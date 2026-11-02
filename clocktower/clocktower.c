/*
 * clocktower — a tall stone clocktower alone on a rock in the sea, reached by
 * a narrow stone causeway that the tide covers and uncovers.
 *
 * (The full description is kept up to date at the end of development; see
 * usage() for the options.)
 */
#define _DEFAULT_SOURCE
#define GL_GLEXT_PROTOTYPES
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <fontconfig/fontconfig.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

#define PI 3.14159265f
#define DEG (PI / 180.0f)
#define D2R (M_PI / 180.0)

/* ---- options ---- */
static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 0;              /* the moment this many seconds after the epoch */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the camera's tour starts at */
static double opt_lat = 51.48, opt_lon = 0.0;
static int loc_given;
static const char *opt_weather = "random";
static const char *opt_font = "C059";
static double time_offset;             /* --time: added to the wall clock */
static float opt_tide = -99;           /* --tide: pin the water level (m), for testing */
static float opt_scale;                /* fixed render scale (0: adapt to the budget) */
static float opt_fov;                  /* spanning: the horizontal angle over the layout */
static float opt_budget = 4.0f;        /* ms per frame the dynamic resolution aims at */
static int opt_shotcam = -1;           /* hold the tour at this shot (testing) */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "clocktower: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9 + time_offset;
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float smooth01(float a, float b, float x) {
    float t = clampf((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

/* the same integer hash as the shaders' (see COMMON) */
static uint32_t hashu(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static float hash1(int32_t a, uint32_t salt) { return (hashu((uint32_t)a * 0x9E3779B9U + salt * 0x85EBCA6BU) >> 8) * (1.0f / 16777216.0f); }
static float hash2(int32_t a, int32_t b, uint32_t salt) {
    return (hashu((uint32_t)a * 0x9E3779B9U ^ hashu((uint32_t)b + salt * 0x85EBCA6BU)) >> 8) * (1.0f / 16777216.0f);
}

static uint32_t rng;

/* value noise and fbm, C side (terrain) */
static float vnoise(float x, float z, uint32_t salt) {
    int32_t ix = (int32_t)floorf(x), iz = (int32_t)floorf(z);
    float fx = x - ix, fz = z - iz;
    fx = fx * fx * (3 - 2 * fx); fz = fz * fz * (3 - 2 * fz);
    float a = hash2(ix, iz, salt), b = hash2(ix + 1, iz, salt), c = hash2(ix, iz + 1, salt), d = hash2(ix + 1, iz + 1, salt);
    return mixf(mixf(a, b, fx), mixf(c, d, fx), fz);
}
static float fbm(float x, float z, int oct, uint32_t salt) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) {
        s += a * vnoise(x, z, salt + i * 17);
        n += a; a *= 0.5f;
        float t = x * 1.7f + z * 0.9f; z = z * 1.7f - x * 0.9f + 3.1f; x = t + 1.3f;
    }
    return s / n;
}

/* ---- small vector and matrix kit ---- */
typedef struct { float x, y, z; } v3;
static v3 V3(float x, float y, float z) { return (v3){x, y, z}; }
static v3 vadd(v3 a, v3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
static v3 vsub(v3 a, v3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
static v3 vscale(v3 a, float s) { return V3(a.x * s, a.y * s, a.z * s); }
static float vdot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 vcross(v3 a, v3 b) { return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(v3 a) { return sqrtf(vdot(a, a)); }
static v3 vnorm(v3 a) { float l = vlen(a); return l > 0 ? vscale(a, 1 / l) : a; }
static v3 vlerp(v3 a, v3 b, float t) { return V3(mixf(a.x, b.x, t), mixf(a.y, b.y, t), mixf(a.z, b.z, t)); }

/* 4x4, column-major */
typedef struct { float m[16]; } m4;
static m4 m4id(void) { m4 r = {{0}}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static m4 m4mul(m4 a, m4 b) {
    m4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static m4 m4trans(v3 t) { m4 r = m4id(); r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z; return r; }
/* rotation about a unit axis */
static m4 m4rot(v3 a, float ang) {
    float c = cosf(ang), s = sinf(ang), t = 1 - c;
    m4 r = m4id();
    r.m[0] = t * a.x * a.x + c;       r.m[4] = t * a.x * a.y - s * a.z; r.m[8] = t * a.x * a.z + s * a.y;
    r.m[1] = t * a.x * a.y + s * a.z; r.m[5] = t * a.y * a.y + c;       r.m[9] = t * a.y * a.z - s * a.x;
    r.m[2] = t * a.x * a.z - s * a.y; r.m[6] = t * a.y * a.z + s * a.x; r.m[10] = t * a.z * a.z + c;
    return r;
}
static v3 m4pt(m4 a, v3 p) {
    return V3(a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12],
              a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13],
              a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14]);
}
static v3 m4dir(m4 a, v3 p) {
    return V3(a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z,
              a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z,
              a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z);
}
/* a view matrix from a camera basis (right, up, back) and position */
static m4 m4view(v3 r, v3 u, v3 b, v3 p) {
    m4 v = m4id();
    v.m[0] = r.x; v.m[4] = r.y; v.m[8] = r.z;
    v.m[1] = u.x; v.m[5] = u.y; v.m[9] = u.z;
    v.m[2] = b.x; v.m[6] = b.y; v.m[10] = b.z;
    v.m[12] = -vdot(r, p); v.m[13] = -vdot(u, p); v.m[14] = -vdot(b, p);
    return v;
}
static m4 m4frustum(float l, float r, float b, float t, float n, float f) {
    m4 p = {{0}};
    p.m[0] = 2 * n / (r - l); p.m[8] = (r + l) / (r - l);
    p.m[5] = 2 * n / (t - b); p.m[9] = (t + b) / (t - b);
    p.m[10] = -(f + n) / (f - n); p.m[14] = -2 * f * n / (f - n); p.m[11] = -1;
    return p;
}
static m4 m4ortho(float l, float r, float b, float t, float n, float f) {
    m4 p = m4id();
    p.m[0] = 2 / (r - l); p.m[5] = 2 / (t - b); p.m[10] = -2 / (f - n);
    p.m[12] = -(r + l) / (r - l); p.m[13] = -(t + b) / (t - b); p.m[14] = -(f + n) / (f - n);
    return p;
}

/* ---- the lock: when the screen was locked (swaylock's start time), as gauges ---- */
static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

static time_t locked_at;
/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else 0 */
static time_t lock_start(void) {
    time_t best = 0, btime = 0;
    char line[256];
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "btime %ld", &btime) == 1) break;
        fclose(f);
    }
    long hz = sysconf(_SC_CLK_TCK);
    DIR *d = btime && hz > 0 ? opendir("/proc") : NULL;
    for (struct dirent *e; d && (e = readdir(d));) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        char path[300], comm[64];
        snprintf(path, sizeof path, "/proc/%s/comm", e->d_name);
        slurp_line(path, comm, sizeof comm);
        if (strncmp(comm, "swaylock", 8)) continue;
        snprintf(path, sizeof path, "/proc/%s/stat", e->d_name);
        char st[1024];
        slurp_line(path, st, sizeof st);
        char *p = strrchr(st, ')');
        unsigned long long start = 0;
        /* fields after the command: state is #3, starttime #22 */
        if (p && sscanf(p + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &start) == 1) {
            time_t t = btime + (time_t)(start / (unsigned long long)hz);
            if (!best || t < best) best = t;
        }
    }
    if (d) closedir(d);
    return best;
}

/* where: --lat/--lon, else ~/.config/wallpapers/location (as nightsky), else Greenwich */
static void load_location(void) {
    if (loc_given) return;
    char path[1024];
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xdg && *xdg) snprintf(path, sizeof path, "%s/wallpapers/location", xdg);
    else if (home) snprintf(path, sizeof path, "%s/.config/wallpapers/location", home);
    else return;
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        for (char *c = line; *c; c++) if (*c == ',') *c = ' ';
        double la, lo;
        if (line[0] != '#' && sscanf(line, "%lf %lf", &la, &lo) == 2 && fabs(la) <= 90 && fabs(lo) <= 360) {
            opt_lat = la; opt_lon = lo;
            break;
        }
    }
    fclose(f);
}

/* ---- the sky: low-precision Sun and Moon, and the tide they raise ---- */
static struct {
    v3 sun, moon;              /* world directions: x east, y up, z south */
    double sun_alt, moon_alt;  /* degrees */
    double ha_sun, ha_moon;    /* local hour angles, radians */
    float illum;               /* the Moon's lit fraction */
} sky;

/* equatorial (ra, dec) and local sidereal time to a world direction */
static v3 eq2world(double ra, double dec, double lst) {
    double lat = opt_lat * D2R, h = lst - ra;
    double e = -cos(dec) * sin(h);
    double n = cos(lat) * sin(dec) - sin(lat) * cos(dec) * cos(h);
    double u = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(h);
    return V3((float)e, (float)u, (float)-n);
}

static void sky_compute(double unix_t) {
    double jd = unix_t / 86400.0 + 2440587.5, d = jd - 2451545.0, T = d / 36525;
    double gmst = fmod(280.46061837 + 360.98564736629 * d, 360) * D2R;
    double lst = gmst + opt_lon * D2R;
    double eps = (23.439291 - 0.0130042 * T) * D2R;
    /* the Sun */
    double L = (280.460 + 0.9856474 * d) * D2R, g = (357.528 + 0.9856003 * d) * D2R;
    double lam = L + (1.915 * sin(g) + 0.020 * sin(2 * g)) * D2R;
    double sra = atan2(cos(eps) * sin(lam), cos(lam)), sdec = asin(sin(eps) * sin(lam));
    sky.sun = eq2world(sra, sdec, lst);
    sky.sun_alt = asin(sky.sun.y) / D2R;
    sky.ha_sun = lst - sra;
    /* the Moon: the larger terms of Meeus ch. 47 */
    double Lp = (218.3164477 + 481267.88123421 * T) * D2R;
    double D = (297.8501921 + 445267.1114034 * T) * D2R;
    double M = (357.5291092 + 35999.0502909 * T) * D2R;
    double Mp = (134.9633964 + 477198.8675055 * T) * D2R;
    double F = (93.2720950 + 483202.0175233 * T) * D2R;
    double ml = Lp + (6.288774 * sin(Mp) + 1.274027 * sin(2 * D - Mp) + 0.658314 * sin(2 * D) + 0.213618 * sin(2 * Mp) -
                      0.185116 * sin(M) - 0.114332 * sin(2 * F) + 0.058793 * sin(2 * D - 2 * Mp) +
                      0.057066 * sin(2 * D - M - Mp) + 0.053322 * sin(2 * D + Mp) + 0.045758 * sin(2 * D - M)) * D2R;
    double mb = (5.128122 * sin(F) + 0.280602 * sin(Mp + F) + 0.277693 * sin(Mp - F) + 0.173237 * sin(2 * D - F)) * D2R;
    double mra = atan2(sin(ml) * cos(eps) - tan(mb) * sin(eps), cos(ml));
    double mdec = asin(sin(mb) * cos(eps) + cos(mb) * sin(eps) * sin(ml));
    /* topocentric parallax, roughly: the Moon sits ~0.95 deg lower near the horizon */
    sky.moon = eq2world(mra, mdec, lst);
    double alt = asin(sky.moon.y) - 0.0166 * cos(asin(sky.moon.y));
    float hz = sqrtf(sky.moon.x * sky.moon.x + sky.moon.z * sky.moon.z);
    if (hz > 1e-6f) {
        sky.moon.x *= (float)cos(alt) / hz; sky.moon.z *= (float)cos(alt) / hz;
        sky.moon.y = (float)sin(alt);
    }
    sky.moon_alt = alt / D2R;
    sky.ha_moon = lst - mra;
    /* the lit fraction from the elongation (geocentric) */
    double ce = cos(sdec) * cos(mdec) * cos(sra - mra) + sin(sdec) * sin(mdec);
    sky.illum = (float)((1 - ce) / 2);
}

/* The tide: the lunar (M2, 12 h 25 m) and solar (S2, 12 h) semi-diurnal
 * constituents, each high a fixed lag after its body crosses the meridian,
 * so springs follow the new and full Moon and neaps the quarters. */
static float tide_at(void) {
    if (opt_tide > -50) return opt_tide;
    double lag = 2.1;                  /* radians of hour angle: ~4 h after the transit */
    return (float)(1.65 * cos(2 * sky.ha_moon - lag) + 0.55 * cos(2 * sky.ha_sun - lag) + 0.15);
}

/* ======================================================================
 * Geometry: everything solid is built here once, as triangles with a
 * position, normal, a material uv (metres for stone, so the coursing is the
 * same everywhere), a material id and a per-object variation.
 * ====================================================================== */
enum { M_STONE, M_RUBBLE, M_IRON, M_BRASS, M_BRONZE, M_WOOD, M_COPPER, M_LANTERN, M_DIAL, M_DRUM,
       M_PAVE, M_ROPE, M_WINDOW, M_LAMP, M_DOOR, M_STEEL };
#define VSZ 10
typedef struct { float *v; int n, cap; } Mesh;

static void mv(Mesh *m, v3 p, v3 n, float u, float v, int mat, float aux) {
    if (m->n + 1 > m->cap) m->v = realloc(m->v, (size_t)(m->cap = m->cap ? m->cap * 2 : 65536) * VSZ * sizeof(float));
    float *o = m->v + (size_t)m->n++ * VSZ;
    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z;
    o[6] = u; o[7] = v; o[8] = (float)mat; o[9] = aux;
}
/* a quad a b c d (counter-clockwise seen from outside), uv per corner */
static void mquad(Mesh *m, v3 a, v3 b, v3 c, v3 d, v3 n, const float *uv, int mat, float aux) {
    mv(m, a, n, uv[0], uv[1], mat, aux); mv(m, b, n, uv[2], uv[3], mat, aux); mv(m, c, n, uv[4], uv[5], mat, aux);
    mv(m, a, n, uv[0], uv[1], mat, aux); mv(m, c, n, uv[4], uv[5], mat, aux); mv(m, d, n, uv[6], uv[7], mat, aux);
}
static void mtri(Mesh *m, v3 a, v3 b, v3 c, int mat, float aux) {
    v3 n = vnorm(vcross(vsub(b, a), vsub(c, a)));
    mv(m, a, n, a.x + a.z, a.y, mat, aux); mv(m, b, n, b.x + b.z, b.y, mat, aux); mv(m, c, n, c.x + c.z, c.y, mat, aux);
}
/* planar uv for a face: metres along two world axes chosen by its normal */
static void face_uv(v3 n, v3 p, float *u, float *v) {
    float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    if (ay >= ax && ay >= az) { *u = p.x; *v = p.z; }
    else if (ax >= az) { *u = n.x > 0 ? -p.z : p.z; *v = p.y; }
    else { *u = n.z > 0 ? p.x : -p.x; *v = p.y; }
}
/* a box: centre, three half-axes (any orientation); uv in metres as if it
 * stood at uvc (so a displaced block keeps its coursing); faces: a bitmask
 * of +x -x +y -y +z -z in its own frame */
static void mbox_uv(Mesh *m, v3 c, v3 ax, v3 ay, v3 az, int mat, float aux, v3 uvc, int faces) {
    v3 p[8];
    for (int i = 0; i < 8; i++)
        p[i] = vadd(c, vadd(vscale(ax, i & 1 ? 1.f : -1.f), vadd(vscale(ay, i & 2 ? 1.f : -1.f), vscale(az, i & 4 ? 1.f : -1.f))));
    static const int F[6][4] = {{1, 3, 7, 5}, {0, 4, 6, 2}, {2, 6, 7, 3}, {0, 1, 5, 4}, {4, 5, 7, 6}, {0, 2, 3, 1}};
    v3 ex = vnorm(ax), ey = vnorm(ay), ez = vnorm(az);
    for (int f = 0; f < 6; f++) {
        if (!(faces & (1 << f))) continue;
        v3 q[4];
        float uv[8];
        v3 fn = f < 2 ? V3(f ? -1.f : 1.f, 0, 0) : f < 4 ? V3(0, f == 2 ? 1.f : -1.f, 0) : V3(0, 0, f == 4 ? 1.f : -1.f);
        for (int k = 0; k < 4; k++) {
            q[k] = p[F[f][k]];
            v3 l = vsub(q[k], c);
            v3 loc = V3(vdot(l, ex), vdot(l, ey), vdot(l, ez));
            face_uv(fn, vadd(loc, uvc), &uv[2 * k], &uv[2 * k + 1]);
        }
        v3 n = vnorm(vcross(vsub(q[1], q[0]), vsub(q[3], q[0])));
        mquad(m, q[0], q[1], q[2], q[3], n, uv, mat, aux);
    }
}
static void mbox(Mesh *m, v3 c, v3 h, int mat, float aux) {
    mbox_uv(m, c, V3(h.x, 0, 0), V3(0, h.y, 0), V3(0, 0, h.z), mat, aux, c, 63);
}
/* a cylinder (or cone) between two points, ends optional */
static void mcyl(Mesh *m, v3 a, v3 b, float r0, float r1, int seg, int caps, int mat, float aux) {
    v3 ax = vnorm(vsub(b, a));
    v3 t = fabsf(ax.y) < 0.9f ? vnorm(vcross(ax, V3(0, 1, 0))) : vnorm(vcross(ax, V3(1, 0, 0)));
    v3 s = vcross(ax, t);
    float len = vlen(vsub(b, a));
    for (int i = 0; i < seg; i++) {
        float a0 = 2 * PI * i / seg, a1 = 2 * PI * (i + 1) / seg;
        v3 d0 = vadd(vscale(t, cosf(a0)), vscale(s, sinf(a0))), d1 = vadd(vscale(t, cosf(a1)), vscale(s, sinf(a1)));
        v3 p0 = vadd(a, vscale(d0, r0)), p1 = vadd(a, vscale(d1, r0));
        v3 q0 = vadd(b, vscale(d0, r1)), q1 = vadd(b, vscale(d1, r1));
        float slope = (r0 - r1) / len;
        v3 n0 = vnorm(vadd(d0, vscale(ax, slope))), n1 = vnorm(vadd(d1, vscale(ax, slope)));
        float u0 = (float)i / seg * 2 * PI * r0, u1 = (float)(i + 1) / seg * 2 * PI * r0;
        mv(m, p0, n0, u0, 0, mat, aux); mv(m, p1, n1, u1, 0, mat, aux); mv(m, q1, n1, u1, len, mat, aux);
        mv(m, p0, n0, u0, 0, mat, aux); mv(m, q1, n1, u1, len, mat, aux); mv(m, q0, n0, u0, len, mat, aux);
        if (caps & 1) { v3 n = vscale(ax, -1); mv(m, a, n, 0, 0, mat, aux); mv(m, p1, n, 0, 0, mat, aux); mv(m, p0, n, 0, 0, mat, aux); }
        if (caps & 2) { mv(m, b, ax, 0, 0, mat, aux); mv(m, q0, ax, 0, 0, mat, aux); mv(m, q1, ax, 0, 0, mat, aux); }
    }
}
/* a surface of revolution about +y through c: profile (r, y) pairs, top down
 * or bottom up as long as the outside is on the right of the walk */
static void mlathe(Mesh *m, v3 c, const float *prof, int np, int seg, int mat, float aux) {
    for (int j = 0; j + 1 < np; j++) {
        float r0 = prof[2 * j], y0 = prof[2 * j + 1], r1 = prof[2 * j + 2], y1 = prof[2 * j + 3];
        float dr = r1 - r0, dy = y1 - y0, l = sqrtf(dr * dr + dy * dy);
        if (l < 1e-6f) continue;
        float nr = dy / l, ny = -dr / l;
        for (int i = 0; i < seg; i++) {
            float a0 = 2 * PI * i / seg, a1 = 2 * PI * (i + 1) / seg;
            float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
            v3 p00 = V3(c.x + r0 * c0, c.y + y0, c.z + r0 * s0), p01 = V3(c.x + r0 * c1, c.y + y0, c.z + r0 * s1);
            v3 p10 = V3(c.x + r1 * c0, c.y + y1, c.z + r1 * s0), p11 = V3(c.x + r1 * c1, c.y + y1, c.z + r1 * s1);
            v3 n0 = V3(nr * c0, ny, nr * s0), n1 = V3(nr * c1, ny, nr * s1);
            float u0 = (float)i / seg * 6.0f, u1 = (float)(i + 1) / seg * 6.0f;
            mv(m, p00, n0, u0, y0, mat, aux); mv(m, p11, n1, u1, y1, mat, aux); mv(m, p01, n1, u1, y0, mat, aux);
            mv(m, p00, n0, u0, y0, mat, aux); mv(m, p10, n0, u0, y1, mat, aux); mv(m, p11, n1, u1, y1, mat, aux);
        }
    }
}

/* A wall panel: a slab on a face (origin o, along t, outward n, up +y)
 * covering s0..s1, y0..y1, th deep, with openings: centre s, half-width,
 * sill, springing (the top, for a square one) and whether it's arched. */
typedef struct { float sc, hw, yb, ys; int arch; } Hole;
static v3 pan_o, pan_t, pan_n;
static v3 PP(float s, float y, float d) { return vadd(pan_o, vadd(vscale(pan_t, s), vadd(V3(0, y, 0), vscale(pan_n, d)))); }
static void pan_quad(Mesh *m, v3 a, v3 b, v3 c, v3 d, int mat, float aux) {
    /* stone coursing runs in panel coordinates: s along, y up, depth */
    v3 n = vnorm(vcross(vsub(b, a), vsub(d, a)));
    v3 q[4] = {a, b, c, d};
    float uv[8];
    for (int i = 0; i < 4; i++) {
        v3 l = vsub(q[i], pan_o);
        float s = vdot(l, pan_t), y = l.y, dd = vdot(l, pan_n);
        if (fabsf(n.y) > 0.7f) { uv[2 * i] = s; uv[2 * i + 1] = dd; }
        else if (fabsf(vdot(n, pan_t)) > 0.7f) { uv[2 * i] = dd; uv[2 * i + 1] = y; }
        else { uv[2 * i] = s; uv[2 * i + 1] = y; }
    }
    mquad(m, a, b, c, d, n, uv, mat, aux);
}
static float hole_top(const Hole *h, float s) {
    if (!h->arch) return h->ys;
    float d = h->hw * h->hw - (s - h->sc) * (s - h->sc);
    return h->ys + sqrtf(d > 0 ? d : 0);
}
static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void panel(Mesh *m, v3 o, v3 t, v3 n, float s0, float s1, float y0, float y1, float th,
                  const Hole *h, int nh, int mat, float aux, int caps) {
    pan_o = o; pan_t = t; pan_n = n;
    float br[256];
    int nb = 0;
    br[nb++] = s0; br[nb++] = s1;
    for (int i = 0; i < nh; i++) {
        int k = h[i].arch ? 16 : 1;
        for (int j = 0; j <= k; j++) br[nb++] = h[i].sc - h[i].hw + 2 * h[i].hw * j / k;
    }
    qsort(br, (size_t)nb, sizeof *br, cmpf);
    for (int i = 0; i + 1 < nb; i++) {
        float a = br[i], b = br[i + 1];
        if (b - a < 1e-4f || a < s0 - 1e-4f || b > s1 + 1e-4f) continue;
        float mid = (a + b) / 2;
        const Hole *hh = NULL;
        for (int j = 0; j < nh; j++)
            if (fabsf(mid - h[j].sc) < h[j].hw) hh = &h[j];
        if (!hh) {
            pan_quad(m, PP(a, y0, 0), PP(b, y0, 0), PP(b, y1, 0), PP(a, y1, 0), mat, aux);
            pan_quad(m, PP(b, y0, -th), PP(a, y0, -th), PP(a, y1, -th), PP(b, y1, -th), mat, aux);
            continue;
        }
        if (hh->yb > y0 + 1e-4f) {     /* below the opening, with the sill on top */
            pan_quad(m, PP(a, y0, 0), PP(b, y0, 0), PP(b, hh->yb, 0), PP(a, hh->yb, 0), mat, aux);
            pan_quad(m, PP(b, y0, -th), PP(a, y0, -th), PP(a, hh->yb, -th), PP(b, hh->yb, -th), mat, aux);
            pan_quad(m, PP(a, hh->yb, 0), PP(b, hh->yb, 0), PP(b, hh->yb, -th), PP(a, hh->yb, -th), mat, aux);
        }
        float ta = hole_top(hh, a), tb = hole_top(hh, b);
        if (ta < y1 - 1e-4f || tb < y1 - 1e-4f) {   /* above it, with the soffit underneath */
            pan_quad(m, PP(a, ta, 0), PP(b, tb, 0), PP(b, y1, 0), PP(a, y1, 0), mat, aux);
            pan_quad(m, PP(b, tb, -th), PP(a, ta, -th), PP(a, y1, -th), PP(b, y1, -th), mat, aux);
            pan_quad(m, PP(b, tb, 0), PP(a, ta, 0), PP(a, ta, -th), PP(b, tb, -th), mat, aux);
        }
    }
    for (int j = 0; j < nh; j++) {     /* the jambs */
        float l = h[j].sc - h[j].hw, r = h[j].sc + h[j].hw, top = h[j].ys;
        pan_quad(m, PP(l, h[j].yb, 0), PP(l, h[j].yb, -th), PP(l, top, -th), PP(l, top, 0), mat, aux);
        pan_quad(m, PP(r, h[j].yb, -th), PP(r, h[j].yb, 0), PP(r, top, 0), PP(r, top, -th), mat, aux);
    }
    if (caps & 1) pan_quad(m, PP(s0, y0, -th), PP(s0, y0, 0), PP(s0, y1, 0), PP(s0, y1, -th), mat, aux);
    if (caps & 2) pan_quad(m, PP(s1, y0, 0), PP(s1, y0, -th), PP(s1, y1, -th), PP(s1, y1, 0), mat, aux);
    if (caps & 4) pan_quad(m, PP(s0, y1, 0), PP(s1, y1, 0), PP(s1, y1, -th), PP(s0, y1, -th), mat, aux);
    if (caps & 8) pan_quad(m, PP(s1, y0, 0), PP(s0, y0, 0), PP(s0, y0, -th), PP(s1, y0, -th), mat, aux);
}

/* A gear about +x through the origin: n teeth on pitch radius r, w thick.
 * Wheels get a rim, spokes and a hub; pinions (no spokes) are solid.
 * kind 1: an escape wheel (long raked teeth). */
static void mgear(Mesh *m, int n, float r, float w, int spokes, int kind, int mat, float aux) {
    float mod = 2 * r / n, ra = r + mod * (kind == 1 ? 2.4f : 1.0f), rr = r - mod * 1.25f;
    int pinion = spokes == 0;
    float rin = pinion ? 0 : rr - fmaxf(mod * 2.2f, r * 0.08f);
    static float px[4096], py[4096];
    int np = 0;
    float pa = 2 * PI / n;
    for (int i = 0; i < n; i++) {
        float a = i * pa;
        if (kind == 1) {
            static const float E[4][2] = {{-0.5f, 0}, {0.16f, 2}, {0.26f, 1}, {0.30f, 0}};
            for (int k = 0; k < 4; k++) {
                float rad = E[k][1] == 2 ? ra : E[k][1] == 1 ? ra - mod * 0.5f : rr;
                float an = a + E[k][0] * pa;
                px[np] = rad * cosf(an); py[np] = rad * sinf(an); np++;
            }
        } else {
            static const float T[8][2] = {{-0.5f, 0}, {-0.30f, 0}, {-0.24f, 1}, {-0.15f, 2}, {0.15f, 2}, {0.24f, 1}, {0.30f, 0}, {0.42f, 0}};
            for (int k = 0; k < 8; k++) {
                float rad = T[k][1] == 0 ? rr : T[k][1] == 1 ? r : ra;
                float an = a + T[k][0] * pa;
                px[np] = rad * cosf(an); py[np] = rad * sinf(an); np++;
            }
        }
    }
    float hx = w / 2;
    /* faces at x = +hx (front) and -hx (back); the outline maps to (y, z) */
    for (int i = 0; i < np; i++) {
        int j = (i + 1) % np;
        v3 a0 = V3(hx, px[i], py[i]), a1 = V3(hx, px[j], py[j]);
        v3 b0 = V3(-hx, px[i], py[i]), b1 = V3(-hx, px[j], py[j]);
        v3 en = vnorm(V3(0, py[j] - py[i], -(px[j] - px[i])));
        float uv[8] = {0};
        mquad(m, a0, b0, b1, a1, en, uv, mat, aux);
        float ai = atan2f(py[i], px[i]), aj = atan2f(py[j], px[j]);
        v3 i0 = V3(hx, rin * cosf(ai), rin * sinf(ai)), i1 = V3(hx, rin * cosf(aj), rin * sinf(aj));
        float fu[8] = {px[i], py[i], px[j], py[j], i1.y, i1.z, i0.y, i0.z};
        mquad(m, a0, a1, i1, i0, V3(1, 0, 0), fu, mat, aux);
        v3 k0 = V3(-hx, i0.y, i0.z), k1 = V3(-hx, i1.y, i1.z);
        mquad(m, b1, b0, k0, k1, V3(-1, 0, 0), fu, mat, aux);
        if (!pinion) {
            v3 in = vnorm(V3(0, -(i0.y + i1.y), -(i0.z + i1.z)));
            mquad(m, i1, k1, k0, i0, in, uv, mat, aux);
        }
    }
    if (pinion) return;
    float hr = fmaxf(r * 0.14f, 0.05f), sw = w * 0.7f;
    mcyl(m, V3(-w * 0.65f, 0, 0), V3(w * 0.65f, 0, 0), hr, hr, 20, 3, mat, aux);
    for (int k = 0; k < spokes; k++) {
        float a = (k + 0.5f) * 2 * PI / spokes;
        v3 d = V3(0, cosf(a), sinf(a)), side = V3(0, -sinf(a), cosf(a));
        float rm = (hr + rin) / 2, hl = (rin - hr) / 2 + 0.01f;
        float wd = fmaxf(r * 0.05f, 0.02f);
        mbox_uv(m, vscale(d, rm), vscale(d, hl), vscale(side, wd), V3(sw / 2, 0, 0), mat, aux, V3(0, 0, 0), 63);
    }
}

/* ======================================================================
 * The place. World metres: x east, y up, z south; mean sea level y = 0.
 * The tower stands on a rock at the origin; the causeway runs south from
 * it to the shore, a bay of headlands wrapping round. The broken wall
 * faces east.
 * ====================================================================== */
#define CW_Z0 14.0f                    /* the causeway: from the foot of the stairs... */
#define CW_Z1 156.0f                   /* ...to the beach */
#define PLAT_Y 4.6f                    /* the rock platform round the tower */
#define WALL_Y0 5.0f                   /* the tower walls start here */
#define GFLOOR 8.6f                    /* the gear room floor */
#define BED_Y 12.4f                    /* the top of the timber bed the movement stands on */
#define CEIL_Y 21.0f
#define DIAL_Y 24.0f
#define DIAL_R 1.8f
#define BELF_Y 27.0f
#define TOP_Y 33.0f

static float cw_x(float z) {
    float u = clampf((z - CW_Z0) / (CW_Z1 - CW_Z0), 0, 1);
    return 6.0f * sinf(PI * u) - 2.2f * sinf(2 * PI * u);
}
static float cw_top(float z) {
    return 0.55f + 0.10f * (vnoise(z * 0.07f, 3.3f, 51) - 0.5f) + 0.45f * smooth01(CW_Z1 - 10, CW_Z1 + 4, z);
}
/* the stairs up the rock: from the causeway (z = CW_Z0) to the platform */
#define ST_Z1 6.4f
#define ST_N 14
static float stair_top(int i) { return cw_top(CW_Z0) + (PLAT_Y - cw_top(CW_Z0)) * (i + 1) / ST_N; }

/* where the shore runs: a bay, the headlands closing in on either side */
/* where the shore runs: high cliffs curving out to a headland in the west;
 * to the east low rocks falling back, so the sea opens out that way */
static float shore_z(float x) {
    float w = x < 0 ? 158 - 100 * (1 - expf(-(x / 190) * (x / 190))) : 158 + 0.22f * x + 18 * (1 - expf(-(x / 120) * (x / 120)));
    return w + 14 * (fbm(x * 0.012f, 7.7f, 3, 60) - 0.5f);
}

/* jointed rock: a Voronoi of tilted facets, cracked along the joints */
static float facets(float x, float z, float cell, uint32_t salt) {
    float gx = x / cell, gz = z / cell;
    int ix = (int)floorf(gx), iz = (int)floorf(gz);
    float d1 = 1e9f, d2 = 1e9f, h1 = 0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int cx = ix + i, cz = iz + j;
            float px = cx + 0.15f + 0.7f * hash2(cx, cz, salt), pz = cz + 0.15f + 0.7f * hash2(cx, cz, salt + 1);
            float dx = gx - px, dz = gz - pz, d = dx * dx + dz * dz;
            if (d < d1) {
                d2 = d1; d1 = d;
                float sx = (hash2(cx, cz, salt + 2) - 0.5f) * 0.55f, sz = (hash2(cx, cz, salt + 3) - 0.5f) * 0.55f;
                h1 = (hash2(cx, cz, salt + 4) - 0.5f) + sx * dx + sz * dz;
            } else if (d < d2) d2 = d;
        }
    float edge = smooth01(0.0f, 0.5f, sqrtf(d2) - sqrtf(d1));
    return (h1 - (1 - edge) * 0.3f) * cell;
}

/* the solid ground (rock, sand, the sea bed), not counting the causeway */
static float terrain(float x, float z) {
    float h = -8.0f + 3.0f * fbm(x * 0.006f, z * 0.006f, 4, 1);
    /* the islet: an irregular crag of rock */
    float ang = atan2f(z, x), r = sqrtf(x * x + z * z);
    float rn = r * (1 + 0.32f * (fbm(cosf(ang) * 1.4f + 5, sinf(ang) * 1.4f + 5, 3, 2) - 0.5f));
    float crag = fbm(x * 0.18f, z * 0.18f, 4, 3), ridge = 1 - fabsf(2 * fbm(x * 0.09f + 3, z * 0.09f, 3, 4) - 1);
    float isl = 5.0f - 7.6f * smooth01(7.5f, 21, rn) - 4.5f * smooth01(17, 42, rn);
    /* crags: higher to the north and west, behind the tower from the shore */
    float nw = clampf((-x * 0.55f - z * 0.85f) / 14 + 0.35f, 0, 1);
    float fa = r < 60 ? facets(x, z, 3.8f, 300) + 0.3f * facets(x + 7, z - 3, 1.5f, 310) : 0;
    isl += fa * 0.6f * smooth01(6.5f, 9.5f, rn);
    isl += smooth01(0.2f, 1.0f, ridge) * (0.8f + 2.8f * nw) * smooth01(7, 10.5f, rn) * (1 - smooth01(13, 24, rn));
    isl += (crag - 0.5f) * 1.2f * smooth01(6.0f, 10, rn);
    h = fmaxf(h, isl);
    /* skerries: rocks out in the bay, some only showing at low water */
    static const float SK[][4] = {{-31, -25, 8, 2.4f}, {26, -31, 6, 0.9f}, {37, 6, 5, 0.2f}, {-40, 8, 7, 1.4f},
                                  {11, -43, 5, 0.5f}, {-17, 33, 4, -0.3f}, {21, 52, 5, 0.4f}, {-23, 88, 5, -0.6f},
                                  {28, 112, 7, 0.8f}, {-44, -41, 6, 3.0f}};
    for (int i = 0; i < 10; i++) {
        float dx = x - SK[i][0], dz = z - SK[i][1];
        /* stretched along the strike of the rock */
        float u = dx * 0.8f + dz * 0.6f, v = (-dx * 0.6f + dz * 0.8f) * 1.7f, d = sqrtf(u * u + v * v);
        if (d > SK[i][2] * 2.0f) continue;
        float sk = SK[i][3] - 3.5f * powf(d / SK[i][2], 1.6f) + facets(x, z, 2.6f, 320 + i) * 0.9f;
        h = fmaxf(h, sk);
    }
    /* the reef under the causeway, with rocks breaking the surface at low water */
    if (z > 6 && z < CW_Z1 + 20) {
        float d = fabsf(x - cw_x(z));
        float reef = -1.3f - 0.05f * d * d + 1.8f * (fbm(x * 0.11f, z * 0.11f, 4, 5) - 0.45f);
        reef += 1.2f * powf(fbm(x * 0.21f + 9, z * 0.21f, 3, 6), 3.0f) * 3 * expf(-d * 0.08f);
        h = fmaxf(h, reef * smooth01(6, 16, z) + h * (1 - smooth01(6, 16, z)));
    }
    /* the shore: shingle, then rock and turf rising into the headlands */
    float zs = shore_z(x), dz = z - zs;
    float cliffs = smooth01(-25, -80, x);
    float hills = 0.35f + 0.65f * smooth01(10, -60, x);
    float land = 0.3f + dz * (0.075f - 0.03f * smooth01(0, 80, x));
    if (dz > 0) {
        land += cliffs * 13 * smooth01(4, 16, dz) + 16 * hills * fbm(x * 0.008f, z * 0.008f, 4, 7) * smooth01(0, 90, dz);
        land += (fbm(x * 0.05f, z * 0.05f, 4, 8) - 0.5f) * 3 * smooth01(2, 12, dz);
    }
    h = fmaxf(h, land);
    /* the platform round the tower and the side tower */
    float px = fmaxf(fabsf(x + 0.6f) - 0.6f, 0), pd = fmaxf(px, fabsf(z));
    float plat = smooth01(7.6f, 6.0f, pd);
    h = mixf(h, PLAT_Y - 0.03f, plat);
    /* the stairs are cut into the rock */
    float sx = fabsf(x - cw_x(z));
    if (z > ST_Z1 - 0.5f && z < CW_Z0 + 1 && sx < 2.2f) {
        float u = clampf((z - ST_Z1) / (CW_Z0 - ST_Z1), 0, 1);
        float sy = PLAT_Y + (cw_top(CW_Z0) - PLAT_Y) * u - 0.25f;
        float k = smooth01(2.2f, 1.4f, sx);
        h = mixf(h, fminf(h, sy), k);
    }
    return h;
}

/* the top of whatever is solid here, the causeway and stairs included (for the water) */
static float solid_top(float x, float z) {
    float h = terrain(x, z);
    if (z >= CW_Z0 - 0.3f && z <= CW_Z1 + 2 && fabsf(x - cw_x(z)) < 1.35f) h = fmaxf(h, cw_top(z));
    if (z > ST_Z1 && z < CW_Z0 && fabsf(x - cw_x(z)) < 1.15f) {
        int i = (int)((CW_Z0 - z) / ((CW_Z0 - ST_Z1) / ST_N));
        if (i > ST_N - 1) i = ST_N - 1;
        h = fmaxf(h, stair_top(i));
    }
    if (fabsf(x) < 4.4f && fabsf(z) < 4.4f) h = fmaxf(h, 30);
    return h;
}

/* ---- the movement: where every arbor is (y, z in the wall's plane) ---- */
typedef struct { float y, z; } P2;
static P2 C_G1 = {13.75f, -1.25f}, C_G2, C_E, C_S1 = {13.75f, 1.55f}, C_S2, C_F;
#define X_WHEEL 0.55f                  /* the plane of the great wheels */
#define X_FRONT 1.08f                  /* escape wheel, count wheel, fly */
#define X_PEND (-0.38f)
#define PEND_TOP 16.55f
#define PEND_L 6.95f
static float pend_T;                   /* its period, s */
static P2 off2(P2 c, float d, float a) { return (P2){c.y + d * cosf(a), c.z + d * sinf(a)}; }
static void movement_layout(void) {
    C_G2 = off2(C_G1, 1.2375f, 0.62f);
    C_E = off2(C_G2, 0.8333f, -0.55f);
    C_S2 = off2(C_S1, 1.2375f, 0.33f);
    C_F = off2(C_S2, 0.7027f, -0.42f);
    pend_T = 2 * PI * sqrtf(PEND_L / 9.81f);
}
/* where the hammer strikes and the bell hangs */
#define BELL_PIV_Y 32.25f
#define HAM_PIV (V3(1.55f, 31.15f, 0))

/* ---- parts that move: each is a range of the dynamic mesh with a pivot ---- */
enum { D_G1, D_G2, D_E, D_ANCHOR, D_PEND, D_S1, D_S2, D_FLY, D_TAIL, D_ROD, D_HAMMER, D_BELL, D_LEAD,
       D_WEIGHT_G, D_WEIGHT_S, D_CABLE_G, D_CABLE_S, D_DRUM0, D_DRUM1, D_DRUM2, D_DRUM3, D_LEVER, D_ARMIL,
       D_ARMIL2, D_N };
static struct { int first, count; v3 pivot; m4 model; float rad; } part[D_N];
static Mesh mstat, mdyn;
static int part_begin(int id, v3 pivot) { part[id].first = mdyn.n; part[id].pivot = pivot; return id; }
static void part_end(int id) {
    part[id].count = mdyn.n - part[id].first;
    float r = 0;
    for (int i = part[id].first; i < mdyn.n; i++) {
        const float *v = mdyn.v + (size_t)i * VSZ;
        r = fmaxf(r, sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    }
    part[id].rad = r;
}

/* lights: lanterns on the causeway, the lamp in the gear room, the window */
#define MAXL 16
static v3 lamp_pos[MAXL];
static int nlamp;

/* E face block coursing: the shaders' pattern (stone()) uses the same rows */
static float row_len(int r) { return 0.75f + 0.55f * hash1(r, 11); }
static float row_off(int r) { return hash1(r, 12) * 3.0f; }
static float breach(float s, float y, float grow) {
    float e = ((s - 0.1f) / (2.7f + grow)) * ((s - 0.1f) / (2.7f + grow)) + ((y - 14.7f) / (5.7f + grow)) * ((y - 14.7f) / (5.7f + grow));
    return e + 0.55f * (fbm(s * 0.7f + 3, y * 0.45f, 3, 7) - 0.5f);
}

static void build_tower(Mesh *m) {
    const v3 up = V3(0, 1, 0);
    (void)up;
    /* faces: S (+z), N (-z) span the corners; E, W fit between */
    v3 So = V3(0, 0, 4), St = V3(1, 0, 0), Sn = V3(0, 0, 1);
    v3 No = V3(0, 0, -4), Nt = V3(-1, 0, 0), Nn = V3(0, 0, -1);
    v3 Eo = V3(4, 0, 0), Et = V3(0, 0, -1), En = V3(1, 0, 0);
    v3 Wo = V3(-4, 0, 0), Wt = V3(0, 0, 1), Wn = V3(-1, 0, 0);
    /* the lower stage: the door, a window, slits for the gear room */
    Hole hs[2] = {{0, 0.85f, WALL_Y0, 7.0f, 1}, {0.0f, 0.36f, 17.4f, 19.0f, 1}};
    panel(m, So, St, Sn, -4, 4, WALL_Y0, CEIL_Y, 1, hs, 2, M_STONE, 0, 3);
    Hole hn[1] = {{1.0f, 0.2f, 15.0f, 16.9f, 1}};
    panel(m, No, Nt, Nn, -4, 4, WALL_Y0, CEIL_Y, 1, hn, 1, M_STONE, 0, 3);
    Hole hw[1] = {{-0.6f, 0.2f, 12.6f, 14.6f, 1}};
    panel(m, Wo, Wt, Wn, -3, 3, WALL_Y0, CEIL_Y, 1, hw, 1, M_STONE, 0, 0);
    /* the east wall, block by block, two skins deep, broken open */
    pan_o = Eo; pan_t = Et; pan_n = En;
    for (int r = (int)(WALL_Y0 / 0.5f); r < (int)(CEIL_Y / 0.5f); r++) {
        float y0 = r * 0.5f, y1 = y0 + 0.5f, L = row_len(r), off = row_off(r);
        int k0 = (int)floorf((-3 + off) / L);
        for (int k = k0; k * L - off < 3; k++) {
            float a = fmaxf(k * L - off, -3), b = fminf((k + 1) * L - off, 3);
            if (b - a < 0.05f) continue;
            float sc = (a + b) / 2, yc = (y0 + y1) / 2;
            for (int layer = 0; layer < 2; layer++) {
                float d0 = layer ? -1.0f : -0.45f, d1 = layer ? -0.45f : 0.0f;
                float br = breach(sc, yc, layer ? -0.45f : 0.0f);
                if (br < 1) continue;
                float loose = br < 1.25f ? hash2(r * 7 + layer, k, 20) : 1;
                float dx = 0, dy = 0, ds = 0, tilt = 0;
                if (loose < 0.45f) {
                    dx = 0.05f * hash2(r, k, 21) * (layer ? 0.3f : 1);
                    dy = -0.03f * hash2(r, k, 22);
                    ds = 0.03f * (hash2(r, k, 23) - 0.5f);
                    tilt = 0.08f * (hash2(r, k, 24) - 0.5f);
                }
                float g = 0.004f;      /* the joints */
                v3 c = PP(sc + ds, yc + dy, (d0 + d1) / 2 + dx);
                v3 ax = vscale(Et, (b - a) / 2 - g), ay = V3(0, 0.25f - g, 0), az = vscale(En, (d1 - d0) / 2 - g);
                if (tilt != 0) {
                    m4 R = m4rot(En, tilt);
                    ax = m4dir(R, ax); ay = m4dir(R, ay);
                }
                /* uv as the panel would give: s along, y up */
                v3 uvc = V3(0, 0, 0);
                (void)uvc;
                v3 p[8];
                for (int i = 0; i < 8; i++)
                    p[i] = vadd(c, vadd(vscale(ax, i & 1 ? 1.f : -1.f), vadd(vscale(ay, i & 2 ? 1.f : -1.f), vscale(az, i & 4 ? 1.f : -1.f))));
                static const int F[6][4] = {{1, 3, 7, 5}, {0, 4, 6, 2}, {2, 6, 7, 3}, {0, 1, 5, 4}, {4, 5, 7, 6}, {0, 2, 3, 1}};
                for (int f = 0; f < 6; f++)
                    pan_quad(m, p[F[f][0]], p[F[f][1]], p[F[f][2]], p[F[f][3]], layer ? M_RUBBLE : M_STONE, hash2(r, k, 25));
            }
        }
    }
    /* fallen blocks: on the rock below the breach and on the gear room floor */
    for (int i = 0; i < 16; i++) {
        float bx, bz, by;
        if (i < 10) { bx = 4.9f + 4.5f * hash1(i, 30); bz = -3.5f + 7 * hash1(i, 31); by = terrain(bx, bz); }
        else { bx = 1.6f + 1.2f * hash1(i, 30); bz = -2.4f + 4.8f * hash1(i, 31); by = GFLOOR; }
        float l = 0.35f + 0.3f * hash1(i, 32), a1 = hash1(i, 33) * 6.28f, a2 = (hash1(i, 34) - 0.5f) * 0.9f;
        m4 R = m4mul(m4rot(V3(0, 1, 0), a1), m4rot(V3(1, 0, 0), a2));
        v3 ax = m4dir(R, V3(l, 0, 0)), ay = m4dir(R, V3(0, 0.24f, 0)), az = m4dir(R, V3(0, 0, 0.3f));
        mbox_uv(m, V3(bx, by + 0.12f, bz), ax, ay, az, M_STONE, hash1(i, 35), V3(hash1(i, 36) * 9, 0.25f, 0), 63);
    }
    /* the dial stage and the belfry */
    panel(m, So, St, Sn, -4, 4, CEIL_Y, BELF_Y, 1, NULL, 0, M_STONE, 0, 3);
    panel(m, No, Nt, Nn, -4, 4, CEIL_Y, BELF_Y, 1, NULL, 0, M_STONE, 0, 3);
    panel(m, Eo, Et, En, -3, 3, CEIL_Y, BELF_Y, 1, NULL, 0, M_STONE, 0, 0);
    panel(m, Wo, Wt, Wn, -3, 3, CEIL_Y, BELF_Y, 1, NULL, 0, M_STONE, 0, 0);
    Hole hb[1] = {{0, 1.45f, BELF_Y + 0.6f, 30.5f, 1}};
    panel(m, So, St, Sn, -4, 4, BELF_Y, TOP_Y, 1, hb, 1, M_STONE, 0, 3);
    panel(m, No, Nt, Nn, -4, 4, BELF_Y, TOP_Y, 1, hb, 1, M_STONE, 0, 3);
    panel(m, Eo, Et, En, -3, 3, BELF_Y, TOP_Y, 1, hb, 1, M_STONE, 0, 0);
    panel(m, Wo, Wt, Wn, -3, 3, BELF_Y, TOP_Y, 1, hb, 1, M_STONE, 0, 0);
    /* plinth, string courses, cornice, parapet */
    mbox(m, V3(0, 3.6f, 0), V3(4.38f, 1.4f, 4.38f), M_STONE, 0.3f);
    mbox_uv(m, V3(0, 5.0f, 0), V3(4.38f, 0, 0), V3(0, 0.08f, 0), V3(0, 0, 4.38f), M_STONE, 0.3f, V3(0, 5, 0), 4);
    static const float courses[] = {CEIL_Y, BELF_Y, TOP_Y - 0.1f};
    for (int i = 0; i < 3; i++) {
        float y = courses[i], o = i == 2 ? 4.42f : 4.2f, hh = i == 2 ? 0.32f : 0.17f;
        mbox(m, V3(0, y, 4.0f + (o - 4) / 2), V3(o, hh, 0.02f + (o - 4) / 2 + 0.2f), M_STONE, 0.6f);
        mbox(m, V3(0, y, -4.0f - (o - 4) / 2), V3(o, hh, 0.02f + (o - 4) / 2 + 0.2f), M_STONE, 0.6f);
        mbox(m, V3(4.0f + (o - 4) / 2, y, 0), V3(0.02f + (o - 4) / 2 + 0.2f, hh, o), M_STONE, 0.6f);
        mbox(m, V3(-4.0f - (o - 4) / 2, y, 0), V3(0.02f + (o - 4) / 2 + 0.2f, hh, o), M_STONE, 0.6f);
    }
    mbox(m, V3(0, TOP_Y + 0.1f, 0), V3(4.25f, 0.18f, 4.25f), M_STONE, 0.6f);
    for (int s = -1; s <= 1; s += 2) {
        mbox(m, V3(0, TOP_Y + 0.75f, s * 4.15f), V3(4.4f, 0.5f, 0.24f), M_STONE, 0.7f);
        mbox(m, V3(s * 4.15f, TOP_Y + 0.75f, 0), V3(0.24f, 0.5f, 3.95f), M_STONE, 0.7f);
        /* the coping, where the gulls sit */
        mbox(m, V3(0, TOP_Y + 1.31f, s * 4.15f), V3(4.48f, 0.07f, 0.31f), M_STONE, 0.9f);
        mbox(m, V3(s * 4.15f, TOP_Y + 1.31f, 0), V3(0.31f, 0.07f, 4.17f), M_STONE, 0.9f);
    }
    /* the spire: copper, a steep pyramid with a lip */
    {
        float b = 3.75f, y0 = TOP_Y + 0.3f, ap = 42.2f;
        v3 A = V3(0, ap, 0);
        v3 c[4] = {V3(-b, y0, b), V3(b, y0, b), V3(b, y0, -b), V3(-b, y0, -b)};
        for (int i = 0; i < 4; i++) {
            v3 p0 = c[i], p1 = c[(i + 1) % 4];
            v3 n = vnorm(vcross(vsub(p1, p0), vsub(A, p0)));
            float sl = vlen(vsub(A, vscale(vadd(p0, p1), 0.5f)));
            mv(m, p0, n, -b, 0, M_COPPER, 0); mv(m, p1, n, b, 0, M_COPPER, 0); mv(m, A, n, 0, sl, M_COPPER, 0);
        }
        /* a dormer-like louvre on each face, to break the cone */
        for (int i = 0; i < 4; i++) {
            float a = i * PI / 2;
            v3 d = V3(sinf(a), 0, cosf(a)), t = V3(cosf(a), 0, -sinf(a));
            v3 c0 = vadd(V3(0, 36.3f, 0), vscale(d, 2.45f));
            mbox_uv(m, c0, vscale(t, 0.55f), V3(0, 0.75f, 0), vscale(d, 0.5f), M_COPPER, 0.5f, c0, 63);
            mbox_uv(m, vadd(c0, vscale(d, 0.45f)), vscale(t, 0.38f), V3(0, 0.45f, 0), vscale(d, 0.08f), M_IRON, 0.2f, c0, 63);
        }
        mcyl(m, A, V3(0, 44.6f, 0), 0.07f, 0.04f, 8, 0, M_IRON, 0);
        float ball[] = {0, 43.28f - 43.6f, 0.17f, -0.27f, 0.24f, -0.15f, 0.25f, 0, 0.21f, 0.14f, 0.13f, 0.24f, 0, 0.28f};
        for (int i = 0; i < 7; i++) ball[2 * i + 1] += 0;
        mlathe(m, V3(0, 42.95f, 0), ball, 7, 16, M_BRASS, 0.7f);
    }
    /* the dials: stone surrounds, the faces drawn by the shader */
    for (int f = 0; f < 4; f++) {
        float a = f * PI / 2;
        v3 n = V3(sinf(a), 0, cosf(a)), rt = V3(cosf(a), 0, -sinf(a));
        v3 c = vadd(V3(0, DIAL_Y, 0), vscale(n, 4.06f));
        int seg = 48;
        for (int i = 0; i < seg; i++) {
            float a0 = 2 * PI * i / seg, a1 = 2 * PI * (i + 1) / seg;
            v3 d0 = vadd(vscale(rt, cosf(a0)), V3(0, sinf(a0), 0)), d1 = vadd(vscale(rt, cosf(a1)), V3(0, sinf(a1), 0));
            /* the glass */
            mv(m, c, n, 0, 0, M_DIAL, (float)f);
            mv(m, vadd(c, vscale(d0, DIAL_R)), n, cosf(a0), sinf(a0), M_DIAL, (float)f);
            mv(m, vadd(c, vscale(d1, DIAL_R)), n, cosf(a1), sinf(a1), M_DIAL, (float)f);
            /* the moulded surround */
            float R0 = DIAL_R, R1 = DIAL_R + 0.38f, pr = 0.16f;
            v3 i0 = vadd(c, vscale(d0, R0)), i1 = vadd(c, vscale(d1, R0));
            v3 o0 = vadd(c, vscale(d0, R1)), o1 = vadd(c, vscale(d1, R1));
            v3 pi0 = vadd(i0, vscale(n, pr)), pi1 = vadd(i1, vscale(n, pr)), po0 = vadd(o0, vscale(n, pr * 0.6f)), po1 = vadd(o1, vscale(n, pr * 0.6f));
            v3 bk0 = vsub(o0, vscale(n, 0.1f)), bk1 = vsub(o1, vscale(n, 0.1f));
            {
                v3 fnm = vnorm(vadd(n, vscale(vnorm(vadd(d0, d1)), 0.25f)));
                float uv[8] = {a0 * 3, 0, a1 * 3, 0, a1 * 3, 0.4f, a0 * 3, 0.4f};
                mquad(m, pi0, pi1, po1, po0, fnm, uv, M_STONE, 0.5f);
                mquad(m, i0, i1, pi1, pi0, vscale(vnorm(vadd(d0, d1)), -1), uv, M_STONE, 0.5f);
                mquad(m, po0, po1, bk1, bk0, vnorm(vadd(d0, d1)), uv, M_STONE, 0.5f);
            }
        }
    }
    /* the door, shut, iron-strapped, in the arch's recess */
    mbox(m, V3(0, 6.4f, 3.42f), V3(0.86f, 1.45f, 0.06f), M_DOOR, 0);
    for (int i = 0; i < 3; i++) mbox(m, V3(0, 5.5f + i * 0.85f, 3.49f), V3(0.8f, 0.045f, 0.012f), M_IRON, 0.4f);
    mbox(m, V3(0.55f, 6.3f, 3.5f), V3(0.06f, 0.06f, 0.03f), M_IRON, 0.3f);
    /* the window's glass, lit at night from the gear room */
    mbox(m, V3(0, 18.0f, 3.25f), V3(0.36f, 1.2f, 0.02f), M_WINDOW, 0);
    /* floors: the gear room's stone, the movement's timber bed, the ceiling,
     * the belfry floor and its frame */
    mbox(m, V3(0, GFLOOR - 0.15f, 0), V3(3, 0.15f, 3), M_STONE, 0.8f);
    for (int i = -1; i <= 1; i++) {
        float bx = i == -1 ? -1.4f : i == 0 ? 0.3f : 2.1f;
        mbox(m, V3(bx, BED_Y - 0.2f, 0), V3(0.16f, 0.2f, 3.0f), M_WOOD, 0.1f + i * 0.2f);
    }
    for (int j = 0; j < 4; j++) mbox(m, V3(-1.6f, BED_Y + 0.02f, -2.5f + j * 0.5f + (j > 1) * 3.0f), V3(1.4f, 0.03f, 0.22f), M_WOOD, 0.5f + j * 0.1f);
    mbox(m, V3(0, CEIL_Y - 0.1f, 0), V3(3, 0.1f, 3), M_WOOD, 0.3f);
    for (int j = -2; j <= 2; j++) mbox(m, V3(0, CEIL_Y - 0.35f, j * 1.25f), V3(3.0f, 0.15f, 0.12f), M_WOOD, 0.6f + j * 0.05f);
    mbox(m, V3(0, BELF_Y + 0.05f, 0), V3(3, 0.08f, 3), M_WOOD, 0.4f);
    mbox(m, V3(0, TOP_Y - 0.15f, 0), V3(3, 0.15f, 3), M_WOOD, 0.2f);
    for (int s = -1; s <= 1; s += 2) {
        mbox(m, V3(0, BELF_Y + 2.5f, s * 0.95f), V3(0.16f, 2.5f, 0.12f), M_WOOD, 0.3f);
        mbox(m, V3(0, BELF_Y + 2.5f, s * 2.5f), V3(0.16f, 2.5f, 0.12f), M_WOOD, 0.35f);
        mbox(m, V3(0, BELF_Y + 5.05f, s * 1.7f), V3(0.18f, 0.14f, 0.95f), M_WOOD, 0.4f);
    }
    /* the movement's bed: two iron rails, and a slim cock rising from each
     * to carry every arbor front and back (a flatbed, open to the eye) */
    float zl = -2.55f, zr = 2.85f;
    for (int i = 0; i < 2; i++) {
        float x = i ? X_FRONT + 0.06f : 0.02f;
        mbox(m, V3(x, BED_Y + 0.1f, (zl + zr) / 2), V3(0.06f, 0.1f, (zr - zl) / 2), M_IRON, 0.1f);
    }
    for (int k = 0; k < 2; k++) {
        float z = k ? zr : zl;
        mbox(m, V3((X_FRONT + 0.08f) / 2, BED_Y + 0.1f, z), V3((X_FRONT + 0.06f) / 2, 0.08f, 0.08f), M_IRON, 0.2f);
    }
    P2 arb[6] = {C_G1, C_G2, C_E, C_S1, C_S2, C_F};
    for (int i = 0; i < 6; i++)
        for (int side = 0; side < 2; side++) {
            float x = side ? X_FRONT + 0.06f : 0.02f, lean = side ? 0.14f : -0.14f;
            /* an A of two legs to each bearing */
            for (int leg = -1; leg <= 1; leg += 2) {
                v3 a = V3(x, BED_Y + 0.2f, arb[i].z + leg * (0.22f + 0.1f * (arb[i].y - BED_Y))), b = V3(x, arb[i].y - 0.05f, arb[i].z);
                (void)lean;
                mcyl(m, a, b, 0.026f, 0.02f, 6, 0, M_IRON, 0.25f);
            }
            mcyl(m, V3(x - 0.05f, arb[i].y, arb[i].z), V3(x + 0.07f, arb[i].y, arb[i].z), 0.075f, 0.075f, 14, 3, M_BRASS, 0.9f);
        }
    /* the pendulum hangs from a beam across the room */
    mbox(m, V3(X_PEND, PEND_TOP + 0.45f, 0), V3(0.14f, 0.18f, 3.0f), M_WOOD, 0.45f);
    mbox(m, V3(X_PEND, PEND_TOP + 0.18f, C_E.z), V3(0.08f, 0.12f, 0.12f), M_IRON, 0.3f);
    /* a candle lantern on the bed by the escapement */
    {
        v3 c = V3(1.75f, BED_Y + 0.42f, -2.1f);
        mbox(m, c, V3(0.09f, 0.14f, 0.09f), M_LAMP, 1);
        mbox(m, vadd(c, V3(0, 0.17f, 0)), V3(0.11f, 0.025f, 0.11f), M_IRON, 0.2f);
        mbox(m, vadd(c, V3(0, -0.16f, 0)), V3(0.11f, 0.025f, 0.11f), M_IRON, 0.2f);
        mcyl(m, vadd(c, V3(0, 0.19f, 0)), vadd(c, V3(0, 0.32f, 0)), 0.08f, 0.01f, 8, 0, M_IRON, 0.2f);
        mbox(m, V3(1.75f, BED_Y + 0.1f, -2.1f), V3(0.35f, 0.1f, 0.35f), M_WOOD, 0.3f);
        lamp_pos[1] = vadd(c, V3(0, 0.02f, 0));
    }
    /* a ladder to the bed, and the lamp's chain */
    for (int s = -1; s <= 1; s += 2) {
        v3 a = V3(-2.55f, GFLOOR, -2.35f + s * 0.25f), b = V3(-2.75f, BED_Y + 0.8f, -2.35f + s * 0.25f);
        mcyl(m, a, b, 0.035f, 0.035f, 6, 0, M_WOOD, 0.7f);
    }
    for (int i = 0; i < 9; i++) {
        float y = GFLOOR + 0.4f + i * 0.45f;
        float x = -2.55f - 0.2f * (y - GFLOOR) / (BED_Y + 0.8f - GFLOOR);
        mcyl(m, V3(x, y, -2.62f), V3(x, y, -2.08f), 0.022f, 0.022f, 6, 0, M_WOOD, 0.8f);
    }
    v3 lampc = V3(-1.3f, 18.6f, 2.1f);
    mcyl(m, V3(lampc.x, lampc.y + 0.3f, lampc.z), V3(lampc.x, CEIL_Y - 0.4f, lampc.z), 0.012f, 0.012f, 5, 0, M_IRON, 0);
    mbox(m, lampc, V3(0.13f, 0.2f, 0.13f), M_LAMP, 0);
    mbox(m, vadd(lampc, V3(0, 0.25f, 0)), V3(0.17f, 0.04f, 0.17f), M_IRON, 0.2f);
    mbox(m, vadd(lampc, V3(0, -0.22f, 0)), V3(0.15f, 0.03f, 0.15f), M_IRON, 0.2f);
    lamp_pos[0] = lampc;
    nlamp = 2;
    /* the bell's frame beam and the hammer's bracket */
    mbox(m, V3(1.75f, HAM_PIV.y, 0), V3(0.2f, 0.12f, 0.18f), M_IRON, 0.3f);
    mbox(m, V3(2.4f, HAM_PIV.y, 0), V3(0.6f, 0.08f, 0.08f), M_IRON, 0.3f);
}

/* the side turret on the south-west corner: round, with a conical roof,
 * and a stone housing on its south side holding a row of numbered drums */
#define SIDE_X (-5.3f)
#define SIDE_Z 3.0f
#define SIDE_R 2.1f
#define SIDE_TOP 20.6f
#define NICHE_Y 8.3f
#define HOUSE_Z (SIDE_Z + SIDE_R + 0.45f)   /* the housing's front face */
#define DRUM_R 0.42f
#define DRUM_HW 0.21f
static float drum_x(int d) { return SIDE_X + (d == 0 ? -0.98f : d == 1 ? -0.5f : d == 2 ? 0.5f : 0.98f); }
static void build_side(Mesh *m) {
    int seg = 36;
    for (int i = 0; i < seg; i++) {
        float a0 = 2 * PI * i / seg, a1 = 2 * PI * (i + 1) / seg;
        v3 d0 = V3(cosf(a0), 0, sinf(a0)), d1 = V3(cosf(a1), 0, sinf(a1));
        float y0 = PLAT_Y - 0.6f, y1 = SIDE_TOP;
        v3 p00 = V3(SIDE_X + d0.x * SIDE_R, y0, SIDE_Z + d0.z * SIDE_R), p01 = V3(SIDE_X + d1.x * SIDE_R, y0, SIDE_Z + d1.z * SIDE_R);
        v3 p10 = vadd(p00, V3(0, y1 - y0, 0)), p11 = vadd(p01, V3(0, y1 - y0, 0));
        float u0 = a0 * SIDE_R, u1 = a1 * SIDE_R;
        float uv[8] = {u0, y0, u1, y0, u1, y1, u0, y1};
        mv(m, p00, d0, uv[0], uv[1], M_STONE, 0.2f); mv(m, p11, d1, uv[4], uv[5], M_STONE, 0.2f); mv(m, p01, d1, uv[2], uv[3], M_STONE, 0.2f);
        mv(m, p00, d0, uv[0], uv[1], M_STONE, 0.2f); mv(m, p10, d0, uv[6], uv[7], M_STONE, 0.2f); mv(m, p11, d1, uv[4], uv[5], M_STONE, 0.2f);
    }
    /* corbelled top and a cone of copper */
    float cor[] = {SIDE_R, SIDE_TOP - 0.01f, SIDE_R + 0.22f, SIDE_TOP + 0.2f, SIDE_R + 0.22f, SIDE_TOP + 0.45f, SIDE_R - 0.3f, SIDE_TOP + 0.45f};
    mlathe(m, V3(SIDE_X, 0, SIDE_Z), cor, 4, seg, M_STONE, 0.6f);
    float cone[] = {SIDE_R + 0.32f, SIDE_TOP + 0.4f, SIDE_R + 0.28f, SIDE_TOP + 0.55f, 0.05f, SIDE_TOP + 5.6f, 0, SIDE_TOP + 5.62f};
    mlathe(m, V3(SIDE_X, 0, SIDE_Z), cone, 4, seg, M_COPPER, 0.3f);
    mcyl(m, V3(SIDE_X, SIDE_TOP + 5.5f, SIDE_Z), V3(SIDE_X, SIDE_TOP + 6.6f, SIDE_Z), 0.04f, 0.02f, 6, 0, M_IRON, 0);
    /* slit windows up the stair */
    for (int k = 0; k < 4; k++) {
        float a = PI * (0.75f + 0.3f * k), y = 9.5f + k * 2.6f;
        v3 d = V3(cosf(a), 0, sinf(a));
        v3 c = V3(SIDE_X + d.x * (SIDE_R - 0.02f), y, SIDE_Z + d.z * (SIDE_R - 0.02f));
        mbox_uv(m, c, vscale(V3(-d.z, 0, d.x), 0.11f), V3(0, 0.55f, 0), vscale(d, 0.04f), M_WINDOW, 0.5f, c, 63);
    }
    /* the housing: a deep stone frame standing out from the turret */
    Hole niche[1] = {{SIDE_X, 1.24f, NICHE_Y - 0.6f, NICHE_Y + 0.6f, 0}};
    panel(m, V3(0, 0, HOUSE_Z), V3(1, 0, 0), V3(0, 0, 1), SIDE_X - 1.55f, SIDE_X + 1.55f, NICHE_Y - 1.0f, NICHE_Y + 0.95f, 1.1f, niche, 1, M_STONE, 0.4f, 15);
    mbox(m, V3(SIDE_X, NICHE_Y, HOUSE_Z - 1.0f), V3(1.3f, 0.65f, 0.1f), M_RUBBLE, 0.5f);
    /* a sloping hood, a sill, and a plinth down to the rock */
    mbox(m, V3(SIDE_X, NICHE_Y + 1.05f, HOUSE_Z - 0.5f), V3(1.7f, 0.1f, 0.75f), M_STONE, 0.6f);
    mbox(m, V3(SIDE_X, NICHE_Y - 1.0f - (NICHE_Y - 1.0f - PLAT_Y + 0.4f) / 2, HOUSE_Z - 0.55f), V3(1.4f, (NICHE_Y - 1.0f - PLAT_Y + 0.4f) / 2, 0.5f), M_STONE, 0.3f);
    mbox(m, V3(SIDE_X, NICHE_Y - 0.62f, HOUSE_Z + 0.03f), V3(1.35f, 0.05f, 0.1f), M_STONE, 0.7f);
    /* the drums' axle and the studs between hours and minutes */
    float zc = HOUSE_Z - DRUM_R - 0.06f;
    mcyl(m, V3(SIDE_X - 1.25f, NICHE_Y, zc), V3(SIDE_X + 1.25f, NICHE_Y, zc), 0.04f, 0.04f, 10, 3, M_STEEL, 0.3f);
    for (int s2 = -1; s2 <= 1; s2 += 2)
        mcyl(m, V3(SIDE_X, NICHE_Y + s2 * 0.13f, zc + DRUM_R - 0.05f), V3(SIDE_X, NICHE_Y + s2 * 0.13f, zc + DRUM_R + 0.03f), 0.05f, 0.05f, 10, 2, M_BRASS, 0.5f);
    mbox(m, V3(SIDE_X, NICHE_Y, zc), V3(0.08f, DRUM_R * 0.97f, DRUM_R * 0.97f), M_IRON, 0.3f);
    /* the lever's quadrant, on the housing's east side */
    v3 lp = V3(SIDE_X + 1.62f, NICHE_Y - 0.2f, HOUSE_Z - 0.45f);
    mbox(m, lp, V3(0.05f, 0.5f, 0.24f), M_IRON, 0.35f);
}

/* the causeway: big dressed blocks on the reef, flagged on top, with
 * lanterns on iron posts; and the stairs up the rock */
static void build_causeway(Mesh *m) {
    float z = CW_Z0 - 0.2f;
    int i = 0;
    while (z < CW_Z1 + 1) {
        float len = 1.0f + 0.4f * hash1(i, 40);
        float zc = z + len / 2;
        v3 c = V3(cw_x(zc), 0, zc);
        float dx = (cw_x(zc + 0.5f) - cw_x(zc - 0.5f));
        v3 T = vnorm(V3(dx, 0, 1)), S = V3(T.z, 0, -T.x);
        float top = cw_top(zc) + 0.03f * (hash1(i, 41) - 0.5f);
        /* some blocks have settled or been nudged by the sea */
        float sink = hash1(i, 42) < 0.1f ? 0.08f * hash1(i, 43) : 0;
        float bot = -2.6f, hh = (top - sink - bot) / 2;
        v3 cc = vadd(c, V3(0.04f * (hash1(i, 44) - 0.5f), bot + hh, 0));
        m4 R = m4rot(T, 0.03f * (hash1(i, 45) - 0.5f));
        mbox_uv(m, cc, vscale(S, 1.32f), m4dir(R, V3(0, hh, 0)), vscale(T, len / 2 - 0.01f), M_PAVE, hash1(i, 46),
                V3(0, cc.y, zc), 63);
        z += len;
        i++;
    }
    /* the stairs */
    float dz = (CW_Z0 - ST_Z1) / ST_N;
    for (int k = 0; k < ST_N; k++) {
        float zc = CW_Z0 - (k + 0.5f) * dz, top = stair_top(k), bot = top - 1.6f;
        v3 c = V3(cw_x(zc), (top + bot) / 2, zc);
        mbox_uv(m, c, V3(1.15f, 0, 0), V3(0, (top - bot) / 2, 0), V3(0, 0, dz / 2 + 0.02f), M_PAVE, hash1(k, 47),
                vadd(c, V3(0, 0, 0)), 63);
    }
    /* a flagged apron before the door */
    mbox(m, V3(0, PLAT_Y - 0.05f, 5.3f), V3(1.7f, 0.12f, 1.1f), M_PAVE, 0.3f);
    /* lanterns */
    for (int k = 0; k < 9; k++) {
        float zl = 26 + k * 15.0f;
        float side = (k & 1) ? 1.0f : -1.0f;
        float dx = (cw_x(zl + 0.5f) - cw_x(zl - 0.5f));
        v3 T = vnorm(V3(dx, 0, 1)), S = V3(T.z, 0, -T.x);
        v3 b = vadd(V3(cw_x(zl), cw_top(zl), zl), vscale(S, side * 1.12f));
        float lean = 0.04f * (hash1(k, 48) - 0.5f);
        v3 top = vadd(b, V3(lean, 2.15f, 0));
        mcyl(m, vadd(b, V3(0, -0.02f, 0)), vadd(b, V3(0, 0.25f, 0)), 0.11f, 0.08f, 10, 2, M_IRON, 0.5f);
        mcyl(m, b, top, 0.055f, 0.045f, 8, 0, M_IRON, 0.5f);
        /* the arm and the lantern hanging over the path */
        v3 arm = vadd(top, vscale(S, -side * 0.42f));
        mcyl(m, top, arm, 0.025f, 0.025f, 6, 0, M_IRON, 0.5f);
        v3 lc = vadd(arm, V3(0, -0.28f, 0));
        mbox(m, lc, V3(0.11f, 0.17f, 0.11f), M_LANTERN, (float)k);
        mbox(m, vadd(lc, V3(0, 0.2f, 0)), V3(0.14f, 0.03f, 0.14f), M_IRON, 0.5f);
        mcyl(m, vadd(lc, V3(0, 0.22f, 0)), vadd(lc, V3(0, 0.36f, 0)), 0.12f, 0.02f, 8, 0, M_IRON, 0.5f);
        mbox(m, vadd(lc, V3(0, -0.19f, 0)), V3(0.12f, 0.025f, 0.12f), M_IRON, 0.5f);
        if (nlamp < MAXL) lamp_pos[nlamp++] = lc;
    }
}

/* the moving parts, each about its own pivot */
static void build_dynamic(Mesh *m) {
    int id;
    /* going train: the great wheel with its barrel and the cable's turns */
    id = part_begin(D_G1, V3(X_WHEEL, C_G1.y, C_G1.z));
    mgear(m, 96, 1.10f, 0.09f, 6, 0, M_BRASS, 0.1f);
    mcyl(m, V3(-0.55f, 0, 0), V3(-0.08f, 0, 0), 0.42f, 0.42f, 28, 3, M_WOOD, 0.9f);
    for (int s = 0; s < 2; s++) mcyl(m, V3(s ? -0.58f : -0.1f, 0, 0), V3(s ? -0.52f : -0.04f, 0, 0), 0.5f, 0.5f, 28, 3, M_IRON, 0.4f);
    mcyl(m, V3(-0.6f, 0, 0), V3(0.6f, 0, 0), 0.04f, 0.04f, 10, 3, M_STEEL, 0);
    part_end(id);
    id = part_begin(D_G2, V3(X_WHEEL, C_G2.y, C_G2.z));
    mgear(m, 72, 0.75f, 0.07f, 5, 0, M_BRASS, 0.3f);
    m4 T = m4trans(V3(-0.32f, 0, 0));
    int f0 = m->n;
    mgear(m, 12, 0.1375f, 0.22f, 0, 0, M_STEEL, 0.2f);
    for (int i = f0; i < m->n; i++) { float *v = m->v + (size_t)i * VSZ; v3 p = m4pt(T, V3(v[0], v[1], v[2])); v[0] = p.x; v[1] = p.y; v[2] = p.z; }
    mcyl(m, V3(-0.55f, 0, 0), V3(0.55f, 0, 0), 0.03f, 0.03f, 10, 3, M_STEEL, 0);
    part_end(id);
    id = part_begin(D_E, V3(X_FRONT - 0.1f, C_E.y, C_E.z));
    mgear(m, 30, 0.40f, 0.04f, 4, 1, M_BRASS, 0.6f);
    T = m4trans(V3(X_WHEEL - X_FRONT + 0.1f, 0, 0));
    f0 = m->n;
    mgear(m, 8, 0.0833f, 0.2f, 0, 0, M_STEEL, 0.2f);
    for (int i = f0; i < m->n; i++) { float *v = m->v + (size_t)i * VSZ; v3 p = m4pt(T, V3(v[0], v[1], v[2])); v[0] = p.x; v[1] = p.y; v[2] = p.z; }
    mcyl(m, V3(-1.1f, 0, 0), V3(0.2f, 0, 0), 0.025f, 0.025f, 10, 3, M_STEEL, 0);
    part_end(id);
    /* the anchor: a bridge over the wheel with two pallets, and the crutch */
    id = part_begin(D_ANCHOR, V3(X_FRONT + 0.0f, C_E.y + 0.62f, C_E.z));
    {
        float pr = 0.62f;
        for (int s = -1; s <= 1; s += 2) {
            float a = s * 0.62f;              /* the pallets sit ~35 deg either side of the top */
            v3 tip = V3(0, C_E.y + 0.47f * cosf(a) - (C_E.y + pr), 0.47f * sinf(a));
            v3 mid = V3(0, -0.05f, s * 0.2f);
            mcyl(m, V3(0, 0, 0), mid, 0.03f, 0.03f, 8, 0, M_STEEL, 0.3f);
            mcyl(m, mid, tip, 0.028f, 0.02f, 8, 0, M_STEEL, 0.3f);
            mbox_uv(m, tip, V3(0.03f, 0, 0), V3(0, 0.05f, 0), V3(0, 0, 0.022f), M_STEEL, 0.9f, tip, 63);
        }
        mcyl(m, V3(-0.04f, 0, 0), V3(0.06f, 0, 0), 0.07f, 0.07f, 14, 3, M_IRON, 0.2f);
        mcyl(m, V3(0.06f, 0, 0), V3(X_PEND - X_FRONT, 0, 0), 0.025f, 0.025f, 8, 0, M_STEEL, 0);
        mcyl(m, V3(X_PEND - X_FRONT, 0, 0), V3(X_PEND - X_FRONT, -1.25f, 0), 0.022f, 0.022f, 8, 0, M_STEEL, 0.2f);
        mbox(m, V3(X_PEND - X_FRONT - 0.04f, -1.25f, 0), V3(0.05f, 0.04f, 0.05f), M_BRASS, 0.5f);
    }
    part_end(id);
    /* the pendulum: a wooden rod and a great iron bob */
    id = part_begin(D_PEND, V3(X_PEND, PEND_TOP, C_E.z));
    mbox(m, V3(0, -0.15f, 0), V3(0.025f, 0.15f, 0.06f), M_STEEL, 0.2f);
    mcyl(m, V3(0, -0.28f, 0), V3(0, -PEND_L + 0.2f, 0), 0.035f, 0.035f, 10, 0, M_WOOD, 0.95f);
    {
        float bob[] = {0, -0.14f, 0.25f, -0.13f, 0.36f, -0.08f, 0.39f, 0, 0.36f, 0.08f, 0.25f, 0.13f, 0, 0.14f};
        int f1 = m->n;
        mlathe(m, V3(0, 0, 0), bob, 7, 32, M_IRON, 0.6f);
        /* lathe is about y; turn it to face the room (axis along x) */
        m4 R = m4mul(m4trans(V3(0, -PEND_L, 0)), m4rot(V3(0, 0, 1), PI / 2));
        for (int i = f1; i < m->n; i++) {
            float *v = m->v + (size_t)i * VSZ;
            v3 p = m4pt(R, V3(v[0], v[1], v[2])), n = m4dir(R, V3(v[3], v[4], v[5]));
            v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = n.x; v[4] = n.y; v[5] = n.z;
        }
        mcyl(m, V3(0, -PEND_L - 0.2f, 0), V3(0, -PEND_L - 0.36f, 0), 0.03f, 0.03f, 8, 2, M_BRASS, 0.3f);
    }
    part_end(id);
    /* striking train: great wheel, its lifting pins, the count wheel, barrel */
    id = part_begin(D_S1, V3(X_WHEEL, C_S1.y, C_S1.z));
    mgear(m, 96, 1.10f, 0.09f, 6, 0, M_BRASS, 0.15f);
    for (int k = 0; k < 8; k++) {
        float a = k * 2 * PI / 8;
        v3 p = V3(0.04f, 0.84f * cosf(a), 0.84f * sinf(a));
        mcyl(m, p, vadd(p, V3(0.2f, 0, 0)), 0.03f, 0.03f, 8, 2, M_STEEL, 0.4f);
    }
    mcyl(m, V3(-0.55f, 0, 0), V3(-0.08f, 0, 0), 0.42f, 0.42f, 28, 3, M_WOOD, 0.85f);
    for (int s = 0; s < 2; s++) mcyl(m, V3(s ? -0.58f : -0.1f, 0, 0), V3(s ? -0.52f : -0.04f, 0, 0), 0.5f, 0.5f, 28, 3, M_IRON, 0.45f);
    mcyl(m, V3(-0.6f, 0, 0), V3(0.65f, 0, 0), 0.04f, 0.04f, 10, 3, M_STEEL, 0);
    T = m4trans(V3(0.58f, 0, 0));
    f0 = m->n;
    mgear(m, 11, 0.36f, 0.03f, 4, 0, M_BRASS, 0.7f);    /* the count wheel: notched */
    for (int i = f0; i < m->n; i++) { float *v = m->v + (size_t)i * VSZ; v3 p = m4pt(T, V3(v[0], v[1], v[2])); v[0] = p.x; v[1] = p.y; v[2] = p.z; }
    part_end(id);
    id = part_begin(D_S2, V3(X_WHEEL, C_S2.y, C_S2.z));
    mgear(m, 60, 0.62f, 0.06f, 5, 0, M_BRASS, 0.4f);
    T = m4trans(V3(-0.3f, 0, 0));
    f0 = m->n;
    mgear(m, 12, 0.1375f, 0.2f, 0, 0, M_STEEL, 0.2f);
    for (int i = f0; i < m->n; i++) { float *v = m->v + (size_t)i * VSZ; v3 p = m4pt(T, V3(v[0], v[1], v[2])); v[0] = p.x; v[1] = p.y; v[2] = p.z; }
    mcyl(m, V3(-0.55f, 0, 0), V3(0.55f, 0, 0), 0.03f, 0.03f, 10, 3, M_STEEL, 0);
    part_end(id);
    /* the fly: an air brake that spins while it strikes */
    id = part_begin(D_FLY, V3(X_WHEEL, C_F.y, C_F.z));
    mgear(m, 8, 0.0827f, 0.18f, 0, 0, M_STEEL, 0.2f);
    mcyl(m, V3(-0.5f, 0, 0), V3(0.95f, 0, 0), 0.025f, 0.025f, 8, 3, M_STEEL, 0);
    for (int s = -1; s <= 1; s += 2) {
        mbox(m, V3(0.75f, s * 0.29f, 0), V3(0.17f, 0.22f, 0.008f), M_IRON, 0.8f);
        mbox(m, V3(0.75f, s * 0.04f, 0), V3(0.17f, 0.04f, 0.02f), M_BRASS, 0.8f);
    }
    part_end(id);
    /* the hammer's tail lever: pins lift it, it pulls the wire to the bell */
    id = part_begin(D_TAIL, V3(X_FRONT + 0.15f, 15.05f, 2.25f));
    mcyl(m, V3(-0.05f, 0, 0), V3(0.06f, 0, 0), 0.06f, 0.06f, 12, 3, M_IRON, 0.3f);
    mcyl(m, V3(0, 0, 0), V3(0, -0.45f, -0.62f), 0.025f, 0.02f, 8, 0, M_STEEL, 0.3f);
    mcyl(m, V3(0, 0, 0), V3(0, 0.12f, 0.6f), 0.025f, 0.025f, 8, 0, M_STEEL, 0.3f);
    part_end(id);
    /* the wire (a rod here) up the south wall through the floors to the bell */
    id = part_begin(D_ROD, V3(X_FRONT + 0.15f, 15.17f, 2.85f));
    mcyl(m, V3(0, 0, 0), V3(0, HAM_PIV.y + 0.35f - 15.2f, 0), 0.015f, 0.015f, 6, 0, M_STEEL, 0);
    part_end(id);
    /* the hammer */
    id = part_begin(D_HAMMER, HAM_PIV);
    mcyl(m, V3(0, 0, -0.08f), V3(0, 0, 0.08f), 0.06f, 0.06f, 10, 3, M_IRON, 0.2f);
    mcyl(m, V3(0, 0, 0), V3(-0.5f, -0.78f, 0), 0.035f, 0.035f, 8, 0, M_IRON, 0.3f);
    mcyl(m, V3(-0.5f, -0.78f, -0.11f), V3(-0.5f, -0.78f, 0.11f), 0.1f, 0.1f, 14, 3, M_IRON, 0.5f);
    mcyl(m, V3(0, 0, 0), V3(-0.2f, 0.22f, 0), 0.03f, 0.03f, 8, 0, M_IRON, 0.3f);
    part_end(id);
    /* the bell: bronze, on its headstock */
    id = part_begin(D_BELL, V3(0, BELL_PIV_Y, 0));
    {
        float pr[] = {0, -0.1f, 0.3f, -0.12f, 0.42f, -0.3f, 0.46f, -0.6f, 0.5f, -0.9f, 0.58f, -1.15f, 0.7f, -1.4f, 0.8f, -1.58f,
                      0.86f, -1.72f, 0.95f, -1.76f, 0.98f, -1.70f, 0.93f, -1.58f, 0.81f, -1.38f, 0.69f, -1.15f, 0.61f, -0.9f,
                      0.56f, -0.6f, 0.53f, -0.3f, 0.47f, -0.08f, 0.36f, 0.0f, 0, 0.0f};
        int f1 = m->n;
        mlathe(m, V3(0, 0, 0), pr, 20, 40, M_BRONZE, 0);
        for (int i = f1; i < m->n; i++) m->v[(size_t)i * VSZ + 1] -= 0.28f;
        mbox(m, V3(0, -0.1f, 0), V3(0.1f, 0.12f, 0.06f), M_IRON, 0.3f);
        mbox(m, V3(0, 0, 0), V3(0.2f, 0.16f, 0.95f), M_WOOD, 0.2f);
        for (int s = -1; s <= 1; s += 2) mbox(m, V3(0, 0, s * 0.5f), V3(0.21f, 0.17f, 0.04f), M_IRON, 0.3f);
        mcyl(m, V3(0, 0, -1.2f), V3(0, 0, 1.2f), 0.05f, 0.05f, 10, 3, M_STEEL, 0);
    }
    part_end(id);
    /* the leading-off rod up to the dials' motion work */
    id = part_begin(D_LEAD, V3(0.25f, 16.8f, -0.1f));
    mcyl(m, V3(0, 0, 0), V3(0, CEIL_Y - 16.8f, 0), 0.03f, 0.03f, 8, 0, M_STEEL, 0);
    for (int s = 0; s < 2; s++) {
        int g0 = m->n;
        mgear(m, 16, 0.14f, 0.04f, 0, 0, M_BRASS, 0.5f);
        m4 R = m4mul(m4trans(V3(0, s ? 0.5f : 0.08f, 0)), m4rot(V3(0, 0, 1), PI / 2));
        for (int i = g0; i < m->n; i++) {
            float *v = m->v + (size_t)i * VSZ;
            v3 p = m4pt(R, V3(v[0], v[1], v[2])), n = m4dir(R, V3(v[3], v[4], v[5]));
            v[0] = p.x; v[1] = p.y; v[2] = p.z; v[3] = n.x; v[4] = n.y; v[5] = n.z;
        }
    }
    part_end(id);
    /* weights and their cables (unit length, scaled) */
    for (int s = 0; s < 2; s++) {
        id = part_begin(s ? D_WEIGHT_S : D_WEIGHT_G, V3(0, 0, 0));
        mcyl(m, V3(0, -0.95f, 0), V3(0, 0, 0), 0.27f, 0.27f, 20, 3, M_IRON, 0.6f + 0.1f * s);
        mcyl(m, V3(0, 0, 0), V3(0, 0.25f, 0), 0.05f, 0.03f, 8, 0, M_IRON, 0.3f);
        part_end(id);
        id = part_begin(s ? D_CABLE_S : D_CABLE_G, V3(0, 0, 0));
        mcyl(m, V3(0, 0, 0), V3(0, 1, 0), 0.018f, 0.018f, 6, 0, M_ROPE, 0);
        part_end(id);
    }
    /* the drums: ten-sided, a digit on each face */
    for (int d = 0; d < 4; d++) {
        float zc = HOUSE_Z - DRUM_R - 0.06f;
        id = part_begin(D_DRUM0 + d, V3(drum_x(d), NICHE_Y, zc));
        float hw = DRUM_HW, ap = 2 * PI / 10, rf = DRUM_R;
        for (int k = 0; k < 10; k++) {
            /* face k faces up from the front by k * 36 deg (see draw: rotation k * 36 shows it) */
            float a0 = (k - 0.5f) * ap, a1 = (k + 0.5f) * ap;
            v3 p0 = V3(0, rf * sinf(a0), rf * cosf(a0)), p1 = V3(0, rf * sinf(a1), rf * cosf(a1));
            v3 n = V3(0, sinf(k * ap), cosf(k * ap));
            float uv[8] = {0, 0, 1, 0, 1, 1, 0, 1};
            mquad(m, vadd(p0, V3(-hw, 0, 0)), vadd(p0, V3(hw, 0, 0)), vadd(p1, V3(hw, 0, 0)), vadd(p1, V3(-hw, 0, 0)), n, uv, M_DRUM, (float)k);
        }
        for (int s = -1; s <= 1; s += 2) mcyl(m, V3(s * hw, 0, 0), V3(s * (hw + 0.025f), 0, 0), rf * 1.04f, rf * 1.04f, 20, s < 0 ? 1 : 2, M_BRASS, 0.6f);
        part_end(id);
    }
    /* the lever by the niche */
    /* (it swings in the plane of the housing's side, about +x) */
    id = part_begin(D_LEVER, V3(SIDE_X + 1.7f, NICHE_Y - 0.5f, HOUSE_Z - 0.45f));
    mcyl(m, V3(-0.06f, 0, 0), V3(0.08f, 0, 0), 0.1f, 0.1f, 12, 3, M_IRON, 0.4f);
    mcyl(m, V3(0.05f, 0, 0), V3(0.05f, 1.35f, 0), 0.04f, 0.03f, 8, 0, M_IRON, 0.5f);
    mcyl(m, V3(0.05f, 0, 0), V3(0.05f, -0.35f, 0), 0.04f, 0.04f, 8, 2, M_IRON, 0.5f);
    {
        float knob[] = {0, -0.08f, 0.065f, -0.055f, 0.075f, 0, 0.06f, 0.065f, 0, 0.09f};
        mlathe(m, V3(0.05f, 1.42f, 0), knob, 5, 14, M_BRASS, 0.3f);
    }
    part_end(id);
    /* an armillary on the spire: rings that turn */
    for (int r = 0; r < 2; r++) {
        id = part_begin(r ? D_ARMIL2 : D_ARMIL, V3(0, 43.95f, 0));
        int seg = 28;
        float R = r ? 0.5f : 0.62f;
        for (int i = 0; i < seg; i++) {
            float a0 = 2 * PI * i / seg, a1 = 2 * PI * (i + 1) / seg;
            v3 p0 = r ? V3(R * cosf(a0), R * sinf(a0), 0) : V3(R * cosf(a0), 0, R * sinf(a0));
            v3 p1 = r ? V3(R * cosf(a1), R * sinf(a1), 0) : V3(R * cosf(a1), 0, R * sinf(a1));
            mcyl(m, p0, p1, 0.022f, 0.022f, 6, 0, M_BRASS, 0.2f + r * 0.3f);
        }
        if (!r) mcyl(m, V3(0, -0.62f, 0), V3(0, 0.75f, 0), 0.015f, 0.015f, 6, 0, M_IRON, 0);
        if (r) mcyl(m, V3(-0.62f, 0, 0), V3(0.62f, 0, 0), 0.02f, 0.008f, 6, 0, M_IRON, 0);
        part_end(id);
    }
}

/* ---- the ground as meshes: finer where it's looked at closely ---- */
typedef struct { float x0, z0, x1, z1, step; } Patch;
static const Patch patches[] = {
    {-48, -48, 48, 48, 0.6f},          /* the islet and its skerries */
    {-40, 48, 44, 196, 1.0f},          /* the reef along the causeway, and its beach */
    {-150, 100, 180, 340, 1.6f},       /* the shore the camera stands on */
    {-720, -640, 720, 880, 8.0f},      /* the bay and the headlands */
};
#define NPATCH 4
static int in_patch(int p, float x, float z) {
    return x > patches[p].x0 && x < patches[p].x1 && z > patches[p].z0 && z < patches[p].z1;
}
static GLuint vao_terr_;
/* fn(i) for i in [0, n), across the cores (as flightsim) */
typedef struct { void (*fn)(int, void *); void *arg; int n, next; pthread_mutex_t mu; } Job;
static void *job_worker(void *p) {
    Job *j = p;
    for (;;) {
        pthread_mutex_lock(&j->mu);
        int i = j->next++;
        pthread_mutex_unlock(&j->mu);
        if (i >= j->n) return NULL;
        j->fn(i, j->arg);
    }
}
static void parallel_for(int n, void (*fn)(int, void *), void *arg) {
    Job j = {fn, arg, n, 0, PTHREAD_MUTEX_INITIALIZER};
    pthread_t th[16];
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    int k = nc < 1 ? 1 : nc > 16 ? 16 : (int)nc;
    for (int i = 1; i < k; i++) pthread_create(&th[i], NULL, job_worker, &j);
    job_worker(&j);
    for (int i = 1; i < k; i++) pthread_join(th[i], NULL);
}

/* The ground in chunks of a patch's grid, each indexed and with its bounds,
 * so what's off screen isn't drawn. */
#define CHUNK_CELLS 32
typedef struct {
    int patch, ci, cj;
    float *v; int nv;                   /* built in parallel, then packed */
    uint32_t *idx; int ni;
    int first, count;                   /* in the packed index buffer */
    float bmin[3], bmax[3];
} Chunk;
#define MAXCHUNK 2048
static Chunk chunks[MAXCHUNK];
static int nchunk;
static int terr_nv, terr_ni;

static void chunk_build(int k, void *arg) {
    (void)arg;
    Chunk *c = &chunks[k];
    const Patch *P = &patches[c->patch];
    int nx = (int)((P->x1 - P->x0) / P->step), nz = (int)((P->z1 - P->z0) / P->step);
    int i0 = c->ci * CHUNK_CELLS, j0 = c->cj * CHUNK_CELLS;
    int ni = nx - i0 < CHUNK_CELLS ? nx - i0 : CHUNK_CELLS, nj = nz - j0 < CHUNK_CELLS ? nz - j0 : CHUNK_CELLS;
    int gw = ni + 3, gh = nj + 3;       /* a margin of one for the normals */
    float *g = malloc(sizeof(float) * (size_t)gw * gh);
    for (int j = 0; j < gh; j++)
        for (int i = 0; i < gw; i++) g[j * gw + i] = terrain(P->x0 + (i0 + i - 1) * P->step, P->z0 + (j0 + j - 1) * P->step);
    int vw = ni + 1, vh = nj + 1;
    c->v = malloc(sizeof(float) * VSZ * ((size_t)vw * vh + 8 * (ni + nj) + 8));
    c->idx = malloc(sizeof(uint32_t) * (6 * (size_t)ni * nj + 24 * (ni + nj) + 24));
    c->nv = c->ni = 0;
    for (int a = 0; a < 3; a++) { c->bmin[a] = 1e9f; c->bmax[a] = -1e9f; }
    for (int j = 0; j < vh; j++)
        for (int i = 0; i < vw; i++) {
            float x = P->x0 + (i0 + i) * P->step, z = P->z0 + (j0 + j) * P->step, h = g[(j + 1) * gw + i + 1];
            float dx = (g[(j + 1) * gw + i + 2] - g[(j + 1) * gw + i]) / (2 * P->step);
            float dz = (g[(j + 2) * gw + i + 1] - g[j * gw + i + 1]) / (2 * P->step);
            v3 n = vnorm(V3(-dx, 1, -dz));
            float *o = c->v + (size_t)c->nv++ * VSZ;
            o[0] = x; o[1] = h; o[2] = z; o[3] = n.x; o[4] = n.y; o[5] = n.z; o[6] = x; o[7] = z; o[8] = 0; o[9] = (float)c->patch;
        }
    for (int j = 0; j < nj; j++)
        for (int i = 0; i < ni; i++) {
            float cx = P->x0 + (i0 + i + 0.5f) * P->step, cz = P->z0 + (j0 + j + 0.5f) * P->step;
            int skip = 0;
            for (int q = 0; q < c->patch; q++) if (in_patch(q, cx, cz)) skip = 1;
            if (skip || (fabsf(cx) < 3.9f && fabsf(cz) < 3.9f)) continue;
            uint32_t a = (uint32_t)(j * vw + i), b = a + 1, d = a + (uint32_t)vw, e = d + 1;
            float hm = fmaxf(fmaxf(c->v[a * VSZ + 1], c->v[b * VSZ + 1]), fmaxf(c->v[d * VSZ + 1], c->v[e * VSZ + 1]));
            if (hm < -7.0f) continue;
            uint32_t q[6] = {a, d, e, a, e, b};
            memcpy(c->idx + c->ni, q, sizeof q);
            c->ni += 6;
            uint32_t qq[4] = {a, b, e, d};
            for (int t = 0; t < 4; t++) {
                const float *o = c->v + (size_t)qq[t] * VSZ;
                for (int ax = 0; ax < 3; ax++) { c->bmin[ax] = fminf(c->bmin[ax], o[ax]); c->bmax[ax] = fmaxf(c->bmax[ax], o[ax]); }
            }
        }
    /* skirts on the patch's outer edges hide the cracks against the next one */
    if (c->patch < NPATCH - 1) {
        int edges[4] = {j0 == 0, i0 + ni == nx, j0 + nj == nz, i0 == 0};
        for (int e = 0; e < 4; e++) {
            if (!edges[e]) continue;
            int len = e & 1 ? nj : ni;
            for (int k2 = 0; k2 < len; k2++) {
                int ia, ib;
                if (e == 0) { ia = k2; ib = k2 + 1; }
                else if (e == 1) { ia = k2 * vw + ni; ib = (k2 + 1) * vw + ni; }
                else if (e == 2) { ia = nj * vw + k2; ib = nj * vw + k2 + 1; }
                else { ia = k2 * vw; ib = (k2 + 1) * vw; }
                uint32_t base = (uint32_t)c->nv;
                for (int s = 0; s < 2; s++) {
                    const float *src = c->v + (size_t)(s ? ib : ia) * VSZ;
                    float *o = c->v + (size_t)c->nv++ * VSZ;
                    memcpy(o, src, sizeof(float) * VSZ);
                    o[1] -= 1.5f;
                }
                uint32_t q[6] = {(uint32_t)ia, base, base + 1, (uint32_t)ia, base + 1, (uint32_t)ib};
                memcpy(c->idx + c->ni, q, sizeof q);
                c->ni += 6;
            }
        }
    }
    free(g);
}

static GLuint vbo_terr, ibo_terr;
static void build_terrain(void) {
    nchunk = 0;
    for (int p = 0; p < NPATCH; p++) {
        const Patch *P = &patches[p];
        int nx = (int)((P->x1 - P->x0) / P->step), nz = (int)((P->z1 - P->z0) / P->step);
        for (int cj = 0; cj * CHUNK_CELLS < nz; cj++)
            for (int ci = 0; ci * CHUNK_CELLS < nx; ci++) {
                if (nchunk == MAXCHUNK) die("too many terrain chunks");
                chunks[nchunk++] = (Chunk){.patch = p, .ci = ci, .cj = cj};
            }
    }
    parallel_for(nchunk, chunk_build, NULL);
    /* pack */
    terr_nv = terr_ni = 0;
    for (int k = 0; k < nchunk; k++) { terr_nv += chunks[k].nv; terr_ni += chunks[k].ni; }
    float *v = malloc(sizeof(float) * VSZ * (size_t)terr_nv);
    uint32_t *ix = malloc(sizeof(uint32_t) * (size_t)terr_ni);
    int nv = 0, ni = 0;
    for (int k = 0; k < nchunk; k++) {
        Chunk *c = &chunks[k];
        memcpy(v + (size_t)nv * VSZ, c->v, sizeof(float) * VSZ * (size_t)c->nv);
        for (int i = 0; i < c->ni; i++) ix[ni + i] = c->idx[i] + (uint32_t)nv;
        c->first = ni; c->count = c->ni;
        nv += c->nv; ni += c->ni;
        free(c->v); free(c->idx);
    }
    glGenVertexArrays(1, &vao_terr_);
    glBindVertexArray(vao_terr_);
    glGenBuffers(1, &vbo_terr);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_terr);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * VSZ * (size_t)nv), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)12);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)24);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)32);
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray((GLuint)i);
    glGenBuffers(1, &ibo_terr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_terr);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * (size_t)ni), ix, GL_STATIC_DRAW);
    glBindVertexArray(0);
    free(v); free(ix);
}

/* is an axis-aligned box (possibly mirrored) at least partly inside clip space? */
static int box_visible(const float *bmin, const float *bmax, m4 vp) {
    int out[6] = {0};
    for (int c = 0; c < 8; c++) {
        float x = c & 1 ? bmax[0] : bmin[0], y = c & 2 ? bmax[1] : bmin[1], z = c & 4 ? bmax[2] : bmin[2];
        float cx = vp.m[0] * x + vp.m[4] * y + vp.m[8] * z + vp.m[12];
        float cy = vp.m[1] * x + vp.m[5] * y + vp.m[9] * z + vp.m[13];
        float cz = vp.m[2] * x + vp.m[6] * y + vp.m[10] * z + vp.m[14];
        float cw = vp.m[3] * x + vp.m[7] * y + vp.m[11] * z + vp.m[15];
        out[0] += cx < -cw; out[1] += cx > cw; out[2] += cy < -cw; out[3] += cy > cw; out[4] += cz < -cw; out[5] += cz > cw;
    }
    for (int i = 0; i < 6; i++) if (out[i] == 8) return 0;
    return 1;
}

/* heights for the water (depth, foam): a coarse map of the whole bay and a
 * fine one of the islet and causeway */
#define HC_N 512
#define HC_X0 (-720.0f)
#define HC_Z0 (-640.0f)
#define HC_S 1440.0f
#define HF_NX 384
#define HF_NZ 896
#define HF_X0 (-48.0f)
#define HF_Z0 (-48.0f)
#define HF_SX 96.0f
#define HF_SZ 224.0f
static GLuint tex_hc, tex_hf, tex_noise, tex_digits;

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum filt, GLenum wrap) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)wrap);
    if (filt == GL_LINEAR_MIPMAP_LINEAR) glGenerateMipmap(GL_TEXTURE_2D);
    return t;
}

static float *hc_map, *hf_map;
static void hc_row(int j, void *arg) {
    (void)arg;
    for (int i = 0; i < HC_N; i++) hc_map[j * HC_N + i] = solid_top(HC_X0 + (i + 0.5f) * HC_S / HC_N, HC_Z0 + (j + 0.5f) * HC_S / HC_N);
}
static void hf_row(int j, void *arg) {
    (void)arg;
    for (int i = 0; i < HF_NX; i++) hf_map[j * HF_NX + i] = solid_top(HF_X0 + (i + 0.5f) * HF_SX / HF_NX, HF_Z0 + (j + 0.5f) * HF_SZ / HF_NZ);
}
static void build_heights(void) {
    hc_map = malloc(sizeof(float) * HC_N * HC_N);
    hf_map = malloc(sizeof(float) * HF_NX * HF_NZ);
    parallel_for(HC_N, hc_row, NULL);
    parallel_for(HF_NZ, hf_row, NULL);
    tex_hc = mk_tex(GL_R32F, HC_N, HC_N, GL_RED, GL_FLOAT, hc_map, GL_LINEAR, GL_CLAMP_TO_EDGE);
    tex_hf = mk_tex(GL_R32F, HF_NX, HF_NZ, GL_RED, GL_FLOAT, hf_map, GL_LINEAR, GL_CLAMP_TO_EDGE);
    free(hc_map); free(hf_map);
}

/* tileable value noise, four frequencies in four channels */
/* the sea's chop: a tile of many crossing wind waves, as slopes; RG long,
 * BA short. Integer wave numbers so it tiles. */
static GLuint tex_wave;
static void build_waves(void) {
    enum { N = 256, M = 48 };
    float *px = calloc((size_t)N * N * 4, sizeof(float));
    float kx[2][M], kz[2][M], am[2][M], ph[2][M];
    for (int set = 0; set < 2; set++)
        for (int i = 0; i < M; i++) {
            float kmin = set ? 10 : 2, kmax = set ? 48 : 14;
            float k = kmin * powf(kmax / kmin, hash1(i, 200 + set));
            float a = -0.6f + (hash1(i, 210 + set) - 0.5f) * 2.4f;     /* round the wind */
            kx[set][i] = roundf(k * cosf(a)); kz[set][i] = roundf(k * sinf(a));
            if (kx[set][i] == 0 && kz[set][i] == 0) kx[set][i] = 2;
            float kl = sqrtf(kx[set][i] * kx[set][i] + kz[set][i] * kz[set][i]);
            am[set][i] = 0.07f / kl * (0.6f + 0.8f * hash1(i, 220 + set));
            ph[set][i] = hash1(i, 230 + set) * 6.2832f;
        }
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            for (int set = 0; set < 2; set++) {
                float sx = 0, sz = 0;
                for (int i = 0; i < M; i++) {
                    float arg = 6.2832f * (kx[set][i] * x + kz[set][i] * y) / N + ph[set][i];
                    float d = -am[set][i] * 6.2832f * sinf(arg);
                    sx += d * kx[set][i]; sz += d * kz[set][i];
                }
                px[((size_t)y * N + x) * 4 + set * 2] = sx;
                px[((size_t)y * N + x) * 4 + set * 2 + 1] = sz;
            }
    /* each set to an rms slope of 0.1 */
    for (int set = 0; set < 2; set++) {
        double ss = 0;
        for (int i = 0; i < N * N; i++) ss += px[i * 4 + set * 2] * px[i * 4 + set * 2] + px[i * 4 + set * 2 + 1] * px[i * 4 + set * 2 + 1];
        float k = 0.1f / (float)sqrt(ss / (N * N));
        for (int i = 0; i < N * N; i++) { px[i * 4 + set * 2] *= k; px[i * 4 + set * 2 + 1] *= k; }
    }
    tex_wave = mk_tex(GL_RGBA16F, N, N, GL_RGBA, GL_FLOAT, px, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
    free(px);
}

static void build_noise(void) {
    enum { N = 256 };
    unsigned char *px = malloc(N * N * 4);
    static const int per[4] = {8, 16, 32, 64};
    for (int c = 0; c < 4; c++) {
        int P = per[c];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                float s = 0, a = 1, norm = 0;
                for (int o = 0; o < 3; o++) {
                    int p = P << o;
                    float fx = (float)x * p / N, fy = (float)y * p / N;
                    int ix = (int)fx, iy = (int)fy;
                    float tx = fx - ix, ty = fy - iy;
                    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
                    float v00 = hash2(ix % p, iy % p, 70 + c * 5 + o), v10 = hash2((ix + 1) % p, iy % p, 70 + c * 5 + o);
                    float v01 = hash2(ix % p, (iy + 1) % p, 70 + c * 5 + o), v11 = hash2((ix + 1) % p, (iy + 1) % p, 70 + c * 5 + o);
                    s += a * mixf(mixf(v00, v10, tx), mixf(v01, v11, tx), ty);
                    norm += a; a *= 0.45f;
                }
                s /= norm;
                s = clampf((s - 0.5f) * 1.6f + 0.5f, 0, 1);
                px[(y * N + x) * 4 + c] = (unsigned char)(s * 255 + 0.5f);
            }
    }
    tex_noise = mk_tex(GL_RGBA8, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
    free(px);
}

/* the drums' numerals: ten cells of a serif face, white on black */
#define DG_W 96
#define DG_H 128
static void build_digits(void) {
    unsigned char *px = calloc(DG_W * 10 * DG_H, 1);
    FT_Library ft;
    FT_Face face = NULL;
    if (FcInit() && !FT_Init_FreeType(&ft)) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:weight=bold:fontformat=CFF,TrueType", opt_font);
        FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *mt = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        if (mt && FcPatternGetString(mt, FC_FILE, 0, &file) == FcResultMatch) FT_New_Face(ft, (const char *)file, 0, &face);
        if (mt) FcPatternDestroy(mt);
        FcPatternDestroy(pat);
    }
    if (face) {
        FT_Set_Pixel_Sizes(face, 0, 104);
        for (int d = 0; d < 10; d++) {
            if (FT_Load_Char(face, (FT_ULong)('0' + d), FT_LOAD_RENDER)) continue;
            FT_GlyphSlot g = face->glyph;
            int w = (int)g->bitmap.width, h = (int)g->bitmap.rows;
            int ox = d * DG_W + (DG_W - w) / 2, oy = (DG_H - h) / 2;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int X = ox + x, Y = oy + y;
                    if (X >= d * DG_W && X < (d + 1) * DG_W && Y >= 0 && Y < DG_H)
                        px[Y * DG_W * 10 + X] = g->bitmap.buffer[y * g->bitmap.pitch + x];
                }
        }
    }
    tex_digits = mk_tex(GL_R8, DG_W * 10, DG_H, GL_RED, GL_UNSIGNED_BYTE, px, GL_LINEAR_MIPMAP_LINEAR, GL_CLAMP_TO_EDGE);
    free(px);
}

/* ======================================================================
 * Shaders
 * ====================================================================== */
#define GLSL(...) #__VA_ARGS__ "\n"

static const char *COMMON = GLSL(
uniform vec3 uCam;
uniform vec3 uSunDir;              /* the real Sun (may be below the horizon) */
uniform vec3 uMoonDir;
uniform vec3 uKeyDir;              /* the light that casts shadows: the Sun, else the Moon */
uniform vec3 uKeyCol;              /* its colour and strength at the ground, clear sky */
uniform vec3 uZen, uHor, uGlow;    /* sky: zenith, horizon, glow round the Sun */
uniform vec3 uAmbSky, uAmbGnd;
uniform vec3 uFogCol;
uniform float uFogDen, uFogH, uMist;
uniform float uTime, uTide, uWet, uCloud, uNight, uMoonIllum, uWaveAmp;
uniform vec2 uWind;
uniform sampler2D uNoise;
uniform sampler2DShadow uShadow;
uniform mat4 uShadowVP;
uniform vec4 uLamp[16];            /* xyz, strength */
uniform int uNLamp;
uniform vec4 uPulse;               /* the bell's last blows: seconds since each (< 0: none) */
uniform float uLowQ;               /* the reflection pass: cheaper */

const float PI = 3.14159265;
const float PLAT = 4.6;
uint hashu(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
float hash1i(int a, uint salt) { return float(hashu(uint(a) * 0x9E3779B9u + salt * 0x85EBCA6Bu) >> 8) * (1.0 / 16777216.0); }
float hash2i(int a, int b, uint salt) { return float(hashu(uint(a) * 0x9E3779B9u ^ hashu(uint(b) + salt * 0x85EBCA6Bu)) >> 8) * (1.0 / 16777216.0); }
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
vec4 nz(vec2 p) { return texture(uNoise, p * 0.125); }
float fbm2(vec2 p) { return nz(p).r * 0.5 + nz(p * 2.31 + 0.37).g * 0.3 + nz(p * 5.17 + 0.71).b * 0.2; }

/* clouds: one layer high up, drifting */
float cloudAt(vec2 xz) {
    vec2 q = xz / 900.0 + uWind * uTime * 0.0002;
    float n = nz(q).r * 0.55 + nz(q * 2.7 + 0.3).g * 0.28 + nz(q * 7.3 + 0.6).b * 0.17;
    return smoothstep(1.02 - uCloud * 0.95, 1.18 - uCloud * 0.75, n + 0.12);
}
float cloudShadow(vec3 p) {
    vec2 q = p.xz + uKeyDir.xz / max(uKeyDir.y, 0.06) * (1600.0 - p.y);
    return 1.0 - 0.88 * cloudAt(q);
}
/* the same from the first octave only: for the sea and the air */
float cloudShadowFast(vec3 p) {
    vec2 q = (p.xz + uKeyDir.xz / max(uKeyDir.y, 0.06) * (1600.0 - p.y)) / 900.0 + uWind * uTime * 0.0002;
    float n = nz(q).r * 0.55 + 0.28 * 0.5 + 0.17 * 0.5;
    return 1.0 - 0.88 * smoothstep(1.02 - uCloud * 0.95, 1.18 - uCloud * 0.75, n + 0.12);
}
float shadowAt(vec3 p, vec3 n) {
    vec4 s = uShadowVP * vec4(p + n * 0.04, 1.0);
    vec3 q = s.xyz * 0.5 + 0.5;
    if (any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0))) || q.z > 1.0) return 1.0;
    float b = 0.0006;
    if (uLowQ > 0.5) return texture(uShadow, vec3(q.xy, q.z - b));
    vec2 o = vec2(1.0 / 2048.0);
    return 0.25 * (texture(uShadow, vec3(q.xy + o * vec2(-0.6, -0.4), q.z - b)) + texture(uShadow, vec3(q.xy + o * vec2(0.4, -0.6), q.z - b)) +
                   texture(uShadow, vec3(q.xy + o * vec2(0.6, 0.4), q.z - b)) + texture(uShadow, vec3(q.xy + o * vec2(-0.4, 0.6), q.z - b)));
}

float shadowFast(vec3 p) {
    vec3 q = (uShadowVP * vec4(p, 1.0)).xyz * 0.5 + 0.5;
    if (any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0))) || q.z > 1.0) return 1.0;
    return texture(uShadow, vec3(q.xy, q.z - 0.0008));
}

/* the sky without clouds or bodies */
vec3 skyBase(vec3 rd) {
    float y = max(rd.y, 0.0);
    /* the warm horizon gathers on the Sun's side when it's low */
    float low = 1.0 - smoothstep(0.05, 0.45, uSunDir.y);
    float side = 0.5 + 0.5 * dot(normalize(rd.xz + vec2(1e-5)), normalize(uSunDir.xz + vec2(1e-5)));
    vec3 hor = mix(uHor, mix(uHor * vec3(0.62, 0.62, 0.8) + uZen * 0.4, uHor, pow(side, 2.0)), low);
    vec3 c = mix(hor, uZen, 1.0 - exp(-y * mix(5.0, 9.0, low)));
    float mu = max(dot(rd, uSunDir), 0.0);
    c += uGlow * (pow(mu, 5.0) * 0.35 + pow(mu, 48.0) * 0.9) * (1.0 - uCloud * 0.5);
    /* a band of haze on the horizon */
    c = mix(c, uFogCol, exp(-y * 14.0) * 0.6);
    if (rd.y < 0.0) c = mix(c, uFogCol * 0.9, min(-rd.y * 8.0, 1.0));
    return c;
}

/* height fog: transmittance along a ray from the camera of length t */
float fogTrans(vec3 rd, float t) {
    float y0 = max(uCam.y, 0.0);
    float k = rd.y * t;
    float a = uFogDen * exp(-y0 / uFogH), m = uMist * exp(-y0 / 5.0);
    float fa = abs(k) > 1e-3 ? uFogH * (1.0 - exp(min(-k / uFogH, 40.0))) / rd.y : t;
    float fm = abs(k) > 1e-3 ? 5.0 * (1.0 - exp(min(-k / 5.0, 40.0))) / rd.y : t;
    return exp(-(a * fa + m * fm));
}
vec3 fogColor(vec3 rd) {
    float mu = max(dot(rd, uSunDir), 0.0);
    return uFogCol + uGlow * pow(mu, 6.0) * 0.25;
}
vec3 applyFog(vec3 c, vec3 p) {
    vec3 d = p - uCam;
    float t = length(d);
    vec3 rd = d / t;
    float T = fogTrans(rd, t);
    return c * T + fogColor(rd) * (1.0 - T);
}

/* how much sky a point inside the tower sees */
float interiorOcc(vec3 p) {
    if (abs(p.x) > 3.02 || abs(p.z) > 3.02 || p.y < 5.0 || p.y > 33.2) {
        /* the side tower's niche */
        return 1.0;
    }
    if (p.y > 27.0) return 0.55;
    if (p.y > 21.0) return 0.12;
    if (p.y < 8.6) return 0.15;
    /* the gear room opens to the east */
    float e = smoothstep(-3.0, 3.2, p.x);
    float v = 1.0 - smoothstep(4.0, 7.0, abs(p.y - 14.7));
    return mix(0.16, 0.6, e * v);
}

float ggx(float NH, float a) { float a2 = a * a, d = NH * NH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }

/* the lanterns: warm points with a soft falloff; the lamp inside lights
 * only the inside */
vec3 lamps(vec3 p, vec3 n, vec3 v, vec3 alb, float rough, vec3 f0) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < uNLamp; i++) {
        /* the gear room's lamp burns day and night; the lanterns only at night */
        float on = i < 2 ? max(uNight, 0.5) : uNight;
        if (on < 0.01) continue;
        vec3 L = uLamp[i].xyz - p;
        float d2 = dot(L, L);
        if (d2 > 900.0) continue;
        bool inside = abs(p.x) < 3.05 && abs(p.z) < 3.05 && p.y > 8.0 && p.y < 21.5;
        if (i < 2 && !inside) continue;
        if (i >= 2 && inside) continue;
        L *= inversesqrt(d2);
        float NL = max(dot(n, L), 0.0);
        float att = uLamp[i].w / (d2 + 0.4) * smoothstep(900.0, 300.0, d2);
        vec3 h = normalize(L + v);
        vec3 spec = f0 * ggx(max(dot(n, h), 0.0), max(rough, 0.08)) * 0.25;
        sum += vec3(1.0, 0.62, 0.3) * att * NL * (alb + spec) * on;
    }
    return sum;
}

/* a surface lit by the key light, the sky, and the lamps */
vec3 shade(vec3 p, vec3 n, vec3 v, vec3 alb, float rough, float metal, float ao, float spec) {
    float occ = interiorOcc(p);
    vec3 f0 = mix(vec3(0.03) * spec, alb, metal);
    vec3 dif = alb * (1.0 - metal);
    float NL = dot(n, uKeyDir);
    vec3 col = vec3(0.0);
    if (NL > 0.0) {
        float sh = (dot(uCam - p, uCam - p) < 3600.0 ? shadowAt(p, n) : shadowFast(p + n * 0.05)) * cloudShadow(p);
        vec3 h = normalize(uKeyDir + v);
        float NH = max(dot(n, h), 0.0), VH = max(dot(v, h), 0.0);
        vec3 F = f0 + (1.0 - f0) * pow(1.0 - VH, 5.0);
        vec3 sp = F * ggx(NH, max(rough * rough, 0.004)) * 0.25 / max(dot(n, v), 0.25);
        col += uKeyCol * sh * NL * (dif / PI * 3.0 + sp);
    }
    /* sky and bounce, and the sky in the sheen */
    vec3 amb = mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5);
    if (occ < 0.99) occ *= 0.55 + 0.6 * max(n.x, 0.0) + 0.1 * n.y;
    col += dif * amb * ao * occ;
    /* inside, light bounced off the sunlit floor and stone, warm */
    if (occ < 0.99) col += dif * ao * (uKeyCol * max(uKeyDir.y, 0.15) * 0.11 * cloudShadow(p) + uAmbSky * 0.3) * vec3(1.0, 0.8, 0.6) * (1.0 - occ);
    vec3 R = reflect(-v, n);
    float fr = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    vec3 Fe = f0 + (max(vec3(1.0 - rough), f0) - f0) * fr;
    vec3 env = R.y > 0.0 ? skyBase(R) : mix(uAmbGnd, uFogCol, 0.5);
    /* indoors, what shines back is warm stone and the lamp */
    vec3 envIn = (uAmbSky * 0.35 + uKeyCol * max(uKeyDir.y, 0.15) * 0.1) * vec3(1.0, 0.78, 0.55) + vec3(0.25, 0.14, 0.05) * max(uNight, 0.45) * 0.4;
    env = mix(envIn, env, smoothstep(0.1, 0.9, occ));
    col += env * Fe * ao * mix(1.0, occ, 0.5) * (1.0 - rough * 0.7);
    col += lamps(p, n, v, dif + 0.0, rough, f0) * ao;
    return col;
}

/* bump from a scalar field: Mikkelsen's surface gradient */
\n#ifndef VERTEX\n
vec3 bump(vec3 p, vec3 n, float h) {
    vec3 dpdx = dFdx(p), dpdy = dFdy(p);
    float dhx = dFdx(h), dhy = dFdy(h);
    vec3 r1 = cross(dpdy, n), r2 = cross(n, dpdx);
    float det = dot(dpdx, r1);
    vec3 g = sign(det) * (dhx * r1 + dhy * r2);
    vec3 b = abs(det) * n - g;
    float l = length(b);
    return l > 1e-12 ? b / l : n;
}
\n#endif\n
);

/* ---- meshes: tower, causeway, movement, bell, drums ---- */
static const char *VS_MESH = GLSL(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec2 aMat;
uniform mat4 uModel, uVP, uMirror;
uniform float uClipY;
out vec3 vP; out vec3 vN; out vec2 vUV; flat out int vMat; out float vAux;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vP = w.xyz;
    vN = mat3(uModel) * aNrm;
    vUV = aUV; vMat = int(aMat.x + 0.5); vAux = aMat.y;
    gl_ClipDistance[0] = w.y - uClipY;
    gl_Position = uVP * (uMirror * w);
}
);

static const char *FS_MESH = GLSL(
in vec3 vP; in vec3 vN; in vec2 vUV; flat in int vMat; in float vAux;
uniform vec4 uHands[4];            /* per dial: hour, minute angles (rad, clockwise from 12) */
uniform sampler2D uDigits;
out vec4 oCol;

/* ashlar coursing: rows half a metre high, block lengths per row */
vec4 blocks(vec2 uv, float rowh) {
    float r = floor(uv.y / rowh);
    int ri = int(r);
    float L = 0.75 + 0.55 * hash1i(ri, 11u), off = hash1i(ri, 12u) * 3.0;
    float u = uv.x + off, k = floor(u / L);
    float fu = u - k * L, fv = uv.y - r * rowh;
    return vec4(min(fu, L - fu), min(fv, rowh - fv), hash2i(ri, int(k), 13u), hash2i(ri, int(k), 14u));
}

void stone(vec3 p, inout vec3 n, vec3 gn, vec2 uv, float aux, float rubble, out vec3 alb, out float rough, out float ao) {
    /* rubble: the wall's core, smaller rougher stones */
    vec2 u2 = rubble > 0.5 ? uv * 2.0 + vec2(0.37 * floor(uv.y * 4.0), 0.0) : uv;
    vec4 b = blocks(u2, 0.5);
    if (rubble > 0.5) b.xy *= 0.5;
    float e = min(b.x, b.y);
    float joint = 1.0 - smoothstep(0.004, 0.016 + rubble * 0.02, e);
    /* weathering across the whole wall, not per block */
    vec2 wp = vec2(p.x + p.z, p.y);
    float W = fbm2(wp * 0.11 + 3.0);
    float grain = fbm2(uv * 1.3 + b.z * 7.0);
    float h = smoothstep(0.0, 0.04, e) * 0.01 + (grain - 0.5) * 0.004;
    if (uLowQ < 0.5 && dot(uCam - p, uCam - p) < 8100.0) n = bump(p, n, h);
    vec3 warm = vec3(0.34, 0.315, 0.28), cool = vec3(0.29, 0.295, 0.29);
    vec3 c = mix(warm, cool, b.z) * (0.9 + 0.2 * b.w);
    c *= 0.86 + 0.28 * grain;
    c *= 0.7 + 0.45 * W;
    if (rubble > 0.5) c *= vec3(0.82, 0.8, 0.78);
    /* grime run down from the ledges, long and vertical */
    float streak = nz(vec2(wp.x * 1.4, wp.y * 0.05)).g * 0.6 + nz(vec2(wp.x * 4.1, wp.y * 0.12)).b * 0.4;
    c *= 1.0 - 0.38 * smoothstep(0.42, 0.8, streak) * (1.0 - abs(gn.y));
    /* lichens: orange spots and grey-green crust, on what faces the weather */
    float lich = smoothstep(0.74, 0.8, nz(wp * 2.6 + 3.7).a * 0.7 + nz(wp * 0.4).b * 0.4) * (0.4 + 0.6 * smoothstep(0.3, 1.0, gn.y + 0.5));
    c = mix(c, vec3(0.50, 0.35, 0.14), lich * 0.55 * (1.0 - rubble));
    float lich2 = smoothstep(0.64, 0.72, nz(wp * 0.23 + 9.1).r + grain * 0.1);
    c = mix(c, vec3(0.36, 0.38, 0.33), lich2 * 0.3);
    /* damp low down: green-black algae, salt, the high-water mark */
    float damp = 1.0 - smoothstep(PLAT + 0.2, PLAT + 2.2 + W * 1.5, p.y);
    c = mix(c, c * vec3(0.55, 0.6, 0.5), damp * 0.8);
    float hw = 2.6;
    float sea = 1.0 - smoothstep(hw - 0.6, hw + 0.8 + grain, p.y);
    c = mix(c, c * vec3(0.38, 0.40, 0.33), sea);
    c = mix(c, vec3(0.05, 0.065, 0.03), (1.0 - smoothstep(uTide - 0.8, uTide + 1.1, p.y)) * smoothstep(0.4, 0.6, grain + 0.2) * sea);
    /* the joints: lime mortar, dirty, a little recessed */
    vec3 mortar = vec3(0.36, 0.345, 0.31) * (0.7 + 0.4 * W);
    c = mix(c, c * 0.6, (1.0 - smoothstep(0.0, 0.03, e)) * 0.5);
    c = mix(c, mortar, joint * 0.75);
    alb = c;
    rough = 0.85 - 0.15 * grain;
    ao = 1.0 - joint * 0.5;
}

void main() {
    float ln = length(vN);
    vec3 n = ln > 1e-6 ? vN / ln : vec3(0.0, 1.0, 0.0);
    vec3 v = normalize(uCam - vP);
    if (dot(n, v) < 0.0) n = -n;
    vec3 gn = n;
    vec3 alb = vec3(0.5);
    float rough = 0.6, metal = 0.0, ao = 1.0, spec = 1.0;
    vec3 emis = vec3(0.0);
    float wet = uWet;
    vec2 uv = vUV;
    int m = vMat;
    if (m == 0 || m == 1) {
        stone(vP, n, gn, uv, vAux, float(m), alb, rough, ao);
        wet = max(wet, 1.0 - smoothstep(uTide + 0.2, uTide + 1.4, vP.y));
    } else if (m == 10) {
        /* the causeway's blocks: worn, weedy, wet after the tide */
        stone(vP, n, gn, uv * vec2(1.0, 1.25) + vec2(0.0, 0.11), vAux, 0.0, alb, rough, ao);
        float g = fbm2(vP.xz * 0.4);
        float tidal = 1.0 - smoothstep(0.4, 2.6, vP.y);
        alb = mix(alb, vec3(0.10, 0.12, 0.05), tidal * smoothstep(0.35, 0.7, g + 0.25 * (1.0 - gn.y)) * 0.85);
        alb = mix(alb, vec3(0.20, 0.26, 0.10), gn.y * tidal * smoothstep(0.55, 0.8, nz(vP.xz * 0.21).b) * 0.6);
        float barn = step(0.8, hash12(floor(vP.xz * 22.0) + floor(vP.y * 22.0))) * (1.0 - gn.y) * tidal;
        alb = mix(alb, vec3(0.55, 0.53, 0.48), barn * 0.6);
        wet = max(wet, 1.0 - smoothstep(uTide + 0.15, uTide + 0.9, vP.y));
        wet = max(wet, 0.35 * tidal);
    } else if (m == 2 || m == 15) {
        float r = fbm2(uv * 1.7 + vP.xz * 0.3 + vP.y * 0.4);
        alb = m == 2 ? vec3(0.075, 0.08, 0.075) : vec3(0.32, 0.32, 0.33);
        alb = mix(alb, vec3(0.24, 0.10, 0.04), smoothstep(0.45, 0.8, r) * (m == 2 ? 0.8 : 0.4));
        rough = m == 2 ? 0.55 : 0.32; metal = m == 2 ? 0.4 : 0.9;
    } else if (m == 3) {
        /* brass, tarnished, rubbed bright on the edges that work */
        float t = fbm2(uv * 1.3 + vP.yz * 0.7 + vAux * 5.0);
        alb = mix(vec3(0.78, 0.58, 0.26), vec3(0.42, 0.30, 0.14), smoothstep(0.35, 0.75, t) * 0.8);
        float verd = smoothstep(0.72, 0.8, nz(vP.yz * 0.9 + vAux).b);
        rough = 0.28 + 0.25 * t;
        metal = 1.0 - verd;
        alb = mix(alb, vec3(0.22, 0.42, 0.33), verd);
    } else if (m == 4) {
        /* bronze with verdigris running down */
        float s = nz(vec2(atan(vP.z, vP.x - 0.0) * 1.3, vP.y * 0.12)).g;
        float t = fbm2(vP.xz * 2.0 + vP.y);
        alb = mix(vec3(0.46, 0.30, 0.15), vec3(0.26, 0.47, 0.39), smoothstep(0.4, 0.75, s + 0.2 * t));
        metal = 1.0 - smoothstep(0.45, 0.7, s + 0.2 * t);
        rough = 0.35 + 0.25 * t;
    } else if (m == 5 || m == 14) {
        float g = nz(vec2(uv.x * 0.6 + vAux, uv.y * 7.0 + vAux * 3.0)).g;
        float plank = m == 14 ? smoothstep(0.0, 0.02, abs(fract(vP.x * 4.5) - 0.5) - 0.46) : 0.0;
        alb = mix(vec3(0.21, 0.14, 0.09), vec3(0.13, 0.09, 0.06), g) * (m == 14 ? 0.8 : 1.0);
        alb *= 1.0 - plank * 0.7;
        rough = 0.75;
    } else if (m == 6) {
        /* copper gone green, with standing seams and dark runs */
        float seam = 1.0 - smoothstep(0.0, 0.035, abs(fract(uv.x * 1.6) - 0.5) * 0.625 - 0.0);
        float run = nz(vec2(uv.x * 3.0, uv.y * 0.08)).g;
        float t = fbm2(uv * 0.8);
        alb = mix(vec3(0.30, 0.52, 0.44), vec3(0.18, 0.30, 0.26), smoothstep(0.4, 0.8, run) * 0.7);
        alb *= 0.85 + 0.3 * t;
        if (uLowQ < 0.5) n = bump(vP, n, seam * 0.02);
        rough = 0.55;
    } else if (m == 7 || m == 13 || m == 12) {
        /* lantern glass, the gear room lamp, the window: lit at night */
        int k = int(vAux + 0.5);
        float on = uNight;
        float flick = 1.0;
        if (m == 7 && k == 4) on = 0.0;                               /* one has gone out */
        if (m == 13) on = max(uNight, 0.5);
        if (m == 7 && k == 6) flick = 0.55 + 0.45 * smoothstep(0.2, 0.6, nz(vec2(uTime * 0.7, 0.3)).r);
        flick *= 0.92 + 0.08 * nz(vec2(uTime * 2.3 + vAux, 0.7)).g;
        alb = vec3(0.05, 0.05, 0.05);
        rough = 0.1; spec = 1.0;
        float frame = 0.0;
        if (m == 12) frame = step(0.92, max(fract(vP.x * 2.6), fract(vP.y * 2.2)));
        vec3 warm = m == 12 ? vec3(1.0, 0.55, 0.22) * 2.2 : vec3(1.0, 0.63, 0.3) * 9.0;
        emis = warm * on * flick * (1.0 - frame);
    } else if (m == 8) {
        /* a dial: opal glass in an iron chapter ring, the hands over it */
        int f = int(vAux + 0.5);
        vec2 q = uv;
        float r = length(q);
        float a = atan(q.x, q.y);                                   /* clockwise from 12 */
        float ink = 0.0;
        ink = max(ink, 1.0 - smoothstep(0.012, 0.022, abs(r - 0.70)));
        ink = max(ink, 1.0 - smoothstep(0.012, 0.022, abs(r - 0.93)));
        float ma = fract(a / (2.0 * PI) * 60.0 + 0.5) - 0.5;
        ink = max(ink, step(abs(ma * 2.0 * PI / 60.0 * r), 0.006) * step(0.86, r) * step(r, 0.93));
        float ha = fract(a / (2.0 * PI) * 12.0 + 0.5) - 0.5;
        float hb = abs(ha * 2.0 * PI / 12.0 * r);
        ink = max(ink, step(hb, 0.035) * step(0.74, r) * step(r, 0.9));
        /* a small mark in place of each hour: a dot inside the ring */
        ink = max(ink, 1.0 - smoothstep(0.022, 0.03, length(q - normalize(q) * 0.62) + step(0.02, hb) ));
        /* hands */
        vec2 hh = vec2(sin(uHands[f].x), cos(uHands[f].x)), mh = vec2(sin(uHands[f].y), cos(uHands[f].y));
        float th = dot(q, hh), tm = dot(q, mh);
        float dh = abs(dot(q, vec2(hh.y, -hh.x))), dm = abs(dot(q, vec2(mh.y, -mh.x)));
        float hand = step(-0.12, th) * step(th, 0.52) * step(dh, 0.045 * (1.0 - 0.6 * clamp(th / 0.52, 0.0, 1.0)));
        hand = max(hand, step(-0.15, tm) * step(tm, 0.84) * step(dm, 0.028 * (1.0 - 0.6 * clamp(tm / 0.84, 0.0, 1.0))));
        hand = max(hand, 1.0 - step(0.06, r));
        /* the hands' shadow on the glass, by day */
        vec2 so = vec2(0.025, -0.03);
        vec2 qs = q - so;
        float ths = dot(qs, hh), tms = dot(qs, mh);
        float sh = step(-0.12, ths) * step(ths, 0.52) * step(abs(dot(qs, vec2(hh.y, -hh.x))), 0.045);
        sh = max(sh, step(-0.15, tms) * step(tms, 0.84) * step(abs(dot(qs, vec2(mh.y, -mh.x))), 0.028));
        float grime = smoothstep(0.55, 1.0, r) * 0.5 + smoothstep(0.5, 0.8, nz(q * 0.7 + float(f)).g) * 0.3;
        alb = mix(vec3(0.78, 0.74, 0.62), vec3(0.45, 0.42, 0.34), grime);
        alb *= 1.0 - sh * 0.35 * (1.0 - uNight);
        alb = mix(alb, vec3(0.03, 0.03, 0.03), max(ink, hand));
        rough = mix(0.25, 0.5, max(ink, hand));
        metal = 0.0;
        emis = vec3(1.0, 0.82, 0.55) * 0.9 * uNight * (1.0 - max(ink, hand)) * (1.0 - grime * 0.6) * (1.0 - sh * 0.5);
    } else if (m == 9) {
        /* a drum face: an ivory enamel numeral plate, the numeral in black */
        int d = int(vAux + 0.5);
        vec2 q = vec2(uv.x, 1.0 - uv.y);
        vec2 g = vec2((float(d) + 0.5 + (q.x - 0.5) * 1.15) / 10.0, 0.5 + (q.y - 0.5) * 0.92);
        float ink = texture(uDigits, g).r;
        float edge = smoothstep(0.0, 0.05, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
        float dirt = fbm2(uv * 1.3 + vAux * 3.0 + vP.xy);
        alb = mix(vec3(0.80, 0.75, 0.62), vec3(0.55, 0.50, 0.40), smoothstep(0.5, 0.9, dirt));
        alb = mix(alb, vec3(0.025, 0.022, 0.02), ink);
        alb *= 0.7 + 0.3 * edge;
        rough = 0.35;
        emis = alb * vec3(1.0, 0.75, 0.45) * 0.35 * uNight;          /* a small lamp above the niche */
    } else if (m == 11) {
        alb = vec3(0.22, 0.17, 0.11) * (0.8 + 0.4 * nz(vec2(vP.y * 9.0, vP.x)).r);
        rough = 0.85;
    }
    if (wet > 0.0 && m != 7 && m != 8 && m != 12 && m != 13) {
        alb *= 1.0 - 0.45 * wet;
        rough = mix(rough, 0.12, wet * 0.85);
        spec = mix(spec, 1.6, wet);
    }
    vec3 col = shade(vP, n, v, alb, rough, metal, ao, spec) + emis;
    /* the niche's own small lamp */
    if (uNight > 0.0) {
        vec3 nl = vec3(-5.3, 8.75, 5.05) - vP;
        float d2 = dot(nl, nl);
        if (d2 < 9.0) col += alb * vec3(1.0, 0.7, 0.4) * uNight * 0.9 / (d2 + 0.2) * max(dot(n, normalize(nl)), 0.0) * (1.0 - smoothstep(4.0, 9.0, d2));
    }
    oCol = vec4(applyFog(col, vP), 1.0);
}
);

/* ---- the ground ---- */
static const char *FS_TERRAIN = GLSL(
in vec3 vP; in vec3 vN; in vec2 vUV; flat in int vMat; in float vAux;
out vec4 oCol;
void main() {
    vec3 p = vP;
    vec3 n = normalize(vN);
    vec3 v = normalize(uCam - p);
    float dist = length(uCam - p);
    /* deep under the water nothing much shows through: keep it cheap */
    if (p.y < uTide - 2.6) {
        vec3 bed = vec3(0.10, 0.11, 0.08) * (uAmbSky * 0.8 + uKeyCol * max(uKeyDir.y, 0.0) * 0.3);
        oCol = vec4(applyFog(bed, p), 1.0);
        return;
    }
    float slope = 1.0 - n.y;
    /* texture coordinates that don't smear down the steep faces */
    vec2 w = n.y > 0.55 ? p.xz : (abs(n.x) > abs(n.z) ? p.zy : p.xy);
    float g1 = fbm2(w * 0.11), g2 = nz(w * 0.53).a, g3 = nz(w * 2.1).b;
    float fineK = 1.0 - smoothstep(30.0, 140.0, dist);
    /* rock: craggy, cracked */
    float cr = 1.0 - smoothstep(0.0, 0.05, abs(nz(w * 0.21 + p.y * 0.05).g - 0.5));
    float h = g2 * 0.15 + g3 * 0.05 * fineK - cr * 0.06 * fineK + g1 * 0.25;
    if (uLowQ < 0.5) n = bump(p, n, h * 1.2);
    vec3 rock = mix(vec3(0.11, 0.105, 0.10), vec3(0.21, 0.195, 0.175), g1) * (0.8 + 0.4 * g2);
    rock *= 1.0 - cr * 0.45;
    /* orange and grey lichens above the splash zone, a black band below */
    float above = smoothstep(2.9, 3.8, p.y);
    float lich = smoothstep(0.7, 0.78, nz(w * 0.37 + 4.0).b + g3 * 0.12) * above * smoothstep(0.2, 0.8, n.y + 0.3);
    rock = mix(rock, vec3(0.47, 0.32, 0.11), lich * 0.7);
    rock = mix(rock, vec3(0.30, 0.30, 0.28), smoothstep(0.64, 0.72, nz(w * 0.29 + 8.0).r + g3 * 0.1) * above * 0.4);
    rock = mix(rock, vec3(0.05, 0.05, 0.045), smoothstep(2.0, 2.6, p.y) * (1.0 - smoothstep(2.9, 3.6, p.y)) * smoothstep(0.3, 0.6, g2) * 0.8);
    /* the intertidal: barnacles, then wrack */
    float tid = 1.0 - smoothstep(1.8, 2.8, p.y + g1 * 0.6);
    float barn = step(0.62, hash12(floor(w * 18.0 + p.y * 7.0))) * smoothstep(-0.5, 0.5, p.y) * tid;
    rock = mix(rock, vec3(0.08, 0.075, 0.04), tid * 0.8);
    rock = mix(rock, vec3(0.5, 0.48, 0.42), barn * 0.5);
    float weed = (1.0 - smoothstep(-0.2, 1.2, p.y + g2 * 0.8)) * smoothstep(0.3, 0.6, g1 + g3 * 0.3);
    rock = mix(rock, vec3(0.09, 0.075, 0.025), weed);
    /* the shore: shingle and sand low down, turf and heather up top */
    float shore = smoothstep(80.0, 120.0, p.z + 40.0 * step(abs(p.x), 30.0) * 0.0);
    float level = smoothstep(0.25, 0.08, slope);
    float sandK = shore * level * (1.0 - smoothstep(1.8, 3.5, p.y)) * smoothstep(-3.0, -0.5, p.y);
    vec3 sand = mix(vec3(0.42, 0.39, 0.33), vec3(0.30, 0.28, 0.25), g3) * (0.85 + 0.3 * step(0.7, hash12(floor(w * 9.0))));
    float turfK = shore * smoothstep(0.35, 0.15, slope) * smoothstep(4.0, 7.0, p.y + g1 * 3.0);
    vec3 turf = mix(vec3(0.075, 0.095, 0.045), vec3(0.16, 0.15, 0.08), g1 * 0.7 + g2 * 0.3);
    turf = mix(turf, vec3(0.12, 0.07, 0.08), smoothstep(0.55, 0.7, nz(w * 0.05).r) * 0.85);
    vec3 alb = rock;
    alb = mix(alb, sand, sandK);
    alb = mix(alb, turf, turfK);
    /* the sea bed */
    alb = mix(alb, vec3(0.16, 0.17, 0.12) * (0.7 + 0.5 * g2), 1.0 - smoothstep(-3.0, -1.0, p.y));
    float wet = max(uWet, 1.0 - smoothstep(uTide + 0.2, uTide + 0.9 + g1, p.y));
    float rough = mix(0.85, 0.15, wet);
    alb *= 1.0 - 0.45 * wet;
    vec3 col = shade(p, n, v, alb, rough, 0.0, 1.0 - cr * 0.3, 1.0 + wet);
    oCol = vec4(applyFog(col, p), 1.0);
}
);

/* ---- the sky: gradient, clouds, Sun, Moon, stars, far islands ---- */
static const char *VS_FULL = GLSL(
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
);

static const char *FS_SKY = GLSL(
in vec2 vUV;
uniform vec4 uViewTan;
uniform mat3 uCamRot;
uniform float uMirrorSky;
uniform float uSunR, uMoonR;
out vec4 oCol;

vec3 skyFull(vec3 rd) {
    vec3 c = skyBase(rd);
    float y = max(rd.y, 0.0);
    /* stars */
    if (uNight > 0.0 && rd.y > 0.0) {
        vec3 q = rd * 420.0;
        vec3 cell = floor(q);
        float hsh = hash13(cell);
        if (hsh > 0.982) {
            vec3 sp = cell + 0.2 + 0.6 * vec3(hash13(cell + 1.3), hash13(cell + 2.7), hash13(cell + 5.1));
            float d = length(q - sp);
            float px = max(length(fwidth(q)) * 0.7, 0.02);
            float tw = 0.75 + 0.25 * sin(uTime * (2.0 + hsh * 6.0) + hsh * 100.0);
            float mag = pow((hsh - 0.982) / 0.018, 7.0);
            c += vec3(0.85, 0.9, 1.0) * mix(0.012, 0.3, mag) * exp(-d * d / (px * px)) * tw * uNight * smoothstep(0.0, 0.15, rd.y);
        }
        /* a faint band of the galaxy */
        float mw = exp(-pow(dot(rd, normalize(vec3(0.3, 0.55, -0.78))) / 0.22, 2.0));
        c += vec3(0.05, 0.055, 0.07) * mw * fbm2(rd.xz * 3.0 + rd.y) * uNight;
    }
    /* the Moon, lit by the Sun: its phase falls out of the geometry */
    float md = dot(rd, uMoonDir);
    if (md > cos(uMoonR * 1.3) && uMoonDir.y > -0.05) {
        vec3 ax = normalize(cross(uMoonDir, vec3(0.0, 1.0, 0.0)));
        vec3 ay = cross(ax, uMoonDir);
        vec2 dq = vec2(dot(rd, ax), dot(rd, ay)) / sin(uMoonR);
        float r2 = dot(dq, dq);
        if (r2 < 1.0) {
            vec3 N = ax * dq.x + ay * dq.y - uMoonDir * sqrt(1.0 - r2);
            float lit = smoothstep(-0.02, 0.06, dot(N, uSunDir));
            float mare = 0.75 + 0.25 * smoothstep(0.35, 0.65, nz(dq * 0.22 + 0.3).r);
            vec3 mc = vec3(1.0, 0.97, 0.9) * (lit * mare * mix(5.0, 1.6, smoothstep(-0.1, 0.1, uSunDir.y)) + 0.015);
            float edge = smoothstep(1.0, 0.92, r2);
            c = mix(c, c * 0.3 + mc, edge * (1.0 - smoothstep(0.0, 0.25, uSunDir.y) * 0.6));
        }
        c += vec3(0.6, 0.65, 0.75) * pow(max(md, 0.0), 2000.0) * 0.08 * uMoonIllum * uNight;
    }
    c += vec3(0.6, 0.65, 0.75) * pow(max(md, 0.0), 300.0) * 0.12 * uMoonIllum * uNight;
    /* the Sun's disc */
    float sd = dot(rd, uSunDir);
    if (uSunDir.y > -0.02) c += uKeyCol * 18.0 * smoothstep(cos(uSunR), cos(uSunR * 0.8), sd);
    /* clouds on a layer 1600 m up */
    if (rd.y > 0.0 && uCloud > 0.01) {
        float t = (1600.0 - uCam.y) / rd.y;
        vec2 q = uCam.xz + rd.xz * t;
        float dens = cloudAt(q);
        if (dens > 0.001) {
            vec2 toS = normalize(uKeyDir.xz + 1e-4) * 240.0;
            float d2 = cloudAt(q + toS), d3 = cloudAt(q + toS * 2.2);
            float light = exp(-(d2 + d3) * 1.4);
            float mu = max(dot(rd, uKeyDir), 0.0);
            vec3 lit = uKeyCol * (0.35 + 0.65 * light) * 0.45 + uAmbSky * 0.9;
            lit += uKeyCol * pow(mu, 10.0) * (1.0 - dens) * 1.6;               /* silver linings */
            vec3 base = mix(uAmbSky * 0.55, uHor * 0.6, 0.3) + uKeyCol * 0.06;
            vec3 cc = mix(base, lit, light * 0.6 + 0.25);
            /* the sky shows through gaps; far clouds fade into the haze */
            float fade = exp(-t / 26000.0);
            c = mix(c, cc, dens * fade * smoothstep(0.0, 0.05, rd.y));
        }
    }
    /* far islands, faint in the haze on the northern horizon */
    float az = atan(rd.x, -rd.z);
    float isl = max(0.0, nz(vec2(az * 1.1, 0.5)).r - 0.55) * 0.05 + max(0.0, nz(vec2(az * 3.7, 0.9)).g - 0.6) * 0.03;
    isl *= smoothstep(1.9, 1.2, abs(az));
    if (rd.y > -0.002 && rd.y < isl) c = mix(c, mix(uHor, uFogCol * 0.65 + uAmbGnd * 0.4, 0.55), 0.8);
    return c;
}

void main() {
    vec3 rd = normalize(uCamRot * vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0));
    if (uMirrorSky > 0.5) rd.y = -rd.y;
    vec3 c = skyFull(rd);
    /* the low mist swallows the horizon */
    float T = fogTrans(rd, 2500.0);
    c = mix(fogColor(rd), c, mix(T, 1.0, smoothstep(0.0, 0.3, rd.y)));
    oCol = vec4(c, 1.0);
}
);

/* ---- the sea ---- */
static const char *VS_WATER = GLSL(
layout(location = 0) in vec2 aP;          /* polar: radius (m), angle */
uniform mat4 uVP;
uniform sampler2D uHC, uHF;
out vec3 vP; out float vDepth; out vec2 vSlope;
float solidAt(vec2 xz) {
    vec2 f = (xz - vec2(-48.0, -48.0)) / vec2(96.0, 224.0);
    if (all(greaterThan(f, vec2(0.002))) && all(lessThan(f, vec2(0.998)))) return textureLod(uHF, f, 0.0).r;
    return textureLod(uHC, (xz - vec2(-720.0, -640.0)) / 1440.0, 0.0).r;
}
/* Gerstner waves, from the open sea to the north-east */
const int NW = 5;
const vec4 WV[NW] = vec4[](vec4(0.42, 0.91, 21.0, 0.32), vec4(0.85, 0.53, 13.0, 0.22), vec4(-0.2, 0.98, 8.5, 0.14),
                           vec4(0.6, 0.8, 5.1, 0.08), vec4(-0.55, 0.83, 3.3, 0.05));
void main() {
    vec2 xz = uCam.xz + vec2(cos(aP.y), sin(aP.y)) * aP.x;
    float ground = solidAt(xz);
    float depth = uTide - ground;
    float damp = smoothstep(-0.2, 2.5, depth) * uWaveAmp;
    vec3 p = vec3(xz.x, uTide, xz.y);
    vec2 sl = vec2(0.0);
    for (int i = 0; i < NW; i++) {
        vec2 d = normalize(WV[i].xy);
        float k = 2.0 * PI / WV[i].z, c = sqrt(9.81 / k);
        float a = WV[i].w * damp * (1.0 - smoothstep(WV[i].z * 2.5, WV[i].z * 5.0, aP.x));
        float ph = k * (dot(d, xz) + c * uTime);
        p.xz -= d * (a * 0.7) * sin(ph);
        p.y += a * cos(ph);
        sl += d * a * k * sin(ph);
    }
    vP = p; vDepth = depth; vSlope = sl;
    gl_Position = uVP * vec4(p, 1.0);
}
);

static const char *FS_WATER = GLSL(
in vec3 vP; in float vDepth; in vec2 vSlope;
uniform sampler2D uRefl, uHC, uHF;
uniform vec2 uScreen;
uniform vec3 uBellPos;
out vec4 oCol;
float solidAt(vec2 xz) {
    vec2 f = (xz - vec2(-48.0, -48.0)) / vec2(96.0, 224.0);
    if (all(greaterThan(f, vec2(0.002))) && all(lessThan(f, vec2(0.998)))) return texture(uHF, f).r;
    return texture(uHC, (xz - vec2(-720.0, -640.0)) / 1440.0).r;
}
/* chop: short wind waves from many directions, faded out where a pixel
 * covers more than a wavelength; and a little noise to break the lattice */
uniform sampler2D uWave;
/* chop: three scales of the wave tile, each drifting its own way */
vec2 detail(vec2 w, float t, float foot) {
    vec2 a = texture(uWave, w / 23.0 + vec2(0.011, -0.006) * t).rg;
    vec2 b = texture(uWave, mat2(0.8, -0.6, 0.6, 0.8) * w / 9.0 + vec2(-0.017, -0.021) * t).ba;
    vec2 c = texture(uWave, mat2(0.6, 0.8, -0.8, 0.6) * w / 3.7 + vec2(0.035, -0.02) * t).ba;
    b = mat2(0.8, 0.6, -0.6, 0.8) * b;
    c = mat2(0.6, -0.8, 0.8, 0.6) * c;
    return a * 1.0 + b * 0.8 + c * 0.6;
}
void main() {
    vec3 p = vP;
    vec3 d = p - uCam;
    float dist = length(d);
    vec3 rd = d / dist;
    float ground = solidAt(p.xz);
    float depth = p.y - ground;
    if (depth < -0.05) discard;
    /* normal: the Gerstner slopes and finer chop, calmer far off */
    float far = smoothstep(60.0, 900.0, dist);
    float foot = length(fwidth(p.xz)) * 1.5;
    vec2 s = vSlope * 0.6 + detail(p.xz, uTime, foot) * (0.45 + uWaveAmp);
    /* rain on the water */
    if (uWet > 0.3) {
        vec2 cell = floor(p.xz * 2.0);
        for (int i = 0; i < 2; i++) {
            vec2 c2 = cell + vec2(float(i), 0.0);
            float h = hash12(c2);
            float ph = fract(uTime * 0.9 + h);
            vec2 cp = (c2 + vec2(hash12(c2 + 1.7), hash12(c2 + 3.1))) / 2.0;
            vec2 dv = p.xz - cp;
            float r = length(dv);
            float ring = sin((r - ph * 0.45) * 60.0) * exp(-r * 8.0) * (1.0 - ph);
            s += dv / max(r, 1e-3) * ring * 0.25 * (1.0 - far) * uWet;
        }
    }
    /* the bell's blows ripple out over the bay */
    for (int i = 0; i < 4; i++) {
        float tb = uPulse[i];
        if (tb < 0.0 || tb > 14.0) break;
        vec2 dv = p.xz - uBellPos.xz;
        float r = length(dv), R = tb * 22.0;
        float env = exp(-abs(r - R) * 0.35) * exp(-tb * 0.22) * 0.5;
        s += dv / max(r, 1e-3) * sin((r - R) * 1.6) * env * 0.25;
    }
    vec3 n = normalize(vec3(-s.x, 1.0, -s.y));
    if (dot(n, -rd) < 0.02) n = normalize(n + rd * (dot(n, -rd) - 0.02));
    float NV = max(dot(n, -rd), 0.0);
    float F = 0.02 + 0.98 * pow(1.0 - NV, 5.0);
    /* reflection: the mirrored scene, wobbled by the waves */
    vec2 suv = gl_FragCoord.xy / uScreen + n.xz * vec2(0.6, 0.6) * 0.05 / (1.0 + dist * 0.02);
    vec4 refl = texture(uRefl, clamp(suv, vec2(0.001), vec2(0.999)));
    /* the body of the water: green in the shallows, slate in the deep */
    float sh = shadowFast(p) * cloudShadowFast(p);
    vec3 lightIn = uAmbSky * 0.9 + uKeyCol * max(uKeyDir.y, 0.0) * sh * 0.45;
    vec3 deep = vec3(0.006, 0.02, 0.026), shallow = vec3(0.035, 0.085, 0.075);
    vec3 body = mix(shallow, deep, smoothstep(0.0, 5.0, depth)) * lightIn;
    /* light through the wave crests */
    body += vec3(0.05, 0.14, 0.11) * uKeyCol * sh * pow(max(dot(rd, uKeyDir) * 0.5 + 0.5, 0.0), 3.0) * max(s.x + s.y, 0.0) * 0.6;
    float aBody = smoothstep(0.0, 3.6, depth) * 0.85 + 0.15 * smoothstep(0.0, 0.4, depth);
    aBody = max(aBody, smoothstep(2.5, 4.5, depth));
    /* the glint of the Sun or Moon */
    vec3 h = normalize(uKeyDir - rd);
    float NH = max(dot(n, h), 0.0);
    float a = mix(0.03, 0.12, far) + uWaveAmp * 0.04;
    vec3 spec = uKeyCol * sh * ggx(NH, a * a) * F * 0.5 * step(0.0, uKeyDir.y);
    /* lamps glitter in it at night */
    if (uNight > 0.01) {
        for (int i = 2; i < uNLamp; i++) {
            vec3 L = uLamp[i].xyz - p;
            float d2 = dot(L, L);
            if (d2 > 2500.0) continue;
            vec3 hl = normalize(normalize(L) - rd);
            spec += vec3(1.0, 0.62, 0.3) * uLamp[i].w * ggx(max(dot(n, hl), 0.0), 0.012) * 0.02 / (1.0 + d2 * 0.02) * uNight;
        }
    }
    /* foam: on the rocks and the causeway, where it's shallow, as the swell runs in */
    float fn = 0.5, fn2 = 0.5;
    if (depth < 1.4 || uWaveAmp > 0.6) {
        fn = fbm2(p.xz * 0.35 + vec2(uTime * 0.03, 0.0));
        fn2 = nz(p.xz * 1.3 - vec2(0.0, uTime * 0.05)).a;
    }
    float surge = 0.5 + 0.5 * sin(uTime * 0.9 - depth * 3.0 + fn * 6.0 + p.x * 0.05);
    float shoreF = smoothstep(0.7, 0.0, depth + (fn - 0.5) * 0.5 - surge * 0.25);
    float lace = smoothstep(0.5, 0.8, fn2 * 0.6 + fn * 0.4 + surge * 0.35);
    float foam = shoreF * mix(lace, 1.0, smoothstep(0.12, 0.0, depth)) * 0.9;
    foam += smoothstep(0.75, 0.95, (vSlope.x + vSlope.y) * 0.9 + fn2 * 0.4) * uWaveAmp * 0.35 * (1.0 - far);
    foam = clamp(foam * (1.0 - far * 0.8), 0.0, 1.0);
    vec3 foamCol = (uKeyCol * max(uKeyDir.y, 0.0) * sh + uAmbSky) * 0.85;
    vec3 col = refl.rgb * F + body * (1.0 - F) * aBody + spec;
    float alpha = clamp(F + (1.0 - F) * aBody, 0.0, 1.0);
    col = mix(col, foamCol, foam * 0.85);
    alpha = mix(alpha, 1.0, foam * 0.85);
    /* fog: the scene behind was fogged to its own distance already */
    float T = fogTrans(rd, dist);
    col = col * T + fogColor(rd) * (1.0 - T) * alpha;
    oCol = vec4(col, alpha);
}
);

/* ---- the gulls ---- */
static const char *VS_BIRD = GLSL(
layout(location = 0) in vec4 aP;
layout(location = 1) in vec4 iA;   /* x, y, z, heading */
layout(location = 2) in vec4 iB;   /* flap, roll, scale, pitch */
uniform mat4 uVP, uMirror;
uniform float uClipY;
out vec3 vP; out vec3 vN; out float vW;
void main() {
    vec3 p = aP.xyz;
    float a = iB.x, s = sign(p.x), ax = abs(p.x);
    vec3 n = vec3(0.0, 1.0, 0.0);
    if (aP.w > 0.5) {
        float r1 = min(ax, 0.45) - 0.08, r2 = max(ax - 0.45, 0.0);
        float a2 = aP.w > 1.5 ? a * 1.6 - 0.25 : a;
        p.x = s * (0.08 + r1 * cos(a) + r2 * cos(a2));
        p.y = r1 * sin(a) + r2 * sin(a2);
        n = vec3(-s * sin(a2), cos(a2), 0.0);
    }
    float cr = cos(iB.y), sr = sin(iB.y), cp = cos(iB.w), sp = sin(iB.w), ch = cos(iA.w), sh = sin(iA.w);
    p = vec3(p.x * cr - p.y * sr, p.x * sr + p.y * cr, p.z);
    n = vec3(n.x * cr - n.y * sr, n.x * sr + n.y * cr, n.z);
    p = vec3(p.x, p.y * cp + p.z * sp, -p.y * sp + p.z * cp);
    n = vec3(n.x, n.y * cp + n.z * sp, -n.y * sp + n.z * cp);
    p = vec3(p.x * ch - p.z * sh, p.y, -p.x * sh - p.z * ch);
    n = vec3(n.x * ch - n.z * sh, n.y, -n.x * sh - n.z * ch);
    vP = iA.xyz + p * iB.z;
    vN = n; vW = aP.w;
    gl_ClipDistance[0] = vP.y - uClipY;
    gl_Position = uVP * (uMirror * vec4(vP, 1.0));
}
);
static const char *FS_BIRD = GLSL(
in vec3 vP; in vec3 vN; in float vW;
out vec4 oCol;
void main() {
    vec3 v = normalize(uCam - vP);
    vec3 n = normalize(vN);
    if (dot(n, v) < 0.0) n = -n;
    vec3 alb = vW > 1.5 ? vec3(0.05) : vec3(0.72, 0.73, 0.74);
    if (vW < 0.5) alb = vec3(0.85);
    vec3 col = shade(vP, n, v, alb, 0.7, 0.0, 1.0, 0.5);
    /* the underside catches the light off the sea */
    col += alb * uAmbSky * 0.3 * max(-n.y, 0.0);
    oCol = vec4(applyFog(col, vP), 1.0);
}
);

/* ---- shadows: depth only ---- */
static const char *VS_DEPTH = GLSL(
layout(location = 0) in vec3 aPos;
uniform mat4 uModel, uVP;
void main() { gl_Position = uVP * (uModel * vec4(aPos, 1.0)); }
);
static const char *FS_DEPTH = GLSL(
void main() {}
);

/* ---- light in the air: shafts through the clouds and the broken wall ---- */
static const char *FS_VOL = GLSL(
in vec2 vUV;
uniform sampler2D uDepth;
uniform vec4 uViewTan;
uniform mat3 uCamRot;
uniform vec2 uNF;                    /* near, far */
uniform float uFrame;
uniform vec3 uBellPos;
out vec4 oCol;
float density(vec3 p) {
    float d = uFogDen * exp(-max(p.y, 0.0) / uFogH) + uMist * exp(-max(p.y, 0.0) / 5.0);
    if (abs(p.x) < 3.0 && abs(p.z) < 3.0 && p.y > 8.6 && p.y < 21.0) d += 0.035;   /* dust in the gear room */
    return d;
}
void main() {
    vec3 rdc = vec3(uViewTan.x + vUV.x * uViewTan.z, uViewTan.y + vUV.y * uViewTan.w, -1.0);
    vec3 rd = normalize(uCamRot * rdc);
    float z = texture(uDepth, vUV).r;
    float zn = z * 2.0 - 1.0;
    float lin = 2.0 * uNF.x * uNF.y / (uNF.y + uNF.x - zn * (uNF.y - uNF.x));
    float tmax = min(lin * length(rdc), z >= 1.0 ? 350.0 : 600.0);
    const int N = 14;
    float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy + uFrame * vec2(5.3, 7.1), vec2(0.06711056, 0.00583715))));
    float dt = tmax / float(N);
    float mu = dot(rd, uKeyDir);
    float g = 0.6;
    float phase = 0.25 * (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5) + 0.08;
    vec3 sum = vec3(0.0);
    float od = 0.0;
    for (int i = 0; i < N; i++) {
        float t = (float(i) + j) * dt;
        vec3 p = uCam + rd * t;
        float dn = density(p);
        float vis = shadowFast(p) * cloudShadowFast(p);
        float T = exp(-od);
        sum += vec3(vis * dn * T * dt);
        od += dn * dt;
    }
    vec3 col = uKeyCol * sum * phase * step(0.0, uKeyDir.y + 0.03);
    /* the bell's note, a faint shell of light spreading out */
    for (int i = 0; i < 4; i++) {
        float tb = uPulse[i];
        if (tb < 0.0 || tb > 6.0) continue;
        float R = tb * 22.0;
        vec3 oc = uCam - uBellPos;
        float b = dot(oc, rd), c = dot(oc, oc) - R * R, disc = b * b - c;
        if (disc > 0.0) {
            float s = sqrt(disc);
            float hit = 0.0;
            float t1 = -b - s, t2 = -b + s;
            if (t1 > 0.0 && t1 < tmax) hit += 1.0;
            if (t2 > 0.0 && t2 < tmax) hit += 1.0;
            float graze = exp(-s * s * 0.03);
            col += vec3(1.0, 0.88, 0.7) * hit * graze * exp(-tb * 0.6) * 0.06 * (uAmbSky + uKeyCol * 0.15 + 0.02);
        }
    }
    oCol = vec4(col, 1.0);
}
);

/* ---- temporal antialiasing: this frame (jittered) over the last, reprojected ---- */
static const char *FS_TAA = GLSL(
in vec2 vUV;
uniform sampler2D uSrc, uDepth, uHistory;
uniform vec4 uViewTanNJ;
uniform mat3 uCamRot;
uniform mat4 uPrevVP;
uniform vec2 uNF;
uniform float uHistOK, uBlend;
out vec4 oCol;
void main() {
    ivec2 ic = ivec2(gl_FragCoord.xy), sz = textureSize(uSrc, 0);
    vec3 c = texelFetch(uSrc, ic, 0).rgb, mn = c, mx = c, m1 = c, m2 = c * c;
    for (int i = 0; i < 8; i++) {
        ivec2 o = ivec2(i < 3 ? i - 1 : i < 5 ? (i == 3 ? -1 : 1) : i - 6, i < 3 ? -1 : i < 5 ? 0 : 1);
        vec3 q = texelFetch(uSrc, clamp(ic + o, ivec2(0), sz - 1), 0).rgb;
        mn = min(mn, q); mx = max(mx, q); m1 += q; m2 += q * q;
    }
    if (uHistOK < 0.5) { oCol = vec4(c, 1.0); return; }
    /* a variance box, tighter than min/max: less ghosting on what moves */
    m1 /= 9.0; m2 /= 9.0;
    vec3 sd = sqrt(max(m2 - m1 * m1, 0.0));
    mn = max(mn, m1 - sd * 1.25); mx = min(mx, m1 + sd * 1.25);
    float z = texelFetch(uDepth, ic, 0).r;
    vec2 uv = (vec2(ic) + 0.5) / vec2(sz);
    vec3 v = vec3(uViewTanNJ.x + uv.x * uViewTanNJ.z, uViewTanNJ.y + uv.y * uViewTanNJ.w, -1.0);
    vec4 pc;
    if (z >= 1.0) pc = uPrevVP * vec4(uCamRot * v, 0.0);
    else {
        float lin = 2.0 * uNF.x * uNF.y / (uNF.y + uNF.x - (z * 2.0 - 1.0) * (uNF.y - uNF.x));
        pc = uPrevVP * vec4(uCam + uCamRot * (v * lin), 1.0);
    }
    vec2 pu = pc.xy / pc.w * 0.5 + 0.5;
    if (pc.w <= 0.0 || any(lessThan(pu, vec2(0.0))) || any(greaterThan(pu, vec2(1.0)))) { oCol = vec4(c, 1.0); return; }
    vec3 h = clamp(texture(uHistory, pu).rgb, mn, mx);
    oCol = vec4(mix(h, c, uBlend), 1.0);
}
);

/* ---- bloom ---- */
static const char *FS_DOWN = GLSL(
in vec2 vUV;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uFirst;
out vec4 oCol;
void main() {
    vec3 c = texture(uSrc, vUV + uTexel * vec2(-1.0, -1.0)).rgb + texture(uSrc, vUV + uTexel * vec2(1.0, -1.0)).rgb +
             texture(uSrc, vUV + uTexel * vec2(-1.0, 1.0)).rgb + texture(uSrc, vUV + uTexel * vec2(1.0, 1.0)).rgb;
    c *= 0.25;
    if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
    if (uFirst > 0.5) {
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c *= smoothstep(0.0, 1.0, l * 0.5) / max(1.0, l * 0.02);
    }
    oCol = vec4(c, 1.0);
}
);
static const char *FS_UP = GLSL(
in vec2 vUV;
uniform sampler2D uSrc;
uniform vec2 uTexel;
out vec4 oCol;
void main() {
    vec3 c = texture(uSrc, vUV).rgb * 4.0;
    c += (texture(uSrc, vUV + vec2(uTexel.x, 0.0)).rgb + texture(uSrc, vUV - vec2(uTexel.x, 0.0)).rgb +
          texture(uSrc, vUV + vec2(0.0, uTexel.y)).rgb + texture(uSrc, vUV - vec2(0.0, uTexel.y)).rgb) * 2.0;
    c += texture(uSrc, vUV + uTexel).rgb + texture(uSrc, vUV - uTexel).rgb +
         texture(uSrc, vUV + vec2(uTexel.x, -uTexel.y)).rgb + texture(uSrc, vUV - vec2(uTexel.x, -uTexel.y)).rgb;
    oCol = vec4(c / 16.0, 1.0);
}
);

/* ---- the grade: tonemap, colour, vignette, grain, rain ---- */
static const char *FS_POST = GLSL(
in vec2 vUV;
uniform sampler2D uScene, uBloom, uVol;
uniform float uExposure, uBloomAmt, uRain;
uniform vec4 uVig;                    /* this view in layout-normalised coords */
uniform vec2 uLayoutPx;               /* the layout's size in pixels */
out vec4 oCol;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 catmull(sampler2D t, vec2 uv) {
    vec2 sz = vec2(textureSize(t, 0)), p = uv * sz - 0.5, c = floor(p) + 0.5, f = p - floor(p);
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2, t0 = (c - 1.0) / sz, t3 = (c + 2.0) / sz, t12 = (c + w2 / w12) / sz;
    vec3 r = texture(t, vec2(t12.x, t0.y)).rgb * w12.x * w0.y + texture(t, vec2(t0.x, t12.y)).rgb * w0.x * w12.y +
             texture(t, t12).rgb * w12.x * w12.y + texture(t, vec2(t3.x, t12.y)).rgb * w3.x * w12.y +
             texture(t, vec2(t12.x, t3.y)).rgb * w12.x * w3.y;
    float ws = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / ws, vec3(0.0));
}
void main() {
    vec3 c = catmull(uScene, vUV);
    if (any(isnan(texture(uScene, vUV).rgb))) { oCol = vec4(1.0, 0.0, 1.0, 1.0); return; }
    c += texture(uVol, vUV).rgb;
    c += texture(uBloom, vUV).rgb * uBloomAmt;
    vec2 L = uVig.xy + vUV * uVig.zw;               /* where on the whole wall */
    /* rain streaks, falling across the whole wall */
    if (uRain > 0.0) {
        vec2 q = L * uLayoutPx / 900.0;
        float r = 0.0;
        for (int i = 0; i < 3; i++) {
            float sc = 1.0 + float(i) * 0.7;
            vec2 rq = vec2(q.x * 26.0 * sc + q.y * 2.0, q.y * 1.3 * sc + uTime * (2.2 + float(i) * 0.6));
            vec2 cell = floor(rq);
            float h = hash12(cell + float(i) * 13.0);
            float x = abs(fract(rq.x) - 0.2 - 0.6 * h);
            float y = fract(rq.y + h * 7.0);
            r += smoothstep(0.06, 0.0, x) * smoothstep(0.0, 0.3, y) * smoothstep(0.75, 0.3, y) * step(0.62, h) * (0.7 - 0.2 * float(i));
        }
        c += vec3(0.55, 0.6, 0.65) * r * uRain * 0.05 * (c + 0.05);
    }
    c *= uExposure;
    /* a filmic curve, then a gentle split-tone: cool shadows, warm lights */
    c = aces(c);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, 0.88);
    c += (vec3(-0.012, 0.0, 0.018) * (1.0 - l) + vec3(0.02, 0.008, -0.016) * l) * 0.9;
    c = max(c, 0.0);
    vec2 vq = L - 0.5;
    c *= 1.0 - 0.32 * smoothstep(0.3, 0.85, length(vq * vec2(1.0, 0.9)));
    c = pow(c, vec3(1.0 / 2.2));
    /* film grain, fine */
    float gr = hash12(gl_FragCoord.xy + fract(uTime * 7.0) * 977.0) - 0.5;
    c += gr * 0.022 * (1.0 - l * 0.6);
    oCol = vec4(c, 1.0);
}
);

/* ======================================================================
 * The moment: weather, light, the clock, the bell, the birds, the camera
 * ====================================================================== */
enum { W_OVERCAST, W_CLEAR, W_MIST, W_RAIN };
static int weather;
static const char *weather_name[] = {"overcast", "clear", "mist", "rain"};
static struct { float cloud, fog, fogh, mist, wet, wave, rain; } WP;

static void choose_weather(void) {
    weather = -1;
    for (int i = 0; i < 4; i++) if (!strcmp(opt_weather, weather_name[i])) weather = i;
    if (weather < 0) {
        float r = hash1((int32_t)opt_seed, 90);
        weather = r < 0.33f ? W_OVERCAST : r < 0.63f ? W_CLEAR : r < 0.83f ? W_MIST : W_RAIN;
    }
    switch (weather) {
    case W_OVERCAST: WP = (typeof(WP)){0.80f, 0.0016f, 150, 0.0f, 0.15f, 0.8f, 0}; break;
    case W_CLEAR:    WP = (typeof(WP)){0.30f, 0.00035f, 260, 0.0f, 0.0f, 0.5f, 0}; break;
    case W_MIST:     WP = (typeof(WP)){0.62f, 0.0009f, 120, 0.12f, 0.35f, 0.3f, 0}; break;
    default:         WP = (typeof(WP)){0.97f, 0.0032f, 120, 0.004f, 1.0f, 0.95f, 1}; break;
    }
}

/* everything the shaders share this frame */
static struct {
    v3 cam, key_dir, key_col, zen, hor, glow, amb_sky, amb_gnd, fog_col;
    float fog_den, fog_h, mist, tide, wet, cloud, night, illum, wave;
    float time, exposure;
    float hands[4][4], pulse[4];
    m4 vp, shadow_vp, mirror;
    float viewtan[4], rot3[9];
    float clip_y, lowq, mirror_sky;
    v3 vrt, vup, vbk;                   /* camera basis */
} F;

static v3 lerp3(v3 a, v3 b, float t) { return vlerp(a, b, clampf(t, 0, 1)); }

static void light_setup(void) {
    float a = (float)sky.sun_alt;
    v3 sunc = vscale(lerp3(V3(1.0f, 0.48f, 0.2f), V3(1.0f, 0.95f, 0.87f), smooth01(1, 28, a)), 3.4f * smooth01(-1.0f, 5.0f, a));
    /* sky colours through the day */
    v3 zd = V3(0.13f, 0.26f, 0.52f), hd = V3(0.60f, 0.68f, 0.76f);
    v3 zg = V3(0.16f, 0.22f, 0.40f), hg = V3(0.98f, 0.62f, 0.36f);
    v3 zu = V3(0.05f, 0.06f, 0.13f), hu = V3(0.36f, 0.22f, 0.24f);
    v3 zn = V3(0.006f, 0.009f, 0.02f), hn = V3(0.012f, 0.017f, 0.03f);
    v3 zen, hor;
    if (a > 4) { float t = smooth01(4, 22, a); zen = lerp3(zg, zd, t); hor = lerp3(hg, hd, t); }
    else if (a > -4) { float t = smooth01(-4, 4, a); zen = lerp3(zu, zg, t); hor = lerp3(hu, hg, t); }
    else { float t = smooth01(-14, -4, a); zen = lerp3(zn, zu, t); hor = lerp3(hn, hu, t); }
    /* the Moon brightens the night sky a little */
    float mup = smooth01(-2, 15, (float)sky.moon_alt) * sky.illum;
    float nightk = smooth01(-4, -14, a);
    zen = vadd(zen, vscale(V3(0.010f, 0.014f, 0.028f), mup * nightk));
    hor = vadd(hor, vscale(V3(0.014f, 0.018f, 0.03f), mup * nightk));
    /* cloud greys it all */
    float cl = WP.cloud;
    float lum = (zen.x + zen.y + zen.z + hor.x + hor.y + hor.z) / 6;
    v3 grey = V3(lum * 0.95f, lum * 0.98f, lum * 1.05f);
    float g = smooth01(0.4f, 1.0f, cl) * 0.85f;
    zen = lerp3(zen, vscale(grey, 0.9f), g);
    hor = lerp3(hor, vscale(grey, 1.15f), g);
    F.zen = zen; F.hor = hor;
    F.glow = vscale(sunc, 0.22f * (1 - g * 0.6f));
    /* the key light: the Sun by day, the Moon by night */
    if (a > -3 || sky.moon_alt < 0) {
        F.key_dir = sky.sun;
        if (F.key_dir.y < 0.03f) F.key_dir = vnorm(V3(F.key_dir.x, 0.03f, F.key_dir.z));
        F.key_col = sunc;
    } else {
        F.key_dir = sky.moon;
        if (F.key_dir.y < 0.05f) F.key_dir = vnorm(V3(F.key_dir.x, 0.05f, F.key_dir.z));
        F.key_col = vscale(V3(0.55f, 0.64f, 0.85f), 0.16f * powf(sky.illum, 1.3f) * smooth01(-2, 12, (float)sky.moon_alt) * nightk);
    }
    if (a > -3 && a < 0 && sky.moon_alt > 0) F.key_col = vscale(F.key_col, 1);
    F.amb_sky = vscale(vadd(vscale(zen, 0.55f), vscale(hor, 0.45f)), 1.25f + 0.6f * g);
    F.amb_gnd = vadd(vscale(F.amb_sky, 0.28f), vscale(F.key_col, 0.035f * fmaxf(F.key_dir.y, 0) * (1 - cl * 0.7f)));
    F.fog_col = vscale(vadd(vscale(hor, 0.75f), vscale(zen, 0.25f)), 0.95f);
    F.fog_den = WP.fog; F.fog_h = WP.fogh; F.mist = WP.mist;
    F.wet = WP.wet; F.cloud = cl; F.wave = WP.wave;
    F.night = smooth01(2.5f, -4.5f, a);
    F.illum = sky.illum;
    /* exposure: aim the scene's typical brightness at mid grey */
    float kl = (F.key_col.x + F.key_col.y + F.key_col.z) / 3 * fmaxf(F.key_dir.y, 0.15f) * (1 - 0.8f * cl);
    float al = (F.amb_sky.x + F.amb_sky.y + F.amb_sky.z) / 3;
    float sl = kl * 0.35f + al * 0.75f + 0.0018f;
    F.exposure = clampf(0.30f / sl, 0.25f, 26.0f) * mixf(1.0f, 0.42f, F.night);
    F.exposure *= weather == W_RAIN ? 0.78f : weather == W_OVERCAST ? 0.9f : 1.0f;
}

/* ---- the clock: local time, and what the bell is doing ---- */
static struct {
    float hour_ang, min_ang;
    float lift;                         /* the hammer, 0..1 */
    float s1;                           /* the striking train, in blows */
    float flyspin;                      /* 0..1 how fast the fly turns */
    float swing;                        /* the bell, radians */
    float pulse[4];
    double t_birds;                     /* seconds since the birds last took off */
    double esc;                         /* the escape wheel, in teeth */
    float pend;                         /* the pendulum, radians */
    float drum[4], lever;
} CK;

static int blows_in(long h) { int k = (int)(((h % 12) + 12) % 12); return k ? k : 12; }

static void clock_update(double wt) {
    time_t ts = (time_t)floor(wt);
    struct tm tm;
    localtime_r(&ts, &tm);
    double frac = wt - floor(wt);
    double sec = tm.tm_sec + frac, mins = tm.tm_min + sec / 60;
    CK.min_ang = (float)(mins * 6 * D2R);
    CK.hour_ang = (float)(((tm.tm_hour % 12) + mins / 60) * 30 * D2R);
    /* the escapement, in step with the pendulum: the wheel lets go half a
     * tooth each time the pendulum passes the middle */
    double beat = pend_T / 2;
    double cyc = 4320 * beat;
    double tp = fmod(wt, cyc);
    double kb = floor(tp / beat), fb = tp / beat - kb;
    double drop = fb < 0.06 ? smooth01(0, 0.06f, (float)fb) : 1.0;
    drop += 0.05 * exp(-fb * 30) * sin(fb * 90) * (fb > 0.06);
    CK.esc = (kb + drop - 1) * 0.5;
    CK.pend = 0.058f * (float)sin(2 * M_PI * tp / pend_T);
    /* the hour: blows at 2 s past and every 3.3 s after */
    double th = wt - (tm.tm_min * 60 + sec);            /* the start of this hour */
    long habs = (long)floor((wt + tm.tm_gmtoff) / 3600);
    int n = blows_in(tm.tm_hour);
    long base = (habs / 12) * 78;
    int k12 = (int)(habs % 12);
    if (k12 > 0) base += 12 + (long)(k12 - 1) * k12 / 2;
    double t = wt - th;
    float lift = 0, part = 0, swing = 0;
    int done = 0;
    for (int i = 0; i < 4; i++) CK.pulse[i] = -1;
    for (int i = 0; i < n; i++) {
        double tb = 2.0 + 3.3 * i;
        if (t >= tb) {
            done++;
            double d = t - tb;
            swing += 0.055f * (float)(exp(-d / 2.4) * sin(d * 2 * M_PI / 1.75));
            for (int j = 3; j > 0; j--) CK.pulse[j] = CK.pulse[j - 1];
            CK.pulse[0] = (float)d;
            lift = fmaxf(lift, 0.12f * (float)(exp(-d * 9) * fabs(sin(d * 20))) * (d < 0.6));
        } else if (t > tb - 1.6) {
            float u = (float)((t - (tb - 1.6)) / 1.55);
            lift = fmaxf(lift, smooth01(0, 1, u));
            part = fmaxf(part, smooth01(0, 1, u));
        }
    }
    CK.lift = lift;
    CK.s1 = (float)(fmod((double)base, 8.0 * 9 * 5) + done + part);
    CK.flyspin = part > 0 && part < 1 ? 1.0f : 0.0f;
    CK.swing = swing;
    /* the gulls go up at the first blow and come back in their own time */
    double t0 = th + 2.0;
    CK.t_birds = wt >= t0 ? wt - t0 : wt - (t0 - 3600);
    /* the drums: the time away, hours and minutes; and the lever that works them */
    double away = wt - (double)locked_at;
    if (away < 0) away = 0;
    long am = (long)floor(away / 60);
    double s_in = away - am * 60.0;
    float roll = smooth01(0.0f, 0.9f, (float)s_in);
    long prev = am > 0 ? am - 1 : 0;
    for (int d = 0; d < 4; d++) {
        long hh = am / 60, mm = am % 60, ph = prev / 60, pm = prev % 60;
        int a = d == 0 ? (int)((hh / 10) % 10) : d == 1 ? (int)(hh % 10) : d == 2 ? (int)(mm / 10) : (int)(mm % 10);
        int b = d == 0 ? (int)((ph / 10) % 10) : d == 1 ? (int)(ph % 10) : d == 2 ? (int)(pm / 10) : (int)(pm % 10);
        int step = ((a - b) % 10 + 10) % 10;
        float over = roll < 1 ? 0.06f * sinf(roll * PI) : 0;
        CK.drum[d] = (float)b + step * (roll + over * (step > 0));
    }
    /* the lever comes down just before the minute and goes back up after */
    double sl = fmod(s_in + 1.2, 60.0);
    float down = sl < 1.8 ? smooth01(0, 0.5f, (float)sl) : 1 - smooth01(1.8f, 4.5f, (float)sl);
    CK.lever = 0.38f - 0.85f * down;
}

/* a gear meshing with another: b's angle from a's, given their tooth
 * counts and the direction from a to b in the wheel plane */
static float mesh_ang(float tha, int na, int nb, P2 a, P2 b) {
    float phi = atan2f(b.z - a.z, b.y - a.y);
    return phi + PI - PI / nb + (phi - tha) * na / nb;
}

static void parts_update(void) {
    /* the going train turns from the escape wheel back */
    float te = (float)(-fmod(CK.esc, 30.0 * 72) * 2 * PI / 30);
    float tg2 = mesh_ang(te, 8, 72, C_E, C_G2);
    float tg1 = mesh_ang(tg2, 12, 96, C_G2, C_G1);
    v3 X = V3(1, 0, 0);
    part[D_E].model = m4mul(m4trans(part[D_E].pivot), m4rot(X, te));
    part[D_G2].model = m4mul(m4trans(part[D_G2].pivot), m4rot(X, tg2));
    part[D_G1].model = m4mul(m4trans(part[D_G1].pivot), m4rot(X, tg1));
    part[D_ANCHOR].model = m4mul(m4trans(part[D_ANCHOR].pivot), m4rot(X, CK.pend));
    part[D_PEND].model = m4mul(m4trans(part[D_PEND].pivot), m4rot(X, CK.pend));
    /* the striking train */
    float ts1 = -CK.s1 * 2 * PI / 8;
    float ts2 = mesh_ang(ts1, 96, 12, C_S1, C_S2);
    float tf = mesh_ang(ts2, 60, 8, C_S2, C_F);
    part[D_S1].model = m4mul(m4trans(part[D_S1].pivot), m4rot(X, ts1));
    part[D_S2].model = m4mul(m4trans(part[D_S2].pivot), m4rot(X, ts2));
    part[D_FLY].model = m4mul(m4trans(part[D_FLY].pivot), m4rot(X, tf));
    part[D_TAIL].model = m4mul(m4trans(part[D_TAIL].pivot), m4rot(X, -0.28f * CK.lift));
    part[D_ROD].model = m4trans(vadd(part[D_ROD].pivot, V3(0, -0.12f * CK.lift, 0)));
    part[D_HAMMER].model = m4mul(m4trans(part[D_HAMMER].pivot), m4rot(V3(0, 0, 1), -0.5f * CK.lift));
    part[D_BELL].model = m4mul(m4trans(part[D_BELL].pivot), m4rot(V3(0, 0, 1), CK.swing));
    part[D_LEAD].model = m4mul(m4trans(part[D_LEAD].pivot), m4rot(V3(0, 1, 0), tg2 * 0.5f));
    /* the weights go down through the hours (and are wound, unseen, at twelve) */
    float hf = fmodf(CK.hour_ang / (2 * PI), 1.0f);
    float yg = 11.4f - 2.0f * hf, ys = 11.4f - 1.9f * fmodf((CK.s1) / 78.0f, 1.0f);
    v3 cg = V3(0.22f, 0, C_G1.z - 0.43f), cs = V3(0.22f, 0, C_S1.z + 0.43f);
    part[D_WEIGHT_G].model = m4trans(V3(cg.x, yg, cg.z));
    part[D_WEIGHT_S].model = m4trans(V3(cs.x, ys, cs.z));
    m4 sg = m4id(); sg.m[5] = C_G1.y - (yg + 0.25f); sg.m[12] = cg.x; sg.m[13] = yg + 0.25f; sg.m[14] = cg.z;
    m4 ss = m4id(); ss.m[5] = C_S1.y - (ys + 0.25f); ss.m[12] = cs.x; ss.m[13] = ys + 0.25f; ss.m[14] = cs.z;
    part[D_CABLE_G].model = sg;
    part[D_CABLE_S].model = ss;
    for (int d = 0; d < 4; d++)
        part[D_DRUM0 + d].model = m4mul(m4trans(part[D_DRUM0 + d].pivot), m4rot(X, CK.drum[d] * 2 * PI / 10));
    part[D_LEVER].model = m4mul(m4trans(part[D_LEVER].pivot), m4rot(V3(1, 0, 0), CK.lever));
    /* the armillary: one ring keeps the Moon's bearing, the other turns slowly */
    float maz = atan2f(sky.moon.x, -sky.moon.z);
    part[D_ARMIL].model = m4mul(m4trans(part[D_ARMIL].pivot), m4rot(V3(0, 1, 0), -maz));
    part[D_ARMIL2].model = m4mul(m4trans(part[D_ARMIL2].pivot), m4mul(m4rot(V3(0, 1, 0), -maz), m4rot(V3(0, 0, 1), F.time * 0.05f)));
    for (int f = 0; f < 4; f++) { F.hands[f][0] = CK.hour_ang; F.hands[f][1] = CK.min_ang; F.hands[f][2] = F.hands[f][3] = 0; }
    for (int i = 0; i < 4; i++) F.pulse[i] = CK.pulse[i];
}

/* ---- gulls: most sit on the parapet until the bell sends them up ---- */
#define NBIRD 21
#define NPERCH 14
static float bird_inst[NBIRD * 8];
static int bird_n;
static void bird_at(int b, double t, v3 *pos, float *flap, float *roll) {
    float h1 = hash1(b, 100), h2 = hash1(b, 101), h3 = hash1(b, 102), h4 = hash1(b, 103);
    if (b >= NPERCH) {      /* free birds, always out over the water */
        float R = 35 + 60 * h1, v = 8 + 4 * h2, dir = h3 < 0.5f ? 1.f : -1.f;
        float ang = h4 * 6.28f + (float)fmod(t * v / R, 2 * M_PI) * dir;
        v3 c = V3(-30 + 60 * h2, 0, 30 + 60 * h3);
        *pos = V3(c.x + R * cosf(ang), 14 + 30 * h1 + 4 * sinf((float)t * 0.11f + h2 * 6), c.z + R * sinf(ang));
        float ft = (float)t + h1 * 10;
        float glide = smooth01(0.1f, 0.5f, sinf(ft * 0.33f + h3 * 5));
        *flap = (1 - glide) * 0.7f * sinf(ft * 10.0f) + glide * 0.1f;
        *roll = -dir * 0.3f;
        return;
    }
    /* the perch: on the coping, facing out */
    int side = b % 4;
    float along = -3.6f + 7.2f * h1;
    v3 perch = side == 0 ? V3(along, 34.42f, 4.18f) : side == 1 ? V3(4.18f, 34.42f, along) : side == 2 ? V3(along, 34.42f, -4.18f) : V3(-4.18f, 34.42f, along);
    /* the orbit */
    float R = 14 + 20 * h2, v = 7 + 3 * h3, dir = h4 < 0.5f ? 1.f : -1.f;
    float ang = h1 * 6.28f + (float)(t * v / R) * dir;
    v3 orb = V3(R * cosf(ang), 38 + 10 * h3 + 3 * sinf((float)t * 0.2f + h1 * 9), R * sinf(ang));
    float back = 70 + 170 * h4;
    float w = smooth01(0, 6, (float)t) * (1 - smooth01(back, back + 16, (float)t));
    float w2 = w * w * (3 - 2 * w);
    *pos = vadd(vlerp(perch, orb, w2), V3(0, 5 * sinf(PI * w2) * (1 - w2), 0));
    float ft = (float)t + h1 * 10;
    if (w <= 0.001f) { *flap = -1.2f; *roll = 0; }
    else {
        float climb = 1 - smooth01(0.6f, 0.95f, w);
        float glide = smooth01(0.1f, 0.5f, sinf(ft * 0.4f + h3 * 5)) * (1 - climb);
        *flap = (1 - glide) * 0.75f * sinf(ft * 11.0f) + glide * 0.12f;
        *roll = -dir * 0.35f * w;
    }
}
static void birds_update(double t) {
    bird_n = 0;
    for (int b = 0; b < NBIRD; b++) {
        v3 p, p2;
        float flap, roll, f2, r2;
        bird_at(b, t, &p, &flap, &roll);
        bird_at(b, t + 0.15, &p2, &f2, &r2);
        v3 vel = vsub(p2, p);
        float hv = sqrtf(vel.x * vel.x + vel.z * vel.z);
        float *o = bird_inst + 8 * bird_n++;
        o[0] = p.x; o[1] = p.y + (flap < -1 ? 0.12f : 0); o[2] = p.z;
        if (hv > 0.01f) o[3] = atan2f(-vel.x, -vel.z);
        else {   /* perched: face out from the tower */
            o[3] = atan2f(p.x, p.z) + PI;
        }
        o[4] = flap; o[5] = roll; o[6] = 1.0f + 0.25f * hash1(b, 104);
        o[7] = hv > 0.01f ? atanf(vel.y / hv) : 0;
    }
}

/* ---- the camera: a slow tour, as one smooth path through key shots ---- */
typedef struct { float t; v3 p, look; float fov, sx, sy, span; } Key;
static const Key keys[] = {
    {0,   {30, 4.6f, 176},   {0, 20, 0},       40, 0.38f, 0.06f, 1},
    {36,  {16, 4.2f, 170},   {0, 19, 0},       42, 0.30f, 0.05f, 1},
    {66,  {7.0f, 3.4f, 162}, {1, 16, 0},       46, 0.18f, 0.05f, 1},
    {96,  {9.0f, 2.6f, 118}, {0, 17, 0},       48, 0.15f, 0.05f, 1},
    {124, {6.5f, 2.4f, 80},  {-1, 15, 0},      48, 0.10f, 0.05f, 1},
    {150, {-1.5f, 3.4f, 30}, {-5.3f, 8.4f, 5.4f}, 30, 0.15f, 0.0f, 0.3f},
    {172, {14, 6.5f, 27},    {0, 16, 0},       50, -0.2f, 0.05f, 1},
    {192, {18, 12, 7},       {2, 14.5f, 0},    44, -0.1f, 0.0f,  0.7f},
    {210, {8.0f, 14.2f, 1.6f}, {0.6f, 14.2f, -0.4f}, 46, 0, 0, 0},
    {228, {3.7f, 14.8f, -0.3f}, {1.0f, 15.4f, -1.0f}, 50, 0, 0, 0},
    {246, {3.5f, 14.0f, 1.2f}, {0.7f, 14.6f, 1.7f}, 52, 0, 0, 0},
    {263, {3.9f, 11.9f, -0.7f}, {1.4f, 10.0f, -1.0f}, 52, 0, 0, 0},
    {282, {9.5f, 21.5f, 5.0f}, {0, 26, 0},     50, 0, 0, 0.5f},
    {300, {7.6f, 30.4f, 2.0f}, {0, 30.6f, 0},  46, 0, 0, 0.3f},
    {318, {15, 37, 13},      {0, 31, 0},       50, -0.1f, 0, 0.8f},
    {346, {75, 30, 95},      {0, 21, 0},       44, 0.3f, 0, 1},
};
#define NKEY ((int)(sizeof keys / sizeof keys[0]))
#define TOUR 376.0f

static v3 herm(v3 p0, v3 p1, v3 m0, v3 m1, float s, float dt) {
    float s2 = s * s, s3 = s2 * s;
    float h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    return vadd(vadd(vscale(p0, h00), vscale(m0, h10 * dt)), vadd(vscale(p1, h01), vscale(m1, h11 * dt)));
}
static void key_at(int i, float *t, v3 *p, v3 *l, float *f, v3 *fr) {
    int n = NKEY, k = ((i % n) + n) % n;
    float wrap = (float)((i - k) / n) * TOUR;
    *t = keys[k].t + wrap; *p = keys[k].p; *l = keys[k].look; *f = keys[k].fov;
    *fr = V3(keys[k].sx, keys[k].sy, keys[k].span);
}
static void tour(float t, v3 *pos, v3 *look, float *fov, v3 *frame) {
    if (opt_shotcam >= 0) t = keys[opt_shotcam % NKEY].t + fmodf(t, 1.0f) * 0;
    t = fmodf(t, TOUR);
    if (t < 0) t += TOUR;
    int i = 0;
    while (i + 1 < NKEY && keys[i + 1].t <= t) i++;
    float ta, tb, tp, tn, fa, fb, fp, fn;
    v3 pa, pb, pp, pn, la, lb, lp, ln, ra, rb, rp, rn;
    key_at(i - 1, &tp, &pp, &lp, &fp, &rp);
    key_at(i, &ta, &pa, &la, &fa, &ra);
    key_at(i + 1, &tb, &pb, &lb, &fb, &rb);
    key_at(i + 2, &tn, &pn, &ln, &fn, &rn);
    if (i == 0 && t < ta) { ta -= TOUR; tb -= TOUR; tp -= TOUR; tn -= TOUR; }
    float dt = tb - ta, s = (t - ta) / dt;
    /* tangents scaled to the time between keys (C1 in time) */
    v3 m0 = vscale(vsub(pb, pp), 1 / (tb - tp)), m1 = vscale(vsub(pn, pa), 1 / (tn - ta));
    v3 n0 = vscale(vsub(lb, lp), 1 / (tb - tp)), n1 = vscale(vsub(ln, la), 1 / (tn - ta));
    *pos = herm(pa, pb, m0, m1, s, dt);
    *look = herm(la, lb, n0, n1, s, dt);
    /* never in the ground: ride a little over it */
    float g = terrain(pos->x, pos->z) + 1.7f;
    if (pos->y < g + 1.0f) pos->y = g + 1.0f * expf((pos->y - g - 1.0f) / 1.0f);
    float e = s * s * (3 - 2 * s);
    *fov = mixf(fa, fb, e);
    *frame = vlerp(ra, rb, e);
}

/* ======================================================================
 * Rendering
 * ====================================================================== */
static GLuint p_mesh, p_terr, p_sky, p_water, p_bird, p_depth, p_vol, p_down, p_up, p_post;
static GLuint vao_stat, vao_dyn, vao_water, vao_bird, vao_empty, vbo_bird;
static int water_n;
static int opt_msaa = 1;
#define SHADOW_N 2048
static GLuint tex_shadow, fbo_shadow;
/* scene targets */
static int sc_w, sc_h;
static GLuint fbo_ms, rb_ms_col, rb_ms_dep;
static GLuint fbo_scene, tex_scene, tex_depth;
static GLuint fbo_refl, tex_refl, rb_refl_dep;
static GLuint fbo_vol, tex_vol;
static GLuint fbo_taa[2], tex_taa[2];
static int taa_cur, taa_hist;
static m4 prev_vp;
static GLuint p_taa;
#define NBLOOM 5
static GLuint fbo_bloom[NBLOOM], tex_bloom[NBLOOM];
static int bloom_w[NBLOOM], bloom_h[NBLOOM];
static float render_scale = 0.6f;
static unsigned frame_no;

static GLuint compile(GLenum type, const char *a, const char *b, const char *name) {
    GLuint s = glCreateShader(type);
    const char *src[3] = {type == GL_VERTEX_SHADER ? "#version 330 core\n#define VERTEX\n" : "#version 330 core\n", a, b};
    glShaderSource(s, b ? 3 : 2, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "clocktower: shader %s:\n%s\n", name, log);
        exit(1);
    }
    return s;
}
/* fragment (and most vertex) shaders get COMMON in front */
static GLuint mk_prog(const char *name, const char *vs, int vs_common, const char *fs, int fs_common) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs_common ? COMMON : vs, vs_common ? vs : NULL, name));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs_common ? COMMON : fs, fs_common ? fs : NULL, name));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "clocktower: link %s:\n%s\n", name, log);
        exit(1);
    }
    return pr;
}
#define U(p, n) glGetUniformLocation(p, n)

static GLuint mesh_vao(Mesh *m) {
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)m->n * VSZ * sizeof(float)), m->v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)12);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)24);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, VSZ * 4, (void *)32);
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray((GLuint)i);
    glBindVertexArray(0);
    return vao;
}

static void gl_init(void) {
    p_mesh = mk_prog("mesh", VS_MESH, 1, FS_MESH, 1);
    p_terr = mk_prog("terrain", VS_MESH, 1, FS_TERRAIN, 1);
    p_sky = mk_prog("sky", VS_FULL, 0, FS_SKY, 1);
    p_water = mk_prog("water", VS_WATER, 1, FS_WATER, 1);
    p_bird = mk_prog("bird", VS_BIRD, 1, FS_BIRD, 1);
    p_depth = mk_prog("depth", VS_DEPTH, 0, FS_DEPTH, 0);
    p_vol = mk_prog("vol", VS_FULL, 0, FS_VOL, 1);
    p_down = mk_prog("down", VS_FULL, 0, FS_DOWN, 1);
    p_up = mk_prog("up", VS_FULL, 0, FS_UP, 1);
    p_post = mk_prog("post", VS_FULL, 0, FS_POST, 1);
    p_taa = mk_prog("taa", VS_FULL, 0, FS_TAA, 1);
    glGenVertexArrays(1, &vao_empty);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    double t0 = now();
    movement_layout();
    build_tower(&mstat);
    build_side(&mstat);
    build_causeway(&mstat);
    build_dynamic(&mdyn);
    build_terrain();
    double t1 = now();
    build_heights();
    build_noise();
    build_waves();
    build_digits();
    if (getenv("CLOCKTOWER_DEBUG"))
        for (int k = 0; k < NKEY; k++)
            fprintf(stderr, "key %2d t %3.0f  cam %6.1f %5.1f %6.1f  ground %5.1f\n", k, keys[k].t, keys[k].p.x, keys[k].p.y, keys[k].p.z,
                    terrain(keys[k].p.x, keys[k].p.z));
    if (getenv("CLOCKTOWER_DEBUG"))
        fprintf(stderr, "clocktower: geometry %d + %d + %d verts (%d tris) in %.0f ms, maps %.0f ms\n", mstat.n, mdyn.n, terr_nv, terr_ni / 3,
                (t1 - t0) * 1e3, (now() - t1) * 1e3);
    vao_stat = mesh_vao(&mstat);
    vao_dyn = mesh_vao(&mdyn);

    /* the sea: a polar grid round the camera, dense near it */
    {
        enum { NR = 150, NA = 200 };
        float *v = malloc(sizeof(float) * 2 * 6 * NR * NA);
        int k = 0;
        for (int i = 0; i < NR; i++) {
            float r0 = 0.25f * powf(1.068f, (float)i) - 0.25f, r1 = 0.25f * powf(1.068f, (float)(i + 1)) - 0.25f;
            if (r1 > 9000) r1 = 9000;
            for (int j = 0; j < NA; j++) {
                float a0 = 2 * PI * j / NA, a1 = 2 * PI * (j + 1) / NA;
                float q[6][2] = {{r0, a0}, {r1, a0}, {r1, a1}, {r0, a0}, {r1, a1}, {r0, a1}};
                for (int c = 0; c < 6; c++) { v[k++] = q[c][0]; v[k++] = q[c][1]; }
            }
        }
        water_n = k / 2;
        GLuint vbo;
        glGenVertexArrays(1, &vao_water);
        glBindVertexArray(vao_water);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * (size_t)k), v, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
        glEnableVertexAttribArray(0);
        free(v);
    }
    /* a gull: body and two jointed wings (as flightsim's) */
    {
        static const float v[][4] = {
            {0, 0, 0.35f, 0}, {-0.08f, 0, 0.05f, 0}, {0.08f, 0, 0.05f, 0},
            {-0.08f, 0, 0.05f, 0}, {0, 0.02f, -0.3f, 0}, {0.08f, 0, 0.05f, 0},
            {0, 0.02f, -0.3f, 0}, {-0.1f, 0, -0.42f, 0}, {0.1f, 0, -0.42f, 0},
            {-0.08f, 0, 0.12f, 1}, {-0.45f, 0, 0.08f, 1}, {-0.08f, 0, -0.08f, 1},
            {-0.08f, 0, -0.08f, 1}, {-0.45f, 0, 0.08f, 1}, {-0.45f, 0, -0.12f, 1},
            {-0.45f, 0, 0.08f, 2}, {-0.86f, 0, -0.10f, 2}, {-0.45f, 0, -0.12f, 2},
            {0.08f, 0, 0.12f, 1}, {0.08f, 0, -0.08f, 1}, {0.45f, 0, 0.08f, 1},
            {0.08f, 0, -0.08f, 1}, {0.45f, 0, -0.12f, 1}, {0.45f, 0, 0.08f, 1},
            {0.45f, 0, 0.08f, 2}, {0.45f, 0, -0.12f, 2}, {0.86f, 0, -0.10f, 2},
        };
        GLuint vbo;
        glGenVertexArrays(1, &vao_bird);
        glBindVertexArray(vao_bird);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
        glGenBuffers(1, &vbo_bird);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_bird);
        glBufferData(GL_ARRAY_BUFFER, sizeof bird_inst, NULL, GL_STREAM_DRAW);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
        glVertexAttribDivisor(1, 1);
        glVertexAttribDivisor(2, 1);
        glBindVertexArray(0);
    }
    /* the shadow map */
    glGenTextures(1, &tex_shadow);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_N, SHADOW_N, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_shadow, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static GLuint mk_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}

static void resize_targets(int w, int h) {
    if (w == sc_w && h == sc_h) return;
    if (sc_w) {
        if (opt_msaa > 1) { glDeleteFramebuffers(1, &fbo_ms); glDeleteRenderbuffers(1, &rb_ms_col); glDeleteRenderbuffers(1, &rb_ms_dep); }
        glDeleteFramebuffers(1, &fbo_scene); glDeleteTextures(1, &tex_scene); glDeleteTextures(1, &tex_depth);
        glDeleteFramebuffers(1, &fbo_refl); glDeleteTextures(1, &tex_refl); glDeleteRenderbuffers(1, &rb_refl_dep);
        glDeleteFramebuffers(1, &fbo_vol); glDeleteTextures(1, &tex_vol);
        glDeleteFramebuffers(2, fbo_taa); glDeleteTextures(2, tex_taa);
        glDeleteFramebuffers(NBLOOM, fbo_bloom); glDeleteTextures(NBLOOM, tex_bloom);
    }
    sc_w = w; sc_h = h;
    int ms = opt_msaa;
    if (ms > 1) {
    glGenFramebuffers(1, &fbo_ms);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
    glGenRenderbuffers(1, &rb_ms_col);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_col);
    if (ms > 1) glRenderbufferStorageMultisample(GL_RENDERBUFFER, ms, GL_RGBA16F, w, h);
    else glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA16F, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_ms_col);
    glGenRenderbuffers(1, &rb_ms_dep);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_dep);
    if (ms > 1) glRenderbufferStorageMultisample(GL_RENDERBUFFER, ms, GL_DEPTH_COMPONENT32F, w, h);
    else glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_ms_dep);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("multisampled target incomplete");
    }

    tex_scene = mk_tex(GL_RGBA16F, w, h, GL_RGBA, GL_HALF_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    tex_depth = mk_tex(GL_DEPTH_COMPONENT32F, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    fbo_scene = mk_fbo(tex_scene);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_depth, 0);

    int rw = (w + 1) / 2, rh = (h + 1) / 2;
    tex_refl = mk_tex(GL_RGBA16F, rw, rh, GL_RGBA, GL_HALF_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    fbo_refl = mk_fbo(tex_refl);
    glGenRenderbuffers(1, &rb_refl_dep);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_refl_dep);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, rw, rh);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_refl_dep);

    for (int i = 0; i < 2; i++) {
        tex_taa[i] = mk_tex(GL_R11F_G11F_B10F, w, h, GL_RGB, GL_HALF_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_taa[i] = mk_fbo(tex_taa[i]);
    }
    taa_hist = 0;
    tex_vol = mk_tex(GL_R11F_G11F_B10F, (w + 3) / 4, (h + 3) / 4, GL_RGB, GL_HALF_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    fbo_vol = mk_fbo(tex_vol);
    int bw = w, bh = h;
    for (int i = 0; i < NBLOOM; i++) {
        bw = (bw + 1) / 2; bh = (bh + 1) / 2;
        if (bw < 1) bw = 1;
        if (bh < 1) bh = 1;
        bloom_w[i] = bw; bloom_h[i] = bh;
        tex_bloom[i] = mk_tex(GL_R11F_G11F_B10F, bw, bh, GL_RGB, GL_HALF_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_bloom[i] = mk_fbo(tex_bloom[i]);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/* ---- the view: one virtual camera; each output renders its own slice ---- */
static float view_x, view_y, view_w, view_h, cam_cx, cam_cy, cam_f, lay_x0, lay_y0, lay_w, lay_h;
static float ref_h;                    /* the front output's height (framing unit) */
static float por_x, por_y;             /* where a portrait output's centre is, relative to the camera centre */
static int has_portrait;
static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = (float)W; view_h = (float)H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f; ref_h = (float)H;
        lay_x0 = lay_y0 = 0; lay_w = (float)W; lay_h = (float)H;
        has_portrait = 0;
        return;
    }
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    lay_x0 = x0; lay_y0 = y0; lay_w = x1 - x0; lay_h = y1 - y0;
    /* centre on the output under the middle of the layout (the "front" one) */
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    cam_cx = span[mid][0] + span[mid][2] / 2;
    cam_cy = span[mid][1] + span[mid][3] / 2;
    ref_h = span[mid][3];
    /* the tallest portrait output, if any, is where the tower goes */
    has_portrait = 0;
    float best = 1.15f;
    for (int i = 0; i < span_n; i++)
        if (span[i][3] / span[i][2] > best) {
            best = span[i][3] / span[i][2];
            has_portrait = 1;
            por_x = span[i][0] + span[i][2] / 2 - cam_cx;
            por_y = span[i][1] + span[i][3] * 0.40f - cam_cy;
        }
    view_x = span[span_me][0]; view_y = span[span_me][1];
    view_w = span[span_me][2]; view_h = span[span_me][3];
}

/* aim: a basis that puts the subject at pixel offset (px, py) from the
 * camera centre (y down), focal length f in pixels */
static void aim(v3 pos, v3 look, float px, float py, float f) {
    v3 w = vnorm(vsub(look, pos));
    v3 dc = V3(px / f, -py / f, -1);
    float yaw = atan2f(w.x, -w.z), pitch = asinf(clampf(w.y, -1, 1));
    for (int it = 0; it < 16; it++) {
        v3 fwd = V3(sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch));
        v3 rt = V3(cosf(yaw), 0, sinf(yaw));
        v3 up = vcross(rt, fwd);
        v3 cur = vnorm(vadd(vadd(vscale(rt, dc.x), vscale(up, dc.y)), vscale(fwd, -dc.z)));
        yaw += atan2f(w.x, -w.z) - atan2f(cur.x, -cur.z);
        pitch += asinf(clampf(w.y, -1, 1)) - asinf(clampf(cur.y, -1, 1));
        pitch = clampf(pitch, -1.45f, 1.45f);
    }
    v3 fwd = V3(sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch));
    F.vrt = V3(cosf(yaw), 0, sinf(yaw));
    F.vup = vcross(F.vrt, fwd);
    F.vbk = vscale(fwd, -1);
}

#define ZNEAR 0.12f
#define ZFAR 9000.0f
static m4 proj_view;
static float jit_x, jit_y;            /* this frame's subpixel jitter, in scene pixels */
static int jit_w = 1, jit_h = 1;
static float viewtan_nj[4];
static m4 vp_nj;
static void camera_setup(float t) {
    v3 pos, look, fr;
    float fov;
    tour(t, &pos, &look, &fov, &fr);
    /* the focal length: standalone from the screen; spanning, the wall's
     * width over ~100 degrees, zoomed with the shot */
    float f;
    float zoom = tanf(46 * DEG / 2) / tanf(fov * DEG / 2);
    float aspect = view_w / view_h;
    if (span_me < 0) {
        f = sqrtf((float)W * H) / 2 / tanf(46 * DEG / 2) * zoom;
        /* portrait: the vertical gets the room, so open up a little */
        if (aspect < 1) f *= 0.92f;
    } else {
        float halfw = fmaxf(cam_cx - lay_x0, lay_x0 + lay_w - cam_cx);
        float base = halfw / tanf((opt_fov > 0 ? opt_fov : 100.0f) * DEG / 2);
        f = base * zoom;
    }
    cam_f = f;
    /* where the subject sits on screen */
    float px = fr.x * ref_h / 2, py = -fr.y * ref_h / 2;
    if (span_me < 0) px *= clampf((aspect - 0.7f) / 0.9f, 0, 1);
    if (span_me >= 0 && has_portrait) {
        float k = fr.z;
        px = mixf(px, por_x, k);
        py = mixf(py, por_y, k);
    }
    aim(pos, look, px, py, f);
    F.cam = pos;
    /* this output's slice of the frustum (tangents at unit distance) */
    float l = (view_x - cam_cx) / f, r = (view_x + view_w - cam_cx) / f;
    float b = (cam_cy - view_y - view_h) / f, tp = (cam_cy - view_y) / f;
    float R[9] = {F.vrt.x, F.vrt.y, F.vrt.z, F.vup.x, F.vup.y, F.vup.z, F.vbk.x, F.vbk.y, F.vbk.z};
    memcpy(F.rot3, R, sizeof R);
    m4 V = m4view(F.vrt, F.vup, F.vbk, pos);
    viewtan_nj[0] = l; viewtan_nj[1] = b; viewtan_nj[2] = r - l; viewtan_nj[3] = tp - b;
    vp_nj = m4mul(m4frustum(l * ZNEAR, r * ZNEAR, b * ZNEAR, tp * ZNEAR, ZNEAR, ZFAR), V);
    /* shifting the frustum by +j pixels moves the image by -j */
    float ox = jit_x * (r - l) / jit_w, oy = jit_y * (tp - b) / jit_h;
    l += ox; r += ox; b += oy; tp += oy;
    F.viewtan[0] = l; F.viewtan[1] = b; F.viewtan[2] = r - l; F.viewtan[3] = tp - b;
    m4 P = m4frustum(l * ZNEAR, r * ZNEAR, b * ZNEAR, tp * ZNEAR, ZNEAR, ZFAR);
    proj_view = m4mul(P, V);
}

/* the key light's shadow map: an orthographic box round the tower */
static void shadow_setup(void) {
    v3 L = F.key_dir;
    v3 c = V3(0, 18, 2);
    v3 fwd = vscale(L, -1);
    v3 rt = vnorm(vcross(fwd, fabsf(fwd.y) > 0.99f ? V3(1, 0, 0) : V3(0, 1, 0)));
    v3 up = vcross(rt, fwd);
    float ext = 31;
    /* snap to the texel grid so the edges don't crawl */
    float tx = 2 * ext / SHADOW_N;
    float cr = vdot(c, rt), cu = vdot(c, up);
    c = vadd(c, vadd(vscale(rt, floorf(cr / tx) * tx - cr), vscale(up, floorf(cu / tx) * tx - cu)));
    v3 eye = vadd(c, vscale(L, 120));
    m4 V = m4view(rt, up, vscale(fwd, -1), eye);
    m4 P = m4ortho(-ext, ext, -ext, ext, 1, 260);
    F.shadow_vp = m4mul(P, V);
}

static void set_common(GLuint p) {
    glUseProgram(p);
    glUniform3f(U(p, "uCam"), F.cam.x, F.cam.y, F.cam.z);
    glUniform3f(U(p, "uSunDir"), sky.sun.x, sky.sun.y, sky.sun.z);
    glUniform3f(U(p, "uMoonDir"), sky.moon.x, sky.moon.y, sky.moon.z);
    glUniform3f(U(p, "uKeyDir"), F.key_dir.x, F.key_dir.y, F.key_dir.z);
    glUniform3f(U(p, "uKeyCol"), F.key_col.x, F.key_col.y, F.key_col.z);
    glUniform3f(U(p, "uZen"), F.zen.x, F.zen.y, F.zen.z);
    glUniform3f(U(p, "uHor"), F.hor.x, F.hor.y, F.hor.z);
    glUniform3f(U(p, "uGlow"), F.glow.x, F.glow.y, F.glow.z);
    glUniform3f(U(p, "uAmbSky"), F.amb_sky.x, F.amb_sky.y, F.amb_sky.z);
    glUniform3f(U(p, "uAmbGnd"), F.amb_gnd.x, F.amb_gnd.y, F.amb_gnd.z);
    glUniform3f(U(p, "uFogCol"), F.fog_col.x, F.fog_col.y, F.fog_col.z);
    glUniform1f(U(p, "uFogDen"), F.fog_den);
    glUniform1f(U(p, "uFogH"), F.fog_h);
    glUniform1f(U(p, "uMist"), F.mist);
    glUniform1f(U(p, "uTime"), F.time);
    glUniform1f(U(p, "uTide"), F.tide);
    glUniform1f(U(p, "uWet"), F.wet);
    glUniform1f(U(p, "uCloud"), F.cloud);
    glUniform1f(U(p, "uNight"), F.night);
    glUniform1f(U(p, "uMoonIllum"), F.illum);
    glUniform1f(U(p, "uWaveAmp"), F.wave);
    glUniform2f(U(p, "uWind"), 1.0f, 0.35f);
    glUniform1f(U(p, "uLowQ"), F.lowq);
    glUniform4fv(U(p, "uPulse"), 1, F.pulse);
    glUniformMatrix4fv(U(p, "uShadowVP"), 1, GL_FALSE, F.shadow_vp.m);
    float lp[MAXL][4];
    for (int i = 0; i < nlamp; i++) {
        lp[i][0] = lamp_pos[i].x; lp[i][1] = lamp_pos[i].y; lp[i][2] = lamp_pos[i].z;
        lp[i][3] = i == 0 ? 3.5f : i == 1 ? 1.6f : (i - 2 == 4 ? 0 : 6.0f);
    }
    glUniform4fv(U(p, "uLamp"), nlamp, &lp[0][0]);
    glUniform1i(U(p, "uNLamp"), nlamp);
    glUniform1i(U(p, "uNoise"), 0);
    glUniform1i(U(p, "uShadow"), 1);
    glUniformMatrix4fv(U(p, "uVP"), 1, GL_FALSE, proj_view.m);
    glUniformMatrix4fv(U(p, "uMirror"), 1, GL_FALSE, F.mirror.m);
    glUniform1f(U(p, "uClipY"), F.clip_y);
}

static void bind_common_tex(void) {
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_noise);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_digits);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_hc);
    glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_hf);
    glActiveTexture(GL_TEXTURE0);
}

static void draw_meshes(int with_dyn, int with_terrain, int terr_max) {
    GLuint p = p_mesh;
    set_common(p);
    glUniform4fv(U(p, "uHands"), 4, &F.hands[0][0]);
    glUniform1i(U(p, "uDigits"), 2);
    m4 I = m4id();
    glUniformMatrix4fv(U(p, "uModel"), 1, GL_FALSE, I.m);
    glBindVertexArray(vao_stat);
    glDrawArrays(GL_TRIANGLES, 0, mstat.n);
    m4 cvp = m4mul(proj_view, F.mirror);
    if (with_dyn) {
        glBindVertexArray(vao_dyn);
        for (int i = 0; i < D_N; i++) {
            if (with_dyn == 2 && i != D_BELL && i != D_HAMMER && i != D_ARMIL && i != D_ARMIL2 && i != D_LEVER) continue;
            v3 c = V3(part[i].model.m[12], part[i].model.m[13], part[i].model.m[14]);
            float r = part[i].rad * (i == D_CABLE_G || i == D_CABLE_S ? 6 : 1);
            float bmin[3] = {c.x - r, c.y - r, c.z - r}, bmax[3] = {c.x + r, c.y + r, c.z + r};
            if (!box_visible(bmin, bmax, cvp)) continue;
            glUniformMatrix4fv(U(p, "uModel"), 1, GL_FALSE, part[i].model.m);
            glDrawArrays(GL_TRIANGLES, part[i].first, part[i].count);
        }
    }
    if (with_terrain) {
        set_common(p_terr);
        glUniformMatrix4fv(U(p_terr, "uModel"), 1, GL_FALSE, I.m);
        glBindVertexArray(vao_terr_);
        for (int k = 0; k < nchunk; k++) {
            if (chunks[k].patch > terr_max || !chunks[k].count || !box_visible(chunks[k].bmin, chunks[k].bmax, cvp)) continue;
            glDrawElements(GL_TRIANGLES, chunks[k].count, GL_UNSIGNED_INT, (void *)(sizeof(uint32_t) * (size_t)chunks[k].first));
        }
    }
}

static void draw_birds(void) {
    set_common(p_bird);
    glBindVertexArray(vao_bird);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bird);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 8 * bird_n, bird_inst);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 27, bird_n);
}

static void draw_sky(void) {
    set_common(p_sky);
    glUniform4fv(U(p_sky, "uViewTan"), 1, F.viewtan);
    glUniformMatrix3fv(U(p_sky, "uCamRot"), 1, GL_FALSE, F.rot3);
    glUniform1f(U(p_sky, "uMirrorSky"), F.mirror_sky);
    glUniform1f(U(p_sky, "uSunR"), 0.55f * DEG);
    glUniform1f(U(p_sky, "uMoonR"), 0.9f * DEG);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void fullscreen(void) {
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* --bench: GPU time per pass */
enum { PR_SHADOW, PR_REFL, PR_OPAQUE, PR_SKY, PR_WATER, PR_TAA, PR_VOL, PR_BLOOM, PR_POST, PR_END };
static const char *prof_name[PR_END] = {"shadow", "reflect", "opaque", "sky", "water", "taa", "vol", "bloom", "post"};
static GLuint prof_q[PR_END + 1];
static double prof_ms[PR_END];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[PR_END + 1];
    for (int i = 0; i <= PR_END; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < PR_END; i++) prof_ms[i] += (t[i + 1] - t[i]) * 1e-6;
}

static void render(GLuint target, double t, double wt) {
    setup_view();
    int sw = (int)(W * render_scale + 0.5f), sh = (int)(H * render_scale + 0.5f);
    sw = sw < 16 ? 16 : sw; sh = sh < 16 ? 16 : sh;
    resize_targets(sw, sh);
    sky_compute(wt);
    F.time = (float)fmod(t, 3600.0);
    F.tide = tide_at();
    light_setup();
    clock_update(wt);
    parts_update();
    birds_update(CK.t_birds);
    {
        static const float hal[8][2] = {{0.5f, 0.333f}, {0.25f, 0.667f}, {0.75f, 0.111f}, {0.125f, 0.444f},
                                        {0.625f, 0.778f}, {0.375f, 0.222f}, {0.875f, 0.556f}, {0.0625f, 0.889f}};
        jit_x = hal[frame_no % 8][0] - 0.5f; jit_y = hal[frame_no % 8][1] - 0.5f;
        jit_w = sw; jit_h = sh;
    }
    camera_setup((float)t);
    shadow_setup();
    if (frame_no == 0 && getenv("CLOCKTOWER_DEBUG"))
        fprintf(stderr, "sun alt %.1f az %.0f, moon alt %.1f lit %.2f, tide %.2f m, exposure %.2f, key %.2f %.2f %.2f, zen %.3f %.3f %.3f hor %.3f %.3f %.3f, %s\n",
                sky.sun_alt, atan2(sky.sun.x, -sky.sun.z) / D2R, sky.moon_alt, sky.illum, F.tide, F.exposure, F.key_col.x, F.key_col.y,
                F.key_col.z, F.zen.x, F.zen.y, F.zen.z, F.hor.x, F.hor.y, F.hor.z, weather_name[weather]);
    F.mirror = m4id();
    F.clip_y = -1e6f; F.lowq = 0; F.mirror_sky = 0;
    bind_common_tex();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    prof_mark(PR_SHADOW);
    /* 1. shadows from the key light: every other frame is plenty for what
     * moves in them (the light itself crawls) */
    if (frame_no % 2 == 0 || opt_shot) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, SHADOW_N, SHADOW_N);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.5f, 2.0f);
    glUseProgram(p_depth);
    glUniformMatrix4fv(U(p_depth, "uVP"), 1, GL_FALSE, F.shadow_vp.m);
    m4 I = m4id();
    glUniformMatrix4fv(U(p_depth, "uModel"), 1, GL_FALSE, I.m);
    glBindVertexArray(vao_stat);
    glDrawArrays(GL_TRIANGLES, 0, mstat.n);
    glBindVertexArray(vao_terr_);
    for (int k = 0; k < nchunk; k++)
        if (chunks[k].patch == 0 && chunks[k].count && box_visible(chunks[k].bmin, chunks[k].bmax, F.shadow_vp))
            glDrawElements(GL_TRIANGLES, chunks[k].count, GL_UNSIGNED_INT, (void *)(sizeof(uint32_t) * (size_t)chunks[k].first));
    glBindVertexArray(vao_dyn);
    for (int i = 0; i < D_N; i++) {
        glUniformMatrix4fv(U(p_depth, "uModel"), 1, GL_FALSE, part[i].model.m);
        glDrawArrays(GL_TRIANGLES, part[i].first, part[i].count);
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
    bind_common_tex();
    }

    prof_mark(PR_REFL);
    /* 2. what the water reflects: the world mirrored in the mean surface */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_refl);
    glViewport(0, 0, (sw + 1) / 2, (sh + 1) / 2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {
        m4 M = m4id();
        M.m[5] = -1; M.m[13] = 2 * F.tide;
        F.mirror = M;
        v3 cam = F.cam;
        F.cam = V3(cam.x, 2 * F.tide - cam.y, cam.z);
        F.clip_y = F.tide - 0.05f; F.lowq = 1; F.mirror_sky = 1;
        glEnable(GL_CLIP_DISTANCE0);
        draw_meshes(2, 1, 3);
        draw_birds();
        glDisable(GL_CLIP_DISTANCE0);
        glDepthMask(GL_FALSE);
        draw_sky();
        glDepthMask(GL_TRUE);
        F.cam = cam;
        F.mirror = m4id();
        F.clip_y = -1e6f; F.lowq = 0; F.mirror_sky = 0;
    }

    prof_mark(PR_OPAQUE);
    /* 3. the scene (straight into the textures unless multisampled) */
    glBindFramebuffer(GL_FRAMEBUFFER, opt_msaa > 1 ? fbo_ms : fbo_scene);
    glViewport(0, 0, sw, sh);
    glClear(GL_DEPTH_BUFFER_BIT);
    draw_meshes(1, 1, 3);
    draw_birds();
    prof_mark(PR_SKY);
    glDepthMask(GL_FALSE);
    draw_sky();
    glDepthMask(GL_TRUE);
    prof_mark(PR_WATER);
    /* the sea over it all */
    set_common(p_water);
    glUniform1i(U(p_water, "uRefl"), 5);
    glUniform1i(U(p_water, "uHC"), 3);
    glUniform1i(U(p_water, "uHF"), 4);
    glUniform1i(U(p_water, "uWave"), 8);
    glActiveTexture(GL_TEXTURE8); glBindTexture(GL_TEXTURE_2D, tex_wave);
    glUniform2f(U(p_water, "uScreen"), (float)sw, (float)sh);
    glUniform3f(U(p_water, "uBellPos"), 0, 30.8f, 0);
    glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, tex_refl);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vao_water);
    glDrawArrays(GL_TRIANGLES, 0, water_n);
    glDisable(GL_BLEND);

    /* resolve */
    prof_mark(PR_TAA);
    if (opt_msaa > 1) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_scene);
        glBlitFramebuffer(0, 0, sw, sh, 0, 0, sw, sh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBlitFramebuffer(0, 0, sw, sh, 0, 0, sw, sh, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    }
    glDisable(GL_DEPTH_TEST);

    /* temporal AA into the history */
    {
        int prev = taa_cur;
        taa_cur ^= 1;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_taa[taa_cur]);
        glViewport(0, 0, sw, sh);
        set_common(p_taa);
        glUniform1i(U(p_taa, "uSrc"), 5);
        glUniform1i(U(p_taa, "uDepth"), 6);
        glUniform1i(U(p_taa, "uHistory"), 7);
        glUniform4fv(U(p_taa, "uViewTanNJ"), 1, viewtan_nj);
        glUniformMatrix3fv(U(p_taa, "uCamRot"), 1, GL_FALSE, F.rot3);
        glUniformMatrix4fv(U(p_taa, "uPrevVP"), 1, GL_FALSE, prev_vp.m);
        glUniform2f(U(p_taa, "uNF"), ZNEAR, ZFAR);
        glUniform1f(U(p_taa, "uHistOK"), (float)taa_hist);
        glUniform1f(U(p_taa, "uBlend"), opt_shot ? 0.2f : 0.12f);
        glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, tex_scene);
        glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, tex_depth);
        glActiveTexture(GL_TEXTURE7); glBindTexture(GL_TEXTURE_2D, tex_taa[prev]);
        glActiveTexture(GL_TEXTURE0);
        fullscreen();
        prev_vp = vp_nj;
        taa_hist = 1;
    }

    prof_mark(PR_VOL);
    /* 4. light in the air, at quarter res */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_vol);
    glViewport(0, 0, (sw + 3) / 4, (sh + 3) / 4);
    F.lowq = 1;
    set_common(p_vol);
    F.lowq = 0;
    glUniform1i(U(p_vol, "uDepth"), 6);
    glUniform4fv(U(p_vol, "uViewTan"), 1, F.viewtan);
    glUniformMatrix3fv(U(p_vol, "uCamRot"), 1, GL_FALSE, F.rot3);
    glUniform2f(U(p_vol, "uNF"), ZNEAR, ZFAR);
    glUniform1f(U(p_vol, "uFrame"), 0.0f);
    glUniform3f(U(p_vol, "uBellPos"), 0, 30.8f, 0);
    glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, tex_depth);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();

    prof_mark(PR_BLOOM);
    /* 5. bloom */
    GLuint src = tex_taa[taa_cur];
    int srcw = sw, srch = sh;
    for (int i = 0; i < NBLOOM; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i]);
        glViewport(0, 0, bloom_w[i], bloom_h[i]);
        glUseProgram(p_down);
        glUniform1i(U(p_down, "uSrc"), 7);
        glUniform2f(U(p_down, "uTexel"), 0.5f / srcw, 0.5f / srch);
        glUniform1f(U(p_down, "uFirst"), i == 0);
        glActiveTexture(GL_TEXTURE7); glBindTexture(GL_TEXTURE_2D, src);
        fullscreen();
        src = tex_bloom[i]; srcw = bloom_w[i]; srch = bloom_h[i];
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    for (int i = NBLOOM - 1; i > 0; i--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i - 1]);
        glViewport(0, 0, bloom_w[i - 1], bloom_h[i - 1]);
        glUseProgram(p_up);
        glUniform1i(U(p_up, "uSrc"), 7);
        glUniform2f(U(p_up, "uTexel"), 1.0f / bloom_w[i], 1.0f / bloom_h[i]);
        glActiveTexture(GL_TEXTURE7); glBindTexture(GL_TEXTURE_2D, tex_bloom[i]);
        fullscreen();
    }
    glDisable(GL_BLEND);

    prof_mark(PR_POST);
    /* 6. the grade, at the output's size */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    set_common(p_post);
    glUniform1i(U(p_post, "uScene"), 5);
    glUniform1i(U(p_post, "uBloom"), 6);
    glUniform1i(U(p_post, "uVol"), 7);
    glUniform1f(U(p_post, "uExposure"), F.exposure);
    glUniform1f(U(p_post, "uBloomAmt"), 0.06f);
    glUniform1f(U(p_post, "uRain"), WP.rain);
    glUniform4f(U(p_post, "uVig"), (view_x - lay_x0) / lay_w, 1 - (view_y + view_h - lay_y0) / lay_h, view_w / lay_w, view_h / lay_h);
    glUniform2f(U(p_post, "uLayoutPx"), lay_w, lay_h);
    glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, tex_taa[taa_cur]);
    glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, tex_bloom[0]);
    glActiveTexture(GL_TEXTURE7); glBindTexture(GL_TEXTURE_2D, tex_vol);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    prof_mark(PR_END);
    frame_no++;
}

/* ---- --layer: a native Wayland wallpaper (wlr-layer-shell + EGL) ---- */
static struct {
    struct wl_display *d;
    struct wl_compositor *comp;
    struct zwlr_layer_shell_v1 *shell;
    struct wl_surface *surf;
    struct zwlr_layer_surface_v1 *ls;
    struct wl_egl_window *ew;
    EGLDisplay ed;
    EGLSurface es;
    int configured, closed;
} wl;

static void wl_global(void *data, struct wl_registry *r, uint32_t name, const char *intf, uint32_t ver) {
    (void)data;
    if (!strcmp(intf, wl_compositor_interface.name))
        wl.comp = wl_registry_bind(r, name, &wl_compositor_interface, ver < 4 ? ver : 4);
    else if (!strcmp(intf, zwlr_layer_shell_v1_interface.name))
        wl.shell = wl_registry_bind(r, name, &zwlr_layer_shell_v1_interface, ver < 4 ? ver : 4);
}
static void wl_global_remove(void *data, struct wl_registry *r, uint32_t name) { (void)data; (void)r; (void)name; }
static const struct wl_registry_listener wl_reg_l = {wl_global, wl_global_remove};

static void ls_configure(void *data, struct zwlr_layer_surface_v1 *ls, uint32_t serial, uint32_t w, uint32_t h) {
    (void)data;
    zwlr_layer_surface_v1_ack_configure(ls, serial);
    if (w && h && ((int)w != W || (int)h != H)) {
        W = (int)w; H = (int)h;
        if (wl.ew) wl_egl_window_resize(wl.ew, W, H, 0, 0);
    }
    wl.configured = 1;
}
static void ls_closed(void *data, struct zwlr_layer_surface_v1 *ls) { (void)data; (void)ls; wl.closed = 1; }
static const struct zwlr_layer_surface_v1_listener ls_l = {ls_configure, ls_closed};

static void wl_init(void) {
    if (!(wl.d = wl_display_connect(NULL))) die("can't connect to the Wayland display");
    struct wl_registry *reg = wl_display_get_registry(wl.d);
    wl_registry_add_listener(reg, &wl_reg_l, NULL);
    wl_display_roundtrip(wl.d);
    if (!wl.comp || !wl.shell) die("the compositor has no wlr-layer-shell");
    wl.surf = wl_compositor_create_surface(wl.comp);
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "clocktower");
    zwlr_layer_surface_v1_add_listener(wl.ls, &ls_l, NULL);
    zwlr_layer_surface_v1_set_anchor(wl.ls, ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP | ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM |
                                                ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT | ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
    zwlr_layer_surface_v1_set_exclusive_zone(wl.ls, -1);
    zwlr_layer_surface_v1_set_size(wl.ls, 0, 0);
    struct wl_region *none = wl_compositor_create_region(wl.comp);
    wl_surface_set_input_region(wl.surf, none);
    wl_region_destroy(none);
    wl_surface_commit(wl.surf);
    while (!wl.configured && !wl.closed && wl_display_dispatch(wl.d) >= 0) {}
    if (!wl.configured) die("the layer surface was never configured");

    PFNEGLGETPLATFORMDISPLAYEXTPROC get_disp = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    wl.ed = get_disp ? get_disp(EGL_PLATFORM_WAYLAND_KHR, wl.d, NULL) : eglGetDisplay((EGLNativeDisplayType)wl.d);
    if (!eglInitialize(wl.ed, NULL, NULL)) die("eglInitialize failed");
    eglBindAPI(EGL_OPENGL_API);
    static const EGLint cfg_attr[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 0, EGL_NONE};
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(wl.ed, cfg_attr, &cfg, 1, &n) || n < 1) die("no suitable EGL config");
    static const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
                                      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(wl.ed, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) die("can't create a GL 3.3 context");
    wl.ew = wl_egl_window_create(wl.surf, W, H);
    wl.es = eglCreateWindowSurface(wl.ed, cfg, (EGLNativeWindowType)wl.ew, NULL);
    if (wl.es == EGL_NO_SURFACE || !eglMakeCurrent(wl.ed, wl.es, wl.es, ctx)) die("can't make the EGL surface current");
    eglSwapInterval(wl.ed, 1);
}

static int wl_poll(void) {
    while (wl_display_prepare_read(wl.d) != 0) wl_display_dispatch_pending(wl.d);
    wl_display_flush(wl.d);
    struct pollfd pf = {wl_display_get_fd(wl.d), POLLIN, 0};
    if (poll(&pf, 1, 0) > 0) wl_display_read_events(wl.d);
    else wl_display_cancel_read(wl.d);
    if (wl_display_dispatch_pending(wl.d) < 0) return 0;
    return !wl.closed;
}

static void usage(void) {
    fprintf(stderr,
        "usage: clocktower [--window | --root | --layer] [--seed N]\n"
        "                  [--weather overcast|clear|mist|rain|random] [--lat DEG --lon DEG]\n"
        "                  [--time 'YYYY-MM-DD HH:MM[:SS]'] [--tide METRES] [--font FAMILY]\n"
        "                  [--scale X] [--budget MS] [--msaa N] [--fov DEG] [--shotcam K]\n"
        "                  [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                  [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
    exit(1);
}

/* "x,y,w,h;x,y,w,h;..." */
static void parse_layout(const char *str) {
    for (span_n = 0; span_n < MAX_OUT; span_n++) {
        float *r = span[span_n];
        int used = 0;
        if (sscanf(str, "%f,%f,%f,%f%n", &r[0], &r[1], &r[2], &r[3], &used) != 4 || r[2] <= 0 || r[3] <= 0)
            break;
        str += used;
        if (*str == ';') str++;
    }
}

static void save_ppm(const char *path, GLuint fbo) {
    unsigned char *px = malloc((size_t)W * H * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; y--) fwrite(px + (size_t)y * W * 3, 1, (size_t)W * 3, f);
    fclose(f);
    free(px);
}

/* --time: the local time to pretend it is */
static void parse_time(const char *s) {
    struct tm tm = {0};
    int sec = 0;
    if (sscanf(s, "%d-%d-%d %d:%d:%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &sec) < 5) usage();
    tm.tm_year -= 1900; tm.tm_mon -= 1; tm.tm_sec = sec; tm.tm_isdst = -1;
    time_t t = mktime(&tm);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    time_offset = (double)t - (ts.tv_sec + ts.tv_nsec * 1e-9);
}

int main(int argc, char **argv) {
    const char *opt_time = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-weather") && i + 1 < argc) opt_weather = argv[++i];
        else if (!strcmp(a, "-lat") && i + 1 < argc) { opt_lat = strtod(argv[++i], NULL); loc_given = 1; }
        else if (!strcmp(a, "-lon") && i + 1 < argc) { opt_lon = strtod(argv[++i], NULL); loc_given = 1; }
        else if (!strcmp(a, "-time") && i + 1 < argc) opt_time = argv[++i];
        else if (!strcmp(a, "-tide") && i + 1 < argc) opt_tide = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-budget") && i + 1 < argc) opt_budget = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-msaa") && i + 1 < argc) opt_msaa = atoi(argv[++i]);
        else if (!strcmp(a, "-fov") && i + 1 < argc) opt_fov = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-shotcam") && i + 1 < argc) opt_shotcam = atoi(argv[++i]);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (opt_msaa < 1) opt_msaa = 1;
    if (opt_time) parse_time(opt_time);
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    load_location();
    choose_weather();
    int offscreen = opt_shot || opt_bench > 0;

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offscreen) wl_init();
    else {
        dpy = XOpenDisplay(NULL);
        if (!dpy) die("can't open display");
        int scr = DefaultScreen(dpy);
        root = RootWindow(dpy, scr);
        int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
        XVisualInfo *vi = glXChooseVisual(dpy, scr, attrs);
        if (!vi) die("no suitable GLX visual");
        XWindowAttributes ra;
        XGetWindowAttributes(dpy, root, &ra);
        XSetWindowAttributes swa = {0};
        swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
        swa.event_mask = StructureNotifyMask | KeyPressMask | ExposureMask;
        swa.override_redirect = opt_root;
        if (opt_root) { W = ra.width; H = ra.height; XSelectInput(dpy, root, StructureNotifyMask); }
        if (offscreen) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Clocktower");
        XClassHint chint = {"clocktower", "Clocktower"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    /* --epoch: when the tour started, shared by every instance; a --shot
     * shows the moment --at seconds after it (or after now) */
    double epoch = opt_epoch ? opt_epoch : wall();
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)(offscreen && !opt_epoch ? epoch - opt_at - 2400 : epoch);

    if (offscreen) {
        GLuint tex = mk_tex(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
        GLuint fbo = mk_fbo(tex);
        render_scale = opt_scale > 0 ? opt_scale : opt_shot ? 1.0f : 0.6f;
        double wt = opt_epoch ? epoch + opt_at : epoch;
        render(fbo, opt_at, wt);
        glFinish();
        if (opt_shot) {
            for (int i = 0; i < 15; i++) render(fbo, opt_at, wt);
            save_ppm(opt_shot, fbo);
            return 0;
        }
        double sum = 0, worst = 0;
        glGenQueries(PR_END + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            render(fbo, opt_at + i / 60.0, wt + i / 60.0);
            glFinish();
            prof_collect();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("clocktower: %dx%d scale %.2f (%dx%d, %dx MSAA), %d frames: avg %.2f ms, worst %.2f ms (%s, %s)\n", W, H,
               render_scale, sc_w, sc_h, opt_msaa, opt_bench, sum / opt_bench, worst, weather_name[weather],
               (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < PR_END; i++) printf("  %-8s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        return 0;
    }

    render_scale = opt_scale > 0 ? opt_scale : 0.6f;
    double avg = opt_budget / 1000, last_change = 0;
    for (int frame = 0;; frame++) {
        if (opt_layer && !wl_poll()) return 0;
        while (dpy && XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win, ev.xconfigure.width, ev.xconfigure.height);
                else { W = ev.xconfigure.width; H = ev.xconfigure.height; }
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        double t = now(), wt = wall();
        render(0, wt - epoch + opt_at, wt);
        glFinish();
        double took = now() - t;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        /* dynamic resolution: keep each frame inside the budget */
        if (frame > 10) avg += (took - avg) * 0.1;
        if (opt_scale <= 0 && frame > 30 && t - last_change > 0.5) {
            float s = render_scale, b = opt_budget / 1000;
            if (avg > b * 1.1f) s *= 0.92f;
            else if (avg < b * 0.75f) s *= 1.05f;
            s = clampf(s, 0.3f, 1.0f);
            if (fabsf(s - render_scale) > 0.02f) { render_scale = s; last_change = t; avg = b; }
        }
        double spare = 1.0 / 30 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
