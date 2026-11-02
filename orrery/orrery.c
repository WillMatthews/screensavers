/*
 * orrery — a grand mechanical orrery on a dark table in a dim study, with
 * the planets where they really are.
 *
 * Mercury, Venus, the Earth and its Moon, Mars, Jupiter (banded, with its
 * four Galilean moons on wires) and Saturn (ringed, the rings held in their
 * true orientation) ride on arms from six concentric tubes round the Sun: a
 * frosted glass globe with a lamp inside, which lights the whole mechanism
 * and throws the arms' shadows across the table. The planets' heliocentric
 * longitudes come from the J2000 Keplerian elements with secular rates
 * (Standish, JPL; good to well under a degree this century), precessed to
 * the equinox of date; the Moon from the main terms of its theory, so its
 * phase as the lamp lights it is the real one; the Earth's axis keeps its
 * 23.44 degree tilt fixed in space.
 *
 * Under a silvered dial plate engraved with the ecliptic (degrees and the
 * signs of the zodiac) the gear train is real: a drive arbor turning once a
 * year carries a wheel for each planet, meshing a wheel on its tube:
 *   Mercury 83:20  Venus 65:40  Earth 50:50  Mars 33:62  Jupiter 7:83
 *   Saturn 10:40 then 11:81 through an idler (0.033951 of a year's turn,
 *   against the true 0.033948)
 * Every wheel turns as the model advances, each meshed with its partner.
 * The drive arbor comes up through the dial plate as the hand of a calendar
 * subdial (days and months engraved where the Earth really is on each date),
 * and a date window shows the model's date.
 *
 * Mostly the model holds the real present; now and then it is wound forward
 * (eased, up to tens of days a second) by weeks to a year and a half, waits,
 * and runs back to now. A slow cinematic camera drifts round it, shot by
 * shot (wide views across the table, the train meshing, a planet sweeping
 * past the lens, the Earth and Moon, Saturn, the dial, the Sun, the plaque),
 * with dips to black between. On the table in front, an engraved brass
 * plaque gives the hostname, the date and time, and how long the screen has
 * been locked (swaylock's start time, from /proc).
 *
 * Rendering: procedural meshes rasterised with 4x MSAA at a reduced internal
 * resolution (upscaled Catmull-Rom; live, it follows a GPU-time budget),
 * anisotropic GGX for turned and grained brass, clear-coated enamel planets,
 * engraving from a freetype glyph atlas rendered into textures, the Sun as a
 * spherical area light with analytic soft shadows from the arms, posts and
 * planets, cool moonlight from the window with a shadow map, depth of field,
 * bloom and a filmic grade.
 *
 * Same plumbing as gauges and clockwork: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] for one frame offscreen and
 * --bench N for timing. --ephem prints the model's positions and exits.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * As in flightsim they all fly one virtual camera, centred on the output
 * under the middle of the layout and framed for it; each renders its own
 * slice of that camera's view, so the scene is continuous across the wall.
 * The orrery's clock and the camera both run off the shared epoch.
 *
 * Keys (window mode): q / Escape quit.
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
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

#define PI 3.14159265358979323846
#define TAU (2 * PI)
#define RAD (PI / 180)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_ephem;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 0;                  /* seconds after the epoch */
static int off_w = 1920, off_h = 1080;
static float opt_scale = 0;                /* render scale; 0 = adapt to the frame budget */
static double opt_epoch;                   /* CLOCK_REALTIME seconds the show starts at */
static double opt_date;                    /* --date: pretend "now" is this unix time */
static int opt_shot_type = -1;             /* --view: hold one kind of shot */
static const char *opt_font = "C059";

/* ---- spanning: one virtual camera across every output ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];             /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "orrery: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}

/* a stateless random number in [0, 1) from the seed and two keys, so every
 * instance (and every moment) agrees without keeping state */
static float h01(uint32_t a, uint32_t b) {
    uint32_t h = opt_seed * 0x9e3779b1u ^ a * 0x85ebca6bu ^ b * 0xc2b2ae35u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return (h >> 8) * (1.0f / 16777216.0f);
}

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

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

/* ---- 4x4 matrices, column-major ---- */
typedef struct { float m[16]; } M4;

static M4 m4_id(void) { M4 r = {{0}}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static M4 m4_mul(M4 a, M4 b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 4; i++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + i] * b.m[c * 4 + k];
            r.m[c * 4 + i] = s;
        }
    return r;
}
static M4 m4_trans(float x, float y, float z) { M4 r = m4_id(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static M4 m4_scale(float x, float y, float z) { M4 r = m4_id(); r.m[0] = x; r.m[5] = y; r.m[10] = z; return r; }
static M4 m4_rotz(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }
static M4 m4_rotx(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
static M4 m4_frustum(float l, float r_, float b, float t, float n, float f) {
    M4 r = {{0}};
    r.m[0] = 2 * n / (r_ - l); r.m[5] = 2 * n / (t - b);
    r.m[8] = (r_ + l) / (r_ - l); r.m[9] = (t + b) / (t - b); r.m[10] = -(f + n) / (f - n); r.m[11] = -1;
    r.m[14] = -2 * f * n / (f - n);
    return r;
}
static M4 m4_ortho(float l, float r_, float b, float t, float n, float f) {
    M4 r = m4_id();
    r.m[0] = 2 / (r_ - l); r.m[5] = 2 / (t - b); r.m[10] = -2 / (f - n);
    r.m[12] = -(r_ + l) / (r_ - l); r.m[13] = -(t + b) / (t - b); r.m[14] = -(f + n) / (f - n);
    return r;
}
static void v3norm(float *v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; } }
static void v3cross(const float *a, const float *b, float *o) {
    o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
static M4 m4_look(const float *e, const float *at, const float *up) {
    float f[3] = {at[0] - e[0], at[1] - e[1], at[2] - e[2]};
    v3norm(f);
    float s[3], u[3];
    v3cross(f, up, s);
    v3norm(s);
    v3cross(s, f, u);
    M4 r = m4_id();
    r.m[0] = s[0]; r.m[4] = s[1]; r.m[8] = s[2];
    r.m[1] = u[0]; r.m[5] = u[1]; r.m[9] = u[2];
    r.m[2] = -f[0]; r.m[6] = -f[1]; r.m[10] = -f[2];
    r.m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
    r.m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
    r.m[14] = f[0] * e[0] + f[1] * e[1] + f[2] * e[2];
    return r;
}
/* a frame whose z axis points along d, at p */
static M4 m4_along(const float *p, const float *d) {
    float z[3] = {d[0], d[1], d[2]};
    v3norm(z);
    float a[3] = {0, 0, 1};
    if (fabsf(z[2]) > 0.9f) { a[0] = 1; a[2] = 0; }
    float x[3], y[3];
    v3cross(a, z, x);
    v3norm(x);
    v3cross(z, x, y);
    M4 r = m4_id();
    r.m[0] = x[0]; r.m[1] = x[1]; r.m[2] = x[2];
    r.m[4] = y[0]; r.m[5] = y[1]; r.m[6] = y[2];
    r.m[8] = z[0]; r.m[9] = z[1]; r.m[10] = z[2];
    r.m[12] = p[0]; r.m[13] = p[1]; r.m[14] = p[2];
    return r;
}

/* ---- the sky ----
 * Heliocentric ecliptic positions from the J2000 Keplerian elements and
 * their rates per Julian century (E. M. Standish, "Keplerian Elements for
 * Approximate Positions of the Major Planets", JPL; table 1, 1800-2050):
 * a (au), e, I, L, long. of perihelion, long. of the node (degrees). */
enum { P_MERCURY, P_VENUS, P_EARTH, P_MARS, P_JUPITER, P_SATURN, NPL };
static const char *PL_NAME[NPL] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"};
static const double ELEM[NPL][12] = {
    {0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
     252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081},
    {0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
     181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418},
    {1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
     100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0},
    {1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
     -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343},
    {5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
     34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106},
    {9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
     49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794},
};

static double jd_of(double unix_t) { return unix_t / 86400.0 + 2440587.5; }
static double wrap360(double a) { a = fmod(a, 360.0); return a < 0 ? a + 360 : a; }

/* heliocentric ecliptic longitude and latitude (degrees, equinox of date)
 * and distance (au) of planet p at Julian date jd */
static void helio(int p, double jd, double *lon, double *lat, double *dist) {
    const double *E = ELEM[p];
    double T = (jd - 2451545.0) / 36525.0;
    double a = E[0] + E[1] * T, e = E[2] + E[3] * T, I = (E[4] + E[5] * T) * RAD;
    double L = E[6] + E[7] * T, wb = E[8] + E[9] * T, Om = (E[10] + E[11] * T) * RAD;
    double M = wrap360(L - wb) * RAD, w = (wb * RAD) - Om;
    if (M > PI) M -= TAU;
    double Ea = M + e * sin(M);
    for (int i = 0; i < 8; i++) Ea -= (Ea - e * sin(Ea) - M) / (1 - e * cos(Ea));
    double xp = a * (cos(Ea) - e), yp = a * sqrt(1 - e * e) * sin(Ea);
    double cw = cos(w), sw = sin(w), cO = cos(Om), sO = sin(Om), cI = cos(I), sI = sin(I);
    double x = (cw * cO - sw * sO * cI) * xp + (-sw * cO - cw * sO * cI) * yp;
    double y = (cw * sO + sw * cO * cI) * xp + (-sw * sO + cw * cO * cI) * yp;
    double z = (sw * sI) * xp + (cw * sI) * yp;
    /* J2000 -> equinox of date: general precession in longitude */
    *lon = wrap360(atan2(y, x) / RAD + 1.396971 * T);
    *lat = atan2(z, sqrt(x * x + y * y)) / RAD;
    *dist = sqrt(x * x + y * y + z * z);
}

/* the Moon's geocentric ecliptic longitude and latitude (degrees, of date):
 * the largest periodic terms of its theory (Meeus ch. 47), to ~0.3 deg */
static void moon_pos(double jd, double *lon, double *lat) {
    double d = jd - 2451545.0;
    double Lp = 218.3164477 + 13.17639648 * d, D = (297.8501921 + 12.19074912 * d) * RAD;
    double M = (357.5291092 + 0.98560028 * d) * RAD, Mp = (134.9633964 + 13.06499295 * d) * RAD;
    double F = (93.2720950 + 13.22935024 * d) * RAD;
    *lon = wrap360(Lp + 6.288774 * sin(Mp) + 1.274027 * sin(2 * D - Mp) + 0.658314 * sin(2 * D) + 0.213618 * sin(2 * Mp) -
                   0.185116 * sin(M) - 0.114332 * sin(2 * F) + 0.058793 * sin(2 * D - 2 * Mp) +
                   0.057066 * sin(2 * D - M - Mp) + 0.053322 * sin(2 * D + Mp) + 0.045758 * sin(2 * D - M) -
                   0.040923 * sin(M - Mp) - 0.034720 * sin(D) - 0.030383 * sin(M + Mp));
    *lat = 5.128122 * sin(F) + 0.280602 * sin(Mp + F) + 0.277693 * sin(Mp - F) + 0.173237 * sin(2 * D - F);
}

/* the Galilean moons' longitudes round Jupiter (degrees, ecliptic, seen from
 * Jupiter; 0 = away from the Sun), from their mean motions (Meeus ch. 44,
 * without the light-time and the Earth's point of view) */
static void jupiter_moons(double jd, double out[4]) {
    double d = jd - 2451545.0;
    double N = (20.020 + 0.0830853 * d) * RAD;
    double B = 5.555 * sin(N) + 0.168 * sin(2 * N);   /* Jupiter's equation of the centre */
    static const double u0[4] = {163.8069, 358.4140, 5.7176, 224.8092};
    static const double un[4] = {203.4058646, 101.2916335, 50.2345180, 21.4879800};
    for (int i = 0; i < 4; i++) out[i] = wrap360(u0[i] + un[i] * d - B);
}

static double sun_lon_geo(double jd) {
    double l, b, r;
    helio(P_EARTH, jd, &l, &b, &r);
    return wrap360(l + 180);
}

/* ---- mesh building ----
 * One vertex buffer for everything. A vertex carries its material, a
 * tangent for the grain (round the local z axis of the part being built:
 * turned and grained parts are), and the part's own x, y for engraving. */
enum { M_BRASS, M_BRASS_DARK, M_STEEL, M_BLUED, M_GILT, M_DIAL, M_MAHOGANY, M_TABLE, M_ROOM, M_SUN,
       M_MERCURY, M_VENUS, M_EARTH, M_MARS, M_JUPITER, M_SATURN, M_MOON, M_RING, M_PLAQUE, M_IVORY, M_BLACK, M_IVORY_BALL };

typedef struct { float p[3], n[3], t[3], m, u, v; } Vtx;
static Vtx *vb;
static int nv, cv;
static M4 xf;                              /* applied to everything emitted */
static int xf_on;
static float cur_mat;

static void vtx(float x, float y, float z, float nx, float ny, float nz) {
    if (nv == cv) vb = realloc(vb, (size_t)(cv = cv ? cv * 2 : 65536) * sizeof *vb);
    Vtx *v = &vb[nv++];
    float tx = -y, ty = x, tl = sqrtf(tx * tx + ty * ty);
    if (tl < 1e-5f) { tx = 1; ty = 0; } else { tx /= tl; ty /= tl; }
    if (xf_on) {
        const float *m = xf.m;
        v->p[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
        v->p[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        v->p[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        v->n[0] = m[0] * nx + m[4] * ny + m[8] * nz;
        v->n[1] = m[1] * nx + m[5] * ny + m[9] * nz;
        v->n[2] = m[2] * nx + m[6] * ny + m[10] * nz;
        v->t[0] = m[0] * tx + m[4] * ty;
        v->t[1] = m[1] * tx + m[5] * ty;
        v->t[2] = m[2] * tx + m[6] * ty;
    } else {
        v->p[0] = x; v->p[1] = y; v->p[2] = z;
        v->n[0] = nx; v->n[1] = ny; v->n[2] = nz;
        v->t[0] = tx; v->t[1] = ty; v->t[2] = 0;
    }
    v->m = cur_mat;
    v->u = x; v->v = y;
}

/* a quad a b c d (as two triangles) with a normal per corner */
static void quad(const float *a, const float *b, const float *c, const float *d,
                 const float *na, const float *nb, const float *nc, const float *nd) {
    vtx(a[0], a[1], a[2], na[0], na[1], na[2]);
    vtx(b[0], b[1], b[2], nb[0], nb[1], nb[2]);
    vtx(c[0], c[1], c[2], nc[0], nc[1], nc[2]);
    vtx(a[0], a[1], a[2], na[0], na[1], na[2]);
    vtx(c[0], c[1], c[2], nc[0], nc[1], nc[2]);
    vtx(d[0], d[1], d[2], nd[0], nd[1], nd[2]);
}

#define MAXC 8192
static float cA[2 * MAXC], cB[2 * MAXC];   /* contour scratch */

/* outward segment normals of a closed CCW contour, and per-vertex normals
 * for the two ends of each segment: smoothed across gentle bends, kept
 * sharp at corners (tooth tips and roots) */
typedef struct { float sn[2 * MAXC], mn[2 * MAXC], ea[2 * MAXC], eb[2 * MAXC]; } Edges;
static Edges eo, ei;

static void edges(const float *c, int n, Edges *e) {
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float dx = c[2 * j] - c[2 * i], dy = c[2 * j + 1] - c[2 * i + 1], l = sqrtf(dx * dx + dy * dy);
        if (l < 1e-6f) { e->sn[2 * i] = 0; e->sn[2 * i + 1] = 0; continue; }
        e->sn[2 * i] = dy / l; e->sn[2 * i + 1] = -dx / l;
    }
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < n; i++)
            if (e->sn[2 * i] == 0 && e->sn[2 * i + 1] == 0) {
                int p = (i + n - 1) % n;
                e->sn[2 * i] = e->sn[2 * p]; e->sn[2 * i + 1] = e->sn[2 * p + 1];
            }
    for (int i = 0; i < n; i++) {
        int p = (i + n - 1) % n;
        float mx = e->sn[2 * p] + e->sn[2 * i], my = e->sn[2 * p + 1] + e->sn[2 * i + 1], l = sqrtf(mx * mx + my * my);
        if (l < 1e-6f) { mx = e->sn[2 * i]; my = e->sn[2 * i + 1]; l = 1; }
        e->mn[2 * i] = mx / l; e->mn[2 * i + 1] = my / l;
    }
    for (int i = 0; i < n; i++) {
        int p = (i + n - 1) % n, j = (i + 1) % n;
        float d0 = e->sn[2 * p] * e->sn[2 * i] + e->sn[2 * p + 1] * e->sn[2 * i + 1];
        float d1 = e->sn[2 * i] * e->sn[2 * j] + e->sn[2 * i + 1] * e->sn[2 * j + 1];
        e->ea[2 * i] = d0 > 0.82f ? e->mn[2 * i] : e->sn[2 * i];
        e->ea[2 * i + 1] = d0 > 0.82f ? e->mn[2 * i + 1] : e->sn[2 * i + 1];
        e->eb[2 * i] = d1 > 0.82f ? e->mn[2 * j] : e->sn[2 * i];
        e->eb[2 * i + 1] = d1 > 0.82f ? e->mn[2 * j + 1] : e->sn[2 * i + 1];
    }
}

/* Extrude a closed CCW contour o (n points) from z0 to z1. With an inner
 * contour `in` (same count, point i at the same angle as o[i]) it's a ring;
 * without, the caps fan from the centroid (the shape must be star-shaped
 * about it). bev > 0 cuts a 45-degree chamfer round the top edges; bottom
 * closes the underside. */
static void extrude(const float *o, const float *in, int n, float z0, float z1, float bev, int bottom) {
    static float oi[2 * MAXC], ii[2 * MAXC];
    edges(o, n, &eo);
    if (in) edges(in, n, &ei);
    float zt = z1 - bev;
    for (int i = 0; i < n; i++) {
        float k = bev / fmaxf(eo.mn[2 * i] * eo.sn[2 * i] + eo.mn[2 * i + 1] * eo.sn[2 * i + 1], 0.5f);
        oi[2 * i] = o[2 * i] - eo.mn[2 * i] * k;
        oi[2 * i + 1] = o[2 * i + 1] - eo.mn[2 * i + 1] * k;
        if (in) {
            float k2 = bev / fmaxf(ei.mn[2 * i] * ei.sn[2 * i] + ei.mn[2 * i + 1] * ei.sn[2 * i + 1], 0.5f);
            ii[2 * i] = in[2 * i] + ei.mn[2 * i] * k2;
            ii[2 * i + 1] = in[2 * i + 1] + ei.mn[2 * i + 1] * k2;
        }
    }
    float cx = 0, cy = 0;
    for (int i = 0; i < n; i++) { cx += o[2 * i]; cy += o[2 * i + 1]; }
    cx /= n; cy /= n;
    const float up[3] = {0, 0, 1}, dn[3] = {0, 0, -1};
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float a0[3] = {o[2 * i], o[2 * i + 1], z0}, b0[3] = {o[2 * j], o[2 * j + 1], z0};
        float a1[3] = {o[2 * i], o[2 * i + 1], zt}, b1[3] = {o[2 * j], o[2 * j + 1], zt};
        float na[3] = {eo.ea[2 * i], eo.ea[2 * i + 1], 0}, nb[3] = {eo.eb[2 * i], eo.eb[2 * i + 1], 0};
        if (zt > z0) quad(a0, b0, b1, a1, na, nb, nb, na);
        float ta[3] = {oi[2 * i], oi[2 * i + 1], z1}, tb[3] = {oi[2 * j], oi[2 * j + 1], z1};
        if (bev > 0) {
            float ca[3] = {na[0], na[1], 1}, cb[3] = {nb[0], nb[1], 1};
            v3norm(ca); v3norm(cb);
            quad(a1, b1, tb, ta, ca, cb, cb, ca);
        }
        if (in) {
            float p0[3] = {in[2 * i], in[2 * i + 1], z0}, q0[3] = {in[2 * j], in[2 * j + 1], z0};
            float p1[3] = {in[2 * i], in[2 * i + 1], zt}, q1[3] = {in[2 * j], in[2 * j + 1], zt};
            float pa[3] = {-ei.ea[2 * i], -ei.ea[2 * i + 1], 0}, pb[3] = {-ei.eb[2 * i], -ei.eb[2 * i + 1], 0};
            if (zt > z0) quad(q0, p0, p1, q1, pb, pa, pa, pb);
            float ua[3] = {ii[2 * i], ii[2 * i + 1], z1}, ub[3] = {ii[2 * j], ii[2 * j + 1], z1};
            if (bev > 0) {
                float ca[3] = {pa[0], pa[1], 1}, cb[3] = {pb[0], pb[1], 1};
                v3norm(ca); v3norm(cb);
                quad(q1, p1, ua, ub, cb, ca, ca, cb);
            }
            quad(ua, ta, tb, ub, up, up, up, up);
            if (bottom) quad(p0, q0, b0, a0, dn, dn, dn, dn);
        } else {
            vtx(cx, cy, z1, 0, 0, 1);
            vtx(ta[0], ta[1], ta[2], 0, 0, 1);
            vtx(tb[0], tb[1], tb[2], 0, 0, 1);
            if (bottom) {
                vtx(cx, cy, z0, 0, 0, -1);
                vtx(b0[0], b0[1], b0[2], 0, 0, -1);
                vtx(a0[0], a0[1], a0[2], 0, 0, -1);
            }
        }
    }
}

/* ---- contours ---- */
static int seg_for(float r) { int n = (int)(r * 20); return n < 16 ? 16 : n > 160 ? 160 : n; }

static int circle_c(float *o, float cx, float cy, float r, int n) {
    for (int i = 0; i < n; i++) {
        double a = TAU * i / n;
        o[2 * i] = cx + r * (float)cos(a);
        o[2 * i + 1] = cy + r * (float)sin(a);
    }
    return n;
}

static void match_circle(const float *o, int n, float cx, float cy, float r, float *in) {
    for (int i = 0; i < n; i++) {
        float dx = o[2 * i] - cx, dy = o[2 * i + 1] - cy, l = sqrtf(dx * dx + dy * dy);
        in[2 * i] = cx + dx / l * r;
        in[2 * i + 1] = cy + dy / l * r;
    }
}

/* A wheel or pinion, tooth 0 centred on angle `rot`: radial flanks below
 * the pitch circle, an ogival addendum above, small root fillets. ha / hf:
 * addendum / dedendum in modules; thick: tooth thickness at the pitch
 * circle as a fraction of half the pitch. */
static int gear_c(float *o, int N, float m, float ha, float hf, float thick, float rot) {
    float rp = m * N * 0.5f, rf = rp - hf * m, Ht = ha * m, pa = (float)TAU / N;
    float hw = thick * (float)PI * m * 0.25f, af = hw / rp;
    float rfil = fminf(0.3f * m, (pa * 0.5f - af) * rf * 0.7f);
    int n = 0;
#define P(a, r) (o[2 * n] = (r) * cosf(a), o[2 * n + 1] = (r) * sinf(a), n++)
    for (int k = 0; k < N; k++) {
        float c = rot + k * pa;
        P(c - pa * 0.5f, rf);
        for (int j = 0; j <= 2; j++) {
            float s = j * (float)PI / 4;
            P(c - af - rfil * (1 - sinf(s)) / rf, rf + rfil * (1 - cosf(s)));
        }
        P(c - af, rp);
        for (int j = 1; j <= 4; j++) {
            float s = j * (float)PI / 8, h = Ht * sinf(s), w = hw * cosf(s);
            P(c - w / (rp + h), rp + h);
        }
        for (int j = 3; j >= 1; j--) {
            float s = j * (float)PI / 8, h = Ht * sinf(s), w = hw * cosf(s);
            P(c + w / (rp + h), rp + h);
        }
        P(c + af, rp);
        for (int j = 2; j >= 0; j--) {
            float s = j * (float)PI / 4;
            P(c + af + rfil * (1 - sinf(s)) / rf, rf + rfil * (1 - cosf(s)));
        }
    }
#undef P
    return n;
}

/* a quadrilateral bar from (x0,y0) to (x1,y1), w0 wide at the start and w1
 * at the end */
static int bar_c(float *o, float x0, float y0, float x1, float y1, float w0, float w1) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy), px = -dy / l, py = dx / l;
    o[0] = x0 + px * w0 * 0.5f; o[1] = y0 + py * w0 * 0.5f;
    o[2] = x0 - px * w0 * 0.5f; o[3] = y0 - py * w0 * 0.5f;
    o[4] = x1 - px * w1 * 0.5f; o[5] = y1 - py * w1 * 0.5f;
    o[6] = x1 + px * w1 * 0.5f; o[7] = y1 + py * w1 * 0.5f;
    float cr = (o[2] - o[0]) * (o[5] - o[3]) - (o[3] - o[1]) * (o[4] - o[2]);
    if (cr < 0) {
        float t[8];
        memcpy(t, o, sizeof t);
        for (int i = 0; i < 4; i++) { o[2 * i] = t[2 * (3 - i)]; o[2 * i + 1] = t[2 * (3 - i) + 1]; }
    }
    return 4;
}

/* ---- primitive parts ---- */
static void cyl(float cx, float cy, float r, float z0, float z1, float bev) {
    int n = circle_c(cA, cx, cy, r, seg_for(r));
    extrude(cA, NULL, n, z0, z1, bev, 0);
}
static void ann(float cx, float cy, float r0, float r1, float z0, float z1, float bev) {
    int n = circle_c(cA, cx, cy, r1, seg_for(r1));
    match_circle(cA, n, cx, cy, r0, cB);
    extrude(cA, cB, n, z0, z1, bev, 0);
}
static void bar(float x0, float y0, float x1, float y1, float w0, float w1, float z0, float z1) {
    bar_c(cA, x0, y0, x1, y1, w0, w1);
    extrude(cA, NULL, 4, z0, z1, 0, 0);
}

/* A surface of revolution round the local z axis: profile points (r, z)
 * traced from the bottom up the outside (r >= 0), normals from the profile,
 * smoothed across gentle bends and kept sharp at corners. */
static void lathe(const float (*pr)[2], int np, int seg) {
    static float sn[256][2], na[256][2], nb[256][2];
    for (int i = 0; i + 1 < np; i++) {
        float dr = pr[i + 1][0] - pr[i][0], dz = pr[i + 1][1] - pr[i][1], l = sqrtf(dr * dr + dz * dz);
        sn[i][0] = l > 1e-7f ? dz / l : 0; sn[i][1] = l > 1e-7f ? -dr / l : 1;
    }
    for (int i = 0; i + 1 < np; i++) {
        for (int e = 0; e < 2; e++) {
            int k = e ? i + 1 : i - 1;   /* the neighbouring segment at this end */
            float *o = e ? nb[i] : na[i];
            o[0] = sn[i][0]; o[1] = sn[i][1];
            if (k < 0 || k + 1 >= np) continue;
            float d = sn[i][0] * sn[k][0] + sn[i][1] * sn[k][1];
            if (d > 0.75f) {
                float x = sn[i][0] + sn[k][0], y = sn[i][1] + sn[k][1], l = sqrtf(x * x + y * y);
                o[0] = x / l; o[1] = y / l;
            }
        }
    }
    for (int s = 0; s < seg; s++) {
        float a0 = (float)TAU * s / seg, a1 = (float)TAU * (s + 1) / seg;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        for (int i = 0; i + 1 < np; i++) {
            float r0 = pr[i][0], z0 = pr[i][1], r1 = pr[i + 1][0], z1 = pr[i + 1][1];
            if (fabsf(r1 - r0) < 1e-7f && fabsf(z1 - z0) < 1e-7f) continue;
            float A0[3] = {r0 * c0, r0 * s0, z0}, A1[3] = {r0 * c1, r0 * s1, z0};
            float B1[3] = {r1 * c1, r1 * s1, z1}, B0[3] = {r1 * c0, r1 * s0, z1};
            float nA0[3] = {na[i][0] * c0, na[i][0] * s0, na[i][1]}, nA1[3] = {na[i][0] * c1, na[i][0] * s1, na[i][1]};
            float nB1[3] = {nb[i][0] * c1, nb[i][0] * s1, nb[i][1]}, nB0[3] = {nb[i][0] * c0, nb[i][0] * s0, nb[i][1]};
            quad(A0, A1, B1, B0, nA0, nA1, nB1, nB0);
        }
    }
}

/* a uv sphere of radius r round the local origin */
static void sphere(float r, int seg, int rings) {
    for (int j = 0; j < rings; j++) {
        float t0 = (float)PI * j / rings - (float)PI / 2, t1 = (float)PI * (j + 1) / rings - (float)PI / 2;
        for (int i = 0; i < seg; i++) {
            float a0 = (float)TAU * i / seg, a1 = (float)TAU * (i + 1) / seg;
            float A[3] = {cosf(t0) * cosf(a0), cosf(t0) * sinf(a0), sinf(t0)}, B[3] = {cosf(t0) * cosf(a1), cosf(t0) * sinf(a1), sinf(t0)};
            float C[3] = {cosf(t1) * cosf(a1), cosf(t1) * sinf(a1), sinf(t1)}, D[3] = {cosf(t1) * cosf(a0), cosf(t1) * sinf(a0), sinf(t1)};
            float pa[3] = {A[0] * r, A[1] * r, A[2] * r}, pb[3] = {B[0] * r, B[1] * r, B[2] * r};
            float pc[3] = {C[0] * r, C[1] * r, C[2] * r}, pd[3] = {D[0] * r, D[1] * r, D[2] * r};
            quad(pa, pb, pc, pd, A, B, C, D);
        }
    }
}

/* a turned part placed with its axis from p along d */
static void lathe_along(const float *p, const float *d, const float (*pr)[2], int np, int seg) {
    M4 save = xf;
    int saved = xf_on;
    M4 a = m4_along(p, d);
    xf = saved ? m4_mul(save, a) : a;
    xf_on = 1;
    lathe(pr, np, seg);
    xf = save;
    xf_on = saved;
}

/* a turned rod from p0 to p1, radius r0 tapering to r1, with rounded ends */
static void rod(const float *p0, const float *p1, float r0, float r1, int seg) {
    float d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    float L = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    float pr[12][2];
    int n = 0;
    pr[n][0] = 0; pr[n][1] = 0; n++;
    for (int i = 1; i <= 3; i++) { float s = (float)PI / 2 * i / 3; pr[n][0] = r0 * sinf(s); pr[n][1] = r0 * (1 - cosf(s)) * 0.6f; n++; }
    for (int i = 0; i < 3; i++) { float s = (float)PI / 2 * (3 - i) / 3; pr[n][0] = r1 * sinf(s); pr[n][1] = L - r1 * (1 - cosf(s)) * 0.6f; n++; }
    pr[n][0] = 0; pr[n][1] = L; n++;
    lathe_along(p0, d, (const float (*)[2])pr, n, seg);
}

/* a lathe-turned part standing at (x, y, z) */
static void turned(float x, float y, float z, const float (*pr)[2], int np, int seg) {
    float p[3] = {x, y, z}, d[3] = {0, 0, 1};
    M4 save = xf;
    int saved = xf_on;
    M4 t = m4_trans(p[0], p[1], p[2]);
    (void)d;
    xf = saved ? m4_mul(save, t) : t;
    xf_on = 1;
    lathe(pr, np, seg);
    xf = save;
    xf_on = saved;
}

/* a slotted screw head standing on z */
static void screw(float x, float y, float z, float r, float a) {
    cur_mat = M_BLUED;
    float pr[][2] = {{0, 0}, {r, 0}, {r, 0.2f * r}, {r * 0.85f, 0.42f * r}, {r * 0.5f, 0.52f * r}, {0, 0.55f * r}};
    turned(x, y, z, (const float (*)[2])pr, 6, 20);
    cur_mat = M_BLACK;
    float c = cosf(a) * r * 0.9f, s = sinf(a) * r * 0.9f;
    bar(x - c, y - s, x + c, y + s, 0.2f * r, 0.2f * r, z + 0.3f * r, z + 0.56f * r);
}

/* a spoked wheel at the local origin: toothed rim, crossings, hub */
static void wheel(int N, float m, float z0, float z1, int spokes, float hub, float bore) {
    cur_mat = M_BRASS;
    float rp = m * N * 0.5f, rf = rp - 1.6f * m, rim = fmaxf(0.16f, rp * 0.08f), ri = rf - rim;
    int n = gear_c(cA, N, m, 1.35f, 1.6f, 0.95f, 0);
    if (ri < hub * 1.8f || spokes == 0) {   /* small: a plain disc */
        match_circle(cA, n, 0, 0, bore, cB);
        extrude(cA, cB, n, z0, z1, 0, 1);
        return;
    }
    match_circle(cA, n, 0, 0, ri, cB);
    extrude(cA, cB, n, z0, z1, 0, 1);
    float w = fmaxf(0.12f, ri * 0.085f);
    for (int k = 0; k < spokes; k++) {
        float a = (float)TAU * k / spokes + 0.3f;
        float pr = hub * 0.8f, pa = a;
        for (int j = 1; j <= 6; j++) {
            float u = j / 6.0f, r = hub * 0.8f + (ri + 0.05f - hub * 0.8f) * u;
            float b = a + 0.3f * sinf(u * (float)PI) * (ri / 4.0f) / (1 + ri / 4.0f);
            float ww0 = w * (1.3f - 0.45f * (j - 1) / 6.0f), ww1 = w * (1.3f - 0.45f * u);
            bar(pr * cosf(pa), pr * sinf(pa), r * cosf(b), r * sinf(b), ww0, ww1, z0 + 0.03f, z1 - 0.03f);
            pr = r; pa = b;
        }
    }
    ann(0, 0, bore, hub, z0 - 0.06f, z1 + 0.08f, 0.04f);
}

static void pinion(int N, float m, float z0, float z1, float bore) {
    cur_mat = M_STEEL;
    int n = gear_c(cA, N, m, 0.7f, 1.6f, 0.85f, 0);
    match_circle(cA, n, 0, 0, bore, cB);
    extrude(cA, cB, n, z0, z1, 0, 1);
}

/* ---- the orrery ----
 * Centimetres; z up from the table top, +x towards the vernal equinox (the
 * zero of ecliptic longitude), the planets going round anticlockwise seen
 * from above. The front (where the date window and the plaque face) is -y. */
#define Z_PLINTH 4.5f
#define Z_BASE 5.0f
#define Z_DIAL0 14.0f
#define Z_DIAL1 14.5f
#define R_DIAL 14.5f
#define R_PLINTH 17.75f
#define Z_PLANE 27.0f                      /* the Sun's centre and the planets' */
#define R_SUN 3.6f
#define R_LAMP 3.2f                        /* the light's radius, for the soft shadows */
static const float ARM_R[NPL] = {5.6f, 8.4f, 12.4f, 16.2f, 22.0f, 28.5f};
static const float ARM_Z[NPL] = {22.2f, 20.9f, 19.6f, 18.3f, 17.0f, 15.7f};
static const float PL_R[NPL] = {0.5f, 0.92f, 1.0f, 0.7f, 2.4f, 2.0f};
static const float TUBE_R[NPL] = {0.55f, 0.72f, 0.89f, 1.06f, 1.23f, 1.40f};
static const float LAYER_Z[7] = {5.55f, 6.75f, 7.95f, 9.15f, 10.35f, 11.55f, 12.75f};
#define WT 0.3f                            /* wheel thickness */
#define R_DRIVE 7.0f                       /* the drive arbor's distance from the centre */
static const float D_ANG = (float)(-30 * RAD);
/* gearing: the drive arbor turns once a year; each planet's tube wheel
 * meshes a wheel on it (Saturn's through an intermediate arbor and an idler) */
static const int N_TUBE[NPL] = {20, 40, 50, 62, 83, 81};
static const int N_DRV[NPL] = {83, 65, 50, 33, 7, 10};
#define N_INT 40
#define N_INTP 11
#define N_IDLER 24
static const float MOD_S1 = 0.2f, MOD_S2 = 0.11f;
static float mod_pair[NPL];
static float Dp[2], P2p[2], Ip[2];         /* drive, intermediate and idler arbors */
static const float JM_R[4] = {3.0f, 3.5f, 4.1f, 4.8f}, JM_Z[4] = {23.6f, 23.0f, 22.4f, 21.8f}, JM_SIZE[4] = {0.2f, 0.18f, 0.26f, 0.24f};
#define MOON_R 2.5f
#define MOON_Z 25.1f
static const float EARTH_TILT = (float)(23.44 * RAD);
/* Saturn's pole: ecliptic longitude 79.5, latitude 61.95 */
static const float SAT_POLE_LON = (float)(79.53 * RAD), SAT_TILT = (float)(28.05 * RAD);

/* the plaque on its stand in front */
static const float PLQ_C[3] = {0, -34.0f, 3.1f};
#define PLQ_W 16.0f
#define PLQ_H 6.0f
static float plq_u[3], plq_v[3], plq_n[3];
/* the date window in the dial plate */
static const float DATE_RECT[4] = {-2.5f, -7.25f, 5.0f, 1.25f};   /* x0 y0 w h */

enum { B_ROOM, B_TABLE, B_STATIC, B_SUN,
       B_TUBE0, B_GLOBE0 = B_TUBE0 + NPL, B_DRV0 = B_GLOBE0 + NPL, B_P2 = B_DRV0 + NPL, B_IDLER, B_MOONARM,
       B_JM0, NB = B_JM0 + 4 };
typedef struct { int first, count, cast; M4 m; } Body;
static Body body[NB];
static void begin_body(int i) { body[i].first = nv; body[i].cast = 1; }
static void end_body(int i) { body[i].count = nv - body[i].first; }

static double mesh_angle(double th1, int n1, int n2, float x1, float y1, float x2, float y2) {
    double phi = atan2(y2 - y1, x2 - x1);
    return phi + PI + PI / n2 - (th1 - phi) * n1 / n2;
}

/* the subdial hand's angle when the Earth is at heliocentric longitude l */
static double hand_angle(double l) {
    return mesh_angle(l, N_TUBE[P_EARTH], N_DRV[P_EARTH], 0, 0, Dp[0], Dp[1]);
}

/* ---- the model's state at a moment ---- */
typedef struct {
    double jd;
    double lon[NPL];                       /* heliocentric longitudes, degrees */
    double moon, jm[4];                    /* the Moon round the Earth; the Galilean moons round Jupiter */
    double drv[NPL], p2, idler;            /* wheel angles, radians */
    float pos[NPL][3];                     /* where the planets' centres are */
} Sky;

static void sky_at(double unix_t, Sky *s) {
    s->jd = jd_of(unix_t);
    for (int p = 0; p < NPL; p++) {
        double b, r;
        helio(p, s->jd, &s->lon[p], &b, &r);
        float a = (float)(s->lon[p] * RAD);
        s->pos[p][0] = ARM_R[p] * cosf(a);
        s->pos[p][1] = ARM_R[p] * sinf(a);
        s->pos[p][2] = Z_PLANE;
    }
    double mb;
    moon_pos(s->jd, &s->moon, &mb);
    jupiter_moons(s->jd, s->jm);
    for (int p = 0; p < P_SATURN; p++)
        s->drv[p] = mesh_angle(s->lon[p] * RAD, N_TUBE[p], N_DRV[p], 0, 0, Dp[0], Dp[1]);
    s->idler = mesh_angle(s->lon[P_SATURN] * RAD, N_TUBE[P_SATURN], N_IDLER, 0, 0, Ip[0], Ip[1]);
    s->p2 = mesh_angle(s->idler, N_IDLER, N_INTP, Ip[0], Ip[1], P2p[0], P2p[1]);
    s->drv[P_SATURN] = mesh_angle(s->p2, N_INT, N_DRV[P_SATURN], P2p[0], P2p[1], Dp[0], Dp[1]);
}

/* every body's model matrix; spin: seconds, for the planets' slow turning */
static void pose(const Sky *s, double spin) {
    body[B_ROOM].m = body[B_TABLE].m = body[B_STATIC].m = body[B_SUN].m = m4_id();
    for (int p = 0; p < NPL; p++) {
        float a = (float)(s->lon[p] * RAD);
        body[B_TUBE0 + p].m = m4_rotz(a);
        M4 t = m4_trans(s->pos[p][0], s->pos[p][1], s->pos[p][2]);
        float sp = (float)fmod(spin * TAU / (p == P_JUPITER ? 40.0 : p == P_SATURN ? 46.0 : p == P_EARTH ? 60.0 : 90.0), TAU);
        if (p == P_EARTH) body[B_GLOBE0 + p].m = m4_mul(t, m4_mul(m4_rotx(-EARTH_TILT), m4_rotz(sp)));
        else if (p == P_SATURN)
            body[B_GLOBE0 + p].m = m4_mul(t, m4_mul(m4_rotz(SAT_POLE_LON - (float)PI / 2), m4_mul(m4_rotx(-SAT_TILT), m4_rotz(sp))));
        else body[B_GLOBE0 + p].m = m4_mul(t, m4_rotz(sp + a));
        body[B_DRV0 + p].m = m4_mul(m4_trans(Dp[0], Dp[1], 0), m4_rotz((float)fmod(s->drv[p], TAU)));
    }
    body[B_P2].m = m4_mul(m4_trans(P2p[0], P2p[1], 0), m4_rotz((float)fmod(s->p2, TAU)));
    body[B_IDLER].m = m4_mul(m4_trans(Ip[0], Ip[1], 0), m4_rotz((float)fmod(s->idler, TAU)));
    body[B_MOONARM].m = m4_mul(m4_trans(s->pos[P_EARTH][0], s->pos[P_EARTH][1], 0), m4_rotz((float)(s->moon * RAD)));
    for (int k = 0; k < 4; k++)
        body[B_JM0 + k].m = m4_mul(m4_trans(s->pos[P_JUPITER][0], s->pos[P_JUPITER][1], 0),
                                   m4_rotz((float)((s->lon[P_JUPITER] + s->jm[k]) * RAD)));
}

/* where a moon is: the Earth's, or one of Jupiter's (k = 0..3) */
static void moon_xyz(const Sky *s, int k, float *o) {
    if (k < 0) {
        float a = (float)(s->moon * RAD);
        o[0] = s->pos[P_EARTH][0] + MOON_R * cosf(a); o[1] = s->pos[P_EARTH][1] + MOON_R * sinf(a); o[2] = Z_PLANE;
    } else {
        float a = (float)((s->lon[P_JUPITER] + s->jm[k]) * RAD);
        o[0] = s->pos[P_JUPITER][0] + JM_R[k] * cosf(a); o[1] = s->pos[P_JUPITER][1] + JM_R[k] * sinf(a); o[2] = Z_PLANE;
    }
}

/* The dial plate is skeletonised, so the lamp lights the train through it:
 * an engraved ring, a hub, the calendar subdial and the date cartouche,
 * joined by spokes, with bosses for the pillar screws and the studs. The
 */
typedef struct { int kind; float a[6]; } PShape;   /* 0 annulus (cx cy r0 r1), 1 disc (cx cy r), 2 bar (x0 y0 x1 y1 w0 w1) */
static PShape pshape[32];
static int npshape;
static void plate_shapes(void) {
    npshape = 0;
#define SH(k, ...) (pshape[npshape++] = (PShape){k, {__VA_ARGS__}})
    SH(0, 0, 0, 11.0f, R_DIAL);
    SH(0, 0, 0, 1.55f, 3.2f);
    SH(1, Dp[0], Dp[1], 4.05f);
    {   /* the date cartouche */
        float x0 = DATE_RECT[0] - 0.55f, x1 = DATE_RECT[0] + DATE_RECT[2] + 0.55f, yc = DATE_RECT[1] + DATE_RECT[3] * 0.5f;
        SH(2, x0, yc, x1, yc, DATE_RECT[3] + 1.1f, DATE_RECT[3] + 1.1f);
    }
    static const float spoke[] = {-90, 23, 105, 160, 230};
    for (int i = 0; i < 5; i++) {
        float a = (float)(spoke[i] * RAD), c = cosf(a), s = sinf(a);
        SH(2, 2.9f * c, 2.9f * s, 11.2f * c, 11.2f * s, 1.0f, 0.62f);
    }
    {   /* the subdial tied to the ring */
        float c = cosf(D_ANG), s = sinf(D_ANG);
        SH(2, Dp[0] + 3.6f * c, Dp[1] + 3.6f * s, 11.2f * c, 11.2f * s, 1.4f, 1.0f);
    }
    for (int k = 0; k < 3; k++) {
        float a = D_ANG + (float)((100 + 120 * k) * RAD);
        SH(1, 10.6f * cosf(a), 10.6f * sinf(a), 0.85f);
    }
    SH(1, P2p[0], P2p[1], 0.8f);
    SH(1, Ip[0], Ip[1], 0.75f);
#undef SH
}

/* a small gilt cup to seat a ball on, its rim at z */
static void cup(float x, float y, float z, float r) {
    cur_mat = M_GILT;
    float pr[][2] = {{0, -0.42f * r}, {0.45f * r, -0.42f * r}, {0.62f * r, -0.2f * r}, {0.9f * r, 0.0f}, {0.95f * r, 0.06f * r},
                     {0.7f * r, 0.08f * r}, {0, 0.02f * r}};
    turned(x, y, z, (const float (*)[2])pr, 7, 24);
}

/* a turned collar round an arbor of radius r at height z */
static void collar(float x, float y, float z, float r, float w, float h) {
    float pr[][2] = {{r, -h}, {r + w * 0.75f, -h}, {r + w, -h * 0.7f}, {r + w, -h * 0.25f}, {r + w * 0.8f, -h * 0.1f},
                     {r + w * 0.8f, h * 0.1f}, {r + w, h * 0.25f}, {r + w, h * 0.7f}, {r + w * 0.75f, h}, {r, h}};
    turned(x, y, z, (const float (*)[2])pr, 10, 32);
}

static void build(void) {
    /* where the arbors go */
    Dp[0] = R_DRIVE * cosf(D_ANG); Dp[1] = R_DRIVE * sinf(D_ANG);
    for (int p = 0; p < P_SATURN; p++) mod_pair[p] = 2 * R_DRIVE / (N_TUBE[p] + N_DRV[p]);
    {
        float d1 = MOD_S1 * (N_INT + N_DRV[P_SATURN]) * 0.5f, rho = 6.2f;
        float al = acosf((rho * rho + R_DRIVE * R_DRIVE - d1 * d1) / (2 * rho * R_DRIVE));
        P2p[0] = rho * cosf(D_ANG + al); P2p[1] = rho * sinf(D_ANG + al);
        float ri = MOD_S2 * (N_TUBE[P_SATURN] + N_IDLER) * 0.5f, d2 = MOD_S2 * (N_IDLER + N_INTP) * 0.5f;
        float be = acosf((rho * rho + ri * ri - d2 * d2) / (2 * rho * ri));
        Ip[0] = ri * cosf(D_ANG + al + be); Ip[1] = ri * sinf(D_ANG + al + be);
    }

    /* ---- the room: a dim study, drawn from the inside ---- */
    begin_body(B_ROOM);
    {
        cur_mat = M_ROOM;
        float X0 = -320, X1 = 320, Y0 = -460, Y1 = 260, Z0 = -76, Z1 = 215;
        float v[8][3] = {{X0, Y0, Z0}, {X1, Y0, Z0}, {X1, Y1, Z0}, {X0, Y1, Z0}, {X0, Y0, Z1}, {X1, Y0, Z1}, {X1, Y1, Z1}, {X0, Y1, Z1}};
        float nu[3] = {0, 0, 1}, nd[3] = {0, 0, -1}, nb[3] = {0, -1, 0}, nf[3] = {0, 1, 0}, nl[3] = {1, 0, 0}, nr[3] = {-1, 0, 0};
        quad(v[0], v[1], v[2], v[3], nu, nu, nu, nu);   /* floor */
        quad(v[4], v[7], v[6], v[5], nd, nd, nd, nd);   /* ceiling */
        quad(v[3], v[2], v[6], v[7], nb, nb, nb, nb);   /* back wall, with the window */
        quad(v[1], v[0], v[4], v[5], nf, nf, nf, nf);   /* front wall */
        quad(v[0], v[3], v[7], v[4], nl, nl, nl, nl);   /* left */
        quad(v[2], v[1], v[5], v[6], nr, nr, nr, nr);   /* right */
    }
    end_body(B_ROOM);
    body[B_ROOM].cast = 0;

    /* ---- the table: a slab of walnut with a rounded edge ---- */
    begin_body(B_TABLE);
    {
        cur_mat = M_TABLE;
        int n = 0;
        const float hx = 165, hy = 78, rc = 7;
        for (int q = 0; q < 4; q++) {
            float cx = q == 0 || q == 3 ? hx - rc : -hx + rc, cy = q < 2 ? hy - rc : -hy + rc;
            for (int k = 0; k <= 8; k++) {
                float a = (float)PI / 2 * (q + k / 8.0f);
                cA[2 * n] = cx + rc * cosf(a); cA[2 * n + 1] = cy + rc * sinf(a); n++;
            }
        }
        extrude(cA, NULL, n, -4.5f, 0, 0.5f, 0);
    }
    end_body(B_TABLE);
    body[B_TABLE].cast = 0;

    /* ---- everything that stands still ---- */
    begin_body(B_STATIC);
    {
        /* the turned mahogany plinth */
        cur_mat = M_MAHOGANY;
        float pl[][2] = {{0, 0}, {17.2f, 0}, {17.25f, 0.45f}, {17.6f, 0.62f}, {R_PLINTH, 0.95f}, {17.6f, 1.28f}, {17.2f, 1.42f},
                         {16.95f, 1.6f}, {16.62f, 2.15f}, {16.5f, 2.85f}, {16.5f, 3.45f}, {16.72f, 3.6f}, {16.88f, 3.88f},
                         {16.8f, 4.2f}, {16.55f, 4.38f}, {16.2f, 4.5f}, {0, 4.5f}};
        turned(0, 0, 0, (const float (*)[2])pl, (int)(sizeof pl / sizeof *pl), 160);
        /* the base plate */
        cur_mat = M_BRASS_DARK;
        int n = circle_c(cA, 0, 0, 14.0f, 160);
        extrude(cA, NULL, n, Z_PLINTH, Z_BASE, 0.12f, 0);
        /* pillars, turned, between the base plate and the dial plate */
        float pp[][2] = {{0, 0}, {0.62f, 0}, {0.62f, 0.22f}, {0.46f, 0.38f}, {0.4f, 0.85f}, {0.52f, 1.05f}, {0.4f, 1.28f},
                         {0.33f, 2.4f}, {0.29f, 4.5f}, {0.33f, 6.6f}, {0.4f, 7.72f}, {0.52f, 7.95f}, {0.4f, 8.15f},
                         {0.46f, 8.62f}, {0.62f, 8.78f}, {0.62f, 9.0f}, {0, 9.0f}};
        for (int k = 0; k < 3; k++) {
            float a = D_ANG + (float)((100 + 120 * k) * RAD);
            cur_mat = M_BRASS;
            turned(10.6f * cosf(a), 10.6f * sinf(a), Z_BASE, (const float (*)[2])pp, (int)(sizeof pp / sizeof *pp), 28);
            screw(10.6f * cosf(a), 10.6f * sinf(a), Z_DIAL1, 0.42f, a + 0.6f);
        }
        /* the dial plate: silvered on top, brass underneath and round the edge */
        cur_mat = M_DIAL;
        plate_shapes();
        for (int i = 0; i < npshape; i++) {
            const float *a = pshape[i].a;
            if (pshape[i].kind == 0) {
                n = circle_c(cA, a[0], a[1], a[3], a[3] > 5 ? 240 : 64);
                match_circle(cA, n, a[0], a[1], a[2], cB);
                extrude(cA, cB, n, Z_DIAL0, Z_DIAL1, i == 0 ? 0.1f : 0.0f, 1);
            } else if (pshape[i].kind == 1) {
                n = circle_c(cA, a[0], a[1], a[2], 64);
                extrude(cA, NULL, n, Z_DIAL0, Z_DIAL1 - 0.001f * i, 0, 1);
            } else {
                bar_c(cA, a[0], a[1], a[2], a[3], a[4], a[5]);
                extrude(cA, NULL, 4, Z_DIAL0 + 0.001f, Z_DIAL1 - 0.001f * i, 0, 1);
            }
        }
        /* a gilt bezel round the date window */
        cur_mat = M_GILT;
        {
            float x0 = DATE_RECT[0], y0 = DATE_RECT[1], x1 = x0 + DATE_RECT[2], y1 = y0 + DATE_RECT[3], b = 0.14f, z = Z_DIAL1;
            bar(x0 - b, y0 - b * 0.5f, x1 + b, y0 - b * 0.5f, b, b, z - 0.01f, z + 0.06f);
            bar(x0 - b, y1 + b * 0.5f, x1 + b, y1 + b * 0.5f, b, b, z - 0.01f, z + 0.06f);
            bar(x0 - b * 0.5f, y0 - b, x0 - b * 0.5f, y1 + b, b, b, z - 0.01f, z + 0.06f);
            bar(x1 + b * 0.5f, y0 - b, x1 + b * 0.5f, y1 + b, b, b, z - 0.01f, z + 0.06f);
        }
        /* the fixed central arbor, carrying the Sun */
        cur_mat = M_STEEL;
        cyl(0, 0, 0.38f, Z_BASE, 23.6f, 0);
        /* studs under the dial plate for the intermediate wheel and the idler */
        cyl(P2p[0], P2p[1], 0.22f, LAYER_Z[5] - 0.3f, Z_DIAL0, 0);
        cyl(Ip[0], Ip[1], 0.2f, LAYER_Z[6] - 0.25f, Z_DIAL0, 0);
        cur_mat = M_BRASS;
        cyl(P2p[0], P2p[1], 0.45f, Z_DIAL0 - 0.35f, Z_DIAL0, 0);
        cyl(Ip[0], Ip[1], 0.42f, Z_DIAL0 - 0.35f, Z_DIAL0, 0);
        screw(P2p[0], P2p[1], Z_DIAL1, 0.32f, 0.4f);
        screw(Ip[0], Ip[1], Z_DIAL1, 0.3f, 1.1f);
        /* a bearing boss on the base plate under each arbor */
        cur_mat = M_BRASS;
        float bp[][2] = {{0, 0}, {0.75f, 0}, {0.72f, 0.12f}, {0.5f, 0.3f}, {0, 0.3f}};
        turned(0, 0, Z_BASE, (const float (*)[2])bp, 5, 28);
        turned(Dp[0], Dp[1], Z_BASE, (const float (*)[2])bp, 5, 28);
        /* the subdial's centre boss round the drive arbor */
        cur_mat = M_GILT;
        ann(Dp[0], Dp[1], 0.3f, 0.5f, Z_DIAL1 - 0.01f, Z_DIAL1 + 0.04f, 0.02f);

        /* the plaque, on a mahogany stand */
        {
            float ex[3] = {1, 0, 0}, ey[3] = {0, 7.0f, 3.8f};
            v3norm(ey);
            float ez[3];
            v3cross(ex, ey, ez);
            memcpy(plq_u, ex, sizeof plq_u); memcpy(plq_v, ey, sizeof plq_v); memcpy(plq_n, ez, sizeof plq_n);
            /* the stand: a wedge, its side profile extruded across */
            M4 sm = m4_id();
            sm.m[0] = 0; sm.m[1] = 1; sm.m[2] = 0;
            sm.m[4] = 0; sm.m[5] = 0; sm.m[6] = 1;
            sm.m[8] = 1; sm.m[9] = 0; sm.m[10] = 0;
            sm.m[12] = 0; sm.m[13] = PLQ_C[1]; sm.m[14] = 0;
            xf = sm; xf_on = 1;
            cur_mat = M_MAHOGANY;
            float wp[][2] = {{-3.5f, 0}, {3.5f, 0}, {3.5f, 5.0f}, {-3.5f, 1.2f}};
            for (int i = 0; i < 4; i++) { cA[2 * i] = wp[i][0]; cA[2 * i + 1] = wp[i][1]; }
            extrude(cA, NULL, 4, -9.0f, 9.0f, 0, 1);
            xf_on = 0;
            /* the brass plate, on the slope */
            M4 pm = m4_id();
            for (int i = 0; i < 3; i++) { pm.m[i] = ex[i]; pm.m[4 + i] = ey[i]; pm.m[8 + i] = ez[i]; pm.m[12 + i] = PLQ_C[i]; }
            xf = pm; xf_on = 1;
            cur_mat = M_PLAQUE;
            float px0 = -PLQ_W / 2, px1 = PLQ_W / 2, py0 = -PLQ_H / 2, py1 = PLQ_H / 2;
            float rc[][2] = {{px0, py0}, {px1, py0}, {px1, py1}, {px0, py1}};
            int nn = 0;
            for (int i = 0; i < 4; i++) {   /* the corners rounded a little */
                float cx = rc[i][0] + (i == 0 || i == 3 ? 0.3f : -0.3f), cy = rc[i][1] + (i < 2 ? 0.3f : -0.3f);
                for (int k = 0; k <= 4; k++) {
                    float a = (float)PI / 2 * (i + 2) + (float)PI / 2 * k / 4.0f;
                    cA[2 * nn] = cx + 0.3f * cosf(a); cA[2 * nn + 1] = cy + 0.3f * sinf(a); nn++;
                }
            }
            extrude(cA, NULL, nn, 0, 0.16f, 0.07f, 0);
            /* four screws in the corners */
            for (int i = 0; i < 4; i++) {
                float sx = (i & 1 ? 1 : -1) * (PLQ_W / 2 - 0.55f), sy = (i & 2 ? 1 : -1) * (PLQ_H / 2 - 0.55f);
                screw(sx, sy, 0.16f, 0.22f, 0.5f + i);
            }
            xf_on = 0;
        }
    }
    end_body(B_STATIC);

    /* ---- the Sun: a frosted glass globe in gilt straps ---- */
    begin_body(B_SUN);
    {
        xf = m4_trans(0, 0, Z_PLANE); xf_on = 1;
        cur_mat = M_SUN;
        sphere(R_SUN, 72, 36);
        cur_mat = M_GILT;
        float band[13][2];
        for (int k = 0; k < 13; k++) {
            float a = (float)PI * (k / 12.0f) - (float)PI / 2;
            band[k][0] = R_SUN + 0.02f + 0.1f * cosf(a); band[k][1] = 0.16f * sinf(a);
        }
        lathe((const float (*)[2])band, 13, 120);
        float o[3] = {0, 0, 0}, ax[3] = {1, 0, 0}, ay[3] = {0, 1, 0};
        float hoop[13][2];
        for (int k = 0; k < 13; k++) {
            float a = (float)PI * (k / 12.0f) - (float)PI / 2;
            hoop[k][0] = R_SUN + 0.02f + 0.07f * cosf(a); hoop[k][1] = 0.09f * sinf(a);
        }
        lathe_along(o, ax, (const float (*)[2])hoop, 13, 120);
        lathe_along(o, ay, (const float (*)[2])hoop, 13, 120);
        /* finial on top, gallery underneath */
        float fin[][2] = {{0, R_SUN - 0.1f}, {0.55f, R_SUN - 0.1f}, {0.6f, R_SUN + 0.15f}, {0.3f, R_SUN + 0.35f}, {0.36f, R_SUN + 0.6f},
                          {0.18f, R_SUN + 0.8f}, {0.06f, R_SUN + 1.5f}, {0, R_SUN + 1.55f}};
        lathe((const float (*)[2])fin, 8, 32);
        float gal[][2] = {{0, -R_SUN - 1.0f}, {0.5f, -R_SUN - 1.0f}, {0.62f, -R_SUN - 0.6f}, {1.1f, -R_SUN - 0.15f},
                          {1.75f, -R_SUN + 0.25f}, {1.86f, -R_SUN + 0.5f}, {1.7f, -R_SUN + 0.48f}, {0, -R_SUN + 0.1f}};
        lathe((const float (*)[2])gal, 8, 48);
        xf_on = 0;
    }
    end_body(B_SUN);

    /* ---- the planets' tubes and arms, each built along +x ---- */
    for (int p = 0; p < NPL; p++) {
        begin_body(B_TUBE0 + p);
        int layer = p == P_SATURN ? 6 : p;
        float tr = TUBE_R[p], z = ARM_Z[p], R = ARM_R[p];
        cur_mat = M_BRASS;
        ann(0, 0, tr - 0.12f, tr, LAYER_Z[layer] - 0.3f, z, 0);
        float m = p == P_SATURN ? MOD_S2 : mod_pair[p];
        int nt = N_TUBE[p];
        wheel(nt, m, LAYER_Z[layer], LAYER_Z[layer] + WT, nt >= 40 ? 5 : 0, tr + 0.28f, tr - 0.02f);
        cur_mat = M_BRASS;
        collar(0, 0, z, tr, 0.38f, 0.42f);
        float a0[3] = {tr + 0.25f, 0, z}, a1[3] = {R - 0.3f, 0, z};
        float r0 = 0.2f + 0.022f * p;
        rod(a0, a1, r0, r0 * 0.68f, 20);
        if (p >= P_MARS) {     /* a counterweight */
            float c0[3] = {-tr - 0.2f, 0, z}, c1[3] = {-2.5f - 0.25f * (p - P_MARS), 0, z};
            rod(c0, c1, r0 * 0.8f, r0 * 0.7f, 16);
            xf = m4_trans(c1[0] - 0.35f, 0, z); xf_on = 1;
            sphere(0.42f + 0.06f * (p - P_MARS), 24, 12);
            xf_on = 0;
        }
        /* the elbow and the post */
        float kn[][2] = {{0, -0.4f}, {0.22f, -0.4f}, {0.36f, -0.22f}, {0.4f, 0}, {0.34f, 0.22f}, {0.2f, 0.42f}, {0, 0.44f}};
        turned(R, 0, z, (const float (*)[2])kn, 7, 24);
        cur_mat = M_STEEL;
        float top = Z_PLANE - PL_R[p] * (p == P_SATURN ? 0.9f : p == P_JUPITER ? 0.935f : 1.0f) - 0.06f;
        float q0[3] = {R, 0, z + 0.3f}, q1[3] = {R, 0, top - 0.1f};
        rod(q0, q1, p >= P_JUPITER ? 0.16f : 0.12f, p >= P_JUPITER ? 0.13f : 0.1f, 16);
        cup(R, 0, top, fmaxf(0.38f, PL_R[p] * 0.45f));
        end_body(B_TUBE0 + p);
    }

    /* ---- the planets themselves, round their own centres ---- */
    for (int p = 0; p < NPL; p++) {
        begin_body(B_GLOBE0 + p);
        float r = PL_R[p];
        int seg = p == P_JUPITER ? 80 : p == P_SATURN ? 72 : p == P_EARTH ? 56 : 40;
        cur_mat = M_MERCURY + p;
        if (p == P_JUPITER || p == P_SATURN) { xf = m4_scale(1, 1, p == P_SATURN ? 0.9f : 0.935f); xf_on = 1; }
        sphere(r, seg, seg / 2);
        xf_on = 0;
        if (p == P_EARTH) {    /* the axis through the poles */
            cur_mat = M_STEEL;
            float a[3] = {0, 0, -1.32f}, b[3] = {0, 0, 1.32f};
            rod(a, b, 0.05f, 0.05f, 12);
            cur_mat = M_GILT;
            float cp[][2] = {{0, 0.96f}, {0.16f, 0.96f}, {0.12f, 1.08f}, {0, 1.1f}};
            lathe((const float (*)[2])cp, 4, 16);
            float cn[][2] = {{0, -1.1f}, {0.12f, -1.08f}, {0.16f, -0.96f}, {0, -0.96f}};
            lathe((const float (*)[2])cn, 4, 16);
        }
        if (p == P_SATURN) {   /* the rings, with the Cassini division */
            cur_mat = M_RING;
            float rr[2][2] = {{2.45f, 3.86f}, {3.98f, 4.62f}};
            for (int k = 0; k < 2; k++) {
                int n = circle_c(cA, 0, 0, rr[k][1], 160);
                match_circle(cA, n, 0, 0, rr[k][0], cB);
                extrude(cA, cB, n, -0.025f, 0.025f, 0, 1);
            }
        }
        end_body(B_GLOBE0 + p);
    }

    /* ---- the train: the drive arbor's wheels ---- */
    for (int p = 0; p < NPL; p++) {
        begin_body(B_DRV0 + p);
        int layer = p;
        int nd = N_DRV[p];
        float m = p == P_SATURN ? MOD_S1 : mod_pair[p];
        if (nd >= 20) wheel(nd, m, LAYER_Z[layer], LAYER_Z[layer] + WT, nd >= 50 ? 6 : 4, 0.62f, 0.27f);
        else pinion(nd, m, LAYER_Z[layer] - 0.2f, LAYER_Z[layer] + WT + 0.2f, 0.27f);
        if (p == P_EARTH) {    /* the arbor itself, and the subdial hand */
            cur_mat = M_STEEL;
            cyl(0, 0, 0.27f, Z_BASE + 0.25f, Z_DIAL1 + 0.3f, 0.03f);
            cur_mat = M_BLUED;
            bar(-0.9f, 0, 3.45f, 0, 0.16f, 0.025f, Z_DIAL1 + 0.06f, Z_DIAL1 + 0.11f);
            cyl(-0.9f, 0, 0.17f, Z_DIAL1 + 0.06f, Z_DIAL1 + 0.11f, 0);
            cyl(0, 0, 0.36f, Z_DIAL1 + 0.04f, Z_DIAL1 + 0.16f, 0.03f);
            cur_mat = M_GILT;
            cyl(0, 0, 0.2f, Z_DIAL1 + 0.16f, Z_DIAL1 + 0.32f, 0.06f);
            /* collets holding the other wheels, so the arbor reads as one */
            cur_mat = M_BRASS;
            for (int k = 0; k < 6; k++)
                if (k != P_EARTH) ann(0, 0, 0.26f, 0.42f, LAYER_Z[k] + WT, LAYER_Z[k] + WT + 0.18f, 0.03f);
        }
        end_body(B_DRV0 + p);
    }
    /* the intermediate arbor: wheel 40 under pinion 11 */
    begin_body(B_P2);
    wheel(N_INT, MOD_S1, LAYER_Z[5], LAYER_Z[5] + WT, 4, 0.55f, 0.24f);
    pinion(N_INTP, MOD_S2, LAYER_Z[6] - 0.15f, LAYER_Z[6] + WT + 0.15f, 0.24f);
    cur_mat = M_BRASS;
    ann(0, 0, 0.24f, 0.4f, LAYER_Z[5] + WT, LAYER_Z[6] - 0.15f, 0);
    end_body(B_P2);
    begin_body(B_IDLER);
    wheel(N_IDLER, MOD_S2, LAYER_Z[6], LAYER_Z[6] + WT, 0, 0.4f, 0.22f);
    end_body(B_IDLER);

    /* ---- the Moon on its arm round the Earth's post ---- */
    begin_body(B_MOONARM);
    {
        cur_mat = M_BRASS;
        collar(0, 0, MOON_Z, 0.12f, 0.26f, 0.28f);
        float a[3] = {0.3f, 0, MOON_Z}, b[3] = {MOON_R, 0, MOON_Z};
        rod(a, b, 0.08f, 0.06f, 12);
        cur_mat = M_STEEL;
        float c[3] = {MOON_R, 0, MOON_Z}, d[3] = {MOON_R, 0, Z_PLANE - 0.36f};
        rod(c, d, 0.05f, 0.045f, 10);
        cup(MOON_R, 0, Z_PLANE - 0.3f, 0.3f);
        cur_mat = M_MOON;
        xf = m4_trans(MOON_R, 0, Z_PLANE); xf_on = 1;
        sphere(0.3f, 32, 16);
        xf_on = 0;
    }
    end_body(B_MOONARM);

    /* ---- Jupiter's moons on their wires ---- */
    for (int k = 0; k < 4; k++) {
        begin_body(B_JM0 + k);
        cur_mat = M_BRASS;
        collar(0, 0, JM_Z[k], 0.16f, 0.16f, 0.14f);
        cur_mat = M_STEEL;
        float a[3] = {0.25f, 0, JM_Z[k]}, b[3] = {JM_R[k], 0, JM_Z[k]}, c[3] = {JM_R[k], 0, Z_PLANE - JM_SIZE[k]};
        rod(a, b, 0.045f, 0.04f, 8);
        rod(b, c, 0.04f, 0.035f, 8);
        cur_mat = M_IVORY_BALL;
        xf = m4_trans(JM_R[k], 0, Z_PLANE); xf_on = 1;
        sphere(JM_SIZE[k], 20, 10);
        xf_on = 0;
        end_body(B_JM0 + k);
    }
}

/* ---- engraving: a freetype glyph atlas, set into textures ----
 * The dial plate's scales (degrees and signs of the ecliptic, the calendar
 * subdial, an inscription) are drawn once into a texture laid over the
 * plate; the plaque and the date window into small ones, redrawn when their
 * text changes. The strokes are glyph quads (rotated along arcs) and solid
 * quads for rules and ticks, max-blended into a coverage mask. */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 24
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[3];               /* regular, bold, italic */
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    static const char *var[3] = {"weight=regular", "weight=bold", "slant=italic"};
    for (int i = 0; i < 3; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:%s:fontformat=CFF,TrueType", opt_font, var[i]);
        FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        if (FT_New_Face(ft, (const char *)file, 0, &faces[i])) die("can't load the font");
        FcPatternDestroy(m);
        FcPatternDestroy(pat);
    }
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 6) p = 6;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) die("too many text styles");
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    return nstyles++;
}

/* texel (1..6, 1..6) is solid, for rules and ticks */
static void build_atlas(void) {
    free(atlas);
    int cap = 8192;
    atlas = calloc((size_t)AW * cap, 1);
    for (int y = 0; y < 8; y++) memset(atlas + (size_t)y * AW, 255, 8);
    int x = 10, y = 1, shelf = 8;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < 96; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c == 95 ? 0xB0 : (FT_ULong)(32 + c);
            if (FT_Load_Char(f, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot g = f->glyph;
            int w = (int)g->bitmap.width, h = (int)g->bitmap.rows;
            if (x + w + 2 > AW) { x = 1; y += shelf + 2; shelf = 0; }
            if (y + h + 2 > cap) die("glyph atlas full");
            for (int yy = 0; yy < h; yy++)
                memcpy(atlas + (size_t)(y + yy) * AW + x, g->bitmap.buffer + yy * g->bitmap.pitch, (size_t)w);
            *gl = (Glyph){(short)x, (short)y, (short)w, (short)h, (short)g->bitmap_left, (short)g->bitmap_top,
                          (short)(g->advance.x >> 6)};
            x += w + 2;
            if (h > shelf) shelf = h;
        }
    }
    atlas_h = y + shelf + 2;
}

static const Glyph *glyph(int s, unsigned char c) {
    int i = c == 0x7f ? 95 : c >= 32 && c < 127 ? c - 32 : 0;
    return &styles[s].g[i];
}

/* text quads: x, y (NDC), u, v (atlas texels), coverage */
static float *tv;
static int tvn, tvcap;
static float tx_o[2], tx_s[2];         /* target space (cm) -> NDC: (p - o) * s */

static void tq(float x, float y, float u, float v, float a) {
    if (tvn + 5 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 65536) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = (x - tx_o[0]) * tx_s[0] - 1; o[1] = (y - tx_o[1]) * tx_s[1] - 1; o[2] = u; o[3] = v; o[4] = a;
    tvn += 5;
}

/* a solid stroke from (x0,y0) to (x1,y1), w wide */
static void tseg(float x0, float y0, float x1, float y1, float w, float a) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1e-6f) return;
    float px = -dy / l * w * 0.5f, py = dx / l * w * 0.5f;
    tq(x0 + px, y0 + py, 4, 4, a); tq(x1 + px, y1 + py, 4, 4, a); tq(x1 - px, y1 - py, 4, 4, a);
    tq(x0 + px, y0 + py, 4, 4, a); tq(x1 - px, y1 - py, 4, 4, a); tq(x0 - px, y0 - py, 4, 4, a);
}
static void tcircle(float cx, float cy, float r, float w, float a) {
    int n = (int)(r * 60) + 48;
    for (int i = 0; i < n; i++) {
        float a0 = (float)TAU * i / n, a1 = (float)TAU * (i + 1.08f) / n;
        float rr = r;
        tseg(cx + rr * cosf(a0), cy + rr * sinf(a0), cx + rr * cosf(a1), cy + rr * sinf(a1), w, a);
    }
}
static void tdot(float x, float y, float r, float a) {
    tseg(x - r, y, x + r, y, r * 2, a);
}

/* one glyph, pen at (x, y) on the baseline, running along (bx, by); k: cm per atlas pixel */
static void tglyph(int s, unsigned char c, float x, float y, float bx, float by, float k, float a) {
    const Glyph *g = glyph(s, c);
    if (!g->w) return;
    float ux = -by, uy = bx;
    float l = g->left * k, r = (g->left + g->w) * k, t = g->top * k, b = (g->top - g->h) * k;
#define C(h, v) x + bx * (h) + ux * (v), y + by * (h) + uy * (v)
    tq(C(l, t), g->x, g->y, a); tq(C(r, t), g->x + g->w, g->y, a); tq(C(r, b), g->x + g->w, g->y + g->h, a);
    tq(C(l, t), g->x, g->y, a); tq(C(r, b), g->x + g->w, g->y + g->h, a); tq(C(l, b), g->x, g->y + g->h, a);
#undef C
}

/* '*' in a string sets a small centred dot */
static float adv_of(int s, unsigned char c, float k) {
    return (c == '*' ? glyph(s, ' ')->adv * 1.6f : glyph(s, c)->adv) * k;
}
static float twidth(int s, const char *str, float track, float k) {
    float w = 0;
    for (const char *p = str; *p; p++) w += adv_of(s, (unsigned char)*p, k) + (p[1] ? track : 0);
    return w;
}

/* a line of text along angle ang, baseline through (x, y); align -1 left, 0 centre, 1 right; track in cm */
static void tline(int s, float x, float y, float ang, int align, const char *str, float track, float k, float a) {
    float w = twidth(s, str, track, k), bx = cosf(ang), by = sinf(ang);
    float pen = align == 0 ? -w * 0.5f : align > 0 ? -w : 0;
    float xh = glyph(s, 'x')->top * k;
    for (const char *p = str; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '*') {
            float aw = adv_of(s, c, k), cx = pen + aw * 0.5f;
            tdot(x + bx * cx - by * xh * 0.55f, y + by * cx + bx * xh * 0.55f, xh * 0.13f, a);
        } else {
            tglyph(s, c, x + bx * pen, y + by * pen, bx, by, k, a);
        }
        pen += adv_of(s, c, k) + track;
    }
}

/* text round an arc of radius r (the baseline) about (cx, cy), centred on
 * angle ang; outward: tops away from the centre, reading clockwise */
static void tarc(int s, float cx, float cy, float r, float ang, const char *str, float track, float k, int outward, float a) {
    float w = twidth(s, str, track, k), pen = -w * 0.5f, dir = outward ? -1.0f : 1.0f;
    for (const char *p = str; *p; p++) {
        unsigned char c = (unsigned char)*p;
        float aw = adv_of(s, c, k);
        float th = ang + dir * (pen + aw * 0.5f) / r;     /* the glyph's middle */
        float ux = cosf(th) * (outward ? 1 : -1), uy = sinf(th) * (outward ? 1 : -1);
        float bx = uy, by = -ux;                         /* baseline: up turned clockwise */
        float mx = cx + r * cosf(th), my = cy + r * sinf(th);
        if (c == '*') {
            float xh = glyph(s, 'x')->top * k;
            tdot(mx + ux * xh * 0.55f, my + uy * xh * 0.55f, xh * 0.13f, a);
        } else {
            tglyph(s, c, mx - bx * aw * 0.5f, my - by * aw * 0.5f, bx, by, k, a);
        }
        pen += aw + track;
    }
}

/* the text pipeline */
static GLuint prog_text, vao_text, vbo_text, tex_atlas;
static GLuint tex_dial, tex_plaque, tex_date, fbo_text;
#define DIAL_TEX 4096
#define PLQ_TEX_W 1536
#define PLQ_TEX_H 576
#define DATE_TEX_W 1024
#define DATE_TEX_H 256

/* styles, by use; set in text_styles() */
static int st_sign, st_deg, st_month, st_inscr, st_host, st_line, st_small, st_date, st_sun;
static const float K_DIAL = 2 * R_DIAL / DIAL_TEX, K_PLQ = PLQ_W / PLQ_TEX_W, K_DATE = 5.0f / DATE_TEX_W;

static void text_styles(void) {
    /* a style's pixel size is its em in cm over the target's cm per texel */
    st_sign = style(0, 0.56f / K_DIAL);
    st_deg = style(0, 0.34f / K_DIAL);
    st_month = style(0, 0.30f / K_DIAL);
    st_inscr = style(2, 0.46f / K_DIAL);
    st_sun = style(0, 0.24f / K_DIAL);
    st_host = style(0, 1.22f / K_PLQ);
    st_line = style(0, 0.56f / K_PLQ);
    st_small = style(2, 0.52f / K_PLQ);
    st_date = style(0, 0.74f / K_DATE);
}

static GLuint mk_mask(int w, int h) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4.0f);
    return t;
}

/* draw the quads gathered in tv into texture t (w x h), clearing it first */
static void flush_text(GLuint t, int w, int h) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_text);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendEquation(GL_MAX);
    glUseProgram(prog_text);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)tvn * (GLsizeiptr)sizeof(float), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 5);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glBindTexture(GL_TEXTURE_2D, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tvn = 0;
}

static void set_space(float x0, float y0, float w, float h) {
    tx_o[0] = x0; tx_o[1] = y0; tx_s[0] = 2 / w; tx_s[1] = 2 / h;
}

static const char *MONTH3[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
static const char *MONTH[12] = {"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER",
                                "OCTOBER", "NOVEMBER", "DECEMBER"};
static const char *WDAY[7] = {"SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"};
static const char *SIGN[12] = {"ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO", "LIBRA", "SCORPIO",
                               "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES"};

/* the dial plate's engraving, for the year containing unix time `now_t` */
static void draw_dial(double now_t) {
    set_space(-R_DIAL, -R_DIAL, 2 * R_DIAL, 2 * R_DIAL);
    const float k = K_DIAL;
    /* the ecliptic ring */
    tcircle(0, 0, 11.05f, 0.03f, 1);
    tcircle(0, 0, 11.78f, 0.02f, 1);
    tcircle(0, 0, 12.45f, 0.022f, 1);
    tcircle(0, 0, 13.72f, 0.03f, 1);
    tcircle(0, 0, 13.84f, 0.012f, 1);
    tcircle(0, 0, 14.18f, 0.035f, 1);
    for (int d = 0; d < 360; d++) {
        float a = (float)(d * RAD), c = cosf(a), s = sinf(a);
        float r0 = d % 10 == 0 ? 11.05f : d % 5 == 0 ? 11.4f : 11.56f, w = d % 10 == 0 ? 0.03f : d % 5 == 0 ? 0.024f : 0.016f;
        tseg(r0 * c, r0 * s, 11.78f * c, 11.78f * s, w, 1);
        if (d % 2 == 0) tdot(13.98f * c, 13.98f * s, d % 10 == 0 ? 0.04f : 0.022f, 1);
    }
    for (int d = 0; d < 360; d += 10) {
        char b[8];
        snprintf(b, sizeof b, "%d", d % 30 == 0 ? 0 : d % 30);
        tarc(st_deg, 0, 0, 11.93f, (float)(d * RAD), b, 0.0f, k, 1, 1);
    }
    for (int i = 0; i < 12; i++) {
        float a = (float)(i * 30 * RAD);
        tseg(12.45f * cosf(a), 12.45f * sinf(a), 13.72f * cosf(a), 13.72f * sinf(a), 0.03f, 1);
        tarc(st_sign, 0, 0, 12.86f, a + (float)(15 * RAD), SIGN[i], 0.09f, k, 1, 1);
    }
    /* round the centre: the sun's rays */
    tcircle(0, 0, 1.9f, 0.025f, 1);
    tcircle(0, 0, 2.02f, 0.012f, 1);
    for (int i = 0; i < 48; i++) {
        float a = (float)TAU * i / 48, l = i % 2 ? 2.5f : 3.0f;
        float c = cosf(a), s = sinf(a);
        float w0 = i % 2 ? 0.05f : 0.08f;
        int segs = 6;
        for (int j = 0; j < segs; j++) {   /* a flame: tapering, with a wave */
            float u0 = (float)j / segs, u1 = (float)(j + 1) / segs;
            float r0 = 2.1f + (l - 2.1f) * u0, r1 = 2.1f + (l - 2.1f) * u1;
            float o0 = (i % 2 ? 0 : 0.06f * sinf(u0 * 6.28f)), o1 = (i % 2 ? 0 : 0.06f * sinf(u1 * 6.28f));
            tseg(r0 * c - o0 * s, r0 * s + o0 * c, r1 * c - o1 * s, r1 * s + o1 * c, w0 * (1 - u0 * 0.85f), 1);
        }
    }
    /* an inscription round the back */
    tarc(st_sun, 0, 0, 3.55f, (float)(150 * RAD), "SOL", 0.12f, k, 1, 1);

    /* the calendar subdial round the drive arbor: each day engraved where the
     * hand points when the Earth is where it is on that date */
    float cx = Dp[0], cy = Dp[1];
    tcircle(cx, cy, 3.78f, 0.035f, 1);
    tcircle(cx, cy, 3.69f, 0.012f, 1);
    tcircle(cx, cy, 3.28f, 0.02f, 1);
    tcircle(cx, cy, 2.58f, 0.022f, 1);
    tcircle(cx, cy, 2.49f, 0.012f, 1);
    time_t tt = (time_t)now_t;
    struct tm tm;
    localtime_r(&tt, &tm);
    struct tm j1 = {0};
    j1.tm_year = tm.tm_year; j1.tm_mon = 0; j1.tm_mday = 1; j1.tm_hour = 12; j1.tm_isdst = -1;
    time_t t0 = mktime(&j1);
    for (int d = 0; d < 366; d++) {
        time_t td = t0 + (time_t)d * 86400;
        struct tm dm;
        localtime_r(&td, &dm);
        if (dm.tm_year != tm.tm_year) break;
        double l, b, r;
        helio(P_EARTH, jd_of((double)td), &l, &b, &r);
        float a = (float)hand_angle(l * RAD), c = cosf(a), s = sinf(a);
        int first = dm.tm_mday == 1, five = dm.tm_mday % 5 == 0;
        float r1 = first ? 3.69f : five ? 3.6f : 3.5f;
        tseg(cx + 3.28f * c, cy + 3.28f * s, cx + r1 * c, cy + r1 * s, first ? 0.03f : 0.014f, 1);
        if (first) tseg(cx + 2.58f * c, cy + 2.58f * s, cx + 3.28f * c, cy + 3.28f * s, 0.02f, 1);
        if (dm.tm_mday == 15 || (dm.tm_mon == 1 && dm.tm_mday == 14)) {
            helio(P_EARTH, jd_of((double)td + 43200), &l, &b, &r);
            float am = (float)hand_angle(l * RAD);
            tarc(st_month, cx, cy, 2.78f, am, MONTH3[dm.tm_mon], 0.04f, k, 1, 1);
        }
    }
    flush_text(tex_dial, DIAL_TEX, DIAL_TEX);
}

/* the plaque: hostname, the date and time, and the time away */
static char host[64];
static time_t locked_at;
static char plaque_key[200];
static void draw_plaque(double wt) {
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    char l2[96], l3[64], key[200];
    snprintf(l2, sizeof l2, "%s * %d %s %d * %02d:%02d", WDAY[tm.tm_wday], tm.tm_mday, MONTH[tm.tm_mon], tm.tm_year + 1900,
             tm.tm_hour, tm.tm_min);
    if (away < 3600) snprintf(l3, sizeof l3, "away %ld min", away / 60);
    else snprintf(l3, sizeof l3, "away %ld h %02ld min", away / 3600, away / 60 % 60);
    snprintf(key, sizeof key, "%s|%s", l2, l3);
    if (!strcmp(key, plaque_key)) return;
    snprintf(plaque_key, sizeof plaque_key, "%s", key);
    set_space(-PLQ_W / 2, -PLQ_H / 2, PLQ_W, PLQ_H);
    const float k = K_PLQ;
    float x0 = -PLQ_W / 2 + 0.95f, x1 = PLQ_W / 2 - 0.95f, y0 = -PLQ_H / 2 + 0.95f, y1 = PLQ_H / 2 - 0.95f;
    for (int i = 0; i < 2; i++) {
        float o = i * 0.13f, w = i ? 0.014f : 0.03f;
        tseg(x0 + o, y0 + o, x1 - o, y0 + o, w, 1); tseg(x1 - o, y0 + o, x1 - o, y1 - o, w, 1);
        tseg(x1 - o, y1 - o, x0 + o, y1 - o, w, 1); tseg(x0 + o, y1 - o, x0 + o, y0 + o, w, 1);
    }
    char hn[64];
    size_t n = strlen(host);
    for (size_t i = 0; i <= n && i < sizeof hn; i++) hn[i] = (char)toupper((unsigned char)host[i]);
    float hw = twidth(st_host, hn, 0.16f, k), hk = k;
    if (hw > 12.5f) hk = k * 12.5f / hw;   /* a long name, set smaller */
    tline(st_host, 0, 0.22f, 0, 0, hn, 0.16f * hk / k, hk, 1);
    tseg(-2.6f, -0.32f, -0.25f, -0.32f, 0.018f, 1);
    tseg(0.25f, -0.32f, 2.6f, -0.32f, 0.018f, 1);
    tseg(-0.16f, -0.32f, 0, -0.32f + 0.12f, 0.03f, 1); tseg(0, -0.32f + 0.12f, 0.16f, -0.32f, 0.03f, 1);
    tseg(0.16f, -0.32f, 0, -0.32f - 0.12f, 0.03f, 1); tseg(0, -0.32f - 0.12f, -0.16f, -0.32f, 0.03f, 1);
    tline(st_line, 0, -1.05f, 0, 0, l2, 0.04f, k, 1);
    tline(st_small, 0, -1.75f, 0, 0, l3, 0.02f, k, 1);
    flush_text(tex_plaque, PLQ_TEX_W, PLQ_TEX_H);
}

/* the date window: the model's date */
static char date_key[64];
static void draw_date(double model_t) {
    time_t ts = (time_t)model_t;
    struct tm tm;
    localtime_r(&ts, &tm);
    char b[64];
    snprintf(b, sizeof b, "%d %s %d", tm.tm_mday, MONTH3[tm.tm_mon], tm.tm_year + 1900);
    if (!strcmp(b, date_key)) return;
    snprintf(date_key, sizeof date_key, "%s", b);
    set_space(0, 0, DATE_RECT[2], DATE_RECT[3]);
    float cap = glyph(st_date, 'H')->top * K_DATE;
    tline(st_date, DATE_RECT[2] * 0.5f, DATE_RECT[3] * 0.5f - cap * 0.5f, 0, 0, b, 0.06f, K_DATE, 1);
    flush_text(tex_date, DATE_TEX_W, DATE_TEX_H);
}

/* ---- shaders ---- */
#define GLSL(...) #__VA_ARGS__

static const char *VS_TEXT = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in float aA;
    uniform vec2 uAtlasSize;
    out vec2 vUV;
    out float vA;
    void main() { vUV = aUV / uAtlasSize; vA = aA; gl_Position = vec4(aPos, 0.0, 1.0); });
static const char *FS_TEXT = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in float vA;
    out vec4 o;
    void main() { o = vec4(vA * texture(uAtlas, vUV).r, 0.0, 0.0, 1.0); });

static const char *VS_SCENE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aP;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec3 aT;
    layout(location = 3) in vec3 aM;          /* material, part x, part y */
    uniform mat4 uModel, uVP;
    invariant gl_Position;
    out vec3 vW, vN, vO, vT;
    out vec2 vUV;
    flat out int vMat;
    void main() {
        vec4 w = uModel * vec4(aP, 1.0);
        vW = w.xyz;
        vN = mat3(uModel) * aN;
        vT = mat3(uModel) * aT;
        vO = aP;
        vUV = aM.yz;
        vMat = int(aM.x + 0.5);
        gl_Position = uVP * w;
    });

static const char *FS_DEPTH = "#version 330 core\n" GLSL(
    void main() {});

/* the scene: lit by the Sun (a spherical area light, with analytic soft
 * shadows from the arms, posts and planets, and the dial plate and plinth
 * shading what's under them) and by moonlight through the window (a shadow
 * map); reflections of a dim study */
static const char *FS_SCENE = GLSL(
    in vec3 vW, vN, vO, vT;
    in vec2 vUV;
    flat in int vMat;
    uniform vec3 uCam, uFwd;
    uniform vec3 uSun, uSunCol;
    uniform float uSunR;
    uniform vec3 uMoonL, uMoonCol;
    uniform sampler2DShadow uShadow;
    uniform mat4 uLVP;
    uniform float uShTexel;
    uniform sampler2D uSunMap;
    uniform mat4 uSVP;
    uniform vec2 uSNF;                         /* the lamp map's near and far */
    uniform float uSTan;                       /* uv per unit of tangent */
    uniform sampler2D uDial, uPlaque, uDate;
    uniform vec4 uDateRect;
    uniform vec3 uPlqC, uPlqU, uPlqV;
    uniform vec2 uPlqSize;
    uniform vec4 uWin;                         /* the window: x0, x1, z0, z1 on the back wall */
    uniform float uWallY;
    layout(location = 0) out vec4 o;
    layout(location = 1) out float oDepth;     /* view depth, for the lens */

    const float PI = 3.14159265;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    float hash3(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
    float noise(vec3 p) {
        vec3 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
                   mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
    }
    float fbm(vec3 p) { return 0.5 * noise(p) + 0.28 * noise(p * 2.03) + 0.16 * noise(p * 4.01) + 0.06; }

    uniform samplerCube uEnv;
    /* the room's reflection, prefiltered: one mip per roughness */
    vec3 env(vec3 r, float rough) { return textureLod(uEnv, r, clamp((rough - 0.04) * 4.0 / 0.96, 0.0, 4.0)).rgb; }
    vec3 winCentre() { return vec3((uWin.x + uWin.y) * 0.5, uWallY, (uWin.z + uWin.w) * 0.5); }
    vec3 lobe(vec3 r, vec3 d, vec3 c, float w, float rough) {
        float s = w * w + rough * rough * 1.6;
        return c * (w * w / s) * exp((dot(r, d) - 1.0) / s);
    }

    /* how much of the lamp a point sees: a shadow map looking straight down
     * from the Sun's centre, a blocker search, then a filter as wide as the
     * penumbra the lamp's size would cast */
    float sunVis(vec3 P, vec3 Ng) {
        if (P.z > 26.6) return 1.0;
        vec4 c = uSVP * vec4(P + Ng * 0.06, 1.0);
        vec3 q = c.xyz / c.w * 0.5 + 0.5;
        if (any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0)))) return 1.0;
        float dr = 27.0 - P.z - 0.04;
        const vec2 k[12] = vec2[](vec2(-0.613, 0.354), vec2(0.170, -0.862), vec2(0.784, 0.373), vec2(-0.265, -0.394),
                                  vec2(0.322, 0.110), vec2(-0.880, -0.260), vec2(0.040, 0.770), vec2(0.580, -0.540),
                                  vec2(-0.45, 0.85), vec2(0.95, -0.05), vec2(-0.2, -0.95), vec2(-0.95, 0.4));
        float rot = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * 6.283;
        mat2 R = mat2(cos(rot), sin(rot), -sin(rot), cos(rot));
        float rs = uSunR * max(1.0 / 5.0 - 1.0 / dr, 0.0) * uSTan;
        float bsum = 0.0, bn = 0.0;
        for (int i = 0; i < 4; i++) {
            float d = texture(uSunMap, q.xy + R * k[i * 3] * rs).r;
            float zb = uSNF.x * uSNF.y / (uSNF.y - d * (uSNF.y - uSNF.x));
            if (zb < dr) { bsum += zb; bn += 1.0; }
        }
        if (bn == 0.0) return 1.0;
        float db = bsum / bn;
        float pr = max(uSunR * (dr - db) / (dr * db) * uSTan, 1.5 / 2048.0);
        float vis = 0.0;
        for (int i = 0; i < 8; i++) {
            float d = texture(uSunMap, q.xy + R * k[i] * pr).r;
            float zb = uSNF.x * uSNF.y / (uSNF.y - d * (uSNF.y - uSNF.x));
            vis += zb < dr ? 0.0 : 1.0;
        }
        return smoothstep(0.0, 1.0, vis / 8.0);
    }

    float moonShadow(vec3 P, vec3 n) {
        vec4 l = uLVP * vec4(P + n * 0.08 + uMoonL * 0.05, 1.0);
        vec3 c = l.xyz * 0.5 + 0.5;
        if (any(lessThan(c.xy, vec2(0.0))) || any(greaterThan(c.xy, vec2(1.0)))) return 1.0;
        float rot = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 7.0, vec2(0.06711056, 0.00583715)))) * 6.283;
        vec2 a = vec2(cos(rot), sin(rot)) * uShTexel * 2.2, b = vec2(-a.y, a.x);
        float s = texture(uShadow, vec3(c.xy + a, c.z - 0.0006)) + texture(uShadow, vec3(c.xy - a, c.z - 0.0006)) +
                  texture(uShadow, vec3(c.xy + b * 0.5, c.z - 0.0006)) + texture(uShadow, vec3(c.xy - b * 0.5, c.z - 0.0006));
        return s * 0.25;
    }

    /* anisotropic GGX: D * Vis * F for light L */
    vec3 ggx(vec3 N, vec3 T, vec3 B, vec3 V, vec3 L, vec3 F0, float ax, float ay) {
        vec3 Hh = normalize(L + V);
        float NoL = max(dot(N, L), 0.0), NoV = max(dot(N, V), 1e-3);
        float ToH = dot(T, Hh), BoH = dot(B, Hh), NoH = dot(N, Hh);
        float d = ToH * ToH / (ax * ax) + BoH * BoH / (ay * ay) + NoH * NoH;
        float D = 1.0 / (PI * ax * ay * d * d);
        float a2 = ax * ay;
        float Vis = 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2) + 1e-5);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(L, Hh), 0.0), 5.0);
        return D * Vis * F * NoL;
    }

    struct Mat { vec3 F0, alb, emit; float rT, rB, coat, coatR; };

    /* light a surface: the Sun (sphere light), the moonlight, the room */
    vec3 light(Mat m, vec3 P, vec3 N, vec3 Ng, vec3 T, vec3 V, float ao) {
        vec3 B = normalize(cross(N, T));
        T = cross(B, N);
        vec3 col = m.emit;
        float NoV = max(dot(N, V), 1e-3);
        /* the Sun */
        vec3 Lc = uSun - P;
        float d2 = dot(Lc, Lc), dist = sqrt(d2);
        vec3 L = Lc / dist;
        float sv = (dot(Ng, L) > -0.2) ? sunVis(P, Ng) : 0.0;
        vec3 E = uSunCol / d2 * sv;
        if (sv > 0.0) {
            /* the representative point on the sphere for the highlight, and
             * the roughness widened to keep the energy */
            vec3 R = reflect(-V, N);
            vec3 cr = dot(Lc, R) * R - Lc;
            vec3 Ls = normalize(Lc + cr * clamp(uSunR / max(length(cr), 1e-4), 0.0, 1.0));
            float ax = max(m.rT * m.rT, 0.004), ay = max(m.rB * m.rB, 0.004);
            float w = uSunR / (2.0 * dist);
            float ax2 = min(ax + w, 1.0), ay2 = min(ay + w, 1.0);
            col += ggx(N, T, B, V, Ls, m.F0, ax2, ay2) * (ax * ay / (ax2 * ay2)) * E;
            col += m.alb * max(dot(N, L), 0.0) * E / PI;
            if (m.coat > 0.0) {
                float c2 = m.coatR * m.coatR, cw = min(c2 + w, 1.0);
                col += ggx(N, T, B, V, Ls, vec3(0.04), cw, cw) * (c2 * c2 / (cw * cw)) * E * m.coat;
            }
        }
        /* moonlight through the window */
        float NoLm = dot(N, uMoonL);
        if (NoLm > 0.0 || dot(Ng, uMoonL) > 0.0) {
            float ms = moonShadow(P, Ng);
            float ax = max(m.rT * m.rT, 0.01), ay = max(m.rB * m.rB, 0.01);
            float wm = 0.06;
            col += (ggx(N, T, B, V, uMoonL, m.F0, min(ax + wm, 1.0), min(ay + wm, 1.0)) * (ax * ay / ((ax + wm) * (ay + wm))) +
                    m.alb * max(NoLm, 0.0) / PI) * uMoonCol * ms;
        }
        /* the room: reflections bent along the grain, and a little fill */
        float r = sqrt(m.rT * m.rB);
        vec3 aniDir = m.rT < m.rB ? B : T;
        float an = abs(m.rB - m.rT) / max(m.rB + m.rT, 1e-3);
        vec3 aT = cross(aniDir, V), aN = cross(aT, aniDir);
        vec3 Nb = normalize(mix(N, aN, clamp(an * 0.8, 0.0, 0.9)));
        vec3 Rr = reflect(-V, Nb);
        vec3 Fe = m.F0 + (max(vec3(1.0 - r), m.F0) - m.F0) * pow(1.0 - NoV, 5.0);
        col += env(Rr, r) * Fe * ao;
        /* inside the train: the lamp-lit base plate below reflects warm light up into the wheels */
        float inTrain = smoothstep(15.0, 12.0, length(P.xy)) * smoothstep(15.0, 13.5, P.z) * step(4.4, P.z);
        if (inTrain > 0.0) {
            vec3 warm = vec3(1.0, 0.66, 0.34);
            col += (lobe(Rr, vec3(0.0, 0.0, -1.0), warm * 0.55, 0.75, r) * Fe + m.alb * warm * 0.25 * max(0.3 - N.z * 0.7, 0.0)) * inTrain;
        }
        col += m.alb * mix(vec3(0.03, 0.019, 0.011), vec3(0.016, 0.014, 0.014), N.z * 0.5 + 0.5) * ao * 2.2;
        if (m.coat > 0.0) col += env(reflect(-V, N), m.coatR) * (0.04 + 0.96 * pow(1.0 - NoV, 5.0)) * ao * m.coat;
        /* the light bouncing off the bright table round the base */
        float rho = length(P.xy);
        float bounce = smoothstep(60.0, 22.0, rho) * smoothstep(30.0, 2.0, P.z) * max(-N.z * 0.6 + 0.4, 0.0);
        col += (m.alb + m.F0 * 0.35) * vec3(1.0, 0.6, 0.3) * 0.05 * bounce * ao;
        return col;
    }

    /* engraving: the mask's gradient bends the normal at the cut's walls */
    float mask(sampler2D t, vec2 uv) { return texture(t, uv).r; }

    vec3 woodCol(vec3 p, vec3 dark, vec3 light, float scale, out float fig) {
        vec3 q = p * scale;
        float wob = fbm(q * vec3(0.06, 1.0, 1.0) + vec3(0.0, 3.1, 7.7));
        float rings = q.y * 2.4 + q.z * 0.4 + wob * 5.0 + 0.6 * sin(q.x * 0.05);
        float g = fract(rings);
        g = smoothstep(0.0, 0.25, g) * smoothstep(1.0, 0.55, g);
        float pores = noise(q * vec3(0.4, 22.0, 22.0));
        fig = g;
        vec3 c = mix(dark, light, g * 0.7 + 0.3 * fbm(q * vec3(0.03, 0.5, 0.5)));
        return c * (0.85 + 0.15 * pores);
    }

    vec3 room(vec3 P, vec3 N) {
        vec3 c = vec3(0.0);
        float d2 = dot(uSun - P, uSun - P);
        vec3 lamp = uSunCol / d2 * max(dot(N, normalize(uSun - P)), 0.0);
        vec3 alb;
        if (N.z > 0.5) {                       /* floor boards, and moonlight through the window on them */
            float b = floor(P.x / 14.0);
            alb = vec3(0.05, 0.03, 0.018) * (0.7 + 0.5 * hash(vec2(b, 3.0))) * (0.8 + 0.2 * noise(vec3(P.x * 0.3, P.y * 0.01, 0.0)));
            vec3 q = P + uMoonL * ((uWallY - P.y) / uMoonL.y);
            float inWin = step(uWin.x, q.x) * step(q.x, uWin.y) * step(uWin.z, q.z) * step(q.z, uWin.w);
            float mull = step(0.06, abs(fract((q.x - uWin.x) / (uWin.y - uWin.x) * 2.0) - 0.5) * 2.0) *
                         step(0.05, abs(fract((q.z - uWin.z) / (uWin.w - uWin.z) * 3.0) - 0.5) * 2.0);
            c += alb * uMoonCol * 0.9 * inWin * mull * max(uMoonL.z, 0.0);
        } else if (N.z < -0.5) {
            alb = vec3(0.03, 0.025, 0.02);
        } else {
            float h = P.z;
            if (h < 18.0) {                    /* panelling */
                alb = vec3(0.07, 0.035, 0.018);
                float u = abs(N.y) > 0.5 ? P.x : P.y;
                float pan = abs(fract(u / 60.0) - 0.5) * 2.0;
                alb *= 0.75 + 0.25 * smoothstep(0.9, 0.96, pan) + 0.2 * smoothstep(-40.0, -38.0, h) * (1.0 - smoothstep(14.0, 16.0, h));
            } else {                           /* dark green paper with a faint damask */
                vec2 u = abs(N.y) > 0.5 ? P.xz : P.yz;
                float dm = smoothstep(0.35, 0.5, abs(sin(u.x * 0.25) * sin(u.y * 0.18)));
                alb = vec3(0.022, 0.035, 0.028) * (0.85 + 0.25 * dm);
            }
            if (N.y < -0.5) {                  /* the back wall: the window and the bookcase */
                if (P.x > uWin.x - 12.0 && P.x < uWin.y + 12.0 && h > uWin.z - 12.0 && h < uWin.w + 14.0) {
                    alb = vec3(0.06, 0.04, 0.025);
                    if (P.x > uWin.x && P.x < uWin.y && h > uWin.z && h < uWin.w) {
                        vec2 q = vec2((P.x - uWin.x) / (uWin.y - uWin.x), (h - uWin.z) / (uWin.w - uWin.z));
                        float mx = abs(fract(q.x * 2.0) - 0.5) * 2.0, mz = abs(fract(q.y * 3.0) - 0.5) * 2.0;
                        float glass = step(0.035, mx) * step(0.03, mz) * step(0.02, q.x) * step(q.x, 0.98) * step(0.02, q.y) * step(q.y, 0.98);
                        /* moonlit sky: a gradient, a few clouds, bare branches */
                        vec3 sky = mix(vec3(0.08, 0.13, 0.24), vec3(0.20, 0.28, 0.42), q.y);
                        float cl = fbm(vec3(q * vec2(3.0, 5.0), 1.0));
                        sky = mix(sky, vec3(0.35, 0.40, 0.50), smoothstep(0.55, 0.8, cl) * 0.6);
                        float br = smoothstep(0.62, 0.66, fbm(vec3(q.x * 9.0, q.y * 3.0 + q.x * 2.0, 4.0))) * (1.0 - q.y * 0.5);
                        sky *= 1.0 - 0.8 * br * step(q.y, 0.75);
                        float moon = smoothstep(0.035, 0.03, length((q - vec2(0.7, 0.78)) * vec2(1.0, 1.6)));
                        c += (sky * 2.2 + vec3(4.0, 4.2, 4.4) * moon) * glass;
                        alb = vec3(0.04, 0.03, 0.02) * (1.0 - glass);
                    }
                } else if (P.x > -20.0 && P.x < 300.0 && h < 175.0) {
                    float sh = floor((h + 76.0) / 36.0), hy = fract((h + 76.0) / 36.0) * 36.0;
                    float bay = abs(fract((P.x + 20.0) / 80.0) - 0.5) * 80.0;
                    alb = vec3(0.05, 0.028, 0.015);
                    float bx = P.x + sh * 37.0;
                    float bi = floor(bx / 3.4);
                    float bh = 22.0 + 9.0 * hash(vec2(bi, sh));
                    if (bay < 37.5 && hy > 2.5 && hy < bh && hash(vec2(bi, sh + 9.0)) > 0.1) {
                        float hu = hash(vec2(bi, sh + 2.0));
                        vec3 bc = hu < 0.3 ? vec3(0.16, 0.035, 0.03) : hu < 0.55 ? vec3(0.04, 0.08, 0.05) : hu < 0.75 ? vec3(0.04, 0.05, 0.11) : vec3(0.17, 0.11, 0.06);
                        bc *= 0.6 + 0.6 * hash(vec2(bi, sh + 4.0));
                        float band = step(0.8, fract(hy / 4.0 + hash(vec2(bi, 1.0)))) * step(hy, bh - 3.0) * step(3.0, hy);
                        alb = mix(bc, vec3(0.35, 0.25, 0.1), band * 0.5);
                    }
                }
            }
        }
        c += alb * (lamp + vec3(0.05, 0.055, 0.07) * 0.35 + vec3(0.09, 0.065, 0.045) * smoothstep(-80.0, 120.0, P.z));
        return c;
    }

    void main() {
        vec3 Ng = normalize(vN);
        vec3 V = normalize(uCam - vW);
        if (!gl_FrontFacing) Ng = -Ng;
        vec3 N = Ng;
        vec3 T = vT - N * dot(vT, N);
        T = dot(T, T) > 1e-6 ? normalize(T) : normalize(cross(N, vec3(0.31, 0.53, 0.79)));
        int mt = vMat;
        Mat m;
        m.F0 = vec3(0.04); m.alb = vec3(0.0); m.emit = vec3(0.0); m.rT = 0.3; m.rB = 0.3; m.coat = 0.0; m.coatR = 0.1;
        float ao = 1.0;
        /* inside the train: less of the room reaches */
        float rho = length(vW.xy);
        if (vW.z > 4.4 && vW.z < 14.1 && rho < 14.6) ao = mix(0.6, 1.0, smoothstep(9.0, 14.6, rho));
        vec3 P = vW;

        if (KIND == 0) {                        /* the room: matte, far away */
            o = vec4(room(P, N), 1.0);
            oDepth = dot(vW - uCam, uFwd);
            return;
        }
        if (KIND == 1) {                        /* the walnut table, waxed */
            float fig;
            m.alb = woodCol(vW, vec3(0.045, 0.02, 0.009), vec3(0.15, 0.075, 0.034), 0.22, fig);
            m.F0 = vec3(0.03);
            m.rT = m.rB = 0.55;
            m.coat = 1.0;
            m.coatR = 0.2 + 0.06 * noise(vW * vec3(0.05, 0.4, 0.4)) + 0.08 * smoothstep(0.55, 0.8, fbm(vW * 0.04));
            o = vec4(min(light(m, P, N, Ng, T, V, 1.0), vec3(60.0)), 1.0);
            oDepth = dot(vW - uCam, uFwd);
            return;
        }
        if (mt == 9) {                          /* the Sun: a frosted globe with a lamp in it */
            vec3 q = normalize(vO - vec3(0.0, 0.0, 27.0));
            float NoV = max(dot(N, V), 0.0);
            float fr = 0.6 + 0.5 * fbm(q * 6.0) ;
            vec3 glow = vec3(1.0, 0.62, 0.28) * (3.0 + 2.6 * pow(NoV, 2.0)) * fr + vec3(1.0, 0.85, 0.6) * 9.0 * pow(NoV, 10.0);
            vec3 R = reflect(-V, N);
            vec3 col = glow + env(R, 0.05) * (0.04 + 0.96 * pow(1.0 - NoV, 5.0)) * 3.0;
            o = vec4(col, 1.0);
            oDepth = dot(vW - uCam, uFwd);
            return;
        }
        bool top = N.z > 0.93;
        if (KIND == 2 && mt == 0) {                          /* brass, turned and grained */
            m.F0 = vec3(0.92, 0.66, 0.34);
            m.rT = 0.08; m.rB = 0.32;
            float ring = hash(vec2(floor(length(vUV) * 9.0), 3.0));
            m.F0 *= 0.92 + 0.1 * ring;
        } else if (KIND == 2 && mt == 1) {                   /* the base plate: older brass, frosted */
            m.F0 = vec3(0.55, 0.40, 0.22);
            m.rT = 0.3; m.rB = 0.42;
            m.F0 *= 0.88 + 0.16 * noise(vW * 0.8);
        } else if (KIND == 2 && mt == 2) {                   /* polished steel */
            m.F0 = vec3(0.62, 0.62, 0.64);
            m.rT = 0.05; m.rB = 0.16;
        } else if (KIND == 2 && mt == 3) {                   /* blued steel */
            m.F0 = vec3(0.06, 0.09, 0.26);
            m.rT = 0.1; m.rB = 0.14;
        } else if (KIND == 2 && mt == 4) {                   /* gilt */
            m.F0 = vec3(1.0, 0.77, 0.40);
            m.rT = 0.06; m.rB = 0.12;
        } else if (KIND == 2 && mt == 5) {                   /* the dial plate: silvered, engraved; brass under it */
            if (top) {
                vec2 uv = vW.xy / 29.0 + 0.5;
                float ink = mask(uDial, uv);
                /* the cut's walls: the ink's slope across the plate, in texels */
                vec2 dx = dFdx(vW.xy), dy = dFdy(vW.xy);
                vec2 gi = vec2(dFdx(ink), dFdy(ink));
                mat2 J = mat2(dx, dy);
                vec2 gw = abs(determinant(J)) > 1e-12 ? gi * inverse(J) : vec2(0.0);
                gw *= 0.6 * 29.0 / 4096.0;
                N = normalize(N + vec3(-gw, 0.0) * 1.2);
                m.F0 = vec3(0.70, 0.68, 0.63) * (0.94 + 0.08 * noise(vW * 6.0));
                m.rT = 0.24; m.rB = 0.34;
                T = normalize(vec3(-vW.y, vW.x, 0.0) + 1e-4);
                /* the date window: an ivory card under glass */
                vec2 dq = (vW.xy - uDateRect.xy) / uDateRect.zw;
                if (all(greaterThan(dq, vec2(0.0))) && all(lessThan(dq, vec2(1.0)))) {
                    float di = texture(uDate, dq).r;
                    m.F0 = vec3(0.04);
                    m.alb = mix(vec3(0.62, 0.55, 0.42), vec3(0.02, 0.018, 0.015), di);
                    m.rT = m.rB = 0.5; m.coat = 1.0; m.coatR = 0.03;
                    ao = 0.8 * smoothstep(0.0, 0.06, min(min(dq.x, 1.0 - dq.x) * 4.0, min(dq.y, 1.0 - dq.y)));
                    N = Ng;
                    ink = 0.0;
                }
                m.F0 = mix(m.F0, vec3(0.025), ink);
                m.alb = mix(m.alb, vec3(0.008, 0.007, 0.006), ink);
                m.rT = mix(m.rT, 0.55, ink); m.rB = mix(m.rB, 0.55, ink);
            } else {
                m.F0 = vec3(0.85, 0.62, 0.32); m.rT = 0.12; m.rB = 0.3;
            }
        } else if (KIND == 2 && mt == 6) {                   /* mahogany: wax over wood */
            float fig;
            vec3 wp = vec3(atan(vW.y, vW.x) * 20.0, length(vW.xy), vW.z);
            if (abs(vW.y + 34.0) < 4.0 && abs(vW.x) < 9.5 && vW.z < 5.5) wp = vW * 1.6;
            m.alb = woodCol(wp, vec3(0.07, 0.016, 0.008), vec3(0.20, 0.06, 0.025), 0.6, fig);
            m.F0 = vec3(0.03);
            m.rT = m.rB = 0.55;
            m.coat = 1.0;
            m.coatR = 0.09;
        } else if ((KIND == 3 || mt == 16) && mt >= 10 && mt <= 16) {      /* the planets: enamel, clear-coated */
            vec3 q = normalize(mt == 16 ? vO - vec3(2.5, 0.0, 27.0) : vO);
            float lat = q.z;
            if (KIND == 3 && mt == 10) {                     /* Mercury: grey, pocked */
                m.alb = vec3(0.30, 0.28, 0.26) * (0.75 + 0.45 * fbm(q * 7.0));
            } else if (KIND == 3 && mt == 11) {              /* Venus: pale cloud */
                m.alb = vec3(0.70, 0.60, 0.40) * (0.88 + 0.16 * fbm(vec3(q.x * 3.0, q.y * 3.0, lat * 9.0)));
            } else if (KIND == 3 && mt == 12) {              /* the Earth: blue enamel, gilt lines */
                m.alb = vec3(0.012, 0.045, 0.16);
                m.alb *= 0.85 + 0.25 * fbm(q * 5.0);
                float la = asin(clamp(lat, -1.0, 1.0)) * 57.2958, lo = atan(q.y, q.x) * 57.2958;
                float wl = fwidth(la) * 0.9 + 0.25, wo = fwidth(lo) * 0.9 + 0.25 / max(sqrt(1.0 - lat * lat), 0.2);
                float g = 0.0;
                g = max(g, smoothstep(wl * 1.6, 0.0, abs(la)));
                g = max(g, smoothstep(wl, 0.0, abs(abs(la) - 23.44)));
                g = max(g, smoothstep(wl, 0.0, abs(abs(la) - 66.56)));
                g = max(g, smoothstep(wo, 0.0, abs(mod(lo + 15.0, 30.0) - 15.0)) * step(abs(la), 80.0));
                m.alb = mix(m.alb, vec3(0.0), g);
                m.F0 = mix(vec3(0.04), vec3(1.0, 0.77, 0.4), g);
                m.rT = m.rB = mix(0.4, 0.15, g);
            } else if (KIND == 3 && mt == 13) {              /* Mars: terracotta, dark seas, a polar cap */
                float f = fbm(q * 4.0 + 2.0);
                m.alb = mix(vec3(0.42, 0.13, 0.05), vec3(0.18, 0.07, 0.04), smoothstep(0.5, 0.65, f));
                m.alb = mix(m.alb, vec3(0.75, 0.72, 0.68), smoothstep(0.88, 0.92, lat));
            } else if (KIND == 3 && mt == 14) {              /* Jupiter: belts and zones, the red spot */
                float t = lat + 0.03 * fbm(q * 8.0) - 0.02 * sin(atan(q.y, q.x) * 3.0 + lat * 20.0) * 0.5;
                float b = sin(t * 26.0) * 0.5 + 0.5, b2 = sin(t * 61.0 + 1.0) * 0.5 + 0.5;
                vec3 zone = vec3(0.72, 0.62, 0.46), belt = vec3(0.38, 0.22, 0.12);
                m.alb = mix(belt, zone, smoothstep(0.3, 0.7, b) * 0.8 + 0.2 * b2);
                m.alb = mix(m.alb, vec3(0.62, 0.55, 0.42), smoothstep(0.45, 0.7, abs(lat)));
                vec2 sp = vec2(atan(q.y, q.x) - 1.2, (lat + 0.36) * 3.2);
                m.alb = mix(m.alb, vec3(0.55, 0.20, 0.10), smoothstep(0.32, 0.24, length(sp * vec2(1.0, 1.7))));
            } else if (KIND == 3 && mt == 15) {              /* Saturn: pale gold bands */
                float b = sin(lat * 30.0 + 0.6 * fbm(q * 6.0)) * 0.5 + 0.5;
                m.alb = mix(vec3(0.62, 0.50, 0.30), vec3(0.74, 0.64, 0.44), b);
                m.alb = mix(m.alb, vec3(0.50, 0.46, 0.38), smoothstep(0.75, 0.9, abs(lat)));
            } else {                            /* the Moon: grey-white, its seas on the side to the Earth */
                float f = fbm(q * 5.0 + 7.0);
                m.alb = vec3(0.58, 0.57, 0.54) * (1.0 - 0.45 * smoothstep(0.45, 0.62, f) * smoothstep(0.0, -0.6, q.x));
            }
            if (KIND != 3 || mt != 12) { m.F0 = vec3(0.04); m.rT = m.rB = 0.45; }
            m.coat = 1.0; m.coatR = 0.05;
            ao = 1.0;
        } else if (KIND == 3 && mt == 17) {                  /* Saturn's rings: enamelled bands */
            float r = length(vO.xy);
            float b = 0.5 + 0.5 * sin(r * 40.0) * sin(r * 13.0 + 1.0);
            m.alb = mix(vec3(0.46, 0.40, 0.30), vec3(0.70, 0.62, 0.48), b) * mix(0.6, 1.0, smoothstep(2.45, 2.9, r));
            m.F0 = vec3(0.04); m.rT = m.rB = 0.4; m.coat = 1.0; m.coatR = 0.06;
        } else if (KIND == 2 && mt == 18) {                  /* the plaque: brushed brass, engraved */
            vec2 uv = vUV / uPlqSize + 0.5;
            m.F0 = vec3(0.90, 0.66, 0.34);
            m.rT = 0.06; m.rB = 0.3;
            T = uPlqU;
            if (dot(Ng, cross(uPlqU, uPlqV)) > 0.9) {
                float ink = mask(uPlaque, uv);
                vec2 dx = dFdx(vUV), dy = dFdy(vUV);
                vec2 gi = vec2(dFdx(ink), dFdy(ink));
                mat2 J = mat2(dx, dy);
                vec2 gw = abs(determinant(J)) > 1e-12 ? gi * inverse(J) : vec2(0.0);
                gw *= 1.2 * 16.0 / 1536.0;
                N = normalize(N - (uPlqU * gw.x + uPlqV * gw.y) * 1.2);
                m.F0 = mix(m.F0, vec3(0.025), ink);
                m.alb = mix(m.alb, vec3(0.008, 0.007, 0.006), ink);
                m.rT = mix(m.rT, 0.55, ink); m.rB = mix(m.rB, 0.55, ink);
            }
        } else if (KIND == 2 && mt == 21) {                  /* Jupiter's moons: little ivory balls */
            m.alb = vec3(0.62, 0.56, 0.45);
            m.coat = 1.0; m.coatR = 0.05;
            m.rT = m.rB = 0.5;
        } else {                                /* slots */
            m.alb = vec3(0.004); m.F0 = vec3(0.02); m.rT = m.rB = 0.6;
        }
        vec3 col = light(m, P, N, Ng, T, V, ao);
        o = vec4(min(col, vec3(60.0)), 1.0);
        oDepth = dot(vW - uCam, uFwd);
    });

/* prefiltering the room's reflection into a cubemap: the analytic lobes,
 * widened by each mip's roughness */
static const char *FS_ENVGEN = "#version 330 core\n" GLSL(
    uniform vec4 uWin;
    uniform float uWallY, uRough;
    uniform int uFace;
    uniform float uSize;
    out vec4 o;
    vec3 winCentre() { return vec3((uWin.x + uWin.y) * 0.5, uWallY, (uWin.z + uWin.w) * 0.5); }

    vec3 lobe(vec3 r, vec3 d, vec3 c, float w, float rough) {
        float s = w * w + rough * rough * 1.6;
        return c * (w * w / s) * exp((dot(r, d) - 1.0) / s);
    }
    /* what a shiny surface near the orrery sees: the dim study, the moonlit
     * window, the lamp-lit table below (the Sun itself is the area light) */
    vec3 envA(vec3 r, float rough) {
        vec3 c = vec3(0.012, 0.009, 0.007) * (0.8 + 0.4 * r.z);
        c += lobe(r, normalize(winCentre() - vec3(0.0, 0.0, 20.0)), vec3(0.42, 0.56, 0.85) * 0.9, 0.11, rough);
        c += lobe(r, vec3(0.0, 0.0, -1.0), vec3(0.075, 0.042, 0.02), 0.55, rough);
        c += lobe(r, normalize(vec3(0.75, -0.55, 0.35)), vec3(1.0, 0.62, 0.32) * 0.10, 0.35, rough);
        c += lobe(r, normalize(vec3(-0.3, -0.9, 0.25)), vec3(0.9, 0.6, 0.35) * 0.05, 0.5, rough);
        c += lobe(r, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.72, 0.45) * 0.09, 0.7, rough);      /* the ceiling, lit by the lamp */
        c += lobe(r, normalize(vec3(0.1, -1.0, 0.45)), vec3(1.0, 0.75, 0.5) * 0.07, 0.35, rough);   /* a pale wall behind the camera */
        return c;
    }


    void main() {
        vec2 q = gl_FragCoord.xy / uSize * 2.0 - 1.0;
        vec3 d;
        if (uFace == 0) d = vec3(1.0, -q.y, -q.x);
        else if (uFace == 1) d = vec3(-1.0, -q.y, q.x);
        else if (uFace == 2) d = vec3(q.x, 1.0, q.y);
        else if (uFace == 3) d = vec3(q.x, -1.0, -q.y);
        else if (uFace == 4) d = vec3(q.x, -q.y, 1.0);
        else d = vec3(-q.x, -q.y, -1.0);
        o = vec4(envA(normalize(d), uRough), 1.0);
    });

static const char *VS_TRI = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* depth of field, at half resolution: a gather over a golden-angle disc
 * sized by the circle of confusion */
static const char *FS_PREP = "#version 330 core\n" GLSL(
    uniform sampler2D uCol;
    uniform sampler2DMS uDep;
    uniform vec2 uSize;
    uniform vec3 uLens;                       /* focus depth, aperture * focal length (px * cm), max */
    uniform int uLast;                        /* the last MSAA sample */
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy * 2.0 / uSize;
        ivec2 p = ivec2(gl_FragCoord.xy) * 2;
        float z = min(min(texelFetch(uDep, p, 0).r, texelFetch(uDep, p + ivec2(1, 0), uLast).r),
                      min(texelFetch(uDep, p + ivec2(0, 1), uLast).r, texelFetch(uDep, p + ivec2(1), 0).r));
        float c = clamp(uLens.y * (1.0 / uLens.x - 1.0 / max(z, 1.0)), -uLens.z, uLens.z);
        o = vec4(texture(uCol, uv).rgb, c);
    });
static const char *FS_DOF = "#version 330 core\n" GLSL(
    uniform sampler2D uPrep;
    uniform vec2 uHalf;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / uHalf;
        vec4 c = texture(uPrep, uv);
        float R = abs(c.a) * 0.5;
        if (R < 0.4) { o = c; return; }
        vec3 acc = c.rgb;
        float ws = 1.0;
        for (int i = 0; i < 16; i++) {
            float fi = float(i) + 0.5, rr = sqrt(fi / 16.0) * R, a = fi * 2.39996;
            vec4 s = texture(uPrep, uv + vec2(cos(a), sin(a)) * rr / uHalf);
            /* a sharper sample in front doesn't bleed back; one behind is
             * only gathered as far as its own blur reaches */
            float w = clamp(abs(s.a) * 0.5 - rr + 1.0, 0.0, 1.0);
            acc += s.rgb * w;
            ws += w;
        }
        o = vec4(acc / ws, c.a);
    });

static const char *FS_FINAL = "#version 330 core\n" GLSL(
    uniform sampler2D uCol, uDof, uPrep;
    uniform vec2 uOut, uSize;
    uniform vec4 uVig;                         /* this view in the camera's frame: x0, y0, w, h, in units of the front's short side */
    uniform float uTime, uExposure, uFade;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
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
        vec2 uv = gl_FragCoord.xy / uOut;
        vec3 sharp = uSize.x < uOut.x - 0.5 ? catmull(uCol, uv) : texture(uCol, uv).rgb;
        float c = texture(uPrep, uv).a;
        vec3 blur = texture(uDof, uv).rgb;
        vec3 col = mix(sharp, blur, smoothstep(0.6, 2.0, abs(c)));
        /* bloom: wide and soft round the Sun, the window, the highlights */
        vec3 bl = textureLod(uDof, uv, 2.0).rgb * 0.35 + textureLod(uDof, uv, 4.0).rgb * 0.4 + textureLod(uDof, uv, 6.0).rgb * 0.25;
        col += max(bl - 0.25, 0.0) * 0.22;
        col *= uExposure;
        /* grade: warm highlights, cool deep shadows */
        float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = mix(col * vec3(0.92, 0.98, 1.12), col * vec3(1.05, 1.0, 0.92), smoothstep(0.0, 0.25, lum));
        col = aces(col);
        /* vignette, in the shared camera's frame so it spans the wall */
        vec2 q = uVig.xy + uv * uVig.zw;
        col *= mix(1.0, 0.55, smoothstep(0.45, 1.25, length(q)));
        col = pow(col, vec3(1.0 / 2.2));
        col = mix(col, vec3(dot(col, vec3(0.33))), 0.06);
        col *= uFade;
        col += (hash(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5) * (2.0 / 255.0);
        o = vec4(col, 1.0);
    });

/* ---- the show: when the model is wound, and what the camera looks at ----
 * Time runs in shots of SHOT_LEN seconds, four to an act: two holding the
 * present, then (in most acts) one winding the model forward and one
 * running it back to now. Everything is a function of the time since the
 * epoch and the seed, so separate instances agree. */
#define SHOT_LEN 20.0
enum { S_WIDE, S_HIGH, S_LOW, S_PLAQUE, S_TRAIN, S_DIAL, S_SUN, S_PLANET, S_EARTH, S_FLYBY, NSHOT };
static const char *SHOT_NAME[NSHOT] = {"wide", "high", "low", "plaque", "train", "dial", "sun", "planet", "earth", "flyby"};

static double smoother(double x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return x * x * x * (x * (x * 6 - 15) + 10); }

/* how far (days) act a winds the model forward; 0 for a quiet act */
static double wind_days(int act) {
    if (act == 0) return 0;
    float r = h01((uint32_t)act, 11);
    if (r < 0.3f) return 0;
    static const double D[6] = {35, 80, 150, 240, 365, 520};
    return D[(int)(h01((uint32_t)act, 12) * 6) % 6];
}

/* the model's offset from the present, days, at show time t */
static double model_offset(double t) {
    if (t < 0) return 0;
    long k = (long)floor(t / SHOT_LEN);
    double u = t - k * SHOT_LEN;
    double D = wind_days((int)(k / 4));
    if (k % 4 == 2) return D * smoother((u - 1.5) / 16.0);
    if (k % 4 == 3) return D * (1 - smoother((u - 1.0) / 16.5));
    return 0;
}

static int shot_type(long k) {
    if (opt_shot_type >= 0) return opt_shot_type;
    if (k <= 0) return S_WIDE;
    static const int hold0[] = {S_WIDE, S_LOW, S_HIGH, S_FLYBY, S_WIDE};
    static const int hold1[] = {S_PLANET, S_EARTH, S_PLAQUE, S_SUN, S_FLYBY, S_PLANET};
    static const int wind[] = {S_TRAIN, S_DIAL, S_WIDE, S_HIGH, S_SUN, S_TRAIN, S_DIAL};
    static const int unwind[] = {S_PLANET, S_HIGH, S_TRAIN, S_DIAL, S_WIDE};
    const int *pool;
    int n;
    int slot = (int)(k % 4), quiet = wind_days((int)(k / 4)) == 0;
    if (slot == 0) { pool = hold0; n = 5; }
    else if (slot == 1 || quiet) { pool = hold1; n = 6; }
    else if (slot == 2) { pool = wind; n = 7; }
    else { pool = unwind; n = 5; }
    int s = pool[(int)(h01((uint32_t)k, 1) * n) % n];
    int prev = pool[(int)(h01((uint32_t)k - 1, 1) * n) % n];
    if (s == prev) s = pool[((int)(h01((uint32_t)k, 1) * n) + 1) % n];
    return s;
}

typedef struct { float eye[3], tgt[3], fov, focus, ap, fp[3]; int has_fp; } Cam;

static void orbit(Cam *c, const float *tgt, float az, float el, float dist) {
    memcpy(c->tgt, tgt, sizeof c->tgt);
    c->eye[0] = tgt[0] + dist * cosf(el) * cosf(az);
    c->eye[1] = tgt[1] + dist * cosf(el) * sinf(az);
    c->eye[2] = tgt[2] + dist * sinf(el);
}

/* the camera at show time t, for a model in state s */
static int cur_shot;
static void camera_at(double t, const Sky *s, Cam *c) {
    long k = (long)floor(t / SHOT_LEN);
    double ul = (t - k * SHOT_LEN) / SHOT_LEN;
    float u = (float)ul, us = (float)smoother(ul * 0.9 + 0.05);
    int type = shot_type(k);
    cur_shot = type;
    uint32_t kk = (uint32_t)k;
    float r1 = h01(kk, 2), r2 = h01(kk, 3), r3 = h01(kk, 4), r4 = h01(kk, 5);
    float sgn = r4 < 0.5f ? -1.0f : 1.0f;
    c->fov = 30; c->ap = 0.9f; c->has_fp = 0;
    const float F = (float)(-PI / 2);   /* the front */
    switch (type) {
    case S_WIDE: {
        float tgt[3] = {0, 0, 16.5f};
        float az = F + (r1 - 0.5f) * 2.2f + sgn * (u - 0.5f) * 0.28f, el = (float)((12 + 10 * r2) * RAD);
        orbit(c, tgt, az, el, 122 - 12 * us);
        c->ap = 0.7f;
        break;
    }
    case S_HIGH: {
        float tgt[3] = {0, 0, 14};
        float az = (float)TAU * r1 + sgn * u * 0.35f, el = (float)((36 + 12 * r2) * RAD);
        orbit(c, tgt, az, el, 112 - 12 * us);
        c->ap = 0.8f;
        break;
    }
    case S_LOW: {
        float az = F + sgn * (0.45f + 0.5f * r1) + sgn * (u - 0.5f) * 0.22f, d = 92 - 10 * us;
        c->eye[0] = d * cosf(az); c->eye[1] = d * sinf(az); c->eye[2] = 4.5f + 2 * r2;
        c->tgt[0] = 0; c->tgt[1] = 0; c->tgt[2] = 19;
        c->fov = 28; c->ap = 1.1f;
        break;
    }
    case S_PLAQUE: {
        float x = sgn * (-7 + 14 * us) * 0.6f;
        c->eye[0] = PLQ_C[0] + x; c->eye[1] = PLQ_C[1] - 25 + 3 * us; c->eye[2] = PLQ_C[2] + 11 - 2 * us;
        c->tgt[0] = PLQ_C[0] + x * 0.25f; c->tgt[1] = PLQ_C[1] + 1.5f; c->tgt[2] = PLQ_C[2] + 1.0f;
        c->fov = 30; c->ap = 0.55f;
        memcpy(c->fp, PLQ_C, sizeof c->fp);
        c->fp[2] -= 0.4f;
        c->has_fp = 1;
        break;
    }
    case S_TRAIN: {
        int L = (int)(r1 * 5.99f);
        float rt = L < 5 ? mod_pair[L] * N_TUBE[L] * 0.5f : MOD_S1 * N_INT * 0.5f;
        float m[3] = {rt * cosf(D_ANG), rt * sinf(D_ANG), LAYER_Z[L] + 0.15f};
        if (L == 5) {   /* the intermediate wheel meshing its pinion on the drive arbor */
            float dx = Dp[0] - P2p[0], dy = Dp[1] - P2p[1], l = sqrtf(dx * dx + dy * dy);
            m[0] = P2p[0] + dx / l * rt; m[1] = P2p[1] + dy / l * rt;
        }
        float az = D_ANG + sgn * (float)((50 + 25 * r2) * RAD) + sgn * (u - 0.5f) * 0.16f;
        float d = 29 - 4 * us, h = 2.5f + 2.5f * r3;
        c->eye[0] = m[0] + d * cosf(az); c->eye[1] = m[1] + d * sinf(az); c->eye[2] = fminf(m[2] + h, 13.6f);
        memcpy(c->tgt, m, sizeof m);
        c->tgt[2] += 0.3f;
        c->fov = 26; c->ap = 0.55f;
        break;
    }
    case S_DIAL: {
        float tgt[3] = {2.2f + 2 * r3, -5.0f, Z_DIAL1};
        float az = F + (r1 - 0.5f) * 0.9f + sgn * (u - 0.5f) * 0.2f, el = (float)((48 + 10 * r2) * RAD);
        orbit(c, tgt, az, el, 34 - 4 * us);
        c->fov = 30; c->ap = 0.9f;
        break;
    }
    case S_SUN: {
        float tgt[3] = {0, 0, Z_PLANE - 1.0f};
        float az = (float)TAU * r1 + sgn * u * 0.3f, el = (float)((16 + 10 * r2) * RAD);
        orbit(c, tgt, az, el, 30 - 6 * us);
        c->ap = 1.0f;
        break;
    }
    case S_PLANET:
    case S_EARTH: {
        int p = type == S_EARTH ? P_EARTH : (int[]){P_JUPITER, P_SATURN, P_MARS, P_JUPITER, P_SATURN, P_VENUS}[(int)(r1 * 5.99f)];
        if (k % 4 == 3 && p < P_JUPITER) p = r1 < 0.5f ? P_JUPITER : P_SATURN;   /* only the slow ones while the model runs */
        float tgt[3] = {s->pos[p][0], s->pos[p][1], s->pos[p][2]};
        float dist = p == P_SATURN ? 17 : p == P_JUPITER ? 14 : p == P_EARTH ? 9.5f : p == P_MARS ? 6 : 7;
        if (p == P_EARTH) {    /* frame the Earth and the Moon */
            float mp[3];
            moon_xyz(s, -1, mp);
            for (int i = 0; i < 3; i++) tgt[i] = tgt[i] * 0.72f + mp[i] * 0.28f;
        }
        float lon = (float)(s->lon[p] * RAD);
        float az = lon + (float)PI + sgn * (float)((48 + 30 * r2) * RAD) + sgn * (u - 0.5f) * 0.3f;
        float el = (float)((p >= P_JUPITER ? 18 : 26) * RAD + r3 * 8 * RAD);
        orbit(c, tgt, az, el, dist * (1.06f - 0.1f * us));
        if (c->eye[2] < 30.0f) c->eye[2] = 30.0f;
        c->fov = 30; c->ap = 0.35f;
        break;
    }
    case S_FLYBY: {
        int p = (int[]){P_JUPITER, P_SATURN, P_MARS, P_EARTH}[(int)(r1 * 3.99f)];
        float lon = (float)(s->lon[p] * RAD);
        float rad[3] = {cosf(lon), sinf(lon), 0}, tan_[3] = {-sinf(lon) * sgn, cosf(lon) * sgn, 0};
        float off = 3.5f + PL_R[p] * 2.2f, run = 16 + 8 * PL_R[p];
        float x = (float)(-1 + 2 * smoother(ul * 0.8 + 0.1)) * run;
        for (int i = 0; i < 3; i++) c->eye[i] = s->pos[p][i] + tan_[i] * x + rad[i] * off;
        c->eye[2] = Z_PLANE + PL_R[p] + 1.6f + 1.4f * r2;
        if (c->eye[2] < 30.2f) c->eye[2] = 30.2f;
        /* look ahead and in, towards the Sun */
        for (int i = 0; i < 3; i++) c->tgt[i] = s->pos[p][i] * 0.55f + tan_[i] * (x + run * 0.6f) * 0.5f;
        c->tgt[2] = Z_PLANE - 2;
        c->fov = 34; c->ap = 0.5f;
        float dp[3] = {s->pos[p][0] - c->eye[0], s->pos[p][1] - c->eye[1], s->pos[p][2] - c->eye[2]};
        c->focus = sqrtf(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]);
        memcpy(c->fp, s->pos[p], sizeof c->fp);
        c->has_fp = 1;
        return;
    }
    }
    if (c->has_fp) return;
    float d[3] = {c->tgt[0] - c->eye[0], c->tgt[1] - c->eye[1], c->tgt[2] - c->eye[2]};
    c->focus = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

/* the fade between shots */
static float shot_fade(double t) {
    double u = fmod(t, SHOT_LEN);
    if (t < 0) return 0;
    return (float)(smoother(u / 0.9) * smoother((SHOT_LEN - u) / 0.9));
}

/* ---- the camera's frame across outputs (as flightsim) ----
 * Canvas point (cam_cx, cam_cy) is straight ahead and the framing is set by
 * the front output's short side; this window shows canvas rect view_*.
 * Canvas y grows downward. Without --layout the canvas is just the window. */
static float cam_cx, cam_cy, view_x, view_y, view_w, view_h, front_min;
static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = (float)W; view_h = (float)H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f;
        front_min = (float)(W < H ? W : H);
        return;
    }
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    cam_cx = span[mid][0] + span[mid][2] / 2;
    cam_cy = span[mid][1] + span[mid][3] / 2;
    front_min = fminf(span[mid][2], span[mid][3]);
    view_x = span[span_me][0]; view_y = span[span_me][1];
    view_w = span[span_me][2]; view_h = span[span_me][3];
}

/* ---- GL ---- */
static GLuint prog_kind[4], prog_scene, prog_depth, prog_prep, prog_dof, prog_final;
static GLuint vao_mesh, vbo_mesh, ibo_mesh, vao_empty;
static int nverts, nidx;
static GLuint fbo_shadow, tex_shadow;
static const int SHADOW = 2048;
static GLuint fbo_ms, rb_ms_col, tex_ms_z, rb_ms_dep, fbo_res, tex_res_col, fbo_prep, tex_prep, fbo_dof, tex_dof;
static int rs_w, rs_h, msaa = 2;
static float render_scale = 1.0f;
static M4 light_vp, sun_vp;
static GLuint fbo_sunmap, tex_sunmap, tex_env;
#define SUN_MAP 2048
static const float SUN_NEAR = 3.8f, SUN_FAR = 200.0f, SUN_TAN = 2.9f;
static float moon_dir[3];
static const float WIN[4] = {-175, -55, 42, 192}, WALL_Y = 260;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "orrery: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *vs, const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "orrery: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, GLenum filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

/* (re)make the render targets for a scene of w x h */
static void targets(int w, int h) {
    if (w == rs_w && h == rs_h) return;
    if (rs_w) {
        glDeleteFramebuffers(1, &fbo_ms); glDeleteRenderbuffers(1, &rb_ms_col); glDeleteTextures(1, &tex_ms_z);
        glDeleteRenderbuffers(1, &rb_ms_dep);
        glDeleteFramebuffers(1, &fbo_res); glDeleteTextures(1, &tex_res_col);
        glDeleteFramebuffers(1, &fbo_dof); glDeleteTextures(1, &tex_dof);
        glDeleteFramebuffers(1, &fbo_prep); glDeleteTextures(1, &tex_prep);
    }
    rs_w = w; rs_h = h;
    glGenRenderbuffers(1, &rb_ms_col);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_col);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, GL_R11F_G11F_B10F, w, h);
    glGenTextures(1, &tex_ms_z);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, msaa, GL_R16F, w, h, GL_TRUE);
    glGenRenderbuffers(1, &rb_ms_dep);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_dep);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, GL_DEPTH_COMPONENT24, w, h);
    glGenFramebuffers(1, &fbo_ms);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_ms_col);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_ms_dep);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("multisampled framebuffer incomplete");

    tex_res_col = mk_tex(GL_R11F_G11F_B10F, w, h, GL_RGB, GL_FLOAT, GL_LINEAR);
    glGenFramebuffers(1, &fbo_res);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_res_col, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("resolve framebuffer incomplete");

    tex_prep = mk_tex(GL_RGBA16F, (w + 1) / 2, (h + 1) / 2, GL_RGBA, GL_HALF_FLOAT, GL_LINEAR);
    glGenFramebuffers(1, &fbo_prep);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_prep);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_prep, 0);
    tex_dof = mk_tex(GL_R11F_G11F_B10F, (w + 1) / 2, (h + 1) / 2, GL_RGB, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGenFramebuffers(1, &fbo_dof);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dof);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_dof, 0);
}

#define U(p, n) glGetUniformLocation(p, n)

static void gl_init(double now_t) {
    for (int k = 0; k < 4; k++) {
        char *src = malloc(strlen(FS_SCENE) + 64);
        sprintf(src, "#version 330 core\nconst int KIND = %d;\n%s", k, FS_SCENE);
        prog_kind[k] = mk_prog(VS_SCENE, src);
        free(src);
    }
    prog_scene = prog_kind[2];
    prog_depth = mk_prog(VS_SCENE, FS_DEPTH);
    prog_prep = mk_prog(VS_TRI, FS_PREP);
    prog_dof = mk_prog(VS_TRI, FS_DOF);
    prog_final = mk_prog(VS_TRI, FS_FINAL);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_empty);
    glEnable(GL_CULL_FACE);

    /* weld identical vertices into an index buffer */
    unsigned *idx = malloc((size_t)nv * sizeof *idx);
    int hsize = 1;
    while (hsize < nv * 2) hsize <<= 1;
    int *ht = malloc((size_t)hsize * sizeof *ht);
    memset(ht, -1, (size_t)hsize * sizeof *ht);
    /* bodies keep their index ranges; the vertices are shared */
    int nu = 0;
    for (int i = 0; i < nv; i++) {
        uint32_t h = 2166136261u;
        const unsigned char *b = (const unsigned char *)&vb[i];
        for (size_t k = 0; k < sizeof(Vtx); k++) h = (h ^ b[k]) * 16777619u;
        int slot = (int)(h & (uint32_t)(hsize - 1));
        while (ht[slot] >= 0 && memcmp(&vb[ht[slot]], &vb[i], sizeof(Vtx))) slot = (slot + 1) & (hsize - 1);
        if (ht[slot] < 0) { vb[nu] = vb[i]; ht[slot] = nu++; }
        idx[i] = (unsigned)ht[slot];
    }
    free(ht);
    nverts = nu;
    nidx = nv;
    glGenVertexArrays(1, &vao_mesh);
    glBindVertexArray(vao_mesh);
    glGenBuffers(1, &vbo_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mesh);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)nu * (GLsizeiptr)sizeof(Vtx), vb, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo_mesh);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_mesh);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)nv * 4, idx, GL_STATIC_DRAW);
    free(idx);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)12);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)24);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)36);
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray((GLuint)i);
    free(vb);
    vb = NULL;

    GLint maxs = 4;
    glGetIntegerv(GL_MAX_SAMPLES, &maxs);
    if (msaa > maxs) msaa = maxs;

    tex_shadow = mk_tex(GL_DEPTH_COMPONENT24, SHADOW, SHADOW, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_shadow, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    /* the room's reflection, prefiltered */
    {
        GLuint pg = mk_prog(VS_TRI, FS_ENVGEN), fb;
        glGenTextures(1, &tex_env);
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env);
        for (int l = 0; l < 5; l++)
            for (int f = 0; f < 6; f++)
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, l, GL_R11F_G11F_B10F, 64 >> l, 64 >> l, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 4);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glUseProgram(pg);
        glUniform4f(U(pg, "uWin"), WIN[0], WIN[1], WIN[2], WIN[3]);
        glUniform1f(U(pg, "uWallY"), WALL_Y);
        glBindVertexArray(vao_empty);
        glDisable(GL_CULL_FACE);
        for (int l = 0; l < 5; l++)
            for (int f = 0; f < 6; f++) {
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, tex_env, l);
                glViewport(0, 0, 64 >> l, 64 >> l);
                glUniform1i(U(pg, "uFace"), f);
                glUniform1f(U(pg, "uSize"), (float)(64 >> l));
                glUniform1f(U(pg, "uRough"), 0.04f + 0.96f * l / 4.0f);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
        glEnable(GL_CULL_FACE);
        glDeleteFramebuffers(1, &fb);
    }

    /* the lamp's shadow map: straight down from the Sun's centre */
    tex_sunmap = mk_tex(GL_DEPTH_COMPONENT24, SUN_MAP, SUN_MAP, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_NEAREST);
    glGenFramebuffers(1, &fbo_sunmap);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_sunmap);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_sunmap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    {
        float e[3] = {0, 0, Z_PLANE}, c[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        float n = SUN_NEAR;
        sun_vp = m4_mul(m4_frustum(-SUN_TAN * n, SUN_TAN * n, -SUN_TAN * n, SUN_TAN * n, n, SUN_FAR), m4_look(e, c, up));
    }

    /* moonlight, through the middle of the window */
    moon_dir[0] = (WIN[0] + WIN[1]) / 2; moon_dir[1] = WALL_Y; moon_dir[2] = (WIN[2] + WIN[3]) / 2 - 15;
    v3norm(moon_dir);
    {
        float c[3] = {0, -8, 12}, e[3] = {c[0] + moon_dir[0] * 300, c[1] + moon_dir[1] * 300, c[2] + moon_dir[2] * 300}, up[3] = {0, 0, 1};
        light_vp = m4_mul(m4_ortho(-62, 62, -50, 50, 150, 480), m4_look(e, c, up));
    }

    for (int kk = 0; kk < 4; kk++) {
    prog_scene = prog_kind[kk];
    glUseProgram(prog_scene);
    glUniform1i(U(prog_scene, "uShadow"), 0);
    glUniform1i(U(prog_scene, "uDial"), 1);
    glUniform1i(U(prog_scene, "uPlaque"), 2);
    glUniform1i(U(prog_scene, "uDate"), 3);
    glUniform1i(U(prog_scene, "uSunMap"), 4);
    glUniform1i(U(prog_scene, "uEnv"), 5);
    glUniformMatrix4fv(U(prog_scene, "uSVP"), 1, GL_FALSE, sun_vp.m);
    glUniform2f(U(prog_scene, "uSNF"), SUN_NEAR, SUN_FAR);
    glUniform1f(U(prog_scene, "uSTan"), 0.5f / SUN_TAN);
    glUniform3fv(U(prog_scene, "uMoonL"), 1, moon_dir);
    glUniform3f(U(prog_scene, "uMoonCol"), 0.30f * 0.42f, 0.30f * 0.56f, 0.30f * 0.85f);
    glUniformMatrix4fv(U(prog_scene, "uLVP"), 1, GL_FALSE, light_vp.m);
    glUniform1f(U(prog_scene, "uShTexel"), 1.0f / SHADOW);
    glUniform3f(U(prog_scene, "uSun"), 0, 0, Z_PLANE);
    glUniform1f(U(prog_scene, "uSunR"), R_LAMP);
    glUniform3f(U(prog_scene, "uSunCol"), 1.0f * 1100, 0.70f * 1100, 0.42f * 1100);
    glUniform4f(U(prog_scene, "uDateRect"), DATE_RECT[0], DATE_RECT[1], DATE_RECT[2], DATE_RECT[3]);
    glUniform3fv(U(prog_scene, "uPlqC"), 1, PLQ_C);
    glUniform3fv(U(prog_scene, "uPlqU"), 1, plq_u);
    glUniform3fv(U(prog_scene, "uPlqV"), 1, plq_v);
    glUniform2f(U(prog_scene, "uPlqSize"), PLQ_W, PLQ_H);
    glUniform4f(U(prog_scene, "uWin"), WIN[0], WIN[1], WIN[2], WIN[3]);
    glUniform1f(U(prog_scene, "uWallY"), WALL_Y);
    }
    glUseProgram(prog_prep);
    glUniform1i(U(prog_prep, "uCol"), 0);
    glUniform1i(U(prog_prep, "uDep"), 1);
    glUseProgram(prog_dof);
    glUniform1i(U(prog_dof, "uPrep"), 3);
    glUseProgram(prog_final);
    glUniform1i(U(prog_final, "uCol"), 0);
    glUniform1i(U(prog_final, "uDof"), 2);
    glUniform1i(U(prog_final, "uPrep"), 3);

    /* the engraving */
    glUseProgram(prog_text);
    glUniform1i(U(prog_text, "uAtlas"), 0);
    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(4 * sizeof(float)));
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);
    text_styles();
    build_atlas();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glUniform2f(U(prog_text, "uAtlasSize"), (float)AW, (float)atlas_h);
    free(atlas);
    atlas = NULL;
    glGenFramebuffers(1, &fbo_text);
    tex_dial = mk_mask(DIAL_TEX, DIAL_TEX);
    tex_plaque = mk_mask(PLQ_TEX_W, PLQ_TEX_H);
    tex_date = mk_mask(DATE_TEX_W, DATE_TEX_H);
    draw_dial(now_t);
}

static void draw_body(int i, GLint lmodel) {
    glUniformMatrix4fv(lmodel, 1, GL_FALSE, body[i].m.m);
    glDrawElements(GL_TRIANGLES, body[i].count, GL_UNSIGNED_INT, (void *)((size_t)body[i].first * 4));
}

/* roughly front to back, so the depth test throws away what's hidden early */
static const int draw_order[] = {B_GLOBE0, B_GLOBE0 + 1, B_GLOBE0 + 2, B_GLOBE0 + 3, B_GLOBE0 + 4, B_GLOBE0 + 5, B_MOONARM,
                                 B_JM0, B_JM0 + 1, B_JM0 + 2, B_JM0 + 3, B_SUN, B_TUBE0, B_TUBE0 + 1, B_TUBE0 + 2, B_TUBE0 + 3,
                                 B_TUBE0 + 4, B_TUBE0 + 5, B_DRV0, B_DRV0 + 1, B_DRV0 + 2, B_DRV0 + 3, B_DRV0 + 4,
                                 B_DRV0 + 5, B_P2, B_IDLER, B_STATIC, B_TABLE, B_ROOM};
#define NDRAW ((int)(sizeof draw_order / sizeof *draw_order))

/* which specialisation of the scene shader draws a body */
static int kind_of(int b) { return b == B_ROOM ? 0 : b == B_TABLE ? 1 : b >= B_GLOBE0 && b < B_GLOBE0 + NPL ? 3 : 2; }

/* the scene's render scale to start with: under a megapixel (4x MSAA),
 * upscaled with Catmull-Rom; live, it then follows the frame budget */
static float default_scale(void) {
    float s = sqrtf(0.72e6f / ((float)W * H));
    return s > 1 ? 1 : s;
}

enum { PR_SHADOW, PR_SCENE, PR_RESOLVE, PR_DOF, PR_FINAL, PR_N };
static const char *prof_name[PR_N] = {"shadow", "scene", "resolve", "dof", "final"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    GLuint64 ts[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &ts[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (ts[i + 1] - ts[i]) * 1e-6;
}

static double date_base, date_wall0;   /* --date: the model's "now" runs from here */
static double model_now(double wt) { return opt_date ? date_base + (wt - date_wall0) : wt; }

/* t: show time (s since the epoch); wt: the wall clock */
static void render(double t, double wt, GLuint target) {
    setup_view();
    int sw = (int)(W * render_scale + 0.5f), sh = (int)(H * render_scale + 0.5f);
    if (sw < 16) sw = 16;
    if (sh < 16) sh = 16;
    targets(sw, sh);

    double mt = model_now(wt) + model_offset(t) * 86400.0;
    Sky sky;
    sky_at(mt, &sky);
    pose(&sky, t);
    draw_plaque(wt);
    draw_date(mt);

    /* the camera */
    Cam cam;
    camera_at(t, &sky, &cam);
    float up[3] = {0, 0, 1};
    M4 view = m4_look(cam.eye, cam.tgt, up);
    float fwd[3] = {cam.tgt[0] - cam.eye[0], cam.tgt[1] - cam.eye[1], cam.tgt[2] - cam.eye[2]};
    v3norm(fwd);
    float cam_f = front_min / 2 / tanf(cam.fov * (float)RAD / 2);
    const float NEAR = 0.6f, FAR = 1400;
    float l = (view_x - cam_cx) / cam_f, r = (view_x + view_w - cam_cx) / cam_f;
    float b = (cam_cy - view_y - view_h) / cam_f, tp = (cam_cy - view_y) / cam_f;
    M4 vp = m4_mul(m4_frustum(l * NEAR, r * NEAR, b * NEAR, tp * NEAR, NEAR, FAR), view);

    /* the shadow maps, moonlight and the lamp's: only redrawn when the
     * model has moved (holding the present, the arms creep imperceptibly) */
    prof_mark(PR_SHADOW);
    glBindVertexArray(vao_mesh);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glUseProgram(prog_depth);
    GLint dmodel = U(prog_depth, "uModel");
    static double shadow_mt = -1e18;
    if (fabs(mt - shadow_mt) > 1200) {
    shadow_mt = mt;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, SHADOW, SHADOW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.5f, 3.0f);
    glDisable(GL_CULL_FACE);
    glUniformMatrix4fv(U(prog_depth, "uVP"), 1, GL_FALSE, light_vp.m);
    for (int j = 0; j < NDRAW; j++)
        if (body[draw_order[j]].cast) draw_body(draw_order[j], dmodel);
    /* the lamp's shadow map: everything under the Sun but the Sun itself */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_sunmap);
    glViewport(0, 0, SUN_MAP, SUN_MAP);
    glClear(GL_DEPTH_BUFFER_BIT);
    glPolygonOffset(1.2f, 2.0f);
    glUniformMatrix4fv(U(prog_depth, "uVP"), 1, GL_FALSE, sun_vp.m);
    for (int j = 0; j < NDRAW; j++)
        if (body[draw_order[j]].cast && draw_order[j] != B_SUN) draw_body(draw_order[j], dmodel);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_CULL_FACE);
    }

    /* the scene */
    prof_mark(PR_SCENE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
    glViewport(0, 0, sw, sh);
    static const float clear_col[4] = {0.01f, 0.008f, 0.006f, 1}, clear_z[4] = {2000, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 0, clear_col);
    glClearBufferfv(GL_COLOR, 1, clear_z);
    glClear(GL_DEPTH_BUFFER_BIT);
    for (int kk = 0; kk < 4; kk++) {
    prog_scene = prog_kind[kk];
    glUseProgram(prog_scene);
    glUniformMatrix4fv(U(prog_scene, "uVP"), 1, GL_FALSE, vp.m);
    glUniform3fv(U(prog_scene, "uCam"), 1, cam.eye);
    glUniform3fv(U(prog_scene, "uFwd"), 1, fwd);

    }
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_dial);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_plaque);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_date);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_sunmap);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glBindVertexArray(vao_mesh);
    /* a depth-only pass first, so the shading runs once per visible sample
     * (the train under the dial plate would otherwise be shaded and then
     * covered); the room is drawn last without it, behind everything */
    glUseProgram(prog_depth);
    glUniformMatrix4fv(U(prog_depth, "uVP"), 1, GL_FALSE, vp.m);
    glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    for (int j = 0; j < NDRAW; j++)
        if (draw_order[j] != B_ROOM) draw_body(draw_order[j], dmodel);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    for (int j = 0; j < NDRAW; j++) {
        int b = draw_order[j], k = kind_of(b);
        if (b == B_ROOM) glDepthMask(GL_TRUE);
        if (j == 0 || k != kind_of(draw_order[j - 1])) glUseProgram(prog_kind[k]);
        draw_body(b, U(prog_kind[k], "uModel"));
    }

    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    /* resolve */
    prof_mark(PR_RESOLVE);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, sw, sh, 0, 0, sw, sh, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    /* depth of field: circles of confusion in scene pixels, from the
     * shared camera's focal length so every output agrees */
    prof_mark(PR_DOF);
    float fpx = cam_f * (float)sw / view_w;
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glBindVertexArray(vao_empty);
    int hw = (sw + 1) / 2, hh = (sh + 1) / 2;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_prep);
    glViewport(0, 0, hw, hh);
    glUseProgram(prog_prep);
    glUniform2f(U(prog_prep, "uSize"), (float)sw, (float)sh);
    glUniform1i(U(prog_prep, "uLast"), msaa - 1);
    const float *fpt = cam.has_fp ? cam.fp : cam.tgt;
    float fd = fmaxf(fwd[0] * (fpt[0] - cam.eye[0]) + fwd[1] * (fpt[1] - cam.eye[1]) + fwd[2] * (fpt[2] - cam.eye[2]), 1.0f);
    glUniform3f(U(prog_prep, "uLens"), fd, cam.ap * fpx, 16.0f * sh / 1080.0f + 4);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_res_col);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dof);
    glUseProgram(prog_dof);
    glUniform2f(U(prog_dof, "uHalf"), (float)hw, (float)hh);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_prep);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_dof);
    glGenerateMipmap(GL_TEXTURE_2D);

    /* tonemap onto the target */
    prof_mark(PR_FINAL);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_final);
    glUniform2f(U(prog_final, "uOut"), (float)W, (float)H);
    glUniform2f(U(prog_final, "uSize"), (float)sw, (float)sh);
    glUniform4f(U(prog_final, "uVig"), (view_x - cam_cx) / front_min, (cam_cy - view_y - view_h) / front_min, view_w / front_min,
                view_h / front_min);
    glUniform1f(U(prog_final, "uTime"), (float)fmod(t, 1000.0));
    glUniform1f(U(prog_final, "uExposure"), 1.0f);
    glUniform1f(U(prog_final, "uFade"), opt_shot ? 1.0f : shot_fade(t));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_res_col);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_mark(PR_N);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

/* ---- --ephem: the model's state, in words ---- */
static void print_ephem(double t, double wt) {
    double mt = model_now(wt) + model_offset(t) * 86400.0;
    Sky s;
    sky_at(mt, &s);
    time_t ts = (time_t)mt;
    struct tm tm;
    gmtime_r(&ts, &tm);
    printf("model time %04d-%02d-%02d %02d:%02d UTC (JD %.4f)%s\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
           tm.tm_min, s.jd, model_offset(t) != 0 ? ", wound forward" : "");
    printf("heliocentric ecliptic longitude (equinox of date):\n");
    for (int p = 0; p < NPL; p++) {
        double l = s.lon[p];
        printf("  %-8s %7.2f  (%2d %s)\n", PL_NAME[p], l, (int)fmod(l, 30), SIGN[(int)(l / 30) % 12]);
    }
    double sun = sun_lon_geo(s.jd), el = wrap360(s.moon - sun);
    printf("Moon: geocentric longitude %.2f, elongation %.1f, %.0f%% lit, %s\n", s.moon, el, 50 * (1 - cos(el * RAD)),
           el < 180 ? "waxing" : "waning");
    printf("Galilean moons (from Jupiter, 0 = away from the Sun): Io %.0f, Europa %.0f, Ganymede %.0f, Callisto %.0f\n",
           s.jm[0], s.jm[1], s.jm[2], s.jm[3]);
    static const double PER[NPL] = {87.9691, 224.701, 365.256, 686.980, 4332.59, 10759.22};
    printf("gear train (tube turns per turn of the drive arbor):\n");
    for (int p = 0; p < NPL; p++) {
        double g = p == P_SATURN ? (double)N_DRV[p] / N_INT * N_INTP / N_TUBE[p] : (double)N_DRV[p] / N_TUBE[p];
        printf("  %-8s %3d:%-3d %s ratio %.6f, true %.6f (%+.3f%%)\n", PL_NAME[p], N_DRV[p], N_TUBE[p],
               p == P_SATURN ? "(10:40, 11:81)" : "              ", g, 365.256 / PER[p], (g / (365.256 / PER[p]) - 1) * 100);
    }
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "orrery");
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
        "usage: orrery [--window | --root | --layer] [--seed N] [--scale X] [--font FAMILY]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm | --bench FRAMES] [--size WxH] [--at SECONDS]\n"
        "              [--date UNIXTIME] [--view NAME] [--ephem]\n"
        "  --scale X     fixed render scale (default: adapts to the frame budget)\n"
        "  --at S        the moment S seconds after the epoch (shots are 20 s, four to an act)\n"
        "  --date T      pretend the present is unix time T\n"
        "  --view NAME   hold one kind of shot: wide high low plaque train dial sun planet earth flyby\n"
        "  --ephem       print the model's positions and the gear ratios, and exit\n");
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

static void save_ppm(const char *path) {
    unsigned char *px = malloc((size_t)W * H * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; y--) fwrite(px + (size_t)y * W * 3, 1, (size_t)W * 3, f);
    fclose(f);
    free(px);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-date") && i + 1 < argc) opt_date = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-ephem")) opt_ephem = 1;
        else if (!strcmp(a, "-view") && i + 1 < argc) {
            const char *v = argv[++i];
            for (int k = 0; k < NSHOT; k++)
                if (!strcmp(v, SHOT_NAME[k])) opt_shot_type = k;
            if (opt_shot_type < 0) usage();
        } else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed * 2654435761u | 1;
    for (int i = 0; i < 4; i++) frand();
    int offscreen = opt_shot || opt_bench;

    /* the show's clock: shared by every instance through --epoch; a shot
     * shows the moment --at seconds after it */
    double wall0 = wallclock();
    double epoch = opt_epoch ? opt_epoch : wall0 - (offscreen ? opt_at : 0);
    double wt_shot = opt_epoch ? opt_epoch + opt_at : wall0;
    date_base = opt_date; date_wall0 = opt_epoch ? opt_epoch : epoch;
    build();
    if (opt_ephem) {
        print_ephem(opt_at, wt_shot);
        return 0;
    }
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)epoch;
    fonts_init();

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
        XStoreName(dpy, win, "Orrery");
        XClassHint chint = {"orrery", "Orrery"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init(model_now(wt_shot));

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        render_scale = opt_scale > 0 ? opt_scale : default_scale();
        render(opt_at, wt_shot, fbo);
        glFinish();
        if (opt_shot) {
            save_ppm(opt_shot);
            fprintf(stderr, "orrery: shot %s (%s)\n", opt_shot, SHOT_NAME[cur_shot]);
            return 0;
        }
        double t0 = now(), worst = 0;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            render(opt_at + (i + 1) / 60.0, wt_shot + (i + 1) / 60.0, fbo);
            glFinish();
            prof_collect();
            double d = now() - f0;
            if (d > worst) worst = d;
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d scale %.2f msaa %d: %.2f ms/frame avg, %.2f worst (%s)\n", W, H, render_scale, msaa, per * 1e3,
               worst * 1e3, (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < PR_N; i++) printf("  %-8s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        printf("  %d vertices, %d triangles\n", nverts, nidx / 3);
        return 0;
    }

    /* GPU time per frame, from timer queries read a couple of frames late,
     * steers the render scale (unless --scale): there may be one of us per
     * monitor on one small GPU */
    GLuint q[3];
    glGenQueries(3, q);
    int qn = 0;
    render_scale = opt_scale > 0 ? opt_scale : default_scale();
    double avg = 0.004, last_change = 0;
    double t0 = now(), w0 = wallclock();
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
        double t = now();
        if (frame >= 3) {
            GLuint64 ns = 0;
            GLint ready = 0;
            glGetQueryObjectiv(q[qn], GL_QUERY_RESULT_AVAILABLE, &ready);
            if (ready) {
                glGetQueryObjectui64v(q[qn], GL_QUERY_RESULT, &ns);
                avg += (ns * 1e-9 - avg) * 0.1;
            }
        }
        glBeginQuery(GL_TIME_ELAPSED, q[qn]);
        double wt = wallclock();
        /* spanning, everything runs off the shared epoch */
        double st = opt_epoch ? opt_at + (wt - opt_epoch) : opt_at + (t - t0);
        (void)w0;
        render(st, wt, 0);
        glEndQuery(GL_TIME_ELAPSED);
        qn = (qn + 1) % 3;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        if (opt_scale <= 0 && frame > 30 && t - last_change > 1.0) {
            float s = render_scale;
            if (avg > 0.0045) s *= 0.92f;
            else if (avg < 0.003) s *= 1.04f;
            float d = default_scale(), lo = 0.6f * d, hi = fminf(1.0f, 1.2f * d);
            s = s < lo ? lo : s > hi ? hi : s;
            if (fabsf(s - render_scale) > 0.02f) { render_scale = s; last_change = t; }
        }
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
