/*
 * sandworm — a deep desert as a lock screen: a wide field of wind-built
 * dunes under a low sun, and something enormous moving under the sand.
 *
 * The dunes are transverse ridges with gentle windward backs, sharp brinks
 * and steep slip faces, broken into crescents where the sand runs thin,
 * with smaller oblique dunes riding on their backs, wind ripples and
 * megaripples, interdune flats of darker grit, and a flat-topped butte or
 * two on the horizon for scale. Sand smokes off the crests on windy days;
 * the far desert shimmers in the heat. The light follows the real clock:
 * dawn and dusk rake long shadows across the ridges, the sun stays low all
 * day (it never climbs past the mid-thirties, so there is always relief),
 * and at night the dunes go silver-blue under the Moon (in its real phase)
 * and a slowly turning star field.
 *
 * Every few minutes a worm passes under the desert. Usually far off: a
 * travelling swell in the sand, its flanks slumping and streaming, a faint
 * plume of dust over its head and a furrow left behind it that the wind
 * soon smooths away. Every third pass it comes closer, sinks out of sight,
 * and then breaches: a ridged, plated body hundreds of metres long arches
 * out of the dunes, sand pouring off it, flows through that arch and dives
 * back in; shock ripples run out across the sand from where it broke the
 * surface and where it went back under, and the dust hangs in the air for
 * a minute after the tail is gone.
 *
 * Rendering: the desert is a polar grid of terrain around the (fixed)
 * camera, columns a couple of pixels apart and rings spaced to match. Its
 * heights come from an analytic function shared, character for character,
 * between the C side and GLSL (TERRAIN_SRC below), so the CPU knows exactly
 * where the sand is. The grid only records each pixel's distance; every
 * pixel is then shaded once, from a cache of what never changes (the sand's
 * colour and normal, ripples and all, worked out when the view is set up)
 * unless the worm has moved or stirred the sand there. Cast shadows are
 * marched on the grid, a slice of it per frame. The worm's work on the sand
 * (swell, rim, furrow, the streaming) is drawn every other frame into a
 * small height map that follows it. Its body is a tube swept along the
 * head's own past track, so the arch stays put while the body slides
 * through it. Falling sand and dust are billboards. Then heat shimmer, tone
 * mapping and grain in a last pass.
 *
 * Palettes (--palette, else picked from the seed): ochre, rust, pale; and
 * dusk and night, which also pin the time of day. --time pins it too.
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch (--schedule prints the passes and breaches). Spanning: run one
 * instance per monitor with the same --seed, --epoch and --layout, and
 * --output = this one's index; one camera looks out of the whole wall, so
 * the desert is one continuous panorama. The output under the middle of the
 * layout is the "front" one: it is centred, and carries the caption.
 *
 * Same plumbing as papercut: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen,
 * --bench N for timings.
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

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_caption = 1, opt_schedule, opt_breach;
static double opt_fps = 30;
static const char *opt_font = "Inter";
static const char *opt_palette;
static const char *opt_time;           /* --time: real (default), dawn, day, dusk, night or HH:MM */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the desert starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "sandworm: %s\n", what); exit(1); }

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

static double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static double sstep(double a, double b, double x) { double u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); }

/* ---- deterministic randomness: everything hashes from the seed ---- */
static uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
typedef struct { uint64_t s; } Rng;
static Rng rng_for(uint64_t a, uint64_t b, uint64_t c) {
    Rng r = {mix64(mix64(mix64(mix64(opt_seed) ^ a) ^ b) ^ c)};
    return r;
}
static double rnd(Rng *r) { r->s = mix64(r->s); return (double)(r->s >> 11) * (1.0 / 9007199254740992.0); }
static double rr(Rng *r, double a, double b) { return a + (b - a) * rnd(r); }

/* ---- the live details for the caption ---- */
static time_t locked_at;

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else now */
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
    return best ? best : time(NULL);
}

static void fmt_span(char *out, size_t n, long s) {
    long m = s / 60, h = m / 60, dd = h / 24;
    if (dd >= 1) snprintf(out, n, "%ld day%s, %ld hour%s", dd, dd == 1 ? "" : "s", h % 24, h % 24 == 1 ? "" : "s");
    else if (h >= 1 && m % 60) snprintf(out, n, "%ld hour%s %ld minute%s", h, h == 1 ? "" : "s", m % 60, m % 60 == 1 ? "" : "s");
    else if (h >= 1) snprintf(out, n, "%ld hour%s", h, h == 1 ? "" : "s");
    else if (m >= 1) snprintf(out, n, "%ld minute%s", m, m == 1 ? "" : "s");
    else snprintf(out, n, "less than a minute");
}

/* ==== the terrain, shared by C and GLSL ====
 * Written in the common subset of the two languages: scalar floats, the
 * GLSL builtins below given C definitions, and nz(), smooth value noise
 * (one lattice cell per unit) that each side defines over the same table.
 * The C compiler builds it from this macro; the shaders get it stringified.
 * terrain() also leaves a few things about the point in the t* globals. */
static float nzt[256 * 256];
static float nz(float x, float y) {
    float fx = floorf(x), fy = floorf(y);
    float u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    int x0 = (int)fx & 255, y0 = (int)fy & 255, x1 = (x0 + 1) & 255, y1 = (y0 + 1) & 255;
    float a = nzt[y0 * 256 + x0], b = nzt[y0 * 256 + x1], c = nzt[y1 * 256 + x0], d = nzt[y1 * 256 + x1];
    float ab = a + (b - a) * u, cd = c + (d - c) * u;
    return ab + (cd - ab) * v;
}
static float max(float a, float b) { return a > b ? a : b; }
static float min(float a, float b) { return a < b ? a : b; }
static float clamp(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smoothstep(float a, float b, float x) { float u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); }
/* the dune field's parameters (uniforms in GLSL) */
static float uWx, uWz, uL1, uA1, uL2, uA2, uMesa[8];

#define TERRAIN_SRC \
float tB; float tA; float tRel; float tRock; float tU; float tC; \
float smin_(float a, float b, float k) { \
    float h = max(k - max(a - b, b - a), 0.0) / k; \
    return min(a, b) - h * h * k * 0.25; \
} \
/* one dune's cross-section over a wavelength: a smooth windward back up \
 * to the crest at c, then the slip face down to the next trough; k rounds \
 * the brink (it is the footprint, so far crests are prefiltered) */ \
float dprof(float u, float c, float k) { \
    float x = clamp(u / c, 0.0, 1.0); \
    float g = x * x * (3.0 - 2.0 * x); \
    float v = (u - c) / (1.0 - c); \
    float l = v < 0.0 ? 1.0 - 1.5 * v : max(1.0 - v, 0.0) * sqrt(max(1.0 - v, 0.0)); \
    return smin_(g, l, k); \
} \
/* a butte: a ragged flat top over cliffs, a step, a talus apron */ \
float mesa1(float x, float z, float cx, float cz, float rad, float hgt) { \
    float dx = x - cx, dz = z - cz; \
    float d = sqrt(dx * dx + dz * dz); \
    if (d > rad * 2.0 + 1.0) return 0.0; \
    float e = rad * (0.78 + 0.45 * nz(dx * 0.005 + 3.0, dz * 0.005)) + 50.0 * (nz(dx * 0.025, dz * 0.025 + 7.0) - 0.5); \
    float cl = clamp((e - d) / 30.0, 0.0, 1.0); \
    float st = clamp((e - d - rad * 0.22) / 25.0, 0.0, 1.0); \
    float ta = clamp((e - d + rad * 0.8) / (rad * 0.8), 0.0, 1.0); \
    return hgt * (0.14 * ta * ta + 0.58 * cl + 0.28 * st); \
} \
float terrain(float x, float z, float fp) { \
    float qx = x * uWx + z * uWz; \
    float qz = z * uWx - x * uWz; \
    float b = 110.0 * (nz(x * 0.00022 + 3.1, z * 0.00022 + 7.7) - 0.5) + 35.0 * (nz(x * 0.0008 + 1.3, z * 0.0008 + 5.2) - 0.5); \
    float a = smoothstep(0.2, 0.72, nz(qx * 0.0007 + 11.0, qz * 0.0015 + 4.0)); \
    float A = uA1 * (0.1 + 0.9 * a); \
    float w = 300.0 * (nz(qx * 0.00035 + 2.0, qz * 0.0007 + 9.0) - 0.5) + 140.0 * (nz(qx * 0.0016 + 5.0, qz * 0.0024 + 1.0) - 0.5) + uL1 * 0.3 * (1.0 - a); \
    float ph = (qx - w) / uL1; \
    float u = ph - floor(ph); \
    float lc = clamp(A * 2.15 / uL1, 0.05, 0.4); \
    float k = clamp(fp * 1.4 / (lc * uL1), 0.004, 0.6); \
    float pr = dprof(u, 1.0 - lc, k); \
    float h = A * pr; \
    tU = u; tC = 1.0 - lc; \
    float rx = qx * 0.8 + qz * 0.6, rz = qz * 0.8 - qx * 0.6; \
    float ph2 = (rx - 90.0 * nz(rx * 0.004, rz * 0.0035 + 3.0)) / uL2; \
    float u2 = ph2 - floor(ph2); \
    float A2 = uA2 * smoothstep(0.38, 0.68, nz(rx * 0.003 + 7.0, rz * 0.005)) * smoothstep(0.0, 0.1, u) * (1.0 - smoothstep(0.75 - lc, 1.0 - lc, u)); \
    float lc2 = 0.38; \
    float k2 = clamp(fp * 1.4 / (lc2 * uL2), 0.004, 0.6); \
    h += A2 * dprof(u2, 1.0 - lc2, k2); \
    float m = max(mesa1(x, z, uMesa[0], uMesa[1], uMesa[2], uMesa[3]), mesa1(x, z, uMesa[4], uMesa[5], uMesa[6], uMesa[7])); \
    tB = b; tA = A; tRel = pr; tRock = clamp((m - h) / 6.0, 0.0, 1.0); \
    return b + max(h, m); \
}

TERRAIN_SRC

#define XSTR(...) #__VA_ARGS__
#define STR(...) XSTR(__VA_ARGS__)
static const char *GL_TERRAIN = STR(TERRAIN_SRC);

/* the smooth value noise table: one random value per lattice point */
static void nz_init(void) {
    for (int i = 0; i < 256 * 256; i++) nzt[i] = (float)((mix64((uint64_t)i * 0x9E37ull ^ 0x5a17d00ull) >> 11) * (1.0 / 9007199254740992.0));
}

/* ==== the wall and the camera ====
 * Canvas units are logical layout pixels (y down). One virtual camera looks
 * out of the whole layout: canvas point (cv_cx, cv_cy) is straight ahead
 * (the middle of the front output), cv_f is the focal length, and this
 * window shows canvas rect cv_x, cv_y, cv_w, cv_h. */
static double cv_cx, cv_cy, cv_f, cv_x, cv_y, cv_w, cv_h;
static double lay_x0, lay_y0, lay_x1, lay_y1;
static double fr[4];                   /* front output: x, y, w, h */

static int front_output(void) {
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
    }
    float mx = (x0 + x1) / 2;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) return i;
    return 0;
}

#define HFOV 54.0                      /* across the front screen, degrees */
static void canvas_update(void) {
    if (span_me >= 0) {
        lay_x0 = lay_y0 = 1e300; lay_x1 = lay_y1 = -1e300;
        for (int i = 0; i < span_n; i++) {
            lay_x0 = fmin(lay_x0, span[i][0]); lay_x1 = fmax(lay_x1, span[i][0] + span[i][2]);
            lay_y0 = fmin(lay_y0, span[i][1]); lay_y1 = fmax(lay_y1, span[i][1] + span[i][3]);
        }
        for (int k = 0; k < 4; k++) fr[k] = span[span_front][k];
        cv_x = span[span_me][0]; cv_y = span[span_me][1]; cv_w = span[span_me][2]; cv_h = span[span_me][3];
    } else {
        lay_x0 = lay_y0 = 0; lay_x1 = W; lay_y1 = H;
        fr[0] = fr[1] = 0; fr[2] = W; fr[3] = H;
        cv_x = cv_y = 0; cv_w = W; cv_h = H;
    }
    cv_cx = fr[0] + fr[2] / 2; cv_cy = fr[1] + fr[3] / 2;
    /* the front screen sees HFOV across (a portrait one no more than ~50
     * degrees up and down); a wide wall no more than 66 degrees either side */
    cv_f = fmax(fr[2] / 2 / tan(HFOV / 2 * M_PI / 180), fr[3] / 2 / tan(25 * M_PI / 180));
    double halfw = fmax(cv_cx - lay_x0, lay_x1 - cv_cx);
    if (halfw / cv_f > tan(66 * M_PI / 180)) cv_f = halfw / tan(66 * M_PI / 180);
}

/* the camera in the world (metres, y up): position, heading (direction
 * (sin psi, cos psi) in xz) and pitch, and its basis */
static double camX, camY, camZ, cam_psi, cam_pitch;
static double cF[3], cR[3], cU[3];
static double horizon_y;               /* the horizon straight ahead, canvas y */

static void cam_basis(void) {
    double cp = cos(cam_pitch), sp = sin(cam_pitch), cs = cos(cam_psi), sn = sin(cam_psi);
    cF[0] = cp * sn; cF[1] = sp; cF[2] = cp * cs;
    cR[0] = cs; cR[1] = 0; cR[2] = -sn;
    /* U = F x R */
    cU[0] = cF[1] * cR[2] - cF[2] * cR[1];
    cU[1] = cF[2] * cR[0] - cF[0] * cR[2];
    cU[2] = cF[0] * cR[1] - cF[1] * cR[0];
}

/* ==== the world, from the seed ==== */
static double windA;                   /* wind towards (cos, sin) in xz */
static double wind_strength;           /* 0..1: how much sand smokes off the crests */
static float cloud_cover;

static void world_init(void) {
    Rng r = rng_for(1, 2, 3);
    windA = rr(&r, 0, 2 * M_PI);
    uWx = (float)cos(windA); uWz = (float)sin(windA);
    uL1 = (float)rr(&r, 430, 580); uA1 = (float)rr(&r, 48, 68);
    uL2 = (float)rr(&r, 150, 220); uA2 = (float)rr(&r, 8, 14);
    wind_strength = rnd(&r) < 0.75 ? rr(&r, 0.4, 1.0) : rr(&r, 0.0, 0.3);
    cloud_cover = rnd(&r) < 0.6 ? (float)rr(&r, 0.1, 0.55) : 0.0f;
    memset(uMesa, 0, sizeof uMesa);
    /* look obliquely down- or upwind, so the ridges run across the view */
    double side = rnd(&r) < 0.5 ? 0 : M_PI;
    double off = rr(&r, 22, 50) * M_PI / 180 * (rnd(&r) < 0.5 ? -1 : 1);
    cam_psi = M_PI / 2 - windA + side + off;
    /* stand on a high crest: of a few dozen, the one that looks out over
     * the most (and the biggest) dunes */
    double best = -1e9, bx = 0, bz = 0, bh = 0;
    for (int c = 0; c < 40; c++) {
        double x0 = rr(&r, -20000, 20000), z0 = rr(&r, -20000, 20000);
        double cx = x0, cz = z0, ch = -1e9;
        for (double s = -uL1; s <= uL1; s += 2) {
            double x = x0 + uWx * s, z = z0 + uWz * s;
            double h = terrain((float)x, (float)z, 0.5f);
            if (h > ch) { ch = h; cx = x; cz = z; }
        }
        double top = -1e9, big = 0;
        for (double a = -0.45; a <= 0.46; a += 0.15)
            for (double d = 300; d < 4000; d *= 1.15) {
                double h = terrain((float)(cx + d * sin(cam_psi + a)), (float)(cz + d * cos(cam_psi + a)), 5.0f);
                top = fmax(top, h + d * 0.03);
                big += tA;
            }
        double score = (ch - top) + 0.02 * big;
        if (score > best) { best = score; bx = cx; bz = cz; bh = ch; }
    }
    /* step back off the brink, onto the windward back of the crest */
    camX = bx - uWx * 6; camZ = bz - uWz * 6;
    camY = fmax(bh, terrain((float)camX, (float)camZ, 0.5f)) + 40;
    /* see over whatever lies ahead */
    for (double a = -0.5; a <= 0.5; a += 0.1)
        for (double d = 60; d < 4000; d *= 1.08) {
            double h = terrain((float)(camX + d * sin(cam_psi + a)), (float)(camZ + d * cos(cam_psi + a)), (float)(d * 0.01));
            camY = fmax(camY, h + d * 0.025 + 4);
        }
    /* buttes on the horizon for scale: one ahead, one off to a side */
    for (int i = 0; i < 2; i++) {
        double a = cam_psi + (i == 0 ? rr(&r, -0.35, 0.35) : (rnd(&r) < 0.5 ? -1 : 1) * rr(&r, 0.6, 0.9));
        double d = i == 0 ? rr(&r, 7000, 11000) : rr(&r, 9000, 15000);
        uMesa[4 * i] = (float)(camX + d * sin(a));
        uMesa[4 * i + 1] = (float)(camZ + d * cos(a));
        uMesa[4 * i + 2] = (float)(i == 0 ? rr(&r, 380, 700) : rr(&r, 500, 1000));
        uMesa[4 * i + 3] = (float)(i == 0 ? rr(&r, 130, 210) : rr(&r, 160, 260));
    }
}

/* the horizon at 36% of the way down the front screen */
static void cam_aim(void) {
    cam_pitch = -atan(0.14 * fr[3] / cv_f);
    cam_basis();
    double d[3] = {sin(cam_psi), 0, cos(cam_psi)};
    double Y = d[0] * cU[0] + d[1] * cU[1] + d[2] * cU[2], Z = d[0] * cF[0] + d[1] * cF[1] + d[2] * cF[2];
    horizon_y = cv_cy - cv_f * Y / Z;
}

/* ==== the worm ====
 * Time runs in episodes of EP seconds. In each one a worm crosses the view
 * under the sand, its head passing the crossing point at tc; every third
 * one (--breach: every one) comes close and breaches there. */
#define EP 200.0
#define TRAIL 1500.0                   /* metres of furrow behind the tail */
typedef struct {
    long e;
    double cx, cz;                     /* the crossing point */
    double dx, dz;                     /* the path's direction */
    double am, lm, ph;                 /* its meander */
    double v, R, Lb, dep;              /* speed, radius, length, depth of the centreline */
    double tc;                         /* the head at the crossing point */
    int breach;
    double Tb, Hpk;                    /* the arch: seconds the head is out, apex height */
} Pass;

static void path_at(const Pass *P, double s, double *x, double *z);
static double ref_level(double x, double z);

/* can the camera see the sand at x, z (a little above it)? */
static int in_sight(double x, double z) {
    double y = ref_level(x, z) + 6;
    for (int k = 1; k < 48; k++) {
        double f = k / 48.0;
        double px = camX + (x - camX) * f, pz = camZ + (z - camZ) * f, py = camY + (y - camY) * f;
        if (terrain((float)px, (float)pz, 6.0f) > py) return 0;
    }
    return 1;
}

static void pass_params(long e, Pass *P) {
    static Pass cache;
    static long cached = -1;
    if (cached == e) { *P = cache; return; }
    Rng r = rng_for((uint64_t)e, 31, 7);
    memset(P, 0, sizeof *P);
    P->e = e;
    P->breach = opt_breach || e % 3 == 1;
    P->am = rr(&r, 40, 150); P->lm = rr(&r, 450, 900); P->ph = rr(&r, 0, 2 * M_PI);
    P->R = P->breach ? rr(&r, 32, 40) : rr(&r, 22, 32);
    P->Lb = P->R * rr(&r, 28, 34);
    P->v = rr(&r, 16, 21);
    P->dep = P->R * rr(&r, 1.1, 1.25);
    P->tc = e * EP + 0.45 * EP;
    P->Hpk = rr(&r, 200, 260);
    P->Tb = 2.7 * P->Hpk / P->v;
    /* where it crosses: of a dozen tries, the one whose track (or, for a
     * breach, where it breaks the sand) the camera can best see */
    double best = -1;
    for (int k = 0; k < 12; k++) {
        Pass Q = *P;
        double D = P->breach ? rr(&r, 950, 1250) : rr(&r, 900, 2600);
        double ang = cam_psi + (P->breach ? rr(&r, -0.1, 0.1) : rr(&r, -0.3, 0.3));
        Q.cx = camX + D * sin(ang); Q.cz = camZ + D * cos(ang);
        double pa = ang + (rnd(&r) < 0.5 ? 1 : -1) * M_PI / 2 + rr(&r, -0.5, 0.5);
        Q.dx = sin(pa); Q.dz = cos(pa);
        double score = rnd(&r) * 0.5, x, z;
        for (int i = -2; i <= 2; i++) {
            double sp = P->breach ? i * P->v * P->Tb / 4 : i * 250.0;
            path_at(&Q, sp, &x, &z);
            score += in_sight(x, z) * (P->breach && abs(i) == 2 ? 2 : 1);
        }
        if (score > best) { best = score; *P = Q; }
    }
    cache = *P;
    cached = e;
}

static void path_at(const Pass *P, double s, double *x, double *z) {
    double m = P->am * (sin(s / P->lm + P->ph) - sin(P->ph));
    *x = P->cx + P->dx * s + P->dz * m;
    *z = P->cz + P->dz * s - P->dx * m;
}

/* the level the worm swims relative to: the sand's surface, smoothed */
static double ref_level(double x, double z) {
    return terrain((float)x, (float)z, 60.0f);
}

static double breach_start(const Pass *P) { return P->tc - P->Tb / 2; }

/* the head's height over the reference level at time tp: buried, except it
 * fades in from the deep at the start of its episode and back out at the
 * end; before a breach it sinks out of sight, then comes up in an arch */
static double head_y(const Pass *P, double tp) {
    double te = tp - P->e * EP;
    double y = -P->dep - 3 * P->R * (1 - sstep(0, 30, te)) - 3 * P->R * sstep(EP - 30, EP, te);
    if (P->breach) {
        double u = (tp - breach_start(P)) / P->Tb;
        if (u > 0 && u < 1) y += (P->Hpk + P->dep) * pow(sin(M_PI * u), 0.85);
        else if (u > -0.9 && u <= 0) y -= 1.6 * P->R * sin(M_PI * (u + 0.9) / 0.9);
    }
    return y;
}

/* the body's radius at tau metres behind the snout */
static double body_r(const Pass *P, double tau) {
    double R = P->R, Lh = 2.3 * R;
    if (tau < Lh) { double u = tau / Lh; return R * fmax(0.12, sqrt(fmax(0, u * (2 - u)))); }
    double tl = P->Lb * 0.72;
    if (tau > tl) return R * (1 - 0.82 * pow((tau - tl) / (P->Lb - tl), 1.3));
    return R;
}

/* the reference level along the path, s metres from the crossing point:
 * worked out once per pass, every 4 m, as it's needed */
#define REFN 4096
static double path_ref(const Pass *P, double s) {
    static double tab[REFN];
    static long tab_e = -1;
    if (tab_e != P->e) { for (int i = 0; i < REFN; i++) tab[i] = NAN; tab_e = P->e; }
    double f = s / 4 + REFN / 2;
    int i = (int)floor(f);
    if (i < 0 || i >= REFN - 1) { double x, z; path_at(P, s, &x, &z); return ref_level(x, z); }
    for (int k = i; k <= i + 1; k++)
        if (isnan(tab[k])) { double x, z; path_at(P, (k - REFN / 2) * 4.0, &x, &z); tab[k] = ref_level(x, z); }
    return tab[i] + (tab[i + 1] - tab[i]) * (f - i);
}

/* a point of the body's centreline at time t: tau behind the head */
static void body_pt(const Pass *P, double t, double tau, double c[3]) {
    double s = P->v * (t - P->tc) - tau;
    path_at(P, s, &c[0], &c[2]);
    c[1] = path_ref(P, s) + head_y(P, t - tau / P->v);
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__
static const char *GL_HEAD = "#version 330 core\n#extension GL_ARB_texture_gather : require\n";

/* noise, the dune parameters, the light and the air: every program has these */
static const char *GL_COMMON = GLSL(
    uniform sampler2D uNz;
    uniform float uWx, uWz, uL1, uA1, uL2, uA2;
    uniform float uMesa[8];
    uniform vec3 uCam;
    uniform float uT, uPix;
    uniform vec3 uSunDir, uSunCol, uMoonDir, uMoonCol, uZen, uHor, uHaze, uGlow, uMoonGlow, uAmb;
    uniform float uFogK, uNight, uLowSun;

    float nz(float x, float y) {
        vec2 p = vec2(x, y);
        vec2 i = floor(p);
        vec2 f = p - i;
        f = f * f * (3.0 - 2.0 * f);
        vec4 g = textureGather(uNz, (i + 1.0) / 256.0);
        return mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y);
    }
    /* the colour of the air looking along d (below the horizon: at it) */
    vec3 hazeCol(vec3 d) {
        float y = max(d.y, 0.0);
        vec3 sd = normalize(vec3(uSunDir.x, max(uSunDir.y, 0.0), uSunDir.z));
        /* with the sun low, the horizon away from it goes dim and violet */
        float ma = dot(normalize(vec3(d.x, 0.0, d.z) + 1e-5), normalize(vec3(sd.x, 0.0, sd.z) + 1e-5));
        float away = uLowSun * (1.0 - pow(0.5 + 0.5 * ma, 1.5));
        vec3 hor = mix(uHor, uHor * vec3(0.42, 0.40, 0.55), away);
        vec3 c = mix(hor, uZen, pow(y, 0.4));
        c = mix(c, mix(uHaze, hor, 0.5 * away), exp(-y * 14.0) * 0.7);
        float mu = max(dot(d, sd), 0.0);
        c += uGlow * (0.30 * pow(mu, 5.0) + 0.9 * pow(mu, 48.0)) * exp(-y * 2.5);
        float mm = max(dot(d, uMoonDir), 0.0);
        c += uMoonGlow * (pow(mm, 24.0) + 0.3 * pow(mm, 4.0));
        return c;
    }
    vec3 fogged(vec3 c, vec3 rel) {
        float d = length(rel);
        return mix(c, hazeCol(rel / d), 1.0 - exp(-d * uFogK));
    }
);

/* things the terrain shaders share: the polar grid, the worm's height map,
 * the shock ripples */
static const char *GL_GRID = GLSL(
    uniform float uPsi, uColK, uI0, uJ0, uDmin, uS0, uD1, uKap;
    uniform sampler2D uDist;
    uniform vec4 uMapBox;
    uniform float uMapOn;
    uniform vec4 uShock[2];
    float ringD(float j) { return j < uJ0 ? uDmin + uS0 * j : uD1 * exp(uKap * (j - uJ0)); }
    float ringS(float j) { return j < uJ0 ? uS0 : uKap * ringD(j); }
    vec2 gridXZ(float ci, float j) {
        float th = uPsi + atan((uI0 + ci) * uColK);
        return uCam.xz + ringD(j) * vec2(sin(th), cos(th));
    }
    vec4 distS(vec2 p) {
        if (uMapOn < 0.5) return vec4(0.0);
        vec2 uv = (p - uMapBox.xy) / uMapBox.z;
        if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0);
        return textureLod(uDist, uv, 0.0);
    }
    /* shock ripples: a packet running outwards from where the sand was
     * broken (xy), age z seconds, amplitude w metres */
    float shockH(vec2 p) {
        float h = 0.0;
        for (int i = 0; i < 2; i++) {
            vec4 s = uShock[i];
            if (s.w <= 0.0 || s.z < 0.0) continue;
            float r = length(p - s.xy), x = r - s.z * 55.0;
            h += s.w * exp(-s.z / 12.0) * exp(-x * x / 12000.0) * sin(x * 0.17) / (1.0 + r / 500.0);
        }
        return h;
    }
);

/* ---- the terrain ---- */
static const char *VS_TERR = GLSL(
    uniform mat4 uVP;
    uniform sampler2D uHgt;
    out vec3 vRel;
    void main() {
        float ci = float(gl_VertexID >> 1);
        float j = float(gl_InstanceID + (gl_VertexID & 1));
        vec2 xz = gridXZ(ci, j);
        float d = ringD(j);
        float h = texelFetch(uHgt, ivec2(int(ci), int(j)), 0).r;
        h += distS(xz).r + shockH(xz);
        h -= d * d / 12.742e6;                   /* the Earth curving away */
        vRel = vec3(xz.x - uCam.x, h - uCam.y, xz.y - uCam.z);
        gl_Position = uVP * vec4(vRel, 1.0);
    });

/* the grid only records how far away the sand is; the shading is done
 * once per pixel afterwards, not for every sliver of a triangle */
static const char *FS_GBUF = GLSL(
    in vec3 vRel;
    out vec4 o;
    void main() { o = vec4(length(vRel)); });

static const char *FS_TERR = GLSL(
    uniform vec3 uSand, uSand2, uRock;
    uniform float uWindy, uShadowSun, uMapTex;
    uniform vec4 uCap[24];
    uniform int uNCap;
    uniform vec4 uCapBox;                        /* where its shadow can fall: x0, z0, x1, z1 */

    /* the nearest approach of the ray p + t L (t > 0) to segment ab */
    float segRay(vec3 p, vec3 L, vec3 a, vec3 b) {
        vec3 u = b - a, w0 = a - p;
        float A = dot(u, u), B = dot(u, L), D = dot(u, w0), E = dot(L, w0);
        float den = max(A - B * B, 1e-4);
        float s = clamp((B * E - D) / den, 0.0, 1.0);
        float t = max(dot(L, a + s * u - p), 0.0);
        s = clamp(dot(p + t * L - a, u) / A, 0.0, 1.0);
        return length(a + s * u - p - t * L);
    }
    /* the worm's shadow, when some of it is out of the sand */
    float wormShadow(vec3 p, vec3 L) {
        float s = 1.0;
        for (int i = 0; i + 1 < uNCap; i++) {
            vec4 a = uCap[i], b = uCap[i + 1];
            if (a.w <= 0.0 || b.w <= 0.0) continue;
            float r = 0.5 * (a.w + b.w);
            s = min(s, 0.15 + 0.85 * smoothstep(r * 0.6, r * 1.4, segRay(p, L, a.xyz, b.xyz)));
        }
        return s;
    }
    /* wind ripples and megaripples, as a height gradient */
    vec2 ripG(vec2 p, float fp) {
        vec2 Wd = vec2(uWx, uWz);
        float qx = dot(p, Wd), qz = dot(p, vec2(-uWz, uWx));
        vec2 g = vec2(0.0);
        float f1 = 1.0 - smoothstep(0.04, 0.22, fp);
        if (f1 > 0.0) {
            float s = 6.2832 * (qx / 0.7 + 2.5 * nz(qx * 0.4, qz * 0.15) + 1.2 * nz(qx * 0.08 + 3.0, qz * 0.05));
            g += Wd * cos(s + 0.7 * sin(s)) * (1.0 + 0.7 * cos(s)) * (0.012 * 6.2832 / 0.7) * f1;
        }
        float f2 = 1.0 - smoothstep(0.35, 1.4, fp);
        if (f2 > 0.0) {
            float s = 6.2832 * (qx / 4.5 + 1.5 * nz(qx * 0.06, qz * 0.025 + 4.0));
            g += Wd * cos(s + 0.7 * sin(s)) * (1.0 + 0.7 * cos(s)) * (0.04 * 6.2832 / 4.5) * f2;
        }
        return g;
    }

    /* what doesn't change about the sand seen at vRel: its colour, its
     * normal (ripples and all), how much it smokes; erase smooths the
     * ripples away where the worm has stirred it */
    void sandStatic(vec3 vRel, float erase, out vec3 alb, out vec3 n, out float veil) {
        vec3 p = vRel + uCam;
        float d = length(vRel);
        float fp = max(d < uD1 ? uS0 : uKap * d, d * uPix * 1.5);
        float e = max(d * uPix * 0.75, 0.05);
        float h0 = terrain(p.x, p.z, fp);
        float rel = tRel, rock = tRock, A = tA, uu = tU, cc = tC;
        float hx = terrain(p.x + e, p.z, fp), hz = terrain(p.x, p.z + e, fp);
        vec2 g = vec2(hx - h0, hz - h0) / e;
        float sl0 = length(g);
        /* ripples on the gentle slopes, not on slip faces or rock */
        float rs = (1.0 - smoothstep(0.3, 0.55, sl0)) * (1.0 - erase) * (1.0 - rock);
        /* (faded by the footprint along the slope: seen edge-on they'd shimmer) */
        float graze = max(abs(dot(vRel / d, normalize(vec3(-g.x, 1.0, -g.y)))), 0.06);
        g += ripG(p.xz, d * uPix / graze) * rs;
        n = normalize(vec3(-g.x, 1.0, -g.y));
        /* the sand: lighter towards the crests, grey grit in the hollows */
        float mott = nz(p.x * 0.011, p.z * 0.011) * 0.6 + nz(p.x * 0.05 + 7.0, p.z * 0.05) * 0.4;
        alb = uSand * (0.9 + 0.2 * mott);
        float hollow = (1.0 - smoothstep(0.03, 0.3, rel)) * (1.0 - smoothstep(0.35, 0.9, A / uA1));
        alb = mix(alb, uSand2 * (0.85 + 0.3 * mott), hollow * 0.85);
        alb *= 1.0 + 0.07 * smoothstep(0.7, 1.0, rel);
        /* slip faces: fresh, a touch paler, in long avalanche streaks */
        float lee = smoothstep(0.45, 0.6, sl0) * step(cc, uu);
        alb *= 1.0 + lee * (0.05 + 0.06 * (nz(dot(p.xz, vec2(-uWz, uWx)) * 0.12, uu * 3.0) - 0.5));
        /* the buttes: banded rock */
        float band = 0.5 + 0.5 * sin(p.y * 0.11 + 2.0 * nz(p.x * 0.01, p.z * 0.01));
        alb = mix(alb, uRock * (0.75 + 0.35 * band) * (0.9 + 0.2 * mott), rock);
        /* where sand smokes off the crests: just downwind of the brink */
        float z = (uu - cc) / max(1.0 - cc, 0.05);
        veil = smoothstep(-0.04, 0.01, z) * (1.0 - smoothstep(0.01, 0.3, z)) * smoothstep(0.35, 0.8, A / uA1);
        veil *= (1.0 - smoothstep(1500.0, 5000.0, d)) * (1.0 - rock);
    }

    /* and what does: the worm's work on it, the light, the blowing sand */
    vec3 sandLit(vec3 vRel, vec3 alb, vec3 n, float veil, float vSh, vec4 ds) {
        vec3 p = vRel + uCam;
        float d = length(vRel);
        vec3 vd = vRel / d;
        /* swell, rim and furrow, shock ripples: tilt the normal */
        vec2 gd = vec2(0.0), gs = vec2(0.0);
        if (ds.g + ds.b > 0.001 || ds.r != 0.0) {
            float tx = uMapTex;
            gd = vec2(distS(p.xz + vec2(tx, 0.0)).r - distS(p.xz - vec2(tx, 0.0)).r,
                      distS(p.xz + vec2(0.0, tx)).r - distS(p.xz - vec2(0.0, tx)).r) / (2.0 * tx);
        }
        if (uShock[0].w + uShock[1].w > 0.0) {
            float se = max(d * uPix, 2.0);
            gs = vec2(shockH(p.xz + vec2(se, 0.0)) - shockH(p.xz - vec2(se, 0.0)),
                      shockH(p.xz + vec2(0.0, se)) - shockH(p.xz - vec2(0.0, se))) / (2.0 * se);
        }
        vec2 g = -n.xz / n.y + gd + gs;
        n = normalize(vec3(-g.x, 1.0, -g.y));
        /* stirred sand: darker, streaming down the swell's flanks */
        float act = clamp(ds.g, 0.0, 1.0), trail = clamp(ds.b, 0.0, 1.0);
        if (act + trail > 0.001) {
            vec2 dn = length(gd) > 1e-4 ? normalize(gd) : vec2(1.0, 0.0);
            float along = dot(p.xz, dn), across = dot(p.xz, vec2(-dn.y, dn.x));
            float st = nz(across * 0.16, along * 0.03 + uT * 0.9) * 0.65 + nz(across * 0.5 + 9.0, along * 0.08 + uT * 1.7) * 0.35;
            float fl = act * (0.4 + 0.6 * smoothstep(0.05, 0.3, length(gd)));
            alb *= 1.0 - 0.16 * trail - 0.14 * act + fl * 0.6 * (st - 0.5);
        }
        /* the light: sun or moon, cast shadows from the grid, the worm's */
        float shS = uShadowSun > 0.5 ? vSh : 1.0, shM = uShadowSun > 0.5 ? 1.0 : vSh;
        if (uNCap > 1 && p.x > uCapBox.x && p.z > uCapBox.y && p.x < uCapBox.z && p.z < uCapBox.w) {
            if (uShadowSun > 0.5) shS *= wormShadow(p, uSunDir);
            else shM *= wormShadow(p, uMoonDir);
        }
        float ds_ = max(dot(n, uSunDir) * 0.92 + 0.08, 0.0) * shS;
        float dm = max(dot(n, uMoonDir) * 0.92 + 0.08, 0.0) * shM;
        vec3 c = alb * (uSunCol * ds_ + uMoonCol * dm + uAmb * (0.55 + 0.45 * n.y));
        /* a little sunlit sand bounces into the shadows */
        c += alb * alb * uSunCol * max(uSunDir.y, 0.0) * 0.12 * (1.0 - n.y * 0.5);
        /* sand smoking off the crests, downwind over the slip faces */
        if (uWindy > 0.0 && veil > 0.0) {
            vec2 Wd = vec2(uWx, uWz);
            float qx = dot(p.xz, Wd), qz = dot(p.xz, vec2(-uWz, uWx));
            float fl = nz(qz * 0.08, qx * 0.02 - uT * 0.9) * nz(qz * 0.21 + 5.0, qx * 0.05 - uT * 2.1);
            float a = veil * smoothstep(0.1, 0.6, fl) * uWindy * 0.3;
            float fw = pow(max(dot(vd, uSunDir), 0.0), 3.0);
            vec3 dust = uSand * (uSunCol * (0.35 + 1.2 * fw) + uAmb * 0.8);
            c = mix(c, dust, a);
        }
        return fogged(c, vRel);
    }
);

/* ---- the sky ---- */
static const char *VS_FULL = GLSL(
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, 1.0, 1.0);
    });

static const char *GL_RAY = GLSL(
    uniform vec4 uCanvas;                        /* this view: x, y, canvas units per device px */
    uniform vec3 uCanvasC;                       /* centre x, y, focal length */
    uniform vec3 uF, uR, uU;
    uniform vec2 uRes;
    vec2 canvasAt(vec2 fc) { return vec2(uCanvas.x + fc.x * uCanvas.z, uCanvas.y + (uRes.y - fc.y) * uCanvas.w); }
    vec3 rayAt(vec2 cp) {
        return normalize(uF + uR * (cp.x - uCanvasC.x) / uCanvasC.z - uU * (cp.y - uCanvasC.y) / uCanvasC.z);
    }
);

static const char *FS_SKY = GLSL(
    uniform vec3 uCloudCol;
    uniform float uCloud, uStarRot, uMoonI, uMoonSide, uMoonR;
    float h31(vec3 p) {
        uvec3 q = uvec3(ivec3(p) + 32768);
        uint h = q.x * 1597334677u ^ q.y * 3812015801u ^ q.z * 2798796415u;
        h = (h ^ (h >> 15)) * 2246822519u;
        h ^= h >> 13;
        return float(h & 0xffffffu) / 16777216.0;
    }
    vec3 skyColour(vec3 d) {
        vec3 c = hazeCol(d);
        float y = d.y;
        float clear = 1.0;
        /* high cirrus, streaked along the wind */
        if (y > 0.0 && uCloud > 0.0) {
            float t = min((7000.0 - uCam.y) / y, 2.0e5);
            vec2 q = uCam.xz + d.xz * t + vec2(uWx, uWz) * uT * 6.0;
            float qa = dot(q, vec2(uWx, uWz)), qb = dot(q, vec2(-uWz, uWx));
            float n = nz(qa * 0.00022, qb * 0.0011) * 0.55 + nz(qa * 0.0009 + 3.0, qb * 0.004) * 0.3 + nz(qa * 0.003, qb * 0.012 + 5.0) * 0.15;
            float cov = smoothstep(1.0 - uCloud, 1.0 - uCloud + 0.28, n) * smoothstep(0.0, 0.12, y);
            float mu = max(dot(d, normalize(vec3(uSunDir.x, max(uSunDir.y, 0.0), uSunDir.z))), 0.0);
            vec3 cc = uCloudCol * (1.0 + 1.5 * pow(mu, 6.0));
            c = mix(c, cc, cov * 0.65);
            clear = 1.0 - cov * 0.6;
        }
        if (y > -0.01) {
            float ext = smoothstep(-0.01, 0.12, y);
            /* stars, turning slowly about a pole behind the camera */
            if (uNight > 0.0) {
                float ca = cos(uStarRot), sa = sin(uStarRot);
                vec3 sd = vec3(ca * d.x - sa * d.y, sa * d.x + ca * d.y, d.z);
                vec3 q = sd * 300.0;
                vec3 cell = floor(q);
                float hs = h31(cell);
                if (hs > 0.975) {
                    vec3 sp = cell + 0.2 + 0.6 * vec3(h31(cell + 17.0), h31(cell + 41.0), h31(cell + 73.0));
                    float ang = length(q - sp) / 300.0;
                    float mag = pow((hs - 0.975) / 0.025, 5.0);
                    float tw = 0.8 + 0.2 * sin(uT * (2.0 + 3.0 * h31(cell + 5.0)) + hs * 60.0);
                    float px = uPix * 0.75;
                    c += vec3(0.85, 0.9, 1.0) * (0.03 + 0.6 * mag) * tw * exp(-ang * ang / (px * px)) * uNight * ext * clear;
                }
            }
            /* the moon, in its phase */
            if (dot(d, uMoonDir) > 0.0 && length(cross(d, uMoonDir)) < uMoonR * 1.5) {
                vec3 mr = normalize(cross(uMoonDir, vec3(0.0, 1.0, 0.0)));
                vec3 mu_ = cross(mr, uMoonDir);
                vec2 l = vec2(dot(d, mr), dot(d, mu_)) / sin(uMoonR);
                float r2 = dot(l, l);
                float edge = 1.0 - smoothstep(1.0 - 2.0 * uPix / uMoonR, 1.0, sqrt(r2));
                if (edge > 0.0) {
                    vec3 n = vec3(l, sqrt(max(1.0 - r2, 0.0)));
                    vec3 L = vec3(-uMoonSide * sin(uMoonI), 0.0, cos(uMoonI));
                    float lit = smoothstep(-0.03, 0.08, dot(n, L));
                    float mare = 0.75 + 0.25 * nz(l.x * 3.0 + 5.0, l.y * 3.0) - 0.2 * smoothstep(0.55, 0.7, nz(l.x * 2.0 + 9.0, l.y * 2.0 + 1.0));
                    vec3 mc = vec3(1.0, 0.97, 0.9) * (lit * mare * 1.6 + 0.012) * ext;
                    /* by day just its lit part, pale against the blue */
                    c = mix(c, c * (1.0 - 0.7 * uNight), edge * clear) + mc * edge * clear * mix(0.25, 1.0, uNight);
                }
            }
            /* the sun */
            /* (by the sine of the angle off it: cos() is too coarse on a GPU) */
            float sr = 0.0052;
            float disc = dot(d, uSunDir) > 0.0 ? 1.0 - smoothstep(sr, sr + 1.5 * uPix, length(cross(d, uSunDir))) : 0.0;
            c += uSunCol * 22.0 * disc * clear * ext;
        }
        return c;
    }
);

/* every pixel: sand where the grid put some, else sky */
static const char *FS_SHADE = GLSL(
    out vec4 o;
    uniform sampler2D uGbuf, uShadow, uCacheA, uCacheB;
    uniform vec2 uShSize;
    void main() {
        vec2 cp = canvasAt(gl_FragCoord.xy);
        vec3 d = rayAt(cp);
        float t = texelFetch(uGbuf, ivec2(gl_FragCoord.xy), 0).r;
        if (t <= 0.0) { o = vec4(skyColour(d), 1.0); return; }
        vec3 rel = d * t;
        /* the cast shadow, from the grid point(s) here */
        float hd = length(rel.xz);
        float th = atan(rel.x, rel.z) - uPsi;
        th = mod(th + 3.14159265, 6.2831853) - 3.14159265;
        float ci = tan(th) / uColK - uI0;
        float j = hd < uD1 ? (hd - uDmin) / uS0 : uJ0 + log(hd / uD1) / uKap;
        float sh = texture(uShadow, (vec2(ci, j) + 0.5) / uShSize).r;
        /* the static look from the cache, unless the worm has moved the
         * sand here (or stirred it) */
        ivec2 px = ivec2(gl_FragCoord.xy);
        vec4 ca = texelFetch(uCacheA, px, 0), cb = texelFetch(uCacheB, px, 0);
        vec4 ds = distS(rel.xz + uCam.xz);
        vec3 alb = ca.rgb, n = cb.xyz;
        float veil = ca.a;
        if (abs(t - cb.w) > 0.02 * t + 1.0 || ds.g + ds.b > 0.02)
            sandStatic(rel, max(ds.g, ds.b), alb, n, veil);
        o = vec4(sandLit(rel, alb, n, veil, sh, ds), 1.0);
    });

/* building that cache: from a grid drawn without the worm */
static const char *FS_CACHE = GLSL(
    layout(location = 0) out vec4 oA;
    layout(location = 1) out vec4 oB;
    uniform sampler2D uGbuf;
    void main() {
        float t = texelFetch(uGbuf, ivec2(gl_FragCoord.xy), 0).r;
        if (t <= 0.0) { oA = vec4(0.0); oB = vec4(0.0); return; }
        vec3 rd = rayAt(canvasAt(gl_FragCoord.xy));
        /* the grid's triangles only approximate the sand: slide along the
         * ray onto the real surface, or the shading would carry the grid's
         * ring pattern (a few secant steps, kept close to where we were) */
        float hd = length(rd.xz) * t;
        float fp = max(hd < uD1 ? uS0 : uKap * hd, hd * uPix * 1.5);
        float ta = t, tb = t * 1.003 + 0.05;
        vec3 pa = uCam + rd * ta, pb = uCam + rd * tb;
        float fa = pa.y - terrain(pa.x, pa.z, fp) + ta * ta * dot(rd.xz, rd.xz) / 12.742e6;
        float fb = pb.y - terrain(pb.x, pb.z, fp) + tb * tb * dot(rd.xz, rd.xz) / 12.742e6;
        float tr = t, best = abs(fa);
        for (int k = 0; k < 3; k++) {
            if (abs(fb - fa) < 1e-6) break;
            float tn = clamp(tb - fb * (tb - ta) / (fb - fa), t * 0.97 - 1.0, t * 1.03 + 1.0);
            vec3 pn = uCam + rd * tn;
            float fn = pn.y - terrain(pn.x, pn.z, fp) + tn * tn * dot(rd.xz, rd.xz) / 12.742e6;
            if (abs(fn) < best) { best = abs(fn); tr = tn; }
            ta = tb; fa = fb; tb = tn; fb = fn;
        }
        vec3 rel = rd * tr;
        vec3 alb, n;
        float veil;
        sandStatic(rel, 0.0, alb, n, veil);
        oA = vec4(alb, veil);
        oB = vec4(n, t);
    });

/* the grid's static heights, once */
static const char *FS_HGT = GLSL(
    out vec4 o;
    void main() {
        float ci = floor(gl_FragCoord.x), j = floor(gl_FragCoord.y);
        vec2 xz = gridXZ(ci, j);
        float d = ringD(j);
        o = vec4(terrain(xz.x, xz.y, max(ringS(j), d * uPix * 1.5)));
    });

/* ---- the worm ---- */
static const char *VS_WORM = GLSL(
    uniform mat4 uVP;
    layout(location = 0) in vec3 aP;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec3 aT;
    layout(location = 3) in vec3 aUV;
    out vec3 vRel, vN, vT, vUV;
    void main() {
        vRel = aP; vN = aN; vT = aT; vUV = aUV;
        gl_Position = uVP * vec4(aP, 1.0);
    });

static const char *FS_WORM = GLSL(
    in vec3 vRel, vN, vT, vUV;
    out vec4 o;
    uniform vec3 uSand, uSkinA, uSkinB, uSkinC;
    uniform float uSegL, uDusty, uShadowSun;
    void main() {
        vec3 n = normalize(vN), T = normalize(vT);
        float tau = vUV.x, phi = vUV.y, head = vUV.z;
        vec3 p = vRel + uCam;
        /* overlapping plates: each rises a little towards its trailing edge,
         * then steps down under the next */
        float segL = uSegL * mix(1.0, 1.9, head);
        float s = tau / segL;
        float fs = fract(s);
        float dh = fs < 0.86 ? 0.25 : -1.6;
        float pw = 1.0 - smoothstep(2.0, 6.0, length(vRel) * uPix / segL * 40.0);
        n = normalize(n - T * dh * 0.35 * pw);
        float seam = smoothstep(0.82, 0.9, fs) * (1.0 - smoothstep(0.94, 1.0, fs));
        /* colour: dark along the back, paler flanks, a pale ribbed belly */
        float ap = abs(phi);
        vec3 alb = mix(uSkinA, uSkinB, smoothstep(0.5, 1.6, ap));
        alb = mix(alb, uSkinC, smoothstep(2.0, 2.6, ap));
        /* each plate its own shade, worn paler towards its trailing edge */
        float pl = floor(s);
        float ph_ = fract(sin(pl * 12.9898) * 43758.5453);
        alb *= (0.85 + 0.3 * ph_) * (0.9 + 0.2 * smoothstep(0.2, 0.85, fs));
        float mott = nz(tau * 0.35 + phi * 5.0, phi * 9.0 + pl * 0.37);
        alb *= 0.85 + 0.3 * mott;
        alb *= 1.0 - 0.6 * seam * pw;
        /* the keels: two low ridges down the back, worn pale */
        alb = mix(alb, uSkinC, exp(-pow((ap - 0.55) / 0.09, 2.0)) * 0.45);
        /* a fine coat of sand on the upper surfaces */
        float dust = uDusty * smoothstep(0.15, 0.85, n.y) * (0.55 + 0.45 * nz(tau * 1.3, phi * 20.0));
        alb = mix(alb, uSand * 0.95, clamp(dust, 0.0, 0.8));
        /* the light, with the dunes' shadow on it */
        vec3 L = uShadowSun > 0.5 ? uSunDir : uMoonDir;
        float sh = 1.0, t = 3.0;
        for (int k = 0; k < 6; k++) {
            vec3 q = p + L * t;
            sh = min(sh, 4.0 * (q.y - terrain(q.x, q.z, t * 0.05)) / t);
            t = t * 2.0 + 4.0;
        }
        sh = clamp(sh, 0.0, 1.0);
        float ds_ = max(dot(n, uSunDir), 0.0) * (uShadowSun > 0.5 ? sh : 1.0);
        float dm = max(dot(n, uMoonDir), 0.0) * (uShadowSun > 0.5 ? 1.0 : sh);
        vec3 c = alb * (uSunCol * ds_ + uMoonCol * dm + uAmb * (0.6 + 0.4 * n.y));
        /* the sky behind it catches the rim */
        float rim = pow(1.0 - max(dot(n, -normalize(vRel)), 0.0), 3.0);
        c += alb * hazeCol(normalize(vRel)) * rim * 0.35;
        /* a low sheen on the plates */
        vec3 vd = normalize(vRel);
        vec3 hv = normalize(uSunDir - vd);
        c += uSunCol * 0.05 * pow(max(dot(n, hv), 0.0), 24.0) * sh * (1.0 - dust);
        /* sand-lit from below */
        c += uSand * uSunCol * max(uSunDir.y, 0.0) * 0.08 * max(-n.y, 0.0);
        o = vec4(fogged(c, vRel), 1.0);
    });

/* ---- falling sand and dust ---- */
static const char *VS_PART = GLSL(
    uniform mat4 uVP;
    uniform vec3 uRw, uUw;
    layout(location = 0) in vec4 aA;             /* position (rel. camera), size */
    layout(location = 1) in vec4 aB;             /* alpha, kind, stretch, seed */
    uniform vec3 uSand;
    out vec2 vQ;
    out vec3 vCol, vFog;
    out float vAlpha, vKind, vSeed, vFogA;
    void main() {
        vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
        vec3 pos = aA.xyz + uRw * c.x * aA.w + uUw * c.y * aA.w * aB.z;
        vQ = c; vAlpha = aB.x; vKind = aB.y; vSeed = aB.w;
        /* lit and fogged once per corner: it's all one colour anyway */
        vec3 vd = normalize(aA.xyz);
        /* forward scattering, toned down when the sun is low and bright behind */
        float fw = pow(max(dot(vd, uSunDir), 0.0), 4.0) * mix(0.35, 1.0, smoothstep(0.05, 0.3, uSunDir.y));
        if (aB.y < 0.5) {
            vec3 sa = mix(uSand, vec3(0.62, 0.52, 0.40), 0.45);
            vCol = sa * (uSunCol * (0.7 + 1.0 * fw) + uMoonCol * 0.6 + uAmb * 0.8);
        } else {
            vec3 dust = mix(uSand, vec3(0.6), 0.3);
            float top = 0.5 + 0.5 * c.y;
            vCol = dust * (uSunCol * (0.25 + 0.55 * top + 1.2 * fw) + uMoonCol * (0.3 + 0.5 * top) + uAmb * (0.7 + 0.4 * top));
        }
        vCol *= 1.0 - 0.55 * uNight;
        float d = length(aA.xyz);
        vFog = hazeCol(vd);
        vFogA = 1.0 - exp(-d * uFogK);
        gl_Position = uVP * vec4(pos, 1.0);
    });

static const char *FS_PART = GLSL(
    in vec2 vQ;
    in vec3 vCol, vFog;
    in float vAlpha, vKind, vSeed, vFogA;
    out vec4 o;
    void main() {
        float r2 = dot(vQ, vQ);
        if (r2 > 1.0) discard;
        float a;
        if (vKind < 0.5) a = exp(-r2 * 3.0) * vAlpha;
        else {
            float n = nz(vQ.x * 1.6 + vSeed * 31.0, vQ.y * 1.6 + vSeed * 17.0) * 0.6 + nz(vQ.x * 4.0 + vSeed * 7.0, vQ.y * 4.0 - vSeed * 5.0) * 0.4;
            a = (1.0 - smoothstep(0.15, 1.0, r2)) * smoothstep(0.2, 0.75, n + 0.35 * (1.0 - r2)) * vAlpha;
        }
        vec3 c = mix(vCol, vFog, vFogA);
        o = vec4(c * a, a);
    });

/* ---- the worm's height map: swell, rim, furrow; stirring ---- */
static const char *FS_DIST = GLSL(
    out vec4 o;
    uniform vec4 uSeg[96];                       /* x, z, centreline y, radius */
    uniform vec4 uSegB[96];                      /* age (s), worm radius, body?, fade */
    uniform int uNSeg;
    uniform vec4 uBox;                           /* x0, z0, texel */
    void main() {
        vec2 p = uBox.xy + gl_FragCoord.xy * uBox.z;
        /* the sand here, only if the body passes near */
        bool near = false;
        for (int k = 0; k + 1 < uNSeg; k++) {
            vec4 a = uSeg[k], b = uSeg[k + 1];
            if (uSegB[k].z < 0.5) break;
            vec2 ab = b.xy - a.xy;
            float tt = clamp(dot(p - a.xy, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
            if (length(p - a.xy - ab * tt) < uSegB[k].y * 6.0) { near = true; break; }
        }
        float Hs = near ? terrain(p.x, p.y, uBox.z) : 0.0;
        float up = 0.0, act = 0.0;
        float tdist = 1e9, tage = 0.0, tR = 1.0, tfade = 0.0;
        for (int k = 0; k + 1 < uNSeg; k++) {
            vec4 a = uSeg[k], b = uSeg[k + 1];
            vec4 A = uSegB[k], B = uSegB[k + 1];
            vec2 ab = b.xy - a.xy;
            float tt = clamp(dot(p - a.xy, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
            float dist = length(p - a.xy - ab * tt);
            float R = A.y, fade = mix(A.w, B.w, tt);
            if (dist > R * 6.0) continue;
            if (A.z > 0.5 && B.z > 0.5) {
                float yc = mix(a.z, b.z, tt), r = mix(a.w, b.w, tt);
                float top = yc + r, bot = yc - r;
                float c = 0.0;
                if (top < Hs) {
                    float dp = Hs - top;
                    float w = 1.1 * R + 0.45 * dp;
                    c = 0.75 * R * exp(-dp / (0.9 * R)) * exp(-dist * dist / (w * w));
                    act = max(act, exp(-dp / (1.2 * R)) * exp(-dist * dist / (2.0 * w * w)) * fade);
                } else if (bot < Hs) {
                    float x = max(dist - r, 0.0) / (0.9 * R);
                    c = 0.45 * R * exp(-x * x);
                    act = max(act, exp(-x * x * 0.5) * fade);
                }
                up = max(up, c * fade);
            } else if (dist < tdist) {
                tdist = dist; tage = mix(A.x, B.x, tt); tR = R; tfade = fade;
            }
        }
        float tr = 0.0, trail = 0.0;
        if (tdist < tR * 6.0) {
            float f = exp(-tage / 35.0) * (1.0 - smoothstep(60.0, 110.0, tage)) * tfade;
            float x = tdist / tR;
            tr = tR * f * (-0.12 * exp(-x * x / 0.8) + 0.08 * exp(-pow((x - 1.5) / 0.6, 2.0)));
            trail = f * exp(-x * x / 3.0);
        }
        o = vec4(up + tr, act, trail, 0.0);
    });

/* ---- the cast-shadow grid: march to the light from every grid point ---- */
static const char *FS_SHADOW = GLSL(
    out vec4 o;
    uniform vec3 uLightDir;
    uniform sampler2D uHgt;
    void main() {
        float ci = floor(gl_FragCoord.x), j = floor(gl_FragCoord.y);
        vec2 xz = gridXZ(ci, j);
        float d = ringD(j);
        float fp = max(ringS(j), d * uPix * 1.5);
        float h = texelFetch(uHgt, ivec2(int(ci), int(j)), 0).r;
        vec3 p = vec3(xz.x, h + 0.4 + 0.1 * fp, xz.y);
        vec3 L = uLightDir;
        float sh = 1.0, t = 1.5;
        for (int k = 0; k < 30; k++) {
            vec3 q = p + L * t;
            if (q.y > 420.0) break;
            float hq = terrain(q.x, q.z, max(t * 0.04, fp));
            sh = min(sh, 4.0 * (q.y - hq) / t);
            if (sh < 0.0) break;
            t = t * 1.21 + 1.5;
        }
        o = vec4(clamp(sh, 0.0, 1.0));
    });

/* ---- the last pass: heat shimmer, exposure, tone, grain ---- */
static const char *FS_POST = GLSL(
    out vec4 o;
    uniform sampler2D uScene;
    uniform float uExpo, uShim, uHorY, uFrH, uDev;
    uniform vec4 uVig;                           /* layout centre, half-size */
    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
    vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    void main() {
        vec2 cp = canvasAt(gl_FragCoord.xy);
        vec2 fc = gl_FragCoord.xy;
        /* the far hot sand just under the horizon wavers */
        float dy = (cp.y - uHorY) / uFrH;
        float band = smoothstep(-0.015, 0.01, dy) * (1.0 - smoothstep(0.02, 0.16, dy));
        if (uShim > 0.0 && band > 0.0) {
            float w = sin(cp.y * 0.45 + uT * 4.0 + 3.0 * sin(cp.x * 0.011 + uT * 0.6)) + 0.5 * sin(cp.y * 1.3 - uT * 7.0 + cp.x * 0.03);
            fc += vec2(w * 0.7, 0.35 * w) * band * uShim * uDev;
        }
        vec3 c = texture(uScene, fc / uRes).rgb * uExpo;
        /* by moonlight the eye loses colour */
        float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(c, lum * vec3(0.78, 0.9, 1.15), uNight * 0.65);
        c = aces(c);
        c = pow(c, vec3(1.0 / 2.2));
        /* a soft vignette over the whole wall */
        vec2 v = (cp - uVig.xy) / uVig.zw;
        c *= 1.0 - 0.22 * smoothstep(0.5, 1.5, dot(v, v));
        c += (h21(gl_FragCoord.xy + fract(uT) * 91.0) - 0.5) * (1.5 / 255.0);
        o = vec4(c, 1.0);
    });

static const char *VS_TEXT = GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in vec4 aCol;
    uniform vec2 uRes;
    out vec2 vUV;
    out vec4 vCol;
    void main() {
        vUV = aUV; vCol = aCol;
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_TEXT = GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vCol;
    out vec4 o;
    void main() { o = vec4(vCol.rgb, vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r); });

static GLuint compile(GLenum type, const char **parts, int n) {
    const char *src[10];
    src[0] = GL_HEAD;
    for (int i = 0; i < n; i++) src[i + 1] = parts[i];
    GLuint s = glCreateShader(type);
    glShaderSource(s, n + 1, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "sandworm: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}
static GLuint mk_prog(const char **vs, int nv, const char **fs, int nf) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs, nv));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs, nf));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "sandworm: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}
#define U(p, n) glGetUniformLocation(p, n)

/* ---- palettes ---- */
static const struct palette {
    const char *name;
    float sand[3], sand2[3], rock[3], haze[3];
    float skinA[3], skinB[3], skinC[3];
    const char *pin;                   /* a time of day it holds, or NULL */
} palettes[] = {
    {"ochre", {0.86f, 0.60f, 0.35f}, {0.62f, 0.47f, 0.33f}, {0.50f, 0.32f, 0.21f}, {1.00f, 0.92f, 0.80f},
     {0.38f, 0.30f, 0.24f}, {0.60f, 0.48f, 0.38f}, {0.90f, 0.77f, 0.60f}, NULL},
    {"rust",  {0.80f, 0.45f, 0.25f}, {0.56f, 0.34f, 0.22f}, {0.40f, 0.20f, 0.13f}, {1.00f, 0.84f, 0.72f},
     {0.36f, 0.26f, 0.21f}, {0.60f, 0.43f, 0.35f}, {0.87f, 0.68f, 0.54f}, NULL},
    {"pale",  {0.92f, 0.84f, 0.70f}, {0.72f, 0.66f, 0.57f}, {0.55f, 0.45f, 0.38f}, {1.00f, 0.97f, 0.93f},
     {0.42f, 0.38f, 0.33f}, {0.66f, 0.60f, 0.52f}, {0.95f, 0.89f, 0.78f}, NULL},
    {"dusk",  {0.88f, 0.58f, 0.34f}, {0.62f, 0.45f, 0.33f}, {0.50f, 0.30f, 0.20f}, {1.00f, 0.88f, 0.80f},
     {0.38f, 0.30f, 0.24f}, {0.60f, 0.48f, 0.38f}, {0.90f, 0.77f, 0.60f}, "dusk"},
    {"night", {0.86f, 0.78f, 0.66f}, {0.66f, 0.61f, 0.54f}, {0.50f, 0.42f, 0.36f}, {0.92f, 0.95f, 1.00f},
     {0.39f, 0.35f, 0.32f}, {0.63f, 0.57f, 0.51f}, {0.93f, 0.85f, 0.75f}, "night"},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float P_sand[3], P_sand2[3], P_rock[3], P_haze[3], P_skin[3][3];

static float lin(float c) { return powf(c, 2.2f); }

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "sandworm: no palette \"%s\" (ochre, rust, pale, dusk, night)\n", opt_palette);
            exit(1);
        }
    } else {
        uint32_t h = opt_seed * 2654435761u;
        h ^= h >> 15;
        pal = (int)(h % 3);            /* the ones that follow the clock */
    }
    const struct palette *P = &palettes[pal];
    for (int c = 0; c < 3; c++) {
        P_sand[c] = lin(P->sand[c]); P_sand2[c] = lin(P->sand2[c]); P_rock[c] = lin(P->rock[c]);
        P_haze[c] = P->haze[c];
        P_skin[0][c] = lin(P->skinA[c]); P_skin[1][c] = lin(P->skinB[c]); P_skin[2][c] = lin(P->skinC[c]);
    }
    if (!opt_time && P->pin) opt_time = P->pin;
}

/* ==== the light: the sun and moon on an artist's path through the day ====
 * The sun rises a little right of straight ahead, swings round behind the
 * camera (never higher than 36 degrees) and sets a little left of it, so
 * dawn and dusk are backlit and the day is side-lit. The moon crosses in
 * front, low, rising about 50 minutes later each day after its real phase. */
static struct {
    double sun_el, moon_el;
    float sun[3], sunc[3], moon[3], moonc[3], zen[3], hor[3], haze[3], glow[3], moonglow[3], amb[3], cloud[3];
    float night, expo, shim, fogk, moon_i, moon_side, shadow_sun;
} lt;

static void dir_of(double el, double azrel, float out[3]) {
    double az = cam_psi + azrel;
    out[0] = (float)(cos(el) * sin(az)); out[1] = (float)sin(el); out[2] = (float)(cos(el) * cos(az));
}

static void key(const double *els, const float (*tab)[3], int n, double el, float out[3]) {
    int i = 0;
    while (i < n - 2 && el > els[i + 1]) i++;
    double f = clamp01((el - els[i]) / (els[i + 1] - els[i]));
    for (int c = 0; c < 3; c++) out[c] = (float)(tab[i][c] + (tab[i + 1][c] - tab[i][c]) * f);
}

static void light_update(double wall) {
    time_t tt = (time_t)wall;
    struct tm tm;
    localtime_r(&tt, &tm);
    double hr = tm.tm_hour + tm.tm_min / 60.0 + (tm.tm_sec + (wall - floor(wall))) / 3600.0;
    if (tm.tm_isdst > 0) hr -= 1;
    double daylen = 12 + 1.8 * sin(2 * M_PI * (tm.tm_yday - 80) / 365.0);
    double rise = 12 - daylen / 2, set = 12 + daylen / 2;
    /* the moon: days since a known new moon */
    double ph = fmod((wall - 947182440.0) / 86400.0 / 29.530588853, 1.0);
    if (ph < 0) ph += 1;
    double mrise = rise + ph * 24.84, mx;
    int pinned_night = 0;
    if (opt_time && strcmp(opt_time, "real")) {
        int hh, mm;
        if (!strcmp(opt_time, "dawn")) hr = rise + 0.02 * daylen;
        else if (!strcmp(opt_time, "day")) hr = rise + 0.3 * daylen;
        else if (!strcmp(opt_time, "dusk")) hr = set - 0.018 * daylen;
        else if (!strcmp(opt_time, "night")) { hr = 0.5; pinned_night = 1; }
        else if (sscanf(opt_time, "%d:%d", &hh, &mm) == 2) hr = hh + mm / 60.0;
        else { fprintf(stderr, "sandworm: --time real, dawn, day, dusk, night or HH:MM\n"); exit(1); }
    }
    double x = (hr - rise) / (set - rise), el, azr;
    if (x >= 0 && x <= 1) { el = 36 * pow(sin(M_PI * x), 0.9); azr = 15 + 330 * x; }
    else {
        double xn = fmod(hr - set + 48, 24.0) / (24 - daylen);
        el = -55 * sin(M_PI * fmin(xn, 1.0));
        azr = xn < 0.5 ? 345 + 60 * xn : 15 - 60 * (1 - xn);
    }
    lt.sun_el = el;
    dir_of(el * M_PI / 180, azr * M_PI / 180, lt.sun);
    mx = fmod(hr - mrise + 48 * 24.84, 24.84) / 12.42;
    double illum = 0.5 - 0.5 * cos(2 * M_PI * ph);
    lt.moon_i = (float)(M_PI * fabs(1 - 2 * ph));    /* 0 full, pi new */
    lt.moon_side = ph < 0.5 ? 1.0f : -1.0f;          /* waxing: lit on the right */
    if (pinned_night) { mx = 0.3; illum = fmax(illum, 0.75); lt.moon_i = (float)fmin(lt.moon_i, 1.0); }
    double mel = mx <= 1 ? 12 * sin(M_PI * mx) : -20 * sin(M_PI * (mx - 1));
    lt.moon_el = mel;
    dir_of(mel * M_PI / 180, (40 - 80 * fmin(mx, 1.0)) * M_PI / 180, lt.moon);

    /* the sky by the sun's elevation */
    static const double els[] = {-20, -10, -5, -1, 2, 6, 12, 25, 40};
    static const float zen[][3] = {{0.0016f, 0.0024f, 0.0065f}, {0.005f, 0.008f, 0.025f}, {0.022f, 0.035f, 0.10f},
                                   {0.065f, 0.095f, 0.22f}, {0.12f, 0.18f, 0.38f}, {0.18f, 0.29f, 0.55f},
                                   {0.15f, 0.33f, 0.78f}, {0.11f, 0.31f, 0.92f}, {0.10f, 0.30f, 0.95f}};
    static const float hor[][3] = {{0.003f, 0.004f, 0.008f}, {0.016f, 0.017f, 0.030f}, {0.11f, 0.065f, 0.075f},
                                   {0.42f, 0.18f, 0.09f}, {0.70f, 0.32f, 0.13f}, {0.85f, 0.52f, 0.28f},
                                   {0.95f, 0.82f, 0.62f}, {0.90f, 0.87f, 0.78f}, {0.88f, 0.88f, 0.84f}};
    static const float glw[][3] = {{0, 0, 0}, {0.01f, 0.005f, 0.002f}, {0.22f, 0.07f, 0.03f},
                                   {0.95f, 0.35f, 0.09f}, {1.40f, 0.70f, 0.26f}, {1.30f, 0.90f, 0.50f},
                                   {1.00f, 0.90f, 0.70f}, {0.80f, 0.78f, 0.70f}, {0.70f, 0.70f, 0.65f}};
    key(els, zen, 9, el, lt.zen);
    key(els, hor, 9, el, lt.hor);
    key(els, glw, 9, el, lt.glow);
    /* the sun through the dusty air */
    double s = sin(fmax(el, -2.0) * M_PI / 180);
    double am = 1.0 / (fmax(s, 0) + 0.50572 * pow(fmax(el, 0) + 6.07995, -1.6364));
    static const double tau[3] = {0.065, 0.135, 0.30};
    double up = sstep(-0.8, 0.8, el);
    for (int c = 0; c < 3; c++) lt.sunc[c] = (float)(4.2 * exp(-tau[c] * fmin(am, 40)) * up);
    for (int c = 0; c < 3; c++) {
        lt.hor[c] *= P_haze[c];
        lt.haze[c] = (float)(lt.hor[c] * (0.85 + 0.15 * P_sand[c] / fmax(P_sand[0], 1e-3)));
        lt.amb[c] = (float)((lt.zen[c] * 0.65 + lt.hor[c] * 0.35) * (0.5 + 0.75 * sstep(3, 20, el)));
        /* the sky itself is brighter than the light it sheds */
        double gain = sstep(4, 18, el);
        lt.zen[c] *= (float)(1.3 + 1.9 * gain); lt.hor[c] *= (float)(0.8 + 1.2 * gain); lt.haze[c] *= (float)(0.8 + 1.2 * gain);
    }
    /* the moon */
    double mup = sstep(-1, 3, mel), mi = (0.25 + 0.75 * illum) * mup;
    static const float mc[3] = {0.36f, 0.44f, 0.62f};
    double night = clamp01((-4 - el) / 9);
    lt.night = (float)night;
    for (int c = 0; c < 3; c++) {
        lt.moonc[c] = (float)(mc[c] * 0.6 * mi * night);
        lt.moonglow[c] = (float)(mc[c] * 0.05 * mi * night);
        lt.amb[c] += (float)(night * (0.026f + 0.016f * c) * (0.6 + 0.6 * mi));
        lt.zen[c] += (float)(night * mi * mc[c] * 0.012);
        lt.hor[c] += (float)(night * mi * mc[c] * 0.02);
        lt.haze[c] += (float)(night * mi * mc[c] * 0.02);
    }
    for (int c = 0; c < 3; c++) {
        double gl = sstep(-6, 4, el);
        lt.cloud[c] = (float)(lt.glow[c] * 0.45 * (1 - sstep(4, 14, el)) * gl + lt.zen[c] * 0.6 + lt.hor[c] * 0.5 * sstep(4, 20, el) + lt.amb[c] * 0.3);
    }
    lt.shadow_sun = el > -1.0 ? 1.0f : 0.0f;
    /* exposure: keep lit sand mid-grey, whatever lights it */
    double sl = 0.2126 * lt.sunc[0] + 0.7152 * lt.sunc[1] + 0.0722 * lt.sunc[2];
    double al = 0.2126 * lt.amb[0] + 0.7152 * lt.amb[1] + 0.0722 * lt.amb[2];
    double ml = 0.2126 * lt.moonc[0] + 0.7152 * lt.moonc[1] + 0.0722 * lt.moonc[2];
    double hl = 0.2126 * lt.hor[0] + 0.7152 * lt.hor[1] + 0.0722 * lt.hor[2];
    double key_l = sl * fmax(s, 0.1) * 0.9 + al * 0.85 + ml * 0.55 + hl * 0.15 + 0.0012;
    lt.expo = (float)(0.30 / pow(key_l, 0.86) * (1 + 1.3 * night));
    lt.shim = (float)sstep(8, 22, el);
    lt.fogk = (float)((1.0 + 0.5 * (1 - sstep(2, 15, el))) / 14000.0);
}

/* ==== GL state ==== */
static GLuint prog_terr, prog_shade, prog_cache, prog_hgt, prog_worm, prog_part, prog_dist, prog_shadow, prog_post, prog_text;
static GLuint vao_empty, vao_worm, vbo_worm, ibo_worm, vao_part, vbo_part, vao_text, vbo_text;
static GLuint tex_nz, tex_shadow, tex_dist, tex_scene, tex_atlas;
static GLuint tex_g, fbo_shadow, fbo_dist, fbo_g, fbo_scene, rb_dep;
static GLuint tex_hgt, fbo_hgt, tex_ca, tex_cb, fbo_cache;
static int tg_w, tg_h, hgt_dirty = 1, cache_dirty = 1;

/* the polar grid: columns COLPX canvas px apart at the horizon, rings from
 * DMIN out, S0 apart near, then a fixed fraction KAP of their distance */
#define COLPX 2.5
#define DMIN 4.0
#define S0 0.45
#define KAP 0.011
#define DMAX 34000.0
static int g_i0, g_ncol, g_nring, g_J0;
static double g_d1;
static int sh_w, sh_h, sh_full = 1;
static long frame_no;

static void grid_update(void) {
    g_J0 = (int)ceil((S0 / KAP - DMIN) / S0);
    g_d1 = DMIN + S0 * g_J0;
    g_nring = g_J0 + (int)ceil(log(DMAX / g_d1) / KAP) + 1;
    /* the columns this view needs: the azimuths of its corners */
    double amin = 1e9, amax = -1e9;
    double Fh[3] = {sin(cam_psi), 0, cos(cam_psi)};
    for (int k = 0; k < 9; k++) {
        double x = cv_x + cv_w * (k % 3) / 2.0, y = cv_y + cv_h * (k / 3) / 2.0;
        double a = (x - cv_cx) / cv_f, b = (y - cv_cy) / cv_f, d[3];
        for (int c = 0; c < 3; c++) d[c] = cF[c] + cR[c] * a - cU[c] * b;
        double fh = d[0] * Fh[0] + d[2] * Fh[2], rh = d[0] * cR[0] + d[2] * cR[2];
        double az = fh > 1e-3 ? atan2(rh, fh) : (rh > 0 ? 1.4 : -1.4);
        az = fmax(-1.4, fmin(1.4, az));
        amin = fmin(amin, az); amax = fmax(amax, az);
    }
    int i0 = (int)floor(cv_f * tan(amin) / COLPX) - 3, i1 = (int)ceil(cv_f * tan(amax) / COLPX) + 3;
    if (i0 != g_i0 || i1 - i0 + 1 != g_ncol || sh_h != g_nring) {
        g_i0 = i0; g_ncol = i1 - i0 + 1;
        sh_w = g_ncol; sh_h = g_nring;
        glBindTexture(GL_TEXTURE_2D, tex_shadow);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, sh_w, sh_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_shadow, 0);
        glBindTexture(GL_TEXTURE_2D, tex_hgt);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sh_w, sh_h, 0, GL_RED, GL_FLOAT, NULL);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_hgt);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_hgt, 0);
        sh_full = 1; hgt_dirty = 1; cache_dirty = 1;
    }
}

static void tex_params(GLint wrap, GLint filt) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
}

/* the targets: the grid's distances, and the HDR scene; one depth buffer */
static void targets_update(void) {
    if (tg_w == W && tg_h == H) return;
    tg_w = W; tg_h = H;
    glBindRenderbuffer(GL_RENDERBUFFER, rb_dep);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glBindTexture(GL_TEXTURE_2D, tex_g);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, W, H, 0, GL_RED, GL_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_g);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_g, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_dep);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("grid framebuffer incomplete");
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_scene, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_dep);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("scene framebuffer incomplete");
    glBindTexture(GL_TEXTURE_2D, tex_ca);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, tex_cb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_cache);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_ca, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_cb, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("cache framebuffer incomplete");
    cache_dirty = 1;
}

/* ---- caption: the time, and how long the screen has been locked ---- */
static FT_Library ft;
static FT_Face face_cap;
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
static Glyph glyphs[2][96];
static int cap_px[2], atlas_w = 1024, atlas_h;
static float *tv;
static int tvn, tvcap;
static char cap_line[2][96];

static FT_Face load_face(const char *spec) {
    FcPattern *pat = FcNameParse((const FcChar8 *)spec);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    FT_Face f;
    if (FT_New_Face(ft, (const char *)file, 0, &f)) die("can't load the font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
    return f;
}

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    char spec[256];
    snprintf(spec, sizeof spec, "%s:weight=medium", opt_font);
    face_cap = load_face(spec);
}

static void build_atlas(void) {
    int px0 = (int)fmax(13, H * 0.024), px1 = (int)fmax(11, H * 0.0145);
    if (px0 == cap_px[0] && px1 == cap_px[1]) return;
    cap_px[0] = px0; cap_px[1] = px1;
    unsigned char *a = calloc((size_t)atlas_w * 1024, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < 2; s++) {
        FT_Set_Pixel_Sizes(face_cap, 0, (FT_UInt)cap_px[s]);
        for (int c = 0; c < 96; c++) {
            Glyph *g = &glyphs[s][c];
            memset(g, 0, sizeof *g);
            if (FT_Load_Char(face_cap, (FT_ULong)(32 + c), FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot gs_ = face_cap->glyph;
            int w = (int)gs_->bitmap.width, h = (int)gs_->bitmap.rows;
            if (x + w + 2 > atlas_w) { x = 1; y += shelf + 2; shelf = 0; }
            if (y + h + 2 > 1024) break;
            for (int yy = 0; yy < h; yy++) memcpy(a + (size_t)(y + yy) * atlas_w + x, gs_->bitmap.buffer + yy * gs_->bitmap.pitch, (size_t)w);
            *g = (Glyph){(short)x, (short)y, (short)w, (short)h, (short)gs_->bitmap_left, (short)gs_->bitmap_top, (short)(gs_->advance.x >> 6)};
            x += w + 2;
            if (h > shelf) shelf = h;
        }
    }
    atlas_h = y + shelf + 2;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlas_w, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, a);
    free(a);
    cap_line[0][0] = 0;                  /* rebuild the quads */
}

static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 8192) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}
static void text_at(int s, float x, float y, const char *str, const float *col) {
    float pen = roundf(x);
    y = roundf(y);
    for (const char *p = str; *p; p++) {
        int c = (unsigned char)*p;
        const Glyph *g = &glyphs[s][c >= 32 && c < 128 ? c - 32 : 0];
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            quad_v(x0, y0, g->x, g->y, col); quad_v(x1, y0, g->x + g->w, g->y, col); quad_v(x1, y1, g->x + g->w, g->y + g->h, col);
            quad_v(x0, y0, g->x, g->y, col); quad_v(x1, y1, g->x + g->w, g->y + g->h, col); quad_v(x0, y1, g->x, g->y + g->h, col);
        }
        pen += g->adv;
    }
}

/* returns the number of vertices */
static int caption(double wall) {
    char l0[96], l1[96], dur[64];
    time_t tt = (time_t)wall;
    struct tm tm;
    localtime_r(&tt, &tm);
    strftime(l0, sizeof l0, "%H:%M", &tm);
    long away = (long)(wall - (double)locked_at);
    fmt_span(dur, sizeof dur, away < 0 ? 0 : away);
    snprintf(l1, sizeof l1, "away %s", dur);
    if (!strcmp(l0, cap_line[0]) && !strcmp(l1, cap_line[1])) return tvn / 8;
    strcpy(cap_line[0], l0); strcpy(cap_line[1], l1);
    tvn = 0;
    float mx = (float)fmax(14, H * 0.035), by = (float)(H - fmax(14, H * 0.035));
    float y1 = by, y0 = by - cap_px[1] * 1.55f;
    float halo[4] = {0.10f, 0.07f, 0.05f, 0.16f}, col[4] = {0.97f, 0.93f, 0.86f, 0.92f};
    float r = fmaxf(1.0f, cap_px[0] / 12.0f);
    for (int k = 0; k < 12; k++) {
        float a = k * 0.5236f, dx = r * cosf(a), dy = r * sinf(a);
        for (int q = 1; q <= 2; q++) {
            text_at(0, mx + dx * q, y0 + dy * q, l0, halo);
            text_at(1, mx + dx * q, y1 + dy * q, l1, halo);
        }
    }
    text_at(0, mx, y0, l0, col);
    col[3] = 0.8f;
    text_at(1, mx, y1, l1, col);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_DYNAMIC_DRAW);
    return tvn / 8;
}

/* ---- the worm's body: a tube swept along its centreline ---- */
#define NRING 400
#define NSIDE 40
static float *wv;                      /* 12 floats a vertex */
static int worm_on;
static float caps[24][4], cap_box[4];
static int ncap;

static void worm_mesh_init(void) {
    wv = malloc(sizeof(float) * 12 * NRING * (NSIDE + 1));
    unsigned *ix = malloc(sizeof(unsigned) * 6 * (NRING - 1) * NSIDE), n = 0;
    for (int i = 0; i + 1 < NRING; i++)
        for (int k = 0; k < NSIDE; k++) {
            unsigned a = (unsigned)(i * (NSIDE + 1) + k), b = a + 1, c = a + NSIDE + 1, d = c + 1;
            ix[n++] = a; ix[n++] = c; ix[n++] = b;
            ix[n++] = b; ix[n++] = c; ix[n++] = d;
        }
    glGenVertexArrays(1, &vao_worm);
    glBindVertexArray(vao_worm);
    glGenBuffers(1, &vbo_worm);
    glGenBuffers(1, &ibo_worm);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_worm);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * n, ix, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_worm);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12 * NRING * (NSIDE + 1), NULL, GL_STREAM_DRAW);
    for (int a = 0; a < 4; a++) {
        glVertexAttribPointer((GLuint)a, 3, GL_FLOAT, GL_FALSE, 48, (void *)(intptr_t)(a * 12));
        glEnableVertexAttribArray((GLuint)a);
    }
    free(ix);
}

static void norm3(double v[3]) {
    double l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; }
}

/* build the body for time t; returns whether any of it is out of the sand */
static int worm_build(const Pass *P, double t) {
    static double C[NRING][3], Rr[NRING];
    double hi = -1e9;
    for (int i = 0; i < NRING; i++) {
        double tau = P->Lb * i / (NRING - 1.0);
        /* rings closer together at the head */
        tau = P->Lb * pow((double)i / (NRING - 1), 1.25);
        body_pt(P, t, tau, C[i]);
        Rr[i] = body_r(P, tau);
        hi = fmax(hi, head_y(P, t - tau / P->v) + Rr[i]);
    }
    ncap = 0;
    if (hi < -0.05 * P->R) return 0;
    for (int i = 0; i < NRING; i++) {
        double tau = P->Lb * pow((double)i / (NRING - 1), 1.25);
        int a = i > 0 ? i - 1 : 0, b = i < NRING - 1 ? i + 1 : NRING - 1;
        double T[3] = {C[b][0] - C[a][0], C[b][1] - C[a][1], C[b][2] - C[a][2]};
        norm3(T);
        double N[3] = {-T[1] * T[0], 1 - T[1] * T[1], -T[1] * T[2]};
        norm3(N);
        double B[3] = {T[1] * N[2] - T[2] * N[1], T[2] * N[0] - T[0] * N[2], T[0] * N[1] - T[1] * N[0]};
        double r = Rr[i], dr = (Rr[b] - Rr[a]) / fmax(1e-3, hypot(hypot(C[b][0] - C[a][0], C[b][1] - C[a][1]), C[b][2] - C[a][2]));
        double headness = 1 - sstep(1.5 * P->R, 3.5 * P->R, tau);
        double flat = 1 - 0.22 * headness;     /* a broad, flattened snout */
        /* around the ring: the angle from the back, two low keels along
         * it, a slightly flattened belly */
        static double kc[NSIDE + 1], ks[NSIDE + 1], kk[NSIDE + 1], kb[NSIDE + 1];
        if (kb[0] == 0)
            for (int k = 0; k <= NSIDE; k++) {
                double ph = 2 * M_PI * k / NSIDE - M_PI, ap = fabs(ph);
                kc[k] = cos(ph); ks[k] = sin(ph);
                kk[k] = 0.07 * exp(-pow((ap - 0.55) / 0.13, 2));
                kb[k] = ap > 2.2 ? 1 - 0.06 * sstep(2.2, M_PI, ap) : 1;
            }
        for (int k = 0; k <= NSIDE; k++) {
            double ph = 2 * M_PI * k / NSIDE - M_PI;   /* 0 along the back */
            double rr_ = r * (1 + kk[k] * (1 - 0.5 * headness)) * kb[k];
            double cy = kc[k] * flat, sy = ks[k];
            float *o = wv + 12 * (i * (NSIDE + 1) + k);
            double rad[3], nn[3];
            for (int c = 0; c < 3; c++) rad[c] = N[c] * cy + B[c] * sy;
            for (int c = 0; c < 3; c++) nn[c] = N[c] * kc[k] / flat + B[c] * sy - T[c] * dr;
            norm3(nn);
            o[0] = (float)(C[i][0] + rad[0] * rr_ - camX);
            o[1] = (float)(C[i][1] + rad[1] * rr_ - camY);
            o[2] = (float)(C[i][2] + rad[2] * rr_ - camZ);
            o[3] = (float)nn[0]; o[4] = (float)nn[1]; o[5] = (float)nn[2];
            o[6] = (float)T[0]; o[7] = (float)T[1]; o[8] = (float)T[2];
            o[9] = (float)tau; o[10] = (float)ph; o[11] = (float)headness;
        }
    }
    /* capsules for its shadow: the parts out of the sand */
    int last = -10;
    for (int i = 0; i < NRING && ncap < 24; i += 12) {
        double tau = P->Lb * pow((double)i / (NRING - 1), 1.25);
        int out = head_y(P, t - tau / P->v) + Rr[i] > -2;
        if (!out) continue;
        if (last != i - 12 && ncap > 0 && ncap < 24) { caps[ncap][3] = 0; caps[ncap][0] = caps[ncap][1] = caps[ncap][2] = 0; ncap++; }
        if (ncap >= 24) break;
        caps[ncap][0] = (float)C[i][0]; caps[ncap][1] = (float)C[i][1]; caps[ncap][2] = (float)C[i][2];
        caps[ncap][3] = (float)Rr[i];
        ncap++;
        last = i;
    }
    /* the ground its shadow can reach: each capsule slid down the light
     * to the dune floor */
    double x0 = 1e18, z0 = 1e18, x1 = -1e18, z1 = -1e18;
    const float *L = lt.shadow_sun > 0.5f ? lt.sun : lt.moon;
    for (int i = 0; i < ncap; i++) {
        if (caps[i][3] <= 0) continue;
        double drop = fmax(caps[i][1] - (ref_level(caps[i][0], caps[i][2]) - 10), 0) / fmax(L[1], 0.03);
        for (int k = 0; k < 2; k++) {
            double x = caps[i][0] - L[0] * drop * k, z = caps[i][2] - L[2] * drop * k;
            x0 = fmin(x0, x - 2 * caps[i][3]); x1 = fmax(x1, x + 2 * caps[i][3]);
            z0 = fmin(z0, z - 2 * caps[i][3]); z1 = fmax(z1, z + 2 * caps[i][3]);
        }
    }
    cap_box[0] = (float)x0; cap_box[1] = (float)z0; cap_box[2] = (float)x1; cap_box[3] = (float)z1;
    return 1;
}

/* ---- the worm's height map ---- */
#define DMAP 384
static float segA[96][4], segB[96][4], map_box[4], map_drawn[4];
static int map_valid;
static int nseg, map_on;
static float shocks[2][4];

static void worm_segments(const Pass *P, double t) {
    nseg = 0;
    double te = t - P->e * EP;
    double fade = sstep(0, 25, te) * (1 - sstep(EP - 25, EP, te));
    double x0 = 1e18, z0 = 1e18, x1 = -1e18, z1 = -1e18;
    /* the body, from a little ahead of the snout (the bow wave) */
    int nb = 56;
    for (int k = 0; k < nb; k++) {
        double tau = -1.2 * P->R + (P->Lb + 1.2 * P->R) * k / (nb - 1.0);
        double c[3];
        body_pt(P, t, fmax(tau, 0), c);
        if (tau < 0) { double s = P->v * (t - P->tc) - tau; path_at(P, s, &c[0], &c[2]); }
        double r = tau < 0 ? P->R * 0.7 : body_r(P, tau);
        if (tau < 0) c[1] += 0.25 * P->R;
        segA[nseg][0] = (float)c[0]; segA[nseg][1] = (float)c[2]; segA[nseg][2] = (float)c[1]; segA[nseg][3] = (float)r;
        segB[nseg][0] = 0; segB[nseg][1] = (float)P->R; segB[nseg][2] = 1; segB[nseg][3] = (float)fade;
        x0 = fmin(x0, c[0]); x1 = fmax(x1, c[0]); z0 = fmin(z0, c[2]); z1 = fmax(z1, c[2]);
        nseg++;
    }
    /* the furrow behind the tail */
    int nt = 36;
    for (int k = 0; k < nt; k++) {
        double tau = P->Lb + TRAIL * k / (nt - 1.0), x, z;
        path_at(P, P->v * (t - P->tc) - tau, &x, &z);
        double age = (tau - P->Lb) / P->v;
        /* the trail began where the tail was at the start of the episode */
        if (t - tau / P->v < P->e * EP - 60) break;
        segA[nseg][0] = (float)x; segA[nseg][1] = (float)z; segA[nseg][2] = 0; segA[nseg][3] = 0;
        segB[nseg][0] = (float)age; segB[nseg][1] = (float)P->R; segB[nseg][2] = 0; segB[nseg][3] = (float)fade;
        x0 = fmin(x0, x); x1 = fmax(x1, x); z0 = fmin(z0, z); z1 = fmax(z1, z);
        nseg++;
    }
    double m = 6 * P->R;
    double size = fmax(x1 - x0, z1 - z0) + 2 * m;
    double tx = ceil((P->Lb + TRAIL + 12 * P->R) / DMAP);   /* fixed per pass, so the map doesn't swim */
    if (size > tx * DMAP) size = tx * DMAP;
    double cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    map_box[0] = (float)(floor((cx - tx * DMAP / 2) / tx) * tx);
    map_box[1] = (float)(floor((cz - tx * DMAP / 2) / tx) * tx);
    map_box[2] = (float)(tx * DMAP);
    map_box[3] = (float)tx;
    map_on = fade > 0.001;
    /* shock ripples where it broke the surface and where it went back in */
    memset(shocks, 0, sizeof shocks);
    if (P->breach) {
        double tb = breach_start(P);
        for (int i = 0; i < 2; i++) {
            double tp = tb + i * P->Tb, x, z;
            path_at(P, P->v * (tp - P->tc), &x, &z);
            shocks[i][0] = (float)x; shocks[i][1] = (float)z;
            shocks[i][2] = (float)(t - tp);
            shocks[i][3] = t >= tp && t - tp < 60 ? (float)(i ? 2.2 : 1.6) : 0.0f;
        }
    }
}

/* ---- falling sand and dust ---- */
#define MAXP 9000
static float pinst[MAXP][8];
static int np;

static void add_p(double x, double y, double z, double size, double alpha, int kind, double stretch, double seed) {
    if (np >= MAXP || alpha <= 0.002) return;
    float *o = pinst[np++];
    o[0] = (float)(x - camX); o[1] = (float)(y - camY); o[2] = (float)(z - camZ); o[3] = (float)size;
    o[4] = (float)alpha; o[5] = (float)kind; o[6] = (float)stretch; o[7] = (float)seed;
}

static double ground(double x, double z) { return terrain((float)x, (float)z, 4.0f); }

static void particles(const Pass *P, double t) {
    np = 0;
    double wx = uWx, wz = uWz;
    double te = t - P->e * EP;
    double fade = sstep(0, 25, te) * (1 - sstep(EP - 25, EP, te));
    /* a faint plume over the buried head */
    for (long k = (long)floor((t - 26) / 1.6); k <= (long)floor(t / 1.6); k++) {
        double ts = k * 1.6, a = t - ts;
        if (a < 0 || a > 26 || ts < P->e * EP) continue;
        double hy = head_y(P, ts);
        if (hy > -0.2 * P->R) continue;   /* breaching: the big plumes take over */
        double vis = exp(-(-hy - P->dep) / P->R) * fade;
        Rng r = rng_for((uint64_t)P->e, 900, (uint64_t)k);
        double x, z;
        path_at(P, P->v * (ts - P->tc), &x, &z);
        x += rr(&r, -0.5, 0.5) * P->R + wx * 4 * a; z += rr(&r, -0.5, 0.5) * P->R + wz * 4 * a;
        double y = ground(x, z) + P->R * 0.3 + 1.2 * a;
        add_p(x, y, z, P->R * (0.9 + 0.15 * a), 0.24 * vis * sstep(0, 3, a) * pow(1 - a / 26, 1.5), 1, 0.7, rnd(&r));
    }
    /* sand streaming down the swell's flanks, over the head */
    {
        double hy = head_y(P, t);
        double vis = exp(-fmax(-hy - P->R - 0, 0) / P->R) * fade;
        double rate2 = 90, life2 = 3.5;
        for (long k = (long)ceil((t - life2) * rate2); k <= (long)floor(t * rate2) && vis > 0.05 && hy < -0.3 * P->R; k++) {
            double ts = k / rate2, a = t - ts;
            if (a < 0 || ts < P->e * EP) continue;
            Rng r = rng_for((uint64_t)P->e, 950, (uint64_t)k);
            double tau = rr(&r, -0.5, 4.0) * P->R, x, z, x2, z2;
            double sp = P->v * (ts - P->tc) - tau;
            path_at(P, sp, &x, &z);
            path_at(P, sp + 1, &x2, &z2);
            double side = rnd(&r) < 0.5 ? -1 : 1, off = rr(&r, 0.2, 1.0) * P->R;
            double nx = (z2 - z) * side, nzz = -(x2 - x) * side;
            double spd = rr(&r, 3, 9);
            x += nx * (off + spd * a); z += nzz * (off + spd * a);
            double y = ground(x, z) + 1.5 + 2.5 * a;
            add_p(x, y, z, rr(&r, 2.0, 4.0) * (1 + 0.6 * a), 0.35 * vis * (1 - a / life2) * sstep(0, 0.3, a), 0, 0.8, rnd(&r));
        }
    }
    if (!P->breach) return;
    double tb = breach_start(P), tend = tb + P->Tb + P->Lb / P->v;
    double tang[3];
    /* sand pouring off the body: shed mostly as it comes out */
    double rate = 650, life = 7.5;
    long k0 = (long)ceil((t - life - tb) * rate), k1 = (long)floor((fmin(t, tend) - tb) * rate);
    if (k0 < 0) k0 = 0;
    /* each grain's launch is worked out once and kept, by its number */
    static struct { long e, k; int ok; float p0[3], v0[3], ref, size, puff, seed; } g[8192];
    for (long k = k0; k <= k1; k++) {
        double ts = tb + k / rate, a = t - ts;
        if (a < 0) continue;
        int slot = (int)(k & 8191);
        if (g[slot].k != k || g[slot].e != P->e) {
            Rng r = rng_for((uint64_t)P->e, 501, (uint64_t)k);
            g[slot].k = k; g[slot].e = P->e; g[slot].ok = 0;
            double lo = fmax(tb, ts - P->Lb / P->v), hi = fmin(ts, tb + P->Tb);
            if (hi <= lo) continue;
            double tp = rnd(&r) < 0.6 ? lo + fmin(hi - lo, P->Tb * 0.22) * pow(rnd(&r), 1.5) : lo + (hi - lo) * rnd(&r);
            double c[3], c2[3];
            double ref = path_ref(P, P->v * (tp - P->tc));
            path_at(P, P->v * (tp - P->tc), &c[0], &c[2]);
            c[1] = ref + head_y(P, tp);
            path_at(P, P->v * (tp + 0.2 - P->tc), &c2[0], &c2[2]);
            c2[1] = path_ref(P, P->v * (tp + 0.2 - P->tc)) + head_y(P, tp + 0.2);
            for (int i = 0; i < 3; i++) tang[i] = c2[i] - c[i];
            norm3(tang);
            double rad = body_r(P, P->v * (ts - tp));
            if (c[1] + rad < ref) continue;
            double N[3] = {-tang[1] * tang[0], 1 - tang[1] * tang[1], -tang[1] * tang[2]};
            norm3(N);
            double B[3] = {tang[1] * N[2] - tang[2] * N[1], tang[2] * N[0] - tang[0] * N[2], tang[0] * N[1] - tang[1] * N[0]};
            double ang = (rnd(&r) < 0.5 ? -1 : 1) * rr(&r, 1.3, 2.4);   /* off the flanks and belly */
            for (int i = 0; i < 3; i++) {
                double o = cos(ang) * N[i] + sin(ang) * B[i];
                g[slot].p0[i] = (float)(c[i] + rad * o);
                g[slot].v0[i] = (float)(tang[i] * P->v * rr(&r, 0.15, 0.5) + o * rr(&r, 0.5, 4));
            }
            g[slot].ref = (float)ref;
            g[slot].size = (float)rr(&r, 1.6, 3.4);
            g[slot].puff = (float)rr(&r, 5, 9);
            g[slot].seed = (float)rnd(&r);
            g[slot].ok = 1;
        }
        if (!g[slot].ok) continue;
        const float *p0 = g[slot].p0, *v0 = g[slot].v0;
        double x = p0[0] + (v0[0] + wx * 3) * a, z = p0[2] + (v0[2] + wz * 3) * a;
        double y = p0[1] + v0[1] * a - 4.9 * a * a;
        double gy = g[slot].ref;      /* (the sand where it fell from: near enough) */
        double al = 0.42 * pow(1 - a / life, 1.3) * sstep(0.5, 1.4, a);
        if (y < gy) {
            /* landed: a short-lived puff of its own */
            double vy = v0[1], tl = (vy + sqrt(fmax(vy * vy + 19.6 * (p0[1] - gy), 0))) / 9.8, since = a - tl;
            if (since > 1.5) continue;
            y = gy + 2 + 3 * since;
            al *= (1 - since / 1.5) * 0.6;
            add_p(x, y, z, g[slot].puff * (1 + since), al, 1, 0.6, g[slot].seed);
            continue;
        }
        double vy = v0[1] - 9.8 * a;
        add_p(x, y, z, g[slot].size * (1 + 0.25 * a), al, 0, 1 + fmin(fabs(vy) * 0.12, 4.0), g[slot].seed);
    }
    /* dust where it broke the sand and where it went back in */
    for (int side = 0; side < 2; side++) {
        double tp0 = tb + side * P->Tb, px, pz;
        path_at(P, P->v * (tp0 - P->tc), &px, &pz);
        double pg = ground(px, pz);
        double dt = 0.75, plife = 65;
        long q0 = (long)ceil((t - plife - tp0) / dt), q1 = (long)floor((fmin(t, tp0 + P->Lb / P->v + 4) - tp0) / dt);
        if (q0 < 0) q0 = 0;
        for (long q = q0; q <= q1; q++) {
            double ts = tp0 + q * dt, a = t - ts;
            if (a < 0) continue;
            Rng r = rng_for((uint64_t)P->e, 700 + side, (uint64_t)q);
            double sp = 1 - exp(-a / 18);
            double x = px + rr(&r, -1, 1) * P->R * (1.3 + 2 * sp) + wx * 5.5 * a;
            double z = pz + rr(&r, -1, 1) * P->R * (1.3 + 2 * sp) + wz * 5.5 * a;
            double y = pg + P->R * rr(&r, 0.3, 1.4) + 45 * sp * rr(&r, 0.3, 1.0);
            double size = P->R * (1.3 + 2.6 * sp) * rr(&r, 0.8, 1.2);
            add_p(x, y, z, size, 0.2 * sstep(0, 2.5, a) * pow(1 - a / plife, 1.6), 1, 1.0, rnd(&r));
        }
    }
    /* a ring of dust thrown up by the shock where it dived */
    double ts = tb + P->Tb, a = t - ts;
    if (a > 0 && a < 20) {
        double px, pz;
        path_at(P, P->v * (ts - P->tc), &px, &pz);
        for (int k = 0; k < 40; k++) {
            Rng r = rng_for((uint64_t)P->e, 800, (uint64_t)k);
            double an = 2 * M_PI * (k + rr(&r, -0.3, 0.3)) / 40, rad = 55 * a * rr(&r, 0.9, 1.05);
            double x = px + rad * cos(an) + wx * 4 * a, z = pz + rad * sin(an) + wz * 4 * a;
            add_p(x, ground(x, z) + 6 + a, z, 10 + 1.5 * a, 0.22 * sstep(0, 1, a) * (1 - a / 20) / (1 + rad / 600), 1, 0.5, rnd(&r));
        }
    }
}

/* far to near, for the blending */
typedef struct { float d; int i; } PKey;
static int part_cmp(const void *a, const void *b) {
    float da = ((const PKey *)a)->d, db = ((const PKey *)b)->d;
    return da < db ? 1 : da > db ? -1 : 0;
}
static void part_sort(void) {
    static PKey key[MAXP];
    static float tmp[MAXP][8];
    for (int i = 0; i < np; i++) {
        const float *p = pinst[i];
        key[i].d = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        key[i].i = i;
    }
    qsort(key, (size_t)np, sizeof key[0], part_cmp);
    for (int i = 0; i < np; i++) memcpy(tmp[i], pinst[key[i].i], sizeof tmp[0]);
    memcpy(pinst, tmp, sizeof tmp[0] * (size_t)np);
}

static void gl_init(void) {
    const char *vs_terr[] = {GL_COMMON, GL_TERRAIN, GL_GRID, VS_TERR};
    const char *fs_gbuf[] = {FS_GBUF};
    prog_terr = mk_prog(vs_terr, 4, fs_gbuf, 1);
    const char *vs_full[] = {VS_FULL};
    const char *fs_shade[] = {GL_COMMON, GL_TERRAIN, GL_GRID, GL_RAY, FS_TERR, FS_SKY, FS_SHADE};
    prog_shade = mk_prog(vs_full, 1, fs_shade, 7);
    const char *fs_cache[] = {GL_COMMON, GL_TERRAIN, GL_GRID, GL_RAY, FS_TERR, FS_CACHE};
    prog_cache = mk_prog(vs_full, 1, fs_cache, 6);
    const char *fs_hgt[] = {GL_COMMON, GL_TERRAIN, GL_GRID, FS_HGT};
    prog_hgt = mk_prog(vs_full, 1, fs_hgt, 4);
    const char *vs_worm[] = {VS_WORM};
    const char *fs_worm[] = {GL_COMMON, GL_TERRAIN, FS_WORM};
    prog_worm = mk_prog(vs_worm, 1, fs_worm, 3);
    const char *vs_part[] = {GL_COMMON, VS_PART};
    const char *fs_part[] = {GL_COMMON, FS_PART};
    prog_part = mk_prog(vs_part, 2, fs_part, 2);
    const char *fs_dist[] = {GL_COMMON, GL_TERRAIN, FS_DIST};
    prog_dist = mk_prog(vs_full, 1, fs_dist, 3);
    const char *fs_sh[] = {GL_COMMON, GL_TERRAIN, GL_GRID, FS_SHADOW};
    prog_shadow = mk_prog(vs_full, 1, fs_sh, 4);
    const char *fs_post[] = {GL_COMMON, GL_RAY, FS_POST};
    prog_post = mk_prog(vs_full, 1, fs_post, 3);
    const char *vs_text[] = {VS_TEXT}, *fs_text[] = {FS_TEXT};
    prog_text = mk_prog(vs_text, 1, fs_text, 1);

    glGenVertexArrays(1, &vao_empty);
    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void *)8);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glGenVertexArrays(1, &vao_part);
    glBindVertexArray(vao_part);
    glGenBuffers(1, &vbo_part);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_part);
    glBufferData(GL_ARRAY_BUFFER, sizeof pinst, NULL, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(0, 1);
    glVertexAttribDivisor(1, 1);
    worm_mesh_init();

    glGenTextures(1, &tex_nz);
    glBindTexture(GL_TEXTURE_2D, tex_nz);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 256, 256, 0, GL_RED, GL_FLOAT, nzt);
    tex_params(GL_REPEAT, GL_NEAREST);
    glGenTextures(1, &tex_shadow);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    tex_params(GL_CLAMP_TO_EDGE, GL_LINEAR);
    glGenFramebuffers(1, &fbo_shadow);
    glGenTextures(1, &tex_dist);
    glBindTexture(GL_TEXTURE_2D, tex_dist);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, DMAP, DMAP, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_LINEAR);
    glGenFramebuffers(1, &fbo_dist);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dist);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_dist, 0);
    glGenTextures(1, &tex_scene);
    glGenTextures(1, &tex_g);
    glGenTextures(1, &tex_hgt);
    glBindTexture(GL_TEXTURE_2D, tex_hgt);
    tex_params(GL_CLAMP_TO_EDGE, GL_NEAREST);
    glGenFramebuffers(1, &fbo_hgt);
    glGenTextures(1, &tex_ca);
    glGenTextures(1, &tex_cb);
    glGenFramebuffers(1, &fbo_cache);
    glGenFramebuffers(1, &fbo_g);
    glGenFramebuffers(1, &fbo_scene);
    glGenRenderbuffers(1, &rb_dep);
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

/* everything the common GLSL wants */
static double cur_t;
static void set_common(GLuint p) {
    glUseProgram(p);
    glUniform1i(U(p, "uNz"), 0);
    glUniform1f(U(p, "uWx"), uWx); glUniform1f(U(p, "uWz"), uWz);
    glUniform1f(U(p, "uL1"), uL1); glUniform1f(U(p, "uA1"), uA1);
    glUniform1f(U(p, "uL2"), uL2); glUniform1f(U(p, "uA2"), uA2);
    glUniform1fv(U(p, "uMesa"), 8, uMesa);
    glUniform3f(U(p, "uCam"), (float)camX, (float)camY, (float)camZ);
    glUniform1f(U(p, "uT"), (float)fmod(cur_t, 3600.0));
    glUniform1f(U(p, "uPix"), (float)(cv_w / W / cv_f));
    glUniform3fv(U(p, "uSunDir"), 1, lt.sun); glUniform3fv(U(p, "uSunCol"), 1, lt.sunc);
    glUniform3fv(U(p, "uMoonDir"), 1, lt.moon); glUniform3fv(U(p, "uMoonCol"), 1, lt.moonc);
    glUniform3fv(U(p, "uZen"), 1, lt.zen); glUniform3fv(U(p, "uHor"), 1, lt.hor);
    glUniform3fv(U(p, "uHaze"), 1, lt.haze); glUniform3fv(U(p, "uGlow"), 1, lt.glow);
    glUniform3fv(U(p, "uMoonGlow"), 1, lt.moonglow); glUniform3fv(U(p, "uAmb"), 1, lt.amb);
    glUniform1f(U(p, "uFogK"), lt.fogk); glUniform1f(U(p, "uNight"), lt.night);
    glUniform1f(U(p, "uLowSun"), (float)(1 - sstep(4, 22, lt.sun_el)) * (float)sstep(-12, -2, lt.sun_el));
    /* the grid */
    glUniform1f(U(p, "uPsi"), (float)cam_psi);
    glUniform1f(U(p, "uColK"), (float)(COLPX / cv_f));
    glUniform1f(U(p, "uI0"), (float)g_i0);
    glUniform1f(U(p, "uJ0"), (float)g_J0);
    glUniform1f(U(p, "uDmin"), (float)DMIN);
    glUniform1f(U(p, "uS0"), (float)S0);
    glUniform1f(U(p, "uD1"), (float)g_d1);
    glUniform1f(U(p, "uKap"), (float)KAP);
    glUniform1i(U(p, "uDist"), 2);
    glUniform4f(U(p, "uMapBox"), map_drawn[0], map_drawn[1], map_drawn[2], map_drawn[3]);
    glUniform1f(U(p, "uMapOn"), (float)(map_on && map_valid));
    glUniform4fv(U(p, "uShock"), 2, &shocks[0][0]);
    glUniform3fv(U(p, "uSand"), 1, P_sand);
    /* the view ray */
    glUniform4f(U(p, "uCanvas"), (float)cv_x, (float)cv_y, (float)(cv_w / W), (float)(cv_h / H));
    glUniform3f(U(p, "uCanvasC"), (float)cv_cx, (float)cv_cy, (float)cv_f);
    glUniform3f(U(p, "uF"), (float)cF[0], (float)cF[1], (float)cF[2]);
    glUniform3f(U(p, "uR"), (float)cR[0], (float)cR[1], (float)cR[2]);
    glUniform3f(U(p, "uU"), (float)cU[0], (float)cU[1], (float)cU[2]);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
}

/* this view's projection: an off-axis slice of the one camera */
static void view_proj(float m[16]) {
    double n = 2, fz = 60000;
    double ax = 2 * (cv_cx - cv_x) / cv_w - 1, ay = 1 - 2 * (cv_cy - cv_y) / cv_h;
    double sx = 2 * cv_f / cv_w, sy = 2 * cv_f / cv_h;
    double A = (fz + n) / (fz - n), B = -2 * fz * n / (fz - n);
    /* row-major: clip = M * (p, 1) */
    for (int c = 0; c < 3; c++) {
        m[0 + c] = (float)(sx * cR[c] + ax * cF[c]);
        m[4 + c] = (float)(sy * cU[c] + ay * cF[c]);
        m[8 + c] = (float)(A * cF[c]);
        m[12 + c] = (float)cF[c];
    }
    m[3] = 0; m[7] = 0; m[11] = (float)B; m[15] = 0;
}

static Pass cur_pass;

/* --bench: GPU time per pass */
#define NPROF 8
static const char *prof_name[NPROF] = {"map", "shadows", "grid", "shade", "worm", "particles", "post", "caption"};
static GLuint prof_q[NPROF + 1];
static double prof_ms[NPROF], prof_min[NPROF] = {1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9};
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[NPROF + 1];
    for (int i = 0; i <= NPROF; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < NPROF; i++) {
        prof_ms[i] += (t[i + 1] - t[i]) / 1e6;
        prof_min[i] = fmin(prof_min[i], (t[i + 1] - t[i]) / 1e6);
    }
}

/* the CPU's share of a frame: where everything is */
static void update(double t, double wall) {
    if (t < 0) t = 0;
    cur_t = t;
    canvas_update();
    cam_aim();
    grid_update();
    targets_update();
    light_update(wall);
    long e = (long)floor(t / EP);
    pass_params(e, &cur_pass);
    worm_segments(&cur_pass, t);
    worm_on = worm_build(&cur_pass, t);
    particles(&cur_pass, t);
    if (np > 0) part_sort();
}

/* and the GPU's */
static void draw(double wall, GLuint target) {
    float VP[16];
    view_proj(VP);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_nz);
    glBindVertexArray(vao_empty);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    prof_mark(0);
    /* the worm's height map */
    /* (every other frame is plenty: it moves a metre) */
    if (!map_on) map_valid = 0;
    else if (!map_valid || !(frame_no & 1)) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_dist);
        glViewport(0, 0, DMAP, DMAP);
        set_common(prog_dist);
        glUniform4fv(U(prog_dist, "uSeg"), nseg, &segA[0][0]);
        glUniform4fv(U(prog_dist, "uSegB"), nseg, &segB[0][0]);
        glUniform1i(U(prog_dist, "uNSeg"), nseg);
        glUniform4f(U(prog_dist, "uBox"), map_box[0], map_box[1], map_box[3], 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        memcpy(map_drawn, map_box, sizeof map_box);
        map_valid = 1;
    }

    prof_mark(1);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_hgt);
    glActiveTexture(GL_TEXTURE0);
    /* the grid's heights, when it is new */
    if (hgt_dirty) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_hgt);
        glViewport(0, 0, sh_w, sh_h);
        set_common(prog_hgt);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        hgt_dirty = 0;
    }
    /* cast shadows: a slice of the grid a frame, all of it when it's new */
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
        glViewport(0, 0, sh_w, sh_h);
        set_common(prog_shadow);
        glUniform1i(U(prog_shadow, "uHgt"), 5);
        glUniform3fv(U(prog_shadow, "uLightDir"), 1, lt.shadow_sun > 0.5f ? lt.sun : lt.moon);
        int slices = 48;
        if (!sh_full) {
            int k = (int)(frame_no % slices);
            glEnable(GL_SCISSOR_TEST);
            glScissor(0, sh_h * k / slices, sh_w, sh_h * (k + 1) / slices - sh_h * k / slices);
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_SCISSOR_TEST);
        sh_full = 0;
    }
    frame_no++;

    prof_mark(2);
    /* the static look of the sand in every pixel, once: from the grid
     * without the worm's work on it */
    if (cache_dirty) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_g);
        glViewport(0, 0, W, H);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        set_common(prog_terr);
        glUniformMatrix4fv(U(prog_terr, "uVP"), 1, GL_TRUE, VP);
        glUniform1i(U(prog_terr, "uHgt"), 5);
        glUniform1f(U(prog_terr, "uMapOn"), 0.0f);
        static const float noshock[8];
        glUniform4fv(U(prog_terr, "uShock"), 2, noshock);
        glBindVertexArray(vao_empty);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 2 * g_ncol, g_nring - 1);
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_cache);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, tex_g);
        glActiveTexture(GL_TEXTURE0);
        set_common(prog_cache);
        glUniform1i(U(prog_cache, "uGbuf"), 4);
        glUniform3fv(U(prog_cache, "uSand2"), 1, P_sand2);
        glUniform3fv(U(prog_cache, "uRock"), 1, P_rock);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        cache_dirty = 0;
    }
    /* the grid: how far away the sand is, in every pixel */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_g);
    glViewport(0, 0, W, H);
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_dist);
    glActiveTexture(GL_TEXTURE0);
    set_common(prog_terr);
    glUniformMatrix4fv(U(prog_terr, "uVP"), 1, GL_TRUE, VP);
    glUniform1i(U(prog_terr, "uHgt"), 5);
    glBindVertexArray(vao_empty);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 2 * g_ncol, g_nring - 1);

    prof_mark(3);
    /* then every pixel shaded once: the sand, or the sky */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_g);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex_ca);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, tex_cb);
    glActiveTexture(GL_TEXTURE0);
    set_common(prog_shade);
    glUniform1i(U(prog_shade, "uGbuf"), 4);
    glUniform1i(U(prog_shade, "uCacheA"), 6);
    glUniform1i(U(prog_shade, "uCacheB"), 7);
    glUniform1i(U(prog_shade, "uShadow"), 1);
    glUniform2f(U(prog_shade, "uShSize"), (float)sh_w, (float)sh_h);
    glUniform3fv(U(prog_shade, "uSand2"), 1, P_sand2);
    glUniform3fv(U(prog_shade, "uRock"), 1, P_rock);
    glUniform1f(U(prog_shade, "uWindy"), (float)wind_strength);
    glUniform1f(U(prog_shade, "uShadowSun"), lt.shadow_sun);
    glUniform1f(U(prog_shade, "uMapTex"), map_drawn[3]);
    glUniform4fv(U(prog_shade, "uCap"), 24, &caps[0][0]);
    glUniform1i(U(prog_shade, "uNCap"), worm_on ? ncap : 0);
    glUniform4fv(U(prog_shade, "uCapBox"), 1, cap_box);
    glUniform3fv(U(prog_shade, "uCloudCol"), 1, lt.cloud);
    glUniform1f(U(prog_shade, "uCloud"), cloud_cover);
    glUniform1f(U(prog_shade, "uStarRot"), (float)fmod(wall / 86164.1 * 2 * M_PI, 2 * M_PI));
    glUniform1f(U(prog_shade, "uMoonI"), lt.moon_i);
    glUniform1f(U(prog_shade, "uMoonSide"), lt.moon_side);
    glUniform1f(U(prog_shade, "uMoonR"), 0.0062f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    prof_mark(4);
    /* the worm, against the grid's depth */
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    if (worm_on) {
        glDepthMask(GL_TRUE);
        glEnable(GL_CULL_FACE);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_worm);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 12 * NRING * (NSIDE + 1), wv);
        set_common(prog_worm);
        glUniformMatrix4fv(U(prog_worm, "uVP"), 1, GL_TRUE, VP);
        glUniform3fv(U(prog_worm, "uSkinA"), 1, P_skin[0]);
        glUniform3fv(U(prog_worm, "uSkinB"), 1, P_skin[1]);
        glUniform3fv(U(prog_worm, "uSkinC"), 1, P_skin[2]);
        glUniform1f(U(prog_worm, "uSegL"), (float)(cur_pass.R * 0.28));
        glUniform1f(U(prog_worm, "uDusty"), 0.55f);
        glUniform1f(U(prog_worm, "uShadowSun"), lt.shadow_sun);
        glBindVertexArray(vao_worm);
        glDrawElements(GL_TRIANGLES, 6 * (NRING - 1) * NSIDE, GL_UNSIGNED_INT, 0);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_FALSE);
    }

    prof_mark(5);
    /* sand and dust */
    if (np > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        set_common(prog_part);
        glUniformMatrix4fv(U(prog_part, "uVP"), 1, GL_TRUE, VP);
        glUniform3f(U(prog_part, "uRw"), (float)cR[0], (float)cR[1], (float)cR[2]);
        glUniform3f(U(prog_part, "uUw"), 0.0f, 1.0f, 0.0f);
        glBindVertexArray(vao_part);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_part);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(sizeof pinst[0] * np), pinst);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, np);
        glDisable(GL_BLEND);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);

    prof_mark(6);
    /* the last pass */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    set_common(prog_post);
    glUniform1i(U(prog_post, "uScene"), 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE0);
    glUniform1f(U(prog_post, "uExpo"), lt.expo);
    glUniform1f(U(prog_post, "uShim"), lt.shim);
    glUniform1f(U(prog_post, "uHorY"), (float)horizon_y);
    glUniform1f(U(prog_post, "uFrH"), (float)fr[3]);
    glUniform1f(U(prog_post, "uDev"), (float)(W / cv_w));
    glUniform4f(U(prog_post, "uVig"), (float)((lay_x0 + lay_x1) / 2), (float)((lay_y0 + lay_y1) / 2),
                (float)((lay_x1 - lay_x0) / 2), (float)((lay_y1 - lay_y0) / 2));
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    prof_mark(7);
    if (opt_caption && (span_me < 0 || span_me == span_front)) {
        build_atlas();
        int nv = caption(wall);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_text);
        glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(U(prog_text, "uAtlas"), 0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glDrawArrays(GL_TRIANGLES, 0, nv);
    }
    glDisable(GL_BLEND);
    prof_mark(8);
}

static void render(double t, double wall, GLuint target) {
    update(t, wall);
    draw(wall, target);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "sandworm");
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
        "usage: sandworm [--window | --root | --layer] [--seed N] [--fps N]\n"
        "                [--palette ochre|rust|pale|dusk|night] [--time real|dawn|day|dusk|night|HH:MM]\n"
        "                [--breach] [--no-caption] [--font FAMILY] [--schedule]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --palette   the sand; default: ochre, rust or pale from the seed (dusk and\n"
        "              night also hold that time of day)\n"
        "  --time      light it for this time of day instead of the clock's\n"
        "  --breach    every pass breaches (normally every third)\n"
        "  --fps       frame rate (default 30)\n"
        "  --schedule  print what happens when (seconds after the epoch) and exit\n");
    exit(1);
}

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

static void print_schedule(void) {
    for (long e = 0; e < 9; e++) {
        Pass P;
        pass_params(e, &P);
        double D = hypot(P.cx - camX, P.cz - camZ);
        printf("%7.1f s  pass %ld: %s %.0f m off, %.0f m long, head crosses at %.1f s\n", e * EP, e,
               P.breach ? "BREACH," : "under the sand,", D, P.Lb, P.tc);
        if (P.breach) {
            double tb = breach_start(&P);
            printf("           breaks the surface %.1f s, apex %.1f s (%.0f m up), dives %.1f s, tail under %.1f s\n",
                   tb, P.tc, P.Hpk, tb + P.Tb, tb + P.Tb + P.Lb / P.v);
        }
    }
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-time") && i + 1 < argc) opt_time = argv[++i];
        else if (!strcmp(a, "-breach")) opt_breach = 1;
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-schedule")) opt_schedule = 1;
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
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    nz_init();
    int offscreen = opt_shot || opt_bench > 0;
    if (offscreen) { W = off_w; H = off_h; }
    canvas_update();
    world_init();
    if (opt_schedule) { print_schedule(); return 0; }
    pick_palette();
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
        XStoreName(dpy, win, "Sandworm");
        XClassHint chint = {"sandworm", "Sandworm"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    gl_init();

    /* --epoch: when the desert started, shared by every instance; a --shot
     * or --bench then shows the moment --at seconds after it */
    if (!opt_epoch) opt_epoch = wallclock() - (offscreen ? opt_at : 0);
    locked_at = lock_start();
    if (offscreen) locked_at = (time_t)opt_epoch;

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        double t0 = now();
        render(opt_at, opt_epoch + opt_at, fbo);
        glFinish();
        double first = (now() - t0) * 1000;
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "sandworm: %s, %dx%d, sun %.1f deg, pass %ld%s, first frame %.0f ms\n", palettes[pal].name, W, H,
                    lt.sun_el, cur_pass.e, worm_on ? " (out of the sand)" : "", first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        glGenQueries(NPROF + 1, prof_q);
        prof_on = 1;
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9, csum = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 30.0;
            double c0 = now();
            update(t, opt_epoch + t);
            csum += now() - c0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            draw(opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            prof_collect();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6);
            if (ms > worst) worst = ms;
        }
        printf("sandworm: %s, %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("sandworm:   GPU %.3f ms (best %.3f), CPU %.3f ms; grid %dx%d, %d particles, worm %s\n", gsum / opt_bench, gmin,
               csum * 1000 / opt_bench, g_ncol, g_nring, np, worm_on ? "out" : "under");
        printf("sandworm:  ");
        for (int i = 0; i < NPROF; i++) printf(" %s %.2f", prof_name[i], prof_ms[i] / opt_bench);
        printf(" ms\nsandworm:   best:");
        for (int i = 0; i < NPROF; i++) printf(" %s %.2f", prof_name[i], prof_min[i]);
        printf(" ms\n");
        printf("sandworm: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
        if (opt_shot) { glBindFramebuffer(GL_FRAMEBUFFER, fbo); save_ppm(opt_shot); }
        return 0;
    }

    for (;;) {
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
        double f0 = now(), wt = wallclock();
        render(wt - opt_epoch, wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - f0);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
