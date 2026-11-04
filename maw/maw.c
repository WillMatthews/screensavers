/*
 * maw — a deep desert as a living film painting, and something vast that
 * lives under it. Layered dune ridges recede through amber haze toward far
 * mesas under a low sun veiled in dust; fine amber glitter hangs in the air
 * and veins the sand. Long, hot, silent lulls; then the ground shudders and
 * a colossal worm rises out of the desert: a plated, cracked, scarred column
 * with a round maw ringed with thousands of needle teeth, glowing faintly
 * from inside, sand pouring off it in sheets. Then it crashes back down.
 *
 * The scene is a sequence of vignettes ("acts", 68-76 s each), shuffled
 * from the seed, each staged as anticipation, climax, aftermath:
 *   harvest  a crawler scooping an amber patch, its plume drifting; a wake
 *            races in across the sand, the ground sinks into a pit and the
 *            maw comes up under the crawler and takes it
 *   refuge   a file of walkers crossing the open sand; a wake behind them;
 *            they run for a rock outcrop and make it, as the worm rears up
 *            over the rock and hangs its maw above them, then withdraws
 *   rise     the signature: the ground swells close in front, the worm
 *            erupts and towers over the frame, tips its maw down at you,
 *            then falls back in an explosion of sand and a rolling gust
 *   horizon  a distant worm arcs out of the dunes, silhouetted against a
 *            huge low sun, watched by tiny figures on the near ridge
 *   storm    a wall of dust rolls in from the horizon and swallows the
 *            light; lightning inside it shows a worm rearing in the murk
 *   night    two moons, cold stars, figures walking out of rhythm along a
 *            ridge; a far-off rumble, and a moonlit breach in the distance
 *   bloom    the ground swells and bursts in a geyser of glowing amber dust
 *            that hangs and glitters and slowly settles; later, drawn by
 *            it, a worm surfaces in the distance
 *
 * Every act has a worm moment, so one comes at least every ~80 s; the big
 * ones (harvest, rise) open a shuffled cycle and refuge or horizon close it.
 *
 * Rendering is 2D and graphic, with a camera model for scale: a camera 35 m
 * up, a focal length of one screen height, and every layer and object at a
 * real distance, so size, height and haze all follow from it. One
 * full-screen pass draws the sky and the eight dune layers (their crest
 * lines computed on the CPU into a small float texture each frame, the
 * camera drifting slowly for parallax) and writes depth; the worm is a
 * triangle strip along its spine shaded in its own (along, across) frame,
 * the maw an analytic ellipse with eleven rings of teeth; the crawler,
 * walkers, flyer, rock, dust, sand curtains and grains are depth-sorted
 * quads with analytic, anti-aliased shapes; a last pass adds the storm,
 * the gust, the glitter, the vignette and the fades between acts.
 *
 * Palettes (--palette, else from the seed): ember, ash, blood, violet.
 *
 * Deterministic: everything is a function of --seed, the layout and the time
 * since --epoch. --schedule prints the acts; --event NAME plays only that
 * vignette and makes a --shot land on its climax (unless --at is given).
 * Spanning: one instance per monitor with the same --seed, --epoch and
 * --layout, and --output = its index; the world lives in wall coordinates,
 * so it is one continuous panorama. The "front" output (under the middle of
 * the layout) sets the scale, holds the horizon and the action, and carries
 * the caption.
 *
 * Same plumbing as burrow: Xlib + GLX for --window / --root, a native
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
static int opt_root, opt_layer, opt_caption = 1, opt_schedule, opt_at_set;
static double opt_fps = 30;
static const char *opt_font = "Inter";
static const char *opt_palette, *opt_event;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the first act starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "maw: %s\n", what); exit(1); }

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
static double sq(double x) { return x * x; }
static double sstep(double a, double b, double x) { double u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); }
static double ease_out(double x) { x = clamp01(x); return 1 - (1 - x) * (1 - x) * (1 - x); }
static double ease_in(double x) { x = clamp01(x); return x * x * x; }
static double mixd(double a, double b, double f) { return a + (b - a) * f; }

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

static double lat(int x, int y, uint32_t salt) {
    uint64_t h = mix64(((uint64_t)(uint32_t)x << 32 | (uint32_t)y) ^ ((uint64_t)salt << 17) ^ 0x5eedull);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}
static double hsh(long i, uint32_t salt) {
    uint64_t h = mix64((uint64_t)i * 0x9E3779B97F4A7C15ull ^ ((uint64_t)salt << 40) ^ mix64(opt_seed + 0x51ull));
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}
static double vn1(double x, uint32_t salt) {
    double f = floor(x), u = x - f;
    long i = (long)f;
    u = u * u * (3 - 2 * u);
    double a = hsh(i, salt), b = hsh(i + 1, salt);
    return a + (b - a) * u;
}
static double fbm1(double x, uint32_t salt) {
    return 0.5 * vn1(x, salt) + 0.3 * vn1(x * 2.03 + 3.1, salt + 1) + 0.2 * vn1(x * 4.1 + 7.7, salt + 2);
}

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

/* ---- the wall ----
 * World units on screen are logical layout pixels (y down). The domain is
 * the layout's bounding box (or, standalone, the screen); the front rect is
 * the output under the middle of the layout, and its shorter side L sets
 * the focal length. */
static double bx0, by0, bx1, by1, bw, bh;
static double fr[4];                   /* front output: x, y, w, h */
static double L;

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

static void domain_init(int dw, int dh) {
    if (span_me >= 0) {
        bx0 = by0 = 1e300; bx1 = by1 = -1e300;
        for (int i = 0; i < span_n; i++) {
            bx0 = fmin(bx0, span[i][0]); bx1 = fmax(bx1, span[i][0] + span[i][2]);
            by0 = fmin(by0, span[i][1]); by1 = fmax(by1, span[i][1] + span[i][3]);
        }
        for (int k = 0; k < 4; k++) fr[k] = span[span_front][k];
    } else {
        bx0 = by0 = 0; bx1 = dw; by1 = dh;
        fr[0] = fr[1] = 0; fr[2] = dw; fr[3] = dh;
    }
    bw = bx1 - bx0; bh = by1 - by0;
    L = fmin(fr[2], fr[3]);
}

/* ==== the world: a camera CAM metres up, focal length F (wall px) ====
 * A point X metres across (in the camera's frame of the act), Y up, z away
 * is at wall px (CX + F (X - camX) / z, YH + F (CAM - Y) / z). */
#define NL 8
#define ZFAR 20000.0
#define CAM 35.0
static double CX, YH, F;
static double camX, shx, shy;          /* this frame: the camera's drift (m); the shake (wall px) */
static const double LZ[NL] = {14000, 6000, 3000, 1500, 800, 420, 210, 95};
static const double LA[NL] = {0, 160, 120, 80, 46, 22, 15, 20};          /* dune amplitude, m */
static const double LLAM[NL] = {0, 2300, 1400, 800, 460, 270, 175, 120}; /* dune wavelength, m */

static double sx_of(double X, double z) { return CX + F * (X - camX) / z + shx; }
static double sy_of(double Y, double z) { return YH + F * (CAM - Y) / z + shy; }
static double X_at(double sx, double z, double cam) { return (sx - CX) * z / F + cam; }

/* ==== acts ==== */
enum { E_HARVEST, E_REFUGE, E_RISE, E_HORIZON, E_STORM, E_NIGHT, E_BLOOM, NEV };
static const char *ev_name[NEV] = {"harvest", "refuge", "rise", "horizon", "storm", "night", "bloom"};
static const double ev_dur[NEV] = {74, 74, 72, 68, 76, 70, 72};
static const double ev_omen[NEV] = {18, 14, 16, 12, 4, 20, 12};       /* the anticipation starts */
static const double ev_climax[NEV] = {39.5, 40, 37, 32, 49, 36, 30};
static const char *ev_what[NEV] = {
    "a wake runs at a crawler on an amber patch; the maw takes it from below",
    "walkers run for a rock; the worm rears over them",
    "the worm erupts close in, tips its maw at you, crashes back",
    "a distant worm arcs across a huge low sun",
    "a dust wall swallows the light; lightning shows a worm",
    "two moons, walkers on a ridge; a far breach",
    "the ground bursts in a geyser of glowing amber dust; a worm comes to it",
};
static int forced_ev = -1;

/* light presets */
enum { LT_GOLD, LT_BLAZE, LT_DUSK, LT_SET, LT_STORM, LT_NIGHT, NLT };
static const int ev_light[NEV] = {LT_GOLD, LT_BLAZE, LT_DUSK, LT_SET, LT_STORM, LT_NIGHT, LT_DUSK};

/* a cycle of all seven, shuffled: a big one first (harvest or rise), refuge
 * or horizon last, and the quieter three (storm, night, bloom) never side
 * by side, so the seams never repeat and the worm always comes soon */
static int quiet(int k) { return k == E_STORM || k == E_NIGHT || k == E_BLOOM; }
static void deck(long c, int *ord) {
    for (int tries = 0;; tries++) {
        Rng r = rng_for((uint64_t)c, (uint64_t)tries, 77);
        for (int i = 0; i < NEV; i++) ord[i] = i;
        for (int i = NEV - 1; i > 0; i--) {
            int j = (int)(rnd(&r) * (i + 1));
            int t = ord[i]; ord[i] = ord[j]; ord[j] = t;
        }
        int ok = (ord[0] == E_HARVEST || ord[0] == E_RISE) && (ord[NEV - 1] == E_REFUGE || ord[NEV - 1] == E_HORIZON);
        for (int i = 0; ok && i + 1 < NEV; i++) if (quiet(ord[i]) && quiet(ord[i + 1])) ok = 0;
        if (ok || tries > 500) return;
    }
}
static int act_kind(long n) {
    if (forced_ev >= 0) return forced_ev;
    static long dc = -1;
    static int ord[NEV];
    long c = n / NEV;
    if (c != dc) { deck(c, ord); dc = c; }
    return ord[n % NEV];
}

#define MAXACT 4096
static double actT0[MAXACT + 1];
static int act_known;
static double act_start(long n) {
    while (act_known < n && act_known < MAXACT) { actT0[act_known + 1] = actT0[act_known] + ev_dur[act_kind(act_known)]; act_known++; }
    return actT0[n];
}
static long act_index(double *t) {
    if (act_start(MAXACT) <= *t) *t = fmod(*t, actT0[MAXACT]);
    long lo = 0, hi = MAXACT;
    while (hi - lo > 1) { long m = (lo + hi) / 2; if (actT0[m] <= *t) lo = m; else hi = m; }
    return lo;
}

typedef struct {
    long n;
    int kind, dir, fgside;
    uint32_t ts;                       /* terrain */
    double T0, dur;
    double fx, fz;                     /* the action: screen x (wall px), distance */
    double cam0, camv;                 /* the camera's drift: start (m) and speed (m/s) */
    double sunx, suny, sunr;           /* wall px, wall px, wall px */
    double fgamp;
    double r[8];
} Act;

static void act_build(Act *A, long n) {
    Rng r = rng_for((uint64_t)n, 13, 3);
    memset(A, 0, sizeof *A);
    A->n = n;
    A->kind = act_kind(n);
    A->T0 = act_start(n);
    A->dur = ev_dur[A->kind];
    A->dir = rnd(&r) < 0.5 ? 1 : -1;
    A->ts = (uint32_t)(rnd(&r) * 1e6);
    A->cam0 = rr(&r, -20000, 20000);
    A->camv = rr(&r, 0.35, 0.8) * (rnd(&r) < 0.5 ? -1 : 1);
    A->fgside = -A->dir;
    A->fgamp = rr(&r, 1.2, 2.6);
    for (int i = 0; i < 8; i++) A->r[i] = rnd(&r);
    double fw = fr[2];
    A->fx = CX + fw * A->dir * rr(&r, 0.0, 0.12);
    A->fz = 400;
    A->sunx = CX - A->dir * fw * rr(&r, 0.10, 0.30);
    A->suny = YH - L * rr(&r, 0.06, 0.13);
    A->sunr = L * 0.06;
    switch (A->kind) {
    case E_HARVEST: A->fz = 330; break;
    case E_REFUGE:
        A->fz = 230;
        A->suny = YH - L * rr(&r, 0.42, 0.52);
        A->sunr = L * 0.045;
        break;
    case E_RISE:
        A->fz = 175;
        A->fx = CX + fw * A->dir * rr(&r, 0.04, 0.10);
        A->sunx = A->fx + A->dir * fw * rr(&r, 0.16, 0.26);   /* the sun just off the worm's shoulder */
        A->suny = YH - L * rr(&r, 0.04, 0.09);
        A->sunr = L * 0.075;
        A->fgamp = rr(&r, 1.6, 2.4);
        break;
    case E_HORIZON:
        A->fz = 1800;
        A->sunx = A->fx;
        A->suny = YH - L * rr(&r, 0.07, 0.11);
        A->sunr = L * rr(&r, 0.19, 0.23);
        A->fgamp = rr(&r, 2.0, 2.8);
        break;
    case E_STORM: A->fz = 650; A->sunr = L * 0.05; A->suny = YH - L * 0.22; break;
    case E_NIGHT:
        A->fz = 2200;
        A->fgamp = rr(&r, 2.2, 3.0);
        A->sunx = A->fx + A->dir * fw * 0.24;      /* the big moon */
        A->suny = YH - L * rr(&r, 0.34, 0.44);
        A->sunr = L * 0.05;
        break;
    case E_BLOOM: A->fz = 420; A->sunx = A->fx - A->dir * fw * rr(&r, 0.12, 0.22); break;
    }
}

#define NAC 3
static Act acache[NAC];
static int acache_ok[NAC];
static const Act *act_get(long n) {
    int s = (int)(n % NAC);
    if (!acache_ok[s] || acache[s].n != n) { act_build(&acache[s], n); acache_ok[s] = 1; }
    return &acache[s];
}

/* ==== the land ==== */
/* a dune's cross-section: a long windward slope up to a sharp crest at a,
 * then the steep slip face; 0 in the trough, 1 on the crest */
static double dune_shape(double ph, double a) {
    double u = ph - floor(ph);
    double w = u < a ? u / a : (1 - u) / (1 - a);
    return pow(sin(w * M_PI / 2), 1.7);
}

static double layer_h(const Act *A, int k, double X, double *shape) {
    uint32_t salt = A->ts * 8u + (uint32_t)k * 3u + 1000u;
    double h, sh;
    if (k == 0) {
        /* mesas and buttes: flat tops, sheer walls, talus */
        double v = vn1(X / 2600.0, salt), v2 = vn1(X / 520.0, salt + 1);
        double m = sstep(0.585, 0.60, v + 0.05 * v2);
        double talus = sstep(0.38, 0.60, v);
        h = m * (200 + 380 * vn1(X / 9000.0, salt + 2)) * (0.86 + 0.14 * fbm1(X / 300.0, salt + 4)) + talus * 90 * (0.7 + 0.3 * v2) + 40 * fbm1(X / 1700.0, salt + 3);
        sh = m;
    } else {
        double lam = LLAM[k];
        double ph = X / lam + 0.8 * fbm1(X / (2.2 * lam), salt);
        double amp = LA[k] * (0.3 + 1.0 * vn1(X / (2.9 * lam), salt + 1));
        sh = dune_shape(ph, 0.72);
        h = amp * (sh + 0.04 * dune_shape(ph * 2.6 + 0.3, 0.6));
        /* open ground in front of the action */
        if (LZ[k] < A->fz * 0.92) {
            double Xf = X_at(A->fx, LZ[k], A->cam0), wm = 0.26 * fr[2] * LZ[k] / F;
            h *= 1 - 0.8 * exp(-sq((X - Xf) / wm));
        }
        /* the near dune rises into a dark mass on one side of the frame */
        if (k == NL - 1) {
            double sxn = (F * (X - A->cam0) / LZ[k] + CX - A->fx) / fr[2] * A->fgside;
            double bump = sstep(0.12, 0.62, sxn) * (1 - sstep(0.85, 1.5, sxn));
            h = h * (1 + A->fgamp * bump) + 14 * A->fgamp * sstep(0.2, 0.7, sxn) * (1 - sstep(0.85, 1.5, sxn));
        }
    }
    if (shape) *shape = sh;
    return h;
}

/* ==== light ==== */
typedef float C3[3];
typedef struct {
    C3 top, hor, sun, haze, sunhaze, sandL, sandD, rim, hide, dust, rock, metal;
    float hazek;                       /* haze per metre */
    float elev;                        /* the sun's elevation for the slopes, rad */
    float suni;                        /* the sun's strength */
} Light;

static const Light presets[NLT] = {
    /* gold: late afternoon, low warm sun */
    {{0.30f, 0.17f, 0.12f}, {0.98f, 0.66f, 0.36f}, {1.00f, 0.92f, 0.72f}, {0.92f, 0.60f, 0.34f}, {0.60f, 0.36f, 0.14f},
     {0.90f, 0.56f, 0.29f}, {0.36f, 0.17f, 0.09f}, {1.00f, 0.80f, 0.52f}, {0.19f, 0.14f, 0.12f}, {0.80f, 0.52f, 0.30f},
     {0.30f, 0.17f, 0.11f}, {0.20f, 0.15f, 0.13f}, 1 / 2600.0f, 0.16f, 1.0f},
    /* blaze: a high white sun, bleached sand */
    {{0.58f, 0.45f, 0.34f}, {0.99f, 0.86f, 0.64f}, {1.00f, 0.99f, 0.94f}, {0.96f, 0.82f, 0.60f}, {0.35f, 0.28f, 0.16f},
     {0.97f, 0.74f, 0.46f}, {0.56f, 0.33f, 0.19f}, {1.00f, 0.94f, 0.80f}, {0.21f, 0.16f, 0.13f}, {0.90f, 0.72f, 0.50f},
     {0.44f, 0.29f, 0.19f}, {0.28f, 0.23f, 0.20f}, 1 / 1800.0f, 0.65f, 1.0f},
    /* dusk: the sun on the horizon, deep orange, everything backlit */
    {{0.15f, 0.07f, 0.08f}, {0.96f, 0.46f, 0.17f}, {1.00f, 0.86f, 0.56f}, {0.88f, 0.42f, 0.17f}, {0.70f, 0.34f, 0.08f},
     {0.80f, 0.40f, 0.17f}, {0.19f, 0.08f, 0.05f}, {1.00f, 0.68f, 0.36f}, {0.13f, 0.085f, 0.075f}, {0.70f, 0.38f, 0.19f},
     {0.16f, 0.08f, 0.06f}, {0.12f, 0.08f, 0.07f}, 1 / 2000.0f, 0.07f, 1.0f},
    /* set: a huge red sun going down */
    {{0.10f, 0.045f, 0.07f}, {0.88f, 0.32f, 0.12f}, {1.00f, 0.72f, 0.42f}, {0.78f, 0.29f, 0.12f}, {0.62f, 0.24f, 0.05f},
     {0.64f, 0.27f, 0.12f}, {0.11f, 0.045f, 0.035f}, {1.00f, 0.52f, 0.25f}, {0.07f, 0.045f, 0.045f}, {0.56f, 0.25f, 0.12f},
     {0.09f, 0.045f, 0.04f}, {0.06f, 0.04f, 0.04f}, 1 / 2300.0f, 0.03f, 1.0f},
    /* storm: brown gloom */
    {{0.24f, 0.16f, 0.11f}, {0.66f, 0.46f, 0.28f}, {0.95f, 0.86f, 0.66f}, {0.60f, 0.42f, 0.25f}, {0.26f, 0.16f, 0.07f},
     {0.68f, 0.45f, 0.26f}, {0.28f, 0.17f, 0.10f}, {0.92f, 0.72f, 0.46f}, {0.19f, 0.14f, 0.11f}, {0.52f, 0.35f, 0.21f},
     {0.24f, 0.15f, 0.10f}, {0.16f, 0.12f, 0.10f}, 1 / 1500.0f, 0.30f, 0.65f},
    /* night: two moons */
    {{0.006f, 0.010f, 0.030f}, {0.065f, 0.085f, 0.15f}, {0.86f, 0.90f, 1.00f}, {0.085f, 0.105f, 0.17f}, {0.10f, 0.12f, 0.19f},
     {0.34f, 0.38f, 0.49f}, {0.030f, 0.035f, 0.060f}, {0.66f, 0.76f, 0.98f}, {0.035f, 0.035f, 0.050f}, {0.22f, 0.25f, 0.33f},
     {0.035f, 0.038f, 0.055f}, {0.03f, 0.03f, 0.04f}, 1 / 2600.0f, 0.45f, 0.8f},
};
static const C3 GLOW = {1.00f, 0.42f, 0.10f};  /* inside the maw */
static const C3 BONE = {0.93f, 0.86f, 0.72f};
static const C3 AMBER = {1.00f, 0.50f, 0.13f}; /* the glittering dust */

/* palettes grade the presets */
static const struct palette {
    const char *name;
    float sat, lift;
    C3 tint;
} palettes[] = {
    {"ember", 1.00f, 0.00f, {1.00f, 1.00f, 1.00f}},
    {"ash", 0.42f, 0.05f, {1.00f, 0.96f, 0.88f}},
    {"blood", 1.15f, 0.00f, {1.10f, 0.74f, 0.66f}},
    {"violet", 0.90f, 0.00f, {0.98f, 0.80f, 1.05f}},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float cap_col[3], cap_halo[3];

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "maw: no palette \"%s\" (ember, ash, blood, violet)\n", opt_palette);
            exit(1);
        }
    } else {
        uint32_t h = opt_seed * 2654435761u;
        h ^= h >> 15;
        pal = (int)(h % NPAL);
    }
    for (int c = 0; c < 3; c++) { cap_col[c] = 0.97f - 0.03f * c; cap_halo[c] = 0.05f; }
}

static void grade(float *c) {
    const struct palette *P = &palettes[pal];
    float l = 0.30f * c[0] + 0.59f * c[1] + 0.11f * c[2];
    for (int i = 0; i < 3; i++) {
        float v = l + (c[i] - l) * P->sat;
        v = v * P->tint[i];
        v = v + P->lift * (1 - v);
        c[i] = v < 0 ? 0 : v;
    }
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

static const char *VS_FULL = "#version 330 core\n" GLSL(
    uniform float uDepth;
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, uDepth * 2.0 - 1.0, 1.0);
    });

/* shared by every fragment shader: the view, the haze and the sun */
static const char *FS_COMMON = "#version 330 core\n" GLSL(
    uniform vec3 uView;
    uniform vec2 uRes;
    uniform float uL, uT, uYH, uF, uCX, uCamX;
    uniform sampler2D uNoise;
    uniform vec3 uHaze, uSunHaze, uFogC, uRim, uSandL, uSandD, uDust, uHide, uAmber, uSunC;
    uniform vec4 uSun;
    uniform float uHazeK, uFogK;
    uniform vec3 uFlashL;
    const float ZF = 20000.0;
    const float CAMH = 35.0;
    vec2 wpos() { return vec2(uView.x + gl_FragCoord.x / uView.z, uView.y + (uRes.y - gl_FragCoord.y) / uView.z); }
    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
    float hs(float n) { return fract(sin(n * 127.1 + 0.37) * 43758.5453); }
    float sunk(vec2 w) { return exp(-length(w - uSun.xy) / uL * 2.2); }
    vec3 hazeAt(vec2 w) {
        float d = length(w - uSun.xy) / uL;
        return uHaze + uSunHaze * (0.85 * exp(-d * 2.0) + 0.9 * exp(-d * 7.0)) * uSun.w;
    }
    vec3 airp(vec3 c, float z, vec2 w) {
        c = mix(c, hazeAt(w), (1.0 - exp(-z * uHazeK)) * 0.9);
        vec3 fc = uFogC + vec3(1.0, 0.86, 0.72) * uFlashL.z * exp(-length(w - uFlashL.xy) / uL * 1.6);
        return mix(c, fc, 1.0 - exp(-z * uFogK));
    }
    /* toward the sun, in (right, up) */
    vec2 sunDir(vec2 w) { vec2 d = uSun.xy - w; d.y = -d.y; return d / max(length(d), 1e-3); }
);

/* the sky and the land: eight layers of ridges, nearest last. Their crest
 * lines (wall y), how sunlit each slope is and how near a crest, per column,
 * come from uRows. Writes the depth of what it draws. */
static const char *FS_LAND = GLSL(
    uniform sampler2D uRows;
    uniform vec3 uRowX;
    uniform float uZ[8];
    uniform vec3 uSkyTop, uSkyHor, uRock;
    uniform float uNight, uSunDisc, uPlume, uVein;
    uniform vec4 uMoon1, uMoon2;
    uniform vec2 uShake;
    out vec4 o;

    vec3 sky(vec2 w) {
        float hy = (uYH - w.y) / uL;
        vec3 c = mix(uSkyHor, uSkyTop, pow(clamp(hy / 1.25, 0.0, 1.0), 0.55));
        c = mix(c, hazeAt(w), exp(-max(hy, 0.0) / 0.07) * 0.85);
        float px = 1.0 / (uView.z * uL);
        /* long bands of dust in the air */
        float band = texture(uNoise, vec2(w.x / uL * 0.05 + uT * 0.0006, w.y / uL * 0.8)).r;
        float band2 = texture(uNoise, vec2(w.x / uL * 0.11 - uT * 0.001, w.y / uL * 4.5)).g;
        float veil = smoothstep(0.25, 0.75, band) * (0.75 + 0.25 * band2);
        c *= 0.94 + 0.10 * veil;
        float d = length(w - uSun.xy) / uL, rs = uSun.z / uL;
        if (uSunDisc > 0.5) {
            float disc = clamp((rs - d) / px + 0.5, 0.0, 1.0);
            /* corona, and long rays through the dust */
            vec2 dv = w - uSun.xy;
            float ang = atan(dv.y, dv.x);
            float rays = texture(uNoise, vec2(ang * 1.3, uT * 0.004)).r * texture(uNoise, vec2(ang * 3.1 + 0.4, 0.5)).g;
            c += uSunC * uSun.w * (0.20 * exp(-d * 2.4) + 0.42 * exp(-d / max(rs * 0.9, 0.02)) + 0.07 * smoothstep(0.25, 0.65, rays) * exp(-d * 2.2)) * smoothstep(-0.05, 0.1, hy);
            float through = mix(0.80, 1.0, veil);
            vec3 sd = uSunC * (1.12 - 0.12 * pow(clamp(d / rs, 0.0, 1.0), 3.0)) * through;
            c = mix(c, max(sd, c), disc * uSun.w);
        }
        if (uNight > 0.0) {
            /* stars, thinning toward the horizon */
            float cell = uL * 0.0055;
            vec2 g = floor(w / cell), f = fract(w / cell) - 0.5;
            float h = h21(g);
            if (h > 0.93) {
                vec2 off = vec2(h21(g + 3.7), h21(g + 9.1)) - 0.5;
                float r = length(f - off * 0.6) * cell * uView.z;
                float b = pow((h - 0.93) / 0.07, 3.0) * (0.6 + 0.4 * sin(uT * (1.0 + 3.0 * h21(g + 1.3)) + h * 50.0));
                c += vec3(0.85, 0.9, 1.0) * b * exp(-r * r * 0.9) * smoothstep(0.0, 0.25, hy) * uNight;
            }
            /* a faint river of stars */
            float mw = exp(-pow((w.y - uYH + (w.x - uCX) * 0.35 + uL * 0.7) / (uL * 0.22), 2.0));
            c += vec3(0.07, 0.08, 0.12) * mw * texture(uNoise, w / uL * 0.6).r * uNight * smoothstep(0.0, 0.3, hy);
            /* the moons */
            for (int i = 0; i < 2; i++) {
                vec4 M = i == 0 ? uMoon1 : uMoon2;
                vec2 q = (w - M.xy) / M.z;
                float r = length(q);
                float cov = clamp((1.0 - r) * M.z * uView.z + 0.5, 0.0, 1.0);
                vec3 mc = i == 0 ? vec3(0.92, 0.93, 0.98) : vec3(0.95, 0.80, 0.70);
                float lit = smoothstep(-0.15, 0.15, q.x * M.w + sqrt(max(1.0 - r * r, 0.0)) * 0.6 - 0.2);
                float mare = texture(uNoise, q * 0.08 + float(i) * 0.3).r;
                vec3 disc = mc * (0.55 + 0.45 * smoothstep(0.35, 0.7, mare)) * mix(0.06, 1.0, lit);
                c += mc * 0.10 * exp(-max(r - 1.0, 0.0) * 1.3) * uNight;
                c = mix(c, disc, cov * uNight);
            }
        }
        return c;
    }

    vec3 land(int k, vec2 w, float ry, float lit, float litb, float crest) {
        float z = uZ[k];
        float s = uF / z;
        float dm = max(w.y - ry, 0.0) / s;
        float Xm = (w.x - uCX) / s + uCamX;
        vec3 c;
        if (k == 0) {
            float g = texture(uNoise, vec2(Xm / 1500.0, dm / 260.0)).r;
            float strata = 0.85 + 0.15 * sin(dm * 0.09 + g * 6.0);
            c = uRock * (0.8 + 0.4 * g) * strata * mix(0.8, 1.35, lit);
        } else {
            /* sunlit and shaded faces run down from the crest line, softening */
            float sh = mix(lit, litb, smoothstep(0.0, 6.0 + 0.02 * z, dm)) * exp(-dm / (18.0 + 0.05 * z));
            c = mix(uSandD, uSandL, sh);
            c = mix(c, mix(uSandD, uSandL, 0.35), 0.25 * smoothstep(10.0, 60.0 + 0.05 * z, dm));
            float rip = texture(uNoise, vec2(Xm / 30.0, dm / 1.2)).g;
            c *= 1.0 + 0.07 * (rip - 0.5) * exp(-z / 400.0) * (0.3 + sh);
            /* veins of amber sand, with glints */
            float vn = texture(uNoise, vec2(Xm / 380.0, dm / 14.0 + float(k) * 0.37)).r;
            float vein = smoothstep(0.035, 0.0, abs(vn - 0.62)) * uVein * exp(-z / 2500.0) * smoothstep(0.0, 3.0, dm);
            c = mix(c, uAmber * mix(0.55, 1.05, sh), vein * 0.4 * exp(-z / 700.0));
            vec2 gc = floor(gl_FragCoord.xy / 2.0);
            float gh = h21(gc + float(k) * 17.0);
            float tw = pow(max(sin(uT * (1.3 + 3.0 * h21(gc + 4.4)) + gh * 40.0), 0.0), 16.0);
            c += uAmber * step(0.93, gh) * tw * vein * 2.0;
        }
        /* the crest catches the light; brightest toward the sun */
        float px = 1.0 / uView.z;
        c += uRim * crest * exp(-(w.y - ry) / (px * 1.4 + 0.12 * s)) * (0.2 + 1.2 * sunk(w)) * uSun.w;
        return airp(c, z, w);
    }

    void main() {
        vec2 w = wpos() - uShake;
        float tu = ((w.x - uRowX.x) / uRowX.y + 0.5) / uRowX.z;
        vec4 A = texture(uRows, vec2(tu, 0.5 / 8.0)), B = texture(uRows, vec2(tu, 1.5 / 8.0));
        vec4 LA = texture(uRows, vec2(tu, 2.5 / 8.0)), LB = texture(uRows, vec2(tu, 3.5 / 8.0));
        vec4 CA = texture(uRows, vec2(tu, 4.5 / 8.0)), CB = texture(uRows, vec2(tu, 5.5 / 8.0));
        vec4 BA = texture(uRows, vec2(tu, 6.5 / 8.0)), BB = texture(uRows, vec2(tu, 7.5 / 8.0));
        float ry[8] = float[8](A.x, A.y, A.z, A.w, B.x, B.y, B.z, B.w);
        float li[8] = float[8](LA.x, LA.y, LA.z, LA.w, LB.x, LB.y, LB.z, LB.w);
        float cr[8] = float[8](CA.x, CA.y, CA.z, CA.w, CB.x, CB.y, CB.z, CB.w);
        float lb[8] = float[8](BA.x, BA.y, BA.z, BA.w, BB.x, BB.y, BB.z, BB.w);
        float px = 1.0 / uView.z;
        int k1 = -1;
        for (int k = 7; k >= 0; k--) if (w.y > ry[k] - 0.5 * px) { k1 = k; break; }
        float cov = k1 >= 0 ? clamp((w.y - ry[k1]) / px + 0.5, 0.0, 1.0) : 0.0;
        vec3 c = vec3(0.0);
        float depth = 1.0;
        if (cov < 1.0) {
            int k2 = -1;
            for (int k = k1 - 1; k >= 0; k--) if (w.y > ry[k] - 0.5 * px) { k2 = k; break; }
            if (k2 >= 0) {
                float c2 = clamp((w.y - ry[k2]) / px + 0.5, 0.0, 1.0);
                c = mix(sky(w), land(k2, w, ry[k2], li[k2], lb[k2], cr[k2]), c2);
                if (c2 > 0.5) depth = uZ[k2] / ZF;
            } else c = sky(w);
        }
        if (k1 >= 0) {
            c = mix(c, land(k1, w, ry[k1], li[k1], lb[k1], cr[k1]), cov);
            if (cov > 0.5) depth = uZ[k1] / ZF;
        }
        /* sand blown off the crests in front */
        if (uPlume > 0.0)
            for (int k = 7; k >= 4; k--) {
                if (k <= k1) break;
                float s = uF / uZ[k];
                float d = (ry[k] - w.y) / s;
                if (d < 0.0 || d > 30.0) continue;
                float Xm = (w.x - uCX) / s + uCamX;
                float n = texture(uNoise, vec2(Xm / 50.0 - uT * 0.22, d / 14.0 - uT * 0.04)).r;
                float a = cr[k] * uPlume * exp(-d / 6.0) * smoothstep(0.35, 0.8, n) * 0.6;
                c = mix(c, airp(uDust * (0.85 + 0.9 * sunk(w)), uZ[k], w), a);
            }
        o = vec4(c, 1.0);
        gl_FragDepth = depth;
    });

/* the worm's body: a strip along its spine carrying (along, across) */
static const char *VS_WORM = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec4 aB;
    layout(location = 2) in vec2 aN;
    uniform vec3 uView;
    uniform vec2 uRes;
    uniform float uDepth;
    out vec4 vB;
    out vec2 vN;
    void main() {
        vB = aB; vN = aN;
        vec2 d = (aPos - uView.xy) * uView.z;
        gl_Position = vec4(d.x / uRes.x * 2.0 - 1.0, 1.0 - d.y / uRes.y * 2.0, uDepth * 2.0 - 1.0, 1.0);
    });

static const char *FS_WORM = GLSL(
    in vec4 vB;                        /* metres from the tip, across (-1..1), radius (m), radius (wall px) */
    in vec2 vN;                        /* the across direction on screen (right, down) */
    uniform float uZw, uGround, uPour, uSil;
    uniform vec4 uMound;               /* x, half width, height (wall px) */
    out vec4 o;
    void main() {
        vec2 w = wpos();
        float px = 1.0 / uView.z;
        float gy = uGround - uMound.z * exp(-pow((w.x - uMound.x) / max(uMound.y, 1.0), 2.0));
        float gcov = clamp((gy - w.y) / px + 0.5, 0.0, 1.0);
        if (gcov <= 0.0) discard;
        float a = clamp(vB.y, -1.0, 1.0), s = vB.x, R = vB.z;
        float ecov = clamp((1.0 - abs(vB.y)) * vB.w / px * 0.9 + 0.5, 0.0, 1.0);
        float nzz = sqrt(max(1.0 - a * a, 0.0));
        float ang = asin(a);
        /* armour: deep folds at irregular intervals, and between them slabs
         * of plate, staggered, weathered, cracked and scarred */
        float wob = texture(uNoise, vec2(s / (R * 9.0), 0.31)).r;
        float q = s / (0.95 * R) + 1.4 * wob;
        float qi = floor(q), f = fract(q);
        float hq = hs(qi);
        float fold = 1.0 - smoothstep(0.0, 0.07 + 0.05 * hq, f);
        float swell = smoothstep(0.0, 0.5, f) * (1.0 - 0.6 * smoothstep(0.75, 1.0, f));
        float q2 = s / (0.32 * R) + 2.0 * wob + hq * 3.0;
        float fine = 0.5 + 0.5 * sin(q2 * 6.28318);
        float g = ang * (1.0 + 1.3 * hq) + hq * 6.0 + 0.6 * sin(qi * 1.7 + ang * 2.0);
        float gi = floor(g), gf = fract(g);
        float hg = hs(qi * 13.0 + gi);
        float vseam = (1.0 - smoothstep(0.0, 0.03, min(gf, 1.0 - gf))) * step(0.3, hg) * (1.0 - fold);
        vec2 cuv = vec2(ang * R / 26.0, s / 26.0);
        float big = texture(uNoise, cuv * 0.22 + vec2(0.13, hq * 0.1)).r;
        float n1 = texture(uNoise, cuv * 0.8 + vec2(hq, hg)).r;
        float n3 = texture(uNoise, cuv * 1.9 + vec2(0.7, hg)).r;
        float crack = max(smoothstep(0.022, 0.0, abs(n1 - 0.5)) * step(0.2, hg), smoothstep(0.018, 0.0, abs(n3 - 0.5)) * 0.7);
        float n2 = texture(uNoise, cuv * 3.1).g;
        float pits = smoothstep(0.72, 0.82, n2) * big;
        float sc = (s * 0.5 + ang * R * 1.1) / (6.0 * R);
        float scar = smoothstep(0.035, 0.0, abs(fract(sc) - 0.5 + 0.08 * (n1 - 0.5))) * step(0.55, hs(floor(sc) + 3.0));
        /* light: the sun low and behind it, sky above, sand below */
        vec3 N = vec3(vN.x * a, -vN.y * a, nzz);
        vec2 sd = sunDir(w);
        vec3 Ls = normalize(vec3(sd * 0.85, -0.5));
        float rim = pow(max(dot(N, Ls), 0.0), 3.0);
        float lam = max(dot(N, normalize(vec3(sd * 0.6, 0.4))), 0.0);
        float slab = 0.75 + 0.5 * hg * hg;
        vec3 c = uHide * (0.55 + 0.55 * swell) * (0.85 + 0.3 * fine) * slab * (0.55 + 0.9 * big);
        c *= (1.0 - 0.8 * fold) * (1.0 - 0.45 * vseam) * (1.0 - 0.65 * crack - 0.35 * pits);
        c *= (0.5 + 0.6 * nzz) * (0.65 + 0.7 * lam);
        c += uHide * 1.1 * scar * nzz * (0.5 + big);
        c = mix(c, uDust * 0.5, clamp(fold * 0.5 + crack * 0.4, 0.0, 1.0) * smoothstep(0.4, 0.7, big) * 0.6);
        c += uRim * rim * (0.5 + 0.7 * swell) * (0.7 + 0.5 * fine) * 2.4 * uSun.w;
        c += uSandL * 0.10 * max(-N.y, 0.0);
        /* sky on the tops of the swells */
        c += uHaze * 0.16 * swell * (1.0 - fold) * (0.4 + 0.6 * nzz) * (0.6 + 0.6 * big);
        float ridge = swell;
        /* sand pouring off it in sheets */
        float st = texture(uNoise, vec2(a * 0.35, (s - uT * 26.0) / (R * 9.0))).r;
        float st2 = texture(uNoise, vec2(a * 1.3 + 0.3, (s - uT * 26.0) / (R * 3.0))).g;
        float pour = uPour * smoothstep(0.5, 0.8, st) * (0.5 + 0.5 * st2) * (0.4 + 0.6 * ridge);
        c = mix(c, uDust * (0.7 + 0.8 * rim + 0.5 * sunk(w)), pour * 0.7);
        c = airp(c, uZw, w);
        c = mix(c, uFogC * 0.08 + uRim * 0.25 * rim, uSil);
        o = vec4(c, gcov * ecov);
    });

/* the maw: an ellipse (a circle seen at an angle) with an armoured lip and,
 * inside, eleven rings of needle teeth going down a glowing throat */
static const char *FS_MAW = GLSL(
    uniform vec2 uMC, uMA, uMT, uShift;
    uniform float uMR, uMK, uOpen, uRot, uGlowI, uZw, uSil;
    uniform vec3 uGlow, uBone;
    out vec4 o;
    void main() {
        vec2 w = wpos();
        vec2 d = w - uMC;
        vec2 p = vec2(dot(d, uMA), dot(d, uMT) / uMK) / uMR;
        float r0 = length(p);
        float th = atan(p.y, p.x);
        /* the lip is a ring of armour plates whose tips stand proud */
        float ns = 23.0;
        float g = th / 6.28318 * ns + 0.2 * sin(th * 3.0 + 1.0);
        float gi = floor(g), gf = fract(g);
        float hg = hs(gi + 7.0);
        float edge = 1.0 + 0.012 * sqrt(max(1.0 - pow(abs(gf - 0.5) * 2.0, 2.0), 0.0)) * (0.4 + hg) - 0.01 * smoothstep(0.7, 0.9, hg) * step(abs(gf - 0.3), 0.15);
        float r = r0 / edge;
        float fw = max(fwidth(r), 1e-4);
        float cov = clamp((1.0 - r) / fw + 0.5, 0.0, 1.0);
        if (cov <= 0.0) discard;
        float rin = 0.82 * uOpen;
        vec2 sd = sunDir(w);
        vec3 c;
        if (r > rin) {
            float v = (r - rin) / max(1.0 - rin, 1e-3);
            float seam = 1.0 - smoothstep(0.0, 0.06, min(gf, 1.0 - gf));
            float bulge = pow(sin(clamp(v, 0.0, 1.0) * 3.14159), 0.5);
            vec2 ds = uMA * p.x + uMT * p.y * uMK;
            ds.y = -ds.y;
            ds /= max(length(ds), 1e-4);
            float rim = max(dot(ds, sd), 0.0) * smoothstep(0.35, 1.0, v);
            float growth = 0.88 + 0.12 * sin(v * 34.0 + hg * 9.0);
            c = uHide * (0.3 + 0.55 * bulge * (0.6 + 0.8 * hg)) * (1.0 - 0.75 * seam) * growth;
            float n1 = texture(uNoise, p * 0.9 + hg).r;
            c *= 1.0 - 0.6 * smoothstep(0.03, 0.0, abs(n1 - 0.5));
            c = mix(c, uDust * 0.6, smoothstep(0.6, 0.85, texture(uNoise, p * 1.6).r) * 0.3 * smoothstep(0.4, 1.0, v));
            c += uRim * rim * 1.5 * uSun.w;
            /* the inner edge of the lip: wet, dark, catching the throat's glow */
            if (uOpen > 0.02) c = mix(c, vec3(0.10, 0.03, 0.02) + uGlow * uGlowI * 0.35, smoothstep(0.25, 0.0, v));
        } else {
            vec3 acc = vec3(0.0);
            float T = 1.0;
            for (int j = 0; j < 12; j++) {
                float fj = float(j);
                float rj = rin * pow(0.83, fj);
                vec2 pj = p - uShift * (1.0 - rj / rin);
                float rr = length(pj);
                float x = (rj - rr) / (rj * mix(0.34, 0.2, min(fj / 5.0, 1.0)));
                if (x < -0.05 || x > 1.0) continue;
                float N = floor(mix(110.0, 210.0, min(fj / 4.0, 1.0)) * (0.5 + 0.5 * rj / 0.82) + 40.0);
                float tj = atan(pj.y, pj.x) / 6.28318 * N + uRot * (mod(fj, 2.0) * 2.0 - 1.0) * (1.0 + 0.3 * fj) + fj * 0.37;
                float ci = floor(tj), cf = fract(tj) - 0.5;
                float hj = hs(ci + fj * 71.0);
                float len = 0.45 + 0.55 * hj * hj;
                float xx = x / len;
                float wdt = (0.36 + 0.18 * hs(ci * 3.1 + fj)) * pow(clamp(1.0 - xx, 0.0, 1.0), 1.5);
                float curl = cf - 0.35 * xx * xx * (hj - 0.5) * 2.0;
                float cellpx = 6.28318 * rr * uMR / N * uView.z;
                float tc = clamp((wdt - abs(curl)) * cellpx + 0.5, 0.0, 1.0) * step(xx, 1.0) * smoothstep(-0.05, 0.03, x) * step(0.08, hs(ci + fj * 9.0));
                float dj = fj / 11.0;
                /* bone, darker at the root, lit by the glow from below */
                vec3 tooth = uBone * mix(0.22, 0.8, smoothstep(0.0, 0.9, xx)) * (0.7 + 0.45 * hj) * mix(0.8, 0.10, dj);
                tooth += uGlow * uGlowI * (0.08 + 1.0 * dj * dj + 0.25 * dj) * (0.3 + 0.7 * xx);
                acc += T * tc * tooth;
                T *= 1.0 - tc;
                if (T < 0.02) break;
            }
            /* the gullet: wet ridged flesh going down to the glow */
            float dr = r / max(rin, 1e-3);
            float depth = -log(max(dr, 0.01));
            vec3 flesh = vec3(0.16, 0.045, 0.03) * (0.7 + 0.5 * sin(depth * 12.0 + th * 2.0 * 0.0));
            flesh += vec3(0.5, 0.2, 0.12) * pow(max(sin(depth * 12.0 + 1.2), 0.0), 18.0) * 0.4;
            vec3 throat = mix(uGlow * uGlowI * 1.7 + vec3(0.25, 0.12, 0.0) * uGlowI, flesh, smoothstep(0.0, 0.55, dr));
            c = acc + T * throat;
        }
        vec3 glowing = r < rin ? c * smoothstep(0.7, 0.0, r / max(rin, 1e-3)) : vec3(0.0);
        c = airp(c, uZw, w);
        c = mix(c, uFogC * 0.08 + glowing * 0.8, uSil);
        o = vec4(c, cov);
    });

/* a screen-space rectangle (uBox: x0, y0, x1, y1 in wall px) */
static const char *VS_BOX = "#version 330 core\n" GLSL(
    uniform vec3 uView;
    uniform vec2 uRes;
    uniform vec4 uBox;
    uniform float uDepth;
    void main() {
        vec2 k[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
        vec2 c = mix(uBox.xy, uBox.zw, k[gl_VertexID]);
        vec2 d = (c - uView.xy) * uView.z;
        gl_Position = vec4(d.x / uRes.x * 2.0 - 1.0, 1.0 - d.y / uRes.y * 2.0, uDepth * 2.0 - 1.0, 1.0);
    });

/* everything else: depth-sorted quads with a local frame (y up) */
static const char *VS_SPR = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aL;
    layout(location = 2) in vec4 aP;
    layout(location = 3) in vec4 aQ;
    uniform vec3 uView;
    uniform vec2 uRes;
    out vec2 vL;
    flat out vec4 vP;
    flat out vec4 vQ;
    void main() {
        vL = aL; vP = aP; vQ = aQ;
        vec2 d = (aPos - uView.xy) * uView.z;
        gl_Position = vec4(d.x / uRes.x * 2.0 - 1.0, 1.0 - d.y / uRes.y * 2.0, aQ.x / 20000.0 * 2.0 - 1.0, 1.0);
    });

static const char *FS_SPR = GLSL(
    in vec2 vL;
    flat in vec4 vP;                   /* type, three parameters */
    flat in vec4 vQ;                   /* distance, wall px per local unit, two more */
    uniform vec3 uMetal, uRockC;
    uniform float uRock[24];
    out vec4 o;
    float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
    float sdSeg(vec2 p, vec2 a, vec2 b) {
        vec2 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h);
    }
    vec3 rimlit(vec3 base, float d, float aa, vec2 w) {
        vec2 g = vec2(dFdx(d), dFdy(d));
        float gl = length(g);
        vec2 n = gl > 0.0 ? g / gl : vec2(0.0);
        float edge = exp(min(d, 0.0) / (aa * 2.0));
        return base + uRim * max(dot(n, sunDir(w)), 0.0) * edge * 0.9 * uSun.w;
    }
    void main() {
        int ty = int(vP.x + 0.5);
        vec2 w = wpos();
        vec2 p = vL;
        float aa = 1.0 / (vQ.y * uView.z);
        float z = vQ.x;
        vec3 c = vec3(0.0), emit = vec3(0.0);
        float a = 0.0;
        if (ty == 0) {                 /* crawler: tracks, hull, cab, mast, the scoop boom */
            p.x *= vP.z;
            float d = sdBox(p - vec2(0.0, 1.8), vec2(14.0, 1.8)) - 0.3;
            d = min(d, sdBox(p - vec2(-1.5, 6.4), vec2(12.0, 3.0)) - 0.6);
            d = min(d, sdBox(p - vec2(8.6, 10.6), vec2(3.3, 1.7)) - 0.3);
            d = min(d, sdBox(p - vec2(-4.0, 13.0), vec2(0.7, 4.5)));
            d = min(d, sdSeg(p, vec2(-4.0, 17.0), vec2(-16.0, 1.4)) - 0.55);
            d = min(d, sdSeg(p, vec2(-4.0, 17.0), vec2(5.0, 10.0)) - 0.3);
            d = min(d, length(p - vec2(-16.0, 1.6)) - 1.9);
            a = clamp(0.5 - d / aa, 0.0, 1.0);
            vec3 base = uMetal * (0.85 + 0.3 * step(3.6, p.y));
            base *= 1.0 - 0.35 * step(p.y, 3.6) * step(0.5, fract(p.x * 0.45 - vP.w * 0.5));
            c = rimlit(base, d, aa, w);
            float win = smoothstep(0.7, 0.2, abs(p.y - 10.9)) * step(abs(p.x - 8.8), 2.6);
            emit = vec3(1.0, 0.72, 0.38) * vP.y * (win * 1.6 + 2.0 * exp(-length(p - vec2(-16.0, 3.5)) * 0.8));
            emit += uAmber * vP.w * 0.0;
        } else if (ty == 1) {          /* flyer: a pod, a tail, two pairs of flickering wings */
            p.x *= vP.z;
            float d = sdSeg(p, vec2(-3.0, 0.0), vec2(3.5, 0.3)) - 1.3;
            d = min(d, length(p - vec2(4.4, 0.5)) - 1.7);
            d = min(d, sdSeg(p, vec2(-3.0, 0.2), vec2(-10.5, 0.9)) - 0.32);
            d = min(d, sdSeg(p, vec2(-0.5, -1.0), vec2(-1.2, -2.0)) - 0.16);
            float sa = clamp(0.5 - d / aa, 0.0, 1.0);
            float wa = 0.0;
            for (int k = 0; k < 2; k++) {
                float ph = vP.y + float(k) * 1.9;
                vec2 anc = vec2(k == 0 ? 1.4 : -1.4, 1.1);
                vec2 cc = anc + vec2(k == 0 ? 1.2 : -1.6, 0.4 + 1.6 * sin(ph));
                vec2 q = (p - cc) / vec2(4.6, 0.30 + 0.75 * abs(cos(ph)));
                float e = length(q);
                wa = max(wa, clamp((1.0 - e) * 4.6 / aa * 0.5 + 0.5, 0.0, 1.0) * 0.6);
            }
            c = rimlit(uMetal * 0.9, d, aa, w);
            c = mix(uRim * 0.6 + uHaze * 0.3, c, sa / max(sa + wa * (1.0 - sa), 1e-3));
            a = sa + wa * (1.0 - sa);
        } else if (ty == 2) {          /* a walker in a hooded robe */
            p.x *= vP.w;
            float lean = 0.10 + 0.22 * vP.z;
            vec2 q = vec2(p.x - lean * p.y, p.y);
            float d = length(q - vec2(0.02, 1.64)) - 0.15;
            float yy = clamp((q.y - 0.55) / 1.0, 0.0, 1.0);
            float hw = mix(0.30, 0.20, yy);
            d = min(d, max(abs(q.x + 0.04 - 0.08 * (1.0 - yy)) - hw, abs(q.y - 1.03) - 0.48));
            float st = 0.20 + 0.14 * vP.z;
            vec2 hip = vec2(0.0, 0.62);
            d = min(d, sdSeg(q, hip, vec2(sin(vP.y) * st, 0.04)) - 0.075);
            d = min(d, sdSeg(q, hip, vec2(-sin(vP.y) * st, 0.04)) - 0.075);
            d = min(d, sdSeg(q, vec2(0.30, 0.0), vec2(0.36, 1.85)) - 0.035 * step(0.5, vQ.z));
            a = clamp(0.5 - d / aa, 0.0, 1.0);
            c = uHide * 0.45 + uSandD * 0.15;
            c += uRim * 0.35 * max(dot(normalize(vec2(dFdx(d), dFdy(d)) + 1e-6), sunDir(w)), 0.0) * exp(min(d, 0.0) / (aa * 1.2)) * uSun.w;
        } else if (ty == 3) {          /* a rock outcrop from a 24-point profile */
            float u = (p.x / vP.y * 0.5 + 0.5) * 23.0;
            int i = int(clamp(floor(u), 0.0, 22.0));
            float f = clamp(u - float(i), 0.0, 1.0);
            float h = mix(uRock[i], uRock[i + 1], f);
            float sl = (uRock[i + 1] - uRock[i]) / (vP.y * 2.0 / 23.0);
            float d = (p.y - h) / sqrt(1.0 + sl * sl);
            d = max(d, -p.y - 4.0);
            a = clamp(0.5 - d / aa, 0.0, 1.0) * step(0.0, u) * step(u, 23.0);
            float n = texture(uNoise, p / vec2(9.0, 26.0)).r;
            float n2 = texture(uNoise, p / vec2(30.0, 30.0) + 0.3).g;
            vec3 base = uRockC * (0.5 + 0.7 * n * n2 + 0.2 * n) * (0.92 + 0.08 * sin(p.y * 1.7 + n * 9.0));
            base *= 1.0 - 0.5 * smoothstep(0.025, 0.0, abs(texture(uNoise, p / vec2(18.0, 50.0)).r - 0.5));
            base *= mix(0.7, 1.15, smoothstep(0.0, 25.0, p.y));
            c = rimlit(base, d, aa, w);
            c += uSandL * 0.25 * exp(d / 1.5);
        } else if (ty == 4) {          /* a swell of sand: a moving wake, or the ground rising */
            float Hh = vP.y, Wd = vP.z, Tr = vP.w;
            float x = p.x;
            float h = Hh * exp(-x * x / (Wd * Wd));
            float xs = x * sign(Tr + 1e-6);
            if (Tr != 0.0) h = max(h, Hh * 0.42 * exp(-max(xs, 0.0) / abs(Tr)) * smoothstep(-0.3 * Wd, 0.5 * Wd, xs));
            float d = p.y - h;
            a = clamp(0.5 - d / aa, 0.0, 1.0) * clamp((p.y + 2.0) / 2.0, 0.0, 1.0) * smoothstep(0.0, Hh * 0.4 + 0.5, h);
            float lit = exp(d / (0.3 * Hh + 0.5));
            vec3 base = mix(uSandD, uSandL, 0.25 + 0.6 * lit);
            float st = texture(uNoise, vec2(x / 9.0, (p.y + uT * 2.5) / 4.0)).r;
            base *= 0.85 + 0.3 * st;
            float cr = smoothstep(0.035, 0.0, abs(texture(uNoise, p / vec2(22.0, 9.0)).r - 0.5));
            emit = uAmber * vQ.z * cr * 2.2 * smoothstep(-Hh, 0.0, d);
            c = rimlit(base, d, aa, w);
        } else if (ty == 5 || ty == 6) { /* a pit opening in the sand; or a patch of amber sand */
            vec2 e = p / vec2(vP.y, vP.z);
            float r = length(e);
            float ang = atan(e.y, e.x);
            if (ty == 5) {
                float sw = texture(uNoise, vec2(ang / 6.283 * 4.0 + (1.0 - r) * 0.9 + vQ.z * 0.05, r * 0.7 - vQ.z * 0.3)).r;
                float deep = smoothstep(1.0, 0.05, r) * vP.w;
                c = mix(uSandL * 0.95, uSandD * 0.35, deep) * (0.75 + 0.5 * sw);
                c += uRim * 0.25 * smoothstep(0.7, 0.98, r) * step(0.0, e.y) * uSun.w;
                a = smoothstep(1.0, 0.8, r);
            } else {
                float n = texture(uNoise, e * 0.5 + vec2(vQ.w, 0.0)).r;
                float m = smoothstep(1.0, 0.55, r + (n - 0.5) * 0.7);
                c = mix(uSandL, uAmber, 0.65) * (0.75 + 0.5 * n);
                vec2 gc = floor(gl_FragCoord.xy / 2.0);
                float gh = h21(gc + 5.5);
                emit = uAmber * step(0.94, gh) * pow(max(sin(uT * (2.0 + 4.0 * h21(gc)) + gh * 30.0), 0.0), 12.0) * 1.8 * m;
                a = m * vP.w;
            }
        } else if (ty == 7) {          /* a billow of dust; amber dust glows and glitters */
            float r = length(p);
            if (r > 1.0) discard;
            float n = texture(uNoise, p * 0.20 + vec2(vP.y, vP.y * 0.71) + vec2(vP.w * 0.004, -vP.w * 0.008)).r;
            float n2 = texture(uNoise, p * 0.55 + vec2(vP.y * 1.3, 0.5)).g;
            float sh = r + (n - 0.5) * 0.95 + (n2 - 0.5) * 0.35;
            a = smoothstep(1.0, 0.3, sh) * smoothstep(1.0, 0.72, r) * vP.z;
            vec2 sd = sunDir(w);
            float lit = clamp(0.55 + 0.45 * dot(p, sd), 0.0, 1.2);
            vec3 base = uDust * mix(0.6, 1.15, lit) * (0.9 + 0.2 * n2);
            base += uSunHaze * sunk(w) * smoothstep(0.3, 1.0, sh) * 1.2 * uSun.w;
            c = mix(base, uAmber * vec3(0.95, 0.75, 0.6) * (0.55 + 0.5 * n2), vQ.z);
            emit = uAmber * vQ.z * 0.22 * smoothstep(1.0, 0.2, sh) * (0.6 + 0.8 * n2);
            vec2 gc = floor(gl_FragCoord.xy / 2.0);
            float gh = h21(gc + floor(vP.y * 10.0));
            emit += uAmber * step(0.985, gh) * pow(max(sin(uT * (2.0 + 5.0 * h21(gc + 2.0)) + gh * 50.0), 0.0), 10.0) * 2.5 * vQ.z * smoothstep(1.0, 0.4, sh);
        } else if (ty == 8) {          /* a falling sheet of sand */
            float u = p.x / vP.y, v = -p.y / vP.z;
            float n = texture(uNoise, vec2(u * 0.45 + vQ.z, v * vP.z / uL * 0.35 - uT * 0.25)).r;
            float n2 = texture(uNoise, vec2(u * 1.7 + vQ.z * 2.0, v * vP.z / uL * 0.9 - uT * 0.6)).g;
            float sheet = smoothstep(0.38, 0.7, n) * (0.45 + 0.75 * n2) * (0.6 + 0.4 * smoothstep(0.0, 0.3, v));
            a = sheet * (1.0 - u * u) * smoothstep(0.0, 0.06, v) * (1.0 - 0.3 * v) * vP.w * 0.8;
            a = max(a, smoothstep(0.75, 1.0, v) * (1.0 - u * u) * vP.w * 0.5 * n);
            c = uDust * (0.8 + 0.7 * sunk(w) + 0.3 * n2);
            c = mix(c, uAmber, vQ.w * 0.6);
        } else {                       /* a grain or a clod in flight */
            float d = length(vec2(max(abs(p.x) - vP.y, 0.0), p.y)) - vP.z;
            a = clamp(0.5 - d * uView.z, 0.0, 1.0) * vP.w;
            c = mix(uSandD * 0.9, uAmber * 1.4, vQ.z);
            emit = uAmber * vQ.z * 0.6;
        }
        c = airp(c, z, w) + emit * exp(-z * uFogK * 0.6);
        o = vec4(c, a);
    });

/* the wall of a dust storm, at one distance (depth-tested) */
static const char *FS_STORM = GLSL(
    uniform float uSZ, uSH, uSA, uFlash;
    uniform vec2 uFlashP;
    out vec4 o;
    void main() {
        vec2 w = wpos();
        float s = uF / uSZ;
        float Xm = (w.x - uCX) / s + uCamX;
        float b1 = texture(uNoise, vec2(Xm / 90000.0 + uT * 0.0002, 0.3)).r;
        float b2 = texture(uNoise, vec2(Xm / 22000.0 - uT * 0.0004, 0.7 + uT * 0.0003)).r;
        /* the top: a long swell with rounded lobes boiling on it */
        float lob = 0.0;
        for (int k = 0; k < 4; k++) {
            float lam = uSH * (0.8 / pow(1.9, float(k)));
            float xx = Xm / lam + texture(uNoise, vec2(Xm / lam * 0.012, 0.2 + float(k) * 0.3 + uT * 0.0002)).r * 2.5 + uT * 0.006 * float(k + 1) + float(k) * 0.37;
            float f = fract(xx) * 2.0 - 1.0;
            float hk = hs(floor(xx) + float(k) * 17.0);
            lob += pow(max(1.0 - f * f, 0.0), 0.65) * (0.26 / pow(1.7, float(k))) * (0.2 + 1.3 * hk * hk);
        }
        float topm = uSH * (0.45 + 0.45 * b1 + 0.1 * b2 + lob);
        float top = uYH + s * (CAMH - topm);
        float Ym = (uYH - w.y) / s + CAMH;
        float into = (w.y - top) / (s * uSH);
        vec2 m = vec2(Xm, Ym) / uSH;
        float n = texture(uNoise, m * 0.09 + vec2(uT * 0.0006, -uT * 0.0009)).r;
        float n2 = texture(uNoise, m * 0.3 + vec2(-uT * 0.0012, uT * 0.001) + n * 0.15).r;
        float bill = n * 0.6 + n2 * 0.4;
        float dens = smoothstep(-0.004, 0.02, into);
        float lit = smoothstep(0.3, 0.8, bill) * exp(-max(into, 0.0) * 2.5);
        vec3 c = uFogC * mix(0.6, 1.15, bill) * mix(1.1, 0.6, clamp(into * 1.2, 0.0, 1.0));
        c += uRim * (0.25 * lit + 0.3 * exp(-max(into, 0.0) * 30.0)) * (0.4 + 1.2 * sunk(w));
        c += vec3(1.0, 0.85, 0.7) * uFlash * exp(-length(w - uFlashP) / uL * 2.2) * (0.6 + 0.6 * n);
        o = vec4(c, dens * uSA);
    });

/* last: storm murk, lightning, the gust of dust, the amber glitter in the
 * air, the fade between acts and the vignette (premultiplied) */
static const char *FS_FX = GLSL(
    uniform float uStormIn, uFlash, uFade, uGlit, uVig;
    uniform vec4 uGust, uGlitC, uBolt, uDom;
    uniform vec2 uFlashP;
    uniform vec3 uFadeC;
    out vec4 o;
    void main() {
        vec2 w = wpos();
        vec3 c = vec3(0.0);
        float a = 0.0;
        if (uStormIn > 0.0) {
            float n = texture(uNoise, vec2(w.x / uL * 0.5 - uT * 0.8, w.y / uL * 1.6 + uT * 0.04)).r;
            float n2 = texture(uNoise, vec2(w.x / uL * 1.5 - uT * 2.0, w.y / uL * 5.0)).g;
            float d = (0.35 + 0.65 * smoothstep(0.3, 0.8, n)) * (0.75 + 0.25 * n2) * uStormIn;
            vec3 sc = uFogC * (0.75 + 0.5 * n2) + vec3(1.0, 0.85, 0.7) * uFlash * 0.5 * exp(-length(w - uFlashP) / uL * 1.5);
            c = sc * d;
            a = d;
        }
        if (uBolt.w > 0.0) {
            /* a jagged bolt from uBolt.xy down by uBolt.z, with a branch */
            float best = 1e9;
            vec2 prev = uBolt.xy, mid = uBolt.xy;
            float xo = 0.0;
            for (int i = 1; i <= 28; i++) {
                float fi = float(i);
                xo += (hs(fi + uBolt.w * 13.0) - 0.5) * uL * 0.035;
                vec2 nx = uBolt.xy + vec2(xo, uBolt.z * fi / 28.0);
                vec2 pa = w - prev, ba = nx - prev;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                best = min(best, length(pa - ba * h));
                prev = nx;
                if (i == 9) mid = nx;
            }
            prev = mid;
            for (int i = 1; i <= 10; i++) {
                float fi = float(i);
                vec2 nx = prev + vec2(uL * 0.012 + (hs(fi * 3.3 + uBolt.w) - 0.5) * uL * 0.03, uBolt.z * 0.025);
                vec2 pa = w - prev, ba = nx - prev;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                best = min(best, length(pa - ba * h) * 1.6);
                prev = nx;
            }
            float dp = best * uView.z;
            c += vec3(1.0, 0.93, 0.88) * uFlash * (exp(-dp * dp * 0.5) * 1.5 + exp(-dp / 30.0) * 0.3);
        }
        if (uGust.w > 0.0) {
            vec2 q = (w - uGust.xy) / vec2(1.0, 0.42);
            float e = length(q) / uGust.z;
            float n = texture(uNoise, vec2(w.x / uL * 0.7 - uT * 0.3, w.y / uL * 1.4 + uT * 0.1)).r;
            float front = exp(-pow((e - 1.0) * 3.5, 2.0));
            float inner = 1.0 - smoothstep(0.85, 1.0, e);
            float d = clamp((front * 0.7 + inner * 0.22) * (0.5 + 0.8 * n) * uGust.w, 0.0, 0.85);
            d *= smoothstep(uGust.y - uGust.z * 0.6 - uL * 0.1, uGust.y - uGust.z * 0.05, w.y + (n - 0.5) * uL * 0.25);
            vec3 gc = uDust * (0.85 + 0.5 * n) + uSunHaze * sunk(w) * 0.8;
            c = gc * d + c * (1.0 - d);
            a = d + a * (1.0 - d);
        }
        if (uGlit > 0.0) {
            /* fine amber glitter adrift in the air, brightest toward the sun and in the haze */
            float wgt = (0.25 + 1.9 * sunk(w)) * (0.35 + 0.9 * exp(-abs(w.y - uYH) / (uL * 0.3)));
            float boost = uGlitC.w * exp(-pow(length(w - uGlitC.xy) / uGlitC.z, 2.0));
            for (int k = 0; k < 2; k++) {
                float sc = uL * (k == 0 ? 0.010 : 0.017);
                vec2 q = (w - vec2(uT * uL * (0.012 + 0.010 * float(k)), -uT * uL * 0.002)) / sc;
                q.y += 0.3 * sin(q.x * 0.21 + uT * 0.3);
                vec2 id = floor(q), f = fract(q) - 0.5;
                float h = h21(id + float(k) * 31.0);
                float th = 1.0 - 0.07 * clamp(uGlit * wgt, 0.0, 1.0) - 0.13 * clamp(boost, 0.0, 1.0);
                if (h < th) continue;
                vec2 off = vec2(h21(id + 3.1), h21(id + 5.7)) - 0.5;
                float dd = length(f - off * 0.7) * sc * uView.z;
                float tw = pow(0.5 + 0.5 * sin(uT * (1.5 + 4.0 * h21(id + 1.9)) + h * 60.0), 5.0);
                c += uAmber * 1.5 * exp(-dd * dd * (k == 0 ? 0.9 : 0.5)) * tw * (wgt * uGlit + boost * 1.5) * (1.0 - a * 0.7);
            }
        }
        c = (1.0 - uFade) * c + uFade * uFadeC;
        a = 1.0 - (1.0 - uFade) * (1.0 - a);
        /* over the whole wall, so the seams between screens stay clean */
        vec2 sp = (w - uDom.xy) / (uDom.zw - uDom.xy) - 0.5;
        float v = uVig * smoothstep(0.35, 0.95, length(sp * vec2(1.0, (uDom.w - uDom.y) / (uDom.z - uDom.x) * 1.3)));
        c *= 1.0 - v;
        a = 1.0 - (1.0 - v) * (1.0 - a);
        o = vec4(c, a);
    });

static const char *VS_TEXT = "#version 330 core\n" GLSL(
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
static const char *FS_TEXT = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vCol;
    out vec4 o;
    void main() { o = vec4(vCol.rgb, vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r); });

static GLuint prog_land, prog_worm, prog_maw, prog_spr, prog_storm, prog_fx, prog_text;
static GLuint vao_empty, vao_text, vbo_text, vao_worm, vbo_worm, vao_spr, vbo_spr;
static GLuint tex_atlas, tex_noise, tex_rows;

/* a tileable 512x512 noise tile: R fbm (16 lattice cells across, 4
 * octaves), G and B value noise (64 cells) */
#define NT 512
static double tile_vn(double x, double y, int Lc, uint32_t salt) {
    double fx = floor(x), fy = floor(y);
    int ix = (int)fx, iy = (int)fy;
    double u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    int x0 = ((ix % Lc) + Lc) % Lc, y0 = ((iy % Lc) + Lc) % Lc, x1 = (x0 + 1) % Lc, y1 = (y0 + 1) % Lc;
    double a = lat(x0, y0, salt), b = lat(x1, y0, salt), c = lat(x0, y1, salt), d = lat(x1, y1, salt);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}
static void tex_params(GLint wrap, int mip) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
}
static void bake_noise(void) {
    unsigned char *t = malloc(NT * NT * 4);
    for (int y = 0; y < NT; y++)
        for (int x = 0; x < NT; x++) {
            double u = (double)x / NT, v = (double)y / NT, f = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 4; o++) {
                int Lc = 16 << o;
                f += amp * tile_vn(u * Lc, v * Lc, Lc, 900 + o);
                norm += amp; amp *= 0.5;
            }
            f /= norm;
            f = 0.5 + (f - 0.5) * 1.6;
            double g = tile_vn(u * 64, v * 64, 64, 911), b = tile_vn(u * 64, v * 64, 64, 913);
            unsigned char *o = t + 4 * (y * NT + x);
            o[0] = (unsigned char)(clamp01(f) * 255); o[1] = (unsigned char)(g * 255); o[2] = (unsigned char)(b * 255);
            o[3] = 255;
        }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NT, NT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tex_params(GL_REPEAT, 1);
    free(t);
}

static GLuint compile(GLenum type, const char *a, const char *b) {
    GLuint s = glCreateShader(type);
    const char *src[2] = {a, b};
    glShaderSource(s, b ? 2 : 1, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "maw: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}
static GLuint mk_prog(const char *vs, const char *fs, int common) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs, NULL));
    glAttachShader(pr, common ? compile(GL_FRAGMENT_SHADER, FS_COMMON, fs) : compile(GL_FRAGMENT_SHADER, fs, NULL));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}
#define U(p, n) glGetUniformLocation(p, n)

/* ==== this frame ==== */
static double v_ox, v_oy, v_sc;
static Light lt;                        /* the act's light, graded */
static float mesa_tint;

static struct {
    double shake, fogk, plume, vein, glit, fade, vig, light;
    C3 fogc;
    double storm_z, storm_h, storm_a, storm_in, flash, flx, fly;
    double bolt_x, bolt_y, bolt_len, bolt_seed;
    double gust_x, gust_y, gust_r, gust_a;
    double glx, gly, glr, gla;
    double moon[2][4];
    int night, sundisc;
} fx;

/* the worm: a spine from the tip (0) back, in metres at one distance */
#define NSP 180
typedef struct {
    int on;
    double z, R;
    double X[NSP + 1], Y[NSP + 1];
    double open, face, pour, glow, sil, rot;
    double mX, mW, mH;                 /* a swell of sand about its base (m) */
} Worm;
static Worm worm;

/* ---- sprites: quads with a local frame, sorted far to near ---- */
enum { T_CRAWLER, T_FLYER, T_WALKER, T_ROCK, T_MOUND, T_PIT, T_PATCH, T_PUFF, T_CURTAIN, T_GRAIN };
#define MAXSPR 2600
typedef struct { float z; float v[6 * 12]; } Spr;
static Spr sprs[MAXSPR];
static int nspr, sord[MAXSPR];
static float sprbuf[MAXSPR * 72];
static float rock_prof[24];

static void spr(int ty, double ax, double ay, double ang, double s, double x0, double x1, double y0, double y1, double z,
                double p1, double p2, double p3, double q2, double q3) {
    if (nspr >= MAXSPR || z < 1) return;
    double ux = cos(ang), uy = -sin(ang), vx = -sin(ang), vy = -cos(ang);
    double lx[4] = {x0, x1, x1, x0}, ly[4] = {y0, y0, y1, y1};
    double cx[4], cy[4], mnx = 1e30, mxx = -1e30, mny = 1e30, mxy = -1e30;
    for (int c = 0; c < 4; c++) {
        cx[c] = ax + (ux * lx[c] + vx * ly[c]) * s;
        cy[c] = ay + (uy * lx[c] + vy * ly[c]) * s;
        mnx = fmin(mnx, cx[c]); mxx = fmax(mxx, cx[c]); mny = fmin(mny, cy[c]); mxy = fmax(mxy, cy[c]);
    }
    if (mxx < v_ox || mnx > v_ox + W / v_sc || mxy < v_oy || mny > v_oy + H / v_sc) return;
    Spr *S = &sprs[nspr++];
    S->z = (float)z;
    static const int idx[6] = {0, 1, 2, 0, 2, 3};
    for (int i = 0; i < 6; i++) {
        int c = idx[i];
        float *o = S->v + i * 12;
        o[0] = (float)cx[c]; o[1] = (float)cy[c]; o[2] = (float)lx[c]; o[3] = (float)ly[c];
        o[4] = (float)ty; o[5] = (float)p1; o[6] = (float)p2; o[7] = (float)p3;
        o[8] = (float)z; o[9] = (float)s; o[10] = (float)q2; o[11] = (float)q3;
    }
}
static int spr_cmp(const void *a, const void *b) {
    float za = sprs[*(const int *)a].z, zb = sprs[*(const int *)b].z;
    return za < zb ? 1 : za > zb ? -1 : (*(const int *)a - *(const int *)b);
}

/* a billow of dust: X, Y (its middle), z in m, radius in m */
static void puff(double X, double Y, double z, double r, double dens, double seed, double age, double amber) {
    if (dens < 0.02) return;
    spr(T_PUFF, sx_of(X, z), sy_of(Y, z), 0, r * F / z, -1, 1, -1, 1, z, seed, fmin(dens, 1), age, amber, 0);
}

/* a burst of dust about (X, 0, z), t seconds after it started */
static void burst(double t, double X, double z, int n, double R0, double R1, double spread, double rise, double life,
                  double dens, uint32_t salt, double amber) {
    for (int i = 0; i < n; i++) {
        double age = t - hsh(i, salt) * 0.6;
        if (age < 0 || age > life) continue;
        double side = (hsh(i, salt + 1) - 0.5) * 2;
        double g = 1 - exp(-age / 1.5);
        double X_ = X + side * (R0 * 0.6 + spread * g);
        double Y_ = R0 * 0.35 + rise * hsh(i, salt + 2) * (1 - exp(-age / 2.5)) - age * 0.6;
        double r = R0 + (R1 - R0) * (1 - exp(-age / 3.5)) * (0.6 + 0.6 * hsh(i, salt + 3));
        double k = age / life;
        puff(X_, fmax(Y_, r * 0.3), z + (hsh(i, salt + 4) - 0.5) * fmin(R0, 0.15 * z), r, dens * sq(1 - k) * sstep(0, 0.25, age),
             hsh(i, salt + 5) * 50, age, amber);
    }
}

/* grains and clods thrown out ballistically */
static void ejecta(double t, double X, double Y0, double z, int n, double v0, double spread, double size, uint32_t salt, double amber) {
    for (int i = 0; i < n; i++) {
        double tau = t - hsh(i, salt) * 0.5;
        if (tau < 0 || tau > 9) continue;
        double ang = (hsh(i, salt + 1) - 0.5) * spread, v = v0 * (0.35 + 0.65 * hsh(i, salt + 2));
        double vx = v * sin(ang), vy = v * cos(ang), x0 = X + (hsh(i, salt + 3) - 0.5) * size * 30;
        double Y = Y0 + vy * tau - 4.9 * tau * tau;
        if (Y < 0) continue;
        double Yp = Y0 + vy * (tau - 0.07) - 4.9 * sq(tau - 0.07);
        double ax = sx_of(x0 + vx * tau, z), ay = sy_of(Y, z), bx = sx_of(x0 + vx * (tau - 0.07), z), by = sy_of(Yp, z);
        double dx = ax - bx, dy = ay - by, len = hypot(dx, dy);
        double rad = fmax(size * (0.25 + 0.5 * hsh(i, salt + 4)) * F / z, 0.6);
        double am = hsh(i, salt + 5) < 0.35 + 0.65 * (amber >= 1) ? amber : 0;
        spr(T_GRAIN, (ax + bx) / 2, (ay + by) / 2, atan2(-dy, dx), 1, -len / 2 - rad - 1, len / 2 + rad + 1, -rad - 1, rad + 1, z,
            len / 2, rad, 0.85 * (1 - sstep(6, 9, tau)), am, 0);
    }
}

/* a sheet of sand falling from (X, Ytop) to the ground, hw metres wide */
static void curtain(double X, double Yt, double z, double hw, double dens, double seed, double amber) {
    if (dens < 0.01 || Yt <= 0) return;
    double ax = sx_of(X, z), ay = sy_of(Yt, z), len = sy_of(0, z) - ay, h = hw * F / z;
    spr(T_CURTAIN, ax, ay, 0, 1, -h, h, -len, 0, z - 0.5, h, len, dens, seed, amber);
}

static void walker(double X, double Y, double z, double ph, double run, int face, int staff) {
    double s = F / z;
    spr(T_WALKER, sx_of(X, z), sy_of(Y, z), 0, s, -0.9, 0.9, -0.1, 2.2, z, ph, run, face, staff, 0);
}
static void crawler(double X, double Y, double z, double tilt, double lights, int face, double t) {
    spr(T_CRAWLER, sx_of(X, z), sy_of(Y, z), tilt, F / z, -19.5, 16, -0.6, 19, z, lights, face, t, 0, 0);
}
static void flyer(double X, double Y, double z, double pitch, int face, double t) {
    spr(T_FLYER, sx_of(X, z), sy_of(Y, z), pitch, F / z, -11.5, 9, -7, 6, z, t * 31, face, 0, 0, 0);
}
static void mound(double X, double z, double Hh, double Wd, double trail, double glow) {
    if (Hh < 0.3) return;
    double ext = 3 * Wd + fabs(trail) * 3;
    spr(T_MOUND, sx_of(X, z), sy_of(0, z), 0, F / z, -ext, ext, -2.5, Hh * 1.25 + 3, z, Hh, Wd, trail, glow, 0);
}
/* an ellipse lying on the ground: a pit or a patch, radius in m */
static void ground_disc(int ty, double X, double z, double Rm, double amt, double t) {
    double rx = Rm * F / z, ry = fmax(Rm * F * CAM / (z * z) * 1.4, 1.5);
    spr(ty, sx_of(X, z), sy_of(0, z), 0, 1, -rx, rx, -ry, ry, z + 0.2 * Rm, rx, ry, amt, t, 0);
}

/* plume of dust left by something moving or working: emitted at intervals
 * from src(te), drifting with the wind */
typedef void (*Emitter)(double te, double *X, double *Y);
static void trail(Emitter src, double t, double t0, double t1, double dt, double life, double z, double r0, double grow,
                  double rise, double wind, double dens, uint32_t salt, double amber) {
    double first = fmax(t0, t - life);
    for (double te = ceil(first / dt) * dt; te <= fmin(t, t1); te += dt) {
        double age = t - te, X, Y;
        long id = (long)floor(te / dt + 0.5);
        src(te, &X, &Y);
        double k = age / life;
        puff(X + wind * age + (hsh(id, salt) - 0.5) * r0, Y + rise * age, z + (hsh(id, salt + 1) - 0.5) * 10, r0 + grow * age,
             dens * sq(1 - k) * sstep(0, 0.4, age), hsh(id, salt + 2) * 40, age, amber);
    }
}

/* ---- the worm's poses ---- */
/* rearing out of the ground at X0: 'out' metres of it above the sand, its
 * base leaning by 'lean', the top 'bend' further over, swaying */
static void worm_rear(Worm *Wm, double X0, double z, double R, double out, double lean, double bend, double t) {
    Wm->on = out > 0.5;
    Wm->z = z; Wm->R = R;
    double len = out + 2.0 * R;
    double ds = len / NSP;
    double x = X0, y = -2.0 * R;
    double px[NSP + 1], py[NSP + 1];
    for (int i = 0; i <= NSP; i++) {
        double sg = i * ds;            /* from the base */
        double u = sg / len;
        double th = lean + bend * sstep(0.25, 1.0, u) + 0.07 * sin(t * 0.7 + u * 3.0) * u + 0.03 * sin(t * 1.3 + u * 7.0);
        px[i] = x; py[i] = y;
        x += sin(th) * ds; y += cos(th) * ds;
    }
    for (int i = 0; i <= NSP; i++) { Wm->X[i] = px[NSP - i]; Wm->Y[i] = py[NSP - i]; }
}

/* surfacing along an arc of radius Ra over (Xc, -0.35 Ra), the head s_h
 * metres along it, the body blen long behind it, travelling dir */
static double arc_ag(void) { return acos(0.35); }
static double arc_len(double Ra) { return 2 * arc_ag() * Ra; }
static void arc_pt(double Xc, double Ra, int dir, double s, double *x, double *y) {
    double ag = arc_ag(), cy = -0.35 * Ra, a, over = 0;
    if (s < 0) { a = -ag; over = s; }
    else if (s > arc_len(Ra)) { a = ag; over = s - arc_len(Ra); }
    else a = -ag + s / Ra;
    *x = Xc + dir * (Ra * sin(a) + cos(a) * over);
    *y = cy + Ra * cos(a) - sin(a) * over;
}
static void worm_arc(Worm *Wm, double Xc, double z, double R, double Ra, int dir, double s_h, double blen) {
    Wm->z = z; Wm->R = R;
    int any = 0;
    for (int i = 0; i <= NSP; i++) {
        double s = s_h - blen * i / NSP;
        arc_pt(Xc, Ra, dir, s, &Wm->X[i], &Wm->Y[i]);
        if (Wm->Y[i] > -R) any = 1;
    }
    Wm->on = any;
}

/* churned sand heaped about the base, and dust boiling off it */
static double skirt_X, skirt_z, skirt_R, skirt_t0, skirt_t1;
static void em_skirt(double te, double *X, double *Y) { *X = skirt_X + (hsh((long)(te * 10), 77) - 0.5) * skirt_R * 2.4; *Y = skirt_R * 0.15; }
static void skirt(double u, double X, double z, double R, double t0, double t1) {
    skirt_X = X; skirt_z = z; skirt_R = R; skirt_t0 = t0; skirt_t1 = t1;
    if (u < t0 || u > t1 + 8) return;
    double g = sstep(t0, t0 + 0.6, u) * (1 - sstep(t1 - 1, t1 + 2, u));
    if (g > 0.01) mound(X, z - R * 1.1, R * 0.38 * g, R * 1.25, 0, 0);
    trail(em_skirt, u, t0, t1, 0.45, 3.5, z - R * 0.6, R * 0.32, R * 0.07, 3, 3, 0.5, 91, 0.1);
}

/* ---- per-act choreography ---- */
static const Act *cur;
static double U_;                      /* act time, for the emitters */
static double crawlerX, crawlerZ, wakeX0, wakeX1, wakeT0, wakeT1, wakeZ;

static double wake_at(double te) { return mixd(wakeX0, wakeX1, sstep(0, 1, (te - wakeT0) / (wakeT1 - wakeT0)) * 0.3 + 0.7 * clamp01((te - wakeT0) / (wakeT1 - wakeT0))); }
static void em_wake(double te, double *X, double *Y) { *X = wake_at(te); *Y = 4; }
static void em_crawler(double te, double *X, double *Y) { (void)te; *X = crawlerX - 16 * cur->dir; *Y = 3; }

/* a wake: the swell running from X0 to X1 between t0 and t1 */
static void wake(double u, double t0, double t1, double X0, double X1, double z, double Hh, uint32_t salt) {
    wakeX0 = X0; wakeX1 = X1; wakeT0 = t0; wakeT1 = t1; wakeZ = z;
    if (u < t0 || u > t1 + 10) return;
    int dir = X1 > X0 ? 1 : -1;
    if (u <= t1) {
        double hx = wake_at(u), g = sstep(t0, t0 + 2.5, u) * (1 - sstep(t1 - 0.4, t1, u));
        mound(hx, z, Hh * g, Hh * 2.6, -dir * Hh * 7, 0);
    }
    trail(em_wake, u, t0, t1, 0.22, 7, z + 4, Hh * 1.4, Hh * 1.2, 2.5, -dir * 3, 0.55, salt, 0.08);
}

/* walkers along a line: who is where */
static void walkers_line(double u, int n, double Xlead, int dir, double gap, double z, int on_layer, double run, double freeze, uint32_t salt) {
    for (int i = 0; i < n; i++) {
        double X = Xlead - dir * gap * i * (0.85 + 0.3 * hsh(i, salt));
        double Y = on_layer >= 0 ? layer_h(cur, on_layer, X, NULL) : 0;
        /* no rhythm: each its own cadence, with hitches */
        double ph = u * (5.2 + 2.0 * run) * (0.9 + 0.2 * hsh(i, salt + 1)) + 0.6 * sin(u * 0.7 + i) + 3.0 * hsh(i, salt + 2);
        ph = mixd(ph, 0.0, freeze);
        walker(X, Y, z, ph, run, dir, hsh(i, salt + 3) < 0.4);
    }
}

static double crest_X(const Act *A, int k, double sxa, double sxb) {
    double best = -1e9, bx = 0;
    for (int i = 0; i <= 200; i++) {
        double X = X_at(mixd(sxa, sxb, i / 200.0), LZ[k], A->cam0), h = layer_h(A, k, X, NULL);
        if (h > best) { best = h; bx = X; }
    }
    return bx;
}

static void base_fx(void) {
    memset(&fx, 0, sizeof fx);
    fx.plume = 0.55; fx.vein = 0.6; fx.glit = 0.35; fx.vig = 0.42;
    fx.sundisc = 1;
}

static double spike(double u, double at, double rise_, double fall) {
    if (u < at - rise_) return 0;
    if (u < at) return sstep(at - rise_, at, u);
    return exp(-(u - at) / fall);
}

static void stage_harvest(const Act *A, double u, double t) {
    int d = A->dir;
    double zc = 320, zw = 330, R = 38;
    double Xc = X_at(A->fx, zc, A->cam0);
    crawlerX = Xc; crawlerZ = zc;
    double fw = fr[2];
    double Xfrom = Xc - d * (1.25 * fw) * zw / F;
    /* the amber patch, worked; gone with the crawler */
    ground_disc(T_PATCH, Xc - d * 6, zc + 6, 85, 0.9 * (1 - sstep(37, 42, u)), A->r[0] * 10);
    if (u < 37.6) {
        double sink = sstep(33, 37.6, u);
        crawler(Xc + 2.0 * sin(u * 0.15), -7 * sink, zc, d * 0.35 * sink, 0.3, d, u);
        trail(em_crawler, u, -10, 37.4, 0.45, 10, zc + 8, 7, 3.5, 2.2, 2.5, 0.42, 501, 0.75);
    } else
        trail(em_crawler, u, -10, 37.4, 0.45, 10, zc + 8, 7, 3.5, 2.2, 2.5, 0.42, 501, 0.75);
    /* the carrier hovering over it, then away */
    {
        double fy = 55 + 4 * sin(u * 0.8), fxp = Xc - d * 35 + 6 * sin(u * 0.3), pitch = 0;
        if (u > 29) { double e = u - 29; fxp -= d * e * e * 1.6; fy += e * 9; pitch = d * 0.25 * sstep(29, 31, u); }
        flyer(fxp, fy, zc - 25, pitch, -d, t);
    }
    wake(u, 18, 36.8, Xfrom, Xc, zw + 10, 7, 601);
    fx.shake = 0.0012 * sstep(22, 36, u) * (1 - sstep(37, 40, u)) + 0.006 * spike(u, 37.4, 0.3, 1.3) + 0.003 * spike(u, 48.5, 0.5, 1.5);
    fx.plume = 0.55 + 0.8 * sstep(24, 36, u);
    if (u > 31 && u < 70) ground_disc(T_PIT, Xc, zc + 2, 60 * sstep(31, 37.4, u) * (1 - 0.6 * sstep(50, 68, u)), 1 - sstep(48, 70, u), u);
    if (u >= 37 && u < 50) {
        double out = 175 * ease_out((u - 37) / 2.6) * (1 - ease_in((u - 45.2) / 4.3));
        double bend = d * (0.35 + 0.45 * sstep(39, 43, u));
        worm_rear(&worm, Xc, zw, R, out, -d * 0.16, bend, t);
        worm.open = 0.25 + 0.75 * sstep(37.0, 38.2, u) - 0.75 * sstep(41.5, 44.5, u);
        worm.face = 0.3 + 0.4 * sstep(39.2, 42.5, u) - 0.3 * sstep(44.5, 47, u);
        worm.pour = sstep(37, 37.8, u) * (1 - sstep(42, 47, u)) + 0.5 * sstep(45, 46, u);
        worm.glow = 0.7 + 0.25 * sin(u * 2.1);
        worm.mX = Xc; worm.mW = 90; worm.mH = 14 * sstep(36.5, 37.5, u);
    }
    burst(u - 37.1, Xc, zw - 15, 18, 26, 75, 140, 70, 13, 0.75, 610, 0.25);
    skirt(u, Xc, zw, R, 37.1, 48.8);
    ejecta(u - 37.2, Xc, 20, zw - 10, 80, 48, 1.7, 1.0, 620, 0.6);
    burst(u - 48.3, Xc, zw - 15, 12, 30, 70, 110, 40, 12, 0.6, 630, 0.15);
    if (u > 37.4 && u < 47) {
        double hx = worm.X[0], hy = worm.Y[0];
        curtain(hx, hy - R * 0.4, zw - 3, R * 0.9, 0.6 * (1 - sstep(42, 47, u)), 3.3, 0.15);
    }
    if (u > 37.4) { fx.gust_x = sx_of(Xc, zw); fx.gust_y = sy_of(0, zw); fx.gust_r = L * (0.05 + 0.25 * (u - 37.4)); fx.gust_a = 0.5 * exp(-(u - 37.4) / 3.5); }
    fx.glx = sx_of(Xc, zw); fx.gly = sy_of(60, zw); fx.glr = L * 0.25; fx.gla = 0.6 * spike(u, 38, 0.5, 6);
}

static double rockX, rockZ, rockHW;
static double rock_h(double X) {
    double u = (X - rockX) / rockHW * 0.5 + 0.5;
    if (u <= 0 || u >= 1) return 0;
    double f = u * 23;
    int i = (int)f;
    if (i > 22) i = 22;
    return rock_prof[i] + (rock_prof[i + 1] - rock_prof[i]) * (f - i);
}

static void stage_refuge(const Act *A, double u, double t) {
    int d = A->dir;
    rockZ = 225; rockHW = 48;
    rockX = X_at(A->fx, rockZ, A->cam0) + d * 25;
    for (int i = 0; i < 24; i++) {
        double v = i / 23.0 * 2 - 1;
        double h = 27 * sstep(-1.0, -0.78, v) * (1 - sstep(0.55, 0.95, v)) + 7 * (hsh(i, 701) - 0.5) * (fabs(v) < 0.8) +
                   14 * sstep(-0.45, -0.25, v) * (1 - sstep(0.05, 0.3, v)) + 6 * sstep(0.2, 0.3, v) * (1 - sstep(0.35, 0.5, v));
        rock_prof[i] = (float)fmax(h, 0);
    }
    rock_prof[0] = rock_prof[23] = 0;
    spr(T_ROCK, sx_of(rockX, rockZ), sy_of(0, rockZ), 0, F / rockZ, -rockHW, rockHW, -4, 50, rockZ, rockHW, 0, 0, 0, 0);
    /* the walkers: a file crossing; the run; up onto the rock */
    double edge = rockX - d * rockHW * 0.85;
    double X0 = edge - d * 70;
    double dist = 1.3 * fmin(u, 17) + 5.4 * fmax(u - 17, 0);
    for (int i = 0; i < 9; i++) {
        double off = i * 4.6 * (0.85 + 0.3 * hsh(i, 711));
        double target = 70 + rockHW * 0.85 + 28 - 5.2 * i;
        double di = fmin(dist - off, target);
        double X = X0 + d * di;
        int arrived = dist - off >= target;
        double run = u > 17 && !arrived ? 1 : 0;
        double ph = arrived ? 0.4 : (u * (5.0 + 3.5 * run) * (0.9 + 0.2 * hsh(i, 712)) + 3 * hsh(i, 713));
        double Y = rock_h(X);
        walker(X, Y, rockZ - 3, ph, run, d, hsh(i, 714) < 0.4);
    }
    double zw = 300, R = 34;
    double Xw = edge - d * 30;
    double Xww = X_at(sx_of(Xw, rockZ), zw, camX);   /* the same sightline, further back */
    wake(u, 14, 33.6, Xww - d * 620, Xww, zw - 10, 6, 721);
    fx.shake = 0.0011 * sstep(18, 33, u) * (1 - sstep(34, 38, u)) + 0.005 * spike(u, 33.8, 0.3, 1.2) + 0.002 * spike(u, 49, 0.5, 1.5);
    fx.plume = 0.6 + 0.6 * sstep(20, 33, u);
    if (u >= 33.4 && u < 51) {
        double out = 235 * ease_out((u - 33.4) / 3.2) * (1 - ease_in((u - 45) / 5.5));
        double bend = d * 2.05 * sstep(35.3, 39.6, u) * (1 - sstep(44, 48, u));
        worm_rear(&worm, Xww, zw, R, out, -d * 0.1, bend, t);
        worm.open = 0.15 + 0.85 * sstep(37.5, 40.0, u) - 0.6 * sstep(44.0, 46.5, u);
        worm.face = 0.75 * sstep(36.0, 39.6, u) * (1 - sstep(44, 47, u)) + 0.1;
        worm.pour = sstep(33.4, 34, u) * (1 - sstep(39, 45, u)) * 0.9;
        worm.glow = 0.75 + 0.25 * sin(u * 1.7);
        worm.mX = Xww; worm.mW = 80; worm.mH = 12 * sstep(33, 34, u);
    }
    burst(u - 33.5, Xww, zw - 12, 16, 24, 70, 120, 60, 13, 0.7, 730, 0.2);
    skirt(u, Xww, zw, R, 33.5, 49.5);
    ejecta(u - 33.6, Xww, 15, zw - 10, 70, 44, 1.6, 1.0, 740, 0.5);
    burst(u - 49.4, Xww, zw - 12, 10, 26, 60, 90, 35, 11, 0.55, 750, 0.1);
    if (u > 34 && u < 46) curtain(worm.X[0], worm.Y[0] - R * 0.3, zw - 3, R * 0.8, 0.55 * (1 - sstep(41, 46, u)), 1.7, 0.1);
    fx.glx = sx_of(Xww, zw); fx.gly = sy_of(80, zw); fx.glr = L * 0.25; fx.gla = 0.5 * spike(u, 34.5, 0.5, 6);
}

static void stage_rise(const Act *A, double u, double t) {
    int d = A->dir;
    double zw = A->fz, R = 41;
    double Xw = X_at(A->fx, zw, A->cam0);
    /* far off, a crawler at work; a file of walkers on the ridge, who stop */
    crawlerX = X_at(A->fx + d * fr[2] * 0.36, 1150, A->cam0); crawlerZ = 1150;
    crawler(crawlerX, 0, 1150, 0, 0.3, -d, u);
    trail(em_crawler, u, -12, 80, 0.6, 12, 1160, 9, 4, 2.4, 3, 0.4, 801, 0.75);
    {
        double Xl = crest_X(A, 7, A->fx + A->fgside * fr[2] * 0.15, A->fx + A->fgside * fr[2] * 0.40) - A->fgside * 24 + A->fgside * 1.1 * fmin(u, 19);
        walkers_line(u, 5, Xl, A->fgside, 2.6, 93, 7, 0, sstep(18.5, 20, fmin(u, 20)), 811);
    }
    /* a flyer crossing in front of it at the height of it */
    if (u > 22) {
        double e = u - 37;
        flyer(Xw + d * (78 + 8 * e), 58 + 4 * sin(e * 0.6) + 1.5 * e, 150, d * 0.05, d, t);
    }
    fx.shake = 0.0013 * sstep(16, 27, u) * (1 - sstep(28, 31, u)) + 0.006 * spike(u, 27.9, 0.4, 1.4) + 0.0016 * sstep(31, 33, u) * (1 - sstep(38, 41, u)) +
               0.010 * spike(u, 43.6, 0.25, 1.6);
    fx.plume = 0.55 + 1.2 * sstep(16, 26, u) * (1 - sstep(50, 60, u));
    double mh = 26 * sstep(21, 27.6, u) * (1 - sstep(28.2, 29, u));
    if (mh > 0.2) mound(Xw, zw + 2, mh, 50, 0, 0);
    if (u >= 27.7 && u < 50) {
        double rise = ease_out((u - 27.7) / 3.4);
        double out = 150 * rise;
        out = mixd(out, 78, sstep(31.0, 34.5, u));
        double fall = ease_in((u - 41.0) / 2.6);
        out *= 1 - fall;
        double lean = d * 0.4 * fall;
        worm_rear(&worm, Xw, zw, R, out, lean, -d * 0.22 * sstep(30, 34, u), t);
        worm.face = 0.32 + 0.66 * sstep(31.2, 34.8, u) - 0.5 * sstep(41, 43, u);
        worm.open = 0.45 + 0.55 * sstep(31.5, 35.0, u) * (1 - 0.06 * sin(u * 1.4)) - 0.5 * sstep(41.5, 43.5, u);
        worm.pour = sstep(27.7, 28.4, u) * (1 - sstep(33, 41, u)) * 1.0 + 0.6 * sstep(41, 42, u);
        worm.glow = 0.8 + 0.2 * sin(u * 1.9);
        worm.mX = Xw; worm.mW = 75; worm.mH = mixd(26, 10, sstep(28, 33, u)) * sstep(27, 28, u);
    }
    burst(u - 27.8, Xw, zw - 12, 16, 28, 70, 170, 55, 10, 0.6, 820, 0.3);
    skirt(u, Xw, zw, R, 27.8, 43.3);
    ejecta(u - 27.9, Xw, 25, zw - 10, 90, 55, 1.6, 1.0, 830, 0.6);
    if (u > 28 && u < 44) {
        double hx = worm.X[0], hy = worm.Y[0];
        double dn = 0.65 * (1 - sstep(38, 43, u)) * sstep(28, 29, u);
        curtain(hx - R * 0.55, hy - R * 0.2, zw - 3, R * 0.45, dn, 2.1, 0.2);
        curtain(hx + R * 0.55, hy - R * 0.4, zw - 3, R * 0.4, dn * 0.8, 5.4, 0.2);
        curtain(worm.X[NSP / 2], worm.Y[NSP / 2], zw - 4, R * 0.7, dn * 0.5, 8.4, 0.1);
    }
    /* the fall: an explosion of sand, and a gust that rolls over everything */
    burst(u - 43.5, Xw + d * 20, zw - 20, 14, 30, 75, 330, 80, 12, 0.8, 840, 0.15);
    ejecta(u - 43.6, Xw + d * 20, 20, zw - 15, 120, 62, 1.9, 1.3, 850, 0.5);
    if (u > 43.6) {
        double e = u - 43.6;
        fx.gust_x = sx_of(Xw, zw); fx.gust_y = sy_of(0, zw);
        fx.gust_r = L * (0.1 + 0.55 * e);
        fx.gust_a = 0.75 * exp(-e / 3.5);
        fx.fogk = 1 / 3500.0 * exp(-e / 12);
    }
    fx.glx = sx_of(Xw, zw); fx.gly = sy_of(60, zw); fx.glr = L * 0.35;
    fx.gla = 0.5 * spike(u, 29, 0.6, 7) + 0.4 * spike(u, 44, 0.4, 7);
}

static void stage_horizon(const Act *A, double u, double t) {
    int d = A->dir;
    double zw = A->fz, R = 72, Ra = 520;
    double Xc = X_at(A->fx, zw, A->cam0);
    /* figures on the near ridge, watching */
    double Xl = crest_X(A, 7, A->fx + A->fgside * fr[2] * 0.12, A->fx + A->fgside * fr[2] * 0.48) - d * 14 + d * 1.2 * fmin(u, 15) + d * 1.2 * fmax(u - 48, 0);
    walkers_line(u, 5, Xl, d, 2.6, 93, 7, 0, sstep(14.5, 16, u) * (1 - sstep(47, 48.5, u)), 911);
    /* a dust line coming along the horizon, then the arc */
    double vel = 62, blen = 1350;
    double alen = arc_len(Ra);
    double s_h = (u - 21) * vel;
    wake(u, 9, 21.5, Xc - d * 1600, Xc - d * Ra * 0.95, zw + 20, 14, 921);
    fx.shake = 0.0005 * sstep(18, 24, u) * (1 - sstep(40, 50, u));
    if (u > 20 && s_h - blen < alen + 200) {
        worm_arc(&worm, Xc, zw, R, Ra, d, s_h, blen);
        worm.open = 0.55; worm.face = 0.15; worm.glow = 0.5; worm.pour = 0.8;
        worm.mX = 0; worm.mW = 1; worm.mH = 0;
        for (int i = 6; i < NSP; i += 9)
            if (worm.Y[i] > R) curtain(worm.X[i], worm.Y[i] - R * 0.7, zw - 5, R * 0.6, 0.5, i * 0.7, 0.05);
    }
    double Xin = Xc - d * Ra * 0.94, Xout = Xc + d * Ra * 0.94;
    burst(u - 21, Xin, zw - 20, 14, 50, 140, 220, 90, 18, 0.7, 930, 0.15);
    burst(u - (21 + alen / vel), Xout, zw - 20, 16, 50, 150, 260, 110, 20, 0.75, 940, 0.15);
    burst(u - (21 + (alen + blen) / vel), Xout, zw - 20, 10, 40, 110, 200, 70, 16, 0.5, 950, 0.1);
    fx.plume = 0.7;
    fx.glit = 0.55;
    (void)t;
}

static void stage_storm(const Act *A, double u, double t) {
    int d = A->dir;
    /* the crawler, working on with its lights on */
    crawlerX = X_at(A->fx - d * fr[2] * 0.22, 600, A->cam0); crawlerZ = 600;
    crawler(crawlerX, 0, 600, 0, 0.4 + 0.6 * sstep(15, 30, u), d, u);
    trail(em_crawler, u, -12, 90, 0.6, 10, 610, 8, 4, 2.0, 2.5, 0.35, 1001, 0.6);
    double zs = 25000 * exp(-0.153 * (u - 4));
    double in = sstep(36, 40.5, u) * (1 - sstep(57, 68, u));
    fx.light = 1 - 0.62 * sstep(14, 39, u) * (1 - 0.6 * sstep(56, 70, u));
    if (u > 4 && u < 40) { fx.storm_z = fmax(zs, 30); fx.storm_h = 3200; fx.storm_a = 0.97 * sstep(4, 8, u); }
    fx.storm_in = 0.6 * in;
    fx.fogk = in * (1 / 260.0);
    fx.plume = 0.6 + 1.8 * sstep(20, 38, u) * (1 - sstep(58, 68, u));
    fx.shake = 0.0006 * in;
    fx.sundisc = 1;
    /* lightning; at 49 it shows a worm rearing in the murk */
    static const double fl[6] = {24.0, 31.5, 43.2, 46.6, 49.0, 54.8};
    for (int i = 0; i < 6; i++) {
        double e = u - fl[i];
        if (e < 0 || e > 1.2) continue;
        double f = exp(-e / 0.16) * (0.6 + 0.4 * (sin(e * 70) > 0)) + 0.6 * exp(-sq((e - 0.25) / 0.05));
        if (i == 4) f = fmax(f, exp(-e / 0.5) * 0.9);
        if (f > fx.flash) {
            fx.flash = f;
            fx.flx = i == 4 ? A->fx + d * fr[2] * 0.10 : A->fx + fr[2] * (hsh(i, 1010) - 0.5) * 0.7;
            fx.fly = YH - L * (i == 4 ? 0.42 : 0.25 + 0.2 * hsh(i, 1011));
            fx.bolt_x = fx.flx + (i == 4 ? d * fr[2] * 0.12 : 0); fx.bolt_y = fx.fly - L * 0.2; fx.bolt_len = L * (0.35 + 0.2 * hsh(i, 1012)); fx.bolt_seed = i + 1;
        }
    }
    if (u >= 45.5 && u < 58) {
        double zw = 420, R = 46;
        double Xw = X_at(A->fx - d * fr[2] * 0.08, zw, A->cam0);
        double out = 260 * ease_out((u - 45.5) / 3.5) * (1 - ease_in((u - 53) / 4.5));
        worm_rear(&worm, Xw, zw, R, out, -d * 0.1, d * 1.2 * sstep(46.5, 49.5, u), t);
        worm.open = 0.9; worm.face = 0.7; worm.glow = 0.9; worm.pour = 0.6;
        worm.sil = 0.0;
        worm.mX = Xw; worm.mW = 1; worm.mH = 0;
    }
    if (fx.storm_in > 0 && fx.flash > 0) fx.storm_in *= 1 - 0.5 * fx.flash;
}

static void stage_night(const Act *A, double u, double t) {
    int d = A->dir;
    fx.night = 1; fx.sundisc = 0;
    fx.moon[0][0] = A->sunx; fx.moon[0][1] = A->suny; fx.moon[0][2] = A->sunr; fx.moon[0][3] = -d * 0.9;
    fx.moon[1][0] = A->fx - d * fr[2] * 0.14; fx.moon[1][1] = YH - L * 0.56; fx.moon[1][2] = L * 0.022; fx.moon[1][3] = d * 0.4;
    double Xl = crest_X(A, 7, A->fx + A->fgside * fr[2] * 0.12, A->fx + A->fgside * fr[2] * 0.48) - d * 22 + d * 1.1 * fmin(u, 24) + d * 1.1 * fmax(u - 44, 0);
    walkers_line(u, 6, Xl, d, 2.8, 93, 7, 0, sstep(23, 25, u) * (1 - sstep(43, 45, u)), 1101);
    double zw = 1500, R = 64, Ra = 430;
    double Xc = X_at(A->fx, zw, A->cam0);
    wake(u, 16, 29.5, Xc - d * 1500, Xc - d * Ra * 0.9, zw + 30, 13, 1121);
    fx.shake = 0.0007 * sstep(20, 26, u) * (1 - sstep(38, 46, u));
    double vel = 58, blen = 1250, s_h = (u - 29) * vel;
    double alen = arc_len(Ra);
    if (u > 28.5 && s_h - blen < alen + 200) {
        worm_arc(&worm, Xc, zw, R, Ra, d, s_h, blen);
        worm.open = 0.5; worm.face = 0.15; worm.glow = 0.9; worm.pour = 0.7;
        worm.mX = 0; worm.mW = 1; worm.mH = 0;
        for (int i = 6; i < NSP; i += 10)
            if (worm.Y[i] > R) curtain(worm.X[i], worm.Y[i] - R * 0.7, zw - 5, R * 0.6, 0.4, i * 0.7, 0);
    }
    burst(u - 29, Xc - d * Ra * 0.94, zw - 20, 12, 50, 130, 200, 80, 16, 0.6, 1130, 0);
    burst(u - (29 + alen / vel), Xc + d * Ra * 0.94, zw - 20, 14, 50, 140, 240, 100, 18, 0.65, 1140, 0);
    fx.plume = 0.35; fx.vein = 0.25; fx.glit = 0.12;
    (void)t;
}

static double bloomX, bloomZ;

static void stage_bloom(const Act *A, double u, double t) {
    int d = A->dir;
    bloomZ = A->fz;
    bloomX = X_at(A->fx, bloomZ, A->cam0);
    ground_disc(T_PATCH, bloomX, bloomZ + 4, 120 + 60 * sstep(27, 40, u), 0.5 + 0.5 * sstep(14, 26, u), 2.2);
    /* far crawler for scale */
    crawlerX = X_at(A->fx - d * fr[2] * 0.33, 900, A->cam0); crawlerZ = 900;
    crawler(crawlerX, 0, 900, 0, 0.2, d, u);
    trail(em_crawler, u, -12, 90, 0.6, 12, 910, 8, 4, 2.0, 2.5, 0.35, 1201, 0.7);
    fx.shake = 0.0009 * sstep(12, 26, u) * (1 - sstep(27, 29, u)) + 0.005 * spike(u, 27.2, 0.3, 1.5);
    double mh = 22 * sstep(15, 27, u) * (1 - sstep(27.2, 28.5, u));
    if (mh > 0.2) mound(bloomX, bloomZ, mh, 70, 0, sstep(18, 26, u));
    /* seeping before it bursts */
    if (u > 19 && u < 27.5)
        for (int i = 0; i < 6; i++) {
            double age = fmod(u - 19 + i * 1.3, 8.0);
            puff(bloomX + (hsh(i, 1210) - 0.5) * 90 + age * 2, mh * 0.8 + age * 4, bloomZ - 2, 8 + age * 3, 0.35 * (1 - age / 8) * sstep(19, 21, u), i * 3.3, age, 1);
        }
    /* the geyser: a column of glowing amber dust, then a hanging cloud that settles */
    double dt = 0.16;
    for (double te = 27.2; te < 34.5 && te <= u; te += dt) {
        long id = (long)((te - 27.2) / dt + 0.5);
        double age = u - te;
        if (age > 44) continue;
        double hgt = (140 + 150 * hsh(id, 1220)) * (1 - exp(-age / 2.0)) - fmax(age - 9, 0) * 2.4;
        double X = bloomX + (hsh(id, 1221) - 0.5) * 40 * (1 + age * 0.5) + age * 3.0 * d;
        double r = 14 + 48 * (1 - exp(-age / 3.5)) + age * 1.1;
        double dens = 0.48 * sq(1 - age / 44) * sstep(0, 0.3, age);
        puff(X, fmax(hgt, r * 0.4), bloomZ - 5 + (hsh(id, 1222) - 0.5) * 30, r, dens, hsh(id, 1223) * 50, age, 0.85 + 0.15 * hsh(id, 1224));
    }
    burst(u - 27.2, bloomX, bloomZ - 10, 16, 30, 80, 160, 40, 14, 0.55, 1230, 0.5);
    ejecta(u - 27.3, bloomX, 15, bloomZ - 8, 160, 60, 1.0, 0.6, 1240, 1.0);
    if (u > 27.2) {
        double e = u - 27.2;
        fx.glx = sx_of(bloomX, bloomZ); fx.gly = sy_of(150, bloomZ) + L * 0.002 * e * e; fx.glr = L * (0.18 + 0.03 * e);
        fx.gla = 0.8 * sstep(0, 1.5, e) * exp(-e / 18);
        fx.gust_x = sx_of(bloomX, bloomZ); fx.gust_y = sy_of(0, bloomZ); fx.gust_r = L * (0.05 + 0.18 * e); fx.gust_a = 0.35 * exp(-e / 4);
    }
    fx.glit = 0.4 + 0.5 * sstep(27, 32, u) * (1 - sstep(50, 70, u));
    fx.vein = 0.8;
    /* drawn to it: a wake, then a worm back arcing in the distance */
    double zw = 1100, Ra = 300, R = 46;
    double Xc = X_at(A->fx + d * fr[2] * 0.30, zw, A->cam0);
    wake(u, 42, 53, Xc + d * 1500, Xc + d * Ra * 0.9, zw + 20, 10, 1250);
    double vel = 55, blen = 1000, s_h = (u - 53) * vel;
    double alen = arc_len(Ra);
    if (u > 52.5 && s_h - blen < alen + 200) {
        worm_arc(&worm, Xc, zw, R, Ra, -d, s_h, blen);
        worm.open = 0.7; worm.face = 0.2; worm.glow = 0.8; worm.pour = 0.8;
        worm.mX = 0; worm.mW = 1; worm.mH = 0;
    }
    burst(u - 53, Xc + d * Ra * 0.94, zw - 20, 12, 40, 110, 170, 70, 15, 0.6, 1260, 0.3);
    burst(u - (53 + alen / vel), Xc - d * Ra * 0.94, zw - 20, 12, 40, 110, 170, 70, 15, 0.6, 1270, 0.3);
    fx.shake += 0.0006 * sstep(46, 52, u) * (1 - sstep(56, 62, u));
    (void)t;
}

/* ---- assemble the frame ---- */
static void frame_build(double t_abs) {
    double t = t_abs < 0 ? 0 : t_abs;
    long n = act_index(&t);
    const Act *A = act_get(n);
    cur = A;
    double u = t - A->T0;
    U_ = u;
    nspr = 0;
    memset(&worm, 0, sizeof worm);
    base_fx();
    fx.light = 1;
    lt = presets[ev_light[A->kind]];
    camX = A->cam0 + A->camv * u;
    shx = shy = 0;
    switch (A->kind) {
    case E_HARVEST: stage_harvest(A, u, t); break;
    case E_REFUGE: stage_refuge(A, u, t); break;
    case E_RISE: stage_rise(A, u, t); break;
    case E_HORIZON: stage_horizon(A, u, t); break;
    case E_STORM: stage_storm(A, u, t); break;
    case E_NIGHT: stage_night(A, u, t); break;
    case E_BLOOM: stage_bloom(A, u, t); break;
    }
    /* the shake moves everything drawn from here on; the sprites already
     * built carry it too because sx_of/sy_of read shx/shy... so set it
     * first next frame: instead shift them now */
    double a = fx.shake * L;
    double sx = a * (0.6 * sin(t * 37.0) + 0.4 * sin(t * 23.0 + 1.0)), sy = a * (0.7 * sin(t * 41.0 + 2.0) + 0.3 * sin(t * 29.0));
    for (int i = 0; i < nspr; i++)
        for (int k = 0; k < 6; k++) { sprs[i].v[k * 12] += (float)sx; sprs[i].v[k * 12 + 1] += (float)sy; }
    shx = sx; shy = sy;
    if (fx.flash > 0) fx.flash = fmin(fx.flash, 1.2);
    /* the cut between acts: a dip through dark haze */
    fx.fade = fmax(1 - sstep(0, 2.2, u), sstep(A->dur - 2.2, A->dur, u)) * 0.92;
    /* the light, graded and dimmed */
    float *cols[12] = {lt.top, lt.hor, lt.sun, lt.haze, lt.sunhaze, lt.sandL, lt.sandD, lt.rim, lt.hide, lt.dust, lt.rock, lt.metal};
    for (int i = 0; i < 12; i++) grade(cols[i]);
    if (fx.light < 1) {
        float k = (float)fx.light;
        for (int i = 0; i < 12; i++) if (i != 2) for (int c = 0; c < 3; c++) cols[i][c] *= 0.35f + 0.65f * k;
        lt.suni *= k * k;
    }
    for (int c = 0; c < 3; c++) fx.fogc[c] = lt.haze[c] * 0.62f * (float)(0.65 + 0.35 * fx.light);
    if (fx.flash > 0) for (int c = 0; c < 3; c++) fx.fogc[c] += (float)(fx.flash * (c == 0 ? 0.12 : c == 1 ? 0.10 : 0.09));
    mesa_tint = 1;
}

/* ---- the land's lines: per column, for each layer, its crest (wall y),
 * how sunlit the slope is and how near a crest ---- */
static float *rowbuf;
static double *hbuf;
static int RW;
static double rx0, rdx;
#define ROWI(li, i) ((((li) / 4) * RW + (i)) * 4 + (li) % 4)

static void view_update(void) {
    if (span_me >= 0) {
        v_sc = W / span[span_me][2];
        v_ox = span[span_me][0]; v_oy = span[span_me][1];
    } else {
        v_sc = fmax(W / fr[2], H / fr[3]);
        v_ox = fr[0] + (fr[2] - W / v_sc) / 2; v_oy = fr[1] + (fr[3] - H / v_sc) / 2;
    }
}

static void rows_update(void) {
    /* padded so the blur below sees the same neighbours on every output */
    double pad = 0.18 * F * LLAM[NL - 1] / LZ[NL - 1] + 6 / v_sc;
    double x0 = v_ox - pad, x1 = v_ox + W / v_sc + pad, dx = 2.0 / v_sc;
    int n = (int)ceil((x1 - x0) / dx) + 2;
    if (n != RW) {
        RW = n;
        rowbuf = realloc(rowbuf, sizeof(float) * 32 * RW);
        hbuf = realloc(hbuf, sizeof(double) * RW * 3);
    }
    rx0 = x0; rdx = dx;
    double ce = cos(lt.elev), se = sin(lt.elev);
    for (int k = 0; k < NL; k++) {
        double z = LZ[k], dX = rdx * z / F;
        for (int i = 0; i < RW; i++) {
            double x = rx0 + i * rdx;
            double X = (x - CX) * z / F + camX, sh;
            hbuf[i] = layer_h(cur, k, X, &sh);
            hbuf[RW + i] = sh;
        }
        for (int i = 0; i < RW; i++) {
            double x = rx0 + i * rdx;
            int a = i > 0 ? i - 1 : 0, b = i < RW - 1 ? i + 1 : RW - 1;
            double slope = (hbuf[b] - hbuf[a]) / ((b - a) * dX);
            double side = tanh((cur->sunx - x) / (0.35 * L));
            double nl = (-slope * side * ce + se) / sqrt(1 + slope * slope);
            rowbuf[ROWI(k, i)] = (float)(YH + F * (CAM - hbuf[i]) / z);
            rowbuf[ROWI(8 + k, i)] = (float)clamp01(0.12 + 1.9 * (nl - 0.35 * se));
            rowbuf[ROWI(16 + k, i)] = (float)(k == 0 ? 0.3 * hbuf[RW + i] : pow(hbuf[RW + i], 5) * fmin(1, hbuf[i] / (0.4 * LA[k] + 1)));
            hbuf[2 * RW + i] = rowbuf[ROWI(8 + k, i)];
        }
        /* the same, blurred over a sixth of a wavelength, for further down the faces */
        int hwn = k ? (int)fmax(1, 0.16 * LLAM[k] / dX) : 1;
        double acc = 0;
        int lo = 0, hi = -1;
        for (int i = 0; i < RW; i++) {
            while (hi < i + hwn && hi < RW - 1) acc += hbuf[2 * RW + ++hi];
            while (lo < i - hwn) acc -= hbuf[2 * RW + lo++];
            rowbuf[ROWI(24 + k, i)] = (float)(acc / (hi - lo + 1));
        }
    }
    static int tex_w;
    glBindTexture(GL_TEXTURE_2D, tex_rows);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    if (tex_w != RW) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, RW, 8, 0, GL_RGBA, GL_FLOAT, rowbuf);
        tex_params(GL_CLAMP_TO_EDGE, 0);
        tex_w = RW;
    } else
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RW, 8, GL_RGBA, GL_FLOAT, rowbuf);
}

/* ---- the worm's strip ---- */
static float wormv[(NSP + 1) * 2 * 8];
static double maw_c[2], maw_t[2];
static void worm_strip(void) {
    double z = worm.z, s = F / z;
    double px[NSP + 1], py[NSP + 1], sg[NSP + 1];
    for (int i = 0; i <= NSP; i++) { px[i] = sx_of(worm.X[i], z); py[i] = sy_of(worm.Y[i], z); }
    sg[0] = 0;
    for (int i = 1; i <= NSP; i++) sg[i] = sg[i - 1] + hypot(worm.X[i] - worm.X[i - 1], worm.Y[i] - worm.Y[i - 1]);
    for (int i = 0; i <= NSP; i++) {
        int a = i > 0 ? i - 1 : 0, c = i < NSP ? i + 1 : NSP;
        double tx = px[a] - px[c], ty = py[a] - py[c], l = hypot(tx, ty);
        if (l < 1e-9) { tx = 0; ty = -1; l = 1; }
        tx /= l; ty /= l;
        double nx = -ty, ny = tx;
        double R = worm.R * (1 + 0.16 * sstep(1.5 * worm.R, 6 * worm.R, sg[i]) + 0.05 * sin(sg[i] / worm.R * 0.45 + 1.0) + 0.03 * sin(sg[i] / worm.R * 1.9) + 0.05 * exp(-sq((sg[i] - 0.4 * worm.R) / (0.3 * worm.R))) + 0.03 * sin(sg[i] / worm.R * 1.3 - U_ * 1.2));
        double rp = R * s * 1.03;
        for (int k = 0; k < 2; k++) {
            float *o = wormv + (i * 2 + k) * 8;
            double sgn = k ? 1 : -1;
            o[0] = (float)(px[i] + nx * rp * sgn); o[1] = (float)(py[i] + ny * rp * sgn);
            o[2] = (float)sg[i]; o[3] = (float)(1.03 * sgn); o[4] = (float)R; o[5] = (float)(R * s);
            o[6] = (float)nx; o[7] = (float)ny;
        }
        if (i == 0) { maw_c[0] = px[0]; maw_c[1] = py[0]; maw_t[0] = tx; maw_t[1] = ty; }
    }
}

/* ---- caption: the time, and how long the screen has been locked ---- */
static FT_Library ft;
static FT_Face face_cap;

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

typedef struct { short x, y, w, h, left, top, adv; } Glyph;
static Glyph glyphs[2][96];
static int cap_px[2], atlas_w = 1024, atlas_h;
static float *tv;
static int tvn, tvcap;
static char cap_line[2][96];

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
    cap_line[0][0] = 0;
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
    float halo[4] = {cap_halo[0], cap_halo[1], cap_halo[2], 0.16f}, col[4] = {cap_col[0], cap_col[1], cap_col[2], 0.90f};
    float r = fmaxf(1.0f, cap_px[0] / 12.0f);
    for (int k = 0; k < 12; k++) {
        float a = k * 0.5236f, dx = r * cosf(a), dy = r * sinf(a);
        for (int q = 1; q <= 2; q++) {
            text_at(0, mx + dx * q, y0 + dy * q, l0, halo);
            text_at(1, mx + dx * q, y1 + dy * q, l1, halo);
        }
    }
    text_at(0, mx, y0, l0, col);
    col[3] = 0.78f;
    text_at(1, mx, y1, l1, col);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_DYNAMIC_DRAW);
    return tvn / 8;
}

static void gl_init(void) {
    prog_land = mk_prog(VS_FULL, FS_LAND, 1);
    prog_worm = mk_prog(VS_WORM, FS_WORM, 1);
    prog_maw = mk_prog(VS_BOX, FS_MAW, 1);
    prog_spr = mk_prog(VS_SPR, FS_SPR, 1);
    prog_storm = mk_prog(VS_FULL, FS_STORM, 1);
    prog_fx = mk_prog(VS_FULL, FS_FX, 1);
    prog_text = mk_prog(VS_TEXT, FS_TEXT, 0);
    glGenVertexArrays(1, &vao_empty);
    glGenVertexArrays(1, &vao_text); glGenBuffers(1, &vbo_text);
    glGenVertexArrays(1, &vao_worm); glGenBuffers(1, &vbo_worm);
    glGenVertexArrays(1, &vao_spr); glGenBuffers(1, &vbo_spr);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void *)8);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray(i);
    glBindVertexArray(vao_worm);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_worm);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)8);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, (void *)24);
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray(i);
    glBindVertexArray(vao_spr);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_spr);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 48, (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 48, (void *)8);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 48, (void *)16);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 48, (void *)32);
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray(i);
    bake_noise();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &tex_rows);
}

static void world_init(void) {
    CX = fr[0] + fr[2] / 2;
    YH = fr[1] + 0.60 * fr[3];
    F = L;
}

static void u3(GLuint pr, const char *n, const float *c) { glUniform3f(U(pr, n), c[0], c[1], c[2]); }

static void set_common(GLuint pr) {
    glUseProgram(pr);
    glUniform3f(U(pr, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(pr, "uRes"), (float)W, (float)H);
    glUniform1f(U(pr, "uL"), (float)L);
    glUniform1f(U(pr, "uT"), (float)fmod(U_ + cur->T0, 3600.0));
    glUniform1f(U(pr, "uYH"), (float)YH);
    glUniform1f(U(pr, "uF"), (float)F);
    glUniform1f(U(pr, "uCX"), (float)CX);
    glUniform1f(U(pr, "uCamX"), (float)camX);
    glUniform1i(U(pr, "uNoise"), 0);
    u3(pr, "uHaze", lt.haze); u3(pr, "uSunHaze", lt.sunhaze); u3(pr, "uRim", lt.rim);
    u3(pr, "uSandL", lt.sandL); u3(pr, "uSandD", lt.sandD); u3(pr, "uDust", lt.dust); u3(pr, "uHide", lt.hide);
    u3(pr, "uAmber", AMBER); u3(pr, "uSunC", lt.sun);
    glUniform3f(U(pr, "uFogC"), fx.fogc[0], fx.fogc[1], fx.fogc[2]);
    glUniform4f(U(pr, "uSun"), (float)(cur->sunx + shx), (float)(cur->suny + shy), (float)cur->sunr, lt.suni);
    glUniform1f(U(pr, "uHazeK"), lt.hazek);
    glUniform1f(U(pr, "uFogK"), (float)fx.fogk);
    glUniform3f(U(pr, "uFlashL"), (float)fx.flx, (float)fx.fly, (float)(fx.storm_in > 0 ? fx.flash * 0.8 : 0));
}

static void render(double t, double wall, GLuint target) {
    view_update();
    frame_build(t);
    rows_update();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glClearDepth(1.0);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_rows);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao_empty);

    /* the sky and the land */
    GLuint pr = prog_land;
    set_common(pr);
    glUniform1f(U(pr, "uDepth"), 1.0f);
    glUniform1i(U(pr, "uRows"), 1);
    glUniform3f(U(pr, "uRowX"), (float)rx0, (float)rdx, (float)RW);
    float zf[NL];
    for (int k = 0; k < NL; k++) zf[k] = (float)LZ[k];
    glUniform1fv(U(pr, "uZ"), NL, zf);
    u3(pr, "uSkyTop", lt.top); u3(pr, "uSkyHor", lt.hor); u3(pr, "uRock", lt.rock);
    glUniform1f(U(pr, "uNight"), (float)fx.night);
    glUniform1f(U(pr, "uSunDisc"), (float)fx.sundisc);
    glUniform1f(U(pr, "uPlume"), (float)fx.plume);
    glUniform1f(U(pr, "uVein"), (float)fx.vein);
    glUniform4f(U(pr, "uMoon1"), (float)(fx.moon[0][0] + shx), (float)(fx.moon[0][1] + shy), (float)fx.moon[0][2], (float)fx.moon[0][3]);
    glUniform4f(U(pr, "uMoon2"), (float)(fx.moon[1][0] + shx), (float)(fx.moon[1][1] + shy), (float)fx.moon[1][2], (float)fx.moon[1][3]);
    glUniform2f(U(pr, "uShake"), (float)shx, (float)shy);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /* the worm, and its maw */
    if (worm.on) {
        worm_strip();
        double z = worm.z;
        float depth = (float)(z / ZFAR);
        double sil = 0;
        if (cur->kind == E_STORM) sil = clamp01(fx.flash * 1.1);
        pr = prog_worm;
        set_common(pr);
        glUniform1f(U(pr, "uDepth"), depth);
        glUniform1f(U(pr, "uZw"), (float)z);
        glUniform1f(U(pr, "uGround"), (float)sy_of(0, z));
        glUniform1f(U(pr, "uPour"), (float)worm.pour);
        glUniform1f(U(pr, "uSil"), (float)sil);
        glUniform4f(U(pr, "uMound"), (float)sx_of(worm.mX, z), (float)(worm.mW * F / z), (float)(worm.mH * F / z), 0);
        glBindVertexArray(vao_worm);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_worm);
        glBufferData(GL_ARRAY_BUFFER, sizeof wormv, wormv, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (NSP + 1) * 2);
        /* the maw, if the head is out of the sand */
        if (worm.Y[0] > -worm.R * 0.2) {
            pr = prog_maw;
            set_common(pr);
            double mk = mixd(0.30, 1.0, clamp01(worm.face)), mr = worm.R * 1.2 * F / z;
            glUniform1f(U(pr, "uDepth"), (float)((z - 2) / ZFAR));
            glUniform2f(U(pr, "uMC"), (float)maw_c[0], (float)maw_c[1]);
            glUniform2f(U(pr, "uMA"), (float)-maw_t[1], (float)maw_t[0]);
            glUniform2f(U(pr, "uMT"), (float)maw_t[0], (float)maw_t[1]);
            glUniform2f(U(pr, "uShift"), 0.0f, (float)(-0.35 * (1 - 0.55 * worm.face)));
            glUniform1f(U(pr, "uMR"), (float)mr);
            glUniform1f(U(pr, "uMK"), (float)mk);
            glUniform1f(U(pr, "uOpen"), (float)clamp01(worm.open));
            glUniform1f(U(pr, "uRot"), (float)(U_ * 0.08));
            glUniform1f(U(pr, "uGlowI"), (float)worm.glow);
            glUniform1f(U(pr, "uZw"), (float)z);
            glUniform1f(U(pr, "uSil"), (float)sil);
            u3(pr, "uGlow", GLOW); u3(pr, "uBone", BONE);
            glBindVertexArray(vao_empty);
            double ex = mr * 1.05 * hypot(maw_t[1], mk * maw_t[0]) + 2, ey = mr * 1.05 * hypot(maw_t[0], mk * maw_t[1]) + 2;
            glUniform4f(U(pr, "uBox"), (float)(maw_c[0] - ex), (float)(maw_c[1] - ey), (float)(maw_c[0] + ex), (float)(maw_c[1] + ey));
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
    }

    /* everything else, far to near */
    if (nspr) {
        for (int i = 0; i < nspr; i++) sord[i] = i;
        qsort(sord, (size_t)nspr, sizeof(int), spr_cmp);
        for (int i = 0; i < nspr; i++) memcpy(sprbuf + i * 72, sprs[sord[i]].v, sizeof(float) * 72);
        pr = prog_spr;
        set_common(pr);
        u3(pr, "uMetal", lt.metal); u3(pr, "uRockC", lt.rock);
        glUniform1fv(U(pr, "uRock"), 24, rock_prof);
        glDepthMask(GL_FALSE);
        glBindVertexArray(vao_spr);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_spr);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nspr * 72 * sizeof(float)), sprbuf, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, nspr * 6);
    }
    glDepthMask(GL_FALSE);
    glBindVertexArray(vao_empty);

    /* the storm wall */
    if (fx.storm_z > 0 && fx.storm_a > 0) {
        pr = prog_storm;
        set_common(pr);
        glUniform1f(U(pr, "uDepth"), (float)(fx.storm_z / ZFAR));
        glUniform1f(U(pr, "uSZ"), (float)fx.storm_z);
        glUniform1f(U(pr, "uSH"), (float)fx.storm_h);
        glUniform1f(U(pr, "uSA"), (float)fx.storm_a);
        glUniform1f(U(pr, "uFlash"), (float)fx.flash);
        glUniform2f(U(pr, "uFlashP"), (float)fx.flx, (float)fx.fly);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    /* murk, lightning, gust, glitter, fade, vignette */
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    pr = prog_fx;
    set_common(pr);
    glUniform1f(U(pr, "uDepth"), 0.0f);
    glUniform1f(U(pr, "uStormIn"), (float)fx.storm_in);
    glUniform1f(U(pr, "uFlash"), (float)fx.flash);
    glUniform2f(U(pr, "uFlashP"), (float)fx.flx, (float)fx.fly);
    glUniform4f(U(pr, "uBolt"), (float)fx.bolt_x, (float)fx.bolt_y, (float)fx.bolt_len, fx.flash > 0.05 ? (float)fx.bolt_seed : 0.0f);
    glUniform4f(U(pr, "uGust"), (float)fx.gust_x, (float)fx.gust_y, (float)fmax(fx.gust_r, 1), (float)fx.gust_a);
    glUniform1f(U(pr, "uGlit"), (float)fx.glit);
    glUniform4f(U(pr, "uGlitC"), (float)fx.glx, (float)fx.gly, (float)fmax(fx.glr, 1), (float)fx.gla);
    glUniform1f(U(pr, "uFade"), (float)fx.fade);
    glUniform3f(U(pr, "uFadeC"), lt.haze[0] * 0.18f, lt.haze[1] * 0.15f, lt.haze[2] * 0.13f);
    glUniform1f(U(pr, "uVig"), (float)fx.vig);
    glUniform4f(U(pr, "uDom"), (float)bx0, (float)by0, (float)bx1, (float)by1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (opt_caption && (span_me < 0 || span_me == span_front)) {
        build_atlas();
        int nv = caption(wall);
        glUseProgram(prog_text);
        glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(U(prog_text, "uAtlas"), 0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glDrawArrays(GL_TRIANGLES, 0, nv);
    }
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "maw");
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
                                      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 0, EGL_DEPTH_SIZE, 24, EGL_NONE};
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

static void usage(void) {
    fprintf(stderr,
        "usage: maw [--window | --root | --layer] [--seed N] [--fps N]\n"
        "           [--palette ember|ash|blood|violet] [--no-caption] [--font FAMILY]\n"
        "           [--event harvest|refuge|rise|horizon|storm|night|bloom] [--schedule]\n"
        "           [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "           [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --palette   default: picked from the seed\n"
        "  --event     play only this vignette, over and over; a --shot without\n"
        "              --at lands on its climax\n"
        "  --fps       frame rate (default 30)\n"
        "  --schedule  print the acts (seconds after the epoch) and exit\n");
    exit(1);
}

static void print_schedule(void) {
    printf("acts of %.0f-%.0f s, shuffled per cycle of %d; each: omen (anticipation), climax, end\n", 68.0, 76.0, NEV);
    for (long n = 0; n < 16; n++) {
        double T0 = act_start(n);
        int k = act_kind(n);
        printf("%7.1f s  act %2ld: %-8s omen %6.1f  climax %6.1f  end %6.1f   %s\n", T0, n, ev_name[k], T0 + ev_omen[k],
               T0 + ev_climax[k], T0 + ev_dur[k], ev_what[k]);
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
        else if (!strcmp(a, "-event") && i + 1 < argc) opt_event = argv[++i];
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-schedule")) opt_schedule = 1;
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) { opt_at = strtod(argv[++i], NULL); opt_at_set = 1; }
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (opt_event) {
        for (forced_ev = 0; forced_ev < NEV; forced_ev++) if (!strcmp(ev_name[forced_ev], opt_event)) break;
        if (forced_ev == NEV) {
            fprintf(stderr, "maw: no event \"%s\" (harvest, refuge, rise, horizon, storm, night, bloom)\n", opt_event);
            exit(1);
        }
        if (!opt_at_set) opt_at = ev_climax[forced_ev];
    }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_schedule) {
        print_schedule();
        return 0;
    }
    int offscreen = opt_shot || opt_bench > 0;
    pick_palette();
    fonts_init();

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    int dw = W, dh = H;
    if (opt_layer && !offscreen) { wl_init(); dw = W; dh = H; }
    else {
        dpy = XOpenDisplay(NULL);
        if (!dpy) die("can't open display");
        int scr = DefaultScreen(dpy);
        root = RootWindow(dpy, scr);
        int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 24, None};
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
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Maw");
        XClassHint chint = {"maw", "Maw"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    domain_init(dw, dh);
    world_init();
    gl_init();

    /* --epoch: when the first act began, shared by every instance; a
     * --shot or --bench then shows the moment --at seconds after it */
    if (!opt_epoch) opt_epoch = wallclock() - (offscreen ? opt_at : 0);
    locked_at = lock_start();
    if (offscreen) locked_at = (time_t)opt_epoch;

    if (offscreen) {
        GLuint tex, fbo, rb;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("offscreen framebuffer incomplete");
        double t0 = now();
        render(opt_at, opt_epoch + opt_at, fbo);
        glFinish();
        double first = (now() - t0) * 1000;
        const char *kn = ev_name[cur->kind];
        double u = U_;
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "maw: %s, %dx%d, %.1f s: act %ld (%s) at %.1f s, first frame %.0f ms\n", palettes[pal].name, W, H, opt_at,
                    cur->n, kn, u, first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9, gmax = 0, csum = 0;
        double *gt = malloc(sizeof(double) * (size_t)opt_bench);
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 30.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            csum += (now() - f0) * 1000;
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            gt[i] = ns / 1e6;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6); gmax = fmax(gmax, ns / 1e6);
            if (ms > worst) worst = ms;
        }
        printf("maw: %s, act %ld (%s), %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n", palettes[pal].name,
               cur->n, kn, W, H, opt_bench, opt_at, sum / opt_bench, worst);
        for (int i = 1; i < opt_bench; i++) for (int j = i; j > 0 && gt[j] < gt[j - 1]; j--) { double x = gt[j]; gt[j] = gt[j - 1]; gt[j - 1] = x; }
        printf("maw:   GPU %.3f ms (median %.3f, best %.3f, worst %.3f); CPU to submit %.3f ms\n", gsum / opt_bench, gt[opt_bench / 2], gmin, gmax, csum / opt_bench);
        printf("maw: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
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
