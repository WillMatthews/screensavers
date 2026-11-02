/*
 * pendulums — chaos made visible.
 *
 * Scenes, one after another (each with its own masses, lengths, starting
 * angles and palette, hashed from --seed and the scene's number):
 *
 *   FAN     800-2000 double pendulums hung from one pivot, started from rest
 *           at angles a hair apart (as little as 1e-13 rad between
 *           neighbours, found by bisection so the fan frays 13-22 s in).
 *           At first they swing as one white line (every colour of the
 *           palette added together); the ensemble then disperses into a
 *           gradient ribbon and blooms into chaos. Rods are fine translucent
 *           lines, bobs small glowing discs, and the lower bobs leave trails
 *           in a decaying HDR buffer: luminous ribbons, shorter comet tails
 *           once it's chaos. After a while every pendulum glides into the
 *           next scene's starting fan and is let go. Variants: low energy
 *           (quasi-periodic: the ribbon shears but never shatters) and a
 *           sweep of energies (order at one end of the fan, chaos at the
 *           other).
 *   MAP     the chaos map on the GPU: every cell a pendulum started from rest
 *           at (θ1, θ2) across the screen, RK4 in a fragment shader on
 *           ping-pong float textures, coloured by when it first flips over;
 *           the fractal grows in real time, then faster and faster, around
 *           the lemon of energies too low ever to flip (drawn as a contour).
 *   PHASE   one double pendulum beside its Poincaré section (θ1 = 0, ω1 > 0),
 *           built up by 220 hidden pendulums on the same energy surface
 *           running at 9x; each has a shadow 1e-8 away whose divergence
 *           (Benettin) marks it chaotic (the sea's quiet tint) or regular
 *           (its own hue: the islands). The featured pendulum's own
 *           crossings are ringed; the allowed region's boundary is drawn.
 *
 * Physics: g = 1, time in sqrt(l/g); the standard equations of motion from
 * the Lagrangian, classical RK4 in double at 180 steps per real second,
 * from t = 0 of each scene (fast-forwarded for --at or a late start), so
 * the state depends only on seed, epoch and time. --check verifies the
 * equations against a mass-matrix solve, measures energy drift and
 * Lyapunov exponents, and that 1 and 8 threads integrate bit-identically.
 * The caption shows time, time away (swaylock's start), and the scene's
 * numbers: the starting spread δθ0, the measured Lyapunov exponent λ from
 * a sparkline of log(neighbour separation), and the energy error |ΔE|.
 *
 * Palettes: spectrum, ember, ice, neon (rotating), and ink (dark on paper;
 * only with --palette ink). --scene fixes the scene kind, --variant the
 * fan's, --count the number of pendulums, --load scales it by CPU load.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout shows the fan (with the
 * clock); a portrait output the phase portrait; the rest, left to right,
 * the chaos map (marking the fan's starting angles) and the phase portrait
 * in turn. All share the scene timeline.
 *
 * Same plumbing as gauges and flightsim: Xlib + GLX for --window / --root,
 * a native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame
 * offscreen, --bench N to time N frames.
 *
 * Keys (window mode): q / Escape quit.
 */
#define _GNU_SOURCE
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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

#define PI 3.14159265358979323846
#define TAU (2 * PI)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check, opt_caption = 1, opt_load, opt_count, opt_variant = -1;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the first scene started at */
static int opt_pal = -1, opt_scene = -1;
static const char *opt_font = "Inter";
static const char *opt_mfont = "Linux Libertine O";

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { SC_FAN, SC_MAP, SC_PHASE, SC_CYCLE };
static int role = SC_CYCLE;            /* what this instance draws: one kind, or them all in turn */

static void die(const char *what) { fprintf(stderr, "pendulums: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static double smooth(double x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return x * x * (3 - 2 * x); }
static double wrap(double a) { return a - TAU * floor((a + PI) / TAU); }

static uint32_t hash32(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
typedef struct { uint64_t s; } Rng;
static double rnd(Rng *r) {
    r->s = r->s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(r->s >> 11) * (1.0 / 9007199254740992.0);
}
static Rng rng_for(int k, int salt) {
    Rng r = {((uint64_t)hash32(opt_seed, (uint32_t)k * 7u + (uint32_t)salt) << 32) | hash32((uint32_t)k, opt_seed ^ 0xA5A5u)};
    rnd(&r); rnd(&r);
    return r;
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

/* CPU load since the last call, 0..1 (for --load) */
static double cpu_load(void) {
    static unsigned long long pb, pt;
    unsigned long long v[10] = {0};
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return 0;
    int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    fclose(f);
    if (n < 4) return 0;
    unsigned long long tot = 0;
    for (int i = 0; i < 8; i++) tot += v[i];
    unsigned long long busy = tot - v[3] - v[4];
    double l = tot > pt ? (double)(busy - pb) / (double)(tot - pt) : 0;
    pb = busy; pt = tot;
    return l < 0 ? 0 : l > 1 ? 1 : l;
}

/* ---- a small thread pool: the ensemble is split across workers, each
 * pendulum integrated by exactly one of them, so the result never depends
 * on how many there are ---- */
#define MAXTH 16
static int nthreads = 1, nworkers;
static pthread_mutex_t pool_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_go = PTHREAD_COND_INITIALIZER, pool_done = PTHREAD_COND_INITIALIZER;
static void (*job_fn)(int, int, void *);
static void *job_arg;
static int job_n, job_gen, job_left, job_split;

static void *worker(void *arg) {
    int id = (int)(intptr_t)arg, gen = 0;
    for (;;) {
        pthread_mutex_lock(&pool_mx);
        while (job_gen == gen) pthread_cond_wait(&pool_go, &pool_mx);
        gen = job_gen;
        int split = job_split;
        pthread_mutex_unlock(&pool_mx);
        if (id < split) job_fn(job_n * id / split, job_n * (id + 1) / split, job_arg);
        pthread_mutex_lock(&pool_mx);
        if (--job_left == 0) pthread_cond_signal(&pool_done);
        pthread_mutex_unlock(&pool_mx);
    }
    return NULL;
}

static void pool_start(int n) {
    if (n > MAXTH) n = MAXTH;
    for (; nworkers < n; nworkers++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, (void *)(intptr_t)nworkers);
        pthread_detach(t);
    }
    nthreads = n;
}

static void par_for(int n, void (*fn)(int, int, void *), void *arg) {
    if (nthreads <= 1 || n < 48) { fn(0, n, arg); return; }
    pthread_mutex_lock(&pool_mx);
    job_fn = fn; job_arg = arg; job_n = n; job_split = nthreads;
    job_left = nworkers;
    job_gen++;
    pthread_cond_broadcast(&pool_go);
    while (job_left) pthread_cond_wait(&pool_done, &pool_mx);
    pthread_mutex_unlock(&pool_mx);
}

/* ---- physics ----
 * The planar double pendulum: point masses m1, m2 on massless rods l1, l2,
 * angles from the downward vertical, g = 1 (time in units of sqrt(l/g)).
 * The equations of motion are the standard ones from the Lagrangian
 *   T = (m1+m2) l1^2 w1^2 / 2 + m2 l2^2 w2^2 / 2 + m2 l1 l2 w1 w2 cos(a1-a2)
 *   V = -(m1+m2) l1 cos a1 - m2 l2 cos a2
 * solved for the angular accelerations; integrated with classical RK4 at a
 * fixed step. --check verifies them against a direct solve of the mass
 * matrix and measures the energy drift. */
typedef struct { double m1, m2, l1, l2; } Phys;
typedef struct { double a1, a2, w1, w2; } St;

static inline void deriv(const Phys *p, const St *s, St *d) {
    double s1, c1, s2, c2;
    sincos(s->a1, &s1, &c1);
    sincos(s->a2, &s2, &c2);
    double sd = s1 * c2 - c1 * s2, cd = c1 * c2 + s1 * s2;      /* sin, cos (a1 - a2) */
    double M = p->m1 + p->m2;
    double den = 2 * p->m1 + p->m2 - p->m2 * (2 * cd * cd - 1);  /* ... - m2 cos(2a1 - 2a2) */
    double sx = sd * c2 - cd * s2;                               /* sin(a1 - 2a2) */
    double w1 = s->w1, w2 = s->w2;
    d->a1 = w1;
    d->a2 = w2;
    d->w1 = (-(2 * p->m1 + p->m2) * s1 - p->m2 * sx - 2 * sd * p->m2 * (w2 * w2 * p->l2 + w1 * w1 * p->l1 * cd)) / (p->l1 * den);
    d->w2 = (2 * sd * (w1 * w1 * p->l1 * M + M * c1 + w2 * w2 * p->l2 * p->m2 * cd)) / (p->l2 * den);
}

static inline void rk4(const Phys *p, St *s, double h) {
    St k1, k2, k3, k4, t;
    deriv(p, s, &k1);
    t = (St){s->a1 + 0.5 * h * k1.a1, s->a2 + 0.5 * h * k1.a2, s->w1 + 0.5 * h * k1.w1, s->w2 + 0.5 * h * k1.w2};
    deriv(p, &t, &k2);
    t = (St){s->a1 + 0.5 * h * k2.a1, s->a2 + 0.5 * h * k2.a2, s->w1 + 0.5 * h * k2.w1, s->w2 + 0.5 * h * k2.w2};
    deriv(p, &t, &k3);
    t = (St){s->a1 + h * k3.a1, s->a2 + h * k3.a2, s->w1 + h * k3.w1, s->w2 + h * k3.w2};
    deriv(p, &t, &k4);
    s->a1 += h / 6 * (k1.a1 + 2 * k2.a1 + 2 * k3.a1 + k4.a1);
    s->a2 += h / 6 * (k1.a2 + 2 * k2.a2 + 2 * k3.a2 + k4.a2);
    s->w1 += h / 6 * (k1.w1 + 2 * k2.w1 + 2 * k3.w1 + k4.w1);
    s->w2 += h / 6 * (k1.w2 + 2 * k2.w2 + 2 * k3.w2 + k4.w2);
}

static double energy(const Phys *p, const St *s) {
    double M = p->m1 + p->m2;
    return 0.5 * M * p->l1 * p->l1 * s->w1 * s->w1 + 0.5 * p->m2 * p->l2 * p->l2 * s->w2 * s->w2 +
           p->m2 * p->l1 * p->l2 * s->w1 * s->w2 * cos(s->a1 - s->a2) - M * p->l1 * cos(s->a1) - p->m2 * p->l2 * cos(s->a2);
}

/* the energy scale for relative errors: the potential's whole range */
static double escale(const Phys *p) { return 2 * ((p->m1 + p->m2) * p->l1 + p->m2 * p->l2); }

#define HZ 180                         /* physics steps per real second */

/* ---- palettes ---- */
typedef struct {
    const char *name;
    int light, sinebow;
    float bg[3], bg2[3], text[3];
    float st[5][3];
    float exposure;
} Pal;
static const Pal PALS[] = {
    {"spectrum", 0, 1, {.010f, .011f, .018f}, {.030f, .032f, .048f}, {.78f, .80f, .86f}, {{0}}, 1.0f},
    {"ember", 0, 0, {.014f, .007f, .005f}, {.045f, .022f, .013f}, {.92f, .78f, .60f},
     {{.62f, .07f, .05f}, {.88f, .26f, .07f}, {.98f, .52f, .14f}, {1.0f, .78f, .38f}, {1.0f, .94f, .78f}}, 1.0f},
    {"ice", 0, 0, {.005f, .012f, .020f}, {.018f, .040f, .056f}, {.74f, .87f, .92f},
     {{.12f, .26f, .66f}, {.10f, .52f, .82f}, {.20f, .78f, .88f}, {.62f, .93f, .96f}, {.96f, 1.0f, 1.0f}}, 1.0f},
    {"neon", 0, 0, {.014f, .005f, .026f}, {.040f, .016f, .064f}, {.86f, .78f, .95f},
     {{1.0f, .16f, .78f}, {.62f, .28f, 1.0f}, {.16f, .45f, 1.0f}, {.0f, .88f, 1.0f}, {.45f, 1.0f, .45f}}, 1.0f},
    {"ink", 1, 0, {.937f, .918f, .878f}, {.965f, .950f, .918f}, {.22f, .21f, .20f},
     {{.10f, .14f, .32f}, {.10f, .36f, .42f}, {.36f, .14f, .40f}, {.62f, .14f, .20f}, {.74f, .44f, .12f}}, 1.5f},
};
#define NPAL 5
#define NDARK 4

/* the colour of fraction f along a palette, as seen */
static void pal_disp(int pi, double f, float *o) {
    const Pal *P = &PALS[pi];
    f = f < 0 ? 0 : f > 1 ? 1 : f;
    if (P->sinebow) {
        double t = 0.02 + 0.84 * f;      /* red .. violet, not quite round to red again */
        for (int i = 0; i < 3; i++) {
            double s = sin(PI * (0.5 - t + i / 3.0));
            o[i] = (float)(s * s);
        }
        return;
    }
    double x = f * 4;
    int i = (int)x;
    if (i > 3) i = 3;
    double u = x - i;
    for (int c = 0; c < 3; c++) o[c] = (float)(P->st[i][c] * (1 - u) + P->st[i + 1][c] * u);
}

/* ... as light added to the dark ground, or as ink's absorbance on paper */
static void pal_acc(int pi, double f, float *o) {
    pal_disp(pi, f, o);
    if (PALS[pi].light)
        for (int c = 0; c < 3; c++) o[c] = -logf(fmaxf(o[c] / PALS[pi].bg[c], 0.04f));
}

/* ---- the scenes ----
 * Time is cut into scenes, each with its own pendulum (masses, lengths),
 * starting angles and palette, all hashed from --seed and the scene's
 * number, so every instance knows every scene without talking. */
enum { V_CLASSIC, V_LOW, V_SWEEP };
typedef struct {
    int k, type, variant, pal, n;
    double t0, dur, ttarget;           /* seconds: start, length, when the fan should fray */
    double tscale;                     /* simulated time units per real second */
    double a1, a2, E;                  /* the featured start (the ensemble's middle) and its energy */
    double pa1, pa2, pE;               /* the phase scene's start: an energy where order and chaos mix */
    double eps;                        /* classic, low: angle step between neighbours */
    double sw1, sw2;                   /* sweep: the far end of the line of starts */
    Phys p;
} Scene;

static int scene_type(int k) {
    static const int pattern[6] = {SC_FAN, SC_FAN, SC_MAP, SC_FAN, SC_FAN, SC_PHASE};
    if (opt_scene >= 0 && opt_scene != SC_CYCLE) return opt_scene;
    if (role != SC_CYCLE) return role;
    return pattern[k % 6];
}

static double scene_dur(int k, double *ttarget) {
    Rng r = rng_for(k, 1);
    double tt = 13 + 9 * rnd(&r);
    if (ttarget) *ttarget = tt;
    int ty = span_me >= 0 ? SC_FAN : scene_type(k);
    if (ty == SC_MAP) return 62 + 8 * rnd(&r);
    if (ty == SC_PHASE) return 72 + 8 * rnd(&r);
    return tt + 40 + 10 * rnd(&r);
}

/* the scene playing el seconds after the epoch */
static int scene_at(double el, double *t0) {
    double t = 0;
    for (int k = 0;; k++) {
        double d = scene_dur(k, NULL);
        if (el < t + d || k > 1000000) { *t0 = t; return k; }
        t += d;
    }
}

/* how many steps until two pendulums started off apart part by `sep` rad;
 * ref holds the first one's angles at every step */
static long diverge_steps(const Phys *p, double a1, double a2, double off, double h, long maxs, const float *ref) {
    St b = {a1 + off, a2 + off, 0, 0};
    for (long s = 0; s < maxs; s++) {
        rk4(p, &b, h);
        if (fabs(b.a1 - ref[2 * s]) + fabs(b.a2 - ref[2 * s + 1]) > 0.5) return s;
    }
    return maxs;
}

static void scene_make(int k, Scene *S) {
    Rng r = rng_for(k, 2);
    memset(S, 0, sizeof *S);
    S->k = k;
    S->type = scene_type(k);
    S->dur = scene_dur(k, &S->ttarget);
    double t0 = 0;
    for (int i = 0; i < k; i++) t0 += scene_dur(i, NULL);
    S->t0 = t0;
    /* palettes in a shuffled rotation, never the same twice running */
    if (opt_pal >= 0) S->pal = opt_pal;
    else {
        int perm[NDARK];
        for (int i = 0; i < NDARK; i++) perm[i] = i;
        uint32_t hs = hash32(opt_seed, 77);
        for (int i = NDARK - 1; i > 0; i--) {
            int j = (int)(hash32(hs, (uint32_t)i) % (uint32_t)(i + 1));
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        S->pal = perm[k % NDARK];
    }
    double mu = rnd(&r) < 0.45 ? 1 : exp(log(0.35) + rnd(&r) * log(3.0 / 0.35));
    double lr = rnd(&r) < 0.5 ? 1 : 0.6 + 0.8 * rnd(&r);
    S->p.m1 = 1; S->p.m2 = mu;
    S->p.l1 = 2 / (1 + lr); S->p.l2 = 2 - S->p.l1;
    S->tscale = 2.2 + 0.6 * rnd(&r);
    double u = rnd(&r);
    S->variant = k == 0 ? V_CLASSIC : u < 0.6 ? V_CLASSIC : u < 0.8 ? V_LOW : V_SWEEP;
    if (opt_variant >= 0) S->variant = opt_variant;
    double nr = rnd(&r);
    S->n = S->variant == V_CLASSIC ? 800 + (int)(1200 * nr) : S->variant == V_LOW ? 500 + (int)(600 * nr) : 900 + (int)(700 * nr);
    if (opt_load) S->n = (int)(S->n * (0.4 + 1.2 * cpu_load()));
    if (opt_count > 0) S->n = opt_count;
    if (S->n < 2) S->n = 2;
    double h = S->tscale / HZ;

    if (S->variant == V_CLASSIC) {
        long tsteps = (long)(S->ttarget * HZ), maxs = tsteps * 2;
        float *ref = malloc(sizeof(float) * 2 * (size_t)maxs);
        for (int tries = 0;; tries++) {
            S->a1 = (rnd(&r) < 0.5 ? -1 : 1) * (1.5 + 1.4 * rnd(&r));
            S->a2 = (rnd(&r) < 0.5 ? -1 : 1) * (1.3 + 1.8 * rnd(&r));
            St a = {S->a1, S->a2, 0, 0};
            for (long s = 0; s < maxs; s++) { rk4(&S->p, &a, h); ref[2 * s] = (float)a.a1; ref[2 * s + 1] = (float)a.a2; }
            /* the edge member is (n-1)/2 steps of eps from the middle */
            double half = (S->n - 1) * 0.5;
            double lo = log10(2e-13 * half), hi = log10(0.02);
            long shi = diverge_steps(&S->p, S->a1, S->a2, pow(10, hi), h, maxs, ref);
            if (shi >= maxs && tries < 12) continue;          /* too regular: another start */
            long slo = diverge_steps(&S->p, S->a1, S->a2, pow(10, lo), h, maxs, ref);
            if (slo < tsteps / 2 && tries < 12) continue;     /* too wild: frays at once */
            if (slo <= tsteps) hi = lo;
            else
                for (int it = 0; it < 26; it++) {
                    double mid = 0.5 * (lo + hi);
                    if (diverge_steps(&S->p, S->a1, S->a2, pow(10, mid), h, maxs, ref) > tsteps) lo = mid;
                    else hi = mid;
                }
            S->eps = pow(10, hi) / half;
            break;
        }
        free(ref);
    } else if (S->variant == V_LOW) {
        S->a1 = (rnd(&r) < 0.5 ? -1 : 1) * (0.3 + 0.6 * rnd(&r));
        S->a2 = (rnd(&r) < 0.5 ? -1 : 1) * (0.2 + 0.9 * rnd(&r));
        S->eps = (0.05 + 0.12 * rnd(&r)) / (S->n - 1);
    } else {
        S->sw1 = (rnd(&r) < 0.5 ? -1 : 1) * (2.1 + 0.9 * rnd(&r));
        S->sw2 = (S->sw1 < 0 ? -1 : 1) * (rnd(&r) < 0.3 ? -1 : 1) * (2.0 + 1.0 * rnd(&r));
        double s = 0.06 + 0.94 * 0.5;
        S->a1 = S->sw1 * s; S->a2 = S->sw2 * s;
    }
    St f = {S->a1, S->a2, 0, 0};
    S->E = energy(&S->p, &f);

    /* the phase portrait wants an energy where islands of order sit in a
     * chaotic sea: a fraction q of the way up from hanging still to
     * balanced upright, started from rest at angles that have it */
    Rng rp = rng_for(k, 4);
    double M = S->p.m1 + S->p.m2, Vr = M * S->p.l1 + S->p.m2 * S->p.l2;
    double q = 0.12 + 0.22 * rnd(&rp);
    S->pE = -Vr + q * 2 * Vr;
    for (int tries = 0;; tries++) {
        double a1 = (rnd(&rp) < 0.5 ? -1 : 1) * acos(fmax(-1, fmin(1, -S->pE / Vr))) * (0.3 + 0.9 * rnd(&rp));
        double c2 = (-S->pE - M * S->p.l1 * cos(a1)) / (S->p.m2 * S->p.l2);
        if (fabs(c2) <= 1 || tries > 50) {
            S->pa1 = a1;
            S->pa2 = (rnd(&rp) < 0.5 ? -1 : 1) * acos(fmax(-1, fmin(1, c2)));
            break;
        }
    }
    St g = {S->pa1, S->pa2, 0, 0};
    S->pE = energy(&S->p, &g);
}

static void member_init(const Scene *S, int j, St *s) {
    if (S->variant == V_SWEEP) {
        double f = (j + 0.5) / S->n, u = 0.06 + 0.94 * f;
        *s = (St){S->sw1 * u, S->sw2 * u, 0, 0};
    } else {
        double o = (j - (S->n - 1) * 0.5) * S->eps;
        *s = (St){S->a1 + o, S->a2 + o, 0, 0};
    }
}

/* ---- the ensemble (the fan; and the one featured pendulum of the phase
 * scene, an ensemble of one) ---- */
#define RECMAX 12                      /* trail points kept per member per frame */
#define MORPH 4.5                      /* seconds the fan glides into the next scene's start */
static Scene cur, nxt;
static int have_cur, have_nxt;
static St *ens;
static double *ens_E0, (*m0)[2];
static int ens_n, ens_cap;
static long fan_step, morph_step = -1;
static int nrec;
static long rec_step[RECMAX + 1];
static float *rec;                     /* [member][RECMAX][a1 a2 w1 w2] */
static int trail_reset = 1;
/* divergence history, a sample every quarter second: log10 of the mean
 * distance between neighbours, and the worst energy error */
#define HIST 4096
static float hist_d[HIST], hist_e[HIST];
static int nhist;

static void ens_alloc(int n) {
    if (n <= ens_cap) return;
    ens_cap = n;
    ens = realloc(ens, sizeof *ens * (size_t)n);
    ens_E0 = realloc(ens_E0, sizeof *ens_E0 * (size_t)n);
    m0 = realloc(m0, sizeof *m0 * (size_t)n);
    rec = realloc(rec, sizeof *rec * 4 * RECMAX * (size_t)n);
}

typedef struct { long from, to; int r0; const Scene *S; } FanJob;
static int feat_cross;                 /* the phase scene's featured pendulum: note its crossings */
static void ph_feature_cross(const St *q, const St *s, long step);

static void fan_job(int lo, int hi, void *arg) {
    const FanJob *J = arg;
    double h = J->S->tscale / HZ;
    for (int i = lo; i < hi; i++) {
        St s = ens[i];
        int r = J->r0;
        for (long st = J->from + 1; st <= J->to; st++) {
            St q = s;
            rk4(&J->S->p, &s, h);
            if (feat_cross) ph_feature_cross(&q, &s, st);
            if (st == morph_step) { m0[i][0] = s.a1; m0[i][1] = s.a2; }
            if (r < nrec && rec_step[r] == st) {
                float *o = rec + ((size_t)i * RECMAX + r) * 4;
                o[0] = (float)s.a1; o[1] = (float)s.a2; o[2] = (float)s.w1; o[3] = (float)s.w2;
                r++;
            }
        }
        ens[i] = s;
    }
}

static void ens_begin(const Scene *S, int n) {
    ens_alloc(n);
    ens_n = n;
    for (int j = 0; j < n; j++) {
        member_init(S, j, &ens[j]);
        ens_E0[j] = energy(&S->p, &ens[j]);
    }
    fan_step = 0;
    nrec = 0;
    nhist = 0;
    trail_reset = 1;
}

static void hist_sample(const Scene *S) {
    if (nhist >= HIST) return;
    double d = 0, de = 0;
    for (int j = 0; j + 1 < ens_n; j++) {
        double x = wrap(ens[j + 1].a1 - ens[j].a1), y = wrap(ens[j + 1].a2 - ens[j].a2);
        d += sqrt(x * x + y * y);
    }
    for (int j = 0; j < ens_n; j += 1 + ens_n / 64) de = fmax(de, fabs(energy(&S->p, &ens[j]) - ens_E0[j]));
    hist_d[nhist] = (float)log10(fmax(d / fmax(ens_n - 1, 1), 1e-300));
    hist_e[nhist] = (float)(de / escale(&S->p));
    nhist++;
}

/* integrate the ensemble to step `to`, keeping up to RECMAX evenly spread
 * trail points from the steps after `keep_from` */
static void ens_advance(const Scene *S, long to, long keep_from) {
    if (to <= fan_step) { nrec = 0; return; }
    long a = keep_from > fan_step ? keep_from : fan_step, nst = to - a;
    nrec = 0;
    if (nst > 0)
        for (int r = 0; r < RECMAX && r < nst; r++) {
            long s = nst <= RECMAX ? a + 1 + r : a + (long)((double)(r + 1) * nst / RECMAX);
            rec_step[nrec++] = s;
        }
    int r0 = 0;
    while (fan_step < to) {
        long next = (fan_step / (HZ / 4) + 1) * (HZ / 4);
        if (next > to) next = to;
        FanJob J = {fan_step, next, r0, S};
        par_for(ens_n, fan_job, &J);
        while (r0 < nrec && rec_step[r0] <= next) r0++;
        fan_step = next;
        if (fan_step % (HZ / 4) == 0) hist_sample(S);
    }
}

/* ---- the phase scene: a Poincare section built by many hidden pendulums
 * on the featured one's energy surface, running ahead of real time ---- */
#define PH_M 220
#define PH_SPEED 3                     /* hidden steps (of 3h each) per real step: 9x real time */
#define PH_BUF 24
typedef struct { float x, y, f, born; } Pt;
static St ph[PH_M], ph_sh[PH_M];        /* each hidden pendulum and its shadow, 1e-8 away */
static float ph_f[PH_M];               /* its hue; < 0: unused */
static double ph_lsum[PH_M];           /* running sum of ln(stretch): the Lyapunov exponent */
#define PH_D0 1e-8
static long ph_step;
static Pt *pts;
static int npts, ptscap, pts_uploaded;
static Pt fpts[2048];                  /* the featured pendulum's own crossings */
static int nfpts;
static float ph_cross[PH_M][PH_BUF][2];
static int ph_ncross[PH_M];
static double ph_wmax, ph_xr;          /* the section's w2 range; the a2 half-range shown */

static void pts_add(float x, float y, float f, float born) {
    if (npts == ptscap) pts = realloc(pts, sizeof *pts * (size_t)(ptscap = ptscap ? ptscap * 2 : 65536));
    pts[npts++] = (Pt){x, y, f, born};
}

/* the section a1 = 0, w1 > 0 at angle a2: the range of w2 the energy allows.
 * On it A w1^2/2 + B w1 w2 + C w2^2/2 = K (the kinetic energy left); the
 * ellipse's extreme w2 needs w1 = -B w2 / A, and where that would be
 * negative the bound is instead where the ellipse meets w1 = 0 */
static int sect_bounds(const Phys *p, double E, double a2, double *lo, double *hi) {
    double M = p->m1 + p->m2;
    double K = E - (-M * p->l1 - p->m2 * p->l2 * cos(a2));
    if (K <= 0) { *lo = *hi = 0; return 0; }
    double A = M * p->l1 * p->l1, B = p->m2 * p->l1 * p->l2 * cos(a2), C = p->m2 * p->l2 * p->l2;
    double top = sqrt(2 * K * A / (A * C - B * B)), edge = sqrt(2 * K / C);
    *hi = B <= 0 ? top : edge;
    *lo = B >= 0 ? -top : -edge;
    return 1;
}

static void ph_begin(const Scene *S) {
    Rng r = rng_for(S->k, 3);
    const Phys *p = &S->p;
    double M = p->m1 + p->m2, E = S->pE;
    ph_wmax = 0;
    for (int i = 0; i < 256; i++) {
        double lo, hi;
        sect_bounds(p, E, -PI + TAU * i / 256, &lo, &hi);
        ph_wmax = fmax(ph_wmax, fmax(hi, -lo));
    }
    /* at low energies the lower arm can't swing far: show just its reach */
    ph_xr = 0;
    for (int i = 0; i <= 512; i++) {
        double a2 = -PI + TAU * i / 512, lo, hi;
        if (sect_bounds(p, E, a2, &lo, &hi)) ph_xr = fmax(ph_xr, fabs(a2));
    }
    ph_xr = fmin(PI, ph_xr * 1.12 + 0.05);
    int m = 0;
    for (int tries = 0; m < PH_M && tries < 400000; tries++) {
        double a2 = -PI + TAU * rnd(&r), w2 = (2 * rnd(&r) - 1) * ph_wmax;
        double K = E - (-M * p->l1 - p->m2 * p->l2 * cos(a2));
        double A = M * p->l1 * p->l1, B = p->m2 * p->l1 * p->l2 * cos(a2), C = p->m2 * p->l2 * p->l2;
        /* A w1^2 / 2 + B w1 w2 + C w2^2 / 2 = K, the larger root (w1 > 0) */
        double disc = B * B * w2 * w2 - A * (C * w2 * w2 - 2 * K);
        if (K <= 0 || disc < 0) continue;
        double w1 = (-B * w2 + sqrt(disc)) / A;
        if (w1 <= 0) continue;
        ph[m] = (St){0, a2, w1, w2};
        ph_sh[m] = (St){0, a2 + PH_D0, w1, w2};
        ph_lsum[m] = 0;
        ph_f[m] = (float)rnd(&r);
        m++;
    }
    for (; m < PH_M; m++) { ph[m] = (St){0, 0, 0, 0}; ph_f[m] = -1; }
    ph_step = 0;
    npts = 0;
    pts_uploaded = 0;
    nfpts = 0;
}

typedef struct { long from, to; const Scene *S; } PhJob;

static void ph_job(int lo, int hi, void *arg) {
    const PhJob *J = arg;
    double h = 3 * J->S->tscale / HZ;
    for (int i = lo; i < hi; i++) {
        St s = ph[i];
        ph_ncross[i] = 0;
        if (ph_f[i] < 0) continue;
        long n = (J->to - J->from) * PH_SPEED;
        double s1 = sin(s.a1);
        for (long k = 0; k < n; k++) {
            double s0 = s1;
            St q = s;
            rk4(&J->S->p, &s, h);
            s1 = sin(s.a1);
            /* upward through a1 = 0 (mod 2 pi): the section */
            if (s0 < 0 && s1 >= 0 && cos(s.a1) > 0 && ph_ncross[i] < PH_BUF) {
                double u = s0 / (s0 - s1);
                ph_cross[i][ph_ncross[i]][0] = (float)wrap(q.a2 + u * (s.a2 - q.a2));
                ph_cross[i][ph_ncross[i]][1] = (float)(q.w2 + u * (s.w2 - q.w2));
                ph_ncross[i]++;
            }
        }
        ph[i] = s;
        /* the shadow: how far it has drifted, then pulled back in (Benettin) */
        St d = ph_sh[i];
        for (long k = 0; k < n; k++) rk4(&J->S->p, &d, h);
        double dx[4] = {d.a1 - s.a1, d.a2 - s.a2, d.w1 - s.w1, d.w2 - s.w2};
        double dist = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + dx[3] * dx[3]);
        if (dist > 0) {
            ph_lsum[i] += log(dist / PH_D0);
            double f = PH_D0 / dist;
            ph_sh[i] = (St){s.a1 + dx[0] * f, s.a2 + dx[1] * f, s.w1 + dx[2] * f, s.w2 + dx[3] * f};
        }
    }
}

/* the hidden pendulum's Lyapunov exponent so far, per time unit */
static double ph_lambda(int i, const Scene *S) {
    double t = (double)ph_step * PH_SPEED * 3 * S->tscale / HZ;
    return t > 0 ? ph_lsum[i] / t : 0;
}

static void ph_advance(const Scene *S, long to) {
    while (ph_step < to) {
        long next = ph_step + 30 < to ? ph_step + 30 : to;
        PhJob J = {ph_step, next, S};
        par_for(PH_M, ph_job, &J);
        float born = (float)((double)next / HZ);
        for (int i = 0; i < PH_M; i++)
            for (int c = 0; c < ph_ncross[i]; c++) pts_add(ph_cross[i][c][0], ph_cross[i][c][1], (float)i, born);
        ph_step = next;
    }
}

/* the featured pendulum's crossings, found between consecutive steps */
static void ph_feature_cross(const St *q, const St *s, long step) {
    double s0 = sin(q->a1), s1 = sin(s->a1);
    if (s0 < 0 && s1 >= 0 && cos(s->a1) > 0 && nfpts < (int)(sizeof fpts / sizeof *fpts)) {
        double u = s0 / (s0 - s1);
        fpts[nfpts++] = (Pt){(float)wrap(q->a2 + u * (s->a2 - q->a2)), (float)(q->w2 + u * (s->w2 - q->w2)), -1,
                             (float)((double)step / HZ)};
    }
}

/* ---- the chaos map (on the GPU): every cell a pendulum started from rest
 * at the angles it sits at, coloured by when it first flips over ---- */
#define MAP_H 0.04                     /* time units per map step */
static long map_step;
static int map_k = -1, map_gw, map_gh, map_cur;
static double map_T(const Scene *S, double t) {
    /* real time at first, then faster and faster: about 450 time units by the end */
    double acc = fmax(0, (440 - S->tscale * 60) * 2 / (60.0 * 60.0));
    return S->tscale * t + 0.5 * acc * t * t;
}

/* ---- glyph atlas: printable ASCII and the few symbols the captions use,
 * in each text style, rasterised at its exact pixel size ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
static const uint32_t XCP[] = {0x3B8, 0x3C9, 0x3BB, 0x3B4, 0x394, 0x3C0, 0x3C4, 0x221A, 0x2080, 0x2081, 0x2082, 0x207B,
                               0x2070, 0xB9, 0xB2, 0xB3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079, 0xB7, 0xD7,
                               0x2248, 0x2212, 0x2192, 0x221E, 0x2113, 0x2264, 0x2265, 0x2032, 0xB0, 0x2009, 0x3BC,
                               0x2016, 0x2014, 0x2013, 0x2026, 0x27E8, 0x27E9, 0x2202, 0xE9};
#define NXCP (int)(sizeof XCP / sizeof *XCP)
#define NGL (95 + NXCP)
typedef struct { int face, px; Glyph g[NGL]; } Style;
#define MAXSTYLE 16
#define NFACE 3
enum { F_SANS, F_MATH, F_MATHI };
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h, atlas_dirty, atlas_upload;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        char pat_s[256];
        if (i == F_SANS) snprintf(pat_s, sizeof pat_s, "%s:weight=light", opt_font);
        else snprintf(pat_s, sizeof pat_s, "%s:slant=%s", opt_mfont, i == F_MATHI ? "italic" : "roman");
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
    if (p < 7) p = 7;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) die("too many text styles");
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    atlas_dirty = 1;
    return nstyles++;
}

static uint32_t gl_cp(int i) { return i < 95 ? (uint32_t)(32 + i) : XCP[i - 95]; }

static void build_atlas(void) {
    free(atlas);
    int cap = 2048;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        for (int c = 0; c < NGL; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            uint32_t cp = gl_cp(c);
            /* a glyph the face lacks comes from the other family */
            FT_Face f = faces[styles[s].face];
            if (!FT_Get_Char_Index(f, cp)) f = faces[styles[s].face == F_SANS ? F_MATH : F_SANS];
            FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
            if (FT_Load_Char(f, cp, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
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
    atlas_dirty = 0;
    atlas_upload = 1;
}

static uint32_t utf8(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t c = *s++;
    if (c >= 0xF0) { c = (c & 7) << 18 | (s[0] & 63u) << 12 | (s[1] & 63u) << 6 | (s[2] & 63u); s += 3; }
    else if (c >= 0xE0) { c = (c & 15) << 12 | (s[0] & 63u) << 6 | (s[1] & 63u); s += 2; }
    else if (c >= 0xC0) { c = (c & 31) << 6 | (s[0] & 63u); s += 1; }
    *p = (const char *)s;
    return c;
}

static const Glyph *glyph(int s, uint32_t c) {
    if (atlas_dirty) build_atlas();
    if (c >= 32 && c < 127) return &styles[s].g[c - 32];
    for (int i = 0; i < NXCP; i++)
        if (XCP[i] == c) return &styles[s].g[95 + i];
    return &styles[s].g[0];
}

/* Text runs: \1 switches to the math italic (for variables), \2 to the
 * math roman, \3 back to the sans; s is a sans style, m an italic one and
 * mr a roman one, all of a size */
typedef struct { int s, m, mr; } TFont;
static int run_style(TFont f, int which) { return which == 1 ? f.m : which == 2 ? f.mr : f.s; }

static float text_width(TFont f, const char *str, float track) {
    float w = 0;
    int cs = f.s;
    for (const char *p = str; *p;) {
        if (*p >= 1 && *p <= 3) { cs = run_style(f, *p == 3 ? 0 : *p); p++; continue; }
        w += glyph(cs, utf8(&p))->adv + track;
    }
    return w - track;
}

/* text quads: x, y, u, v, r, g, b, a per vertex */
static float *tv;
static int tvn, tvcap;

static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 8192) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x;
 * returns the width */
static float text(TFont f, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(f, str, track);
    float pen = roundf(x - (align == 0 ? w * 0.5f : align > 0 ? w : 0));
    y = roundf(y);
    int cs = f.s;
    for (const char *p = str; *p;) {
        if (*p >= 1 && *p <= 3) { cs = run_style(f, *p == 3 ? 0 : *p); p++; continue; }
        const Glyph *g = glyph(cs, utf8(&p));
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y0, u1, v0, col); quad_v(x1, y1, u1, v1, col);
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y1, u1, v1, col); quad_v(x0, y1, u0, v1, col);
        }
        pen += g->adv + track;
    }
    return w;
}

static TFont font_px(float px) {
    TFont f = {style(F_SANS, px), style(F_MATHI, px * 1.12f), style(F_MATH, px * 1.12f)};
    return f;
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS_FULL = "#version 330 core\n" GLSL(
    out vec2 vUV;
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        vUV = p;
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

static const char *FS_NONE = "#version 330 core\n" GLSL(
    out vec4 o;
    void main() { o = vec4(0.0); });

/* the trails' and the section's light on the ground: added light on the
 * dark palettes (rolling off softly), absorbed by the paper on ink */
static const char *FS_FINAL = "#version 330 core\n" GLSL(
    uniform sampler2D uAcc;
    uniform vec3 uBg, uBg2;
    uniform vec2 uRes, uGlow;
    uniform float uExp, uLight, uFade;
    in vec2 vUV;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec3 a = texture(uAcc, vUV).rgb * uExp * uFade;
        vec2 q = (vUV - uGlow) * uRes / min(uRes.x, uRes.y);
        vec3 bg = mix(uBg2, uBg, smoothstep(0.0, 0.85, length(q)));
        vec3 c = uLight > 0.5 ? bg * exp(-a) : 1.0 - (1.0 - bg) * exp(-a);
        c += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
        o = vec4(c, 1.0);
    });

/* trails: triangle strips with a signed across-coordinate for antialiasing */
static const char *VS_STRIP = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in float aSide;
    layout(location = 2) in vec4 aCol;
    uniform vec2 uRes;
    out float vSide;
    out vec4 vCol;
    void main() {
        vSide = aSide; vCol = aCol;
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_STRIP = "#version 330 core\n" GLSL(
    uniform float uHW, uExt;
    in float vSide;
    in vec4 vCol;
    out vec4 o;
    void main() {
        float d = abs(vSide) * uExt;
        float c = clamp(uHW + 0.5 - d, 0.0, 1.0);
        o = vec4(vCol.rgb * vCol.a * c, 0.0);
    });

/* capsules (rods, bobs, rules, marks): a segment a-b of radius r with a
 * soft glow around it, drawn instanced */
static const char *VS_CAPS = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 iAB;
    layout(location = 1) in vec4 iCol;
    layout(location = 2) in vec3 iRG;
    uniform vec2 uRes;
    out vec2 vP;
    flat out vec4 vAB;
    flat out vec4 vCol;
    flat out vec3 vRG;
    void main() {
        vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
        vec2 a = iAB.xy, b = iAB.zw, d = b - a;
        float L = length(d);
        vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0), n = vec2(-t.y, t.x);
        float ext = max(iRG.x, iRG.z > 0.0 ? iRG.y * 2.2 : 0.0) + 1.5;
        vec2 p = (a + b) * 0.5 + t * corner.x * (L * 0.5 + ext) + n * corner.y * ext;
        vP = p; vAB = iAB; vCol = iCol; vRG = iRG;
        gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_CAPS = "#version 330 core\n" GLSL(
    uniform float uLight;
    in vec2 vP;
    flat in vec4 vAB;
    flat in vec4 vCol;
    flat in vec3 vRG;
    out vec4 o;
    void main() {
        vec2 pa = vP - vAB.xy, ba = vAB.zw - vAB.xy;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        float d = length(pa - ba * h);
        float r = max(vRG.x, 0.5);
        float core = clamp(r + 0.5 - d, 0.0, 1.0) * min(vRG.x * 2.0, 1.0);
        float glow = vRG.z * exp(-d * d / (vRG.y * vRG.y + 1e-4));
        float a = clamp(core * vCol.a + glow, 0.0, 1.0);
        vec3 c = uLight > 0.5 ? (1.0 - vCol.rgb) * a : vCol.rgb * a;
        o = vec4(c, a);
    });

/* the section's points: tiny soft dots */
static const char *VS_PTS = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aP;          /* a2, w2, which pendulum, born */
    uniform vec2 uRes;
    uniform vec4 uRect, uRange;               /* x0 y0 x1 y1 px; a2 lo hi, w2 lo hi */
    uniform float uSize, uT, uI;
    uniform sampler2D uTC;                    /* each hidden pendulum's colour */
    out vec3 vCol;
    void main() {
        float x = mix(uRect.x, uRect.z, (aP.x - uRange.x) / (uRange.y - uRange.x));
        float y = mix(uRect.w, uRect.y, (aP.y - uRange.z) / (uRange.w - uRange.z));
        float age = uT - aP.w;
        float fresh = age < 0.0 ? 0.0 : 1.0 + 2.5 * exp(-age * 1.5);
        vCol = texelFetch(uTC, ivec2(int(aP.z), 0), 0).rgb * uI * fresh;
        gl_PointSize = uSize;
        gl_Position = vec4(x / uRes.x * 2.0 - 1.0, 1.0 - y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_PTS = "#version 330 core\n" GLSL(
    in vec3 vCol;
    out vec4 o;
    void main() {
        vec2 q = gl_PointCoord * 2.0 - 1.0;
        float a = exp(-dot(q, q) * 3.0);
        o = vec4(vCol * a, 0.0);
    });

/* the chaos map: RK4 for every cell, several steps a pass; a cell stops
 * once it has flipped and keeps the time it did */
#define MAP_COMMON GLSL( \
    const float PI = 3.14159265358979; \
    uniform vec4 uP; \
    vec4 dv(vec4 s) { \
        float s1 = sin(s.x), c1 = cos(s.x), s2 = sin(s.y), c2 = cos(s.y); \
        float sd = s1 * c2 - c1 * s2, cd = c1 * c2 + s1 * s2; \
        float m1 = uP.x, m2 = uP.y, l1 = uP.z, l2 = uP.w, M = m1 + m2; \
        float den = 2.0 * m1 + m2 - m2 * (2.0 * cd * cd - 1.0); \
        float sx = sd * c2 - cd * s2; \
        float w1 = s.z, w2 = s.w; \
        float d1 = (-(2.0 * m1 + m2) * s1 - m2 * sx - 2.0 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd)) / (l1 * den); \
        float d2 = (2.0 * sd * (w1 * w1 * l1 * M + M * c1 + w2 * w2 * l2 * m2 * cd)) / (l2 * den); \
        return vec4(w1, w2, d1, d2); \
    })
static const char *FS_MAP_INIT = "#version 330 core\n" MAP_COMMON GLSL(
    uniform vec2 uG, uView;                   /* grid size; radians across the grid */
    layout(location = 0) out vec4 oS;
    layout(location = 1) out vec4 oA;
    void main() {
        vec2 th = (gl_FragCoord.xy / uG - 0.5) * uView;
        th = th - 2.0 * PI * floor((th + PI) / (2.0 * PI));
        oS = vec4(th, 0.0, 0.0);
        oA = vec4(-1.0, 0.0, 0.0, 0.0);
    });
static const char *FS_MAP_STEP = "#version 330 core\n" MAP_COMMON GLSL(
    uniform sampler2D uS, uA;
    uniform float uH, uT0;
    uniform int uN;
    layout(location = 0) out vec4 oS;
    layout(location = 1) out vec4 oA;
    void main() {
        ivec2 ij = ivec2(gl_FragCoord.xy);
        vec4 s = texelFetch(uS, ij, 0), a = texelFetch(uA, ij, 0);
        if (a.x < 0.0) {
            float t = uT0;
            for (int k = 0; k < uN; k++) {
                vec4 k1 = dv(s), k2 = dv(s + 0.5 * uH * k1), k3 = dv(s + 0.5 * uH * k2), k4 = dv(s + uH * k3);
                s += uH / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                t += uH;
                if (abs(s.x) > PI || abs(s.y) > PI) { a = vec4(t, abs(s.x) > PI ? 1.0 : 2.0, 0.0, 0.0); break; }
            }
        }
        oS = s;
        oA = a;
    });
static const char *FS_MAP_SHOW = "#version 330 core\n" GLSL(
    const float PI = 3.14159265358979;
    uniform sampler2D uS, uA;
    uniform vec2 uG, uRes, uView;
    uniform vec3 uBg, uBg2, uSt[9];
    uniform vec4 uP;
    uniform float uT, uTmax, uFade, uLight, uExp, uEmin;
    in vec2 vUV;
    out vec4 o;
    vec3 pal(float f) {
        float x = clamp(f, 0.0, 1.0) * 8.0;
        int i = min(int(x), 7);
        return mix(uSt[i], uSt[i + 1], x - float(i));
    }
    vec3 cell(ivec2 ij) {
        ij = clamp(ij, ivec2(0), ivec2(uG) - 1);
        vec4 a = texelFetch(uA, ij, 0);
        if (a.x >= 0.0) {
            /* flipped: hue by the log of when, the early broad lobes dim and
             * the late fractal filaments bright, with faint isochrones */
            float lt = log(1.0 + a.x), f = clamp(lt / log(1.0 + uTmax), 0.0, 1.0);
            float iso = 0.8 + 0.2 * smoothstep(0.0, 0.12, abs(fract(lt * 2.0) - 0.5));
            float flash = exp(-max(uT - a.x, 0.0) * 0.5);
            return pal(f) * ((0.3 + 1.9 * f * f) * iso + 2.2 * flash);
        }
        vec4 s = texelFetch(uS, ij, 0);
        /* not yet flipped: a faint live shade of where its arms point */
        float v = 0.5 - 0.5 * cos(s.y), u = 0.5 - 0.5 * cos(s.x);
        return pal(0.5) * (0.035 + 0.10 * v * v + 0.05 * u);
    }
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 g = vUV * uG - 0.5;
        ivec2 i0 = ivec2(floor(g));
        vec2 f = fract(g);
        vec3 e = mix(mix(cell(i0), cell(i0 + ivec2(1, 0)), f.x), mix(cell(i0 + ivec2(0, 1)), cell(i0 + ivec2(1, 1)), f.x), f.y);
        vec2 th = (vUV - 0.5) * uView;
        bool inside = abs(th.x) <= PI && abs(th.y) <= PI;
        if (!inside) e = mix(vec3(dot(e, vec3(0.3, 0.5, 0.2))), e, 0.5) * 0.24;
        /* the energy contour inside which nothing can ever flip */
        float M = uP.x + uP.y;
        float V = -M * uP.z * cos(th.x) - uP.y * uP.w * cos(th.y);
        float w = fwidth(V);
        float line = 1.0 - smoothstep(0.0, 1.2 * w, abs(V - uEmin));
        e += pal(0.15) * line * (inside ? 0.55 : 0.2);
        e *= uFade;
        vec2 q = (vUV - 0.5) * uRes / min(uRes.x, uRes.y);
        vec3 bg = mix(uBg2, uBg, smoothstep(0.0, 0.85, length(q)));
        vec3 c = uLight > 0.5 ? bg * exp(-e * uExp) : 1.0 - (1.0 - bg) * exp(-e * uExp);
        c += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
        o = vec4(c, 1.0);
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
    void main() {
        float a = vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r;
        o = vec4(vCol.rgb * a, a);
    });

static GLuint prog_none, prog_final, prog_strip, prog_caps, prog_pts, prog_minit, prog_mstep, prog_mshow, prog_text;
static GLuint vao_empty, vao_strip, vbo_strip, vao_caps, vbo_caps, vao_pts, vbo_pts, vao_text, vbo_text;
static GLuint tex_atlas, tex_acc, fbo_acc, tex_tc;
static GLuint tex_ms[2], tex_ma[2], fbo_map[2];
static int acc_w, acc_h, pts_cap_gpu;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "pendulums: shader:\n%s\n", log);
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
    if (!ok) die("shader link failed");
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static void gl_init(void) {
    prog_none = mk_prog(VS_FULL, FS_NONE);
    prog_final = mk_prog(VS_FULL, FS_FINAL);
    prog_strip = mk_prog(VS_STRIP, FS_STRIP);
    prog_caps = mk_prog(VS_CAPS, FS_CAPS);
    prog_pts = mk_prog(VS_PTS, FS_PTS);
    prog_minit = mk_prog(VS_FULL, FS_MAP_INIT);
    prog_mstep = mk_prog(VS_FULL, FS_MAP_STEP);
    prog_mshow = mk_prog(VS_FULL, FS_MAP_SHOW);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_empty);

    glGenVertexArrays(1, &vao_strip);
    glGenBuffers(1, &vbo_strip);
    glBindVertexArray(vao_strip);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);

    glGenVertexArrays(1, &vao_caps);
    glGenBuffers(1, &vbo_caps);
    glBindVertexArray(vao_caps);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_caps);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void *)(4 * sizeof(float)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void *)(8 * sizeof(float)));
    for (int i = 0; i < 3; i++) { glEnableVertexAttribArray((GLuint)i); glVertexAttribDivisor((GLuint)i, 1); }

    glGenVertexArrays(1, &vao_pts);
    glGenBuffers(1, &vbo_pts);
    glBindVertexArray(vao_pts);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pts);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);
    glGenTextures(1, &tex_atlas);
    glEnable(GL_PROGRAM_POINT_SIZE);
}

static GLuint mk_fbo(GLuint t0, GLuint t1) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t0, 0);
    if (t1) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, t1, 0);
        GLenum db[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, db);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("incomplete framebuffer");
    return f;
}

static void acc_clear(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    trail_reset = 1;
}

static void ensure_size(void) {
    if (acc_w == W && acc_h == H) return;
    if (tex_acc) { glDeleteTextures(1, &tex_acc); glDeleteFramebuffers(1, &fbo_acc); }
    tex_acc = mk_tex(GL_RGBA16F, W, H, GL_NEAREST);
    fbo_acc = mk_fbo(tex_acc, 0);
    acc_clear();
    acc_w = W; acc_h = H;
    map_k = -1;                        /* the map's grid follows the size: start it over */
}

/* ---- per-frame geometry ---- */
static float *cv;                      /* capsule instances, 11 floats each */
static int cvn, cvcap;
static void cap_add(float ax, float ay, float bx, float by, float r, float gr, float ga, const float *rgb, float a) {
    if (cvn + 11 > cvcap) cv = realloc(cv, (size_t)(cvcap = cvcap ? cvcap * 2 : 1 << 16) * sizeof *cv);
    float *o = cv + cvn;
    o[0] = ax; o[1] = ay; o[2] = bx; o[3] = by;
    o[4] = rgb[0]; o[5] = rgb[1]; o[6] = rgb[2]; o[7] = a;
    o[8] = r; o[9] = gr; o[10] = ga;
    cvn += 11;
}

static float *sv;                      /* trail strip vertices, 7 floats each */
static int svn, svcap;
static int nstrips;

static void draw_caps(int light) {
    if (!cvn) return;
    glUseProgram(prog_caps);
    glUniform2f(U(prog_caps, "uRes"), (float)W, (float)H);
    glUniform1f(U(prog_caps, "uLight"), (float)light);
    glEnable(GL_BLEND);
    /* screen on the dark grounds, multiply on paper: never clips */
    if (light) glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
    else glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
    glBindVertexArray(vao_caps);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_caps);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(cvn * sizeof *cv), cv, GL_STREAM_DRAW);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, cvn / 11);
    cvn = 0;
}

static void draw_text(void) {
    if (!tvn) return;
    if (atlas_dirty) build_atlas();
    if (atlas_upload) {
        atlas_upload = 0;
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glUseProgram(prog_text);
    glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glUniform1i(U(prog_text, "uAtlas"), 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof *tv), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    tvn = 0;
}

/* palette stops for the shaders, as accumulated light / absorbance */
static void set_stops(GLuint p, const char *name, int pal) {
    float st[9][3];
    for (int i = 0; i < 9; i++) pal_acc(pal, i / 8.0, st[i]);
    glUniform3fv(U(p, name), 9, &st[0][0]);
}

/* ---- the views ---- */
typedef struct { float cx, cy, s; } View;   /* pivot in px, px per unit length */

/* the bobs' positions and velocities (per time unit) for angles a, rates w */
static void bobs(const Phys *p, double a1, double a2, double w1, double w2, double *o) {
    double s1 = sin(a1), c1 = cos(a1), s2 = sin(a2), c2 = cos(a2);
    o[0] = p->l1 * s1; o[1] = p->l1 * c1;
    o[2] = o[0] + p->l2 * s2; o[3] = o[1] + p->l2 * c2;
    o[4] = p->l2 * c2 * w2 + p->l1 * c1 * w1; o[5] = -p->l2 * s2 * w2 - p->l1 * s1 * w1;
}

/* the morph: display member j of the next scene is played by member i of
 * this one; as s goes 0 -> 1 it glides from i's own motion to j's start */
static int disp_n(void) { return morph_step >= 0 && fan_step >= morph_step && have_nxt ? nxt.n : ens_n; }
static int disp_src(int j, int dn) { return dn == ens_n ? j : (int)((double)j * (ens_n - 1) / (dn > 1 ? dn - 1 : 1) + 0.5); }
static double morph_s(long step) {
    if (morph_step < 0 || step < morph_step || !have_nxt) return 0;
    return smooth((double)(step - morph_step) / (MORPH * HZ));
}
static void disp_angles(int j, int i, double s, double a1, double a2, double *o1, double *o2) {
    if (s <= 0) { *o1 = a1; *o2 = a2; return; }
    St t;
    member_init(&nxt, j, &t);
    double t1 = m0[i][0] + wrap(t.a1 - m0[i][0]), t2 = m0[i][1] + wrap(t.a2 - m0[i][1]);
    *o1 = (1 - s) * a1 + s * t1;
    *o2 = (1 - s) * a2 + s * t2;
}

/* last trail point per display member: x, y, vx, vy (px, px per step) */
static float *lastp;
static int lastp_cap;
static long last_step;

static void member_color(int j, int dn, double s, int acc, float *c) {
    float a[3], b[3];
    double f = dn > 1 ? (double)j / (dn - 1) : 0.5;
    if (acc) pal_acc(cur.pal, f, a); else pal_disp(cur.pal, f, a);
    if (s > 0 && have_nxt) {
        if (acc) pal_acc(nxt.pal, f, b); else pal_disp(nxt.pal, f, b);
        for (int k = 0; k < 3; k++) a[k] = (float)(a[k] * (1 - s) + b[k] * s);
    }
    memcpy(c, a, sizeof a);
}

/* the ensemble's trails into the accumulation buffer, its rods and bobs
 * into the capsule list. fade: overall presence; single: one pendulum */
/* the strips are built in parallel, each member into its own slot */
#define SLOT (2 * (RECMAX * 8 + 2))
static float *slots;
static int *slot_n, slots_cap;
static struct { const View *v; float fade, ka, ext, vref; int single, reset, dn; double h; } TJ;

static void trail_job(int lo, int hi, void *arg) {
    (void)arg;
    const View *v = TJ.v;
    const Phys *p = &cur.p;
    for (int j = lo; j < hi; j++) {
        float *out = slots + (size_t)j * SLOT * 7;
        int on = 0;
        slot_n[j] = 0;
        int i = disp_src(j, TJ.dn);
        float col[4];
        member_color(j, TJ.dn, morph_s(rec_step[nrec - 1]), 1, col);
        if (TJ.single) {
            if (PALS[cur.pal].light) pal_acc(cur.pal, 0.0, col);
            else { col[0] = 1.0f; col[1] = 0.95f; col[2] = 0.85f; }
        }
        /* the points: the last one drawn, then this frame's */
        float P[RECMAX + 1][4];
        long PS[RECMAX + 1];
        int np = 0;
        if (!TJ.reset) { memcpy(P[0], lastp + 4 * j, sizeof P[0]); PS[0] = last_step; np = 1; }
        for (int r = 0; r < nrec; r++) {
            const float *q = rec + ((size_t)i * RECMAX + r) * 4;
            double s = morph_s(rec_step[r]), a1, a2, b[6];
            disp_angles(j, i, s, q[0], q[1], &a1, &a2);
            bobs(p, a1, a2, q[2] * (1 - s), q[3] * (1 - s), b);
            P[np][0] = (float)(v->cx + v->s * b[2]);
            P[np][1] = (float)(v->cy + v->s * b[3]);
            P[np][2] = (float)(v->s * b[4]);
            P[np][3] = (float)(v->s * b[5]);
            PS[np] = rec_step[r];
            np++;
        }
        memcpy(lastp + 4 * j, P[np - 1], sizeof P[0]);
        if (np < 2) continue;
        double ms = morph_s(rec_step[nrec - 1]);
        float inten = TJ.ka * TJ.fade * (float)((1 - ms) * (1 - ms));
        if (inten <= 0.0005f) continue;
        for (int k = 0; k + 1 < np; k++) {
            float dt = (float)((double)(PS[k + 1] - PS[k]) * TJ.h);
            float *A = P[k], *B = P[k + 1];
            float len = hypotf(B[0] - A[0], B[1] - A[1]);
            int sub = (int)ceilf(len / 5.0f);
            if (sub < 1) sub = 1;
            if (sub > 8) sub = 8;
            for (int u = k ? 1 : 0; u <= sub; u++) {
                /* cubic Hermite through the steps, with the true velocities */
                float t = (float)u / sub, t2 = t * t, t3 = t2 * t;
                float h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
                float x = h00 * A[0] + h10 * dt * A[2] + h01 * B[0] + h11 * dt * B[2];
                float y = h00 * A[1] + h10 * dt * A[3] + h01 * B[1] + h11 * dt * B[3];
                float vx = A[2] + (B[2] - A[2]) * t, vy = A[3] + (B[3] - A[3]) * t;
                float vl = hypotf(vx, vy);
                if (vl < 1e-3f) { vx = B[0] - A[0]; vy = B[1] - A[1]; vl = hypotf(vx, vy) + 1e-6f; }
                float nx = -vy / vl * TJ.ext, ny = vx / vl * TJ.ext;
                /* a long exposure: where the bob lingers it burns brighter */
                float bright = inten * sqrtf(fminf(1.0f, TJ.vref / (vl + 1e-3f)));
                float *o = out + on * 7;
                o[0] = x + nx; o[1] = y + ny; o[2] = 1;
                o[3] = col[0]; o[4] = col[1]; o[5] = col[2]; o[6] = bright;
                o[7] = x - nx; o[8] = y - ny; o[9] = -1;
                o[10] = col[0]; o[11] = col[1]; o[12] = col[2]; o[13] = bright;
                on += 2;
            }
        }
        slot_n[j] = on;
    }
}

static void draw_ensemble_trails(const View *v, float fade, int single) {
    int dn = disp_n();
    if (lastp_cap < dn) lastp = realloc(lastp, sizeof *lastp * 4 * (size_t)(lastp_cap = dn));
    if (slots_cap < dn) {
        slots_cap = dn;
        slots = realloc(slots, sizeof *slots * (size_t)dn * SLOT * 7);
        slot_n = realloc(slot_n, sizeof *slot_n * (size_t)dn);
    }
    svn = 0;
    nstrips = 0;
    int reset = trail_reset || !nrec;
    if (morph_step >= 0 && last_step < morph_step && fan_step >= morph_step) reset = 1;
    float hw = single ? 1.1f : 0.6f;
    TJ.v = v; TJ.fade = fade; TJ.single = single; TJ.reset = reset; TJ.dn = dn;
    TJ.ka = single ? 0.9f : fminf(1.0f, 34.0f / dn);
    TJ.ext = hw + 1.0f;
    TJ.vref = v->s * 2.0f;             /* px per time unit: a gentle long-exposure falloff above it */
    TJ.h = cur.tscale / HZ;
    if (nrec) par_for(dn, trail_job, NULL);
    /* one strip for all: members joined by degenerate triangles */
    for (int j = 0; j < dn && nrec; j++) {
        int n = slot_n[j];
        if (!n) continue;
        if (svn + (n + 2) * 7 > svcap) sv = realloc(sv, (size_t)(svcap = (svn + (n + 2) * 7) * 2) * sizeof *sv);
        const float *src = slots + (size_t)j * SLOT * 7;
        if (svn) {
            memcpy(sv + svn, sv + svn - 7, sizeof *sv * 7);
            memcpy(sv + svn + 7, src, sizeof *sv * 7);
            svn += 14;
        }
        memcpy(sv + svn, src, sizeof *sv * 7 * (size_t)n);
        svn += n * 7;
        nstrips = 1;
    }
    float ext = TJ.ext;
    if (nrec) {
        last_step = fan_step;
        trail_reset = 0;
    }
    if (!nstrips) return;
    glUseProgram(prog_strip);
    glUniform2f(U(prog_strip, "uRes"), (float)W, (float)H);
    glUniform1f(U(prog_strip, "uHW"), fmaxf(hw, 0.5f));
    glUniform1f(U(prog_strip, "uExt"), ext);
    glBindVertexArray(vao_strip);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(svn * sizeof *sv), sv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, svn / 7);
}

static void draw_ensemble_rods(const View *v, float fade, int single) {
    int dn = disp_n();
    const Phys *p = &cur.p;
    double s = morph_s(fan_step);
    int light = PALS[cur.pal].light;
    float ra = single ? 0.85f : fmaxf(fminf(5.0f / dn, 0.5f), 0.022f);
    float m2r = (float)sqrt(p->m2 / p->m1);
    for (int pass = 0; pass < 2; pass++)
        for (int j = 0; j < dn; j++) {
            int i = disp_src(j, dn);
            double a1, a2, b[6];
            disp_angles(j, i, s, ens[i].a1, ens[i].a2, &a1, &a2);
            bobs(p, a1, a2, 0, 0, b);
            float x1 = (float)(v->cx + v->s * b[0]), y1 = (float)(v->cy + v->s * b[1]);
            float x2 = (float)(v->cx + v->s * b[2]), y2 = (float)(v->cy + v->s * b[3]);
            float c[3];
            member_color(j, dn, s, 0, c);
            if (single) { c[0] = light ? .2f : 1; c[1] = light ? .2f : .96f; c[2] = light ? .22f : .9f; }
            if (pass == 0) {
                cap_add(v->cx, v->cy, x1, y1, single ? 1.1f : 0.55f, 0, 0, c, ra * fade);
                cap_add(x1, y1, x2, y2, single ? 1.1f : 0.55f, 0, 0, c, ra * fade);
            } else {
                float ba = single ? 1.0f : fmaxf(fminf(12.0f / dn, 0.9f), 0.10f);
                float ga = single ? 0.35f : fmaxf(fminf(1.5f / dn, 0.2f), 0.03f);
                cap_add(x1, y1, x1, y1, single ? 4.0f : 1.4f, single ? 9.0f : 4.0f, ga * 0.6f * fade, c, ba * fade);
                cap_add(x2, y2, x2, y2, (single ? 4.0f : 1.6f) * fminf(fmaxf(m2r, 0.7f), 1.5f), single ? 12.0f : 6.0f,
                        ga * fade, c, ba * fade);
            }
        }
    /* the pivot */
    float pc[3];
    memcpy(pc, PALS[cur.pal].text, sizeof pc);
    cap_add(v->cx, v->cy, v->cx, v->cy, 3.2f, 8.0f, 0.12f * fade, pc, 0.9f * fade);
}

/* ---- caption ---- */
static time_t locked_at;
static double t_start_wall;

static void fmt_pow10(char *o, size_t n, double v) {
    /* 3.2 × 10⁻⁹ */
    static const char *sup[10] = {"⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"};
    if (v <= 0) { snprintf(o, n, "0"); return; }
    int e = (int)floor(log10(v));
    double m = v / pow(10, e);
    if (m >= 9.95) { m /= 10; e++; }
    char es[32] = "";
    int ae = abs(e);
    if (e < 0) strcat(es, "⁻");
    char dg[16];
    snprintf(dg, sizeof dg, "%d", ae);
    for (char *q = dg; *q; q++) strcat(es, sup[*q - '0']);
    if (e >= -2 && e <= 2) snprintf(o, n, "%.*f", e < 0 ? 1 - e : 1, v);
    else snprintf(o, n, "%.1f×10%s", m, es);
}

/* the caption: time, time away, and a line of numbers about the scene */
static float cap_x, cap_y, cap_u;      /* where it sits: left, baseline of the clock; size unit */

static void caption(double wt, const char *line, int clock, float fade) {
    if (!opt_caption) return;
    const Pal *P = &PALS[cur.pal];
    float u = cap_u;
    float col[4] = {P->text[0], P->text[1], P->text[2], 0.92f * fade};
    float dim[4] = {P->text[0], P->text[1], P->text[2], 0.55f * fade};
    float x = cap_x, y = cap_y;
    if (clock) {
        time_t ts = (time_t)wt;
        struct tm tm;
        localtime_r(&ts, &tm);
        char buf[64];
        strftime(buf, sizeof buf, "%H:%M", &tm);
        TFont big = font_px(u * 2.6f);
        text(big, x - u * 0.12f, y, -1, buf, col, u * 0.02f);
        double away = wt - (double)locked_at;
        if (away < 0) away = 0;
        long a = (long)away;
        if (a >= 3600) snprintf(buf, sizeof buf, "AWAY  %ld:%02ld:%02ld", a / 3600, a / 60 % 60, a % 60);
        else snprintf(buf, sizeof buf, "AWAY  %ld:%02ld", a / 60, a % 60);
        TFont sm = font_px(u * 0.78f);
        text(sm, x, y + u * 1.25f, -1, buf, dim, u * 0.14f);
        y += u * 2.55f;
    }
    /* the numbers, a line at a time */
    TFont m = font_px(u * 0.84f);
    for (const char *p = line; p && *p;) {
        const char *e = strchr(p, '\n');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        char buf[256];
        if (n >= sizeof buf) n = sizeof buf - 1;
        memcpy(buf, p, n);
        buf[n] = 0;
        text(m, x, y, -1, buf, dim, 0);
        y += u * 1.45f;
        p = e ? e + 1 : p + n;
    }
}

/* the divergence sparkline: log10 of the neighbours' mean separation over the scene */
static void sparkline(float x, float y, float w, float h, float fade) {
    if (nhist < 2) return;
    const Pal *P = &PALS[cur.pal];
    float lo = hist_d[0] - 0.5f, hi = 0.8f;
    double tmax = cur.dur;
    float c[3] = {P->text[0], P->text[1], P->text[2]};
    cap_add(x, y + h, x + w, y + h, 0.5f, 0, 0, c, 0.25f * fade);
    cap_add(x, y, x, y + h, 0.5f, 0, 0, c, 0.25f * fade);
    float px = 0, py = 0;
    for (int i = 0; i < nhist; i++) {
        float t = (float)((i + 1) * 0.25 / tmax);
        float qx = x + w * t, qy = y + h * (1 - clampf((hist_d[i] - lo) / (hi - lo), 0, 1));
        if (i) {
            float cc[3];
            pal_disp(cur.pal, 0.15 + 0.7 * (double)i / nhist, cc);
            if (P->sinebow || P->light) memcpy(cc, c, sizeof cc);
            cap_add(px, py, qx, qy, 0.7f, 0, 0, cc, 0.85f * fade);
        }
        px = qx; py = qy;
    }
    cap_add(px, py, px, py, 1.8f, 5, 0.25f * fade, c, 0.9f * fade);
}

/* the Lyapunov exponent: the slope of ln(separation) through its
 * exponential stretch (grown tenfold, not yet saturating), per real second */
static double lyapunov(void) {
    if (nhist < 6) return -1;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (int i = 0; i < nhist; i++) {
        if (hist_d[i] < hist_d[0] + 1.0f || hist_d[i] > -1.6f) continue;
        double x = (i + 1) * 0.25, y = hist_d[i] * log(10.0);
        sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
    }
    if (n < 6) return -1;
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

/* ---- per-role setup and frames ---- */
static int last_role_drawn = -1;
static double scene_el;                /* seconds into the current scene */

/* the next scene is worked out (its fray timing searched for) on a thread
 * of its own while this one plays, so a scene change never stalls a frame;
 * it is needed only once the morph begins */
static pthread_t nxt_thread;
static int nxt_pending;
static Scene nxt_tmp;
static void *nxt_worker(void *arg) {
    scene_make((int)(intptr_t)arg, &nxt_tmp);
    return NULL;
}
static void nxt_wait(void) {
    if (!nxt_pending) return;
    pthread_join(nxt_thread, NULL);
    nxt = nxt_tmp;
    have_nxt = 1;
    nxt_pending = 0;
}

static void scene_enter(int k) {
    nxt_wait();
    if (have_nxt && nxt.k == k) cur = nxt;
    else scene_make(k, &cur);
    have_cur = 1;
    have_nxt = 0;
    if (pthread_create(&nxt_thread, NULL, nxt_worker, (void *)(intptr_t)(k + 1)) == 0) nxt_pending = 1;
    else { scene_make(k + 1, &nxt); have_nxt = 1; }
    int ty = role == SC_CYCLE ? cur.type : role;
    morph_step = -1;
    feat_cross = 0;
    if (ty == SC_FAN) {
        ens_begin(&cur, cur.n);
        int nty = role == SC_CYCLE ? scene_type(k + 1) : role;
        if (nty == SC_FAN) morph_step = (long)((cur.dur - MORPH) * HZ);
    } else if (ty == SC_PHASE) {
        Scene one = cur;
        one.variant = V_CLASSIC; one.eps = 0; one.n = 1;
        one.a1 = cur.pa1; one.a2 = cur.pa2;
        ens_begin(&one, 1);
        ph_begin(&cur);
        feat_cross = 1;
    }
    acc_clear();
}

static int cur_type(void) { return role == SC_CYCLE ? cur.type : role; }

/* fade in at the start of a scene and out at its end, unless the fan is
 * gliding straight into the next */
static float scene_fade(void) {
    double in = smooth(scene_el / 2.0), out = 1 - smooth((scene_el - (cur.dur - 3.0)) / 3.0);
    int ty = cur_type();
    if (ty == SC_FAN && cur.k > 0) {
        Scene pv;
        int pty = role == SC_CYCLE ? scene_type(cur.k - 1) : role;
        (void)pv;
        if (pty == SC_FAN) in = 1;
    }
    if (ty == SC_FAN && morph_step >= 0) out = 1;
    return (float)(in * out);
}

/* the simulation, to el seconds after the epoch */
static void sim_to(double el, long keep_from_step) {
    double t0;
    int k = scene_at(el, &t0);
    if (!have_cur || k != cur.k) { scene_enter(k); keep_from_step = -1; }
    scene_el = el - cur.t0;
    long target = (long)floor(scene_el * HZ);
    int ty = cur_type();
    if (ty == SC_FAN && morph_step >= 0 && target >= morph_step) nxt_wait();
    if (ty == SC_FAN) ens_advance(&cur, target, keep_from_step);
    else if (ty == SC_PHASE) {
        ens_advance(&cur, target, keep_from_step);
        ph_advance(&cur, target);
    }
}

/* the map: run the grid to the scene's time */
static void map_run(const Scene *S, double t) {
    int gw = (W + 1) / 2, gh = (H + 1) / 2;
    if (map_k != S->k || gw != map_gw || gh != map_gh) {
        if (map_gw != gw || map_gh != gh || !tex_ms[0]) {
            for (int i = 0; i < 2; i++) {
                if (tex_ms[i]) { glDeleteTextures(1, &tex_ms[i]); glDeleteTextures(1, &tex_ma[i]); glDeleteFramebuffers(1, &fbo_map[i]); }
                tex_ms[i] = mk_tex(GL_RGBA32F, gw, gh, GL_NEAREST);
                tex_ma[i] = mk_tex(GL_RGBA32F, gw, gh, GL_NEAREST);
                fbo_map[i] = mk_fbo(tex_ms[i], tex_ma[i]);
            }
        }
        map_gw = gw; map_gh = gh;
        map_k = S->k;
        map_cur = 0;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_map[0]);
        glViewport(0, 0, gw, gh);
        glDisable(GL_BLEND);
        glUseProgram(prog_minit);
        float sc = (float)(fminf((float)W, (float)H) * 0.86f / TAU);
        glUniform2f(U(prog_minit, "uG"), (float)gw, (float)gh);
        glUniform2f(U(prog_minit, "uView"), W / sc, H / sc);
        glBindVertexArray(vao_empty);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        map_step = 0;
    }
    long target = (long)(map_T(S, t) / MAP_H);
    if (target <= map_step) return;
    glDisable(GL_BLEND);
    glUseProgram(prog_mstep);
    glUniform4f(U(prog_mstep, "uP"), (float)S->p.m1, (float)S->p.m2, (float)S->p.l1, (float)S->p.l2);
    glUniform1f(U(prog_mstep, "uH"), (float)MAP_H);
    glUniform1i(U(prog_mstep, "uS"), 0);
    glUniform1i(U(prog_mstep, "uA"), 1);
    glBindVertexArray(vao_empty);
    glViewport(0, 0, map_gw, map_gh);
    while (map_step < target) {
        int n = (int)(target - map_step < 48 ? target - map_step : 48);
        int dst = 1 - map_cur;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_map[dst]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_ms[map_cur]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_ma[map_cur]);
        glUniform1f(U(prog_mstep, "uT0"), (float)(map_step * MAP_H));
        glUniform1i(U(prog_mstep, "uN"), n);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        map_cur = dst;
        map_step += n;
    }
    glActiveTexture(GL_TEXTURE0);
}

static void final_pass(GLuint target, float fade, float gx, float gy) {
    const Pal *P = &PALS[cur.pal];
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_final);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_acc);
    glUniform1i(U(prog_final, "uAcc"), 0);
    glUniform3fv(U(prog_final, "uBg"), 1, P->bg);
    glUniform3fv(U(prog_final, "uBg2"), 1, P->bg2);
    glUniform2f(U(prog_final, "uRes"), (float)W, (float)H);
    glUniform2f(U(prog_final, "uGlow"), gx, gy);
    glUniform1f(U(prog_final, "uExp"), P->exposure);
    glUniform1f(U(prog_final, "uLight"), (float)P->light);
    glUniform1f(U(prog_final, "uFade"), fade);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static double frame_dt = 1.0 / 60;     /* seconds since the last frame drawn */
static double trail_tau = 1.3;        /* seconds the trails take to fade to 1/e */

static void acc_decay(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
    glViewport(0, 0, W, H);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ZERO, GL_CONSTANT_ALPHA);
    glBlendColor(0, 0, 0, (float)exp(-frame_dt / trail_tau));
    glUseProgram(prog_none);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBlendFunc(GL_ONE, GL_ONE);
}

static void set_cap_place(void) {
    float u = fminf((float)W, (float)H) * 0.017f;
    if (u < 9) u = 9;
    cap_u = u;
    cap_x = roundf(u * 2.6f);
    cap_y = roundf(H - u * 6.4f);
}

static void render_fan(double wt, GLuint target) {
    float fade = scene_fade();
    View v = {W * 0.5f, H * 0.5f, fminf((float)W, (float)H) * 0.46f / 2.0f};
    /* long luminous ribbons while the fan holds together, shorter comet
     * tails once it is chaos, or the whole disc fogs over */
    if (cur.variant == V_CLASSIC && nhist > 0) {
        double fr = smooth((hist_d[nhist - 1] - hist_d[0]) / (0.2 - hist_d[0]));
        trail_tau = 1.5 - 0.85 * fr;
    } else trail_tau = cur.variant == V_LOW ? 1.8 : 1.1;
    acc_decay();
    draw_ensemble_trails(&v, fade, 0);
    final_pass(target, 1, 0.5f, 0.5f);
    draw_ensemble_rods(&v, fade, 0);
    /* the reach of the arms, very faint */
    set_cap_place();
    char line[256] = "";
    char ebuf[64], dbuf[64];
    double lam = lyapunov();
    if (cur.variant == V_SWEEP) {
        St a, b;
        member_init(&cur, 0, &a); member_init(&cur, cur.n - 1, &b);
        snprintf(line, sizeof line, "\3%d pendulums   \1E\2 from %.2f to %.2f", cur.n, energy(&cur.p, &a), energy(&cur.p, &b));
    } else {
        fmt_pow10(dbuf, sizeof dbuf, cur.eps);
        if (cur.variant == V_LOW || lam < 0)
            snprintf(line, sizeof line, "\3%d pendulums   \1δθ\2₀ = %s   \1E\2 = %.2f", cur.n, dbuf, cur.E);
        else
            snprintf(line, sizeof line, "\3%d pendulums   \1δθ\2₀ = %s   \1λ\2 ≈ %.2f s\2⁻¹", cur.n, dbuf, lam);
    }
    if (nhist > 0) {
        fmt_pow10(ebuf, sizeof ebuf, fmax(hist_e[nhist - 1], 1e-16));
        size_t l = strlen(line);
        snprintf(line + l, sizeof line - l, "   \2|Δ\1E\2| < %s", ebuf);
    }
    caption(wt, line, 1, 1);
    if (opt_caption && cur.variant != V_SWEEP) {
        float sw = cap_u * 9, sh = cap_u * 2.2f;
        sparkline(cap_x + cap_u * 9.5f, cap_y - cap_u * 2.0f, sw, sh, 1);
    }
    draw_caps(PALS[cur.pal].light);
    draw_text();
}

/* the map's θ <-> px */
static float map_sc(void) { return fminf((float)W, (float)H) * 0.86f / (float)TAU; }

static void render_map(double wt, GLuint target, int clock) {
    float fade = scene_fade();
    const Pal *P = &PALS[cur.pal];
    double T = map_T(&cur, scene_el);
    map_run(&cur, scene_el);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    GLuint p = prog_mshow;
    glUseProgram(p);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_ms[map_cur]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_ma[map_cur]);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(U(p, "uS"), 0);
    glUniform1i(U(p, "uA"), 1);
    float sc = map_sc();
    glUniform2f(U(p, "uG"), (float)map_gw, (float)map_gh);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform2f(U(p, "uView"), W / sc, H / sc);
    glUniform3fv(U(p, "uBg"), 1, P->bg);
    glUniform3fv(U(p, "uBg2"), 1, P->bg2);
    set_stops(p, "uSt", cur.pal);
    glUniform4f(U(p, "uP"), (float)cur.p.m1, (float)cur.p.m2, (float)cur.p.l1, (float)cur.p.l2);
    glUniform1f(U(p, "uT"), (float)(map_step * MAP_H));
    glUniform1f(U(p, "uTmax"), (float)map_T(&cur, cur.dur));
    glUniform1f(U(p, "uFade"), fade);
    glUniform1f(U(p, "uLight"), (float)P->light);
    glUniform1f(U(p, "uExp"), P->exposure);
    /* the lowest energy a flip needs: the lower arm over the top, the upper hanging */
    double M = cur.p.m1 + cur.p.m2;
    glUniform1f(U(p, "uEmin"), (float)fmin(cur.p.m2 * cur.p.l2 - M * cur.p.l1, M * cur.p.l1 - cur.p.m2 * cur.p.l2));
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* the frame of the fundamental square, ticks at -π, 0, π, and the marker */
    float c[3] = {P->text[0], P->text[1], P->text[2]};
    float cx = W * 0.5f, cy = H * 0.5f, hs = (float)(PI * sc);
    float x0 = cx - hs, x1 = cx + hs, y0 = cy - hs, y1 = cy + hs;
    float fa = 0.30f * fade;
    cap_add(x0, y0, x1, y0, 0.5f, 0, 0, c, fa);
    cap_add(x0, y1, x1, y1, 0.5f, 0, 0, c, fa);
    cap_add(x0, y0, x0, y1, 0.5f, 0, 0, c, fa);
    cap_add(x1, y0, x1, y1, 0.5f, 0, 0, c, fa);
    float tk = fminf((float)W, (float)H) * 0.008f;
    TFont f = font_px(fmaxf(fminf((float)W, (float)H) * 0.0145f, 10));
    float tc[4] = {c[0], c[1], c[2], 0.6f * fade};
    static const char *lab[3] = {"−\1π", "0", "\1π"};
    for (int i = 0; i < 3; i++) {
        float x = cx + (i - 1) * hs, y = cy - (i - 1) * hs;
        cap_add(x, y1, x, y1 + tk, 0.5f, 0, 0, c, fa * 1.5f);
        cap_add(x0 - tk, y, x0, y, 0.5f, 0, 0, c, fa * 1.5f);
        text(f, x, y1 + tk * 3.4f, 0, lab[i], tc, 0);
        text(f, x0 - tk * 1.8f, y + tk * 0.6f, 1, lab[i], tc, 0);
    }
    text(f, cx, y1 + tk * 6.2f, 0, "\1θ\2₁", tc, 0);
    text(f, x0 - tk * 6.0f, cy + tk * 0.6f, 1, "\1θ\2₂", tc, 0);
    /* the featured start (the fan's, when spanning) */
    float mx = cx + (float)(wrap(cur.a1) * sc), my = cy - (float)(wrap(cur.a2) * sc);
    float mc[3] = {1, 1, 1};
    if (P->light) memcpy(mc, P->text, sizeof mc);
    float rr = tk * 1.6f;
    for (int i = 0; i < 24; i++) {
        float a0 = (float)(i * TAU / 24), a1 = (float)((i + 0.6) * TAU / 24);
        cap_add(mx + rr * cosf(a0), my + rr * sinf(a0), mx + rr * cosf(a1), my + rr * sinf(a1), 0.6f, 0, 0, mc, 0.8f * fade);
    }
    set_cap_place();
    char line[200];
    snprintf(line, sizeof line, "\3time to first flip   \1t\2 = %.0f √\1ℓ\2/\1g\2\n\1m\2₂/\1m\2₁ = %.2f   \1ℓ\2₂/\1ℓ\2₁ = %.2f",
             T, cur.p.m2 / cur.p.m1, cur.p.l2 / cur.p.l1);
    caption(wt, line, clock, fade);
    draw_caps(P->light);
    draw_text();
}

/* phase scene layout: pendulum view and the section's rectangle */
static void phase_layout(View *v, float *rect) {
    if (H > W) {
        float R = fminf(W * 0.40f, H * 0.21f);
        *v = (View){W * 0.5f, H * 0.25f, R / 2};
        rect[0] = W * 0.10f; rect[2] = W * 0.90f;
        rect[1] = H * 0.50f; rect[3] = H * 0.85f;
    } else {
        float R = fminf(W * 0.21f, H * 0.38f);
        *v = (View){W * 0.27f, H * 0.47f, R / 2};
        rect[0] = W * 0.53f; rect[2] = W * 0.93f;
        rect[1] = H * 0.16f; rect[3] = H * 0.80f;
    }
}

static void render_phase(double wt, GLuint target, int clock) {
    float fade = scene_fade();
    const Pal *P = &PALS[cur.pal];
    View v;
    float rect[4];
    phase_layout(&v, rect);
    double wr = ph_wmax * 1.08 + 0.05;
    trail_tau = 1.3;
    acc_decay();
    draw_ensemble_trails(&v, fade, 1);
    /* the section's points: each drawn every frame at the rate the buffer
     * forgets, so it holds them at a steady brightness */
    if (npts) {
        glBindVertexArray(vao_pts);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts);
        if (npts > pts_cap_gpu) {
            int nc = pts_cap_gpu ? pts_cap_gpu : 65536;
            while (nc < npts) nc *= 2;
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nc * sizeof(Pt)), NULL, GL_DYNAMIC_DRAW);
            pts_cap_gpu = nc;
            pts_uploaded = 0;
        }
        if (pts_uploaded < npts) {
            glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)(pts_uploaded * sizeof(Pt)), (GLsizeiptr)((npts - pts_uploaded) * sizeof(Pt)), pts + pts_uploaded);
            pts_uploaded = npts;
        }
        glUseProgram(prog_pts);
        glUniform2f(U(prog_pts, "uRes"), (float)W, (float)H);
        glUniform4f(U(prog_pts, "uRect"), rect[0], rect[1], rect[2], rect[3]);
        glUniform4f(U(prog_pts, "uRange"), (float)-ph_xr, (float)ph_xr, (float)-wr, (float)wr);
        float ps = fmaxf(fminf((float)W, (float)H) / 600.0f, 1.6f) + 1.0f;
        glUniform1f(U(prog_pts, "uSize"), ps);
        glUniform1f(U(prog_pts, "uT"), (float)scene_el);
        float keep = (float)(1 - exp(-frame_dt / trail_tau));
        glUniform1f(U(prog_pts, "uI"), keep * 0.55f * fade);
        /* each hidden pendulum's colour: its own hue while it looks regular
         * (an island, a torus), the sea's quiet tint once its shadow shows
         * it to be chaotic */
        static float tc[PH_M][4];
        double tu = (double)ph_step * PH_SPEED * 3 * cur.tscale / HZ;
        float sea[3];
        pal_acc(cur.pal, 0.12, sea);
        if (P->sinebow) { sea[0] = 0.32f; sea[1] = 0.40f; sea[2] = 0.55f; }
        for (int i = 0; i < PH_M; i++) {
            float own[3];
            pal_acc(cur.pal, 0.08 + 0.92 * ph_f[i], own);
            double lam = ph_lambda(i, &cur);
            /* until the exponent has settled, lean on how far it has got */
            float ch = (float)smooth((lam - 0.015) / 0.035) * (float)smooth(tu / 40.0);
            ch = fmaxf(ch, (float)smooth((lam - 0.05) / 0.05));
            for (int c = 0; c < 3; c++) tc[i][c] = own[c] * 1.5f * (1 - ch) + sea[c] * 0.42f * ch;
            tc[i][3] = 1;
        }
        if (!tex_tc) tex_tc = mk_tex(GL_RGBA32F, PH_M, 1, GL_NEAREST);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tc);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, PH_M, 1, GL_RGBA, GL_FLOAT, tc);
        glUniform1i(U(prog_pts, "uTC"), 0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDrawArrays(GL_POINTS, 0, npts);
    }
    final_pass(target, 1, v.cx / W, 1 - v.cy / H);
    draw_ensemble_rods(&v, fade, 1);

    /* the section's frame, the energy boundary, ticks and labels */
    float c[3] = {P->text[0], P->text[1], P->text[2]};
    float fa = 0.28f * fade;
    float x0 = rect[0], y0 = rect[1], x1 = rect[2], y1 = rect[3];
    cap_add(x0, y1, x1, y1, 0.5f, 0, 0, c, fa);
    cap_add(x0, y0, x0, y1, 0.5f, 0, 0, c, fa);
    float ym = (y0 + y1) * 0.5f;
    cap_add(x0, ym, x1, ym, 0.5f, 0, 0, c, fa * 0.4f);
    float bc[3];
    pal_disp(cur.pal, 0.2, bc);
    if (P->sinebow) memcpy(bc, c, sizeof bc);
    for (int side = 0; side < 2; side++) {
        float px = 0, py = 0;
        int had = 0;
        for (int i = 0; i <= 400; i++) {
            double a2 = ph_xr * (i / 200.0 - 1), lo, hi;
            int ok = sect_bounds(&cur.p, cur.pE, a2, &lo, &hi);
            double w = side ? hi : lo;
            float qx = x0 + (x1 - x0) * (float)(i / 400.0), qy = ym - (float)(w / wr) * (y1 - y0) * 0.5f;
            if (had && ok) cap_add(px, py, qx, qy, 0.6f, 0, 0, bc, 0.5f * fade);
            px = qx; py = qy;
            had = ok;
        }
    }
    float tk = fminf((float)W, (float)H) * 0.008f;
    TFont f = font_px(fmaxf(fminf((float)W, (float)H) * 0.0145f, 10));
    float tc[4] = {c[0], c[1], c[2], 0.6f * fade};
    static const char *lab[9] = {"−\1π", "−3\1π\2/4", "−\1π\2/2", "−\1π\2/4", "0", "\1π\2/4", "\1π\2/2", "3\1π\2/4", "\1π"};
    for (int k = -4; k <= 4; k++) {
        double a = k * PI / 4;
        if ((ph_xr > 2 && k % 2) || fabs(a) > ph_xr + 1e-6) continue;
        float x = x0 + (x1 - x0) * (float)((a + ph_xr) / (2 * ph_xr));
        cap_add(x, y1, x, y1 + tk, 0.5f, 0, 0, c, fa * 1.5f);
        text(f, x, y1 + tk * 3.4f, 0, lab[k + 4], tc, 0);
    }
    text(f, (x0 + x1) * 0.5f, y1 + tk * 6.4f, 0, "\1θ\2₂", tc, 0);
    text(f, x0 - tk * 1.5f, ym + tk * 0.6f, 1, "\1ω\2₂", tc, 0);
    text(f, x0, y0 - tk * 2.0f, -1, "\3Poincaré section   \1θ\2₁ = 0,  \1ω\2₁ > 0", tc, 0);
    /* the featured pendulum's own crossings, ringed */
    float wc[3] = {1, 0.97f, 0.9f};
    if (P->light) memcpy(wc, P->text, sizeof wc);
    for (int i = 0; i < nfpts; i++) {
        float x = x0 + (x1 - x0) * (float)((fpts[i].x + ph_xr) / (2 * ph_xr));
        float y = ym - (float)(fpts[i].y / wr) * (y1 - y0) * 0.5f;
        float age = (float)scene_el - fpts[i].born;
        if (age < 0) continue;
        float pulse = expf(-age * 1.2f);
        cap_add(x, y, x, y, 2.2f, 6 + 14 * pulse, (0.1f + 0.5f * pulse) * fade, wc, 0.95f * fade);
    }
    set_cap_place();
    char line[200];
    snprintf(line, sizeof line, "\1E\2 = %.3f   %d crossings\n\1m\2₂/\1m\2₁ = %.2f   \1ℓ\2₂/\1ℓ\2₁ = %.2f",
             cur.pE, npts, cur.p.m2 / cur.p.m1, cur.p.l2 / cur.p.l1);
    caption(wt, line, clock, fade);
    draw_caps(P->light);
    draw_text();
}

static double prof_sim, prof_draw;     /* --bench: process CPU seconds in each */
static double cputime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void render(double el, double wt, GLuint target, long keep_from) {
    ensure_size();
    double c0 = cputime();
    sim_to(el, keep_from);
    double c1 = cputime();
    prof_sim += c1 - c0;
    int ty = cur_type();
    int clock = span_me < 0 || role == SC_FAN;
    if (ty == SC_FAN) render_fan(wt, target);
    else if (ty == SC_MAP) render_map(wt, target, clock);
    else render_phase(wt, target, clock);
    prof_draw += cputime() - c1;
    last_role_drawn = ty;
}

/* frames between `from` and `to` seconds, drawn but not shown, so the
 * trails are there from the first frame shown */
static void warm_up(double el, double wt, GLuint target) {
    double from = el - 3.0, t0;
    scene_at(el, &t0);
    if (from < t0) from = t0;
    sim_to(from, -1);
    acc_clear();
    for (double t = from; t < el - 1e-9; t += 1.0 / 60) {
        frame_dt = 1.0 / 60;
        long ks = fan_step;
        render(t, wt - (el - t), target, ks);
    }
    frame_dt = 1.0 / 60;
}

/* ---- --check: the physics, verified ---- */
static int check_physics(void) {
    printf("pendulums: physics check\n");
    /* 1. the closed-form accelerations against a direct solve of
     *    M(q) q'' = f(q, q') from the Lagrangian */
    Rng r = {12345};
    double worst = 0;
    for (int t = 0; t < 100000; t++) {
        Phys p = {0.3 + 2 * rnd(&r), 0.3 + 2 * rnd(&r), 0.4 + 1.2 * rnd(&r), 0.4 + 1.2 * rnd(&r)};
        St s = {TAU * rnd(&r) - PI, TAU * rnd(&r) - PI, 6 * rnd(&r) - 3, 6 * rnd(&r) - 3}, d;
        deriv(&p, &s, &d);
        double M = p.m1 + p.m2, c = cos(s.a1 - s.a2), sn = sin(s.a1 - s.a2);
        double A = M * p.l1 * p.l1, B = p.m2 * p.l1 * p.l2 * c, C = p.m2 * p.l2 * p.l2;
        double f1 = -p.m2 * p.l1 * p.l2 * sn * s.w2 * s.w2 - M * p.l1 * sin(s.a1);
        double f2 = p.m2 * p.l1 * p.l2 * sn * s.w1 * s.w1 - p.m2 * p.l2 * sin(s.a2);
        double det = A * C - B * B;
        double x1 = (C * f1 - B * f2) / det, x2 = (A * f2 - B * f1) / det;
        double e = fmax(fabs(x1 - d.w1), fabs(x2 - d.w2)) / (1 + fabs(x1) + fabs(x2));
        if (e > worst) worst = e;
    }
    printf("  equations of motion vs mass-matrix solve, 100000 random states: worst relative difference %.2e\n", worst);

    /* 2. energy drift over a long run, at the step used on screen (h = tscale/240) and twice it */
    static const struct { double a1, a2; const char *what; } cases[] = {
        {0.4, 0.6, "low energy (quasi-periodic)"}, {1.5, 1.2, "middling"}, {2.6, 2.9, "high energy (chaotic)"},
        {3.0, -3.1, "near the top (very chaotic)"}};
    Phys pq[3] = {{1, 1, 1, 1}, {1, 2.5, 1.2, 0.8}, {1, 0.4, 0.8, 1.2}};
    for (int pi = 0; pi < 3; pi++)
        for (int c = 0; c < 4; c++) {
            printf("  m2/m1 %.1f, l2/l1 %.2f, start (%.1f, %.1f) %s:\n", pq[pi].m2, pq[pi].l2 / pq[pi].l1, cases[c].a1, cases[c].a2, cases[c].what);
            for (int hm = 1; hm <= 2; hm++) {
                double h = 2.8 / HZ * hm, T = 2.8 * 80;     /* 80 s of the most energetic time scale */
                St s = {cases[c].a1, cases[c].a2, 0, 0};
                double E0 = energy(&pq[pi], &s), me = 0;
                long n = (long)(T / h);
                for (long i = 0; i < n; i++) {
                    rk4(&pq[pi], &s, h);
                    double e = fabs(energy(&pq[pi], &s) - E0);
                    if (e > me) me = e;
                }
                printf("    h = %.4f: max |dE| / E-range over %.0f time units = %.2e\n", h, T, me / escale(&pq[pi]));
            }
        }

    /* 3. the largest Lyapunov exponent (Benettin's method) of a few starts */
    printf("  largest Lyapunov exponents (per time unit sqrt(l/g)):\n");
    Phys p1 = {1, 1, 1, 1};
    static const double starts[][2] = {{0.3, 0.4}, {1.0, 1.0}, {2.0, 2.0}, {2.6, 2.9}, {3.0, -3.1}};
    for (int i = 0; i < 5; i++) {
        St a = {starts[i][0], starts[i][1], 0, 0}, b = a;
        double d0 = 1e-9, sum = 0, h = 0.01;
        b.a2 += d0;
        long n = 0;
        for (int blk = 0; blk < 2000; blk++) {
            for (int k = 0; k < 50; k++) { rk4(&p1, &a, h); rk4(&p1, &b, h); }
            double dx[4] = {b.a1 - a.a1, b.a2 - a.a2, b.w1 - a.w1, b.w2 - a.w2};
            double d = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + dx[3] * dx[3]);
            sum += log(d / d0);
            n++;
            b = (St){a.a1 + dx[0] * d0 / d, a.a2 + dx[1] * d0 / d, a.w1 + dx[2] * d0 / d, a.w2 + dx[3] * d0 / d};
        }
        printf("    start (%.1f, %.1f), E = %+.2f: lambda = %.3f\n", starts[i][0], starts[i][1], energy(&p1, &(St){starts[i][0], starts[i][1], 0, 0}), sum / (n * 50 * h));
    }

    /* 4. determinism: the fan of the first scene integrated with 1 and 8 threads */
    Scene S;
    opt_seed = opt_seed ? opt_seed : 1;
    scene_make(0, &S);
    printf("  scene 0 (seed %u): %d pendulums, eps %.3e, fray target %.1f s\n", opt_seed, S.n, S.eps, S.ttarget);
    St *keep = malloc(sizeof(St) * (size_t)S.n);
    for (int pass = 0; pass < 2; pass++) {
        pool_start(pass ? 8 : 1);
        ens_begin(&S, S.n);
        double t0 = now();
        ens_advance(&S, 40 * HZ, -1);
        double el = now() - t0;
        if (!pass) memcpy(keep, ens, sizeof(St) * (size_t)S.n);
        else printf("  1 vs 8 threads after 40 s: %s (%.0f ms for 40 s of %d pendulums on 8 threads)\n",
                    memcmp(keep, ens, sizeof(St) * (size_t)S.n) ? "DIFFERENT" : "bit-identical", el * 1000, S.n);
        if (!pass) {
            printf("  divergence of neighbours (log10 rad), every 2 s:");
            for (int i = 7; i < nhist; i += 8) printf(" %.1f", hist_d[i]);
            printf("\n  ensemble energy error: %.2e of the energy range\n", hist_e[nhist - 1]);
        }
    }
    free(keep);
    return 0;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "pendulums");
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
        "usage: pendulums [--window | --root | --layer] [--seed N] [--fps N]\n"
        "                 [--scene cycle|fan|map|phase] [--palette spectrum|ember|ice|neon|ink]\n"
        "                 [--variant classic|low|sweep] [--count N] [--load] [--no-caption]\n"
        "                 [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                 [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check]\n");
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

/* Spanning, the output under the middle of the layout is the front one and
 * gets the fan; a portrait output gets the phase portrait, and the rest,
 * left to right, take the chaos map and the phase portrait in turn. */
static int pick_role(void) {
    if (span_me < 0) return SC_CYCLE;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    if (span_me == front) return SC_FAN;
    if (span[span_me][3] > span[span_me][2]) return SC_PHASE;
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == front || i == span_me || span[i][3] > span[i][2]) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && (span[i][1] < span[span_me][1] ||
                                               (span[i][1] == span[span_me][1] && i < span_me)))) rank++;
    }
    return rank % 2 ? SC_PHASE : SC_MAP;
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

static int pal_by_name(const char *s) {
    for (int i = 0; i < NPAL; i++)
        if (!strcmp(s, PALS[i].name)) return i;
    usage();
    return 0;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-check")) opt_check = 1;
        else if (!strcmp(a, "-load")) opt_load = 1;
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-count") && i + 1 < argc) opt_count = atoi(argv[++i]);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_pal = pal_by_name(argv[++i]);
        else if (!strcmp(a, "-scene") && i + 1 < argc) {
            const char *s = argv[++i];
            opt_scene = !strcmp(s, "fan") ? SC_FAN : !strcmp(s, "map") ? SC_MAP : !strcmp(s, "phase") ? SC_PHASE
                      : !strcmp(s, "cycle") ? SC_CYCLE : -2;
            if (opt_scene == -2) usage();
        } else if (!strcmp(a, "-variant") && i + 1 < argc) {
            const char *s = argv[++i];
            opt_variant = !strcmp(s, "classic") ? V_CLASSIC : !strcmp(s, "low") ? V_LOW : !strcmp(s, "sweep") ? V_SWEEP : -2;
            if (opt_variant == -2) usage();
        }
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
    role = pick_role();
    if (opt_scene >= 0 && opt_scene != SC_CYCLE && span_me < 0) role = opt_scene;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_check) return check_physics();
    int offscreen = opt_shot || opt_bench > 0;
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    pool_start(nc > 1 ? (int)nc : 1);
    cpu_load();
    fonts_init();

    /* --epoch: when the first scene began, shared by every instance; a
     * --shot then shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    t_start_wall = t_wall0;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;

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
        XStoreName(dpy, win, "Pendulums");
        XClassHint chint = {"pendulums", "Pendulums"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        double wt = opt_epoch ? opt_epoch + opt_at : wall();
        ensure_size();
        double t0 = now();
        warm_up(opt_at, wt, fbo);
        double tw = now() - t0;
        if (opt_shot) {
            frame_dt = 1.0 / 60;
            render(opt_at, wt, fbo, fan_step);
            glFinish();
            save_ppm(opt_shot, fbo);
            fprintf(stderr, "pendulums: scene %d (%s), %.1f of %.1f s; caught up in %.0f ms\n", cur.k,
                    cur_type() == SC_FAN ? "fan" : cur_type() == SC_MAP ? "map" : "phase", scene_el, cur.dur, tw * 1000);
            return 0;
        }
        pool_start(nc > 4 ? 4 : (int)nc);
        /* the machine may be busy: GPU time from timer queries, CPU time
         * as the work done by every thread */
        double sum = 0, worst = 0, gpu = 0, gworst = 0;
        GLuint q;
        glGenQueries(1, &q);
        struct timespec c0, c1;
        double gbest = 1e9;
        prof_sim = prof_draw = 0;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = opt_at + (i + 1) / 60.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, wt + (i + 1) / 60.0, fbo, fan_step);
            glEndQuery(GL_TIME_ELAPSED);
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            gpu += ns * 1e-6;
            if (ns * 1e-6 > gworst) gworst = ns * 1e-6;
            if (ns * 1e-6 < gbest) gbest = ns * 1e-6;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
        double cpu = ((c1.tv_sec - c0.tv_sec) + (c1.tv_nsec - c0.tv_nsec) * 1e-9) * 1000;
        printf("pendulums: %dx%d, %s, %d frames: GPU %.3f ms avg (best %.3f, worst %.3f), CPU work %.3f ms/frame "
               "(all threads; sim %.3f, draw %.3f); wall %.3f ms avg (worst %.3f); warm-up %.0f ms\n",
               W, H, cur_type() == SC_FAN ? "fan" : cur_type() == SC_MAP ? "map" : "phase", opt_bench, gpu / opt_bench, gbest,
               gworst, cpu / opt_bench, prof_sim * 1000 / opt_bench, prof_draw * 1000 / opt_bench, sum / opt_bench, worst, tw * 1000);
        printf("pendulums: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    /* live: a few cores for the ensemble, leaving the rest alone */
    double el0 = opt_epoch ? fmax(wall() - opt_epoch, 0) : 0;
    double tstart = now() - el0;
    ensure_size();
    pool_start(nc > 1 ? (int)nc : 1);
    warm_up(el0, wall(), 0);
    pool_start(nc > 4 ? 4 : (int)nc);
    double last = now();
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
        double t = now(), wt = wall();
        double el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - tstart;
        frame_dt = fmin(fmax(t - last, 1e-3), 0.25);
        last = t;
        /* fell far behind (a suspend, a stall): catch up afresh */
        if (have_cur && el - (cur.t0 + (double)fan_step / HZ) > 2.0 && cur_type() != SC_MAP) {
            pool_start(nc > 1 ? (int)nc : 1);
            warm_up(el, wt, 0);
            pool_start(nc > 4 ? 4 : (int)nc);
        }
        render(el, wt, 0, fan_step);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
