/*
 * burrow — a side-on cut-paper desert with something alive in it. Above the
 * line of the sand, dune ridges recede into haze under a paper sun (or a
 * moon and stars); below it, the cross-section: bands of sandstone with
 * their bedding, pockets of gravel, rocks, ammonites and the bones of
 * things that died when this was sea. Through it swims a great worm.
 *
 * The worm is a creature, not a tube: a body tapered from a thick forebody
 * to a fine tail, a blunt head with a jaw, an eye and trailing barbels, and
 * a smooth saddle (the clitellum) a fifth of the way back. Its rings are of
 * uneven length; they bunch up on the inside of a bend and stretch on the
 * outside, and peristaltic waves run down the body, each one fattening and
 * shortening the segments it passes. The body follows the path the head
 * dug, with a small travelling wriggle (larger in the open air).
 *
 * Each pass the worm crosses the whole wall, alternately left and right. A
 * quiet pass rises close under the surface somewhere, heaving the sand up
 * over it; every other pass it breaches, leaving the sand in a long arc
 * over the dunes with sand pouring off it and spraying from the holes,
 * jaw open, before diving back. The tunnel stays open behind it, then the
 * roof falls in and it becomes a filled burrow with crescent backfill that
 * fades away; the sand above a shallow burrow slumps. Every fourth pass the
 * worm leaves galleries above its tunnel that spell the time or the host
 * name, which collapse in their turn.
 *
 * Rendering: one full-screen pass draws the paper sheets (sky bands, sun,
 * clouds, the dune ridges, the strata and what is buried in them), each
 * cut edge throwing a soft shadow up onto the sheet behind, like papercut.
 * The ridge and strata lines are 1D functions computed on the CPU into a
 * small float texture (the surface row every frame, for the heave). The
 * burrows and the worm are triangle strips along their spines carrying
 * (along, across) body coordinates, so the fragment shader works in the
 * worm's own frame: radius, rings, markings and head are analytic there and
 * anti-aliased by distance. The falling sand is a few hundred stateless
 * grains drawn as anti-aliased capsules.
 *
 * Palettes (--palette, else picked from the seed): ochre, dusk, noon,
 * night, paper.
 *
 * Deterministic: everything is a function of --seed, the layout and the
 * time since --epoch (--schedule prints the passes; --breach jumps a --shot
 * to the top of the first breach). Spanning: one instance per monitor with
 * the same --seed, --epoch and --layout, and --output = this one's index;
 * the desert lives in wall coordinates, so it is one continuous scene and
 * the worm swims from screen to screen. The "front" output (under the
 * middle of the layout) sets the scale and the sand line, and carries the
 * caption and the galleries.
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
static const char *opt_text;           /* --text: what the host-name cuts spell instead */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the stack starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "burrow: %s\n", what); exit(1); }

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

/* ---- the live details for the caption ---- */
static char host[64];
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

/* ---- the wall ----
 * World units are logical layout pixels (y down). The domain is the layout's
 * bounding box (or, standalone, the screen); the front rect is the output
 * under the middle of the layout, and its height L sets the scale. */
static double bx0, by0, bx1, by1, bw, bh;
static double fr[4];                   /* front output: x, y, w, h */
static double L;                       /* front output height: the unit of size */

/* the output under the middle of the layout (flightsim's rule) */
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
    /* the scale: the front screen's shorter side, so a portrait front
     * output doesn't get giant shapes */
    L = fmin(fr[2], fr[3]);
}

/* the outputs to place things on: the layout, or the one screen */
static int n_outs(void) { return span_me >= 0 ? span_n : 1; }
static void out_rect(int i, double r[4]) {
    if (span_me >= 0) for (int k = 0; k < 4; k++) r[k] = span[i][k];
    else for (int k = 0; k < 4; k++) r[k] = fr[k];
}
/* ==== the desert ====
 * Y0 is the resting level of the sand on the front screen; RM the worm's
 * greatest radius and LW its length; V0 its cruising speed. */
static double Y0, RM, LW, V0, DPASS, fold_ph;
#define GAP 8.0                        /* seconds of empty desert between passes */
#define TOPEN 3.0                      /* a burrow stays open this long after the tail has left it */
#define TC 22.0                        /* then its roof falls in over this long */
#define TF 110.0                       /* and the filled burrow fades over this long */
#define LAM_P 0.34                     /* peristaltic wavelength, in body lengths */
#define PER_P 3.4                      /* its period, s */
#define TXT_OPEN 40.0                  /* the galleries stay open this long after they are dug */

static double sq(double x) { return x * x; }
static double sstep(double a, double b, double x) { double u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); }

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
static double smin(double a, double b, double k) {
    double h = clamp01(0.5 + 0.5 * (b - a) / k);
    return b + (a - b) * h - k * h * (1 - h);
}

/* a dune's cross-section: a long windward slope up to the crest at a, then
 * the steep slip face; 0 in the trough, 1 on the crest */
static double dune_shape(double ph, double a) {
    double u = ph - floor(ph);
    double w = u < a ? 0.5 * u / a : 0.5 + 0.5 * (u - a) / (1 - a);
    return 0.5 - 0.5 * cos(2 * M_PI * w);
}

/* the sand line at rest */
static double surf_base(double x) {
    double ph = x / (0.70 * L) + 0.9 * fbm1(x / (1.9 * L), 101);
    double amp = 0.040 * L * (0.40 + 0.80 * vn1(x / (1.4 * L), 102));
    double h = dune_shape(ph, 0.72) + 0.10 * dune_shape(ph * 3.1 + 0.4, 0.62);
    return Y0 - amp * (h - 0.45);
}

/* the far ridges, 0 the farthest */
#define NFAR 4
static double far_y(int k, double x) {
    double base = Y0 - L * (0.205 - 0.044 * k);
    double lam = L * (0.30 + 0.17 * k);
    double amp = L * (0.026 + 0.016 * k) * (0.45 + 0.8 * vn1(x / (1.2 * L) + k * 5.1, 110 + k));
    double ph = x / lam + 0.8 * fbm1(x / (1.6 * L) + 3.7 * k, 120 + k);
    return base - amp * dune_shape(ph, 0.66 + 0.04 * k);
}

/* the strata: the top of each band under the sand */
#define NSTRATA 9
static const double strata_d[NSTRATA] = {0.075, 0.15, 0.235, 0.33, 0.44, 0.56, 0.70, 0.86, 1.04};
static double strata_y(int j, double x) {
    double d = strata_d[j] + 0.035 * (hsh(j, 150) - 0.5) * (j > 0);
    return Y0 + d * L + (surf_base(x) - Y0) * exp(-d / 0.10) +
           L * (0.007 + 0.03 * d) * (2 * fbm1(x / (L * (0.45 + 0.4 * d)) + 9.1 * j, 130 + j) - 1) +
           L * 0.07 * d * sin(x / (2.9 * L) + 0.5 * j + fold_ph) + L * 0.025 * d * sin(x / (1.3 * L) + 0.3 * j + 2 * fold_ph);
}

/* ==== the worm ==== */
/* radius along the body (sn: 0 at the back of the head, 1 the tail tip),
 * in RM; the shader has the same function */
static double prof(double sn) {
    if (sn < 0) sn = 0;
    double r = 0.88 + 0.12 * sstep(0, 0.34, sn) + 0.07 * exp(-sq(sn / 0.012)) - 0.08 * exp(-sq((sn - 0.035) / 0.012)) + 0.035 * exp(-sq((sn - 0.08) / 0.02));
    double tl = clamp01((sn - 0.40) / 0.60);
    return r * (1 - 0.93 * pow(tl, 1.35));
}
/* the saddle, by material position */
static double clit_of(double mu) { double u = mu / LW; return sstep(0.10, 0.145, u) * (1 - sstep(0.185, 0.23, u)); }
/* the peristaltic wave: +1 where the body is contracted (short and fat) */
static double peri(double mu, double t) { return sin(2 * M_PI * mu / (LAM_P * LW) - 2 * M_PI * t / PER_P); }

/* ring number along the body: rings are of uneven length, longest in the
 * thick of the body; a material property, so they travel with it */
#define NQ 4096
static double qtab[NQ], q_mu0, q_dmu;
static void build_rings(void) {
    double seg0 = 0.85 * RM;
    q_mu0 = -4 * RM;
    q_dmu = (1.3 * LW - q_mu0) / (NQ - 1);
    double q = q_mu0 / seg0;
    for (int i = 0; i < NQ; i++) {
        double mu = q_mu0 + i * q_dmu;
        qtab[i] = q;
        double dens = mu < 0 ? 1 / seg0
                             : fmax(0.45, 1 + 1.0 * (vn1(mu / (seg0 * 1.7), 140) - 0.5) + 0.6 * (vn1(mu / (seg0 * 0.8), 141) - 0.5)) /
                                   (seg0 * (0.32 + 0.68 * prof(mu / LW)));
        q += dens * q_dmu;
    }
}
static double ring_of(double mu) {
    double u = (mu - q_mu0) / q_dmu;
    if (u <= 0) return qtab[0] + u * (qtab[1] - qtab[0]);
    if (u >= NQ - 1) return qtab[NQ - 1] + (u - (NQ - 1)) * (qtab[NQ - 1] - qtab[NQ - 2]);
    int i = (int)u;
    return qtab[i] + (qtab[i + 1] - qtab[i]) * (u - i);
}

/* ==== passes ====
 * The worm crosses the wall once per pass, along a path that is a function
 * of x (it never loops), tabulated with its arclength. */
enum { K_PLAIN, K_BREACH, K_TEXT };
enum { T_NONE, T_TIME, T_HOST };
static const char *kind_name[3] = {"quiet", "breach", "galleries"};
typedef struct {
    long p;
    int dir, kind, txt;
    double T0, v;
    double ph[2], xs;                  /* the depth's wander; where a quiet pass comes up shallow */
    double xc, hw, hh, ys;             /* a breach: centre, half width, height, sand level there */
    double x0, dx;                     /* table: entry i is at x0 + dir * i * dx */
    int n;
    double *y, *sg;
    double s_start, s_end;             /* the head's arclength at the start and at the end of the move */
    double s_up, s_down;               /* a breach: where the path leaves and re-enters the sand */
} Pass;

static int pass_kind(long p) {
    static const int order[4] = {K_PLAIN, K_BREACH, K_TEXT, K_BREACH};
    return order[(p + (opt_seed >> 3)) & 3];
}

/* the path's depth below Y0 at x */
static double path_rel(const Pass *P, double x) {
    double u = x / L;
    double dep = L * (0.25 + 0.10 * sin(2 * M_PI * u / 2.6 + P->ph[0]) + 0.06 * sin(2 * M_PI * u / 1.7 + P->ph[1]));
    if (P->kind == K_PLAIN) dep += (0.065 * L - dep) * exp(-sq((x - P->xs) / (0.62 * L)));
    if (P->kind == K_TEXT) {
        double g = sstep(fr[0] - 0.9 * L, fr[0] - 0.05 * L, x) * (1 - sstep(fr[0] + fr[2] + 0.05 * L, fr[0] + fr[2] + 0.9 * L, x));
        dep += (0.40 * L - dep) * g;
    }
    if (P->kind == K_BREACH) {
        double v = (x - P->xc) / P->hw;
        dep = smin(P->ys + P->hh * (v * v - 1), dep, 0.2 * L);
    }
    return dep;
}

static double sg_at_x(const Pass *P, double x) {
    double u = (x - P->x0) * P->dir / P->dx;
    int i = (int)floor(u);
    if (i < 0) i = 0;
    if (i > P->n - 2) i = P->n - 2;
    return P->sg[i] + (P->sg[i + 1] - P->sg[i]) * (u - i);
}

/* position (and unit tangent, in the direction of travel) at arclength s;
 * straight on past either end */
static void path_at(const Pass *P, double s, double *x, double *y, double *tx, double *ty) {
    int lo = 0, hi = P->n - 1;
    if (s <= P->sg[0]) hi = 1;
    else if (s >= P->sg[P->n - 1]) lo = P->n - 2;
    else while (hi - lo > 1) { int m = (lo + hi) / 2; if (P->sg[m] <= s) lo = m; else hi = m; }
    double ds = P->sg[hi] - P->sg[lo], f = (s - P->sg[lo]) / ds;
    double xa = P->x0 + P->dir * lo * P->dx, xb = xa + P->dir * P->dx;
    *x = xa + (xb - xa) * f;
    *y = P->y[lo] + (P->y[hi] - P->y[lo]) * f;
    if (tx) { *tx = (xb - xa) / ds; *ty = (P->y[hi] - P->y[lo]) / ds; }
}

static int pick_output(Rng *r) {
    if (n_outs() < 2) return 0;
    double tot = 0, pick;
    for (int i = 0; i < span_n; i++) tot += span[i][2];
    pick = rnd(r) * tot;
    for (int i = 0; i < span_n; i++) if ((pick -= span[i][2]) <= 0) return i;
    return span_n - 1;
}

static void build_pass(Pass *P, long p) {
    Rng r = rng_for((uint64_t)p, 31, 7);
    P->p = p;
    P->dir = ((p + (opt_seed >> 5)) & 1) ? -1 : 1;
    P->kind = pass_kind(p);
    P->txt = (((p >> 2) + opt_seed) & 1) ? T_HOST : T_TIME;
    P->T0 = p * DPASS;
    P->ph[0] = rr(&r, 0, 2 * M_PI);
    P->ph[1] = rr(&r, 0, 2 * M_PI);
    double o[4];
    out_rect(pick_output(&r), o);
    P->xs = o[0] + o[2] * rr(&r, 0.3, 0.7);
    out_rect(pick_output(&r), o);
    P->xc = o[0] + o[2] * rr(&r, 0.40, 0.60);
    P->hw = fmin(L * rr(&r, 0.50, 0.60), 0.45 * o[2]);
    P->ys = surf_base(P->xc) - Y0;
    P->hh = fmin(L * rr(&r, 0.25, 0.32), Y0 + P->ys - (o[1] + 0.13 * o[3]));
    double m = LW + 0.6 * L;
    P->dx = 0.004 * L;
    P->n = (int)((bw + 2 * m) / P->dx) + 2;
    P->x0 = P->dir > 0 ? bx0 - m : bx1 + m;
    P->y = malloc(sizeof(double) * P->n);
    P->sg = malloc(sizeof(double) * P->n);
    for (int i = 0; i < P->n; i++) {
        double x = P->x0 + P->dir * i * P->dx;
        P->y[i] = Y0 + path_rel(P, x);
        P->sg[i] = i ? P->sg[i - 1] + hypot(P->dx, P->y[i] - P->y[i - 1]) : 0;
    }
    /* the head starts just off the near edge; the move ends with the tail off the far one */
    P->s_start = sg_at_x(P, P->dir > 0 ? bx0 - 0.2 * L : bx1 + 0.2 * L);
    P->s_end = sg_at_x(P, P->dir > 0 ? bx1 + 0.2 * L : bx0 - 0.2 * L) + LW;
    P->v = (P->s_end - P->s_start) / (DPASS - GAP);
    P->s_up = P->s_down = -1;
    if (P->kind == K_BREACH) {
        int was = 0;
        for (int i = 0; i < P->n; i++) {
            double x = P->x0 + P->dir * i * P->dx;
            int above = P->y[i] < surf_base(x);
            if (above && !was && P->s_up < 0) P->s_up = P->sg[i];
            if (!above && was && P->s_up >= 0 && P->s_down < 0) P->s_down = P->sg[i];
            was = above;
        }
        if (P->s_down < 0) P->s_up = -1;
    }
}

#define NPC 4
static Pass pcache[NPC];
static long pc_used[NPC], pc_frame;    /* a pass handed out this frame is never evicted */
static Pass *pass_get(long p) {
    int best = -1;
    for (int i = 0; i < NPC; i++)
        if (pcache[i].n && pcache[i].p == p) { pc_used[i] = pc_frame; return &pcache[i]; }
    for (int i = 0; i < NPC; i++) {
        if (!pcache[i].n) { best = i; break; }
        if (pc_used[i] != pc_frame && (best < 0 || pc_used[i] < pc_used[best])) best = i;
    }
    if (best < 0) best = (int)(pc_frame % NPC);
    Pass *P = &pcache[best];
    free(P->y); free(P->sg);
    memset(P, 0, sizeof *P);
    build_pass(P, p);
    pc_used[best] = pc_frame;
    return P;
}
static double head_s(const Pass *P, double t) { return P->s_start + P->v * (t - P->T0); }

/* when a breach is at its height: the middle of the body over the peak */
static double breach_peak(const Pass *P) {
    return P->T0 + (sg_at_x(P, P->xc) + 0.45 * LW - P->s_start) / P->v;
}

/* the body this frame, sampled head to tail */
#define NB 160
typedef struct { double x, y, nx, ny, s, R, mu, Q, dq, kap, air; } BodyPt;
static BodyPt body[NB + 1];
static double mouth;

static double air_at(const Pass *P, double s) {
    double x, y;
    path_at(P, s, &x, &y, NULL, NULL);
    return surf_base(x) - y;
}

static void body_build(const Pass *P, double t) {
    double sh = head_s(P, t);
    double sA = -2.4 * RM, sB = LW + 0.25 * RM, ds = (sB - sA) / NB;
    double px[NB + 1], py[NB + 1];
    for (int i = 0; i <= NB; i++) {
        BodyPt *b = &body[i];
        double s = sA + ds * i, x, y, tx, ty;
        path_at(P, sh - s, &x, &y, &tx, &ty);
        b->s = s;
        b->air = surf_base(x) - y;
        /* a wriggle running down the body, bigger out of the sand */
        double amp = RM * (0.07 + 0.38 * clamp01(b->air / (1.5 * RM))) * sstep(0, 0.18 * LW, s) * (1 - 0.4 * s / LW);
        double off = amp * sin(2 * M_PI * s / (0.42 * LW) - 2 * M_PI * t / 3.6);
        double nx = P->dir * ty, ny = -P->dir * tx;
        px[i] = x + nx * off; py[i] = y + ny * off;
    }
    double tvx[NB + 1], tvy[NB + 1];
    for (int i = 0; i <= NB; i++) {
        int a = i > 0 ? i - 1 : 0, c = i < NB ? i + 1 : NB;
        double ex = px[a] - px[c], ey = py[a] - py[c], l = hypot(ex, ey);   /* toward the head */
        tvx[i] = ex / l; tvy[i] = ey / l;
    }
    double mu = sA;
    for (int i = 0; i <= NB; i++) {
        BodyPt *b = &body[i];
        int a = i > 0 ? i - 1 : 0, c = i < NB ? i + 1 : NB;
        b->x = px[i]; b->y = py[i];
        b->nx = P->dir * tvy[i]; b->ny = -P->dir * tvx[i];
        /* curvature toward the back (dorsal) side: + where the back is the inside */
        double kx = (tvx[a] - tvx[c]) / ((c - a) * ds), ky = (tvy[a] - tvy[c]) / ((c - a) * ds);
        b->kap = kx * b->nx + ky * b->ny;
        if (i) {
            double cc = peri(mu, t);
            mu += b->s <= 0 ? ds : ds / (1 - 0.24 * cc);
        }
        b->mu = mu;
        b->Q = ring_of(mu);
        b->R = RM * prof(b->s / LW) * (1 + 0.08 * peri(mu, t) + 0.05 * clit_of(mu));
    }
    for (int i = 0; i <= NB; i++) {
        int a = i > 0 ? i - 1 : 0, c = i < NB ? i + 1 : NB;
        body[i].dq = (body[c].Q - body[a].Q) / ((c - a) * ds);
    }
    /* the jaw opens when the head is out, and shuts a little after it goes back in */
    double a = -1e9;
    for (int k = 0; k < 4; k++) a = fmax(a, air_at(P, head_s(P, t - 0.5 * k) + 0.5 * RM));
    mouth = sstep(0.0, 1.4 * RM, a);
    /* and now and then it works its jaw underground */
    mouth = fmax(mouth, 0.22 * pow(fmax(0, sin(t * 0.55 + P->p)), 12));
}
/* ==== the cut text ==== */
static FT_Library ft;
static FT_Face face_cap, face_big;

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
    snprintf(spec, sizeof spec, "%s:weight=bold", opt_font);
    face_big = load_face(spec);
}

/* set a line of text at px into a canvas with a margin; returns it (w x h, 0..255) */
static unsigned char *set_text(const char *line, int px, int pad, int *ow, int *oh) {
    FT_Set_Pixel_Sizes(face_big, 0, (FT_UInt)px);
    int w = 0;
    for (const char *c = line; *c; c++)
        if (!FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_DEFAULT)) w += (int)(face_big->glyph->advance.x >> 6);
    int asc = (int)(face_big->size->metrics.ascender >> 6), dsc = (int)(-face_big->size->metrics.descender >> 6);
    int cw = w + 2 * pad, ch = asc + dsc + 2 * pad;
    unsigned char *cv = calloc((size_t)cw * ch, 1);
    int pen = pad, base = pad + asc;
    for (const char *c = line; *c; c++) {
        if (FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_RENDER)) continue;
        FT_GlyphSlot g = face_big->glyph;
        for (int y = 0; y < (int)g->bitmap.rows; y++)
            for (int x = 0; x < (int)g->bitmap.width; x++) {
                int X = pen + g->bitmap_left + x, Y = base - g->bitmap_top + y;
                if (X < 0 || Y < 0 || X >= cw || Y >= ch) continue;
                int v = g->bitmap.buffer[y * g->bitmap.pitch + x];
                if (v > cv[Y * cw + X]) cv[Y * cw + X] = (unsigned char)v;
            }
        pen += (int)(g->advance.x >> 6);
    }
    *ow = cw; *oh = ch;
    return cv;
}

/* squared Euclidean distance transform (Felzenszwalb & Huttenlocher), 1D */
static void edt1(const float *f, int n, float *d, int *v, float *z) {
    int k = 0;
    v[0] = 0; z[0] = -1e30f; z[1] = 1e30f;
    for (int q = 1; q < n; q++) {
        float s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        while (s <= z[k]) {
            k--;
            s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = 1e30f;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        d[q] = (q - v[k]) * (float)(q - v[k]) + f[v[k]];
    }
}
static void edt2(float *g, int w, int h) {
    int n = w > h ? w : h;
    float *f = calloc(n, sizeof(float)), *d = malloc(sizeof(float) * n), *z = malloc(sizeof(float) * (n + 1));
    int *v = malloc(sizeof(int) * n);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) f[y] = g[y * w + x];
        edt1(f, h, d, v, z);
        for (int y = 0; y < h; y++) g[y * w + x] = d[y];
    }
    for (int y = 0; y < h; y++) {
        edt1(g + (size_t)y * w, w, d, v, z);
        memcpy(g + (size_t)y * w, d, sizeof(float) * w);
    }
    free(f); free(d); free(z); free(v);
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

/* full-screen triangle */
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, 0.0, 1.0);
    });

/* the paper sheets: sky, sun or moon, clouds, the ridges, the sand and the
 * strata, what is buried in them and the galleries. Sheet k is in front of
 * sheet k-1; every cut edge throws a soft shadow up onto the sheets behind. */
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform sampler2D uRows, uNoise, uPaper, uText;
    uniform vec3 uView, uRowX;
    uniform vec2 uRes;
    uniform float uL, uY0, uT, uGap, uSoft0, uSoft1, uShadow, uEmb, uStars, uSeedF;
    uniform vec3 uSky0, uSky1, uSun, uCloudC, uBone, uRock, uTunC;
    uniform vec4 uSunP;
    uniform vec3 uCol[15];
    uniform vec4 uObj[32];
    uniform vec4 uObjB[32];
    uniform int uNObj;
    uniform vec4 uCloud[6];
    uniform int uNCloud;
    uniform vec4 uTBox, uTxt;
    uniform float uTHw, uTO, uTC, uTF;
    uniform vec3 uLamp;
    uniform float uObjY;
    out vec4 o;

    float vn(vec2 p) { return texture(uNoise, p * (1.0 / 64.0)).g; }
    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
    float h11(float n) { return fract(sin(n * 91.7 + uSeedF) * 43758.5453); }
    vec2 rot(vec2 p, float a) { float c = cos(a), s = sin(a); return vec2(c * p.x - s * p.y, s * p.x + c * p.y); }
    float sq(float x) { return x * x; }
    float sdSeg(vec2 p, vec2 a, vec2 b) {
        vec2 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h);
    }

    vec3 paper(vec3 c, vec2 pp) {
        vec4 t0 = texture(uPaper, pp * (1.0 / 1024.0));
        float h1 = texture(uPaper, (pp + vec2(0.5, 1.1)) * (1.0 / 1024.0)).r;
        return c * (0.97 + 0.06 * t0.g) * (1.0 + uEmb * (t0.r - h1)) * (0.985 + 0.03 * t0.r);
    }

    /* a cut-paper cloud: a few round puffs on a flat bottom */
    float sdCloud(vec2 q, vec4 C) {
        float rr[5] = float[5](0.55, 0.85, 1.05, 0.78, 0.5);
        float d = 1e5;
        for (int j = 0; j < 5; j++) {
            float r = C.w * rr[j];
            d = min(d, length(q - vec2((-0.78 + 0.39 * float(j)) * C.z, -0.35 * r)) - r);
        }
        vec2 a = vec2(-0.85 * C.z, -0.32 * C.w), b = vec2(0.85 * C.z, -0.32 * C.w);
        d = min(d, sdSeg(q, a, b) - 0.42 * C.w);
        return max(d, q.y);
    }

    vec3 skyCol(vec2 p, float px) {
        float h = (uY0 - p.y) / (0.62 * uL);
        /* stepped bands, gently curved, each throwing a shadow up onto the next */
        float bs = 0.62 * uL / 4.5;
        float hb = h * 4.5 + 0.22 * sin(p.x / (1.3 * uL) + 1.7) + 0.12 * sin(p.x / (0.55 * uL) + 0.4);
        float b = floor(hb), f = hb - b;
        vec3 c0 = mix(uSky1, uSky0, clamp((b - 1.0) / 4.0, 0.0, 1.0));
        vec3 c = mix(uSky1, uSky0, clamp(b / 4.0, 0.0, 1.0));
        c = mix(c0, c, clamp(f * bs / px + 0.5, 0.0, 1.0));
        c *= 1.0 - 0.35 * uShadow * exp(-f * bs / (uGap * 1.4)) * step(0.5, b);
        vec2 sv = p - uSunP.xy;
        float r = length(sv), R0 = uSunP.z;
        if (uSunP.w < 0.5) {
            /* the sun, cut in three rings */
            for (int i = 2; i >= 0; i--) {
                float Ri = R0 * (1.0 + 0.6 * float(i));
                vec3 ci = mix(c, uSun, i == 0 ? 1.0 : (i == 1 ? 0.42 : 0.2));
                float rs = length(sv + vec2(0.0, uGap * 0.9));
                c *= 1.0 - 0.55 * uShadow * smoothstep(1.2 * uGap, -0.6 * uGap, rs - Ri) * clamp((r - Ri) / px + 0.5, 0.0, 1.0);
                c = mix(c, ci, clamp((Ri - r) / px + 0.5, 0.0, 1.0));
            }
        } else {
            /* stars */
            vec2 cell = floor(p / (0.03 * uL));
            float hs = h21(cell + 7.1);
            if (hs < 0.34) {
                vec2 sp = (cell + 0.15 + 0.7 * vec2(h21(cell + 1.3), h21(cell + 2.9))) * 0.03 * uL;
                float sr = uL * (0.0009 + 0.0024 * pow(h21(cell + 5.5), 3.0));
                float tw = 0.7 + 0.3 * sin(hs * 40.0);
                float scv = clamp((sr - length(p - sp)) / px + 0.5, 0.0, 1.0);
                c = mix(c, uSun, uStars * scv * tw * smoothstep(0.15, 0.4, h) * smoothstep(R0 * 2.5, R0 * 4.0, r));
            }
            /* a crescent moon and its halo */
            float rh = R0 * 1.7;
            c = mix(c, mix(c, uSun, 0.10), clamp((rh - r) / px + 0.5, 0.0, 1.0));
            float dm = r - R0, dc = length(sv - vec2(0.36, -0.14) * R0) - 0.9 * R0;
            float dd = max(dm, -dc);
            float sh = max(length(sv + vec2(0.0, uGap)) - R0, -(length(sv + vec2(0.0, uGap) - vec2(0.36, -0.14) * R0) - 0.9 * R0));
            c *= 1.0 - 0.6 * uShadow * smoothstep(uGap, -0.5 * uGap, sh) * clamp(dd / px + 0.5, 0.0, 1.0);
            c = mix(c, mix(c, uSun, 0.16), clamp(-dm / px + 0.5, 0.0, 1.0));
            c = mix(c, uSun, clamp(-dd / px + 0.5, 0.0, 1.0));
        }
        for (int i = 0; i < uNCloud; i++) {
            vec4 C = uCloud[i];
            vec2 q = p - C.xy;
            if (abs(q.x) > C.z * 1.4 || q.y > uGap * 3.0 || q.y < -C.w * 2.8) continue;
            float dsh = sdCloud(q + vec2(0.0, uGap * 1.2), C);
            float d = sdCloud(q, C);
            c *= 1.0 - 0.75 * uShadow * smoothstep(1.4 * uGap, -0.5 * uGap, dsh) * clamp(d / px + 0.5, 0.0, 1.0);
            vec3 cc = uCloudC * (1.0 - 0.06 * smoothstep(-0.2 * C.w, 0.0, q.y));
            c = mix(c, cc, clamp(0.5 - d / px, 0.0, 1.0));
        }
        return c;
    }

    /* the bedding inside a band of sandstone */
    vec3 strataDetail(int k, vec2 p, float top, float bot, vec3 c, float px) {
        float z = p.y - top, th = max(bot - top, 1.0);
        float hk = h11(float(k));
        float lw = 0.0007 * uL;
        if (hk < 0.3) {
            /* fine wavy laminae */
            float sp = uL * 0.012;
            float lam = (z + 0.010 * uL * sin(p.x / (0.37 * uL) + float(k)) + 0.006 * uL * sin(p.x / (0.13 * uL))) / sp;
            float dl = abs(fract(lam) - 0.5) * sp;
            c *= 1.0 - 0.075 * (1.0 - smoothstep(lw, lw + 1.2 * px, dl));
        } else if (hk < 0.62) {
            /* trough cross-bedding: nested scoops, cut off by the set above */
            float mid = th * (0.45 + 0.1 * sin(p.x / (0.9 * uL) + float(k)));
            float set = z < mid ? 0.0 : 1.0;
            float zz = z - (set > 0.5 ? mid : 0.0);
            float cw = uL * (0.30 + 0.12 * h11(float(k) + 4.0));
            float xc = p.x / cw + set * 0.47 + float(k) * 0.31;
            float cell = floor(xc), xl = (fract(xc) - 0.5) * cw;
            float sp = uL * 0.013;
            float lam = (zz + 0.9 * xl * xl / cw - 0.25 * xl * (h11(cell + float(k)) - 0.5)) / sp;
            float dl = abs(fract(lam) - 0.5) * sp;
            c *= 1.0 - 0.08 * (1.0 - smoothstep(lw, lw + 1.2 * px, dl));
            c *= 1.0 - 0.10 * (1.0 - smoothstep(lw * 1.4, lw * 1.4 + 1.2 * px, abs(z - mid)));
        } else if (hk < 0.82) {
            /* pebbly */
            vec2 cp0 = floor(p / (0.016 * uL));
            float hp = h21(cp0 + float(k) * 3.1);
            if (hp < 0.4) {
                vec2 cp = (cp0 + 0.25 + 0.5 * vec2(h21(cp0 + 1.7), h21(cp0 + 4.2))) * 0.016 * uL;
                float pr = uL * (0.0022 + 0.0042 * h21(cp0 + 9.3));
                vec2 dq = p - cp;
                dq.y *= 1.35;
                float dp = length(dq) - pr;
                float cov = clamp(0.5 - dp / px, 0.0, 1.0);
                c = mix(c, c * (0.80 + 0.32 * h21(cp0 + 2.2)), cov);
            }
        }
        /* and a speckle of grains everywhere */
        vec2 cs = floor(p / (0.006 * uL));
        float hs = h21(cs + float(k) * 1.31);
        if (hs < 0.16) {
            vec2 sp2 = (cs + 0.2 + 0.6 * vec2(h21(cs + 3.3), h21(cs + 6.1))) * 0.006 * uL;
            float sd = length(p - sp2) - uL * 0.0009;
            c *= 1.0 - (hs < 0.08 ? 0.16 : -0.10) * clamp(0.5 - sd / px, 0.0, 1.0);
        }
        return c;
    }

    /* sheet k at p: yk its cut, yb the next cut down, yf the three cuts in
     * front of it (their shadows) */
    float ysRest;
    vec3 shade(int k, vec2 p, float px, float yk, float yb, vec3 yf) {
        vec3 c = k == 0 ? skyCol(p, px) : uCol[k];
        if (k >= 5) c = strataDetail(k, p, k == 5 ? ysRest : yk, yb, c, px);
        c = paper(c, p + vec2(float(k) * 137.3, float(k) * 71.9));
        /* shadows thrown up by the sheets in front */
        float lit = 1.0;
        for (int j = 0; j < 3; j++) {
            float gj = float(j + 1);
            float w = uSoft0 + uSoft1 * gj;
            float s = smoothstep(-w, w * 0.5, p.y - (yf[j] - uGap * gj));
            lit *= 1.0 - uShadow * s / (1.0 + 0.3 * (gj - 1.0));
            lit *= 1.0 - 0.15 * exp(min(p.y - yf[j], 0.0) / (0.7 * w)) / (1.0 + 0.6 * (gj - 1.0));
        }
        /* a pale cut edge along the top of the sheet */
        if (k > 0) {
            float e = 1.0 - smoothstep(0.3 * px, 2.2 * px, p.y - yk);
            c += e * 0.07 * (0.5 + 0.5 * c);
        }
        return c * lit;
    }

    /* buried things: 0 rock, 1 pocket of gravel, 2 ammonite, 3 fish, 4 the ribs
     * of something big. d < 0 inside; ink: the lines drawn on it */
    float objSD(int ty, vec2 q, float s, vec4 B, float px, out float ink) {
        ink = 0.0;
        if (ty == 0 || ty == 1) {
            float asp = B.z;
            vec2 r = vec2(q.x, q.y * asp);
            float a = atan(r.y, r.x);
            float rad = s * (1.0 + 0.10 * sin(3.0 * a + B.y) + 0.06 * sin(5.0 * a + 2.0 * B.y) + 0.035 * sin(9.0 * a + 3.1 * B.y));
            return (length(r) - rad) / asp;
        }
        if (ty == 2) {
            float r = length(q), a = atan(q.y, q.x);
            float t = log(max(r, 1e-3) / s) / log(0.5) + a / 6.2831853;
            float dw = (0.5 - abs(fract(t) - 0.5)) * r * 0.693;
            float rb = (0.5 - abs(fract(a * 30.0 / 6.2831853 - 1.5 * log(max(r, 1e-3) / s)) - 0.5)) * r * 6.2831853 / 30.0;
            ink = max(1.0 - smoothstep(0.012 * s, 0.012 * s + 1.2 * px, dw),
                      0.45 * (1.0 - smoothstep(0.006 * s, 0.006 * s + 1.2 * px, rb)) * smoothstep(0.18 * s, 0.3 * s, r));
            return r - s;
        }
        if (ty == 3) {
            float d1 = max(abs(q.y - 0.02 * s * sin(q.x / s * 3.0)) - 0.016 * s, abs(q.x + 0.12 * s) - 0.74 * s);
            float sp = 0.075 * s;
            float xr = q.x + 0.5 * abs(q.y);
            float fr2 = abs(fract(xr / sp) - 0.5) * sp;
            float hgt = 0.30 * s * sqrt(max(0.0, 1.0 - sq(q.x / (0.62 * s))));
            float d2 = max(max(fr2 - 0.010 * s, abs(q.y) - hgt), abs(q.x + 0.03 * s) - 0.53 * s);
            vec2 kq = (q - vec2(0.76 * s, 0.0)) / vec2(0.21, 0.16);
            float dk = (length(kq) - s) * 0.16;
            dk = max(dk, -(length(q - vec2(0.83 * s, -0.04 * s)) - 0.045 * s));
            float dt = min(sdSeg(q, vec2(-0.84 * s, 0.0), vec2(-1.0 * s, -0.2 * s)), sdSeg(q, vec2(-0.84 * s, 0.0), vec2(-1.0 * s, 0.2 * s))) - 0.014 * s;
            return min(min(d1, d2), min(dk, dt));
        }
        /* the ribcage: an arched spine of vertebrae, ribs hanging from it, a skull */
        float xs = q.x / s;
        float ysp = -0.10 * s * (1.0 - xs * xs);
        float dy = q.y - ysp;
        float vsp = 0.06 * s;
        float fv = abs(fract(q.x / vsp) - 0.5) * vsp;
        float hv = 0.032 * s * clamp((xs + 1.7) / 1.0, 0.3, 1.0);
        float dv = max(max(abs(dy) - hv, fv - 0.36 * vsp), max(q.x - 0.9 * s, -q.x - 1.6 * s));
        float rsp = 0.09 * s;
        float xr = q.x + 0.28 * dy - 1.1 * dy * dy / s;
        float cell = floor(xr / rsp);
        float xcr = (cell + 0.5) * rsp / s;
        float lr = 0.44 * s * sqrt(max(0.0, 1.0 - sq((xcr - 0.12) / 0.62)));
        float frr = abs(fract(xr / rsp) - 0.5) * rsp;
        float dr = max(frr - 0.017 * s * (1.0 - 0.6 * clamp(dy / max(lr, 1e-3), 0.0, 1.0)), max(-dy, dy - lr));
        dr = max(dr, abs(xcr - 0.12) > 0.6 ? 1e5 : -1e5);
        vec2 kq = q - vec2(1.12 * s, 0.02 * s);
        kq = rot(kq, 0.25);
        float dk = (length(kq / vec2(0.26, 0.12)) - s) * 0.12;
        dk = max(dk, -(length(kq - vec2(0.06 * s, -0.025 * s)) - 0.035 * s));
        dk = max(dk, -max(abs(kq.y - 0.045 * s) - 0.008 * s, -kq.x - 0.1 * s));
        ink = 0.0;
        return min(min(dv, dr), dk);
    }

    void main() {
        vec2 dev = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 p = uView.xy + dev / uView.z;
        float px = 1.0 / uView.z;
        float u = ((p.x - uRowX.x) / uRowX.y + 0.5) / uRowX.z;
        vec4 A = texture(uRows, vec2(u, 0.125)), Bv = texture(uRows, vec2(u, 0.375));
        vec4 Cv = texture(uRows, vec2(u, 0.625)), Dv = texture(uRows, vec2(u, 0.875));
        ysRest = Dv.w;
        float ys[15];
        ys[0] = -1e6;
        ys[1] = A.x; ys[2] = A.y; ys[3] = A.z; ys[4] = A.w;
        ys[5] = Bv.x; ys[6] = Bv.y; ys[7] = Bv.z; ys[8] = Bv.w;
        ys[9] = Cv.x; ys[10] = Cv.y; ys[11] = Cv.z; ys[12] = Cv.w;
        ys[13] = Dv.x; ys[14] = Dv.y;
        /* a scalpel's wobble along every cut */
        float w1 = vn(vec2(p.x / (uL * 0.004), 3.7)) - 0.5, w2 = vn(vec2(p.x / (uL * 0.03), 9.1)) - 0.5;
        float sl[15];
        for (int k = 1; k < 15; k++) {
            float fk = float(k);
            ys[k] += uL * (0.0009 * w1 * sin(fk * 2.1 + 0.5) + 0.0022 * w2 * cos(fk * 1.3 + 0.2));
            sl[k] = dFdx(ys[k]) * uView.z;
        }
        ys[0] = -1e6;
        float ysx[18];
        for (int k = 0; k < 15; k++) ysx[k] = ys[k];
        ysx[15] = ys[14] + 0.3 * uL; ysx[16] = 1e6; ysx[17] = 1e6;
        /* the sheet on top, and on its anti-aliased edge the one behind */
        int top = 0;
        for (int k = 14; k >= 1; k--) if (p.y > ys[k] - 1.5 * px) { top = k; break; }
        float ct = top == 0 ? 1.0 : clamp((p.y - ys[top]) * uView.z / sqrt(1.0 + sl[top] * sl[top]) + 0.5, 0.0, 1.0);
        vec3 col = shade(top, p, px, ysx[top], ysx[top + 1], vec3(ysx[top + 1], ysx[top + 2], ysx[top + 3]));
        if (ct < 1.0) {
            int nx = 0;
            for (int k = 14; k >= 1; k--) if (k < top && p.y > ys[k] - 1.5 * px) { nx = k; break; }
            col = mix(shade(nx, p, px, ysx[nx], ysx[nx + 1], vec3(ysx[nx + 1], ysx[nx + 2], ysx[nx + 3])), col, ct);
        }
        /* what is buried, lying on the cut face */
        if (p.y > ys[5] && p.y > uObjY) {
            for (int i = 0; i < 32; i++) {
                if (i >= uNObj) break;
                vec4 O = uObj[i], B = uObjB[i];
                vec2 q = p - O.xy;
                float bound = O.z * (O.w > 3.5 ? 1.75 : 1.35) + uGap * 2.0;
                if (dot(q, q) > bound * bound) continue;
                int ty = int(O.w + 0.5);
                float ink, ink2;
                vec2 ql = rot(q, -B.x);
                float d = objSD(ty, ql, O.z, B, px, ink);
                float ds = objSD(ty, rot(q + vec2(0.0, uGap * 1.2), -B.x), O.z, B, px, ink2);
                float cov = clamp(0.5 - d / px, 0.0, 1.0) * clamp((p.y - ys[5]) / px - 2.0, 0.0, 1.0);
                col *= 1.0 - 0.8 * uShadow * smoothstep(uGap, -0.5 * uGap, ds) * (1.0 - cov);
                vec3 oc;
                if (ty == 0) {
                    float lit = dot(normalize(ql + 1e-4), normalize(vec2(-0.4, -1.0)));
                    oc = uRock * (0.85 + 0.3 * B.w) * (1.0 + 0.16 * smoothstep(0.2, 0.6, lit) * smoothstep(-0.25 * O.z, -0.05 * O.z, -length(ql) + 0.0 * O.z + 0.6 * O.z));
                } else if (ty == 1) {
                    oc = col * 1.10 + 0.02;
                    vec2 gc = floor(p / (0.006 * uL));
                    float gh = h21(gc + 3.0);
                    vec2 gp = (gc + 0.5) * 0.006 * uL;
                    oc *= 1.0 - 0.35 * step(gh, 0.45) * clamp(0.5 - (length(p - gp) - 0.0016 * uL) / px, 0.0, 1.0);
                } else {
                    oc = uBone * (0.9 + 0.1 * B.w);
                    oc *= 1.0 - 0.45 * ink;
                    oc *= 0.92 + 0.1 * smoothstep(0.0, -0.01 * uL, d);
                }
                oc = paper(oc, p * 1.3 + vec2(float(i) * 31.0));
                col = mix(col, oc, cov);
            }
        }
        /* the galleries: letters dug out above the tunnel, falling in later */
        if (uTxt.x > 0.5) {
            vec2 tu = (p - uTBox.xy) / uTBox.zw;
            float age = (uTxt.y - p.x) * uTxt.z / uTxt.w;
            if (tu.x > 0.0 && tu.y > 0.0 && tu.x < 1.0 && tu.y < 1.0 && age > 0.0) {
                float sd = texture(uText, tu).r;
                float sds = texture(uText, (p + vec2(0.0, uGap * 1.3) - uTBox.xy) / uTBox.zw).r;
                float at = age - uTO;
                float k = smoothstep(0.0, uTC, at);
                float fade = 1.0 - smoothstep(uTC, uTC + uTF, at);
                /* dug out over a few seconds, then filled with sand from the top down */
                float grow = (uTHw + 2.0) * (1.0 - smoothstep(0.0, 3.0, age));
                float inl = clamp(0.5 - (sd + grow) / px, 0.0, 1.0);
                float fy = uTBox.y + uTBox.w * (0.12 + 0.8 * k) + 0.012 * uL * (vn(vec2(p.x / (0.012 * uL), 4.3)) - 0.5);
                float fill = k > 0.0 ? clamp((fy - p.y) / px + 0.5, 0.0, 1.0) : 0.0;
                float hole = inl * (1.0 - fill);
                col = mix(col, col * 1.14 + 0.025, 0.75 * inl * fill * fade);
                /* the hole: the back wall in shade, darkest under the lower rim;
                 * a pale cut edge round it */
                float sh = smoothstep(-px, 0.9 * uGap, sds + grow);
                vec3 hc = mix(col, uTunC, 0.72) * (1.0 - 0.45 * sh);
                hc += 0.10 * (1.0 - smoothstep(0.4 * px, 2.0 * px, -(sd + grow))) * (1.0 - sh);
                col = mix(col, hc, hole);
            }
        }
        float lamp = 1.03 - uLamp.z * length((p - uLamp.xy) / uL);
        col *= lamp;
        col += (h21(gl_FragCoord.xy) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });

/* the burrow strips and the worm: world position plus the body frame */
static const char *VS_STRIP = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec4 aA;
    layout(location = 2) in vec2 aB;
    uniform vec3 uView;
    uniform vec2 uRes, uShift;
    out vec4 vA;
    out vec2 vB;
    out vec2 vW;
    void main() {
        vA = aA; vB = aB; vW = aPos + uShift;
        vec2 d = (aPos + uShift - uView.xy) * uView.z;
        gl_Position = vec4(d.x / uRes.x * 2.0 - 1.0, 1.0 - d.y / uRes.y * 2.0, 0.0, 1.0);
    });

/* a burrow: open behind the worm, its roof falling in, then a filled burrow
 * with crescent backfill that fades. (age, across, radius, arclength) */
static const char *FS_TUN = "#version 330 core\n" GLSL(
    uniform sampler2D uRows, uNoise;
    uniform vec3 uRowX, uTunC;
    uniform float uTB, uTO, uTC, uTF, uPx, uRM, uGap;
    in vec4 vA;
    in vec2 vB;
    in vec2 vW;
    out vec4 o;
    void main() {
        float ys = texture(uRows, vec2(((vW.x - uRowX.x) / uRowX.y + 0.5) / uRowX.z, 0.375)).x;
        float under = clamp((vW.y - ys) / uPx - 0.5, 0.0, 1.0);
        if (under <= 0.0) discard;
        float age = vA.x, v = vA.y, Rt = vA.z, sg = vA.w;
        float at = age - uTB - uTO;
        float nz = texture(uNoise, vec2(sg / (uRM * 3.0), 0.37) * (1.0 / 16.0)).r;
        float k = smoothstep(0.0, uTC, at + 8.0 * (nz - 0.5));
        float roof = Rt * (1.0 - 2.2 * k) * (1.0 + 0.08 * (nz - 0.5)), flo = -Rt * (1.0 - 0.4 * k);
        float dv = max(v - roof, flo - v);
        float cv = clamp(0.5 - dv / uPx, 0.0, 1.0);
        /* the hole's lower rim throws its shadow up the back wall */
        float shd = smoothstep(uGap * 1.5, 0.0, v - flo);
        vec3 hole = uTunC * (1.0 - 0.45 * shd) * (0.9 + 0.2 * nz);
        /* the filled burrow: paler sand with crescent laminae */
        float fade = 1.0 - smoothstep(uTC, uTC + uTF, at);
        float Rf = uRM * (1.0 + 0.10 * (nz - 0.5));
        float inb = clamp(0.5 - (abs(v) - Rf) / uPx, 0.0, 1.0) * smoothstep(-6.0, 6.0, at) * fade;
        float aV = 0.7 * cv;
        float aB = inb * 0.075 * (0.8 + 0.4 * nz);
        vec3 cB = vec3(1.0, 0.97, 0.9);
        float a = aV + aB * (1.0 - aV);
        vec3 col = (hole * aV + cB * aB * (1.0 - aV)) / max(a, 1e-4);
        o = vec4(col, a * under);
    });

/* the worm, in its own frame: s along (0 at the back of the head, + toward
 * the tail), v across (+ its back), mu material position, Q ring number */
static const char *FS_WORM = "#version 330 core\n" GLSL(
    uniform sampler2D uPaper, uNoise;
    uniform float uLW, uRM, uT, uLam, uPer, uMouth, uPx, uMode, uShA, uNight, uEmb;
    uniform vec3 uBack, uBelly, uMark, uClit, uThroat, uBone, uShC, uGlow;
    in vec4 vA;
    in vec2 vB;
    in vec2 vW;
    out vec4 o;

    float sq(float x) { return x * x; }
    float prof(float sn) {
        sn = max(sn, 0.0);
        float r = 0.88 + 0.12 * smoothstep(0.0, 0.34, sn) + 0.07 * exp(-sq(sn / 0.012)) - 0.08 * exp(-sq((sn - 0.035) / 0.012)) + 0.035 * exp(-sq((sn - 0.08) / 0.02));
        float tl = clamp((sn - 0.40) / 0.60, 0.0, 1.0);
        return r * (1.0 - 0.93 * pow(tl, 1.35));
    }
    float h11(float n) { return fract(sin(n * 127.1 + 0.37) * 43758.5453); }
    vec2 rot(vec2 p, float a) { float c = cos(a), s = sin(a); return vec2(c * p.x - s * p.y, s * p.x + c * p.y); }
    /* the closed head, x forward from the back of the head */
    float headSD(vec2 q, float Rh) {
        float brow = 0.07 * Rh * exp(-sq((q.x / Rh + 0.25) / 0.4)) * smoothstep(0.0, 0.5 * Rh, q.y);
        if (q.x <= 0.0) return abs(q.y) - Rh - brow;
        float ax = 1.3 * Rh;
        float y = q.y + 0.16 * Rh * sq(q.x / ax);
        return (length(vec2(q.x / ax, y / Rh)) - 1.0) * Rh * 0.92 - brow;
    }
    float aa(float d, float w) { return clamp(0.5 - d / w, 0.0, 1.0); }

    void main() {
        float s = vA.x, v = vA.y, mu = vA.z, Q = vA.w, kap = vB.x, dqs = max(vB.y, 1e-6);
        float segl = 1.0 / dqs;
        float c = sin(6.2831853 * mu / (uLam * uLW) - 6.2831853 * uT / uPer);
        float un = mu / uLW;
        float clit = smoothstep(0.10, 0.145, un) * (1.0 - smoothstep(0.185, 0.23, un));
        float R = uRM * prof(s / uLW) * (1.0 + 0.08 * c + 0.05 * clit);
        float Rh = uRM * prof(0.0);
        float nv0 = clamp(v / R, -1.0, 1.0);
        /* the rings are seen a little from above, so each is an arc */
        float qb = Q - 0.22 * R * dqs * sqrt(max(1.0 - nv0 * nv0, 0.0));
        float k = floor(qb), f = qb - k;
        float hk = h11(k), hk2 = h11(k + 71.3), hk3 = h11(k + 13.7);
        float ins = clamp(kap * v * 6.0, -1.0, 1.0);
        float bul = sin(3.14159265 * f);
        float body = smoothstep(0.5 * uRM, 2.0 * uRM, s) * (1.0 - smoothstep(0.85, 1.0, s / uLW));
        float notch = (0.04 + 0.05 * hk2) * (1.0 + 1.6 * max(ins, 0.0) - 0.6 * max(-ins, 0.0)) * (1.0 - 0.8 * clit) * body;
        float sc = 1.0 + notch * (pow(max(bul, 0.0), 0.35) - 0.85);
        float dBody = s > uLW ? length(vec2(s - uLW, v)) - R : abs(v) - R * sc;

        /* the head: a blunt snout split by a jaw that opens */
        vec2 q = vec2(-s, v);
        float dClosed = s > 0.0 ? dBody : headSD(q, Rh);
        float th = uMouth * 0.62;
        vec2 Hj = vec2(0.10 * Rh, -0.14 * Rh);
        float dBack = max(dClosed, q.x - Hj.x);
        vec2 qu = Hj + rot(q - Hj, -th), ql = Hj + rot(q - Hj, th);
        /* the pieces overlap a little, or the seams between them would show */
        float ov = 0.04 * Rh * (1.0 - smoothstep(0.02, 0.08, th));
        float dU = max(headSD(qu, Rh), max(Hj.y - ov - qu.y, Hj.x - 0.2 * Rh - qu.x));
        float dL = max(headSD(ql, Rh), max(ql.y - Hj.y - ov, Hj.x - 0.2 * Rh - ql.x));
        float dSkin = min(dBack, min(dU, dL));
        vec2 rq = q - Hj;
        float dThroat = 1e5, dTU = 1e5, dTL = 1e5;
        if (th > 0.02) {
            float sU = dot(rq, vec2(-sin(th), cos(th))), sL = dot(rq, vec2(-sin(th), -cos(th)));
            dThroat = max(max(max(sU, sL), length(rq) - 1.08 * Rh), -rq.x - 0.02 * Rh);
            float tw = 0.16 * Rh;
            float xu = qu.x - Hj.x, yu = Hj.y - qu.y;
            float tlen = 0.22 * Rh * smoothstep(0.08 * Rh, 0.3 * Rh, xu) * (1.0 - smoothstep(0.95 * Rh, 1.15 * Rh, xu));
            float tri = abs(fract(xu / tw + 0.7 * yu / tw) - 0.5) * 2.0;
            dTU = max(max(-yu, yu - tlen * (1.0 - tri)), dThroat - 0.01 * Rh);
            float xl = ql.x - Hj.x, yl = ql.y - Hj.y;
            float tlen2 = 0.18 * Rh * smoothstep(0.12 * Rh, 0.3 * Rh, xl) * (1.0 - smoothstep(0.85 * Rh, 1.05 * Rh, xl));
            float tri2 = abs(fract(xl / tw + 0.33 + 0.7 * yl / tw) - 0.5) * 2.0;
            dTL = max(max(-yl, yl - tlen2 * (1.0 - tri2)), dThroat - 0.01 * Rh);
        }
        /* two short feelers under the chin, curling forward */
        float dBarb = 1e5;
        if (s < 2.0 * Rh) {
            for (int i = 0; i < 2; i++) {
                float bx = (i == 0 ? 0.42 : 0.78) * Rh;
                float by = -sqrt(max(1.0 - sq(bx / (1.3 * Rh)), 0.0)) * Rh - 0.16 * Rh * sq(bx / (1.3 * Rh)) + 0.06 * Rh;
                float len = (i == 0 ? 0.85 : 0.6) * Rh;
                float Y = (by - ql.y) / Rh;
                float sw = 0.3 * sin(uT * 1.7 + float(i) * 2.3 - Y * 3.0);
                float cu = (i == 0 ? 1.3 : 1.0) + sw;
                float xb = bx + (-0.15 * Y + cu * Y * Y) * Rh;
                float slp = -0.15 + 2.0 * cu * Y;
                float wb = Rh * (0.05 - 0.04 * clamp(Y * Rh / len, 0.0, 1.0)) + 0.3 * uPx;
                float db = abs(ql.x - xb) / sqrt(1.0 + slp * slp) - wb;
                dBarb = min(dBarb, max(db, max(-Y * Rh, Y * Rh - len)));
            }
        }
        float dTeeth = min(dTU, dTL);
        float d = min(min(dSkin, dThroat), min(dTeeth, dBarb));
        if (uMode > 0.5) {
            o = vec4(uShC, uShA * smoothstep(0.35 * uRM, -0.3 * uRM, min(min(dSkin, dThroat), dBarb + 0.1 * uRM)));
            return;
        }
        float w = clamp(fwidth(d), 0.6 * uPx, 3.0 * uPx);
        float cov = aa(d, w);
        if (cov <= 0.0) discard;

        /* the skin: countershaded, a dark mottle over the back */
        float nv = clamp(v / (R * sc), -1.0, 1.0);
        vec3 skin = mix(uBelly, uBack, smoothstep(-0.55, 0.10, nv));
        float nm = texture(uNoise, vec2(mu / (uRM * 2.4), nv * 0.5 + 0.37) * (1.0 / 16.0)).r;
        float blot = smoothstep(0.50, 0.57, nm + 0.22 * (hk - 0.5));
        blot *= smoothstep(-0.35, 0.15, nv) * (1.0 - clit) * smoothstep(1.2 * uRM, 3.0 * uRM, s);
        skin = mix(skin, uMark, 0.78 * blot);
        float nm2 = texture(uNoise, vec2(mu / (uRM * 0.7), nv * 1.3 + 5.1) * (1.0 / 16.0)).r;
        skin = mix(skin, uMark, 0.35 * smoothstep(0.62, 0.68, nm2) * smoothstep(-0.7, -0.1, nv) * (1.0 - blot) * (1.0 - clit));
        skin = mix(skin, uClit * (0.92 + 0.12 * smoothstep(-0.8, 0.4, nv)), 0.92 * clit);
        /* light from above, a sheen along the upper flank, the belly in shade */
        float sh = 0.70 + 0.36 * smoothstep(-0.95, 0.55, nv);
        sh += 0.09 * exp(-sq((nv - 0.45) / 0.2));
        sh *= 1.0 - 0.18 * smoothstep(0.7, 1.0, -nv);
        /* the grooves between rings: a fine line, strongest toward the
         * flanks, deeper on the inside of a bend and where the body bunches */
        float dqp = min(f, 1.0 - f) * segl;
        float edge = smoothstep(0.2, 0.92, abs(nv));
        float cs = (0.3 + 0.7 * hk2) * (0.55 + 0.8 * max(ins, 0.0) - 0.3 * max(-ins, 0.0) + 0.3 * c) * (1.0 - 0.85 * clit) * body;
        float gw = uRM * (0.012 + 0.008 * max(ins, 0.0));
        float groove = 1.0 - smoothstep(gw, gw + 1.2 * uPx, dqp);
        float fall = 1.0 - smoothstep(0.0, 0.35 * segl, dqp);
        sh *= 1.0 - clamp(cs, 0.0, 1.2) * (groove * (0.16 + 0.42 * edge) + fall * 0.13);
        sh *= 0.95 + 0.07 * bul;
        if (hk3 > 0.55 && segl > 0.6 * uRM) {
            float dw = abs(f - (0.42 + 0.16 * hk)) * segl;
            sh *= 1.0 - 0.22 * edge * cs * (1.0 - smoothstep(0.007 * uRM, 0.007 * uRM + uPx, dw));
        }
        sh *= 1.0 - 0.05 * c;
        vec3 col = skin * sh;

        /* the head's colouring, in whichever piece of it this is */
        if (s < 1.5 * Rh) {
            vec2 qq = dU < dL ? (dU < dBack ? qu : q) : (dL < dBack ? ql : q);
            float nh = clamp(qq.y / Rh, -1.0, 1.0);
            vec3 hc = mix(uBelly, uBack, smoothstep(-0.55, 0.1, nh)) * (0.72 + 0.36 * smoothstep(-0.95, 0.55, nh));
            hc = mix(hc, uMark, 0.25 * smoothstep(0.2, 1.0, qq.x / Rh));
            /* the eye, half under the edge of the brow plate */
            vec2 E = vec2(Hj.x - 0.40 * Rh, Hj.y + 0.36 * Rh);
            float de = length(q - E);
            hc = mix(hc, mix(uBelly, uClit, 0.5), aa(de - 0.19 * Rh, 1.2 * uPx));
            hc = mix(hc, vec3(0.06, 0.04, 0.045) + 0.35 * uGlow * uNight, aa(de - 0.125 * Rh, 1.0 * uPx));
            hc = mix(hc, vec3(1.0, 0.97, 0.9), 0.85 * aa(length(q - E - vec2(0.03, -0.035) * Rh) - 0.025 * Rh, uPx));
            /* the plate: over the snout and brow, its back edge curved */
            float bx = qq.x - (-0.80 * Rh + 0.40 * Rh * sq(nh - 0.3));
            float byy = qq.y - (E.y + 0.03 * Rh + 0.10 * qq.x);
            float dv = -min(bx, byy);
            float pl = aa(dv, 1.0 * uPx);
            vec3 pc = mix(uBone, uBack, 0.38) * (0.80 + 0.30 * smoothstep(-0.1, 0.9, nh));
            pc *= 1.0 - 0.12 * (1.0 - smoothstep(0.0, 0.03 * Rh + uPx, abs(fract(bx / (0.13 * Rh)) - 0.5) * 0.13 * Rh)) * step(0.1 * Rh, bx);
            pc *= 1.0 - 0.25 * smoothstep(0.25 * Rh, 0.0, min(bx, byy));
            pc += 0.10 * exp(-sq((nh - 0.68) / 0.09)) * smoothstep(-0.5 * Rh, 0.2 * Rh, qq.x) * (vec3(1.0) - pc);
            for (int i = 0; i < 4; i++) {
                vec2 pp2 = vec2((0.32 + 0.24 * float(i)) * Rh, E.y + (0.16 - 0.03 * float(i)) * Rh + 0.06 * (0.32 + 0.24 * float(i)) * Rh);
                pc *= 1.0 - 0.5 * aa(length(qq - pp2) - 0.028 * Rh, 1.2 * uPx);
            }
            hc = mix(hc, pc, pl);
            hc *= 1.0 - 0.45 * (1.0 - smoothstep(0.0, 0.018 * Rh + uPx, abs(dv))) * step(-0.5, nh);
            col = mix(col, hc, smoothstep(0.8 * Rh, 0.1 * Rh, s));
            /* the line of the mouth, turning down at the corner */
            float cx = Hj.x - q.x;
            float lipy = Hj.y - 0.07 * Rh * sq(clamp(cx / (0.45 * Rh), 0.0, 1.0));
            float dlip = max(abs(q.y - lipy) - 0.02 * Rh, max(-cx - (th > 0.06 ? 0.0 : 2.0 * Rh), cx - 0.45 * Rh));
            col *= 1.0 - 0.6 * aa(dlip, 1.2 * uPx);
        }
        /* the mouth: a dark gullet, pale hooked teeth, the feelers */
        float rl = length(rq) / Rh;
        vec3 thc = uThroat * (0.35 + 0.75 * smoothstep(0.05, 1.0, rl)) * (1.0 - 0.25 * (1.0 - smoothstep(0.0, 0.25, abs(fract(rl * 4.0) - 0.5))));
        vec3 base = mix(col, thc, aa(dThroat, w));
        col = mix(base, col, aa(dSkin, w));
        col = mix(col, uBone * (0.85 + 0.15 * smoothstep(-0.05 * Rh, 0.0, -dTeeth)), aa(dTeeth, max(fwidth(dTeeth), 0.6 * uPx)));
        col = mix(col, mix(uMark, uBelly, 0.35), aa(dBarb, w));
        /* night: a line of glowing spots along the flank */
        if (uNight > 0.5 && s > 0.8 * uRM) {
            float dk = h11(k + 3.3);
            float dd = length(vec2((f - 0.5) * segl, v + 0.30 * R)) - 0.11 * R;
            float pls = 0.5 + 0.5 * sin(mu / (0.18 * uLW) * 6.2831853 - uT * 1.8);
            col = mix(col, uGlow, step(0.3, dk) * aa(dd, uPx) * (0.45 + 0.55 * pls));
        }
        /* paper: the worm is cut from it too */
        vec2 pp = vec2(mu, v) * 1.2 + vec2(311.0, 97.0);
        vec4 t0 = texture(uPaper, pp * (1.0 / 1024.0));
        float h1 = texture(uPaper, (pp + vec2(0.5, 1.1)) * (1.0 / 1024.0)).r;
        col *= (0.97 + 0.06 * t0.g) * (1.0 + uEmb * (t0.r - h1));
        o = vec4(col, cov);
    });

/* the falling sand: capsules in device pixels */
static const char *VS_GR = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aLoc;
    layout(location = 2) in vec4 aP;
    uniform vec2 uRes;
    out vec2 vL;
    out vec4 vP;
    void main() {
        vL = aLoc; vP = aP;
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_GR = "#version 330 core\n" GLSL(
    uniform vec3 uGrain;
    in vec2 vL;
    in vec4 vP;
    out vec4 o;
    void main() {
        float d = length(vec2(max(abs(vL.x) - vP.x, 0.0), vL.y)) - vP.y;
        o = vec4(uGrain * vP.w, clamp(0.5 - d, 0.0, 1.0) * vP.z);
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

static GLuint prog_bg, prog_tun, prog_worm, prog_gr, prog_text;
static GLuint vao_empty, vao_text, vbo_text, vao_body, vbo_body, vao_tun, vbo_tun, vao_gr, vbo_gr;
static GLuint tex_atlas, tex_noise, tex_paper, tex_rows, tex_text;
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

/* the paper itself, 1024x1024 and tileable, a texel per layout pixel:
 * R a height map: the cloudy "formation" of the pulp at a few pixels to a
 * few dozen, short fibres lying in it and a fine tooth; G a broad mottle */
#define PT 1024
static void bake_paper(void) {
    float *h = calloc((size_t)PT * PT, sizeof(float));
    Rng r = rng_for(0x9a9e7, 1, 2);
    for (int y = 0; y < PT; y++)
        for (int x = 0; x < PT; x++) {
            double u = (double)x / PT, v = (double)y / PT;
            double fm = 0.45 * tile_vn(u * 256, v * 256, 256, 960) + 0.30 * tile_vn(u * 128, v * 128, 128, 961) +
                        0.17 * tile_vn(u * 64, v * 64, 64, 962) + 0.08 * tile_vn(u * 32, v * 32, 32, 963);
            h[y * PT + x] = (float)(fm - 0.5) + (float)(rnd(&r) - 0.5) * 0.18f;
        }
    /* fibres: short, nearly straight, lying in the sheet */
    for (int f = 0; f < 5000; f++) {
        double x = rnd(&r) * PT, y = rnd(&r) * PT, a = rnd(&r) * 2 * M_PI;
        int steps = 6 + (int)(rnd(&r) * 22);
        float amp = (float)((rnd(&r) < 0.7 ? 1 : -1) * (0.04 + 0.08 * rnd(&r)));
        double bend = (rnd(&r) - 0.5) * 0.08;
        for (int s = 0; s < steps; s++) {
            x += 0.8 * cos(a); y += 0.8 * sin(a);
            a += bend + (rnd(&r) - 0.5) * 0.06;
            int ix = (int)floor(x), iy = (int)floor(y);
            for (int dy = -1; dy <= 2; dy++)
                for (int dx = -1; dx <= 2; dx++) {
                    double ddx = ix + dx - x, ddy = iy + dy - y;
                    float wgt = (float)exp(-(ddx * ddx + ddy * ddy) / 0.45);
                    h[(((iy + dy) % PT + PT) % PT) * PT + (((ix + dx) % PT + PT) % PT)] += amp * wgt;
                }
        }
    }
    double mean = 0, var = 0;
    for (int i = 0; i < PT * PT; i++) mean += h[i];
    mean /= PT * PT;
    for (int i = 0; i < PT * PT; i++) var += (h[i] - mean) * (h[i] - mean);
    double sd = sqrt(var / (PT * PT)) + 1e-9;
    unsigned char *t = malloc((size_t)PT * PT * 4);
    for (int y = 0; y < PT; y++)
        for (int x = 0; x < PT; x++) {
            double u = (double)x / PT, v = (double)y / PT, m = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 4; o++) {
                int Lc = 6 << o;
                m += amp * tile_vn(u * Lc, v * Lc, Lc, 940 + o);
                norm += amp; amp *= 0.5;
            }
            m /= norm;
            unsigned char *o = t + 4 * (y * PT + x);
            o[0] = (unsigned char)(clamp01(0.5 + (h[y * PT + x] - mean) / sd * 0.16) * 255);
            o[1] = (unsigned char)(clamp01(0.5 + (m - 0.5) * 2.2) * 255);
            o[2] = 128; o[3] = 255;
        }
    glGenTextures(1, &tex_paper);
    glBindTexture(GL_TEXTURE_2D, tex_paper);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PT, PT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tex_params(GL_REPEAT, 1);
    free(t); free(h);
}

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "burrow: shader:\n%s\n", log);
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

/* ---- palettes ---- */
typedef float C3[3];
static const struct palette {
    const char *name;
    C3 sky0, sky1, sun;                /* sky at the top, at the horizon; the sun or moon */
    float sunx, suny, sunr;            /* on the front screen (fractions of it); radius in L */
    int moon;
    C3 far0, far1, sand;               /* the farthest ridge, the nearest; the sand on top */
    C3 st[4];                          /* the strata, from under the sand down */
    C3 back, belly, mark, clit, throat, bone, rock, grain, cloud, glow;
    float shadow, emboss;
} palettes[] = {
    {"ochre", {0.36f, 0.60f, 0.76f}, {0.98f, 0.90f, 0.74f}, {1.00f, 0.97f, 0.86f}, 0.76f, 0.16f, 0.055f, 0,
     {0.97f, 0.88f, 0.72f}, {0.95f, 0.74f, 0.45f}, {0.86f, 0.55f, 0.27f},
     {{0.76f, 0.44f, 0.22f}, {0.64f, 0.34f, 0.18f}, {0.47f, 0.23f, 0.14f}, {0.25f, 0.12f, 0.09f}},
     {0.44f, 0.20f, 0.22f}, {0.89f, 0.68f, 0.58f}, {0.22f, 0.08f, 0.11f}, {0.74f, 0.42f, 0.36f},
     {0.40f, 0.05f, 0.08f}, {0.97f, 0.92f, 0.80f}, {0.42f, 0.33f, 0.29f}, {0.82f, 0.52f, 0.25f},
     {0.99f, 0.97f, 0.91f}, {0, 0, 0}, 0.42f, 0.32f},
    {"dusk", {0.19f, 0.14f, 0.31f}, {0.99f, 0.62f, 0.38f}, {1.00f, 0.44f, 0.25f}, 0.30f, 0.30f, 0.085f, 0,
     {0.86f, 0.46f, 0.43f}, {0.58f, 0.24f, 0.30f}, {0.62f, 0.25f, 0.22f},
     {{0.52f, 0.20f, 0.19f}, {0.40f, 0.14f, 0.17f}, {0.26f, 0.09f, 0.14f}, {0.11f, 0.05f, 0.09f}},
     {0.20f, 0.12f, 0.16f}, {0.92f, 0.62f, 0.46f}, {0.08f, 0.04f, 0.07f}, {0.56f, 0.27f, 0.24f},
     {0.95f, 0.36f, 0.22f}, {0.98f, 0.86f, 0.70f}, {0.20f, 0.10f, 0.13f}, {0.92f, 0.52f, 0.34f},
     {0.95f, 0.55f, 0.48f}, {0, 0, 0}, 0.45f, 0.30f},
    {"noon", {0.74f, 0.84f, 0.87f}, {0.98f, 0.96f, 0.90f}, {1.00f, 1.00f, 0.97f}, 0.55f, 0.10f, 0.05f, 0,
     {0.97f, 0.93f, 0.84f}, {0.95f, 0.85f, 0.67f}, {0.95f, 0.83f, 0.62f},
     {{0.92f, 0.79f, 0.58f}, {0.84f, 0.67f, 0.47f}, {0.70f, 0.52f, 0.36f}, {0.48f, 0.34f, 0.25f}},
     {0.58f, 0.40f, 0.40f}, {0.96f, 0.85f, 0.76f}, {0.36f, 0.22f, 0.24f}, {0.80f, 0.58f, 0.52f},
     {0.48f, 0.12f, 0.14f}, {1.00f, 0.98f, 0.92f}, {0.60f, 0.53f, 0.48f}, {0.97f, 0.88f, 0.68f},
     {1.00f, 1.00f, 0.99f}, {0, 0, 0}, 0.38f, 0.32f},
    {"night", {0.04f, 0.05f, 0.13f}, {0.21f, 0.24f, 0.42f}, {0.97f, 0.95f, 0.85f}, 0.70f, 0.17f, 0.045f, 1,
     {0.28f, 0.30f, 0.50f}, {0.19f, 0.19f, 0.37f}, {0.21f, 0.20f, 0.38f},
     {{0.17f, 0.16f, 0.32f}, {0.13f, 0.11f, 0.26f}, {0.09f, 0.07f, 0.19f}, {0.04f, 0.03f, 0.10f}},
     {0.50f, 0.50f, 0.62f}, {0.84f, 0.82f, 0.84f}, {0.26f, 0.26f, 0.40f}, {0.72f, 0.62f, 0.70f},
     {0.22f, 0.05f, 0.14f}, {0.82f, 0.84f, 0.88f}, {0.12f, 0.11f, 0.22f}, {0.52f, 0.52f, 0.70f},
     {0.30f, 0.31f, 0.50f}, {0.50f, 1.00f, 0.86f}, 0.50f, 0.30f},
    {"paper", {0.925f, 0.925f, 0.915f}, {0.965f, 0.96f, 0.945f}, {1.00f, 1.00f, 0.99f}, 0.72f, 0.18f, 0.05f, 0,
     {0.965f, 0.962f, 0.952f}, {0.955f, 0.95f, 0.935f}, {0.968f, 0.96f, 0.94f},
     {{0.955f, 0.95f, 0.935f}, {0.945f, 0.94f, 0.925f}, {0.935f, 0.93f, 0.915f}, {0.92f, 0.915f, 0.90f}},
     {0.84f, 0.82f, 0.80f}, {0.975f, 0.97f, 0.955f}, {0.68f, 0.66f, 0.64f}, {0.93f, 0.90f, 0.87f},
     {0.50f, 0.46f, 0.45f}, {1.00f, 1.00f, 0.985f}, {0.86f, 0.85f, 0.83f}, {0.90f, 0.89f, 0.86f},
     {1.00f, 1.00f, 0.995f}, {0, 0, 0}, 0.34f, 0.45f},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float pal_col[15][3], tun_col[3], sh_col[3], cap_col[3], cap_halo[3];

static float lin(float c) { return powf(c, 2.2f); }
static float gam(float c) { return powf(c, 1 / 2.2f); }

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "burrow: no palette \"%s\" (ochre, dusk, noon, night, paper)\n", opt_palette);
            exit(1);
        }
    } else {
        uint32_t h = opt_seed * 2654435761u;
        h ^= h >> 15;
        pal = (int)(h % NPAL);
    }
    const struct palette *P = &palettes[pal];
    for (int k = 1; k < 15; k++) {
        const float *a, *b;
        float f;
        if (k <= 4) { a = P->far0; b = P->far1; f = (k - 1) / 3.0f; }
        else if (k == 5) { a = b = P->sand; f = 0; }
        else {
            float u = (k - 6) / 8.0f * 3;
            int i = (int)u;
            if (i > 2) i = 2;
            a = P->st[i]; b = P->st[i + 1]; f = u - i;
        }
        float alt = k >= 6 ? ((k & 1) ? 1.035f : 0.97f) : 1.0f;
        for (int c = 0; c < 3; c++) pal_col[k][c] = gam(lin(a[c]) * (1 - f) + lin(b[c]) * f) * alt;
        /* now and then a pale band, or a grey one */
        double hb = hsh(k, 151);
        if (k >= 7 && pal != 4) {
            if (hb < 0.22) for (int c = 0; c < 3; c++) pal_col[k][c] += (P->far0[c] * (1 - 0.4f * (k - 7) / 7.0f) - pal_col[k][c]) * 0.26f;
            else if (hb > 0.84) for (int c = 0; c < 3; c++) pal_col[k][c] += (P->rock[c] - pal_col[k][c]) * 0.30f;
        }
    }
    for (int c = 0; c < 3; c++) {
        tun_col[c] = pal == 4 ? 0.62f - 0.01f * c : P->st[3][c] * 0.42f;
        sh_col[c] = pal == 4 ? 0.3f : P->st[3][c] * 0.3f;
    }
    /* the caption sits on the deep strata */
    float *bc = pal_col[12];
    float lum = 0.3f * bc[0] + 0.59f * bc[1] + 0.11f * bc[2];
    for (int c = 0; c < 3; c++) {
        cap_col[c] = lum < 0.45f ? 0.96f - 0.04f * c : 0.16f - 0.02f * c;
        cap_halo[c] = bc[c] * 0.9f;
    }
}

/* ---- buried things, clouds ---- */
#define MAXOBJ 32
static float objA[MAXOBJ][4], objB[MAXOBJ][4];
static int nobj;
static void add_obj(int ty, double x, double y, double s, double ang, double sd, double asp, double tint) {
    if (nobj >= MAXOBJ) return;
    float *A = objA[nobj], *B = objB[nobj];
    A[0] = (float)x; A[1] = (float)y; A[2] = (float)s; A[3] = (float)ty;
    B[0] = (float)ang; B[1] = (float)sd; B[2] = (float)asp; B[3] = (float)tint;
    nobj++;
}
static void place_objects(void) {
    Rng r = rng_for(77, 1, 0);
    nobj = 0;
    /* something big, long dead, deep under the front screen */
    add_obj(4, fr[0] + fr[2] * rr(&r, 0.3, 0.7), Y0 + L * rr(&r, 0.62, 0.72), L * rr(&r, 0.16, 0.2), rr(&r, -0.1, 0.1), 0, 1,
            rnd(&r));
    if (rnd(&r) < 0.5) objB[0][0] += (float)M_PI, objA[0][1] += 0;
    int want = (int)(bw / (0.15 * L));
    if (want < 10) want = 10;
    if (want > MAXOBJ - 1) want = MAXOBJ - 1;
    for (int tries = 0; tries < 600 && nobj < want + 1; tries++) {
        double x = bx0 + bw * rnd(&r), y = Y0 + L * rr(&r, 0.13, 0.08) + (by1 - Y0 - 0.16 * L) * rnd(&r);
        double u = rnd(&r), s;
        int ty;
        if (u < 0.36) { ty = 0; s = L * rr(&r, 0.012, 0.042); }
        else if (u < 0.58) { ty = 1; s = L * rr(&r, 0.04, 0.085); }
        else if (u < 0.82) { ty = 2; s = L * rr(&r, 0.018, 0.038); }
        else { ty = 3; s = L * rr(&r, 0.05, 0.075); }
        int ok = 1;
        for (int j = 0; j < nobj && ok; j++) {
            double rj = objA[j][2] * (objA[j][3] > 3.5 ? 1.7 : 1.3);
            if (hypot(x - objA[j][0], y - objA[j][1]) < rj + 1.4 * s + 0.02 * L) ok = 0;
        }
        if (!ok) continue;
        double ang = ty == 0 ? rr(&r, -0.4, 0.4) : ty == 1 ? rr(&r, -0.12, 0.12) : ty == 2 ? rr(&r, 0, 2 * M_PI) : rr(&r, -0.3, 0.3);
        if (ty == 3 && rnd(&r) < 0.5) ang += M_PI;
        add_obj(ty, x, y, s, ang, rr(&r, 0, 2 * M_PI), ty == 0 ? rr(&r, 1.2, 1.9) : ty == 1 ? rr(&r, 2.6, 4.0) : 1, rnd(&r));
    }
}

#define MAXCLOUD 6
static float clouds[MAXCLOUD][4];
static int ncloud;
static void place_clouds(double t) {
    Rng r = rng_for(78, 2, 0);
    ncloud = (int)(bw / (1.0 * L)) + 1;
    if (ncloud > MAXCLOUD) ncloud = MAXCLOUD;
    if (pal == 2) ncloud = ncloud > 2 ? 2 : ncloud;
    double x0 = bx0 - 0.4 * L, span_w = bw + 0.8 * L;
    for (int i = 0; i < ncloud; i++) {
        double x = x0 + span_w * (i + rr(&r, 0.1, 0.9)) / ncloud + L * 0.004 * rr(&r, 0.6, 1.4) * t;
        x = x0 + fmod(fmod(x - x0, span_w) + span_w, span_w);
        clouds[i][0] = (float)x;
        clouds[i][1] = (float)(Y0 - L * rr(&r, 0.33, 0.52));
        clouds[i][2] = (float)(L * rr(&r, 0.08, 0.15));
        clouds[i][3] = (float)(L * rr(&r, 0.035, 0.05));
    }
}

/* ---- the lines of the land: 14 functions of x, sampled every other
 * device pixel into an RW x 4 RGBA32F texture: far ridges 0-3, the sand 4,
 * the strata 5-13 ---- */
static float *rowbuf;
static double *surfb, *heave, *sink;
static int RW;
static double rx0, rdx;
static double v_ox, v_oy, v_sc;
#define ROWI(li, i) ((((li) / 4) * RW + (i)) * 4 + (li) % 4)

static void rows_layout(void) {
    double x0 = v_ox - 0.05 * L, x1 = v_ox + W / v_sc + 0.05 * L, dx = 2.0 / v_sc;
    int n = (int)ceil((x1 - x0) / dx) + 2;
    if (n == RW && x0 == rx0 && dx == rdx) return;
    RW = n; rx0 = x0; rdx = dx;
    rowbuf = realloc(rowbuf, sizeof(float) * 16 * RW);
    surfb = realloc(surfb, sizeof(double) * RW);
    heave = realloc(heave, sizeof(double) * RW);
    sink = realloc(sink, sizeof(double) * RW);
    for (int i = 0; i < RW; i++) {
        double x = rx0 + i * rdx;
        for (int k = 0; k < NFAR; k++) rowbuf[ROWI(k, i)] = (float)far_y(k, x);
        surfb[i] = surf_base(x);
        rowbuf[ROWI(4, i)] = (float)surfb[i];
        for (int j = 0; j < NSTRATA; j++) rowbuf[ROWI(5 + j, i)] = (float)strata_y(j, x);
        rowbuf[ROWI(14, i)] = 0;
        rowbuf[ROWI(15, i)] = (float)surfb[i];   /* the sand at rest: its bedding hangs from this */
    }
    glBindTexture(GL_TEXTURE_2D, tex_rows);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, RW, 4, 0, GL_RGBA, GL_FLOAT, rowbuf);
    tex_params(GL_CLAMP_TO_EDGE, 0);
}

static void splat(double *acc, double xc, double w, double a) {
    int i0 = (int)floor((xc - 3 * w - rx0) / rdx), i1 = (int)ceil((xc + 3 * w - rx0) / rdx);
    if (i0 < 0) i0 = 0;
    if (i1 > RW - 1) i1 = RW - 1;
    for (int i = i0; i <= i1; i++) acc[i] += a * exp(-sq((rx0 + i * rdx - xc) / w));
}

/* the sand line this frame: heaved up over the worm where it runs shallow,
 * slumped over burrows that have fallen in */
static void surface_update(const Pass *trail[3], int body_on, double t) {
    memset(heave, 0, sizeof(double) * RW);
    memset(sink, 0, sizeof(double) * RW);
    double ds = (body[NB].s - body[0].s) / NB;
    if (body_on)
        for (int j = 0; j <= NB; j += 3) {
            const BodyPt *b = &body[j];
            if (b->s < -RM || -b->air < -1.2 * b->R) continue;
            double dp = fmax(-b->air - b->R, 0);   /* sand over its back */
            double a = 1.2 * b->R * exp(-dp / (0.07 * L)) * (1 + 0.8 * exp(-fmax(b->s, 0) / (0.15 * LW)));
            double w = 0.8 * dp + 1.6 * b->R + 0.02 * L;
            splat(heave, b->x, w, a * 3 * ds / (w * sqrt(M_PI)));
        }
    /* sand heaped round the holes of a breach while the body runs through
     * them, slumping back afterwards */
    if (trail[0] && trail[0]->kind == K_BREACH && trail[0]->s_up >= 0)
        for (int h = 0; h < 2; h++) {
            const Pass *P = trail[0];
            double hole = h ? P->s_down : P->s_up, x, y;
            double pass = (head_s(P, t) - hole) / LW;
            double m = sstep(-0.02, 0.06, pass) * (1 - sstep(1.0, 1.6, pass));
            if (m <= 0) continue;
            path_at(P, hole, &x, &y, NULL, NULL);
            double w = 2.2 * RM;
            splat(heave, x - 0.8 * RM, w, 0.75 * RM * m);
            splat(heave, x + 0.8 * RM, w, 0.75 * RM * m);
        }
    for (int k = 0; k < 3; k++) {
        const Pass *P = trail[k];
        if (!P) continue;
        double tb = LW / P->v, sh = fmin(head_s(P, t), P->s_end);
        double lo = fmax(P->s_start, P->s_start + P->v * (t - P->T0 - tb - TOPEN - TC - TF)), hi = sh - LW;
        if (hi <= lo) continue;
        int n = (int)ceil((hi - lo) / (0.03 * L));
        if (n > 200) n = 200;
        double dsg = (hi - lo) / n;
        for (int i = 0; i <= n; i++) {
            double sg = lo + dsg * i, x, y;
            path_at(P, sg, &x, &y, NULL, NULL);
            double dp = y - surf_base(x);
            if (dp < 0) continue;
            double at = t - (P->T0 + (sg - P->s_start) / P->v) - tb - TOPEN;
            double a = 0.45 * RM * exp(-dp / (0.075 * L)) * sstep(0, TC, at) * (1 - sstep(TC, TC + TF, at));
            if (a < 0.02) continue;
            double w = 0.7 * dp + 1.4 * RM;
            splat(sink, x, w, a * dsg / (w * sqrt(M_PI)));
        }
    }
    double bm = 0.9 * RM;
    for (int i = 0; i < RW; i++)
        rowbuf[ROWI(4, i)] = (float)(surfb[i] - bm * (1 - exp(-heave[i] / bm)) + sink[i]);
    glBindTexture(GL_TEXTURE_2D, tex_rows);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 1, RW, 1, GL_RGBA, GL_FLOAT, rowbuf + 4 * RW);
}

/* ---- strips ---- */
static float bodyv[(NB + 1) * 2 * 8];
static void body_strip(void) {
    for (int i = 0; i <= NB; i++) {
        const BodyPt *b = &body[i];
        double hw = b->R * 1.2 + 0.18 * RM + 1.7 * RM * (1 - sstep(0, 3.0 * RM, b->s));
        for (int sd = 0; sd < 2; sd++) {
            double v = sd ? -hw : hw;
            float *o = bodyv + (i * 2 + sd) * 8;
            o[0] = (float)(b->x + b->nx * v); o[1] = (float)(b->y + b->ny * v);
            o[2] = (float)b->s; o[3] = (float)v; o[4] = (float)b->mu; o[5] = (float)b->Q;
            o[6] = (float)b->kap; o[7] = (float)b->dq;
        }
    }
}

#define NTUN 300
static float tunv[3][(NTUN + 1) * 2 * 8];
static int tun_cnt[3];
static int tunnel_build(const Pass *P, double t, float *out) {
    double tb = LW / P->v, sh0 = head_s(P, t), sh = fmin(sh0, P->s_end);
    double lo = fmax(P->s_start, P->s_start + P->v * (t - P->T0 - tb - TOPEN - TC - TF)), hi = sh;
    if (hi - lo < 0.01 * L) return 0;
    int n = (int)ceil((hi - lo) / (0.012 * L));
    if (n > NTUN) n = NTUN;
    if (n < 2) n = 2;
    for (int i = 0; i <= n; i++) {
        double sg = hi - (hi - lo) * i / n, x, y, tx, ty;
        path_at(P, sg, &x, &y, &tx, &ty);
        double nx = P->dir * ty, ny = -P->dir * tx;
        double age = t - (P->T0 + (sg - P->s_start) / P->v);
        /* the tunnel hugs the body; behind the tail the loose sand closes
         * in to a crack, which then falls in altogether */
        double Rt = RM * (fmax(prof(fmin(sh0 - sg, LW) / LW), 0.12) * (1 + 0.05 * sstep(0, 0.2 * LW, sh0 - sg)) + 0.08);
        double hw = fmax(Rt, 1.1 * RM) * 1.15 + 2 / v_sc;
        for (int sd = 0; sd < 2; sd++) {
            double v = sd ? -hw : hw;
            float *o = out + (i * 2 + sd) * 8;
            o[0] = (float)(x + nx * v); o[1] = (float)(y + ny * v);
            o[2] = (float)age; o[3] = (float)v; o[4] = (float)Rt; o[5] = (float)sg;
            o[6] = o[7] = 0;
        }
    }
    return 2 * (n + 1);
}

/* ---- the sand in the air when the worm breaches. Pouring: NSTREAM
 * streams from points fixed on the body, each drawn as the ribbon of grains
 * it shed over the last second, falling ballistically; it thins and breaks
 * into drips as that bit of body dries out. Spraying: grain slots at the
 * holes, each reborn every second or so with a hashed velocity. ---- */
#define NGR 4000
#define NSTREAM 110
#define NSEG 24
#define SEG_DT 0.035
static float grv[NGR * 6 * 8];
static int gr_cnt;
static double gh(long i, long c, int k) { return hsh(i * 1000003L + c * 7919L + k, 200 + k); }

/* a capsule from a to b, wid wide, in device pixels */
static void grain_seg(double xa, double ya, double xb, double yb, double wid, double alpha, double bright) {
    if (gr_cnt >= NGR) return;
    double ax = (xa - v_ox) * v_sc, ay = (ya - v_oy) * v_sc, bx2 = (xb - v_ox) * v_sc, by2 = (yb - v_oy) * v_sc;
    double cx = (ax + bx2) / 2, cy = (ay + by2) / 2, dx = bx2 - ax, dy = by2 - ay, len = hypot(dx, dy);
    double ux = len > 1e-6 ? dx / len : 0, uy = len > 1e-6 ? dy / len : 1;
    double hl = len / 2, hwd = fmax(wid * 0.5 * v_sc, 0.45);
    if (cx < -hl - 8 || cy < -hl - 8 || cx > W + hl + 8 || cy > H + hl + 8) return;
    double ea = hl + hwd + 1.0, eb = hwd + 1.0;
    float *o = grv + gr_cnt * 48;
    static const int cs[6][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, -1}, {1, 1}, {-1, 1}};
    for (int k = 0; k < 6; k++) {
        double la = cs[k][0] * ea, lb = cs[k][1] * eb;
        o[k * 8 + 0] = (float)(cx + ux * la - uy * lb);
        o[k * 8 + 1] = (float)(cy + uy * la + ux * lb);
        o[k * 8 + 2] = (float)la; o[k * 8 + 3] = (float)lb;
        o[k * 8 + 4] = (float)hl; o[k * 8 + 5] = (float)hwd;
        o[k * 8 + 6] = (float)alpha; o[k * 8 + 7] = (float)bright;
    }
    gr_cnt++;
}

static void grains_build(const Pass *P, double t) {
    gr_cnt = 0;
    if (P->kind != K_BREACH || P->s_up < 0) return;
    const double G = 1.35 * L;
    /* the streams */
    double t0 = floor(t / SEG_DT) * SEG_DT, ft = t - t0;
    for (long j = 0; j < NSTREAM; j++) {
        uint32_t key = (uint32_t)(j + 97 * P->p);
        double si = LW * (0.04 + 0.86 * hsh(key, 193));
        double across = -0.80 + 0.22 * hsh(key, 195);
        double strength = 0.4 + 0.6 * hsh(key, 194), drip = 2.5 + 5.0 * pow(hsh(key, 197), 2.0);
        double px0 = 0, py0 = 0;
        int have = 0;
        for (int k = 0; k < NSEG; k++) {
            double te = t0 - k * SEG_DT, age = ft + k * SEG_DT;
            double sg = head_s(P, te) - si;
            if (te < P->T0 || sg <= P->s_up || sg >= P->s_down) { have = 0; continue; }
            double tau = (sg - P->s_up) / P->v;
            double flow = strength * exp(-tau / drip);
            if (flow < 0.08) { have = 0; continue; }
            double x, y, tx, ty;
            path_at(P, sg, &x, &y, &tx, &ty);
            if (surf_base(x) - y < RM * 0.5) { have = 0; continue; }
            double nx = P->dir * ty, ny = -P->dir * tx, R = RM * prof(si / LW);
            double x0 = x + nx * across * R, y0 = y + ny * across * R;
            double vx = tx * P->v * 0.8 + (hsh(key, 198) - 0.5) * 0.03 * L, vy = ty * P->v * 0.8;
            double gx = x0 + vx * age, gy = y0 + vy * age + 0.5 * G * age * age;
            if (gy > surf_base(gx)) break;
            /* thin streams break up into drops */
            double br = 0.85 + 0.25 * hsh(key, 199);
            if (flow > 0.28) {
                if (have) grain_seg(px0, py0, gx, gy, L * (0.0014 + 0.0045 * flow) * (1 - 0.3 * fmin(age, 1)), 0.95, br);
            } else if (((long)floor(te / SEG_DT) + j) % 2 == 0) {
                /* a thin stream falls as separate grains */
                grain_seg(gx, gy - 0.0015 * L, gx, gy, L * 0.0026, 0.92, br);
            }
            px0 = gx; py0 = gy; have = 1;
        }
    }
    /* the spray from the holes */
    for (long i = 0; i < 1200; i++) {
        double per = 1.2 + 0.8 * hsh(i, 191), ph = hsh(i, 192) * per;
        long c = (long)floor((t + ph) / per);
        double ts = c * per - ph, age = t - ts;
        if (ts < P->T0) continue;
        double hole = (i & 1) ? P->s_up : P->s_down;
        double pass = head_s(P, ts) - hole;
        if (pass < -0.02 * LW || pass > LW) continue;
        double I = 0.4 + 0.6 * exp(-fmax(pass, 0) / (0.15 * LW));
        if (gh(i, c, 1) > I) continue;
        double x, y;
        path_at(P, hole, &x, &y, NULL, NULL);
        x += (gh(i, c, 2) - 0.5) * 2.6 * RM;
        y = surf_base(x) - 0.004 * L;
        double spd = (0.2 + 0.6 * gh(i, c, 3)) * L * sqrt(I), a = (gh(i, c, 4) - 0.5) * 1.3 + 0.35 * P->dir * ((i & 1) ? -1 : 1);
        double vx = sin(a) * spd, vy = -cos(a) * spd;
        double gx = x + vx * age, gy = y + vy * age + 0.5 * G * age * age;
        if (gy > surf_base(gx) + 0.002 * L && age > 0.1) continue;
        double vyy = vy + G * age;
        grain_seg(gx - vx * 0.02, gy - vyy * 0.02, gx, gy, L * (0.0018 + 0.0022 * gh(i, c, 7)), 0.95 * fmin(1, age / 0.04), 0.8 + 0.35 * gh(i, c, 8));
    }
}

/* ---- the galleries' letters ---- */
static float txt_box[4], txt_hw;
static long txt_key = -1;
static void build_text(const Pass *P) {
    if (txt_key == P->p) return;
    txt_key = P->p;
    char buf[64];
    double box[4] = {fr[0], Y0 + 0.11 * L, fr[2], 0.19 * L};
    if (P->txt == T_TIME) {
        double tc = P->T0 + (sg_at_x(P, fr[0] + fr[2] / 2) - P->s_start) / P->v;
        time_t tt = (time_t)(opt_epoch + tc);
        struct tm tm;
        localtime_r(&tt, &tm);
        strftime(buf, sizeof buf, "%H:%M", &tm);
    } else {
        snprintf(buf, sizeof buf, "%s", opt_text ? opt_text : host);
        for (char *c = buf; *c; c++) *c = (char)toupper((unsigned char)*c);
    }
    int cw, ch;
    unsigned char *cv = set_text(buf, 100, 0, &cw, &ch);
    int x0 = cw, y0 = ch, x1 = 0, y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    free(cv);
    if (x1 <= x0) { txt_hw = 0; return; }
    double fitw = fmin(box[2] * 0.55 / (x1 - x0 + 1), box[3] * 0.8 / (y1 - y0 + 1)) * 100;
    double tx = fmax(1.0, (x1 - x0) * fitw / 100 / 3000);
    int px = (int)(fitw / tx), pad = (int)(0.04 * L / tx) + 4;
    cv = set_text(buf, px, pad, &cw, &ch);
    x0 = cw; y0 = ch; x1 = 0; y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    size_t n = (size_t)cw * ch;
    float *gi = malloc(sizeof(float) * n), *go = malloc(sizeof(float) * n), *sd = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) {
        int in = cv[i] >= 128;
        go[i] = in ? 0 : 1e20f;
        gi[i] = in ? 1e20f : 0;
    }
    edt2(go, cw, ch);
    edt2(gi, cw, ch);
    float deepest = 0;
    for (size_t i = 0; i < n; i++) {
        float a2 = cv[i] / 255.0f, s;
        if (a2 > 0.02f && a2 < 0.98f) s = 0.5f - a2;
        else if (cv[i] >= 128) s = -(sqrtf(gi[i]) - 0.5f);
        else s = sqrtf(go[i]) - 0.5f;
        sd[i] = s * (float)tx;
        if (-sd[i] > deepest) deepest = -sd[i];
    }
    double cx = box[0] + box[2] / 2, cyy = box[1] + box[3] / 2;
    txt_box[0] = (float)(cx - (x0 + x1 + 1) * 0.5 * tx);
    txt_box[1] = (float)(cyy - (y0 + y1 + 1) * 0.5 * tx);
    txt_box[2] = (float)(cw * tx); txt_box[3] = (float)(ch * tx);
    txt_hw = deepest;
    glBindTexture(GL_TEXTURE_2D, tex_text);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, cw, ch, 0, GL_RED, GL_FLOAT, sd);
    free(cv); free(gi); free(go); free(sd);
}
/* ---- caption: the time, and how long the screen has been locked ---- */
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
    float halo[4] = {cap_halo[0], cap_halo[1], cap_halo[2], 0.16f}, col[4] = {cap_col[0], cap_col[1], cap_col[2], 0.92f};
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

static void attrib8(GLuint vao, GLuint vbo, int a, int b, int c) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, a, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, b, GL_FLOAT, GL_FALSE, 32, (void *)(sizeof(float) * a));
    glVertexAttribPointer(2, c, GL_FLOAT, GL_FALSE, 32, (void *)(sizeof(float) * (a + b)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
}

static void gl_init(void) {
    prog_bg = mk_prog(VS_FULL, FS_BG);
    prog_tun = mk_prog(VS_STRIP, FS_TUN);
    prog_worm = mk_prog(VS_STRIP, FS_WORM);
    prog_gr = mk_prog(VS_GR, FS_GR);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_empty);
    GLuint *vaos[4] = {&vao_text, &vao_body, &vao_tun, &vao_gr}, *vbos[4] = {&vbo_text, &vbo_body, &vbo_tun, &vbo_gr};
    for (int i = 0; i < 4; i++) { glGenVertexArrays(1, vaos[i]); glGenBuffers(1, vbos[i]); }
    attrib8(vao_text, vbo_text, 2, 2, 4);
    attrib8(vao_body, vbo_body, 2, 4, 2);
    attrib8(vao_tun, vbo_tun, 2, 4, 2);
    attrib8(vao_gr, vbo_gr, 2, 2, 4);
    bake_noise();
    bake_paper();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &tex_text);
    glBindTexture(GL_TEXTURE_2D, tex_text);
    tex_params(GL_CLAMP_TO_EDGE, 0);
    {
        float big = 1e3f;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, &big);
    }
    glGenTextures(1, &tex_rows);
}

/* ---- the sheets' cache: the background barely changes, so it is drawn
 * into a texture and only the parts that move are redrawn each frame: the
 * sand line where it heaves or slumps, the clouds, the galleries ---- */
static GLuint fbo_bg, tex_bg;
static int bg_w, bg_h, bg_ok, bg_rw, bg_txt;
static double bg_ox, bg_oy, bg_sc;
static float *bg_row, bg_cloud[MAXCLOUD][4];

static void bg_prepare(void) {
    if (!fbo_bg) { glGenFramebuffers(1, &fbo_bg); glGenTextures(1, &tex_bg); }
    if (bg_w != W || bg_h != H) {
        glBindTexture(GL_TEXTURE_2D, tex_bg);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        tex_params(GL_CLAMP_TO_EDGE, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bg);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bg, 0);
        bg_w = W; bg_h = H; bg_ok = 0;
    }
    if (bg_rw != RW || bg_ox != v_ox || bg_oy != v_oy || bg_sc != v_sc) bg_ok = 0;
}

/* draw the sheets over the world rect x0..x1, y0..y1 */
static void bg_rect(double x0, double y0, double x1, double y1) {
    int a = (int)floor((x0 - v_ox) * v_sc) - 2, b = (int)ceil((x1 - v_ox) * v_sc) + 2;
    int c = (int)floor((y0 - v_oy) * v_sc) - 2, d = (int)ceil((y1 - v_oy) * v_sc) + 2;
    if (a < 0) a = 0;
    if (c < 0) c = 0;
    if (b > W) b = W;
    if (d > H) d = H;
    if (b <= a || d <= c) return;
    glScissor(a, H - d, b - a, d - c);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void bg_draw_dirty(int txt_on) {
    glBindVertexArray(vao_empty);
    if (!bg_ok) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
        bg_row = realloc(bg_row, sizeof(float) * RW);
        for (int i = 0; i < RW; i++) bg_row[i] = rowbuf[ROWI(4, i)];
    } else {
        glEnable(GL_SCISSOR_TEST);
        /* the sand line, where it moved */
        int i0 = -1, i1 = -1;
        double lo = 1e30, hi = -1e30;
        for (int i = 0; i < RW; i++) {
            float y = rowbuf[ROWI(4, i)];
            if (fabsf(y - bg_row[i]) * v_sc < 0.01f) continue;
            if (i0 < 0) i0 = i;
            i1 = i;
            lo = fmin(lo, fmin(y, bg_row[i]));
            hi = fmax(hi, fmax(y, bg_row[i]));
        }
        if (i0 >= 0) {
            /* the cache now holds the line as it is over that stretch (and
             * a margin), so slow slumping can't creep past the threshold */
            double xa = rx0 + i0 * rdx - 0.03 * L, xb = rx0 + i1 * rdx + 0.03 * L;
            bg_rect(xa, lo - 0.05 * L, xb, hi + 0.03 * L);
            for (int i = 0; i < RW; i++) {
                double x = rx0 + i * rdx;
                if (x >= xa + 0.005 * L && x <= xb - 0.005 * L) bg_row[i] = rowbuf[ROWI(4, i)];
            }
        }
        /* the clouds, where they were and where they are */
        double g = 0.0075 * L;
        for (int i = 0; i < ncloud; i++) {
            const float *a = clouds[i], *b = bg_cloud[i];
            if (a[0] == b[0] && a[1] == b[1]) continue;
            for (int k = 0; k < 2; k++) {
                const float *C = k ? b : a;
                bg_rect(C[0] - 1.45 * C[2] - g, C[1] - 2.9 * C[3] - g, C[0] + 1.45 * C[2] + g, C[1] + 4 * g);
            }
        }
        /* the galleries */
        if (txt_on || bg_txt) bg_rect(txt_box[0], txt_box[1] - 0.02 * L, txt_box[0] + txt_box[2], txt_box[1] + txt_box[3] + 0.02 * L);
        glDisable(GL_SCISSOR_TEST);
    }
    memcpy(bg_cloud, clouds, sizeof clouds);
    bg_txt = txt_on;
    bg_ok = 1; bg_rw = RW; bg_ox = v_ox; bg_oy = v_oy; bg_sc = v_sc;
}

static void world_init(void) {
    Y0 = fr[1] + 0.44 * fr[3];
    RM = 0.05 * L;
    LW = 1.75 * L;
    V0 = 0.06 * L;
    DPASS = (bw + LW + 0.45 * L) * 1.12 / V0 + GAP;
    Rng r = rng_for(5, 5, 5);
    fold_ph = rr(&r, 0, 2 * M_PI);
    build_rings();
    place_objects();
}

/* ---- the view: world -> this window's pixels ---- */
static void view_update(void) {
    if (span_me >= 0) {
        v_sc = W / span[span_me][2];
        v_ox = span[span_me][0]; v_oy = span[span_me][1];
    } else {
        v_sc = fmax(W / fr[2], H / fr[3]);
        v_ox = fr[0] + (fr[2] - W / v_sc) / 2; v_oy = fr[1] + (fr[3] - H / v_sc) / 2;
    }
}

static void set_view(GLuint pr) {
    glUniform3f(U(pr, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(pr, "uRes"), (float)W, (float)H);
}
static void u3(GLuint pr, const char *n, const float *c) { glUniform3f(U(pr, n), c[0], c[1], c[2]); }

static void render(double t, double wall, GLuint target) {
    if (t < 0) t = 0;
    view_update();
    rows_layout();
    pc_frame++;
    long p = (long)floor(t / DPASS);
    const Pass *P = pass_get(p);
    const Pass *trail[3] = {P, p >= 1 ? pass_get(p - 1) : NULL, p >= 2 ? pass_get(p - 2) : NULL};
    int on = t - P->T0 < DPASS - GAP;
    if (on) body_build(P, t);
    surface_update(trail, on, t);
    for (int k = 0; k < 3; k++) tun_cnt[k] = trail[k] ? tunnel_build(trail[k], t, tunv[k]) : 0;
    grains_build(P, t);
    place_clouds(t);
    const struct palette *C = &palettes[pal];
    float gap = (float)(0.0075 * L);

    /* the galleries, while a galleries pass's letters are still there */
    float txt[4] = {0, 0, 1, 1};
    for (int k = 0; k < 2; k++) {
        const Pass *T = trail[k];
        if (!T || T->kind != K_TEXT) continue;
        build_text(T);
        double xh, yh, xa, xb, y;
        path_at(T, head_s(T, t), &xh, &yh, NULL, NULL);
        path_at(T, T->s_start, &xa, &y, NULL, NULL);
        path_at(T, T->s_end, &xb, &y, NULL, NULL);
        txt[0] = txt_hw > 0; txt[1] = (float)xh; txt[2] = (float)T->dir; txt[3] = (float)(fabs(xb - xa) / (DPASS - GAP));
        break;
    }

    /* the sheets: kept in a cache and redrawn only where they change */
    bg_prepare();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_bg);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    GLuint pr = prog_bg;
    glUseProgram(pr);
    set_view(pr);
    glUniform3f(U(pr, "uRowX"), (float)rx0, (float)rdx, (float)RW);
    glUniform1f(U(pr, "uL"), (float)L);
    glUniform1f(U(pr, "uY0"), (float)Y0);
    glUniform1f(U(pr, "uT"), (float)fmod(t, 20000.0));
    glUniform1f(U(pr, "uGap"), gap);
    glUniform1f(U(pr, "uSoft0"), (float)(0.0025 * L));
    glUniform1f(U(pr, "uSoft1"), (float)(0.0045 * L));
    glUniform1f(U(pr, "uShadow"), C->shadow);
    glUniform1f(U(pr, "uEmb"), C->emboss);
    glUniform1f(U(pr, "uStars"), C->moon ? 1.0f : 0.0f);
    glUniform1f(U(pr, "uSeedF"), (float)(opt_seed % 1000) * 0.137f);
    u3(pr, "uSky0", C->sky0); u3(pr, "uSky1", C->sky1); u3(pr, "uSun", C->sun);
    u3(pr, "uCloudC", C->cloud); u3(pr, "uBone", C->bone); u3(pr, "uRock", C->rock); u3(pr, "uTunC", tun_col);
    glUniform4f(U(pr, "uSunP"), (float)(fr[0] + C->sunx * fr[2]), (float)(fr[1] + C->suny * fr[3]), (float)(C->sunr * L), (float)C->moon);
    glUniform3fv(U(pr, "uCol"), 15, &pal_col[0][0]);
    /* only the buried things on this screen */
    {
        float va[MAXOBJ][4], vb[MAXOBJ][4], ymin = 1e30f;
        int nv = 0;
        double x0 = v_ox, x1 = v_ox + W / v_sc, y1 = v_oy + H / v_sc;
        for (int i = 0; i < nobj; i++) {
            double r = objA[i][2] * 1.8 + 0.02 * L;
            if (objA[i][0] + r < x0 || objA[i][0] - r > x1 || objA[i][1] - r > y1 || objA[i][1] + r < v_oy) continue;
            memcpy(va[nv], objA[i], sizeof va[0]);
            memcpy(vb[nv], objB[i], sizeof vb[0]);
            ymin = fminf(ymin, (float)(objA[i][1] - r));
            nv++;
        }
        glUniform4fv(U(pr, "uObj"), nv, &va[0][0]);
        glUniform4fv(U(pr, "uObjB"), nv, &vb[0][0]);
        glUniform1i(U(pr, "uNObj"), nv);
        glUniform1f(U(pr, "uObjY"), ymin);
    }
    glUniform4fv(U(pr, "uCloud"), ncloud, &clouds[0][0]);
    glUniform1i(U(pr, "uNCloud"), ncloud);
    glUniform4f(U(pr, "uTBox"), txt_box[0], txt_box[1], txt_box[2], txt_box[3]);
    glUniform4f(U(pr, "uTxt"), txt[0], txt[1], txt[2], txt[3]);
    glUniform1f(U(pr, "uTHw"), txt_hw);
    glUniform1f(U(pr, "uTO"), (float)TXT_OPEN);
    glUniform1f(U(pr, "uTC"), (float)TC);
    glUniform1f(U(pr, "uTF"), (float)TF);
    glUniform3f(U(pr, "uLamp"), (float)(fr[0] + 0.5 * fr[2]), (float)(fr[1] - 0.3 * fr[3]), 0.05f);
    glUniform1i(U(pr, "uRows"), 1);
    glUniform1i(U(pr, "uNoise"), 0);
    glUniform1i(U(pr, "uPaper"), 2);
    glUniform1i(U(pr, "uText"), 3);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_rows);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_paper);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_text);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    bg_draw_dirty(txt[0] > 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_bg);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
    glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, target);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    /* the burrows, oldest first */
    pr = prog_tun;
    glUseProgram(pr);
    set_view(pr);
    glUniform2f(U(pr, "uShift"), 0, 0);
    glUniform3f(U(pr, "uRowX"), (float)rx0, (float)rdx, (float)RW);
    u3(pr, "uTunC", tun_col);
    glUniform1f(U(pr, "uTO"), (float)TOPEN);
    glUniform1f(U(pr, "uTC"), (float)TC);
    glUniform1f(U(pr, "uTF"), (float)TF);
    glUniform1f(U(pr, "uPx"), (float)(1 / v_sc));
    glUniform1f(U(pr, "uRM"), (float)RM);
    glUniform1f(U(pr, "uGap"), gap);
    glUniform1i(U(pr, "uRows"), 1);
    glUniform1i(U(pr, "uNoise"), 0);
    glBindVertexArray(vao_tun);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_tun);
    for (int k = 2; k >= 0; k--) {
        if (!tun_cnt[k]) continue;
        glUniform1f(U(pr, "uTB"), (float)(LW / trail[k]->v));
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tun_cnt[k] * 8 * sizeof(float)), tunv[k], GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, tun_cnt[k]);
    }

    /* the worm: its shadow, then itself */
    if (on) {
        body_strip();
        pr = prog_worm;
        glUseProgram(pr);
        set_view(pr);
        glUniform1f(U(pr, "uLW"), (float)LW);
        glUniform1f(U(pr, "uRM"), (float)RM);
        glUniform1f(U(pr, "uT"), (float)fmod(t, 20000.0));
        glUniform1f(U(pr, "uLam"), (float)LAM_P);
        glUniform1f(U(pr, "uPer"), (float)PER_P);
        glUniform1f(U(pr, "uMouth"), (float)mouth);
        glUniform1f(U(pr, "uPx"), (float)(1 / v_sc));
        glUniform1f(U(pr, "uShA"), 0.38f);
        glUniform1f(U(pr, "uNight"), C->moon ? 1.0f : 0.0f);
        glUniform1f(U(pr, "uEmb"), C->emboss * 0.8f);
        u3(pr, "uBack", C->back); u3(pr, "uBelly", C->belly); u3(pr, "uMark", C->mark); u3(pr, "uClit", C->clit);
        u3(pr, "uThroat", C->throat); u3(pr, "uBone", C->bone); u3(pr, "uShC", sh_col); u3(pr, "uGlow", C->glow);
        glUniform1i(U(pr, "uPaper"), 2);
        glUniform1i(U(pr, "uNoise"), 0);
        glBindVertexArray(vao_body);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_body);
        glBufferData(GL_ARRAY_BUFFER, sizeof bodyv, bodyv, GL_STREAM_DRAW);
        glUniform1f(U(pr, "uMode"), 1);
        glUniform2f(U(pr, "uShift"), (float)(0.003 * L), (float)(-0.011 * L));
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (NB + 1) * 2);
        glUniform1f(U(pr, "uMode"), 0);
        glUniform2f(U(pr, "uShift"), 0, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (NB + 1) * 2);
    }

    /* the sand in the air */
    if (gr_cnt) {
        pr = prog_gr;
        glUseProgram(pr);
        glUniform2f(U(pr, "uRes"), (float)W, (float)H);
        u3(pr, "uGrain", C->grain);
        glBindVertexArray(vao_gr);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_gr);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(gr_cnt * 48 * sizeof(float)), grv, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, gr_cnt * 6);
    }

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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "burrow");
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
        "usage: burrow [--window | --root | --layer] [--seed N] [--fps N]\n"
        "              [--palette ochre|dusk|noon|night|paper] [--no-caption]\n"
        "              [--text WORD] [--font FAMILY] [--schedule]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS | --breach]] [--bench N] [--size WxH]\n"
        "  --palette   default: picked from the seed\n"
        "  --text      what the galleries spell instead of the host name\n"
        "  --fps       frame rate (default 30)\n"
        "  --breach    for --shot / --bench: the moment the first breach is at its height\n"
        "  --schedule  print the passes (seconds after the epoch) for --size / --layout and exit\n");
    exit(1);
}

static long first_breach(void) {
    for (long p = 0; p < 64; p++) {
        pc_frame++;
        const Pass *P = pass_get(p);
        if (P->kind == K_BREACH && P->s_up >= 0) return p;
    }
    return -1;
}

static void print_schedule(void) {
    printf("a pass every %.1f s: the worm takes %.1f s to cross, then %.0f s of empty desert\n", DPASS, DPASS - GAP, GAP);
    for (long p = 0; p < 12; p++) {
        pc_frame++;
        const Pass *P = pass_get(p);
        printf("%7.1f s  pass %2ld: %-9s %s", P->T0, p, kind_name[P->kind], P->dir > 0 ? "left to right" : "right to left");
        if (P->kind == K_BREACH && P->s_up >= 0)
            printf(", out of the sand at %.1f s, highest at %.1f s, back in at %.1f s", P->T0 + (P->s_up - 1.2 * RM - P->s_start) / P->v,
                   breach_peak(P), P->T0 + (P->s_down - 1.2 * RM - P->s_start) / P->v);
        if (P->kind == K_TEXT) printf(", spelling %s", P->txt == T_TIME ? "the time" : "the host name");
        printf("\n");
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
        else if (!strcmp(a, "-text") && i + 1 < argc) opt_text = argv[++i];
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-schedule")) opt_schedule = 1;
        else if (!strcmp(a, "-breach")) opt_breach = 1;
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
    if (opt_schedule) {
        if (span_n > 0 && span_me < 0) { span_me = 0; span_front = front_output(); }
        domain_init(off_w, off_h);
        world_init();
        print_schedule();
        return 0;
    }
    int offscreen = opt_shot || opt_bench > 0;
    pick_palette();
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
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
        /* a window gets its real size from the WM: lay the desert out for
         * the screen it will most likely fill */
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Burrow");
        XClassHint chint = {"burrow", "Burrow"};
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
    if (offscreen && opt_breach) {
        long b = first_breach();
        if (b >= 0) opt_at = breach_peak(pass_get(b));
    }

    /* --epoch: when the first pass began, shared by every instance; a
     * --shot or --bench then shows the moment --at seconds after it */
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
        long p = (long)floor(fmax(opt_at, 0) / DPASS);
        const char *kn = kind_name[pass_get(p)->kind];
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "burrow: %s, %dx%d, %.1f s: pass %ld (%s), first frame %.0f ms\n", palettes[pal].name, W, H, opt_at, p, kn, first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9, csum = 0;
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
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6);
            if (ms > worst) worst = ms;
        }
        printf("burrow: %s, pass %ld (%s), %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, p, kn, W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("burrow:   GPU %.3f ms (best %.3f); CPU to submit %.3f ms\n", gsum / opt_bench, gmin, csum / opt_bench);
        printf("burrow: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
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
