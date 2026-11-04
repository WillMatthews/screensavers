/*
 * kluver — Klüver's form constants as a lock screen: the tunnels, spirals,
 * honeycomb lattices and cobwebs that recur in psychedelic (and migraine,
 * and hypnagogic) imagery, drawn flat, the way they are seen.
 *
 * The model is Bressloff, Cowan, Golubitsky, Thomas & Wiener (2001):
 * the hallucination is a pattern of activity in primary visual cortex,
 * a Turing-like planform (stripes, squares, hexagons) formed when the
 * cortex is destabilised, and what is "seen" is that planform carried
 * back to the visual field through the inverse retinocortical map, which
 * is close to a complex logarithm about the centre of gaze:
 *
 *     cortex (x, y)  =  (log r, theta)
 *
 * So a cortical plane wave cos(a log r + n theta) is, on screen,
 *     a != 0, n == 0   concentric rings      (the tunnel)
 *     a == 0, n != 0   radial spokes         (the funnel)
 *     both             n-armed log spirals   (the spiral)
 * three such waves at 120 degrees in the cortex are a honeycomb that
 * grows with eccentricity and winds into a spiral lattice, and a square
 * planform of rings and spokes, sagging between its spokes, is the cobweb.
 * The angular wavenumbers are integers so every pattern closes around the
 * centre of gaze.
 *
 * The activity is drawn as a Huichol-like stack of contour bands: each
 * level set of the field is a crisp, anti-aliased outline and the bands
 * between them take the colours of a small palette in turn, so a hexagonal
 * planform becomes nested rosettes. The level offset creeps, so the bands
 * pour slowly out of every cell; the whole visual field breathes (a slow
 * dilation of log r), drifts outwards and turns; the cortex is gently
 * warped by a slow noise flow so nothing is ever ruled.
 *
 * Over a six-minute cycle the trip comes up, peaks and comes down: at the
 * bottom of the cycle the bands follow a flat, marbled "surface" (an organic
 * field in screen space) that breathes; as the cortical planform gains
 * amplitude it crystallises out of that surface into geometry; at the peak
 * the form constants turn into one another (every 48 s, re-forming over
 * 16 s) and the contours are densest; on the way down the cortex warps
 * harder and the geometry melts back into the surface.
 *
 * Palettes (--palette, else from the seed): jewel, pastel, blacklight, oil.
 *
 * The time and how long the screen has been locked sit bottom left, on the
 * front output.
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch. Spanning: run one instance per monitor with the same --seed,
 * --epoch and --layout, and --output = this one's index; the field lives
 * in wall coordinates around a centre of gaze in the middle of the front
 * output (where the eyes are), so the side screens carry the wide, slow
 * outer rings and arms of the same pattern.
 *
 * Same plumbing as papercut: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame
 * offscreen, --bench N for timings. --form and --phase hold a form constant
 * or a point of the cycle for testing.
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
#include "../common/hidpi.h"

static int W = 1280, H = 800;
static float dpr = 1;                  /* buffer px per logical px: the output's scale in --layer, else 1 */
static int opt_root, opt_layer, opt_caption = 1, opt_schedule;
static const char *opt_font = "Inter";
static double opt_fps = 30;
static const char *opt_palette;
static const char *opt_form;
static const char *opt_phase;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;

#define MAX_OUT 8
static float span[MAX_OUT][4];
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "kluver: %s\n", what); exit(1); }

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
static double ease(double u) { u = clamp01(u); return u * u * u * (u * (u * 6 - 15) + 10); }

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
static int ri(Rng *r, int a, int b) { return a + (int)(rnd(r) * (b - a + 1)) % (b - a + 1); }

static double lat(int x, int y, uint32_t salt) {
    uint64_t h = mix64(((uint64_t)(uint32_t)x << 32 | (uint32_t)y) ^ ((uint64_t)salt << 17) ^ 0x5eedull);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
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


/* ---- the wall ---- */
static double fr[4];                   /* front output: x, y, w, h */
static double L;                       /* the unit of size: front output's shorter side */

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
    if (span_me >= 0) for (int k = 0; k < 4; k++) fr[k] = span[span_front][k];
    else { fr[0] = fr[1] = 0; fr[2] = dw; fr[3] = dh; }
    L = fmin(fr[2], fr[3]);
}

/* ==== the schedule ==== */
#define ACT 48.0                       /* seconds a form constant is held, including its re-forming */
#define MORPH 16.0                     /* the last MORPH seconds turn into the next one */
#define CYCLE 360.0
#define SKEW_MAX 1.0                   /* the most lopsided a lattice may be */
#define LINES 0.07                     /* isoline spacing, in units of the planform's peak */
#define CREEP 5.0                      /* seconds for the lines to creep one spacing */
#define REST 16.0                      /* fewer isolines than this per cortical unit: a flat place, let it rest */
#define WIRE 0.008                     /* the wire's half-width, cortical units (x r on screen) */                    /* come-up, peak, comedown */
enum { F_TUNNEL, F_SPIRAL, F_LATTICE, F_COBWEB, NFORM };
static const char *form_name[NFORM] = {"tunnel", "spiral", "lattice", "cobweb"};
static int forced_form = -1;
enum { P_COMEUP, P_PEAK, P_COMEDOWN, NPHASE };
static const char *phase_name[NPHASE] = {"comeup", "peak", "comedown"};
static const double phase_at[NPHASE] = {0.11, 0.45, 0.80};   /* where in the cycle --phase holds */
static int forced_phase = -1;

/* a shuffle bag of the forms, never repeating across the seam */
static int act_form(long a) {
    if (forced_form >= 0) return forced_form;
    long g = a / NFORM;
    int p[NFORM], prev_last = -1;
    for (long gg = g - 1; gg <= g; gg++) {
        Rng r = rng_for(0xba9, (uint64_t)gg, 0);
        for (int i = 0; i < NFORM; i++) p[i] = i;
        for (int i = NFORM - 1; i > 0; i--) {
            int j = (int)(rnd(&r) * (i + 1));
            int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        if (gg == g && p[0] == prev_last) { int t = p[0]; p[0] = p[1]; p[1] = t; }
        prev_last = p[NFORM - 1];
    }
    return p[a % NFORM];
}

/* a planform: cortical plane waves cos(kx log r + ky theta + phase), ky an
 * integer. Tier 0 is the planform itself; tier 1 its harmonics (sums and
 * differences of the base wavevectors, phases locked to theirs), which
 * fill the cells with stars and rosettes as the trip peaks. Plus the
 * cobweb's sag between its spokes. */
#define MAXW 10
typedef struct {
    int form, nw;
    float w[MAXW][4];                  /* kx, ky, amplitude, phase */
    int tier[MAXW];
    float sag, sagm, sagp;
    float feat;                        /* the level drawn strongest (normalised field) */
} Form;

static void wave(Form *f, int tier, double kx, double ky, double amp, double ph) {
    float *w = f->w[f->nw];
    f->tier[f->nw++] = tier;
    w[0] = (float)kx; w[1] = (float)ky; w[2] = (float)amp; w[3] = (float)fmod(ph, 2 * M_PI);
}
/* a harmonic: i*a + j*b of two existing waves */
static void harm(Form *f, int a, double i, int b, double j, double amp) {
    const float *A = f->w[a], *B = f->w[b];
    wave(f, 1, i * A[0] + j * B[0], i * A[1] + j * B[1], amp, i * A[3] + j * B[3] + 8 * M_PI);
}

static void act_build(long a, int try_, Form *f) {
    Rng r = rng_for(0xac7, (uint64_t)a, 1 + (uint64_t)try_);
    memset(f, 0, sizeof *f);
    f->form = act_form(a);
    double s = rnd(&r) < 0.5 ? -1 : 1;
    double hb = rnd(&r) < 0.5 ? -1 : 1, hc = rnd(&r) < 0.5 ? -1 : 1;
    switch (f->form) {
    case F_TUNNEL: {
        /* rings, sharpened by their harmonic; faint spokes segment them
         * into a mirror-symmetric lattice of lozenges as the trip peaks */
        int n = 2 * ri(&r, 6, 9);
        double q = n * rr(&r, 0.55, 0.7);
        wave(f, 0, q, 0, 1.0, rr(&r, 0, 2 * M_PI));
        wave(f, 1, 0, n, 0.10, rr(&r, 0, 2 * M_PI));
        harm(f, 0, 2, 0, 0, 0.30 * hb);
        harm(f, 0, 1, 1, 1, 0.13);
        harm(f, 0, 1, 1, -1, 0.13);
        f->feat = 0.5f;
        break;
    }
    case F_SPIRAL: {
        /* n log-spiral arms, their crests sharpened by the second and
         * third harmonics; a faint counter-spiral of the same order
         * diamonds them */
        int n = ri(&r, 5, 9);
        double q = n * rr(&r, 0.6, 1.1);
        wave(f, 0, q, s * n, 1.0, rr(&r, 0, 2 * M_PI));
        wave(f, 1, q, -s * n, 0.14, rr(&r, 0, 2 * M_PI));
        harm(f, 0, 2, 0, 0, 0.30 * hb);
        harm(f, 0, 3, 0, 0, 0.12 * hc);
        f->feat = 0.5f;
        break;
    }
    case F_LATTICE: {
        /* a honeycomb: three waves at 120 degrees in the cortex, the first
         * with n1 cells round, the second n2: twisted into a spiral
         * lattice unless n2 == 0. Harmonics: the root-3 shell (stars) and
         * the doubled shell (rosettes). */
        int n1 = ri(&r, 10, 14), n2 = (int)s * ri(&r, 0, 3);
        double kx = (2.0 * n2 + n1) / sqrt(3.0), ky = n1;
        double ph[3] = {rr(&r, 0, 2 * M_PI), rr(&r, 0, 2 * M_PI), 0};
        ph[2] = -ph[0] - ph[1] + (rnd(&r) < 0.5 ? 0 : M_PI);   /* a symmetric cell: maxima or minima on the centres */
        for (int i = 0; i < 3; i++) {
            double c = cos(i * 2 * M_PI / 3), sn = sin(i * 2 * M_PI / 3);
            wave(f, 0, kx * c - ky * sn, round(kx * sn + ky * c), 0.62, ph[i]);
        }
        for (int i = 0; i < 3; i++) harm(f, i, 1, (i + 1) % 3, -1, 0.30 * hb);
        for (int i = 0; i < 3; i++) harm(f, i, 2, i, 0, 0.20 * hc);
        f->feat = (float)(rnd(&r) < 0.5 ? 0.30 : 0.62);   /* quantile: the honeycomb's walls, or rings round its cells */
        break;
    }
    case F_COBWEB: {
        /* rings and spokes, cos(q log r) cos(m theta), sagging between the
         * 2m spokes; harmonics add finer threads */
        int m = ri(&r, 6, 10);
        double q = 2 * m * rr(&r, 0.6, 0.8), p0 = rr(&r, 0, 2 * M_PI), p1 = rr(&r, 0, 2 * M_PI);
        wave(f, 0, q, m, 0.8, p0 + p1);
        wave(f, 0, q, -m, 0.8, p0 - p1);
        harm(f, 0, 1, 1, 1, 0.30 * hb);     /* rings at 2q */
        harm(f, 0, 1, 1, -1, 0.26 * hc);    /* spokes at 2m */
        harm(f, 0, 2, 1, 1, 0.14);
        harm(f, 0, 1, 1, 2, 0.14);
        f->sag = (float)rr(&r, 0.10, 0.16);
        f->sagm = (float)m;
        f->sagp = (float)p1;
        f->feat = 0.5f;
        break;
    }
    }
}

static int cmp_f(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

/* an act's planform, with its feature level turned from a quantile of the
 * field (over a patch of cortex, at the peak) into a level */
static void act_params(long a, Form *f) {
    static long ca[2] = {-1, -1};
    static Form cf[2];
    for (int i = 0; i < 2; i++) if (ca[i] == a) { *f = cf[i]; return; }
    enum { NS = 64 };
    static float v[NS * NS];
    /* a lattice whose field is lopsided (skewed) draws as sparse polka
     * dots on an empty floor, or holes in an empty plateau: re-roll it
     * until it is a packed honeycomb */
    for (int k = 0;; k++) {
        act_build(a, k, f);
        double asum = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int i = 0; i < f->nw; i++) asum += fabs(f->w[i][2]);
        for (int y = 0; y < NS; y++)
            for (int x = 0; x < NS; x++) {
                double X = -2.5 + 2.5 * (x + 0.5) / NS, Y = 2 * M_PI * (y + 0.37) / NS, u = 0;
                double Xs = X + f->sag * fabs(cos(f->sagm * Y + f->sagp));
                for (int i = 0; i < f->nw; i++) u += f->w[i][2] * cos(f->w[i][0] * Xs + f->w[i][1] * Y + f->w[i][3]);
                v[y * NS + x] = (float)(u / asum);
                m1 += u / asum;
            }
        m1 /= NS * NS;
        for (int i = 0; i < NS * NS; i++) { double d = v[i] - m1; m2 += d * d; m3 += d * d * d; }
        m2 /= NS * NS; m3 /= NS * NS;
        double skew = m3 / pow(m2, 1.5);
        if (f->form != F_LATTICE || fabs(skew) < SKEW_MAX || k == 15) break;
    }
    qsort(v, NS * NS, sizeof v[0], cmp_f);
    f->feat = v[(int)(f->feat * (NS * NS - 1))];
    int slot = (int)(a & 1);
    ca[slot] = a; cf[slot] = *f;
}

/* the arc of the cycle: intensity 0.12 .. 1, and how far into the comedown */
static void arc(double t, double *I, double *cd) {
    double p = forced_phase >= 0 ? phase_at[forced_phase] : fmod(t, CYCLE) / CYCLE;
    if (p < 0.30) *I = 0.12 + 0.88 * ease(p / 0.30);
    else if (p < 0.62) *I = 1;
    else *I = 1 - 0.88 * ease((p - 0.62) / 0.38);
    *cd = p < 0.62 ? 0 : sin(M_PI * clamp01((p - 0.62) / 0.38));
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, 0.0, 1.0);
    });

static const char *FS_MAIN = "#version 330 core\n" GLSL(
    out vec4 oCol;
    uniform sampler2D uNoise;
    uniform vec3 uView;                /* world origin of this output, device px per world px */
    uniform vec2 uRes, uC;             /* device size; centre of gaze (world) */
    uniform float uL;
    uniform vec4 uW[20];               /* waves: kx, ky, amp, phase (the motion already in the phase) */
    uniform int uNW, uNA;              /* waves in all; the first uNA belong to form A */
    uniform vec4 uSag;                 /* sag A, m A, sag B, m B */
    uniform vec2 uSagPh;               /* the spokes' angular phase, A and B */
    uniform vec4 uDrift;               /* noise drifts, wrapped */
    uniform vec3 uMod;                 /* warp, surface amplitude, planform amplitude */
    uniform float uNorm;               /* 1 / the planform's peak */
    uniform float uDL, uCreep, uFeat;  /* line spacing, its creep (lines), the wire's level */
    uniform float uWire, uWireOn, uInkW;       /* the wire's half-width (cortical), line width (px) */
    uniform float uPx;                 /* device px per logical px: the thresholds below are logical */
    uniform float uSat, uTint, uRest;
    uniform vec3 uRamp[4], uWireCol, uGround;
    uniform int uNR, uFilm;

    const float TAU = 6.28318530718;

    /* thin-film interference: the colour reflected by a film of optical
     * thickness d (microns), dark where it is too thin to colour */
    vec3 film(float d) {
        vec3 c = 0.5 - 0.5 * cos(TAU * 2.0 * d / vec3(0.640, 0.540, 0.460));
        c = mix(vec3(dot(c, vec3(0.333))), c, 0.7);
        return mix(vec3(0.16, 0.15, 0.16), 0.15 + 0.8 * c, smoothstep(0.05, 0.16, d));
    }

    vec3 ramp(float u) {
        float lev = clamp(u * 0.8 + 0.5, 0.0, 1.0);
        if (uFilm == 1) return film(0.10 + 0.42 * lev);
        lev *= float(uNR - 1);
        int i = int(min(floor(lev), float(uNR - 2)));
        return mix(uRamp[i], uRamp[i + 1], lev - float(i));
    }

    void main() {
        vec2 dev = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 wp = uView.xy + dev / uView.z;
        vec2 p = (wp - uC) / uL;
        float r = max(length(p), 1e-5);

        /* the inverse retinocortical map: cortex = (log r, theta) */
        float X = log(r), Y = atan(p.y, p.x);

        /* a slow flow over the cortex: two octaves of noise, periodic in
         * theta (3 and 6 tiles round) */
        vec2 nc = vec2(X * 0.22 + uDrift.x, Y * (3.0 / TAU));
        vec2 w1 = textureLod(uNoise, nc, 0.0).gb - 0.5;
        vec2 w2 = textureLod(uNoise, nc * 2.0 + vec2(uDrift.y, 0.37), 0.0).gb - 0.5;
        vec2 wv = w1 + 0.5 * w2;
        X += uMod.x * wv.x;
        Y += uMod.x * wv.y * 0.6;

        /* the cobweb's threads sag between its spokes */
        float cA = cos(uSag.y * Y + uSagPh.x), cB = cos(uSag.w * Y + uSagPh.y);
        float XA = X + uSag.x * abs(cA), XB = X + uSag.z * abs(cB);
        float dA = -uSag.x * uSag.y * sin(uSag.y * Y + uSagPh.x) * sign(cA);
        float dB = -uSag.z * uSag.w * sin(uSag.w * Y + uSagPh.y) * sign(cB);

        /* the planform and its cortical gradient */
        float u = 0.0;
        vec2 gc = vec2(0.0);
        for (int i = 0; i < uNW; i++) {
            bool isA = i < uNA;
            float ar = uW[i].x * (isA ? XA : XB) + uW[i].y * Y + uW[i].w;
            u += uW[i].z * cos(ar);
            gc += -uW[i].z * sin(ar) * vec2(uW[i].x, uW[i].x * (isA ? dA : dB) + uW[i].y);
        }
        float gs = uMod.z * uNorm;
        u *= gs; gc *= gs;

        /* the surface the geometry grows out of: flat, marbled, in screen
         * space */
        vec2 sp = wp / uL;
        vec2 sw = textureLod(uNoise, sp * 0.31 + uDrift.zw, 0.0).gb - 0.5;
        float org = uMod.y * 2.2 * (textureLod(uNoise, sp * 0.23 + sw * 0.25 - uDrift.wz, 0.0).r - 0.5);
        u += org;

        /* fine isolines, coloured by level, creeping out of the cells */
        float sl = u / uDL + uCreep;
        float fl = max(fwidth(sl), 1e-4);
        float flL = fl * uPx;          /* the same, per logical px */
        float nl = floor(sl + 0.5);
        float dl = abs(sl - nl);
        float ul = (nl - uCreep) * uDL;
        float major = exp(-pow((ul - uFeat) / 0.10, 2.0));
        float wpx = uInkW * mix(0.8, 1.6, major);
        float cov = clamp(0.5 * wpx - dl / fl + 0.5, 0.0, 1.0);
        /* in flat places the lines wander: let them fade where they are
         * far apart; where they crowd, drop every other one, then every
         * other again, like the contours of a map */
        float rpx = r * uL * uView.z;  /* one cortical unit, in device px */
        cov *= mix(smoothstep(0.012, 0.045, flL) * smoothstep(uRest * 0.5, uRest, fl * rpx), 1.0, major);
        float m4 = mod(nl, 4.0);
        float lod = m4 == 0.0 ? 1.0 : m4 == 2.0 ? 1.0 - smoothstep(0.17, 0.30, flL) : 1.0 - smoothstep(0.08, 0.15, flL);
        cov *= max(lod, major);
        vec3 lm = ramp(u);
        vec3 lc = mix(lm, uWireCol, major * 0.7);
        vec3 base = mix(uGround, lm, uTint);
        /* finer than a couple of pixels (the centre of gaze): their average */
        float dense = smoothstep(0.45, 0.9, flL);
        vec3 col = mix(base, lc, mix(cov, min(0.3, 0.15 * wpx * fl), dense) * uSat);

        /* the wire: a double line of constant cortical width along the
         * feature level. The log map is conformal, so the gradient in
         * world px is the cortical one over r. */
        float rw = r * uL;
        vec2 er = p / r, et = vec2(-er.y, er.x);
        vec2 gw = (gc.x * er + gc.y * et) / rw + vec2(dFdx(org), -dFdy(org)) * uView.z;
        float ad = abs(u - uFeat) / max(length(gw), 1e-6) * uView.z;
        float hw = uWire * rw * uView.z;
        float two = smoothstep(1.2, 2.5, hw / uPx);
        float l1 = clamp(0.55 * uInkW - abs(ad - hw) + 0.5, 0.0, 1.0);
        float l0 = clamp(0.65 * uInkW - ad + 0.5, 0.0, 1.0);
        float wire = max(l1 * two, l0 * (1.0 - two)) * (1.0 - dense) * uWireOn;
        col = mix(col, uGround, clamp(hw + 1.5 * uInkW - ad + 0.5, 0.0, 1.0) * two * uSat * uWireOn);
        col = mix(col, uWireCol, wire * mix(0.6, 1.0, uSat));

        /* dither */
        vec2 h = fract(sin(vec2(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)), dot(gl_FragCoord.xy, vec2(39.3468, 11.135)))) * 43758.5453);
        col += (h.x - h.y) / 255.0;
        oCol = vec4(col, 1.0);
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

static GLuint prog_main, prog_text, vao_empty, vao_text, vbo_text, tex_noise, tex_atlas;

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


static double tile_vn(double x, double y, int Lc, uint32_t salt) {
    double fx = floor(x), fy = floor(y);
    int ix = (int)fx, iy = (int)fy;
    double u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    int x0 = ((ix % Lc) + Lc) % Lc, y0 = ((iy % Lc) + Lc) % Lc, x1 = (x0 + 1) % Lc, y1 = (y0 + 1) % Lc;
    double a = lat(x0, y0, salt), b = lat(x1, y0, salt), c = lat(x0, y1, salt), d = lat(x1, y1, salt);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}

/* R: a 3-octave fbm (the surface); G, B: smooth low-frequency noise (the
 * flow); all tileable */
#define NT 256
static void bake_noise(void) {
    unsigned char *t = malloc(NT * NT * 4);
    for (int y = 0; y < NT; y++)
        for (int x = 0; x < NT; x++) {
            double u = (double)x / NT, v = (double)y / NT, f = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 3; o++) {
                int Lc = 4 << o;
                f += amp * tile_vn(u * Lc, v * Lc, Lc, 900 + o);
                norm += amp; amp *= 0.45;
            }
            f /= norm;
            f = 0.5 + (f - 0.5) * 1.8;
            double g = 0.7 * tile_vn(u * 4, v * 4, 4, 911) + 0.3 * tile_vn(u * 8, v * 8, 8, 912);
            double b = 0.7 * tile_vn(u * 4, v * 4, 4, 913) + 0.3 * tile_vn(u * 8, v * 8, 8, 914);
            unsigned char *o = t + 4 * (y * NT + x);
            o[0] = (unsigned char)(clamp01(f) * 255 + 0.5);
            o[1] = (unsigned char)(clamp01(0.5 + (g - 0.5) * 1.6) * 255 + 0.5);
            o[2] = (unsigned char)(clamp01(0.5 + (b - 0.5) * 1.6) * 255 + 0.5);
            o[3] = 255;
        }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NT, NT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    free(t);
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
        fprintf(stderr, "kluver: shader:\n%s\n", log);
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

/* ---- palettes: the ground, the wire, the lines' colours from the lowest
 * level to the highest ---- */
static const struct palette {
    const char *name;
    int film, n;
    float ground[3], wire[3], ramp[4][3];
    float inkw, tint;                  /* line width (px); how much the lines' colour tints the ground */
} palettes[] = {
    /* sapphire, emerald and ruby lines and gold wire on black */
    {"jewel", 0, 3, {0.010f, 0.009f, 0.022f}, {0.95f, 0.75f, 0.38f},
     {{0.20f, 0.36f, 1.00f}, {0.10f, 0.85f, 0.62f}, {1.00f, 0.22f, 0.62f}}, 1.3f, 0.13f},
    /* watercolour inks and a sepia wire on cream paper */
    {"pastel", 0, 3, {0.955f, 0.93f, 0.865f}, {0.25f, 0.18f, 0.16f},
     {{0.30f, 0.44f, 0.68f}, {0.36f, 0.58f, 0.48f}, {0.80f, 0.40f, 0.44f}}, 1.4f, 0.12f},
    /* fluorescent poster inks on deep violet */
    {"blacklight", 0, 3, {0.06f, 0.015f, 0.14f}, {1.00f, 0.24f, 0.66f},
     {{0.25f, 0.30f, 1.00f}, {0.10f, 0.85f, 1.00f}, {1.00f, 0.55f, 0.10f}}, 1.4f, 0.10f},
    /* thin film on dark water, with a pale wire */
    {"oil", 1, 1, {0.018f, 0.02f, 0.028f}, {0.86f, 0.86f, 0.84f}, {{0}}, 1.3f, 0.05f},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float cap_col[3], cap_halo[3];

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "kluver: no palette \"%s\" (jewel, pastel, blacklight, oil)\n", opt_palette);
            exit(1);
        }
    } else {
        uint32_t h = opt_seed * 2654435761u;
        h ^= h >> 15;
        pal = (int)(h % NPAL);
    }
    /* the caption: light on a dark ground, dark on a light one, haloed in
     * the ground */
    const float *g = palettes[pal].ground;
    float lum = 0.3f * g[0] + 0.59f * g[1] + 0.11f * g[2];
    for (int c = 0; c < 3; c++) {
        cap_col[c] = lum < 0.45f ? 0.94f - 0.03f * c : 0.20f - 0.02f * c;
        cap_halo[c] = g[c];
    }
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
    cap_line[0][0] = 0;                 /* lay the caption out again at the new size */
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
    float halo[4] = {cap_halo[0], cap_halo[1], cap_halo[2], 0.30f}, col[4] = {cap_col[0], cap_col[1], cap_col[2], 0.92f};
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

static void gl_init(void) {
    prog_main = mk_prog(VS_FULL, FS_MAIN);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
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
    bake_noise();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

static double v_ox, v_oy, v_sc;
static void view_update(void) {
    if (span_me >= 0) {
        v_sc = W / span[span_me][2];
        v_ox = span[span_me][0]; v_oy = span[span_me][1];
    } else {
        v_sc = fmax(W / fr[2], H / fr[3]);
        v_ox = fr[0] + (fr[2] - W / v_sc) / 2; v_oy = fr[1] + (fr[3] - H / v_sc) / 2;
    }
}

static double wrap(double x, double m) { return x - m * floor(x / m); }

static void render(double t, double wall, GLuint target) {
    if (t < 0) t = 0;
    view_update();
    long a = (long)floor(t / ACT);
    double ts = t - a * ACT, mu = ease((ts - (ACT - MORPH)) / MORPH);
    Form A, B;
    act_params(a, &A);
    if (mu > 0) act_params(a + 1, &B);
    else { B = A; B.nw = 0; }

    double I, cd;
    arc(t, &I, &cd);

    /* motion: a steady outward drift and turn, with a breath on top, put
     * into the wave phases in double precision */
    double flow = 0.020 * t;
    double breath = (0.025 + 0.030 * I) * sin(2 * M_PI * t / 10.0);
    double rot = 0.006 * t;
    double fil = ease((I - 0.4) / 0.6);   /* the filigree comes in towards the peak */
    float wv[2 * MAXW][4];
    int nw = 0;
    double asum = 0;
    Form *fs[2] = {&A, &B};
    double wt[2] = {1 - mu, mu};
    for (int f = 0; f < 2; f++)
        for (int i = 0; i < fs[f]->nw; i++) {
            float *w = wv[nw++];
            const float *s = fs[f]->w[i];
            w[0] = s[0]; w[1] = s[1];
            w[2] = (float)(s[2] * wt[f] * (fs[f]->tier[i] ? fil : 1));
            w[3] = (float)wrap(s[3] - s[0] * (flow + breath) + s[1] * rot, 2 * M_PI);
            asum += fabs(w[2]);
        }
    const struct palette *P = &palettes[pal];
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    GLuint pr = prog_main;
    glUseProgram(pr);
    glUniform3f(U(pr, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(pr, "uRes"), (float)W, (float)H);
    glUniform2f(U(pr, "uC"), (float)(fr[0] + fr[2] / 2), (float)(fr[1] + fr[3] / 2));
    glUniform1f(U(pr, "uL"), (float)L);
    glUniform4fv(U(pr, "uW"), nw, &wv[0][0]);
    glUniform1i(U(pr, "uNW"), nw);
    glUniform1i(U(pr, "uNA"), A.nw);
    glUniform4f(U(pr, "uSag"), A.sag * (float)wt[0], A.sagm, B.sag * (float)wt[1], B.sagm);
    glUniform2f(U(pr, "uSagPh"), (float)wrap(A.sagp + A.sagm * rot, M_PI), (float)wrap(B.sagp + B.sagm * rot, M_PI));
    glUniform4f(U(pr, "uDrift"), (float)wrap(-0.004 * t, 1), (float)wrap(0.006 * t, 1), (float)wrap(0.0021 * t, 1),
                (float)wrap(0.0013 * t, 1));
    double warp = 0.008 + 0.07 * (1 - I) + 0.25 * cd;
    double org = 0.98 * pow(1 - I, 1.5);
    double geo = 0.15 + 0.85 * I;
    glUniform3f(U(pr, "uMod"), (float)warp, (float)org, (float)geo);
    glUniform1f(U(pr, "uNorm"), (float)(1.0 / fmax(asum, 1e-3)));
    glUniform1f(U(pr, "uDL"), (float)(LINES * (1 + 1.2 * (1 - I))));
    glUniform1f(U(pr, "uCreep"), (float)wrap(t / CREEP, 4));
    glUniform1f(U(pr, "uFeat"), (float)(A.feat * (1 - mu) + B.feat * mu));
    glUniform1f(U(pr, "uWire"), (float)WIRE);
    glUniform1f(U(pr, "uWireOn"), (float)ease((I - 0.3) / 0.5));
    /* line widths are logical px; v_sc beyond the output's scale is a --shot zoom */
    glUniform1f(U(pr, "uInkW"), (float)(P->inkw * dpr * fmax(1.0, v_sc / dpr * 0.75)));
    glUniform1f(U(pr, "uPx"), dpr);
    glUniform1f(U(pr, "uSat"), (float)(0.55 + 0.45 * I));
    glUniform1f(U(pr, "uTint"), P->tint);
    glUniform1f(U(pr, "uRest"), (float)REST);
    glUniform3fv(U(pr, "uRamp"), 4, &P->ramp[0][0]);
    glUniform1i(U(pr, "uNR"), P->n);
    glUniform1i(U(pr, "uFilm"), P->film);
    glUniform3fv(U(pr, "uWireCol"), 1, P->wire);
    glUniform3fv(U(pr, "uGround"), 1, P->ground);
    glUniform1i(U(pr, "uNoise"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    if (opt_caption && (span_me < 0 || span_me == span_front)) {
        build_atlas();
        int nv = caption(wall);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_text);
        glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(U(prog_text, "uAtlas"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glDrawArrays(GL_TRIANGLES, 0, nv);
        glDisable(GL_BLEND);
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
static struct hidpi hd;
static double opt_render_scale = 1;

static void wl_global(void *data, struct wl_registry *r, uint32_t name, const char *intf, uint32_t ver) {
    (void)data;
    if (hidpi_bind(&hd, r, name, intf, ver)) return;
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
    hidpi_logical(&hd, (int)w, (int)h);
    wl.configured = 1;
}
static void ls_closed(void *data, struct zwlr_layer_surface_v1 *ls) { (void)data; (void)ls; wl.closed = 1; }
static const struct zwlr_layer_surface_v1_listener ls_l = {ls_configure, ls_closed};

/* the buffer follows the logical size and the output's scale (hidpi.h) */
static void wl_resize(void) {
    if (!hidpi_take_change(&hd)) return;
    int w = W, h = H;
    hidpi_size(&hd, &w, &h);
    if (w == W && h == H) return;
    W = w; H = h;
    dpr = (float)hidpi_factor(&hd, W);
    if (wl.ew) wl_egl_window_resize(wl.ew, W, H, 0, 0);
}

static void wl_init(void) {
    hidpi_init(&hd, opt_render_scale);
    if (!(wl.d = wl_display_connect(NULL))) die("can't connect to the Wayland display");
    struct wl_registry *reg = wl_display_get_registry(wl.d);
    wl_registry_add_listener(reg, &wl_reg_l, NULL);
    wl_display_roundtrip(wl.d);
    if (!wl.comp || !wl.shell) die("the compositor has no wlr-layer-shell");
    wl.surf = wl_compositor_create_surface(wl.comp);
    hidpi_attach(&hd, wl.surf);
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "kluver");
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
    wl_display_roundtrip(wl.d);                    /* the preferred scale, before the first buffer */
    wl_resize();

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
    wl_resize();
    return !wl.closed;
}

static void usage(void) {
    fprintf(stderr,
        "usage: kluver [--window | --root | --layer] [--seed N] [--fps N]\n"
        "              [--palette jewel|pastel|blacklight|oil]\n"
        "              [--form tunnel|spiral|lattice|cobweb] [--phase comeup|peak|comedown]\n"
        "              [--no-caption] [--font FAMILY]\n"
        "              [--schedule] [--epoch UNIXTIME] [--layout 'x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "              [--render-scale F]\n"
        "  --form      hold one form constant (re-formed every %.0f s) instead of cycling\n"
        "  --phase     hold a point of the %.0f s come-up / peak / comedown cycle\n"
        "  --schedule  print what happens when (seconds after the epoch) and exit\n"
        "  --render-scale F  with --layer, draw F x the output's native pixels (0 < F <= 1; default 1)\n", ACT, CYCLE);
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
    for (long a = 0; a < 16; a++) {
        double I, cd;
        arc(a * ACT, &I, &cd);
        printf("%7.1f s  act %2ld: %-8s intensity %.2f%s  (turns into %s from %.1f s)\n", a * ACT, a, form_name[act_form(a)],
               I, cd > 0 ? "  comedown" : "", form_name[act_form(a + 1)], (a + 1) * ACT - MORPH);
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
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-form") && i + 1 < argc) opt_form = argv[++i];
        else if (!strcmp(a, "-phase") && i + 1 < argc) opt_phase = argv[++i];
        else if (!strcmp(a, "-schedule")) opt_schedule = 1;
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (opt_form) {
        for (forced_form = 0; forced_form < NFORM; forced_form++) if (!strcmp(form_name[forced_form], opt_form)) break;
        if (forced_form == NFORM) { fprintf(stderr, "kluver: no form \"%s\" (tunnel, spiral, lattice, cobweb)\n", opt_form); exit(1); }
    }
    if (opt_phase) {
        for (forced_phase = 0; forced_phase < NPHASE; forced_phase++) if (!strcmp(phase_name[forced_phase], opt_phase)) break;
        if (forced_phase == NPHASE) { fprintf(stderr, "kluver: no phase \"%s\" (comeup, peak, comedown)\n", opt_phase); exit(1); }
    }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_schedule) { print_schedule(); return 0; }
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
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Kluver");
        XClassHint chint = {"kluver", "Kluver"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    domain_init(dw, dh);
    gl_init();
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
        long a = (long)floor(fmax(opt_at, 0) / ACT);
        double I, cd;
        arc(opt_at, &I, &cd);
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "kluver: %s, %dx%d, %s (act %ld), intensity %.2f, first frame %.0f ms\n", palettes[pal].name, W, H,
                    form_name[act_form(a)], a, I, first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 30.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6);
            if (ms > worst) worst = ms;
        }
        printf("kluver: %s, %s, %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, form_name[act_form(a)], W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("kluver:   GPU %.3f ms (best %.3f)\n", gsum / opt_bench, gmin);
        printf("kluver: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
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
