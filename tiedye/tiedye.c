/*
 * tiedye — tie-dyed cotton as a lock screen, made the way the real thing is
 * made: the cloth is folded and bound, dye is squirted on (or the bundle is
 * dipped), the dye wicks through the damp cotton, and the cloth is opened
 * out flat. A flat print of the result, with restraint: no lighting, no
 * drape, just dye on white cotton and the faintest grain.
 *
 * Each piece is made in three stages:
 *
 *   1. folded space: a GPU simulation of the dye in the folded bundle. Each
 *      fold has its own folded-space domain (the twisted disc of a spiral,
 *      the bound tube of a bullseye).
 *      Dye and water are laid down there and the water spreads by capillary
 *      (porous-medium) diffusion through cotton of uneven permeability,
 *      carrying the dye with it a little slower than itself; bindings and
 *      clamps are cotton the water can hardly enter. The simulation also
 *      records when the dye reached each point.
 *   2. unfolding: for every point of the flat cloth, where it lay in the
 *      bundle and how buried it was there (the doubled-back middle of each
 *      pleat and the crease valleys take less dye: the white veins). The
 *      lookup point is jittered along the warp and the weft, a little
 *      differently on every yarn: the feathery, fibrous wicked edges.
 *   3. the print: dyes mix subtractively (Kubelka-Munk, per channel), with a
 *      darker tide line where a wet front stopped, over natural white cotton
 *      with a barely-there grain.
 *
 * The finished piece, and when the dye reached each pixel, are rendered
 * once into a cache (spread over the first frames of the act, while the
 * cloth is still white); every frame after that is one cheap pass over it.
 *
 * Each act (ACT seconds): a fresh white cloth; the dye goes on, squirt by
 * squirt from the centre of the fold outwards (wedge after wedge round a
 * spiral, band after band down a bullseye), the wet fronts creeping on as it wicks, the
 * cloth dark and wet; it dries to its true colours; it is held; then it
 * fades back to a white cloth and the next is folded.
 *
 * Folds (--fold): spiral (pinched at the centre, twisted flat and squirted
 * in wedges) and bullseye (pulled up into a tube by its centre, bound with
 * rubber bands and squirted in bands).
 * Palettes (--palette): pastel, rainbow, sunset, ocean, indigo.
 * By default both are picked per act from the seed: pastel most often.
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch. Spanning (like papercut): run one instance per monitor with the
 * same --seed, --epoch and --layout, and --output = this one's index; the
 * cloth lives in wall coordinates, so it is one piece of cotton across the
 * bezels, its fold centred on the "front" output (the one under the middle
 * of the layout), which also carries the caption.
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
static int opt_root, opt_layer, opt_caption = 1, opt_schedule;
static double opt_fps = 30;
static const char *opt_font = "Inter";
static const char *opt_palette, *opt_fold;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 90;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;

/* ---- the act ---- */
#define ACT 150.0                      /* seconds per piece */
#define T_WHITE 3.0                    /* a fresh white cloth (the piece is made meanwhile) */
#define T_BLOOM 30.0                   /* the dye goes on and wicks in */
#define T_DRY 8.0                      /* the wet cloth dries to its true colours */
#define T_FADE 6.0                     /* back to white at the end */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "tiedye: %s\n", what); exit(1); }

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
static int ri(Rng *r, int n) { return (int)(rnd(r) * n) % n; }

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
 * restart of the saver doesn't reset it), else now */
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
 * World units are logical layout pixels (y down). The front rect is the
 * output under the middle of the layout; its shorter side L is the unit the
 * cloth is measured in, and the fold sits on its centre. */
static double bx0, by0, bx1, by1;
static double fr[4];
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
    L = fmin(fr[2], fr[3]);
}

/* ---- palettes: each dye's colour at full strength (sRGB) ---- */
#define MAXDYE 6
static const struct palette {
    const char *name;
    int n;
    float col[MAXDYE][3];
    float strength;                    /* how much dye goes on: < 1 for dilute (pastel) */
    float cloth[3];                    /* the undyed cotton */
} palettes[] = {
    {"pastel",  5, {{0.90f, 0.42f, 0.55f}, {0.96f, 0.62f, 0.38f}, {0.95f, 0.84f, 0.42f}, {0.45f, 0.72f, 0.60f}, {0.50f, 0.58f, 0.85f}}, 0.38f, {0.955f, 0.94f, 0.905f}},
    {"rainbow", 6, {{0.86f, 0.10f, 0.47f}, {0.97f, 0.42f, 0.08f}, {0.98f, 0.83f, 0.10f},
                    {0.03f, 0.62f, 0.66f}, {0.10f, 0.25f, 0.66f}, {0.40f, 0.16f, 0.56f}}, 1.0f, {0.955f, 0.94f, 0.905f}},
    {"sunset",  4, {{0.80f, 0.08f, 0.42f}, {0.96f, 0.42f, 0.32f}, {0.97f, 0.70f, 0.10f}, {0.52f, 0.12f, 0.42f}}, 1.0f, {0.955f, 0.94f, 0.905f}},
    {"ocean",   4, {{0.13f, 0.17f, 0.45f}, {0.02f, 0.45f, 0.52f}, {0.35f, 0.76f, 0.80f}, {0.10f, 0.55f, 0.47f}}, 1.0f, {0.955f, 0.94f, 0.905f}},
    {"indigo",  1, {{0.09f, 0.13f, 0.32f}}, 1.0f, {0.955f, 0.94f, 0.905f}},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))

/* ---- folds ---- */
enum { F_SPIRAL, F_BULLSEYE, NFOLD };
static const char *fold_name[NFOLD] = {"spiral", "bullseye"};
static int forced_fold = -1, forced_pal = -1;

/* this act's piece */
static int fold, pal;
#define NF 8
#define NH 10
static float uF[NF][4];                /* fold parameters (meaning per fold) */
static float uI[1][4];                 /* the init pass's parameters */
static float uH[NH][4];                /* pleat harmonics: k, amplitude, phase, drift */
static float uWD[16];                  /* dye per wedge / band (-1: none) */
static float uBand[16];                /* bullseye: binding positions along the tube */
static float uOrd[4];                  /* the order the dye goes on */
static int sim_w, sim_h, sim_steps, sim_polar;
static float sim_D, sim_m, sim_ret, sim_dm, sim_kx;
static float ks[MAXDYE][3], ks_cloth[3], cloth_lin[3];
static float fib_comb, fib_mid, tide;
static float cap_col[3], cap_halo[3];

/* the fold of act a: spirals and bullseyes in turn, three spirals to two
 * (S B S B S, S B S B S, ...) */
static int act_fold(long a) {
    if (forced_fold >= 0) return forced_fold;
    long k = ((a % 5) + 5) % 5;
    return k & 1 ? F_BULLSEYE : F_SPIRAL;
}
/* the palette of act a: dealt from a shuffled bag of eight (pastel three
 * times, rainbow twice, sunset, ocean, indigo), never the same twice
 * running, across bags too */
static const int pal_bag[8] = {0, 0, 0, 1, 1, 2, 3, 4};
static void deal(long b, int out[8]) {
    Rng r = rng_for((uint64_t)b, 0xba9, 0);
    for (int tries = 0; tries < 64; tries++) {
        memcpy(out, pal_bag, sizeof pal_bag);
        for (int i = 7; i > 0; i--) { int j = ri(&r, i + 1), t = out[i]; out[i] = out[j]; out[j] = t; }
        int ok = 1;
        for (int i = 1; i < 8; i++) if (out[i] == out[i - 1]) ok = 0;
        if (ok) break;
    }
}
static int act_pal(long a) {
    if (forced_pal >= 0) return forced_pal;
    long b = (long)floor(a / 8.0);
    int bag[8];
    deal(b, bag);
    int i = (int)(a - b * 8), prev[8];
    /* if the bag's first would repeat the last bag's last, swap its first two */
    deal(b - 1, prev);
    if (i < 2 && bag[0] == prev[7]) i ^= 1;
    return bag[i];
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, 0.0, 1.0);
    });

/* hashing and noise shared by every pass */
static const char *LIB = GLSL(
    uint hsh(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
    float h2(ivec2 i, uint s) { return float(hsh(uint(i.x) * 0x8da6b343u ^ hsh(uint(i.y) * 0xd8163841u ^ s))) * (1.0 / 4294967296.0); }
    float vnoise(vec2 p, uint s) {
        vec2 i = floor(p), f = p - i;
        vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        ivec2 ii = ivec2(i);
        return mix(mix(h2(ii, s), h2(ii + ivec2(1, 0), s), u.x), mix(h2(ii + ivec2(0, 1), s), h2(ii + ivec2(1, 1), s), u.x), u.y);
    }
    float fbm(vec2 p, uint s) {
        float a = 0.5, f = 0.0;
        for (int o = 0; o < 5; o++) { f += a * vnoise(p, s + uint(o) * 101u); p = mat2(1.6, 1.2, -1.2, 1.6) * p + vec2(17.1, 9.3); a *= 0.5; }
        return f / 0.96875;
    }
    float fbm3(vec2 p, uint s) {
        float a = 0.5, f = 0.0;
        for (int o = 0; o < 3; o++) { f += a * vnoise(p, s + uint(o) * 101u); p = mat2(1.6, 1.2, -1.2, 1.6) * p + vec2(17.1, 9.3); a *= 0.5; }
        return f / 0.875;
    }
    const float TAU = 6.2831853;
);

/* stage 1a: lay the dye and water down in folded space, and the cotton's
 * permeability there (0 under a binding or a clamp). B.b is when the dye
 * got there (as a fraction of the simulation; 1: not yet) */
static const char *FS_INIT_MAIN = GLSL(
    uniform int uFold;
    uniform vec2 uSim;
    uniform vec4 uI[1];
    uniform float uWD[16], uBand[16];
    layout(location = 0) out vec4 oA;
    layout(location = 1) out vec4 oB;
    layout(location = 2) out vec4 oP;
    float dye[6];
    void put(float k, float amt) {
        int i = int(k + 0.5);
        if (k < -0.5) return;
        for (int j = 0; j < 6; j++) if (j == i) dye[j] += amt;
    }
    void main() {
        vec2 q = gl_FragCoord.xy / uSim;
        for (int j = 0; j < 6; j++) dye[j] = 0.0;
        float water = uI[0].z, perm = 0.45 + 0.8 * fbm(q * vec2(uSim.x / uSim.y, 1.0) * 9.0, 7u);
        if (uFold == 0) {
            /* spiral: pie wedges squirted on the twisted disc */
            vec2 disc = q.y * vec2(cos(TAU * q.x), sin(TAU * q.x));
            float nW = uI[0].x, gap = uI[0].y;
            float x = q.x + 0.020 * (fbm(disc * 5.0, 3u) - 0.5) / max(q.y, 0.15);
            float wf = fract(x) * nW, w = floor(wf), f = wf - w;
            float amt = 0.80 + uI[0].w * (fbm(disc * 7.0, 4u) - 0.5);
            /* missed spots: the bottle didn't reach everywhere */
            amt *= smoothstep(0.18, 0.32, fbm(disc * 4.0 + 3.1, 5u) + 0.25 * q.y);
            amt *= smoothstep(gap * 0.5, gap * 0.5 + 0.05, f) * smoothstep(gap * 0.5, gap * 0.5 + 0.05, 1.0 - f);
            int wi = int(w);
            float k = -1.0;
            for (int j = 0; j < 16; j++) if (j == wi) k = uWD[j];
            if (k > -0.5) { put(k, amt); water = max(water, amt); }
        } else if (uFold == 1) {
            /* bullseye: bands squirted between the rubber bands on the tube */
            float t = q.y, nb = uI[0].x;
            vec2 ring = vec2(cos(TAU * q.x), sin(TAU * q.x));
            /* each rubber band: its width varies round the tube, and where
             * it sat loose the dye crept under it */
            float bw = uI[0].w * (0.45 + 1.1 * fbm(ring * 1.8 + t * 3.0, 6u));
            float loose = smoothstep(0.55, 0.75, fbm(ring * 1.3 + t * 5.0 + 4.0, 10u));
            int band = 0;
            float dmin = 1.0;
            for (int j = 0; j < 16; j++) {
                if (float(j) >= nb) break;
                if (t > uBand[j]) band = j + 1;
                dmin = min(dmin, abs(t - uBand[j]));
            }
            float k = -1.0;
            for (int j = 0; j < 16; j++) if (j == band) k = uWD[j];
            float amt = (0.82 + uI[0].y * (fbm(ring * 2.0 + t * 9.0, 4u) - 0.5)) * smoothstep(bw * 0.5, bw * 0.5 + 0.015, dmin);
            if (k > -0.5) { put(k, amt); water = max(water, amt); }
            if (k > -0.5 && dmin < bw * 0.5) put(k, 0.45 * loose * smoothstep(0.0, bw * 0.5, dmin));
            perm *= mix(mix(0.01, 0.35, loose), 1.0, smoothstep(bw * 0.35, bw * 0.55, dmin));
        }
        float tot = dye[0] + dye[1] + dye[2] + dye[3] + dye[4] + dye[5];
        oA = vec4(dye[0], dye[1], dye[2], dye[3]);
        oB = vec4(dye[4], dye[5], tot > 0.05 ? 0.0 : 1.0, water);
        oP = vec4(clamp(perm, 0.0, 1.0), 0.0, 0.0, 1.0);
    });

/* stage 1b: one step of capillary wicking. Water moves down its own
 * gradient with a diffusivity that grows with wetness (porous-medium
 * diffusion: a sharp wet front), the dye rides along with it, a little
 * slower (it sticks to the fibre), and diffuses a touch on its own where
 * the cloth is wet. Fluxes are per face, so everything is conserved. */
static const char *FS_STEP_MAIN = GLSL(
    uniform sampler2D uA, uB, uP;
    uniform float uD, uM, uRet, uDm, uKx, uFrac;
    uniform int uPolar;
    layout(location = 0) out vec4 oA;
    layout(location = 1) out vec4 oB;
    void main() {
        ivec2 c = ivec2(gl_FragCoord.xy), N = textureSize(uA, 0);
        vec4 a = texelFetch(uA, c, 0), b = texelFetch(uB, c, 0), p = texelFetch(uP, c, 0);
        float w = b.a, rho = (float(c.y) + 0.5) / float(N.y);
        float kx = uPolar == 1 ? min(1.0, 1.0 / pow(TAU * max(rho, 0.02) * float(N.y) / float(N.x), 2.0)) : uKx;
        vec4 da = vec4(0.0);
        vec2 db = vec2(0.0);
        float dw = 0.0;
        vec4 ca0 = a / max(w, 1e-4);
        vec2 cb0 = b.rg / max(w, 1e-4);
        for (int i = 0; i < 4; i++) {
            ivec2 o = i == 0 ? ivec2(1, 0) : i == 1 ? ivec2(-1, 0) : i == 2 ? ivec2(0, 1) : ivec2(0, -1);
            ivec2 n = c + o;
            if (uPolar > 0) n.x = (n.x + N.x) % N.x; else n.x = clamp(n.x, 0, N.x - 1);
            n.y = clamp(n.y, 0, N.y - 1);
            if (n == c) continue;
            vec4 an = texelFetch(uA, n, 0), bn = texelFetch(uB, n, 0);
            float k = (i < 2 ? kx : 1.0) * min(p.r, texelFetch(uP, n, 0).r);
            float wn = bn.a, wm = 0.5 * (w + wn);
            float F = uD * k * pow(wm, uM) * (wn - w);
            vec4 ca = min(F > 0.0 ? an / max(wn, 1e-4) : ca0, vec4(1.5));
            vec2 cb = min(F > 0.0 ? bn.rg / max(wn, 1e-4) : cb0, vec2(1.5));
            float dm = uDm * k * smoothstep(0.08, 0.35, min(w, wn));
            da += F * uRet * ca + dm * (an - a);
            db += F * uRet * cb + dm * (bn.rg - b.rg);
            dw += F;
        }
        vec4 na = max(a + da, 0.0);
        vec2 nb = max(b.rg + db, 0.0);
        float tot = na.r + na.g + na.b + na.a + nb.r + nb.g;
        float arr = b.b < 0.999 ? b.b : tot > 0.05 ? uFrac : 1.0;
        oA = na;
        oB = vec4(nb, arr, max(w + dw, 0.0));
    });

/* stages 2 and 3: unfold and print, into the cache: rgb the finished
 * piece (linear), a when its dye arrived (0..1 over the bloom; > 1 never) */
static const char *FS_COMP_MAIN = GLSL(
    uniform sampler2D uSA, uSB;
    uniform int uFold, uND;
    uniform vec4 uF[8];
    uniform vec4 uH[10];
    uniform vec4 uOrd;
    uniform vec3 uKS[6];
    uniform vec3 uKSc;
    uniform float uStr, uTP, uTide, uComb, uMid;
    uniform vec4 uView;                  /* ox, oy, scale, device height */
    uniform vec3 uFC;                    /* front centre (world), L */
    out vec4 frag;

    /* yarn wicking: each warp yarn (vertical) carries dye a different
     * distance along itself, and each weft yarn likewise across; plus a
     * mid-scale fibrous wander of the wet front */
    vec2 wick(vec2 p) {
        float dy = vnoise(vec2(p.x / uTP, p.y / (uTP * 10.0)), 11u) - 0.5;
        float dx = vnoise(vec2(p.x / (uTP * 10.0), p.y / uTP), 12u) - 0.5;
        dy += 0.5 * (vnoise(vec2(p.x / (uTP * 0.5), p.y / (uTP * 40.0)), 15u) - 0.5);
        dx += 0.5 * (vnoise(vec2(p.x / (uTP * 40.0), p.y / (uTP * 0.5)), 16u) - 0.5);
        vec2 m = vec2(fbm(p * 30.0, 13u), fbm(p * 30.0, 14u)) - 0.5;
        m += 0.5 * (vec2(fbm(p * 110.0, 17u), fbm(p * 110.0, 18u)) - 0.5);
        return uComb * vec2(dx, dy) + uMid * m;
    }

    float pl;
    void pleats(float ph, float r) {
        pl = 0.0;
        for (int i = 0; i < 10; i++) pl += uH[i].y * sin(uH[i].x * ph + uH[i].z + uH[i].w * r);
    }

    /* pleats are Z folds: the cloth runs forward (the long leg), doubles
     * back (the short one, sandwiched in the middle) and runs forward
     * again. Their spacing wanders (harmonics with integer k keep it
     * periodic round the circle), each pleat has its own depth, the peaks
     * are a little rounded, and finer folds sit inside the big ones */
    float smin(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
    float zigf(float psi, float a, float k, out float u, out float id) {
        id = floor(psi); u = psi - id;
        return smin(u / a, (1.0 - u) / (1.0 - a), k);
    }
    float spiral_phi(float th, float r, vec2 d, out float cd) {
        float tw = uF[0].w * pow(r, 0.8) + 0.5 * (fbm3(vec2(r * 2.0, 0.5), 21u) - 0.5);
        float ph = th + uF[1].x * r;
        pleats(ph, r);
        float N = uF[2].w, a = uF[3].x, u, id;
        float psi = N * ph / TAU + uF[3].y * pl + 0.8 * fbm3(vec2(r * 3.0, 1.5), 26u);
        float zig = zigf(psi, a, 0.12, u, id);
        float A = uF[1].y * smoothstep(0.0, 0.25, r) * (0.45 + 0.9 * fbm3(d * 3.0, 22u));
        A *= 0.35 + 1.3 * h2(ivec2(int(mod(id, N)), 3), 27u);
        cd = min(u, abs(u - a));
        float N2 = floor(N * 2.7), u2, id2;
        float psi2 = N2 * (th + uF[1].x * 1.4 * r) / TAU + 0.3 * pl + 1.5 * fbm3(vec2(r * 4.0, 2.5), 28u);
        float zig2 = zigf(psi2, a, 0.15, u2, id2);
        float A2 = 0.7 * uF[1].y * (0.1 + 1.3 * fbm3(d * 5.0 + 3.0, 25u)) * h2(ivec2(int(mod(id2, N2)), 4), 29u);
        return th + tw + A * TAU / N * zig + A2 * TAU / N2 * zig2;
    }

    /* where this point of the flat cloth lay in the bundle: q in xy,
     * penetration in z, a per-layer shift of the dose in w */
    vec4 unfold(vec2 p) {
        if (uFold == 0) {
            vec2 d = p - uF[0].xy;
            float r = length(d) / uF[0].z, th = atan(d.y, d.x);
            float cd, cd1;
            float phi = spiral_phi(th, r, d, cd);
            /* d phi / d theta: < 0 on a fold-back */
            float eps = 0.0015 / max(r, 0.02);
            vec2 d1 = r * uF[0].z * vec2(cos(th + eps), sin(th + eps));
            float s = (spiral_phi(th + eps, r, d1, cd1) - phi) / eps;
            float mid = smoothstep(0.2, -0.7, s);
            float crease = exp(-cd * cd / 0.0016);
            float pen = 1.0 - uF[1].z * mid - uF[1].w * crease * smoothstep(0.03, 0.15, r);
            pen *= mix(uF[2].x, 1.0, smoothstep(0.0, uF[2].y, r));   /* the tight pinch */
            return vec4(fract(phi / TAU), clamp(r * uF[2].z, 0.002, 0.998), pen, 0.0);
        } else {
            vec2 d = p - uF[0].xy;
            float r = length(d) / uF[0].z, th = atan(d.y, d.x);
            /* pulled up into a tube: the cloth gathers in long pleats down
             * it, so each band's edge is pulled into teeth and the pleats'
             * insides are paler streaks */
            float ph = th + uF[1].x * r;
            pleats(ph, r);
            float N = uF[2].w, u, id;
            float psi = N * ph / TAU + 0.10 * pl + 0.6 * fbm3(vec2(r * 3.0, 0.5), 26u);
            float zig = zigf(psi, uF[3].x, 0.10, u, id);
            float A = uF[1].y * (0.4 + 1.2 * h2(ivec2(int(mod(id, N)), 5), 27u)) * (0.6 + 0.8 * fbm3(d * 3.0, 23u));
            float t = r * (1.0 + 0.10 * (fbm3(d * 1.6, 24u) - 0.5)) + A * min(r, 0.6) * (zig - 0.5);
            float valley = 1.0 - smoothstep(0.0, 0.3, zig);
            float pen = 1.0 - uF[1].z * valley * smoothstep(0.02, 0.1, r);
            pen *= mix(uF[2].x, 1.0, smoothstep(0.0, uF[2].y, r));
            return vec4(fract(th / TAU), clamp(t * uF[2].z, 0.002, 0.998), pen, 0.0);
        }
    }

    /* cubic B-spline sampling: smooth, no bilinear creases in the field */
    vec4 tex_bs(sampler2D t, vec2 uv) {
        vec2 ts = vec2(textureSize(t, 0));
        vec2 st = uv * ts - 0.5, i = floor(st), f = st - i;
        vec2 f2 = f * f, f3 = f2 * f;
        vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
        vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
        vec2 g0 = w0 + w1, g1 = w2 + w3;
        vec2 h0 = (i - 0.5 + w1 / g0) / ts, h1 = (i + 1.5 + w3 / g1) / ts;
        return g0.y * (g0.x * texture(t, h0) + g1.x * texture(t, vec2(h1.x, h0.y)))
             + g1.y * (g0.x * texture(t, vec2(h0.x, h1.y)) + g1.x * texture(t, h1));
    }
    vec3 km(vec3 k) { return 1.0 + k - sqrt(k * k + 2.0 * k); }

    vec2 cloth_at(vec2 frag) {
        vec2 w = uView.xy + vec2(frag.x, uView.w - frag.y) / uView.z;
        return vec2(w.x - uFC.x, uFC.y - w.y) / uFC.z;
    }

    void main() {
        vec2 p0 = cloth_at(gl_FragCoord.xy);
        vec4 m0 = unfold(p0 + wick(p0));
        vec4 A = tex_bs(uSA, m0.xy), B = tex_bs(uSB, m0.xy);
        float dose[6] = float[6](A.r, A.g, A.b, A.a, B.r, B.g);
        float px = 1.0 / (uView.z * uFC.z);
        /* uneven soak: some parts of the bundle simply took more */
        float soak = 0.70 + 0.45 * fbm(p0 * 2.5 + 7.0, 61u) + 0.25 * (fbm3(p0 * 11.0, 62u) - 0.5);
        float raw = 0.0;
        for (int k = 0; k < 6; k++) if (k < uND) raw += dose[k];
        float tl = 1.0 + uTide / max(raw, 0.03) * smoothstep(0.03, 0.10, raw) * (1.0 - smoothstep(0.10, 0.30, raw));
        vec3 acc = vec3(0.0);
        for (int sidx = 0; sidx < 4; sidx++) {
            vec2 off = vec2(sidx & 1, sidx >> 1) - 0.5;
            off = mat2(0.894, 0.447, -0.447, 0.894) * off * 0.5;
            vec2 p = p0 + off * px;
            /* plain weave: each yarn's appetite for dye varies, yarn to
             * yarn and slub to slub */
            vec2 tq = p / uTP;
            float over = 0.5 + 0.5 * sin(3.14159 * tq.x) * sin(3.14159 * tq.y);
            float yw = vnoise(vec2(tq.x, tq.y / 24.0), 51u) + 0.6 * vnoise(vec2(tq.x * 0.5, tq.y / 6.0), 53u);
            float yf = vnoise(vec2(tq.x / 24.0, tq.y), 52u) + 0.6 * vnoise(vec2(tq.x / 6.0, tq.y * 0.5), 54u);
            float yarn = mix(yf, yw, 0.5 + 0.3 * (over - 0.5)) / 1.6 - 0.5;
            /* fuzz: loose fibres at the very edge of the wet front */
            float fibre = 0.7 * (vnoise(tq * 1.4, 55u) - 0.5) + 0.3 * ((yw + yf) / 3.2 - 0.5);
            float gap = 1.0 - max(abs(sin(3.14159 * tq.x)), abs(sin(3.14159 * tq.y)));
            float lift = m0.z * (1.0 + m0.w) * soak * (1.0 + 0.14 * yarn);
            /* the feathered wet front: where the dye ran thin it didn't
             * take, and where it stopped it piled up (the tide line) */
            float take = smoothstep(0.035, 0.12, raw * lift + 0.10 * fibre);
            vec3 K = uKSc;
            for (int k = 0; k < 6; k++) {
                if (k >= uND) break;
                K += uStr * dose[k] * lift * take * tl * uKS[k];
            }
            vec3 R = km(K);
            R *= 1.0 - 0.012 * gap - 0.015 * yarn;
            acc += R;
        }
        /* when the dye got here: the order it went on (centre outwards,
         * wedge by wedge or band by band), then how long it took to wick */
        float sim = B.b;
        float order = 0.0;
        if (uFold == 0) {
            float w = floor(fract(m0.x) * uOrd.x);
            order = (mod(w - uOrd.y, uOrd.x) + 0.8 * m0.y) / uOrd.x;
        } else order = m0.y;
        float arr = 0.62 * order + 0.36 * sim;
        arr += 0.012 * (fbm3(p0 * 40.0, 63u) - 0.5);
        if (raw < 0.02) arr = 2.0;
        frag = vec4(acc * 0.25, arr);
    });

/* every frame: the cached piece, as far as the dye has got */
static const char *FS_SHOW_MAIN = GLSL(
    uniform sampler2D uC;
    uniform vec3 uCloth;                 /* linear */
    uniform float uT, uWet, uFade;
    out vec4 frag;
    float tosrgb(float c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
    void main() {
        vec4 c = texelFetch(uC, ivec2(gl_FragCoord.xy), 0);
        /* dyed where the dye has arrived, with a feathered front */
        float m = 1.0 - smoothstep(uT - 0.012, uT, c.a);
        /* the water runs a little ahead of the dye: damp cloth, greyer */
        float damp = uWet * (1.0 - smoothstep(uT, uT + 0.05, c.a)) * (1.0 - m);
        /* wet dye is darker and deeper; it lightens as it dries */
        vec3 dyed = pow(c.rgb, vec3(1.0 + 0.35 * uWet)) * (1.0 - 0.10 * uWet);
        vec3 col = mix(uCloth, dyed, m) * (1.0 - 0.06 * damp);
        col = mix(col, uCloth, uFade);
        frag = vec4(tosrgb(col.r), tosrgb(col.g), tosrgb(col.b), 1.0);
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
static const char *FS_TEXT = GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vCol;
    out vec4 o;
    void main() { o = vec4(vCol.rgb, vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r); });

static GLuint compile(GLenum type, const char *a, const char *b) {
    GLuint s = glCreateShader(type);
    const char *src[3] = {"#version 330 core\n", a, b};
    if (b) glShaderSource(s, 3, src, NULL);
    else glShaderSource(s, 1, &a, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "tiedye: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}
static GLuint link_prog(GLuint vs, GLuint fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, vs);
    glAttachShader(pr, fs);
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}
static GLuint mk_prog(const char *fs) {
    return link_prog(compile(GL_VERTEX_SHADER, VS_FULL, NULL), compile(GL_FRAGMENT_SHADER, LIB, fs));
}
#define U(p, n) glGetUniformLocation(p, n)

static GLuint prog_init, prog_step, prog_comp, prog_show, prog_text, vao, vao_text, vbo_text, tex_atlas;
static GLuint sim_tex[2][2], sim_perm, sim_fbo[2], init_fbo;
static GLuint cache_tex, cache_fbo;
static int cache_w, cache_h;

static void tex_params(GLint wrap_s, GLint filter) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
static void fbo_check(void) {
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete");
}

/* ---- the piece: fold, binding and dye placement for act a ---- */
static float srgb2lin(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
static float ks_of(float r) { r = srgb2lin(r); if (r < 0.002f) r = 0.002f; return (1 - r) * (1 - r) / (2 * r); }

/* the farthest the cloth reaches from (cx, cy), in cloth units */
static double reach(double cx, double cy) {
    double x0 = (bx0 - (fr[0] + fr[2] / 2)) / L, x1 = (bx1 - (fr[0] + fr[2] / 2)) / L;
    double y0 = ((fr[1] + fr[3] / 2) - by1) / L, y1 = ((fr[1] + fr[3] / 2) - by0) / L;
    double far = 0;
    for (int i = 0; i < 4; i++) {
        double dx = (i & 1 ? x1 : x0) - cx, dy = (i & 2 ? y1 : y0) - cy;
        far = fmax(far, sqrt(dx * dx + dy * dy));
    }
    return far;
}

static void setup(long a) {
    fold = act_fold(a);
    pal = act_pal(a);
    const struct palette *P = &palettes[pal];
    memset(ks, 0, sizeof ks);
    for (int k = 0; k < P->n; k++)
        for (int c = 0; c < 3; c++) ks[k][c] = (ks_of(P->col[k][c]) - ks_of(P->cloth[c])) / 2.0f;
    for (int c = 0; c < 3; c++) { ks_cloth[c] = ks_of(P->cloth[c]); cloth_lin[c] = srgb2lin(P->cloth[c]); }
    Rng r = rng_for((uint64_t)a, (uint64_t)fold, 0x71ed);
    double fa = fmax(fr[2], fr[3]) / L;  /* the front output's aspect */
    memset(uF, 0, sizeof uF); memset(uH, 0, sizeof uH); memset(uI, 0, sizeof uI); memset(uOrd, 0, sizeof uOrd);
    for (int i = 0; i < 16; i++) uWD[i] = -1;
    sim_polar = 0; sim_kx = 1; sim_D = 0.22f; sim_m = 1.6f; sim_ret = 0.85f; sim_dm = 0.015f;
    fib_comb = 4.5f / 1080; fib_mid = 11.0f / 1080; tide = 0.10f;
    int nd = P->n, off = ri(&r, nd);
    switch (fold) {
    case F_SPIRAL: {
        /* the centre a little off the middle of the front screen */
        double cx = rr(&r, -0.30, 0.30) * fa / 1.78, cy = rr(&r, -0.12, 0.12);
        uF[0][0] = (float)cx; uF[0][1] = (float)cy; uF[0][2] = (float)reach(cx, cy);
        uF[0][3] = (float)((rnd(&r) < 0.5 ? -1 : 1) * rr(&r, 4.0, 6.0));   /* twist */
        uF[1][0] = (float)(-uF[0][3] * rr(&r, 0.2, 0.6));                    /* pleat lines cross the arms */
        uF[1][1] = (float)rr(&r, 0.45, 0.7);                                 /* pleat throw, in pleat widths */
        uF[1][2] = 0.45f; uF[1][3] = 0.30f;                                  /* inner layer, crease resist */
        uF[2][0] = 0.6f; uF[2][1] = 0.06f; uF[2][2] = 1.0f;
        uF[2][3] = (float)(16 + ri(&r, 10));                                 /* pleats round the disc */
        uF[3][0] = (float)rr(&r, 0.62, 0.75);                                /* forward share of each Z */
        uF[3][1] = 0.30f;                                                    /* pleat spacing wander */
        int nW = nd >= 5 ? nd : nd == 1 ? 2 * (3 + ri(&r, 2)) : 2 * nd;
        uI[0][0] = (float)nW; uI[0][1] = (float)rr(&r, 0.02, 0.16); uI[0][2] = 0.18f; uI[0][3] = 0.6f;
        for (int i = 0; i < nW; i++) uWD[i] = nd == 1 ? (i & 1 ? -1.0f : 0.0f) : (float)((i + off) % nd);
        for (int i = 0; i < NH; i++) {
            uH[i][0] = (float)(5 + ri(&r, 12));
            uH[i][1] = (float)rr(&r, 0.3, 1.0);
            uH[i][2] = (float)rr(&r, 0, 2 * M_PI);
            uH[i][3] = (float)rr(&r, -9, 9);
        }
        double sa = 0;
        for (int i = 0; i < NH; i++) sa += uH[i][1] * uH[i][1];
        for (int i = 0; i < NH; i++) uH[i][1] /= (float)sqrt(sa / 2);
        uOrd[0] = (float)nW; uOrd[1] = (float)ri(&r, nW);
        sim_w = 640; sim_h = 512; sim_polar = 1; sim_steps = 1600;
        break;
    }
    default: {
        double cx = rr(&r, -0.25, 0.25) * fa / 1.78, cy = rr(&r, -0.10, 0.10);
        uF[0][0] = (float)cx; uF[0][1] = (float)cy; uF[0][2] = (float)reach(cx, cy);
        uF[1][0] = (float)rr(&r, -0.6, 0.6);
        uF[1][1] = (float)rr(&r, 0.035, 0.06);
        uF[1][2] = 0.5f;
        uF[2][0] = 0.7f; uF[2][1] = 0.04f; uF[2][2] = 1.0f;
        uF[2][3] = (float)(26 + ri(&r, 16));
        uF[3][0] = (float)rr(&r, 0.6, 0.8);
        int nb = 5 + ri(&r, 3);
        double t = 0;
        for (int i = 0; i < nb; i++) {
            t += rr(&r, 0.7, 1.3) / (nb + 0.6);
            uBand[i] = (float)(pow(t, 1.15));
        }
        for (int i = 0; i <= nb; i++) uWD[i] = nd == 1 ? (i & 1 ? -1.0f : 0.0f) : (float)((i + off) % nd);
        if (nd > 1 && rnd(&r) < 0.5) uWD[1 + ri(&r, nb - 1)] = -1;  /* a band left white */
        uI[0][0] = (float)nb; uI[0][1] = 0.35f; uI[0][2] = 0.18f; uI[0][3] = (float)rr(&r, 0.012, 0.022);
        for (int i = 0; i < NH; i++) {
            uH[i][0] = (float)(14 + ri(&r, 34));
            uH[i][1] = (float)rr(&r, 0.4, 1.0);
            uH[i][2] = (float)rr(&r, 0, 2 * M_PI);
            uH[i][3] = (float)rr(&r, -14, 14);
        }
        double sa = 0;
        for (int i = 0; i < NH; i++) sa += uH[i][1] * uH[i][1];
        for (int i = 0; i < NH; i++) uH[i][1] /= (float)sqrt(sa);
        sim_w = 512; sim_h = 640; sim_polar = 2; sim_kx = 0.25f; sim_steps = 900;
        break;
    }
    }
}

/* ---- making the piece, a little at a time ---- */
static long built_act = -1000;         /* the act the cache holds */
static long making = -1000;            /* the act being made */
static int make_stage, make_step, make_strip;
#define STRIPS 8
#define STEPS_PER_FRAME 40

static void sim_alloc(void) {
    static int aw, ah, apolar = -1;
    GLint wrap = sim_polar ? GL_REPEAT : GL_CLAMP_TO_EDGE;
    if (aw != sim_w || ah != sim_h || apolar != (sim_polar != 0)) {
        if (aw) { glDeleteTextures(4, &sim_tex[0][0]); glDeleteTextures(1, &sim_perm); }
        GLuint t[5];
        glGenTextures(5, t);
        for (int i = 0; i < 5; i++) {
            glBindTexture(GL_TEXTURE_2D, t[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sim_w, sim_h, 0, GL_RGBA, GL_FLOAT, NULL);
        }
        sim_tex[0][0] = t[0]; sim_tex[0][1] = t[1]; sim_tex[1][0] = t[2]; sim_tex[1][1] = t[3]; sim_perm = t[4];
        aw = sim_w; ah = sim_h; apolar = sim_polar != 0;
        static const GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
        if (!init_fbo) { glGenFramebuffers(1, &init_fbo); glGenFramebuffers(2, sim_fbo); }
        glBindFramebuffer(GL_FRAMEBUFFER, init_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sim_tex[0][0], 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, sim_tex[0][1], 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, sim_perm, 0);
        glDrawBuffers(3, bufs);
        fbo_check();
        for (int i = 0; i < 2; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, sim_fbo[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sim_tex[i][0], 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, sim_tex[i][1], 0);
            glDrawBuffers(2, bufs);
            fbo_check();
        }
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) { glBindTexture(GL_TEXTURE_2D, sim_tex[i][j]); tex_params(wrap, GL_NEAREST); }
    glBindTexture(GL_TEXTURE_2D, sim_perm);
    tex_params(wrap, GL_NEAREST);
}

static void sim_init(void) {
    sim_alloc();
    glViewport(0, 0, sim_w, sim_h);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, init_fbo);
    glUseProgram(prog_init);
    glUniform1i(U(prog_init, "uFold"), fold);
    glUniform2f(U(prog_init, "uSim"), (float)sim_w, (float)sim_h);
    glUniform4fv(U(prog_init, "uI"), 1, &uI[0][0]);
    glUniform1fv(U(prog_init, "uWD"), 16, uWD);
    glUniform1fv(U(prog_init, "uBand"), 16, uBand);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* n more steps; the result always ends in slot 0 */
static void sim_steps_run(int from, int n) {
    glViewport(0, 0, sim_w, sim_h);
    glBindVertexArray(vao);
    glUseProgram(prog_step);
    glUniform1f(U(prog_step, "uD"), sim_D);
    glUniform1f(U(prog_step, "uM"), sim_m);
    glUniform1f(U(prog_step, "uRet"), sim_ret);
    glUniform1f(U(prog_step, "uDm"), sim_dm);
    glUniform1f(U(prog_step, "uKx"), sim_kx);
    glUniform1i(U(prog_step, "uPolar"), sim_polar);
    glUniform1i(U(prog_step, "uA"), 0);
    glUniform1i(U(prog_step, "uB"), 1);
    glUniform1i(U(prog_step, "uP"), 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, sim_perm);
    for (int s = 0; s < n; s++) {
        int cur = s & 1;
        glUniform1f(U(prog_step, "uFrac"), (float)(from + s + 1) / sim_steps);
        glBindFramebuffer(GL_FRAMEBUFFER, sim_fbo[cur ^ 1]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sim_tex[cur][0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, sim_tex[cur][1]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glActiveTexture(GL_TEXTURE0);
    if (n & 1) {
        /* one more step's worth of bookkeeping: swap the slots */
        GLuint t0 = sim_tex[0][0], t1 = sim_tex[0][1];
        sim_tex[0][0] = sim_tex[1][0]; sim_tex[0][1] = sim_tex[1][1];
        sim_tex[1][0] = t0; sim_tex[1][1] = t1;
        GLuint f = sim_fbo[0]; sim_fbo[0] = sim_fbo[1]; sim_fbo[1] = f;
    }
}

/* ---- the view: world -> this window's pixels ---- */
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

static void cache_resize(void) {
    if (cache_w == W && cache_h == H) return;
    if (cache_w) glDeleteTextures(1, &cache_tex);
    glGenTextures(1, &cache_tex);
    glBindTexture(GL_TEXTURE_2D, cache_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, GL_NEAREST);
    if (!cache_fbo) glGenFramebuffers(1, &cache_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, cache_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cache_tex, 0);
    fbo_check();
    cache_w = W; cache_h = H;
    built_act = -1000;                  /* a new size: make it again */
    if (making > -1000 && make_stage == 2) make_strip = 0;
}

static void comp_strip(int k) {
    float tp = 3.2f / 1080;
    for (int j = 0; j < 2; j++) { glBindTexture(GL_TEXTURE_2D, sim_tex[0][j]); tex_params(sim_polar ? GL_REPEAT : GL_CLAMP_TO_EDGE, GL_LINEAR); }
    glBindFramebuffer(GL_FRAMEBUFFER, cache_fbo);
    glViewport(0, 0, W, H);
    int y0 = H * k / STRIPS, y1 = H * (k + 1) / STRIPS;
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, y0, W, y1 - y0);
    glBindVertexArray(vao);
    glUseProgram(prog_comp);
    glUniform1i(U(prog_comp, "uFold"), fold);
    glUniform1i(U(prog_comp, "uND"), palettes[pal].n);
    glUniform4fv(U(prog_comp, "uF"), 8, &uF[0][0]);
    glUniform4fv(U(prog_comp, "uH"), NH, &uH[0][0]);
    glUniform4fv(U(prog_comp, "uOrd"), 1, uOrd);
    glUniform3fv(U(prog_comp, "uKS"), MAXDYE, &ks[0][0]);
    glUniform3fv(U(prog_comp, "uKSc"), 1, ks_cloth);
    glUniform1f(U(prog_comp, "uStr"), palettes[pal].strength);
    glUniform1f(U(prog_comp, "uTP"), tp);
    glUniform1f(U(prog_comp, "uTide"), tide);
    glUniform1f(U(prog_comp, "uComb"), fib_comb);
    glUniform1f(U(prog_comp, "uMid"), fib_mid);
    glUniform4f(U(prog_comp, "uView"), (float)v_ox, (float)v_oy, (float)v_sc, (float)H);
    glUniform3f(U(prog_comp, "uFC"), (float)(fr[0] + fr[2] / 2), (float)(fr[1] + fr[3] / 2), (float)L);
    glUniform1i(U(prog_comp, "uSA"), 0);
    glUniform1i(U(prog_comp, "uSB"), 1);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sim_tex[0][0]);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sim_tex[0][1]);
    glActiveTexture(GL_TEXTURE0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
    for (int j = 0; j < 2; j++) { glBindTexture(GL_TEXTURE_2D, sim_tex[0][j]); tex_params(sim_polar ? GL_REPEAT : GL_CLAMP_TO_EDGE, GL_NEAREST); }
}

/* the caption's colours: dark on a light piece, light on a dark one,
 * judged from the cached piece under where it goes */
static void caption_colours(void) {
    int w = (int)fmin(W, fmax(64, H * 0.4)), h = (int)fmin(H, fmax(32, H * 0.12));
    float *px = malloc((size_t)w * h * 4 * sizeof(float));
    glBindFramebuffer(GL_FRAMEBUFFER, cache_fbo);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, px);
    double lum = 0;
    for (int i = 0; i < w * h; i++) {
        float *c = px + 4 * i;
        if (c[3] > 1.5f) { c[0] = cloth_lin[0]; c[1] = cloth_lin[1]; c[2] = cloth_lin[2]; }
        lum += 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    free(px);
    lum /= w * h;
    int light = lum < 0.22;
    for (int c = 0; c < 3; c++) {
        cap_col[c] = light ? 0.97f - 0.02f * c : 0.17f - 0.01f * c;
        cap_halo[c] = light ? 0.05f : palettes[pal].cloth[c];
    }
}

/* do some of the work of making act a's piece; all of it if `all` */
static void make_piece(long a, int all) {
    if (making != a) {
        setup(a);
        making = a; make_stage = 0; make_step = 0; make_strip = 0;
    }
    double budget = all ? 1e9 : 1;
    while (budget > 0 && built_act != a) {
        if (make_stage == 0) { sim_init(); make_stage = 1; budget -= 0.2; }
        else if (make_stage == 1) {
            int n = sim_steps - make_step;
            /* about the same GPU time per frame whatever the grid */
            int cap = (int)fmin(800, STEPS_PER_FRAME * (640.0 * 512.0) / ((double)sim_w * sim_h));
            if (!all && n > cap) n = cap;
            sim_steps_run(make_step, n);
            make_step += n;
            if (make_step >= sim_steps) make_stage = 2;
            budget -= 1;
        } else {
            comp_strip(make_strip++);
            budget -= 1;
            if (make_strip >= STRIPS) { caption_colours(); built_act = a; }
        }
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
static float cap_last[3];

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
    if (!strcmp(l0, cap_line[0]) && !strcmp(l1, cap_line[1]) && !memcmp(cap_last, cap_col, sizeof cap_last)) return tvn / 8;
    strcpy(cap_line[0], l0); strcpy(cap_line[1], l1);
    memcpy(cap_last, cap_col, sizeof cap_last);
    tvn = 0;
    float mx = (float)fmax(14, H * 0.035), by = (float)(H - fmax(14, H * 0.035));
    float y1 = by, y0 = by - cap_px[1] * 1.55f;
    float halo[4] = {cap_halo[0], cap_halo[1], cap_halo[2], 0.14f}, col[4] = {cap_col[0], cap_col[1], cap_col[2], 0.88f};
    float r = fmaxf(1.0f, cap_px[0] / 12.0f);
    for (int k = 0; k < 12; k++) {
        float an = k * 0.5236f, dx = r * cosf(an), dy = r * sinf(an);
        for (int q = 1; q <= 2; q++) {
            text_at(0, mx + dx * q, y0 + dy * q, l0, halo);
            text_at(1, mx + dx * q, y1 + dy * q, l1, halo);
        }
    }
    text_at(0, mx, y0, l0, col);
    col[3] = 0.75f;
    text_at(1, mx, y1, l1, col);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_DYNAMIC_DRAW);
    return tvn / 8;
}

static void gl_init(void) {
    prog_init = mk_prog(FS_INIT_MAIN);
    prog_step = mk_prog(FS_STEP_MAIN);
    prog_comp = mk_prog(FS_COMP_MAIN);
    prog_show = mk_prog(FS_SHOW_MAIN);
    prog_text = link_prog(compile(GL_VERTEX_SHADER, VS_TEXT, NULL), compile(GL_FRAGMENT_SHADER, FS_TEXT, ""));
    glGenVertexArrays(1, &vao);
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
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

/* ---- a frame ---- */
static void render(double t, double wall, GLuint target) {
    if (t < 0) t = 0;
    view_update();
    cache_resize();
    long a = (long)floor(t / ACT);
    double ts = t - a * ACT;
    /* the piece is made while the cloth is white; if it isn't done by the
     * time the dye should go on (a slow GPU, or a fresh start mid-act),
     * finish it now */
    if (built_act != a) make_piece(a, ts >= T_WHITE);
    int ready = built_act == a;
    double bt = (ts - T_WHITE) / T_BLOOM;
    /* the wicking slows as it goes */
    float uT = ready ? (float)(1.03 * (1 - pow(1 - clamp01(bt), 1.6))) : -1.0f;
    float wet = (float)(1 - ease((ts - T_WHITE - T_BLOOM * 0.85) / T_DRY));
    float fade = (float)ease((ts - (ACT - T_FADE)) / T_FADE);
    if (!ready || bt <= 0) uT = -1;

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glBindVertexArray(vao);
    glUseProgram(prog_show);
    glUniform1i(U(prog_show, "uC"), 0);
    glUniform3fv(U(prog_show, "uCloth"), 1, cloth_lin);
    glUniform1f(U(prog_show, "uT"), uT);
    glUniform1f(U(prog_show, "uWet"), wet);
    glUniform1f(U(prog_show, "uFade"), fade);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cache_tex);
    glDrawArrays(GL_TRIANGLES, 0, 3);

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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "tiedye");
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
        "usage: tiedye [--window | --root | --layer] [--seed N] [--fps N]\n"
        "              [--fold spiral|bullseye]\n"
        "              [--palette pastel|rainbow|sunset|ocean|indigo]\n"
        "              [--no-caption] [--font FAMILY] [--schedule]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --shot 'f%%04d.ppm' --bench N   save each of N frames (from --at, at --fps)\n"
        "  --fold, --palette   hold one; default: picked per piece from the seed\n"
        "  --fps       frame rate (default 30)\n"
        "  --schedule  print which piece is made when (seconds after the epoch) and exit\n");
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

static void print_schedule(void) {
    for (long a = 0; a < 12; a++) {
        int f = act_fold(a);
        printf("%7.1f s  piece %2ld: %-8s %-7s  dye on from %.0f s, dry by %.0f s, fades from %.0f s\n", a * ACT, a,
               fold_name[f], palettes[act_pal(a)].name, a * ACT + T_WHITE, a * ACT + T_WHITE + T_BLOOM * 0.85 + T_DRY,
               (a + 1) * ACT - T_FADE);
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
        else if (!strcmp(a, "-fold") && i + 1 < argc) opt_fold = argv[++i];
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
    if (opt_fold) {
        for (forced_fold = 0; forced_fold < NFOLD; forced_fold++) if (!strcmp(fold_name[forced_fold], opt_fold)) break;
        if (forced_fold == NFOLD) { fprintf(stderr, "tiedye: no fold \"%s\" (spiral, bullseye)\n", opt_fold); exit(1); }
    }
    if (opt_palette) {
        for (forced_pal = 0; forced_pal < NPAL; forced_pal++) if (!strcmp(palettes[forced_pal].name, opt_palette)) break;
        if (forced_pal == NPAL) { fprintf(stderr, "tiedye: no palette \"%s\" (pastel, rainbow, sunset, ocean, indigo)\n", opt_palette); exit(1); }
    }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_schedule) { print_schedule(); return 0; }
    int offscreen = opt_shot || opt_bench > 0;
    if (opt_caption) fonts_init();

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
        XStoreName(dpy, win, "Tiedye");
        XClassHint chint = {"tiedye", "Tiedye"};
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
        /* a shot shows the piece as it is at --at, made in full first */
        long a = (long)floor(fmax(opt_at, 0) / ACT);
        view_update();
        cache_resize();
        make_piece(a, 1);
        glFinish();
        double make_ms = (now() - t0) * 1000;
        render(opt_at, opt_epoch + opt_at, fbo);
        glFinish();
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "tiedye: %s, %s, %dx%d, piece %ld at %.1f s: made in %.0f ms\n", fold_name[fold],
                    palettes[pal].name, W, H, a, opt_at - a * ACT, make_ms);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / opt_fps;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6);
            if (ms > worst) worst = ms;
            /* --shot 'frames/%04d.ppm' --bench N: keep every frame */
            if (opt_shot && strchr(opt_shot, '%')) {
                char path[512];
                snprintf(path, sizeof path, opt_shot, i);
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                save_ppm(path);
            }
        }
        printf("tiedye: %s, %s, %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               fold_name[fold], palettes[pal].name, W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("tiedye:   GPU %.3f ms (best %.3f); making the piece took %.0f ms\n", gsum / opt_bench, gmin, make_ms);
        printf("tiedye: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        if (opt_shot && !strchr(opt_shot, '%')) { glBindFramebuffer(GL_FRAMEBUFFER, fbo); save_ppm(opt_shot); }
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
