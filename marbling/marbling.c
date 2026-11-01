/*
 * marbling — paper marbling (ebru, suminagashi) as a slow live process.
 *
 * Inks are dropped onto a water bath one at a time. Each drop spreads into
 * a disc and pushes everything already floating outward into rings. Then
 * tools are drawn through the bath: a stylus pulled across in straight
 * lines, a rake pulled back and forth (get-gel), a fine comb (nonpareil),
 * a wavy pass (bouquet), a stylus stirred in circles (whirls). When the
 * pattern is done it holds a while, then a sheet of paper is laid on,
 * soaks up the print and is lifted off, and the clean bath starts over
 * with another recipe and another set of inks.
 *
 * Nothing is simulated. Every operation is an exact, area-preserving
 * deformation of the bath (after Jaffer, "Mathematical Marbling"):
 *
 *   drop of radius r at c:  p' = c + (p - c) sqrt(1 + r^2 / |p - c|^2)
 *   tine along L through b: p' = p + L z(d),  z = a l / (d + l),
 *                           d = distance from the tine's line
 *   comb: tines every S, the two nearest pulling (alternate tines
 *         the opposite way for a rake);  wave: p' = p + L A sin(w p.N + f);
 *   whirl: p turned about c by A exp(-|p - c|^2 / s^2)
 *
 * and a pixel's colour is found by running the inverses back from now
 * until a drop captures it (or it reaches the bare bath), so ink edges
 * stay razor sharp no matter how far they've been dragged. The finished
 * operations are baked into a cached image (4 samples a pixel, in strips
 * over a few frames while the next operation plays); the one in motion is
 * applied live on top, its drop edge analytic, its tool tip a brass pin.
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen,
 * --bench N to time N frames offscreen.
 *
 * A portrait screen gets the bath turned on its side: the recipes are laid
 * out for a landscape one, and the sheet is shown rotated a quarter turn.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * with --layout (every output's logical rect) and --output = this one's
 * index, as for flightsim. Simulating one bath across the whole wall would
 * cost too much, so each output instead marbles its own sheet in the shared
 * palette: the "front" output (the one under the middle of the layout)
 * plays the recipes in the seed's order from the epoch, and the others play
 * the same order a recipe or more ahead and part way through it, staggered
 * so their papers come off at different times.
 *
 * The inks come from one of several palettes after traditional marbled
 * papers (brass, ebru, florentine, sea, ink), picked from the seed for the
 * whole session, or fixed with --palette NAME.
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
static int opt_root, opt_layer, opt_bench, opt_recipe = -1;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot / --bench: from this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the clock starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
static int rec_shift;                  /* this output plays recipes from this one on */
static double span_toff;               /* and this far into them */

static void die(const char *what) { fprintf(stderr, "marbling: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}
static float frange(float a, float b) { return a + (b - a) * frand(); }
static int irange(int a, int b) { int v = a + (int)(frand() * (b - a + 1)); return v > b ? b : v; }

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x ? x : 1;
}

/* ---- inks: a palette per session, picked from the seed ----
 * Each palette has eight inks, two baths (a dark one and a pale size), a
 * paper and harmonious ink sets for each bath. The first ink of a set is
 * the one that ends up on top and, with the second, makes the stones of
 * "stone", so those two always stand off the bath and from each other. */
#define NINK 8
typedef struct {
    const char *name;
    const char *ink[NINK];
    const char *bath_dark, *bath_cream;
    float paper[3];
    int dark[5][5], cream[5][5];
} Palette;

static const Palette PALETTES[] = {
    /* Aged Brass: 0 cream 1 gold 2 gold2 3 amber 4 rust 5 terra 6 sage 7 brown */
    {"brass", {"#EFE3CE", "#FFC332", "#D6A12A", "#E9994A", "#C66B1B", "#A53C2C", "#8FA672", "#2A211B"},
     "#1B1511", "#EADCC3", {0.957f, 0.925f, 0.866f},
     {{0, 1, 5, 6, -1}, {3, 0, 5, 6, -1}, {1, 5, 0, 4, -1}, {0, 2, 5, -1, -1}, {1, 0, 6, -1, -1}},
     {{5, 7, 2, 6, -1}, {7, 4, 6, -1, -1}, {7, 5, 2, -1, -1}, {6, 7, 4, 1, -1}, {4, 7, 6, 5, -1}}},
    /* Turkish ebru: 0 white 1 indigo 2 madder 3 ochre 4 deep red 5 sky 6 rose 7 ink black */
    {"ebru", {"#F3ECDD", "#2C3D78", "#B3322B", "#D9A43C", "#6E1C22", "#7E9CC6", "#C26C60", "#1C1B26"},
     "#121423", "#EEE2C8", {0.957f, 0.93f, 0.875f},
     {{0, 2, 3, 5, -1}, {3, 0, 2, 5, -1}, {0, 3, 5, 2, -1}, {2, 0, 3, 6, -1}, {5, 3, 0, 2, -1}},
     {{1, 2, 3, -1, -1}, {2, 1, 3, 7, -1}, {1, 3, 4, 5, -1}, {7, 2, 3, 1, -1}, {2, 1, 5, 3, -1}}},
    /* Florentine end-papers: 0 white 1 navy 2 Prussian 3 red 4 sky 5 pale blue 6 deep red 7 black */
    {"florentine", {"#F2EFE7", "#1B2A4E", "#1E5A8C", "#C2362E", "#86A9CC", "#C9D6E2", "#6C1A20", "#121620"},
     "#0E1424", "#EDE8DC", {0.96f, 0.945f, 0.91f},
     {{0, 3, 4, 2, -1}, {4, 0, 3, -1, -1}, {3, 0, 2, 4, -1}, {0, 4, 6, -1, -1}, {5, 3, 2, 0, -1}},
     {{1, 3, 2, -1, -1}, {2, 3, 1, 4, -1}, {3, 1, 4, -1, -1}, {1, 2, 6, 4, -1}, {3, 7, 2, 4, -1}}},
    /* sea: 0 white 1 teal 2 navy 3 sand 4 coral 5 aqua 6 deep teal 7 pale sand */
    {"sea", {"#F1EEE5", "#2C8580", "#1C3052", "#D8C397", "#E07A5F", "#86C2B8", "#164A50", "#EBDDBE"},
     "#0B1A1F", "#E9E3D3", {0.955f, 0.945f, 0.915f},
     {{0, 1, 3, 5, -1}, {5, 0, 3, 4, -1}, {3, 1, 0, 5, -1}, {0, 5, 1, -1, -1}, {7, 1, 5, 4, -1}},
     {{1, 2, 3, -1, -1}, {2, 1, 5, 4, -1}, {6, 3, 2, 5, -1}, {1, 2, 4, 3, -1}, {2, 5, 6, 3, -1}}},
    /* ink (suminagashi): 0 off-white 1 pale grey 2 grey 3 charcoal 4 black 5 deep red 6 mid grey 7 dark grey */
    {"ink", {"#EEECE6", "#B9B7B1", "#7C7A76", "#3C3B39", "#151515", "#8E1D1D", "#9D9A93", "#55524D"},
     "#121212", "#ECE9E2", {0.95f, 0.945f, 0.93f},
     /* the red only ever a vein, in two sets of five, so it stays an accent */
     {{0, 2, 1, 5, -1}, {1, 0, 3, -1, -1}, {0, 6, 3, -1, -1}, {6, 0, 3, -1, -1}, {0, 1, 7, 5, -1}},
     {{4, 2, 1, 5, -1}, {3, 4, 1, -1, -1}, {4, 7, 1, 5, -1}, {2, 4, 3, -1, -1}, {3, 6, 4, -1, -1}}},
};
#define NPAL ((int)(sizeof PALETTES / sizeof *PALETTES))
static const Palette *P;               /* this session's */
static int opt_palette = -1;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

/* ---- recipes: a timed list of operations ---- */
#define MAXOPS 160
enum { OP_DROP = 1, OP_COMB, OP_WAVE, OP_WHIRL };
enum { K_STONE, K_RIPPLE, K_NONPAREIL, K_CHEVRON, K_BOUQUET, K_WHIRL, NKIND };
static const char *KIND_NAME[NKIND] = {"stone", "ripple", "nonpareil", "chevron", "bouquet", "whirl"};

/* per op, three vec4s (see the shaders):
 *   drop:  (cx, cy, r, rim)      (r, g, b, -)          (-, -, -, type)
 *   comb:  (bx, by, Lx, Ly)      (a, l, spacing, alt)  (ramp, tip0, tip1, type)
 *   wave:  (Lx, Ly, A, w)        (phase, -, -, -)      (-, -, -, type)
 *   whirl: (cx, cy, A, sigma)    (-, -, -, -)          (-, -, -, type)
 * positions in screen heights, origin bottom left */
typedef struct { float t0, dur; float v[12]; } Op;

#define HOLD 10.0
#define COVER 2.2
#define REST 0.8
#define LIFT 2.2

typedef struct {
    int kind, n, dark;
    float aspect;
    float bath[3];
    double ops_end, paper_at, dur;     /* local times */
    float pdir;                        /* angle the paper is laid on along */
    Op op[MAXOPS];
} Recipe;

static Recipe *R;                      /* being generated */
static double gen_t;                   /* its clock */

static Op *add_op(int type, float dur, float gap) {
    if (R->n >= MAXOPS) return NULL;
    Op *o = &R->op[R->n++];
    memset(o, 0, sizeof *o);
    o->t0 = (float)gen_t; o->dur = dur;
    o->v[11] = (float)type;
    gen_t += dur + gap;
    return o;
}

static void add_drop(float x, float y, float r, int ink, float dur) {
    Op *o = add_op(OP_DROP, dur, dur * 0.2f + frange(0.0f, 0.12f));
    if (!o) return;
    float c[3];
    hex(P->ink[ink], c);
    float tone = frange(0.95f, 1.04f), lum = 0.3f * c[0] + 0.55f * c[1] + 0.15f * c[2];
    for (int i = 0; i < 3; i++) c[i] = lum + (c[i] - lum) * 0.93f;   /* real inks are a touch muted */
    o->v[0] = x; o->v[1] = y; o->v[2] = r; o->v[3] = frange(0.03f, 0.07f);
    for (int i = 0; i < 3; i++) o->v[4 + i] = fminf(1.0f, c[i] * tone);
}

/* a straight pass (spacing 0: one stylus) of tines along unit L through b,
 * the tip travelling right across the screen */
static void add_comb(float bx, float by, float ang, float a, float l, float sp, int alt, float dur, float gap) {
    Op *o = add_op(OP_COMB, dur, gap);
    if (!o) return;
    float Lx = cosf(ang), Ly = sinf(ang);
    float zmax = a * (1.0f + (sp > 0 ? l / (l + sp * 0.5f) : 0.0f));
    float ramp = 2.0f * zmax;          /* keeps the moving tip's map invertible */
    /* how far along L the screen (plus the cache margin) reaches from b */
    float S = 0;
    float cx[4] = {-0.1f, R->aspect + 0.1f, -0.1f, R->aspect + 0.1f}, cy[4] = {-0.1f, -0.1f, 1.1f, 1.1f};
    for (int i = 0; i < 4; i++) S = fmaxf(S, fabsf((cx[i] - bx) * Lx + (cy[i] - by) * Ly));
    float v[12] = {bx, by, Lx, Ly, a, l, sp, (float)alt, ramp, -S - 0.02f, S + ramp + a + 0.02f, OP_COMB};
    memcpy(o->v, v, sizeof v);
}

static void add_wave(float ang, float A, float wl, float dur) {
    Op *o = add_op(OP_WAVE, dur, 0.6f);
    if (!o) return;
    float v[12] = {cosf(ang), sinf(ang), A, 6.2831853f / wl, frange(0, 6.28f), 0, 0, 0, 0, 0, 0, OP_WAVE};
    memcpy(o->v, v, sizeof v);
}

static void add_whirl(float x, float y, float A, float sig, float dur) {
    Op *o = add_op(OP_WHIRL, dur, 0.4f);
    if (!o) return;
    float v[12] = {x, y, A, sig, 0, 0, 0, 0, 0, 0, 0, OP_WHIRL};
    memcpy(o->v, v, sizeof v);
}

/* the inks of a recipe, the first being the "top" one */
static int inks[5], ninks;

static int pick_ink(int avoid) {
    int k;
    do k = inks[irange(0, ninks - 1)]; while (ninks > 1 && k == avoid);
    return k;
}

/* drops scattered evenly: each the best of a few candidates, away from
 * the others of this scatter, and a different ink from its nearest */
static void scatter(int n, float rmin, float rmax, int from, int to, float dur) {
    float hx[64], hy[64];
    int hk[64], nh = 0;
    for (int i = 0; i < n; i++) {
        float bx = 0, by = 0, best = -1;
        int near = -1;
        for (int c = 0; c < 8; c++) {
            float x = frange(0.02f, R->aspect - 0.02f), y = frange(0.02f, 0.98f), d = 1e9f;
            if (!R->n) { x = R->aspect * frange(0.3f, 0.7f); y = frange(0.3f, 0.7f); }   /* open in the middle */
            int k = -1;
            for (int j = 0; j < nh; j++) {
                float dd = (x - hx[j]) * (x - hx[j]) + (y - hy[j]) * (y - hy[j]);
                if (dd < d) { d = dd; k = j; }
            }
            if (d > best) { best = d; bx = x; by = y; near = k; }
        }
        int ink;
        do ink = inks[irange(from, to)]; while (to > from && near >= 0 && ink == hk[near]);
        if (nh < 64) { hx[nh] = bx; hy[nh] = by; hk[nh] = ink; nh++; }
        add_drop(bx, by, frange(rmin, rmax), ink, dur * frange(0.85f, 1.15f));
    }
}

/* concentric rings: drops of alternating inks, all at one spot */
static void bullseye(float x, float y, int rings, float r, float dur) {
    int a = pick_ink(-1), b = pick_ink(a), c = ninks > 2 && frand() < 0.5f ? pick_ink(b) : a;
    int seq[3] = {a, b, c};
    for (int i = 0; i < rings; i++) add_drop(x, y, r * frange(0.9f, 1.1f), seq[i % 3], dur);
}

static float A_;                       /* aspect, shorthand while generating */

/* the rake back and forth across the bath, rows `sp` apart */
static void getgel(float ang, float sp) {
    add_comb(A_ * 0.5f, 0.5f, ang, frange(0.2f, 0.26f), frange(0.045f, 0.06f), sp, 1, 3.6f, 0.8f);
}

static int kind_of(int idx) {
    if (opt_recipe >= 0) return opt_recipe;
    idx += rec_shift;
    int deck = idx / NKIND, pos = idx % NKIND, order[NKIND], prev_last = -1;
    for (int d = deck - 1; d <= deck; d++) {
        if (d < 0) continue;
        rng = mix32(opt_seed * 2654435761U ^ (uint32_t)d * 40503U ^ 0xdeca1U);
        for (int i = 0; i < NKIND; i++) order[i] = i;
        for (int i = NKIND - 1; i > 0; i--) { int j = irange(0, i); int t = order[i]; order[i] = order[j]; order[j] = t; }
        if (d < deck) prev_last = order[NKIND - 1];
    }
    if (order[0] == prev_last) { int t = order[0]; order[0] = order[1]; order[1] = t; }
    return order[pos];
}

static int dark_of(int idx) {
    idx += rec_shift;
    rng = mix32(opt_seed * 747796405U ^ (uint32_t)idx * 2891336453U ^ 0xba7eU);
    return frand() < 0.58f;
}

static void gen_recipe(int idx, float aspect, Recipe *out) {
    R = out;
    memset(R, 0, sizeof *R);
    R->kind = kind_of(idx);
    R->dark = dark_of(idx);
    R->aspect = A_ = aspect;
    hex(R->dark ? P->bath_dark : P->bath_cream, R->bath);
    rng = mix32(opt_seed ^ (uint32_t)(idx + rec_shift) * 0x9E3779B9U);
    const int *set = R->dark ? P->dark[irange(0, 4)] : P->cream[irange(0, 4)];
    for (ninks = 0; ninks < 5 && set[ninks] >= 0; ninks++) inks[ninks] = set[ninks];
    R->pdir = frange(0, 6.2831853f);
    gen_t = idx == 0 ? 0.6 : 0.8;      /* after the paper's gone */

    switch (R->kind) {
    case K_STONE:                      /* big drops that end as veins, then the stones */
        scatter(12, 0.22f, 0.3f, 2, ninks - 1, 0.9f);
        scatter(irange(38, 42), 0.15f, 0.19f, 0, 1, 0.5f);
        scatter(irange(6, 9), 0.035f, 0.055f, 2, ninks - 1, 0.45f);
        break;
    case K_RIPPLE: {                   /* rings, then a few stylus lines through them */
        scatter(9, 0.27f, 0.34f, 0, ninks - 1, 0.9f);
        int nc = irange(3, 4);
        for (int i = 0; i < nc; i++)
            bullseye(A_ * (i + frange(0.25f, 0.75f)) / nc, frange(0.25f, 0.75f), irange(8, 10), frange(0.08f, 0.1f), 0.5f);
        float ang = (frand() < 0.5f ? 0.0f : 1.5707963f) + frange(-0.45f, 0.45f);
        int nl = irange(4, 6);
        float span = fabsf(A_ * sinf(ang)) + fabsf(cosf(ang)), a = frange(0.1f, 0.16f), l = frange(0.018f, 0.03f);
        float nx = -sinf(ang), ny = cosf(ang);
        for (int i = 0; i < nl; i++) {
            float off = span * ((i + 0.5f) / nl - 0.5f) + frange(-0.03f, 0.03f);
            add_comb(A_ * 0.5f + nx * off, 0.5f + ny * off, ang + (i & 1 ? 3.1415926f : 0.0f), a, l, 0, 0, 1.9f, 0.35f);
        }
        break;
    }
    case K_NONPAREIL: {                /* stone, get-gel, then a fine comb */
        scatter(28, 0.15f, 0.22f, 0, ninks - 1, 0.6f);
        float ang = frand() < 0.7f ? 0.0f : 1.5707963f;
        getgel(ang, frange(0.09f, 0.12f));
        getgel(ang + 1.5707963f, frange(0.09f, 0.12f));
        add_comb(A_ * 0.5f, 0.5f, ang + 1.5707963f, frange(0.07f, 0.09f), frange(0.008f, 0.012f), frange(0.024f, 0.032f), 0, 4.0f, 0.5f);
        break;
    }
    case K_CHEVRON: {                  /* stone, get-gel, then a wide rake across it */
        scatter(28, 0.15f, 0.22f, 0, ninks - 1, 0.6f);
        float ang = frand() < 0.7f ? 0.0f : 1.5707963f;
        getgel(ang, frange(0.08f, 0.1f));
        add_comb(A_ * 0.5f, 0.5f, ang + 1.5707963f, frange(0.12f, 0.16f), frange(0.04f, 0.06f), frange(0.13f, 0.18f), 1, 3.6f, 0.5f);
        break;
    }
    case K_BOUQUET: {                  /* nonpareil, then drawn back in waves */
        scatter(26, 0.15f, 0.22f, 0, ninks - 1, 0.6f);
        float ang = frand() < 0.7f ? 0.0f : 1.5707963f;
        getgel(ang, frange(0.09f, 0.11f));
        add_comb(A_ * 0.5f, 0.5f, ang + 1.5707963f, frange(0.07f, 0.09f), frange(0.008f, 0.012f), frange(0.025f, 0.03f), 0, 4.0f, 0.5f);
        add_wave(ang + 1.5707963f, frange(0.035f, 0.05f), frange(0.16f, 0.22f), 3.0f);
        break;
    }
    case K_WHIRL: {                    /* rings, stirred round */
        scatter(6, 0.26f, 0.34f, 0, ninks - 1, 0.9f);
        int nc = irange(4, 5);
        float cx[5], cy[5];
        for (int i = 0; i < nc; i++) {
            cx[i] = A_ * (i + frange(0.3f, 0.7f)) / nc; cy[i] = (i & 1 ? 0.62f : 0.38f) + frange(-0.1f, 0.1f);
            bullseye(cx[i], cy[i], irange(7, 9), frange(0.07f, 0.09f), 0.42f);
        }
        float dir = frand() < 0.5f ? 1.0f : -1.0f;
        for (int i = 0; i < nc; i++, dir = -dir)
        {
            /* stirred a little off centre, or the rings would just turn */
            float oa = frange(0, 6.2831853f), od = frange(0.07f, 0.12f);
            add_whirl(cx[i] + od * cosf(oa), cy[i] + od * sinf(oa), dir * frange(3.0f, 4.5f), frange(0.13f, 0.19f), 3.4f);
        }
        break;
    }
    }
    R->ops_end = 0;
    for (int i = 0; i < R->n; i++) R->ops_end = fmax(R->ops_end, R->op[i].t0 + R->op[i].dur);
    R->paper_at = R->ops_end + HOLD;
    R->dur = R->paper_at + COVER + REST + LIFT;
}

/* where recipe idx starts on the clock; recipes play back to back */
static double *rstart;
static int nrstart, caprstart;
static double recipe_start(int idx) {
    static Recipe tmp;
    if (!nrstart) { rstart = malloc(64 * sizeof *rstart); caprstart = 64; rstart[0] = 0; nrstart = 1; }
    while (nrstart <= idx) {
        gen_recipe(nrstart - 1, 1.6f, &tmp);   /* the timing doesn't depend on the shape */
        if (nrstart == caprstart) rstart = realloc(rstart, (caprstart *= 2) * sizeof *rstart);
        rstart[nrstart] = rstart[nrstart - 1] + tmp.dur;
        nrstart++;
    }
    return rstart[idx];
}

static int recipe_at(double t, double *local) {
    if (t < 0) t = 0;
    int i = 0;
    while (recipe_start(i + 1) <= t) i++;
    *local = t - recipe_start(i);
    return i;
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the operations' inverses, shared by both passes */
static const char *FS_COMMON = "#version 330 core\n" GLSL(
    uniform sampler2D uOps;     /* MAXOPS rows of 3 RGBA32F texels */
    uniform float uH;           /* screen height in px: positions are in heights */
    vec4 opv(int i, int j) { return texelFetch(uOps, ivec2(j, i), 0); }

    /* the two tines either side of n: their pulls (signed by direction) */
    void tines(vec4 b, float n, out float z0, out float z1, out float g0, out float g1) {
        float a = b.x, l = b.y, sp = b.z;
        g0 = 1.0; g1 = 1.0;
        if (sp <= 0.0) { z0 = a * l / (abs(n) + l); z1 = 0.0; return; }
        float k = floor(n / sp), d0 = n - k * sp, d1 = sp - d0;
        if (b.w > 0.5) { g0 = mod(k, 2.0) < 0.5 ? 1.0 : -1.0; g1 = -g0; }
        z0 = a * l / (d0 + l); z1 = a * l / (d1 + l);
    }

    /* a comb pass undone; tip < -1e8 for the finished pass, else the tip's
     * way along L (along -L for the tines going the other way), with the
     * pull easing in over c.x behind it */
    vec2 combInv(vec2 p, vec4 a, vec4 b, vec4 c, float tip) {
        vec2 L = a.zw, N = vec2(-L.y, L.x), q = p - a.xy;
        float s = dot(q, L), n = dot(q, N), z0, z1, s0, s1;
        tines(b, n, z0, z1, s0, s1);
        if (tip < -1e8) s -= s0 * z0 + s1 * z1;
        else {
            /* solve s + sum(s_i z_i g(tip - s_i s)) = s' by Newton */
            float sp = s, w = c.x;
            for (int it = 0; it < 5; it++) {
                float x0 = clamp((tip - s0 * s) / w, 0.0, 1.0), x1 = clamp((tip - s1 * s) / w, 0.0, 1.0);
                float h = s + s0 * z0 * x0 * x0 * (3.0 - 2.0 * x0) + s1 * z1 * x1 * x1 * (3.0 - 2.0 * x1) - sp;
                float dh = 1.0 - (z0 * 6.0 * x0 * (1.0 - x0) + z1 * 6.0 * x1 * (1.0 - x1)) / w;
                s -= h / max(dh, 0.2);
            }
        }
        return a.xy + L * s + N * n;
    }
    vec2 waveInv(vec2 p, vec4 a, vec4 b, float k) {
        vec2 N = vec2(-a.y, a.x);
        return p - a.xy * (k * a.z * sin(a.w * dot(p, N) + b.x));
    }
    vec2 whirlInv(vec2 p, vec4 a, float k) {
        vec2 d = p - a.xy;
        float th = -k * a.z * exp(-dot(d, d) / (a.w * a.w)), cs = cos(th), sn = sin(th);
        return a.xy + vec2(cs * d.x - sn * d.y, sn * d.x + cs * d.y);
    }
    /* ink, denser towards the rim the drop first spread to */
    vec3 ink(vec4 a, vec4 b, float q) { return b.rgb * (1.0 - a.w * smoothstep(0.72, 1.0, q)); }
);

/* bake: the bath after the first uN operations, 4 samples a pixel */
static const char *FS_BUILD = GLSL(
    uniform int uN;
    uniform float uM;           /* the cache's margin round the screen, in heights */
    uniform vec3 uBath;
    out vec4 o;
    void main() {
        vec2 base = gl_FragCoord.xy / uH - uM;
        vec2 ps[4];
        ps[0] = base + vec2(-0.125, -0.375) / uH; ps[1] = base + vec2(0.375, -0.125) / uH;
        ps[2] = base + vec2(0.125, 0.375) / uH;   ps[3] = base + vec2(-0.375, 0.125) / uH;
        vec3 col[4];
        bool done[4];
        for (int s = 0; s < 4; s++) { col[s] = uBath; done[s] = false; }
        int left = 4;
        for (int i = uN - 1; i >= 0 && left > 0; i--) {
            vec4 a = opv(i, 0), b = opv(i, 1), c = opv(i, 2);
            int ty = int(c.w);
            for (int s = 0; s < 4; s++) {
                if (done[s]) continue;
                vec2 p = ps[s];
                if (ty == 1) {
                    vec2 d = p - a.xy;
                    float d2 = dot(d, d), r2 = a.z * a.z;
                    if (d2 < r2) { col[s] = ink(a, b, sqrt(d2 / r2)); done[s] = true; left--; continue; }
                    p = a.xy + d * sqrt(1.0 - r2 / d2);
                } else if (ty == 2) p = combInv(p, a, b, c, -2e8);
                else if (ty == 3) p = waveInv(p, a, b, 1.0);
                else p = whirlInv(p, a, 1.0);
                ps[s] = p;
            }
        }
        o = vec4((col[0] + col[1] + col[2] + col[3]) * 0.25, 1.0);
    });

/* show: the cache through the operation in motion, then the water, the
 * paper and the tools */
static const char *FS_SHOW = GLSL(
    uniform sampler2D uCache;
    uniform vec2 uRes, uCacheRes;
    uniform float uM, uLight;   /* the window light's sway, a phase */
    uniform vec4 uDrift;        /* the swell's two layers, drifting (wrapped) */
    uniform int uAnim;          /* the op in motion, or -1 */
    uniform float uProg;        /* its progress, 0..1 */
    uniform vec3 uNext;         /* the next recipe's bath */
    uniform vec4 uPaper;        /* phase (0 none, 1 laying, 2 down, 3 lifting), its progress, soak, - */
    uniform vec3 uPaperCol;
    uniform vec2 uPdir;
    out vec4 o;

    uniform sampler2D uNoise;   /* 256x256 tiling: smooth, smooth, fibre, white */
    float swell(vec2 p) {
        return 0.6 * texture(uNoise, p * 0.31 + uDrift.xy).r + 0.4 * texture(uNoise, p * 0.73 + uDrift.zw).g;
    }
    float fibre(vec2 px, vec2 sc) {
        return 0.6 * texture(uNoise, px * sc / 256.0).b + 0.4 * texture(uNoise, px * sc * 0.37 / 256.0 + 0.31).b - 0.5;
    }
    vec3 screen(vec3 c, vec3 l) { return 1.0 - (1.0 - c) * (1.0 - l); }
    vec3 cache(vec2 q) { return texture(uCache, (q + uM) * uH / uCacheRes).rgb; }

    /* a brass pin seen from above, with its shadow on the water */
    vec3 pin(vec3 col, vec2 p, vec2 at, float fade) {
        float R = 0.0062, px = 1.0 / uH;
        float sd = length(p - at - vec2(0.004, -0.005));
        col *= 1.0 - 0.5 * fade * smoothstep(R * 2.4, R * 0.5, sd);
        vec2 d = p - at;
        float a = fade * clamp((R - length(d)) / px + 0.5, 0.0, 1.0);
        float hl = clamp(1.0 - length(d / R - vec2(-0.35, 0.4)), 0.0, 1.0);
        vec3 brass = mix(vec3(0.36, 0.25, 0.09), vec3(1.0, 0.9, 0.62), hl * hl);
        return mix(col, brass, a);
    }

    void main() {
        float A = uRes.x / uRes.y;
        vec2 p = gl_FragCoord.xy / uH;
        vec2 q = p;
        vec4 a = vec4(0), b = vec4(0), c = vec4(0);
        int ty = 0;
        float cover = 0.0, ring = 0.0;
        vec3 dropc = vec3(0);
        if (uAnim >= 0) {
            a = opv(uAnim, 0); b = opv(uAnim, 1); c = opv(uAnim, 2);
            ty = int(c.w);
            if (ty == 1) {
                /* the drop spreads fast, then slows */
                float r = a.z * sqrt(1.0 - pow(1.0 - uProg, 3.0));
                vec2 d = p - a.xy;
                float dl = length(d);
                cover = clamp((r - dl) * uH + 0.5, 0.0, 1.0);
                dropc = ink(a, b, dl / max(r, 1e-6));
                q = a.xy + d * sqrt(max(1.0 - r * r / max(dl * dl, 1e-12), 0.0));
                /* a ripple runs out from where it landed */
                float rr = r + 0.012 + 0.16 * uProg;
                ring = exp(-pow((dl - rr) * 90.0, 2.0)) * pow(1.0 - uProg, 2.0);
            } else if (ty == 2) q = combInv(p, a, b, c, mix(c.y, c.z, uProg));
            else if (ty == 3) q = waveInv(p, a, b, uProg * uProg * (3.0 - 2.0 * uProg));
            else if (ty == 4) q = whirlInv(p, a, uProg * uProg * (3.0 - 2.0 * uProg));
        }
        vec3 col = mix(cache(q), dropc, cover);
        if (uPaper.x > 2.5) {          /* lifting: the fresh bath shows where it's gone */
            float e = dot(p - vec2(A, 1.0) * 0.5, uPdir), E = 0.5 * (A * abs(uPdir.x) + abs(uPdir.y)) + 0.12;
            float f = mix(-E, E, uPaper.y);
            if (e < f) col = uNext;
        }

        /* the water: fine grain, a slow swell, a soft window of light */
        float sw = swell(p);
        float tex = fibre(gl_FragCoord.xy, vec2(0.55, 0.8));
        col *= 1.0 + 0.04 * tex + 0.035 * (sw - 0.5);
        vec2 lp = vec2(A * 0.3 + 0.06 * sin(uLight), 0.86);
        float win = smoothstep(1.05, 0.0, length((p - lp) * vec2(0.55, 1.0)));
        float sheen = win * win * (0.55 + 0.9 * sw) * 0.075 + ring * 0.12;
        col = screen(col, vec3(1.0, 0.95, 0.84) * sheen);

        /* the tool in motion */
        if (ty == 2) {
            vec2 L = a.zw, N = vec2(-L.y, L.x), d = p - a.xy;
            float s = dot(d, L), n = dot(d, N), tip = mix(c.y, c.z, uProg) - 0.35 * c.x;
            float fade = smoothstep(0.0, 0.05, uProg) * smoothstep(1.0, 0.95, uProg);
            if (b.z <= 0.0) col = pin(col, p, a.xy + L * tip, fade);
            else {
                float k = floor(n / b.z);
                for (int j = 0; j < 2; j++) {
                    float kk = k + float(j), sg = b.w > 0.5 && mod(kk, 2.0) > 0.5 ? -1.0 : 1.0;
                    col = pin(col, p, a.xy + L * (sg * tip) + N * (kk * b.z), fade);
                }
            }
        } else if (ty == 4) {
            /* the stylus, carried round with the water at its radius */
            float e = uProg * uProg * (3.0 - 2.0 * uProg), rr = a.w * 0.45;
            float th = a.z * e * exp(-0.2025);
            float fade = smoothstep(0.0, 0.08, uProg) * smoothstep(1.0, 0.9, uProg);
            col = pin(col, p, a.xy + rr * vec2(cos(th + 0.5), sin(th + 0.5)), fade);
        }

        /* the paper */
        if (uPaper.x > 0.5) {
            float e = dot(p - vec2(A, 1.0) * 0.5, uPdir), E = 0.5 * (A * abs(uPdir.x) + abs(uPdir.y)) + 0.12;
            float f = mix(-E, E, uPaper.y);
            float under = uPaper.x < 2.5 ? f - e : e - f;      /* > 0 on the paper */
            float lifting = uPaper.x > 2.5 ? 1.0 : 0.0;
            /* the sheet's shadow on the water, just off its edge */
            if (under < 0.0 && uPaper.x != 2.0) col *= 1.0 - 0.38 * exp(under * 26.0);
            vec3 paper = uPaperCol * (1.0 + 0.07 * fibre(gl_FragCoord.xy, vec2(1.1, 0.7)) + 0.04 * (sw - 0.5));
            vec3 print = cache(p);
            paper *= mix(vec3(1.0), mix(print, vec3(1.0), 0.3) * 1.06, uPaper.z * 0.38);
            /* the edge, bent up off the water: a light crease, then the curl */
            float uu = max(under, 0.0), curl = exp(-uu * mix(70.0, 40.0, lifting));
            paper = screen(paper, vec3(0.22) * curl) * (1.0 - 0.1 * exp(-uu * 500.0));
            float aa = clamp(under * uH + 0.5, 0.0, 1.0);
            col = mix(col, paper, aa);
        }

        /* a gentle vignette, and dither so the light doesn't band */
        vec2 v = (p - vec2(A, 1.0) * 0.5) / vec2(A, 1.0);
        col *= 1.0 - 0.28 * dot(v, v) * 1.4;
        col += (texelFetch(uNoise, ivec2(gl_FragCoord.xy) & 255, 0).a - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });

static GLuint prog_build, prog_show, vao, tex_ops, tex_noise;
static GLuint cache_tex[2], cache_fbo[2];
static int cache_w, cache_h, cache_m, cache_for_w, cache_for_h;
static int cache_cnt[2] = {-1, -1}, cache_rows[2], cur;
static Recipe rec;
static int rec_idx = -1;
static float rec_aspect;
#define NSTRIP 20
#define MARGIN 0.07f

static GLuint compile(GLenum type, const char *a, const char *b) {
    GLuint s = glCreateShader(type);
    const char *src[2] = {a, b ? b : ""};
    glShaderSource(s, 2, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "marbling: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, VS, NULL));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, FS_COMMON, fs));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}

/* tiling value noise, `per` cells across the 256 texels */
static float lattice(const float *g, int per, int x, int y) {
    float fx = (float)x * per / 256, fy = (float)y * per / 256;
    int ix = (int)fx, iy = (int)fy;
    float u = fx - ix, v = fy - iy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
#define G(i, j) g[(((iy + (j)) % per) * per + (ix + (i)) % per)]
    return (G(0, 0) * (1 - u) + G(1, 0) * u) * (1 - v) + (G(0, 1) * (1 - u) + G(1, 1) * u) * v;
#undef G
}

static void noise_init(void) {
    unsigned char *px = malloc(256 * 256 * 4);
    float *acc = calloc(256 * 256 * 3, sizeof *acc), *g = malloc(256 * 256 * sizeof *g);
    static const int per[3][3] = {{4, 8, 16}, {4, 8, 16}, {64, 128, 256}};
    static const float wt[3] = {0.55f, 0.3f, 0.15f};
    rng = 0x5eed1234U;
    for (int c = 0; c < 3; c++)
        for (int o = 0; o < 3; o++) {
            int P = per[c][o];
            for (int i = 0; i < P * P; i++) g[i] = frand();
            for (int y = 0; y < 256; y++)
                for (int x = 0; x < 256; x++) acc[(y * 256 + x) * 3 + c] += wt[o] * lattice(g, P, x, y);
        }
    for (int i = 0; i < 256 * 256; i++) {
        for (int c = 0; c < 3; c++) {
            float v = acc[i * 3 + c];
            if (c < 2) v = 0.5f + (v - 0.5f) * 2.2f;       /* spread the smooth ones out */
            px[i * 4 + c] = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255);
        }
        px[i * 4 + 3] = (unsigned char)(frand() * 255);
    }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    free(px); free(acc); free(g);
}

static void gl_init(void) {
    prog_build = mk_prog(FS_BUILD);
    prog_show = mk_prog(FS_SHOW);
    glGenVertexArrays(1, &vao);
    glGenTextures(1, &tex_ops);
    glBindTexture(GL_TEXTURE_2D, tex_ops);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 3, MAXOPS, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(prog_build);
    glUniform1i(glGetUniformLocation(prog_build, "uOps"), 1);
    glUseProgram(prog_show);
    glUniform1i(glGetUniformLocation(prog_show, "uCache"), 0);
    glUniform1i(glGetUniformLocation(prog_show, "uOps"), 1);
    glUniform1i(glGetUniformLocation(prog_show, "uNoise"), 2);
    noise_init();
}

/* (re)make the two caches for the window's size: the screen plus a margin,
 * so ink dragged in from just off screen is really there */
static void cache_size(void) {
    if (cache_for_w == W && cache_for_h == H) return;
    if (cache_for_w) { glDeleteTextures(2, cache_tex); glDeleteFramebuffers(2, cache_fbo); }
    cache_for_w = W; cache_for_h = H;
    cache_m = (int)ceilf(MARGIN * H);
    cache_w = W + 2 * cache_m; cache_h = H + 2 * cache_m;
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &cache_tex[i]);
        glBindTexture(GL_TEXTURE_2D, cache_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cache_w, cache_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1, &cache_fbo[i]);
        glBindFramebuffer(GL_FRAMEBUFFER, cache_fbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cache_tex[i], 0);
        cache_cnt[i] = -1;
    }
}

/* bake rows [y0, y1) of cache i with the first n ops */
static void bake(int i, int n, int y0, int y1) {
    if (y1 > cache_h) y1 = cache_h;
    if (y0 >= y1) return;
    glBindFramebuffer(GL_FRAMEBUFFER, cache_fbo[i]);
    glViewport(0, 0, cache_w, cache_h);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, y0, cache_w, y1 - y0);
    glUseProgram(prog_build);
    glUniform1i(glGetUniformLocation(prog_build, "uN"), n);
    glUniform1f(glGetUniformLocation(prog_build, "uH"), (float)H);
    glUniform1f(glGetUniformLocation(prog_build, "uM"), (float)cache_m / H);
    glUniform3fv(glGetUniformLocation(prog_build, "uBath"), 1, rec.bath);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
}

static void load_recipe(int idx) {
    float aspect = (float)W / H;
    if (idx == rec_idx && aspect == rec_aspect) return;
    gen_recipe(idx, aspect, &rec);
    rec_idx = idx; rec_aspect = aspect;
    float buf[MAXOPS * 12] = {0};
    for (int i = 0; i < rec.n; i++) memcpy(buf + i * 12, rec.op[i].v, sizeof rec.op[i].v);
    glBindTexture(GL_TEXTURE_2D, tex_ops);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 3, MAXOPS, GL_RGBA, GL_FLOAT, buf);
    cache_cnt[0] = cache_cnt[1] = -1;
}

static float ease(float x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return x * x * (3 - 2 * x); }

/* the bath at clock time t, into the framebuffer target */
static void render_bath(double t, GLuint target) {
    double lt;
    int idx = recipe_at(t, &lt);
    glBindVertexArray(vao);
    cache_size();
    load_recipe(idx);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_ops);
    glActiveTexture(GL_TEXTURE0);

    /* done: ops finished; anim: the one in motion */
    int done = 0, anim = -1;
    float prog = 0;
    while (done < rec.n && lt >= rec.op[done].t0 + rec.op[done].dur) done++;
    if (done < rec.n && lt >= rec.op[done].t0) { anim = done; prog = (float)((lt - rec.op[done].t0) / rec.op[done].dur); }

    /* the shown cache must hold exactly the finished ops: take the one
     * baked in the background if it's ready, else bake it now */
    if (cache_cnt[cur] != done) {
        if (cache_cnt[cur ^ 1] == done && cache_rows[cur ^ 1] >= cache_h) cur ^= 1;
        else { bake(cur, done, 0, cache_h); cache_cnt[cur] = done; cache_rows[cur] = cache_h; }
    }
    /* meanwhile bake the next one, a strip a frame */
    if (done < rec.n) {
        int nx = cur ^ 1;
        if (cache_cnt[nx] != done + 1) { cache_cnt[nx] = done + 1; cache_rows[nx] = 0; }
        if (cache_rows[nx] < cache_h) {
            int strip = (cache_h + NSTRIP - 1) / NSTRIP;
            bake(nx, done + 1, cache_rows[nx], cache_rows[nx] + strip);
            cache_rows[nx] += strip;
        }
    }

    /* the paper */
    float paper[4] = {0, 0, 0, 0};
    double pt = lt - rec.paper_at;
    if (pt >= 0) {
        if (pt < COVER) { paper[0] = 1; paper[1] = ease((float)(pt / COVER)); }
        else if (pt < COVER + REST) { paper[0] = 2; paper[1] = 1; }
        else { paper[0] = 3; paper[1] = ease((float)((pt - COVER - REST) / LIFT)); }
        paper[2] = ease((float)(pt / (COVER + REST + 0.4)));
    }
    float next[3];
    hex(dark_of(idx + 1) ? P->bath_dark : P->bath_cream, next);

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_show);
    glBindTexture(GL_TEXTURE_2D, cache_tex[cur]);
    glUniform2f(glGetUniformLocation(prog_show, "uRes"), (float)W, (float)H);
    glUniform2f(glGetUniformLocation(prog_show, "uCacheRes"), (float)cache_w, (float)cache_h);
    glUniform1f(glGetUniformLocation(prog_show, "uH"), (float)H);
    glUniform1f(glGetUniformLocation(prog_show, "uM"), (float)cache_m / H);
    /* slow drifts, wrapped so they stay smooth however long it runs */
    glUniform4f(glGetUniformLocation(prog_show, "uDrift"), (float)fmod(t * 0.006, 1.0), (float)fmod(-t * 0.004, 1.0),
                (float)fmod(-t * 0.0102, 1.0), (float)fmod(t * 0.0068, 1.0));
    glUniform1f(glGetUniformLocation(prog_show, "uLight"), (float)fmod(t * 0.05, 6.283185307179586));
    glUniform1i(glGetUniformLocation(prog_show, "uAnim"), anim);
    glUniform1f(glGetUniformLocation(prog_show, "uProg"), prog);
    glUniform3fv(glGetUniformLocation(prog_show, "uNext"), 1, next);
    glUniform4fv(glGetUniformLocation(prog_show, "uPaper"), 1, paper);
    glUniform3fv(glGetUniformLocation(prog_show, "uPaperCol"), 1, P->paper);
    glUniform2f(glGetUniformLocation(prog_show, "uPdir"), cosf(rec.pdir), sinf(rec.pdir));
    glDrawArrays(GL_TRIANGLES, 0, 3);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "marbling");
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

/* portrait: the bath is made landscape (W x H swapped) in here, then turned */
static GLuint prog_rot, rot_tex, rot_fbo;
static int rot_w, rot_h;
static const char *FS_ROT = GLSL(
    uniform sampler2D uSrc;
    uniform int uW;             /* the real window's width */
    out vec4 o;
    void main() { o = texelFetch(uSrc, ivec2(int(gl_FragCoord.y), uW - 1 - int(gl_FragCoord.x)), 0); });

static void render(double t, GLuint target) {
    t += span_toff;
    if (W >= H) { render_bath(t, target); return; }
    int rw = W, rh = H;
    if (!prog_rot) {
        prog_rot = mk_prog(FS_ROT);
        glUseProgram(prog_rot);
        glUniform1i(glGetUniformLocation(prog_rot, "uSrc"), 0);
    }
    if (rot_w != rh || rot_h != rw) {
        if (rot_w) { glDeleteTextures(1, &rot_tex); glDeleteFramebuffers(1, &rot_fbo); }
        rot_w = rh; rot_h = rw;
        glGenTextures(1, &rot_tex);
        glBindTexture(GL_TEXTURE_2D, rot_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rot_w, rot_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glGenFramebuffers(1, &rot_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, rot_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rot_tex, 0);
    }
    W = rh; H = rw;
    render_bath(t, rot_fbo);
    W = rw; H = rh;
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_rot);
    glUniform1i(glGetUniformLocation(prog_rot, "uW"), W);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, rot_tex);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

/* spanning: this output's place in the round, counted from the front one
 * (under the middle of the layout), sets which recipes it plays and when */
static void span_setup(void) {
    if (span_me < 0) return;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
    }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    int k = (span_me - front + span_n) % span_n;
    /* a recipe ahead per step round (so neighbours differ), and that
     * fraction of the way into it (so they don't finish together) */
    rec_shift = k;
    span_toff = recipe_start(1) * k / span_n;
}

static void usage(void) {
    fprintf(stderr,
        "usage: marbling [--window | --root | --layer] [--seed N] [--recipe NAME]\n"
        "                [--palette NAME] [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--shot FILE.ppm | --bench N] [--size WxH] [--at SECONDS]\n"
        "recipes: stone ripple nonpareil chevron bouquet whirl\n"
        "palettes: brass ebru florentine sea ink (default: from the seed)\n");
    exit(1);
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

static GLuint offscreen(void) {
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NPAL; k++) if (!strcmp(n, PALETTES[k].name)) opt_palette = k;
            if (opt_palette < 0) usage();
        }
        else if (!strcmp(a, "-recipe") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NKIND; k++) if (!strcmp(n, KIND_NAME[k])) opt_recipe = k;
            if (opt_recipe < 0) usage();
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
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (!opt_seed) opt_seed = 1;
    P = &PALETTES[opt_palette >= 0 ? opt_palette : (int)(mix32(opt_seed * 2246822519U ^ 0x9a1e77eU) % NPAL)];
    span_setup();
    int offline = opt_shot || opt_bench > 0;

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offline) wl_init();
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
        if (offline) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Marbling");
        XClassHint chint = {"marbling", "Marbling"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offline) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (opt_shot) {
        GLuint fbo = offscreen();
        render(opt_at, fbo);
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }
    if (opt_bench > 0) {
        GLuint fbo = offscreen();
        for (int i = 5; i > 0; i--) render(opt_at - i / 60.0, fbo);   /* warm up: shaders, the first bake */
        glFinish();
        double sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double t0 = now();
            render(opt_at + i / 60.0, fbo);
            glFinish();
            double ms = (now() - t0) * 1e3;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("marbling: %d frames at %dx%d from %.1f s: avg %.3f ms, worst %.3f ms\nrenderer: %s\n",
               opt_bench, W, H, opt_at, sum / opt_bench, worst, (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    if (span_me >= 0 && opt_epoch) t0 -= wallclock() - opt_epoch;   /* the clock runs from the shared epoch */
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
        double t = now();
        render(t - t0, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
