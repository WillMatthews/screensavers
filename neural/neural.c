/*
 * neural — neural networks visibly learning, live, on the lock screen.
 *
 * Every network here is trained for real while you watch: forward and
 * backward passes, Adam, minibatches, all written out by hand in C (no ML
 * library) and run single-threaded on the CPU at a capped rate (the nets are
 * small; a lock screen should cost a few percent of one core). The GPU only
 * draws. Four acts take turns, a few minutes each, each opening with a small
 * title card:
 *
 *   I    IMPLICIT IMAGE   a coordinate network, (x, y) -> (r, g, b), learns
 *                         one procedurally drawn picture (the clock at the
 *                         time the act began, the hostname in large type, a
 *                         mountain sunset, a zone plate). A SIREN (sine
 *                         activations; --model fourier for a ReLU MLP on random Fourier
 *                         features instead). Its output is shown beside the target,
 *                         sharpening from blur to crisp, with the loss on a
 *                         log axis and an error spectrogram (error per spatial
 *                         frequency band over time): the low bands go dark
 *                         first, which is spectral bias.
 *   II   DECISION BOUNDARIES  a 2-12-12-12-1 tanh MLP on a 2D point set
 *                         (spirals, moons, checkerboard, circles, xor); the
 *                         boundary field animates as it trains, every hidden
 *                         unit's activation map gets a tile, and the network
 *                         diagram has its edges coloured and thickened by
 *                         weight sign and size, pulsing with the gradient
 *                         flowing back through them.
 *   III  CHARACTER MODEL  an MLP language model (8 characters of context,
 *                         learned embeddings, one tanh layer, softmax) learns
 *                         to write away-messages from a small embedded corpus;
 *                         it types a fresh sample every few seconds, with its
 *                         next-character distribution under the cursor, and
 *                         keeps a log of samples epoch by epoch, plus its
 *                         character embeddings projected to 2D (PCA).
 *   IV   OPTIMISERS       SGD, momentum, RMSProp and Adam race on Rosenbrock, then
 *                         Rastrigin (noisy gradients) or Himmelblau, drawn in 3D and as
 *                         a contour map, with their losses.
 *
 * Training is deterministic: each act's net starts from an init seeded by
 * --seed and the act's number, and runs fixed steps at a fixed rate from the
 * act's start, so any moment can be replayed (that is what --shot does, and
 * what a late-starting instance does to catch up).
 *
 * Palettes (--palette, or picked from the seed): spectral (dark, a spectral
 * diverging map), mono (dark greys, one accent), paper (light), terminal
 * (green phosphor). Lock info, small: the time and how long the screen has
 * been locked (swaylock's start time).
 *
 * Spanning: run one instance per monitor with the same --seed, --epoch and
 * --layout (every output's logical rect), and --output = this one's index.
 * Each trains its own copy of the same nets in lockstep off the shared epoch.
 * The output under the middle of the layout (flightsim's front-output rule)
 * shows the main view; the others, left to right, take turns at the
 * structural view (network diagram, unit maps, embeddings, contour map) and
 * the numbers (loss curves, spectrogram, activation tiles, sample log). On its
 * own, an output shows a composition of all of it, landscape or portrait.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size (and the training cost per
 * act); --gradcheck checks every net's backward pass by finite differences.
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
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include "proto-layer-shell.h"
#include "../common/hidpi.h"

#define PI 3.14159265f
#define TAU 6.28318531f

static int W = 1280, H = 800;
static float PX = 1;                    /* buffer pixels per logical pixel (--layer on a scaled output) */
static int opt_root, opt_layer, opt_bench, opt_gradcheck, opt_act = -1;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot, *opt_palette, *opt_name = "Will";
static double opt_at = 60;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the first act starts at */
static double act_len = 180;           /* --act-len: seconds per act */
static double opt_budget = 6;          /* ms of training per frame at most (live) */
static int opt_pic = -1, opt_model = -1;   /* Act I: force the picture, the model */

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { R_FULL, R_FRONT, R_DIAG, R_STATS, R_SIDE };
static int role = R_FULL;

static void die(const char *what) { fprintf(stderr, "neural: %s\n", what); exit(1); }

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

static double cpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }

/* ---- deterministic randomness ---- */
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
typedef struct { uint64_t s; } Rng;
static Rng rng_new(uint32_t a, uint32_t b) { Rng r = {((uint64_t)hash32(a) << 32 | hash32(b ^ 0x9e3779b9u)) | 1}; return r; }
static uint32_t rnd_u(Rng *r) {
    uint64_t x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return (uint32_t)((x * 0x2545F4914F6CDD1Dull) >> 32);
}
static float rnd_f(Rng *r) { return (rnd_u(r) >> 8) * (1.0f / 16777216.0f); }
static float rnd_s(Rng *r) { return rnd_f(r) * 2 - 1; }
static float rnd_n(Rng *r) {
    float u = fmaxf(rnd_f(r), 1e-7f), v = rnd_f(r);
    return sqrtf(-2 * logf(u)) * cosf(TAU * v);
}

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

static char host[64];
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

/* ---- glyph atlas: printable ASCII in each text style used, rasterised at
 * its exact pixel size (as gauges) ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 48
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
enum { F_REG, F_SEMI, F_LIGHT, F_MONO, NFACE };
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    static const char *pats[NFACE] = {
        "Inter,Noto Sans,DejaVu Sans,sans-serif:weight=regular",
        "Inter,Noto Sans,DejaVu Sans,sans-serif:weight=semibold",
        "Inter,Noto Sans,DejaVu Sans,sans-serif:weight=light",
        "JetBrainsMono Nerd Font Mono,JetBrains Mono,DejaVu Sans Mono,monospace:weight=regular",
    };
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        FcPattern *pat = FcNameParse((const FcChar8 *)pats[i]);
        FcPatternAddString(pat, FC_FONTFORMAT, (const FcChar8 *)"TrueType");
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        int idx = 0;
        FcPatternGetInteger(m, FC_INDEX, 0, &idx);
        if (FT_New_Face(ft, (const char *)file, idx, &faces[i])) die("can't load the font");
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
    return nstyles++;
}

static void build_atlas(void) {
    free(atlas);
    int cap = 4096;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < 96; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c == 95 ? 0xB7 : (FT_ULong)(32 + c);    /* 0x7f: a middle dot */
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

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p; p++) w += glyph(s, (unsigned char)*p)->adv + (p[1] ? track : 0);
    return w;
}

static float cap_h(int s) { return glyph(s, 'H')->top; }

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
 * track is extra spacing in px; returns the width */
static float text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    pen = roundf(pen);
    y = roundf(y);
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph(s, (unsigned char)*p);
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

static float textf(int s, float x, float y, int align, const float *col, float track, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return text(s, x, y, align, buf, col, track);
}

/* ---- palettes ---- */
typedef struct {
    const char *name;
    int light, mono_text, img_mode;    /* img_mode: 0 colour, 1 grey, 2 green phosphor */
    const char *bg, *panel, *grid, *fg, *dim, *accent, *neg, *mid, *pos;
    const char *seq[5], *cls[2], *opt[4];
} PalDef;

static const PalDef PALS[] = {
    {"spectral", 0, 0, 0, "#0c0e13", "#12151c", "#262b36", "#e8eaf0", "#7c8494", "#fdae61",
     "#3288bd", "#161a22", "#d53e4f",
     {"#0f1220", "#3288bd", "#66c2a5", "#e6f598", "#fee08b"}, {"#f46d43", "#3288bd"},
     {"#fee08b", "#66c2a5", "#5e9ce0", "#f46d43"}},
    {"mono", 0, 0, 1, "#0b0b0c", "#121214", "#2a2a2e", "#ececec", "#7d7d82", "#ff6a3d",
     "#6b7380", "#141416", "#f2f2f2",
     {"#0e0e10", "#3a3a3e", "#77777c", "#b8b8bc", "#ffffff"}, {"#ff6a3d", "#d8d8d8"},
     {"#8a8a90", "#c4c4c8", "#ffffff", "#ff6a3d"}},
    {"paper", 1, 0, 0, "#f5f3ee", "#fbfaf7", "#dedad1", "#1f2328", "#6d7178", "#c2410c",
     "#2b6cb0", "#f5f3ee", "#c53030",
     {"#fbfaf7", "#f6d38a", "#e07b39", "#9b3a5a", "#2d1e40"}, {"#d0582a", "#2b6cb0"},
     {"#a07d10", "#2f855a", "#2b6cb0", "#c53030"}},
    {"terminal", 0, 1, 2, "#030703", "#071007", "#113011", "#7dff86", "#2f8f3a", "#c8ffc8",
     "#1e6b2a", "#051005", "#b4ffb0",
     {"#020602", "#0b3d12", "#1f8a2e", "#5cff6e", "#d8ffd8"}, {"#ffb000", "#4dff6a"},
     {"#2f9f3f", "#5cff6e", "#ffb000", "#d8ffd8"}},
};
#define NPAL ((int)(sizeof PALS / sizeof *PALS))

static struct {
    int light, mono_text, img_mode;
    float bg[4], panel[4], grid[4], fg[4], dim[4], accent[4], neg[4], mid[4], pos[4];
    float seq[5][4], cls[2][4], opt[4][4];
} P;
static const char *pal_name;

static void hex4(const char *h, float *rgba) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgba[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
    rgba[3] = 1;
}

static void pal_set(const char *name) {
    int k = (int)(hash32(opt_seed * 7 + 3) % NPAL);
    if (name) {
        for (k = 0; k < NPAL && strcmp(PALS[k].name, name); k++) {}
        if (k == NPAL) die("unknown palette (spectral, mono, paper, terminal)");
    }
    const PalDef *d = &PALS[k];
    pal_name = d->name;
    P.light = d->light; P.mono_text = d->mono_text; P.img_mode = d->img_mode;
    hex4(d->bg, P.bg); hex4(d->panel, P.panel); hex4(d->grid, P.grid); hex4(d->fg, P.fg);
    hex4(d->dim, P.dim); hex4(d->accent, P.accent); hex4(d->neg, P.neg); hex4(d->mid, P.mid);
    hex4(d->pos, P.pos);
    for (int i = 0; i < 5; i++) hex4(d->seq[i], P.seq[i]);
    for (int i = 0; i < 2; i++) hex4(d->cls[i], P.cls[i]);
    for (int i = 0; i < 4; i++) hex4(d->opt[i], P.opt[i]);
}

/* a colour with alpha */
static const float *ca(const float *c, float a) {
    static float ring[64][4];
    static int k;
    float *o = ring[k++ & 63];
    o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = c[3] * a;
    return o;
}
static const float *cmix(const float *a, const float *b, float t) {
    static float ring[64][4];
    static int k;
    float *o = ring[k++ & 63];
    for (int i = 0; i < 4; i++) o[i] = a[i] + (b[i] - a[i]) * t;
    return o;
}
/* the diverging map at v in [-1, 1] */
static const float *cdiv(float v) {
    v = clampf(v, -1, 1);
    return v >= 0 ? cmix(P.mid, P.pos, v) : cmix(P.mid, P.neg, -v);
}

/* ---- the network core: dense layers, Adam ---- */
typedef struct {
    int in, out;
    float *W, *b;                      /* W[in][out]: out is contiguous */
    float *gW, *gb, *mW, *vW, *mb, *vb;
} Dense;

static void dense_alloc(Dense *L, int in, int out) {
    L->in = in; L->out = out;
    size_t n = (size_t)in * out;
    float *m = calloc(n * 4 + (size_t)out * 4, sizeof *m);
    L->W = m; L->gW = m + n; L->mW = m + 2 * n; L->vW = m + 3 * n;
    L->b = m + 4 * n; L->gb = L->b + out; L->mb = L->gb + out; L->vb = L->mb + out;
}

static void dense_free(Dense *L) { free(L->W); memset(L, 0, sizeof *L); }

/* weights uniform in [-a, a], biases uniform in [-ab, ab] */
static void dense_init(Dense *L, Rng *r, float a, float ab) {
    size_t n = (size_t)L->in * L->out;
    memset(L->gW, 0, n * 3 * sizeof(float));
    memset(L->gb, 0, (size_t)L->out * 3 * sizeof(float));
    for (size_t i = 0; i < n; i++) L->W[i] = a * rnd_s(r);
    for (int o = 0; o < L->out; o++) L->b[o] = ab * rnd_s(r);
}

/* Z[B][out] = X[B][in] W + b */
static void dense_fwd(const Dense *L, const float *restrict X, int B, float *restrict Z) {
    const int in = L->in, out = L->out;
    for (int s = 0; s < B; s++) {
        float *restrict z = Z + (size_t)s * out;
        const float *restrict x = X + (size_t)s * in;
        memcpy(z, L->b, (size_t)out * sizeof *z);
        for (int i = 0; i < in; i++) {
            const float xi = x[i];
            if (xi == 0) continue;
            const float *restrict w = L->W + (size_t)i * out;
            for (int o = 0; o < out; o++) z[o] += xi * w[o];
        }
    }
}

/* accumulate dL/dW, dL/db from dZ; dX = dZ W^T if wanted */
static void dense_bwd(Dense *L, const float *restrict X, const float *restrict dZ, int B, float *restrict dX) {
    const int in = L->in, out = L->out;
    for (int s = 0; s < B; s++) {
        const float *restrict dz = dZ + (size_t)s * out;
        const float *restrict x = X + (size_t)s * in;
        for (int o = 0; o < out; o++) L->gb[o] += dz[o];
        for (int i = 0; i < in; i++) {
            const float xi = x[i];
            float *restrict g = L->gW + (size_t)i * out;
            const float *restrict w = L->W + (size_t)i * out;
            if (dX) {
                float acc = 0;
                for (int o = 0; o < out; o++) acc += dz[o] * w[o];
                dX[(size_t)s * in + i] = acc;
            }
            if (xi == 0) continue;
            for (int o = 0; o < out; o++) g[o] += xi * dz[o];
        }
    }
}

static void dense_zero(Dense *L) {
    memset(L->gW, 0, (size_t)L->in * L->out * sizeof(float));
    memset(L->gb, 0, (size_t)L->out * sizeof(float));
}

static void adam_vec(float *restrict p, const float *restrict g, float *restrict m, float *restrict v, size_t n,
                     float lr, int t, float wd) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float c1 = 1 - powf(b1, (float)t), c2 = 1 - powf(b2, (float)t);
    const float a = lr * sqrtf(c2) / c1;
    for (size_t i = 0; i < n; i++) {
        float gi = g[i] + wd * p[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= a * m[i] / (sqrtf(v[i]) + eps);
    }
}

static void dense_adam(Dense *L, float lr, int t, float wd) {
    adam_vec(L->W, L->gW, L->mW, L->vW, (size_t)L->in * L->out, lr, t, wd);
    adam_vec(L->b, L->gb, L->mb, L->vb, (size_t)L->out, lr, t, 0);
}

static int dense_params(const Dense *L) { return L->in * L->out + L->out; }

/* sin and cos, fast and vectorisable: reduce to [-pi, pi], reflect into
 * [-pi/2, pi/2], odd Taylor to x^9 (error < 4e-6) */
static inline float fsin(float x) {
    float r = x - TAU * floorf(x * (1.0f / TAU) + 0.5f);
    r = r > PI * 0.5f ? PI - r : r;
    r = r < -PI * 0.5f ? -PI - r : r;
    float r2 = r * r;
    return r * (1 + r2 * (-1.0f / 6 + r2 * (1.0f / 120 + r2 * (-1.0f / 5040 + r2 * (1.0f / 362880)))));
}
static inline float fcos(float x) { return fsin(x + PI * 0.5f); }

/* ---- schedule: which act is on, and how far into it ---- */
enum { A_FIT, A_CLS, A_CHAR, A_OPT, NACT };
#define T_TRAIN 4.0                    /* training starts this far into an act */
static const double SPS[NACT] = {6, 30, 20, 12};   /* training steps a second */
static int nphase(int act) { return act == A_CLS || act == A_OPT ? 2 : 1; }
static double phase_len(int act) { return (act_len - T_TRAIN) / nphase(act); }
static int phase_steps(int act) { return (int)floor(phase_len(act) * SPS[act]); }

static struct {
    long cycle;                        /* which act slot since the epoch */
    int act, phase, variant;
    int live;                          /* a run is set up */
    int step;                          /* steps done in this phase */
    double t_act;                      /* seconds into the act (for drawing) */
    double wt_act0;                    /* wall-clock time the act began */
} run;

static double t_wall0;                 /* the epoch: wall time the first act starts */

/* ======================================================================
 * Act I: an implicit image. A coordinate MLP learns one picture.
 * ====================================================================== */
#define IMG 128                        /* training resolution */
#define NPX (IMG * IMG)
#define FH 64                          /* hidden width */
#define FFM 64                         /* random Fourier features (sin + cos each) */
#define BATCH1 1024
#define EVAL_E 8                       /* steps per full-image evaluation (done a slice a step) */
#define EROWS (IMG / EVAL_E)
#define TILE 32                        /* unit activation maps are evaluated at TILE x TILE */
#define NBAND 20                       /* spectrogram frequency bands */
#define MAXSNAP 1024
enum { M_SIREN, M_FOURIER };
enum { PIC_CLOCK, PIC_HOST, PIC_LAND, PIC_ZONE, NPIC };
static const char *PIC_NAME[NPIC] = {"the clock", "the hostname", "a mountain sunset", "a zone plate"};

static struct {
    int model, pic, nl, nsteps;
    float tgt[NPX * 3];
    float omega[4];
    float Bf[FFM * 2];                 /* Fourier feature frequencies (cycles per unit) */
    Dense L[4], S[4];                  /* the net, and the snapshot being evaluated */
    float *a[5], *z[4], *g0, *g1, *xy; /* per-sample buffers, MAXB rows */
    int *idx;
    float *loss;                       /* minibatch loss per step */
    /* evaluation */
    float ev[NPX * 3];                 /* the snapshot's image, being filled */
    float pub[2][NPX * 3];             /* the last two finished ones: [1] newest */
    int pub_step[2], pub_ver;
    int snap_step;
    int nsnap;
    float psnr[MAXSNAP];               /* per snapshot */
    int snap_at[MAXSNAP];
    float spec[MAXSNAP][NBAND];        /* log10 relative error per band */
    float tiles[3][FH][TILE * TILE];   /* hidden activations of the newest snapshot */
    int ntl;                           /* hidden layers with tiles */
    float lr;
} F;
#define MAXB 2048

static float fit_xy(int i) { return (i + 0.5f) / IMG * 2 - 1; }

/* ---- the pictures, rendered at 4x and box-filtered down ---- */
static float pic_mask[512 * 512];      /* type for the hostname */

static void lerp3(float *o, const float *a, const float *b, float t) {
    for (int i = 0; i < 3; i++) o[i] = a[i] + (b[i] - a[i]) * t;
}
static void hex3(const char *h, float *rgb) { float c[4]; hex4(h, c); memcpy(rgb, c, 3 * sizeof *rgb); }

static float vnoise1(float x, uint32_t salt) {
    int i = (int)floorf(x);
    float f = x - i, u = f * f * (3 - 2 * f);
    float a = (hash32((uint32_t)i * 747796405u + salt) & 0xffff) / 65535.0f;
    float b = (hash32((uint32_t)(i + 1) * 747796405u + salt) & 0xffff) / 65535.0f;
    return a + (b - a) * u;
}
static float fbm1(float x, uint32_t salt) {
    float s = 0, a = 0.5f;
    for (int o = 0; o < 5; o++) { s += a * vnoise1(x, salt + o * 17); x *= 2.03f; a *= 0.5f; }
    return s;
}

static float seg_dist(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
    float t = clampf((wx * vx + wy * vy) / (vx * vx + vy * vy), 0, 1);
    float dx = wx - vx * t, dy = wy - vy * t;
    return sqrtf(dx * dx + dy * dy);
}

static float str_adv(FT_Face f, const char *s) {
    float w = 0;
    for (const char *p = s; *p; p++)
        if (!FT_Load_Char(f, (unsigned char)*p, FT_LOAD_DEFAULT)) w += f->glyph->advance.x / 64.0f;
    return w;
}

static void host_mask(void) {
    memset(pic_mask, 0, sizeof pic_mask);
    FT_Face f = faces[F_SEMI];
    const char *s = host[0] ? host : "localhost";
    float px = 200;
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
    px = fminf(px * 512 * 0.84f / str_adv(f, s), 230);
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
    float w = str_adv(f, s);
    int pen = (int)((512 - w) / 2), base = (int)(256 + px * 0.30f);
    for (const char *p = s; *p; p++) {
        if (FT_Load_Char(f, (unsigned char)*p, FT_LOAD_RENDER)) continue;
        FT_GlyphSlot g = f->glyph;
        for (unsigned yy = 0; yy < g->bitmap.rows; yy++)
            for (unsigned xx = 0; xx < g->bitmap.width; xx++) {
                int X = pen + g->bitmap_left + (int)xx, Y = base - g->bitmap_top + (int)yy;
                if (X >= 0 && X < 512 && Y >= 0 && Y < 512)
                    pic_mask[Y * 512 + X] = fmaxf(pic_mask[Y * 512 + X], g->bitmap.buffer[yy * g->bitmap.pitch + xx] / 255.0f);
            }
        pen += (int)(g->advance.x >> 6);
    }
}

/* one sample of picture pic at (u, v) in [0, 1] (v down) */
static void pic_at(int pic, float u, float v, const struct tm *tm, float *o) {
    float c1[3], c2[3], c3[3];
    if (pic == PIC_CLOCK) {
        float x = u - 0.5f, y = v - 0.5f, r = sqrtf(x * x + y * y);
        hex3("#1d2d50", c1); hex3("#0a0f1e", c2);
        lerp3(o, c1, c2, clampf(r * 1.6f, 0, 1));
        float R = 0.43f;
        if (r < R + 0.025f) { hex3("#c9a24a", c1); lerp3(o, o, c1, 1); }
        if (r < R) {
            hex3("#f4ecdc", c1); hex3("#ddd0b8", c2);
            lerp3(o, c1, c2, r / R);
            float ang = atan2f(x, -y), sec = ang / TAU * 60;
            if (sec < 0) sec += 60;
            float near = fabsf(sec - roundf(sec));
            int tick = (int)roundf(sec) % 60;
            int hour = tick % 5 == 0;
            float len = hour ? 0.075f : 0.03f, wid = hour ? 0.012f : 0.004f;
            hex3("#1b1f2a", c1);
            if (r > R - 0.02f - len && r < R - 0.02f && near * TAU / 60 * r < wid) lerp3(o, o, c1, 1);
            float hh = (tm->tm_hour % 12 + tm->tm_min / 60.0f) / 12 * TAU;
            float mm = (tm->tm_min + tm->tm_sec / 60.0f) / 60 * TAU, ss = tm->tm_sec / 60.0f * TAU;
            float d = seg_dist(x, y, -sinf(hh) * 0.05f, cosf(hh) * 0.05f, sinf(hh) * 0.24f, -cosf(hh) * 0.24f);
            if (d < 0.022f) lerp3(o, o, c1, 1);
            d = seg_dist(x, y, -sinf(mm) * 0.06f, cosf(mm) * 0.06f, sinf(mm) * 0.36f, -cosf(mm) * 0.36f);
            if (d < 0.014f) lerp3(o, o, c1, 1);
            hex3("#d64933", c2);
            d = seg_dist(x, y, -sinf(ss) * 0.09f, cosf(ss) * 0.09f, sinf(ss) * 0.38f, -cosf(ss) * 0.38f);
            if (d < 0.0055f || r < 0.028f) lerp3(o, o, c2, 1);
            if (r < 0.01f) lerp3(o, o, c1, 1);
        }
    } else if (pic == PIC_HOST) {
        hex3("#1b1446", c1); hex3("#d1495b", c2); hex3("#edae49", c3);
        lerp3(o, c1, c2, clampf(u * 0.55f + v * 0.6f - 0.1f, 0, 1));
        float vig = 1 - 0.35f * ((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));
        for (int i = 0; i < 3; i++) o[i] *= vig;
        int X = (int)(u * 512), Y = (int)(v * 512);
        X = X < 0 ? 0 : X > 511 ? 511 : X;
        Y = Y < 0 ? 0 : Y > 511 ? 511 : Y;
        float m = pic_mask[Y * 512 + X];
        if (v > 0.66f && v < 0.675f && u > 0.3f && u < 0.7f) lerp3(o, o, c3, 1);
        float wh[3] = {1, 0.97f, 0.93f};
        lerp3(o, o, wh, m);
    } else if (pic == PIC_LAND) {
        hex3("#2b1b4b", c1); hex3("#e8673c", c2); hex3("#f9c784", c3);
        float hz = 0.62f, sv = v < hz ? v : 2 * hz - v;    /* the lake mirrors the sky */
        if (sv < 0.45f) lerp3(o, c1, c2, sv / 0.45f);
        else lerp3(o, c2, c3, (sv - 0.45f) / (hz - 0.45f));
        float sx = u - 0.66f, sy = sv - 0.40f, sr = sqrtf(sx * sx + sy * sy);
        float sun[3] = {1, 0.95f, 0.78f};
        lerp3(o, o, sun, sr < 0.075f ? 1 : 0.35f * expf(-(sr - 0.075f) * 14));
        static const char *rc[3] = {"#8a4f7d", "#4b2a5e", "#1c1130"};
        for (int k = 0; k < 3; k++) {
            float hgt = hz - 0.06f - 0.22f * (1 - k * 0.28f) * fbm1(u * (2.5f + k * 1.7f) + k * 9, 101 + k * 31) + k * 0.04f;
            hex3(rc[k], c1);
            if (sv > hgt) lerp3(o, o, c1, 1);
        }
        if (v > hz) {
            for (int i = 0; i < 3; i++) o[i] *= 0.72f;
            float rip = 0.5f + 0.5f * sinf(v * 260 + 3 * sinf(u * 9));
            if (rip > 0.93f) for (int i = 0; i < 3; i++) o[i] = o[i] * 0.8f + 0.12f;
        }
    } else {
        float x = u - 0.5f, y = v - 0.5f, r2 = x * x + y * y;
        hex3("#12131a", c1); hex3("#f4a259", c2); hex3("#2e7d9a", c3);
        o[0] = c1[0]; o[1] = c1[1]; o[2] = c1[2];
        if (r2 < 0.44f * 0.44f) lerp3(o, c3, c2, 0.5f + 0.5f * cosf(330 * r2));
        if (fabsf(x) > 0.47f || fabsf(y) > 0.47f) {
            int ch = ((int)floorf(u * 32) + (int)floorf(v * 32)) & 1;
            if (ch) { o[0] = 0.85f; o[1] = 0.85f; o[2] = 0.82f; }
        }
    }
}

static void fit_target(const struct tm *tm) {
    if (F.pic == PIC_HOST) host_mask();
    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++) {
            float acc[3] = {0, 0, 0}, o[3];
            for (int my = 0; my < 4; my++)
                for (int mx = 0; mx < 4; mx++) {
                    pic_at(F.pic, (x + (mx + 0.5f) / 4) / IMG, (y + (my + 0.5f) / 4) / IMG, tm, o);
                    for (int i = 0; i < 3; i++) acc[i] += o[i];
                }
            for (int i = 0; i < 3; i++) F.tgt[(y * IMG + x) * 3 + i] = clampf(acc[i] / 16, 0, 1);
        }
}

/* forward B samples at xy through layers L: a[0] is the encoded input,
 * a[nl] the output (centred: the picture is fitted minus 0.5) */
static void fit_forward(Dense *L, int B, const float *xy) {
    if (F.model == M_SIREN) memcpy(F.a[0], xy, (size_t)B * 2 * sizeof(float));
    else
        for (int s = 0; s < B; s++) {
            float *e = F.a[0] + (size_t)s * 2 * FFM;
            for (int k = 0; k < FFM; k++) {
                float p = TAU * (F.Bf[k * 2] * xy[s * 2] + F.Bf[k * 2 + 1] * xy[s * 2 + 1]);
                e[k] = fsin(p);
                e[FFM + k] = fcos(p);
            }
        }
    for (int l = 0; l < F.nl; l++) {
        dense_fwd(&L[l], F.a[l], B, F.z[l]);
        size_t n = (size_t)B * L[l].out;
        float *restrict z = F.z[l], *restrict a = F.a[l + 1];
        if (l == F.nl - 1) memcpy(a, z, n * sizeof *a);
        else if (F.model == M_SIREN) { const float w = F.omega[l]; for (size_t i = 0; i < n; i++) a[i] = fsin(w * z[i]); }
        else for (size_t i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0;
    }
}

/* the minibatch for step t: B pixels drawn from r */
static void fit_batch(Rng *r, int B) {
    for (int s = 0; s < B; s++) {
        int i = (int)(rnd_u(r) % NPX);
        F.idx[s] = i;
        F.xy[s * 2] = fit_xy(i % IMG);
        F.xy[s * 2 + 1] = fit_xy(i / IMG);
    }
}

/* forward and backward on the batch in F.xy / F.idx (gradients accumulate
 * in the layers); returns the minibatch MSE */
static float fit_grad(int B) {
    fit_forward(F.L, B, F.xy);
    float *out = F.a[F.nl], *g = F.g0;
    double loss = 0;
    for (int s = 0; s < B; s++)
        for (int c = 0; c < 3; c++) {
            float d = out[s * 3 + c] - (F.tgt[F.idx[s] * 3 + c] - 0.5f);
            loss += d * d;
            g[s * 3 + c] = 2 * d / (B * 3);
        }
    for (int l = F.nl - 1; l >= 0; l--) {
        size_t n = (size_t)B * F.L[l].out;
        float *restrict z = F.z[l];
        if (l < F.nl - 1) {
            if (F.model == M_SIREN) { const float w = F.omega[l]; for (size_t i = 0; i < n; i++) g[i] *= w * fcos(w * z[i]); }
            else for (size_t i = 0; i < n; i++) g[i] = z[i] > 0 ? g[i] : 0;
        }
        dense_bwd(&F.L[l], F.a[l], g, B, l > 0 ? F.g1 : NULL);
        float *sw = F.g0; F.g0 = F.g1; F.g1 = sw;
        g = F.g0;
    }
    return (float)(loss / (B * 3));
}

static void snap_take(void) {
    for (int l = 0; l < F.nl; l++) {
        memcpy(F.S[l].W, F.L[l].W, (size_t)F.L[l].in * F.L[l].out * sizeof(float));
        memcpy(F.S[l].b, F.L[l].b, (size_t)F.L[l].out * sizeof(float));
    }
}

/* evaluate rows [y0, y1) of the snapshot into ev */
static void snap_rows(int y0, int y1) {
    for (int y = y0; y < y1; y += MAXB / IMG) {
        int ye = y + MAXB / IMG < y1 ? y + MAXB / IMG : y1, B = (ye - y) * IMG;
        for (int s = 0; s < B; s++) { F.xy[s * 2] = fit_xy(s % IMG); F.xy[s * 2 + 1] = fit_xy(y + s / IMG); }
        fit_forward(F.S, B, F.xy);
        for (int s = 0; s < B * 3; s++) F.ev[(size_t)y * IMG * 3 + s] = F.a[F.nl][s] + 0.5f;
    }
}

/* in-place radix-2 FFT of n complex values */
static void fft(float *re, float *im, int n, int stride) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i * stride]; re[i * stride] = re[j * stride]; re[j * stride] = t;
            t = im[i * stride]; im[i * stride] = im[j * stride]; im[j * stride] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -TAU / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                float wr = cosf(ang * k), wi = sinf(ang * k);
                int a = (i + k) * stride, b = (i + k + len / 2) * stride;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
    }
}

/* power per frequency band of a luminance image (minus another) */
static void band_power(const float *img3, const float *sub3, double *pw) {
    static float re[NPX], im[NPX];
    for (int i = 0; i < NPX; i++) {
        float y = 0.299f * img3[i * 3] + 0.587f * img3[i * 3 + 1] + 0.114f * img3[i * 3 + 2];
        if (sub3) y -= 0.299f * sub3[i * 3] + 0.587f * sub3[i * 3 + 1] + 0.114f * sub3[i * 3 + 2];
        re[i] = y; im[i] = 0;
    }
    for (int y = 0; y < IMG; y++) fft(re + y * IMG, im + y * IMG, IMG, 1);
    for (int x = 0; x < IMG; x++) fft(re + x, im + x, IMG, IMG);
    for (int b = 0; b < NBAND; b++) pw[b] = 0;
    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++) {
            int fx = x < IMG / 2 ? x : x - IMG, fy = y < IMG / 2 ? y : y - IMG;
            float r = sqrtf((float)(fx * fx + fy * fy));
            if (r < 1 || r > IMG / 2) continue;
            int b = (int)(log2f(r) / log2f(IMG / 2.0f) * NBAND);
            if (b >= NBAND) b = NBAND - 1;
            pw[b] += (double)re[y * IMG + x] * re[y * IMG + x] + (double)im[y * IMG + x] * im[y * IMG + x];
        }
}

static double tgt_band[NBAND];

/* the snapshot is complete: score it, its spectrum, its unit maps; publish */
static void snap_done(void) {
    double se = 0;
    for (int i = 0; i < NPX * 3; i++) { double d = F.ev[i] - F.tgt[i]; se += d * d; }
    double mse = se / (NPX * 3);
    if (F.nsnap < MAXSNAP) {
        int k = F.nsnap++;
        F.psnr[k] = (float)(-10 * log10(fmax(mse, 1e-10)));
        F.snap_at[k] = F.snap_step;
        double pw[NBAND];
        band_power(F.ev, F.tgt, pw);
        for (int b = 0; b < NBAND; b++) F.spec[k][b] = (float)log10(fmax(pw[b] / fmax(tgt_band[b], 1e-12), 1e-6));
        /* the low bands that no integer frequency falls in: their neighbours' */
        for (int b = 0; b < NBAND; b++)
            if (tgt_band[b] <= 0) {
                int lo = b, hi = b;
                while (lo >= 0 && tgt_band[lo] <= 0) lo--;
                while (hi < NBAND && tgt_band[hi] <= 0) hi++;
                F.spec[k][b] = lo < 0 ? F.spec[k][hi < NBAND ? hi : 0] : hi >= NBAND ? F.spec[k][lo]
                             : (F.spec[k][lo] * (hi - b) + F.spec[k][hi] * (b - lo)) / (hi - lo);
            }
    }
    /* unit maps: every hidden unit over a TILE x TILE grid */
    int B = TILE * TILE;
    for (int s = 0; s < B; s++) {
        F.xy[s * 2] = ((s % TILE) + 0.5f) / TILE * 2 - 1;
        F.xy[s * 2 + 1] = (s / TILE + 0.5f) / TILE * 2 - 1;
    }
    fit_forward(F.S, B, F.xy);
    for (int l = 0; l < F.ntl; l++)
        for (int s = 0; s < B; s++)
            for (int u = 0; u < FH; u++) F.tiles[l][u][s] = F.a[l + 1][(size_t)s * FH + u];
    if (F.model == M_FOURIER)   /* ReLU maps: scale each unit to its own peak */
        for (int l = 0; l < F.ntl; l++)
            for (int u = 0; u < FH; u++) {
                float m = 1e-6f;
                for (int i = 0; i < B; i++) m = fmaxf(m, F.tiles[l][u][i]);
                for (int i = 0; i < B; i++) F.tiles[l][u][i] = F.tiles[l][u][i] / m * 2 - 1;
            }
    memcpy(F.pub[0], F.pub[1], sizeof F.pub[0]);
    memcpy(F.pub[1], F.ev, sizeof F.ev);
    F.pub_step[0] = F.pub_step[1];
    F.pub_step[1] = F.snap_step;
    F.pub_ver++;
}

static float fit_lr(int t) {
    float base = F.model == M_SIREN ? 1e-4f : 1e-3f;
    float x = (float)t / F.nsteps;
    return base * (0.1f + 0.9f * 0.5f * (1 + cosf(PI * x)));
}

static Rng fit_rng;

static void fit_init(uint32_t var, const struct tm *tm) {
    for (int l = 0; l < 4; l++) { dense_free(&F.L[l]); dense_free(&F.S[l]); }
    free(F.loss);
    if (!F.xy) {
        for (int l = 0; l < 5; l++) F.a[l] = malloc((size_t)MAXB * 2 * FFM * sizeof(float));
        for (int l = 0; l < 4; l++) F.z[l] = malloc((size_t)MAXB * 2 * FFM * sizeof(float));
        F.g0 = malloc((size_t)MAXB * 2 * FFM * sizeof(float));
        F.g1 = malloc((size_t)MAXB * 2 * FFM * sizeof(float));
        F.xy = malloc((size_t)MAXB * 2 * sizeof(float));
        F.idx = malloc((size_t)MAXB * sizeof(int));
    }
    F.pic = opt_pic >= 0 ? opt_pic : (int)(var % NPIC);
    F.model = opt_model >= 0 ? opt_model : M_SIREN;   /* Fourier features: --model fourier (noisier) */
    F.nsteps = phase_steps(A_FIT);
    F.loss = calloc((size_t)F.nsteps + 2, sizeof(float));
    fit_target(tm);
    band_power(F.tgt, NULL, tgt_band);
    Rng r = rng_new(opt_seed ^ 0xF17u, var);
    if (F.model == M_SIREN) {
        int dims[5] = {2, FH, FH, FH, 3};
        F.nl = 4;
        for (int l = 0; l < 4; l++) {
            dense_alloc(&F.L[l], dims[l], dims[l + 1]);
            dense_alloc(&F.S[l], dims[l], dims[l + 1]);
            F.omega[l] = 30;
            float a = l == 0 ? 1.0f / dims[l] : sqrtf(6.0f / dims[l]) / 30;
            dense_init(&F.L[l], &r, a, l == 3 ? 0 : 1 / sqrtf((float)dims[l]));
        }
        F.ntl = 3;
    } else {
        int dims[4] = {2 * FFM, FH, FH, 3};
        F.nl = 3;
        for (int k = 0; k < FFM * 2; k++) F.Bf[k] = 2.0f * rnd_n(&r);
        for (int l = 0; l < 3; l++) {
            dense_alloc(&F.L[l], dims[l], dims[l + 1]);
            dense_alloc(&F.S[l], dims[l], dims[l + 1]);
            float a = sqrtf(6.0f / dims[l]);
            dense_init(&F.L[l], &r, l == 2 ? a * 0.5f : a, 0.01f);
        }
        F.ntl = 2;
    }
    fit_rng = rng_new(opt_seed ^ 0xB47Cu, var);
    F.nsnap = 0;
    F.snap_step = 0;
    snap_take();
    snap_rows(0, IMG);
    snap_done();
    snap_done();                       /* both published slots hold the init */
    F.nsnap = 1;
    F.lr = fit_lr(0);
}

/* training step t (from 1) */
static void fit_step(int t) {
    F.lr = fit_lr(t);
    fit_batch(&fit_rng, BATCH1);
    F.loss[t] = fit_grad(BATCH1);
    for (int l = 0; l < F.nl; l++) { dense_adam(&F.L[l], F.lr, t, 0); dense_zero(&F.L[l]); }
    int j = t % EVAL_E;
    if (j == 0) { snap_take(); F.snap_step = t; }
    if (F.snap_step > 0) {
        snap_rows(j * EROWS, (j + 1) * EROWS);
        if (j == EVAL_E - 1) snap_done();
    }
}

static int fit_params(void) { int n = 0; for (int l = 0; l < F.nl; l++) n += dense_params(&F.L[l]); return n; }

/* ======================================================================
 * Act II: decision boundaries. A 2-12-12-12-1 tanh MLP on 2D points.
 * ====================================================================== */
#define NP2 320
#define H2 12
#define NL2 4
enum { DS_SPIRAL, DS_MOONS, DS_CHECKER, DS_CIRCLES, DS_XOR, NDS };
static const char *DS_NAME[NDS] = {"spirals", "moons", "checkerboard", "circles", "xor"};
static const int CLS_DIMS[NL2 + 1] = {2, H2, H2, H2, 1};
#define IN_SCALE 1.6f                  /* inputs are scaled by this into the net */

static struct {
    int ds, nsteps;
    float x[NP2 * 2], y[NP2];
    Dense L[NL2];
    float *a[NL2 + 1], *z[NL2], *g0, *g1;
    float *loss, *acc;
    float gm[NL2][H2 * H2];            /* recent |dL/dw| per weight, for the pulses */
    float lr;
} C;

static void cls_data(int ds, uint32_t salt) {
    Rng r = rng_new(opt_seed ^ 0xDA7Au, salt);
    for (int i = 0; i < NP2; i++) {
        float x = 0, y = 0;
        int c = i & 1;
        if (ds == DS_SPIRAL) {
            float t = (float)(i / 2) / (NP2 / 2);
            float rad = 0.07f + 0.86f * t, ang = t * 3.4f * PI + c * PI;
            x = rad * cosf(ang) + 0.025f * rnd_n(&r);
            y = rad * sinf(ang) + 0.025f * rnd_n(&r);
        } else if (ds == DS_MOONS) {
            float th = rnd_f(&r) * PI;
            if (!c) { x = cosf(th); y = sinf(th); }
            else { x = 1 - cosf(th); y = 0.5f - sinf(th); }
            x = (x - 0.5f) * 0.78f + 0.10f * rnd_n(&r);
            y = (y - 0.25f) * 0.95f + 0.10f * rnd_n(&r);
        } else if (ds == DS_CHECKER) {
            x = rnd_s(&r) * 0.95f; y = rnd_s(&r) * 0.95f;
            c = ((int)floorf((x + 1) * 2) + (int)floorf((y + 1) * 2)) & 1;
        } else if (ds == DS_CIRCLES) {
            float ang = rnd_f(&r) * TAU, rad = c ? 0.5f + 0.4f * rnd_f(&r) : 0.55f * sqrtf(rnd_f(&r));
            x = rad * cosf(ang); y = rad * sinf(ang);
        } else {
            do { x = rnd_s(&r) * 0.95f; y = rnd_s(&r) * 0.95f; } while (fabsf(x) < 0.06f || fabsf(y) < 0.06f);
            c = x * y > 0;
        }
        C.x[i * 2] = x; C.x[i * 2 + 1] = y; C.y[i] = (float)c;
    }
}

static void cls_forward(Dense *L, int B, const float *x) {
    for (int s = 0; s < B * 2; s++) C.a[0][s] = x[s] * IN_SCALE;
    for (int l = 0; l < NL2; l++) {
        dense_fwd(&L[l], C.a[l], B, C.z[l]);
        size_t n = (size_t)B * L[l].out;
        for (size_t i = 0; i < n; i++) C.a[l + 1][i] = l == NL2 - 1 ? C.z[l][i] : tanhf(C.z[l][i]);
    }
}

/* full-batch BCE (with logits): gradients accumulate; returns the loss and accuracy */
static float cls_grad(float *acc) {
    cls_forward(C.L, NP2, C.x);
    double loss = 0;
    float *g = C.g0;
    int ok = 0;
    for (int s = 0; s < NP2; s++) {
        float z = C.a[NL2][s], y = C.y[s];
        loss += fmaxf(z, 0) - z * y + log1pf(expf(-fabsf(z)));
        g[s] = (1 / (1 + expf(-z)) - y) / NP2;
        ok += (z > 0) == (y > 0.5f);
    }
    for (int l = NL2 - 1; l >= 0; l--) {
        size_t n = (size_t)NP2 * C.L[l].out;
        if (l < NL2 - 1) for (size_t i = 0; i < n; i++) g[i] *= 1 - C.a[l + 1][i] * C.a[l + 1][i];
        dense_bwd(&C.L[l], C.a[l], g, NP2, l > 0 ? C.g1 : NULL);
        float *sw = C.g0; C.g0 = C.g1; C.g1 = sw;
        g = C.g0;
    }
    if (acc) *acc = (float)ok / NP2;
    return (float)(loss / NP2);
}

static float cls_lr(int t) {
    float x = (float)t / C.nsteps;
    float base = C.ds == DS_SPIRAL ? 0.012f : C.ds == DS_CHECKER ? 0.004f : 0.003f;
    return base * (0.08f + 0.92f * 0.5f * (1 + cosf(PI * x)));
}

static void cls_init(uint32_t var, int phase) {
    for (int l = 0; l < NL2; l++) dense_free(&C.L[l]);
    free(C.loss); free(C.acc);
    if (!C.g0) {
        for (int l = 0; l <= NL2; l++) C.a[l] = malloc(NP2 * H2 * sizeof(float));
        for (int l = 0; l < NL2; l++) C.z[l] = malloc(NP2 * H2 * sizeof(float));
        C.g0 = malloc(NP2 * H2 * sizeof(float));
        C.g1 = malloc(NP2 * H2 * sizeof(float));
    }
    /* the first phase a hard set, the second an easier one */
    static const int hard[] = {DS_SPIRAL, DS_CHECKER}, easy[] = {DS_MOONS, DS_CIRCLES, DS_XOR};
    C.ds = phase == 0 ? hard[var % 2] : easy[(var >> 4) % 3];
    cls_data(C.ds, var * 2 + (uint32_t)phase);
    C.nsteps = phase_steps(A_CLS);
    C.loss = calloc((size_t)C.nsteps + 2, sizeof(float));
    C.acc = calloc((size_t)C.nsteps + 2, sizeof(float));
    Rng r = rng_new(opt_seed ^ 0xC1A5u, var * 2 + (uint32_t)phase);
    for (int l = 0; l < NL2; l++) {
        dense_alloc(&C.L[l], CLS_DIMS[l], CLS_DIMS[l + 1]);
        dense_init(&C.L[l], &r, sqrtf(6.0f / (CLS_DIMS[l] + CLS_DIMS[l + 1])), 0.1f);
    }
    memset(C.gm, 0, sizeof C.gm);
    C.loss[0] = cls_grad(&C.acc[0]);
    for (int l = 0; l < NL2; l++) dense_zero(&C.L[l]);
    C.lr = cls_lr(0);
}

static void cls_step(int t) {
    C.lr = cls_lr(t);
    C.loss[t] = cls_grad(&C.acc[t]);
    for (int l = 0; l < NL2; l++) {
        int n = C.L[l].in * C.L[l].out;
        for (int i = 0; i < n; i++) C.gm[l][i] = 0.9f * C.gm[l][i] + 0.1f * fabsf(C.L[l].gW[i]);
        dense_adam(&C.L[l], C.lr, t, 0);
        dense_zero(&C.L[l]);
    }
}

static int cls_params(void) { int n = 0; for (int l = 0; l < NL2; l++) n += dense_params(&C.L[l]); return n; }

/* ======================================================================
 * Act III: a character-level language model learns away-messages.
 * ====================================================================== */
#define CTX 8
#define EMB 12
#define HID 128
#define MAXV 96
#define BATCH3 64
#define SAMPLE_E 80                    /* steps between samples */
#define MAXSMP 256
#define SMPLEN 96

/* $N name, $H host, $U login (cf. stir's MSGS) */
static const char *CORPUS[] = {
    "This is $N's computer.",
    "$N is away from this computer.",
    "$H is locked.",
    "Back soon, probably.",
    "Pressing keys won't help, but it won't hurt either.",
    "$N is most likely by the kettle.",
    "If it's urgent, try Slack.",
    "$H is guarding the desk.",
    "Please do not feed the laptop.",
    "$N will be right back. Define 'right'.",
    "The cursor is resting. So is $N.",
    "Gone for a walk. Or a coffee. Or a meeting that should have been an email.",
    "Leave a note on paper. Paper still works.",
    "All the tabs are exactly where $N left them.",
    "Unattended computer. Attended desk plant.",
    "Out of office, but only by a few metres.",
    "The tea is probably going cold.",
    "$N is thinking. Possibly about lunch.",
    "No, you can't check your email on this one.",
    "Please hold. Your call is important to $H.",
    "The build is still running. Probably.",
    "Away from keyboard. Near a kettle.",
    "This computer belongs to $N. The chair is negotiable.",
    "Status: away. Mood: optimistic.",
    "$N will return shortly. Shortly is a relative term.",
    "Somewhere, $N has a perfectly good reason for being away.",
    "Nobody is typing. That is the point.",
    "Come back later. $H will still be here.",
    "The desk is unoccupied. The computer is not impressed.",
    "Messages will be answered in the order they are noticed.",
    "Gone fishing. Metaphorically.",
    "Back in a bit. A bit is somewhere between a minute and an hour.",
    "$N has left the building. The building is fine.",
    "If found, please return $N to $H.",
    "No unsaved work was harmed in the making of this screensaver.",
    "$N is AFK.",
    "Still away. Still locked. Still $H.",
    "Do not adjust your set. $N has merely stepped out.",
    "$H is keeping $N's seat warm. Literally: check the fans.",
    "There is nobody at this desk. There is a computer, and it is locked.",
    "The keyboard misses $N. The keyboard is like that.",
    "$N is in a meeting, at lunch, or lost in thought. Possibly all three.",
    "Knock knock. Who's there? Not $N.",
    "This screen will self-unlock when $N returns. Not before.",
    "$N is out. $H is in.",
    "Recalculating the whereabouts of $N.",
    "Away, but not far.",
    "The network is learning to write this. Slowly.",
    "$U@$H:~$ whoami  # not here",
    "$U@$H:~$ ping $N  # no reply, try again later",
    "$U@$H:~$ cat status  # away",
    "$U@$H:~$ sleep 60  # $N will be back after this",
};
#define NCORPUS ((int)(sizeof CORPUS / sizeof *CORPUS))

typedef struct {
    int step, len;
    float epoch, loss;
    char txt[SMPLEN + 1];
    unsigned char top[SMPLEN][4];      /* the four likeliest next characters at each position */
    float topp[SMPLEN][4];
} Sample;

static struct {
    char *text;                        /* the corpus, lines joined by \n */
    unsigned char *ids;
    int n, V, nsteps;
    char chars[MAXV];
    int lut[256];
    float E[MAXV * EMB], gE[MAXV * EMB], mE[MAXV * EMB], vE[MAXV * EMB];
    Dense L1, L2;
    float *x, *h, *p, *g, *gx;
    int *tg;
    float *loss, lr;
    Sample smp[MAXSMP];
    int nsmp;
    float e2d[MAXV][2], pc[2][EMB];
} T;
static char login[64];

static void char_corpus(void) {
    size_t cap = 16384, n = 0;
    T.text = malloc(cap);
    T.text[n++] = '\n';
    for (int i = 0; i < NCORPUS; i++) {
        for (const char *p = CORPUS[i]; *p && n < cap - 80; p++) {
            const char *sub = NULL;
            if (p[0] == '$' && p[1] == 'N') sub = opt_name;
            else if (p[0] == '$' && p[1] == 'H') sub = host[0] ? host : "localhost";
            else if (p[0] == '$' && p[1] == 'U') sub = login;
            if (sub) { for (; *sub && n < cap - 80; sub++) T.text[n++] = *sub; p++; }
            else T.text[n++] = *p;
        }
        T.text[n++] = '\n';
    }
    T.text[n] = 0;
    T.n = (int)n;
    T.ids = malloc(n);
    memset(T.lut, -1, sizeof T.lut);
    T.V = 0;
    /* the vocabulary in code order, so ids are stable */
    for (int c = 0; c < 256; c++)
        for (size_t i = 0; i < n; i++)
            if ((unsigned char)T.text[i] == c) {
                if (T.V < MAXV) { T.lut[c] = T.V; T.chars[T.V++] = (char)c; }
                break;
            }
    for (size_t i = 0; i < n; i++) T.ids[i] = (unsigned char)T.lut[(unsigned char)T.text[i]];
}

/* forward B contexts (ids ctx[B][CTX]): fills x, h, p (softmax) */
static void char_forward(const unsigned char *ctx, int B) {
    for (int s = 0; s < B; s++)
        for (int k = 0; k < CTX; k++)
            memcpy(T.x + (size_t)s * CTX * EMB + k * EMB, T.E + ctx[s * CTX + k] * EMB, EMB * sizeof(float));
    dense_fwd(&T.L1, T.x, B, T.h);
    for (int i = 0; i < B * HID; i++) T.h[i] = tanhf(T.h[i]);
    dense_fwd(&T.L2, T.h, B, T.p);
    for (int s = 0; s < B; s++) {
        float *q = T.p + (size_t)s * T.V, m = -1e30f, sum = 0;
        for (int v = 0; v < T.V; v++) m = fmaxf(m, q[v]);
        for (int v = 0; v < T.V; v++) sum += q[v] = expf(q[v] - m);
        for (int v = 0; v < T.V; v++) q[v] /= sum;
    }
}

static unsigned char char_ctx[BATCH3 * CTX];

static void char_batch(Rng *r) {
    for (int s = 0; s < BATCH3; s++) {
        int pos = 1 + (int)(rnd_u(r) % (uint32_t)(T.n - 1));
        /* each line on its own: the context before its start is newlines,
         * as when sampling, so samples don't all start like the first line */
        int ls = pos;
        while (ls > 0 && T.text[ls - 1] != '\n') ls--;
        for (int k = 0; k < CTX; k++) {
            int q = pos - CTX + k;
            char_ctx[s * CTX + k] = q < ls ? T.ids[0] : T.ids[q];
        }
        T.tg[s] = T.ids[pos];
    }
}

/* cross-entropy on the batch in char_ctx / tg; gradients accumulate */
static float char_grad(void) {
    const int B = BATCH3;
    char_forward(char_ctx, B);
    double loss = 0;
    for (int s = 0; s < B; s++) {
        float *q = T.p + (size_t)s * T.V;
        loss -= logf(fmaxf(q[T.tg[s]], 1e-12f));
        for (int v = 0; v < T.V; v++) T.g[(size_t)s * T.V + v] = (q[v] - (v == T.tg[s])) / B;
    }
    dense_bwd(&T.L2, T.h, T.g, B, T.gx);
    for (int i = 0; i < B * HID; i++) T.gx[i] *= 1 - T.h[i] * T.h[i];
    static float gin[BATCH3 * CTX * EMB];
    dense_bwd(&T.L1, T.x, T.gx, B, gin);
    for (int s = 0; s < B; s++)
        for (int k = 0; k < CTX; k++) {
            float *ge = T.gE + char_ctx[s * CTX + k] * EMB;
            const float *gi = gin + (size_t)s * CTX * EMB + k * EMB;
            for (int e = 0; e < EMB; e++) ge[e] += gi[e];
        }
    return (float)(loss / B);
}

static void char_pca(void) {
    float mean[EMB] = {0}, cov[EMB][EMB] = {{0}};
    for (int v = 0; v < T.V; v++) for (int e = 0; e < EMB; e++) mean[e] += T.E[v * EMB + e] / T.V;
    for (int v = 0; v < T.V; v++)
        for (int i = 0; i < EMB; i++)
            for (int j = 0; j < EMB; j++)
                cov[i][j] += (T.E[v * EMB + i] - mean[i]) * (T.E[v * EMB + j] - mean[j]);
    for (int k = 0; k < 2; k++) {
        float b[EMB];
        for (int e = 0; e < EMB; e++) b[e] = T.pc[k][e] != 0 ? T.pc[k][e] : 1.0f / (1 + e + k * 3);
        for (int it = 0; it < 60; it++) {
            float nb[EMB] = {0}, nn = 0;
            for (int i = 0; i < EMB; i++) for (int j = 0; j < EMB; j++) nb[i] += cov[i][j] * b[j];
            if (k == 1) {   /* deflate: keep it orthogonal to the first */
                float d = 0;
                for (int i = 0; i < EMB; i++) d += nb[i] * T.pc[0][i];
                for (int i = 0; i < EMB; i++) nb[i] -= d * T.pc[0][i];
            }
            for (int i = 0; i < EMB; i++) nn += nb[i] * nb[i];
            nn = sqrtf(fmaxf(nn, 1e-20f));
            for (int i = 0; i < EMB; i++) b[i] = nb[i] / nn;
        }
        float d = 0;   /* keep the sign of the previous axis, so the plot doesn't flip */
        for (int i = 0; i < EMB; i++) d += b[i] * T.pc[k][i];
        for (int i = 0; i < EMB; i++) T.pc[k][i] = d < 0 ? -b[i] : b[i];
    }
    float sc = 1e-6f;
    for (int v = 0; v < T.V; v++)
        for (int k = 0; k < 2; k++) {
            float d = 0;
            for (int e = 0; e < EMB; e++) d += (T.E[v * EMB + e] - mean[e]) * T.pc[k][e];
            T.e2d[v][k] = d;
            sc = fmaxf(sc, fabsf(d));
        }
    for (int v = 0; v < T.V; v++) { T.e2d[v][0] /= sc; T.e2d[v][1] /= sc; }
}

/* a sample from the current net, as of step t */
static void char_sample(int t) {
    Sample *S = &T.smp[T.nsmp < MAXSMP ? T.nsmp++ : MAXSMP - 1];
    Rng r = rng_new(opt_seed ^ 0x5A3Bu, (uint32_t)t);
    unsigned char ctx[CTX];
    memset(ctx, T.ids[0], CTX);
    S->step = t;
    S->epoch = (float)t * BATCH3 / T.n;
    S->loss = T.loss[t];
    float temp = 0.75f;
    int n = 0;
    for (; n < SMPLEN; n++) {
        char_forward(ctx, 1);
        float q[MAXV], sum = 0;
        for (int v = 0; v < T.V; v++) sum += q[v] = powf(T.p[v], 1 / temp);
        float u = rnd_f(&r) * sum;
        int pick = T.V - 1;
        for (int v = 0; v < T.V; v++) { u -= q[v]; if (u <= 0) { pick = v; break; } }
        /* the top four, by the untempered distribution */
        for (int k = 0; k < 4; k++) {
            int best = 0;
            float bp = -1;
            for (int v = 0; v < T.V; v++) {
                int used = 0;
                for (int j = 0; j < k; j++) used |= S->top[n][j] == v;
                if (!used && T.p[v] > bp) { bp = T.p[v]; best = v; }
            }
            S->top[n][k] = (unsigned char)best;
            S->topp[n][k] = bp;
        }
        if (T.chars[pick] == '\n') break;
        S->txt[n] = T.chars[pick];
        memmove(ctx, ctx + 1, CTX - 1);
        ctx[CTX - 1] = (unsigned char)pick;
    }
    S->txt[n] = 0;
    S->len = n;
    char_pca();
}

static Rng char_rng;

static float char_lr(int t) {
    float x = (float)t / T.nsteps;
    return 3e-3f * (0.1f + 0.9f * 0.5f * (1 + cosf(PI * x)));
}

static void char_init(uint32_t var) {
    if (!T.text) {
        char_corpus();
        T.x = malloc(sizeof(float) * BATCH3 * CTX * EMB);
        T.h = malloc(sizeof(float) * BATCH3 * HID);
        T.gx = malloc(sizeof(float) * BATCH3 * HID);
        T.p = malloc(sizeof(float) * BATCH3 * MAXV);
        T.g = malloc(sizeof(float) * BATCH3 * MAXV);
        T.tg = malloc(sizeof(int) * BATCH3);
    }
    dense_free(&T.L1); dense_free(&T.L2);
    free(T.loss);
    T.nsteps = phase_steps(A_CHAR);
    T.loss = calloc((size_t)T.nsteps + 2, sizeof(float));
    Rng r = rng_new(opt_seed ^ 0xC4A2u, var);
    for (int i = 0; i < T.V * EMB; i++) T.E[i] = rnd_n(&r) * 0.5f;
    memset(T.gE, 0, sizeof T.gE); memset(T.mE, 0, sizeof T.mE); memset(T.vE, 0, sizeof T.vE);
    dense_alloc(&T.L1, CTX * EMB, HID);
    dense_alloc(&T.L2, HID, T.V);
    dense_init(&T.L1, &r, sqrtf(3.0f / (CTX * EMB)), 0.01f);
    dense_init(&T.L2, &r, sqrtf(3.0f / HID) * 0.5f, 0);
    memset(T.pc, 0, sizeof T.pc);
    char_rng = rng_new(opt_seed ^ 0x7E47u, var);
    T.loss[0] = logf((float)T.V);
    T.nsmp = 0;
    char_sample(0);
    T.lr = char_lr(0);
}

static void char_step(int t) {
    T.lr = char_lr(t);
    char_batch(&char_rng);
    T.loss[t] = char_grad();
    dense_adam(&T.L1, T.lr, t, 0); dense_zero(&T.L1);
    dense_adam(&T.L2, T.lr, t, 0); dense_zero(&T.L2);
    adam_vec(T.E, T.gE, T.mE, T.vE, (size_t)T.V * EMB, T.lr, t, 0);
    memset(T.gE, 0, sizeof T.gE);
    if (t % SAMPLE_E == 0) char_sample(t);
}

static int char_params(void) { return T.V * EMB + dense_params(&T.L1) + dense_params(&T.L2); }

/* ======================================================================
 * Act IV: optimisers racing on test surfaces.
 * ====================================================================== */
enum { FN_ROSEN, FN_RASTRIGIN, FN_HIMMEL, NFN };
static const char *FN_NAME[NFN] = {"Rosenbrock", "Rastrigin", "Himmelblau"};
static const char *OPT_NAME[4] = {"SGD", "Momentum", "RMSProp", "Adam"};
#define NOPT 4

static struct {
    int fn, nsteps;
    double x[NOPT][2], m[NOPT][2], v[NOPT][2], lr[NOPT];
    float *tr[NOPT];                   /* the path: (nsteps + 1) x 2 */
    float *lo[NOPT];                   /* f along it */
    float dom[4];                      /* x0 y0 x1 y1 */
    float fmax;
    float noise;
    Rng rng;
} O;

#define RA 4.0                         /* Rastrigin's A: low enough bumps that momentum can roll over some */
static double fn_f(int fn, double x, double y, double *gx, double *gy) {
    if (fn == FN_ROSEN) {
        double a = 1 - x, b = y - x * x;
        if (gx) { *gx = -2 * a - 400 * x * b; *gy = 200 * b; }
        return a * a + 100 * b * b;
    }
    if (fn == FN_RASTRIGIN) {
        double w = 2 * M_PI;
        if (gx) { *gx = 2 * x + RA * w * sin(w * x); *gy = 2 * y + RA * w * sin(w * y); }
        return 2 * RA + x * x - RA * cos(w * x) + y * y - RA * cos(w * y);
    }
    double a = x * x + y - 11, b = x + y * y - 7;
    if (gx) { *gx = 4 * a * x + 2 * b; *gy = 2 * a + 4 * b * y; }
    return a * a + b * b;
}

static void opt_init(uint32_t var, int phase) {
    for (int k = 0; k < NOPT; k++) { free(O.tr[k]); free(O.lo[k]); }
    O.fn = phase == 0 ? FN_ROSEN : (var >> 5) % 2 ? FN_RASTRIGIN : FN_HIMMEL;
    O.nsteps = phase_steps(A_OPT);
    Rng r = rng_new(opt_seed ^ 0x0971u, var * 2 + (uint32_t)phase);
    double sx, sy;
    if (O.fn == FN_ROSEN) {
        O.dom[0] = -2; O.dom[1] = -1; O.dom[2] = 2; O.dom[3] = 3;
        sx = -1.6 + 0.3 * rnd_s(&r); sy = 2.4 + 0.3 * rnd_s(&r);
        double lr[NOPT] = {4e-4, 1.2e-4, 6e-3, 4e-2};
        memcpy(O.lr, lr, sizeof lr);
    } else if (O.fn == FN_RASTRIGIN) {
        O.dom[0] = -5.12f; O.dom[1] = -5.12f; O.dom[2] = 5.12f; O.dom[3] = 5.12f;
        sx = 4.3 + 0.15 * rnd_s(&r); sy = 4.6 + 0.15 * rnd_s(&r);
        double lr[NOPT] = {2e-3, 2e-3, 0.12, 0.4};
        memcpy(O.lr, lr, sizeof lr);
    } else {
        O.dom[0] = -5; O.dom[1] = -5; O.dom[2] = 5; O.dom[3] = 5;
        sx = -0.27 + 0.25 * rnd_s(&r); sy = -0.92 + 0.25 * rnd_s(&r);
        double lr[NOPT] = {4e-4, 1e-4, 5e-3, 0.03};
        memcpy(O.lr, lr, sizeof lr);
    }
    O.noise = O.fn == FN_RASTRIGIN ? 8 : 0;   /* minibatch-like noise in the gradient */
    O.rng = rng_new(opt_seed ^ 0x401Eu, var * 2 + (uint32_t)phase);
    O.fmax = 0;
    for (int j = 0; j <= 64; j++)
        for (int i = 0; i <= 64; i++) {
            double x = O.dom[0] + (O.dom[2] - O.dom[0]) * i / 64, y = O.dom[1] + (O.dom[3] - O.dom[1]) * j / 64;
            O.fmax = fmaxf(O.fmax, (float)fn_f(O.fn, x, y, NULL, NULL));
        }
    for (int k = 0; k < NOPT; k++) {
        O.tr[k] = malloc(sizeof(float) * 2 * ((size_t)O.nsteps + 2));
        O.lo[k] = malloc(sizeof(float) * ((size_t)O.nsteps + 2));
        O.x[k][0] = sx; O.x[k][1] = sy;
        O.m[k][0] = O.m[k][1] = O.v[k][0] = O.v[k][1] = 0;
        O.tr[k][0] = (float)sx; O.tr[k][1] = (float)sy;
        O.lo[k][0] = (float)fn_f(O.fn, sx, sy, NULL, NULL);
    }
}

static void opt_step(int t) {
    for (int k = 0; k < NOPT; k++) {
        double g[2];
        fn_f(O.fn, O.x[k][0], O.x[k][1], &g[0], &g[1]);
        if (O.noise > 0) { g[0] += O.noise * rnd_n(&O.rng); g[1] += O.noise * rnd_n(&O.rng); }
        for (int d = 0; d < 2; d++) {
            double step;
            if (k == 0) step = O.lr[k] * g[d];
            else if (k == 1) { O.m[k][d] = 0.9 * O.m[k][d] + g[d]; step = O.lr[k] * O.m[k][d]; }
            else if (k == 2) { O.v[k][d] = 0.9 * O.v[k][d] + 0.1 * g[d] * g[d]; step = O.lr[k] * g[d] / (sqrt(O.v[k][d]) + 1e-8); }
            else {
                O.m[k][d] = 0.9 * O.m[k][d] + 0.1 * g[d];
                O.v[k][d] = 0.999 * O.v[k][d] + 0.001 * g[d] * g[d];
                double mh = O.m[k][d] / (1 - pow(0.9, t)), vh = O.v[k][d] / (1 - pow(0.999, t));
                step = O.lr[k] * mh / (sqrt(vh) + 1e-8);
            }
            if (step > 0.5) step = 0.5;
            if (step < -0.5) step = -0.5;
            O.x[k][d] -= step;
        }
        /* walls: stay on the map */
        O.x[k][0] = fmin(fmax(O.x[k][0], O.dom[0]), O.dom[2]);
        O.x[k][1] = fmin(fmax(O.x[k][1], O.dom[1]), O.dom[3]);
        O.tr[k][t * 2] = (float)O.x[k][0];
        O.tr[k][t * 2 + 1] = (float)O.x[k][1];
        O.lo[k][t] = (float)fn_f(O.fn, O.x[k][0], O.x[k][1], NULL, NULL);
    }
}

/* ======================================================================
 * The schedule: set up the run for the act and phase on now, and step it
 * ====================================================================== */
static void run_setup(long cycle, int act, int phase) {
    run.cycle = cycle; run.act = act; run.phase = phase;
    run.variant = (int)(hash32(opt_seed * 0x9E3779B9u + (uint32_t)cycle * 0x85EBCA6Bu + 17) & 0x7fffffff);
    run.step = 0;
    run.live = 1;
    run.wt_act0 = t_wall0 + cycle * act_len;
    uint32_t var = (uint32_t)run.variant;
    if (act == A_FIT) {
        time_t ts = (time_t)(run.wt_act0 + T_TRAIN);
        struct tm tm;
        localtime_r(&ts, &tm);
        fit_init(var, &tm);
    } else if (act == A_CLS) cls_init(var, phase);
    else if (act == A_CHAR) char_init(var);
    else opt_init(var, phase);
}

static void run_step(void) {
    int t = ++run.step;
    if (run.act == A_FIT) fit_step(t);
    else if (run.act == A_CLS) cls_step(t);
    else if (run.act == A_CHAR) char_step(t);
    else opt_step(t);
}

static int cur_act(long cycle) {
    if (opt_act >= 0) return opt_act;
    return (int)((cycle + (long)(hash32(opt_seed) % NACT)) % NACT);
}

static double step_frac;               /* fractional steps due (for smooth drawing) */

/* bring training up to e seconds after the epoch; budget_ms < 0: no limit */
static void advance(double e, double budget_ms) {
    long cycle = (long)floor(e / act_len);
    int act = cur_act(cycle);
    double lt = e - cycle * act_len;
    int ph = 0;
    if (lt > T_TRAIN) ph = (int)((lt - T_TRAIN) / phase_len(act));
    if (ph >= nphase(act)) ph = nphase(act) - 1;
    if (!run.live || run.cycle != cycle || run.phase != ph || run.act != act) run_setup(cycle, act, ph);
    run.t_act = lt;
    double pt = lt - T_TRAIN - ph * phase_len(act);
    double due = pt * SPS[act];
    int max = phase_steps(act);
    if (due > max) due = max;
    if (due < 0) due = 0;
    double c0 = now();
    while (run.step < (int)due) {
        run_step();
        if (budget_ms >= 0 && (now() - c0) * 1000 > budget_ms) break;
    }
    step_frac = fmin(due, run.step + 1.0);
}

/* ======================================================================
 * GL
 * ====================================================================== */
#define GLSL(...) #__VA_ARGS__

/* the background: a soft vignette and a faint dot grid */
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform vec3 uBg, uGrid;
    uniform float uU, uLight;
    out vec4 o;
    void main() {
        vec2 q = gl_FragCoord.xy / uRes - 0.5;
        float v = 1.0 - dot(q, q) * (uLight > 0.5 ? 0.25 : 0.6);
        vec3 c = uBg * (uLight > 0.5 ? mix(0.97, 1.01, v) : mix(0.75, 1.12, v));
        float s = 28.0 * uU;
        vec2 g = abs(mod(gl_FragCoord.xy, s) - s * 0.5);
        float dot_ = clamp(1.2 * uU - length(g), 0.0, 1.0);
        c = mix(c, uGrid, dot_ * 0.55);
        o = vec4(c, 1.0);
    });

/* shapes: antialiased capsules, rounded rects and circles from their SDFs */
static const char *VS_SHAPE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aLoc;
    layout(location = 2) in vec4 aCol;
    layout(location = 3) in vec4 aPar;
    uniform vec2 uRes;
    out vec2 vLoc;
    out vec4 vCol;
    flat out vec4 vPar;
    void main() {
        vLoc = aLoc; vCol = aCol; vPar = aPar;
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_SHAPE = "#version 330 core\n" GLSL(
    in vec2 vLoc;
    in vec4 vCol;
    flat in vec4 vPar;
    out vec4 o;
    void main() {
        float d;
        if (vPar.x < 0.5) {
            vec2 q = abs(vLoc) - vPar.yz + vPar.w;
            d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - vPar.w;
        } else if (vPar.x < 1.5) {
            d = length(vec2(max(abs(vLoc.x) - vPar.y, 0.0), vLoc.y)) - vPar.z;
        } else {
            d = length(vLoc) - vPar.y;
            if (vPar.w > 0.0) d = abs(d + vPar.w * 0.5) - vPar.w * 0.5;
            if (vPar.z > 0.0) { o = vec4(vCol.rgb, vCol.a * exp(-max(d + vPar.y, 0.0) * vPar.z)); return; }
        }
        o = vec4(vCol.rgb, vCol.a * clamp(0.5 - d, 0.0, 1.0));
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

/* a rect on screen (px, y down) with uv across it */
static const char *VS_RECT = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform vec4 uRect;
    out vec2 vUV;
    void main() {
        vec2 c[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));
        vec2 q = c[gl_VertexID];
        vUV = q;
        vec2 p = mix(uRect.xy, uRect.zw, q);
        gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
    });

#define CMAPS GLSL( \
    uniform vec3 uNeg, uMid, uPos; \
    uniform vec3 uSeq[5]; \
    vec3 seqc(float t) { \
        t = clamp(t, 0.0, 1.0) * 4.0; \
        int i = int(min(floor(t), 3.0)); \
        return mix(uSeq[i], uSeq[i + 1], t - float(i)); \
    } \
    vec3 divc(float v) { \
        v = clamp(v, -1.0, 1.0); \
        return v > 0.0 ? mix(uMid, uPos, v) : mix(uMid, uNeg, -v); \
    })

/* images: a picture (mode 0), a signed map (1) or a magnitude map (2) */
static const char *FS_IMG = "#version 330 core\n" CMAPS GLSL(
    uniform sampler2D uA, uB;
    uniform float uMix, uGain, uOff, uAlpha;
    uniform int uMode, uImg;
    uniform vec4 uSub;
    in vec2 vUV;
    out vec4 o;
    void main() {
        vec2 uv = mix(uSub.xy, uSub.zw, vUV);
        vec4 a = texture(uA, uv);
        if (uMix > 0.0) a = mix(a, texture(uB, uv), uMix);
        vec3 c;
        if (uMode == 0) {
            c = clamp(a.rgb, 0.0, 1.0);
            float l = dot(c, vec3(0.299, 0.587, 0.114));
            if (uImg == 1) c = vec3(l);
            else if (uImg == 2) c = seqc(l * 0.95 + 0.05);
        } else if (uMode == 1) c = divc(a.r * uGain);
        else c = seqc(a.r * uGain + uOff);
        o = vec4(c, uAlpha);
    });

/* Act II: the classifier evaluated per pixel: its output (unit < 0) or one
 * unit's activation map (layer 0 is the inputs) */
static const char *FS_FIELD = "#version 330 core\n" CMAPS GLSL(
    uniform float uW1[24], uB1[12], uW2[144], uB2[12], uW3[144], uB3[12], uW4[12], uB4;
    uniform vec4 uDom;
    uniform int uLayer, uUnit;
    uniform vec3 uC0, uC1, uBg, uFg;
    uniform float uScale, uAlpha, uPx;
    in vec2 vUV;
    out vec4 o;
    void main() {
        vec2 p = mix(uDom.xy, uDom.zw, vec2(vUV.x, 1.0 - vUV.y));
        float x0 = p.x * uScale, x1 = p.y * uScale;
        float h1[12], h2[12], h3[12];
        float v = 0.0;
        float z = uB4;
        /* a unit's map only needs the layers up to it (uniform branches) */
        if (uLayer == 0) v = uUnit == 0 ? p.x : p.y;
        else if (uLayer == 1) v = tanh(uB1[uUnit] + x0 * uW1[uUnit] + x1 * uW1[12 + uUnit]);
        else {
            for (int i = 0; i < 12; i++) h1[i] = tanh(uB1[i] + x0 * uW1[i] + x1 * uW1[12 + i]);
            if (uLayer == 2) {
                float s = uB2[uUnit];
                for (int i = 0; i < 12; i++) s += h1[i] * uW2[i * 12 + uUnit];
                v = tanh(s);
            } else {
                for (int j = 0; j < 12; j++) {
                    float s = uB2[j];
                    for (int i = 0; i < 12; i++) s += h1[i] * uW2[i * 12 + j];
                    h2[j] = tanh(s);
                }
                if (uLayer == 3) {
                    float s = uB3[uUnit];
                    for (int i = 0; i < 12; i++) s += h2[i] * uW3[i * 12 + uUnit];
                    v = tanh(s);
                } else {
                    for (int j = 0; j < 12; j++) {
                        float s = uB3[j];
                        for (int i = 0; i < 12; i++) s += h2[i] * uW3[i * 12 + j];
                        h3[j] = tanh(s);
                    }
                    for (int i = 0; i < 12; i++) z += h3[i] * uW4[i];
                }
            }
        }
        vec3 c;
        if (uLayer >= 0) c = divc(v);
        else {
            float pr = 1.0 / (1.0 + exp(-z));
            float k = abs(pr - 0.5) * 2.0;
            c = mix(uBg, mix(uC0, uC1, step(0.5, pr)), 0.16 + 0.42 * k * k);
            float fw = max(fwidth(z) * uPx, 1e-4);   /* line widths in logical px */
            float iso = abs(fract(z * 0.5 + 0.5) - 0.5) * 2.0 / (fw * 0.5);
            c = mix(c, uFg, 0.10 * clamp(1.0 - iso, 0.0, 1.0));
            c = mix(c, uFg, 0.85 * clamp(1.6 - abs(z) / fw, 0.0, 1.0));
        }
        o = vec4(c, uAlpha);
    });

/* Act IV: the loss surface, a grid mesh lifted in the vertex shader */
#define SURF_FN GLSL( \
    uniform int uFn; \
    uniform float uFmax; \
    float fnf(vec2 p) { \
        if (uFn == 0) { float a = 1.0 - p.x, b = p.y - p.x * p.x; return a * a + 100.0 * b * b; } \
        if (uFn == 1) return 8.0 + p.x * p.x - 4.0 * cos(6.2831853 * p.x) + p.y * p.y - 4.0 * cos(6.2831853 * p.y); \
        float a = p.x * p.x + p.y - 11.0, b = p.x + p.y * p.y - 7.0; \
        return a * a + b * b; \
    } \
    float hgt(vec2 p) { return log(1.0 + fnf(p)) / log(1.0 + uFmax); })
static const char *VS_SURF = "#version 330 core\n" SURF_FN GLSL(
    uniform int uN;
    uniform mat4 uMVP;
    uniform vec4 uDom;
    uniform float uHs;
    out vec3 vN;
    out float vH;
    out vec2 vP;
    void main() {
        int cell = gl_VertexID / 6, k = gl_VertexID - cell * 6;
        ivec2 off[6] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));
        ivec2 c = ivec2(cell % (uN - 1), cell / (uN - 1)) + off[k];
        vec2 q = vec2(c) / float(uN - 1);
        vec2 p = mix(uDom.xy, uDom.zw, q);
        float h = hgt(p);
        vec2 e = (uDom.zw - uDom.xy) / float(uN - 1);
        float hx = hgt(p + vec2(e.x, 0.0)) - hgt(p - vec2(e.x, 0.0));
        float hy = hgt(p + vec2(0.0, e.y)) - hgt(p - vec2(0.0, e.y));
        float dq = 2.0 / float(uN - 1);
        vN = normalize(vec3(-hx * uHs / (2.0 * dq), 1.0, hy * uHs / (2.0 * dq)));
        vH = h;
        vP = p;
        gl_Position = uMVP * vec4(q.x * 2.0 - 1.0, h * uHs, 1.0 - q.y * 2.0, 1.0);
    });
static const char *FS_SURF = "#version 330 core\n" CMAPS GLSL(
    uniform vec3 uFg, uBg;
    uniform float uTop, uPx;
    in vec3 vN;
    in float vH;
    in vec2 vP;
    out vec4 o;
    void main() {
        vec3 c = seqc(0.12 + vH * 0.85);
        float l = uTop > 0.5 ? 1.0 : 0.55 + 0.45 * max(dot(normalize(vN), normalize(vec3(-0.4, 0.8, 0.45))), 0.0);
        c *= l;
        float lv = vH * 16.0;
        float d = abs(fract(lv) - 0.5) * 2.0;
        float fw = max(fwidth(lv) * 2.0 * uPx, 1e-4);
        c = mix(c, uFg, 0.28 * clamp((d - (1.0 - fw)) / fw, 0.0, 1.0));
        vec2 g = abs(fract(vP + 0.5) - 0.5) * 2.0;
        vec2 gw = max(fwidth(vP) * 2.0 * uPx, 1e-4);
        float gl = max(clamp((g.x - (1.0 - gw.x)) / gw.x, 0.0, 1.0), clamp((g.y - (1.0 - gw.y)) / gw.y, 0.0, 1.0));
        c = mix(c, uBg, 0.25 * gl);
        o = vec4(c, 1.0);
    });

static GLuint prog_bg, prog_shape, prog_text, prog_img, prog_field, prog_surf;
static GLuint vao, vao_shape, vbo_shape, vao_text, vbo_text;
static GLuint tex_atlas, tex_tgt, tex_pred[2], tex_tiles, tex_spec, tex_w3;
static int up_ver = -1, up_cycle = -1;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "neural: shader:\n%s\n", log);
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

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, int nearest) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static void set_cmaps(GLuint p) {
    glUseProgram(p);
    glUniform3fv(U(p, "uNeg"), 1, P.neg);
    glUniform3fv(U(p, "uMid"), 1, P.mid);
    glUniform3fv(U(p, "uPos"), 1, P.pos);
    float s[5][3];
    for (int i = 0; i < 5; i++) memcpy(s[i], P.seq[i], sizeof s[i]);
    glUniform3fv(U(p, "uSeq"), 5, &s[0][0]);
}

static void gl_init(void) {
    prog_bg = mk_prog(VS_FULL, FS_BG);
    prog_shape = mk_prog(VS_SHAPE, FS_SHAPE);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    prog_img = mk_prog(VS_RECT, FS_IMG);
    prog_field = mk_prog(VS_RECT, FS_FIELD);
    prog_surf = mk_prog(VS_SURF, FS_SURF);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_shape);
    glGenBuffers(1, &vbo_shape);
    glBindVertexArray(vao_shape);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_shape);
    for (int i = 0; i < 4; i++) {
        static const int sz[4] = {2, 2, 4, 4}, off[4] = {0, 2, 4, 8};
        glVertexAttribPointer((GLuint)i, sz[i], GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void *)(off[i] * sizeof(float)));
        glEnableVertexAttribArray((GLuint)i);
    }
    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glGenTextures(1, &tex_atlas);
    tex_tgt = mk_tex(GL_RGB32F, IMG, IMG, GL_RGB, 1);
    tex_pred[0] = mk_tex(GL_RGB32F, IMG, IMG, GL_RGB, 1);
    tex_pred[1] = mk_tex(GL_RGB32F, IMG, IMG, GL_RGB, 1);
    tex_tiles = mk_tex(GL_R32F, 8 * TILE, 3 * 8 * TILE, GL_RED, 0);
    tex_spec = mk_tex(GL_R32F, MAXSNAP, NBAND, GL_RED, 1);
    tex_w3 = mk_tex(GL_R32F, MAXV, HID, GL_RED, 1);
    set_cmaps(prog_img);
    set_cmaps(prog_field);
    set_cmaps(prog_surf);
}

/* ---- the shape batch: 12 floats a vertex ---- */
static float *sv;
static int svn, svcap;

static void shape_quad(float cx, float cy, float ux, float uy, float hx, float hy, const float *col, const float *par) {
    /* a quad centred at c with axes u (unit) and v = perp(u), half extents hx, hy (incl. the AA pad) */
    if (svn + 72 > svcap) sv = realloc(sv, (size_t)(svcap = svcap ? svcap * 2 : 65536) * sizeof *sv);
    float vx = -uy, vy = ux;
    static const float cs[6][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, -1}, {1, 1}, {-1, 1}};
    for (int k = 0; k < 6; k++) {
        float lx = cs[k][0] * hx, ly = cs[k][1] * hy;
        float *o = sv + svn;
        o[0] = cx + ux * lx + vx * ly; o[1] = cy + uy * lx + vy * ly;
        o[2] = lx; o[3] = ly;
        memcpy(o + 4, col, 4 * sizeof(float));
        memcpy(o + 8, par, 4 * sizeof(float));
        svn += 12;
    }
}

/* a capsule from a to b, w px wide */
static void seg(float ax, float ay, float bx, float by, float w, const float *col) {
    float dx = bx - ax, dy = by - ay, l = sqrtf(dx * dx + dy * dy);
    float ux = 1, uy = 0;
    if (l > 1e-4f) { ux = dx / l; uy = dy / l; }
    float hw = w * 0.5f, a = 1;
    if (hw < 0.5f) { a = hw / 0.5f; hw = 0.5f; }
    float c[4] = {col[0], col[1], col[2], col[3] * a};
    float par[4] = {1, l * 0.5f, hw, 0};
    shape_quad((ax + bx) * 0.5f, (ay + by) * 0.5f, ux, uy, l * 0.5f + hw + 1, hw + 1, c, par);
}

static void rect_fill(float x0, float y0, float x1, float y1, float rad, const float *col) {
    float par[4] = {0, (x1 - x0) * 0.5f, (y1 - y0) * 0.5f, rad};
    shape_quad((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, 1, 0, (x1 - x0) * 0.5f + 1, (y1 - y0) * 0.5f + 1, col, par);
}

static void circle(float x, float y, float r, const float *col) {
    float par[4] = {2, r, 0, 0};
    shape_quad(x, y, 1, 0, r + 1, r + 1, col, par);
}

static void ring(float x, float y, float r, float th, const float *col) {
    float par[4] = {2, r, 0, th};
    shape_quad(x, y, 1, 0, r + 1, r + 1, col, par);
}

/* a soft glow of radius r (falls off as exp(-d * k)) */
static void glow(float x, float y, float r, const float *col) {
    float par[4] = {2, r * 0.25f, 4.0f / r, 0};
    shape_quad(x, y, 1, 0, r * 1.3f, r * 1.3f, col, par);
}

static void rect_line(float x0, float y0, float x1, float y1, float w, const float *col) {
    seg(x0, y0, x1, y0, w, col); seg(x1, y0, x1, y1, w, col);
    seg(x1, y1, x0, y1, w, col); seg(x0, y1, x0, y0, w, col);
}

static void flush_shapes(void) {
    if (!svn) return;
    glUseProgram(prog_shape);
    glUniform2f(U(prog_shape, "uRes"), (float)W, (float)H);
    glBindVertexArray(vao_shape);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_shape);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(svn * sizeof(float)), sv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, svn / 12);
    svn = 0;
}

static void flush_text(void) {
    if (!tvn) return;
    glUseProgram(prog_text);
    glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_text, "uAtlas"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    tvn = 0;
}

static void flush(void) { flush_shapes(); flush_text(); }

/* a texture into a screen rect: mode 0 picture, 1 signed, 2 magnitude */
static void draw_tex(float x0, float y0, float x1, float y1, GLuint a, GLuint b, float mix, int mode, float gain,
                     float off, const float *sub) {
    flush();
    GLuint p = prog_img;
    glUseProgram(p);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform4f(U(p, "uRect"), x0, y0, x1, y1);
    static const float full[4] = {0, 0, 1, 1};
    glUniform4fv(U(p, "uSub"), 1, sub ? sub : full);
    glUniform1f(U(p, "uMix"), mix);
    glUniform1f(U(p, "uGain"), gain);
    glUniform1f(U(p, "uOff"), off);
    glUniform1f(U(p, "uAlpha"), 1);
    glUniform1i(U(p, "uMode"), mode);
    glUniform1i(U(p, "uImg"), P.img_mode);
    glUniform1i(U(p, "uA"), 0);
    glUniform1i(U(p, "uB"), 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, b ? b : a);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, a);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

/* the act's data to the GPU when it changes */
static void upload(void) {
    if (run.act == A_FIT) {
        if (up_cycle != (int)run.cycle) {
            glBindTexture(GL_TEXTURE_2D, tex_tgt);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMG, IMG, GL_RGB, GL_FLOAT, F.tgt);
            up_ver = -1;
        }
        if (up_ver != F.pub_ver) {
            for (int i = 0; i < 2; i++) {
                glBindTexture(GL_TEXTURE_2D, tex_pred[i]);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMG, IMG, GL_RGB, GL_FLOAT, F.pub[i]);
            }
            static float tl[3 * 8 * TILE][8 * TILE];
            for (int l = 0; l < 3; l++)
                for (int u = 0; u < FH; u++)
                    for (int y = 0; y < TILE; y++)
                        for (int x = 0; x < TILE; x++)
                            tl[l * 8 * TILE + (u / 8) * TILE + y][(u % 8) * TILE + x] = l < F.ntl ? F.tiles[l][u][y * TILE + x] : 0;
            glBindTexture(GL_TEXTURE_2D, tex_tiles);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8 * TILE, 3 * 8 * TILE, GL_RED, GL_FLOAT, tl);
            static float sp[NBAND][MAXSNAP];
            for (int b = 0; b < NBAND; b++)
                for (int k = 0; k < MAXSNAP; k++) sp[b][k] = k < F.nsnap ? F.spec[k][b] : 0;
            glBindTexture(GL_TEXTURE_2D, tex_spec);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MAXSNAP, NBAND, GL_RED, GL_FLOAT, sp);
            up_ver = F.pub_ver;
        }
    } else if (run.act == A_CHAR) {
        static int last = -1;
        if (last != run.step || up_cycle != (int)run.cycle) {
            static float w[HID][MAXV];
            for (int h = 0; h < HID; h++)
                for (int v = 0; v < MAXV; v++) w[h][v] = v < T.V ? T.L2.W[h * T.V + v] : 0;
            glBindTexture(GL_TEXTURE_2D, tex_w3);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MAXV, HID, GL_RED, GL_FLOAT, w);
            last = run.step;
        }
    }
    up_cycle = (int)run.cycle;
}

/* ======================================================================
 * Drawing: panels, plots, the four acts, and the frame around them
 * ====================================================================== */
typedef struct { float x0, y0, x1, y1; } Rect;
static float rw(Rect r) { return r.x1 - r.x0; }
static float rh(Rect r) { return r.y1 - r.y0; }
static Rect rect(float x0, float y0, float x1, float y1) { Rect r = {x0, y0, x1, y1}; return r; }
/* split off the left (or top) fraction f, with a gap */
static void hsplit(Rect r, float f, float gap, Rect *a, Rect *b) {
    float x = r.x0 + (rw(r) - gap) * f;
    *a = rect(r.x0, r.y0, x, r.y1);
    *b = rect(x + gap, r.y0, r.x1, r.y1);
}
static void vsplit(Rect r, float f, float gap, Rect *a, Rect *b) {
    float y = r.y0 + (rh(r) - gap) * f;
    *a = rect(r.x0, r.y0, r.x1, y);
    *b = rect(r.x0, y + gap, r.x1, r.y1);
}

static float Uu = 1;                   /* a px unit: the short side / 1000 */
static int st_label, st_tiny, st_small, st_body, st_mono, st_monob, st_title, st_card, st_cardk, st_time;
static int styled_w, styled_h;
static double draw_t;                  /* seconds into the act, for animation */

#define SEP " \x7f "

static void layout_styles(void) {
    nstyles = 0;
    /* from the logical size, so a scale-2 output gets the same layout at twice the pixels */
    Uu = fmaxf(fminf((float)W, (float)H) / PX / 1000.0f, 0.5f) * PX;
    int reg = P.mono_text ? F_MONO : F_REG, semi = P.mono_text ? F_MONO : F_SEMI, light = P.mono_text ? F_MONO : F_LIGHT;
    st_label = style(semi, 11.5f * Uu);
    st_tiny = style(F_MONO, 10.5f * Uu);
    st_small = style(reg, 14 * Uu);
    st_body = style(reg, 17 * Uu);
    st_mono = style(F_MONO, 13.5f * Uu);
    st_monob = style(F_MONO, 27 * Uu);
    st_title = style(light, 29 * Uu);
    st_card = style(light, 52 * Uu);
    st_cardk = style(semi, 13 * Uu);
    st_time = style(light, 29 * Uu);
    build_atlas();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    styled_w = W; styled_h = H;
}

static void fmt_int(char *o, size_t n, long v) {
    char b[32];
    snprintf(b, sizeof b, "%ld", v < 0 ? -v : v);
    int len = (int)strlen(b), k = 0;
    if (v < 0 && k < (int)n - 1) o[k++] = '-';
    for (int i = 0; i < len && k < (int)n - 2; i++) {
        o[k++] = b[i];
        if ((len - 1 - i) % 3 == 0 && i < len - 1) o[k++] = ',';
    }
    o[k] = 0;
}
static void fmt_val(char *o, size_t n, double v) {
    if (v >= 100) snprintf(o, n, "%.0f", v);
    else if (v >= 1) snprintf(o, n, "%.3f", v);
    else if (v >= 0.01) snprintf(o, n, "%.4f", v);
    else snprintf(o, n, "%.2e", v);
}

static const float *cseq(float t) {
    t = clampf(t, 0, 1) * 4;
    int i = (int)fminf(floorf(t), 3);
    return cmix(P.seq[i], P.seq[i + 1], t - i);
}

/* a panel's label (small caps, tracked) with an optional note on the right;
 * returns the space below it */
static Rect panel(Rect r, const char *label, const char *note) {
    char up[128];
    int i = 0;
    for (; label[i] && i < 127; i++) up[i] = (char)toupper((unsigned char)label[i]);
    up[i] = 0;
    float y = r.y0 + cap_h(st_label);
    float lw = text(st_label, r.x0, y, -1, up, P.dim, 1.6f * Uu);
    if (note) {
        float nw = text_width(st_small, note, 0);
        if (r.x0 + lw + 16 * Uu + nw < r.x1) text(st_small, r.x1, y, 1, note, ca(P.fg, 0.8f), 0);
    }
    return rect(r.x0, y + 11 * Uu, r.x1, r.y1);
}

/* ---- line plots ---- */
typedef struct {
    const float *y, *x;                /* x NULL: y per step 0..n-1 */
    int n;
    const float *col;
    int raw;                           /* noisy: draw a min/max band and the bucket means */
    const char *name;
} Series;

static float plot_lo, plot_hi;         /* the y range used (log10 when logy) */

/* a 1, 2 or 5 times a power of ten, giving at most n ticks over span */
static float nice_step(float span, int n) {
    float st = powf(10, floorf(log10f(span / n)));
    if (span / st > n) st *= 2;
    if (span / st > n) st *= 2.5f;
    if (span / st > n) st *= 2;
    return st;
}

static void plot(Rect r, const Series *S, int ns, float xmax, int logy, float ylo, float yhi, const char *xlab) {
    Rect a = rect(r.x0 + 44 * Uu, r.y0 + 4 * Uu, r.x1 - 6 * Uu, r.y1 - 17 * Uu);
    if (rw(a) < 20 || rh(a) < 20) return;
    float lo = ylo, hi = yhi;
    if (logy) {
        float mn = 1e30f, mx = -1e30f;
        for (int s = 0; s < ns; s++)
            for (int i = 0; i < S[s].n; i++) {
                float v = S[s].y[i];
                if (v > 0 && isfinite(v)) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
            }
        if (mn > mx) { mn = 0.1f; mx = 1; }
        lo = floorf(log10f(mn)); hi = ceilf(log10f(mx));
        if (hi - lo < 1) hi = lo + 1;
        if (ylo > lo) lo = ylo;
    }
    plot_lo = lo; plot_hi = hi;
#define PYV(v) (a.y1 - ((logy ? log10f(fmaxf(v, 1e-30f)) : (v)) - lo) / (hi - lo) * rh(a))
#define PXV(i) (a.x0 + (i) / fmaxf(xmax, 1e-6f) * rw(a))
    /* grid */
    const float *gc = P.grid;
    if (logy) {
        int step = hi - lo > 6 ? 2 : 1;
        for (int d = (int)lo; d <= (int)hi; d++) {
            float y = a.y1 - (d - lo) / (hi - lo) * rh(a);
            seg(a.x0, y, a.x1, y, 1 * Uu, ca(gc, 0.8f));
            if ((d - (int)lo) % step == 0) {
                char b[16];
                if (d >= 0 && d <= 3) snprintf(b, sizeof b, "%g", pow(10, d));
                else if (d >= -2 && d < 0) snprintf(b, sizeof b, "%g", pow(10, d));
                else snprintf(b, sizeof b, "1e%d", d);
                text(st_tiny, a.x0 - 7 * Uu, y + cap_h(st_tiny) * 0.5f, 1, b, P.dim, 0);
            }
            if (d < (int)hi)
                for (int k = 2; k < 10; k++) {
                    float yy = a.y1 - (d + log10f((float)k) - lo) / (hi - lo) * rh(a);
                    seg(a.x0 - 3 * Uu, yy, a.x0, yy, 1 * Uu, ca(gc, 1));
                }
        }
    } else {
        float span = hi - lo, st = nice_step(span, 4);
        for (float v = ceilf(lo / st) * st; v <= hi + st * 0.01f; v += st) {
            float y = a.y1 - (v - lo) / (hi - lo) * rh(a);
            seg(a.x0, y, a.x1, y, 1 * Uu, ca(gc, 0.8f));
            textf(st_tiny, a.x0 - 7 * Uu, y + cap_h(st_tiny) * 0.5f, 1, P.dim, 0, "%g", v);
        }
    }
    seg(a.x0, a.y1, a.x1, a.y1, 1.2f * Uu, P.dim);
    /* x ticks */
    {
        float st = fmaxf(nice_step(fmaxf(xmax, 1), rw(a) > 500 * Uu ? 5 : 3), 1);
        for (float v = 0; v <= xmax * 1.0001f; v += st) {
            float x = PXV(v);
            seg(x, a.y1, x, a.y1 + 3 * Uu, 1 * Uu, P.dim);
            char b[24];
            fmt_int(b, sizeof b, (long)v);
            text(st_tiny, x, a.y1 + 5 * Uu + cap_h(st_tiny), 0, b, P.dim, 0);
        }
        if (xlab) text(st_tiny, a.x1, a.y1 - 5 * Uu, 1, xlab, ca(P.dim, 0.8f), 0);
    }
    /* series */
    for (int s = 0; s < ns; s++) {
        const Series *q = &S[s];
        if (q->n < 1) continue;
        float lw = 1.7f * Uu;
        if (!q->x) {
            int nb = (int)fminf((float)q->n, rw(a) / (2.5f * Uu));
            if (nb < 1) nb = 1;
            float px = 0, py = 0;
            for (int b = 0; b < nb; b++) {
                int i0 = (int)((long)b * q->n / nb), i1 = (int)((long)(b + 1) * q->n / nb);
                if (i1 <= i0) i1 = i0 + 1;
                float mn = 1e30f, mx = -1e30f, sum = 0;
                int c = 0;
                for (int i = i0; i < i1 && i < q->n; i++) {
                    float v = q->y[i];
                    if (!isfinite(v) || (logy && v <= 0)) continue;
                    mn = fminf(mn, v); mx = fmaxf(mx, v);
                    sum += logy ? logf(v) : v;
                    c++;
                }
                if (!c) continue;
                float mean = logy ? expf(sum / c) : sum / c;
                float x = PXV((i0 + i1 - 1) * 0.5f);
                float y = clampf(PYV(mean), a.y0, a.y1);
                if (q->raw && i1 - i0 > 1)
                    seg(x, clampf(PYV(mx), a.y0, a.y1), x, clampf(PYV(mn), a.y0, a.y1), 2.2f * Uu, ca(q->col, 0.18f));
                if (b) seg(px, py, x, y, lw, q->col);
                px = x; py = y;
            }
        } else {
            float px = 0, py = 0;
            for (int i = 0; i < q->n; i++) {
                float x = PXV(q->x[i]), y = clampf(PYV(q->y[i]), a.y0, a.y1);
                if (i) seg(px, py, x, y, lw, q->col);
                px = x; py = y;
            }
        }
        if (q->raw == 2) continue;   /* a reference line: no head */
        /* the head */
        int i = q->n - 1;
        float x = q->x ? PXV(q->x[i]) : PXV((float)i), y = clampf(PYV(q->y[i]), a.y0, a.y1);
        glow(x, y, 12 * Uu, ca(q->col, 0.5f));
        circle(x, y, 3.2f * Uu, q->col);
    }
#undef PYV
#undef PXV
}

static void legend(float x, float y, const char *name, const float *col, const char *val) {
    circle(x + 4 * Uu, y - cap_h(st_small) * 0.5f, 4 * Uu, col);
    float w = text(st_small, x + 14 * Uu, y, -1, name, P.fg, 0);
    if (val) text(st_mono, x + 14 * Uu + w + 8 * Uu, y, -1, val, P.dim, 0);
}

/* ---------------- Act I ---------------- */
static void fit_pair(Rect r, int side) {
    float gap = 22 * Uu, lab = cap_h(st_label) + 11 * Uu, foot = 26 * Uu;
    int stack = !side && rh(r) > rw(r) * 1.5f;
    float s = stack ? fminf(rw(r), (rh(r) - gap) / 2 - lab - foot) : fminf((rw(r) - gap) / 2, rh(r) - lab - foot);
    if (s >= IMG * 2 && floorf(s / IMG) * IMG >= 0.8f * s) s = floorf(s / IMG) * IMG;
    s = floorf(s);
    float tw = stack ? s : 2 * s + gap, th = stack ? 2 * (s + lab + foot) + gap : s + lab + foot;
    float x0 = roundf(r.x0 + (rw(r) - tw) / 2), y0 = roundf(r.y0 + (rh(r) - th) / 2);
    for (int k = 0; k < 2; k++) {
        float bx = stack ? x0 : x0 + k * (s + gap), by = stack ? y0 + k * (s + lab + foot + gap) : y0;
        char note[64];
        if (k == 0) snprintf(note, sizeof note, "%s", PIC_NAME[F.pic]);
        else {
            char b[24];
            fmt_int(b, sizeof b, run.step);
            snprintf(note, sizeof note, "step %s", b);
        }
        Rect in = panel(rect(bx, by, bx + s, by + s + lab), k ? "network output" : "target", note);
        float ix0 = bx, iy0 = in.y0;
        if (k == 0) draw_tex(ix0, iy0, ix0 + s, iy0 + s, tex_tgt, 0, 0, 0, 1, 0, NULL);
        else {
            float mixv = clampf((float)(step_frac - (F.pub_step[1] + EVAL_E - 1)) / EVAL_E, 0, 1);
            if (F.pub_step[1] == 0) mixv = 1;
            draw_tex(ix0, iy0, ix0 + s, iy0 + s, tex_pred[0], tex_pred[1], mixv, 0, 1, 0, NULL);
        }
        rect_line(ix0 - 0.5f, iy0 - 0.5f, ix0 + s + 0.5f, iy0 + s + 0.5f, 1 * Uu, P.grid);
        float fy = iy0 + s + 9 * Uu + cap_h(st_small);
        if (k == 0) textf(st_small, ix0, fy, -1, P.dim, 0, "%d x %d" SEP "%s values", IMG, IMG, "49,152");
        else {
            float ps = F.psnr[F.nsnap - 1];
            textf(st_small, ix0, fy, -1, P.dim, 0, "%s" SEP "%d params", F.model == M_SIREN ? "SIREN, sine activations" : "Fourier features + ReLU", fit_params());
            textf(st_mono, ix0 + s, fy, 1, P.fg, 0, "PSNR %.1f dB", ps);
        }
    }
}

static void fit_loss(Rect r) {
    char b[32];
    fmt_val(b, sizeof b, F.loss[run.step > 0 ? run.step : 0]);
    Rect in = panel(r, "loss", NULL);
    static float mx[MAXSNAP], my[MAXSNAP];
    int n = F.nsnap;
    for (int i = 0; i < n; i++) { mx[i] = (float)F.snap_at[i]; my[i] = powf(10, -F.psnr[i] / 10); }
    Series S[2] = {
        {F.loss + 1, NULL, run.step, P.accent, 1, "minibatch"},
        {my, mx, n, P.fg, 0, "full image"},
    };
    plot(rect(in.x0, in.y0 + 22 * Uu, in.x1, in.y1), S, run.step > 0 ? 2 : 1, (float)F.nsteps, 1, -6, 0, "step");
    float ly = in.y0 + cap_h(st_small);
    legend(in.x0 + 44 * Uu, ly, "minibatch MSE", P.accent, b);
    fmt_val(b, sizeof b, my[n - 1]);
    legend(in.x0 + rw(in) * 0.55f, ly, "full image", P.fg, b);
}

static void colorbar(float x0, float y, float w, const char *lo, const char *hi) {
    int n = 24;
    float h = 6 * Uu;
    for (int i = 0; i < n; i++)
        rect_fill(x0 + w * i / n, y - h, x0 + w * (i + 1) / n + 0.5f, y, 0, cseq((i + 0.5f) / n));
    text(st_tiny, x0 - 6 * Uu, y, 1, lo, P.dim, 0);
    text(st_tiny, x0 + w + 6 * Uu, y, -1, hi, P.dim, 0);
}

static void fit_spec(Rect r) {
    Rect in = panel(r, "error by spatial frequency", NULL);
    Rect a = rect(in.x0 + 44 * Uu, in.y0 + 22 * Uu, in.x1 - 6 * Uu, in.y1 - 17 * Uu);
    if (rh(a) < 20) return;
    colorbar(in.x1 - 150 * Uu, in.y0 + 8 * Uu, 90 * Uu, "learned", "not yet");
    text(st_small, in.x0 + 44 * Uu, in.y0 + 8 * Uu, -1, "broad shapes first, fine detail later", ca(P.fg, 0.8f), 0);
    int total = F.nsteps / EVAL_E + 1;
    float fx = a.x0 + rw(a) * fminf((float)F.nsnap / total, 1);
    float sub[4] = {0, 1, (float)F.nsnap / MAXSNAP, 0};
    flush();
    if (F.nsnap > 0) draw_tex(a.x0, a.y0, fx, a.y1, tex_spec, 0, 0, 2, 1 / 3.5f, 3 / 3.5f, sub);
    rect_line(a.x0, a.y0, a.x1, a.y1, 1 * Uu, P.grid);
    /* frequency axis: cycles per image */
    for (int f = 1; f <= IMG / 2; f *= 4) {
        float y = a.y1 - log2f((float)f) / log2f(IMG / 2.0f) * rh(a);
        seg(a.x0 - 3 * Uu, y, a.x0, y, 1 * Uu, P.dim);
        textf(st_tiny, a.x0 - 7 * Uu, y + cap_h(st_tiny) * 0.5f, 1, P.dim, 0, "%d", f);
    }
    float st = F.nsteps > 600 ? 250 : 100;
    for (float v = 0; v <= F.nsteps; v += st) {
        float x = a.x0 + v / F.nsteps * rw(a);
        seg(x, a.y1, x, a.y1 + 3 * Uu, 1 * Uu, P.dim);
        textf(st_tiny, x, a.y1 + 5 * Uu + cap_h(st_tiny), 0, P.dim, 0, "%d", (int)v);
    }
}

static void fit_tiles(Rect r, int layer, int cols) {
    char lab[64];
    snprintf(lab, sizeof lab, "layer %d units", layer + 1);
    Rect in = panel(r, lab, F.model == M_SIREN ? "sin(30 (Wx + b))" : "max(0, Wx + b)");
    if (layer >= F.ntl) layer = F.ntl - 1;
    int rows = FH / cols;
    float gap = fmaxf(2 * PX, 3 * Uu);
    float t = fminf((rw(in) - gap * (cols - 1)) / cols, (rh(in) - gap * (rows - 1)) / rows);
    t = floorf(t);
    if (t < 4) return;
    float x0 = roundf(in.x0 + (rw(in) - (t * cols + gap * (cols - 1))) / 2), y0 = in.y0;
    flush();
    for (int u = 0; u < FH; u++) {
        int cx = u % cols, cy = u / cols;
        float sx = (u % 8) * TILE, sy = layer * 8 * TILE + (u / 8) * TILE;
        float sub[4] = {(sx + 0.5f) / (8 * TILE), (sy + 0.5f) / (3 * 8 * TILE), (sx + TILE - 0.5f) / (8 * TILE),
                        (sy + TILE - 0.5f) / (3 * 8 * TILE)};
        float bx = x0 + cx * (t + gap), by = y0 + cy * (t + gap);
        draw_tex(bx, by, bx + t, by + t, tex_tiles, 0, 0, 1, 1, 0, sub);
    }
}

/* the strongest connections of the snapshot, per layer */
#define EK 420
static struct { short i, o; float w; } fedge[4][EK];
static int nfedge[4], fedge_ver = -1;
static float fedge_max[4];

static int cmp_absw(const void *a, const void *b) {
    float x = fabsf(*(const float *)a), y = fabsf(*(const float *)b);
    return x < y ? 1 : x > y ? -1 : 0;
}

static void fit_edges(void) {
    static float tmp[2 * FFM * FH];
    for (int l = 0; l < F.nl; l++) {
        Dense *L = &F.L[l];
        int n = L->in * L->out;
        memcpy(tmp, L->W, (size_t)n * sizeof(float));
        qsort(tmp, (size_t)n, sizeof(float), cmp_absw);
        float th = fabsf(tmp[n < EK ? n - 1 : EK - 1]);
        fedge_max[l] = fabsf(tmp[0]) + 1e-9f;
        int k = 0;
        for (int i = 0; i < L->in && k < EK; i++)
            for (int o = 0; o < L->out && k < EK; o++)
                if (fabsf(L->W[i * L->out + o]) >= th) fedge[l][k++] = (typeof(fedge[0][0])){(short)i, (short)o, L->W[i * L->out + o]};
        nfedge[l] = k;
    }
    fedge_ver = F.pub_ver;
}

static void fit_net(Rect r) {
    Rect in = panel(r, "the network", "strongest connections");
    if (fedge_ver != F.pub_ver) fit_edges();
    int ncol = F.nl + 1;
    int size[5];
    for (int l = 0; l < F.nl; l++) size[l] = F.L[l].in;
    size[F.nl] = 3;
    float colx[5], top = in.y0 + 26 * Uu, bot = in.y1 - 8 * Uu;
    for (int c = 0; c < ncol; c++) colx[c] = in.x0 + 30 * Uu + (rw(in) - 60 * Uu) * c / (ncol - 1);
#define NODEY(c, i) (size[c] <= 3 ? (top + bot) / 2 + ((i) - (size[c] - 1) / 2.0f) * 60 * Uu \
                                  : top + (bot - top) * ((i) + 0.5f) / size[c])
    for (int l = 0; l < F.nl; l++)
        for (int k = 0; k < nfedge[l]; k++) {
            float m = fabsf(fedge[l][k].w) / fedge_max[l];
            float y0 = NODEY(l, fedge[l][k].i), y1 = NODEY(l + 1, fedge[l][k].o);
            seg(colx[l], y0, colx[l + 1], y1, (0.5f + 1.6f * m) * Uu, ca(cdiv(fedge[l][k].w / fedge_max[l]), 0.10f + 0.55f * m));
        }
    static const char *rgbn[3] = {"#e8524a", "#56b870", "#4a86e8"};
    for (int c = 0; c < ncol; c++) {
        float rr = fminf(3.2f * Uu, (bot - top) / size[c] * 0.32f);
        if (size[c] <= 3) rr = 6 * Uu;
        for (int i = 0; i < size[c]; i++) {
            float y = NODEY(c, i);
            float col[4];
            if (c == ncol - 1) hex4(rgbn[i], col);
            else memcpy(col, P.fg, sizeof col);
            circle(colx[c], y, rr, P.bg);
            ring(colx[c], y, rr, fmaxf(PX, 1.2f * Uu), ca(col, c == ncol - 1 ? 1 : 0.75f));
        }
        const char *nm = c == 0 ? (F.model == M_SIREN ? "x, y" : "sin, cos (Bx)") : c == ncol - 1 ? "r g b" : "64";
        text(st_tiny, colx[c], top - 12 * Uu, 0, nm, P.dim, 0);
    }
#undef NODEY
}

/* ---------------- Act II ---------------- */
static const float CLS_DOM = 1.15f;

static void cls_uniforms(void) {
    GLuint p = prog_field;
    glUseProgram(p);
    static const char *wn[NL2] = {"uW1", "uW2", "uW3", "uW4"}, *bn[NL2] = {"uB1", "uB2", "uB3", "uB4"};
    for (int l = 0; l < NL2; l++) {
        glUniform1fv(U(p, wn[l]), C.L[l].in * C.L[l].out, C.L[l].W);
        glUniform1fv(U(p, bn[l]), C.L[l].out, C.L[l].b);
    }
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform1f(U(p, "uScale"), IN_SCALE);
    glUniform1f(U(p, "uPx"), PX);
    glUniform3fv(U(p, "uC0"), 1, P.cls[0]);
    glUniform3fv(U(p, "uC1"), 1, P.cls[1]);
    glUniform3fv(U(p, "uBg"), 1, P.panel);
    glUniform3fv(U(p, "uFg"), 1, P.fg);
    glUniform1f(U(p, "uAlpha"), 1);
}

static void field(float x0, float y0, float x1, float y1, int layer, int unit, float dom) {
    flush();
    GLuint p = prog_field;
    glUseProgram(p);
    glUniform4f(U(p, "uRect"), x0, y0, x1, y1);
    glUniform4f(U(p, "uDom"), -dom, -dom, dom, dom);
    glUniform1i(U(p, "uLayer"), layer);
    glUniform1i(U(p, "uUnit"), unit);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static float cls_acc_now(void) { return C.acc[run.step]; }

static void cls_field(Rect r) {
    char note[64];
    snprintf(note, sizeof note, "%s" SEP "%d points", DS_NAME[C.ds], NP2);
    float foot = 26 * Uu;
    Rect in = panel(r, "decision boundary", note);
    float s = floorf(fminf(rw(in), rh(in) - foot));
    float x0 = roundf(in.x0 + (rw(in) - s) / 2), y0 = in.y0;
    if (s < 40) return;
    cls_uniforms();
    field(x0, y0, x0 + s, y0 + s, -1, 0, CLS_DOM);
    rect_line(x0 - 0.5f, y0 - 0.5f, x0 + s + 0.5f, y0 + s + 0.5f, 1 * Uu, P.grid);
    float pr = fmaxf(2.2f * PX, 3.4f * Uu * s / PX / 700);   /* s / PX: the panel in logical px */
    for (int i = 0; i < NP2; i++) {
        float px = x0 + (C.x[i * 2] / CLS_DOM * 0.5f + 0.5f) * s, py = y0 + (0.5f - C.x[i * 2 + 1] / CLS_DOM * 0.5f) * s;
        int c = C.y[i] > 0.5f;
        int wrong = (C.a[NL2][i] > 0) != c;
        circle(px, py, pr + 1.2f * Uu, P.light ? P.panel : P.bg);
        circle(px, py, pr, P.cls[c]);
        if (wrong) ring(px, py, pr + 3 * Uu, 1.2f * Uu, P.fg);
    }
    float fy = y0 + s + 9 * Uu + cap_h(st_small);
    textf(st_small, x0, fy, -1, P.dim, 0, "2-12-12-12-1, tanh" SEP "full batch, Adam");
    textf(st_mono, x0 + s, fy, 1, P.fg, 0, "accuracy %.1f%%", cls_acc_now() * 100);
}

static void cls_net(Rect r) {
    Rect in = panel(r, "the network", "edges: weights" SEP "pulses: gradients");
    int ncol = NL2 + 1;
    float top = in.y0 + 22 * Uu, bot = in.y1 - 4 * Uu;
    float cw = rw(in) / ncol;
    float rowh = (bot - top) / H2;
    float ts = floorf(fminf(cw * 0.34f, rowh * 0.8f));
    float colx[NL2 + 1];
    for (int c = 0; c < ncol; c++) colx[c] = in.x0 + cw * (c + 0.5f);
#define NY(c, i) (top + (bot - top) / 2 + ((i) - (CLS_DIMS[c] - 1) / 2.0f) * rowh)
    float gmax = 1e-12f;
    for (int l = 0; l < NL2; l++)
        for (int i = 0; i < C.L[l].in * C.L[l].out; i++) gmax = fmaxf(gmax, C.gm[l][i]);
    for (int l = 0; l < NL2; l++) {
        float wmax = 1e-6f;
        for (int i = 0; i < C.L[l].in * C.L[l].out; i++) wmax = fmaxf(wmax, fabsf(C.L[l].W[i]));
        wmax = fmaxf(wmax, 1.5f);
        for (int i = 0; i < C.L[l].in; i++)
            for (int o = 0; o < C.L[l].out; o++) {
                float w = C.L[l].W[i * C.L[l].out + o], m = fminf(fabsf(w) / wmax, 1);
                float ax = colx[l] + ts / 2, ay = NY(l, i), bx = colx[l + 1] - ts / 2, by = NY(l + 1, o);
                seg(ax, ay, bx, by, (0.4f + 3.6f * m) * Uu, ca(w > 0 ? P.pos : P.neg, 0.12f + 0.7f * m));
                /* the gradient running back along it */
                float g = sqrtf(C.gm[l][i * C.L[l].out + o] / gmax);
                if (g > 0.05f) {
                    uint32_t h = hash32((uint32_t)(l * 4096 + i * 64 + o));
                    float ph = (float)fmod(draw_t * 0.45 + (h & 0xffff) / 65536.0, 1.0);
                    float px = bx + (ax - bx) * ph, py = by + (ay - by) * ph;
                    float a = g * sinf(ph * PI);
                    glow(px, py, 9 * Uu, ca(P.accent, 0.5f * a));
                    circle(px, py, 1.6f * Uu, ca(P.accent, 0.9f * a));
                }
            }
    }
    flush();
    cls_uniforms();
    for (int c = 0; c < ncol; c++)
        for (int i = 0; i < CLS_DIMS[c]; i++) {
            float x = colx[c] - ts / 2, y = NY(c, i) - ts / 2;
            field(x, y, x + ts, y + ts, c == NL2 ? -1 : c, i, CLS_DOM);
        }
    for (int c = 0; c < ncol; c++) {
        for (int i = 0; i < CLS_DIMS[c]; i++) {
            float x = colx[c] - ts / 2, y = NY(c, i) - ts / 2;
            rect_line(x, y, x + ts, y + ts, 1 * Uu, ca(P.fg, 0.35f));
        }
        const char *nm = c == 0 ? "input" : c == NL2 ? "output" : "tanh";
        text(st_tiny, colx[c], top - 8 * Uu, 0, nm, P.dim, 0);
    }
    text(st_tiny, colx[0] - ts / 2 - 6 * Uu, NY(0, 0) + cap_h(st_tiny) / 2, 1, "x", P.dim, 0);
    text(st_tiny, colx[0] - ts / 2 - 6 * Uu, NY(0, 1) + cap_h(st_tiny) / 2, 1, "y", P.dim, 0);
#undef NY
}

static void cls_tiles(Rect r) {
    Rect in = panel(r, "hidden unit activations", "over the plane");
    int cols = rw(in) > rh(in) * 1.2f ? 12 : 6;
    int rows = 3 * (12 / cols);
    float lab = 18 * Uu, gap = 4 * Uu;
    float t = floorf(fminf((rw(in) - gap * (cols - 1)) / cols, (rh(in) - 3 * lab - gap * rows) / rows));
    if (t < 6) return;
    flush();
    cls_uniforms();
    float y = in.y0;
    float x0 = roundf(in.x0 + (rw(in) - (t * cols + gap * (cols - 1))) / 2);
    for (int l = 1; l <= 3; l++) {
        textf(st_tiny, x0, y + cap_h(st_tiny), -1, P.dim, 0, "layer %d", l);
        y += lab;
        for (int u = 0; u < H2; u++) {
            int cx = u % cols, cy = u / cols;
            float bx = x0 + cx * (t + gap), by = y + cy * (t + gap);
            field(bx, by, bx + t, by + t, l, u, CLS_DOM);
        }
        y += (12 / cols) * (t + gap);
    }
}

static void cls_loss(Rect r) {
    Rect a, b;
    vsplit(r, 0.6f, 14 * Uu, &a, &b);
    char v[32];
    fmt_val(v, sizeof v, C.loss[run.step]);
    Rect in = panel(a, "loss", v);
    Series S = {C.loss, NULL, run.step + 1, P.accent, 0, "loss"};
    plot(in, &S, 1, (float)C.nsteps, 1, -5, 0, "step");
    snprintf(v, sizeof v, "%.1f%%", C.acc[run.step] * 100);
    in = panel(b, "train accuracy", v);
    Series A = {C.acc, NULL, run.step + 1, P.fg, 0, "acc"};
    plot(in, &A, 1, (float)C.nsteps, 0, 0.4f, 1.0f, "step");
}

/* ---------------- Act III ---------------- */
#define CPS 32                         /* typing speed, characters a second */
static float e2d_prev[MAXV][2];
static int e2d_prev_n = -1;

static double smp_time(const Sample *s) { return T_TRAIN + s->step / SPS[A_CHAR]; }

static void char_type(Rect r) {
    const Sample *s = &T.smp[T.nsmp - 1];
    char note[64];
    snprintf(note, sizeof note, "epoch %.1f" SEP "temperature 0.75", s->epoch);
    Rect in = panel(r, "a sample, written now", note);
    float adv = glyph(st_monob, 'M')->adv, lh = cap_h(st_monob) * 2.1f;
    int cols = (int)(rw(in) / adv);
    if (cols < 8) return;
    int typed = (int)((draw_t - smp_time(s)) * CPS);
    if (typed > s->len) typed = s->len;
    if (typed < 0) typed = 0;
    /* wrap at spaces */
    int brk[16], nl = 0, start = 0;
    while (start < s->len && nl < 15) {
        int end = start + cols;
        if (end >= s->len) { brk[nl++] = s->len; break; }
        int b = end;
        while (b > start && s->txt[b] != ' ') b--;
        if (b == start) b = end;
        brk[nl++] = b;
        start = b + (s->txt[b] == ' ');
    }
    if (!nl) brk[nl++] = 0;
    float y = in.y0 + 14 * Uu + cap_h(st_monob);
    float cx = in.x0, cy = y;
    int pos = 0;
    for (int l = 0; l < nl; l++) {
        char line[SMPLEN + 1];
        int e = brk[l], n = 0;
        for (int i = pos; i < e && i < typed; i++) line[n++] = s->txt[i];
        line[n] = 0;
        text(st_monob, in.x0, y, -1, line, P.fg, 0);
        if (typed >= pos && (typed <= e || l == nl - 1)) { cx = in.x0 + n * adv; cy = y; }
        pos = e + (e < s->len && s->txt[e] == ' ');
        y += lh;
    }
    float blink = typed < s->len ? 1 : (fmod(draw_t, 1.0) < 0.55 ? 1 : 0.15f);
    rect_fill(cx + 1 * Uu, cy - cap_h(st_monob) * 1.15f, cx + adv * 0.85f, cy + cap_h(st_monob) * 0.25f, 0, ca(P.accent, blink));
    /* what it expects next */
    int at = typed < s->len ? typed : s->len;
    float bx = cx, by = cy + 22 * Uu, bw = 120 * Uu;
    if (bx + bw + 60 * Uu > in.x1) bx = in.x1 - bw - 60 * Uu;
    if (bx < in.x0) bx = in.x0;
    for (int k = 0; k < 4; k++) {
        float p = s->topp[at][k];
        char c = T.chars[s->top[at][k]];
        char lab[8];
        if (c == '\n') snprintf(lab, sizeof lab, "\\n");
        else if (c == ' ') snprintf(lab, sizeof lab, "_");
        else snprintf(lab, sizeof lab, "%c", c);
        float yy = by + k * 19 * Uu + cap_h(st_mono);
        text(st_mono, bx, yy, -1, lab, k == 0 ? P.accent : P.fg, 0);
        rect_fill(bx + 26 * Uu, yy - cap_h(st_mono), bx + 26 * Uu + bw * p + 1, yy, 1.5f * Uu, ca(k == 0 ? P.accent : P.dim, 0.8f));
        textf(st_tiny, bx + 32 * Uu + bw * p, yy, -1, P.dim, 0, "%.0f%%", p * 100);
    }
}

static void char_log(Rect r) {
    Rect in = panel(r, "samples, epoch by epoch", "newest first");
    float lh = cap_h(st_mono) * 2.0f;
    float adv = glyph(st_mono, 'M')->adv;
    float y = in.y0 + cap_h(st_mono) + 4 * Uu;
    float tx = in.x0 + 9 * adv;
    int maxc = (int)((in.x1 - tx) / adv);
    int shown = 0;
    for (int k = T.nsmp - 2; k >= 0 && y < in.y1; k--, shown++) {
        const Sample *s = &T.smp[k];
        float a = fmaxf(0.25f, 1 - shown * 0.05f);
        textf(st_mono, in.x0, y, -1, ca(P.dim, a), 0, "ep %5.1f", s->epoch);
        char line[SMPLEN + 4];
        int n = s->len < maxc ? s->len : maxc - 1;
        if (n < 0) n = 0;
        memcpy(line, s->txt, (size_t)n);
        line[n] = 0;
        if (n < s->len) strcat(line, "~");
        text(st_mono, tx, y, -1, line, ca(P.fg, a), 0);
        y += lh;
    }
}

static void char_loss(Rect r) {
    char v[48];
    snprintf(v, sizeof v, "%.2f nats" SEP "%.2f bits/char", T.loss[run.step], T.loss[run.step] / logf(2));
    Rect in = panel(r, "cross-entropy", v);
    static float sm[8192];
    int n = run.step + 1 < 8192 ? run.step + 1 : 8192;
    Series S[2] = {{T.loss, NULL, n, P.accent, 1, "loss"}, {sm, NULL, 2, P.dim, 2, "uniform"}};
    float base = logf((float)T.V);
    float xs[2] = {0, (float)T.nsteps};
    sm[0] = sm[1] = base;
    S[1].x = xs;
    plot(in, S, 2, (float)T.nsteps, 1, -2, 1, "step");
    text(st_tiny, in.x1 - 6 * Uu, in.y0 + 14 * Uu, 1, "grey: guessing uniformly", P.dim, 0);
}

static void char_emb(Rect r) {
    Rect in = panel(r, "character embeddings", "first two principal components");
    float s = fminf(rw(in), rh(in)) * 0.92f;
    float cx = (in.x0 + in.x1) / 2, cy = (in.y0 + in.y1) / 2;
    seg(cx - s / 2, cy, cx + s / 2, cy, 1 * Uu, P.grid);
    seg(cx, cy - s / 2, cx, cy + s / 2, 1 * Uu, P.grid);
    const Sample *sm = &T.smp[T.nsmp - 1];
    if (e2d_prev_n != T.nsmp) {
        /* T.e2d is already the newest; keep the previous one for the glide */
        static float last[MAXV][2];
        static int have;
        if (have) memcpy(e2d_prev, last, sizeof last);
        else memcpy(e2d_prev, T.e2d, sizeof e2d_prev);
        memcpy(last, T.e2d, sizeof last);
        have = 1;
        e2d_prev_n = T.nsmp;
    }
    float f = smooth(0, 1.5f, (float)(draw_t - smp_time(sm)));
    for (int v = 0; v < T.V; v++) {
        float x = mixf(e2d_prev[v][0], T.e2d[v][0], f), y = mixf(e2d_prev[v][1], T.e2d[v][1], f);
        float px = cx + x * s * 0.46f, py = cy - y * s * 0.46f;
        char c = T.chars[v], lab[4];
        const float *col = P.fg;
        if (strchr("aeiouAEIOU", c)) col = P.accent;
        else if (isdigit((unsigned char)c)) col = P.pos;
        else if (!isalpha((unsigned char)c)) col = P.dim;
        if (c == '\n') snprintf(lab, sizeof lab, "\\n");
        else if (c == ' ') snprintf(lab, sizeof lab, "_");
        else snprintf(lab, sizeof lab, "%c", c);
        text(st_mono, px, py + cap_h(st_mono) / 2, 0, lab, col, 0);
    }
    float ly = in.y1 - 4 * Uu;
    legend(in.x0, ly, "vowels", P.accent, NULL);
    legend(in.x0 + 90 * Uu, ly, "consonants", P.fg, NULL);
    legend(in.x0 + 205 * Uu, ly, "punctuation", P.dim, NULL);
}

static void char_weights(Rect r) {
    Rect in = panel(r, "output weights", "hidden units x characters");
    float w = rw(in), h = rh(in);
    if (w < 20 || h < 20) return;
    float sub[4] = {0, 0, (float)T.V / MAXV, 1};
    float mx = 1e-6f;
    for (int i = 0; i < HID * T.V; i++) mx = fmaxf(mx, fabsf(T.L2.W[i]));
    draw_tex(in.x0, in.y0, in.x1, in.y1, tex_w3, 0, 0, 1, 1 / mx, 0, sub);
}

/* ---------------- Act IV ---------------- */
static float opt_hgt(float x, float y) {
    return logf(1 + (float)fn_f(O.fn, x, y, NULL, NULL)) / logf(1 + O.fmax);
}
#define HS 0.55f

static void mat_mul(float *o, const float *a, const float *b) {
    float t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
            t[c * 4 + r] = s;
        }
    memcpy(o, t, sizeof t);
}

static void opt_mvp(float aspect, int top, float *m) {
    memset(m, 0, 16 * sizeof(float));
    if (top) {
        float k = 1.0f;
        float kx = aspect >= 1 ? k / aspect : k, ky = aspect >= 1 ? k : k * aspect;
        m[0] = kx; m[6] = -0.3f; m[9] = -ky; m[15] = 1; m[14] = 0;
        return;
    }
    float az = 0.75f + (float)draw_t * 0.035f, el = 0.58f, R = 4.1f, ty = 0.12f;
    float ex = R * cosf(el) * cosf(az), ey = R * sinf(el) + ty, ez = R * cosf(el) * sinf(az);
    float fx = -ex, fy = ty - ey, fz = -ez, fl = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0 - fz * 1, sy = fz * 0 - fx * 0, sz = fx * 1 - fy * 0;   /* f x up(0,1,0) */
    float sl = sqrtf(sx * sx + sy * sy + sz * sz);
    sx /= sl; sy /= sl; sz /= sl;
    float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    float V[16] = {sx, ux, -fx, 0, sy, uy, -fy, 0, sz, uz, -fz, 0,
                   -(sx * ex + sy * ey + sz * ez), -(ux * ex + uy * ey + uz * ez), fx * ex + fy * ey + fz * ez, 1};
    float fov = 0.62f, f = 1 / tanf(fov / 2), n = 0.1f, fa = 20;
    float Pm[16] = {0};
    Pm[0] = f / aspect; Pm[5] = f; Pm[10] = (fa + n) / (n - fa); Pm[11] = -1; Pm[14] = 2 * fa * n / (n - fa);
    mat_mul(m, Pm, V);
}

static int opt_project(const float *m, Rect a, float x, float y, float lift, float *px, float *py) {
    float qx = (x - O.dom[0]) / (O.dom[2] - O.dom[0]), qy = (y - O.dom[1]) / (O.dom[3] - O.dom[1]);
    float wx = qx * 2 - 1, wy = opt_hgt(x, y) * HS + lift, wz = 1 - qy * 2;
    float cx = m[0] * wx + m[4] * wy + m[8] * wz + m[12];
    float cy = m[1] * wx + m[5] * wy + m[9] * wz + m[13];
    float cw = m[3] * wx + m[7] * wy + m[11] * wz + m[15];
    if (cw <= 1e-4f) return 0;
    *px = a.x0 + (cx / cw * 0.5f + 0.5f) * rw(a);
    *py = a.y0 + (0.5f - cy / cw * 0.5f) * rh(a);
    return 1;
}

static void opt_view(Rect r, int top) {
    char note[64];
    snprintf(note, sizeof note, "%s" SEP "log(1 + f)", FN_NAME[O.fn]);
    Rect a = panel(r, top ? "contour map" : "loss surface", note);
    if (top) {
        float s = fminf(rw(a), rh(a));
        a = rect(roundf((a.x0 + a.x1 - s) / 2), a.y0, roundf((a.x0 + a.x1 + s) / 2), a.y0 + s);
    }
    if (rw(a) < 20 || rh(a) < 20) return;
    flush();
    float m[16];
    opt_mvp(rw(a) / rh(a), top, m);
    int vx = (int)a.x0, vy = H - (int)a.y1, vw = (int)rw(a), vh = (int)rh(a);
    glViewport(vx, vy, vw, vh);
    glEnable(GL_SCISSOR_TEST);
    glScissor(vx, vy, vw, vh);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    GLuint p = prog_surf;
    glUseProgram(p);
    int N = top ? 200 : 150;
    glUniform1i(U(p, "uN"), N);
    glUniformMatrix4fv(U(p, "uMVP"), 1, GL_FALSE, m);
    glUniform4fv(U(p, "uDom"), 1, O.dom);
    glUniform1f(U(p, "uHs"), HS);
    glUniform1i(U(p, "uFn"), O.fn);
    glUniform1f(U(p, "uFmax"), O.fmax);
    glUniform1f(U(p, "uTop"), (float)top);
    glUniform1f(U(p, "uPx"), PX);
    glUniform3fv(U(p, "uFg"), 1, P.light ? P.fg : P.bg);
    glUniform3fv(U(p, "uBg"), 1, P.bg);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, (N - 1) * (N - 1) * 6);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);
    glViewport(0, 0, W, H);
    if (top) rect_line(a.x0, a.y0, a.x1, a.y1, 1 * Uu, P.grid);
    /* the minimum(s) */
    static const float mins[3][4][2] = {{{1, 1}}, {{0, 0}}, {{3, 2}, {-2.805118f, 3.131312f}, {-3.779310f, -3.283186f}, {3.584428f, -1.848126f}}};
    for (int k = 0; k < (O.fn == FN_HIMMEL ? 4 : 1); k++) {
        float px, py;
        if (opt_project(m, a, mins[O.fn][k][0], mins[O.fn][k][1], 0.01f, &px, &py)) {
            seg(px - 5 * Uu, py - 5 * Uu, px + 5 * Uu, py + 5 * Uu, 1.4f * Uu, P.fg);
            seg(px - 5 * Uu, py + 5 * Uu, px + 5 * Uu, py - 5 * Uu, 1.4f * Uu, P.fg);
        }
    }
    int n = run.step;
    for (int k = 0; k < NOPT; k++) {
        int stride = n / 400 + 1;
        float lx = 0, ly = 0;
        int have = 0;
        for (int i = 0; i <= n; i += stride) {
            int j = i + stride > n ? n : i;
            float px, py;
            if (!opt_project(m, a, O.tr[k][j * 2], O.tr[k][j * 2 + 1], 0.012f, &px, &py)) continue;
            if (have) seg(lx, ly, px, py, 1.8f * Uu, ca(P.opt[k], 0.35f + 0.65f * (float)i / (n + 1)));
            lx = px; ly = py; have = 1;
            if (j == n) break;
        }
        float px, py;
        if (opt_project(m, a, O.tr[k][n * 2], O.tr[k][n * 2 + 1], 0.012f, &px, &py)) {
            glow(px, py, 16 * Uu, ca(P.opt[k], 0.6f));
            circle(px, py, 4.5f * Uu, P.opt[k]);
            ring(px, py, 4.5f * Uu, 1.2f * Uu, P.light ? P.panel : P.bg);
        }
    }
    if (!top) {
        float ly = a.y0 + 10 * Uu + cap_h(st_small);
        for (int k = 0; k < NOPT; k++) {
            char v[32];
            fmt_val(v, sizeof v, O.lo[k][n]);
            legend(a.x0 + 4 * Uu, ly + k * 22 * Uu, OPT_NAME[k], P.opt[k], v);
        }
    }
}

static void opt_loss(Rect r) {
    Rect in = panel(r, "f(x) - min f", "log scale");
    Series S[NOPT];
    for (int k = 0; k < NOPT; k++) S[k] = (Series){O.lo[k], NULL, run.step + 1, P.opt[k], 0, OPT_NAME[k]};
    float ly = in.y0 + cap_h(st_small);
    for (int k = 0; k < NOPT; k++) legend(in.x0 + 44 * Uu + k * rw(in) * 0.22f, ly, OPT_NAME[k], P.opt[k], NULL);
    plot(rect(in.x0, in.y0 + 24 * Uu, in.x1, in.y1), S, NOPT, (float)O.nsteps, 1, -8, 0, "step");
}

/* ---------------- the frame ---------------- */
static const char *ACT_NUM[NACT] = {"I", "II", "III", "IV"};
static const char *ACT_TITLE[NACT] = {"Implicit neural representation", "Decision boundaries",
                                      "A character-level language model", "Optimisers"};
static const char *ACT_SUB[NACT] = {
    "a coordinate network learns one picture: (x, y) in, (r, g, b) out",
    "a small tanh network learns to separate two classes of points",
    "learning to write away-messages, one character at a time",
    "four ways downhill, racing on the same surface"};

static void fmt_away(char *o, size_t n, long s) {
    long m = s / 60, h = m / 60;
    if (h >= 24) snprintf(o, n, "away %ld d %ld h", h / 24, h % 24);
    else if (h >= 1) snprintf(o, n, "away %ld h %02ld min", h, m % 60);
    else if (m >= 1) snprintf(o, n, "away %ld min", m);
    else snprintf(o, n, "away under a minute");
}

static void stats_line(char *o, size_t n) {
    char st[24], tot[24], pr[24], lo[24];
    int act = run.act;
    fmt_int(st, sizeof st, run.step);
    fmt_int(tot, sizeof tot, phase_steps(act));
    if (act == A_FIT) {
        fmt_int(pr, sizeof pr, fit_params());
        fmt_val(lo, sizeof lo, F.loss[run.step]);
        snprintf(o, n, "step %s / %s" SEP "epoch %.1f" SEP "loss %s" SEP "lr %.1e" SEP "%s params", st, tot,
                 (double)run.step * BATCH1 / NPX, lo, F.lr, pr);
    } else if (act == A_CLS) {
        fmt_int(pr, sizeof pr, cls_params());
        fmt_val(lo, sizeof lo, C.loss[run.step]);
        snprintf(o, n, "step %s / %s" SEP "epoch %d" SEP "loss %s" SEP "lr %.1e" SEP "%s params", st, tot, run.step, lo, C.lr, pr);
    } else if (act == A_CHAR) {
        fmt_int(pr, sizeof pr, char_params());
        fmt_val(lo, sizeof lo, T.loss[run.step]);
        snprintf(o, n, "step %s / %s" SEP "epoch %.1f" SEP "loss %s" SEP "lr %.1e" SEP "%s params", st, tot,
                 (double)run.step * BATCH3 / T.n, lo, T.lr, pr);
    } else {
        snprintf(o, n, "step %s / %s" SEP "%s" SEP "start (%.2f, %.2f)", st, tot, FN_NAME[O.fn], O.tr[0][0], O.tr[0][1]);
    }
}

static void frame_chrome(Rect *content, double wt) {
    float m = 40 * Uu;
    int act = run.act;
    float y = m + cap_h(st_title);
    float x = m;
    float nw = text(st_title, x, y, -1, ACT_NUM[act], P.accent, 0) + 6 * Uu;
    text(st_title, x + nw + 14 * Uu, y, -1, ACT_TITLE[act], P.fg, 0);
    text(st_small, x + nw + 14 * Uu, y + 12 * Uu + cap_h(st_small), -1, ACT_SUB[act], P.dim, 0);
    if (role == R_FULL || role == R_FRONT) {
        time_t ts = (time_t)wt;
        struct tm tm;
        localtime_r(&ts, &tm);
        char b[32];
        strftime(b, sizeof b, "%H:%M", &tm);
        text(st_time, W - m, y, 1, b, P.fg, 0);
        fmt_away(b, sizeof b, (long)(wt - (double)locked_at));
        text(st_small, W - m, y + 12 * Uu + cap_h(st_small), 1, b, P.dim, 0);
    } else {
        text(st_small, W - m, y, 1, host, P.dim, 1 * Uu);
    }
    /* footer: the numbers, and where we are in the cycle */
    char line[256];
    stats_line(line, sizeof line);
    float fy = H - m * 0.75f;
    text(st_mono, m, fy, -1, line, P.dim, 0);
    float px = W - m;
    for (int k = NACT - 1; k >= 0; k--) {
        float w = text_width(st_cardk, ACT_NUM[k], 1.5f * Uu);
        text(st_cardk, px, fy, 1, ACT_NUM[k], k == act ? P.fg : ca(P.dim, 0.6f), 1.5f * Uu);
        if (k == act) {
            float f = clampf((float)(run.t_act / act_len), 0, 1);
            seg(px - w, fy + 6 * Uu, px - w + w * f, fy + 6 * Uu, 1.6f * Uu, P.accent);
        }
        px -= w + 16 * Uu;
    }
    *content = rect(m, m + cap_h(st_title) + 46 * Uu, W - m, H - m * 0.75f - cap_h(st_mono) - 24 * Uu);
}

/* the panels of the act on now, for this output's role and shape */
static void compose(Rect c) {
    int land = rw(c) >= rh(c);
    float g = 34 * Uu;
    Rect a, b, d, e, f, h;
    int act = run.act, rl = role;
    if (rl == R_SIDE) {
        if (land) hsplit(c, 0.5f, g, &a, &b); else vsplit(c, 0.5f, g, &a, &b);
        role = R_DIAG; compose(a);
        role = R_STATS; compose(b);
        role = R_SIDE;
        return;
    }
    if (act == A_FIT) {
        if (rl == R_FRONT) {
            vsplit(c, 0.76f, g, &a, &b);
            fit_pair(a, 1);
            fit_loss(b);
        } else if (rl == R_DIAG) {
            if (land) { hsplit(c, 0.5f, g, &a, &b); fit_net(a); fit_tiles(b, 1, 8); }
            else { vsplit(c, 0.55f, g, &a, &b); fit_net(a); fit_tiles(b, 1, 8); }
        } else if (rl == R_STATS) {
            if (land) { hsplit(c, 0.5f, g, &a, &b); fit_loss(a); fit_spec(b); }
            else { vsplit(c, 0.34f, g, &a, &b); fit_loss(a); vsplit(b, 0.45f, g, &d, &e); fit_spec(d); fit_tiles(e, 2, 8); }
        } else if (land) {
            hsplit(c, 0.6f, g * 1.5f, &a, &b);
            vsplit(a, 0.7f, g, &d, &e);
            fit_pair(d, 1);
            fit_tiles(e, 1, 16);
            vsplit(b, 0.5f, g, &d, &e);
            fit_loss(d);
            fit_spec(e);
        } else {
            vsplit(c, 0.42f, g, &a, &b);
            fit_pair(a, 1);
            vsplit(b, 0.36f, g, &d, &e);
            fit_loss(d);
            vsplit(e, 0.5f, g, &f, &h);
            fit_spec(f);
            fit_tiles(h, 1, 16);
        }
    } else if (act == A_CLS) {
        if (rl == R_FRONT) {
            if (land) { hsplit(c, 0.62f, g, &a, &b); cls_field(a); cls_loss(b); }
            else { vsplit(c, 0.7f, g, &a, &b); cls_field(a); cls_loss(b); }
        } else if (rl == R_DIAG) cls_net(c);
        else if (rl == R_STATS) {
            if (land) { hsplit(c, 0.55f, g, &a, &b); cls_tiles(a); cls_loss(b); }
            else { vsplit(c, 0.55f, g, &a, &b); cls_tiles(a); cls_loss(b); }
        } else if (land) {
            hsplit(c, 0.5f, g * 1.5f, &a, &b);
            cls_field(a);
            vsplit(b, 0.64f, g, &d, &e);
            cls_net(d);
            cls_loss(e);
        } else {
            vsplit(c, 0.44f, g, &a, &b);
            cls_field(a);
            vsplit(b, 0.62f, g, &d, &e);
            cls_net(d);
            cls_loss(e);
        }
    } else if (act == A_CHAR) {
        if (rl == R_FRONT) { vsplit(c, 0.36f, g, &a, &b); char_type(a); char_log(b); }
        else if (rl == R_DIAG) {
            if (land) { hsplit(c, 0.55f, g, &a, &b); char_emb(a); char_weights(b); }
            else { vsplit(c, 0.6f, g, &a, &b); char_emb(a); char_weights(b); }
        } else if (rl == R_STATS) {
            if (land) { hsplit(c, 0.5f, g, &a, &b); char_loss(a); char_log(b); }
            else { vsplit(c, 0.4f, g, &a, &b); char_loss(a); char_log(b); }
        } else if (land) {
            hsplit(c, 0.6f, g * 1.5f, &a, &b);
            vsplit(a, 0.34f, g, &d, &e);
            char_type(d);
            char_log(e);
            vsplit(b, 0.45f, g, &d, &e);
            char_loss(d);
            char_emb(e);
        } else {
            vsplit(c, 0.3f, g, &a, &b);
            char_type(a);
            vsplit(b, 0.38f, g, &d, &e);
            char_log(d);
            vsplit(e, 0.42f, g, &f, &h);
            char_loss(f);
            char_emb(h);
        }
    } else {
        if (rl == R_FRONT) opt_view(c, 0);
        else if (rl == R_DIAG) {
            if (land) opt_view(c, 1);
            else { vsplit(c, 0.62f, g, &a, &b); opt_view(a, 1); opt_loss(b); }
        }
        else if (rl == R_STATS) opt_loss(c);
        else if (land) {
            hsplit(c, 0.6f, g * 1.5f, &a, &b);
            opt_view(a, 0);
            vsplit(b, 0.58f, g, &d, &e);
            opt_view(d, 1);
            opt_loss(e);
        } else {
            vsplit(c, 0.45f, g, &a, &b);
            opt_view(a, 0);
            vsplit(b, 0.55f, g, &d, &e);
            opt_view(d, 1);
            opt_loss(e);
        }
    }
}

/* the title card at the start of an act */
static void title_card(float al) {
    if (al <= 0.001f) return;
    int act = run.act;
    float cx = W / 2.0f, cy = H / 2.0f;
    float tw = text_width(st_card, ACT_TITLE[act], 0), sw = text_width(st_body, ACT_SUB[act], 0);
    float bw = fmaxf(tw, sw) + 120 * Uu, bh = 200 * Uu;
    rect_fill(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 10 * Uu, ca(P.panel, 0.92f * al));
    rect_line(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 1 * Uu, ca(P.grid, al));
    char k[32];
    snprintf(k, sizeof k, "ACT %s", ACT_NUM[act]);
    text(st_cardk, cx, cy - 46 * Uu, 0, k, ca(P.accent, al), 3 * Uu);
    text(st_card, cx, cy + 18 * Uu, 0, ACT_TITLE[act], ca(P.fg, al), 0);
    seg(cx - 24 * Uu, cy + 38 * Uu, cx + 24 * Uu, cy + 38 * Uu, 1.5f * Uu, ca(P.accent, al));
    text(st_body, cx, cy + 70 * Uu, 0, ACT_SUB[act], ca(P.dim, al), 0);
}

static void render(double wt, GLuint target) {
    if (styled_w != W || styled_h != H) layout_styles();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    upload();
    draw_t = run.t_act;
    GLuint p = prog_bg;
    glUseProgram(p);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform3fv(U(p, "uBg"), 1, P.bg);
    glUniform3fv(U(p, "uGrid"), 1, P.grid);
    glUniform1f(U(p, "uU"), Uu);
    glUniform1f(U(p, "uLight"), (float)P.light);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    Rect c;
    frame_chrome(&c, wt);
    compose(c);
    flush();
    /* fades: in after the card, out at the end, a blink between phases */
    float t = (float)run.t_act;
    float vis = smooth(T_TRAIN - 1.6f, T_TRAIN + 0.2f, t) * (1 - smooth((float)act_len - 1.2f, (float)act_len, t));
    if (nphase(run.act) > 1) {
        float pb = T_TRAIN + (float)phase_len(run.act);
        vis *= 1 - 0.85f * (1 - smooth(0, 0.5f, fabsf(t - pb)));
    }
    if (vis < 1) rect_fill(-2, -2, W + 2, H + 2, 0, ca(P.bg, 1 - vis));
    float card = smooth(0.3f, 1.0f, t) * (1 - smooth(T_TRAIN - 1.0f, T_TRAIN, t));
    title_card(card);
    flush();
}

/* ======================================================================
 * Checks: finite differences against every backward pass, and full runs
 * ====================================================================== */
typedef struct { float *p, *g; } PG;

/* compare the analytic gradient g (saved) with central differences of loss() at k params */
static void fd_report(const char *name, PG *pg, int n, float (*loss)(void), float eps) {
    double worst = 0, sum = 0;
    int cnt = 0;
    float gmax = 0;
    for (int i = 0; i < n; i++) gmax = fmaxf(gmax, fabsf(*pg[i].g));
    for (int i = 0; i < n; i++) {
        float a = *pg[i].g;
        if (fabsf(a) < gmax * 1e-2f) continue;
        float old = *pg[i].p;
        *pg[i].p = old + eps;
        double lp = loss();
        *pg[i].p = old - eps;
        double lm = loss();
        *pg[i].p = old;
        double num = (lp - lm) / (2 * eps);
        double rel = fabs(num - a) / fmax(fabs(num) + fabs(a), 1e-12);
        worst = fmax(worst, rel);
        sum += rel;
        cnt++;
    }
    printf("gradcheck %-28s %3d params: mean rel err %.2e, worst %.2e\n", name, cnt, cnt ? sum / cnt : 0, worst);
}

static float gc_fit_B;
static float gc_fit(void) { for (int l = 0; l < F.nl; l++) dense_zero(&F.L[l]); return fit_grad((int)gc_fit_B); }
static float gc_cls(void) { for (int l = 0; l < NL2; l++) dense_zero(&C.L[l]); return cls_grad(NULL); }
static float gc_char(void) { dense_zero(&T.L1); dense_zero(&T.L2); memset(T.gE, 0, sizeof T.gE); return char_grad(); }
static int gc_fn;
static double gc_xy[2];

/* pick k params (with their gradient copies) from a dense layer */
static int gc_pick(Dense *L, PG *pg, float *store, int k, Rng *r) {
    int n = L->in * L->out;
    for (int i = 0; i < k; i++) {
        int j = (int)(rnd_u(r) % (uint32_t)(n + L->out));
        if (j < n) { pg[i].p = &L->W[j]; store[i] = L->gW[j]; }
        else { pg[i].p = &L->b[j - n]; store[i] = L->gb[j - n]; }
        pg[i].g = &store[i];
    }
    return k;
}

static void gradcheck(void) {
    static PG pg[512];
    static float st[512];
    Rng r = rng_new(1234, 5678);
    struct tm tm = {0};
    tm.tm_hour = 10; tm.tm_min = 9;
    for (int model = 0; model < 2; model++) {
        fit_init((uint32_t)(model << 8 | PIC_CLOCK), &tm);
        /* a few steps in, so it isn't the init's special case */
        for (int t = 1; t <= 20; t++) fit_step(t);
        gc_fit_B = 32;
        fit_batch(&r, 32);
        gc_fit();
        int n = 0;
        for (int l = 0; l < F.nl; l++) n += gc_pick(&F.L[l], pg + n, st + n, 30, &r);
        fd_report(model == M_SIREN ? "I   SIREN (w0 = 30)" : "I   Fourier features + ReLU", pg, n, gc_fit,
                  model == M_SIREN ? 2e-4f : 2e-3f);
    }
    cls_init(0, 0);
    for (int t = 1; t <= 50; t++) cls_step(t);
    gc_cls();
    int n = 0;
    for (int l = 0; l < NL2; l++) n += gc_pick(&C.L[l], pg + n, st + n, 30, &r);
    fd_report("II  tanh MLP + BCE", pg, n, gc_cls, 1e-2f);
    char_init(0);
    for (int t = 1; t <= 50; t++) char_step(t);
    char_batch(&r);
    gc_char();
    n = gc_pick(&T.L1, pg, st, 40, &r);
    n += gc_pick(&T.L2, pg + n, st + n, 40, &r);
    for (int i = 0; i < 40; i++) {
        int j = char_ctx[rnd_u(&r) % (BATCH3 * CTX)] * EMB + (int)(rnd_u(&r) % EMB);
        pg[n].p = &T.E[j]; st[n] = T.gE[j]; pg[n].g = &st[n]; n++;
    }
    fd_report("III char MLP + softmax CE", pg, n, gc_char, 1e-2f);
    for (gc_fn = 0; gc_fn < NFN; gc_fn++) {
        double worst = 0;
        for (int i = 0; i < 50; i++) {
            double x = rnd_s(&r) * 2, y = rnd_s(&r) * 2, g[2];
            fn_f(gc_fn, x, y, &g[0], &g[1]);
            for (int d = 0; d < 2; d++) {
                double e = 1e-6;
                gc_xy[0] = x + (d == 0) * e; gc_xy[1] = y + (d == 1) * e;
                double lp = fn_f(gc_fn, gc_xy[0], gc_xy[1], NULL, NULL);
                gc_xy[0] = x - (d == 0) * e; gc_xy[1] = y - (d == 1) * e;
                double lm = fn_f(gc_fn, gc_xy[0], gc_xy[1], NULL, NULL);
                double num = (lp - lm) / (2 * e);
                worst = fmax(worst, fabs(num - g[d]) / fmax(fabs(num) + fabs(g[d]), 1e-9));
            }
        }
        printf("gradcheck IV  %-24s analytic vs numeric: worst rel err %.2e\n", FN_NAME[gc_fn], worst);
    }
}

/* run every act's phases to the end, printing how training went */
static void converge(void) {
    for (int act = 0; act < NACT; act++) {
        if (opt_act >= 0 && act != opt_act) continue;
        int nv = act == A_FIT ? 2 * NPIC : act == A_CLS ? 4 : act == A_OPT ? 2 : 1;
        for (int v = 0; v < nv; v++)
            for (int ph = 0; ph < nphase(act); ph++) {
                uint32_t var = act == A_FIT ? (uint32_t)((v / NPIC) << 8 | (v % NPIC))
                             : act == A_CLS ? (uint32_t)(v % 2 | (v / 2) << 4) : (uint32_t)(v << 5);
                if (act == A_CLS && ph == 0 && v >= 2) continue;
                run.act = act; run.phase = ph; run.step = 0;
                struct tm tm = {0};
                tm.tm_hour = 10; tm.tm_min = 9; tm.tm_sec = 30;
                if (act == A_FIT) fit_init(var, &tm);
                else if (act == A_CLS) cls_init(var, ph);
                else if (act == A_CHAR) char_init(var);
                else opt_init(var, ph);
                int n = phase_steps(act);
                double c0 = cpu_time();
                char line[512] = "";
                size_t L = 0;
                for (int t = 1; t <= n; t++) {
                    run.step = t;
                    if (act == A_FIT) fit_step(t);
                    else if (act == A_CLS) cls_step(t);
                    else if (act == A_CHAR) char_step(t);
                    else opt_step(t);
                    if (t == n / 10 || t == n / 4 || t == n / 2 || t == n) {
                        if (act == A_FIT) L += (size_t)snprintf(line + L, sizeof line - L, " @%d psnr %.1f", t, F.psnr[F.nsnap - 1]);
                        else if (act == A_CLS) L += (size_t)snprintf(line + L, sizeof line - L, " @%d loss %.4f acc %.3f", t, C.loss[t], C.acc[t]);
                        else if (act == A_CHAR) {
                            float m = 0;
                            for (int k = t - 49; k <= t; k++) m += T.loss[k] / 50;
                            L += (size_t)snprintf(line + L, sizeof line - L, " @%d loss %.3f", t, m);
                        } else {
                            L += (size_t)snprintf(line + L, sizeof line - L, " @%d", t);
                            for (int k = 0; k < NOPT; k++) L += (size_t)snprintf(line + L, sizeof line - L, " %s %.2g", OPT_NAME[k], O.lo[k][t]);
                        }
                    }
                }
                double ms = (cpu_time() - c0) * 1000 / n;
                const char *what = act == A_FIT ? (F.model == M_SIREN ? "SIREN" : "Fourier") : act == A_CLS ? DS_NAME[C.ds]
                                 : act == A_CHAR ? "char" : FN_NAME[O.fn];
                printf("act %-3s %-10s %-14s %.2f ms/step = %4.1f%% of a core:%s\n", ACT_NUM[act],
                       act == A_FIT ? PIC_NAME[F.pic] + 0 : "", what, ms, ms * SPS[act] / 10, line);
                if (act == A_CHAR)
                    for (int k = T.nsmp - 3; k < T.nsmp; k++) printf("    ep %5.1f  %s\n", T.smp[k].epoch, T.smp[k].txt);
            }
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
    PX = (float)hidpi_factor(&hd, W);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "neural");
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
                                      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 0,
                                      EGL_DEPTH_SIZE, 24, EGL_NONE};
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
        "usage: neural [--window | --root | --layer] [--seed N] [--palette spectral|mono|paper|terminal]\n"
        "              [--act 1-4] [--act-len SECONDS] [--fps N] [--budget MS] [--name NAME]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--render-scale F (--layer: draw F x the output's native pixels, 0 < F <= 1; default 1)]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--soak SECONDS] [--size WxH]\n"
        "              [--picture clock|host|sunset|zone] [--model siren|fourier]\n"
        "              [--gradcheck] [--converge]\n");
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

/* Spanning, the output under the middle of the layout is the front one
 * (flightsim's rule) and gets the main view; the rest, left to right, take
 * the structure and the numbers in turn (with just two outputs, the other
 * one gets both). */
static int pick_role(void) {
    if (span_me < 0 || span_n < 2) return R_FULL;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    if (span_me == front) return R_FRONT;
    if (span_n == 2) return R_SIDE;
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == front || i == span_me) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && (span[i][1] < span[span_me][1] ||
                                               (span[i][1] == span[span_me][1] && i < span_me)))) rank++;
    }
    return rank % 2 ? R_STATS : R_DIAG;
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
    double opt_soak = 0;
    int opt_converge = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-name") && i + 1 < argc) opt_name = argv[++i];
        else if (!strcmp(a, "-act") && i + 1 < argc) opt_act = atoi(argv[++i]) - 1;
        else if (!strcmp(a, "-act-len") && i + 1 < argc) act_len = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-budget") && i + 1 < argc) opt_budget = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-soak") && i + 1 < argc) opt_soak = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-picture") && i + 1 < argc) { const char *v = argv[++i]; opt_pic = !strcmp(v, "clock") ? 0 : !strcmp(v, "host") ? 1 : !strcmp(v, "sunset") ? 2 : !strcmp(v, "zone") ? 3 : -1; }
        else if (!strcmp(a, "-model") && i + 1 < argc) { const char *v = argv[++i]; opt_model = !strcmp(v, "siren") ? 0 : !strcmp(v, "fourier") ? 1 : -1; }
        else if (!strcmp(a, "-gradcheck")) opt_gradcheck = 1;
        else if (!strcmp(a, "-converge")) opt_converge = 1;
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    role = pick_role();
    if (opt_act >= NACT) opt_act = NACT - 1;
    if (act_len < 30) act_len = 30;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    pal_set(opt_palette);
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    {
        const char *u = getenv("USER");
        snprintf(login, sizeof login, "%s", u && *u ? u : "user");
    }
    fonts_init();
    if (opt_gradcheck || opt_converge) {
        if (opt_gradcheck) gradcheck();
        if (opt_converge) converge();
        return 0;
    }
    int offscreen = opt_shot || opt_bench > 0 || opt_soak > 0;

    /* --epoch: when the first act starts, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    t_wall0 = opt_epoch ? opt_epoch : wall();
    if (offscreen && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)(opt_epoch ? opt_epoch : wall() - (opt_shot ? opt_at : 0));

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offscreen) wl_init();
    else {
        dpy = XOpenDisplay(NULL);
        if (!dpy) die("can't open display");
        int scr = DefaultScreen(dpy);
        root = RootWindow(dpy, scr);
        int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                       GLX_DEPTH_SIZE, 24, None};
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
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Neural");
        XClassHint chint = {"neural", "Neural"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        /* a 3.3 core context if we can get one */
        GLXContext ctx = NULL;
        typedef GLXContext (*CCA)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
        CCA cca = (CCA)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
        int nfb = 0;
        int fbattr[] = {GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_DOUBLEBUFFER, True, GLX_DEPTH_SIZE, 24, None};
        GLXFBConfig *fbc = cca ? glXChooseFBConfig(dpy, scr, fbattr, &nfb) : NULL;
        if (fbc && nfb > 0) {
            int cattr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                           GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None};
            for (int i = 0; i < nfb && !ctx; i++) {
                XVisualInfo *fv = glXGetVisualFromFBConfig(dpy, fbc[i]);
                if (fv && fv->visualid == vi->visualid) ctx = cca(dpy, fbc[i], NULL, True, cattr);
                if (fv) XFree(fv);
            }
        }
        if (!ctx) ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

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
        if (opt_shot) {
            double c0 = cpu_time();
            advance(opt_at, -1);
            fprintf(stderr, "neural: act %s, step %d, replayed in %.1f s of CPU\n", ACT_NUM[run.act], run.step, cpu_time() - c0);
            render(t_wall0 + opt_at, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        if (opt_soak > 0) {
            /* the live loop, offscreen and in real time: what it costs */
            double t0 = now(), c0 = cpu_time(), e0 = opt_at;
            int frames = 0;
            while (now() - t0 < opt_soak) {
                double t = now();
                advance(e0 + (t - t0), opt_budget);
                render(t_wall0 + e0 + (t - t0), fbo);
                glFinish();
                frames++;
                double spare = 1.0 / opt_fps - (now() - t);
                if (spare > 0) { struct timespec ts = {0, (long)(spare * 1e9)}; nanosleep(&ts, NULL); }
            }
            double el = now() - t0;
            printf("neural: soak %.0f s at %dx%d, act %s: %.1f fps, %.1f%% of a core\n", el, W, H, ACT_NUM[run.act],
                   frames / el, (cpu_time() - c0) / el * 100);
            return 0;
        }
        for (int act = 0; act < NACT; act++) {
            if (opt_act >= 0 && act != opt_act) continue;
            int keep = opt_act;
            opt_act = act;
            run.live = 0;
            double e = act_len * 0.55, c0 = cpu_time();
            advance(e, -1);
            double ms_step = (cpu_time() - c0) * 1000 / fmax(run.step, 1);
            render(t_wall0 + e, fbo);
            glFinish();
            double sum = 0, worst = 0, csum = 0;
            for (int i = 0; i < opt_bench; i++) {
                double f0 = now(), cc = cpu_time();
                render(t_wall0 + e + i / 60.0, fbo);
                glFinish();
                double ms = (now() - f0) * 1000;
                csum += (cpu_time() - cc) * 1000;
                sum += ms;
                if (ms > worst) worst = ms;
            }
            printf("neural: act %-3s %dx%d, %d frames: avg %.3f ms, worst %.3f ms per frame (CPU %.2f ms); "
                   "training %.2f ms/step x %.0f/s = %.1f%% of a core\n",
                   ACT_NUM[act], W, H, opt_bench, sum / opt_bench, worst, csum / opt_bench, ms_step, SPS[act], ms_step * SPS[act] / 10);
            opt_act = keep;
        }
        printf("neural: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
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
        double t = now(), wt = wall(), el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        advance(el, opt_budget);
        render(wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
