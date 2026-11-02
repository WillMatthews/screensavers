/*
 * algorithms — a theatre of algorithms at work: sorts and mazes, played as a
 * programme of acts, each with a small title card and live counters.
 *
 * SORTING. Twenty-odd sorts (insertion, selection, bubble, cocktail shaker,
 * gnome, odd-even transposition, cycle, pancake, Shell, comb, merge top-down
 * and bottom-up, quicksort with Lomuto and Hoare partitions, heap, radix LSD
 * and MSD, counting, a timsort, Batcher's bitonic and odd-even merge networks,
 * and now and then a bogosort that gives up) run on inputs of different shapes
 * (random, nearly sorted, reversed, few unique, sawtooth) and are shown in one
 * of several forms:
 *   bars      the classic; horizontal on a portrait screen
 *   dots      index against value, a scatter that collapses onto the diagonal
 *   wheel     a disparity circle: hue is the value, the radius how close it
 *             is to home; sorting settles a spiral into a perfect ring
 *   image     a picture whose stripes were shuffled, sorted back together
 *   network   a sorting network drawn as wires and comparators, the wires
 *             coloured by the values they carry, braiding into order
 *   heap      heapsort with its binary heap drawn as a tree over the bars
 * Comparisons and swaps flash briefly; the counters below count comparisons,
 * swaps, and reads and writes of the array.
 *
 * MAZES. A generator carves a maze on a square, hex or circular (theta) grid
 * (recursive backtracker, Prim's, Kruskal's with its union-find sets
 * coloured, Wilson's loop-erased walks, Aldous-Broder, Eller's row by row,
 * binary tree, sidewinder, recursive division, hunt-and-kill, growing tree),
 * the passages coloured by when they were cut; then a solver runs (BFS
 * flood-fill coloured by distance, DFS, A*, Dijkstra, wall follower, dead-end
 * filling) and the path is drawn as a glowing line.
 *
 * Every algorithm is run once, up front, into a log of events (compare,
 * swap, write, carve, visit...), and the log is played back on a pacing curve
 * that starts slow enough to follow each step and accelerates. Which act runs
 * when is a function of the seed and the time since --epoch, so --at can show
 * any moment by replaying, and instances on several monitors agree without
 * talking.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index. The
 * output under the middle of the layout is the front one (it keeps the clock).
 * Acts then alternate between
 *   a race      the same input sorted by a different algorithm on each
 *               screen, all at the same step rate, so they finish in turn
 *   a triptych  one sort shown in a different form on each screen
 *   a maze      one maze over the whole wall, carved and then solved across
 *               the bezels
 *
 * Everything is drawn as signed-distance quads (crisp at any size) and
 * freetype text (Inter and a mono, from fontconfig). Palettes: spectral,
 * mono, terminal, paper (light), sunset, viridis (--palette NAME, or one is
 * picked from the seed).
 *
 * Same plumbing as gauges and flightsim: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen,
 * --bench N to time N frames. --check verifies every algorithm and exits.
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

#define PI 3.14159265358979f
#define TAU (2 * PI)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check, opt_list;
static float opt_fps = 60;
static const char *opt_font = "Inter";
static const char *opt_mono = "JetBrainsMono Nerd Font Mono,JetBrains Mono,DejaVu Sans Mono";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 20;             /* --shot: the moment this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the programme started at */
static const char *opt_palette;
static double opt_speed = 1;
/* forcing what plays, for previews and tests */
static const char *opt_kind, *opt_sort, *opt_form, *opt_input, *opt_maze, *opt_solve, *opt_grid;
static int opt_n;

/* spanning */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
static int front;                      /* the output under the middle of the layout */
static int out_rank[MAX_OUT];          /* left-to-right order of each output, front first */

static void die(const char *what) { fprintf(stderr, "algorithms: %s\n", what); exit(1); }

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
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }

/* ---- deterministic randomness ---- */
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) { return hash32(a ^ hash32(b ^ hash32(c + 0x9e3779b9u))); }

typedef struct { uint64_t s; } Rng;
static uint32_t rnext(Rng *r) {
    uint64_t z = (r->s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}
static int rint_(Rng *r, int n) { return n > 0 ? (int)(((uint64_t)rnext(r) * (uint64_t)n) >> 32) : 0; }
static Rng rng_of(uint32_t a, uint32_t b) { Rng r = {((uint64_t)hash3(a, b, 17) << 32) | hash3(b, a, 91)}; return r; }

/* ---- the lock ---- */
static time_t locked_at;

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

/* ---- palettes ---- */
typedef struct {
    const char *name;
    int light;
    const char *bg, *bg2, *fg, *dim, *faint, *hi, *hot, *path;
    int nramp;
    const char *ramp[9];
} PalDef;

static const PalDef PALS[] = {
    {"spectral", 0, "#0b0d12", "#151a24", "#eef0f5", "#8a92a4", "#262c38", "#ffffff", "#ff4f6d", "#fff4cc",
     9, {"#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#6a5acd"}},
    {"mono", 0, "#0d0d0f", "#18181c", "#f2f2f2", "#8c8c94", "#2a2a30", "#ffb03a", "#ff5a1f", "#ff7a3a",
     4, {"#55575f", "#8d8f99", "#c8cad2", "#ffffff"}},
    {"terminal", 0, "#030805", "#08140c", "#c4ffd2", "#4e9a64", "#11291a", "#f4fff6", "#ffc94a", "#d8ffe2",
     5, {"#125a2a", "#1d8a3c", "#2ec257", "#63ff8c", "#d2ffdc"}},
    {"paper", 1, "#f3efe6", "#faf7f1", "#1d1e22", "#77736b", "#dcd5c6", "#2f5fb3", "#d62828", "#d62828",
     6, {"#e9c46a", "#f08a4b", "#d1495b", "#8f3f71", "#45417a", "#1f2f4a"}},
    {"sunset", 0, "#0f0a16", "#1b1226", "#fdf2e6", "#a08fae", "#2a1f37", "#ffffff", "#ffe08a", "#fff0c4",
     6, {"#5a1d8a", "#912b84", "#c93d72", "#f0645c", "#fb9f6b", "#fdd9a0"}},
    {"viridis", 0, "#070d16", "#0f1a28", "#eef4f8", "#7f94a8", "#1c2a3a", "#ffffff", "#ff6b5a", "#fff8c8",
     7, {"#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#6ece58", "#fde725"}},
};
#define NPAL ((int)(sizeof PALS / sizeof PALS[0]))

static struct {
    int light;
    float bg[3], bg2[3], fg[3], dim[3], faint[3], hi[3], hot[3], path[3];
    int nramp;
    float ramp[9][3];
    const char *name;
} pal;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

static void choose_palette(void) {
    int k = -1;
    for (int i = 0; i < NPAL; i++)
        if (opt_palette && !strcmp(opt_palette, PALS[i].name)) k = i;
    if (opt_palette && k < 0) {
        fprintf(stderr, "algorithms: no palette '%s'; one of:", opt_palette);
        for (int i = 0; i < NPAL; i++) fprintf(stderr, " %s", PALS[i].name);
        fprintf(stderr, "\n");
        exit(1);
    }
    if (k < 0) k = (int)(hash3(opt_seed, 3, 5) % NPAL);
    const PalDef *p = &PALS[k];
    pal.name = p->name;
    pal.light = p->light;
    hex(p->bg, pal.bg); hex(p->bg2, pal.bg2); hex(p->fg, pal.fg); hex(p->dim, pal.dim);
    hex(p->faint, pal.faint); hex(p->hi, pal.hi); hex(p->hot, pal.hot); hex(p->path, pal.path);
    pal.nramp = p->nramp;
    for (int i = 0; i < p->nramp; i++) hex(p->ramp[i], pal.ramp[i]);
}

/* the value ramp at t in [0, 1] */
static void ramp(float t, float *o) {
    t = clampf(t, 0, 1) * (pal.nramp - 1);
    int i = (int)t;
    if (i >= pal.nramp - 1) i = pal.nramp - 2;
    float f = t - i;
    for (int c = 0; c < 3; c++) o[c] = pal.ramp[i][c] + (pal.ramp[i + 1][c] - pal.ramp[i][c]) * f;
}

static void mix3(const float *a, const float *b, float t, float *o) {
    for (int c = 0; c < 3; c++) o[c] = a[c] + (b[c] - a[c]) * t;
}

/* ---- text: a freetype glyph atlas ---- */
#define NEXTRA 22
static const unsigned extra_cp[NEXTRA] = {0xB7, 0xD7, 0x2192, 0x2013, 0x2014, 0x2026, 0x3A3, 0x2248, 0x2264,
                                          0xB0, 0x2022, 0x221E, 0x2032, 0x2033, 0x2009, 0xBD, 0x2191, 0x2193,
                                          0x2190, 0x2265, 0x2019, 0xA0};
#define NGLYPH (95 + NEXTRA)
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGLYPH]; } Style;
#define MAXSTYLE 32
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
enum { F_REG, F_SEMI, F_LIGHT, F_MONO, NFACE };
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        char pat_s[300];
        static const char *wt[NFACE] = {"regular", "semibold", "light", "regular"};
        snprintf(pat_s, sizeof pat_s, "%s:weight=%s:fontformat=CFF,TrueType", i == F_MONO ? opt_mono : opt_font, wt[i]);
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

static unsigned glyph_cp(int i) { return i < 95 ? (unsigned)(32 + i) : extra_cp[i - 95]; }

static void build_atlas(void) {
    free(atlas);
    int cap = 4096;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < NGLYPH; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            if (FT_Load_Char(f, glyph_cp(c), FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
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

/* next code point of a UTF-8 string */
static unsigned utf8(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    unsigned c = *s++;
    if (c >= 0xF0 && s[0] && s[1] && s[2]) { c = (c & 7) << 18 | (s[0] & 63) << 12 | (s[1] & 63) << 6 | (s[2] & 63); s += 3; }
    else if (c >= 0xE0 && s[0] && s[1]) { c = (c & 15) << 12 | (s[0] & 63) << 6 | (s[1] & 63); s += 2; }
    else if (c >= 0xC0 && s[0]) { c = (c & 31) << 6 | (s[0] & 63); s += 1; }
    *p = (const char *)s;
    return c;
}

static const Glyph *glyph(int s, unsigned c) {
    if (c >= 32 && c < 127) return &styles[s].g[c - 32];
    for (int i = 0; i < NEXTRA; i++)
        if (extra_cp[i] == c) return &styles[s].g[95 + i];
    return &styles[s].g[0];
}

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p;) {
        unsigned c = utf8(&p);
        w += glyph(s, c)->adv + (*p ? track : 0);
    }
    return w;
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
 * col is rgb + alpha */
static float text(int s, float x, float y, int align, const char *str, const float *rgb, float alpha, float track) {
    if (alpha <= 0.003f) return 0;
    float col[4] = {rgb[0], rgb[1], rgb[2], alpha};
    float w = text_width(s, str, track);
    float pen = roundf(x - (align == 0 ? w * 0.5f : align > 0 ? w : 0));
    y = roundf(y);
    for (const char *p = str; *p;) {
        const Glyph *g = glyph(s, utf8(&p));
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

/* 12345 -> "12,345" */
static void commas(char *o, size_t n, long v) {
    char b[32];
    snprintf(b, sizeof b, "%ld", v < 0 ? -v : v);
    int len = (int)strlen(b), j = 0;
    if (v < 0 && j < (int)n - 1) o[j++] = '-';
    for (int i = 0; i < len && j < (int)n - 1; i++) {
        o[j++] = b[i];
        if ((len - 1 - i) % 3 == 0 && i < len - 1 && j < (int)n - 1) o[j++] = ',';
    }
    o[j] = 0;
}

/* ---- SDF primitives: instanced quads ----
 * type 0: a box along a segment, its ends extended by the half-width (a
 *         capsule when the corner radius equals the half-width; a disc or a
 *         rounded square when both ends coincide)
 * type 1: an arc band: centre, radius, angles a0 < a1, half-width
 * type 2: a box along a segment with flat, unextended ends
 * colours run from ca at the start to cb at the end; glow adds a soft halo */
typedef struct { float a[4], b[4]; unsigned char ca[4], cb[4]; } Prim;
static Prim *pv;
static int pn, pcap;
static void flush_prims(void);

static void ubc(unsigned char *o, const float *c, float a) {
    a = clampf(a, 0, 1);
    for (int i = 0; i < 3; i++) o[i] = (unsigned char)lroundf(clampf(c[i], 0, 1) * 255);
    o[3] = (unsigned char)lroundf(a * 255);
}

static Prim *prim_new(void) {
    if (pn == pcap) pv = realloc(pv, (size_t)(pcap = pcap ? pcap * 2 : 16384) * sizeof *pv);
    return &pv[pn++];
}

static void seg2(float x0, float y0, float x1, float y1, float w, float r, const float *c0, float a0,
                 const float *c1, float a1, float glow, int flat) {
    if (a0 <= 0.002f && a1 <= 0.002f && glow <= 0) return;
    Prim *p = prim_new();
    p->a[0] = x0; p->a[1] = y0; p->a[2] = x1; p->a[3] = y1;
    p->b[0] = r; p->b[1] = w; p->b[2] = flat ? 2 : 0; p->b[3] = glow;
    ubc(p->ca, c0, a0);
    ubc(p->cb, c1, a1);
}
static void seg(float x0, float y0, float x1, float y1, float w, const float *c, float a, float glow) {
    seg2(x0, y0, x1, y1, w, w, c, a, c, a, glow, 0);
}
static void disc(float x, float y, float r, const float *c, float a, float glow) {
    seg2(x, y, x, y, r, r, c, a, c, a, glow, 0);
}
static void box(float x0, float y0, float x1, float y1, float r, const float *c, float a) {
    /* an axis-aligned rect as a flat segment along its longer side */
    if (x1 - x0 >= y1 - y0) seg2(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2, (y1 - y0) / 2, r, c, a, c, a, 0, 1);
    else seg2((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, (x1 - x0) / 2, r, c, a, c, a, 0, 1);
}
static void arc2(float cx, float cy, float R, float a0, float a1, float w, const float *c0, float al0,
                 const float *c1, float al1, float glow) {
    if (al0 <= 0.002f && al1 <= 0.002f && glow <= 0) return;
    Prim *p = prim_new();
    p->a[0] = cx; p->a[1] = cy; p->a[2] = R; p->a[3] = a0;
    p->b[0] = a1; p->b[1] = w; p->b[2] = 1; p->b[3] = glow;
    ubc(p->ca, c0, al0);
    ubc(p->cb, c1, al1);
}

/* ---- event logs ---- */
typedef struct { int a, b; unsigned char k; } Ev;
typedef struct {
    Ev *e;
    int n, cap;
    int *tp;                           /* index of the k-th timed event */
    int nt;
} Log;
static Log *rec;                       /* the log being recorded into */

/* sorting events; those from E_PIVOT on are markers and take no time */
enum { E_CMP, E_CMPV, E_CMPX, E_SWAP, E_WRITE, E_READ, E_MOVE,
       E_PIVOT, E_RANGE, E_HEAPN, E_MARK, E_NOTE };
/* maze events: generation, then solving; from M_UNTIMED on they take no time */
enum { M_LINK = 32, M_UNLINK, M_HEAD, M_STATE, M_WPUSH, M_WPOP, M_VISIT, M_OPEN, M_CLOSE, M_PUSH, M_POP,
       M_FILL, M_STEP,
       M_UNTIMED, M_SETID = M_UNTIMED, M_RECT, M_ROW, M_WCLEAR, M_PARENT };
enum { NOTE_GAP, NOTE_PASS, NOTE_RUNS, NOTE_SHUF, NOTE_GIVEUP, NOTE_LAYER, NOTE_WIDTH, NOTE_SORTED, NNOTE };

static int timed(int k) { return k < E_PIVOT || (k >= M_LINK && k < M_UNTIMED); }

static void emit(int k, int a, int b) {
    Log *L = rec;
    if (!L) return;
    if (L->n == L->cap) L->e = realloc(L->e, (size_t)(L->cap = L->cap ? L->cap * 2 : 4096) * sizeof *L->e);
    L->e[L->n++] = (Ev){a, b, (unsigned char)k};
}

static void log_finish(Log *L) {
    free(L->tp);
    L->nt = 0;
    for (int i = 0; i < L->n; i++) L->nt += timed(L->e[i].k);
    L->tp = malloc((size_t)(L->nt + 1) * sizeof *L->tp);
    int k = 0;
    for (int i = 0; i < L->n; i++)
        if (timed(L->e[i].k)) L->tp[k++] = i;
    L->tp[k] = L->n;
}

static void log_free(Log *L) { free(L->e); free(L->tp); memset(L, 0, sizeof *L); }

/* ---- pacing: a slow start ramping exponentially to a cruising rate,
 * solved so that S steps take T seconds ---- */
typedef struct { double r0, Tr, k, rc, A, T; long S; } Pace;

static double pace_area(double r0, double Tr, double T, double rc) {
    return (rc - r0) * Tr / log(rc / r0) + rc * (T - Tr);
}

static void pace_init(Pace *p, long S, double T, double r0) {
    memset(p, 0, sizeof *p);
    p->S = S; p->T = T; p->r0 = r0;
    if (S <= 0 || T <= 0) { p->rc = 1; return; }
    p->Tr = fmin(11.0, T * 0.32);
    if (S <= r0 * T * 1.15) { p->Tr = 0; p->rc = S / T; return; }
    double lo = r0 * 1.0001, hi = (double)S * 2;
    for (int i = 0; i < 80; i++) {
        double mid = sqrt(lo * hi);
        if (pace_area(r0, p->Tr, T, mid) > S) hi = mid; else lo = mid;
    }
    p->rc = sqrt(lo * hi);
    p->k = log(p->rc / r0) / p->Tr;
    p->A = (p->rc - r0) / p->k;
}

static double pace_steps(const Pace *p, double t) {
    if (t <= 0) return 0;
    double s = p->Tr > 0 && t < p->Tr ? p->r0 * (exp(p->k * t) - 1) / p->k : p->A + p->rc * (t - p->Tr);
    return s > p->S ? (double)p->S : s;
}

static double pace_tau(const Pace *p, double s) {
    if (s <= 0) return 0;
    if (p->Tr > 0 && s < p->A) return log(1 + s * p->k / p->r0) / p->k;
    return p->Tr + (s - p->A) / p->rc;
}

static double pace_rate(const Pace *p, double t) {
    if (p->Tr > 0 && t < p->Tr) return p->r0 * exp(p->k * fmax(t, 0));
    return p->rc;
}

/* ======================================================================
 * SORTING
 * ====================================================================== */
static int *A, N, *B, *C;              /* the array being sorted, two scratch arrays */
static long t_cmp, t_swp, t_rd, t_wr;  /* tallies kept by the helpers (checked against playback) */
static Rng srng;                       /* for random pivots and shuffles */

static int lt(int i, int j) { emit(E_CMP, i, j); t_cmp++; t_rd += 2; return A[i] < A[j]; }
static int ltv(int i, int v) { emit(E_CMPV, i, v); t_cmp++; t_rd++; return A[i] < v; }
static int gtv(int i, int v) { emit(E_CMPV, i, v); t_cmp++; t_rd++; return A[i] > v; }
static int eqv(int i, int v) { emit(E_CMPV, i, v); t_cmp++; t_rd++; return A[i] == v; }
static int lex(int at, int a, int b) { emit(E_CMPX, at, 0); t_cmp++; return a <= b; }  /* two held values */
static int rd(int i) { emit(E_READ, i, 0); t_rd++; return A[i]; }
static void wr(int i, int v) { emit(E_WRITE, i, v); t_wr++; A[i] = v; }
static void mov(int dst, int src) { emit(E_MOVE, dst, src); t_rd++; t_wr++; A[dst] = A[src]; }
static void swp(int i, int j) {
    if (i == j) return;
    emit(E_SWAP, i, j); t_swp++; t_rd += 2; t_wr += 2;
    int t = A[i]; A[i] = A[j]; A[j] = t;
}
static void note(int kind, int v) { emit(E_NOTE, kind, v); }
static void range(int lo, int hi) { emit(E_RANGE, lo, hi); }

static void s_insertion(void) {
    for (int i = 1; i < N; i++)
        for (int j = i; j > 0 && lt(j, j - 1); j--) swp(j, j - 1);
}

static void s_selection(void) {
    for (int i = 0; i < N - 1; i++) {
        int m = i;
        emit(E_MARK, m, 0);
        for (int j = i + 1; j < N; j++)
            if (lt(j, m)) { m = j; emit(E_MARK, m, 0); }
        swp(i, m);
    }
    emit(E_MARK, -1, 0);
}

static void s_bubble(void) {
    for (int end = N - 1; end > 0;) {
        int last = 0;
        range(0, end);
        for (int j = 0; j < end; j++)
            if (lt(j + 1, j)) { swp(j, j + 1); last = j; }
        end = last;
    }
}

static void s_cocktail(void) {
    int lo = 0, hi = N - 1;
    while (lo < hi) {
        int last = lo;
        range(lo, hi);
        for (int j = lo; j < hi; j++)
            if (lt(j + 1, j)) { swp(j, j + 1); last = j; }
        hi = last;
        last = hi;
        range(lo, hi);
        for (int j = hi; j > lo; j--)
            if (lt(j, j - 1)) { swp(j, j - 1); last = j; }
        lo = last;
    }
}

static void s_gnome(void) {
    for (int i = 0; i < N;) {
        if (i == 0 || !lt(i, i - 1)) i++;
        else { swp(i, i - 1); i--; }
    }
}

static void s_oddeven_t(void) {
    for (int r = 0; r < N; r++) {
        note(NOTE_LAYER, r);
        for (int i = r & 1; i + 1 < N; i += 2)
            if (lt(i + 1, i)) swp(i, i + 1);
    }
}

static void s_cycle(void) {
    for (int cs = 0; cs < N - 1; cs++) {
        emit(E_MARK, cs, 0);
        int item = rd(cs), pos = cs;
        for (int i = cs + 1; i < N; i++)
            if (ltv(i, item)) pos++;
        if (pos == cs) continue;
        while (eqv(pos, item)) pos++;
        int t = rd(pos); wr(pos, item); item = t;
        while (pos != cs) {
            pos = cs;
            for (int i = cs + 1; i < N; i++)
                if (ltv(i, item)) pos++;
            while (pos != cs && eqv(pos, item)) pos++;
            if (pos == cs) { wr(cs, item); break; }
            t = rd(pos); wr(pos, item); item = t;
        }
    }
    emit(E_MARK, -1, 0);
}

static void flip(int l, int r) {
    range(l, r);
    while (l < r) swp(l++, r--);
}
static void s_pancake(void) {
    for (int size = N; size > 1; size--) {
        int m = 0;
        for (int i = 1; i < size; i++)
            if (lt(m, i)) m = i;
        emit(E_MARK, m, 0);
        if (m != size - 1) {
            if (m) flip(0, m);
            flip(0, size - 1);
        }
    }
    emit(E_MARK, -1, 0);
}

static void s_shell(void) {
    int gaps[24] = {1, 4, 10, 23, 57, 132, 301, 701}, ng = 8;
    while (ng < 24 && gaps[ng - 1] < N) { gaps[ng] = (int)(gaps[ng - 1] * 2.25); ng++; }
    for (int g = ng - 1; g >= 0; g--) {
        int gap = gaps[g];
        if (gap >= N) continue;
        note(NOTE_GAP, gap);
        for (int i = gap; i < N; i++)
            for (int j = i; j >= gap && lt(j, j - gap); j -= gap) swp(j, j - gap);
    }
}

static void s_comb(void) {
    int gap = N, sorted = 0;
    while (!sorted) {
        gap = gap * 10 / 13;
        if (gap <= 1) { gap = 1; sorted = 1; }
        note(NOTE_GAP, gap);
        for (int i = 0; i + gap < N; i++)
            if (lt(i + gap, i)) { swp(i, i + gap); sorted = 0; }
    }
}

/* merge [lo, mid) and [mid, hi) through the scratch array */
static void merge(int lo, int mid, int hi) {
    range(lo, hi - 1);
    for (int k = lo; k < hi; k++) B[k] = rd(k);
    int i = lo, j = mid;
    for (int k = lo; k < hi; k++)
        if (i < mid && (j >= hi || lex(k, B[i], B[j]))) wr(k, B[i++]);
        else wr(k, B[j++]);
}
static void msort(int lo, int hi) {
    if (hi - lo < 2) return;
    int mid = (lo + hi) / 2;
    msort(lo, mid);
    msort(mid, hi);
    merge(lo, mid, hi);
}
static void s_merge(void) { msort(0, N); }
static void s_merge_bu(void) {
    for (int w = 1; w < N; w *= 2) {
        note(NOTE_WIDTH, w);
        for (int lo = 0; lo < N - w; lo += 2 * w) merge(lo, lo + w, lo + 2 * w < N ? lo + 2 * w : N);
    }
}

static void qs_lomuto(int lo, int hi) {
    if (lo >= hi) return;
    range(lo, hi);
    swp(lo + rint_(&srng, hi - lo + 1), hi);
    emit(E_PIVOT, A[hi], 0);
    int i = lo;
    for (int j = lo; j < hi; j++)
        if (lt(j, hi)) { swp(i, j); i++; }
    swp(i, hi);
    qs_lomuto(lo, i - 1);
    qs_lomuto(i + 1, hi);
}
static void s_quick_lomuto(void) { qs_lomuto(0, N - 1); emit(E_PIVOT, -1, 0); }

static void qs_hoare(int lo, int hi) {
    if (lo >= hi) return;
    range(lo, hi);
    int pv = rd(lo + (hi - lo) / 2);
    emit(E_PIVOT, pv, 0);
    int i = lo - 1, j = hi + 1;
    for (;;) {
        do i++; while (ltv(i, pv));
        do j--; while (gtv(j, pv));
        if (i >= j) break;
        swp(i, j);
    }
    qs_hoare(lo, j);
    qs_hoare(j + 1, hi);
}
static void s_quick_hoare(void) { qs_hoare(0, N - 1); emit(E_PIVOT, -1, 0); }

static void sift(int i, int n) {
    for (;;) {
        int l = 2 * i + 1;
        if (l >= n) break;
        int c = l;
        if (l + 1 < n && lt(l, l + 1)) c = l + 1;
        if (!lt(i, c)) break;
        swp(i, c);
        i = c;
    }
}
static void s_heap(void) {
    emit(E_HEAPN, N, 0);
    for (int i = N / 2 - 1; i >= 0; i--) sift(i, N);
    for (int end = N - 1; end > 0; end--) {
        swp(0, end);
        emit(E_HEAPN, end, 0);
        sift(0, end);
    }
    emit(E_HEAPN, 0, 0);
}

static int maxval(void) { int m = 0; for (int i = 0; i < N; i++) if (A[i] > m) m = A[i]; return m; }
#define RADIX 4
static void s_radix_lsd(void) {
    int mx = maxval(), pass = 0;
    for (long ex = 1; mx / ex > 0; ex *= RADIX) {
        note(NOTE_PASS, pass++);
        int cnt[RADIX] = {0};
        for (int i = 0; i < N; i++) { C[i] = rd(i); cnt[(C[i] / ex) % RADIX]++; }
        for (int d = 1; d < RADIX; d++) cnt[d] += cnt[d - 1];
        for (int i = N - 1; i >= 0; i--) B[--cnt[(C[i] / ex) % RADIX]] = C[i];
        for (int i = 0; i < N; i++) wr(i, B[i]);
    }
}
static void msd(int lo, int hi, long ex) {
    if (hi - lo < 2 || ex == 0) return;
    range(lo, hi - 1);
    int cnt[RADIX + 1] = {0};
    for (int i = lo; i < hi; i++) { C[i] = rd(i); cnt[(C[i] / ex) % RADIX + 1]++; }
    for (int d = 1; d <= RADIX; d++) cnt[d] += cnt[d - 1];
    int start[RADIX + 1];
    memcpy(start, cnt, sizeof start);
    for (int i = lo; i < hi; i++) B[lo + cnt[(C[i] / ex) % RADIX]++] = C[i];
    for (int i = lo; i < hi; i++) wr(i, B[i]);
    for (int d = 0; d < RADIX; d++) msd(lo + start[d], lo + start[d + 1], ex / RADIX);
}
static void s_radix_msd(void) {
    int mx = maxval();
    long ex = 1;
    while (ex * RADIX <= mx) ex *= RADIX;
    msd(0, N, ex);
}

static void s_counting(void) {
    int mx = maxval();
    int *cnt = calloc((size_t)mx + 1, sizeof *cnt);
    for (int i = 0; i < N; i++) cnt[rd(i)]++;
    note(NOTE_PASS, 1);
    for (int v = 0, k = 0; v <= mx; v++)
        for (int c = 0; c < cnt[v]; c++) wr(k++, v);
    free(cnt);
}

/* a timsort: natural runs (descending ones reversed), short runs extended by
 * binary insertion to minrun, and timsort's stack invariants deciding the
 * merges (without galloping) */
static void bin_insert(int lo, int start, int hi) {
    for (int k = start; k < hi; k++) {
        int pv = rd(k), l = lo, r = k;
        while (l < r) {
            int m = (l + r) / 2;
            if (gtv(m, pv)) r = m; else l = m + 1;
        }
        for (int p = k; p > l; p--) mov(p, p - 1);
        if (l != k) wr(l, pv);
    }
}
static void s_tim(void) {
    int n = N, r = 0;
    while (n >= 32) { r |= n & 1; n >>= 1; }
    int minrun = n + r;
    int rb[64], rl[64], nr = 0, runs = 0;
    for (int i = 0; i < N;) {
        int j = i + 1;
        if (j < N) {
            if (lt(j, i)) {
                while (j + 1 < N && lt(j + 1, j)) j++;
                flip(i, j);
            } else {
                while (j + 1 < N && !lt(j + 1, j)) j++;
            }
            j++;
        }
        int len = j - i;
        if (len < minrun) {
            int force = N - i < minrun ? N - i : minrun;
            bin_insert(i, i + len, i + force);
            len = force;
        }
        rb[nr] = i; rl[nr] = len; nr++;
        note(NOTE_RUNS, ++runs);
        i += len;
        /* merge_collapse */
        while (nr > 1) {
            int k = nr - 2;
            if ((k > 0 && rl[k - 1] <= rl[k] + rl[k + 1]) || (k > 1 && rl[k - 2] <= rl[k - 1] + rl[k])) {
                if (rl[k - 1] < rl[k + 1]) k--;
            } else if (rl[k] > rl[k + 1]) break;
            merge(rb[k], rb[k + 1], rb[k + 1] + rl[k + 1]);
            rl[k] += rl[k + 1];
            for (int q = k + 1; q < nr - 1; q++) { rb[q] = rb[q + 1]; rl[q] = rl[q + 1]; }
            nr--;
        }
    }
    while (nr > 1) {
        int k = nr - 2;
        if (k > 0 && rl[k - 1] < rl[k + 1]) k--;
        merge(rb[k], rb[k + 1], rb[k + 1] + rl[k + 1]);
        rl[k] += rl[k + 1];
        for (int q = k + 1; q < nr - 1; q++) { rb[q] = rb[q + 1]; rl[q] = rl[q + 1]; }
        nr--;
    }
}

static void cex(int a, int b) { if (lt(b, a)) swp(a, b); }
static void s_bitonic(void) {
    int layer = 0;
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            note(NOTE_LAYER, layer++);
            for (int i = 0; i < N; i++) {
                int l = i ^ j;
                if (l > i) {
                    if ((i & k) == 0) cex(i, l);
                    else cex(l, i);
                }
            }
        }
}
static void s_oddeven_merge(void) {
    int layer = 0;
    for (int p = 1; p < N; p <<= 1)
        for (int k = p; k >= 1; k >>= 1) {
            note(NOTE_LAYER, layer++);
            for (int j = k % p; j + k < N; j += 2 * k)
                for (int i = 0; i < k && i + j + k < N; i++)
                    if ((i + j) / (2 * p) == (i + j + k) / (2 * p)) cex(i + j, i + j + k);
        }
}

static int bogo_limit = 200;
static void s_bogo(void) {
    for (int sh = 0;; sh++) {
        int ok = 1;
        for (int i = 1; i < N && ok; i++)
            if (lt(i, i - 1)) ok = 0;
        if (ok) { note(NOTE_SORTED, sh); return; }
        if (sh == bogo_limit) { note(NOTE_GIVEUP, sh); return; }
        for (int i = N - 1; i > 0; i--) swp(i, rint_(&srng, i + 1));
        note(NOTE_SHUF, sh + 1);
    }
}

enum { CL_QUAD, CL_FAST, CL_JOKE };
typedef struct {
    const char *key, *name, *detail;
    int cls, pow2, net;
    void (*fn)(void);
} SortDef;

static const SortDef SORTS[] = {
    {"insertion", "Insertion sort", "swapping each new element down into place", CL_QUAD, 0, 0, s_insertion},
    {"selection", "Selection sort", "the minimum of the rest, swapped to the front", CL_QUAD, 0, 0, s_selection},
    {"bubble", "Bubble sort", "adjacent swaps \xc2\xb7 early exit", CL_QUAD, 0, 0, s_bubble},
    {"cocktail", "Cocktail shaker sort", "bubble sort in both directions", CL_QUAD, 0, 0, s_cocktail},
    {"gnome", "Gnome sort", "step forward when in order, swap back when not", CL_QUAD, 0, 0, s_gnome},
    {"oddeven", "Odd\xe2\x80\x93" "even transposition sort", "brick sort \xc2\xb7 a network of n rounds", CL_QUAD, 0, 1, s_oddeven_t},
    {"cycle", "Cycle sort", "each element written once, straight home", CL_QUAD, 0, 0, s_cycle},
    {"pancake", "Pancake sort", "only prefix reversals", CL_QUAD, 0, 0, s_pancake},
    {"shell", "Shellsort", "Ciura's gap sequence", CL_FAST, 0, 0, s_shell},
    {"comb", "Comb sort", "shrinking gaps \xc2\xb7 factor 1.3", CL_FAST, 0, 0, s_comb},
    {"merge", "Merge sort", "top-down", CL_FAST, 0, 0, s_merge},
    {"merge-bu", "Merge sort", "bottom-up", CL_FAST, 0, 0, s_merge_bu},
    {"quick-lomuto", "Quicksort", "Lomuto partition \xc2\xb7 random pivot", CL_FAST, 0, 0, s_quick_lomuto},
    {"quick-hoare", "Quicksort", "Hoare partition \xc2\xb7 middle pivot", CL_FAST, 0, 0, s_quick_hoare},
    {"heap", "Heapsort", "binary max-heap", CL_FAST, 0, 0, s_heap},
    {"radix-lsd", "Radix sort", "LSD \xc2\xb7 base 4", CL_FAST, 0, 0, s_radix_lsd},
    {"radix-msd", "Radix sort", "MSD \xc2\xb7 base 4", CL_FAST, 0, 0, s_radix_msd},
    {"counting", "Counting sort", "count, then write out in order", CL_FAST, 0, 0, s_counting},
    {"tim", "Timsort", "natural runs \xc2\xb7 binary insertion \xc2\xb7 balanced merges", CL_FAST, 0, 0, s_tim},
    {"bitonic", "Bitonic sort", "Batcher's bitonic network", CL_FAST, 1, 1, s_bitonic},
    {"oem", "Odd\xe2\x80\x93" "even merge sort", "Batcher's merging network", CL_FAST, 1, 1, s_oddeven_merge},
    {"bogo", "Bogosort", "shuffle until sorted", CL_JOKE, 0, 0, s_bogo},
};
#define NSORT ((int)(sizeof SORTS / sizeof SORTS[0]))

enum { I_RANDOM, I_NEARLY, I_REVERSED, I_FEW, I_SAW, NINPUT };
static const char *INPUTS[NINPUT] = {"random", "nearly-sorted", "reversed", "few-unique", "sawtooth"};
static const char *INPUT_NAME[NINPUT] = {"random", "nearly sorted", "reversed", "few unique", "sawtooth"};

enum { F_BARS, F_DOTS, F_WHEEL, F_IMAGE, F_NET, F_HEAP, NFORM };
static const char *FORMS[NFORM] = {"bars", "dots", "wheel", "image", "network", "heap"};

static int cmp_float_idx_keys;
static const float *qkeys;
static int by_key(const void *a, const void *b) {
    float x = qkeys[*(const int *)a], y = qkeys[*(const int *)b];
    (void)cmp_float_idx_keys;
    return x < y ? -1 : x > y;
}

/* fill v[0..n) with an input of the given shape; values 0..n-1 (a
 * permutation, except few-unique) */
static void make_input(int *v, int n, int shape, Rng *r) {
    for (int i = 0; i < n; i++) v[i] = i;
    switch (shape) {
    case I_RANDOM:
        for (int i = n - 1; i > 0; i--) { int j = rint_(r, i + 1), t = v[i]; v[i] = v[j]; v[j] = t; }
        break;
    case I_NEARLY: {
        int sw = n / 14 + 1, d = n / 18 + 2;
        for (int s = 0; s < sw; s++) {
            int i = rint_(r, n), j = i + 1 + rint_(r, d);
            if (j >= n) j = n - 1;
            int t = v[i]; v[i] = v[j]; v[j] = t;
        }
        break;
    }
    case I_REVERSED:
        for (int i = 0; i < n; i++) v[i] = n - 1 - i;
        break;
    case I_FEW: {
        int k = 4 + rint_(r, 4);
        for (int i = 0; i < n; i++) v[i] = rint_(r, k) * (n - 1) / (k - 1);
        break;
    }
    case I_SAW: {
        int teeth = 3 + rint_(r, 4);
        float *key = malloc((size_t)n * sizeof *key);
        int *idx = malloc((size_t)n * sizeof *idx);
        for (int i = 0; i < n; i++) {
            int t = i * teeth / n, s0 = t * n / teeth, s1 = (t + 1) * n / teeth;
            key[i] = (float)(i - s0) / (float)(s1 - s0) + t * 1e-5f;
            idx[i] = i;
        }
        qkeys = key;
        qsort(idx, (size_t)n, sizeof *idx, by_key);
        for (int k = 0; k < n; k++) v[idx[k]] = k;
        free(key);
        free(idx);
        break;
    }
    }
}

/* run sort `alg` on v[0..n) into log L (if L), leaving the result in v */
static void run_sort(int alg, int *v, int n, Log *L, uint32_t seed) {
    A = v; N = n;
    B = malloc((size_t)(n + 1) * sizeof *B);
    C = malloc((size_t)(n + 1) * sizeof *C);
    t_cmp = t_swp = t_rd = t_wr = 0;
    srng = rng_of(seed, 4242);
    rec = L;
    SORTS[alg].fn();
    rec = NULL;
    free(B); free(C);
    if (L) log_finish(L);
}

/* ---- sort playback ---- */
typedef struct {
    int alg, form, input, n, maxv;
    Log log;
    Pace pace;
    double t0;                         /* act-local time the sort starts */
    int *init, *v, *lastc, *lasts;     /* lastc/lasts: timed index of the last compare / swap-or-write touching it */
    int *home0, *home1;                /* where each value belongs: [home0, home1] */
    long cmp, swp, rd, wr;
    int pivot, lo, hi, heapn, mark, notes[NNOTE], hasnote[NNOTE];
    int pos, done;
    long S;                            /* timed events in all */
    double t_end;                      /* act-local time it finishes */
    /* network form */
    int ncomp, ncol, *comp_a, *comp_b, *comp_col, *snap, *wire_last, cc, *comp_ev, *comp_sw;
    /* image form */
    unsigned char *pic;
    int pic_w, pic_kind, pic_vert;
    int pic_dirty;
} SortPlay;

static void sp_reset(SortPlay *s) {
    memcpy(s->v, s->init, (size_t)s->n * sizeof *s->v);
    for (int i = 0; i < s->n; i++) s->lastc[i] = s->lasts[i] = -1000000;
    s->cmp = s->swp = s->rd = s->wr = 0;
    s->pivot = -1; s->lo = -1; s->hi = -1; s->heapn = -1; s->mark = -1;
    memset(s->hasnote, 0, sizeof s->hasnote);
    s->pos = 0; s->done = 0; s->cc = 0;
    if (s->wire_last) for (int i = 0; i < s->n; i++) s->wire_last[i] = -1;
    s->pic_dirty = 1;
}

static void sp_apply(SortPlay *s, const Ev *e) {
    int t = s->done;   /* timed index of this event (if timed) */
    int *v = s->v;
    switch (e->k) {
    case E_CMP:
        s->cmp++; s->rd += 2; s->lastc[e->a] = s->lastc[e->b] = t;
        if (s->comp_a) {
            int c = s->cc;
            if (c < s->ncomp) { s->comp_ev[c] = t; s->comp_sw[c] = 0; s->wire_last[e->a] = s->wire_last[e->b] = s->comp_col[c]; }
            s->cc++;
        }
        break;
    case E_CMPV: s->cmp++; s->rd++; s->lastc[e->a] = t; break;
    case E_CMPX: s->cmp++; s->lastc[e->a] = t; break;
    case E_SWAP: {
        s->swp++; s->rd += 2; s->wr += 2;
        int x = v[e->a]; v[e->a] = v[e->b]; v[e->b] = x;
        s->lasts[e->a] = s->lasts[e->b] = t;
        if (s->comp_a && s->cc > 0) s->comp_sw[s->cc - 1] = 1;
        s->pic_dirty = 1;
        break;
    }
    case E_WRITE: s->wr++; v[e->a] = e->b; s->lasts[e->a] = t; s->pic_dirty = 1; break;
    case E_READ: s->rd++; s->lastc[e->a] = t; break;
    case E_MOVE: s->rd++; s->wr++; v[e->a] = v[e->b]; s->lasts[e->a] = t; s->lastc[e->b] = t; s->pic_dirty = 1; break;
    case E_PIVOT: s->pivot = e->a; break;
    case E_RANGE: s->lo = e->a; s->hi = e->b; break;
    case E_HEAPN: s->heapn = e->a; break;
    case E_MARK: s->mark = e->a; break;
    case E_NOTE: s->notes[e->a] = e->b; s->hasnote[e->a] = 1; break;
    }
    if (timed(e->k)) s->done++;
}

/* play to `steps` timed events (and the markers that follow them) */
static void sp_seek(SortPlay *s, long steps) {
    if (steps > s->S) steps = s->S;
    if (steps < s->done) sp_reset(s);
    const Log *L = &s->log;
    int end = steps >= L->nt ? L->n : L->tp[steps];
    while (s->pos < end) sp_apply(s, &L->e[s->pos++]);
}

/* the network form: comparators from the log, packed into columns */
static void sp_network(SortPlay *s) {
    const Log *L = &s->log;
    int nc = 0;
    for (int i = 0; i < L->n; i++) nc += L->e[i].k == E_CMP;
    s->ncomp = nc;
    s->comp_a = malloc((size_t)nc * sizeof(int)); s->comp_b = malloc((size_t)nc * sizeof(int));
    s->comp_col = malloc((size_t)nc * sizeof(int)); s->comp_ev = malloc((size_t)nc * sizeof(int));
    s->comp_sw = malloc((size_t)nc * sizeof(int));
    s->wire_last = malloc((size_t)s->n * sizeof(int));
    int col0 = 0, ncol = 0, c = 0;
    for (int i = 0; i < L->n; i++) {
        const Ev *e = &L->e[i];
        if (e->k == E_NOTE && e->a == NOTE_LAYER) { col0 = ncol; continue; }
        if (e->k != E_CMP) continue;
        int a = e->a < e->b ? e->a : e->b, b = e->a < e->b ? e->b : e->a;
        s->comp_a[c] = a; s->comp_b[c] = b;
        /* first column of this layer where it overlaps nothing */
        int col = col0;
        for (;; col++) {
            int clash = 0;
            for (int q = c - 1; q >= 0 && s->comp_col[q] >= col0; q--)
                if (s->comp_col[q] == col && s->comp_a[q] <= b && a <= s->comp_b[q]) { clash = 1; break; }
            if (!clash) break;
        }
        s->comp_col[c] = col;
        if (col + 1 > ncol) ncol = col + 1;
        c++;
    }
    s->ncol = ncol;
    /* the value on each wire after each column */
    s->snap = malloc((size_t)(ncol + 1) * s->n * sizeof(int));
    int *v = malloc((size_t)s->n * sizeof(int));
    memcpy(v, s->init, (size_t)s->n * sizeof(int));
    c = 0;
    for (int col = 0; col <= ncol; col++) {
        memcpy(s->snap + (size_t)col * s->n, v, (size_t)s->n * sizeof(int));
        for (; c < nc && s->comp_col[c] == col; c++)
            if (v[s->comp_b[c]] < v[s->comp_a[c]]) { int t = v[s->comp_a[c]]; v[s->comp_a[c]] = v[s->comp_b[c]]; v[s->comp_b[c]] = t; }
    }
    free(v);
}

static void sp_free(SortPlay *s) {
    log_free(&s->log);
    free(s->init); free(s->v); free(s->lastc); free(s->lasts); free(s->home0); free(s->home1);
    free(s->comp_a); free(s->comp_b); free(s->comp_col); free(s->snap); free(s->wire_last);
    free(s->comp_ev); free(s->comp_sw); free(s->pic);
    memset(s, 0, sizeof *s);
}

/* set up a sort: input, the recorded run, homes; pacing is the caller's */
static void sp_build(SortPlay *s, int alg, int form, int input, int n, uint32_t seed) {
    sp_free(s);
    s->alg = alg; s->form = form; s->input = input; s->n = n;
    s->init = malloc((size_t)n * sizeof(int)); s->v = malloc((size_t)n * sizeof(int));
    s->lastc = malloc((size_t)n * sizeof(int)); s->lasts = malloc((size_t)n * sizeof(int));
    Rng r = rng_of(seed, 77);
    make_input(s->init, n, input, &r);
    int *work = malloc((size_t)n * sizeof(int));
    memcpy(work, s->init, (size_t)n * sizeof(int));
    if (SORTS[alg].cls == CL_JOKE) bogo_limit = 160 + (int)(seed % 120);
    run_sort(alg, work, n, &s->log, seed);
    s->S = s->log.nt;
    s->maxv = 1;
    for (int i = 0; i < n; i++) if (s->init[i] > s->maxv) s->maxv = s->init[i];
    /* homes: sorted, value v occupies [home0, home1] */
    s->home0 = malloc((size_t)(s->maxv + 1) * sizeof(int)); s->home1 = malloc((size_t)(s->maxv + 1) * sizeof(int));
    int *cnt = calloc((size_t)s->maxv + 2, sizeof(int));
    for (int i = 0; i < n; i++) cnt[s->init[i] + 1]++;
    for (int x = 1; x <= s->maxv + 1; x++) cnt[x] += cnt[x - 1];
    for (int x = 0; x <= s->maxv; x++) { s->home0[x] = cnt[x]; s->home1[x] = cnt[x + 1] - 1; }
    free(cnt);
    free(work);
    if (form == F_NET) sp_network(s);
    sp_reset(s);
}

/* ======================================================================
 * MAZES (generic graph: square, hex, theta)
 * ====================================================================== */
#define MAXNB 8
enum { G_SQUARE, G_HEX, G_THETA, NGRID };
static const char *GRIDS[NGRID] = {"square", "hex", "theta"};
typedef struct {
    float x, y;                        /* centre (logical units) */
    int nn, nb[MAXNB];                 /* neighbours, in angular order */
    int ring, col, row;
    float a0, a1, r;                   /* theta: angular span and radius */
} Cell;

typedef struct {
    int grid, nc, cols, rows, rings;
    Cell *c;
    int *sq;                           /* square: row * cols + col -> cell, or -1 */
    float pitch, cx, cy;               /* theta centre */
    int start, goal;
} Graph;

/* ======================================================================
 * THE PROGRAMME
 * ====================================================================== */
enum { K_SORT, K_RACE, K_TRIO, K_MAZE, NKIND };
static const char *KINDS[NKIND] = {"sort", "race", "triptych", "maze"};

typedef struct {
    int kind;
    double t0, dur;
    uint32_t seed;
    int alg, form, input, n, pic;
    int algs[MAX_OUT], forms[MAX_OUT];  /* by output rank (race / triptych) */
    double T;                           /* sorting time, or maze generation time */
    int gen, solve, grid, density;
    double Ts;                          /* maze solving time */
} Act;

/* act-local timeline of a sort */
#define SORT_LEAD 2.2                  /* the title alone, data still */
#define SORT_SWEEP 1.6                 /* the check sweep once sorted */
#define SORT_HOLD 4.0
#define FADE 1.2

static int nsched, capsched;
static Act *sched;
static int recent_alg[6], recent_gen[3];

static int find_name(const char *s, const char *const *names, int n) {
    for (int i = 0; i < n; i++) if (!strcmp(s, names[i])) return i;
    return -1;
}
static int find_sort(const char *s) {
    for (int i = 0; i < NSORT; i++) if (!strcmp(s, SORTS[i].key)) return i;
    return -1;
}

static int is_recent(int alg) {
    for (int i = 0; i < 6; i++) if (recent_alg[i] == alg + 1) return 1;
    return 0;
}
static void push_recent(int alg) {
    memmove(recent_alg + 1, recent_alg, 5 * sizeof(int));
    recent_alg[0] = alg + 1;
}

static int forced_sort(void) { return opt_sort ? find_sort(opt_sort) : -1; }

/* pick a sort of class cls (-1 any but the joke), not played lately */
static int pick_sort(Rng *r, int cls, int allow_joke) {
    if (forced_sort() >= 0) return forced_sort();
    if (allow_joke && rint_(r, 28) == 0 && !is_recent(NSORT - 1)) return NSORT - 1;
    for (int tries = 0; tries < 200; tries++) {
        int a = rint_(r, NSORT - 1);
        if (cls >= 0 && SORTS[a].cls != cls) continue;
        if (tries < 150 && is_recent(a)) continue;
        return a;
    }
    return 0;
}

static int pow2_floor(int n) { int p = 1; while (p * 2 <= n) p *= 2; return p; }

/* the size of a sort in a form */
static int pick_n(Rng *r, int alg, int form, int portrait) {
    if (opt_n > 0) return SORTS[alg].pow2 ? pow2_floor(opt_n) : opt_n;
    const SortDef *d = &SORTS[alg];
    int n;
    if (d->cls == CL_JOKE) return 6 + rint_(r, 2);
    if (form == F_NET) return d->cls == CL_QUAD ? 16 : (rint_(r, 3) ? 32 : 16);
    if (form == F_HEAP) return portrait ? 127 : 255;
    static const int quad[NFORM][3] = {{64, 96, 128}, {128, 160, 200}, {120, 160, 200}, {160, 200, 240}};
    static const int fast[NFORM][3] = {{256, 384, 512}, {512, 768, 1024}, {480, 640, 800}, {320, 480, 640}};
    n = d->cls == CL_QUAD ? quad[form][rint_(r, 3)] : fast[form][rint_(r, 3)];
    if (d->pow2) n = pow2_floor(n);
    return n;
}

static int pick_input(Rng *r, int alg, int form) {
    int f = opt_input ? find_name(opt_input, INPUTS, NINPUT) : -1;
    if (f >= 0) return f;
    int x = rint_(r, 100);
    int in = x < 50 ? I_RANDOM : x < 62 ? I_NEARLY : x < 72 ? I_REVERSED : x < 86 ? I_FEW : I_SAW;
    if (form == F_IMAGE && in == I_FEW) in = I_RANDOM;
    if (SORTS[alg].cls == CL_JOKE || form == F_NET) in = I_RANDOM;
    if (!strcmp(SORTS[alg].key, "counting") && x % 2) in = I_FEW;
    return in;
}

static int pick_form(Rng *r, int alg) {
    int f = opt_form ? find_name(opt_form, FORMS, NFORM) : -1;
    if (f == F_NET && !SORTS[alg].net) f = -1;
    if (f == F_HEAP && strcmp(SORTS[alg].key, "heap")) f = -1;
    if (f >= 0) return f;
    if (SORTS[alg].cls == CL_JOKE) return rint_(r, 2) ? F_BARS : F_DOTS;
    if (SORTS[alg].net && rint_(r, 100) < 45) return F_NET;
    if (!strcmp(SORTS[alg].key, "heap") && rint_(r, 100) < 50) return F_HEAP;
    int x = rint_(r, 100);
    return x < 32 ? F_BARS : x < 52 ? F_DOTS : x < 76 ? F_WHEEL : F_IMAGE;
}

static double sort_time(int alg, int form) {
    const SortDef *d = &SORTS[alg];
    if (d->cls == CL_JOKE) return 26;
    if (form == F_NET) return 44;
    return d->cls == CL_QUAD ? 50 : 38;
}

static int spanning(void) { return span_me >= 0 && span_n >= 2; }
static int portrait_out(int o) {
    if (o < 0) return H > W * 1.05f;
    return span[o][3] > span[o][2] * 1.05f;
}

static void maze_params(Act *a, Rng *r, int span_mode);

static Act next_act(int k, double t0) {
    Act a;
    memset(&a, 0, sizeof a);
    a.t0 = t0;
    a.seed = hash3(opt_seed, (uint32_t)k, 0xac7);
    Rng r = rng_of(a.seed, 1);
    int kind;
    int fk = opt_kind ? find_name(opt_kind, KINDS, NKIND) : -1;
    if (opt_sort || opt_form || opt_input) fk = fk < 0 ? (spanning() ? K_TRIO : K_SORT) : fk;
    if (opt_maze || opt_solve || opt_grid) fk = K_MAZE;
    if (fk >= 0) kind = fk;
    else if (spanning()) { static const int cyc[6] = {K_RACE, K_MAZE, K_TRIO, K_RACE, K_TRIO, K_MAZE}; kind = cyc[k % 6]; }
    else kind = k % 3 == 2 ? K_MAZE : K_SORT;
    if (!spanning() && (kind == K_RACE || kind == K_TRIO)) kind = K_SORT;
    a.kind = kind;
    int portrait = portrait_out(spanning() ? front : -1);
    if (kind == K_SORT) {
        a.alg = pick_sort(&r, -1, 1);
        a.form = pick_form(&r, a.alg);
        a.n = pick_n(&r, a.alg, a.form, portrait);
        a.input = pick_input(&r, a.alg, a.form);
        a.T = sort_time(a.alg, a.form);
        push_recent(a.alg);
        a.dur = SORT_LEAD + a.T + SORT_SWEEP + SORT_HOLD + FADE;
    } else if (kind == K_RACE) {
        int cls = rint_(&r, 5) < 2 ? CL_QUAD : CL_FAST;
        int x = rint_(&r, 100);
        a.form = x < 40 ? F_BARS : x < 60 ? F_DOTS : x < 80 ? F_WHEEL : F_IMAGE;
        int f = opt_form ? find_name(opt_form, FORMS, NFORM) : -1;
        if (f >= 0 && f <= F_IMAGE) a.form = f;
        for (int i = 0; i < span_n; i++) {
            int al;
            for (int tries = 0;; tries++) {
                al = pick_sort(&r, cls, 0);
                int dup = 0;
                for (int j = 0; j < i; j++) dup |= a.algs[j] == al;
                if (!dup || tries > 100) break;
            }
            a.algs[i] = al;
            a.forms[i] = a.form;
            push_recent(al);
        }
        a.alg = a.algs[0];
        a.n = opt_n > 0 ? opt_n : cls == CL_QUAD ? (a.form == F_BARS ? 96 : 140) : (a.form == F_BARS ? 320 : 480);
        a.input = pick_input(&r, a.alg, a.form);
        if (a.input == I_FEW && a.form == F_IMAGE) a.input = I_RANDOM;
        a.T = cls == CL_QUAD ? 56 : 44;
        a.dur = SORT_LEAD + a.T + SORT_SWEEP + SORT_HOLD + 2 + FADE;
    } else if (kind == K_TRIO) {
        a.alg = pick_sort(&r, -1, 0);
        int pool[NFORM], np = 0;
        int base[4] = {F_WHEEL, F_BARS, F_IMAGE, F_DOTS};
        if (SORTS[a.alg].net) pool[np++] = F_NET;
        if (!strcmp(SORTS[a.alg].key, "heap")) pool[np++] = F_HEAP;
        int rot = rint_(&r, 4);
        for (int i = 0; i < 4; i++) pool[np++] = base[(i + rot) % 4];
        for (int i = 0; i < span_n; i++) a.forms[i] = pool[i % np];
        int f = opt_form ? find_name(opt_form, FORMS, NFORM) : -1;
        if (f >= 0) for (int i = 0; i < span_n; i++) a.forms[i] = f;
        /* one n for all: the network and heap forms want few */
        a.form = a.forms[0];
        a.n = pick_n(&r, a.alg, F_WHEEL, 0);
        for (int i = 0; i < span_n; i++)
            if (a.forms[i] == F_NET || a.forms[i] == F_HEAP) { a.n = pick_n(&r, a.alg, a.forms[i], 0); break; }
        if (opt_n > 0) a.n = opt_n;
        if (SORTS[a.alg].pow2) a.n = pow2_floor(a.n);
        int img = 0;
        for (int i = 0; i < span_n; i++) img |= a.forms[i] == F_IMAGE || a.forms[i] == F_NET;
        a.input = pick_input(&r, a.alg, img ? F_IMAGE : F_BARS);
        if (img && a.input == I_FEW) a.input = I_RANDOM;
        a.T = sort_time(a.alg, a.forms[0] == F_NET ? F_NET : F_BARS);
        push_recent(a.alg);
        a.dur = SORT_LEAD + a.T + SORT_SWEEP + SORT_HOLD + FADE;
    } else {
        maze_params(&a, &r, spanning());
    }
    a.pic = rint_(&r, 3);
    return a;
}

static const Act *act_at(double el, int *index) {
    if (el < 0) el = 0;
    while (nsched == 0 || sched[nsched - 1].t0 + sched[nsched - 1].dur <= el) {
        if (nsched == capsched) sched = realloc(sched, (size_t)(capsched = capsched ? capsched * 2 : 64) * sizeof *sched);
        double t0 = nsched ? sched[nsched - 1].t0 + sched[nsched - 1].dur : 0;
        sched[nsched] = next_act(nsched, t0);
        nsched++;
    }
    int lo = 0, hi = nsched - 1;
    while (lo < hi) {
        int m = (lo + hi + 1) / 2;
        if (sched[m].t0 <= el) lo = m; else hi = m - 1;
    }
    *index = lo;
    return &sched[lo];
}

/* ---- maze graphs ---- */
static Graph G;

static void graph_free(Graph *g) { free(g->c); free(g->sq); memset(g, 0, sizeof *g); }

static int add_cell(Graph *g, float x, float y) {
    if ((g->nc & 1023) == 0) g->c = realloc(g->c, (size_t)(g->nc + 1024) * sizeof *g->c);
    Cell *c = &g->c[g->nc];
    memset(c, 0, sizeof *c);
    c->x = x; c->y = y;
    return g->nc++;
}

static void add_nb(Graph *g, int a, int b) {
    if (a < 0 || b < 0 || a == b) return;
    Cell *ca = &g->c[a], *cb = &g->c[b];
    for (int i = 0; i < ca->nn; i++) if (ca->nb[i] == b) return;
    if (ca->nn < MAXNB && cb->nn < MAXNB) { ca->nb[ca->nn++] = b; cb->nb[cb->nn++] = a; }
}

/* neighbours in angular order (clockwise on screen, from east) */
static void sort_nb(Graph *g) {
    for (int i = 0; i < g->nc; i++) {
        Cell *c = &g->c[i];
        float ang[MAXNB];
        for (int k = 0; k < c->nn; k++) ang[k] = atan2f(g->c[c->nb[k]].y - c->y, g->c[c->nb[k]].x - c->x);
        for (int k = 1; k < c->nn; k++)
            for (int j = k; j > 0 && ang[j] < ang[j - 1]; j--) {
                float t = ang[j]; ang[j] = ang[j - 1]; ang[j - 1] = t;
                int u = c->nb[j]; c->nb[j] = c->nb[j - 1]; c->nb[j - 1] = u;
            }
    }
}

static int nb_slot(const Graph *g, int a, int b) {
    const Cell *c = &g->c[a];
    for (int k = 0; k < c->nn; k++) if (c->nb[k] == b) return k;
    return -1;
}

/* keep only the largest connected piece (masks can leave islands) */
static void graph_prune(Graph *g) {
    int *comp = malloc((size_t)g->nc * sizeof(int)), *q = malloc((size_t)g->nc * sizeof(int));
    for (int i = 0; i < g->nc; i++) comp[i] = -1;
    int best = -1, bestn = 0, nco = 0;
    for (int s = 0; s < g->nc; s++) {
        if (comp[s] >= 0) continue;
        int qh = 0, qt = 0;
        q[qt++] = s; comp[s] = nco;
        while (qh < qt) {
            int c = q[qh++];
            for (int k = 0; k < g->c[c].nn; k++) {
                int n = g->c[c].nb[k];
                if (comp[n] < 0) { comp[n] = nco; q[qt++] = n; }
            }
        }
        if (qt > bestn) { bestn = qt; best = nco; }
        nco++;
    }
    if (nco > 1) {
        int *map = q;
        int m = 0;
        for (int i = 0; i < g->nc; i++) map[i] = comp[i] == best ? m++ : -1;
        for (int i = 0; i < g->nc; i++) {
            if (map[i] < 0) continue;
            Cell c = g->c[i];
            int nn = 0;
            for (int k = 0; k < c.nn; k++) if (map[c.nb[k]] >= 0) c.nb[nn++] = map[c.nb[k]];
            c.nn = nn;
            g->c[map[i]] = c;
        }
        if (g->sq) for (int i = 0; i < g->cols * g->rows; i++) if (g->sq[i] >= 0) g->sq[i] = map[g->sq[i]];
        g->nc = m;
    }
    free(comp); free(q);
}

typedef int (*MaskFn)(float x, float y, float half);

/* a square lattice of pitch p, cols x rows from (x0, y0); cells kept where
 * mask (if any) says so */
static void build_square(Graph *g, float x0, float y0, int cols, int rows, float p, MaskFn mask) {
    graph_free(g);
    g->grid = G_SQUARE; g->cols = cols; g->rows = rows; g->pitch = p;
    g->sq = malloc((size_t)cols * rows * sizeof(int));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            float x = x0 + (c + 0.5f) * p, y = y0 + (r + 0.5f) * p;
            int id = -1;
            if (!mask || mask(x, y, p * 0.5f)) {
                id = add_cell(g, x, y);
                g->c[id].col = c; g->c[id].row = r;
            }
            g->sq[r * cols + c] = id;
        }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int id = g->sq[r * cols + c];
            if (id < 0) continue;
            if (c + 1 < cols) add_nb(g, id, g->sq[r * cols + c + 1]);
            if (r + 1 < rows) add_nb(g, id, g->sq[(r + 1) * cols + c]);
        }
}

static void build_hex(Graph *g, float x0, float y0, int cols, int rows, float p, MaskFn mask) {
    graph_free(g);
    g->grid = G_HEX; g->cols = cols; g->rows = rows; g->pitch = p;
    g->sq = malloc((size_t)cols * rows * sizeof(int));
    float dy = p * 0.8660254f;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            float x = x0 + (c + 0.5f + (r & 1) * 0.5f) * p, y = y0 + p * 0.5f + r * dy;
            int id = -1;
            if (!mask || mask(x, y, p * 0.5f)) {
                id = add_cell(g, x, y);
                g->c[id].col = c; g->c[id].row = r;
            }
            g->sq[r * cols + c] = id;
        }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int id = g->sq[r * cols + c];
            if (id < 0) continue;
            if (c + 1 < cols) add_nb(g, id, g->sq[r * cols + c + 1]);
            if (r + 1 < rows) {
                int c0 = (r & 1) ? c : c - 1;
                for (int cc = c0; cc <= c0 + 1; cc++)
                    if (cc >= 0 && cc < cols) add_nb(g, id, g->sq[(r + 1) * cols + cc]);
            }
        }
}

/* rings of cells around (cx, cy), each ring one pitch deep, splitting cells
 * as the circumference grows */
static void build_theta(Graph *g, float cx, float cy, int rings, float p) {
    graph_free(g);
    g->grid = G_THETA; g->rings = rings; g->pitch = p; g->cx = cx; g->cy = cy;
    int *first = malloc((size_t)(rings + 1) * sizeof(int)), *count = malloc((size_t)(rings + 1) * sizeof(int));
    for (int i = 0; i < rings; i++) {
        int prev = i ? count[i - 1] : 1, n;
        if (i == 0) n = 1;
        else {
            float est = TAU * i / prev;
            int ratio = (int)lroundf(est);
            if (ratio < 1) ratio = 1;
            n = prev * ratio;
        }
        count[i] = n;
        first[i] = g->nc;
        for (int j = 0; j < n; j++) {
            float r = i ? (i + 0.5f) * p : 0;
            float a0 = (float)j / n * TAU - PI / 2, a1 = (float)(j + 1) / n * TAU - PI / 2, am = (a0 + a1) / 2;
            int id = add_cell(g, cx + r * cosf(am), cy + r * sinf(am));
            Cell *c = &g->c[id];
            c->ring = i; c->col = j; c->row = i; c->a0 = a0; c->a1 = a1; c->r = r;
        }
    }
    for (int i = 1; i < rings; i++) {
        int n = count[i], prev = count[i - 1];
        for (int j = 0; j < n; j++) {
            int id = first[i] + j;
            if (n > 1) add_nb(g, id, first[i] + (j + 1) % n);
            add_nb(g, id, first[i - 1] + j * prev / n);
        }
    }
    free(first); free(count);
}

/* ---- generation ---- */
enum { GEN_DFS, GEN_PRIM, GEN_KRUSKAL, GEN_WILSON, GEN_AB, GEN_ELLER, GEN_BTREE, GEN_SIDEWINDER, GEN_DIVISION,
       GEN_HUNT, GEN_GROWING, NGEN };
typedef struct { const char *key, *name, *detail; int square_only, slow; } GenDef;
static const GenDef GENS[NGEN] = {
    {"backtracker", "Recursive backtracker", "depth-first search with a stack", 0, 0},
    {"prim", "Prim's algorithm", "growing from a random frontier", 0, 0},
    {"kruskal", "Kruskal's algorithm", "random edges \xc2\xb7 union-find sets", 0, 0},
    {"wilson", "Wilson's algorithm", "loop-erased random walks", 0, 1},
    {"aldous-broder", "Aldous\xe2\x80\x93" "Broder", "a random walk, cutting into the unvisited", 0, 1},
    {"eller", "Eller's algorithm", "row by row, sets merged and carried down", 1, 0},
    {"binary-tree", "Binary tree", "each cell opens north or east", 1, 0},
    {"sidewinder", "Sidewinder", "runs eastward, closed by a cut north", 1, 0},
    {"division", "Recursive division", "walls dividing chambers", 1, 0},
    {"hunt-and-kill", "Hunt-and-kill", "random walk, then hunt for a way back in", 0, 0},
    {"growing-tree", "Growing tree", "half newest, half random", 0, 0},
};

enum { SOL_BFS, SOL_DFS, SOL_ASTAR, SOL_DIJKSTRA, SOL_WALL, SOL_DEADEND, NSOL };
typedef struct { const char *key, *name, *detail; } SolDef;
static const SolDef SOLS[NSOL] = {
    {"bfs", "Breadth-first search", "flood fill \xc2\xb7 coloured by distance"},
    {"dfs", "Depth-first search", "a stack, backing out of dead ends"},
    {"astar", "A* search", "open and closed sets \xc2\xb7 straight-line heuristic"},
    {"dijkstra", "Dijkstra's algorithm", "nearest first \xc2\xb7 weighted by length"},
    {"wall", "Wall follower", "left hand on the wall"},
    {"deadend", "Dead-end filling", "filling dead ends until one corridor is left"},
};

static unsigned char *glk;             /* recording: links */
static unsigned char *gst;             /* recording: generation state */
static Rng mrng;

static void set_link(const Graph *g, unsigned char *lk, int a, int b, int on) {
    int sa = nb_slot(g, a, b), sb = nb_slot(g, b, a);
    if (sa < 0 || sb < 0) return;
    if (on) { lk[a] |= (unsigned char)(1 << sa); lk[b] |= (unsigned char)(1 << sb); }
    else { lk[a] &= (unsigned char)~(1 << sa); lk[b] &= (unsigned char)~(1 << sb); }
}

enum { M_LINKU = 64, M_HEADU, M_STATEU, M_OPENU };   /* untimed variants (>= M_UNTIMED) */

static void carve(int a, int b, int timed_) { set_link(&G, glk, a, b, 1); emit(timed_ ? M_LINK : M_LINKU, a, b); }
static void gstate(int c, int s, int timed_) { gst[c] = (unsigned char)s; emit(timed_ ? M_STATE : M_STATEU, c, s); }

static void g_dfs(void) {
    int *stack = malloc((size_t)G.nc * sizeof(int)), sp = 0;
    int s = rint_(&mrng, G.nc);
    stack[sp++] = s; gstate(s, 3, 1); emit(M_HEADU, s, 0);
    while (sp) {
        int c = stack[sp - 1], cand[MAXNB], nc = 0;
        for (int k = 0; k < G.c[c].nn; k++) if (!gst[G.c[c].nb[k]]) cand[nc++] = G.c[c].nb[k];
        if (nc) {
            int n = cand[rint_(&mrng, nc)];
            carve(c, n, 1); gstate(n, 3, 0); emit(M_HEADU, n, 0);
            stack[sp++] = n;
        } else {
            gstate(c, 1, 1);
            sp--;
            emit(M_HEADU, sp ? stack[sp - 1] : -1, 0);
        }
    }
    free(stack);
}

static void g_prim(void) {
    int *fr = malloc((size_t)G.nc * sizeof(int)), nf = 0;
    int s = rint_(&mrng, G.nc);
    gstate(s, 1, 1);
    for (int k = 0; k < G.c[s].nn; k++) { int n = G.c[s].nb[k]; gstate(n, 2, 0); fr[nf++] = n; }
    while (nf) {
        int i = rint_(&mrng, nf), f = fr[i];
        fr[i] = fr[--nf];
        int in[MAXNB], ni = 0;
        for (int k = 0; k < G.c[f].nn; k++) if (gst[G.c[f].nb[k]] == 1) in[ni++] = G.c[f].nb[k];
        carve(in[rint_(&mrng, ni)], f, 1);
        gstate(f, 1, 0);
        emit(M_HEADU, f, 0);
        for (int k = 0; k < G.c[f].nn; k++) {
            int n = G.c[f].nb[k];
            if (!gst[n]) { gstate(n, 2, 0); fr[nf++] = n; }
        }
    }
    emit(M_HEADU, -1, 0);
    free(fr);
}

static int *uf, *ufn, *ufnext, *uftail;
static int uf_find(int x) { while (uf[x] != x) x = uf[x] = uf[uf[x]]; return x; }

static void g_kruskal(void) {
    int ne = 0;
    for (int i = 0; i < G.nc; i++) for (int k = 0; k < G.c[i].nn; k++) ne += G.c[i].nb[k] > i;
    int *ea = malloc((size_t)ne * sizeof(int)), *eb = malloc((size_t)ne * sizeof(int));
    ne = 0;
    for (int i = 0; i < G.nc; i++) for (int k = 0; k < G.c[i].nn; k++) if (G.c[i].nb[k] > i) { ea[ne] = i; eb[ne] = G.c[i].nb[k]; ne++; }
    for (int i = ne - 1; i > 0; i--) {
        int j = rint_(&mrng, i + 1), t;
        t = ea[i]; ea[i] = ea[j]; ea[j] = t;
        t = eb[i]; eb[i] = eb[j]; eb[j] = t;
    }
    uf = malloc((size_t)G.nc * sizeof(int)); ufn = malloc((size_t)G.nc * sizeof(int));
    ufnext = malloc((size_t)G.nc * sizeof(int)); uftail = malloc((size_t)G.nc * sizeof(int));
    for (int i = 0; i < G.nc; i++) { uf[i] = i; ufn[i] = 1; ufnext[i] = -1; uftail[i] = i; gst[i] = 1; }
    for (int e = 0; e < ne; e++) {
        int a = uf_find(ea[e]), b = uf_find(eb[e]);
        if (a == b) continue;
        carve(ea[e], eb[e], 1);
        if (ufn[a] < ufn[b]) { int t = a; a = b; b = t; }
        /* the smaller set takes the larger one's colour */
        for (int m = b; m >= 0; m = ufnext[m]) emit(M_SETID, m, a);
        uf[b] = a; ufn[a] += ufn[b];
        ufnext[uftail[a]] = b; uftail[a] = uftail[b];
    }
    free(ea); free(eb); free(uf); free(ufn); free(ufnext); free(uftail);
}

static void g_wilson(void) {
    int *order = malloc((size_t)G.nc * sizeof(int)), *walk = malloc((size_t)G.nc * sizeof(int));
    int *inwalk = malloc((size_t)G.nc * sizeof(int));
    for (int i = 0; i < G.nc; i++) { order[i] = i; inwalk[i] = -1; }
    for (int i = G.nc - 1; i > 0; i--) { int j = rint_(&mrng, i + 1), t = order[i]; order[i] = order[j]; order[j] = t; }
    gstate(order[0], 1, 1);
    for (int oi = 1; oi < G.nc; oi++) {
        int s = order[oi];
        if (gst[s]) continue;
        int nw = 0;
        walk[nw++] = s; inwalk[s] = 0; emit(M_WPUSH, s, 0);
        int c = s;
        while (!gst[c]) {
            int n = G.c[c].nb[rint_(&mrng, G.c[c].nn)];
            if (inwalk[n] >= 0) {
                /* erase the loop back to n */
                while (walk[nw - 1] != n) { inwalk[walk[nw - 1]] = -1; nw--; emit(M_WPOP, 0, 0); }
            } else {
                walk[nw++] = n; inwalk[n] = nw - 1; emit(M_WPUSH, n, 0);
            }
            c = n;
        }
        /* the walk reached the maze: it joins */
        for (int k = 0; k + 1 < nw; k++) { carve(walk[k], walk[k + 1], 0); gstate(walk[k], 1, 0); }
        for (int k = 0; k < nw; k++) inwalk[walk[k]] = -1;
        emit(M_WCLEAR, 0, 0);
    }
    free(order); free(walk); free(inwalk);
}

static void g_ab(void) {
    int c = rint_(&mrng, G.nc), left = G.nc - 1;
    gstate(c, 1, 1); emit(M_HEADU, c, 0);
    while (left) {
        int n = G.c[c].nb[rint_(&mrng, G.c[c].nn)];
        if (!gst[n]) { carve(c, n, 1); gstate(n, 1, 0); left--; emit(M_HEADU, n, 0); }
        else emit(M_HEAD, n, 0);
        c = n;
    }
    emit(M_HEADU, -1, 0);
}

static int sqc(int col, int row) { return col >= 0 && row >= 0 && col < G.cols && row < G.rows ? G.sq[row * G.cols + col] : -1; }

static void g_eller(void) {
    int cols = G.cols, *set = malloc((size_t)G.nc * sizeof(int)), next = 0;
    for (int r = 0; r < G.rows; r++) {
        emit(M_ROW, r, 0);
        for (int c = 0; c < cols; c++) {
            int id = sqc(c, r);
            if (r == 0 || set[id] < 0) set[id] = next++;
            emit(M_SETID, id, set[id]);
            gstate(id, 3, 0);
        }
        /* join neighbours in this row */
        for (int c = 0; c + 1 < cols; c++) {
            int a = sqc(c, r), b = sqc(c + 1, r);
            if (set[a] != set[b] && (r == G.rows - 1 || rint_(&mrng, 2))) {
                carve(a, b, 1);
                int old = set[b];
                for (int k = 0; k < cols; k++) {
                    int x = sqc(k, r);
                    if (set[x] == old) { set[x] = set[a]; emit(M_SETID, x, set[a]); }
                }
            }
        }
        if (r == G.rows - 1) break;
        /* at least one way down from each set */
        for (int c = 0; c < cols; c++) set[sqc(c, r + 1)] = -1;
        for (int c = 0; c < cols;) {
            int s = set[sqc(c, r)], e = c;
            while (e < cols && set[sqc(e, r)] == s) e++;
            /* a run of the same set is contiguous only by luck; gather the whole set */
            int mem[4096], nm = 0;
            for (int k = 0; k < cols; k++) if (set[sqc(k, r)] == s && nm < 4096) mem[nm++] = k;
            if (mem[0] == c) {
                int forced = mem[rint_(&mrng, nm)];
                for (int q = 0; q < nm; q++) {
                    int k = mem[q];
                    if (k == forced || rint_(&mrng, 3) == 0) {
                        int a = sqc(k, r), b = sqc(k, r + 1);
                        carve(a, b, 1);
                        set[b] = s;
                    }
                }
            }
            c = e;
        }
        for (int c = 0; c < cols; c++) gstate(sqc(c, r), 1, 0);
    }
    for (int c = 0; c < cols; c++) gstate(sqc(c, G.rows - 1), 1, 0);
    emit(M_ROW, -1, 0);
    free(set);
}

static void g_btree(void) {
    for (int r = 0; r < G.rows; r++)
        for (int c = 0; c < G.cols; c++) {
            int id = sqc(c, r), n = sqc(c, r - 1), e = sqc(c + 1, r), to = -1;
            if (n >= 0 && e >= 0) to = rint_(&mrng, 2) ? n : e;
            else to = n >= 0 ? n : e;
            gstate(id, 1, to < 0);
            if (to >= 0) carve(id, to, 1);
            emit(M_HEADU, id, 0);
        }
    emit(M_HEADU, -1, 0);
}

static void g_sidewinder(void) {
    int *run = malloc((size_t)G.cols * sizeof(int));
    for (int r = 0; r < G.rows; r++) {
        int nr = 0;
        for (int c = 0; c < G.cols; c++) {
            int id = sqc(c, r);
            run[nr++] = id;
            gstate(id, 3, 0);
            emit(M_HEADU, id, 0);
            int close = c == G.cols - 1 || (r > 0 && rint_(&mrng, 2));
            if (close) {
                if (r > 0) {
                    int m = run[rint_(&mrng, nr)];
                    carve(m, sqc(G.c[m].col, r - 1), 1);
                } else emit(M_STEP, id, 0);
                for (int k = 0; k < nr; k++) gstate(run[k], 1, 0);
                nr = 0;
            } else carve(id, sqc(c + 1, r), 1);
        }
    }
    emit(M_HEADU, -1, 0);
    free(run);
}

static void divide(int x, int y, int w, int h) {
    if (w < 2 || h < 2) return;
    emit(M_RECT, x | y << 16, w | h << 16);
    int horiz = h > w ? 1 : w > h ? 0 : rint_(&mrng, 2);
    if (horiz) {
        int wy = y + rint_(&mrng, h - 1), gap = x + rint_(&mrng, w);
        for (int c = x; c < x + w; c++)
            if (c != gap) { int a = sqc(c, wy), b = sqc(c, wy + 1); set_link(&G, glk, a, b, 0); emit(M_UNLINK, a, b); }
        divide(x, y, w, wy - y + 1);
        divide(x, wy + 1, w, y + h - wy - 1);
    } else {
        int wx = x + rint_(&mrng, w - 1), gap = y + rint_(&mrng, h);
        for (int r = y; r < y + h; r++)
            if (r != gap) { int a = sqc(wx, r), b = sqc(wx + 1, r); set_link(&G, glk, a, b, 0); emit(M_UNLINK, a, b); }
        divide(x, y, wx - x + 1, h);
        divide(wx + 1, y, x + w - wx - 1, h);
    }
}
static void g_division(void) {
    for (int i = 0; i < G.nc; i++) {
        gstate(i, 1, 0);
        for (int k = 0; k < G.c[i].nn; k++) if (G.c[i].nb[k] > i) carve(i, G.c[i].nb[k], 0);
    }
    divide(0, 0, G.cols, G.rows);
    emit(M_RECT, 0, 0);
}

static void g_hunt(void) {
    int c = rint_(&mrng, G.nc);
    gstate(c, 1, 1); emit(M_HEADU, c, 0);
    int hunted_from = 0;
    for (;;) {
        int cand[MAXNB], nc = 0;
        for (int k = 0; k < G.c[c].nn; k++) if (!gst[G.c[c].nb[k]]) cand[nc++] = G.c[c].nb[k];
        if (nc) {
            int n = cand[rint_(&mrng, nc)];
            carve(c, n, 1); gstate(n, 1, 0); emit(M_HEADU, n, 0);
            c = n;
            continue;
        }
        /* hunt: scan for an unvisited cell beside the maze */
        int found = -1, from = -1, row = -1;
        emit(M_HEADU, -1, 0);
        for (int i = hunted_from; i < G.nc && found < 0; i++) {
            if (G.c[i].row != row) { row = G.c[i].row; emit(M_ROW, row, 0); emit(M_STEP, i, 0); }
            if (gst[i]) { if (i == hunted_from) hunted_from++; continue; }
            int in[MAXNB], ni = 0;
            for (int k = 0; k < G.c[i].nn; k++) if (gst[G.c[i].nb[k]]) in[ni++] = G.c[i].nb[k];
            if (ni) { found = i; from = in[rint_(&mrng, ni)]; }
        }
        if (found < 0) break;
        emit(M_ROW, -1, 0);
        carve(from, found, 1); gstate(found, 1, 0); emit(M_HEADU, found, 0);
        c = found;
    }
    emit(M_ROW, -1, 0);
}

static void g_growing(void) {
    int *list = malloc((size_t)G.nc * sizeof(int)), nl = 0;
    int s = rint_(&mrng, G.nc);
    gstate(s, 3, 1); list[nl++] = s;
    while (nl) {
        int i = rint_(&mrng, 2) ? nl - 1 : rint_(&mrng, nl), c = list[i];
        int cand[MAXNB], nc = 0;
        for (int k = 0; k < G.c[c].nn; k++) if (!gst[G.c[c].nb[k]]) cand[nc++] = G.c[c].nb[k];
        if (nc) {
            int n = cand[rint_(&mrng, nc)];
            carve(c, n, 1); gstate(n, 3, 0); emit(M_HEADU, n, 0);
            list[nl++] = n;
        } else {
            gstate(c, 1, 1);
            list[i] = list[--nl];
        }
    }
    emit(M_HEADU, -1, 0);
    free(list);
}

static void (*const GEN_FN[NGEN])(void) = {g_dfs, g_prim, g_kruskal, g_wilson, g_ab, g_eller, g_btree,
                                           g_sidewinder, g_division, g_hunt, g_growing};

/* ---- solving (on the finished maze, links in glk) ---- */
static int *tdist, *tpar;              /* true distances and BFS parents from the start */

static void true_paths(const unsigned char *lk, int start) {
    int *q = malloc((size_t)G.nc * sizeof(int)), qh = 0, qt = 0;
    for (int i = 0; i < G.nc; i++) { tdist[i] = -1; tpar[i] = -1; }
    tdist[start] = 0; q[qt++] = start;
    while (qh < qt) {
        int c = q[qh++];
        for (int k = 0; k < G.c[c].nn; k++)
            if (lk[c] >> k & 1) {
                int n = G.c[c].nb[k];
                if (tdist[n] < 0) { tdist[n] = tdist[c] + 1; tpar[n] = c; q[qt++] = n; }
            }
    }
    free(q);
}

static void v_bfs(int s, int g) {
    (void)g;
    int *q = malloc((size_t)G.nc * sizeof(int)), *d = malloc((size_t)G.nc * sizeof(int)), qh = 0, qt = 0;
    for (int i = 0; i < G.nc; i++) d[i] = -1;
    d[s] = 0; q[qt++] = s; emit(M_VISIT, s, 0);
    while (qh < qt) {
        int c = q[qh++];
        for (int k = 0; k < G.c[c].nn; k++)
            if (glk[c] >> k & 1) {
                int n = G.c[c].nb[k];
                if (d[n] < 0) { d[n] = d[c] + 1; q[qt++] = n; emit(M_PARENT, n, c); emit(M_VISIT, n, d[n]); }
            }
    }
    free(q); free(d);
}

static void v_dfs(int s, int g) {
    int *stack = malloc((size_t)G.nc * sizeof(int)), sp = 0, *seen = calloc((size_t)G.nc, sizeof(int));
    int *it = calloc((size_t)G.nc, sizeof(int));
    stack[sp++] = s; seen[s] = 1; emit(M_PUSH, s, 0);
    while (sp) {
        int c = stack[sp - 1];
        if (c == g) break;
        int moved = 0;
        while (it[c] < G.c[c].nn) {
            int k = it[c]++, n = G.c[c].nb[k];
            if ((glk[c] >> k & 1) && !seen[n]) { seen[n] = 1; stack[sp++] = n; emit(M_PUSH, n, 0); moved = 1; break; }
        }
        if (!moved) { sp--; emit(M_POP, c, 0); }
    }
    free(stack); free(seen); free(it);
}

/* a binary heap of (key, cell) */
static float *hk;
static int *hc, hn;
static void hpush(float k, int c) {
    int i = hn++;
    while (i > 0 && hk[(i - 1) / 2] > k) { hk[i] = hk[(i - 1) / 2]; hc[i] = hc[(i - 1) / 2]; i = (i - 1) / 2; }
    hk[i] = k; hc[i] = c;
}
static int hpop(float *key) {
    int top = hc[0];
    *key = hk[0];
    float k = hk[--hn];
    int c = hc[hn], i = 0;
    for (;;) {
        int l = 2 * i + 1;
        if (l >= hn) break;
        if (l + 1 < hn && hk[l + 1] < hk[l]) l++;
        if (hk[l] >= k) break;
        hk[i] = hk[l]; hc[i] = hc[l]; i = l;
    }
    hk[i] = k; hc[i] = c;
    return top;
}

static float cdist(int a, int b) { return hypotf(G.c[a].x - G.c[b].x, G.c[a].y - G.c[b].y); }

/* the length of a link as walked: theta ring links follow the arc */
static float link_len(int a, int b) {
    if (G.grid == G_THETA && G.c[a].ring == G.c[b].ring && G.c[a].ring > 0) {
        float da = fabsf((G.c[a].a0 + G.c[a].a1) - (G.c[b].a0 + G.c[b].a1)) / 2;
        if (da > PI) da = TAU - da;
        return da * G.c[a].r;
    }
    return cdist(a, b);
}

static void v_astar_dijkstra(int s, int g, int astar) {
    hk = malloc((size_t)G.nc * 8 * sizeof(float)); hc = malloc((size_t)G.nc * 8 * sizeof(int)); hn = 0;
    float *gs = malloc((size_t)G.nc * sizeof(float));
    unsigned char *closed = calloc((size_t)G.nc, 1);
    for (int i = 0; i < G.nc; i++) gs[i] = 1e30f;
    gs[s] = 0;
    hpush(astar ? cdist(s, g) : 0, s);
    emit(M_OPENU, s, -1);
    while (hn) {
        float key;
        int c = hpop(&key);
        if (closed[c]) continue;
        closed[c] = 1;
        emit(M_CLOSE, c, (int)lroundf(gs[c] / G.pitch * 16));
        if (astar && c == g) break;
        for (int k = 0; k < G.c[c].nn; k++)
            if (glk[c] >> k & 1) {
                int n = G.c[c].nb[k];
                float ng = gs[c] + link_len(c, n);
                if (!closed[n] && ng < gs[n]) {
                    gs[n] = ng;
                    hpush(ng + (astar ? cdist(n, g) : 0), n);
                    emit(M_OPENU, n, c);
                }
            }
    }
    free(hk); free(hc); free(gs); free(closed);
}
static void v_astar(int s, int g) { v_astar_dijkstra(s, g, 1); }
static void v_dijkstra(int s, int g) { v_astar_dijkstra(s, g, 0); }

static void v_wall(int s, int g) {
    int c = s, from = -1, guard = 4 * G.nc + 8;
    emit(M_STEP, c, 0);
    while (c != g && guard--) {
        int base = from >= 0 ? nb_slot(&G, c, from) : G.c[c].nn - 1, nn = G.c[c].nn, next = -1;
        for (int i = 1; i <= nn; i++) {
            int k = (base + i) % nn;
            if (glk[c] >> k & 1) { next = G.c[c].nb[k]; break; }
        }
        if (next < 0) break;
        from = c; c = next;
        emit(M_STEP, c, 0);
    }
}

static void v_deadend(int s, int g) {
    int *deg = calloc((size_t)G.nc, sizeof(int)), *q = malloc((size_t)G.nc * sizeof(int)), qh = 0, qt = 0;
    unsigned char *filled = calloc((size_t)G.nc, 1);
    for (int i = 0; i < G.nc; i++)
        for (int k = 0; k < G.c[i].nn; k++) deg[i] += glk[i] >> k & 1;
    for (int i = 0; i < G.nc; i++) if (deg[i] <= 1 && i != s && i != g) q[qt++] = i;
    while (qh < qt) {
        int c = q[qh++];
        if (filled[c]) continue;
        filled[c] = 1;
        emit(M_FILL, c, 0);
        for (int k = 0; k < G.c[c].nn; k++)
            if (glk[c] >> k & 1) {
                int n = G.c[c].nb[k];
                if (!filled[n] && --deg[n] == 1 && n != s && n != g) q[qt++] = n;
            }
    }
    free(deg); free(q); free(filled);
}

static void (*const SOL_FN[NSOL])(int, int) = {v_bfs, v_dfs, v_astar, v_dijkstra, v_wall, v_deadend};

/* ---- maze playback ---- */
typedef struct {
    int gen, solve;
    Log lg, ls;
    Pace pg, ps;
    int nc;
    unsigned char *lk, *st, *ss;       /* links, generation state, solve state */
    int *order, *setid, *lastg;        /* when each cell joined (timed index), its set, last touched */
    int *walk, nwalk, head, row, rect[4];
    int gpos, gdone;
    int *dist, *par, *lasts, *stack, nstack, shead, spos, sdone;
    long visited, steps, filled;
    int *path, npath, maxd, *tdist;
    unsigned char *flk;                /* the finished maze's links */
} MazePlay;
static MazePlay MP;

static void mp_reset_gen(MazePlay *m) {
    memset(m->lk, 0, (size_t)m->nc);
    memset(m->st, 0, (size_t)m->nc);
    for (int i = 0; i < m->nc; i++) { m->order[i] = -1; m->setid[i] = i; m->lastg[i] = -1000000; }
    m->nwalk = 0; m->head = -1; m->row = -1; memset(m->rect, 0, sizeof m->rect);
    m->gpos = m->gdone = 0;
}
static void mp_reset_sol(MazePlay *m) {
    memset(m->ss, 0, (size_t)m->nc);
    for (int i = 0; i < m->nc; i++) { m->dist[i] = -1; m->par[i] = -1; m->lasts[i] = -1000000; }
    m->nstack = 0; m->shead = -1; m->spos = m->sdone = 0;
    m->visited = m->steps = m->filled = 0;
}

static void mp_apply_gen(MazePlay *m, const Ev *e) {
    int t = m->gdone;
    switch (e->k) {
    case M_LINK: case M_LINKU:
        set_link(&G, m->lk, e->a, e->b, 1);
        if (m->order[e->a] < 0) m->order[e->a] = t;
        if (m->order[e->b] < 0) m->order[e->b] = t;
        m->lastg[e->a] = m->lastg[e->b] = t;
        break;
    case M_UNLINK: set_link(&G, m->lk, e->a, e->b, 0); m->lastg[e->a] = m->lastg[e->b] = t; break;
    case M_HEAD: case M_HEADU: m->head = e->a; if (e->a >= 0) m->lastg[e->a] = t; break;
    case M_STATE: case M_STATEU:
        m->st[e->a] = (unsigned char)e->b;
        if (e->b == 1 && m->order[e->a] < 0) m->order[e->a] = t;
        m->lastg[e->a] = t;
        break;
    case M_WPUSH: m->walk[m->nwalk++] = e->a; m->lastg[e->a] = t; break;
    case M_WPOP: if (m->nwalk) m->nwalk--; break;
    case M_WCLEAR: m->nwalk = 0; break;
    case M_SETID: m->setid[e->a] = e->b; break;
    case M_RECT: m->rect[0] = e->a & 0xffff; m->rect[1] = e->a >> 16; m->rect[2] = e->b & 0xffff; m->rect[3] = e->b >> 16; break;
    case M_ROW: m->row = e->a; break;
    case M_STEP: if (e->a >= 0) m->lastg[e->a] = t; break;
    }
    if (timed(e->k)) m->gdone++;
}

static void mp_apply_sol(MazePlay *m, const Ev *e) {
    int t = m->sdone;
    switch (e->k) {
    case M_VISIT: m->ss[e->a] = 2; m->dist[e->a] = e->b * 16; m->visited++; m->lasts[e->a] = t; break;
    case M_PARENT: m->par[e->a] = e->b; break;
    case M_OPENU: if (m->ss[e->a] != 2) m->ss[e->a] = 1; m->par[e->a] = e->b; break;
    case M_CLOSE: m->ss[e->a] = 2; m->dist[e->a] = e->b; m->visited++; m->lasts[e->a] = t; m->shead = e->a; break;
    case M_PUSH:
        m->stack[m->nstack++] = e->a; m->ss[e->a] = 3; m->visited++; m->lasts[e->a] = t;
        m->dist[e->a] = (m->nstack - 1) * 16;
        break;
    case M_POP: if (m->nstack) m->nstack--; m->ss[e->a] = 2; m->lasts[e->a] = t; break;
    case M_STEP:
        if (!m->ss[e->a]) { m->visited++; m->dist[e->a] = (int)m->steps * 16; }
        m->ss[e->a] = m->ss[e->a] >= 2 ? 3 : 2;
        m->shead = e->a; m->lasts[e->a] = t;
        break;
    case M_FILL: m->ss[e->a] = 4; m->filled++; m->lasts[e->a] = t; break;
    }
    if (timed(e->k)) { m->sdone++; m->steps++; }
}

static void mp_seek_gen(MazePlay *m, long steps) {
    if (steps > m->lg.nt) steps = m->lg.nt;
    if (steps < m->gdone) mp_reset_gen(m);
    int end = steps >= m->lg.nt ? m->lg.n : m->lg.tp[steps];
    while (m->gpos < end) mp_apply_gen(m, &m->lg.e[m->gpos++]);
}
static void mp_seek_sol(MazePlay *m, long steps) {
    if (steps > m->ls.nt) steps = m->ls.nt;
    if (steps < m->sdone) mp_reset_sol(m);
    int end = steps >= m->ls.nt ? m->ls.n : m->ls.tp[steps];
    while (m->spos < end) mp_apply_sol(m, &m->ls.e[m->spos++]);
}

static void mp_free(MazePlay *m) {
    log_free(&m->lg); log_free(&m->ls);
    free(m->lk); free(m->st); free(m->ss); free(m->order); free(m->setid); free(m->lastg); free(m->walk);
    free(m->dist); free(m->par); free(m->lasts); free(m->stack); free(m->path); free(m->flk); free(m->tdist);
    memset(m, 0, sizeof *m);
}

/* record generation and solving on the current graph G */
static int maze_record(MazePlay *m, int gen, int solve, uint32_t seed) {
    mp_free(m);
    m->gen = gen; m->solve = solve; m->nc = G.nc;
    int n = G.nc;
    glk = calloc((size_t)n, 1); gst = calloc((size_t)n, 1);
    mrng = rng_of(seed, 555);
    rec = &m->lg;
    GEN_FN[gen]();
    rec = NULL;
    log_finish(&m->lg);
    /* start and goal */
    tdist = malloc((size_t)n * sizeof(int)); tpar = malloc((size_t)n * sizeof(int));
    if (G.grid == G_THETA) {
        /* from the centre to the rim, where the way out is longest */
        G.start = 0;
        true_paths(glk, G.start);
        int outer0 = n - 1;
        while (outer0 > 0 && G.c[outer0 - 1].ring == G.c[n - 1].ring) outer0--;
        G.goal = outer0;
        for (int i = outer0; i < n; i++) if (tdist[i] > tdist[G.goal]) G.goal = i;
    } else {
        int s = 0, g = 0;
        for (int i = 0; i < n; i++) {
            if (G.c[i].x + G.c[i].y < G.c[s].x + G.c[s].y) s = i;
            if (G.c[i].x + G.c[i].y > G.c[g].x + G.c[g].y) g = i;
        }
        G.start = s; G.goal = g;
    }
    true_paths(glk, G.start);
    m->maxd = 1;
    for (int i = 0; i < n; i++) if (tdist[i] > m->maxd) m->maxd = tdist[i];
    m->npath = tdist[G.goal] + 1;
    m->path = malloc((size_t)(m->npath > 0 ? m->npath : 1) * sizeof(int));
    for (int c = G.goal, k = m->npath - 1; c >= 0 && k >= 0; c = tpar[c], k--) m->path[k] = c;
    rec = &m->ls;
    SOL_FN[solve](G.start, G.goal);
    rec = NULL;
    log_finish(&m->ls);
    m->flk = glk; glk = NULL;
    free(gst); gst = NULL;
    m->tdist = tdist; tdist = NULL;
    free(tpar);
    m->lk = calloc((size_t)n, 1); m->st = calloc((size_t)n, 1); m->ss = calloc((size_t)n, 1);
    m->order = malloc((size_t)n * sizeof(int)); m->setid = malloc((size_t)n * sizeof(int));
    m->lastg = malloc((size_t)n * sizeof(int)); m->walk = malloc((size_t)n * sizeof(int));
    m->dist = malloc((size_t)n * sizeof(int)); m->par = malloc((size_t)n * sizeof(int));
    m->lasts = malloc((size_t)n * sizeof(int)); m->stack = malloc((size_t)n * sizeof(int));
    mp_reset_gen(m);
    mp_reset_sol(m);
    return 0;
}

/* ---- maze acts ---- */
#define MZ_LEAD 1.6
#define MZ_GAP 2.4
#define MZ_TRACE 2.6
#define MZ_HOLD 5.0

static int find_gen(const char *s) { for (int i = 0; i < NGEN; i++) if (!strcmp(s, GENS[i].key)) return i; return -1; }
static int find_sol(const char *s) { for (int i = 0; i < NSOL; i++) if (!strcmp(s, SOLS[i].key)) return i; return -1; }

static void maze_params(Act *a, Rng *r, int span_mode) {
    int g = opt_grid ? find_name(opt_grid, GRIDS, NGRID) : -1;
    int fg = opt_maze ? find_gen(opt_maze) : -1;
    int fs = opt_solve ? find_sol(opt_solve) : -1;
    if (g < 0) {
        int x = rint_(r, 100);
        g = span_mode ? (x < 70 ? G_SQUARE : G_HEX) : (x < 55 ? G_SQUARE : x < 78 ? G_HEX : G_THETA);
        if (fg >= 0 && GENS[fg].square_only) g = G_SQUARE;
    }
    if (span_mode && g == G_THETA) g = G_SQUARE;
    a->grid = g;
    int gen = fg;
    if (gen >= 0 && GENS[gen].square_only && (g != G_SQUARE || span_mode)) gen = -1;
    for (int tries = 0; gen < 0; tries++) {
        int x = rint_(r, NGEN);
        if (GENS[x].square_only && (g != G_SQUARE || span_mode)) continue;
        if (GENS[x].slow && span_mode) continue;
        int rec_ = 0;
        for (int i = 0; i < 3; i++) rec_ |= recent_gen[i] == x + 1;
        if (rec_ && tries < 60) continue;
        gen = x;
    }
    memmove(recent_gen + 1, recent_gen, 2 * sizeof(int));
    recent_gen[0] = gen + 1;
    a->gen = gen;
    int sol = fs;
    while (sol < 0) {
        int x = rint_(r, NSOL);
        if (x == SOL_DIJKSTRA && g != G_THETA) continue;
        if (x == SOL_BFS && g == G_THETA && rint_(r, 2)) continue;
        sol = x;
    }
    a->solve = sol;
    a->density = rint_(r, 3);
    a->T = GENS[gen].slow ? 46 : 34;
    static const double ts[NSOL] = {22, 20, 18, 22, 22, 18};
    a->Ts = ts[sol];
    a->dur = MZ_LEAD + a->T + MZ_GAP + a->Ts + MZ_TRACE + MZ_HOLD + FADE;
}

/* ---- screen layout (in units of u, so the same in logical units) ---- */
static struct {
    float u, mx, my;
    float dx0, dy0, dx1, dy1;          /* the data rect */
    float title_y, sub_y, lab_y, num_y, head_h, foot_h;
    int portrait;
    int s_title, s_sub, s_label, s_num, s_clock, s_small, s_mid;
    int w, h;
} L;

/* header and footer heights for a screen whose short side is `shortside` */
static void hud_metrics(float shortside, float *head, float *foot, float *mx, float *my) {
    float u = shortside / 1080.0f;
    *mx = roundf(54 * u); *my = roundf(46 * u);
    *head = *my + 30 * u + 26 * u + 30 * u;
    *foot = *my + 21 * u + 12 * u + 36 * u;
}

static GLuint tex_atlas;

static void layout(void) {
    L.w = W; L.h = H;
    L.portrait = H > W * 1.05f;
    float u = fminf(W, H) / 1080.0f;
    L.u = u;
    hud_metrics(fminf(W, H), &L.head_h, &L.foot_h, &L.mx, &L.my);
    nstyles = 0;
    L.s_title = style(F_SEMI, 30 * u);
    L.s_sub = style(F_REG, 16.5f * u);
    L.s_label = style(F_SEMI, 10.5f * u);
    L.s_num = style(F_MONO, 21 * u);
    L.s_clock = style(F_LIGHT, 34 * u);
    L.s_small = style(F_REG, 13.5f * u);
    L.s_mid = style(F_LIGHT, 22 * u);
    build_atlas();
    L.title_y = L.my + 30 * u * 0.73f;
    L.sub_y = L.title_y + 26 * u;
    L.num_y = H - L.my;
    L.lab_y = L.num_y - 21 * u - 10 * u;
    L.dx0 = L.mx; L.dx1 = W - L.mx;
    L.dy0 = L.head_h; L.dy1 = H - L.foot_h;
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

/* ---- the act being played ---- */
static struct {
    int index, w, h;
    int lane, place, nlanes;           /* race: this output's lane and finishing place */
    double fin[MAX_OUT];               /* race: each lane's finishing time (act-local) */
    int alg, form;
    SortPlay sp;
    GLuint tex_pic, tex_perm;
    float mz_ox, mz_oy, mz_sc;         /* maze: logical -> this output's pixels */
    float hw;                          /* maze passage half-width (pixels) */
    int maxd16;
    int nlinks;
} cur = {.index = -1};

/* outputs: left-to-right order, and front-first order */
static int lane_of(int o) {
    int r = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] < span[o][0] || (span[i][0] == span[o][0] && i < o)) r++;
    return r;
}
static int ff_of(int o) {
    if (o == front) return 0;
    return 1 + lane_of(o) - (lane_of(front) < lane_of(o));
}

/* ---- pictures for the image form ---- */
static float vhash(int x, int y, uint32_t s) { return (hash3((uint32_t)x, (uint32_t)y, s) & 0xffffff) / 16777216.0f; }
static float vnoise(float x, float y, uint32_t s) {
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float fx = x - xi, fy = y - yi;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    float a = vhash(xi, yi, s), b = vhash(xi + 1, yi, s), c = vhash(xi, yi + 1, s), d = vhash(xi + 1, yi + 1, s);
    return mixf(mixf(a, b, fx), mixf(c, d, fx), fy);
}
static float fbm(float x, float y, uint32_t s, int oct) {
    float v = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) { v += a * vnoise(x, y, s + i * 101); n += a; a *= 0.5f; x *= 2.03f; y *= 2.03f; }
    return v / n;
}

#define PIC_W 480
/* the picture at (x, y) in [0,1]^2 (y down), asp = width / height */
static void picture(int kind, uint32_t s, float x, float y, float asp, float *o) {
    float deep[3];
    memcpy(deep, pal.light ? pal.fg : pal.bg, sizeof deep);
    if (kind == 0) {
        /* ridges at dusk under a low sun */
        float sky = clampf(y / 0.62f, 0, 1);
        ramp(mixf(0.95f, 0.18f, sky), o);
        float sx = 0.68f, sy = 0.40f, dx = (x - sx) * asp, dy = y - sy, d = sqrtf(dx * dx + dy * dy);
        float sun[3];
        ramp(0.12f, sun);
        mix3(sun, pal.hi, 0.35f, sun);
        mix3(o, sun, clampf((0.085f - d) * 400, 0, 1), o);
        mix3(o, sun, 0.25f * expf(-d * 9), o);
        for (int l = 0; l < 5; l++) {
            float base = 0.48f + l * 0.09f, amp = 0.05f + l * 0.025f;
            float h = base - amp * (fbm(x * asp * (2.2f + l * 0.9f), l * 7.3f, s + l, 5) * 2 - 0.6f);
            if (y > h) {
                float c[3];
                ramp(0.30f + l * 0.13f, c);
                mix3(c, deep, 0.25f + l * 0.16f, c);
                float shade = clampf((y - h) * 6, 0, 1);
                mix3(c, deep, shade * 0.25f, c);
                memcpy(o, c, sizeof c);
            }
        }
    } else if (kind == 1) {
        /* interference of two ring systems */
        float ax = 0.33f * asp, ay = 0.45f, bx = 0.70f * asp, by = 0.58f;
        float d1 = hypotf(x * asp - ax, y - ay), d2 = hypotf(x * asp - bx, y - by);
        float t = 0.5f + 0.5f * sinf(d1 * 46) * cosf(d2 * 38);
        float band = fmodf(d1 * 3.2f + d2 * 1.4f, 1.0f);
        ramp(band, o);
        mix3(o, deep, (1 - t) * 0.55f, o);
    } else {
        /* a warped flow of colour with contour lines */
        float px = x * asp * 2.2f, py = y * 2.2f;
        float qx = fbm(px, py, s, 4), qy = fbm(px + 5.2f, py + 1.3f, s, 4);
        float t = fbm(px + 2.2f * qx, py + 2.2f * qy, s + 9, 3);
        t = clampf((t - 0.28f) / 0.44f, 0, 1);
        /* soft terraces */
        float band = floorf(t * 7) / 7, f = t * 7 - floorf(t * 7);
        t = band + smooth(0.75f, 1.0f, f) / 7;
        ramp(t, o);
        mix3(o, deep, 0.18f * (1 - smooth(0.0f, 0.25f, f)), o);
    }
}

static void make_picture(SortPlay *s, int kind, uint32_t seed, int vert, float asp) {
    free(s->pic);
    int n = s->n;
    s->pic = malloc((size_t)PIC_W * n * 3);
    s->pic_w = PIC_W; s->pic_kind = kind; s->pic_vert = vert;
    for (int st = 0; st < n; st++)
        for (int k = 0; k < PIC_W; k++) {
            float al = (st + 0.5f) / n, ac = (k + 0.5f) / PIC_W, o[3];
            picture(kind, seed, vert ? ac : al, vert ? al : ac, asp, o);
            unsigned char *p = s->pic + ((size_t)st * PIC_W + k) * 3;
            for (int c = 0; c < 3; c++) p[c] = (unsigned char)lroundf(clampf(o[c], 0, 1) * 255);
        }
    glBindTexture(GL_TEXTURE_2D, cur.tex_pic);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, PIC_W, n, 0, GL_RGB, GL_UNSIGNED_BYTE, s->pic);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/* ---- spanning maze mask (logical units) ---- */
static float span_head, span_foot, span_margin;
static int in_union(float x, float y) {
    for (int i = 0; i < span_n; i++)
        if (x >= span[i][0] && x <= span[i][0] + span[i][2] && y >= span[i][1] && y <= span[i][1] + span[i][3]) return 1;
    return 0;
}
static int span_mask(float x, float y, float half) {
    float e = half + span_margin;
    if (!in_union(x - e, y - e) || !in_union(x + e, y - e) || !in_union(x - e, y + e) || !in_union(x + e, y + e) ||
        !in_union(x, y))
        return 0;
    const float *f = span[front];
    if (x + half > f[0] && x - half < f[0] + f[2]) {
        if (y - half < f[1] + span_head && y + half > f[1]) return 0;
        if (y + half > f[1] + f[3] - span_foot && y - half < f[1] + f[3]) return 0;
    }
    return 1;
}

static void build_maze_act(const Act *a) {
    float p;
    if (spanning()) {
        const float *f = span[front];
        float fshort = fminf(f[2], f[3]), mx, my;
        hud_metrics(fshort, &span_head, &span_foot, &mx, &my);
        static const float rows_on_front[3] = {22, 30, 40};
        p = f[3] / rows_on_front[a->density];
        span_margin = my * 0.6f;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int i = 0; i < span_n; i++) {
            x0 = fminf(x0, span[i][0]); y0 = fminf(y0, span[i][1]);
            x1 = fmaxf(x1, span[i][0] + span[i][2]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
        }
        /* align the lattice to the front output's centre */
        float cx = f[0] + f[2] / 2, cy = f[1] + f[3] / 2;
        float ox = cx - ceilf((cx - x0) / p) * p, oy = cy - ceilf((cy - y0) / p) * p;
        int cols = (int)((x1 - ox) / p) + 1, rows = (int)((y1 - oy) / (a->grid == G_HEX ? p * 0.8660254f : p)) + 1;
        if (a->grid == G_HEX) build_hex(&G, ox, oy, cols, rows, p, span_mask);
        else build_square(&G, ox, oy, cols, rows, p, span_mask);
        graph_prune(&G);
        cur.mz_ox = span[span_me][0]; cur.mz_oy = span[span_me][1]; cur.mz_sc = W / span[span_me][2];
    } else {
        float w = L.dx1 - L.dx0, h = L.dy1 - L.dy0;
        if (a->grid == G_THETA) {
            static const int rings[3] = {9, 12, 16};
            int R = rings[a->density];
            p = fminf(w, h) / 2 / (R + 0.15f);
            build_theta(&G, (L.dx0 + L.dx1) / 2, (L.dy0 + L.dy1) / 2, R, p);
        } else {
            static const float shortcells[3] = {17, 24, 33};
            p = fminf(w, h) / shortcells[a->density];
            if (a->grid == G_HEX) {
                int cols = (int)((w - p * 0.5f) / p), rows = (int)((h - p) / (p * 0.8660254f)) + 1;
                float gw = (cols + 0.5f) * p, gh = p + (rows - 1) * p * 0.8660254f;
                build_hex(&G, L.dx0 + (w - gw) / 2, L.dy0 + (h - gh) / 2, cols, rows, p, NULL);
            } else {
                int cols = (int)(w / p), rows = (int)(h / p);
                build_square(&G, L.dx0 + (w - cols * p) / 2, L.dy0 + (h - rows * p) / 2, cols, rows, p, NULL);
            }
        }
        cur.mz_ox = 0; cur.mz_oy = 0; cur.mz_sc = 1;
    }
    sort_nb(&G);
    maze_record(&MP, a->gen, a->solve, a->seed);
    pace_init(&MP.pg, MP.lg.nt, a->T, 4);
    pace_init(&MP.ps, MP.ls.nt, a->Ts, 5);
    /* the colour scale of the solver's distances: replay it once */
    mp_seek_sol(&MP, MP.ls.nt);
    cur.maxd16 = 16;
    for (int i = 0; i < MP.nc; i++) if (MP.dist[i] > cur.maxd16) cur.maxd16 = MP.dist[i];
    mp_reset_sol(&MP);
    cur.hw = G.pitch * cur.mz_sc * (G.grid == G_HEX ? 0.27f : 0.31f);
}

static void build_act(const Act *a, int index) {
    cur.index = index; cur.w = W; cur.h = H;
    cur.lane = spanning() ? lane_of(span_me) : 0;
    cur.nlanes = spanning() ? span_n : 1;
    sp_free(&cur.sp);
    if (a->kind == K_MAZE) { build_maze_act(a); return; }
    int ff = spanning() ? ff_of(span_me) : 0;
    cur.alg = a->kind == K_RACE ? a->algs[cur.lane] : a->alg;
    cur.form = a->kind == K_TRIO ? a->forms[ff] : a->form;
    int portrait = L.portrait;
    sp_build(&cur.sp, cur.alg, cur.form, a->input, a->n, a->seed);
    if (a->kind == K_RACE) {
        /* every lane at the same rate: the pace is set by the slowest */
        long S[MAX_OUT], smax = 1;
        for (int i = 0; i < span_n; i++) {
            int *v = malloc((size_t)a->n * sizeof(int));
            Rng r = rng_of(a->seed, 77);
            make_input(v, a->n, a->input, &r);
            Log tmp = {0};
            run_sort(a->algs[i], v, a->n, &tmp, a->seed);
            S[i] = tmp.nt;
            log_free(&tmp);
            free(v);
            if (S[i] > smax) smax = S[i];
        }
        pace_init(&cur.sp.pace, smax, a->T, 3);
        for (int i = 0; i < span_n; i++) cur.fin[i] = SORT_LEAD + pace_tau(&cur.sp.pace, (double)S[i]);
        cur.place = 1;
        for (int i = 0; i < span_n; i++)
            if (cur.fin[i] < cur.fin[cur.lane] - 1e-9 || (fabs(cur.fin[i] - cur.fin[cur.lane]) < 1e-9 && i < cur.lane)) cur.place++;
    } else {
        pace_init(&cur.sp.pace, cur.sp.S, a->T, 3);
    }
    cur.sp.t_end = SORT_LEAD + pace_tau(&cur.sp.pace, (double)cur.sp.S);
    if (cur.form == F_IMAGE)
        make_picture(&cur.sp, a->pic, a->seed, portrait, (L.dx1 - L.dx0) / (L.dy1 - L.dy0));
}

/* ======================================================================
 * DRAWING
 * ====================================================================== */
static float fade_a;                   /* the act's fade in/out */

static float flash_of(const Pace *p, int ev, double ts, double tf, double steps_now, double rate) {
    if (ev < 0) return 0;
    if (ev < steps_now - rate * tf * 7 - 2) return 0;
    double age = ts - pace_tau(p, ev + 1);
    if (age < 0) age = 0;
    return (float)exp(-age / tf);
}

static void shade(const float *base, float fc, float fs, float *o) {
    mix3(base, pal.hi, 0.85f * fc, o);
    mix3(o, pal.hot, 0.9f * fs, o);
}

typedef struct { const SortPlay *s; double ts, tf, now, rate, fin; } SortView;

static void sv_flash(const SortView *v, int i, float *fc, float *fs) {
    *fc = flash_of(&v->s->pace, v->s->lastc[i], v->ts, v->tf, v->now, v->rate);
    *fs = flash_of(&v->s->pace, v->s->lasts[i], v->ts, v->tf * 1.6, v->now, v->rate);
}

/* the post-sort check sweep: 0..1 brightness at index i */
static float sweep_at(const SortView *v, int i) {
    if (v->fin < 0) return 0;
    double k = v->fin / SORT_SWEEP * v->s->n * 1.15;
    if (i > k) return 0;
    return (float)exp(-(k - i) / (v->s->n * 0.07 + 2));
}

static void draw_bars(const SortView *sv, float x0, float y0, float x1, float y1, int vert) {
    const SortPlay *s = sv->s;
    int n = s->n;
    float u = L.u, along = vert ? y1 - y0 : x1 - x0, vlen = vert ? x1 - x0 : y1 - y0, pitch = along / n;
    float hw = pitch * (pitch >= 6 ? 0.36f : pitch >= 3 ? 0.41f : 0.5f);
    if (hw < 0.45f) hw = 0.45f;
    float rr = fminf(hw * 0.45f, 2.5f * u);
    for (int i = 0; i < n; i++) {
        int val = s->v[i];
        float e = (val + 1.0f) / (s->maxv + 1.0f) * vlen, c = (vert ? y0 : x0) + (i + 0.5f) * pitch;
        float base[3], col[3], fc, fs;
        ramp((float)val / s->maxv, base);
        sv_flash(sv, i, &fc, &fs);
        if (s->pivot >= 0 && val == s->pivot && sv->fin < 0) { mix3(base, pal.hot, 0.65f, base); fs = fmaxf(fs, 0.25f); }
        if (s->heapn >= 0 && i >= s->heapn && sv->fin < 0 && s->form == F_BARS) mix3(base, pal.bg, 0.15f, base);
        shade(base, fc, fs, col);
        float sw = sweep_at(sv, i);
        mix3(col, pal.hi, 0.6f * sw, col);
        float glow = pitch > 2.5f ? 0.55f * fmaxf(fmaxf(fc, fs), sw) : 0;
        if (vert) seg2(x0, c, x0 + e, c, hw, rr, col, 1, col, 1, glow, 1);
        else seg2(c, y1, c, y1 - e, hw, rr, col, 1, col, 1, glow, 1);
    }
    /* the active range, and the selection's marked element */
    float off = 8 * u;
    if (s->lo >= 0 && s->hi >= s->lo && sv->fin < 0) {
        float a = (vert ? y0 : x0) + s->lo * pitch, b = (vert ? y0 : x0) + (s->hi + 1) * pitch;
        if (vert) seg(x0 - off, a, x0 - off, b, 1.3f * u, pal.dim, 0.9f, 0);
        else seg(a, y1 + off, b, y1 + off, 1.3f * u, pal.dim, 0.9f, 0);
    }
    if (s->mark >= 0 && sv->fin < 0) {
        float c = (vert ? y0 : x0) + (s->mark + 0.5f) * pitch;
        if (vert) disc(x0 - off, c, 3.2f * u, pal.hot, 1, 0.6f);
        else disc(c, y1 + off, 3.2f * u, pal.hot, 1, 0.6f);
    }
}

static void draw_dots(const SortView *sv, float x0, float y0, float x1, float y1, int vert) {
    const SortPlay *s = sv->s;
    int n = s->n;
    float u = L.u, along = vert ? y1 - y0 : x1 - x0, pitch = along / n;
    float r = clampf(pitch * 0.5f, 1.6f * u, 6.5f * u);
    if (vert) seg(x0, y0, x1, y1, 0.6f * u, pal.faint, 1, 0);
    else seg(x0, y1, x1, y0, 0.6f * u, pal.faint, 1, 0);
    for (int i = 0; i < n; i++) {
        int val = s->v[i];
        float f = (float)val / s->maxv, c = (vert ? y0 : x0) + (i + 0.5f) * pitch;
        float px = vert ? x0 + f * (x1 - x0) : c, py = vert ? c : y1 - f * (y1 - y0);
        float base[3], col[3], fc, fs;
        ramp(f, base);
        sv_flash(sv, i, &fc, &fs);
        if (s->pivot >= 0 && val == s->pivot && sv->fin < 0) { mix3(base, pal.hot, 0.7f, base); fs = fmaxf(fs, 0.4f); }
        shade(base, fc, fs, col);
        float sw = sweep_at(sv, i);
        mix3(col, pal.hi, 0.6f * sw, col);
        float k = fmaxf(fmaxf(fc, fs), sw);
        disc(px, py, r * (1 + 0.7f * k), col, 1, 0.7f * k);
    }
}

static long wheel_disp;
static void draw_wheel(const SortView *sv, float x0, float y0, float x1, float y1) {
    const SortPlay *s = sv->s;
    int n = s->n;
    float u = L.u, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, R = 0.48f * fminf(x1 - x0, y1 - y0);
    float rd = clampf(TAU * R / n * 0.34f, 1.6f * u, 7 * u);
    arc2(cx, cy, R, -PI, PI, 0.5f * u, pal.faint, 1, pal.faint, 1, 0);
    static float *px, *py;
    float hub = fmaxf(R * 0.1f, 62 * u);
    static int cap;
    if (cap < n) { cap = n; px = realloc(px, (size_t)n * sizeof *px); py = realloc(py, (size_t)n * sizeof *py); }
    wheel_disp = 0;
    for (int i = 0; i < n; i++) {
        int val = s->v[i], h0 = s->home0[val], h1 = s->home1[val];
        int d = i < h0 ? h0 - i : i > h1 ? i - h1 : 0;
        wheel_disp += d;
        if (d > n - d) d = n - d;
        float dn = d / (n * 0.5f), r = R * (1 - 0.8f * dn);
        float th = -PI / 2 + TAU * (i + 0.5f) / n;
        px[i] = cx + r * cosf(th); py[i] = cy + r * sinf(th);
    }
    float la = pal.light ? 0.45f : 0.32f;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float a[3], b[3];
        ramp((float)s->v[i] / s->maxv, a);
        ramp((float)s->v[j] / s->maxv, b);
        seg2(px[i], py[i], px[j], py[j], 0.55f * u, 0.55f * u, a, la, b, la, 0, 0);
    }
    /* a quiet hub for the displacement figure */
    for (int k = 0; k < 6; k++) disc(cx, cy, hub * (1.25f - k * 0.05f), pal.bg, 0.22f, 0);
    for (int i = 0; i < n; i++) {
        float base[3], col[3], fc, fs;
        ramp((float)s->v[i] / s->maxv, base);
        sv_flash(sv, i, &fc, &fs);
        if (s->pivot >= 0 && s->v[i] == s->pivot && sv->fin < 0) { mix3(base, pal.hot, 0.7f, base); fs = fmaxf(fs, 0.4f); }
        shade(base, fc, fs, col);
        float sw = sweep_at(sv, i);
        mix3(col, pal.hi, 0.6f * sw, col);
        float k = fmaxf(fmaxf(fc, fs), sw);
        disc(px[i], py[i], rd * (1 + 0.6f * k), col, 1, 0.6f * k);
    }
}

static GLuint prog_img, prog_prim, prog_text, prog_bg, vao, vbo_prim, prim_vao, vao_text, vbo_text;
#define U(p, n) glGetUniformLocation(p, n)

static void draw_image(const SortView *sv, float x0, float y0, float x1, float y1, int vert) {
    const SortPlay *s = sv->s;
    int n = s->n;
    static float *perm;
    static int cap;
    if (cap < n) { cap = n; perm = realloc(perm, (size_t)n * sizeof *perm); }
    for (int i = 0; i < n; i++) perm[i] = (float)s->v[i];
    glBindTexture(GL_TEXTURE_2D, cur.tex_perm);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, n, 1, 0, GL_RED, GL_FLOAT, perm);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(prog_img);
    glUniform2f(U(prog_img, "uRes"), (float)W, (float)H);
    glUniform4f(U(prog_img, "uRect"), x0, y0, x1, y1);
    glUniform1i(U(prog_img, "uN"), n);
    glUniform1i(U(prog_img, "uVert"), vert);
    glUniform1f(U(prog_img, "uAlpha"), fade_a);
    glUniform1i(U(prog_img, "uPic"), 0);
    glUniform1i(U(prog_img, "uPerm"), 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, cur.tex_perm);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cur.tex_pic);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    /* flashes over the stripes being touched */
    float along = vert ? y1 - y0 : x1 - x0, pitch = along / n;
    for (int i = 0; i < n; i++) {
        float fc, fs;
        sv_flash(sv, i, &fc, &fs);
        float sw = sweep_at(sv, i), k = fmaxf(fmaxf(fc * 0.6f, fs), sw * 0.7f);
        if (k < 0.02f) continue;
        float a = (vert ? y0 : x0) + i * pitch, b = a + pitch;
        const float *c = fs > fc ? pal.hot : pal.hi;
        if (vert) box(x0, a, x1, b, 0, c, 0.3f * k);
        else box(a, y0, b, y1, 0, c, 0.3f * k);
    }
}

static void draw_network(const SortView *sv, float x0, float y0, float x1, float y1, int vert) {
    const SortPlay *s = sv->s;
    int n = s->n, C = s->ncol;
    float u = L.u;
    /* time runs along t (left to right, or top to bottom); wires across */
    float t0 = (vert ? y0 : x0) + 26 * u, t1 = (vert ? y1 : x1) - 26 * u;
    float w0 = vert ? x0 : y0, w1 = vert ? x1 : y1;
    float sp = (w1 - w0) / n;
    float ww = clampf(sp * 0.13f, 1.2f * u, 3.2f * u);
#define NP(t, w, X, Y) do { float t_ = (t), w_ = w0 + ((w) + 0.5f) * sp; if (vert) { X = w_; Y = t_; } else { X = t_; Y = w_; } } while (0)
    float tc[2048];
    if (C > 2047) C = 2047;
    for (int c = 0; c < C; c++) tc[c] = t0 + (c + 0.5f) / C * (t1 - t0);
    int front_col = s->cc > 0 ? s->comp_col[s->cc - 1] : -1;
    if (s->cc >= s->ncomp) front_col = C;
    for (int w = 0; w < n; w++) {
        for (int j = 0; j <= C; j++) {
            float ta = j == 0 ? t0 - 18 * u : tc[j - 1], tb = j == C ? t1 + 18 * u : tc[j];
            int coloured = j - 1 < front_col || (j - 1 == front_col && s->wire_last[w] == front_col);
            if (j == 0) coloured = 1;
            float xa, ya, xb, yb, col[3];
            NP(ta, w, xa, ya); NP(tb, w, xb, yb);
            if (coloured) {
                ramp((float)s->snap[(size_t)j * n + w] / s->maxv, col);
                seg2(xa, ya, xb, yb, ww, ww, col, 1, col, 1, 0, 0);
            } else seg2(xa, ya, xb, yb, ww * 0.6f, ww * 0.6f, pal.faint, 1, pal.faint, 1, 0, 0);
        }
    }
    for (int c = 0; c < s->ncomp; c++) {
        int col_ = s->comp_col[c];
        if (col_ >= C) continue;
        float xa, ya, xb, yb, k = 0;
        NP(tc[col_], s->comp_a[c], xa, ya);
        NP(tc[col_], s->comp_b[c], xb, yb);
        const float *cc;
        float al;
        if (c < s->cc) {
            k = flash_of(&s->pace, s->comp_ev[c], sv->ts, sv->tf * 1.8, sv->now, sv->rate);
            cc = s->comp_sw[c] ? pal.fg : pal.dim;
            al = 0.85f;
        } else { cc = pal.faint; al = 1; }
        float col[3];
        mix3(cc, s->comp_sw[c] && c < s->cc ? pal.hot : pal.hi, k, col);
        float lw = 0.9f * u * (1 + k);
        seg(xa, ya, xb, yb, lw, col, al, 0.6f * k);
        disc(xa, ya, 2.6f * u * (1 + 0.5f * k), col, al, 0.5f * k);
        disc(xb, yb, 2.6f * u * (1 + 0.5f * k), col, al, 0.5f * k);
    }
    /* the values at the ends */
    for (int w = 0; w < n; w++) {
        float x, y, col[3];
        NP(t0 - 18 * u, w, x, y);
        ramp((float)s->snap[w] / s->maxv, col);
        disc(x, y, fminf(sp * 0.32f, 7 * u), col, 1, 0);
        if (front_col >= C) {
            NP(t1 + 18 * u, w, x, y);
            ramp((float)s->snap[(size_t)C * n + w] / s->maxv, col);
            float sw = sweep_at(sv, w);
            mix3(col, pal.hi, 0.6f * sw, col);
            disc(x, y, fminf(sp * 0.32f, 7 * u), col, 1, 0.6f * sw);
        }
    }
#undef NP
}

static void draw_heap(const SortView *sv, float x0, float y0, float x1, float y1) {
    const SortPlay *s = sv->s;
    int n = s->n, levels = 1;
    float u = L.u;
    while ((1 << levels) - 1 < n) levels++;
    float h = y1 - y0, ty1 = y0 + h * 0.6f, by0 = y0 + h * 0.68f;
    int leaves = 1 << (levels - 1);
    float lsp = (x1 - x0) / leaves, vsp = (ty1 - y0) / levels;
    float r = fminf(fminf(lsp * 0.38f, vsp * 0.3f), 13 * u);
    int heapn = s->heapn < 0 ? n : s->heapn;
#define HX(i, X, Y) do { int L_ = 31 - __builtin_clz((unsigned)(i) + 1), k_ = (i) + 1 - (1 << L_); X = x0 + (k_ + 0.5f) * (x1 - x0) / (1 << L_); Y = y0 + (L_ + 0.5f) * vsp; } while (0)
    for (int i = 1; i < n; i++) {
        float xa, ya, xb, yb;
        HX((i - 1) / 2, xa, ya); HX(i, xb, yb);
        int in = i < heapn;
        seg(xa, ya, xb, yb, 0.7f * u, in ? pal.dim : pal.faint, in ? 0.7f : 1, 0);
    }
    for (int i = 0; i < n; i++) {
        float x, y, base[3], col[3], fc, fs;
        HX(i, x, y);
        if (i >= heapn && sv->fin < 0) { disc(x, y, r * 0.45f, pal.faint, 1, 0); continue; }
        ramp((float)s->v[i] / s->maxv, base);
        sv_flash(sv, i, &fc, &fs);
        shade(base, fc, fs, col);
        float sw = sweep_at(sv, i);
        mix3(col, pal.hi, 0.5f * sw, col);
        float k = fmaxf(fc, fs);
        if (i >= heapn) disc(x, y, r * 0.45f, col, 1, 0);
        else disc(x, y, r * (1 + 0.35f * k), col, 1, 0.6f * k);
    }
#undef HX
    draw_bars(sv, x0, by0, x1, y1, 0);
    /* the boundary between heap and sorted tail */
    if (s->heapn >= 0 && s->heapn < n && sv->fin < 0) {
        float x = x0 + (x1 - x0) * s->heapn / n;
        seg(x, by0 - 6 * u, x, y1 + 6 * u, 0.8f * u, pal.dim, 0.9f, 0);
    }
}

/* ---- maze drawing ---- */
#define MXP(x) (((x) - cur.mz_ox) * cur.mz_sc)
#define MYP(y) (((y) - cur.mz_oy) * cur.mz_sc)

static void maze_node(int c, float w, const float *col, float a, float glow) {
    const Cell *k = &G.c[c];
    float x = MXP(k->x), y = MYP(k->y);
    if (G.grid == G_THETA && k->ring > 0) {
        float R = k->r * cur.mz_sc, gap = (G.pitch * cur.mz_sc * 0.5f - w) / R;
        arc2(MXP(G.cx), MYP(G.cy), R, k->a0 + gap, k->a1 - gap, w, col, a, col, a, glow);
    } else if (G.grid == G_SQUARE) seg2(x, y, x, y, w, w * 0.3f, col, a, col, a, glow, 0);
    else disc(x, y, G.grid == G_THETA ? w * 1.3f : w, col, a, glow);
}

static int link_round;                 /* overlays: round ends, and theta paths follow the rings */
static void maze_link(int a, int b, float w, const float *ca, const float *cb, float al, float glow) {
    const Cell *p = &G.c[a], *q = &G.c[b];
    if (G.grid == G_THETA && p->ring == q->ring && p->ring > 0) {
        float am = (p->a0 + p->a1) / 2, bm = (q->a0 + q->a1) / 2, d = bm - am;
        if (d > PI) d -= TAU;
        if (d < -PI) d += TAU;
        if (d > 0) arc2(MXP(G.cx), MYP(G.cy), p->r * cur.mz_sc, am, am + d, w, ca, al, cb, al, glow);
        else arc2(MXP(G.cx), MYP(G.cy), p->r * cur.mz_sc, am + d, am, w, cb, al, ca, al, glow);
    } else if (G.grid == G_THETA && p->ring != q->ring) {
        const Cell *ch = p->ring > q->ring ? p : q, *pa = p->ring > q->ring ? q : p;
        const float *cc = p->ring > q->ring ? ca : cb, *cp = p->ring > q->ring ? cb : ca;
        float am = (ch->a0 + ch->a1) / 2;
        float xe = MXP(G.cx + pa->r * cosf(am)), ye = MYP(G.cy + pa->r * sinf(am));
        seg2(MXP(ch->x), MYP(ch->y), xe, ye, w, link_round ? w : 0, cc, al, cp, al, glow, link_round ? 0 : 1);
        if (link_round && pa->ring > 0) {
            /* a path follows the parent's ring to its centre */
            float pm = (pa->a0 + pa->a1) / 2, d = pm - am;
            if (d > PI) d -= TAU;
            if (d < -PI) d += TAU;
            if (fabsf(d) > 1e-4f) arc2(MXP(G.cx), MYP(G.cy), pa->r * cur.mz_sc, d > 0 ? am : pm, d > 0 ? pm : am, w, cp, al, cp, al, glow);
        }
    } else {
        seg2(MXP(p->x), MYP(p->y), MXP(q->x), MYP(q->y), w, link_round ? w : 0, ca, al, cb, al, glow, link_round ? 0 : 1);
    }
}

static int cell_visible(int c, float m) {
    float x = MXP(G.c[c].x), y = MYP(G.c[c].y);
    return x > -m && y > -m && x < W + m && y < H + m;
}

static void neutral(float *o) { mix3(pal.bg, pal.fg, pal.light ? 0.2f : 0.17f, o); }

static float *mz_col;                  /* each cell's colour this frame */
static int mz_cap;

static void draw_maze(const Act *a, double tau) {
    MazePlay *m = &MP;
    int n = m->nc;
    if (mz_cap < n) { mz_cap = n; mz_col = realloc(mz_col, (size_t)n * 3 * sizeof *mz_col); }
    double tg = tau - MZ_LEAD, gen_end = MZ_LEAD + a->T, ts = tau - (gen_end + MZ_GAP);
    double gsteps = pace_steps(&m->pg, tg), ssteps = pace_steps(&m->ps, ts);
    mp_seek_gen(m, (long)gsteps);
    mp_seek_sol(m, ts > 0 ? (long)ssteps : 0);
    double grate = pace_rate(&m->pg, tg), srate = pace_rate(&m->ps, ts);
    double gtf = clampf((float)(5 / grate), 0.07f, 0.35f), stf = clampf((float)(5 / srate), 0.07f, 0.35f);
    float w = cur.hw;
    float relax = smooth((float)gen_end, (float)gen_end + 1.8f, (float)tau);
    int setmode = m->gen == GEN_KRUSKAL || m->gen == GEN_ELLER;
    float neu[3];
    neutral(neu);
    float maxsum = 1;
    if (m->gen == GEN_DIVISION) for (int i = 0; i < n; i++) maxsum = fmaxf(maxsum, G.c[i].x + G.c[i].y);
    float minsum = maxsum;
    if (m->gen == GEN_DIVISION) for (int i = 0; i < n; i++) minsum = fminf(minsum, G.c[i].x + G.c[i].y);
    /* cell colours */
    for (int c = 0; c < n; c++) {
        float *o = mz_col + 3 * c, base[3];
        if (setmode && (m->gen == GEN_KRUSKAL || m->st[c] == 3)) ramp((hash32((uint32_t)m->setid[c] * 2654435761u) & 1023) / 1023.0f, base);
        else if (m->gen == GEN_DIVISION) ramp((G.c[c].x + G.c[c].y - minsum) / (maxsum - minsum + 1e-3f), base);
        else ramp(m->order[c] >= 0 ? (float)m->order[c] / (m->lg.nt + 1) : 0, base);
        if (!pal.light) mix3(base, pal.bg, 0.12f, base);
        if (m->st[c] == 3 && tau < gen_end) mix3(base, pal.hi, 0.3f, base);
        float fl = tau < gen_end + 0.5 ? flash_of(&m->pg, m->lastg[c], tg, gtf, gsteps, grate) : 0;
        mix3(base, pal.hi, 0.6f * fl, base);
        mix3(base, neu, relax * 0.82f, o);
        if (ts > -0.5 && m->ss[c]) {
            float sc[3], d = m->dist[c] >= 0 ? (float)m->dist[c] / cur.maxd16 : 0;
            ramp(d, sc);
            float fs = flash_of(&m->ps, m->lasts[c], ts, stf, ssteps, srate);
            switch (m->ss[c]) {
            case 1: mix3(o, pal.hot, 0.5f, o); break;
            case 2:
                if (m->solve == SOL_DFS) { mix3(sc, neu, 0.55f, sc); }
                else if (m->solve == SOL_WALL) { mix3(sc, neu, 0.25f, sc); }
                memcpy(o, sc, sizeof sc);
                break;
            case 3:
                if (m->solve == SOL_WALL) { mix3(sc, neu, 0.65f, sc); memcpy(o, sc, sizeof sc); }
                else memcpy(o, pal.hot, sizeof sc);
                break;
            case 4: mix3(neu, pal.bg, 0.72f, o); break;
            }
            mix3(o, pal.hi, 0.7f * fs, o);
        }
        if (m->solve == SOL_DEADEND && ts > 0 && m->ss[c] != 4) {
            /* the corridors still standing light up as the dead ends fill */
            float k = smooth(0, 0.6f, (float)m->filled / (float)(n - m->npath + 1)), dc[3];
            ramp((float)m->tdist[c] / m->maxd, dc);
            mix3(o, dc, k, o);
        }
    }
    float cullm = G.pitch * cur.mz_sc * 2;
    /* ghosts of cells not yet in the maze */
    if (relax < 1)
        for (int c = 0; c < n; c++) {
            if (m->st[c] || m->order[c] >= 0 || !cell_visible(c, cullm)) continue;
            float x = MXP(G.c[c].x), y = MYP(G.c[c].y);
            disc(x, y, fmaxf(0.9f * L.u, w * 0.16f), pal.faint, 1 - relax, 0);
        }
    /* passages */
    for (int c = 0; c < n; c++) {
        if (!m->lk[c] || !cell_visible(c, cullm)) continue;
        for (int k = 0; k < G.c[c].nn; k++) {
            int b = G.c[c].nb[k];
            if (b < c && cell_visible(b, cullm)) continue;
            if (!(m->lk[c] >> k & 1)) continue;
            maze_link(c, b, w, mz_col + 3 * c, mz_col + 3 * b, 1, 0);
        }
    }
    /* cells */
    for (int c = 0; c < n; c++) {
        if (!cell_visible(c, cullm)) continue;
        if (m->st[c] == 2 && !m->lk[c]) { disc(MXP(G.c[c].x), MYP(G.c[c].y), w * 0.42f, pal.dim, 0.9f, 0); continue; }
        if (!m->lk[c] && m->st[c] != 1 && m->st[c] != 3) continue;
        maze_node(c, w, mz_col + 3 * c, 1, 0);
    }
    /* generation overlays */
    link_round = 1;
    if (tau < gen_end + 0.4) {
        float k = 1 - smooth((float)gen_end, (float)gen_end + 0.4f, (float)tau);
        for (int i = 0; i + 1 < m->nwalk; i++) maze_link(m->walk[i], m->walk[i + 1], w * 0.55f, pal.hot, pal.hot, k, 0.4f);
        if (m->nwalk) maze_node(m->walk[m->nwalk - 1], w * 1.05f, pal.hi, k, 0.8f);
        if (m->head >= 0) maze_node(m->head, w * 1.15f, pal.hi, k, 1.0f);
        if (m->gen == GEN_DIVISION && m->rect[2] > 0 && G.grid == G_SQUARE) {
            int c0 = sqc(m->rect[0], m->rect[1]), c1 = sqc(m->rect[0] + m->rect[2] - 1, m->rect[1] + m->rect[3] - 1);
            if (c0 >= 0 && c1 >= 0) {
                float p = G.pitch * cur.mz_sc * 0.5f;
                float xa = MXP(G.c[c0].x) - p, ya = MYP(G.c[c0].y) - p, xb = MXP(G.c[c1].x) + p, yb = MYP(G.c[c1].y) + p;
                float lw = 1.2f * L.u;
                seg(xa, ya, xb, ya, lw, pal.hot, 0.8f * k, 0); seg(xa, yb, xb, yb, lw, pal.hot, 0.8f * k, 0);
                seg(xa, ya, xa, yb, lw, pal.hot, 0.8f * k, 0); seg(xb, ya, xb, yb, lw, pal.hot, 0.8f * k, 0);
            }
        }
        if (m->row >= 0 && (m->gen == GEN_HUNT || m->gen == GEN_ELLER)) {
            /* the row being hunted or built: a faint band */
            float ymin = 1e9f, ymax = -1e9f, xmin = 1e9f, xmax = -1e9f;
            for (int c = 0; c < n; c++)
                if (G.c[c].row == m->row) {
                    ymin = fminf(ymin, MYP(G.c[c].y)); ymax = fmaxf(ymax, MYP(G.c[c].y));
                    xmin = fminf(xmin, MXP(G.c[c].x)); xmax = fmaxf(xmax, MXP(G.c[c].x));
                }
            if (ymax >= ymin && G.grid != G_THETA) {
                float p = G.pitch * cur.mz_sc * 0.5f;
                box(xmin - p, ymin - p, xmax + p, ymax + p, 3 * L.u, pal.hi, 0.07f * k);
            }
        }
    }
    /* solving overlays */
    if (ts > -MZ_GAP * 0.5) {
        float k = smooth(-1.2f, 0, (float)ts);
        if (m->solve == SOL_DFS)
            for (int i = 0; i + 1 < m->nstack; i++) maze_link(m->stack[i], m->stack[i + 1], w * 0.45f, pal.hot, pal.hot, 1, 0.35f);
        if (m->solve == SOL_ASTAR && m->shead >= 0 && ts < a->Ts + 0.5) {
            int guard = n;
            for (int c = m->shead; m->par[c] >= 0 && guard--; c = m->par[c]) maze_link(c, m->par[c], w * 0.32f, pal.hi, pal.hi, 0.85f, 0.3f);
        }
        if ((m->solve == SOL_WALL || m->solve == SOL_ASTAR) && m->shead >= 0 && ts < a->Ts + 0.5) maze_node(m->shead, w * 1.15f, pal.hi, 1, 1.0f);
        /* start and goal */
        float rr = w * 1.25f;
        if (cell_visible(G.start, cullm)) {
            disc(MXP(G.c[G.start].x), MYP(G.c[G.start].y), rr, pal.path, k, 0.6f * k);
        }
        if (cell_visible(G.goal, cullm)) {
            disc(MXP(G.c[G.goal].x), MYP(G.c[G.goal].y), rr, pal.bg, k, 0);
            arc2(MXP(G.c[G.goal].x), MYP(G.c[G.goal].y), rr * 0.8f, -PI, PI, rr * 0.2f, pal.path, k, pal.path, k, 0.6f * k);
        }
    }
    /* the path, traced */
    double tt = ts - a->Ts;
    if (tt > 0 && m->npath > 1) {
        float prog = (float)fmin(tt / MZ_TRACE, 1.0);
        prog = 1 - (1 - prog) * (1 - prog);
        float fl = (m->npath - 1) * prog;
        int full = (int)fl;
        float pw = w * 0.42f;
        flush_prims();
        for (int pass = 0; pass < 2; pass++) {
            float gl = pass ? 0 : (pal.light ? 0.3f : 0.75f);
            glBlendEquation(pass ? GL_FUNC_ADD : GL_MAX);
            for (int i = 0; i < full && i + 1 < m->npath; i++) {
                if (!cell_visible(m->path[i], cullm) && !cell_visible(m->path[i + 1], cullm)) continue;
                maze_link(m->path[i], m->path[i + 1], pw, pal.path, pal.path, 1, gl);
            }
            if (full + 1 < m->npath) {
                const Cell *p = &G.c[m->path[full]], *q = &G.c[m->path[full + 1]];
                float f = fl - full, x = p->x + (q->x - p->x) * f, y = p->y + (q->y - p->y) * f;
                seg(MXP(p->x), MYP(p->y), MXP(x), MYP(y), pw, pal.path, 1, gl);
                disc(MXP(x), MYP(y), pw * 1.5f, pal.hi, 1, gl * 1.4f);
            }
            flush_prims();
        }
    }
    link_round = 0;
}

/* ======================================================================
 * HUD
 * ====================================================================== */
static void counters(const char *const *labels, const long *vals, int nc, float alpha) {
    float x = L.mx, gap = 34 * L.u;
    for (int i = 0; i < nc; i++) {
        char b[32];
        commas(b, sizeof b, vals[i]);
        float lw = text_width(L.s_label, labels[i], 1.2f * L.u), nw = text_width(L.s_num, b, 0);
        text(L.s_label, x, L.lab_y, -1, labels[i], pal.dim, alpha, 1.2f * L.u);
        text(L.s_num, x, L.num_y, -1, b, pal.fg, alpha, 0);
        /* columns wide enough for the number to grow */
        float cw = fmaxf(fmaxf(lw, nw), text_width(L.s_num, "000,000", 0));
        x += cw + gap;
    }
}

static void title_card(const char *title, const char *sub, float alpha, float slide) {
    text(L.s_title, L.mx + slide, L.title_y, -1, title, pal.fg, alpha, 0);
    text(L.s_sub, L.mx + slide * 0.6f, L.sub_y, -1, sub, pal.dim, alpha, 0);
}

static void status(const char *s, float alpha) {
    text(L.s_sub, W - L.mx, L.num_y, 1, s, pal.dim, alpha, 0);
}

static void draw_clock(double wt) {
    if (spanning() && span_me != front) return;
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    char b[32], c[64];
    snprintf(b, sizeof b, "%02d:%02d", tm.tm_hour, tm.tm_min);
    text(L.s_clock, W - L.mx, L.title_y + 4 * L.u, 1, b, pal.fg, 0.9f, 0);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    if (away < 60) snprintf(c, sizeof c, "locked just now");
    else if (away < 3600) snprintf(c, sizeof c, "locked %ld min", away / 60);
    else snprintf(c, sizeof c, "locked %ld h %02ld min", away / 3600, away / 60 % 60);
    text(L.s_small, W - L.mx, L.sub_y + 2 * L.u, 1, c, pal.dim, 0.9f, 0);
}

static const char *ordinal(int k) {
    static const char *o[] = {"1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"};
    return o[(k - 1) & 7];
}

static void draw_sort(const Act *a, double tau) {
    SortPlay *s = &cur.sp;
    double ts = tau - SORT_LEAD;
    double steps = pace_steps(&s->pace, ts);
    sp_seek(s, (long)steps);
    double rate = pace_rate(&s->pace, ts);
    SortView sv = {s, ts, clampf((float)(6 / rate), 0.06f, 0.3f), steps, rate, -1};
    int done = s->done >= s->S;
    if (done) sv.fin = tau - s->t_end;
    if (sv.fin > SORT_SWEEP * 1.6) sv.fin = -1;
    float x0 = L.dx0, y0 = L.dy0, x1 = L.dx1, y1 = L.dy1, u = L.u;
    int vert = L.portrait;
    switch (cur.form) {
    case F_BARS: draw_bars(&sv, x0, y0, x1, y1, vert); break;
    case F_DOTS: draw_dots(&sv, x0, y0, x1, y1, vert); break;
    case F_WHEEL: draw_wheel(&sv, x0, y0, x1, y1); break;
    case F_IMAGE: draw_image(&sv, x0, y0, x1, y1, vert); break;
    case F_NET: draw_network(&sv, x0, y0, x1, y1, vert); break;
    case F_HEAP: draw_heap(&sv, x0, y0, x1, y1); break;
    }
    /* the title card */
    const SortDef *d = &SORTS[cur.alg];
    char sub[256], nb[32];
    commas(nb, sizeof nb, s->n);
    float ta = smooth(0.15f, 1.0f, (float)tau), slide = (1 - ta) * -10 * u;
    if (a->kind == K_RACE)
        snprintf(sub, sizeof sub, "%s \xc2\xb7 n = %s \xc2\xb7 %s \xc2\xb7 lane %d of %d", d->detail, nb, INPUT_NAME[a->input], cur.lane + 1, span_n);
    else
        snprintf(sub, sizeof sub, "%s \xc2\xb7 n = %s \xc2\xb7 %s input", d->detail, nb, INPUT_NAME[a->input]);
    title_card(d->name, sub, ta, slide);
    static const char *lab[4] = {"COMPARISONS", "SWAPS", "READS", "WRITES"};
    long vals[4] = {s->cmp, s->swp, s->rd, s->wr};
    counters(lab, vals, 4, ta);
    if (cur.form == F_WHEEL && wheel_disp >= 0) {
        char b[32];
        commas(b, sizeof b, wheel_disp);
        float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        text(L.s_label, cx, cy - 6 * u, 0, "DISPLACEMENT", pal.dim, 0.9f, 1.2f * u);
        text(L.s_mid, cx, cy + 22 * u, 0, b, pal.fg, 0.9f, 0);
    }
    /* status, bottom right */
    char st[128];
    if (d->cls == CL_JOKE) {
        if (s->hasnote[NOTE_GIVEUP]) snprintf(st, sizeof st, "gave up after %d shuffles", s->notes[NOTE_GIVEUP]);
        else if (s->hasnote[NOTE_SORTED]) snprintf(st, sizeof st, "sorted, by sheer luck, after %d shuffles", s->notes[NOTE_SORTED]);
        else snprintf(st, sizeof st, "shuffle %d", s->hasnote[NOTE_SHUF] ? s->notes[NOTE_SHUF] : 0);
    } else if (done) {
        double secs = s->t_end - SORT_LEAD;
        if (a->kind == K_RACE) snprintf(st, sizeof st, "%s of %d \xc2\xb7 sorted in %.1f s", ordinal(cur.place), span_n, secs);
        else snprintf(st, sizeof st, "sorted in %.1f s", secs);
    } else if (ts < 0) snprintf(st, sizeof st, a->kind == K_RACE ? "on your marks" : "ready");
    else if (s->hasnote[NOTE_GAP] && (!strcmp(d->key, "shell") || !strcmp(d->key, "comb"))) snprintf(st, sizeof st, "gap %d", s->notes[NOTE_GAP]);
    else if (s->hasnote[NOTE_PASS] && !strncmp(d->key, "radix-lsd", 9)) snprintf(st, sizeof st, "digit %d", s->notes[NOTE_PASS] + 1);
    else if (s->hasnote[NOTE_WIDTH]) snprintf(st, sizeof st, "runs of %d", s->notes[NOTE_WIDTH]);
    else if (s->hasnote[NOTE_RUNS] && !strcmp(d->key, "tim")) snprintf(st, sizeof st, "run %d", s->notes[NOTE_RUNS]);
    else if (s->heapn >= 0 && s->heapn < s->n) snprintf(st, sizeof st, "heap of %d", s->heapn);
    else if (s->hasnote[NOTE_LAYER] && d->net) snprintf(st, sizeof st, "layer %d", s->notes[NOTE_LAYER] + 1);
    else snprintf(st, sizeof st, "%s", a->kind == K_RACE ? "racing" : "sorting");
    status(st, ta);
}

static void maze_hud(const Act *a, double tau) {
    MazePlay *m = &MP;
    if (spanning() && span_me != front) return;
    double gen_end = MZ_LEAD + a->T, ts = tau - (gen_end + MZ_GAP);
    float u = L.u;
    char sub[256];
    char size[96];
    if (G.grid == G_THETA) snprintf(size, sizeof size, "%d rings \xc2\xb7 %d cells", G.rings, G.nc);
    else if (spanning()) snprintf(size, sizeof size, "%d %s cells across %d screens", G.nc, GRIDS[G.grid], span_n);
    else snprintf(size, sizeof size, "%d \xc3\x97 %d %s cells", G.cols, G.rows, GRIDS[G.grid]);
    /* the generator's card, crossfading into the solver's */
    float k = smooth((float)gen_end + 0.3f, (float)gen_end + 1.3f, (float)tau);
    float ta = smooth(0.15f, 1.0f, (float)tau);
    if (k < 1) {
        snprintf(sub, sizeof sub, "%s \xc2\xb7 %s", GENS[m->gen].detail, size);
        title_card(GENS[m->gen].name, sub, ta * (1 - k), -10 * u * (1 - ta) - 10 * u * k);
    }
    if (k > 0) {
        snprintf(sub, sizeof sub, "%s \xc2\xb7 %s", SOLS[m->solve].detail, size);
        title_card(SOLS[m->solve].name, sub, k, -10 * u * (1 - k));
    }
    char st[128];
    if (ts < -MZ_GAP * 0.6) {
        int nl = 0, in = 0;
        for (int c = 0; c < m->nc; c++) { nl += __builtin_popcount(m->lk[c]); in += m->order[c] >= 0 || m->st[c] == 1; }
        nl /= 2;
        if (m->gen == GEN_KRUSKAL) {
            static const char *lab[3] = {"PASSAGES", "SETS", "STEPS"};
            long v[3] = {nl, m->nc - nl, m->gdone};
            counters(lab, v, 3, ta);
        } else {
            static const char *lab[3] = {"CELLS", "PASSAGES", "STEPS"};
            long v[3] = {in, nl, m->gdone};
            counters(lab, v, 3, ta);
        }
        snprintf(st, sizeof st, tau < gen_end ? "carving" : "carved");
    } else {
        long pathlen = ts > a->Ts ? m->npath : 0;
        if (m->solve == SOL_DEADEND) {
            static const char *lab[3] = {"FILLED", "STEPS", "PATH"};
            long v[3] = {m->filled, m->steps, pathlen};
            counters(lab, v, 3, k);
        } else {
            static const char *lab[3] = {"VISITED", "STEPS", "PATH"};
            long v[3] = {m->visited, m->steps, pathlen};
            counters(lab, v, 3, k);
        }
        if (ts > a->Ts) snprintf(st, sizeof st, "solved \xc2\xb7 %d cells from start to goal", m->npath);
        else snprintf(st, sizeof st, "solving");
    }
    status(st, ta);
}

/* ======================================================================
 * GL
 * ====================================================================== */
#define GLSL(...) #__VA_ARGS__
static const char *VS_PRIM = "#version 330 core\n" GLSL(
layout(location = 0) in vec4 iA;
layout(location = 1) in vec4 iB;
layout(location = 2) in vec4 iCa;
layout(location = 3) in vec4 iCb;
uniform vec2 uRes;
out vec2 vP;
flat out vec4 vA;
flat out vec4 vB;
flat out vec4 vCa;
flat out vec4 vCb;
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    float w = iB.y, glow = iB.w;
    float m = w + 1.5 + (glow > 0.0 ? 2.5 * w + 10.0 : 0.0);
    vec2 lo, hi;
    if (abs(iB.z - 1.0) < 0.5) {
        vec2 ctr = iA.xy;
        float R = iA.z, a0 = iA.w, a1 = iB.x;
        vec2 p0 = ctr + R * vec2(cos(a0), sin(a0)), p1 = ctr + R * vec2(cos(a1), sin(a1));
        lo = min(p0, p1); hi = max(p0, p1);
        lo = min(lo, ctr + (R - w) * vec2(cos(a0), sin(a0))); hi = max(hi, ctr + (R - w) * vec2(cos(a0), sin(a0)));
        lo = min(lo, ctr + (R - w) * vec2(cos(a1), sin(a1))); hi = max(hi, ctr + (R - w) * vec2(cos(a1), sin(a1)));
        for (int k = -4; k <= 4; k++) {
            float a = float(k) * 1.5707963;
            if (a > a0 && a < a1) { vec2 p = ctr + R * vec2(cos(a), sin(a)); lo = min(lo, p); hi = max(hi, p); }
        }
        lo -= m; hi += m;
        vP = mix(lo, hi, c);
    } else {
        /* an oriented quad hugging the segment */
        vec2 ab = iA.zw - iA.xy;
        float Ln = length(ab);
        vec2 dir = Ln > 1e-4 ? ab / Ln : vec2(1.0, 0.0), nrm = vec2(-dir.y, dir.x);
        vP = 0.5 * (iA.xy + iA.zw) + dir * (c.x * 2.0 - 1.0) * (0.5 * Ln + m) + nrm * (c.y * 2.0 - 1.0) * m;
    }
    vA = iA; vB = iB; vCa = iCa; vCb = iCb;
    gl_Position = vec4(vP.x / uRes.x * 2.0 - 1.0, 1.0 - vP.y / uRes.y * 2.0, 0.0, 1.0);
}
);

static const char *FS_PRIM = "#version 330 core\n" GLSL(
in vec2 vP;
flat in vec4 vA;
flat in vec4 vB;
flat in vec4 vCa;
flat in vec4 vCb;
out vec4 o;
void main() {
    float w = vB.y, type = vB.z, glow = vB.w, d, t;
    if (abs(type - 1.0) < 0.5) {
        vec2 q = vP - vA.xy;
        float r = length(q), a0 = vA.w, a1 = vB.x, am = 0.5 * (a0 + a1), h = 0.5 * (a1 - a0);
        float a = atan(q.y, q.x) - am;
        a = mod(a + 3.14159265, 6.2831853) - 3.14159265;
        float dr = abs(r - vA.z) - w, da = (abs(a) - h) * r;
        d = max(dr, da);
        t = clamp(a / max(2.0 * h, 1e-4) + 0.5, 0.0, 1.0);
    } else {
        vec2 A = vA.xy, Bq = vA.zw, ab = Bq - A;
        float Ln = length(ab);
        vec2 dir = Ln > 1e-4 ? ab / Ln : vec2(1.0, 0.0);
        vec2 q = vP - 0.5 * (A + Bq);
        q = vec2(dot(q, dir), dot(q, vec2(-dir.y, dir.x)));
        float ext = type < 0.5 ? w : 0.0;
        vec2 hb = vec2(0.5 * Ln + ext, w);
        float r = min(vB.x, w);
        vec2 dd = abs(q) - hb + r;
        d = length(max(dd, 0.0)) + min(max(dd.x, dd.y), 0.0) - r;
        t = Ln > 1e-4 ? clamp(q.x / Ln + 0.5, 0.0, 1.0) : 0.5;
    }
    float cov = clamp(0.5 - d, 0.0, 1.0);
    vec4 col = mix(vCa, vCb, t);
    float gm = 2.5 * w + 10.0;
    float g = glow > 0.0 ? glow * exp(-max(d, 0.0) / (0.9 * w + 4.0)) * (1.0 - cov) * (1.0 - smoothstep(0.5 * gm, gm, d)) : 0.0;
    if (cov <= 0.0 && g < 0.002) discard;
    o = vec4(col.rgb * col.a * (cov + g), col.a * cov);
}
);

static const char *VS_TEXT = "#version 330 core\n" GLSL(
layout(location = 0) in vec2 aP;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aC;
uniform vec2 uRes, uAt;
out vec2 vUV;
out vec4 vC;
void main() {
    vUV = aUV / uAt;
    vC = aC;
    gl_Position = vec4(aP.x / uRes.x * 2.0 - 1.0, 1.0 - aP.y / uRes.y * 2.0, 0.0, 1.0);
}
);

static const char *FS_TEXT = "#version 330 core\n" GLSL(
in vec2 vUV;
in vec4 vC;
uniform sampler2D uTex;
out vec4 o;
void main() {
    float a = texture(uTex, vUV).r * vC.a;
    o = vec4(vC.rgb * a, a);
}
);

static const char *VS_FULL = "#version 330 core\n" GLSL(
void main() {
    vec2 p = vec2(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID >> 1) * 4 - 1));
    gl_Position = vec4(p, 0.0, 1.0);
}
);

static const char *FS_BG = "#version 330 core\n" GLSL(
uniform vec3 uBg, uBg2;
uniform vec2 uRes;
out vec4 o;
void main() {
    vec2 p = gl_FragCoord.xy;
    float r = length((p - 0.5 * uRes) / uRes.y);
    vec3 c = mix(uBg2, uBg, smoothstep(0.0, 0.95, r));
    float n = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    o = vec4(c + (n - 0.5) / 255.0, 1.0);
}
);

static const char *VS_IMG = "#version 330 core\n" GLSL(
uniform vec4 uRect;
uniform vec2 uRes;
out vec2 vP;
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    vP = mix(uRect.xy, uRect.zw, c);
    gl_Position = vec4(vP.x / uRes.x * 2.0 - 1.0, 1.0 - vP.y / uRes.y * 2.0, 0.0, 1.0);
}
);

static const char *FS_IMG = "#version 330 core\n" GLSL(
in vec2 vP;
uniform vec4 uRect;
uniform int uN, uVert;
uniform float uAlpha;
uniform sampler2D uPic, uPerm;
out vec4 o;
void main() {
    vec2 f = (vP - uRect.xy) / (uRect.zw - uRect.xy);
    float al = uVert == 1 ? f.y : f.x, ac = uVert == 1 ? f.x : f.y;
    float x = al * float(uN);
    int i = clamp(int(x), 0, uN - 1);
    float src = texelFetch(uPerm, ivec2(i, 0), 0).r;
    vec3 c = texture(uPic, vec2(ac, (src + 0.5) / float(uN))).rgb;
    float pitch = (uVert == 1 ? uRect.w - uRect.y : uRect.z - uRect.x) / float(uN);
    float fr = fract(x), e = min(fr, 1.0 - fr) * pitch;
    if (pitch >= 5.0) c *= mix(0.84, 1.0, smoothstep(0.0, 1.2, e));
    o = vec4(c * uAlpha, uAlpha);
}
);

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "algorithms: shader:\n%s\n", log);
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

static void gl_init(void) {
    prog_prim = mk_prog(VS_PRIM, FS_PRIM);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    prog_bg = mk_prog(VS_FULL, FS_BG);
    prog_img = mk_prog(VS_IMG, FS_IMG);
    glGenVertexArrays(1, &vao);
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
    glGenVertexArrays(1, &prim_vao);
    glGenBuffers(1, &vbo_prim);
    glBindVertexArray(prim_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Prim), (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Prim), (void *)(4 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Prim), (void *)(8 * sizeof(float)));
    glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Prim), (void *)(8 * sizeof(float) + 4));
    for (int i = 0; i < 4; i++) { glEnableVertexAttribArray(i); glVertexAttribDivisor(i, 1); }
    glGenTextures(1, &tex_atlas);
    glGenTextures(1, &cur.tex_pic);
    glGenTextures(1, &cur.tex_perm);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}

static void flush_prims(void) {
    if (!pn) return;
    glUseProgram(prog_prim);
    glUniform2f(U(prog_prim, "uRes"), (float)W, (float)H);
    glBindVertexArray(prim_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)pn * sizeof(Prim)), pv, GL_STREAM_DRAW);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pn);
    pn = 0;
}

static void flush_text(void) {
    if (!tvn) return;
    glUseProgram(prog_text);
    glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
    glUniform2f(U(prog_text, "uAt"), (float)AW, (float)atlas_h);
    glUniform1i(U(prog_text, "uTex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)tvn * sizeof(float)), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    tvn = 0;
}

/* el: seconds into the programme; wt: wall-clock time */
static void render(double el, double wt, GLuint target) {
    if (L.w != W || L.h != H) layout();
    int ai;
    const Act *a = act_at(el, &ai);
    if (cur.index != ai || cur.w != W || cur.h != H) build_act(a, ai);
    double tau = el - a->t0;
    fade_a = smooth(0, 0.9f, (float)tau) * (1 - smooth((float)(a->dur - FADE), (float)a->dur, (float)tau));
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_bg);
    glUniform3fv(U(prog_bg, "uBg"), 1, pal.bg);
    glUniform3fv(U(prog_bg, "uBg2"), 1, pal.bg2);
    glUniform2f(U(prog_bg, "uRes"), (float)W, (float)H);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_BLEND);
    pn = 0; tvn = 0;
    if (a->kind == K_MAZE) { draw_maze(a, tau); maze_hud(a, tau); }
    else draw_sort(a, tau);
    flush_prims();
    flush_text();
    if (fade_a < 1) {
        box(0, 0, (float)W, (float)H, 0, pal.bg, 1 - fade_a);
        flush_prims();
    }
    draw_clock(wt);
    flush_text();
}

/* ======================================================================
 * --check: every algorithm, verified
 * ====================================================================== */
static int icmp(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

static int check_all(void) {
    int fails = 0, runs = 0;
    static const int sizes[] = {1, 2, 3, 5, 8, 16, 33, 64, 100, 128, 255, 256, 512, 1000, 1024};
    printf("sorts: every algorithm x input shape x size; sorted, a permutation of the input,\n"
           "the log replayed gives the same array and counts, and known exact counts hold\n\n");
    printf("%-14s %10s %10s %10s %10s   (n = 512, random)\n", "", "compares", "swaps", "reads", "writes");
    for (int alg = 0; alg < NSORT; alg++) {
        const SortDef *d = &SORTS[alg];
        for (int in = 0; in < NINPUT + 1; in++)
            for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
                int n = sizes[si];
                if (d->pow2 && (n & (n - 1))) continue;
                if (d->cls == CL_JOKE && n > 6) continue;
                int *v = malloc((size_t)n * sizeof(int)), *orig = malloc((size_t)n * sizeof(int)), *ref = malloc((size_t)n * sizeof(int));
                Rng r = rng_of(1234 + (uint32_t)n, (uint32_t)in);
                if (in < NINPUT) make_input(v, n, in, &r);
                else for (int i = 0; i < n; i++) v[i] = i;          /* already sorted */
                memcpy(orig, v, (size_t)n * sizeof(int));
                memcpy(ref, v, (size_t)n * sizeof(int));
                qsort(ref, (size_t)n, sizeof(int), icmp);
                Log lg = {0};
                if (d->cls == CL_JOKE) bogo_limit = 100000;
                run_sort(alg, v, n, &lg, 99);
                runs++;
                char why[160] = "";
                if (memcmp(v, ref, (size_t)n * sizeof(int))) snprintf(why, sizeof why, "result not the sorted input");
                /* replay */
                SortPlay sp = {0};
                sp.n = n; sp.init = orig; sp.v = malloc((size_t)n * sizeof(int));
                sp.lastc = malloc((size_t)n * sizeof(int)); sp.lasts = malloc((size_t)n * sizeof(int));
                sp.log = lg; sp.S = lg.nt;
                sp_reset(&sp);
                sp_seek(&sp, sp.S);
                if (!why[0] && memcmp(sp.v, v, (size_t)n * sizeof(int))) snprintf(why, sizeof why, "replay differs");
                if (!why[0] && (sp.cmp != t_cmp || sp.swp != t_swp || sp.rd != t_rd || sp.wr != t_wr))
                    snprintf(why, sizeof why, "replay counts %ld/%ld/%ld/%ld vs %ld/%ld/%ld/%ld", sp.cmp, sp.swp, sp.rd, sp.wr, t_cmp, t_swp, t_rd, t_wr);
                long q = (long)n * (n - 1) / 2;
                int lg2 = 0;
                while ((1 << lg2) < n) lg2++;
                if (!why[0]) {
                    const char *k = d->key;
                    if (!strcmp(k, "selection") && t_cmp != q) snprintf(why, sizeof why, "selection: %ld compares, want n(n-1)/2 = %ld", t_cmp, q);
                    if (!strcmp(k, "pancake") && t_cmp != q) snprintf(why, sizeof why, "pancake: %ld compares, want %ld", t_cmp, q);
                    if ((!strcmp(k, "insertion") || !strcmp(k, "bubble") || !strcmp(k, "cocktail")) && in == I_REVERSED && n > 1 && (t_cmp != q || t_swp != q))
                        snprintf(why, sizeof why, "%s reversed: %ld compares %ld swaps, want %ld each", k, t_cmp, t_swp, q);
                    if ((!strcmp(k, "insertion") || !strcmp(k, "bubble") || !strcmp(k, "cocktail")) && in == NINPUT && n > 1 && (t_cmp != n - 1 || t_swp != 0))
                        snprintf(why, sizeof why, "%s sorted: %ld compares %ld swaps, want n-1 and 0", k, t_cmp, t_swp);
                    if (!strcmp(k, "bitonic") && t_cmp != (long)n * lg2 * (lg2 + 1) / 4)
                        snprintf(why, sizeof why, "bitonic: %ld comparators, want n lg n (lg n + 1) / 4 = %ld", t_cmp, (long)n * lg2 * (lg2 + 1) / 4);
                    if (!strcmp(k, "oem") && n >= 2 && t_cmp != (long)(lg2 * lg2 - lg2 + 4) * (1L << lg2) / 4 - 1)
                        snprintf(why, sizeof why, "odd-even merge: %ld comparators, want (p^2-p+4)2^(p-2)-1 = %ld", t_cmp, (long)(lg2 * lg2 - lg2 + 4) * (1L << lg2) / 4 - 1);
                    if (!strcmp(k, "oddeven")) {
                        long want = 0;
                        for (int rr = 0; rr < n; rr++) want += (n - (rr & 1)) / 2;
                        if (t_cmp != want) snprintf(why, sizeof why, "odd-even transposition: %ld comparators, want %ld", t_cmp, want);
                    }
                    if (!strcmp(k, "cycle") && in != I_FEW) {
                        long mis = 0;
                        for (int i = 0; i < n; i++) mis += orig[i] != ref[i];
                        if (t_wr != mis) snprintf(why, sizeof why, "cycle: %ld writes, want one per misplaced element = %ld", t_wr, mis);
                    }
                    if (!strcmp(k, "merge") && n > 1 && t_cmp > (long)n * lg2 - (1L << lg2) + 1)
                        snprintf(why, sizeof why, "merge: %ld compares, more than n lg n - 2^lg n + 1", t_cmp);
                    if ((!strcmp(k, "merge-bu") || !strcmp(k, "tim")) && t_cmp > (long)n * (lg2 + 1))
                        snprintf(why, sizeof why, "%s: %ld compares, more than n (lg n + 1)", k, t_cmp);
                    if (!strcmp(k, "heap") && n > 1 && t_cmp > 2L * n * lg2 + 2L * n)
                        snprintf(why, sizeof why, "heap: %ld compares, more than 2n lg n + 2n", t_cmp);
                    if ((!strncmp(k, "radix", 5) || !strcmp(k, "counting")) && t_cmp)
                        snprintf(why, sizeof why, "%s compared %ld times", k, t_cmp);
                    if (!strcmp(k, "counting") && t_wr != n) snprintf(why, sizeof why, "counting: %ld writes, want n", t_wr);
                }
                if (why[0]) { fails++; printf("FAIL %-14s %-14s n=%-5d %s\n", d->key, in < NINPUT ? INPUTS[in] : "sorted", n, why); }
                if (n == 512 && in == I_RANDOM && !why[0])
                    printf("%-14s %10ld %10ld %10ld %10ld\n", d->key, t_cmp, t_swp, t_rd, t_wr);
                free(sp.v); free(sp.lastc); free(sp.lasts);
                log_free(&lg);
                free(v); free(orig); free(ref);
            }
    }
    printf("\nmazes: every generator x grid gives a perfect maze (a spanning tree), its log\n"
           "replays to it, and every solver finds the one path\n\n");
    for (int g = 0; g < NGRID; g++)
        for (int gen = 0; gen < NGEN; gen++) {
            if (GENS[gen].square_only && g != G_SQUARE) continue;
            for (int sz = 0; sz < 2; sz++)
                for (int sol = 0; sol < NSOL; sol++) {
                    if (g == G_THETA) build_theta(&G, 0, 0, sz ? 14 : 6, 10);
                    else if (g == G_HEX) build_hex(&G, 0, 0, sz ? 41 : 7, sz ? 23 : 5, 10, NULL);
                    else build_square(&G, 0, 0, sz ? 48 : 6, sz ? 27 : 4, 10, NULL);
                    sort_nb(&G);
                    maze_record(&MP, gen, sol, 7 + (uint32_t)sz * 31 + (uint32_t)sol);
                    runs++;
                    char why[160] = "";
                    int nl = 0;
                    for (int c = 0; c < G.nc; c++) nl += __builtin_popcount(MP.flk[c]);
                    nl /= 2;
                    int *d = malloc((size_t)G.nc * sizeof(int)), *par = malloc((size_t)G.nc * sizeof(int));
                    tdist = d; tpar = par;
                    true_paths(MP.flk, G.start);
                    int reach = 0;
                    for (int c = 0; c < G.nc; c++) reach += d[c] >= 0;
                    if (nl != G.nc - 1 || reach != G.nc) snprintf(why, sizeof why, "%d passages, %d of %d reachable: not a spanning tree", nl, reach, G.nc);
                    mp_seek_gen(&MP, MP.lg.nt);
                    if (!why[0] && memcmp(MP.lk, MP.flk, (size_t)G.nc)) snprintf(why, sizeof why, "generation replay differs");
                    mp_seek_sol(&MP, MP.ls.nt);
                    if (!why[0]) {
                        int np = MP.npath, ok = 1;
                        if (sol == SOL_BFS || sol == SOL_DIJKSTRA) {
                            if (MP.visited != G.nc) ok = 0;
                            for (int c = 0; c < G.nc && ok; c++) if (sol == SOL_BFS && MP.dist[c] != d[c] * 16) ok = 0;
                        } else if (sol == SOL_DFS) {
                            if (MP.nstack != np) ok = 0;
                            for (int i = 0; i < np && ok; i++) if (MP.stack[i] != MP.path[i]) ok = 0;
                        } else if (sol == SOL_ASTAR) {
                            int k = np - 1;
                            for (int c = G.goal; ok && c >= 0; c = MP.par[c], k--) if (k < 0 || MP.path[k] != c) ok = 0;
                            if (k != -1) ok = 0;
                        } else if (sol == SOL_WALL) {
                            if (MP.shead != G.goal || MP.steps > 2L * G.nc) ok = 0;
                        } else if (sol == SOL_DEADEND) {
                            int left = 0;
                            for (int c = 0; c < G.nc; c++) left += MP.ss[c] != 4;
                            if (left != np) ok = 0;
                            for (int i = 0; i < np && ok; i++) if (MP.ss[MP.path[i]] == 4) ok = 0;
                        }
                        if (!ok) snprintf(why, sizeof why, "%s didn't find the path", SOLS[sol].key);
                    }
                    if (why[0]) { fails++; printf("FAIL %-14s %-7s %-9s %s\n", GENS[gen].key, GRIDS[g], SOLS[sol].key, why); }
                    else if (sz && sol == 0)
                        printf("%-14s %-7s %5d cells, %6d generation steps, path %4d\n", GENS[gen].key, GRIDS[g], G.nc, MP.lg.nt, MP.npath);
                    free(d); free(par);
                }
        }
    printf("\n%d runs, %d failures\n", runs, fails);
    return fails ? 1 : 0;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "algorithms");
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
        "usage: algorithms [--window | --root | --layer] [--seed N] [--fps N] [--palette NAME]\n"
        "                  [--font FAMILY] [--mono FAMILY] [--speed X]\n"
        "                  [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                  [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "                  [--kind sort|race|triptych|maze] [--sort NAME] [--form NAME] [--input NAME] [--n N]\n"
        "                  [--maze NAME] [--solve NAME] [--grid square|hex|theta]\n"
        "                  [--check] [--list]\n");
    exit(1);
}

static void list(void) {
    printf("sorts:");
    for (int i = 0; i < NSORT; i++) printf(" %s", SORTS[i].key);
    printf("\nforms:");
    for (int i = 0; i < NFORM; i++) printf(" %s", FORMS[i]);
    printf("\ninputs:");
    for (int i = 0; i < NINPUT; i++) printf(" %s", INPUTS[i]);
    printf("\nmazes:");
    for (int i = 0; i < NGEN; i++) printf(" %s", GENS[i].key);
    printf("\nsolvers:");
    for (int i = 0; i < NSOL; i++) printf(" %s", SOLS[i].key);
    printf("\ngrids:");
    for (int i = 0; i < NGRID; i++) printf(" %s", GRIDS[i]);
    printf("\npalettes:");
    for (int i = 0; i < NPAL; i++) printf(" %s", PALS[i].name);
    printf("\n");
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

/* the front output: the one under the middle of the layout */
static void pick_front(void) {
    if (span_n < 1) return;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    for (int i = 0; i < span_n; i++) out_rank[i] = lane_of(i);
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
        else if (!strcmp(a, "-check")) opt_check = 1;
        else if (!strcmp(a, "-list")) opt_list = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-mono") && i + 1 < argc) opt_mono = argv[++i];
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-kind") && i + 1 < argc) opt_kind = argv[++i];
        else if (!strcmp(a, "-sort") && i + 1 < argc) opt_sort = argv[++i];
        else if (!strcmp(a, "-form") && i + 1 < argc) opt_form = argv[++i];
        else if (!strcmp(a, "-input") && i + 1 < argc) opt_input = argv[++i];
        else if (!strcmp(a, "-n") && i + 1 < argc) opt_n = atoi(argv[++i]);
        else if (!strcmp(a, "-maze") && i + 1 < argc) opt_maze = argv[++i];
        else if (!strcmp(a, "-solve") && i + 1 < argc) opt_solve = argv[++i];
        else if (!strcmp(a, "-grid") && i + 1 < argc) opt_grid = argv[++i];
        else usage();
    }
    if (opt_list) { list(); return 0; }
    if (opt_check) return check_all();
    if (opt_sort && forced_sort() < 0) { fprintf(stderr, "algorithms: no sort '%s' (see --list)\n", opt_sort); return 1; }
    if (opt_form && find_name(opt_form, FORMS, NFORM) < 0) { fprintf(stderr, "algorithms: no form '%s'\n", opt_form); return 1; }
    if (opt_input && find_name(opt_input, INPUTS, NINPUT) < 0) { fprintf(stderr, "algorithms: no input '%s'\n", opt_input); return 1; }
    if (opt_maze && find_gen(opt_maze) < 0) { fprintf(stderr, "algorithms: no maze '%s'\n", opt_maze); return 1; }
    if (opt_solve && find_sol(opt_solve) < 0) { fprintf(stderr, "algorithms: no solver '%s'\n", opt_solve); return 1; }
    if (opt_grid && find_name(opt_grid, GRIDS, NGRID) < 0) { fprintf(stderr, "algorithms: no grid '%s'\n", opt_grid); return 1; }
    if (opt_kind && find_name(opt_kind, KINDS, NKIND) < 0) { fprintf(stderr, "algorithms: no kind '%s'\n", opt_kind); return 1; }
    if (span_me >= span_n) span_me = -1;
    pick_front();
    if (opt_fps < 1) opt_fps = 1;
    if (opt_speed <= 0) opt_speed = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;
    choose_palette();
    fonts_init();

    /* --epoch: when the programme started, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
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
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Algorithms");
        XClassHint chint = {"algorithms", "Algorithms"};
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
        if (opt_shot) {
            double wt = opt_epoch ? opt_epoch + opt_at : wall();
            render(opt_at * opt_speed, wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double sum = 0, worst = 0, gsum = 0;
        int nb = 0;
        GLuint q;
        glGenQueries(1, &q);
        for (int i = 0; i < opt_bench; i++) {
            double t = opt_at + i / 60.0;
            int ai;
            act_at(t * opt_speed, &ai);
            if (ai != cur.index) { render(t * opt_speed, wall(), fbo); glFinish(); continue; }   /* act build: not a frame */
            double f0 = now();
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t * opt_speed, wall(), fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            gsum += ns * 1e-6;
            double ms = (now() - f0) * 1000;
            sum += ms;
            nb++;
            if (ms > worst) worst = ms;
        }
        printf("algorithms: %dx%d, %d frames: avg %.3f ms, worst %.3f ms per frame (GPU %.3f ms)\n", W, H, nb, sum / (nb ? nb : 1), worst, gsum / (nb ? nb : 1));
        printf("algorithms: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
        render(el * opt_speed, wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
