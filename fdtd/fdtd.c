/*
 * fdtd — Maxwell's equations solved live, on the GPU, with the 2D
 * finite-difference time-domain (Yee) method.
 *
 * The field lives on a staggered Yee grid in float textures: Fz at the cell
 * centres, Fx and Fy half a cell up and right, and every time step is two
 * fragment passes (the transverse curl, then the z curl) ping-ponging
 * between them. One code path does both polarisations by duality:
 *
 *   TMz   Fz = Ez, (Fx, Fy) = (Hx, Hy)      line currents, PEC = Ez = 0
 *   TEz   Fz = Hz, (Fx, Fy) = -(Ex, Ey)     in-plane dipoles and strips
 *
 * The time step is a quarter of a cell per step for light (Courant number
 * S = c dt / h = 0.5, under the 2D limit 1/sqrt 2). Materials are painted
 * into coefficient textures from signed-distance shapes in units of the
 * wavelength: dielectrics (sub-cell averaged eps, graded for the Luneburg
 * lens), PEC conductors and lossy media. The domain is wrapped in a 14-cell
 * convolutional PML (Roden and Gedney's CPML, polynomial grading with
 * kappa and alpha) that sits just off-screen. Sources are soft: point and
 * line currents, continuous, Gaussian-pulsed, or modulated.
 *
 * Acts, about 48 s each, in a shuffled order:
 *   dipole    a half-wave dipole radiating: near field into far field
 *   array     a phased array of eight line sources steering its beam
 *   yagi      a Yagi-Uda: reflector, driven element, four directors
 *   horn      a sectoral horn fed from a parallel-plate guide
 *   dish      a parabolic reflector gathering plane pulses to its focus
 *   lens      a dielectric lens focusing a plane wave
 *   luneburg  a Luneburg lens turning a point on its rim into a plane wave
 *   bend      a parallel-plate waveguide round two bends
 *   crystal   a photonic crystal with an L-shaped line defect
 *   slits     a double slit's interference fringes
 *   tir       total internal reflection and its evanescent tail
 *   cavity    a resonant cavity ringing down through an iris, with its Q
 * each with a small title card and the curl equations typeset.
 *
 * Rendering: the signed field through a diverging colour map with a soft
 * glow from the time-averaged energy density; materials as translucent
 * labelled shapes; optionally (--poynting) arrows along the averaged
 * Poynting vector. Antenna acts get a far-field polar plot computed live
 * from the field (a running DFT on a box round the antenna, then the 2D
 * near-to-far-field transform of its equivalent currents) with the analytic
 * pattern dashed over it; every act has a probe trace. Palettes (--palette):
 * field, phosphor, plasma, paper. Lock info: the time and time away.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * Every instance runs the same deterministic simulation of one domain that
 * covers the whole layout (scaled so the scene fits the front output, the
 * one under the middle of the layout, and coarsened to a cell budget), and
 * draws its own slice of it. The insets and title go to a portrait output
 * if there is one, else the front.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size. --check runs the physics
 * checks (PML reflection, dipole pattern, array steering, cavity modes).
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
#define DEG (PI / 180.0f)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check, opt_poynting;
static float opt_fps = 60;
static float opt_speed = 1;            /* wave periods per second, times the act's own */
static float opt_act_secs = 48;
static int opt_cells = 420000;         /* the cell budget for the domain */
static uint32_t opt_seed;
static const char *opt_shot, *opt_palette, *opt_act;
static double opt_at = 30;             /* --shot: the frame this many seconds into the act */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the show started at */

/* ---- spanning: one instance per output ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "fdtd: %s\n", what); exit(1); }

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

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smoothf(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

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

/* ---- glyph atlas: printable ASCII plus the Greek and maths the title cards
 * and equations use, in each text style, rasterised at its exact size ---- */
static const uint32_t extra_cp[] = {
    0x2207, 0x00D7, 0x2212, 0x2202, 0x1D715, 0x1D404, 0x1D407, 0x1D46C, 0x1D46F, 0x1D707, 0x1D700, 0x1D70E,
    0x1D461, 0x1D438, 0x1D43B, 0x1D467, 0x1D465, 0x1D466, 0x1D45F, 0x1D45A, 0x1D45B, 0x1D706, 0x1D703, 0x1D711,
    0x03BB, 0x03B8, 0x03C6, 0x0394, 0x00B7, 0x2248, 0x00B0, 0x03B5, 0x03BC, 0x03C3, 0x03C0, 0x00B2, 0x2192,
    0x2014, 0x2013, 0x2265, 0x2032, 0x2009, 0x2080, 0x2081, 0x2082, 0x2083, 0x2192, 0x03BA, 0x03C9, 0x2026,
    0x2218, 0x221A, 0x00BD, 0x2022, 0x2195, 0x2194, 0x2016, 0x1D450,
};
#define NEXTRA ((int)(sizeof extra_cp / sizeof extra_cp[0]))
#define NGLYPH (95 + NEXTRA)
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGLYPH]; } Style;
#define MAXSTYLE 32
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
enum { F_UI, F_UIB, F_MATH, F_FALL, NFACE };
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    static const char *pats[NFACE] = {
        "Inter:weight=regular", "Inter:weight=semibold", "Latin Modern Math", "DejaVu Sans"};
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:fontformat=CFF,TrueType", pats[i]);
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
    styles[nstyles].g[0].adv = -1;     /* not rasterised yet */
    return nstyles++;
}

static uint32_t glyph_cp(int i) { return i < 95 ? (uint32_t)(32 + i) : extra_cp[i - 95]; }

/* a solid 6x6 block at (1,1) for rules, then the glyphs */
static int atlas_dirty;
static void build_atlas(void) {
    free(atlas);
    int cap = 4096;
    atlas = calloc((size_t)AW * cap, 1);
    for (int y = 1; y < 7; y++) memset(atlas + (size_t)y * AW + 1, 255, 6);
    int x = 9, y = 1, shelf = 6;
    for (int s = 0; s < nstyles; s++) {
        for (int c = 0; c < NGLYPH; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            uint32_t code = glyph_cp(c);
            /* the style's face, else the maths face, else DejaVu */
            FT_Face f = faces[styles[s].face];
            if (!FT_Get_Char_Index(f, code)) f = faces[F_MATH];
            if (!FT_Get_Char_Index(f, code)) f = faces[F_FALL];
            FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
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
    atlas_dirty = 1;
}

static uint32_t utf8(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t c = *s++;
    int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    if (n) c &= 0x3F >> n;
    for (; n && (*s & 0xC0) == 0x80; n--) c = c << 6 | (*s++ & 0x3F);
    *p = (const char *)s;
    return c;
}

static const Glyph *glyph(int s, uint32_t c) {
    int i = 0;
    if (c >= 32 && c < 127) i = (int)c - 32;
    else for (int k = 0; k < NEXTRA; k++) if (extra_cp[k] == c) { i = 95 + k; break; }
    return &styles[s].g[i];
}

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p;) { w += glyph(s, utf8(&p))->adv + (*p ? track : 0); }
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

/* a solid rectangle, from the atlas's solid block */
static void rect(float x0, float y0, float x1, float y1, const float *c) {
    quad_v(x0, y0, 3, 3, c); quad_v(x1, y0, 3, 3, c); quad_v(x1, y1, 3, 3, c);
    quad_v(x0, y0, 3, 3, c); quad_v(x1, y1, 3, 3, c); quad_v(x0, y1, 3, 3, c);
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x;
 * returns the width */
static float text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    pen = roundf(pen);
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

/* A line of maths in the maths face at size px, baseline y: plain runs,
 * {num|den} as a stacked fraction, and _c / ^c as a sub/superscript of one
 * character. Returns the width; draws only if col. */
static float math_line(float px, float x, float y, const char *src, const float *col) {
    int s = style(F_MATH, px), sf = style(F_MATH, px * 0.78f), ss = style(F_MATH, px * 0.68f);
    float pen = x, axis = px * 0.25f;
    char run[256];
    for (const char *p = src; *p;) {
        if (*p == '{') {
            const char *bar = strchr(p, '|'), *end = strchr(p, '}');
            if (!bar || !end || bar > end) break;
            char num[128], den[128];
            snprintf(num, sizeof num, "%.*s", (int)(bar - p - 1), p + 1);
            snprintf(den, sizeof den, "%.*s", (int)(end - bar - 1), bar + 1);
            float wn = text_width(sf, num, 0), wd = text_width(sf, den, 0), w = fmaxf(wn, wd) + px * 0.2f;
            if (col) {
                float cx = pen + w / 2, th = fmaxf(1, roundf(px * 0.05f));
                text(sf, cx, y - axis - th - px * 0.12f, 0, num, col, 0);
                text(sf, cx, y - axis + th + px * 0.12f + cap_h(sf), 0, den, col, 0);
                rect(roundf(pen + px * 0.06f), roundf(y - axis - th / 2), roundf(pen + w - px * 0.06f), roundf(y - axis - th / 2) + th, col);
            }
            pen += w + px * 0.08f;
            p = end + 1;
        } else if (*p == '_' || *p == '^') {
            int sub = *p == '_';
            p++;
            const char *q = p;
            utf8(&q);
            snprintf(run, sizeof run, "%.*s", (int)(q - p), p);
            float yy = sub ? y + px * 0.2f : y - px * 0.42f;
            pen += col ? text(ss, pen, yy, -1, run, col, 0) : text_width(ss, run, 0);
            pen += px * 0.04f;
            p = q;
        } else {
            int n = 0;
            while (p[n] && p[n] != '{' && p[n] != '_' && p[n] != '^' && n < 250) n++;
            snprintf(run, sizeof run, "%.*s", n, p);
            pen += col ? text(s, pen, y, -1, run, col, 0) : text_width(s, run, 0);
            p += n;
        }
    }
    return pen - x;
}

/* ---- palettes ---- */
typedef struct {
    const char *name;
    const char *bg, *neg0, *neg1, *pos0, *pos1, *top, *glow, *diel, *pec, *ink, *dim, *accent;
    int light;
} Palette;
static const Palette palettes[] = {
    /* blue-white-red on near black */
    {"field", "#06080c", "#1f4f9c", "#7fb6ff", "#a3271f", "#ff8a5c", "#fbf6ee", "#7e9fd8",
     "#86b8c9", "#9aa3b0", "#dfe4ec", "#7d8794", "#f1c27d", 0},
    /* a green phosphor scope */
    {"phosphor", "#020604", "#0b4a3c", "#33c79a", "#3d7011", "#b6ff5c", "#f2ffe4", "#5fd38a",
     "#5fbf8f", "#7f9c88", "#b9f5c8", "#5d8c6a", "#e6ff9a", 0},
    /* the ends of plasma on deep violet */
    {"plasma", "#0a0414", "#3a0f86", "#8c3fd6", "#c8366c", "#f88d3a", "#fbf3a0", "#c860c0",
     "#b89ae0", "#a69ab8", "#f1e0ff", "#8b7aa3", "#f9d65a", 0},
    /* an engineering print: ink on cream */
    {"paper", "#f2ede1", "#2a5aa0", "#163a72", "#b03a2a", "#6e1d12", "#1a1410", "#c9a97a",
     "#5d7f96", "#5c5650", "#2b2620", "#7a7266", "#9c2b20", 1},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

/* ---- the domain: one grid over the whole layout ----
 * Logical pixels, y down. The grid's cell (i, j) has its centre at
 * x = gx0 + (i + .5) cp, y = gy0 + (NY - j - .5) cp: j counts up the screen,
 * so the grid is right-handed like the scene. The scene's own coordinates
 * are in wavelengths with x forward, centred on the front output's centre
 * at pxl pixels per wavelength, turned a quarter (x up) when the front
 * output is portrait. */
#define S_COURANT 0.5f
#define PML 14
static int NX, NY;
static float cp, gx0, gy0, pxl, nl;     /* px per cell, grid origin, px and cells per wavelength */
static float scx, scy;                  /* the scene centre, logical px */
static int rot90;                       /* the scene turned a quarter, x up */
static float front[4], inset[4], view[4];
static int inset_here, front_here;

static void local_to_cell(float lx, float ly, float *ci, float *cj) {
    float sx = rot90 ? -ly : lx, sy = rot90 ? lx : ly;       /* screen offsets, y up, in wavelengths */
    float px = scx + sx * pxl, py = scy - sy * pxl;
    *ci = (px - gx0) / cp - 0.5f;
    *cj = NY - (py - gy0) / cp - 0.5f;
}
static void cell_to_local(float ci, float cj, float *lx, float *ly) {
    float px = gx0 + (ci + 0.5f) * cp, py = gy0 + (NY - cj - 0.5f) * cp;
    float sx = (px - scx) / pxl, sy = (scy - py) / pxl;
    *lx = rot90 ? sy : sx;
    *ly = rot90 ? -sx : sy;
}
/* scene point -> this instance's framebuffer pixels (y down) */
static void local_to_fb(float lx, float ly, float *fx, float *fy) {
    float sx = rot90 ? -ly : lx, sy = rot90 ? lx : ly;
    *fx = (scx + sx * pxl - view[0]) / view[2] * W;
    *fy = (scy - sy * pxl - view[1]) / view[3] * H;
}

/* ---- shapes, in wavelengths ---- */
enum { SH_CIRCLE, SH_BOX, SH_SEG, SH_ARC, SH_HALF, SH_LENS, SH_FRAME };
enum { M_AIR, M_DIEL, M_PEC, M_LOSS, M_LUNE };
typedef struct {
    int kind, mat;
    float p[8];
    float eps, mu, sig;
} Shape;
#define MAXSH 400
static Shape sh[MAXSH];
static int nsh;

static Shape *add_shape(int kind, int mat, float eps) {
    if (nsh == MAXSH) die("too many shapes");
    Shape *s = &sh[nsh++];
    memset(s, 0, sizeof *s);
    s->kind = kind; s->mat = mat; s->eps = eps; s->mu = 1;
    return s;
}
static Shape *add_circle(float x, float y, float r, int mat, float eps) {
    Shape *s = add_shape(SH_CIRCLE, mat, eps);
    s->p[0] = x; s->p[1] = y; s->p[2] = r;
    return s;
}
static Shape *add_box(float x, float y, float hx, float hy, int mat, float eps) {
    Shape *s = add_shape(SH_BOX, mat, eps);
    s->p[0] = x; s->p[1] = y; s->p[2] = hx; s->p[3] = hy;
    return s;
}
/* a strip from (x0,y0) to (x1,y1), th thick (at least a cell or so) */
static Shape *add_seg(float x0, float y0, float x1, float y1, float th, int mat) {
    Shape *s = add_shape(SH_SEG, mat, 1);
    s->p[0] = x0; s->p[1] = y0; s->p[2] = x1; s->p[3] = y1; s->p[4] = fmaxf(th, 1.15f / nl) / 2;
    return s;
}
static Shape *add_arc(float cx, float cy, float r, float a0, float a1, float th, int mat) {
    Shape *s = add_shape(SH_ARC, mat, 1);
    s->p[0] = cx; s->p[1] = cy; s->p[2] = r; s->p[3] = a0 * DEG; s->p[4] = a1 * DEG; s->p[5] = fmaxf(th, 1.15f / nl) / 2;
    return s;
}

static float sd_seg(float x, float y, float x0, float y0, float x1, float y1) {
    float ax = x - x0, ay = y - y0, bx = x1 - x0, by = y1 - y0;
    float t = clampf((ax * bx + ay * by) / (bx * bx + by * by + 1e-12f), 0, 1);
    return hypotf(ax - bx * t, ay - by * t);
}
static float sd_box(float x, float y, float hx, float hy) {
    float dx = fabsf(x) - hx, dy = fabsf(y) - hy;
    return hypotf(fmaxf(dx, 0), fmaxf(dy, 0)) + fminf(fmaxf(dx, dy), 0);
}

static float shape_sd(const Shape *s, float x, float y) {
    const float *p = s->p;
    switch (s->kind) {
    case SH_CIRCLE: return hypotf(x - p[0], y - p[1]) - p[2];
    case SH_BOX: return sd_box(x - p[0], y - p[1], p[2], p[3]);
    case SH_SEG: return sd_seg(x, y, p[0], p[1], p[2], p[3]) - p[4];
    case SH_ARC: {
        float dx = x - p[0], dy = y - p[1], a = atan2f(dy, dx);
        float a0 = p[3], a1 = p[4];
        while (a < a0) a += 2 * PI;
        while (a >= a0 + 2 * PI) a -= 2 * PI;
        if (a <= a1) return fabsf(hypotf(dx, dy) - p[2]) - p[5];
        float e0 = hypotf(dx - p[2] * cosf(a0), dy - p[2] * sinf(a0));
        float e1 = hypotf(dx - p[2] * cosf(a1), dy - p[2] * sinf(a1));
        return fminf(e0, e1) - p[5];
    }
    case SH_HALF: return (x - p[0]) * p[2] + (y - p[1]) * p[3];
    case SH_LENS: {   /* two discs' intersection, clipped to |y - cy| < h */
        float d1 = hypotf(x - p[0], y - p[1]) - p[4], d2 = hypotf(x - p[2], y - p[3]) - p[4];
        return fmaxf(fmaxf(d1, d2), fabsf(y - (p[1] + p[3]) / 2) - p[5]);
    }
    case SH_FRAME: {  /* a hollow box: outer half extents, wall thickness */
        float o = sd_box(x - p[0], y - p[1], p[2], p[3]);
        float i = sd_box(x - p[0], y - p[1], p[2] - p[4], p[3] - p[4]);
        return fmaxf(o, -i);
    }
    }
    return 1e9f;
}

/* a generous box round a shape, in wavelengths */
static void shape_bounds(const Shape *s, float *b) {
    const float *p = s->p;
    float m = 2.5f / nl;
    switch (s->kind) {
    case SH_CIRCLE: b[0] = p[0] - p[2]; b[1] = p[1] - p[2]; b[2] = p[0] + p[2]; b[3] = p[1] + p[2]; break;
    case SH_BOX: case SH_FRAME: b[0] = p[0] - p[2]; b[1] = p[1] - p[3]; b[2] = p[0] + p[2]; b[3] = p[1] + p[3]; break;
    case SH_SEG: b[0] = fminf(p[0], p[2]) - p[4]; b[1] = fminf(p[1], p[3]) - p[4];
                 b[2] = fmaxf(p[0], p[2]) + p[4]; b[3] = fmaxf(p[1], p[3]) + p[4]; break;
    case SH_ARC: b[0] = p[0] - p[2] - p[5]; b[1] = p[1] - p[2] - p[5]; b[2] = p[0] + p[2] + p[5]; b[3] = p[1] + p[2] + p[5]; break;
    case SH_LENS: b[0] = fminf(p[0], p[2]) - p[4]; b[2] = fmaxf(p[0], p[2]) + p[4];
                  b[1] = (p[1] + p[3]) / 2 - p[5]; b[3] = (p[1] + p[3]) / 2 + p[5]; break;
    default: b[0] = b[1] = -1e6f; b[2] = b[3] = 1e6f;
    }
    b[0] -= m; b[1] -= m; b[2] += m; b[3] += m;
}

/* ---- sources ---- */
enum { C_Z, C_X, C_Y };
enum { W_CW, W_TRAIN, W_RICKER, W_BURST };
typedef struct {
    int line, comp, profile;            /* profile (lines): 0 tapered ends, 1 half sine (a guide's mode) */
    float x0, y0, x1, y1;               /* wavelengths; a point source uses x0, y0 */
    float amp, phase;                   /* amplitude, phase lag (radians) */
    float width;                        /* a point source's footprint (~sigma), cells */
} Source;
#define MAXSRC 16
static Source src[MAXSRC];
static int nsrc;

static Source *add_point(float x, float y, int comp, float amp, float phase) {
    if (nsrc == MAXSRC) die("too many sources");
    Source *s = &src[nsrc++];
    memset(s, 0, sizeof *s);
    s->x0 = x; s->y0 = y; s->comp = comp; s->amp = amp; s->phase = phase; s->width = 0.5f;
    return s;
}
static Source *add_line(float x0, float y0, float x1, float y1, int comp, float amp, int profile) {
    Source *s = add_point(x0, y0, comp, amp, 0);
    s->line = 1; s->x1 = x1; s->y1 = y1; s->profile = profile;
    return s;
}

/* ---- labels: text placed in the scene ---- */
typedef struct { float x, y; int align; char s[64]; } Label;
#define MAXLAB 12
static Label lab_list[MAXLAB];
static int nlab;
static void add_label(float x, float y, int align, const char *s) {
    if (nlab == MAXLAB) return;
    lab_list[nlab].x = x; lab_list[nlab].y = y; lab_list[nlab].align = align;
    snprintf(lab_list[nlab].s, sizeof lab_list[nlab].s, "%s", s);
    nlab++;
}

/* ---- the acts ---- */
enum { A_DIPOLE, A_ARRAY, A_YAGI, A_HORN, A_DISH, A_LENS, A_LUNE, A_BEND, A_CRYSTAL, A_SLITS, A_TIR, A_CAVITY, NACT };
static const char *act_names[NACT] = {"dipole", "array", "yagi", "horn", "dish", "lens", "luneburg", "bend",
                                      "crystal", "slits", "tir", "cavity"};
static const char *numerals[] = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"};
static int act, act_no;                 /* the act, and its place in the show (for the numeral) */
static uint32_t act_seed;               /* varies an act from one showing to the next */
static int te;                          /* TEz (else TMz) */
static float roi[4];                    /* what must be in view: x0 y0 x1 y1, wavelengths */
static float nl_pref;                   /* cells per wavelength wanted */
static float fv;                        /* wave periods per second */
static int wave;
static float w_tau, w_rep, w_t0, w_fm;  /* pulse width, repeat, first pulse (periods), carrier / f0 */
static float probe[2];
static int has_ntff, ana_kind;          /* a far-field plot; its analytic overlay (1 dipole, 2 array) */
static float nbox[4], ntff_tau;         /* the NTFF box (wavelengths), its DFT's memory (periods) */
static char title[64], sub[160];
static float dphi;                      /* array: the phase step between elements (radians) */
static float steer;                     /* array: the angle it should steer to (radians) */
static double q_est;                    /* cavity: Q from the probe's ring-down */
static float cav_f;                     /* cavity: the mode's frequency / f0 */
static float act_gain;                  /* the picture's exposure, relative */
static int check_fixed;                 /* --check: hold the array still / cavity broadband */
static float check_dphi;

/* the stretch of each act we want in view, and how finely it wants sampling */
static void act_frame(void) {
    static const float R[NACT][4] = {
        {-4.0f, -2.6f, 4.0f, 2.6f},  /* dipole */
        {-3.6f, -3.6f, 6.4f, 3.6f},  /* array */
        {-2.6f, -2.9f, 5.8f, 2.9f},  /* yagi */
        {-4.6f, -3.0f, 6.0f, 3.0f},  /* horn */
        {-5.0f, -4.0f, 5.6f, 4.0f},  /* dish */
        {-5.6f, -3.4f, 6.0f, 3.4f},  /* lens */
        {-4.4f, -3.4f, 6.2f, 3.4f},  /* luneburg */
        {-7.4f, -2.6f, 4.4f, 2.8f},  /* bend */
        {-3.6f, -3.2f, 3.8f, 3.6f},  /* crystal */
        {-4.6f, -3.6f, 7.0f, 3.6f},  /* slits */
        {-5.0f, -3.4f, 5.0f, 1.8f},  /* tir */
        {-3.4f, -2.2f, 3.4f, 2.2f},  /* cavity */
    };
    static const float NL[NACT] = {44, 30, 40, 32, 28, 30, 32, 32, 36, 28, 36, 40};
    static const float FV[NACT] = {0.9f, 1.0f, 0.9f, 1.0f, 1.3f, 1.0f, 1.0f, 1.0f, 0.8f, 1.0f, 0.9f, 1.0f};
    static const float G[NACT] = {1, 1, 1, 0.7f, 1, 1, 1, 0.45f, 0.75f, 1, 0.9f, 0.3f};
    memcpy(roi, R[act], sizeof roi);
    act_gain = G[act];
    nl_pref = NL[act];
    fv = FV[act] * opt_speed;
}

/* the cavity: interior a x b wavelengths, centred, its TM mode (m, n) */
#define CAV_A 2.0f
#define CAV_B 1.5f
#define CAV_M 3
#define CAV_N 2

static void act_build(void) {
    nsh = nsrc = nlab = 0;
    te = 0;
    wave = W_CW; w_fm = 1; w_tau = 1; w_rep = 1e9f; w_t0 = 0;
    has_ntff = 0; ana_kind = 0; ntff_tau = 6;
    probe[0] = 2; probe[1] = 0;
    sub[0] = 0;
    q_est = 0;
    char b[96];
    switch (act) {
    case A_DIPOLE: {
        te = 1;
        snprintf(title, sizeof title, "Half-wave dipole");
        float L = 0.47f, g = 1.5f / nl;
        add_seg(0, g, 0, L / 2, 1.2f / nl, M_PEC);
        add_seg(0, -g, 0, -L / 2, 1.2f / nl, M_PEC);
        add_point(0, 0, C_Y, 1, 0);
        has_ntff = 1; ana_kind = 1;
        nbox[0] = -1.1f; nbox[1] = -1.1f; nbox[2] = 1.1f; nbox[3] = 1.1f;
        probe[0] = 2.0f; probe[1] = 0;
        add_label(0.12f, 0.30f, -1, "PEC strip, L = 0.47λ");
        add_label(0.12f, -0.02f, -1, "feed");
        snprintf(sub, sizeof sub, "near field into far field · pattern ≈ sin θ");
        break;
    }
    case A_ARRAY: {
        for (int n = 0; n < 8; n++) add_point(-2.5f, (n - 3.5f) * 0.5f, C_Z, 1, 0);
        has_ntff = 1; ana_kind = 2; ntff_tau = 3;
        nbox[0] = -3.0f; nbox[1] = -2.4f; nbox[2] = -2.0f; nbox[3] = 2.4f;
        probe[0] = 2.5f; probe[1] = 0;
        add_label(-2.75f, 2.15f, 1, "8 line sources");
        add_label(-2.75f, 1.85f, 1, "d = λ/2");
        snprintf(title, sizeof title, "Phased array");
        break;
    }
    case A_YAGI: {
        te = 1;
        snprintf(title, sizeof title, "Yagi–Uda");
        float g = 1.5f / nl, th = 1.2f / nl;
        /* tuned in 2D (--check): strips infinite in z resonate differently
         * from wires, so the reflector is long and the directors short */
        float X[6] = {-0.15f, 0, 0.2f, 0.4f, 0.6f, 0.8f};
        float L[6] = {0.62f, 0.47f, 0.26f, 0.25f, 0.24f, 0.23f};
        const char *ov = getenv("FDTD_YAGI");
        if (ov) {
            float rl, rs, dl, ds;
            if (sscanf(ov, "%f %f %f %f", &rl, &rs, &dl, &ds) == 4) {
                L[0] = rl; X[0] = -rs;
                for (int i = 2; i < 6; i++) { L[i] = dl - 0.01f * (i - 2); X[i] = ds * (i - 1); }
            }
        }
        for (int i = 0; i < 6; i++) {
            if (i == 1) {
                add_seg(X[i], g, X[i], L[i] / 2, th, M_PEC);
                add_seg(X[i], -g, X[i], -L[i] / 2, th, M_PEC);
            } else add_seg(X[i], -L[i] / 2, X[i], L[i] / 2, th, M_PEC);
        }
        add_point(0, 0, C_Y, 1, 0);
        has_ntff = 1;
        nbox[0] = -0.6f; nbox[1] = -0.7f; nbox[2] = 1.3f; nbox[3] = 0.7f;
        probe[0] = 3.2f; probe[1] = 0;
        add_label(-0.15f, 0.45f, 0, "reflector");
        add_label(0.0f, -0.36f, 0, "driven");
        add_label(0.5f, 0.26f, 0, "directors");
        snprintf(sub, sizeof sub, "parasitic strips: one reflector, four directors");
        break;
    }
    case A_HORN: {
        snprintf(title, sizeof title, "Sectoral horn");
        float a = 0.4f, w = 0.1f, x0 = -4.2f, x1 = -2.2f, x2 = 1.4f, A = 1.9f;
        add_seg(x0, a + w / 2, x1, a + w / 2, w, M_PEC);
        add_seg(x0, -a - w / 2, x1, -a - w / 2, w, M_PEC);
        add_seg(x0 - w / 2, -a - w, x0 - w / 2, a + w, w, M_PEC);
        add_seg(x1, a + w / 2, x2, A + w / 2, w, M_PEC);
        add_seg(x1, -a - w / 2, x2, -A - w / 2, w, M_PEC);
        add_line(x0 + 0.32f, -a, x0 + 0.32f, a, C_Z, 1, 1);
        has_ntff = 1;
        nbox[0] = -4.5f; nbox[1] = -2.4f; nbox[2] = 1.8f; nbox[3] = 2.4f;
        probe[0] = 1.3f; probe[1] = 0;
        add_label(-3.2f, 0.62f, 0, "TE₁ guide");
        add_label(-0.3f, 1.45f, 0, "PEC flare");
        snprintf(sub, sizeof sub, "a parallel-plate guide flared to a 3.8λ aperture");
        break;
    }
    case A_DISH: {
        snprintf(title, sizeof title, "Parabolic reflector");
        float f = 2.0f, xv = -4.5f, ym = 3.6f, w = 0.12f;
        int n = 48;
        for (int i = 0; i < n; i++) {
            float ya = -ym + 2 * ym * i / n, yb = -ym + 2 * ym * (i + 1) / n;
            add_seg(xv + ya * ya / (4 * f), ya, xv + yb * yb / (4 * f), yb, w, M_PEC);
        }
        add_line(4.6f, -3.9f, 4.6f, 3.9f, C_Z, 1, 0);
        wave = W_TRAIN; w_tau = 1.2f; w_t0 = 3; w_rep = 15.0f;
        probe[0] = xv + f; probe[1] = 0;
        add_label(xv + f, -0.35f, 0, "focus");
        add_label(xv + 0.55f, 3.25f, -1, "PEC paraboloid, f = 2λ");
        snprintf(sub, sizeof sub, "plane pulses gathered to the focus");
        break;
    }
    case A_LENS: {
        snprintf(title, sizeof title, "Dielectric lens");
        float R = 4.0f, h = 2.6f, xc = -1.5f, sag = R - sqrtf(R * R - h * h);
        Shape *s = add_shape(SH_LENS, M_DIEL, 2.25f);
        s->p[0] = xc + R - sag; s->p[1] = 0; s->p[2] = xc - R + sag; s->p[3] = 0; s->p[4] = R; s->p[5] = h;
        add_line(-5.2f, -3.6f, -5.2f, 3.6f, C_Z, 1, 0);
        probe[0] = 2.3f; probe[1] = 0;
        add_label(xc, h + 0.2f, 0, "n = 1.5");
        snprintf(sub, sizeof sub, "a plane wave brought to a focus");
        break;
    }
    case A_LUNE: {
        snprintf(title, sizeof title, "Luneburg lens");
        add_circle(0, 0, 2.6f, M_LUNE, 2);
        add_point(-2.68f, 0, C_Z, 1, 0)->width = 1.3f;   /* it moves: wide, or it sheds grid noise */
        has_ntff = 1; ntff_tau = 4;
        nbox[0] = -3.1f; nbox[1] = -3.1f; nbox[2] = 3.1f; nbox[3] = 3.1f;
        probe[0] = 4.0f; probe[1] = 0;
        add_label(0, 2.85f, 0, "ε(r) = 2 − (r/R)²");
        snprintf(sub, sizeof sub, "a point on the rim leaves as a plane wave");
        break;
    }
    case A_BEND: {
        snprintf(title, sizeof title, "Waveguide bend");
        float a = 0.75f, w = 0.1f, R = 1.4f, h = a / 2 + w / 2;
        float y0 = -1.6f, xs = -7.0f, xb = -1.4f;
        add_seg(xs, y0 + h, xb, y0 + h, w, M_PEC);
        add_seg(xs, y0 - h, xb, y0 - h, w, M_PEC);
        add_seg(xs - w / 2, y0 - h - w / 2, xs - w / 2, y0 + h + w / 2, w, M_PEC);
        add_arc(xb, y0 + R, R - h, -90, 0, w, M_PEC);
        add_arc(xb, y0 + R, R + h, -90, 0, w, M_PEC);
        float yv0 = y0 + R, yv1 = -y0 - R;      /* the short upright between the bends */
        add_seg(xb + R - h, yv0, xb + R - h, yv1, w, M_PEC);
        add_seg(xb + R + h, yv0, xb + R + h, yv1, w, M_PEC);
        add_arc(xb + 2 * R, yv1, R + h, 90, 180, w, M_PEC);
        add_arc(xb + 2 * R, yv1, R - h, 90, 180, w, M_PEC);
        add_seg(xb + 2 * R, -y0 + h, 3.6f, -y0 + h, w, M_PEC);
        add_seg(xb + 2 * R, -y0 - h, 3.6f, -y0 - h, w, M_PEC);
        add_line(xs + 0.3f, y0 - a / 2, xs + 0.3f, y0 + a / 2, C_Z, 1, 1);
        probe[0] = 2.6f; probe[1] = -y0;
        add_label(-4.2f, y0 - 0.65f, 0, "parallel-plate guide, a = 0.75λ");
        add_label(xb + R + 0.6f, 0, -1, "R = 1.4λ");
        snprintf(sub, sizeof sub, "the TE₁ mode follows two quarter bends");
        break;
    }
    case A_CRYSTAL: {
        snprintf(title, sizeof title, "Photonic crystal");
        float a = 0.38f;
        for (int i = -8; i <= 8; i++)
            for (int j = -7; j <= 8; j++) {
                if (j == 0 && i <= 0) continue;   /* the line defect: in from the left... */
                if (i == 0 && j >= 0) continue;   /* ...and round the corner, up */
                add_circle(i * a, j * a, 0.2f * a, M_DIEL, 8.9f);
            }
        add_point(-8.6f * a, 0, C_Z, 1, 0);
        probe[0] = 0; probe[1] = 8.6f * a;
        add_label(-4.5f * a, -8.2f * a, 0, "rods ε = 8.9, r = 0.2a, a/λ = 0.38");
        snprintf(sub, sizeof sub, "in the band gap, light keeps to the defect and turns the corner");
        break;
    }
    case A_SLITS: {
        snprintf(title, sizeof title, "Double slit");
        float xs = -1.2f, t = 0.2f, d = 2.0f, sw = 0.45f;
        add_box(xs, 0, t / 2, 100, M_PEC, 1);
        Shape *ab = add_box(xs - t / 2 - 0.25f, 0, 0.25f, 100, M_LOSS, 1);
        ab->sig = 0.08f;
        add_box(xs - 0.25f, d / 2, 0.4f, sw / 2, M_AIR, 1);
        add_box(xs - 0.25f, -d / 2, 0.4f, sw / 2, M_AIR, 1);
        add_line(-4.3f, -100, -4.3f, 100, C_Z, 1, 0);
        probe[0] = 5.0f; probe[1] = 0;
        add_label(xs - 0.5f, -2.6f, 1, "absorber σ");
        add_label(xs + 0.25f, -2.6f, -1, "PEC");
        snprintf(b, sizeof b, "d = %.0fλ · bright fringes at sin θ = mλ/d", d);
        snprintf(sub, sizeof sub, "%s", b);
        break;
    }
    case A_TIR: {
        snprintf(title, sizeof title, "Total internal reflection");
        Shape *s = add_shape(SH_HALF, M_DIEL, 4);
        s->p[0] = 0; s->p[1] = 0; s->p[2] = 0; s->p[3] = 1;
        float th = 40 * DEG, dx = sinf(th), dy = cosf(th);   /* travelling up and right, 40 deg off the normal */
        float cx = -3.0f * dx, cy = -3.0f * dy, hw = 2.0f;
        add_line(cx - dy * hw, cy + dx * hw, cx + dy * hw, cy - dx * hw, C_Z, 1, 0);
        probe[0] = 0; probe[1] = 0.12f;
        add_label(-4.6f, -0.35f, -1, "n = 2");
        add_label(-4.6f, 0.2f, -1, "n = 1");
        add_label(0.5f, 0.42f, -1, "evanescent");
        snprintf(sub, sizeof sub, "40° from the normal, past the critical 30°");
        break;
    }
    case A_CAVITY: {
        snprintf(title, sizeof title, "Resonant cavity");
        float w = 0.12f;
        Shape *s = add_shape(SH_FRAME, M_PEC, 1);
        s->p[0] = 0; s->p[1] = 0; s->p[2] = CAV_A / 2 + w; s->p[3] = CAV_B / 2 + w; s->p[4] = w;
        add_box(CAV_A / 2 + w / 2, 0.375f, w, 0.09f, M_AIR, 1);   /* the iris */
        /* drive at one antinode of TM32, listen at another */
        float ax = CAV_A / (2 * CAV_M), by = CAV_B / (2 * CAV_N);
        add_point(-CAV_A / 2 + ax, -CAV_B / 2 + by, C_Z, 1, 0);
        probe[0] = -CAV_A / 2 + 3 * ax; probe[1] = -CAV_B / 2 + 3 * by;
        cav_f = 0.5f * sqrtf((CAV_M / CAV_A) * (CAV_M / CAV_A) + (CAV_N / CAV_B) * (CAV_N / CAV_B));
        wave = W_BURST; w_fm = cav_f; w_tau = 2.5f; w_t0 = 5; w_rep = 24 * fv;
        if (check_fixed) { wave = W_RICKER; w_tau = 0.35f; w_t0 = 1.5f; }
        add_label(CAV_A / 2 + w + 0.12f, 0.375f, -1, "iris");
        snprintf(b, sizeof b, "%.0fλ × %.1fλ PEC box", CAV_A, CAV_B);
        add_label(0, -CAV_B / 2 - w - 0.3f, 0, b);
        break;
    }
    }
}

static float act_ramp(float tp) { return tp <= 0 ? 0 : tp >= 1.5f ? 1 : 0.5f - 0.5f * cosf(PI * tp / 1.5f); }

/* the array's steering programme: a few angles from the act's seed */
static float array_angle(float ts) {
    static const float set[] = {0, 30, -20, 45, -40, 15, -30, 50};
    uint32_t h = hash32(act_seed * 3u + 7);
    float hold = 9.5f;
    int k = (int)(ts / hold);
    float u = ts - k * hold;
    float a0 = set[(h + (uint32_t)k * 5u) % 8], a1 = set[(h + (uint32_t)(k + 1) * 5u) % 8];
    if (k == 0) a0 = 0;
    return (a0 + (a1 - a0) * smoothf(hold - 1.8f, hold, u)) * DEG;
}

/* things that move with time tp (periods since the act began) */
static void act_dynamic(float tp) {
    float ts = tp / fv;
    if (act == A_ARRAY) {
        steer = array_angle(ts);
        dphi = check_fixed ? check_dphi : PI * sinf(steer);
        if (check_fixed) steer = asinf(dphi / PI);
        for (int n = 0; n < 8; n++) src[n].phase = n * dphi;
    } else if (act == A_LUNE) {
        static const float set[] = {0, 24, -20, 12, -28};
        float hold = 11.0f;
        int k = (int)(ts / hold);
        float u = ts - k * hold;
        uint32_t h = hash32(act_seed * 5u + 1);
        float a0 = k ? set[(h + (uint32_t)k) % 5] : 0, a1 = set[(h + (uint32_t)k + 1) % 5];
        /* hop between positions: fade out, move while silent, fade in (a
         * source sliding round the rim would go at a fair fraction of c) */
        float b = (u < hold - 1.5f ? a0 : a1) * DEG;
        float dip = fabsf(u - (hold - 1.5f)) / 1.5f;
        src[0].amp = dip >= 1 ? 1 : 0.5f - 0.5f * cosf(PI * dip);
        src[0].x0 = -2.68f * cosf(b);
        src[0].y0 = -2.68f * sinf(b);
    }
}

/* a source's strength at time tp */
static float src_value(const Source *s, float tp) {
    float c = sinf(2 * PI * w_fm * tp - s->phase);
    switch (wave) {
    case W_CW: return s->amp * act_ramp(tp) * c;
    case W_TRAIN: case W_BURST: {
        float k = floorf((tp - w_t0) / w_rep + 0.5f);
        if (k < 0) k = 0;
        float d = (tp - w_t0 - k * w_rep) / w_tau;
        return s->amp * expf(-d * d) * c;
    }
    case W_RICKER: {
        float d = (tp - w_t0) / w_tau;
        return s->amp * -d * expf(-d * d) * 2.33f;
    }
    }
    return 0;
}

/* ---- shaders ---- */
#define GLSL(...) #__VA_ARGS__
#define VERSION "#version 330 core\n"

/* a full-screen triangle */
static const char *VS = VERSION GLSL(
    void main() {
        vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* a rectangle in framebuffer pixels (y down), with a -1..1 coordinate across it */
static const char *VS_RECT = VERSION GLSL(
    uniform vec4 uRect;
    uniform vec2 uRes;
    out vec2 vP;
    void main() {
        int i = gl_VertexID;
        vec2 c = vec2(i == 1 || i == 2 || i == 4 ? 1.0 : 0.0, i == 2 || i == 4 || i == 5 ? 1.0 : 0.0);
        vec2 px = uRect.xy + c * uRect.zw;
        vP = vec2(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0);
        gl_Position = vec4(px.x / uRes.x * 2.0 - 1.0, 1.0 - px.y / uRes.y * 2.0, 0.0, 1.0);
    });

/* the CPML grading and the soft sources, shared by both curl passes */
#define SIM_COMMON GLSL( \
    uniform ivec2 uN; \
    uniform vec4 uPml; \
    uniform int uNps; \
    uniform vec4 uPs[16]; \
    uniform vec4 uLs0[16]; \
    uniform vec4 uLs1[16]; \
    vec3 pml(float p, float n) { \
        float L = uPml.x; \
        float d = max(L - p, p - (n - 1.0 - L)) / L; \
        if (d <= 0.0) return vec3(0.0, 0.0, 1.0); \
        d = min(d, 1.0); \
        float d3 = d * d * d; \
        float s = uPml.y * d3, k = 1.0 + (uPml.z - 1.0) * d3, a = uPml.w * (1.0 - d); \
        float b = exp(-(s / k + a)); \
        return vec3(b, s / (s * k + k * k * a) * (b - 1.0), k); \
    } \
    float src(vec2 q, float comp) { \
        float v = 0.0; \
        float L = uPml.x; \
        if (q.x < L || q.y < L || q.x > float(uN.x) - 1.0 - L || q.y > float(uN.y) - 1.0 - L) return 0.0; \
        for (int i = 0; i < uNps; i++) { \
            if (uLs1[i].x != comp) continue; \
            if (uLs1[i].z < 0.0) { \
                vec2 d = q - uPs[i].xy; \
                float r2 = dot(d, d); \
                float R2 = 9.0 * uLs1[i].w * uLs1[i].w, k = 1.0 - r2 / R2; \
                if (k > 0.0) v += uLs1[i].y * k * k * k * 1.2732395 / R2; \
            } else { \
                vec2 a = uLs0[i].xy, ab = uLs0[i].zw - a; \
                float t = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0); \
                vec2 d = q - (a + t * ab); \
                float r2 = dot(d, d); \
                if (r2 < 6.0) { \
                    float pr = uLs1[i].z < 0.5 ? smoothstep(0.0, 0.18, t) * smoothstep(1.0, 0.82, t) : sin(3.14159265 * t); \
                    v += uLs1[i].y * pr * exp(-2.0 * r2) * 0.79788; \
                } \
            } \
        } \
        return v; \
    })

/* the transverse fields from the curl of Fz: (Fx, Fy, psi_x, psi_y) */
static const char *FS_STEP_T = VERSION SIM_COMMON GLSL(
    uniform sampler2D uFz, uFt, uMatT;
    out vec4 o;
    void main() {
        ivec2 c = ivec2(gl_FragCoord.xy);
        vec4 t = texelFetch(uFt, c, 0);
        float ez = texelFetch(uFz, c, 0).r;
        float er = c.x + 1 < uN.x ? texelFetch(uFz, c + ivec2(1, 0), 0).r : 0.0;
        float eu = c.y + 1 < uN.y ? texelFetch(uFz, c + ivec2(0, 1), 0).r : 0.0;
        vec4 m = texelFetch(uMatT, c, 0);
        vec3 py = pml(float(c.y) + 0.5, float(uN.y));
        vec3 px = pml(float(c.x) + 0.5, float(uN.x));
        float dy = eu - ez, dx = er - ez;
        t.b = py.x * t.b + py.y * dy;
        t.a = px.x * t.a + px.y * dx;
        vec2 q = vec2(c);
        t.r = m.x * t.r - m.y * (dy / py.z + t.b) + src(q + vec2(0.0, 0.5), 1.0);
        t.g = m.z * t.g + m.w * (dx / px.z + t.a) + src(q + vec2(0.5, 0.0), 2.0);
        o = t;
    });

/* Fz from the curl of the transverse fields: (Fz, psi_x, psi_y, 0) */
static const char *FS_STEP_Z = VERSION SIM_COMMON GLSL(
    uniform sampler2D uFz, uFt, uMatZ;
    out vec4 o;
    void main() {
        ivec2 c = ivec2(gl_FragCoord.xy);
        vec4 z = texelFetch(uFz, c, 0);
        vec4 t = texelFetch(uFt, c, 0);
        float hl = c.x > 0 ? texelFetch(uFt, c - ivec2(1, 0), 0).g : 0.0;
        float hd = c.y > 0 ? texelFetch(uFt, c - ivec2(0, 1), 0).r : 0.0;
        vec2 m = texelFetch(uMatZ, c, 0).xy;
        vec3 px = pml(float(c.x), float(uN.x)), py = pml(float(c.y), float(uN.y));
        float dhy = t.g - hl, dhx = t.r - hd;
        z.g = px.x * z.g + px.y * dhy;
        z.b = py.x * z.b + py.y * dhx;
        z.r = m.x * z.r + m.y * (dhy / px.z + z.g - dhx / py.z - z.b) + src(vec2(c), 0.0);
        if (c.x == 0 || c.y == 0 || c.x == uN.x - 1 || c.y == uN.y - 1) z.r = 0.0;
        o = z;
    });

/* the probe: Fz there, into one texel of the ring */
static const char *FS_PROBE = VERSION GLSL(
    uniform sampler2D uFz;
    uniform vec2 uAt;
    out vec4 o;
    void main() { o = vec4(texture(uFz, (uAt + 0.5) / vec2(textureSize(uFz, 0))).r); });

/* the running DFT of the fields on the NTFF box: (Ez re, im, Hx re, im), (Hy re, im) */
static const char *FS_NTFF = VERSION GLSL(
    uniform sampler2D uPos, uA, uB, uFz, uFt;
    uniform vec2 uRotZ, uRotT;
    uniform float uDecay;
    layout(location = 0) out vec4 oA;
    layout(location = 1) out vec4 oB;
    void main() {
        ivec2 k = ivec2(gl_FragCoord.xy);
        vec2 p = texelFetch(uPos, k, 0).xy;
        vec2 inv = 1.0 / vec2(textureSize(uFz, 0));
        float ez = texture(uFz, (p + 0.5) * inv).r;
        float hx = texture(uFt, (p + vec2(0.5, 0.0)) * inv).r;
        float hy = texture(uFt, (p + vec2(0.0, 0.5)) * inv).g;
        vec4 a = texelFetch(uA, k, 0) * uDecay, b = texelFetch(uB, k, 0) * uDecay;
        a.xy += ez * vec2(uRotZ.x, -uRotZ.y);
        a.zw += hx * vec2(uRotT.x, -uRotT.y);
        b.xy += hy * vec2(uRotT.x, -uRotT.y);
        oA = a;
        oB = vec4(b.xy, 0.0, 0.0);
    });

/* the far field, one texel per degree (screen angle): the 2D transform of
 * the box's equivalent currents, J = n x H and M = -n x E */
static const char *FS_PATTERN = VERSION GLSL(
    uniform sampler2D uPos, uA, uB;
    uniform int uNp;
    uniform float uK;
    uniform vec2 uC;
    uniform float uSgn;
    out vec4 o;
    void main() {
        float ph = (floor(gl_FragCoord.x) + 0.5) * (6.28318531 / 360.0);
        vec2 r = vec2(cos(ph), sin(ph));
        vec2 acc = vec2(0.0);
        for (int k = int(gl_FragCoord.y); k < uNp; k += 16) {
            vec4 pn = texelFetch(uPos, ivec2(k, 0), 0);
            vec4 a = texelFetch(uA, ivec2(k, 0), 0);
            vec2 hy = texelFetch(uB, ivec2(k, 0), 0).xy;
            vec2 jz = pn.z * hy - pn.w * a.zw;
            vec2 f = a.xy * dot(pn.zw, r) + uSgn * jz;
            float arg = uK * dot(r, pn.xy - uC);
            vec2 e = vec2(cos(arg), sin(arg));
            acc += vec2(f.x * e.x - f.y * e.y, f.x * e.y + f.y * e.x);
        }
        o = vec4(acc, 0.0, 0.0);
    });

/* the partial sums added up: the far field's magnitude */
static const char *FS_PATSUM = VERSION GLSL(
    uniform sampler2D uT;
    out vec4 o;
    void main() {
        vec2 s = vec2(0.0);
        for (int j = 0; j < 16; j++) s += texelFetch(uT, ivec2(int(gl_FragCoord.x), j), 0).xy;
        o = vec4(length(s));
    });

/* the largest of a row of texels, into one */
static const char *FS_MAX = VERSION GLSL(
    uniform sampler2D uT;
    uniform int uN0, uN1, uW, uStride;
    out vec4 o;
    void main() {
        float m = 1e-12;
        for (int i = uN0 + int(gl_FragCoord.x); i < uN1; i += uStride) m = max(m, abs(texelFetch(uT, ivec2((i % uW + uW) % uW, 0), 0).r));
        o = vec4(m);
    });

/* once a frame: the time-averaged energy density and Poynting vector, and
 * Fz between the last two steps: (<u>, <Sx>, <Sy>, Fz) */
static const char *FS_PREP = VERSION GLSL(
    uniform sampler2D uFz, uFzPrev, uFt, uAcc, uMatV;
    uniform float uFrac, uK;
    out vec4 o;
    void main() {
        ivec2 c = ivec2(gl_FragCoord.xy);
        float z = mix(texelFetch(uFzPrev, c, 0).r, texelFetch(uFz, c, 0).r, uFrac);
        vec4 t = texelFetch(uFt, c, 0);
        float hx = 0.5 * (t.r + texelFetch(uFt, max(c - ivec2(0, 1), ivec2(0)), 0).r);
        float hy = 0.5 * (t.g + texelFetch(uFt, max(c - ivec2(1, 0), ivec2(0)), 0).g);
        float u = z * z + hx * hx + hy * hy;
        vec4 a = texelFetch(uAcc, c, 0);
        a.rgb = mix(a.rgb, vec3(u, -z * hy, z * hx), uK);
        o = vec4(a.rgb, z);
    });

/* the exposure: a slow average of the mean field amplitude, sqrt <u>,
 * sampled over the domain inside the PML (robust to the sources' peaks) */
static const char *FS_EXPO_S = VERSION GLSL(
    uniform sampler2D uAcc;
    uniform vec2 uLo, uHi;
    out vec4 o;
    void main() {
        vec2 uv = mix(uLo, uHi, gl_FragCoord.xy / vec2(32.0, 18.0));
        o = vec4(sqrt(max(texture(uAcc, uv).r, 0.0)));
    });
static const char *FS_EXPO = VERSION GLSL(
    uniform sampler2D uS, uPrev;
    uniform vec2 uK2;
    out vec4 o;
    void main() {
        float m = 0.0;
        for (int j = 0; j < 18; j++)
            for (int i = 0; i < 32; i++) m += texelFetch(uS, ivec2(i, j), 0).r;
        m /= 576.0;
        float p = texelFetch(uPrev, ivec2(0), 0).r;
        o = vec4(mix(p, m, m > p ? uK2.x : uK2.y));
    });

/* the glow: the averaged energy, blurred (at half the grid) */
static const char *FS_BLUR = VERSION GLSL(
    uniform sampler2D uT;
    uniform vec2 uDir, uOut;
    uniform float uLod;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / uOut;
        float w[5] = float[](0.2270, 0.1945, 0.1216, 0.0540, 0.0162);
        float s = textureLod(uT, uv, uLod).r * w[0];
        for (int i = 1; i < 5; i++) {
            vec2 d = uDir * float(i) * 1.5 / uOut;
            s += (textureLod(uT, uv + d, uLod).r + textureLod(uT, uv - d, uLod).r) * w[i];
        }
        o = vec4(s);
    });

/* the picture: the field, its glow, the materials and the Poynting arrows */
static const char *FS_COMP = VERSION GLSL(
    uniform sampler2D uAcc, uGlow, uMat, uExp;
    uniform vec4 uMap;
    uniform vec3 uBg, uNeg0, uNeg1, uPos0, uPos1, uTop, uGlowC, uDiel, uPec;
    uniform float uLight, uGain, uFade, uPoynt, uArrow, uTime, uCardR, uH;
    uniform vec4 uCard[3];
    uniform int uNCard;
    out vec4 o;
    vec4 cubic(float v) {
        vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
        vec4 s = n * n * n;
        float x = s.x, y = s.y - 4.0 * s.x, z = s.z - 4.0 * s.y + 6.0 * s.x;
        return vec4(x, y, z, 6.0 - x - y - z) * (1.0 / 6.0);
    }
    vec4 bicubic(sampler2D t, vec2 uv) {
        vec2 ts = vec2(textureSize(t, 0));
        vec2 tc = uv * ts - 0.5;
        vec2 f = fract(tc);
        tc -= f;
        vec4 xc = cubic(f.x), yc = cubic(f.y);
        vec4 c = tc.xxyy + vec2(-0.5, 1.5).xyxy;
        vec4 s = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
        vec4 off = (c + vec4(xc.yw, yc.yw) / s) / ts.xxyy;
        vec4 s0 = textureLod(t, off.xz, 0.0), s1 = textureLod(t, off.yz, 0.0);
        vec4 s2 = textureLod(t, off.xw, 0.0), s3 = textureLod(t, off.yw, 0.0);
        float sx = s.x / (s.x + s.y), sy = s.z / (s.z + s.w);
        return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
    }
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 uv = uMap.xy + gl_FragCoord.xy * uMap.zw;
        vec4 a = bicubic(uAcc, uv);
        float e = texelFetch(uExp, ivec2(0), 0).r;
        float g = uGain / (e + 1e-12);
        float f = tanh(a.a * g * 0.36);
        float m = abs(f);
        vec3 lo = f < 0.0 ? uNeg0 : uPos0, hi = f < 0.0 ? uNeg1 : uPos1;
        vec3 col = mix(uBg, lo, smoothstep(0.0, 0.5, m));
        col = mix(col, hi, smoothstep(0.35, 0.88, m));
        col = mix(col, uTop, smoothstep(0.86, 1.0, m) * 0.7);
        /* the glow of the averaged energy */
        float gl = texture(uGlow, uv).r * g * g * 0.16;
        gl = gl / (1.0 + gl);
        if (uLight > 0.5) col *= 1.0 - 0.28 * gl * vec3(0.3, 0.55, 0.9);
        else col += uGlowC * gl * 0.42;
        /* materials */
        vec4 mt = texture(uMat, uv);
        float de = mt.a, dg = mt.r;
        float edge = de > 0.02 && de < 0.98 ? 1.0 - smoothstep(0.3, 1.3, abs(de - 0.5) / max(fwidth(de), 1e-4)) : 0.0;
        col = mix(col, uDiel, de * 0.05 + dg * 0.11);
        col = mix(col, uDiel, edge * 0.55);
        float pe = smoothstep(0.25, 0.6, mt.g);
        float pedge = mt.g > 0.02 && mt.g < 0.98 ? 1.0 - smoothstep(0.0, 1.2, abs(mt.g - 0.5) / max(fwidth(mt.g), 1e-4)) : 0.0;
        col = mix(col, uPec * (uLight > 0.5 ? 0.8 : 0.55), pe * 0.85);
        col = mix(col, uPec, pedge * 0.8);
        float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 7.0));
        col = mix(col, uPec, mt.b * (0.12 + 0.18 * hatch));
        /* Poynting arrows on a grid */
        if (uPoynt > 0.5) {
            vec2 cell = floor(gl_FragCoord.xy / uArrow);
            vec2 cc = (cell + 0.5) * uArrow;
            vec2 cuv = uMap.xy + cc * uMap.zw;
            vec2 S = texture(uAcc, cuv).gb;
            float sm = length(S) * g * g;
            if (sm > 0.02) {
                vec2 d = normalize(S);
                vec2 q = gl_FragCoord.xy - cc;
                q = vec2(dot(q, d), dot(q, vec2(-d.y, d.x)));
                float L = uArrow * 0.38 * clamp(sqrt(sm) * 0.9, 0.3, 1.0);
                float shaft = max(abs(q.y) - 0.6, max(q.x - L * 0.35, -L - q.x));
                float hd = max(abs(q.y) - (L - q.x) * 0.45, max(q.x - L, L * 0.3 - q.x));
                float dist = min(shaft, hd);
                float al = (1.0 - smoothstep(-0.5, 0.7, dist)) * smoothstep(0.02, 0.12, sm) * 0.55;
                col = mix(col, uLight > 0.5 ? vec3(0.1) : vec3(0.92, 0.94, 0.96), al);
            }
        }
        /* shade behind the text */
        vec2 fp = vec2(gl_FragCoord.x, uH - gl_FragCoord.y);
        float sh = 0.0;
        for (int i = 0; i < uNCard; i++) {
            vec2 c = 0.5 * (uCard[i].xy + uCard[i].zw), hb = 0.5 * (uCard[i].zw - uCard[i].xy);
            vec2 q = abs(fp - c) - hb;
            float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
            sh = max(sh, 1.0 - smoothstep(-uCardR * 0.2, uCardR, d));
        }
        col = mix(col, uBg, sh * 0.62);
        /* fade, dither */
        vec2 sc = gl_FragCoord.xy * uMap.zw;
        col = mix(uBg, col, uFade);
        col += (hash(gl_FragCoord.xy + fract(uTime)) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });

/* the far-field polar plot */
static const char *FS_POLAR = VERSION GLSL(
    uniform sampler2D uPat, uPatMax, uAna;
    uniform vec3 uInk, uDim, uAcc, uBg;
    uniform float uPx, uLight, uMark, uShowMark, uFade, uHasAna;
    in vec2 vP;
    out vec4 o;
    float stroke(float d, float w) { return 1.0 - smoothstep(w - 0.5, w + 0.5, abs(d)); }
    void main() {
        float R = 0.86;
        float r = length(vP) / R, ang = atan(vP.y, vP.x);
        float rp = r * R / uPx * 0.5;          /* r in pixels */
        float pxr = 2.0 / (uPx * R);           /* one pixel in r */
        float u = fract(ang / 6.28318531 + 1.0);
        float mx = texelFetch(uPatMax, ivec2(0), 0).r;
        float m = texture(uPat, vec2(u, 0.5)).r / mx;
        float an = texture(uAna, vec2(u, 0.5)).r;
        float al = 0.0;
        vec3 col = uBg;
        /* backing disc */
        float disc = 1.0 - smoothstep(1.08 - pxr, 1.08 + pxr, r);
        al = disc * 0.62;
        /* rings and spokes */
        float g = 0.0;
        for (int i = 1; i <= 4; i++) g = max(g, stroke((r - float(i) * 0.25) / pxr, 0.5));
        float sp = abs(fract(ang / 0.5235988 + 0.5) - 0.5) * 0.5235988 * r / pxr;
        g = max(g, stroke(sp, 0.45) * step(r, 1.0) * step(0.08, r));
        col = mix(col, uDim, g * 0.5);
        al = max(al, g * 0.5 * disc);
        /* the measured pattern, filled faintly and stroked */
        float inside = 1.0 - smoothstep(m - pxr, m + pxr, r);
        col = mix(col, uAcc, inside * 0.16);
        float dm = (r - m) / pxr;
        float sk = stroke(dm, 0.9);
        col = mix(col, uAcc, sk);
        al = max(al, sk);
        /* the analytic pattern, dashed */
        if (uHasAna > 0.5) {
            float dash = step(0.45, fract(ang / 6.28318531 * 90.0));
            float sa = stroke((r - an) / pxr, 0.6) * dash;
            col = mix(col, uInk, sa * 0.85);
            al = max(al, sa * 0.85);
        }
        /* where it should point */
        if (uShowMark > 0.5) {
            vec2 d = vec2(cos(uMark), sin(uMark));
            float along = dot(vP / R, d), across = abs(dot(vP / R, vec2(-d.y, d.x))) / pxr;
            float mk = stroke(across, 0.5) * step(0.0, along) * step(along, 1.06) * step(0.5, fract(along * 14.0));
            col = mix(col, uInk, mk * 0.7);
            al = max(al, mk * 0.7);
        }
        o = vec4(col, al * uFade);
    });

/* the probe trace */
static const char *FS_TRACE = VERSION GLSL(
    uniform sampler2D uRing, uMaxT;
    uniform vec3 uInk, uDim, uAcc, uBg;
    uniform float uHead, uSpan, uPxW, uPxH, uFade;
    in vec2 vP;
    out vec4 o;
    float val(float x) {
        float i = uHead - (1.0 - x) * uSpan;
        float w = float(textureSize(uRing, 0).x);
        float a = mod(floor(i), w), b = mod(floor(i) + 1.0, w);
        float f = fract(i);
        float v = mix(texelFetch(uRing, ivec2(int(a), 0), 0).r, texelFetch(uRing, ivec2(int(b), 0), 0).r, f);
        return v / texelFetch(uMaxT, ivec2(0), 0).r * 0.82;
    }
    void main() {
        vec2 p = vP;
        float x = p.x * 0.5 + 0.5;
        float dx = 1.0 / uPxW;
        float v0 = val(x - dx), v1 = val(x), v2 = val(x + dx);
        float lo = min(v1, min(0.5 * (v0 + v1), 0.5 * (v1 + v2))), hi = max(v1, max(0.5 * (v0 + v1), 0.5 * (v1 + v2)));
        float py = 2.0 / uPxH;
        float d = max(max(lo - p.y, p.y - hi), 0.0) / py;
        float line = 1.0 - smoothstep(0.4, 1.4, d);
        float axis = (1.0 - smoothstep(0.0, 1.0, abs(p.y) / py)) * 0.5;
        float grid = (1.0 - smoothstep(0.0, 1.0, abs(fract(x * 8.0 + 0.5) - 0.5) / (8.0 * dx))) * 0.18;
        float fadein = smoothstep(0.0, 0.12, x);
        vec3 col = mix(uBg, uDim, max(axis, grid));
        col = mix(col, uAcc, line * fadein);
        float al = max(0.62, max(axis, line * fadein));
        o = vec4(col, al * uFade);
    });

static const char *VS_TEXT = VERSION GLSL(
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
static const char *FS_TEXT = VERSION GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vCol;
    out vec4 o;
    void main() { o = vec4(vCol.rgb, vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r); });

/* ---- GL plumbing ---- */
static GLuint compile(GLenum type, const char *src_) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src_, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "fdtd: shader:\n%s\n", log);
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

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static GLuint mk_fbo(GLuint t0, GLuint t1) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t0, 0);
    if (t1) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, t1, 0);
        static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, bufs);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete");
    return f;
}
static void clear_fbo(GLuint f) {
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
}

static GLuint prog_patsum, tPatPart, fPatPart, tMaxPart, fMaxPart, prog_expo_s, tExpS, fExpS;
static GLuint prog_t, prog_z, prog_probe, prog_ntff, prog_pat, prog_max, prog_prep, prog_expo, prog_blur, prog_comp,
              prog_polar, prog_trace, prog_text;
static GLuint vao, vao_text, vbo_text, tex_atlas;
static GLuint tFz[2], tFt[2], fFz[2], fFt[2];
static int cz, ct;
static GLuint tMatZ, tMatT, tMatV;
static GLuint tAcc[2], fAcc[2];
static int ca;
static GLuint tGlow[2], fGlow[2];
static int gw, gh;
static GLuint tExp[2], fExp[2];
static int ce;
#define RING 4096
static GLuint tRing, fRing;
#define MAXNP 4096
static GLuint tNPos, tNA[2], tNB[2], fN[2];
static int cn, np;
static float ncen[2];
static GLuint tPat, fPat, tPatMax, fPatMax, tAna, tTMax, fTMax;
static int grid_w, grid_h;
static long step_n;
static float frac = 1;
static float pal_c[13][3];
static time_t locked_at;

static void gl_init(void) {
    prog_t = mk_prog(VS, FS_STEP_T);
    prog_z = mk_prog(VS, FS_STEP_Z);
    prog_probe = mk_prog(VS, FS_PROBE);
    prog_ntff = mk_prog(VS, FS_NTFF);
    prog_pat = mk_prog(VS, FS_PATTERN);
    prog_max = mk_prog(VS, FS_MAX);
    prog_patsum = mk_prog(VS, FS_PATSUM);
    prog_expo_s = mk_prog(VS, FS_EXPO_S);
    tExpS = mk_tex(GL_R32F, 32, 18, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
    fExpS = mk_fbo(tExpS, 0);
    prog_prep = mk_prog(VS, FS_PREP);
    prog_expo = mk_prog(VS, FS_EXPO);
    prog_blur = mk_prog(VS, FS_BLUR);
    prog_comp = mk_prog(VS, FS_COMP);
    prog_polar = mk_prog(VS_RECT, FS_POLAR);
    prog_trace = mk_prog(VS_RECT, FS_TRACE);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
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
    glGenTextures(1, &tex_atlas);

    tRing = mk_tex(GL_R32F, RING, 1, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
    fRing = mk_fbo(tRing, 0);
    tNPos = mk_tex(GL_RGBA32F, MAXNP, 1, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
    for (int i = 0; i < 2; i++) {
        tNA[i] = mk_tex(GL_RGBA32F, MAXNP, 1, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
        tNB[i] = mk_tex(GL_RGBA32F, MAXNP, 1, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
        fN[i] = mk_fbo(tNA[i], tNB[i]);
        tExp[i] = mk_tex(GL_R32F, 1, 1, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
        fExp[i] = mk_fbo(tExp[i], 0);
    }
    tPat = mk_tex(GL_R32F, 360, 1, GL_RED, GL_FLOAT, NULL, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    fPat = mk_fbo(tPat, 0);
    tPatPart = mk_tex(GL_RG32F, 360, 16, GL_RG, GL_FLOAT, NULL, GL_NEAREST);
    fPatPart = mk_fbo(tPatPart, 0);
    tMaxPart = mk_tex(GL_R32F, 64, 1, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
    fMaxPart = mk_fbo(tMaxPart, 0);
    tPatMax = mk_tex(GL_R32F, 1, 1, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
    fPatMax = mk_fbo(tPatMax, 0);
    tTMax = mk_tex(GL_R32F, 1, 1, GL_RED, GL_FLOAT, NULL, GL_NEAREST);
    fTMax = mk_fbo(tTMax, 0);
    tAna = mk_tex(GL_R32F, 360, 1, GL_RED, GL_FLOAT, NULL, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

    const Palette *P = &palettes[pal];
    const char *cs[13] = {P->bg, P->neg0, P->neg1, P->pos0, P->pos1, P->top, P->glow, P->diel, P->pec, P->ink, P->dim, P->accent, P->bg};
    for (int i = 0; i < 13; i++) hex(cs[i], pal_c[i]);
}
enum { PC_BG, PC_NEG0, PC_NEG1, PC_POS0, PC_POS1, PC_TOP, PC_GLOW, PC_DIEL, PC_PEC, PC_INK, PC_DIM, PC_ACC };

/* the grid's textures, (re)made when its size changes */
static void grid_alloc(void) {
    if (grid_w == NX && grid_h == NY) return;
    if (grid_w) {
        glDeleteTextures(2, tFz); glDeleteTextures(2, tFt); glDeleteTextures(2, tAcc); glDeleteTextures(2, tGlow);
        glDeleteFramebuffers(2, fFz); glDeleteFramebuffers(2, fFt); glDeleteFramebuffers(2, fAcc); glDeleteFramebuffers(2, fGlow);
        glDeleteTextures(1, &tMatZ); glDeleteTextures(1, &tMatT); glDeleteTextures(1, &tMatV);
    }
    grid_w = NX; grid_h = NY;
    gw = (NX + 1) / 2; gh = (NY + 1) / 2;
    for (int i = 0; i < 2; i++) {
        tFz[i] = mk_tex(GL_RGBA32F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR);
        tFt[i] = mk_tex(GL_RGBA32F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR);
        fFz[i] = mk_fbo(tFz[i], 0);
        fFt[i] = mk_fbo(tFt[i], 0);
        tAcc[i] = mk_tex(GL_RGBA16F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR);
        fAcc[i] = mk_fbo(tAcc[i], 0);
        tGlow[i] = mk_tex(GL_R16F, gw, gh, GL_RED, GL_FLOAT, NULL, GL_LINEAR);
        fGlow[i] = mk_fbo(tGlow[i], 0);
    }
    tMatZ = mk_tex(GL_RGBA32F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
    tMatT = mk_tex(GL_RGBA32F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
    tMatV = mk_tex(GL_RGBA8, NX, NY, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_LINEAR);
}

/* ---- materials: paint the shapes onto the three Yee sites of every cell,
 * then turn eps, mu, sigma and PEC into update coefficients ---- */
static void paint_materials(void) {
    size_t n = (size_t)NX * NY;
    float *eps = malloc(n * 3 * sizeof *eps), *mu = malloc(n * 3 * sizeof *mu);
    float *sig = calloc(n * 3, sizeof *sig);
    unsigned char *pec = calloc(n * 3, 1);
    for (size_t i = 0; i < n * 3; i++) eps[i] = mu[i] = 1;
    static const float off[3][2] = {{0, 0}, {0, 0.5f}, {0.5f, 0}};   /* Fz, Fx, Fy sites */
    for (int k = 0; k < nsh; k++) {
        const Shape *s = &sh[k];
        float b[4], i0 = 1e9f, i1 = -1e9f, j0 = 1e9f, j1 = -1e9f;
        shape_bounds(s, b);
        for (int c = 0; c < 4; c++) {
            float ci, cj;
            local_to_cell(b[c & 1 ? 2 : 0], b[c & 2 ? 3 : 1], &ci, &cj);
            i0 = fminf(i0, ci); i1 = fmaxf(i1, ci); j0 = fminf(j0, cj); j1 = fmaxf(j1, cj);
        }
        int ia = (int)fmaxf(0, floorf(i0) - 1), ib = (int)fminf(NX - 1, ceilf(i1) + 1);
        int ja = (int)fmaxf(0, floorf(j0) - 1), jb = (int)fminf(NY - 1, ceilf(j1) + 1);
        for (int j = ja; j <= jb; j++)
            for (int i = ia; i <= ib; i++)
                for (int st = 0; st < 3; st++) {
                    float lx, ly;
                    cell_to_local(i + off[st][0], j + off[st][1], &lx, &ly);
                    float dc = shape_sd(s, lx, ly) * nl;
                    if (dc > 1) continue;
                    float cv = clampf(0.5f - dc, 0, 1);
                    size_t q = ((size_t)j * NX + i) * 3 + st;
                    switch (s->mat) {
                    case M_PEC: if (dc < 0) pec[q] = 1; break;
                    case M_AIR:
                        eps[q] += (1 - eps[q]) * cv; mu[q] += (1 - mu[q]) * cv; sig[q] -= sig[q] * cv;
                        if (cv > 0.5f) pec[q] = 0;
                        break;
                    case M_LUNE: {
                        float r = hypotf(lx - s->p[0], ly - s->p[1]) / s->p[2];
                        eps[q] += (2 - fminf(r * r, 1) - eps[q]) * cv;
                        break;
                    }
                    default:
                        eps[q] += (s->eps - eps[q]) * cv;
                        mu[q] += (s->mu - mu[q]) * cv;
                        sig[q] += (s->sig - sig[q]) * cv;
                    }
                }
    }
    float *mz = malloc(n * 4 * sizeof *mz), *mt = malloc(n * 4 * sizeof *mt);
    unsigned char *mv = malloc(n * 4);
    for (size_t c = 0; c < n; c++) {
        float co[3][2];
        for (int st = 0; st < 3; st++) {
            size_t q = c * 3 + st;
            int electric = te ? st > 0 : st == 0;
            if (!electric) { co[st][0] = 1; co[st][1] = S_COURANT / mu[q]; continue; }
            if (pec[q]) { co[st][0] = co[st][1] = 0; continue; }
            float l = sig[q] / (2 * eps[q]);
            co[st][0] = (1 - l) / (1 + l);
            co[st][1] = S_COURANT / eps[q] / (1 + l);
        }
        mz[c * 4] = co[0][0]; mz[c * 4 + 1] = co[0][1]; mz[c * 4 + 2] = mz[c * 4 + 3] = 0;
        mt[c * 4] = co[1][0]; mt[c * 4 + 1] = co[1][1]; mt[c * 4 + 2] = co[2][0]; mt[c * 4 + 3] = co[2][1];
        /* for the eye: dielectric strength, conductor, loss, dielectric presence */
        float e = te ? (eps[c * 3 + 1] + eps[c * 3 + 2]) / 2 : eps[c * 3];
        int p = pec[c * 3] || pec[c * 3 + 1] || pec[c * 3 + 2];
        mv[c * 4] = (unsigned char)(clampf((e - 1) / 1.25f, 0, 1) * 255);
        mv[c * 4 + 1] = p ? 255 : 0;
        mv[c * 4 + 2] = (unsigned char)(clampf(sig[c * 3 + (te ? 1 : 0)] * 12, 0, 1) * 255);
        mv[c * 4 + 3] = (unsigned char)(clampf((e - 1) * 8, 0, 1) * 255);
    }
    glBindTexture(GL_TEXTURE_2D, tMatZ);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RGBA, GL_FLOAT, mz);
    glBindTexture(GL_TEXTURE_2D, tMatT);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RGBA, GL_FLOAT, mt);
    glBindTexture(GL_TEXTURE_2D, tMatV);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RGBA, GL_UNSIGNED_BYTE, mv);
    free(eps); free(mu); free(sig); free(pec); free(mz); free(mt); free(mv);
}

/* the NTFF box: points a cell apart round it, outward normals */
static void ntff_setup(void) {
    np = 0;
    if (!has_ntff) return;
    float a0, b0, a1, b1;
    local_to_cell(nbox[0], nbox[1], &a0, &b0);
    local_to_cell(nbox[2], nbox[3], &a1, &b1);
    int i0 = (int)lroundf(fminf(a0, a1)), i1 = (int)lroundf(fmaxf(a0, a1));
    int j0 = (int)lroundf(fminf(b0, b1)), j1 = (int)lroundf(fmaxf(b0, b1));
    i0 = i0 < PML + 2 ? PML + 2 : i0; j0 = j0 < PML + 2 ? PML + 2 : j0;
    i1 = i1 > NX - PML - 3 ? NX - PML - 3 : i1; j1 = j1 > NY - PML - 3 ? NY - PML - 3 : j1;
    float *p = calloc(MAXNP * 4, sizeof *p);
    #define PT(x, y, nx_, ny_) do { if (np < MAXNP) { float *q_ = p + np * 4; q_[0] = x; q_[1] = y; q_[2] = nx_; q_[3] = ny_; np++; } } while (0)
    for (int i = i0; i < i1; i++) PT(i + 0.5f, (float)j0, 0, -1);
    for (int j = j0; j < j1; j++) PT((float)i1, j + 0.5f, 1, 0);
    for (int i = i1; i > i0; i--) PT(i - 0.5f, (float)j1, 0, 1);
    for (int j = j1; j > j0; j--) PT((float)i0, j - 0.5f, -1, 0);
    #undef PT
    ncen[0] = (i0 + i1) / 2.0f; ncen[1] = (j0 + j1) / 2.0f;
    glBindTexture(GL_TEXTURE_2D, tNPos);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MAXNP, 1, GL_RGBA, GL_FLOAT, p);
    free(p);
    for (int i = 0; i < 2; i++) clear_fbo(fN[i]);
}

/* the analytic far field, per screen degree, normalised */
static void analytic_upload(void) {
    if (!ana_kind) return;
    float a[360], mx = 1e-9f;
    for (int i = 0; i < 360; i++) {
        float ph = (i + 0.5f) * DEG - (rot90 ? PI / 2 : 0);   /* back to the scene's angle */
        float v;
        if (ana_kind == 1) {          /* half-wave dipole along y: cos(pi/2 cos th) / sin th */
            float c = sinf(ph), s = fabsf(cosf(ph));
            v = s < 1e-3f ? 0 : fabsf(cosf(PI / 2 * c) / s);
        } else {                       /* eight elements at half a wavelength */
            float u = PI * sinf(ph) - dphi;
            float dn = 8 * sinf(u / 2);
            v = fabsf(dn) < 1e-5f ? 1 : fabsf(sinf(4 * u) / dn);
        }
        a[i] = v; mx = fmaxf(mx, v);
    }
    for (int i = 0; i < 360; i++) a[i] /= mx;
    glBindTexture(GL_TEXTURE_2D, tAna);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 360, 1, GL_RED, GL_FLOAT, a);
}

/* ---- the domain over the layout ---- */
static float lay[4];
static void domain_setup(void) {
    rot90 = front[3] > front[2] * 1.05f;
    float rw = roi[2] - roi[0], rh = roi[3] - roi[1];
    float sw = rot90 ? rh : rw, shh = rot90 ? rw : rh;
    pxl = fminf(front[2] * 0.86f / sw, front[3] * 0.78f / shh);
    float rcx = (roi[0] + roi[2]) / 2, rcy = (roi[1] + roi[3]) / 2;
    float sx = rot90 ? -rcy : rcx, sy = rot90 ? rcx : rcy;
    scx = front[0] + front[2] / 2 - sx * pxl;
    scy = front[1] + front[3] / 2 + sy * pxl + front[3] * 0.02f;
    cp = pxl / nl_pref;
    for (int it = 0; it < 40; it++) {
        NX = (int)ceilf(lay[2] / cp) + 2 * PML;
        NY = (int)ceilf(lay[3] / cp) + 2 * PML;
        if ((double)NX * NY <= opt_cells || pxl / cp <= 12) break;
        cp *= (float)fmax(1.01, sqrt((double)NX * NY / opt_cells));
    }
    nl = pxl / cp;
    gx0 = lay[0] - PML * cp;
    gy0 = lay[1] - PML * cp;
}

/* ---- running the simulation ---- */
static float rate;                      /* steps per second */
static float steps_period(void) { return nl / S_COURANT; }

static void set_sources(GLuint p, float tp) {
    float ps[MAXSRC][4], l0[MAXSRC][4], l1[MAXSRC][4];
    for (int i = 0; i < nsrc; i++) {
        const Source *s = &src[i];
        int comp = s->comp;
        if (rot90 && comp) comp = comp == C_X ? C_Y : C_X;
        float a, b, c, d;
        local_to_cell(s->x0, s->y0, &a, &b);
        ps[i][0] = a; ps[i][1] = b; ps[i][2] = ps[i][3] = 0;
        l0[i][0] = a; l0[i][1] = b;
        if (s->line) { local_to_cell(s->x1, s->y1, &c, &d); l0[i][2] = c; l0[i][3] = d; }
        else { l0[i][2] = a; l0[i][3] = b; }
        l1[i][0] = (float)comp;
        l1[i][1] = src_value(s, tp) * (s->line ? 1.0f : 1.0f);
        l1[i][2] = s->line ? (float)s->profile : -1.0f;
        l1[i][3] = s->width;
    }
    glUniform1i(U(p, "uNps"), nsrc);
    if (nsrc) {
        glUniform4fv(U(p, "uPs"), nsrc, &ps[0][0]);
        glUniform4fv(U(p, "uLs0"), nsrc, &l0[0][0]);
        glUniform4fv(U(p, "uLs1"), nsrc, &l1[0][0]);
    }
}

static void sim_uniforms(GLuint p) {
    glUniform2i(U(p, "uN"), NX, NY);
    /* CPML: cubic grading, sigma at its optimum (m + 1) S 0.8, kappa to 6, alpha 0.05 */
    glUniform4f(U(p, "uPml"), (float)PML, 0.8f * 4 * S_COURANT * 1.0f, 6.0f, 0.05f);
}

static void bind_tex(int unit, GLuint t) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, t);
}

static void sim_step(void) {
    float dtp = 1.0f / steps_period();
    float tz = (step_n + 1) * dtp, tt = tz - 0.5f * dtp;
    act_dynamic(tz);
    glBindVertexArray(vao);
    glViewport(0, 0, NX, NY);
    glDisable(GL_BLEND);
    /* the transverse fields */
    glUseProgram(prog_t);
    sim_uniforms(prog_t);
    set_sources(prog_t, tt);
    glUniform1i(U(prog_t, "uFz"), 0); glUniform1i(U(prog_t, "uFt"), 1); glUniform1i(U(prog_t, "uMatT"), 2);
    bind_tex(0, tFz[cz]); bind_tex(1, tFt[ct]); bind_tex(2, tMatT);
    glBindFramebuffer(GL_FRAMEBUFFER, fFt[ct ^ 1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ct ^= 1;
    /* then Fz */
    glUseProgram(prog_z);
    sim_uniforms(prog_z);
    set_sources(prog_z, tz);
    glUniform1i(U(prog_z, "uFz"), 0); glUniform1i(U(prog_z, "uFt"), 1); glUniform1i(U(prog_z, "uMatZ"), 2);
    bind_tex(0, tFz[cz]); bind_tex(1, tFt[ct]); bind_tex(2, tMatZ);
    glBindFramebuffer(GL_FRAMEBUFFER, fFz[cz ^ 1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    cz ^= 1;
    step_n++;
    /* the probe, into the ring */
    {
        float a, b;
        local_to_cell(probe[0], probe[1], &a, &b);
        glUseProgram(prog_probe);
        glUniform1i(U(prog_probe, "uFz"), 0);
        glUniform2f(U(prog_probe, "uAt"), a, b);
        bind_tex(0, tFz[cz]);
        glBindFramebuffer(GL_FRAMEBUFFER, fRing);
        glEnable(GL_SCISSOR_TEST);
        int x = (int)(step_n % RING);
        glViewport(0, 0, RING, 1);
        glScissor(x, 0, 1, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_SCISSOR_TEST);
    }
    /* the far-field DFT on the box */
    if (np) {
        float w = 2 * PI * dtp;
        glUseProgram(prog_ntff);
        glUniform1i(U(prog_ntff, "uPos"), 0); glUniform1i(U(prog_ntff, "uA"), 1); glUniform1i(U(prog_ntff, "uB"), 2);
        glUniform1i(U(prog_ntff, "uFz"), 3); glUniform1i(U(prog_ntff, "uFt"), 4);
        glUniform2f(U(prog_ntff, "uRotZ"), cosf(w * step_n), sinf(w * step_n));
        glUniform2f(U(prog_ntff, "uRotT"), cosf(w * (step_n - 0.5f)), sinf(w * (step_n - 0.5f)));
        glUniform1f(U(prog_ntff, "uDecay"), expf(-dtp / ntff_tau));
        bind_tex(0, tNPos); bind_tex(1, tNA[cn]); bind_tex(2, tNB[cn]); bind_tex(3, tFz[cz]); bind_tex(4, tFt[ct]);
        glBindFramebuffer(GL_FRAMEBUFFER, fN[cn ^ 1]);
        glViewport(0, 0, np, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        cn ^= 1;
    }
}

/* ---- the show: which act when ---- */
static int cur_k = -1, need_restart;

static int act_for(int k) {
    if (opt_act) {
        for (int a = 0; a < NACT; a++) if (!strcmp(opt_act, act_names[a])) return a;
        return 0;
    }
    int cyc = k / NACT, slot = k % NACT;
    int perm[NACT];
    for (int c = 0;; c++) {
        uint32_t r = hash32(opt_seed * 2654435761u + (uint32_t)(cyc - c) * 40503u + 17);
        for (int i = 0; i < NACT; i++) perm[i] = i;
        for (int i = NACT - 1; i > 0; i--) {
            r = hash32(r + (uint32_t)i);
            int j = (int)(r % (uint32_t)(i + 1));
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        break;
    }
    return perm[slot];
}

static void act_start(int k) {
    act = act_for(k);
    act_no = act;
    act_seed = hash32(opt_seed ^ (uint32_t)k * 0x9E3779B9u);
    act_frame();
    domain_setup();
    act_build();
    grid_alloc();
    paint_materials();
    for (int i = 0; i < 2; i++) { clear_fbo(fFz[i]); clear_fbo(fFt[i]); clear_fbo(fAcc[i]); clear_fbo(fGlow[i]); }
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fExp[i]);
        glClearColor(1e-6f, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    clear_fbo(fRing);
    ntff_setup();
    step_n = 0;
    rate = fv * steps_period();
    act_dynamic(0);
    analytic_upload();
    cur_k = k;
    need_restart = 0;
}

/* advance the show to t seconds in, stepping at most maxsteps; returns the
 * seconds into the act */
static double show_advance(double t, int maxsteps) {
    if (t < 0) t = 0;
    int k = (int)floor(t / opt_act_secs);
    if (k != cur_k || need_restart) act_start(k);
    double ta = t - k * (double)opt_act_secs;
    double want = ta * rate;
    int n = 0;
    while (step_n < (long)floor(want) && n < maxsteps) { sim_step(); n++; }
    frac = clampf((float)(want - floor(want)), 0, 1);
    if (step_n < (long)floor(want)) frac = 1;
    return ta;
}

/* ---- a frame ---- */
static int atlas_n = -1;
static float ppol[4], ptr[4];           /* the polar plot's and the trace's rects (fb px) */
static int show_polar, show_trace;
static float trace_span;
static double q_at;

/* the cavity's Q from the probe's ring-down: fit the log of the peaks
 * since the last pulse died away */
static void estimate_q(void) {
    static float ring[RING];
    glBindFramebuffer(GL_FRAMEBUFFER, fRing);
    glReadPixels(0, 0, RING, 1, GL_RED, GL_FLOAT, ring);
    float spp = steps_period();
    float tp = step_n / spp;
    float k = floorf((tp - w_t0) / w_rep);
    if (k < 0) k = 0;
    float quiet = (w_t0 + k * w_rep + 2.6f * w_tau) * spp;   /* steps: the burst is over */
    long s0 = (long)ceilf(quiet), s1 = step_n;
    if (s1 - s0 > RING - 8) s0 = s1 - RING + 8;
    if (s1 - s0 < 4 * spp) return;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (long s = s0 + 1; s < s1 - 1; s++) {
        float a = fabsf(ring[(s - 1) % RING]), b = fabsf(ring[s % RING]), c = fabsf(ring[(s + 1) % RING]);
        if (b > a && b >= c && b > 1e-9f) {
            sx += s; sy += log(b); sxx += (double)s * s; sxy += s * log(b); n++;
        }
    }
    if (n < 6) return;
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    if (slope >= 0) return;
    double tau = -1 / slope;                    /* steps */
    q_est = PI * (w_fm / spp) * tau;
}

/* the text blocks' extents (fb px, y down), shaded behind in the next frame */
static float cards[3][4];
static int ncards;

static void build_text(double ta) {
    tvn = 0;
    ncards = 0;
    const float *ink = pal_c[PC_INK], *dim = pal_c[PC_DIM], *acc = pal_c[PC_ACC];
    float cink[4] = {ink[0], ink[1], ink[2], 0.95f}, cdim[4] = {dim[0], dim[1], dim[2], 0.95f};
    float cacc[4] = {acc[0], acc[1], acc[2], 0.95f};
    float fade = smoothf(0.2f, 1.6f, (float)ta) * smoothf(opt_act_secs, opt_act_secs - 1.2f, (float)ta);
    cink[3] *= fade; cdim[3] *= fade; cacc[3] *= fade;
    float bgc[4] = {pal_c[PC_BG][0], pal_c[PC_BG][1], pal_c[PC_BG][2], 0.5f * fade};
    char b[160];
    float sxs = W / view[2];                 /* framebuffer px per logical px */
    /* labels in the scene */
    {
        float u = fminf(front[2], front[3]) * sxs;
        int s = style(F_UI, fmaxf(11, u * 0.0145f));
        for (int i = 0; i < nlab; i++) {
            float x, y;
            local_to_fb(lab_list[i].x, lab_list[i].y, &x, &y);
            if (x < -400 || x > W + 400 || y < -50 || y > H + 50) continue;
            text(s, x, y + cap_h(s) * 0.5f, lab_list[i].align, lab_list[i].s, cdim, 0.3f);
        }
    }
    /* the time and time away, on the front output */
    if (front_here) {
        float u = fminf(front[2], front[3]) * sxs, m = u * 0.04f;
        int st = style(F_UIB, u * 0.03f), sa = style(F_UI, u * 0.0155f);
        time_t now_t = time(NULL);
        struct tm tm;
        localtime_r(&now_t, &tm);
        snprintf(b, sizeof b, "%02d:%02d", tm.tm_hour, tm.tm_min);
        float ink2[4] = {ink[0], ink[1], ink[2], 0.92f};
        text(st, W - m, m + cap_h(st), 1, b, ink2, 0.5f);
        long away = (long)(wall() - (double)locked_at);
        if (away < 0) away = 0;
        if (away < 3600) snprintf(b, sizeof b, "away %ld min", away / 60);
        else snprintf(b, sizeof b, "away %ldh %02ld min", away / 3600, away / 3600 ? (away / 60) % 60 : 0);
        float dim2[4] = {dim[0], dim[1], dim[2], 0.92f};
        float wa = text(sa, W - m, m + cap_h(st) + cap_h(sa) * 2.1f, 1, b, dim2, 0.4f);
        float *c = cards[ncards++];
        c[0] = W - m - fmaxf(wa, text_width(st, "00:00", 0.5f)); c[1] = m; c[2] = W - m; c[3] = m + cap_h(st) + cap_h(sa) * 2.4f;
    }
    if (!inset_here) return;
    float u = fminf(inset[2], inset[3]) * sxs, m = u * 0.04f;
    /* the title card */
    int sn = style(F_UI, u * 0.0145f), stt = style(F_UIB, u * 0.032f), ss = style(F_UI, u * 0.0165f);
    float y = m + cap_h(sn);
    snprintf(b, sizeof b, "%s  ·  %s", numerals[act_no], te ? "TEz" : "TMz");
    float x0 = m;
    float wmax = text(sn, x0, y, -1, b, cacc, 1.5f);
    y += cap_h(stt) * 1.9f;
    wmax = fmaxf(wmax, text(stt, x0, y, -1, title, cink, 0.2f));
    y += cap_h(ss) * 2.3f;
    if (act == A_ARRAY)
        snprintf(sub, sizeof sub, "Δφ = %.0f° between elements  →  sin θ = Δφ λ / 2πd,  θ = %.1f°", dphi / DEG, steer / DEG);
    if (act == A_CAVITY) {
        if (q_est > 0) snprintf(sub, sizeof sub, "TM₃₂ at %.3f f₀ rings down through an iris  ·  Q ≈ %.0f", cav_f, q_est);
        else snprintf(sub, sizeof sub, "TM₃₂ at %.3f f₀ rings down through an iris", cav_f);
    }
    wmax = fmaxf(wmax, text(ss, x0, y, -1, sub, cdim, 0.1f));
    /* the curl equations */
    float pe = u * 0.024f;
    y += pe * 2.3f;
    const char *e1 = "∇ × 𝐄 = −𝜇 {∂𝐇|∂𝑡}", *e2 = "∇ × 𝐇 = 𝜀 {∂𝐄|∂𝑡} + 𝜎𝐄";
    math_line(pe, x0, y, e1, cink);
    y += pe * 2.6f;
    math_line(pe, x0, y, e2, cink);
    y += pe * 1.9f;
    if (te) snprintf(b, sizeof b, "𝐻_𝑧 , 𝐸_𝑥 , 𝐸_𝑦   on a Yee grid,  𝑐Δ𝑡 = ½ Δ𝑥");
    else snprintf(b, sizeof b, "𝐸_𝑧 , 𝐻_𝑥 , 𝐻_𝑦   on a Yee grid,  𝑐Δ𝑡 = ½ Δ𝑥");
    wmax = fmaxf(wmax, math_line(pe * 0.78f, x0, y, b, cdim));
    {
        float *c = cards[ncards++];
        c[0] = x0; c[1] = m; c[2] = x0 + wmax; c[3] = y + pe * 0.3f;
    }
    /* captions for the insets */
    int sc = style(F_UI, u * 0.0135f);
    if (show_polar) {
        snprintf(b, sizeof b, "far field  |%s|", te ? "Hz" : "Ez");
        text(sc, ppol[0] + ppol[2] / 2, ppol[1] - cap_h(sc) * 0.6f, 0, b, cdim, 0.6f);
        if (ana_kind) {
            text(sc, ppol[0] + ppol[2] / 2, ppol[1] + ppol[3] + cap_h(sc) * 1.8f, 0,
                 ana_kind == 1 ? "measured  ·  dashed: cos(½π cos θ) / sin θ" : "measured  ·  dashed: array factor", cdim, 0.3f);
        }
    }
    if (show_trace) {
        snprintf(b, sizeof b, "probe  %s(t)", te ? "Hz" : "Ez");
        text(sc, ptr[0], ptr[1] - cap_h(sc) * 0.8f, -1, b, cdim, 0.6f);
    }
    (void)bgc;
}

static void inset_layout(void) {
    float sxs = W / view[2];
    float u = fminf(inset[2], inset[3]) * sxs, m = u * 0.04f;
    show_polar = inset_here && has_ntff;
    show_trace = inset_here;
    float P = u * 0.27f;
    ppol[0] = W - m - P; ppol[1] = H - m - P - u * 0.03f; ppol[2] = P; ppol[3] = P;
    ptr[2] = u * 0.42f; ptr[3] = u * 0.1f;
    ptr[0] = m; ptr[1] = H - m - ptr[3];
    trace_span = steps_period() * (act == A_CAVITY ? 40.0f : act == A_DISH ? 30.0f : 6.0f);
    if (trace_span > RING - 2) trace_span = RING - 2;
}

/* the largest |value| of texels n0..n1 (wrapping at w) of a row, into a 1x1
 * target: 64 strided partial maxima, then their max */
static void max_of(GLuint tex, int n0, int n1, int w, GLuint dst) {
    glUseProgram(prog_max);
    glUniform1i(U(prog_max, "uT"), 0);
    glUniform1i(U(prog_max, "uN0"), n0); glUniform1i(U(prog_max, "uN1"), n1);
    glUniform1i(U(prog_max, "uW"), w); glUniform1i(U(prog_max, "uStride"), 64);
    bind_tex(0, tex);
    glViewport(0, 0, 64, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, fMaxPart);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUniform1i(U(prog_max, "uN0"), 0); glUniform1i(U(prog_max, "uN1"), 64);
    glUniform1i(U(prog_max, "uW"), 64); glUniform1i(U(prog_max, "uStride"), 1);
    bind_tex(0, tMaxPart);
    glViewport(0, 0, 1, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, dst);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void set_col(GLuint p, const char *n, int c) { glUniform3fv(U(p, n), 1, pal_c[c]); }

/* --bench: GPU timestamps between the stages of a frame */
enum { PR_SIM, PR_PREP, PR_EXPO, PR_GLOW, PR_PLOTS, PR_COMP, PR_INSETS, PR_TEXT, PR_N };
static const char *prof_name[PR_N] = {"sim", "average", "exposure", "glow", "plots", "picture", "insets", "text"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (t[i + 1] - t[i]) * 1e-6;
}

static void render(GLuint target, double ta, float dt) {
    prof_mark(PR_PREP);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
    /* the averages */
    glViewport(0, 0, NX, NY);
    glUseProgram(prog_prep);
    glUniform1i(U(prog_prep, "uFz"), 0); glUniform1i(U(prog_prep, "uFzPrev"), 1);
    glUniform1i(U(prog_prep, "uFt"), 2); glUniform1i(U(prog_prep, "uAcc"), 3);
    glUniform1f(U(prog_prep, "uFrac"), frac);
    glUniform1f(U(prog_prep, "uK"), 1 - expf(-dt * fv / 1.2f));
    bind_tex(0, tFz[cz]); bind_tex(1, tFz[cz ^ 1]); bind_tex(2, tFt[ct]); bind_tex(3, tAcc[ca]);
    glBindFramebuffer(GL_FRAMEBUFFER, fAcc[ca ^ 1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ca ^= 1;
    prof_mark(PR_EXPO);
    /* the exposure */
    glViewport(0, 0, 32, 18);
    glUseProgram(prog_expo_s);
    glUniform1i(U(prog_expo_s, "uAcc"), 0);
    glUniform2f(U(prog_expo_s, "uLo"), (PML + 0.5f) / NX, (PML + 0.5f) / NY);
    glUniform2f(U(prog_expo_s, "uHi"), 1 - (PML + 0.5f) / NX, 1 - (PML + 0.5f) / NY);
    bind_tex(0, tAcc[ca]);
    glBindFramebuffer(GL_FRAMEBUFFER, fExpS);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glViewport(0, 0, 1, 1);
    glUseProgram(prog_expo);
    glUniform1i(U(prog_expo, "uS"), 0); glUniform1i(U(prog_expo, "uPrev"), 1);
    glUniform2f(U(prog_expo, "uK2"), 1 - expf(-dt / 0.5f), 1 - expf(-dt / 4.0f));
    bind_tex(0, tExpS); bind_tex(1, tExp[ce]);
    glBindFramebuffer(GL_FRAMEBUFFER, fExp[ce ^ 1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ce ^= 1;
    prof_mark(PR_GLOW);
    /* the glow */
    glUseProgram(prog_blur);
    glUniform1i(U(prog_blur, "uT"), 0);
    glUniform2f(U(prog_blur, "uOut"), (float)gw, (float)gh);
    glViewport(0, 0, gw, gh);
    glUniform2f(U(prog_blur, "uDir"), 1, 0);
    glUniform1f(U(prog_blur, "uLod"), 0);
    bind_tex(0, tAcc[ca]);
    glBindFramebuffer(GL_FRAMEBUFFER, fGlow[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUniform2f(U(prog_blur, "uDir"), 0, 1);
    glUniform1f(U(prog_blur, "uLod"), 0);
    bind_tex(0, tGlow[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, fGlow[1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_mark(PR_PLOTS);
    /* the far field and the trace's scale */
    inset_layout();
    if (show_polar || (opt_check && np)) {
        glUseProgram(prog_pat);
        glUniform1i(U(prog_pat, "uPos"), 0); glUniform1i(U(prog_pat, "uA"), 1); glUniform1i(U(prog_pat, "uB"), 2);
        glUniform1i(U(prog_pat, "uNp"), np);
        glUniform1f(U(prog_pat, "uK"), 2 * PI / nl);
        glUniform2f(U(prog_pat, "uC"), ncen[0], ncen[1]);
        glUniform1f(U(prog_pat, "uSgn"), -1);
        bind_tex(0, tNPos); bind_tex(1, tNA[cn]); bind_tex(2, tNB[cn]);
        glViewport(0, 0, 360, 16);
        glBindFramebuffer(GL_FRAMEBUFFER, fPatPart);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glUseProgram(prog_patsum);
        glUniform1i(U(prog_patsum, "uT"), 0);
        bind_tex(0, tPatPart);
        glViewport(0, 0, 360, 1);
        glBindFramebuffer(GL_FRAMEBUFFER, fPat);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        max_of(tPat, 0, 360, 360, fPatMax);
        if (ana_kind == 2) analytic_upload();
    }
    if (show_trace) max_of(tRing, (int)(step_n - (long)trace_span), (int)step_n + 1, RING, fTMax);
    if (act == A_CAVITY && show_trace && ta - q_at > 1.0) { q_at = ta; estimate_q(); }
    if (ta < q_at) q_at = ta;

    prof_mark(PR_COMP);
    /* the picture */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    GLuint p = prog_comp;
    glUseProgram(p);
    glUniform1i(U(p, "uAcc"), 0); glUniform1i(U(p, "uGlow"), 1); glUniform1i(U(p, "uMat"), 2); glUniform1i(U(p, "uExp"), 3);
    bind_tex(0, tAcc[ca]); bind_tex(1, tGlow[1]); bind_tex(2, tMatV); bind_tex(3, tExp[ce]);
    {
        float su = view[2] / (W * cp * NX), sv = view[3] / (H * cp * NY);
        float u0 = (view[0] - gx0) / (cp * NX), v0 = (NY - (view[1] + view[3] - gy0) / cp) / NY;
        glUniform4f(U(p, "uMap"), u0, v0, su, sv);
    }
    set_col(p, "uBg", PC_BG); set_col(p, "uNeg0", PC_NEG0); set_col(p, "uNeg1", PC_NEG1);
    set_col(p, "uPos0", PC_POS0); set_col(p, "uPos1", PC_POS1); set_col(p, "uTop", PC_TOP);
    set_col(p, "uGlowC", PC_GLOW); set_col(p, "uDiel", PC_DIEL); set_col(p, "uPec", PC_PEC);
    glUniform1f(U(p, "uLight"), (float)palettes[pal].light);
    glUniform1f(U(p, "uGain"), act_gain);
    float fade = smoothf(0, 1.2f, (float)ta) * smoothf(opt_act_secs, opt_act_secs - 1.2f, (float)ta);
    glUniform1f(U(p, "uFade"), fade);
    glUniform1f(U(p, "uPoynt"), (float)opt_poynting);
    glUniform1f(U(p, "uArrow"), fmaxf(14, fminf(W, H) * 0.028f));
    glUniform1f(U(p, "uTime"), (float)fmod(ta, 100.0));
    glUniform1f(U(p, "uH"), (float)H);
    glUniform1f(U(p, "uCardR"), fminf(W, H) * 0.035f);
    glUniform1i(U(p, "uNCard"), ncards);
    if (ncards) glUniform4fv(U(p, "uCard"), ncards, &cards[0][0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    prof_mark(PR_INSETS);
    /* the insets */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    float ifade = smoothf(0.2f, 1.6f, (float)ta) * smoothf(opt_act_secs, opt_act_secs - 1.2f, (float)ta);
    if (show_polar) {
        GLuint q = prog_polar;
        glUseProgram(q);
        glUniform4f(U(q, "uRect"), ppol[0], ppol[1], ppol[2], ppol[3]);
        glUniform2f(U(q, "uRes"), (float)W, (float)H);
        glUniform1i(U(q, "uPat"), 0); glUniform1i(U(q, "uPatMax"), 1); glUniform1i(U(q, "uAna"), 2);
        bind_tex(0, tPat); bind_tex(1, tPatMax); bind_tex(2, tAna);
        set_col(q, "uInk", PC_INK); set_col(q, "uDim", PC_DIM); set_col(q, "uAcc", PC_ACC); set_col(q, "uBg", PC_BG);
        glUniform1f(U(q, "uPx"), ppol[2]);
        glUniform1f(U(q, "uHasAna"), (float)(ana_kind != 0));
        glUniform1f(U(q, "uShowMark"), act == A_ARRAY ? 1.0f : 0.0f);
        glUniform1f(U(q, "uMark"), steer + (rot90 ? PI / 2 : 0));
        glUniform1f(U(q, "uFade"), ifade);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    if (show_trace) {
        GLuint q = prog_trace;
        glUseProgram(q);
        glUniform4f(U(q, "uRect"), ptr[0], ptr[1], ptr[2], ptr[3]);
        glUniform2f(U(q, "uRes"), (float)W, (float)H);
        glUniform1i(U(q, "uRing"), 0); glUniform1i(U(q, "uMaxT"), 1);
        bind_tex(0, tRing); bind_tex(1, tTMax);
        set_col(q, "uInk", PC_INK); set_col(q, "uDim", PC_DIM); set_col(q, "uAcc", PC_ACC); set_col(q, "uBg", PC_BG);
        glUniform1f(U(q, "uHead"), (float)step_n);
        glUniform1f(U(q, "uSpan"), trace_span);
        glUniform1f(U(q, "uPxW"), ptr[2]); glUniform1f(U(q, "uPxH"), ptr[3]);
        glUniform1f(U(q, "uFade"), ifade);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    prof_mark(PR_TEXT);
    /* text */
    build_text(ta);
    if (nstyles != atlas_n) {
        build_atlas();
        atlas_n = nstyles;
        build_text(ta);
    }
    if (atlas_dirty) {
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        atlas_dirty = 0;
    }
    if (tvn) {
        glUseProgram(prog_text);
        glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(U(prog_text, "uAtlas"), 0);
        bind_tex(0, tex_atlas);
        glBindVertexArray(vao_text);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    }
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    prof_mark(PR_N);
}

/* ---- where we are on the wall ---- */
static void place(void) {
    if (span_me < 0) {
        lay[0] = lay[1] = 0; lay[2] = (float)W; lay[3] = (float)H;
        memcpy(front, lay, sizeof front); memcpy(inset, lay, sizeof inset); memcpy(view, lay, sizeof view);
        inset_here = front_here = 1;
        return;
    }
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    lay[0] = x0; lay[1] = y0; lay[2] = x1 - x0; lay[3] = y1 - y0;
    /* the front output: the one under the middle of the layout */
    float mx = (x0 + x1) / 2;
    int fi = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { fi = i; break; }
    /* the insets: the biggest portrait output besides it, else the front */
    int ii = fi;
    float best = 0;
    for (int i = 0; i < span_n; i++)
        if (i != fi && span[i][3] > span[i][2] && span[i][2] * span[i][3] > best) { best = span[i][2] * span[i][3]; ii = i; }
    memcpy(front, span[fi], sizeof front);
    memcpy(inset, span[ii], sizeof inset);
    memcpy(view, span[span_me], sizeof view);
    front_here = span_me == fi;
    inset_here = span_me == ii;
}

/* ---- --check: the physics, measured ---- */
static void read_ring(float *out, long n) {
    static float ring[RING];
    glBindFramebuffer(GL_FRAMEBUFFER, fRing);
    glReadPixels(0, 0, RING, 1, GL_RED, GL_FLOAT, ring);
    for (long s = 0; s < n; s++) out[s] = ring[(step_n - n + 1 + s + RING * 4L) % RING];
}

/* a bare vacuum domain of n x n interior cells */
static void bare_domain(int n, float nlv) {
    nl = nlv; cp = 1; pxl = nlv; rot90 = 0;
    NX = NY = n + 2 * PML;
    gx0 = gy0 = 0;
    scx = NX / 2.0f; scy = NY / 2.0f;
    nsh = nsrc = nlab = 0; te = 0; has_ntff = 0; np = 0;
}

/* the probe's record of a Gaussian-derivative pulse, nsteps long */
static void run_pulse(int n, float nlv, float px, float py, int nsteps, float *rec) {
    bare_domain(n, nlv);
    wave = W_RICKER; w_tau = 0.3f; w_t0 = 1.2f; w_fm = 1;
    add_point(0, 0, C_Z, 1, 0);
    probe[0] = px; probe[1] = py;
    grid_alloc();
    paint_materials();
    for (int i = 0; i < 2; i++) { clear_fbo(fFz[i]); clear_fbo(fFt[i]); }
    clear_fbo(fRing);
    step_n = 0;
    for (int s = 0; s < nsteps; s++) sim_step();
    read_ring(rec, nsteps);
}

static void check_pml(void) {
    int n = 120, M = 260, N = 1000;
    float nlv = 20;
    /* probes four cells inside the PML's inner face: head on, and near a corner */
    float d = (n / 2 - 4) / nlv;
    float P[2][2] = {{d, 0}, {d, d * 0.85f}};
    const char *nm[2] = {"normal incidence", "near a corner (~40°)"};
    float *a = malloc(N * sizeof *a), *r = malloc(N * sizeof *r);
    for (int k = 0; k < 2; k++) {
        run_pulse(n, nlv, P[k][0], P[k][1], N, a);
        run_pulse(n + 2 * M, nlv, P[k][0], P[k][1], N, r);
        float e = 0, m = 0;
        for (int s = 0; s < N; s++) { e = fmaxf(e, fabsf(a[s] - r[s])); m = fmaxf(m, fabsf(r[s])); }
        printf("  CPML reflection, %-22s %6.1f dB  (max |small - big domain| / max |incident|)\n", nm[k], 20 * log10f(e / m));
    }
    free(a); free(r);
}

static void read_pattern(float *pat) {
    glBindFramebuffer(GL_FRAMEBUFFER, fPat);
    glReadPixels(0, 0, 360, 1, GL_RED, GL_FLOAT, pat);
    float mx = 1e-20f;
    for (int i = 0; i < 360; i++) mx = fmaxf(mx, pat[i]);
    for (int i = 0; i < 360; i++) pat[i] /= mx;
}

static void run_act_for(int a, float periods) {
    opt_act = act_names[a];
    cur_k = -1;
    act_start(0);
    long want = (long)(periods * steps_period());
    while (step_n < want) sim_step();
    render(0, 10, 1.0f / 60);
}

static void check_dipole(void) {
    float pat[360];
    run_act_for(A_DIPOLE, 40);
    read_pattern(pat);
    double e_sin = 0, e_hw = 0, s_sin = 0;
    for (int i = 0; i < 360; i++) {
        float ph = (i + 0.5f) * DEG;
        float st = fabsf(cosf(ph)), ct = sinf(ph);        /* the dipole is along y */
        float hw = st < 1e-3f ? 0 : fabsf(cosf(PI / 2 * ct) / st);
        e_sin = fmax(e_sin, fabsf(pat[i] - st)); e_hw = fmax(e_hw, fabsf(pat[i] - hw));
        s_sin += (pat[i] - st) * (pat[i] - st);
    }
    printf("  dipole (nl %.1f cells/λ): |F(φ)| vs sin θ: rms %.3f, max %.3f; vs cos(π/2 cos θ)/sin θ: max %.3f\n",
           nl, sqrt(s_sin / 360), e_sin, e_hw);
    printf("    at θ = 90°, 60°, 30°, 10° from the axis: %.3f %.3f %.3f %.3f  (sin θ: 1 .866 .5 .174)\n",
           (pat[0] + pat[359]) / 2, pat[30], pat[60], pat[80]);
}

static void check_array(void) {
    float want[3] = {30, -20, 45};
    check_fixed = 1;
    for (int k = 0; k < 3; k++) {
        check_dphi = PI * sinf(want[k] * DEG);
        float pat[360];
        run_act_for(A_ARRAY, 30);
        read_pattern(pat);
        int bi = 0;
        for (int i = 0; i < 360; i++) {
            int ii = i < 180 ? i : i - 360;
            if (abs(ii) < 90 && pat[i] > pat[bi]) bi = i;
        }
        float a = pat[(bi + 359) % 360], b = pat[bi], c = pat[(bi + 1) % 360];
        float off = 0.5f * (a - c) / (a - 2 * b + c);
        float ang = bi + 0.5f + off;
        if (ang > 180) ang -= 360;
        printf("  phased array: Δφ = %6.2f°, sin θ = Δφ λ/(2π d) gives %6.2f°, the field steers to %6.2f°\n",
               check_dphi / DEG, want[k], ang);
    }
    check_fixed = 0;
}

static void check_yagi(void) {
    float pat[360];
    run_act_for(A_YAGI, 40);
    read_pattern(pat);
    int bi = 0;
    for (int i = 0; i < 360; i++) if (pat[i] > pat[bi]) bi = i;
    float fwd = (pat[0] + pat[359]) / 2, back = (pat[179] + pat[180]) / 2;
    printf("  yagi: front %.3f back %.3f F/B %.1f dB, peak at %d°\n", fwd, back, 20 * log10f(fwd / fmaxf(back, 1e-6f)), bi);
}

static void check_cavity(void) {
    check_fixed = 1;
    int N = RING;
    opt_act = act_names[A_CAVITY];
    cur_k = -1;
    act_start(0);
    long pulse = (long)(3 * steps_period());
    while (step_n < pulse + N) sim_step();
    float *rec = malloc(N * sizeof *rec);
    read_ring(rec, N);
    /* the cavity's true size in cells: Ez is free between the PEC sites */
    int cx = 0, cy = 0;
    for (int i = -400; i < 400; i++) {
        float lx, ly, ci, cj;
        local_to_cell(0, 0, &ci, &cj);
        cell_to_local(roundf(ci) + i, roundf(cj), &lx, &ly);
        if (fabsf(lx) < CAV_A / 2 + 0.05f && shape_sd(&sh[0], lx, ly) * nl >= 0) cx++;
        cell_to_local(roundf(ci), roundf(cj) + i, &lx, &ly);
        if (fabsf(ly) < CAV_B / 2 + 0.05f && shape_sd(&sh[0], lx, ly) * nl >= 0) cy++;
    }
    float A = (cx + 1) / nl, B = (cy + 1) / nl;     /* wavelengths */
    printf("  cavity: %d x %d free cells (nl %.1f): %.4fλ x %.4fλ\n", cx, cy, nl, A, B);
    /* the spectrum, Hann-windowed, finely */
    float spp = steps_period();
    for (int m = 1; m <= 4; m++)
        for (int nn = 1; nn <= 3; nn++) {
            float fa = 0.5f * sqrtf((m / A) * (m / A) + (nn / B) * (nn / B));
            if (fa < 0.5f || fa > 1.25f) continue;
            /* only the modes the source and the probe both couple to */
            float xs = src[0].x0 + A / 2, ys = src[0].y0 + B / 2, xp = probe[0] + A / 2, yp = probe[1] + B / 2;
            float cpl = fabsf(sinf(m * PI * xs / A) * sinf(nn * PI * ys / B) * sinf(m * PI * xp / A) * sinf(nn * PI * yp / B));
            if (cpl < 0.15f) continue;
            /* with the Yee grid's dispersion */
            float kx = m * PI / (A * nl), ky = nn * PI / (B * nl);
            float fn = asinf(S_COURANT * sqrtf(sinf(kx / 2) * sinf(kx / 2) + sinf(ky / 2) * sinf(ky / 2))) / PI * spp;
            float best = 0, bf = 0;
            for (float f = fa * 0.97f; f < fa * 1.03f; f += 0.00005f) {
                double re = 0, im = 0, w = 2 * PI * f / spp;
                for (int s = 0; s < N; s++) {
                    double h = 0.5 - 0.5 * cos(2 * PI * s / (N - 1));
                    re += rec[s] * h * cos(w * s); im += rec[s] * h * sin(w * s);
                }
                float pw = (float)(re * re + im * im);
                if (pw > best) { best = pw; bf = f; }
            }
            printf("    TM%d%d: analytic %.4f f0, Yee-dispersion %.4f f0, measured peak %.4f f0 (%+.2f%% / %+.2f%%)\n",
                   m, nn, fa, fn, bf, (bf / fa - 1) * 100, (bf / fn - 1) * 100);
        }
    free(rec);
    check_fixed = 0;
}

static void check_q(void) {
    opt_act = act_names[A_CAVITY];
    cur_k = -1;
    act_start(0);
    long want = (long)((w_t0 + w_rep - 0.5f) * steps_period());   /* just before the next burst */
    while (step_n < want) sim_step();
    inset_here = 1;
    estimate_q();
    printf("  cavity ring-down: Q ≈ %.0f (from the probe's envelope between bursts)\n", q_est);
}

static void run_checks(void) {
    if (getenv("FDTD_YAGI")) { check_yagi(); return; }
    printf("fdtd: checks (%s)\n", (const char *)glGetString(GL_RENDERER));
    check_pml();
    check_dipole();
    check_array();
    check_yagi();
    check_cavity();
    check_q();
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
        need_restart = 1;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "fdtd");
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
        "usage: fdtd [--window | --root | --layer] [--seed N] [--fps N] [--palette field|phosphor|plasma|paper]\n"
        "            [--act NAME] [--act-secs S] [--speed X] [--cells N] [--poynting]\n"
        "            [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "            [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check]\n"
        "acts: dipole array yagi horn dish lens luneburg bend crystal slits tir cavity\n");
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
        else if (!strcmp(a, "-poynting")) opt_poynting = 1;
        else if (!strcmp(a, "-check")) opt_check = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-act") && i + 1 < argc) opt_act = argv[++i];
        else if (!strcmp(a, "-act-secs") && i + 1 < argc) opt_act_secs = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-cells") && i + 1 < argc) opt_cells = atoi(argv[++i]);
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
    if (opt_fps < 1) opt_fps = 1;
    if (opt_act_secs < 8) opt_act_secs = 8;
    if (opt_speed <= 0) opt_speed = 1;
    if (opt_cells < 20000) opt_cells = 20000;
    if (opt_act) {
        int ok = 0;
        for (int a = 0; a < NACT; a++) ok |= !strcmp(opt_act, act_names[a]);
        if (!ok) usage();
    }
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    /* --palette NAME, else from the seed */
    pal = (int)(hash32(opt_seed * 7919u + 3) % 4u);
    pal = pal == 3 ? 0 : pal;          /* paper is for asking for */
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) usage();
    }
    int offscreen = opt_shot || opt_bench > 0 || opt_check;
    fonts_init();

    double t_wall0 = opt_epoch ? opt_epoch : wall();
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
        XStoreName(dpy, win, "FDTD");
        XClassHint chint = {"fdtd", "Fdtd"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();
    place();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (opt_check) { run_checks(); return 0; }
        /* the show at --at seconds (into the act, with --act; else since the epoch) */
        double t = opt_at;
        if (opt_epoch && !opt_act) t = opt_at;
        if (opt_shot) {
            /* the last few seconds frame by frame, so the averages settle */
            for (double f = fmax(0, t - 4); f < t; f += 1.0 / 60) {
                double ta = show_advance(f, 1 << 30);
                render(fbo, ta, 1.0f / 60);
            }
            double ta = show_advance(t, 1 << 30);
            render(fbo, ta, 1.0f / 60);
            glFinish();
            if (getenv("FDTD_DEBUG")) {
                float e;
                glBindFramebuffer(GL_FRAMEBUFFER, fExp[ce]);
                glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, &e);
                float *z = malloc((size_t)NX * NY * 16);
                glBindFramebuffer(GL_FRAMEBUFFER, fAcc[ca]);
                glReadPixels(0, 0, NX, NY, GL_RGBA, GL_FLOAT, z);
                double su = 0, mz = 0;
                for (long i = 0; i < (long)NX * NY; i++) { su += z[i * 4]; mz = fmax(mz, fabs(z[i * 4 + 3])); }
                fprintf(stderr, "exp %g mean u %g max|Fz| %g\n", e, su / NX / NY, mz);
                free(z);
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            }
            save_ppm(opt_shot);
            printf("fdtd: %s, %s, grid %dx%d (%.0fk cells), %.1f cells/λ, %.1f px/cell, %ld steps\n",
                   act_names[act], palettes[pal].name, NX, NY, NX * NY / 1e3, nl, cp * W / view[2], step_n);
            return 0;
        }
        double ta = show_advance(t, 1 << 30);
        render(fbo, ta, 1.0f / 60);
        glFinish();
        double sum = 0, worst = 0;
        long steps0 = step_n;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            prof_mark(PR_SIM);
            ta = show_advance(t + (i + 1) / 60.0, 64);
            render(fbo, ta, 1.0f / 60);
            glFinish();
            double ms = (now() - f0) * 1000;
            prof_collect();
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("fdtd: %s %dx%d, grid %dx%d (%.0fk cells, %.1f cells/λ), %.2f steps/frame\n", act_names[act], W, H,
               NX, NY, NX * NY / 1e3, nl, (double)(step_n - steps0) / opt_bench);
        printf("fdtd: %d frames: avg %.3f ms wall (submit to finish), worst %.3f ms\n", opt_bench, sum / opt_bench, worst);
        double gpu = 0;
        for (int i = 0; i < PR_N; i++) gpu += prof_ms[i];
        printf("fdtd: GPU %.3f ms/frame:", gpu / opt_bench);
        for (int i = 0; i < PR_N; i++) printf(" %s %.3f", prof_name[i], prof_ms[i] / opt_bench);
        printf("\nfdtd: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now(), last = t0;
    for (;;) {
        if (opt_layer && !wl_poll()) return 0;
        while (dpy && XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win, ev.xconfigure.width, ev.xconfigure.height);
                else if (ev.xconfigure.width != W || ev.xconfigure.height != H) {
                    W = ev.xconfigure.width; H = ev.xconfigure.height;
                    need_restart = 1;
                }
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        if (need_restart) place();
        double t = now(), wt = wall();
        double el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        double ta = show_advance(el, 48);
        render(0, ta, (float)fmin(t - last, 0.1));
        last = t;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
