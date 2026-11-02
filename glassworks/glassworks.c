/*
 * glassworks — lava and glass: a glassblower's hot shop at night.
 *
 * A dark workshop lit by its fires. The glory hole roars orange-white at one
 * end of the bench, the key light, throwing long shadows and shafts through
 * the haze; the main furnace glows through its gathering port; moonlight
 * falls through a high mullioned window. Between them the gaffer, a
 * silhouette in a leather apron, makes one piece after another:
 *
 *   gather     a gob of molten glass taken from the furnace on a blowpipe
 *   marver     rolled on the polished steel table into a cylinder
 *   blow       a first bubble, through a blow hose, at the bench
 *   reheat     back into the glory hole, white-hot again
 *   shape      blown out and necked with the jacks, the pipe rolling on the
 *              bench arms all the while
 *   punty      transferred onto a punty, the open end now facing out
 *   open       the lip opened and flared with the jacks
 *   cool       the glow fading from yellow through orange and cherry red
 *              until it is glass: clear, cobalt, amber, green, smoke, a
 *              latticino twist or millefiori, refracting the fires
 *   anneal     carried to the annealer; a new piece begins
 *
 * Each piece takes four minutes and has its own form (vase, bowl, goblet,
 * bottle, trumpet, tumbler) and glass. Finished pieces join a gallery on the
 * shelves along the wall, which grows through the session, their caustics
 * thrown onto the boards by the glory hole.
 *
 * Hot glass is drawn as a blackbody by its temperature (thin parts cool
 * first); cool glass as a dielectric: screen-space refraction with a hint of
 * dispersion, Beer-Lambert tint, Fresnel reflections of the fires and the
 * window. The glory hole's light is shadow-mapped and scatters in drifting
 * haze (ray-marched at quarter resolution), the moonlight shafts through the
 * window's mullions, heat shimmers above the fires and the glass, embers
 * drift up, and a director cuts between macro shots of the turning gather and
 * wide shots of the shop with a physically-based depth of field, bloom and a
 * filmic grade. The roar of the fires follows the CPU load.
 *
 * Everything is a deterministic function of the time since --epoch and the
 * --seed, so instances on several outputs agree without talking.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout gets the director's cut (and the
 * caption); the others, left to right, take turns at
 *   GALLERY   slow moves along the shelves of finished pieces
 *   FIRES     the glory hole, furnace and bench, from the far side
 * all in the same shop at the same moment.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size (GW_PROF=1 in the environment
 * breaks a frame down by pass, from GPU timestamps). The 3-D work renders at
 * an internal resolution of about a megapixel, 4x multisampled (--scale F,
 * --msaa N to change), and is upscaled under the depth of field, bloom and
 * grain. --no-caption drops the time; --role front|gallery|fires and --cam N
 * (a director's shot kind) pin a view, for testing.
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

#define PI 3.14159265f

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "EB Garamond";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 25;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the shop opened at */
static float opt_scale;                /* internal render scale; 0 = automatic */
static int opt_caption = 1;
static int opt_role = -1;              /* --role: force a role (testing) */
static int opt_shotlock = -1;          /* --cam: force a director shot kind (testing) */

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { R_FRONT, R_GALLERY, R_FIRES };
static int role = R_FRONT;

static void die(const char *what) { fprintf(stderr, "glassworks: %s\n", what); exit(1); }

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
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float sstep(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float ease(float t) { t = clampf(t, 0, 1); return t * t * (3 - 2 * t); }

/* stateless hashing: everything that varies is a function of (seed, salt, i) */
static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu ^ opt_seed * 0x9e3779b9u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; h *= 0x27d4eb2du; h ^= h >> 16;
    return h;
}
static float hf(uint32_t a, uint32_t b, uint32_t c) { return (hash3(a, b, c) & 0xffffff) / 16777216.0f; }

/* smooth 1-D value noise in t */
static float noise1(float t, uint32_t salt) {
    float f = floorf(t), u = t - f;
    int32_t i = (int32_t)f;
    float a = hf((uint32_t)i, salt, 77), b = hf((uint32_t)(i + 1), salt, 77);
    u = u * u * (3 - 2 * u);
    return a + (b - a) * u;
}

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* ---- vectors ---- */
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { return (V3){x, y, z}; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vmul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a) { float l = vlen(a); return l > 1e-9f ? vmul(a, 1 / l) : v3(0, 1, 0); }
static V3 vlerp(V3 a, V3 b, float t) { return vadd(a, vmul(vsub(b, a), t)); }
/* any two unit vectors perpendicular to unit d */
static void basis(V3 d, V3 *u, V3 *v) {
    V3 a = fabsf(d.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0);
    *u = vnorm(vcross(a, d));
    *v = vcross(d, *u);
}

/* column-major 4x4 */
static void mat_mul(float *o, const float *a, const float *b) {
    float t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            t[c * 4 + r] = 0;
            for (int k = 0; k < 4; k++) t[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
        }
    memcpy(o, t, sizeof t);
}
static void look_at(float *m, V3 eye, V3 at, V3 up) {
    V3 f = vnorm(vsub(at, eye)), s = vnorm(vcross(f, up)), u = vcross(s, f);
    float r[16] = {s.x, u.x, -f.x, 0, s.y, u.y, -f.y, 0, s.z, u.z, -f.z, 0,
                   -vdot(s, eye), -vdot(u, eye), vdot(f, eye), 1};
    memcpy(m, r, sizeof r);
}
/* off-centre perspective from tangents of the frustum's sides */
static void frustum(float *m, float l, float r, float b, float t, float n, float f) {
    float p[16] = {0};
    p[0] = 2 * n / (r - l); p[5] = 2 * n / (t - b);
    p[8] = (r + l) / (r - l); p[9] = (t + b) / (t - b);
    p[10] = -(f + n) / (f - n); p[11] = -1;
    p[14] = -2 * f * n / (f - n);
    memcpy(m, p, sizeof p);
}
static int mat_inv(float *o, const float *m) {
    float inv[16];
    inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (fabsf(det) < 1e-20f) return 0;
    for (int i = 0; i < 16; i++) o[i] = inv[i] / det;
    return 1;
}
static V3 mat_apply(const float *m, V3 p, float *w) {
    float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    float ww = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (w) *w = ww;
    return v3(x, y, z);
}

/* ---- sensors: the lock time, and the CPU load that feeds the fires ---- */
static time_t locked_at;
static float cpu_load, cpu_load_s;     /* latest reading, and smoothed */

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

static void read_cpu(void) {
    static unsigned long long pb, pt;
    char line[256];
    slurp_line("/proc/stat", line, sizeof line);
    unsigned long long v[8] = {0}, t = 0;
    sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    for (int i = 0; i < 8; i++) t += v[i];
    unsigned long long b = t - v[3] - v[4];
    if (pt && t > pt) cpu_load = (float)(b - pb) / (float)(t - pt);
    pb = b; pt = t;
}

/* ---- glyph atlas: printable ASCII in each text style used ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 8
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* regular, italic */
#define AW 1024
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < 2; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:slant=%s:weight=regular:fontformat=CFF,TrueType", opt_font, i ? "italic" : "roman");
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

static void build_atlas(void) {
    free(atlas);
    int cap = 1024;
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

/* text quads: x, y, u, v, r, g, b, a per vertex */
static float *tv;
static int tvn, tvcap;

static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 8192) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x */
static void text(int s, float x, float y, int align, const char *str, const float *col, float track) {
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
}

/* ---- meshes: unindexed triangles of pos, normal, uv, material, seed ---- */
#define VF 10
typedef struct { float *v; int n, cap; } Mesh;
static float cur_seed;                 /* per-object variation, stamped on each vertex */

enum {
    M_FLOOR, M_BRICK, M_STEEL, M_POLISH, M_WOOD, M_WALL, M_CLOTH, M_SKIN, M_LEATHER, M_PIPE,
    M_FURNACE, M_GLORY, M_WINDOW, M_HOSE, M_CEIL, M_FIBER, M_PLANK, M_HAIR, M_ANNEAL, M_LAMP, M_DOORHOT
};

static void vtx(Mesh *m, V3 p, V3 n, float u, float v, int mat) {
    if (m->n + 1 > m->cap) m->v = realloc(m->v, (size_t)(m->cap = m->cap ? m->cap * 2 : 65536) * VF * sizeof(float));
    float *o = m->v + (size_t)m->n * VF;
    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z;
    o[6] = u; o[7] = v; o[8] = (float)mat; o[9] = cur_seed;
    m->n++;
}
static void tri(Mesh *m, V3 a, V3 b, V3 c, V3 na, V3 nb, V3 nc, float ua, float va, float ub, float vb,
                float uc, float vc, int mat) {
    vtx(m, a, na, ua, va, mat); vtx(m, b, nb, ub, vb, mat); vtx(m, c, nc, uc, vc, mat);
}
/* a flat quad p, p+u, p+u+v, p+v facing u x v; uv in metres */
static void quad(Mesh *m, V3 p, V3 u, V3 v, int mat) {
    V3 n = vnorm(vcross(u, v));
    float lu = vlen(u), lv = vlen(v);
    V3 b = vadd(p, u), c = vadd(b, v), d = vadd(p, v);
    tri(m, p, b, c, n, n, n, 0, 0, lu, 0, lu, lv, mat);
    tri(m, p, c, d, n, n, n, 0, 0, lu, lv, 0, lv, mat);
}
/* an oriented box: centre, unit axes, half extents */
static void obox(Mesh *m, V3 c, V3 ax, V3 ay, V3 az, float hx, float hy, float hz, int mat) {
    V3 X = vmul(ax, hx), Y = vmul(ay, hy), Z = vmul(az, hz);
    V3 o = vsub(vsub(vsub(c, X), Y), Z);
    V3 X2 = vmul(X, 2), Y2 = vmul(Y, 2), Z2 = vmul(Z, 2);
    quad(m, vadd(o, Z2), X2, Y2, mat);                    /* +z */
    quad(m, vadd(o, X2), vmul(X2, -1), Y2, mat);          /* -z */
    quad(m, vadd(o, X2), Z2, Y2, mat);                    /* +x */
    quad(m, o, Y2, Z2, mat);                              /* -x */
    quad(m, vadd(o, Y2), Z2, X2, mat);                    /* +y */
    quad(m, o, X2, Z2, mat);                              /* -y */
}
static void box(Mesh *m, float x0, float y0, float z0, float x1, float y1, float z1, int mat) {
    obox(m, v3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), v3(1, 0, 0), v3(0, 1, 0), v3(0, 0, 1),
         (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, mat);
}
/* a box from a to b with a square section of half-width r (beams, blades) */
static void beam(Mesh *m, V3 a, V3 b, float rw, float rh, V3 up, int mat) {
    V3 d = vsub(b, a);
    float l = vlen(d);
    V3 az = vmul(d, 1 / l), ax = vnorm(vcross(up, az)), ay = vcross(az, ax);
    obox(m, vlerp(a, b, 0.5f), ax, ay, az, rw, rh, l / 2, mat);
}
/* a tapered capsule from a (radius ra) to b (rb); inward = 0 */
static void capsule(Mesh *m, V3 a, V3 b, float ra, float rb, int seg, int mat) {
    V3 d = vsub(b, a);
    float l = vlen(d);
    if (l < 1e-6f) { d = v3(0, 1, 0); l = 1e-6f; }
    V3 ax = vmul(d, 1 / l), u, v;
    basis(ax, &u, &v);
    /* the cone's side tilts its normals by the taper */
    float sa = clampf((ra - rb) / l, -0.99f, 0.99f), ca = sqrtf(1 - sa * sa);
    int rings = seg / 2;
    /* profile: hemisphere at a (from the pole), the side, hemisphere at b */
    int np = 2 * rings + 2;
    V3 P[64]; float R[64], NA[64];         /* centre, radius, normal angle */
    int k = 0;
    float phi0 = asinf(sa);
    for (int i = 0; i <= rings; i++) {       /* a cap: angle from -pi/2 to phi0 */
        float t = -PI / 2 + (phi0 + PI / 2) * i / rings;
        P[k] = a; R[k] = ra; NA[k] = t; k++;
    }
    for (int i = 0; i <= rings; i++) {       /* b cap: phi0 to pi/2 */
        float t = phi0 + (PI / 2 - phi0) * i / rings;
        P[k] = b; R[k] = rb; NA[k] = t; k++;
    }
    (void)np; (void)ca;
    for (int i = 0; i + 1 < k; i++) {
        for (int j = 0; j < seg; j++) {
            float a0 = 2 * PI * j / seg, a1 = 2 * PI * (j + 1) / seg;
            V3 q[4], n[4];
            for (int c = 0; c < 4; c++) {
                int ii = i + (c >= 2), jj = c == 1 || c == 2;
                float ang = jj ? a1 : a0;
                V3 rad = vadd(vmul(u, cosf(ang)), vmul(v, sinf(ang)));
                V3 nn = vadd(vmul(rad, cosf(NA[ii])), vmul(ax, sinf(NA[ii])));
                n[c] = nn;
                q[c] = vadd(P[ii], vmul(nn, R[ii]));
            }
            float s0 = (float)i / k, s1 = (float)(i + 1) / k;
            tri(m, q[0], q[2], q[1], n[0], n[2], n[1], a0, s0, a1, s1, a1, s0, mat);
            tri(m, q[0], q[3], q[2], n[0], n[3], n[2], a0, s0, a0, s1, a1, s1, mat);
        }
    }
}
/* an open cylinder from a to b; inward flips it to face its axis */
static void cyl(Mesh *m, V3 a, V3 b, float r, int seg, int inward, int mat) {
    V3 d = vsub(b, a);
    float l = vlen(d);
    V3 ax = vmul(d, 1 / l), u, v;
    basis(ax, &u, &v);
    for (int j = 0; j < seg; j++) {
        float a0 = 2 * PI * j / seg, a1 = 2 * PI * (j + 1) / seg;
        V3 n0 = vadd(vmul(u, cosf(a0)), vmul(v, sinf(a0))), n1 = vadd(vmul(u, cosf(a1)), vmul(v, sinf(a1)));
        V3 p0 = vadd(a, vmul(n0, r)), p1 = vadd(a, vmul(n1, r)), p2 = vadd(p1, d), p3 = vadd(p0, d);
        float u0 = a0 * r, u1 = a1 * r;
        if (inward) {
            V3 m0 = vmul(n0, -1), m1 = vmul(n1, -1);
            tri(m, p0, p2, p1, m0, m1, m1, u0, 0, u1, l, u1, 0, mat);
            tri(m, p0, p3, p2, m0, m0, m1, u0, 0, u0, l, u1, l, mat);
        } else {
            tri(m, p0, p1, p2, n0, n1, n1, u0, 0, u1, 0, u1, l, mat);
            tri(m, p0, p2, p3, n0, n1, n0, u0, 0, u1, l, u0, l, mat);
        }
    }
}
/* an annulus (or disc, r0 = 0) at c facing n */
static void disc(Mesh *m, V3 c, V3 n, float r0, float r1, int seg, int mat) {
    V3 u, v;
    basis(n, &u, &v);
    for (int j = 0; j < seg; j++) {
        float a0 = 2 * PI * j / seg, a1 = 2 * PI * (j + 1) / seg;
        V3 d0 = vadd(vmul(u, cosf(a0)), vmul(v, sinf(a0))), d1 = vadd(vmul(u, cosf(a1)), vmul(v, sinf(a1)));
        V3 p0 = vadd(c, vmul(d0, r0)), p1 = vadd(c, vmul(d1, r0)), p2 = vadd(c, vmul(d1, r1)), p3 = vadd(c, vmul(d0, r1));
        tri(m, p0, p3, p2, n, n, n, d0.x * r0, d0.y * r0, d0.x * r1, d0.y * r1, d1.x * r1, d1.y * r1, mat);
        tri(m, p0, p2, p1, n, n, n, d0.x * r0, d0.y * r0, d1.x * r1, d1.y * r1, d1.x * r0, d1.y * r0, mat);
    }
}
/* a closed cylinder */
static void rod(Mesh *m, V3 a, V3 b, float r, int seg, int mat) {
    cyl(m, a, b, r, seg, 0, mat);
    V3 d = vnorm(vsub(b, a));
    disc(m, b, d, 0, r, seg, mat);
    disc(m, a, vmul(d, -1), 0, r, seg, mat);
}
/* an ellipsoid at c with semi-axes along ax, ay, az */
static void ellipsoid(Mesh *m, V3 c, V3 ax, V3 ay, V3 az, float rx, float ry, float rz, int seg, int mat) {
    int rings = seg / 2;
    for (int i = 0; i < rings; i++)
        for (int j = 0; j < seg; j++) {
            V3 q[4], n[4];
            for (int k = 0; k < 4; k++) {
                int ii = i + (k >= 2), jj = j + (k == 1 || k == 2);
                float th = PI * ii / rings - PI / 2, ph = 2 * PI * jj / seg;
                float cx = cosf(th) * cosf(ph), cy = sinf(th), cz = cosf(th) * sinf(ph);
                q[k] = vadd(c, vadd(vadd(vmul(ax, cx * rx), vmul(ay, cy * ry)), vmul(az, cz * rz)));
                n[k] = vnorm(vadd(vadd(vmul(ax, cx / rx), vmul(ay, cy / ry)), vmul(az, cz / rz)));
            }
            tri(m, q[0], q[1], q[2], n[0], n[1], n[2], 0, 0, 0, 0, 0, 0, mat);
            tri(m, q[0], q[2], q[3], n[0], n[2], n[3], 0, 0, 0, 0, 0, 0, mat);
        }
}

/* ---- the shop: fixed geometry, in metres, y up; the bench faces +z ---- */
/* the glory hole: a drum on its side, mouth facing +x towards the bench */
static const V3 GH_C = {-2.45f, 1.05f, -1.15f};
#define GH_R 0.52f
#define GH_LEN 0.95f
#define GH_MOUTH 0.18f
static V3 gh_mouth(void) { return v3(GH_C.x + GH_LEN / 2, GH_C.y, GH_C.z); }
/* the furnace's gathering port, in its front face (z = FUR_Z) */
#define FUR_Z (-3.4f)
static const V3 PORT_C = {-0.7f, 1.05f, -3.4f};
#define PORT_W 0.40f
#define PORT_H 0.30f
/* the bench: the gaffer sits facing +z between two arms, the pipe across them */
static const V3 SEAT = {0.62f, 0.5f, -1.2f};
#define RAIL_Y 0.80f
#define RAIL_XL 0.22f                  /* the arm on the gaffer's right (-x) */
#define RAIL_XR 1.02f
#define PIPE_Z (-0.80f)                /* where the pipe lies across the arms */
/* the marver */
#define MV_X0 1.55f
#define MV_X1 2.45f
#define MV_Z0 (-2.75f)
#define MV_Z1 (-2.15f)
#define MV_Y 0.90f
/* the annealer, its door in its front face */
#define AN_X0 3.1f
#define AN_X1 4.5f
#define AN_Z0 (-3.95f)
#define AN_Z1 (-2.95f)
#define AN_H 1.3f
#define AN_DX0 3.4f
#define AN_DX1 4.2f
#define AN_DY0 0.55f
#define AN_DY1 1.05f
/* the window: tall, in the right-hand wall that the glory hole faces */
#define ROOM_X 6.0f
#define WIN_X (ROOM_X - 0.02f)
#define WIN_Y0 1.0f
#define WIN_Y1 3.45f
#define WIN_Z0 (-2.7f)
#define WIN_Z1 (0.9f)
/* the gallery: shelves across the window, backlit by the moon */
#define SH_X (ROOM_X - 0.03f)
#define SH_D 0.30f
#define SH_Z0 WIN_Z0
#define SH_Z1 WIN_Z1
#define SH_N 4
#define SH_SLOTS 10
static float shelf_y(int i) { return WIN_Y0 + 0.02f + 0.58f * i; }
#define ROOM_Z0 (-5.0f)
#define ROOM_Z1 6.0f
#define ROOM_H 4.6f

static Mesh shop;

static void build_shop(void) {
    Mesh *m = &shop;
    m->n = 0;
    cur_seed = 0;
    /* room */
    quad(m, v3(-ROOM_X, 0, ROOM_Z1), v3(2 * ROOM_X, 0, 0), v3(0, 0, ROOM_Z0 - ROOM_Z1), M_FLOOR);
    quad(m, v3(-ROOM_X, 0, ROOM_Z0), v3(2 * ROOM_X, 0, 0), v3(0, ROOM_H, 0), M_WALL);
    cur_seed = 1;
    quad(m, v3(ROOM_X, 0, ROOM_Z0), v3(0, 0, ROOM_Z1 - ROOM_Z0), v3(0, ROOM_H, 0), M_WALL);
    cur_seed = 2;
    quad(m, v3(-ROOM_X, 0, ROOM_Z1), v3(0, 0, ROOM_Z0 - ROOM_Z1), v3(0, ROOM_H, 0), M_WALL);
    cur_seed = 3;
    quad(m, v3(ROOM_X, 0, ROOM_Z1), v3(-2 * ROOM_X, 0, 0), v3(0, ROOM_H, 0), M_WALL);
    cur_seed = 0;
    quad(m, v3(-ROOM_X, ROOM_H, ROOM_Z0), v3(2 * ROOM_X, 0, 0), v3(0, 0, ROOM_Z1 - ROOM_Z0), M_CEIL);
    /* the window: moonlit panes, a deep reveal, and a sill */
    quad(m, v3(WIN_X, WIN_Y0, WIN_Z0), v3(0, 0, WIN_Z1 - WIN_Z0), v3(0, WIN_Y1 - WIN_Y0, 0), M_WINDOW);
    box(m, ROOM_X - 0.36f, WIN_Y0 - 0.12f, WIN_Z0 - 0.1f, ROOM_X, WIN_Y0 - 0.0f, WIN_Z1 + 0.1f, M_WALL);
    box(m, ROOM_X - 0.18f, WIN_Y1, WIN_Z0 - 0.1f, ROOM_X, WIN_Y1 + 0.12f, WIN_Z1 + 0.1f, M_WALL);
    box(m, ROOM_X - 0.18f, WIN_Y0, WIN_Z0 - 0.12f, ROOM_X, WIN_Y1, WIN_Z0, M_WALL);
    box(m, ROOM_X - 0.18f, WIN_Y0, WIN_Z1, ROOM_X, WIN_Y1, WIN_Z1 + 0.12f, M_WALL);
    /* roof trusses and purlins, black against the haze */
    for (int i = 0; i < 6; i++) {
        float z = ROOM_Z0 + 1.0f + i * 1.9f;
        cur_seed = 10 + i;
        box(m, -ROOM_X, 4.0f, z - 0.07f, ROOM_X, 4.22f, z + 0.07f, M_STEEL);
        for (int k = -5; k <= 5; k++) {
            float x = k * 1.15f;
            beam(m, v3(x, 4.2f, z), v3(x + 0.55f, ROOM_H, z), 0.035f, 0.035f, v3(0, 0, 1), M_STEEL);
            beam(m, v3(x, 4.2f, z), v3(x - 0.55f, ROOM_H, z), 0.035f, 0.035f, v3(0, 0, 1), M_STEEL);
        }
    }
    for (int k = -2; k <= 2; k++) box(m, k * 2.4f - 0.06f, 4.22f, ROOM_Z0, k * 2.4f + 0.06f, 4.36f, ROOM_Z1, M_STEEL);

    /* the furnace: a brick block with a steel frame, the port in its face */
    cur_seed = 40;
    float fx0 = -1.75f, fx1 = 0.35f, fy1 = 1.85f, fz0 = -4.95f;
    float px0 = PORT_C.x - PORT_W / 2, px1 = PORT_C.x + PORT_W / 2, py0 = PORT_C.y - PORT_H / 2, py1 = PORT_C.y + PORT_H / 2;
    box(m, fx0, 0, fz0, fx1, fy1, FUR_Z - 0.35f, M_STEEL);           /* the body behind the face */
    box(m, fx0, 0, FUR_Z - 0.35f, px0 - 0.3f, fy1, FUR_Z, M_STEEL);  /* steel cladding... */
    box(m, px1 + 0.3f, 0, FUR_Z - 0.35f, fx1, fy1, FUR_Z, M_STEEL);
    box(m, px0 - 0.3f, 0, FUR_Z - 0.35f, px1 + 0.3f, py0 - 0.25f, FUR_Z, M_STEEL);
    box(m, px0 - 0.3f, py1 + 0.3f, FUR_Z - 0.35f, px1 + 0.3f, fy1, FUR_Z, M_STEEL);
    cur_seed = 42;                                                   /* ...round a brick face */
    box(m, px0 - 0.3f, py0 - 0.25f, FUR_Z - 0.33f, px0, py1 + 0.3f, FUR_Z - 0.02f, M_BRICK);
    box(m, px1, py0 - 0.25f, FUR_Z - 0.33f, px1 + 0.3f, py1 + 0.3f, FUR_Z - 0.02f, M_BRICK);
    box(m, px0, py0 - 0.25f, FUR_Z - 0.33f, px1, py0, FUR_Z - 0.02f, M_BRICK);
    box(m, px0, py1, FUR_Z - 0.33f, px1, py1 + 0.3f, FUR_Z - 0.02f, M_BRICK);
    cur_seed = 40;
    /* the port's throat and the glowing pot beyond */
    quad(m, v3(px0, py0, FUR_Z), v3(PORT_W, 0, 0), v3(0, 0, -0.35f), M_FURNACE);
    quad(m, v3(px1, py1, FUR_Z), v3(-PORT_W, 0, 0), v3(0, 0, -0.35f), M_FURNACE);
    quad(m, v3(px0, py1, FUR_Z), v3(0, -PORT_H, 0), v3(0, 0, -0.35f), M_FURNACE);
    quad(m, v3(px1, py0, FUR_Z), v3(0, PORT_H, 0), v3(0, 0, -0.35f), M_FURNACE);
    quad(m, v3(px0, py0, FUR_Z - 0.35f), v3(PORT_W, 0, 0), v3(0, PORT_H, 0), M_FURNACE);
    /* steel angle at the corners and bands */
    for (int i = 0; i < 2; i++) {
        float x = i ? fx1 : fx0;
        box(m, x - 0.04f, 0, FUR_Z - 0.04f, x + 0.04f, fy1 + 0.02f, FUR_Z + 0.04f, M_STEEL);
    }
    box(m, fx0 - 0.04f, fy1, FUR_Z - 0.04f, fx1 + 0.04f, fy1 + 0.08f, FUR_Z + 0.04f, M_STEEL);
    box(m, fx0 - 0.04f, 0.35f, FUR_Z - 0.03f, fx1 + 0.04f, 0.43f, FUR_Z + 0.03f, M_STEEL);
    /* the port's sliding door, raised, in its guides */
    cur_seed = 41;
    box(m, px0 - 0.09f, py1 - 0.04f, FUR_Z + 0.03f, px1 + 0.09f, py1 + 0.42f, FUR_Z + 0.07f, M_DOORHOT);
    box(m, px0 - 0.14f, py0 - 0.05f, FUR_Z, px0 - 0.10f, py1 + 0.75f, FUR_Z + 0.09f, M_STEEL);
    box(m, px1 + 0.10f, py0 - 0.05f, FUR_Z, px1 + 0.14f, py1 + 0.75f, FUR_Z + 0.09f, M_STEEL);
    /* a ledge under the port, and the flue */
    box(m, px0 - 0.15f, py0 - 0.07f, FUR_Z, px1 + 0.15f, py0, FUR_Z + 0.22f, M_STEEL);
    cyl(m, v3(-0.9f, fy1, -4.3f), v3(-0.9f, ROOM_H, -4.3f), 0.2f, 20, 0, M_STEEL);

    /* the glory hole: the drum, its face ring and the hot throat */
    cur_seed = 50;
    V3 gm = gh_mouth(), gb = v3(GH_C.x - GH_LEN / 2, GH_C.y, GH_C.z), X = v3(1, 0, 0);
    cyl(m, gb, gm, GH_R, 48, 0, M_STEEL);
    disc(m, gm, X, GH_MOUTH, GH_MOUTH + 0.11f, 48, M_FIBER);
    disc(m, vadd(gm, v3(-0.005f, 0, 0)), X, GH_MOUTH + 0.11f, GH_R + 0.02f, 48, M_STEEL);
    disc(m, gb, vmul(X, -1), 0, GH_R, 48, M_STEEL);
    cyl(m, v3(gm.x - 0.75f, gm.y, gm.z), gm, GH_MOUTH, 32, 1, M_GLORY);
    disc(m, v3(gm.x - 0.75f, gm.y, gm.z), X, 0, GH_MOUTH, 32, M_GLORY);
    /* bands round the drum, the burner at the back */
    for (int i = 0; i < 3; i++) {
        float x = gb.x + 0.08f + i * 0.4f;
        cyl(m, v3(x, gm.y, gm.z), v3(x + 0.05f, gm.y, gm.z), GH_R + 0.015f, 48, 0, M_STEEL);
    }
    rod(m, vsub(gb, v3(0.45f, 0, 0)), gb, 0.07f, 16, M_STEEL);
    rod(m, vsub(gb, v3(0.3f, 0, 0)), vsub(gb, v3(0.3f, 1.05f, 0)), 0.035f, 12, M_STEEL);
    /* the two half-moon doors, swung open */
    /* (the near one swung wide, the far one half shut, so the light leaves in a fan) */
    for (int s = -1; s <= 1; s += 2) {
        V3 hinge = v3(gm.x + 0.03f, gm.y, gm.z + s * 0.40f);
        V3 dir = s > 0 ? vnorm(v3(0.35f, 0, 1)) : vnorm(v3(0.85f, 0, 0.5f));
        obox(m, vadd(hinge, vmul(dir, 0.21f)), dir, v3(0, 1, 0), vnorm(vcross(dir, v3(0, 1, 0))), 0.21f, 0.33f, 0.02f, M_STEEL);
    }
    /* the stand */
    for (int i = 0; i < 4; i++) {
        float x = gb.x + 0.12f + (i & 1) * (GH_LEN - 0.24f), z = gm.z + ((i & 2) ? 0.35f : -0.35f);
        box(m, x - 0.035f, 0, z - 0.035f, x + 0.035f, GH_C.y - 0.35f, z + 0.035f, M_STEEL);
    }
    box(m, gb.x + 0.05f, GH_C.y - 0.45f, gm.z - 0.4f, gm.x - 0.05f, GH_C.y - 0.38f, gm.z + 0.4f, M_STEEL);
    /* the yoke the pipe rests on in front of the mouth */
    rod(m, v3(gm.x + 0.55f, 0, gm.z), v3(gm.x + 0.55f, 0.95f, gm.z), 0.025f, 10, M_STEEL);
    beam(m, v3(gm.x + 0.55f, 0.95f, gm.z - 0.08f), v3(gm.x + 0.55f, 0.95f, gm.z + 0.08f), 0.015f, 0.015f, v3(0, 1, 0), M_STEEL);
    box(m, gm.x + 0.4f, 0, gm.z - 0.18f, gm.x + 0.7f, 0.03f, gm.z + 0.18f, M_STEEL);
    /* the hood over it and its duct */
    {
        V3 c = v3(GH_C.x + 0.2f, 0, GH_C.z);
        cur_seed = 52;
        /* a spun steel cone with a rolled rim, and its duct to the roof */
        V3 a = v3(c.x, 2.45f, c.z), b = v3(c.x, 3.0f, c.z);
        int seg = 40;
        for (int j = 0; j < seg; j++) {
            float a0 = 2 * PI * j / seg, a1 = 2 * PI * (j + 1) / seg;
            V3 d0 = v3(cosf(a0), 0, sinf(a0)), d1 = v3(cosf(a1), 0, sinf(a1));
            float r0 = 0.85f, r1 = 0.24f, h = b.y - a.y, sl = (r0 - r1) / h;
            V3 n0 = vnorm(v3(d0.x, sl, d0.z)), n1 = vnorm(v3(d1.x, sl, d1.z));
            V3 p0 = vadd(a, vmul(d0, r0)), p1 = vadd(a, vmul(d1, r0)), p2 = vadd(b, vmul(d1, r1)), p3 = vadd(b, vmul(d0, r1));
            tri(m, p0, p1, p2, n0, n1, n1, a0, 0, a1, 0, a1, 1, M_STEEL);
            tri(m, p0, p2, p3, n0, n1, n0, a0, 0, a1, 1, a0, 1, M_STEEL);
        }
        cyl(m, vadd(a, v3(0, -0.03f, 0)), a, 0.86f, 40, 0, M_STEEL);
        cyl(m, vadd(a, v3(0, -0.03f, 0)), a, 0.845f, 40, 1, M_STEEL);
        cyl(m, b, v3(c.x, ROOM_H, c.z), 0.24f, 32, 0, M_STEEL);
        for (int i = 0; i < 3; i++) {
            float an = i * 2.1f + 0.4f;
            beam(m, vadd(a, v3(cosf(an) * 0.8f, 0.05f, sinf(an) * 0.8f)), v3(c.x + cosf(an) * 0.5f, 4.0f, c.z + sinf(an) * 0.5f), 0.008f, 0.008f, v3(0, 1, 0), M_STEEL);
        }
    }

    /* the bench: two long arms, a seat, legs; a bucket and blocks beside it */
    cur_seed = 60;
    for (int i = 0; i < 2; i++) {
        float x = i ? RAIL_XR : RAIL_XL;
        box(m, x - 0.035f, RAIL_Y - 0.06f, -1.62f, x + 0.035f, RAIL_Y, -0.30f, M_WOOD);
        /* steel strap on top, where the pipe rolls */
        box(m, x - 0.03f, RAIL_Y, -1.2f, x + 0.03f, RAIL_Y + 0.006f, -0.32f, M_POLISH);
        box(m, x - 0.03f, 0, -1.55f, x + 0.03f, RAIL_Y - 0.06f, -1.49f, M_WOOD);
        box(m, x - 0.03f, 0, -0.43f, x + 0.03f, RAIL_Y - 0.06f, -0.37f, M_WOOD);
        box(m, x - 0.025f, 0.12f, -1.5f, x + 0.025f, 0.17f, -0.4f, M_WOOD);
    }
    box(m, RAIL_XL + 0.035f, SEAT.y - 0.05f, -1.52f, RAIL_XR - 0.035f, SEAT.y, -0.92f, M_WOOD);
    box(m, RAIL_XL + 0.035f, 0.12f, -1.0f, RAIL_XR - 0.035f, 0.17f, -0.95f, M_WOOD);
    /* the bucket of water with wooden blocks soaking */
    cur_seed = 61;
    cyl(m, v3(1.45f, 0, -0.45f), v3(1.45f, 0.36f, -0.45f), 0.17f, 24, 0, M_STEEL);
    disc(m, v3(1.45f, 0.30f, -0.45f), v3(0, 1, 0), 0, 0.165f, 24, M_POLISH);
    rod(m, v3(1.4f, 0.28f, -0.48f), v3(1.32f, 0.62f, -0.6f), 0.012f, 8, M_WOOD);
    ellipsoid(m, v3(1.47f, 0.33f, -0.40f), v3(1, 0, 0), v3(0, 1, 0), v3(0, 0, 1), 0.05f, 0.05f, 0.05f, 12, M_WOOD);
    /* a tool tray at the end of the right arm: tweezers, shears, a paddle */
    cur_seed = 62;
    box(m, RAIL_XR + 0.04f, RAIL_Y - 0.02f, -0.95f, RAIL_XR + 0.30f, RAIL_Y, -0.35f, M_WOOD);
    beam(m, v3(RAIL_XR + 0.08f, RAIL_Y + 0.005f, -0.9f), v3(RAIL_XR + 0.12f, RAIL_Y + 0.005f, -0.45f), 0.008f, 0.004f, v3(0, 1, 0), M_POLISH);
    beam(m, v3(RAIL_XR + 0.18f, RAIL_Y + 0.006f, -0.88f), v3(RAIL_XR + 0.16f, RAIL_Y + 0.006f, -0.5f), 0.012f, 0.004f, v3(0, 1, 0), M_POLISH);
    box(m, RAIL_XR + 0.20f, RAIL_Y, -0.8f, RAIL_XR + 0.28f, RAIL_Y + 0.012f, -0.42f, M_WOOD);

    /* the marver: a thick polished plate on a steel stand */
    cur_seed = 70;
    box(m, MV_X0, MV_Y - 0.05f, MV_Z0, MV_X1, MV_Y, MV_Z1, M_POLISH);
    for (int i = 0; i < 4; i++) {
        float x = (i & 1) ? MV_X1 - 0.08f : MV_X0 + 0.04f, z = (i & 2) ? MV_Z1 - 0.08f : MV_Z0 + 0.04f;
        box(m, x, 0, z, x + 0.04f, MV_Y - 0.05f, z + 0.04f, M_STEEL);
    }
    box(m, MV_X0 + 0.04f, 0.2f, MV_Z0 + 0.04f, MV_X1 - 0.04f, 0.23f, MV_Z1 - 0.04f, M_STEEL);

    /* the annealer: an insulated box with a door (drawn with the moving parts) */
    cur_seed = 80;
    box(m, AN_X0, 0.35f, AN_Z0, AN_X1, AN_H, AN_Z1 - 0.12f, M_ANNEAL);
    box(m, AN_X0, 0.35f, AN_Z1 - 0.12f, AN_DX0, AN_H, AN_Z1, M_ANNEAL);
    box(m, AN_DX1, 0.35f, AN_Z1 - 0.12f, AN_X1, AN_H, AN_Z1, M_ANNEAL);
    box(m, AN_DX0, 0.35f, AN_Z1 - 0.12f, AN_DX1, AN_DY0, AN_Z1, M_ANNEAL);
    box(m, AN_DX0, AN_DY1, AN_Z1 - 0.12f, AN_DX1, AN_H, AN_Z1, M_ANNEAL);
    quad(m, v3(AN_DX0, AN_DY0, AN_Z1 - 0.12f), v3(AN_DX1 - AN_DX0, 0, 0), v3(0, AN_DY1 - AN_DY0, 0), M_FIBER);
    for (int i = 0; i < 4; i++) {
        float x = (i & 1) ? AN_X1 - 0.1f : AN_X0 + 0.05f, z = (i & 2) ? AN_Z1 - 0.1f : AN_Z0 + 0.05f;
        box(m, x, 0, z, x + 0.05f, 0.35f, z + 0.05f, M_STEEL);
    }
    box(m, AN_DX0 - 0.05f, AN_H, AN_Z1 - 0.06f, AN_DX0 - 0.01f, AN_H + 0.7f, AN_Z1 + 0.02f, M_STEEL);
    box(m, AN_DX1 + 0.01f, AN_H, AN_Z1 - 0.06f, AN_DX1 + 0.05f, AN_H + 0.7f, AN_Z1 + 0.02f, M_STEEL);
    box(m, AN_DX0 - 0.05f, AN_H + 0.7f, AN_Z1 - 0.06f, AN_DX1 + 0.05f, AN_H + 0.74f, AN_Z1 + 0.02f, M_STEEL);
    /* its controller: a little box with a red readout */
    box(m, AN_X1 + 0.02f, 0.9f, AN_Z1 - 0.4f, AN_X1 + 0.12f, 1.15f, AN_Z1 - 0.2f, M_STEEL);

    /* the gallery: boards across the window on iron brackets (the sill is the first) */
    for (int i = 1; i < SH_N; i++) {
        cur_seed = 90 + i;
        box(m, SH_X - SH_D, shelf_y(i) - 0.03f, SH_Z0, SH_X, shelf_y(i), SH_Z1, M_PLANK);
        for (int k = 0; k < 3; k++) {
            float z = SH_Z0 + 0.25f + (SH_Z1 - SH_Z0 - 0.5f) * k / 2;
            box(m, SH_X - 0.04f, shelf_y(i) - 0.16f, z - 0.012f, SH_X, shelf_y(i) - 0.03f, z + 0.012f, M_STEEL);
        }
    }
    /* a long low cabinet under the window */
    cur_seed = 99;
    box(m, ROOM_X - 0.5f, 0, WIN_Z0 - 0.2f, ROOM_X, 0.8f, WIN_Z1 + 0.2f, M_WOOD);
    box(m, ROOM_X - 0.52f, 0.8f, WIN_Z0 - 0.22f, ROOM_X, 0.84f, WIN_Z1 + 0.22f, M_PLANK);

    /* the tool wall: a board with jacks, shears, tweezers and paddles hung on pegs */
    cur_seed = 110;
    box(m, -5.2f, 1.05f, ROOM_Z0, -2.4f, 2.05f, ROOM_Z0 + 0.03f, M_WOOD);
    for (int i = 0; i < 9; i++) {
        float x = -5.0f + i * 0.3f, y = 1.85f;
        cur_seed = 111 + i;
        rod(m, v3(x, y, ROOM_Z0 + 0.03f), v3(x, y + 0.01f, ROOM_Z0 + 0.12f), 0.008f, 6, M_STEEL);
        int kind = i % 4;
        float sw = 0.04f + 0.05f * hf(i, 3, 3);
        if (kind == 0 || kind == 2) {          /* jacks: two blades from a spring bow */
            V3 top = v3(x, y, ROOM_Z0 + 0.07f);
            beam(m, top, v3(x - sw, y - 0.42f, ROOM_Z0 + 0.06f), 0.007f, 0.003f, v3(0, 0, 1), M_POLISH);
            beam(m, top, v3(x + sw, y - 0.42f, ROOM_Z0 + 0.08f), 0.007f, 0.003f, v3(0, 0, 1), M_POLISH);
        } else if (kind == 1) {                /* a paddle */
            box(m, x - 0.06f, y - 0.6f, ROOM_Z0 + 0.05f, x + 0.06f, y - 0.25f, ROOM_Z0 + 0.07f, M_WOOD);
            rod(m, v3(x, y - 0.25f, ROOM_Z0 + 0.06f), v3(x, y, ROOM_Z0 + 0.06f), 0.012f, 8, M_WOOD);
        } else {                               /* shears */
            beam(m, v3(x, y, ROOM_Z0 + 0.07f), v3(x - 0.02f, y - 0.32f, ROOM_Z0 + 0.07f), 0.009f, 0.003f, v3(0, 0, 1), M_STEEL);
            beam(m, v3(x, y, ROOM_Z0 + 0.075f), v3(x + 0.03f, y - 0.32f, ROOM_Z0 + 0.075f), 0.009f, 0.003f, v3(0, 0, 1), M_STEEL);
        }
    }
    /* a rack of blowpipes and punties leaning by the glory hole */
    cur_seed = 130;
    box(m, -4.3f, 0.0f, -2.95f, -3.3f, 0.05f, -2.75f, M_STEEL);
    box(m, -4.3f, 1.1f, -3.1f, -3.3f, 1.14f, -3.0f, M_STEEL);
    for (int i = 0; i < 8; i++) {
        float x = -4.2f + i * 0.12f + 0.03f * hf(i, 5, 5);
        rod(m, v3(x, 0.05f, -2.85f), v3(x + 0.03f, 1.6f + 0.1f * hf(i, 6, 6), -3.08f), 0.012f, 8, M_PIPE);
    }
    /* sheet steel on the floor round the hot end */
    cur_seed = 150;
    box(m, -3.3f, 0, -2.6f, -0.5f, 0.006f, 0.2f, M_STEEL);
}

/* ---- the pieces ---- */
enum { F_VASE, F_BOWL, F_GOBLET, F_BOTTLE, F_TRUMPET, F_TUMBLER, NFORM };
enum { G_TWIST, G_COBALT, G_AMBER, G_GREEN, G_MILLE, G_SMOKE, G_RUBY, G_TEAL, G_WRAP, G_CLEAR, NGLASS };
static const char *form_name[NFORM] = {"vase", "bowl", "goblet", "bottle", "trumpet vase", "tumbler"};
static const char *glass_name[NGLASS] = {"latticino", "cobalt", "amber", "green", "millefiori", "smoke",
                                         "cranberry", "teal", "wrapped", "clear"};
/* Beer-Lambert absorption, per metre */
static const float glass_abs[NGLASS][3] = {
    {4, 2, 4}, {260, 140, 10}, {6, 55, 260}, {150, 22, 130}, {5, 3, 5}, {75, 80, 88},
    {10, 260, 140}, {230, 25, 45}, {5, 2, 5}, {6, 2, 5}};
/* the colour of the threads / murrine, where there are any */
static const float glass_thread[NGLASS][3] = {
    {0.92f, 0.90f, 0.86f}, {0}, {0}, {0}, {0.85f, 0.2f, 0.15f}, {0}, {0}, {0}, {0.15f, 0.35f, 0.85f}, {0}};

typedef struct { int form, glass; float size, aspect, twist, var; } PieceDef;

#define CYCLE 240.0
/* the k-th piece of the session (negative: made before it, already on the shelves) */
static PieceDef piece_def(int k) {
    PieceDef d;
    uint32_t h = hash3((uint32_t)k, 11, 0);
    d.form = (int)(h % NFORM);
    /* not the same form or glass twice running */
    if (k != 0) {
        uint32_t hp = hash3((uint32_t)(k - 1), 11, 0);
        if ((int)(hp % NFORM) == d.form) d.form = (d.form + 1 + (int)(h >> 20) % (NFORM - 1)) % NFORM;
    }
    d.glass = (int)(hash3((uint32_t)k, 12, 0) % NGLASS);
    if (k != 0 && (int)(hash3((uint32_t)(k - 1), 12, 0) % NGLASS) == d.glass) d.glass = (d.glass + 3) % NGLASS;
    d.size = 0.88f + 0.24f * hf((uint32_t)k, 13, 0);
    d.aspect = 0.88f + 0.24f * hf((uint32_t)k, 14, 0);
    d.twist = (hf((uint32_t)k, 15, 0) < 0.5f ? -1 : 1) * (6 + 10 * hf((uint32_t)k, 16, 0));
    d.var = hf((uint32_t)k, 17, 0);
    return d;
}

/* profiles: NP points along the meridian from the bottom pole (s = 0) to the
 * lip (s = 1): axial a, radius r, wall w (0: solid glass) */
#define NP 36
typedef struct { float a[NP], r[NP], w[NP]; } Prof;

typedef struct { float a, r, solid; } CP;
static const CP F_CP[NFORM][14] = {
    /* vase: a baluster with a short neck and a flared lip */
    {{0, 0, 0}, {0, .035f, 0}, {.012f, .058f, 0}, {.045f, .079f, 0}, {.10f, .083f, 0}, {.15f, .066f, 0},
     {.19f, .034f, 0}, {.212f, .027f, 0}, {.235f, .033f, 0}, {.25f, .047f, 0}, {-1, 0, 0}},
    /* bowl */
    {{0, 0, 0}, {0, .036f, 0}, {.006f, .052f, 0}, {.025f, .092f, 0}, {.05f, .12f, 0}, {.075f, .135f, 0},
     {.095f, .143f, 0}, {.104f, .149f, 0}, {-1, 0, 0}},
    /* goblet: a solid foot and stem under the cup */
    {{0, 0, 1}, {0, .044f, 1}, {.004f, .046f, 1}, {.008f, .040f, 1}, {.016f, .011f, 1}, {.032f, .0065f, 1},
     {.072f, .0065f, 1}, {.086f, .013f, 1}, {.10f, .031f, 0}, {.125f, .049f, 0}, {.16f, .053f, 0},
     {.195f, .050f, 0}, {.206f, .051f, 0}, {-1, 0, 0}},
    /* bottle: a round shoulder and a long neck */
    {{0, 0, 0}, {0, .045f, 0}, {.01f, .062f, 0}, {.06f, .068f, 0}, {.12f, .066f, 0}, {.155f, .05f, 0},
     {.18f, .022f, 0}, {.20f, .016f, 0}, {.27f, .016f, 0}, {.285f, .023f, 0}, {-1, 0, 0}},
    /* trumpet */
    {{0, 0, 0}, {0, .038f, 0}, {.008f, .046f, 0}, {.03f, .041f, 0}, {.08f, .030f, 0}, {.14f, .033f, 0},
     {.20f, .052f, 0}, {.24f, .080f, 0}, {.262f, .101f, 0}, {-1, 0, 0}},
    /* tumbler */
    {{0, 0, 0}, {0, .040f, 0}, {.006f, .045f, 0}, {.05f, .047f, 0}, {.10f, .051f, 0}, {.13f, .053f, 0},
     {.136f, .054f, 0}, {-1, 0, 0}},
};
/* where the closed bubble's dome begins, before the lip is opened */
static const float F_CLOSE[NFORM] = {0.8f, 0.45f, 0.62f, 0.86f, 0.55f, 0.55f};

/* Catmull-Rom through the points, resampled evenly by arc length */
static void resample(const CP *cp, int n, Prof *o, float wall) {
    enum { DN = 240 };
    float da[DN + 1], dr[DN + 1], ds[DN + 1], dl[DN + 1];
    for (int i = 0; i <= DN; i++) {
        float f = (float)i / DN * (n - 1);
        int k = (int)f;
        if (k > n - 2) k = n - 2;
        float t = f - k;
        const CP *p0 = &cp[k > 0 ? k - 1 : 0], *p1 = &cp[k], *p2 = &cp[k + 1], *p3 = &cp[k + 2 < n ? k + 2 : n - 1];
        float t2 = t * t, t3 = t2 * t;
#define CR(c) (0.5f * (2 * p1->c + (-p0->c + p2->c) * t + (2 * p0->c - 5 * p1->c + 4 * p2->c - p3->c) * t2 + (-p0->c + 3 * p1->c - 3 * p2->c + p3->c) * t3))
        da[i] = CR(a); dr[i] = fmaxf(CR(r), 0);
#undef CR
        ds[i] = t < 0.5f ? p1->solid : p2->solid;
        dl[i] = i ? dl[i - 1] + hypotf(da[i] - da[i - 1], dr[i] - dr[i - 1]) : 0;
    }
    for (int j = 0, i = 0; j < NP; j++) {
        float L = dl[DN] * j / (NP - 1);
        while (i < DN - 1 && dl[i + 1] < L) i++;
        float t = dl[i + 1] > dl[i] ? (L - dl[i]) / (dl[i + 1] - dl[i]) : 0;
        t = clampf(t, 0, 1);
        o->a[j] = mixf(da[i], da[i + 1], t);
        o->r[j] = mixf(dr[i], dr[i + 1], t);
        o->w[j] = (ds[i] > 0.5f ? 0 : wall);
    }
}

static int cp_count(const CP *cp) { int n = 0; while (n < 14 && cp[n].a >= 0) n++; return n; }

/* the finished form, scaled for this piece */
static void prof_final(const PieceDef *d, Prof *o, float wall) {
    CP cp[14];
    int n = cp_count(F_CP[d->form]);
    for (int i = 0; i < n; i++) {
        cp[i] = F_CP[d->form][i];
        cp[i].a *= d->size * d->aspect;
        cp[i].r *= d->size / sqrtf(d->aspect);
    }
    resample(cp, n, o, wall);
    /* a heavier base, as they are */
    for (int i = 0; i < NP; i++)
        if (o->w[i] > 0) o->w[i] *= 1 + 2.5f * (1 - sstep(0.0f, 0.14f, (float)i / (NP - 1)));
}
/* the same body, still a closed bubble: a dome from F_CLOSE up to the neck */
static void prof_closed(const PieceDef *d, Prof *o, float wall, float neck) {
    CP cp[24];
    int n = cp_count(F_CP[d->form]);
    Prof f;
    prof_final(d, &f, wall);
    int m = 0, ic = (int)(F_CLOSE[d->form] * (NP - 1));
    for (int i = 0; i <= ic; i += 3) cp[m++] = (CP){f.a[i], f.r[i], f.w[i] == 0 ? 1.0f : 0.0f};
    float a0 = f.a[ic], r0 = f.r[ic];
    float dz = fmaxf(r0 * 0.85f, f.a[NP - 1] - a0 > 0 ? fminf(f.a[NP - 1] - a0, r0 * 1.2f) : 0.02f);
    for (int i = 1; i <= 6; i++) {
        float th = (PI / 2) * i / 6;
        cp[m++] = (CP){a0 + dz * sinf(th), neck + (r0 - neck) * cosf(th), 0};
    }
    cp[m++] = (CP){a0 + dz + 0.012f, neck, 0};
    (void)n;
    resample(cp, m, o, wall);
}
/* simple blobs for the early stages */
static void prof_blob(Prof *o, float len, float rad, float neck, float wall, float squash) {
    CP cp[8] = {{0, 0, 1}, {len * 0.06f, rad * 0.62f, 1}, {len * 0.22f, rad * 0.95f, 1}, {len * 0.5f, rad, 1},
                {len * (0.72f + 0.1f * squash), rad * (0.92f + 0.06f * squash), 1}, {len * 0.92f, neck + (rad - neck) * 0.45f, 1},
                {len, neck, 1}};
    if (wall > 0) for (int i = 0; i < 7; i++) cp[i].solid = 0;
    resample(cp, 7, o, wall);
}
static void prof_mix(Prof *o, const Prof *a, const Prof *b, float t) {
    t = ease(t);
    for (int i = 0; i < NP; i++) {
        o->a[i] = mixf(a->a[i], b->a[i], t);
        o->r[i] = mixf(a->r[i], b->r[i], t);
        float wa = a->w[i], wb = b->w[i];
        /* solid parts stay solid through a morph until the other is nearly reached */
        if (wa == 0 && wb == 0) o->w[i] = 0;
        else if (wa == 0) o->w[i] = t < 0.3f ? 0 : mixf(fmaxf(a->r[i], 0.004f), wb, (t - 0.3f) / 0.7f);
        else if (wb == 0) o->w[i] = t > 0.7f ? 0 : mixf(wa, fmaxf(b->r[i], 0.004f), t / 0.7f);
        else o->w[i] = mixf(wa, wb, t);
    }
}

/* ---- the timeline of one piece ---- */
enum {
    S_GATHER, S_TO_MARVER, S_MARVER, S_TO_BENCH1, S_BLOW, S_TO_GH1, S_REHEAT1, S_TO_BENCH2, S_SHAPE,
    S_TO_GH2, S_REHEAT2, S_TO_BENCH3, S_PUNTY, S_TO_GH3, S_REHEAT3, S_TO_BENCH4, S_OPEN, S_COOL,
    S_TO_ANNEAL, S_ANNEAL, S_TO_FURNACE, NSTAGE
};
static const float st_len[NSTAGE] = {14, 4, 16, 4, 12, 4, 12, 4, 40, 4, 10, 4, 12, 4, 10, 4, 30, 30, 6, 10, 6};
static float st_t0[NSTAGE + 1];
/* where each stage happens */
enum { P_FURN, P_MARVER, P_BENCH, P_GH, P_ANNEAL, P_WALK };
static const int st_place[NSTAGE] = {P_FURN, P_WALK, P_MARVER, P_WALK, P_BENCH, P_WALK, P_GH, P_WALK, P_BENCH,
                                     P_WALK, P_GH, P_WALK, P_BENCH, P_WALK, P_GH, P_WALK, P_BENCH, P_BENCH,
                                     P_WALK, P_ANNEAL, P_WALK};

/* the core temperature of the glass through the cycle (kelvin) */
static const float temp_key[][2] = {
    {0, 1450}, {14, 1440}, {18, 1390}, {34, 1250}, {38, 1235}, {50, 1185}, {54, 1170}, {57, 1300}, {66, 1460},
    {70, 1430}, {110, 1170}, {114, 1160}, {117, 1300}, {124, 1460}, {128, 1430}, {140, 1300}, {144, 1290},
    {147, 1350}, {154, 1460}, {158, 1430}, {188, 1140}, {196, 960}, {206, 820}, {218, 730}, {224, 700},
    {240, 690}};

static void timeline_init(void) {
    st_t0[0] = 0;
    for (int i = 0; i < NSTAGE; i++) st_t0[i + 1] = st_t0[i] + st_len[i];
}
static int stage_at(float c, float *lt) {
    int s = 0;
    while (s < NSTAGE - 1 && c >= st_t0[s + 1]) s++;
    *lt = c - st_t0[s];
    return s;
}
static float core_temp(float c) {
    int n = (int)(sizeof temp_key / sizeof temp_key[0]);
    for (int i = 0; i + 1 < n; i++)
        if (c < temp_key[i + 1][0]) {
            float f = (c - temp_key[i][0]) / (temp_key[i + 1][0] - temp_key[i][0]);
            return mixf(temp_key[i][1], temp_key[i + 1][1], ease(f));
        }
    return temp_key[n - 1][1];
}

/* the piece's form at cycle time c */
static void piece_prof(const PieceDef *d, float c, Prof *o) {
    float lt;
    int s = stage_at(c, &lt);
    Prof a, b;
    float grow = clampf(c / 12.0f, 0, 1);
    float gr = 0.034f + 0.016f * grow;
    if (s <= S_TO_MARVER) {
        prof_blob(o, 0.07f + 0.035f * grow, gr, 0.016f, 0, 0);
    } else if (s == S_MARVER) {
        prof_blob(&a, 0.105f, 0.05f, 0.016f, 0, 0);
        prof_blob(&b, 0.13f, 0.042f, 0.017f, 0, 1);
        prof_mix(o, &a, &b, lt / st_len[s]);
    } else if (s <= S_BLOW) {
        prof_blob(&a, 0.13f, 0.042f, 0.017f, 0, 1);
        prof_blob(&b, 0.14f, 0.047f, 0.017f, 0.017f, 1);
        prof_mix(o, &a, &b, s == S_BLOW ? lt / st_len[s] : 0);
    } else if (s < S_SHAPE) {
        prof_blob(o, 0.14f, 0.047f, 0.017f, 0.017f, 1);
    } else if (s == S_SHAPE) {
        float f = lt / st_len[s];
        Prof bl, cl;
        prof_blob(&a, 0.14f, 0.047f, 0.017f, 0.017f, 1);
        prof_closed(d, &cl, 0.0045f, 0.017f);
        /* blown out to about two thirds, then shaped and necked */
        bl = cl;
        for (int i = 0; i < NP; i++) { bl.a[i] *= 0.7f; bl.r[i] = 0.017f + (bl.r[i] - 0.017f) * 0.66f; bl.w[i] = bl.w[i] ? 0.008f : 0; }
        if (f < 0.4f) prof_mix(o, &a, &bl, f / 0.4f);
        else prof_mix(o, &bl, &cl, (f - 0.4f) / 0.6f);
    } else if (s < S_OPEN) {
        prof_closed(d, o, 0.0045f, 0.017f);
    } else if (s == S_OPEN) {
        prof_closed(d, &a, 0.0045f, 0.017f);
        prof_final(d, &b, 0.0038f);
        prof_mix(o, &a, &b, sstep(0.08f, 0.85f, lt / st_len[s]));
    } else {
        prof_final(d, o, 0.0038f);
    }
}

/* ---- the gaffer and the rod ---- */
#define PIPE_LEN 1.45f
#define PIPE_R 0.011f
#define PUNTY_R 0.008f
typedef struct {
    V3 root; float yaw, seated, walk, lean, gait;   /* walk: gait phase (radians); gait: its stride, 0 standing */
    V3 tip, back;                    /* the rod: hot end and unit direction to the gaffer's end */
    float rodlen; int punty;         /* which rod */
    float dl, dr;                    /* hands along the rod from the tip (dr < 0: the right hand is elsewhere) */
    V3 hand_r;                       /* the right hand when it isn't on the rod */
    int jacks; V3 jtip[2], jdir;     /* the jacks: tips and pointing direction */
    float hose;                      /* the blow hose, 0..1 */
    float spin;                      /* the rod's turn, radians */
    int attach;                      /* 0 lip at the tip (pipe), 1 bottom at the tip (punty), 2 in the annealer */
    float door;                      /* the annealer door: 0 shut, 1 up */
    V3 look;
} Pose;

static V3 fwd(float yaw) { return v3(sinf(yaw), 0, cosf(yaw)); }

static void station(int place, float lt, float h, Pose *p) {
    V3 gm = gh_mouth();
    p->seated = 0; p->lean = 0; p->gait = 0; p->walk = 0;
    p->dl = 0; p->dr = 0;
    switch (place) {
    case P_FURN:
        p->root = v3(-0.52f, 0, -2.0f); p->yaw = PI;
        p->tip = v3(PORT_C.x + 0.02f * sinf(lt * 0.7f), PORT_C.y - 0.04f, FUR_Z - 0.24f);
        p->back = vnorm(v3(0.1f, 0.05f, 1));
        p->dl = 1.36f; p->dr = 0.98f; p->lean = 0.15f;
        break;
    case P_MARVER: {
        float z = (MV_Z0 + MV_Z1) / 2 + 0.11f * sinf(lt * 2 * PI / 3.4f);
        p->root = v3(3.3f, 0, z + 0.06f); p->yaw = -PI / 2;
        p->back = vnorm(v3(1, 0.18f, 0));
        p->tip = v3(MV_X0 + 0.25f + h, MV_Y + 0.045f, z);
        p->dl = 1.34f; p->dr = 0.98f; p->lean = 0.25f;
        break;
    }
    case P_BENCH: {
        float z = PIPE_Z + 0.13f * sinf(lt * 2 * PI / 2.6f);
        p->root = v3(SEAT.x, 0, SEAT.z); p->yaw = 0; p->seated = 1;
        p->tip = v3(RAIL_XL - 0.07f, RAIL_Y + PIPE_R + 0.006f, z);
        p->back = v3(1, 0, 0);
        p->dl = 1.0f; p->dr = -1; p->lean = 0.12f;
        p->hand_r = v3(SEAT.x - 0.2f, 0.62f, SEAT.z + 0.32f);
        break;
    }
    case P_GH:
        p->root = v3(gm.x + 1.75f, 0, gm.z + 0.32f); p->yaw = -PI / 2;
        p->tip = v3(gm.x - 0.32f, gm.y - 0.02f, gm.z + 0.01f);
        p->back = vnorm(v3(1, 0.07f, 0.16f));
        p->dl = 1.40f; p->dr = 1.02f; p->lean = 0.12f;
        break;
    case P_ANNEAL: {
        float in = sstep(1.5f, 4.0f, lt) - sstep(6.5f, 8.5f, lt);
        p->root = v3((AN_DX0 + AN_DX1) / 2 + 0.12f, 0, AN_Z1 + 0.95f + 0.35f * in * 0); p->yaw = PI;
        p->tip = v3((AN_DX0 + AN_DX1) / 2, AN_DY0 + 0.12f + 0.06f * (1 - in), AN_Z1 + 0.15f - 0.55f * in);
        p->root.z = p->tip.z + 1.25f;
        p->back = vnorm(v3(0, 0.25f, 1));
        p->dl = 1.12f; p->dr = 0.78f; p->lean = 0.2f;
        break;
    }
    }
}

/* walking: carrying the rod before him, the hot end up and ahead */
static void carry(V3 root, float yaw, Pose *p) {
    V3 f = fwd(yaw);
    p->root = root; p->yaw = yaw;
    p->back = vnorm(vadd(vmul(f, -1), v3(0, 0.22f, 0)));
    p->tip = vadd(vadd(root, vmul(f, 1.0f)), v3(0, 0.95f, 0));
    p->tip = vsub(p->tip, vmul(p->back, -0.25f));
    p->dl = PIPE_LEN - 0.15f; p->dr = PIPE_LEN - 0.55f;
    p->seated = 0; p->lean = 0.05f;
}

static float angle_lerp(float a, float b, float t) {
    float d = fmodf(b - a + 3 * PI, 2 * PI) - PI;
    return a + d * t;
}
static void pose_lerp(Pose *o, const Pose *a, const Pose *b, float t) {
    t = ease(t);
    *o = *a;
    o->root = vlerp(a->root, b->root, t);
    o->yaw = angle_lerp(a->yaw, b->yaw, t);
    o->seated = mixf(a->seated, b->seated, t);
    o->lean = mixf(a->lean, b->lean, t);
    o->tip = vlerp(a->tip, b->tip, t);
    o->back = vnorm(vlerp(a->back, b->back, t));
    o->dl = mixf(a->dl, b->dl, t);
    o->dr = a->dr < 0 || b->dr < 0 ? (t < 0.5f ? a->dr : b->dr) : mixf(a->dr, b->dr, t);
    o->hand_r = a->dr < 0 ? a->hand_r : b->hand_r;
}

/* the rod's turning, accumulated through the stages so it never jumps */
static float spin_in(int s, float lt, float h, float rmax) {
    int pl = st_place[s];
    if (pl == P_BENCH) return 0.13f * sinf(lt * 2 * PI / 2.6f) / PIPE_R * 0.35f + 0.6f * lt;
    if (pl == P_MARVER) return 0.11f * sinf(lt * 2 * PI / 3.4f) / fmaxf(rmax, 0.03f);
    (void)h;
    return (pl == P_WALK ? 1.6f : pl == P_ANNEAL ? 0.4f : 2.6f) * lt;
}
static float spin_at(int s, float lt, float h, float rmax) {
    float a = 0;
    for (int i = 0; i < s; i++) a += spin_in(i, st_len[i], h, rmax) - spin_in(i, 0, h, rmax);
    return a + spin_in(s, lt, h, rmax) - spin_in(s, 0, h, rmax);
}

static float prof_height(const Prof *pr) { float h = 0; for (int i = 0; i < NP; i++) h = fmaxf(h, pr->a[i]); return h; }
static float prof_rmax(const Prof *pr) { float r = 0; for (int i = 0; i < NP; i++) r = fmaxf(r, pr->r[i]); return r; }

/* the whole state of the gaffer at cycle time c, for a piece of height h */
static void pose_at(float c, float h, float rmax, Pose *p) {
    float lt;
    int s = stage_at(c, &lt);
    memset(p, 0, sizeof *p);
    int pl = st_place[s];
    if (pl != P_WALK) {
        station(pl, lt, h, p);
    } else {
        int from = st_place[s - 1], to = st_place[(s + 1) % NSTAGE];
        Pose A, B, C;
        float L = st_len[s];
        station(from, st_len[s - 1], h, &A);
        station(to, 0, h, &B);
        V3 d = vsub(B.root, A.root);
        float wy = atan2f(d.x, d.z);
        float tu = 1.0f, td = L - 1.0f;
        V3 r = vlerp(A.root, B.root, sstep(tu * 0.5f, td + 0.5f, lt));
        if (lt < tu) {
            carry(r, angle_lerp(A.yaw, wy, ease(lt / tu)), &C);
            pose_lerp(p, &A, &C, lt / tu);
            p->yaw = C.yaw;
        } else if (lt > td) {
            carry(r, angle_lerp(wy, B.yaw, ease((lt - td) / (L - td))), &C);
            pose_lerp(p, &C, &B, (lt - td) / (L - td));
            p->yaw = C.yaw;
        } else {
            carry(r, wy, p);
        }
        p->root = r;
        /* the gait, while moving */
        p->gait = sstep(tu * 0.4f, tu, lt) * (1 - sstep(td, td + 0.5f, lt));
        p->walk = lt * 2 * PI / 1.1f;
    }
    p->rodlen = PIPE_LEN;
    p->punty = c >= st_t0[S_PUNTY] + 6.0f;
    p->attach = p->punty ? 1 : 0;
    if (s == S_ANNEAL && lt > 5.0f) p->attach = 2;
    if (s == S_TO_FURNACE) p->attach = 2;
    if (p->punty) p->rodlen = 1.2f;
    p->spin = spin_at(s, lt, h, rmax);
    if (s == S_ANNEAL) p->door = sstep(0.5f, 2.0f, lt) * (1 - sstep(7.5f, 9.5f, lt));
    if (s == S_BLOW) p->hose = sstep(0.0f, 1.5f, lt) * (1 - sstep(10.5f, 12, lt));
    if (s == S_SHAPE) p->hose = 1 - sstep(0, 1.5f, lt) + sstep(8, 9, lt) * (1 - sstep(14, 15.5f, lt));
    return;
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__

/* shared: noise, blackbody, the lights and the environment seen in reflections */
#define COMMON GLSL( \
    uniform sampler3D uNoise; \
    uniform float uTime; \
    uniform vec3 uEye; \
    uniform vec3 uGhPos, uGhAxis, uGhCol; \
    uniform vec3 uFpPos, uFpAxis, uFpCol; \
    uniform vec3 uHgPos, uHgCol; \
    uniform float uHgRad; \
    uniform vec3 uMoonDir, uMoonCol; \
    uniform vec4 uWin; \
    uniform float uGhRad, uFpRad; \
    uniform mat4 uShadowM; \
    uniform sampler2DShadow uShadow; \
    float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); } \
    float hash13(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); } \
    float n3(vec3 p) { return texture(uNoise, p * (1.0 / 16.0)).r; } \
    float fbm3(vec3 p) { vec4 t = texture(uNoise, p * (1.0 / 16.0)); return t.r * 0.5 + t.g * 0.3 + t.b * 0.2; } \
    float vn2(vec2 p) { return texture(uNoise, vec3(p * 0.125, 0.37)).r; } \
    float fbm2(vec2 p) { return dot(texture(uNoise, vec3(p * 0.125, 0.61)).rgb, vec3(0.5, 0.3, 0.2)); } \
    vec3 planck(float T) { \
        vec3 lam = vec3(0.610, 0.550, 0.465); \
        vec3 b = 1.0 / (lam * lam * lam * lam * lam * (exp(14388.0 / (lam * max(T, 300.0))) - 1.0)); \
        return b * (0.61 * 0.61 * 0.61 * 0.61 * 0.61 * (exp(14388.0 / (0.61 * 1400.0)) - 1.0)); } \
    vec3 glow(float T) { vec3 b = planck(T); float m = max(b.r, 1e-20); return b / m * pow(m, 0.42); } \
    float window_mask(vec2 yz) { \
        vec2 w = (yz - uWin.xy) / (uWin.zw - uWin.xy); \
        if (any(lessThan(w, vec2(0.0))) || any(greaterThan(w, vec2(1.0)))) return 0.0; \
        vec2 g = fract(w * vec2(4.0, 5.0)); \
        float bar = smoothstep(0.0, 0.035, g.x) * smoothstep(1.0, 0.965, g.x) * smoothstep(0.0, 0.03, g.y) * smoothstep(1.0, 0.97, g.y); \
        return bar * smoothstep(0.0, 0.02, min(min(w.x, 1.0 - w.x), min(w.y, 1.0 - w.y))); } \
    /* does p see the moon through the window? */ \
    float moonlit(vec3 p) { \
        float s = (5.98 - p.x) / -uMoonDir.x; \
        if (s < 0.0) return 0.0; \
        vec3 q = p - uMoonDir * s; \
        return window_mask(q.yz); } \
    /* the glory hole and the port as seen in a mirror, blurred by roughness */ \
    /* a disc of light of angular radius rad seen along d (small angles: 1 - cos ~ a^2 / 2) */ \
    float disc_seen(vec3 d, vec3 L, float rad, float bl) { \
        float x = 2.0 * max(1.0 - dot(d, L), 0.0); \
        float r1 = (rad + bl) * (rad + bl), r0 = max(rad - bl, 0.0); r0 *= r0; \
        return (1.0 - smoothstep(r0, r1, x)) * rad * rad / r1; } \
    vec3 env(vec3 p, vec3 d, float rough) { \
        vec3 c = mix(vec3(0.006, 0.0045, 0.0035), vec3(0.03, 0.016, 0.008), smoothstep(-0.3, 0.1, d.y) * smoothstep(0.6, 0.0, d.y)); \
        c *= 0.6 + 0.8 * smoothstep(0.3, -0.8, d.x); \
        float bl = rough * 0.9 + 0.004; \
        vec3 L = uGhPos - p; float dl = length(L); L /= dl; \
        float facing = max(dot(-L, uGhAxis), 0.0); \
        float cs = dot(d, L); \
        c += uGhCol * uGhRad * disc_seen(d, L, 0.18 * sqrt(facing) / dl, bl); \
        c += uGhCol * 0.05 * exp((cs - 1.0) * 2.0 / (0.04 + rough)) * facing / (1.0 + dl * dl * 0.2); \
        L = uFpPos - p; dl = length(L); L /= dl; \
        facing = max(dot(-L, uFpAxis), 0.0); \
        cs = dot(d, L); \
        c += uFpCol * uFpRad * disc_seen(d, L, 0.17 * sqrt(facing) / dl, bl); \
        c += uFpCol * 0.04 * exp((cs - 1.0) * 2.0 / (0.04 + rough)) * facing / (1.0 + dl * dl * 0.2); \
        if (d.x > 0.05) { \
            float s = (5.98 - p.x) / d.x; vec3 q = p + d * s; \
            float wm = window_mask(q.yz + (hash12(q.yz * 40.0) - 0.5) * rough * 0.5) * smoothstep(0.1, 0.25, (q.y - uWin.x) / (uWin.z - uWin.x)); \
            c += vec3(0.12, 0.16, 0.26) * wm * (1.0 - rough); } \
        return c; } \
    float shadow_gh(vec3 p, vec3 n) { \
        vec3 l = uGhPos - p; float dl = length(l); \
        float nl = abs(dot(n, l / dl)); \
        vec4 s = uShadowM * vec4(p + n * (0.01 + 0.03 * (1.0 - nl)) * (0.5 + dl * 0.25), 1.0); \
        if (s.w <= 0.0) return 0.0; \
        vec3 q = s.xyz / s.w * 0.5 + 0.5; \
        if (any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0)))) return 1.0; \
        q.z -= 0.0006; \
        return texture(uShadow, q); } \
)

static const char *VS_SCENE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aUV;
    layout(location = 3) in float aMat;
    layout(location = 4) in float aSeed;
    uniform mat4 uVP;
    out vec3 vP, vN; out vec2 vUV; flat out int vMat; flat out float vSeed;
    void main() {
        vP = aPos; vN = aN; vUV = aUV; vMat = int(aMat + 0.5); vSeed = aSeed;
        gl_Position = uVP * vec4(aPos, 1.0);
    });

static const char *VS_SHADOW = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aPos;
    uniform mat4 uVP;
    void main() { gl_Position = uVP * vec4(aPos, 1.0); });
static const char *FS_SHADOW = "#version 330 core\n" GLSL(
    out vec4 o;
    void main() { o = vec4(1.0); });

static const char *FS_SCENE = "#version 330 core\n" COMMON GLSL(
    in vec3 vP, vN; in vec2 vUV; flat in int vMat; flat in float vSeed;
    uniform vec3 uRodTip, uRodDir; uniform float uRodT;
    uniform vec3 uAmb;
    uniform float uAnnealDoor;
    uniform int uDbgS;
    out vec4 o;

    float ggx(vec3 N, vec3 V, vec3 L, float r) {
        vec3 Hh = normalize(V + L);
        float a = r * r, a2 = a * a;
        float nh = max(dot(N, Hh), 0.0), nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3);
        float d = nh * nh * (a2 - 1.0) + 1.0;
        float k = (r + 1.0) * (r + 1.0) / 8.0;
        return a2 / (3.14159 * d * d) * nl / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k)) * 0.25;
    }
    /* bricks in the plane of a face: 230 x 76 mm with 8 mm joints */
    vec3 bricks(vec2 p, out float joint, out float id) {
        vec2 b = vec2(0.238, 0.084);
        float row = floor(p.y / b.y);
        float x = p.x / b.x + 0.5 * mod(row, 2.0);
        vec2 cell = vec2(floor(x), row);
        vec2 f = vec2(fract(x) * b.x, fract(p.y / b.y) * b.y);
        joint = smoothstep(0.0, 0.006, min(min(f.x, b.x - f.x), min(f.y, b.y - f.y)));
        id = hash12(cell + 7.0);
        return vec3(f, 0.0);
    }

    void main() {
        if (uDbgS == 9) { o = vec4(0.1); return; }
        vec3 P = vP, N = normalize(vN), V = normalize(uEye - P);
        if (dot(N, V) < 0.0) N = -N;
        vec3 alb = vec3(0.1); float rough = 0.7, metal = 0.0, wrap = 0.0, sheen = 0.0, ao = 1.0;
        vec3 emis = vec3(0.0);
        int m = vMat;
        vec3 an = abs(N);
        vec2 fp = an.x > an.y && an.x > an.z ? P.zy : an.z > an.y ? P.xy : P.xz;
        if (m == 0) {                              /* concrete floor */
            vec4 nt = texture(uNoise, vec3(P.xz * 0.16, 0.61));
            float n = dot(nt.rgb, vec3(0.5, 0.3, 0.2)), g = vn2(P.xz * 22.0);
            alb = vec3(0.16, 0.15, 0.14) * (0.7 + 0.5 * n) * (0.9 + 0.2 * g);
            float stain = smoothstep(0.55, 0.75, fbm2(P.xz * 0.6 + 4.0));
            alb *= 1.0 - 0.45 * stain;
            vec2 j = abs(fract(P.xz / 2.4 + 0.5) - 0.5) * 2.4;
            alb *= 0.55 + 0.45 * smoothstep(0.0, 0.012, min(j.x, j.y));
            rough = mix(0.75, 0.32, stain) - 0.15 * smoothstep(0.55, 0.75, nt.b);
            ao = 0.75 + 0.25 * smoothstep(0.0, 0.6, length(P.xz - vec2(0.62, -1.0)));
        } else if (m == 1 || m == 5) {             /* fire brick (furnace) / old brick (walls) */
            float j, id;
            bricks(fp + vSeed * 3.1, j, id);
            float n = fbm2(fp * 9.0);
            if (m == 1) {
                alb = mix(vec3(0.36, 0.27, 0.18), vec3(0.46, 0.36, 0.24), id) * (0.75 + 0.4 * n);
                /* soot above the port */
                float soot = smoothstep(0.5, 0.0, abs(P.x - uFpPos.x) - 0.1) * smoothstep(uFpPos.y + 0.05, uFpPos.y + 0.7, P.y);
                alb *= 1.0 - 0.75 * soot * (0.6 + 0.4 * n);
                /* the bricks round the port are hot */
                vec2 dp = abs(P.xy - uFpPos.xy) - vec2(0.2, 0.15);
                float dd = length(max(dp, 0.0)) + max(0.0, P.z - uFpPos.z) * 0.0;
                if (P.z < uFpPos.z + 0.01) dd = max(dd, 0.0);
                float T = 520.0 + 900.0 * exp(-dd / 0.05) * step(uFpPos.z - 0.36, P.z);
                emis += glow(T) * 2.0 * (0.8 + 0.4 * n);
            } else {
                alb = mix(vec3(0.075, 0.04, 0.028), vec3(0.11, 0.06, 0.04), id) * (0.6 + 0.5 * n);
                alb *= mix(1.0, 0.45, smoothstep(2.6, 4.2, P.y));
            }
            alb *= mix(0.35, 1.0, j);
            rough = 0.85;
        } else if (m == 2 || m == 18) {            /* dark steel; the annealer's painted case */
            float n = fbm2(fp * 6.0 + vSeed), s = vn2(fp * vec2(40.0, 9.0)) * 0.4;
            alb = m == 2 ? vec3(0.055, 0.05, 0.045) * (0.7 + 0.6 * n) : vec3(0.10, 0.11, 0.115) * (0.8 + 0.4 * n);
            alb = mix(alb, vec3(0.11, 0.05, 0.025), smoothstep(0.62, 0.8, fbm2(fp * 3.0 + vSeed * 2.0)) * 0.7);
            metal = 0.5; rough = 0.5 - 0.15 * s;
        } else if (m == 3) {                       /* polished steel */
            float s = vn2(fp * vec2(3.0, 180.0));
            alb = vec3(0.55, 0.54, 0.52); metal = 1.0; rough = 0.13 + 0.08 * s + 0.1 * smoothstep(0.6, 0.9, fbm2(fp * 5.0));
        } else if (m == 4 || m == 16) {            /* wood: bench, handles; shelf boards */
            vec2 g = vUV * vec2(1.0, 1.0);
            float grain = vn2(vec2(g.x * 2.0, (g.y + fbm2(g * 3.0) * 0.2) * 60.0)) * 0.6 + vn2(g * vec2(0.5, 150.0)) * 0.4;
            alb = (m == 4 ? vec3(0.085, 0.05, 0.03) : vec3(0.30, 0.19, 0.11)) * (0.7 + 0.5 * grain);
            /* charred where the hot end has lain */
            if (m == 4) alb *= mix(0.3, 1.0, smoothstep(-0.2, 0.6, P.x + (P.y > 0.7 ? 0.0 : 1.0)));
            rough = m == 16 ? 0.45 : 0.6;
        } else if (m == 6) {                       /* work clothes */
            alb = vec3(0.03, 0.032, 0.04) * (0.85 + 0.3 * vn2(vUV * 30.0)); rough = 0.9; sheen = 0.35;
        } else if (m == 7) {                       /* skin */
            alb = vec3(0.16, 0.09, 0.065); rough = 0.55; wrap = 0.4;
        } else if (m == 8) {                       /* leather apron */
            alb = vec3(0.17, 0.085, 0.04) * (0.7 + 0.5 * fbm2(P.xy * 14.0 + P.z * 3.0)); rough = 0.55; sheen = 0.4;
        } else if (m == 9) {                       /* blowpipe: bright where worn, glowing near the glass */
            alb = vec3(0.35, 0.34, 0.33); metal = 1.0; rough = 0.32;
            vec3 dp = P - uRodTip;
            float along = dot(dp, uRodDir);
            float d = max(along, 0.0);
            float T = 300.0 + (uRodT - 300.0) * exp(-d / 0.075) * step(length(dp - uRodDir * along), 0.03);
            emis += glow(T) * 3.0;
            alb *= mix(0.25, 1.0, smoothstep(0.02, 0.35, d));
        } else if (m == 10) {                      /* the furnace's throat and the pot of glass */
            float depth = clamp((uFpPos.z - P.z) / 0.35, 0.0, 1.0);
            float n = fbm3(P * 9.0 + vec3(0.0, -uTime * 0.7, 0.0));
            emis = glow(1250.0 + 330.0 * depth + 60.0 * n) * (7.0 + 4.0 * depth);
            alb = vec3(0.0);
        } else if (m == 11) {                      /* the glory hole's throat: roaring */
            float depth = clamp((uGhPos.x - P.x) / 0.75, 0.0, 1.0);
            vec3 q = P - uGhPos;
            float a = atan(q.z, q.y);
            float n = fbm3(vec3(a * 1.8 + uTime * 1.3, q.x * 6.0 - uTime * 2.6, uTime * 0.4));
            float n2 = n3(vec3(a * 4.0 - uTime * 2.0, q.x * 14.0 + uTime * 5.0, 3.0));
            emis = glow(1350.0 + 200.0 * depth + 300.0 * n + 100.0 * n2) * (1.3 + 1.6 * depth) * (length(uGhCol) / 17.0);
            alb = vec3(0.0);
        } else if (m == 12) {                      /* the night outside: moonlit sky over the rooftops */
            vec2 yz = P.yz;
            float wm = window_mask(yz);
            vec2 w = (yz - uWin.xy) / (uWin.zw - uWin.xy);
            float pane = hash12(floor(w * vec2(4.0, 5.0)) + 3.0);
            /* old glass: each pane a little different, faintly wavy */
            vec2 wv = w + vec2(vn2(w * vec2(9.0, 3.0) + pane * 7.0) - 0.5, 0.0) * 0.006;
            vec3 sky = mix(vec3(0.03, 0.042, 0.08), vec3(0.012, 0.02, 0.05), smoothstep(0.1, 1.0, wv.x));
            sky += vec3(0.12, 0.15, 0.22) * exp(-length((wv - vec2(1.1, 0.15)) * vec2(1.0, 0.8)) * 2.6);
            vec2 sgp = wv * vec2(110.0, 160.0), sg = floor(sgp);
            float st = step(0.975, hash12(sg)) * smoothstep(0.3, 0.7, wv.x) * smoothstep(0.25, 0.05, length(fract(sgp) - 0.5));
            sky += vec3(0.5, 0.55, 0.7) * st * hash12(sg + 1.3);
            /* rooftops and chimneys against it, a few windows lit */
            float x = wv.y * 7.0;
            float hgt = 0.05 + 0.07 * hash12(vec2(floor(x), 3.0)) + 0.04 * step(0.85, hash12(vec2(floor(x * 5.0), 5.0)));
            if (wv.x < hgt) {
                sky = vec3(0.006, 0.007, 0.012);
                vec2 lw = floor(vec2(wv.x * 140.0, wv.y * 200.0));
                if (step(0.93, hash12(lw)) > 0.0 && fract(wv.x * 140.0) > 0.35 && fract(wv.y * 200.0) > 0.4)
                    sky = vec3(0.45, 0.28, 0.1) * (0.3 + 0.7 * hash12(lw + 2.0));
            }
            sky *= 0.85 + 0.3 * pane;
            emis = sky * wm * 1.0;
            alb = vec3(0.02) * (1.0 - wm); rough = 0.8;
        } else if (m == 13) {                      /* rubber hose, lamp cords */
            alb = vec3(0.07, 0.025, 0.02); rough = 0.55;
        } else if (m == 14) {                      /* the roof, lost in the dark */
            alb = vec3(0.025, 0.022, 0.02); rough = 0.9;
        } else if (m == 15) {                      /* ceramic fibre: the glory hole's face, the annealer's inside */
            alb = vec3(0.42, 0.38, 0.33) * (0.75 + 0.4 * fbm2(fp * 12.0)); rough = 0.95;
            float d = length(P.yz - uGhPos.yz);
            if (abs(P.x - uGhPos.x) < 0.05 && d < 0.6) {
                emis += glow(700.0 + 900.0 * exp(-(d - 0.18) / 0.06)) * 2.5;
                alb *= mix(0.25, 1.0, smoothstep(0.18, 0.45, d));
            } else {
                emis += glow(760.0) * 6.0 * uAnnealDoor;
            }
        } else if (m == 17) {                      /* hair */
            alb = vec3(0.025, 0.02, 0.018); rough = 0.6; sheen = 0.5;
        } else if (m == 19) {                      /* enamel lamp shades */
            alb = vec3(0.03, 0.07, 0.05); rough = 0.25; metal = 0.2;
        } else if (m == 20) {                      /* the port's door: hot along its lower edge */
            alb = vec3(0.06, 0.05, 0.045) * (0.7 + 0.6 * fbm2(fp * 7.0)); metal = 0.5; rough = 0.55;
            float d = P.y - (uFpPos.y + 0.11);
            emis += glow(600.0 + 700.0 * exp(-max(d, 0.0) / 0.04)) * 2.0;
        }

        /* lighting */
        vec3 F0 = mix(vec3(0.04), alb, metal);
        vec3 dif = alb * (1.0 - metal);
        vec3 col = vec3(0.0);
        /* the glory hole: a disc of fire facing +x, shadow-mapped */
        {
            vec3 L = uGhPos - P; float d2 = dot(L, L); L *= inversesqrt(d2);
            float e = max(dot(-L, uGhAxis), 0.0);
            e = e * e * e * 0.92 + 0.08 * smoothstep(-0.2, 0.3, dot(-L, uGhAxis));
            float nl = dot(N, L);
            float sh = shadow_gh(P, N);
            vec3 li = uGhCol * e / (d2 + 0.04) * sh;
            col += li * (dif * max((nl + wrap) / (1.0 + wrap), 0.0) + F0 * ggx(N, V, L, max(rough, 0.08)) * 3.14159);
            col += li * sheen * 0.15 * pow(1.0 - max(dot(N, V), 0.0), 3.0) * max(nl + 0.5, 0.0);
        }
        /* the furnace port */
        {
            vec3 L = uFpPos - P; float d2 = dot(L, L); L *= inversesqrt(d2);
            float e = max(dot(-L, uFpAxis), 0.0);
            e *= e;
            float nl = dot(N, L);
            vec3 li = uFpCol * e / (d2 + 0.04);
            col += li * (dif * max((nl + wrap) / (1.0 + wrap), 0.0) + F0 * ggx(N, V, L, max(rough, 0.08)) * 3.14159);
            col += li * sheen * 0.15 * pow(1.0 - max(dot(N, V), 0.0), 3.0) * max(nl + 0.5, 0.0);
        }
        /* the hot glass */
        {
            vec3 L = uHgPos - P; float d2 = dot(L, L); L *= inversesqrt(d2);
            float nl = dot(N, L);
            vec3 li = uHgCol / (d2 + uHgRad * uHgRad);
            float nh = max(dot(N, normalize(L + V)), 0.0);
            float sp = pow(nh, 2.0 / max(rough * rough * rough * rough, 0.002));
            col += li * (dif * max((nl + wrap) / (1.0 + wrap), 0.0) + F0 * sp * (0.5 / max(rough * rough, 0.01)) * max(nl, 0.0));
            col += li * sheen * 0.2 * (1.0 - max(dot(N, V), 0.0)) * max(nl + 0.5, 0.0);
        }
        /* the moon, through the window */
        {
            float ml = moonlit(P);
            vec3 L = -uMoonDir;
            if (ml > 0.0) col += uMoonCol * ml * (dif * max(dot(N, L), 0.0) + F0 * ggx(N, V, L, max(rough, 0.1)) * 3.14159);
        }
        col += uAmb * dif * (0.6 + 0.4 * N.y) * ao;
        /* mirror-ish surfaces see the fires */
        if (rough < 0.6) {
            vec3 R = reflect(-V, N);
            float nv = max(dot(N, V), 0.0);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - nv, 5.0) * (1.0 - rough);
            col += env(P, R, rough) * F * (1.0 - rough * 0.8);
        }
        col += emis;
        if (uDbgS == 1) col = alb;
        if (uDbgS == 2) col = vec3(shadow_gh(P, N));
        if (uDbgS == 3) col = N * 0.5 + 0.5;
        o = vec4(col, 1.0);
    });

/* glass: back faces first (the background seen through it, refracted and
 * tinted), then front faces over them (glow, reflections, threads), blended
 * as rgb + alpha * what's there */
static const char *VS_GLASS = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec4 aA;          /* s, u, temperature, wall/chord */
    layout(location = 3) in vec2 aB;          /* side (0 outside, 1 rim, 2 inside), piece */
    uniform mat4 uVP, uModel;
    out vec3 vP, vN; out vec4 vA; flat out int vPid; out float vSide;
    void main() {
        vec4 w = uModel * vec4(aPos, 1.0);
        vP = w.xyz; vN = mat3(uModel) * aN; vA = aA; vSide = aB.x; vPid = int(aB.y + 0.5);
        gl_Position = uVP * w;
    });

static const char *FS_GLASS = "#version 330 core\n" COMMON GLSL(
    in vec3 vP, vN; in vec4 vA; flat in int vPid; in float vSide;
    uniform int uPass;                        /* 0 back faces, 1 front faces */
    uniform sampler2D uScene;
    uniform vec2 uRes;
    uniform mat4 uView;
    uniform float uProjY;
    uniform vec3 uAbs[48];
    uniform vec4 uKind[48];                   /* glass, twist, thread r g b packed: x kind, y twist, z height */
    uniform vec3 uThread[48];
    out vec4 o;

    /* how much of the surface is coloured cane / murrine here, and its colour */
    float pattern(int kind, float twist, float s, float u, out vec3 tc) {
        tc = uThread[vPid];
        float a = u / 6.28318;
        if (kind == 0) {                       /* latticino: a twist of fine white threads */
            float f = fract(a * 14.0 + s * twist);
            float g = fract(a * 7.0 - s * twist * 0.5 + 0.25);
            return smoothstep(0.42, 0.5, 0.5 - abs(f - 0.5)) * 0.8 + smoothstep(0.46, 0.5, 0.5 - abs(g - 0.5)) * 0.5 * step(0.5, fract(s * 3.0 + a));
        }
        if (kind == 8) {                       /* a coloured thread wound round the upper part */
            float f = fract(s * 38.0 + a);
            return smoothstep(0.4, 0.5, 0.5 - abs(f - 0.5)) * smoothstep(0.35, 0.55, s);
        }
        if (kind == 4) {                       /* millefiori: slices of cane, each a little flower */
            vec2 g = vec2(a * 22.0, s * 15.0);
            vec2 c = floor(g + 0.5 * vec2(0.0, mod(floor(g.x), 2.0)));
            vec2 f = g + 0.5 * vec2(0.0, mod(floor(g.x), 2.0)) - c - 0.5;
            float h = hash12(c);
            float r = length(f * vec2(1.0, 1.4)) * 2.0;
            float ang = atan(f.y, f.x);
            float petals = 0.5 + 0.5 * cos(ang * (5.0 + floor(h * 4.0)));
            int pal = int(h * 5.0);
            vec3 c1 = pal == 0 ? vec3(0.85, 0.15, 0.1) : pal == 1 ? vec3(0.1, 0.25, 0.85) : pal == 2 ? vec3(0.95, 0.75, 0.15) : pal == 3 ? vec3(0.1, 0.6, 0.3) : vec3(0.9, 0.9, 0.85);
            vec3 c2 = pal == 4 ? vec3(0.8, 0.1, 0.4) : vec3(0.95, 0.93, 0.88);
            float ring = step(0.55 + 0.15 * petals, r);
            tc = mix(c2, c1, ring);
            if (r < 0.18) tc = vec3(0.95, 0.85, 0.2);
            return smoothstep(0.98, 0.85, r) * 0.95;
        }
        return 0.0;
    }

    void main() {
        vec3 N = normalize(vN), V = normalize(uEye - vP);
        float s = vA.x, u = vA.y, T = vA.z, wall = vA.w;
        vec4 kind = uKind[vPid];
        vec3 ab = uAbs[vPid];
        vec3 tcol;
        float pat = pattern(int(kind.x + 0.5), kind.y, s, u, tcol);
        float nv = abs(dot(N, V));
        /* path through the glass along this ray: two walls of a vessel, or a chord of a solid */
        float path = wall < 0.0 ? -wall * 2.0 * max(nv, 0.2) : 2.0 * wall / max(nv, 0.18);
        vec3 tint = exp(-ab * path);
        float hot = smoothstep(800.0, 1100.0, T);
        /* Kirchhoff: it glows where it absorbs */
        vec3 emiss = 1.0 - exp(-(vec3(24.0) + ab * 0.5) * path);
        vec3 glo = glow(T) * emiss * 15.0 * (1.0 + 0.1 * pat);

        if (uPass == 0) {
            /* refraction: bend the background by the surface's tilt in view */
            vec3 nview = mat3(uView) * N;
            vec4 cv = uView * vec4(vP, 1.0);
            vec2 uv = gl_FragCoord.xy / uRes;
            vec2 off = -nview.xy * (0.03 + path * 2.0) * (1.0 + 1.5 * (1.0 - nv)) * uProjY / max(-cv.z, 0.05) * vec2(uRes.y / uRes.x, 1.0);
            vec3 bg;
            bg.r = texture(uScene, uv + off * 0.94).r;
            bg.g = texture(uScene, uv + off).g;
            bg.b = texture(uScene, uv + off * 1.07).b;
            vec3 R = reflect(-V, -N);
            float F = 0.04 + 0.96 * pow(1.0 - nv, 5.0);
            vec3 c = bg * tint * (1.0 - F) * (1.0 - pat * 0.85 * (1.0 - hot)) * (1.0 - emiss * 0.8);
            c += env(vP, R, 0.02) * F * tint * 0.7;
            /* at grazing angles the light inside is turned back: the dark rims of real glass */
            float tir = smoothstep(0.38, 0.08, nv) * (1.0 - hot);
            c = mix(c, env(vP, R, 0.05) * tint * 0.6, tir);
            /* threads on the far side, lit dimly by the fires */
            vec3 Lg = normalize(uGhPos - vP);
            float lit = max(dot(-N, Lg), 0.0) * 0.5 + 0.5;
            c += tcol * pat * (dot(uGhCol, vec3(0.3)) * 0.004 + 0.02) * lit * (1.0 - hot);
            o = vec4(c, 1.0);
        } else {
            vec3 R = reflect(-V, N);
            float F = 0.04 + 0.96 * pow(1.0 - nv, 5.0);
            vec3 c = env(vP, R, 0.015) * F;
            /* the threads, lit by the fires */
            vec3 Lg = uGhPos - vP; float dg = dot(Lg, Lg); Lg *= inversesqrt(dg);
            vec3 Lf = uFpPos - vP; float df = dot(Lf, Lf); Lf *= inversesqrt(df);
            vec3 irr = uGhCol * max(dot(-Lg, uGhAxis), 0.0) * (max(dot(N, Lg), 0.0) * 0.7 + 0.3) / (dg + 0.04) * shadow_gh(vP, N)
                     + uFpCol * max(dot(-Lf, uFpAxis), 0.0) * (max(dot(N, Lf), 0.0) * 0.7 + 0.3) / (df + 0.04) + vec3(0.01);
            c += tcol * pat * irr * 0.25 * (1.0 - hot);
            c += glo;
            o = vec4(c, (1.0 - F) * (1.0 - pat * 0.8 * (1.0 - hot)) * (1.0 - dot(emiss, vec3(0.33)) * 0.7));
        }
    });

/* embers, sparks and motes: streaks expanded on the CPU */
static const char *VS_SPARK = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in vec3 aCol;
    uniform mat4 uVP;
    out vec2 vUV; out vec3 vCol;
    void main() { vUV = aUV; vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); });
static const char *FS_SPARK = "#version 330 core\n" GLSL(
    in vec2 vUV; in vec3 vCol;
    out vec4 o;
    void main() {
        float d = dot(vUV, vUV);
        o = vec4(vCol * exp(-d * 3.5) * smoothstep(1.0, 0.6, d), 1.0);
    });

/* fullscreen passes */
static const char *VS_FULL = "#version 330 core\n" GLSL(
    out vec2 vUV;
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        vUV = p;
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* light in the haze: march the view ray through drifting smoke */
static const char *FS_VOL = "#version 330 core\n" COMMON GLSL(
    in vec2 vUV;
    uniform sampler2D uDepth;
    uniform mat4 uInvVP;
    uniform float uFog;
    uniform int uSteps;
    uniform float uFrame;
    out vec4 o;
    void main() {
        float d = texture(uDepth, vUV).r;
        vec4 w = uInvVP * vec4(vUV * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
        w /= w.w;
        vec3 ro = uEye, rd = w.xyz - ro;
        float tmax = min(length(rd), 16.0);
        rd = normalize(rd);
        float dt = tmax / float(uSteps);
        float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))) + uFrame * 0.618);
        vec3 sc = vec3(0.0);
        float tr = 1.0;
        vec3 drift = vec3(uTime * 0.03, -uTime * 0.05, uTime * 0.02);
        for (int i = 0; i < 48; i++) {
            if (i >= uSteps) break;
            float t = (float(i) + j) * dt;
            vec3 p = ro + rd * t;
            float nz = fbm3(p * 0.7 + drift);
            float dens = uFog * (0.25 + 0.75 * smoothstep(0.5, 4.0, p.y)) * (0.15 + 3.0 * nz * nz * nz);
            /* heat and smoke rising off the fires */
            vec2 dg = p.xz - (uGhPos.xz + vec2(0.05, 0.0));
            dens += uFog * 2.5 * exp(-dot(dg, dg) * 5.0) * smoothstep(uGhPos.y - 0.2, uGhPos.y + 0.6, p.y) * fbm3(p * 2.0 - vec3(0.0, uTime * 0.4, 0.0));
            vec3 li = vec3(0.0);
            vec3 L = uGhPos - p; float d2 = dot(L, L); L *= inversesqrt(d2);
            float e = max(dot(-L, uGhAxis), 0.0);
            float c = dot(rd, -L);
            float ph = 0.3 + 1.2 * pow(max(c, 0.0), 4.0) + 0.25 * (1.0 + c * c);
            if (e > 0.0) {
                vec4 s = uShadowM * vec4(p, 1.0);
                vec3 q = s.xyz / s.w * 0.5 + 0.5;
                float sh = s.w > 0.0 ? texture(uShadow, q - vec3(0.0, 0.0, 0.002)) : 0.0;
                li += uGhCol * e * e * e * sh / (d2 + 0.1) * ph;
            }
            L = uFpPos - p; d2 = dot(L, L); L *= inversesqrt(d2);
            e = max(dot(-L, uFpAxis), 0.0);
            c = dot(rd, -L);
            li += uFpCol * e * e / (d2 + 0.1) * (0.3 + 1.2 * pow(max(c, 0.0), 4.0) + 0.25 * (1.0 + c * c));
            L = uHgPos - p; d2 = dot(L, L);
            li += uHgCol * 0.6 / (d2 + 0.02);
            float ml = moonlit(p);
            c = dot(rd, -uMoonDir);
            li += uMoonCol * ml * (0.4 + 2.0 * pow(max(c, 0.0), 6.0)) * 1.1;
            sc += tr * dens * dt * li;
            tr *= exp(-dens * dt * 0.6);
        }
        o = vec4(sc, tr);
    });

/* heat shimmer: plumes of hot air rising off the fires and the glass bend
 * whatever is behind them */
#define SHIMMER GLSL( \
    uniform vec4 uHeat[4]; \
    uniform vec2 uHeatK[4]; \
    uniform vec2 uNF; \
    uniform float uAspect; \
    float lin_z(float d) { float z = d * 2.0 - 1.0; return 2.0 * uNF.x * uNF.y / (uNF.y + uNF.x - z * (uNF.y - uNF.x)); } \
    vec2 shimmer(vec2 uv, float z) { \
        vec2 off = vec2(0.0); \
        for (int i = 0; i < 4; i++) { \
            vec4 h = uHeat[i]; \
            if (uHeatK[i].x <= 0.0 || z < uHeatK[i].y) continue; \
            float dy = (uv.y - h.y) / h.w; \
            if (dy < -0.1 || dy > 1.0) continue; \
            float wdt = h.z * (1.0 + 1.5 * max(dy, 0.0)); \
            float dx = (uv.x - h.x) * uAspect / wdt; \
            float m = exp(-dx * dx) * smoothstep(-0.1, 0.12, dy) * (1.0 - dy) * uHeatK[i].x; \
            if (m < 0.002) continue; \
            vec3 q = vec3(uv.x * uAspect * 7.0 / h.w, uv.y * 5.0 / h.w - uTime * 1.4, uTime * 0.35 + float(i)); \
            vec4 n = texture(uNoise, q * 0.25); \
            off += (n.gb - 0.5) * m; \
        } \
        return off * 0.012; } \
)

static const char *FS_PREP = "#version 330 core\n" GLSL(uniform sampler3D uNoise; uniform float uTime;) SHIMMER GLSL(
    in vec2 vUV;
    uniform sampler2D uScene, uDepth, uVol;
    uniform vec3 uDof;                         /* focus distance, CoC scale (half-res px), max CoC */
    out vec4 o;
    void main() {
        float z = lin_z(texture(uDepth, vUV).r);
        vec2 uv = vUV + shimmer(vUV, z);
        vec4 v = texture(uVol, vUV);
        vec3 c = texture(uScene, uv).rgb * v.a + v.rgb;
        float coc = clamp(uDof.y * (z - uDof.x) / z, -uDof.z, uDof.z);
        o = vec4(c, coc);
    });

/* a bokeh gather (after Dennis Gustafsson's single-pass depth of field) */
static const char *FS_DOF = "#version 330 core\n" GLSL(
    in vec2 vUV;
    uniform sampler2D uPrep;
    uniform vec2 uPx;
    uniform float uMax;
    out vec4 o;
    void main() {
        vec4 c0 = texture(uPrep, vUV);
        float cs = abs(c0.a);
        vec3 col = c0.rgb;
        float tot = 1.0;
        float r = 1.0;
        float ang = 0.0;
        for (int i = 0; i < 64; i++) {
            if (r >= uMax) break;
            vec2 tc = vUV + vec2(cos(ang), sin(ang)) * uPx * r;
            vec4 s = texture(uPrep, tc);
            float ss = abs(s.a);
            if (s.a > c0.a) ss = clamp(ss, 0.0, cs * 2.0);
            float m = smoothstep(r - 0.5, r + 0.5, ss);
            col += mix(col / tot, s.rgb, m);
            tot += 1.0;
            ang += 2.39996323;
            r += 1.6 / r;
        }
        o = vec4(col / tot, c0.a);
    });

static const char *FS_DOWN = "#version 330 core\n" GLSL(
    in vec2 vUV;
    uniform sampler2D uSrc;
    uniform vec2 uPx;
    uniform int uFirst;
    out vec4 o;
    void main() {
        vec3 c = texture(uSrc, vUV).rgb * 4.0;
        c += texture(uSrc, vUV + uPx * vec2(-1, -1)).rgb + texture(uSrc, vUV + uPx * vec2(1, -1)).rgb;
        c += texture(uSrc, vUV + uPx * vec2(-1, 1)).rgb + texture(uSrc, vUV + uPx * vec2(1, 1)).rgb;
        c /= 8.0;
        if (uFirst == 1) c = max(c - 0.6, 0.0) + min(c, 0.6) * 0.08;   /* only the brights bloom much */
        o = vec4(min(c, vec3(400.0)), 1.0);
    });
static const char *FS_UP = "#version 330 core\n" GLSL(
    in vec2 vUV;
    uniform sampler2D uSrc;
    uniform vec2 uPx;
    out vec4 o;
    void main() {
        vec3 c = texture(uSrc, vUV + uPx * vec2(-1, 0)).rgb + texture(uSrc, vUV + uPx * vec2(1, 0)).rgb;
        c += texture(uSrc, vUV + uPx * vec2(0, -1)).rgb + texture(uSrc, vUV + uPx * vec2(0, 1)).rgb;
        c *= 2.0;
        c += texture(uSrc, vUV + uPx * vec2(-1, -1)).rgb + texture(uSrc, vUV + uPx * vec2(1, -1)).rgb;
        c += texture(uSrc, vUV + uPx * vec2(-1, 1)).rgb + texture(uSrc, vUV + uPx * vec2(1, 1)).rgb;
        o = vec4(c / 12.0, 1.0);
    });

/* everything together: depth of field, bloom, a filmic curve and grade, grain */
static const char *FS_FINAL = "#version 330 core\n" GLSL(uniform sampler3D uNoise; uniform float uTime;) SHIMMER GLSL(
    in vec2 vUV;
    uniform sampler2D uScene, uDepth, uVol, uDofTex, uBloom;
    uniform vec3 uDof;
    uniform float uExposure, uBloomK, uFade;
    uniform int uDbg;
    uniform vec4 uVig;                         /* this view's rect in the vignette's frame */
    uniform vec2 uOutRes;
    out vec4 o;
    vec3 aces(vec3 x) { return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    void main() {
        float z = lin_z(texture(uDepth, vUV).r);
        vec2 uv = vUV + shimmer(vUV, z);
        vec4 v = texture(uVol, vUV);
        vec3 sharp = texture(uScene, uv).rgb * v.a + v.rgb;
        float coc = abs(clamp(uDof.y * (z - uDof.x) / z, -uDof.z, uDof.z));
        vec4 b = texture(uDofTex, vUV);
        float m = smoothstep(0.4, 1.4, max(coc, abs(b.a) * 0.6));
        vec3 c = mix(sharp, b.rgb, m);
        c += texture(uBloom, vUV).rgb * uBloomK;
        if (uDbg == 1) c = texture(uScene, vUV).rgb;
        if (uDbg == 2) c = v.rgb;
        if (uDbg == 3) c = vec3(v.a);
        if (uDbg == 4) c = texture(uBloom, vUV).rgb * uBloomK;
        c *= uExposure;
        /* a warm, slightly lifted filmic look; highlights roll to yellow-white */
        c = aces(c);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(vec3(l), c, 1.08);
        c = pow(c, vec3(1.0 / 2.2));
        c += vec3(-0.004, 0.006, 0.014) * (1.0 - smoothstep(0.0, 0.35, l));
        vec2 vp = uVig.xy + vUV * uVig.zw - 0.5;
        c *= 1.0 - 0.55 * dot(vp * vec2(1.0, 0.8), vp * vec2(1.0, 0.8));
        float g = fract(sin(dot(gl_FragCoord.xy + fract(uTime * 7.1) * 97.0, vec2(12.9898, 78.233))) * 43758.5453);
        c += (g - 0.5) * 0.035 * (0.4 + 0.6 * (1.0 - l));
        o = vec4(c * uFade, 1.0);
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

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "glassworks: shader:\n%s\n", log);
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
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "glassworks: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

static GLuint p_scene, p_shadow, p_glass, p_spark, p_vol, p_prep, p_dof, p_down, p_up, p_final, p_text;
static GLuint vao_full, vao_stat, vbo_stat, vao_dyn, vbo_dyn, vao_glass, vbo_glass, ebo_glass;
static GLuint vao_gal, vbo_gal, ebo_gal, vao_spark, vbo_spark, vao_text, vbo_text;
static GLuint tex_noise, tex_atlas;
#define SHADOW_N 1024
static GLuint fbo_shadow, tex_shadow;
/* render targets at the internal resolution (iw x ih) and below */
static int iw, ih, rt_w, rt_h;
static GLuint fbo_scene, tex_scene, tex_depth, fbo_copy, tex_copy;
static GLuint fbo_ms, rb_ms_col, rb_ms_dep;        /* the scene, multisampled */
static int opt_msaa = 4;
static GLuint fbo_vol, tex_vol, fbo_prep, tex_prep, fbo_dof, tex_dof;
#define NBLOOM 5
static GLuint fbo_bloom[NBLOOM], tex_bloom[NBLOOM];
static int bw[NBLOOM], bh[NBLOOM];

/* tileable 3-D value noise, three octaves in r, g, b and white noise in a */
static void make_noise(void) {
    enum { N = 64 };
    unsigned char *d = malloc((size_t)N * N * N * 4);
    for (int c = 0; c < 3; c++) {
        int L = 8 << c;                       /* lattice cells across */
        float *g = malloc(sizeof(float) * L * L * L);
        for (int i = 0; i < L * L * L; i++) g[i] = hf((uint32_t)i, 900 + c, 1);
        for (int z = 0; z < N; z++)
            for (int y = 0; y < N; y++)
                for (int x = 0; x < N; x++) {
                    float fx = (float)x * L / N, fy = (float)y * L / N, fz = (float)z * L / N;
                    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
                    float tx = fx - ix, ty = fy - iy, tz = fz - iz;
                    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty); tz = tz * tz * (3 - 2 * tz);
                    float v = 0;
                    for (int k = 0; k < 8; k++) {
                        int ax = (ix + (k & 1)) % L, ay = (iy + ((k >> 1) & 1)) % L, az = (iz + (k >> 2)) % L;
                        float w = ((k & 1) ? tx : 1 - tx) * ((k & 2) ? ty : 1 - ty) * ((k & 4) ? tz : 1 - tz);
                        v += w * g[(az * L + ay) * L + ax];
                    }
                    d[(((size_t)z * N + y) * N + x) * 4 + c] = (unsigned char)(v * 255.0f + 0.5f);
                }
        free(g);
    }
    for (int i = 0; i < N * N * N; i++) d[(size_t)i * 4 + 3] = (unsigned char)(hash3((uint32_t)i, 950, 2) >> 24);
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_3D, tex_noise);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, N, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, d);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    free(d);
}

static GLuint mk_tex(int w, int h, GLenum ifmt, GLenum fmt, GLenum type, GLenum filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static GLuint mk_fbo(GLuint color, GLuint depth) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    if (color) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    if (depth) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    if (!color) { glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE); }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete");
    return f;
}

/* the internal resolution: about two megapixels unless --scale says otherwise */
static void targets(void) {
    float sc = opt_scale > 0 ? opt_scale : fminf(1.0f, sqrtf(0.95e6f / ((float)W * H)));
    int w = (int)(W * sc + 0.5f), h = (int)(H * sc + 0.5f);
    if (w == rt_w && h == rt_h && tex_scene) return;
    if (tex_scene) {
        if (fbo_ms) {
            glDeleteFramebuffers(1, &fbo_ms);
            glDeleteRenderbuffers(1, &rb_ms_col);
            glDeleteRenderbuffers(1, &rb_ms_dep);
            fbo_ms = 0;
        }
        GLuint t[] = {tex_scene, tex_depth, tex_copy, tex_vol, tex_prep, tex_dof};
        GLuint f[] = {fbo_scene, fbo_copy, fbo_vol, fbo_prep, fbo_dof};
        glDeleteTextures(6, t);
        glDeleteFramebuffers(5, f);
        glDeleteTextures(NBLOOM, tex_bloom);
        glDeleteFramebuffers(NBLOOM, fbo_bloom);
    }
    rt_w = w; rt_h = h; iw = w; ih = h;
    tex_scene = mk_tex(iw, ih, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_LINEAR);
    tex_depth = mk_tex(iw, ih, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_NEAREST);
    fbo_scene = mk_fbo(tex_scene, tex_depth);
    tex_copy = mk_tex(iw, ih, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_LINEAR);
    fbo_copy = mk_fbo(tex_copy, 0);
    if (opt_msaa > 1) {
        glGenRenderbuffers(1, &rb_ms_col);
        glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_col);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, opt_msaa, GL_R11F_G11F_B10F, iw, ih);
        glGenRenderbuffers(1, &rb_ms_dep);
        glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_dep);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, opt_msaa, GL_DEPTH_COMPONENT24, iw, ih);
        glGenFramebuffers(1, &fbo_ms);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_ms_col);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_ms_dep);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("multisampled framebuffer incomplete");
    }
    tex_vol = mk_tex((iw + 3) / 4, (ih + 3) / 4, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR);
    fbo_vol = mk_fbo(tex_vol, 0);
    tex_prep = mk_tex((iw + 1) / 2, (ih + 1) / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR);
    fbo_prep = mk_fbo(tex_prep, 0);
    tex_dof = mk_tex((iw + 1) / 2, (ih + 1) / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR);
    fbo_dof = mk_fbo(tex_dof, 0);
    int x = (iw + 1) / 2, y = (ih + 1) / 2;
    for (int i = 0; i < NBLOOM; i++) {
        x = (x + 1) / 2; y = (y + 1) / 2;
        if (x < 1) x = 1;
        if (y < 1) y = 1;
        bw[i] = x; bh[i] = y;
        tex_bloom[i] = mk_tex(x, y, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_LINEAR);
        fbo_bloom[i] = mk_fbo(tex_bloom[i], 0);
    }
}

static void attribs_scene(void) {
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(6 * sizeof(float)));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(8 * sizeof(float)));
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(9 * sizeof(float)));
    for (int i = 0; i < 5; i++) glEnableVertexAttribArray((GLuint)i);
}
#define GF 12                          /* glass vertex: pos, normal, s u T wall, side piece */
static void attribs_glass(void) {
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, GF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, GF * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, GF * sizeof(float), (void *)(6 * sizeof(float)));
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, GF * sizeof(float), (void *)(10 * sizeof(float)));
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray((GLuint)i);
}

static void gl_init(void) {
    p_scene = mk_prog(VS_SCENE, FS_SCENE);
    p_shadow = mk_prog(VS_SHADOW, FS_SHADOW);
    p_glass = mk_prog(VS_GLASS, FS_GLASS);
    p_spark = mk_prog(VS_SPARK, FS_SPARK);
    p_vol = mk_prog(VS_FULL, FS_VOL);
    p_prep = mk_prog(VS_FULL, FS_PREP);
    p_dof = mk_prog(VS_FULL, FS_DOF);
    p_down = mk_prog(VS_FULL, FS_DOWN);
    p_up = mk_prog(VS_FULL, FS_UP);
    p_final = mk_prog(VS_FULL, FS_FINAL);
    p_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_full);

    glGenVertexArrays(1, &vao_stat);
    glGenBuffers(1, &vbo_stat);
    glBindVertexArray(vao_stat);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_stat);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)shop.n * VF * sizeof(float)), shop.v, GL_STATIC_DRAW);
    attribs_scene();

    glGenVertexArrays(1, &vao_dyn);
    glGenBuffers(1, &vbo_dyn);
    glBindVertexArray(vao_dyn);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_dyn);
    attribs_scene();

    glGenVertexArrays(1, &vao_glass);
    glGenBuffers(1, &vbo_glass);
    glGenBuffers(1, &ebo_glass);
    glBindVertexArray(vao_glass);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_glass);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_glass);
    attribs_glass();

    glGenVertexArrays(1, &vao_gal);
    glGenBuffers(1, &vbo_gal);
    glGenBuffers(1, &ebo_gal);
    glBindVertexArray(vao_gal);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_gal);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_gal);
    attribs_glass();

    glGenVertexArrays(1, &vao_spark);
    glGenBuffers(1, &vbo_spark);
    glBindVertexArray(vao_spark);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_spark);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(5 * sizeof(float)));
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);

    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);
    glGenTextures(1, &tex_atlas);

    make_noise();

    tex_shadow = mk_tex(SHADOW_N, SHADOW_N, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    fbo_shadow = mk_fbo(0, tex_shadow);
}

/* ---- one moment of the shop ---- */
static V3 glow_c(float T) {
    const float lam[3] = {0.610f, 0.550f, 0.465f};
    float b[3];
    double ref = 1.0 / (pow(0.61, 5) * (exp(14388.0 / (0.61 * 1400.0)) - 1));
    for (int i = 0; i < 3; i++) b[i] = (float)(1.0 / (pow(lam[i], 5) * (exp(14388.0 / (lam[i] * fmax(T, 300))) - 1)) / ref);
    float m = fmaxf(b[0], 1e-20f), s = powf(m, 0.42f) / m;
    return v3(b[0] * s, b[1] * s, b[2] * s);
}

typedef struct {
    double t; int k; float c, lt; int stage;
    PieceDef def; Prof prof; float h, rmax, T;
    Pose pose;
    float model[16];                   /* the piece: lathe space (y up the axis) to world */
    V3 pc, ax;                         /* its centre, and the axis away from the rod */
    int show;                          /* drawn at all */
    float inside;                      /* how far into a fire it is (its own light hidden) */
} Frame;

/* the rod's tip and the piece's frame, without the gaffer's limbs */
static void place_piece(Frame *F) {
    Pose *p = &F->pose;
    V3 ax = vmul(p->back, -1);
    F->ax = ax;
    V3 y = p->attach == 1 ? ax : vmul(ax, -1);
    V3 o = p->attach == 1 ? vadd(p->tip, vmul(ax, 0.004f)) : vadd(p->tip, vmul(ax, F->h + 0.004f));
    V3 u, v;
    basis(y, &u, &v);
    float cs = cosf(p->spin), sn = sinf(p->spin);
    V3 X = vadd(vmul(u, cs), vmul(v, sn)), Z = vcross(X, y);
    float m[16] = {X.x, X.y, X.z, 0, y.x, y.y, y.z, 0, Z.x, Z.y, Z.z, 0, o.x, o.y, o.z, 1};
    memcpy(F->model, m, sizeof m);
    F->pc = vadd(o, vmul(y, F->h * 0.5f));
}

static void sim(double t, Frame *F) {
    memset(F, 0, sizeof *F);
    F->t = t;
    F->k = (int)floor(t / CYCLE);
    F->c = (float)(t - F->k * CYCLE);
    F->stage = stage_at(F->c, &F->lt);
    F->def = piece_def(F->k);
    piece_prof(&F->def, F->c, &F->prof);
    F->h = prof_height(&F->prof);
    F->rmax = prof_rmax(&F->prof);
    F->T = core_temp(F->c);
    pose_at(F->c, F->h, F->rmax, &F->pose);
    if (F->pose.punty && st_place[F->stage] == P_BENCH) {
        /* on the punty the piece sits nearer him, so he can reach the lip */
        F->pose.tip.x += 0.12f;
    }
    if (F->stage == S_ANNEAL && F->lt > 5.0f) {
        /* left in the annealer where it was set down */
        Frame G;
        float c = st_t0[S_ANNEAL] + 5.0f;
        G.def = F->def; piece_prof(&G.def, c, &G.prof);
        G.h = prof_height(&G.prof);
        pose_at(c, G.h, prof_rmax(&G.prof), &G.pose);
        G.pose.attach = 1;
        place_piece(&G);
        memcpy(F->model, G.model, sizeof G.model);
        F->pc = G.pc; F->ax = G.ax;
    } else {
        place_piece(F);
    }
    F->show = !(F->stage == S_TO_FURNACE || (F->stage == S_ANNEAL && F->lt > 9.5f));
    /* in the glory hole or the furnace: its glow is lost in theirs */
    V3 gm = gh_mouth();
    float in_gh = sstep(gm.x + 0.05f, gm.x - 0.1f, F->pc.x) * (fabsf(F->pc.z - gm.z) < 0.4f);
    float in_f = sstep(FUR_Z + 0.05f, FUR_Z - 0.1f, F->pc.z);
    float in_a = F->stage == S_ANNEAL ? sstep(AN_Z1 + 0.05f, AN_Z1 - 0.1f, F->pc.z) : 0;
    F->inside = fmaxf(fmaxf(in_gh, in_f), in_a);
}

/* the piece's centre, averaged over the last second: what the camera follows */
static V3 follow(double t) {
    V3 a = v3(0, 0, 0);
    for (int i = 0; i < 6; i++) {
        Frame F;
        sim(t - i * 0.2, &F);
        a = vadd(a, F.pc);
    }
    return vmul(a, 1.0f / 6);
}

/* ---- the gaffer's body, the rod, the tools: rebuilt every frame ---- */
static Mesh dyn;

/* two-bone IK: the joint between a and the target, bending towards pole */
static V3 ik(V3 a, V3 tgt, float l1, float l2, V3 pole, V3 *end) {
    V3 d = vsub(tgt, a);
    float L = vlen(d);
    float Lm = (l1 + l2) * 0.999f;
    if (L > Lm) { d = vmul(d, Lm / L); L = Lm; }
    if (L < fabsf(l1 - l2) + 1e-3f) L = fabsf(l1 - l2) + 1e-3f;
    V3 dn = vnorm(d);
    float x = (l1 * l1 - l2 * l2 + L * L) / (2 * L);
    float y = sqrtf(fmaxf(l1 * l1 - x * x, 0));
    V3 pp = vsub(pole, vmul(dn, vdot(pole, dn)));
    pp = vnorm(pp);
    *end = vadd(a, vmul(dn, L));
    return vadd(vadd(a, vmul(dn, x)), vmul(pp, y));
}

static float rmax_at_s(const Prof *pr, float a_from_end, float h) {
    /* the radius a distance in from the lip end */
    float best = 0.02f, bd = 1e9f;
    for (int i = 0; i < NP; i++) {
        float d = fabsf(pr->a[i] - (h - a_from_end));
        if (d < bd && i > NP / 2) { bd = d; best = pr->r[i]; }
    }
    return best;
}

static void build_dyn(const Frame *F) {
    Mesh *m = &dyn;
    m->n = 0;
    cur_seed = 200;
    const Pose *p = &F->pose;
    V3 up = v3(0, 1, 0), f = fwd(p->yaw), r = vcross(f, up);
    /* the rod */
    V3 rend = vadd(p->tip, vmul(p->back, p->rodlen));
    if (F->stage != S_TO_FURNACE || F->lt > 0) {
        if (!p->punty) {
            cyl(m, p->tip, rend, PIPE_R, 10, 0, M_PIPE);
            rod(m, vadd(rend, vmul(p->back, -0.12f)), vadd(rend, vmul(p->back, 0.0f)), PIPE_R * 1.5f, 10, M_PIPE);
            disc(m, p->tip, vmul(p->back, -1), PIPE_R * 0.4f, PIPE_R, 10, M_PIPE);
        } else {
            rod(m, p->tip, rend, PUNTY_R, 8, M_PIPE);
        }
    }

    /* jacks */
    int jacks = 0;
    V3 jt[2], jd = v3(0, 0, 1), hand_r = p->hand_r;
    float lt = F->lt;
    if (F->stage == S_SHAPE) {
        float on = sstep(17, 19, lt) * (1 - sstep(37.5f, 39.5f, lt));
        if (on > 0.01f) {
            jacks = 1;
            V3 ax = F->ax;
            V3 neck = vadd(p->tip, vmul(ax, 0.016f + 0.03f * (1 - on)));
            float rn = rmax_at_s(&F->prof, 0.016f, F->h);
            jd = vnorm(v3(-0.25f, -0.5f, 0.8f));
            V3 o = vnorm(vcross(jd, ax));
            float sq = rn * (1.0f + 0.25f * (0.5f + 0.5f * sinf(lt * 2.3f))) + 0.004f + 0.05f * (1 - on);
            jt[0] = vadd(vadd(neck, vmul(o, sq)), vmul(jd, 0.03f));
            jt[1] = vadd(vsub(neck, vmul(o, sq)), vmul(jd, 0.03f));
            hand_r = vsub(neck, vmul(jd, 0.30f + 0.15f * (1 - on)));
        }
    } else if (F->stage == S_OPEN) {
        float on = sstep(1, 3, lt) * (1 - sstep(27.5f, 29.5f, lt));
        if (on > 0.01f) {
            jacks = 1;
            V3 ax = F->ax;
            int i = NP - 1;
            V3 lc = vadd(p->tip, vmul(ax, F->h - 0.004f));
            jd = vnorm(v3(0.35f, -0.55f, 0.75f));
            V3 o = vnorm(vcross(jd, ax));
            float rl = F->prof.r[i] - F->prof.w[i] - 0.003f;
            float sp = rl * (0.82f + 0.12f * (0.5f + 0.5f * sinf(lt * 1.9f)));
            V3 in = vsub(lc, vmul(ax, 0.025f));
            jt[0] = vadd(in, vmul(o, sp * on));
            jt[1] = vsub(in, vmul(o, sp * on));
            hand_r = vsub(vmul(vadd(jt[0], jt[1]), 0.5f), vmul(jd, 0.32f + 0.2f * (1 - on)));
        }
    }

    /* the body */
    float hipy = mixf(0.93f, SEAT.y + 0.1f, p->seated);
    float bob = p->gait * 0.02f * fabsf(sinf(p->walk));
    V3 lean_side = v3(0, 0, 0);
    if (F->stage == S_OPEN || F->stage == S_SHAPE) lean_side = vmul(r, jacks ? 0.12f : 0.0f);
    V3 pelvis = vadd(p->root, v3(0, hipy + bob, 0));
    V3 tup = vnorm(vadd(vadd(up, vmul(f, p->lean + (jacks ? 0.15f : 0))), lean_side));
    V3 chest = vadd(pelvis, vmul(tup, 0.44f));
    V3 neck = vadd(chest, vmul(tup, 0.09f));
    V3 head = vadd(vadd(neck, vmul(tup, 0.12f)), vmul(f, 0.03f));
    V3 look = vnorm(vsub(F->pc, head));
    V3 lookh = vnorm(vadd(vmul(f, 1.4f), look));
    V3 sh[2] = {vadd(vsub(chest, vmul(tup, 0.03f)), vmul(r, -0.185f)), vadd(vsub(chest, vmul(tup, 0.03f)), vmul(r, 0.185f))};
    V3 hip[2] = {vadd(pelvis, vmul(r, -0.095f)), vadd(pelvis, vmul(r, 0.095f))};
    /* hands: [0] left, [1] right */
    V3 hand[2];
    hand[0] = vadd(p->tip, vmul(p->back, p->dl));
    hand[1] = jacks ? hand_r : p->dr >= 0 ? vadd(p->tip, vmul(p->back, p->dr)) : hand_r;
    if (p->hose > 0.5f && F->stage == S_BLOW) hand[1] = vadd(F->pc, vadd(vmul(F->ax, 0.0f), v3(0.05f, 0.12f, 0.12f)));
    /* the hands sit on top of the rod */
    for (int i = 0; i < 2; i++) if (!(i == 1 && (jacks || p->dr < 0))) hand[i] = vadd(hand[i], v3(0, 0.03f, 0));
    cur_seed = 210;
    for (int i = 0; i < 2; i++) {
        V3 wrist, pole = vadd(vadd(vmul(up, -1), vmul(r, i ? 0.7f : -0.7f)), vmul(f, -0.4f));
        V3 hdir = vnorm(vsub(hand[i], sh[i]));
        V3 tg = vsub(hand[i], vmul(hdir, 0.06f));
        V3 el = ik(sh[i], tg, 0.30f, 0.28f, pole, &wrist);
        capsule(m, sh[i], el, 0.058f, 0.047f, 10, M_CLOTH);
        capsule(m, el, wrist, 0.047f, 0.036f, 10, M_CLOTH);
        V3 fa = vnorm(vsub(wrist, el)), u2, v2;
        basis(fa, &u2, &v2);
        /* gloved fists */
        ellipsoid(m, vadd(wrist, vmul(fa, 0.045f)), fa, u2, v2, 0.05f, 0.042f, 0.036f, 10, M_LEATHER);
    }
    /* legs */
    for (int i = 0; i < 2; i++) {
        float side = i ? 1.0f : -1.0f;
        V3 foot;
        if (p->seated > 0.5f) {
            foot = vadd(vadd(p->root, vmul(r, side * 0.16f)), vadd(vmul(f, 0.5f), v3(0, 0.09f, 0)));
        } else {
            float ph = p->walk + (i ? PI : 0);
            float sw = p->gait * 0.24f * sinf(ph);
            float lift = p->gait * 0.08f * fmaxf(0, cosf(ph));
            foot = vadd(vadd(p->root, vmul(r, side * 0.12f)), vadd(vmul(f, sw + 0.03f), v3(0, 0.09f + lift, 0)));
        }
        if (p->seated > 0.02f && p->seated < 0.98f) {
            V3 fs = vadd(vadd(p->root, vmul(r, side * 0.16f)), vadd(vmul(f, 0.5f), v3(0, 0.09f, 0)));
            V3 fd = vadd(vadd(p->root, vmul(r, side * 0.12f)), v3(0, 0.09f, 0));
            foot = vlerp(fd, fs, p->seated);
        }
        V3 ankle, pole = vadd(f, vmul(up, 0.3f));
        V3 knee = ik(hip[i], foot, 0.45f, 0.44f, pole, &ankle);
        capsule(m, hip[i], knee, 0.085f, 0.06f, 10, M_CLOTH);
        capsule(m, knee, ankle, 0.058f, 0.045f, 10, M_CLOTH);
        ellipsoid(m, vadd(ankle, vadd(vmul(f, 0.06f), v3(0, -0.045f, 0))), f, up, r, 0.13f, 0.05f, 0.055f, 10, M_LEATHER);
    }
    /* torso, shoulders, neck, head */
    cur_seed = 220;
    capsule(m, pelvis, chest, 0.15f, 0.165f, 14, M_CLOTH);
    capsule(m, sh[0], sh[1], 0.075f, 0.075f, 10, M_CLOTH);
    capsule(m, neck, vadd(neck, vmul(tup, 0.08f)), 0.05f, 0.048f, 10, M_SKIN);
    {
        V3 hf_ = vnorm(vsub(lookh, vmul(tup, vdot(lookh, tup)))), hr = vnorm(vcross(hf_, tup)), hu = vcross(hr, hf_);
        hf_ = vnorm(vadd(hf_, vmul(hu, -0.3f * vdot(look, up))));
        hu = vcross(hr, hf_);
        ellipsoid(m, head, hr, hu, hf_, 0.078f, 0.105f, 0.095f, 16, M_SKIN);
        ellipsoid(m, vadd(head, vadd(vmul(hu, 0.03f), vmul(hf_, -0.02f))), hr, hu, hf_, 0.084f, 0.09f, 0.095f, 14, M_HAIR);
        ellipsoid(m, vadd(head, vadd(vmul(hu, -0.06f), vmul(hf_, 0.035f))), hr, hu, hf_, 0.068f, 0.06f, 0.07f, 12, M_HAIR);
        /* safety glasses: a dark band with a glint */
        ellipsoid(m, vadd(head, vadd(vmul(hu, 0.018f), vmul(hf_, 0.07f))), hr, hu, hf_, 0.075f, 0.018f, 0.035f, 12, M_POLISH);
        if (p->hose > 0.3f) {
            /* the blow hose from the end of the pipe to his mouth */
            V3 mouth = vadd(head, vadd(vmul(hu, -0.04f), vmul(hf_, 0.09f)));
            V3 a = rend, b = vadd(rend, vmul(p->back, 0.25f)), c = vadd(mouth, v3(0, -0.45f, 0.1f)), d = mouth;
            V3 prev = a;
            cur_seed = 230;
            for (int i = 1; i <= 12; i++) {
                float s = i / 12.0f, q = 1 - s;
                V3 pt = vadd(vadd(vmul(a, q * q * q), vmul(b, 3 * q * q * s)), vadd(vmul(c, 3 * q * s * s), vmul(d, s * s * s)));
                capsule(m, prev, pt, 0.009f, 0.009f, 6, M_HOSE);
                prev = pt;
            }
        }
    }
    /* the apron, over the chest and down the thighs */
    {
        V3 knees = vmul(vadd(vadd(hip[0], hip[1]), vmul(f, 0.0f)), 0.5f);
        V3 a0 = vadd(vsub(chest, vmul(tup, 0.05f)), vmul(f, 0.135f));
        V3 a1 = vadd(pelvis, vmul(f, 0.14f));
        V3 kn = p->seated > 0.5f ? vadd(vadd(knees, vmul(f, 0.46f)), v3(0, 0.07f, 0)) : vadd(vadd(knees, vmul(f, 0.13f)), v3(0, -0.42f, 0));
        cur_seed = 240;
        beam(m, a0, a1, 0.13f, 0.006f, f, M_LEATHER);
        beam(m, a1, kn, 0.17f, 0.006f, f, M_LEATHER);
        if (p->seated > 0.5f) beam(m, kn, vadd(kn, v3(0, -0.2f, 0.03f)), 0.17f, 0.006f, f, M_LEATHER);
        (void)knees;
    }
    /* the jacks */
    if (jacks) {
        cur_seed = 250;
        V3 hinge = vadd(hand[1], vmul(jd, 0.07f));
        V3 o = vnorm(vsub(jt[0], jt[1]));
        V3 bowa = vsub(hand[1], vmul(jd, 0.02f));
        for (int i = 0; i < 2; i++) {
            V3 mid = vadd(hinge, vmul(o, (i ? -1 : 1) * 0.015f));
            beam(m, hinge, mid, 0.005f, 0.005f, o, M_POLISH);
            beam(m, mid, jt[i], 0.009f, 0.0025f, o, M_POLISH);
            beam(m, bowa, vadd(hinge, vmul(o, (i ? -1 : 1) * 0.02f)), 0.006f, 0.004f, o, M_STEEL);
        }
    }
    /* the annealer's door */
    cur_seed = 260;
    float dy = p->door * 0.52f * (F->stage == S_ANNEAL);
    box(m, AN_DX0 - 0.03f, AN_DY0 - 0.03f + dy, AN_Z1 + 0.005f, AN_DX1 + 0.03f, AN_DY1 + 0.03f + dy, AN_Z1 + 0.06f, M_ANNEAL);
    box(m, (AN_DX0 + AN_DX1) / 2 - 0.12f, AN_DY0 + 0.05f + dy, AN_Z1 + 0.06f, (AN_DX0 + AN_DX1) / 2 + 0.12f, AN_DY0 + 0.07f + dy, AN_Z1 + 0.1f, M_POLISH);
}

/* ---- glass meshes: a lathe of the profile, outside, lip and inside ---- */
typedef struct { float *v; int n, cap; unsigned *ix; int ni, icap; } GMesh;
static GMesh gact, ggal;

static void gvert(GMesh *g, V3 p, V3 n, float s, float u, float T, float wall, float side, float pid) {
    if (g->n + 1 > g->cap) g->v = realloc(g->v, (size_t)(g->cap = g->cap ? g->cap * 2 : 16384) * GF * sizeof(float));
    float *o = g->v + (size_t)g->n * GF;
    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z;
    o[6] = s; o[7] = u; o[8] = T; o[9] = wall; o[10] = side; o[11] = pid;
    g->n++;
}
static void gidx(GMesh *g, unsigned a) {
    if (g->ni + 1 > g->icap) g->ix = realloc(g->ix, (size_t)(g->icap = g->icap ? g->icap * 2 : 32768) * sizeof(unsigned));
    g->ix[g->ni++] = a;
}

/* xf: where the piece's base sits and its turn about the vertical (NULL: lathe space) */
static void build_glass(GMesh *g, const Prof *pr, int seg, float Tcore, float pid, const float *xf) {
    enum { RK = 6 };
    float ra[2 * NP + RK], rr[2 * NP + RK], na[2 * NP + RK], nr[2 * NP + RK], rs[2 * NP + RK], rw[2 * NP + RK], rside[2 * NP + RK];
    int nring = 0;
    float ta[NP], tr[NP];
    for (int i = 0; i < NP; i++) {
        int i0 = i > 0 ? i - 1 : 0, i1 = i < NP - 1 ? i + 1 : NP - 1;
        float da = pr->a[i1] - pr->a[i0], dr = pr->r[i1] - pr->r[i0], l = hypotf(da, dr);
        if (l < 1e-7f) { da = 0; dr = 1; l = 1; }
        ta[i] = da / l; tr[i] = dr / l;
    }
    /* the inside: offset inwards by the wall, collapsing to the axis where solid */
    float ia[NP], ir[NP];
    int ih_ = -1;
    for (int i = 0; i < NP; i++) if (pr->w[i] > 0 && ih_ < 0) ih_ = i;
    for (int i = 0; i < NP; i++) {
        float w = pr->w[i];
        if (w > 0) { ia[i] = pr->a[i] + tr[i] * w; ir[i] = fmaxf(pr->r[i] - ta[i] * w, 0); }
        else { ia[i] = 0; ir[i] = 0; }
    }
    float a_floor = ih_ >= 0 ? ia[ih_] : pr->a[NP - 1];
    for (int i = 0; i < NP; i++) if (pr->w[i] <= 0 && (ih_ < 0 || i < ih_)) ia[i] = a_floor;
    for (int i = 1; i < NP; i++) if (ia[i] < ia[i - 1] && ir[i] < 0.004f) ia[i] = ia[i - 1];
    /* outside */
    for (int i = 0; i < NP; i++) {
        ra[nring] = pr->a[i]; rr[nring] = pr->r[i]; na[nring] = -tr[i]; nr[nring] = ta[i];
        rs[nring] = (float)i / (NP - 1); rw[nring] = pr->w[i] > 0 ? pr->w[i] : -fmaxf(pr->r[i], 0.002f); rside[nring] = 0;
        nring++;
    }
    /* the lip: a half round from outside to inside */
    {
        float oa = pr->a[NP - 1], orr = pr->r[NP - 1], ia_ = ia[NP - 1], ir_ = ir[NP - 1];
        float ma = (oa + ia_) / 2, mr = (orr + ir_) / 2;
        float ea = oa - ma, er = orr - mr, el = hypotf(ea, er);
        float hw = el;
        if (el < 1e-6f) { ea = -tr[NP - 1]; er = ta[NP - 1]; el = 1; }
        ea /= el; er /= el;
        for (int k = 1; k < RK; k++) {
            float th = PI * k / RK;
            float pa = ma + ea * hw * cosf(th) + ta[NP - 1] * hw * sinf(th);
            float prr = mr + er * hw * cosf(th) + tr[NP - 1] * hw * sinf(th);
            float qa = ea * cosf(th) + ta[NP - 1] * sinf(th), qr = er * cosf(th) + tr[NP - 1] * sinf(th);
            ra[nring] = pa; rr[nring] = fmaxf(prr, 0); na[nring] = qa; nr[nring] = qr;
            rs[nring] = 1; rw[nring] = pr->w[NP - 1] > 0 ? pr->w[NP - 1] : -fmaxf(orr, 0.002f); rside[nring] = 1;
            nring++;
        }
    }
    /* inside, back down */
    for (int i = NP - 1; i >= 0; i--) {
        ra[nring] = ia[i]; rr[nring] = ir[i]; na[nring] = tr[i]; nr[nring] = -ta[i];
        rs[nring] = (float)i / (NP - 1); rw[nring] = pr->w[i] > 0 ? pr->w[i] : -fmaxf(pr->r[i], 0.002f); rside[nring] = 2;
        nring++;
    }
    unsigned base = (unsigned)g->n;
    float c = xf ? cosf(xf[3]) : 1, s_ = xf ? sinf(xf[3]) : 0;
    for (int i = 0; i < nring; i++) {
        float thick = rw[i] > 0 ? rw[i] : -rw[i];
        float T = Tcore - (Tcore - 640) * 0.4f * expf(-thick / 0.006f) - (rside[i] == 1 ? (Tcore - 640) * 0.08f : 0);
        if (Tcore < 700) T = Tcore;
        for (int j = 0; j <= seg; j++) {
            float u = 2 * PI * j / seg, cu = cosf(u), su = sinf(u);
            V3 p = v3(rr[i] * cu, ra[i], rr[i] * su), n = v3(nr[i] * cu, na[i], nr[i] * su);
            if (xf) {
                p = v3(xf[0] + p.x * c - p.z * s_, xf[1] + p.y, xf[2] + p.x * s_ + p.z * c);
                n = v3(n.x * c - n.z * s_, n.y, n.x * s_ + n.z * c);
            }
            gvert(g, p, n, rs[i], u, T, rw[i], rside[i], pid);
        }
    }
    for (int i = 0; i + 1 < nring; i++)
        for (int j = 0; j < seg; j++) {
            unsigned a = base + (unsigned)(i * (seg + 1) + j), b = a + (unsigned)(seg + 1);
            gidx(g, a); gidx(g, b); gidx(g, b + 1);
            gidx(g, a); gidx(g, b + 1); gidx(g, a + 1);
        }
}

/* ---- the gallery: finished pieces on the shelves ---- */
#define PRE 22                         /* pieces already there when the session starts */
#define NSLOT (SH_N * SH_SLOTS)
static int gal_k = -9999, gal_n;
static int gal_pid[48];                /* which piece index each glass pid shows */
static int gal_first[48], gal_count[48];
static V3 gal_c[48];
static float gal_r[48];
static void slot_pos(int slot, float *x, float *y, float *z) {
    static const int order[SH_N] = {1, 2, 0, 3};
    int sh = order[(slot / SH_SLOTS) % SH_N], j = slot % SH_SLOTS;
    int perm = (int)(hash3((uint32_t)sh, 31, 0) % SH_SLOTS);
    j = (j * 3 + perm) % SH_SLOTS;
    float step = (SH_Z1 - SH_Z0 - 0.36f) / (SH_SLOTS - 1);
    *z = SH_Z0 + 0.18f + step * j + 0.06f * (hf((uint32_t)slot, 32, 0) - 0.5f);
    *x = SH_X - SH_D * 0.5f + 0.06f * (hf((uint32_t)slot, 33, 0) - 0.5f);
    *y = shelf_y(sh);
}
static void build_gallery(int k) {
    ggal.n = ggal.ni = 0;
    gal_n = 0;
    int first = -PRE, last = k - 1;
    if (last - first + 1 > NSLOT) first = last - NSLOT + 1;
    for (int j = first; j <= last; j++) {
        PieceDef d = piece_def(j);
        Prof pr;
        prof_final(&d, &pr, 0.0038f);
        int slot = (j + PRE) % NSLOT;
        float xf[4];
        slot_pos(slot, &xf[0], &xf[1], &xf[2]);
        xf[3] = hf((uint32_t)j, 34, 0) * 6.283f;
        gal_pid[1 + gal_n] = j;
        gal_first[gal_n] = ggal.ni;
        build_glass(&ggal, &pr, 28, 300, (float)(1 + gal_n), xf);
        gal_count[gal_n] = ggal.ni - gal_first[gal_n];
        float hgt = prof_height(&pr), rad = prof_rmax(&pr);
        gal_c[gal_n] = v3(xf[0], xf[1] + hgt / 2, xf[2]);
        gal_r[gal_n] = sqrtf(hgt * hgt / 4 + rad * rad);
        gal_n++;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_gal);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)ggal.n * GF * sizeof(float)), ggal.v, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_gal);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)((size_t)ggal.ni * sizeof(unsigned)), ggal.ix, GL_STATIC_DRAW);
    gal_k = k;
}

/* ---- the camera ---- */
typedef struct { V3 eye, at; float vfov, focus, aperture, exposure; } Cam;
enum { C_MACRO, C_BENCH, C_WIDE, C_LOW, C_GLORY, C_RACK, C_FURNACE, C_MARVER, C_ANNEAL, C_HANDS, NCAM };
/* which shots suit which stretch of the cycle: [from, to) seconds, then the choices */
static const struct { float t0, t1; int n; int kinds[5]; } groups[] = {
    {0, 18, 2, {C_FURNACE, C_WIDE}},
    {18, 38, 2, {C_MARVER, C_WIDE}},
    {38, 50, 2, {C_BENCH, C_MACRO}},
    {50, 70, 2, {C_GLORY, C_WIDE}},
    {70, 110, 4, {C_MACRO, C_BENCH, C_RACK, C_MACRO}},
    {110, 128, 2, {C_GLORY, C_WIDE}},
    {128, 134, 1, {C_BENCH}},
    {134, 140, 1, {C_MACRO}},
    {140, 158, 2, {C_GLORY, C_WIDE}},
    {158, 188, 3, {C_MACRO, C_BENCH, C_MACRO}},
    {188, 218, 3, {C_MACRO, C_RACK, C_MACRO}},
    {218, 240, 2, {C_ANNEAL, C_WIDE}},
};
#define NGROUP ((int)(sizeof groups / sizeof groups[0]))

/* the director: which shot is running at cycle time c, its start and length */
static int shot_at(int k, float c, float *s0, float *len, uint32_t *id) {
    for (int g = 0; g < NGROUP; g++) {
        if (c >= groups[g].t1) continue;
        float span_ = groups[g].t1 - groups[g].t0;
        int n = (int)(span_ / 11.0f);
        if (n < 1) n = 1;
        float L = span_ / n;
        int i = (int)((c - groups[g].t0) / L);
        if (i >= n) i = n - 1;
        *s0 = groups[g].t0 + i * L; *len = L;
        *id = hash3((uint32_t)k, (uint32_t)g, (uint32_t)i);
        int pick = (int)(*id % (uint32_t)groups[g].n);
        if (i > 0 && groups[g].n > 1) {
            uint32_t pid = hash3((uint32_t)k, (uint32_t)g, (uint32_t)(i - 1));
            if ((int)(pid % (uint32_t)groups[g].n) == pick) pick = (pick + 1) % groups[g].n;
        }
        return opt_shotlock >= 0 ? opt_shotlock : groups[g].kinds[pick];
    }
    *s0 = 0; *len = 10; *id = 0;
    return C_WIDE;
}

static float hu(uint32_t id, int i) { return hf(id, (uint32_t)i, 4242); }

static void director(double t, const Frame *F, Cam *cm) {
    float s0, len;
    uint32_t id;
    int kind = shot_at(F->k, F->c, &s0, &len, &id);
    float u = F->c - s0, q = u / len;
    V3 pc = follow(t);
    V3 gm = gh_mouth();
    float side = hu(id, 1) < 0.5f ? -1 : 1;
    cm->exposure = 1.0f;
    cm->aperture = 0.006f;
    switch (kind) {
    case C_MACRO: {
        float d = 0.34f + F->rmax * 3.4f;
        int v = (int)(hu(id, 4) * 2);
        /* from beyond the piece, the gaffer a silhouette behind; or square on from the front */
        float th = (v == 0 ? -1.35f + 0.6f * hu(id, 2) : -0.3f + 0.35f * hu(id, 2)) + side * 0.025f * u;
        float el = 0.35f + 0.3f * hu(id, 3);
        cm->eye = vadd(pc, v3(sinf(th) * d, el * d + 0.02f, cosf(th) * d));
        cm->at = vadd(pc, v3(0, -0.01f, 0));
        cm->vfov = 24; cm->aperture = 0.012f; cm->exposure = 0.8f;
        break;
    }
    case C_BENCH: {
        /* over his shoulder, or from the glory hole's side with him against the dark */
        if (hu(id, 5) < 0.0f) {
            cm->eye = v3(SEAT.x + 0.55f + 0.2f * hu(id, 2), 1.55f + 0.15f * hu(id, 3), SEAT.z - 1.25f + 0.1f * q);
            cm->at = vadd(pc, v3(0.05f, -0.05f, 0));
            cm->vfov = 30;
        } else {
            cm->eye = v3(SEAT.x - 2.5f + 0.15f * q, 1.3f + 0.2f * hu(id, 3), SEAT.z + 1.6f + 0.6f * hu(id, 2));
            cm->at = vlerp(v3(SEAT.x, 1.0f, SEAT.z), pc, 0.5f);
            cm->vfov = 32;
        }
        cm->aperture = 0.012f;
        break;
    }
    case C_WIDE: {
        int v = (int)(hu(id, 2) * 3);
        if (v == 0) { cm->eye = v3(4.4f - 0.3f * q, 1.75f, 3.2f - 0.2f * q); cm->at = v3(-0.8f, 1.05f, -1.8f); }
        else if (v == 1) { cm->eye = v3(-4.3f + 0.25f * q, 1.45f, 2.6f); cm->at = v3(0.6f, 1.0f, -1.4f); }
        else { cm->eye = v3(1.2f + 0.4f * q, 2.6f, 3.6f); cm->at = v3(0.2f, 0.85f, -1.6f); }
        cm->vfov = 46; cm->aperture = 0.005f; cm->exposure = 1.0f;
        cm->at = vlerp(cm->at, pc, 0.25f);
        break;
    }
    case C_LOW: {
        cm->eye = vadd(pc, v3(0.95f + 0.2f * hu(id, 2), -0.42f, 0.7f + 0.3f * hu(id, 3) - 0.12f * q));
        if (cm->eye.y < 0.15f) cm->eye.y = 0.15f;
        cm->at = vadd(pc, v3(-0.35f, 0.12f, -0.1f));
        cm->vfov = 36; cm->aperture = 0.014f;
        break;
    }
    case C_GLORY: {
        /* from the far side of the mouth, the gaffer beyond it */
        /* over the gaffer's shoulder, him dark against the fire */
        cm->eye = vadd(gm, v3(2.75f + 0.3f * hu(id, 2) - 0.2f * q, 0.65f + 0.15f * hu(id, 3), 0.85f + 0.35f * hu(id, 4)));
        cm->at = vadd(gm, v3(0.25f, -0.05f, 0.1f));
        cm->vfov = 36; cm->aperture = 0.01f; cm->exposure = 0.85f;
        break;
    }
    case C_RACK: {
        /* from among the finished pieces on the shelves, across the shop to the bench */
        float z = -1.9f + 2.2f * hu(id, 2);
        cm->eye = v3(SH_X - 0.4f, shelf_y(1) + 0.1f + 0.5f * hu(id, 3), z);
        cm->at = pc;
        cm->vfov = 26; cm->aperture = 0.02f; cm->exposure = 1.1f;
        break;
    }
    case C_FURNACE: {
        cm->eye = vadd(PORT_C, v3(1.3f - 0.15f * q, 0.25f + 0.15f * hu(id, 2), 2.0f - 0.2f * q));
        cm->at = vadd(PORT_C, v3(0.1f, -0.1f, 0.6f));
        cm->vfov = 40; cm->aperture = 0.008f; cm->exposure = 0.9f;
        break;
    }
    case C_MARVER: {
        V3 mc = v3((MV_X0 + MV_X1) / 2, MV_Y, (MV_Z0 + MV_Z1) / 2);
        cm->eye = vadd(mc, v3(-0.45f + 0.2f * hu(id, 2), 0.3f, 0.95f - 0.1f * q));
        cm->at = vlerp(mc, pc, 0.7f);
        cm->vfov = 30; cm->aperture = 0.014f;
        break;
    }
    case C_ANNEAL: {
        cm->eye = v3(AN_DX0 - 1.8f - 0.2f * q, 1.55f, AN_Z1 + 3.2f);
        cm->at = v3((AN_DX0 + AN_DX1) / 2, 0.85f, AN_Z1);
        cm->vfov = 40; cm->aperture = 0.008f;
        break;
    }
    case C_HANDS: default: {
        cm->eye = vadd(pc, v3(0.25f + 0.2f * hu(id, 2), 0.35f, 0.75f - 0.08f * q));
        cm->at = vadd(pc, v3(0.12f, 0.03f, -0.05f));
        cm->vfov = 30; cm->aperture = 0.016f; cm->exposure = 0.85f;
        break;
    }
    }
    if (kind == C_MACRO || kind == C_HANDS || kind == C_LOW || kind == C_MARVER)
        cm->exposure *= mixf(1.0f, 0.5f, sstep(900, 1400, F->T) * (1 - F->inside));
    cm->focus = vlen(vsub(pc, cm->eye));
    if (kind == C_MACRO) cm->focus -= F->rmax * 0.6f;   /* on its near face */
    if (kind == C_FURNACE || kind == C_GLORY) cm->focus = vlen(vsub(cm->at, cm->eye));
    if (kind == C_ANNEAL || kind == C_WIDE) cm->focus = vlen(vsub(vlerp(cm->at, pc, 0.6f), cm->eye));
    if (kind == C_RACK) {
        /* rack focus from a piece on the shelf to the work */
        float near_ = 0.55f, far_ = cm->focus;
        float f = sstep(0.35f, 0.55f, q);
        cm->focus = 1 / mixf(1 / near_, 1 / far_, f);
    }
}

/* the other outputs: slow moves of their own */
static void role_cam(double t, const Frame *F, Cam *cm) {
    double per = role == R_GALLERY ? 41.0 : 53.0;
    int n = (int)floor(t / per);
    float q = (float)((t - n * per) / per);
    uint32_t id = hash3((uint32_t)n, 77, (uint32_t)role);
    V3 pc = follow(t);
    cm->exposure = 1.0f;
    if (role == R_GALLERY) {
        float hv = hu(id, 1);
        int v = hv < 0.45f ? 0 : hv < 0.8f ? 1 : 2;
        float zc = SH_Z0 + 0.9f + (SH_Z1 - SH_Z0 - 1.8f) * hu(id, 2);
        if (v == 0) {          /* along a shelf, against the moonlit glass */
            static const int full[3] = {1, 2, 0};
            int sh = full[(int)(hu(id, 3) * 2.99f)];
            cm->eye = v3(SH_X - 0.95f, shelf_y(sh) + 0.2f, zc + 0.55f - 0.5f * q);
            cm->at = v3(SH_X - 0.15f, shelf_y(sh) + 0.1f, zc - 0.45f - 0.5f * q);
            cm->vfov = 34; cm->aperture = 0.012f;
            cm->focus = 1.15f;
        } else if (v == 1) {   /* the whole window, drifting */
            cm->eye = v3(SH_X - 2.9f, 1.55f + 0.25f * q, zc + 0.4f - 0.5f * q);
            cm->at = v3(SH_X - 0.2f, 1.95f, zc - 0.3f - 0.5f * q);
            cm->vfov = 46; cm->aperture = 0.006f;
            cm->focus = 2.8f;
        } else {               /* through the pieces to the work */
            cm->eye = v3(SH_X - 0.3f, shelf_y(1) + 0.12f + 0.5f * hu(id, 3), WIN_Z0 + 0.4f + 0.3f * q);
            cm->at = pc;
            cm->vfov = 30; cm->aperture = 0.014f;
            cm->focus = vlen(vsub(pc, cm->eye));
        }
    } else {
        int v = (int)(hu(id, 1) * 3);
        V3 gm = gh_mouth();
        if (v == 0) { cm->eye = v3(-0.6f - 0.3f * q, 1.55f, 1.9f - 0.2f * q); cm->at = v3(-1.9f, 1.0f, -1.3f); }
        else if (v == 1) { cm->eye = v3(0.9f - 0.2f * q, 1.5f, -1.75f); cm->at = v3(-0.7f, 1.0f, -3.3f); }
        else { cm->eye = v3(-4.7f + 0.3f * q, 1.6f, 0.9f); cm->at = v3(0.6f, 1.0f, -1.1f); }
        cm->vfov = 44; cm->aperture = 0.008f; cm->exposure = 0.95f;
        cm->focus = vlen(vsub(vlerp(gm, pc, 0.5f), cm->eye));
    }
    (void)F;
}

/* ---- embers, sparks, dust ---- */
typedef struct { float *v; int n, cap; } SBuf;
static SBuf sb;
static float view_m[16], proj_m[16], vp_m[16], cam_fpx;
static V3 cam_eye, cam_right, cam_up;

static void spark_quad(V3 a, V3 b, float w, V3 col) {
    /* a streak from a to b, w wide, at least ~1.2 px; its light kept when thinned */
    V3 mid = vmul(vadd(a, b), 0.5f);
    float dist = fmaxf(vlen(vsub(mid, cam_eye)), 0.05f);
    float wmin = 1.2f * dist / cam_fpx;
    if (w < wmin) { col = vmul(col, (w / wmin) * (w / wmin)); w = wmin; }
    V3 ax = vsub(b, a);
    V3 vd = vnorm(vsub(mid, cam_eye));
    V3 axp = vsub(ax, vmul(vd, vdot(ax, vd)));
    float l = vlen(axp);
    V3 dir = l > 1e-5f ? vmul(axp, 1 / l) : cam_up;
    V3 sd = vnorm(vcross(dir, vd));
    /* streaks spread their light over their length */
    col = vmul(col, w / (w + l * 0.5f));
    V3 e0 = vsub(a, vmul(dir, w)), e1 = vadd(b, vmul(dir, w));
    V3 P[8] = {vsub(e0, vmul(sd, w)), vadd(e0, vmul(sd, w)), vsub(a, vmul(sd, w)), vadd(a, vmul(sd, w)),
               vsub(b, vmul(sd, w)), vadd(b, vmul(sd, w)), vsub(e1, vmul(sd, w)), vadd(e1, vmul(sd, w))};
    float UV[8][2] = {{-1, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 0}, {1, 0}, {-1, 1}, {1, 1}};
    static const int T[18] = {0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4, 4, 5, 7, 4, 7, 6};
    if (sb.n + 18 > sb.cap) sb.v = realloc(sb.v, (size_t)(sb.cap = sb.cap ? sb.cap * 2 : 8192) * 8 * sizeof(float));
    for (int i = 0; i < 18; i++) {
        float *o = sb.v + (size_t)sb.n * 8;
        int k = T[i];
        o[0] = P[k].x; o[1] = P[k].y; o[2] = P[k].z; o[3] = UV[k][0]; o[4] = UV[k][1];
        o[5] = col.x; o[6] = col.y; o[7] = col.z;
        sb.n++;
    }
}

static float win_mask_c(float y, float z) {
    float wy = (y - WIN_Y0) / (WIN_Y1 - WIN_Y0), wz = (z - WIN_Z0) / (WIN_Z1 - WIN_Z0);
    if (wy < 0 || wy > 1 || wz < 0 || wz > 1) return 0;
    float gy = fmodf(wy * 4, 1), gz = fmodf(wz * 5, 1);
    return (gy > 0.035f && gy < 0.965f && gz > 0.03f && gz < 0.97f) ? 1.0f : 0.0f;
}
static V3 moon_dir(void) { return vnorm(v3(-0.78f, -0.45f, 0.22f)); }

/* an ember from a source: born at period-spaced times, rising on the hot air */
static void embers(double t, int salt, int n, V3 src, V3 vel, float spread, float life, float T0, float bright) {
    for (int i = 0; i < n; i++) {
        float per = life * (1.0f + hf((uint32_t)i, (uint32_t)salt, 1));
        double ph = hf((uint32_t)i, (uint32_t)salt, 2) * per;
        double cyc = (t + ph) / per;
        int gen = (int)floor(cyc);
        float age = (float)((cyc - gen) * per);
        uint32_t s = (uint32_t)salt * 7919u + (uint32_t)i;
        float L = life * (0.5f + 0.7f * hf(s, (uint32_t)gen, 3));
        if (age > L) continue;
        V3 v0 = vadd(vel, v3((hf(s, gen, 4) - 0.5f) * spread, (hf(s, gen, 5) - 0.3f) * spread, (hf(s, gen, 6) - 0.5f) * spread));
        V3 o = vadd(src, v3((hf(s, gen, 7) - 0.5f) * 0.2f, (hf(s, gen, 8) - 0.5f) * 0.15f, (hf(s, gen, 9) - 0.5f) * 0.2f));
        float k = 1.2f, buoy = 0.25f + 0.25f * hf(s, gen, 10);
        V3 p[2];
        for (int j = 0; j < 2; j++) {
            float a = fmaxf(age - j * 0.035f, 0);
            float e = (1 - expf(-k * a)) / k;
            V3 wob = v3(0.12f * (noise1(a * 1.3f + i, s) - 0.5f), 0, 0.12f * (noise1(a * 1.1f + i, s + 9) - 0.5f));
            p[j] = vadd(vadd(o, vmul(v0, e)), vadd(v3(0, buoy * a, 0), vmul(wob, a)));
        }
        float T = T0 * (0.75f + 0.25f * hf(s, gen, 11)) - age * 220.0f;
        float fl = 0.6f + 0.4f * noise1(age * 9.0f + i, s + 3);
        V3 c = vmul(glow_c(T), bright * fl * sstep(0, 0.15f, age) * (1 - sstep(L * 0.7f, L, age)));
        spark_quad(p[1], p[0], 0.0025f + 0.002f * hf(s, gen, 12), c);
    }
}

static void build_sparks(double t, const Frame *F, float roar) {
    sb.n = 0;
    V3 gm = gh_mouth();
    embers(t, 1, 34, vadd(gm, v3(0.08f, 0.1f, 0)), v3(0.35f, 0.45f, 0), 0.5f, 3.2f, 1500, 30 * roar);
    embers(t, 2, 16, v3(PORT_C.x, PORT_C.y + 0.17f, FUR_Z + 0.08f), v3(0, 0.4f, 0.25f), 0.4f, 3.5f, 1450, 26);
    /* a spray of hot chips as the piece is knocked off the pipe onto the punty */
    float tp = st_t0[S_PUNTY] + 6.0f;
    if (F->c > tp && F->c < tp + 1.2f) {
        float age = F->c - tp;
        for (int i = 0; i < 26; i++) {
            uint32_t s = hash3((uint32_t)F->k, 500, (uint32_t)i);
            V3 v0 = v3((hf(s, 1, 1) - 0.5f) * 2.2f, 0.4f + hf(s, 1, 2) * 1.8f, (hf(s, 1, 3) - 0.5f) * 2.2f);
            V3 p[2];
            for (int j = 0; j < 2; j++) {
                float a = fmaxf(age - j * 0.03f, 0);
                p[j] = vadd(F->pc, vadd(vmul(v0, a), v3(0, -4.9f * a * a, 0)));
            }
            if (p[0].y < 0.01f) continue;
            V3 c = vmul(glow_c(1350 - age * 400), 40 * (1 - sstep(0.6f, 1.2f, age)));
            spark_quad(p[1], p[0], 0.002f, c);
        }
    }
    /* dust motes in the moonlight */
    V3 md = moon_dir();
    for (int i = 0; i < 160; i++) {
        uint32_t s = 7000u + (uint32_t)i;
        /* scattered through the shaft's volume, drifting */
        float d = 0.5f + 5.5f * hf(s, 1, 1);
        float wy = WIN_Y0 + (WIN_Y1 - WIN_Y0) * hf(s, 1, 2), wz = WIN_Z0 + (WIN_Z1 - WIN_Z0) * hf(s, 1, 3);
        V3 p = vadd(v3(5.95f, wy, wz), vmul(md, d));
        float tt = (float)t * 0.05f;
        p = vadd(p, v3(0.3f * (noise1(tt + i, s) - 0.5f), 0.3f * (noise1(tt * 0.8f + i, s + 1) - 0.5f) - fmodf(tt * 0.3f + hf(s, 1, 4), 1.0f) * 0.2f,
                       0.3f * (noise1(tt * 0.9f + i, s + 2) - 0.5f)));
        float sdist = (5.98f - p.x) / -md.x;
        V3 q = vsub(p, vmul(md, sdist));
        float lit = win_mask_c(q.y, q.z);
        float tw = 0.5f + 0.5f * sinf((float)t * (1.0f + hf(s, 1, 5)) + i);
        V3 c = vmul(v3(0.5f, 0.6f, 0.8f), lit * 0.25f * tw);
        if (lit > 0) spark_quad(p, vadd(p, v3(0.0005f, 0, 0)), 0.0012f, c);
    }
}

/* ---- rendering ---- */
static float frame_no;
static double t_start_wall;

static void set_lights(GLuint p, const Frame *F, double t, float roar) {
    V3 gm = gh_mouth();
    float fl = 0.88f + 0.12f * noise1((float)t * 7.0f, 91) + 0.05f * noise1((float)t * 23.0f, 92);
    float I = 15.0f * fl * roar;
    glUniform1f(U(p, "uTime"), (float)fmod(t, 1000.0));
    glUniform3f(U(p, "uEye"), cam_eye.x, cam_eye.y, cam_eye.z);
    glUniform3f(U(p, "uGhPos"), gm.x, gm.y, gm.z);
    glUniform3f(U(p, "uGhAxis"), 1, 0, 0);
    glUniform3f(U(p, "uGhCol"), 1.0f * I, 0.50f * I, 0.19f * I);
    glUniform1f(U(p, "uGhRad"), 2.2f);
    float fi = 4.5f * (0.92f + 0.08f * noise1((float)t * 5.0f, 93));
    glUniform3f(U(p, "uFpPos"), PORT_C.x, PORT_C.y, PORT_C.z);
    glUniform3f(U(p, "uFpAxis"), 0, 0, 1);
    glUniform3f(U(p, "uFpCol"), 1.0f * fi, 0.48f * fi, 0.16f * fi);
    glUniform1f(U(p, "uFpRad"), 3.0f);
    V3 g = glow_c(F->T);
    float hk = F->show ? 0.35f * (F->rmax / 0.05f) * (F->rmax / 0.05f) * (1 - F->inside) : 0;
    glUniform3f(U(p, "uHgPos"), F->pc.x, F->pc.y, F->pc.z);
    glUniform3f(U(p, "uHgCol"), g.x * hk, g.y * hk, g.z * hk);
    glUniform1f(U(p, "uHgRad"), fmaxf(F->rmax * 2.0f, 0.12f));
    V3 md = moon_dir();
    glUniform3f(U(p, "uMoonDir"), md.x, md.y, md.z);
    glUniform3f(U(p, "uMoonCol"), 0.30f * 0.16f, 0.40f * 0.16f, 0.66f * 0.16f);
    glUniform4f(U(p, "uWin"), WIN_Y0, WIN_Z0, WIN_Y1, WIN_Z1);
    glUniform1i(U(p, "uNoise"), 5);
    glUniform1i(U(p, "uShadow"), 6);
}

static float shadow_m[16];

static void project(V3 p, float *x, float *y, float *depth) {
    float w;
    V3 c = mat_apply(vp_m, p, &w);
    *x = (c.x / w) * 0.5f + 0.5f; *y = (c.y / w) * 0.5f + 0.5f;
    V3 vv = mat_apply(view_m, p, NULL);
    *depth = -vv.z;
}

/* the gallery's pieces that are in view */
static void draw_gallery(float tv, float aspect) {
    for (int i = 0; i < gal_n; i++) {
        V3 c = mat_apply(view_m, gal_c[i], NULL);
        float r = gal_r[i], z = -c.z;
        if (z < -r || fabsf(c.x) > z * tv * aspect + r * 1.5f || fabsf(c.y) > z * tv + r * 1.5f) continue;
        glDrawElements(GL_TRIANGLES, gal_count[i], GL_UNSIGNED_INT, (void *)(sizeof(unsigned) * (size_t)gal_first[i]));
    }
}

static int caption_style[2];
static void draw_caption(double wt, const Frame *F);

/* GW_PROF=1: GPU time of each stage, from timestamp queries */
static int prof_on = -1, prof_n;
static double prof_acc[16], prof_cpu;
static const char *prof_name[16];
static GLuint prof_q[16];
static void prof_mark(int i, const char *name) {
    if (prof_on < 0) {
        prof_on = getenv("GW_PROF") != NULL;
        if (prof_on) glGenQueries(16, prof_q);
    }
    if (!prof_on) return;
    glQueryCounter(prof_q[i], GL_TIMESTAMP);
    prof_name[i] = name;
    if (i + 1 > prof_n) prof_n = i + 1;
}
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[16];
    for (int i = 0; i < prof_n; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 1; i < prof_n; i++) prof_acc[i] += (t[i] - t[i - 1]) * 1e-6;
}
static void prof_dump(int frames) {
    if (!prof_on) return;
    double sum = 0;
    for (int i = 1; i < prof_n; i++) { printf("  %-10s %.3f ms\n", prof_name[i], prof_acc[i] / frames); sum += prof_acc[i] / frames; }
    printf("  GPU total  %.3f ms; CPU (sim, meshes) %.3f ms\n", sum, prof_cpu * 1000 / frames);
}

static void render(double t, double wt, GLuint target) {
    double cpu0 = now();
    targets();
    int skip = getenv("GW_SKIP") ? atoi(getenv("GW_SKIP")) : 0;
    Frame F;
    sim(t, &F);
    float roar = 0.85f + 0.4f * cpu_load_s;

    /* the camera */
    Cam cm;
    if (role == R_FRONT) director(t, &F, &cm);
    else role_cam(t, &F, &cm);
    float aspect = (float)W / H;
    float tv = tanf(cm.vfov * 0.5f * PI / 180);
    /* framed for 16:9; narrower frames widen to keep most of the width */
    float th = tv * 16.0f / 9.0f * 0.74f;
    if (th / aspect > tv) tv = th / aspect;
    cam_eye = cm.eye;
    if (getenv("GW_VERBOSE"))
        fprintf(stderr, "t %.2f k %d c %.2f stage %d eye %.2f %.2f %.2f at %.2f %.2f %.2f fov %.1f focus %.2f pc %.2f %.2f %.2f T %.0f h %.3f r %.3f\n",
                t, F.k, F.c, F.stage, cm.eye.x, cm.eye.y, cm.eye.z, cm.at.x, cm.at.y, cm.at.z, cm.vfov, cm.focus, F.pc.x, F.pc.y, F.pc.z, F.T, F.h, F.rmax);
    look_at(view_m, cm.eye, cm.at, v3(0, 1, 0));
    float nearp = 0.03f, farp = 40.0f;
    frustum(proj_m, -tv * aspect * nearp, tv * aspect * nearp, -tv * nearp, tv * nearp, nearp, farp);
    mat_mul(vp_m, proj_m, view_m);
    float inv_vp[16];
    mat_inv(inv_vp, vp_m);
    cam_right = v3(view_m[0], view_m[4], view_m[8]);
    cam_up = v3(view_m[1], view_m[5], view_m[9]);
    cam_fpx = ih / (2 * tv);

    /* geometry that moves */
    build_dyn(&F);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_dyn);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)dyn.n * VF * sizeof(float)), dyn.v, GL_STREAM_DRAW);
    gact.n = gact.ni = 0;
    if (F.show) build_glass(&gact, &F.prof, 56, F.T, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_glass);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)gact.n * GF * sizeof(float)), gact.v, GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_glass);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)((size_t)gact.ni * sizeof(unsigned)), gact.ix, GL_STREAM_DRAW);
    if (gal_k != F.k) build_gallery(F.k);
    build_sparks(t, &F, roar);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_spark);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)sb.n * 8 * sizeof(float)), sb.v, GL_STREAM_DRAW);

    prof_cpu += now() - cpu0;
    prof_mark(0, "");
    /* the glory hole's shadow map */
    {
        V3 gm = gh_mouth();
        float lv[16], lp[16];
        look_at(lv, vadd(gm, v3(0.01f, 0, 0)), vadd(gm, v3(1, 0, 0)), v3(0, 1, 0));
        float n = 0.25f, tt = tanf(66 * PI / 180);
        frustum(lp, -tt * n, tt * n, -tt * n, tt * n, n, 16.0f);
        mat_mul(shadow_m, lp, lv);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
        glViewport(0, 0, SHADOW_N, SHADOW_N);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.5f, 2.0f);
        glUseProgram(p_shadow);
        glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, shadow_m);
        glBindVertexArray(vao_stat);
        glDrawArrays(GL_TRIANGLES, 0, shop.n);
        glBindVertexArray(vao_dyn);
        glDrawArrays(GL_TRIANGLES, 0, dyn.n);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_3D, tex_noise);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE0);

    prof_mark(1, "shadow");
    /* with multisampling, the depth the later passes read is drawn on its own:
     * resolving a multisampled depth buffer costs more than drawing it again */
    if (fbo_ms) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
        glViewport(0, 0, iw, ih);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(p_shadow);
        glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, vp_m);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glBindVertexArray(vao_stat);
        glDrawArrays(GL_TRIANGLES, 0, shop.n);
        glBindVertexArray(vao_dyn);
        glDrawArrays(GL_TRIANGLES, 0, dyn.n);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glBindVertexArray(vao_gal);
        draw_gallery(tv, aspect);
        if (gact.ni) {
            float mvp[16];
            mat_mul(mvp, vp_m, F.model);
            glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, mvp);
            glBindVertexArray(vao_glass);
            glDrawElements(GL_TRIANGLES, gact.ni, GL_UNSIGNED_INT, 0);
        }
        glDisable(GL_CULL_FACE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }

    /* the scene */
    GLuint fbo_draw = fbo_ms ? fbo_ms : fbo_scene;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_draw);
    glViewport(0, 0, iw, ih);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    /* depth first, so the costly shading runs once a pixel */
    glUseProgram(p_shadow);
    glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, vp_m);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glBindVertexArray(vao_stat);
    glDrawArrays(GL_TRIANGLES, 0, shop.n);
    glBindVertexArray(vao_dyn);
    glDrawArrays(GL_TRIANGLES, 0, dyn.n);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_EQUAL);
    glDepthMask(GL_FALSE);
    glUseProgram(p_scene);
    set_lights(p_scene, &F, t, roar);
    glUniformMatrix4fv(U(p_scene, "uVP"), 1, GL_FALSE, vp_m);
    glUniformMatrix4fv(U(p_scene, "uShadowM"), 1, GL_FALSE, shadow_m);
    glUniform3f(U(p_scene, "uRodTip"), F.pose.tip.x, F.pose.tip.y, F.pose.tip.z);
    glUniform3f(U(p_scene, "uRodDir"), F.pose.back.x, F.pose.back.y, F.pose.back.z);
    glUniform1f(U(p_scene, "uRodT"), F.stage == S_TO_FURNACE ? 300 : 300 + (F.T - 300) * 0.72f);
    glUniform3f(U(p_scene, "uAmb"), 0.012f, 0.009f, 0.007f);
    glUniform1i(U(p_scene, "uDbgS"), getenv("GW_DBGS") ? atoi(getenv("GW_DBGS")) : 0);
    glUniform1f(U(p_scene, "uAnnealDoor"), F.stage == S_ANNEAL ? F.pose.door : 0);
    glBindVertexArray(vao_stat);
    glDrawArrays(GL_TRIANGLES, 0, shop.n);
    glBindVertexArray(vao_dyn);
    glDrawArrays(GL_TRIANGLES, 0, dyn.n);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    prof_mark(2, "scene");
    /* the glass: what's behind it first */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_draw);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_copy);
    glBlitFramebuffer(0, 0, iw, ih, 0, 0, iw, ih, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_draw);
    glUseProgram(p_glass);
    set_lights(p_glass, &F, t, roar);
    glUniformMatrix4fv(U(p_glass, "uVP"), 1, GL_FALSE, vp_m);
    glUniformMatrix4fv(U(p_glass, "uView"), 1, GL_FALSE, view_m);
    glUniformMatrix4fv(U(p_glass, "uShadowM"), 1, GL_FALSE, shadow_m);
    glUniform1f(U(p_glass, "uProjY"), proj_m[5]);
    glUniform2f(U(p_glass, "uRes"), (float)iw, (float)ih);
    glUniform1i(U(p_glass, "uScene"), 0);
    {
        float ab[48][3], kd[48][4], th_[48][3];
        int np_ = 1 + gal_n;
        for (int i = 0; i < np_; i++) {
            PieceDef d = i == 0 ? F.def : piece_def(gal_pid[i]);
            memcpy(ab[i], glass_abs[d.glass], sizeof ab[i]);
            memcpy(th_[i], glass_thread[d.glass], sizeof th_[i]);
            kd[i][0] = (float)d.glass; kd[i][1] = d.twist; kd[i][2] = 0; kd[i][3] = 0;
        }
        glUniform3fv(U(p_glass, "uAbs"), np_, &ab[0][0]);
        glUniform4fv(U(p_glass, "uKind"), np_, &kd[0][0]);
        glUniform3fv(U(p_glass, "uThread"), np_, &th_[0][0]);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_copy);
    glEnable(GL_CULL_FACE);
    static const float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int pass = 0; pass < 2; pass++) {
        glUniform1i(U(p_glass, "uPass"), pass);
        glCullFace(pass == 0 ? GL_FRONT : GL_BACK);
        if (pass == 1) { glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_SRC_ALPHA); glDepthFunc(GL_LEQUAL); }
        glUniformMatrix4fv(U(p_glass, "uModel"), 1, GL_FALSE, ident);
        glBindVertexArray(vao_gal);
        draw_gallery(tv, aspect);
        if (gact.ni) {
            glUniformMatrix4fv(U(p_glass, "uModel"), 1, GL_FALSE, F.model);
            glBindVertexArray(vao_glass);
            glDrawElements(GL_TRIANGLES, gact.ni, GL_UNSIGNED_INT, 0);
        }
    }
    glDisable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);

    prof_mark(3, "glass");
    /* embers: additive, behind whatever's in front of them */
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    glUseProgram(p_spark);
    glUniformMatrix4fv(U(p_spark, "uVP"), 1, GL_FALSE, vp_m);
    glBindVertexArray(vao_spark);
    glDrawArrays(GL_TRIANGLES, 0, sb.n);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    if (fbo_ms) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_scene);
        glBlitFramebuffer(0, 0, iw, ih, 0, 0, iw, ih, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    prof_mark(4, "sparks+res");
    /* the haze */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_vol);
    glViewport(0, 0, (iw + 3) / 4, (ih + 3) / 4);
    glUseProgram(p_vol);
    set_lights(p_vol, &F, t, roar);
    glUniformMatrix4fv(U(p_vol, "uShadowM"), 1, GL_FALSE, shadow_m);
    glUniformMatrix4fv(U(p_vol, "uInvVP"), 1, GL_FALSE, inv_vp);
    glUniform1i(U(p_vol, "uDepth"), 0);
    glUniform1f(U(p_vol, "uFog"), 0.016f);
    glUniform1i(U(p_vol, "uSteps"), 14);
    glUniform1f(U(p_vol, "uFrame"), frame_no);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    prof_mark(5, "haze");
    /* heat shimmer sources: the glory hole, the port, the hot glass */
    float heat[4][4] = {{0}}, heatk[4][2] = {{0}};
    {
        V3 gm = gh_mouth();
        V3 src[3] = {vadd(gm, v3(0.1f, 0.12f, 0)), v3(PORT_C.x, PORT_C.y + 0.16f, FUR_Z + 0.05f), vadd(F.pc, v3(0, F.rmax, 0))};
        float hgt[3] = {0.9f, 0.8f, 0.35f}, rad[3] = {0.2f, 0.18f, F.rmax * 1.2f};
        float str[3] = {1.0f, 0.8f, F.show ? sstep(900, 1300, F.T) * (1 - F.inside) * 0.8f : 0};
        for (int i = 0; i < 3; i++) {
            float x0, y0, d0, x1, y1, d1;
            project(src[i], &x0, &y0, &d0);
            project(vadd(src[i], v3(0, hgt[i], 0)), &x1, &y1, &d1);
            if (d0 < 0.05f || y1 - y0 < 0.002f) continue;
            heat[i][0] = x0; heat[i][1] = y0; heat[i][3] = y1 - y0; heat[i][2] = rad[i] / hgt[i] * (y1 - y0);
            heatk[i][0] = str[i] * clampf(0.15f / (y1 - y0), 0.3f, 1.5f); heatk[i][1] = d0;
        }
    }
    /* depth of field: the circle of confusion at half resolution */
    float hw = (iw + 1) / 2, hh = (ih + 1) / 2;
    float fpx = hh / (2 * tv);
    float K = cm.aperture * fpx / fmaxf(cm.focus, 0.05f);
    float cmax = fminf(K * 1.2f, 11.0f * hh / 600.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_prep);
    glViewport(0, 0, (int)hw, (int)hh);
    glUseProgram(p_prep);
    glUniform1f(U(p_prep, "uTime"), (float)fmod(t, 1000.0));
    glUniform1i(U(p_prep, "uNoise"), 5);
    glUniform4fv(U(p_prep, "uHeat"), 4, &heat[0][0]);
    glUniform2fv(U(p_prep, "uHeatK"), 4, &heatk[0][0]);
    glUniform2f(U(p_prep, "uNF"), nearp, farp);
    glUniform1f(U(p_prep, "uAspect"), aspect);
    glUniform3f(U(p_prep, "uDof"), cm.focus, K, cmax);
    glUniform1i(U(p_prep, "uScene"), 0);
    glUniform1i(U(p_prep, "uDepth"), 1);
    glUniform1i(U(p_prep, "uVol"), 2);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_depth);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_vol);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dof);
    if (!(skip & 2)) {
    glUseProgram(p_dof);
    glUniform1i(U(p_dof, "uPrep"), 0);
    glUniform2f(U(p_dof, "uPx"), 1.0f / hw, 1.0f / hh);
    glUniform1f(U(p_dof, "uMax"), fmaxf(cmax, 1.0f));
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_prep);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    prof_mark(6, "dof");
    /* bloom */
    if (!(skip & 1)) {
    glUseProgram(p_down);
    glUniform1i(U(p_down, "uSrc"), 0);
    for (int i = 0; i < NBLOOM; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i]);
        glViewport(0, 0, bw[i], bh[i]);
        int sw = i ? bw[i - 1] : (int)hw, sh_ = i ? bh[i - 1] : (int)hh;
        glUniform2f(U(p_down, "uPx"), 1.0f / sw, 1.0f / sh_);
        glUniform1i(U(p_down, "uFirst"), i == 0);
        glBindTexture(GL_TEXTURE_2D, i ? tex_bloom[i - 1] : tex_prep);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glUseProgram(p_up);
    glUniform1i(U(p_up, "uSrc"), 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    for (int i = NBLOOM - 1; i > 0; i--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i - 1]);
        glViewport(0, 0, bw[i - 1], bh[i - 1]);
        glUniform2f(U(p_up, "uPx"), 1.0f / bw[i], 1.0f / bh[i]);
        glBindTexture(GL_TEXTURE_2D, tex_bloom[i]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glDisable(GL_BLEND);
    }

    prof_mark(7, "bloom");
    /* the final image */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_final);
    glUniform1f(U(p_final, "uTime"), (float)fmod(t, 1000.0));
    glUniform1i(U(p_final, "uNoise"), 5);
    glUniform4fv(U(p_final, "uHeat"), 4, &heat[0][0]);
    glUniform2fv(U(p_final, "uHeatK"), 4, &heatk[0][0]);
    glUniform2f(U(p_final, "uNF"), nearp, farp);
    glUniform1f(U(p_final, "uAspect"), aspect);
    glUniform3f(U(p_final, "uDof"), cm.focus, K, cmax);
    glUniform1f(U(p_final, "uExposure"), cm.exposure * 1.25f);
    glUniform1f(U(p_final, "uBloomK"), 0.05f);
    glUniform1i(U(p_final, "uDbg"), getenv("GW_DEBUG") ? atoi(getenv("GW_DEBUG")) : 0);
    /* a breath of black at each cut */
    float fade = 1;
    if (role == R_FRONT) {
        float s0, len;
        uint32_t id;
        shot_at(F.k, F.c, &s0, &len, &id);
        float u = F.c - s0;
        fade = sstep(0.0f, 0.35f, u) * (1 - 0.0f * sstep(len - 0.3f, len, u));
    }
    glUniform1f(U(p_final, "uFade"), fminf(fade, 1.0f));
    glUniform4f(U(p_final, "uVig"), 0, 0, 1, 1);
    glUniform1i(U(p_final, "uScene"), 0);
    glUniform1i(U(p_final, "uDepth"), 1);
    glUniform1i(U(p_final, "uVol"), 2);
    glUniform1i(U(p_final, "uDofTex"), 3);
    glUniform1i(U(p_final, "uBloom"), 4);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_depth);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_vol);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_dof);
    glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_bloom[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    prof_mark(8, "final");
    if (role == R_FRONT && opt_caption) draw_caption(wt, &F);
    prof_mark(9, "caption");
    prof_collect();
    frame_no += 1;
}

/* the lock caption: the time, how long away, and what's being made */
static int cap_w, cap_h;
static void draw_caption(double wt, const Frame *F) {
    if (cap_w != W || cap_h != H) {
        nstyles = 0;
        float base = fminf(W, H) / 1080.0f;
        caption_style[0] = style(0, 44 * base);
        caption_style[1] = style(1, 21 * base);
        build_atlas();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        cap_w = W; cap_h = H;
    }
    tvn = 0;
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    char a[64], b[160];
    snprintf(a, sizeof a, "%02d:%02d", tm.tm_hour, tm.tm_min);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    char aw[48];
    if (away < 60) snprintf(aw, sizeof aw, "away a moment");
    else if (away < 3600) snprintf(aw, sizeof aw, "away %ld min", away / 60);
    else snprintf(aw, sizeof aw, "away %ld h %02ld min", away / 3600, (away / 60) % 60);
    snprintf(b, sizeof b, "%s   \x7f   no. %d, %s %s", aw, F->k + 1, glass_name[F->def.glass], form_name[F->def.form]);
    float m = fminf(W, H) * 0.045f;
    float y = H - m;
    const float c1[4] = {0.93f, 0.86f, 0.74f, 0.82f}, c2[4] = {0.85f, 0.76f, 0.62f, 0.62f};
    text(caption_style[1], m, y, -1, b, c2, 0.4f);
    text(caption_style[0], m - 2, y - styles[caption_style[1]].px * 1.55f, -1, a, c1, 1.0f);
    glUseProgram(p_text);
    glUniform2f(U(p_text, "uRes"), (float)W, (float)H);
    glUniform1i(U(p_text, "uAtlas"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
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
    /* a resize rebuilds the render targets at the next frame */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "glassworks");
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
        "usage: glassworks [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY]\n"
        "                  [--scale F] [--msaa N] [--no-caption]\n"
        "                  [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                  [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "                  [--role front|gallery|fires] [--cam N]\n");
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
 * gets the director's cut; the rest, left to right, take the gallery and the
 * fires in turn. */
static int pick_role(void) {
    if (span_me < 0) return R_FRONT;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    if (span_me == front) return R_FRONT;
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == front || i == span_me) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && (span[i][1] < span[span_me][1] ||
                                               (span[i][1] == span[span_me][1] && i < span_me)))) rank++;
    }
    return rank % 2 ? R_FIRES : R_GALLERY;
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

/* the CPU load, read once a second (on the wall-clock second when spanning,
 * so every output's fires roar together), and eased */
static double last_read = -1e9;
static void sense(double t, double wt, float dt) {
    if (span_me >= 0 ? floor(wt) != last_read : t - last_read >= 1.0) {
        read_cpu();
        last_read = span_me >= 0 ? floor(wt) : t;
    }
    cpu_load_s += (cpu_load - cpu_load_s) * clampf(dt * 0.8f, 0, 1);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-msaa") && i + 1 < argc) opt_msaa = atoi(argv[++i]);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-cam") && i + 1 < argc) opt_shotlock = atoi(argv[++i]);
        else if (!strcmp(a, "-role") && i + 1 < argc) {
            const char *r = argv[++i];
            opt_role = !strcmp(r, "gallery") ? R_GALLERY : !strcmp(r, "fires") ? R_FIRES : R_FRONT;
        }
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    role = opt_role >= 0 ? opt_role : pick_role();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;

    timeline_init();
    build_shop();
    fonts_init();
    read_cpu();
    {   /* a first CPU-load sample over a tenth of a second */
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
        read_cpu();
        cpu_load_s = cpu_load;
    }

    /* --epoch: when the shop opened, shared by every instance; a --shot then
     * shows the moment --at seconds after it */
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
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Glassworks");
        XClassHint chint = {"glassworks", "Glassworks"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        /* a 3.3 core context, as on the layer surface */
        typedef GLXContext (*CCA)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
        CCA cca = (CCA)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
        int fbattr[] = {GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_DOUBLEBUFFER, True, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
                        GLX_BLUE_SIZE, 8, None};
        int nfb = 0;
        GLXFBConfig *fbc = glXChooseFBConfig(dpy, scr, fbattr, &nfb);
        GLXContext ctx = NULL;
        if (cca && fbc && nfb > 0) {
            int ca[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None};
            ctx = cca(dpy, fbc[0], NULL, True, ca);
        }
        if (!ctx) ctx = glXCreateContext(dpy, vi, NULL, True);
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
            render(opt_at, wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double sum = 0, worst = 0;
        double t0b = opt_at;
        for (int i = 0; i < opt_bench + 3; i++) {
            double f0 = now();
            double t = t0b + i / 60.0;
            render(t, wall(), fbo);
            glFinish();
            double ms = (now() - f0) * 1000;
            if (i < 3) continue;   /* warm-up: the gallery and shaders settle */
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("glassworks: %dx%d (internal %dx%d), %d frames: avg %.3f ms, worst %.3f ms per frame\n",
               W, H, iw, ih, opt_bench, sum / opt_bench, worst);
        printf("glassworks: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        prof_dump(opt_bench + 3);
        if (prof_on) printf("  verts: shop %d dyn %d, glass idx: gallery %d active %d, sparks %d\n", shop.n, dyn.n, ggal.ni, gact.ni, sb.n);
        return 0;
    }

    double t0 = now(), prev = 0;
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
        sense(el, wt, (float)(el - prev));
        prev = el;
        render(el, wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
