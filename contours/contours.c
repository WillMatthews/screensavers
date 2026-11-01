/*
 * contours — a living topographic map: the contour lines of an endless
 * procedural mountain country, drawn like a fine cartographic sheet, drifting
 * slowly past under a moving sun.
 *
 * The ground is flightsim's (domain-warped fbm hills, ridged mountain ranges
 * where a broad mask allows) gentled for a map: gradient noise with its
 * octaves turned (value noise shows its lattice in contour lines), fewer fine
 * octaves, larger forms. Rivers cut valleys through the lower ground along
 * the level sets of a linear function of a gently warped position, so they
 * wind but never close into loops, and run from lake to lake; the water is
 * everything below datum. The ground is baked on the GPU into a float height
 * texture one screen pixel per texel, addressed toroidally: as the view
 * drifts only the strips of new ground that slide into view are baked, so a
 * frame costs one cheap shading pass however heavy the terrain is.
 *
 * Every pixel then draws, from the height and its central differences:
 *   - contours every --interval metres, index contours every fifth, as
 *     constant-width anti-aliased lines (distance to the nearest level =
 *     |fract(h / I) - 1/2| * I / |grad h|), thin ones fading where they crowd;
 *   - a soft hillshade from the north-west, the sun swaying a little;
 *   - water below datum (which rises and falls a few metres, so the shores
 *     breathe), with engraved waterlines following the shore outwards;
 *   - rivers along their valleys, widening downstream; woodland stipple;
 *   - a faint kilometre grid.
 * On the CPU the same terrain finds summits (spot heights with triangles)
 * and straight stretches of index contour to carry their height, set along
 * the line, readable side up, knocking a gap in the lines beneath. Round it:
 * a margin with a zebra border and grid numbers, and a title cartouche (the
 * host and the time, the contour interval, north arrow and scale bar).
 *
 * --seed N picks the country and the drift, --light prints it on paper,
 * --interval sets the contour interval (metres, default 20), --speed scales
 * the drift (~6 px/s at 1080p) and --fps the frame rate (default 30: the
 * motion is slow, and one instance runs per monitor on the lock screen).
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen;
 * --bench N times N offscreen frames at --size (with --shot, it saves the
 * last of them: the incrementally baked frame).
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
static int opt_root, opt_layer, opt_light, opt_bench, opt_fps = 30;
static float opt_interval = 20;        /* metres between contours */
static float opt_speed = 1;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds in */
static int off_w = 1920, off_h = 1080;

static float S = 1;                    /* UI scale: 1 at 1920x1080 */
static float MPP = 8;                  /* metres per pixel */
static float seed_x, seed_z;           /* where in the world this map is, metres */
static int E0, N0;                     /* grid numbering offsets */
static double ph[6];                   /* phases of the drift */

static void die(const char *what) { fprintf(stderr, "contours: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- noise and terrain (flightsim's, ported 1:1 to GLSL below) ---- */
static uint32_t hash2(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
static float smooth(float t) { return t * t * (3 - 2 * t); }
static float smoothstep(float a, float b, float x) {
    float t = (x - a) / (b - a);
    return smooth(t < 0 ? 0 : t > 1 ? 1 : t);
}
/* gradient noise, ~[0, 1]: smoother level sets than flightsim's value noise,
 * which on a map shows its lattice as square-shouldered blobs */
static float grad_dot(int32_t ix, int32_t iz, uint32_t salt, float dx, float dz) {
    uint32_t h = hash2(ix, iz, salt);
    return ((h & 0xffff) / 32768.0f - 1) * dx + ((h >> 16) / 32768.0f - 1) * dz;
}
static float vnoise(float x, float z, uint32_t salt) {
    float fx = floorf(x), fz = floorf(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    float rx = x - fx, rz = z - fz;
    float tx = rx * rx * rx * (rx * (rx * 6 - 15) + 10), tz = rz * rz * rz * (rz * (rz * 6 - 15) + 10);
    float a = grad_dot(ix, iz, salt, rx, rz), b = grad_dot(ix + 1, iz, salt, rx - 1, rz);
    float c = grad_dot(ix, iz + 1, salt, rx, rz - 1), d = grad_dot(ix + 1, iz + 1, salt, rx - 1, rz - 1);
    return 0.5f + (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}
/* octaves turned against each other, so no lattice lines up */
#define ROT(x, z, k) do { float x_ = (0.8f * x - 0.6f * z) * k; z = (0.6f * x + 0.8f * z) * k; x = x_; } while (0)
static float fbm(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        s += vnoise(x, z, salt + i) * amp;
        norm += amp; amp *= 0.47f; ROT(x, z, 2.03f);
    }
    return s / norm;
}
static float ridged(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        float n = fmaxf(0, 1 - fabsf(vnoise(x, z, salt + i) * 2 - 1));
        s += n * n * amp;
        norm += amp; amp *= 0.42f; ROT(x, z, 2.07f);
    }
    return s / norm;
}

/* The ground at (x, z) metres: out = height (m, 0 = datum), river field
 * (rivers run along its zero set), woodland, and how far below the river
 * sources this is. Keep in step with terrain() in FS_BAKE. */
static void terrain(float x, float z, float out[4]) {
    float wx = fbm(x / 6000, z / 6000, 3, 71) - 0.5f;
    float wz = fbm(x / 6000 + 5.2f, z / 6000 + 1.3f, 3, 72) - 0.5f;
    float X = x + wx * 2000, Z = z + wz * 2000;
    float base = fbm(X / 2800, Z / 2800, 3, 11) * 260;
    float mask = smoothstep(0.42f, 0.66f, fbm(x / 9000, z / 9000, 3, 23));
    float mtn = mask > 0 ? powf(ridged(X / 2600, Z / 2600, 3, 37), 1.4f) * 1000 * mask : 0;
    float h = base + mtn - 110;
    float qx = X + (fbm(X / 4200 + 3.1f, Z / 4200 - 7.7f, 3, 91) - 0.5f) * 2600;
    float qz = Z + (fbm(X / 4200 - 1.9f, Z / 4200 + 4.4f, 3, 92) - 0.5f) * 2600;
    float rv = 0.5f * sinf(3.14159265f * (0.6f * qx + 0.8f * qz) / 6000);
    float ron = (1 - smoothstep(120, 380, h)) * smoothstep(0.38f, 0.5f, fbm(x / 7000 - 4.4f, z / 7000 + 2.9f, 2, 93));
    float carve = (1 - smoothstep(0, 1, fabsf(rv) / 0.05f)) * ron;
    float floor_ = fminf(h, h * 0.6f - 6);
    h = h + (floor_ - h) * carve;
    out[0] = h;
    out[1] = rv;
    out[2] = fbm(x / 1400 + 11.5f, z / 1400 - 3.3f, 3, 55);
    out[3] = ron;
}

/* height at world pixel (x, y) */
static float T(double x, double y) {
    float o[4];
    terrain((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, o);
    return o[0];
}
static void grad(double x, double y, double *gx, double *gy) {
    *gx = 0.5 * (T(x + 1, y) - T(x - 1, y));
    *gy = 0.5 * (T(x, y + 1) - T(x, y - 1));
}

/* the drift: where the middle of the screen is at time t, world px. Three
 * circles of different sizes and periods summed (an epicycle wandering over
 * ~300 km), so the path turns gently, never stops and never repeats. */
static void view_at(double t, double *x, double *y) {
    double mx = 33000 * cos(t / 830 + ph[0]) + 6000 * cos(-t / 377 + ph[1]) + 90000 * cos(t / 5400 + ph[2]);
    double my = 33000 * sin(t / 830 + ph[0]) + 6000 * sin(-t / 377 + ph[1]) + 90000 * sin(t / 6950 + ph[3]);
    *x = mx / MPP; *y = my / MPP;
}
static float water_at(double t) { return 3.5f * (float)sin(t / 170 + ph[4]); }
#define WATER_MAX 3.5f

/* ---- glyph atlas: a few faces at their exact pixel sizes, plus drawn
 * sprites, shelf-packed into one R8 texture ---- */
typedef struct { int x, y, w, h, bx, by; float adv; } Glyph;
typedef struct { Glyph g[96]; float px; } Face;   /* ' '..'~', then 127 = middle dot */
enum { F_NUM, F_SPOT, F_COORD, F_TITLE, F_SUB, F_SMALL, NF };
enum { SP_TRI, SP_ARROW, NSP };
static Face fonts[NF];
static Glyph sprites[NSP];
#define ATW 1024
#define ATH 1024
static unsigned char *atlas;
static int sh_x = 1, sh_y = 1, sh_h;

static void atlas_place(int w, int h, int *x, int *y) {
    if (sh_x + w + 3 > ATW) { sh_x = 1; sh_y += sh_h + 3; sh_h = 0; }
    if (sh_y + h + 3 > ATH) die("glyph atlas full");
    *x = sh_x; *y = sh_y;
    sh_x += w + 3;
    if (h > sh_h) sh_h = h;
}

static void load_font(Face *f, const char *pattern, float px) {
    FcPattern *pat = FcNameParse((const FcChar8 *)pattern);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    FT_Library ft;
    FT_Face face;
    if (FT_Init_FreeType(&ft) || FT_New_Face(ft, (const char *)file, 0, &face)) die("can't load the font");
    FT_Set_Pixel_Sizes(face, 0, (FT_UInt)lroundf(px));
    f->px = px;
    for (int i = 0; i < 96; i++) {
        FT_ULong code = i == 95 ? 0xB7 : (FT_ULong)(32 + i);
        Glyph *g = &f->g[i];
        memset(g, 0, sizeof *g);
        if (FT_Load_Char(face, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
        FT_GlyphSlot s = face->glyph;
        g->adv = s->advance.x / 64.0f;
        g->w = (int)s->bitmap.width; g->h = (int)s->bitmap.rows;
        g->bx = s->bitmap_left; g->by = s->bitmap_top;
        if (!g->w || !g->h) continue;
        atlas_place(g->w, g->h, &g->x, &g->y);
        for (int y = 0; y < g->h; y++)
            memcpy(atlas + (size_t)(g->y + y) * ATW + g->x, s->bitmap.buffer + y * s->bitmap.pitch, (size_t)g->w);
    }
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
}

/* sprites, 4x4 supersampled */
static int in_poly(const float *p, int n, float x, float y) {
    int in = 0;
    for (int i = 0, j = n - 1; i < n; j = i++)
        if ((p[2 * i + 1] > y) != (p[2 * j + 1] > y) &&
            x < (p[2 * j] - p[2 * i]) * (y - p[2 * i + 1]) / (p[2 * j + 1] - p[2 * i + 1]) + p[2 * i])
            in = !in;
    return in;
}
static float edge_dist(const float *p, int n, float x, float y) {
    float best = 1e9f;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        float ax = p[2 * j], ay = p[2 * j + 1], ex = p[2 * i] - ax, ey = p[2 * i + 1] - ay;
        float t = ((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey);
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        float dx = x - ax - t * ex, dy = y - ay - t * ey, d = sqrtf(dx * dx + dy * dy);
        if (d < best) best = d;
    }
    return best;
}
static void make_sprites(void) {
    /* spot-height triangle */
    {
        Glyph *g = &sprites[SP_TRI];
        g->w = (int)lroundf(9 * S) | 1; g->h = (int)lroundf(g->w * 0.88f);
        float p[6] = {g->w * 0.5f, 0.3f, g->w - 0.3f, g->h - 0.3f, 0.3f, g->h - 0.3f};
        atlas_place(g->w, g->h, &g->x, &g->y);
        for (int y = 0; y < g->h; y++)
            for (int x = 0; x < g->w; x++) {
                int c = 0;
                for (int k = 0; k < 16; k++) c += in_poly(p, 3, x + (k % 4 + 0.5f) / 4, y + (k / 4 + 0.5f) / 4);
                atlas[(size_t)(g->y + y) * ATW + g->x + x] = (unsigned char)(c * 255 / 16);
            }
    }
    /* north arrow: a slim needle, the west half solid, the east half outlined */
    {
        Glyph *g = &sprites[SP_ARROW];
        g->w = (int)lroundf(15 * S) | 1; g->h = (int)lroundf(54 * S);
        float cx = g->w * 0.5f, lw = fmaxf(0.9f, 1.0f * S);
        float L[6] = {cx, 0.4f, 0.4f, g->h - 0.4f, cx, g->h * 0.76f};
        float R[6] = {cx, 0.4f, cx, g->h * 0.76f, g->w - 0.4f, g->h - 0.4f};
        atlas_place(g->w, g->h, &g->x, &g->y);
        for (int y = 0; y < g->h; y++)
            for (int x = 0; x < g->w; x++) {
                float c = 0;
                for (int k = 0; k < 16; k++) {
                    float sx = x + (k % 4 + 0.5f) / 4, sy = y + (k / 4 + 0.5f) / 4;
                    if (in_poly(L, 3, sx, sy)) c += 1;
                    else if (in_poly(R, 3, sx, sy)) c += edge_dist(R, 3, sx, sy) < lw ? 1 : 0.1f;
                }
                atlas[(size_t)(g->y + y) * ATW + g->x + x] = (unsigned char)lroundf(c * 255 / 16);
            }
    }
}

static void build_atlas(void) {
    if (!FcInit()) die("fontconfig init failed");
    atlas = calloc((size_t)ATW * ATH, 1);
    load_font(&fonts[F_NUM], "Linux Libertine O,serif", 13 * S);
    load_font(&fonts[F_SPOT], "Linux Libertine O,serif:italic", 13.5f * S);
    load_font(&fonts[F_COORD], "Linux Libertine O,serif", 13 * S);
    load_font(&fonts[F_TITLE], "EB Garamond 12,serif", 25 * S);
    load_font(&fonts[F_SUB], "EB Garamond 12,serif:italic", 15.5f * S);
    load_font(&fonts[F_SMALL], "Linux Libertine O,serif", 12 * S);
    make_sprites();
}

/* ---- overlay geometry: quads of x, y, u, v, r, g, b, a, mode ---- */
typedef struct { float *v; int n, cap; } Vbuf;
static Vbuf ov, mk;                    /* the overlay, the knock-out mask */
static int ov_split;                   /* ov quads before this are clipped to the map */

static void vpush(Vbuf *b, float x, float y, float u, float v, const float c[4], float mode) {
    if (b->n + 9 > b->cap) b->v = realloc(b->v, sizeof(float) * (size_t)(b->cap = b->cap ? b->cap * 2 : 4096));
    float *p = b->v + b->n;
    p[0] = x; p[1] = y; p[2] = u; p[3] = v; p[4] = c[0]; p[5] = c[1]; p[6] = c[2]; p[7] = c[3]; p[8] = mode;
    b->n += 9;
}
/* a quad from corner positions q[8] and uvs t[8] */
static void quad(Vbuf *b, const float *q, const float *t, const float c[4], float mode) {
    static const int idx[6] = {0, 1, 2, 0, 2, 3};
    for (int i = 0; i < 6; i++) vpush(b, q[2 * idx[i]], q[2 * idx[i] + 1], t[2 * idx[i]], t[2 * idx[i] + 1], c, mode);
}
static void rect(Vbuf *b, float x0, float y0, float x1, float y1, const float c[4]) {
    float q[8] = {x0, y0, x1, y0, x1, y1, x0, y1}, t[8] = {0};
    quad(b, q, t, c, 1);
}
/* a glyph or sprite with its bitmap's top-left at local (lx, ly), in a frame
 * at (ox, oy) turned by (co, si) */
static void blit(Vbuf *b, const Glyph *g, float ox, float oy, float co, float si, float lx, float ly, const float c[4]) {
    float q[8], t[8];
    float lxs[4] = {lx, lx + g->w, lx + g->w, lx}, lys[4] = {ly, ly, ly + g->h, ly + g->h};
    float us[4] = {(float)g->x, (float)(g->x + g->w), (float)(g->x + g->w), (float)g->x};
    float vs[4] = {(float)g->y, (float)g->y, (float)(g->y + g->h), (float)(g->y + g->h)};
    for (int i = 0; i < 4; i++) {
        q[2 * i] = ox + lxs[i] * co - lys[i] * si;
        q[2 * i + 1] = oy + lxs[i] * si + lys[i] * co;
        t[2 * i] = us[i] / ATW; t[2 * i + 1] = vs[i] / ATH;
    }
    quad(b, q, t, c, 0);
}
static int gi(unsigned char c) { return c == 127 ? 95 : c >= 32 && c < 127 ? c - 32 : 0; }
static float text_w(const Face *f, const char *s, float track) {
    float w = 0;
    for (const char *p = s; *p; p++) w += f->g[gi((unsigned char)*p)].adv + (p[1] ? track : 0);
    return w;
}
/* the height of a digit, and the offset that centres digits on a point */
static float digit_h(const Face *f) { return (float)f->g['0' - 32].h; }
static float digit_mid(const Face *f) { return f->g['0' - 32].by - f->g['0' - 32].h * 0.5f; }
static float cap_h(const Face *f) { return (float)f->g['H' - 32].by; }

/* text with its anchor (fraction ax of its width along, baseline dy below)
 * at (x, y), turned by ang; upright text is snapped to whole pixels */
static float text(Vbuf *b, const Face *f, const char *s, float x, float y, float ang, float ax, float dy, float track,
                  const float c[4]) {
    float w = text_w(f, s, track), co = cosf(ang), si = sinf(ang), pen = -ax * w;
    if (ang == 0) { x = roundf(x + pen) - pen; y = roundf(y); }
    for (const char *p = s; *p; p++) {
        const Glyph *g = &f->g[gi((unsigned char)*p)];
        float lx = pen + g->bx;
        if (ang == 0) lx = roundf(lx);
        if (g->w) blit(b, g, x, y, co, si, lx, dy - g->by, c);
        pen += g->adv + track;
    }
    return w;
}
/* a soft-edged gap in the lines: a rounded box, half extents hw x hh, turned */
static void knock(float x, float y, float ang, float hw, float hh) {
    float co = cosf(ang), si = sinf(ang), pad = 2;
    float lx[4] = {-hw - pad, hw + pad, hw + pad, -hw - pad}, ly[4] = {-hh - pad, -hh - pad, hh + pad, hh + pad};
    float q[8], t[8], c[4] = {hw, hh, fminf(hh, 3 * S), 1};
    for (int i = 0; i < 4; i++) {
        q[2 * i] = x + lx[i] * co - ly[i] * si;
        q[2 * i + 1] = y + lx[i] * si + ly[i] * co;
        t[2 * i] = lx[i]; t[2 * i + 1] = ly[i];
    }
    quad(&mk, q, t, c, 2);
}

/* ---- features: per world cell, found once and cached ----
 * Each cell may hold a spot height (a summit, found by climbing from the
 * highest of a coarse grid of samples) and a contour label (a point on an
 * index contour where the line runs straight and the lines aren't crowded). */
#define CK 32
typedef struct {
    int cx, cy, ok;
    int lab, lv;
    double lx, ly;
    float la;
    int spot, sv;
    double sx, sy;
} Cell;
static Cell cells[CK * CK];
static float CELL;                     /* cell size, world px */

/* walk p onto the contour at level L along the gradient */
static int to_level(double *x, double *y, float L) {
    for (int i = 0; i < 8; i++) {
        float h = T(*x, *y);
        if (fabsf(h - L) < 0.05f) return 1;
        double gx, gy;
        grad(*x, *y, &gx, &gy);
        double g2 = gx * gx + gy * gy;
        if (g2 < 1e-6) return 0;
        double k = (h - L) / g2;
        if (fabs(k * sqrt(g2)) > 12) k = (k > 0 ? 12 : -12) / sqrt(g2);
        *x -= k * gx; *y -= k * gy;
    }
    return fabsf(T(*x, *y) - L) < 0.3f;
}

static void cell_build(Cell *c, int cx, int cy) {
    memset(c, 0, sizeof *c);
    c->cx = cx; c->cy = cy; c->ok = 1;
    double x0 = cx * (double)CELL, y0 = cy * (double)CELL;
    float I5 = opt_interval * 5;

    /* summit */
    double bx = 0, by = 0;
    float best = -1e9f;
    for (int j = 0; j < 7; j++)
        for (int i = 0; i < 7; i++) {
            double x = x0 + (i + 0.5) * CELL / 7, y = y0 + (j + 0.5) * CELL / 7;
            float h = T(x, y);
            if (h > best) { best = h; bx = x; by = y; }
        }
    double step = CELL / 14;
    for (int it = 0; it < 60 && step > 0.2; it++) {
        double gx, gy;
        grad(bx, by, &gx, &gy);
        double gl = sqrt(gx * gx + gy * gy);
        if (gl < 1e-5) break;
        double nx = bx + gx / gl * step, ny = by + gy / gl * step;
        float h = T(nx, ny);
        if (h > best) { best = h; bx = nx; by = ny; }
        else step *= 0.5;
    }
    if (bx > x0 + 4 && bx < x0 + CELL - 4 && by > y0 + 4 && by < y0 + CELL - 4 && best > WATER_MAX + 60) {
        int peak = 1;
        float ring = 1e9f;
        for (int k = 0; k < 12 && peak; k++) {
            double a = k * M_PI / 6;
            if (T(bx + 3 * cos(a), by + 3 * sin(a)) > best) peak = 0;
            float r = T(bx + 34 * S * cos(a), by + 34 * S * sin(a));
            if (r < ring) ring = r;
        }
        if (peak && best - ring > 25) { c->spot = 1; c->sx = bx; c->sy = by; c->sv = (int)lroundf(best); }
    }

    /* contour label */
    for (int tries = 0; tries < 6 && !c->lab; tries++) {
        uint32_t hsh = hash2(cx, cy, 300 + tries + opt_seed * 7);
        double x = x0 + CELL * (0.15 + 0.7 * (hsh & 0xffff) / 65536.0);
        double y = y0 + CELL * (0.15 + 0.7 * (hsh >> 16) / 65536.0);
        float h = T(x, y), L = roundf(h / I5) * I5;
        if (L < WATER_MAX + opt_interval) continue;
        if (!to_level(&x, &y, L)) continue;
        if (x < x0 || x >= x0 + CELL || y < y0 || y >= y0 + CELL) continue;
        double gx, gy;
        grad(x, y, &gx, &gy);
        double gl = sqrt(gx * gx + gy * gy);
        double spacing = opt_interval / gl;               /* px between contours */
        if (spacing < 3.2 * S || spacing > 26 * S) continue;
        /* runs straight: points half a label either side, walked back onto
         * the line, stay close to the tangent */
        double tx = -gy / gl, ty = gx / gl, half = 18 * S;
        int straight = 1;
        for (int sgn = -1; sgn <= 1 && straight; sgn += 2) {
            double px = x + sgn * tx * half, py = y + sgn * ty * half;
            double qx = px, qy = py;
            if (!to_level(&qx, &qy, L) || hypot(qx - px, qy - py) > 2.2 * S) straight = 0;
            double hx, hy;
            grad(qx, qy, &hx, &hy);
            double hl = sqrt(hx * hx + hy * hy);
            if (hl < 1e-6 || (hx * gx + hy * gy) / (hl * gl) < 0.93) straight = 0;
            if (opt_interval / hl < 3.0 * S) straight = 0;
        }
        if (!straight) continue;
        float ground[4];
        terrain((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, ground);
        if (fabsf(ground[1]) < 0.03f && ground[3] > 0.05f) continue;   /* in a river */
        if (c->spot && hypot(x - c->sx, y - c->sy) < 60 * S) continue;
        float a = (float)atan2(ty, tx);
        if (a > (float)M_PI / 2) a -= (float)M_PI;
        if (a <= -(float)M_PI / 2) a += (float)M_PI;
        c->lab = 1; c->lx = x; c->ly = y; c->la = a; c->lv = (int)L;
    }
}

/* the cell, built if need be; NULL if it isn't built and !build */
static Cell *cell_get(int cx, int cy, int build) {
    Cell *c = &cells[((cy % CK + CK) % CK) * CK + (cx % CK + CK) % CK];
    if (!c->ok || c->cx != cx || c->cy != cy) {
        if (!build) return NULL;
        cell_build(c, cx, cy);
    }
    return c;
}

/* ---- colours ---- */
typedef struct { float r, g, b; } Rgb;
static Rgb rgb(const char *h) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    return (Rgb){((v >> 16) & 255) / 255.0f, ((v >> 8) & 255) / 255.0f, (v & 255) / 255.0f};
}
static Rgb mixc(Rgb a, Rgb b, float t) { return (Rgb){a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t}; }
static void rgba(float out[4], Rgb c, float a) { out[0] = c.r; out[1] = c.g; out[2] = c.b; out[3] = a; }

static struct {
    Rgb paper, paper2, margin, hi, lo, thin, index, water, waterink, wood, grid, frame, river;
    Rgb label, spot, tri, title, sub, panel;
    float a_thin, a_index, a_grid, a_wood, hi_k, lo_k;
} P;

static void palette(void) {
    /* Aged Brass: bg pushed darker for a full screen, bg_alt, cream, fg_dim,
     * gold, dim gold, sage, amber; and for water a muted grey-green, not blue */
    Rgb bgd = rgb("#17120E"), bga = rgb("#3A2E25"), cream = rgb("#EFE3CE"), dim = rgb("#A89880");
    Rgb gold = rgb("#FFC332"), goldd = rgb("#D6A12A"), sage = rgb("#8FA672"), amber = rgb("#E9994A");
    Rgb slate = rgb("#7E9C9A");
    if (!opt_light) {
        /* gold and amber lines on near-black brown */
        P.paper = bgd; P.paper2 = mixc(bgd, bga, 0.75f); P.margin = rgb("#100C09");
        P.hi = cream; P.lo = rgb("#000000"); P.hi_k = 0.14f; P.lo_k = 0.5f;
        P.thin = amber; P.a_thin = 0.5f;
        P.index = goldd; P.a_index = 0.78f;
        P.water = mixc(bgd, slate, 0.10f); P.waterink = slate;
        P.wood = sage; P.a_wood = 0.42f;
        P.grid = cream; P.a_grid = 0.09f;
        P.frame = dim; P.river = mixc(slate, cream, 0.15f);
        P.label = gold; P.spot = cream; P.tri = gold; P.title = gold; P.sub = dim; P.panel = rgb("#120E0B");
    } else {
        /* --light, a paper map: ink (the theme's bg) on paper, contours in
         * amber darkened towards burnt sienna */
        Rgb paper = rgb("#F4EFE3"), ink = rgb("#2A211B");
        P.paper = paper; P.paper2 = mixc(paper, rgb("#E6D9BF"), 0.9f); P.margin = rgb("#EAE3D3");
        P.hi = rgb("#FFFFFF"); P.lo = mixc(ink, amber, 0.2f); P.hi_k = 0.35f; P.lo_k = 0.22f;
        P.thin = mixc(amber, ink, 0.35f); P.a_thin = 0.55f;
        P.index = mixc(amber, ink, 0.55f); P.a_index = 0.9f;
        P.water = mixc(paper, rgb("#6F8F98"), 0.16f); P.waterink = rgb("#4F7480");
        P.wood = mixc(sage, ink, 0.25f); P.a_wood = 0.55f;
        P.grid = ink; P.a_grid = 0.09f;
        P.frame = ink; P.river = rgb("#4F7480");
        P.label = mixc(amber, ink, 0.6f); P.spot = ink; P.tri = ink; P.title = ink; P.sub = mixc(ink, paper, 0.35f);
        P.panel = paper;
    }
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the terrain, ported from terrain() above */
#define NOISE_GLSL GLSL(                                                                   \
    uint hash2(ivec2 i, uint salt) {                                                      \
        uint h = uint(i.x) * 0x8da6b343u ^ uint(i.y) * 0xd8163841u ^ salt * 0xcb1ab31fu;   \
        h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;                                    \
        return h;                                                                          \
    }                                                                                      \
    float gdot(ivec2 i, uint s, vec2 d) {                                                 \
        uint h = hash2(i, s);                                                              \
        return (float(h & 0xffffu) / 32768.0 - 1.0) * d.x + (float(h >> 16u) / 32768.0 - 1.0) * d.y; \
    }                                                                                      \
    float vnoise(vec2 p, uint s) {                                                         \
        vec2 f = floor(p);                                                                 \
        ivec2 i = ivec2(f);                                                                \
        vec2 r = p - f;                                                                    \
        vec2 t = r * r * r * (r * (r * 6.0 - 15.0) + 10.0);                                \
        float a = gdot(i, s, r), b = gdot(i + ivec2(1, 0), s, r - vec2(1.0, 0.0));         \
        float c = gdot(i + ivec2(0, 1), s, r - vec2(0.0, 1.0)), d = gdot(i + ivec2(1, 1), s, r - 1.0); \
        return 0.5 + (a + (b - a) * t.x) + ((c + (d - c) * t.x) - (a + (b - a) * t.x)) * t.y; \
    }                                                                                      \
    vec2 rot(vec2 p, float k) { return vec2(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y) * k; } \
    float fbm(vec2 p, int oct, uint s) {                                                   \
        float sum = 0.0, amp = 0.5, norm = 0.0;                                            \
        for (int i = 0; i < oct; i++) {                                                    \
            sum += vnoise(p, s + uint(i)) * amp;                                           \
            norm += amp; amp *= 0.47; p = rot(p, 2.03);                                     \
        }                                                                                  \
        return sum / norm;                                                                 \
    }                                                                                      \
    float ridged(vec2 p, int oct, uint s) {                                                \
        float sum = 0.0, amp = 0.5, norm = 0.0;                                            \
        for (int i = 0; i < oct; i++) {                                                    \
            float n = max(0.0, 1.0 - abs(vnoise(p, s + uint(i)) * 2.0 - 1.0));            \
            sum += n * n * amp;                                                            \
            norm += amp; amp *= 0.42; p = rot(p, 2.07);                                     \
        }                                                                                  \
        return sum / norm;                                                                 \
    }                                                                                      \
    vec4 terrain(vec2 p) {                                                                 \
        vec2 w = vec2(fbm(p / 6000.0, 3, 71u), fbm(p / 6000.0 + vec2(5.2, 1.3), 3, 72u)) - 0.5; \
        vec2 P = p + w * 2000.0;                                                           \
        float base = fbm(P / 2800.0, 3, 11u) * 260.0;                                      \
        float mask = smoothstep(0.42, 0.66, fbm(p / 9000.0, 3, 23u));                      \
        float mtn = mask > 0.0 ? pow(ridged(P / 2600.0, 3, 37u), 1.4) * 1000.0 * mask : 0.0; \
        float h = base + mtn - 110.0;                                                      \
        vec2 q = P + (vec2(fbm(P / 4200.0 + vec2(3.1, -7.7), 3, 91u), fbm(P / 4200.0 + vec2(-1.9, 4.4), 3, 92u)) - 0.5) * 2600.0; \
        float rv = 0.5 * sin(3.14159265 * dot(q, vec2(0.6, 0.8)) / 6000.0);                        \
        float ron = (1.0 - smoothstep(120.0, 380.0, h)) * smoothstep(0.38, 0.5, fbm(p / 7000.0 + vec2(-4.4, 2.9), 2, 93u));                                     \
        float carve = (1.0 - smoothstep(0.0, 1.0, abs(rv) / 0.05)) * ron;                  \
        h = mix(h, min(h, h * 0.6 - 6.0), carve);                                          \
        float forest = fbm(p / 1400.0 + vec2(11.5, -3.3), 3, 55u);                         \
        return vec4(h, rv, forest, ron);                                                   \
    })

/* bake: texel t of the toroidal height texture holds world pixel
 * uB0 + ((t - uB0) mod N) */
static const char *FS_BAKE = "#version 330 core\n" NOISE_GLSL GLSL(
    uniform ivec2 uB0, uB0m, uN;
    uniform vec2 uSeed;
    uniform float uMpp;
    layout(location = 0) out float oH;
    layout(location = 1) out vec4 oAux;
    void main() {
        ivec2 t = ivec2(gl_FragCoord.xy);
        ivec2 i = uB0 + (t - uB0m + uN) % uN;
        vec4 g = terrain(vec2(i) * uMpp + uSeed);
        oH = g.x;
        oAux = vec4(g.yzw, 1.0);
    });

static const char *FS_MAP = "#version 330 core\n" GLSL(
    uniform sampler2D uH, uAux, uMask;
    uniform vec2 uRes, uN;
    uniform vec2 uOrgT;     /* world px of screen (0, 0), mod the texture size */
    uniform vec2 uOrgS;     /* ditto, mod the stipple period */
    uniform vec2 uOrgG;     /* world metres of screen (0, 0), mod 2 km */
    uniform vec4 uMap, uFrame;
    uniform float uMpp, uI, uWater, uSun, uS;
    uniform vec3 cPaper, cPaper2, cMargin, cHi, cLo, cThin, cIndex, cWater, cWaterInk, cWood, cGrid, cFrame, cRiver;
    uniform float aThin, aIndex, aGrid, aWood, kHi, kLo;
    out vec4 o;

    float Hh(vec2 w) { return texture(uH, (w + 0.5) / uN).r; }
    vec4 Ax(vec2 w) { return texture(uAux, (w + 0.5) / uN); }
    /* coverage of a line w px wide at distance d px */
    float cov(float d, float w) { return clamp(0.5 * w + 0.5 - d, 0.0, min(w, 1.0)); }
    uint ihash(uvec2 c, uint s) {
        uint h = c.x * 0x8da6b343u ^ c.y * 0xd8163841u ^ s * 0xcb1ab31fu;
        h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;
        return h;
    }
    float r01(uvec2 c, uint s) { return float(ihash(c, s) & 0xffffu) / 65536.0; }

    vec3 margin(vec2 sp, vec2 gm) {
        vec3 col = cMargin;
        bool inF = all(greaterThanEqual(sp, uFrame.xy)) && all(lessThan(sp, uFrame.zw));
        if (inF) {
            /* the zebra border: kilometres alternately filled */
            bool tb = sp.y < uMap.y || sp.y >= uMap.w, lr = sp.x < uMap.x || sp.x >= uMap.z;
            float k = tb && lr ? 0.0 : floor((tb ? gm.x : gm.y) / 1000.0);
            if (mod(k, 2.0) < 0.5) col = mix(col, cFrame, 0.55);
            /* frame and neat lines */
            vec2 a = sp - uFrame.xy, b = uFrame.zw - sp;
            if (min(min(a.x, a.y), min(b.x, b.y)) < 1.0) col = mix(col, cFrame, 0.8);
        }
        vec2 d = max(uMap.xy - sp, sp - uMap.zw);
        if (max(d.x, d.y) < 1.0) col = mix(col, cFrame, 0.9);
        return col;
    }

    void main() {
        vec2 sp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 gm = sp * uMpp + uOrgG;
        if (any(lessThan(sp, uMap.xy)) || any(greaterThanEqual(sp, uMap.zw))) {
            o = vec4(margin(sp, gm), 1.0);
            return;
        }
        vec2 w = sp + uOrgT;
        vec2 ex = vec2(1.0, 0.0), ey = vec2(0.0, 1.0);
        float h = Hh(w);
        vec2 g = 0.5 * vec2(Hh(w + ex) - Hh(w - ex), Hh(w + ey) - Hh(w - ey));   /* m per px */
        vec4 ax = Ax(w);
        float gl = max(length(g), 1e-4);
        float knock = texelFetch(uMask, ivec2(gl_FragCoord.xy), 0).r;

        /* paper, warming with height, and the hillshade */
        vec3 col = mix(cPaper, cPaper2, 0.25 + 0.75 * smoothstep(-50.0, 1100.0, h));
        vec3 n = normalize(vec3(-g * (2.2 / uMpp), 1.0));
        vec3 L = normalize(vec3(cos(uSun), sin(uSun), 1.1));
        float sh = dot(n, L) - L.z;                        /* 0 on the flat */
        col = sh > 0.0 ? mix(col, cHi, min(sh, 1.0) * kHi) : mix(col, cLo, min(-sh, 1.0) * kLo);

        /* water: below datum, a soft shore */
        float dsh = (h - uWater) / gl;                      /* px from the shore, + on land */
        float wet = clamp(0.5 - dsh, 0.0, 1.0);
        col = mix(col, cWater, wet);

        /* woodland stipple */
        float slope = gl / uMpp;
        float dens = smoothstep(0.56, 0.7, ax.z) * (1.0 - smoothstep(550.0, 800.0, h)) *
                     smoothstep(4.0, 12.0, dsh) * (1.0 - smoothstep(0.5, 1.0, slope));
        if (dens > 0.0) {
            float cs = 5.0 * uS;
            vec2 ws = sp + uOrgS;
            vec2 c = floor(ws / cs);
            uvec2 uc = uvec2(ivec2(c) & 4095);
            vec2 ctr = (c + 0.25 + 0.5 * vec2(r01(uc, 1u), r01(uc, 2u))) * cs;
            float dot_ = clamp(0.8 * uS + 0.5 - length(ws - ctr), 0.0, 1.0);
            if (r01(uc, 3u) < dens) col = mix(col, cWood, dot_ * aWood * (1.0 - knock));
        }

        /* kilometre grid */
        vec2 dg = abs(fract(gm / 1000.0 + 0.5) - 0.5) * 1000.0 / uMpp;
        col = mix(col, cGrid, max(cov(dg.x, 1.0), cov(dg.y, 1.0)) * aGrid * (1.0 - knock));

        /* waterlines: depths D (k / 1)^(4/3) out from the shore, fading */
        if (wet > 0.0) {
            float depth = max(uWater - h, 1e-3), D1 = 9.0;
            float q = pow(depth / D1, 0.75);
            float gq = 0.75 * pow(depth / D1, -0.25) / D1 * gl;
            float k = floor(q + 0.5);
            float dq = abs(q - k) / max(gq, 1e-5);
            float wl = k >= 1.0 && k <= 5.0 ? cov(dq, 0.8 * uS) * exp(-0.42 * (k - 1.0)) * smoothstep(0.6, 2.5, 1.0 / gq) : 0.0;
            col = mix(col, cWaterInk, wl * 0.5 * wet);
        }

        /* contours */
        float lines = 1.0 - wet;
        float spc = uI / gl;                                 /* px between contours */
        float f = h / uI;
        float dt = abs(fract(f + 0.5) - 0.5) * spc;
        float thin = cov(dt, 0.85 * uS) * smoothstep(1.6, 3.6, spc);
        float f5 = h / (5.0 * uI);
        float di = abs(fract(f5 + 0.5) - 0.5) * spc * 5.0;
        float idx = cov(di, 1.4 * uS) * smoothstep(1.4, 3.0, spc * 5.0);
        col = mix(col, cThin, thin * aThin * lines * (1.0 - knock));
        col = mix(col, cIndex, idx * aIndex * lines * (1.0 - knock));

        /* rivers: along their valleys, widening as they come down */
        if (abs(ax.x) < 0.06 && ax.w > 0.45) {
            vec2 gr = 0.5 * vec2(Ax(w + ex).x - Ax(w - ex).x, Ax(w + ey).x - Ax(w - ey).x);
            float dr = abs(ax.x) / max(length(gr), 1e-7);
            float rw = uS * mix(0.6, 2.0, smoothstep(200.0, 10.0, h));
            float ra = cov(dr, rw) * smoothstep(0.45, 0.75, ax.w) * (1.0 - wet) * (1.0 - knock);
            col = mix(col, cRiver, ra * 0.8);
        }

        /* the shoreline itself */
        col = mix(col, cWaterInk, cov(abs(dsh), 1.1 * uS) * 0.8);
        o = vec4(col, 1.0);
    });

static const char *VS_QUAD = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in vec4 aCol;
    layout(location = 3) in float aMode;
    uniform vec2 uRes;
    out vec2 vUV;
    out vec4 vCol;
    flat out int vMode;
    void main() {
        vec2 p = aPos / uRes * 2.0 - 1.0;
        gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
        vUV = aUV; vCol = aCol; vMode = int(aMode + 0.5);
    });
static const char *FS_QUAD = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vCol;
    flat in int vMode;
    out vec4 o;
    void main() {
        if (vMode == 0) o = vec4(vCol.rgb, vCol.a * texture(uAtlas, vUV).r);
        else if (vMode == 1) o = vCol;
        else {                                  /* knock-out: a rounded box, uv local px */
            vec2 q = abs(vUV) - vCol.xy + vCol.z;
            float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - vCol.z;
            o = vec4(clamp(0.5 - d / 1.5, 0.0, 1.0));
        }
    });

static GLuint prog_bake, prog_map, prog_quad, vao, vao_q, vbo_q, tex_atlas;
static GLuint tex_h, tex_aux, fbo_h, tex_mask, fbo_mask;
static int TN, TM, tex_w, tex_h_, baked, bake_x, bake_y;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "contours: shader:\n%s\n", log);
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
static GLuint mk_tex(GLint ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLint filter, GLint wrap) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    return t;
}
#define U(p, n) glGetUniformLocation(p, n)
static void u3(GLuint p, const char *n, Rgb c) { glUniform3f(U(p, n), c.r, c.g, c.b); }

/* (re)make the size-dependent targets: the height texture, the mask */
static void targets(void) {
    if (tex_w == W && tex_h_ == H) return;
    if (tex_w) {
        GLuint t[3] = {tex_h, tex_aux, tex_mask}, f[2] = {fbo_h, fbo_mask};
        glDeleteTextures(3, t);
        glDeleteFramebuffers(2, f);
    }
    tex_w = W; tex_h_ = H;
    TN = W + 16; TM = H + 16;
    tex_h = mk_tex(GL_R32F, TN, TM, GL_RED, GL_FLOAT, NULL, GL_LINEAR, GL_REPEAT);
    tex_aux = mk_tex(GL_RGBA16F, TN, TM, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_REPEAT);
    glGenFramebuffers(1, &fbo_h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_h);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_h, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_aux, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("height framebuffer incomplete");
    tex_mask = mk_tex(GL_R8, W, H, GL_RED, GL_UNSIGNED_BYTE, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo_mask);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_mask);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_mask, 0);
    baked = 0;
}

static void gl_init(void) {
    prog_bake = mk_prog(VS, FS_BAKE);
    prog_map = mk_prog(VS, FS_MAP);
    prog_quad = mk_prog(VS_QUAD, FS_QUAD);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_q);
    glGenBuffers(1, &vbo_q);
    glBindVertexArray(vao_q);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_q);
    for (int i = 0, off = 0, n[4] = {2, 2, 4, 1}; i < 4; off += n[i++]) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, n[i], GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(sizeof(float) * off));
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    tex_atlas = mk_tex(GL_R8, ATW, ATH, GL_RED, GL_UNSIGNED_BYTE, atlas, GL_LINEAR, GL_CLAMP_TO_EDGE);

    GLuint p = prog_map;
    glUseProgram(p);
    glUniform1i(U(p, "uH"), 0);
    glUniform1i(U(p, "uAux"), 1);
    glUniform1i(U(p, "uMask"), 2);
    u3(p, "cPaper", P.paper); u3(p, "cPaper2", P.paper2); u3(p, "cMargin", P.margin);
    u3(p, "cHi", P.hi); u3(p, "cLo", P.lo); u3(p, "cThin", P.thin); u3(p, "cIndex", P.index);
    u3(p, "cWater", P.water); u3(p, "cWaterInk", P.waterink); u3(p, "cWood", P.wood);
    u3(p, "cGrid", P.grid); u3(p, "cFrame", P.frame); u3(p, "cRiver", P.river);
    glUniform1f(U(p, "aThin"), P.a_thin); glUniform1f(U(p, "aIndex"), P.a_index);
    glUniform1f(U(p, "aGrid"), P.a_grid); glUniform1f(U(p, "aWood"), P.a_wood);
    glUniform1f(U(p, "kHi"), P.hi_k); glUniform1f(U(p, "kLo"), P.lo_k);
    glUniform1f(U(p, "uI"), opt_interval);
    glUniform1f(U(p, "uS"), S);
    glUniform1f(U(p, "uMpp"), MPP);
    glUseProgram(prog_bake);
    glUniform2f(U(prog_bake, "uSeed"), seed_x, seed_z);
    glUniform1f(U(prog_bake, "uMpp"), MPP);
    glUseProgram(prog_quad);
    glUniform1i(U(prog_quad, "uAtlas"), 0);
}

static int posmod(int a, int n) { return (a % n + n) % n; }

/* bake world texels [x0, x1) x all rows, or all columns x [y0, y1) */
static void bake_span(int horiz, int a, int b) {
    int n = horiz ? TN : TM, len = b - a, s = posmod(a, n);
    while (len > 0) {
        int run = len < n - s ? len : n - s;
        if (horiz) glScissor(s, 0, run, TM);
        else glScissor(0, s, TN, run);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        len -= run; s = 0;
    }
}

/* make the height texture cover world px [nx, nx + TN) x [ny, ny + TM) */
static void bake(int nx, int ny) {
    if (baked && nx == bake_x && ny == bake_y) return;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_h);
    glViewport(0, 0, TN, TM);
    glUseProgram(prog_bake);
    glUniform2i(U(prog_bake, "uB0"), nx, ny);
    glUniform2i(U(prog_bake, "uB0m"), posmod(nx, TN), posmod(ny, TM));
    glUniform2i(U(prog_bake, "uN"), TN, TM);
    glBindVertexArray(vao);
    glEnable(GL_SCISSOR_TEST);
    if (!baked || abs(nx - bake_x) >= TN || abs(ny - bake_y) >= TM) {
        glScissor(0, 0, TN, TM);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    } else {
        if (nx > bake_x) bake_span(1, bake_x + TN, nx + TN);
        if (nx < bake_x) bake_span(1, nx, bake_x);
        if (ny > bake_y) bake_span(0, bake_y + TM, ny + TM);
        if (ny < bake_y) bake_span(0, ny, bake_y);
    }
    glDisable(GL_SCISSOR_TEST);
    baked = 1; bake_x = nx; bake_y = ny;
}

/* ---- the overlay: labels on the map, then the margin and the cartouche ---- */
static char host[64];
static int mx0, my0, mx1, my1;         /* the map's rectangle, px */

typedef struct { float x, y, r; } Disc;

static void build_features(double ox, double oy) {
    float lc[4], sc[4], tc[4];
    rgba(lc, P.label, 0.95f); rgba(sc, P.spot, 0.85f); rgba(tc, P.tri, 0.95f);
    /* everything near the view, then placed greedily: summits first, highest
     * first, then contour labels in a fixed order, skipping any that would
     * crowd one already placed */
    enum { MAXF = 512 };
    struct { int kind, pri; double x, y; Cell *c; } f[MAXF];
    int nf = 0;
    int cx0 = (int)floor((ox + mx0 - 60 * S) / CELL), cx1 = (int)floor((ox + mx1 + 60 * S) / CELL);
    int cy0 = (int)floor((oy + my0 - 60 * S) / CELL), cy1 = (int)floor((oy + my1 + 60 * S) / CELL);
    /* build the ring of cells just outside ahead of time, a couple a frame,
     * so the drift never has to build a whole row at once */
    int budget = 2;
    for (int cy = cy0 - 1; cy <= cy1 + 1 && budget > 0; cy++)
        for (int cx = cx0 - 1; cx <= cx1 + 1 && budget > 0; cx++)
            if ((cy < cy0 || cy > cy1 || cx < cx0 || cx > cx1) && !cell_get(cx, cy, 0)) {
                cell_get(cx, cy, 1);
                budget--;
            }
    for (int cy = cy0; cy <= cy1; cy++)
        for (int cx = cx0; cx <= cx1; cx++) {
            Cell *c = cell_get(cx, cy, 1);
            if (c->spot && nf < MAXF) { f[nf].kind = 0; f[nf].pri = 1000000 - c->sv; f[nf].x = c->sx; f[nf].y = c->sy; f[nf++].c = c; }
            if (c->lab && nf < MAXF) { f[nf].kind = 1; f[nf].pri = 2000000 + (int)(hash2(cx, cy, 5) & 0xfffff); f[nf].x = c->lx; f[nf].y = c->ly; f[nf++].c = c; }
        }
    for (int i = 1; i < nf; i++)                        /* insertion sort by priority */
        for (int j = i; j > 0 && f[j].pri < f[j - 1].pri; j--) {
            __typeof__(f[0]) t = f[j]; f[j] = f[j - 1]; f[j - 1] = t;
        }
    Disc placed[MAXF];
    int np = 0;
    for (int i = 0; i < nf; i++) {
        float x = (float)(f[i].x - ox), y = (float)(f[i].y - oy);
        Cell *c = f[i].c;
        char buf[16];
        snprintf(buf, sizeof buf, "%d", f[i].kind == 0 ? c->sv : c->lv);
        const Face *fn = &fonts[f[i].kind == 0 ? F_SPOT : F_NUM];
        float tw = text_w(fn, buf, 0);
        Disc d;
        if (f[i].kind == 0) d = (Disc){x + tw * 0.5f, y, tw * 0.5f + 10 * S};
        else d = (Disc){x, y, tw * 0.5f + 8 * S};
        int ok = 1;
        for (int k = 0; k < np && ok; k++)
            if (hypotf(d.x - placed[k].x, d.y - placed[k].y) < d.r + placed[k].r + 6 * S) ok = 0;
        if (!ok) continue;
        placed[np++] = d;
        if (f[i].kind == 0) {
            const Glyph *g = &sprites[SP_TRI];
            blit(&ov, g, x, y, 1, 0, -g->w * 0.5f, -g->h * 0.62f, tc);
            float tx = x + g->w * 0.5f + 3.5f * S;
            text(&ov, fn, buf, tx, y, 0, 0, digit_mid(fn), 0, sc);
            knock(x, y, 0, g->w * 0.5f + 1.5f * S, g->h * 0.5f + 1.5f * S);
            knock(tx + tw * 0.5f, y, 0, tw * 0.5f + 2 * S, digit_h(fn) * 0.5f + 2 * S);
        } else {
            text(&ov, fn, buf, x, y, c->la, 0.5f, digit_mid(fn), 0, lc);
            knock(x, y, c->la, tw * 0.5f + 3 * S, digit_h(fn) * 0.5f + 2.5f * S);
        }
    }
}

static void build_margin(double ox, double oy) {
    float cc[4];
    rgba(cc, P.frame, 0.95f);
    const Face *fn = &fonts[F_COORD];
    float bw = roundf(5 * S), mid_t = (my0 - bw) * 0.5f, mid_l = (mx0 - bw) * 0.5f;
    char buf[16];
    /* eastings along the top and bottom, northings up the sides */
    for (long k = (long)ceil((ox + mx0) * MPP / 1000); k * 1000.0 / MPP - ox < mx1; k++) {
        float x = (float)(k * 1000.0 / MPP - ox);
        if (x < mx0 + 14 * S || x > mx1 - 14 * S) continue;
        snprintf(buf, sizeof buf, "%02ld", ((E0 + k) % 100 + 100) % 100);
        text(&ov, fn, buf, x, mid_t, 0, 0.5f, digit_mid(fn), 0, cc);
        text(&ov, fn, buf, x, H - mid_t, 0, 0.5f, digit_mid(fn), 0, cc);
    }
    for (long k = (long)ceil((oy + my0) * MPP / 1000); k * 1000.0 / MPP - oy < my1; k++) {
        float y = (float)(k * 1000.0 / MPP - oy);
        if (y < my0 + 14 * S || y > my1 - 14 * S) continue;
        snprintf(buf, sizeof buf, "%02ld", ((N0 - k) % 100 + 100) % 100);
        text(&ov, fn, buf, mid_l, y, 0, 0.5f, digit_mid(fn), 0, cc);
        text(&ov, fn, buf, W - mid_l, y, 0, 0.5f, digit_mid(fn), 0, cc);
    }
}

/* the title block: north arrow | HOST · HH:MM, interval, scale bar */
static void build_cartouche(void) {
    float tc[4], sc[4], fc[4], fc2[4], pc[4], bc[4];
    rgba(tc, P.title, 1); rgba(sc, P.sub, 1); rgba(fc, P.frame, 0.9f); rgba(fc2, P.frame, 0.45f);
    rgba(pc, P.panel, 1); rgba(bc, P.label, 0.9f);
    char title[96], sub[64];
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char hm[8];
    strftime(hm, sizeof hm, "%H:%M", &tm);
    snprintf(title, sizeof title, "%s \x7f %s", host, hm);
    snprintf(sub, sizeof sub, "Contours at %g metre intervals", opt_interval);
    const Face *ft = &fonts[F_TITLE], *fs = &fonts[F_SUB], *fm = &fonts[F_SMALL];
    float track = roundf(3 * S);
    float tw = text_w(ft, title, track), sw = text_w(fs, sub, 0);
    /* the scale bar: whole kilometres, about 230 px long */
    int km = (int)fmaxf(1, roundf(230 * S * MPP / 1000));
    float bar = km * 1000 / MPP;
    float pad = roundf(16 * S), col = roundf(36 * S);
    float inner = fmaxf(fmaxf(tw, sw), bar + 30 * S);
    float pw = roundf(pad + col + pad + inner + pad), ph_ = roundf(124 * S);
    float x1 = mx1 - roundf(20 * S), y1 = my1 - roundf(20 * S), x0 = x1 - pw, y0 = y1 - ph_;

    rect(&ov, x0, y0, x1, y1, pc);
    float l = fmaxf(1, roundf(S));
    rect(&ov, x0, y0, x1, y0 + l, fc); rect(&ov, x0, y1 - l, x1, y1, fc);
    rect(&ov, x0, y0, x0 + l, y1, fc); rect(&ov, x1 - l, y0, x1, y1, fc);
    float in = roundf(4 * S);
    rect(&ov, x0 + in, y0 + in, x1 - in, y0 + in + 1, fc2); rect(&ov, x0 + in, y1 - in - 1, x1 - in, y1 - in, fc2);
    rect(&ov, x0 + in, y0 + in, x0 + in + 1, y1 - in, fc2); rect(&ov, x1 - in - 1, y0 + in, x1 - in, y1 - in, fc2);

    /* north arrow, with an N over it */
    float ax = roundf(x0 + pad + col * 0.5f);
    const Glyph *ga = &sprites[SP_ARROW];
    float ay = roundf(y0 + ph_ * 0.5f - ga->h * 0.5f + 9 * S);
    text(&ov, ft, "N", ax, ay - 6 * S, 0, 0.5f, 0, 0, tc);
    blit(&ov, ga, ax, ay, 1, 0, -floorf(ga->w * 0.5f), 0, tc);
    float dx = roundf(x0 + pad + col + pad * 0.5f);
    rect(&ov, dx, y0 + pad, dx + 1, y1 - pad, fc2);

    float tx = x0 + pad + col + pad;
    float by = y0 + pad + cap_h(ft);
    text(&ov, ft, title, tx, by, 0, 0, 0, track, tc);
    float ry = roundf(by + 10 * S);
    rect(&ov, tx, ry, tx + inner, ry + 1, fc2);
    float sy = ry + roundf(10 * S) + cap_h(fs);
    text(&ov, fs, sub, tx, sy, 0, 0, 0, 0, sc);

    /* scale bar: alternate halves of each kilometre filled */
    float bt = roundf(sy + 13 * S), bh = roundf(5 * S);
    float seg = bar / (km * 2);
    rect(&ov, tx, bt, tx + bar, bt + 1, fc); rect(&ov, tx, bt + bh - 1, tx + bar, bt + bh, fc);
    for (int i = 0; i < km * 2; i++) {
        float a = roundf(tx + i * seg), b = roundf(tx + (i + 1) * seg);
        if (!(i & 1)) rect(&ov, a, bt, b, bt + bh, bc);
        rect(&ov, a, bt, a + 1, bt + bh, fc);
    }
    rect(&ov, roundf(tx + bar) - 1, bt, roundf(tx + bar), bt + bh, fc);
    char buf[16];
    float ly = bt + bh + roundf(5 * S) + cap_h(fm);
    for (int i = 0; i <= km; i++) {
        snprintf(buf, sizeof buf, "%d", i);
        text(&ov, fm, buf, roundf(tx + i * bar / km), ly, 0, 0.5f, 0, 0, sc);
    }
    text(&ov, fm, "km", roundf(tx + bar + 8 * S), ly, 0, 0, 0, 0, sc);
}

/* ---- a frame ---- */
static void draw_quads(int from, int to) {
    if (to <= from) return;
    glUseProgram(prog_quad);
    glUniform2f(U(prog_quad, "uRes"), (float)W, (float)H);
    glBindVertexArray(vao_q);
    glDrawArrays(GL_TRIANGLES, from / 9, (to - from) / 9);
}

/* t: drift time, seconds */
static void render(double t, GLuint target) {
    targets();
    float m = roundf(36 * S);
    mx0 = (int)m; my0 = (int)m; mx1 = W - (int)m; my1 = H - (int)m;

    double cx, cy;
    view_at(t, &cx, &cy);
    double ox = cx - W * 0.5, oy = cy - H * 0.5;        /* world px of the screen's top-left */
    bake((int)floor(ox) - 8, (int)floor(oy) - 8);

    /* the overlay and the knock-outs */
    ov.n = mk.n = 0;
    build_features(ox, oy);
    ov_split = ov.n;
    build_margin(ox, oy);
    build_cartouche();

    glBindBuffer(GL_ARRAY_BUFFER, vbo_q);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (size_t)(ov.n + mk.n), NULL, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * (size_t)ov.n, ov.v);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * (size_t)ov.n, sizeof(float) * (size_t)mk.n, mk.v);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_mask);
    glViewport(0, 0, W, H);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendEquation(GL_MAX);
    glBlendFunc(GL_ONE, GL_ONE);
    draw_quads(ov.n, ov.n + mk.n);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_map);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_h);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_mask);
    glActiveTexture(GL_TEXTURE0);
    GLuint p = prog_map;
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform2f(U(p, "uN"), (float)TN, (float)TM);
    double sp = 5.0 * S * 4096;
    glUniform2f(U(p, "uOrgT"), (float)(ox - floor(ox / TN) * TN), (float)(oy - floor(oy / TM) * TM));
    glUniform2f(U(p, "uOrgS"), (float)(ox - floor(ox / sp) * sp), (float)(oy - floor(oy / sp) * sp));
    double gx = ox * MPP, gy = oy * MPP;
    glUniform2f(U(p, "uOrgG"), (float)(gx - floor(gx / 2000) * 2000), (float)(gy - floor(gy / 2000) * 2000));
    glUniform4f(U(p, "uMap"), (float)mx0, (float)my0, (float)mx1, (float)my1);
    float b = roundf(5 * S);
    glUniform4f(U(p, "uFrame"), mx0 - b, my0 - b, mx1 + b, my1 + b);
    glUniform1f(U(p, "uWater"), water_at(t));
    glUniform1f(U(p, "uSun"), (float)(-2.356 + 0.3 * sin(t / 230 + ph[5])));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glEnable(GL_SCISSOR_TEST);
    glScissor(mx0, H - my1, mx1 - mx0, my1 - my0);
    draw_quads(0, ov_split);
    glDisable(GL_SCISSOR_TEST);
    draw_quads(ov_split, ov.n);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "contours");
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
        "usage: contours [--window | --root | --layer] [--seed N] [--light] [--interval METRES]\n"
        "                [--speed X] [--fps N]\n"
        "                [--shot FILE.ppm [--size WxH] [--at SECONDS]] [--bench N [--size WxH]]\n");
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

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-light")) opt_light = 1;
        else if (!strcmp(a, "-interval") && i + 1 < argc) opt_interval = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (opt_interval < 1 || opt_fps < 1 || off_w < 64 || off_h < 64) usage();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench;

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    for (char *p = host; *p; p++) *p = (char)toupper((unsigned char)*p);
    if (!host[0]) strcpy(host, "MAP");

    /* where in the world, and how the drift is phased */
    uint32_t h = hash2((int32_t)opt_seed, 17, 1);
    seed_x = ((h & 0xffff) / 65536.0f - 0.5f) * 100000;
    seed_z = ((h >> 16) / 65536.0f - 0.5f) * 100000;
    for (int i = 0; i < 6; i++) ph[i] = (hash2((int32_t)opt_seed, i, 2) & 0xffff) / 65536.0 * 2 * M_PI;
    E0 = (int)(hash2((int32_t)opt_seed, 3, 3) % 100); N0 = (int)(hash2((int32_t)opt_seed, 4, 3) % 100);

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
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput,
                            vi->visual, CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Contours");
        XClassHint chint = {"contours", "Contours"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    /* scale the type and the map for the screen: a window gets its real size
     * from the WM, so go by the display it's on */
    int gw = W, gh = H;
    if (dpy && !opt_root && !offscreen) {
        gw = DisplayWidth(dpy, DefaultScreen(dpy));
        gh = DisplayHeight(dpy, DefaultScreen(dpy));
    }
    S = fminf(gw / 1920.0f, gh / 1080.0f);
    if (S < 0.6f) S = 0.6f;
    MPP = 6.5f / S;
    CELL = 190 * S;
    palette();
    build_atlas();
    gl_init();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (opt_bench) {
            double sum = 0, worst = 0, first = 0;
            for (int i = 0; i < opt_bench; i++) {
                double t0 = now();
                render((opt_at + (double)i / opt_fps) * opt_speed, fbo);
                glFinish();
                double ms = (now() - t0) * 1000;
                if (i == 0) { first = ms; continue; }
                sum += ms;
                if (ms > worst) worst = ms;
            }
            int n = opt_bench > 1 ? opt_bench - 1 : 1;
            printf("%s\n%dx%d: %d frames, avg %.2f ms, worst %.2f ms (first frame, with the full bake: %.1f ms)\n",
                   (const char *)glGetString(GL_RENDERER), W, H, opt_bench, sum / n, worst, first);
        }
        if (opt_shot) {                                 /* with --bench: the last frame benched */
            if (!opt_bench) render(opt_at * opt_speed, fbo);
            glFinish();
            save_ppm(opt_shot);
        }
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
        double t = now();
        render((t - t0) * opt_speed, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
