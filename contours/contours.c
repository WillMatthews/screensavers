/*
 * contours — a living topographic map: the contour lines of an endless
 * procedural mountain country, drawn like a fine cartographic sheet, drifting
 * slowly past under a moving sun.
 *
 * The ground is carved by its own drainage (see "the ground"): a hierarchy
 * of valleys, rivers to gills, each falling to the one below it, the land
 * the lowest of them all, so water runs off everywhere to the rivers and
 * lakes; mountain ranges with arêtes, corries and tarns, U-shaped dales with
 * flat floors, rolling hills and low lake country, by province. The water is
 * everything below datum, and the tarns, each at its own level. The ground
 * is baked on the GPU into a float height
 * texture one screen pixel per texel, addressed toroidally: as the view
 * drifts only the strips of new ground that slide into view are baked, so a
 * frame costs one cheap shading pass however heavy the terrain is.
 *
 * Every pixel then draws, from the height and its central differences:
 *   - contours every --interval metres, index contours every fifth, as
 *     constant-width anti-aliased lines (distance to the nearest level =
 *     |fract(h / I) - 1/2| * I / |grad h|), thin ones narrowing then fading
 *     where they crowd; lit Tanaka-fashion, brighter and bolder on slopes
 *     facing the sun, darker and finer in shade, with a little ink spread;
 *     ticks on the downhill side of contours closing round a hollow;
 *   - a soft multi-directional hillshade over a fine and a broad normal,
 *     the sun swinging across the north, hollows shadowed and crests lifted, on a
 *     hypsometric tint from the palette's paper, woodland and highlight;
 *   - water below datum (which rises and falls a few metres, so the shores
 *     breathe), with engraved waterlines following the shore outwards;
 *   - woodland stipple,
 *     with a tree symbol here and there; wedge hachures on crags;
 *   - a coastal vignette and engraved hatching inside every shore;
 *   - a faint kilometre grid; and paper grain over it all.
 * And a map's furniture, invented but consistent (see "cartography"):
 * rivers and becks traced down their valleys from the heads of the
 * drainage network to the lakes, villages joined by roads
 * routed through the valleys and over the passes (tracks and footpaths where
 * they must climb), and place names: villages upright, water in italic,
 * hills in small italic, districts in spaced capitals.
 * On the CPU the same terrain finds summits (spot heights with triangles)
 * and straight stretches of index contour to carry their height, set along
 * the line, readable side up, knocking a gap in the lines beneath. Round it:
 * a margin with a zebra border and grid numbers, and a title cartouche (the
 * host and the time, the contour interval, north arrow and scale bar).
 *
 * --seed N picks the country, the drift and the palette; --palette NAME
 * fixes the palette: brass (gold and amber on near-black brown, the theme's
 * own), cream (sienna on paper; --light is an alias), slate (pale blue-grey
 * on deep slate), blueprint (cyan and white on Prussian blue) or survey (a
 * printed survey sheet: brown contours, blue water, a green-to-buff tint);
 * It lives: the drift lingers and glides (pace()); the sun swings from
 * west-south-west to north-east and back over ~16 minutes, low at the ends,
 * and casts soft shadows that sweep across the country (FS_SHADE marches
 * towards it through the height texture, at quarter resolution, every other
 * frame; the ground is baked SH_REACH beyond the screen for it); cloud
 * shadows drift over on the wind, changing shape; swash lines run in to the
 * shores and glints twinkle out on the water; and a surveyor's pen walks
 * the country, keeping to the contours where it can, leaving a dotted track
 * that fades over a few minutes (build_walker()).
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
 * last of them: the incrementally baked frame). CONTOURS_VERIFY=1 in the
 * environment compares a sample of the first full bake with the CPU's
 * terrain() (they should agree to a centimetre or so).
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
#include "../common/hidpi.h"

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_fps = 30;
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
static double ph2[8];                  /* phases of the pace, the light, the walker */

static void die(const char *what) { fprintf(stderr, "contours: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- noise (flightsim's, ported 1:1 to GLSL below) ---- */
static uint32_t hash2(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
/* floor, exact, without a call into libm (it is hot) */
static inline float ifloor(float x) { float t = (float)(int)x; return t > x ? t - 1 : t; }
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
    float fx = ifloor(x), fz = ifloor(z);
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
/* a min-heap, for the flood and the routes */
typedef struct { float f; int i; } Hn;
static void hpush(Hn *hp, int *n, Hn v) {
    int i = (*n)++;
    while (i && hp[(i - 1) / 2].f > v.f) { hp[i] = hp[(i - 1) / 2]; i = (i - 1) / 2; }
    hp[i] = v;
}
static Hn hpop(Hn *hp, int *n) {
    Hn top = hp[0], v = hp[--*n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && hp[c + 1].f < hp[c].f) c++;
        if (hp[c].f >= v.f) break;
        hp[i] = hp[c]; i = c;
    }
    hp[i] = v;
    return top;
}
/* ---- the ground: a drainage network, and the valleys it cuts ----
 * The country is built the way water would have built it. A hierarchy of
 * jittered grids, cells LS[k] across (7.2 km, 2.4 km, 800 m, 267 m), each
 * with a node joined to where its water goes:
 *   level 0, the rivers: each node to the neighbour lowest on a broad
 *     control surface (ctrl(): the lie of the land, low in the lake
 *     country, high in the mountains), or, lowest of its neighbours, to
 *     nothing: a sink, made a lake;
 *   levels 1 to 3, the becks, gills and gullies: each node partway up the
 *     valley side it stands on (lambda_at(): shallow in the hills and the
 *     finer levels, deep in the mountains), joined straight down to the
 *     floor of the valley below it (or, if that's far, to the next node of
 *     its own level on the way there), bending through a point off its
 *     middle;
 *   high in the mountains, where the ground rises well behind it, a valley
 *     head may be a corrie (corrie()): a scoop with a floor and a cliffed
 *     headwall, holding a tarn at the level it would spill at (tarn_level()).
 * Every segment is a valley whose floor falls along it, flat for a width
 * w (falling gently to a meandering line where the river runs), then
 * rising at the slope R, a U rounded over LC[k] that opens out down the
 * valley and steepens towards the crest, scree-steep under it; the ground is
 * the lowest of all of them (softened where they meet high up, so crests
 * round over in the hills and stay sharp in the mountains). Since every
 * floor falls to a lower one, and the lowest of valleys only bottoms out
 * at a sink, water runs off everywhere to the rivers and lakes: no hollows
 * but the lakes, branching valleys, spurs and arêtes between, flat valley
 * floors. A province field (province(), drawn out along a strike, so the
 * mountains run in ranges) sets the relief: steep, high and sharp in the
 * mountains, rolling hills, low lake country. The whole network is laid
 * out in a gently warped space (warp()), so no valley runs straight.
 * Nodes are a pure function of their cell and are cached (node_get());
 * the GPU bake reads them from a texture filled from the same cache, so
 * the two agree exactly about the network. Keep env() and terrain() in
 * step with FS_BAKE. */
#define NL 4
static const float LS[NL] = {7200, 2400, 800, 267};         /* cell sizes, m */
static const float LC[NL] = {800, 420, 110, 40};            /* valley-bottom rounding, m (mountains) */
/* a node, its valley bending through (mx, my) on its way to (tx, ty) */
/* and perhaps a corrie: a scoop up behind the node, centred (cx, cy), cr
 * in radius, its floor cq below the node */
typedef struct { float x, y, h, w, mx, my, tx, ty, th, riv, cq, cr, cx, cy, b0x, b0y, b1x, b1y; } Node;

/* the provinces: 0 the low lake country, through the hills, to 1 the
 * mountains */
static float province(float x, float y) {
    /* ranges, not blobs: the field is drawn out ~2.5 : 1 along a strike */
    return smoothstep(0.3f, 0.7f, fbm((0.8f * x - 0.6f * y) / 60000, (0.6f * x + 0.8f * y) / 24000, 3, 23));
}
/* the lie of the land, for the rivers' floors */
static float ctrl(float x, float y) {
    float u = province(x, y);
    return 30 + 400 * smoothstep(0.12f, 0.92f, u) + 220 * (fbm(x / 15000 + 7.1f, y / 15000 - 2.3f, 2, 31) - 0.5f);
}
/* the slope of valley sides, the softening of crests, valley floors' flat
 * half width, and how far up its valley side a tributary rises */
static float slope_at(float x, float y, float u) {
    return (0.08f + 0.56f * u * u) * (0.7f + 0.6f * fbm(x / 4000, y / 4000, 2, 29));
}
static float soft_at(float u) { return 150 - 125 * u; }
static float flat_at(int k, float u) { return k == 0 ? 330 - 250 * u : k == 1 ? 80 - 50 * u : k == 2 ? 10 : 0; }
/* a tributary's head is this far (0 .. 1) up the valley side from where it
 * joins: high (shallow valleys) in the hills and the finer levels, and near
 * the water, so the shores don't fray */
static float lambda_at(int k, float u, float jh) {
    static const float hill[NL] = {0, 0.45f, 0.65f, 0.85f}, mtn[NL] = {0, 0.32f, 0.5f, 0.7f};
    float l = hill[k] + (mtn[k] - hill[k]) * u;
    return k < 2 ? l : l + (1 - l) * 0.75f * (1 - smoothstep(4, 70, jh));
}

/* warped space: the network is laid out in it */
static void warp(float x, float y, float *qx, float *qy) {
    *qx = x + (fbm(x / 9000, y / 9000, 3, 71) - 0.5f) * 1800 + (fbm(x / 1700, y / 1700, 2, 73) - 0.5f) * 260;
    *qy = y + (fbm(x / 9000 + 5.2f, y / 9000 + 1.3f, 3, 72) - 0.5f) * 1800 + (fbm(x / 1700 + 3.7f, y / 1700 - 8.1f, 2, 74) - 0.5f) * 260;
}

static float smin(float a, float b, float k) {
    if (k <= 0) return fminf(a, b);
    float h = fmaxf(k - fabsf(a - b), 0) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

/* the node of level k in cell (i, j), with the pieces cached */
#define NC 256
typedef struct { int i, j, st, tst; float x, y, c, tl, tr; int down; Node n; } Slot;
static Slot *ncache[NL];
static Slot *slot(int k, int i, int j) {
    if (!ncache[k]) ncache[k] = calloc(NC * NC, sizeof(Slot));
    Slot *s = &ncache[k][(j & (NC - 1)) * NC + (i & (NC - 1))];
    if (s->st && s->i == i && s->j == j) return s;
    s->i = i; s->j = j; s->st = 1; s->tst = 0;
    uint32_t h = hash2(i, j, 600 + k);
    s->x = (i + 0.15f + 0.7f * (h & 0xffff) / 65536.0f) * LS[k];
    s->y = (j + 0.15f + 0.7f * (h >> 16) / 65536.0f) * LS[k];
    s->c = k == 0 ? ctrl(s->x, s->y) : 0;
    s->down = -1;
    return s;
}
static const int NB[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
/* level 0: where the water goes, 8 for nowhere (a sink) */
static int down0(int i, int j) {
    Slot *s = slot(0, i, j);
    if (s->down >= 0) return s->down;
    float best = s->c, c;
    int d = 8;
    for (int k = 0; k < 8; k++)
        if ((c = slot(0, i + NB[k][0], j + NB[k][1])->c) < best) { best = c; d = k; }
    return slot(0, i, j)->down = d;
}
/* the floor at a level-0 node: a sink is a lake, if it's anywhere low */
static float floor0(int i, int j) {
    float c = slot(0, i, j)->c;
    if (down0(i, j) < 8) return c;
    return c < 90 ? fminf(c, 0) - 30 : c - 12;
}
/* nodes (up to depth steps) upstream of a level-0 node, itself included */
static int upstream0(int i, int j, int depth) {
    int n = 1;
    if (depth > 0)
        for (int k = 0; k < 8; k++)
            if (down0(i + NB[k][0], j + NB[k][1]) == ((k + 4) & 7)) n += upstream0(i + NB[k][0], j + NB[k][1], depth - 1);
    return n;
}

typedef struct { float e, ab, best, jx, jy, jh, rd, rs; int ri, rj; } Env;
static const Node *node_get(int k, int i, int j);
/* one stretch of valley, (ax, ay, ah) to (bx, by, bh), into the envelope */
static void valley(Env *o, float qx, float qy, float ax, float ay, float ah, float bx, float by, float bh,
                   float w, float lc, float R, float ks, float t0) {
    float vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy;
    float t = l2 > 0 ? ((qx - ax) * vx + (qy - ay) * vy) / l2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    float px = ax + vx * t, py = ay + vy * t, dx = qx - px, dy = qy - py;
    float d = sqrtf(dx * dx + dy * dy), fl = ah + (bh - ah) * t;
    /* a U, steepening as it rises: gentle low on the valley side, scree-steep
     * under the crest */
    /* the U opens out down the valley: its head is a steep headwall */
    lc *= 0.25f + 0.75f * smoothstep(0, 0.55f, t0 + 0.5f * t);
    /* a flat floor falls gently to a line meandering from side to side of it,
     * where the river runs */
    float dm = d;
    if (w > 40 && d < 2 * w && t > 0 && t < 1) {
        float l = sqrtf(l2), nw = fmaxf(1, floorf(l / (5 * w))), st = sinf(3.14159265f * t);
        float cr = ((qx - ax) * vy - (qy - ay) * vx) / l, m = 0.55f * w * st * sinf(3.14159265f * t * nw);
        dm = fabsf(cr - m);
    }
    float z = fmaxf(d - w, 0), a = R * z * z / (z + lc) * fminf(0.55f + z * (1 / 1300.0f), 1.1f) + 0.012f * fminf(dm, w), V = fl + a;
    float kk = ks * smoothstep(0, 2 * ks, fminf(o->ab, a));
    o->e = smin(o->e, V, kk);
    o->ab = fminf(o->ab, a);
    if (V < o->best) { o->best = V; o->jx = px; o->jy = py; o->jh = fl; }
    o->rd = d;                                          /* (for the caller) */
}
/* a corrie into the envelope: a floor rising gently, then the headwall
 * round it, steepening to crags at its rim (r^3), and on at that slope */
/* how far out from a corrie's middle, in its radii: it's wider than it's
 * deep, and opens out towards its lip */
static float corrie_r(const Node *n, float qx, float qy) {
    float ax = n->cx - n->x, ay = n->cy - n->y, al = sqrtf(ax * ax + ay * ay);
    ax /= al; ay /= al;
    float dx = qx - n->cx, dy = qy - n->cy, along = dx * ax + dy * ay, across = dy * ax - dx * ay;
    along /= along < 0 ? 1.6f : 1.0f;
    return sqrtf(along * along + across * across * 0.7f) / n->cr;
}
static void corrie(Env *o, float qx, float qy, const Node *n, float ks) {
    float r = corrie_r(n, qx, qy), fb = n->h - n->cq, hw = 0.5f * n->cr;
    float a = (r < 1 ? r * r * r : 1 + 3 * (r - 1)) * hw, V = fb + a;
    float kk = ks * smoothstep(0, 2 * ks, fminf(o->ab, a));
    o->e = smin(o->e, V, kk);
    o->ab = fminf(o->ab, a);
    if (V < o->best) { o->best = V; o->jx = n->cx; o->jy = n->cy; o->jh = fb; }
}
static const int ORD[5] = {0, 1, -1, 2, -2};
/* the envelope of the valleys of levels below kn at (qx, qy) in warped
 * space; ks the crests' softening, R the valley sides' slope */
static void env(float qx, float qy, int kn, float u, float R, Env *o) {
    float rd = 1e9f, ks = soft_at(u);
    o->e = o->ab = o->best = 1e9f; o->rs = 0; o->ri = o->rj = 0; o->jx = qx; o->jy = qy; o->jh = 0;
    for (int k = kn - 1; k >= 0; k--) {                /* (finest first: the CPU skips more) */
        /* level 0's segments reach two cells; the others' a cell and a half,
         * so the four cells nearest will do */
        float fx = qx / LS[k], fy = qy / LS[k];
        int ci = (int)ifloor(fx), cj = (int)ifloor(fy), sx = fx - ci < 0.5f ? -1 : 1, sy = fy - cj < 0.5f ? -1 : 1;
        int nw = k ? 4 : 5;
        float lc = LC[k] * (0.6f + 0.6f * u);
        /* the cell it's in first, then outwards, the nearer side first */
        for (int mj = 0; mj < nw; mj++)
            for (int mi = 0; mi < nw; mi++) {
                int di = ORD[mi] * sx, dj = ORD[mj] * sy;
                const Node *n = node_get(k, ci + di, cj + dj);
                if (n->cq <= 0) {
                    /* (on the CPU, skip what can't reach the envelope: its
                     * lowest floor at its nearest stays a crest's softening
                     * above it; the GPU's sum is the same without it) */
                    float bx = fmaxf(fmaxf(n->b0x - qx, qx - n->b1x), 0), by = fmaxf(fmaxf(n->b0y - qy, qy - n->b1y), 0);
                    float dm = sqrtf(bx * bx + by * by), z = fmaxf(dm - n->w, 0);
                    float a = R * z * z / (z + lc) * fminf(0.55f + z * (1 / 1300.0f), 1.1f);
                    if (fminf(n->h, n->th) + a >= o->e + ks && a >= o->ab) {
                        if (k == 0 && n->riv > 0 && dm < rd) {        /* (but a river's distance counts) */
                            Env t = *o;
                            float hm = 0.5f * (n->h + n->th);
                            valley(&t, qx, qy, n->x, n->y, n->h, n->mx, n->my, hm, n->w, lc, R, ks, 0);
                            float d = t.rd;
                            valley(&t, qx, qy, n->mx, n->my, hm, n->tx, n->ty, n->th, n->w, lc, R, ks, 0.5f);
                            d = fminf(d, t.rd);
                            if (d < rd) { rd = d; o->rs = n->riv; o->ri = ci + di; o->rj = cj + dj; }
                        }
                        continue;
                    }
                }
                float hm = 0.5f * (n->h + n->th);
                valley(o, qx, qy, n->x, n->y, n->h, n->mx, n->my, hm, n->w, lc, R, ks, 0);
                float d = o->rd;
                valley(o, qx, qy, n->mx, n->my, hm, n->tx, n->ty, n->th, n->w, lc, R, ks, 0.5f);
                d = fminf(d, o->rd);
                if (n->cq > 0) corrie(o, qx, qy, n, ks);
                if (k == 0 && n->riv > 0 && d < rd) { rd = d; o->rs = n->riv; o->ri = ci + di; o->rj = cj + dj; }
            }
    }
    o->rd = rd;
}

static const Node *node_get(int k, int i, int j) {
    Slot *s = slot(k, i, j);
    if (s->st == 2) return &s->n;
    Node n;
    float x = s->x, y = s->y, u = province(x, y), R = slope_at(x, y, u);
    n.x = x; n.y = y;
    n.w = flat_at(k, u);
    n.riv = 0;
    if (k == 0) {
        int d = down0(i, j);
        n.h = floor0(i, j);
        if (d < 8) {
            Slot *t = slot(0, i + NB[d][0], j + NB[d][1]);
            n.tx = t->x; n.ty = t->y; n.th = floor0(i + NB[d][0], j + NB[d][1]);
        } else { n.tx = x; n.ty = y; n.th = n.h; }
        int up = upstream0(i, j, 5);
        n.riv = smoothstep(1.5f, 5, (float)up) * (1 - smoothstep(260, 420, n.h));
    } else {
        Env e;
        env(x, y, k, u, R, &e);
        float dj = hypotf(e.jx - x, e.jy - y), lam = lambda_at(k, u, e.jh);
        if (dj <= 1.5f * LS[k]) {
            n.tx = e.jx; n.ty = e.jy; n.th = e.jh;
            n.h = fmaxf(e.jh + lam * (e.e - e.jh), e.jh + 1);
        } else {
            /* far from the valley below: on to the neighbour lowest on it */
            float best = e.e;
            int bi = -1;
            for (int m = 0; m < 8; m++) {
                Slot *t = slot(k, i + NB[m][0], j + NB[m][1]);
                Env f;
                float tu = province(t->x, t->y);
                env(t->x, t->y, k, tu, slope_at(t->x, t->y, tu), &f);
                if (f.e < best) { best = f.e; bi = m; }
            }
            if (bi >= 0) {
                Node t = *node_get(k, i + NB[bi][0], j + NB[bi][1]);
                n.tx = t.x; n.ty = t.y; n.th = t.h;
                n.h = t.h + fmaxf(lam * (e.e - best), 0.5f * lam * R * hypotf(t.x - x, t.y - y));
            } else { n.tx = x; n.ty = y; n.h = n.th = e.e; }
        }
    }
    /* high in the mountains, a valley's head may be a corrie */
    n.cq = n.cr = 0;
    if (k == 1 || k == 2) {
        uint32_t hc = hash2(i, j, 670 + k);
        float p = (hc & 0xffff) / 65536.0f, q = (hc >> 16) / 65536.0f;
        if (p < (k == 1 ? 0.36f : 0.1f) * smoothstep(0.5f, 0.75f, u) * smoothstep(350, 550, n.h)) {
            /* only where the ground rises well behind the valley's head:
             * there the scoop leaves a headwall */
            float vx = n.x - n.mx, vy = n.y - n.my, l = fmaxf(sqrtf(vx * vx + vy * vy), 1), cr = (k == 1 ? 430 : 250) * (0.8f + 0.4f * q);
            float bx = n.x + vx / l * 1.3f * cr, by = n.y + vy / l * 1.3f * cr;
            Env b;
            float bu = province(bx, by);
            env(bx, by, k, bu, slope_at(bx, by, bu), &b);
            if (b.e - n.h > 0.45f * cr) {
                n.cq = 12 + 20 * q;
                n.cr = cr;
                n.cx = n.x + vx / l * 0.8f * cr; n.cy = n.y + vy / l * 0.8f * cr;
            }
        }
    }
    /* the valley bends: through a point off its middle */
    float vx = n.tx - n.x, vy = n.ty - n.y, b = ((hash2(i, j, 650 + k) & 0xffff) / 65536.0f - 0.5f) * 0.6f;
    n.mx = n.x + 0.5f * vx - b * vy; n.my = n.y + 0.5f * vy + b * vx;
    n.b0x = fminf(fminf(n.x, n.mx), n.tx); n.b1x = fmaxf(fmaxf(n.x, n.mx), n.tx);
    n.b0y = fminf(fminf(n.y, n.my), n.ty); n.b1y = fmaxf(fmaxf(n.y, n.my), n.ty);
    s = slot(k, i, j);                                  /* (the recursion may have moved it) */
    s->n = n; s->st = 2;
    return &s->n;
}

/* a corrie's tarn: the level the water stands at (where it would spill,
 * found by flooding a grid over the corrie from its edges inwards), and how
 * far out (in corrie_r()) the water reaches; level -1000 if it holds none */
static float tarn_level(int k, int i, int j, float *reach) {
    const Node *n = node_get(k, i, j);
    *reach = 0;
    if (n->cq <= 0) return -1000;
    Slot *s = slot(k, i, j);
    if (s->tst) { *reach = s->tr; return s->tl; }
    Node c = *n;
    enum { G = 27 };
    static float hh[G * G], fl[G * G];
    static unsigned char done[G * G];
    static Hn hp[G * G * 8];
    float sp = 3.6f * c.cr / (G - 1);
    for (int y = 0; y < G; y++)
        for (int x = 0; x < G; x++) {
            float qx = c.cx + (x - G / 2) * sp, qy = c.cy + (y - G / 2) * sp, u = province(qx, qy);
            Env e;
            env(qx, qy, NL, u, slope_at(qx, qy, u), &e);
            hh[y * G + x] = e.e;
        }
    /* priority flood: fl = the level water would stand at over each cell */
    int nh = 0;
    memset(done, 0, sizeof done);
    for (int q = 0; q < G * G; q++) {
        int x = q % G, y = q / G;
        if (x == 0 || y == 0 || x == G - 1 || y == G - 1) { fl[q] = hh[q]; done[q] = 1; hpush(hp, &nh, (Hn){hh[q], q}); }
    }
    while (nh) {
        Hn v = hpop(hp, &nh);
        int x = v.i % G, y = v.i / G;
        for (int d = 0; d < 8; d++) {
            int X = x + NB[d][0], Y = y + NB[d][1], q = Y * G + X;
            if (X < 0 || Y < 0 || X >= G || Y >= G || done[q]) continue;
            done[q] = 1;
            fl[q] = fmaxf(hh[q], v.f);
            hpush(hp, &nh, (Hn){fl[q], q});
        }
    }
    /* the deepest flooded cell near the middle holds the tarn */
    int best = -1;
    float depth = 2;
    for (int q = 0; q < G * G; q++) {
        int x = q % G - G / 2, y = q / G - G / 2;
        if (x * x + y * y < (G / 4) * (G / 4) && fl[q] - hh[q] > depth) { depth = fl[q] - hh[q]; best = q; }
    }
    float tl = -1000, tr = 0;
    if (best >= 0) {
        tl = fl[best] - 0.6f;
        for (int q = 0; q < G * G; q++)
            if (fl[q] - 0.01f > hh[q] && fabsf(fl[q] - fl[best]) < 0.01f) {
                float r = corrie_r(&c, c.cx + (q % G - G / 2) * sp, c.cy + (q / G - G / 2) * sp);
                if (r > tr) tr = r;
            }
        tr += sp / c.cr;
    }
    s = slot(k, i, j);
    s->tl = tl; s->tr = tr; s->tst = 1;
    *reach = tr;
    return tl;
}

/* The ground at (x, z) metres: out = height (m, 0 = datum), distance to
 * the nearest river (km; rivers run along its zero set), woodland, and the
 * river's size (0 none .. 1). Keep in step with terrain() in FS_BAKE. */
static void terrain_(float x, float z, float out[5], int aux) {
    float qx, qy;
    warp(x, z, &qx, &qy);
    float u = province(qx, qy);
    Env e;
    env(qx, qy, NL, u, slope_at(qx, qy, u), &e);
    out[0] = e.e;
    if (!aux) return;
    out[1] = e.rd / 1000;
    out[2] = fbm(x / 1400 + 11.5f, z / 1400 - 3.3f, 3, 55);
    out[3] = e.rs;
    /* the tarns' level, if one is near */
    out[4] = -1000;
    for (int k = 1; k <= 2; k++) {
        int ci = (int)ifloor(qx / LS[k]), cj = (int)ifloor(qy / LS[k]);
        for (int dj = -2; dj <= 2; dj++)
            for (int di = -2; di <= 2; di++) {
                const Node *n = node_get(k, ci + di, cj + dj);
                if (n->cq <= 0) continue;
                float reach, tl = tarn_level(k, ci + di, cj + dj, &reach);
                if (corrie_r(n, qx, qy) < reach) out[4] = fmaxf(out[4], tl);
            }
    }
}
static void terrain(float x, float z, float out[5]) { terrain_(x, z, out, 1); }

/* height at world pixel (x, y) */
static float T(double x, double y) {
    float o[5];
    terrain_((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, o, 0);
    return o[0];
}
static void grad(double x, double y, double *gx, double *gy) {
    *gx = 0.5 * (T(x + 1, y) - T(x - 1, y));
    *gy = 0.5 * (T(x, y + 1) - T(x, y - 1));
}

/* the drift: where the middle of the screen is at time t, world px. Three
 * circles of different sizes and periods summed (an epicycle wandering over
 * ~300 km), so the path turns gently, never stops and never repeats. */
/* the pace: the drift lingers and glides rather than running steady.
 * Drift time runs at 1 + 0.55 cos + 0.25 cos of real time: from a fifth of
 * the speed (a few minutes lingering over one stretch of country) to nearly
 * twice it, never stopping, never going back. */
static double pace(double t) {
    return t + 0.55 * 110 * sin(t / 110 + ph2[0]) + 0.25 * 41 * sin(t / 41 + ph2[1]);
}

static void view_at(double t, double *x, double *y) {
    t = pace(t);
    double mx = 33000 * cos(t / 830 + ph[0]) + 6000 * cos(-t / 377 + ph[1]) + 90000 * cos(t / 5400 + ph[2]);
    double my = 33000 * sin(t / 830 + ph[0]) + 6000 * sin(-t / 377 + ph[1]) + 90000 * sin(t / 6950 + ph[3]);
    *x = mx / MPP; *y = my / MPP;
}
static float water_at(double t) { return 3.5f * (float)sin(t / 170 + ph[4]); }
#define WATER_MAX 3.5f

/* the sun: swinging back and forth across the north over ~16 minutes, from
 * west-south-west to north-east (never from the south, where relief reads
 * inside out), high at the middle of its swing and low at the ends, where
 * the shadows run long. az is the screen angle towards the sun (y down). */
static void sun_at(double t, float *az, float *el) {
    double th = t / 150 + ph2[2];
    *az = (float)(-0.6 * M_PI + 0.95 * sin(th));
    *el = (float)((15 + 25 * (0.5 + 0.5 * cos(2 * th))) * M_PI / 180);
}

/* ---- glyph atlas: a few faces at their exact pixel sizes, plus drawn
 * sprites, shelf-packed into one R8 texture ---- */
typedef struct { int x, y, w, h, bx, by; float adv; } Glyph;
typedef struct { Glyph g[96]; float px; } Face;   /* ' '..'~', then 127 = middle dot */
enum { F_NUM, F_SPOT, F_COORD, F_TITLE, F_SUB, F_SMALL, F_TOWN, F_PLACE, F_WATER, F_HILL, F_REGION, NF };
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

/* the walker's sprites: a dot of its track, and the ring of its pen, drawn
 * a pixel larger all round than they need so they filter smoothly when blit
 * at fractional positions */
static Glyph wk_dot, wk_pen;
static void make_walker_sprites(void) {
    for (int k = 0; k < 2; k++) {
        Glyph *g = k ? &wk_pen : &wk_dot;
        float r = k ? 5.5f * S : 1.35f * S, lw = fmaxf(1.0f, 1.1f * S);
        g->w = g->h = ((int)ceilf(r * 2) + 3) | 1;
        atlas_place(g->w, g->h, &g->x, &g->y);
        float c0 = g->w * 0.5f;
        for (int y = 0; y < g->h; y++)
            for (int x = 0; x < g->w; x++) {
                int c = 0;
                for (int j = 0; j < 16; j++) {
                    float d = hypotf(x + (j % 4 + 0.5f) / 4 - c0, y + (j / 4 + 0.5f) / 4 - c0);
                    c += k ? (fabsf(d - r + lw * 0.5f) < lw * 0.5f || d < 1.3f * S) : d < r;
                }
                atlas[(size_t)(g->y + y) * ATW + g->x + x] = (unsigned char)(c * 255 / 16);
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
    /* place names */
    load_font(&fonts[F_TOWN], "Linux Libertine O,serif", 17 * S);
    load_font(&fonts[F_PLACE], "Linux Libertine O,serif", 13.5f * S);
    load_font(&fonts[F_WATER], "Linux Libertine O,serif:italic", 15 * S);
    load_font(&fonts[F_HILL], "Linux Libertine O,serif:italic", 12 * S);
    load_font(&fonts[F_REGION], "Linux Libertine O,serif", 14 * S);
    make_sprites();
    make_walker_sprites();
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
        float ground[5];
        terrain((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, ground);
        if ((ground[1] < 0.03f && ground[3] > 0.05f) || L < ground[4] + 2) continue;   /* in a river or a tarn */
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
    Rgb road;                          /* road fill: carto_palette(), from the others */
} P;

/* the palettes: brass is the theme's own; the others are other kinds of map
 * sheet, so that the lock screen isn't the same gold every time */
static const char *pal_names[] = {"brass", "cream", "slate", "blueprint", "survey"};
enum { NPAL = sizeof pal_names / sizeof *pal_names };
static int pal = -1;                   /* -1: from the seed */

static void palette(void) {
    /* Aged Brass: bg pushed darker for a full screen, bg_alt, cream, fg_dim,
     * gold, dim gold, sage, amber; and for water a muted grey-green, not blue */
    Rgb bgd = rgb("#17120E"), bga = rgb("#3A2E25"), cream = rgb("#EFE3CE"), dim = rgb("#A89880");
    Rgb gold = rgb("#FFC332"), goldd = rgb("#D6A12A"), sage = rgb("#8FA672"), amber = rgb("#E9994A");
    Rgb slate = rgb("#7E9C9A");
    if (pal == 0) {
        /* brass: gold and amber lines on near-black brown */
        P.paper = bgd; P.paper2 = mixc(bgd, bga, 0.75f); P.margin = rgb("#100C09");
        P.hi = cream; P.lo = rgb("#000000"); P.hi_k = 0.14f; P.lo_k = 0.5f;
        P.thin = amber; P.a_thin = 0.5f;
        P.index = goldd; P.a_index = 0.78f;
        P.water = mixc(bgd, slate, 0.10f); P.waterink = slate;
        P.wood = sage; P.a_wood = 0.42f;
        P.grid = cream; P.a_grid = 0.09f;
        P.frame = dim; P.river = mixc(slate, cream, 0.15f);
        P.label = gold; P.spot = cream; P.tri = gold; P.title = gold; P.sub = dim; P.panel = rgb("#120E0B");
    } else if (pal == 1) {
        /* cream, a paper map: ink (the theme's bg) on paper, contours in
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
    } else if (pal == 2) {
        /* slate: pale blue-grey and white lines on deep slate, darker water */
        Rgb bg = rgb("#171B21"), bg2 = rgb("#2C333D"), pale = rgb("#C9D4E0"), white = rgb("#F2F5F8");
        Rgb mid = rgb("#8796A8"), sea = rgb("#6E9CC4");
        P.paper = bg; P.paper2 = mixc(bg, bg2, 0.8f); P.margin = rgb("#111419");
        P.hi = white; P.lo = rgb("#000000"); P.hi_k = 0.12f; P.lo_k = 0.5f;
        P.thin = mixc(pale, bg, 0.1f); P.a_thin = 0.42f;
        P.index = white; P.a_index = 0.72f;
        P.water = rgb("#0D1624"); P.waterink = sea;
        P.wood = rgb("#7FA39A"); P.a_wood = 0.36f;
        P.grid = pale; P.a_grid = 0.08f;
        P.frame = mid; P.river = mixc(sea, white, 0.2f);
        P.label = white; P.spot = pale; P.tri = white; P.title = white; P.sub = mid; P.panel = rgb("#13171C");
    } else if (pal == 3) {
        /* blueprint: cyan and white lines on Prussian blue, like a cyanotype */
        Rgb bg = rgb("#0F2A4A"), bg2 = rgb("#16375E"), cyan = rgb("#8FD8F0"), white = rgb("#EAF6FC");
        Rgb mid = rgb("#7FA8C8");
        P.paper = bg; P.paper2 = mixc(bg, bg2, 0.9f); P.margin = rgb("#0B2240");
        P.hi = white; P.lo = rgb("#04101F"); P.hi_k = 0.08f; P.lo_k = 0.45f;
        P.thin = cyan; P.a_thin = 0.45f;
        P.index = white; P.a_index = 0.8f;
        P.water = rgb("#0A1F3A"); P.waterink = cyan;
        P.wood = rgb("#9FE0E8"); P.a_wood = 0.3f;
        P.grid = white; P.a_grid = 0.1f;
        P.frame = mid; P.river = cyan;
        P.label = white; P.spot = white; P.tri = cyan; P.title = white; P.sub = mid; P.panel = rgb("#0C2544");
    } else {
        /* survey: a printed survey sheet, brown contours, blue water, a faint
         * green-to-buff layer tint, black lettering */
        Rgb paper = rgb("#F3F2E8"), low = rgb("#E3EBD2"), high = rgb("#EEDDBE"), ink = rgb("#1E1E1E");
        Rgb brown = rgb("#9A5B2A"), blue = rgb("#2F7BBF");
        P.paper = mixc(paper, low, 0.9f); P.paper2 = high; P.margin = paper;
        P.hi = rgb("#FFFFFF"); P.lo = rgb("#5A5A50"); P.hi_k = 0.3f; P.lo_k = 0.2f;
        P.thin = brown; P.a_thin = 0.6f;
        P.index = mixc(brown, ink, 0.2f); P.a_index = 0.9f;
        P.water = rgb("#C6DFF0"); P.waterink = blue;
        P.wood = rgb("#4E8A3E"); P.a_wood = 0.5f;
        P.grid = blue; P.a_grid = 0.14f;
        P.frame = ink; P.river = blue;
        P.label = brown; P.spot = ink; P.tri = ink; P.title = ink; P.sub = mixc(ink, paper, 0.4f);
        P.panel = paper;
    }
}

/* the cartography's one colour of its own */
static void carto_palette(void) { P.road = mixc(P.paper, P.label, 0.62f); }

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
    )

/* the ground, ported from env(), corrie() and terrain_() above: the nodes
 * come from the CPU's cache in uNodes, four texels each (x, y, h, w), (mx,
 * my, tx, ty), (th, riv, cq, cr), (cx, cy, tarn level, tarn reach), for
 * cells uNB[k] .. uNB[k] + NT - 1 of level k */
#define TERRAIN_GLSL GLSL( \
    uniform sampler2D uNodes; \
    uniform ivec2 uNB[4]; \
    uniform int uNT; \
    const float LS[4] = float[4](7200.0, 2400.0, 800.0, 267.0); \
    const float LC[4] = float[4](800.0, 420.0, 110.0, 40.0); \
    const int ORD[5] = int[5](0, 1, -1, 2, -2); \
    float province(vec2 p) { \
        return smoothstep(0.3, 0.7, fbm(vec2((0.8 * p.x - 0.6 * p.y) / 60000.0, (0.6 * p.x + 0.8 * p.y) / 24000.0), 3, 23u)); \
    } \
    float slope_at(vec2 p, float u) { return (0.08 + 0.56 * u * u) * (0.7 + 0.6 * fbm(p / 4000.0, 2, 29u)); } \
    float soft_at(float u) { return 150.0 - 125.0 * u; } \
    vec2 warp(vec2 p) { \
        return p + vec2((fbm(p / 9000.0, 3, 71u) - 0.5) * 1800.0 + (fbm(p / 1700.0, 2, 73u) - 0.5) * 260.0, \
                        (fbm(p / 9000.0 + vec2(5.2, 1.3), 3, 72u) - 0.5) * 1800.0 + (fbm(p / 1700.0 + vec2(3.7, -8.1), 2, 74u) - 0.5) * 260.0); \
    } \
    float smin(float a, float b, float k) { \
        if (k <= 0.0) return min(a, b); \
        float h = max(k - abs(a - b), 0.0) / k; \
        return min(a, b) - h * h * k * 0.25; \
    } \
    vec4 nd(int k, ivec2 c, int m) { \
        ivec2 t = c - uNB[k]; \
        return texelFetch(uNodes, ivec2(t.x * 4 + m, k * uNT + t.y), 0); \
    } \
    float valley(inout float e, inout float ab, vec2 q, vec2 a, float ah, vec2 b, float bh, float w, float lc, float R, float ks, float t0) { \
        vec2 v = b - a; \
        float l2 = v.x * v.x + v.y * v.y; \
        float t = l2 > 0.0 ? ((q.x - a.x) * v.x + (q.y - a.y) * v.y) / l2 : 0.0; \
        t = clamp(t, 0.0, 1.0); \
        vec2 dd = q - (a + v * t); \
        float d = sqrt(dd.x * dd.x + dd.y * dd.y), fl = ah + (bh - ah) * t; \
        lc *= 0.25 + 0.75 * smoothstep(0.0, 0.55, t0 + 0.5 * t); \
        float dm = d; \
        if (w > 40.0 && d < 2.0 * w && t > 0.0 && t < 1.0) { \
            float l = sqrt(l2), nw = max(1.0, floor(l / (5.0 * w))), st = sin(3.14159265 * t); \
            float cr = ((q.x - a.x) * v.y - (q.y - a.y) * v.x) / l, m = 0.55 * w * st * sin(3.14159265 * t * nw); \
            dm = abs(cr - m); \
        } \
        float z = max(d - w, 0.0), aa = R * z * z / (z + lc) * min(0.55 + z * (1.0 / 1300.0), 1.1) + 0.012 * min(dm, w), V = fl + aa; \
        float kk = ks * smoothstep(0.0, 2.0 * ks, min(ab, aa)); \
        e = smin(e, V, kk); \
        ab = min(ab, aa); \
        return d; \
    } \
    float corrie_r(vec4 A, vec4 D, float cr, vec2 q) { \
        vec2 ax = D.xy - A.xy; \
        ax /= sqrt(ax.x * ax.x + ax.y * ax.y); \
        vec2 dd = q - D.xy; \
        float along = dd.x * ax.x + dd.y * ax.y, across = dd.y * ax.x - dd.x * ax.y; \
        along /= along < 0.0 ? 1.6 : 1.0; \
        return sqrt(along * along + across * across * 0.7) / cr; \
    } \
    /* height, distance to a river (km), woodland, river size; tarn level */ \
    vec4 terrain(vec2 p, out float tarn) { \
        vec2 q = warp(p); \
        float u = province(q), R = slope_at(q, u), ks = soft_at(u); \
        float e = 1e9, ab = 1e9, rd = 1e9, rs = 0.0; \
        for (int k = 3; k >= 0; k--) { \
            vec2 f = q / LS[k]; \
            ivec2 c = ivec2(floor(f)); \
            ivec2 sg = ivec2(f.x - float(c.x) < 0.5 ? -1 : 1, f.y - float(c.y) < 0.5 ? -1 : 1); \
            int nw = k > 0 ? 4 : 5; \
            float lc = LC[k] * (0.6 + 0.6 * u); \
            for (int mj = 0; mj < nw; mj++) \
                for (int mi = 0; mi < nw; mi++) { \
                    ivec2 cc = c + ivec2(ORD[mi], ORD[mj]) * sg; \
                    vec4 A = nd(k, cc, 0), B = nd(k, cc, 1), C = nd(k, cc, 2); \
                    float hm = 0.5 * (A.z + C.x); \
                    float d = valley(e, ab, q, A.xy, A.z, B.xy, hm, A.w, lc, R, ks, 0.0); \
                    d = min(d, valley(e, ab, q, B.xy, hm, B.zw, C.x, A.w, lc, R, ks, 0.5)); \
                    if (C.z > 0.0) { \
                        vec4 D = nd(k, cc, 3); \
                        float r = corrie_r(A, D, C.w, q), hw = 0.5 * C.w; \
                        float aa = (r < 1.0 ? r * r * r : 1.0 + 3.0 * (r - 1.0)) * hw, V = A.z - C.z + aa; \
                        float kk = ks * smoothstep(0.0, 2.0 * ks, min(ab, aa)); \
                        e = smin(e, V, kk); \
                        ab = min(ab, aa); \
                    } \
                    if (k == 0 && C.y > 0.0 && d < rd) { rd = d; rs = C.y; } \
                } \
        } \
        tarn = -1000.0; \
        for (int k = 1; k <= 2; k++) { \
            ivec2 c = ivec2(floor(q / LS[k])); \
            for (int dj = -2; dj <= 2; dj++) \
                for (int di = -2; di <= 2; di++) { \
                    ivec2 cc = c + ivec2(di, dj); \
                    vec4 C = nd(k, cc, 2); \
                    if (C.z <= 0.0) continue; \
                    vec4 A = nd(k, cc, 0), D = nd(k, cc, 3); \
                    if (corrie_r(A, D, C.w, q) < D.w) tarn = max(tarn, D.z); \
                } \
        } \
        return vec4(e, rd / 1000.0, fbm(p / 1400.0 + vec2(11.5, -3.3), 3, 55u), rs); \
    })

/* bake: texel t of the toroidal height texture holds world pixel
 * uB0 + ((t - uB0) mod N) */
static const char *FS_BAKE = "#version 330 core\n" NOISE_GLSL TERRAIN_GLSL GLSL(
    uniform ivec2 uB0, uB0m, uN;
    uniform vec2 uSeed;
    uniform float uMpp;
    layout(location = 0) out float oH;
    layout(location = 1) out vec4 oAux;
    void main() {
        ivec2 t = ivec2(gl_FragCoord.xy);
        ivec2 i = uB0 + (t - uB0m + uN) % uN;
        float tarn;
        vec4 g = terrain(vec2(i) * uMpp + uSeed, tarn);
        oH = g.x;
        oAux = vec4(g.yzw, tarn);
    });

static const char *FS_MAP = "#version 330 core\n" GLSL(
    uniform sampler2D uH, uAux, uMask;
    uniform sampler2D uNoise;   /* 256x256 RGBA of independent noise, repeating */
    uniform vec2 uRes, uN;
    uniform vec2 uOrgT;     /* world px of screen (0, 0), mod the texture size */
    uniform vec2 uOrgS;     /* ditto, mod the stipple period */
    uniform vec2 uOrgG;     /* world metres of screen (0, 0), mod 2 km */
    uniform vec4 uMap, uFrame;
    uniform float uMpp, uI, uWater, uSun, uS;
    uniform vec3 cPaper, cPaper2, cMargin, cHi, cLo, cThin, cIndex, cWater, cWaterInk, cWood, cGrid, cFrame, cRiver;
    uniform float aThin, aIndex, aGrid, aWood, kHi, kLo;
    uniform sampler2D uShade;                 /* the light: sun lit, cloud shadow */
    uniform float uT, kShadow, kCloud;
    uniform vec3 cRoad, cSpot;                          /* cartography */
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
            if (min(min(a.x, a.y), min(b.x, b.y)) < max(1.0, floor(uS + 0.5))) col = mix(col, cFrame, 0.8);
        }
        vec2 d = max(uMap.xy - sp, sp - uMap.zw);
        if (max(d.x, d.y) < max(1.0, floor(uS + 0.5))) col = mix(col, cFrame, 0.9);
        return col;
    }

    /* ---- the print: relief, hypsometric tint, illuminated contours, grain.
     * Everything here is driven by the palette's paper / hi / lo / wood /
     * line colours, so any palette gets the same treatment. ---- */
    /* noise from uNoise, one texel per k * uS px of the world-anchored
     * stipple frame; k divides 80, so the tile repeats whole across
     * uOrgS's wrap (5 * 4096 * uS px) and the grain never jumps */
    vec4 nz(vec2 ws, vec2 k) { return texture(uNoise, ws / (256.0 * k * uS)); }
    /* soft multi-directional hillshade, 0 on the flat, over a blend of a fine
     * and a broad normal: the sun's light; a fill from 90 degrees round it
     * that only lifts the slopes it faces (so slopes edge-on to the sun still
     * model); and steepness darkening whatever way the slope faces */
    float relief(vec2 g, vec2 gb) {
        vec2 sd = vec2(cos(uSun), sin(uSun));
        vec3 nf = normalize(vec3(-g * (2.6 / uMpp), 1.0)), nb = normalize(vec3(-gb * (3.4 / uMpp), 1.0));
        vec3 n = normalize(0.45 * nf + 0.55 * nb);
        float key = dot(n.xy, sd) * 0.7071 + (n.z - 1.0) * 0.7071;
        float fill = max(dot(n.xy, vec2(sd.y, -sd.x)) * 0.7071 + (n.z - 1.0) * 0.7071, 0.0);
        float shade = 0.85 * key + 0.3 * fill - 0.4 * (1.0 - n.z);
        /* a gentle s-curve: mid-slopes carry, the steepest don't burn out */
        return sign(shade) * (1.0 - exp(-2.4 * abs(shade)));
    }
    /* ground colour from height: low ground leans to the woodland ink, the
     * paper warms with height, and the high tops catch the light */
    vec3 hypso(float h) {
        vec3 c = mix(cPaper, cPaper2, 0.05 + 0.95 * smoothstep(-30.0, 900.0, h));
        c = mix(c, cWood, 0.05 * (1.0 - smoothstep(20.0, 260.0, h)));
        c = mix(c, cHi, 0.07 * smoothstep(600.0, 1150.0, h));
        return c;
    }
    /* paper grain: a fine tooth and a broad mottling, -1..1 */
    float grain(vec2 ws) {
        float tooth = nz(ws, vec2(1.25)).r - 0.5;
        float fibre = nz(ws, vec2(2.5, 5.0)).g - 0.5;              /* lengthwise fibres */
        float mott = nz(ws, vec2(40.0)).b * 0.6 + nz(ws, vec2(16.0)).a * 0.4 - 0.5;
        return tooth * 0.9 + fibre * 0.5 + mott * 1.1;
    }
    /* depression ticks, for a pixel at w a distance dt down from the contour
     * at level L, gn the unit way up, hxx and hyy the broad second
     * differences (m per px^2). The contour has to curve round the low side
     * (the divergence of gn, from the Hessian, is its curvature k), and for
     * one of a few sizes R a ring round a point R further down has to stand
     * above L all round, with dry, lower ground in the middle: then the line
     * closes round a hollow. Returns the tick coverage. */
    float hollow(vec2 w, vec2 gn, float gl, float hxx, float hyy, float L, float dt) {
        float tl = 3.6 * uS;
        if (dt > tl + 1.0) return 0.0;
        vec2 e = vec2(5.0 * uS, -5.0 * uS);
        float hxy = (Hh(w + e.xx) + Hh(w + e.yy) - Hh(w + e.xy) - Hh(w + e.yx)) / (4.0 * e.x * e.x);
        float k = (hxx * gn.y * gn.y - 2.0 * hxy * gn.x * gn.y + hyy * gn.x * gn.x) / gl;
        if (k < 1.0 / (60.0 * uS) || k > 0.25 / uS) return 0.0;      /* too open, or too small to tick */
        /* twelve ticks round the loop, set by the direction of the slope;
         * only the pixels under a tick go on to ask whether it's a hollow */
        vec2 dn = -gn;
        float u = atan(dn.y, dn.x) * 12.0 / 6.2831853;
        float du = abs(fract(u + 0.5) - 0.5) * 6.2831853 / 12.0 / k;
        float tick = cov(du, 0.9 * uS) * clamp(tl + 0.5 - dt, 0.0, 1.0);
        if (tick <= 0.0) return 0.0;
        vec2 p0 = w + gn * dt, tn = vec2(-dn.y, dn.x);
        for (int i = 0; i < 3; i++) {
            float R = uS * (i == 0 ? 6.0 : i == 1 ? 13.0 : 26.0);
            vec2 c = p0 + dn * R;
            float hc = Hh(c);
            if (hc < max(uWater, Ax(c).w) + 1.0 || hc > L - 0.5) continue;   /* a lake's basin, or not lower */
            bool found = true;
            for (int j = 0; j < 9 && found; j++) {
                float a = 0.6981317 * float(j);               /* the far side first */
                vec2 q = c + 1.3 * R * (dn * cos(a) + tn * sin(a));
                vec2 qs = q - uOrgT;                          /* off the baked ground: no */
                found = Hh(q) > L && all(greaterThan(qs, vec2(-7.0))) && all(lessThan(qs, uRes + 7.0));
            }
            if (found) return tick;
        }
        return 0.0;
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
        vec4 mw = texelFetch(uMask, ivec2(gl_FragCoord.xy), 0);   /* r: knock-out; g, b, a: cartography */
        float knock = max(mw.r, clamp(mw.b * 4.0, 0.0, 1.0));     /* roads clear a gap too */

        /* paper, tinted by height, and the hillshade: a broad gradient and
         * hollowness from four samples further out soften and deepen it */
        vec3 col = hypso(h);
        float rb = 5.0 * uS;
        float hxp = Hh(w + ex * rb), hxm = Hh(w - ex * rb), hyp = Hh(w + ey * rb), hym = Hh(w - ey * rb);
        vec2 gb = vec2(hxp - hxm, hyp - hym) / (2.0 * rb);
        float sh = relief(g, gb);
        float cav = (hxp + hxm + hyp + hym) * 0.25 - h;     /* m: + in hollows, - on crests */
        col = sh > 0.0 ? mix(col, cHi, min(sh, 1.0) * kHi * 1.5) : mix(col, cLo, min(-sh, 1.0) * kLo * 1.15);
        col = mix(col, cLo, smoothstep(0.0, 14.0 * uMpp / 8.0, cav) * kLo * 0.45);
        col = mix(col, cHi, smoothstep(0.0, 14.0 * uMpp / 8.0, -cav) * kHi * 0.5);
        vec2 ws = sp + uOrgS;
        float grn = grain(ws);

        /* water: below datum, a soft shore */
        float wlv = max(uWater, ax.w);                      /* the sea's level, or a tarn's */
        float dsh = (h - wlv) / gl;                      /* px from the shore, + on land */
        float wet = clamp(0.5 - dsh, 0.0, 1.0);
        col = mix(col, cWater, wet);
        col = mix(col, cLo, wet * smoothstep(0.0, 60.0, wlv - h) * kLo * 0.6);   /* deeper, darker */

        /* woodland stipple */
        float slope = gl / uMpp;
        float dens = smoothstep(0.56, 0.7, ax.y) * (1.0 - smoothstep(550.0, 800.0, h)) *
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

        /* and in the thicker woods here and there a tree: a ring on a stalk,
         * one at most to a 20 px world cell */
        if (dens > 0.4) {
            float ct = 20.0 * uS;
            vec2 ws = sp + uOrgS, c = floor(ws / ct);
            uvec2 uc = uvec2(ivec2(c) & 4095);
            if (r01(uc, 13u) < 0.45 * smoothstep(0.4, 0.9, dens)) {
                vec2 ctr = (c + 0.25 + 0.5 * vec2(r01(uc, 14u), r01(uc, 15u))) * ct;
                float ring = cov(abs(length(ws - ctr) - 2.3 * uS), 0.9 * uS);
                float stalk = cov(abs(ws.x - ctr.x), 0.9 * uS) * step(ctr.y + 2.3 * uS, ws.y) * step(ws.y, ctr.y + 5.5 * uS);
                col = mix(col, cWood, max(ring, stalk) * min(1.0, aWood * 1.8) * (1.0 - knock));
            }
        }

        /* kilometre grid */
        vec2 dg = abs(fract(gm / 1000.0 + 0.5) - 0.5) * 1000.0 / uMpp;
        col = mix(col, cGrid, max(cov(dg.x, max(1.0, uS)), cov(dg.y, max(1.0, uS))) * aGrid * (1.0 - knock));

        /* waterlines: depths D (k / 1)^(4/3) out from the shore, fading */
        if (wet > 0.0) {
            float depth = max(wlv - h, 1e-3), D1 = 9.0;
            float q = pow(depth / D1, 0.75);
            float gq = 0.75 * pow(depth / D1, -0.25) / D1 * gl;
            float k = floor(q + 0.5);
            float dq = abs(q - k) / max(gq, 1e-5);
            float wl = k >= 1.0 && k <= 5.0 ? cov(dq, 0.8 * uS) * exp(-0.42 * (k - 1.0)) * smoothstep(0.6, 2.5, 1.0 / gq) : 0.0;
            col = mix(col, cWaterInk, wl * 0.5 * wet);
        }

        /* the water moving: two swash lines at a time running in to the
         * shore and fading as they go, and glints twinkling out on the open
         * water where a slow band of breeze catches the sun */
        if (wet > 0.0) {
            float q = pow(max(wlv - h, 1e-3) / 9.0, 0.75);
            float gq = max(0.75 * pow(max(wlv - h, 1e-3) / 9.0, -0.25) / 9.0 * gl, 1e-5);
            float var = 0.5 * sin(6.2832 * (gm.x / 2000.0 + gm.y / 1000.0)) + 0.5 * sin(6.2832 * gm.y / 2000.0);
            float sw = 0.0;
            for (int k = 0; k < 2; k++) {
                float p = fract(uT / 13.0 + 0.5 * float(k) + 0.08 * var);
                float d = abs(q - (1.45 - 1.35 * p)) / gq;
                sw = max(sw, cov(d, 0.75 * uS) * sin(3.14159 * p) * sin(3.14159 * p));
            }
            col = mix(col, cWaterInk, sw * 0.45 * wet * smoothstep(0.6, 2.0, 1.0 / gq));
            float cs = 10.0 * uS;
            vec2 ws = sp + uOrgS;
            vec2 c = floor(ws / cs);
            uvec2 uc = uvec2(ivec2(c) & 2047);
            float band = smoothstep(0.35, 0.95, 0.5 + 0.5 * sin(6.2832 * (gm.x / 2000.0 - gm.y / 1000.0) - uT * 0.05));
            if (r01(uc, 7u) < 0.2 * band && dsh < -6.0) {
                vec2 ctr = (c + 0.2 + 0.6 * vec2(r01(uc, 8u), r01(uc, 9u))) * cs;
                float tw = pow(max(sin(uT * (0.5 + 0.4 * r01(uc, 11u)) + 6.2832 * r01(uc, 10u)), 0.0), 10.0);
                float dot_ = clamp(0.9 * uS + 0.5 - length(ws - ctr), 0.0, 1.0);
                col = mix(col, cHi, dot_ * tw * 0.75 * smoothstep(-6.0, -14.0, dsh));
            }
        }

        /* the coast: a vignette inside the shore and engraved hatching,
         * level lines close in, thinning and parting further out */
        if (wet > 0.0 && dsh > -40.0 * uS) {
            float ds = -dsh / uS;
            col = mix(col, cWaterInk, 0.16 * exp(-ds / 7.0) * wet);
            float hp = 3.2 * uS, hy = (sp.y + uOrgS.y) / hp;  /* periods dividing uOrgS's */
            float dh = abs(fract(hy) - 0.5) * hp;
            float hx = (sp.x + uOrgS.x) / (40.0 * uS);
            ivec2 hc = ivec2(floor(hy), floor(hx));
            float run = mix(r01(uvec2(hc & 4095), 7u), r01(uvec2((hc + ivec2(0, 1)) & 4095), 7u), smoothstep(0.0, 1.0, fract(hx)));
            float reach = 8.0 + 30.0 * run;
            col = mix(col, cWaterInk, cov(dh, 0.55 * uS) * 0.45 * (1.0 - smoothstep(0.4 * reach, reach, ds)) * wet);
        }

        /* contours, illuminated (Tanaka style): brighter and a touch bolder where the
         * slope faces the sun, darker and finer in its shade; intermediates
         * thin and then fade as they crowd, so steep ground stays legible;
         * a little ink spread along every line */
        float lines = (1.0 - wet) * (1.0 - knock);
        float spc = uI / gl;                                 /* px between contours */
        vec2 gn = g / gl;                                    /* unit, uphill */
        float face = -dot(gn, vec2(cos(uSun), sin(uSun)));
        float lit = max(face, 0.0), shd = max(-face, 0.0);
        float ink = 0.86 + 0.14 * nz(ws, vec2(2.5)).a + 0.03 * grn;
        float f = h / uI;
        float dt = abs(fract(f + 0.5) - 0.5) * spc;
        float wt = 0.95 * uS * (1.0 + 0.25 * face) * mix(0.6, 1.0, smoothstep(2.5, 7.0, spc));
        float thin = cov(dt, wt) * smoothstep(2.2, 4.6, spc) * ink;
        float f5 = h / (5.0 * uI);
        float di = abs(fract(f5 + 0.5) - 0.5) * spc * 5.0;
        float wi = 1.6 * uS * (1.0 + 0.35 * face) * mix(0.7, 1.0, smoothstep(5.0, 16.0, spc * 5.0));
        float fi = smoothstep(2.6, 5.5, spc * 5.0);
        float idx = cov(di, wi) * fi * ink;
        float halo = exp(-di * di / (6.0 * uS * uS)) * fi * 0.16 * (0.4 + 0.6 * lit);
        vec3 lthin = mix(mix(cThin, cHi, 0.4 * lit), cLo, 0.4 * shd);
        vec3 lidx = mix(mix(cIndex, cHi, 0.45 * lit), cLo, 0.42 * shd);
        float ath = aThin * (0.8 + 0.25 * lit - 0.3 * shd), aix = aIndex * (1.0 + 0.2 * lit - 0.12 * shd);
        col = mix(col, lidx, halo * aIndex * lines);
        col = mix(col, lthin, min(thin * ath, 1.0) * lines);
        /* depressions: ticks on the downhill side, on intermediates and index */
        if (spc > 3.5 && wet < 1.0) {
            float Lv = floor(f + 0.5) * uI;
            if (h < Lv) {
                float tk = hollow(w, gn, gl, (hxp + hxm - 2.0 * h) / (rb * rb), (hyp + hym - 2.0 * h) / (rb * rb), Lv, dt);
                bool isI = abs(fract(Lv / (5.0 * uI) + 0.5) - 0.5) < 0.1;
                col = mix(col, isI ? lidx : lthin, tk * (isI ? aix : ath) * lines * ink);
            }
        }
        col = mix(col, lidx, min(idx * aix, 1.0) * lines);

        /* crags: wedge hachures down the steepest ground, one to a 5 px world
         * cell, turned by the slope at the cell's middle */
        if (slope > 0.75 && wet < 0.5) {
            float cs = 5.0 * uS;
            vec2 ws = sp + uOrgS, c = floor(ws / cs), wc = (c + 0.5) * cs - uOrgS + uOrgT;
            vec2 gc = 0.5 * vec2(Hh(wc + ex) - Hh(wc - ex), Hh(wc + ey) - Hh(wc - ey));
            float sc = length(gc) / uMpp, k = smoothstep(1.05, 1.6, sc);
            uvec2 uc = uvec2(ivec2(c) & 4095);
            if (k > 0.0 && r01(uc, 9u) < 0.2 + 0.8 * k) {
                vec2 dir = gc / max(length(gc), 1e-5);       /* uphill */
                vec2 rel = ws - (c + 0.5 + 0.3 * (vec2(r01(uc, 10u), r01(uc, 11u)) - 0.5)) * cs;
                float al = dot(rel, dir), ac = abs(dot(rel, vec2(-dir.y, dir.x)));
                float len = cs * (0.22 + 0.12 * k + 0.16 * r01(uc, 12u));
                float wid = mix(0.25, 1.15, clamp(al / len * 0.5 + 0.5, 0.0, 1.0)) * uS;
                float hach = cov(ac, wid) * (1.0 - smoothstep(len - 0.5, len + 0.5, abs(al)));
                col = mix(col, cSpot, hach * 0.6 * k * (1.0 - knock));
            }
        }

        /* streams (G), roads (B: 0.25 the gap, 0.5 the casing, 1 the fill)
         * and tracks and paths (A), stroked into the mask on the CPU; names
         * knock them out */
        float nk = 1.0 - mw.r;
        col = mix(col, cRiver, mw.g * 0.9 * (1.0 - wet) * (1.0 - knock));
        col = mix(col, cSpot, clamp(mw.b * 4.0 - 1.0, 0.0, 1.0) * 0.85 * nk);
        col = mix(col, cRoad, clamp(mw.b * 2.0 - 1.0, 0.0, 1.0) * nk);
        col = mix(col, cSpot, mw.a * 0.75 * nk);

        /* the shoreline itself */
        col = mix(col, cWaterInk, cov(abs(dsh), 1.1 * uS) * 0.8);
        /* the paper's grain over everything printed on it */
        col = mix(col, grn > 0.0 ? cHi : cLo, min(abs(grn), 1.0) * (grn > 0.0 ? 0.022 : 0.04));

        /* the light over everything: the moving sun's cast shadows and the
         * drifting clouds' (from the quarter-resolution FS_SHADE pass) */
        vec2 csh = texture(uShade, gl_FragCoord.xy / uRes).rg;
        col = mix(col, cLo, min((1.0 - csh.r) * kShadow + csh.g * kCloud, 0.8));
        o = vec4(col, 1.0);
    });

/* the light, at quarter resolution (it is all soft): r = how lit by the sun,
 * marching from each place towards the sun through the height texture and
 * keeping the lowest clearance of the ray over the ground, as a soft
 * penumbra, the far steps counting for less so long shadows fade out at
 * their tips; g = cloud shadow, two layers of noise blown at different
 * winds over the ground so the clouds change shape as they go */
static const char *FS_SHADE = "#version 330 core\n" NOISE_GLSL GLSL(
    uniform sampler2D uH;
    uniform vec2 uRes, uN, uOrgT, uSunDir, uC1, uC2;
    uniform float uMpp, uWater, uRise, uReach, uCover, uDiv;
    out vec2 o;
    float Hs(vec2 w) { return max(texture(uH, (w + 0.5) / uN).r, uWater); }
    void main() {
        vec2 sp = vec2(gl_FragCoord.x, uRes.y / uDiv - gl_FragCoord.y) * uDiv;
        vec2 w = sp + uOrgT;
        float h0 = Hs(w) + 1.5;
        float lit = 1.0;
        for (int i = 1; i <= 28; i++) {
            float f = float(i) / 28.0, d = uReach * f * f;
            float clear = h0 + d * uRise - Hs(w + uSunDir * d);     /* m the ray clears the ground by */
            float l = clamp(0.5 + clear / (d * uMpp * 0.07 + 3.0), 0.0, 1.0);
            lit = min(lit, mix(l, 1.0, smoothstep(0.2, 1.0, f * f)));
        }
        vec2 gm = sp * uMpp;
        float n = 0.65 * fbm((gm + uC1) / 3000.0, 3, 501u) + 0.35 * fbm((gm + uC2) / 1300.0, 2, 502u);
        o = vec2(lit, smoothstep(uCover, uCover + 0.18, 0.5 + 3.0 * (n - 0.5)));
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
        else if (vMode == 2) {                  /* knock-out: a rounded box, uv local px */
            vec2 q = abs(vUV) - vCol.xy + vCol.z;
            float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - vCol.z;
            o = vec4(clamp(0.5 - d / 1.5, 0.0, 1.0), 0.0, 0.0, 0.0);
        }
        else                                    /* a stroke into the mask channels vCol */
            o = vCol * clamp(vUV.y + 0.5 - abs(vUV.x), 0.0, 1.0);
    });

static GLuint prog_bake, prog_map, prog_quad, vao, vao_q, vbo_q, tex_atlas;
static GLuint tex_h, tex_aux, fbo_h, tex_mask, fbo_mask;
static int TN, TM, tex_w, tex_h_, baked, bake_x, bake_y;
static GLuint prog_shade, tex_shade, fbo_shade;
static int shade_ok;                   /* tex_shade holds a frame's light (it is redone every other frame) */
static int BM = 8;                     /* ground baked beyond the screen each side: the shadow march's reach */
#define SH_REACH 230                   /* px at S = 1: how far a shadow can run */
#define SH_DIV 4                       /* the light pass's resolution divisor */

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
    BM = (int)ceilf(SH_REACH * S) + 8;
    TN = W + 2 * BM; TM = H + 2 * BM;
    tex_h = mk_tex(GL_R32F, TN, TM, GL_RED, GL_FLOAT, NULL, GL_LINEAR, GL_REPEAT);
    tex_aux = mk_tex(GL_RGBA16F, TN, TM, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_REPEAT);
    glGenFramebuffers(1, &fbo_h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_h);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_h, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_aux, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("height framebuffer incomplete");
    tex_mask = mk_tex(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo_mask);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_mask);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_mask, 0);
    if (tex_shade) { glDeleteTextures(1, &tex_shade); glDeleteFramebuffers(1, &fbo_shade); }
    shade_ok = 0;
    tex_shade = mk_tex(GL_RG8, (W + SH_DIV - 1) / SH_DIV, (H + SH_DIV - 1) / SH_DIV, GL_RG, GL_UNSIGNED_BYTE, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo_shade);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shade);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_shade, 0);
    baked = 0;
}

/* the paper's grain: a 256x256 RGBA texture of independent noise, left
 * bound on unit 3 for the map pass */
static void noise_tex(GLuint p) {
    unsigned char *n = malloc(256 * 256 * 4);
    for (int i = 0; i < 256 * 256 * 4; i++) n[i] = (unsigned char)(hash2(i, i >> 10, 4242) >> 24);
    glActiveTexture(GL_TEXTURE4);
    mk_tex(GL_RGBA8, 256, 256, GL_RGBA, GL_UNSIGNED_BYTE, n, GL_LINEAR, GL_REPEAT);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(U(p, "uNoise"), 4);
    free(n);
}

static void gl_init(void) {
    prog_bake = mk_prog(VS, FS_BAKE);
    prog_map = mk_prog(VS, FS_MAP);
    prog_quad = mk_prog(VS_QUAD, FS_QUAD);
    prog_shade = mk_prog(VS, FS_SHADE);
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
    glUniform1i(U(p, "uShade"), 3);
    /* shadows and cloud shadows darken towards the hillshade's dark side */
    glUniform1f(U(p, "kShadow"), 0.2f + 0.25f * P.lo_k);
    glUniform1f(U(p, "kCloud"), 0.1f + 0.15f * P.lo_k);
    noise_tex(p);
    u3(p, "cPaper", P.paper); u3(p, "cPaper2", P.paper2); u3(p, "cMargin", P.margin);
    u3(p, "cHi", P.hi); u3(p, "cLo", P.lo); u3(p, "cThin", P.thin); u3(p, "cIndex", P.index);
    u3(p, "cWater", P.water); u3(p, "cWaterInk", P.waterink); u3(p, "cWood", P.wood);
    u3(p, "cGrid", P.grid); u3(p, "cFrame", P.frame); u3(p, "cRiver", P.river);
    u3(p, "cRoad", P.road); u3(p, "cSpot", P.spot);
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

/* the network's nodes for the bake, from the CPU's cache: NT x NT cells of
 * each level, recentred when the ground to bake strays off them (every few
 * km of drift); the warp moves the ground at most ~1.1 km */
#define NT 96
static GLuint tex_nodes;
static int nb_ok, nb_x[NL], nb_y[NL];
static void nodes_cover(int nx, int ny) {
    float x0 = nx * MPP + seed_x - 1200, y0 = ny * MPP + seed_z - 1200;
    float x1 = (nx + TN) * MPP + seed_x + 1200, y1 = (ny + TM) * MPP + seed_z + 1200;
    int ok = nb_ok, nk[NL];
    for (int k = 0; k < NL; k++) nk[k] = (int)fminf(NT, 25600 / LS[k] + 6);    /* cells each level needs */
    for (int k = 0; k < NL && ok; k++)
        ok = (int)floorf(x0 / LS[k]) - 2 >= nb_x[k] && (int)floorf(x1 / LS[k]) + 2 < nb_x[k] + nk[k] &&
             (int)floorf(y0 / LS[k]) - 2 >= nb_y[k] && (int)floorf(y1 / LS[k]) + 2 < nb_y[k] + nk[k];
    if (ok) return;
    static float buf[NT * NL * NT * 4 * 4];
    for (int k = 0; k < NL; k++) {
        nb_x[k] = (int)floorf(0.5f * (x0 + x1) / LS[k]) - nk[k] / 2;
        nb_y[k] = (int)floorf(0.5f * (y0 + y1) / LS[k]) - nk[k] / 2;
        for (int j = 0; j < nk[k]; j++)
            for (int i = 0; i < nk[k]; i++) {
                int ci = nb_x[k] + i, cj = nb_y[k] + j;
                Node n = *node_get(k, ci, cj);
                float reach = 0, tl = n.cq > 0 ? tarn_level(k, ci, cj, &reach) : -1000;
                float v[16] = {n.x, n.y, n.h, n.w, n.mx, n.my, n.tx, n.ty, n.th, n.riv, n.cq, n.cr, n.cx, n.cy, tl, reach};
                memcpy(buf + ((size_t)(k * NT + j) * NT * 4 + i * 4) * 4, v, sizeof v);
            }
    }
    glActiveTexture(GL_TEXTURE5);
    if (!tex_nodes) tex_nodes = mk_tex(GL_RGBA32F, NT * 4, NT * NL, GL_RGBA, GL_FLOAT, buf, GL_NEAREST, GL_CLAMP_TO_EDGE);
    else { glBindTexture(GL_TEXTURE_2D, tex_nodes); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NT * 4, NT * NL, GL_RGBA, GL_FLOAT, buf); }
    glActiveTexture(GL_TEXTURE0);
    glUseProgram(prog_bake);
    glUniform1i(U(prog_bake, "uNodes"), 5);
    glUniform1i(U(prog_bake, "uNT"), NT);
    GLint nb[2 * NL];
    for (int k = 0; k < NL; k++) { nb[2 * k] = nb_x[k]; nb[2 * k + 1] = nb_y[k]; }
    glUniform2iv(U(prog_bake, "uNB"), NL, nb);
    nb_ok = 1;
}

/* make the height texture cover world px [nx, nx + TN) x [ny, ny + TM) */
static void bake(int nx, int ny) {
    if (baked && nx == bake_x && ny == bake_y) return;
    nodes_cover(nx, ny);
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
        if (getenv("CONTOURS_VERIFY")) {                /* the GPU's ground against the CPU's */
            float *hb = malloc(sizeof(float) * TN * TM), *ab = malloc(sizeof(float) * 4 * TN * TM);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glReadPixels(0, 0, TN, TM, GL_RED, GL_FLOAT, hb);
            glReadBuffer(GL_COLOR_ATTACHMENT1);
            glReadPixels(0, 0, TN, TM, GL_RGBA, GL_FLOAT, ab);
            int mx = posmod(nx, TN), my = posmod(ny, TM);
            double worst = 0, sum = 0, wa[4] = {0};
            int n = 0, big = 0;
            for (int k = 0; k < 40000; k++) {
                int tx = (int)(hash2(k, 1, 9) % TN), ty = (int)(hash2(k, 2, 9) % TM);
                int wx = nx + posmod(tx - mx, TN), wy = ny + posmod(ty - my, TM);
                float o[5];
                terrain(wx * MPP + seed_x, wy * MPP + seed_z, o);
                double d = fabs(o[0] - hb[ty * TN + tx]);
                if (d > worst) worst = d;
                if (d > 0.05) big++;
                sum += d; n++;
                for (int c = 0; c < 4; c++) {
                    double g = ab[4 * (ty * TN + tx) + c], cc = c < 3 ? o[c + 1] : o[4];
                    if (c == 0 && cc > 0.5) continue;   /* far from rivers: half floats */
                    if (fabs(g - cc) > wa[c]) wa[c] = fabs(g - cc);
                }
            }
            fprintf(stderr, "verify: height max |gpu-cpu| %.4f m, mean %.5f, >5cm %d/%d; aux max %.4f %.4f %.4f %.4f\n",
                    worst, sum / n, big, n, wa[0], wa[1], wa[2], wa[3]);
            free(hb); free(ab);
        }
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

/* ---- cartography: streams, settlements, roads and paths, place names ----
 * All of it hangs off world cells (CELL, cached apart from the labels'),
 * each a pure function of its cell, so nothing changes as the map drifts.
 *   - rivers and becks: from the head of each valley of the drainage
 *     network, traced downhill until they reach water, a tarn, a river or
 *     another beck (see "the becks");
 *   - settlements on low, dry, gentle ground, a few blocks round a centre;
 *   - ways: neighbouring settlements (a relative neighbourhood graph) joined
 *     by the cheapest route over the ground on a coarse grid (A*, climbing
 *     dear, height a little dear, water barred), which finds the valleys and
 *     the passes; a route that has to go high becomes a track or a footpath;
 *   - names, invented: settlements upright, water in italic (lakes, and
 *     rivers by the lake they run to), hills in small italic, regions in
 *     spaced capitals; lakes named (up to four) per 7x7 cells,
 *     a district per 4x4.
 * Streams, roads and paths go into the mask's G, B and A channels (R stays
 * the labels' knock-out; a road's B has three levels, gap, casing and fill)
 * and FS_MAP inks them, so where they overlap they merge. Live, cells and
 * becks are built a few ms a frame (a long river a slice at a time),
 * nearest first; offscreen, all at once. */
typedef struct { float x, y, w; } Pt;                  /* world px; half width or alpha */
typedef struct { int kind, n; Pt *p; } Way;
enum { W_MAIN, W_ROAD, W_TRACK, W_PATH };
typedef struct { int cx, cy, ok, has, size; double x, y; } Settle;
typedef struct { int cx, cy, ok, nw, river; Way w[8]; double rx, ry; float ra; char rname[32]; } Carto;
typedef struct { int rx, ry, ok, lake; double lx[4], ly[4]; char lname[4][32]; } Tract;
#define SK 64
#define CCK 32
#define RK 8
#define RG 7                                           /* cells per region */
static Settle settles[SK * SK];
static Carto cartos[CCK * CCK];
static Tract regions[RK * RK];

static const char *const NM_A[] = {
    "Ash", "Brack", "Thorn", "Hal", "Kel", "Mar", "Os", "Ral", "Sker", "Wen", "Gar", "Lang", "Bor", "Fen", "Hawk",
    "Cal", "Dun", "Ell", "Gris", "Har", "Kirk", "Lof", "Mel", "Nor", "Pen", "Rook", "Sel", "Wray", "Ulls", "War",
    "Yew", "Bran", "Cold", "Stan", "Whit", "Black", "Red", "Hart", "Swin", "Ald", "Brath", "Cray", "Esk", "Hest",
    "Lune", "Moss", "Ry", "Sty", "Thirl", "Wast", "Bar", "Dent", "Gis", "Haw", "Ing", "Kes", "Lin", "Mun", "Ned",
    "Ock", "Peel", "Rav", "Scar", "Tod", "Ulf", "Wyth", "Gow", "Arn", "Bleas", "Cop", "Dal", "Sca", "Grise"};
static const char *const NM_TOWN[] = {
    "by", "thwaite", "ton", "ham", "wick", "dale", "ford", "ley", "stead", "holme", "bury", "thorpe",
    "side", "garth", "with", "burn", "worth", "field", "bridge", "beck", "ing", "sett", "hope", "wood"};
#define NMN(a) ((uint32_t)(sizeof(a) / sizeof(a[0])))

/* an invented name from a hash: 0 settlement, 1 lake, 2 river, 3 hill, 4 high
 * region, 5 low region, 6 wooded region */
static void name_make(uint32_t h, int kind, char *out, int n) {
    const char *a = NM_A[h % NMN(NM_A)];
    uint32_t v = h / NMN(NM_A);
    char lo[24];
    snprintf(lo, sizeof lo, "%s", a);
    for (char *p = lo; *p; p++) *p = (char)tolower((unsigned char)*p);
    switch (kind) {
    case 0: snprintf(out, (size_t)n, "%s%s", a, NM_TOWN[v % NMN(NM_TOWN)]); break;
    case 1: {
        static const char *f[] = {"%swater", "%s Water", "%smere", "%s Mere", "%s Tarn", "%s Reservoir", "Loch %s"};
        snprintf(out, (size_t)n, f[v % 7], a);
        break;
    }
    case 2: {
        static const char *e[] = {"", "e", "en", "ey", "an"};
        const char *s = a[strlen(a) - 1] == 'e' || a[strlen(a) - 1] == 'y' ? "" : e[v % 5];
        snprintf(out, (size_t)n, "River %s%s", a, s);
        break;
    }
    case 3: {
        static const char *f[] = {"%s Pike", "%s Fell", "High %s", "%s Crag", "%s Knott", "%s Howe", "%s Edge", "%s Top"};
        snprintf(out, (size_t)n, f[v % 8], a);
        break;
    }
    default: {
        static const char *hi[] = {"%s FELLS", "%s MOOR", "HIGH %s", "%s FELL"};
        static const char *low[] = {"%s MOSS", "%s COMMON", "VALE OF %s", "%sDALE"};
        static const char *wd[] = {"%s FOREST", "%s CHASE", "%s WOODS"};
        char up[24];
        snprintf(up, sizeof up, "%s", a);
        for (char *p = up; *p; p++) *p = (char)toupper((unsigned char)*p);
        if (kind == 4) snprintf(out, (size_t)n, hi[v % 4], up);
        else if (kind == 5) snprintf(out, (size_t)n, low[v % 4], up);
        else snprintf(out, (size_t)n, wd[v % 3], up);
    }
    }
    (void)lo;
}

/* the settlement in a cell, if any: the gentlest of a few low dry spots */
static Settle *settle_get(int cx, int cy) {
    Settle *s = &settles[posmod(cy, SK) * SK + posmod(cx, SK)];
    if (s->ok && s->cx == cx && s->cy == cy) return s;
    memset(s, 0, sizeof *s);
    s->ok = 1; s->cx = cx; s->cy = cy;
    uint32_t h = hash2(cx, cy, 900 + opt_seed * 13);
    if ((h & 0xff) > 100) return s;
    double best = 1e9;
    float bh = 0;
    for (int k = 0; k < 6; k++) {
        uint32_t q = hash2(cx, cy, 910 + k + opt_seed * 13);
        double x = (cx + 0.12 + 0.76 * (q & 0xffff) / 65536.0) * CELL, y = (cy + 0.12 + 0.76 * (q >> 16) / 65536.0) * CELL;
        float t = T(x, y);
        if (t < WATER_MAX + 6 || t > 320) continue;
        double gx, gy;
        grad(x, y, &gx, &gy);
        double sl = sqrt(gx * gx + gy * gy) / MPP, c = sl * 5 + t / 400;
        if (sl < 0.14 && c < best) { best = c; s->x = x; s->y = y; bh = t; }
    }
    if (best > 1e8) return s;
    s->has = 1;
    s->size = ((h >> 8) & 7) == 0 && bh < 150 ? 2 : (h >> 11) & 1 ? 1 : 0;
    return s;
}

/* Douglas-Peucker: drop the points within tol px of the line; returns n */
static int dp_keep(const Pt *p, unsigned char *keep, int a, int b, float tol) {
    float dx = p[b].x - p[a].x, dy = p[b].y - p[a].y, l = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f), worst = 0;
    int wi = -1;
    for (int i = a + 1; i < b; i++) {
        float d = fabsf((p[i].x - p[a].x) * dy - (p[i].y - p[a].y) * dx) / l;
        if (d > worst) { worst = d; wi = i; }
    }
    if (wi < 0 || worst < tol) return 0;
    keep[wi] = 1;
    return 1 + dp_keep(p, keep, a, wi, tol) + dp_keep(p, keep, wi, b, tol);
}
static int simplify(Pt *p, int n, float tol) {
    if (n < 3) return n;
    unsigned char *keep = calloc((size_t)n, 1);
    keep[0] = keep[n - 1] = 1;
    dp_keep(p, keep, 0, n - 1, tol);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (keep[i]) p[m++] = p[i];
    free(keep);
    return m;
}

/* the cheapest route from a to b over the ground on a grid; 0 if none */
static int route(double ax, double ay, double bx, double by, Way *out, float *maxh) {
    double G = 12 * S, L = hypot(bx - ax, by - ay), pad = fmax(0.35 * L, 60 * S);
    double x0 = fmin(ax, bx) - pad, y0 = fmin(ay, by) - pad;
    int nx = (int)((fabs(bx - ax) + 2 * pad) / G) + 2, ny = (int)((fabs(by - ay) + 2 * pad) / G) + 2, N = nx * ny;
    float *ht = malloc(sizeof(float) * N * 2), *gs = ht + N;
    int *from = malloc(sizeof(int) * N);
    Hn *hp = malloc(sizeof(Hn) * N * 16);
    unsigned char *closed = calloc((size_t)N, 1);
    for (int i = 0; i < N; i++) { ht[i] = NAN; gs[i] = 1e30f; }
    int si = (int)lround((ax - x0) / G) + nx * (int)lround((ay - y0) / G);
    int gi_ = (int)lround((bx - x0) / G) + nx * (int)lround((by - y0) / G);
    int gx_ = gi_ % nx, gy_ = gi_ / nx, sx_ = si % nx, sy_ = si / nx;
    int nh = 0, found = 0;
    float dm = (float)(G * MPP);
    gs[si] = 0; from[si] = -1;
    hpush(hp, &nh, (Hn){0, si});
    while (nh) {
        Hn u = hpop(hp, &nh);
        if (closed[u.i]) continue;
        closed[u.i] = 1;
        if (u.i == gi_) { found = 1; break; }
        int ux = u.i % nx, uy = u.i / nx;
        if (isnan(ht[u.i])) ht[u.i] = T(x0 + ux * G, y0 + uy * G);
        for (int k = 0; k < 16; k++) {                 /* kings' and knights' moves */
            static const int dx[16] = {1, -1, 0, 0, 1, 1, -1, -1, 2, 2, -2, -2, 1, 1, -1, -1};
            static const int dy[16] = {0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 2, -2, 2, -2};
            int vx = ux + dx[k], vy = uy + dy[k];
            if (vx < 0 || vy < 0 || vx >= nx || vy >= ny) continue;
            int v = vy * nx + vx;
            if (closed[v]) continue;
            if (isnan(ht[v])) ht[v] = T(x0 + vx * G, y0 + vy * G);
            int nearend = abs(vx - gx_) + abs(vy - gy_) <= 2 || abs(vx - sx_) + abs(vy - sy_) <= 2;
            if (ht[v] < WATER_MAX + 1.5f && !nearend) continue;
            float d = k < 4 ? dm : k < 8 ? dm * 1.41421f : dm * 2.23607f, gr = fabsf(ht[v] - ht[u.i]) / d / 0.07f;
            float c = gs[u.i] + d * (1 + gr * gr + fmaxf(ht[v], 0) / 500);
            if (c < gs[v]) {
                gs[v] = c; from[v] = u.i;
                float hx = (float)((vx - gx_) * G * MPP), hy = (float)((vy - gy_) * G * MPP);
                hpush(hp, &nh, (Hn){c + 1.25f * sqrtf(hx * hx + hy * hy), v});
                if (nh >= N * 16 - 16) nh = 0;          /* give up */
            }
        }
    }
    int n = 0;
    if (found) {
        for (int i = gi_; i >= 0; i = from[i]) n++;
        if (gs[gi_] > 6 * L * MPP + 3000) found = 0;   /* too dear: no way */
    }
    if (found) {
        /* the grid path, its ends at the settlements, eased off the grid,
         * then Chaikin thrice */
        Pt *p = malloc(sizeof(Pt) * (size_t)n);
        int k = n;
        float mh = 0;
        for (int i = gi_; i >= 0; i = from[i]) {
            k--;
            p[k].x = (float)(x0 + (i % nx) * G); p[k].y = (float)(y0 + (i / nx) * G);
            if (ht[i] > mh) mh = ht[i];
        }
        p[0].x = (float)ax; p[0].y = (float)ay; p[n - 1].x = (float)bx; p[n - 1].y = (float)by;
        for (int r = 0; r < 5; r++) {
            Pt prev = p[0];
            for (int i = 1; i + 1 < n; i++) {
                Pt cur = p[i];
                p[i].x = 0.25f * prev.x + 0.5f * cur.x + 0.25f * p[i + 1].x;
                p[i].y = 0.25f * prev.y + 0.5f * cur.y + 0.25f * p[i + 1].y;
                prev = cur;
            }
        }
        for (int r = 0; r < 3 && n > 2; r++) {
            Pt *q = malloc(sizeof(Pt) * (size_t)(2 * n));
            int m = 0;
            q[m++] = p[0];
            for (int i = 0; i + 1 < n; i++) {
                q[m++] = (Pt){0.75f * p[i].x + 0.25f * p[i + 1].x, 0.75f * p[i].y + 0.25f * p[i + 1].y, 0};
                q[m++] = (Pt){0.25f * p[i].x + 0.75f * p[i + 1].x, 0.25f * p[i].y + 0.75f * p[i + 1].y, 0};
            }
            q[m++] = p[n - 1];
            free(p); p = q; n = m;
        }
        n = simplify(p, n, 0.2f);
        out->p = p; out->n = n; *maxh = mh;
    }
    free(ht); free(from); free(hp); free(closed);
    return found;
}

/* ---- the becks: the drainage network's rivers and becks, drawn ----
 * Each node of levels 0 (the rivers, wider the more runs into them), 1 and
 * 2 is the head of a beck: from a little way down its valley it's traced
 * downhill over the ground itself (not along the network's segment: a
 * tributary alongside may run a few metres lower and take the water, and
 * the line should be where the water is), until it reaches a lake, a tarn,
 * a river or a beck already traced. Points with w = half width, -w where it
 * rises (drawn fainter). Tracing stops where another beck has run, so the
 * work is the length of the network in view, not of every beck to the sea;
 * it goes a slice at a time when live. */
/* a beck being traced: ok 1 under way, 2 done */
typedef struct {
    int i, j, ok, n, cap, it, end, spill, stall, flat, id;
    double x, y, dx, dy, len, lowlen;
    float h, lowest, riv;
    Pt *p;
} Beck;
#define BK 64
static Beck becks[3][BK * BK];

/* warped space back to the map: Newton on warp() */
static void unwarp(float qx, float qy, float *px, float *py) {
    float x = qx, y = qy;
    for (int it = 0; it < 6; it++) {
        float wx, wy, ax, ay, bx, by;
        warp(x, y, &wx, &wy);
        float rx = wx - qx, ry = wy - qy;
        if (rx * rx + ry * ry < 0.01f) break;
        warp(x + 1, y, &ax, &ay); warp(x, y + 1, &bx, &by);
        float j11 = ax - wx, j21 = ay - wy, j12 = bx - wx, j22 = by - wy, det = j11 * j22 - j12 * j21;
        if (fabsf(det) < 1e-6f) break;
        x -= (j22 * rx - j12 * ry) / det; y -= (j11 * ry - j21 * rx) / det;
    }
    *px = x; *py = y;
}

/* where becks have run, so the next to get there can stop: 6 px cells */
#define OCC 65536
static struct { int x, y, id; } occ[OCC];
static int occ_at(int x, int y, int id, int mark) {
    uint32_t h = hash2(x, y, 31) & (OCC - 1);
    if (occ[h].id && occ[h].x == x && occ[h].y == y) return occ[h].id;
    if (mark) { occ[h].x = x; occ[h].y = y; occ[h].id = id; }
    return id;
}

/* the beck of node (k, i, j): NULL if it isn't traced and !build; build
 * traces it on by up to `steps` steps (0: to the end), and it's drawn once
 * it's done (ok == 2) */
static Beck *beck_get(int k, int i, int j, int build, int steps) {
    Beck *b = &becks[k][posmod(j, BK) * BK + posmod(i, BK)];
    int mine = b->ok && b->i == i && b->j == j;
    if (mine && b->ok == 2) return b;
    if (!build) return NULL;
    static const float w0[3] = {0.8f, 0.55f, 0.35f}, w1[3] = {1.05f, 0.85f, 0.6f};
    double step = (k ? 5 : 7) * S;
    if (!mine) {
        free(b->p);
        memset(b, 0, sizeof *b);
        b->ok = 1; b->i = i; b->j = j;
        Node n = *node_get(k, i, j);
        b->riv = n.riv;
        b->id = (int)(hash2(i, j, k) | 1);
        /* from the valley's head, a little way down it, downhill over the
         * ground itself (a tributary alongside may run lower and take the
         * water: it goes where the water goes) */
        float px, py, hx = n.x + 0.12f * (n.mx - n.x), hy = n.y + 0.12f * (n.my - n.y);
        unwarp(hx, hy, &px, &py);
        b->x = (px - seed_x) / MPP; b->y = (py - seed_z) / MPP;
        b->h = b->lowest = T(b->x, b->y);
        if (b->h < WATER_MAX) { b->ok = 2; return b; }
        b->cap = 128;
        b->p = malloc(sizeof(Pt) * (size_t)b->cap);
        b->p[b->n++] = (Pt){(float)b->x, (float)b->y, 0.01f};
    }
    int lim = k ? 240 : 900;
    for (int s_ = 0; b->it < lim && !b->end && (!steps || s_ < steps); s_++, b->it++) {
        double x = b->x, y = b->y, h = b->h;
        double gx = T(x + 1, y) - h, gy = T(x, y + 1) - h, gl = sqrt(gx * gx + gy * gy);
        if (b->spill) {                                /* across a hollow: straight on */
            if (++b->spill > 31) { b->end = 2; break; }
        } else if (gl > 1e-3) {
            double ex = -gx / gl, ey = -gy / gl;
            if (b->it) { ex += 0.6 * b->dx; ey += 0.6 * b->dy; }
            double el = hypot(ex, ey);
            if (el < 1e-6) { b->end = 2; break; }
            b->dx = ex / el; b->dy = ey / el;
            b->flat = 0;
        } else if (++b->flat > 15 || !b->it) { b->end = 2; break; }
        x += b->dx * step; y += b->dy * step; b->len += step;
        b->x = x; b->y = y;
        float hn = b->h = T(x, y);
        if (hn > b->lowest + 12) { b->end = 2; break; }      /* a hollow it can't spill from */
        if (hn < b->lowest - 0.02f) { b->lowest = hn; b->lowlen = b->len; b->spill = b->stall = 0; }
        else if (!b->spill && (hn > b->lowest + 1.5f || ++b->stall > 15)) b->spill = 1;
        if (hn < 0) b->end = 1;                        /* into the water */
        else if (b->it % 3 == 2) {
            float g[5];
            terrain((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, g);
            if ((k && g[1] < 0.015f && g[3] > 0.3f) || hn < g[4]) b->end = 1;   /* into a river or a tarn */
        }
        if (occ_at((int)floor(x / (6 * S)), (int)floor(y / (6 * S)), b->id, 0) != b->id && b->it > 2) b->end = 1;   /* into another beck */
        if (b->n == b->cap) b->p = realloc(b->p, sizeof(Pt) * (size_t)(b->cap *= 2));
        b->p[b->n++] = (Pt){(float)x, (float)y, (float)b->len};
    }
    if (!b->end && b->it < lim) return NULL;          /* more next time */
    /* done: lost in a hollow, only down to its lowest */
    Pt *p = b->p;
    int nn = b->n;
    if (b->end != 1) {
        while (nn > 1 && p[nn - 1].w > b->lowlen) nn--;
        if (b->lowlen < 120 * S) nn = 0;
    }
    if (b->len < 40 * S) nn = 0;
    /* smoothed: on a flat floor it wanders from side to side of the fall */
    for (int r = 0; r < 4; r++) {
        Pt prev = p[0];
        for (int m = 1; m + 1 < nn; m++) {
            Pt cur = p[m];
            p[m].x = 0.25f * prev.x + 0.5f * cur.x + 0.25f * p[m + 1].x;
            p[m].y = 0.25f * prev.y + 0.5f * cur.y + 0.25f * p[m + 1].y;
            prev = cur;
        }
    }
    for (int m = 0; m < nn; m++) {
        float l = p[m].w, w = S * (w0[k] + (w1[k] - w0[k]) * smoothstep(0, 600 * S, l) + (k ? 0 : 1.1f * b->riv));
        p[m].w = l < 25 * S ? -w : w;
    }
    /* where it runs, for the becks after it to join */
    for (int m = 0; m + 1 < nn; m++)
        for (int e = 0; e < 3; e++) {
            float x = p[m].x + (p[m + 1].x - p[m].x) * e / 3, y = p[m].y + (p[m + 1].y - p[m].y) * e / 3;
            occ_at((int)floorf(x / (6 * S)), (int)floorf(y / (6 * S)), b->id, 1);
        }
    b->n = nn ? simplify(p, nn, 0.25f) : 0;
    b->ok = 2;
    return b;
}

/* a point on a river in the cell, if one runs through it (onto the zero of
 * terrain()'s distance to the rivers), the way it runs and its name */
static int river_spot(int cx, int cy, double *ox_, double *oy_, float *oa, char *buf, int n) {
    double x = (cx + 0.5) * CELL, y = (cy + 0.5) * CELL;
    float g[5];
    int ok = 0;
    for (int it = 0; it < 6; it++) {           /* Newton onto rv = 0 */
        terrain((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, g);
        float e[5], f2[5];
        terrain((float)((x + 1) * MPP) + seed_x, (float)(y * MPP) + seed_z, e);
        terrain((float)(x * MPP) + seed_x, (float)((y + 1) * MPP) + seed_z, f2);
        double rx = e[1] - g[1], ry = f2[1] - g[1], r2 = rx * rx + ry * ry;
        if (r2 < 1e-14) break;
        double k = g[1] / r2;
        if (fabs(k) * sqrt(r2) > CELL * 0.5) break;
        x -= k * rx; y -= k * ry;
        if (fabsf(g[1]) < 1e-4f) { ok = 1; break; }
    }
    if (!ok || g[3] < 0.7f || g[0] < WATER_MAX + 2) return 0;
    if (x < cx * CELL || x >= (cx + 1) * CELL || y < cy * CELL || y >= (cy + 1) * CELL) return 0;
    float e[5];
    terrain((float)((x + 1) * MPP) + seed_x, (float)(y * MPP) + seed_z, e);
    float f2[5];
    terrain((float)(x * MPP) + seed_x, (float)((y + 1) * MPP) + seed_z, f2);
    float a = atan2f(-(e[1] - g[1]), f2[1] - g[1]);   /* along the river */
    if (a > (float)M_PI / 2) a -= (float)M_PI;
    if (a <= -(float)M_PI / 2) a += (float)M_PI;
    /* which river: named for the lake it runs to */
    float qx, qy;
    warp((float)(x * MPP) + seed_x, (float)(y * MPP) + seed_z, &qx, &qy);
    Env ev;
    env(qx, qy, 1, 0.5f, 0.2f, &ev);
    int ri = ev.ri, rj = ev.rj;
    for (int k = 0, d; k < 64 && (d = down0(ri, rj)) < 8; k++) { ri += NB[d][0]; rj += NB[d][1]; }
    int id = (int)hash2(ri, rj, 77);
    name_make(hash2(id, 0, 1400 + opt_seed * 5), 2, buf, n);
    *ox_ = x; *oy_ = y; *oa = a;
    return 1;
}

static void carto_build(Carto *c, int cx, int cy) {
    for (int i = 0; i < c->nw; i++) free(c->w[i].p);
    memset(c, 0, sizeof *c);
    c->cx = cx; c->cy = cy; c->ok = 1;
    /* a river through it, for a name */
    if (!(hash2(cx, cy, 1300) & 1)) c->river = river_spot(cx, cy, &c->rx, &c->ry, &c->ra, c->rname, sizeof c->rname);
    /* ways to the neighbours this cell's settlement owns */
    Settle *a = settle_get(cx, cy);
    if (!a->has) return;
    for (int by = cy - 3; by <= cy + 3; by++)
        for (int bx = cx - 3; bx <= cx + 3; bx++) {
            if (by < cy || (by == cy && bx <= cx) || c->nw == 8) continue;
            Settle *b = settle_get(bx, by);
            if (!b->has) continue;
            double ab = hypot(b->x - a->x, b->y - a->y);
            if (ab > 3.1 * CELL) continue;
            int rng = 1;
            for (int qy = cy - 4; qy <= cy + 4 && rng; qy++)
                for (int qx = cx - 4; qx <= cx + 4 && rng; qx++) {
                    Settle *q = settle_get(qx, qy);
                    if (!q->has || q == a || q == b) continue;
                    if (hypot(q->x - a->x, q->y - a->y) < ab && hypot(q->x - b->x, q->y - b->y) < ab) rng = 0;
                }
            if (!rng) continue;
            Way *w = &c->w[c->nw];
            float mh;
            if (!route(a->x, a->y, b->x, b->y, w, &mh)) continue;
            w->kind = mh > 300 ? W_PATH : mh > 190 ? W_TRACK : a->size + b->size >= 3 ? W_MAIN : W_ROAD;
            c->nw++;
        }
}

static Carto *carto_get(int cx, int cy, int build) {
    Carto *c = &cartos[posmod(cy, CCK) * CCK + posmod(cx, CCK)];
    if (!c->ok || c->cx != cx || c->cy != cy) {
        if (!build) return NULL;
        carto_build(c, cx, cy);
    }
    return c;
}

/* lake labels: the deepest water with room round it, then the next deepest
 * well away from those, up to four */
static Tract *region_get(int rx, int ry) {
    Tract *r = &regions[posmod(ry, RK) * RK + posmod(rx, RK)];
    if (r->ok && r->rx == rx && r->ry == ry) return r;
    memset(r, 0, sizeof *r);
    r->ok = 1; r->rx = rx; r->ry = ry;
    double R = RG * (double)CELL, x0 = rx * R, y0 = ry * R;
    struct { float t; double x, y; } c[144];
    int nc = 0;
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
            double x = x0 + (i + 0.5) * R / 12, y = y0 + (j + 0.5) * R / 12;
            float t = T(x, y);
            if (t > -6) continue;
            int room = 1;
            for (int k = 0; k < 6 && room; k++)
                if (T(x + 90 * S * cos(k * M_PI / 3), y + 35 * S * sin(k * M_PI / 3)) > -1) room = 0;
            if (room) { c[nc].t = t; c[nc].x = x; c[nc++].y = y; }
        }
    for (int i = 1; i < nc; i++)
        for (int j = i; j > 0 && c[j].t < c[j - 1].t; j--) { __typeof__(c[0]) s = c[j]; c[j] = c[j - 1]; c[j - 1] = s; }
    for (int i = 0; i < nc && r->lake < 4; i++) {
        int far = 1;
        for (int k = 0; k < r->lake && far; k++) far = hypot(c[i].x - r->lx[k], c[i].y - r->ly[k]) > 650 * S;
        if (!far) continue;
        r->lx[r->lake] = c[i].x; r->ly[r->lake] = c[i].y;
        name_make(hash2(rx * 4 + r->lake, ry, 1100 + opt_seed * 5), 1, r->lname[r->lake], sizeof r->lname[0]);
        r->lake++;
    }
    return r;
}
/* a district's name, if it has one, and where it goes: per 4x4 cells */
static int district(int rx, int ry, double *x, double *y, char *buf, int n) {
    uint32_t h = hash2(rx, ry, 1200 + opt_seed * 5);
    if ((h & 7) > 3) return 0;
    *x = (rx + 0.25 + 0.5 * ((h >> 3) & 0xfff) / 4096.0) * 4 * CELL;
    *y = (ry + 0.25 + 0.5 * (h >> 20) / 4096.0) * 4 * CELL;
    float g[5];
    terrain((float)(*x * MPP) + seed_x, (float)(*y * MPP) + seed_z, g);
    int kind = g[0] > 360 ? 4 : g[2] > 0.6f && g[0] > 20 ? 6 : g[0] > 30 && g[0] < 220 ? 5 : -1;
    if (kind < 0) return 0;
    name_make(hash2(rx, ry, 1250 + opt_seed * 5), kind, buf, n);
    return 1;
}

/* AA strokes into the mask, uv = (px across, half width), colour = channels */
static void mseg(float x0, float y0, float x1, float y1, float hw, const float ch[4]) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1e-3f) return;
    dx /= l; dy /= l;
    float e = hw * 0.7f + 0.5f, s = hw + 1;
    float q[8] = {x0 - dx * e - dy * s, y0 - dy * e + dx * s, x1 + dx * e - dy * s, y1 + dy * e + dx * s,
                  x1 + dx * e + dy * s, y1 + dy * e - dx * s, x0 - dx * e + dy * s, y0 - dy * e - dx * s};
    float t[8] = {s, hw, s, hw, -s, hw, -s, hw};
    quad(&mk, q, t, ch, 3);
}
/* a polyline at screen offset (ox, oy): dashes on/off px (on = 0: solid) */
static void mline(const Pt *p, int n, double ox, double oy, float hw, float on, float off, const float ch[4]) {
    float ph_ = 0;
    for (int i = 0; i + 1 < n; i++) {
        float x0 = (float)(p[i].x - ox), y0 = (float)(p[i].y - oy), x1 = (float)(p[i + 1].x - ox), y1 = (float)(p[i + 1].y - oy);
        float l = hypotf(x1 - x0, y1 - y0);
        if (fmaxf(x0, x1) < mx0 - 20 || fminf(x0, x1) > mx1 + 20 || fmaxf(y0, y1) < my0 - 20 || fminf(y0, y1) > my1 + 20) {
            ph_ = on > 0 ? fmodf(ph_ + l, on + off) : 0;
            continue;
        }
        if (on <= 0) { mseg(x0, y0, x1, y1, hw, ch); continue; }
        float a = 0;
        while (a < l) {
            float r = fmaxf(ph_ < on ? on - ph_ : on + off - ph_, 0.05f), b = fminf(l, a + r);
            if (b <= a) break;
            if (ph_ < on && b - a > 0.3f)
                mseg(x0 + (x1 - x0) * a / l, y0 + (y1 - y0) * a / l, x0 + (x1 - x0) * b / l, y0 + (y1 - y0) * b / l, hw, ch);
            ph_ = fmodf(ph_ + (b - a), on + off);
            a = b;
        }
    }
}

/* the streams, ways and settlement blocks round the view */
static void build_carto(double ox, double oy) {
    int M = 4;
    int cx0 = (int)floor((ox + mx0) / CELL) - M, cx1 = (int)floor((ox + mx1) / CELL) + M;
    int cy0 = (int)floor((oy + my0) / CELL) - M, cy1 = (int)floor((oy + my1) / CELL) + M;
    /* build: everything offscreen; live, nearest the middle first, ~4 ms */
    int missing = 0;
    for (int cy = cy0; cy <= cy1 && !missing; cy++)
        for (int cx = cx0; cx <= cx1 && !missing; cx++) missing = !carto_get(cx, cy, 0);
    double t0 = now(), budget = opt_shot || opt_bench ? 1e9 : 0.004;
    double mx = (ox + W * 0.5) / CELL, my = (oy + H * 0.5) / CELL;
    int rings = (int)fmax(fmax(mx - cx0, cx1 + 1 - mx), fmax(my - cy0, cy1 + 1 - my)) + 1;
    for (int ring = 0; missing && ring <= rings && now() - t0 < budget; ring++)
        for (int cy = cy0; cy <= cy1; cy++)
            for (int cx = cx0; cx <= cx1; cx++) {
                int d = (int)fmax(fabs(cx + 0.5 - mx), fabs(cy + 0.5 - my));
                if (d == ring && !carto_get(cx, cy, 0) && now() - t0 < budget) carto_get(cx, cy, 1);
            }
    static const float CH_ST[4] = {0, 1, 0, 0}, CH_RD[4] = {0, 0, 1, 0}, CH_PT[4] = {0, 0, 0, 1};
    /* the becks whose valleys may reach the view (the warp moves things
     * ~1.1 km at most), fading in from their heads */
    float q0x = (float)((ox + mx0) * MPP) + seed_x - 1200, q1x = (float)((ox + mx1) * MPP) + seed_x + 1200;
    float q0y = (float)((oy + my0) * MPP) + seed_z - 1200, q1y = (float)((oy + my1) * MPP) + seed_z + 1200;
    for (int k = 0; k < 3; k++)
        for (int j = (int)floorf(q0y / LS[k]) - 2; j <= (int)floorf(q1y / LS[k]) + 2; j++)
            for (int i = (int)floorf(q0x / LS[k]) - 2; i <= (int)floorf(q1x / LS[k]) + 2; i++) {
                Beck *bk = beck_get(k, i, j, 0, 0);
                if (!bk && now() - t0 < budget) bk = beck_get(k, i, j, 1, budget > 1 ? 0 : 100);
                if (!bk) continue;
                for (int m = 0; m + 1 < bk->n; m++) {
                    const Pt *a = &bk->p[m], *b = &bk->p[m + 1];
                    if (a->w == 0 || b->w == 0) continue;
                    float x0 = (float)(a->x - ox), y0 = (float)(a->y - oy), x1 = (float)(b->x - ox), y1 = (float)(b->y - oy);
                    if (fmaxf(x0, x1) < mx0 - 8 || fminf(x0, x1) > mx1 + 8 || fmaxf(y0, y1) < my0 - 8 || fminf(y0, y1) > my1 + 8) continue;
                    float ch[4] = {0, a->w < 0 ? 0.5f : 1, 0, 0};
                    mseg(x0, y0, x1, y1, fabsf(b->w), a->w < 0 ? ch : CH_ST);
                }
            }
    static const float CH_KN[4] = {0, 0, 0.25f, 0}, CH_RD2[4] = {0, 0, 0.5f, 0};
    for (int cy = cy0; cy <= cy1; cy++)
        for (int cx = cx0; cx <= cx1; cx++) {
            Carto *c = carto_get(cx, cy, 0);
            if (!c) continue;
            for (int i = 0; i < c->nw; i++) {
                const Way *w = &c->w[i];
                switch (w->kind) {
                case W_MAIN:
                    mline(w->p, w->n, ox, oy, 2.9f * S, 0, 0, CH_KN);
                    mline(w->p, w->n, ox, oy, 2.1f * S, 0, 0, CH_RD2);
                    mline(w->p, w->n, ox, oy, 1.3f * S, 0, 0, CH_RD);
                    break;
                case W_ROAD:
                    mline(w->p, w->n, ox, oy, 2.2f * S, 0, 0, CH_KN);
                    mline(w->p, w->n, ox, oy, 1.5f * S, 0, 0, CH_RD2);
                    mline(w->p, w->n, ox, oy, 0.75f * S, 0, 0, CH_RD);
                    break;
                case W_TRACK:
                    mline(w->p, w->n, ox, oy, 0.65f * S, 7 * S, 3.5f * S, CH_PT);
                    break;
                default:
                    mline(w->p, w->n, ox, oy, 0.6f * S, 2.6f * S, 2.6f * S, CH_PT);
                }
            }
        }
    /* settlements: a few blocks round the centre, more for a bigger place */
    float bc[4];
    rgba(bc, P.spot, 0.9f);
    for (int cy = cy0 + M - 1; cy <= cy1 - M + 1; cy++)
        for (int cx = cx0 + M - 1; cx <= cx1 - M + 1; cx++) {
            Settle *s = settle_get(cx, cy);
            if (!s->has) continue;
            int nb = s->size == 2 ? 11 : s->size == 1 ? 6 : 3;
            float rad = (s->size == 2 ? 13 : s->size == 1 ? 8 : 5) * S;
            for (int k = 0; k < nb; k++) {
                uint32_t q = hash2(cx * 31 + k, cy, 950 + opt_seed);
                float a = (q & 0xff) / 256.0f * 6.2832f, r = rad * sqrtf(((q >> 8) & 0xff) / 256.0f);
                float bx = (float)(s->x - ox) + r * cosf(a), by = (float)(s->y - oy) + r * sinf(a);
                float hw = (1.6f + 1.4f * ((q >> 16) & 0xff) / 256.0f) * S, hh = hw * 0.65f;
                float an = (q >> 24) / 256.0f * 1.5708f, co = cosf(an), si = sinf(an);
                float lx[4] = {-hw, hw, hw, -hw}, ly[4] = {-hh, -hh, hh, hh}, qq[8], tt[8] = {0};
                for (int j = 0; j < 4; j++) { qq[2 * j] = bx + lx[j] * co - ly[j] * si; qq[2 * j + 1] = by + lx[j] * si + ly[j] * co; }
                quad(&ov, qq, tt, bc, 1);
                knock(bx, by, an, hw, hh);
            }
        }
}

/* room for a box (centre, angle, w x h) among the placed discs, bar any
 * centred within claim_skip.r of claim_skip (a name's own spot height); take
 * it if so */
static Disc claim_skip;
static int claim(Disc *pl, int *np, int cap, float x, float y, float ang, float w, float h) {
    int n = (int)ceilf(w / h) + 1;
    if (*np + n > cap) return 0;
    float co = cosf(ang), si = sinf(ang), r = h * 0.75f;
    for (int i = 0; i < n; i++) {
        float t = n > 1 ? (i / (float)(n - 1) - 0.5f) * (w - h) : 0, dx = x + t * co, dy = y + t * si;
        if (dx < mx0 + 12 * S || dx > mx1 - 12 * S || dy < my0 + 12 * S || dy > my1 - 12 * S) return 0;
        for (int k = 0; k < *np; k++)
            if (hypotf(dx - pl[k].x, dy - pl[k].y) < r + pl[k].r + 3 * S &&
                hypotf(pl[k].x - claim_skip.x, pl[k].y - claim_skip.y) >= claim_skip.r) return 0;
    }
    for (int i = 0; i < n; i++) {
        float t = n > 1 ? (i / (float)(n - 1) - 0.5f) * (w - h) : 0;
        pl[(*np)++] = (Disc){x + t * co, y + t * si, r};
    }
    return 1;
}

/* place names, after the spot heights and before the contour labels */
static void place_names(double ox, double oy, Disc *pl, int *np, int cap) {
    float sc[4], wc[4], gc[4], hc[4];
    rgba(sc, P.spot, 0.95f); rgba(wc, P.waterink, 0.95f); rgba(gc, P.frame, 0.9f); rgba(hc, P.spot, 0.75f);
    char buf[48];
    /* settlements biggest first, then lakes, rivers, hills; districts last */
    int cx0 = (int)floor((ox + mx0) / CELL), cx1 = (int)floor((ox + mx1) / CELL);
    int cy0 = (int)floor((oy + my0) / CELL), cy1 = (int)floor((oy + my1) / CELL);
    for (int pass = 2; pass >= 0; pass--)
        for (int cy = cy0; cy <= cy1; cy++)
            for (int cx = cx0; cx <= cx1; cx++) {
                Settle *s = settle_get(cx, cy);
                if (!s->has || s->size != pass) continue;
                name_make(hash2(cx, cy, 990 + opt_seed * 3), 0, buf, sizeof buf);
                const Face *f = &fonts[pass == 2 ? F_TOWN : pass == 1 ? F_PLACE : F_SMALL];
                float tw = text_w(f, buf, 0), th = cap_h(f), rad = (pass == 2 ? 15 : pass == 1 ? 10 : 7) * S;
                float x = (float)(s->x - ox), y = (float)(s->y - oy);
                /* right, left, above, below */
                float ax[4] = {x + rad + 4 * S + tw * 0.5f, x - rad - 4 * S - tw * 0.5f, x, x};
                float ay[4] = {y, y, y - rad - 4 * S - th * 0.5f, y + rad + 4 * S + th * 0.5f};
                for (int k = 0; k < 4; k++)
                    if (claim(pl, np, cap, ax[k], ay[k], 0, tw + 4 * S, th + 6 * S)) {
                        text(&ov, f, buf, ax[k], ay[k], 0, 0.5f, th * 0.5f, 0, sc);
                        knock(ax[k], ay[k], 0, tw * 0.5f + 2 * S, th * 0.5f + 2.5f * S);
                        break;
                    }
            }
    int rx0 = (int)floor(((ox + mx0) / CELL - 0) / RG), rx1 = (int)floor((ox + mx1) / CELL / RG);
    int ry0 = (int)floor((oy + my0) / CELL / RG), ry1 = (int)floor((oy + my1) / CELL / RG);
    for (int ry = ry0; ry <= ry1; ry++)
        for (int rx = rx0; rx <= rx1; rx++) {
            Tract *r = region_get(rx, ry);
            for (int i = 0; i < r->lake; i++) {
                const Face *f = &fonts[F_WATER];
                float tr = roundf(2 * S), tw = text_w(f, r->lname[i], tr), th = cap_h(f);
                float x = (float)(r->lx[i] - ox), y = (float)(r->ly[i] - oy);
                if (claim(pl, np, cap, x, y, 0, tw + 6 * S, th + 6 * S))
                    text(&ov, f, r->lname[i], x, y, 0, 0.5f, th * 0.5f, tr, wc);
            }
        }
    /* rivers: a point on each river in view, named by its level set */
    for (int cy = cy0; cy <= cy1; cy++)
        for (int cx = cx0; cx <= cx1; cx++) {
            Carto *cc = carto_get(cx, cy, 0);
            if (!cc || !cc->river) continue;
            double x = cc->rx, y = cc->ry;
            float a = cc->ra;
            snprintf(buf, sizeof buf, "%s", cc->rname);
            const Face *f = &fonts[F_WATER];
            float tr = roundf(1.5f * S), tw = text_w(f, buf, tr), th = cap_h(f);
            /* beside the water, not on it */
            float off = th * 0.5f + 8 * S, nx = -sinf(a), ny = cosf(a);
            float px = (float)(x - ox) - nx * off, py = (float)(y - oy) - ny * off;
            if (claim(pl, np, cap, px, py, a, tw + 6 * S, th + 5 * S)) {
                text(&ov, f, buf, px, py, a, 0.5f, th * 0.5f, tr, wc);
                knock(px, py, a, tw * 0.5f + 2 * S, th * 0.5f + 2 * S);
            }
        }
    /* hills: the higher spot heights, named in small italic beneath */
    for (int cy = (int)floor((oy + my0) / CELL); cy <= (int)floor((oy + my1) / CELL); cy++)
        for (int cx = (int)floor((ox + mx0) / CELL); cx <= (int)floor((ox + mx1) / CELL); cx++) {
            Cell *c = cell_get(cx, cy, 0);
            if (!c || !c->spot || c->sv < 380 || (hash2(cx, cy, 1500 + opt_seed) & 3) == 0) continue;
            name_make(hash2(cx, cy, 1600 + opt_seed * 5), 3, buf, sizeof buf);
            const Face *f = &fonts[F_HILL];
            float tw = text_w(f, buf, 0), th = cap_h(f);
            float x = (float)(c->sx - ox), y = (float)(c->sy - oy) + 8 * S + th * 0.5f + 3 * S;
            claim_skip = (Disc){x + 14 * S, (float)(c->sy - oy), 20 * S};
            int ok = claim(pl, np, cap, x + tw * 0.5f - 4 * S, y, 0, tw + 4 * S, th + 4 * S);
            claim_skip.r = 0;
            if (ok) {
                text(&ov, f, buf, x - 4 * S, y, 0, 0, th * 0.5f, 0, hc);
                knock(x + tw * 0.5f - 4 * S, y, 0, tw * 0.5f + 2 * S, th * 0.5f + 2 * S);
            }
        }
    /* regions: spaced capitals, overprinted lightly */
    for (int ry = (int)floor((oy + my0) / CELL / 4); ry <= (int)floor((oy + my1) / CELL / 4); ry++)
        for (int rx = (int)floor((ox + mx0) / CELL / 4); rx <= (int)floor((ox + mx1) / CELL / 4); rx++) {
            double gx, gy;
            if (!district(rx, ry, &gx, &gy, buf, sizeof buf)) continue;
            const Face *f = &fonts[F_REGION];
            float tr = roundf(f->px * 0.55f), tw = text_w(f, buf, tr), th = cap_h(f);
            float x = (float)(gx - ox), y = (float)(gy - oy);
            if (claim(pl, np, cap, x, y, 0, tw + 8 * S, th + 8 * S)) {
                text(&ov, f, buf, x, y, 0, 0.5f, th * 0.5f, tr, gc);
                knock(x, y, 0, tw * 0.5f + 3 * S, th * 0.5f + 2 * S);
            }
        }
}

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
    Disc placed[2 * MAXF];
    int np = 0, named = 0;
    for (int i = 0; i <= nf; i++) {
        if (!named && (i == nf || f[i].kind == 1)) { place_names(ox, oy, placed, &np, MAXF); named = 1; }
        if (i == nf || np >= 2 * MAXF) break;
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
    float l = fmaxf(1, roundf(S));     /* a hairline, scaled with the type */
    rect(&ov, x0, y0, x1, y0 + l, fc); rect(&ov, x0, y1 - l, x1, y1, fc);
    rect(&ov, x0, y0, x0 + l, y1, fc); rect(&ov, x1 - l, y0, x1, y1, fc);
    float in = roundf(4 * S);
    rect(&ov, x0 + in, y0 + in, x1 - in, y0 + in + l, fc2); rect(&ov, x0 + in, y1 - in - l, x1 - in, y1 - in, fc2);
    rect(&ov, x0 + in, y0 + in, x0 + in + l, y1 - in, fc2); rect(&ov, x1 - in - l, y0 + in, x1 - in, y1 - in, fc2);

    /* north arrow, with an N over it */
    float ax = roundf(x0 + pad + col * 0.5f);
    const Glyph *ga = &sprites[SP_ARROW];
    float ay = roundf(y0 + ph_ * 0.5f - ga->h * 0.5f + 9 * S);
    text(&ov, ft, "N", ax, ay - 6 * S, 0, 0.5f, 0, 0, tc);
    blit(&ov, ga, ax, ay, 1, 0, -floorf(ga->w * 0.5f), 0, tc);
    float dx = roundf(x0 + pad + col + pad * 0.5f);
    rect(&ov, dx, y0 + pad, dx + l, y1 - pad, fc2);

    float tx = x0 + pad + col + pad;
    float by = y0 + pad + cap_h(ft);
    text(&ov, ft, title, tx, by, 0, 0, 0, track, tc);
    float ry = roundf(by + 10 * S);
    rect(&ov, tx, ry, tx + inner, ry + l, fc2);
    float sy = ry + roundf(10 * S) + cap_h(fs);
    text(&ov, fs, sub, tx, sy, 0, 0, 0, 0, sc);

    /* scale bar: alternate halves of each kilometre filled */
    float bt = roundf(sy + 13 * S), bh = roundf(5 * S);
    float seg = bar / (km * 2);
    rect(&ov, tx, bt, tx + bar, bt + l, fc); rect(&ov, tx, bt + bh - l, tx + bar, bt + bh, fc);
    for (int i = 0; i < km * 2; i++) {
        float a = roundf(tx + i * seg), b = roundf(tx + (i + 1) * seg);
        if (!(i & 1)) rect(&ov, a, bt, b, bt + bh, bc);
        rect(&ov, a, bt, a + l, bt + bh, fc);
    }
    rect(&ov, roundf(tx + bar) - l, bt, roundf(tx + bar), bt + bh, fc);
    char buf[16];
    float ly = bt + bh + roundf(5 * S) + cap_h(fm);
    for (int i = 0; i <= km; i++) {
        snprintf(buf, sizeof buf, "%d", i);
        text(&ov, fm, buf, roundf(tx + i * bar / km), ly, 0, 0.5f, 0, 0, sc);
    }
    text(&ov, fm, "km", roundf(tx + bar + 8 * S), ly, 0, 0, 0, 0, sc);
}

/* ---- the walker: a surveyor's pen crossing the country, leaving a dotted
 * track ----
 * It heads for a point that wanders about the middle of the view (so it
 * stays in sight as the view drifts), but like a path it would rather keep
 * to the contour than climb: the uphill part of its heading is mostly
 * turned away, so it zigzags up slopes and swings round hills. It turns
 * smoothly, and a dot is set down every few pixels it walks; dots are world
 * points, so the track drifts with the map, and fade over a few minutes. */
#define WK_N 2048
#define WK_LIFE 300.0                  /* s a dot of the track lasts */
static struct { double x, y, t; } wk[WK_N];
static int wk_n, wk_i;                 /* dots held, the next slot */
static double wk_t = -1e18, wk_x, wk_y, wk_hd, wk_s;    /* time, place, heading, px since the last dot */

static void walker_goal(double t, double *x, double *y) {
    double cx, cy;
    view_at(t, &cx, &cy);
    *x = cx + W * (0.27 * sin(t / 157 + ph2[5]) + 0.07 * sin(t / 61 + ph2[6]));
    *y = cy + H * (0.24 * sin(t / 119 + ph2[7]) + 0.07 * cos(t / 53 + ph2[5]));
}

static void walker_step(double t, double dt) {
    double gx, gy, ex, ey;
    walker_goal(t, &gx, &gy);
    ex = gx - wk_x; ey = gy - wk_y;
    double dist = hypot(ex, ey) + 1e-9, dx = ex / dist, dy = ey / dist;
    /* keep to the contour: take most of the climb out of the heading,
     * unless it has fallen well behind */
    double hx, hy;
    grad(wk_x, wk_y, &hx, &hy);
    double hl = hypot(hx, hy);
    if (hl > 1e-6) {
        double c = (dx * hx + dy * hy) / hl, k = 0.85 * smoothstep(0.02f, 0.12f, (float)(hl / MPP)) *
                               (1 - smoothstep(80 * S, 300 * S, (float)dist));
        dx -= k * c * hx / hl; dy -= k * c * hy / hl;
    }
    /* turn smoothly towards that */
    double want = atan2(dy, dx), d = remainder(want - wk_hd, 2 * M_PI), turn = 0.35 * dt;
    wk_hd += d > turn ? turn : d < -turn ? -turn : d;
    double v = 3.5 * S + 0.07 * dist;                         /* px/s: hurrying if left behind */
    double sx = cos(wk_hd) * v * dt, sy = sin(wk_hd) * v * dt;
    wk_x += sx; wk_y += sy;
    wk_s += hypot(sx, sy);
    if (wk_s >= 6.5 * S) {
        wk_s = 0;
        wk[wk_i].x = wk_x; wk[wk_i].y = wk_y; wk[wk_i].t = t + dt;
        wk_i = (wk_i + 1) % WK_N;
        if (wk_n < WK_N) wk_n++;
    }
}

/* walk on to time t (from a few minutes before it, the first time or after
 * a jump), then draw the track and the pen */
static void build_walker(double t, double ox, double oy) {
    if (t < wk_t || t - wk_t > WK_LIFE) {
        wk_t = t - WK_LIFE;
        walker_goal(wk_t, &wk_x, &wk_y);
        double gx, gy;
        walker_goal(wk_t + 1, &gx, &gy);
        wk_hd = atan2(gy - wk_y, gx - wk_x);
        wk_n = wk_i = 0; wk_s = 0;
    }
    while (wk_t < t) {
        double dt = fmin(0.25, t - wk_t);
        walker_step(wk_t, dt);
        wk_t += dt;
    }
    float c[4];
    for (int k = 0; k < wk_n; k++) {
        int i = ((wk_i - 1 - k) % WK_N + WK_N) % WK_N;
        double age = t - wk[i].t;
        if (age > WK_LIFE) { wk_n = k; break; }
        float x = (float)(wk[i].x - ox), y = (float)(wk[i].y - oy);
        if (x < mx0 - 8 || x > mx1 + 8 || y < my0 - 8 || y > my1 + 8) continue;
        float a = 0.85f * (1 - smoothstep(0.35f * WK_LIFE, WK_LIFE, (float)age)) * smoothstep(0, 1.5f, (float)age);
        rgba(c, P.label, a);
        blit(&ov, &wk_dot, x, y, 1, 0, -wk_dot.w * 0.5f, -wk_dot.h * 0.5f, c);
    }
    float x = (float)(wk_x - ox), y = (float)(wk_y - oy);
    rgba(c, P.label, 0.75f + 0.2f * (float)sin(t * 1.3));
    blit(&ov, &wk_pen, x, y, 1, 0, -wk_pen.w * 0.5f, -wk_pen.h * 0.5f, c);
}

/* ---- a frame ---- */
static void draw_quads(int from, int to) {
    if (to <= from) return;
    glUseProgram(prog_quad);
    glUniform2f(U(prog_quad, "uRes"), (float)W, (float)H);
    glBindVertexArray(vao_q);
    glDrawArrays(GL_TRIANGLES, from / 9, (to - from) / 9);
}

/* the light pass: sun and cloud shadows into tex_shade, at 1 / SH_DIV resolution */
static void shade(double t, double ox, double oy, float az, float el) {
    GLuint p = prog_shade;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shade);
    glViewport(0, 0, (W + SH_DIV - 1) / SH_DIV, (H + SH_DIV - 1) / SH_DIV);
    glUseProgram(p);
    glBindTexture(GL_TEXTURE_2D, tex_h);
    glUniform1i(U(p, "uH"), 0);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform2f(U(p, "uN"), (float)TN, (float)TM);
    glUniform2f(U(p, "uOrgT"), (float)(ox - floor(ox / TN) * TN), (float)(oy - floor(oy / TM) * TM));
    glUniform2f(U(p, "uSunDir"), cosf(az), sinf(az));
    glUniform1f(U(p, "uMpp"), MPP);
    glUniform1f(U(p, "uWater"), water_at(t));
    glUniform1f(U(p, "uRise"), MPP * tanf(el) / 2.0f);          /* relief exaggerated twice, as the hillshade */
    glUniform1f(U(p, "uReach"), SH_REACH * S);
    glUniform1f(U(p, "uDiv"), SH_DIV);
    /* clouds: two layers on steady winds (per seed), over the ground */
    double gx = ox * MPP + seed_x, gy = oy * MPP + seed_z, wa = ph2[3];
    glUniform2f(U(p, "uC1"), (float)(gx - 9 * cos(wa) * t), (float)(gy - 9 * sin(wa) * t));
    glUniform2f(U(p, "uC2"), (float)(gx - 14 * cos(wa + 0.5) * t + 7000), (float)(gy - 14 * sin(wa + 0.5) * t - 3000));
    glUniform1f(U(p, "uCover"), (float)(0.6 + 0.08 * sin(t / 700 + ph2[4])));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* t: drift time, seconds */
static void render(double t, GLuint target) {
    targets();
    float m = roundf(36 * S);
    mx0 = (int)m; my0 = (int)m; mx1 = W - (int)m; my1 = H - (int)m;

    double cx, cy;
    view_at(t, &cx, &cy);
    double ox = cx - W * 0.5, oy = cy - H * 0.5;        /* world px of the screen's top-left */
    bake((int)floor(ox) - BM, (int)floor(oy) - BM);
    float sun_az, sun_el;
    sun_at(t, &sun_az, &sun_el);
    /* the light changes slowly and is soft: a frame late is a fraction of a
     * pixel, so it's redone every other frame */
    if (!shade_ok || ++shade_ok % 2) shade(t, ox, oy, sun_az, sun_el);
    if (!shade_ok) shade_ok = 1;

    /* the overlay and the knock-outs */
    ov.n = mk.n = 0;
    build_walker(t, ox, oy);
    build_carto(ox, oy);
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
    glClearColor(0, 0, 0, 0);
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
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_shade);
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
    glUniform1f(U(p, "uSun"), sun_az);
    glUniform1f(U(p, "uT"), (float)fmod(t, 13.0 * 2000));
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
        "usage: contours [--window | --root | --layer] [--seed N] [--interval METRES]\n"
        "                [--palette brass|cream|slate|blueprint|survey | --light] [--speed X] [--fps N]\n"
        "                [--shot FILE.ppm [--size WxH] [--at SECONDS]] [--bench N [--size WxH]]\n"
        "                [--render-scale F]\n"
        "  --render-scale F  with --layer, draw F x the output's native pixels (0 < F <= 1; default 1)\n");
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
        else if (!strcmp(a, "-light")) pal = 1;
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            pal = -2;
            for (int k = 0; k < NPAL; k++) if (!strcmp(n, pal_names[k])) pal = k;
            if (pal == -2) usage();
        }
        else if (!strcmp(a, "-interval") && i + 1 < argc) opt_interval = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
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
    if (pal < 0) pal = (int)(hash2((int32_t)opt_seed, 5, 3) % NPAL);
    for (int i = 0; i < 8; i++) ph2[i] = (hash2((int32_t)opt_seed, i, 9) & 0xffff) / 65536.0 * 2 * M_PI;
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
    carto_palette();
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
