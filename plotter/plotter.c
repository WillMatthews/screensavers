/*
 * plotter — a pen plotter slowly drawing generative line art on cream paper.
 *
 * Seen from above: a sheet of paper on the plotter's bed, a gantry beam riding
 * on two rails, and a carriage on the beam holding the pen. The pen lowers,
 * draws one stroke at a time with believable acceleration (slowing into
 * curves and corners), lifts, travels, and so on until the piece is done;
 * then it signs the sheet in the corner with the host and the time in its own
 * single-stroke script, parks, and the sheet slides away for a fresh one.
 *
 * Each sheet is one of a handful of classic pen-plotter pieces, picked at
 * random: a flow field of evenly spaced streamlines, a topographic map with
 * hatched hillshading and coastal ripple lines, a ridgeline landscape with
 * hidden lines removed (and a hatched sun behind it), a harmonograph, a
 * spirograph rosette, warped concentric rings, and a multi-strand Truchet
 * weave. Mostly one ink, with an accent pen swapped in for part of it: the
 * carriage parks for each pen change.
 *
 * Pen sets: each run picks one from the seed (--palette NAME to choose),
 * with paper, mat and machine to match. sepia: sepia ink with brass gold,
 * terracotta or sage on cream, a brown machine with brass. blue: blue-black
 * fountain-pen ink with royal blue or red on white, an aluminium machine on a
 * slate mat. technical: black technical pen with red or fine grey on bright
 * white, a black machine. colour: felt tips (teal, magenta, orange,
 * ultramarine; a fresh main colour each sheet) on a green cutting mat.
 *
 * Ink: each stroke is resampled and given a motion profile, so every point
 * of the pen's path has a time. Ink is laid down as antialiased capsules into
 * a persistent texture the size of the sheet (min-blended, so the joins of a
 * stroke don't darken), a little wider and darker where the pen moves slowly
 * (pooling where it lands and lifts), and with a slight wobble. The state at
 * any time is a pure function of seed + time: --at replays the plotter up to
 * that moment.
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), --shot
 * FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen and
 * --bench N to time N frames. --kind NAME draws only one kind of piece,
 * --speed X runs time faster, and --at also starts a live run that far in.
 * Rendering is cheap: the bed and the sheet's paper are baked once per size,
 * ink goes into its texture a few segments a frame, and the per-frame pass
 * is two texel fetches except near the gantry.
 *
 * Environment, for testing: PLOTTER_DEBUG=1 prints each sheet's plan,
 * PLOTTER_HOST=TEXT stands in for the host name in the signature.
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
static int opt_root, opt_layer, opt_bench, opt_kind = -1;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 120;            /* --shot / --bench: this many seconds in (live: start there, if given) */
static int opt_at_set;
static double opt_speed = 1;           /* time runs this much faster */
static int off_w = 1920, off_h = 1080;

#define TAU 6.28318531f

static void die(const char *what) { fprintf(stderr, "plotter: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n);
    if (!p) die("out of memory");
    return p;
}
#define GROW(arr, n, cap, need) do { if ((need) > (cap)) { (cap) = (cap) ? (cap) * 2 : 256; \
    (cap) = (cap) < (need) ? (need) : (cap); (arr) = xrealloc((arr), sizeof *(arr) * (size_t)(cap)); } } while (0)

/* ---- randomness: one generator per sheet, so sheets are reproducible ---- */
typedef struct { uint32_t s; } Rng;
static uint32_t mix32(uint32_t h) {
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
static float rnd(Rng *r) {
    r->s ^= r->s << 13; r->s ^= r->s >> 17; r->s ^= r->s << 5;
    return (r->s >> 8) * (1.0f / 16777216.0f);
}
static float rr(Rng *r, float a, float b) { return a + (b - a) * rnd(r); }
static int ri(Rng *r, int n) { int k = (int)(rnd(r) * n); return k < n ? k : n - 1; }

/* ---- value noise (as flightsim's) ---- */
static uint32_t hash2(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
static float rand01(int32_t x, int32_t z, uint32_t salt) { return (hash2(x, z, salt) & 0xffffff) / (float)0x1000000; }
static float smooth(float t) { return t * t * (3 - 2 * t); }
static float vnoise(float x, float z, uint32_t salt) {
    float fx = floorf(x), fz = floorf(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    float tx = smooth(x - fx), tz = smooth(z - fz);
    float a = rand01(ix, iz, salt), b = rand01(ix + 1, iz, salt);
    float c = rand01(ix, iz + 1, salt), d = rand01(ix + 1, iz + 1, salt);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}
static float fbm(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        s += vnoise(x, z, salt + i) * amp;
        norm += amp; amp *= 0.5f; x *= 2.03f; z *= 2.03f;
    }
    return s / norm;
}
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smoothstepf(float a, float b, float x) { return smooth(clampf((x - a) / (b - a), 0, 1)); }

/* ---- the sheet, in mm; y down ---- */
static float PW = 420, PH = 297;       /* A3, landscape (portrait on a tall screen) */

/* pens: the main ink and three accent slots, filled from the pen set */
enum { INK, ACC1, ACC2, ACC3, NPEN };

/* pen sets, one per run (from the seed, or --palette): the inks, the paper,
 * the bed's mat and its grid, and the machine (dark, mid and light faces,
 * and its fittings). A set with shuffle picks a fresh main ink per sheet
 * from all four, the accent from the other three. */
typedef struct {
    const char *name, *pen[NPEN], *paper, *bed, *bedline;
    float dark[3], mid[3], hi[3], metal[3];
    int shuffle;
} Pal;
static const Pal PALS[] = {
    /* sepia ink, brass gold / terracotta / sage, on cream; a brown machine with brass */
    {"sepia", {"#2A211B", "#C99529", "#A53C2C", "#5A7548"}, "#F4EFE3", "#D6CDB7", "#EAE3D2",
     {0.165f, 0.13f, 0.106f}, {0.25f, 0.205f, 0.17f}, {0.42f, 0.355f, 0.29f}, {0.79f, 0.585f, 0.16f}, 0},
    /* blue-black fountain-pen ink, a royal blue or a red, on white; an
     * aluminium machine with steel fittings on a slate mat */
    {"blue", {"#1E2740", "#2A5BC8", "#C23A30", "#2A5BC8"}, "#F3F3EF", "#26292D", "#33373C",
     {0.25f, 0.265f, 0.285f}, {0.44f, 0.46f, 0.485f}, {0.63f, 0.65f, 0.675f}, {0.74f, 0.76f, 0.79f}, 0},
    /* black technical pen, red (or a fine grey), on bright white; a black machine */
    {"technical", {"#131313", "#C9282A", "#C9282A", "#77787A"}, "#F8F8F6", "#18191B", "#26282B",
     {0.045f, 0.045f, 0.05f}, {0.10f, 0.10f, 0.11f}, {0.21f, 0.215f, 0.225f}, {0.55f, 0.57f, 0.60f}, 0},
    /* felt tips: teal, magenta, orange, ultramarine, on white; a graphite
     * machine on a green cutting mat */
    {"colour", {"#0B8378", "#C8246E", "#E8730E", "#2B3BB0"}, "#F6F5F1", "#1D3529", "#2C4D3B",
     {0.11f, 0.115f, 0.12f}, {0.20f, 0.21f, 0.22f}, {0.36f, 0.37f, 0.385f}, {0.70f, 0.71f, 0.73f}, 1},
};
#define NPAL ((int)(sizeof PALS / sizeof PALS[0]))
static int opt_pal = -1;
static const Pal *pal;
static int sheet_pen[NPEN];             /* this sheet's slots -> the set's pens */

/* ---- polylines: what a generator hands the planner ---- */
typedef struct {
    float *p; int np, pcap;            /* x, y pairs */
    int *st, *cnt; unsigned char *pen; float *w;
    int ns, scap;
} Paths;

static void paths_free(Paths *P) { free(P->p); free(P->st); free(P->cnt); free(P->pen); free(P->w); memset(P, 0, sizeof *P); }

static void path_begin(Paths *P, int pen, float w) {
    if (P->ns + 1 > P->scap) {
        P->scap = P->scap ? P->scap * 2 : 256;
        P->st = xrealloc(P->st, sizeof *P->st * P->scap);
        P->cnt = xrealloc(P->cnt, sizeof *P->cnt * P->scap);
        P->pen = xrealloc(P->pen, P->scap);
        P->w = xrealloc(P->w, sizeof *P->w * P->scap);
    }
    P->st[P->ns] = P->np; P->cnt[P->ns] = 0; P->pen[P->ns] = (unsigned char)pen; P->w[P->ns] = w;
}
static void path_pt(Paths *P, float x, float y) {
    GROW(P->p, P->pcap, P->pcap, 2 * (P->np + 1));
    P->p[2 * P->np] = x; P->p[2 * P->np + 1] = y;
    P->np++;
    P->cnt[P->ns]++;
}
/* finish the path; it's dropped if it has fewer than two points */
static void path_end(Paths *P) {
    if (P->cnt[P->ns] < 2) { P->np = P->st[P->ns]; return; }
    P->ns++;
}
static void path_line(Paths *P, int pen, float w, float x0, float y0, float x1, float y1) {
    path_begin(P, pen, w); path_pt(P, x0, y0); path_pt(P, x1, y1); path_end(P);
}
static void path_rect(Paths *P, int pen, float w, float x0, float y0, float x1, float y1) {
    path_begin(P, pen, w);
    path_pt(P, x0, y0); path_pt(P, x1, y0); path_pt(P, x1, y1); path_pt(P, x0, y1); path_pt(P, x0, y0);
    path_end(P);
}
static float paths_length(const Paths *P) {
    double L = 0;
    for (int i = 0; i < P->ns; i++)
        for (int k = 1; k < P->cnt[i]; k++) {
            const float *a = P->p + 2 * (P->st[i] + k - 1);
            L += hypotf(a[2] - a[0], a[3] - a[1]);
        }
    return (float)L;
}

/* join polylines whose ends meet (within eps, same pen) into long ones */
static void chain(const Paths *in, Paths *out, float eps) {
    int n = in->ns;
    if (!n) return;
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (int i = 0; i < in->np; i++) {
        float x = in->p[2 * i], y = in->p[2 * i + 1];
        minx = fminf(minx, x); maxx = fmaxf(maxx, x);
        miny = fminf(miny, y); maxy = fmaxf(maxy, y);
    }
    float cell = eps * 4 > 0.5f ? eps * 4 : 0.5f;
    int gw = (int)((maxx - minx) / cell) + 3, gh = (int)((maxy - miny) / cell) + 3;
    int *head = malloc(sizeof(int) * (size_t)gw * gh), *nxt = malloc(sizeof(int) * 2 * (size_t)n);
    char *used = calloc((size_t)n, 1);
    for (int i = 0; i < gw * gh; i++) head[i] = -1;
#define ENDPT(e) (in->p + 2 * (in->st[(e) >> 1] + (((e) & 1) ? in->cnt[(e) >> 1] - 1 : 0)))
#define CELL(x, y) (((int)(((y) - miny) / cell) + 1) * gw + (int)(((x) - minx) / cell) + 1)
    for (int e = 0; e < 2 * n; e++) {
        const float *q = ENDPT(e);
        int c = CELL(q[0], q[1]);
        nxt[e] = head[c]; head[c] = e;
    }
    /* find an unused end near (x, y) with this pen */
    int *seq = malloc(sizeof(int) * 2 * (size_t)n + 2);   /* deque of signed path ids: ~id = reversed */
    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) continue;
        used[i0] = 1;
        int lo = n, hi = n;                                   /* seq[lo..hi) */
        seq[hi++] = i0;
        for (int dir = 0; dir < 2; dir++) {
            for (;;) {
                /* the free end: tail of the last (dir 0) or head of the first (dir 1) */
                int id = dir ? seq[lo] : seq[hi - 1];
                int rev = id < 0, pid = rev ? ~id : id;
                int end = (dir == 0) ^ rev ? 1 : 0;
                const float *q = ENDPT(2 * pid + end);
                int found = -1;
                int cx = (int)((q[0] - minx) / cell) + 1, cy = (int)((q[1] - miny) / cell) + 1;
                for (int yy = cy - 1; yy <= cy + 1 && found < 0; yy++)
                    for (int xx = cx - 1; xx <= cx + 1 && found < 0; xx++) {
                        if (xx < 0 || yy < 0 || xx >= gw || yy >= gh) continue;
                        for (int e = head[yy * gw + xx]; e >= 0; e = nxt[e]) {
                            int pj = e >> 1;
                            if (used[pj] || in->pen[pj] != in->pen[pid]) continue;
                            const float *r2 = ENDPT(e);
                            if (fabsf(r2[0] - q[0]) < eps && fabsf(r2[1] - q[1]) < eps) { found = e; break; }
                        }
                    }
                if (found < 0) break;
                int pj = found >> 1;
                used[pj] = 1;
                /* going forward we enter pj at the matched end: reversed if that's its tail */
                if (dir == 0) seq[hi++] = (found & 1) ? ~pj : pj;
                else seq[--lo] = (found & 1) ? pj : ~pj;
            }
        }
        path_begin(out, in->pen[i0], in->w[i0]);
        for (int k = lo; k < hi; k++) {
            int id = seq[k], rev = id < 0, pid = rev ? ~id : id;
            int c = in->cnt[pid];
            for (int m = k > lo ? 1 : 0; m < c; m++) {
                const float *q = in->p + 2 * (in->st[pid] + (rev ? c - 1 - m : m));
                path_pt(out, q[0], q[1]);
            }
        }
        path_end(out);
    }
#undef ENDPT
#undef CELL
    free(seq); free(head); free(nxt); free(used);
}

/* ---- hatching: parallel lines across a box, kept where inside() says ---- */
typedef int (*Inside)(void *ctx, float x, float y);

static void hatch(Paths *P, int pen, float w, float ang, float gap, float x0, float y0, float x1, float y1,
                  Inside inside, void *ctx, float phase) {
    float c = cosf(ang), s = sinf(ang);
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, R = hypotf(x1 - x0, y1 - y0) / 2;
    int k = 0;
    for (float o = -R + fmodf(phase, gap) ; o <= R; o += gap, k++) {
        /* alternate directions, as a plotter would */
        int run = 0;
        float lx = 0, ly = 0;
        for (float t = -R; t <= R + 0.2f; t += 0.4f) {
            float tt = (k & 1) ? -t : t;
            float x = cx + c * tt - s * o, y = cy + s * tt + c * o;
            int in = x >= x0 && x <= x1 && y >= y0 && y <= y1 && inside(ctx, x, y);
            if (in && !run) { path_begin(P, pen, w); path_pt(P, x, y); run = 1; }
            else if (!in && run) { path_pt(P, lx, ly); if (P->cnt[P->ns] == 2 && hypotf(lx - P->p[2 * P->st[P->ns]], ly - P->p[2 * P->st[P->ns] + 1]) < 1.5f) P->cnt[P->ns] = 0; path_end(P); run = 0; }
            lx = x; ly = y;
        }
        if (run) { path_pt(P, lx, ly); path_end(P); }
    }
}

/* ---- the art box: centred a little high, clear of the caption ---- */
static float art_cx(void) { return PW * 0.5f; }
static float art_cy(void) { return PH * 0.47f; }

/* ===================================================================== */
/* generators: each fills P with ~8-10 m of line                         */
/* ===================================================================== */

#define NIB 0.5f                        /* fineliner, mm */
#define BUDGET 8500.0f                  /* mm of ink a sheet aims for */

/* -- flow field: evenly spaced streamlines (after Jobard & Lefer) -- */
typedef struct {
    int circ; float cx, cy, rad, hx, hy;
    float sc, turns, ox, oy; uint32_t salt;
    float cell; int gw, gh; int *head, *next, *line, *ord; float *pt; int npt, cap;
    char *dead;
} Flow;

static int flow_in(Flow *F, float x, float y) {
    if (F->circ) return (x - F->cx) * (x - F->cx) + (y - F->cy) * (y - F->cy) < F->rad * F->rad;
    return fabsf(x - F->cx) < F->hx && fabsf(y - F->cy) < F->hy;
}
static float flow_ang(Flow *F, float x, float y) {
    return fbm(x / F->sc + F->ox, y / F->sc + F->oy, 2, F->salt) * TAU * F->turns;
}
/* is there a point of another line (or of this one, far enough back) within d? */
static int flow_near(Flow *F, float x, float y, float d, int line, int ord) {
    int gx = (int)(x / F->cell), gy = (int)(y / F->cell), rc = (int)ceilf(d / F->cell);
    for (int yy = gy - rc; yy <= gy + rc; yy++)
        for (int xx = gx - rc; xx <= gx + rc; xx++) {
            if (xx < 0 || yy < 0 || xx >= F->gw || yy >= F->gh) continue;
            for (int i = F->head[yy * F->gw + xx]; i >= 0; i = F->next[i]) {
                if (F->dead[F->line[i]]) continue;
                if (F->line[i] == line && abs(F->ord[i] - ord) < (int)(d * 3 / 0.5f) + 4) continue;
                float dx = F->pt[2 * i] - x, dy = F->pt[2 * i + 1] - y;
                if (dx * dx + dy * dy < d * d) return 1;
            }
        }
    return 0;
}
static void flow_add(Flow *F, float x, float y, int line, int ord) {
    if (F->npt + 1 > F->cap) {
        F->cap = F->cap ? F->cap * 2 : 4096;
        F->pt = xrealloc(F->pt, sizeof(float) * 2 * F->cap);
        F->next = xrealloc(F->next, sizeof(int) * F->cap);
        F->line = xrealloc(F->line, sizeof(int) * F->cap);
        F->ord = xrealloc(F->ord, sizeof(int) * F->cap);
    }
    int i = F->npt++, gx = (int)(x / F->cell), gy = (int)(y / F->cell);
    F->pt[2 * i] = x; F->pt[2 * i + 1] = y; F->line[i] = line; F->ord[i] = ord;
    if (gx < 0 || gy < 0 || gx >= F->gw || gy >= F->gh) { F->next[i] = -1; return; }
    F->next[i] = F->head[gy * F->gw + gx]; F->head[gy * F->gw + gx] = i;
}

static void gen_flow(Rng *R, Paths *P, int accent) {
    Flow F = {0};
    F.circ = rnd(R) < 0.5f;
    F.cx = art_cx(); F.cy = art_cy();
    F.rad = 0.36f * fminf(PW, PH);
    F.hx = PW * 0.31f; F.hy = PH * 0.33f;
    if (PW < PH) { F.hx = PW * 0.36f; F.hy = PH * 0.28f; }
    float area = F.circ ? 3.14159f * F.rad * F.rad : 4 * F.hx * F.hy;
    float dsep = clampf(area / BUDGET * 0.92f, 2.6f, 5.0f), dtest = dsep * 0.55f, step = 0.5f;
    F.sc = rr(R, 85, 160); F.turns = rr(R, 0.7f, 1.3f);
    F.ox = rr(R, 0, 100); F.oy = rr(R, 0, 100); F.salt = (uint32_t)ri(R, 1 << 20);
    F.cell = dtest;
    F.gw = (int)(PW / F.cell) + 1; F.gh = (int)(PH / F.cell) + 1;
    F.head = malloc(sizeof(int) * (size_t)F.gw * F.gh);
    for (int i = 0; i < F.gw * F.gh; i++) F.head[i] = -1;
    int maxlines = 20000;
    F.dead = calloc((size_t)maxlines, 1);
    float maxlen = rr(R, 140, 280);
    uint32_t csalt = F.salt + 77;
    float cthresh = rr(R, 0.56f, 0.62f), csc = rr(R, 40, 80);

    /* seeds: a jittered grid in shuffled order */
    int sx = (int)(PW / dsep), sy = (int)(PH / dsep), ns = sx * sy;
    float *seeds = malloc(sizeof(float) * 2 * ns);
    for (int j = 0; j < sy; j++)
        for (int i = 0; i < sx; i++) {
            seeds[2 * (j * sx + i)] = (i + rnd(R)) * dsep;
            seeds[2 * (j * sx + i) + 1] = (j + rnd(R)) * dsep;
        }
    for (int i = ns - 1; i > 0; i--) {
        int j = ri(R, i + 1);
        float a = seeds[2 * i], b = seeds[2 * i + 1];
        seeds[2 * i] = seeds[2 * j]; seeds[2 * i + 1] = seeds[2 * j + 1];
        seeds[2 * j] = a; seeds[2 * j + 1] = b;
    }
    int maxn = (int)(maxlen / step) + 2;
    float *fw = malloc(sizeof(float) * 2 * maxn), *bw = malloc(sizeof(float) * 2 * maxn);
    int line = 0;
    for (int s = 0; s < ns && line < maxlines; s++) {
        float x = seeds[2 * s], y = seeds[2 * s + 1];
        if (!flow_in(&F, x, y) || flow_near(&F, x, y, dsep * 0.95f, -1, 0)) continue;
        int nf = 0, nb = 0;
        for (int dir = 0; dir < 2; dir++) {
            float px = x, py = y, *buf = dir ? bw : fw;
            int *n = dir ? &nb : &nf;
            float sg = dir ? -1.0f : 1.0f;
            if (!dir) { buf[0] = x; buf[1] = y; *n = 1; flow_add(&F, x, y, line, 0); }
            while ((nf + nb) * step < maxlen && *n < maxn) {
                float a = flow_ang(&F, px, py);
                float mx = px + sg * cosf(a) * step * 0.5f, my = py + sg * sinf(a) * step * 0.5f;
                float a2 = flow_ang(&F, mx, my);
                float qx = px + sg * cosf(a2) * step, qy = py + sg * sinf(a2) * step;
                int ord = dir ? -(*n + 1) : *n;
                if (!flow_in(&F, qx, qy) || flow_near(&F, qx, qy, dtest, line, ord)) break;
                buf[2 * *n] = qx; buf[2 * *n + 1] = qy; (*n)++;
                flow_add(&F, qx, qy, line, ord);
                px = qx; py = qy;
            }
        }
        if ((nf + nb) * step < 6) { F.dead[line++] = 1; continue; }
        int pen = INK;
        if (accent && fbm(x / csc + 9.1f, y / csc + 3.7f, 2, csalt) > cthresh) pen = accent;
        path_begin(P, pen, NIB);
        for (int i = nb - 1; i >= 0; i--) path_pt(P, bw[2 * i], bw[2 * i + 1]);
        for (int i = 0; i < nf; i++) path_pt(P, fw[2 * i], fw[2 * i + 1]);
        path_end(P);
        line++;
    }
    free(seeds); free(fw); free(bw);
    free(F.head); free(F.next); free(F.line); free(F.ord); free(F.pt); free(F.dead);
}

/* -- topographic map: contours, coastal ripples, hatched hillshade -- */
typedef struct { float x0, y0, step; int nx, ny; float *h, *shade; float wl; } Topo;

static float topo_at(const Topo *T, const float *f, float x, float y) {
    float gx = (x - T->x0) / T->step, gy = (y - T->y0) / T->step;
    int ix = (int)gx, iy = (int)gy;
    ix = ix < 0 ? 0 : ix > T->nx - 2 ? T->nx - 2 : ix;
    iy = iy < 0 ? 0 : iy > T->ny - 2 ? T->ny - 2 : iy;
    float tx = clampf(gx - ix, 0, 1), ty = clampf(gy - iy, 0, 1);
    const float *r0 = f + iy * T->nx + ix, *r1 = r0 + T->nx;
    return (r0[0] * (1 - tx) + r0[1] * tx) * (1 - ty) + (r1[0] * (1 - tx) + r1[1] * tx) * ty;
}
typedef struct { Topo *T; float lo, hi; } ShadeBand;
static int shade_in(void *c, float x, float y) {
    ShadeBand *b = c;
    if (topo_at(b->T, b->T->h, x, y) < b->T->wl + 0.01f) return 0;
    float s = topo_at(b->T, b->T->shade, x, y);
    return s >= b->lo && s < b->hi;
}

/* marching squares at level L into 2-point segments */
static void contour(const Topo *T, float L, Paths *seg, int pen, float w) {
    for (int j = 0; j < T->ny - 1; j++)
        for (int i = 0; i < T->nx - 1; i++) {
            const float *r0 = T->h + j * T->nx + i, *r1 = r0 + T->nx;
            float v[4] = {r0[0], r0[1], r1[1], r1[0]};                 /* tl tr br bl */
            int c = (v[0] > L) | (v[1] > L) << 1 | (v[2] > L) << 2 | (v[3] > L) << 3;
            if (c == 0 || c == 15) continue;
            float x0 = T->x0 + i * T->step, y0 = T->y0 + j * T->step, s = T->step;
            /* edge points, interpolated canonically from the lower-index node */
            float ex[4], ey[4];
            ex[0] = x0 + s * (L - v[0]) / (v[1] - v[0]); ey[0] = y0;          /* top */
            ex[1] = x0 + s; ey[1] = y0 + s * (L - v[1]) / (v[2] - v[1]);      /* right */
            ex[2] = x0 + s * (L - v[3]) / (v[2] - v[3]); ey[2] = y0 + s;      /* bottom */
            ex[3] = x0; ey[3] = y0 + s * (L - v[0]) / (v[3] - v[0]);          /* left */
            static const signed char tab[16][4] = {
                {-1, -1, -1, -1}, {3, 0, -1, -1}, {0, 1, -1, -1}, {3, 1, -1, -1},
                {1, 2, -1, -1}, {3, 0, 1, 2}, {0, 2, -1, -1}, {3, 2, -1, -1},
                {2, 3, -1, -1}, {0, 2, -1, -1}, {0, 1, 2, 3}, {1, 2, -1, -1},
                {1, 3, -1, -1}, {0, 1, -1, -1}, {3, 0, -1, -1}, {-1, -1, -1, -1}};
            if (c == 5 || c == 10) {                                       /* saddle: decide by the centre */
                int up = (v[0] + v[1] + v[2] + v[3]) / 4 > L;
                static const signed char sad[2][2][4] = {{{3, 0, 1, 2}, {0, 1, 3, 2}}, {{0, 1, 2, 3}, {3, 0, 1, 2}}};
                const signed char *e = sad[c == 10][up];
                path_line(seg, pen, w, ex[e[0]], ey[e[0]], ex[e[1]], ey[e[1]]);
                path_line(seg, pen, w, ex[e[2]], ey[e[2]], ex[e[3]], ey[e[3]]);
                continue;
            }
            path_line(seg, pen, w, ex[tab[c][0]], ey[tab[c][0]], ex[tab[c][1]], ey[tab[c][1]]);
        }
}

static void gen_topo(Rng *R, Paths *P, int accent) {
    Topo T;
    float bw = PW * 0.54f, bh = PH * 0.58f;
    if (PW < PH) { bw = PW * 0.72f; bh = PH * 0.5f; }
    T.step = 1.0f;
    T.nx = (int)(bw / T.step) + 1; T.ny = (int)(bh / T.step) + 1;
    bw = (T.nx - 1) * T.step; bh = (T.ny - 1) * T.step;
    T.x0 = art_cx() - bw / 2; T.y0 = art_cy() - bh / 2;
    T.h = malloc(sizeof(float) * T.nx * T.ny);
    T.shade = malloc(sizeof(float) * T.nx * T.ny);
    float sc = rr(R, 70, 120), ox = rr(R, 0, 200), oy = rr(R, 0, 200);
    uint32_t salt = (uint32_t)ri(R, 1 << 20);
    float lo = 1e9f, hi = -1e9f;
    for (int j = 0; j < T.ny; j++)
        for (int i = 0; i < T.nx; i++) {
            float x = (T.x0 + i * T.step) / sc + ox, y = (T.y0 + j * T.step) / sc + oy;
            float wx = fbm(x * 0.7f + 3.3f, y * 0.7f, 2, salt + 40) - 0.5f;
            float wy = fbm(x * 0.7f, y * 0.7f + 7.1f, 2, salt + 50) - 0.5f;
            float v = fbm(x + wx * 0.9f, y + wy * 0.9f, 4, salt);
            v = v * v;
            T.h[j * T.nx + i] = v;
            lo = fminf(lo, v); hi = fmaxf(hi, v);
        }
    for (int k = 0; k < T.nx * T.ny; k++) T.h[k] = (T.h[k] - lo) / (hi - lo);
    /* hillshade of a smoothed copy, light from the upper left, so the
     * hatching follows the big forms rather than every wrinkle */
    float *sm = malloc(sizeof(float) * T.nx * T.ny), *tmp = malloc(sizeof(float) * T.nx * T.ny);
    memcpy(sm, T.h, sizeof(float) * T.nx * T.ny);
    for (int pass = 0; pass < 3; pass++) {
        const int rad = 3;
        for (int j = 0; j < T.ny; j++)
            for (int i = 0; i < T.nx; i++) {
                float acc = 0; int n = 0;
                for (int k = -rad; k <= rad; k++) { int x = i + k; if (x >= 0 && x < T.nx) { acc += sm[j * T.nx + x]; n++; } }
                tmp[j * T.nx + i] = acc / n;
            }
        for (int j = 0; j < T.ny; j++)
            for (int i = 0; i < T.nx; i++) {
                float acc = 0; int n = 0;
                for (int k = -rad; k <= rad; k++) { int y = j + k; if (y >= 0 && y < T.ny) { acc += tmp[y * T.nx + i]; n++; } }
                sm[j * T.nx + i] = acc / n;
            }
    }
    free(tmp);
    float relief = 60.0f;
    for (int j = 0; j < T.ny; j++)
        for (int i = 0; i < T.nx; i++) {
            int i0 = i ? i - 1 : i, i1 = i < T.nx - 1 ? i + 1 : i, j0 = j ? j - 1 : j, j1 = j < T.ny - 1 ? j + 1 : j;
            float dx = (sm[j * T.nx + i1] - sm[j * T.nx + i0]) * relief / ((i1 - i0) * T.step);
            float dy = (sm[j1 * T.nx + i] - sm[j0 * T.nx + i]) * relief / ((j1 - j0) * T.step);
            float nx = -dx, ny = -dy, nz = 1, nl = sqrtf(nx * nx + ny * ny + nz * nz);
            float lx = -0.55f, ly = -0.55f, lz = 0.63f;
            T.shade[j * T.nx + i] = (nx * lx + ny * ly + nz * lz) / nl;
        }
    free(sm);
    T.wl = rr(R, 0.22f, 0.34f);
    int nlev = 9;
    float dl = (1 - T.wl) / (nlev + 0.5f);
    int wpen = accent ? accent : INK;
    Paths seg = {0};
    contour(&T, T.wl, &seg, INK, 0.5f);                                      /* the coast */
    for (int k = 1; k <= nlev; k++) contour(&T, T.wl + k * dl, &seg, INK, k % 4 == 0 ? 0.5f : 0.3f);
    for (int k = 1; k <= 4; k++) contour(&T, T.wl - k * dl * 0.42f * (1 + 0.25f * k), &seg, wpen, 0.3f);  /* ripples */
    chain(&seg, P, 0.01f);
    paths_free(&seg);
    /* hatching on the shaded slopes: one way, then crossed where darker */
    /* as much of it as the budget leaves room for, darkest first */
    float room = BUDGET * 1.4f - paths_length(P);
    for (float lim = 0.52f;; lim -= 0.04f) {
        Paths h = {0};
        ShadeBand b1 = {&T, -2, lim}, b2 = {&T, -2, lim - 0.15f};
        hatch(&h, INK, 0.3f, 0.785f, 1.7f, T.x0, T.y0, T.x0 + bw, T.y0 + bh, shade_in, &b1, 0);
        hatch(&h, INK, 0.3f, -0.785f, 1.7f, T.x0, T.y0, T.x0 + bw, T.y0 + bh, shade_in, &b2, 0.7f);
        if (paths_length(&h) < room || lim < 0.2f) {
            for (int i = 0; i < h.ns; i++) {
                path_begin(P, h.pen[i], h.w[i]);
                for (int k = 0; k < h.cnt[i]; k++) path_pt(P, h.p[2 * (h.st[i] + k)], h.p[2 * (h.st[i] + k) + 1]);
                path_end(P);
            }
            paths_free(&h);
            break;
        }
        paths_free(&h);
    }
    /* a double-ruled border */
    path_rect(P, INK, 0.5f, T.x0, T.y0, T.x0 + bw, T.y0 + bh);
    path_rect(P, INK, 0.3f, T.x0 - 2.5f, T.y0 - 2.5f, T.x0 + bw + 2.5f, T.y0 + bh + 2.5f);
    free(T.h); free(T.shade);
}

/* -- ridgelines: a landscape of stacked profiles, hidden lines removed -- */
typedef struct { float *hz; float x0, dx; int n; float cx, cy, r; } Sun;
static int sun_in(void *c, float x, float y) {
    Sun *s = c;
    if ((x - s->cx) * (x - s->cx) + (y - s->cy) * (y - s->cy) > s->r * s->r) return 0;
    int i = (int)((x - s->x0) / s->dx + 0.5f);
    if (i < 0 || i >= s->n) return 1;
    return y < s->hz[i] - 1.6f;
}

static void gen_ridges(Rng *R, Paths *P, int accent) {
    float bw = PW * (PW > PH ? 0.58f : 0.74f);
    float x0 = art_cx() - bw / 2, dx = 0.5f;
    int n = (int)(bw / dx) + 1;
    int nl = 42 + ri(R, 12);
    float dy = (PW > PH ? PH * 0.40f : PH * 0.32f) / nl;
    float yb = art_cy() + PH * (PW > PH ? 0.25f : 0.2f);
    float amp = PH * rr(R, 0.13f, 0.19f);
    float sc = rr(R, 28, 48), ox = rr(R, 0, 100);
    uint32_t salt = (uint32_t)ri(R, 1 << 20);
    float pk = rr(R, -0.35f, 0.35f), pk2 = rr(R, -0.8f, 0.8f);
    float *hz = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) hz[i] = 1e9f;
    for (int l = 0; l < nl; l++) {                          /* front to back */
        float depth = (float)l / nl;
        int run = 0;
        float lxv = 0, lyv = 0;
        for (int i = 0; i < n; i++) {
            float x = x0 + i * dx, u = (x - x0) / bw * 2 - 1;
            float env = expf(-(u - pk) * (u - pk) * 3.0f) + 0.45f * expf(-(u - pk2) * (u - pk2) * 9.0f);
            float edge = smoothstepf(1.0f, 0.55f, fabsf(u));
            float f = fbm(x / sc + ox, l * 0.11f + depth * 2, 4, salt);
            float h = amp * env * edge * powf(f, 1.7f) * 2.2f * (0.55f + 0.45f * depth);
            float y = yb - l * dy - h;
            int vis = y < hz[i] - 0.12f;
            if (vis && !run) { path_begin(P, INK, 0.38f); if (i) path_pt(P, lxv, fminf(lyv, hz[i - 1])); run = 1; }
            if (vis) path_pt(P, x, y);
            if (!vis && run) { path_end(P); run = 0; }
            lxv = x; lyv = y;
            if (y < hz[i]) hz[i] = y;
        }
        if (run) path_end(P);
    }
    if (accent && rnd(R) < 0.85f) {
        Sun s = {hz, x0, dx, n, art_cx() + rr(R, -0.22f, 0.22f) * bw, yb - nl * dy - amp * rr(R, 0.6f, 1.2f), PH * rr(R, 0.09f, 0.12f)};
        hatch(P, accent, 0.45f, 0, 1.25f, s.cx - s.r, s.cy - s.r, s.cx + s.r, s.cy + s.r, sun_in, &s, 0.3f);
    }
    free(hz);
}

/* -- curves: harmonograph, spirograph, warped rings -- */
static void gen_harmono(Rng *R, Paths *P, int accent) {
    static const int ratios[][2] = {{1, 2}, {2, 3}, {3, 4}, {3, 5}, {1, 3}, {2, 5}, {4, 5}};
    int k = ri(R, 7);
    float base = 1.0f, A = 0.34f * fminf(PW, PH);
    float f1 = ratios[k][0] * base + rr(R, -0.006f, 0.006f), f2 = ratios[k][1] * base + rr(R, -0.006f, 0.006f);
    float f3 = f1 + rr(R, -0.02f, 0.02f), f4 = f2 + rr(R, -0.02f, 0.02f);
    float p1 = rr(R, 0, TAU), p2 = rr(R, 0, TAU), p3 = rr(R, 0, TAU), p4 = rr(R, 0, TAU);
    float mixa = rr(R, 0.25f, 0.45f);
    float cx = art_cx(), cy = art_cy();
    float dt = 0.4f / (A * (f2 > f1 ? f2 : f1) * 1.2f);
    /* damping such that the whole decay (to 30%: evenly spaced loops) takes the budget */
    double L0 = 0;
    for (float t = dt; t < 200; t += dt) {
        float x0 = (1 - mixa) * sinf(f1 * (t - dt) + p1) + mixa * sinf(f4 * (t - dt) + p4), x1 = (1 - mixa) * sinf(f1 * t + p1) + mixa * sinf(f4 * t + p4);
        float y0 = (1 - mixa) * sinf(f2 * (t - dt) + p2) + mixa * sinf(f3 * (t - dt) + p3), y1 = (1 - mixa) * sinf(f2 * t + p2) + mixa * sinf(f3 * t + p3);
        L0 += A * hypotf(x1 - x0, y1 - y0);
    }
    float d = (float)(L0 / 200 * 0.7 / (BUDGET * 1.5f));
    double L = 0;
    float px = 0, py = 0;
    float split = accent && rnd(R) < 0.7f ? BUDGET * rr(R, 0.22f, 0.35f) : -1;
    int pen = split > 0 ? accent : INK;
    path_begin(P, pen, NIB);
    for (float t = 0;; t += dt) {
        float e = expf(-d * t);
        if (e < 0.3f || L > BUDGET * 1.6f) break;
        float x = cx + A * e * ((1 - mixa) * sinf(f1 * t + p1) + mixa * sinf(f4 * t + p4));
        float y = cy + A * e * ((1 - mixa) * sinf(f2 * t + p2) + mixa * sinf(f3 * t + p3));
        if (t > 0 && hypotf(x - px, y - py) < 0.4f) continue;
        if (t > 0) L += hypotf(x - px, y - py);
        path_pt(P, x, y);
        if (split > 0 && L > split) {                          /* change pens here */
            path_end(P); path_begin(P, INK, NIB); path_pt(P, x, y); split = -1;
        }
        px = x; py = y;
    }
    path_end(P);
}

static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

static float hypo(Paths *P, int pen, float cx, float cy, float Rm, int Ri, int ri_, float dr, float rot, int dry) {
    float Rr = (float)Ri, r = (float)ri_, dd = dr * r;
    float s = Rm / (Rr - r + dd);
    int g = gcd(Ri, ri_);
    float T = TAU * ri_ / g;
    int n = (int)(T / 0.004f);
    double L = 0;
    float px = 0, py = 0;
    if (!dry) path_begin(P, pen, NIB);
    for (int i = 0; i <= n; i++) {
        float t = T * i / n;
        float x = (Rr - r) * cosf(t) + dd * cosf((Rr - r) / r * t);
        float y = (Rr - r) * sinf(t) - dd * sinf((Rr - r) / r * t);
        float X = cx + s * (x * cosf(rot) - y * sinf(rot)), Y = cy + s * (x * sinf(rot) + y * cosf(rot));
        if (i) L += hypotf(X - px, Y - py);
        if (!dry) path_pt(P, X, Y);
        px = X; py = Y;
    }
    if (!dry) path_end(P);
    return (float)L;
}

static void gen_spiro(Rng *R, Paths *P, int accent) {
    float Rm = 0.37f * fminf(PW, PH), cx = art_cx(), cy = art_cy();
    int bR = 7, br = 3;
    float bd = 0.8f, L = 0;
    for (int tries = 0; tries < 400; tries++) {
        int Ri = 5 + ri(R, 14), r = 2 + ri(R, Ri - 3);
        float dr = rr(R, 0.5f, 1.3f);
        float l = hypo(P, 0, cx, cy, Rm, Ri, r, dr, 0, 1);
        if (l > BUDGET * 0.12f && l < BUDGET * 0.4f) { bR = Ri; br = r; bd = dr; L = l; break; }
    }
    if (L <= 0) L = hypo(P, 0, cx, cy, Rm, bR, br, bd, 0, 1);
    /* the same wheel again and again, each pass smaller and turned a little */
    float rot = rr(R, 0, TAU), twist = rr(R, -0.12f, 0.12f), shrink = rr(R, 0.07f, 0.12f);
    int every = 2 + ri(R, 2);
    float used = 0;
    for (int i = 0; i < 12; i++) {
        float s = 1 - i * shrink;
        if (s < 0.25f || used + L * s > BUDGET * 1.05f) break;
        used += hypo(P, accent && i % every == every - 1 ? accent : INK, cx, cy, Rm * s, bR, br, bd, rot + i * twist, 0);
    }
}

static void gen_rings(Rng *R, Paths *P, int accent) {
    float cx = art_cx(), cy = art_cy();
    float rmax = 0.38f * fminf(PW, PH), r0 = rr(R, 3, 12);
    /* sum of 2 pi r ~ budget */
    int N = (int)(BUDGET / (3.14159f * (rmax + r0)));
    float gap = (rmax - r0) / N;
    float fq = rr(R, 0.8f, 1.6f), amp = rr(R, 0.18f, 0.3f) * rmax;
    float drx = rr(R, -0.12f, 0.12f) * rmax, dry = rr(R, -0.12f, 0.12f) * rmax;
    uint32_t salt = (uint32_t)ri(R, 1 << 20);
    int a0 = accent ? N / 3 + ri(R, N / 3) : -1, a1 = a0 + 3 + ri(R, 4);
    float ox = rr(R, 0, 50), oy = rr(R, 0, 50);
    for (int i = 0; i < N; i++) {
        float r = r0 + i * gap, grow = powf((float)i / N, 1.3f);
        int n = (int)(TAU * r / 0.6f) + 12;
        float start = rr(R, 0, TAU);
        path_begin(P, i >= a0 && i < a1 ? accent : INK, NIB);
        for (int k = 0; k <= n; k++) {
            float a = start + TAU * k / n;
            float nx = cosf(a) * fq + ox, ny = sinf(a) * fq + oy;
            float rr_ = r + amp * grow * (fbm(nx, ny + i * 0.035f, 3, salt) - 0.5f) * 2;
            path_pt(P, cx + drx * (1 - grow) + rr_ * cosf(a), cy + dry * (1 - grow) + rr_ * sinf(a));
        }
        path_end(P);
    }
}

/* -- truchet: quarter arcs, several strands, woven into long curves -- */
static void gen_truchet(Rng *R, Paths *P, int accent) {
    int k = rnd(R) < 0.6f ? 3 : 2;
    float c = rr(R, 20, 27);
    float per_tile = 2 * k * (TAU / 4) * (c / 2);
    int tiles = (int)(BUDGET * 0.95f / per_tile);
    float aspect = PW > PH ? 1.35f : 0.75f;
    int nx = (int)roundf(sqrtf(tiles * aspect)), ny = tiles / (nx ? nx : 1);
    nx = nx < 2 ? 2 : nx; ny = ny < 2 ? 2 : ny;
    float x0 = art_cx() - nx * c / 2, y0 = art_cy() - ny * c / 2;
    Paths seg = {0};
    float g = c / (k + 1) * (k == 3 ? 0.9f : 0.8f);
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            int o = rnd(R) < 0.5f;
            float tx = x0 + i * c, ty = y0 + j * c;
            for (int a = 0; a < 2; a++) {
                /* corner and the quarter it sweeps: o=0 tl & br, o=1 tr & bl */
                int corner = o ? (a ? 3 : 1) : (a ? 2 : 0);   /* 0 tl, 1 tr, 2 br, 3 bl */
                float ccx = tx + (corner == 1 || corner == 2 ? c : 0), ccy = ty + (corner >= 2 ? c : 0);
                float a0 = corner == 0 ? 0 : corner == 1 ? TAU / 4 : corner == 2 ? TAU / 2 : 3 * TAU / 4;
                for (int s = 0; s < k; s++) {
                    float r = c / 2 + (s - (k - 1) * 0.5f) * g;
                    int pen = (k == 3 && s == 1 && accent) ? accent : INK;
                    path_begin(&seg, pen, NIB);
                    for (int m = 0; m <= 10; m++) {
                        float an = a0 + TAU / 4 * m / 10;
                        float px = ccx + r * cosf(an), py = ccy + r * sinf(an);
                        if (m == 0 || m == 10) { px = roundf(px * 1000) / 1000; py = roundf(py * 1000) / 1000; }
                        path_pt(&seg, px, py);
                    }
                    path_end(&seg);
                }
            }
        }
    chain(&seg, P, 0.02f);
    paths_free(&seg);
    if (accent && k == 2)                              /* two strands: colour a few whole curves */
        for (int i = 0; i < P->ns; i++) if (rnd(R) < 0.18f) P->pen[i] = (unsigned char)accent;
    path_rect(P, INK, 0.5f, x0, y0, x0 + nx * c, y0 + ny * c);
}

enum { K_FLOW, K_TOPO, K_RIDGES, K_HARMONO, K_SPIRO, K_RINGS, K_TRUCHET, NKIND };
static const char *KIND_NAME[NKIND] = {"flow", "topo", "ridges", "harmonograph", "spirograph", "rings", "truchet"};
static void (*const GEN[NKIND])(Rng *, Paths *, int) = {gen_flow, gen_topo, gen_ridges, gen_harmono, gen_spiro, gen_rings, gen_truchet};

/* ===================================================================== */
/* the caption: a single-stroke script                                   */
/* ===================================================================== */
/* Units: baseline 0, x-height 7, caps 11, descenders -4; y up. Points are
 * "x,y"; "!" after one makes it a corner; "|" starts a new stroke. The
 * strokes are smoothed through Catmull-Rom splines and slanted. '~' is an
 * em dash. */
typedef struct { char c; float w; const char *s; } Glyph;
static const Glyph FONT[] = {
    {'a', 7, "5.5,5.5 4,7 2,6.6 0.5,4 0.8,1 2.5,0 4.3,1 5.5,3.5 | 5.8,7 5.6,1.5 6.6,0"},
    {'b', 7, "1,11 1,0.5 | 1,3.8 2.5,6.3 4.5,7 6,5.2 6,2 4.5,0.2 2.5,0.1 1,1.2"},
    {'c', 6, "5.5,5.8 4,7 2,6.6 0.5,4 0.8,1.2 2.5,0 4.2,0.3 5.5,1.5"},
    {'d', 7, "5.5,5 4,7 2,6.6 0.5,4 0.8,1 2.5,0 4.3,1 5.5,3.5 | 5.8,11 5.6,1.5 6.6,0"},
    {'e', 6.5, "0.8,3.5 5.5,3.8 5.6,5.5 4,7 2,6.7 0.5,4.3 0.8,1.4 2.6,0 4.5,0.3 5.6,1.3"},
    {'f', 5, "4.8,10.2 3.5,11 2.3,10.3 2,8 2,0 | 0.3,7 4.2,7"},
    {'g', 7, "5.5,5.5 4,7 2,6.6 0.5,4 0.8,1.3 2.5,0.3 4.3,1.2 5.5,3.5 | 5.8,7 5.6,-2.5 4.3,-4 2,-4 0.6,-3"},
    {'h', 7, "1,11 1,0 | 1,4 2.5,6.4 4.3,7 5.6,6 5.8,4 5.8,0"},
    {'i', 3, "1.5,7 1.4,1 2.4,0 | 1.6,9.6 1.6,9.4"},
    {'j', 3.5, "2.2,7 2.1,-2.5 1,-4 -0.5,-3.6 | 2.3,9.6 2.3,9.4"},
    {'k', 6.5, "1,11 1,0 | 5.5,7 1.1,3! 5.8,0"},
    {'l', 3, "1.6,11 1.4,1.2 2.6,0"},
    {'m', 10, "1,7 1,0 | 1,4.5 2.3,6.6 3.6,7 4.8,6 5,4 5,0 | 5,4.5 6.3,6.6 7.6,7 8.8,6 9,4 9,0"},
    {'n', 7, "1,7 1,0 | 1,4.5 2.4,6.5 4.2,7 5.5,6 5.8,4 5.8,0"},
    {'o', 7, "3.3,7 1.2,6 0.4,3.5 1,1 3,0 5.2,0.8 6.2,3.5 5.6,6 3.3,7 2.6,6.8"},
    {'p', 7, "1,7 1,-4 | 1,3.8 2.5,6.3 4.5,7 6,5.2 6,2 4.5,0.2 2.5,0.1 1,1.2"},
    {'q', 7, "5.5,5.5 4,7 2,6.6 0.5,4 0.8,1.3 2.5,0.3 4.3,1.2 5.5,3.5 | 5.8,7 5.6,-4 6.6,-3"},
    {'r', 5, "1,7 1,0 | 1,4 2.3,6.3 3.6,7 4.8,6.6"},
    {'s', 6, "5,6.2 3.4,7 1.5,6.7 0.8,5.3 1.8,4 4.2,3 5.2,1.6 4.3,0.2 2.3,0 0.6,0.8"},
    {'t', 5, "2,10 1.9,1.2 3,0 4.5,0.6 | 0.2,7 4.2,7"},
    {'u', 7, "1,7 1,2.5 1.8,0.4 3.2,0 4.6,0.8 5.8,3 | 5.9,7 5.8,1.2 6.8,0"},
    {'v', 6.5, "0.3,7 3.2,0! 6.2,7"},
    {'w', 9.5, "0.3,7 2.5,0! 4.7,6! 7,0! 9.2,7"},
    {'x', 6.5, "0.5,7 6,0 | 6,7 0.5,0"},
    {'y', 6.5, "0.5,7 3.4,0.5 | 6.2,7 3,-2 1.8,-3.7 0.4,-4"},
    {'z', 6.5, "0.6,7 5.8,7! 0.6,0! 6,0"},
    {'A', 9, "0.3,0 4.5,11! 8.7,0 | 2,4 7,4"},
    {'B', 8, "1,0 1,11! 4.5,11 6.5,10 7,8.3 6.2,6.4 4,5.8 1,5.8 | 4,5.8 6.4,5.2 7.4,3.2 6.8,1.2 4.8,0 1,0"},
    {'C', 8.5, "8,9 6.5,10.7 4.5,11 2.3,10 0.8,7.8 0.5,4.5 1.5,1.7 3.6,0.1 5.8,0.1 7.6,1.3 8.2,2.3"},
    {'D', 8.5, "1,0 1,11! 4,11 6.5,10 7.9,7.8 8.1,4.5 7.2,1.8 5,0.2 1,0"},
    {'E', 7.5, "7,11 1,11! 1,0! 7.2,0 | 1,5.7 5.8,5.7"},
    {'F', 7, "7,11 1,11! 1,0 | 1,5.7 5.5,5.7"},
    {'G', 9, "8,9 6.5,10.7 4.5,11 2.3,10 0.8,7.8 0.5,4.5 1.5,1.7 3.6,0.1 5.8,0.1 7.6,1.3 8.3,3.5! 8.3,5! 5,5"},
    {'H', 8.5, "1,11 1,0 | 7.5,11 7.5,0 | 1,5.7 7.5,5.7"},
    {'I', 3, "1.5,11 1.5,0"},
    {'J', 6, "5,11 5,3 4.2,0.8 2.6,0 1,0.6 0.2,2"},
    {'K', 8, "1,11 1,0 | 7.3,11 1.1,4.5 | 3.2,6.6 7.6,0"},
    {'L', 7, "1,11 1,0! 6.8,0"},
    {'M', 10.5, "1,0 1.2,11! 5.2,2.5! 9.3,11! 9.5,0"},
    {'N', 9, "1,0 1,11! 7.8,0! 7.8,11"},
    {'O', 9.5, "4.7,11 2.3,10 0.7,7.5 0.5,4 1.6,1.4 4,0 6.6,0.5 8.4,2.8 8.8,6.2 7.8,9.2 5.6,10.9 4.7,11 3.6,10.6"},
    {'P', 8, "1,0 1,11! 4.5,11 6.6,10.2 7.3,8.3 6.6,6.4 4.5,5.5 1,5.5"},
    {'Q', 9.5, "4.7,11 2.3,10 0.7,7.5 0.5,4 1.6,1.4 4,0 6.6,0.5 8.4,2.8 8.8,6.2 7.8,9.2 5.6,10.9 4.7,11 3.6,10.6 | 5.5,2.5 8.8,-0.5"},
    {'R', 8, "1,0 1,11! 4.5,11 6.6,10.2 7.3,8.3 6.6,6.4 4.5,5.5 1,5.5 | 4,5.5 7.5,0"},
    {'S', 7.5, "6.8,9.6 5.3,10.8 3.4,11 1.6,10.3 1,8.6 1.7,7 3.8,6 5.8,4.9 6.9,3.2 6.5,1.2 4.6,0 2.4,0 0.6,1.3"},
    {'T', 8, "0.2,11 7.8,11 | 4,11 4,0"},
    {'U', 8.5, "1,11 1,3.5 1.8,1 4,0 6.2,0.8 7.4,3.2 7.5,11"},
    {'V', 8.5, "0.3,11 4.2,0! 8.2,11"},
    {'W', 12, "0.3,11 3,0! 6,8.5! 9,0! 11.7,11"},
    {'X', 8, "0.5,11 7.5,0 | 7.5,11 0.5,0"},
    {'Y', 8, "0.3,11 4,5.5! 7.7,11 | 4,5.5 4,0"},
    {'Z', 8, "0.8,11 7.4,11! 0.6,0! 7.6,0"},
    {'0', 7, "3.5,10.5 1.5,9.6 0.5,6.5 0.5,4 1.4,1 3.5,0 5.5,0.9 6.5,4 6.5,6.5 5.6,9.5 3.5,10.5 2.8,10.3"},
    {'1', 5.5, "1.3,8.6 4,10.5! 4,0"},
    {'2', 7, "0.8,8.6 1.8,10 3.5,10.5 5.3,9.9 6.1,8.2 5.6,6.2 0.6,0! 6.5,0"},
    {'3', 7, "0.8,9.4 2.5,10.4 4.3,10.4 5.8,9.3 5.8,7.4 4.4,6 2.6,5.8 | 4.4,6 6.2,4.8 6.6,2.6 5.6,0.8 3.7,0 1.8,0.2 0.4,1.4"},
    {'4', 7, "5,0 5,10.5! 0.3,3! 6.8,3"},
    {'5', 7, "6,10.5 1.4,10.5! 0.9,5.6! 2.7,6.4 4.6,6.4 6.2,5.2 6.6,3 5.8,1 3.8,0 1.8,0.2 0.4,1.3"},
    {'6', 7, "5.6,10 4,10.5 2.2,9.8 0.9,7.6 0.5,4.5 0.9,1.6 2.6,0.1 4.6,0.1 6.1,1.4 6.4,3.4 5.6,5.4 3.8,6.2 2,5.8 0.7,4.3"},
    {'7', 7, "0.5,10.5 6.6,10.5! 2.6,0"},
    {'8', 7, "3.5,5.8 1.5,6.8 1,8.7 2,10.2 3.5,10.5 5,10.2 6,8.7 5.5,6.8 3.5,5.8 1.2,4.8 0.5,2.8 1.4,0.7 3.5,0 5.6,0.7 6.5,2.8 5.8,4.8 3.5,5.8"},
    {'9', 7, "6.2,6.2 5.2,4.6 3.4,4.2 1.6,4.9 0.7,6.8 1.1,9 2.6,10.4 4.6,10.4 6,9.1 6.5,6.6 6.3,3.2 5.2,1 3.4,0 1.6,0.4"},
    {',', 2.5, "1.4,0.6 1.4,0.3 0.6,-1.8"},
    {'.', 2.5, "1.2,0.4 1.2,0.2"},
    {':', 2.5, "1.4,6.4 1.4,6.2 | 1.2,0.4 1.2,0.2"},
    {'-', 5, "0.8,4.5 4.2,4.5"},
    {'_', 6, "0.2,-1 5.8,-1"},
    {'~', 13, "0.8,4.3 12.2,4.5"},
    {'\'', 2, "1.2,11 1,8"},
    {'/', 5, "0.3,-1 4.7,11.5"},
    {' ', 3.5, ""},
};

static const Glyph *glyph(char c) {
    for (size_t i = 0; i < sizeof FONT / sizeof *FONT; i++) if (FONT[i].c == c) return &FONT[i];
    return NULL;
}
#define GAP 1.3f

static float text_width(const char *s) {
    float w = 0;
    for (; *s; s++) { const Glyph *g = glyph(*s); w += (g ? g->w : 3.5f) + GAP; }
    return w - GAP;
}

/* lay out s with its baseline at y, starting at x; sc mm per unit */
static void text_paths(Rng *R, Paths *P, const char *s, float x, float y, float sc, int pen, float w) {
    const float slant = 0.2f;
    float pts[64][2];
    for (; *s; s++) {
        const Glyph *g = glyph(*s);
        float gw = g ? g->w : 3.5f;
        if (g && *g->s) {
            float jy = rr(R, -0.25f, 0.25f), rot = rr(R, -0.03f, 0.03f), gs = rr(R, 0.96f, 1.04f);
            const char *p = g->s;
            while (*p) {
                int n = 0;
                while (*p && *p != '|') {
                    char *e;
                    float px = strtof(p, &e);
                    if (e == p) { p++; continue; }
                    p = e + 1;
                    float py = strtof(p, &e);
                    p = e;
                    int corner = *p == '!';
                    if (corner) p++;
                    for (int k = 0; k <= corner && n < 64; k++) { pts[n][0] = px; pts[n][1] = py; n++; }
                    while (*p == ' ') p++;
                }
                if (*p == '|') p++;
                if (n < 2) continue;
                path_begin(P, pen, w);
                for (int i = 0; i + 1 < n; i++) {
                    const float *p0 = pts[i ? i - 1 : 0], *p1 = pts[i], *p2 = pts[i + 1], *p3 = pts[i + 2 < n ? i + 2 : n - 1];
                    float e0[2] = {p0[0], p0[1]}, e3[2] = {p3[0], p3[1]};
                    if (i == 0) { e0[0] = 2 * p1[0] - p2[0]; e0[1] = 2 * p1[1] - p2[1]; }
                    if (i + 2 >= n) { e3[0] = 2 * p2[0] - p1[0]; e3[1] = 2 * p2[1] - p1[1]; }
                    int steps = (p1[0] == p2[0] && p1[1] == p2[1]) ? 0 : 6;
                    for (int k = (i == 0 ? 0 : 1); k <= steps; k++) {
                        float t = (float)k / (steps ? steps : 1), t2 = t * t, t3 = t2 * t;
                        float q[2];
                        for (int d = 0; d < 2; d++)
                            q[d] = 0.5f * (2 * p1[d] + (-e0[d] + p2[d]) * t + (2 * e0[d] - 5 * p1[d] + 4 * p2[d] - e3[d]) * t2 +
                                           (-e0[d] + 3 * p1[d] - 3 * p2[d] + e3[d]) * t3);
                        float gx = q[0] * gs, gy = q[1] * gs + jy;
                        float rx = gx * cosf(rot) - gy * sinf(rot), ry = gx * sinf(rot) + gy * cosf(rot);
                        path_pt(P, x + (rx + ry * slant) * sc, y - ry * sc);
                    }
                }
                path_end(P);
            }
        }
        x += (gw + GAP) * sc;
    }
}

/* ===================================================================== */
/* the planner: strokes -> a timed path for the pen                      */
/* ===================================================================== */
typedef struct { float x, y, ix, iy, t, v, w, k, z; unsigned char pen; } Smp;

#define VMAX 42.0f       /* pen down, mm/s */
#define ACC 320.0f       /* mm/s^2 */
#define VTRAV 120.0f     /* pen up */
#define ATRAV 600.0f
#define T_LOWER 0.11f
#define T_RAISE 0.09f
#define TARGET 200.0f    /* seconds a drawing aims to take */

typedef struct {
    Smp *s; int n, cap;
    float t_end;                        /* plotting done, parked */
    int kind;
} Plan;

static Plan plan;
static float pl_x, pl_y, pl_t;
static int pl_pen;
static uint32_t pl_salt;
static float park_x, park_y;

static void push(float x, float y, float ix, float iy, float v, float w, float k, float z) {
    GROW(plan.s, plan.n, plan.cap, plan.n + 1);
    plan.s[plan.n++] = (Smp){x, y, ix, iy, pl_t, v, w, k, z, (unsigned char)pl_pen};
    pl_x = x; pl_y = y;
}

static void travel(float x, float y) {
    float D = hypotf(x - pl_x, y - pl_y);
    if (D < 0.05f) { if (D > 0) push(x, y, x, y, 0, 0, 0, 1); return; }
    float ta = VTRAV / ATRAV, da = 0.5f * ATRAV * ta * ta, vpk = VTRAV;
    if (2 * da > D) { ta = sqrtf(D / ATRAV); da = D / 2; vpk = ATRAV * ta; }
    float tc = (D - 2 * da) / vpk, T = 2 * ta + tc;
    int n = (int)clampf(T * 30, 4, 60);
    float x0 = pl_x, y0 = pl_y, t0 = pl_t;
    for (int i = 1; i <= n; i++) {
        float t = T * i / n, s;
        if (t < ta) s = 0.5f * ATRAV * t * t;
        else if (t < ta + tc) s = da + vpk * (t - ta);
        else { float u = T - t; s = D - 0.5f * ATRAV * u * u; }
        pl_t = t0 + t;
        float f = s / D;
        push(x0 + (x - x0) * f, y0 + (y - y0) * f, 0, 0, 0, 0, 0, 1);
    }
}

static void pen_change(int pen) {
    if (pen == pl_pen) return;
    travel(park_x, park_y);
    pl_t += 0.3f; push(park_x, park_y, 0, 0, 0, 0, 0, 2.5f);     /* the pen comes out */
    pl_t += 1.1f; pl_pen = pen; push(park_x, park_y, 0, 0, 0, 0, 0, 2.5f);
    pl_t += 0.5f; push(park_x, park_y, 0, 0, 0, 0, 0, 1);         /* the new one goes in */
    pl_t += 0.4f; push(park_x, park_y, 0, 0, 0, 0, 0, 1);
}

/* draw one polyline: resample, wobble, plan the speed along it */
static float *rs, *rv, *rl;
static int rcap;

static void stroke(const float *pts, int n, float w, float k) {
    /* resample to <= 0.5 mm */
    int m = 0;
    for (int i = 0; i < n; i++) {
        int sub = 1;
        if (i) {
            float d = hypotf(pts[2 * i] - pts[2 * i - 2], pts[2 * i + 1] - pts[2 * i - 1]);
            if (d < 1e-4f) continue;
            sub = (int)ceilf(d / 0.5f);
        }
        GROW(rs, rcap, rcap, 2 * (m + sub + 1));
        if (!i) { rs[0] = pts[0]; rs[1] = pts[1]; m = 1; continue; }
        for (int j = 1; j <= sub; j++) {
            float f = (float)j / sub;
            rs[2 * m] = pts[2 * i - 2] + (pts[2 * i] - pts[2 * i - 2]) * f;
            rs[2 * m + 1] = pts[2 * i - 1] + (pts[2 * i + 1] - pts[2 * i - 1]) * f;
            m++;
        }
    }
    rv = xrealloc(rv, sizeof(float) * (m + 1));
    rl = xrealloc(rl, sizeof(float) * (m + 1));
    /* speed limits: zero at the ends, and v^2 / r <= ACC round curves */
    for (int i = 0; i < m; i++) {
        if (i == 0 || i == m - 1) { rv[i] = 0; continue; }
        float ax = rs[2 * i] - rs[2 * i - 2], ay = rs[2 * i + 1] - rs[2 * i - 1];
        float bx = rs[2 * i + 2] - rs[2 * i], by = rs[2 * i + 3] - rs[2 * i + 1];
        float la = hypotf(ax, ay), lb = hypotf(bx, by);
        float c = (ax * bx + ay * by) / (la * lb + 1e-9f);
        float phi = acosf(clampf(c, -1, 1));
        float r = 0.5f * (la + lb) / fmaxf(phi, 1e-5f);
        rv[i] = fminf(VMAX, sqrtf(ACC * 0.6f * r));
        if (phi > 1.2f) rv[i] = fminf(rv[i], 1.5f);              /* sharp corners: all but stop */
    }
    for (int i = 1; i < m; i++) {
        float d = hypotf(rs[2 * i] - rs[2 * i - 2], rs[2 * i + 1] - rs[2 * i - 1]);
        rv[i] = fminf(rv[i], sqrtf(rv[i - 1] * rv[i - 1] + 2 * ACC * d));
    }
    for (int i = m - 2; i >= 0; i--) {
        float d = hypotf(rs[2 * i + 2] - rs[2 * i], rs[2 * i + 3] - rs[2 * i + 1]);
        rv[i] = fminf(rv[i], sqrtf(rv[i + 1] * rv[i + 1] + 2 * ACC * d));
    }
    /* go there, lower the pen, let the ink pool, draw, lift */
    travel(rs[0], rs[1]);
    float s = 0;
    uint32_t salt = pl_salt++;
    float x = rs[0], y = rs[1];
    pl_t += T_LOWER; push(x, y, x, y, 0, w, k, 0);
    pl_t += 0.07f; push(x, y, x, y, 0, w, k, 0);
    for (int i = 1; i < m; i++) {
        float d = hypotf(rs[2 * i] - rs[2 * i - 2], rs[2 * i + 1] - rs[2 * i - 1]);
        s += d;
        pl_t += 2 * d / fmaxf(rv[i - 1] + rv[i], 0.2f);
        int a = i > 1 ? i - 1 : 0, b = i < m - 1 ? i + 1 : m - 1;
        float tx = rs[2 * b] - rs[2 * a], ty = rs[2 * b + 1] - rs[2 * a + 1], tl = hypotf(tx, ty) + 1e-9f;
        float off = (vnoise(s / 14.0f, 0.5f, salt) - 0.5f) * 0.3f + (vnoise(s / 3.0f, 3.5f, salt) - 0.5f) * 0.09f;
        float ix = rs[2 * i] - ty / tl * off, iy = rs[2 * i + 1] + tx / tl * off;
        push(rs[2 * i], rs[2 * i + 1], ix, iy, rv[i], w, k, 0);
    }
    float ex = plan.s[plan.n - 1].ix, ey = plan.s[plan.n - 1].iy;
    pl_t += 0.03f; push(pl_x, pl_y, ex, ey, 0, w, k, 0);
    pl_t += T_RAISE; push(pl_x, pl_y, 0, 0, 0, 0, 0, 1);
}

/* plot all paths of the set: one pen at a time (accents first), each
 * layer ordered nearest-neighbour; keep_order for handwriting */
static void plot_paths(const Paths *P, Rng *R, int keep_order) {
    char *done = calloc((size_t)P->ns + 1, 1);
    float *buf = NULL;
    int bcap = 0;
    static const int order[NPEN] = {ACC1, ACC2, ACC3, INK};
    for (int oi = 0; oi < NPEN; oi++) {
        int pen = order[oi], left = 0;
        for (int i = 0; i < P->ns; i++) left += P->pen[i] == pen;
        if (!left) continue;
        pen_change(pen);
        while (left--) {
            int best = -1, rev = 0;
            float bd = 1e30f;
            for (int i = 0; i < P->ns; i++) {
                if (done[i] || P->pen[i] != pen) continue;
                const float *a = P->p + 2 * P->st[i], *b = P->p + 2 * (P->st[i] + P->cnt[i] - 1);
                float da = (a[0] - pl_x) * (a[0] - pl_x) + (a[1] - pl_y) * (a[1] - pl_y);
                float db = (b[0] - pl_x) * (b[0] - pl_x) + (b[1] - pl_y) * (b[1] - pl_y);
                if (keep_order) { best = i; rev = 0; break; }
                if (da < bd) { bd = da; best = i; rev = 0; }
                if (db < bd) { bd = db; best = i; rev = 1; }
            }
            done[best] = 1;
            int c = P->cnt[best];
            GROW(buf, bcap, bcap, 2 * c);
            for (int j = 0; j < c; j++) {
                const float *q = P->p + 2 * (P->st[best] + (rev ? c - 1 - j : j));
                buf[2 * j] = q[0]; buf[2 * j + 1] = q[1];
            }
            stroke(buf, c, P->w[best], rr(R, 0.9f, 1.0f));
        }
    }
    free(buf);
    free(done);
}

/* ===================================================================== */
/* sheets and the timeline                                                */
/* ===================================================================== */
#define T_SLIDE 2.8
#define T_SETTLE 1.2
#define T_REST 10.0

static char host[64];
static time_t wall0;                    /* the wall clock at t = 0 */
static int sheet_k = -1, prev_kind = -1;
static double sheet_T;                  /* global time this sheet started sliding in */
static double sheet_cycle;

static float paper_rgb[3], pen_T[NPEN][3], pen_rgb[NPEN][3];
static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

/* this sheet's pens: the set's own order, or (shuffle) a fresh main ink
 * and the rest as accents; on its own stream so the art stays the same */
static void pick_pens(int k) {
    for (int p = 0; p < NPEN; p++) sheet_pen[p] = p;
    if (pal->shuffle) {
        Rng Q = {mix32(opt_seed * 0x2545f491u + (uint32_t)k * 0x9e3779b9u + 7) | 1};
        for (int i = NPEN - 1; i > 0; i--) { int j = ri(&Q, i + 1), t = sheet_pen[i]; sheet_pen[i] = sheet_pen[j]; sheet_pen[j] = t; }
    }
    hex(pal->paper, paper_rgb);
    for (int p = 0; p < NPEN; p++) {
        hex(pal->pen[sheet_pen[p]], pen_rgb[p]);
        for (int i = 0; i < 3; i++) pen_T[p][i] = pen_rgb[p][i] / paper_rgb[i];
    }
}

static void make_sheet(int k) {
    Rng R = {mix32(opt_seed * 0x9e3779b9u + (uint32_t)k * 0x85ebca6bu + 1) | 1};
    pick_pens(k);
    int kind = opt_kind >= 0 ? opt_kind : ri(&R, NKIND);
    if (opt_kind < 0 && kind == prev_kind) kind = (kind + 1 + ri(&R, NKIND - 1)) % NKIND;
    prev_kind = kind;
    int accent = rnd(&R) < 0.85f ? 1 + ri(&R, 3) : 0;

    Paths art = {0};
    GEN[kind](&R, &art, accent);

    plan.n = 0; plan.kind = kind;
    pl_t = 0; pl_salt = R.s;
    park_x = -0.035f * PW; park_y = -0.03f * PH;
    pl_x = park_x; pl_y = park_y;
    pl_pen = INK;
    for (int i = 0; i < art.ns; i++) if (art.pen[i] != INK) { pl_pen = art.pen[i]; break; }
    push(park_x, park_y, 0, 0, 0, 0, 0, 1);
    plot_paths(&art, &R, 0);
    /* a plotter speed setting, so the piece takes a few minutes */
    float f = clampf(pl_t / TARGET, 0.8f, 2.6f);
    for (int i = 0; i < plan.n; i++) plan.s[i].t /= f;
    pl_t /= f;
    float art_len = paths_length(&art);
    paths_free(&art);

    /* sign it: host and the time it's signed at, bottom right */
    time_t when = wall0 + (time_t)(sheet_T + T_SLIDE + T_SETTLE + pl_t + 4);
    struct tm tmv;
    localtime_r(&when, &tmv);
    char date[64], cap[160];
    strftime(date, sizeof date, "%A %-d %B, %H:%M", &tmv);
    snprintf(cap, sizeof cap, "%s ~ %s", host, date);
    float sc = 0.4f;
    float tw = text_width(cap) * sc;
    Paths txt = {0};
    text_paths(&R, &txt, cap, PW - PW * 0.06f - tw, PH - PH * 0.045f, sc, INK, 0.36f);
    pen_change(INK);
    plot_paths(&txt, &R, 1);
    paths_free(&txt);
    travel(park_x, park_y);
    plan.t_end = pl_t;
    sheet_cycle = 2 * T_SLIDE + T_SETTLE + plan.t_end + T_REST;
    if (getenv("PLOTTER_DEBUG"))
        fprintf(stderr, "sheet %d: %s, accent %d, %.1f m of ink, %d samples, speed x%.2f, %.0f s\n", k, KIND_NAME[kind],
                accent, art_len / 1000, plan.n, f, plan.t_end);
}

/* ===================================================================== */
/* GL                                                                     */
/* ===================================================================== */
#define GLSL(...) #__VA_ARGS__

/* ink: one antialiased capsule per instance, widths and densities varying
 * along it, min-blended into the sheet (white = no ink) */
static const char *VS_SEG = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aP;      /* ends, px */
    layout(location = 1) in vec4 aW;      /* radii px, densities */
    layout(location = 2) in vec3 aC;      /* the ink's transmission */
    uniform vec2 uSize;
    flat out vec4 vP;
    flat out vec4 vW;
    flat out vec3 vC;
    void main() {
        vec2 a = aP.xy, b = aP.zw, d = b - a;
        float r = max(max(aW.x, aW.y), 0.5) + 1.0, L = length(d);
        vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0), n = vec2(-t.y, t.x);
        vec2 c = ((gl_VertexID & 1) == 0 ? a - t * r : b + t * r) + ((gl_VertexID & 2) == 0 ? -n : n) * r;
        vP = aP; vW = aW; vC = aC;
        gl_Position = vec4(c / uSize * 2.0 - 1.0, 0.0, 1.0);
    });
static const char *FS_SEG = "#version 330 core\n" GLSL(
    flat in vec4 vP;
    flat in vec4 vW;
    flat in vec3 vC;
    out vec4 o;
    void main() {
        vec2 p = gl_FragCoord.xy, a = vP.xy, ba = vP.zw - a, pa = p - a;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        float d = length(pa - ba * h);
        float r = mix(vW.x, vW.y, h), rr = max(r, 0.5);
        float cov = clamp(rr - d + 0.5, 0.0, 1.0) * (r / rr);
        o = vec4(mix(vec3(1.0), vC, cov * mix(vW.z, vW.w, h)), 1.0);
    });

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* shared by the baked layers and the per-frame pass */
static const char *LIB = GLSL(
    float box(vec2 p, vec2 c, vec2 hs, float r) {
        vec2 d = abs(p - c) - hs + r;
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
    }
    /* soft shadow from a shape's distance: k darkness, b blur */
    float shade(float d, float k, float b) { return 1.0 - k * (1.0 - smoothstep(-b * 0.6, b, d)); }
    vec3 over(vec3 col, vec3 c, float d) { return mix(col, c, clamp(0.5 - d, 0.0, 1.0)); }
    uint hh(uvec2 v) {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * 1664525u; v.y += v.x * 1664525u;
        v ^= v >> 16u;
        v.x += v.y * 1664525u; v.y += v.x * 1664525u;
        v ^= v >> 16u;
        return v.x;
    }
    float h21(vec2 p) { return float(hh(uvec2(ivec2(floor(p)) + 65536)) & 0xffffffu) / 16777216.0; }
    float vn(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
    }
    uniform vec3 DARK, MID, HI;             /* the machine's faces */
    uniform vec3 METAL;                     /* and its fittings */
);

/* the sheet itself, baked once per size: paper colour with mottling,
 * fibres and tooth, its edges a touch shadowed; alpha keeps the grain for
 * the ink to take */
static const char *FS_PAPER = GLSL(
    uniform float uMm;          /* px per mm */
    uniform float uU;
    uniform vec2 uSize;
    uniform vec3 uPaperCol;
    out vec4 o;
    void main() {
        vec2 px = gl_FragCoord.xy, mm = px / uMm;
        float cloud = vn(mm / 11.0) * 0.5 + vn(mm / 4.5 + 17.0) * 0.3 + vn(mm / 1.6 + 31.0) * 0.2;
        float fib = 0.0;
        for (int i = 0; i < 4; i++) {
            float a = float(i) * 1.7 + 0.3;
            vec2 q = mat2(cos(a), sin(a), -sin(a), cos(a)) * mm;
            q.x += vn(q * 0.15 + float(i) * 9.0) * 6.0;        /* bend them */
            float n = vn(q * vec2(0.35, 3.2) + float(i) * 13.0);
            fib += pow(1.0 - abs(2.0 * n - 1.0), 14.0) * (vn(q * 0.2 + 5.0) > 0.55 ? 1.0 : 0.0) * (i % 2 == 0 ? -1.0 : 0.7);
        }
        float tooth = vn(px * 0.7) - 0.5 + (h21(px) - 0.5) * 0.6;
        float g = 0.30 * (cloud - 0.5) + 0.22 * fib + 0.10 * tooth;
        vec3 pap = uPaperCol * (1.0 + 0.085 * g);
        /* a sheet never lies quite flat: its edges catch a little shadow,
         * the top left corner a little more (row 0 is the top) */
        float e = min(min(px.x, px.y), min(uSize.x - px.x, uSize.y - px.y));
        pap *= mix(0.94, 1.0, smoothstep(0.0, 28.0 * uU, e));
        vec2 uv = px / uSize;
        pap *= 1.0 - 0.03 * smoothstep(0.35, 0.0, uv.y) * smoothstep(0.6, 0.0, uv.x);
        o = vec4(pap, g + 0.5);
    });

/* the bed, baked once per size: a cutting mat with a faint
 * 10 mm grid, darker towards the edges */
static const char *FS_BED = GLSL(
    uniform vec2 uRes;
    uniform float uU, uMm;
    uniform vec3 uBed, uBedLine;
    out vec4 o;
    void main() {
        vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        float u = uU;
        vec3 col = uBed;
        vec2 gq = abs(fract((p - uRes * 0.5) / (uMm * 10.0) + 0.5) - 0.5) * uMm * 10.0;
        float gl = 1.0 - clamp(min(gq.x, gq.y) - 0.3, 0.0, 1.0);
        vec2 gq5 = abs(fract((p - uRes * 0.5) / (uMm * 50.0) + 0.5) - 0.5) * uMm * 50.0;
        gl = max(gl * 0.5, 1.0 - clamp(min(gq5.x, gq5.y) - 0.5, 0.0, 1.0));
        col = mix(col, uBedLine, gl * 0.35);
        col *= 0.985 + 0.03 * h21(p);
        vec2 q = p / uRes - 0.5;
        col *= 1.0 - 0.35 * dot(q, q);
        o = vec4(col, 1.0);
    });

/* each frame: the bed, the sheet (paper and ink) and the moving machine */
static const char *FS_COMP = GLSL(
    uniform vec2 uRes;
    uniform vec4 uPaper;        /* x, y (top left, y down; whole px), w, h */
    uniform sampler2D uBg, uInk, uSheet;
    uniform vec2 uPen;          /* the pen tip, px, y down */
    uniform float uLift;        /* 0 down, 1 up, 2.5 out for a change */
    uniform vec3 uPenCol;
    uniform vec2 uRails;        /* x of the two rails */
    uniform float uU;           /* a size unit: min(W, H) / 1000 */
    out vec4 o;

    void main() {
        vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        float u = uU;
        vec3 col = texelFetch(uBg, ivec2(gl_FragCoord.xy), 0).rgb;

        /* the sheet, on its shadow */
        vec2 pc = uPaper.xy + uPaper.zw * 0.5, phs = uPaper.zw * 0.5;
        col *= shade(box(p, pc + vec2(2.0, 5.0) * u, phs, 2.0), 0.28, 14.0 * u);
        float dp = box(p, pc, phs, 1.5);
        if (dp < 1.0) {
            ivec2 ip = clamp(ivec2(p - uPaper.xy), ivec2(0), ivec2(uPaper.zw) - 1);
            vec4 sh = texelFetch(uSheet, ip, 0);
            vec3 ink = texelFetch(uInk, ip, 0).rgb;
            ink = 1.0 - (1.0 - ink) * (0.9 + 0.5 * (sh.a - 0.5));   /* the ink takes the paper's tooth */
            col = mix(col, sh.rgb * ink, clamp(0.5 - dp, 0.0, 1.0));
        }

        /* the rails, standing over the sheet as it slides past: a dark
         * extrusion with a slot, on its shadow */
        float rw = 10.0 * u, rx = p.x < uRes.x * 0.5 ? uRails.x : uRails.y;
        float dr = abs(p.x - rx) - rw;
        if (dr < 14.0 * u) {
            col *= shade(abs(p.x - rx - 2.7 * u) - rw, 0.25, 6.0 * u);
            vec3 c = mix(MID, HI, smoothstep(rw, -rw, p.x - rx) * 0.5);
            c = mix(c, DARK, 1.0 - smoothstep(1.2 * u, 2.4 * u, abs(p.x - rx)));
            col = over(col, c, dr);
        }

        /* the machine; most of the screen is clear of it */
        float by = uPen.y - 58.0 * u, bh = 15.0 * u;           /* beam centre, half height */
        if (p.y < by - bh - 12.0 * u || p.y > uPen.y + 50.0 * u) { o = vec4(col, 1.0); return; }
        float bx0 = uRails.x, bx1 = uRails.y;
        vec2 sv = vec2(9.0, 14.0) * u;                          /* light from the upper left */
        float near = abs(p.x - uPen.x) - 60.0 * u;              /* > 0: clear of the carriage */
        float cy0 = by - bh - 9.0 * u, cy1 = uPen.y + 20.0 * u;
        float dB = box(p - sv, vec2((bx0 + bx1) * 0.5, by), vec2((bx1 - bx0) * 0.5 + 18.0 * u, bh), 4.0 * u);
        float dC = near > 0.0 ? 1e4 : box(p - sv * 1.15, vec2(uPen.x, (cy0 + cy1) * 0.5), vec2(27.0 * u, (cy1 - cy0) * 0.5), 7.0 * u);
        col *= shade(min(dB, dC), 0.30, 16.0 * u);

        /* the beam, with end blocks on the rails */
        float d = box(p, vec2((bx0 + bx1) * 0.5, by), vec2((bx1 - bx0) * 0.5 + 6.0 * u, bh), 3.0 * u);
        if (d < 1.0) {
            float v = (p.y - by) / bh;                          /* -1 top .. 1 bottom */
            vec3 c = mix(HI, MID, smoothstep(-1.0, -0.5, v));
            c = mix(c, DARK, smoothstep(0.55, 1.0, v));
            c = mix(c, DARK * 0.8, 1.0 - smoothstep(1.0 * u, 2.2 * u, abs(p.y - by)));   /* the belt slot */
            col = over(col, c, d);
        }
        float de = box(p, vec2(rx, by), vec2(19.0 * u, bh + 9.0 * u), 5.0 * u);
        if (de < 1.0) {
            col = over(col, mix(MID, DARK, smoothstep(-bh, bh + 9.0 * u, p.y - by)), de);
            vec2 sc = vec2(rx, by + sign(p.y - by) * (bh + 1.0 * u));
            col = over(col, mix(METAL, METAL * 0.6, smoothstep(-3.0 * u, 3.0 * u, p.y - sc.y)), length(p - sc) - 3.2 * u);
        }
        if (near > 0.0) { o = vec4(col, 1.0); return; }

        /* the carriage */
        vec2 cc = vec2(uPen.x, (cy0 + cy1) * 0.5), ch = vec2(27.0 * u, (cy1 - cy0) * 0.5);
        vec2 psh = vec2(2.0, 3.0) * u * (1.0 + 2.2 * uLift);
        col *= shade(length(p - uPen - psh) - 13.0 * u, 0.35, (3.0 + 4.0 * uLift) * u);
        d = box(p, cc, ch, 7.0 * u);
        float v = (p.y - cy0) / (cy1 - cy0);
        vec3 c = mix(HI * 0.95, MID, smoothstep(0.0, 0.25, v));
        c = mix(c, DARK, smoothstep(0.75, 1.0, v));
        c = mix(c, DARK, smoothstep(-2.5 * u, 0.0, d) * 0.6);                            /* bevel */
        col = over(col, c, d);
        /* a brass plate across it */
        float dpl = box(p, vec2(uPen.x, by), vec2(21.0 * u, 4.0 * u), 2.0 * u);
        col = over(col, mix(METAL * 1.1, METAL * 0.65, smoothstep(by - 4.0 * u, by + 4.0 * u, p.y)), dpl);
        /* the pen in its clamp: seen from above, its cap in the ink's colour */
        float s = 1.0 + 0.08 * uLift;
        vec2 pp = uPen - vec2(1.0, 1.5) * u * uLift;
        float rr = length(p - pp);
        col = over(col, mix(METAL, METAL * 0.55, smoothstep(-16.0 * u, 16.0 * u, p.y - pp.y + (p.x - pp.x) * 0.5)), rr - 16.0 * s * u);
        col = over(col, DARK, rr - 12.5 * s * u);
        vec3 pcap = uPenCol * (0.75 + 0.45 * smoothstep(9.0 * u, -9.0 * u, (p.x - pp.x) + (p.y - pp.y)));
        col = over(col, pcap, rr - 10.5 * s * u);
        col = over(col, mix(pcap, vec3(1.0), 0.35), length(p - pp + vec2(3.5, 3.5) * u) - 2.5 * u);
        o = vec4(col, 1.0);
    });

static GLuint prog_seg, prog_paper, prog_bed, prog_comp, vao_seg, vbo_seg, vao_empty;
static GLuint tex_ink, fbo_ink, tex_sheet, fbo_sheet, tex_bg, fbo_bg;
static float bed_rgb[3], bedline_rgb[3];
static int ink_w, ink_h, lay_w, lay_h;
static float px_mm, pap_x, pap_y;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "plotter: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}
/* a fragment shader with the shared functions in front (kept for good) */
static const char *with_lib(const char *fs) {
    size_t n = strlen(LIB) + strlen(fs) + 32;
    char *src = malloc(n);
    snprintf(src, n, "#version 330 core\n%s\n%s", LIB, fs);
    return src;
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
static GLuint mk_target(GLuint *tex, GLenum fmt, int w, int h) {
    GLuint f;
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt == GL_R8 ? GL_RED : GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    return f;
}

static void gl_init(void) {
    prog_seg = mk_prog(VS_SEG, FS_SEG);
    prog_paper = mk_prog(VS_FULL, with_lib(FS_PAPER));
    prog_bed = mk_prog(VS_FULL, with_lib(FS_BED));
    prog_comp = mk_prog(VS_FULL, with_lib(FS_COMP));
    glGenVertexArrays(1, &vao_empty);
    glGenVertexArrays(1, &vao_seg);
    glGenBuffers(1, &vbo_seg);
    glBindVertexArray(vao_seg);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_seg);
    for (int i = 0; i < 3; i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, i == 2 ? 3 : 4, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void *)(sizeof(float) * 4 * i));
        glVertexAttribDivisor(i, 1);
    }
    hex(pal->paper, paper_rgb);
    hex(pal->bed, bed_rgb);
    hex(pal->bedline, bedline_rgb);                                    /* the mat's grid */
    glUseProgram(prog_comp);
    glUniform3fv(glGetUniformLocation(prog_comp, "DARK"), 1, pal->dark);
    glUniform3fv(glGetUniformLocation(prog_comp, "MID"), 1, pal->mid);
    glUniform3fv(glGetUniformLocation(prog_comp, "HI"), 1, pal->hi);
    glUniform3fv(glGetUniformLocation(prog_comp, "METAL"), 1, pal->metal);
    glUniform1i(glGetUniformLocation(prog_comp, "uBg"), 0);
    glUniform1i(glGetUniformLocation(prog_comp, "uInk"), 1);
    glUniform1i(glGetUniformLocation(prog_comp, "uSheet"), 2);
}

/* ---- laying ink ---- */
static float *ibuf;
static int in_, icap;
static int ink_cursor;                  /* segments up to this sample are down */

static void ink_flush(void) {
    if (!in_) return;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ink);
    glViewport(0, 0, ink_w, ink_h);
    glEnable(GL_BLEND);
    glBlendEquation(GL_MIN);
    glUseProgram(prog_seg);
    glUniform2f(glGetUniformLocation(prog_seg, "uSize"), (float)ink_w, (float)ink_h);
    glBindVertexArray(vao_seg);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_seg);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 11 * in_, ibuf, GL_STREAM_DRAW);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, in_);
    glDisable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    in_ = 0;
}

/* ink from sample a to b (f of the way there) */
static void ink_seg(const Smp *a, const Smp *b, float f) {
    if (a->z > 0.01f || b->z > 0.01f) return;
    if (in_ + 1 > 8192) ink_flush();
    GROW(ibuf, in_, icap, 11 * (in_ + 1));
    float *o = ibuf + 11 * in_++;
    float bx = a->ix + (b->ix - a->ix) * f, by = a->iy + (b->iy - a->iy) * f, bv = a->v + (b->v - a->v) * f;
    /* slow ink spreads and darkens: wide pools where the pen stands */
    float wa = a->w * (1 + 0.55f * expf(-a->v / 4)), wb = b->w * (1 + 0.55f * expf(-bv / 4));
    float da = a->k * (0.78f + 0.22f * expf(-a->v / 9)), db = b->k * (0.78f + 0.22f * expf(-bv / 9));
    o[0] = a->ix * px_mm; o[1] = a->iy * px_mm; o[2] = bx * px_mm; o[3] = by * px_mm;
    o[4] = 0.5f * wa * px_mm; o[5] = 0.5f * wb * px_mm; o[6] = da; o[7] = db;
    memcpy(o + 8, pen_T[b->pen], sizeof(float) * 3);
}

static void ink_clear(void) {
    ink_cursor = 0;
    if (!fbo_ink) return;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ink);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

/* lay all ink up to plan time tau */
static void ink_upto(float tau) {
    const Smp *s = plan.s;
    int i = ink_cursor;
    while (i + 1 < plan.n && s[i + 1].t <= tau) { ink_seg(&s[i], &s[i + 1], 1); i++; }
    ink_cursor = i;
    if (i + 1 < plan.n && s[i].t < tau) ink_seg(&s[i], &s[i + 1], (tau - s[i].t) / (s[i + 1].t - s[i].t));
    ink_flush();
}

static float rails_x(void) {
    float U = (float)(H < W ? H : W) / 1000;
    return fmaxf(pap_x * 0.45f, 16 * U);
}

/* the sheet's place on screen, and its textures at that size */
static void layout(void) {
    if (lay_w == W && lay_h == H) return;
    int first = !lay_w;
    lay_w = W; lay_h = H;
    float aspect = PW / PH;
    float ph = 0.86f * H, pw = ph * aspect;
    if (pw > 0.8f * W) { pw = 0.8f * W; ph = pw / aspect; }
    int nw = (int)lroundf(pw), nh = (int)lroundf(ph);
    pap_x = floorf((W - nw) * 0.5f);
    pap_y = floorf((H - nh) * 0.5f);
    float U = (float)(H < W ? H : W) / 1000;
    /* the bed: redone at every size */
    if (!first) { glDeleteTextures(1, &tex_bg); glDeleteFramebuffers(1, &fbo_bg); }
    fbo_bg = mk_target(&tex_bg, GL_RGBA8, W, H);
    glViewport(0, 0, W, H);
    glUseProgram(prog_bed);
    glUniform2f(glGetUniformLocation(prog_bed, "uRes"), (float)W, (float)H);
    glUniform1f(glGetUniformLocation(prog_bed, "uU"), U);
    glUniform1f(glGetUniformLocation(prog_bed, "uMm"), nw / PW);
    glUniform3fv(glGetUniformLocation(prog_bed, "uBed"), 1, bed_rgb);
    glUniform3fv(glGetUniformLocation(prog_bed, "uBedLine"), 1, bedline_rgb);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    if (!first && nw == ink_w && nh == ink_h) return;
    /* the sheet and its ink: only when its size changes */
    if (!first) {
        glDeleteTextures(1, &tex_ink); glDeleteFramebuffers(1, &fbo_ink);
        glDeleteTextures(1, &tex_sheet); glDeleteFramebuffers(1, &fbo_sheet);
    }
    ink_w = nw; ink_h = nh;
    px_mm = ink_w / PW;
    fbo_ink = mk_target(&tex_ink, GL_RGBA8, ink_w, ink_h);
    fbo_sheet = mk_target(&tex_sheet, GL_RGBA8, ink_w, ink_h);
    glViewport(0, 0, ink_w, ink_h);
    glUseProgram(prog_paper);
    glUniform1f(glGetUniformLocation(prog_paper, "uMm"), px_mm);
    glUniform1f(glGetUniformLocation(prog_paper, "uU"), U);
    glUniform2f(glGetUniformLocation(prog_paper, "uSize"), (float)ink_w, (float)ink_h);
    glUniform3fv(glGetUniformLocation(prog_paper, "uPaperCol"), 1, paper_rgb);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ink_clear();                        /* and replay what's been drawn so far */
}

/* ---- the timeline ---- */
static double t_cur;

static void advance(double t) {
    if (sheet_k < 0) { sheet_k = 0; sheet_T = 0; make_sheet(0); ink_clear(); }
    while (t >= sheet_T + sheet_cycle) {
        sheet_T += sheet_cycle;
        make_sheet(++sheet_k);
        ink_clear();
    }
    t_cur = t;
}

static float ease(float x) { x = clampf(x, 0, 1); return x * x * x * (x * (x * 6 - 15) + 10); }

static void render(GLuint target) {
    layout();
    double u = t_cur - sheet_T;
    float tau = (float)(u - T_SLIDE - T_SETTLE);
    ink_upto(tau < 0 ? 0 : tau);

    /* where the pen is */
    const Smp *s = plan.s;
    float px, py, lift;
    int pen;
    if (tau <= 0 || plan.n < 2) { px = s[0].x; py = s[0].y; lift = s[0].z; pen = s[0].pen; }
    else if (tau >= s[plan.n - 1].t) { px = s[plan.n - 1].x; py = s[plan.n - 1].y; lift = s[plan.n - 1].z; pen = s[plan.n - 1].pen; }
    else {
        int lo = 0, hi = plan.n - 1;
        while (hi - lo > 1) { int m = (lo + hi) / 2; if (s[m].t <= tau) lo = m; else hi = m; }
        float f = (tau - s[lo].t) / fmaxf(s[hi].t - s[lo].t, 1e-6f);
        px = s[lo].x + (s[hi].x - s[lo].x) * f;
        py = s[lo].y + (s[hi].y - s[lo].y) * f;
        lift = s[lo].z + (s[hi].z - s[lo].z) * f;
        pen = s[lo].pen;
    }
    /* the sheet slides in from the right and out to the left */
    float dx = 0, U = (float)(H < W ? H : W) / 1000;
    double out_at = sheet_cycle - T_SLIDE;
    if (u < T_SLIDE) dx = (1 - ease((float)(u / T_SLIDE))) * (W - pap_x + 30 * U);
    else if (u > out_at) dx = -ease((float)((u - out_at) / T_SLIDE)) * (pap_x + ink_w + 30 * U);

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_comp);
    GLuint texs[3] = {tex_bg, tex_ink, tex_sheet};
    for (int i = 2; i >= 0; i--) { glActiveTexture(GL_TEXTURE0 + i); glBindTexture(GL_TEXTURE_2D, texs[i]); }
    glUniform2f(glGetUniformLocation(prog_comp, "uRes"), (float)W, (float)H);
    glUniform4f(glGetUniformLocation(prog_comp, "uPaper"), pap_x + roundf(dx), pap_y, (float)ink_w, (float)ink_h);
    glUniform2f(glGetUniformLocation(prog_comp, "uPen"), pap_x + px * px_mm, pap_y + py * px_mm);
    glUniform1f(glGetUniformLocation(prog_comp, "uLift"), lift);
    glUniform3fv(glGetUniformLocation(prog_comp, "uPenCol"), 1, pen_rgb[pen]);
    glUniform2f(glGetUniformLocation(prog_comp, "uRails"), rails_x(), W - rails_x());
    glUniform1f(glGetUniformLocation(prog_comp, "uU"), U);
    glBindVertexArray(vao_empty);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "plotter");
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
        "usage: plotter [--window | --root | --layer] [--seed N] [--speed X] [--kind NAME]\n"
        "               [--palette NAME]\n"
        "               [--shot FILE.ppm] [--bench N] [--size WxH] [--at SECONDS]\n"
        "kinds: flow topo ridges harmonograph spirograph rings truchet\n"
        "palettes: sepia blue technical colour (default: picked from the seed)\n");
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
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-kind") && i + 1 < argc) {
            const char *k = argv[++i];
            for (int j = 0; j < NKIND; j++) if (!strcmp(k, KIND_NAME[j])) opt_kind = j;
            if (opt_kind < 0) usage();
        }
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *k = argv[++i];
            for (int j = 0; j < NPAL; j++) if (!strcmp(k, PALS[j].name)) opt_pal = j;
            if (opt_pal < 0) usage();
        }
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) { opt_at = strtod(argv[++i], NULL); opt_at_set = 1; }
        else usage();
    }
    if (opt_speed <= 0) usage();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    pal = &PALS[opt_pal >= 0 ? opt_pal : (int)(mix32(opt_seed ^ 0x51ed270bu) % NPAL)];
    gethostname(host, sizeof host - 1);
    if (getenv("PLOTTER_HOST")) snprintf(host, sizeof host, "%s", getenv("PLOTTER_HOST"));   /* for testing the script */
    host[strcspn(host, ".")] = 0;
    for (char *c = host; *c; c++) if (!glyph(*c)) *c = '-';
    wall0 = time(NULL);
    int offscreen = opt_shot || opt_bench;

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
        else if (!offscreen) { W = DisplayWidth(dpy, scr); H = DisplayHeight(dpy, scr); }
        if (offscreen) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Plotter");
        XClassHint chint = {"plotter", "Plotter"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    /* the sheet stands the way the screen does */
    if (H > W) { PW = 297; PH = 420; }
    gl_init();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        layout();
        advance(opt_at);
        render(fbo);
        glFinish();
        if (opt_shot) { save_ppm(opt_shot); return 0; }
        double t0 = now(), worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            advance(opt_at + (i + 1) / 60.0 * opt_speed);
            render(fbo);
            glFinish();
            worst = fmax(worst, now() - f0);
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d: %.3f ms/frame avg, %.3f worst (%s)\n", W, H, per * 1e3, worst * 1e3,
               (const char *)glGetString(GL_RENDERER));
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
        advance((opt_at_set ? opt_at : 0) + (t - t0) * opt_speed);
        render(0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
