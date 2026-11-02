/*
 * billiards — mathematical billiards: a point ball bouncing with perfect
 * specular reflection inside a table, its path drawn as fine luminous lines
 * that build up, over a minute or two, into the table's characteristic
 * pattern, and slowly fade.
 *
 * The acts, each with a small title card naming the table and its property:
 *
 *   circle      integrable: each orbit keeps its angle, its chords wrap a
 *               concentric caustic circle
 *   ellipse     integrable: every chord of an orbit is tangent to the same
 *               confocal conic, an ellipse or a hyperbola (the foci marked)
 *   square      a rational slope closes up, an irrational one is dense
 *   stadium     Bunimovich: chaotic and ergodic, one orbit fills it uniformly
 *   Sinai       a square with a round scatterer: dispersing, chaotic
 *   cardioid    chaotic
 *   limaçon     r = 1 + 0.3 cos θ: islands in a chaotic sea
 *   mushroom    Bunimovich's mushroom: regular in the cap, chaotic in the stem
 *   lemon       two circular arcs: a stable island and a chaotic sea
 *   triangle    equilateral: Fagnano's period-3 orbit and its neighbours, with
 *               the unfolding (reflect the table, not the ball: the orbit is a
 *               straight line through the triangular lattice)
 *   ensembles   thousands of balls launched from almost the same state, a
 *               glowing swarm: the circle keeps them together, the stadium
 *               tears them apart exponentially, the mushroom does both
 *
 * Beside each table, the Birkhoff phase portrait: arc length s along the
 * boundary against cos θ of the outgoing angle (the coordinates in which the
 * bounce map preserves area), built up a dot per bounce: invariant curves,
 * islands and chaotic seas.
 *
 * The geometry is exact, in double precision: rays are intersected with the
 * pieces of the boundary in closed form (segments, circular arcs, the ellipse;
 * the root on the curve the ball is leaving is factored out analytically), and
 * with the limaçon / cardioid quartic (x²+y²−εx)² = x²+y² by isolating real
 * roots through the critical points and polishing by Newton. A ball's position
 * is never integrated: it is the last bounce point plus the direction times
 * the path length since, and the path length is a closed-form function of the
 * act's clock. So the orbits don't depend on the frame rate, and every
 * instance of a wall computes bit-identical orbits. --check prints the maths
 * checks (boundary residuals, conserved quantities, reversibility, Lyapunov
 * exponents, uniform filling, the mean-free-path law).
 *
 * The lines are drawn additively into a half-float accumulation buffer (plus a
 * quarter-size one for the glow) that decays gently, on a fixed tick of the
 * act's clock, so a pattern builds and slowly fades; joining late (a restart, a
 * new output) replays the act so far with the decay folded into each line's
 * weight. Palettes: spectral, gold (gold leaf on black, picked sparingly), ice,
 * paper (ink on cream), neon; --palette NAME, else from the seed.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index. The
 * outputs, left to right, show a regular, a mixed and a chaotic table in
 * sync, each with its phase portrait (circle | mushroom | stadium, ellipse |
 * limaçon | cardioid, triangle | lemon | Sinai, and the three swarms). The
 * output under the middle of the layout (the "front" one) carries the time.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), --shot
 * FILE.ppm [--at SECONDS] [--size WxH] renders one frame offscreen, --bench N
 * times N frames. --scene NAME holds one act.
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

#define TAU (2 * M_PI)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check;
static float opt_fps = 60;
static const char *opt_font = "EB Garamond 12";
static uint32_t opt_seed;
static const char *opt_shot, *opt_scene, *opt_palette;
static double opt_at = 40;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the show started at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
static int is_front = 1, slot = -1;    /* slot: which of a trio this output shows, -1 alone */

static void die(const char *what) { fprintf(stderr, "billiards: %s\n", what); exit(1); }

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

/* splitmix64: every random choice is a pure function of the seed and the act */
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static double urand(uint64_t *s) { return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0); }

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static double smooth(double a, double b, double x) {
    double t = (x - a) / (b - a);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return t * t * (3 - 2 * t);
}

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

/* ==================================================================== */
/* ---- geometry: a table is a closed chain of pieces, interior on the left
 * (the outer boundary counter-clockwise, a scatterer clockwise) ---- */

enum { P_SEG, P_ARC, P_ELL, P_LIM };
typedef struct {
    int type;
    double ax, ay, bx, by;             /* SEG: from a to b */
    double cx, cy, r, a0, da;          /* ARC: centre, radius, start angle, signed sweep; ELL, LIM: centre */
    double ea, eb;                     /* ELL: semi-axes */
    double eps;                        /* LIM: r = 1 + eps cos θ about the centre (eps = 1: the cardioid) */
    double len, s0;                    /* length, and arc length where it starts */
    double *tab;                       /* ELL, LIM: arc length at parameter 2π i / NTAB */
} Piece;
#define NTAB 8192
#define MAXPC 12
typedef struct {
    Piece pc[MAXPC];
    int n;
    double L, area, mfp, size;         /* perimeter, area, mean free path πA/L, sqrt(area) */
    double x0, y0, x1, y1;             /* bounding box */
} Table;

static Piece *tb_add(Table *T, int type) {
    if (T->n == MAXPC) die("too many pieces");
    Piece *P = &T->pc[T->n++];
    memset(P, 0, sizeof *P);
    P->type = type;
    return P;
}
static void tb_seg(Table *T, double ax, double ay, double bx, double by) {
    Piece *P = tb_add(T, P_SEG);
    P->ax = ax; P->ay = ay; P->bx = bx; P->by = by;
}
static void tb_arc(Table *T, double cx, double cy, double r, double a0, double da) {
    Piece *P = tb_add(T, P_ARC);
    P->cx = cx; P->cy = cy; P->r = r; P->a0 = a0; P->da = da;
}

/* a point of the piece at u in [0, 1] (the curve's own parameter) */
static void pc_point(const Piece *P, double u, double *x, double *y) {
    switch (P->type) {
    case P_SEG: *x = P->ax + u * (P->bx - P->ax); *y = P->ay + u * (P->by - P->ay); break;
    case P_ARC: { double f = P->a0 + u * P->da; *x = P->cx + P->r * cos(f); *y = P->cy + P->r * sin(f); } break;
    case P_ELL: { double t = TAU * u; *x = P->cx + P->ea * cos(t); *y = P->cy + P->eb * sin(t); } break;
    default: { double t = TAU * u, r = 1 + P->eps * cos(t); *x = P->cx + r * cos(t); *y = P->cy + r * sin(t); } break;
    }
}

/* the limaçon's implicit form f = (|q|² − ε q_x)² − |q|², negative inside */
static double lim_f(const Piece *P, double x, double y, double *gx, double *gy) {
    x -= P->cx; y -= P->cy;
    double q2 = x * x + y * y, h = q2 - P->eps * x;
    if (gx) { *gx = 2 * h * (2 * x - P->eps) - 2 * x; *gy = 2 * h * 2 * y - 2 * y; }
    return h * h - q2;
}

/* inward unit normal and the tangent (direction of travel, interior on the left) at q */
static void pc_frame(const Piece *P, double x, double y, double *nx, double *ny, double *tx, double *ty) {
    double a, b, l;
    switch (P->type) {
    case P_SEG: a = P->bx - P->ax; b = P->by - P->ay; l = hypot(a, b); *tx = a / l; *ty = b / l; break;
    case P_ARC: { double sg = P->da > 0 ? 1 : -1; a = x - P->cx; b = y - P->cy; l = hypot(a, b);
                  *tx = -sg * b / l; *ty = sg * a / l; } break;
    case P_ELL: a = -(x - P->cx) / (P->ea * P->ea); b = -(y - P->cy) / (P->eb * P->eb); l = hypot(a, b);
                *tx = b / l; *ty = -a / l; break;
    default: lim_f(P, x, y, &a, &b); l = hypot(a, b); a = -a / l; b = -b / l; *tx = b; *ty = -a; break;
    }
    *nx = -*ty; *ny = *tx;
}

static double tab_at(const double *tab, double ang) {
    ang = fmod(ang, TAU);
    if (ang < 0) ang += TAU;
    double f = ang / TAU * NTAB;
    int i = (int)f;
    if (i >= NTAB) i = NTAB - 1;
    return tab[i] + (f - i) * (tab[i + 1] - tab[i]);
}

/* arc length from the piece's start to q (on it) */
static double pc_local(const Piece *P, double x, double y) {
    double v;
    switch (P->type) {
    case P_SEG: v = hypot(x - P->ax, y - P->ay); break;
    case P_ARC: {
        double f = atan2(y - P->cy, x - P->cx);
        v = P->da > 0 ? f - P->a0 : P->a0 - f;
        v = fmod(v, TAU); if (v < 0) v += TAU;
        if (v > fabs(P->da) + 1e-9) v = v > (TAU + fabs(P->da)) / 2 ? 0 : fabs(P->da);
        v *= P->r;
    } break;
    case P_ELL: v = tab_at(P->tab, atan2((y - P->cy) / P->eb, (x - P->cx) / P->ea)); break;
    default: v = tab_at(P->tab, atan2(y - P->cy, x - P->cx)); break;
    }
    return v < 0 ? 0 : v > P->len ? P->len : v;
}

/* distance from q to the piece's curve (a numerical-drift check) */
static double pc_resid(const Piece *P, double x, double y) {
    double gx, gy, f;
    switch (P->type) {
    case P_SEG: { double tx = P->bx - P->ax, ty = P->by - P->ay, l = hypot(tx, ty);
                  return fabs((x - P->ax) * ty - (y - P->ay) * tx) / l; }
    case P_ARC: return fabs(hypot(x - P->cx, y - P->cy) - P->r);
    case P_ELL: { double u = (x - P->cx) / P->ea, v = (y - P->cy) / P->eb;
                  f = u * u + v * v - 1; gx = 2 * u / P->ea; gy = 2 * v / P->eb; return fabs(f) / hypot(gx, gy); }
    default: f = lim_f(P, x, y, &gx, &gy); return fabs(f) / hypot(gx, gy);
    }
}

static int arc_has(const Piece *P, double x, double y) {
    if (fabs(P->da) >= TAU - 1e-12) return 1;
    double lo = P->da > 0 ? P->a0 : P->a0 + P->da;
    double u = fmod(atan2(y - P->cy, x - P->cx) - lo, TAU);
    if (u < 0) u += TAU;
    return u <= fabs(P->da) + 1e-12 || u >= TAU - 1e-12;
}

/* real roots of c[0] + c[1] t + ... + c[n] t^n in (lo, hi), ascending: the
 * critical points (roots of the derivative, recursively) split the interval
 * into monotone pieces, and each piece with a sign change is bisected */
static double peval(const double *c, int n, double t) {
    double v = c[n];
    for (int i = n - 1; i >= 0; i--) v = v * t + c[i];
    return v;
}
static int proots(const double *c, int n, double lo, double hi, double *out) {
    while (n > 0 && c[n] == 0) n--;
    if (n == 0) return 0;
    if (n == 1) {
        double t = -c[0] / c[1];
        if (t > lo && t < hi) { out[0] = t; return 1; }
        return 0;
    }
    double dc[4], cr[4], pts[6];
    for (int i = 1; i <= n; i++) dc[i - 1] = i * c[i];
    int nc = proots(dc, n - 1, lo, hi, cr), np = 0, nr = 0;
    pts[np++] = lo;
    for (int i = 0; i < nc; i++) pts[np++] = cr[i];
    pts[np++] = hi;
    for (int i = 0; i + 1 < np; i++) {
        double a = pts[i], b = pts[i + 1], fa = peval(c, n, a), fb = peval(c, n, b);
        if ((fa > 0) == (fb > 0)) continue;
        for (int k = 0; k < 200; k++) {
            double m = 0.5 * (a + b);
            if (m <= a || m >= b) break;
            double fm = peval(c, n, m);
            if ((fm > 0) == (fa > 0)) { a = m; fa = fm; } else b = m;
        }
        out[nr++] = 0.5 * (a + b);
    }
    return nr;
}

/* the ray from (px, py) along unit (dx, dy): the nearest piece it meets beyond
 * tmin (the ball is leaving piece `from`, or -1 from inside) */
static int tb_hit(const Table *T, int from, double px, double py, double dx, double dy, double *tt) {
    const double tmin = 1e-9 * T->size, tmax = 4 * hypot(T->x1 - T->x0, T->y1 - T->y0);
    double best = 1e300;
    int bi = -1;
#define CAND(t_) do { double t__ = (t_); if (t__ > tmin && t__ < best) { best = t__; bi = i; } } while (0)
    for (int i = 0; i < T->n; i++) {
        const Piece *P = &T->pc[i];
        if (P->type == P_SEG) {
            if (i == from) continue;
            double ex = P->bx - P->ax, ey = P->by - P->ay, den = dx * ey - dy * ex;
            if (fabs(den) < 1e-300) continue;
            double wx = P->ax - px, wy = P->ay - py;
            double t = (wx * ey - wy * ex) / den, u = (wx * dy - wy * dx) / den;
            if (u >= -1e-12 && u <= 1 + 1e-12) CAND(t);
        } else if (P->type == P_ARC) {
            double ox = px - P->cx, oy = py - P->cy, b = ox * dx + oy * dy;
            if (i == from) {
                /* leaving this circle: t = 0 is one root, -2 o·d the other;
                 * a convex scatterer can't be met again */
                if (P->da < 0) continue;
                double t = -2 * b;
                if (arc_has(P, px + t * dx, py + t * dy)) CAND(t);
                continue;
            }
            double c = ox * ox + oy * oy - P->r * P->r, disc = b * b - c;
            if (disc < 0) continue;
            double sq = sqrt(disc), t1, t2;
            if (b > 0) { t1 = -b - sq; t2 = c / t1; } else { t2 = -b + sq; t1 = t2 != 0 ? c / t2 : 0; }
            if (arc_has(P, px + t1 * dx, py + t1 * dy)) CAND(t1);
            if (arc_has(P, px + t2 * dx, py + t2 * dy)) CAND(t2);
        } else if (P->type == P_ELL) {
            double ia = 1 / (P->ea * P->ea), ib = 1 / (P->eb * P->eb);
            double ox = px - P->cx, oy = py - P->cy;
            double A = dx * dx * ia + dy * dy * ib, B = 2 * (ox * dx * ia + oy * dy * ib);
            if (i == from) { CAND(-B / A); continue; }
            double C = ox * ox * ia + oy * oy * ib - 1, disc = B * B - 4 * A * C;
            if (disc < 0) continue;
            double q = -0.5 * (B + (B > 0 ? sqrt(disc) : -sqrt(disc)));
            if (q != 0) { CAND(q / A); CAND(C / q); }
        } else {
            /* (|q|² − ε q_x)² − |q|² along q = o + t d: a quartic in t, monic
             * since |d| = 1; leaving the curve, its constant term is the t = 0
             * root, divided out to leave a cubic */
            double ox = px - P->cx, oy = py - P->cy, e = P->eps;
            double b = ox * dx + oy * dy, oo = ox * ox + oy * oy;
            double h1 = 2 * b - e * dx, h0 = oo - e * ox;
            double c[5] = {h0 * h0 - oo, 2 * h1 * h0 - 2 * b, h1 * h1 + 2 * h0 - 1, 2 * h1, 1};
            double r[4];
            int nr = i == from ? proots(c + 1, 3, tmin, tmax, r) : proots(c, 4, tmin, tmax, r);
            for (int k = 0; k < nr; k++) {
                double t = r[k];
                for (int it = 0; it < 3; it++) {            /* polish on the geometry itself */
                    double qx = ox + t * dx, qy = oy + t * dy, q2 = qx * qx + qy * qy, h = q2 - e * qx;
                    double f = h * h - q2, fp = 2 * h * (2 * t + h1) - (2 * t + 2 * b);
                    if (fp == 0) break;
                    double tn = t - f / fp;
                    if (fabs(tn - t) > 1e-6 * T->size) break;
                    t = tn;
                }
                double qx = ox + t * dx, qy = oy + t * dy;
                /* below the cusp the origin is a touching (double) root, not the boundary */
                if (e < 1 && qx * qx + qy * qy < 1e-12) continue;
                CAND(t);
                break;
            }
        }
    }
#undef CAND
    *tt = best;
    return bi;
}

static double pc_len_tab(Piece *P) {
    P->tab = malloc((NTAB + 1) * sizeof *P->tab);
    P->tab[0] = 0;
    double h = TAU / NTAB;
    for (int i = 0; i < NTAB; i++) {
        /* Simpson on |dγ/dt| over each step */
        double s = 0;
        for (int k = 0; k <= 2; k++) {
            double t = (i + k * 0.5) * h, v;
            if (P->type == P_ELL) v = hypot(P->ea * sin(t), P->eb * cos(t));
            else { double r = 1 + P->eps * cos(t), rp = -P->eps * sin(t); v = hypot(r, rp); }
            s += v * (k == 1 ? 4 : 1);
        }
        P->tab[i + 1] = P->tab[i] + s * h / 6;
    }
    return P->tab[NTAB];
}

static void tb_finish(Table *T) {
    double s = 0, area = 0;
    T->x0 = T->y0 = 1e9; T->x1 = T->y1 = -1e9;
    for (int i = 0; i < T->n; i++) {
        Piece *P = &T->pc[i];
        P->s0 = s;
        P->len = P->type == P_SEG ? hypot(P->bx - P->ax, P->by - P->ay)
               : P->type == P_ARC ? P->r * fabs(P->da) : pc_len_tab(P);
        s += P->len;
        double px = 0, py = 0;
        for (int k = 0; k <= 2048; k++) {
            double x, y;
            pc_point(P, k / 2048.0, &x, &y);
            if (k) area += px * y - x * py;
            px = x; py = y;
            if (x < T->x0) T->x0 = x;
            if (x > T->x1) T->x1 = x;
            if (y < T->y0) T->y0 = y;
            if (y > T->y1) T->y1 = y;
        }
    }
    T->L = s;
    T->area = area / 2;
    T->size = sqrt(T->area);
    T->mfp = M_PI * T->area / T->L;
}

/* ---- the tables ---- */
enum { TB_CIRCLE, TB_ELLIPSE, TB_SQUARE, TB_STADIUM, TB_SINAI, TB_CARDIOID, TB_LIMACON, TB_MUSHROOM, TB_LEMON,
       TB_TRIANGLE, NTB };
static const char *tb_names[NTB] = {"circle", "ellipse", "square", "stadium", "sinai", "cardioid", "limacon",
                                    "mushroom", "lemon", "triangle"};
#define ELL_A 1.35
#define ELL_B 0.85
#define SINAI_R 0.42
#define LIMACON_EPS 0.3
#define MUSH_A 0.4                     /* the stem's half-width */
#define MUSH_H 1.1                     /* and its length */
#define LEMON_D 1.1                    /* distance between the unit circles' centres */
#define TRI_S 2.4

static void table_build(Table *T, int id) {
    memset(T, 0, sizeof *T);
    switch (id) {
    case TB_CIRCLE: tb_arc(T, 0, 0, 1, 0, TAU); break;
    case TB_ELLIPSE: { Piece *P = tb_add(T, P_ELL); P->ea = ELL_A; P->eb = ELL_B; } break;
    case TB_SQUARE:
        tb_seg(T, -1, -1, 1, -1); tb_seg(T, 1, -1, 1, 1); tb_seg(T, 1, 1, -1, 1); tb_seg(T, -1, 1, -1, -1);
        break;
    case TB_STADIUM:
        tb_seg(T, -1, -1, 1, -1); tb_arc(T, 1, 0, 1, -M_PI / 2, M_PI);
        tb_seg(T, 1, 1, -1, 1); tb_arc(T, -1, 0, 1, M_PI / 2, M_PI);
        break;
    case TB_SINAI:
        tb_seg(T, -1, -1, 1, -1); tb_seg(T, 1, -1, 1, 1); tb_seg(T, 1, 1, -1, 1); tb_seg(T, -1, 1, -1, -1);
        tb_arc(T, 0, 0, SINAI_R, 0, -TAU);
        break;
    case TB_CARDIOID: { Piece *P = tb_add(T, P_LIM); P->eps = 1; P->cx = -0.75; } break;
    case TB_LIMACON: { Piece *P = tb_add(T, P_LIM); P->eps = LIMACON_EPS; P->cx = -0.1; } break;
    case TB_MUSHROOM: {
        double a = MUSH_A, h = MUSH_H, y = 0.45;            /* lifted so the whole is centred */
        tb_arc(T, 0, y, 1, 0, M_PI);
        tb_seg(T, -1, y, -a, y); tb_seg(T, -a, y, -a, y - h); tb_seg(T, -a, y - h, a, y - h);
        tb_seg(T, a, y - h, a, y); tb_seg(T, a, y, 1, y);
    } break;
    case TB_LEMON: {
        double c = LEMON_D / 2, w = sqrt(1 - c * c), a0 = atan2(c, w);
        tb_arc(T, 0, -c, 1, a0, M_PI - 2 * a0);
        tb_arc(T, 0, c, 1, -M_PI + a0, M_PI - 2 * a0);
    } break;
    case TB_TRIANGLE: {
        double s = TRI_S, h = s * sqrt(3) / 2;
        tb_seg(T, -s / 2, -h / 3, s / 2, -h / 3); tb_seg(T, s / 2, -h / 3, 0, 2 * h / 3); tb_seg(T, 0, 2 * h / 3, -s / 2, -h / 3);
    } break;
    }
    tb_finish(T);
}

/* ---- balls: a chord at a time ---- */
#define NH 8
typedef struct {
    double px, py, dx, dy;             /* the current chord's start (the last bounce) and direction */
    double L0, len;                    /* path length at its start, its length */
    double Ld;                         /* drawn up to this path length */
    int from, to;                      /* the piece left and the piece the chord ends on */
    float hx[NH], hy[NH];              /* recent bounce points (hx[(hh - 1) % NH] newest) */
    double hl[NH];
    int hh;
    float u;                           /* colour, 0..1 along the palette */
    float ps, pp;                      /* phase point of the last bounce */
    int nb;                            /* bounces so far */
} Ball;
static long n_escape;                  /* rays that found no wall (should stay 0) */

static void ball_chord(Ball *b, const Table *T) {
    double t;
    int to = tb_hit(T, b->from, b->px, b->py, b->dx, b->dy, &t);
    if (to < 0) {
        /* rounding put the ball outside: turn it back, retrace */
        n_escape++;
        b->dx = -b->dx; b->dy = -b->dy;
        to = tb_hit(T, -1, b->px, b->py, b->dx, b->dy, &t);
        if (to < 0) { b->px = (T->x0 + T->x1) / 2; b->py = (T->y0 + T->y1) / 2; to = tb_hit(T, -1, b->px, b->py, b->dx, b->dy, &t); }
        if (to < 0) die("lost a ball");
    }
    b->to = to;
    b->len = t;
}

static void ball_push(Ball *b) {
    int k = b->hh++ % NH;
    b->hx[k] = (float)b->px; b->hy[k] = (float)b->py; b->hl[k] = b->L0;
}

static void ball_in(Ball *b, const Table *T, double x, double y, double ang, double L0) {
    memset(b, 0, sizeof *b);
    b->px = x; b->py = y; b->dx = cos(ang); b->dy = sin(ang); b->from = -1; b->L0 = b->Ld = L0;
    ball_push(b);
    ball_chord(b, T);
    b->ps = -1;
}

/* from the boundary: piece k at parameter u, at angle ang (0..π) to the tangent */
static void ball_on(Ball *b, const Table *T, int k, double u, double ang) {
    double x, y, nx, ny, tx, ty;
    pc_point(&T->pc[k], u, &x, &y);
    pc_frame(&T->pc[k], x, y, &nx, &ny, &tx, &ty);
    memset(b, 0, sizeof *b);
    b->px = x; b->py = y; b->from = k;
    b->dx = cos(ang) * tx + sin(ang) * nx; b->dy = cos(ang) * ty + sin(ang) * ny;
    ball_push(b);
    ball_chord(b, T);
    b->ps = -1;
}

/* bounce at the end of the chord; the phase point (s/L, cos θ) of the outgoing ray */
static double drift_d;                 /* largest | |d| − 1 | seen before renormalising */
static void ball_bounce(Ball *b, const Table *T) {
    double x = b->px + b->len * b->dx, y = b->py + b->len * b->dy, nx, ny, tx, ty;
    const Piece *P = &T->pc[b->to];
    /* on to the curve exactly, so rounding can't accumulate off it */
    if (P->type == P_ARC) {
        double ox = x - P->cx, oy = y - P->cy, l = hypot(ox, oy);
        x = P->cx + ox * P->r / l; y = P->cy + oy * P->r / l;
    } else if (P->type == P_ELL) {
        double ox = x - P->cx, oy = y - P->cy, f = sqrt(ox * ox / (P->ea * P->ea) + oy * oy / (P->eb * P->eb));
        x = P->cx + ox / f; y = P->cy + oy / f;
    } else if (P->type == P_LIM) {
        double gx, gy, f = lim_f(P, x, y, &gx, &gy), g2 = gx * gx + gy * gy;
        if (g2 > 1e-20) { x -= f * gx / g2; y -= f * gy / g2; }
    }
    pc_frame(P, x, y, &nx, &ny, &tx, &ty);
    double dn = b->dx * nx + b->dy * ny;
    double dx = b->dx - 2 * dn * nx, dy = b->dy - 2 * dn * ny, l = hypot(dx, dy);
    if (fabs(l - 1) > drift_d) drift_d = fabs(l - 1);
    b->dx = dx / l; b->dy = dy / l;
    b->ps = (float)((P->s0 + pc_local(P, x, y)) / T->L);
    b->pp = (float)(b->dx * tx + b->dy * ty);
    b->L0 += b->len;
    b->px = x; b->py = y; b->from = b->to;
    b->nb++;
    ball_push(b);
    ball_chord(b, T);
}

/* ==================================================================== */
/* ---- --check: the maths, measured ---- */

static void run_bounces(Ball *b, const Table *T, long n) { for (long i = 0; i < n; i++) ball_bounce(b, T); }

/* the ellipse's conserved quantity: the parameter λ of the confocal conic
 * x²/(a²−λ) + y²/(b²−λ) = 1 all of the orbit's chords are tangent to; an
 * ellipse for 0 < λ < b², a hyperbola for b² < λ < a² */
static double ell_lambda(const Ball *b) {
    double c = b->px * b->dy - b->py * b->dx;
    return ELL_A * ELL_A * b->dy * b->dy + ELL_B * ELL_B * b->dx * b->dx - c * c;
}

/* n bounces out, turn round in the middle of the chord, n back: how far
 * from home */
static double reversal(int tb, long n, double u, double ang) {
    Table T;
    table_build(&T, tb);
    Ball b;
    ball_on(&b, &T, 0, u, ang);
    double x0 = b.px, y0 = b.py;
    run_bounces(&b, &T, n);
    double mx = b.px + 0.5 * b.len * b.dx, my = b.py + 0.5 * b.len * b.dy;
    ball_in(&b, &T, mx, my, atan2(-b.dy, -b.dx), 0);
    run_bounces(&b, &T, n + 1);
    return hypot(b.px - x0, b.py - y0);
}

static void check(void) {
    printf("billiards --check: the maths, in double precision\n\n");
    printf("1. tables: perimeter L, area A, mean free path πA/L, and 10^6 bounces of one orbit:\n"
           "   the largest distance of a bounce point from its wall, the largest | |d|−1 | before\n"
           "   renormalising, rays that found no wall, the mean chord (Santaló: = πA/L for an\n"
           "   ergodic orbit), and the time per bounce.\n");
    for (int id = 0; id < NTB; id++) {
        Table T;
        table_build(&T, id);
        Ball b;
        ball_on(&b, &T, 0, 0.3141, 1.0);
        double res = 0, sum = 0;
        long N = 1000000, esc0 = n_escape;
        drift_d = 0;
        double t0 = now();
        for (long i = 0; i < N; i++) {
            sum += b.len;
            ball_bounce(&b, &T);
            double r = pc_resid(&T.pc[b.from], b.px, b.py);
            if (r > res) res = r;
        }
        double dt = now() - t0;
        printf("   %-9s L %7.4f  A %7.4f  πA/L %.4f | resid %.1e  |d| %.1e  lost %ld  chord %.4f  %4.0f ns\n",
               tb_names[id], T.L, T.area, T.mfp, res, drift_d, n_escape - esc0, sum / N, dt / N * 1e9);
    }

    printf("\n2. conserved quantities (integrable tables), drift over 10^6 bounces:\n");
    {
        Table T;
        table_build(&T, TB_CIRCLE);
        Ball b;
        ball_on(&b, &T, 0, 0.1, 0.4021 * M_PI);
        run_bounces(&b, &T, 1);
        double p0 = b.pp, dmax = 0;
        for (long i = 0; i < 1000000; i++) { ball_bounce(&b, &T); if (fabs(b.pp - p0) > dmax) dmax = fabs(b.pp - p0); }
        printf("   circle: cos θ (the caustic radius) %.6f, max change %.1e (float phase coords)\n", p0, dmax);
    }
    {
        Table T;
        table_build(&T, TB_ELLIPSE);
        double angs[2] = {0.35, 1.45};
        for (int k = 0; k < 2; k++) {
            Ball b;
            ball_on(&b, &T, 0, 0.13, angs[k]);
            double l0 = ell_lambda(&b), dmax = 0;
            for (long i = 0; i < 1000000; i++) {
                ball_bounce(&b, &T);
                double d = fabs(ell_lambda(&b) - l0);
                if (d > dmax) dmax = d;
            }
            printf("   ellipse: caustic λ = %.6f (%s; b² = %.4f, a² = %.4f), max drift %.1e\n", l0,
                   l0 < ELL_B * ELL_B ? "a confocal ellipse" : "a confocal hyperbola", ELL_B * ELL_B, ELL_A * ELL_A, dmax);
        }
    }
    {
        Table T;
        table_build(&T, TB_MUSHROOM);
        Ball b;
        ball_on(&b, &T, 0, 0.3, acos(0.6));
        double ymin = 1e9;
        for (long i = 0; i < 1000000; i++) { ball_bounce(&b, &T); if (b.py < ymin) ymin = b.py; }
        printf("   mushroom: a cap orbit with caustic radius 0.6 > stem half-width %.2f: lowest bounce y = %.6f (cap floor 0.45)\n",
               MUSH_A, ymin);
    }

    printf("\n3. reversibility: n bounces out, reverse, n back; distance from the start.\n"
           "   Linear growth of the error in the integrable tables, exponential in the chaotic\n"
           "   ones: there the computed orbit is only a shadow of a true one after ~30 bounces.\n");
    printf("   %-9s", "n");
    long ns[6] = {10, 20, 40, 80, 160, 320};
    for (int k = 0; k < 6; k++) printf(" %9ld", ns[k]);
    printf("\n");
    int rt[] = {TB_CIRCLE, TB_ELLIPSE, TB_SQUARE, TB_STADIUM, TB_SINAI, TB_CARDIOID, TB_MUSHROOM};
    for (unsigned j = 0; j < sizeof rt / sizeof rt[0]; j++) {
        printf("   %-9s", tb_names[rt[j]]);
        for (int k = 0; k < 6; k++) printf(" %9.1e", reversal(rt[j], ns[k], 0.3141, 1.0));
        printf("\n");
    }

    printf("\n4. Lyapunov exponents: two orbits from one wall point, 1e-12 rad apart; the\n"
           "   growth rate of their separation until it reaches 1e-5 (or 3000 bounces), averaged\n"
           "   over 200 starts. Per bounce, and per unit length. ~0 (ln n / n) when integrable.\n");
    int lt[] = {TB_CIRCLE, TB_ELLIPSE, TB_SQUARE, TB_STADIUM, TB_SINAI, TB_CARDIOID, TB_LEMON, TB_LIMACON};
    for (unsigned j = 0; j < sizeof lt / sizeof lt[0]; j++) {
        Table T;
        table_build(&T, lt[j]);
        uint64_t rs = 12345;
        double sum = 0, slen = 0;
        int trials = 200;
        for (int k = 0; k < trials; k++) {
            double u = urand(&rs), ang = 0.2 + 2.7 * urand(&rs);
            Ball a, b;
            ball_on(&a, &T, 0, u, ang);
            ball_on(&b, &T, 0, u, ang + 1e-12);
            double d = 1e-12, len = 0;
            int n = 0;
            while (n < 3000 && d < 1e-5) {
                len += a.len;
                ball_bounce(&a, &T); ball_bounce(&b, &T); n++;
                d = sqrt((a.px - b.px) * (a.px - b.px) + (a.py - b.py) * (a.py - b.py) +
                         (a.dx - b.dx) * (a.dx - b.dx) + (a.dy - b.dy) * (a.dy - b.dy));
                if (d == 0) d = 1e-300;
            }
            sum += log(d / 1e-12) / n;
            slen += log(d / 1e-12) / len;
        }
        printf("   %-9s λ = %.3f per bounce, %.3f per unit length\n", tb_names[lt[j]], sum / trials, slen / trials);
    }

    printf("\n5. filling: one orbit's time in each cell of a grid (cells wholly inside the table),\n"
           "   and its bounces in a 20x20 grid of the phase portrait (s, cos θ). An ergodic\n"
           "   orbit spreads evenly over both (Liouville measure: uniform in area, uniform in\n"
           "   (s, cos θ)); spread = coefficient of variation; ~1/sqrt(visits per cell) is noise.\n");
    int ft[] = {TB_CIRCLE, TB_ELLIPSE, TB_STADIUM, TB_SINAI, TB_CARDIOID, TB_MUSHROOM, TB_LIMACON};
    for (unsigned j = 0; j < sizeof ft / sizeof ft[0]; j++) {
        Table T;
        table_build(&T, ft[j]);
        enum { G = 24 };
        static double cell[G][G];
        static long ph[20][20];
        static int inside[G][G];
        memset(cell, 0, sizeof cell); memset(ph, 0, sizeof ph);
        double cw = (T.x1 - T.x0) / G, chh = (T.y1 - T.y0) / G;
        /* a cell is inside if 25 sample points are (ray parity from each) */
        for (int gx = 0; gx < G; gx++)
            for (int gy = 0; gy < G; gy++) {
                int ok = 1;
                for (int k = 0; k < 25 && ok; k++) {
                    double x = T.x0 + (gx + (k % 5) / 4.0) * cw, y = T.y0 + (gy + (k / 5) / 4.0) * chh;
                    /* count crossings of a ray to +x */
                    int cross = 0;
                    double px = x, py = y, t;
                    for (int it = 0; it < 20; it++) {
                        int h = tb_hit(&T, -1, px, py, 1, 1e-7, &t);
                        if (h < 0) break;
                        cross++;
                        px += t * 1 + 1e-9; py += t * 1e-7;
                    }
                    if (!(cross & 1)) ok = 0;
                }
                inside[gx][gy] = ok;
            }
        Ball b;
        if (ft[j] == TB_MUSHROOM) ball_on(&b, &T, 3, 0.37, 1.2);      /* from the stem: the chaotic sea */
        else ball_on(&b, &T, 0, 0.123, 1.1);
        long N = 400000;
        for (long i = 0; i < N; i++) {
            /* sample the chord every 0.01 units */
            int ns = (int)(b.len / 0.01) + 1;
            for (int k = 0; k < ns; k++) {
                double s = (k + 0.5) * b.len / ns, x = b.px + s * b.dx, y = b.py + s * b.dy;
                int gx = (int)((x - T.x0) / cw), gy = (int)((y - T.y0) / chh);
                if (gx >= 0 && gx < G && gy >= 0 && gy < G) cell[gx][gy] += b.len / ns;
            }
            ball_bounce(&b, &T);
            int sx = (int)(b.ps * 20), sy = (int)((b.pp + 1) / 2 * 20);
            if (sx >= 0 && sx < 20 && sy >= 0 && sy < 20) ph[sx][sy]++;
        }
        double s1 = 0, s2 = 0;
        int nc = 0;
        for (int gx = 0; gx < G; gx++)
            for (int gy = 0; gy < G; gy++)
                if (inside[gx][gy]) { s1 += cell[gx][gy]; s2 += cell[gx][gy] * cell[gx][gy]; nc++; }
        double m = s1 / nc, cv = sqrt(fmax(s2 / nc - m * m, 0)) / m;
        double p1 = 0, p2 = 0, empty = 0;
        for (int a = 0; a < 20; a++)
            for (int c = 0; c < 20; c++) { p1 += ph[a][c]; p2 += (double)ph[a][c] * ph[a][c]; if (!ph[a][c]) empty++; }
        double pm = p1 / 400, pcv = sqrt(fmax(p2 / 400 - pm * pm, 0)) / pm;
        printf("   %-9s space: %3d cells, spread %.3f | phase: spread %.3f, %3.0f of 400 cells empty\n",
               tb_names[ft[j]], nc, cv, pcv, empty);
    }
    printf("\n   (the stadium, Sinai and cardioid spread evenly, to the noise; the circle and\n"
           "   ellipse, confined to their caustics, don't; the mushroom's chaotic orbit and the\n"
           "   limaçon's leave the regular islands of phase space empty.)\n");

    printf("\n6. time reversal maps (s, cos θ) to (s, −cos θ), so a chaotic sea is symmetric: the\n"
           "   share of bounces with cos θ < 0 after n bounces, from a start with cos θ > 0. It\n"
           "   reaches 1/2 at once in the stadium; in the limaçon partial barriers (cantori) hold\n"
           "   an orbit in one half for thousands of bounces.\n");
    printf("   %-9s %9s %9s %9s %9s\n", "n", "1e3", "1e4", "1e5", "1e6");
    int st[] = {TB_STADIUM, TB_CARDIOID, TB_LIMACON, TB_LEMON};
    for (unsigned j = 0; j < sizeof st / sizeof st[0]; j++) {
        Table T;
        table_build(&T, st[j]);
        Ball b;
        ball_on(&b, &T, 0, 0.0003, 1.21);
        long neg = 0, n = 0;
        printf("   %-9s", tb_names[st[j]]);
        for (long lim = 1000; lim <= 1000000; lim *= 10) {
            for (; n < lim; n++) { ball_bounce(&b, &T); neg += b.pp < 0; }
            printf(" %9.3f", (double)neg / n);
        }
        printf("\n");
    }
}

/* ==================================================================== */
/* ---- palettes ---- */
typedef struct {
    const char *name;
    int ink;                           /* light paper: lines absorb rather than glow */
    int weight;                        /* how often the seed picks it */
    const char *bg, *bg2, *bound, *text, *frame;
    const char *ramp[8];
    float umin, exposure, halo;
} PalDef;
static const PalDef PALS[] = {
    {"spectral", 0, 3, "#06070b", "#0d1018", "#e9e4d8", "#cfc8b8", "#7d786c",
     {"#5e4fa2", "#3288bd", "#66c2a5", "#c7e9a0", "#fee08b", "#fdae61", "#f46d43", "#d53e4f"}, 0.0f, 1.0f, 0.55f},
    {"gold", 0, 1, "#070504", "#120d07", "#f3dca2", "#d8c08a", "#7a6640",
     {"#6e4a18", "#a8792c", "#d4a64a", "#ecc874", "#f8e2a8", "#fff3d4", 0, 0}, 0.25f, 1.0f, 0.6f},
    {"ice", 0, 2, "#03060b", "#08121c", "#dff2ff", "#b4d0e2", "#5d7a90",
     {"#1d3f7a", "#2f6db5", "#4aa3dc", "#86d0f0", "#c4ecfa", "#f0fbff", 0, 0}, 0.25f, 1.0f, 0.6f},
    {"paper", 1, 2, "#f2ebdb", "#e6dcc6", "#1b1814", "#2a241c", "#8c8070",
     {"#1d2b55", "#244f7a", "#1f5c58", "#6b4a1e", "#a8322a", "#6a2e5a", 0, 0}, 0.0f, 1.0f, 0.25f},
    {"neon", 0, 2, "#05030a", "#0d0718", "#f4ecff", "#cbbde6", "#6c5a8a",
     {"#ff2e88", "#ff7a1f", "#ffe23d", "#3dffa8", "#2ec8ff", "#8f5bff", "#ff2ee0", 0}, 0.0f, 1.0f, 0.6f},
};
#define NPAL ((int)(sizeof PALS / sizeof PALS[0]))
static struct {
    int ink, nramp;
    float bg[3], bg2[3], bound[3], text[3], frame[3], ramp[8][3];
    float umin, exposure, halo;
    const char *name;
} pal;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

static void pal_set(int k) {
    const PalDef *p = &PALS[k];
    pal.name = p->name;
    pal.ink = p->ink;
    hex(p->bg, pal.bg); hex(p->bg2, pal.bg2); hex(p->bound, pal.bound); hex(p->text, pal.text); hex(p->frame, pal.frame);
    pal.nramp = 0;
    for (int i = 0; i < 8 && p->ramp[i]; i++) hex(p->ramp[i], pal.ramp[pal.nramp++]);
    pal.umin = p->umin; pal.exposure = p->exposure; pal.halo = p->halo;
}

/* a colour along the palette: what a line adds (or, on paper, absorbs) */
static void pal_col(float u, float *c) {
    u = pal.umin + (1 - pal.umin) * clampf(u, 0, 1);
    float f = u * (pal.nramp - 1);
    int i = (int)f;
    if (i >= pal.nramp - 1) i = pal.nramp - 2;
    f -= i;
    for (int k = 0; k < 3; k++) {
        float v = pal.ramp[i][k] + f * (pal.ramp[i + 1][k] - pal.ramp[i][k]);
        c[k] = pal.ink ? 1 - v : v;
    }
}
static void pal_fixed(const float *rgb, float *c) {
    for (int k = 0; k < 3; k++) c[k] = pal.ink ? 1 - rgb[k] : rgb[k];
}

/* ==================================================================== */
/* ---- the acts ---- */
enum { M_ORBITS, M_SWARM };
enum { S_PHASE, S_UNFOLD };
typedef struct {
    int table, mode, side;
    int variant;                       /* which set of orbits / which swarm */
    const char *title, *sub, *label;   /* label: the short name when two tables share a screen */
} PanelSpec;
typedef struct {
    const char *name, *title, *sub;    /* title/sub: for a two-table act */
    float dur;
    int np;
    PanelSpec p[3];
} SceneSpec;

#define PS_CIRCLE {TB_CIRCLE, M_ORBITS, S_PHASE, 0, "The circle", "integrable · each orbit keeps its angle; its chords wrap a concentric caustic circle", "circle"}
#define PS_ELLIPSE {TB_ELLIPSE, M_ORBITS, S_PHASE, 0, "The ellipse", "integrable · every chord of an orbit touches one confocal conic — an ellipse, or a hyperbola", "ellipse"}
#define PS_SQUARE {TB_SQUARE, M_ORBITS, S_PHASE, 0, "The square", "a rational slope closes up; an irrational slope is dense", "square"}
#define PS_STADIUM {TB_STADIUM, M_ORBITS, S_PHASE, 0, "Bunimovich stadium", "chaotic · ergodic — one orbit fills the table uniformly", "stadium"}
#define PS_SINAI {TB_SINAI, M_ORBITS, S_PHASE, 0, "Sinai billiard", "dispersing · the convex scatterer makes every orbit chaotic", "Sinai"}
#define PS_CARDIOID {TB_CARDIOID, M_ORBITS, S_PHASE, 0, "The cardioid", "chaotic · r = 1 + cos θ, the limit of the limaçons", "cardioid"}
#define PS_LIMACON {TB_LIMACON, M_ORBITS, S_PHASE, 0, "Limaçon", "mixed · r = 1 + 0.3 cos θ — islands of order in a chaotic sea", "limaçon"}
#define PS_MUSHROOM {TB_MUSHROOM, M_ORBITS, S_PHASE, 0, "Bunimovich mushroom", "mixed · regular in the cap, chaotic through the stem", "mushroom"}
#define PS_LEMON {TB_LEMON, M_ORBITS, S_PHASE, 0, "Lemon", "mixed · two circular arcs — a stable island about the short axis, chaos around it", "lemon"}
#define PS_TRIANGLE {TB_TRIANGLE, M_ORBITS, S_UNFOLD, 0, "The equilateral triangle", "Fagnano’s period-3 orbit and its neighbours; unfolded, every path is straight", "triangle"}
#define PS_SW_CIRCLE {TB_CIRCLE, M_SWARM, S_PHASE, 0, "The circle, two thousand balls", "launched together, they stay together — drifting apart only slowly", "circle"}
#define PS_SW_STADIUM {TB_STADIUM, M_SWARM, S_PHASE, 0, "The stadium, two thousand balls", "launched together, torn apart exponentially — chaos", "stadium"}
#define PS_SW_MUSHROOM {TB_MUSHROOM, M_SWARM, S_PHASE, 1, "The mushroom, two swarms", "one launched in the regular cap holds together; one in the chaotic sea scatters", "mushroom"}

static const SceneSpec SOLO[] = {
    {"circle", 0, 0, 85, 1, {PS_CIRCLE}},
    {"stadium", 0, 0, 105, 1, {PS_STADIUM}},
    {"ellipse", 0, 0, 100, 1, {PS_ELLIPSE}},
    {"mushroom", 0, 0, 105, 1, {PS_MUSHROOM}},
    {"swarms", "Two thousand balls, one start",
     "the circle keeps them together; the stadium tears them apart exponentially", 80, 2, {PS_SW_CIRCLE, PS_SW_STADIUM}},
    {"square", 0, 0, 80, 1, {PS_SQUARE}},
    {"sinai", 0, 0, 95, 1, {PS_SINAI}},
    {"limacon", 0, 0, 110, 1, {PS_LIMACON}},
    {"triangle", 0, 0, 95, 1, {PS_TRIANGLE}},
    {"cardioid", 0, 0, 95, 1, {PS_CARDIOID}},
    {"lemon", 0, 0, 105, 1, {PS_LEMON}},
    {"mushroom-swarms", 0, 0, 80, 1, {PS_SW_MUSHROOM}},
};
/* a wall: regular | mixed | chaotic, left to right */
static const SceneSpec TRIO[] = {
    {"trio-circle", 0, 0, 100, 3, {PS_CIRCLE, PS_MUSHROOM, PS_STADIUM}},
    {"trio-ellipse", 0, 0, 100, 3, {PS_ELLIPSE, PS_LIMACON, PS_CARDIOID}},
    {"trio-swarms", 0, 0, 85, 3, {PS_SW_CIRCLE, PS_SW_MUSHROOM, PS_SW_STADIUM}},
    {"trio-triangle", 0, 0, 100, 3, {PS_TRIANGLE, PS_LEMON, PS_SINAI}},
};
#define NSOLO ((int)(sizeof SOLO / sizeof SOLO[0]))
#define NTRIO ((int)(sizeof TRIO / sizeof TRIO[0]))

/* ---- pacing ---- */
#define DECAY_TAU 32.0                 /* seconds for the trace to fade to 1/e */
#define TICK 0.25                      /* the decay is applied on this tick of the act's clock */
#define FADE_IN 2.5
#define FADE_OUT 4.0

/* ---- a table on screen, with its balls ---- */
#define MAXSWARM 2400
typedef struct {
    const PanelSpec *spec;
    Table T;
    int nb;
    Ball *b;
    float bu_dir;                      /* colour the lines by direction (single chaotic orbits) */
    double r0, r1, ramp;               /* bounces a second: from r0 up to r1 over `ramp` seconds */
    /* layout, px */
    float tx, ty, ts;                  /* screen = (tx + ts x, ty - ts y) */
    float box[4], side[4];             /* table box, side panel box: x0 y0 x1 y1 */
    float sigma, I_line, I_dot, I_head;
    /* the unfolding */
    double ux, uy, udx, udy;           /* start and direction of the unfolded orbit */
    int uorb;
} Panel;

/* path length run at act time τ: the rate eases from r0 to r1 bounces a
 * second (smoothstep, integrated in closed form) and holds */
static double path_at(const Panel *P, double tau) {
    if (tau <= 0) return 0;
    double x = fmin(tau / P->ramp, 1), I = x * x * x - 0.5 * x * x * x * x;
    double n = P->r0 * tau + (P->r1 - P->r0) * (P->ramp * I + fmax(tau - P->ramp, 0));
    return n * P->T.mfp;
}
static double rate_at(const Panel *P, double tau) {
    return P->r0 + (P->r1 - P->r0) * smooth(0, P->ramp, tau);
}

/* the ellipse: a direction from boundary point (piece 0 at u) whose chord
 * touches the confocal conic of parameter λ */
static double ell_angle_for(const Table *T, double u, double lam) {
    double lo = 1e-3, hi = M_PI / 2, x, y, nx, ny, tx, ty;
    pc_point(&T->pc[0], u, &x, &y);
    pc_frame(&T->pc[0], x, y, &nx, &ny, &tx, &ty);
    /* λ grows as the chord turns from the tangent to the normal */
    for (int i = 0; i < 80; i++) {
        double m = 0.5 * (lo + hi), dx = cos(m) * tx + sin(m) * nx, dy = cos(m) * ty + sin(m) * ny;
        double c = x * dy - y * dx, l = ELL_A * ELL_A * dy * dy + ELL_B * ELL_B * dx * dx - c * c;
        if (l < lam) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
}

static void panel_init(Panel *P, const PanelSpec *ps, uint64_t seed) {
    uint64_t rs = seed;
    free(P->b);
    for (int i = 0; i < P->T.n; i++) free(P->T.pc[i].tab);
    memset(P, 0, sizeof *P);
    P->spec = ps;
    table_build(&P->T, ps->table);
    const Table *T = &P->T;
    P->r0 = 0.45; P->r1 = 20; P->ramp = 45;
    P->b = calloc(MAXSWARM, sizeof *P->b);
    Ball *b = P->b;
    int n = 0;
    double j1 = urand(&rs), j2 = urand(&rs);
    if (ps->mode == M_SWARM) {
        P->r0 = 0.3; P->r1 = 0.85; P->ramp = 12;
        double cx, cy, ang, rad = 0.010, spread = 0.004;
        int clouds = 1;
        switch (ps->table) {
        case TB_STADIUM: cx = -0.2; cy = -0.42; ang = 0.08 + 0.15 * j1; break;
        case TB_MUSHROOM: clouds = 2; cx = 0; cy = 0; ang = 0; break;
        default: cx = -0.2; cy = -0.42; ang = 0.08 + 0.15 * j1; break;      /* caustic radius ~0.4 */
        }
        int per = 2000 / clouds;
        for (int c = 0; c < clouds; c++) {
            if (ps->table == TB_MUSHROOM) {
                /* the cap's rim (caustic radius 0.82 > the stem's 0.4: regular), and the stem */
                if (c == 0) { cx = -0.15 + 0.1 * j2; cy = 0.45 + 0.82; ang = 0.02; }
                else { cx = 0.08; cy = -0.35; ang = 1.25 + 0.2 * j1; }
            }
            for (int i = 0; i < per; i++) {
                /* a small disc of starts, the angle spread by the offset along it */
                double r = rad * sqrt(urand(&rs)), a = TAU * urand(&rs), da = spread * (urand(&rs) - 0.5);
                double ox = r * cos(a), oy = r * sin(a);
                ball_in(&b[n], T, cx + ox, cy + oy, ang + da, 0);
                /* colour by where in the cloud it started, across the beam */
                float across = (float)((-ox * sin(ang) + oy * cos(ang)) / rad);
                b[n].u = clouds == 2 ? (c ? 0.62f : 0.12f) + 0.22f * (across * 0.5f + 0.5f)
                                     : 0.5f + 0.5f * across;
                n++;
            }
        }
    } else {
        switch (ps->table) {
        case TB_CIRCLE: {
            /* rotation numbers near 2/5, 2/7 and 1/8: slowly precessing stars */
            double th[3] = {0.4021, 0.2834 + 0.004 * j1, 0.1309};
            for (int i = 0; i < 3; i++) { ball_on(&b[n], T, 0, urand(&rs), th[i] * M_PI); b[n].u = (i + 0.5f) / 3; n++; }
            P->r1 = 5;                  /* slow enough that the stars stay distinct */
        } break;
        case TB_ELLIPSE: {
            /* two orbits round confocal ellipses, two between confocal hyperbolas */
            double a2 = ELL_A * ELL_A, b2 = ELL_B * ELL_B;
            double lam[4] = {0.22 * b2, 0.72 * b2, b2 + 0.30 * (a2 - b2), b2 + 0.78 * (a2 - b2)};
            double us[4] = {0.02, 0.4, 0.25, 0.27};
            for (int i = 0; i < 4; i++) {
                double u = us[i] + 0.05 * urand(&rs);
                ball_on(&b[n], T, 0, u, ell_angle_for(T, u, lam[i]));
                b[n].u = (i + 0.5f) / 4; n++;
            }
            P->r1 = 10;
        } break;
        case TB_SQUARE:
            ball_in(&b[n], T, -0.31, -0.77, atan2(2, 3), 0); b[n].u = 0.15f; n++;             /* slope 2/3 */
            ball_in(&b[n], T, 0.21, 0.13, atan2(0.5 * (sqrt(5) - 1), 1), 0); b[n].u = 0.8f; n++; /* slope 1/φ */
            P->r1 = 10;
            break;
        case TB_TRIANGLE: {
            double s = TRI_S, h = s * sqrt(3) / 2;
            /* Fagnano: the midpoints of the sides, parallel to them; a parallel
             * neighbour (period 6); and a generic slope */
            double mx = 0, my = -h / 3, m2x = s / 4, m2y = (-h / 3 + 2 * h / 3) / 2;
            double fang = atan2(m2y - my, m2x - mx);
            ball_on(&b[n], T, 0, 0.5, fang); b[n].u = 0.85f; n++;
            ball_on(&b[n], T, 0, 0.36, fang); b[n].u = 0.6f; n++;
            ball_on(&b[n], T, 0, 0.29 + 0.05 * j1, 0.7854 * 0.9787); b[n].u = 0.2f; n++;
            P->uorb = 2;
            P->ux = b[2].px; P->uy = b[2].py; P->udx = b[2].dx; P->udy = b[2].dy;
            P->r1 = 6;
        } break;
        case TB_MUSHROOM: {
            /* two cap orbits (caustic radius 0.84 and 0.6 > the stem's 0.4: they
             * never find the stem), one chaotic orbit from the stem */
            ball_on(&b[n], T, 0, 0.1 + 0.2 * j1, acos(0.84)); b[n].u = 0.9f; n++;
            ball_on(&b[n], T, 0, 0.6, acos(0.6)); b[n].u = 0.7f; n++;
            ball_on(&b[n], T, 3, 0.37, 1.2 + 0.3 * j2); b[n].u = 0.2f; n++;
            P->r1 = 16;
        } break;
        case TB_LIMACON: {
            /* launched from the rightmost point at a fan of angles: some ride
             * invariant curves and islands, the rest share the chaotic sea */
            double a[7] = {0.26, 0.52, 0.80, 1.02, 1.21, 1.38, 1.53};
            for (int i = 0; i < 7; i++) { ball_on(&b[n], T, 0, 0.0005 * j1, a[i] + 0.01 * urand(&rs)); b[n].u = (i + 0.5f) / 7; n++; }
            P->r1 = 16;
        } break;
        case TB_LEMON: {
            /* near the short axis (the stable period-2 orbit): the island; and the sea */
            ball_on(&b[n], T, 0, 0.5 + 0.03, M_PI / 2 - 0.05); b[n].u = 0.85f; n++;
            ball_on(&b[n], T, 0, 0.5 + 0.10, M_PI / 2 - 0.12); b[n].u = 0.6f; n++;
            ball_on(&b[n], T, 0, 0.2 + 0.1 * j1, 0.7); b[n].u = 0.2f; n++;
            P->r1 = 16;
        } break;
        default:
            /* chaotic: one orbit, its lines coloured by direction */
            ball_on(&b[n], T, 0, urand(&rs), 0.35 + 1.0 * urand(&rs)); b[n].u = 0.5f; n++;
            P->bu_dir = 1;
            P->r1 = 28;
            break;
        }
    }
    P->nb = n;
}

/* ==================================================================== */
/* ---- glyph atlas: printable ASCII and a few more (Greek, dashes, the
 * middle dot) in each text style used, rasterised at its exact size ---- */
static const unsigned EXTRA[] = {0x3b8, 0xb7, 0x2014, 0x2013, 0x2212, 0x3b5, 0x3c0, 0xd7, 0x2248, 0x3bb, 0x2019,
                                 0xe7, 0x2202, 0x3a9, 0xb0, 0x3c1};
#define NGLYPH (95 + (int)(sizeof EXTRA / sizeof EXTRA[0]))
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGLYPH]; } Style;
#define MAXSTYLE 16
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* regular, italic */
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < 2; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:slant=%s", opt_font, i ? "italic" : "roman");
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
    int cap = 2048;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < NGLYPH; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c < 95 ? (FT_ULong)(32 + c) : EXTRA[c - 95];
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

/* the next code point of a UTF-8 string, as a glyph index */
static int next_glyph(const char **sp) {
    const unsigned char *s = (const unsigned char *)*sp;
    unsigned cp = *s++;
    if (cp >= 0xf0) { cp = (cp & 7) << 18 | (s[0] & 63) << 12 | (s[1] & 63) << 6 | (s[2] & 63); s += 3; }
    else if (cp >= 0xe0) { cp = (cp & 15) << 12 | (s[0] & 63) << 6 | (s[1] & 63); s += 2; }
    else if (cp >= 0xc0) { cp = (cp & 31) << 6 | (s[0] & 63); s += 1; }
    *sp = (const char *)s;
    if (cp >= 32 && cp < 127) return (int)cp - 32;
    for (int i = 0; i < (int)(sizeof EXTRA / sizeof EXTRA[0]); i++)
        if (EXTRA[i] == cp) return 95 + i;
    return '?' - 32;
}

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p;) {
        w += styles[s].g[next_glyph(&p)].adv;
        if (*p) w += track;
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

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x */
static void text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = roundf(x - (align == 0 ? w * 0.5f : align > 0 ? w : 0));
    y = roundf(y);
    for (const char *p = str; *p;) {
        const Glyph *g = &styles[s].g[next_glyph(&p)];
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y0, u1, v0, col); quad_v(x1, y1, u1, v1, col);
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y1, u1, v1, col); quad_v(x0, y1, u0, v1, col);
        }
        pen += g->adv + track;
    }
}

/* ==================================================================== */
/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* a line (or a dot) per instance: a Gaussian profile across, flat ends, so
 * the pieces of a path drawn frame by frame add up to one even line */
static const char *VS_LINE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aSeg;          /* x0 y0 x1 y1, px */
    layout(location = 1) in vec4 aCol;          /* rgb, intensity */
    layout(location = 2) in vec4 aPar;          /* sigma px, fade at start, fade at end, kind (0 line, 1 dot) */
    uniform vec2 uRes;
    uniform float uScale, uSigma;
    out vec2 vL;
    out vec3 vP;
    out vec4 vCol;
    out vec2 vF;
    out float vG;
    void main() {
        vec2 a = aSeg.xy * uScale, b = aSeg.zw * uScale;
        float s0 = aPar.x * uScale;
        float sg = uSigma > 0.0 ? max(uSigma, s0) : s0;
        /* the glow spreads the same light wider: keep its total */
        vG = aPar.w > 0.5 ? (s0 / sg) * (s0 / sg) : s0 / sg;
        vec2 d = b - a;
        float len = length(d);
        vec2 t = len > 1e-5 ? d / len : vec2(1.0, 0.0), n = vec2(-t.y, t.x);
        float ext = 2.6 * sg;
        float sx = float(gl_VertexID & 1), sy = (gl_VertexID & 2) != 0 ? 1.0 : -1.0;
        float al = aPar.w > 0.5 ? mix(-ext, len + ext, sx) : mix(0.0, len, sx);
        vec2 p = a + t * al + n * (sy * ext);
        vL = vec2(al, sy * ext);
        vP = vec3(len, sg, aPar.w);
        vCol = aCol;
        vF = aPar.yz;
        gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_LINE = "#version 330 core\n" GLSL(
    in vec2 vL;
    in vec3 vP;
    in vec4 vCol;
    in vec2 vF;
    in float vG;
    uniform float uGain;
    out vec4 o;
    void main() {
        float d = abs(vL.y);
        if (vP.z > 0.5) d = length(vec2(vL.x - clamp(vL.x, 0.0, vP.x), vL.y));
        float g = exp(-d * d / (vP.y * vP.y));
        float f = mix(vF.x, vF.y, vP.x > 1e-5 ? clamp(vL.x / vP.x, 0.0, 1.0) : 0.0);
        o = vec4(vCol.rgb * (vCol.a * g * f * uGain * vG), 0.0);
    });

/* the accumulated light (or ink) on the background, toned */
static const char *FS_SHOW = "#version 330 core\n" GLSL(
    uniform sampler2D uAcc, uHalo;
    uniform vec2 uRes;
    uniform vec3 uBg, uBg2;
    uniform float uExp, uHaloK, uFade;
    uniform int uInk;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec3 v = texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0).rgb + uHaloK * texture(uHalo, uv).rgb;
        v = max(v, 0.0) * uFade * uExp;
        vec2 c = (uv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
        float r = length(c);
        vec3 bg = mix(uBg2, uBg, smoothstep(0.0, 0.9, r));
        vec3 col;
        if (uInk == 1) col = bg * exp(-v);
        else col = bg + (1.0 - exp(-v)) * (1.0 - bg * 0.5);
        col += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });
static const char *FS_ZERO = "#version 330 core\n" GLSL(
    out vec4 o;
    void main() { o = vec4(0.0); });

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

static GLuint prog_line, prog_show, prog_zero, prog_text, vao, vao_line, vbo_line, vao_text, vbo_text;
static GLuint tex_atlas, tex_acc, tex_halo, fbo_acc, fbo_halo;
static int acc_w, acc_h, atlas_dirty = 1;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "billiards: shader:\n%s\n", log);
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

static void gl_init(void) {
    prog_line = mk_prog(VS_LINE, FS_LINE);
    prog_show = mk_prog(VS_FULL, FS_SHOW);
    prog_zero = mk_prog(VS_FULL, FS_ZERO);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_line);
    glGenBuffers(1, &vbo_line);
    glBindVertexArray(vao_line);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_line);
    for (int i = 0; i < 3; i++) {
        glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void *)(4 * i * sizeof(float)));
        glEnableVertexAttribArray(i);
        glVertexAttribDivisor(i, 1);
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
    glGenTextures(1, &tex_acc);
    glGenTextures(1, &tex_halo);
    glGenFramebuffers(1, &fbo_acc);
    glGenFramebuffers(1, &fbo_halo);
}

static void mk_target(GLuint tex, GLuint fbo, int w, int h, GLenum filt) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("half-float render target unsupported");
}

#define HALO_DIV 4
static void acc_clear(void) {
    glClearColor(0, 0, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_halo);
    glClear(GL_COLOR_BUFFER_BIT);
}

/* ---- line lists ---- */
typedef struct { float *v; int n, cap; } LineList;
static LineList acc_l, ovl_l;
static float acc_sigma_px;             /* the glow's width (in halo px) */

static void flush_acc(void);
static void emit(LineList *l, float x0, float y0, float x1, float y1, const float *c, float I, float sg, float f0,
                 float f1, int kind) {
    if (I <= 0) return;
    if (l->n == l->cap) {
        if (l == &acc_l && l->cap >= 65536) flush_acc();
        else l->v = realloc(l->v, (size_t)(l->cap = l->cap ? l->cap * 2 : 4096) * 12 * sizeof(float));
    }
    float *o = l->v + (size_t)l->n++ * 12;
    o[0] = x0; o[1] = y0; o[2] = x1; o[3] = y1;
    o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = I;
    o[8] = sg; o[9] = f0; o[10] = f1; o[11] = (float)kind;
}

static void draw_list(LineList *l, int w, int h, float scale, float sigma, float gain) {
    if (!l->n) return;
    glUseProgram(prog_line);
    glUniform2f(U(prog_line, "uRes"), (float)w, (float)h);
    glUniform1f(U(prog_line, "uScale"), scale);
    glUniform1f(U(prog_line, "uSigma"), sigma);
    glUniform1f(U(prog_line, "uGain"), gain);
    glBindVertexArray(vao_line);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_line);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)l->n * 12 * sizeof(float), l->v, GL_STREAM_DRAW);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, l->n);
}

/* into the accumulation buffers: the fine line, and its glow at a quarter size */
static void flush_acc(void) {
    if (!acc_l.n) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
    glViewport(0, 0, acc_w, acc_h);
    draw_list(&acc_l, acc_w, acc_h, 1, 0, 1);
    int hw = (acc_w + HALO_DIV - 1) / HALO_DIV, hh = (acc_h + HALO_DIV - 1) / HALO_DIV;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_halo);
    glViewport(0, 0, hw, hh);
    draw_list(&acc_l, hw, hh, 1.0f / HALO_DIV, acc_sigma_px, 1);
    acc_l.n = 0;
}

static void acc_decay(double f) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_ZERO, GL_CONSTANT_COLOR);
    glBlendColor((float)f, (float)f, (float)f, (float)f);
    glUseProgram(prog_zero);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
    glViewport(0, 0, acc_w, acc_h);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_halo);
    glViewport(0, 0, (acc_w + HALO_DIV - 1) / HALO_DIV, (acc_h + HALO_DIV - 1) / HALO_DIV);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* ==================================================================== */
/* ---- the scene on this screen ---- */
static struct {
    int idx;                           /* which act of the show (the running count) */
    const SceneSpec *spec;
    int np;
    Panel P[2];
    const char *title, *sub;
    double dur, tau;                   /* act length; simulated up to */
    double next_tick;
    int built_w, built_h;
} sc = {.idx = -1};
static double fade_now = 1;
static float min_dim, ui_m;
static int st_title, st_sub, st_small, st_cap;
static float px_title, px_sub;

/* the side panel's aspect */
#define PH_ASPECT 1.6f

static unsigned utf8_next(const char **sp) {
    const unsigned char *s = (const unsigned char *)*sp;
    unsigned cp = *s++;
    if (cp >= 0xf0) { cp = (cp & 7) << 18 | (s[0] & 63) << 12 | (s[1] & 63) << 6 | (s[2] & 63); s += 3; }
    else if (cp >= 0xe0) { cp = (cp & 15) << 12 | (s[0] & 63) << 6 | (s[1] & 63); s += 2; }
    else if (cp >= 0xc0) { cp = (cp & 31) << 6 | (s[0] & 63); s += 1; }
    *sp = (const char *)s;
    return cp;
}

/* the width of (the first n bytes of) str in a face at px, before the atlas is built */
static float measure(int face, float px, const char *str, int n) {
    FT_Set_Pixel_Sizes(faces[face], 0, (FT_UInt)lroundf(px));
    float w = 0;
    for (const char *p = str; *p && (n < 0 || p < str + n);)
        if (!FT_Load_Char(faces[face], utf8_next(&p), FT_LOAD_DEFAULT)) w += faces[face]->glyph->advance.x >> 6;
    return w;
}

/* the subtitle, broken at spaces into lines no wider than maxw */
#define MAXSUB 4
static char sub_l[MAXSUB][200];
static int n_sub;
static float sub_w;
static void wrap_sub(const char *str, float px, float maxw) {
    n_sub = 0;
    sub_w = 0;
    const char *p = str;
    while (*p && n_sub < MAXSUB) {
        const char *brk = NULL, *e = p;
        for (;;) {
            while (*e && *e != ' ') e++;
            if (brk && measure(1, px, p, (int)(e - p)) > maxw) break;
            brk = e;
            if (!*e) break;
            e++;
        }
        if (n_sub == MAXSUB - 1) brk = p + strlen(p);
        snprintf(sub_l[n_sub], sizeof sub_l[0], "%.*s", (int)(brk - p), p);
        float w = measure(1, px, sub_l[n_sub], -1);
        if (w > sub_w) sub_w = w;
        n_sub++;
        p = *brk ? brk + 1 : brk;
    }
}
static float cap_block_h(void) { return px_title * 1.35f + n_sub * px_sub * 1.45f; }
static float cap_x, cap_y;                    /* the caption block's top left */

/* Lay out the table(s) and side panel(s). One table: the side panel and the
 * caption either share a row under the table (caption left, panel right) or
 * a column beside it, whichever lets the table be bigger; the panel then
 * grows into what's spare. Two tables (one scale, a fair comparison): side
 * by side or stacked, each with its panel, the caption along the bottom. */
static void layout(void) {
    min_dim = (float)(W < H ? W : H);
    float m = 0.05f * min_dim, gap = 0.045f * min_dim;
    ui_m = m;
    nstyles = 0;
    px_title = clampf(0.030f * min_dim, 15, 60);
    px_sub = clampf(0.0175f * min_dim, 11, 34);
    st_title = style(1, px_title);
    st_sub = style(1, px_sub);
    st_small = style(1, clampf(0.0145f * min_dim, 10, 28));
    st_cap = style(0, clampf(0.0150f * min_dim, 10, 28));
    float cw = W - 2 * m, ch = H - 2 * m, ph0 = 0.25f * min_dim, pw0 = ph0 * PH_ASPECT;
    float title_w = measure(1, px_title, sc.title, -1);
    float bx[2][4], bs[2][4];
    float s = 0;
    if (sc.np == 1) {
        const Table *T = &sc.P[0].T;
        float wt = (float)(T->x1 - T->x0), ht = (float)(T->y1 - T->y0);
        /* below: caption | panel */
        float capmax = fmaxf(cw - pw0 - gap * 1.5f, 0.3f * cw);
        wrap_sub(sc.sub, px_sub, capmax);
        float capw_v = fmaxf(title_w, sub_w), rowh = fmaxf(ph0, cap_block_h());
        float sv = fminf(cw / wt, (ch - gap - rowh) / ht);
        /* beside: panel over caption */
        wrap_sub(sc.sub, px_sub, pw0);
        float colh = ph0 + gap * 0.8f + cap_block_h();
        float sh = fminf((cw - gap * 1.4f - pw0) / wt, ch / ht);
        if (colh > ch) sh = 0;
        int vert = sv >= sh;
        s = vert ? sv : sh;
        float w = wt * s, h = ht * s, ph, pw;
        if (vert) {
            wrap_sub(sc.sub, px_sub, capmax);
            /* grow the panel into the spare height, as wide as the row allows */
            float spare = ch - (h + gap + rowh);
            float rw = fminf(fmaxf(w, capw_v + gap * 1.5f + pw0), cw);
            ph = fminf(ph0 + spare, fminf(0.42f * min_dim, (rw - capw_v - gap * 1.5f) / PH_ASPECT));
            ph = fmaxf(ph, ph0);
            pw = ph * PH_ASPECT;
            rowh = fmaxf(ph, cap_block_h());
            float x0 = m + (cw - rw) / 2, y0 = m + (ch - (h + gap + rowh)) / 2;
            bx[0][0] = x0 + (rw - w) / 2; bx[0][1] = y0;
            bs[0][0] = x0 + rw - pw; bs[0][1] = y0 + h + gap + (rowh - ph);
            cap_x = x0;
            cap_y = y0 + h + gap + rowh - cap_block_h();
        } else {
            float spare = cw - (w + gap * 1.4f + pw0);
            pw = fminf(pw0 + spare, 0.42f * min_dim * PH_ASPECT);
            wrap_sub(sc.sub, px_sub, pw);
            ph = fminf(pw / PH_ASPECT, fmaxf(ph0, h - gap * 0.8f - cap_block_h()));
            pw = ph * PH_ASPECT;
            wrap_sub(sc.sub, px_sub, pw);
            float colh2 = ph + gap * 0.8f + cap_block_h(), bh = fmaxf(h, colh2);
            float x0 = m + (cw - (w + gap * 1.4f + pw)) / 2, y0 = m + (ch - bh) / 2;
            bx[0][0] = x0; bx[0][1] = y0 + (bh - h) / 2;
            bs[0][0] = x0 + w + gap * 1.4f; bs[0][1] = y0 + (bh - colh2) / 2;
            cap_x = bs[0][0];
            cap_y = bs[0][1] + ph + gap * 0.8f;
        }
        bx[0][2] = bx[0][0] + w; bx[0][3] = bx[0][1] + h;
        bs[0][2] = bs[0][0] + pw; bs[0][3] = bs[0][1] + ph;
    } else {
        wrap_sub(sc.sub, px_sub, cw * 0.7f);
        float band = cap_block_h() + gap * 0.6f;
        ch -= band;
        float ph = ph0 * 0.9f, pw = ph * PH_ASPECT, best = -1;
        int barr = 0, bor = 0;
        for (int arr = 0; arr < 2; arr++)
            for (int o = 0; o < 2; o++) {
                float lo = 0, hi = 1e5;
                for (int it = 0; it < 50; it++) {
                    float sm = 0.5f * (lo + hi), tw = 0, th = 0;
                    for (int i = 0; i < 2; i++) {
                        const Table *T = &sc.P[i].T;
                        float w = (float)(T->x1 - T->x0) * sm, h = (float)(T->y1 - T->y0) * sm;
                        float pwid = o == 0 ? w + gap + pw : fmaxf(w, pw), phei = o == 0 ? fmaxf(h, ph) : h + gap + ph;
                        if (arr == 0) { tw += pwid + (i ? gap * 1.5f : 0); th = fmaxf(th, phei); }
                        else { th += phei + (i ? gap * 1.5f : 0); tw = fmaxf(tw, pwid); }
                    }
                    if (tw <= cw && th <= ch) lo = sm; else hi = sm;
                }
                if (lo > best * 1.001f) { best = lo; barr = arr; bor = o; }
            }
        s = best;
        float x = 0, y = 0, tw = 0, th = 0, ext[2][2];
        for (int i = 0; i < 2; i++) {
            const Table *T = &sc.P[i].T;
            float w = (float)(T->x1 - T->x0) * s, h = (float)(T->y1 - T->y0) * s;
            float pwid = bor == 0 ? w + gap + pw : fmaxf(w, pw), phei = bor == 0 ? fmaxf(h, ph) : h + gap + ph;
            if (bor == 0) {
                bx[i][0] = x; bx[i][1] = y + (phei - h) / 2;
                bs[i][0] = x + w + gap; bs[i][1] = y + (phei - ph) / 2;
            } else {
                bx[i][0] = x + (pwid - w) / 2; bx[i][1] = y;
                bs[i][0] = x + (pwid - pw) / 2; bs[i][1] = y + h + gap;
            }
            bx[i][2] = bx[i][0] + w; bx[i][3] = bx[i][1] + h;
            bs[i][2] = bs[i][0] + pw; bs[i][3] = bs[i][1] + ph;
            ext[i][0] = pwid; ext[i][1] = phei;
            if (barr == 0) { x += pwid + gap * 1.5f; th = fmaxf(th, phei); } else { y += phei + gap * 1.5f; tw = fmaxf(tw, pwid); }
        }
        if (barr == 0) tw = x - gap * 1.5f; else th = y - gap * 1.5f;
        for (int i = 0; i < 2; i++) {
            /* centre the lot, and each panel across the other's extent */
            float dx = m + (cw - tw) / 2, dy = m + (ch - th) / 2;
            if (barr == 0) dy += (th - ext[i][1]) / 2; else dx += (tw - ext[i][0]) / 2;
            for (int k = 0; k < 4; k++) { bx[i][k] += k & 1 ? dy : dx; bs[i][k] += k & 1 ? dy : dx; }
        }
        cap_x = m;
        cap_y = H - m - cap_block_h();
    }
    for (int i = 0; i < sc.np; i++) {
        Panel *P = &sc.P[i];
        for (int k = 0; k < 4; k++) { P->box[k] = roundf(bx[i][k]); P->side[k] = roundf(bs[i][k]); }
        const Table *T = &P->T;
        P->ts = s;
        P->tx = P->box[0] - (float)T->x0 * s;
        P->ty = P->box[3] + (float)T->y0 * s;
    }
    build_atlas();
    atlas_dirty = 1;
}

/* the line widths and intensities, normalised so that each act reaches about
 * the same mean density however long its lines and however big the screen */
static void normalise(Panel *P) {
    float sg = fmaxf(0.62f, 0.00062f * min_dim);
    P->sigma = sg;
    /* path length drawn, weighted by the decay it has had by the act's end */
    double Lw = 0;
    for (double t = 0; t < sc.dur; t += 0.25) Lw += rate_at(P, t) * 0.25 * exp(-(sc.dur - t) / DECAY_TAU);
    double nbounce = Lw * P->nb;                          /* bounces of all the balls, weighted */
    Lw *= P->T.mfp * P->nb;
    double area_px = P->T.area * P->ts * P->ts;
    double side_px = (double)(P->side[2] - P->side[0]) * (P->side[3] - P->side[1]);
    double dl = P->spec->mode == M_SWARM ? 0.30 : 0.55;   /* target mean density, in exposure units */
    double I = dl * area_px / (sg * sqrt(M_PI) * Lw * P->ts);
    P->I_line = clampf((float)I, 0.004f, 1.6f);
    double sd = fmaxf(0.85f, 0.0009f * min_dim);
    double Id = (P->spec->mode == M_SWARM ? 0.25 : 0.6) * side_px / (M_PI * sd * sd * nbounce);
    P->I_dot = clampf((float)Id, P->spec->mode == M_SWARM ? 0.01f : 0.2f, 2.4f);
}

static uint64_t scene_seed(int idx, int k) {
    return (uint64_t)opt_seed * 0x9e3779b97f4a7c15ULL ^ (uint64_t)(idx + 1) * 0xc2b2ae3d27d4eb4fULL ^ (uint64_t)(k + 7) * 0x165667b19e3779f9ULL;
}

/* which act runs at show time t, and how far into it */
static const SceneSpec *sched(double t, int *idx, double *tau) {
    const SceneSpec *list = slot >= 0 ? TRIO : SOLO;
    int n = slot >= 0 ? NTRIO : NSOLO;
    int fixed = -1;
    if (opt_scene)
        for (int i = 0; i < n; i++)
            if (!strcmp(opt_scene, list[i].name)) fixed = i;
    if (fixed >= 0) {
        double d = list[fixed].dur;
        int k = (int)floor(t / d);
        *idx = k;
        *tau = t - k * d;
        return &list[fixed];
    }
    double total = 0;
    for (int i = 0; i < n; i++) total += list[i].dur;
    int start = (int)(opt_seed % (uint32_t)n);
    long cyc = (long)floor(t / total);
    double r = t - cyc * total;
    for (int i = 0; i < n; i++) {
        const SceneSpec *s = &list[(start + i) % n];
        if (r < s->dur || i == n - 1) { *idx = (int)(cyc * n + i); *tau = r; return s; }
        r -= s->dur;
    }
    *idx = 0; *tau = 0;
    return list;
}

/* ---- drawing the path ---- */
static void to_px(const Panel *P, double x, double y, float *sx, float *sy) {
    *sx = P->tx + P->ts * (float)x;
    *sy = P->ty - P->ts * (float)y;
}

static void ball_col(const Panel *P, const Ball *b, float *c) {
    if (P->bu_dir) {
        /* by direction (mod π): the same chord either way has the same colour */
        float a = (float)atan2(b->dy, b->dx);
        pal_col(0.5f + 0.5f * cosf(2 * a + 0.6f), c);
    } else pal_col(b->u, c);
}

static void phase_px(const Panel *P, float s, float p, float *x, float *y) {
    *x = P->side[0] + s * (P->side[2] - P->side[0]);
    *y = P->side[3] - (p + 1) * 0.5f * (P->side[3] - P->side[1]);
}

/* draw ball b's path on to path length L into the accumulation, weight w;
 * each bounce leaves a dot in the phase portrait */
static void ball_draw_to(Panel *P, Ball *b, double L, float w) {
    float c[3], dot = P->spec->mode == M_SWARM ? P->I_dot * w : P->I_dot * w;
    float Il = P->I_line * w;
    for (;;) {
        double Lend = b->L0 + b->len, a = b->Ld - b->L0, e = fmin(L, Lend) - b->L0;
        if (e > a) {
            float x0, y0, x1, y1;
            to_px(P, b->px + a * b->dx, b->py + a * b->dy, &x0, &y0);
            to_px(P, b->px + e * b->dx, b->py + e * b->dy, &x1, &y1);
            ball_col(P, b, c);
            emit(&acc_l, x0, y0, x1, y1, c, Il, P->sigma, 1, 1, 0);
        }
        if (L < Lend) { b->Ld = L; return; }
        b->Ld = Lend;
        ball_bounce(b, &P->T);
        if (P->spec->side == S_PHASE) {
            float x, y;
            phase_px(P, b->ps, b->pp, &x, &y);
            pal_col(P->bu_dir ? 0.5f + 0.5f * b->pp * b->pp : b->u, c);
            if (P->bu_dir) ball_col(P, b, c);
            emit(&acc_l, x, y, x, y, c, dot, fmaxf(0.9f, 0.00085f * min_dim), 1, 1, 1);
        }
    }
}

static void panel_draw_to(Panel *P, double tau, float w) {
    double L = path_at(P, tau);
    /* a swarm leaves its trace only once it has begun to spread: two thousand
     * balls on one chord would burn it in for minutes */
    if (P->spec->mode == M_SWARM) w *= (float)smooth(2, 20, tau);
    for (int i = 0; i < P->nb; i++) ball_draw_to(P, &P->b[i], L, w);
}

/* set up act idx (at its start), then catch up to τ: the path so far, each
 * stretch weighted by the decay it would have had */
static void scene_start(const SceneSpec *s, int idx, double tau) {
    sc.idx = idx;
    sc.spec = s;
    sc.dur = s->dur;
    if (slot >= 0) {
        sc.np = 1;
        panel_init(&sc.P[0], &s->p[slot % s->np], scene_seed(idx, slot % s->np));
        sc.title = s->p[slot % s->np].title; sc.sub = s->p[slot % s->np].sub;
    } else {
        sc.np = s->np;
        for (int i = 0; i < s->np; i++) panel_init(&sc.P[i], &s->p[i], scene_seed(idx, i));
        sc.title = s->title ? s->title : s->p[0].title;
        sc.sub = s->sub ? s->sub : s->p[0].sub;
    }
    layout();
    sc.built_w = W; sc.built_h = H;
    for (int i = 0; i < sc.np; i++) normalise(&sc.P[i]);
    acc_clear();
    for (int i = 0; i < sc.np; i++) {
        Panel *P = &sc.P[i];
        double step = P->spec->mode == M_SWARM ? 1.0 : TICK;
        for (double t = step; t < tau + step; t += step) {
            double te = fmin(t, tau);
            panel_draw_to(P, te, (float)exp(-(tau - te) / DECAY_TAU));
            if (te >= tau) break;
        }
    }
    flush_acc();
    sc.tau = tau;
    sc.next_tick = (floor(tau / TICK) + 1) * TICK;
}

static void scene_advance(double tau) {
    while (sc.next_tick <= tau) {
        for (int i = 0; i < sc.np; i++) panel_draw_to(&sc.P[i], sc.next_tick, 1);
        flush_acc();
        acc_decay(exp(-TICK / DECAY_TAU));
        sc.next_tick += TICK;
    }
    for (int i = 0; i < sc.np; i++) panel_draw_to(&sc.P[i], tau, 1);
    flush_acc();
    sc.tau = tau;
}

/* ---- the overlay: walls, heads and trails, the swarm, the side panel ---- */
static void boundary(const Panel *P, float I) {
    float c[3], cg[3];
    pal_fixed(pal.bound, c);
    for (int k = 0; k < 3; k++) cg[k] = c[k];
    float sg = P->sigma * 0.95f;
    for (int i = 0; i < P->T.n; i++) {
        const Piece *pc = &P->T.pc[i];
        int n = pc->type == P_SEG ? 1 : (int)fmax(24, pc->len * P->ts / 3);
        float px = 0, py = 0;
        for (int k = 0; k <= n; k++) {
            double x, y;
            pc_point(pc, (double)k / n, &x, &y);
            float sx, sy;
            to_px(P, x, y, &sx, &sy);
            if (k) {
                emit(&ovl_l, px, py, sx, sy, c, I * 0.85f, sg, 1, 1, 0);
                emit(&ovl_l, px, py, sx, sy, cg, I * 0.07f, sg * 4.5f, 1, 1, 0);
            }
            px = sx; py = sy;
        }
        /* corners */
        float sx, sy;
        to_px(P, pc->type == P_SEG ? pc->ax : 0, pc->type == P_SEG ? pc->ay : 0, &sx, &sy);
        if (pc->type == P_SEG) emit(&ovl_l, sx, sy, sx, sy, c, I * 0.85f, sg, 1, 1, 1);
    }
    if (P->T.pc[0].type == P_ELL) {
        /* the foci */
        double f = sqrt(ELL_A * ELL_A - ELL_B * ELL_B);
        for (int s = -1; s <= 1; s += 2) {
            float sx, sy;
            to_px(P, s * f, 0, &sx, &sy);
            emit(&ovl_l, sx, sy, sx, sy, c, I * 0.55f, sg * 1.6f, 1, 1, 1);
        }
    }
}

/* the last stretch of path, back from the head: a fading trail */
static void trail(const Panel *P, const Ball *b, double L, double len, const float *c, float I, float sg) {
    double hx = b->px + (L - b->L0) * b->dx, hy = b->py + (L - b->L0) * b->dy;
    double back = 0;
    float x0, y0, x1, y1;
    to_px(P, hx, hy, &x0, &y0);
    for (int k = 1; k <= NH && back < len; k++) {
        int j = (b->hh - k) % NH;
        if (b->hh - k < 0) break;
        double bx = b->hx[j], by = b->hy[j], seg = L - back - b->hl[j];
        if (seg <= 0) { continue; }
        double use = fmin(seg, len - back);
        double ex = hx + (bx - hx) * (use / seg), ey = hy + (by - hy) * (use / seg);
        to_px(P, ex, ey, &x1, &y1);
        float f0 = (float)(1 - back / len), f1 = (float)(1 - (back + use) / len);
        emit(&ovl_l, x0, y0, x1, y1, c, I, sg, f0 * f0, f1 * f1, 0);
        back += use;
        hx = bx; hy = by; x0 = x1; y0 = y1;
    }
}

static void side_frame(const Panel *P, float I) {
    float c[3];
    pal_fixed(pal.frame, c);
    const float *s = P->side;
    float sg = P->sigma * 0.8f;
    emit(&ovl_l, s[0], s[1], s[2], s[1], c, I, sg, 1, 1, 0);
    emit(&ovl_l, s[2], s[1], s[2], s[3], c, I, sg, 1, 1, 0);
    emit(&ovl_l, s[2], s[3], s[0], s[3], c, I, sg, 1, 1, 0);
    emit(&ovl_l, s[0], s[3], s[0], s[1], c, I, sg, 1, 1, 0);
    if (P->spec->side != S_PHASE) return;
    /* where one piece of the wall gives way to the next */
    float tk = 0.012f * min_dim;
    for (int i = 1; i < P->T.n; i++) {
        float x = s[0] + (float)(P->T.pc[i].s0 / P->T.L) * (s[2] - s[0]);
        emit(&ovl_l, x, s[3], x, s[3] + tk, c, I, sg, 1, 1, 0);
        emit(&ovl_l, x, s[1], x, s[1] - tk, c, I, sg, 1, 1, 0);
    }
    /* cos θ = 0 (the normal), dotted */
    float ym = (s[1] + s[3]) / 2, step = 0.008f * min_dim;
    for (float x = s[0] + step; x < s[2] - step * 0.5f; x += step * 2)
        emit(&ovl_l, x, ym, x + step * 0.5f, ym, c, I * 0.6f, sg, 1, 1, 0);
    emit(&ovl_l, s[0] - tk, ym, s[0], ym, c, I, sg, 1, 1, 0);
}

/* clip the line x = p + t d (t in [t0, t1]) to the box; returns 0 if outside */
static int clip(double px, double py, double dx, double dy, double *t0, double *t1, const double *bx) {
    double lo = *t0, hi = *t1;
    for (int a = 0; a < 2; a++) {
        double p = a ? py : px, d = a ? dy : dx, mn = bx[a], mx = bx[a + 2];
        if (fabs(d) < 1e-15) { if (p < mn || p > mx) return 0; continue; }
        double u0 = (mn - p) / d, u1 = (mx - p) / d;
        if (u0 > u1) { double t = u0; u0 = u1; u1 = t; }
        if (u0 > lo) lo = u0;
        if (u1 < hi) hi = u1;
    }
    if (lo >= hi) return 0;
    *t0 = lo; *t1 = hi;
    return 1;
}

/* The unfolding: reflect the table instead of the ball and the orbit is a
 * straight line through the lattice of reflected triangles. The view backs
 * off as the line grows, then follows its head. */
static void unfold(const Panel *P, double L, float I) {
    const float *s = P->side;
    float pw = s[2] - s[0], ph = s[3] - s[1];
    double S = TRI_S, hgt = S * sqrt(3) / 2;
    double emin = 3 * S, emax = S * fmin(pw, ph) / (0.022 * min_dim);
    double ext = fmin(fmax(L * 1.15 + 2 * S, emin), emax);
    double su = fmin(pw, ph) / ext;
    double kfol = 0.40 * fmin(pw, ph) / su;
    double along = fmax(L / 2, L - kfol);
    double cx = P->ux + P->udx * along, cy = P->uy + P->udy * along;
    if (L * 1.15 + 2 * S < emin) { cx = P->ux + P->udx * L / 2; cy = P->uy + P->udy * L / 2; }
    double bx[4] = {cx - pw / 2 / su, cy - ph / 2 / su, cx + pw / 2 / su, cy + ph / 2 / su};
#define UPX(x_, y_, ox_, oy_) do { ox_ = s[0] + pw / 2 + (float)(((x_) - cx) * su); oy_ = s[1] + ph / 2 - (float)(((y_) - cy) * su); } while (0)
    float c[3], cl[3];
    pal_fixed(pal.frame, c);
    /* three families of lines, parallel to the sides, a height apart */
    for (int k = 0; k < 3; k++) {
        const Piece *pc = &P->T.pc[k];
        double tx = (pc->bx - pc->ax) / S, ty = (pc->by - pc->ay) / S, nx = -ty, ny = tx;
        double c0 = nx * pc->ax + ny * pc->ay;
        double lo = 1e9, hi = -1e9;
        for (int q = 0; q < 4; q++) {
            double v = nx * bx[(q & 1) * 2] + ny * bx[1 + (q & 2)] - c0;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        for (long j = (long)floor(lo / hgt); j <= (long)ceil(hi / hgt); j++) {
            double ox = pc->ax + nx * j * hgt, oy = pc->ay + ny * j * hgt, t0 = -1e9, t1 = 1e9;
            if (!clip(ox, oy, tx, ty, &t0, &t1, bx)) continue;
            float x0, y0, x1, y1;
            UPX(ox + tx * t0, oy + ty * t0, x0, y0);
            UPX(ox + tx * t1, oy + ty * t1, x1, y1);
            emit(&ovl_l, x0, y0, x1, y1, c, I * 0.45f, P->sigma * 0.8f, 1, 1, 0);
        }
    }
    /* the home triangle, brighter */
    pal_fixed(pal.bound, cl);
    for (int k = 0; k < 3; k++) {
        const Piece *pc = &P->T.pc[k];
        double t0 = 0, t1 = 1;
        if (!clip(pc->ax, pc->ay, pc->bx - pc->ax, pc->by - pc->ay, &t0, &t1, bx)) continue;
        float x0, y0, x1, y1;
        UPX(pc->ax + (pc->bx - pc->ax) * t0, pc->ay + (pc->by - pc->ay) * t0, x0, y0);
        UPX(pc->ax + (pc->bx - pc->ax) * t1, pc->ay + (pc->by - pc->ay) * t1, x1, y1);
        emit(&ovl_l, x0, y0, x1, y1, cl, I * 0.7f, P->sigma, 1, 1, 0);
    }
    /* the straight path, and its head */
    double t0 = 0, t1 = L;
    pal_col(P->b[P->uorb].u, cl);
    if (L > 0 && clip(P->ux, P->uy, P->udx, P->udy, &t0, &t1, bx)) {
        float x0, y0, x1, y1;
        UPX(P->ux + P->udx * t0, P->uy + P->udy * t0, x0, y0);
        UPX(P->ux + P->udx * t1, P->uy + P->udy * t1, x1, y1);
        emit(&ovl_l, x0, y0, x1, y1, cl, I * 0.9f, P->sigma * 1.1f, 1, 1, 0);
        emit(&ovl_l, x0, y0, x1, y1, cl, I * 0.08f, P->sigma * 5, 1, 1, 0);
    }
    float hx, hy;
    UPX(P->ux + P->udx * L, P->uy + P->udy * L, hx, hy);
    emit(&ovl_l, hx, hy, hx, hy, cl, I, P->sigma * 2.2f, 1, 1, 1);
#undef UPX
}

static void overlay(Panel *P, double tau, float I) {
    boundary(P, I);
    side_frame(P, I * 0.8f);
    double L = path_at(P, tau), rate = rate_at(P, tau);
    float c[3];
    if (P->spec->mode == M_SWARM) {
        /* the swarm: each ball a soft point with a short trail; its last
         * bounce a point in the phase portrait */
        double tl = 0.10 * P->T.mfp;
        float sg = P->sigma * 1.05f, sd = fmaxf(0.9f, 0.001f * min_dim);
        for (int i = 0; i < P->nb; i++) {
            Ball *b = &P->b[i];
            pal_col(b->u, c);
            trail(P, b, L, tl, c, I * 0.22f, sg);
            float x, y;
            to_px(P, b->px + (L - b->L0) * b->dx, b->py + (L - b->L0) * b->dy, &x, &y);
            emit(&ovl_l, x, y, x, y, c, I * 0.30f, sg * 1.3f, 1, 1, 1);
            if (b->ps >= 0) {
                phase_px(P, b->ps, b->pp, &x, &y);
                emit(&ovl_l, x, y, x, y, c, I * 0.5f, sd, 1, 1, 1);
            }
        }
        return;
    }
    /* heads, bright at a walking pace and fading as the trace speeds up */
    double tl = fmin(fmax(rate * 0.35, 0.35), 2.2) * P->T.mfp;
    float hi = (float)(1.0 / (1.0 + rate / 5.0));
    for (int i = 0; i < P->nb; i++) {
        Ball *b = &P->b[i];
        if (P->bu_dir) pal_fixed(pal.bound, c); else pal_col(b->u, c);
        trail(P, b, L, tl, c, I * (0.25f + 0.6f * hi), P->sigma * 1.1f);
        float x, y;
        to_px(P, b->px + (L - b->L0) * b->dx, b->py + (L - b->L0) * b->dy, &x, &y);
        emit(&ovl_l, x, y, x, y, c, I * (0.35f + 0.65f * hi), P->sigma * 2.4f, 1, 1, 1);
        emit(&ovl_l, x, y, x, y, c, I * 0.12f * (0.3f + hi), P->sigma * 9, 1, 1, 1);
        if (P->spec->side == S_PHASE && b->ps >= 0) {
            phase_px(P, b->ps, b->pp, &x, &y);
            emit(&ovl_l, x, y, x, y, c, I * (0.3f + 0.6f * hi), P->sigma * 2.4f, 1, 1, 1);
        }
    }
    if (P->spec->side == S_UNFOLD) unfold(P, L, I);
}

/* ---- the frame ---- */
static time_t locked_at;

static void away_str(char *buf, size_t n, double away) {
    long m = (long)(away / 60);
    if (m < 1) snprintf(buf, n, "away a moment");
    else if (m < 60) snprintf(buf, n, "away %ld min", m);
    else snprintf(buf, n, "away %ld h %02ld min", m / 60, m % 60);
}

static void render(double t, double wt, GLuint target) {
    int idx = 0;
    double tau = 0;
    const SceneSpec *s = sched(t, &idx, &tau);
    if (acc_w != W || acc_h != H) {
        acc_w = W; acc_h = H;
        mk_target(tex_acc, fbo_acc, W, H, GL_NEAREST);
        mk_target(tex_halo, fbo_halo, (W + HALO_DIV - 1) / HALO_DIV, (H + HALO_DIV - 1) / HALO_DIV, GL_LINEAR);
        sc.idx = -1;
    }
    acc_sigma_px = 2.2f * fmaxf(1.0f, fminf(W, H) / 1200.0f);
    if (idx != sc.idx || s != sc.spec || sc.built_w != W || sc.built_h != H || tau < sc.tau || tau - sc.tau > 3)
        scene_start(s, idx, tau);
    else scene_advance(tau);
    fade_now = smooth(0, FADE_IN, tau) * smooth(sc.dur, sc.dur - FADE_OUT, tau);

    /* the accumulation, toned on to the background */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_show);
    glUniform1i(U(prog_show, "uAcc"), 0);
    glUniform1i(U(prog_show, "uHalo"), 1);
    glUniform2f(U(prog_show, "uRes"), (float)W, (float)H);
    glUniform3fv(U(prog_show, "uBg"), 1, pal.bg);
    glUniform3fv(U(prog_show, "uBg2"), 1, pal.bg2);
    glUniform1f(U(prog_show, "uExp"), pal.exposure);
    glUniform1f(U(prog_show, "uHaloK"), pal.halo);
    glUniform1f(U(prog_show, "uFade"), (float)fade_now);
    glUniform1i(U(prog_show, "uInk"), pal.ink);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_halo);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_acc);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* walls, heads, swarms, side panels */
    ovl_l.n = 0;
    for (int i = 0; i < sc.np; i++) overlay(&sc.P[i], tau, 1);
    glEnable(GL_BLEND);
    if (pal.ink) glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
    else glBlendFunc(GL_ONE, GL_ONE);
    draw_list(&ovl_l, W, H, 1, 0, (float)fade_now);

    /* words */
    tvn = 0;
    float a_card = (float)(smooth(1.5, 4.5, tau) * smooth(sc.dur, sc.dur - FADE_OUT, tau));
    float ct[4] = {pal.text[0], pal.text[1], pal.text[2], 0.92f * a_card};
    float cs[4] = {pal.text[0], pal.text[1], pal.text[2], 0.62f * a_card};
    float cf[4] = {pal.frame[0], pal.frame[1], pal.frame[2], 0.9f * (float)fade_now};
    float base = H - ui_m * 0.45f;
    text(st_title, cap_x, cap_y + px_title, -1, sc.title, ct, 0);
    for (int i = 0; i < n_sub; i++) text(st_sub, cap_x, cap_y + px_title * 1.35f + px_sub * (1.0f + 1.45f * i), -1, sub_l[i], cs, 0);
    for (int i = 0; i < sc.np; i++) {
        Panel *P = &sc.P[i];
        float y = P->side[1] - 0.012f * min_dim;
        if (P->spec->side == S_PHASE) {
            text(st_small, P->side[0], y, -1, "phase portrait", cf, 0);
            text(st_small, P->side[2], y, 1, "s  ×  cos θ", cf, 0);
        } else {
            text(st_small, P->side[0], y, -1, "the unfolding", cf, 0);
            text(st_small, P->side[2], y, 1, "reflect the table, not the ball", cf, 0);
        }
        if (sc.np > 1) text(st_small, P->box[0], P->box[1] - 0.012f * min_dim, -1, P->spec->label, cf, 0);
    }
    if (is_front) {
        char tbuf[32], abuf[48], line[96];
        time_t ts = (time_t)wt;
        struct tm tm;
        localtime_r(&ts, &tm);
        strftime(tbuf, sizeof tbuf, "%H:%M", &tm);
        double away = wt - (double)locked_at;
        away_str(abuf, sizeof abuf, away < 0 ? 0 : away);
        snprintf(line, sizeof line, "%s  ·  %s", tbuf, abuf);
        float cc[4] = {pal.text[0], pal.text[1], pal.text[2], 0.55f};
        text(st_cap, W - ui_m, base, 1, line, cc, 0.5f);
    }
    if (tvn) {
        if (atlas_dirty) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glBindTexture(GL_TEXTURE_2D, tex_atlas);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            atlas_dirty = 0;
        }
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_text);
        glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(U(prog_text, "uAtlas"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)tvn * sizeof(float), tv, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    }
    glDisable(GL_BLEND);
}

/* ==================================================================== */
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
    /* a resize rebuilds the layout and replays the act at the next frame */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "billiards");
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
        "usage: billiards [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY]\n"
        "                 [--palette spectral|gold|ice|paper|neon] [--scene NAME]\n"
        "                 [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                 [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check]\n"
        "scenes:");
    for (int i = 0; i < NSOLO; i++) fprintf(stderr, " %s", SOLO[i].name);
    fprintf(stderr, "\nacross outputs:");
    for (int i = 0; i < NTRIO; i++) fprintf(stderr, " %s", TRIO[i].name);
    fprintf(stderr, "\n");
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

/* Spanning: the output under the middle of the layout is the front one (it
 * carries the time); the outputs left to right take the regular, mixed and
 * chaotic table of each trio (two outputs: regular and chaotic). */
static void pick_roles(void) {
    if (span_me < 0 || span_n < 2) { is_front = 1; slot = -1; return; }
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    is_front = span_me == front;
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == span_me) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && (span[i][1] < span[span_me][1] ||
                                               (span[i][1] == span[span_me][1] && i < span_me)))) rank++;
    }
    slot = span_n == 2 ? rank * 2 : rank % 3;
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
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-scene") && i + 1 < argc) opt_scene = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (opt_check) { check(); return 0; }
    if (span_me >= span_n) span_me = -1;
    pick_roles();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    {   /* the palette: asked for, else from the seed (gold leaf sparingly) */
        int k = -1;
        if (opt_palette) {
            for (int i = 0; i < NPAL; i++) if (!strcmp(opt_palette, PALS[i].name)) k = i;
            if (k < 0) usage();
        } else {
            uint64_t s = (uint64_t)opt_seed * 0x2545f4914f6cdd1dULL + 99;
            int tot = 0;
            for (int i = 0; i < NPAL; i++) tot += PALS[i].weight;
            int r = (int)(sm64(&s) % (uint64_t)tot);
            for (k = 0; k < NPAL - 1 && r >= PALS[k].weight; k++) r -= PALS[k].weight;
        }
        pal_set(k);
    }
    if (opt_scene) {
        int ok = 0;
        for (int i = 0; i < NSOLO; i++) ok |= !strcmp(opt_scene, SOLO[i].name);
        for (int i = 0; i < NTRIO; i++) ok |= !strcmp(opt_scene, TRIO[i].name);
        if (!ok) usage();
    }
    int offscreen = opt_shot || opt_bench > 0;
    fonts_init();

    /* --epoch: when the show started, shared by every instance; a --shot
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
        XStoreName(dpy, win, "Billiards");
        XClassHint chint = {"billiards", "Billiards"};
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
            /* with --bench N too: arrive at --at frame by frame, from N/60 s
             * before it (the live path, decay ticks and all) */
            for (int i = opt_bench; i > 0; i--) render(opt_at - i / 60.0, wt - i / 60.0, fbo);
            render(opt_at, wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            if (n_escape) fprintf(stderr, "billiards: %ld rays found no wall\n", n_escape);
            return 0;
        }
        double t0 = now();
        render(opt_at, wall(), fbo);
        glFinish();
        double tb = now() - t0, sum = 0, worst = 0, cpu = 0, gpu = 0, gmin = 1e9;
        GLuint q;
        glGenQueries(1, &q);
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = opt_at + (i + 1) / 60.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, wall(), fbo);
            glEndQuery(GL_TIME_ELAPSED);
            cpu += now() - f0;
            glFinish();
            double ms = (now() - f0) * 1000;
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            gpu += ns * 1e-6;
            if (ns * 1e-6 < gmin) gmin = ns * 1e-6;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("billiards: %dx%d, %d frames: avg %.3f ms, worst %.3f ms per frame (act set up and caught up in %.1f ms)\n",
               W, H, opt_bench, sum / opt_bench, worst, tb * 1000);
        printf("billiards: of which CPU %.3f ms; GPU time avg %.3f ms, best %.3f ms\n", cpu * 1000 / opt_bench,
               gpu / opt_bench, gmin);
        printf("billiards: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
