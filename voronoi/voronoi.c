/*
 * voronoi — Lloyd's relaxation as a lock screen: a Voronoi diagram that
 * settles, cell by cell, into a honeycomb, is stirred up, and settles again.
 *
 * Several hundred to a few thousand sites (--cells: how many fall on the
 * front screen). Their Voronoi diagram is exact: a Delaunay triangulation on
 * the CPU (a port of delaunator's sweep-hull), each cell drawn as a fan of
 * triangles that carry the bisector lines of the cell's edges, so every
 * pixel knows its exact distance to the cell wall: the edges are crisp,
 * anti-aliased and can be given any profile (lead came, wax walls, ink,
 * cracks). The relaxation runs over a margin past the screens, so the
 * squashed boundary row of the tessellation is out of sight.
 *
 * Relaxation: every half-second tick each site moves to the centroid of its
 * cell (clipped to the screen), and the sites glide between ticks on a
 * Catmull-Rom spline through the iterates, so the motion is smooth. Over a
 * few dozen steps the cells even out into a centroidal Voronoi tessellation,
 * a near-hexagonal honeycomb. Then a disturbance arrives and it begins again:
 *
 *   ripple     a ring wave rolls out from a point and leaves rings behind
 *   swirl      one or two vortices twist the honeycomb
 *   wind       a gust front sweeps across, dragging sites in eddies
 *   cells      a cluster of cells divides (new seeds dropped in) while
 *              another cluster fuses pairwise
 *   text       the time (hours over minutes) or the host name becomes a
 *              density: cells under it bud new seeds, written left to right,
 *              and weighted Lloyd (weighted Voronoi stippling) holds them
 *              there, the small cells drawing the letters; then the budded
 *              cells fuse back into their neighbours and it dissolves
 *   dissolve   every few minutes the honeycomb melts into a fresh random
 *              scatter (which is also what bounds the replay below)
 *
 * Divisions and fusions stay continuous: a new site starts on its parent
 * (the two cells together are exactly the parent's) and glides apart while
 * the line between them fades in; a fusing site glides onto its partner.
 *
 * Now and then the relaxation shows its workings: the Delaunay dual, the
 * centroid each site is heading for, or the cells shaded by their area.
 *
 * Renderings (--palette, else picked from the seed):
 *   glass      stained glass: jewel colours lit from behind, hammered glass,
 *              bevelled lead came
 *   giraffe    reticulated giraffe: rounded liver-brown patches on cream
 *   crackle    celadon crackle glaze: fine dark cracks and crazing
 *   blueprint  white lines on blueprint blue, centroids marked
 *   paper      ink on cream paper, with the odd watercolour wash
 *   honey      honeycomb: wax walls, amber honey, capped cells
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch. Time runs in 200 s eras of five cycles (dissolve, two
 * disturbances, text, untext; --schedule prints them); each era relaxes
 * from a canonical random scatter, so a (re)started instance replays at most
 * two eras of Lloyd steps to catch up and lands on exactly the frames a
 * running one shows.
 *
 * Spanning (like flightsim): run one instance per monitor with the same
 * --seed, --epoch and --layout (every output's logical rect), and --output =
 * this one's index. The sites live in wall coordinates over the layout's
 * bounding box and every instance runs the same simulation, so the diagram is
 * one continuous sheet across the bezels; each triangulates and draws just
 * the sites within a few cells of its own slice. The
 * cell size comes from the "front" output (the one under the middle of the
 * layout), which also carries the caption and the time when it is drawn in
 * cells.
 *
 * Same plumbing as flightsim: Xlib + GLX for --window / --root, a native
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
static int opt_cells = 650;            /* sites on the front screen */
static double opt_fps = 60;
static const char *opt_font = "Inter";
static const char *opt_palette;
static const char *opt_text;           /* --text: what the host-name cycles spell instead */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the simulation starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "voronoi: %s\n", what); exit(1); }

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
static int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
/* smootherstep: zero velocity and acceleration at both ends */
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

/* value noise in [0, 1) on the integer lattice, smooth in between */
static double lat(int x, int y, uint32_t salt) {
    uint64_t h = mix64(((uint64_t)(uint32_t)x << 32 | (uint32_t)y) ^ ((uint64_t)salt << 17) ^ opt_seed);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}
static double vnoise(double x, double y, uint32_t salt) {
    double fx = floor(x), fy = floor(y);
    int ix = (int)fx, iy = (int)fy;
    double u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    double a = lat(ix, iy, salt), b = lat(ix + 1, iy, salt), c = lat(ix, iy + 1, salt), d = lat(ix + 1, iy + 1, salt);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}

/* ---- the live details for the caption ---- */
static char host[64];
static time_t locked_at;

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else now */
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
 * World units are logical layout pixels (y down). The domain is the layout's
 * bounding box (or, standalone, the screen); the front rect is the output
 * under the middle of the layout. */
static double bx0, by0, bx1, by1, bw, bh;
static double fr[4];                   /* front output: x, y, w, h */
static double sp;                      /* mean site spacing: a settled cell is ~sp across */
static int N0;                         /* sites over the whole domain */

/* the output under the middle of the layout (flightsim's rule) */
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

/* standalone: the domain is the screen the saver starts on */
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
    int cells = opt_cells < 20 ? 20 : opt_cells;
    sp = sqrt(fr[2] * fr[3] / cells);
    /* relax over a margin past the screens, so the boundary row of a
     * centroidal tessellation (squashed against the edge) is out of sight */
    bx0 -= 1.5 * sp; by0 -= 1.5 * sp; bx1 += 1.5 * sp; by1 += 1.5 * sp;
    bw = bx1 - bx0; bh = by1 - by0;
    N0 = (int)lround(bw * bh / (sp * sp));
    if (N0 < 8) N0 = 8;
}

/* a random point well inside some output (weighted by area) */
static void visible_point(Rng *r, double *x, double *y, double inset) {
    double rx = fr[0], ry = fr[1], rw = fr[2], rh = fr[3];
    if (span_me >= 0 && span_n > 1) {
        double tot = 0, pick;
        for (int i = 0; i < span_n; i++) tot += (double)span[i][2] * span[i][3];
        pick = rnd(r) * tot;
        for (int i = 0; i < span_n; i++) {
            pick -= (double)span[i][2] * span[i][3];
            if (pick <= 0 || i == span_n - 1) { rx = span[i][0]; ry = span[i][1]; rw = span[i][2]; rh = span[i][3]; break; }
        }
    }
    *x = rx + rw * (inset + (1 - 2 * inset) * rnd(r));
    *y = ry + rh * (inset + (1 - 2 * inset) * rnd(r));
}

/* ==== Delaunay: a port of delaunator (Mapbox, ISC) ==== */
typedef struct {
    int cap, n, nt, hsz, hstart;
    const double *p;
    int *tri, *half, *hprev, *hnext, *htri, *hhash, *ids, *inedge;
    double *dist, *ccx, *ccy, cx, cy;
} Del;
static Del D;

#define NXT(e) ((e) % 3 == 2 ? (e) - 2 : (e) + 1)
#define PRV(e) ((e) % 3 == 0 ? (e) + 2 : (e) - 1)

static void del_alloc(int cap) {
    int mt = 6 * cap + 16;
    D.cap = cap;
    D.tri = malloc(sizeof(int) * mt); D.half = malloc(sizeof(int) * mt);
    D.hprev = malloc(sizeof(int) * cap); D.hnext = malloc(sizeof(int) * cap);
    D.htri = malloc(sizeof(int) * cap); D.ids = malloc(sizeof(int) * cap);
    D.inedge = malloc(sizeof(int) * cap); D.dist = malloc(sizeof(double) * cap);
    D.hhash = malloc(sizeof(int) * ((int)ceil(sqrt(cap)) + 1));
    D.ccx = malloc(sizeof(double) * (2 * cap + 8)); D.ccy = malloc(sizeof(double) * (2 * cap + 8));
}

static void link_he(int a, int b) { D.half[a] = b; if (b != -1) D.half[b] = a; }
static int add_tri(int i0, int i1, int i2, int a, int b, int c) {
    int t = D.nt;
    D.tri[t] = i0; D.tri[t + 1] = i1; D.tri[t + 2] = i2;
    link_he(t, a); link_he(t + 1, b); link_he(t + 2, c);
    D.nt += 3;
    return t;
}
static int orient_neg(double px, double py, double qx, double qy, double rx, double ry) {
    return (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0;
}
static int incircle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) {
    double dx = ax - px, dy = ay - py, ex = bx - px, ey = by - py, fx = cx - px, fy = cy - py;
    double ap = dx * dx + dy * dy, bp = ex * ex + ey * ey, cp = fx * fx + fy * fy;
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0;
}
static double circumr2(double ax, double ay, double bx, double by, double cx, double cy) {
    double dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
    double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, den = dx * ey - dy * ex;
    if (den == 0) return 1e300;
    double d = 0.5 / den, x = (ey * bl - dy * cl) * d, y = (dx * cl - ex * bl) * d;
    return x * x + y * y;
}
static void circumc(double ax, double ay, double bx, double by, double cx, double cy, double *ox, double *oy) {
    double dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
    double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, den = dx * ey - dy * ex;
    if (fabs(den) < 1e-12 * (bl + cl)) { *ox = (ax + bx + cx) / 3; *oy = (ay + by + cy) / 3; return; }
    double d = 0.5 / den;
    *ox = ax + (ey * bl - dy * cl) * d;
    *oy = ay + (dx * cl - ex * bl) * d;
}
static int hkey(double x, double y) {
    double dx = x - D.cx, dy = y - D.cy, s = fabs(dx) + fabs(dy);
    if (s == 0) return 0;
    double p = dx / s, a = (dy > 0 ? 3 - p : 1 + p) / 4;
    int k = (int)floor(a * D.hsz) % D.hsz;
    return k < 0 ? k + D.hsz : k;
}
static int legalize(int a) {
    static int stack[2048];
    int i = 0, ar = 0;
    const double *p = D.p;
    for (;;) {
        int b = D.half[a], a0 = a - a % 3;
        ar = a0 + (a + 2) % 3;
        if (b == -1) {
            if (i == 0) break;
            a = stack[--i];
            continue;
        }
        int b0 = b - b % 3, al = a0 + (a + 1) % 3, bl = b0 + (b + 2) % 3;
        int p0 = D.tri[ar], pr = D.tri[a], pl = D.tri[al], p1 = D.tri[bl];
        if (incircle(p[2 * p0], p[2 * p0 + 1], p[2 * pr], p[2 * pr + 1], p[2 * pl], p[2 * pl + 1], p[2 * p1], p[2 * p1 + 1])) {
            D.tri[a] = p1; D.tri[b] = p0;
            int hbl = D.half[bl];
            if (hbl == -1) {
                int e = D.hstart;
                do {
                    if (D.htri[e] == bl) { D.htri[e] = a; break; }
                    e = D.hprev[e];
                } while (e != D.hstart);
            }
            link_he(a, hbl);
            link_he(b, D.half[ar]);
            link_he(ar, bl);
            int br = b0 + (b + 1) % 3;
            if (i < 2048) stack[i++] = br;
        } else {
            if (i == 0) break;
            a = stack[--i];
        }
    }
    return ar;
}
static void sort_ids(int *ids, const double *d, int lo, int hi) {
    while (hi - lo > 16) {
        int mid = (lo + hi) >> 1;
        double a = d[ids[lo]], b = d[ids[mid]], c = d[ids[hi]];
        double pv = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
        int i = lo, j = hi;
        while (i <= j) {
            while (d[ids[i]] < pv) i++;
            while (d[ids[j]] > pv) j--;
            if (i <= j) { int t = ids[i]; ids[i] = ids[j]; ids[j] = t; i++; j--; }
        }
        if (j - lo < hi - i) { sort_ids(ids, d, lo, j); lo = i; }
        else { sort_ids(ids, d, i, hi); hi = j; }
    }
    for (int i = lo + 1; i <= hi; i++) {
        int t = ids[i];
        double v = d[t];
        int j = i - 1;
        while (j >= lo && d[ids[j]] > v) { ids[j + 1] = ids[j]; j--; }
        ids[j + 1] = t;
    }
}

/* triangulate n points (x, y pairs); then the circumcentres and, per point,
 * one outgoing half-edge (-1 for a skipped duplicate) */
static void del_build(const double *p, int n) {
    D.p = p; D.n = n; D.nt = 0;
    D.hsz = (int)ceil(sqrt(n));
    double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
    for (int i = 0; i < n; i++) {
        double x = p[2 * i], y = p[2 * i + 1];
        minx = fmin(minx, x); miny = fmin(miny, y);
        maxx = fmax(maxx, x); maxy = fmax(maxy, y);
        D.ids[i] = i;
        D.inedge[i] = -1;
    }
    double cx = (minx + maxx) / 2, cy = (miny + maxy) / 2, md = 1e300;
    int i0 = 0, i1 = 0, i2 = 0;
    for (int i = 0; i < n; i++) {
        double d = (p[2 * i] - cx) * (p[2 * i] - cx) + (p[2 * i + 1] - cy) * (p[2 * i + 1] - cy);
        if (d < md) { i0 = i; md = d; }
    }
    double i0x = p[2 * i0], i0y = p[2 * i0 + 1];
    md = 1e300;
    for (int i = 0; i < n; i++) {
        if (i == i0) continue;
        double d = (p[2 * i] - i0x) * (p[2 * i] - i0x) + (p[2 * i + 1] - i0y) * (p[2 * i + 1] - i0y);
        if (d < md && d > 0) { i1 = i; md = d; }
    }
    double i1x = p[2 * i1], i1y = p[2 * i1 + 1], mr = 1e300;
    for (int i = 0; i < n; i++) {
        if (i == i0 || i == i1) continue;
        double r = circumr2(i0x, i0y, i1x, i1y, p[2 * i], p[2 * i + 1]);
        if (r < mr) { i2 = i; mr = r; }
    }
    if (mr >= 1e300) return;
    double i2x = p[2 * i2], i2y = p[2 * i2 + 1];
    if (orient_neg(i0x, i0y, i1x, i1y, i2x, i2y)) {
        int t = i1; i1 = i2; i2 = t;
        double tx = i1x, ty = i1y; i1x = i2x; i1y = i2y; i2x = tx; i2y = ty;
    }
    circumc(i0x, i0y, i1x, i1y, i2x, i2y, &D.cx, &D.cy);
    for (int i = 0; i < n; i++)
        D.dist[i] = (p[2 * i] - D.cx) * (p[2 * i] - D.cx) + (p[2 * i + 1] - D.cy) * (p[2 * i + 1] - D.cy);
    sort_ids(D.ids, D.dist, 0, n - 1);
    D.hstart = i0;
    D.hnext[i0] = D.hprev[i2] = i1;
    D.hnext[i1] = D.hprev[i0] = i2;
    D.hnext[i2] = D.hprev[i1] = i0;
    D.htri[i0] = 0; D.htri[i1] = 1; D.htri[i2] = 2;
    for (int k = 0; k < D.hsz; k++) D.hhash[k] = -1;
    D.hhash[hkey(i0x, i0y)] = i0;
    D.hhash[hkey(i1x, i1y)] = i1;
    D.hhash[hkey(i2x, i2y)] = i2;
    add_tri(i0, i1, i2, -1, -1, -1);
    double xp = 0, yp = 0;
    for (int k = 0; k < n; k++) {
        int i = D.ids[k];
        double x = p[2 * i], y = p[2 * i + 1];
        if (k > 0 && fabs(x - xp) <= 1e-9 && fabs(y - yp) <= 1e-9) continue;
        xp = x; yp = y;
        if (i == i0 || i == i1 || i == i2) continue;
        int start = 0, key = hkey(x, y);
        for (int j = 0; j < D.hsz; j++) {
            start = D.hhash[(key + j) % D.hsz];
            if (start != -1 && start != D.hnext[start]) break;
        }
        start = D.hprev[start];
        int e = start, q;
        while (q = D.hnext[e], !orient_neg(x, y, p[2 * e], p[2 * e + 1], p[2 * q], p[2 * q + 1])) {
            e = q;
            if (e == start) { e = -1; break; }
        }
        if (e == -1) continue;
        int t = add_tri(e, i, D.hnext[e], -1, -1, D.htri[e]);
        D.htri[i] = legalize(t + 2);
        D.htri[e] = t;
        int nn = D.hnext[e];
        while (q = D.hnext[nn], orient_neg(x, y, p[2 * nn], p[2 * nn + 1], p[2 * q], p[2 * q + 1])) {
            t = add_tri(nn, i, q, D.htri[i], -1, D.htri[nn]);
            D.htri[i] = legalize(t + 2);
            D.hnext[nn] = nn;
            nn = q;
        }
        if (e == start) {
            while (q = D.hprev[e], orient_neg(x, y, p[2 * q], p[2 * q + 1], p[2 * e], p[2 * e + 1])) {
                t = add_tri(q, i, e, -1, D.htri[e], D.htri[q]);
                legalize(t + 2);
                D.htri[q] = t;
                D.hnext[e] = e;
                e = q;
            }
        }
        D.hstart = D.hprev[i] = e;
        D.hnext[e] = D.hprev[nn] = i;
        D.hnext[i] = nn;
        D.hhash[hkey(x, y)] = i;
        D.hhash[hkey(p[2 * e], p[2 * e + 1])] = e;
    }
    for (int e = 0; e < D.nt; e++) {
        int v = D.tri[e];
        if (D.inedge[v] == -1 || D.half[e] == -1) D.inedge[v] = e;
    }
    for (int t = 0; t < D.nt / 3; t++) {
        int a = D.tri[3 * t], b = D.tri[3 * t + 1], c = D.tri[3 * t + 2];
        circumc(p[2 * a], p[2 * a + 1], p[2 * b], p[2 * b + 1], p[2 * c], p[2 * c + 1], &D.ccx[t], &D.ccy[t]);
    }
}

/* the Voronoi cell of point i: its vertices (circumcentres, in order) and
 * the neighbour across the edge ending at each; 0 if it has no closed cell */
#define MAXV 64
static int cell_of(int i, double *vx, double *vy, int *nb) {
    int e0 = D.inedge[i];
    if (e0 < 0) return 0;
    int e = e0, m = 0;
    do {
        if (m == MAXV) return 0;
        int t = e / 3;
        vx[m] = D.ccx[t]; vy[m] = D.ccy[t];
        nb[m] = D.tri[NXT(e)];
        m++;
        e = D.half[PRV(e)];
        if (e < 0) return 0;
    } while (e != e0);
    return m;
}

static int clip1(const double *ix, const double *iy, int n, double *ox, double *oy, int axis, double v, double sgn) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int j = i + 1 == n ? 0 : i + 1;
        double a = ((axis ? iy[i] : ix[i]) - v) * sgn, b = ((axis ? iy[j] : ix[j]) - v) * sgn;
        if (a >= 0) { ox[m] = ix[i]; oy[m] = iy[i]; m++; }
        if ((a >= 0) != (b >= 0)) {
            double t = a / (a - b);
            ox[m] = ix[i] + t * (ix[j] - ix[i]);
            oy[m] = iy[i] + t * (iy[j] - iy[i]);
            m++;
        }
    }
    return m;
}
/* clip a convex polygon to the domain, in place; its area and centroid */
static double poly_clip_ac(double *x, double *y, int n, double *cx, double *cy) {
    double tx[MAXV + 8], ty[MAXV + 8];
    n = clip1(x, y, n, tx, ty, 0, bx0, 1);
    n = clip1(tx, ty, n, x, y, 0, bx1, -1);
    n = clip1(x, y, n, tx, ty, 1, by0, 1);
    n = clip1(tx, ty, n, x, y, 1, by1, -1);
    double a = 0, sx = 0, sy = 0;
    for (int i = 0; i < n; i++) {
        int j = i + 1 == n ? 0 : i + 1;
        double c = x[i] * y[j] - x[j] * y[i];
        a += c; sx += (x[i] + x[j]) * c; sy += (y[i] + y[j]) * c;
    }
    if (fabs(a) < 1e-12) { *cx = n ? x[0] : 0; *cy = n ? y[0] : 0; return 0; }
    *cx = sx / (3 * a); *cy = sy / (3 * a);
    return fabs(a) * 0.5;
}

/* ==== the simulation ==== */
#define TICK 0.5                       /* seconds per Lloyd step */
#define ERA_TICKS 400
#define RING 8
#define NFAR 8
enum { K_START, K_DISSOLVE, K_RIPPLE, K_SWIRL, K_WIND, K_CELLS, K_TEXT_TIME, K_TEXT_HOST, K_UNTEXT };
#define NCYC 5
static const char *kind_name[] = {"start", "dissolve", "ripple", "swirl", "wind", "cells", "text (time)", "text (host)", "untext"};
static int kind_len(int k) { return k <= K_DISSOLVE ? 80 : k == K_UNTEXT ? 50 : k >= K_TEXT_TIME ? 70 : 100; }
static int kind_dist(int k) {
    switch (k) {
    case K_DISSOLVE: return 14;
    case K_RIPPLE: return 16;
    case K_SWIRL: return 12;
    case K_WIND: return 14;
    case K_CELLS: return 12;
    case K_TEXT_TIME: case K_TEXT_HOST: return 10;
    case K_UNTEXT: return 12;
    default: return 0;
    }
}

/* an era: a dissolve into a fresh scatter, then two disturbances and a
 * piece of text (written, then dissolved) in a seeded order;
 * 80 + 100 + 100 + 70 + 50 ticks */
static void era_plan(long era, int kind[NCYC]) {
    Rng r = rng_for((uint64_t)era, 7, 0);
    kind[0] = era ? K_DISSOLVE : K_START;
    int tp = 1 + (int)(rnd(&r) * 3);
    int mo[4] = {K_RIPPLE, K_SWIRL, K_WIND, K_CELLS};
    for (int i = 3; i > 0; i--) {
        int j = (int)(rnd(&r) * (i + 1));
        int t = mo[i]; mo[i] = mo[j]; mo[j] = t;
    }
    int mi = 0;
    for (int c = 1; c < NCYC; c++) {
        if (c == tp) kind[c] = ((era + (opt_seed & 1)) & 1) ? K_TEXT_HOST : K_TEXT_TIME;
        else if (c == tp + 1) kind[c] = K_UNTEXT;
        else kind[c] = mo[mi++];
    }
}

typedef struct { long era, start; int ci, kind, len, dist; } Where;
static Where locate(long k) {
    Where w;
    if (k < 0) k = 0;
    w.era = k / ERA_TICKS;
    int kinds[NCYC];
    era_plan(w.era, kinds);
    long s = w.era * ERA_TICKS;
    for (int c = 0; c < NCYC; c++) {
        int L = kind_len(kinds[c]);
        if (k < s + L || c == NCYC - 1) {
            w.ci = c; w.kind = kinds[c]; w.len = L; w.dist = kind_dist(kinds[c]); w.start = s;
            break;
        }
        s += L;
    }
    return w;
}

/* the sites: slots that live and die (cells divide and fuse) */
static struct {
    int cap, n;                        /* slots 0..n-1 in use, some dead */
    double *x, *y;
    uint32_t *key;                     /* colour */
    unsigned char *alive;
    unsigned char *extra;              /* budded to write text: fused away after */
} S;
static double *ring_x[RING], *ring_y[RING];
static long sim_k = -1;                /* the last tick computed */

/* the current cycle and its disturbance */
enum { M_NONE, M_MOVE, M_BIRTH, M_DEATH };
static struct {
    Where w;
    int planned, settled;
    int nc;
    double c[3][2], amp[3], sig[3], del[3];
    double dir[2], A, B, kw, rmax, u0, u1, width;
    double W;                          /* text weight */
} cy = {.w = {.start = -1}};
static double *pl_tx, *pl_ty, *pl_dl;  /* per slot: path end, start delay */
static int *pl_twin;
static unsigned char *pl_mode;
static uint32_t *pl_kA, *pl_kB;        /* colour at the start and end of the disturbance */
static double *qx, *qy;                /* an era's scatter */
static uint32_t *qk;

/* text density: grid cells with weight, in row order */
static double gs;                      /* density grid step */
static int mk_n, mk_cap;
static double *mk_x, *mk_y, *mk_w;
static double mk_sum;

/* scratch for the Lloyd step and the frame */
static int *g_slot;                    /* compact index -> slot */
static int *g_ofs, *g_cnt, *g_nb;      /* neighbour lists (CSR) */
static double *g_pts, *g_ax, *g_ay, *g_a, *g_mx, *g_my, *g_m;

static void sim_alloc(void) {
    S.cap = 3 * N0 + 256;
    int c = S.cap;
    S.x = calloc(c, sizeof(double)); S.y = calloc(c, sizeof(double));
    S.key = calloc(c, sizeof(uint32_t)); S.alive = calloc(c, 1); S.extra = calloc(c, 1);
    for (int i = 0; i < RING; i++) { ring_x[i] = calloc(c, sizeof(double)); ring_y[i] = calloc(c, sizeof(double)); }
    pl_tx = calloc(c, sizeof(double)); pl_ty = calloc(c, sizeof(double)); pl_dl = calloc(c, sizeof(double));
    pl_twin = calloc(c, sizeof(int)); pl_mode = calloc(c, 1);
    pl_kA = calloc(c, sizeof(uint32_t)); pl_kB = calloc(c, sizeof(uint32_t));
    qx = calloc(N0, sizeof(double)); qy = calloc(N0, sizeof(double)); qk = calloc(N0, sizeof(uint32_t));
    g_slot = calloc(c + NFAR, sizeof(int));
    g_ofs = calloc(c + NFAR, sizeof(int)); g_cnt = calloc(c + NFAR, sizeof(int));
    g_nb = calloc((size_t)(c + NFAR) * 8 + MAXV, sizeof(int));
    g_pts = calloc(2 * (c + NFAR), sizeof(double));
    g_ax = calloc(c, sizeof(double)); g_ay = calloc(c, sizeof(double)); g_a = calloc(c, sizeof(double));
    g_mx = calloc(c, sizeof(double)); g_my = calloc(c, sizeof(double)); g_m = calloc(c, sizeof(double));
    del_alloc(c + NFAR);
}

static void gen_scatter(long era) {
    Rng r = rng_for((uint64_t)era, 3, 0);
    for (int i = 0; i < N0; i++) {
        qx[i] = bx0 + bw * rnd(&r);
        qy[i] = by0 + bh * rnd(&r);
        qk[i] = (uint32_t)(rnd(&r) * 4294967296.0);
    }
}

/* the canonical state an era relaxes from */
static void sim_reset(long era) {
    gen_scatter(era);
    S.n = N0;
    memset(S.extra, 0, S.cap);
    for (int i = 0; i < N0; i++) { S.x[i] = qx[i]; S.y[i] = qy[i]; S.key[i] = qk[i]; S.alive[i] = 1; }
}

/* the far ring of points that closes every real cell */
static void put_far(double *p, int m) {
    double cx = (bx0 + bx1) / 2, cy2 = (by0 + by1) / 2, R = 12 * fmax(bw, bh);
    for (int k = 0; k < NFAR; k++) {
        double a = k * (2 * M_PI / NFAR) + 0.3;
        p[2 * (m + k)] = cx + R * cos(a);
        p[2 * (m + k) + 1] = cy2 + R * sin(a);
    }
}

/* triangulate the live sites at positions (px, py) per slot; fills g_slot
 * and the CSR neighbour lists; returns how many */
static int triangulate(const double *px, const double *py, const double *keep) {
    int m = 0;
    for (int s = 0; s < S.n; s++)
        if (S.alive[s] && (!keep || (px[s] > keep[0] && px[s] < keep[2] && py[s] > keep[1] && py[s] < keep[3]))) { g_slot[m] = s; g_pts[2 * m] = px[s]; g_pts[2 * m + 1] = py[s]; m++; }
    put_far(g_pts, m);
    del_build(g_pts, m + NFAR);
    return m;
}

/* the nearest site to (x, y), walking the Delaunay graph from site i */
static int walk_nearest(int i, double x, double y) {
    for (;;) {
        double bd = (g_pts[2 * i] - x) * (g_pts[2 * i] - x) + (g_pts[2 * i + 1] - y) * (g_pts[2 * i + 1] - y);
        int best = i;
        for (int k = 0; k < g_cnt[i]; k++) {
            int j = g_nb[g_ofs[i] + k];
            double d = (g_pts[2 * j] - x) * (g_pts[2 * j] - x) + (g_pts[2 * j + 1] - y) * (g_pts[2 * j + 1] - y);
            if (d < bd) { bd = d; best = j; }
        }
        if (best == i) return i;
        i = best;
    }
}

/* one Lloyd step (weighted by the text density at weight wt); returns the
 * largest move */
static double lloyd(double wt, double alpha) {
    int m = triangulate(S.x, S.y, NULL);
    double vx[MAXV + 8], vy[MAXV + 8];
    int nb[MAXV], o = 0;
    for (int i = 0; i < m; i++) {
        int k = cell_of(i, vx, vy, nb);
        g_ofs[i] = o; g_cnt[i] = 0;
        for (int j = 0; j < k; j++)
            if (nb[j] < m) g_nb[o + g_cnt[i]++] = nb[j];
        o += g_cnt[i];
        g_a[i] = k ? poly_clip_ac(vx, vy, k, &g_ax[i], &g_ay[i]) : 0;
        g_mx[i] = g_my[i] = g_m[i] = 0;
    }
    if (wt > 0 && mk_n) {
        int cur = 0;
        double cell = gs * gs * wt;
        for (int q = 0; q < mk_n; q++) {
            cur = walk_nearest(cur, mk_x[q], mk_y[q]);
            double w = mk_w[q] * cell;
            g_m[cur] += w; g_mx[cur] += w * mk_x[q]; g_my[cur] += w * mk_y[q];
        }
    }
    double worst = 0;
    for (int i = 0; i < m; i++) {
        int s = g_slot[i];
        double mass = g_a[i] + g_m[i];
        if (mass <= 0) continue;
        double nx = (g_a[i] * g_ax[i] + g_mx[i]) / mass, ny = (g_a[i] * g_ay[i] + g_my[i]) / mass;
        double mv = hypot(nx - S.x[s], ny - S.y[s]);
        if (mv > worst) worst = mv;
        nx = S.x[s] + alpha * (nx - S.x[s]); ny = S.y[s] + alpha * (ny - S.y[s]);
        S.x[s] = fmin(fmax(nx, bx0 + 0.25), bx1 - 0.25); S.y[s] = fmin(fmax(ny, by0 + 0.25), by1 - 0.25);
    }
    return worst;
}

/* ---- text density ---- */
static FT_Library ft;
static FT_Face face_cap, face_big;

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
    snprintf(spec, sizeof spec, "%s:weight=bold", opt_font);
    face_big = load_face(spec);
}

/* set lines of text at px into a canvas; returns it (w x h, 0..255) */
static unsigned char *set_text(const char **lines, int nl, int px, int *ow, int *oh) {
    FT_Set_Pixel_Sizes(face_big, 0, (FT_UInt)px);
    int lh = (int)(px * 1.05), w = 0;
    for (int l = 0; l < nl; l++) {
        int lw = 0;
        for (const char *c = lines[l]; *c; c++)
            if (!FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_DEFAULT)) lw += (int)(face_big->glyph->advance.x >> 6);
        if (lw > w) w = lw;
    }
    int cw = w + 2 * px, ch = lh * nl + 2 * px;
    unsigned char *cv = calloc((size_t)cw * ch, 1);
    for (int l = 0; l < nl; l++) {
        int lw = 0;
        for (const char *c = lines[l]; *c; c++)
            if (!FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_DEFAULT)) lw += (int)(face_big->glyph->advance.x >> 6);
        int pen = px + (w - lw) / 2, base = px + lh * l + (int)(px * 0.85);
        for (const char *c = lines[l]; *c; c++) {
            if (FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_RENDER)) continue;
            FT_GlyphSlot g = face_big->glyph;
            for (int y = 0; y < (int)g->bitmap.rows; y++)
                for (int x = 0; x < (int)g->bitmap.width; x++) {
                    int X = pen + g->bitmap_left + x, Y = base - g->bitmap_top + y;
                    if (X < 0 || Y < 0 || X >= cw || Y >= ch) continue;
                    int v = g->bitmap.buffer[y * g->bitmap.pitch + x];
                    if (v > cv[Y * cw + X]) cv[Y * cw + X] = (unsigned char)v;
                }
            pen += (int)(g->advance.x >> 6);
        }
    }
    *ow = cw; *oh = ch;
    return cv;
}

/* the box text is fitted to: the time on the front output; the host name
 * along the band every output covers, if there is a decent one */
static void text_box(int host_text, double b[4]) {
    b[0] = fr[0]; b[1] = fr[1]; b[2] = fr[2]; b[3] = fr[3];
    if (!host_text || span_me < 0 || span_n < 2) return;
    double y0 = -1e300, y1 = 1e300;
    for (int i = 0; i < span_n; i++) { y0 = fmax(y0, span[i][1]); y1 = fmin(y1, span[i][1] + span[i][3]); }
    if (y1 - y0 < 0.4 * fr[3]) return;
    b[0] = bx0; b[1] = y0; b[2] = bw; b[3] = y1 - y0;
}

static void build_mask(int kind, long cycle_start) {
    char buf[64], l0[16], l1[16];
    const char *lines[2];
    int nl = 1;
    double box[4];
    text_box(kind == K_TEXT_HOST, box);
    if (kind == K_TEXT_TIME) {
        time_t tt = (time_t)(opt_epoch + cycle_start * TICK);
        struct tm tm;
        localtime_r(&tt, &tm);
        strftime(buf, sizeof buf, "%H:%M", &tm);
        /* the hours over the minutes, unless the box is very wide: big
         * figures are what make it legible in cells */
        if (box[2] < box[3] * 2.4) {
            strftime(l0, sizeof l0, "%H", &tm); strftime(l1, sizeof l1, "%M", &tm);
            lines[0] = l0; lines[1] = l1; nl = 2;
        } else lines[0] = buf;
    } else {
        snprintf(buf, sizeof buf, "%s", opt_text ? opt_text : host);
        for (char *c = buf; *c; c++) *c = (char)toupper((unsigned char)*c);
        lines[0] = buf;
    }
    /* measure at 100 px, then fit */
    int cw, ch;
    unsigned char *cv = set_text(lines, nl, 100, &cw, &ch);
    int x0 = cw, y0 = ch, x1 = 0, y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    free(cv);
    if (x1 <= x0) { mk_n = 0; return; }
    double fit = fmin(box[2] * 0.9 / (x1 - x0 + 1), box[3] * (nl > 1 ? 0.8 : 0.66) / (y1 - y0 + 1)) / gs;
    int px = (int)(100 * fit);
    if (px < 8) px = 8;
    cv = set_text(lines, nl, px, &cw, &ch);
    x0 = cw; y0 = ch; x1 = 0; y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    /* ink centre onto the box centre, in grid cells */
    double ox = box[0] + box[2] / 2 - (x0 + x1 + 1) * 0.5 * gs, oy = box[1] + box[3] / 2 - (y0 + y1 + 1) * 0.5 * gs;
    mk_n = 0; mk_sum = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            double v = cv[y * cw + x] / 255.0;
            if (v < 0.02) continue;
            double wx = ox + (x + 0.5) * gs, wy = oy + (y + 0.5) * gs;
            if (wx < bx0 || wx > bx1 || wy < by0 || wy > by1) continue;
            if (mk_n == mk_cap) {
                mk_cap = mk_cap ? mk_cap * 2 : 4096;
                mk_x = realloc(mk_x, mk_cap * sizeof(double)); mk_y = realloc(mk_y, mk_cap * sizeof(double));
                mk_w = realloc(mk_w, mk_cap * sizeof(double));
            }
            mk_x[mk_n] = wx; mk_y[mk_n] = wy; mk_w[mk_n] = v; mk_n++;
            mk_sum += v;
        }
    free(cv);
    /* the weight that pulls about 62% of the sites into the letters */
    double f = 0.66, tm2 = mk_sum * gs * gs;
    cy.W = tm2 > 0 ? f * bw * bh / ((1 - f) * tm2) : 0;
}

/* the text weight over a text cycle's relaxation (ticks) */
static double text_w(double r) { return r < 0 ? 0 : ease(r / 4); }

/* ---- disturbances ---- */
static int alloc_slot(void) {
    for (int s = 0; s < S.n; s++)
        if (!S.alive[s]) return s;
    if (S.n == S.cap) return -1;
    return S.n++;
}

static uint32_t hilbert(double x, double y) {
    uint32_t n = 1024, X = (uint32_t)clampi((int)((x - bx0) / bw * 1023), 0, 1023), Y = (uint32_t)clampi((int)((y - by0) / bh * 1023), 0, 1023), d = 0;
    for (uint32_t s = n / 2; s > 0; s /= 2) {
        uint32_t rx = (X & s) > 0, ry = (Y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) { X = n - 1 - X; Y = n - 1 - Y; }
            uint32_t t = X; X = Y; Y = t;
        }
    }
    return d;
}
static uint64_t *srt;
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void plan_dissolve(void) {
    int nc = 0;
    gen_scatter(cy.w.era);
    if (!srt) srt = malloc(sizeof(uint64_t) * (S.cap + N0));
    for (int s = 0; s < S.n; s++)
        if (S.alive[s]) srt[nc++] = (uint64_t)hilbert(S.x[s], S.y[s]) << 32 | (uint32_t)s;
    qsort(srt, nc, sizeof *srt, cmp_u64);
    int *cur = malloc(sizeof(int) * nc), *tg = malloc(sizeof(int) * N0), *rep = malloc(sizeof(int) * N0);
    for (int i = 0; i < nc; i++) cur[i] = (int)(srt[i] & 0xffffffffu);
    for (int i = 0; i < N0; i++) srt[i] = (uint64_t)hilbert(qx[i], qy[i]) << 32 | (uint32_t)i;
    qsort(srt, N0, sizeof *srt, cmp_u64);
    for (int i = 0; i < N0; i++) { tg[i] = (int)(srt[i] & 0xffffffffu); rep[i] = -1; }
    for (int r = 0; r < nc; r++) {
        int s = cur[r], tq = tg[(long)r * N0 / nc];
        pl_tx[s] = qx[tq]; pl_ty[s] = qy[tq]; pl_kA[s] = S.key[s]; pl_kB[s] = qk[tq];
        if (rep[tq] < 0) { rep[tq] = s; pl_mode[s] = M_MOVE; }
        else { pl_mode[s] = M_DEATH; pl_twin[s] = rep[tq]; }
    }
    for (int r = 0; r < N0; r++) {
        int tq = tg[r];
        if (rep[tq] >= 0) continue;
        int par = cur[(long)r * nc / N0], c = alloc_slot();
        if (c < 0) continue;
        S.alive[c] = 1; S.x[c] = S.x[par]; S.y[c] = S.y[par]; S.key[c] = qk[tq];
        pl_tx[c] = qx[tq]; pl_ty[c] = qy[tq]; pl_kA[c] = S.key[par]; pl_kB[c] = qk[tq];
        pl_mode[c] = M_BIRTH; pl_twin[c] = par; pl_dl[c] = 0;
        rep[tq] = c;
    }
    /* the melt spreads in patches */
    for (int s = 0; s < S.n; s++)
        if (S.alive[s]) pl_dl[s] = 0.4 * clamp01(vnoise(S.x[s] / (sp * 5), S.y[s] / (sp * 5), 41 + (uint32_t)cy.w.era) * 1.6 - 0.3);
    free(cur); free(tg); free(rep);
}

/* candidates near (x, y), nearest first */
static int near_list(double x, double y, int *out, int max, const unsigned char *used) {
    int n = 0;
    for (int s = 0; s < S.n; s++)
        if (S.alive[s] && !used[s]) {
            double d = hypot(S.x[s] - x, S.y[s] - y);
            srt[n++] = (uint64_t)(d * 1024) << 24 | (uint32_t)s;
        }
    qsort(srt, n, sizeof *srt, cmp_u64);
    if (n > max) n = max;
    for (int i = 0; i < n; i++) out[i] = (int)(srt[i] & 0xffffff);
    return n;
}

/* a cluster of cells divides; another cluster fuses in pairs */
static void plan_cells(Rng *r) {
    if (!srt) srt = malloc(sizeof(uint64_t) * (S.cap + N0));
    int M = (int)(opt_cells * 0.06);
    if (M < 12) M = 12;
    if (M > 160) M = 160;
    unsigned char *used = calloc(S.cap, 1);
    int *lst = malloc(sizeof(int) * (4 * M + 8));
    double ax, ay, bxp = 0, byp = 0;
    visible_point(r, &ax, &ay, 0.2);
    double best = -1;
    for (int k = 0; k < 6; k++) {
        double x, y;
        visible_point(r, &x, &y, 0.2);
        double d = hypot(x - ax, y - ay);
        if (d > best) { best = d; bxp = x; byp = y; }
    }
    cy.c[0][0] = ax; cy.c[0][1] = ay; cy.c[1][0] = bxp; cy.c[1][1] = byp; cy.nc = 2;
    /* divisions: the M nearest the drop, the inner ones first */
    int n = near_list(ax, ay, lst, M, used);
    double R = n ? hypot(S.x[lst[n - 1]] - ax, S.y[lst[n - 1]] - ay) + 1e-9 : 1;
    for (int i = 0; i < n; i++) used[lst[i]] = 1;
    for (int i = 0; i < n; i++) {
        int p = lst[i], c = alloc_slot();
        if (c < 0) break;
        double a = rnd(r) * 2 * M_PI, d = 0.2 * sp;
        S.alive[c] = 1; S.x[c] = S.x[p]; S.y[c] = S.y[p];
        S.key[c] = (uint32_t)(rnd(r) * 4294967296.0);
        used[c] = 1;
        pl_mode[c] = M_BIRTH; pl_twin[c] = p; pl_kA[c] = S.key[p]; pl_kB[c] = S.key[c];
        pl_tx[c] = S.x[p] + d * cos(a); pl_ty[c] = S.y[p] + d * sin(a);
        pl_tx[p] = S.x[p] - d * cos(a); pl_ty[p] = S.y[p] - d * sin(a);
        pl_dl[p] = pl_dl[c] = 0.45 * hypot(S.x[p] - ax, S.y[p] - ay) / R;
    }
    /* fusions: nearest pairs around the other point */
    n = near_list(bxp, byp, lst, 4 * M, used);
    int merged = 0;
    double Rb = n ? hypot(S.x[lst[n - 1]] - bxp, S.y[lst[n - 1]] - byp) + 1e-9 : 1;
    for (int i = 0; i < n && merged < M; i++) {
        int a = lst[i];
        if (used[a]) continue;
        int bj = -1;
        double bd = 1.8 * sp;
        for (int j = i + 1; j < n; j++) {
            int b = lst[j];
            if (used[b]) continue;
            double d = hypot(S.x[a] - S.x[b], S.y[a] - S.y[b]);
            if (d < bd) { bd = d; bj = b; }
        }
        if (bj < 0) continue;
        used[a] = used[bj] = 1;
        pl_mode[bj] = M_DEATH; pl_twin[bj] = a; pl_kA[bj] = S.key[bj]; pl_kB[bj] = S.key[a];
        pl_tx[bj] = S.x[a]; pl_ty[bj] = S.y[a];
        pl_dl[bj] = 0.45 * hypot(S.x[a] - bxp, S.y[a] - byp) / Rb;
        merged++;
    }
    free(used); free(lst);
}

/* write the text: cells under it bud children, sampled from the density
 * and placed left to right, until the letters hold ~5x the density */
static void plan_text(Rng *r) {
    if (!mk_n) return;
    double area = mk_sum * gs * gs, st = sp * 0.32;
    int nb = (int)(area / (st * st) - area / (sp * sp));
    int room = S.cap - S.n - 8;
    if (nb > room) nb = room;
    if (nb > (int)(1.6 * N0)) nb = (int)(1.6 * N0);
    if (nb < 0) nb = 0;
    double *cum = malloc(sizeof(double) * mk_n), acc = 0, xmin = 1e300, xmax = -1e300;
    for (int q = 0; q < mk_n; q++) {
        cum[q] = acc += mk_w[q];
        xmin = fmin(xmin, mk_x[q]); xmax = fmax(xmax, mk_x[q]);
    }
    int n0 = S.n;
    unsigned char *old = malloc(n0);
    memcpy(old, S.alive, n0);
    for (int b = 0; b < nb; b++) {
        double u = rnd(r) * acc;
        int lo = 0, hi = mk_n - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (cum[mid] < u) lo = mid + 1; else hi = mid; }
        double x = mk_x[lo] + (rnd(r) - 0.5) * gs, y = mk_y[lo] + (rnd(r) - 0.5) * gs;
        int par = -1;
        double bd = 1e300;
        for (int s = 0; s < n0; s++)
            if (old[s]) {
                double d = (S.x[s] - x) * (S.x[s] - x) + (S.y[s] - y) * (S.y[s] - y);
                if (d < bd) { bd = d; par = s; }
            }
        int c = alloc_slot();
        if (c < 0 || par < 0) break;
        S.alive[c] = 1; S.extra[c] = 1;
        S.x[c] = S.x[par]; S.y[c] = S.y[par];
        S.key[c] = (uint32_t)(rnd(r) * 4294967296.0);
        pl_mode[c] = M_BIRTH; pl_twin[c] = par; pl_kA[c] = S.key[par]; pl_kB[c] = S.key[c];
        pl_tx[c] = x; pl_ty[c] = y;
        pl_dl[c] = 0.45 * clamp01((x - xmin) / (xmax - xmin + 1e-9));
    }
    free(cum); free(old);
    /* hold the letters at the density just made: a centroidal tessellation
     * puts sites at the square root of the density, so the weight goes as
     * the fourth power of the spacing ratio */
    cy.W = 1.15 * pow(sp / st, 4);
}

/* and dissolve it: each budded cell fuses into its nearest original */
static void plan_untext(void) {
    for (int s = 0; s < S.n; s++) {
        if (!S.alive[s] || !S.extra[s]) continue;
        int k = -1;
        double bd = 1e300;
        for (int o = 0; o < S.n; o++)
            if (S.alive[o] && !S.extra[o]) {
                double d = (S.x[o] - S.x[s]) * (S.x[o] - S.x[s]) + (S.y[o] - S.y[s]) * (S.y[o] - S.y[s]);
                if (d < bd) { bd = d; k = o; }
            }
        if (k < 0) continue;
        pl_mode[s] = M_DEATH; pl_twin[s] = k; pl_kA[s] = S.key[s]; pl_kB[s] = S.key[k];
        pl_tx[s] = S.x[k]; pl_ty[s] = S.y[k];
        pl_dl[s] = 0.45 * clamp01(vnoise(S.x[s] / (sp * 3), S.y[s] / (sp * 3), 55) * 1.6 - 0.3);
    }
}

static void begin_cycle(Where w, int j) {
    cy.w = w;
    cy.planned = 0; cy.settled = 0; cy.W = 0;
    mk_n = 0;
    if (w.kind == K_TEXT_TIME || w.kind == K_TEXT_HOST) build_mask(w.kind, w.start);
    if (j != 0 || w.dist == 0) return;
    Rng r = rng_for((uint64_t)w.era, (uint64_t)w.ci, 1);
    for (int s = 0; s < S.cap; s++) {
        pl_tx[s] = S.x[s]; pl_ty[s] = S.y[s]; pl_dl[s] = 0;
        pl_twin[s] = -1; pl_mode[s] = M_NONE; pl_kA[s] = pl_kB[s] = S.key[s];
    }
    double diag = hypot(bw, bh);
    switch (w.kind) {
    case K_DISSOLVE:
        plan_dissolve();
        break;
    case K_RIPPLE: {
        visible_point(&r, &cy.c[0][0], &cy.c[0][1], 0.25);
        double cx = cy.c[0][0], cyy = cy.c[0][1];
        cy.rmax = 0;
        double corners[4][2] = {{bx0, by0}, {bx1, by0}, {bx0, by1}, {bx1, by1}};
        for (int k = 0; k < 4; k++) cy.rmax = fmax(cy.rmax, hypot(corners[k][0] - cx, corners[k][1] - cyy));
        cy.A = 0.32 * sp; cy.B = 0.9 * sp; cy.kw = 2 * M_PI / (3.2 * sp);
        cy.del[0] = rnd(&r) * 2 * M_PI;
        break;
    }
    case K_SWIRL:
        cy.nc = 1 + (rnd(&r) < 0.55);
        for (int k = 0; k < cy.nc; k++) {
            visible_point(&r, &cy.c[k][0], &cy.c[k][1], 0.25);
            cy.amp[k] = (1.4 + 1.3 * rnd(&r)) * (rnd(&r) < 0.5 ? -1 : 1);
            if (k == 1) cy.amp[1] = -copysign(fabs(cy.amp[1]), cy.amp[0]);
            cy.sig[k] = fr[3] * (0.16 + 0.12 * rnd(&r));
            cy.del[k] = k * 0.25;
        }
        break;
    case K_WIND: {
        double a = rnd(&r) * 2 * M_PI;
        cy.dir[0] = cos(a); cy.dir[1] = sin(a);
        double us[4] = {bx0 * cy.dir[0] + by0 * cy.dir[1], bx1 * cy.dir[0] + by0 * cy.dir[1],
                        bx0 * cy.dir[0] + by1 * cy.dir[1], bx1 * cy.dir[0] + by1 * cy.dir[1]};
        cy.u0 = 1e300; cy.u1 = -1e300;
        for (int k = 0; k < 4; k++) { cy.u0 = fmin(cy.u0, us[k]); cy.u1 = fmax(cy.u1, us[k]); }
        cy.width = 0.25 * (cy.u1 - cy.u0) + 2 * sp;
        cy.A = 1.3 * sp; cy.B = 0.8 * sp;
        cy.del[0] = rnd(&r) * 1000;
        break;
    }
    case K_CELLS:
        plan_cells(&r);
        break;
    case K_TEXT_TIME:
    case K_TEXT_HOST:
        plan_text(&r);
        break;
    case K_UNTEXT:
        plan_untext();
        break;
    }
    (void)diag;
    cy.planned = 1;
}

/* where slot s is at fraction tau through the disturbance */
static void dist_pos(int s, double tau, double *ox, double *oy) {
    double x = S.x[s], y = S.y[s];
    switch (cy.w.kind) {
    case K_RIPPLE: {
        double dx = x - cy.c[0][0], dy = y - cy.c[0][1], r = hypot(dx, dy);
        if (r > 1e-9) {
            double a = r / cy.rmax, f = clamp01((tau * 1.3 - a) / 0.3);
            double perm = cy.A * sin(cy.kw * r + cy.del[0]) * ease(f);
            double pulse = cy.B * sin(M_PI * f) * (1 - 0.5 * a);
            x += dx / r * (perm + pulse); y += dy / r * (perm + pulse);
        }
        break;
    }
    case K_SWIRL:
        for (int k = 0; k < cy.nc; k++) {
            double dx = x - cy.c[k][0], dy = y - cy.c[k][1];
            double th = cy.amp[k] * exp(-(dx * dx + dy * dy) / (cy.sig[k] * cy.sig[k])) * ease((tau - cy.del[k]) / 0.75);
            double c = cos(th), sn = sin(th);
            x = cy.c[k][0] + c * dx - sn * dy;
            y = cy.c[k][1] + sn * dx + c * dy;
        }
        break;
    case K_WIND: {
        double u = x * cy.dir[0] + y * cy.dir[1];
        double front = cy.u0 - cy.width + (cy.u1 - cy.u0 + 2 * cy.width) * tau;
        double f = clamp01((front - u) / cy.width), e = ease(f);
        /* eddies: the curl of a noise field */
        double L = 5 * sp, h = 0.05;
        double nx = x / L + cy.del[0], ny = y / L;
        double dpx = (vnoise(nx + h, ny, 77) - vnoise(nx - h, ny, 77)) / (2 * h);
        double dpy = (vnoise(nx, ny + h, 77) - vnoise(nx, ny - h, 77)) / (2 * h);
        double g = cy.A * e + cy.B * sin(M_PI * f);
        x += g * (cy.dir[0] + 0.9 * dpy);
        y += g * (cy.dir[1] - 0.9 * dpx);
        break;
    }
    case K_DISSOLVE:
    case K_CELLS:
    case K_TEXT_TIME:
    case K_TEXT_HOST:
    case K_UNTEXT:
        if (pl_mode[s] != M_NONE) {
            double e = ease((tau - pl_dl[s]) / (1 - 0.45));
            double tx = pl_tx[s], ty = pl_ty[s];
            x += (tx - x) * e; y += (ty - y) * e;
            if (cy.w.kind == K_DISSOLVE) {   /* a sideways bow, so the melt swirls */
                double bow = 0.22 * sin(M_PI * e) * ((pl_kB[s] & 1) ? 1 : -1);
                x += -(ty - S.y[s]) * bow; y += (tx - S.x[s]) * bow;
            }
        }
        break;
    }
    double m = 0.25;
    *ox = x < bx0 + m ? bx0 + m : x > bx1 - m ? bx1 - m : x;
    *oy = y < by0 + m ? by0 + m : y > by1 - m ? by1 - m : y;
}

static void finalize(void) {
    if (cy.w.kind == K_DISSOLVE) { sim_reset(cy.w.era); return; }
    if (!cy.planned) return;
    double *nx = g_ax, *ny = g_ay;    /* scratch */
    for (int s = 0; s < S.n; s++)
        if (S.alive[s]) dist_pos(s, 1.0, &nx[s], &ny[s]);
    for (int s = 0; s < S.n; s++) {
        if (!S.alive[s]) continue;
        S.x[s] = nx[s]; S.y[s] = ny[s];
        S.key[s] = pl_kB[s];
        if (pl_mode[s] == M_DEATH) S.alive[s] = S.extra[s] = 0;
    }
}

static void snapshot(long k) {
    memcpy(ring_x[k % RING], S.x, sizeof(double) * S.n);
    memcpy(ring_y[k % RING], S.y, sizeof(double) * S.n);
}

static void sim_step(long k) {
    Where w = locate(k);
    int j = (int)(k - w.start);
    if (cy.w.start != w.start) begin_cycle(w, j);
    if (j < w.dist) {
        snapshot(k);
    } else if (j == w.dist) {
        if (w.dist > 0) finalize();
        snapshot(k);
    } else {
        int r = j - w.dist;
        double wt = cy.W * text_w(r), wp = cy.W * text_w(r - 1);
        if (!cy.settled || wt != wp) {
            /* over-relaxed while text forms and fades: plain Lloyd is
             * slow to move sites any distance */
            int text = w.kind == K_TEXT_TIME || w.kind == K_TEXT_HOST;
            double mv = lloyd(wt, text ? 1.3 : w.kind == K_UNTEXT && r < 30 ? 1.5 : 1.0);
            cy.settled = mv < 0.002 * sp && wt == wp;
        }
        snapshot(k);
    }
    sim_k = k;
}

/* start (or restart) at tick k: from the canonical scatter of k's era, or
 * of the era before when k is inside a dissolve (which needs its end) */
static void sim_init(long k) {
    Where w = locate(k);
    long e = w.era;
    if (e > 0 && k - e * ERA_TICKS < kind_dist(K_DISSOLVE)) e--;
    sim_reset(e);
    cy.w.start = -1;
    long s = e * ERA_TICKS + (e ? kind_dist(K_DISSOLVE) : 0);
    sim_k = s - 1;
}

static void sim_advance(long target) {
    if (sim_k < 0 && target < 0) return;
    if (target - sim_k > 2 * ERA_TICKS) sim_init(target);
    while (sim_k < target) sim_step(sim_k + 1);
}

/* ==== the frame: where every site is at time t ==== */
static double *fx_, *fy_, *fcx, *fcy;
static float *fkA, *fkB, *fmix, *fea;
static int *ftw;
static double f_area_on, f_dual_on, f_cen_on, f_text_on;   /* overlay strengths */

static void catmull(int s, double rr, int rl, long base, double *ox, double *oy) {
    int i = (int)floor(rr);
    double f = rr - i;
    int ii[4];
    for (int k = 0; k < 4; k++) ii[k] = (int)((base + clampi(i - 1 + k, 0, rl)) % RING);
    double f2 = f * f, f3 = f2 * f;
    double w0 = -0.5 * f3 + f2 - 0.5 * f, w1 = 1.5 * f3 - 2.5 * f2 + 1, w2 = -1.5 * f3 + 2 * f2 + 0.5 * f, w3 = 0.5 * f3 - 0.5 * f2;
    *ox = w0 * ring_x[ii[0]][s] + w1 * ring_x[ii[1]][s] + w2 * ring_x[ii[2]][s] + w3 * ring_x[ii[3]][s];
    *oy = w0 * ring_y[ii[0]][s] + w1 * ring_y[ii[1]][s] + w2 * ring_y[ii[2]][s] + w3 * ring_y[ii[3]][s];
}

static double ramp(double x, double a, double b) { return ease((x - a) / (b - a)); }

static void frame_positions(double t) {
    if (t < 0) t = 0;
    double T = t / TICK;
    long k = (long)floor(T);
    double f = T - k;
    Where w = locate(k);
    int j = (int)(k - w.start);
    long target = k + 3;
    if (target > w.start + w.len - 1) target = w.start + w.len - 1;
    if (j < w.dist && target > w.start + w.dist - 1) target = w.start + w.dist - 1;
    if (sim_k < 0 || target < sim_k - RING + 4) sim_init(k);
    sim_advance(target);

    double ct = (j + f) * TICK, cl = w.len * TICK;   /* seconds into / length of the cycle */
    Rng r = rng_for((uint64_t)w.era, (uint64_t)w.ci, 99);
    int want_dual = rnd(&r) < 0.3, want_cen = rnd(&r) < 0.45, want_area = rnd(&r) < 0.3;
    double fade = ramp(ct, 0, 4) * (1 - ramp(ct, cl - 3, cl));
    double relax0 = w.dist * TICK;
    f_dual_on = want_dual && w.kind < K_TEXT_TIME ? fade : 0;
    f_cen_on = want_cen ? ramp(ct, relax0 + 0.5, relax0 + 3) * (1 - ramp(ct, cl - 3, cl)) : 0;
    f_text_on = 0;
    if (w.kind == K_TEXT_TIME || w.kind == K_TEXT_HOST) {
        f_text_on = ramp(ct, 0.5, 4);
        f_cen_on = 0;
    } else if (w.kind == K_UNTEXT) {
        f_text_on = 1 - ramp(ct, 1, w.dist * TICK + 2);
        f_cen_on = 0;
    }
    f_area_on = fmax(want_area ? fade : 0, f_text_on);

    if (j < w.dist && cy.planned) {
        double tau = (j + f) / w.dist;
        for (int s = 0; s < S.n; s++) {
            if (!S.alive[s]) continue;
            dist_pos(s, tau, &fx_[s], &fy_[s]);
            fcx[s] = fx_[s]; fcy[s] = fy_[s];
            double e = pl_mode[s] != M_NONE ? ease((tau - pl_dl[s]) / (1 - 0.45)) : 0;
            fkA[s] = (float)(pl_kA[s] * (1.0 / 4294967296.0));
            fkB[s] = (float)(pl_kB[s] * (1.0 / 4294967296.0));
            fmix[s] = (float)e;
            ftw[s] = pl_mode[s] == M_BIRTH || pl_mode[s] == M_DEATH ? pl_twin[s] : -1;
            fea[s] = (float)(pl_mode[s] == M_BIRTH ? e : pl_mode[s] == M_DEATH ? 1 - e : 1);
        }
    } else {
        int rl = w.len - 1 - w.dist;
        double rr = (j - w.dist) + f;
        if (j < w.dist) rr = 0;                         /* a dist cycle we joined late */
        long base = w.start + w.dist;
        for (int s = 0; s < S.n; s++) {
            if (!S.alive[s]) continue;
            catmull(s, rr, rl, base, &fx_[s], &fy_[s]);
            catmull(s, rr + 1, rl, base, &fcx[s], &fcy[s]);
            fkA[s] = fkB[s] = (float)(S.key[s] * (1.0 / 4294967296.0));
            fmix[s] = 0; ftw[s] = -1; fea[s] = 1;
        }
    }
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

/* shared by the cell shaders: uniforms, noise, colours */
static const char *CELL_COMMON = GLSL(
    uniform int uStyle;
    uniform float uSp, uPx, uArea, uCen, uTime, uTextOn;
    uniform vec3 uLight;

    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
    /* noise from the baked tile (see bake_noise): value noise with one
     * lattice cell per unit, fbm, and a second value noise for warps */
    uniform sampler2D uNoise;
    float vn(vec2 p) { return texture(uNoise, p * (1.0 / 64.0)).g; }
    float vn2(vec2 p) { return texture(uNoise, p * (1.0 / 64.0)).b; }
    float fbm(vec2 p) { return texture(uNoise, p * (1.0 / 16.0)).r; }
    float kh(float k, float salt) { return fract(sin(k * 1271.13 + salt * 311.7) * 43758.5453); }
    vec2 warp(vec2 p, float s, float a) { vec4 t = texture(uNoise, p / (s * 64.0)); return p + a * (t.gb - 0.5) * 2.0; }

    /* stained glass: a background of blues and teals that drifts across the
     * window, and jewel accents by key */
    vec3 glassCol(float k, vec2 s) {
        float f = vn(s / (uSp * 7.0) + 3.7), g = vn(s / (uSp * 11.0) + 9.1);
        vec3 bg = mix(vec3(0.05, 0.14, 0.55), vec3(0.03, 0.38, 0.47), smoothstep(0.3, 0.7, f));
        bg = mix(bg, vec3(0.25, 0.08, 0.50), smoothstep(0.68, 0.9, g));
        vec3 c = bg * (0.6 + 0.6 * kh(k, 2.0));
        if (k < 0.06) c = vec3(0.62, 0.04, 0.07);
        else if (k < 0.10) c = vec3(0.88, 0.52, 0.06);
        else if (k < 0.13) c = vec3(0.05, 0.40, 0.15);
        else if (k < 0.145) c = vec3(0.62, 0.62, 0.50);
        else if (k < 0.16) c = vec3(0.45, 0.10, 0.36);
        return c;
    }
    vec3 giraffeCol(float k) { return mix(vec3(0.45, 0.22, 0.08), vec3(0.62, 0.34, 0.12), kh(k, 3.0)); }
    vec3 crackleCol(float k) { return vec3(0.63, 0.74, 0.66) + (kh(k, 4.0) - 0.5) * vec3(0.05, 0.05, 0.045); }
    vec3 washCol(float k) {
        float h = kh(k, 6.0);
        return h < 0.25 ? vec3(0.86, 0.66, 0.36) : h < 0.5 ? vec3(0.56, 0.66, 0.50) : h < 0.75 ? vec3(0.48, 0.58, 0.72) : vec3(0.84, 0.56, 0.50);
    }
    vec3 cellCol(float k, vec2 s) {
        if (uStyle == 0) return glassCol(k, s);
        if (uStyle == 1) return giraffeCol(k);
        if (uStyle == 2) return crackleCol(k);
        if (uStyle == 4) return washCol(k);
        if (uStyle == 5) return vec3(kh(k, 5.0));
        return vec3(0.0);
    }
    /* small cells hot, large ones cool */
    vec3 areaRamp(float t) {
        if (uStyle == 0) return t < 0.45 ? mix(vec3(0.03, 0.08, 0.38), vec3(0.06, 0.22, 0.55), t / 0.45) : t < 0.75 ? mix(vec3(0.06, 0.22, 0.55), vec3(0.75, 0.08, 0.10), (t - 0.45) / 0.3) : mix(vec3(0.75, 0.08, 0.10), vec3(1.0, 0.74, 0.18), (t - 0.75) / 0.25);
        if (uStyle == 1) return mix(vec3(0.36, 0.16, 0.05), vec3(0.85, 0.52, 0.20), t);
        if (uStyle == 2) return mix(vec3(0.42, 0.56, 0.50), vec3(0.80, 0.86, 0.76), t);
        if (uStyle == 3) return mix(vec3(0.06, 0.20, 0.44), vec3(0.40, 0.66, 0.92), t);
        if (uStyle == 4) return mix(vec3(0.95, 0.93, 0.86), vec3(0.72, 0.30, 0.22), t);
        return vec3(t);
    }

    );

/* one instance per fan triangle of a cell (site, vertex j, vertex j+1),
 * carrying the bisector lines of its edge and the two either side, so the
 * fragment shader needs no lookups */
static const char *VS_CELL = GLSL(
    layout(location = 0) in vec4 aP01;
    layout(location = 1) in vec2 aP2;
    layout(location = 2) in vec4 aSite;
    layout(location = 3) in vec4 aKey;
    layout(location = 4) in vec4 aE0;
    layout(location = 5) in vec4 aE1;
    layout(location = 6) in vec4 aE2;
    layout(location = 7) in vec4 aE3;
    layout(location = 8) in vec4 aE4;
    layout(location = 9) in vec4 aE5;
    layout(location = 10) in vec4 aE6;
    layout(location = 11) in vec4 aE7;
    uniform vec3 uView;
    uniform vec2 uRes;
    out vec2 vW;
    flat out vec4 vSite, vKey, vE0, vE1, vE2, vE3, vE4, vE5, vE6, vE7;
    flat out vec3 vFill;
    void main() {
        vFill = mix(cellCol(aKey.x, aSite.xy), cellCol(aKey.y, aSite.xy), aSite.w);
        vec2 w = gl_VertexID == 0 ? aP01.xy : gl_VertexID == 1 ? aP01.zw : aP2;
        vW = w; vSite = aSite; vKey = aKey;
        vE0 = aE0; vE1 = aE1; vE2 = aE2; vE3 = aE3; vE4 = aE4; vE5 = aE5; vE6 = aE6; vE7 = aE7;
        vec2 px = (w - uView.xy) * uView.z;
        gl_Position = vec4(px.x / uRes.x * 2.0 - 1.0, 1.0 - px.y / uRes.y * 2.0, 0.0, 1.0);
    });

static const char *FS_CELL = GLSL(
    in vec2 vW;
    flat in vec4 vSite, vKey, vE0, vE1, vE2, vE3, vE4, vE5, vE6, vE7;
    flat in vec3 vFill;
    out vec4 o;

    void main() {
        vec4 t1 = vec4(vKey.xy, vSite.w, vSite.z);
        vec2 s = vSite.xy, p = vW;
        float aa = 0.5 * uPx;

        /* each style's wall: half width, corner rounding, a warp for organic edges */
        float hw, k = 0.0;
        vec2 q = p;
        if (uStyle == 0) hw = max(0.08 * uSp, 1.6 * uPx);
        else if (uStyle == 1) { hw = 0.05 * uSp; k = 0.09 * uSp; q = warp(p, 0.45 * uSp, 0.045 * uSp); aa *= 2.5; }
        else if (uStyle == 2) { hw = max(0.009 * uSp, 0.55 * uPx); q = warp(warp(p, 0.3 * uSp, 0.035 * uSp), 0.07 * uSp, 0.012 * uSp); }
        else if (uStyle == 3) hw = max(0.009 * uSp, 0.6 * uPx);
        else if (uStyle == 4) { hw = (max(0.011 * uSp, 0.7 * uPx)) * (0.75 + 0.5 * vn(p / (0.5 * uSp))); q = warp(p, 0.4 * uSp, 0.012 * uSp); }
        else { hw = 0.06 * uSp; k = 0.075 * uSp; }
        /* the thick walls scale with the cell, so a tiny cell in a letter
         * is still mostly glass, honey or hide */
        float cs = clamp(sqrt(max(vSite.z, 0.0)), 0.35, 1.3);
        if (uStyle == 0 || uStyle == 1 || uStyle == 5) { hw = max(hw * cs, 1.2 * uPx); k *= cs; }

        vec4 E[8] = vec4[8](vE0, vE1, vE2, vE3, vE4, vE5, vE6, vE7);
        float dmin = 1e9, cov = 0.0, dcov = 1e9, sm = 1e9, ss = 0.0;
        vec2 gmin = vec2(0.0), gcov = vec2(0.0, -1.0), sg = vec2(0.0);
        for (int i = 0; i < 8; i++) {
            vec2 nr = E[i].xy;
            float d = dot(q, nr) - E[i].z, a = E[i].w;
            /* interior shading ignores a line still fading in */
            float da = d + (1.0 - a) * 0.6 * uSp;
            if (da < dmin) { dmin = da; gmin = nr; }
            float c = a * (1.0 - smoothstep(hw - aa, hw + aa, abs(d)));
            if (c > cov) { cov = c; dcov = d; gcov = nr; }
            if (k > 0.0) {
                if (d < sm) { float f = exp((d - sm) / k); ss *= f; sg *= f; sm = d; }
                float wgt = a * exp(-(d - sm) / k);
                ss += wgt; sg += wgt * nr;
            }
        }
        float Ds = ss > 0.0 ? sm - k * log(ss) : 1e9;
        vec2 G = ss > 0.0 ? normalize(sg + 1e-6) : gmin;

        float area = t1.w, tA = clamp(0.3 - 0.5 * log2(max(area, 1e-3)), 0.0, 1.0);
        /* area colouring: everything while text is up, otherwise only the
         * cells whose size stands out */
        float am = max(uTextOn, uArea * clamp(abs(log2(max(area, 1e-3))) * 2.5, 0.0, 1.0));
        vec3 fill = vFill;
        vec3 col, ink = vec3(0.0);
        vec2 cen = vKey.zw;
        if (uTime < 0.0) { o = vec4(fill * (1.0 - cov) + 0.001 * Ds, 1.0); return; }

        if (uStyle == 0) {                       /* stained glass */
            vec3 g = mix(fill, areaRamp(tA), am * 0.9);
            vec2 dl = p - uLight.xy;
            float bl = 0.30 + 0.95 * exp(-dot(dl, dl) / (uLight.z * uLight.z)) + 0.35 * (fbm(p / (uSp * 6.0)) - 0.5);
            float kk = t1.z < 0.5 ? t1.x : t1.y;
            float ham = 0.80 + 0.40 * fbm((p + kk * 913.0) / (uSp * 0.16));
            float ang = kh(kk, 9.0) * 6.283;
            vec2 sd = vec2(cos(ang), sin(ang));
            float streak = vn(vec2(dot(p, sd) / (uSp * 0.9), dot(p, vec2(-sd.y, sd.x)) / (uSp * 0.03)) + kk * 77.0);
            float edge = smoothstep(hw, hw + 0.3 * uSp * cs, dmin);
            col = g * bl * ham * (0.85 + 0.25 * streak) * (0.35 + 0.65 * edge);
            col += g * g * 0.35 * bl * bl * edge * edge;
            float u = clamp(abs(dcov) / hw, 0.0, 1.0);
            vec3 N = normalize(vec3(gcov * u * 1.4, sqrt(max(1.0 - u * u, 0.0)) + 0.08));
            vec3 Ld = normalize(vec3(-0.45, -0.6, 0.66));
            float dif = max(dot(N, Ld), 0.0), spec = pow(max(reflect(-Ld, N).z, 0.0), 20.0);
            vec3 lead = vec3(0.075, 0.075, 0.085) * (0.25 + 1.3 * dif) + vec3(0.75, 0.74, 0.70) * spec * 0.55;
            lead *= 0.85 + 0.3 * vn(p / (uSp * 0.05));
            col = mix(col, lead, cov);
            ink = vec3(1.0, 0.86, 0.45);
        } else if (uStyle == 1) {                /* giraffe */
            vec3 pt = mix(fill, areaRamp(tA), am * 0.85);
            pt *= 0.82 + 0.22 * smoothstep(hw, hw + 0.35 * uSp * cs, Ds);
            pt *= 0.9 + 0.16 * vn(vec2(p.x / (uSp * 0.012), p.y / (uSp * 0.05)));
            vec3 net = vec3(0.93, 0.86, 0.70) * (0.93 + 0.08 * vn(p / (uSp * 0.08)));
            float m = smoothstep(hw - aa, hw + aa, Ds);
            col = mix(net, pt, m);
            ink = vec3(0.22, 0.10, 0.03);
        } else if (uStyle == 2) {                /* celadon crackle */
            vec3 gl = mix(fill, areaRamp(tA), am * 0.8);
            gl *= 0.92 + 0.12 * fbm(p / (uSp * 3.0));
            float halo = exp(-max(abs(dmin), 0.0) / (0.05 * uSp));
            gl = mix(gl, gl * vec3(0.78, 0.76, 0.62), 0.45 * halo);
            /* fine crazing: a smaller cellular net */
            float cz = texture(uNoise, p / (0.3 * uSp * 16.0)).a;
            gl *= 1.0 - 0.13 * (1.0 - smoothstep(0.0, 0.08, cz)) * smoothstep(0.3, 0.6, vn(p / (uSp * 1.3)));
            gl += vec3(0.07) * pow(fbm(p / (uSp * 8.0) + 3.0), 3.0);
            col = mix(gl, vec3(0.15, 0.12, 0.09), cov);
            ink = vec3(0.35, 0.18, 0.10);
        } else if (uStyle == 3) {                /* blueprint */
            vec3 bg = vec3(0.07, 0.22, 0.46) * (0.93 + 0.1 * fbm(p / (uSp * 1.5)));
            bg = mix(bg, areaRamp(tA), am * 0.55);
            float g = uSp * 0.5;
            vec2 gd = abs(fract(p / g + 0.5) - 0.5) * g / uPx;
            vec2 gD = abs(fract(p / (4.0 * g) + 0.5) - 0.5) * 4.0 * g / uPx;
            float gl = max(1.0 - smoothstep(0.3, 1.0, min(gd.x, gd.y)), 0.0) * 0.07 + max(1.0 - smoothstep(0.4, 1.2, min(gD.x, gD.y)), 0.0) * 0.12;
            col = mix(bg, vec3(0.75, 0.86, 1.0), gl);
            col = mix(col, vec3(0.90, 0.95, 1.0), cov);
            ink = vec3(1.0, 1.0, 1.0);
        } else if (uStyle == 4) {                /* ink on paper */
            vec3 paper = vec3(0.955, 0.93, 0.865) * (0.965 + 0.05 * fbm(p / (uSp * 0.07)));
            float kk = t1.z < 0.5 ? t1.x : t1.y;
            float wash = (kh(kk, 7.0) < 0.28 ? 1.0 : 0.0);
            float pool = exp(-max(dmin, 0.0) / (0.05 * uSp));
            float gran = 0.75 + 0.5 * fbm(p / (uSp * 0.05));
            vec3 wc = fill;
            col = mix(paper, paper * wc, wash * (0.32 + 0.35 * pool) * gran);
            col = mix(col, areaRamp(tA) * paper, am * 0.25);
            vec3 inkc = vec3(0.12, 0.10, 0.09);
            col = mix(col, inkc, cov * (0.85 + 0.15 * vn(p / (uSp * 0.03))));
            ink = vec3(0.70, 0.16, 0.10);
        } else {                                 /* honeycomb */
            float kk = t1.z < 0.5 ? t1.x : t1.y;
            float depth = smoothstep(hw, hw + 0.5 * uSp * cs, Ds);
            vec3 honey = mix(vec3(0.92, 0.58, 0.12), vec3(0.42, 0.16, 0.01), pow(depth, 0.6));
            honey += vec3(0.45, 0.22, 0.0) * exp(-dot(p - s, p - s) / (0.05 * uSp * uSp)) * 0.5;
            float rim = (1.0 - smoothstep(hw, hw + 0.16 * uSp * cs, Ds)) * max(dot(G, normalize(vec2(0.6, 0.8))), 0.0);
            honey += vec3(1.0, 0.90, 0.60) * rim * 0.4;
            /* capped cells: a few at random; while text is up, exactly the small ones */
            float capped = mix(step(kh(kk, 5.0), 0.16), smoothstep(0.55, 0.75, tA), am);
            vec3 cap = vec3(0.94, 0.80, 0.47) * (0.82 + 0.22 * depth) * (0.95 + 0.08 * vn(p / (uSp * 0.06)));
            vec3 cell = mix(honey, cap, capped);
            cell = mix(cell, cell * vec3(0.55, 0.40, 0.30), step(0.94, kh(kk, 8.0)) * (1.0 - capped));
            float u = clamp(Ds / hw, 0.0, 1.0);
            vec3 wax = vec3(0.90, 0.74, 0.40) * (0.70 + 0.32 * (1.0 - u * u)) * (0.95 + 0.08 * vn(p / (uSp * 0.04)));
            float m = smoothstep(hw - aa, hw + aa, Ds);
            col = mix(wax, cell, m);
            ink = vec3(0.30, 0.13, 0.02);
        }

        /* the relaxation's workings: the site, the centroid it is heading
         * for, and the step between */
        if (uCen > 0.0) {
            float rd = max(0.045 * uSp, 1.5 * uPx), rr = max(0.085 * uSp, 3.0 * uPx), lw = max(0.6 * uPx, 0.006 * uSp);
            float dot_ = 1.0 - smoothstep(rd - aa, rd + aa, length(p - s));
            float ring = 1.0 - smoothstep(lw - aa, lw + aa, abs(length(p - cen) - rr));
            vec2 sc = cen - s;
            float h = clamp(dot(p - s, sc) / max(dot(sc, sc), 1e-6), 0.0, 1.0);
            float seg = (1.0 - smoothstep(lw - aa, lw + aa, length(p - s - sc * h))) * smoothstep(rr * 0.9, rr * 1.4, length(sc));
            col = mix(col, ink, uCen * max(max(dot_, ring), seg * 0.8));
        }
        /* stippling: in the text, a dot per site */
        if ((uStyle == 3 || uStyle == 4) && am > 0.0) {
            float rd = 0.11 * uSp;
            float dt = 1.0 - smoothstep(rd - aa, rd + aa, length(p - s));
            col = mix(col, uStyle == 3 ? vec3(0.92, 0.96, 1.0) : vec3(0.12, 0.10, 0.09), am * dt);
        }
        o = vec4(col, 1.0);
    });

/* Delaunay dual: instanced, anti-aliased segments */
static const char *VS_LINE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aSeg;
    layout(location = 1) in float aA;
    uniform vec3 uView;
    uniform vec2 uRes;
    uniform float uW;
    out vec2 vL;
    flat out float vLen, vA;
    void main() {
        vec2 a = (aSeg.xy - uView.xy) * uView.z, b = (aSeg.zw - uView.xy) * uView.z;
        vec2 d = b - a;
        float L = max(length(d), 1e-4);
        vec2 t = d / L, n = vec2(-t.y, t.x);
        int id = gl_VertexID;
        vec2 c = vec2((id == 1 || id == 2 || id == 4) ? 1.0 : 0.0, (id == 2 || id == 4 || id == 5) ? 1.0 : 0.0);
        float e = uW + 1.5;
        vL = vec2(mix(-e, L + e, c.x), mix(-e, e, c.y));
        vec2 px = a + t * vL.x + n * vL.y;
        vLen = L; vA = aA;
        gl_Position = vec4(px.x / uRes.x * 2.0 - 1.0, 1.0 - px.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_LINE = "#version 330 core\n" GLSL(
    uniform vec4 uCol;
    uniform float uW, uDash;
    in vec2 vL;
    flat in float vLen, vA;
    out vec4 o;
    void main() {
        float dx = max(max(-vL.x, vL.x - vLen), 0.0);
        float c = 1.0 - smoothstep(uW - 0.5, uW + 0.5, length(vec2(dx, vL.y)));
        if (uDash > 0.0) {
            float ph = fract(vL.x / uDash);
            c *= smoothstep(0.0, 0.1, ph) * (1.0 - smoothstep(0.5, 0.6, ph));
        }
        o = vec4(uCol.rgb, uCol.a * c * vA);
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

static GLuint prog_cell, prog_line, prog_text;
static GLuint vao_cell, vbo_cell, vao_line, vbo_line, vao_text, vbo_text;
static GLuint tex_atlas, tex_noise;

/* a tileable 512x512 noise tile: R fbm (16 lattice cells across, 4
 * octaves), G and B value noise (64 cells), A the F2 - F1 distance of a
 * jittered cellular net (16 cells), for crazing */
#define NT 512
static double tile_vn(double x, double y, int L, uint32_t salt) {
    double fx = floor(x), fy = floor(y);
    int ix = (int)fx, iy = (int)fy;
    double u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    int x0 = ((ix % L) + L) % L, y0 = ((iy % L) + L) % L, x1 = (x0 + 1) % L, y1 = (y0 + 1) % L;
    double a = lat(x0, y0, salt), b = lat(x1, y0, salt), c = lat(x0, y1, salt), d = lat(x1, y1, salt);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}
static void bake_noise(void) {
    unsigned char *t = malloc(NT * NT * 4);
    for (int y = 0; y < NT; y++)
        for (int x = 0; x < NT; x++) {
            double u = (double)x / NT, v = (double)y / NT, f = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 4; o++) {
                int L = 16 << o;
                f += amp * tile_vn(u * L, v * L, L, 900 + o);
                norm += amp; amp *= 0.5;
            }
            f /= norm;
            f = 0.5 + (f - 0.5) * 1.6;
            double g = tile_vn(u * 64, v * 64, 64, 911), b = tile_vn(u * 64, v * 64, 64, 913);
            double cx = u * 16, cy2 = v * 16, f1 = 9, f2 = 9;
            int ix = (int)floor(cx), iy = (int)floor(cy2);
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int jx = ix + dx, jy = iy + dy, wx = ((jx % 16) + 16) % 16, wy = ((jy % 16) + 16) % 16;
                    double px = jx + 0.15 + 0.7 * lat(wx, wy, 920), py = jy + 0.15 + 0.7 * lat(wx, wy, 921);
                    double d = hypot(px - cx, py - cy2);
                    if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d;
                }
            unsigned char *o = t + 4 * (y * NT + x);
            o[0] = (unsigned char)(clamp01(f) * 255); o[1] = (unsigned char)(g * 255); o[2] = (unsigned char)(b * 255);
            o[3] = (unsigned char)(clamp01(f2 - f1) * 255);
        }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NT, NT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    free(t);
}
#define TRI_FLOATS 46                  /* per fan triangle: see VS_CELL */

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "voronoi: shader:\n%s\n", log);
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

/* ---- palettes ---- */
static const struct palette {
    const char *name;
    float cap[3], halo[3], dual[4];
} palettes[] = {
    {"glass",     {0.96f, 0.90f, 0.78f}, {0.02f, 0.02f, 0.03f}, {1.00f, 0.86f, 0.45f, 0.55f}},
    {"giraffe",   {0.20f, 0.10f, 0.04f}, {0.95f, 0.89f, 0.75f}, {0.25f, 0.12f, 0.04f, 0.55f}},
    {"crackle",   {0.16f, 0.13f, 0.10f}, {0.80f, 0.86f, 0.80f}, {0.40f, 0.22f, 0.12f, 0.50f}},
    {"blueprint", {0.92f, 0.96f, 1.00f}, {0.05f, 0.16f, 0.36f}, {0.80f, 0.90f, 1.00f, 0.45f}},
    {"paper",     {0.14f, 0.12f, 0.10f}, {0.96f, 0.94f, 0.88f}, {0.72f, 0.18f, 0.12f, 0.55f}},
    {"honey",     {1.00f, 0.93f, 0.75f}, {0.18f, 0.08f, 0.01f}, {0.35f, 0.15f, 0.02f, 0.55f}},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) return;
        fprintf(stderr, "voronoi: no palette \"%s\" (glass, giraffe, crackle, blueprint, paper, honey)\n", opt_palette);
        exit(1);
    }
    uint32_t h = opt_seed * 2654435761u;
    h ^= h >> 15;
    pal = (int)(h % NPAL);
}

/* ---- caption: the time, and how long the screen has been locked ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
static Glyph glyphs[2][96];
static int cap_px[2], atlas_w = 1024, atlas_h;
static float *tv;
static int tvn, tvcap;
static char cap_line[2][96];

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
    cap_line[0][0] = 0;                  /* rebuild the quads */
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
    if (!strcmp(l0, cap_line[0]) && !strcmp(l1, cap_line[1])) return tvn / 8;
    strcpy(cap_line[0], l0); strcpy(cap_line[1], l1);
    tvn = 0;
    const struct palette *P = &palettes[pal];
    float mx = (float)fmax(14, H * 0.035), by = (float)(H - fmax(14, H * 0.035));
    float y1 = by, y0 = by - cap_px[1] * 1.55f;
    float halo[4] = {P->halo[0], P->halo[1], P->halo[2], 0.16f}, col[4] = {P->cap[0], P->cap[1], P->cap[2], 0.92f};
    float r = fmaxf(1.0f, cap_px[0] / 12.0f);
    for (int k = 0; k < 12; k++) {
        float a = k * 0.5236f, dx = r * cosf(a), dy = r * sinf(a);
        for (int rr = 1; rr <= 2; rr++) {
            text_at(0, mx + dx * rr, y0 + dy * rr, l0, halo);
            text_at(1, mx + dx * rr, y1 + dy * rr, l1, halo);
        }
    }
    text_at(0, mx, y0, l0, col);
    col[3] = 0.8f;
    text_at(1, mx, y1, l1, col);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_DYNAMIC_DRAW);
    return tvn / 8;
}

static void gl_init(void) {
    {
        static char vs[32768], fs[65536];
        snprintf(vs, sizeof vs, "#version 330 core\n%s\n%s", CELL_COMMON, VS_CELL);
        snprintf(fs, sizeof fs, "#version 330 core\n%s\n%s", CELL_COMMON, FS_CELL);
        prog_cell = mk_prog(vs, fs);
    }
    prog_line = mk_prog(VS_LINE, FS_LINE);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_cell);
    glGenBuffers(1, &vbo_cell);
    glBindVertexArray(vao_cell);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cell);
    {
        static const int sz[12] = {4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        int off = 0;
        for (int i = 0; i < 12; i++) {
            glVertexAttribPointer(i, sz[i], GL_FLOAT, GL_FALSE, TRI_FLOATS * 4, (void *)(intptr_t)(off * 4));
            glVertexAttribDivisor(i, 1);
            glEnableVertexAttribArray(i);
            off += sz[i];
        }
    }
    glGenVertexArrays(1, &vao_line);
    glGenBuffers(1, &vbo_line);
    glBindVertexArray(vao_line);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_line);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 20, (void *)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 20, (void *)16);
    glVertexAttribDivisor(0, 1);
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
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
    bake_noise();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int c = S.cap + NFAR;
    fx_ = calloc(c, sizeof(double)); fy_ = calloc(c, sizeof(double));
    fcx = calloc(c, sizeof(double)); fcy = calloc(c, sizeof(double));
    fkA = calloc(c, sizeof(float)); fkB = calloc(c, sizeof(float));
    fmix = calloc(c, sizeof(float)); fea = calloc(c, sizeof(float));
    ftw = calloc(c, sizeof(int));
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

/* ---- per-frame geometry ---- */
static float *tri_buf, *line_buf;
static int *slot2i;
static int n_tri, n_lines;
static double t_tri;                   /* CPU ms spent triangulating this frame */

/* the edge alpha between compact sites i and n: a division or fusion fades
 * the line between the pair */
static float edge_alpha(int si, int n, int m) {
    if (n >= m) return 1;
    int sn = g_slot[n];
    if (ftw[si] == sn) return fea[si];
    if (ftw[sn] == si) return fea[sn];
    return 1;
}

/* everything the geometry depends on, to skip rebuilding a still frame */
static unsigned char *last_key;
static size_t last_len;
static int frame_unchanged(void) {
    size_t n = (size_t)S.n, len = n * (4 * sizeof(double) + 4 * sizeof(float) + sizeof(int) + 1) + 6 * sizeof(double);
    static unsigned char *k;
    static size_t cap;
    if (len > cap) { k = realloc(k, len); last_key = realloc(last_key, len); cap = len; }
    unsigned char *o = k;
#define PUT(src, sz) do { memcpy(o, (src), (sz)); o += (sz); } while (0)
    PUT(fx_, n * sizeof(double)); PUT(fy_, n * sizeof(double)); PUT(fcx, n * sizeof(double)); PUT(fcy, n * sizeof(double));
    PUT(fkA, n * sizeof(float)); PUT(fkB, n * sizeof(float)); PUT(fmix, n * sizeof(float)); PUT(fea, n * sizeof(float));
    PUT(ftw, n * sizeof(int)); PUT(S.alive, n);
    double v[6] = {v_ox, v_oy, v_sc, (double)W, (double)H, f_dual_on > 0};
    PUT(v, sizeof v);
#undef PUT
    int same = len == last_len && !memcmp(k, last_key, len);
    memcpy(last_key, k, len);
    last_len = len;
    return same;
}

static void frame_geometry(void) {
    double t0 = now();
    if (frame_unchanged()) { t_tri = (now() - t0) * 1000; return; }
    /* only the sites near this output: 5 spacings out, every cell that
     * reaches the screen still has all its true neighbours (cells are about
     * one spacing across, a little more in the sparse parts of a text
     * cycle), so the slices of neighbouring outputs agree exactly */
    double mg = 5 * sp, keep[4] = {v_ox - mg, v_oy - mg, v_ox + W / v_sc + mg, v_oy + H / v_sc + mg};
    int m = triangulate(fx_, fy_, getenv("VORONOI_FULL") ? NULL : keep);
    int c = S.cap + NFAR;
    if (!tri_buf) {
        slot2i = malloc(sizeof(int) * c);
        tri_buf = malloc(sizeof(float) * TRI_FLOATS * ((size_t)c * 8 + MAXV));
        line_buf = malloc(sizeof(float) * 5 * 3 * c);
    }
    double vx0 = v_ox - sp, vy0 = v_oy - sp, vx1 = v_ox + W / v_sc + sp, vy1 = v_oy + H / v_sc + sp;
    double mean = bw * bh / N0;
    double vx[MAXV + 8], vy[MAXV + 8], cx, cyy, ex[MAXV][4];
    int nb[MAXV];
    n_tri = 0;
    for (int i = 0; i < m; i++) {
        int s = g_slot[i], k = cell_of(i, vx, vy, nb);
        if (k < 3) continue;
        double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
        for (int j = 0; j < k; j++) {
            x0 = fmin(x0, vx[j]); x1 = fmax(x1, vx[j]); y0 = fmin(y0, vy[j]); y1 = fmax(y1, vy[j]);
        }
        if (x1 < vx0 || x0 > vx1 || y1 < vy0 || y0 > vy1) continue;
        double sx = fx_[s], sy = fy_[s];
        /* edge j runs from vertex j to j+1, shared with neighbour nb[j+1] */
        for (int j = 0; j < k; j++) {
            int n = nb[j + 1 == k ? 0 : j + 1];
            double nx = sx - g_pts[2 * n], ny = sy - g_pts[2 * n + 1], L = hypot(nx, ny);
            if (L < 1e-9) { ex[j][0] = 0; ex[j][1] = 1; ex[j][2] = -1e9; ex[j][3] = 0; continue; }
            nx /= L; ny /= L;
            ex[j][0] = nx; ex[j][1] = ny;
            ex[j][2] = nx * (sx + g_pts[2 * n]) * 0.5 + ny * (sy + g_pts[2 * n + 1]) * 0.5;
            ex[j][3] = edge_alpha(s, n, m);
        }
        double tx[MAXV + 8], ty[MAXV + 8];
        memcpy(tx, vx, sizeof(double) * k); memcpy(ty, vy, sizeof(double) * k);
        float area = (float)(poly_clip_ac(tx, ty, k, &cx, &cyy) / mean);
        for (int j = 0; j < k; j++) {
            int j2 = j + 1 == k ? 0 : j + 1;
            float *tb = tri_buf + TRI_FLOATS * n_tri++;
            tb[0] = (float)sx; tb[1] = (float)sy; tb[2] = (float)vx[j]; tb[3] = (float)vy[j];
            tb[4] = (float)vx[j2]; tb[5] = (float)vy[j2];
            tb[6] = (float)sx; tb[7] = (float)sy; tb[8] = area; tb[9] = fmix[s];
            tb[10] = fkA[s]; tb[11] = fkB[s]; tb[12] = (float)fcx[s]; tb[13] = (float)fcy[s];
            /* a cell of up to eight edges passes them all; a bigger one the
             * eight nearest this wedge (the edge itself first) */
            int pick[8];
            if (k > 8) {
                double wx = (sx + vx[j] + vx[j2]) / 3, wy = (sy + vy[j] + vy[j2]) / 3, dd[MAXV];
                for (int q = 0; q < k; q++) dd[q] = ex[q][0] * wx + ex[q][1] * wy - ex[q][2];
                dd[j] = -1e300;
                for (int e = 0; e < 8; e++) {
                    int bi = 0;
                    for (int q = 1; q < k; q++) if (dd[q] < dd[bi]) bi = q;
                    pick[e] = bi; dd[bi] = 1e300;
                }
            }
            for (int e = 0; e < 8; e++) {
                int off = (e + 1) / 2 * (e & 1 ? 1 : -1), jj = k > 8 ? pick[e] : ((j + off) % k + k) % k;
                int dup = e >= k;
                float *eb = tb + 14 + 4 * e;
                eb[0] = (float)ex[jj][0]; eb[1] = (float)ex[jj][1]; eb[2] = (float)ex[jj][2];
                eb[3] = dup ? 0.0f : (float)ex[jj][3];
            }
        }
    }
    n_lines = 0;
    if (f_dual_on > 0 || palettes[pal].name[0] == 'b') {
        for (int e = 0; e < D.nt; e++) {
            int h = D.half[e];
            if (h >= 0 && h < e) continue;
            int a = D.tri[e], b = D.tri[NXT(e)];
            if (a >= m || b >= m) continue;
            double ax = g_pts[2 * a], ay = g_pts[2 * a + 1], bx = g_pts[2 * b], by = g_pts[2 * b + 1];
            if (fmax(ax, bx) < vx0 || fmin(ax, bx) > vx1 || fmax(ay, by) < vy0 || fmin(ay, by) > vy1) continue;
            float *lb = line_buf + 5 * n_lines++;
            lb[0] = (float)ax; lb[1] = (float)ay; lb[2] = (float)bx; lb[3] = (float)by; lb[4] = 1;
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cell);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * TRI_FLOATS * n_tri, tri_buf, GL_STREAM_DRAW);
    if (n_lines) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_line);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 5 * n_lines, line_buf, GL_STREAM_DRAW);
    }
    t_tri = (now() - t0) * 1000;
}

static double t_sim;                   /* CPU ms in the simulation this frame */

static void render(double t, double wall, GLuint target) {
    double t0 = now();
    frame_positions(t);
    t_sim = (now() - t0) * 1000;
    view_update();
    frame_geometry();

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_cell);
    glUniform3f(U(prog_cell, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(prog_cell, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_cell, "uStyle"), pal);
    glUniform1f(U(prog_cell, "uSp"), (float)sp);
    glUniform1f(U(prog_cell, "uPx"), (float)(1.0 / v_sc));
    glUniform1f(U(prog_cell, "uArea"), (float)f_area_on);
    glUniform1f(U(prog_cell, "uTextOn"), (float)f_text_on);
    glUniform1f(U(prog_cell, "uCen"), (float)(palettes[pal].name[0] == 'b' ? fmax(f_cen_on, 0.6 * (1 - f_text_on)) : f_cen_on));
    glUniform1f(U(prog_cell, "uTime"), getenv("VPROBE") ? -1.0f : (float)fmod(t, 1000.0));
    /* the light behind the glass drifts slowly across the front screen */
    glUniform3f(U(prog_cell, "uLight"), (float)(fr[0] + fr[2] * (0.5 + 0.3 * cos(t * 0.011))),
                (float)(fr[1] + fr[3] * (0.4 + 0.25 * sin(t * 0.017))), (float)(span_me >= 0 ? 0.4 * bw : 0.75 * fmax(fr[2], fr[3])));
    glUniform1i(U(prog_cell, "uNoise"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_cell);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, n_tri);

    double dual = palettes[pal].name[0] == 'b' ? fmax(f_dual_on, 0.5) : f_dual_on;
    if (n_lines && dual > 0) {
        const struct palette *P = &palettes[pal];
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_line);
        glUniform3f(U(prog_line, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
        glUniform2f(U(prog_line, "uRes"), (float)W, (float)H);
        glUniform1f(U(prog_line, "uW"), (float)fmax(0.55, sp * v_sc * 0.007));
        glUniform1f(U(prog_line, "uDash"), P->name[0] == 'b' ? (float)fmax(6, sp * v_sc * 0.1) : 0.0f);
        glUniform4f(U(prog_line, "uCol"), P->dual[0], P->dual[1], P->dual[2], (float)(P->dual[3] * dual));
        glBindVertexArray(vao_line);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n_lines);
    }

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
    }
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "voronoi");
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
        "usage: voronoi [--window | --root | --layer] [--seed N] [--cells N] [--fps N]\n"
        "               [--palette glass|giraffe|crackle|blueprint|paper|honey] [--no-caption]\n"
        "               [--text WORD] [--font FAMILY] [--schedule]\n"
        "               [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "               [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --cells     sites on the front screen (default 650)\n"
        "  --palette   rendering; default: picked from the seed\n"
        "  --text      what the host-name cycles spell instead of the host name\n"
        "  --schedule  print what happens when (seconds after the epoch) and exit\n");
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
    for (long e = 0; e < 3; e++) {
        int kinds[NCYC];
        era_plan(e, kinds);
        long s = e * ERA_TICKS;
        for (int c = 0; c < NCYC; c++) {
            int d = kind_dist(kinds[c]), L = kind_len(kinds[c]);
            printf("%7.1f s  era %ld cycle %d: %-11s  (disturbance %4.1f s, relaxing from %6.1f s, %5.1f s long)\n",
                   s * TICK, e, c, kind_name[kinds[c]], d * TICK, (s + d) * TICK, L * TICK);
            s += L;
        }
    }
    printf("then every %.0f s the same pattern with other draws\n", ERA_TICKS * TICK);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-cells") && i + 1 < argc) opt_cells = atoi(argv[++i]);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-text") && i + 1 < argc) opt_text = argv[++i];
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
    if (opt_schedule) { print_schedule(); return 0; }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;
    pick_palette();
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    fonts_init();

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
        /* a window gets its real size from the WM: lay the diagram out for
         * the screen it will most likely fill */
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Voronoi");
        XClassHint chint = {"voronoi", "Voronoi"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    domain_init(dw, dh);
    gs = sp / 4;
    sim_alloc();
    gl_init();

    /* --epoch: when the relaxation began, shared by every instance; a --shot
     * or --bench then shows the moment --at seconds after it */
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
        render(opt_at, opt_epoch + opt_at, fbo);
        glFinish();
        double first = (now() - t0) * 1000;
        if (opt_shot && !opt_bench) {
            save_ppm(opt_shot);
            fprintf(stderr, "voronoi: %s, %dx%d, %d sites (%s), first frame %.0f ms\n", palettes[pal].name, W, H, N0,
                    kind_name[locate((long)(opt_at / TICK)).kind], first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, ssum = 0, tsum = 0, sworst = 0, gmin = 1e9, tmin = 1e9;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 60.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6); tmin = fmin(tmin, t_tri); ssum += t_sim; tsum += t_tri;
            if (ms > worst) worst = ms;
            if (t_sim > sworst) sworst = t_sim;
        }
        printf("voronoi: %s, %dx%d, %d sites, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, W, H, N0, opt_bench, opt_at, sum / opt_bench, worst);
        printf("voronoi:   GPU %.3f ms (best %.3f), CPU: simulation %.3f ms (worst %.2f), triangulation + upload %.3f ms (best %.3f)\n",
               gsum / opt_bench, gmin, ssum / opt_bench, sworst, tsum / opt_bench, tmin);
        printf("voronoi: GL renderer: %s (first frame, replaying to --at, %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
        if (opt_shot) save_ppm(opt_shot);   /* the last frame: a running instance, to compare with a fresh --shot */
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
