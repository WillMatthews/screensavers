/*
 * bridge — a structural analysis screen come alive: a finite-element solver
 * (written here, no libraries) running live behind a modern CAE display.
 *
 * Three acts, each opened by a title card and its governing equations:
 *
 *   I    BRIDGES UNDER MOVING LOADS  a Warren or Pratt truss, a tied arch, a
 *        cable-stayed or a suspension bridge (the cables tension-only),
 *        built of 2D frame and truss elements under their own dead load.
 *        Trains cross, their axle loads moving along the deck; the response
 *        is integrated in time (Newmark-beta, average acceleration, with
 *        Rayleigh damping) so the deck rings after a train has gone. Members
 *        are coloured by axial stress (tension blue, compression red), the
 *        live-load deflection is drawn exaggerated (labelled xN), and the
 *        midspan deflection is plotted against time.
 *   II   CONTINUUM  plane stress, bilinear quads with incompatible modes
 *        (QM6, condensed): a plate with a hole in tension (the Kirsch
 *        concentration, Kt -> 3, checked along the ligament), a cantilever
 *        in von Mises with its principal stress trajectories (the tip
 *        deflection checked against PL^3/3EI), and an L-bracket under a
 *        swinging load. Each solve is shown converging, iterate by iterate.
 *   III  TOPOLOGY OPTIMISATION  SIMP with a sensitivity filter and the
 *        optimality-criteria update (the classic 88-line code, ported): the
 *        MBB beam or a deck-bridge domain growing from grey mush into an
 *        organic truss.
 *
 * Every stiffness matrix is assembled sparse (CSR) and solved by Jacobi-
 * preconditioned conjugate gradients, warm-started from the last solution.
 * The solver runs at a lower rate than the display (the bridge every 10 ms
 * of model time, the optimiser a few iterations a second, time-sliced) and
 * frames interpolate between solutions.
 *
 * --check runs the verification cases and prints them: a cantilever's tip
 * deflection against PL^3/3EI, the Kirsch Kt, a pin-jointed truss against
 * the method of joints, and the 88-line MBB result.
 *
 * Palettes: cae (dark, turbo), blueprint, paper (a light drawing office),
 * mono. The lock info sits in the drawing's title block: the time, how long
 * the screen has been locked (swaylock's start time, from /proc), and the
 * hostname as the PROJECT.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The model is laid out once across the whole wall (a bridge from one end
 * of it to the other, the train crossing screen to screen); every instance
 * solves the same deterministic model and shows its own window onto it. A
 * portrait output, if any, carries the equations, plot and readouts above
 * and below the model; the front output (the one under the middle of the
 * layout, as in flightsim) carries the title block.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size.
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

#define PI 3.14159265358979
#define MAX_OUT 8

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot, *opt_palette, *opt_model;
static int opt_act;                    /* 1..3: loop one act */
static double opt_at = 30;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;
static float opt_budget = 3;           /* ms of solver work per frame, live */
static int opt_msaa = 4;

static float span[MAX_OUT][4];
static int span_n, span_me = -1, front = 1;

static void die(const char *what) { fprintf(stderr, "bridge: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double cpu_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
static double hashf(uint32_t a, uint32_t b, uint32_t c) { return (hash3(a, b, c) >> 8) * (1.0 / 16777216.0); }

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

/* ======================================================================
 * sparse matrices and the solver
 * ====================================================================== */

/* symmetric, both triangles stored, CSR */
typedef struct {
    int n, nnz;
    int *rp, *ci;
    double *v, *diag;
} Csr;

static int cmp_pair(const void *a, const void *b) {
    const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}

/* the pattern of ne elements of nd dofs each (dofs[e*nd + i], -1 = none),
 * and map[e*nd*nd + i*nd + j] = where element entry (i, j) lands in v */
static void csr_pattern(Csr *K, int n, int ne, int nd, const int *dofs, int **map) {
    int64_t *pr = malloc(sizeof *pr * (size_t)ne * nd * nd);
    size_t np = 0;
    for (int e = 0; e < ne; e++)
        for (int i = 0; i < nd; i++)
            for (int j = 0; j < nd; j++) {
                int a = dofs[e * nd + i], b = dofs[e * nd + j];
                if (a >= 0 && b >= 0) pr[np++] = (int64_t)a * n + b;
            }
    /* every diagonal, so a dof nothing touches still has a row */
    pr = realloc(pr, sizeof *pr * (np + n));
    for (int i = 0; i < n; i++) pr[np++] = (int64_t)i * n + i;
    qsort(pr, np, sizeof *pr, cmp_pair);
    size_t u = 0;
    for (size_t k = 0; k < np; k++)
        if (!u || pr[k] != pr[u - 1]) pr[u++] = pr[k];
    K->n = n;
    K->nnz = (int)u;
    K->rp = calloc((size_t)n + 1, sizeof(int));
    K->ci = malloc(sizeof(int) * u);
    K->v = calloc(u, sizeof(double));
    K->diag = calloc((size_t)n, sizeof(double));
    for (size_t k = 0; k < u; k++) {
        K->rp[pr[k] / n + 1]++;
        K->ci[k] = (int)(pr[k] % n);
    }
    for (int i = 0; i < n; i++) K->rp[i + 1] += K->rp[i];
    free(pr);
    *map = malloc(sizeof(int) * (size_t)ne * nd * nd);
    for (int e = 0; e < ne; e++)
        for (int i = 0; i < nd; i++)
            for (int j = 0; j < nd; j++) {
                int a = dofs[e * nd + i], b = dofs[e * nd + j], at = -1;
                if (a >= 0 && b >= 0) {
                    int lo = K->rp[a], hi = K->rp[a + 1] - 1;
                    while (lo <= hi) {
                        int m = (lo + hi) / 2;
                        if (K->ci[m] < b) lo = m + 1; else if (K->ci[m] > b) hi = m - 1; else { at = m; break; }
                    }
                }
                (*map)[(size_t)e * nd * nd + i * nd + j] = at;
            }
}

static void csr_free(Csr *K) { free(K->rp); free(K->ci); free(K->v); free(K->diag); memset(K, 0, sizeof *K); }

static void csr_diag(Csr *K) {
    for (int i = 0; i < K->n; i++)
        for (int k = K->rp[i]; k < K->rp[i + 1]; k++)
            if (K->ci[k] == i) K->diag[i] = K->v[k];
}

static void csr_mul(const Csr *K, const double *x, double *y) {
    for (int i = 0; i < K->n; i++) {
        double s = 0;
        for (int k = K->rp[i]; k < K->rp[i + 1]; k++) s += K->v[k] * x[K->ci[k]];
        y[i] = s;
    }
}

/* preconditioned CG on A = cK K + cM diag(M) over the free dofs (fixed
 * ones held at zero), resumable a few iterations at a time; Jacobi unless
 * a preconditioner is plugged in (the optimiser's multigrid) */
typedef struct {
    int n;
    void (*prec)(void *ctx, const double *r, double *z);
    void *pctx;
    const Csr *K;
    double cK, cM;
    const double *M;
    const unsigned char *fix;
    double *x, *r, *z, *p, *q, *dinv;
    double rz, bn, tol, res;
    int it, total, done;
} Pcg;

static void pcg_init(Pcg *s, const Csr *K, const unsigned char *fix) {
    memset(s, 0, sizeof *s);
    s->n = K->n; s->K = K; s->fix = fix; s->cK = 1; s->tol = 1e-8;
    s->x = calloc((size_t)s->n, sizeof(double));
    s->r = calloc((size_t)s->n, sizeof(double));
    s->z = calloc((size_t)s->n, sizeof(double));
    s->p = calloc((size_t)s->n, sizeof(double));
    s->q = calloc((size_t)s->n, sizeof(double));
    s->dinv = calloc((size_t)s->n, sizeof(double));
}

static void pcg_free(Pcg *s) { free(s->x); free(s->r); free(s->z); free(s->p); free(s->q); free(s->dinv); memset(s, 0, sizeof *s); }

static void pcg_apply(Pcg *s, const double *x, double *y) {
    csr_mul(s->K, x, y);
    for (int i = 0; i < s->n; i++) {
        y[i] = s->cK * y[i] + (s->M ? s->cM * s->M[i] * x[i] : 0);
        if (s->fix[i]) y[i] = 0;
    }
}

static void pcg_prec(Pcg *s) {
    if (s->prec) {
        s->prec(s->pctx, s->r, s->z);
        for (int i = 0; i < s->n; i++) if (s->fix[i]) s->z[i] = 0;
    } else
        for (int i = 0; i < s->n; i++) s->z[i] = s->dinv[i] * s->r[i];
}

/* start solving A x = b from the x already held (the warm start) */
static void pcg_begin(Pcg *s, const double *b) {
    int n = s->n;
    for (int i = 0; i < n; i++) {
        double d = s->cK * s->K->diag[i] + (s->M ? s->cM * s->M[i] : 0);
        s->dinv[i] = s->fix[i] || d <= 0 ? 0 : 1 / d;
        if (s->fix[i]) s->x[i] = 0;
    }
    pcg_apply(s, s->x, s->q);
    double bn = 0, rz = 0;
    for (int i = 0; i < n; i++) {
        double bi = s->fix[i] ? 0 : b[i];
        bn += bi * bi;
        s->r[i] = bi - s->q[i];
    }
    pcg_prec(s);
    for (int i = 0; i < n; i++) {
        s->p[i] = s->z[i];
        rz += s->r[i] * s->z[i];
    }
    s->bn = sqrt(bn);
    s->rz = rz;
    s->it = 0;
    double rr = 0;
    for (int i = 0; i < n; i++) rr += s->r[i] * s->r[i];
    s->res = s->bn > 0 ? sqrt(rr) / s->bn : 0;
    s->done = s->bn == 0 || s->res <= s->tol;
}

/* up to maxit more iterations; returns done */
static int pcg_run(Pcg *s, int maxit) {
    int n = s->n;
    for (int k = 0; k < maxit && !s->done; k++) {
        pcg_apply(s, s->p, s->q);
        double pq = 0;
        for (int i = 0; i < n; i++) pq += s->p[i] * s->q[i];
        if (pq <= 0) { s->done = 1; break; }
        double a = s->rz / pq, rr = 0, rz = 0;
        for (int i = 0; i < n; i++) {
            s->x[i] += a * s->p[i];
            s->r[i] -= a * s->q[i];
            rr += s->r[i] * s->r[i];
        }
        pcg_prec(s);
        for (int i = 0; i < n; i++) rz += s->r[i] * s->z[i];
        double bt = rz / s->rz;
        s->rz = rz;
        for (int i = 0; i < n; i++) s->p[i] = s->z[i] + bt * s->p[i];
        s->it++;
        s->total++;
        s->res = sqrt(rr) / s->bn;
        if (s->res <= s->tol || s->it > 20 * n) s->done = 1;
    }
    return s->done;
}

static void pcg_solve(Pcg *s, const double *b) {
    pcg_begin(s, b);
    while (!pcg_run(s, 1000)) {}
}

/* ======================================================================
 * elements
 * ====================================================================== */

/* 2D frame (Euler-Bernoulli) or truss bar in global axes; dofs u1 v1 r1 u2 v2 r2 */
static void bar_k(double x1, double y1, double x2, double y2, double E, double A, double I, int truss, double k[36]) {
    double dx = x2 - x1, dy = y2 - y1, L = sqrt(dx * dx + dy * dy), c = dx / L, s = dy / L;
    double kl[36] = {0};
    double ea = E * A / L;
    kl[0 * 6 + 0] = kl[3 * 6 + 3] = ea;
    kl[0 * 6 + 3] = kl[3 * 6 + 0] = -ea;
    if (!truss) {
        double a = 12 * E * I / (L * L * L), b = 6 * E * I / (L * L), d = 4 * E * I / L, e = 2 * E * I / L;
        kl[1 * 6 + 1] = kl[4 * 6 + 4] = a;
        kl[1 * 6 + 4] = kl[4 * 6 + 1] = -a;
        kl[1 * 6 + 2] = kl[2 * 6 + 1] = kl[1 * 6 + 5] = kl[5 * 6 + 1] = b;
        kl[4 * 6 + 2] = kl[2 * 6 + 4] = kl[4 * 6 + 5] = kl[5 * 6 + 4] = -b;
        kl[2 * 6 + 2] = kl[5 * 6 + 5] = d;
        kl[2 * 6 + 5] = kl[5 * 6 + 2] = e;
    }
    double T[36] = {0};
    T[0] = c; T[1] = s; T[6] = -s; T[7] = c; T[14] = 1;
    T[21] = c; T[22] = s; T[27] = -s; T[28] = c; T[35] = 1;
    /* k = T^T kl T */
    double t1[36];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double sum = 0;
            for (int m = 0; m < 6; m++) sum += kl[i * 6 + m] * T[m * 6 + j];
            t1[i * 6 + j] = sum;
        }
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double sum = 0;
            for (int m = 0; m < 6; m++) sum += T[m * 6 + i] * t1[m * 6 + j];
            k[i * 6 + j] = sum;
        }
}

/* plane stress D */
static void dmat(double E, double nu, double D[9]) {
    double c = E / (1 - nu * nu);
    D[0] = c; D[1] = c * nu; D[2] = 0;
    D[3] = c * nu; D[4] = c; D[5] = 0;
    D[6] = 0; D[7] = 0; D[8] = c * (1 - nu) / 2;
}

static void inv4(const double *a, double *o);

/* bilinear quad with Wilson-Taylor incompatible modes (QM6), the four
 * internal dofs condensed out: k (8x8), and kr (4x8) = Kii^-1 Kic so the
 * internal modes can be recovered for the stresses (alpha = -kr u) */
static const double QX[4] = {-1, 1, 1, -1}, QY[4] = {-1, -1, 1, 1};

static void q6_jac(const double *xy, double xi, double eta, double J[4], double dN[8]) {
    for (int a = 0; a < 4; a++) {
        dN[a * 2] = 0.25 * QX[a] * (1 + eta * QY[a]);
        dN[a * 2 + 1] = 0.25 * QY[a] * (1 + xi * QX[a]);
    }
    J[0] = J[1] = J[2] = J[3] = 0;
    for (int a = 0; a < 4; a++) {
        J[0] += dN[a * 2] * xy[a * 2];     J[1] += dN[a * 2] * xy[a * 2 + 1];
        J[2] += dN[a * 2 + 1] * xy[a * 2]; J[3] += dN[a * 2 + 1] * xy[a * 2 + 1];
    }
}

/* the 3x12 B at (xi, eta): 8 compatible columns then the 4 incompatible */
static double q6_b(const double *xy, double xi, double eta, double B[36]) {
    double J[4], dN[8], J0[4], dN0[8];
    q6_jac(xy, xi, eta, J, dN);
    q6_jac(xy, 0, 0, J0, dN0);
    double dj = J[0] * J[3] - J[1] * J[2], dj0 = J0[0] * J0[3] - J0[1] * J0[2];
    double Ji[4] = {J[3] / dj, -J[1] / dj, -J[2] / dj, J[0] / dj};
    double J0i[4] = {J0[3] / dj0, -J0[1] / dj0, -J0[2] / dj0, J0[0] / dj0};
    memset(B, 0, sizeof(double) * 36);
    for (int a = 0; a < 4; a++) {
        double nx = Ji[0] * dN[a * 2] + Ji[1] * dN[a * 2 + 1];
        double ny = Ji[2] * dN[a * 2] + Ji[3] * dN[a * 2 + 1];
        B[0 * 12 + a * 2] = nx;
        B[1 * 12 + a * 2 + 1] = ny;
        B[2 * 12 + a * 2] = ny;
        B[2 * 12 + a * 2 + 1] = nx;
    }
    /* P1 = 1 - xi^2, P2 = 1 - eta^2, derivatives through J0, scaled detJ0/detJ */
    double dP[2][2] = {{-2 * xi, 0}, {0, -2 * eta}}, f = dj0 / dj;
    for (int m = 0; m < 2; m++) {
        double px = f * (J0i[0] * dP[m][0] + J0i[1] * dP[m][1]);
        double py = f * (J0i[2] * dP[m][0] + J0i[3] * dP[m][1]);
        B[0 * 12 + 8 + m] = px;          /* u modes */
        B[2 * 12 + 8 + m] = py;
        B[1 * 12 + 10 + m] = py;         /* v modes */
        B[2 * 12 + 10 + m] = px;
    }
    return dj;
}

static void q6_k(const double *xy, const double D[9], double t, double k[64], double kr[32]) {
    double K[144] = {0}, B[36], DB[36];
    const double g = 1 / sqrt(3.0);
    for (int gp = 0; gp < 4; gp++) {
        double dj = q6_b(xy, QX[gp] * g, QY[gp] * g, B);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 12; j++)
                DB[i * 12 + j] = D[i * 3] * B[j] + D[i * 3 + 1] * B[12 + j] + D[i * 3 + 2] * B[24 + j];
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
                K[i * 12 + j] += (B[i] * DB[j] + B[12 + i] * DB[12 + j] + B[24 + i] * DB[24 + j]) * dj * t;
    }
    double Kii[16], Kiinv[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) Kii[i * 4 + j] = K[(8 + i) * 12 + 8 + j];
    inv4(Kii, Kiinv);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) {
            double s = 0;
            for (int m = 0; m < 4; m++) s += Kiinv[i * 4 + m] * K[(8 + m) * 12 + j];
            kr[i * 8 + j] = s;
        }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double s = K[i * 12 + j];
            for (int m = 0; m < 4; m++) s -= K[i * 12 + 8 + m] * kr[m * 8 + j];
            k[i * 8 + j] = s;
        }
}

/* stress (sx, sy, txy) at (xi, eta) in an element with nodal displacements u */
static void q6_stress(const double *xy, const double D[9], const double kr[32], const double *u, double xi, double eta, double s[3]) {
    double B[36], d[12], e[3];
    q6_b(xy, xi, eta, B);
    for (int i = 0; i < 8; i++) d[i] = u[i];
    for (int i = 0; i < 4; i++) {
        double a = 0;
        for (int j = 0; j < 8; j++) a -= kr[i * 8 + j] * u[j];
        d[8 + i] = a;
    }
    for (int i = 0; i < 3; i++) {
        e[i] = 0;
        for (int j = 0; j < 12; j++) e[i] += B[i * 12 + j] * d[j];
    }
    for (int i = 0; i < 3; i++) s[i] = D[i * 3] * e[0] + D[i * 3 + 1] * e[1] + D[i * 3 + 2] * e[2];
}

static void inv4(const double *a, double *o) {
    double m[4][8];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) m[i][j] = j < 4 ? a[i * 4 + j] : (j - 4 == i);
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++) if (fabs(m[r][c]) > fabs(m[p][c])) p = r;
        for (int j = 0; j < 8; j++) { double t = m[c][j]; m[c][j] = m[p][j]; m[p][j] = t; }
        double d = m[c][c];
        for (int j = 0; j < 8; j++) m[c][j] /= d;
        for (int r = 0; r < 4; r++)
            if (r != c) {
                double f = m[r][c];
                for (int j = 0; j < 8; j++) m[r][j] -= f * m[c][j];
            }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) o[i * 4 + j] = m[i][4 + j];
}

static void vm_principal(const double s[3], double *vm, double *s1, double *s2, double *ang) {
    double c = 0.5 * (s[0] + s[1]), r = sqrt(0.25 * (s[0] - s[1]) * (s[0] - s[1]) + s[2] * s[2]);
    *s1 = c + r; *s2 = c - r;
    *vm = sqrt(s[0] * s[0] - s[0] * s[1] + s[1] * s[1] + 3 * s[2] * s[2]);
    *ang = 0.5 * atan2(2 * s[2], s[0] - s[1]);
}


/* ======================================================================
 * II  continuum: plane stress QM6 meshes
 * ====================================================================== */

enum { C_PLATE, C_CANTI, C_BRACKET, NCONT };
#define MAXSNAP 72

typedef struct {
    int kind;
    int nn, ne;
    double *xy;                 /* nn*2 (mm) */
    int *q;                     /* ne*4, counter-clockwise */
    double E, nu, t, D[9];
    Csr K;
    int *map, *edof;
    double *kr;                 /* ne*32 */
    unsigned char *fix;
    int ncase;                  /* unit load cases, superposed for the display */
    double *f[2], *u[2], *sn[2];/* loads, displacements, nodal stresses (sx sy txy) */
    double load;                /* the magnitude of each case's unit load (N or MPa) */
    /* the solve, iterate by iterate: residuals and a few snapshots */
    int nres, nsnap, snap_it[MAXSNAP];
    float *res, *snap;          /* nres; nsnap * 2nn */
    int iters;
    double solve_ms;
    /* geometry, for annotations */
    double bx0, by0, bx1, by1;  /* bounding box */
    double a, b, len, h;        /* plate: hole radius, half width, half length; cantilever: length, depth */
    int gx, gy;                 /* structured: grid cells */
    double gdx;
    int *gnode;                 /* (gx+1)(gy+1) -> node or -1 */
    double tipx, tipy;          /* where the load acts (bracket, cantilever) */
    double check_fe, check_ref, check_ref2;
    /* principal stress trajectories (cantilever) */
    int ntraj, *traj_n, *traj_fam;
    float *traj;                /* concatenated polylines, x y */
    int traj_cap, traj_total;
} Cont;

/* node dedup by position */
typedef struct { int cap; int64_t *key; int *val; } PosHash;
static int node_at(Cont *c, PosHash *h, double x, double y) {
    int64_t kx = llround(x * 1e4), ky = llround(y * 1e4), key = kx * 1000003LL + ky;
    uint64_t i = ((uint64_t)key * 0x9E3779B97F4A7C15ULL) % (uint64_t)h->cap;
    while (h->val[i] >= 0) {
        if (h->key[i] == key) return h->val[i];
        i = (i + 1) % (uint64_t)h->cap;
    }
    h->key[i] = key;
    h->val[i] = c->nn;
    c->xy[c->nn * 2] = x;
    c->xy[c->nn * 2 + 1] = y;
    return c->nn++;
}

static void cont_alloc(Cont *c, int maxn, int maxe, PosHash *h) {
    c->xy = malloc(sizeof(double) * 2 * maxn);
    c->q = malloc(sizeof(int) * 4 * maxe);
    h->cap = maxn * 3 + 7;
    h->key = malloc(sizeof(int64_t) * h->cap);
    h->val = malloc(sizeof(int) * h->cap);
    for (int i = 0; i < h->cap; i++) h->val[i] = -1;
}

static void add_quad(Cont *c, int a, int b, int d, int e) {
    /* make it counter-clockwise */
    double *p = c->xy;
    double ar = (p[b * 2] - p[a * 2]) * (p[e * 2 + 1] - p[a * 2 + 1]) - (p[b * 2 + 1] - p[a * 2 + 1]) * (p[e * 2] - p[a * 2]);
    int *q = c->q + c->ne * 4;
    if (ar > 0) { q[0] = a; q[1] = b; q[2] = d; q[3] = e; }
    else { q[0] = a; q[1] = e; q[2] = d; q[3] = b; }
    c->ne++;
}

/* a structured block of nx x ny quads */
static void add_block(Cont *c, PosHash *h, int nx, int ny, double x0, double y0, double x1, double y1) {
    int *id = malloc(sizeof(int) * (nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; j++)
        for (int i = 0; i <= nx; i++)
            id[j * (nx + 1) + i] = node_at(c, h, x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny);
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            add_quad(c, id[j * (nx + 1) + i], id[j * (nx + 1) + i + 1], id[(j + 1) * (nx + 1) + i + 1], id[(j + 1) * (nx + 1) + i]);
    free(id);
}

/* plate with a central hole: an O-grid around the hole out to a square,
 * structured blocks either side */
static void mesh_plate(Cont *c, PosHash *h) {
    double a = 16, b = 80, L = 240;          /* hole radius, half width, half length (mm): W/D = 5 */
    c->a = a; c->b = b; c->len = L;
    int ns = 16, nr = 18, nt = 4 * ns;
    double g = 1.16;
    int nx = (int)lround((L - b) / (2 * b / ns));
    cont_alloc(c, nt * (nr + 1) + 2 * (nx + 1) * (ns + 1) + 16, nt * nr + 2 * nx * ns, h);
    int *id = malloc(sizeof(int) * nt * (nr + 1));
    for (int k = 0; k < nt; k++) {
        /* outer point k: along the square's perimeter, from (b, -b) counter-clockwise */
        int side = k / ns, i = k % ns;
        double s = -b + 2 * b * i / ns, ox, oy;
        if (side == 0) { ox = b; oy = s; }
        else if (side == 1) { ox = -s; oy = b; }
        else if (side == 2) { ox = -b; oy = -s; }
        else { ox = s; oy = -b; }
        double th = -0.25 * PI + 2 * PI * k / nt;
        double ix = a * cos(th), iy = a * sin(th);
        for (int r = 0; r <= nr; r++) {
            double f = (pow(g, r) - 1) / (pow(g, nr) - 1);
            id[k * (nr + 1) + r] = node_at(c, h, ix + f * (ox - ix), iy + f * (oy - iy));
        }
    }
    for (int k = 0; k < nt; k++)
        for (int r = 0; r < nr; r++) {
            int k2 = (k + 1) % nt;
            add_quad(c, id[k * (nr + 1) + r], id[k * (nr + 1) + r + 1], id[k2 * (nr + 1) + r + 1], id[k2 * (nr + 1) + r]);
        }
    free(id);
    add_block(c, h, nx, ns, b, -b, L, b);
    add_block(c, h, nx, ns, -L, -b, -b, b);
    c->bx0 = -L; c->bx1 = L; c->by0 = -b; c->by1 = b;
}

/* a structured grid of square cells dx, kept where keep() says */
static void mesh_grid(Cont *c, PosHash *h, int gx, int gy, double dx, int (*keep)(double, double)) {
    cont_alloc(c, (gx + 1) * (gy + 1), gx * gy, h);
    c->gx = gx; c->gy = gy; c->gdx = dx;
    c->gnode = malloc(sizeof(int) * (gx + 1) * (gy + 1));
    for (int i = 0; i < (gx + 1) * (gy + 1); i++) c->gnode[i] = -1;
    for (int j = 0; j < gy; j++)
        for (int i = 0; i < gx; i++) {
            if (!keep((i + 0.5) * dx, (j + 0.5) * dx)) continue;
            int n[4];
            for (int m = 0; m < 4; m++) {
                int ii = i + (m == 1 || m == 2), jj = j + (m >= 2);
                int *gn = &c->gnode[jj * (gx + 1) + ii];
                if (*gn < 0) *gn = node_at(c, h, ii * dx, jj * dx);
                n[m] = *gn;
            }
            add_quad(c, n[0], n[1], n[2], n[3]);
        }
    c->bx0 = 1e30; c->by0 = 1e30; c->bx1 = -1e30; c->by1 = -1e30;
    for (int i = 0; i < c->nn; i++) {
        c->bx0 = fmin(c->bx0, c->xy[i * 2]); c->bx1 = fmax(c->bx1, c->xy[i * 2]);
        c->by0 = fmin(c->by0, c->xy[i * 2 + 1]); c->by1 = fmax(c->by1, c->xy[i * 2 + 1]);
    }
}

static int keep_all(double x, double y) { (void)x; (void)y; return 1; }
/* L-bracket: a leg 60 wide, 240 tall, an arm 360 long and 60 deep across its top */
static int keep_bracket(double x, double y) { return x < 60 || y > 180; }

static void cont_solve_case(Cont *c, int k, int record) {
    Pcg s;
    pcg_init(&s, &c->K, c->fix);
    s.tol = 1e-9;
    double t0 = now();
    if (record) {
        c->res = malloc(sizeof(float) * 20000);
        c->snap = malloc(sizeof(float) * MAXSNAP * 2 * c->nn);
        c->nres = c->nsnap = 0;
        pcg_begin(&s, c->f[k]);
        int next = 0;
        for (;;) {
            if (c->nres < 20000) c->res[c->nres++] = (float)s.res;
            if ((s.it >= next || s.done) && c->nsnap < MAXSNAP) {
                c->snap_it[c->nsnap] = s.it;
                for (int i = 0; i < 2 * c->nn; i++) c->snap[(size_t)c->nsnap * 2 * c->nn + i] = (float)s.x[i];
                c->nsnap++;
                next = s.it < 8 ? s.it + 1 : (int)(s.it * 1.12) + 1;
            }
            if (s.done) break;
            pcg_run(&s, 1);
        }
        c->iters = s.it;
    } else
        pcg_solve(&s, c->f[k]);
    c->solve_ms += (now() - t0) * 1000;
    c->u[k] = malloc(sizeof(double) * 2 * c->nn);
    memcpy(c->u[k], s.x, sizeof(double) * 2 * c->nn);
    pcg_free(&s);
    /* nodal stresses: corner values averaged over the elements meeting there */
    c->sn[k] = calloc((size_t)c->nn * 3, sizeof(double));
    int *cnt = calloc((size_t)c->nn, sizeof(int));
    for (int e = 0; e < c->ne; e++) {
        double xy[8], ue[8], st[3];
        for (int a = 0; a < 4; a++) {
            int n = c->q[e * 4 + a];
            xy[a * 2] = c->xy[n * 2]; xy[a * 2 + 1] = c->xy[n * 2 + 1];
            ue[a * 2] = c->u[k][n * 2]; ue[a * 2 + 1] = c->u[k][n * 2 + 1];
        }
        for (int a = 0; a < 4; a++) {
            q6_stress(xy, c->D, c->kr + e * 32, ue, QX[a], QY[a], st);
            int n = c->q[e * 4 + a];
            for (int i = 0; i < 3; i++) c->sn[k][n * 3 + i] += st[i];
            cnt[n]++;
        }
    }
    for (int n = 0; n < c->nn; n++)
        for (int i = 0; i < 3; i++) c->sn[k][n * 3 + i] /= cnt[n] ? cnt[n] : 1;
    free(cnt);
}

static void cont_assemble(Cont *c) {
    dmat(c->E, c->nu, c->D);
    int n = 2 * c->nn;
    c->edof = malloc(sizeof(int) * 8 * c->ne);
    for (int e = 0; e < c->ne; e++)
        for (int a = 0; a < 4; a++) {
            c->edof[e * 8 + a * 2] = c->q[e * 4 + a] * 2;
            c->edof[e * 8 + a * 2 + 1] = c->q[e * 4 + a] * 2 + 1;
        }
    csr_pattern(&c->K, n, c->ne, 8, c->edof, &c->map);
    c->kr = malloc(sizeof(double) * 32 * c->ne);
    for (int e = 0; e < c->ne; e++) {
        double xy[8], k[64];
        for (int a = 0; a < 4; a++) {
            xy[a * 2] = c->xy[c->q[e * 4 + a] * 2];
            xy[a * 2 + 1] = c->xy[c->q[e * 4 + a] * 2 + 1];
        }
        q6_k(xy, c->D, c->t, k, c->kr + e * 32);
        for (int i = 0; i < 64; i++) c->K.v[c->map[e * 64 + i]] += k[i];
    }
    csr_diag(&c->K);
    c->fix = calloc((size_t)n, 1);
    for (int k = 0; k < 2; k++) c->f[k] = calloc((size_t)n, sizeof(double));
}

/* nodes on a straight edge (coordinate axis == v), sorted along it */
static int edge_nodes(const Cont *c, int axis, double v, int *out) {
    int m = 0;
    for (int i = 0; i < c->nn; i++)
        if (fabs(c->xy[i * 2 + axis] - v) < 1e-6) out[m++] = i;
    for (int i = 1; i < m; i++)
        for (int j = i; j > 0 && c->xy[out[j] * 2 + !axis] < c->xy[out[j - 1] * 2 + !axis]; j--) {
            int t = out[j]; out[j] = out[j - 1]; out[j - 1] = t;
        }
    return m;
}

/* a traction tr(s) (N/mm^2, s along the edge) in direction dir on the edge
 * axis == v: consistent nodal loads, Simpson along each element edge */
static void edge_load(Cont *c, int k, int axis, double v, int dir, double (*tr)(const Cont *, double)) {
    int *en = malloc(sizeof(int) * c->nn), m = edge_nodes(c, axis, v, en);
    for (int i = 0; i + 1 < m; i++) {
        double s0 = c->xy[en[i] * 2 + !axis], s1 = c->xy[en[i + 1] * 2 + !axis], ln = s1 - s0;
        double f0 = tr(c, s0), fm = tr(c, 0.5 * (s0 + s1)), f1 = tr(c, s1);
        c->f[k][en[i] * 2 + dir] += ln * c->t * (f0 + 2 * fm) / 6;
        c->f[k][en[i + 1] * 2 + dir] += ln * c->t * (f1 + 2 * fm) / 6;
    }
    free(en);
}

static double tr_unit(const Cont *c, double s) { (void)c; (void)s; return 1; }
static double tr_unit_neg(const Cont *c, double s) { (void)c; (void)s; return -1; }
/* parabolic shear over the cantilever's depth, totalling -1 N */
static double tr_shear(const Cont *c, double s) {
    double y = s - c->h / 2, A = c->h * c->t;
    return -1.5 / A * (1 - 4 * y * y / (c->h * c->h));
}

static int nearest_node(const Cont *c, double x, double y) {
    int best = 0;
    double bd = 1e300;
    for (int i = 0; i < c->nn; i++) {
        double d = (c->xy[i * 2] - x) * (c->xy[i * 2] - x) + (c->xy[i * 2 + 1] - y) * (c->xy[i * 2 + 1] - y);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

static void cont_trajectories(Cont *c);

static void cont_build(Cont *c, int kind) {
    memset(c, 0, sizeof *c);
    c->kind = kind;
    PosHash h;
    c->E = 200e3; c->nu = 0.3;              /* N/mm^2: steel */
    if (kind == C_PLATE) {
        c->t = 10;
        mesh_plate(c, &h);
        cont_assemble(c);
        c->ncase = 1;
        c->load = 100;                         /* sigma_inf, MPa */
        edge_load(c, 0, 0, c->len, 0, tr_unit);
        edge_load(c, 0, 0, -c->len, 0, tr_unit_neg);
        /* self-equilibrated: pin one end's middle, roller the other's */
        int l = nearest_node(c, -c->len, 0), r = nearest_node(c, c->len, 0);
        c->fix[l * 2] = c->fix[l * 2 + 1] = 1;
        c->fix[r * 2 + 1] = 1;
    } else if (kind == C_CANTI) {
        c->t = 20; c->len = 2000; c->h = 250;
        mesh_grid(c, &h, 64, 8, c->h / 8, keep_all);
        cont_assemble(c);
        c->ncase = 1;
        c->load = 20e3;                        /* P, N */
        edge_load(c, 0, 0, c->len, 1, tr_shear);
        for (int i = 0; i < c->nn; i++)
            if (c->xy[i * 2] < 1e-6) c->fix[i * 2] = c->fix[i * 2 + 1] = 1;
        c->tipx = c->len; c->tipy = c->h / 2;
    } else {
        c->t = 12;
        mesh_grid(c, &h, 72, 48, 5, keep_bracket);
        cont_assemble(c);
        c->ncase = 2;
        c->load = 5e3;                         /* N */
        /* the load spread over the arm's end face: x and y unit cases */
        int *en = malloc(sizeof(int) * c->nn), m = edge_nodes(c, 0, 360, en);
        for (int i = 0; i < m; i++) {
            double w = (i == 0 || i == m - 1 ? 0.5 : 1.0) / (m - 1);
            c->f[0][en[i] * 2] += w;
            c->f[1][en[i] * 2 + 1] += w;
        }
        free(en);
        for (int i = 0; i < c->nn; i++)
            if (c->xy[i * 2 + 1] < 1e-6) c->fix[i * 2] = c->fix[i * 2 + 1] = 1;
        c->tipx = 360; c->tipy = 210;
    }
    free(h.key); free(h.val);
    for (int k = 0; k < c->ncase; k++) cont_solve_case(c, k, k == 0);
    if (kind == C_PLATE) {
        /* Kt on the gross section: sigma_xx at the top of the hole over sigma_inf (unit) */
        int n = nearest_node(c, 0, c->a);
        c->check_fe = c->sn[0][n * 3];
        c->check_ref = 3.0;
        double dW = c->a / c->b;
        c->check_ref2 = (2 + pow(1 - dW, 3)) / (1 - dW);  /* Heywood's finite-width fit */
    } else if (kind == C_CANTI) {
        double I = c->t * pow(c->h, 3) / 12, G = c->E / (2 * (1 + c->nu)), A = c->t * c->h;
        int n = nearest_node(c, c->len, c->h / 2);
        c->check_fe = -c->u[0][n * 2 + 1];      /* per newton */
        c->check_ref = pow(c->len, 3) / (3 * c->E * I);
        c->check_ref2 = c->check_ref + c->len / (5.0 / 6 * G * A);
        cont_trajectories(c);
    }
}

static void cont_free(Cont *c) {
    free(c->xy); free(c->q); csr_free(&c->K); free(c->map); free(c->edof); free(c->kr); free(c->fix);
    for (int k = 0; k < 2; k++) { free(c->f[k]); free(c->u[k]); free(c->sn[k]); }
    free(c->res); free(c->snap); free(c->gnode);
    free(c->traj_n); free(c->traj_fam); free(c->traj);
    memset(c, 0, sizeof *c);
}

/* the nodal stress field at a point of a structured mesh, bilinear */
static int grid_stress(const Cont *c, const double *sn, double x, double y, double s[3]) {
    double fx = x / c->gdx, fy = y / c->gdx;
    int i = (int)floor(fx), j = (int)floor(fy);
    if (i < 0 || j < 0 || i >= c->gx || j >= c->gy) return 0;
    int n[4] = {c->gnode[j * (c->gx + 1) + i], c->gnode[j * (c->gx + 1) + i + 1],
                c->gnode[(j + 1) * (c->gx + 1) + i + 1], c->gnode[(j + 1) * (c->gx + 1) + i]};
    for (int m = 0; m < 4; m++) if (n[m] < 0) return 0;
    double u = fx - i, v = fy - j, w[4] = {(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v};
    for (int k = 0; k < 3; k++) s[k] = w[0] * sn[n[0] * 3 + k] + w[1] * sn[n[1] * 3 + k] + w[2] * sn[n[2] * 3 + k] + w[3] * sn[n[3] * 3 + k];
    return 1;
}

static int dir_at(const Cont *c, double x, double y, int fam, double *dx, double *dy) {
    double s[3];
    if (!grid_stress(c, c->sn[0], x, y, s)) return 0;
    double th = 0.5 * atan2(2 * s[2], s[0] - s[1]);
    if (fam) th += 0.5 * PI;
    *dx = cos(th); *dy = sin(th);
    return 1;
}

static void traj_push(Cont *c, float x, float y) {
    if (c->traj_total + 1 > c->traj_cap) c->traj = realloc(c->traj, sizeof(float) * 2 * (c->traj_cap = c->traj_cap ? c->traj_cap * 2 : 4096));
    c->traj[c->traj_total * 2] = x;
    c->traj[c->traj_total * 2 + 1] = y;
    c->traj_total++;
}

/* principal stress trajectories: lines everywhere tangent to sigma1 (fam
 * 0) or sigma2 (fam 1), traced both ways from seeds along the top and
 * bottom edges, RK2 */
static void cont_trajectories(Cont *c) {
    c->traj_n = malloc(sizeof(int) * 64);
    c->traj_fam = malloc(sizeof(int) * 64);
    double step = c->gdx * 0.25, eps = 1e-3;
    static float buf[2][4096 * 2];
    for (int fam = 0; fam < 2; fam++)
        for (int k = 0; k < 14; k++) {
            double sx = c->len * (k < 7 ? 0.04 + 0.135 * k : 0.105 + 0.135 * (k - 7));
            double sy = (k < 7) == !fam ? c->h - eps : eps;
            int nb[2] = {0, 0};
            for (int way = 0; way < 2; way++) {
                double x = sx, y = sy, pdx, pdy;
                if (!dir_at(c, x, y, fam, &pdx, &pdy)) break;
                if (way) { pdx = -pdx; pdy = -pdy; }
                for (int it = 0; it < 4000 && nb[way] < 4096; it++) {
                    buf[way][nb[way] * 2] = (float)x; buf[way][nb[way] * 2 + 1] = (float)y; nb[way]++;
                    double dx, dy, ex, ey;
                    if (!dir_at(c, x, y, fam, &dx, &dy)) break;
                    if (dx * pdx + dy * pdy < 0) { dx = -dx; dy = -dy; }
                    if (!dir_at(c, x + 0.5 * step * dx, y + 0.5 * step * dy, fam, &ex, &ey)) break;
                    if (ex * dx + ey * dy < 0) { ex = -ex; ey = -ey; }
                    x += step * ex; y += step * ey;
                    pdx = ex; pdy = ey;
                    if (x < 0 || y < 0 || x > c->len || y > c->h) break;
                }
            }
            if (nb[0] + nb[1] < 6 || c->ntraj >= 64) continue;
            int start = c->traj_total;
            for (int i = nb[1] - 1; i > 0; i--) traj_push(c, buf[1][i * 2], buf[1][i * 2 + 1]);
            for (int i = 0; i < nb[0]; i++) traj_push(c, buf[0][i * 2], buf[0][i * 2 + 1]);
            c->traj_n[c->ntraj] = c->traj_total - start;
            c->traj_fam[c->ntraj] = fam;
            c->ntraj++;
        }
}

/* ======================================================================
 * III  topology optimisation: the 88-line SIMP code (Andreassen, Clausen,
 * Schevenels, Lazarov, Sigmund 2011), sensitivity filter (ft = 1), OC
 * update, with the direct solve swapped for warm-started PCG; it runs a
 * piece at a time (top_work) so it can share a frame with the display
 * ====================================================================== */


/* geometric multigrid on the optimiser's structured grid, as the CG's
 * preconditioner (MGCG, after Amir, Aage & Lazarov 2014): bilinear
 * prolongation P, Galerkin coarse operators P^T A P built element by
 * element (a coarse element's matrix is the sum over its 2x2 children of
 * P_s^T k_child P_s), damped Jacobi smoothing (the same before and after,
 * so the V-cycle is symmetric), a dense Cholesky at the bottom */
#define MAXLVL 6
typedef struct {
    int nl, nx[MAXLVL], ny[MAXLVL], n[MAXLVL];
    Csr A[MAXLVL];
    int *map[MAXLVL];           /* element -> CSR slots, levels >= 1 */
    double *ke[MAXLVL];         /* element matrices, levels >= 1 */
    unsigned char *fx[MAXLVL];  /* fixed dofs at each level */
    int *par[MAXLVL];           /* fine dof -> 4 coarse parents (-1 none) */
    double *pw[MAXLVL];
    double *dinv[MAXLVL], *b[MAXLVL], *x[MAXLVL], *t[MAXLVL];
    double Ps[4][64];           /* a child element's dofs from its parent's */
    double Ms[4][64];           /* P_s^T KE P_s */
    double *L;                  /* coarsest, dense Cholesky factor */
} Mg;

static void grid_edof(int nx, int ny, int *edof) {
    for (int ex = 0; ex < nx; ex++)
        for (int ey = 0; ey < ny; ey++) {
            int e = ex * ny + ey, n1 = ex * (ny + 1) + ey, n2 = n1 + ny + 1;
            int d[8] = {2 * (n1 + 1), 2 * (n1 + 1) + 1, 2 * (n2 + 1), 2 * (n2 + 1) + 1, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1};
            memcpy(edof + e * 8, d, sizeof d);
        }
}

/* o = P^T k P, all 8x8 */
static void ptkp(const double *Pm, const double *k, double *o) {
    double t[64];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double s = 0;
            for (int m = 0; m < 8; m++) s += k[i * 8 + m] * Pm[m * 8 + j];
            t[i * 8 + j] = s;
        }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double s = 0;
            for (int m = 0; m < 8; m++) s += Pm[m * 8 + i] * t[m * 8 + j];
            o[i * 8 + j] = s;
        }
}

static void mg_init(Mg *g, const Csr *K, int nelx, int nely, const unsigned char *fix, const double *KE) {
    memset(g, 0, sizeof *g);
    /* corners in the element's dof order (LL, LR, UR, UL), as (column, row) with rows downward */
    static const int cc[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    for (int s = 0; s < 4; s++) {
        int sx = s & 1, sy = s >> 1;
        memset(g->Ps[s], 0, sizeof g->Ps[s]);
        for (int f = 0; f < 4; f++) {
            double X = (sx + cc[f][0]) * 0.5, Y = (sy + cc[f][1]) * 0.5;
            for (int c = 0; c < 4; c++) {
                double w = (cc[c][0] ? X : 1 - X) * (cc[c][1] ? Y : 1 - Y);
                for (int d = 0; d < 2; d++) g->Ps[s][(2 * f + d) * 8 + 2 * c + d] = w;
            }
        }
        ptkp(g->Ps[s], KE, g->Ms[s]);
    }
    int nx = nelx, ny = nely;
    for (int l = 0; l < MAXLVL; l++) {
        g->nx[l] = nx; g->ny[l] = ny; g->n[l] = 2 * (nx + 1) * (ny + 1);
        g->nl = l + 1;
        int n = g->n[l];
        g->fx[l] = calloc((size_t)n, 1);
        if (l == 0) {
            Csr *A = &g->A[0];
            *A = *K;
            A->rp = malloc(sizeof(int) * (K->n + 1)); memcpy(A->rp, K->rp, sizeof(int) * (K->n + 1));
            A->ci = malloc(sizeof(int) * K->nnz); memcpy(A->ci, K->ci, sizeof(int) * K->nnz);
            A->v = malloc(sizeof(double) * K->nnz);
            A->diag = malloc(sizeof(double) * K->n);
            memcpy(g->fx[0], fix, (size_t)n);
        } else {
            int *ed = malloc(sizeof(int) * 8 * nx * ny);
            grid_edof(nx, ny, ed);
            csr_pattern(&g->A[l], n, nx * ny, 8, ed, &g->map[l]);
            free(ed);
            g->ke[l] = malloc(sizeof(double) * 64 * nx * ny);
            /* a coarse dof is held where the fine node it sits on is */
            int fny = g->ny[l - 1];
            for (int ix = 0; ix <= nx; ix++)
                for (int iy = 0; iy <= ny; iy++)
                    for (int d = 0; d < 2; d++)
                        g->fx[l][2 * (ix * (ny + 1) + iy) + d] = g->fx[l - 1][2 * (2 * ix * (fny + 1) + 2 * iy) + d];
        }
        g->dinv[l] = malloc(sizeof(double) * n);
        g->b[l] = malloc(sizeof(double) * n);
        g->x[l] = malloc(sizeof(double) * n);
        g->t[l] = malloc(sizeof(double) * n);
        if (nx % 2 || ny % 2 || n <= 400 || l == MAXLVL - 1) break;
        /* the prolongation from the next level down */
        g->par[l] = malloc(sizeof(int) * 4 * n);
        g->pw[l] = malloc(sizeof(double) * 4 * n);
        int cny = ny / 2;
        for (int ix = 0; ix <= nx; ix++)
            for (int iy = 0; iy <= ny; iy++) {
                int cx[2], cy[2], ncx, ncy;
                double wx[2], wy[2];
                if (ix % 2 == 0) { cx[0] = ix / 2; wx[0] = 1; ncx = 1; }
                else { cx[0] = ix / 2; cx[1] = ix / 2 + 1; wx[0] = wx[1] = 0.5; ncx = 2; }
                if (iy % 2 == 0) { cy[0] = iy / 2; wy[0] = 1; ncy = 1; }
                else { cy[0] = iy / 2; cy[1] = iy / 2 + 1; wy[0] = wy[1] = 0.5; ncy = 2; }
                for (int d = 0; d < 2; d++) {
                    int i = 2 * (ix * (ny + 1) + iy) + d, k = 0;
                    for (int a = 0; a < ncx; a++)
                        for (int b = 0; b < ncy; b++) {
                            g->par[l][i * 4 + k] = 2 * (cx[a] * (cny + 1) + cy[b]) + d;
                            g->pw[l][i * 4 + k] = wx[a] * wy[b];
                            k++;
                        }
                    for (; k < 4; k++) { g->par[l][i * 4 + k] = -1; g->pw[l][i * 4 + k] = 0; }
                }
            }
        nx /= 2; ny /= 2;
    }
    int nc = g->n[g->nl - 1];
    g->L = malloc(sizeof(double) * nc * nc);
}

static void mg_free(Mg *g) {
    for (int l = 0; l < g->nl; l++) {
        csr_free(&g->A[l]);
        free(g->map[l]); free(g->ke[l]); free(g->fx[l]);
        free(g->par[l]); free(g->pw[l]); free(g->dinv[l]); free(g->b[l]); free(g->x[l]); free(g->t[l]);
    }
    free(g->L);
    memset(g, 0, sizeof *g);
}

/* held dofs: rows and columns out, a one on the diagonal */
static void csr_mask(Csr *A, const unsigned char *fx) {
    for (int i = 0; i < A->n; i++)
        for (int k = A->rp[i]; k < A->rp[i + 1]; k++)
            if (fx[i] || fx[A->ci[k]]) A->v[k] = A->ci[k] == i ? 1 : 0;
}

/* the operators for the current design (Ee: each element's modulus) */
static void mg_setup(Mg *g, const Csr *K, const double *Ee) {
    Csr *A0 = &g->A[0];
    memcpy(A0->v, K->v, sizeof(double) * K->nnz);
    csr_mask(A0, g->fx[0]);
    for (int l = 1; l < g->nl; l++) {
        int nx = g->nx[l], ny = g->ny[l], fny = g->ny[l - 1];
        for (int cx = 0; cx < nx; cx++)
            for (int cy = 0; cy < ny; cy++) {
                double *k = g->ke[l] + (size_t)(cx * ny + cy) * 64, m[64];
                memset(k, 0, sizeof(double) * 64);
                for (int s = 0; s < 4; s++) {
                    int fe = (2 * cx + (s & 1)) * fny + 2 * cy + (s >> 1);
                    if (l == 1) { for (int i = 0; i < 64; i++) k[i] += Ee[fe] * g->Ms[s][i]; }
                    else {
                        ptkp(g->Ps[s], g->ke[l - 1] + (size_t)fe * 64, m);
                        for (int i = 0; i < 64; i++) k[i] += m[i];
                    }
                }
            }
        Csr *A = &g->A[l];
        memset(A->v, 0, sizeof(double) * A->nnz);
        for (int e = 0; e < nx * ny; e++)
            for (int i = 0; i < 64; i++) A->v[g->map[l][e * 64 + i]] += g->ke[l][(size_t)e * 64 + i];
        csr_mask(A, g->fx[l]);
    }
    for (int l = 0; l < g->nl; l++) {
        csr_diag(&g->A[l]);
        for (int i = 0; i < g->n[l]; i++) g->dinv[l][i] = g->A[l].diag[i] > 0 ? 1 / g->A[l].diag[i] : 0;
    }
    /* the coarsest, dense, factored */
    const Csr *C = &g->A[g->nl - 1];
    int n = C->n;
    double *L = g->L;
    memset(L, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
        for (int k = C->rp[i]; k < C->rp[i + 1]; k++) L[i * n + C->ci[k]] = C->v[k];
    for (int j = 0; j < n; j++) {
        double d = L[j * n + j];
        for (int k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
        d = sqrt(fmax(d, 1e-300));
        L[j * n + j] = d;
        for (int i = j + 1; i < n; i++) {
            double s = L[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / d;
        }
    }
}

static void mg_smooth(Mg *g, int l, int nu) {
    const double w = 0.6;
    for (int s = 0; s < nu; s++) {
        csr_mul(&g->A[l], g->x[l], g->t[l]);
        for (int i = 0; i < g->n[l]; i++) g->x[l][i] += w * g->dinv[l][i] * (g->b[l][i] - g->t[l][i]);
    }
}

static void mg_vcycle(Mg *g, int l) {
    int n = g->n[l];
    if (l == g->nl - 1) {
        double *L = g->L, *x = g->x[l];
        for (int i = 0; i < n; i++) {
            double s = g->b[l][i];
            for (int k = 0; k < i; k++) s -= L[i * n + k] * x[k];
            x[i] = s / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = x[i];
            for (int k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
            x[i] = s / L[i * n + i];
        }
        return;
    }
    memset(g->x[l], 0, sizeof(double) * n);
    mg_smooth(g, l, 2);
    csr_mul(&g->A[l], g->x[l], g->t[l]);
    int nc = g->n[l + 1];
    memset(g->b[l + 1], 0, sizeof(double) * nc);
    for (int i = 0; i < n; i++) {
        double r = g->b[l][i] - g->t[l][i];
        for (int a = 0; a < 4 && g->par[l][i * 4 + a] >= 0; a++) g->b[l + 1][g->par[l][i * 4 + a]] += g->pw[l][i * 4 + a] * r;
    }
    mg_vcycle(g, l + 1);
    for (int i = 0; i < n; i++)
        for (int a = 0; a < 4 && g->par[l][i * 4 + a] >= 0; a++) g->x[l][i] += g->pw[l][i * 4 + a] * g->x[l + 1][g->par[l][i * 4 + a]];
    mg_smooth(g, l, 2);
}

static void mg_prec(void *ctx, const double *r, double *z) {
    Mg *g = ctx;
    memcpy(g->b[0], r, sizeof(double) * g->n[0]);
    mg_vcycle(g, 0);
    memcpy(z, g->x[0], sizeof(double) * g->n[0]);
}

enum { T_MBB, T_BRIDGE, NTOP };
#define TOP_MAXIT 240

typedef struct {
    int kind, nelx, nely, ne, n;
    double volfrac, penal, rmin, E0, Emin, nu, KE[64];
    int *edof, *map;
    Csr K;
    double *x, *xPhys, *xnew, *dc, *ce, *F, *U, *Ee;
    unsigned char *fix, *passive;
    int *Hp, *Hi;               /* the filter, element-by-element (CSR) */
    double *Hv, *Hs;
    Pcg pcg;
    Mg mg;
    int iter, phase, done, cg_last;
    double c, change, vol;
    /* the history, one design per iteration, for the display */
    float *hist;                /* (TOP_MAXIT+1) * ne */
    float hc[TOP_MAXIT + 1], hvol[TOP_MAXIT + 1], hchg[TOP_MAXIT + 1];
    int hcg[TOP_MAXIT + 1];
    int nhist;
    double work_ms;
} Top;

static void top_ke(double nu, double KE[64]) {
    static const double A11[16] = {12, 3, -6, -3, 3, 12, 3, 0, -6, 3, 12, -3, -3, 0, -3, 12};
    static const double A12[16] = {-6, -3, 0, 3, -3, -6, -3, -6, 0, -3, -6, 3, 3, -6, 3, -6};
    static const double B11[16] = {-4, 3, -2, 9, 3, -4, -9, 4, -2, -9, -4, -3, 9, 4, -3, -4};
    static const double B12[16] = {2, -3, 4, -9, -3, 2, 9, -2, 4, 9, 2, 3, -9, -2, 3, 2};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            int bi = i / 4, bj = j / 4, ii = i % 4, jj = j % 4;
            double a, b;
            if (bi == bj) { a = A11[ii * 4 + jj]; b = B11[ii * 4 + jj]; }
            else if (bi == 0) { a = A12[ii * 4 + jj]; b = B12[ii * 4 + jj]; }
            else { a = A12[jj * 4 + ii]; b = B12[jj * 4 + ii]; }     /* the transposes */
            KE[i * 8 + j] = (a + nu * b) / (1 - nu * nu) / 24;
        }
}

static void top_build(Top *T, int kind, int nelx, int nely, double volfrac, double rmin) {
    memset(T, 0, sizeof *T);
    T->kind = kind; T->nelx = nelx; T->nely = nely;
    T->ne = nelx * nely;
    T->n = 2 * (nelx + 1) * (nely + 1);
    T->volfrac = volfrac; T->penal = 3; T->rmin = rmin;
    T->E0 = 1; T->Emin = 1e-9; T->nu = 0.3;
    top_ke(T->nu, T->KE);
    T->edof = malloc(sizeof(int) * 8 * T->ne);
    grid_edof(nelx, nely, T->edof);
    csr_pattern(&T->K, T->n, T->ne, 8, T->edof, &T->map);
    T->x = malloc(sizeof(double) * T->ne);
    T->xPhys = malloc(sizeof(double) * T->ne);
    T->xnew = malloc(sizeof(double) * T->ne);
    T->dc = malloc(sizeof(double) * T->ne);
    T->ce = malloc(sizeof(double) * T->ne);
    T->F = calloc((size_t)T->n, sizeof(double));
    T->U = calloc((size_t)T->n, sizeof(double));
    T->fix = calloc((size_t)T->n, 1);
    T->passive = calloc((size_t)T->ne, 1);
    if (kind == T_MBB) {
        /* half the MBB beam: a load down at the top left, symmetry on the left edge, a roller at the bottom right */
        T->F[1] = -1;
        for (int j = 0; j <= nely; j++) T->fix[2 * j] = 1;
        T->fix[T->n - 1] = 1;
    } else {
        /* half a deck bridge: a solid deck along the top under a uniform
         * load, symmetry at midspan (left), pinned at the abutment (bottom right) */
        for (int i = 0; i <= nelx; i++) T->F[2 * (i * (nely + 1)) + 1] = -(i == 0 || i == nelx ? 0.5 : 1.0) / nelx;
        for (int j = 0; j <= nely; j++) T->fix[2 * j] = 1;
        T->fix[T->n - 1] = T->fix[T->n - 2] = 1;
        for (int ex = 0; ex < nelx; ex++) T->passive[ex * nely] = 1;
    }
    for (int e = 0; e < T->ne; e++) T->x[e] = T->xPhys[e] = T->passive[e] ? 1 : volfrac;
    /* the filter weights */
    int R = (int)ceil(rmin) - 1, maxn = (2 * R + 1) * (2 * R + 1);
    T->Hp = malloc(sizeof(int) * (T->ne + 1));
    T->Hi = malloc(sizeof(int) * T->ne * maxn);
    T->Hv = malloc(sizeof(double) * T->ne * maxn);
    T->Hs = calloc((size_t)T->ne, sizeof(double));
    int k = 0;
    for (int i1 = 0; i1 < nelx; i1++)
        for (int j1 = 0; j1 < nely; j1++) {
            int e1 = i1 * nely + j1;
            T->Hp[e1] = k;
            for (int i2 = i1 - R > 0 ? i1 - R : 0; i2 <= i1 + R && i2 < nelx; i2++)
                for (int j2 = j1 - R > 0 ? j1 - R : 0; j2 <= j1 + R && j2 < nely; j2++) {
                    double w = rmin - sqrt((double)(i1 - i2) * (i1 - i2) + (double)(j1 - j2) * (j1 - j2));
                    if (w <= 0) continue;
                    T->Hi[k] = i2 * nely + j2;
                    T->Hv[k] = w;
                    T->Hs[e1] += w;
                    k++;
                }
        }
    T->Hp[T->ne] = k;
    pcg_init(&T->pcg, &T->K, T->fix);
    T->pcg.tol = 1e-6;
    mg_init(&T->mg, &T->K, nelx, nely, T->fix, T->KE);
    T->Ee = malloc(sizeof(double) * T->ne);
    T->pcg.prec = mg_prec;
    T->pcg.pctx = &T->mg;
    T->hist = malloc(sizeof(float) * (size_t)(TOP_MAXIT + 1) * T->ne);
    for (int e = 0; e < T->ne; e++) T->hist[e] = (float)T->xPhys[e];
    T->hvol[0] = (float)volfrac;
    T->nhist = 1;
}

static void top_free(Top *T) {
    free(T->edof); free(T->map); csr_free(&T->K);
    free(T->Ee); free(T->x); free(T->xPhys); free(T->xnew); free(T->dc); free(T->ce); free(T->F); free(T->U);
    free(T->fix); free(T->passive); free(T->Hp); free(T->Hi); free(T->Hv); free(T->Hs);
    pcg_free(&T->pcg); mg_free(&T->mg); free(T->hist);
    memset(T, 0, sizeof *T);
}

static void top_assemble(Top *T) {
    memset(T->K.v, 0, sizeof(double) * T->K.nnz);
    for (int e = 0; e < T->ne; e++) {
        double Ee = T->Ee[e] = T->Emin + pow(T->xPhys[e], T->penal) * (T->E0 - T->Emin);
        const int *m = T->map + e * 64;
        for (int i = 0; i < 64; i++) T->K.v[m[i]] += Ee * T->KE[i];
    }
    csr_diag(&T->K);
}

/* after the solve: sensitivities, filter, OC */
static void top_update(Top *T) {
    double *U = T->pcg.x;
    memcpy(T->U, U, sizeof(double) * T->n);
    double c = 0;
    for (int e = 0; e < T->ne; e++) {
        const int *d = T->edof + e * 8;
        double ce = 0;
        for (int i = 0; i < 8; i++) {
            double s = 0;
            for (int j = 0; j < 8; j++) s += T->KE[i * 8 + j] * U[d[j]];
            ce += s * U[d[i]];
        }
        T->ce[e] = ce;
        c += (T->Emin + pow(T->xPhys[e], T->penal) * (T->E0 - T->Emin)) * ce;
        T->dc[e] = -T->penal * (T->E0 - T->Emin) * pow(T->xPhys[e], T->penal - 1) * ce;
    }
    T->c = c;
    /* sensitivity filter */
    for (int e = 0; e < T->ne; e++) {
        double s = 0;
        for (int k = T->Hp[e]; k < T->Hp[e + 1]; k++) s += T->Hv[k] * T->x[T->Hi[k]] * T->dc[T->Hi[k]];
        T->xnew[e] = s / T->Hs[e] / fmax(1e-3, T->x[e]);
    }
    memcpy(T->dc, T->xnew, sizeof(double) * T->ne);
    /* optimality criteria, bisecting on the volume */
    double l1 = 0, l2 = 1e9, move = 0.2, target = T->volfrac * T->ne;
    while ((l2 - l1) / (l1 + l2) > 1e-3) {
        double lmid = 0.5 * (l2 + l1), vol = 0;
        for (int e = 0; e < T->ne; e++) {
            double xe = T->x[e], v;
            if (T->passive[e]) v = 1;
            else {
                v = xe * sqrt(fmax(-T->dc[e], 0) / lmid);
                v = fmin(fmin(v, xe + move), 1);
                v = fmax(fmax(v, xe - move), 0);
            }
            T->xnew[e] = v;
            vol += v;
        }
        if (vol > target) l1 = lmid; else l2 = lmid;
    }
    double ch = 0, vol = 0;
    for (int e = 0; e < T->ne; e++) {
        ch = fmax(ch, fabs(T->xnew[e] - T->x[e]));
        T->x[e] = T->xPhys[e] = T->xnew[e];
        vol += T->xnew[e];
    }
    T->change = ch;
    T->vol = vol / T->ne;
    T->iter++;
    int h = T->nhist;
    if (h <= TOP_MAXIT) {
        for (int e = 0; e < T->ne; e++) T->hist[(size_t)h * T->ne + e] = (float)T->xPhys[e];
        T->hc[h - 1] = (float)c;          /* the compliance of the design before this update */
        T->hvol[h] = (float)T->vol;
        T->hchg[h] = (float)ch;
        T->hcg[h] = T->cg_last;
        T->nhist++;
    }
    if (ch < 0.01 || T->iter >= TOP_MAXIT) T->done = 1;
}

/* do up to budget ms of work (budget <= 0: until done), a small piece at
 * a time so a frame never waits long: assemble, set up the multigrid, a
 * couple of CG iterations, the update; returns done */
static int top_work(Top *T, double budget_ms) {
    double t0 = now();
    while (!T->done) {
        if (T->phase == 0) { top_assemble(T); T->phase = 1; }
        else if (T->phase == 1) {
            mg_setup(&T->mg, &T->K, T->Ee);
            pcg_begin(&T->pcg, T->F);       /* pcg.x still holds the last U: the warm start */
            T->phase = 2;
        } else if (T->phase == 2) {
            if (pcg_run(&T->pcg, 2)) T->phase = 3;
        } else {
            T->cg_last = T->pcg.it;
            top_update(T);
            T->phase = 0;
        }
        if (budget_ms > 0 && (now() - t0) * 1000 > budget_ms) break;
    }
    T->work_ms += (now() - t0) * 1000;
    return T->done;
}

/* ======================================================================
 * I  bridges: 2D frames and (tension-only) truss bars under trains
 * ====================================================================== */

enum { B_WARREN, B_PRATT, B_ARCH, B_STAYED, B_SUSP, NBRIDGE };
enum { G_CHORD, G_WEB, G_DECK, G_CABLE, G_HANGER, G_PYLON, G_ARCH };

typedef struct { int a, b, truss, tonly, grp; double A, I, E, rho, L, pre; } Bar;   /* pre: cable pretension (N) */

#define MAXTRAIN 6
#define MAXAXLE 64
typedef struct {
    double t0, v, s0;           /* starts at time t0 with its head at s0, speed v (m/s, signed) */
    int naxle;
    double ax[MAXAXLE], P[MAXAXLE];   /* axle offsets behind the head (m), loads (N) */
    int ncar;
    double car0[12], car1[12];  /* car extents behind the head */
    int loco[12];
    double len;
} Train;

typedef struct {
    int kind;
    const char *name;
    int nn, nb, n;
    double *px, *py;
    Bar *e;
    Csr K;
    int *map, *edof;
    unsigned char *fix, *slack;
    double *M, *ud, *fd, *Nd;
    int ndeck, *deck;           /* deck bars by x */
    double x0, x1, span, height, deck_y, ground, mon_x, sag;
    int nsup, sup_n[8], sup_t[8];   /* 0 pin, 1 roller, 2 fixed */
    double q_dead;
    /* dynamics: Newmark (average acceleration) with Rayleigh damping */
    double dt, alpha, betar, f1, zeta;
    double *u, *v, *a, *u0, *f, *rhs, *w, *t1;
    Pcg pcg;
    int step, cg_last, cg_sum;
    double scale, smax;         /* the deflection exaggeration; colour scale (Pa) */
    double vmax_live;
    int ntrain;
    Train tr[MAXTRAIN];
    float hist[12000];          /* the monitored deflection, every step (m) */
    int nhist;
    double env_max, env_min;    /* the live deflection envelope so far */
    double smax_now, smin_now;
} Br;

static double br_bar_len(const Br *B, int i) {
    double dx = B->px[B->e[i].b] - B->px[B->e[i].a], dy = B->py[B->e[i].b] - B->py[B->e[i].a];
    return sqrt(dx * dx + dy * dy);
}

static int br_node(Br *B, double x, double y) {
    for (int i = 0; i < B->nn; i++)
        if (fabs(B->px[i] - x) < 1e-6 && fabs(B->py[i] - y) < 1e-6) return i;
    B->px[B->nn] = x; B->py[B->nn] = y;
    return B->nn++;
}

static void br_bar(Br *B, int a, int b, int grp, int truss, int tonly, double A, double I) {
    Bar *e = &B->e[B->nb++];
    memset(e, 0, sizeof *e);
    e->a = a; e->b = b; e->grp = grp; e->truss = truss; e->tonly = tonly;
    e->A = A; e->I = I; e->E = 210e9; e->rho = 7850;
}

static void br_sup(Br *B, int n, int t) { B->sup_n[B->nsup] = n; B->sup_t[B->nsup] = t; B->nsup++; }

static void br_geometry(Br *B, int kind) {
    B->px = malloc(sizeof(double) * 256);
    B->py = malloc(sizeof(double) * 256);
    B->e = malloc(sizeof(Bar) * 512);
    B->deck = malloc(sizeof(int) * 128);
    int dn[64];
    if (kind == B_WARREN || kind == B_PRATT) {
        int n = 8;
        double L = 80, h = kind == B_WARREN ? 9 : 11, p = L / n;
        B->name = kind == B_WARREN ? "WARREN TRUSS" : "PRATT TRUSS";
        B->q_dead = 60e3;
        for (int i = 0; i <= n; i++) dn[i] = br_node(B, i * p, 0);
        for (int i = 0; i < n; i++) { B->deck[B->ndeck++] = B->nb; br_bar(B, dn[i], dn[i + 1], G_DECK, 0, 0, 0.030, 0.012); }
        if (kind == B_WARREN) {
            int tn[16];
            for (int j = 0; j < n; j++) tn[j] = br_node(B, (j + 0.5) * p, h);
            for (int j = 0; j + 1 < n; j++) br_bar(B, tn[j], tn[j + 1], G_CHORD, 0, 0, 0.032, 0.004);
            for (int j = 0; j < n; j++) {
                br_bar(B, dn[j], tn[j], G_WEB, 1, 0, 0.014, 0);
                br_bar(B, tn[j], dn[j + 1], G_WEB, 1, 0, 0.014, 0);
            }
        } else {
            int tn[16];
            for (int i = 1; i < n; i++) tn[i] = br_node(B, i * p, h);
            for (int i = 1; i + 1 < n; i++) br_bar(B, tn[i], tn[i + 1], G_CHORD, 0, 0, 0.034, 0.004);
            br_bar(B, dn[0], tn[1], G_CHORD, 0, 0, 0.034, 0.004);
            br_bar(B, tn[n - 1], dn[n], G_CHORD, 0, 0, 0.034, 0.004);
            for (int i = 1; i < n; i++) br_bar(B, dn[i], tn[i], G_WEB, 1, 0, 0.010, 0);
            for (int i = 1; i < n - 1; i++) {
                if (i < n / 2) br_bar(B, tn[i], dn[i + 1], G_WEB, 1, 0, 0.012, 0);
                else br_bar(B, tn[i + 1], dn[i], G_WEB, 1, 0, 0.012, 0);
            }
        }
        br_sup(B, dn[0], 0); br_sup(B, dn[n], 1);
        B->x0 = 0; B->x1 = L; B->height = h; B->ground = -12; B->mon_x = L / 2;
    } else if (kind == B_ARCH) {
        int n = 12;
        double L = 96, f = 18, p = L / n;
        B->name = "TIED ARCH";
        B->q_dead = 70e3;
        int an[16];
        for (int i = 0; i <= n; i++) dn[i] = br_node(B, i * p, 0);
        for (int i = 0; i < n; i++) { B->deck[B->ndeck++] = B->nb; br_bar(B, dn[i], dn[i + 1], G_DECK, 0, 0, 0.040, 0.030); }
        an[0] = dn[0]; an[n] = dn[n];
        for (int i = 1; i < n; i++) {
            double x = i * p;
            an[i] = br_node(B, x, 4 * f * x * (L - x) / (L * L));
        }
        for (int i = 0; i < n; i++) br_bar(B, an[i], an[i + 1], G_ARCH, 0, 0, 0.060, 0.020);
        for (int i = 1; i < n; i++) br_bar(B, dn[i], an[i], G_HANGER, 1, 1, 0.004, 0);
        br_sup(B, dn[0], 0); br_sup(B, dn[n], 1);
        B->x0 = 0; B->x1 = L; B->height = f; B->ground = -14; B->mon_x = L / 2;
    } else if (kind == B_STAYED) {
        int n = 20;
        double L = 160, p = L / n, hp = 42, base = -16;
        B->name = "CABLE-STAYED";
        B->q_dead = 90e3;
        for (int i = 0; i <= n; i++) dn[i] = br_node(B, i * p, 0);
        for (int i = 0; i < n; i++) { B->deck[B->ndeck++] = B->nb; br_bar(B, dn[i], dn[i + 1], G_DECK, 0, 0, 0.060, 0.050); }
        int pb = br_node(B, L / 2, base), m = n / 2;
        br_bar(B, pb, dn[m], G_PYLON, 0, 0, 0.5, 0.8);
        int prev = dn[m], anc[12];
        for (int k = 2; k <= m; k++) {
            double y = hp - (m - k) * 2.2;
            anc[k] = br_node(B, L / 2, y);
        }
        for (int k = 2; k <= m; k++) { br_bar(B, prev, anc[k], G_PYLON, 0, 0, 0.4, 0.6); prev = anc[k]; }
        for (int k = 2; k <= m; k++) {
            br_bar(B, dn[m - k], anc[k], G_CABLE, 1, 1, 0.005, 0);
            br_bar(B, dn[m + k], anc[k], G_CABLE, 1, 1, 0.005, 0);
        }
        br_sup(B, dn[0], 1); br_sup(B, dn[n], 1); br_sup(B, pb, 2);
        B->x0 = 0; B->x1 = L; B->height = hp; B->ground = base; B->mon_x = L * 0.25;
    } else {
        int n = 16;
        double L = 128, p = L / n, ht = 30, sag = 24, base = -15;
        B->name = "SUSPENSION";
        B->q_dead = 60e3;
        for (int i = 0; i <= n; i++) dn[i] = br_node(B, i * p, 0);
        for (int i = 0; i < n; i++) { B->deck[B->ndeck++] = B->nb; br_bar(B, dn[i], dn[i + 1], G_DECK, 0, 0, 0.040, 0.025); }
        int t0b = br_node(B, 0, base), t1b = br_node(B, L, base);
        int t0 = br_node(B, 0, ht), t1 = br_node(B, L, ht);
        int t0m = br_node(B, 0, -1.5), t1m = br_node(B, L, -1.5);
        br_bar(B, t0b, t0m, G_PYLON, 0, 0, 0.4, 0.5); br_bar(B, t0m, t0, G_PYLON, 0, 0, 0.4, 0.5);
        br_bar(B, t1b, t1m, G_PYLON, 0, 0, 0.4, 0.5); br_bar(B, t1m, t1, G_PYLON, 0, 0, 0.4, 0.5);
        /* the deck rides on the towers' cross beams: short posts to its ends */
        br_bar(B, t0m, dn[0], G_PYLON, 1, 0, 0.1, 0);
        br_bar(B, t1m, dn[n], G_PYLON, 1, 0, 0.1, 0);
        int cn[20];
        cn[0] = t0; cn[n] = t1;
        for (int i = 1; i < n; i++) {
            double x = i * p;
            cn[i] = br_node(B, x, ht - sag * 4 * x * (L - x) / (L * L));
        }
        for (int i = 0; i < n; i++) br_bar(B, cn[i], cn[i + 1], G_CABLE, 1, 1, 0.030, 0);
        for (int i = 1; i < n; i++) br_bar(B, dn[i], cn[i], G_HANGER, 1, 1, 0.003, 0);
        int a0 = br_node(B, -0.45 * L, -3), a1 = br_node(B, 1.45 * L, -3);
        br_bar(B, a0, t0, G_CABLE, 1, 1, 0.030, 0);
        br_bar(B, t1, a1, G_CABLE, 1, 1, 0.030, 0);
        br_sup(B, t0b, 2); br_sup(B, t1b, 2); br_sup(B, a0, 0); br_sup(B, a1, 0);
        /* the deck's ends: a longitudinal restraint at one end */
        br_sup(B, dn[0], 3);
        B->x0 = 0; B->x1 = L; B->height = ht; B->ground = base; B->mon_x = L / 2; B->sag = sag;
    }
    B->span = B->x1 - B->x0;
    B->deck_y = 0;
}

/* (re)assemble K with the slack bars all but switched off */
static void br_assemble(Br *B) {
    memset(B->K.v, 0, sizeof(double) * B->K.nnz);
    for (int i = 0; i < B->nb; i++) {
        Bar *e = &B->e[i];
        double k[36];
        bar_k(B->px[e->a], B->py[e->a], B->px[e->b], B->py[e->b], e->E, e->A * (B->slack[i] ? 1e-6 : 1), e->I, e->truss, k);
        for (int j = 0; j < 36; j++)
            if (B->map[i * 36 + j] >= 0) B->K.v[B->map[i * 36 + j]] += k[j];
    }
    csr_diag(&B->K);
}

static double bar_axial(const Br *B, int i, const double *u1, const double *u2) {
    const Bar *e = &B->e[i];
    double dx = B->px[e->b] - B->px[e->a], dy = B->py[e->b] - B->py[e->a], L = e->L;
    double du = 0, dv = 0;
    if (u1) { du += u1[e->b * 3] - u1[e->a * 3]; dv += u1[e->b * 3 + 1] - u1[e->a * 3 + 1]; }
    if (u2) { du += u2[e->b * 3] - u2[e->a * 3]; dv += u2[e->b * 3 + 1] - u2[e->a * 3 + 1]; }
    return e->pre + e->E * e->A / L * (du * dx + dv * dy) / L;
}

/* the consistent nodal loads of a point load P (down) at x on the deck */
static void br_point_load(const Br *B, double *f, double x, double P) {
    if (x < B->x0 || x > B->x1) return;
    for (int k = 0; k < B->ndeck; k++) {
        const Bar *e = &B->e[B->deck[k]];
        double xa = B->px[e->a], xb = B->px[e->b];
        if (x < xa || x > xb) continue;
        double L = xb - xa, s = (x - xa) / L;
        f[e->a * 3 + 1] -= P * (1 - 3 * s * s + 2 * s * s * s);
        f[e->a * 3 + 2] -= P * L * (s - 2 * s * s + s * s * s);
        f[e->b * 3 + 1] -= P * (3 * s * s - 2 * s * s * s);
        f[e->b * 3 + 2] -= P * L * (-s * s + s * s * s);
        return;
    }
}

/* the live deck deflection at x (Hermite on the deck bar) */
static double br_deck_v(const Br *B, const double *u, double x) {
    if (x <= B->x0 || x >= B->x1) return 0;
    for (int k = 0; k < B->ndeck; k++) {
        const Bar *e = &B->e[B->deck[k]];
        double xa = B->px[e->a], xb = B->px[e->b];
        if (x < xa || x > xb) continue;
        double L = xb - xa, s = (x - xa) / L;
        return u[e->a * 3 + 1] * (1 - 3 * s * s + 2 * s * s * s) + u[e->a * 3 + 2] * L * (s - 2 * s * s + s * s * s) +
               u[e->b * 3 + 1] * (3 * s * s - 2 * s * s * s) + u[e->b * 3 + 2] * L * (-s * s + s * s * s);
    }
    return 0;
}

static void train_axles(Train *T, int nloco, int ncar, double Ploco, double Pcar) {
    T->naxle = 0; T->ncar = 0;
    double s = 0;
    for (int c = 0; c < nloco + ncar; c++) {
        int lo = c < nloco;
        double len = lo ? 21 : 19.5, b = lo ? 3.4 : 3.0;
        T->car0[T->ncar] = s + 0.6; T->car1[T->ncar] = s + len - 0.6; T->loco[T->ncar] = lo; T->ncar++;
        double ax[4] = {s + b - 1.3, s + b + 1.3, s + len - b - 1.3, s + len - b + 1.3};
        for (int i = 0; i < 4; i++) { T->ax[T->naxle] = ax[i]; T->P[T->naxle] = lo ? Ploco : Pcar; T->naxle++; }
        s += len;
    }
    T->len = s;
}

/* where axle i is at time t (x), or a big number before/after */
static double axle_x(const Train *T, int i, double t) {
    double head = T->s0 + T->v * (t - T->t0);
    return T->v > 0 ? head - T->ax[i] : head + T->ax[i];
}

static void br_loads(const Br *B, double t, double *f) {
    memset(f, 0, sizeof(double) * B->n);
    for (int k = 0; k < B->ntrain; k++) {
        const Train *T = &B->tr[k];
        if (t < T->t0) continue;
        for (int i = 0; i < T->naxle; i++) br_point_load(B, f, axle_x(T, i, t), T->P[i]);
    }
}

/* rounded up to a tidy scale value: 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k */
static double nice_up(double v) {
    static const double c[] = {1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10};
    double e = pow(10, floor(log10(v))), m = v / e;
    for (int i = 0; i < 11; i++) if (c[i] >= m - 1e-9) return c[i] * e;
    return 10 * e;
}

static double nice_num(double v, int down) {
    double e = pow(10, floor(log10(v))), m = v / e;
    double c[4] = {1, 2, 5, 10};
    if (down) { for (int i = 3; i >= 0; i--) if (c[i] <= m + 1e-9) return c[i] * e; return e; }
    for (int i = 0; i < 4; i++) if (c[i] >= m - 1e-9) return c[i] * e;
    return 10 * e;
}

#define BR_DT 0.01

static void br_build(Br *B, int kind, uint32_t seed) {
    memset(B, 0, sizeof *B);
    B->kind = kind;
    br_geometry(B, kind);
    B->n = 3 * B->nn;
    for (int i = 0; i < B->nb; i++) B->e[i].L = br_bar_len(B, i);
    /* stays and hangers are tensioned to carry their share of the deck (the
     * vertical component of the pretension is the deck's weight per panel) */
    {
        double p = B->span / B->ndeck, q = B->q_dead * 1.06;
        for (int i = 0; i < B->nb; i++) {
            Bar *e = &B->e[i];
            if (!(e->grp == G_HANGER || (e->grp == G_CABLE && B->kind == B_STAYED))) continue;
            double sn = fabs(B->py[e->b] - B->py[e->a]) / e->L;
            e->pre = q * p / fmax(sn, 0.2);
        }
        /* the suspension's main cable and backstays: the funicular's
         * horizontal tension H = qL^2/8f throughout */
        if (B->kind == B_SUSP) {
            double H = q * B->span * B->span / (8 * B->sag);
            for (int i = 0; i < B->nb; i++) {
                Bar *e = &B->e[i];
                if (e->grp != G_CABLE) continue;
                double cs = fabs(B->px[e->b] - B->px[e->a]) / e->L;
                e->pre = H / cs;
            }
        }
    }
    B->edof = malloc(sizeof(int) * 6 * B->nb);
    for (int i = 0; i < B->nb; i++)
        for (int j = 0; j < 3; j++) {
            B->edof[i * 6 + j] = B->e[i].a * 3 + j;
            B->edof[i * 6 + 3 + j] = B->e[i].b * 3 + j;
        }
    csr_pattern(&B->K, B->n, B->nb, 6, B->edof, &B->map);
    B->fix = calloc((size_t)B->n, 1);
    B->slack = calloc((size_t)B->nb, 1);
    /* rotations only where a frame member meets the node */
    for (int nd = 0; nd < B->nn; nd++) {
        int fr = 0;
        for (int i = 0; i < B->nb; i++) if (!B->e[i].truss && (B->e[i].a == nd || B->e[i].b == nd)) fr = 1;
        if (!fr) B->fix[nd * 3 + 2] = 1;
    }
    for (int s = 0; s < B->nsup; s++) {
        int nd = B->sup_n[s];
        if (B->sup_t[s] == 0) B->fix[nd * 3] = B->fix[nd * 3 + 1] = 1;
        else if (B->sup_t[s] == 1) B->fix[nd * 3 + 1] = 1;
        else if (B->sup_t[s] == 2) B->fix[nd * 3] = B->fix[nd * 3 + 1] = B->fix[nd * 3 + 2] = 1;
        else B->fix[nd * 3] = 1;
    }
    int n = B->n;
    B->M = calloc((size_t)n, sizeof(double));
    B->fd = calloc((size_t)n, sizeof(double));
    B->ud = calloc((size_t)n, sizeof(double));
    B->Nd = calloc((size_t)B->nb, sizeof(double));
    const double g = 9.81;
    for (int i = 0; i < B->nb; i++) {
        Bar *e = &B->e[i];
        double m = e->rho * e->A * e->L * 1.15;    /* +15% for connections, bracing */
        B->M[e->a * 3] += m / 2; B->M[e->a * 3 + 1] += m / 2;
        B->M[e->b * 3] += m / 2; B->M[e->b * 3 + 1] += m / 2;
        B->fd[e->a * 3 + 1] -= m / 2 * g; B->fd[e->b * 3 + 1] -= m / 2 * g;
    }
    for (int k = 0; k < B->ndeck; k++) {
        Bar *e = &B->e[B->deck[k]];
        double w = B->q_dead, L = e->L, m = w / g * L;
        B->M[e->a * 3] += m / 2; B->M[e->a * 3 + 1] += m / 2;
        B->M[e->b * 3] += m / 2; B->M[e->b * 3 + 1] += m / 2;
        B->fd[e->a * 3 + 1] -= w * L / 2; B->fd[e->b * 3 + 1] -= w * L / 2;
        B->fd[e->a * 3 + 2] -= w * L * L / 12; B->fd[e->b * 3 + 2] += w * L * L / 12;
    }
    for (int i = 0; i < B->nb; i++) {
        Bar *e = &B->e[i];
        if (!e->pre) continue;
        double ex = (B->px[e->b] - B->px[e->a]) / e->L, ey = (B->py[e->b] - B->py[e->a]) / e->L;
        B->fd[e->a * 3] += e->pre * ex; B->fd[e->a * 3 + 1] += e->pre * ey;
        B->fd[e->b * 3] -= e->pre * ex; B->fd[e->b * 3 + 1] -= e->pre * ey;
    }
    /* dead load, tension-only bars iterated slack until consistent */
    Pcg *s = &B->pcg;
    pcg_init(s, &B->K, B->fix);
    s->tol = 1e-10;
    for (int it = 0; it < 12; it++) {
        br_assemble(B);
        pcg_solve(s, B->fd);
        int changed = 0;
        for (int i = 0; i < B->nb; i++) {
            if (!B->e[i].tonly) continue;
            double N = bar_axial(B, i, s->x, NULL);
            int sl = N < 0;
            if (sl != B->slack[i]) { B->slack[i] = (unsigned char)sl; changed = 1; }
        }
        if (!changed) break;
    }
    memcpy(B->ud, s->x, sizeof(double) * n);
    for (int i = 0; i < B->nb; i++) B->Nd[i] = B->slack[i] ? 0 : bar_axial(B, i, B->ud, NULL);
    /* the fundamental frequency by inverse iteration, K x = M y */
    double *y = calloc((size_t)n, sizeof(double)), *x = calloc((size_t)n, sizeof(double)), lam = 0;
    for (int i = 0; i < n; i++) x[i] = B->fix[i] ? 0 : (i % 3 == 1 ? 1 : 0.1);
    memset(s->x, 0, sizeof(double) * n);
    for (int it = 0; it < 40; it++) {
        for (int i = 0; i < n; i++) y[i] = B->M[i] * x[i];
        pcg_solve(s, y);
        double xm = 0, xk = 0, nr = 0;
        for (int i = 0; i < n; i++) { xm += s->x[i] * B->M[i] * s->x[i]; xk += s->x[i] * y[i]; nr += s->x[i] * s->x[i]; }
        lam = xk / xm;
        nr = sqrt(nr);
        for (int i = 0; i < n; i++) x[i] = s->x[i] / nr;
    }
    free(y); free(x);
    double w1 = sqrt(lam), w2 = 5 * w1;
    B->f1 = w1 / (2 * PI);
    B->zeta = 0.02;
    B->alpha = 2 * B->zeta * w1 * w2 / (w1 + w2);
    B->betar = 2 * B->zeta / (w1 + w2);
    B->dt = BR_DT;
    /* trains: alternate directions, a pause between */
    double L = B->span, t = 3.0;
    B->ntrain = 0;
    for (int k = 0; k < MAXTRAIN && t < 88; k++) {
        Train *T = &B->tr[B->ntrain];
        memset(T, 0, sizeof *T);
        uint32_t hs = hash3(seed, (uint32_t)kind, (uint32_t)k);
        int freight = hs & 1, ncar = 3 + (int)((hs >> 4) % 4);
        train_axles(T, 1 + (ncar > 4), ncar, 220e3, freight ? 200e3 : 130e3);
        double v = (freight ? 9 : 13) + 6 * hashf(seed, (uint32_t)kind, 50u + (uint32_t)k);
        int dir = k % 2 == 0 ? 1 : -1;
        T->v = dir * v;
        T->t0 = t;
        double approach = 0.75 * L;   /* starts this far beyond the bridge's end, nose first */
        T->s0 = dir > 0 ? B->x0 - approach : B->x1 + approach;
        /* the next one sets off as this one leaves the bridge, plus a breath */
        t += (approach + L + T->len) / v + 4 + 4 * hashf(seed, (uint32_t)kind, 90u + (uint32_t)k);
        B->ntrain++;
    }
    /* the exaggeration and colour scale: a static solve, a locomotive at the monitored point */
    {
        double *f = calloc((size_t)n, sizeof(double));
        Train *T = &B->tr[0];
        for (int i = 0; i < 4; i++) br_point_load(B, f, B->mon_x + T->ax[i] - T->ax[1] - 1.3, T->P[i]);
        for (int i = 4; i < 8 && i < T->naxle; i++) br_point_load(B, f, B->mon_x + T->ax[i] - T->ax[1] - 1.3, T->P[i]);
        memset(s->x, 0, sizeof(double) * n);
        pcg_solve(s, f);
        double vm = 1e-9, sm = 1;
        for (int k = 0; k < B->ndeck; k++) vm = fmax(vm, fabs(s->x[B->e[B->deck[k]].a * 3 + 1]));
        for (int i = 0; i < B->nb; i++) sm = fmax(sm, fabs((B->slack[i] ? 0 : bar_axial(B, i, B->ud, s->x)) / B->e[i].A));
        B->vmax_live = vm;
        B->scale = nice_num(0.045 * L / vm, 1);
        B->smax = nice_up(sm * 1.05);
        free(f);
    }
    B->u = calloc((size_t)n, sizeof(double));
    B->v = calloc((size_t)n, sizeof(double));
    B->a = calloc((size_t)n, sizeof(double));
    B->u0 = calloc((size_t)n, sizeof(double));
    B->f = calloc((size_t)n, sizeof(double));
    B->rhs = calloc((size_t)n, sizeof(double));
    B->w = calloc((size_t)n, sizeof(double));
    B->t1 = calloc((size_t)n, sizeof(double));
    memset(s->x, 0, sizeof(double) * n);
    s->tol = 1e-9;
}

static void br_free(Br *B) {
    free(B->px); free(B->py); free(B->e); free(B->deck);
    csr_free(&B->K); free(B->map); free(B->edof); free(B->fix); free(B->slack);
    free(B->M); free(B->ud); free(B->fd); free(B->Nd);
    free(B->u); free(B->v); free(B->a); free(B->u0); free(B->f); free(B->rhs); free(B->w); free(B->t1);
    pcg_free(&B->pcg);
    memset(B, 0, sizeof *B);
}

/* one Newmark step of the live response (trains only; the dead load is
 * the equilibrium it's measured from) */
static void br_step(Br *B) {
    int n = B->n;
    const double be = 0.25, ga = 0.5, dt = B->dt;
    double a0 = 1 / (be * dt * dt), a1 = ga / (be * dt), a2 = 1 / (be * dt), a3 = 1 / (2 * be) - 1;
    double a4 = ga / be - 1, a5 = dt / 2 * (ga / be - 2);
    double tn = (B->step + 1) * dt;
    br_loads(B, tn, B->f);
    for (int i = 0; i < n; i++) B->w[i] = a1 * B->u[i] + a4 * B->v[i] + a5 * B->a[i];
    csr_mul(&B->K, B->w, B->t1);
    for (int i = 0; i < n; i++)
        B->rhs[i] = B->f[i] + B->M[i] * ((a0 + B->alpha * a1) * B->u[i] + (a2 + B->alpha * a4) * B->v[i] + (a3 + B->alpha * a5) * B->a[i]) +
                    B->betar * B->t1[i];
    Pcg *s = &B->pcg;
    s->cK = 1 + a1 * B->betar;
    s->cM = a0 + a1 * B->alpha;
    s->M = B->M;
    /* warm start: a constant-acceleration guess */
    for (int i = 0; i < n; i++) s->x[i] = B->u[i] + dt * B->v[i] + 0.5 * dt * dt * B->a[i];
    pcg_solve(s, B->rhs);
    B->cg_last = s->it;
    B->cg_sum += s->it;
    memcpy(B->u0, B->u, sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double un = s->x[i], an = a0 * (un - B->u[i]) - a2 * B->v[i] - a3 * B->a[i];
        B->v[i] += dt * ((1 - ga) * B->a[i] + ga * an);
        B->a[i] = an;
        B->u[i] = un;
    }
    B->step++;
    /* tension-only bars: slack when the total force would be compressive */
    int changed = 0;
    for (int i = 0; i < B->nb; i++) {
        if (!B->e[i].tonly) continue;
        double N = bar_axial(B, i, B->ud, B->u);
        int sl = N < 0;
        if (sl != B->slack[i]) { B->slack[i] = (unsigned char)sl; changed = 1; }
    }
    if (changed) br_assemble(B);
    double vm = br_deck_v(B, B->u, B->mon_x);
    if (B->nhist < 12000) B->hist[B->nhist++] = (float)vm;
    B->env_max = fmax(B->env_max, vm);
    B->env_min = fmin(B->env_min, vm);
}

static void br_advance(Br *B, double t) {
    while ((B->step + 1) * B->dt <= t) br_step(B);
}

/* axial force in bar i now (dead + live), interpolated between steps */
static double br_N(const Br *B, int i, double fr) {
    if (B->slack[i]) return 0;
    double N1 = bar_axial(B, i, B->ud, B->u0), N2 = bar_axial(B, i, B->ud, B->u);
    return N1 + (N2 - N1) * fr;
}

/* ======================================================================
 * --check: the verification cases
 * ====================================================================== */

/* a pin-jointed Pratt truss, 4 panels of 4 m, 3 m deep, 100 kN at each
 * inner bottom joint; forces against the method of sections */
static void check_truss(void) {
    double X[10], Y[10];
    int nn = 0, bars[20][2], nb = 0;
    for (int i = 0; i <= 4; i++) { X[nn] = 4 * i; Y[nn] = 0; nn++; }          /* 0..4 bottom */
    for (int i = 1; i <= 3; i++) { X[nn] = 4 * i; Y[nn] = 3; nn++; }          /* 5..7 top, above 1..3 */
#define BAR(a, b) (bars[nb][0] = (a), bars[nb][1] = (b), nb++)
    for (int i = 0; i < 4; i++) BAR(i, i + 1);           /* bottom chord 0-3 */
    BAR(5, 6); BAR(6, 7);                                 /* top chord 4-5 */
    BAR(0, 5); BAR(7, 4);                                 /* end posts 6-7 */
    BAR(1, 5); BAR(2, 6); BAR(3, 7);                      /* verticals 8-10 */
    BAR(5, 2); BAR(7, 2);                                 /* diagonals 11-12 */
#undef BAR
    int n = nn * 3, *dofs = malloc(sizeof(int) * 6 * nb), *map;
    for (int e = 0; e < nb; e++)
        for (int j = 0; j < 3; j++) { dofs[e * 6 + j] = bars[e][0] * 3 + j; dofs[e * 6 + 3 + j] = bars[e][1] * 3 + j; }
    Csr K;
    csr_pattern(&K, n, nb, 6, dofs, &map);
    double E = 200e9, A = 0.01;
    for (int e = 0; e < nb; e++) {
        double k[36];
        bar_k(X[bars[e][0]], Y[bars[e][0]], X[bars[e][1]], Y[bars[e][1]], E, A, 0, 1, k);
        for (int j = 0; j < 36; j++) if (map[e * 36 + j] >= 0) K.v[map[e * 36 + j]] += k[j];
    }
    csr_diag(&K);
    unsigned char *fix = calloc((size_t)n, 1);
    for (int i = 0; i < nn; i++) fix[i * 3 + 2] = 1;
    fix[0] = fix[1] = 1; fix[4 * 3 + 1] = 1;
    double *f = calloc((size_t)n, sizeof(double));
    for (int i = 1; i <= 3; i++) f[i * 3 + 1] = -100e3;
    Pcg s;
    pcg_init(&s, &K, fix);
    s.tol = 1e-12;
    pcg_solve(&s, f);
    double N[20];
    for (int e = 0; e < nb; e++) {
        int a = bars[e][0], b = bars[e][1];
        double dx = X[b] - X[a], dy = Y[b] - Y[a], L = sqrt(dx * dx + dy * dy);
        N[e] = E * A / L * ((s.x[b * 3] - s.x[a * 3]) * dx + (s.x[b * 3 + 1] - s.x[a * 3 + 1]) * dy) / L;
    }
    struct { const char *what; int bar; double hand; } c[] = {
        {"bottom chord, end panel   R*4/3", 0, 200e3},
        {"bottom chord, panel 2     R*4/3", 1, 200e3},
        {"top chord, mid            -(R*8-P*4)/3", 4, -266.667e3},
        {"end post                  -R/sin", 6, -250e3},
        {"diagonal, panel 2         (R-P)/sin", 11, 83.333e3},
        {"vertical at quarter       P", 8, 100e3},
        {"vertical at midspan       0", 9, 0},
    };
    printf("truss (pin-jointed Pratt, 4 x 4 m, 3 m deep, 3 x 100 kN), %d CG iterations:\n", s.it);
    for (size_t i = 0; i < sizeof c / sizeof *c; i++)
        printf("  %-40s FE %9.2f kN   hand %9.2f kN\n", c[i].what, N[c[i].bar] / 1e3, c[i].hand / 1e3);
    pcg_free(&s);
    csr_free(&K);
    free(map); free(dofs); free(fix); free(f);
}

static void check_top(int kind, int nelx, int nely, double vf, double rmin) {
    Top T;
    top_build(&T, kind, nelx, nely, vf, rmin);
    double t0 = now();
    top_work(&T, 0);
    double ms = (now() - t0) * 1000;
    int cg = 0;
    for (int i = 1; i < T.nhist; i++) cg += T.hcg[i];
    printf("SIMP %s %dx%d, volfrac %.2f, penal 3, rmin %.1f: %d iterations, c = %.2f, change %.4f, vol %.3f\n",
           kind == T_MBB ? "MBB" : "bridge", nelx, nely, vf, rmin, T.iter, T.c, T.change, T.vol);
    printf("  %.0f ms in all, %.1f ms an iteration, %.0f CG iterations a solve on average\n", ms, ms / T.iter, (double)cg / T.iter);
    for (int ey = 0; ey < nely; ey += nely > 30 ? 2 : 1) {
        printf("  ");
        for (int ex = 0; ex < nelx; ex += nelx > 100 ? 2 : 1) {
            double x = T.xPhys[ex * nely + ey];
            putchar(x > 0.75 ? '#' : x > 0.5 ? '+' : x > 0.25 ? '.' : ' ');
        }
        putchar('\n');
    }
    top_free(&T);
}

static void run_check(void) {
    printf("bridge --check\n\n");
    check_truss();
    printf("\n");
    Cont c;
    cont_build(&c, C_CANTI);
    printf("cantilever 2000 x 250 x 20 mm, QM6 64x8, tip shear P (parabolic), %d CG iterations (%.0f ms):\n", c.iters, c.solve_ms);
    printf("  tip deflection FE %.4f mm/kN   PL^3/3EI %.4f mm/kN (%.2f%%)   + Timoshenko shear %.4f mm/kN (%.2f%%)\n",
           c.check_fe * 1e3, c.check_ref * 1e3, 100 * (c.check_fe / c.check_ref - 1), c.check_ref2 * 1e3,
           100 * (c.check_fe / c.check_ref2 - 1));
    {
        int n = nearest_node(&c, 0, c.h);
        double I = c.t * pow(c.h, 3) / 12;
        printf("  root top-fibre stress per kN: FE %.3f MPa, Mc/I %.3f MPa (nodal average at a clamped corner)\n",
               c.sn[0][n * 3] * 1e3, 1e3 * c.len * c.h / 2 / I);
        int m = nearest_node(&c, c.len / 2, c.h);
        printf("  midspan top-fibre stress per kN: FE %.4f MPa, Mc/I %.4f MPa\n", c.sn[0][m * 3] * 1e3, 1e3 * c.len / 2 * c.h / 2 / I);
        printf("  %d principal stress trajectories\n", c.ntraj);
    }
    cont_free(&c);
    cont_build(&c, C_PLATE);
    printf("\nplate with a hole, W/D = 5, %d QM6 elements, %d dofs, %d CG iterations (%.0f ms):\n", c.ne, 2 * c.nn, c.iters, c.solve_ms);
    printf("  Kt = sigma_max / sigma_inf: FE %.3f   Kirsch (infinite plate) %.3f   Heywood (W/D = 5) %.3f\n",
           c.check_fe, c.check_ref, c.check_ref2);
    {
        int n = nearest_node(&c, c.a, 0);
        printf("  sigma_yy at the side of the hole (theta = 0): FE %.3f, Kirsch -1\n", c.sn[0][n * 3 + 1]);
        printf("  along the ligament, sigma_xx / sigma_inf (FE vs Kirsch 1 + a^2/2y^2 + 3a^4/2y^4):\n   ");
        for (double y = 10; y <= 40.1; y += 5) {
            int k = nearest_node(&c, 0, y);
            double yy = c.xy[k * 2 + 1], r = c.a / yy;
            printf(" y=%.1f %.3f/%.3f", yy, c.sn[0][k * 3], 1 + 0.5 * r * r + 1.5 * r * r * r * r);
        }
        printf("\n");
    }
    cont_free(&c);
    cont_build(&c, C_BRACKET);
    printf("\nL-bracket: %d elements, %d dofs, %d CG iterations (%.0f ms)\n", c.ne, 2 * c.nn, c.iters, c.solve_ms);
    cont_free(&c);
    printf("\nbridges:\n");
    for (int k = 0; k < NBRIDGE; k++) {
        Br B;
        double t0 = now();
        br_build(&B, k, 1);
        double tb = now() - t0;
        t0 = now();
        br_advance(&B, 60);
        double ts = now() - t0;
        printf("  %-13s %3d nodes %3d bars %3d dofs  f1 %.2f Hz  x%-5g  smax %3.0f MPa  live max %.1f mm  "
               "built %.0f ms, 60 s of steps %.0f ms (%.1f CG its/step)\n",
               B.name, B.nn, B.nb, B.n, B.f1, B.scale, B.smax / 1e6, B.vmax_live * 1e3, tb * 1e3, ts * 1e3, (double)B.cg_sum / B.step);
        br_free(&B);
    }
    printf("\n");
    check_top(T_MBB, 60, 20, 0.5, 2.4);
    check_top(T_MBB, 120, 40, 0.5, 4.8);
    check_top(T_BRIDGE, 120, 40, 0.3, 3.0);
}

/* ======================================================================
 * drawing: one vertex stream, painter's order, one shader. A vertex is
 * canvas position, colour, (u, v, w) and a kind s:
 *   s in [0,1]  banded sequential colour map (fringe plot)
 *   s in [2,3]  smooth diverging map (tension / compression)
 *   s in [4,5]  smooth sequential map
 *   s = -1      flat colour, alpha cut by the distance |(u,v)| against w
 *               (anti-aliased lines and discs)
 *   s = -2      flat colour, hatched
 *   s = -3      text: (u, v) are atlas texels
 *   s = -4      the optimiser's density field, (u, v) its texture coords
 * ====================================================================== */

typedef struct { float x, y, r, g, b, a, u, v, w, s; } Vtx;
static Vtx *vb;
static int nvb, cvb;
static float sc = 1;                   /* physical pixels per canvas unit */

static void vput(float x, float y, const float *c, float u, float v, float w, float s) {
    if (nvb == cvb) vb = realloc(vb, sizeof(Vtx) * (size_t)(cvb = cvb ? cvb * 2 : 65536));
    Vtx *o = &vb[nvb++];
    o->x = x; o->y = y; o->r = c[0]; o->g = c[1]; o->b = c[2]; o->a = c[3];
    o->u = u; o->v = v; o->w = w; o->s = s;
}

static void tri(float x0, float y0, float x1, float y1, float x2, float y2, const float *c) {
    vput(x0, y0, c, 0, 0, 1e6f, -1); vput(x1, y1, c, 0, 0, 1e6f, -1); vput(x2, y2, c, 0, 0, 1e6f, -1);
}

static void rect(float x0, float y0, float x1, float y1, const float *c) {
    tri(x0, y0, x1, y0, x1, y1, c);
    tri(x0, y0, x1, y1, x0, y1, c);
}

static void rect_hatch(float x0, float y0, float x1, float y1, const float *c) {
    vput(x0, y0, c, 0, 0, 1e6f, -2); vput(x1, y0, c, 0, 0, 1e6f, -2); vput(x1, y1, c, 0, 0, 1e6f, -2);
    vput(x0, y0, c, 0, 0, 1e6f, -2); vput(x1, y1, c, 0, 0, 1e6f, -2); vput(x0, y1, c, 0, 0, 1e6f, -2);
}

/* a line w canvas units wide (thinner than a pixel: a pixel, fainter),
 * with s >= 0 its colour comes from a map at s0 .. s1 */
static void line_s(float x0, float y0, float x1, float y1, float w, const float *c, float s0, float s1) {
    float dx = x1 - x0, dy = y1 - y0, L = sqrtf(dx * dx + dy * dy);
    if (L < 1e-6f) return;
    dx /= L; dy /= L;
    float hw = w * 0.5f, cc[4] = {c[0], c[1], c[2], c[3]};
    float minw = 0.6f / sc;
    if (hw < minw) { cc[3] *= hw / minw; hw = minw; }
    float e = hw + 1.0f / sc;           /* the quad is a pixel wider than the line, for the fade */
    float nx = -dy * e, ny = dx * e, ax = dx * hw * 0.5f, ay = dy * hw * 0.5f;
    float X0 = x0 - ax, Y0 = y0 - ay, X1 = x1 + ax, Y1 = y1 + ay;
    vput(X0 + nx, Y0 + ny, cc, e, 0, hw, s0); vput(X1 + nx, Y1 + ny, cc, e, 0, hw, s1); vput(X1 - nx, Y1 - ny, cc, -e, 0, hw, s1);
    vput(X0 + nx, Y0 + ny, cc, e, 0, hw, s0); vput(X1 - nx, Y1 - ny, cc, -e, 0, hw, s1); vput(X0 - nx, Y0 - ny, cc, -e, 0, hw, s0);
}

static void line(float x0, float y0, float x1, float y1, float w, const float *c) { line_s(x0, y0, x1, y1, w, c, -1, -1); }

static void dashed(float x0, float y0, float x1, float y1, float w, float on, float off, const float *c) {
    float dx = x1 - x0, dy = y1 - y0, L = sqrtf(dx * dx + dy * dy);
    if (L < 1e-6f) return;
    for (float s = 0; s < L; s += on + off) {
        float e = fminf(s + on, L);
        line(x0 + dx * s / L, y0 + dy * s / L, x0 + dx * e / L, y0 + dy * e / L, w, c);
    }
}

static void disc(float x, float y, float r, const float *c) {
    float e = r + 1.0f / sc;
    vput(x - e, y - e, c, -e, -e, r, -1); vput(x + e, y - e, c, e, -e, r, -1); vput(x + e, y + e, c, e, e, r, -1);
    vput(x - e, y - e, c, -e, -e, r, -1); vput(x + e, y + e, c, e, e, r, -1); vput(x - e, y + e, c, -e, e, r, -1);
}

static void ring(float x, float y, float r, float w, const float *c) {
    int n = 12 + (int)(r * sc * 0.6f);
    if (n > 96) n = 96;
    for (int i = 0; i < n; i++) {
        float a0 = 2 * (float)PI * i / n, a1 = 2 * (float)PI * (i + 1) / n;
        line(x + r * cosf(a0), y + r * sinf(a0), x + r * cosf(a1), y + r * sinf(a1), w, c);
    }
}

/* an arrow from (x0, y0) to its point at (x1, y1) */
static void arrow(float x0, float y0, float x1, float y1, float w, float head, const float *c) {
    float dx = x1 - x0, dy = y1 - y0, L = sqrtf(dx * dx + dy * dy);
    if (L < 1e-6f) return;
    dx /= L; dy /= L;
    float hl = fminf(head, L * 0.6f), hw = hl * 0.38f;
    line(x0, y0, x1 - dx * hl * 0.8f, y1 - dy * hl * 0.8f, w, c);
    tri(x1, y1, x1 - dx * hl - dy * hw, y1 - dy * hl + dx * hw, x1 - dx * hl + dy * hw, y1 - dy * hl - dx * hw, c);
}

/* ---- text: a lazily-filled glyph atlas, any codepoint, any size ---- */
enum { F_SANS, F_SEMI, F_MONO, F_MATH, NFACE };
static FT_Library ft;
static FT_Face faces[NFACE];
#define AT_W 2048
#define AT_H 2048
static unsigned char *atlas;
static int at_x = 1, at_y = 1, at_shelf, at_dirty = 1;
typedef struct { uint32_t cp; short x, y, w, h, left, top; float adv; } Glyph;
#define GSLOTS 256
typedef struct { int face, px; Glyph g[GSLOTS]; int used[GSLOTS]; float asc, desc; } Style;
#define MAXSTYLE 96
static Style styles[MAXSTYLE];
static int nstyles;

static void fonts_init(void) {
    static const char *pat[NFACE] = {
        "Inter:weight=regular", "Inter:weight=semibold", "JetBrainsMono Nerd Font Mono:weight=regular", "Latin Modern Math",
    };
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        FcPattern *p = FcNameParse((const FcChar8 *)pat[i]);
        FcConfigSubstitute(NULL, p, FcMatchPattern);
        FcDefaultSubstitute(p);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, p, &res);
        FcChar8 *file = NULL;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        if (FT_New_Face(ft, (const char *)file, 0, &faces[i])) die("can't load a font");
        FcPatternDestroy(m);
        FcPatternDestroy(p);
    }
    atlas = calloc((size_t)AT_W * AT_H, 1);
}

static void atlas_reset(void) {
    memset(atlas, 0, (size_t)AT_W * AT_H);
    at_x = at_y = 1; at_shelf = 0; at_dirty = 1;
    nstyles = 0;
}

/* a style: a face at a size in physical pixels */
static int style(int face, float px) {
    int p = (int)lroundf(px * sc);
    if (p < 6) p = 6;
    if (p > 400) p = 400;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) atlas_reset();
    Style *s = &styles[nstyles];
    memset(s, 0, sizeof *s);
    s->face = face; s->px = p;
    FT_Set_Pixel_Sizes(faces[face], 0, (FT_UInt)p);
    s->asc = faces[face]->size->metrics.ascender / 64.0f;
    s->desc = -faces[face]->size->metrics.descender / 64.0f;
    return nstyles++;
}

static const Glyph *glyph(int si, uint32_t cp) {
    Style *s = &styles[si];
    uint32_t h = (cp * 2654435761u) % GSLOTS;
    for (int k = 0; k < GSLOTS; k++, h = (h + 1) % GSLOTS) {
        if (s->used[h] && s->g[h].cp == cp) return &s->g[h];
        if (!s->used[h]) break;
    }
    if (s->used[h]) return &s->g[0];
    FT_Face f = faces[s->face];
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)s->px);
    Glyph *g = &s->g[h];
    memset(g, 0, sizeof *g);
    g->cp = cp;
    s->used[h] = 1;
    FT_UInt gi = FT_Get_Char_Index(f, cp);
    if (!gi && s->face != F_MATH) gi = 0;
    if (FT_Load_Glyph(f, gi, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) return g;
    FT_GlyphSlot gs = f->glyph;
    int w = (int)gs->bitmap.width, hh = (int)gs->bitmap.rows;
    if (at_x + w + 2 > AT_W) { at_x = 1; at_y += at_shelf + 2; at_shelf = 0; }
    if (at_y + hh + 2 > AT_H) { atlas_reset(); return glyph(style(s->face, s->px / sc), cp); }
    for (int y = 0; y < hh; y++) memcpy(atlas + (size_t)(at_y + y) * AT_W + at_x, gs->bitmap.buffer + y * gs->bitmap.pitch, (size_t)w);
    g->x = (short)at_x; g->y = (short)at_y; g->w = (short)w; g->h = (short)hh;
    g->left = (short)gs->bitmap_left; g->top = (short)gs->bitmap_top;
    g->adv = gs->advance.x / 64.0f;
    at_x += w + 2;
    if (hh > at_shelf) at_shelf = hh;
    at_dirty = 1;
    return g;
}

static uint32_t utf8(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t c = *s++;
    if (c >= 0xF0) { c = (c & 7) << 18; c |= (*s++ & 63u) << 12; c |= (*s++ & 63u) << 6; c |= *s++ & 63u; }
    else if (c >= 0xE0) { c = (c & 15) << 12; c |= (*s++ & 63u) << 6; c |= *s++ & 63u; }
    else if (c >= 0xC0) { c = (c & 31) << 6; c |= *s++ & 63u; }
    *p = (const char *)s;
    return c;
}

/* one glyph with its pen at (x, y) canvas, baseline; returns the advance (canvas) */
static float put_glyph(int si, uint32_t cp, float x, float y, const float *col) {
    const Glyph *g = glyph(si, cp);
    if (g->w && col) {
        float X = roundf(x * sc) + g->left, Y = roundf(y * sc) - g->top;
        float x0 = X / sc, y0 = Y / sc, x1 = (X + g->w) / sc, y1 = (Y + g->h) / sc;
        float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
        vput(x0, y0, col, u0, v0, 0, -3); vput(x1, y0, col, u1, v0, 0, -3); vput(x1, y1, col, u1, v1, 0, -3);
        vput(x0, y0, col, u0, v0, 0, -3); vput(x1, y1, col, u1, v1, 0, -3); vput(x0, y1, col, u0, v1, 0, -3);
    }
    return g->adv / sc;
}

static float text_w(int face, float px, const char *str, float track) {
    int si = style(face, px);
    float w = 0;
    for (const char *p = str; *p;) {
        w += glyph(si, utf8(&p))->adv / sc;
        if (*p) w += track * px;
    }
    return w;
}

/* str with its baseline at y; align -1 left, 0 centre, 1 right; track in ems */
static float text(int face, float px, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_w(face, px, str, track);
    int si = style(face, px);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    for (const char *p = str; *p;) {
        pen += put_glyph(si, utf8(&p), pen, y, col);
        if (*p) pen += track * px;
    }
    return w;
}

static float cap_h(int face, float px) { return glyph(style(face, px), 'H')->top / sc; }

/* ---- a little mathematical typesetting, TeX-flavoured:
 *   letters are math italic, digits and \r{...} upright, \b{...} bold
 *   x^{..} x_{..}  superscripts and subscripts (a single character needs no braces)
 *   \f{num}{den}   a fraction       \q{...}  a square root
 *   \d{x} \dd{x}   dot, double dot  \, \;    thin, medium space
 *   = + - and the like are spaced as relations and operators */
static uint32_t math_cp(uint32_t c, int mode) {
    if (mode == 1) return c == '-' ? 0x2212 : c;
    if (mode == 2) {
        if (c >= 'A' && c <= 'Z') return 0x1D400 + (c - 'A');
        if (c >= 'a' && c <= 'z') return 0x1D41A + (c - 'a');
        if (c >= 0x3B1 && c <= 0x3C9) return 0x1D6C2 + (c - 0x3B1);
        return c;
    }
    if (c == 'h') return 0x210E;
    if (c >= 'A' && c <= 'Z') return 0x1D434 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 0x1D44E + (c - 'a');
    if (c >= 0x3B1 && c <= 0x3C9) return 0x1D6FC + (c - 0x3B1);
    if (c == '-') return 0x2212;
    return c;
}

static int is_rel(uint32_t c) { return c == '=' || c == 0x2248 || c == 0x2264 || c == 0x2265 || c == 0x2192 || c == 0x2190 || c == '<' || c == '>' || c == 0x221D; }
static int is_bin(uint32_t c) { return c == '+' || c == '-' || c == 0x2212 || c == 0x00D7 || c == 0x22C5 || c == 0x00B1; }

typedef struct { float asc, desc; } MBox;

static float mrun(const char **pp, float px, int mode, int depth, float x, float y, const float *col, MBox *bx);

/* one atom: a group in braces, or a single character */
static float matom(const char **pp, float px, int mode, int depth, float x, float y, const float *col, MBox *bx) {
    if (**pp == '{') {
        (*pp)++;
        float w = mrun(pp, px, mode, depth, x, y, col, bx);
        if (**pp == '}') (*pp)++;
        return w;
    }
    const char *s = *pp;
    uint32_t c = utf8(&s);
    *pp = s;
    int si = style(F_MATH, px);
    const Glyph *g = glyph(si, math_cp(c, mode));
    bx->asc = fmaxf(bx->asc, g->top / sc);
    bx->desc = fmaxf(bx->desc, (g->h - g->top) / sc);
    return put_glyph(si, math_cp(c, mode), x, y, col);
}

static float mrun(const char **pp, float px, int mode, int depth, float x, float y, const float *col, MBox *bx) {
    const char *p = *pp;
    float x0 = x, last_x = x, last_w = 0, last_top = 0;
    int si = style(F_MATH, px);
    while (*p && *p != '}') {
        if (*p == '^' || *p == '_') {
            /* a script, or a sub and a sup on the same atom */
            int sup = *p == '^';
            p++;
            float spx = px * (depth ? 0.78f : 0.7f);
            MBox b = {0, 0};
            float sx = last_x + last_w;
            float dy = sup ? -fmaxf(px * 0.42f, last_top - spx * 0.3f) : px * 0.24f;
            float w = matom(&p, spx, mode, depth + 1, sx, y + dy, col, &b);
            bx->asc = fmaxf(bx->asc, b.asc - dy);
            bx->desc = fmaxf(bx->desc, b.desc + dy);
            if ((*p == '^' || *p == '_') && (*p == '^') != sup) {
                int sup2 = *p == '^';
                p++;
                MBox b2 = {0, 0};
                float dy2 = sup2 ? -fmaxf(px * 0.42f, last_top - spx * 0.3f) : px * 0.24f;
                float w2 = matom(&p, spx, mode, depth + 1, sx, y + dy2, col, &b2);
                w = fmaxf(w, w2);
                bx->asc = fmaxf(bx->asc, b2.asc - dy2);
                bx->desc = fmaxf(bx->desc, b2.desc + dy2);
            }
            x = sx + w + px * 0.04f;
            last_w = 0;
            continue;
        }
        if (*p == '\\') {
            p++;
            if (*p == ',') { p++; x += px * 0.17f; continue; }
            if (*p == ';') { p++; x += px * 0.28f; continue; }
            if (*p == '!') { p++; x -= px * 0.12f; continue; }
            if (*p == 'r' || *p == 'b') {
                int m = *p == 'r' ? 1 : 2;
                p++;
                MBox b = {0, 0};
                last_x = x;
                last_w = matom(&p, px, m, depth, x, y, col, &b);
                last_top = b.asc;
                bx->asc = fmaxf(bx->asc, b.asc); bx->desc = fmaxf(bx->desc, b.desc);
                x += last_w;
                continue;
            }
            if (*p == 'f') {
                p++;
                float fpx = depth ? px * 0.78f : px * 0.86f;
                const char *q = p;
                MBox bn = {0, 0}, bd = {0, 0};
                float wn = matom(&q, fpx, mode, depth + 1, 0, 0, NULL, &bn);
                float wd = matom(&q, fpx, mode, depth + 1, 0, 0, NULL, &bd);
                float w = fmaxf(wn, wd) + px * 0.2f, axis = px * 0.26f, gap = px * 0.14f;
                float th = fmaxf(px * 0.045f, 1.0f / sc);
                matom(&p, fpx, mode, depth + 1, x + (w - wn) * 0.5f, y - axis - gap - bn.desc, col, &bn);
                matom(&p, fpx, mode, depth + 1, x + (w - wd) * 0.5f, y - axis + gap + bd.asc, col, &bd);
                if (col) rect(x + px * 0.05f, y - axis - th * 0.5f, x + w - px * 0.05f, y - axis + th * 0.5f, col);
                bx->asc = fmaxf(bx->asc, axis + gap + bn.desc + bn.asc);
                bx->desc = fmaxf(bx->desc, gap + bd.asc + bd.desc - axis);
                last_x = x; last_w = w; last_top = axis + gap + bn.desc + bn.asc;
                x += w + px * 0.06f;
                continue;
            }
            if (*p == 'q') {
                p++;
                const char *q = p;
                MBox b = {0, 0};
                float w = matom(&q, px, mode, depth, 0, 0, NULL, &b);
                /* the radical, grown to span what's under it, its top on the vinculum */
                const Glyph *rg = glyph(si, 0x221A);
                float need = b.asc + px * 0.14f, span_h = need + b.desc + px * 0.04f;
                float f = fmaxf(1, fminf(2.5f, span_h / fmaxf(rg->h / sc, 1e-3f)));
                int rs = style(F_MATH, px * f);
                const Glyph *rg2 = glyph(rs, 0x221A);
                float top = need, th = fmaxf(px * 0.045f, 1.0f / sc);
                float rw = put_glyph(rs, 0x221A, x, y - need + rg2->top / sc, col);
                matom(&p, px, mode, depth, x + rw + px * 0.04f, y, col, &b);
                if (col) rect(x + rw - th * 0.5f, y - top, x + rw + w + px * 0.1f, y - top + th, col);
                rw += px * 0.04f;
                bx->asc = fmaxf(bx->asc, top); bx->desc = fmaxf(bx->desc, b.desc);
                last_x = x; last_w = rw + w + px * 0.08f; last_top = top;
                x += last_w;
                continue;
            }
            if (*p == 'd') {
                int two = p[1] == 'd';
                p += two ? 2 : 1;
                MBox b = {0, 0};
                float w = matom(&p, px, mode, depth, x, y, col, &b);
                const Glyph *dg = glyph(si, two ? 0xA8 : 0x2D9);
                float dw = dg->adv / sc;
                put_glyph(si, two ? 0xA8 : 0x2D9, x + (w - dw) * 0.5f + px * 0.04f, y - b.asc - px * 0.06f + (dg->top - dg->h) / sc, col);
                bx->asc = fmaxf(bx->asc, b.asc + px * 0.12f); bx->desc = fmaxf(bx->desc, b.desc);
                last_x = x; last_w = w; last_top = b.asc + px * 0.12f;
                x += w;
                continue;
            }
            continue;
        }
        if (*p == '{') {
            MBox b = {0, 0};
            last_x = x;
            last_w = matom(&p, px, mode, depth, x, y, col, &b);
            last_top = b.asc;
            bx->asc = fmaxf(bx->asc, b.asc); bx->desc = fmaxf(bx->desc, b.desc);
            x += last_w;
            continue;
        }
        if (*p == ' ') { p++; x += px * 0.25f; continue; }
        const char *q = p;
        uint32_t c = utf8(&q);
        float sp = depth ? 0 : is_rel(c) ? px * 0.26f : is_bin(c) ? px * 0.2f : 0;
        x += sp;
        MBox b = {0, 0};
        last_x = x;
        last_w = matom(&p, px, mode, depth, x, y, col, &b);
        last_top = b.asc;
        bx->asc = fmaxf(bx->asc, b.asc); bx->desc = fmaxf(bx->desc, b.desc);
        x += last_w + sp;
    }
    *pp = p;
    return x - x0;
}

/* typeset src with its baseline at (x, y); returns the width */
static float math(const char *src, float px, float x, float y, const float *col, MBox *out) {
    const char *p = src;
    MBox b = {0, 0};
    float w = mrun(&p, px, 1 * 0, 0, x, y, col, &b);
    if (out) *out = b;
    return w;
}

static float math_w(const char *src, float px, MBox *out) { return math(src, px, 0, 0, NULL, out); }

/* ======================================================================
 * palettes and colour maps
 * ====================================================================== */

typedef struct {
    const char *name;
    float bg0[3], bg1[3], grid[4], ink[4], dim[4], faint[4], accent[4], panel[4], solid[4], warn[4];
    int seq;                    /* 0 turbo, 1 viridis, 2 blueprint, 3 grey */
    float neg[3], mid[3], pos[3];   /* diverging: compression, zero, tension */
    int light;
} Pal;

#define C4(r, g, b, a) {r / 255.0f, g / 255.0f, b / 255.0f, a}
#define C3(r, g, b) {r / 255.0f, g / 255.0f, b / 255.0f}
static const Pal pals[] = {
    {"cae", C3(13, 17, 23), C3(24, 31, 42), C4(130, 160, 200, 0.07f), C4(214, 224, 234, 1), C4(130, 146, 166, 1),
     C4(130, 146, 166, 0.35f), C4(255, 180, 70, 1), C4(12, 16, 22, 0.90f), C4(205, 214, 226, 1), C4(255, 96, 80, 1), 0,
     C3(214, 47, 39), C3(200, 204, 212), C3(46, 110, 230), 0},
    {"blueprint", C3(10, 52, 102), C3(16, 72, 132), C4(220, 235, 255, 0.10f), C4(238, 245, 255, 1), C4(160, 196, 232, 1),
     C4(190, 215, 245, 0.35f), C4(255, 214, 102, 1), C4(8, 42, 84, 0.90f), C4(232, 242, 255, 1), C4(255, 130, 90, 1), 2,
     C3(255, 150, 60), C3(225, 236, 250), C3(70, 225, 255), 0},
    {"paper", C3(243, 239, 228), C3(235, 230, 216), C4(60, 120, 110, 0.10f), C4(31, 42, 54, 1), C4(93, 107, 120, 1),
     C4(93, 107, 120, 0.35f), C4(200, 70, 30, 1), C4(250, 248, 241, 0.94f), C4(40, 50, 62, 1), C4(200, 40, 30, 1), 1,
     C3(200, 30, 40), C3(170, 170, 170), C3(30, 80, 200), 1},
    {"mono", C3(14, 14, 14), C3(24, 24, 24), C4(255, 255, 255, 0.05f), C4(232, 232, 232, 1), C4(140, 140, 140, 1),
     C4(140, 140, 140, 0.35f), C4(255, 255, 255, 1), C4(10, 10, 10, 0.90f), C4(225, 225, 225, 1), C4(255, 255, 255, 1), 3,
     C3(70, 70, 70), C3(150, 150, 150), C3(250, 250, 250), 0},
};
#define NPAL (int)(sizeof pals / sizeof *pals)
static const Pal *P = &pals[0];

static void seq_rgb(int map, float x, float *o) {
    x = x < 0 ? 0 : x > 1 ? 1 : x;
    if (map == 0) {
        /* turbo, Mikhailov's polynomial fit */
        o[0] = 0.13572138f + x * (4.61539260f + x * (-42.66032258f + x * (132.13108234f + x * (-152.94239396f + x * 59.28637943f))));
        o[1] = 0.09140261f + x * (2.19418839f + x * (4.84296658f + x * (-14.18503333f + x * (4.27729857f + x * 2.82956604f))));
        o[2] = 0.10667330f + x * (12.64194608f + x * (-60.58204836f + x * (110.36276771f + x * (-89.90310912f + x * 27.34824973f))));
    } else if (map == 1) {
        /* viridis, polynomial fit */
        static const float c[7][3] = {{0.2777273f, 0.0054073f, 0.3340998f}, {0.1050930f, 1.4046135f, 1.3845902f},
                                      {-0.3308618f, 0.2148476f, 0.0950952f}, {-4.6342305f, -5.7991010f, -19.3324410f},
                                      {6.2282699f, 14.1799334f, 56.6905526f}, {4.7763850f, -13.7451454f, -65.3530326f},
                                      {-5.4354559f, 4.6458526f, 26.3124352f}};
        for (int i = 0; i < 3; i++) {
            float v = c[6][i];
            for (int k = 5; k >= 0; k--) v = c[k][i] + x * v;
            o[i] = v;
        }
    } else if (map == 2) {
        static const float k[7][3] = {{0.11f, 0.20f, 0.52f}, {0.18f, 0.45f, 0.82f}, {0.33f, 0.80f, 1.0f}, {0.80f, 0.96f, 1.0f},
                                      {1.0f, 0.95f, 0.66f}, {1.0f, 0.70f, 0.30f}, {1.0f, 0.40f, 0.22f}};
        float f = x * 6;
        int i = (int)f;
        if (i > 5) i = 5;
        f -= i;
        for (int j = 0; j < 3; j++) o[j] = k[i][j] + (k[i + 1][j] - k[i][j]) * f;
    } else {
        float v = 0.16f + 0.80f * x;
        o[0] = o[1] = o[2] = v;
    }
    for (int i = 0; i < 3; i++) o[i] = o[i] < 0 ? 0 : o[i] > 1 ? 1 : o[i];
}

static void div_rgb(float x, float *o) {
    x = x < 0 ? 0 : x > 1 ? 1 : x;
    const float *a = x < 0.5f ? P->neg : P->mid, *b = x < 0.5f ? P->mid : P->pos;
    float f = x < 0.5f ? x * 2 : x * 2 - 1;
    f = f * f * (3 - 2 * f) * 0.35f + f * 0.65f;
    for (int i = 0; i < 3; i++) o[i] = a[i] + (b[i] - a[i]) * f;
}

/* ======================================================================
 * layout: the canvas is the window, or spanning, the whole wall; every
 * position below is in canvas units
 * ====================================================================== */

typedef struct { float x, y, w, h, u; int on; } Box;
enum { BX_TITLE, BX_EQN, BX_PLOT, BX_READ, BX_LEGEND, BX_BLOCK, NBOX };
static Box box[NBOX];
static float view[4];                  /* this instance's rect of the canvas */
static float stage[4];                 /* where the model goes */
static float U = 1;                    /* the model's annotation unit */
static float canvas_w, canvas_h;
static int portrait_layout;

static Box mkbox(float x, float y, float w, float h, float u) { Box b = {x, y, w, h, u, 1}; return b; }

static void layout(void) {
    for (int i = 0; i < NBOX; i++) box[i].on = 0;
    if (span_me < 0) {
        view[0] = view[1] = 0; view[2] = (float)W; view[3] = (float)H;
        canvas_w = (float)W; canvas_h = (float)H;
        sc = 1;
        float u = fminf((float)W, (float)H) / 1080.0f, m = roundf(30 * u);
        U = u;
        if (W >= H) {
            portrait_layout = 0;
            float hh = roundf(0.165f * H), fh = roundf(0.25f * H), g = roundf(14 * u);
            box[BX_TITLE] = mkbox(m, m, 0.30f * W, hh, u);
            box[BX_EQN] = mkbox(W - m - 0.56f * W, m, 0.56f * W, hh, u);
            float fy = H - m - fh, x = m, aw = W - 2 * m - 3 * g;
            box[BX_PLOT] = mkbox(x, fy, 0.33f * aw, fh, u); x += 0.33f * aw + g;
            box[BX_READ] = mkbox(x, fy, 0.24f * aw, fh, u); x += 0.24f * aw + g;
            box[BX_LEGEND] = mkbox(x, fy, 0.15f * aw, fh, u); x += 0.15f * aw + g;
            box[BX_BLOCK] = mkbox(x, fy, W - m - x, fh, u);
            stage[0] = m; stage[1] = m + hh + g; stage[2] = W - 2 * m; stage[3] = fy - g - stage[1];
        } else {
            portrait_layout = 1;
            float g = roundf(14 * u), y = m, aw = W - 2 * m;
            box[BX_TITLE] = mkbox(m, y, aw, 0.075f * H, u); y += 0.075f * H + g;
            box[BX_EQN] = mkbox(m, y, aw, 0.15f * H, u); y += 0.15f * H + g;
            float sh = 0.30f * H;
            stage[0] = m; stage[1] = y; stage[2] = aw; stage[3] = sh; y += sh + g;
            box[BX_PLOT] = mkbox(m, y, aw, 0.15f * H, u); y += 0.15f * H + g;
            box[BX_READ] = mkbox(m, y, aw * 0.6f - g / 2, 0.135f * H, u);
            box[BX_LEGEND] = mkbox(m + aw * 0.6f + g / 2, y, aw * 0.4f - g / 2, 0.135f * H, u);
            y += 0.135f * H + g;
            box[BX_BLOCK] = mkbox(m, y, aw, H - m - y, u);
        }
        return;
    }
    /* spanning */
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    front = span_me == mid;
    /* the canvas origin at the wall's top left */
    for (int i = 0; i < 4; i++) view[i] = span[span_me][i];
    view[0] -= x0; view[1] -= y0;
    canvas_w = x1 - x0; canvas_h = y1 - y0;
    sc = (float)W / view[2];
    const float *F = span[mid];
    float fx = F[0] - x0, fy = F[1] - y0, fw = F[2], fhh = F[3];
    /* the band every output shares */
    float b0 = -1e9f, b1 = 1e9f;
    for (int i = 0; i < span_n; i++) { b0 = fmaxf(b0, span[i][1] - y0); b1 = fminf(b1, span[i][1] + span[i][3] - y0); }
    if (b1 - b0 < 0.4f * fhh) { b0 = fy; b1 = fy + fhh; }
    float u = fminf(fw, fhh) / 1080.0f, m = roundf(30 * u), g = roundf(14 * u);
    U = u;
    portrait_layout = 0;
    /* a portrait output with room above and below the band */
    int por = -1;
    float best = 1.2f;
    for (int i = 0; i < span_n; i++) {
        float a = span[i][3] / span[i][2];
        if (i != mid && a > best) { best = a; por = i; }
    }
    float footer = roundf(0.21f * (b1 - b0));
    if (por >= 0) {
        const float *Q = span[por];
        float qx = Q[0] - x0, qy = Q[1] - y0, qw = Q[2], qh = Q[3], qu = fminf(qw, qh) / 1080.0f, qm = roundf(30 * qu);
        float above = b0 - qy, below = qy + qh - b1;
        if (above > 0.12f * qh) {
            float th = fminf(0.3f * above, 150 * qu);
            box[BX_TITLE] = mkbox(qx + qm, qy + qm, qw - 2 * qm, th, qu);
            box[BX_EQN] = mkbox(qx + qm, qy + qm + th + g, qw - 2 * qm, above - 2 * qm - th - g, qu);
        }
        if (below > 0.12f * qh) {
            float ph = 0.56f * (below - 2 * qm);
            box[BX_PLOT] = mkbox(qx + qm, b1 + qm, qw - 2 * qm, ph, qu);
            box[BX_READ] = mkbox(qx + qm, b1 + qm + ph + g, qw - 2 * qm, below - 2 * qm - ph - g, qu);
        }
    }
    /* whatever is left goes on the front output: a header row and a footer row */
    float hdr = 0;
    if (!box[BX_TITLE].on) {
        hdr = roundf(0.15f * fhh);
        box[BX_TITLE] = mkbox(fx + m, fy + m, 0.3f * fw, hdr, u);
        box[BX_EQN] = mkbox(fx + fw - m - 0.58f * fw, fy + m, 0.58f * fw, hdr, u);
        hdr += g;
    }
    {
        float fy0 = b1 - m - footer, x = fx + m, aw = fw - 2 * m - 3 * g;
        int haveplot = box[BX_PLOT].on;
        if (!haveplot) {
            box[BX_PLOT] = mkbox(x, fy0, 0.33f * aw, footer, u); x += 0.33f * aw + g;
            box[BX_READ] = mkbox(x, fy0, 0.24f * aw, footer, u); x += 0.24f * aw + g;
            box[BX_LEGEND] = mkbox(x, fy0, 0.15f * aw, footer, u); x += 0.15f * aw + g;
            box[BX_BLOCK] = mkbox(x, fy0, fx + fw - m - x, footer, u);
        } else {
            box[BX_LEGEND] = mkbox(x, fy0, 0.36f * aw, footer, u);
            box[BX_BLOCK] = mkbox(fx + fw - m - 0.5f * aw, fy0, 0.5f * aw, footer, u);
        }
        stage[0] = m; stage[2] = canvas_w - 2 * m;
        stage[1] = b0 + m + hdr; stage[3] = fy0 - g - stage[1];
    }
}

/* the world -> canvas map of the current model */
static double wk = 1, wox, woy;
static float WX(double x) { return (float)(wox + x * wk); }
static float WY(double y) { return (float)(woy - y * wk); }

/* fit the world box (x0..x1, y0..y1) in the stage, centred, a margin around */
static void fit(double x0, double y0, double x1, double y1, float marg) {
    float sw = stage[2] - 2 * marg, sh = stage[3] - 2 * marg;
    wk = fmin(sw / (x1 - x0), sh / (y1 - y0));
    wox = stage[0] + stage[2] * 0.5 - (x0 + x1) * 0.5 * wk;
    woy = stage[1] + stage[3] * 0.5 + (y0 + y1) * 0.5 * wk;
}

/* ======================================================================
 * the programme: a cycle of scenes, each a sheet of the drawing
 * ====================================================================== */

enum { S_BRIDGE, S_CONT, S_TOP };
typedef struct { int type, kind, sheet, nsheet; double t0, dur; uint32_t seed; } Scene;

#define DUR_BRIDGE 96.0
#define DUR_CONT 36.0
#define DUR_TOP 92.0

static const char *bridge_names[NBRIDGE] = {"warren", "pratt", "arch", "stayed", "suspension"};
static const char *cont_names[NCONT] = {"plate", "cantilever", "bracket"};
static const char *top_names[NTOP] = {"mbb", "deck"};
static int force_type = -1, force_kind = -1;

static Scene scene_at(double t) {
    Scene s;
    memset(&s, 0, sizeof s);
    if (t < 0) t = 0;
    if (force_type >= 0) {
        double d = force_type == S_BRIDGE ? DUR_BRIDGE : force_type == S_CONT ? DUR_CONT : DUR_TOP;
        long c = (long)floor(t / d);
        s.type = force_type; s.kind = force_kind; s.t0 = c * d; s.dur = d;
        s.sheet = 1; s.nsheet = 1;
        s.seed = hash3(opt_seed, (uint32_t)c, 7);
        return s;
    }
    /* the cycle: a bridge, the three continuum sheets, an optimisation */
    double cyc_d;
    int first = 0, last = 4;
    if (opt_act == 1) { first = last = 0; }
    else if (opt_act == 2) { first = 1; last = 3; }
    else if (opt_act == 3) { first = last = 4; }
    double d[5] = {DUR_BRIDGE, DUR_CONT, DUR_CONT, DUR_CONT, DUR_TOP};
    cyc_d = 0;
    for (int i = first; i <= last; i++) cyc_d += d[i];
    long c = (long)floor(t / cyc_d);
    double tc = t - c * cyc_d, acc = 0;
    int k = first;
    for (; k < last && tc >= acc + d[k]; k++) acc += d[k];
    s.t0 = c * cyc_d + acc;
    s.dur = d[k];
    s.sheet = k - first + 1;
    s.nsheet = last - first + 1;
    s.seed = hash3(opt_seed, (uint32_t)c, 11);
    if (k == 0) {
        s.type = S_BRIDGE;
        /* every kind once before any repeats, in a seeded order */
        int perm[NBRIDGE] = {0, 1, 2, 3, 4};
        uint32_t ps = hash3(opt_seed, (uint32_t)(c / NBRIDGE), 3);
        for (int i = NBRIDGE - 1; i > 0; i--) {
            int j = (int)(hash3(ps, (uint32_t)i, 1) % (uint32_t)(i + 1));
            int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
        s.kind = perm[c % NBRIDGE];
    } else if (k <= 3) {
        s.type = S_CONT;
        s.kind = (k - 1 + (int)(c % 3)) % 3;
    } else {
        s.type = S_TOP;
        s.kind = (int)((c + (opt_seed & 1)) % 2);
    }
    return s;
}

/* ======================================================================
 * the scene in hand
 * ====================================================================== */

static Scene cur;
static int have_cur;
static Br br;
static Cont ct;
static Top top;
static double ts;                       /* time into the scene */
static int top_shown;                   /* the optimiser's iteration on display */
static float top_frac;

#define TOP_START 5.0
#define TOP_RATE 1.6
#define SOLVE0 4.6
#define SOLVE1 10.5

static int same_scene(const Scene *a, const Scene *b) {
    return a->type == b->type && a->kind == b->kind && a->t0 == b->t0;
}

static void scene_free(void) {
    if (!have_cur) return;
    if (cur.type == S_BRIDGE) br_free(&br);
    else if (cur.type == S_CONT) cont_free(&ct);
    else top_free(&top);
    have_cur = 0;
}

static void scene_build(const Scene *s) {
    scene_free();
    cur = *s;
    if (s->type == S_BRIDGE) br_build(&br, s->kind, s->seed);
    else if (s->type == S_CONT) cont_build(&ct, s->kind);
    else if (s->kind == T_MBB) top_build(&top, T_MBB, 120, 40, 0.5, 4.8);
    else top_build(&top, T_BRIDGE, 120, 40, 0.3, 3.0);
    have_cur = 1;
}

/* bring the scene to time t; budget_ms < 0: everything it needs, now */
static void scene_update(double t, double budget_ms) {
    Scene s = scene_at(t);
    if (!have_cur || !same_scene(&s, &cur)) scene_build(&s);
    ts = t - cur.t0;
    if (cur.type == S_BRIDGE) br_advance(&br, ts);
    else if (cur.type == S_TOP) {
        double want = (ts - TOP_START) * TOP_RATE;
        if (want < 0) want = 0;
        if (budget_ms < 0) while (!top.done && top.nhist - 1 < (int)want + 1) top_work(&top, 50);
        else if (!top.done) top_work(&top, budget_ms);
        int k = (int)want;
        top_frac = (float)(want - k);
        if (k >= top.nhist - 1) { k = top.nhist - 1; top_frac = 0; }
        top_shown = k;
    }
}

/* ======================================================================
 * I  drawing a bridge
 * ====================================================================== */

static float smooth01(float a, float b, float x) {
    float f = (x - a) / (b - a);
    f = f < 0 ? 0 : f > 1 ? 1 : f;
    return f * f * (3 - 2 * f);
}

static void col_a(float *o, const float *c, float a) { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = c[3] * a; }
static void col3(float *o, const float *c, float a) { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = a; }

/* the ground: abutments at the bridge's ends, a valley under it */
static double ground_y(const Br *B, double x) {
    double L = B->span, top = B->deck_y - 1.4, bot = B->ground;
    double a0 = B->x0 + 0.03 * L, a1 = B->x0 + 0.22 * L, b0 = B->x1 - 0.22 * L, b1 = B->x1 - 0.03 * L;
    if (x <= a0 || x >= b1) return top;
    if (x < a1) { double f = (x - a0) / (a1 - a0); return top + (bot - top) * (0.5 - 0.5 * cos(PI * f)); }
    if (x > b0) { double f = (x - b0) / (b1 - b0); return bot + (top - bot) * (0.5 - 0.5 * cos(PI * f)); }
    return bot;
}

static void support(float x, float y, int type, float s, float rot_dn) {
    float ink[4], hat[4];
    col_a(ink, P->ink, 0.9f);
    col_a(hat, P->dim, 0.55f);
    (void)rot_dn;
    if (type == 2) {
        rect_hatch(x - 1.2f * s, y, x + 1.2f * s, y + 0.7f * s, hat);
        line(x - 1.3f * s, y, x + 1.3f * s, y, 0.14f * s, ink);
        return;
    }
    float h = s, b = 0.62f * s;
    float lw = fmaxf(1.2f / sc, 0.09f * s);
    line(x, y, x - b, y + h, lw, ink);
    line(x - b, y + h, x + b, y + h, lw, ink);
    line(x + b, y + h, x, y, lw, ink);
    disc(x, y, 0.16f * s, ink);
    float gy = y + h;
    if (type == 1 || type == 3) {
        for (int i = -1; i <= 1; i += 2) ring(x + i * 0.36f * s, y + h + 0.2f * s, 0.18f * s, lw, ink);
        gy += 0.4f * s;
    }
    line(x - 1.1f * s, gy, x + 1.1f * s, gy, lw, ink);
    rect_hatch(x - 1.1f * s, gy, x + 1.1f * s, gy + 0.45f * s, hat);
}

static void dim_h(float x0, float x1, float y, float ext0, float ext1, const char *label, float u) {
    float c[4];
    col_a(c, P->dim, 0.9f);
    float lw = fmaxf(1.0f / sc, 0.9f * u), a = 8 * u;
    line(x0, ext0, x0, y + 4 * u * (y > ext0 ? 1 : -1), lw, c);
    line(x1, ext1, x1, y + 4 * u * (y > ext1 ? 1 : -1), lw, c);
    arrow(x0 + (x1 - x0) * 0.5f, y, x0, y, lw, a, c);
    arrow(x0 + (x1 - x0) * 0.5f, y, x1, y, lw, a, c);
    float tw = text_w(F_MONO, 12 * u, label, 0) + 10 * u, cx = (x0 + x1) * 0.5f;
    float bg[4];
    col3(bg, P->bg0, 1);
    rect(cx - tw / 2, y - 8 * u, cx + tw / 2, y + 8 * u, bg);
    text(F_MONO, 12 * u, cx, y + cap_h(F_MONO, 12 * u) * 0.5f, 0, label, c, 0);
}

static void dim_v(float x, float y0, float y1, const char *label, float u, int side) {
    float c[4];
    col_a(c, P->dim, 0.9f);
    float lw = fmaxf(1.0f / sc, 0.9f * u), a = 8 * u;
    arrow(x, (y0 + y1) * 0.5f, x, y0, lw, a, c);
    arrow(x, (y0 + y1) * 0.5f, x, y1, lw, a, c);
    line(x - 5 * u, y0, x + 5 * u, y0, lw, c);
    line(x - 5 * u, y1, x + 5 * u, y1, lw, c);
    if (side == 0) side = x + 12 * u + text_w(F_MONO, 12 * u, label, 0) > stage[0] + stage[2] ? -1 : 1;
    text(F_MONO, 12 * u, x + side * 8 * u, (y0 + y1) * 0.5f + cap_h(F_MONO, 12 * u) * 0.5f, -side, label, c, 0);
}

/* the live displacements now, between the last two steps */
static double *ulive;
static void br_live(void) {
    static int cap;
    if (cap < br.n) { ulive = realloc(ulive, sizeof(double) * br.n); cap = br.n; }
    double fr = (ts - br.step * br.dt) / br.dt + 1;
    if (br.step == 0) fr = 1;
    fr = fr < 0 ? 0 : fr > 1 ? 1 : fr;
    for (int i = 0; i < br.n; i++) ulive[i] = br.u0[i] + (br.u[i] - br.u0[i]) * fr;
}

/* a point along bar i at xi in [0,1], displaced (exaggerated) */
static void bar_point(const Br *B, int i, double xi, const double *u, double k, double *X, double *Y) {
    const Bar *e = &B->e[i];
    double xa = B->px[e->a], ya = B->py[e->a], xb = B->px[e->b], yb = B->py[e->b];
    double L = e->L, c = (xb - xa) / L, s = (yb - ya) / L;
    const double *ua = u + e->a * 3, *ub = u + e->b * 3;
    double la = c * ua[0] + s * ua[1], lb = c * ub[0] + s * ub[1];
    double va = -s * ua[0] + c * ua[1], vb2 = -s * ub[0] + c * ub[1];
    double ax = la + (lb - la) * xi, v;
    if (e->truss) v = va + (vb2 - va) * xi;
    else {
        double x2 = xi * xi, x3 = x2 * xi;
        v = va * (1 - 3 * x2 + 2 * x3) + ua[2] * L * (xi - 2 * x2 + x3) + vb2 * (3 * x2 - 2 * x3) + ub[2] * L * (-x2 + x3);
    }
    *X = xa + (xb - xa) * xi + k * (c * ax - s * v);
    *Y = ya + (yb - ya) * xi + k * (s * ax + c * v);
}

static float stage_fade(float X) {
    float e = 50 * U;
    return smooth01(stage[0] - e, stage[0] + e, X) * (1 - smooth01(stage[0] + stage[2] - e, stage[0] + stage[2] + e, X));
}

static void draw_train(const Br *B, const Train *T, double t) {
    if (t < T->t0) return;
    double head = T->s0 + T->v * (t - T->t0);
    int dir = T->v > 0 ? 1 : -1;
    float u = U;
    for (int c = 0; c < T->ncar; c++) {
        double xa = head - dir * T->car0[c], xb = head - dir * T->car1[c];
        if (xa > xb) { double tmp = xa; xa = xb; xb = tmp; }
        double ya = B->deck_y + br_deck_v(B, ulive, xa) * B->scale, yb = B->deck_y + br_deck_v(B, ulive, xb) * B->scale;
        float fa = stage_fade(WX(xa)), fb = stage_fade(WX(xb)), f = fminf(fa, fb);
        if (f <= 0.01f) continue;
        int lo = T->loco[c];
        double hgt = lo ? 4.3 : 3.9, z0 = 1.05;
        /* the body, tilted with the deck */
        double dx = xb - xa, dy = yb - ya, L = sqrt(dx * dx + dy * dy), nx = -dy / L, ny = dx / L;
        float fill[4], ink[4], win[4];
        col_a(fill, P->panel, 1); fill[3] = 0.92f * f;
        col_a(ink, P->ink, 0.95f * f);
        col_a(win, P->dim, 0.55f * f);
#define PX(x_, z_) WX(xa + (x_) * dx / L + nx * (z_)), WY(ya + (x_) * dy / L + ny * (z_))
        double cl = L;
        double nose = lo ? 2.2 : 0;    /* the cab end, raked */
        int nose_front = lo && ((dir > 0) ? 1 : 0);
        double r0 = nose_front ? 0 : nose, r1 = nose_front ? nose : 0;
        float q[6][2] = {{PX(0, z0)}, {PX(cl, z0)}, {PX(cl, z0 + hgt * 0.55)}, {PX(cl - r1, z0 + hgt)},
                         {PX(r0, z0 + hgt)}, {PX(0, z0 + hgt * 0.55)}};
        (void)cl;
        for (int i = 1; i < 5; i++) tri(q[0][0], q[0][1], q[i][0], q[i][1], q[i + 1][0], q[i + 1][1], fill);
        float lw = fmaxf(1.0f / sc, 1.4f * u);
        for (int i = 0; i < 6; i++) line(q[i][0], q[i][1], q[(i + 1) % 6][0], q[(i + 1) % 6][1], lw, ink);
        /* windows or ribs */
        if (lo) {
            double wx0 = nose_front ? L - 3.6 : 0.9, wx1 = wx0 + 2.7;
            float w4[4][2] = {{PX(wx0, z0 + hgt * 0.62)}, {PX(wx1, z0 + hgt * 0.62)}, {PX(wx1, z0 + hgt * 0.88)}, {PX(wx0, z0 + hgt * 0.88)}};
            tri(w4[0][0], w4[0][1], w4[1][0], w4[1][1], w4[2][0], w4[2][1], win);
            tri(w4[0][0], w4[0][1], w4[2][0], w4[2][1], w4[3][0], w4[3][1], win);
            float a0[2] = {PX(0.6, z0 + hgt * 0.3)}, a1[2] = {PX(L - 0.6, z0 + hgt * 0.3)};
            line(a0[0], a0[1], a1[0], a1[1], lw, win);
        } else if (T->P[0] < 160e3 || c % 2) {
            for (double x = 1.4; x + 1.2 < L - 0.8; x += 2.1) {
                float w4[4][2] = {{PX(x, z0 + hgt * 0.5)}, {PX(x + 1.3, z0 + hgt * 0.5)}, {PX(x + 1.3, z0 + hgt * 0.8)}, {PX(x, z0 + hgt * 0.8)}};
                tri(w4[0][0], w4[0][1], w4[1][0], w4[1][1], w4[2][0], w4[2][1], win);
                tri(w4[0][0], w4[0][1], w4[2][0], w4[2][1], w4[3][0], w4[3][1], win);
            }
        } else {
            for (double x = 1.2; x < L - 0.8; x += 1.6) {
                float a0[2] = {PX(x, z0 + 0.3)}, a1[2] = {PX(x, z0 + hgt - 0.3)};
                line(a0[0], a0[1], a1[0], a1[1], lw * 0.7f, win);
            }
        }
#undef PX
    }
    /* wheels and the axle loads */
    float ink[4], acc[4];
    for (int i = 0; i < T->naxle; i++) {
        double x = axle_x(T, i, t);
        float f = stage_fade(WX(x));
        if (f < 0.01f) continue;
        double y = B->deck_y + br_deck_v(B, ulive, x) * B->scale;
        col_a(ink, P->ink, 0.95f * f);
        float r = (float)fmax(0.46 * wk, 2.0 * U);
        disc(WX(x), WY(y) - r, r, ink);
        if (x > B->x0 && x < B->x1 && (i % 2 == 0)) {
            double xm = 0.5 * (x + axle_x(T, i + 1 < T->naxle ? i + 1 : i, t));
            double ym = B->deck_y + br_deck_v(B, ulive, xm) * B->scale;
            col_a(acc, P->accent, 0.95f * f);
            float top = WY(ym + 4.3 + 1.05 + 2.8), bot = WY(ym + 4.3 + 1.05 + 0.25);
            arrow(WX(xm), top, WX(xm), bot, fmaxf(1.2f / sc, 1.6f * U), 7 * U, acc);
        }
    }
}

static void draw_bridge(void) {
    const Br *B = &br;
    br_live();
    double L = B->span;
    /* fit the superstructure (the valley may run off the bottom) */
    double nx0 = 1e9, nx1 = -1e9, ny1 = -1e9;
    for (int i = 0; i < B->nn; i++) {
        nx0 = fmin(nx0, B->px[i]); nx1 = fmax(nx1, B->px[i]);
        ny1 = fmax(ny1, B->py[i]);
    }
    double padx = 0.07 * L;
    fit(nx0 - padx, span_me < 0 ? fmin(B->deck_y - 6.5, B->ground - 1.5) : B->deck_y - 6.5, nx1 + padx, ny1 + 4.5, 0);
    float u = U;
    float bot = span_me < 0 ? stage[1] + stage[3] : canvas_h;
    double reveal = smooth01(0.2f, 3.2f, (float)ts);

    /* ground, hatched, and water in the valley */
    float gnd[4], gl[4], wat[4];
    col_a(gnd, P->dim, 0.32f);
    col_a(gl, P->ink, 0.7f);
    col_a(wat, P->dim, 0.6f);
    double gx0 = (stage[0] - wox) / wk - 10, gx1 = (stage[0] + stage[2] - wox) / wk + 10;
    int ng = 160;
    for (int i = 0; i < ng; i++) {
        double xa = gx0 + (gx1 - gx0) * i / ng, xb = gx0 + (gx1 - gx0) * (i + 1) / ng;
        float Xa = WX(xa), Xb = WX(xb), Ya = WY(ground_y(B, xa)), Yb = WY(ground_y(B, xb));
        float bb = fmaxf(bot, fmaxf(Ya, Yb) + 4 * u);
        vput(Xa, Ya, gnd, 0, 0, 1e6f, -2); vput(Xb, Yb, gnd, 0, 0, 1e6f, -2); vput(Xb, bb, gnd, 0, 0, 1e6f, -2);
        vput(Xa, Ya, gnd, 0, 0, 1e6f, -2); vput(Xb, bb, gnd, 0, 0, 1e6f, -2); vput(Xa, bb, gnd, 0, 0, 1e6f, -2);
        line(Xa, Ya, Xb, Yb, fmaxf(1.0f / sc, 1.5f * u), gl);
    }
    {
        double wy = B->ground + 0.3 * (B->deck_y - 1.4 - B->ground);
        double xa = B->x0, xb = B->x1;
        for (double x = B->x0; x < B->x1; x += 0.25) if (ground_y(B, x) < wy) { xa = x; break; }
        for (double x = B->x1; x > B->x0; x -= 0.25) if (ground_y(B, x) < wy) { xb = x; break; }
        float bgw[4];
        col3(bgw, P->bg1, 1);
        rect(WX(xa), WY(wy), WX(xb), WY(B->ground) + 1, bgw);
        float wfill[4];
        col_a(wfill, P->accent, 0.0f);
        (void)wfill;
        line(WX(xa), WY(wy), WX(xb), WY(wy), fmaxf(1.0f / sc, 1.2f * u), wat);
        for (int k = 1; k <= 3; k++) {
            double yy = wy - k * 0.9, inset = k * 0.08 * (xb - xa);
            dashed(WX(xa + inset), WY(yy), WX(xb - inset), WY(yy), fmaxf(1.0f / sc, 1.0f * u), 18 * u, 14 * u, wat);
        }
        float xm = WX(xa + 0.7 * (xb - xa)), ym = WY(wy);
        tri(xm - 6 * u, ym - 10 * u, xm + 6 * u, ym - 10 * u, xm, ym - 1 * u, wat);
    }
    /* approach tracks */
    {
        float tr[4];
        col_a(tr, P->ink, 0.8f);
        float lw = fmaxf(1.0f / sc, 2.0f * u);
        line(stage[0] - 60 * u, WY(B->deck_y), WX(B->x0), WY(B->deck_y), lw, tr);
        line(WX(B->x1), WY(B->deck_y), stage[0] + stage[2] + 60 * u, WY(B->deck_y), lw, tr);
        for (double x = gx0; x < gx1; x += 2.0) {
            if (x > B->x0 - 0.5 && x < B->x1 + 0.5) continue;
            line(WX(x), WY(B->deck_y), WX(x), WY(B->deck_y) + 0.3f * (float)wk + 2 * u, lw * 0.7f, tr);
        }
    }
    /* abutments */
    {
        float ab[4], abl[4];
        col_a(ab, P->dim, 0.45f);
        col_a(abl, P->ink, 0.8f);
        for (int side = 0; side < 2; side++) {
            double x = side ? B->x1 : B->x0, w = 0.05 * L, s = side ? 1 : -1;
            double ytop = B->deck_y - 1.4;
            float X0 = WX(x - s * 0.006 * L), X1 = WX(x + s * w);
            if (X0 > X1) { float t2 = X0; X0 = X1; X1 = t2; }
            rect_hatch(X0, WY(ytop), X1, WY(ytop - 5), ab);
            line(X0, WY(ytop), X1, WY(ytop), fmaxf(1.0f / sc, 1.4f * u), abl);
        }
    }
    /* trains, behind the near truss plane */
    for (int k = 0; k < B->ntrain; k++) draw_train(B, &B->tr[k], ts);
    /* the undeformed shape, faint */
    float ghost[4];
    col_a(ghost, P->dim, 0.45f);
    for (int i = 0; i < B->nb; i++) {
        const Bar *e = &B->e[i];
        dashed(WX(B->px[e->a]), WY(B->py[e->a]), WX(B->px[e->b]), WY(B->py[e->b]), fmaxf(1.0f / sc, 0.8f * u), 6 * u, 5 * u, ghost);
    }
    /* members by axial stress, deformed (live load, exaggerated) */
    double fr = (ts - br.step * br.dt) / br.dt + 1;
    fr = fr < 0 ? 0 : fr > 1 ? 1 : fr;
    double smx = 0, smn = 0;
    for (int i = 0; i < B->nb; i++) {
        const Bar *e = &B->e[i];
        double xm = 0.5 * (B->px[e->a] + B->px[e->b]);
        float rv = smooth01((float)((xm - B->x0) / L - 0.15), (float)((xm - B->x0) / L + 0.15), (float)(reveal * 1.3 - 0.15));
        if (rv <= 0.01f) continue;
        double N = br_N(B, i, fr), sg = N / e->A;
        smx = fmax(smx, sg); smn = fmin(smn, sg);
        float s = 2.5f + 0.5f * (float)fmax(-1, fmin(1, sg / B->smax));
        float base = e->grp == G_DECK || e->grp == G_CHORD || e->grp == G_ARCH ? 0.34f : e->grp == G_PYLON ? 0.6f :
                     e->grp == G_WEB ? 0.2f : e->grp == G_CABLE ? 0.16f : 0.11f;
        float w = fmaxf(base * (float)wk, (e->grp == G_HANGER || e->grp == G_CABLE ? 1.4f : 2.0f) * u);
        w = fminf(w, (e->grp == G_PYLON ? 14.0f : 8.0f) * u);
        float cc[4] = {1, 1, 1, rv};
        int nseg = e->truss ? 1 : 6;
        double px0, py0;
        bar_point(B, i, 0, ulive, B->scale, &px0, &py0);
        for (int k = 1; k <= nseg; k++) {
            double px1, py1;
            bar_point(B, i, (double)k / nseg, ulive, B->scale, &px1, &py1);
            if (B->slack[i]) {
                float sl[4];
                col_a(sl, P->dim, 0.8f * rv);
                dashed(WX(px0), WY(py0), WX(px1), WY(py1), w * 0.6f, 6 * u, 5 * u, sl);
            } else
                line_s(WX(px0), WY(py0), WX(px1), WY(py1), w, cc, s, s);
            px0 = px1; py0 = py1;
        }
    }
    br.smax_now = smx; br.smin_now = smn;
    /* joints */
    float jc[4];
    col_a(jc, P->ink, 0.9f * (float)reveal);
    for (int i = 0; i < B->nn; i++) {
        int frame = 0, any = 0;
        for (int k = 0; k < B->nb; k++)
            if (B->e[k].a == i || B->e[k].b == i) { any = 1; if (!B->e[k].truss) frame++; }
        if (!any) continue;
        double X = B->px[i] + ulive[i * 3] * B->scale, Y = B->py[i] + ulive[i * 3 + 1] * B->scale;
        disc(WX(X), WY(Y), fmaxf(1.8f * u, 0.16f * (float)wk), jc);
    }
    /* supports */
    float ss = fmaxf(9 * u, fminf(22 * u, 1.3f * (float)wk));
    for (int k = 0; k < B->nsup; k++) {
        int nd = B->sup_n[k];
        support(WX(B->px[nd]), WY(B->py[nd]) + (B->sup_t[k] == 2 ? 0 : 0.18f * ss), B->sup_t[k], ss, 0);
    }
    /* the monitored point */
    {
        double x = B->mon_x, y = B->deck_y + br_deck_v(B, ulive, x) * B->scale;
        float ac[4];
        col_a(ac, P->accent, 1);
        float X = WX(x), Y = WY(y), r = 9 * u;
        ring(X, Y, r, fmaxf(1.0f / sc, 1.6f * u), ac);
        line(X - 1.6f * r, Y, X - 0.5f * r, Y, fmaxf(1.0f / sc, 1.4f * u), ac);
        line(X + 0.5f * r, Y, X + 1.6f * r, Y, fmaxf(1.0f / sc, 1.4f * u), ac);
        math("δ_{\\r{mid}}", 15 * u, X + 1.8f * r, Y + 22 * u, ac, NULL);
    }
    /* dimensions and the scale */
    {
        char lab[64];
        snprintf(lab, sizeof lab, "L = %.1f m", L);
        float yd = span_me < 0 ? fminf(stage[1] + stage[3] - 14 * u, WY(B->ground) + 4 * u) : WY(B->deck_y - 3.6 - B->scale * B->vmax_live * 1.3);
        dim_h(WX(B->x0), WX(B->x1), yd, WY(B->deck_y - 1.4) + 4 * u, WY(B->deck_y - 1.4) + 4 * u, lab, u);
        snprintf(lab, sizeof lab, "%.1f m", B->height);
        double hx = B->kind == B_STAYED ? B->x0 + 0.5 * L : B->kind == B_SUSP ? B->x1 : B->kind == B_ARCH ? B->x0 + 0.5 * L : B->x1 - 0.5 * L / 8;
        float xo = B->kind == B_STAYED || B->kind == B_ARCH ? 28 * u : B->kind == B_SUSP ? 24 * u : 0;
        if (B->kind == B_WARREN || B->kind == B_PRATT) hx = B->x1 + 0.035 * L, xo = 0;
        dim_v(WX(hx) + xo, WY(B->deck_y), WY(B->deck_y + B->height), lab, u, 0);
        float c[4];
        col_a(c, P->dim, 1);
        char sl[64];
        snprintf(sl, sizeof sl, "deflection (live load) ×%g", B->scale);
        text(F_SANS, 13 * u, WX(nx0) + 4 * u, WY(ny1 + 3.0), -1, sl, c, 0.02f);
    }
}

/* ======================================================================
 * II  drawing a continuum
 * ====================================================================== */

static float *cfield, *cxy;            /* per node: the colour coordinate, the drawn position */
static int ccap;
static double c_lam, c_phi, c_fmax, c_umax, c_dscale;
static int c_solving, c_snap;
static double c_fe_now;                /* the quantity under check, now */

/* the load now: the plate cycles (fatigue, R = 0.1), the cantilever
 * breathes, the bracket's load swings */
static void cont_load(double t, double *lam, double *phi) {
    if (ct.kind == C_PLATE) { *lam = 0.55 - 0.45 * cos(2 * PI * fmax(t - SOLVE1, 0) / 7.0); *phi = 0; }
    else if (ct.kind == C_CANTI) { *lam = 0.85 + 0.15 * cos(2 * PI * fmax(t - SOLVE1, 0) / 6.0); *phi = 0; }
    else { *lam = 1; *phi = -0.5 * PI + 0.95 * sin(2 * PI * fmax(t - SOLVE1, 0) / 12.0); }
}

static void cont_u(int n, double lam, double phi, double *ux, double *uy) {
    if (ct.kind == C_BRACKET) {
        double a = cos(phi) * ct.load, b = sin(phi) * ct.load;
        *ux = a * ct.u[0][n * 2] + b * ct.u[1][n * 2];
        *uy = a * ct.u[0][n * 2 + 1] + b * ct.u[1][n * 2 + 1];
    } else {
        *ux = lam * ct.load * ct.u[0][n * 2];
        *uy = lam * ct.load * ct.u[0][n * 2 + 1];
    }
}

static void cont_stress(int n, double lam, double phi, double s[3]) {
    for (int k = 0; k < 3; k++) {
        if (ct.kind == C_BRACKET) s[k] = ct.load * (cos(phi) * ct.sn[0][n * 3 + k] + sin(phi) * ct.sn[1][n * 3 + k]);
        else s[k] = lam * ct.load * ct.sn[0][n * 3 + k];
    }
}

static double cont_field(int n, double lam, double phi) {
    double s[3], vm, s1, s2, an;
    cont_stress(n, lam, phi, s);
    if (ct.kind == C_PLATE) return s[0];
    vm_principal(s, &vm, &s1, &s2, &an);
    return vm;
}

/* the colour scale's top: the field at full load, its peak (the bracket's
 * corner is singular: a high percentile instead) */
static double cont_fmax(void) {
    int n = ct.nn;
    double *v = malloc(sizeof(double) * n), m = 0;
    for (int i = 0; i < n; i++) {
        double a = 0;
        if (ct.kind == C_BRACKET)
            for (int k = 0; k < 16; k++) a = fmax(a, cont_field(i, 1, -0.5 * PI + 0.95 * sin(2 * PI * k / 16)));
        else a = cont_field(i, 1, 0);
        v[i] = a;
        m = fmax(m, a);
    }
    if (ct.kind == C_BRACKET) {
        /* the 99.5th percentile */
        for (int i = 1; i < n; i++)
            for (int j = i; j > 0 && v[j] < v[j - 1]; j--) { double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
        m = v[(int)(n * 0.995)];
    }
    free(v);
    return nice_up(m * 1.02);
}

static void draw_cont(void) {
    int nn = ct.nn;
    if (ccap < nn) { cfield = realloc(cfield, sizeof(float) * nn); cxy = realloc(cxy, sizeof(float) * 2 * nn); ccap = nn; }
    float u = U;
    double mx = ct.bx1 - ct.bx0, my = ct.by1 - ct.by0;
    /* room around for the loads, supports and dimensions */
    if (ct.kind == C_PLATE) fit(ct.bx0 - 0.17 * mx, ct.by0 - 0.32 * my, ct.bx1 + 0.17 * mx, ct.by1 + 0.25 * my, 0);
    else if (ct.kind == C_CANTI) fit(ct.bx0 - 0.08 * mx, ct.by0 - 1.5 * my, ct.bx1 + 0.08 * mx, ct.by1 + 1.4 * my, 0);
    else fit(ct.bx0 - 0.25 * mx, ct.by0 - 0.12 * my, ct.bx1 + 0.3 * mx, ct.by1 + 0.3 * my, 0);
    static Cont *fm_for;
    static uint64_t fm_id;
    uint64_t id = (uint64_t)(cur.t0 * 1000) * 4 + (uint64_t)ct.kind;
    if (fm_for != &ct || fm_id != id) { c_fmax = cont_fmax(); fm_for = &ct; fm_id = id; }
    double lam, phi;
    cont_load(ts, &lam, &phi);
    c_lam = lam; c_phi = phi;
    /* the exaggeration: the biggest displacement at full load a few % of the size */
    double um = 1e-12;
    for (int i = 0; i < nn; i++) {
        double ux, uy;
        if (ct.kind == C_BRACKET) {
            for (int k = 0; k < 2; k++) um = fmax(um, ct.load * hypot(ct.u[k][i * 2], ct.u[k][i * 2 + 1]));
        } else {
            cont_u(i, 1, 0, &ux, &uy);
            um = fmax(um, hypot(ux, uy));
        }
    }
    c_umax = um;
    c_dscale = nice_num((ct.kind == C_PLATE ? 0.025 : 0.06) * fmax(mx, my) / um, 1);
    /* solving: the iterate on display */
    c_solving = ts < SOLVE1;
    float a_mesh = smooth01(0.3f, 1.6f, (float)ts);
    double fsolve = (ts - SOLVE0) / (SOLVE1 - SOLVE0 - 0.6);
    fsolve = fsolve < 0 ? 0 : fsolve > 1 ? 1 : fsolve;
    /* iterations on a log-ish clock: the early ones are where it all happens */
    double itf = fsolve <= 0 ? -1 : pow(fsolve, 1.6) * ct.iters;
    int snap = 0;
    for (int k = 0; k < ct.nsnap; k++) if (ct.snap_it[k] <= itf) snap = k;
    c_snap = itf < 0 ? -1 : snap;
    double snorm = 1e-30;
    if (c_solving && c_snap >= 0)
        for (int i = 0; i < nn; i++)
            snorm = fmax(snorm, hypot(ct.snap[(size_t)snap * 2 * nn + i * 2], ct.snap[(size_t)snap * 2 * nn + i * 2 + 1]));
    float blend = smooth01((float)SOLVE1 - 0.6f, (float)SOLVE1 + 0.4f, (float)ts);   /* iterate -> stress */
    double fnorm = 0;
    for (int i = 0; i < nn; i++) fnorm = fmax(fnorm, hypot(ct.u[0][i * 2], ct.u[0][i * 2 + 1]));
    for (int i = 0; i < nn; i++) {
        double ux, uy, sx = 0, sy = 0;
        cont_u(i, lam, phi, &ux, &uy);
        double fld = cont_field(i, lam, phi) / c_fmax;
        if (ct.kind == C_PLATE) fld = (cont_field(i, lam, phi) + 0.1 * c_fmax) / (1.1 * c_fmax);
        double it_mag = 0;
        if (c_snap >= 0) {
            double a = ct.snap[(size_t)snap * 2 * nn + i * 2], b = ct.snap[(size_t)snap * 2 * nn + i * 2 + 1];
            it_mag = hypot(a, b) / fnorm;
            double k = (ct.kind == C_BRACKET ? cos(-0.5 * PI) * 0 + 1 : lam) * ct.load;
            if (ct.kind == C_BRACKET) { sx = a * ct.load * cos(phi); sy = b * ct.load * cos(phi); sx = a * ct.load; sy = b * ct.load; (void)k; }
            else { sx = a * k; sy = b * k; }
        }
        double dx = c_solving ? (c_snap >= 0 ? sx : 0) : ux, dy = c_solving ? (c_snap >= 0 ? sy : 0) : uy;
        if (blend > 0 && blend < 1) { dx = dx + (ux - dx) * blend; dy = dy + (uy - dy) * blend; }
        cxy[i * 2] = WX(ct.xy[i * 2] + c_dscale * dx);
        cxy[i * 2 + 1] = WY(ct.xy[i * 2 + 1] + c_dscale * dy);
        float fs = (float)(fld < 0 ? 0 : fld > 1 ? 1 : fld);
        float fi = (float)(it_mag > 1 ? 1 : it_mag);
        /* encode: the iterate's |u| on the smooth map, the stress banded */
        cfield[i] = blend >= 1 ? fs : fi;
        if (blend > 0 && blend < 1) cfield[i] = fi + (fs - fi) * blend;
    }
    (void)snorm;
    int banded = blend >= 0.5f;
    float fillc[4] = {1, 1, 1, a_mesh * (c_snap >= 0 || !c_solving ? 1.0f : 0.0f)};
    if (fillc[3] > 0)
        for (int e = 0; e < ct.ne; e++) {
            const int *q = ct.q + e * 4;
            float sbase = banded ? 0 : 4;
            for (int t = 0; t < 2; t++) {
                int a = q[0], b = q[1 + t], c = q[2 + t];
                vput(cxy[a * 2], cxy[a * 2 + 1], fillc, 0, 0, 1e6f, sbase + cfield[a]);
                vput(cxy[b * 2], cxy[b * 2 + 1], fillc, 0, 0, 1e6f, sbase + cfield[b]);
                vput(cxy[c * 2], cxy[c * 2 + 1], fillc, 0, 0, 1e6f, sbase + cfield[c]);
            }
        }
    /* the mesh, then the outline (edges with one element) */
    float mc[4], oc[4];
    col3(mc, P->bg0, (P->light ? 0.5f : 0.4f) * (c_snap >= 0 || !c_solving ? 1 : 0.0f));
    if (c_solving && c_snap < 0) col_a(mc, P->ink, 0.5f * a_mesh);
    col_a(oc, P->ink, 0.95f * a_mesh);
    float mw = fmaxf(0.7f / sc, 0.6f * u);
    for (int e = 0; e < ct.ne; e++) {
        const int *q = ct.q + e * 4;
        for (int k = 0; k < 4; k++) {
            int a = q[k], b = q[(k + 1) % 4];
            if (a > b) continue;
            line(cxy[a * 2], cxy[a * 2 + 1], cxy[b * 2], cxy[b * 2 + 1], mw, mc);
        }
    }
    /* outline: boundary edges */
    {
        static int *bnd, nbnd, bnd_for_ne;
        static uint64_t bnd_id;
        if (bnd_id != id || bnd_for_ne != ct.ne) {
            free(bnd);
            bnd = malloc(sizeof(int) * 2 * 4 * ct.ne);
            nbnd = 0;
            /* an edge is on the boundary if it isn't shared: hash on (min, max) */
            int cap = 16 * ct.ne + 7;
            int64_t *key = malloc(sizeof(int64_t) * cap);
            int *cnt = calloc((size_t)cap, sizeof(int));
            for (int i = 0; i < cap; i++) key[i] = -1;
            for (int pass = 0; pass < 2; pass++)
                for (int e = 0; e < ct.ne; e++)
                    for (int k = 0; k < 4; k++) {
                        int a = ct.q[e * 4 + k], b = ct.q[e * 4 + (k + 1) % 4];
                        int64_t kk = (int64_t)(a < b ? a : b) * 1000003 + (a < b ? b : a);
                        uint64_t h = ((uint64_t)kk * 0x9E3779B97F4A7C15ULL) % (uint64_t)cap;
                        while (key[h] >= 0 && key[h] != kk) h = (h + 1) % (uint64_t)cap;
                        if (pass == 0) { key[h] = kk; cnt[h]++; }
                        else if (cnt[h] == 1) { bnd[nbnd * 2] = a; bnd[nbnd * 2 + 1] = b; nbnd++; }
                    }
            free(key); free(cnt);
            bnd_id = id; bnd_for_ne = ct.ne;
        }
        float ow = fmaxf(1.0f / sc, 1.6f * u), gw[4];
        col_a(gw, P->dim, 0.5f * a_mesh);
        for (int k = 0; k < nbnd; k++) {
            int a = bnd[k * 2], b = bnd[k * 2 + 1];
            dashed(WX(ct.xy[a * 2]), WY(ct.xy[a * 2 + 1]), WX(ct.xy[b * 2]), WY(ct.xy[b * 2 + 1]), fmaxf(0.8f / sc, 0.8f * u), 5 * u, 4 * u, gw);
        }
        for (int k = 0; k < nbnd; k++) {
            int a = bnd[k * 2], b = bnd[k * 2 + 1];
            line(cxy[a * 2], cxy[a * 2 + 1], cxy[b * 2], cxy[b * 2 + 1], ow, oc);
        }
    }
    /* principal stress trajectories, over the cantilever once solved */
    if (ct.kind == C_CANTI && !c_solving) {
        float tc[2][4];
        float fa = smooth01((float)SOLVE1 + 0.5f, (float)SOLVE1 + 3, (float)ts);
        col_a(tc[0], P->light ? P->ink : P->ink, 0.85f * fa);
        col3(tc[1], P->bg0, 0.8f * fa);
        int off = 0;
        float tw = fmaxf(1.0f / sc, 1.3f * u);
        for (int k = 0; k < ct.ntraj; k++) {
            int n = ct.traj_n[k], fam = ct.traj_fam[k];
            float acc = 0;
            for (int i = 0; i + 1 < n; i++) {
                double x0 = ct.traj[(off + i) * 2], y0 = ct.traj[(off + i) * 2 + 1];
                double x1 = ct.traj[(off + i + 1) * 2], y1 = ct.traj[(off + i + 1) * 2 + 1];
                /* displaced with the beam: the deflection at x of the bottom fibre is close enough */
                int ia = nearest_node(&ct, x0, 0);
                (void)ia;
                double s = c_dscale * lam * ct.load;
                double d0 = 0, d1 = 0;
                {
                    int gi = (int)fmin(ct.gx, fmax(0, x0 / ct.gdx)), gj = (int)fmin(ct.gy, fmax(0, y0 / ct.gdx));
                    int nd = ct.gnode[gj * (ct.gx + 1) + gi];
                    if (nd >= 0) d0 = ct.u[0][nd * 2 + 1] * s;
                    gi = (int)fmin(ct.gx, fmax(0, x1 / ct.gdx)); gj = (int)fmin(ct.gy, fmax(0, y1 / ct.gdx));
                    nd = ct.gnode[gj * (ct.gx + 1) + gi];
                    if (nd >= 0) d1 = ct.u[0][nd * 2 + 1] * s;
                }
                float X0 = WX(x0), Y0 = WY(y0 + d0), X1 = WX(x1), Y1 = WY(y1 + d1);
                float seg = hypotf(X1 - X0, Y1 - Y0);
                /* σ2 dashed */
                if (fam == 0 || fmodf(acc, 12 * u) < 7 * u) line(X0, Y0, X1, Y1, tw, tc[fam]);
                acc += seg;
            }
            off += n;
        }
    }
    /* loads and supports */
    float ac[4], ink[4], hat[4], dm[4];
    col_a(ac, P->accent, a_mesh);
    col_a(ink, P->ink, 0.9f * a_mesh);
    col_a(hat, P->dim, 0.55f * a_mesh);
    col_a(dm, P->dim, a_mesh);
    float lw = fmaxf(1.0f / sc, 1.8f * u), ah = 11 * u;
    char lab[96];
    if (ct.kind == C_PLATE) {
        double L = ct.len, b = ct.b;
        float alen = (float)(0.11 * L * wk) * (0.4f + 0.6f * (float)lam);
        for (int k = 0; k <= 8; k++) {
            double y = -b + 2 * b * k / 8;
            int n = nearest_node(&ct, L, y), m = nearest_node(&ct, -L, y);
            arrow(cxy[n * 2] + 4 * u, cxy[n * 2 + 1], cxy[n * 2] + 4 * u + alen, cxy[n * 2 + 1], lw, ah, ac);
            arrow(cxy[m * 2] - 4 * u, cxy[m * 2 + 1], cxy[m * 2] - 4 * u - alen, cxy[m * 2 + 1], lw, ah, ac);
        }
        snprintf(lab, sizeof lab, "σ_{∞} = %.0f \\r{MPa}", lam * ct.load);
        math(lab, 16 * u, WX(L) + 10 * u, WY(b) - 14 * u, ac, NULL);
        snprintf(lab, sizeof lab, "W = %.0f", 2 * b);
        dim_v(WX(L) + (float)(0.13 * L * wk), WY(-b), WY(b), lab, u, 0);
        snprintf(lab, sizeof lab, "L = %.0f mm", 2 * L);
        dim_h(WX(-L), WX(L), WY(-b) + 34 * u, WY(-b) + 6 * u, WY(-b) + 6 * u, lab, u);
        /* the check: sigma_max at the top of the hole */
        if (!c_solving) {
            int n = nearest_node(&ct, 0, ct.a);
            double smax = cont_field(n, lam, 0);
            c_fe_now = smax;
            float X = cxy[n * 2], Y = cxy[n * 2 + 1];
            float fa = smooth01((float)SOLVE1, (float)SOLVE1 + 1.5f, (float)ts);
            float cc[4];
            col_a(cc, P->ink, fa);
            disc(X, Y, 3.5f * u, cc);
            float lx = X + (float)(0.10 * L * wk), ly = WY(b) - 42 * u;
            line(X, Y, lx, ly, fmaxf(1.0f / sc, 1.2f * u), cc);
            line(lx, ly, lx + 190 * u, ly, fmaxf(1.0f / sc, 1.2f * u), cc);
            snprintf(lab, sizeof lab, "σ_{\\r{max}} = %.0f \\r{MPa}", smax);
            math(lab, 15 * u, lx + 4 * u, ly - 7 * u, cc, NULL);
            snprintf(lab, sizeof lab, "K_{t} = %.3f", ct.check_fe);
            math(lab, 15 * u, lx + 4 * u, ly + 20 * u, cc, NULL);
            snprintf(lab, sizeof lab, "Ø%.0f", 2 * ct.a);
            text(F_MONO, 12 * u, WX(0), WY(-ct.a) + 20 * u, 0, lab, dm, 0);
        }
        float sx = cxy[nearest_node(&ct, -L, -b) * 2], sy = cxy[nearest_node(&ct, -L, -b) * 2 + 1];
        (void)sx; (void)sy;
    } else if (ct.kind == C_CANTI) {
        /* the clamp */
        float X = WX(0), Y0 = WY(ct.h), Y1 = WY(0), cw = 18 * u;
        rect_hatch(X - cw, Y0 - 14 * u, X, Y1 + 14 * u, hat);
        line(X, Y0 - 14 * u, X, Y1 + 14 * u, lw, ink);
        int n = nearest_node(&ct, ct.len, ct.h);
        float tx = cxy[n * 2], ty = cxy[n * 2 + 1];
        float alen = 70 * u * (0.5f + 0.5f * (float)lam);
        arrow(tx, ty - alen - 6 * u, tx, ty - 6 * u, lw * 1.2f, ah * 1.3f, ac);
        snprintf(lab, sizeof lab, "P = %.1f \\r{kN}", lam * ct.load / 1e3);
        math(lab, 16 * u, tx + 10 * u, ty - alen + 4 * u, ac, NULL);
        snprintf(lab, sizeof lab, "L = %.0f mm", ct.len);
        dim_h(WX(0), WX(ct.len), WY(0) + (float)(1.05 * ct.h * wk), WY(0) + 16 * u, WY(0) + 16 * u, lab, u);
        snprintf(lab, sizeof lab, "h = %.0f", ct.h);
        dim_v(WX(0) - 40 * u, WY(0), WY(ct.h), lab, u, -1);
        if (!c_solving) {
            int m = nearest_node(&ct, ct.len, ct.h / 2);
            double d = -lam * ct.load * ct.u[0][m * 2 + 1];
            c_fe_now = d;
            float fa = smooth01((float)SOLVE1, (float)SOLVE1 + 1.5f, (float)ts), cc[4];
            col_a(cc, P->ink, fa);
            float X2 = WX(ct.len) + 14 * u, Yu = WY(ct.h / 2), Yd = cxy[m * 2 + 1];
            dashed(WX(ct.len) - 6 * u, Yu, X2 + 14 * u, Yu, fmaxf(1.0f / sc, 1.0f * u), 4 * u, 3 * u, cc);
            arrow(X2 + 8 * u, Yu, X2 + 8 * u, Yd, fmaxf(1.0f / sc, 1.2f * u), 7 * u, cc);
            snprintf(lab, sizeof lab, "δ = %.2f \\r{mm}", d);
            math(lab, 15 * u, X2 + 16 * u, (Yu + Yd) * 0.5f + 5 * u, cc, NULL);
            float la[4];
            col_a(la, P->dim, fa);
            text(F_SANS, 12 * u, WX(ct.len * 0.5), WY(ct.h) - (float)(0.9 * ct.h * wk), 0,
                 "principal stress trajectories:  solid σ1,  dashed σ2", la, 0.02f);
        }
    } else {
        /* the bracket's foot, bolted down */
        float Y = WY(0);
        rect_hatch(WX(-40), Y, WX(100), Y + 16 * u, hat);
        line(WX(-40), Y, WX(100), Y, lw, ink);
        int n = nearest_node(&ct, 360, 210);
        float tx = cxy[n * 2], ty = cxy[n * 2 + 1];
        float alen = 90 * u, dx = (float)cos(phi), dy = -(float)sin(phi);
        arrow(tx - dx * (alen + 8 * u) + 0, ty - dy * (alen + 8 * u), tx - dx * 8 * u, ty - dy * 8 * u, lw * 1.2f, ah * 1.3f, ac);
        snprintf(lab, sizeof lab, "F = %.0f \\r{kN},  φ = %.0f°", ct.load / 1e3, phi * 180 / PI);
        math(lab, 16 * u, WX(300), WY(240) - (float)(0.22 * 240 * wk), ac, NULL);
        snprintf(lab, sizeof lab, "%.0f mm", 360.0);
        dim_h(WX(0), WX(360), WY(240) + -18 * u, WY(240) - 4 * u, WY(240) - 4 * u, lab, u);
        snprintf(lab, sizeof lab, "%.0f", 240.0);
        dim_v(WX(0) - 30 * u, WY(0), WY(240), lab, u, -1);
        if (!c_solving) {
            int m = nearest_node(&ct, 60, 180);
            double vm = cont_field(m, lam, phi);
            c_fe_now = vm;
            float fa = smooth01((float)SOLVE1, (float)SOLVE1 + 1.5f, (float)ts), cc[4];
            col_a(cc, P->ink, fa);
            float X = cxy[m * 2], Yc = cxy[m * 2 + 1];
            ring(X, Yc, 7 * u, fmaxf(1.0f / sc, 1.4f * u), cc);
            float lx = X + 120 * u, ly = Yc + 70 * u;
            line(X + 5 * u, Yc + 5 * u, lx, ly, fmaxf(1.0f / sc, 1.2f * u), cc);
            line(lx, ly, lx + 230 * u, ly, fmaxf(1.0f / sc, 1.2f * u), cc);
            snprintf(lab, sizeof lab, "\\r{re-entrant corner:}\\; σ_{\\r{vM}} = %.0f \\r{MPa}", vm);
            math(lab, 14 * u, lx + 4 * u, ly - 7 * u, cc, NULL);
            math("\\r{singular:}\\; σ ∼ r^{-0.455}", 14 * u, lx + 4 * u, ly + 20 * u, cc, NULL);
        }
    }
    {
        char sl[64];
        snprintf(sl, sizeof sl, c_solving ? "displacement |u| of the CG iterate,  ×%g" : "deformation ×%g", c_dscale);
        if (ct.kind == C_BRACKET) text(F_SANS, 13 * u, WX(ct.bx1), WY(0) + 30 * u, 1, sl, dm, 0.02f);
        else text(F_SANS, 13 * u, WX(ct.bx0), WY(ct.by1) - 16 * u, -1, sl, dm, 0.02f);
    }
}

/* ======================================================================
 * III  drawing the optimiser
 * ====================================================================== */

static GLuint tex_topo;
static float top_sharp = 1;
static unsigned char *topo_px;
static int topo_w, topo_h, topo_tw, topo_th;

static void draw_top(void) {
    const Top *T = &top;
    int nx = T->nelx, ny = T->nely, full = 2 * nx;
    float u = U;
    fit(-0.05 * full, -0.30 * ny, full * 1.05, ny * 1.30, 0);
    /* the design on display, between two iterations; mirrored about the left edge */
    if (topo_w != full || topo_h != ny) {
        free(topo_px);
        topo_px = malloc((size_t)full * ny * 4);
        topo_w = full; topo_h = ny;
    }
    int k = top_shown, k2 = k + 1 < T->nhist ? k + 1 : k;
    const float *h0 = T->hist + (size_t)k * T->ne, *h1 = T->hist + (size_t)k2 * T->ne;
    float fr = top_frac;
    /* the energy display once converged */
    float emix = T->done && k >= T->nhist - 1 ? smooth01(0, 3, (float)(ts - TOP_START - (T->nhist - 1) / TOP_RATE - 2)) : 0;
    double emax = 1e-30;
    if (emix > 0) for (int e = 0; e < T->ne; e++) emax = fmax(emax, T->ce[e] * pow(T->xPhys[e], T->penal));
    for (int X = 0; X < full; X++) {
        int ex = X < nx ? nx - 1 - X : X - nx;
        for (int ey = 0; ey < ny; ey++) {
            int e = ex * ny + ey;
            float d = h0[e] + (h1[e] - h0[e]) * fr;
            unsigned char *o = topo_px + ((size_t)ey * full + X) * 4;
            o[0] = (unsigned char)(d * 255 + 0.5f);
            float en = emix > 0 ? (float)sqrt(T->ce[e] * pow(T->xPhys[e], T->penal) / emax) : 0;
            o[1] = (unsigned char)(fminf(en, 1) * 255);
            o[2] = 0; o[3] = 255;
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex_topo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (topo_tw != full || topo_th != ny) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, full, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, topo_px);
        topo_tw = full; topo_th = ny;
    } else
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, full, ny, GL_RGBA, GL_UNSIGNED_BYTE, topo_px);
    float a = smooth01(0.3f, 1.6f, (float)ts);
    float sc4[4];
    col_a(sc4, P->solid, a);
    float X0 = WX(0), X1 = WX(full), Y0 = WY(ny), Y1 = WY(0);
    /* the domain's frame */
    float fc[4];
    col_a(fc, P->dim, 0.55f * a);
    vput(X0, Y0, sc4, 0, 0, emix, -4); vput(X1, Y0, sc4, 1, 0, emix, -4); vput(X1, Y1, sc4, 1, 1, emix, -4);
    vput(X0, Y0, sc4, 0, 0, emix, -4); vput(X1, Y1, sc4, 1, 1, emix, -4); vput(X0, Y1, sc4, 0, 1, emix, -4);
    float lw = fmaxf(1.0f / sc, 1.0f * u);
    line(X0, Y0, X1, Y0, lw, fc); line(X1, Y0, X1, Y1, lw, fc); line(X1, Y1, X0, Y1, lw, fc); line(X0, Y1, X0, Y0, lw, fc);
    /* the element grid, faintly, while it's still grey */
    float gfa = 0.22f * (1 - smooth01(4, 12, (float)ts)) * a;
    if (gfa > 0.01f && wk * sc > 5) {
        float gc[4];
        col_a(gc, P->dim, gfa);
        for (int i = 1; i < full; i++) line(WX(i), Y0, WX(i), Y1, 0.6f / sc, gc);
        for (int j = 1; j < ny; j++) line(X0, WY(j), X1, WY(j), 0.6f / sc, gc);
    }
    /* the symmetry line */
    float cl[4];
    col_a(cl, P->accent, 0.6f * a);
    dashed(WX(nx), Y0 - 18 * u, WX(nx), Y1 + 18 * u, fmaxf(1.0f / sc, 1.0f * u), 14 * u, 5 * u, cl);
    text(F_MONO, 12 * u, WX(nx) + 6 * u, Y1 + 30 * u, -1, "CL", cl, 0);
    /* supports and loads */
    float ss = fmaxf(16 * u, fminf(26 * u, 1.2f * (float)wk));
    float ac[4];
    col_a(ac, P->accent, a);
    top_sharp = 1.1f + 2.4f * smooth01(10, 60, (float)k);
    float aw = fmaxf(1.0f / sc, 1.8f * u);
    char lab[96];
    if (T->kind == T_MBB) {
        support(X0, Y1 + 0.18f * ss, 0, ss, 0);
        support(X1, Y1 + 0.18f * ss, 1, ss, 0);
        arrow(WX(nx), Y0 - 90 * u, WX(nx), Y0 - 4 * u, aw * 1.3f, 14 * u, ac);
        math("F", 17 * u, WX(nx) + 10 * u, Y0 - 60 * u, ac, NULL);
    } else {
        support(X0, Y1 + 0.18f * ss, 0, ss, 0);
        support(X1, Y1 + 0.18f * ss, 0, ss, 0);
        for (int i = 0; i <= 24; i++) {
            float x = X0 + (X1 - X0) * i / 24;
            arrow(x, Y0 - 44 * u, x, Y0 - 3 * u, aw * 0.8f, 9 * u, ac);
        }
        line(X0, Y0 - 44 * u, X1, Y0 - 44 * u, aw * 0.8f, ac);
        math("q", 17 * u, X0 + 8 * u, Y0 - 52 * u, ac, NULL);
        float dm[4];
        col_a(dm, P->dim, a);
        text(F_SANS, 12 * u, X1 - 4 * u, Y0 - 52 * u, 1, "deck: passive solid elements", dm, 0.02f);
    }
    float dm[4];
    col_a(dm, P->dim, a);
    snprintf(lab, sizeof lab, "%d × %d elements (half modelled, mirrored)", full, ny);
    dim_h(X0, X1, Y1 + ss * 2.2f + 12 * u, Y1 + ss * 2.0f, Y1 + ss * 2.0f, lab, u);
    snprintf(lab, sizeof lab, "iteration %d", k);
    text(F_MONO, 14 * u, X0, Y0 - 16 * u, -1, lab, dm, 0);
}

/* ======================================================================
 * the panels around the model
 * ====================================================================== */

static float panel(const Box *b, const char *label) {
    float u = b->u, fc[4], bc[4], lc[4], ac[4];
    col_a(fc, P->panel, 1);
    col_a(bc, P->dim, 0.35f);
    col_a(lc, P->dim, 1);
    col_a(ac, P->accent, 1);
    rect(b->x, b->y, b->x + b->w, b->y + b->h, fc);
    float lw = fmaxf(1.0f / sc, 1.0f * u);
    line(b->x, b->y, b->x + b->w, b->y, lw, bc);
    line(b->x + b->w, b->y, b->x + b->w, b->y + b->h, lw, bc);
    line(b->x + b->w, b->y + b->h, b->x, b->y + b->h, lw, bc);
    line(b->x, b->y + b->h, b->x, b->y, lw, bc);
    /* corner ticks */
    float t = 8 * u;
    line(b->x, b->y, b->x + t, b->y, 2 * lw, ac);
    line(b->x, b->y, b->x, b->y + t, 2 * lw, ac);
    float hy = b->y + 14 * u + cap_h(F_SEMI, 11 * u);
    if (label) text(F_SEMI, 11 * u, b->x + 14 * u, hy, -1, label, lc, 0.12f);
    return hy + 10 * u;
}

static void scene_titles(char *act, char *title, char *sub, size_t n) {
    if (cur.type == S_BRIDGE) {
        snprintf(act, n, "ACT I  ·  BRIDGES UNDER MOVING LOADS");
        snprintf(title, n, "%s", br.name);
        snprintf(sub, n, "%d nodes · %d members · %d dof · f1 = %.2f Hz · Newmark-β, Rayleigh ζ = 2%%", br.nn, br.nb, br.n, br.f1);
    } else if (cur.type == S_CONT) {
        static const char *t[] = {"PLATE WITH A HOLE", "CANTILEVER", "L-BRACKET"};
        snprintf(act, n, "ACT II  ·  CONTINUUM");
        snprintf(title, n, "%s", t[ct.kind]);
        snprintf(sub, n, "plane stress · QM6 quads · %d elements · %d dof · Jacobi-PCG", ct.ne, 2 * ct.nn);
    } else {
        snprintf(act, n, "ACT III  ·  TOPOLOGY OPTIMISATION");
        snprintf(title, n, "%s", top.kind == T_MBB ? "MBB BEAM" : "DECK BRIDGE");
        snprintf(sub, n, "SIMP p = 3 · sensitivity filter r = %.1f · OC · multigrid-CG · V = %.0f%%", top.rmin, 100 * top.volfrac);
    }
}

/* a few words on what's being shown */
static int scene_desc(const char **d) {
    if (cur.type == S_BRIDGE) {
        static const char *a[] = {"Trains cross at speed; axle loads move along the deck.",
                                  "Members coloured by axial stress (dead + live), live-load",
                                  "deflection exaggerated; the deck rings after each train."};
        static const char *c[] = {"Trains cross at speed; axle loads move along the deck.",
                                  "Cables and hangers are tension-only (slack: dashed).",
                                  "Live-load deflection exaggerated; the deck rings after."};
        const char **x = br.kind >= B_ARCH ? c : a;
        for (int i = 0; i < 3; i++) d[i] = x[i];
        return 3;
    }
    if (cur.type == S_CONT) {
        static const char *a[][3] = {
            {"A strip in tension with a central hole, load cycling.", "The concentration at the hole against Kirsch's", "solution and Heywood's finite-width fit."},
            {"Clamped at the root, a shear load at the tip.", "Von Mises fringes, principal stress trajectories;", "tip deflection checked against PL³/3EI."},
            {"Bolted at the foot, a load swinging at the arm's end:", "two unit cases superposed. The re-entrant corner", "is a singularity, its peak grows with refinement."}};
        for (int i = 0; i < 3; i++) d[i] = a[ct.kind][i];
        return 3;
    }
    static const char *a[][3] = {
        {"Minimum compliance at half the volume: the MBB beam,", "half modelled. Densities start uniform grey;", "material migrates to where the load path runs."},
        {"A deck under uniform load, pinned at both abutments,", "30% of the material. The optimiser finds an arch", "and its hangers: a bridge, from first principles."}};
    for (int i = 0; i < 3; i++) d[i] = a[top.kind][i];
    return 3;
}

static float fit_px(int face, float px, const char *s, float maxw) {
    float w = text_w(face, px, s, 0);
    return w > maxw ? px * maxw / w : px;
}

static void draw_title_box(void) {
    Box *b = &box[BX_TITLE];
    if (!b->on) return;
    char act[128], title[128], sub[160];
    scene_titles(act, title, sub, sizeof act);
    float u = b->u, y = panel(b, NULL);
    float ink[4], dim[4], ac[4];
    col_a(ink, P->ink, 1); col_a(dim, P->dim, 1); col_a(ac, P->accent, 1);
    float mw = b->w - 28 * u;
    y = b->y + 16 * u;
    float pa = fit_px(F_SEMI, 12 * u, act, mw);
    y += cap_h(F_SEMI, pa);
    text(F_SEMI, pa, b->x + 14 * u, y, -1, act, ac, 0.1f);
    float pt = fit_px(F_SEMI, fminf(34 * u, (b->h - 40 * u) * 0.42f), title, mw);
    y += 12 * u + cap_h(F_SEMI, pt);
    text(F_SEMI, pt, b->x + 14 * u, y, -1, title, ink, 0.03f);
    float ps = fit_px(F_SANS, 13 * u, sub, mw);
    y += 10 * u + cap_h(F_SANS, ps) + 2 * u;
    if (y < b->y + b->h - 6 * u) text(F_SANS, ps, b->x + 14 * u, y, -1, sub, dim, 0.01f);
    const char *d[4];
    int nd = scene_desc(d);
    float dp = 13 * u;
    for (int i = 0; i < nd; i++) dp = fminf(dp, fit_px(F_SANS, 13 * u, d[i], mw));
    float lh = dp * 1.45f, room = b->y + b->h - 10 * u - y;
    if (room > lh * nd + 6 * u) {
        float c[4];
        col_a(c, P->ink, 0.72f);
        y += 10 * u;
        for (int i = 0; i < nd; i++) { y += lh; text(F_SANS, dp, b->x + 14 * u, y, -1, d[i], c, 0); }
    }
}

static int scene_eqns(const char **e, int max) {
    int n = 0;
#define EQ(s) do { if (n < max) e[n++] = s; } while (0)
    if (cur.type == S_BRIDGE) {
        EQ("\\b{M}\\dd{\\b{u}} + \\b{C}\\d{\\b{u}} + \\b{K}\\b{u} = \\b{f}(t),   \\b{C} = a_{0}\\b{M} + a_{1}\\b{K}");
        EQ("\\b{u}_{n+1} = \\b{u}_{n} + Δt\\,\\d{\\b{u}}_{n} + Δt^{2}[(\\f{1}{2} - β)\\dd{\\b{u}}_{n} + β\\dd{\\b{u}}_{n+1}],   β = \\f{1}{4}");
        EQ("N = \\f{EA}{L}(\\b{u}_{j} - \\b{u}_{i})⋅\\b{e}_{ij},   N_{\\r{cable}} ≥ 0");
    } else if (cur.type == S_CONT) {
        EQ("∇⋅\\b{σ} + \\b{b} = \\b{0},   \\b{σ} = \\b{D}\\b{ε},   \\b{ε} = \\f{1}{2}(∇\\b{u} + ∇\\b{u}^{\\r{T}})");
        EQ("\\b{K}_{e} = ∫_{Ω_{e}}\\b{B}^{\\r{T}}\\b{D}\\b{B}\\,t\\,\\r{d}Ω,   \\b{K}\\b{u} = \\b{f}");
        if (ct.kind == C_PLATE) EQ("σ_{θθ}(a,θ) = σ_{∞}(1 - 2\\r{cos}\\,2θ),   K_{t} = \\f{σ_{\\r{max}}}{σ_{∞}} = 3");
        else if (ct.kind == C_CANTI) EQ("δ = \\f{PL^{3}}{3EI} + \\f{PL}{κGA},   σ_{\\r{vM}} = \\q{σ_{x}^{2} - σ_{x}σ_{y} + σ_{y}^{2} + 3τ_{xy}^{2}}");
        else EQ("\\b{u} = F(\\r{cos}\\,φ\\,\\b{u}_{x} + \\r{sin}\\,φ\\,\\b{u}_{y}),   σ ∼ r^{λ-1},  λ = 0.545");
    } else {
        EQ("\\r{min}_{\\b{x}}\\; c = \\b{U}^{\\r{T}}\\b{K}\\b{U} = Σ_{e}(E_{\\r{min}} + x_{e}^{p}(E_{0} - E_{\\r{min}}))\\,\\b{u}_{e}^{\\r{T}}\\b{k}_{0}\\b{u}_{e}");
        EQ("\\r{s.t.}\\;\\; V(\\b{x}) = fV_{0},   \\b{K}\\b{U} = \\b{F},   0 ≤ x_{e} ≤ 1");
        EQ("x_{e} ← x_{e}B_{e}^{1/2},  B_{e} = \\f{-∂c/∂x_{e}}{λ\\,∂V/∂x_{e}},   \\f{∂c}{∂x_{e}} ← \\f{Σ_{i}H_{ei}x_{i}∂c/∂x_{i}}{x_{e}Σ_{i}H_{ei}}");
    }
#undef EQ
    return n;
}

static void draw_eqn_box(void) {
    Box *b = &box[BX_EQN];
    if (!b->on) return;
    float u = b->u, y0 = panel(b, "GOVERNING EQUATIONS");
    const char *e[6];
    int n = scene_eqns(e, 6);
    float ink[4];
    col_a(ink, P->ink, 1);
    /* one size for all, the largest that fits */
    float px = 23 * u, avail_h = b->y + b->h - y0 - 8 * u, mw = b->w - 32 * u;
    for (int i = 0; i < n; i++) {
        float w = math_w(e[i], px, NULL);
        if (w > mw) px *= mw / w;
    }
    MBox mb[6];
    float tot = 0;
    for (int i = 0; i < n; i++) { math_w(e[i], px, &mb[i]); tot += mb[i].asc + mb[i].desc; }
    float gap = (avail_h - tot) / (n + 0.5f);
    if (gap < 4 * u) {
        float f = avail_h / (tot + (n + 0.5f) * 4 * u);
        px *= f;
        tot = 0;
        for (int i = 0; i < n; i++) { math_w(e[i], px, &mb[i]); tot += mb[i].asc + mb[i].desc; }
        gap = (avail_h - tot) / (n + 0.5f);
    }
    gap = fminf(gap, 26 * u);
    float y = y0 + gap * 0.6f;
    for (int i = 0; i < n; i++) {
        y += mb[i].asc;
        math(e[i], px, b->x + 16 * u, y, ink, NULL);
        y += mb[i].desc + gap;
    }
}

/* a plot frame: returns the inner rect; ticks and labels */
typedef struct { float x, y, w, h; double x0, x1, y0, y1; } Plot;
static float PXf(const Plot *p, double x) { return p->x + (float)((x - p->x0) / (p->x1 - p->x0)) * p->w; }
static float PYf(const Plot *p, double y) { return p->y + p->h - (float)((y - p->y0) / (p->y1 - p->y0)) * p->h; }

static void fmt_tick(char *o, size_t n, double v, double step) {
    if (fabs(v) < step * 1e-6) v = 0;
    int dec = step >= 1 ? 0 : (int)ceil(-log10(step) - 1e-9);
    snprintf(o, n, "%.*f", dec, v);
}

static Plot plot_frame(const Box *b, float top, double x0, double x1, double y0, double y1, const char *xl, const char *yl, int logy) {
    float u = b->u;
    Plot p;
    p.x = b->x + 58 * u; p.y = top + 8 * u; p.w = b->x + b->w - 18 * u - p.x; p.h = b->y + b->h - 34 * u - p.y;
    p.x0 = x0; p.x1 = x1; p.y0 = y0; p.y1 = y1;
    float gc[4], dc[4], ic[4];
    col_a(gc, P->dim, 0.18f); col_a(dc, P->dim, 1); col_a(ic, P->ink, 0.6f);
    float lw = fmaxf(0.8f / sc, 0.8f * u);
    double xs = nice_num((x1 - x0) / 5, 0), ys = logy ? 2 : nice_num((y1 - y0) / 4, 0);
    char s[32];
    float fp = 11 * u;
    for (double v = ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) {
        line(PXf(&p, v), p.y, PXf(&p, v), p.y + p.h, lw, gc);
        fmt_tick(s, sizeof s, v, xs);
        text(F_MONO, fp, PXf(&p, v), p.y + p.h + 15 * u, 0, s, dc, 0);
    }
    for (double v = ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) {
        line(p.x, PYf(&p, v), p.x + p.w, PYf(&p, v), lw, fabs(v) < 1e-9 && !logy ? ic : gc);
        if (logy) snprintf(s, sizeof s, "1e%d", (int)lround(v));
        else fmt_tick(s, sizeof s, v, ys);
        text(F_MONO, fp, p.x - 6 * u, PYf(&p, v) + cap_h(F_MONO, fp) * 0.5f, 1, s, dc, 0);
    }
    line(p.x, p.y, p.x, p.y + p.h, lw, ic);
    line(p.x, p.y + p.h, p.x + p.w, p.y + p.h, lw, ic);
    math(xl, 13 * u, p.x + p.w - math_w(xl, 13 * u, NULL), p.y + p.h + 31 * u, dc, NULL);
    math(yl, 13 * u, p.x + 6 * u, p.y + 14 * u, dc, NULL);
    return p;
}

static void polyline_plot(const Plot *p, const double *x, const double *y, int n, float w, const float *c) {
    for (int i = 0; i + 1 < n; i++) {
        double ya = fmax(p->y0, fmin(p->y1, y[i])), yb = fmax(p->y0, fmin(p->y1, y[i + 1]));
        if (x[i + 1] < p->x0 || x[i] > p->x1) continue;
        line(PXf(p, x[i]), PYf(p, ya), PXf(p, x[i + 1]), PYf(p, yb), w, c);
    }
}

static double pbx[4096], pby[4096];

static void draw_plot_box(void) {
    Box *b = &box[BX_PLOT];
    if (!b->on) return;
    float u = b->u, ac[4], ink[4], dim[4];
    col_a(ac, P->accent, 1); col_a(ink, P->ink, 1); col_a(dim, P->dim, 1);
    float lw = fmaxf(1.2f / sc, 1.8f * u);
    if (cur.type == S_BRIDGE) {
        float py0 = panel(b, "MIDSPAN DEFLECTION  (LIVE LOAD)");
        double win = 30, t1 = fmax(ts, win), t0 = t1 - win;
        double R = nice_num(fmax(br.vmax_live, fmax(-br.env_min, br.env_max)) * 1e3 * 1.25, 0);
        Plot p = plot_frame(b, py0, t0, t1, -R, 0.4 * R, "t \\r{(s)}", "δ_{\\r{mid}} \\r{(mm)}", 0);
        int n = 0;
        int i0 = (int)(t0 / br.dt), step = 4;
        for (int i = i0 < 0 ? 0 : i0; i < br.nhist && n < 4096; i += step) {
            pbx[n] = (i + 1) * br.dt;
            pby[n] = br.hist[i] * 1e3;
            n++;
        }
        polyline_plot(&p, pbx, pby, n, lw, ac);
        if (n) disc(PXf(&p, pbx[n - 1]), PYf(&p, fmax(p.y0, fmin(p.y1, pby[n - 1]))), 3.5f * u, ac);
        /* the static deflection under one locomotive, for scale */
        float sc2[4];
        col_a(sc2, P->dim, 0.7f);
        dashed(p.x, PYf(&p, -br.vmax_live * 1e3), p.x + p.w, PYf(&p, -br.vmax_live * 1e3), fmaxf(0.8f / sc, 1.0f * u), 6 * u, 5 * u, sc2);
        text(F_SANS, 11 * u, p.x + p.w - 4 * u, PYf(&p, -br.vmax_live * 1e3) - 5 * u, 1, "static, one locomotive", sc2, 0.01f);
    } else if (cur.type == S_CONT) {
        if (c_solving || ct.kind == C_BRACKET) {
            float py0 = panel(b, "PCG CONVERGENCE");
            double itf = fmax(0, (ts - SOLVE0) / (SOLVE1 - SOLVE0 - 0.6));
            itf = itf > 1 ? 1 : itf;
            int upto = (int)(pow(itf, 1.6) * ct.iters);
            if (!c_solving) upto = ct.iters;
            Plot p = plot_frame(b, py0, 0, ct.iters, -10, 0, "iteration", "‖\\b{r}‖/‖\\b{f}‖", 1);
            int n = 0, step = ct.nres / 1500 + 1;
            for (int i = 0; i < ct.nres && i <= upto && n < 4096; i += step) {
                pbx[n] = i; pby[n] = log10(fmax(ct.res[i], 1e-12)); n++;
            }
            polyline_plot(&p, pbx, pby, n, lw, ac);
            if (n) disc(PXf(&p, pbx[n - 1]), PYf(&p, fmax(-10, pby[n - 1])), 3.5f * u, ac);
            float tc[4];
            col_a(tc, P->dim, 0.7f);
            dashed(p.x, PYf(&p, -9), p.x + p.w, PYf(&p, -9), fmaxf(0.8f / sc, 1.0f * u), 6 * u, 5 * u, tc);
        } else if (ct.kind == C_PLATE) {
            float py0 = panel(b, "LIGAMENT STRESS  vs  KIRSCH");
            Plot p = plot_frame(b, py0, 1, 6, 0.8, 3.2, "y/a", "σ_{xx}/σ_{∞}", 0);
            int n = 0;
            for (double r = 1; r <= 6.001 && n < 4096; r += 0.05) {
                pbx[n] = r; pby[n] = 1 + 0.5 / (r * r) + 1.5 / (r * r * r * r); n++;
            }
            polyline_plot(&p, pbx, pby, n, lw, dim);
            for (int i = 0; i < ct.nn; i++) {
                if (fabs(ct.xy[i * 2]) > 1e-6 || ct.xy[i * 2 + 1] < ct.a - 1e-6) continue;
                double r = ct.xy[i * 2 + 1] / ct.a;
                if (r > 6) continue;
                disc(PXf(&p, r), PYf(&p, ct.sn[0][i * 3]), 3.2f * u, ac);
            }
            text(F_SANS, 12 * u, p.x + p.w - 4 * u, p.y + 14 * u, 1, "— Kirsch    ● FE", dim, 0.01f);
        } else {
            float py0 = panel(b, "DEFLECTION LINE  vs  BEAM THEORY");
            double I = ct.t * pow(ct.h, 3) / 12, P0 = c_lam * ct.load, dt = P0 * pow(ct.len, 3) / (3 * ct.E * I);
            double R = nice_num(ct.check_ref * ct.load * 1.15, 0);
            Plot p = plot_frame(b, py0, 0, ct.len, -R, 0.15 * R, "x \\r{(mm)}", "w \\r{(mm)}", 0);
            int n = 0;
            for (double x = 0; x <= ct.len + 1e-9 && n < 4096; x += ct.len / 200) {
                pbx[n] = x; pby[n] = -P0 * x * x * (3 * ct.len - x) / (6 * ct.E * I); n++;
            }
            polyline_plot(&p, pbx, pby, n, lw, dim);
            for (int i = 0; i < ct.nn; i++) {
                if (fabs(ct.xy[i * 2 + 1] - ct.h / 2) > 1e-6) continue;
                if (((int)lround(ct.xy[i * 2] / ct.gdx)) % 2) continue;
                disc(PXf(&p, ct.xy[i * 2]), PYf(&p, P0 * ct.u[0][i * 2 + 1]), 3.0f * u, ac);
            }
            (void)dt;
            text(F_SANS, 12 * u, p.x + p.w - 4 * u, p.y + 14 * u, 1, "— Euler–Bernoulli    ● FE", dim, 0.01f);
        }
    } else {
        float py0 = panel(b, "COMPLIANCE  &  VOLUME");
        int k = top_shown;
        double cmax = 1e-9, cmin = 1e30;
        for (int i = 0; i < top.nhist - 1; i++) { cmax = fmax(cmax, top.hc[i]); cmin = fmin(cmin, top.hc[i]); }
        if (top.nhist < 2) cmax = 1, cmin = 0;
        double hi = nice_num(cmax * 1.1, 0);
        double xmax = fmax(40, nice_num(fmax(k, top.nhist) * 1.1, 0));
        Plot p = plot_frame(b, py0, 0, xmax, 0, hi, "iteration", "c", 0);
        int n = 0;
        for (int i = 0; i < top.nhist - 1 && i <= k && n < 4096; i++) { pbx[n] = i; pby[n] = top.hc[i]; n++; }
        polyline_plot(&p, pbx, pby, n, lw, ac);
        if (n) disc(PXf(&p, pbx[n - 1]), PYf(&p, pby[n - 1]), 3.5f * u, ac);
        n = 0;
        for (int i = 0; i < top.nhist && i <= k && n < 4096; i++) { pbx[n] = i; pby[n] = top.hvol[i] * hi; n++; }
        polyline_plot(&p, pbx, pby, n, lw * 0.7f, dim);
        text(F_SANS, 12 * u, p.x + p.w - 4 * u, p.y + 14 * u, 1, "— compliance    — volume fraction", dim, 0.01f);
    }
}

static void readout(const Box *b, float *y, float rh, const char *k, const char *v) {
    float u = b->u, dim[4], ink[4];
    col_a(dim, P->dim, 1); col_a(ink, P->ink, 1);
    float px = fminf(rh * 0.6f, 15 * u);
    *y += rh;
    if (*y > b->y + b->h - 4 * u) return;
    math(k, px, b->x + 14 * u, *y, dim, NULL);
    text(F_MONO, px, b->x + b->w - 14 * u, *y, 1, v, ink, 0);
}

static void draw_read_box(void) {
    Box *b = &box[BX_READ];
    if (!b->on) return;
    float u = b->u, y = panel(b, cur.type == S_TOP ? "OPTIMISER" : "READOUTS") - 6 * u;
    char v[96];
    int rows = cur.type == S_BRIDGE ? 7 : 6;
    float rh = fminf(28 * u, (b->y + b->h - y - 10 * u) / rows);
    if (cur.type == S_BRIDGE) {
        const Train *on = NULL;
        double tot = 0;
        for (int k = 0; k < br.ntrain; k++) {
            const Train *T = &br.tr[k];
            if (ts < T->t0) continue;
            double head = T->s0 + T->v * (ts - T->t0), tail = head - (T->v > 0 ? 1 : -1) * T->len;
            if (fmax(head, tail) > (stage[0] - wox) / wk && fmin(head, tail) < (stage[0] + stage[2] - wox) / wk) on = T;
        }
        if (on) {
            for (int i = 0; i < on->naxle; i++) tot += on->P[i];
            snprintf(v, sizeof v, "%s %.0f km/h · %d cars · %.1f MN", on->v > 0 ? "→" : "←", fabs(on->v) * 3.6, on->ncar, tot / 1e6);
        } else snprintf(v, sizeof v, "—");
        readout(b, &y, rh, "\\r{train}", v);
        snprintf(v, sizeof v, "%+.0f MPa", br.smax_now / 1e6);
        readout(b, &y, rh, "σ_{\\r{max}}\\;\\r{(tension)}", v);
        snprintf(v, sizeof v, "%+.0f MPa", br.smin_now / 1e6);
        readout(b, &y, rh, "σ_{\\r{min}}\\;\\r{(compression)}", v);
        double dm = br.nhist ? br.hist[br.nhist - 1] * 1e3 : 0;
        snprintf(v, sizeof v, "%+.1f mm", dm);
        readout(b, &y, rh, "δ_{\\r{mid}}", v);
        snprintf(v, sizeof v, "%+.1f / %+.1f mm", br.env_min * 1e3, br.env_max * 1e3);
        readout(b, &y, rh, "\\r{envelope}", v);
        snprintf(v, sizeof v, "%.2f Hz · ζ 2%%", br.f1);
        readout(b, &y, rh, "f_{1}", v);
        snprintf(v, sizeof v, "%d it · Δt %.0f ms · %d dof", br.cg_last, br.dt * 1e3, br.n);
        readout(b, &y, rh, "\\r{PCG}", v);
    } else if (cur.type == S_CONT) {
        if (c_solving) {
            double itf = fmax(0, (ts - SOLVE0) / (SOLVE1 - SOLVE0 - 0.6));
            itf = itf > 1 ? 1 : itf;
            int it = (int)(pow(itf, 1.6) * ct.iters);
            snprintf(v, sizeof v, "%d / %d", it, ct.iters);
            readout(b, &y, rh, "\\r{CG iteration}", v);
            snprintf(v, sizeof v, "%.2e", ct.res[it < ct.nres ? it : ct.nres - 1]);
            readout(b, &y, rh, "‖\\b{r}‖/‖\\b{f}‖", v);
            snprintf(v, sizeof v, "%d", 2 * ct.nn);
            readout(b, &y, rh, "\\r{dof}", v);
            snprintf(v, sizeof v, "%d", ct.K.nnz);
            readout(b, &y, rh, "\\r{nonzeros in }\\b{K}", v);
            snprintf(v, sizeof v, "%.0f ms", ct.solve_ms);
            readout(b, &y, rh, "\\r{solve time}", v);
        } else if (ct.kind == C_PLATE) {
            snprintf(v, sizeof v, "%.1f MPa", c_lam * ct.load);
            readout(b, &y, rh, "σ_{∞}", v);
            snprintf(v, sizeof v, "%.1f MPa", c_fe_now);
            readout(b, &y, rh, "σ_{\\r{max}}\\;\\r{FE}", v);
            snprintf(v, sizeof v, "%.3f", ct.check_fe);
            readout(b, &y, rh, "K_{t}\\;\\r{FE}", v);
            snprintf(v, sizeof v, "%.3f", ct.check_ref);
            readout(b, &y, rh, "K_{t}\\;\\r{Kirsch (infinite)}", v);
            snprintf(v, sizeof v, "%.3f", ct.check_ref2);
            readout(b, &y, rh, "K_{t}\\;\\r{Heywood},\\; W/D = 5", v);
            snprintf(v, sizeof v, "%+.1f%%", 100 * (ct.check_fe / ct.check_ref2 - 1));
            readout(b, &y, rh, "\\r{FE vs Heywood}", v);
        } else if (ct.kind == C_CANTI) {
            double P0 = c_lam * ct.load;
            snprintf(v, sizeof v, "%.2f kN", P0 / 1e3);
            readout(b, &y, rh, "P", v);
            snprintf(v, sizeof v, "%.3f mm", c_fe_now);
            readout(b, &y, rh, "δ_{\\r{tip}}\\;\\r{FE}", v);
            snprintf(v, sizeof v, "%.3f mm", P0 * ct.check_ref);
            readout(b, &y, rh, "PL^{3}/3EI", v);
            snprintf(v, sizeof v, "%.3f mm", P0 * ct.check_ref2);
            readout(b, &y, rh, "\\r{+ shear}\\; PL/κGA", v);
            snprintf(v, sizeof v, "%+.2f%% / %+.2f%%", 100 * (ct.check_fe / ct.check_ref - 1), 100 * (ct.check_fe / ct.check_ref2 - 1));
            readout(b, &y, rh, "\\r{FE vs EB / Timoshenko}", v);
            double I = ct.t * pow(ct.h, 3) / 12;
            snprintf(v, sizeof v, "%.0f MPa", P0 * ct.len * ct.h / 2 / I);
            readout(b, &y, rh, "Mc/I\\;\\r{at the root}", v);
        } else {
            snprintf(v, sizeof v, "%.0f kN", ct.load / 1e3);
            readout(b, &y, rh, "F", v);
            snprintf(v, sizeof v, "%+.0f°", c_phi * 180 / PI);
            readout(b, &y, rh, "φ", v);
            snprintf(v, sizeof v, "%.0f MPa", c_fe_now);
            readout(b, &y, rh, "σ_{\\r{vM}}\\;\\r{corner}", v);
            double um = 0;
            for (int i = 0; i < ct.nn; i++) {
                double ux, uy;
                cont_u(i, c_lam, c_phi, &ux, &uy);
                um = fmax(um, hypot(ux, uy));
            }
            snprintf(v, sizeof v, "%.3f mm", um);
            readout(b, &y, rh, "|\\b{u}|_{\\r{max}}", v);
            snprintf(v, sizeof v, "%d it · 2 cases", ct.iters);
            readout(b, &y, rh, "\\r{PCG}", v);
        }
    } else {
        int k = top_shown;
        snprintf(v, sizeof v, "%d%s", k, top.done && k >= top.nhist - 1 ? "  converged" : "");
        readout(b, &y, rh, "\\r{iteration}", v);
        snprintf(v, sizeof v, "%.3f", k < top.nhist - 1 ? top.hc[k] : top.c);
        readout(b, &y, rh, "c", v);
        snprintf(v, sizeof v, "%.3f", top.hvol[k]);
        readout(b, &y, rh, "V/V_{0}", v);
        snprintf(v, sizeof v, "%.4f", k ? top.hchg[k] : 1.0);
        readout(b, &y, rh, "\\r{max}\\;|Δx_{e}|", v);
        snprintf(v, sizeof v, "%d it · %d levels", k ? top.hcg[k] : 0, top.mg.nl);
        readout(b, &y, rh, "\\r{MGCG}", v);
        snprintf(v, sizeof v, "%d × %d · %d dof", top.nelx, top.nely, top.n);
        readout(b, &y, rh, "\\r{grid}", v);
    }
}

static void draw_legend_box(void) {
    Box *b = &box[BX_LEGEND];
    if (!b->on) return;
    float u = b->u, dim[4], ink[4];
    col_a(dim, P->dim, 1); col_a(ink, P->ink, 1);
    const char *title = cur.type == S_BRIDGE ? "AXIAL STRESS  (MPa)" :
                        cur.type == S_TOP ? "DENSITY" :
                        c_solving ? "|u|  (ITERATE)" : ct.kind == C_PLATE ? "σxx  (MPa)" : "VON MISES  (MPa)";
    float y0 = panel(b, title);
    float bx = b->x + 16 * u, bw = 16 * u, by0 = y0 + 4 * u, by1 = b->y + b->h - 16 * u;
    if (by1 - by0 < 20 * u) return;
    float white[4] = {1, 1, 1, 1};
    double lo, hi;
    int nt = 5;
    if (cur.type == S_BRIDGE) {
        vput(bx, by1, white, 0, 0, 1e6f, 2); vput(bx + bw, by1, white, 0, 0, 1e6f, 2); vput(bx + bw, by0, white, 0, 0, 1e6f, 3);
        vput(bx, by1, white, 0, 0, 1e6f, 2); vput(bx + bw, by0, white, 0, 0, 1e6f, 3); vput(bx, by0, white, 0, 0, 1e6f, 3);
        lo = -br.smax / 1e6; hi = br.smax / 1e6;
        text(F_SEMI, 11 * u, bx + bw + 52 * u, by0 + 10 * u, -1, "T", ink, 0);
        text(F_SEMI, 11 * u, bx + bw + 52 * u, by1, -1, "C", ink, 0);
    } else if (cur.type == S_CONT) {
        float s0 = c_solving ? 4 : 0;
        vput(bx, by1, white, 0, 0, 1e6f, s0); vput(bx + bw, by1, white, 0, 0, 1e6f, s0); vput(bx + bw, by0, white, 0, 0, 1e6f, s0 + 0.999f);
        vput(bx, by1, white, 0, 0, 1e6f, s0); vput(bx + bw, by0, white, 0, 0, 1e6f, s0 + 0.999f); vput(bx, by0, white, 0, 0, 1e6f, s0 + 0.999f);
        if (c_solving) { lo = 0; hi = 1; }
        else if (ct.kind == C_PLATE) { lo = -0.1 * c_fmax; hi = c_fmax; }
        else { lo = 0; hi = c_fmax; }
    } else {
        float gc[4];
        for (int i = 0; i < 32; i++) {
            float f = (i + 0.5f) / 32;
            col_a(gc, P->solid, f);
            float ya = by1 - (by1 - by0) * i / 32, yb = by1 - (by1 - by0) * (i + 1) / 32;
            rect(bx, yb, bx + bw, ya, gc);
        }
        lo = 0; hi = 1;
    }
    float bc[4];
    col_a(bc, P->dim, 0.6f);
    float lw = fmaxf(0.8f / sc, 0.8f * u);
    line(bx, by0, bx + bw, by0, lw, bc); line(bx + bw, by0, bx + bw, by1, lw, bc);
    line(bx + bw, by1, bx, by1, lw, bc); line(bx, by1, bx, by0, lw, bc);
    char s[32];
    for (int i = 0; i < nt; i++) {
        double v = lo + (hi - lo) * i / (nt - 1);
        float y = by1 - (by1 - by0) * i / (nt - 1);
        line(bx + bw, y, bx + bw + 5 * u, y, lw, bc);
        snprintf(s, sizeof s, hi - lo >= 10 ? "%.0f" : "%.2f", v);
        text(F_MONO, 11 * u, bx + bw + 8 * u, y + cap_h(F_MONO, 11 * u) * 0.5f, -1, s, dim, 0);
    }
}

/* the drawing's title block: where the lock info lives */
static void draw_block_box(double wt) {
    Box *b = &box[BX_BLOCK];
    if (!b->on || !front) return;
    float u = b->u;
    panel(b, NULL);
    float ink[4], dim[4], ac[4], bc[4];
    col_a(ink, P->ink, 1); col_a(dim, P->dim, 1); col_a(ac, P->accent, 1); col_a(bc, P->dim, 0.35f);
    float lw = fmaxf(1.0f / sc, 1.0f * u);
    time_t tt = (time_t)wt;
    struct tm tm;
    localtime_r(&tt, &tm);
    char hm[16], date[32], away[64], up[64], sheet[32], scale[32];
    strftime(hm, sizeof hm, "%H:%M", &tm);
    strftime(date, sizeof date, "%Y-%m-%d", &tm);
    long a = locked_at ? (long)(tt - locked_at) : 0;
    if (a < 0) a = 0;
    if (a < 60) snprintf(away, sizeof away, "a moment");
    else if (a < 3600) snprintf(away, sizeof away, "%ld min", a / 60);
    else snprintf(away, sizeof away, "%ld h %02ld min", a / 3600, a / 60 % 60);
    snprintf(sheet, sizeof sheet, "%d / %d", cur.sheet, cur.nsheet);
    if (cur.type == S_BRIDGE) snprintf(scale, sizeof scale, "δ ×%g", br.scale);
    else if (cur.type == S_CONT) snprintf(scale, sizeof scale, "δ ×%g", c_dscale);
    else snprintf(scale, sizeof scale, "—");
    for (char *p = host; *p; p++) up[p - host] = (char)toupper((unsigned char)*p), up[p - host + 1] = 0;
    /* rows: PROJECT | TIME, AWAY | SHEET, SCALE, DATE */
    float x0 = b->x, x1 = b->x + b->w, y0 = b->y, y1 = b->y + b->h;
    float r1 = y0 + (y1 - y0) * 0.36f, r2 = y0 + (y1 - y0) * 0.74f;
    line(x0, r1, x1, r1, lw, bc);
    line(x0, r2, x1, r2, lw, bc);
    float cx1 = x0 + (x1 - x0) * 0.5f;
    line(cx1, r1, cx1, r2, lw, bc);
    float c3a = x0 + (x1 - x0) / 3, c3b = x0 + 2 * (x1 - x0) / 3;
    line(c3a, r2, c3a, y1, lw, bc);
    line(c3b, r2, c3b, y1, lw, bc);
    float lp = 10 * u, pad = 12 * u;
#define LABEL(s, x, y) text(F_SEMI, lp, (x) + pad, (y) + pad + cap_h(F_SEMI, lp), -1, s, dim, 0.12f)
    LABEL("PROJECT", x0, y0);
    float hp = fit_px(F_SEMI, fminf(40 * u, (r1 - y0) * 0.42f), up, x1 - x0 - 2 * pad);
    text(F_SEMI, hp, x0 + pad, r1 - pad, -1, up, ink, 0.06f);
    LABEL("TIME", x0, r1);
    float tp = fminf(46 * u, (r2 - r1) * 0.5f);
    tp = fit_px(F_MONO, tp, hm, cx1 - x0 - 2 * pad);
    text(F_MONO, tp, x0 + pad, r2 - pad, -1, hm, ink, 0);
    LABEL("AWAY", cx1, r1);
    float ap = fit_px(F_SANS, fminf(26 * u, (r2 - r1) * 0.32f), away, x1 - cx1 - 2 * pad);
    text(F_SANS, ap, cx1 + pad, r2 - pad, -1, away, ac, 0);
    float sp = fminf(15 * u, (y1 - r2) * 0.32f);
    LABEL("SHEET", x0, r2);
    text(F_MONO, fit_px(F_MONO, sp, sheet, c3a - x0 - 2 * pad), x0 + pad, y1 - pad * 0.8f, -1, sheet, ink, 0);
    LABEL("SCALE", c3a, r2);
    text(F_MONO, fit_px(F_MONO, sp, scale, c3b - c3a - 2 * pad), c3a + pad, y1 - pad * 0.8f, -1, scale, ink, 0);
    LABEL("DATE", c3b, r2);
    text(F_MONO, fit_px(F_MONO, sp, date, x1 - c3b - 2 * pad), c3b + pad, y1 - pad * 0.8f, -1, date, ink, 0);
#undef LABEL
}

/* the act's title card, over the middle of the stage as a scene opens */
static void draw_title_card(void) {
    float a = smooth01(0.2f, 0.9f, (float)ts) * (1 - smooth01(4.2f, 5.2f, (float)ts));
    if (a <= 0.01f) return;
    char act[128], title[128], sub[160];
    scene_titles(act, title, sub, sizeof act);
    float u = U;
    float cx = stage[0] + stage[2] * 0.5f, cy = stage[1] + stage[3] * 0.5f;
    float tp = 54 * u, w = fmaxf(text_w(F_SEMI, tp, title, 0.04f), text_w(F_SANS, 16 * u, sub, 0.01f)) + 90 * u;
    float maxw = fminf(stage[2], span_me >= 0 ? box[BX_BLOCK].w * 2.2f : stage[2]) - 20 * u;
    if (w > maxw) { tp *= maxw / w; w = maxw; }
    float h = 190 * u * (tp / (54 * u) * 0.5f + 0.5f);
    float fc[4], bc[4], ac[4], ink[4], dim[4];
    col_a(fc, P->panel, a); fc[3] = fminf(1, P->panel[3] + 0.12f) * a;
    col_a(bc, P->accent, a); col_a(ac, P->accent, a); col_a(ink, P->ink, a); col_a(dim, P->dim, a);
    float x0 = cx - w / 2, y0 = cy - h / 2;
    rect(x0, y0, x0 + w, y0 + h, fc);
    float lw = fmaxf(1.0f / sc, 1.5f * u);
    line(x0, y0, x0 + w, y0, lw, bc); line(x0, y0 + h, x0 + w, y0 + h, lw, bc);
    float slide = (1 - smooth01(0.2f, 1.2f, (float)ts)) * 30 * u;
    text(F_SEMI, 14 * u, cx, y0 + 0.27f * h, 0, act, ac, 0.16f);
    text(F_SEMI, tp, cx + slide, y0 + 0.27f * h + 18 * u + cap_h(F_SEMI, tp), 0, title, ink, 0.04f);
    float sp = fit_px(F_SANS, 16 * u, sub, w - 40 * u);
    text(F_SANS, sp, cx, y0 + h - 0.16f * h, 0, sub, dim, 0.01f);
}

/* ======================================================================
 * GL
 * ====================================================================== */

#define GLSL(...) #__VA_ARGS__

static const char *VS_BG = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(p, 0.0, 1.0);
    });

/* the sheet: a gradient, a vignette, a fine grid with a coarser one over it */
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform vec4 uView;
    uniform vec2 uCanvas;
    uniform vec3 uBg0, uBg1;
    uniform vec4 uGrid;
    uniform float uStep, uSc, uRes;
    out vec4 o;
    float gridline(vec2 cp, float s, float w) {
        vec2 f = abs(fract(cp / s + 0.5) - 0.5) * s * uSc;
        return 1.0 - clamp(min(f.x, f.y) - w + 0.5, 0.0, 1.0);
    }
    void main() {
        vec2 cp = vec2(uView.x + gl_FragCoord.x / uSc, uView.y + (uRes - gl_FragCoord.y) / uSc);
        vec2 q = cp / uCanvas;
        vec3 c = mix(uBg1, uBg0, smoothstep(0.0, 1.0, q.y));
        vec2 d = (q - 0.5) * vec2(uCanvas.x / uCanvas.y, 1.0);
        c *= 1.0 - 0.18 * smoothstep(0.35, 1.1, length(d) / max(1.0, uCanvas.x / uCanvas.y * 0.6));
        float minor = gridline(cp, uStep, 0.5), major = gridline(cp, uStep * 5.0, 0.7);
        c = mix(c, uGrid.rgb, clamp(uGrid.a * minor + uGrid.a * 1.6 * major, 0.0, 1.0));
        /* a whisper of noise against banding */
        float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        o = vec4(c + (n - 0.5) / 255.0, 1.0);
    });

static const char *VS_MAIN = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec4 aCol;
    layout(location = 2) in vec3 aUVW;
    layout(location = 3) in float aS;
    uniform vec4 uView;
    out vec4 vCol;
    out vec3 vUVW;
    out float vS;
    void main() {
        vCol = aCol; vUVW = aUVW; vS = aS;
        vec2 p = (aPos - uView.xy) / uView.zw;
        gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    });

static const char *FS_MAIN = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas, uCmap, uTopo;
    uniform float uSc, uSharp;
    uniform vec2 uAtlasSize;
    in vec4 vCol;
    in vec3 vUVW;
    in float vS;
    out vec4 o;
    vec3 seq(float x) { return texture(uCmap, vec2(clamp(x, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.25)).rgb; }
    vec3 dvg(float x) { return texture(uCmap, vec2(clamp(x, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.75)).rgb; }
    void main() {
        vec4 c = vCol;
        if (vS < -3.5) {
            vec4 t = texture(uTopo, vUVW.xy);
            float d = t.r;
            c.rgb = mix(c.rgb, seq(0.12 + 0.88 * t.g), vUVW.z * smoothstep(0.3, 0.6, d));
            c.a *= clamp((d - 0.5) * uSharp + 0.5, 0.0, 1.0);
        } else if (vS < -2.5) {
            c.a *= texture(uAtlas, vUVW.xy / uAtlasSize).r;
        } else {
            if (vS > -0.5) {
                if (vS < 1.5) {
                    /* a fringe plot: 12 bands, a fine dark line between */
                    float x = clamp(vS, 0.0, 0.9999) * 12.0;
                    c.rgb *= seq((floor(x) + 0.5) / 12.0);
                    float e = abs(x - floor(x + 0.5)) / max(fwidth(x), 1e-4);
                    c.rgb *= mix(0.62, 1.0, clamp(e - 0.2, 0.0, 1.0));
                } else if (vS < 3.5) c.rgb *= dvg(vS - 2.0);
                else c.rgb *= seq(vS - 4.0);
            }
            c.a *= clamp((vUVW.z - length(vUVW.xy)) * uSc + 0.5, 0.0, 1.0);
            if (vS < -1.5) c.a *= mod(gl_FragCoord.x + gl_FragCoord.y, 9.0) < 1.6 ? 1.0 : 0.22;
        }
        o = c;
    });

static GLuint prog_bg, prog_main, vao_bg, vao, vbo, tex_atlas, tex_cmap;
static GLuint msaa_fbo, msaa_rb;
static int msaa_w, msaa_h;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "bridge: shader:\n%s\n", log);
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

#define UL(p, n) glGetUniformLocation(p, n)

static void cmap_upload(void) {
    unsigned char px[256 * 2 * 4];
    for (int i = 0; i < 256; i++) {
        float c[3], d[3];
        seq_rgb(P->seq, i / 255.0f, c);
        div_rgb(i / 255.0f, d);
        for (int k = 0; k < 3; k++) {
            px[i * 4 + k] = (unsigned char)(c[k] * 255 + 0.5f);
            px[(256 + i) * 4 + k] = (unsigned char)(d[k] * 255 + 0.5f);
        }
        px[i * 4 + 3] = px[(256 + i) * 4 + 3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D, tex_cmap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

static void gl_init(void) {
    prog_bg = mk_prog(VS_BG, FS_BG);
    prog_main = mk_prog(VS_MAIN, FS_MAIN);
    glGenVertexArrays(1, &vao_bg);
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(6 * sizeof(float)));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(9 * sizeof(float)));
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray(i);
    GLuint tx[3];
    glGenTextures(3, tx);
    tex_atlas = tx[0]; tex_cmap = tx[1]; tex_topo = tx[2];
    for (int i = 0; i < 3; i++) {
        glBindTexture(GL_TEXTURE_2D, tx[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, i == 0 ? GL_NEAREST : GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, i == 0 ? GL_NEAREST : GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AT_W, AT_H, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    unsigned char z[4] = {0, 0, 0, 255};
    glBindTexture(GL_TEXTURE_2D, tex_topo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, z);
    cmap_upload();
    glGenFramebuffers(1, &msaa_fbo);
    glGenRenderbuffers(1, &msaa_rb);
}

static void msaa_size(void) {
    if (msaa_w == W && msaa_h == H) return;
    msaa_w = W; msaa_h = H;
    glBindRenderbuffer(GL_RENDERBUFFER, msaa_rb);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, opt_msaa, GL_RGBA8, W, H);
    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaa_rb);
}

static int box_visible(const Box *b) {
    return b->on && b->x < view[0] + view[2] && b->x + b->w > view[0] && b->y < view[1] + view[3] && b->y + b->h > view[1];
}

/* everything for one frame into the vertex stream */
static void build_frame(double wt) {
    nvb = 0;
    if (cur.type == S_BRIDGE) draw_bridge();
    else if (cur.type == S_CONT) draw_cont();
    else draw_top();
    /* the fade between sheets, over the model only */
    float f = 1 - smooth01(0, 0.7f, (float)ts) * (1 - smooth01((float)cur.dur - 0.8f, (float)cur.dur, (float)ts));
    if (f > 0.003f) {
        float c[4] = {P->bg0[0], P->bg0[1], P->bg0[2], f * 0.97f};
        rect(view[0], view[1], view[0] + view[2], view[1] + view[3], c);
    }
    if (box_visible(&box[BX_TITLE])) draw_title_box();
    if (box_visible(&box[BX_EQN])) draw_eqn_box();
    if (box_visible(&box[BX_PLOT])) draw_plot_box();
    if (box_visible(&box[BX_READ])) draw_read_box();
    if (box_visible(&box[BX_LEGEND])) draw_legend_box();
    if (box_visible(&box[BX_BLOCK])) draw_block_box(wt);
    draw_title_card();
}

static void render(double wt, GLuint target) {
    layout();
    build_frame(wt);
    msaa_size();
    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_bg);
    glUniform4f(UL(prog_bg, "uView"), view[0], view[1], view[2], view[3]);
    glUniform2f(UL(prog_bg, "uCanvas"), canvas_w, canvas_h);
    glUniform3fv(UL(prog_bg, "uBg0"), 1, P->bg0);
    glUniform3fv(UL(prog_bg, "uBg1"), 1, P->bg1);
    glUniform4fv(UL(prog_bg, "uGrid"), 1, P->grid);
    glUniform1f(UL(prog_bg, "uStep"), roundf(22 * U));
    glUniform1f(UL(prog_bg, "uSc"), sc);
    glUniform1f(UL(prog_bg, "uRes"), (float)H);
    glBindVertexArray(vao_bg);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    if (at_dirty) {
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        int rows = at_y + at_shelf + 2;
        if (rows > AT_H) rows = AT_H;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, AT_W, rows, GL_RED, GL_UNSIGNED_BYTE, atlas);
        at_dirty = 0;
    }
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog_main);
    glUniform4f(UL(prog_main, "uView"), view[0], view[1], view[2], view[3]);
    glUniform1f(UL(prog_main, "uSc"), sc);
    glUniform1f(UL(prog_main, "uSharp"), top_sharp);
    glUniform2f(UL(prog_main, "uAtlasSize"), AT_W, AT_H);
    glUniform1i(UL(prog_main, "uAtlas"), 0);
    glUniform1i(UL(prog_main, "uCmap"), 1);
    glUniform1i(UL(prog_main, "uTopo"), 2);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_cmap);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_topo);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(Vtx) * nvb), vb, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, nvb);
    glDisable(GL_BLEND);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
    glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "bridge");
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
        "usage: bridge [--window | --root | --layer] [--seed N] [--fps N] [--palette cae|blueprint|paper|mono]\n"
        "              [--act 1|2|3] [--model warren|pratt|arch|stayed|suspension|plate|cantilever|bracket|mbb|deck]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--budget MS] [--msaa 0|2|4]\n"
        "              [--check]\n");
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

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void pick_model(const char *m) {
    for (int i = 0; i < NBRIDGE; i++) if (!strcmp(m, bridge_names[i])) { force_type = S_BRIDGE; force_kind = i; }
    for (int i = 0; i < NCONT; i++) if (!strcmp(m, cont_names[i])) { force_type = S_CONT; force_kind = i; }
    for (int i = 0; i < NTOP; i++) if (!strcmp(m, top_names[i])) { force_type = S_TOP; force_kind = i; }
    if (force_type < 0) usage();
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
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-act") && i + 1 < argc) opt_act = atoi(argv[++i]);
        else if (!strcmp(a, "-model") && i + 1 < argc) opt_model = argv[++i];
        else if (!strcmp(a, "-budget") && i + 1 < argc) opt_budget = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-msaa") && i + 1 < argc) opt_msaa = atoi(argv[++i]);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (opt_check) { run_check(); return 0; }
    if (span_me >= span_n) span_me = -1;
    if (opt_fps < 1) opt_fps = 1;
    if (opt_act < 0 || opt_act > 3) usage();
    if (opt_model) pick_model(opt_model);
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_palette) {
        P = NULL;
        for (int i = 0; i < NPAL; i++) if (!strcmp(opt_palette, pals[i].name)) P = &pals[i];
        if (!P) usage();
    } else {
        /* mostly the dark ones */
        uint32_t h = hash3(opt_seed, 99, 5) % 20;
        P = &pals[h < 9 ? 0 : h < 15 ? 1 : h < 18 ? 3 : 2];
    }
    int offscreen = opt_shot || opt_bench > 0;

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    fonts_init();

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
        XStoreName(dpy, win, "Bridge");
        XClassHint chint = {"bridge", "Bridge"};
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
            layout();
            scene_update(opt_at, -1);
            render(wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        /* the scene at --at, then N frames on from there */
        layout();
        scene_update(opt_at, -1);
        render(wall(), fbo);
        glFinish();
        double sum = 0, worst = 0, solve = 0, wsolve = 0;
        double *gpu = malloc(sizeof(double) * opt_bench);
        GLuint q;
        glGenQueries(1, &q);
        double cpu0 = cpu_now();
        for (int i = 0; i < opt_bench; i++) {
            double t = opt_at + i / 60.0;
            double f0 = now();
            scene_update(t, opt_budget);
            double f1 = now();
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(wall(), fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            gpu[i] = ns * 1e-6;
            double ms = (now() - f1) * 1000;
            solve += (f1 - f0) * 1000;
            wsolve = fmax(wsolve, (f1 - f0) * 1000);
            sum += ms;
            if (ms > worst) worst = ms;
        }
        double cpu_frame = (cpu_now() - cpu0) * 1000 / opt_bench;
        double c0 = now();
        for (int i = 0; i < opt_bench; i++) { layout(); build_frame(wall()); }
        double build = (now() - c0) * 1000 / opt_bench;
        printf("bridge: %dx%d, %d frames from t = %.0f s (%s): render avg %.3f ms, worst %.3f ms (CPU frame build %.3f ms "
               "of that, %d vertices); solver avg %.3f ms, worst %.3f ms a frame\n",
               W, H, opt_bench, opt_at,
               cur.type == S_BRIDGE ? bridge_names[cur.kind] : cur.type == S_CONT ? cont_names[cur.kind] : top_names[cur.kind],
               sum / opt_bench, worst, build, nvb, solve / opt_bench, wsolve);
        qsort(gpu, (size_t)opt_bench, sizeof *gpu, cmp_dbl);
        printf("bridge: GPU time (timer query): median %.3f ms, 10th percentile %.3f ms, 90th %.3f ms\n",
               gpu[opt_bench / 2], gpu[opt_bench / 10], gpu[opt_bench * 9 / 10]);
        printf("bridge: process CPU (user + sys, all threads) %.3f ms a frame, solver included\n", cpu_frame);
        /* the solver over a whole scene, live: stepped at 60 fps with the
         * per-frame budget, from the scene's start to its end */
        {
            Scene sc0 = cur;
            have_cur = 0;
            scene_free();
            double t1 = cpu_now(), w0 = now();
            scene_build(&sc0);
            int nf = (int)(sc0.dur * 60);
            for (int i = 0; i < nf; i++) scene_update(sc0.t0 + i / 60.0, opt_budget);
            double c = cpu_now() - t1;
            printf("bridge: the solver over the whole %.0f s scene, live at 60 fps: %.2f s of CPU, %.1f%% of a core "
                   "(%.2f s wall)\n", sc0.dur, c, 100 * c / sc0.dur, now() - w0);
        }
        printf("bridge: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
        layout();
        scene_update(el, opt_budget);
        render(wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec tsl = {0, (long)(spare * 1e9)};
            nanosleep(&tsl, NULL);
        }
    }
}
