/*
 * kalman — state estimation made visible, on a tactical display.
 *
 * Simulated targets (a car on a winding road, aircraft flying legs and
 * coordinated turns, a ship making way) are watched by noisy sensors: GPS
 * fixes, range-bearing radars with a rotating beam, missed detections and
 * clutter, and a passive sensor that hears only bearings. Trackers written
 * here from scratch estimate where the targets are, and every part of the
 * estimate is drawn: truth paths (faint), raw detections with their
 * uncertainty wedges, each track with its 1-sigma / 2-sigma covariance
 * ellipses (growing while predicted, shrinking on update), gates, track IDs
 * and velocity leaders, and around the plot the filter's own diagnostics: an
 * NIS strip chart against its chi-square bounds, position error vs time, the
 * covariance P and gain K as live heatmaps.
 *
 * The acts cycle, each opened by a title card with its key equations:
 *   1  KF vs raw GPS                      linear Kalman filters, CV and CA
 *   2  EKF vs UKF on a turning target     one radar; linearised vs sigma points
 *   3  IMM through manoeuvres             CV + coordinated-turn models, their
 *                                         probabilities switching in the turns
 *   4  particle filter, bearings-only     a cloud of hypotheses collapsing as
 *                                         the listening ship manoeuvres
 *   5  multi-target association in clutter
 *                                         GNN (Hungarian) assignment, gates,
 *                                         M-of-N confirmation, deletion, and
 *                                         crossing targets that may swap
 * The scene at any moment is a pure function of --seed and the time since
 * --epoch, so instances never need to talk.
 *
 * Palettes (--palette): tactical (dark green / cyan), night (red, night
 * ops), paper (light), mono. Lock info: the time and how long the screen
 * has been locked (swaylock's start time, from /proc).
 *
 * Spanning: one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout carries the tactical picture;
 * the others, left to right, take turns at the filter internals (equations,
 * strip charts, matrices) and the sensor view (measurement space, the
 * whitened innovations, a track table). A single output gets the picture
 * and the internals side by side (or stacked, in portrait).
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] renders one frame offscreen,
 * --bench N times N frames, --check runs the filter consistency tests.
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
#include "proto-layer-shell.h"

#define PI 3.14159265358979323846
#define TAU (2 * PI)
#define DEGR (PI / 180)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check;
static float opt_fps = 30;
static const char *opt_font = "DejaVu Sans Mono", *opt_math = "Latin Modern Math";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = -1;              /* --shot: this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;                /* CLOCK_REALTIME seconds the show started at */
static int opt_act = -1, opt_pal = -1;
static double opt_act_len = 90;         /* real seconds per act */

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];
static int span_n, span_me = -1;
enum { R_FRONT, R_INTERNALS, R_SENSOR };
static int role = R_FRONT, spanned;

static void die(const char *what) { fprintf(stderr, "kalman: %s\n", what); exit(1); }

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
static double clampd(double x, double a, double b) { return x < a ? a : x > b ? b : x; }
static float smooth01(float x) { x = clampf(x, 0, 1); return x * x * (3 - 2 * x); }
static double wrap_pi(double a) { a = fmod(a + PI, TAU); if (a < 0) a += TAU; return a - PI; }

/* a counter-based generator per purpose (truth, measurements, filters), so
 * the scenario is a pure function of the seed */
typedef struct { uint64_t s; } Rng;
static uint64_t rn_next(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double rn_u(Rng *r) { return (rn_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static double rn_range(Rng *r, double a, double b) { return a + (b - a) * rn_u(r); }
static double rn_n(Rng *r) {
    double u1 = rn_u(r), u2 = rn_u(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2 * log(u1)) * cos(TAU * u2);
}
static double rn_sign(Rng *r) { return rn_u(r) < 0.5 ? -1 : 1; }
/* Poisson by inversion (small means) */
static int rn_poisson(Rng *r, double lam) {
    double L = exp(-lam), p = 1;
    int k = 0;
    do { k++; p *= rn_u(r); } while (p > L && k < 1000);
    return k - 1;
}

/* ---- small dense linear algebra, row-major doubles ---- */
#define MX 6
static void mm(const double *A, const double *B, double *C, int n, int m, int p) {
    double T[MX * MX] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s += A[i * m + k] * B[k * p + j];
            T[i * p + j] = s;
        }
    memcpy(C, T, sizeof(double) * n * p);
}
/* C (n x p) = A (n x m) * B^T, B is p x m */
static void mmT(const double *A, const double *B, double *C, int n, int m, int p) {
    double T[MX * MX] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s += A[i * m + k] * B[j * m + k];
            T[i * p + j] = s;
        }
    memcpy(C, T, sizeof(double) * n * p);
}
static void symm(double *P, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) P[i * n + j] = P[j * n + i] = 0.5 * (P[i * n + j] + P[j * n + i]);
}
/* Gauss-Jordan with partial pivoting; returns 0 if singular */
static int inv(const double *A, double *Ai, int n) {
    double M[MX * 2 * MX];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 2 * n; j++) M[i * 2 * n + j] = j < n ? A[i * n + j] : (j - n == i);
    for (int c = 0; c < n; c++) {
        int p = c;
        for (int r = c + 1; r < n; r++) if (fabs(M[r * 2 * n + c]) > fabs(M[p * 2 * n + c])) p = r;
        if (fabs(M[p * 2 * n + c]) < 1e-300) return 0;
        if (p != c)
            for (int j = 0; j < 2 * n; j++) { double t = M[c * 2 * n + j]; M[c * 2 * n + j] = M[p * 2 * n + j]; M[p * 2 * n + j] = t; }
        double d = 1 / M[c * 2 * n + c];
        for (int j = 0; j < 2 * n; j++) M[c * 2 * n + j] *= d;
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            double f = M[r * 2 * n + c];
            if (f != 0) for (int j = 0; j < 2 * n; j++) M[r * 2 * n + j] -= f * M[c * 2 * n + j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) Ai[i * n + j] = M[i * 2 * n + n + j];
    return 1;
}
/* lower Cholesky factor; non-positive pivots are floored */
static void chol(const double *A, double *L, int n) {
    memset(L, 0, sizeof(double) * n * n);
    for (int j = 0; j < n; j++) {
        double s = A[j * n + j];
        for (int k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
        L[j * n + j] = s > 1e-12 ? sqrt(s) : 1e-6;
        for (int i = j + 1; i < n; i++) {
            double t = A[i * n + j];
            for (int k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = t / L[j * n + j];
        }
    }
}
static double det2(const double *S) { return S[0] * S[3] - S[1] * S[2]; }

/* ---- chi-square quantiles: exact for 1 and 2 degrees of freedom,
 * Wilson-Hilferty beyond ---- */
static double norm_inv(double p) {             /* Acklam's rational approximation */
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    if (p < 0.02425) {
        double q = sqrt(-2 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - 0.02425) return -norm_inv(1 - p);
    double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
static double chi2inv(double p, int k) {
    if (k == 1) { double z = norm_inv(0.5 + p / 2); return z * z; }
    if (k == 2) return -2 * log(1 - p);
    double z = norm_inv(p), h = 2.0 / (9 * k), c = 1 - h + z * sqrt(h);
    return k * c * c * c;
}

/* ---- motion models ---- */
enum { M_CV, M_CA, M_CV5, M_CT };      /* state x y vx vy [ax ay | omega] */
static int model_n(int m) { return m == M_CV ? 4 : m == M_CA ? 6 : 5; }

static void model_f(int model, double *x, double dt) {
    if (model == M_CA) {
        x[0] += x[2] * dt + 0.5 * x[4] * dt * dt;
        x[1] += x[3] * dt + 0.5 * x[5] * dt * dt;
        x[2] += x[4] * dt;
        x[3] += x[5] * dt;
    } else if (model == M_CT && fabs(x[4]) > 1e-7) {
        double w = x[4], s = sin(w * dt), c = cos(w * dt), vx = x[2], vy = x[3];
        x[0] += s / w * vx - (1 - c) / w * vy;
        x[1] += (1 - c) / w * vx + s / w * vy;
        x[2] = c * vx - s * vy;
        x[3] = s * vx + c * vy;
    } else {
        x[0] += x[2] * dt;
        x[1] += x[3] * dt;
        if (model == M_CV5) x[4] = 0;  /* the IMM's CV model: no turn */
    }
}

static void model_F(int model, const double *x, double dt, double *F) {
    int n = model_n(model);
    memset(F, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++) F[i * n + i] = 1;
    if (model == M_CT) {               /* numerical Jacobian of the turn */
        for (int j = 0; j < n; j++) {
            double a[MX], b[MX], e = j < 2 ? 1 : j < 4 ? 0.01 : 1e-5;
            memcpy(a, x, sizeof a); memcpy(b, x, sizeof b);
            a[j] += e; b[j] -= e;
            model_f(model, a, dt); model_f(model, b, dt);
            for (int i = 0; i < n; i++) F[i * n + j] = (a[i] - b[i]) / (2 * e);
        }
        return;
    }
    F[0 * n + 2] = F[1 * n + 3] = dt;
    if (model == M_CV5) F[24] = 0;
    if (model == M_CA) {
        F[0 * n + 4] = F[1 * n + 5] = 0.5 * dt * dt;
        F[2 * n + 4] = F[3 * n + 5] = dt;
    }
}

/* continuous white-noise acceleration (CV, CT) or jerk (CA) of intensity q;
 * the turn rate random-walks with intensity qw */
static void model_Q(int model, double dt, double q, double qw, double *Q) {
    int n = model_n(model);
    memset(Q, 0, sizeof(double) * n * n);
    double d2 = dt * dt, d3 = d2 * dt;
    for (int a = 0; a < 2; a++) {
        int p = a, v = 2 + a;
        if (model == M_CA) {
            int c = 4 + a;
            double d4 = d3 * dt, d5 = d4 * dt;
            Q[p * n + p] = q * d5 / 20; Q[p * n + v] = Q[v * n + p] = q * d4 / 8; Q[p * n + c] = Q[c * n + p] = q * d3 / 6;
            Q[v * n + v] = q * d3 / 3;  Q[v * n + c] = Q[c * n + v] = q * d2 / 2;  Q[c * n + c] = q * dt;
        } else {
            Q[p * n + p] = q * d3 / 3; Q[p * n + v] = Q[v * n + p] = q * d2 / 2; Q[v * n + v] = q * dt;
        }
    }
    if (n == 5) Q[24] = qw * dt;
}

/* ---- a Kalman-family filter: the state, and what the display needs of its
 * last update ---- */
typedef struct {
    int n, model;
    double q, qw;
    double x[MX], P[MX * MX], t;       /* estimate valid at sim time t */
    double K[MX * 2];                  /* last gain, n x m */
    int m, nupd;
    double nis, lik, nuw[2], S[4];     /* last innovation: NIS, likelihood, whitened, covariance */
    double t_upd, ex, ey, eP[3];       /* easing: prior minus posterior position, prior position cov */
} Filt;

static void f_init(Filt *f, int model, double q, double qw) {
    memset(f, 0, sizeof *f);
    f->model = model; f->n = model_n(model); f->q = q; f->qw = qw;
    f->t_upd = -1e9;
}

static void f_predict(Filt *f, double t) {
    double dt = t - f->t;
    if (dt <= 0) return;
    int n = f->n;
    double F[MX * MX], Q[MX * MX];
    model_F(f->model, f->x, dt, F);
    model_f(f->model, f->x, dt);
    mm(F, f->P, f->P, n, n, n);
    mmT(f->P, F, f->P, n, n, n);
    model_Q(f->model, dt, f->q, f->qw, Q);
    for (int i = 0; i < n * n; i++) f->P[i] += Q[i];
    symm(f->P, n);
    f->t = t;
}

static void f_ease_begin(Filt *f) {
    f->ex = f->x[0]; f->ey = f->x[1];
    f->eP[0] = f->P[0]; f->eP[1] = f->P[1]; f->eP[2] = f->P[f->n + 1];
}
static void f_ease_end(Filt *f) {
    f->ex -= f->x[0]; f->ey -= f->x[1];
    f->t_upd = f->t;
    f->nupd++;
}

/* innovation statistics from S (m x m) and nu */
static void innov_stats(Filt *f, const double *S, const double *nu, int m) {
    double Si[4], L[4];
    inv(S, Si, m);
    f->m = m;
    f->nis = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) f->nis += nu[i] * Si[i * m + j] * nu[j];
    double dS = m == 2 ? det2(S) : S[0];
    f->lik = exp(-0.5 * f->nis) / (pow(TAU, m / 2.0) * sqrt(fmax(dS, 1e-300)));
    chol(S, L, m);
    f->nuw[0] = nu[0] / L[0];
    f->nuw[1] = m == 2 ? (nu[1] - L[2] * f->nuw[0]) / L[3] : 0;
    memcpy(f->S, S, sizeof(double) * m * m);
}

/* the linearised update: z ~ zh + H dx + v, R; component `ang` is an angle */
static void f_update(Filt *f, const double *z, const double *zh, const double *Hm, const double *R, int m, int ang) {
    int n = f->n;
    double PHt[MX * 2], S[4], Si[4], nu[2], K[MX * 2];
    mmT(f->P, Hm, PHt, n, n, m);
    mm(Hm, PHt, S, m, n, m);
    for (int i = 0; i < m * m; i++) S[i] += R[i];
    for (int i = 0; i < m; i++) nu[i] = z[i] - zh[i];
    if (ang >= 0) nu[ang] = wrap_pi(nu[ang]);
    innov_stats(f, S, nu, m);
    inv(S, Si, m);
    mm(PHt, Si, K, n, m, m);
    f_ease_begin(f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) f->x[i] += K[i * m + j] * nu[j];
    /* Joseph form: P = (I - KH) P (I - KH)^T + K R K^T */
    double IKH[MX * MX], T[MX * MX], KR[MX * 2];
    mm(K, Hm, IKH, n, m, n);
    for (int i = 0; i < n * n; i++) IKH[i] = (i % (n + 1) == 0) - IKH[i];
    mm(IKH, f->P, T, n, n, n);
    mmT(T, IKH, f->P, n, n, n);
    mm(K, R, KR, n, m, m);
    mmT(KR, K, T, n, m, n);
    for (int i = 0; i < n * n; i++) f->P[i] += T[i];
    symm(f->P, n);
    memcpy(f->K, K, sizeof(double) * n * m);
    f_ease_end(f);
}

static void f_update_pos(Filt *f, const double *z, const double *R) {
    double Hm[2 * MX] = {0}, zh[2] = {f->x[0], f->x[1]};
    Hm[0] = 1; Hm[f->n + 1] = 1;
    f_update(f, z, zh, Hm, R, 2, -1);
}

static void rb_of(const double *x, double sx, double sy, double *z) {
    double dx = x[0] - sx, dy = x[1] - sy;
    z[0] = sqrt(dx * dx + dy * dy);
    z[1] = atan2(dy, dx);
}

static void ekf_update_rb(Filt *f, const double *z, double sx, double sy, const double *R) {
    double zh[2], Hm[2 * MX] = {0};
    rb_of(f->x, sx, sy, zh);
    double dx = f->x[0] - sx, dy = f->x[1] - sy, r2 = fmax(zh[0] * zh[0], 1);
    Hm[0] = dx / zh[0]; Hm[1] = dy / zh[0];
    Hm[f->n] = -dy / r2; Hm[f->n + 1] = dx / r2;
    f_update(f, z, zh, Hm, R, 2, 1);
}

/* the unscented update (alpha 1, beta 2, kappa 0): sigma points through h */
static void ukf_sigma(const double *x, const double *P, int n, double sig[][MX], double *wm, double *wc) {
    double A[MX * MX], L[MX * MX];
    double lam = 0;                     /* alpha^2 (n + kappa) - n */
    for (int i = 0; i < n * n; i++) A[i] = (n + lam) * P[i];
    chol(A, L, n);
    for (int k = 0; k < 2 * n + 1; k++) {
        memcpy(sig[k], x, sizeof(double) * n);
        if (k == 0) continue;
        int c = (k - 1) % n;
        double s = k <= n ? 1 : -1;
        for (int i = 0; i < n; i++) sig[k][i] += s * L[i * n + c];
    }
    wm[0] = lam / (n + lam); wc[0] = wm[0] + 2;      /* + 1 - alpha^2 + beta */
    for (int k = 1; k < 2 * n + 1; k++) wm[k] = wc[k] = 0.5 / (n + lam);
}

static void ukf_update_rb(Filt *f, const double *z, double sx, double sy, const double *R) {
    int n = f->n, ns = 2 * n + 1;
    double sig[2 * MX + 1][MX], wm[2 * MX + 1], wc[2 * MX + 1], Z[2 * MX + 1][2], zh[2] = {0, 0}, ref[2];
    ukf_sigma(f->x, f->P, n, sig, wm, wc);
    rb_of(f->x, sx, sy, ref);
    for (int k = 0; k < ns; k++) {
        rb_of(sig[k], sx, sy, Z[k]);
        Z[k][1] = ref[1] + wrap_pi(Z[k][1] - ref[1]);
        zh[0] += wm[k] * Z[k][0];
        zh[1] += wm[k] * Z[k][1];
    }
    double S[4] = {R[0], R[1], R[2], R[3]}, Pxz[MX * 2] = {0};
    for (int k = 0; k < ns; k++) {
        double d0 = Z[k][0] - zh[0], d1 = Z[k][1] - zh[1];
        S[0] += wc[k] * d0 * d0; S[1] += wc[k] * d0 * d1; S[3] += wc[k] * d1 * d1;
        for (int i = 0; i < n; i++) {
            double dx = sig[k][i] - f->x[i];
            Pxz[i * 2] += wc[k] * dx * d0;
            Pxz[i * 2 + 1] += wc[k] * dx * d1;
        }
    }
    S[2] = S[1];
    double nu[2] = {z[0] - zh[0], wrap_pi(z[1] - zh[1])}, Si[4], K[MX * 2];
    innov_stats(f, S, nu, 2);
    inv(S, Si, 2);
    mm(Pxz, Si, K, n, 2, 2);
    f_ease_begin(f);
    for (int i = 0; i < n; i++) f->x[i] += K[i * 2] * nu[0] + K[i * 2 + 1] * nu[1];
    double KS[MX * 2], T[MX * MX];
    mm(K, S, KS, n, 2, 2);
    mmT(KS, K, T, n, 2, n);
    for (int i = 0; i < n * n; i++) f->P[i] -= T[i];
    symm(f->P, n);
    memcpy(f->K, K, sizeof(double) * n * 2);
    f_ease_end(f);
}

/* normalised estimation error squared over the first d states */
static double nees(const Filt *f, const double *truth, int d) {
    double Pd[MX * MX], Pi[MX * MX], e[MX], s = 0;
    for (int i = 0; i < d; i++) {
        e[i] = f->x[i] - truth[i];
        for (int j = 0; j < d; j++) Pd[i * d + j] = f->P[i * f->n + j];
    }
    if (!inv(Pd, Pi, d)) return 0;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) s += e[i] * Pi[i * d + j] * e[j];
    return s;
}

/* ---- truth: piecewise legs, each with a closed form ---- */
enum { S_CV, S_TURN, S_ACC };
typedef struct { double t0, x, y, v, h, p; int kind; } Seg;
#define MAXSEG 64
typedef struct { Seg s[MAXSEG]; int n; } Path;

/* o: x y vx vy heading speed */
static void seg_eval(const Seg *g, double t, double *o) {
    double tau = t - g->t0, h = g->h, v = g->v, x = g->x, y = g->y;
    if (g->kind == S_TURN && fabs(g->p) > 1e-9) {
        double w = g->p;
        h = g->h + w * tau;
        x += v / w * (sin(h) - sin(g->h));
        y -= v / w * (cos(h) - cos(g->h));
    } else if (g->kind == S_ACC) {
        double s = v * tau + 0.5 * g->p * tau * tau;
        v += g->p * tau;
        x += cos(h) * s; y += sin(h) * s;
    } else {
        x += cos(h) * v * tau; y += sin(h) * v * tau;
    }
    o[0] = x; o[1] = y; o[2] = v * cos(h); o[3] = v * sin(h); o[4] = h; o[5] = v;
}

static void path_eval(const Path *p, double t, double *o) {
    int i = p->n - 1;
    while (i > 0 && p->s[i].t0 > t) i--;
    seg_eval(&p->s[i], t, o);
}

static void path_start(Path *p, double t0, double x, double y, double v, double h) {
    p->n = 1;
    p->s[0] = (Seg){t0, x, y, v, h, 0, S_CV};
}

/* the open leg becomes `kind` for dur seconds; a CV leg follows */
static void path_add(Path *p, int kind, double dur, double par) {
    if (p->n >= MAXSEG) return;
    Seg *g = &p->s[p->n - 1];
    g->kind = kind; g->p = par;
    double o[6];
    seg_eval(g, g->t0 + dur, o);
    p->s[p->n++] = (Seg){g->t0 + dur, o[0], o[1], o[5], o[4], 0, S_CV};
}


/* ==================================================================== */
/* ---- the acts ---- */
enum { A_KF, A_UKF, A_IMM, A_PF, A_MTT, NACT };
static const char *ACT_NAME[NACT] = {
    "KF vs raw GPS", "EKF vs UKF on a turning target", "IMM through manoeuvres",
    "Particle filter, bearings-only", "Multi-target association in clutter"};
static const char *ACT_LINE[NACT] = {
    "a car on winding roads, one GPS fix a second (\xcf\x83 5 m), filtered with constant-velocity and constant-acceleration models",
    "one radar, range and bearing: the same CV model linearised (EKF) or carried through by sigma points (UKF)",
    "an aircraft flying legs and hard turns; constant-velocity and coordinated-turn models, mixed by their probabilities",
    "a listening ship hears only bearings; 2000 hypotheses of range, course and speed, culled as it manoeuvres",
    "a radar in clutter with missed detections: gates, global-nearest-neighbour assignment, M-of-N track management"};
static const float ACT_TS[NACT] = {1, 3, 3, 15, 3};   /* sim seconds per real second */

/* palette slots */
enum { C_BG, C_PANEL, C_GRID, C_FRAME, C_TEXT, C_DIM, C_TRUTH, C_MEAS, C_CLUT, C_F1, C_F2, C_F3, C_WARN, C_SWEEP, NC };

#define NSER 1536
typedef struct { float t[NSER], v[NSER]; int n, col, style; const char *name; } Series;
enum { ST_LINE, ST_DOTS, ST_STEP, ST_BAND };
static void ser_add(Series *s, double t, double v) {
    if (s->n < NSER) { s->t[s->n] = (float)t; s->v[s->n] = (float)v; s->n++; }
}

typedef struct {
    const char *title, *note;          /* note: the bounds, in words */
    int logy; float y0, y1;
    float lo, hi;                      /* 95% bounds on one sample (dashed) */
    float blo, bhi;                    /* 95% bounds on the moving average (band) */
    int avgn;                          /* moving-average window, samples */
    int ns; Series *s[5];
    int shade, bars;                   /* shade the manoeuvres; leave room for the IMM bars */
} Chart;

#define TRL 400
typedef struct { float x[TRL], y[TRL]; int n, h; } Trail;
static void trail_push(Trail *t, double x, double y) {
    t->x[t->h] = (float)x; t->y[t->h] = (float)y;
    t->h = (t->h + 1) % TRL;
    if (t->n < TRL) t->n++;
}
#define TRAIL_AT(tr, i) (((tr)->h - (tr)->n + (i) + TRL) % TRL)

typedef struct { double t, r, b, x, y; int tgt, used; } Det;
typedef struct { double x, y, T, sr, sb, pd, lam, rmin, rmax, th0; } Radar;

#define MAXT 10
#define MAXQ 256
#define DH 1024
#define NP 2000
#define NBRG 400
#define MAXTRK 192
#define NSECT 12
#define RING 160

typedef struct {
    Filt f;
    Trail tr;
    int id, st, hits, miss, nsc, truth, swaps;
    double t_born, t_dead, t_swap, t_conf;
} Track;
enum { TK_TENT, TK_CONF, TK_DEAD };

static const double PIJ[2][2] = {{0.93, 0.07}, {0.12, 0.88}};
#define GATE 9.21                       /* chi2(2) 99% */

static struct {
    int id; long cyc;
    double TS, s, len_s;
    Rng rt, rm, rf;
    float view[4];                     /* world box to frame: x0 y0 x1 y1 */
    int follow; float follow_w, follow_b, follow_f;     /* camera: window (m), sim s behind / ahead */
    int nt; Path tp[MAXT]; double tb[MAXT], te[MAXT];
    float man[32][2]; int nman;        /* manoeuvre intervals, real seconds */

    Radar rd; int has_radar, scan, sector;
    int q[MAXQ], qn, qi;               /* this scan's detections (indices into dh), by time */
    Det dh[DH]; int dhn;               /* every detection so far (ring) */
    int nfix;

    int nf, finit; Filt f[3]; Trail ft[3]; int fcol[3]; const char *fname[3];
    Filt mo[2]; double mu[2];          /* IMM: CV, CT */

    Path own;                          /* PF: the listening ship */
    double px[NP][4], pw[NP], s_pf, neff; int nres, nbrg;
    double brg_t[NBRG], brg_b[NBRG], brg_ox[NBRG], brg_oy[NBRG], brg_pm[NBRG], brg_ps[NBRG];

    Track tk[MAXTRK]; int ntk, next_id, nswaps, last_trk;
    float cost[6][8]; int cost_r, cost_c, cost_id[6], cost_asg[6]; double cost_t;

    float inn[RING][2]; int inn_n, inn_h, inn_m;      /* whitened innovations */
    float es[3][RING][2]; int es_n, es_h;              /* error space: meas, f0, f1 minus truth */

    Series ser[14]; Chart ch[3];
    double acc_nis[3], acc_nees[3]; int acc_nn[3], acc_ne[3];
} A;

static double rt_of(double s) { return s / A.TS; }      /* sim -> real seconds into the act */

static void ring_inn(double a, double b) {
    A.inn[A.inn_h][0] = (float)a; A.inn[A.inn_h][1] = (float)b;
    A.inn_h = (A.inn_h + 1) % RING;
    if (A.inn_n < RING) A.inn_n++;
}

static int add_det(double t, double r, double b, int tgt) {
    int i = A.dhn % DH;
    A.dh[i] = (Det){t, r, b, A.rd.x + r * cos(b), A.rd.y + r * sin(b), tgt, 0};
    A.dhn++;
    return i;
}

static double pass_time(double b, int n) {
    double f = (A.rd.th0 - b) / TAU;
    f -= floor(f);
    return (n + f) * A.rd.T;
}
static int sector_of(double b) {
    double f = (A.rd.th0 - b) / TAU;
    f -= floor(f);
    int k = (int)(f * NSECT);
    return k < 0 ? 0 : k >= NSECT ? NSECT - 1 : k;
}

static void det_R(const Det *d, double *R) {
    double c = cos(d->b), s = sin(d->b), a = A.rd.sr * A.rd.sr, e = d->r * A.rd.sb;
    e *= e;
    R[0] = c * c * a + s * s * e; R[1] = R[2] = c * s * (a - e); R[3] = s * s * a + c * c * e;
}

/* one sweep of the beam: each target where the beam passes it (if detected),
 * and Poisson clutter uniform over the coverage */
static void radar_scan(int n) {
    Radar *rd = &A.rd;
    A.qn = A.qi = 0;
    for (int i = 0; i < A.nt; i++) {
        double o[6];
        path_eval(&A.tp[i], n * rd->T, o);
        double b = atan2(o[1] - rd->y, o[0] - rd->x), t = pass_time(b, n);
        path_eval(&A.tp[i], t, o);
        b = atan2(o[1] - rd->y, o[0] - rd->x);
        t = pass_time(b, n);
        path_eval(&A.tp[i], t, o);
        double r = hypot(o[0] - rd->x, o[1] - rd->y);
        double u = rn_u(&A.rm), nr = rn_n(&A.rm), nb = rn_n(&A.rm);
        if (t < A.tb[i] || t > A.te[i] || r < rd->rmin || r > rd->rmax || u > rd->pd) continue;
        if (A.qn < MAXQ) A.q[A.qn++] = add_det(t, r + rd->sr * nr, wrap_pi(b + rd->sb * nb), i);
    }
    int nc = rd->lam > 0 ? rn_poisson(&A.rm, rd->lam) : 0;
    for (int c = 0; c < nc && A.qn < MAXQ; c++) {
        double r = sqrt(rn_range(&A.rm, rd->rmin * rd->rmin, rd->rmax * rd->rmax));
        double b = rn_range(&A.rm, -PI, PI);
        A.q[A.qn++] = add_det(pass_time(b, n), r, b, -1);
    }
    for (int i = 1; i < A.qn; i++)            /* by time */
        for (int j = i; j > 0 && A.dh[A.q[j]].t < A.dh[A.q[j - 1]].t; j--) {
            int t = A.q[j]; A.q[j] = A.q[j - 1]; A.q[j - 1] = t;
        }
}

/* manoeuvre intervals of path p, for shading the charts */
static void note_manoeuvres(const Path *p) {
    for (int i = 0; i < p->n - 1 && A.nman < 32; i++)
        if (p->s[i].kind != S_CV) {
            A.man[A.nman][0] = (float)rt_of(p->s[i].t0);
            A.man[A.nman][1] = (float)rt_of(p->s[i + 1].t0);
            A.nman++;
        }
}

/* frame the paths over the act (and the radar) */
static void view_fit(int with_radar, double margin, double minsz) {
    double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (int i = 0; i < A.nt; i++)
        for (double t = A.tb[i]; t <= fmin(A.te[i], A.len_s); t += A.len_s / 200) {
            double o[6];
            path_eval(&A.tp[i], t, o);
            x0 = fmin(x0, o[0]); x1 = fmax(x1, o[0]); y0 = fmin(y0, o[1]); y1 = fmax(y1, o[1]);
        }
    if (with_radar) { x0 = fmin(x0, A.rd.x); x1 = fmax(x1, A.rd.x); y0 = fmin(y0, A.rd.y); y1 = fmax(y1, A.rd.y); }
    double cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, w = fmax(x1 - x0, minsz) * (1 + margin), h = fmax(y1 - y0, minsz) * (1 + margin);
    A.view[0] = (float)(cx - w / 2); A.view[1] = (float)(cy - h / 2);
    A.view[2] = (float)(cx + w / 2); A.view[3] = (float)(cy + h / 2);
}

/* random legs and manoeuvres: CV legs of leg0..leg1 s, turns at w0..w1 deg/s
 * through a0..a1 degrees, and (if acc) speed changes */
static void gen_path(Path *p, double tend, double leg0, double leg1, double w0, double w1, double a0, double a1,
                     int acc, double vmin, double vmax) {
    double t = p->s[0].t0;
    while (t < tend && p->n < MAXSEG - 2) {
        double leg = rn_range(&A.rt, leg0, leg1);
        path_add(p, S_CV, leg, 0);
        t += leg;
        double v = p->s[p->n - 1].v;
        if (!acc || rn_u(&A.rt) < 0.62) {
            double w = rn_range(&A.rt, w0, w1) * DEGR * rn_sign(&A.rt), ang = rn_range(&A.rt, a0, a1) * DEGR;
            path_add(p, S_TURN, ang / fabs(w), w);
            t += ang / fabs(w);
        } else {
            double vt = rn_range(&A.rt, vmin, vmax);
            if (fabs(vt - v) < 3) vt = v > (vmin + vmax) / 2 ? vmin : vmax;
            double a = (vt > v ? 1 : -1) * rn_range(&A.rt, 0.8, 2.0), dur = (vt - v) / a;
            path_add(p, S_ACC, dur, a);
            t += dur;
        }
    }
}

/* does path i stay within r0..r1 of the radar over the act? */
static int path_in_ring(const Path *p, double r0, double r1, double tend) {
    for (double t = 0; t <= tend; t += 2) {
        double o[6];
        path_eval(p, t, o);
        double r = hypot(o[0] - A.rd.x, o[1] - A.rd.y);
        if (r < r0 || r > r1) return 0;
    }
    return 1;
}

static void acc_stats(int k, double nis, double ne) {
    A.acc_nis[k] += nis; A.acc_nn[k]++;
    if (ne >= 0) { A.acc_nees[k] += ne; A.acc_ne[k]++; }
}

static Series *S_(int i, int col, int style, const char *name) {
    Series *s = &A.ser[i];
    s->col = col; s->style = style; s->name = name;
    return s;
}

static void chart(int k, const char *title, const char *note, int logy, float y0, float y1) {
    Chart *c = &A.ch[k];
    memset(c, 0, sizeof *c);
    c->title = title; c->note = note; c->logy = logy; c->y0 = y0; c->y1 = y1;
}
static void chart_bounds(int k, int dof, int avgn) {
    Chart *c = &A.ch[k];
    c->lo = (float)chi2inv(0.025, dof); c->hi = (float)chi2inv(0.975, dof);
    c->avgn = avgn;
    c->blo = (float)(chi2inv(0.025, dof * avgn) / avgn); c->bhi = (float)(chi2inv(0.975, dof * avgn) / avgn);
}
static void chart_ser(int k, Series *s) { Chart *c = &A.ch[k]; if (c->ns < 5) c->s[c->ns++] = s; }

/* ---------------- act 1: KF vs raw GPS ---------------- */
#define GPS_SIG 5.0
static void kf_init(void) {
    Path *p = &A.tp[0];
    path_start(p, 0, 0, 0, rn_range(&A.rt, 10, 16), rn_range(&A.rt, 0, TAU));
    gen_path(p, A.len_s + 20, 5, 12, 6, 16, 30, 120, 1, 6, 26);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    note_manoeuvres(p);
    A.follow = 1; A.follow_w = 260; A.follow_b = 3; A.follow_f = 3;
    A.nf = 2;
    f_init(&A.f[0], M_CV, 3.0, 0); A.fname[0] = "KF-CV"; A.fcol[0] = C_F1;
    f_init(&A.f[1], M_CA, 1.0, 0); A.fname[1] = "KF-CA"; A.fcol[1] = C_F3;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%", 1, 0.01f, 100);
    chart_bounds(0, 2, 10);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "CV"));
    chart_ser(0, S_(1, C_F3, ST_DOTS, "CA"));
    chart(1, "POSITION ERROR", "m, RMS over 10 s", 0, 0, 12);
    chart_ser(1, S_(2, C_MEAS, ST_LINE, "GPS"));
    chart_ser(1, S_(3, C_F1, ST_LINE, "CV"));
    chart_ser(1, S_(4, C_F3, ST_LINE, "CA"));
    chart(2, "SPEED", "m/s", 0, 0, 30);
    chart_ser(2, S_(5, C_TRUTH, ST_LINE, "truth"));
    chart_ser(2, S_(6, C_F1, ST_LINE, "CV"));
    chart_ser(2, S_(7, C_F3, ST_LINE, "CA"));
    A.ch[0].shade = A.ch[1].shade = A.ch[2].shade = 1;
}

/* raw errors kept separately for the running RMS */
static float kf_err[3][16]; static int kf_errn;

static void kf_fix(void) {
    double t = 0.5 + A.nfix++, o[6];
    path_eval(&A.tp[0], t, o);
    double nx = rn_n(&A.rm) * GPS_SIG, ny = rn_n(&A.rm) * GPS_SIG, u = rn_u(&A.rm);
    if (u > 0.93) {                     /* no fix this second */
        for (int k = 0; k < 2; k++) if (A.finit) f_predict(&A.f[k], t);
        return;
    }
    double z[2] = {o[0] + nx, o[1] + ny}, R[4] = {GPS_SIG * GPS_SIG, 0, 0, GPS_SIG * GPS_SIG};
    add_det(t, 0, 0, 0);
    Det *d = &A.dh[(A.dhn - 1) % DH];
    d->x = z[0]; d->y = z[1];
    if (!A.finit) {
        for (int k = 0; k < 2; k++) {
            Filt *f = &A.f[k];
            f->t = t; f->x[0] = z[0]; f->x[1] = z[1];
            for (int i = 0; i < f->n; i++) f->P[i * f->n + i] = i < 2 ? R[0] : i < 4 ? 15 * 15 : 3 * 3;
            f->t_upd = t; f->nupd = 1;
            trail_push(&A.ft[k], z[0], z[1]);
        }
        A.finit = 1;
        return;
    }
    double rt = rt_of(t), ev[3] = {hypot(nx, ny), 0, 0};
    for (int k = 0; k < 2; k++) {
        Filt *f = &A.f[k];
        f_predict(f, t);
        f_update_pos(f, z, R);
        trail_push(&A.ft[k], f->x[0], f->x[1]);
        double tr[4] = {o[0], o[1], o[2], o[3]}, ne = nees(f, tr, 4);
        acc_stats(k, f->nis, ne);
        ser_add(&A.ser[k], rt, f->nis);
        ser_add(&A.ser[6 + k], rt, hypot(f->x[2], f->x[3]));
        ev[1 + k] = hypot(f->x[0] - o[0], f->x[1] - o[1]);
        A.es[1 + k][A.es_h][0] = (float)(f->x[0] - o[0]); A.es[1 + k][A.es_h][1] = (float)(f->x[1] - o[1]);
    }
    A.es[0][A.es_h][0] = (float)nx; A.es[0][A.es_h][1] = (float)ny;
    A.es_h = (A.es_h + 1) % RING;
    if (A.es_n < RING) A.es_n++;
    ring_inn(A.f[0].nuw[0], A.f[0].nuw[1]);
    A.inn_m = 2;
    /* RMS over the last 10 fixes */
    for (int k = 0; k < 3; k++) kf_err[k][kf_errn % 10] = (float)ev[k];
    kf_errn++;
    int n = kf_errn < 10 ? kf_errn : 10;
    for (int k = 0; k < 3; k++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += kf_err[k][i] * kf_err[k][i];
        ser_add(&A.ser[2 + k], rt, sqrt(s / n));
    }
    ser_add(&A.ser[5], rt, o[5]);
}

/* ---------------- acts 2 and 3: one target, one radar ---------------- */
static void radar_setup(double T, double sr, double sb_deg, double pd, double lam, double rmin, double rmax) {
    A.has_radar = 1;
    A.rd = (Radar){0, 0, T, sr, sb_deg * DEGR, pd, lam, rmin, rmax, rn_range(&A.rt, 0, TAU)};
    A.scan = -1;
}

/* an aircraft that stays between r0 and r1 of the radar */
static void aircraft(Path *p, double r0, double r1, double rs0, double rs1, double v0, double v1,
                     double leg0, double leg1, double w0, double w1, double a0, double a1) {
    for (int tries = 0; tries < 200; tries++) {
        double r = rn_range(&A.rt, rs0, rs1), b = rn_range(&A.rt, 0, TAU);
        double h = b + PI + rn_range(&A.rt, -70, 70) * DEGR;
        path_start(p, 0, r * cos(b), r * sin(b), rn_range(&A.rt, v0, v1), h);
        gen_path(p, A.len_s + 10, leg0, leg1, w0, w1, a0, a1, 0, 0, 0);
        if (path_in_ring(p, r0, r1, A.len_s)) return;
    }
}

static void ukf_init(void) {
    radar_setup(4, 20, 3.0, 0.95, 0, 2000, 80000);
    aircraft(&A.tp[0], 6000, 40000, 14000, 26000, 200, 250, 25, 50, 2, 4, 60, 150);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    note_manoeuvres(&A.tp[0]);
    view_fit(1, 0.12, 20000);
    A.follow = 1; A.follow_w = 24000; A.follow_b = 12; A.follow_f = 6;
    A.nf = 2;
    f_init(&A.f[0], M_CV, 150, 0); A.fname[0] = "UKF"; A.fcol[0] = C_F1;
    f_init(&A.f[1], M_CV, 150, 0); A.fname[1] = "EKF"; A.fcol[1] = C_F2;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%", 1, 0.01f, 100);
    chart_bounds(0, 2, 10);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "UKF"));
    chart_ser(0, S_(1, C_F2, ST_DOTS, "EKF"));
    chart(1, "POSITION ERROR", "m", 1, 10, 3000);
    chart_ser(1, S_(2, C_MEAS, ST_DOTS, "raw"));
    chart_ser(1, S_(3, C_F1, ST_LINE, "UKF"));
    chart_ser(1, S_(4, C_F2, ST_LINE, "EKF"));
    chart(2, "NEES", "\xcf\x87\xc2\xb2\xe2\x82\x84 95%", 1, 0.03f, 300);
    chart_bounds(2, 4, 10);
    chart_ser(2, S_(5, C_F1, ST_DOTS, "UKF"));
    chart_ser(2, S_(6, C_F2, ST_DOTS, "EKF"));
    for (int k = 0; k < 3; k++) A.ch[k].shade = 1;
}

static void init_from_det(Filt *f, const Det *d, double sv, double sw) {
    double R[4];
    det_R(d, R);
    f->t = d->t;
    memset(f->x, 0, sizeof f->x);
    memset(f->P, 0, sizeof f->P);
    f->x[0] = d->x; f->x[1] = d->y;
    int n = f->n;
    f->P[0] = R[0]; f->P[1] = f->P[n] = R[1]; f->P[n + 1] = R[3];
    f->P[2 * n + 2] = f->P[3 * n + 3] = sv * sv;
    if (n == 5) f->P[24] = sw * sw;
    f->t_upd = d->t; f->nupd = 1;
}

static void ukf_det(const Det *d) {
    double o[6], z[2] = {d->r, d->b}, R[4] = {A.rd.sr * A.rd.sr, 0, 0, A.rd.sb * A.rd.sb};
    path_eval(&A.tp[0], d->t, o);
    if (!A.finit) {
        for (int k = 0; k < 2; k++) { init_from_det(&A.f[k], d, 150, 0); trail_push(&A.ft[k], d->x, d->y); }
        A.finit = 1;
        return;
    }
    double rt = rt_of(d->t);
    for (int k = 0; k < 2; k++) {
        Filt *f = &A.f[k];
        f_predict(f, d->t);
        if (k == 0) ukf_update_rb(f, z, A.rd.x, A.rd.y, R);
        else ekf_update_rb(f, z, A.rd.x, A.rd.y, R);
        trail_push(&A.ft[k], f->x[0], f->x[1]);
        double tr[4] = {o[0], o[1], o[2], o[3]}, ne = nees(f, tr, 4);
        acc_stats(k, f->nis, ne);
        ser_add(&A.ser[k], rt, f->nis);
        ser_add(&A.ser[3 + k], rt, hypot(f->x[0] - o[0], f->x[1] - o[1]));
        ser_add(&A.ser[5 + k], rt, ne);
    }
    ser_add(&A.ser[2], rt, hypot(d->x - o[0], d->y - o[1]));
    ring_inn(A.f[0].nuw[0], A.f[0].nuw[1]);
    A.inn_m = 2;
}

/* IMM: CV and coordinated turn, both 5-state (x y vx vy omega) */
static void imm_init(void) {
    radar_setup(3, 25, 0.3, 0.95, 0, 2000, 80000);
    aircraft(&A.tp[0], 6000, 55000, 18000, 32000, 180, 240, 18, 40, 4, 8, 70, 180);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    note_manoeuvres(&A.tp[0]);
    view_fit(1, 0.12, 16000);
    A.follow = 1; A.follow_w = 15000; A.follow_b = 10; A.follow_f = 5;
    A.nf = 2;
    f_init(&A.mo[0], M_CV5, 1.0, 1e-6);
    f_init(&A.mo[1], M_CT, 10.0, 6e-3);
    f_init(&A.f[0], M_CT, 10.0, 6e-3); A.fname[0] = "IMM"; A.fcol[0] = C_F1;
    f_init(&A.f[1], M_CV, 30.0, 0); A.fname[1] = "CV-EKF"; A.fcol[1] = C_F2;
    A.mu[0] = 0.8; A.mu[1] = 0.2;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%", 1, 0.01f, 100);
    chart_bounds(0, 2, 10);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "IMM"));
    chart_ser(0, S_(1, C_F2, ST_DOTS, "CV"));
    chart(1, "POSITION ERROR", "m", 1, 3, 3000);
    chart_ser(1, S_(2, C_MEAS, ST_DOTS, "raw"));
    chart_ser(1, S_(3, C_F1, ST_LINE, "IMM"));
    chart_ser(1, S_(4, C_F2, ST_LINE, "CV"));
    chart(2, "MODE PROBABILITY", "\xce\xbc(CT)", 0, 0, 1);
    chart_ser(2, S_(5, C_F3, ST_LINE, "CT"));
    for (int k = 0; k < 3; k++) A.ch[k].shade = 1;
    A.ch[2].bars = 1;
}

/* combine the models into one estimate */
static void imm_combine(const Filt *m, const double *mu, double *x, double *P) {
    memset(x, 0, sizeof(double) * 5);
    memset(P, 0, sizeof(double) * 25);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 5; i++) x[i] += mu[j] * m[j].x[i];
    for (int j = 0; j < 2; j++)
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < 5; b++)
                P[a * 5 + b] += mu[j] * (m[j].P[a * 5 + b] + (m[j].x[a] - x[a]) * (m[j].x[b] - x[b]));
}

static void imm_det(const Det *d) {
    double o[6], z[2] = {d->r, d->b}, R[4] = {A.rd.sr * A.rd.sr, 0, 0, A.rd.sb * A.rd.sb};
    path_eval(&A.tp[0], d->t, o);
    if (!A.finit) {
        for (int j = 0; j < 2; j++) init_from_det(&A.mo[j], d, 200, 0.03);
        init_from_det(&A.f[0], d, 200, 0.03);
        init_from_det(&A.f[1], d, 200, 0);
        for (int k = 0; k < 2; k++) trail_push(&A.ft[k], d->x, d->y);
        A.finit = 1;
        return;
    }
    double rt = rt_of(d->t);
    /* mixing */
    double cb[2], mix[2][2];
    for (int j = 0; j < 2; j++) cb[j] = PIJ[0][j] * A.mu[0] + PIJ[1][j] * A.mu[1];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) mix[i][j] = PIJ[i][j] * A.mu[i] / cb[j];
    Filt m0[2] = {A.mo[0], A.mo[1]};
    for (int j = 0; j < 2; j++) {
        double w[2] = {mix[0][j], mix[1][j]};
        imm_combine(m0, w, A.mo[j].x, A.mo[j].P);
    }
    /* the prior, for the display's easing and the combined innovation */
    for (int j = 0; j < 2; j++) f_predict(&A.mo[j], d->t);
    double xp[5], Pp[25];
    imm_combine(A.mo, cb, xp, Pp);
    for (int j = 0; j < 2; j++) ekf_update_rb(&A.mo[j], z, A.rd.x, A.rd.y, R);
    /* mode probabilities */
    double c = A.mo[0].lik * cb[0] + A.mo[1].lik * cb[1];
    for (int j = 0; j < 2; j++) A.mu[j] = c > 1e-300 ? A.mo[j].lik * cb[j] / c : cb[j];
    for (int j = 0; j < 2; j++) A.mu[j] = clampd(A.mu[j], 1e-4, 1 - 1e-4);
    Filt *f = &A.f[0];
    f->ex = xp[0]; f->ey = xp[1];
    f->eP[0] = Pp[0]; f->eP[1] = Pp[1]; f->eP[2] = Pp[6];
    imm_combine(A.mo, A.mu, f->x, f->P);
    f->t = d->t;
    f_ease_end(f);
    /* the mixture's innovation: z against the prior's predicted measurement */
    {
        double zp[2], Hm[10] = {0}, dx = xp[0] - A.rd.x, dy = xp[1] - A.rd.y, r2;
        rb_of(xp, A.rd.x, A.rd.y, zp);
        r2 = fmax(zp[0] * zp[0], 1);
        Hm[0] = dx / zp[0]; Hm[1] = dy / zp[0]; Hm[5] = -dy / r2; Hm[6] = dx / r2;
        double PHt[10], S[4], nu[2] = {z[0] - zp[0], wrap_pi(z[1] - zp[1])};
        mmT(Pp, Hm, PHt, 5, 5, 2);
        mm(Hm, PHt, S, 2, 5, 2);
        for (int i = 0; i < 4; i++) S[i] += R[i];
        innov_stats(f, S, nu, 2);
        memcpy(f->K, A.mo[1].K, sizeof f->K);
    }
    double tr[5] = {o[0], o[1], o[2], o[3], 0};
    acc_stats(0, f->nis, nees(f, tr, 4));
    ser_add(&A.ser[0], rt, f->nis);
    ser_add(&A.ser[3], rt, hypot(f->x[0] - o[0], f->x[1] - o[1]));
    ser_add(&A.ser[5], rt, A.mu[1]);
    trail_push(&A.ft[0], f->x[0], f->x[1]);
    ring_inn(f->nuw[0], f->nuw[1]);
    A.inn_m = 2;
    /* the lone CV filter, for contrast */
    Filt *g = &A.f[1];
    f_predict(g, d->t);
    ekf_update_rb(g, z, A.rd.x, A.rd.y, R);
    trail_push(&A.ft[1], g->x[0], g->x[1]);
    acc_stats(1, g->nis, nees(g, tr, 4));
    ser_add(&A.ser[1], rt, g->nis);
    ser_add(&A.ser[4], rt, hypot(g->x[0] - o[0], g->x[1] - o[1]));
    ser_add(&A.ser[2], rt, hypot(d->x - o[0], d->y - o[1]));
}

/* ---------------- act 4: particle filter, bearings only ---------------- */
#define BRG_T 4.0
#define BRG_SIG (1.0 * DEGR)
static void pf_init(void) {
    Path *own = &A.own;
    path_start(own, 0, 0, 0, 10, rn_range(&A.rt, 0, TAU));
    double t = 0;
    while (t < A.len_s + 60 && own->n < MAXSEG - 2) {
        double leg = rn_range(&A.rt, 150, 240);
        path_add(own, S_CV, leg, 0);
        double w = 2.0 * DEGR * rn_sign(&A.rt), ang = rn_range(&A.rt, 80, 140) * DEGR;
        path_add(own, S_TURN, ang / fabs(w), w);
        t += leg + ang / fabs(w);
    }
    double r = rn_range(&A.rt, 5000, 9000), b = own->s[0].h + rn_range(&A.rt, -70, 70) * DEGR;
    double crs = b + PI / 2 * rn_sign(&A.rt) + rn_range(&A.rt, -40, 40) * DEGR;
    path_start(&A.tp[0], 0, r * cos(b), r * sin(b), rn_range(&A.rt, 4, 8), crs);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    /* frame both ships' tracks */
    A.nt = 2; A.tp[1] = *own; A.tb[1] = 0; A.te[1] = 1e9;
    view_fit(0, 0.25, 12000);
    A.nt = 1;
    note_manoeuvres(own);
    A.nf = 1;
    f_init(&A.f[0], M_CV, 1e-4, 0); A.fname[0] = "PF"; A.fcol[0] = C_F1;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x81 95%", 1, 0.001f, 100);
    chart_bounds(0, 1, 10);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "PF"));
    chart(1, "POSITION ERROR", "m", 1, 10, 30000);
    chart_ser(1, S_(1, C_F1, ST_LINE, "PF"));
    chart_ser(1, S_(2, C_F2, ST_LINE, "2\xcf\x83"));
    chart(2, "RANGE", "km, estimate \xc2\xb1 2\xcf\x83", 0, 0, 25);
    chart_ser(2, S_(3, C_F1, ST_BAND, ""));
    chart_ser(2, S_(4, C_F1, ST_BAND, ""));
    chart_ser(2, S_(5, C_TRUTH, ST_LINE, "truth"));
    chart_ser(2, S_(6, C_F1, ST_LINE, "PF"));
    A.ch[0].shade = A.ch[1].shade = A.ch[2].shade = 1;
}

static void pf_estimate(void) {
    Filt *f = &A.f[0];
    double m[4] = {0}, P[16] = {0};
    for (int i = 0; i < NP; i++)
        for (int k = 0; k < 4; k++) m[k] += A.pw[i] * A.px[i][k];
    for (int i = 0; i < NP; i++)
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) P[a * 4 + b] += A.pw[i] * (A.px[i][a] - m[a]) * (A.px[i][b] - m[b]);
    memcpy(f->x, m, sizeof m);
    memcpy(f->P, P, sizeof P);
}

static void pf_bearing(void) {
    double t = BRG_T * A.nbrg, oo[6], ot[6];
    path_eval(&A.own, t, oo);
    path_eval(&A.tp[0], t, ot);
    double beta = wrap_pi(atan2(ot[1] - oo[1], ot[0] - oo[0]) + BRG_SIG * rn_n(&A.rm));
    if (A.nbrg < NBRG) {
        A.brg_t[A.nbrg] = t; A.brg_b[A.nbrg] = beta; A.brg_ox[A.nbrg] = oo[0]; A.brg_oy[A.nbrg] = oo[1];
        A.brg_pm[A.nbrg] = beta; A.brg_ps[A.nbrg] = BRG_SIG;
    }
    A.nbrg++;
    Filt *f = &A.f[0];
    double rt = rt_of(t);
    if (A.nbrg == 1) {                  /* hypotheses along the first bearing */
        for (int i = 0; i < NP; i++) {
            double r = exp(rn_range(&A.rf, log(1500.0), log(25000.0))), b = beta + BRG_SIG * rn_n(&A.rf);
            double v = rn_range(&A.rf, 1, 12), c = rn_range(&A.rf, 0, TAU);
            A.px[i][0] = oo[0] + r * cos(b); A.px[i][1] = oo[1] + r * sin(b);
            A.px[i][2] = v * cos(c); A.px[i][3] = v * sin(c);
            A.pw[i] = 1.0 / NP;
        }
        A.s_pf = t;
        pf_estimate();
        f->t = t; f->t_upd = t; f->nupd = 1;
        A.neff = NP;
        return;
    }
    double dt = t - A.s_pf, sq = sqrt(f->q * dt);
    for (int i = 0; i < NP; i++) {
        A.px[i][0] += A.px[i][2] * dt; A.px[i][1] += A.px[i][3] * dt;
        A.px[i][2] += sq * rn_n(&A.rf); A.px[i][3] += sq * rn_n(&A.rf);
    }
    A.s_pf = t;
    /* the prior's predicted bearing, for the NIS */
    double mb = 0, vb = 0;
    for (int i = 0; i < NP; i++) mb += A.pw[i] * wrap_pi(atan2(A.px[i][1] - oo[1], A.px[i][0] - oo[0]) - beta);
    for (int i = 0; i < NP; i++) {
        double e = wrap_pi(atan2(A.px[i][1] - oo[1], A.px[i][0] - oo[0]) - beta) - mb;
        vb += A.pw[i] * e * e;
    }
    if (A.nbrg <= NBRG) { A.brg_pm[A.nbrg - 1] = beta + mb; A.brg_ps[A.nbrg - 1] = sqrt(vb); }
    pf_estimate();
    f->t = t;
    f_ease_begin(f);
    double sw = 0, s2 = 0;
    for (int i = 0; i < NP; i++) {
        double e = wrap_pi(beta - atan2(A.px[i][1] - oo[1], A.px[i][0] - oo[0])) / BRG_SIG;
        A.pw[i] *= exp(-0.5 * e * e) + 1e-300;
        sw += A.pw[i];
    }
    for (int i = 0; i < NP; i++) { A.pw[i] /= sw; s2 += A.pw[i] * A.pw[i]; }
    A.neff = 1 / s2;
    pf_estimate();
    f_ease_end(f);
    if (A.neff < NP / 2.0) {           /* systematic resampling, then roughening */
        static double nx[NP][4];
        double u = rn_u(&A.rf) / NP, c = A.pw[0], L[16];
        int j = 0;
        for (int i = 0; i < NP; i++) {
            double uu = u + (double)i / NP;
            while (uu > c && j < NP - 1) c += A.pw[++j];
            memcpy(nx[i], A.px[j], sizeof nx[i]);
        }
        /* regularised: jitter shaped by the cloud's own covariance (so it
         * spreads along the bearing line, not across it), at half the
         * optimal kernel bandwidth (4 / (N (d + 2)))^(1 / (d + 4)) */
        chol(f->P, L, 4);
        double hk = 0.5 * pow(4.0 / (NP * 6.0), 1.0 / 8);
        for (int i = 0; i < NP; i++) {
            double w[4] = {rn_n(&A.rf), rn_n(&A.rf), rn_n(&A.rf), rn_n(&A.rf)};
            for (int k = 0; k < 4; k++) {
                A.px[i][k] = nx[i][k];
                for (int l = 0; l <= k; l++) A.px[i][k] += hk * L[k * 4 + l] * w[l];
            }
            A.pw[i] = 1.0 / NP;
        }
        A.nres++;
        pf_estimate();
    }
    double nis = mb * mb / (vb + BRG_SIG * BRG_SIG);
    f->nis = nis; f->m = 1; f->nuw[0] = -mb / sqrt(vb + BRG_SIG * BRG_SIG); f->nuw[1] = 0;
    double tr[4] = {ot[0], ot[1], ot[2], ot[3]};
    acc_stats(0, nis, nees(f, tr, 4));
    ser_add(&A.ser[0], rt, nis);
    double err = hypot(f->x[0] - ot[0], f->x[1] - ot[1]);
    ser_add(&A.ser[1], rt, err);
    ser_add(&A.ser[2], rt, 2 * sqrt(f->P[0] + f->P[5]));
    double rr = hypot(f->x[0] - oo[0], f->x[1] - oo[1]);
    /* the range's spread, along the line of sight */
    double ux = (f->x[0] - oo[0]) / rr, uy = (f->x[1] - oo[1]) / rr;
    double sr = sqrt(fmax(ux * ux * f->P[0] + 2 * ux * uy * f->P[1] + uy * uy * f->P[5], 0));
    ser_add(&A.ser[3], rt, fmax(rr - 2 * sr, 0) / 1000);
    ser_add(&A.ser[4], rt, (rr + 2 * sr) / 1000);
    ser_add(&A.ser[5], rt, hypot(ot[0] - oo[0], ot[1] - oo[1]) / 1000);
    ser_add(&A.ser[6], rt, rr / 1000);
    trail_push(&A.ft[0], f->x[0], f->x[1]);
    ring_inn(f->nuw[0], 0);
    A.inn_m = 1;
}

/* ---------------- act 5: many targets in clutter ---------------- */
static void mtt_init(void) {
    radar_setup(4, 40, 0.5, 0.9, 50, 3000, 60000);
    int n = 0;
    /* a pair crossing at a shallow angle, mid-act */
    double cr = rn_range(&A.rt, 18000, 28000), cb = rn_range(&A.rt, 0, TAU), tc = rn_range(&A.rt, 0.45, 0.6) * A.len_s;
    double h = rn_range(&A.rt, 0, TAU), dh = rn_range(&A.rt, 12, 22) * DEGR * rn_sign(&A.rt);
    for (int k = 0; k < 2; k++) {
        double v = rn_range(&A.rt, 190, 235), hh = h + k * dh;
        path_start(&A.tp[n], 0, cr * cos(cb) - v * tc * cos(hh), cr * sin(cb) - v * tc * sin(hh), v, hh);
        A.tb[n] = 0;
        n++;
    }
    for (int k = 0; k < 6; k++, n++) {
        int early = k < 3;
        double tb = early ? 0 : rn_range(&A.rt, 15, A.len_s * 0.6);
        double r = early ? rn_range(&A.rt, 15000, 48000) : rn_range(&A.rt, 52000, 57000), b = rn_range(&A.rt, 0, TAU);
        double hh = b + PI + rn_range(&A.rt, -55, 55) * DEGR;
        path_start(&A.tp[n], tb, r * cos(b), r * sin(b), rn_range(&A.rt, 150, 280), hh);
        if (rn_u(&A.rt) < 0.6) {
            path_add(&A.tp[n], S_CV, rn_range(&A.rt, 30, 100), 0);
            double w = rn_range(&A.rt, 2, 4) * DEGR * rn_sign(&A.rt);
            path_add(&A.tp[n], S_TURN, rn_range(&A.rt, 40, 100) * DEGR / fabs(w), w);
        }
        A.tb[n] = tb;
    }
    A.nt = n;
    for (int i = 0; i < n; i++) {      /* gone once out of cover */
        A.te[i] = 1e9;
        for (double t = A.tb[i] + 1; t < A.len_s + 30; t += 1) {
            double o[6];
            path_eval(&A.tp[i], t, o);
            if (hypot(o[0], o[1]) > A.rd.rmax + 3000) { A.te[i] = t; break; }
        }
    }
    double R = A.rd.rmax * 1.04;
    A.view[0] = (float)-R; A.view[1] = (float)-R; A.view[2] = (float)R; A.view[3] = (float)R;
    A.next_id = 1;
    A.sector = 0;
    radar_scan(0);
    A.scan = 0;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%, confirmed tracks", 1, 0.01f, 100);
    chart_bounds(0, 2, 20);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "tracks"));
    chart(1, "POSITION ERROR", "m, RMS over confirmed tracks", 1, 30, 3000);
    chart_ser(1, S_(1, C_F1, ST_LINE, "RMS"));
    chart(2, "TRACK COUNT", "", 0, 0, 12);
    chart_ser(2, S_(2, C_TRUTH, ST_STEP, "true"));
    chart_ser(2, S_(3, C_F1, ST_STEP, "confirmed"));
    chart_ser(2, S_(4, C_F2, ST_STEP, "tentative"));
}

/* e-maxx Hungarian: rows n <= cols m, cost a (n x m), row -> col */
static void hungarian(int n, int m, const double *a, int *ans) {
    double u[n + 1], v[m + 1], minv[m + 1];
    int p[m + 1], way[m + 1];
    char used[m + 1];
    for (int i = 0; i <= n; i++) u[i] = 0;
    for (int j = 0; j <= m; j++) { v[j] = 0; p[j] = 0; way[j] = 0; }
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        for (int j = 0; j <= m; j++) { minv[j] = 1e18; used[j] = 0; }
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            double delta = 1e18;
            for (int j = 1; j <= m; j++)
                if (!used[j]) {
                    double cur = a[(i0 - 1) * m + j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            for (int j = 0; j <= m; j++)
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    for (int i = 0; i < n; i++) ans[i] = -1;
    for (int j = 1; j <= m; j++) if (p[j]) ans[p[j] - 1] = j - 1;
}

static void trk_pos_at(const Filt *f, double t, double *x, double *y) {
    *x = f->x[0] + f->x[2] * (t - f->t);
    *y = f->x[1] + f->x[3] * (t - f->t);
}

/* squared Mahalanobis distance of detection d from track f (predicted to d's time) */
static double gate_d2(const Filt *f, const Det *d, double *lndS) {
    if (d->t < f->t) return 1e9;
    Filt g = *f;
    f_predict(&g, d->t);
    double R[4], S[4], Si[4];
    det_R(d, R);
    S[0] = g.P[0] + R[0]; S[1] = g.P[1] + R[1]; S[2] = g.P[4] + R[2]; S[3] = g.P[5] + R[3];
    inv(S, Si, 2);
    double nx = d->x - g.x[0], ny = d->y - g.x[1];
    if (lndS) *lndS = log(det2(S));
    return nx * (Si[0] * nx + Si[1] * ny) + ny * (Si[2] * nx + Si[3] * ny);
}

static void trk_update(Track *k, Det *d) {
    double R[4], z[2] = {d->x, d->y};
    det_R(d, R);
    f_predict(&k->f, d->t);
    f_update_pos(&k->f, z, R);
    trail_push(&k->tr, k->f.x[0], k->f.x[1]);
    k->hits++; k->miss = 0; k->nsc++;
    d->used = 1 + (k->st == TK_CONF);
    if (k->st == TK_CONF) {
        A.last_trk = (int)(k - A.tk);
        double rt = rt_of(d->t);
        ser_add(&A.ser[0], rt, k->f.nis);
        ring_inn(k->f.nuw[0], k->f.nuw[1]);
        A.inn_m = 2;
        /* which target is it following? */
        int best = -1;
        double bd = 2500;
        for (int i = 0; i < A.nt; i++) {
            if (d->t < A.tb[i] || d->t > A.te[i]) continue;
            double o[6];
            path_eval(&A.tp[i], d->t, o);
            double e = hypot(o[0] - k->f.x[0], o[1] - k->f.x[1]);
            if (e < bd) { bd = e; best = i; }
        }
        if (best >= 0) {
            if (k->truth >= 0 && best != k->truth) { k->swaps++; k->t_swap = d->t; A.nswaps++; }
            k->truth = best;
            double o[6];
            path_eval(&A.tp[best], d->t, o);
            double tr[4] = {o[0], o[1], o[2], o[3]};
            acc_stats(0, k->f.nis, nees(&k->f, tr, 4));
        } else acc_stats(0, k->f.nis, -1);
    }
}

static void trk_new(const Det *d) {
    if (A.ntk >= MAXTRK) return;
    Track *k = &A.tk[A.ntk++];
    memset(k, 0, sizeof *k);
    f_init(&k->f, M_CV, 25, 0);
    init_from_det(&k->f, d, 160, 0);
    k->st = TK_TENT; k->hits = 1; k->truth = -1;
    k->t_born = d->t; k->t_swap = -1e9;
    trail_push(&k->tr, d->x, d->y);
}

static void mtt_sector(int j) {
    double Ts = A.rd.T / NSECT, p = j * Ts;
    int due_sec = ((j - 2) % NSECT + NSECT) % NSECT;
    int cand[96], nc = 0;
    for (int i = A.dhn - 1; i >= 0 && i >= A.dhn - 2 * MAXQ && nc < 96; i--) {
        Det *d = &A.dh[i % DH];
        if (!d->used && d->t > p - 3 * Ts && d->t <= p) cand[nc++] = i % DH;
    }
    for (int pass = 0; pass < 2; pass++) {
        int due[48], nd = 0;
        for (int i = 0; i < A.ntk && nd < 48; i++) {
            Track *k = &A.tk[i];
            if (k->st != (pass ? TK_TENT : TK_CONF)) continue;
            double x, y;
            trk_pos_at(&k->f, p, &x, &y);
            if (sector_of(atan2(y - A.rd.y, x - A.rd.x)) == due_sec) due[nd++] = i;
        }
        if (!nd) continue;
        int m = nc + nd;
        double *cost = malloc(sizeof(double) * nd * m), d2s[48][96];
        int ans[48];
        for (int r = 0; r < nd; r++) {
            for (int c = 0; c < nc; c++) {
                Det *d = &A.dh[cand[c]];
                double d2 = A.dh[cand[c]].used ? 1e9 : gate_d2(&A.tk[due[r]].f, d, NULL);
                d2s[r][c] = d2;
                cost[r * m + c] = d2 < GATE ? d2 : 1e6;
            }
            for (int c = 0; c < nd; c++) cost[r * m + nc + c] = c == r ? GATE : 1e6;
        }
        hungarian(nd, m, cost, ans);
        free(cost);
        if (pass == 0) {               /* keep the problem for the display */
            int cols[8], ncol = 0;
            for (int c = 0; c < nc && ncol < 8; c++) {
                int any = 0;
                for (int r = 0; r < nd; r++) any |= d2s[r][c] < GATE * 4;
                if (any) cols[ncol++] = c;
            }
            A.cost_r = nd < 6 ? nd : 6; A.cost_c = ncol; A.cost_t = p;
            for (int r = 0; r < A.cost_r; r++) {
                A.cost_id[r] = A.tk[due[r]].id;
                A.cost_asg[r] = -1;
                for (int c = 0; c < ncol; c++) {
                    A.cost[r][c] = (float)d2s[r][cols[c]];
                    if (ans[r] == cols[c]) A.cost_asg[r] = c;
                }
            }
        }
        for (int r = 0; r < nd; r++) {
            Track *k = &A.tk[due[r]];
            int c = ans[r];
            if (c >= 0 && c < nc && d2s[r][c] < GATE && !A.dh[cand[c]].used) trk_update(k, &A.dh[cand[c]]);
            else { k->miss++; k->nsc++; }
            if (k->st == TK_TENT) {
                if (k->hits >= 3) { k->st = TK_CONF; k->id = A.next_id++; k->t_conf = p; }
                else if (k->miss >= 2 || k->nsc >= 5) k->st = -1;
            } else if (k->miss >= 3) { k->st = TK_DEAD; k->t_dead = p; }
            double sp = sqrt(k->f.P[0] + k->f.P[5]);
            if (k->st == TK_CONF && sp > 6000) { k->st = TK_DEAD; k->t_dead = p; }
        }
    }
    /* detections nobody wanted start tentative tracks */
    for (int c = 0; c < nc; c++) {
        Det *d = &A.dh[cand[c]];
        if (!d->used && d->t > p - 3 * Ts && d->t <= p - 2 * Ts) { trk_new(d); d->used = 1; }
    }
    /* compact: drop the failed, and the dead once faded */
    int w = 0;
    for (int i = 0; i < A.ntk; i++) {
        Track *k = &A.tk[i];
        if (k->st < 0 || (k->st == TK_DEAD && p - k->t_dead > 3 * A.TS)) continue;
        if (w != i) A.tk[w] = *k;
        if (A.last_trk == i) A.last_trk = w;
        w++;
    }
    A.ntk = w;
    if (j % NSECT == 0) {              /* once a scan: counts and the RMS error */
        double rt = rt_of(p), se = 0;
        int nconf = 0, ntent = 0, ntrue = 0, ne = 0;
        for (int i = 0; i < A.nt; i++) {
            if (p < A.tb[i] || p > A.te[i]) continue;
            double o[6];
            path_eval(&A.tp[i], p, o);
            double r = hypot(o[0], o[1]);
            ntrue += r >= A.rd.rmin && r <= A.rd.rmax;
        }
        for (int i = 0; i < A.ntk; i++) {
            Track *k = &A.tk[i];
            if (k->st == TK_TENT && k->hits >= 2) ntent++;
            if (k->st != TK_CONF) continue;
            nconf++;
            if (k->truth >= 0) {
                double o[6];
                path_eval(&A.tp[k->truth], k->f.t, o);
                double e = hypot(o[0] - k->f.x[0], o[1] - k->f.x[1]);
                if (e < 3000) { se += e * e; ne++; }
            }
        }
        if (ne) ser_add(&A.ser[1], rt, sqrt(se / ne));
        ser_add(&A.ser[2], rt, ntrue);
        ser_add(&A.ser[3], rt, nconf);
        ser_add(&A.ser[4], rt, ntent);
    }
}

/* ---------------- running an act ---------------- */
static uint64_t mix64(uint64_t a, uint64_t b) {
    Rng r = {a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull)};
    rn_next(&r);
    return rn_next(&r);
}

static void act_setup(long cyc, int id) {
    kf_errn = 0;
    memset(&A, 0, sizeof A);
    A.cyc = cyc; A.id = id;
    A.TS = ACT_TS[id];
    A.len_s = opt_act_len * A.TS;
    uint64_t base = mix64(opt_seed, (uint64_t)cyc * 8 + (uint64_t)id);
    A.rt.s = mix64(base, 1); A.rm.s = mix64(base, 2); A.rf.s = mix64(base, 3);
    A.last_trk = -1;
    switch (id) {
    case A_KF: kf_init(); break;
    case A_UKF: ukf_init(); break;
    case A_IMM: imm_init(); break;
    case A_PF: pf_init(); break;
    default: mtt_init(); break;
    }
}

/* run the act's events up to sim time s */
static void act_advance(double s) {
    for (int guard = 0; guard < 1000000; guard++) {
        if (A.id == A_KF) {
            double t = 0.5 + A.nfix;
            if (t > s) break;
            kf_fix();
        } else if (A.id == A_PF) {
            double t = BRG_T * A.nbrg;
            if (t > s) break;
            pf_bearing();
        } else if (A.id == A_MTT) {
            int j = A.sector + 1;
            double t = j * A.rd.T / NSECT;
            if (t > s) break;
            mtt_sector(j);
            A.sector = j;
            if (j % NSECT == 0) { A.scan = j / NSECT; radar_scan(A.scan); }
        } else {
            double tn = (A.scan + 1) * A.rd.T, td = A.qi < A.qn ? A.dh[A.q[A.qi]].t : 1e18;
            if (fmin(tn, td) > s) break;
            if (tn <= td) { A.scan++; radar_scan(A.scan); }
            else {
                const Det *d = &A.dh[A.q[A.qi++]];
                if (A.id == A_UKF) ukf_det(d); else imm_det(d);
            }
        }
    }
    A.s = s;
}

/* ==================================================================== */
/* ---- palettes ---- */
typedef struct { const char *name; const char *c[NC]; int glow; } Pal;
/*                        bg         panel      grid       frame      text       dim        truth      meas       clutter    f1         f2         f3         warn       sweep */
static const Pal PALS[] = {
    {"tactical", {"#030b09", "#06130f", "#0f3a2d", "#1f5b48", "#93ecc4", "#56a084", "#c8dccf", "#d8f59a", "#3f7f55", "#3fe6e0", "#f0b347", "#d77be0", "#ff5a4a", "#39ff8a"}, 1},
    {"night",    {"#080101", "#100303", "#3c0e08", "#5e1b10", "#ff6e4c", "#a63e27", "#ffb3a0", "#ffcfa4", "#7c2a1a", "#ff4a2e", "#ffa04a", "#ff79a6", "#ffe3a3", "#ff3a20"}, 1},
    {"paper",    {"#f1ede2", "#e9e4d5", "#d3ccb7", "#8a8270", "#2a2620", "#6e675a", "#57524a", "#2e2b26", "#aaa290", "#1f5fa8", "#c0461f", "#2f8a4a", "#c01818", "#1f5fa8"}, 0},
    {"mono",     {"#040404", "#0b0b0b", "#282828", "#4c4c4c", "#e6e6e6", "#8c8c8c", "#bdbdbd", "#dadada", "#5c5c5c", "#ffffff", "#a8a8a8", "#787878", "#ffffff", "#e0e0e0"}, 1},
};
#define NPAL ((int)(sizeof PALS / sizeof PALS[0]))
static float pal[NC][3];
static int glow_on;
static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}
static void set_palette(int i) {
    for (int k = 0; k < NC; k++) hex(PALS[i].c[k], pal[k]);
    glow_on = PALS[i].glow;
}

/* ---- lock info: when swaylock started (gauges' rule) ---- */
static time_t locked_at;
static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}
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
        if (p && sscanf(p + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &start) == 1) {
            time_t t = btime + (time_t)(start / (unsigned long long)hz);
            if (!best || t < best) best = t;
        }
    }
    if (d) closedir(d);
    return best;
}

/* ---- glyphs: a lazily filled atlas keyed by (style, code point) ---- */
enum { F_MONO, F_MONOB, F_MATH, F_SERIFI, F_SANS, NFACE };
static FT_Library ft;
static FT_Face faces[NFACE];
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
#define AW 2048
#define AH 2048
static unsigned char *atlas;
static int ax = 1, ay = 1, ashelf, atlas_dirty;
typedef struct { int face, px; } Style;
#define MAXSTYLE 64
static Style styles[MAXSTYLE];
static int nstyles;
#define GH 8192
static struct { uint32_t key; Glyph g; } gtab[GH];

static FT_Face load_face(const char *pat_s) {
    FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    FT_Face f = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    if (FT_New_Face(ft, (const char *)file, 0, &f)) die("can't load a font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
    return f;
}

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    char b[256];
    snprintf(b, sizeof b, "%s:weight=regular", opt_font); faces[F_MONO] = load_face(b);
    snprintf(b, sizeof b, "%s:weight=bold", opt_font); faces[F_MONOB] = load_face(b);
    faces[F_MATH] = load_face(opt_math);
    faces[F_SERIFI] = load_face("DejaVu Serif:style=Italic");
    faces[F_SANS] = load_face("DejaVu Sans");
    atlas = calloc((size_t)AW * AH, 1);
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 7) p = 7;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) return nstyles - 1;
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    return nstyles++;
}

/* math italic letters back to plain ones (for a fallback face) */
static uint32_t unitalic(uint32_t c) {
    if (c >= 0x1D434 && c < 0x1D44E) return 'A' + (c - 0x1D434);
    if (c >= 0x1D44E && c < 0x1D468) return 'a' + (c - 0x1D44E);
    if (c == 0x210E) return 'h';
    if (c >= 0x1D6FC && c <= 0x1D714) return 0x3B1 + (c - 0x1D6FC);
    return 0;
}

static const Glyph *glyph(int s, uint32_t cp) {
    static const Glyph none = {0, 0, 0, 0, 0, 0, 0};
    uint32_t key = ((uint32_t)(s + 1) << 21) | (cp & 0x1FFFFF);
    uint32_t h = (key * 2654435761u) >> 19;
    for (int n = 0; n < GH; n++, h = (h + 1) & (GH - 1)) {
        if (gtab[h].key == key) return &gtab[h].g;
        if (!gtab[h].key) break;
    }
    if (!gtab[h].key) gtab[h].key = key;
    else return &none;
    Glyph *gl = &gtab[h].g;
    memset(gl, 0, sizeof *gl);
    FT_Face f = faces[styles[s].face];
    uint32_t c = cp;
    if (!FT_Get_Char_Index(f, c)) {
        uint32_t u = unitalic(c);
        if (u) { f = faces[F_SERIFI]; c = u; }
        if (!FT_Get_Char_Index(f, c)) f = faces[F_SANS];
    }
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
    if (FT_Load_Char(f, c, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) return gl;
    FT_GlyphSlot g = f->glyph;
    int w = (int)g->bitmap.width, hh = (int)g->bitmap.rows;
    if (ax + w + 2 > AW) { ax = 1; ay += ashelf + 2; ashelf = 0; }
    if (ay + hh + 2 > AH) { gl->adv = (short)(g->advance.x >> 6); return gl; }
    for (int yy = 0; yy < hh; yy++)
        memcpy(atlas + (size_t)(ay + yy) * AW + ax, g->bitmap.buffer + yy * g->bitmap.pitch, (size_t)w);
    *gl = (Glyph){(short)ax, (short)ay, (short)w, (short)hh, (short)g->bitmap_left, (short)g->bitmap_top,
                  (short)(g->advance.x >> 6)};
    ax += w + 2;
    if (hh > ashelf) ashelf = hh;
    atlas_dirty = 1;
    return gl;
}

static uint32_t u8next(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t c = s[0];
    int n = 0;
    if (c < 0x80) n = 0;
    else if ((c & 0xE0) == 0xC0) { c &= 0x1F; n = 1; }
    else if ((c & 0xF0) == 0xE0) { c &= 0x0F; n = 2; }
    else { c &= 0x07; n = 3; }
    s++;
    for (int i = 0; i < n && (*s & 0xC0) == 0x80; i++) c = (c << 6) | (*s++ & 0x3F);
    *p = (const char *)s;
    return c;
}

/* ---- the vertex stream: x y, local u v, rgba, kind, param ---- */
typedef struct { float x, y, u, v, r, g, b, a, k, p; } Vtx;
static Vtx *vb;
static int vn, vcap;
enum { K_FLAT, K_LINE, K_DISK, K_RING, K_TEXT, K_BLOB, K_SOFT };

static void V(float x, float y, float u, float v, const float *c, float a, int k, float p) {
    if (vn == vcap) vb = realloc(vb, (size_t)(vcap = vcap ? vcap * 2 : 65536) * sizeof *vb);
    vb[vn++] = (Vtx){x, y, u, v, c[0], c[1], c[2], a, (float)k, p};
}
static void quad(float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1, const float *c, float a, int k, float p) {
    V(x0, y0, u0, v0, c, a, k, p); V(x1, y0, u1, v0, c, a, k, p); V(x1, y1, u1, v1, c, a, k, p);
    V(x0, y0, u0, v0, c, a, k, p); V(x1, y1, u1, v1, c, a, k, p); V(x0, y1, u0, v1, c, a, k, p);
}
static void rectf(float x, float y, float w, float h, const float *c, float a) { quad(x, y, x + w, y + h, 0, 0, 0, 0, c, a, K_FLAT, 0); }
static void disk(float x, float y, float r, const float *c, float a) { float e = r + 1; quad(x - e, y - e, x + e, y + e, -e, -e, e, e, c, a, K_DISK, r); }
static void ring(float x, float y, float r, const float *c, float a) { float e = r + 2; quad(x - e, y - e, x + e, y + e, -e, -e, e, e, c, a, K_RING, r); }
static void blob(float x, float y, float r, const float *c, float a) { quad(x - r, y - r, x + r, y + r, -r, -r, r, r, c, a, K_BLOB, r); }

/* an anti-aliased segment w px wide (soft: a glow falloff instead) */
static void seg_k(float x0, float y0, float x1, float y1, float w, const float *c, float a, int k) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1e-4f) return;
    float hw = w * 0.5f, e = hw + 1, nx = -dy / l * e, ny = dx / l * e, tx = dx / l * 0.5f, ty = dy / l * 0.5f;
    x0 -= tx; y0 -= ty; x1 += tx; y1 += ty;
    V(x0 + nx, y0 + ny, e, 0, c, a, k, hw); V(x1 + nx, y1 + ny, e, 0, c, a, k, hw); V(x1 - nx, y1 - ny, -e, 0, c, a, k, hw);
    V(x0 + nx, y0 + ny, e, 0, c, a, k, hw); V(x1 - nx, y1 - ny, -e, 0, c, a, k, hw); V(x0 - nx, y0 - ny, -e, 0, c, a, k, hw);
}
static void seg(float x0, float y0, float x1, float y1, float w, const float *c, float a) { seg_k(x0, y0, x1, y1, w, c, a, K_LINE); }

/* a mitred polyline; al: per-point alpha multipliers or NULL */
static float *poly_o;
static int poly_cap;
static void poly_k(const float *P, int n, int closed, float w, const float *c, float a, const float *al, int k) {
    if (n < 2) return;
    if (n > poly_cap) poly_o = realloc(poly_o, sizeof(float) * 2 * (size_t)(poly_cap = n * 2));
    float hw = w * 0.5f, e = hw + 1;
    for (int i = 0; i < n; i++) {
        int ip = closed ? (i - 1 + n) % n : (i > 0 ? i - 1 : 0), in = closed ? (i + 1) % n : (i < n - 1 ? i + 1 : n - 1);
        float d0x = P[2 * i] - P[2 * ip], d0y = P[2 * i + 1] - P[2 * ip + 1];
        float d1x = P[2 * in] - P[2 * i], d1y = P[2 * in + 1] - P[2 * i + 1];
        float l0 = sqrtf(d0x * d0x + d0y * d0y), l1 = sqrtf(d1x * d1x + d1y * d1y);
        if (l0 < 1e-5f) { d0x = d1x; d0y = d1y; l0 = l1; }
        if (l1 < 1e-5f) { d1x = d0x; d1y = d0y; l1 = l0; }
        if (l0 < 1e-5f) { poly_o[2 * i] = 0; poly_o[2 * i + 1] = e; continue; }
        d0x /= l0; d0y /= l0; d1x /= l1; d1y /= l1;
        float tx = d0x + d1x, ty = d0y + d1y, tl = sqrtf(tx * tx + ty * ty);
        if (tl < 1e-4f) { tx = d1x; ty = d1y; tl = 1; }
        tx /= tl; ty /= tl;
        float nx = -ty, ny = tx, dt = nx * -d1y + ny * d1x;
        float sc = 1 / fmaxf(dt, 0.5f);
        poly_o[2 * i] = nx * e * sc; poly_o[2 * i + 1] = ny * e * sc;
    }
    int ns = closed ? n : n - 1;
    for (int i = 0; i < ns; i++) {
        int j = (i + 1) % n;
        float ai = a * (al ? al[i] : 1), aj = a * (al ? al[j] : 1);
        if (ai <= 0.003f && aj <= 0.003f) continue;
        float ax0 = P[2 * i] + poly_o[2 * i], ay0 = P[2 * i + 1] + poly_o[2 * i + 1];
        float bx0 = P[2 * i] - poly_o[2 * i], by0 = P[2 * i + 1] - poly_o[2 * i + 1];
        float ax1 = P[2 * j] + poly_o[2 * j], ay1 = P[2 * j + 1] + poly_o[2 * j + 1];
        float bx1 = P[2 * j] - poly_o[2 * j], by1 = P[2 * j + 1] - poly_o[2 * j + 1];
        V(ax0, ay0, e, 0, c, ai, k, hw); V(ax1, ay1, e, 0, c, aj, k, hw); V(bx1, by1, -e, 0, c, aj, k, hw);
        V(ax0, ay0, e, 0, c, ai, k, hw); V(bx1, by1, -e, 0, c, aj, k, hw); V(bx0, by0, -e, 0, c, ai, k, hw);
    }
}
static void poly(const float *P, int n, int closed, float w, const float *c, float a, const float *al) {
    poly_k(P, n, closed, w, c, a, al, K_LINE);
}
/* dashes along a polyline: on px drawn, off px skipped */
static void poly_dash(const float *P, int n, int closed, float w, const float *c, float a, float on, float off) {
    float pos = 0, per = on + off;
    int ns = closed ? n : n - 1;
    for (int i = 0; i < ns; i++) {
        int j = (i + 1) % n;
        float x0 = P[2 * i], y0 = P[2 * i + 1], dx = P[2 * j] - x0, dy = P[2 * j + 1] - y0, l = sqrtf(dx * dx + dy * dy);
        if (l < 1e-5f) continue;
        float s = 0;
        while (s < l) {
            float ph = fmodf(pos + s, per), rem;
            if (ph < on) {
                rem = fminf(on - ph, l - s);
                seg(x0 + dx * s / l, y0 + dy * s / l, x0 + dx * (s + rem) / l, y0 + dy * (s + rem) / l, w, c, a);
            } else rem = fminf(per - ph, l - s);
            s += fmaxf(rem, 0.01f);
        }
        pos += l;
    }
}

/* points around an ellipse in screen space, semi-axes a b, rotated by ang */
#define ELN 56
static void ellipse_pts(float cx, float cy, float a, float b, float ang, float *P) {
    float ca = cosf(ang), sa = sinf(ang);
    for (int i = 0; i < ELN; i++) {
        float t = (float)(TAU * i / ELN), x = a * cosf(t), y = b * sinf(t);
        P[2 * i] = cx + ca * x - sa * y;
        P[2 * i + 1] = cy + sa * x + ca * y;
    }
}
static void fan(const float *P, int n, float cx, float cy, const float *c, float a) {
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        V(cx, cy, 0, 0, c, a, K_FLAT, 0); V(P[2 * i], P[2 * i + 1], 0, 0, c, a, K_FLAT, 0); V(P[2 * j], P[2 * j + 1], 0, 0, c, a, K_FLAT, 0);
    }
}

/* ---- text ---- */
static float text_w(int s, const char *str) {
    float w = 0;
    for (const char *p = str; *p;) w += glyph(s, u8next(&p))->adv;
    return w;
}
static void glyph_at(int s, uint32_t cp, float x, float y, const float *c, float a) {
    const Glyph *g = glyph(s, cp);
    if (!g->w) return;
    float x0 = roundf(x) + g->left, y0 = roundf(y) - g->top;
    quad(x0, y0, x0 + g->w, y0 + g->h, g->x, g->y, g->x + g->w, g->y + g->h, c, a, K_TEXT, 0);
}
/* baseline at y; align -1 left, 0 centre, 1 right; returns the width */
static float text(int s, float x, float y, int align, const char *str, const float *c, float a) {
    float w = text_w(s, str), pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    for (const char *p = str; *p;) {
        uint32_t cp = u8next(&p);
        glyph_at(s, cp, pen, y, c, a);
        pen += glyph(s, cp)->adv;
    }
    return w;
}
static float cap_h(int s) { return glyph(s, 'H')->top; }
static float textf(int s, float x, float y, int align, const float *c, float a, const char *fmt, ...) __attribute__((format(printf, 7, 8)));
static float textf(int s, float x, float y, int align, const float *c, float a, const char *fmt, ...) {
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    return text(s, x, y, align, b, c, a);
}
/* word-wrapped; returns the number of lines */
static int text_wrap(int s, float x, float y, float maxw, float lh, const char *str, const float *c, float a, int draw) {
    char line[1200] = "", word[128];
    int lines = 0;
    const char *p = str;
    while (*p) {
        while (*p == ' ') p++;
        int n = 0;
        while (*p && *p != ' ' && n < 127) word[n++] = *p++;
        word[n] = 0;
        if (!n) break;
        char tryl[1200];
        snprintf(tryl, sizeof tryl, "%s%s%s", line, line[0] ? " " : "", word);
        if (line[0] && text_w(s, tryl) > maxw) {
            if (draw) text(s, x, y + lines * lh, -1, line, c, a);
            lines++;
            snprintf(line, sizeof line, "%s", word);
        } else snprintf(line, sizeof line, "%s", tryl);
    }
    if (line[0]) { if (draw) text(s, x, y + lines * lh, -1, line, c, a); lines++; }
    return lines;
}

/* ---- a little math typesetter ----
 * letters and lowercase Greek set in math italic; ^x / ^{..} superscript,
 * _x / _{..} subscript (both may follow one base), \h{..} a hat over,
 * \r{..} upright, \, \; \q spaces, a plain space a medium one */
static int eq_s[2];
static uint32_t math_cp(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return 0x1D434 + (c - 'A');
    if (c == 'h') return 0x210E;
    if (c >= 'a' && c <= 'z') return 0x1D44E + (c - 'a');
    if (c >= 0x3B1 && c <= 0x3C9) return 0x1D6FC + (c - 0x3B1);
    return c;
}
static float eq_seq(const char **pp, float x, float y, int lvl, int up, int draw, const float *c, float a);
static float eq_atom(const char **pp, float x, float y, int lvl, int up, int draw, const float *c, float a) {
    const char *p = *pp;
    int s = eq_s[lvl > 0];
    float em = (float)styles[s].px, w = 0;
    if (*p == '{') {
        p++;
        w = eq_seq(&p, x, y, lvl, up, draw, c, a);
        if (*p == '}') p++;
    } else if (*p == '\\') {
        char cmd = p[1];
        p += 2;
        if (cmd == ',') w = 0.17f * em;
        else if (cmd == ';') w = 0.3f * em;
        else if (cmd == 'q') w = 1.0f * em;
        else if ((cmd == 'h' || cmd == 'r') && *p == '{') {
            p++;
            w = eq_seq(&p, x, y, lvl, cmd == 'r' ? 1 : up, draw, c, a);
            if (*p == '}') p++;
            if (cmd == 'h' && draw) {      /* a hat: a little chevron over the base */
                float cx = x + w * 0.55f, top = y - glyph(s, 'x')->top - em * 0.12f, hw = em * 0.16f, hh = em * 0.11f;
                float lw = fmaxf(em * 0.05f, 0.9f);
                seg(cx - hw, top, cx, top - hh, lw, c, a);
                seg(cx, top - hh, cx + hw, top, lw, c, a);
            }
        }
    } else if (*p == ' ') {
        p++;
        w = 0.22f * em;
    } else {
        uint32_t cp = u8next(&p);
        if (!up) cp = math_cp(cp);
        if (draw) glyph_at(s, cp, x, y, c, a);
        w = glyph(s, cp)->adv;
    }
    *pp = p;
    return w;
}
static float eq_seq(const char **pp, float x, float y, int lvl, int up, int draw, const float *c, float a) {
    const char *p = *pp;
    float x0 = x;
    float em = (float)styles[eq_s[lvl > 0]].px;
    while (*p && *p != '}') {
        if (*p == '^' || *p == '_') {           /* a script with no base */
            p++;
            x += eq_atom(&p, x, y - (p[-1] == '^' ? 0.42f : -0.2f) * em, lvl + 1, up, draw, c, a);
            continue;
        }
        x += eq_atom(&p, x, y, lvl, up, draw, c, a);
        float sw = 0;
        while (*p == '^' || *p == '_') {
            int sup = *p == '^';
            p++;
            float yy = y + (sup ? -0.42f : 0.24f) * em;
            sw = fmaxf(sw, eq_atom(&p, x, yy, lvl + 1, up, draw, c, a));
        }
        x += sw;
    }
    *pp = p;
    return x - x0;
}
static float eq(float x, float y, const char *str, const float *c, float a, int draw) {
    const char *p = str;
    float w = 0;
    while (*p) {
        w += eq_seq(&p, x + w, y, 0, 0, draw, c, a);
        if (*p == '}') p++;
    }
    return w;
}

/* the key equations of each act */
static const char *EQS[NACT][6] = {
    {"\\h{x}_{k|k−1} = F\\h{x}_{k−1} ,\\q P_{k|k−1} = FP_{k−1}F^⊤ + Q",
     "K_k = P_{k|k−1}H^⊤(HP_{k|k−1}H^⊤ + R)^{−1}",
     "\\h{x}_k = \\h{x}_{k|k−1} + K_k(z_k − H\\h{x}_{k|k−1})",
     "P_k = (I − K_kH)P_{k|k−1}",
     "ε_k = ν^⊤_kS^{−1}_kν_k \\; ∼ \\; χ^2_2", NULL},
    {"z = h(x) + v ,\\q h(x) = (‖p − s‖ , \\r{atan2}(Δy, Δx))",
     "\\r{EKF:}\\q H_k = ∂h/∂x |_{\\h{x}_{k|k−1}}",
     "\\r{UKF:}\\q χ_i = \\h{x} ± ((n+λ)P)^{1/2}_i",
     "\\h{z} = ∑_iW_ih(χ_i) ,\\q P_{xz} = ∑_iW_i(χ_i − \\h{x})(h(χ_i) − \\h{z})^⊤",
     "K = P_{xz}S^{−1} ,\\q P_k = P_{k|k−1} − KSK^⊤", NULL},
    {"c_j = ∑_ip_{ij}μ_i ,\\q μ_{i|j} = p_{ij}μ_i / c_j",
     "\\h{x}^{0j} = ∑_iμ_{i|j}\\h{x}^i \\q \\r{(mixing)}",
     "Λ_j = \\r{N}(ν_j; 0, S_j) ,\\q μ_j = Λ_jc_j / ∑_lΛ_lc_l",
     "\\h{x} = ∑_jμ_j\\h{x}^j ,\\q P = ∑_jμ_j(P^j + δ^jδ^{j⊤})",
     "\\r{CT:}\\q \\h{ω}\\, \\r{rides in the state;  CV:}\\; ω = 0", NULL},
    {"x^{(i)}_k ∼ p(x_k | x^{(i)}_{k−1})",
     "β(x) = \\r{atan2}(y − y_o , x − x_o)",
     "w^{(i)}_k ∝ w^{(i)}_{k−1} \\r{exp}(−(β_k − β(x^{(i)}_k))^2 / 2σ^2_β)",
     "N_{\\r{eff}} = 1 / ∑_i(w^{(i)})^2 \\; < \\; N/2 \\q ⇒ \\q \\r{resample}", NULL, NULL},
    {"z = s + r(\\r{cos} θ, \\r{sin} θ) ,\\q R_c = JRJ^⊤",
     "d^2_{ij} = ν^⊤_{ij}S^{−1}_iν_{ij} \\; ≤ \\; γ = 9.21",
     "\\r{min} ∑_{ij}a_{ij}d^2_{ij} ,\\q ∑_ja_{ij} ≤ 1 ,\\; ∑_ia_{ij} ≤ 1",
     "\\r{confirm: 3 hits;  delete: 3 misses in a row}", NULL, NULL},
};

/* ==================================================================== */
/* ---- layout ---- */
typedef struct { float x, y, w, h; } Rect;
enum { P_PLOT, P_TITLE, P_NIS, P_ERR, P_ACT, P_MATS, P_CLOCK, P_MEAS, P_INNOV, P_TABLE };
static struct { int kind; Rect r; } pan[16];
static int npan;
static float U;
static int s_small, s_hud, s_hudb, s_head, s_title, s_desc, s_card, s_cardsub, s_clock, s_clocks;

static void addp(int k, float x, float y, float w, float h) {
    if (npan < 16) { pan[npan].kind = k; pan[npan].r = (Rect){roundf(x), roundf(y), roundf(w), roundf(h)}; npan++; }
}
static void vstack(float x, float y, float w, float h, float g, int n, const int *k, const float *wt) {
    float tw = 0;
    for (int i = 0; i < n; i++) tw += wt[i];
    float avail = h - g * (n - 1);
    for (int i = 0; i < n; i++) {
        float hh = avail * wt[i] / tw;
        addp(k[i], x, y, w, hh);
        y += hh + g;
    }
}

static void layout(void) {
    npan = 0;
    U = clampf(fminf(W, H) / 1000.0f, 0.6f, 2.6f);
    float m = roundf(14 * U), g = roundf(10 * U), asp = (float)W / H;
    s_small = style(F_MONO, 11 * U); s_hud = style(F_MONO, 13 * U); s_hudb = style(F_MONOB, 13 * U);
    s_head = style(F_MONOB, 11.5f * U); s_title = style(F_MONOB, 22 * U); s_desc = style(F_MONO, 12.5f * U);
    s_card = style(F_MONOB, 34 * U); s_cardsub = style(F_MONO, 15 * U);
    s_clock = style(F_MONO, 42 * U); s_clocks = style(F_MONO, 13 * U);
    if (role == R_FRONT && spanned) {
        addp(P_PLOT, m, m, W - 2 * m, H - 2 * m);
        addp(P_CLOCK, W - m - 320 * U, H - m - 100 * U, 300 * U, 88 * U);
    } else if (role == R_FRONT) {
        if (asp >= 0.8f) {
            float pw = fminf(H - 2 * m, W * (asp >= 1.25f ? 0.62f : 0.56f));
            addp(P_PLOT, m, m, pw, H - 2 * m);
            float sx = m + pw + g;
            static const int k[] = {P_TITLE, P_NIS, P_ERR, P_ACT, P_MATS, P_CLOCK};
            static const float wt[] = {2.0f, 1, 1, 1, 1.25f, 0.5f};
            vstack(sx, m, W - sx - m, H - 2 * m, g, 6, k, wt);
        } else {
            float ph = fminf(W - 2 * m, H * 0.62f);
            addp(P_PLOT, m, m, W - 2 * m, ph);
            float y0 = m + ph + g, hh = H - y0 - m, cw = (W - 2 * m - g) / 2;
            static const int kl[] = {P_TITLE, P_MATS, P_CLOCK}, kr[] = {P_NIS, P_ERR, P_ACT};
            static const float wl[] = {2.0f, 1.2f, 0.45f}, wr[] = {1, 1, 1};
            vstack(m, y0, cw, hh, g, 3, kl, wl);
            vstack(m + cw + g, y0, cw, hh, g, 3, kr, wr);
        }
    } else if (role == R_INTERNALS) {
        if (asp < 1) {
            static const int k[] = {P_TITLE, P_NIS, P_ERR, P_ACT, P_MATS, P_CLOCK};
            static const float wt[] = {1.5f, 1, 1, 1, 1.4f, 0.42f};
            vstack(m, m, W - 2 * m, H - 2 * m, g, 6, k, wt);
        } else {
            float cw = (W - 2 * m - g) / 2;
            static const int kl[] = {P_TITLE, P_MATS, P_CLOCK}, kr[] = {P_NIS, P_ERR, P_ACT};
            static const float wl[] = {2.0f, 1.4f, 0.5f}, wr[] = {1, 1, 1};
            vstack(m, m, cw, H - 2 * m, g, 3, kl, wl);
            vstack(m + cw + g, m, cw, H - 2 * m, g, 3, kr, wr);
        }
    } else {
        if (asp >= 1) {
            float mw = fminf(H - 2 * m, W * 0.58f);
            addp(P_MEAS, m, m, mw, H - 2 * m);
            static const int k[] = {P_TABLE, P_INNOV};
            static const float wt[] = {1.1f, 1};
            vstack(m + mw + g, m, W - 2 * m - mw - g, H - 2 * m, g, 2, k, wt);
        } else {
            float mh = fminf(W - 2 * m, H * 0.58f), cw = (W - 2 * m - g) / 2, y0 = m + mh + g;
            addp(P_MEAS, m, m, W - 2 * m, mh);
            addp(P_TABLE, m, y0, cw, H - y0 - m);
            addp(P_INNOV, m + cw + g, y0, cw, H - y0 - m);
        }
    }
}

static int paper(void) { return pal[C_BG][0] > 0.5f; }

static void frame(const Rect *r, const char *head) {
    rectf(r->x, r->y, r->w, r->h, pal[C_PANEL], 1);
    float x0 = r->x + 0.5f, y0 = r->y + 0.5f, x1 = r->x + r->w - 0.5f, y1 = r->y + r->h - 0.5f, l = 9 * U, w = fmaxf(U, 1);
    seg(x0, y0, x1, y0, w, pal[C_FRAME], 0.5f); seg(x0, y1, x1, y1, w, pal[C_FRAME], 0.5f);
    seg(x0, y0, x0, y1, w, pal[C_FRAME], 0.5f); seg(x1, y0, x1, y1, w, pal[C_FRAME], 0.5f);
    float cw = 1.6f * w;
    seg(x0, y0, x0 + l, y0, cw, pal[C_DIM], 1); seg(x0, y0, x0, y0 + l, cw, pal[C_DIM], 1);
    seg(x1, y0, x1 - l, y0, cw, pal[C_DIM], 1); seg(x1, y0, x1, y0 + l, cw, pal[C_DIM], 1);
    seg(x0, y1, x0 + l, y1, cw, pal[C_DIM], 1); seg(x0, y1, x0, y1 - l, cw, pal[C_DIM], 1);
    seg(x1, y1, x1 - l, y1, cw, pal[C_DIM], 1); seg(x1, y1, x1, y1 - l, cw, pal[C_DIM], 1);
    if (head) text(s_head, r->x + 10 * U, r->y + 9 * U + cap_h(s_head), -1, head, pal[C_DIM], 1);
}

static void fmt_mmss(char *b, size_t n, double s) {
    int t = (int)floor(s + 0.5);
    snprintf(b, n, "%d:%02d", t / 60, t % 60);
}

/* ---- strip charts ---- */
static Rect chart_in(const Rect *r, int bars) {
    float top = 28 * U, bot = 17 * U, left = 38 * U, right = 10 * U;
    Rect in = {r->x + left, r->y + top, r->w - left - right, r->h - top - bot};
    if (bars) in.w -= r->w * 0.3f;
    return in;
}
static float chart_y(const Chart *c, const Rect *in, float v) {
    float t = c->logy ? (log10f(fmaxf(v, 1e-9f)) - log10f(c->y0)) / (log10f(c->y1) - log10f(c->y0)) : (v - c->y0) / (c->y1 - c->y0);
    return in->y + in->h * (1 - clampf(t, -0.015f, 1.015f));
}
static float chart_x(const Rect *in, double rt) { return in->x + in->w * (float)(rt / opt_act_len); }

static double nice_step(double range, int want) {
    double raw = range / want, p = pow(10, floor(log10(raw))), f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

static void chart_static(const Rect *r, const Chart *c) {
    char head[160];
    snprintf(head, sizeof head, "%s%s%s", c->title, c->note && c->note[0] ? "  \xc2\xb7  " : "", c->note ? c->note : "");
    frame(r, head);
    Rect in = chart_in(r, c->bars);
    const float *gc = pal[C_GRID], *dc = pal[C_DIM];
    float lw = fmaxf(U, 1);
    /* y grid */
    if (c->logy) {
        for (int e = (int)ceil(log10(c->y0) - 1e-6); e <= (int)floor(log10(c->y1) + 1e-6); e++) {
            float y = chart_y(c, &in, powf(10, (float)e));
            seg(in.x, y, in.x + in.w, y, lw, gc, 0.8f);
            char b[16];
            if (e < 0) snprintf(b, sizeof b, "%.*f", -e, pow(10, e));
            else if (e < 4) snprintf(b, sizeof b, "%.0f", pow(10, e));
            else snprintf(b, sizeof b, "%.0fk", pow(10, e - 3));
            text(s_small, in.x - 5 * U, y + cap_h(s_small) * 0.5f, 1, b, dc, 1);
        }
    } else {
        double st = nice_step(c->y1 - c->y0, 4);
        for (double v = ceil(c->y0 / st) * st; v <= c->y1 + 1e-9; v += st) {
            float y = chart_y(c, &in, (float)v);
            seg(in.x, y, in.x + in.w, y, lw, gc, 0.8f);
            char b[16];
            snprintf(b, sizeof b, st < 1 ? "%.1f" : "%.0f", v);
            text(s_small, in.x - 5 * U, y + cap_h(s_small) * 0.5f, 1, b, dc, 1);
        }
    }
    /* x: sim time */
    double sl = opt_act_len * A.TS, st = nice_step(sl, 5);
    for (double v = st; v <= sl + 1e-6; v += st) {
        float x = chart_x(&in, v / A.TS);
        seg(x, in.y, x, in.y + in.h, lw, gc, 0.45f);
        char b[16];
        fmt_mmss(b, sizeof b, v);
        text(s_small, x, in.y + in.h + 4 * U + cap_h(s_small), 0, b, dc, 1);
    }
    /* chi-square bounds: a band for the moving average, dashes for one sample */
    if (c->bhi > 0) {
        float y0 = chart_y(c, &in, c->bhi), y1 = chart_y(c, &in, c->blo);
        rectf(in.x, y0, in.w, y1 - y0, pal[C_F1], paper() ? 0.12f : 0.09f);
        seg(in.x, y0, in.x + in.w, y0, lw, pal[C_F1], 0.35f);
        seg(in.x, y1, in.x + in.w, y1, lw, pal[C_F1], 0.35f);
    }
    if (c->hi > 0) {
        float pts[4];
        for (int k = 0; k < 2; k++) {
            float y = chart_y(c, &in, k ? c->hi : c->lo);
            pts[0] = in.x; pts[1] = y; pts[2] = in.x + in.w; pts[3] = y;
            poly_dash(pts, 2, 0, lw, pal[C_WARN], 0.55f, 5 * U, 4 * U);
        }
    }
    seg(in.x, in.y + in.h, in.x + in.w, in.y + in.h, lw, pal[C_FRAME], 0.9f);
    seg(in.x, in.y, in.x, in.y + in.h, lw, pal[C_FRAME], 0.9f);
    /* legend, right of the header */
    float x = r->x + r->w - 10 * U, y = r->y + 9 * U + cap_h(s_small);
    for (int i = c->ns - 1; i >= 0; i--) {
        const Series *s = c->s[i];
        if (!s->name || !s->name[0]) continue;
        x -= text_w(s_small, s->name);
        text(s_small, x, y, -1, s->name, pal[s->col], 1);
        x -= 6 * U;
        if (s->style == ST_DOTS) disk(x - 3 * U, y - cap_h(s_small) * 0.4f, 2.2f * U, pal[s->col], 1);
        else seg(x - 12 * U, y - cap_h(s_small) * 0.4f, x, y - cap_h(s_small) * 0.4f, 2 * U, pal[s->col], 1);
        x -= 18 * U;
    }
}

static void chart_dynamic(const Rect *r, const Chart *c, double rt) {
    Rect in = chart_in(r, c->bars);
    if (c->shade)
        for (int i = 0; i < A.nman; i++) {
            if (A.man[i][0] > rt) continue;
            float x0 = chart_x(&in, A.man[i][0]), x1 = chart_x(&in, fmin(A.man[i][1], rt));
            rectf(x0, in.y, fmaxf(x1 - x0, 1), in.h, pal[C_TEXT], paper() ? 0.07f : 0.05f);
        }
    static float *P, *Q;
    static int cap;
    for (int i = 0; i < c->ns; i++) {
        const Series *s = c->s[i];
        if (s->n > cap) { cap = NSER * 2; P = realloc(P, sizeof(float) * cap * 2); Q = realloc(Q, sizeof(float) * cap * 2); }
        const float *col = pal[s->col];
        if (s->style == ST_BAND) {
            if (i + 1 >= c->ns) continue;
            const Series *t = c->s[++i];
            int n = s->n < t->n ? s->n : t->n;
            for (int k = 0; k + 1 < n; k++) {
                float xa = chart_x(&in, s->t[k]), xb = chart_x(&in, s->t[k + 1]);
                float a0 = chart_y(c, &in, s->v[k]), a1 = chart_y(c, &in, s->v[k + 1]);
                float b0 = chart_y(c, &in, t->v[k]), b1 = chart_y(c, &in, t->v[k + 1]);
                V(xa, a0, 0, 0, col, 0.16f, K_FLAT, 0); V(xb, a1, 0, 0, col, 0.16f, K_FLAT, 0); V(xb, b1, 0, 0, col, 0.16f, K_FLAT, 0);
                V(xa, a0, 0, 0, col, 0.16f, K_FLAT, 0); V(xb, b1, 0, 0, col, 0.16f, K_FLAT, 0); V(xa, b0, 0, 0, col, 0.16f, K_FLAT, 0);
            }
            continue;
        }
        int n = 0;
        if (s->style == ST_DOTS) {
            double sum = 0;
            for (int k = 0; k < s->n; k++) {
                float x = chart_x(&in, s->t[k]), y = chart_y(c, &in, s->v[k]);
                disk(x, y, 1.5f * U, col, 0.55f);
                sum += s->v[k];
                if (k >= c->avgn && c->avgn > 0) sum -= s->v[k - c->avgn];
                int m = c->avgn > 0 && k + 1 > c->avgn ? c->avgn : k + 1;
                if (c->avgn > 0) { P[2 * n] = x; P[2 * n + 1] = chart_y(c, &in, (float)(sum / m)); n++; }
            }
            if (n > 1) poly(P, n, 0, 1.8f * U, col, 0.95f, NULL);
            continue;
        }
        for (int k = 0; k < s->n; k++) {
            float x = chart_x(&in, s->t[k]), y = chart_y(c, &in, s->v[k]);
            if (s->style == ST_STEP && n) { P[2 * n] = x; P[2 * n + 1] = P[2 * n - 1]; n++; }
            P[2 * n] = x; P[2 * n + 1] = y; n++;
        }
        if (s->style == ST_STEP && n) { P[2 * n] = chart_x(&in, rt); P[2 * n + 1] = P[2 * n - 1]; n++; }
        if (n > 1) poly(P, n, 0, 1.6f * U, col, 0.95f, NULL);
        if (glow_on && n > 1) poly_k(P, n, 0, 5 * U, col, 0.12f, NULL, K_SOFT);
    }
    (void)Q;
    float x = chart_x(&in, rt);
    seg(x, in.y, x, in.y + in.h, fmaxf(U, 1), pal[C_DIM], 0.6f);
    /* the running mean of the NIS-like charts, against its expectation */
    if (c->bhi > 0 && c->ns && c->s[0]->n) {
        float ty = in.y + 4 * U + cap_h(s_small);
        float tx = in.x + in.w - 4 * U;
        for (int i = c->ns - 1; i >= 0; i--) {
            const Series *s = c->s[i];
            double sum = 0;
            for (int k = 0; k < s->n; k++) sum += s->v[k];
            tx -= textf(s_small, tx, ty, 1, pal[s->col], 0.95f, "%s %.2f", s->name, s->n ? sum / s->n : 0) + 10 * U;
        }
        text(s_small, tx, ty, 1, "mean", pal[C_DIM], 0.95f);
    }
}

/* ---- the plot ---- */
static Rect vr;
static float vk;
static double vcx, vcy;
static float WX(double x) { return vr.x + vr.w * 0.5f + (float)((x - vcx) * vk); }
static float WY(double y) { return vr.y + vr.h * 0.5f - (float)((y - vcy) * vk); }

static void cov_axes(const double *P2, double *l1, double *l2, double *th) {
    double a = P2[0], b = P2[1], c = P2[2], tr = (a + c) / 2, d = sqrt(fmax((a - c) * (a - c) / 4 + b * b, 0));
    *l1 = fmax(tr + d, 0); *l2 = fmax(tr - d, 0);
    *th = 0.5 * atan2(2 * b, a - c);
}
static void cov_ellipse(double x, double y, const double *P2, double k, float *pts) {
    double l1, l2, th;
    cov_axes(P2, &l1, &l2, &th);
    ellipse_pts(WX(x), WY(y), fmaxf((float)(k * sqrt(l1) * vk), 1.2f), fmaxf((float)(k * sqrt(l2) * vk), 1.2f), (float)-th, pts);
}

/* a filter's estimate at sim time s: predicted from its last update, with
 * the update itself eased over a moment */
static void fdisp(const Filt *f, double s, double *x, double *P2) {
    Filt g = *f;
    if (A.id == A_IMM && f == &A.f[0]) {
        Filt m[2] = {A.mo[0], A.mo[1]};
        f_predict(&m[0], s); f_predict(&m[1], s);
        imm_combine(m, A.mu, g.x, g.P);
        g.n = 5;
    } else f_predict(&g, s);
    double e = smooth01((float)((s - f->t_upd) / A.TS / 0.4));
    int n = g.n;
    x[0] = g.x[0] + f->ex * (1 - e); x[1] = g.x[1] + f->ey * (1 - e); x[2] = g.x[2]; x[3] = g.x[3];
    P2[0] = g.P[0] * e + f->eP[0] * (1 - e);
    P2[1] = g.P[1] * e + f->eP[1] * (1 - e);
    P2[2] = g.P[n + 1] * e + f->eP[2] * (1 - e);
}

static void draw_trail(const Trail *t, int col, float a, float w) {
    static float P[2 * TRL], al[TRL];
    int n = t->n;
    for (int i = 0; i < n; i++) {
        int k = TRAIL_AT(t, i);
        P[2 * i] = WX(t->x[k]); P[2 * i + 1] = WY(t->y[k]);
        al[i] = powf((i + 1.0f) / n, 0.8f);
    }
    poly(P, n, 0, w, pal[col], a, al);
}

static void draw_truth(const Path *p, double t0, double t1, float a) {
    if (t1 <= t0) return;
    float P[2 * 241];
    int n = 0;
    for (int i = 0; i <= 240; i++) {
        double o[6];
        path_eval(p, t0 + (t1 - t0) * i / 240, o);
        P[2 * n] = WX(o[0]); P[2 * n + 1] = WY(o[1]); n++;
    }
    poly(P, n, 0, 1.1f * U, pal[C_TRUTH], a, NULL);
    double o[6];
    path_eval(p, t1, o);
    float x = WX(o[0]), y = WY(o[1]), h = 3.5f * U, lw = fmaxf(U, 1);
    seg(x - h, y - h, x + h, y - h, lw, pal[C_TRUTH], a * 2); seg(x + h, y - h, x + h, y + h, lw, pal[C_TRUTH], a * 2);
    seg(x + h, y + h, x - h, y + h, lw, pal[C_TRUTH], a * 2); seg(x - h, y + h, x - h, y - h, lw, pal[C_TRUTH], a * 2);
}

static void draw_est(const double *x, const double *P2, int col, float a, const char *label, double lead, int fill, float ldy) {
    float P[2 * ELN], sx = WX(x[0]), sy = WY(x[1]);
    const float *c = pal[col];
    cov_ellipse(x[0], x[1], P2, 1, P);
    if (fill) fan(P, ELN, sx, sy, c, 0.10f * a);
    poly(P, ELN, 1, 1.4f * U, c, 0.9f * a, NULL);
    cov_ellipse(x[0], x[1], P2, 2, P);
    poly_dash(P, ELN, 1, fmaxf(U, 1), c, 0.55f * a, 5 * U, 4 * U);
    if (lead > 0) seg(sx, sy, WX(x[0] + x[2] * lead), WY(x[1] + x[3] * lead), 1.5f * U, c, 0.85f * a);
    if (glow_on) blob(sx, sy, 9 * U, c, 0.3f * a);
    ring(sx, sy, 3.6f * U, c, a);
    disk(sx, sy, 1.3f * U, c, a);
    if (label) text(s_hudb, sx + 8 * U, sy - 7 * U + ldy, -1, label, c, a);
}

/* a big circle as a polyline (a ring quad would shade its whole box) */
static void circle(float cx, float cy, float r, float w, const float *c, float a, int dashed) {
    int n = (int)clampf(r * 0.5f, 24, 256);
    static float P[2 * 256];
    for (int i = 0; i < n; i++) { P[2 * i] = cx + r * cosf((float)(TAU * i / n)); P[2 * i + 1] = cy + r * sinf((float)(TAU * i / n)); }
    if (dashed) poly_dash(P, n, 1, w, c, a, 3 * U, 5 * U);
    else poly(P, n, 1, w, c, a, NULL);
}

static void draw_grid(void) {
    double wx0 = vcx - vr.w / 2 / vk, wx1 = vcx + vr.w / 2 / vk, wy0 = vcy - vr.h / 2 / vk, wy1 = vcy + vr.h / 2 / vk;
    double st = nice_step(fmin(wx1 - wx0, wy1 - wy0), 6);
    float lw = fmaxf(U, 1);
    for (double x = ceil(wx0 / st) * st; x <= wx1; x += st) seg(WX(x), vr.y, WX(x), vr.y + vr.h, lw, pal[C_GRID], 0.55f);
    for (double y = ceil(wy0 / st) * st; y <= wy1; y += st) seg(vr.x, WY(y), vr.x + vr.w, WY(y), lw, pal[C_GRID], 0.55f);
    /* scale bar */
    float bx = vr.x + 16 * U, by = vr.y + vr.h - 16 * U, bl = (float)(st * vk);
    seg(bx, by, bx + bl, by, 1.6f * U, pal[C_DIM], 1);
    seg(bx, by - 4 * U, bx, by + 1, 1.2f * U, pal[C_DIM], 1);
    seg(bx + bl, by - 4 * U, bx + bl, by + 1, 1.2f * U, pal[C_DIM], 1);
    char b[32];
    if (st >= 1000) snprintf(b, sizeof b, "%g km", st / 1000); else snprintf(b, sizeof b, "%g m", st);
    text(s_small, bx + bl + 6 * U, by + cap_h(s_small) * 0.5f, -1, b, pal[C_DIM], 1);
    /* north */
    float nx = vr.x + vr.w - 22 * U, ny = vr.y + 30 * U;
    seg(nx, ny + 10 * U, nx, ny - 10 * U, 1.4f * U, pal[C_DIM], 1);
    seg(nx, ny - 10 * U, nx - 4 * U, ny - 4 * U, 1.4f * U, pal[C_DIM], 1);
    seg(nx, ny - 10 * U, nx + 4 * U, ny - 4 * U, 1.4f * U, pal[C_DIM], 1);
    text(s_small, nx, ny + 12 * U + cap_h(s_small), 0, "N", pal[C_DIM], 1);
}

static double beam(double s) { return A.rd.th0 - TAU * s / A.rd.T; }

static void draw_radar(double s, int rings) {
    const Radar *rd = &A.rd;
    float cx = WX(rd->x), cy = WY(rd->y), R = (float)(rd->rmax * vk);
    if (rings) {
        double st = nice_step(rd->rmax, 5);
        for (double r = st; r <= rd->rmax + 1; r += st) {
            circle(cx, cy, (float)(r * vk), fmaxf(U, 1), pal[C_GRID], 0.9f, 0);
            char b[16];
            snprintf(b, sizeof b, "%g", r / 1000);
            text(s_small, cx + (float)(r * vk) * 0.7071f + 3 * U, cy - (float)(r * vk) * 0.7071f - 3 * U, -1, b, pal[C_DIM], 0.9f);
        }
    }
    /* the beam and its afterglow, trailing behind the sweep */
    double th = beam(s);
    const int nseg = 28;
    float span = 0.75f, Rr = fminf(R, 4000.0f);
    for (int i = 0; i < nseg; i++) {
        float a0 = (float)th + span * i / nseg, a1 = (float)th + span * (i + 1) / nseg;
        float al0 = 0.1f * powf(1 - (float)i / nseg, 2), al1 = 0.1f * powf(1 - (float)(i + 1) / nseg, 2);
        if (paper()) { al0 *= 0.6f; al1 *= 0.6f; }
        V(cx, cy, 0, 0, pal[C_SWEEP], al0 * 0.4f, K_FLAT, 0);
        V(cx + Rr * cosf(a0), cy - Rr * sinf(a0), 0, 0, pal[C_SWEEP], al0, K_FLAT, 0);
        V(cx + Rr * cosf(a1), cy - Rr * sinf(a1), 0, 0, pal[C_SWEEP], al1, K_FLAT, 0);
    }
    float bx = cx + Rr * cosf((float)th), by = cy - Rr * sinf((float)th);
    if (glow_on) seg_k(cx, cy, bx, by, 6 * U, pal[C_SWEEP], 0.25f, K_SOFT);
    seg(cx, cy, bx, by, 1.4f * U, pal[C_SWEEP], 0.8f);
    /* the site */
    float h = 6 * U;
    seg(cx, cy - h, cx + h, cy, 1.4f * U, pal[C_TEXT], 1); seg(cx + h, cy, cx, cy + h, 1.4f * U, pal[C_TEXT], 1);
    seg(cx, cy + h, cx - h, cy, 1.4f * U, pal[C_TEXT], 1); seg(cx - h, cy, cx, cy - h, 1.4f * U, pal[C_TEXT], 1);
}

/* detections, fading over a scan and a half; wedges for the newest */
static void draw_dets(double s, int wedges) {
    double pers = A.rd.T * 1.5;
    int shown = 0;
    for (int i = A.dhn - 1; i >= 0 && i >= A.dhn - DH; i--) {
        const Det *d = &A.dh[i % DH];
        if (d->t > s) continue;
        double age = s - d->t;
        if (age > pers) { if (++shown > 600) break; continue; }
        float a = powf(1 - (float)(age / pers), 1.3f), x = WX(d->x), y = WY(d->y);
        int col = d->tgt < 0 && !d->used ? C_CLUT : C_MEAS;
        if (A.id == A_MTT) col = d->used == 2 ? C_MEAS : C_CLUT;
        if (glow_on && col == C_MEAS) blob(x, y, 6 * U, pal[col], 0.35f * a);
        disk(x, y, (col == C_MEAS ? 2.2f : 1.7f) * U, pal[col], a);
        if (wedges && age < A.rd.T * 2.5) {
            /* the measurement's +-2 sigma cell: an annular sector */
            float P[2 * 34];
            int n = 0;
            double r0 = d->r - 2 * A.rd.sr, r1 = d->r + 2 * A.rd.sr, b0 = d->b - 2 * A.rd.sb, b1 = d->b + 2 * A.rd.sb;
            for (int k = 0; k <= 16; k++) {
                double b = b0 + (b1 - b0) * k / 16;
                P[2 * n] = WX(A.rd.x + r1 * cos(b)); P[2 * n + 1] = WY(A.rd.y + r1 * sin(b)); n++;
            }
            for (int k = 16; k >= 0; k--) {
                double b = b0 + (b1 - b0) * k / 16;
                P[2 * n] = WX(A.rd.x + r0 * cos(b)); P[2 * n + 1] = WY(A.rd.y + r0 * sin(b)); n++;
            }
            float wa = (float)(1 - age / (A.rd.T * 2.5));
            poly(P, n, 1, fmaxf(U, 1), pal[C_MEAS], 0.75f * wa, NULL);
            fan(P, n, x, y, pal[C_MEAS], 0.08f * wa);
        }
    }
}

static void draw_card(const Rect *r, double rt) {
    float a = smooth01((float)(rt / 0.7)) * (1 - smooth01((float)((rt - 6.5) / 1.2)));
    if (a <= 0.003f) return;
    float w = fminf(r->w * 0.82f, 980 * U), pad = 22 * U, x = r->x + (r->w - w) / 2;
    int nl = text_wrap(s_cardsub, 0, 0, w - 2 * pad, styles[s_cardsub].px * 1.35f, ACT_LINE[A.id], pal[C_DIM], a, 0);
    int nt = text_wrap(s_card, 0, 0, w - 2 * pad, styles[s_card].px * 1.15f, ACT_NAME[A.id], pal[C_TEXT], a, 0);
    float h = pad * 2 + cap_h(s_head) + 14 * U + nt * styles[s_card].px * 1.15f + 6 * U + nl * styles[s_cardsub].px * 1.35f;
    float y = r->y + (r->h - h) * 0.42f;
    rectf(x, y, w, h, pal[C_BG], 0.86f * a);
    float lw = fmaxf(U, 1);
    seg(x, y, x + w, y, lw, pal[C_FRAME], a); seg(x, y + h, x + w, y + h, lw, pal[C_FRAME], a);
    seg(x, y, x, y + h, lw, pal[C_FRAME], a); seg(x + w, y, x + w, y + h, lw, pal[C_FRAME], a);
    seg(x, y, x + w * 0.18f, y, 2.5f * U, pal[C_F1], a);
    float ty = y + pad + cap_h(s_head);
    textf(s_head, x + pad, ty, -1, pal[C_F1], a, "ACT %d / %d", A.id + 1, NACT);
    ty += 14 * U + cap_h(s_card);
    text_wrap(s_card, x + pad, ty, w - 2 * pad, styles[s_card].px * 1.15f, ACT_NAME[A.id], pal[C_TEXT], a, 1);
    ty += (nt - 1) * styles[s_card].px * 1.15f + 8 * U + styles[s_cardsub].px * 1.2f;
    text_wrap(s_cardsub, x + pad, ty, w - 2 * pad, styles[s_cardsub].px * 1.35f, ACT_LINE[A.id], pal[C_DIM], a, 1);
}

static void legend_line(float *y, float x, int col, int dot, const char *name) {
    float c = cap_h(s_small);
    if (dot) disk(x + 6 * U, *y - c * 0.4f, 2.3f * U, pal[col], 1);
    else seg(x, *y - c * 0.4f, x + 12 * U, *y - c * 0.4f, 2 * U, pal[col], 1);
    text(s_small, x + 18 * U, *y, -1, name, pal[col == C_CLUT ? C_DIM : col], 1);
    *y += c * 1.9f;
}

static void draw_plot(const Rect *r, double s, double rt) {
    (void)rt;
    vr = *r;
    vr.x += 1; vr.y += 1; vr.w -= 2; vr.h -= 2;
    if (A.follow) {                     /* a camera that eases after the car */
        double sx = 0, sy = 0;
        for (int i = 0; i < 9; i++) {
            double o[6];
            path_eval(&A.tp[0], fmax(s - A.follow_b + (A.follow_b + A.follow_f) * i / 8, 0), o);
            sx += o[0]; sy += o[1];
        }
        vcx = sx / 9; vcy = sy / 9;
        vk = fminf(vr.w, vr.h) / A.follow_w;
    } else {
        vcx = (A.view[0] + A.view[2]) / 2; vcy = (A.view[1] + A.view[3]) / 2;
        vk = fminf(vr.w / (A.view[2] - A.view[0]), vr.h / (A.view[3] - A.view[1]));
    }
    draw_grid();
    char lab[64];
    if (A.has_radar) draw_radar(s, 1);
    if (A.id == A_KF) {
        draw_truth(&A.tp[0], fmax(0, s - 60), s, 0.35f);
        for (int i = A.dhn - 1; i >= 0 && i >= A.dhn - 40; i--) {
            const Det *d = &A.dh[i % DH];
            if (d->t > s) continue;
            float a = 1 - (float)((s - d->t) / 40), x = WX(d->x), y = WY(d->y);
            if (a <= 0) continue;
            if (s - d->t < 3) circle(x, y, (float)(GPS_SIG * vk), fmaxf(U, 1), pal[C_MEAS], 0.4f * a * (float)(1 - (s - d->t) / 3), 0);
            if (glow_on) blob(x, y, 5 * U, pal[C_MEAS], 0.3f * a);
            disk(x, y, 2.2f * U, pal[C_MEAS], a);
        }
    } else if (A.id == A_UKF || A.id == A_IMM) {
        draw_truth(&A.tp[0], 0, s, 0.3f);
        draw_dets(s, 1);
    } else if (A.id == A_PF) {
        draw_truth(&A.tp[0], 0, s, 0.35f);
        /* the listening ship and its bearing lines */
        double oo[6];
        path_eval(&A.own, s, oo);
        float L = (float)(fmax(A.view[2] - A.view[0], A.view[3] - A.view[1]) * 1.5);
        for (int i = A.nbrg - 1; i >= 0 && i >= A.nbrg - 8; i--) {
            if (i >= NBRG || A.brg_t[i] > s) continue;
            float a = 1 - (A.nbrg - 1 - i) / 8.0f;
            double b = A.brg_b[i];
            seg(WX(A.brg_ox[i]), WY(A.brg_oy[i]), WX(A.brg_ox[i] + L * cos(b)), WY(A.brg_oy[i] + L * sin(b)),
                (i == A.nbrg - 1 ? 1.4f : 1.0f) * U, pal[C_MEAS], 0.5f * a * a);
        }
        /* the hypotheses, drifting with their own velocities */
        if (A.nbrg > 0) {
            double dt = s - A.s_pf;
            for (int i = 0; i < NP; i++) {
                float a = clampf((float)(A.pw[i] * NP) * 0.6f, 0.12f, 1);
                disk(WX(A.px[i][0] + A.px[i][2] * dt), WY(A.px[i][1] + A.px[i][3] * dt), 1.25f * U, pal[C_F1], 0.55f * a);
            }
        }
        float P[2 * 241];
        int n = 0;
        for (int i = 0; i <= 240; i++) {
            double o[6];
            path_eval(&A.own, s * i / 240, o);
            P[2 * n] = WX(o[0]); P[2 * n + 1] = WY(o[1]); n++;
        }
        poly(P, n, 0, 1.6f * U, pal[C_TEXT], 0.7f, NULL);
        float ox = WX(oo[0]), oy = WY(oo[1]), h = oo[4];
        float hx = cosf(h), hy = -sinf(h), sz = 8 * U;
        V(ox + hx * sz, oy + hy * sz, 0, 0, pal[C_TEXT], 1, K_FLAT, 0);
        V(ox - hx * sz * 0.6f - hy * sz * 0.5f, oy - hy * sz * 0.6f + hx * sz * 0.5f, 0, 0, pal[C_TEXT], 1, K_FLAT, 0);
        V(ox - hx * sz * 0.6f + hy * sz * 0.5f, oy - hy * sz * 0.6f - hx * sz * 0.5f, 0, 0, pal[C_TEXT], 1, K_FLAT, 0);
        text(s_small, ox + 10 * U, oy + 14 * U, -1, "OWN SHIP", pal[C_TEXT], 0.9f);
    } else {
        draw_truth(&A.tp[0], 0, 0, 0);
        for (int i = 0; i < A.nt; i++) {
            double t0 = fmax(A.tb[i], 0), t1 = fmin(s, A.te[i]);
            if (t1 > t0) draw_truth(&A.tp[i], t0, t1, 0.22f);
        }
        draw_dets(s, 0);
    }

    /* the estimates */
    if (A.id == A_MTT) {
        for (int i = 0; i < A.ntk; i++) {
            Track *k = &A.tk[i];
            if (k->st == TK_TENT && k->hits < 2) continue;
            double x[4], P2[3];
            fdisp(&k->f, s, x, P2);
            if (k->st == TK_TENT) {
                float pts[2 * ELN];
                double G[3] = {P2[0], P2[1], P2[2]};
                Det fake = {0, hypot(x[0], x[1]), atan2(x[1], x[0]), 0, 0, 0, 0};
                double R[4];
                det_R(&fake, R);
                G[0] += R[0]; G[1] += R[1]; G[2] += R[3];
                cov_ellipse(x[0], x[1], G, sqrt(GATE), pts);
                poly_dash(pts, ELN, 1, fmaxf(U, 1), pal[C_F2], 0.45f, 2 * U, 4 * U);
                ring(WX(x[0]), WY(x[1]), 3 * U, pal[C_F2], 0.6f);
                continue;
            }
            float a = 1;
            if (k->st == TK_DEAD) a = 1 - clampf((float)((s - k->t_dead) / (3 * A.TS)), 0, 1);
            if (a <= 0) continue;
            draw_trail(&k->tr, C_F1, 0.5f * a, 1.3f * U);
            snprintf(lab, sizeof lab, "T%02d", k->id);
            /* the gate: the predicted measurement's 99% region */
            float pts[2 * ELN];
            double G[3] = {P2[0], P2[1], P2[2]}, R[4];
            Det fake = {0, hypot(x[0], x[1]), atan2(x[1], x[0]), 0, 0, 0, 0};
            det_R(&fake, R);
            G[0] += R[0]; G[1] += R[1]; G[2] += R[3];
            cov_ellipse(x[0], x[1], G, sqrt(GATE), pts);
            poly_dash(pts, ELN, 1, fmaxf(U, 1), pal[C_F1], 0.4f * a, 2 * U, 3 * U);
            draw_est(x, P2, C_F1, a, lab, 30, 1, 0);
            if (k->st == TK_DEAD) {
                float sx = WX(x[0]), sy = WY(x[1]), h = 6 * U;
                seg(sx - h, sy - h, sx + h, sy + h, 1.6f * U, pal[C_WARN], a);
                seg(sx - h, sy + h, sx + h, sy - h, 1.6f * U, pal[C_WARN], a);
            }
            double sw = (s - k->t_swap) / A.TS;
            if (sw >= 0 && sw < 5) {
                float fa = (float)(1 - sw / 5) * (0.6f + 0.4f * (float)fabs(sin(sw * 6)));
                circle(WX(x[0]), WY(x[1]), 16 * U, 1.6f * U, pal[C_WARN], fa, 0);
                text(s_hudb, WX(x[0]) + 8 * U, WY(x[1]) + 20 * U, -1, "SWAP?", pal[C_WARN], fa);
            }
        }
    } else if (A.finit || (A.id == A_PF && A.nbrg > 0)) {
        for (int k = A.nf - 1; k >= 0; k--) {
            double x[4], P2[3];
            fdisp(&A.f[k], s, x, P2);
            draw_trail(&A.ft[k], A.fcol[k], 0.75f, 1.4f * U);
            double lead = A.id == A_KF ? 2 : A.id == A_PF ? 300 : 15;
            draw_est(x, P2, A.fcol[k], k == 0 ? 1 : 0.85f, A.fname[k], lead, k == 0, k * 16 * U);
            if (A.id == A_UKF && k == 0) {   /* the sigma points of the predicted covariance */
                Filt g = A.f[0];
                f_predict(&g, s);
                double sig[2 * MX + 1][MX], wm[2 * MX + 1], wc[2 * MX + 1];
                ukf_sigma(g.x, g.P, 4, sig, wm, wc);
                for (int i = 0; i < 9; i++) {
                    float sx = WX(sig[i][0]), sy = WY(sig[i][1]), h = 3 * U;
                    V(sx, sy - h, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0); V(sx + h, sy, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0); V(sx, sy + h, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0);
                    V(sx, sy - h, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0); V(sx, sy + h, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0); V(sx - h, sy, 0, 0, pal[C_F3], 0.9f, K_FLAT, 0);
                }
            }
            if (A.id == A_IMM && k == 0) {   /* the CT probability as an arc round the marker */
                float sx = WX(x[0]), sy = WY(x[1]), rr = 7 * U, P[2 * 49];
                int n = 0;
                for (int i = 0; i <= 48; i++) {
                    float a = (float)(-PI / 2 + TAU * A.mu[1] * i / 48);
                    P[2 * n] = sx + rr * cosf(a); P[2 * n + 1] = sy + rr * sinf(a); n++;
                }
                poly(P, n, 0, 2.2f * U, pal[C_F3], 1, NULL);
            }
        }
    }

    /* the act, the sim clock and a legend, top left */
    float x = vr.x + 14 * U, y = vr.y + 14 * U + cap_h(s_hudb);
    textf(s_hudb, x, y, -1, pal[C_TEXT], 1, "%d/%d  %s", A.id + 1, NACT, ACT_NAME[A.id]);
    y += cap_h(s_hudb) * 1.9f;
    char tb[32];
    fmt_mmss(tb, sizeof tb, s);
    textf(s_small, x, y, -1, pal[C_DIM], 1, "T+%s  sim \xc3\x97%g", tb, A.TS);
    y += cap_h(s_small) * 2.4f;
    legend_line(&y, x, C_TRUTH, 0, "truth");
    if (A.id == A_KF) legend_line(&y, x, C_MEAS, 1, "GPS fix");
    else if (A.id == A_PF) legend_line(&y, x, C_MEAS, 0, "bearing");
    else if (A.id == A_MTT) { legend_line(&y, x, C_MEAS, 1, "assigned"); legend_line(&y, x, C_CLUT, 1, "clutter / unused"); }
    else legend_line(&y, x, C_MEAS, 1, "detection \xc2\xb1" "2\xcf\x83");
    if (A.id == A_MTT) { legend_line(&y, x, C_F1, 0, "confirmed"); legend_line(&y, x, C_F2, 0, "tentative"); }
    else for (int k = 0; k < A.nf; k++) legend_line(&y, x, A.fcol[k], 0, A.fname[k]);
    if (A.id == A_UKF) legend_line(&y, x, C_F3, 1, "sigma points");
    if (A.id == A_PF) legend_line(&y, x, C_F1, 1, "particles");
    if (A.id == A_IMM) legend_line(&y, x, C_F3, 0, "\xce\xbc(CT)");
    y += cap_h(s_small);
    textf(s_small, x, y, -1, pal[C_DIM], 1, "ellipses 1\xcf\x83 / 2\xcf\x83%s", A.id == A_MTT ? ", gates 99%" : "");
    if (A.id == A_PF) {
        y += cap_h(s_small) * 1.9f;
        textf(s_small, x, y, -1, pal[C_DIM], 1, "N_eff %4.0f / %d  resampled %d\xc3\x97", A.neff, NP, A.nres);
    }
    if (A.id == A_MTT) {
        y += cap_h(s_small) * 1.9f;
        int nc = 0;
        for (int i = 0; i < A.ntk; i++) nc += A.tk[i].st == TK_CONF;
        textf(s_small, x, y, -1, pal[C_DIM], 1, "%d tracks  %d IDs issued  %d swaps", nc, A.next_id - 1, A.nswaps);
    }
}

/* ---- the matrices ---- */
static void fmt_cell(char *b, size_t n, double v) {
    double a = fabs(v);
    const char *sg = v < 0 ? "-" : "";
    if (a >= 1e6) snprintf(b, n, "%s%.0fM", sg, a / 1e6);
    else if (a >= 1e4) snprintf(b, n, "%s%.0fk", sg, a / 1e3);
    else if (a >= 1e3) snprintf(b, n, "%s%.1fk", sg, a / 1e3);
    else if (a >= 10) snprintf(b, n, "%s%.0f", sg, a);
    else if (a >= 1) snprintf(b, n, "%s%.1f", sg, a);
    else if (a >= 0.005) snprintf(b, n, "%s.%02.0f", sg, a * 100);
    else snprintf(b, n, "0");
}

/* mode 0: covariance (colour by correlation), 1: gain (by the largest),
 * 2: assignment costs (by gate) */
static void heat(float x, float y, float cs, int nr, int nc, const double *M, int mode, const char *const *rl,
                 const char *const *cl, const int *asg) {
    double mx[8];                       /* gains: each column by its largest (their units differ) */
    for (int j = 0; j < nc && j < 8; j++) {
        mx[j] = 1e-30;
        for (int i = 0; i < nr; i++) mx[j] = fmax(mx[j], fabs(M[i * nc + j]));
    }
    int numbers = cs >= 21 * U, sc = style(F_MONO, fminf(11 * U, cs * 0.3f));
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nc; j++) {
            double v = M[i * nc + j], c;
            if (mode == 0) c = v / sqrt(fmax(M[i * nc + i] * M[j * nc + j], 1e-30));
            else if (mode == 1) c = v / mx[j < 8 ? j : 7];
            else c = v < GATE ? 1 - v / GATE * 0.8 : 0;
            int col = c >= 0 ? C_F1 : C_F2;
            float cx = x + j * cs, cy = y + i * cs;
            rectf(cx + 1, cy + 1, cs - 2, cs - 2, pal[col], 0.1f + 0.7f * (float)fmin(fabs(c), 1));
            if (numbers) {
                char b[16];
                if (mode == 2 && v >= GATE) snprintf(b, sizeof b, "\xc2\xb7");
                else fmt_cell(b, sizeof b, v);
                text(sc, cx + cs / 2, cy + cs / 2 + cap_h(sc) / 2, 0, b, pal[C_TEXT], 0.95f);
            }
            if (asg && asg[i] == j) {
                float lw = 1.6f * U;
                seg(cx + 1, cy + 1, cx + cs - 1, cy + 1, lw, pal[C_TEXT], 1); seg(cx + cs - 1, cy + 1, cx + cs - 1, cy + cs - 1, lw, pal[C_TEXT], 1);
                seg(cx + cs - 1, cy + cs - 1, cx + 1, cy + cs - 1, lw, pal[C_TEXT], 1); seg(cx + 1, cy + cs - 1, cx + 1, cy + 1, lw, pal[C_TEXT], 1);
            }
        }
    for (int i = 0; rl && i < nr; i++) text(s_small, x - 5 * U, y + i * cs + cs / 2 + cap_h(s_small) / 2, 1, rl[i], pal[C_DIM], 1);
    for (int j = 0; cl && j < nc; j++) text(s_small, x + j * cs + cs / 2, y - 5 * U, 0, cl[j], pal[C_DIM], 1);
}

static const char *const LBL4[] = {"x", "y", "vx", "vy", "ax", "ay"}, *const LBL5[] = {"x", "y", "vx", "vy", "\xcf\x89"};
static const char *const LBLZ[] = {"zx", "zy"}, *const LBLRB[] = {"r", "\xce\xb2"};

static const char *mats_head(void) {
    switch (A.id) {
    case A_PF: return "COVARIANCE P (particles)  \xc2\xb7  RANGE DISTRIBUTION";
    case A_MTT: return "ASSIGNMENT d\xc2\xb2 (track \xc3\x97 detection)  \xc2\xb7  P OF THE LATEST UPDATE";
    case A_IMM: return "COVARIANCE P (combined)  \xc2\xb7  GAIN K (CT model)";
    default: return "COVARIANCE P  \xc2\xb7  KALMAN GAIN K";
    }
}

static void draw_mats(const Rect *r, double s) {
    (void)s;
    Rect in = {r->x + 34 * U, r->y + 46 * U, r->w - 44 * U, r->h - 54 * U};
    double M[36];
    if (A.id == A_MTT) {
        float cs = fminf(fminf(in.h / 6.0f, in.w * 0.55f / 8.0f), 40 * U);
        int nr = A.cost_r, nc = A.cost_c;
        double C[48];
        char rl[6][8], cl[8][8];
        const char *rp[6], *cp[8];
        for (int i = 0; i < nr; i++) {
            if (A.cost_id[i]) snprintf(rl[i], 8, "T%02d", A.cost_id[i]); else snprintf(rl[i], 8, "?");
            rp[i] = rl[i];
            for (int j = 0; j < nc; j++) C[i * nc + j] = A.cost[i][j];
        }
        for (int j = 0; j < nc; j++) { snprintf(cl[j], 8, "z%d", j + 1); cp[j] = cl[j]; }
        if (nr && nc) heat(in.x + 6 * U, in.y, cs, nr, nc, C, 2, rp, cp, A.cost_asg);
        else text(s_small, in.x, in.y + cs, -1, "no confirmed track in this sector", pal[C_DIM], 1);
        if (A.last_trk >= 0 && A.last_trk < A.ntk) {
            const Track *k = &A.tk[A.last_trk];
            float cs2 = fminf(fminf(in.h / 4.0f, in.w * 0.4f / 4.0f), 48 * U), x2 = in.x + in.w - cs2 * 4;
            heat(x2, in.y, cs2, 4, 4, k->f.P, 0, LBL4, LBL4, NULL);
            textf(s_small, x2, in.y + cs2 * 4 + 14 * U, -1, pal[C_DIM], 1, "T%02d", k->id);
        }
        return;
    }
    if (!A.finit && !(A.id == A_PF && A.nbrg)) return;
    const Filt *f = &A.f[0];
    int n = f->n > 6 ? 6 : f->n;
    float cs = fminf(fminf(in.h / n, in.w * 0.58f / n), 48 * U);
    memcpy(M, f->P, sizeof(double) * n * n);
    heat(in.x, in.y, cs, n, n, M, 0, n == 5 ? LBL5 : LBL4, n == 5 ? LBL5 : LBL4, NULL);
    float x2 = in.x + cs * n + 50 * U;
    if (A.id == A_PF) {                 /* where the particles think the ship is, by range */
        double hist[24] = {0}, mx = 1e-12, oo[6];
        path_eval(&A.own, A.s_pf, oo);
        for (int i = 0; i < NP; i++) {
            int b = (int)(hypot(A.px[i][0] - oo[0], A.px[i][1] - oo[1]) / 1000);
            if (b >= 0 && b < 24) hist[b] += A.pw[i];
        }
        for (int b = 0; b < 24; b++) mx = fmax(mx, hist[b]);
        float bw = (r->x + r->w - 12 * U - x2) / 24, by = in.y + in.h - 14 * U, bh = in.h - 24 * U;
        double oot[6];
        path_eval(&A.tp[0], A.s_pf, oot);
        double tr = hypot(oot[0] - oo[0], oot[1] - oo[1]) / 1000;
        for (int b = 0; b < 24; b++) {
            float h = (float)(hist[b] / mx) * bh;
            rectf(x2 + b * bw + 1, by - h, bw - 2, h, pal[C_F1], 0.75f);
        }
        float tx = x2 + (float)tr * bw;
        seg(tx, by - bh, tx, by, 1.4f * U, pal[C_TRUTH], 0.9f);
        seg(x2, by, x2 + 24 * bw, by, fmaxf(U, 1), pal[C_FRAME], 1);
        for (int b = 0; b <= 24; b += 5) textf(s_small, x2 + b * bw, by + 4 * U + cap_h(s_small), 0, pal[C_DIM], 1, "%d", b);
        text(s_small, x2 + 24 * bw, by + 4 * U + cap_h(s_small), 1, "km", pal[C_DIM], 1);
        return;
    }
    const Filt *g = A.id == A_IMM ? &A.mo[1] : f;
    int gn = g->n;
    float cs2 = fminf(fminf(in.h / gn, (r->x + r->w - 12 * U - x2) / 2), 48 * U);
    heat(x2, in.y, cs2, gn, 2, g->K, 1, gn == 5 ? LBL5 : LBL4, A.id == A_KF ? LBLZ : LBLRB, NULL);
}

static void draw_mats_static(const Rect *r) {
    frame(r, mats_head());
}

/* ---- sensor view: measurement space ---- */
static const char *meas_head(void) {
    switch (A.id) {
    case A_KF: return "ERROR SPACE  \xc2\xb7  fix \xe2\x88\x92 truth, estimate \xe2\x88\x92 truth (m)";
    case A_UKF: return "SIGMA POINTS  \xc2\xb7  state space (x, y)  \xe2\x86\x92  measurement space (r, \xce\xb2)";
    case A_IMM: return "MODE-CONDITIONED ESTIMATES  \xc2\xb7  close in on the target";
    case A_PF: return "BEARING-TIME RECORD  \xc2\xb7  measured, truth, particle spread \xc2\xb1" "2\xcf\x83";
    default: return "B-SCOPE  \xc2\xb7  range vs bearing, one scan and a half";
    }
}

static void axes_box(const Rect *in) {
    float lw = fmaxf(U, 1);
    seg(in->x, in->y, in->x + in->w, in->y, lw, pal[C_FRAME], 0.8f); seg(in->x, in->y + in->h, in->x + in->w, in->y + in->h, lw, pal[C_FRAME], 0.8f);
    seg(in->x, in->y, in->x, in->y + in->h, lw, pal[C_FRAME], 0.8f); seg(in->x + in->w, in->y, in->x + in->w, in->y + in->h, lw, pal[C_FRAME], 0.8f);
}

static void draw_meas(const Rect *r, double s, double rt) {
    Rect in = {r->x + 40 * U, r->y + 36 * U, r->w - 56 * U, r->h - 60 * U};
    float lw = fmaxf(U, 1);
    if (A.id == A_KF) {
        float sz = fminf(in.w, in.h), cx = in.x + in.w / 2, cy = in.y + in.h / 2, k = sz / 2 / 22.0f;
        seg(cx - sz / 2, cy, cx + sz / 2, cy, lw, pal[C_GRID], 1); seg(cx, cy - sz / 2, cx, cy + sz / 2, lw, pal[C_GRID], 1);
        for (int m = 1; m <= 3; m++) {
            circle(cx, cy, (float)(m * GPS_SIG * k), lw, pal[C_MEAS], 0.5f, 1);
            textf(s_small, cx + (float)(m * GPS_SIG * k) * 0.7071f + 3 * U, cy - (float)(m * GPS_SIG * k) * 0.7071f, -1, pal[C_DIM], 1, "%d\xcf\x83", m);
        }
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < A.es_n; i++) {
                int idx = (A.es_h - 1 - i + RING) % RING;
                float a = 1 - (float)i / RING;
                int col = c == 0 ? C_MEAS : c == 1 ? C_F1 : C_F3;
                disk(cx + A.es[c][idx][0] * k, cy - A.es[c][idx][1] * k, (c ? 2.0f : 1.7f) * U, pal[col], 0.75f * a);
            }
        if (A.finit) {
            double x[4], P2[3];
            for (int f = 0; f < 2; f++) {
                fdisp(&A.f[f], s, x, P2);
                double l1, l2, th;
                cov_axes(P2, &l1, &l2, &th);
                float P[2 * ELN];
                ellipse_pts(cx, cy, (float)(2 * sqrt(l1) * k), (float)(2 * sqrt(l2) * k), (float)-th, P);
                poly(P, ELN, 1, 1.5f * U, pal[f ? C_F3 : C_F1], 0.9f, NULL);
            }
        }
        float y = in.y + cap_h(s_small);
        legend_line(&y, in.x, C_MEAS, 1, "GPS");
        legend_line(&y, in.x, C_F1, 1, "KF-CV, 2\xcf\x83 of P");
        legend_line(&y, in.x, C_F3, 1, "KF-CA, 2\xcf\x83 of P");
        return;
    }
    if (A.id == A_UKF) {
        if (!A.finit) return;
        Filt g = A.f[0];
        f_predict(&g, s);
        double sig[2 * MX + 1][MX], wm[2 * MX + 1], wc[2 * MX + 1];
        ukf_sigma(g.x, g.P, 4, sig, wm, wc);
        Rect L = {in.x, in.y, in.w * 0.46f, in.h}, Rr = {in.x + in.w * 0.54f, in.y, in.w * 0.46f, in.h};
        axes_box(&L); axes_box(&Rr);
        /* state space: the predicted ellipse and its sigma points */
        double l1, l2, th, P2[3] = {g.P[0], g.P[1], g.P[5]};
        cov_axes(P2, &l1, &l2, &th);
        double ext = 3.2 * sqrt(l1) + 1;
        float k = fminf(L.w, L.h) / 2 / (float)ext, cx = L.x + L.w / 2, cy = L.y + L.h / 2, P[2 * ELN];
        ellipse_pts(cx, cy, (float)(sqrt(l1) * k), (float)(sqrt(l2) * k), (float)-th, P);
        poly(P, ELN, 1, 1.4f * U, pal[C_F1], 0.9f, NULL);
        ellipse_pts(cx, cy, (float)(2 * sqrt(l1) * k), (float)(2 * sqrt(l2) * k), (float)-th, P);
        poly_dash(P, ELN, 1, lw, pal[C_F1], 0.5f, 5 * U, 4 * U);
        for (int i = 0; i < 9; i++) disk(cx + (float)((sig[i][0] - g.x[0]) * k), cy - (float)((sig[i][1] - g.x[1]) * k), 3 * U, pal[C_F3], 0.95f);
        textf(s_small, L.x + 6 * U, L.y + L.h - 6 * U, -1, pal[C_DIM], 1, "\xc2\xb1%.0f m", ext);
        /* measurement space: h of each sigma point; UKF vs EKF predicted S */
        double Z[9][2], zh[2] = {0, 0}, ref[2], R[4] = {A.rd.sr * A.rd.sr, 0, 0, A.rd.sb * A.rd.sb};
        rb_of(g.x, A.rd.x, A.rd.y, ref);
        for (int i = 0; i < 9; i++) {
            rb_of(sig[i], A.rd.x, A.rd.y, Z[i]);
            Z[i][1] = ref[1] + wrap_pi(Z[i][1] - ref[1]);
            zh[0] += wm[i] * Z[i][0]; zh[1] += wm[i] * Z[i][1];
        }
        double Su[4] = {R[0], 0, 0, R[3]};
        for (int i = 0; i < 9; i++) {
            double d0 = Z[i][0] - zh[0], d1 = Z[i][1] - zh[1];
            Su[0] += wc[i] * d0 * d0; Su[1] += wc[i] * d0 * d1; Su[3] += wc[i] * d1 * d1;
        }
        double dx = g.x[0] - A.rd.x, dy = g.x[1] - A.rd.y, r2 = dx * dx + dy * dy, rr = sqrt(r2);
        double Hm[8] = {dx / rr, dy / rr, 0, 0, -dy / r2, dx / r2, 0, 0}, PHt[8], Se[4];
        mmT(g.P, Hm, PHt, 4, 4, 2);
        mm(Hm, PHt, Se, 2, 4, 2);
        Se[0] += R[0]; Se[3] += R[3];
        /* axes in metres along range and across (r * beta) */
        double ext2 = 0;
        for (int i = 0; i < 9; i++) ext2 = fmax(ext2, fmax(fabs(Z[i][0] - ref[0]), fabs((Z[i][1] - ref[1]) * ref[0])));
        ext2 = fmax(ext2, 3 * sqrt(fmax(Se[0], Se[3] * r2))) * 1.15 + 1;
        float k2 = fminf(Rr.w, Rr.h) / 2 / (float)ext2, mx = Rr.x + Rr.w / 2, my = Rr.y + Rr.h / 2;
        for (int pass = 0; pass < 2; pass++) {
            const double *S = pass ? Se : Su;
            double Sm[3] = {S[3] * r2, S[1] * rr, S[0]};      /* (cross, range) in metres */
            cov_axes(Sm, &l1, &l2, &th);
            ellipse_pts(mx + (float)((pass ? 0 : (zh[1] - ref[1]) * ref[0]) * k2), my - (float)((pass ? 0 : zh[0] - ref[0]) * k2),
                        (float)(2 * sqrt(l1) * k2), (float)(2 * sqrt(l2) * k2), (float)-th, P);
            poly(P, ELN, 1, 1.5f * U, pal[pass ? C_F2 : C_F1], 0.9f, NULL);
        }
        for (int i = 0; i < 9; i++)
            disk(mx + (float)((Z[i][1] - ref[1]) * ref[0] * k2), my - (float)((Z[i][0] - ref[0]) * k2), 3 * U, pal[C_F3], 0.95f);
        text(s_small, Rr.x + 6 * U, Rr.y + Rr.h - 6 * U, -1, "\xe2\x86\x91 range   \xe2\x86\x92 r\xc2\xb7\xce\xb2", pal[C_DIM], 1);
        float y = Rr.y + 6 * U + cap_h(s_small);
        legend_line(&y, Rr.x + 6 * U, C_F1, 0, "UKF S, 2\xcf\x83");
        legend_line(&y, Rr.x + 6 * U, C_F2, 0, "EKF HPH\xe1\xb5\x80+R, 2\xcf\x83");
        legend_line(&y, Rr.x + 6 * U, C_F3, 1, "h(\xcf\x87\xe1\xb5\xa2)");
        return;
    }
    if (A.id == A_IMM) {
        if (!A.finit) return;
        axes_box(&in);
        double o[6];
        path_eval(&A.tp[0], s, o);
        Rect save = vr;
        float svk = vk;
        double scx = vcx, scy = vcy;
        vr = in; vcx = o[0]; vcy = o[1]; vk = fminf(in.w, in.h) / 6000.0f;
        draw_grid();
        draw_truth(&A.tp[0], fmax(0, s - 60), s, 0.5f);
        draw_dets(s, 1);
        for (int j = 0; j < 2; j++) {
            Filt g = A.mo[j];
            f_predict(&g, s);
            double x[4] = {g.x[0], g.x[1], g.x[2], g.x[3]}, P2[3] = {g.P[0], g.P[1], g.P[6]};
            draw_est(x, P2, j ? C_F3 : C_F2, 0.35f + 0.65f * (float)A.mu[j], j ? "CT" : "CV", 15, 0, (j + 1) * 15 * U);
        }
        double x[4], P2[3];
        fdisp(&A.f[0], s, x, P2);
        draw_est(x, P2, C_F1, 1, "IMM", 15, 1, 0);
        vr = save; vk = svk; vcx = scx; vcy = scy;
        return;
    }
    if (A.id == A_PF) {
        axes_box(&in);
        if (A.nbrg < 1) return;
        /* x: bearing (deg) around the act's span; y: time, newest at the top */
        int n = A.nbrg < NBRG ? A.nbrg : NBRG;
        double c0 = A.brg_b[n - 1], lo = 1e9, hi = -1e9;
        for (int i = 0; i < n; i++) { double d = wrap_pi(A.brg_b[i] - c0); lo = fmin(lo, d); hi = fmax(hi, d); }
        double half = fmax((hi - lo) / 2 + 6 * DEGR, 15 * DEGR), mid = c0 + (hi + lo) / 2;
        float tspan = (float)A.len_s;
        #define BX(b) (in.x + in.w * (float)(0.5 + wrap_pi((b) - mid) / (2 * half)))
        #define TY(t) (in.y + in.h * (1 - (float)((t) / tspan)))
        for (int d = -180; d < 180; d += 10) {
            double b = d * DEGR;
            if (fabs(wrap_pi(b - mid)) > half) continue;
            float x = BX(b);
            seg(x, in.y, x, in.y + in.h, lw, pal[C_GRID], 0.6f);
            double cmp = fmod(90 - d + 720, 360);      /* compass */
            textf(s_small, x, in.y + in.h + 4 * U + cap_h(s_small), 0, pal[C_DIM], 1, "%03.0f", cmp);
        }
        static float B[2 * NBRG], C[2 * NBRG], T[2 * 200];
        for (int i = 0; i < n; i++) {
            float y = TY(A.brg_t[i]);
            B[2 * i] = BX(A.brg_pm[i] - 2 * A.brg_ps[i]); B[2 * i + 1] = y;
            C[2 * i] = BX(A.brg_pm[i] + 2 * A.brg_ps[i]); C[2 * i + 1] = y;
        }
        for (int i = 0; i + 1 < n; i++) {
            V(B[2 * i], B[2 * i + 1], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0); V(C[2 * i], C[2 * i + 1], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0);
            V(C[2 * i + 2], C[2 * i + 3], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0);
            V(B[2 * i], B[2 * i + 1], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0); V(C[2 * i + 2], C[2 * i + 3], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0);
            V(B[2 * i + 2], B[2 * i + 3], 0, 0, pal[C_F1], 0.22f, K_FLAT, 0);
        }
        int nt = 0;
        for (int i = 0; i < 200; i++) {
            double t = s * i / 199, oo[6], ot[6];
            path_eval(&A.own, t, oo); path_eval(&A.tp[0], t, ot);
            double b = atan2(ot[1] - oo[1], ot[0] - oo[0]);
            if (fabs(wrap_pi(b - mid)) > half) continue;
            T[2 * nt] = BX(b); T[2 * nt + 1] = TY(t); nt++;
        }
        poly(T, nt, 0, 1.3f * U, pal[C_TRUTH], 0.8f, NULL);
        for (int i = 0; i < n; i++) disk(BX(A.brg_b[i]), TY(A.brg_t[i]), 1.8f * U, pal[C_MEAS], 0.85f);
        #undef BX
        #undef TY
        for (int i = 0; i < A.nman; i++) {          /* own ship's turns */
            float y0 = in.y + in.h * (1 - A.man[i][0] / (float)opt_act_len), y1 = in.y + in.h * (1 - fminf(A.man[i][1], (float)rt) / (float)opt_act_len);
            if (A.man[i][0] > rt) continue;
            rectf(in.x, y1, in.w, y0 - y1, pal[C_TEXT], 0.05f);
            text(s_small, in.x + 4 * U, y0 - 3 * U, -1, "own ship turns", pal[C_DIM], 0.8f);
        }
        return;
    }
    /* B-scope */
    axes_box(&in);
    const Radar *rd = &A.rd;
    #define SBX(b) (in.x + in.w * (float)(fmod(90 - (b) / DEGR + 720, 360) / 360))
    #define SRY(r) (in.y + in.h * (1 - (float)((r) / rd->rmax)))
    for (int d = 0; d <= 360; d += 45) {
        float x = in.x + in.w * d / 360.0f;
        seg(x, in.y, x, in.y + in.h, lw, pal[C_GRID], 0.7f);
        textf(s_small, x, in.y + in.h + 4 * U + cap_h(s_small), 0, pal[C_DIM], 1, "%03d", d % 360);
    }
    for (int k = 10; k <= 60; k += 10) {
        float y = SRY(k * 1000.0);
        seg(in.x, y, in.x + in.w, y, lw, pal[C_GRID], 0.7f);
        textf(s_small, in.x - 5 * U, y + cap_h(s_small) / 2, 1, pal[C_DIM], 1, "%d", k);
    }
    double pers = rd->T * 1.5;
    for (int i = A.dhn - 1; i >= 0 && i >= A.dhn - 400; i--) {
        const Det *d = &A.dh[i % DH];
        if (d->t > s || s - d->t > pers) continue;
        float a = 1 - (float)((s - d->t) / pers);
        disk(SBX(d->b), SRY(d->r), 1.9f * U, pal[d->used == 2 ? C_MEAS : C_CLUT], a);
    }
    for (int i = 0; i < A.ntk; i++) {
        const Track *k = &A.tk[i];
        if (k->st != TK_CONF) continue;
        double x[4], P2[3];
        fdisp(&k->f, s, x, P2);
        double b = atan2(x[1], x[0]), rr = hypot(x[0], x[1]);
        float sx = SBX(b), sy = SRY(rr);
        ring(sx, sy, 4 * U, pal[C_F1], 1);
        textf(s_small, sx + 6 * U, sy - 5 * U, -1, pal[C_F1], 1, "T%02d", k->id);
    }
    float x = SBX(beam(s));
    seg(x, in.y, x, in.y + in.h, 1.4f * U, pal[C_SWEEP], 0.7f);
    #undef SBX
    #undef SRY
}

/* ---- whitened innovations ---- */
static void draw_innov(const Rect *r) {
    Rect in = {r->x + 16 * U, r->y + 34 * U, r->w - 32 * U, r->h - 64 * U};
    float sz = fminf(in.w, in.h), cx = in.x + in.w / 2, cy = in.y + in.h / 2, k = sz / 2 / 3.6f, lw = fmaxf(U, 1);
    seg(cx - sz / 2, cy, cx + sz / 2, cy, lw, pal[C_GRID], 1);
    int in2 = 0, n = A.inn_n;
    if (A.inn_m == 2) {
        seg(cx, cy - sz / 2, cx, cy + sz / 2, lw, pal[C_GRID], 1);
        for (int m = 1; m <= 3; m++) circle(cx, cy, m * k, lw, pal[C_DIM], 0.6f, m != 2);
        for (int i = 0; i < n; i++) {
            int idx = (A.inn_h - 1 - i + RING) % RING;
            float a = 1 - (float)i / RING, u = A.inn[idx][0], v = A.inn[idx][1];
            in2 += u * u + v * v < 4;
            disk(cx + clampf(u, -3.5f, 3.5f) * k, cy - clampf(v, -3.5f, 3.5f) * k, (i == 0 ? 3.0f : 1.9f) * U, pal[i == 0 ? C_TEXT : C_F1], 0.85f * a);
        }
    } else {
        for (int m = -3; m <= 3; m++) seg(cx + m * k, cy - sz / 2, cx + m * k, cy + sz / 2, lw, pal[m ? C_GRID : C_DIM], 0.8f);
        /* the standard normal, for comparison */
        float P[2 * 81];
        for (int i = 0; i <= 80; i++) {
            float u = -3.6f + 7.2f * i / 80;
            P[2 * i] = cx + u * k; P[2 * i + 1] = cy + sz / 2 - expf(-0.5f * u * u) * sz * 0.8f;
        }
        poly(P, 81, 0, 1.4f * U, pal[C_DIM], 0.8f, NULL);
        for (int i = 0; i < n; i++) {
            int idx = (A.inn_h - 1 - i + RING) % RING;
            float u = A.inn[idx][0], a = 1 - (float)i / RING;
            in2 += fabsf(u) < 2;
            float jy = cy + sz / 2 - 4 * U - (float)((i * 37) % 97) / 97.0f * sz * 0.75f;
            disk(cx + clampf(u, -3.5f, 3.5f) * k, jy, (i == 0 ? 3.0f : 1.9f) * U, pal[i == 0 ? C_TEXT : C_F1], 0.85f * a);
        }
    }
    if (n) textf(s_small, r->x + 12 * U, r->y + r->h - 10 * U, -1, pal[C_DIM], 1, "inside 2\xcf\x83: %.0f%% of %d  (expect %.0f%%)",
                 100.0 * in2 / n, n, A.inn_m == 2 ? 86.5 : 95.4);
}

/* ---- a table of the filters, or of the tracks ---- */
static void draw_table(const Rect *r, double s) {
    float x = r->x + 12 * U, y = r->y + 40 * U + cap_h(s_hud), lh = cap_h(s_hud) * 2.0f;
    static const float cx[] = {0, 0.17f, 0.32f, 0.47f, 0.6f, 0.73f, 0.87f};
    float w = r->w - 24 * U;
    if (A.id == A_MTT) {
        static const char *hd[] = {"TRACK", "STATE", "RANGE", "BRG", "SPEED", "\xcf\x83 POS", "HITS"};
        for (int c = 0; c < 7; c++) text(s_small, x + w * cx[c], y, -1, hd[c], pal[C_DIM], 1);
        y += lh;
        int rows = 0, maxr = (int)((r->y + r->h - y) / lh);
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i < A.ntk && rows < maxr; i++) {
                const Track *k = &A.tk[i];
                if (pass == 0 ? k->st != TK_CONF && k->st != TK_DEAD : !(k->st == TK_TENT && k->hits >= 2)) continue;
                double xx[4], P2[3];
                fdisp(&k->f, s, xx, P2);
                int col = k->st == TK_CONF ? C_F1 : k->st == TK_DEAD ? C_WARN : C_F2;
                char b[32];
                if (k->id) snprintf(b, sizeof b, "T%02d", k->id); else snprintf(b, sizeof b, "--");
                text(s_hud, x, y, -1, b, pal[col], 1);
                text(s_hud, x + w * cx[1], y, -1, k->st == TK_CONF ? "CONF" : k->st == TK_DEAD ? "LOST" : "TENT", pal[col], 1);
                textf(s_hud, x + w * cx[2], y, -1, pal[C_TEXT], 1, "%.1f", hypot(xx[0], xx[1]) / 1000);
                textf(s_hud, x + w * cx[3], y, -1, pal[C_TEXT], 1, "%03.0f", fmod(90 - atan2(xx[1], xx[0]) / DEGR + 720, 360));
                textf(s_hud, x + w * cx[4], y, -1, pal[C_TEXT], 1, "%.0f", hypot(xx[2], xx[3]));
                textf(s_hud, x + w * cx[5], y, -1, pal[C_TEXT], 1, "%.0f", sqrt(P2[0] + P2[2]));
                textf(s_hud, x + w * cx[6], y, -1, pal[C_TEXT], 1, "%d", k->hits);
                y += lh;
                rows++;
            }
        return;
    }
    static const char *hd[] = {"FILTER", "ERROR m", "\xcf\x83 POS m", "NIS", "SPEED", "COURSE", "UPDATES"};
    for (int c = 0; c < 7; c++) text(s_small, x + w * cx[c], y, -1, hd[c], pal[C_DIM], 1);
    y += lh;
    double o[6];
    path_eval(&A.tp[0], s, o);
    text(s_hud, x, y, -1, "truth", pal[C_TRUTH], 1);
    textf(s_hud, x + w * cx[4], y, -1, pal[C_TRUTH], 1, "%.1f", o[5]);
    textf(s_hud, x + w * cx[5], y, -1, pal[C_TRUTH], 1, "%03.0f", fmod(90 - o[4] / DEGR + 720, 360));
    y += lh;
    if (!A.finit && !(A.id == A_PF && A.nbrg)) return;
    for (int k = 0; k < A.nf; k++) {
        double xx[4], P2[3];
        fdisp(&A.f[k], s, xx, P2);
        int col = A.fcol[k];
        text(s_hud, x, y, -1, A.fname[k], pal[col], 1);
        textf(s_hud, x + w * cx[1], y, -1, pal[C_TEXT], 1, "%.0f", hypot(xx[0] - o[0], xx[1] - o[1]));
        textf(s_hud, x + w * cx[2], y, -1, pal[C_TEXT], 1, "%.0f", sqrt(P2[0] + P2[2]));
        textf(s_hud, x + w * cx[3], y, -1, pal[C_TEXT], 1, "%.2f", A.f[k].nis);
        textf(s_hud, x + w * cx[4], y, -1, pal[C_TEXT], 1, "%.1f", hypot(xx[2], xx[3]));
        textf(s_hud, x + w * cx[5], y, -1, pal[C_TEXT], 1, "%03.0f", fmod(90 - atan2(xx[3], xx[2]) / DEGR + 720, 360));
        textf(s_hud, x + w * cx[6], y, -1, pal[C_TEXT], 1, "%d", A.f[k].nupd);
        y += lh;
    }
    y += lh * 0.5f;
    if (A.id == A_IMM) {
        textf(s_hud, x, y, -1, pal[C_F3], 1, "\xce\xbc(CV) %.2f   \xce\xbc(CT) %.2f   \xcf\x89 est %+.2f\xc2\xb0/s",
              A.mu[0], A.mu[1], (A.mu[0] * A.mo[0].x[4] + A.mu[1] * A.mo[1].x[4]) / DEGR);
    } else if (A.id == A_PF) {
        textf(s_hud, x, y, -1, pal[C_F1], 1, "N_eff %.0f of %d   resampled %d times", A.neff, NP, A.nres);
    }
}

/* the IMM's mode probability bars, right of its chart */
static void draw_bars(const Rect *r) {
    float x0 = r->x + r->w * 0.72f, w = r->w * 0.26f - 8 * U, y = r->y + 38 * U, bh = fminf((r->h - 60 * U) / 3, 22 * U);
    static const char *nm[] = {"CV", "CT"};
    for (int j = 0; j < 2; j++) {
        float yy = y + j * (bh + 14 * U);
        text(s_hudb, x0, yy + bh / 2 + cap_h(s_hudb) / 2, -1, nm[j], pal[j ? C_F3 : C_F2], 1);
        float bx = x0 + 30 * U, bw = w - 30 * U - 40 * U;
        rectf(bx, yy, bw, bh, pal[C_GRID], 0.8f);
        rectf(bx, yy, bw * (float)A.mu[j], bh, pal[j ? C_F3 : C_F2], 0.85f);
        textf(s_small, bx + bw + 5 * U, yy + bh / 2 + cap_h(s_small) / 2, -1, pal[C_TEXT], 1, "%.2f", A.mu[j]);
    }
    double w_est = (A.mu[0] * A.mo[0].x[4] + A.mu[1] * A.mo[1].x[4]) / DEGR, o[6], o2[6];
    path_eval(&A.tp[0], A.s, o);
    path_eval(&A.tp[0], A.s + 0.5, o2);
    double w_true = wrap_pi(o2[4] - o[4]) / 0.5 / DEGR;
    float yy = y + 2 * (bh + 14 * U) + cap_h(s_small);
    textf(s_small, x0, yy, -1, pal[C_TEXT], 1, "\xcf\x89 est   %+5.1f\xc2\xb0/s", w_est);
    textf(s_small, x0, yy + cap_h(s_small) * 1.8f, -1, pal[C_TRUTH], 1, "\xcf\x89  %+5.1f\xc2\xb0/s", w_true);
}

static void draw_clock(const Rect *r, double wt) {
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    char b[32];
    strftime(b, sizeof b, "%H:%M", &tm);
    float x = r->x + r->w - 12 * U, y = r->y + r->h * 0.42f + cap_h(s_clock) * 0.35f;
    if (role == R_FRONT && spanned) rectf(r->x, r->y, r->w, r->h, pal[C_BG], 0.6f);
    text(s_clock, x, y, 1, b, pal[C_TEXT], 1);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    snprintf(b, sizeof b, "AWAY %ld:%02ld:%02ld", away / 3600, away / 60 % 60, away % 60);
    text(s_clocks, x, y + cap_h(s_clocks) * 2.4f, 1, b, pal[C_DIM], 1);
    strftime(b, sizeof b, "%a %d %b", &tm);
    text(s_clocks, r->x + 12 * U, y, -1, b, pal[C_DIM], 1);
}

static void draw_title(const Rect *r) {
    frame(r, NULL);
    float pad = 14 * U, x = r->x + pad, y = r->y + pad, w = r->w - 2 * pad;
    y += cap_h(s_head);
    textf(s_head, x, y, -1, pal[C_F1], 1, "ACT %d / %d", A.id + 1, NACT);
    y += 12 * U + cap_h(s_title);
    float lh = styles[s_title].px * 1.2f;
    int nl = text_wrap(s_title, x, y, w, lh, ACT_NAME[A.id], pal[C_TEXT], 1, 1);
    y += (nl - 1) * lh + 8 * U + styles[s_desc].px;
    float dl = styles[s_desc].px * 1.35f;
    nl = text_wrap(s_desc, x, y, w, dl, ACT_LINE[A.id], pal[C_DIM], 1, 1);
    y += (nl - 1) * dl + 12 * U;
    seg(x, y, x + w, y, fmaxf(U, 1), pal[C_FRAME], 1);
    int ne = 0;
    while (ne < 6 && EQS[A.id][ne]) ne++;
    float base = 21 * U;
    eq_s[0] = style(F_MATH, base); eq_s[1] = style(F_MATH, base * 0.7f);
    float widest = 1;
    for (int i = 0; i < ne; i++) widest = fmaxf(widest, eq(0, 0, EQS[A.id][i], pal[C_TEXT], 1, 0));
    float avail = r->y + r->h - pad - y, sc = fminf(1.7f, fminf(w / widest, avail / (ne * 1.75f * base)));
    float px = fmaxf(base * sc, 9);
    eq_s[0] = style(F_MATH, px); eq_s[1] = style(F_MATH, px * 0.7f);
    float el = fminf(px * 2.0f, avail / ne);
    y += el * 0.5f + px * 0.6f;
    for (int i = 0; i < ne; i++) eq(x, y + i * el, EQS[A.id][i], pal[C_TEXT], 1, 1);
}

/* ==================================================================== */
/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aP;
    layout(location = 1) in vec2 aL;
    layout(location = 2) in vec4 aC;
    layout(location = 3) in vec2 aK;
    uniform vec2 uRes;
    out vec2 vL;
    out vec4 vC;
    flat out vec2 vK;
    void main() {
        vL = aL; vC = aC; vK = aK;
        gl_Position = vec4(aP.x / uRes.x * 2.0 - 1.0, 1.0 - aP.y / uRes.y * 2.0, 0.0, 1.0);
    });
/* kinds: 0 flat, 1 AA line (u across, p half-width), 2 disk (p radius),
 * 3 thin ring, 4 text (u v texels), 5 soft blob, 6 soft line */
static const char *FS = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    in vec2 vL;
    in vec4 vC;
    flat in vec2 vK;
    out vec4 o;
    void main() {
        float a = vC.a;
        int k = int(vK.x + 0.5);
        if (k == 1) a *= clamp(vK.y + 0.5 - abs(vL.x), 0.0, 1.0);
        else if (k == 2) a *= clamp(vK.y + 0.5 - length(vL), 0.0, 1.0);
        else if (k == 3) a *= clamp(1.2 - abs(length(vL) - vK.y), 0.0, 1.0);
        else if (k == 4) a *= texture(uAtlas, vL / vec2(2048.0)).r;
        else if (k == 5) { float d = length(vL) / vK.y; a *= exp(-d * d * 4.0) * step(d, 1.0); }
        else if (k == 6) { float d = clamp(1.0 - abs(vL.x) / (vK.y + 1.0), 0.0, 1.0); a *= d * d; }
        if (a < 0.002) discard;
        o = vec4(vC.rgb, a);
    });
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(p, 0.0, 1.0);
    });
static const char *FS_BLIT = "#version 330 core\n" GLSL(
    uniform sampler2D uTex;
    out vec4 o;
    void main() { o = vec4(texelFetch(uTex, ivec2(gl_FragCoord.xy), 0).rgb, 1.0); });

static GLuint prog, prog_blit, vao, vbo, vao_empty, tex_atlas, tex_static, fbo_static;
static int st_w, st_h, st_act = -1, lay_w, lay_h;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "kalman: shader:\n%s\n", log);
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

static void gl_init(void) {
    prog = mk_prog(VS, FS);
    prog_blit = mk_prog(VS_FULL, FS_BLIT);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_empty);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(4 * sizeof(float)));
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)(8 * sizeof(float)));
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray(i);
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, AH, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenTextures(1, &tex_static);
    glGenFramebuffers(1, &fbo_static);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}

/* draw what has been queued */
static void flush(void) {
    if (!vn) return;
    if (atlas_dirty) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, AW, ay + ashelf + 2 < AH ? ay + ashelf + 2 : AH, GL_RED, GL_UNSIGNED_BYTE, atlas);
        atlas_dirty = 0;
    }
    glUseProgram(prog);
    glUniform2f(UL(prog, "uRes"), (float)W, (float)H);
    glUniform1i(UL(prog, "uAtlas"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vn * sizeof(Vtx)), vb, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, vn);
    vn = 0;
}

/* everything that changes only with the size or the act, drawn once */
static void build_static(void) {
    glBindTexture(GL_TEXTURE_2D, tex_static);
    if (st_w != W || st_h != H) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_static);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_static, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_static);
    glViewport(0, 0, W, H);
    glClearColor(pal[C_BG][0], pal[C_BG][1], pal[C_BG][2], 1);
    glClear(GL_COLOR_BUFFER_BIT);
    vn = 0;
    for (int i = 0; i < npan; i++) {
        const Rect *r = &pan[i].r;
        switch (pan[i].kind) {
        case P_PLOT:
            frame(r, NULL);
            rectf(r->x + 1, r->y + 1, r->w - 2, r->h - 2, pal[C_BG], paper() ? 0.35f : 0.6f);
            break;
        case P_TITLE: draw_title(r); break;
        case P_NIS: chart_static(r, &A.ch[0]); break;
        case P_ERR: chart_static(r, &A.ch[1]); break;
        case P_ACT: chart_static(r, &A.ch[2]); break;
        case P_MATS: draw_mats_static(r); break;
        case P_MEAS: frame(r, meas_head()); break;
        case P_INNOV: frame(r, A.inn_m == 1 || A.id == A_PF ? "WHITENED INNOVATIONS  \xc2\xb7  (\xce\xb2 \xe2\x88\x92 \xce\xb2\xcc\x82)/\xe2\x88\x9aS vs N(0,1)"
                                                         : "WHITENED INNOVATIONS  \xc2\xb7  L\xe2\x81\xbb\xc2\xb9\xce\xbd, S = LL\xe1\xb5\x80"); break;
        case P_TABLE: frame(r, A.id == A_MTT ? "TRACK TABLE" : "FILTERS"); break;
        case P_CLOCK: if (!(role == R_FRONT && spanned)) frame(r, NULL); break;
        }
    }
    flush();
    st_w = W; st_h = H; st_act = A.id;
}

/* s: sim seconds into the act, rt: real seconds into it, wt: wall clock */
static void render(double s, double rt, double wt, GLuint target) {
    if (W != lay_w || H != lay_h) { layout(); lay_w = W; lay_h = H; st_act = -1; }
    if (st_act != A.id || st_w != W || st_h != H) build_static();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_blit);
    glUniform1i(UL(prog_blit, "uTex"), 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_static);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_BLEND);
    vn = 0;
    for (int i = 0; i < npan; i++) {
        const Rect *r = &pan[i].r;
        switch (pan[i].kind) {
        case P_PLOT:
            flush();
            glEnable(GL_SCISSOR_TEST);
            glScissor((int)r->x, H - (int)(r->y + r->h), (int)r->w, (int)r->h);
            draw_plot(r, s, rt);
            if (role == R_FRONT) draw_card(r, rt);
            flush();
            glDisable(GL_SCISSOR_TEST);
            break;
        case P_NIS: chart_dynamic(r, &A.ch[0], rt); break;
        case P_ERR: chart_dynamic(r, &A.ch[1], rt); break;
        case P_ACT: chart_dynamic(r, &A.ch[2], rt); if (A.ch[2].bars) draw_bars(r); break;
        case P_MATS: draw_mats(r, s); break;
        case P_MEAS:
            flush();
            glEnable(GL_SCISSOR_TEST);
            glScissor((int)r->x, H - (int)(r->y + r->h), (int)r->w, (int)r->h);
            draw_meas(r, s, rt);
            flush();
            glDisable(GL_SCISSOR_TEST);
            break;
        case P_INNOV: draw_innov(r); break;
        case P_TABLE: draw_table(r, s); break;
        case P_CLOCK: draw_clock(r, wt); break;
        }
    }
    flush();
}

/* ---- the show's clock: act and moment from the time since the epoch ---- */
static int act_live;
static void sim_to(double t) {
    if (t < 0) t = 0;
    long cyc = (long)floor(t / opt_act_len);
    int id = opt_act >= 0 ? opt_act : (int)((cyc + opt_seed % NACT) % NACT);
    if (!act_live || cyc != A.cyc || id != A.id) { act_setup(cyc, id); act_live = 1; }
    act_advance((t - cyc * opt_act_len) * A.TS);
}

/* ---- --check: filter consistency ---- */
static void check_linear(int model, double q) {
    enum { M = 200, K = 120 };
    int n = model_n(model);
    static double anees[K], anis[K];
    memset(anees, 0, sizeof anees); memset(anis, 0, sizeof anis);
    Rng r = {777 + (uint64_t)model};
    double R[4] = {GPS_SIG * GPS_SIG, 0, 0, GPS_SIG * GPS_SIG}, dt = 1;
    for (int m = 0; m < M; m++) {
        double x[MX] = {0, 0, 12, -4, 0, 0}, F[MX * MX], Q[MX * MX], L[MX * MX], P0[MX * MX] = {0}, L0[MX * MX];
        Filt f;
        f_init(&f, model, q, 0);
        for (int i = 0; i < n; i++) P0[i * n + i] = i < 2 ? R[0] : i < 4 ? 10 * 10 : 2 * 2;
        chol(P0, L0, n);
        double w[MX];
        for (int i = 0; i < n; i++) w[i] = rn_n(&r);
        for (int i = 0; i < n; i++) {
            f.x[i] = x[i];
            for (int j = 0; j <= i; j++) f.x[i] += L0[i * n + j] * w[j];
        }
        memcpy(f.P, P0, sizeof(double) * n * n);
        model_F(model, x, dt, F);
        model_Q(model, dt, q, 0, Q);
        chol(Q, L, n);
        for (int k = 0; k < K; k++) {
            double y[MX];
            mm(F, x, y, n, n, 1);
            for (int i = 0; i < n; i++) w[i] = rn_n(&r);
            for (int i = 0; i < n; i++) {
                x[i] = y[i];
                for (int j = 0; j <= i; j++) x[i] += L[i * n + j] * w[j];
            }
            double z[2] = {x[0] + GPS_SIG * rn_n(&r), x[1] + GPS_SIG * rn_n(&r)};
            f_predict(&f, (k + 1) * dt);
            f_update_pos(&f, z, R);
            anees[k] += nees(&f, x, n) / M;
            anis[k] += f.nis / M;
        }
    }
    double lo_e = chi2inv(0.025, n * M) / M, hi_e = chi2inv(0.975, n * M) / M, lo_n = chi2inv(0.025, 2 * M) / M, hi_n = chi2inv(0.975, 2 * M) / M;
    int in_e = 0, in_n = 0;
    double me = 0, mn = 0;
    for (int k = 0; k < K; k++) {
        in_e += anees[k] >= lo_e && anees[k] <= hi_e;
        in_n += anis[k] >= lo_n && anis[k] <= hi_n;
        me += anees[k] / K; mn += anis[k] / K;
    }
    double glo_e = chi2inv(0.025, n * M * K) / (M * K), ghi_e = chi2inv(0.975, n * M * K) / (M * K);
    double glo_n = chi2inv(0.025, 2 * M * K) / (M * K), ghi_n = chi2inv(0.975, 2 * M * K) / (M * K);
    printf("  KF-%s, matched model (%d Monte Carlo runs x %d steps):\n", model == M_CA ? "CA" : "CV", M, K);
    printf("    ANEES per step: %d/%d steps inside [%.2f, %.2f] (95%%); overall %.3f, bounds [%.3f, %.3f] %s\n",
           in_e, K, lo_e, hi_e, me, glo_e, ghi_e, me >= glo_e && me <= ghi_e ? "OK" : "OUTSIDE");
    printf("    ANIS  per step: %d/%d steps inside [%.2f, %.2f] (95%%); overall %.3f, bounds [%.3f, %.3f] %s\n",
           in_n, K, lo_n, hi_n, mn, glo_n, ghi_n, mn >= glo_n && mn <= ghi_n ? "OK" : "OUTSIDE");
}

static void run_check(void) {
    printf("kalman --check\n");
    printf("chi2 quantiles: chi2(2) 95%% [%.4f, %.3f] (exact 0.0506, 7.378); chi2(4) [%.3f, %.2f] (exact 0.484, 11.14)\n",
           chi2inv(0.025, 2), chi2inv(0.975, 2), chi2inv(0.025, 4), chi2inv(0.975, 4));
    printf("1. linear filters against their own model\n");
    check_linear(M_CV, 1.5);
    check_linear(M_CA, 1.0);
    printf("2. the acts as shown (manoeuvring truth the filters don't model; time-averaged, 8 seeds each)\n");
    static const char *dofs[NACT] = {"2", "2", "2", "1", "2"};
    for (int id = 0; id < NACT; id++) {
        double nis[3] = {0}, ne[3] = {0};
        int nn[3] = {0}, nne[3] = {0}, swaps = 0, ids = 0, maxres = 0;
        double t0 = now();
        for (uint32_t sd = 1; sd <= 8; sd++) {
            opt_seed = sd;
            act_setup(0, id);
            act_advance(A.len_s);
            for (int k = 0; k < 3; k++) { nis[k] += A.acc_nis[k]; nn[k] += A.acc_nn[k]; ne[k] += A.acc_nees[k]; nne[k] += A.acc_ne[k]; }
            swaps += A.nswaps; ids += A.next_id - 1;
            if (A.nres > maxres) maxres = A.nres;
        }
        printf("  %d %-38s (%.0f ms for 8 acts)\n", id + 1, ACT_NAME[id], (now() - t0) * 1000);
        int nf = id == A_MTT ? 1 : A.nf;
        for (int k = 0; k < nf; k++)
            printf("      %-8s mean NIS %6.2f (dof %s)   mean NEES %7.2f (dof 4)   over %d updates\n",
                   id == A_MTT ? "tracks" : A.fname[k], nn[k] ? nis[k] / nn[k] : 0, dofs[id], nne[k] ? ne[k] / nne[k] : 0, nn[k]);
        if (id == A_MTT) printf("      %d track IDs issued over 8 acts (8 x 8 targets), %d swaps flagged\n", ids, swaps);
        if (id == A_PF) printf("      up to %d resamplings in an act\n", maxres);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "kalman");
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
        "usage: kalman [--window | --root | --layer] [--seed N] [--fps N] [--palette tactical|night|paper|mono]\n"
        "              [--act 1-5] [--act-len SECONDS] [--font FAMILY] [--math-font FAMILY]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check]\n");
    exit(1);
}

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
 * gets the tactical picture; the rest, left to right, take the filter
 * internals and the sensor view in turn. */
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
    return rank % 2 ? R_SENSOR : R_INTERNALS;
}

static void save_ppm(const char *path, GLuint fbo) {
    unsigned char *px = malloc((size_t)W * H * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
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
        else if (!strcmp(a, "-math-font") && i + 1 < argc) opt_math = argv[++i];
        else if (!strcmp(a, "-act") && i + 1 < argc) { opt_act = atoi(argv[++i]) - 1; if (opt_act < 0 || opt_act >= NACT) usage(); }
        else if (!strcmp(a, "-act-len") && i + 1 < argc) { opt_act_len = strtod(argv[++i], NULL); if (opt_act_len < 20) opt_act_len = 20; }
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2 || off_w < 64 || off_h < 64) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            for (opt_pal = 0; opt_pal < NPAL && strcmp(n, PALS[opt_pal].name); opt_pal++) {}
            if (opt_pal == NPAL) usage();
        }
        else usage();
    }
    if (opt_check) { run_check(); return 0; }
    if (span_me >= span_n) span_me = -1;
    spanned = span_me >= 0 && span_n > 1;
    role = pick_role();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    /* the palette: from the seed unless asked for */
    if (opt_pal < 0) opt_pal = (int)(((opt_seed * 0x9E3779B1u) >> 16) % NPAL);
    set_palette(opt_pal);
    int offscreen = opt_shot || opt_bench > 0;
    if (opt_at < 0) opt_at = 45;
    fonts_init();

    /* --epoch: when the show started, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (offscreen && !opt_epoch) t_wall0 -= opt_at;
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
        XStoreName(dpy, win, "Kalman");
        XClassHint chint = {"kalman", "Kalman"};
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
            double t = opt_at;
            sim_to(t);
            render(A.s, A.s / A.TS, t_wall0 + t, fbo);
            glFinish();
            save_ppm(opt_shot, fbo);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        sim_to(opt_at);
        render(A.s, A.s / A.TS, t_wall0 + opt_at, fbo);      /* warm-up: static layer, glyphs */
        glFinish();
        double sum = 0, worst = 0, gsum = 0, gworst = 0, csum = 0, rsum = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 30.0;
            sim_to(t);
            double c0 = now();
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(A.s, A.s / A.TS, t_wall0 + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            rsum += (now() - c0) * 1000;
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000, gms = ns / 1e6;
            csum += (c0 - f0) * 1000;
            sum += ms; gsum += gms;
            if (ms > worst) worst = ms;
            if (gms > gworst) gworst = gms;
        }
        printf("kalman: act %d, %dx%d, %d frames: GPU avg %.3f ms (worst %.3f); frame wall avg %.3f ms (worst %.3f): sim %.3f ms, drawing (CPU) %.3f ms\n",
               A.id + 1, W, H, opt_bench, gsum / opt_bench, gworst, sum / opt_bench, worst, csum / opt_bench, rsum / opt_bench);
        printf("kalman: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
        sim_to(el);
        render(A.s, A.s / A.TS, wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
