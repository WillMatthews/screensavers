/*
 * kalman — state estimation made visible, on a tactical display.
 *
 * Simulated targets (a car on a winding road, aircraft flying legs and
 * coordinated turns, a ship making way) are watched by noisy sensors: GPS
 * fixes, range-bearing radars (scans, no rotating beam drawn), missed detections and
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
static int opt_root, opt_layer, opt_bench, opt_check, opt_resize;
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
enum { M_CV, M_CA, M_CV5, M_CT, M_CVB };      /* state x y vx vy [ax ay | omega | odometer bias bx by] */
static int model_n(int m) { return m == M_CV ? 4 : m == M_CA || m == M_CVB ? 6 : 5; }

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
    if (model == M_CVB) Q[28] = Q[35] = qw * dt;     /* the bias random-walks */
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
enum { A_KF, A_UKF, A_IMM, A_PF, A_MTT, A_RTS, A_FUSE, A_NUM, A_SLAM, NACT };
static const char *ACT_NAME[NACT] = {
    "KF vs raw GPS", "EKF vs UKF on a turning target", "IMM through manoeuvres",
    "Particle filter, bearings-only", "Multi-target association in clutter",
    "Smoothing: Rauch\xe2\x80\x93Tung\xe2\x80\x93Striebel", "Sensor fusion in information form",
    "Square root vs naive: float32", "EKF-SLAM: mapping as you go"};
static const char *ACT_LINE[NACT] = {
    "a car on winding roads, one GPS fix a second (\xcf\x83 5 m), filtered with constant-velocity and constant-acceleration models",
    "one radar, range and bearing: the same CV model linearised (EKF) or carried through by sigma points (UKF)",
    "an aircraft flying legs and hard turns; constant-velocity and coordinated-turn models, mixed by their probabilities",
    "a listening ship hears only bearings; 2000 hypotheses of range, course and speed, culled as it manoeuvres",
    "a radar in clutter with missed detections: gates, global-nearest-neighbour assignment, M-of-N track management",
    "the same car, GPS through two tunnels: the forward filter, and every fix the whole past re-estimated backwards; the estimate eight seconds late is the fixed-lag smoother",
    "a rover with GPS (lost in an urban canyon), wheel odometry with a wandering bias and two range beacons; information adds, and the base station's track, which has swallowed our own reports, is merged by covariance intersection",
    "a coarse cue (\xcf\x83 30 m) hands a drone to a laser tracker (\xcf\x83 1 mm): one update shrinks P a billionfold, past float32's seven digits",
    "a robot drives a loop among landmarks it has never seen: range and bearing, odometry between, one state for robot and map, and the map pulls together when it comes round again"};
static const float ACT_TS[NACT] = {1, 3, 3, 15, 3, 2, 1, 1, 2};   /* sim seconds per real second */

/* palette slots */
enum { C_BG, C_PANEL, C_GRID, C_FRAME, C_TEXT, C_DIM, C_TRUTH, C_MEAS, C_CLUT, C_F1, C_F2, C_F3, C_WARN, NC };

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
typedef struct { double x, y, T, sr, sb, pd, lam, rmin, rmax, ph; } Radar;

#define MAXT 10
#define MAXQ 256
#define DH 1024
#define NP 2000
#define NBRG 400
#define MAXTRK 320
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

    Radar rd; int has_radar, scan;
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

    Series ser[16]; Chart ch[3];
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

/* when in scan n the radar looks at target i: each target keeps its own
 * phase within the scan (no beam is drawn; the timing just spreads the
 * detections out so they arrive one by one, not all at once) */
static double look_time(int i, int n) {
    double f = A.rd.ph + i * 0.3819660112501051;
    return (n + f - floor(f)) * A.rd.T;
}

static void det_R(const Det *d, double *R) {
    double c = cos(d->b), s = sin(d->b), a = A.rd.sr * A.rd.sr, e = d->r * A.rd.sb;
    e *= e;
    R[0] = c * c * a + s * s * e; R[1] = R[2] = c * s * (a - e); R[3] = s * s * a + c * c * e;
}

/* one scan: each target at its look time (if detected), and Poisson clutter
 * uniform over the coverage and at random moments within the scan */
static void radar_scan(int n) {
    Radar *rd = &A.rd;
    A.qn = A.qi = 0;
    for (int i = 0; i < A.nt; i++) {
        double o[6], t = look_time(i, n);
        path_eval(&A.tp[i], t, o);
        double b = atan2(o[1] - rd->y, o[0] - rd->x), r = hypot(o[0] - rd->x, o[1] - rd->y);
        double u = rn_u(&A.rm), nr = rn_n(&A.rm), nb = rn_n(&A.rm);
        if (t < A.tb[i] || t > A.te[i] || r < rd->rmin || r > rd->rmax || u > rd->pd) continue;
        if (A.qn < MAXQ) A.q[A.qn++] = add_det(t, r + rd->sr * nr, wrap_pi(b + rd->sb * nb), i);
    }
    int nc = rd->lam > 0 ? rn_poisson(&A.rm, rd->lam) : 0;
    for (int c = 0; c < nc && A.qn < MAXQ; c++) {
        double r = sqrt(rn_range(&A.rm, rd->rmin * rd->rmin, rd->rmax * rd->rmax));
        double b = rn_range(&A.rm, -PI, PI), t = (n + rn_u(&A.rm)) * rd->T;
        A.q[A.qn++] = add_det(t, r, b, -1);
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

/* the end of scan n: global-nearest-neighbour assignment of its detections,
 * confirmed tracks first, then tentative ones; the rest start tracks */
static void mtt_scan(int n) {
    double p = (n + 1) * A.rd.T;
    int nc = A.qn, *cand = A.q;
    for (int pass = 0; pass < 2; pass++) {
        int due[MAXTRK], nd = 0;
        for (int i = 0; i < A.ntk; i++)
            if (A.tk[i].st == (pass ? TK_TENT : TK_CONF)) due[nd++] = i;
        if (!nd) continue;
        int m = nc + nd, *ans = malloc(sizeof(int) * nd);
        double *cost = malloc(sizeof(double) * nd * m), *d2s = malloc(sizeof(double) * nd * (nc + 1));
        for (int r = 0; r < nd; r++) {
            for (int c = 0; c < nc; c++) {
                Det *d = &A.dh[cand[c]];
                double d2 = d->used ? 1e9 : gate_d2(&A.tk[due[r]].f, d, NULL);
                d2s[r * (nc + 1) + c] = d2;
                cost[r * m + c] = d2 < GATE ? d2 : 1e6;
            }
            for (int c = 0; c < nd; c++) cost[r * m + nc + c] = c == r ? GATE : 1e6;
        }
        hungarian(nd, m, cost, ans);
        if (pass == 0) {               /* keep (part of) the problem for the display */
            int rows[6], nr = 0, cols[8], ncol = 0;
            for (int r = 0; r < nd && nr < 6; r++) {
                int any = 0;
                for (int c = 0; c < nc; c++) any |= d2s[r * (nc + 1) + c] < GATE * 4;
                if (any) rows[nr++] = r;
            }
            for (int c = 0; c < nc && ncol < 8; c++) {
                int any = 0;
                for (int k = 0; k < nr; k++) any |= d2s[rows[k] * (nc + 1) + c] < GATE * 4;
                if (any) cols[ncol++] = c;
            }
            A.cost_r = nr; A.cost_c = ncol; A.cost_t = p;
            for (int k = 0; k < nr; k++) {
                int r = rows[k];
                A.cost_id[k] = A.tk[due[r]].id;
                A.cost_asg[k] = -1;
                for (int c = 0; c < ncol; c++) {
                    A.cost[k][c] = (float)d2s[r * (nc + 1) + cols[c]];
                    if (ans[r] == cols[c]) A.cost_asg[k] = c;
                }
            }
        }
        for (int r = 0; r < nd; r++) {
            Track *k = &A.tk[due[r]];
            int c = ans[r];
            if (c >= 0 && c < nc && d2s[r * (nc + 1) + c] < GATE && !A.dh[cand[c]].used) {
                trk_update(k, &A.dh[cand[c]]);
                k->f.t_upd = p;            /* ease the update in now, when it is made */
            } else { k->miss++; k->nsc++; }
            if (k->st == TK_TENT) {
                if (k->hits >= 3) { k->st = TK_CONF; k->id = A.next_id++; k->t_conf = p; }
                else if (k->miss >= 2 || k->nsc >= 5) k->st = -1;
            } else if (k->miss >= 3) { k->st = TK_DEAD; k->t_dead = p; }
            double sp = sqrt(k->f.P[0] + k->f.P[5]);
            if (k->st == TK_CONF && sp > 6000) { k->st = TK_DEAD; k->t_dead = p; }
        }
        free(cost); free(d2s); free(ans);
    }
    /* detections nobody wanted start tentative tracks */
    for (int c = 0; c < nc; c++) {
        Det *d = &A.dh[cand[c]];
        if (!d->used) { trk_new(d); d->used = 1; }
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
    {                                  /* counts and the RMS error */
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

/* ---------------- act 6: the Rauch-Tung-Striebel smoother ---------------- */
/* A car with GPS (and two outages, tunnels), a forward Kalman filter, and on
 * every fix the whole history smoothed backwards with the data so far; the
 * estimate of eight seconds ago is the fixed-lag smoother's. */
#define RK 400
#define RTS_LAG 8
#define RTS_SIG 6.0
static struct {
    int n, n0;                          /* steps so far (one a second); before the latest */
    double t[RK], xf[RK][4], Pf[RK][16], xp[RK][4], Pp[RK][16];
    double xs[RK][4], Ps[RK][16], C[RK][16], xs0[RK][4], Ps0[RK][16];
    double xl[RK][4], Pl[RK][16];       /* the fixed-lag estimate of step k, made at k + L */
    double tr[RK][4];                   /* truth at each step */
    int fix[RK];
    double zx[RK], zy[RK];
    double out[2][2];                   /* the outages, sim s */
    double se[3], sn[3]; int ne[3];     /* squared errors, NEES: filter, lag, smoothed */
} RT;

static int rts_outage(double t) {
    for (int i = 0; i < 2; i++) if (t >= RT.out[i][0] && t < RT.out[i][1]) return 1;
    return 0;
}

/* the backward pass over steps 0..n-1: x_{k|N}, P_{k|N} and the gains C_k */
static void rts_smooth(int n, double (*xf)[4], double (*Pf)[16], double (*xp)[4], double (*Pp)[16],
                       double (*xs)[4], double (*Ps)[16], double (*C)[16], const double *F) {
    memcpy(xs[n - 1], xf[n - 1], sizeof xs[0]);
    memcpy(Ps[n - 1], Pf[n - 1], sizeof Ps[0]);
    for (int k = n - 2; k >= 0; k--) {
        double PFt[16], Pi[16], d[4], D[16], T[16];
        mmT(Pf[k], F, PFt, 4, 4, 4);
        inv(Pp[k + 1], Pi, 4);
        mm(PFt, Pi, C[k], 4, 4, 4);
        for (int i = 0; i < 4; i++) d[i] = xs[k + 1][i] - xp[k + 1][i];
        for (int i = 0; i < 16; i++) D[i] = Ps[k + 1][i] - Pp[k + 1][i];
        for (int i = 0; i < 4; i++) {
            xs[k][i] = xf[k][i];
            for (int j = 0; j < 4; j++) xs[k][i] += C[k][i * 4 + j] * d[j];
        }
        mm(C[k], D, T, 4, 4, 4);
        mmT(T, C[k], Ps[k], 4, 4, 4);
        for (int i = 0; i < 16; i++) Ps[k][i] += Pf[k][i];
        symm(Ps[k], 4);
    }
}

static double nees_pos(const double *x, const double *P, const double *tr) {
    double Pi[4], P2[4] = {P[0], P[1], P[4], P[5]}, e0 = x[0] - tr[0], e1 = x[1] - tr[1];
    if (!inv(P2, Pi, 2)) return 0;
    return e0 * (Pi[0] * e0 + Pi[1] * e1) + e1 * (Pi[2] * e0 + Pi[3] * e1);
}

static void rts_init(void) {
    memset(&RT, 0, sizeof RT);
    Path *p = &A.tp[0];
    path_start(p, 0, 0, 0, rn_range(&A.rt, 11, 16), rn_range(&A.rt, 0, TAU));
    gen_path(p, A.len_s + 20, 6, 14, 6, 14, 40, 110, 1, 8, 22);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    double t0 = rn_range(&A.rt, 0.18, 0.3) * A.len_s, t1 = rn_range(&A.rt, 0.6, 0.72) * A.len_s;
    RT.out[0][0] = t0; RT.out[0][1] = t0 + rn_range(&A.rt, 10, 14);
    RT.out[1][0] = t1; RT.out[1][1] = t1 + rn_range(&A.rt, 10, 14);
    for (int i = 0; i < 2; i++) {
        A.man[A.nman][0] = (float)rt_of(RT.out[i][0]); A.man[A.nman][1] = (float)rt_of(RT.out[i][1]); A.nman++;
    }
    A.follow = 1; A.follow_w = 300; A.follow_b = 14; A.follow_f = 2;
    A.nf = 1;
    f_init(&A.f[0], M_CV, 3.0, 0); A.fname[0] = "FILTER"; A.fcol[0] = C_F2;
    chart(0, "POSITION \xcf\x83", "m, \xe2\x88\x9a" "trace of the x, y block", 1, 0.5f, 100);
    chart_ser(0, S_(0, C_F2, ST_LINE, "filter"));
    chart_ser(0, S_(1, C_F3, ST_LINE, "lag 8 s"));
    chart_ser(0, S_(2, C_F1, ST_LINE, "smoothed"));
    chart(1, "POSITION ERROR", "m", 1, 0.1f, 100);
    chart_ser(1, S_(3, C_F2, ST_LINE, "filter"));
    chart_ser(1, S_(4, C_F3, ST_LINE, "lag 8 s"));
    chart_ser(1, S_(5, C_F1, ST_LINE, "smoothed"));
    chart(2, "NEES", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%, position", 1, 0.01f, 100);
    chart_bounds(2, 2, 10);
    chart_ser(2, S_(6, C_F2, ST_DOTS, "filter"));
    chart_ser(2, S_(7, C_F1, ST_DOTS, "smoothed"));
    for (int k = 0; k < 3; k++) A.ch[k].shade = 1;
}

static void rts_step(void) {
    int k = RT.n;
    if (k >= RK) { RT.n++; return; }
    double t = 0.5 + k, o[6];
    path_eval(&A.tp[0], t, o);
    double nx = rn_n(&A.rm) * RTS_SIG, ny = rn_n(&A.rm) * RTS_SIG, u = rn_u(&A.rm);
    int fix = !rts_outage(t) && u < 0.95;
    double z[2] = {o[0] + nx, o[1] + ny}, R[4] = {RTS_SIG * RTS_SIG, 0, 0, RTS_SIG * RTS_SIG};
    Filt *f = &A.f[0];
    RT.t[k] = t; RT.fix[k] = fix; RT.zx[k] = z[0]; RT.zy[k] = z[1];
    memcpy(RT.tr[k], o, sizeof RT.tr[k]);
    if (k == 0) {                       /* the first second always has a fix */
        f->t = t; f->x[0] = z[0]; f->x[1] = z[1];
        f->P[0] = f->P[5] = R[0]; f->P[10] = f->P[15] = 15 * 15;
        f->t_upd = t; f->nupd = 1; RT.fix[0] = 1;
        memcpy(RT.xp[0], f->x, sizeof RT.xp[0]); memcpy(RT.Pp[0], f->P, sizeof RT.Pp[0]);
    } else {
        f_predict(f, t);
        memcpy(RT.xp[k], f->x, sizeof RT.xp[0]); memcpy(RT.Pp[k], f->P, sizeof RT.Pp[0]);
        if (fix) {
            f_update_pos(f, z, R);
            ring_inn(f->nuw[0], f->nuw[1]);
            A.inn_m = 2;
        } else { f_ease_begin(f); f_ease_end(f); f->nupd--; }
    }
    memcpy(RT.xf[k], f->x, sizeof RT.xf[0]); memcpy(RT.Pf[k], f->P, sizeof RT.Pf[0]);
    trail_push(&A.ft[0], f->x[0], f->x[1]);
    /* smooth everything so far; keep the last pass to ease from */
    memcpy(RT.xs0, RT.xs, sizeof(double) * 4 * k);
    memcpy(RT.Ps0, RT.Ps, sizeof(double) * 16 * k);
    RT.n0 = k;
    RT.n = k + 1;
    double F[16];
    model_F(M_CV, f->x, 1.0, F);
    rts_smooth(RT.n, RT.xf, RT.Pf, RT.xp, RT.Pp, RT.xs, RT.Ps, RT.C, F);
    if (k >= RTS_LAG) {
        int j = k - RTS_LAG;
        memcpy(RT.xl[j], RT.xs[j], sizeof RT.xl[0]); memcpy(RT.Pl[j], RT.Ps[j], sizeof RT.Pl[0]);
        double e = hypot(RT.xl[j][0] - RT.tr[j][0], RT.xl[j][1] - RT.tr[j][1]);
        ser_add(&A.ser[1], rt_of(RT.t[j]), sqrt(RT.Pl[j][0] + RT.Pl[j][5]));
        ser_add(&A.ser[4], rt_of(RT.t[j]), e);
        RT.se[1] += e * e; RT.sn[1] += nees_pos(RT.xl[j], RT.Pl[j], RT.tr[j]); RT.ne[1]++;
    }
    double rt = rt_of(t), e = hypot(f->x[0] - o[0], f->x[1] - o[1]), ne = nees_pos(f->x, f->P, o);
    ser_add(&A.ser[0], rt, sqrt(f->P[0] + f->P[5]));
    ser_add(&A.ser[3], rt, e);
    ser_add(&A.ser[6], rt, ne);
    RT.se[0] += e * e; RT.sn[0] += ne; RT.ne[0]++;
    if (fix) { A.acc_nis[0] += f->nis; A.acc_nn[0]++; }
    A.acc_nees[0] += ne; A.acc_ne[0]++;
    /* the smoothed curves are redrawn whole: the past moves */
    A.ser[2].n = A.ser[5].n = A.ser[7].n = 0;
    RT.se[2] = RT.sn[2] = 0; RT.ne[2] = 0;
    for (int i = 0; i < RT.n; i++) {
        double es = hypot(RT.xs[i][0] - RT.tr[i][0], RT.xs[i][1] - RT.tr[i][1]), ns = nees_pos(RT.xs[i], RT.Ps[i], RT.tr[i]);
        ser_add(&A.ser[2], rt_of(RT.t[i]), sqrt(RT.Ps[i][0] + RT.Ps[i][5]));
        ser_add(&A.ser[5], rt_of(RT.t[i]), es);
        ser_add(&A.ser[7], rt_of(RT.t[i]), ns);
        RT.se[2] += es * es; RT.sn[2] += ns; RT.ne[2]++;
    }
    A.finit = 1;
}

/* ---------------- act 7: sensor fusion in information form ---------------- */
/* A rover with a GPS (that drops out in an urban canyon), wheel odometry
 * (velocity, with a slowly wandering bias the filter estimates), and two
 * range beacons, fused by an information filter. The base station keeps its
 * own track of the rover from its radar and from our position reports, which
 * it takes as news; its errors and ours are then correlated by an amount
 * nobody knows, so the two tracks are combined by covariance intersection
 * (and, for contrast, naively, which double-counts). */
#define FU_DT 0.1
#define FU_GPS 4.0
#define FU_ODO 0.06
#define FU_RNG 1.0
#define FU_BRG (0.5 * DEGR)
#define FU_BRNG 1.5
#define FU_QB 8e-5                      /* the odometer bias's random walk, (m/s)^2 per s */
#define FN 6
enum { SN_GPS, SN_ODO, SN_B1, SN_B2, NSN };
static const char *const SN_NAME[NSN] = {"GPS", "odometry", "beacon 1", "beacon 2"};
static const int SN_COL[NSN] = {C_MEAS, C_F2, C_F3, C_F3};
static struct {
    long tick;
    double Y[FN * FN], y[FN];           /* the information filter */
    Filt kf;                            /* the same filter in covariance form (for --check) */
    Filt rem;                           /* the base station's track: GPS and its bearing camera */
    double dr[2];                       /* dead reckoning: odometry integrated */
    double bias[2];
    double bx[3], by[3];                /* the beacons, and the base station */
    double canyon[2];                   /* no GPS between these sim seconds */
    int nupd[NSN];
    double bits[NSN], bits_pred, last_nis[NSN];
    double bin[NSN][200];               /* bits gained in each sim second */
    /* the latest update, for the picture: prior and posterior position
     * covariance, and what the sensor added (H'R^-1 H) */
    int last_sn; double t_last, Pm[3], Pp[3], HRH[FN * FN];
    double upd_t[NSN], upd_Pm[NSN][3], upd_Pp[NSN][3], upd_J[NSN][3];
    double rng_t[2], rng_r[2];          /* each beacon's latest range */
    double brg_t, brg_b, brg_r;         /* the base's latest range and bearing */
    /* covariance intersection against the base's track, once a second */
    double ci_t, ci_a[2], ci_A[3], ci_b[2], ci_B[3], ci_n[2], ci_N[3], ci_c[2], ci_C[3], ci_w;
    double nees_a, nees_n, nees_c, wsum; int nci;
    double se_f, se_dr, se_g; int nse, nseg;
} FU;

static void fu_cov(void) {                /* the covariance form, for display */
    Filt *f = &A.f[0];
    inv(FU.Y, f->P, FN);
    symm(f->P, FN);
    mm(f->P, FU.y, f->x, FN, FN, 1);
}

static double logdetn(const double *M, int n) {
    double L[MX * MX], s = 0;
    chol(M, L, n);
    for (int i = 0; i < n; i++) s += log(L[i * n + i]);
    return 2 * s;
}

/* an update in information form: z ~ h(x) + v, linearised at the prior */
static void fu_update(int sn, const double *z, const double *zh, const double *Hm, const double *R, int m) {
    Filt *f = &A.f[0];
    double Ri[4], HtRi[FN * 2], HRH[FN * FN], nu[2], ld0 = logdetn(FU.Y, FN);
    inv(R, Ri, m);
    for (int i = 0; i < FN; i++)            /* H' R^-1, FN x m */
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s += Hm[k * FN + i] * Ri[k * m + j];
            HtRi[i * m + j] = s;
        }
    mm(HtRi, Hm, HRH, FN, m, FN);
    for (int i = 0; i < m; i++) nu[i] = z[i] - zh[i];
    {                                       /* the NIS, from the prior */
        double PHt[FN * 2], S[4];
        mmT(f->P, Hm, PHt, FN, FN, m);
        mm(Hm, PHt, S, m, FN, m);
        for (int i = 0; i < m * m; i++) S[i] += R[i];
        innov_stats(f, S, nu, m);
        FU.last_nis[sn] = f->nis;
    }
    double Hx[2];
    mm(Hm, f->x, Hx, m, FN, 1);
    for (int i = 0; i < FN; i++)
        for (int j = 0; j < m; j++) FU.y[i] += HtRi[i * m + j] * (nu[j] + Hx[j]);
    for (int i = 0; i < FN * FN; i++) FU.Y[i] += HRH[i];
    symm(FU.Y, FN);
    FU.Pm[0] = f->P[0]; FU.Pm[1] = f->P[1]; FU.Pm[2] = f->P[FN + 1];
    f_ease_begin(f);
    fu_cov();
    f_ease_end(f);
    FU.Pp[0] = f->P[0]; FU.Pp[1] = f->P[1]; FU.Pp[2] = f->P[FN + 1];
    memcpy(FU.HRH, HRH, sizeof HRH);
    FU.last_sn = sn; FU.t_last = f->t;
    FU.upd_t[sn] = f->t;
    memcpy(FU.upd_Pm[sn], FU.Pm, sizeof FU.Pm); memcpy(FU.upd_Pp[sn], FU.Pp, sizeof FU.Pp);
    FU.upd_J[sn][0] = HRH[0]; FU.upd_J[sn][1] = HRH[1]; FU.upd_J[sn][2] = HRH[FN + 1];
    double b = 0.5 * (logdetn(FU.Y, FN) - ld0) / log(2.0);
    FU.bits[sn] += b;
    int sec = (int)f->t;
    if (sec >= 0 && sec < 200) FU.bin[sn][sec] += b;
    FU.nupd[sn]++;
    /* the same update in covariance form (Joseph), linearised at the same
     * point: the two must agree (--check compares them) */
    f_update(&FU.kf, z, zh, Hm, R, m, -1);
    if (sn == SN_GPS) { ring_inn(f->nuw[0], f->nuw[1]); A.inn_m = 2; }
}

static void fu_init(void) {
    memset(&FU, 0, sizeof FU);
    Path *p = &A.tp[0];
    path_start(p, 0, 0, 0, rn_range(&A.rt, 2.5, 4), rn_range(&A.rt, 0, TAU));
    gen_path(p, A.len_s + 20, 6, 14, 8, 20, 50, 120, 0, 0, 0);
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    for (int j = 0; j < 3; j++) {        /* beacons and base, off the route */
        double o[6], tt = A.len_s * (j < 2 ? 0.25 + 0.45 * j : 0.5);
        path_eval(p, tt, o);
        double side = rn_sign(&A.rt) * rn_range(&A.rt, 25, 45) * (j == 2 ? 1.5 : 1), h = o[4] + PI / 2;
        FU.bx[j] = o[0] + side * cos(h); FU.by[j] = o[1] + side * sin(h);
    }
    FU.rng_t[0] = FU.rng_t[1] = FU.brg_t = -1e9;
    for (int s = 0; s < NSN; s++) FU.upd_t[s] = -1e9;
    FU.canyon[0] = rn_range(&A.rt, 0.3, 0.4) * A.len_s; FU.canyon[1] = FU.canyon[0] + rn_range(&A.rt, 20, 28);
    A.man[0][0] = (float)rt_of(FU.canyon[0]); A.man[0][1] = (float)rt_of(FU.canyon[1]); A.nman = 1;
    A.follow = 1; A.follow_w = 130; A.follow_b = 6; A.follow_f = 2;
    A.nf = 1;
    f_init(&A.f[0], M_CVB, 1.0, FU_QB); A.fname[0] = "FUSED"; A.fcol[0] = C_F1;
    f_init(&FU.kf, M_CVB, 1.0, FU_QB);
    f_init(&FU.rem, M_CV, 1.0, 0);
    double o[6];
    path_eval(p, 0, o);
    FU.bias[0] = 0.05 * rn_n(&A.rm); FU.bias[1] = 0.05 * rn_n(&A.rm);
    double P0[FN * FN] = {0};
    /* started from a GPS fix and a first odometer reading */
    P0[0] = P0[7] = FU_GPS * FU_GPS; P0[14] = P0[21] = 0.1 * 0.1; P0[28] = P0[35] = 0.05 * 0.05;
    double x0[FN] = {o[0] + FU_GPS * rn_n(&A.rf), o[1] + FU_GPS * rn_n(&A.rf), o[2] + 0.1 * rn_n(&A.rf), o[3] + 0.1 * rn_n(&A.rf), 0, 0};
    inv(P0, FU.Y, FN);
    mm(FU.Y, x0, FU.y, FN, FN, 1);
    Filt *f = &A.f[0];
    fu_cov();
    f->t = 0; f->t_upd = -1e9;
    FU.kf.t = 0; memcpy(FU.kf.x, x0, sizeof x0); memcpy(FU.kf.P, P0, sizeof P0);
    double P4[16] = {16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01};
    FU.rem.t = 0; memcpy(FU.rem.x, x0, sizeof(double) * 4); memcpy(FU.rem.P, P4, sizeof P4);
    FU.dr[0] = x0[0]; FU.dr[1] = x0[1];
    FU.ci_t = -1e9;
    A.finit = 1;
    chart(0, "POSITION ERROR", "m", 1, 0.1f, 100);
    chart_ser(0, S_(0, C_MEAS, ST_DOTS, "GPS"));
    chart_ser(0, S_(1, C_F2, ST_LINE, "dead reckoning"));
    chart_ser(0, S_(2, C_F1, ST_LINE, "fused"));
    chart(1, "INFORMATION GAINED", "bits per second, 4 s average", 1, 0.01f, 100);
    chart_ser(1, S_(3, C_MEAS, ST_LINE, "GPS"));
    chart_ser(1, S_(4, C_F2, ST_LINE, "odometry"));
    chart_ser(1, S_(5, C_F3, ST_LINE, "beacons"));
    chart(2, "NEES", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%, position", 1, 0.01f, 100);
    chart_bounds(2, 2, 10);
    chart_ser(2, S_(6, C_F1, ST_DOTS, "fused"));
    chart_ser(2, S_(7, C_WARN, ST_DOTS, "naive"));
    chart_ser(2, S_(8, C_F3, ST_DOTS, "CI"));
    for (int k = 0; k < 3; k++) A.ch[k].shade = 1;
}

static void fu_predict(double t) {
    Filt *f = &A.f[0];
    double ld0 = logdetn(FU.Y, FN);
    f_predict(f, t);                    /* Y- = (F Y^-1 F' + Q)^-1 */
    inv(f->P, FU.Y, FN);
    symm(FU.Y, FN);
    mm(FU.Y, f->x, FU.y, FN, FN, 1);
    FU.bits_pred += 0.5 * (logdetn(FU.Y, FN) - ld0) / log(2.0);
    f_predict(&FU.kf, t);
    f_predict(&FU.rem, t);
}

/* the CI weight: the omega in [0, 1] that minimises det P (golden section) */
static double ci_omega(const double *Ai, const double *Bi) {
    double a = 0, b = 1, g = 0.6180339887498949;
    for (int it = 0; it < 60; it++) {
        double c = b - g * (b - a), d = a + g * (b - a), Mc[4], Md[4];
        for (int i = 0; i < 4; i++) { Mc[i] = c * Ai[i] + (1 - c) * Bi[i]; Md[i] = d * Ai[i] + (1 - d) * Bi[i]; }
        if (det2(Mc) > det2(Md)) b = d; else a = c;      /* maximise det of the information */
    }
    return (a + b) / 2;
}

/* fuse (a, A) and (b, B): naively as if independent, and by CI */
static void ci_fuse(const double *a, const double *A2, const double *b, const double *B2, double w,
                    double *xn, double *Pn, double *xc, double *Pc) {
    double Ai[4], Bi[4], M[4], Mi[4], u[2];
    inv(A2, Ai, 2); inv(B2, Bi, 2);
    for (int pass = 0; pass < 2; pass++) {
        double wa = pass ? w : 1, wb = pass ? 1 - w : 1;
        for (int i = 0; i < 4; i++) M[i] = wa * Ai[i] + wb * Bi[i];
        inv(M, Mi, 2);
        for (int i = 0; i < 2; i++) u[i] = wa * (Ai[i * 2] * a[0] + Ai[i * 2 + 1] * a[1]) + wb * (Bi[i * 2] * b[0] + Bi[i * 2 + 1] * b[1]);
        double *x = pass ? xc : xn, *P = pass ? Pc : Pn;
        x[0] = Mi[0] * u[0] + Mi[1] * u[1]; x[1] = Mi[2] * u[0] + Mi[3] * u[1];
        memcpy(P, Mi, sizeof Mi);
    }
}

static void fu_tick(void) {
    long i = ++FU.tick;
    double t = i * FU_DT, o[6];
    path_eval(&A.tp[0], t, o);
    Filt *f = &A.f[0];
    fu_predict(t);
    /* the odometer: velocity plus a wandering bias, five times a second */
    if (i % 2 == 0) {
        for (int k = 0; k < 2; k++) FU.bias[k] += sqrt(FU_QB * 2 * FU_DT) * rn_n(&A.rm);
        double z[2] = {o[2] + FU.bias[0] + FU_ODO * rn_n(&A.rm), o[3] + FU.bias[1] + FU_ODO * rn_n(&A.rm)};
        FU.dr[0] += z[0] * 2 * FU_DT; FU.dr[1] += z[1] * 2 * FU_DT;
        double Hm[2 * FN] = {0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1}, zh[2], R[4] = {FU_ODO * FU_ODO, 0, 0, FU_ODO * FU_ODO};
        zh[0] = f->x[2] + f->x[4]; zh[1] = f->x[3] + f->x[5];
        fu_update(SN_ODO, z, zh, Hm, R, 2);
    }
    /* GPS once a second, but not in the canyon */
    if (i % 10 == 5) {
        double z[2] = {o[0] + FU_GPS * rn_n(&A.rm), o[1] + FU_GPS * rn_n(&A.rm)}, u = rn_u(&A.rm);
        if (!(t >= FU.canyon[0] && t < FU.canyon[1]) && u < 0.95) {
            double Hm[2 * FN] = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, zh[2] = {f->x[0], f->x[1]}, R[4] = {FU_GPS * FU_GPS, 0, 0, FU_GPS * FU_GPS};
            fu_update(SN_GPS, z, zh, Hm, R, 2);
            add_det(t, 0, 0, 0);
            Det *d = &A.dh[(A.dhn - 1) % DH];
            d->x = z[0]; d->y = z[1];
            ser_add(&A.ser[0], rt_of(t), hypot(z[0] - o[0], z[1] - o[1]));
            FU.se_g += (z[0] - o[0]) * (z[0] - o[0]) + (z[1] - o[1]) * (z[1] - o[1]); FU.nseg++;
        }
    }
    /* each beacon every two seconds, staggered */
    for (int j = 0; j < 2; j++)
        if (i % 20 == 3 + 10 * j) {
            double r = hypot(o[0] - FU.bx[j], o[1] - FU.by[j]) + FU_RNG * rn_n(&A.rm), R[1] = {FU_RNG * FU_RNG};
            /* iterated: relinearise at the posterior a few times (a range is
             * curved enough, close in, to bias a single linearisation) */
            double xl[2] = {f->x[0], f->x[1]}, Hm[FN] = {0}, z[1] = {r}, zh[1];
            for (int it = 0; it < 4; it++) {
                double dx = xl[0] - FU.bx[j], dy = xl[1] - FU.by[j], rh = fmax(hypot(dx, dy), 1);
                Hm[0] = dx / rh; Hm[1] = dy / rh;
                zh[0] = rh + Hm[0] * (f->x[0] - xl[0]) + Hm[1] * (f->x[1] - xl[1]);
                if (it == 3) break;
                double Y2[FN * FN], y2[FN], P2[FN * FN], x2[FN], c = (r - zh[0] + Hm[0] * f->x[0] + Hm[1] * f->x[1]) / R[0];
                memcpy(Y2, FU.Y, sizeof Y2); memcpy(y2, FU.y, sizeof y2);
                for (int u = 0; u < 2; u++) {
                    y2[u] += Hm[u] * c;
                    for (int v = 0; v < 2; v++) Y2[u * FN + v] += Hm[u] * Hm[v] / R[0];
                }
                inv(Y2, P2, FN);
                mm(P2, y2, x2, FN, FN, 1);
                xl[0] = x2[0]; xl[1] = x2[1];
            }
            fu_update(SN_B1 + j, z, zh, Hm, R, 1);
            FU.rng_t[j] = t; FU.rng_r[j] = r;
        }
    /* the base station's own tracker: range and bearing to the rover, once a second */
    if (i % 10 == 7) {
        double z[2] = {hypot(o[0] - FU.bx[2], o[1] - FU.by[2]) + FU_BRNG * rn_n(&A.rm),
                       wrap_pi(atan2(o[1] - FU.by[2], o[0] - FU.bx[2]) + FU_BRG * rn_n(&A.rm))};
        double R[4] = {FU_BRNG * FU_BRNG, 0, 0, FU_BRG * FU_BRG};
        ekf_update_rb(&FU.rem, z, FU.bx[2], FU.by[2], R);
        FU.brg_t = t; FU.brg_b = z[1]; FU.brg_r = z[0];
    }
    /* ...and our own position reports, which it folds in as if they were
     * news: from then on its errors and ours are correlated, by an amount
     * neither side knows */
    if (i % 10 == 8) {
        double z[2] = {f->x[0], f->x[1]}, R[4] = {f->P[0], f->P[1], f->P[FN], f->P[FN + 1]};
        f_update_pos(&FU.rem, z, R);
    }
    /* once a second: the scores, and CI against the base's track */
    if (i % 10 == 0) {
        double rt = rt_of(t), tr[2] = {o[0], o[1]};
        double e = hypot(f->x[0] - o[0], f->x[1] - o[1]), edr = hypot(FU.dr[0] - o[0], FU.dr[1] - o[1]);
        ser_add(&A.ser[1], rt, edr);
        ser_add(&A.ser[2], rt, e);
        FU.se_f += e * e; FU.se_dr += edr * edr; FU.nse++;
        double Pf4[16] = {f->P[0], f->P[1], 0, 0, f->P[FN], f->P[FN + 1]};
        double ne = nees_pos(f->x, Pf4, o);
        ser_add(&A.ser[6], rt, ne);
        A.acc_nees[0] += ne; A.acc_ne[0]++;
        int sec = (int)t;
        for (int k = 0; k < 3; k++) {
            double s = 0;
            for (int b = sec - 4; b < sec; b++) {
                if (b < 0) continue;
                s += k < 2 ? FU.bin[k][b] : FU.bin[SN_B1][b] + FU.bin[SN_B2][b];
            }
            ser_add(&A.ser[3 + k], rt, s / 4);
        }
        double a[2] = {f->x[0], f->x[1]}, A2[4] = {f->P[0], f->P[1], f->P[FN], f->P[FN + 1]};
        double b[2] = {FU.rem.x[0], FU.rem.x[1]}, B2[4] = {FU.rem.P[0], FU.rem.P[1], FU.rem.P[4], FU.rem.P[5]};
        double Ai[4], Bi[4], xn[2], Pn[4], xc[2], Pc[4];
        inv(A2, Ai, 2); inv(B2, Bi, 2);
        double w = ci_omega(Ai, Bi);
        ci_fuse(a, A2, b, B2, w, xn, Pn, xc, Pc);
        FU.ci_t = t; FU.ci_w = w;
        memcpy(FU.ci_a, a, sizeof a); memcpy(FU.ci_b, b, sizeof b); memcpy(FU.ci_n, xn, sizeof xn); memcpy(FU.ci_c, xc, sizeof xc);
        FU.ci_A[0] = A2[0]; FU.ci_A[1] = A2[1]; FU.ci_A[2] = A2[3];
        FU.ci_B[0] = B2[0]; FU.ci_B[1] = B2[1]; FU.ci_B[2] = B2[3];
        FU.ci_N[0] = Pn[0]; FU.ci_N[1] = Pn[1]; FU.ci_N[2] = Pn[3];
        FU.ci_C[0] = Pc[0]; FU.ci_C[1] = Pc[1]; FU.ci_C[2] = Pc[3];
        double Pn4[16] = {Pn[0], Pn[1], 0, 0, Pn[2], Pn[3]}, Pc4[16] = {Pc[0], Pc[1], 0, 0, Pc[2], Pc[3]};
        double nn = nees_pos(xn, Pn4, tr), nc = nees_pos(xc, Pc4, tr);
        ser_add(&A.ser[7], rt, nn);
        ser_add(&A.ser[8], rt, nc);
        FU.nees_a += ne; FU.nees_n += nn; FU.nees_c += nc; FU.nci++; FU.wsum += w;
        trail_push(&A.ft[1], FU.dr[0], FU.dr[1]);
        trail_push(&A.ft[2], FU.rem.x[0], FU.rem.x[1]);
    }
    if (i % 2 == 0) trail_push(&A.ft[0], f->x[0], f->x[1]);
}

/* ---------------- act 8: numerical stability in float32 ---------------- */
/* A coarse cue (sigma 30 m, correlated) hands a hovering drone to a laser
 * tracker (sigma 1 mm): one update shrinks P by a factor of 10^9, beyond
 * float32's 7 digits. The textbook update loses symmetry and positive
 * definiteness; the Joseph form and the square-root (array) form hold. */
#define NU_DT 0.1
#define NU_SIG 1e-3
#define NU_CYC 22.5
#define NU_COAST 4.0                    /* seconds on the coarse cue alone before the laser */
#define NU_Q 1e-4
enum { NF_NAIVE, NF_JOSEPH, NF_SQRT, NNF };
static const char *const NF_NAME[NNF] = {"naive", "Joseph", "sqrt"};
static const int NF_COL[NNF] = {C_WARN, C_F2, C_F1};
#define NUH 1024
static struct {
    long k; int cyc;
    double xt[4];                       /* truth */
    double xr[4], Pr[16];               /* reference: double precision, Joseph */
    float x[NNF][4], P[NNF][16];        /* P, or for sqrt its factor S (P = S S') */
    int bad[NNF], broken[NNF], nbad[NNF];
    double t_cue, t_bad[NNF];
    double Pd[NNF][16];                 /* each P in double, for the display */
    double nis[NNF], rel[NNF], asym[NNF], lmin[NNF], err[NNF], eig[NNF + 1][4];
    double hs[NUH]; int hn;             /* log10 of the reference sigma, each step */
    double ht[NUH];
    double tx[NUH][2]; int txn;         /* the truth's trail */
    double se[NNF + 1]; int nse;
} NU;

/* L (n x n, lower) with L L' = A A', A n x p, by Householder reflections
 * from the right: the "tria" of the square-root filters */
#define DEF_TRIA(T, NAME, SQ)                                                              \
    static void NAME(const T *A0, int n, int p, T *L) {                                     \
        T Aw[12 * 12], v[12];                                                               \
        memcpy(Aw, A0, sizeof(T) * n * p);                                                  \
        for (int i = 0; i < n; i++) {                                                       \
            T nrm = 0;                                                                      \
            for (int j = i; j < p; j++) nrm += Aw[i * p + j] * Aw[i * p + j];               \
            nrm = SQ(nrm);                                                                  \
            if (nrm == 0) continue;                                                         \
            T al = Aw[i * p + i] > 0 ? -nrm : nrm, vv = 0;                                  \
            for (int j = i; j < p; j++) v[j] = Aw[i * p + j];                               \
            v[i] -= al;                                                                     \
            for (int j = i; j < p; j++) vv += v[j] * v[j];                                  \
            if (vv == 0) continue;                                                          \
            for (int r = i; r < n; r++) {                                                   \
                T d = 0;                                                                    \
                for (int j = i; j < p; j++) d += Aw[r * p + j] * v[j];                      \
                d = 2 * d / vv;                                                             \
                for (int j = i; j < p; j++) Aw[r * p + j] -= d * v[j];                      \
            }                                                                               \
        }                                                                                   \
        for (int i = 0; i < n; i++)                                                         \
            for (int j = 0; j < n; j++) L[i * n + j] = j <= i ? Aw[i * p + j] : 0;          \
    }
DEF_TRIA(float, tria_f, sqrtf)
DEF_TRIA(double, tria_d, sqrt)

/* the three float filters and their double twins (for --check), one
 * template: CV, position measured, R = r I */
#define DEF_NUMF(T, SUF, SQ)                                                                \
    static void predP_##SUF(T *P, const T *F, const T *Q) {                                 \
        T A1[16], B1[16];                                                                   \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += F[i * 4 + k] * P[k * 4 + j];          \
            A1[i * 4 + j] = s; }                                                            \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += A1[i * 4 + k] * F[j * 4 + k];         \
            B1[i * 4 + j] = s + Q[i * 4 + j]; }                                             \
        memcpy(P, B1, sizeof B1);                                                           \
    }                                                                                       \
    static void predx_##SUF(T *x, const T *F) {                                             \
        T y[4];                                                                             \
        for (int i = 0; i < 4; i++) { T s = 0; for (int k = 0; k < 4; k++) s += F[i * 4 + k] * x[k]; y[i] = s; } \
        memcpy(x, y, sizeof y);                                                             \
    }                                                                                       \
    /* S (n x n lower): S- = tria([F S, Lq]) */                                             \
    static void predS_##SUF(T *S, const T *F, const T *Lq) {                                \
        T Aw[4 * 8];                                                                        \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += F[i * 4 + k] * S[k * 4 + j];          \
            Aw[i * 8 + j] = s; Aw[i * 8 + 4 + j] = Lq[i * 4 + j]; }                         \
        tria_##SUF(Aw, 4, 8, S);                                                            \
    }                                                                                       \
    /* 2x2 S^-1; returns 0 if not positive */                                              \
    static int gainK_##SUF(const T *P, T r, T *K, T *Sm) {                                  \
        T a = P[0] + r, b = P[1], c = P[4], d = P[5] + r, det = a * d - b * c;               \
        Sm[0] = a; Sm[1] = b; Sm[2] = c; Sm[3] = d;                                         \
        if (!(det > 0) || !(a > 0)) return 0;                                               \
        T i00 = d / det, i01 = -b / det, i10 = -c / det, i11 = a / det;                     \
        for (int i = 0; i < 4; i++) {                                                       \
            K[i * 2] = P[i * 4] * i00 + P[i * 4 + 1] * i10;                                 \
            K[i * 2 + 1] = P[i * 4] * i01 + P[i * 4 + 1] * i11; }                           \
        return 1;                                                                           \
    }                                                                                       \
    static int upd_naive_##SUF(T *x, T *P, const T *z, T r, T *nis) {                       \
        T K[8], Sm[4], IKH[16], N[16];                                                      \
        if (!gainK_##SUF(P, r, K, Sm)) return 0;                                            \
        T n0 = z[0] - x[0], n1 = z[1] - x[1], dt = Sm[0] * Sm[3] - Sm[1] * Sm[2];             \
        *nis = (n0 * (Sm[3] * n0 - Sm[1] * n1) + n1 * (-Sm[2] * n0 + Sm[0] * n1)) / dt;     \
        for (int i = 0; i < 4; i++) x[i] += K[i * 2] * n0 + K[i * 2 + 1] * n1;              \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)                              \
            IKH[i * 4 + j] = (i == j) - (j < 2 ? K[i * 2 + j] : 0);                         \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += IKH[i * 4 + k] * P[k * 4 + j];        \
            N[i * 4 + j] = s; }                                                             \
        memcpy(P, N, sizeof N);                                                             \
        return 1;                                                                           \
    }                                                                                       \
    static int upd_joseph_##SUF(T *x, T *P, const T *z, T r, T *nis) {                      \
        T K[8], Sm[4], IKH[16], N[16], M[16];                                               \
        if (!gainK_##SUF(P, r, K, Sm)) return 0;                                            \
        T n0 = z[0] - x[0], n1 = z[1] - x[1], dt = Sm[0] * Sm[3] - Sm[1] * Sm[2];             \
        *nis = (n0 * (Sm[3] * n0 - Sm[1] * n1) + n1 * (-Sm[2] * n0 + Sm[0] * n1)) / dt;     \
        for (int i = 0; i < 4; i++) x[i] += K[i * 2] * n0 + K[i * 2 + 1] * n1;              \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)                              \
            IKH[i * 4 + j] = (i == j) - (j < 2 ? K[i * 2 + j] : 0);                         \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += IKH[i * 4 + k] * P[k * 4 + j];        \
            N[i * 4 + j] = s; }                                                             \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {                           \
            T s = 0; for (int k = 0; k < 4; k++) s += N[i * 4 + k] * IKH[j * 4 + k];        \
            M[i * 4 + j] = s + r * (K[i * 2] * K[j * 2] + K[i * 2 + 1] * K[j * 2 + 1]); }   \
        memcpy(P, M, sizeof M);                                                             \
        return 1;                                                                           \
    }                                                                                       \
    /* the array form: tria([sqrt(r) I, H S; 0, S]) = [Se^1/2, 0; Kbar, S+], K = Kbar Se^-1/2 */ \
    static T lastw_##SUF[2];                                                                \
    static int upd_sqrt_##SUF(T *x, T *S, const T *z, T r, T *nis) {                        \
        T Aw[6 * 6] = {0}, L[36], sr = SQ(r);                                               \
        Aw[0] = sr; Aw[7] = sr;                                                             \
        for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) Aw[i * 6 + 2 + j] = S[i * 4 + j]; \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) Aw[(2 + i) * 6 + 2 + j] = S[i * 4 + j]; \
        tria_##SUF(Aw, 6, 6, L);                                                            \
        T x00 = L[0], x10 = L[6], x11 = L[7];                                               \
        if (x00 == 0 || x11 == 0) return 0;                                                 \
        T n0 = z[0] - x[0], n1 = z[1] - x[1];                                               \
        T w0 = n0 / x00, w1 = (n1 - x10 * w0) / x11;      /* Se^-1/2 nu */                   \
        *nis = w0 * w0 + w1 * w1;                                                           \
        lastw_##SUF[0] = w0; lastw_##SUF[1] = w1;                                           \
        for (int i = 0; i < 4; i++) x[i] += L[(2 + i) * 6] * w0 + L[(2 + i) * 6 + 1] * w1;  \
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) S[i * 4 + j] = L[(2 + i) * 6 + 2 + j]; \
        return 1;                                                                           \
    }
DEF_NUMF(float, f, sqrtf)
DEF_NUMF(double, d, sqrt)

static void nu_FQ(double *F, double *Q, double *Lq) {
    model_F(M_CV, NULL, NU_DT, F);
    model_Q(M_CV, NU_DT, NU_Q, 0, Q);
    chol(Q, Lq, 4);
}

/* eigenvalues of a symmetric 4x4 (cyclic Jacobi), ascending */
static void eig4(const double *M, double *ev) {
    double a[16];
    memcpy(a, M, sizeof a);
    for (int sw = 0; sw < 30; sw++) {
        double off = 0;
        for (int p = 0; p < 4; p++) for (int q = p + 1; q < 4; q++) off += a[p * 4 + q] * a[p * 4 + q];
        if (off < 1e-300) break;
        for (int p = 0; p < 4; p++)
            for (int q = p + 1; q < 4; q++) {
                double apq = a[p * 4 + q];
                if (fabs(apq) < 1e-300) continue;
                double th = (a[q * 4 + q] - a[p * 4 + p]) / (2 * apq);
                double t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1)), c = 1 / sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 4; k++) {
                    double akp = a[k * 4 + p], akq = a[k * 4 + q];
                    a[k * 4 + p] = c * akp - s * akq; a[k * 4 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; k++) {
                    double apk = a[p * 4 + k], aqk = a[q * 4 + k];
                    a[p * 4 + k] = c * apk - s * aqk; a[q * 4 + k] = s * apk + c * aqk;
                }
            }
    }
    for (int i = 0; i < 4; i++) ev[i] = a[i * 4 + i];
    for (int i = 1; i < 4; i++)
        for (int j = i; j > 0 && ev[j] < ev[j - 1]; j--) { double t = ev[j]; ev[j] = ev[j - 1]; ev[j - 1] = t; }
}

static void nu_init(void) {
    memset(&NU, 0, sizeof NU);
    NU.cyc = -1;
    NU.xt[0] = rn_range(&A.rt, -3, 3); NU.xt[1] = rn_range(&A.rt, -3, 3);
    NU.xt[2] = rn_range(&A.rt, -0.03, 0.03); NU.xt[3] = rn_range(&A.rt, -0.03, 0.03);
    A.nf = 0;
    for (int c = 0; c * NU_CYC < A.len_s && A.nman < 32; c++) {
        A.man[A.nman][0] = (float)rt_of(c * NU_CYC); A.man[A.nman][1] = (float)rt_of(c * NU_CYC + NU_COAST); A.nman++;
    }
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%", 1, 0.01f, 1e4f);
    chart_bounds(0, 2, 10);
    for (int i = 0; i < NNF; i++) chart_ser(0, S_(i, NF_COL[i], ST_DOTS, NF_NAME[i]));
    chart(1, "\xe2\x80\x96P \xe2\x88\x92 P_ref\xe2\x80\x96 / \xe2\x80\x96P_ref\xe2\x80\x96", "float32 against float64", 1, 1e-9f, 1e3f);
    for (int i = 0; i < NNF; i++) chart_ser(1, S_(3 + i, NF_COL[i], ST_LINE, NF_NAME[i]));
    chart(2, "POSITION ERROR", "mm", 1, 0.01f, 1e5f);
    chart_ser(2, S_(6, C_TRUTH, ST_LINE, "float64"));
    for (int i = 0; i < NNF; i++) chart_ser(2, S_(7 + i, NF_COL[i], ST_LINE, NF_NAME[i]));
    for (int k = 0; k < 3; k++) A.ch[k].shade = 1;
}

/* every filter's P in double (for sqrt: S S') */
static void nu_Pd(int i, double *P) {
    if (i == NF_SQRT) {
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
                double s = 0;
                for (int k = 0; k < 4; k++) s += (double)NU.P[i][a * 4 + k] * NU.P[i][b * 4 + k];
                P[a * 4 + b] = s;
            }
    } else for (int k = 0; k < 16; k++) P[k] = NU.P[i][k];
}

static void nu_step(void) {
    long k = ++NU.k;
    double t = k * NU_DT, F[16], Q[16], Lq[16];
    nu_FQ(F, Q, Lq);
    /* truth: the same CV process the filters assume */
    {
        double w[4] = {rn_n(&A.rt), rn_n(&A.rt), rn_n(&A.rt), rn_n(&A.rt)}, y[4];
        mm(F, NU.xt, y, 4, 4, 1);
        for (int i = 0; i < 4; i++) {
            NU.xt[i] = y[i];
            for (int j = 0; j <= i; j++) NU.xt[i] += Lq[i * 4 + j] * w[j];
        }
    }
    if (NU.txn < NUH) { NU.tx[NU.txn][0] = NU.xt[0]; NU.tx[NU.txn][1] = NU.xt[1]; NU.txn++; }
    int cyc = (int)floor(t / NU_CYC + 1e-9);
    if (cyc != NU.cyc) {                 /* a new cue: every filter starts from it */
        NU.cyc = cyc; NU.t_cue = t;
        double Aq[16], w[4], P0[16];
        static const double sc[4] = {30, 30, 2, 2};
        for (int i = 0; i < 16; i++) Aq[i] = sc[i / 4] * rn_n(&A.rf);
        mmT(Aq, Aq, P0, 4, 4, 4);
        for (int i = 0; i < 4; i++) w[i] = rn_n(&A.rf);
        double L0[16];
        chol(P0, L0, 4);
        for (int i = 0; i < 4; i++) {
            NU.xr[i] = NU.xt[i];
            for (int j = 0; j <= i; j++) NU.xr[i] += L0[i * 4 + j] * w[j];
        }
        memcpy(NU.Pr, P0, sizeof P0);
        for (int f = 0; f < NNF; f++) {
            for (int i = 0; i < 4; i++) NU.x[f][i] = (float)NU.xr[i];
            for (int i = 0; i < 16; i++) NU.P[f][i] = (float)(f == NF_SQRT ? L0[i] : P0[i]);
            NU.bad[f] = NU.broken[f] = 0;
        }
        return;
    }
    float Ff[16], Qf[16], Lqf[16];
    for (int i = 0; i < 16; i++) { Ff[i] = (float)F[i]; Qf[i] = (float)Q[i]; Lqf[i] = (float)Lq[i]; }
    double z[2] = {NU.xt[0] + NU_SIG * rn_n(&A.rm), NU.xt[1] + NU_SIG * rn_n(&A.rm)}, r = NU_SIG * NU_SIG, nisr = 0;
    float zf[2] = {(float)z[0], (float)z[1]}, rf = (float)r;
    int laser = t - NU.t_cue >= NU_COAST - 1e-9;      /* before that, the filters coast on the cue */
    predx_d(NU.xr, F); predP_d(NU.Pr, F, Q);
    if (laser) upd_joseph_d(NU.xr, NU.Pr, z, r, &nisr);
    symm(NU.Pr, 4);
    for (int f = 0; f < NNF; f++) {
        if (NU.broken[f]) continue;
        predx_f(NU.x[f], Ff);
        if (f == NF_SQRT) predS_f(NU.P[f], Ff, Lqf); else predP_f(NU.P[f], Ff, Qf);
        float nis = 0;
        int ok = !laser ? 1 : f == NF_NAIVE ? upd_naive_f(NU.x[f], NU.P[f], zf, rf, &nis)
               : f == NF_JOSEPH ? upd_joseph_f(NU.x[f], NU.P[f], zf, rf, &nis) : upd_sqrt_f(NU.x[f], NU.P[f], zf, rf, &nis);
        int fin = 1;
        for (int i = 0; i < 16; i++) fin &= isfinite(NU.P[f][i]);
        for (int i = 0; i < 4; i++) fin &= isfinite(NU.x[f][i]);
        if (!ok || !fin) { NU.broken[f] = 1; NU.t_bad[f] = t; NU.nbad[f]++; continue; }
        NU.nis[f] = nis;
        if (f == NF_SQRT && laser) { ring_inn(lastw_f[0], lastw_f[1]); A.inn_m = 2; }   /* the array form whitens it */
    }
    /* how far each has strayed from the reference */
    double rt = rt_of(t), mr = 0;
    for (int i = 0; i < 16; i++) mr = fmax(mr, fabs(NU.Pr[i]));
    eig4(NU.Pr, NU.eig[NNF]);
    double e0 = hypot(NU.xr[0] - NU.xt[0], NU.xr[1] - NU.xt[1]) * 1000;
    ser_add(&A.ser[6], rt, e0);
    int settled = t - NU.t_cue > NU_COAST + 1;      /* RMS once the laser has settled */
    if (settled) NU.se[NNF] += e0 * e0;
    for (int f = 0; f < NNF; f++) {
        if (NU.broken[f]) {             /* a gap in its curves */
            if (A.ser[3 + f].n && A.ser[3 + f].v[A.ser[3 + f].n - 1] == A.ser[3 + f].v[A.ser[3 + f].n - 1]) {
                ser_add(&A.ser[3 + f], rt, NAN); ser_add(&A.ser[7 + f], rt, NAN);
            }
            continue;
        }
        double *P = NU.Pd[f], Ps[16], d = 0, as = 0, mp = 0;
        nu_Pd(f, P);
        for (int i = 0; i < 16; i++) { d = fmax(d, fabs(P[i] - NU.Pr[i])); mp = fmax(mp, fabs(P[i])); }
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
            as = fmax(as, fabs(P[i * 4 + j] - P[j * 4 + i]));
            Ps[i * 4 + j] = 0.5 * (P[i * 4 + j] + P[j * 4 + i]);
        }
        eig4(Ps, NU.eig[f]);
        NU.rel[f] = d / mr; NU.asym[f] = as / fmax(mp, 1e-300); NU.lmin[f] = NU.eig[f][0];
        int bad = NU.eig[f][0] <= 0;
        if (bad && !NU.bad[f]) { NU.t_bad[f] = t; NU.nbad[f]++; }
        NU.bad[f] = bad;
        NU.err[f] = hypot(NU.x[f][0] - NU.xt[0], NU.x[f][1] - NU.xt[1]) * 1000;
        if (settled) NU.se[f] += NU.err[f] * NU.err[f];
        if (laser) ser_add(&A.ser[f], rt, NU.nis[f]);
        ser_add(&A.ser[3 + f], rt, fmax(NU.rel[f], 1e-12));
        ser_add(&A.ser[7 + f], rt, NU.err[f]);
    }
    NU.nse += settled;
    double a = NU.Pr[0], b = NU.Pr[1], c = NU.Pr[5], l1 = (a + c) / 2 + sqrt(fmax((a - c) * (a - c) / 4 + b * b, 0));
    if (NU.hn < NUH) { NU.hs[NU.hn] = log10(sqrt(fmax(l1, 1e-30))); NU.ht[NU.hn] = t; NU.hn++; }
    double tr4[4] = {NU.xt[0], NU.xt[1], NU.xt[2], NU.xt[3]};
    Filt g; memset(&g, 0, sizeof g); g.n = 4; memcpy(g.x, NU.xr, sizeof NU.xr); memcpy(g.P, NU.Pr, sizeof NU.Pr);
    if (laser) acc_stats(0, nisr, nees(&g, tr4, 4));
    A.finit = 1;
}

/* ---------------- act 9: EKF-SLAM ---------------- */
/* A robot drives a loop among landmarks it has never seen, measuring range
 * and bearing to those within reach and dead-reckoning between with noisy
 * odometry; robot and map share one state and one covariance, and when it
 * comes round again the map pulls itself together. */
#define LMAX 16
#define NSL (3 + 2 * LMAX)
#define SL_DT 0.5
#define SL_RMAX 14.0
#define SL_SR 0.25
#define SL_SB (2.0 * DEGR)
#define SL_SV 0.08
#define SL_SW (1.5 * DEGR)
#define SLH 512
static struct {
    int L;                              /* true landmarks */
    double mx[LMAX], my[LMAX];
    int slot[LMAX], id[LMAX], nl, n;    /* map slots <-> landmarks; state size */
    double x[NSL], P[NSL * NSL], x0[NSL], P0[NSL * NSL];
    long k; double t;
    double first[LMAX], last[LMAX];
    double t_close[8]; int nclose, close_lm[8];
    double rx[SLH][3]; int rn;          /* the estimated robot path */
    int seen[LMAX];                     /* observed in the latest step */
    double nees_r, nees_m; int nnr, nnm;
} SL;

static double sl_at(int i, int j) { return SL.P[i * NSL + j]; }

static void sl_init(void) {
    memset(&SL, 0, sizeof SL);
    Path *p = &A.tp[0];
    double v = 1.5, r = rn_range(&A.rt, 5, 8), lx = rn_range(&A.rt, 26, 38), ly = rn_range(&A.rt, 14, 24), sg = rn_sign(&A.rt);
    double h0 = rn_range(&A.rt, 0, TAU);
    path_start(p, 0, 0, 0, v, h0);
    for (int lap = 0; lap < 3; lap++)
        for (int side = 0; side < 4; side++) {
            path_add(p, S_CV, (side % 2 ? ly : lx) / v, 0);
            path_add(p, S_TURN, (PI / 2) * r / v, sg * v / r);
        }
    A.nt = 1; A.tb[0] = 0; A.te[0] = 1e9;
    /* landmarks scattered about the loop, not too close together */
    double cx = 0, cy = 0, o[6], lap_t = (2 * (lx + ly) + TAU * r) / v;
    for (int i = 0; i < 40; i++) { path_eval(p, lap_t * i / 40, o); cx += o[0] / 40; cy += o[1] / 40; }
    SL.L = 0;
    for (int tries = 0; tries < 400 && SL.L < 14; tries++) {
        path_eval(p, rn_range(&A.rt, 0, lap_t), o);
        double side = rn_sign(&A.rt) * rn_range(&A.rt, 3, 10), hh = o[4] + PI / 2;
        double x = o[0] + side * cos(hh), y = o[1] + side * sin(hh);
        int ok = 1;
        for (int j = 0; j < SL.L; j++) ok &= hypot(x - SL.mx[j], y - SL.my[j]) > 7;
        for (int j = 0; j < 40 && ok; j++) { double q[6]; path_eval(p, lap_t * j / 40, q); ok &= hypot(x - q[0], y - q[1]) > 2.5; }
        if (!ok) continue;
        SL.mx[SL.L] = x; SL.my[SL.L] = y; SL.L++;
    }
    for (int j = 0; j < LMAX; j++) SL.slot[j] = -1;
    path_eval(p, 0, o);
    SL.x[0] = o[0]; SL.x[1] = o[1]; SL.x[2] = o[4];   /* the map's frame is the start pose */
    SL.n = 3;
    memcpy(SL.x0, SL.x, sizeof SL.x);
    /* view: the loop and the landmarks */
    A.nt = 1;
    view_fit(0, 0.18, 30);
    for (int j = 0; j < SL.L; j++) {
        A.view[0] = fminf(A.view[0], (float)SL.mx[j] - 4); A.view[2] = fmaxf(A.view[2], (float)SL.mx[j] + 4);
        A.view[1] = fminf(A.view[1], (float)SL.my[j] - 4); A.view[3] = fmaxf(A.view[3], (float)SL.my[j] + 4);
    }
    (void)cx; (void)cy;
    A.nf = 0;
    chart(0, "NIS", "\xcf\x87\xc2\xb2\xe2\x82\x82 95%, every landmark sighting", 1, 0.01f, 100);
    chart_bounds(0, 2, 20);
    chart_ser(0, S_(0, C_F1, ST_DOTS, "sightings"));
    chart(1, "ROBOT POSITION", "m, error and 2\xcf\x83", 0, 0, 5);
    chart_ser(1, S_(1, C_F2, ST_LINE, "2\xcf\x83"));
    chart_ser(1, S_(2, C_F1, ST_LINE, "error"));
    chart(2, "MAP UNCERTAINTY", "m, landmark \xcf\x83 (mean, worst)", 0, 0, 3);
    chart_ser(2, S_(3, C_F3, ST_LINE, "worst"));
    chart_ser(2, S_(4, C_F1, ST_LINE, "mean"));
}

/* the robot moves by odometry (v, w) over dt: an exact arc */
static void sl_move(double *x, double v, double w, double dt, double *Gx, double *Gu) {
    double th = x[2];
    if (fabs(w) < 1e-6) {
        double c = cos(th), s = sin(th);
        x[0] += v * dt * c; x[1] += v * dt * s;
        if (Gx) { double G[9] = {1, 0, -v * dt * s, 0, 1, v * dt * c, 0, 0, 1}; memcpy(Gx, G, sizeof G); }
        if (Gu) { double G[6] = {dt * c, -0.5 * v * dt * dt * s, dt * s, 0.5 * v * dt * dt * c, 0, dt}; memcpy(Gu, G, sizeof G); }
    } else {
        double th1 = th + w * dt, r = v / w;
        x[0] += r * (sin(th1) - sin(th)); x[1] -= r * (cos(th1) - cos(th)); x[2] = wrap_pi(th1);
        if (Gx) { double G[9] = {1, 0, r * (cos(th1) - cos(th)), 0, 1, r * (sin(th1) - sin(th)), 0, 0, 1}; memcpy(Gx, G, sizeof G); }
        if (Gu) {
            double dxv = (sin(th1) - sin(th)) / w, dyv = -(cos(th1) - cos(th)) / w;
            double dxw = -v / (w * w) * (sin(th1) - sin(th)) + r * cos(th1) * dt;
            double dyw = v / (w * w) * (cos(th1) - cos(th)) + r * sin(th1) * dt;
            double G[6] = {dxv, dxw, dyv, dyw, 0, dt};
            memcpy(Gu, G, sizeof G);
        }
    }
}

/* the odometry the robot would read over [t, t + dt]: the arc that joins the poses */
static void sl_odo_true(double t, double dt, double *v, double *w) {
    double a[6], b[6];
    path_eval(&A.tp[0], t, a); path_eval(&A.tp[0], t + dt, b);
    double dth = wrap_pi(b[4] - a[4]), ch = hypot(b[0] - a[0], b[1] - a[1]);
    *w = dth / dt;
    *v = fabs(dth) > 1e-9 ? ch / dt * (dth / 2) / sin(dth / 2) : ch / dt;
}

static void sl_step(void) {
    long k = ++SL.k;
    double t = k * SL_DT, t0 = t - SL_DT;
    int n = SL.n;
    memcpy(SL.x0, SL.x, sizeof SL.x);
    memcpy(SL.P0, SL.P, sizeof SL.P);
    /* predict: robot by noisy odometry; robot-map cross terms through Gx */
    double v, w, Gx[9], Gu[6];
    sl_odo_true(t0, SL_DT, &v, &w);
    v += SL_SV * rn_n(&A.rm); w += SL_SW * rn_n(&A.rm);
    sl_move(SL.x, v, w, SL_DT, Gx, Gu);
    {
        double M[4] = {SL_SV * SL_SV, 0, 0, SL_SW * SL_SW}, T[NSL * 3];
        /* P_rr */
        double Prr[9], A1[9], B1[9], GM[6], GQ[9];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Prr[i * 3 + j] = sl_at(i, j);
        mm(Gx, Prr, A1, 3, 3, 3); mmT(A1, Gx, B1, 3, 3, 3);
        mm(Gu, M, GM, 3, 2, 2); mmT(GM, Gu, GQ, 3, 2, 3);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) SL.P[i * NSL + j] = B1[i * 3 + j] + GQ[i * 3 + j];
        /* P_rm = Gx P_rm */
        for (int i = 0; i < 3; i++)
            for (int j = 3; j < n; j++) {
                double s = 0;
                for (int l = 0; l < 3; l++) s += Gx[i * 3 + l] * sl_at(l, j);
                T[i * NSL + j] = s;
            }
        for (int i = 0; i < 3; i++)
            for (int j = 3; j < n; j++) SL.P[i * NSL + j] = SL.P[j * NSL + i] = T[i * NSL + j];
    }
    /* observe every landmark within reach */
    double o[6];
    path_eval(&A.tp[0], t, o);
    for (int j = 0; j < SL.L; j++) {
        SL.seen[j] = 0;
        double dx = SL.mx[j] - o[0], dy = SL.my[j] - o[1], r = hypot(dx, dy);
        if (r > SL_RMAX) continue;
        double z[2] = {r + SL_SR * rn_n(&A.rm), wrap_pi(atan2(dy, dx) - o[4] + SL_SB * rn_n(&A.rm))};
        double R[4] = {SL_SR * SL_SR, 0, 0, SL_SB * SL_SB};
        SL.seen[j] = 1;
        int s = SL.slot[j];
        n = SL.n;
        if (s < 0) {                    /* a new landmark: augment the state */
            if (SL.nl >= LMAX) continue;
            s = SL.slot[j] = SL.nl; SL.id[s] = j; SL.nl++;
            int a = 3 + 2 * s;
            double th = SL.x[2] + z[1], c = cos(th), sn = sin(th);
            SL.x[a] = SL.x[0] + z[0] * c; SL.x[a + 1] = SL.x[1] + z[0] * sn;
            double Gr[6] = {1, 0, -z[0] * sn, 0, 1, z[0] * c}, Gz[4] = {c, -z[0] * sn, sn, z[0] * c};
            /* P_mx = Gr P_r. (2 x n), P_mm = Gr Prr Gr' + Gz R Gz' */
            double Pmx[2 * NSL];
            for (int i = 0; i < 2; i++)
                for (int jj = 0; jj < n; jj++) {
                    double q = 0;
                    for (int l = 0; l < 3; l++) q += Gr[i * 3 + l] * sl_at(l, jj);
                    Pmx[i * NSL + jj] = q;
                }
            for (int i = 0; i < 2; i++)
                for (int jj = 0; jj < n; jj++) SL.P[(a + i) * NSL + jj] = SL.P[jj * NSL + a + i] = Pmx[i * NSL + jj];
            for (int i = 0; i < 2; i++)
                for (int jj = 0; jj < 2; jj++) {
                    double q = 0;
                    for (int l = 0; l < 3; l++) q += Pmx[i * NSL + l] * Gr[jj * 3 + l];
                    double gz = 0;
                    for (int l = 0; l < 2; l++) gz += Gz[i * 2 + l] * R[l * 2 + l] * Gz[jj * 2 + l];
                    SL.P[(a + i) * NSL + a + jj] = q + gz;
                }
            SL.n = n + 2;
            SL.first[j] = SL.last[j] = t;
            continue;
        }
        /* an update: H touches the robot and this landmark only */
        int a = 3 + 2 * s;
        double ex = SL.x[a] - SL.x[0], ey = SL.x[a + 1] - SL.x[1], q = fmax(ex * ex + ey * ey, 1e-9), sq = sqrt(q);
        double zh[2] = {sq, wrap_pi(atan2(ey, ex) - SL.x[2])};
        double Hr[6] = {-ex / sq, -ey / sq, 0, ey / q, -ex / q, -1}, Hm[4] = {ex / sq, ey / sq, -ey / q, ex / q};
        static double PHt[NSL * 2];
        for (int i = 0; i < n; i++)
            for (int m = 0; m < 2; m++) {
                double acc = 0;
                for (int l = 0; l < 3; l++) acc += sl_at(i, l) * Hr[m * 3 + l];
                for (int l = 0; l < 2; l++) acc += sl_at(i, a + l) * Hm[m * 2 + l];
                PHt[i * 2 + m] = acc;
            }
        double S[4], Si[4], nu[2] = {z[0] - zh[0], wrap_pi(z[1] - zh[1])};
        for (int m = 0; m < 2; m++)
            for (int mm2 = 0; mm2 < 2; mm2++) {
                double acc = R[m * 2 + mm2];
                for (int l = 0; l < 3; l++) acc += Hr[m * 3 + l] * PHt[l * 2 + mm2];
                for (int l = 0; l < 2; l++) acc += Hm[m * 2 + l] * PHt[(a + l) * 2 + mm2];
                S[m * 2 + mm2] = acc;
            }
        inv(S, Si, 2);
        double nis = nu[0] * (Si[0] * nu[0] + Si[1] * nu[1]) + nu[1] * (Si[2] * nu[0] + Si[3] * nu[1]);
        static double K[NSL * 2];
        for (int i = 0; i < n; i++)
            for (int m = 0; m < 2; m++) K[i * 2 + m] = PHt[i * 2] * Si[m] + PHt[i * 2 + 1] * Si[2 + m];
        for (int i = 0; i < n; i++) SL.x[i] += K[i * 2] * nu[0] + K[i * 2 + 1] * nu[1];
        SL.x[2] = wrap_pi(SL.x[2]);
        /* P -= K S K' = K (P H')' */
        for (int i = 0; i < n; i++)
            for (int jj = 0; jj <= i; jj++) {
                double d = K[i * 2] * PHt[jj * 2] + K[i * 2 + 1] * PHt[jj * 2 + 1];
                SL.P[i * NSL + jj] -= d;
                if (jj != i) SL.P[jj * NSL + i] = SL.P[i * NSL + jj];
            }
        ser_add(&A.ser[0], rt_of(t), nis);
        acc_stats(0, nis, -1);
        {
            double L[4], sv[2];
            chol(S, L, 2);
            sv[0] = nu[0] / L[0]; sv[1] = (nu[1] - L[2] * sv[0]) / L[3];
            ring_inn(sv[0], sv[1]);
            A.inn_m = 2;
        }
        if (t - SL.last[j] > 40 && SL.nclose < 8) { SL.t_close[SL.nclose] = t; SL.close_lm[SL.nclose] = j; SL.nclose++; }
        SL.last[j] = t;
    }
    /* scores */
    double rt = rt_of(t), e = hypot(SL.x[0] - o[0], SL.x[1] - o[1]);
    ser_add(&A.ser[1], rt, 2 * sqrt(sl_at(0, 0) + sl_at(1, 1)));
    ser_add(&A.ser[2], rt, e);
    {
        double Pr[9], Pi[9], er[3] = {SL.x[0] - o[0], SL.x[1] - o[1], wrap_pi(SL.x[2] - o[4])}, ne = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Pr[i * 3 + j] = sl_at(i, j);
        if (inv(Pr, Pi, 3))
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) ne += er[i] * Pi[i * 3 + j] * er[j];
        if (k > 4) { SL.nees_r += ne; SL.nnr++; A.acc_nees[0] += ne; A.acc_ne[0]++; }
    }
    if (SL.nl) {
        double sm = 0, sw = 0;
        for (int s = 0; s < SL.nl; s++) {
            int a = 3 + 2 * s;
            double sg = sqrt(sl_at(a, a) + sl_at(a + 1, a + 1));
            sm += sg / SL.nl; sw = fmax(sw, sg);
        }
        ser_add(&A.ser[3], rt, sw);
        ser_add(&A.ser[4], rt, sm);
    }
    if (SL.rn < SLH) { SL.rx[SL.rn][0] = SL.x[0]; SL.rx[SL.rn][1] = SL.x[1]; SL.rx[SL.rn][2] = t; SL.rn++; }
    SL.t = t;
    A.finit = 1;
}

/* the landmarks' NEES at the end of a run (2 dof each) */
static double sl_map_nees(int *cnt) {
    double s = 0;
    *cnt = 0;
    for (int sl = 0; sl < SL.nl; sl++) {
        int a = 3 + 2 * sl, j = SL.id[sl];
        double P2[4] = {sl_at(a, a), sl_at(a, a + 1), sl_at(a + 1, a), sl_at(a + 1, a + 1)}, Pi[4];
        double e0 = SL.x[a] - SL.mx[j], e1 = SL.x[a + 1] - SL.my[j];
        if (!inv(P2, Pi, 2)) continue;
        s += e0 * (Pi[0] * e0 + Pi[1] * e1) + e1 * (Pi[2] * e0 + Pi[3] * e1);
        (*cnt)++;
    }
    return s;
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
    case A_MTT: mtt_init(); break;
    case A_RTS: rts_init(); break;
    case A_FUSE: fu_init(); break;
    case A_NUM: nu_init(); break;
    default: sl_init(); break;
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
        } else if (A.id == A_RTS) {
            if (0.5 + RT.n > s || RT.n >= RK) break;
            rts_step();
        } else if (A.id == A_FUSE) {
            if ((FU.tick + 1) * FU_DT > s) break;
            fu_tick();
        } else if (A.id == A_NUM) {
            if ((NU.k + 1) * NU_DT > s) break;
            nu_step();
        } else if (A.id == A_SLAM) {
            if ((SL.k + 1) * SL_DT > s) break;
            sl_step();
        } else if (A.id == A_MTT) {
            double t = (A.scan + 1) * A.rd.T;
            if (t > s) break;
            mtt_scan(A.scan);
            A.scan++;
            radar_scan(A.scan);
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
/*                        bg         panel      grid       frame      text       dim        truth      meas       clutter    f1         f2         f3         warn */
static const Pal PALS[] = {
    {"tactical", {"#030b09", "#06130f", "#0f3a2d", "#1f5b48", "#93ecc4", "#56a084", "#c8dccf", "#d8f59a", "#3f7f55", "#3fe6e0", "#f0b347", "#d77be0", "#ff5a4a"}, 1},
    {"night",    {"#080101", "#100303", "#3c0e08", "#5e1b10", "#ff6e4c", "#a63e27", "#ffb3a0", "#ffcfa4", "#7c2a1a", "#ff4a2e", "#ffa04a", "#ff79a6", "#ffe3a3"}, 1},
    {"paper",    {"#f1ede2", "#e9e4d5", "#d3ccb7", "#8a8270", "#2a2620", "#6e675a", "#57524a", "#2e2b26", "#aaa290", "#1f5fa8", "#c0461f", "#2f8a4a", "#c01818"}, 0},
    {"mono",     {"#040404", "#0b0b0b", "#282828", "#4c4c4c", "#e6e6e6", "#8c8c8c", "#bdbdbd", "#dadada", "#5c5c5c", "#ffffff", "#a8a8a8", "#787878", "#ffffff"}, 1},
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
static int nstyles, text_full, text_resets;
#define GH 8192
static struct { uint32_t key; Glyph g; } gtab[GH];

/* styles are keyed by pixel size, which follows the window: forget every
 * style and glyph (on a resize, or when a table fills) and start again */
static void text_reset(void) {
    nstyles = 0;
    memset(gtab, 0, sizeof gtab);
    if (atlas) memset(atlas, 0, (size_t)AW * AH);
    ax = ay = 1; ashelf = 0;
    atlas_dirty = 1;
    text_full = 0;
    text_resets++;
}

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
    if (nstyles == MAXSTYLE) { text_full = 1; return nstyles - 1; }
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
    else { text_full = 1; return &none; }
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
    if (ay + hh + 2 > AH) { gl->adv = (short)(g->advance.x >> 6); text_full = 1; return gl; }
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
    {"C_k = P_{k|k}F^⊤P^{−1}_{k+1|k}",
     "\\h{x}_{k|N} = \\h{x}_{k|k} + C_k(\\h{x}_{k+1|N} − \\h{x}_{k+1|k})",
     "P_{k|N} = P_{k|k} + C_k(P_{k+1|N} − P_{k+1|k})C^⊤_k",
     "\\r{fixed lag:}\\q \\h{x}_{k−L|k} ,\\; L \\r{steps back from} k ,\\; L = 8",
     "P_{k|N} ⪯ P_{k|k} \\q \\r{(the future only adds information)}", NULL},
    {"Y = P^{−1} ,\\q y = P^{−1}\\h{x} ,\\q Y_{k|k−1} = (FY^{−1}_{k−1}F^⊤ + Q)^{−1}",
     "Y_k = Y_{k|k−1} + ∑_sH^⊤_sR^{−1}_sH_s",
     "y_k = y_{k|k−1} + ∑_sH^⊤_sR^{−1}_s(z_s − h_s(\\h{x}) + H_s\\h{x})",
     "\\r{bits} = ½\\,\\r{log}_2(\\r{det}\\,Y_k / \\r{det}\\,Y_{k|k−1})",
     "\\r{CI:}\\q P^{−1} = ωP^{−1}_a + (1−ω)P^{−1}_b ,\\; ω = \\r{arg min det}\\,P", NULL},
    {"\\r{naive:}\\q P^+ = (I − KH)P",
     "\\r{Joseph:}\\q P^+ = (I − KH)P(I − KH)^⊤ + KRK^⊤",
     "\\r{square root:}\\q P = SS^⊤ ,\\q S^− = \\r{tria}(FS ,\\, Q^{1/2})",
     "\\r{tria}(R^{1/2}\\, HS \\,;\\, 0\\, S) = (S^{1/2}_e\\, 0 \\,;\\, \\h{K}\\, S^+) ,\\q K = \\h{K}S^{−1/2}_e",
     "P^−/P^+ ≈ 30^2/0.001^2 ≈ 10^9 \\;≫\\; 1/ε_{32} ≈ 1.7·10^7", NULL},
    {"x = (x_r ,\\, y_r ,\\, θ ,\\, m_1 ,\\, … ,\\, m_L)",
     "z_i = (‖m_i − p‖ ,\\; \\r{atan2}(Δy, Δx) − θ) + v",
     "P_{k|k−1} = G_xPG^⊤_x + G_uMG^⊤_u \\q \\r{(the map does not move)}",
     "m_{\\r{new}} = p + r(\\r{cos}(θ+β), \\r{sin}(θ+β)) ,\\q P_{mm} = G_rP_{rr}G^⊤_r + G_zRG^⊤_z",
     "H_i \\r{touches only the robot and} m_i \\r{; the gain reaches every landmark}", NULL},
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
        int e0 = (int)ceil(log10(c->y0) - 1e-6), e1 = (int)floor(log10(c->y1) + 1e-6), est = e1 - e0 > 6 ? (e1 - e0 + 5) / 6 : 1;
        for (int e = e0; e <= e1; e++) {
            float y = chart_y(c, &in, powf(10, (float)e));
            seg(in.x, y, in.x + in.w, y, lw, gc, (e - e0) % est ? 0.35f : 0.8f);
            if ((e - e0) % est) continue;
            char b[16];
            if (e < -2) snprintf(b, sizeof b, "1e%d", e);
            else if (e < 0) snprintf(b, sizeof b, "%.*f", -e, pow(10, e));
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
            if (s->v[k] != s->v[k]) {   /* NaN: a gap */
                if (n > 1) poly(P, n, 0, 1.6f * U, col, 0.95f, NULL);
                n = 0;
                continue;
            }
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
    if (st >= 1000) snprintf(b, sizeof b, "%g km", st / 1000);
    else if (st >= 1) snprintf(b, sizeof b, "%g m", st);
    else if (st >= 0.01) snprintf(b, sizeof b, "%g cm", st * 100);
    else snprintf(b, sizeof b, "%g mm", st * 1000);
    text(s_small, bx + bl + 6 * U, by + cap_h(s_small) * 0.5f, -1, b, pal[C_DIM], 1);
    /* north */
    float nx = vr.x + vr.w - 22 * U, ny = vr.y + 30 * U;
    seg(nx, ny + 10 * U, nx, ny - 10 * U, 1.4f * U, pal[C_DIM], 1);
    seg(nx, ny - 10 * U, nx - 4 * U, ny - 4 * U, 1.4f * U, pal[C_DIM], 1);
    seg(nx, ny - 10 * U, nx + 4 * U, ny - 4 * U, 1.4f * U, pal[C_DIM], 1);
    text(s_small, nx, ny + 12 * U + cap_h(s_small), 0, "N", pal[C_DIM], 1);
}

static void draw_radar(int rings) {
    const Radar *rd = &A.rd;
    float cx = WX(rd->x), cy = WY(rd->y);
    if (rings) {
        double st = nice_step(rd->rmax, 5);
        for (double r = st; r <= rd->rmax + 1; r += st) {
            circle(cx, cy, (float)(r * vk), fmaxf(U, 1), pal[C_GRID], 0.9f, 0);
            char b[16];
            snprintf(b, sizeof b, "%g", r / 1000);
            text(s_small, cx + (float)(r * vk) * 0.7071f + 3 * U, cy - (float)(r * vk) * 0.7071f - 3 * U, -1, b, pal[C_DIM], 0.9f);
        }
    }
    /* the site: a small diamond, nothing that turns */
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

static void np_plot(double s);
static void flush(void);
static Rect clip_cur;                   /* the panel being drawn, scissored */
static void clip_to(const Rect *r) {   /* scissor to r within the panel (NULL: back to the panel) */
    flush();
    Rect c = r ? *r : clip_cur;
    if (r) {
        float x0 = fmaxf(c.x, clip_cur.x), y0 = fmaxf(c.y, clip_cur.y);
        float x1 = fminf(c.x + c.w, clip_cur.x + clip_cur.w), y1 = fminf(c.y + c.h, clip_cur.y + clip_cur.h);
        c = (Rect){x0, y0, fmaxf(x1 - x0, 0), fmaxf(y1 - y0, 0)};
    }
    glScissor((int)c.x, H - (int)(c.y + c.h), (int)c.w, (int)c.h);
}
static void np_mats(const Rect *r);
static void np_meas(const Rect *r, double s);
static void np_table(const Rect *r, double s);
static void nu_camera(double s);
static void axes_box(const Rect *in);

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
    if (A.id == A_NUM) nu_camera(s);
    draw_grid();
    if (A.id >= A_RTS) { np_plot(s); return; }
    char lab[64];
    if (A.has_radar) draw_radar(1);
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
                float fa = smooth01((float)(sw / 0.6)) * (1 - smooth01((float)((sw - 3) / 2)));   /* a steady fade, no blinking */
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
            float ins = cs < 9 * U ? 0 : 1;
            rectf(cx + ins, cy + ins, cs - 2 * ins, cs - 2 * ins, pal[col], 0.1f + 0.7f * (float)fmin(fabs(c), 1));
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
    case A_RTS: return "8 s AGO: FILTERED P  \xc2\xb7  SMOOTHED P  \xc2\xb7  SMOOTHER GAIN C";
    case A_FUSE: return "INFORMATION Y = P\xe2\x81\xbb\xc2\xb9  \xc2\xb7  COVARIANCE P  (position, velocity, odometer bias)";
    case A_NUM: return "P AS STORED: NAIVE float32  \xc2\xb7  float64  \xc2\xb7  THE SQUARE ROOT S";
    case A_SLAM: return "CORRELATION OF THE WHOLE STATE  \xc2\xb7  robot (r) and every landmark";
    case A_PF: return "COVARIANCE P (particles)  \xc2\xb7  RANGE DISTRIBUTION";
    case A_MTT: return "ASSIGNMENT d\xc2\xb2 (track \xc3\x97 detection)  \xc2\xb7  P OF THE LATEST UPDATE";
    case A_IMM: return "COVARIANCE P (combined)  \xc2\xb7  GAIN K (CT model)";
    default: return "COVARIANCE P  \xc2\xb7  KALMAN GAIN K";
    }
}

static void draw_mats(const Rect *r, double s) {
    (void)s;
    if (A.id >= A_RTS) { np_mats(r); return; }
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
        else text(s_small, in.x, in.y + cs, -1, "no confirmed track near a detection", pal[C_DIM], 1);
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
    case A_RTS: return "ERROR AGAINST TIME  \xc2\xb7  filter and smoother, \xc2\xb1" "2\xcf\x83";
    case A_FUSE: return "INFORMATION ADDS UP  \xc2\xb7  COVARIANCE INTERSECTION";
    case A_NUM: return "EIGENVALUES OF P  \xc2\xb7  each filter, after the latest update";
    case A_SLAM: return "THE ROBOT'S VIEW  \xc2\xb7  landmarks relative to it, nose up";
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
    if (A.id >= A_RTS) { np_meas(r, s); return; }
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
    if (A.id >= A_RTS) { np_table(r, s); return; }
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
/* ---- the new acts on screen ---- */

/* the act's name and clock, top left of the plot; returns where the legend goes */
static void plot_head(double s, float *x, float *y, int lines) {
    *x = vr.x + 14 * U; *y = vr.y + 14 * U + cap_h(s_hudb);
    float h = 14 * U + cap_h(s_hudb) * 1.9f + cap_h(s_small) * (2.4f + 1.9f * lines + 1) + 8 * U;
    rectf(vr.x + 4 * U, vr.y + 4 * U, 470 * U, h, pal[C_BG], 0.72f);
    textf(s_hudb, *x, *y, -1, pal[C_TEXT], 1, "%d/%d  %s", A.id + 1, NACT, ACT_NAME[A.id]);
    *y += cap_h(s_hudb) * 1.9f;
    char tb[32];
    fmt_mmss(tb, sizeof tb, s);
    textf(s_small, *x, *y, -1, pal[C_DIM], 1, "T+%s  sim \xc3\x97%g", tb, A.TS);
    *y += cap_h(s_small) * 2.4f;
}

/* a 2x2 covariance [a b; b c] as an ellipse around (cx, cy) in a local
 * frame of k px per unit (y up); kk sigmas */
static void cov_ell_local(float cx, float cy, const double *P3, double k, double kk, float *pts) {
    double l1, l2, th, Q[3] = {P3[0], P3[1], P3[2]};
    cov_axes(Q, &l1, &l2, &th);
    ellipse_pts(cx, cy, fmaxf((float)(kk * sqrt(l1) * k), 1.0f), fmaxf((float)(kk * sqrt(l2) * k), 1.0f), (float)-th, pts);
}

/* a band of a world polyline (an outage, a canyon): a wide faint stroke */
static void path_band(const Path *p, double t0, double t1, const char *label, float a) {
    if (t1 <= t0) return;
    float P[2 * 121];
    for (int i = 0; i <= 120; i++) {
        double o[6];
        path_eval(p, t0 + (t1 - t0) * i / 120, o);
        P[2 * i] = WX(o[0]); P[2 * i + 1] = WY(o[1]);
    }
    poly(P, 121, 0, 16 * U, pal[C_WARN], 0.10f * a, NULL);
    poly_dash(P, 121, 0, fmaxf(U, 1), pal[C_WARN], 0.5f * a, 4 * U, 5 * U);
    if (label) text(s_small, P[120] + 12 * U, P[121] - 10 * U, -1, label, pal[C_WARN], 0.85f * a);
}

/* -------- RTS -------- */
static double rts_ease(double s) {
    return RT.n ? smooth01((float)((s - RT.t[RT.n - 1]) / A.TS / 0.5)) : 1;
}
static void rts_xs(int k, double e, double *x, double *P3) {
    const double *a = RT.xs[k], *A4 = RT.Ps[k];
    if (k < RT.n0 && k < RT.n - 1) {
        const double *b = RT.xs0[k], *B4 = RT.Ps0[k];
        x[0] = b[0] + (a[0] - b[0]) * e; x[1] = b[1] + (a[1] - b[1]) * e;
        P3[0] = B4[0] + (A4[0] - B4[0]) * e; P3[1] = B4[1] + (A4[1] - B4[1]) * e; P3[2] = B4[5] + (A4[5] - B4[5]) * e;
    } else { x[0] = a[0]; x[1] = a[1]; P3[0] = A4[0]; P3[1] = A4[1]; P3[2] = A4[5]; }
}

static void rts_plot(double s) {
    double win = 70;
    draw_truth(&A.tp[0], fmax(0, s - win - 10), s, 0.3f);
    for (int i = 0; i < 2; i++)
        if (RT.out[i][0] < s) path_band(&A.tp[0], RT.out[i][0], fmin(RT.out[i][1], s), "GPS OUTAGE", 1);
    int n = RT.n;
    if (!n) return;
    for (int k = 0; k < n; k++) {
        if (!RT.fix[k] || RT.t[k] < s - win) continue;
        float a = 1 - (float)((s - RT.t[k]) / win);
        disk(WX(RT.zx[k]), WY(RT.zy[k]), 2.0f * U, pal[C_MEAS], 0.75f * a);
    }
    double e = rts_ease(s);
    static float Pf[2 * RK], Ps[2 * RK], al[RK];
    int m = 0, k0 = n - 1;
    while (k0 > 0 && RT.t[k0 - 1] >= s - win) k0--;
    float pts[2 * ELN];
    for (int k = k0; k < n; k++, m++) {
        double x[2], P3[3];
        rts_xs(k, e, x, P3);
        Pf[2 * m] = WX(RT.xf[k][0]); Pf[2 * m + 1] = WY(RT.xf[k][1]);
        Ps[2 * m] = WX(x[0]); Ps[2 * m + 1] = WY(x[1]);
        al[m] = smooth01((float)((RT.t[k] - (s - win)) / 12));
        if ((n - 1 - k) % 3 == 0 && k < n - 1) {
            double Pf3[3] = {RT.Pf[k][0], RT.Pf[k][1], RT.Pf[k][5]};
            cov_ellipse(RT.xf[k][0], RT.xf[k][1], Pf3, 1, pts);
            poly_dash(pts, ELN, 1, fmaxf(U, 1), pal[C_F2], 0.55f * al[m], 3 * U, 3 * U);
            cov_ellipse(x[0], x[1], P3, 1, pts);
            fan(pts, ELN, WX(x[0]), WY(x[1]), pal[C_F1], 0.10f * al[m]);
            poly(pts, ELN, 1, 1.2f * U, pal[C_F1], 0.8f * al[m], NULL);
        }
    }
    poly(Pf, m, 0, 1.1f * U, pal[C_F2], 0.8f, al);
    poly(Ps, m, 0, 2.0f * U, pal[C_F1], 0.95f, al);
    if (glow_on) poly_k(Ps, m, 0, 6 * U, pal[C_F1], 0.10f, al, K_SOFT);
    /* the fixed-lag estimate, eight steps behind */
    int j = n - 1 - RTS_LAG;
    if (j >= 0) {
        double x[4], P3[3];
        rts_xs(j, e, x, P3);
        x[2] = RT.xs[j][2]; x[3] = RT.xs[j][3];
        draw_est(x, P3, C_F3, 1, "LAG 8 s", 0, 0, 0);
        double o[6];
        path_eval(&A.tp[0], RT.t[j], o);
        float h = 3.5f * U;
        seg(WX(o[0]) - h, WY(o[1]), WX(o[0]) + h, WY(o[1]), fmaxf(U, 1), pal[C_TRUTH], 0.8f);
        seg(WX(o[0]), WY(o[1]) - h, WX(o[0]), WY(o[1]) + h, fmaxf(U, 1), pal[C_TRUTH], 0.8f);
    }
    double x[4], P2[3];
    fdisp(&A.f[0], s, x, P2);
    draw_est(x, P2, C_F2, 1, "FILTER", 2, 0, 0);
    float lx, ly;
    plot_head(s, &lx, &ly, 7);
    legend_line(&ly, lx, C_TRUTH, 0, "truth");
    legend_line(&ly, lx, C_MEAS, 1, "GPS fix");
    legend_line(&ly, lx, C_F2, 0, "forward filter, 1\xcf\x83 dashed");
    legend_line(&ly, lx, C_F1, 0, "smoothed with all data so far");
    legend_line(&ly, lx, C_F3, 0, "fixed-lag smoother, 8 s");
    ly += cap_h(s_small);
    double sf = sqrt(RT.Pf[n - 1][0] + RT.Pf[n - 1][5]), sl = j >= 0 ? sqrt(RT.Ps[j][0] + RT.Ps[j][5]) : 0, sfj = j >= 0 ? sqrt(RT.Pf[j][0] + RT.Pf[j][5]) : 0;
    textf(s_small, lx, ly, -1, pal[C_DIM], 1, "\xcf\x83 now %.1f m;  8 s ago: filtered %.1f m, smoothed %.1f m", sf, sfj, sl);
}

static void rts_mats(const Rect *r) {
    Rect in = {r->x + 34 * U, r->y + 52 * U, r->w - 44 * U, r->h - 60 * U};
    int j = RT.n - 1 - RTS_LAG;
    if (j < 0) return;
    float gap = 40 * U, cs = fminf(fminf(in.h / 4.4f, (in.w - 2 * gap) / 12.0f), 48 * U);
    const double *M[3] = {RT.Pf[j], RT.Ps[j], RT.C[j]};
    static const char *nm[3] = {"P(k|k)  filtered", "P(k|N)  smoothed", "C(k)  smoother gain"};
    for (int i = 0; i < 3; i++) {
        float x = in.x + i * (4 * cs + gap);
        heat(x, in.y, cs, 4, 4, M[i], i == 2 ? 1 : 0, LBL4, LBL4, NULL);
        text(s_small, x, in.y - 20 * U, -1, nm[i], pal[i == 0 ? C_F2 : i == 1 ? C_F1 : C_DIM], 1);
    }
}

static void rts_meas(const Rect *r, double s) {
    Rect in = {r->x + 44 * U, r->y + 40 * U, r->w - 60 * U, r->h - 64 * U};
    float lw = fmaxf(U, 1), gh = 14 * U, hh = (in.h - gh) / 2;
    double win = 80, e = rts_ease(s);
    int n = RT.n, k0 = n - 1;
    if (n < 2) return;
    while (k0 > 0 && RT.t[k0 - 1] >= s - win) k0--;
    double sc = 12;
    for (int k = k0; k < n; k++) sc = fmax(sc, 2.3 * sqrt(fmax(RT.Pf[k][0], RT.Pf[k][5])));
    sc = fmin(sc, 150);
    #define TX(t) (in.x + in.w * (float)(1 - (s - (t)) / win))
    for (int c = 0; c < 2; c++) {
        Rect b = {in.x, in.y + c * (hh + gh), in.w, hh};
        axes_box(&b);
        float cy = b.y + b.h / 2, ky = (float)(b.h / 2 / sc);
        seg(b.x, cy, b.x + b.w, cy, lw, pal[C_GRID], 1);
        double st = nice_step(sc, 3);
        for (double v = st; v < sc; v += st) {
            seg(b.x, cy - (float)v * ky, b.x + b.w, cy - (float)v * ky, lw, pal[C_GRID], 0.5f);
            seg(b.x, cy + (float)v * ky, b.x + b.w, cy + (float)v * ky, lw, pal[C_GRID], 0.5f);
            textf(s_small, b.x - 5 * U, cy - (float)v * ky + cap_h(s_small) / 2, 1, pal[C_DIM], 1, "%g", v);
        }
        text(s_small, b.x + 6 * U, b.y + 6 * U + cap_h(s_small), -1, c ? "y error, m" : "x error, m", pal[C_DIM], 1);
        for (int i = 0; i < 2; i++) {
            double a0 = fmax(RT.out[i][0], s - win), a1 = fmin(RT.out[i][1], s);
            if (a1 > a0) rectf(TX(a0), b.y, TX(a1) - TX(a0), b.h, pal[C_WARN], 0.07f);
        }
        for (int pass = 0; pass < 2; pass++) {       /* filter, then smoother: +-2 sigma bands */
            int col = pass ? C_F1 : C_F2;
            for (int k = k0; k + 1 < n; k++) {
                double xa[2], Pa[3], xb[2], Pb[3];
                if (pass) { rts_xs(k, e, xa, Pa); rts_xs(k + 1, e, xb, Pb); }
                else {
                    xa[0] = RT.xf[k][0]; xa[1] = RT.xf[k][1]; Pa[0] = RT.Pf[k][0]; Pa[2] = RT.Pf[k][5];
                    xb[0] = RT.xf[k + 1][0]; xb[1] = RT.xf[k + 1][1]; Pb[0] = RT.Pf[k + 1][0]; Pb[2] = RT.Pf[k + 1][5];
                }
                double ea = xa[c] - RT.tr[k][c], eb = xb[c] - RT.tr[k + 1][c], sa = 2 * sqrt(c ? Pa[2] : Pa[0]), sb = 2 * sqrt(c ? Pb[2] : Pb[0]);
                float x0 = TX(RT.t[k]), x1 = TX(RT.t[k + 1]);
                float y0a = cy - (float)clampd(ea + sa, -sc, sc) * ky, y0b = cy - (float)clampd(ea - sa, -sc, sc) * ky;
                float y1a = cy - (float)clampd(eb + sb, -sc, sc) * ky, y1b = cy - (float)clampd(eb - sb, -sc, sc) * ky;
                float fa = pass ? 0.22f : 0.13f;
                V(x0, y0a, 0, 0, pal[col], fa, K_FLAT, 0); V(x1, y1a, 0, 0, pal[col], fa, K_FLAT, 0); V(x1, y1b, 0, 0, pal[col], fa, K_FLAT, 0);
                V(x0, y0a, 0, 0, pal[col], fa, K_FLAT, 0); V(x1, y1b, 0, 0, pal[col], fa, K_FLAT, 0); V(x0, y0b, 0, 0, pal[col], fa, K_FLAT, 0);
                seg(x0, cy - (float)clampd(ea, -sc, sc) * ky, x1, cy - (float)clampd(eb, -sc, sc) * ky, (pass ? 1.8f : 1.1f) * U, pal[col], 0.95f);
            }
        }
        for (int k = k0; k < n; k++) {
            if (!RT.fix[k]) continue;
            double ez = (c ? RT.zy[k] : RT.zx[k]) - RT.tr[k][c];
            if (fabs(ez) < sc) disk(TX(RT.t[k]), cy - (float)ez * ky, 1.6f * U, pal[C_MEAS], 0.7f);
        }
    }
    #undef TX
    float y = in.y + in.h + 4 * U + cap_h(s_small);
    text(s_small, in.x, y, -1, "\xe2\x86\x90 80 s", pal[C_DIM], 1);
    text(s_small, in.x + in.w, y, 1, "now", pal[C_DIM], 1);
    float lx = in.x + in.w * 0.3f;
    lx += text(s_small, lx, y, -1, "\xc2\xb1" "2\xcf\x83: ", pal[C_DIM], 1);
    lx += text(s_small, lx, y, -1, "filter  ", pal[C_F2], 1);
    lx += text(s_small, lx, y, -1, "smoother  ", pal[C_F1], 1);
    text(s_small, lx, y, -1, "GPS", pal[C_MEAS], 1);
}

/* the table helper: a header row and rows of cells */
static float table_head(const Rect *r, int nc, const float *cx, const char *const *hd, float *x, float *w) {
    *x = r->x + 12 * U; *w = r->w - 24 * U;
    float y = r->y + 40 * U + cap_h(s_hud);
    for (int c = 0; c < nc; c++) text(s_small, *x + *w * cx[c], y, -1, hd[c], pal[C_DIM], 1);
    return y + cap_h(s_hud) * 2.0f;
}

static void rts_table(const Rect *r, double s) {
    (void)s;
    static const float cx[] = {0, 0.3f, 0.48f, 0.66f, 0.84f};
    static const char *hd[] = {"ESTIMATE", "RMS ERR m", "NEES (2)", "\xcf\x83 NOW m", "LATENCY"};
    float x, w, lh = cap_h(s_hud) * 2.0f, y = table_head(r, 5, cx, hd, &x, &w);
    static const char *nm[3] = {"forward filter", "fixed lag", "smoothed"};
    static const char *lat[3] = {"0 s", "8 s", "all data"};
    static const int col[3] = {C_F2, C_F3, C_F1};
    int n = RT.n, j = n - 1 - RTS_LAG;
    for (int k = 0; k < 3; k++) {
        text(s_hud, x, y, -1, nm[k], pal[col[k]], 1);
        if (RT.ne[k]) {
            textf(s_hud, x + w * cx[1], y, -1, pal[C_TEXT], 1, "%.2f", sqrt(RT.se[k] / RT.ne[k]));
            textf(s_hud, x + w * cx[2], y, -1, pal[C_TEXT], 1, "%.2f", RT.sn[k] / RT.ne[k]);
        }
        double sg = k == 0 && n ? sqrt(RT.Pf[n - 1][0] + RT.Pf[n - 1][5]) : j >= 0 ? sqrt(RT.Ps[j][0] + RT.Ps[j][5]) : 0;
        if (k == 2 && n) sg = sqrt(RT.Ps[n / 2][0] + RT.Ps[n / 2][5]);
        textf(s_hud, x + w * cx[3], y, -1, pal[C_TEXT], 1, "%.2f", sg);
        text(s_hud, x + w * cx[4], y, -1, lat[k], pal[C_TEXT], 1);
        y += lh;
    }
    y += lh * 0.5f;
    int nf = 0;
    for (int k = 0; k < n; k++) nf += RT.fix[k];
    textf(s_small, x, y, -1, pal[C_DIM], 1, "%d steps, %d GPS fixes; the smoothed \xcf\x83 is at mid-history", n, nf);
    y += lh * 0.8f;
    text(s_small, x, y, -1, "RTS = batch least squares over the whole past (--check)", pal[C_DIM], 1);
}

/* -------- fusion -------- */
static void fu_site(double x, double y, const char *label, int square) {
    float cx = WX(x), cy = WY(y), h = 5.5f * U, lw = 1.4f * U;
    if (square) {
        seg(cx - h, cy - h, cx + h, cy - h, lw, pal[C_TEXT], 1); seg(cx + h, cy - h, cx + h, cy + h, lw, pal[C_TEXT], 1);
        seg(cx + h, cy + h, cx - h, cy + h, lw, pal[C_TEXT], 1); seg(cx - h, cy + h, cx - h, cy - h, lw, pal[C_TEXT], 1);
    } else {
        seg(cx, cy - h * 1.2f, cx + h, cy + h * 0.7f, lw, pal[C_TEXT], 1); seg(cx + h, cy + h * 0.7f, cx - h, cy + h * 0.7f, lw, pal[C_TEXT], 1);
        seg(cx - h, cy + h * 0.7f, cx, cy - h * 1.2f, lw, pal[C_TEXT], 1);
    }
    text(s_small, cx + 9 * U, cy + 4 * U, -1, label, pal[C_TEXT], 0.9f);
}

/* the zoom: mode 0, information adding up in the latest fix; mode 1, CI */
static void fu_zoom_in(const Rect *b, double s, int mode, int small);
static void fu_zoom(const Rect *b, double s, int mode, int small) {
    clip_to(b);
    fu_zoom_in(b, s, mode, small);
    clip_to(NULL);
}
static void fu_zoom_in(const Rect *b, double s, int mode, int small) {
    axes_box(b);
    float cx = b->x + b->w / 2, cy = b->y + b->h / 2, pts[2 * ELN], lw = fmaxf(U, 1);
    float ty = b->y + 6 * U + cap_h(s_small);
    if (mode == 0) {
        /* cycle GPS, beacon 1, beacon 2, two seconds each; crossfade */
        static const int seq[3] = {SN_GPS, SN_B1, SN_B2};
        double ph = s / 2.0;
        int slot = (int)floor(ph) % 3, sn = seq[slot];
        float a = smooth01((float)((ph - floor(ph)) / 0.25)) * (1 - smooth01((float)((ph - floor(ph) - 0.85) / 0.15)));
        text(s_small, b->x + 6 * U, ty, -1, "Y\xe2\x81\xba = Y\xe2\x81\xbb + H\xe1\xb5\x80R\xe2\x81\xbb\xc2\xb9H", pal[C_DIM], 1);
        textf(s_small, b->x + b->w - 6 * U, ty, 1, pal[SN_COL[sn]], a, "%s", SN_NAME[sn]);
        if (FU.upd_t[sn] < 0 || s - FU.upd_t[sn] > 6) {
            text(s_small, cx, cy, 0, sn == SN_GPS ? "no GPS fix lately" : "no range lately", pal[C_WARN], a);
            return;
        }
        const double *Pm = FU.upd_Pm[sn], *Pp = FU.upd_Pp[sn], *J = FU.upd_J[sn];
        double l1, l2, th;
        cov_axes(Pm, &l1, &l2, &th);
        double k = fminf(b->w, b->h) / 2 * 0.8 / (2.0 * sqrt(l1));
        /* the sensor's information as a covariance: its null direction runs off the box */
        double jl1, jl2, jth;
        cov_axes(J, &jl1, &jl2, &jth);
        float big = fmaxf(b->w, b->h);
        float ax = (float)(jl1 > 0 ? 2 / sqrt(jl1) * k : big), ay = (float)(jl2 > jl1 * 1e-6 ? 2 / sqrt(jl2) * k : big);
        ellipse_pts(cx, cy, fminf(ax, big), fminf(ay, big), (float)-jth, pts);
        fan(pts, ELN, cx, cy, pal[SN_COL[sn]], 0.10f * a);
        poly_dash(pts, ELN, 1, lw, pal[SN_COL[sn]], 0.8f * a, 4 * U, 3 * U);
        cov_ell_local(cx, cy, Pm, k, 2, pts);
        poly_dash(pts, ELN, 1, 1.3f * U, pal[C_F1], 0.75f * a, 5 * U, 4 * U);
        cov_ell_local(cx, cy, Pp, k, 2, pts);
        fan(pts, ELN, cx, cy, pal[C_F1], 0.22f * a);
        poly(pts, ELN, 1, 1.8f * U, pal[C_F1], a, NULL);
        if (!small) {
            float y = b->y + b->h - 8 * U - cap_h(s_small) * 1.9f * 2;
            legend_line(&y, b->x + 6 * U, C_F1, 0, "prior, then posterior (2\xcf\x83)");
            legend_line(&y, b->x + 6 * U, SN_COL[sn], 0, "what the sensor knows (2\xcf\x83 of (H\xe1\xb5\x80R\xe2\x81\xbb\xc2\xb9H)\xe2\x81\xbb\xc2\xb9)");
        }
        textf(s_small, b->x + 6 * U, b->y + b->h - 6 * U, -1, pal[C_DIM], 1, "\xc2\xb1%.1f m", b->w / 2 / k);
    } else {
        text(s_small, b->x + 6 * U, ty, -1, "COVARIANCE INTERSECTION", pal[C_DIM], 1);
        if (FU.ci_t < 0) return;
        textf(s_small, b->x + b->w - 6 * U, ty, 1, pal[C_DIM], 1, "\xcf\x89 %.2f", FU.ci_w);
        double ext = 0, l1, l2, th;
        cov_axes(FU.ci_A, &l1, &l2, &th); ext = fmax(ext, 2 * sqrt(l1));
        cov_axes(FU.ci_B, &l1, &l2, &th); ext = fmax(ext, 2 * sqrt(l1));
        ext = fmax(ext, fmax(fabs(FU.ci_b[0] - FU.ci_a[0]), fabs(FU.ci_b[1] - FU.ci_a[1])) + 1);
        double k = fminf(b->w, b->h) / 2 * 0.82 / ext;
        #define LX(v) (cx + (float)(((v) - FU.ci_a[0]) * k))
        #define LY(v) (cy - (float)(((v) - FU.ci_a[1]) * k))
        cov_ell_local(LX(FU.ci_a[0]), LY(FU.ci_a[1]), FU.ci_A, k, 2, pts);
        poly(pts, ELN, 1, 1.4f * U, pal[C_F1], 0.9f, NULL);
        cov_ell_local(LX(FU.ci_b[0]), LY(FU.ci_b[1]), FU.ci_B, k, 2, pts);
        poly(pts, ELN, 1, 1.4f * U, pal[C_F3], 0.9f, NULL);
        cov_ell_local(LX(FU.ci_n[0]), LY(FU.ci_n[1]), FU.ci_N, k, 2, pts);
        poly_dash(pts, ELN, 1, 1.4f * U, pal[C_WARN], 0.9f, 3 * U, 3 * U);
        cov_ell_local(LX(FU.ci_c[0]), LY(FU.ci_c[1]), FU.ci_C, k, 2, pts);
        fan(pts, ELN, LX(FU.ci_c[0]), LY(FU.ci_c[1]), pal[C_TEXT], 0.08f);
        poly(pts, ELN, 1, 2.0f * U, pal[C_TEXT], 0.95f, NULL);
        double o[6];
        path_eval(&A.tp[0], FU.ci_t, o);
        float h = 4 * U, X = LX(o[0]), Y = LY(o[1]);
        seg(X - h, Y, X + h, Y, lw, pal[C_TRUTH], 1); seg(X, Y - h, X, Y + h, lw, pal[C_TRUTH], 1);
        #undef LX
        #undef LY
        if (!small) {
            float y = b->y + b->h - 8 * U - cap_h(s_small) * 1.9f * 3;
            legend_line(&y, b->x + 6 * U, C_F1, 0, "ours, fused");
            legend_line(&y, b->x + 6 * U, C_F3, 0, "the base's (radar + our own reports)");
            legend_line(&y, b->x + 6 * U, C_WARN, 0, "naive: as if independent");
            legend_line(&y, b->x + 6 * U, C_TEXT, 0, "CI: safe for any correlation");
        }
        textf(s_small, b->x + 6 * U, b->y + b->h - 6 * U, -1, pal[C_DIM], 1, "\xc2\xb1%.1f m, 2\xcf\x83", b->w / 2 / k);
    }
}

static void fu_plot(double s) {
    draw_truth(&A.tp[0], fmax(0, s - 50), s, 0.3f);
    if (FU.canyon[0] < s) path_band(&A.tp[0], FU.canyon[0], fmin(FU.canyon[1], s), "URBAN CANYON: NO GPS", 1);
    for (int i = A.dhn - 1; i >= 0 && i >= A.dhn - 30; i--) {
        const Det *d = &A.dh[i % DH];
        if (d->t > s) continue;
        float a = 1 - (float)((s - d->t) / 30), x = WX(d->x), y = WY(d->y);
        if (a <= 0) continue;
        if (s - d->t < 2.5) circle(x, y, (float)(FU_GPS * vk), fmaxf(U, 1), pal[C_MEAS], 0.35f * a * (float)(1 - (s - d->t) / 2.5), 0);
        disk(x, y, 2.2f * U, pal[C_MEAS], a);
    }
    double x[4], P2[3];
    fdisp(&A.f[0], s, x, P2);
    /* beacons, their latest ranges as arcs, the base station and its bearing */
    for (int j = 0; j < 2; j++) {
        fu_site(FU.bx[j], FU.by[j], j ? "B2" : "B1", 0);
        double age = s - FU.rng_t[j];
        if (age < 0 || age > 3.5) continue;
        float a = (1 - smooth01((float)((age - 1.5) / 2))) * smooth01((float)(age / 0.3));
        double c = atan2(x[1] - FU.by[j], x[0] - FU.bx[j]), span = fmin(60 / fmax(FU.rng_r[j], 1), 0.6);
        for (int d = -1; d <= 1; d++) {
            float P[2 * 41];
            double rr = FU.rng_r[j] + 2 * d * FU_RNG;
            for (int i = 0; i <= 40; i++) {
                double b = c - span + 2 * span * i / 40;
                P[2 * i] = WX(FU.bx[j] + rr * cos(b)); P[2 * i + 1] = WY(FU.by[j] + rr * sin(b));
            }
            if (d) poly_dash(P, 41, 0, fmaxf(U, 1), pal[C_F3], 0.5f * a, 3 * U, 3 * U);
            else poly(P, 41, 0, 1.5f * U, pal[C_F3], 0.85f * a, NULL);
        }
    }
    fu_site(FU.bx[2], FU.by[2], "BASE", 1);
    {
        double age = s - FU.brg_t;
        if (age >= 0 && age < 2.5) {
            float a = 0.45f * (1 - smooth01((float)((age - 1) / 1.5)));
            double L = FU.brg_r, c = cos(FU.brg_b), sn = sin(FU.brg_b), h = 3 * FU_BRNG;
            seg(WX(FU.bx[2]), WY(FU.by[2]), WX(FU.bx[2] + L * c), WY(FU.by[2] + L * sn), 1.1f * U, pal[C_F3], a);
            seg(WX(FU.bx[2] + L * c - h * sn), WY(FU.by[2] + L * sn + h * c), WX(FU.bx[2] + L * c + h * sn), WY(FU.by[2] + L * sn - h * c), 1.4f * U, pal[C_F3], 2 * a);
        }
    }
    draw_trail(&A.ft[1], C_F2, 0.85f, 1.4f * U);
    if (A.ft[1].n) {
        int k = TRAIL_AT(&A.ft[1], A.ft[1].n - 1);
        float dx = WX(A.ft[1].x[k]), dy = WY(A.ft[1].y[k]);
        ring(dx, dy, 3.5f * U, pal[C_F2], 1);
        text(s_small, dx + 7 * U, dy + 12 * U, -1, "DEAD RECKONING", pal[C_F2], 0.9f);
    }
    draw_trail(&A.ft[2], C_F3, 0.45f, 1.0f * U);
    draw_trail(&A.ft[0], C_F1, 0.8f, 1.5f * U);
    draw_est(x, P2, C_F1, 1, "FUSED", 2, 1, 0);
    float lx, ly;
    plot_head(s, &lx, &ly, 6);
    legend_line(&ly, lx, C_TRUTH, 0, "truth");
    legend_line(&ly, lx, C_MEAS, 1, "GPS fix (\xcf\x83 4 m)");
    legend_line(&ly, lx, C_F3, 0, "range to a beacon (\xcf\x83 1 m)");
    legend_line(&ly, lx, C_F2, 0, "odometry alone (bias \xe2\x89\x88 0.05 m/s)");
    legend_line(&ly, lx, C_F1, 0, "information filter, all four");
    legend_line(&ly, lx, C_F3, 0, "base's track: its radar + our reports");
    /* the zooms, top right, when the plot is wide enough to spare them */
    if (role == R_FRONT) {
        float zs = fminf(vr.w, vr.h) * 0.3f;
        if (zs > 150 * U) zs = 150 * U + (zs - 150 * U) * 0.5f;
        Rect z0 = {vr.x + vr.w - zs - 46 * U, vr.y + 14 * U, zs, zs}, z1 = {z0.x, z0.y + zs + 10 * U, zs, zs};
        rectf(z0.x, z0.y, z0.w, z0.h * 2 + 10 * U, pal[C_BG], 0.82f);
        fu_zoom(&z0, s, 0, 1);
        fu_zoom(&z1, s, 1, 1);
    }
}

static void fu_mats(const Rect *r) {
    Rect in = {r->x + 34 * U, r->y + 52 * U, r->w - 44 * U, r->h - 60 * U};
    static const char *const LB[] = {"x", "y", "vx", "vy", "bx", "by"};
    float gap = 46 * U, cs = fminf(fminf(in.h / 6.4f, (in.w - gap) / 12.0f), 44 * U);
    heat(in.x, in.y, cs, 6, 6, FU.Y, 0, LB, LB, NULL);
    text(s_small, in.x, in.y - 20 * U, -1, "Y = P\xe2\x81\xbb\xc2\xb9  information", pal[C_F1], 1);
    heat(in.x + 6 * cs + gap, in.y, cs, 6, 6, A.f[0].P, 0, LB, LB, NULL);
    text(s_small, in.x + 6 * cs + gap, in.y - 20 * U, -1, "P  covariance", pal[C_DIM], 1);
}

static void fu_table(const Rect *r, double s) {
    (void)s;
    static const float cx[] = {0, 0.26f, 0.42f, 0.58f, 0.76f};
    static const char *hd[] = {"SENSOR", "RATE", "\xcf\x83", "UPDATES", "BITS GAINED"};
    float x, w, lh = cap_h(s_hud) * 2.0f, y = table_head(r, 5, cx, hd, &x, &w);
    static const char *rate[NSN] = {"1 Hz", "5 Hz", "0.5 Hz", "0.5 Hz"}, *sg[NSN] = {"4 m", "0.06 m/s", "1 m", "1 m"};
    for (int k = 0; k < NSN; k++) {
        text(s_hud, x, y, -1, SN_NAME[k], pal[SN_COL[k]], 1);
        text(s_hud, x + w * cx[1], y, -1, rate[k], pal[C_TEXT], 1);
        text(s_hud, x + w * cx[2], y, -1, sg[k], pal[C_TEXT], 1);
        textf(s_hud, x + w * cx[3], y, -1, pal[C_TEXT], 1, "%d", FU.nupd[k]);
        textf(s_hud, x + w * cx[4], y, -1, pal[C_TEXT], 1, "%.0f", FU.bits[k]);
        y += lh;
    }
    text(s_hud, x, y, -1, "prediction", pal[C_DIM], 1);
    textf(s_hud, x + w * cx[4], y, -1, pal[C_TEXT], 1, "%.0f", FU.bits_pred);
    y += lh * 1.4f;
    if (FU.nci) {
        textf(s_small, x, y, -1, pal[C_DIM], 1, "mean position NEES (expect 2):  fused %.2f   naive merge %.2f   CI %.2f",
              FU.nees_a / FU.nci, FU.nees_n / FU.nci, FU.nees_c / FU.nci);
        y += lh * 0.8f;
    }
    textf(s_small, x, y, -1, pal[C_DIM], 1, "odometer bias: true (%+.3f, %+.3f)  estimated (%+.3f, %+.3f) m/s",
          FU.bias[0], FU.bias[1], A.f[0].x[4], A.f[0].x[5]);
}

/* -------- numerics -------- */
static void nu_camera(double s) {
    /* the window follows 7 sigma of the reference in log space, but at a
     * limited pace: out at 3 decades a second, in at 1.4, then a light
     * average to round off the corners; a function of the history only */
    double c = NAN, last[6];
    int i0 = NU.hn - 1, nl = 0;
    while (i0 > 0 && NU.ht[i0 - 1] > s - 8) i0--;
    for (int i = i0; i < NU.hn && NU.ht[i] <= s; i++) {
        double tgt = clampd(NU.hs[i] + log10(7.0), log10(0.012), log10(260.0));
        if (c != c) c = tgt;
        else c += clampd(tgt - c, -1.4 * NU_DT, 3.0 * NU_DT);
        last[nl++ % 6] = c;
    }
    double w = 300;
    if (nl) {
        int m = nl < 6 ? nl : 6;
        double a = 0;
        for (int k = 0; k < m; k++) a += last[k];
        w = pow(10, a / m);
    }
    vcx = NU.xt[0]; vcy = NU.xt[1];
    if (NU.txn) {                       /* the truth at s */
        int k = (int)clampd(s / NU_DT - 1, 0, NU.txn - 1);
        vcx = NU.tx[k][0]; vcy = NU.tx[k][1];
    }
    vk = fminf(vr.w, vr.h) / (float)w;
}

static void nu_plot(double s) {
    float lw = fmaxf(U, 1), pts[2 * ELN];
    /* the truth's trail */
    {
        static float P[2 * 240];
        int n = 0;
        for (int i = NU.txn - 1; i >= 0 && n < 240; i -= 1) {
            if ((i + 1) * NU_DT > s) continue;
            if ((i + 1) * NU_DT < s - 3) break;
            P[2 * n] = WX(NU.tx[i][0]); P[2 * n + 1] = WY(NU.tx[i][1]); n++;
        }
        poly(P, n, 0, 1.1f * U, pal[C_TRUTH], 0.35f, NULL);
        float h = 4 * U;
        seg(vr.x + vr.w / 2 - h, vr.y + vr.h / 2, vr.x + vr.w / 2 + h, vr.y + vr.h / 2, lw, pal[C_TRUTH], 0.9f);
        seg(vr.x + vr.w / 2, vr.y + vr.h / 2 - h, vr.x + vr.w / 2, vr.y + vr.h / 2 + h, lw, pal[C_TRUTH], 0.9f);
    }
    if (!NU.k) return;
    double since = s - NU.t_cue;
    /* the reference, then the three float filters */
    for (int f = NNF; f >= 0; f--) {
        int col = f == NNF ? C_TRUTH : NF_COL[f];
        double x0, y0, P3[3];
        if (f == NNF) { x0 = NU.xr[0]; y0 = NU.xr[1]; P3[0] = NU.Pr[0]; P3[1] = NU.Pr[1]; P3[2] = NU.Pr[5]; }
        else {
            x0 = NU.x[f][0]; y0 = NU.x[f][1];
            double *P = NU.Pd[f];
            if (since < NU_DT * 1.5) nu_Pd(f, P);
            P3[0] = P[0]; P3[1] = 0.5 * (P[1] + P[4]); P3[2] = P[5];
        }
        float sx = WX(x0), sy = WY(y0), ldy = (float)(f == NNF ? 3 : f) * 15 * U;
        const char *lab = f == NNF ? "float64" : NF_NAME[f];
        int indef = f < NNF && (NU.bad[f] || NU.broken[f] || P3[0] <= 0 || P3[2] <= 0 || P3[0] * P3[2] - P3[1] * P3[1] <= 0);
        if (indef) {
            float h = 9 * U;
            seg(sx - h, sy - h, sx + h, sy + h, 2 * U, pal[C_WARN], 1);
            seg(sx - h, sy + h, sx + h, sy - h, 2 * U, pal[C_WARN], 1);
            textf(s_hudb, sx - 12 * U, sy + 4 * U, 1, pal[C_WARN], 1, "%s: %s", lab, NU.broken[f] ? "DIVERGED" : "P \xe2\x8a\x81 0");
            continue;
        }
        cov_ellipse(x0, y0, P3, 1, pts);
        if (f == NF_SQRT) fan(pts, ELN, sx, sy, pal[col], 0.10f);
        poly(pts, ELN, 1, (f == NNF ? 1.2f : 1.6f) * U, pal[col], 0.9f, NULL);
        cov_ellipse(x0, y0, P3, 2, pts);
        poly_dash(pts, ELN, 1, lw, pal[col], 0.5f, 5 * U, 4 * U);
        ring(sx, sy, 3.4f * U, pal[col], 1);
        text(s_hudb, sx + 8 * U, sy - 7 * U + ldy, -1, lab, pal[col], 1);
    }
    float lx, ly;
    plot_head(s, &lx, &ly, 7);
    legend_line(&ly, lx, C_TRUTH, 0, "truth; float64 reference (Joseph)");
    for (int f = NNF - 1; f >= 0; f--) legend_line(&ly, lx, NF_COL[f], 0, f == NF_SQRT ? "float32 square root" : f == NF_JOSEPH ? "float32 Joseph" : "float32 naive (I \xe2\x88\x92 KH)P");
    ly += cap_h(s_small);
    textf(s_small, lx, ly, -1, pal[C_DIM], 1, "cue %d: \xcf\x83 30 m \xe2\x86\x92 laser \xcf\x83 1 mm, %.1f s ago", NU.cyc + 1, fmax(since, 0));
    ly += cap_h(s_small) * 1.9f;
    textf(s_small, lx, ly, -1, pal[C_DIM], 1, "naive P lost definiteness after %d of %d cues", NU.nbad[NF_NAIVE] > NU.cyc + 1 ? NU.cyc + 1 : NU.nbad[NF_NAIVE], NU.cyc + 1);
}

/* compact scientific: 1.2e-6 */
static void fmt_sci(char *b, size_t n, double v) {
    if (v == 0) { snprintf(b, n, "0"); return; }
    if (!isfinite(v)) { snprintf(b, n, "nan"); return; }
    int e = (int)floor(log10(fabs(v)));
    double m = v / pow(10, e);
    if (fabs(m) >= 9.95) { m /= 10; e++; }
    snprintf(b, n, "%.*f%s%d", fabs(m - round(m)) < 0.05 ? 0 : 1, m, "e", e);
}

/* a covariance-like matrix with exponents, invalid correlations in warning colour */
static void heat_sci(float x, float y, float cs, int n, const double *M, const char *const *lb, int lower) {
    int numbers = cs >= 30 * U, sc = style(F_MONO, fminf(10 * U, cs * 0.24f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double v = M[i * n + j], d = M[i * n + i] * M[j * n + j], c = d > 0 ? v / sqrt(d) : 2;
            if (lower) c = j <= i ? 0.6 : 0;
            int bad = !lower && (fabs(c) > 1 + 1e-6 || M[i * n + i] <= 0 || (i != j && fabs(v - M[j * n + i]) > 1e-3 * fmax(fabs(v), 1e-30)));
            int col = bad ? C_WARN : c >= 0 ? C_F1 : C_F2;
            float cx = x + j * cs, cy = y + i * cs;
            rectf(cx + 1, cy + 1, cs - 2, cs - 2, pal[col], bad ? 0.75f : 0.1f + 0.7f * (float)fmin(fabs(c), 1));
            if (numbers && (!lower || j <= i)) {
                char b[24];
                fmt_sci(b, sizeof b, v);
                text(sc, cx + cs / 2, cy + cs / 2 + cap_h(sc) / 2, 0, b, pal[C_TEXT], 0.95f);
            }
        }
    for (int i = 0; lb && i < n; i++) {
        text(s_small, x - 5 * U, y + i * cs + cs / 2 + cap_h(s_small) / 2, 1, lb[i], pal[C_DIM], 1);
        text(s_small, x + i * cs + cs / 2, y - 5 * U, 0, lb[i], pal[C_DIM], 1);
    }
}

static void nu_mats(const Rect *r) {
    Rect in = {r->x + 34 * U, r->y + 52 * U, r->w - 44 * U, r->h - 60 * U};
    if (!NU.k) return;
    float gap = 40 * U, cs = fminf(fminf(in.h / 4.4f, (in.w - 2 * gap) / 12.0f), 52 * U);
    double S[16];
    for (int i = 0; i < 16; i++) S[i] = NU.P[NF_SQRT][i];
    const double *M[3] = {NU.Pd[NF_NAIVE], NU.Pr, S};
    static const char *nm[3] = {"naive float32 P", "float64 P", "square root S, P = SS\xe1\xb5\x80"};
    static const int col[3] = {C_WARN, C_TRUTH, C_F1};
    for (int i = 0; i < 3; i++) {
        float x = in.x + i * (4 * cs + gap);
        heat_sci(x, in.y, cs, 4, M[i], LBL4, i == 2);
        text(s_small, x, in.y - 20 * U, -1, nm[i], pal[col[i]], 1);
    }
}

static void nu_meas(const Rect *r, double s) {
    (void)s;
    Rect in = {r->x + 90 * U, r->y + 44 * U, r->w - 110 * U, r->h - 76 * U};
    float lw = fmaxf(U, 1);
    double e0 = -14, e1 = 4;
    #define EX(v) (in.x + in.w * (float)((v - e0) / (e1 - e0)))
    for (int e = (int)e0; e <= (int)e1; e += 2) {
        float x = EX(e);
        seg(x, in.y, x, in.y + in.h, lw, pal[C_GRID], 0.7f);
        textf(s_small, x, in.y + in.h + 4 * U + cap_h(s_small), 0, pal[C_DIM], 1, "1e%d", e);
    }
    float ex = EX(log10(NU_SIG * NU_SIG));
    seg(ex, in.y, ex, in.y + in.h, 1.4f * U, pal[C_MEAS], 0.6f);
    text(s_small, ex + 4 * U, in.y + cap_h(s_small) + 2 * U, -1, "R = (1 mm)\xc2\xb2", pal[C_MEAS], 0.9f);
    float rh = in.h / 4;
    for (int f = NNF; f >= 0; f--) {
        int row = NNF - f, col = f == NNF ? C_TRUTH : NF_COL[f];
        float y = in.y + rh * row + rh * 0.5f;
        text(s_hud, in.x - 8 * U, y + cap_h(s_hud) / 2, 1, f == NNF ? "float64" : NF_NAME[f], pal[col], 1);
        seg(in.x, y, in.x + in.w, y, lw, pal[C_GRID], 1);
        if (!NU.k || (f < NNF && NU.broken[f])) {
            if (f < NNF && NU.broken[f]) text(s_small, in.x + 6 * U, y - 6 * U, -1, "diverged: S not positive", pal[C_WARN], 1);
            continue;
        }
        const double *ev = NU.eig[f];
        for (int i = 0; i < 4; i++) {
            double v = ev[i];
            float x = EX(clampd(log10(fmax(fabs(v), 1e-30)), e0, e1)), h = rh * 0.28f;
            if (v > 0) { seg(x, y - h, x, y + h, 2.2f * U, pal[col], 1); }
            else {
                seg(x, y - h, x, y + h, 2.2f * U, pal[C_WARN], 1);
                text(s_hudb, x, y - h - 3 * U, 0, "\xe2\x88\x92", pal[C_WARN], 1);
            }
        }
        if (f < NNF) textf(s_small, in.x + in.w, y + rh * 0.42f, 1, pal[C_DIM], 1, "asymmetry %.0e   \xe2\x80\x96" "\xce\x94P\xe2\x80\x96 %.0e", NU.asym[f], NU.rel[f]);
    }
    #undef EX
    text(s_small, r->x + 12 * U, r->y + r->h - 10 * U, -1, "|\xce\xbb| of P on a log scale; red: negative (P is not a covariance)", pal[C_DIM], 1);
}

static void nu_table(const Rect *r, double s) {
    (void)s;
    static const float cx[] = {0, 0.17f, 0.36f, 0.53f, 0.7f, 0.86f};
    static const char *hd[] = {"FILTER", "P", "\xce\xbb MIN", "\xe2\x80\x96\xce\x94P\xe2\x80\x96 REL", "ERR mm", "NIS"};
    float x, w, lh = cap_h(s_hud) * 2.0f, y = table_head(r, 6, cx, hd, &x, &w);
    for (int f = NNF; f >= 0; f--) {
        int col = f == NNF ? C_TRUTH : NF_COL[f];
        char b[24];
        text(s_hud, x, y, -1, f == NNF ? "float64" : NF_NAME[f], pal[col], 1);
        if (f == NNF) {
            text(s_hud, x + w * cx[1], y, -1, "ok", pal[C_TEXT], 1);
            fmt_sci(b, sizeof b, NU.eig[NNF][0]); text(s_hud, x + w * cx[2], y, -1, b, pal[C_TEXT], 1);
            text(s_hud, x + w * cx[3], y, -1, "\xe2\x80\x94", pal[C_TEXT], 1);
        } else {
            const char *st = NU.broken[f] ? "DIVERGED" : NU.bad[f] ? "\xe2\x8a\x81 0" : "ok";
            text(s_hud, x + w * cx[1], y, -1, st, pal[NU.broken[f] || NU.bad[f] ? C_WARN : C_TEXT], 1);
            if (!NU.broken[f]) {
                fmt_sci(b, sizeof b, NU.lmin[f]); text(s_hud, x + w * cx[2], y, -1, b, pal[NU.lmin[f] <= 0 ? C_WARN : C_TEXT], 1);
                fmt_sci(b, sizeof b, NU.rel[f]); text(s_hud, x + w * cx[3], y, -1, b, pal[C_TEXT], 1);
            }
        }
        if (f == NNF || !NU.broken[f]) {
            double err = f == NNF ? hypot(NU.xr[0] - NU.xt[0], NU.xr[1] - NU.xt[1]) * 1000 : NU.err[f];
            textf(s_hud, x + w * cx[4], y, -1, pal[C_TEXT], 1, err < 100 ? "%.2f" : "%.0f", err);
            if (f < NNF) textf(s_hud, x + w * cx[5], y, -1, pal[C_TEXT], 1, NU.nis[f] < 1000 ? "%.2f" : "%.0e", NU.nis[f]);
        }
        y += lh;
    }
    y += lh * 0.5f;
    text(s_small, x, y, -1, "float32 keeps ~7 digits; one update here asks for 9", pal[C_DIM], 1);
    y += lh * 0.8f;
    text(s_small, x, y, -1, "in float64 the square-root and Joseph forms agree to 1e-11 (--check)", pal[C_DIM], 1);
}

/* -------- SLAM -------- */
static double sl_e(double s) { return clampd((s - SL.t) / SL_DT, 0, 1); }
/* the state shown at s: a step behind, between the last two posteriors */
static double sl_x(int i, double e) {
    if (i == 2) return SL.x0[2] + wrap_pi(SL.x[2] - SL.x0[2]) * e;
    double a = SL.x0[i];
    if (i >= 3 && a == 0 && SL.P0[i * NSL + i] == 0) a = SL.x[i];   /* added this step */
    return a + (SL.x[i] - a) * e;
}
static double sl_p(int i, int j, double e) {
    double a = SL.P0[i * NSL + j], b = SL.P[i * NSL + j];
    if (SL.P0[i * NSL + i] == 0 && i >= 3) a = b;
    if (SL.P0[j * NSL + j] == 0 && j >= 3) a = b;
    return a + (b - a) * e;
}

static void sl_plot(double s) {
    float lw = fmaxf(U, 1), pts[2 * ELN];
    double e = sl_e(s), sd = fmax(s - SL_DT, 0);
    for (int j = 0; j < SL.L; j++) {
        float x = WX(SL.mx[j]), y = WY(SL.my[j]), h = 4 * U;
        seg(x - h, y, x + h, y, lw, pal[C_TRUTH], 0.8f); seg(x, y - h, x, y + h, lw, pal[C_TRUTH], 0.8f);
    }
    draw_truth(&A.tp[0], 0, sd, 0.28f);
    if (!SL.k) return;
    double rx = sl_x(0, e), ry = sl_x(1, e), rth = sl_x(2, e);
    double o[6];
    path_eval(&A.tp[0], sd, o);
    circle(WX(rx), WY(ry), (float)(SL_RMAX * vk), lw, pal[C_DIM], 0.4f, 1);
    /* rays to what the robot can see */
    for (int j = 0; j < SL.L; j++) {
        int sl = SL.slot[j];
        if (sl < 0) continue;
        double d = hypot(SL.mx[j] - o[0], SL.my[j] - o[1]);
        float a = smooth01((float)((SL_RMAX - d) / 1.5));
        if (a <= 0) continue;
        int i = 3 + 2 * sl;
        seg(WX(rx), WY(ry), WX(sl_x(i, e)), WY(sl_x(i + 1, e)), 1.1f * U, pal[C_MEAS], 0.4f * a);
    }
    /* the robot's estimated path */
    {
        static float P[2 * SLH];
        int n = 0;
        for (int i = 0; i < SL.rn; i++) { P[2 * n] = WX(SL.rx[i][0]); P[2 * n + 1] = WY(SL.rx[i][1]); n++; }
        if (n) { P[2 * n - 2] = WX(rx); P[2 * n - 1] = WY(ry); }
        poly(P, n, 0, 1.4f * U, pal[C_F2], 0.75f, NULL);
    }
    /* landmarks: 2 sigma */
    for (int sl = 0; sl < SL.nl; sl++) {
        int i = 3 + 2 * sl, j = SL.id[sl];
        double x = sl_x(i, e), y = sl_x(i + 1, e), P3[3] = {sl_p(i, i, e), sl_p(i, i + 1, e), sl_p(i + 1, i + 1, e)};
        float a = smooth01((float)((s - SL.first[j]) / A.TS / 0.6));
        cov_ellipse(x, y, P3, 2, pts);
        fan(pts, ELN, WX(x), WY(y), pal[C_F1], 0.12f * a);
        poly(pts, ELN, 1, 1.4f * U, pal[C_F1], 0.9f * a, NULL);
        disk(WX(x), WY(y), 1.8f * U, pal[C_F1], a);
        textf(s_small, WX(x) + 7 * U, WY(y) - 6 * U, -1, pal[C_F1], 0.8f * a, "L%02d", j + 1);
    }
    /* the robot */
    {
        double P3[3] = {sl_p(0, 0, e), sl_p(0, 1, e), sl_p(1, 1, e)};
        cov_ellipse(rx, ry, P3, 2, pts);
        fan(pts, ELN, WX(rx), WY(ry), pal[C_F2], 0.12f);
        poly(pts, ELN, 1, 1.5f * U, pal[C_F2], 0.95f, NULL);
        float ox = WX(rx), oy = WY(ry), hx = cosf((float)rth), hy = -sinf((float)rth), sz = 9 * U;
        V(ox + hx * sz, oy + hy * sz, 0, 0, pal[C_F2], 1, K_FLAT, 0);
        V(ox - hx * sz * 0.6f - hy * sz * 0.5f, oy - hy * sz * 0.6f + hx * sz * 0.5f, 0, 0, pal[C_F2], 1, K_FLAT, 0);
        V(ox - hx * sz * 0.6f + hy * sz * 0.5f, oy - hy * sz * 0.6f - hx * sz * 0.5f, 0, 0, pal[C_F2], 1, K_FLAT, 0);
        float tx = WX(o[0]), ty = WY(o[1]), h = 3.5f * U;
        seg(tx - h, ty - h, tx + h, ty + h, lw, pal[C_TRUTH], 0.9f); seg(tx - h, ty + h, tx + h, ty - h, lw, pal[C_TRUTH], 0.9f);
    }
    for (int c = 0; c < SL.nclose; c++) {
        double age = (s - SL.t_close[c]) / A.TS;
        if (age < 0 || age > 9) continue;
        float a = smooth01((float)(age / 0.6)) * (1 - smooth01((float)((age - 6) / 3)));
        int j = SL.close_lm[c], sl = SL.slot[j];
        if (sl < 0) continue;
        int i = 3 + 2 * sl;
        float x = WX(SL.x[i]), y = WY(SL.x[i + 1]);
        circle(x, y, 18 * U, 1.4f * U, pal[C_F3], 0.8f * a, 0);
        int left = x > vr.x + vr.w * 0.6f;
        text(s_hudb, x + (left ? -22 : 22) * U, y + 26 * U, left ? 1 : -1, "LOOP CLOSURE", pal[C_F3], a);
    }
    float lx, ly;
    plot_head(s, &lx, &ly, 7);
    legend_line(&ly, lx, C_TRUTH, 0, "truth: path, landmarks +");
    legend_line(&ly, lx, C_F2, 0, "robot estimate, 2\xcf\x83");
    legend_line(&ly, lx, C_F1, 0, "landmark estimates, 2\xcf\x83");
    legend_line(&ly, lx, C_MEAS, 0, "range-bearing sightings");
    ly += cap_h(s_small);
    textf(s_small, lx, ly, -1, pal[C_DIM], 1, "%d of %d landmarks mapped; state %d", SL.nl, SL.L, SL.n);
    ly += cap_h(s_small) * 1.9f;
    textf(s_small, lx, ly, -1, pal[C_DIM], 1, "odometry \xcf\x83 %.2f m/s, %.1f\xc2\xb0/s; range %.2f m, bearing %.0f\xc2\xb0", SL_SV, SL_SW / DEGR, SL_SR, SL_SB / DEGR);
}

static void sl_mats(const Rect *r) {
    Rect in = {r->x + 34 * U, r->y + 44 * U, r->w - 44 * U, r->h - 56 * U};
    int n = SL.n;
    if (n < 3) return;
    float cs = fminf(in.h / n, in.w * 0.62f / n);
    double C[NSL * NSL];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i * n + j] = SL.P[i * NSL + j];
    heat(in.x, in.y, cs, n, n, C, 0, NULL, NULL, NULL);
    float lw = fmaxf(U, 1), tot = cs * n;
    seg(in.x + 3 * cs, in.y, in.x + 3 * cs, in.y + tot, 1.4f * U, pal[C_TEXT], 0.8f);
    seg(in.x, in.y + 3 * cs, in.x + tot, in.y + 3 * cs, 1.4f * U, pal[C_TEXT], 0.8f);
    text(s_small, in.x - 5 * U, in.y + 1.5f * cs + cap_h(s_small) / 2, 1, "r", pal[C_DIM], 1);
    text(s_small, in.x + 1.5f * cs, in.y - 4 * U, 0, "r", pal[C_DIM], 1);
    for (int sl = 0; sl < SL.nl; sl += (cs < 8 * U ? 2 : 1)) {
        float c = in.x + (3 + 2 * sl + 1) * cs;
        textf(s_small, c, in.y - 4 * U, 0, pal[C_DIM], 1, "%d", SL.id[sl] + 1);
    }
    (void)lw;
    /* how much of the map moves with the robot */
    float x2 = in.x + tot + 30 * U, y = in.y + cap_h(s_small);
    double mr = 0, mm2 = 0;
    int cr = 0, cm = 0;
    for (int a = 0; a < SL.nl; a++) {
        int i = 3 + 2 * a;
        mr += fabs(SL.P[0 * NSL + i] / sqrt(SL.P[0] * SL.P[i * NSL + i] + 1e-30)); cr++;
        for (int b = a + 1; b < SL.nl; b++) {
            int j = 3 + 2 * b;
            mm2 += fabs(SL.P[i * NSL + j] / sqrt(SL.P[i * NSL + i] * SL.P[j * NSL + j] + 1e-30)); cm++;
        }
    }
    if (x2 < r->x + r->w - 60 * U) {
        textf(s_small, x2, y, -1, pal[C_DIM], 1, "correlation \xcf\x81\xe1\xb5\xa2\xe2\x82\x97");
        y += cap_h(s_small) * 2;
        textf(s_small, x2, y, -1, pal[C_TEXT], 1, "robot\xe2\x80\x93map  |\xcf\x81| %.2f", cr ? mr / cr : 0);
        y += cap_h(s_small) * 1.8f;
        textf(s_small, x2, y, -1, pal[C_TEXT], 1, "map\xe2\x80\x93map    |\xcf\x81| %.2f", cm ? mm2 / cm : 0);
        y += cap_h(s_small) * 1.8f;
        textf(s_small, x2, y, -1, pal[C_DIM], 1, "%d \xc3\x97 %d", n, n);
    }
}

static void sl_meas(const Rect *r, double s) {
    Rect in = {r->x + 30 * U, r->y + 40 * U, r->w - 60 * U, r->h - 64 * U};
    float lw = fmaxf(U, 1), pts[2 * ELN];
    axes_box(&in);
    float sz = fminf(in.w, in.h), cx = in.x + in.w / 2, cy = in.y + in.h * 0.62f, k = sz / 2 / (float)(SL_RMAX * 1.25);
    if (!SL.k) return;
    double e = sl_e(s), rx = sl_x(0, e), ry = sl_x(1, e), th = sl_x(2, e);
    circle(cx, cy, (float)(SL_RMAX * k), lw, pal[C_DIM], 0.6f, 1);
    for (int m = 5; m < SL_RMAX; m += 5) circle(cx, cy, (float)(m * k), lw, pal[C_GRID], 0.8f, 0);
    seg(cx, cy - (float)(SL_RMAX * k), cx, cy + (float)(SL_RMAX * k), lw, pal[C_GRID], 1);
    seg(cx - (float)(SL_RMAX * k), cy, cx + (float)(SL_RMAX * k), cy, lw, pal[C_GRID], 1);
    /* the robot, nose up */
    float hs = 9 * U;
    V(cx, cy - hs, 0, 0, pal[C_F2], 1, K_FLAT, 0); V(cx - hs * 0.55f, cy + hs * 0.6f, 0, 0, pal[C_F2], 1, K_FLAT, 0); V(cx + hs * 0.55f, cy + hs * 0.6f, 0, 0, pal[C_F2], 1, K_FLAT, 0);
    double c = cos(th), sn = sin(th);
    for (int sl = 0; sl < SL.nl; sl++) {
        int i = 3 + 2 * sl;
        double dx = sl_x(i, e) - rx, dy = sl_x(i + 1, e) - ry;
        double fx = c * dx + sn * dy, fy = -sn * dx + c * dy;      /* forward, left */
        if (hypot(fx, fy) > SL_RMAX * 1.25) continue;
        /* the relative position's covariance: J P J' with J = d(rel)/d(robot, landmark) */
        double Jr[2][5] = {{-c, -sn, -sn * dx + c * dy, c, sn}, {sn, -c, -c * dx - sn * dy, -sn, c}};
        int idx[5] = {0, 1, 2, i, i + 1};
        double Pr[3] = {0, 0, 0};
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < 5; b++) {
                double p = sl_p(idx[a], idx[b], e);
                Pr[0] += Jr[0][a] * p * Jr[0][b]; Pr[1] += Jr[0][a] * p * Jr[1][b]; Pr[2] += Jr[1][a] * p * Jr[1][b];
            }
        /* screen: forward up, left to the left */
        double Ps[3] = {Pr[2], -Pr[1], Pr[0]};
        float sx = cx - (float)(fy * k), sy = cy - (float)(fx * k);
        int vis = SL.seen[SL.id[sl]];
        cov_ell_local(sx, sy, Ps, k, 2, pts);
        poly(pts, ELN, 1, 1.4f * U, pal[C_F1], vis ? 0.95f : 0.4f, NULL);
        if (vis) fan(pts, ELN, sx, sy, pal[C_F1], 0.12f);
        textf(s_small, sx + 6 * U, sy - 5 * U, -1, pal[C_F1], vis ? 0.9f : 0.45f, "L%02d", SL.id[sl] + 1);
    }
    float y = in.y + 6 * U + cap_h(s_small);
    legend_line(&y, in.x + 6 * U, C_F1, 0, "landmark relative to the robot, 2\xcf\x83 of J P J\xe1\xb5\x80");
    textf(s_small, in.x + 6 * U, in.y + in.h - 6 * U, -1, pal[C_DIM], 1, "rings 5 m, reach %.0f m; dim: mapped, out of sight", SL_RMAX);
}

static void sl_table(const Rect *r, double s) {
    (void)s;
    static const float cx[] = {0, 0.16f, 0.34f, 0.52f, 0.72f};
    static const char *hd[] = {"LMK", "\xcf\x83 m", "ERROR m", "\xcf\x81 ROBOT x", "FIRST SEEN"};
    float x, w, lh = cap_h(s_hud) * 1.8f, y = table_head(r, 5, cx, hd, &x, &w);
    int maxr = (int)((r->y + r->h - y - lh) / lh);
    for (int sl = 0; sl < SL.nl && sl < maxr; sl++) {
        int i = 3 + 2 * sl, j = SL.id[sl];
        double sg = sqrt(SL.P[i * NSL + i] + SL.P[(i + 1) * NSL + i + 1]);
        textf(s_hud, x, y, -1, pal[SL.seen[j] ? C_F1 : C_DIM], 1, "L%02d", j + 1);
        textf(s_hud, x + w * cx[1], y, -1, pal[C_TEXT], 1, "%.2f", sg);
        textf(s_hud, x + w * cx[2], y, -1, pal[C_TEXT], 1, "%.2f", hypot(SL.x[i] - SL.mx[j], SL.x[i + 1] - SL.my[j]));
        textf(s_hud, x + w * cx[3], y, -1, pal[C_TEXT], 1, "%+.2f", SL.P[i] / sqrt(SL.P[0] * SL.P[i * NSL + i] + 1e-30));
        char tb[16];
        fmt_mmss(tb, sizeof tb, SL.first[j]);
        text(s_hud, x + w * cx[4], y, -1, tb, pal[C_TEXT], 1);
        y += lh;
    }
}

/* -------- dispatch -------- */
static void np_plot(double s) {
    switch (A.id) {
    case A_RTS: rts_plot(s); break;
    case A_FUSE: fu_plot(s); break;
    case A_NUM: nu_plot(s); break;
    default: sl_plot(s); break;
    }
}
static void np_mats(const Rect *r) {
    switch (A.id) {
    case A_RTS: rts_mats(r); break;
    case A_FUSE: fu_mats(r); break;
    case A_NUM: nu_mats(r); break;
    default: sl_mats(r); break;
    }
}
static void np_meas(const Rect *r, double s) {
    switch (A.id) {
    case A_RTS: rts_meas(r, s); break;
    case A_FUSE: {
        Rect in = {r->x + 16 * U, r->y + 36 * U, r->w - 32 * U, r->h - 52 * U};
        float g = 14 * U, w = (in.w - g) / 2, h = fminf(in.h, w);
        Rect a = {in.x, in.y + (in.h - h) / 2, w, h}, b = {in.x + w + g, a.y, w, h};
        if (in.h > in.w * 1.2f) {         /* portrait: stacked */
            h = fminf((in.h - g) / 2, in.w);
            a = (Rect){in.x + (in.w - h) / 2, in.y, h, h}; b = (Rect){a.x, in.y + h + g, h, h};
        }
        fu_zoom(&a, s, 0, 0);
        fu_zoom(&b, s, 1, 0);
        break;
    }
    case A_NUM: nu_meas(r, s); break;
    default: sl_meas(r, s); break;
    }
}
static void np_table(const Rect *r, double s) {
    switch (A.id) {
    case A_RTS: rts_table(r, s); break;
    case A_FUSE: fu_table(r, s); break;
    case A_NUM: nu_table(r, s); break;
    default: sl_table(r, s); break;
    }
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
        case P_TABLE: frame(r, A.id == A_MTT ? "TRACK TABLE" : A.id == A_RTS ? "ESTIMATES" : A.id == A_FUSE ? "SENSORS" : A.id == A_SLAM ? "THE MAP" : "FILTERS"); break;
        case P_CLOCK: if (!(role == R_FRONT && spanned)) frame(r, NULL); break;
        }
    }
    flush();
    st_w = W; st_h = H; st_act = A.id;
}

/* s: sim seconds into the act, rt: real seconds into it, wt: wall clock */
static void render(double s, double rt, double wt, GLuint target) {
    /* a new size (or a full glyph cache): new styles, a new layout */
    if (W != lay_w || H != lay_h || text_full || (st_act != A.id && (nstyles > MAXSTYLE * 3 / 4 || ay > AH * 3 / 4))) {
        text_reset();
        layout();
        lay_w = W; lay_h = H; st_act = -1;
    }
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
            clip_cur = *r;
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
            clip_cur = *r;
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

/* dense inverse of any size (Gauss-Jordan, partial pivoting) */
static int inv_big(const double *A0, double *Ai, int n) {
    double *M = malloc(sizeof(double) * n * 2 * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 2 * n; j++) M[i * 2 * n + j] = j < n ? A0[i * n + j] : (j - n == i);
    int ok = 1;
    for (int c = 0; c < n && ok; c++) {
        int p = c;
        for (int r = c + 1; r < n; r++) if (fabs(M[r * 2 * n + c]) > fabs(M[p * 2 * n + c])) p = r;
        if (fabs(M[p * 2 * n + c]) < 1e-300) { ok = 0; break; }
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
    for (int i = 0; i < n && ok; i++)
        for (int j = 0; j < n; j++) Ai[i * n + j] = M[i * 2 * n + n + j];
    free(M);
    return ok;
}

/* RTS against batch least squares: the smoothed means and covariances are
 * the solution of the normal equations over the whole trajectory, and the
 * fixed-lag estimate is that solution over the data so far */
static void check_rts(void) {
    enum { N = 40 };
    int D = 4 * N;
    double F[16], Q[16], Qi[16], LQ[16], P0[16] = {0}, P0i[16], m0[4] = {0, 0, 8, -3}, Rv = 25;
    model_F(M_CV, NULL, 1.0, F);
    model_Q(M_CV, 1.0, 2.0, 0, Q);
    inv(Q, Qi, 4); chol(Q, LQ, 4);
    P0[0] = P0[5] = 100; P0[10] = P0[15] = 16;
    inv(P0, P0i, 4);
    double wx = 0, wP = 0, wl = 0, wlP = 0;
    double *Lam = malloc(sizeof(double) * D * D), *Li = malloc(sizeof(double) * D * D), *eta = malloc(sizeof(double) * D), *xb = malloc(sizeof(double) * D);
    for (int trial = 0; trial < 6; trial++) {
        Rng r = {4242 + (uint64_t)trial};
        double x[4], tr[N][4], z[N][2];
        int has[N];
        for (int i = 0; i < 4; i++) x[i] = m0[i] + sqrt(P0[i * 5]) * rn_n(&r);
        for (int k = 0; k < N; k++) {
            if (k) {
                double y[4], w[4] = {rn_n(&r), rn_n(&r), rn_n(&r), rn_n(&r)};
                mm(F, x, y, 4, 4, 1);
                for (int i = 0; i < 4; i++) { x[i] = y[i]; for (int j = 0; j <= i; j++) x[i] += LQ[i * 4 + j] * w[j]; }
            }
            memcpy(tr[k], x, sizeof x);
            has[k] = !(k >= 12 && k < 20) && rn_u(&r) < 0.9;
            z[k][0] = x[0] + 5 * rn_n(&r); z[k][1] = x[1] + 5 * rn_n(&r);
        }
        /* forward filter + RTS */
        static double xf[N][4], Pf[N][16], xp[N][4], Pp[N][16], xs[N][4], Ps[N][16], C[N][16];
        int lagk = 30, L = 8;
        double xlag[4], Plag[16];
        for (int k = 0; k < N; k++) {
            Filt f;
            f_init(&f, M_CV, 2.0, 0);
            if (k == 0) { memcpy(f.x, m0, sizeof m0); memcpy(f.P, P0, sizeof P0); f.t = 0; }
            else { memcpy(f.x, xf[k - 1], sizeof xf[0]); memcpy(f.P, Pf[k - 1], sizeof Pf[0]); f.t = k - 1; f_predict(&f, k); }
            memcpy(xp[k], f.x, sizeof xp[0]); memcpy(Pp[k], f.P, sizeof Pp[0]);
            double R[4] = {Rv, 0, 0, Rv};
            if (has[k]) f_update_pos(&f, z[k], R);
            memcpy(xf[k], f.x, sizeof xf[0]); memcpy(Pf[k], f.P, sizeof Pf[0]);
            if (k == lagk) {            /* the fixed-lag smoother's view at step k */
                rts_smooth(k + 1, xf, Pf, xp, Pp, xs, Ps, C, F);
                memcpy(xlag, xs[k - L], sizeof xlag); memcpy(Plag, Ps[k - L], sizeof Plag);
            }
        }
        rts_smooth(N, xf, Pf, xp, Pp, xs, Ps, C, F);
        /* batch: the information matrix of the whole trajectory */
        for (int pass = 0; pass < 2; pass++) {
            int n = pass ? lagk + 1 : N, Dn = 4 * n;
            memset(Lam, 0, sizeof(double) * D * D); memset(eta, 0, sizeof(double) * D);
            #define LA(i, j) Lam[(i) * Dn + (j)]
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) LA(i, j) += P0i[i * 4 + j];
                for (int j = 0; j < 4; j++) eta[i] += P0i[i * 4 + j] * m0[j];
            }
            for (int k = 1; k < n; k++) {
                /* (x_k - F x_{k-1})' Qi (x_k - F x_{k-1}) */
                double FtQi[16], FtQiF[16];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
                    double s = 0; for (int l = 0; l < 4; l++) s += F[l * 4 + i] * Qi[l * 4 + j];
                    FtQi[i * 4 + j] = s; }
                mm(FtQi, F, FtQiF, 4, 4, 4);
                int a = 4 * (k - 1), b = 4 * k;
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
                    LA(a + i, a + j) += FtQiF[i * 4 + j];
                    LA(b + i, b + j) += Qi[i * 4 + j];
                    LA(a + i, b + j) -= FtQi[i * 4 + j];
                    LA(b + j, a + i) -= FtQi[i * 4 + j];
                }
            }
            for (int k = 0; k < n; k++)
                if (has[k])
                    for (int i = 0; i < 2; i++) { LA(4 * k + i, 4 * k + i) += 1 / Rv; eta[4 * k + i] += z[k][i] / Rv; }
            #undef LA
            inv_big(Lam, Li, Dn);
            for (int i = 0; i < Dn; i++) { double s = 0; for (int j = 0; j < Dn; j++) s += Li[i * Dn + j] * eta[j]; xb[i] = s; }
            if (pass == 0) {
                for (int k = 0; k < N; k++)
                    for (int i = 0; i < 4; i++) {
                        wx = fmax(wx, fabs(xs[k][i] - xb[4 * k + i]));
                        for (int j = 0; j < 4; j++)
                            wP = fmax(wP, fabs(Ps[k][i * 4 + j] - Li[(4 * k + i) * Dn + 4 * k + j]) / sqrt(Ps[k][i * 5] * Ps[k][j * 5]));
                    }
            } else {
                int k = lagk - L;
                for (int i = 0; i < 4; i++) {
                    wl = fmax(wl, fabs(xlag[i] - xb[4 * k + i]));
                    for (int j = 0; j < 4; j++)
                        wlP = fmax(wlP, fabs(Plag[i * 4 + j] - Li[(4 * k + i) * Dn + 4 * k + j]) / sqrt(Plag[i * 5] * Plag[j * 5]));
                }
            }
        }
    }
    free(Lam); free(Li); free(eta); free(xb);
    printf("  RTS vs batch least squares (6 runs x %d steps, CV, GPS with an outage):\n", N);
    printf("    smoothed mean max |x_RTS - x_batch| = %.2e m;  covariance max |dP|/sqrt(PiiPjj) = %.2e  %s\n", wx, wP,
           wx < 1e-6 && wP < 1e-6 ? "OK" : "MISMATCH");
    printf("    fixed lag 8 at step 30 vs batch on steps 0..30: |dx| = %.2e m, |dP| = %.2e  %s\n", wl, wlP,
           wl < 1e-6 && wlP < 1e-6 ? "OK" : "MISMATCH");
}

/* the information filter against the covariance filter, over the fusion act */
static void check_fusion(void) {
    double wsum = 0, wx = 0, wP = 0, nf = 0, nn = 0, nc = 0, rf = 0, rdr = 0, rg = 0, nees_f = 0;
    int cnt = 0, nupd = 0, nne = 0, runs = 24;
    double rmin = 1e9, rmax = 0;
    for (uint32_t sd = 1; sd <= (uint32_t)runs; sd++) {
        opt_seed = sd;
        act_setup(0, A_FUSE);
        for (long i = 0; (i + 1) * FU_DT <= A.len_s; i++) {
            fu_tick();
            double mp = 0;
            for (int k = 0; k < FN * FN; k++) mp = fmax(mp, fabs(FU.kf.P[k]));
            for (int k = 0; k < FN; k++) wx = fmax(wx, fabs(A.f[0].x[k] - FU.kf.x[k]));
            for (int k = 0; k < FN * FN; k++) wP = fmax(wP, fabs(A.f[0].P[k] - FU.kf.P[k]) / mp);
        }
        rmin = fmin(rmin, FU.nees_a / FU.nci); rmax = fmax(rmax, FU.nees_a / FU.nci);
        wsum += FU.wsum; nf += FU.nees_a; nn += FU.nees_n; nc += FU.nees_c; cnt += FU.nci;
        rf += FU.se_f / FU.nse; rdr += FU.se_dr / FU.nse; rg += FU.nseg ? FU.se_g / FU.nseg : 0;
        nees_f += A.acc_nees[0]; nne += A.acc_ne[0];
        for (int s = 0; s < NSN; s++) nupd += FU.nupd[s];
    }
    printf("  information filter vs covariance (Joseph) filter, same data, %d fusion acts (%d updates):\n", runs, nupd);
    printf("    max |x_IF - x_KF| = %.2e m;  max |P_IF - P_KF| / max|P| = %.2e  %s\n", wx, wP, wx < 1e-6 && wP < 1e-6 ? "OK" : "MISMATCH");
    printf("    RMS position error: GPS %.2f m, dead reckoning %.2f m, fused %.2f m\n", sqrt(rg / runs), sqrt(rdr / runs), sqrt(rf / runs));
    double lo = chi2inv(0.025, 2 * cnt) / cnt, hi = chi2inv(0.975, 2 * cnt) / cnt;
    printf("    position NEES (dof 2, 95%% band of the mean [%.2f, %.2f]): fused %.2f; fused with the base's track\n"
           "      naively (as if independent) %.2f, by covariance intersection %.2f  (%d fusions, mean \xcf\x89 %.2f;\n      per-run fused NEES %.2f to %.2f)\n", lo, hi, nf / cnt, nn / cnt, nc / cnt, cnt, wsum / cnt, rmin, rmax);
    /* CI is consistent whatever the cross-correlation: P_CI - (true error covariance) >= 0 */
    Rng r = {99};
    int bad_ci = 0, bad_naive = 0, M = 20000;
    double worst = 1e9;
    for (int t = 0; t < M; t++) {
        double G[16], J[16];
        for (int i = 0; i < 16; i++) G[i] = rn_n(&r) * (i < 8 ? 1 : 2);
        mmT(G, G, J, 4, 4, 4);                  /* a joint covariance of the two errors */
        double A2[4] = {J[0], J[1], J[4], J[5]}, B2[4] = {J[10], J[11], J[14], J[15]}, X[4] = {J[2], J[3], J[6], J[7]};
        double Ai[4], Bi[4], xn[2], Pn[4], xc[2], Pc[4], zero[2] = {0, 0};
        inv(A2, Ai, 2); inv(B2, Bi, 2);
        double w = ci_omega(Ai, Bi);
        ci_fuse(zero, A2, zero, B2, w, xn, Pn, xc, Pc);
        for (int pass = 0; pass < 2; pass++) {
            double wa = pass ? w : 1, wb = pass ? 1 - w : 1, *P = pass ? Pc : Pn, Wa[4], Wb[4], T1[4], T2[4], E[4] = {0};
            double sA[4], sB[4];
            for (int i = 0; i < 4; i++) { sA[i] = wa * Ai[i]; sB[i] = wb * Bi[i]; }
            mm(P, sA, Wa, 2, 2, 2); mm(P, sB, Wb, 2, 2, 2);
            /* true covariance of Wa ea + Wb eb */
            mm(Wa, A2, T1, 2, 2, 2); mmT(T1, Wa, T2, 2, 2, 2); for (int i = 0; i < 4; i++) E[i] += T2[i];
            mm(Wb, B2, T1, 2, 2, 2); mmT(T1, Wb, T2, 2, 2, 2); for (int i = 0; i < 4; i++) E[i] += T2[i];
            mm(Wa, X, T1, 2, 2, 2); mmT(T1, Wb, T2, 2, 2, 2);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) E[i * 2 + j] += T2[i * 2 + j] + T2[j * 2 + i];
            double D2[4] = {P[0] - E[0], P[1] - E[1], P[2] - E[2], P[3] - E[3]};
            double tr = (D2[0] + D2[3]) / 2, dd = sqrt(fmax((D2[0] - D2[3]) * (D2[0] - D2[3]) / 4 + D2[1] * D2[1], 0));
            double lmin = (tr - dd) / (P[0] + P[3]);
            if (pass) { if (lmin < -1e-9) bad_ci++; worst = fmin(worst, lmin); } else if (lmin < -1e-9) bad_naive++;
        }
    }
    printf("    random joint covariances (%d): P_CI - E[ee'] indefinite in %d (worst min eigenvalue %.1e of trace)  %s;\n"
           "      the naive P is too small in %d (%.0f%%)\n", M, bad_ci, worst, bad_ci ? "FAIL" : "OK", bad_naive, 100.0 * bad_naive / M);
}

/* the square-root filter and the Joseph form agree in double precision;
 * in float32 the naive form does not */
static void check_sqrt(void) {
    double F[16], Q[16], Lq[16];
    nu_FQ(F, Q, Lq);
    double worst = 0, wx = 0;
    Rng r = {31337};
    for (int trial = 0; trial < 20; trial++) {
        double Aq[16], P[16], S[16], x1[4] = {0}, x2[4] = {0}, xt[4] = {0};
        static const double sc[4] = {30, 30, 2, 2};
        for (int i = 0; i < 16; i++) Aq[i] = sc[i / 4] * rn_n(&r);
        mmT(Aq, Aq, P, 4, 4, 4);
        chol(P, S, 4);
        for (int k = 0; k < 200; k++) {
            double z[2] = {xt[0] + NU_SIG * rn_n(&r), xt[1] + NU_SIG * rn_n(&r)}, n1, n2;
            predx_d(x1, F); predP_d(P, F, Q); upd_joseph_d(x1, P, z, NU_SIG * NU_SIG, &n1);
            predx_d(x2, F); predS_d(S, F, Lq); upd_sqrt_d(x2, S, z, NU_SIG * NU_SIG, &n2);
            double SS[16], mp = 0, d = 0;
            mmT(S, S, SS, 4, 4, 4);
            for (int i = 0; i < 16; i++) { mp = fmax(mp, fabs(P[i])); d = fmax(d, fabs(SS[i] - P[i])); }
            worst = fmax(worst, d / mp);
            for (int i = 0; i < 4; i++) wx = fmax(wx, fabs(x1[i] - x2[i]));
        }
    }
    /* and the naive form in float64: does it hold where float32 failed? */
    int nlost = 0;
    {
        Rng r2 = {31337};
        for (int trial = 0; trial < 20; trial++) {
            double Aq[16], P[16], x1[4] = {0}, xt[4] = {0};
            static const double sc[4] = {30, 30, 2, 2};
            for (int i = 0; i < 16; i++) Aq[i] = sc[i / 4] * rn_n(&r2);
            mmT(Aq, Aq, P, 4, 4, 4);
            int lost = 0;
            for (int k = 0; k < 200; k++) {
                double z[2] = {xt[0] + NU_SIG * rn_n(&r2), xt[1] + NU_SIG * rn_n(&r2)}, n1, Ps[16], ev[4];
                predx_d(x1, F); predP_d(P, F, Q);
                if (!upd_naive_d(x1, P, z, NU_SIG * NU_SIG, &n1)) { lost = 1; break; }
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) Ps[i * 4 + j] = 0.5 * (P[i * 4 + j] + P[j * 4 + i]);
                eig4(Ps, ev);
                lost |= ev[0] <= 0;
            }
            nlost += lost;
        }
    }
    printf("  square root (array form) vs Joseph, both float64, 20 cues x 200 laser updates:\n");
    printf("    max |SS' - P_Joseph| / max|P| = %.2e;  max |x_sqrt - x_Joseph| = %.2e m  %s\n", worst, wx,
           worst < 1e-6 && wx < 1e-9 ? "OK" : "MISMATCH");
    printf("    the naive form in float64 lost positive definiteness after %d of 20 cues\n", nlost);
    /* the act, in float32 */
    int cues = 0, lost[NNF] = {0}, broken[NNF] = {0};
    double maxrel[NNF] = {0}, se[NNF + 1] = {0};
    long nse = 0;
    for (uint32_t sd = 1; sd <= 8; sd++) {
        opt_seed = sd;
        act_setup(0, A_NUM);
        int cyc = -1, was[NNF] = {0}, wasb[NNF] = {0};
        for (long i = 0; (i + 1) * NU_DT <= A.len_s; i++) {
            nu_step();
            if (NU.cyc != cyc) { cyc = NU.cyc; cues++; memset(was, 0, sizeof was); memset(wasb, 0, sizeof wasb); }
            for (int f = 0; f < NNF; f++) {
                if ((NU.bad[f] || NU.broken[f]) && !was[f]) { lost[f]++; was[f] = 1; }
                if (NU.broken[f] && !wasb[f]) { broken[f]++; wasb[f] = 1; }
                if (!NU.broken[f]) maxrel[f] = fmax(maxrel[f], NU.rel[f]);
            }
        }
        for (int f = 0; f <= NNF; f++) se[f] += NU.se[f];
        nse += NU.nse;
    }
    printf("  float32 over 8 acts (%d cues, \xcf\x83 30 m -> 1 mm):\n", cues);
    for (int f = 0; f < NNF; f++)
        printf("    %-7s P lost positive definiteness after %2d of %d cues (diverged, S not positive: %d); worst |P - P_ref|/|P_ref| %.1e;\n"
               "            RMS error %.3g mm once settled\n", NF_NAME[f], lost[f], cues, broken[f], maxrel[f], sqrt(se[f] / nse));
    printf("    float64 reference RMS error %.3g mm (laser \xcf\x83 %.1f mm)\n", sqrt(se[NNF] / nse), NU_SIG * 1000);
}

/* EKF-SLAM: robot pose NEES over time and the map's NEES at the end */
static void check_slam(void) {
    double nr = 0, nm = 0, nis = 0;
    int cr = 0, cm = 0, cn = 0, closes = 0, runs = 16;
    for (uint32_t sd = 1; sd <= (uint32_t)runs; sd++) {
        opt_seed = sd;
        act_setup(0, A_SLAM);
        act_advance(A.len_s);
        nr += SL.nees_r; cr += SL.nnr;
        int c;
        nm += sl_map_nees(&c); cm += c;
        nis += A.acc_nis[0]; cn += A.acc_nn[0];
        closes += SL.nclose > 0;
    }
    printf("  EKF-SLAM, %d runs (one loop and most of a second, %d landmarks seen in all):\n", runs, cm);
    printf("    mean NIS %.2f (dof 2, %d sightings, band [%.2f, %.2f])\n", nis / cn, cn, chi2inv(0.025, 2 * cn) / cn, chi2inv(0.975, 2 * cn) / cn);
    printf("    robot pose NEES %.2f (dof 3, %d steps; time-correlated, so the band is loose)\n", nr / cr, cr);
    printf("    map NEES at the end %.2f per landmark (dof 2, band of the mean [%.2f, %.2f]); loop closed in %d of %d runs\n",
           nm / cm, chi2inv(0.025, 2 * cm) / cm, chi2inv(0.975, 2 * cm) / cm, closes, runs);
}

static void run_check(void) {
    printf("kalman --check\n");
    printf("chi2 quantiles: chi2(2) 95%% [%.4f, %.3f] (exact 0.0506, 7.378); chi2(4) [%.3f, %.2f] (exact 0.484, 11.14)\n",
           chi2inv(0.025, 2), chi2inv(0.975, 2), chi2inv(0.025, 4), chi2inv(0.975, 4));
    printf("1. linear filters against their own model\n");
    check_linear(M_CV, 1.5);
    check_linear(M_CA, 1.0);
    printf("2. the acts as shown (manoeuvring truth the filters don't model; time-averaged, 8 seeds each)\n");
    static const char *dofs[NACT] = {"2", "2", "2", "1", "2", "2", "1-2", "2", "2"};
    for (int id = 0; id < A_RTS; id++) {
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
    printf("3. the new acts\n");
    {
        double sf[3] = {0}, nf[3] = {0};
        int cf[3] = {0};
        for (uint32_t sd = 1; sd <= 8; sd++) {
            opt_seed = sd;
            act_setup(0, A_RTS);
            act_advance(A.len_s);
            for (int k = 0; k < 3; k++) { sf[k] += RT.se[k]; nf[k] += RT.sn[k]; cf[k] += RT.ne[k]; }
        }
        static const char *nm[3] = {"filter", "fixed lag 8 s", "smoothed"};
        printf("  RTS act, 8 runs: RMS position error and mean position NEES (dof 2):\n");
        for (int k = 0; k < 3; k++) printf("    %-14s %6.2f m   NEES %5.2f  (%d steps)\n", nm[k], sqrt(sf[k] / cf[k]), nf[k] / cf[k], cf[k]);
    }
    check_rts();
    check_fusion();
    check_sqrt();
    check_slam();
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
        "              [--act 1-9] [--act-len SECONDS] [--font FAMILY] [--math-font FAMILY]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check] [--resize-test N]\n");
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
        else if (!strcmp(a, "-resize-test") && i + 1 < argc) opt_resize = atoi(argv[++i]);
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
    int offscreen = opt_shot || opt_bench > 0 || opt_resize > 0;
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
        if (opt_resize > 0) {          /* many sizes in one process, every act */
            int maxst = 0, r0 = text_resets;
            Rng rr = {opt_seed};
            for (int i = 0; i < opt_resize; i++) {
                W = 200 + (int)(rn_u(&rr) * (off_w - 200)); H = 160 + (int)(rn_u(&rr) * (off_h - 160));
                if (i % 7 == 6) { W = off_w; H = off_h; }
                double t = opt_at + i * 37.0;
                sim_to(t);
                render(A.s, A.s / A.TS, t_wall0 + t, fbo);
                if (nstyles > maxst) maxst = nstyles;
            }
            glFinish();
            printf("kalman: %d resizes: %d text resets, at most %d of %d styles, atlas rows used %d of %d, %s\n",
                   opt_resize, text_resets - r0, maxst, MAXSTYLE, ay + ashelf, AH, text_full ? "TABLE FULL" : "ok");
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
