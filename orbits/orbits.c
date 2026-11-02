/*
 * orbits — a mission-control trajectory display, running real astrodynamics.
 *
 * (description filled in below, with the display)
 */
#define _GNU_SOURCE
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

/* ================================================================== */
/* ---- astrodynamics ----                                             */
/* ================================================================== */
#define PI_D 3.14159265358979323846
#define TAU_D (2 * PI_D)
#define RAD (PI_D / 180)
#define MU_SUN 1.32712440018e11        /* km^3/s^2 */
#define MU_EARTH 398600.4418
#define MU_MARS 42828.37
#define MU_JUP 1.26686534e8
#define R_EARTH 6378.137               /* km */
#define R_MARS 3389.5
#define R_JUP 71492.0
#define R_MOON 1737.4
#define AU 1.495978707e8               /* km */
#define DAY 86400.0
#define J2000 2451545.0
#define R_GEO 42164.17

static double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static double norm3(const double *a) { return sqrt(dot3(a, a)); }
static void cross3(const double *a, const double *b, double *o) {
    double x = a[1] * b[2] - a[2] * b[1], y = a[2] * b[0] - a[0] * b[2], z = a[0] * b[1] - a[1] * b[0];
    o[0] = x; o[1] = y; o[2] = z;
}
static void set3(double *o, double x, double y, double z) { o[0] = x; o[1] = y; o[2] = z; }
static void sub3(const double *a, const double *b, double *o) { o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2]; }
static void add3(const double *a, const double *b, double *o) { o[0] = a[0] + b[0]; o[1] = a[1] + b[1]; o[2] = a[2] + b[2]; }
static void scale3(const double *a, double s, double *o) { o[0] = a[0] * s; o[1] = a[1] * s; o[2] = a[2] * s; }
static void unit3(const double *a, double *o) { double n = norm3(a); scale3(a, n > 0 ? 1 / n : 0, o); }
static double wrap_pi(double a) { a = fmod(a + PI_D, TAU_D); return a < 0 ? a + PI_D : a - PI_D; }

/* Stumpff functions C(z), S(z) */
static double stumpC(double z) {
    if (z > 1e-6) return (1 - cos(sqrt(z))) / z;
    if (z < -1e-6) return (cosh(sqrt(-z)) - 1) / -z;
    return 0.5 - z / 24 + z * z / 720;
}
static double stumpS(double z) {
    if (z > 1e-6) { double s = sqrt(z); return (s - sin(s)) / (s * s * s); }
    if (z < -1e-6) { double s = sqrt(-z); return (sinh(s) - s) / (s * s * s); }
    return 1.0 / 6 - z / 120 + z * z / 5040;
}

/* two-body propagation of (r0, v0) by dt with the universal variable
 * (any conic): Newton on Kepler's equation in chi, then f and g */
static void kepler_uv(double mu, const double *r0, const double *v0, double dt, double *r, double *v) {
    double R0 = norm3(r0), V0 = norm3(v0), vr0 = dot3(r0, v0) / R0, smu = sqrt(mu);
    double alpha = 2 / R0 - V0 * V0 / mu;
    /* F(chi) rises monotonically: Newton inside an expanding bracket,
     * falling back to bisection, so hyperbolas far out converge too */
    double C = 0.5, S = 1.0 / 6, sg = dt < 0 ? -1 : 1;
#define UVF(x_, F_, dF_) do { double z_ = alpha * (x_) * (x_); C = stumpC(z_); S = stumpS(z_); \
        F_ = R0 * vr0 / smu * (x_) * (x_) * C + (1 - alpha * R0) * (x_) * (x_) * (x_) * S + R0 * (x_) - smu * dt; \
        dF_ = R0 * vr0 / smu * (x_) * (1 - z_ * S) + (1 - alpha * R0) * (x_) * (x_) * C + R0; } while (0)
    double lo = 0, hi = sg * smu * fabs(dt) / R0, F, dF;
    if (alpha > 1e-12) hi = sg * fmin(fabs(hi), smu * alpha * fabs(dt) * 1.5 + 1e-9);
    for (int it = 0; it < 200; it++) { UVF(hi, F, dF); if (F * sg >= 0) break; lo = hi; hi *= 2; }
    if (sg < 0) { double t = lo; lo = hi; hi = t; }
    double x = 0.5 * (lo + hi);
    for (int it = 0; it < 200; it++) {
        UVF(x, F, dF);
        if (F < 0) lo = x; else hi = x;
        double xn = x - F / dF;
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        if (fabs(xn - x) < 1e-13 * (1 + fabs(x))) { x = xn; break; }
        x = xn;
    }
#undef UVF
    double z = alpha * x * x;
    C = stumpC(z); S = stumpS(z);
    double f = 1 - x * x / R0 * C, g = dt - x * x * x / smu * S;
    double rr[3] = {f * r0[0] + g * v0[0], f * r0[1] + g * v0[1], f * r0[2] + g * v0[2]};
    double R = norm3(rr);
    double fd = smu / (R * R0) * (z * x * S - x), gd = 1 - x * x / R * C;
    for (int k = 0; k < 3; k++) { v[k] = fd * r0[k] + gd * v0[k]; r[k] = rr[k]; }
}

/* ---- conics by their elements; a < 0 and e > 1 for a hyperbola ---- */
typedef struct { double mu, a, e, i, raan, argp, M0, t0; } Orbit;

static void orb_basis(const Orbit *o, double *P, double *Q) {
    double cO = cos(o->raan), sO = sin(o->raan), cw = cos(o->argp), sw = sin(o->argp), ci = cos(o->i), si = sin(o->i);
    set3(P, cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si);
    set3(Q, -cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si);
}
static double orb_p(const Orbit *o) { return o->a * (1 - o->e * o->e); }
static double orb_n(const Orbit *o) { return sqrt(o->mu / fabs(o->a * o->a * o->a)); }
static double orb_period(const Orbit *o) { return TAU_D / orb_n(o); }

/* position and velocity at true anomaly nu */
static void orb_at_nu(const Orbit *o, double nu, double *r, double *v) {
    double P[3], Q[3], p = orb_p(o);
    orb_basis(o, P, Q);
    double R = p / (1 + o->e * cos(nu)), c = cos(nu), s = sin(nu), k = sqrt(o->mu / p);
    for (int j = 0; j < 3; j++) {
        if (r) r[j] = R * (c * P[j] + s * Q[j]);
        if (v) v[j] = k * (-s * P[j] + (o->e + c) * Q[j]);
    }
}

/* Kepler's equation: M = E - e sin E, and its hyperbolic twin M = e sinh H - H */
static double kep_E(double M, double e) {
    M = wrap_pi(M);
    double E = e < 0.8 ? M + e * sin(M) : (M < 0 ? -PI_D : PI_D) * 0.9;
    for (int i = 0; i < 50; i++) {
        double d = (E - e * sin(E) - M) / (1 - e * cos(E));
        E -= d;
        if (fabs(d) < 1e-14) break;
    }
    return E;
}
static double kep_H(double M, double e) {
    double H = asinh(M / e);
    for (int i = 0; i < 80; i++) {
        double d = (e * sinh(H) - H - M) / (e * cosh(H) - 1);
        H -= d;
        if (fabs(d) < 1e-14 * (1 + fabs(H))) break;
    }
    return H;
}
static double nu_of_M(double M, double e) {
    if (e < 1) { double E = kep_E(M, e); return 2 * atan2(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2)); }
    double H = kep_H(M, e);
    return 2 * atan(sqrt((e + 1) / (e - 1)) * tanh(H / 2));
}
static double M_of_nu(double nu, double e) {
    if (e < 1) { double E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2)); return E - e * sin(E); }
    double H = 2 * atanh(sqrt((e - 1) / (e + 1)) * tan(nu / 2));
    return e * sinh(H) - H;
}
static double orb_nu_at(const Orbit *o, double t) { return nu_of_M(o->M0 + orb_n(o) * (t - o->t0), o->e); }
static void orb_rv(const Orbit *o, double t, double *r, double *v) { orb_at_nu(o, orb_nu_at(o, t), r, v); }
/* the time (after t0's periapsis passage reference) at which the conic is at nu, nearest t0 */
static double orb_t_of_nu(const Orbit *o, double nu) { return o->t0 + (M_of_nu(nu, o->e) - (o->e < 1 ? wrap_pi(o->M0) : o->M0)) / orb_n(o); }

/* elements from a state vector at time t */
static void orb_from_rv(double mu, const double *r, const double *v, double t, Orbit *o) {
    double h[3], ev[3], Wv[3], Pv[3], Qv[3], nv[3];
    cross3(r, v, h);
    double R = norm3(r), V2 = dot3(v, v), rv = dot3(r, v);
    for (int k = 0; k < 3; k++) ev[k] = ((V2 - mu / R) * r[k] - rv * v[k]) / mu;
    double e = norm3(ev);
    unit3(h, Wv);
    double si = sqrt(Wv[0] * Wv[0] + Wv[1] * Wv[1]);
    o->mu = mu;
    o->i = atan2(si, Wv[2]);
    o->raan = si > 1e-11 ? atan2(Wv[0], -Wv[1]) : 0;
    set3(nv, cos(o->raan), sin(o->raan), 0);
    if (e > 1e-10) scale3(ev, 1 / e, Pv);
    else if (si > 1e-11) memcpy(Pv, nv, sizeof Pv);
    else { set3(Pv, 1, 0, 0); if (Wv[2] < 0) set3(Pv, 1, 0, 0); }
    cross3(Wv, Pv, Qv);
    double wn[3];
    cross3(Wv, nv, wn);
    o->argp = atan2(dot3(Pv, wn), dot3(Pv, nv));
    o->e = e;
    o->a = 1 / (2 / R - V2 / mu);
    double nu = atan2(dot3(r, Qv), dot3(r, Pv));
    o->M0 = M_of_nu(nu, e);
    o->t0 = t;
}

/* ---- Lambert's problem: the conic from r1 to r2 in time dt (single
 * revolution, universal variables, after Bate/Mueller/White and Curtis).
 * Time of flight rises monotonically with z, so z is bracketed and bisected
 * (robust everywhere a porkchop wanders, ridges included). 0 on failure. */
static int lambert(double mu, const double *r1, const double *r2, double dt, int prograde, double *v1, double *v2) {
    double R1 = norm3(r1), R2 = norm3(r2), c[3];
    cross3(r1, r2, c);
    double cth = dot3(r1, r2) / (R1 * R2);
    if (cth > 1) cth = 1;
    if (cth < -1) cth = -1;
    double th = acos(cth);
    if ((prograde && c[2] < 0) || (!prograde && c[2] >= 0)) th = TAU_D - th;
    if (1 - cos(th) < 1e-12) return 0;
    double A = sin(th) * sqrt(R1 * R2 / (1 - cos(th)));
    if (fabs(A) < 1e-9) return 0;
    double smu = sqrt(mu), lo = -4 * PI_D * PI_D * 4, hi = 4 * PI_D * PI_D - 1e-9, y = 0, z = 0;
    for (int it = 0; it < 100; it++) {
        z = 0.5 * (lo + hi);
        double C = stumpC(z), S = stumpS(z);
        y = R1 + R2 + A * (z * S - 1) / sqrt(C);
        if (y < 0) { lo = z; continue; }
        double F = pow(y / C, 1.5) * S + A * sqrt(y) - smu * dt;
        if (F < 0) lo = z; else hi = z;
        if (hi - lo < 1e-12) break;
    }
    if (y <= 0) return 0;
    double f = 1 - y / R1, g = A * sqrt(y / mu), gd = 1 - y / R2;
    for (int k = 0; k < 3; k++) {
        v1[k] = (r2[k] - f * r1[k]) / g;
        v2[k] = (gd * r2[k] - r1[k]) / g;
    }
    return isfinite(v1[0]) && isfinite(v2[0]);
}

/* ---- the planets: J2000 Keplerian elements and their rates per Julian
 * century (E. M. Standish, "Keplerian Elements for Approximate Positions of
 * the Major Planets", JPL, table 1, 1800-2050): a (au), e, I, L, long. of
 * perihelion, long. of the node (degrees). Heliocentric ecliptic J2000. */
enum { P_MERCURY, P_VENUS, P_EARTH, P_MARS, P_JUPITER, P_SATURN, NPL };
static const double ELEM[NPL][12] = {
    {0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
     252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081},
    {0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
     181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418},
    {1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
     100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0},
    {1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
     -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343},
    {5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
     34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106},
    {9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
     49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794},
};
static double jd_of(double unix_t) { return unix_t / 86400.0 + 2440587.5; }

/* the osculating heliocentric orbit of planet p at Julian date jd (time
 * reference: seconds past J2000) */
static void planet_orbit(int p, double jd, Orbit *o) {
    const double *E = ELEM[p];
    double T = (jd - J2000) / 36525.0;
    double L = E[6] + E[7] * T, wb = E[8] + E[9] * T, Om = E[10] + E[11] * T;
    o->mu = MU_SUN;
    o->a = (E[0] + E[1] * T) * AU;
    o->e = E[2] + E[3] * T;
    o->i = (E[4] + E[5] * T) * RAD;
    o->raan = Om * RAD;
    o->argp = (wb - Om) * RAD;
    o->M0 = wrap_pi((L - wb) * RAD);
    o->t0 = (jd - J2000) * DAY;
}
static void planet_rv(int p, double jd, double *r, double *v) {
    Orbit o;
    planet_orbit(p, jd, &o);
    orb_at_nu(&o, nu_of_M(o.M0, o.e), r, v);
}

/* sidereal time at Greenwich (radians) and the Sun's direction in
 * equatorial coordinates, for the globe's spin and its terminator */
static double gmst(double jd) { return fmod((280.46061837 + 360.98564736629 * (jd - J2000)) * RAD, TAU_D); }
static void sun_dir_eq(double jd, double *s) {
    double d = jd - J2000, g = (357.529 + 0.98560028 * d) * RAD;
    double lam = (280.459 + 0.98564736 * d + 1.915 * sin(g) + 0.020 * sin(2 * g)) * RAD, eps = 23.439 * RAD;
    set3(s, cos(lam), cos(eps) * sin(lam), sin(eps) * sin(lam));
}

/* ---- the circular restricted three-body problem, planar, in the frame
 * rotating with the primaries: lengths in their separation, time in
 * 1/(mean motion), masses 1-mu at (-mu, 0) and mu at (1-mu, 0) ---- */
static double cr_mu = 0.012150585;     /* Earth-Moon */
static void cr_f(const double *s, double *d) {
    double x = s[0], y = s[1], m = cr_mu;
    double r1 = sqrt((x + m) * (x + m) + y * y), r2 = sqrt((x - 1 + m) * (x - 1 + m) + y * y);
    double k1 = (1 - m) / (r1 * r1 * r1), k2 = m / (r2 * r2 * r2);
    d[0] = s[2]; d[1] = s[3];
    d[2] = 2 * s[3] + x - k1 * (x + m) - k2 * (x - 1 + m);
    d[3] = -2 * s[2] + y - k1 * y - k2 * y;
}
static double cr_2omega(double x, double y) {
    double m = cr_mu, r1 = sqrt((x + m) * (x + m) + y * y), r2 = sqrt((x - 1 + m) * (x - 1 + m) + y * y);
    return x * x + y * y + 2 * (1 - m) / r1 + 2 * m / r2;
}
static double cr_jacobi(const double *s) { return cr_2omega(s[0], s[1]) - s[2] * s[2] - s[3] * s[3]; }

/* one adaptive Dormand-Prince 5(4) step; h is updated. Returns 1 if the
 * step was taken (s, t advanced; k7 = f at the new state) */
static int dp45(double *s, double *t, double *h, double tol, double *fs) {
    static const double c2 = 1. / 5, c3 = 3. / 10, c4 = 4. / 5, c5 = 8. / 9;
    static const double a21 = 1. / 5, a31 = 3. / 40, a32 = 9. / 40, a41 = 44. / 45, a42 = -56. / 15, a43 = 32. / 9,
                        a51 = 19372. / 6561, a52 = -25360. / 2187, a53 = 64448. / 6561, a54 = -212. / 729,
                        a61 = 9017. / 3168, a62 = -355. / 33, a63 = 46732. / 5247, a64 = 49. / 176, a65 = -5103. / 18656,
                        b1 = 35. / 384, b3 = 500. / 1113, b4 = 125. / 192, b5 = -2187. / 6784, b6 = 11. / 84,
                        e1 = 71. / 57600, e3 = -71. / 16695, e4 = 71. / 1920, e5 = -17253. / 339200, e6 = 22. / 525, e7 = -1. / 40;
    (void)c2; (void)c3; (void)c4; (void)c5;
    double k1[4], k2[4], k3[4], k4[4], k5[4], k6[4], k7[4], y[4], yn[4];
    double hh = *h;
    memcpy(k1, fs, sizeof k1);
    for (int i = 0; i < 4; i++) y[i] = s[i] + hh * a21 * k1[i];
    cr_f(y, k2);
    for (int i = 0; i < 4; i++) y[i] = s[i] + hh * (a31 * k1[i] + a32 * k2[i]);
    cr_f(y, k3);
    for (int i = 0; i < 4; i++) y[i] = s[i] + hh * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    cr_f(y, k4);
    for (int i = 0; i < 4; i++) y[i] = s[i] + hh * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    cr_f(y, k5);
    for (int i = 0; i < 4; i++) y[i] = s[i] + hh * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    cr_f(y, k6);
    for (int i = 0; i < 4; i++) yn[i] = s[i] + hh * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
    cr_f(yn, k7);
    double err = 0;
    for (int i = 0; i < 4; i++) {
        double ei = hh * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        double sc = tol * (1 + fmax(fabs(s[i]), fabs(yn[i])));
        err = fmax(err, fabs(ei) / sc);
    }
    double fac = err > 0 ? 0.9 * pow(err, -0.2) : 5;
    if (fac > 5) fac = 5;
    if (fac < 0.2) fac = 0.2;
    if (err <= 1) {
        memcpy(s, yn, sizeof yn);
        memcpy(fs, k7, sizeof k7);
        *t += hh;
        *h = hh * fac;
        return 1;
    }
    *h = hh * fac;
    return 0;
}

/* cubic Hermite between two states (and their derivatives) at fraction u of h */
static void hermite4(const double *a, const double *fa, const double *b, const double *fb, double h, double u, double *o) {
    double u2 = u * u, u3 = u2 * u;
    double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    for (int i = 0; i < 4; i++) o[i] = h00 * a[i] + h10 * h * fa[i] + h01 * b[i] + h11 * h * fb[i];
}

/* integrate s0 for time T, sampling every dt into out (4 doubles each);
 * stops early on hitting a primary (radius r1c / r2c). Returns the count. */
#define CR_TOL 1e-11
static int cr_run(const double *s0, double T, double dt, double *out, int max, double r1c, double r2c, double *jdrift) {
    double s[4], fs[4], t = 0, h = 1e-3, next = 0, C0 = cr_jacobi(s0);
    memcpy(s, s0, sizeof s);
    cr_f(s, fs);
    int n = 0;
    if (jdrift) *jdrift = 0;
    while (n < max && next <= T) {
        double sa[4], fa[4], ta = t;
        memcpy(sa, s, sizeof s);
        memcpy(fa, fs, sizeof fs);
        if (h > dt) h = dt;
        if (!dp45(s, &t, &h, CR_TOL, fs)) continue;
        while (next <= t && n < max && next <= T) {
            hermite4(sa, fa, s, fs, t - ta, (next - ta) / (t - ta), out + 4 * n);
            n++;
            next = n * dt;
        }
        if (jdrift) *jdrift = fmax(*jdrift, fabs(cr_jacobi(s) - C0));
        double m = cr_mu;
        double r1 = hypot(s[0] + m, s[1]), r2 = hypot(s[0] - 1 + m, s[1]);
        if (r1 < r1c || r2 < r2c) break;
    }
    return n;
}

/* the collinear Lagrange points by Newton on the x-axis force balance */
static double cr_lx(double x) {
    double g[4] = {x, 0, 0, 0}, d[4];
    for (int i = 0; i < 60; i++) {
        g[0] = x;
        cr_f(g, d);
        double e = 1e-7, g2[4] = {x + e, 0, 0, 0}, d2[4];
        cr_f(g2, d2);
        double dx = d[2] / ((d2[2] - d[2]) / e);
        x -= dx;
        if (fabs(dx) < 1e-15) break;
    }
    return x;
}
static double Lx[5], Ly[5], LC[5];
static void cr_lpoints(void) {
    double m = cr_mu, h = cbrt(m / 3);
    Lx[0] = cr_lx(1 - m - h); Lx[1] = cr_lx(1 - m + h); Lx[2] = cr_lx(-1 - 5 * m / 12);
    Lx[3] = Lx[4] = 0.5 - m;
    Ly[0] = Ly[1] = Ly[2] = 0; Ly[3] = sqrt(3) / 2; Ly[4] = -sqrt(3) / 2;
    for (int i = 0; i < 5; i++) LC[i] = cr_2omega(Lx[i], Ly[i]);
}

/* a planar Lyapunov orbit about L1 by differential correction: from
 * (xL1 - ax, 0) moving straight up, find the speed that crosses y = 0 again
 * perpendicularly (secant on vx there). Returns the period; s0 the start. */
static double cr_cross_vx(double x0, double vy0, double *thalf) {
    double s[4] = {x0, 0, 0, vy0}, fs[4], t = 0, h = 1e-3;
    cr_f(s, fs);
    for (int it = 0; it < 200000; it++) {
        double sa[4], fa[4], ta = t;
        memcpy(sa, s, sizeof s);
        memcpy(fa, fs, sizeof fs);
        if (h > 0.02) h = 0.02;
        if (!dp45(s, &t, &h, 1e-13, fs)) continue;
        if (t > 0.3 && sa[1] > 0 && s[1] <= 0) {
            double lo = 0, hi = 1, o[4];
            for (int k = 0; k < 60; k++) {
                double mid = 0.5 * (lo + hi);
                hermite4(sa, fa, s, fs, t - ta, mid, o);
                if (o[1] > 0) lo = mid; else hi = mid;
            }
            hermite4(sa, fa, s, fs, t - ta, 0.5 * (lo + hi), o);
            *thalf = ta + 0.5 * (lo + hi) * (t - ta);
            return o[2];
        }
        if (t > 10) break;
    }
    *thalf = 0;
    return NAN;
}
static double cr_lyapunov(double ax, double *s0) {
    double m = cr_mu, g = 1 - m - Lx[0];
    double c2 = m / (g * g * g) + (1 - m) / pow(1 - g, 3);
    double w = sqrt((2 - c2 + sqrt(9 * c2 * c2 - 8 * c2)) / 2), k = (w * w + 1 + 2 * c2) / (2 * w);
    /* bracketed: the crossing jumps branch below the linear guess, so bisect
     * from either side of the true root */
    double x0 = Lx[0] - ax, lo = k * ax * w * 0.95, hi = k * ax * w * 1.45, th, v1 = 0;
    double flo = cr_cross_vx(x0, lo, &th);
    for (int it = 0; it < 64; it++) {
        v1 = 0.5 * (lo + hi);
        double f = cr_cross_vx(x0, v1, &th);
        if ((f < 0) == (flo < 0)) { lo = v1; flo = f; } else hi = v1;
        if (hi - lo < 1e-15) break;
    }
    cr_cross_vx(x0, v1, &th);
    set3(s0, x0, 0, 0);
    s0[3] = v1;
    return 2 * th;
}

/* ---- the atmosphere: exponential between tabulated base altitudes
 * (Vallado, "Fundamentals of Astrodynamics", table 8-4) ---- */
static const double ATM[][3] = {
    {0, 1.225, 7.249}, {25, 3.899e-2, 6.349}, {30, 1.774e-2, 6.682}, {40, 3.972e-3, 7.554},
    {50, 1.057e-3, 8.382}, {60, 3.206e-4, 7.714}, {70, 8.770e-5, 6.549}, {80, 1.905e-5, 5.799},
    {90, 3.396e-6, 5.382}, {100, 5.297e-7, 5.877}, {110, 9.661e-8, 7.263}, {120, 2.438e-8, 9.473},
    {130, 8.484e-9, 12.636}, {140, 3.845e-9, 16.149}, {150, 2.070e-9, 22.523}, {180, 5.464e-10, 29.740},
    {200, 2.789e-10, 37.105}, {250, 7.248e-11, 45.546}, {300, 2.418e-11, 53.628}, {350, 9.518e-12, 53.298},
    {400, 3.725e-12, 58.515}, {450, 1.585e-12, 60.828}, {500, 6.967e-13, 63.822}, {600, 1.454e-13, 71.835},
    {700, 3.614e-14, 88.667}, {800, 1.170e-14, 124.64}, {900, 5.245e-15, 181.05}, {1000, 3.019e-15, 268.00},
};
#define NATM ((int)(sizeof ATM / sizeof ATM[0]))
static double atm_rho(double h) {
    if (h < 0) h = 0;
    int i = NATM - 1;
    while (i > 0 && h < ATM[i][0]) i--;
    return ATM[i][1] * exp(-(h - ATM[i][0]) / ATM[i][2]);
}

/* ---- monotone cubic interpolation (Fritsch-Carlson): the eased mapping
 * from display time to mission time, and its slope (the time warp) ---- */
static double pchip(const double *x, const double *y, int n, double t, double *slope) {
    if (t <= x[0]) { if (slope) *slope = (y[1] - y[0]) / (x[1] - x[0]) * 0; return y[0]; }
    if (t >= x[n - 1]) { if (slope) *slope = 0; return y[n - 1]; }
    int k = 0;
    while (k < n - 2 && t > x[k + 1]) k++;
    double m[2];
    for (int j = 0; j < 2; j++) {
        int i = k + j;
        double dl = i > 0 ? (y[i] - y[i - 1]) / (x[i] - x[i - 1]) : 0;
        double dr = i < n - 1 ? (y[i + 1] - y[i]) / (x[i + 1] - x[i]) : 0;
        if (i == 0 || i == n - 1) m[j] = 0;
        else if (dl * dr <= 0) m[j] = 0;
        else {
            double hl = x[i] - x[i - 1], hr = x[i + 1] - x[i];
            double w1 = 2 * hr + hl, w2 = hr + 2 * hl;
            m[j] = (w1 + w2) / (w1 / dl + w2 / dr);
        }
    }
    double h = x[k + 1] - x[k], u = (t - x[k]) / h, u2 = u * u, u3 = u2 * u;
    double v = (2 * u3 - 3 * u2 + 1) * y[k] + (u3 - 2 * u2 + u) * h * m[0] + (-2 * u3 + 3 * u2) * y[k + 1] + (u3 - u2) * h * m[1];
    if (slope)
        *slope = ((6 * u2 - 6 * u) * y[k] + (3 * u2 - 4 * u + 1) * h * m[0] + (-6 * u2 + 6 * u) * y[k + 1] + (3 * u2 - 2 * u) * h * m[1]) / h;
    return v;
}

/* ================================================================== */
/* ---- the display ----                                               */
/* ================================================================== */
static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check, opt_act = -1;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot, *opt_palette = "mission";
static double opt_at = 40;             /* --shot: the moment this many seconds after the epoch */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the show started at */
static const char *opt_font_mono = "JetBrains Mono,JetBrainsMono Nerd Font Mono,JetBrainsMono NFM,DejaVu Sans Mono";
static const char *opt_font_sans = "Inter,Cantarell,DejaVu Sans";
static double t_wall0;

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { R_FRONT, R_PANEL, R_SIDE };
static int role = R_FRONT, has_panel, has_side, side_rank;

static void die(const char *what) { fprintf(stderr, "orbits: %s\n", what); exit(1); }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static double wall(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static double clampd(double x, double a, double b) { return x < a ? a : x > b ? b : x; }
static double smooth(double a, double b, double x) { double t = clampd((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static double lerpd(double a, double b, double t) { return a + (b - a) * t; }

static uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static double hrand(uint32_t a, uint32_t b) { return (hash32(a * 0x9E3779B9U ^ hash32(b + 0x632BE5ABU)) >> 8) * (1.0 / 16777216.0); }

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

static char host[64], mission_name[64];
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

/* ---- palettes ---- */
typedef struct {
    const char *name;
    float bg[3], bg2[3], ink[3], dim[3], faint[3], orb[3], orb2[3], hi[3], alt[3], warn[3], ok[3];
    float sea[3], land[3], mars[3], moon[3], jup[3], sun[3];
    int paper;
    float glow;
} Pal;
#define RGB(h) {((h) >> 16 & 255) / 255.0f, ((h) >> 8 & 255) / 255.0f, ((h) & 255) / 255.0f}
static const Pal PALS[] = {
    {"mission", RGB(0x04070c), RGB(0x0b1622), RGB(0xd8e7f3), RGB(0x7f96ab), RGB(0x24364a), RGB(0x57d6ff), RGB(0x3f78b5),
     RGB(0xffb547), RGB(0xc48dff), RGB(0xff5d52), RGB(0x6fe3a0),
     RGB(0x173f6e), RGB(0x3c6e55), RGB(0xc4683d), RGB(0x9b9a96), RGB(0xc9a47c), RGB(0xffd27a), 0, 1.0f},
    {"phosphor", RGB(0x010501), RGB(0x041207), RGB(0xa6ffbe), RGB(0x46b066), RGB(0x123f1e), RGB(0x62ff8f), RGB(0x2b9e4d),
     RGB(0xd2ff70), RGB(0x3fd6a6), RGB(0xfff07a), RGB(0x62ff8f),
     RGB(0x0b3518), RGB(0x2a7a3f), RGB(0x5fbf66), RGB(0x58a868), RGB(0x6fcf7a), RGB(0xc8ff9a), 0, 1.25f},
    {"paper", RGB(0xf1ece0), RGB(0xe8e1d0), RGB(0x1d2532), RGB(0x5c6573), RGB(0xc9c0ad), RGB(0x1f4f8f), RGB(0x6c7b90),
     RGB(0xc0392b), RGB(0x2e7d4f), RGB(0xa61e1e), RGB(0x2e7d4f),
     RGB(0xa8bdcf), RGB(0xd2c497), RGB(0xc98a63), RGB(0xb9b4a8), RGB(0xcdb38c), RGB(0xe3a443), 1, 0.0f},
    {"mono", RGB(0x050505), RGB(0x0e0e0e), RGB(0xececec), RGB(0x8c8c8c), RGB(0x2e2e2e), RGB(0xffffff), RGB(0x9c9c9c),
     RGB(0xffffff), RGB(0xc4c4c4), RGB(0xffffff), RGB(0xffffff),
     RGB(0x1c1c1c), RGB(0x555555), RGB(0x8a8a8a), RGB(0x8a8a8a), RGB(0x9a9a9a), RGB(0xf0f0f0), 0, 0.75f},
};
#define NPAL ((int)(sizeof PALS / sizeof PALS[0]))
static const Pal *pal = &PALS[0];

/* ---- glyph atlas: printable ASCII plus the symbols astrodynamics needs,
 * in each text style used, rasterised at its exact pixel size ---- */
static const uint32_t XCP[] = {
    0xB0, 0x394, 0x3BC, 0x221A, 0xB2, 0xB3, 0x2080, 0x2081, 0x2082, 0x2083, 0x2084, 0x2085, 0x2086, 0x2087, 0x2088, 0x2089,
    0x221E, 0xB7, 0x2192, 0x2248, 0x3B8, 0x3BD, 0x3C9, 0x3A9, 0x3C0, 0x3B4, 0x3C6, 0x3BB, 0x3B5, 0x3C1, 0x2212, 0xD7,
    0x2295, 0x2264, 0x2265, 0x2032, 0x25B2, 0x25CF, 0x1E8B, 0x1E8D, 0x1E8F, 0xFF, 0x2202, 0xB1, 0x2014, 0x2013, 0x207B,
    0x2070, 0xB9, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079, 0x3B3, 0x3B1, 0x25CB, 0x25B6, 0x25C6, 0x2190, 0x2191, 0x2193, 0xBD, 0x207A,
};
#define NX ((int)(sizeof XCP / sizeof XCP[0]))
#define NGLYPH (95 + NX)
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGLYPH]; } Style;
enum { F_MONO, F_MONOB, F_SANS, F_SANSB, F_LIGHT, NFACE };
#define MAXSTYLE 48
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h, atlas_dirty;

static FT_Face load_face(const char *fam, const char *weight) {
    char pat_s[400];
    snprintf(pat_s, sizeof pat_s, "%s:weight=%s", fam, weight);
    FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    FT_Face f;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    int idx = 0;
    FcPatternGetInteger(m, FC_INDEX, 0, &idx);
    if (FT_New_Face(ft, (const char *)file, idx, &f)) die("can't load a font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
    return f;
}

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    faces[F_MONO] = load_face(opt_font_mono, "regular");
    faces[F_MONOB] = load_face(opt_font_mono, "bold");
    faces[F_SANS] = load_face(opt_font_sans, "regular");
    faces[F_SANSB] = load_face(opt_font_sans, "semibold");
    faces[F_LIGHT] = load_face(opt_font_sans, "light");
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 8) p = 8;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) die("too many text styles");
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    atlas_dirty = 1;
    return nstyles++;
}

static void build_atlas(void) {
    free(atlas);
    int cap = 4096;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < NGLYPH; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c >= 95 ? XCP[c - 95] : (FT_ULong)(32 + c);
            if (!FT_Get_Char_Index(f, code) || FT_Load_Char(f, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
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
    atlas_dirty = 2;
}

/* UTF-8 */
static uint32_t utf8_next(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t c = s[0];
    int n = 1;
    if (c >= 0xF0 && s[1] && s[2] && s[3]) { c = ((c & 7) << 18) | ((s[1] & 63) << 12) | ((s[2] & 63) << 6) | (s[3] & 63); n = 4; }
    else if (c >= 0xE0 && s[1] && s[2]) { c = ((c & 15) << 12) | ((s[1] & 63) << 6) | (s[2] & 63); n = 3; }
    else if (c >= 0xC0 && s[1]) { c = ((c & 31) << 6) | (s[1] & 63); n = 2; }
    *p += n;
    return c;
}
static int gidx(uint32_t c) {
    if (c >= 32 && c < 127) return (int)c - 32;
    for (int i = 0; i < NX; i++) if (XCP[i] == c) return 95 + i;
    return 0;
}
static const Glyph *glyph(int s, uint32_t c) { return &styles[s].g[gidx(c)]; }
static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p;) { uint32_t c = utf8_next(&p); w += glyph(s, c)->adv + (*p ? track : 0); }
    return w;
}
static float cap_h(int s) { return glyph(s, 'H')->top; }
static float line_h(int s) { return styles[s].px * 1.32f; }

/* ---- the primitive stream: every shape is an instanced quad whose
 * fragment shader works out its own anti-aliased coverage. Per instance:
 * centre, half extents, angle, kind, 4 params, rgba, 2 more params. ---- */
enum { K_LINE, K_GLOWL, K_CIRC, K_TEXT, K_RECT, K_GLOW, K_PLANET, K_MAP, K_FIELD, K_PORK, K_TRI };
#define IF 16
static float *ib;
static int ibn, ibcap;
static float alpha_mul = 1;

static void inst(float cx, float cy, float hx, float hy, float ang, int k, const float *p, const float *c, float q0, float q1) {
    if (ibn + 1 > ibcap) ib = realloc(ib, (size_t)(ibcap = ibcap ? ibcap * 2 : 8192) * IF * sizeof *ib);
    float *o = ib + (size_t)ibn * IF;
    o[0] = cx; o[1] = cy; o[2] = hx; o[3] = hy; o[4] = ang; o[5] = (float)k;
    o[6] = p[0]; o[7] = p[1]; o[8] = p[2]; o[9] = p[3];
    o[10] = c[0]; o[11] = c[1]; o[12] = c[2]; o[13] = c[3] * alpha_mul;
    o[14] = q0; o[15] = q1;
    ibn++;
}

#define C4(r, g, b, a) ((const float[4]){(r), (g), (b), (a)})
static const float *ca(const float *c, float a) {
    static float buf[64][4];
    static int bi;
    float *o = buf[bi++ & 63];
    o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = a;
    return o;
}
/* additive: a negative alpha (glows; nothing on paper) */
static const float *cadd(const float *c, float a) { return ca(c, pal->paper ? 0 : -a * pal->glow); }
static const float *mixc(const float *a, const float *b, float t, float al) {
    static float buf[16][4];
    static int bi;
    float *o = buf[bi++ & 15];
    for (int i = 0; i < 3; i++) o[i] = a[i] + (b[i] - a[i]) * t;
    o[3] = al;
    return o;
}

/* a capsule from a to b, w px wide; dash > 0: dashes of that period, the
 * pattern offset by phase px (so a polyline's dashes run on) */
static void line_dp(float x0, float y0, float x1, float y1, float w, const float *c, float dash, float phase) {
    if (c[3] == 0) return;
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    float hw = w * 0.5f;
    float p[4] = {l * 0.5f, hw, dash, phase};
    inst((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, l * 0.5f + hw + 1.5f, hw + 1.5f, atan2f(dy, dx), K_LINE, p, c, 0.5f, 0);
}
static void line(float x0, float y0, float x1, float y1, float w, const float *c) { line_dp(x0, y0, x1, y1, w, c, 0, 0); }
static void line_d(float x0, float y0, float x1, float y1, float w, const float *c, float dash) { line_dp(x0, y0, x1, y1, w, c, dash, 0); }
/* a soft halo along a segment, additive */
static void glowline(float x0, float y0, float x1, float y1, float w, const float *c) {
    if (c[3] == 0) return;
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    float p[4] = {l * 0.5f, w * 0.5f, 0, 0};
    inst((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, l * 0.5f + w * 0.5f, w * 0.5f, atan2f(dy, dx), K_GLOWL, p, c, 0, 0);
}
static void disc(float x, float y, float r, const float *c) {
    float p[4] = {r, 0, 0, 0};
    inst(x, y, r + 1.5f, r + 1.5f, 0, K_CIRC, p, c, 0, 0);
}
static void ring(float x, float y, float r, float w, const float *c) {
    float p[4] = {r, w * 0.5f, 0, 0};
    inst(x, y, r + w * 0.5f + 1.5f, r + w * 0.5f + 1.5f, 0, K_CIRC, p, c, 0, 0);
}
static void ring_d(float x, float y, float r, float w, const float *c, float ndash) {
    float p[4] = {r, w * 0.5f, ndash, 0};
    inst(x, y, r + w * 0.5f + 1.5f, r + w * 0.5f + 1.5f, 0, K_CIRC, p, c, 0, 0);
}
static void glow(float x, float y, float r, const float *c) {
    float p[4] = {r, 0, 0, 0};
    inst(x, y, r, r, 0, K_GLOW, p, c, 0, 0);
}
static void rect(float x0, float y0, float x1, float y1, float rad, const float *c) {
    float p[4] = {(x1 - x0) * 0.5f, (y1 - y0) * 0.5f, rad, 0};
    inst((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, p[0] + 1.5f, p[1] + 1.5f, 0, K_RECT, p, c, 0, 0);
}
static void rect_o(float x0, float y0, float x1, float y1, float rad, float w, const float *c) {
    float p[4] = {(x1 - x0) * 0.5f, (y1 - y0) * 0.5f, rad, w};
    inst((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, p[0] + 1.5f, p[1] + 1.5f, 0, K_RECT, p, c, 0, 0);
}
/* a triangle pointing along ang, s px from centre to tip; w > 0 an outline */
static void tri(float x, float y, float ang, float s, const float *c, float w) {
    float p[4] = {s, w, 0, 0};
    inst(x, y, s * 1.2f + 2, s * 1.2f + 2, ang, K_TRI, p, c, 0, 0);
}
static void arrow(float x0, float y0, float x1, float y1, float w, const float *c, float head) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1) return;
    dx /= l; dy /= l;
    float h = fminf(head, l * 0.6f);
    line(x0, y0, x1 - dx * h * 0.8f, y1 - dy * h * 0.8f, w, c);
    tri(x1 - dx * h * 0.5f, y1 - dy * h * 0.5f, atan2f(dy, dx), h * 0.55f, c, 0);
}

/* set str with its baseline at (x, y); align -1 left, 0 centred, 1 right */
static float text_t(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = roundf(x - (align == 0 ? w * 0.5f : align > 0 ? w : 0));
    y = roundf(y);
    static const float zp[4] = {0, 0, 0, 0};
    for (const char *p = str; *p;) {
        uint32_t cp = utf8_next(&p);
        const Glyph *g = glyph(s, cp);
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top;
            float pp[4] = {g->x + g->w * 0.5f, g->y + g->h * 0.5f, 0, 0};
            (void)zp;
            inst(x0 + g->w * 0.5f, y0 + g->h * 0.5f, g->w * 0.5f, g->h * 0.5f, 0, K_TEXT, pp, col, 0, 0);
        }
        pen += g->adv + track;
    }
    return w;
}
static float text(int s, float x, float y, int align, const char *str, const float *col) { return text_t(s, x, y, align, str, col, 0); }
/* text with a halo of the background behind it, for labels over lines */
static float text_h(int s, float x, float y, int align, const char *str, const float *col) {
    float w = text_width(s, str, 0), ch = cap_h(s);
    float x0 = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    rect(x0 - 3, y - ch - 3, x0 + w + 3, y + 4, 2, ca(pal->bg, 0.72f * col[3]));
    return text(s, x, y, align, str, col);
}

/* ---- text styles, sized to the output ---- */
static float UI = 1;                   /* px per "point" */
static int st_tiny, st_lab, st_small, st_body, st_bodyb, st_eq, st_h, st_title, st_clock, st_num, st_numb, st_sub;
static void styles_init(void) {
    float m = fminf((float)W, (float)H);
    nstyles = 0;                       /* sizes follow the output: start the cache afresh */
    atlas_dirty = 1;
    UI = clampf(m / 1080.0f, 0.55f, 3.0f);
    st_tiny = style(F_MONO, 11 * UI);
    st_lab = style(F_MONO, 12.5f * UI);
    st_small = style(F_SANS, 13.5f * UI);
    st_body = style(F_SANS, 15.5f * UI);
    st_bodyb = style(F_SANSB, 15.5f * UI);
    st_eq = style(F_SANS, 18 * UI);
    st_h = style(F_SANSB, 13 * UI);
    st_sub = style(F_SANS, 16.5f * UI);
    st_title = style(F_SANSB, 31 * UI);
    st_clock = style(F_LIGHT, 50 * UI);
    st_num = style(F_MONO, 14.5f * UI);
    st_numb = style(F_MONOB, 14.5f * UI);
    if (atlas_dirty) build_atlas();
}

/* ---- views: an orthographic camera onto a world, into a screen rect ---- */
typedef struct {
    float x0, y0, x1, y1;              /* screen rect */
    float cx, cy;                      /* where the world centre lands */
    double s;                          /* px per world unit */
    double c[3];                       /* world centre */
    double R[3], D[3], O[3];           /* screen right, down, and out towards the viewer, in world axes */
    double yaw, el;
    int nocc;
    double occ[4][4];                  /* spheres that hide what's behind them: centre, radius */
} View;

static void view_init(View *v, float x0, float y0, float x1, float y1) {
    memset(v, 0, sizeof *v);
    v->x0 = x0; v->y0 = y0; v->x1 = x1; v->y1 = y1;
    v->cx = (x0 + x1) * 0.5f; v->cy = (y0 + y1) * 0.5f;
    v->s = 1;
}
/* looking from azimuth yaw, elevation el (radians) above the world's xy-plane */
static void view_cam(View *v, double yaw, double el) {
    double ce = cos(el), se = sin(el);
    v->yaw = yaw; v->el = el;
    set3(v->O, ce * cos(yaw), ce * sin(yaw), se);
    set3(v->R, -sin(yaw), cos(yaw), 0);
    double up[3];
    cross3(v->O, v->R, up);
    scale3(up, -1, v->D);
}
static void view_occ(View *v, const double *c, double r) {
    if (v->nocc < 4) { memcpy(v->occ[v->nocc], c, 3 * sizeof *c); v->occ[v->nocc][3] = r; v->nocc++; }
}
static void proj(const View *v, const double *p, float *x, float *y) {
    double d[3];
    sub3(p, v->c, d);
    *x = v->cx + (float)(v->s * dot3(v->R, d));
    *y = v->cy + (float)(v->s * dot3(v->D, d));
}
static int hidden(const View *v, const double *p) {
    for (int i = 0; i < v->nocc; i++) {
        double d[3];
        sub3(p, v->occ[i], d);
        double z = dot3(v->O, d), r = v->occ[i][3];
        if (z >= 0) continue;
        double a = dot3(v->R, d), b = dot3(v->D, d);
        if (a * a + b * b < r * r) return 1;
    }
    return 0;
}

/* a polyline through world points: w px, dashed if dash > 0, a halo if
 * halo > 0; alpha runs a0 -> a1 along it (trails fade) */
static void poly(const View *v, const double *pts, int n, float w, const float *c, float dash, float halo, float a0, float a1) {
    float px = 0, py = 0, phase = 0;
    int ph = 0;
    float m = 40;
    for (int i = 0; i < n; i++) {
        float x, y;
        proj(v, pts + 3 * i, &x, &y);
        int hd = hidden(v, pts + 3 * i);
        if (i > 0) {
            float dx = x - px, dy = y - py, l = sqrtf(dx * dx + dy * dy);
            int vis = !(hd && ph);
            int off = (x < v->x0 - m && px < v->x0 - m) || (x > v->x1 + m && px > v->x1 + m) ||
                      (y < v->y0 - m && py < v->y0 - m) || (y > v->y1 + m && py > v->y1 + m) || l > 6000;
            if (vis && !off && l > 0.01f) {
                float t = n > 1 ? (float)i / (n - 1) : 1;
                float al = c[3] * (a0 + (a1 - a0) * t);
                if (hd || ph) al *= 0.25f;
                if (halo > 0 && !pal->paper) glowline(px, py, x, y, w * halo, cadd(c, al * 0.22f));
                line_dp(px, py, x, y, w, ca(c, al), dash, -phase);
            }
            phase += l;
            if (dash > 0) phase = fmodf(phase, dash);
        }
        px = x; py = y; ph = hd;
    }
}

/* sample a conic by true anomaly from nu0 to nu1 into pts (3 per point) */
static int conic_pts(const Orbit *o, double nu0, double nu1, int n, double *pts) {
    for (int i = 0; i < n; i++) {
        double nu = nu0 + (nu1 - nu0) * i / (n - 1);
        orb_at_nu(o, nu, pts + 3 * i, NULL);
    }
    return n;
}
static void draw_conic(const View *v, const Orbit *o, double nu0, double nu1, float w, const float *c, float dash, float halo) {
    static double pts[3 * 400];
    int n = (int)clampd(fabs(nu1 - nu0) / TAU_D * 360, 24, 400);
    conic_pts(o, nu0, nu1, n, pts);
    poly(v, pts, n, w, c, dash, halo, 1, 1);
}
/* a circle of radius r in the plane through c with normal (0,0,1) */
static void draw_circle(const View *v, const double *c, double r, float w, const float *col, float dash) {
    static double pts[3 * 241];
    for (int i = 0; i <= 240; i++) {
        double a = TAU_D * i / 240;
        set3(pts + 3 * i, c[0] + r * cos(a), c[1] + r * sin(a), c[2]);
    }
    poly(v, pts, 241, w, col, dash, 0, 1, 1);
}

/* a planet: a shaded disc lit from sun (world direction), at least minr px */
enum { B_EARTH, B_MOON, B_MARS, B_JUPITER, B_SUN, B_PLAIN };
static void planet(const View *v, const double *p, double r, int body, const double *sun, float spin, float minr, const float *col) {
    float x, y;
    proj(v, p, &x, &y);
    float rr = fmaxf((float)(r * v->s), minr);
    double sd[3] = {0, 0, 1};
    if (sun) unit3(sun, sd);
    float pp[4] = {rr, (float)dot3(v->R, sd), (float)dot3(v->D, sd), (float)dot3(v->O, sd)};
    if (body == B_EARTH && !pal->paper) glow(x, y, rr * 1.25f, cadd(pal->orb, 0.22f));
    if (body == B_SUN && !pal->paper) { glow(x, y, rr * 5, cadd(pal->sun, 0.35f)); glow(x, y, rr * 2.2f, cadd(pal->sun, 0.5f)); }
    float e = body == B_EARTH ? rr * 0.06f + 2 : 1.5f;
    inst(x, y, rr + e, rr + e, (float)v->el, K_PLANET, pp, ca(col, 1), (float)body, spin - (float)v->yaw);
}

/* ---- the world map: a coarse coastline, from memory (lon, lat pairs,
 * each polygon closed by its first point). Black Sea and Caspian are holes
 * in Eurasia (even-odd). ---- */
static const short COAST[] = {
    /* North America */
    -168,66, -162,70, -156,71, -141,70, -128,70, -115,68, -95,68, -90,69, -82,68, -80,63, -93,61, -94,58, -88,56, -82,55,
    -79,52, -77,55, -77,60, -72,61, -65,60, -61,56, -56,52, -60,48, -64,45, -66,44, -70,43, -70,41, -74,40, -76,37, -75,35,
    -78,33, -81,31, -80,27, -80,25, -82,26, -83,28, -84,30, -89,30, -94,29, -97,27, -97,24, -97,21, -95,18, -91,19, -90,21,
    -87,21, -88,18, -88,16, -84,15, -83,11, -80,9, -77,8, -79,8, -82,8, -85,10, -87,13, -91,14, -96,16, -101,17, -105,20,
    -105,23, -109,26, -112,29, -114,31, -112,27, -110,23, -112,24, -114,27, -115,29, -117,32, -120,34, -122,37, -124,40,
    -124,46, -125,48, -127,50, -130,54, -133,57, -137,59, -141,60, -146,61, -150,59, -154,57, -158,56, -162,55, -158,58,
    -162,59, -165,61, -165,63, -161,64, -166,65, 9999,
    /* Greenland */
    -73,78, -60,82, -40,83, -22,82, -18,77, -21,72, -24,69, -32,68, -40,65, -43,60, -48,61, -52,64, -54,68, -55,71, -60,76,
    -68,77, 9999,
    /* Baffin, Ellesmere, Victoria */
    -80,73, -71,70, -62,66, -64,63, -68,62, -74,64, -78,64, -73,68, -79,70, -89,72, 9999,
    -90,81, -62,82, -70,78, -80,76, -90,77, 9999,
    -118,73, -102,73, -100,69, -112,68, -118,70, 9999,
    /* South America */
    -77,8, -75,11, -71,12, -68,10, -62,10, -60,8, -57,6, -52,5, -50,1, -48,-1, -44,-2, -38,-4, -35,-7, -35,-9, -38,-13,
    -39,-18, -40,-21, -44,-23, -48,-26, -48,-28, -51,-31, -53,-34, -57,-35, -57,-37, -62,-39, -65,-41, -63,-42, -65,-45,
    -67,-46, -66,-48, -68,-50, -68,-52, -66,-55, -70,-55, -74,-52, -75,-48, -74,-44, -73,-40, -73,-36, -71,-32, -71,-28,
    -70,-23, -70,-18, -75,-15, -77,-12, -79,-7, -81,-5, -80,-2, -80,0, -79,1, -78,4, -77,7, 9999,
    /* Africa */
    -17,21, -16,24, -14,26, -13,28, -10,29, -9,32, -7,34, -6,36, -2,35, 3,37, 10,37, 11,35, 10,34, 11,33, 15,32, 19,30,
    20,31, 22,33, 25,32, 29,31, 32,31, 34,28, 35,24, 37,21, 39,18, 40,15, 42,13, 43,12, 44,10, 51,12, 51,10, 49,7, 46,2,
    43,0, 40,-3, 39,-5, 39,-8, 40,-10, 40,-15, 37,-18, 35,-22, 33,-26, 33,-29, 30,-31, 27,-33, 25,-34, 22,-34, 20,-35,
    18,-34, 18,-31, 16,-28, 15,-26, 14,-23, 13,-20, 12,-17, 12,-13, 14,-11, 13,-8, 12,-6, 12,-3, 9,-1, 10,2, 10,4, 8,4,
    6,4, 4,6, 1,6, -2,5, -4,5, -7,4, -9,5, -11,7, -13,9, -15,11, -17,12, -17,15, -16,16, -16,18, -17,20, 9999,
    /* Madagascar */
    49,-12, 50,-15, 49,-17, 47,-25, 45,-25, 44,-23, 44,-20, 44,-17, 46,-16, 48,-14, 9999,
    /* Eurasia */
    -9,39, -9,43, -8,44, -2,43, -1,46, -3,47, -5,48, -2,49, 1,50, 3,51, 4,52, 6,53, 9,54, 8,55, 8,57, 10,58, 11,56, 11,54,
    13,54, 14,54, 18,55, 21,55, 21,57, 23,57, 24,59, 28,59, 30,60, 26,60, 22,60, 21,63, 25,65, 22,66, 17,62, 17,61, 19,59,
    16,57, 14,55, 13,55, 12,58, 11,59, 8,58, 6,59, 5,61, 7,63, 11,64, 14,67, 16,69, 19,70, 23,71, 28,71, 31,70, 33,69,
    41,67, 39,66, 35,66, 34,64, 37,64, 40,64, 44,66, 43,68, 46,68, 53,69, 59,68, 68,68, 69,73, 72,73, 73,68, 76,72, 80,73,
    87,74, 96,76, 104,78, 113,74, 120,73, 128,73, 129,71, 140,72, 150,71, 160,70, 170,70, 180,69, 180,65, 177,62, 172,61,
    164,60, 163,56, 162,55, 160,53, 157,51, 156,57, 160,61, 155,59, 152,59, 143,59, 137,54, 141,52, 140,48, 135,43, 132,43,
    130,41, 129,36, 126,34, 126,37, 125,40, 121,39, 121,41, 118,39, 119,37, 122,37, 120,35, 121,32, 122,30, 119,25, 116,23,
    113,22, 110,21, 108,21, 106,20, 105,18, 108,15, 109,11, 106,10, 105,9, 103,11, 100,13, 99,10, 100,7, 103,5, 103,1,
    101,3, 98,8, 98,12, 97,16, 94,16, 94,19, 92,21, 90,22, 88,21, 86,20, 85,19, 82,16, 80,15, 80,13, 79,10, 77,8, 76,9,
    74,13, 73,16, 72,19, 72,21, 70,21, 68,23, 66,25, 61,25, 57,26, 56,27, 54,26, 51,28, 50,30, 48,30, 48,28, 50,26, 51,25,
    51,24, 53,24, 56,25, 56,24, 59,22, 58,20, 55,17, 52,16, 48,14, 45,13, 43,13, 42,16, 41,19, 39,22, 37,25, 35,28, 34,28,
    35,30, 34,31, 35,33, 36,36, 32,36, 29,36, 27,37, 26,39, 26,40, 23,40, 23,38, 22,36, 21,38, 19,40, 19,42, 16,43, 14,45,
    12,45, 12,44, 14,42, 16,41, 18,40, 17,39, 16,38, 15,40, 14,41, 12,42, 10,43, 9,44, 7,44, 5,43, 3,43, 3,42, 1,41, 0,39,
    0,38, -1,37, -2,37, -4,37, -6,36, -6,37, -7,37, -9,37, 9999,
    /* Black Sea and Caspian (holes) */
    28,41, 28,43, 30,45, 33,46, 35,45, 37,45, 38,47, 39,47, 37,45, 40,43, 41,41, 36,42, 33,42, 30,41, 9999,
    47,45, 49,46, 53,47, 53,45, 51,44, 52,42, 54,41, 54,37, 51,37, 49,38, 49,40, 47,43, 9999,
    /* Chukotka past the date line */
    -180,69, -175,67, -170,66, -172,64, -180,65, 9999,
    /* Britain, Ireland, Iceland */
    -6,50, 1,51, 2,53, 0,53, -2,56, -2,58, -3,59, -5,59, -6,57, -6,56, -5,55, -3,54, -3,53, -5,53, -5,52, -3,51, 9999,
    -6,52, -6,54, -6,55, -8,55, -10,54, -10,52, -8,51, 9999,
    -24,65, -22,66, -16,67, -14,65, -15,64, -19,63, -23,64, 9999,
    /* Japan, Taiwan, Philippines */
    131,31, 132,34, 133,35, 136,36, 137,37, 140,39, 140,41, 142,41, 142,40, 141,36, 140,35, 139,35, 137,34, 135,34, 133,33, 9999,
    140,42, 141,45, 145,44, 146,43, 143,42, 9999,
    120,23, 121,25, 122,25, 121,22, 9999,
    121,18, 122,18, 122,16, 124,13, 121,14, 120,16, 9999,
    122,7, 126,7, 126,9, 125,10, 124,8, 9999,
    /* Indonesia, New Guinea, Sri Lanka */
    109,2, 111,2, 113,3, 115,5, 117,7, 119,5, 118,4, 118,1, 117,0, 116,-2, 116,-4, 113,-3, 111,-3, 110,-2, 9999,
    95,6, 97,5, 100,2, 104,-1, 106,-3, 106,-6, 104,-6, 102,-4, 100,-1, 99,2, 9999,
    105,-7, 108,-6, 111,-6, 114,-8, 114,-9, 111,-8, 108,-8, 9999,
    131,-1, 134,-1, 138,-2, 141,-3, 145,-4, 147,-6, 150,-10, 147,-10, 144,-8, 143,-9, 141,-9, 139,-8, 138,-6, 135,-4, 133,-4, 9999,
    80,8, 80,10, 82,7, 81,6, 9999,
    /* Australia, New Zealand */
    113,-22, 114,-26, 115,-30, 115,-34, 117,-35, 122,-34, 124,-33, 126,-32, 131,-31, 134,-32, 136,-35, 138,-33, 138,-35,
    140,-36, 140,-38, 143,-39, 146,-39, 150,-37, 151,-34, 153,-30, 153,-26, 151,-23, 149,-21, 146,-19, 145,-15, 144,-14,
    142,-11, 142,-13, 141,-16, 140,-18, 137,-16, 136,-15, 137,-12, 133,-11, 130,-13, 129,-15, 127,-14, 125,-15, 122,-18,
    119,-20, 116,-21, 9999,
    173,-34, 175,-37, 178,-38, 177,-39, 175,-41, 175,-40, 174,-39, 174,-37, 9999,
    173,-40, 174,-42, 173,-43, 171,-45, 169,-46, 166,-46, 168,-44, 170,-43, 9999,
    /* Cuba, Hispaniola */
    -85,22, -82,23, -77,22, -74,20, -77,20, -80,22, 9999,
    -74,19, -72,20, -68,19, -70,18, -72,18, 9999,
    /* Antarctica, closed along the pole */
    -180,-78, -150,-76, -120,-73, -100,-73, -75,-72, -62,-65, -58,-63, -62,-66, -62,-72, -40,-78, -20,-74, 0,-70, 30,-69,
    60,-67, 90,-66, 120,-66, 150,-68, 170,-71, 165,-78, 180,-78, 180,-90, -180,-90, 9999,
    10000,
};

#define LW 720
#define LH 360
static unsigned char *land;            /* equirectangular land mask, lon -180 at x = 0, lat 90 at y = 0 */
static void land_init(void) {
    land = calloc((size_t)LW * LH, 1);
    static float xs[512];
    for (int row = 0; row < LH; row++) {
        double lat = 90 - (row + 0.5) * 180.0 / LH;
        int nx = 0;
        const short *p = COAST;
        while (*p != 10000) {
            const short *s = p;
            int n = 0;
            while (s[2 * n] != 9999) n++;
            for (int i = 0; i < n; i++) {
                double x0 = s[2 * i], y0 = s[2 * i + 1], x1 = s[2 * ((i + 1) % n)], y1 = s[2 * ((i + 1) % n) + 1];
                if ((y0 > lat) != (y1 > lat) && nx < 512) xs[nx++] = (float)(x0 + (lat - y0) / (y1 - y0) * (x1 - x0));
            }
            p = s + 2 * n + 1;
        }
        for (int i = 1; i < nx; i++) for (int j = i; j > 0 && xs[j] < xs[j - 1]; j--) { float t = xs[j]; xs[j] = xs[j - 1]; xs[j - 1] = t; }
        for (int i = 0; i + 1 < nx; i += 2) {
            int a = (int)lround((xs[i] + 180) / 360 * LW), b = (int)lround((xs[i + 1] + 180) / 360 * LW);
            for (int x = a < 0 ? 0 : a; x < b && x < LW; x++) land[row * LW + x] = 255;
        }
    }
}

/* ================================================================== */
/* ---- the acts ----                                                  */
/* ================================================================== */
#define NACT 6
#define ACT_LEN 90.0
enum { A_HOHMANN, A_MARS, A_FLYBY, A_CR3BP, A_WALKER, A_DECAY };

typedef struct { float x0, y0, x1, y1; } Rect;
static void flush(const Rect *clip);
typedef struct { char k[32], v[48]; } KV;
typedef struct { char name[48], val[24]; int st; } DvRow;   /* st: 0 planned, 1 done, 2 total, 3 note */
typedef struct {
    const char *title, *sub, *eq[6], *dv_head, *frame;
    int neq, ntel, ndv;
    KV tel[14];
    DvRow dv[10];
    char clock[48];
    double warp;
} Info;

static void tel(Info *in, const char *k, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void tel(Info *in, const char *k, const char *fmt, ...) {
    if (in->ntel >= 14) return;
    KV *e = &in->tel[in->ntel++];
    snprintf(e->k, sizeof e->k, "%s", k);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->v, sizeof e->v, fmt, ap);
    va_end(ap);
}
static void dvrow(Info *in, const char *name, double v, int st) {
    if (in->ndv >= 10) return;
    DvRow *d = &in->dv[in->ndv++];
    snprintf(d->name, sizeof d->name, "%s", name);
    if (isnan(v)) d->val[0] = 0;
    else snprintf(d->val, sizeof d->val, "%.3f", v);
    d->st = st;
}
static void dvrow_s(Info *in, const char *name, const char *val, int st) {
    if (in->ndv >= 10) return;
    DvRow *d = &in->dv[in->ndv++];
    snprintf(d->name, sizeof d->name, "%s", name);
    snprintf(d->val, sizeof d->val, "%s", val);
    d->st = st;
}

/* "T+ 1d 04:12:33" */
static void fmt_met(char *b, size_t n, double s) {
    char sg = s < 0 ? '-' : '+';
    s = fabs(s);
    long t = (long)s, d = t / 86400;
    if (d > 0) snprintf(b, n, "T%c %ldd %02ld:%02ld:%02ld", sg, d, t / 3600 % 24, t / 60 % 60, t % 60);
    else snprintf(b, n, "T%c %02ld:%02ld:%02ld", sg, t / 3600, t / 60 % 60, t % 60);
}
static void fmt_date(char *b, size_t n, double jd) {
    time_t t = (time_t)((jd - 2440587.5) * 86400);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(b, n, "%Y-%m-%d", &tm);
}
static void fmt_dur(char *b, size_t n, double s) {
    if (s >= 2 * 86400) snprintf(b, n, "%.1f d", s / 86400);
    else if (s >= 3600) snprintf(b, n, "%.2f h", s / 3600);
    else snprintf(b, n, "%.1f min", s / 60);
}
/* thousands with thin spaces: 35786 -> "35 786" */
static void fmt_k(char *b, size_t n, double v) {
    long x = lround(fabs(v));
    char t[32];
    snprintf(t, sizeof t, "%ld", x);
    int L = (int)strlen(t), o = 0;
    if (v < 0 && o < (int)n - 1) b[o++] = '-';
    for (int i = 0; i < L && o < (int)n - 2; i++) {
        b[o++] = t[i];
        if ((L - i - 1) % 3 == 0 && i < L - 1) b[o++] = ' ';
    }
    b[o] = 0;
}

typedef struct { int n; double u[12], t[12]; } Warp;
static double warp_at(const Warp *w, double u, double *rate) {
    double r;
    double t = pchip(w->u, w->t, w->n, u, &r);
    if (rate) *rate = r;
    return t;
}

/* the act now showing and how far into it */
static int act, act_k = -1;
static double act_u, act_wt0;          /* seconds into the act; its start (wall clock) */
static uint32_t act_seed;
static float act_alpha = 1;

/* ---- shared drawing for the views ---- */

/* concentric range rings in the xy-plane about c out to rmax, every step,
 * with a label on each in unit (scaled by us) */
static void range_rings(const View *v, const double *c, double rmax, double step, double us, const char *unit, float a) {
    char b[48], k[32];
    for (int i = 1; i * step <= rmax * 1.0001; i++) {
        double r = i * step;
        draw_circle(v, c, r, 1, ca(pal->faint, a * (i % 2 ? 0.55f : 0.9f)), 0);
        double p[3] = {c[0] + r * cos(-0.5), c[1] + r * sin(-0.5), c[2]};
        float x, y;
        proj(v, p, &x, &y);
        if (step * us < 1) snprintf(k, sizeof k, "%.2f", r * us);
        else fmt_k(k, sizeof k, r * us);
        snprintf(b, sizeof b, "%s %s", k, unit);
        if (x > v->x0 + 20 && x < v->x1 - 60 && y > v->y0 + 20 && y < v->y1 - 20)
            text(st_tiny, x + 4, y - 3, -1, b, ca(pal->dim, a * 0.75f));
    }
    for (int i = 0; i < 12; i++) {
        double t = i * TAU_D / 12, p0[3] = {c[0] + step * 0.5 * cos(t), c[1] + step * 0.5 * sin(t), c[2]};
        double p1[3] = {c[0] + rmax * cos(t), c[1] + rmax * sin(t), c[2]};
        double pts[6] = {p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]};
        poly(v, pts, 2, 1, ca(pal->faint, a * 0.35f), 0, 0, 1, 1);
    }
}

/* the spacecraft: a bright point with a halo and a short velocity tick */
static void craft(const View *v, const double *p, const double *vel, const float *c, const char *label) {
    float x, y;
    proj(v, p, &x, &y);
    if (hidden(v, p)) return;
    glow(x, y, 26 * UI, cadd(c, 0.35f));
    disc(x, y, 3.6f * UI, ca(pal->paper ? pal->ink : C4(1, 1, 1, 1), 1));
    ring(x, y, 7 * UI, 1.2f * UI, ca(c, 0.9f));
    if (vel) {
        double q[3], u[3];
        unit3(vel, u);
        double d = 30 * UI / v->s;
        for (int i = 0; i < 3; i++) q[i] = p[i] + u[i] * d;
        float x1, y1;
        proj(v, q, &x1, &y1);
        float dx = x1 - x, dy = y1 - y, l = sqrtf(dx * dx + dy * dy);
        if (l > 4) line(x + dx / l * 9 * UI, y + dy / l * 9 * UI, x1, y1, 1.3f * UI, ca(c, 0.8f));
    }
    if (label) text_h(st_lab, x + 12 * UI, y - 10 * UI, -1, label, ca(pal->ink, 0.9f));
}

/* a burn: a chevron along the thrust at the place, Δv and its state */
static void burn_mark(const View *v, const double *p, const double *dir, const char *name, double dv, double dt_to, double flash, const double *ctr) {
    float x, y;
    proj(v, p, &x, &y);
    double q[3], u[3];
    unit3(dir, u);
    for (int i = 0; i < 3; i++) q[i] = p[i] + u[i] * (1 / v->s);
    float x1, y1;
    proj(v, q, &x1, &y1);
    float ang = atan2f(y1 - y, x1 - x);
    int done = dt_to <= 0;
    const float *c = pal->hi;
    if (flash > 0) glow(x, y, 60 * UI * (float)flash, cadd(c, 0.7f * (float)flash));
    tri(x + cosf(ang) * 11 * UI, y + sinf(ang) * 11 * UI, ang, 6.5f * UI, ca(c, done ? 0.95f : 0.8f), done ? 0 : 1.2f * UI);
    ring(x, y, 5 * UI, 1.2f * UI, ca(c, 0.9f));
    char b[64], t[32];
    snprintf(b, sizeof b, "%s  %+.3f km/s", name, dv);
    float cx, cy;
    proj(v, ctr, &cx, &cy);
    float ox = x - cx, oy = y - cy, ol = fmaxf(hypotf(ox, oy), 1);
    ox /= ol; oy /= ol;
    float lx = x + ox * 22 * UI, ly = y + oy * 22 * UI + (oy > 0 ? cap_h(st_lab) : 0);
    int al = ox > 0.35f ? -1 : ox < -0.35f ? 1 : 0;
    text_h(st_lab, lx, ly, al, b, ca(c, 0.95f));
    if (!done && dt_to < 1e8) {
        fmt_met(t, sizeof t, -dt_to);
        text_h(st_tiny, lx, ly + line_h(st_lab), al, t, ca(pal->dim, 0.9f));
    }
}

/* ---- plots for the panels ---- */
typedef struct { Rect r; double x0, x1, y0, y1; int lx, ly; } Plot;
static float pl_x(const Plot *p, double x) {
    double t = p->lx ? (log(x) - log(p->x0)) / (log(p->x1) - log(p->x0)) : (x - p->x0) / (p->x1 - p->x0);
    return p->r.x0 + (float)t * (p->r.x1 - p->r.x0);
}
static float pl_y(const Plot *p, double y) {
    double t = p->ly ? (log(y) - log(p->y0)) / (log(p->y1) - log(p->y0)) : (y - p->y0) / (p->y1 - p->y0);
    return p->r.y1 - (float)t * (p->r.y1 - p->r.y0);
}
static void pl_frame(const Plot *p, const char *xl, const char *yl) {
    const Rect *r = &p->r;
    rect_o(r->x0, r->y0, r->x1, r->y1, 0, 1, ca(pal->faint, 1));
    if (xl) text(st_tiny, r->x1, r->y1 + line_h(st_tiny) * 2.1f, 1, xl, ca(pal->dim, 0.9f));
    if (yl) text(st_tiny, r->x0, r->y0 - line_h(st_tiny) * 0.6f, -1, yl, ca(pal->dim, 0.9f));
}
static void pl_tick_x(const Plot *p, double x, const char *lab, int gridline) {
    float X = pl_x(p, x);
    if (X < p->r.x0 - 0.5f || X > p->r.x1 + 0.5f) return;
    if (gridline) line(X, p->r.y0, X, p->r.y1, 1, ca(pal->faint, 0.45f));
    line(X, p->r.y1, X, p->r.y1 + 5 * UI, 1, ca(pal->dim, 0.8f));
    if (lab) text(st_tiny, X, p->r.y1 + 6 * UI + cap_h(st_tiny) + 2, 0, lab, ca(pal->dim, 0.9f));
}
static void pl_tick_y(const Plot *p, double y, const char *lab, int gridline) {
    float Y = pl_y(p, y);
    if (Y < p->r.y0 - 0.5f || Y > p->r.y1 + 0.5f) return;
    if (gridline) line(p->r.x0, Y, p->r.x1, Y, 1, ca(pal->faint, 0.45f));
    line(p->r.x0 - 5 * UI, Y, p->r.x0, Y, 1, ca(pal->dim, 0.8f));
    if (lab) text(st_tiny, p->r.x0 - 8 * UI, Y + cap_h(st_tiny) * 0.5f, 1, lab, ca(pal->dim, 0.9f));
}
static void pl_curve(const Plot *p, const double *xs, const double *ys, int n, float w, const float *c, float dash) {
    float ph = 0;
    for (int i = 1; i < n; i++) {
        if (isnan(ys[i]) || isnan(ys[i - 1])) continue;
        float x0 = pl_x(p, xs[i - 1]), y0 = pl_y(p, ys[i - 1]), x1 = pl_x(p, xs[i]), y1 = pl_y(p, ys[i]);
        if (y0 < p->r.y0 - 0.5f || y1 < p->r.y0 - 0.5f || y0 > p->r.y1 + 0.5f || y1 > p->r.y1 + 0.5f) continue;
        line_dp(x0, y0, x1, y1, w, c, dash, -ph);
        ph += hypotf(x1 - x0, y1 - y0);
    }
}

/* ================================================================== */
/* ---- act 1: LEO to GEO, Hohmann then bi-elliptic ----               */
/* ================================================================== */
static struct {
    double r1, r2, rb, P1;
    double tb1, tb2, tb3, bb2, bb3;    /* burn times: Hohmann (tb1, tb2), bi-elliptic (tb1, bb2, bb3) */
    Orbit leo, geo, xf, geoH, b1, b2, geoB;
    double dv1, dv2, e1, e2, e3;
    Warp wa, wb;
    double yaw0;
    /* now */
    int ph;
    double t, sp[3], sv[3];
    Orbit cur;
} ho;

static Orbit circ(double mu, double r, double M0, double t0) { Orbit o = {mu, r, 0, 0, 0, 0, M0, t0}; return o; }

static void ho_init(uint32_t seed) {
    ho.r1 = R_EARTH + 300;
    ho.r2 = R_GEO;
    ho.rb = 125000;
    ho.yaw0 = (hrand(seed, 1) - 0.5) * 1.2;      /* added to the Sun's azimuth: a lit Earth */
    double mu = MU_EARTH, r1 = ho.r1, r2 = ho.r2, rb = ho.rb;
    ho.P1 = TAU_D * sqrt(r1 * r1 * r1 / mu);
    ho.tb1 = 0.62 * ho.P1;
    ho.leo = circ(mu, r1, -0.62 * TAU_D, 0);
    double at = (r1 + r2) / 2;
    ho.xf = (Orbit){mu, at, (r2 - r1) / (r2 + r1), 0, 0, 0, 0, ho.tb1};
    ho.tb2 = ho.tb1 + PI_D * sqrt(at * at * at / mu);
    ho.geoH = circ(mu, r2, PI_D, ho.tb2);
    ho.dv1 = sqrt(mu * (2 / r1 - 1 / at)) - sqrt(mu / r1);
    ho.dv2 = sqrt(mu / r2) - sqrt(mu * (2 / r2 - 1 / at));
    double a1 = (r1 + rb) / 2, a2 = (rb + r2) / 2;
    ho.b1 = (Orbit){mu, a1, (rb - r1) / (rb + r1), 0, 0, 0, 0, ho.tb1};
    ho.bb2 = ho.tb1 + PI_D * sqrt(a1 * a1 * a1 / mu);
    ho.b2 = (Orbit){mu, a2, (rb - r2) / (rb + r2), 0, 0, 0, PI_D, ho.bb2};
    ho.bb3 = ho.bb2 + PI_D * sqrt(a2 * a2 * a2 / mu);
    ho.geoB = circ(mu, r2, 0, ho.bb3);
    ho.e1 = sqrt(mu * (2 / r1 - 1 / a1)) - sqrt(mu / r1);
    ho.e2 = sqrt(mu * (2 / rb - 1 / a2)) - sqrt(mu * (2 / rb - 1 / a1));
    ho.e3 = sqrt(mu * (2 / r2 - 1 / a2)) - sqrt(mu / r2);
    Warp wa = {7, {0, 6.5, 9.5, 27, 31, 40, 44}, {0, ho.tb1 - 500, ho.tb1 + 500, ho.tb2 - 2500, ho.tb2 + 1500, ho.tb2 + 9000, ho.tb2 + 11000}};
    Warp wb = {9, {0, 6.0, 9.0, 18, 22.5, 33, 37.5, 42, 45},
               {0, ho.tb1 - 500, ho.tb1 + 600, ho.bb2 - 9000, ho.bb2 + 9000, ho.bb3 - 5000, ho.bb3 + 3000, ho.bb3 + 12000, ho.bb3 + 14000}};
    ho.wa = wa;
    ho.wb = wb;
}

static void ho_update(double u, Info *in) {
    double rate;
    ho.ph = u >= 45;
    ho.t = ho.ph ? warp_at(&ho.wb, u - 45, &rate) : warp_at(&ho.wa, u, &rate);
    const Orbit *o;
    if (ho.t < ho.tb1) o = &ho.leo;
    else if (!ho.ph) o = ho.t < ho.tb2 ? &ho.xf : &ho.geoH;
    else o = ho.t < ho.bb2 ? &ho.b1 : ho.t < ho.bb3 ? &ho.b2 : &ho.geoB;
    ho.cur = *o;
    orb_rv(o, ho.t, ho.sp, ho.sv);
    in->title = ho.ph ? "Bi-elliptic transfer" : "Hohmann transfer";
    in->sub = ho.ph ? "LEO \xe2\x86\x92 GEO through a far apogee: three burns" : "LEO 300 km \xe2\x86\x92 geostationary orbit, two tangential burns";
    in->frame = "GEOCENTRIC EQUATORIAL";
    in->eq[0] = "v = \xe2\x88\x9a(\xce\xbc (2/r \xe2\x88\x92 1/a))";
    in->eq[1] = "a = (r\xe2\x82\x81 + r\xe2\x82\x82) / 2  (transfer)";
    in->eq[2] = "\xce\x94v\xe2\x82\x81 = \xe2\x88\x9a(\xce\xbc/r\xe2\x82\x81) (\xe2\x88\x9a(2r\xe2\x82\x82/(r\xe2\x82\x81+r\xe2\x82\x82)) \xe2\x88\x92 1)";
    in->eq[3] = "t = \xcf\x80 \xe2\x88\x9a(a\xc2\xb3/\xce\xbc)  (half an orbit)";
    in->neq = 4;
    in->warp = rate;
    fmt_met(in->clock, sizeof in->clock, ho.t);
    double r = norm3(ho.sp), v = norm3(ho.sv);
    char b[32];
    fmt_k(b, sizeof b, r - R_EARTH);
    tel(in, "ALTITUDE", "%s km", b);
    tel(in, "VELOCITY", "%.3f km/s", v);
    fmt_k(b, sizeof b, ho.cur.a);
    tel(in, "SEMI-MAJOR AXIS", "%s km", b);
    tel(in, "ECCENTRICITY", "%.4f", ho.cur.e);
    char d[32];
    fmt_dur(d, sizeof d, orb_period(&ho.cur));
    tel(in, "PERIOD", "%s", d);
    tel(in, "TRUE ANOMALY", "%.1f\xc2\xb0", fmod(orb_nu_at(&ho.cur, ho.t) / RAD + 360, 360));
    tel(in, "FLIGHT PATH \xce\xb3", "%.2f\xc2\xb0", asin(dot3(ho.sp, ho.sv) / (r * v)) / RAD);
    in->dv_head = "\xce\x94v BUDGET  km/s";
    if (!ho.ph) {
        dvrow(in, "\xce\x94v\xe2\x82\x81  perigee burn, LEO", ho.dv1, ho.t >= ho.tb1);
        dvrow(in, "\xce\x94v\xe2\x82\x82  apogee burn, circularise", ho.dv2, ho.t >= ho.tb2);
        dvrow(in, "TOTAL  Hohmann", ho.dv1 + ho.dv2, 2);
        fmt_dur(d, sizeof d, ho.tb2 - ho.tb1);
        dvrow_s(in, "transfer time", d, 3);
    } else {
        dvrow(in, "\xce\x94v\xe2\x82\x81  perigee burn, LEO", ho.e1, ho.t >= ho.tb1);
        dvrow(in, "\xce\x94v\xe2\x82\x82  raise perigee at r\xe2\x82\x90", ho.e2, ho.t >= ho.bb2);
        dvrow(in, "\xce\x94v\xe2\x82\x83  circularise at GEO", -ho.e3, ho.t >= ho.bb3);
        dvrow(in, "TOTAL  bi-elliptic", ho.e1 + ho.e2 - ho.e3, 2);
        dvrow(in, "vs Hohmann", ho.e1 + ho.e2 - ho.e3 - ho.dv1 - ho.dv2, 3);
        fmt_dur(d, sizeof d, ho.bb3 - ho.tb1);
        dvrow_s(in, "transfer time", d, 3);
    }
}

static double fit_scale(const View *v, double rad, double el, double margin) {
    double w = (v->x1 - v->x0) * 0.5 - margin, h = (v->y1 - v->y0) * 0.5 - margin;
    return fmin(w / rad, h / (rad * fmax(sin(el), 0.25)));
}

static void ho_draw(Rect R, double u) {
    View v;
    view_init(&v, R.x0, R.y0, R.x1, R.y1);
    double el = lerpd(50, 60, smooth(45, 52, u)) * RAD;
    double jd0 = jd_of(act_wt0), sun0[3];
    sun_dir_eq(jd0, sun0);
    view_cam(&v, atan2(sun0[1], sun0[0]) + 0.7 + ho.yaw0 + u * 0.0045, el);
    double s0 = fit_scale(&v, ho.r2 * 1.12, el, 30 * UI), s1 = fit_scale(&v, ho.rb * 1.05, el, 30 * UI);
    v.s = exp(lerpd(log(s0), log(s1), smooth(46, 56, u)));
    double O[3] = {0, 0, 0};
    view_occ(&v, O, R_EARTH);
    double jd = jd_of(act_wt0) + ho.t / DAY, sun[3];
    sun_dir_eq(jd, sun);
    double step = ho.ph && u > 50 ? 25000 : 10000;
    range_rings(&v, O, ho.ph && u > 50 ? ho.rb * 1.05 : ho.r2 * 1.12, step, 1, "km", 0.9f);
    planet(&v, O, R_EARTH, B_EARTH, sun, (float)gmst(jd), 4, pal->sea);
    /* the orbits */
    draw_circle(&v, O, ho.r1, 1.2f * UI, ca(pal->orb2, 0.8f), 0);
    draw_circle(&v, O, ho.r2, 1.2f * UI, ca(pal->orb2, 0.8f), 0);
    double t = ho.t;
    const float *oc = pal->orb;
    if (!ho.ph) {
        if (t < ho.tb1) draw_conic(&v, &ho.xf, 0, PI_D, 1.4f * UI, ca(oc, 0.8f), 9 * UI, 0);
        else if (t < ho.tb2) {
            double nu = orb_nu_at(&ho.xf, t);
            if (nu < 0) nu += TAU_D;
            draw_conic(&v, &ho.xf, 0, nu, 2 * UI, ca(oc, 1), 0, 4);
            draw_conic(&v, &ho.xf, nu, PI_D, 1.4f * UI, ca(oc, 0.8f), 9 * UI, 0);
        } else draw_conic(&v, &ho.xf, 0, PI_D, 2 * UI, ca(oc, 0.85f), 0, 3);
        draw_conic(&v, &ho.xf, PI_D, TAU_D, 1 * UI, ca(oc, 0.18f), 6 * UI, 0);
    } else {
        draw_conic(&v, &ho.xf, 0, PI_D, 1 * UI, ca(pal->dim, 0.5f), 7 * UI, 0);
        const Orbit *legs[2] = {&ho.b1, &ho.b2};
        double ts[3] = {ho.tb1, ho.bb2, ho.bb3}, nus[2][2] = {{0, PI_D}, {PI_D, TAU_D}};
        for (int k = 0; k < 2; k++) {
            if (t < ts[k]) draw_conic(&v, legs[k], nus[k][0], nus[k][1], 1.4f * UI, ca(oc, 0.8f), 9 * UI, 0);
            else if (t < ts[k + 1]) {
                double nu = orb_nu_at(legs[k], t);
                while (nu < nus[k][0]) nu += TAU_D;
                draw_conic(&v, legs[k], nus[k][0], nu, 2 * UI, ca(oc, 1), 0, 4);
                draw_conic(&v, legs[k], nu, nus[k][1], 1.4f * UI, ca(oc, 0.8f), 9 * UI, 0);
            } else draw_conic(&v, legs[k], nus[k][0], nus[k][1], 2 * UI, ca(oc, 0.85f), 0, 3);
        }
    }
    /* labels on the circles */
    {
        double p[3] = {ho.r2 * cos(2.4), ho.r2 * sin(2.4), 0};
        float x, y;
        proj(&v, p, &x, &y);
        text_h(st_lab, x, y - 8 * UI, 0, "GEO  r = 42 164 km", ca(pal->orb2, 1));
        double q[3] = {ho.r1 * cos(-2.2), ho.r1 * sin(-2.2), 0};
        if (!hidden(&v, q)) { proj(&v, q, &x, &y); text_h(st_lab, x, y + 18 * UI, 0, "LEO 300 km", ca(pal->orb2, 1)); }
        if (ho.ph && u > 50) {
            double w[3] = {-ho.rb, 0, 0};
            proj(&v, w, &x, &y);
            text_h(st_lab, x, y - 14 * UI, 0, "apogee 125 000 km", ca(pal->ink, 0.85f));
        }
    }
    /* the burns */
    double tg[3] = {0, 1, 0}, tm[3] = {0, -1, 0};
    double p1[3] = {ho.r1, 0, 0};
    double fl;
    if (!ho.ph) {
        fl = fmax(0, 1 - fabs(t - ho.tb1) / 900);
        burn_mark(&v, p1, tg, "\xce\x94v\xe2\x82\x81", ho.dv1, ho.tb1 - t, fl, O);
        double p2[3] = {-ho.r2, 0, 0};
        fl = fmax(0, 1 - fabs(t - ho.tb2) / 2500);
        burn_mark(&v, p2, tm, "\xce\x94v\xe2\x82\x82", ho.dv2, ho.tb2 - t, fl, O);
    } else {
        fl = fmax(0, 1 - fabs(t - ho.tb1) / 900);
        burn_mark(&v, p1, tg, "\xce\x94v\xe2\x82\x81", ho.e1, ho.tb1 - t, fl, O);
        double p2[3] = {-ho.rb, 0, 0};
        fl = fmax(0, 1 - fabs(t - ho.bb2) / 9000);
        burn_mark(&v, p2, tm, "\xce\x94v\xe2\x82\x82", ho.e2, ho.bb2 - t, fl, O);
        double p3[3] = {ho.r2, 0, 0};
        fl = fmax(0, 1 - fabs(t - ho.bb3) / 4000);
        double rev[3] = {0, -1, 0};
        burn_mark(&v, p3, rev, "\xce\x94v\xe2\x82\x83", -ho.e3, ho.bb3 - t, fl, O);
    }
    craft(&v, ho.sp, ho.sv, pal->orb, NULL);
}

/* the panel's plot: total Δv against the radius ratio, Hohmann vs
 * bi-elliptic (r_b -> infinity and r_b = 125 000 km's ratio) */
static double hoh_dv(double R) { return sqrt(2 * R / (1 + R)) - 1 + sqrt(1 / R) * (1 - sqrt(2 / (1 + R))); }
static double bie_dv(double R, double Rb) {
    return sqrt(2 * Rb / (1 + Rb)) - 1 + sqrt(2 / Rb) * (sqrt(R / (R + Rb)) - sqrt(1 / (1 + Rb))) + sqrt(1 / R) * (sqrt(2 * Rb / (R + Rb)) - 1);
}
static void ho_plot(Rect r, double u) {
    (void)u;
    Plot p = {r, 1, 100, 0.0, 0.62, 1, 0};
    pl_frame(&p, "r\xe2\x82\x82 / r\xe2\x82\x81", "\xce\x94v / v\xe2\x82\x81");
    char b[16];
    double xt[] = {1, 2, 5, 10, 20, 50, 100};
    for (int i = 0; i < 7; i++) { snprintf(b, sizeof b, "%g", xt[i]); pl_tick_x(&p, xt[i], b, 1); }
    for (int i = 0; i <= 6; i++) { snprintf(b, sizeof b, "%.1f", i * 0.1); pl_tick_y(&p, i * 0.1, b, 1); }
    double xs[200], y1[200], y2[200], y3[200];
    for (int i = 0; i < 200; i++) {
        xs[i] = exp(log(100) * i / 199);
        y1[i] = hoh_dv(xs[i]);
        y2[i] = (sqrt(2) - 1) * (1 + 1 / sqrt(xs[i]));
        double Rb = ho.rb / ho.r1;
        y3[i] = xs[i] <= Rb ? bie_dv(xs[i], Rb) : NAN;
    }
    pl_curve(&p, xs, y2, 200, 1.4f * UI, ca(pal->alt, 0.9f), 6 * UI);
    pl_curve(&p, xs, y3, 200, 1.4f * UI, ca(pal->alt, 0.9f), 0);
    pl_curve(&p, xs, y1, 200, 2 * UI, ca(pal->orb, 1), 0);
    float X = pl_x(&p, 11.94);
    line_d(X, r.y0, X, r.y1, 1, ca(pal->hi, 0.8f), 5 * UI);
    text(st_tiny, X + 4 * UI, r.y0 + line_h(st_tiny) * 1.2f, -1, "11.94: bi-elliptic wins (r\xe2\x82\x90\xe2\x86\x92\xe2\x88\x9e)", ca(pal->hi, 0.95f));
    double R = ho.r2 / ho.r1;
    float px = pl_x(&p, R), py = pl_y(&p, hoh_dv(R));
    disc(px, py, 4 * UI, ca(pal->orb, 1));
    glow(px, py, 18 * UI, cadd(pal->orb, 0.5f));
    snprintf(b, sizeof b, "LEO\xe2\x86\x92GEO  %.2f", R);
    text_h(st_tiny, px + 8 * UI, py + 14 * UI, -1, b, ca(pal->ink, 1));
    float ly = r.y1 - line_h(st_tiny) * 3.2f;
    line(r.x1 - 150 * UI, ly, r.x1 - 120 * UI, ly, 2 * UI, ca(pal->orb, 1));
    text(st_tiny, r.x1 - 114 * UI, ly + 4 * UI, -1, "Hohmann", ca(pal->ink, 0.9f));
    ly += line_h(st_tiny) * 1.1f;
    line(r.x1 - 150 * UI, ly, r.x1 - 120 * UI, ly, 1.4f * UI, ca(pal->alt, 1));
    text(st_tiny, r.x1 - 114 * UI, ly + 4 * UI, -1, "bi-elliptic", ca(pal->ink, 0.9f));
}

/* ================================================================== */
/* ---- act 2: Earth to Mars, a porkchop plot from Lambert's problem ---- */
/* ================================================================== */
#define PCX 96
#define PCY 96
static struct {
    double jd0, dep0, dep1, arr0, arr1;
    float grid[PCY][PCX][2];           /* C3 (km^2/s^2), arrival v-infinity (km/s); -1 no transfer; -2 not yet computed */
    int done, dirty;
    double er[PCX][3], ev[PCX][3], mr[PCY][3], mv[PCY][3];
    int bi, bj, mi, mj;
    double bc3, bvi, bdv1, bdv2, mc3;
    double tdep, tarr;                 /* jd */
    Orbit tr;
    double nu0, nu1, yaw0;
    Warp w;
    /* now */
    double jd, met, sp[3], sv[3];
    int flying;
} mx;

static void mx_cell(int k) {
    int i = k / PCY, j = k % PCY;
    double td = mx.dep0 + (mx.dep1 - mx.dep0) * i / (PCX - 1), ta = mx.arr0 + (mx.arr1 - mx.arr0) * j / (PCY - 1);
    double tof = ta - td, v1[3], v2[3];
    mx.grid[j][i][0] = mx.grid[j][i][1] = -1;
    if (tof < 15 || !lambert(MU_SUN, mx.er[i], mx.mr[j], tof * DAY, 1, v1, v2)) { mx.grid[j][i][0] = 999; mx.grid[j][i][1] = 99; return; }
    double a[3], b[3];
    sub3(v1, mx.ev[i], a);
    sub3(v2, mx.mv[j], b);
    mx.grid[j][i][0] = (float)dot3(a, a);
    mx.grid[j][i][1] = (float)norm3(b);
}
static double dv_dep(double c3) { double r = R_EARTH + 300; return sqrt(c3 + 2 * MU_EARTH / r) - sqrt(MU_EARTH / r); }
static double dv_arr(double vi) { double r = R_MARS + 400; return sqrt(vi * vi + 2 * MU_MARS / r) - sqrt(MU_MARS / r); }

static void mx_finish(void) {
    double best = 1e9, mc = 1e9;
    for (int i = 0; i < PCX; i++)
        for (int j = 0; j < PCY; j++) {
            float c3 = mx.grid[j][i][0], vi = mx.grid[j][i][1];
            if (c3 < 0 || c3 > 900) continue;
            double d = dv_dep(c3) + dv_arr(vi);
            if (d < best) { best = d; mx.bi = i; mx.bj = j; }
            if (c3 < mc) { mc = c3; mx.mi = i; mx.mj = j; }
        }
    mx.mc3 = mc;
    int i = mx.bi, j = mx.bj;
    mx.bc3 = mx.grid[j][i][0];
    mx.bvi = mx.grid[j][i][1];
    mx.bdv1 = dv_dep(mx.bc3);
    mx.bdv2 = dv_arr(mx.bvi);
    mx.tdep = mx.dep0 + (mx.dep1 - mx.dep0) * i / (PCX - 1);
    mx.tarr = mx.arr0 + (mx.arr1 - mx.arr0) * j / (PCY - 1);
    double v1[3], v2[3];
    lambert(MU_SUN, mx.er[i], mx.mr[j], (mx.tarr - mx.tdep) * DAY, 1, v1, v2);
    orb_from_rv(MU_SUN, mx.er[i], v1, 0, &mx.tr);
    mx.nu0 = orb_nu_at(&mx.tr, 0);
    mx.nu1 = orb_nu_at(&mx.tr, (mx.tarr - mx.tdep) * DAY);
    if (mx.nu1 < mx.nu0) mx.nu1 += TAU_D;
    double tof = mx.tarr - mx.tdep;
    Warp w = {7, {30, 36, 40, 77, 81, 87, 90}, {-30, -2, 3, tof - 6, tof + 2, tof + 18, tof + 20}};
    mx.w = w;
}

/* compute cells up to n (all of them in a --shot) */
static void mx_compute(int n) {
    if (n > PCX * PCY) n = PCX * PCY;
    if (mx.done >= n) return;
    while (mx.done < n) mx_cell(mx.done++);
    mx.dirty = 1;
    if (mx.done == PCX * PCY) mx_finish();
}

static void mx_init(uint32_t seed, double wt) {
    mx.yaw0 = hrand(seed, 2) * TAU_D;
    mx.jd0 = floor(jd_of(wt) - 0.5) + 0.5;
    /* the next window: the first dip in the best C3 over departure dates */
    double prev = 1e9, prev2 = 1e9, dmin = 0;
    for (double d = -90; d < 900; d += 8) {
        double best = 1e9, re[3], ve[3], rm[3], vm[3], v1[3], v2[3];
        planet_rv(P_EARTH, mx.jd0 + d, re, ve);
        for (double tof = 120; tof <= 420; tof += 15) {
            planet_rv(P_MARS, mx.jd0 + d + tof, rm, vm);
            if (!lambert(MU_SUN, re, rm, tof * DAY, 1, v1, v2)) continue;
            double a[3];
            sub3(v1, ve, a);
            best = fmin(best, dot3(a, a));
        }
        if (prev < prev2 && prev <= best && prev < 40) { dmin = d - 8; break; }
        prev2 = prev;
        prev = best;
    }
    {
        double re[3], ve[3];
        planet_rv(P_EARTH, mx.jd0 + dmin, re, ve);
        mx.yaw0 = atan2(re[1], re[0]) - 135 * RAD + (hrand(seed, 2) - 0.5) * 0.3;
    }
    mx.dep0 = mx.jd0 + dmin - 75;
    mx.dep1 = mx.jd0 + dmin + 75;
    mx.arr0 = mx.dep0 + 110;
    mx.arr1 = mx.dep1 + 420;
    for (int i = 0; i < PCX; i++) planet_rv(P_EARTH, mx.dep0 + (mx.dep1 - mx.dep0) * i / (PCX - 1), mx.er[i], mx.ev[i]);
    for (int j = 0; j < PCY; j++) planet_rv(P_MARS, mx.arr0 + (mx.arr1 - mx.arr0) * j / (PCY - 1), mx.mr[j], mx.mv[j]);
    for (int i = 0; i < PCX; i++) for (int j = 0; j < PCY; j++) mx.grid[j][i][0] = mx.grid[j][i][1] = -2;
    mx.done = 0;
    mx.dirty = 1;
}

static double mx_reveal(double u) { return smooth(0, 1, (u - 2) / 24) * 0.0 + clampd((u - 2) / 24, 0, 1); }

static void mx_update(double u, Info *in, int shot) {
    int need = (int)ceil(mx_reveal(u) * PCX * PCY);
    if (shot || u > 27) need = PCX * PCY;
    else need = need + 200;
    mx_compute(need);
    double rate = 0;
    mx.flying = u >= 30 && mx.done == PCX * PCY;
    if (mx.flying) {
        mx.met = warp_at(&mx.w, u, &rate);
        mx.jd = mx.tdep + mx.met;
        rate *= DAY;
    } else {
        mx.met = -1e9;
        mx.jd = mx.jd0 + u * 0.4;
        rate = 0.4 * 86400;
    }
    double tof = mx.tarr - mx.tdep;
    double pe[3], ve[3], pm[3], vm[3];
    planet_rv(P_EARTH, mx.jd, pe, ve);
    planet_rv(P_MARS, mx.jd, pm, vm);
    if (!mx.flying || mx.met < 0) { memcpy(mx.sp, pe, sizeof pe); memcpy(mx.sv, ve, sizeof ve); }
    else if (mx.met > tof) { memcpy(mx.sp, pm, sizeof pm); memcpy(mx.sv, vm, sizeof vm); }
    else orb_rv(&mx.tr, mx.met * DAY, mx.sp, mx.sv);
    in->title = "Earth \xe2\x86\x92 Mars";
    in->sub = mx.flying ? "Lambert transfer flown on the real ephemeris" : "searching the launch window: Lambert's problem, cell by cell";
    in->frame = "HELIOCENTRIC ECLIPTIC J2000";
    in->eq[0] = "(r\xe2\x82\x81, r\xe2\x82\x82, \xce\x94t) \xe2\x86\x92 v\xe2\x82\x81, v\xe2\x82\x82   (Lambert)";
    in->eq[1] = "\xe2\x88\x9a\xce\xbc \xce\x94t = (y/C)^1.5 S(z) + A \xe2\x88\x9ay";
    in->eq[2] = "C\xe2\x82\x83 = v\xe2\x88\x9e\xc2\xb2 = |v\xe2\x82\x81 \xe2\x88\x92 V(Earth)|\xc2\xb2";
    in->eq[3] = "Hohmann: \xcf\x80\xe2\x88\x9a(a\xc2\xb3/\xce\xbc) = 259 d";
    in->neq = 4;
    in->warp = rate;
    char b[48], d2[24];
    fmt_date(b, sizeof b, mx.jd);
    if (mx.flying) {
        char m[32];
        fmt_met(m, sizeof m, mx.met * DAY);
        snprintf(in->clock, sizeof in->clock, "%s  %s", b, m);
    } else snprintf(in->clock, sizeof in->clock, "%s UTC", b);
    double rs = norm3(mx.sp), de[3], dm[3];
    sub3(mx.sp, pe, de);
    sub3(mx.sp, pm, dm);
    tel(in, "DATE", "%s", b);
    tel(in, "SUN DISTANCE", "%.4f AU", rs / AU);
    tel(in, "HELIO VELOCITY", "%.3f km/s", norm3(mx.sv));
    tel(in, "EARTH RANGE", "%.2f Gm", norm3(de) / 1e6);
    tel(in, "MARS RANGE", "%.2f Gm", norm3(dm) / 1e6);
    tel(in, "LIGHT TIME", "%.1f min", norm3(de) / 299792.458 / 60);
    tel(in, "LAMBERT SOLVES", "%d / %d", mx.done, PCX * PCY);
    in->dv_head = "\xce\x94v BUDGET  km/s";
    if (mx.done == PCX * PCY) {
        fmt_date(b, sizeof b, mx.tdep);
        snprintf(d2, sizeof d2, "%s", b);
        char nm[48];
        snprintf(nm, sizeof nm, "TMI  C\xe2\x82\x83 %.1f km\xc2\xb2/s\xc2\xb2", mx.bc3);
        dvrow(in, nm, mx.bdv1, mx.flying && mx.met >= 0);
        snprintf(nm, sizeof nm, "MOI  v\xe2\x88\x9e %.2f km/s", mx.bvi);
        dvrow(in, nm, mx.bdv2, mx.flying && mx.met >= tof);
        dvrow(in, "TOTAL", mx.bdv1 + mx.bdv2, 2);
        dvrow_s(in, "launch", d2, 3);
        fmt_date(b, sizeof b, mx.tarr);
        dvrow_s(in, "arrival", b, 3);
        snprintf(b, sizeof b, "%.0f d", tof);
        dvrow_s(in, "time of flight", b, 3);
    } else {
        dvrow_s(in, "TMI  from LEO 300 km", "\xe2\x80\x94", 0);
        dvrow_s(in, "MOI  to 400 km circular", "\xe2\x80\x94", 0);
        snprintf(b, sizeof b, "%d%%", mx.done * 100 / (PCX * PCY));
        dvrow_s(in, "porkchop", b, 3);
    }
}

static void mx_draw(Rect R, double u) {
    View v;
    view_init(&v, R.x0, R.y0, R.x1, R.y1);
    double el = lerpd(68, 52, smooth(30, 60, u)) * RAD;
    view_cam(&v, mx.yaw0 + u * 0.004, el);
    v.s = fit_scale(&v, 1.72 * AU, el, 24 * UI);
    double O[3] = {0, 0, 0}, sunv[3] = {0, 0, 1};
    range_rings(&v, O, 1.75 * AU, 0.25 * AU, 1 / AU, "AU", 0.8f);
    Orbit oe, om;
    planet_orbit(P_EARTH, mx.jd, &oe);
    planet_orbit(P_MARS, mx.jd, &om);
    draw_conic(&v, &oe, 0, TAU_D, 1.3f * UI, ca(pal->orb2, 0.85f), 0, 0);
    draw_conic(&v, &om, 0, TAU_D, 1.3f * UI, ca(pal->orb2, 0.85f), 0, 0);
    planet(&v, O, 696000, B_SUN, sunv, 0, 7 * UI, pal->sun);
    double pe[3], pm[3], tmp[3];
    planet_rv(P_EARTH, mx.jd, pe, tmp);
    planet_rv(P_MARS, mx.jd, pm, tmp);
    /* the search: a fan of trial Lambert arcs from the column being solved */
    if (!mx.flying && mx.done > 0) {
        int col = (int)(mx_reveal(u) * PCX);
        if (col >= PCX) col = PCX - 1;
        for (int j = 4; j < PCY; j += 9) {
            float c3 = mx.grid[j][col][0];
            if (c3 < 0 || c3 > 60) continue;
            double v1[3], v2[3], ta = mx.arr0 + (mx.arr1 - mx.arr0) * j / (PCY - 1);
            double td = mx.dep0 + (mx.dep1 - mx.dep0) * col / (PCX - 1);
            if (!lambert(MU_SUN, mx.er[col], mx.mr[j], (ta - td) * DAY, 1, v1, v2)) continue;
            Orbit o;
            orb_from_rv(MU_SUN, mx.er[col], v1, 0, &o);
            double n0 = orb_nu_at(&o, 0), n1 = orb_nu_at(&o, (ta - td) * DAY);
            if (n1 < n0) n1 += TAU_D;
            float q = (float)clampd(1 - (c3 - 8) / 50, 0.15, 1);
            draw_conic(&v, &o, n0, n1, 1 * UI, ca(q > 0.8f ? pal->hi : pal->orb, 0.25f + 0.5f * q), 0, 0);
        }
        float x, y;
        proj(&v, mx.er[col], &x, &y);
        ring(x, y, 9 * UI, 1, ca(pal->hi, 0.8f));
    }
    if (mx.done == PCX * PCY) {
        float a = (float)smooth(26, 30, u);
        double tof = mx.tarr - mx.tdep, nu = mx.nu0;
        if (mx.flying && mx.met > 0) nu = mx.met >= tof ? mx.nu1 : orb_nu_at(&mx.tr, mx.met * DAY);
        if (nu < mx.nu0) nu += TAU_D;
        if (nu > mx.nu0) draw_conic(&v, &mx.tr, mx.nu0, nu, 2 * UI, ca(pal->orb, a), 0, 4);
        if (nu < mx.nu1) draw_conic(&v, &mx.tr, nu, mx.nu1, 1.5f * UI, ca(pal->orb, a * 0.9f), 9 * UI, 0);
        /* where Mars is at launch, and Earth at arrival */
        double gm[3], ge[3], vv[3];
        planet_rv(P_MARS, mx.tdep, gm, vv);
        planet_rv(P_EARTH, mx.tarr, ge, vv);
        float x, y;
        proj(&v, gm, &x, &y);
        ring(x, y, 5 * UI, 1, ca(pal->mars, 0.6f * a));
        text_h(st_tiny, x + 8 * UI, y + 4 * UI, -1, "MARS AT LAUNCH", ca(pal->dim, 0.9f * a));
        proj(&v, ge, &x, &y);
        ring(x, y, 5 * UI, 1, ca(pal->orb2, 0.6f * a));
        text_h(st_tiny, x + 8 * UI, y + 4 * UI, -1, "EARTH AT ARRIVAL", ca(pal->dim, 0.9f * a));
        /* the burns */
        double r0[3], v0[3], r1[3], v1[3];
        orb_rv(&mx.tr, 0, r0, v0);
        orb_rv(&mx.tr, tof * DAY, r1, v1);
        if (a > 0) {
            alpha_mul = act_alpha * a;
            double fl = mx.flying ? fmax(0, 1 - fabs(mx.met) / 3) : 0;
            burn_mark(&v, r0, v0, "TMI", mx.bdv1, mx.flying ? -mx.met * DAY : 1e9, fl, O);
            fl = mx.flying ? fmax(0, 1 - fabs(mx.met - tof) / 3) : 0;
            double rv[3];
            scale3(v1, -1, rv);
            burn_mark(&v, r1, rv, "MOI", mx.bdv2, mx.flying ? (tof - mx.met) * DAY : 1e9, fl, O);
            alpha_mul = act_alpha;
        }
    }
    double sd[3];
    scale3(pe, -1, sd);
    planet(&v, pe, R_EARTH, B_EARTH, sd, 0, 5.5f * UI, pal->sea);
    scale3(pm, -1, sd);
    planet(&v, pm, R_MARS, B_MARS, sd, 0, 4.5f * UI, pal->mars);
    float x, y;
    proj(&v, pe, &x, &y);
    text_h(st_lab, x - 12 * UI, y - 12 * UI, 1, "EARTH", ca(pal->ink, 0.9f));
    proj(&v, pm, &x, &y);
    text_h(st_lab, x + 12 * UI, y - 12 * UI, -1, "MARS", ca(pal->ink, 0.9f));
    if (mx.flying && mx.met > 0 && mx.met < mx.tarr - mx.tdep) craft(&v, mx.sp, mx.sv, pal->orb, NULL);
}

static GLuint tex_pork;
static void mx_plot(Rect r, double u) {
    if (mx.dirty) {
        glBindTexture(GL_TEXTURE_2D, tex_pork);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, PCX, PCY, 0, GL_RG, GL_FLOAT, mx.grid);
        mx.dirty = 0;
    }
    Plot p = {r, mx.dep0, mx.dep1, mx.arr0, mx.arr1, 0, 0};
    float rev = (float)(mx.done >= PCX * PCY ? 1.0 : fmin(mx_reveal(u), (double)mx.done / (PCX * PCY)));
    float pp[4] = {(r.x1 - r.x0) * 0.5f, (r.y1 - r.y0) * 0.5f, (float)mx.mc3, 0};
    inst((r.x0 + r.x1) * 0.5f, (r.y0 + r.y1) * 0.5f, pp[0], pp[1], 0, K_PORK, pp, ca(pal->orb, 1), rev, 0);
    /* the reveal sweep */
    if (rev < 1) {
        float X = r.x0 + rev * (r.x1 - r.x0);
        line(X, r.y0, X, r.y1, 1.5f * UI, ca(pal->hi, 0.9f));
        glowline(X, r.y0, X, r.y1, 12 * UI, cadd(pal->hi, 0.3f));
    }
    pl_frame(&p, "DEPARTURE (EARTH)", "ARRIVAL (MARS)");
    /* month ticks */
    char b[24];
    for (int axis = 0; axis < 2; axis++) {
        double a0 = axis ? mx.arr0 : mx.dep0, a1 = axis ? mx.arr1 : mx.dep1;
        time_t t = (time_t)((a0 - 2440587.5) * 86400);
        struct tm tm;
        gmtime_r(&t, &tm);
        tm.tm_mday = 1; tm.tm_hour = 0; tm.tm_min = 0; tm.tm_sec = 0;
        int every = axis ? 2 : 1;
        for (int m = 0; m < 30; m++) {
            struct tm q = tm;
            q.tm_mon += m + 1;
            time_t tt = timegm(&q);
            double jd = jd_of((double)tt);
            if (jd > a1) break;
            gmtime_r(&tt, &q);
            if (q.tm_mon % every) continue;
            if (q.tm_mon == 0) strftime(b, sizeof b, "%b '%y", &q);
            else strftime(b, sizeof b, "%b", &q);
            for (char *c = b; *c; c++) *c = (char)toupper((unsigned char)*c);
            if (axis) pl_tick_y(&p, jd, b, 0);
            else pl_tick_x(&p, jd, b, 0);
        }
    }
    /* lines of constant time of flight */
    for (int tof = 100; tof <= 500; tof += 100) {
        double x0 = fmax(mx.dep0, mx.arr0 - tof), x1 = fmin(mx.dep1, mx.arr1 - tof);
        if (x1 <= x0) continue;
        line_d(pl_x(&p, x0), pl_y(&p, x0 + tof), pl_x(&p, x1), pl_y(&p, x1 + tof), 1, ca(pal->dim, 0.55f), 4 * UI);
        snprintf(b, sizeof b, "%d d", tof);
        text_h(st_tiny, pl_x(&p, x1) - 4 * UI, pl_y(&p, x1 + tof) + 12 * UI, 1, b, ca(pal->dim, 0.85f));
    }
    if (mx.done == PCX * PCY) {
        float a = (float)smooth(24, 28, u);
        double xd = mx.dep0 + (mx.dep1 - mx.dep0) * mx.mi / (PCX - 1), ya = mx.arr0 + (mx.arr1 - mx.arr0) * mx.mj / (PCY - 1);
        float X = pl_x(&p, xd), Y = pl_y(&p, ya);
        ring(X, Y, 5 * UI, 1, ca(pal->ink, 0.8f * a));
        snprintf(b, sizeof b, "C\xe2\x82\x83 min %.2f", mx.mc3);
        text_h(st_tiny, X - 9 * UI, Y - 8 * UI, 1, b, ca(pal->ink, 0.9f * a));
        X = pl_x(&p, mx.tdep); Y = pl_y(&p, mx.tarr);
        line(X - 14 * UI, Y, X - 5 * UI, Y, 1.4f * UI, ca(pal->hi, a));
        line(X + 5 * UI, Y, X + 14 * UI, Y, 1.4f * UI, ca(pal->hi, a));
        line(X, Y - 14 * UI, X, Y - 5 * UI, 1.4f * UI, ca(pal->hi, a));
        line(X, Y + 5 * UI, X, Y + 14 * UI, 1.4f * UI, ca(pal->hi, a));
        glow(X, Y, 26 * UI, cadd(pal->hi, 0.4f * a));
        snprintf(b, sizeof b, "\xce\x94v %.2f km/s  %.0f d", mx.bdv1 + mx.bdv2, mx.tarr - mx.tdep);
        text_h(st_tiny, X + 16 * UI, Y + 16 * UI, -1, b, ca(pal->hi, a));
        if (mx.flying) {
            double t = mx.tdep + clampd(mx.met, 0, mx.tarr - mx.tdep);
            (void)t;
        }
    }
    /* legend */
    text(st_tiny, r.x1, r.y0 - line_h(st_tiny) * 0.6f, 1, "C\xe2\x82\x83 contours  \xc2\xb7  v\xe2\x88\x9e arrival dashed", ca(pal->dim, 0.95f));
}
/* ================================================================== */
/* ---- act 3: a gravity assist at Jupiter ----                        */
/* ================================================================== */
static struct {
    double vinf, rp, e, a, delta, b, n, p;
    double uin[2], uout[2], ph[2];
    int sense, trailing;
    double Vp[2], vin[2], vout[2], dv, tmax, rJ, ra;
    Orbit hin, hout;
    double nuenc_in, nuenc_out, nuout_end;
    Warp w;
    double yaw0;
    /* now */
    double t, pos[3], vel[3];
} fb;

static void fb_state(double t, double *p, double *v) {
    double H = kep_H(fb.n * t, fb.e), e = fb.e;
    double nu = 2 * atan(sqrt((e + 1) / (e - 1)) * tanh(H / 2));
    double r = fb.p / (1 + e * cos(nu)), th = atan2(fb.ph[1], fb.ph[0]) + fb.sense * nu;
    double k = sqrt(MU_JUP / fb.p), vr = k * e * sin(nu), vt = k * (1 + e * cos(nu)) * fb.sense;
    set3(p, r * cos(th), r * sin(th), 0);
    if (v) set3(v, vr * cos(th) - vt * sin(th), vr * sin(th) + vt * cos(th), 0);
}
static double fb_t_at_r(double r) {
    double ch = (1 - r / fb.a) / fb.e, H = acosh(fmax(ch, 1));
    return (fb.e * sinh(H) - H) / fb.n;
}

static void fb_init(uint32_t seed) {
    fb.yaw0 = (hrand(seed, 3) - 0.5) * 0.6;
    fb.ra = (5.45 + 0.9 * hrand(seed, 4)) * AU;
    fb.rp = (3.0 + 4.0 * hrand(seed, 5)) * R_JUP;
    fb.trailing = hrand(seed, 6) < 0.8;
    fb.rJ = 5.2026 * AU;
    double mu = MU_SUN, rJ = fb.rJ, ain = (AU + fb.ra) / 2, ein = (fb.ra - AU) / (fb.ra + AU);
    fb.Vp[0] = 0; fb.Vp[1] = sqrt(mu / rJ);
    double v = sqrt(mu * (2 / rJ - 1 / ain)), h = sqrt(mu * ain * (1 - ein * ein)), vt = h / rJ, vr = sqrt(fmax(v * v - vt * vt, 0));
    fb.vin[0] = vr; fb.vin[1] = vt;
    double vi[2] = {fb.vin[0] - fb.Vp[0], fb.vin[1] - fb.Vp[1]};
    fb.vinf = hypot(vi[0], vi[1]);
    fb.uin[0] = vi[0] / fb.vinf; fb.uin[1] = vi[1] / fb.vinf;
    fb.e = 1 + fb.rp * fb.vinf * fb.vinf / MU_JUP;
    fb.delta = 2 * asin(1 / fb.e);
    double best = fb.trailing ? -1 : 1e9;
    for (int s = -1; s <= 1; s += 2) {
        double c = cos(s * fb.delta), sn = sin(s * fb.delta);
        double uo[2] = {c * fb.uin[0] - sn * fb.uin[1], sn * fb.uin[0] + c * fb.uin[1]};
        double sp = hypot(fb.Vp[0] + fb.vinf * uo[0], fb.Vp[1] + fb.vinf * uo[1]);
        if (fb.trailing ? sp > best : sp < best) { best = sp; fb.uout[0] = uo[0]; fb.uout[1] = uo[1]; }
    }
    fb.sense = fb.uin[0] * fb.uout[1] - fb.uin[1] * fb.uout[0] > 0 ? 1 : -1;
    double d[2] = {fb.uin[0] - fb.uout[0], fb.uin[1] - fb.uout[1]}, dl = hypot(d[0], d[1]);
    fb.ph[0] = d[0] / dl; fb.ph[1] = d[1] / dl;
    fb.a = -MU_JUP / (fb.vinf * fb.vinf);
    fb.p = fb.a * (1 - fb.e * fb.e);
    fb.n = sqrt(MU_JUP / pow(-fb.a, 3));
    fb.b = -fb.a * sqrt(fb.e * fb.e - 1);
    fb.vout[0] = fb.Vp[0] + fb.vinf * fb.uout[0];
    fb.vout[1] = fb.Vp[1] + fb.vinf * fb.uout[1];
    fb.dv = hypot(fb.vout[0] - fb.vin[0], fb.vout[1] - fb.vin[1]);
    fb.tmax = fb_t_at_r(90 * R_JUP);
    double tq = fb_t_at_r(fb.rp * 2.2);
    Warp w = {6, {0, 28, 37, 46, 80, 90}, {-fb.tmax, -tq, 0, tq, fb.tmax, fb.tmax * 1.12}};
    fb.w = w;
    double r0[3] = {rJ, 0, 0}, v0[3] = {fb.vin[0], fb.vin[1], 0}, v1[3] = {fb.vout[0], fb.vout[1], 0};
    orb_from_rv(MU_SUN, r0, v0, 0, &fb.hin);
    orb_from_rv(MU_SUN, r0, v1, 0, &fb.hout);
    fb.nuenc_in = orb_nu_at(&fb.hin, 0);
    fb.nuenc_out = orb_nu_at(&fb.hout, 0);
    if (fb.hout.e >= 1) {
        double rr = 14 * AU, pp = orb_p(&fb.hout);
        fb.nuout_end = acos(clampd((pp / rr - 1) / fb.hout.e, -1, 1));
    } else fb.nuout_end = fb.nuenc_out + TAU_D * 0.999;
}

static void fb_update(double u, Info *in) {
    double rate;
    fb.t = warp_at(&fb.w, u, &rate);
    fb_state(fb.t, fb.pos, fb.vel);
    in->title = "Gravity assist";
    in->sub = fb.trailing ? "a Jupiter flyby behind the planet: momentum borrowed, speed gained" : "a Jupiter flyby ahead of the planet: speed given up";
    in->frame = "JOVICENTRIC  \xc2\xb7  HELIOCENTRIC";
    in->eq[0] = "e = 1 + r(p) v\xe2\x88\x9e\xc2\xb2 / \xce\xbc";
    in->eq[1] = "\xce\xb4 = 2 arcsin(1/e)";
    in->eq[2] = "|\xce\x94v| = 2 v\xe2\x88\x9e sin(\xce\xb4/2)";
    in->eq[3] = "v\xe2\x81\xba = V(J) + R(\xce\xb4) v\xe2\x88\x9e\xe2\x81\xbb";
    in->neq = 4;
    in->warp = rate;
    fmt_met(in->clock, sizeof in->clock, fb.t);
    snprintf(in->clock + strlen(in->clock), sizeof in->clock - strlen(in->clock), "  PJ");
    double r = norm3(fb.pos), vr = norm3(fb.vel);
    double vh[2] = {fb.Vp[0] + fb.vel[0], fb.Vp[1] + fb.vel[1]};
    tel(in, "RANGE", "%.2f R(J)", r / R_JUP);
    tel(in, "ALTITUDE", "%.0f km", r - R_JUP);
    tel(in, "V JOVICENTRIC", "%.3f km/s", vr);
    tel(in, "V HELIOCENTRIC", "%.3f km/s", hypot(vh[0], vh[1]));
    double ang = atan2(fb.vel[1], fb.vel[0]) - atan2(fb.uin[1], fb.uin[0]);
    tel(in, "TURNED SO FAR", "%.1f\xc2\xb0", fabs(wrap_pi(ang)) / RAD);
    tel(in, "IMPACT PARAM b", "%.2f R(J)", fb.b / R_JUP);
    tel(in, "SOI RADIUS", "%.1f Gm", fb.rJ * pow(MU_JUP / MU_SUN, 0.4) / 1e6);
    in->dv_head = "FLYBY  km/s";
    char b[24];
    dvrow(in, "v\xe2\x88\x9e  hyperbolic excess", fb.vinf, 1);
    snprintf(b, sizeof b, "%.2f R(J)", fb.rp / R_JUP);
    dvrow_s(in, "periapsis r(p)", b, 1);
    snprintf(b, sizeof b, "%.1f\xc2\xb0", fb.delta / RAD);
    dvrow_s(in, "turn angle \xce\xb4", b, 1);
    dvrow(in, "|\xce\x94v| from Jupiter", fb.dv, 2);
    dvrow(in, "v helio, in", hypot(fb.vin[0], fb.vin[1]), 3);
    dvrow(in, "v helio, out", hypot(fb.vout[0], fb.vout[1]), 3);
    dvrow(in, "solar escape at 5.2 AU", sqrt(2 * MU_SUN / fb.rJ), 3);
}

static void fb_vectors(Rect r) {
    /* fit 0, Vp, vin, vout and the v-infinity circle */
    double xs[5] = {0, fb.Vp[0], fb.vin[0], fb.vout[0], fb.Vp[0]}, ys[5] = {0, fb.Vp[1], fb.vin[1], fb.vout[1], fb.Vp[1]};
    double x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (int i = 0; i < 4; i++) { x0 = fmin(x0, xs[i]); x1 = fmax(x1, xs[i]); y0 = fmin(y0, ys[i]); y1 = fmax(y1, ys[i]); }
    double vmax = fb.vinf * 1.15;
    x0 = fmin(x0, fb.Vp[0] - vmax); x1 = fmax(x1, fb.Vp[0] + vmax);
    y0 = fmin(y0, fb.Vp[1] - vmax); y1 = fmax(y1, fb.Vp[1] + vmax);
    float m = 26 * UI;
    double s = fmin((r.x1 - r.x0 - 2 * m) / (x1 - x0), (r.y1 - r.y0 - 2 * m - 20 * UI) / (y1 - y0));
    float ox = (r.x0 + r.x1) * 0.5f - (float)((x0 + x1) * 0.5 * s), oy = (r.y0 + r.y1) * 0.5f + 10 * UI + (float)((y0 + y1) * 0.5 * s);
#define VX(x) (ox + (float)((x) * s))
#define VY(y) (oy - (float)((y) * s))
    text(st_h, r.x0 + 4 * UI, r.y0 + cap_h(st_h) + 4 * UI, -1, "VELOCITY DIAGRAM  km/s", ca(pal->dim, 1));
    float tx = VX(fb.Vp[0]), ty = VY(fb.Vp[1]);
    ring_d(tx, ty, (float)(fb.vinf * s), 1, ca(pal->alt, 0.55f), 48);
    disc(VX(0), VY(0), 2.5f * UI, ca(pal->ink, 0.8f));
    arrow(VX(0), VY(0), tx, ty, 1.5f * UI, ca(pal->dim, 0.95f), 10 * UI);
    arrow(VX(0), VY(0), VX(fb.vin[0]), VY(fb.vin[1]), 1.6f * UI, ca(pal->orb2, 1), 10 * UI);
    arrow(VX(0), VY(0), VX(fb.vout[0]), VY(fb.vout[1]), 2 * UI, ca(pal->orb, 1), 11 * UI);
    arrow(tx, ty, VX(fb.vin[0]), VY(fb.vin[1]), 1.2f * UI, ca(pal->alt, 0.7f), 8 * UI);
    arrow(tx, ty, VX(fb.vout[0]), VY(fb.vout[1]), 1.2f * UI, ca(pal->alt, 0.7f), 8 * UI);
    arrow(VX(fb.vin[0]), VY(fb.vin[1]), VX(fb.vout[0]), VY(fb.vout[1]), 1.8f * UI, ca(pal->hi, 1), 10 * UI);
    /* the relative velocity turning, v-infinity long: from v-inf in to v-inf out */
    double a0 = atan2(fb.uin[1], fb.uin[0]), a1 = atan2(fb.vel[1], fb.vel[0]), da = wrap_pi(a1 - a0);
    float px = tx, py = ty, rr = (float)(fb.vinf * s);
    for (int i = 0; i <= 48; i++) {
        double a = a0 + da * i / 48;
        float x = tx + rr * (float)cos(a), y = ty - rr * (float)sin(a);
        if (i) line(px, py, x, y, 2 * UI, ca(pal->alt, 0.95f));
        px = x; py = y;
    }
    arrow(tx, ty, px, py, 1.6f * UI, ca(pal->alt, 1), 9 * UI);
    glow(px, py, 16 * UI, cadd(pal->alt, 0.5f));
    char b[48];
    snprintf(b, sizeof b, "V(J) %.1f", hypot(fb.Vp[0], fb.Vp[1]));
    text_h(st_tiny, (VX(0) + tx) * 0.5f + 8 * UI, (VY(0) + ty) * 0.5f, -1, b, ca(pal->dim, 1));
    snprintf(b, sizeof b, "v\xe2\x81\xbb %.1f", hypot(fb.vin[0], fb.vin[1]));
    text_h(st_tiny, VX(fb.vin[0]) + 6 * UI, VY(fb.vin[1]) + 14 * UI, -1, b, ca(pal->orb2, 1));
    snprintf(b, sizeof b, "v\xe2\x81\xba %.1f", hypot(fb.vout[0], fb.vout[1]));
    text_h(st_tiny, VX(fb.vout[0]) + 6 * UI, VY(fb.vout[1]) - 6 * UI, -1, b, ca(pal->orb, 1));
    snprintf(b, sizeof b, "\xce\x94v %.2f", fb.dv);
    text_h(st_tiny, (VX(fb.vin[0]) + VX(fb.vout[0])) * 0.5f + 8 * UI, (VY(fb.vin[1]) + VY(fb.vout[1])) * 0.5f + 4 * UI, -1, b, ca(pal->hi, 1));
    snprintf(b, sizeof b, "v\xe2\x88\x9e %.2f", fb.vinf);
    text_h(st_tiny, tx + (float)(fb.vinf * s) * 0.72f, ty + (float)(fb.vinf * s) * 0.72f + 6 * UI, -1, b, ca(pal->alt, 1));
#undef VX
#undef VY
}

static void fb_helio(Rect r) {
    View v;
    view_init(&v, r.x0, r.y0, r.x1, r.y1);
    view_cam(&v, -PI_D / 2, PI_D / 2);
    v.c[0] = 2.2 * AU;
    v.s = fmin((r.x1 - r.x0) * 0.5 / (7.6 * AU), (r.y1 - r.y0 - 30 * UI) * 0.5 / (6.2 * AU));
    v.cy += 10 * UI;
    text(st_h, r.x0 + 4 * UI, r.y0 + cap_h(st_h) + 4 * UI, -1, "HELIOCENTRIC", ca(pal->dim, 1));
    double O[3] = {0, 0, 0}, z[3] = {0, 0, 1};
    draw_circle(&v, O, AU, 1, ca(pal->orb2, 0.6f), 0);
    draw_circle(&v, O, fb.rJ, 1.2f * UI, ca(pal->orb2, 0.85f), 0);
    planet(&v, O, 696000, B_SUN, z, 0, 5 * UI, pal->sun);
    double nu0 = 0, nu1 = fb.nuenc_in;
    if (nu1 < nu0) nu1 += TAU_D;
    draw_conic(&v, &fb.hin, nu0, nu1, 1.5f * UI, ca(pal->orb2, 1), 0, 0);
    draw_conic(&v, &fb.hin, nu1, nu0 + TAU_D, 1, ca(pal->orb2, 0.3f), 5 * UI, 0);
    double e1 = fb.nuout_end;
    if (fb.hout.e < 1) { while (e1 < fb.nuenc_out) e1 += TAU_D; }
    int past = fb.t > 0;
    draw_conic(&v, &fb.hout, fb.nuenc_out, e1, (past ? 2.0f : 1.4f) * UI, ca(pal->orb, past ? 1 : 0.6f), past ? 0 : 7 * UI, past ? 4 : 0);
    double J[3] = {fb.rJ, 0, 0}, sd[3] = {-1, 0, 0};
    planet(&v, J, R_JUP, B_JUPITER, sd, 0, 4 * UI, pal->jup);
    float x, y;
    proj(&v, J, &x, &y);
    text_h(st_tiny, x + 8 * UI, y + 14 * UI, -1, "JUPITER", ca(pal->ink, 0.9f));
    double E[3] = {AU, 0, 0};
    proj(&v, E, &x, &y);
    text_h(st_tiny, x + 6 * UI, y - 6 * UI, -1, "1 AU", ca(pal->dim, 0.9f));
    if (fb.hout.e >= 1) text(st_tiny, r.x1 - 6 * UI, r.y1 - 8 * UI, 1, "e > 1: SOLAR ESCAPE", ca(pal->hi, 0.95f));
    else {
        char b[48];
        snprintf(b, sizeof b, "aphelion %.1f AU", fb.hout.a * (1 + fb.hout.e) / AU);
        text(st_tiny, r.x1 - 6 * UI, r.y1 - 8 * UI, 1, b, ca(pal->hi, 0.95f));
    }
}

static void fb_draw(Rect R, double u) {
    int port = (R.y1 - R.y0) > (R.x1 - R.x0) * 1.05f;
    Rect A = R, Bv, Bh;
    if (port) {
        A.y1 = R.y0 + (R.y1 - R.y0) * 0.58f;
        float w = (R.x1 - R.x0) * 0.5f;
        Bv = (Rect){R.x0 + 10 * UI, A.y1 + 10 * UI, R.x0 + w - 6 * UI, R.y1 - 10 * UI};
        Bh = (Rect){R.x0 + w + 6 * UI, A.y1 + 10 * UI, R.x1 - 10 * UI, R.y1 - 10 * UI};
    } else {
        A.x1 = R.x0 + (R.x1 - R.x0) * 0.62f;
        float top = R.y0 + 150 * UI, h = (R.y1 - top) * 0.5f;
        Bv = (Rect){A.x1 + 10 * UI, top, R.x1 - 24 * UI, top + h - 6 * UI};
        Bh = (Rect){A.x1 + 10 * UI, top + h + 6 * UI, R.x1 - 24 * UI, R.y1 - 24 * UI};
    }
    View v;
    view_init(&v, A.x0, A.y0, A.x1, A.y1);
    double el = 78 * RAD;
    view_cam(&v, -PI_D / 2 + fb.yaw0 + sin(u * 0.03) * 0.12, el);
    double ae0 = -fb.a * fb.e;
    v.s = fit_scale(&v, fmax(0.5 * ae0 + 12 * R_JUP, 30 * R_JUP), el, 10 * UI);
    v.c[0] = fb.ph[0] * ae0 * 0.5;
    v.c[1] = fb.ph[1] * ae0 * 0.5;
    double O[3] = {0, 0, 0};
    view_occ(&v, O, R_JUP);
    range_rings(&v, O, ae0 + 20 * R_JUP, 10 * R_JUP, 1 / R_JUP, "R(J)", 0.8f);
    /* asymptotes through the hyperbola's centre, and the impact parameter */
    double ae = -fb.a * fb.e, C[3] = {fb.ph[0] * ae, fb.ph[1] * ae, 0};
    double L = 130 * R_JUP;
    double a0[6] = {C[0], C[1], 0, C[0] - fb.uin[0] * L, C[1] - fb.uin[1] * L, 0};
    double a1[6] = {C[0], C[1], 0, C[0] + fb.uout[0] * L, C[1] + fb.uout[1] * L, 0};
    poly(&v, a0, 2, 1, ca(pal->dim, 0.7f), 6 * UI, 0, 1, 1);
    double a2[6] = {C[0], C[1], 0, C[0] + fb.uin[0] * 14 * R_JUP, C[1] + fb.uin[1] * 14 * R_JUP, 0};
    poly(&v, a2, 2, 1, ca(pal->dim, 0.45f), 2 * UI, 0, 1, 1);
    poly(&v, a1, 2, 1, ca(pal->dim, 0.7f), 6 * UI, 0, 1, 1);
    double k = -(C[0] * fb.uin[0] + C[1] * fb.uin[1]);
    double F[3] = {C[0] + fb.uin[0] * k, C[1] + fb.uin[1] * k, 0}, bl[6] = {0, 0, 0, F[0], F[1], 0};
    poly(&v, bl, 2, 1.2f * UI, ca(pal->alt, 0.8f), 3 * UI, 0, 1, 1);
    float x, y;
    char b[48];
    proj(&v, F, &x, &y);
    snprintf(b, sizeof b, "b = %.1f R(J)", fb.b / R_JUP);
    text_h(st_tiny, x + 8 * UI, y + 4 * UI, -1, b, ca(pal->alt, 1));
    /* the turn angle at the centre */
    {
        double a_in = atan2(fb.uin[1], fb.uin[0]), d = wrap_pi(atan2(fb.uout[1], fb.uout[0]) - a_in);
        static double arc[3 * 41];
        double rr = 7 * R_JUP;
        for (int i = 0; i <= 40; i++) set3(arc + 3 * i, C[0] + rr * cos(a_in + d * i / 40), C[1] + rr * sin(a_in + d * i / 40), 0);
        poly(&v, arc, 41, 1.2f * UI, ca(pal->hi, 0.9f), 0, 0, 1, 1);
        proj(&v, arc + 3 * 20, &x, &y);
        snprintf(b, sizeof b, "\xce\xb4 = %.1f\xc2\xb0", fb.delta / RAD);
        text_h(st_lab, x + 6 * UI, y - 6 * UI, -1, b, ca(pal->hi, 1));
    }
    /* the hyperbola: flown, then to come */
    static double pts[3 * 301];
    int n = 300;
    double ta = -fb.tmax * 1.15, tb = fb.tmax * 1.15;
    for (int i = 0; i <= n; i++) {
        /* denser near periapsis: time through a sinh */
        double s = -1 + 2.0 * i / n, t = sinh(s * asinh(tb / (fb.tmax * 0.02))) * fb.tmax * 0.02;
        (void)ta;
        fb_state(t, pts + 3 * i, NULL);
    }
    int k1 = 0;
    for (int i = 0; i <= n; i++) {
        double s = -1 + 2.0 * i / n, t = sinh(s * asinh(tb / (fb.tmax * 0.02))) * fb.tmax * 0.02;
        if (t <= fb.t) k1 = i;
    }
    poly(&v, pts, k1 + 1, 2 * UI, pal->orb, 0, 4, 1, 1);
    poly(&v, pts + 3 * k1, n + 1 - k1, 1.4f * UI, ca(pal->orb, 0.85f), 9 * UI, 0, 1, 1);
    double sd[3] = {-1, 0, 0};
    planet(&v, O, R_JUP, B_JUPITER, sd, (float)(u * 0.05), 6 * UI, pal->jup);
    /* periapsis */
    double P[3] = {fb.ph[0] * fb.rp, fb.ph[1] * fb.rp, 0};
    proj(&v, P, &x, &y);
    float jx, jy;
    proj(&v, O, &jx, &jy);
    float dx = x - jx, dy = y - jy, dl = fmaxf(hypotf(dx, dy), 1);
    line(x + dx / dl * 4 * UI, y + dy / dl * 4 * UI, x + dx / dl * 14 * UI, y + dy / dl * 14 * UI, 1.4f * UI, ca(pal->ink, 0.9f));
    snprintf(b, sizeof b, "PERIJOVE  %.2f R(J)", fb.rp / R_JUP);
    text_h(st_tiny, x + dx / dl * 20 * UI, y + dy / dl * 20 * UI + 4 * UI, dx > 0 ? -1 : 1, b, ca(pal->ink, 0.9f));
    /* Jupiter's own motion and the way to the Sun */
    double Vd[3] = {0, 9 * R_JUP, 0}, Sd[3] = {-9 * R_JUP, 0, 0};
    float x1, y1;
    proj(&v, Vd, &x1, &y1);
    arrow(jx, jy, x1, y1, 1.3f * UI, ca(pal->dim, 0.9f), 9 * UI);
    text_h(st_tiny, x1 + 6 * UI, y1, -1, "V(J)", ca(pal->dim, 1));
    proj(&v, Sd, &x1, &y1);
    line_d(jx, jy, x1, y1, 1, ca(pal->sun, 0.6f), 4 * UI);
    text_h(st_tiny, x1, y1 - 6 * UI, 0, "\xe2\x86\x90 SUN", ca(pal->dim, 1));
    craft(&v, fb.pos, fb.vel, pal->orb, NULL);
    flush(&A);
    fb_vectors(Bv);
    flush(&Bv);
    fb_helio(Bh);
    flush(&Bh);
}

/* ================================================================== */
/* ---- act 4: the circular restricted three-body problem ----          */
/* ================================================================== */
#define MU_EM 0.012150585
#define MU_SJ 9.537e-4
#define CR_DT 0.01
#define CR_T 36.0
#define NCH 5
static struct {
    double Le[5][3], Ls[5][3];
    double ly0[4], lyT, lyC, lydrift;
    double *ly;
    int nly;
    double *ch[NCH];
    int chn[NCH], nch;
    double chdrift;
    double *hs, *tp;
    int nhs, ntp;
    double hsC, tpC, sjdrift;
    Warp wa, wb;
    /* now */
    int ph;
    double tm, rate;
} cr;

static void cr_init(uint32_t seed) {
    cr_mu = MU_SJ;
    cr_lpoints();
    for (int i = 0; i < 5; i++) { cr.Ls[i][0] = Lx[i]; cr.Ls[i][1] = Ly[i]; cr.Ls[i][2] = LC[i]; }
    {
        /* horseshoe and tadpole: near-circular starts beside L3 and L4 */
        double dr = 0.006 + 0.004 * hrand(seed, 7), r = 1 + dr, vt = sqrt(1 / r) - r;
        double s0[4] = {-r - MU_SJ, 0, 0, -vt}, d;
        if (!cr.hs) cr.hs = malloc(sizeof(double) * 4 * 6002);
        cr.nhs = cr_run(s0, 600, 0.1, cr.hs, 6001, 1e-3, 1e-4, &d);
        cr.hsC = cr_jacobi(s0);
        cr.sjdrift = d;
        double th = (72 + 6 * hrand(seed, 8)) * RAD;
        r = 1.008 + 0.006 * hrand(seed, 9);
        vt = sqrt(1 / r) - r;
        double s1[4] = {r * cos(th) - MU_SJ, r * sin(th), -vt * sin(th), vt * cos(th)};
        if (!cr.tp) cr.tp = malloc(sizeof(double) * 4 * 6002);
        cr.ntp = cr_run(s1, 600, 0.1, cr.tp, 6001, 1e-3, 1e-4, &d);
        cr.tpC = cr_jacobi(s1);
        cr.sjdrift = fmax(cr.sjdrift, d);
    }
    cr_mu = MU_EM;
    cr_lpoints();
    for (int i = 0; i < 5; i++) { cr.Le[i][0] = Lx[i]; cr.Le[i][1] = Ly[i]; cr.Le[i][2] = LC[i]; }
    double ax = 0.0105 + 0.0035 * hrand(seed, 10);
    cr.lyT = cr_lyapunov(ax, cr.ly0);
    cr.lyC = cr_jacobi(cr.ly0);
    if (!cr.ly) cr.ly = malloc(sizeof(double) * 4 * 1002);
    cr.nly = cr_run(cr.ly0, cr.lyT, cr.lyT / 400, cr.ly, 1001, 0, 0, &cr.lydrift);
    /* transits: starts in the Earth's realm at the Lyapunov orbit's energy
     * that pass through the L1 neck into the Moon's */
    double rE = R_EARTH / 384400.0, rM = R_MOON / 384400.0;
    cr.nch = 0;
    cr.chdrift = 0;
    int full = (int)(CR_T / CR_DT);
    for (int tr = 0; tr < 140 && cr.nch < NCH; tr++) {
        double x = 0.50 + 0.28 * hrand(seed, 100 + tr), y = (hrand(seed, 200 + tr) - 0.5) * 0.3;
        double v2 = cr_2omega(x, y) - cr.lyC;
        if (v2 <= 0) continue;
        double a = hrand(seed, 300 + tr) * TAU_D, s0[4] = {x, y, sqrt(v2) * cos(a), sqrt(v2) * sin(a)}, d;
        if (!cr.ch[cr.nch]) cr.ch[cr.nch] = malloc(sizeof(double) * 4 * (full + 2));
        int n = cr_run(s0, CR_T, CR_DT, cr.ch[cr.nch], full + 1, rE, rM, &d);
        if (n < full * 0.8) continue;
        int crossed = 0;
        for (int i = 0; i < n * 3 / 5; i++) if (cr.ch[cr.nch][4 * i] > cr.Le[0][0] + 0.02) { crossed = 1; break; }
        if (!crossed) continue;
        cr.chn[cr.nch] = n;
        cr.chdrift = fmax(cr.chdrift, d);
        cr.nch++;
    }
    Warp wa = {5, {0, 3, 49, 53, 54}, {0, 0.25, CR_T - 0.4, CR_T, CR_T}};
    Warp wb = {5, {54, 57, 86, 89, 90}, {0, 6, 590, 600, 600}};
    cr.wa = wa;
    cr.wb = wb;
}

static void cr_update(double u, Info *in) {
    cr.ph = u >= 54;
    cr.tm = cr.ph ? warp_at(&cr.wb, u, &cr.rate) : warp_at(&cr.wa, u, &cr.rate);
    in->title = "Three-body problem";
    in->sub = cr.ph ? "Sun\xe2\x80\x93Jupiter, rotating frame: a tadpole (Trojan) and a horseshoe orbit"
                    : "Earth\xe2\x80\x93Moon, rotating frame: transits through the L1 neck";
    in->frame = "ROTATING (SYNODIC), NONDIMENSIONAL";
    in->eq[0] = "\xe1\xba\x8d \xe2\x88\x92 2\xe1\xba\x8f = \xe2\x88\x82\xce\xa9/\xe2\x88\x82x";
    in->eq[1] = "\xc3\xbf + 2\xe1\xba\x8b = \xe2\x88\x82\xce\xa9/\xe2\x88\x82y";
    in->eq[2] = "\xce\xa9 = \xc2\xbd(x\xc2\xb2 + y\xc2\xb2) + (1\xe2\x88\x92\xce\xbc)/r\xe2\x82\x81 + \xce\xbc/r\xe2\x82\x82";
    in->eq[3] = "C = 2\xce\xa9 \xe2\x88\x92 (\xe1\xba\x8b\xc2\xb2 + \xe1\xba\x8f\xc2\xb2)";
    in->neq = 4;
    double tu = cr.ph ? 11.862 * 365.25 / TAU_D : 27.321661 / TAU_D;     /* days per time unit */
    in->warp = cr.rate * tu * 86400;
    if (cr.ph) snprintf(in->clock, sizeof in->clock, "t = %.1f yr", cr.tm * tu / 365.25);
    else snprintf(in->clock, sizeof in->clock, "t = %.2f d", cr.tm * tu);
    double (*L)[3] = cr.ph ? cr.Ls : cr.Le;
    tel(in, "MASS RATIO \xce\xbc", cr.ph ? "%.4e" : "%.9f", cr.ph ? MU_SJ : MU_EM);
    if (!cr.ph) {
        tel(in, "JACOBI C", "%.6f", cr.lyC);
        tel(in, "C(L1) / C(L2)", "%.5f / %.5f", L[0][2], L[1][2]);
        tel(in, "LYAPUNOV PERIOD", "%.2f d", cr.lyT * tu);
        tel(in, "|\xce\x94" "C| MAX", "%.1e", fmax(cr.chdrift, cr.lydrift));
        tel(in, "TRANSITS", "%d", cr.nch);
        tel(in, "LENGTH UNIT", "384 400 km");
    } else {
        tel(in, "C HORSESHOE", "%.6f", cr.hsC);
        tel(in, "C TADPOLE", "%.6f", cr.tpC);
        tel(in, "C(L3) / C(L4)", "%.5f / %.5f", L[2][2], L[3][2]);
        tel(in, "|\xce\x94" "C| MAX", "%.1e", cr.sjdrift);
        tel(in, "LENGTH UNIT", "5.20 AU");
    }
    tel(in, "INTEGRATOR", "RK45 DP, tol 1e-11");
    in->dv_head = "LAGRANGE POINTS   x, y   C";
    char nm[48], v[24];
    for (int i = 0; i < 5; i++) {
        snprintf(nm, sizeof nm, "L%d  %+.5f, %+.4f", i + 1, L[i][0], L[i][1]);
        snprintf(v, sizeof v, "%.5f", L[i][2]);
        dvrow_s(in, nm, v, i == 0 && !cr.ph ? 1 : 3);
    }
    if (!cr.ph) {
        char k[24];
        fmt_k(k, sizeof k, (L[0][0] + MU_EM) * 384400);
        snprintf(v, sizeof v, "%s km", k);
        dvrow_s(in, "L1 from Earth", v, 2);
    }
}

static const float *cr_col(int i) {
    const float *c[5] = {pal->hi, pal->alt, pal->ok, pal->ink, pal->warn};
    return c[i % 5];
}

static void cr_trail(const View *v, const double *s, int n, int upto, int tail, const float *c, float w) {
    static double pts[3 * 8000];
    if (upto >= n) upto = n - 1;
    if (upto < 1) return;
    /* the whole path so far, faint and thinned */
    int k = 0, step = upto > 2000 ? 4 : 2;
    for (int i = 0; i <= upto && k < 8000; i += step) { set3(pts + 3 * k, s[4 * i], s[4 * i + 1], 0); k++; }
    poly(v, pts, k, 1, ca(c, 0.16f), 0, 0, 1, 1);
    /* the recent tail, bright and fading */
    int i0 = upto - tail < 0 ? 0 : upto - tail;
    k = 0;
    for (int i = i0; i <= upto && k < 8000; i++) { set3(pts + 3 * k, s[4 * i], s[4 * i + 1], 0); k++; }
    poly(v, pts, k, w, c, 0, 3, 0, 1);
    float x, y;
    proj(v, pts + 3 * (k - 1), &x, &y);
    glow(x, y, 14 * UI, cadd(c, 0.6f));
    disc(x, y, 2.6f * UI, ca(c, 1));
}

static int g_solo;                     /* the front output is alone: it carries every card */
static double cr_zoom = -1;            /* >= 0: force the Earth-Moon framing (0 whole, 1 the neck) */
static double cr_zoom_at(double u) { return smooth(13, 22, u) * (1 - smooth(38, 48, u)); }
static void cr_draw(Rect R, double u) {
    View v;
    view_init(&v, R.x0, R.y0, R.x1, R.y1);
    view_cam(&v, -PI_D / 2, PI_D / 2);
    float w = R.x1 - R.x0, h = R.y1 - R.y0;
    double mu = cr.ph ? MU_SJ : MU_EM, C = cr.ph ? cr.hsC : cr.lyC;
    double (*L)[3] = cr.ph ? cr.Ls : cr.Le;
    if (!cr.ph) {
        double z = cr_zoom >= 0 ? cr_zoom : cr_zoom_at(u);
        double s0 = fmin(w * 0.5 / 1.28, h * 0.5 / 1.02), s1 = fmin(w * 0.5 / 0.34, h * 0.5 / 0.26);
        v.s = exp(lerpd(log(s0), log(s1), z));
        v.c[0] = lerpd(0.08, 0.99, z);
    } else {
        v.s = fmin(w * 0.5 / 1.2, h * 0.5 / 1.12);
        v.c[0] = 0;
    }
    /* the forbidden realm and its zero-velocity curve */
    {
        /* over the whole output: insets are scissored */
        float cx = W * 0.5f, cy = H * 0.5f;
        float pp[4] = {(float)(v.c[0] + (cx - v.cx) / v.s), (float)(v.c[1] - (cy - v.cy) / v.s), (float)(1 / v.s), (float)C};
        inst(cx, cy, W * 0.5f, H * 0.5f, 0, K_FIELD, pp, ca(pal->orb2, 1), (float)mu, 0);
    }
    /* axes */
    double ax[6] = {-1.6, 0, 0, 1.6, 0, 0}, ay[6] = {0, -1.3, 0, 0, 1.3, 0};
    poly(&v, ax, 2, 1, ca(pal->faint, 0.8f), 0, 0, 1, 1);
    poly(&v, ay, 2, 1, ca(pal->faint, 0.5f), 0, 0, 1, 1);
    double O[3] = {-mu, 0, 0};
    draw_circle(&v, O, 1, 1, ca(pal->faint, 0.8f), 5 * UI);
    char b[48];
    if (!cr.ph) {
        int nl = cr.nly;
        static double pts[3 * 1002];
        for (int i = 0; i < nl; i++) set3(pts + 3 * i, cr.ly[4 * i], cr.ly[4 * i + 1], 0);
        poly(&v, pts, nl, 1.6f * UI, pal->orb, 0, 4, 1, 1);
        int idx = (int)(fmod(cr.tm, cr.lyT) / cr.lyT * (nl - 1));
        float x, y;
        proj(&v, pts + 3 * idx, &x, &y);
        disc(x, y, 2.5f * UI, ca(pal->orb, 1));
        glow(x, y, 12 * UI, cadd(pal->orb, 0.6f));
        int upto = (int)(cr.tm / CR_DT);
        for (int i = 0; i < cr.nch; i++) cr_trail(&v, cr.ch[i], cr.chn[i], upto, 500, cr_col(i), 1.4f * UI);
        double E[3] = {-mu, 0, 0}, M[3] = {1 - mu, 0, 0}, sd[3] = {-0.6, 0.5, 0.62};
        planet(&v, E, R_EARTH / 384400.0, B_EARTH, sd, 0, 4 * UI, pal->sea);
        planet(&v, M, R_MOON / 384400.0, B_MOON, sd, 0, 2.5f * UI, pal->moon);
        proj(&v, E, &x, &y);
        text_h(st_tiny, x, y + 22 * UI, 0, "EARTH", ca(pal->ink, 0.9f));
        proj(&v, M, &x, &y);
        text_h(st_tiny, x, y + 18 * UI, 0, "MOON", ca(pal->ink, 0.9f));
        snprintf(b, sizeof b, "LYAPUNOV  T = %.1f d", cr.lyT * 4.3484);
        double q[3] = {cr.ly0[0], 0, 0};
        double top = 0;
        for (int i = 0; i < nl; i++) top = fmax(top, cr.ly[4 * i + 1]);
        q[1] = top;
        proj(&v, q, &x, &y);
        text_h(st_tiny, x, y - 10 * UI, 0, b, ca(pal->orb, 1));
    } else {
        int upto = (int)(cr.tm / 0.1);
        cr_trail(&v, cr.hs, cr.nhs, upto, 600, pal->orb, 1.4f * UI);
        cr_trail(&v, cr.tp, cr.ntp, upto, 600, pal->hi, 1.4f * UI);
        double S[3] = {-mu, 0, 0}, J[3] = {1 - mu, 0, 0}, sd[3] = {-1, 0, 0.3}, z[3] = {0, 0, 1};
        planet(&v, S, 0.0009, B_SUN, z, 0, 7 * UI, pal->sun);
        planet(&v, J, 0.0001, B_JUPITER, sd, 0, 5 * UI, pal->jup);
        float x, y;
        proj(&v, J, &x, &y);
        text_h(st_tiny, x + 10 * UI, y + 16 * UI, -1, "JUPITER", ca(pal->ink, 0.9f));
        proj(&v, S, &x, &y);
        text_h(st_tiny, x, y + 24 * UI, 0, "SUN", ca(pal->ink, 0.9f));
        int i = cr.ntp * 3 / 4;
        double q[3] = {cr.tp[4 * i], cr.tp[4 * i + 1], 0};
        proj(&v, q, &x, &y);
        (void)x;
        double Lq[3] = {L[3][0], L[3][1] + 0.16, 0};
        proj(&v, Lq, &x, &y);
        text_h(st_tiny, x, y, 0, "TADPOLE", ca(pal->hi, 1));
        double Hq[3] = {-1.12, 0.08, 0};
        proj(&v, Hq, &x, &y);
        text_h(st_tiny, x, y, 1, "HORSESHOE", ca(pal->orb, 1));
    }
    /* the Lagrange points */
    for (int i = 0; i < 5; i++) {
        double p[3] = {L[i][0], L[i][1], 0};
        float x, y;
        proj(&v, p, &x, &y);
        if (x < R.x0 || x > R.x1 || y < R.y0 || y > R.y1) continue;
        float s = 5 * UI;
        line(x - s, y, x + s, y, 1.4f * UI, ca(pal->ink, 0.95f));
        line(x, y - s, x, y + s, 1.4f * UI, ca(pal->ink, 0.95f));
        snprintf(b, sizeof b, "L%d", i + 1);
        text_h(st_lab, x + 7 * UI, y - 7 * UI, -1, b, ca(pal->ink, 1));
    }
    snprintf(b, sizeof b, "ZERO-VELOCITY CURVE  C = %.5f", C);
    text(st_tiny, R.x0 + 14 * UI, R.y1 - 14 * UI, -1, b, ca(pal->orb2, 1));
}

/* ================================================================== */
/* ---- act 5: a Walker-delta constellation and its ground tracks ---- */
/* ================================================================== */
typedef struct { const char *name; int T, P, F; double inc, alt; } WalkerDef;
static const WalkerDef WALKERS[] = {
    {"LEO SHELL", 48, 6, 1, 53, 1150},
    {"GALILEO-LIKE MEO", 24, 3, 1, 56, 23222},
    {"POLAR LEO", 36, 6, 2, 86, 780},
};
static struct {
    WalkerDef d;
    double a, n, per, jd0, yaw0, lam;
    int S, hl;
    Warp w;
    /* now */
    double tm, jd, th, rate, cov;
} wk;

static void wk_sat(int k, double t, double *r) {
    int j = k / wk.S, i = k % wk.S;
    double O = j * TAU_D / wk.d.P, u = wk.n * t + i * TAU_D / wk.S + j * wk.d.F * TAU_D / wk.d.T, ci = cos(wk.d.inc * RAD), si = sin(wk.d.inc * RAD);
    set3(r, wk.a * (cos(O) * cos(u) - sin(O) * sin(u) * ci), wk.a * (sin(O) * cos(u) + cos(O) * sin(u) * ci), wk.a * sin(u) * si);
}
static void wk_ground(int k, double t, double *lon, double *lat) {
    double r[3];
    wk_sat(k, t, r);
    *lon = wrap_pi(atan2(r[1], r[0]) - gmst(wk.jd0 + t / DAY));
    *lat = asin(r[2] / wk.a);
}

static void wk_init(uint32_t seed, double wt) {
    wk.d = WALKERS[hash32(seed * 7 + 3) % 3];
    wk.S = wk.d.T / wk.d.P;
    wk.a = R_EARTH + wk.d.alt;
    wk.n = sqrt(MU_EARTH / (wk.a * wk.a * wk.a));
    wk.per = TAU_D / wk.n;
    wk.jd0 = jd_of(wt);
    double sd[3];
    sun_dir_eq(wk.jd0, sd);
    wk.yaw0 = atan2(sd[1], sd[0]) + 0.15 + (hrand(seed, 11) - 0.5) * 0.6;
    wk.hl = (int)(hrand(seed, 12) * wk.d.T);
    double eps = 10 * RAD;
    wk.lam = acos(R_EARTH * cos(eps) / wk.a) - eps;
    double span = wk.per * 1.5;
    Warp w = {5, {0, 4, 45, 86, 90}, {0, span * 0.012, span * 0.5, span * 0.988, span}};
    wk.w = w;
}

static void wk_update(double u, Info *in) {
    wk.tm = warp_at(&wk.w, u, &wk.rate);
    wk.jd = wk.jd0 + wk.tm / DAY;
    wk.th = gmst(wk.jd);
    /* one-fold coverage over a coarse grid, area weighted */
    double tot = 0, cov = 0, cl = cos(wk.lam);
    static double sp[64][3];
    for (int k = 0; k < wk.d.T && k < 64; k++) { wk_sat(k, wk.tm, sp[k]); unit3(sp[k], sp[k]); }
    for (int la = 0; la < 18; la++) {
        double lat = (-85 + la * 10) * RAD, wgt = cos(lat);
        for (int lo = 0; lo < 36; lo++) {
            double lon = (lo * 10 - 180) * RAD + wk.th, g[3] = {cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)};
            int seen = 0;
            for (int k = 0; k < wk.d.T && k < 64 && !seen; k++) seen = dot3(g, sp[k]) >= cl;
            tot += wgt;
            cov += seen * wgt;
        }
    }
    wk.cov = cov / tot;
    static char sub[96];
    snprintf(sub, sizeof sub, "Walker delta %.0f\xc2\xb0:%d/%d/%d at %.0f km, ground tracks over a turning Earth", wk.d.inc, wk.d.T, wk.d.P, wk.d.F, wk.d.alt);
    in->title = "Walker constellation";
    in->sub = sub;
    in->frame = "ECI  \xc2\xb7  EARTH-FIXED";
    in->eq[0] = "\xce\xa9(j) = j \xc2\xb7 360\xc2\xb0 / P";
    in->eq[1] = "u(j,k) = k \xc2\xb7 360\xc2\xb0/S + j \xc2\xb7 F \xc2\xb7 360\xc2\xb0/T";
    in->eq[2] = "\xce\xbb = arccos(R(E) cos \xce\xb5 / r) \xe2\x88\x92 \xce\xb5";
    in->eq[3] = "T(orbit) = 2\xcf\x80 \xe2\x88\x9a(a\xc2\xb3/\xce\xbc)";
    in->neq = 4;
    in->warp = wk.rate;
    char b[48];
    fmt_met(in->clock, sizeof in->clock, wk.tm);
    time_t tt = (time_t)((wk.jd - 2440587.5) * 86400);
    struct tm tm;
    gmtime_r(&tt, &tm);
    strftime(b, sizeof b, "  %H:%M UTC", &tm);
    snprintf(in->clock + strlen(in->clock), sizeof in->clock - strlen(in->clock), "%s", b);
    tel(in, "CONSTELLATION", "%s", wk.d.name);
    tel(in, "ALTITUDE", "%.0f km", wk.d.alt);
    fmt_dur(b, sizeof b, wk.per);
    tel(in, "PERIOD", "%s", b);
    tel(in, "VELOCITY", "%.3f km/s", sqrt(MU_EARTH / wk.a));
    tel(in, "FOOTPRINT \xce\xbb (\xce\xb5 10\xc2\xb0)", "%.1f\xc2\xb0", wk.lam / RAD);
    double lon, lat;
    wk_ground(wk.hl, wk.tm, &lon, &lat);
    tel(in, "TRACKED SAT", "P%d-%02d", wk.hl / wk.S + 1, wk.hl % wk.S + 1);
    tel(in, "SUB-POINT", "%.1f\xc2\xb0%c %.1f\xc2\xb0%c", fabs(lat / RAD), lat >= 0 ? 'N' : 'S', fabs(lon / RAD), lon >= 0 ? 'E' : 'W');
    in->dv_head = "CONSTELLATION  i:T/P/F";
    snprintf(b, sizeof b, "%.0f\xc2\xb0:%d/%d/%d", wk.d.inc, wk.d.T, wk.d.P, wk.d.F);
    dvrow_s(in, "Walker notation", b, 1);
    snprintf(b, sizeof b, "%d", wk.d.P);
    dvrow_s(in, "planes P, 360\xc2\xb0/P apart", b, 3);
    snprintf(b, sizeof b, "%d", wk.S);
    dvrow_s(in, "satellites per plane S", b, 3);
    snprintf(b, sizeof b, "%.1f\xc2\xb0", wk.d.F * 360.0 / wk.d.T);
    dvrow_s(in, "inter-plane phasing F\xc2\xb7" "360\xc2\xb0/T", b, 3);
    snprintf(b, sizeof b, "%.1f%%", wk.cov * 100);
    dvrow_s(in, "Earth covered now (\xce\xb5 \xe2\x89\xa5 10\xc2\xb0)", b, 2);
}

static const float *wk_plane_col(int j) { return mixc(pal->orb, pal->alt, wk.d.P > 1 ? (float)j / (wk.d.P - 1) : 0, 1); }

static void wk_globe(Rect R, double u) {
    View v;
    view_init(&v, R.x0, R.y0, R.x1, R.y1);
    double el = (22 + 8 * sin(u * 0.02)) * RAD;
    view_cam(&v, wk.yaw0 + u * 0.006, el);
    v.s = fmin((R.x1 - R.x0) * 0.5 - 20 * UI, (R.y1 - R.y0) * 0.5 - 20 * UI) / (wk.a * 1.04);
    double O[3] = {0, 0, 0}, sun[3];
    view_occ(&v, O, R_EARTH);
    sun_dir_eq(wk.jd, sun);
    planet(&v, O, R_EARTH, B_EARTH, sun, (float)wk.th, 4, pal->sea);
    static double pts[3 * 129];
    double ci = cos(wk.d.inc * RAD), si = sin(wk.d.inc * RAD);
    for (int j = 0; j < wk.d.P; j++) {
        double Om = j * TAU_D / wk.d.P;
        for (int i = 0; i <= 128; i++) {
            double uu = TAU_D * i / 128;
            set3(pts + 3 * i, wk.a * (cos(Om) * cos(uu) - sin(Om) * sin(uu) * ci), wk.a * (sin(Om) * cos(uu) + cos(Om) * sin(uu) * ci), wk.a * sin(uu) * si);
        }
        poly(&v, pts, 129, 1.1f * UI, ca(wk_plane_col(j), 0.6f), 0, 2.5f, 1, 1);
    }
    for (int k = 0; k < wk.d.T; k++) {
        double r[3];
        wk_sat(k, wk.tm, r);
        if (hidden(&v, r)) continue;
        float x, y;
        proj(&v, r, &x, &y);
        const float *c = wk_plane_col(k / wk.S);
        if (k == wk.hl) { glow(x, y, 22 * UI, cadd(pal->hi, 0.6f)); ring(x, y, 7 * UI, 1.2f * UI, ca(pal->hi, 1)); }
        glow(x, y, 9 * UI, cadd(c, 0.5f));
        disc(x, y, 2.4f * UI, ca(k == wk.hl ? pal->hi : c, 1));
    }
}

static void wk_map(Rect R, double u) {
    (void)u;
    float w = R.x1 - R.x0, h = R.y1 - R.y0;
    /* 2:1, centred in R */
    if (w > 2 * h) { float c = (R.x0 + R.x1) * 0.5f; w = 2 * h; R.x0 = c - h; R.x1 = c + h; }
    else { float c = (R.y0 + R.y1) * 0.5f; h = w / 2; R.y0 = c - h / 2; R.y1 = c + h / 2; }
    double s[3];
    sun_dir_eq(wk.jd, s);
    float slon = (float)wrap_pi(atan2(s[1], s[0]) - wk.th), slat = (float)asin(s[2]);
    float pp[4] = {w * 0.5f, h * 0.5f, 0, 0};
    inst((R.x0 + R.x1) * 0.5f, (R.y0 + R.y1) * 0.5f, w * 0.5f, h * 0.5f, 0, K_MAP, pp, ca(pal->sea, 1), slon, slat);
#define MX(lon) (R.x0 + (float)(((lon) + PI_D) / TAU_D) * w)
#define MY(lat) (R.y0 + (float)((PI_D / 2 - (lat)) / PI_D) * h)
    /* the coast */
    const short *p = COAST;
    while (*p != 10000) {
        int n = 0;
        while (p[2 * n] != 9999) n++;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            if (abs(p[2 * i] - p[2 * j]) > 180 || p[2 * i + 1] <= -89 || p[2 * j + 1] <= -89) continue;
            line(MX(p[2 * i] * RAD), MY(p[2 * i + 1] * RAD), MX(p[2 * j] * RAD), MY(p[2 * j + 1] * RAD), 1.1f * UI, ca(pal->ink, pal->paper ? 0.7f : 0.45f));
        }
        p += 2 * n + 1;
    }
    rect_o(R.x0, R.y0, R.x1, R.y1, 0, 1, ca(pal->dim, 0.6f));
    char b[16];
    for (int lo = -150; lo <= 150; lo += 30) {
        snprintf(b, sizeof b, "%d\xc2\xb0", lo);
        text(st_tiny, MX(lo * RAD), R.y1 + line_h(st_tiny), 0, b, ca(pal->dim, 0.8f));
    }
    for (int la = -60; la <= 60; la += 30) {
        snprintf(b, sizeof b, "%d\xc2\xb0", la);
        text(st_tiny, R.x0 - 5 * UI, MY(la * RAD) + cap_h(st_tiny) * 0.5f, 1, b, ca(pal->dim, 0.8f));
    }
    /* tracks: the last half orbit of each, the next orbit of the tracked one */
    for (int k = 0; k < wk.d.T; k++) {
        int hl = k == wk.hl;
        const float *c = hl ? pal->hi : wk_plane_col(k / wk.S);
        int n = 40;
        double span = wk.per * 0.5;
        float px = 0, py = 0;
        double plon = 0;
        for (int i = 0; i <= n; i++) {
            double t = wk.tm - span + span * i / n, lon, lat;
            wk_ground(k, t, &lon, &lat);
            float x = MX(lon), y = MY(lat);
            if (i && fabs(lon - plon) < PI_D) line(px, py, x, y, (hl ? 1.8f : 1.2f) * UI, ca(c, (hl ? 0.95f : 0.6f) * (float)i / n));
            px = x; py = y; plon = lon;
        }
        if (hl) {
            float ph = 0;
            for (int i = 0; i <= 120; i++) {
                double t = wk.tm + wk.per * i / 120, lon, lat;
                wk_ground(k, t, &lon, &lat);
                float x = MX(lon), y = MY(lat);
                if (i && fabs(lon - plon) < PI_D) { line_dp(px, py, x, y, 1.3f * UI, ca(c, 0.8f), 8 * UI, -ph); ph += hypotf(x - px, y - py); }
                px = x; py = y; plon = lon;
            }
        }
        /* footprint */
        double lon0, lat0;
        wk_ground(k, wk.tm, &lon0, &lat0);
        float fx = 0, fy = 0;
        double flon = 0;
        for (int i = 0; i <= 48; i++) {
            double th = TAU_D * i / 48, l = wk.lam;
            double lat = asin(sin(lat0) * cos(l) + cos(lat0) * sin(l) * cos(th));
            double lon = wrap_pi(lon0 + atan2(sin(th) * sin(l) * cos(lat0), cos(l) - sin(lat0) * sin(lat)));
            float x = MX(lon), y = MY(lat);
            if (i && fabs(lon - flon) < PI_D && (hl || wk.lam < 32 * RAD)) line(fx, fy, x, y, 1, ca(c, hl ? 0.7f : 0.22f));
            fx = x; fy = y; flon = lon;
        }
        float x = MX(lon0), y = MY(lat0);
        glow(x, y, (hl ? 20 : 10) * UI, cadd(c, 0.55f));
        disc(x, y, (hl ? 3.4f : 2.4f) * UI, ca(c, 1));
    }
    /* the subsolar point */
    float sx = MX(slon), sy = MY(slat);
    ring(sx, sy, 6 * UI, 1.2f * UI, ca(pal->sun, 0.9f));
    disc(sx, sy, 2 * UI, ca(pal->sun, 0.9f));
#undef MX
#undef MY
}

static void wk_draw(Rect R, double u) {
    int port = H > W;
    if (port) {
        float m = R.y0 + (R.y1 - R.y0) * 0.55f;
        wk_globe((Rect){R.x0, R.y0, R.x1, m}, u);
        wk_map((Rect){R.x0 + 40 * UI, m + 10 * UI, R.x1 - 20 * UI, R.y1 - 30 * UI}, u);
    } else {
        float m = R.x0 + (R.x1 - R.x0) * 0.36f;
        wk_globe((Rect){R.x0, R.y0, m, R.y1}, u);
        float w = R.x1 - 30 * UI - (m + 40 * UI), top = g_solo ? R.y0 + 150 * UI : fmaxf(R.y0 + 150 * UI, (R.y0 + R.y1) * 0.5f - w * 0.25f);
        wk_map((Rect){m + 40 * UI, top, R.x1 - 30 * UI, top + w * 0.5f}, u);
    }
}

/* the pattern: every satellite at its node and argument of latitude */
static void wk_plot(Rect r, double u) {
    (void)u;
    Plot p = {r, 0, 360, 0, 360, 0, 0};
    pl_frame(&p, "RIGHT ASCENSION OF THE NODE \xce\xa9", "ARGUMENT OF LATITUDE u");
    char b[16];
    for (int i = 0; i <= 360; i += 60) {
        snprintf(b, sizeof b, "%d\xc2\xb0", i);
        pl_tick_x(&p, i, b, 1);
        pl_tick_y(&p, i, b, 1);
    }
    for (int k = 0; k < wk.d.T; k++) {
        int j = k / wk.S, i = k % wk.S;
        double O = j * 360.0 / wk.d.P + 360.0 / wk.d.P * 0.5;
        double uu = fmod((wk.n * wk.tm + i * TAU_D / wk.S + j * wk.d.F * TAU_D / wk.d.T) / RAD, 360);
        const float *c = k == wk.hl ? pal->hi : wk_plane_col(j);
        float x = pl_x(&p, O), y = pl_y(&p, uu);
        glow(x, y, 12 * UI, cadd(c, 0.5f));
        disc(x, y, 3 * UI, ca(c, 1));
        if (i == 0) line(x, r.y0, x, r.y1, 1, ca(c, 0.25f));
    }
}

/* ================================================================== */
/* ---- act 6: orbital decay in the atmosphere ----                    */
/* ================================================================== */
static struct {
    float *st, *cdv;                   /* x, y, vx, vy every DC_DT; cumulative drag delta-v */
    int n, ns;
    double life, hp0, ha0, B, yaw0;
    float *sd, *shp, *sha;             /* the plot: day, perigee, apogee */
    Warp w;
    /* now */
    double t, rate, p[3], v[3];
    Orbit osc;
    int gone;
} dc;
#define DC_DT 10.0
static void dc_acc(const double *s, double *d, double *ad) {
    double r = hypot(s[0], s[1]), v = hypot(s[2], s[3]);
    double rho = atm_rho(r - R_EARTH), k = 500 * rho * v / dc.B;
    d[0] = s[2]; d[1] = s[3];
    d[2] = -MU_EARTH * s[0] / (r * r * r) - k * s[2];
    d[3] = -MU_EARTH * s[1] / (r * r * r) - k * s[3];
    if (ad) *ad = k * v;
}
static void dc_init(void) {
    dc.hp0 = 185; dc.ha0 = 800; dc.B = 50;
    double rp = R_EARTH + dc.hp0, ra = R_EARTH + dc.ha0, a = (rp + ra) / 2;
    double s[4] = {rp, 0, 0, sqrt(MU_EARTH * (2 / rp - 1 / a))}, t = 0, h = DC_DT, dv = 0;
    int cap = 400000;
    dc.st = malloc(sizeof(float) * 4 * cap);
    dc.cdv = malloc(sizeof(float) * cap);
    dc.sd = malloc(sizeof(float) * 4000); dc.shp = malloc(sizeof(float) * 4000); dc.sha = malloc(sizeof(float) * 4000);
    dc.n = 0; dc.ns = 0;
    double next_s = 0;
    for (;;) {
        for (int i = 0; i < 4; i++) dc.st[4 * dc.n + i] = (float)s[i];
        dc.cdv[dc.n] = (float)dv;
        dc.n++;
        double r = hypot(s[0], s[1]);
        if (t >= next_s && dc.ns < 4000) {
            double R3[3] = {s[0], s[1], 0}, V3[3] = {s[2], s[3], 0};
            Orbit o;
            orb_from_rv(MU_EARTH, R3, V3, 0, &o);
            dc.sd[dc.ns] = (float)(t / DAY);
            dc.shp[dc.ns] = (float)(o.a * (1 - o.e) - R_EARTH);
            dc.sha[dc.ns] = (float)(o.a * (1 + o.e) - R_EARTH);
            dc.ns++;
            next_s += 0.04 * DAY;
        }
        if (r - R_EARTH < 75 || dc.n >= cap) break;
        double k1[4], k2[4], k3[4], k4[4], y[4], ad;
        dc_acc(s, k1, &ad);
        for (int i = 0; i < 4; i++) y[i] = s[i] + h / 2 * k1[i];
        dc_acc(y, k2, NULL);
        for (int i = 0; i < 4; i++) y[i] = s[i] + h / 2 * k2[i];
        dc_acc(y, k3, NULL);
        for (int i = 0; i < 4; i++) y[i] = s[i] + h * k3[i];
        dc_acc(y, k4, NULL);
        for (int i = 0; i < 4; i++) s[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        dv += ad * h;
        t += h;
    }
    /* the end: entry interface, 120 km, after the last perigee pass */
    dc.life = (dc.n - 1) * DC_DT;
    for (int i = 0; i < dc.n; i++)
        if (hypot(dc.st[4 * i], dc.st[4 * i + 1]) - R_EARTH < 120) { dc.life = i * DC_DT; break; }
    double L = dc.life, end = (dc.n - 1) * DC_DT;
    Warp w = {8, {0, 6, 18, 58, 72, 80, 86, 90}, {0, 2400, 0.10 * L, 0.90 * L, L - 4 * 3600, L - 1200, end, end}};
    dc.w = w;
}
static void dc_state(double t, double *p, double *v, double *cdv) {
    double f = t / DC_DT;
    int i = (int)f;
    if (i >= dc.n - 1) { i = dc.n - 2; f = i + 1; }
    if (i < 0) { i = 0; f = 0; }
    double u = f - i;
    const float *a = dc.st + 4 * i, *b = a + 4;
    set3(p, a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, 0);
    if (v) set3(v, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u, 0);
    if (cdv) *cdv = dc.cdv[i] + (dc.cdv[i + 1] - dc.cdv[i]) * u;
}

static void dc_update(double u, Info *in) {
    dc.t = warp_at(&dc.w, u, &dc.rate);
    double cdv;
    dc.gone = dc.t >= dc.life;
    if (dc.gone) dc.rate = 1;
    dc.t = fmin(dc.t, dc.life);           /* the last we heard */
    dc_state(dc.t, dc.p, dc.v, &cdv);
    orb_from_rv(MU_EARTH, dc.p, dc.v, dc.t, &dc.osc);
    double r = norm3(dc.p), h = r - R_EARTH, vv = norm3(dc.v);
    in->title = "Orbital decay";
    in->sub = "185 \xc3\x97 800 km, B = 50 kg/m\xc2\xb2: drag eats the apogee, then the spiral down";
    in->sub = "185 \xc3\x97 800 km, ballistic coefficient 50 kg/m\xc2\xb2: drag eats the apogee, then the spiral in";
    in->frame = "GEOCENTRIC, ORBIT PLANE";
    in->eq[0] = "a(D) = \xe2\x88\x92\xc2\xbd \xcf\x81 v\xc2\xb2 / B,   B = m / (C(D) A)";
    in->eq[1] = "\xcf\x81(h) = \xcf\x81\xe2\x82\x80 e^(\xe2\x88\x92(h \xe2\x88\x92 h\xe2\x82\x80)/H)";
    in->eq[2] = "\xce\x94" "a per rev \xe2\x89\x88 \xe2\x88\x92 2\xcf\x80 a\xc2\xb2 \xcf\x81 / B";
    in->eq[3] = "drag paradox: v rises as a falls";
    in->neq = 4;
    in->warp = dc.rate;
    fmt_met(in->clock, sizeof in->clock, dc.t);
    tel(in, "ALTITUDE", "%.1f km", h);
    double hp = dc.osc.a * (1 - dc.osc.e) - R_EARTH, ha = dc.osc.a * (1 + dc.osc.e) - R_EARTH;
    tel(in, "PERIGEE \xc2\xb7 APOGEE", "%.0f \xc2\xb7 %.0f km", hp, ha);
    tel(in, "VELOCITY", "%.3f km/s", vv);
    char b[32];
    fmt_dur(b, sizeof b, dc.osc.a > 0 ? orb_period(&dc.osc) : 0);
    tel(in, "PERIOD", "%s", b);
    double rho = atm_rho(h);
    tel(in, "DENSITY \xcf\x81", "%.2e kg/m\xc2\xb3", rho);
    tel(in, "DRAG", "%.2e m/s\xc2\xb2", 0.5 * rho * vv * vv * 1e6 / dc.B);
    in->dv_head = "DRAG LOSSES  km/s";
    dvrow(in, "\xce\x94v taken by drag so far", cdv, 1);
    double cdv2;
    dc_state(fmin(dc.t + DAY, (dc.n - 1) * DC_DT), b[0] ? dc.p : dc.p, NULL, &cdv2);
    dc_state(dc.t, dc.p, dc.v, &cdv);
    dvrow(in, "\xce\x94v per day to hold the orbit", cdv2 - cdv, 3);
    fmt_dur(b, sizeof b, dc.life);
    dvrow_s(in, "predicted entry (120 km)", b, 2);
    snprintf(b, sizeof b, "%s", dc.gone ? "LOSS OF SIGNAL" : "TRACKING");
    dvrow_s(in, "status", b, dc.gone ? 2 : 3);
}

/* the drawing exaggerates altitude, or a LEO orbit is a hair round the Earth */
#define DC_K 8.0
static void dc_xf(const double *p, double *o) {
    double r = norm3(p), k = (R_EARTH + DC_K * (r - R_EARTH)) / r;
    scale3(p, k, o);
}
static void dc_orbit(const View *v, const Orbit *o, float w, const float *c, float halo) {
    static double pts[3 * 241];
    for (int i = 0; i <= 240; i++) {
        double p[3];
        orb_at_nu(o, TAU_D * i / 240, p, NULL);
        dc_xf(p, pts + 3 * i);
    }
    poly(v, pts, 241, w, c, 0, halo, 1, 1);
}
static void dc_draw(Rect R, double u) {
    View v;
    view_init(&v, R.x0, R.y0, R.x1, R.y1);
    double el = 52 * RAD;
    view_cam(&v, dc.yaw0 + u * 0.005, el);
    double ha = dc.osc.a > 0 ? dc.osc.a * (1 + dc.osc.e) - R_EARTH : 150;
    ha = fmax(ha, 250);
    v.s = fit_scale(&v, (R_EARTH + DC_K * ha) * 1.04, el, 20 * UI);
    double O[3] = {0, 0, 0}, sun[3] = {0.4, -0.85, 0.35};
    view_occ(&v, O, R_EARTH);
    planet(&v, O, R_EARTH, B_EARTH, sun, (float)(dc.t * 7.292115e-5), 4, pal->sea);
    /* shells */
    double sh[] = {120, 200, 400, 800};
    char b[48];
    for (int i = 0; i < 4; i++) {
        double rr = R_EARTH + DC_K * sh[i];
        if (sh[i] > ha * 1.05 && i) continue;
        draw_circle(&v, O, rr, 1, i == 0 ? ca(pal->warn, 0.7f) : ca(pal->faint, 0.9f), i == 0 ? 5 * UI : 0);
        double p[3] = {rr * cos(2.0 + i * 0.25), rr * sin(2.0 + i * 0.25), 0};
        if (hidden(&v, p)) continue;
        float x, y;
        proj(&v, p, &x, &y);
        snprintf(b, sizeof b, i == 0 ? "ENTRY INTERFACE %.0f km" : "%.0f km", sh[i]);
        text_h(st_tiny, x, y - 4 * UI, 0, b, i == 0 ? ca(pal->warn, 0.95f) : ca(pal->dim, 0.9f));
    }
    /* the orbit at earlier times */
    for (int k = 6; k >= 1; k--) {
        double t = dc.t - k * dc.life / 14;
        if (t < 0) continue;
        double p[3], vv[3];
        dc_state(t, p, vv, NULL);
        Orbit o;
        orb_from_rv(MU_EARTH, p, vv, t, &o);
        if (o.e < 1) dc_orbit(&v, &o, 1, ca(pal->orb2, 0.55f - k * 0.06f), 0);
    }
    if (!dc.gone && dc.osc.e < 1) dc_orbit(&v, &dc.osc, 1.4f * UI, ca(pal->orb, 0.55f), 0);
    /* the actual path over the last revolution, hotter as it goes deeper */
    double per = dc.osc.a > 0 ? fmin(orb_period(&dc.osc), 6000) : 5300;
    static double pts[3 * 201];
    int n = 200;
    for (int i = 0; i <= n; i++) {
        double p[3];
        dc_state(dc.t - per + per * i / n, p, NULL, NULL);
        dc_xf(p, pts + 3 * i);
    }
    for (int i = 0; i < n; i++) {
        double h = (norm3(pts + 3 * i) - R_EARTH) / DC_K;
        float hot = (float)clampd((170 - h) / 60, 0, 1);
        const float *c = mixc(pal->orb, pal->warn, hot, 1);
        poly(&v, pts + 3 * i, 2, (1.8f + 1.5f * hot) * UI, c, 0, 3 + 3 * hot, 0.15f + 0.85f * i / n, 0.15f + 0.85f * (i + 1) / n);
    }
    double P[3];
    dc_xf(dc.p, P);
    float x, y;
    proj(&v, P, &x, &y);
    double h = norm3(dc.p) - R_EARTH;
    if (!dc.gone) {
        if (h < 160) glow(x, y, (float)(70 - h * 0.3) * UI, cadd(pal->warn, (float)clampd((160 - h) / 60, 0, 0.9)));
        craft(&v, P, dc.v, h < 150 ? pal->hi : pal->orb, NULL);
    } else {
        float a = (float)clampd(1 - (dc.t - dc.life) / 1200, 0.3, 1);
        glow(x, y, 50 * UI, cadd(pal->warn, 0.6f * a));
        text_h(st_bodyb, x, y - 26 * UI, 0, "LOSS OF SIGNAL", ca(pal->warn, 1));
    }
    snprintf(b, sizeof b, "ALTITUDE DRAWN Ã%.0f", DC_K);
    text(st_tiny, (R.x0 + R.x1) * 0.5f, R.y0 + 30 * UI + 2 * line_h(st_tiny), 0, b, ca(pal->dim, 0.8f));
}

static void dc_plot(Rect r, double u) {
    (void)u;
    Plot p = {r, 0, dc.life / DAY * 1.04, 0, 850, 0, 0};
    pl_frame(&p, "DAYS", "ALTITUDE  km");
    char b[16];
    int step = dc.life / DAY > 30 ? 10 : dc.life / DAY > 12 ? 4 : 2;
    for (int d = 0; d <= dc.life / DAY * 1.04; d += step) { snprintf(b, sizeof b, "%d", d); pl_tick_x(&p, d, b, 1); }
    for (int a = 0; a <= 800; a += 200) { snprintf(b, sizeof b, "%d", a); pl_tick_y(&p, a, b, 1); }
    float ey = pl_y(&p, 120);
    line_d(r.x0, ey, r.x1, ey, 1, ca(pal->warn, 0.7f), 5 * UI);
    text(st_tiny, r.x0 + 6 * UI, ey - 5 * UI, -1, "120 km", ca(pal->warn, 0.9f));
    static double xs[4000], ya[4000], yp[4000];
    int n = 0, cut = 0;
    for (int i = 0; i < dc.ns; i++) {
        if (dc.sd[i] * DAY > dc.life) break;
        xs[n] = dc.sd[i]; ya[n] = dc.sha[i]; yp[n] = dc.shp[i];
        if (dc.sd[i] * DAY <= dc.t) cut = n;
        n++;
    }
    pl_curve(&p, xs, ya, n, 1.2f * UI, ca(pal->orb, 0.35f), 0);
    pl_curve(&p, xs, yp, n, 1.2f * UI, ca(pal->alt, 0.35f), 0);
    pl_curve(&p, xs, ya, cut + 1, 2 * UI, ca(pal->orb, 1), 0);
    pl_curve(&p, xs, yp, cut + 1, 2 * UI, ca(pal->alt, 1), 0);
    float X = pl_x(&p, fmin(dc.t, dc.life) / DAY);
    line(X, r.y0, X, r.y1, 1.2f * UI, ca(pal->hi, 0.8f));
    text(st_tiny, r.x1 - 6 * UI, pl_y(&p, 760), 1, "APOGEE", ca(pal->orb, 1));
    text(st_tiny, r.x1 - 6 * UI, pl_y(&p, 680), 1, "PERIGEE", ca(pal->alt, 1));
}
/* ================================================================== */
/* ---- GL ----                                                        */
/* ================================================================== */
#define GLSL(...) #__VA_ARGS__
static const char *VS_PRIM = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aA;
    layout(location = 1) in vec2 aB;
    layout(location = 2) in vec4 aP;
    layout(location = 3) in vec4 aC;
    layout(location = 4) in vec2 aQ;
    uniform vec2 uRes;
    out vec2 vL;
    out vec4 vP;
    out vec4 vC;
    out vec2 vQ;
    out float vAng;
    flat out int vK;
    void main() {
        int i = gl_VertexID;
        int k = (i == 0 || i == 3) ? 0 : (i == 1) ? 1 : (i == 2 || i == 4) ? 2 : 3;
        vec2 c = vec2((k == 1 || k == 2) ? 1.0 : -1.0, k >= 2 ? 1.0 : -1.0);
        vec2 L = c * aA.zw;
        vK = int(aB.y + 0.5);
        float an = (vK == 6) ? 0.0 : aB.x;
        float cs = cos(an), sn = sin(an);
        vec2 P = aA.xy + vec2(L.x * cs - L.y * sn, L.x * sn + L.y * cs);
        vL = L; vP = aP; vC = aC; vQ = aQ; vAng = aB.x;
        gl_Position = vec4(P.x / uRes.x * 2.0 - 1.0, 1.0 - P.y / uRes.y * 2.0, 0.0, 1.0);
    });

static const char *FS_PRIM = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas, uLand, uPork;
    uniform vec3 uLandC, uBg, uBg2, uInk, uFaint, uAlt, uHi;
    uniform float uPaper;
    in vec2 vL;
    in vec4 vP;
    in vec4 vC;
    in vec2 vQ;
    in float vAng;
    flat in int vK;
    out vec4 o;
    float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
    float sdb(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
    float gridline(float v, float step) {
        float f = v / step, d = abs(fract(f + 0.5) - 0.5) / max(fwidth(f), 1e-5);
        return 1.0 - smoothstep(0.3, 1.2, d);
    }
    float landAt(float lon, float lat) {
        return texture(uLand, vec2(lon / 6.2831853 + 0.5, 0.5 - lat / 3.14159265)).r;
    }
    void main() {
        vec3 c = vC.rgb;
        float a = 1.0;
        if (vK == 0) {
            float hw = vP.y, t = max(hw, 0.5);
            float d = length(vec2(max(abs(vL.x) - vP.x, 0.0), vL.y)) - t;
            a = clamp(0.5 - d, 0.0, 1.0) * min(hw * 2.0, 1.0);
            if (vP.z > 0.0) {
                float u = mod(vL.x + vP.x - vP.w, vP.z), on = vP.z * 0.55;
                a *= clamp(0.5 - max(-u, u - on), 0.0, 1.0) + clamp(u - vP.z + 0.5, 0.0, 1.0);
            }
        } else if (vK == 1) {
            float d = length(vec2(max(abs(vL.x) - vP.x, 0.0), vL.y)) / vP.y;
            a = exp(-d * d * 3.5) * (1.0 - smoothstep(0.8, 1.0, d));
        } else if (vK == 2) {
            float r = length(vL);
            if (vP.y > 0.0) a = clamp(0.5 - (abs(r - vP.x) - max(vP.y, 0.5)), 0.0, 1.0) * min(vP.y * 2.0, 1.0);
            else a = clamp(0.5 - (r - vP.x), 0.0, 1.0);
            if (vP.z > 0.0) {
                float f = fract(atan(vL.y, vL.x) / 6.2831853 * vP.z);
                a *= smoothstep(0.0, 0.08, f) * (1.0 - smoothstep(0.5, 0.58, f));
            }
        } else if (vK == 3) {
            a = texture(uAtlas, (vP.xy + vL) / vec2(textureSize(uAtlas, 0))).r;
        } else if (vK == 4) {
            float d = sdb(vL, vP.xy, vP.z);
            if (vP.w > 0.0) a = clamp(0.5 - (abs(d + vP.w * 0.5) - vP.w * 0.5), 0.0, 1.0);
            else a = clamp(0.5 - d, 0.0, 1.0);
        } else if (vK == 5) {
            float r = length(vL) / vP.x;
            a = exp(-r * r * 4.0) * (1.0 - smoothstep(0.85, 1.0, r));
        } else if (vK == 6) {
            /* a planet: lit from vP.yzw (view space), its pole the world's z */
            float r = vP.x, dist = length(vL);
            vec2 q = vL / r;
            float d2 = dot(q, q);
            int ty = int(vQ.x + 0.5);
            vec3 sun = vP.yzw;
            if (dist > r) {
                /* the atmosphere's rim */
                float h = (dist - r) / (r * 0.06 + 2.0);
                float lit = clamp(dot(normalize(vec3(q, 0.0)), sun) * 0.8 + 0.4, 0.0, 1.0);
                a = (ty == 0 && uPaper < 0.5) ? exp(-h * 3.0) * 0.55 * lit : 0.0;
                c = mix(c, vec3(0.55, 0.8, 1.0), 0.6);
            } else {
                a = clamp(0.5 - (dist - r), 0.0, 1.0);
                vec3 n = vec3(q, sqrt(max(1.0 - d2, 0.0)));
                float se = sin(vAng), ce = cos(vAng);
                vec3 nw = vec3(se * n.y + ce * n.z, n.x, -ce * n.y + se * n.z);
                float lon = atan(nw.y, nw.x) - vQ.y, lat = asin(clamp(nw.z, -1.0, 1.0));
                lon = mod(lon + 3.14159265, 6.2831853) - 3.14159265;
                vec3 base = c;
                float lam = dot(n, sun);
                if (ty == 0) {
                    float l = landAt(lon, lat);
                    base = mix(c, uLandC, smoothstep(0.3, 0.7, l));
                    base = mix(base, vec3(0.85), smoothstep(1.3, 1.42, abs(lat)) * (1.0 - uPaper) * 0.5);
                    float sp = pow(max(dot(reflect(-sun, n), vec3(0, 0, 1)), 0.0), 24.0) * (1.0 - l);
                    base += vec3(0.35, 0.45, 0.55) * sp * (1.0 - uPaper);
                } else if (ty == 3) {
                    base = c * (0.82 + 0.18 * sin(lat * 17.0 + sin(lat * 5.0) * 1.4)) * (0.92 + 0.08 * sin(lat * 41.0));
                } else if (ty == 2) {
                    base = c * (0.88 + 0.12 * sin(lon * 3.0 + lat * 5.0) * sin(lat * 4.0));
                    base = mix(base, vec3(0.95), smoothstep(1.25, 1.35, abs(lat)));
                } else if (ty == 1) {
                    base = c * (0.85 + 0.15 * sin(lon * 7.0) * sin(lat * 6.0));
                }
                if (ty == 4) {
                    c = c * (1.15 - 0.35 * d2);
                } else {
                    float day = smoothstep(-0.06, 0.14, lam);
                    float amb = uPaper > 0.5 ? 0.55 : 0.09;
                    c = base * (amb + (1.0 - amb) * day * (0.45 + 0.55 * max(lam, 0.0)));
                    c *= 0.78 + 0.22 * n.z;
                    if (ty == 0 || ty == 2) {
                        float g = max(gridline(lon, 0.5235988), gridline(lat, 0.5235988));
                        c = mix(c, uInk, g * (0.03 + 0.09 * day));
                    }
                    if (uPaper > 0.5) c = mix(c, uInk, (1.0 - smoothstep(0.5, 1.0, dist / r)) * 0.0 + smoothstep(r - 1.5, r, dist) * 0.8);
                }
            }
        } else if (vK == 7) {
            /* the map: land, sea, night, a graticule */
            float lon = vL.x / vP.x * 3.14159265, lat = -vL.y / vP.y * 1.5707963;
            float l = landAt(lon, lat);
            vec3 base = mix(c, uLandC, smoothstep(0.3, 0.7, l));
            float cz = sin(lat) * sin(vQ.y) + cos(lat) * cos(vQ.y) * cos(lon - vQ.x);
            float day = smoothstep(-0.10, 0.10, cz);
            if (uPaper > 0.5) {
                c = base * (0.86 + 0.14 * day);
                float hatch = step(fract((gl_FragCoord.x - gl_FragCoord.y) / 6.0), 0.2);
                c = mix(c, uInk, (1.0 - day) * hatch * 0.18);
            } else c = base * (0.30 + 0.70 * day);
            float g = max(gridline(lon, 0.5235988), gridline(lat, 0.5235988));
            c = mix(c, uInk, g * 0.13);
            float eq = 1.0 - smoothstep(0.5, 1.5, abs(vL.y) );
            c = mix(c, uInk, eq * 0.18);
        } else if (vK == 8) {
            /* the zero-velocity curve: 2 Omega(x, y) = C, the realm where it's below hatched */
            vec2 w = vP.xy + vec2(vL.x, -vL.y) * vP.z;
            float m = vQ.x;
            float r1 = length(w - vec2(-m, 0.0)), r2 = length(w - vec2(1.0 - m, 0.0));
            float f = dot(w, w) + 2.0 * (1.0 - m) / r1 + 2.0 * m / r2 - vP.w;
            float px = f / max(fwidth(f), 1e-7);
            float ln = 1.0 - smoothstep(0.6, 1.6, abs(px));
            float forb = smoothstep(0.5, -0.5, px);
            float hatch = 1.0 - smoothstep(0.5, 1.4, abs(mod(gl_FragCoord.x + gl_FragCoord.y, 9.0) - 4.5) * 1.0 - 3.0);
            hatch = step(mod(gl_FragCoord.x + gl_FragCoord.y, 9.0), 1.6);
            a = max(ln * 0.95, forb * (0.10 + 0.22 * hatch));
            /* a few faint level curves of the potential outside */
            float lv = log(max(f + vP.w, 1e-3)) * 24.0;
            a = max(a, (1.0 - forb) * (1.0 - smoothstep(0.0, 1.0, abs(fract(lv + 0.5) - 0.5) / max(fwidth(lv), 1e-5))) * 0.06);
        } else if (vK == 9) {
            /* the porkchop: C3 contours (relative to the minimum) over a fill, v-infinity dashed */
            vec2 uv = vec2(vL.x / vP.x * 0.5 + 0.5, 0.5 - vL.y / vP.y * 0.5);
            vec2 g = texture(uPork, uv).rg;
            float c3 = g.r;
            a = step(uv.x, vQ.x) * step(0.5, c3) * (1.0 - smoothstep(160.0, 420.0, c3));
            float l = log(max(c3, 0.5) / vP.z) / log(1.22);
            float dl = max(fwidth(l), 1e-4);
            float f = abs(fract(l + 0.5) - 0.5) / dl;
            float ln = (1.0 - smoothstep(0.35, 1.2, f)) * step(l, 13.5);
            float major = step(abs(mod(floor(l + 0.5), 3.0)), 0.5);
            float t = clamp(1.0 - l / 14.0, 0.0, 1.0);
            vec3 fill = mix(uBg2, c, pow(t, 1.8) * 0.42);
            fill = mix(fill, uHi, smoothstep(0.86, 1.0, t) * 0.30);
            vec3 col = mix(fill, mix(c, uInk, 0.25), ln * (0.35 + 0.45 * major));
            float vl = g.g, vf = abs(fract(vl + 0.5) - 0.5) / max(fwidth(vl), 1e-4);
            float dash = step(fract((gl_FragCoord.x + gl_FragCoord.y) / 9.0), 0.5);
            col = mix(col, uAlt, (1.0 - smoothstep(0.35, 1.1, vf)) * dash * 0.6 * step(g.g, 6.5) * step(l, 13.5));
            c = col;
            a *= 0.96;
        } else if (vK == 10) {
            vec2 p = vL / vP.x;
            float d = max(-p.x - 0.5, abs(p.y) * 0.866 + p.x * 0.5 - 0.5) * vP.x;
            if (vP.y > 0.0) a = clamp(0.5 - (abs(d + vP.y * 0.5) - vP.y * 0.5), 0.0, 1.0);
            else a = clamp(0.5 - d, 0.0, 1.0);
        }
        float al = abs(vC.a) * a;
        o = vec4(c * al, vC.a < 0.0 ? 0.0 : al);
    });

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });
/* the backdrop: deep space and its stars, or a sheet of plotting paper */
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform vec2 uRes, uOff;
    uniform vec3 uBg, uBg2, uStar, uGrid;
    uniform float uPaper;
    out vec4 o;
    float h21(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
    float stars(vec2 p, float cell, float dens) {
        vec2 c = floor(p / cell), f = p - c * cell;
        float h = h21(c);
        if (h < 1.0 - dens) return 0.0;
        vec2 at = vec2(h21(c + 17.1), h21(c + 31.7)) * (cell - 4.0) + 2.0;
        float r = length(f - at), b = pow(h21(c + 5.3), 6.0);
        return exp(-r * r * 1.6) * (0.25 + 0.95 * b);
    }
    void main() {
        vec2 p = gl_FragCoord.xy, q = p / uRes;
        float v = length((q - 0.5) * vec2(uRes.x / uRes.y, 1.0));
        vec3 c = mix(uBg2, uBg, smoothstep(0.0, 0.9, v));
        vec2 w = vec2(p.x, uRes.y - p.y) + uOff;
        if (uPaper > 0.5) {
            float n = h21(floor(w * 0.7)) * 0.025 + h21(floor(w * 0.11)) * 0.02;
            c = uBg * (0.985 + n);
            float g1 = max(step(mod(w.x, 24.0), 1.0), step(mod(w.y, 24.0), 1.0));
            float g2 = max(step(mod(w.x, 120.0), 1.0), step(mod(w.y, 120.0), 1.0));
            c = mix(c, uGrid, g1 * 0.10 + g2 * 0.14);
        } else {
            float s = stars(w, 7.0, 0.045) * 0.55 + stars(w + 1000.0, 23.0, 0.12) + stars(w + 77.0, 61.0, 0.18) * 1.3;
            c += uStar * s;
        }
        o = vec4(c, 1.0);
    });

static GLuint prog_prim, prog_bg, vao_prim, vbo_prim, vao_full, tex_atlas, tex_land, target_fbo;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "orbits: shader:\n%s\n", log);
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
    prog_prim = mk_prog(VS_PRIM, FS_PRIM);
    prog_bg = mk_prog(VS_FULL, FS_BG);
    glGenVertexArrays(1, &vao_full);
    glGenVertexArrays(1, &vao_prim);
    glGenBuffers(1, &vbo_prim);
    glBindVertexArray(vao_prim);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    int off[5] = {0, 4, 6, 10, 14}, sz[5] = {4, 2, 4, 4, 2};
    for (int i = 0; i < 5; i++) {
        glVertexAttribPointer((GLuint)i, sz[i], GL_FLOAT, GL_FALSE, IF * sizeof(float), (void *)(off[i] * sizeof(float)));
        glVertexAttribDivisor((GLuint)i, 1);
        glEnableVertexAttribArray((GLuint)i);
    }
    glGenTextures(1, &tex_atlas);
    glGenTextures(1, &tex_land);
    glGenTextures(1, &tex_pork);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_land);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, LW, LH, 0, GL_RED, GL_UNSIGNED_BYTE, land);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, tex_pork);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, PCX, PCY, 0, GL_RG, GL_FLOAT, NULL);
}

static void upload_atlas(void) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    atlas_dirty = 0;
}

/* draw what's queued, clipped to a rect (or not) */
static void flush(const Rect *clip) {
    if (!ibn) return;
    if (clip) {
        glEnable(GL_SCISSOR_TEST);
        glScissor((GLint)clip->x0, (GLint)(H - clip->y1), (GLsizei)(clip->x1 - clip->x0), (GLsizei)(clip->y1 - clip->y0));
    }
    GLuint p = prog_prim;
    glUseProgram(p);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform1i(U(p, "uAtlas"), 0);
    glUniform1i(U(p, "uLand"), 1);
    glUniform1i(U(p, "uPork"), 2);
    glUniform3fv(U(p, "uLandC"), 1, pal->land);
    glUniform3fv(U(p, "uBg"), 1, pal->bg);
    glUniform3fv(U(p, "uBg2"), 1, pal->bg2);
    glUniform3fv(U(p, "uInk"), 1, pal->ink);
    glUniform3fv(U(p, "uFaint"), 1, pal->faint);
    glUniform3fv(U(p, "uAlt"), 1, pal->alt);
    glUniform3fv(U(p, "uHi"), 1, pal->hi);
    glUniform1f(U(p, "uPaper"), (float)pal->paper);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_pork);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_land);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_prim);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)ibn * IF * sizeof(float)), ib, GL_STREAM_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, ibn);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    ibn = 0;
}

/* ================================================================== */
/* ---- the acts, scheduled; the panels; a frame ----                  */
/* ================================================================== */
static const char *ACT_TAG[NACT] = {"ORBIT TRANSFER", "INTERPLANETARY", "FLYBY", "RESTRICTED THREE-BODY", "CONSTELLATION", "REENTRY"};
static Info info, info_side;
static int side_act, shot_mode;

static int side_act_for(int a, int k) {
    int s = a == A_CR3BP ? A_WALKER : a == A_WALKER ? A_CR3BP : (k % 2 ? A_CR3BP : A_WALKER);
    if (side_rank % 2) s = s == A_CR3BP ? A_WALKER : A_CR3BP;
    return s;
}

static void act_init(int a, uint32_t seed, double wt0) {
    switch (a) {
    case A_HOHMANN: ho_init(seed); break;
    case A_MARS: mx_init(seed, wt0); break;
    case A_FLYBY: fb_init(seed); break;
    case A_CR3BP: cr_init(seed); break;
    case A_WALKER: wk_init(seed, wt0); break;
    case A_DECAY: dc.yaw0 = hrand(seed, 13) * TAU_D; break;
    }
}
static void act_update(int a, double u, Info *in) {
    memset(in, 0, sizeof *in);
    switch (a) {
    case A_HOHMANN: ho_update(u, in); break;
    case A_MARS: mx_update(u, in, shot_mode); break;
    case A_FLYBY: fb_update(u, in); break;
    case A_CR3BP: cr_update(u, in); break;
    case A_WALKER: wk_update(u, in); break;
    case A_DECAY: dc_update(u, in); break;
    }
}
static void act_draw(int a, Rect r, double u) {
    switch (a) {
    case A_HOHMANN: ho_draw(r, u); break;
    case A_MARS: mx_draw(r, u); break;
    case A_FLYBY: fb_draw(r, u); break;
    case A_CR3BP: cr_draw(r, u); break;
    case A_WALKER: wk_draw(r, u); break;
    case A_DECAY: dc_draw(r, u); break;
    }
}
static int act_has_plot(int a) { return a != A_FLYBY; }
static void act_plot(int a, Rect r, double u) {
    switch (a) {
    case A_HOHMANN: ho_plot(r, u); break;
    case A_MARS: mx_plot(r, u); break;
    case A_FLYBY: fb_vectors(r); break;
    case A_CR3BP: {
        /* a closer look: the L1 neck, or the tadpole */
        Rect q = r;
        float w = r.x1 - r.x0, h = r.y1 - r.y0;
        (void)w; (void)h;
        cr_zoom = cr_zoom_at(u) > 0.5 ? 0 : 1;
        cr_draw(q, u);
        cr_zoom = -1;
        break;
    }
    case A_WALKER: wk_plot(r, u); break;
    case A_DECAY: dc_plot(r, u); break;
    }
}
static const char *act_plot_title(int a) {
    if (a == A_CR3BP) return cr.ph ? "TADPOLE AND HORSESHOE" : cr_zoom_at(act_u) > 0.5 ? "THE WHOLE SYSTEM" : "THE L1 NECK";
    static const char *t[NACT] = {"\xce\x94v AGAINST RADIUS RATIO", "PORKCHOP  \xc2\xb7  LAMBERT SOLVER", "VELOCITY DIAGRAM",
                                  "THE L1 NECK", "WALKER PATTERN  \xce\xa9 \xc3\x97 u", "APOGEE AND PERIGEE"};
    return t[a];
}

static void act_at(double wt) {
    double el = wt - t_wall0;
    if (el < 0) el = 0;
    int k = (int)floor(el / ACT_LEN);
    act_u = el - k * ACT_LEN;
    if (k != act_k) {
        int start = opt_act >= 0 ? opt_act : (int)(opt_seed % NACT);
        act_k = k;
        act = (start + k) % NACT;
        act_seed = hash32(opt_seed ^ (uint32_t)k * 0x9E3779B9u);
        act_wt0 = t_wall0 + k * ACT_LEN;
        if (role == R_SIDE) {
            side_act = side_act_for(act, k);
            act_init(side_act, act_seed, act_wt0);
        } else act_init(act, act_seed, act_wt0);
    }
    act_alpha = (float)(smooth(0, 1.4, act_u) * (1 - smooth(ACT_LEN - 1.4, ACT_LEN, act_u)));
}

/* ---- the panels ---- */
static void card(Rect r) {
    rect(r.x0, r.y0, r.x1, r.y1, 3 * UI, ca(pal->bg2, pal->paper ? 0.92f : 0.88f));
    rect_o(r.x0, r.y0, r.x1, r.y1, 3 * UI, 1, ca(pal->faint, 0.9f));
    float t = 9 * UI;
    const float *c = ca(pal->dim, 0.9f);
    line(r.x0, r.y0, r.x0 + t, r.y0, 1.5f * UI, c); line(r.x0, r.y0, r.x0, r.y0 + t, 1.5f * UI, c);
    line(r.x1, r.y1, r.x1 - t, r.y1, 1.5f * UI, c); line(r.x1, r.y1, r.x1, r.y1 - t, 1.5f * UI, c);
}
/* a section header with a hairline; returns the y below it */
static float header(float x0, float x1, float y, const char *s, const char *right) {
    y += cap_h(st_h);
    text_t(st_h, x0, y, -1, s, ca(pal->dim, 1), 1.2f * UI);
    if (right) text_t(st_h, x1, y, 1, right, ca(pal->hi, 0.95f), 1 * UI);
    y += 7 * UI;
    line(x0, y, x1, y, 1, ca(pal->faint, 1));
    return y + 8 * UI;
}
/* wrapped text; returns the y below it */
static float wrap(int s, float x0, float x1, float y, const char *str, const float *c) {
    char buf[256], ln[256];
    snprintf(buf, sizeof buf, "%s", str);
    float lh = line_h(s);
    char *p = buf;
    ln[0] = 0;
    y += cap_h(s);
    while (*p) {
        char *sp = strchr(p, ' ');
        size_t n = sp ? (size_t)(sp - p) : strlen(p);
        char cand[256];
        snprintf(cand, sizeof cand, "%s%s%.*s", ln, ln[0] ? " " : "", (int)n, p);
        if (ln[0] && text_width(s, cand, 0) > x1 - x0) {
            text(s, x0, y, -1, ln, c);
            y += lh;
            snprintf(ln, sizeof ln, "%.*s", (int)n, p);
        } else snprintf(ln, sizeof ln, "%s", cand);
        p += n;
        while (*p == ' ') p++;
    }
    if (ln[0]) { text(s, x0, y, -1, ln, c); y += lh; }
    return y - cap_h(s);
}

static float title_block(float x0, float x1, float y, const Info *in, int a) {
    char b[64];
    snprintf(b, sizeof b, "%02d / %02d   %s", a + 1, NACT, ACT_TAG[a]);
    y += cap_h(st_h);
    text_t(st_h, x0, y, -1, b, ca(pal->hi, 1), 1.5f * UI);
    y += 14 * UI + cap_h(st_title);
    text(st_title, x0, y, -1, in->title, ca(pal->ink, 1));
    y += 12 * UI;
    y = wrap(st_sub, x0, x1, y, in->sub, ca(pal->dim, 1));
    return y;
}
static float eq_block(float x0, float x1, float y, const Info *in) {
    y = header(x0, x1, y, "KEY EQUATIONS", NULL);
    for (int i = 0; i < in->neq; i++) {
        y += cap_h(st_eq);
        text(st_eq, x0, y, -1, in->eq[i], ca(pal->ink, 0.95f));
        y += line_h(st_eq) * 0.62f;
    }
    return y;
}
static float dv_block(float x0, float x1, float y, const Info *in) {
    y = header(x0, x1, y, in->dv_head ? in->dv_head : "\xce\x94v BUDGET", NULL);
    float lh = line_h(st_small) * 1.12f;
    for (int i = 0; i < in->ndv; i++) {
        const DvRow *d = &in->dv[i];
        if (d->st == 2) { line(x0, y - 2 * UI, x1, y - 2 * UI, 1, ca(pal->faint, 1)); y += 4 * UI; }
        float by = y + cap_h(st_small);
        const float *c = d->st == 3 ? ca(pal->dim, 1) : ca(pal->ink, 1);
        if (d->st == 1) disc(x0 + 3 * UI, by - cap_h(st_small) * 0.5f, 2.6f * UI, ca(pal->hi, 1));
        else if (d->st == 0) ring(x0 + 3 * UI, by - cap_h(st_small) * 0.5f, 2.6f * UI, 1, ca(pal->dim, 1));
        text(d->st == 2 ? st_bodyb : st_small, x0 + 12 * UI, by, -1, d->name, c);
        text(d->st == 2 ? st_numb : st_num, x1, by, 1, d->val, d->st == 2 ? ca(pal->hi, 1) : c);
        y += lh;
    }
    return y;
}
static float tel_block(float x0, float x1, float y, const Info *in, int cols) {
    y = header(x0, x1, y, "TELEMETRY", in->frame);
    float cw = (x1 - x0) / cols, lh = line_h(st_num) * 1.0f;
    int rows = (in->ntel + cols - 1) / cols;
    for (int i = 0; i < in->ntel; i++) {
        int c = i / rows, r = i % rows;
        float x = x0 + c * cw, yy = y + r * lh + cap_h(st_num);
        text(st_tiny, x, yy, -1, in->tel[i].k, ca(pal->dim, 1));
        text(st_num, x + cw - (c < cols - 1 ? 16 * UI : 0), yy, 1, in->tel[i].v, ca(pal->ink, 1));
    }
    return y + rows * lh;
}
static float dv_height(const Info *in) { return 38 * UI + in->ndv * line_h(st_small) * 1.12f + 8 * UI; }
static float tel_height(const Info *in, int cols) { return 38 * UI + (in->ntel + cols - 1) / cols * line_h(st_num); }
static float eq_height(const Info *in) { return 38 * UI + in->neq * (cap_h(st_eq) + line_h(st_eq) * 0.62f); }

/* the mission clock and the time warp: a log scale, 1 to 10^7 */
static void warp_block(float cx, float y, const Info *in, float w) {
    float x0 = cx - w * 0.5f, x1 = cx + w * 0.5f;
    text(st_numb, cx, y, 0, in->clock, ca(pal->ink, 1));
    y += 14 * UI;
    for (int d = 0; d <= 7; d++) {
        float x = x0 + (x1 - x0) * d / 7.0f;
        line(x, y, x, y + (d % 2 ? 4 : 7) * UI, 1, ca(pal->dim, 0.9f));
        for (int m = 2; m < 10 && d < 7; m++) {
            float xm = x + (x1 - x0) / 7.0f * log10f((float)m);
            line(xm, y, xm, y + 2.5f * UI, 1, ca(pal->faint, 1));
        }
    }
    line(x0, y, x1, y, 1, ca(pal->dim, 0.9f));
    float wv = (float)clampd(log10(fmax(in->warp, 1)), 0, 7);
    float wx = x0 + (x1 - x0) * wv / 7.0f;
    line(x0, y, wx, y, 2 * UI, ca(pal->hi, 1));
    tri(wx, y - 6 * UI, PI_D / 2, 5 * UI, ca(pal->hi, 1), 0);
    char b[48], k[32];
    fmt_k(k, sizeof k, in->warp);
    snprintf(b, sizeof b, "TIME WARP  \xc3\x97%s", k);
    text(st_tiny, cx, y + 9 * UI + cap_h(st_tiny), 0, b, ca(pal->hi, 1));
}

/* the lock: the time, how long away, and whose mission */
static void lock_block(float x1, float y, double wt, int align) {
    float sv = alpha_mul;
    alpha_mul = 1;
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    char b[64], d[64];
    strftime(b, sizeof b, "%H:%M", &tm);
    y += cap_h(st_clock);
    text(st_clock, x1, y, align, b, ca(pal->ink, 1));
    y += 12 * UI + cap_h(st_lab);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    strftime(d, sizeof d, "%a %d %b", &tm);
    for (char *c = d; *c; c++) *c = (char)toupper((unsigned char)*c);
    snprintf(b, sizeof b, "%s  \xc2\xb7  AWAY %ld:%02ld:%02ld", d, away / 3600, away / 60 % 60, away % 60);
    text(st_lab, x1, y, align, b, ca(pal->dim, 1));
    y += line_h(st_lab);
    snprintf(b, sizeof b, "MISSION %s", mission_name);
    text_t(st_lab, x1, y, align, b, ca(pal->hi, 1), 1.5f * UI);
    alpha_mul = sv;
}

/* a frame of the front output: the act's main view with its cards */
static void render_front(double wt) {
    int port = H > W;
    int solo = !has_panel;
    g_solo = solo;
    float m = 30 * UI;
    const Info *in = &info;
    if (!port) {
        float cw = clampf(W * 0.25f, 330 * UI, 470 * UI);
        Rect view = {cw + 10 * UI, 0, (float)W, (float)H};
        if (solo && act_has_plot(act) && act != A_WALKER) { view.x1 = W - clampf(W * 0.27f, 300 * UI, 560 * UI) * 0.3f; view.y1 = H - clampf(H * 0.36f, 220 * UI, 460 * UI) * 0.3f; }
        int inset = solo && act_has_plot(act);
        act_draw(act, view, act_u);
        flush(NULL);
        /* left column */
        float x0 = m, x1 = cw - 10 * UI, y = m;
        float colh = 0;
        (void)colh;
        Rect cr0 = {x0 - 14 * UI, y - 14 * UI, x1 + 14 * UI, 0};
        float ty = title_block(x0, x1, y, in, act);
        float ey = ty + 22 * UI;
        float bottom = ey + eq_height(in);
        if (solo) bottom += 18 * UI + dv_height(in) + 18 * UI + tel_height(in, 1);
        cr0.y1 = bottom + 4 * UI;
        if (cr0.y1 > H - m) cr0.y1 = H - m;
        card(cr0);
        title_block(x0, x1, y, in, act);
        float yy = eq_block(x0, x1, ey, in);
        if (solo) {
            yy = dv_block(x0, x1, yy + 18 * UI, in);
            tel_block(x0, x1, yy + 18 * UI, in, 1);
        }
        if (inset) {
            float pw = clampf(W * 0.27f, 300 * UI, 560 * UI), ph = clampf(H * 0.36f, 220 * UI, 460 * UI);
            if (act == A_MARS) { pw *= 1.12f; ph *= 1.12f; }
            Rect pr = {W - m - pw, H - m - ph, W - m, H - m};
            card(pr);
            header(pr.x0 + 14 * UI, pr.x1 - 14 * UI, pr.y0 + 12 * UI, act_plot_title(act), NULL);
            Rect inner = {pr.x0 + 58 * UI, pr.y0 + 52 * UI, pr.x1 - 18 * UI, pr.y1 - 44 * UI};
            flush(NULL);
            act_plot(act, inner, act_u);
            flush(&pr);
        }
        float wx = view.x0 + (W - view.x0) * (inset ? 0.38f : 0.5f);
        warp_block(wx, H - m - 46 * UI, in, 330 * UI);
        lock_block(W - m, m, wt, 1);
        if (!solo) text(st_tiny, wx, H - m - 46 * UI - line_h(st_numb) * 1.3f, 0, in->frame, ca(pal->dim, 0.8f));
    } else {
        /* portrait: title, the view, then the tables */
        float x0 = m, x1 = W - m, y = m;
        float ty = title_block(x0, x1 - 300 * UI, y, in, act);
        lock_block(x1, y, wt, 1);
        float vy0 = fmaxf(ty, y + 120 * UI) + 10 * UI;
        float vy1 = H * 0.60f;
        Rect view = {0, vy0, (float)W, vy1};
        flush(NULL);
        act_draw(act, view, act_u);
        flush(&view);
        float yy = vy1 + 6 * UI;
        warp_block(W * 0.5f, yy + cap_h(st_numb), in, 330 * UI);
        yy += 70 * UI;
        float mid = W * 0.5f;
        Rect cl = {m - 14 * UI, yy, mid - 10 * UI, H - m};
        Rect crr = {mid + 10 * UI, yy, W - m + 14 * UI, H - m};
        card(cl);
        card(crr);
        float ly = eq_block(x0, mid - 24 * UI, yy + 14 * UI, in);
        ly = dv_block(x0, mid - 24 * UI, ly + 16 * UI, in);
        float ry = yy + 14 * UI;
        if (act_has_plot(act) && act != A_CR3BP) {
            float ph = fminf((crr.x1 - crr.x0) * 0.8f, (H - m - ry) * 0.55f);
            ry = header(mid + 24 * UI, x1, ry, act_plot_title(act), NULL);
            Rect inner = {mid + 70 * UI, ry + 14 * UI, x1, ry + ph};
            flush(NULL);
            act_plot(act, inner, act_u);
            flush(&crr);
            ry = inner.y1 + 46 * UI;
        }
        tel_block(mid + 24 * UI, x1, ry, in, 1);
    }
}

/* the portrait panel beside the front: plot, budget, telemetry */
static void render_panel(double wt) {
    (void)wt;
    float m = 34 * UI, x0 = m, x1 = W - m;
    const Info *in = &info;
    float y = m;
    char b[96];
    snprintf(b, sizeof b, "FLIGHT DYNAMICS  \xc2\xb7  MISSION %s", mission_name);
    y += cap_h(st_h);
    text_t(st_h, x0, y, -1, b, ca(pal->hi, 1), 1.5f * UI);
    snprintf(b, sizeof b, "%02d / %02d", act + 1, NACT);
    text_t(st_h, x1, y, 1, b, ca(pal->dim, 1), 1.5f * UI);
    y += 16 * UI + cap_h(st_title);
    text(st_title, x0, y, -1, in->title, ca(pal->ink, 1));
    y += 14 * UI;
    y = wrap(st_sub, x0, x1, y, in->sub, ca(pal->dim, 1)) + 18 * UI;
    /* the plot, big */
    float ph = fminf((x1 - x0) * (act == A_MARS ? 1.0f : 0.82f), H * 0.44f);
    Rect pc = {x0 - 14 * UI, y, x1 + 14 * UI, y + ph};
    card(pc);
    float hy = header(x0, x1, y + 14 * UI, act_plot_title(act), NULL);
    Rect inner = {x0 + 52 * UI, hy + 22 * UI, x1 - 8 * UI, pc.y1 - 50 * UI};
    if (act == A_FLYBY || act == A_CR3BP) inner = (Rect){x0, hy + 6 * UI, x1, pc.y1 - 12 * UI};
    flush(NULL);
    act_plot(act, inner, act_u);
    flush(&pc);
    y = pc.y1 + 22 * UI;
    float bh = dv_height(in), th = tel_height(in, 2), eh = eq_height(in);
    Rect c1 = {x0 - 14 * UI, y, x1 + 14 * UI, y + bh + 20 * UI};
    card(c1);
    dv_block(x0, x1, y + 14 * UI, in);
    y = c1.y1 + 22 * UI;
    Rect c2 = {x0 - 14 * UI, y, x1 + 14 * UI, y + th + 20 * UI};
    card(c2);
    tel_block(x0, x1, y + 14 * UI, in, 2);
    y = c2.y1 + 22 * UI;
    if (y + eh + 20 * UI < H - m) {
        Rect c3 = {x0 - 14 * UI, y, x1 + 14 * UI, y + eh + 20 * UI};
        card(c3);
        eq_block(x0, x1, y + 14 * UI, in);
        y = c3.y1 + 22 * UI;
    }
    if (y + 80 * UI < H - m) warp_block(W * 0.5f, H - m - 50 * UI, in, fminf(W * 0.6f, 420 * UI));
}

/* the side output: the ground track or the rotating frame, whole */
static void render_side(double wt) {
    (void)wt;
    float m = 30 * UI;
    Rect view = {0, 0, (float)W, (float)H};
    const Info *in = &info_side;
    if (side_act == A_WALKER) {
        float gw = W * 0.30f;
        wk_globe((Rect){0, H * 0.18f, gw, H * 0.82f}, act_u);
        flush(NULL);
        wk_map((Rect){gw + 20 * UI, 120 * UI, W - m, H - 60 * UI}, act_u);
    } else act_draw(side_act, view, act_u);
    flush(NULL);
    float y = m;
    char b[96];
    snprintf(b, sizeof b, "%s   \xc2\xb7   %s", ACT_TAG[side_act], in->frame);
    y += cap_h(st_h);
    text_t(st_h, m, y, -1, b, ca(pal->hi, 1), 1.5f * UI);
    y += 12 * UI + cap_h(st_bodyb);
    text(st_bodyb, m, y, -1, in->title, ca(pal->ink, 1));
    y += 8 * UI;
    wrap(st_small, m, fminf(W * 0.45f, 700 * UI), y, in->sub, ca(pal->dim, 1));
    warp_block(W - m - 170 * UI, m + cap_h(st_numb), in, 300 * UI);
}

static int lay_w, lay_h;
static void render(double wt) {
    if (W != lay_w || H != lay_h) { lay_w = W; lay_h = H; styles_init(); }
    act_at(wt);
    if (atlas_dirty) upload_atlas();
    glBindFramebuffer(GL_FRAMEBUFFER, target_fbo);
    glViewport(0, 0, W, H);
    glUseProgram(prog_bg);
    glUniform2f(U(prog_bg, "uRes"), (float)W, (float)H);
    glUniform2f(U(prog_bg, "uOff"), span_me >= 0 ? span[span_me][0] * 1.37f : 0, span_me >= 0 ? span[span_me][1] * 1.37f : 0);
    glUniform3fv(U(prog_bg, "uBg"), 1, pal->bg);
    glUniform3fv(U(prog_bg, "uBg2"), 1, pal->bg2);
    float star[3] = {0.8f, 0.85f, 0.95f};
    if (!strcmp(pal->name, "phosphor")) { star[0] = 0.35f; star[1] = 0.9f; star[2] = 0.5f; }
    glUniform3fv(U(prog_bg, "uStar"), 1, star);
    glUniform3fv(U(prog_bg, "uGrid"), 1, pal->orb);
    glUniform1f(U(prog_bg, "uPaper"), (float)pal->paper);
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ibn = 0;
    alpha_mul = act_alpha;
    if (role == R_SIDE) {
        act_update(side_act, act_u, &info_side);
        render_side(wt);
    } else {
        act_update(act, act_u, &info);
        if (role == R_PANEL) render_panel(wt);
        else render_front(wt);
    }
    flush(NULL);
    alpha_mul = 1;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "orbits");
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
        "usage: orbits [--window | --root | --layer] [--seed N] [--fps N] [--palette mission|phosphor|paper|mono]\n"
        "              [--act N] [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--check]\n"
        "              [--font-mono FAMILY] [--font-sans FAMILY]\n");
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

/* Spanning: the output under the middle of the layout (flightsim's front
 * rule) shows the trajectory; portrait outputs the flight-dynamics panel;
 * other landscape outputs the ground track or the rotating frame. */
static void pick_role(void) {
    if (span_me < 0) { role = R_FRONT; return; }
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    for (int i = 0; i < span_n; i++) {
        if (i == front) continue;
        if (span[i][3] > span[i][2]) has_panel = 1;
        else has_side = 1;
    }
    if (span_me == front) { role = R_FRONT; return; }
    if (span[span_me][3] > span[span_me][2]) { role = R_PANEL; return; }
    role = R_SIDE;
    side_rank = 0;
    for (int i = 0; i < span_n; i++)
        if (i != front && i != span_me && span[i][3] <= span[i][2] && span[i][0] < span[span_me][0]) side_rank++;
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

/* ---- --check: the numbers, against the textbooks ---- */
static int check(double wt) {
    int bad = 0;
#define EXPECT(cond, ...) do { int ok_ = (cond); bad += !ok_; printf("  [%s] ", ok_ ? " ok " : "FAIL"); printf(__VA_ARGS__); printf("\n"); } while (0)
    printf("orbits --check\n");
    ho_init(1);
    printf("Hohmann LEO 300 km -> GEO:\n");
    EXPECT(fabs(ho.dv1 + ho.dv2 - 3.893) < 0.01, "dv1 %.4f + dv2 %.4f = %.4f km/s (textbook ~3.89), transfer %.2f h", ho.dv1, ho.dv2, ho.dv1 + ho.dv2, (ho.tb2 - ho.tb1) / 3600);
    printf("  bi-elliptic via r_b = %.0f km: %.4f + %.4f + %.4f = %.4f km/s, %.1f h\n", ho.rb, ho.e1, ho.e2, -ho.e3, ho.e1 + ho.e2 - ho.e3, (ho.bb3 - ho.tb1) / 3600);
    EXPECT(fabs(bie_dv(11.94, 1e12) - hoh_dv(11.94)) < 2e-3, "Hohmann = bi-elliptic(r_b->inf) at R = 11.94: %.5f vs %.5f (v/v1)", hoh_dv(11.94), bie_dv(11.94, 1e12));
    double a1 = ELEM[P_EARTH][0] * AU, a2 = ELEM[P_MARS][0] * AU, am = (a1 + a2) / 2;
    double tH = PI_D * sqrt(am * am * am / MU_SUN) / DAY;
    printf("Earth -> Mars:\n");
    EXPECT(fabs(tH - 259) < 1.5, "minimum-energy (Hohmann) transfer %.1f days (expect ~259)", tH);
    /* Lambert round trips: solve, then propagate v1 for dt with the universal-variable Kepler solver */
    double jd = jd_of(wt), worst = 0, worst_el = 0;
    int nfail = 0;
    for (int i = 0; i < 400; i++) {
        double td = jd + hrand(9, i) * 800, tof = 60 + hrand(10, i) * 400;
        double r1[3], v[3], r2[3], v1[3], v2[3], rp[3], vp[3], d[3];
        planet_rv(P_EARTH, td, r1, v);
        planet_rv(P_MARS, td + tof, r2, v);
        if (!lambert(MU_SUN, r1, r2, tof * DAY, 1, v1, v2)) { nfail++; continue; }
        kepler_uv(MU_SUN, r1, v1, tof * DAY, rp, vp);
        sub3(rp, r2, d);
        worst = fmax(worst, norm3(d));
        Orbit o;
        orb_from_rv(MU_SUN, r1, v1, 0, &o);
        orb_rv(&o, tof * DAY, rp, vp);
        sub3(rp, r2, d);
        worst_el = fmax(worst_el, norm3(d));
    }
    EXPECT(worst < 1 && worst_el < 1, "Lambert -> propagate round trips (400 Earth-Mars pairs): worst miss %.2e km (universal variable), %.2e km (elements); %d unsolved", worst, worst_el, nfail);
    mx_init(1, wt);
    double t0 = now();
    mx_compute(PCX * PCY);
    double tp = now() - t0;
    char d1[24], d2[24];
    fmt_date(d1, sizeof d1, mx.tdep);
    fmt_date(d2, sizeof d2, mx.tarr);
    printf("  porkchop %dx%d (%.0f ms, %.1f us/solve): C3 min %.2f km^2/s^2; best total dv %.3f km/s: launch %s, arrive %s (%.0f d), C3 %.2f, v_inf %.2f\n",
           PCX, PCY, tp * 1e3, tp * 1e6 / (PCX * PCY), mx.mc3, mx.bdv1 + mx.bdv2, d1, d2, mx.tarr - mx.tdep, mx.bc3, mx.bvi);
    printf("Gravity assist:\n");
    fb_init(5);
    {
        double p[3], v[3];
        fb_state(-fb.tmax * 40, p, v);
        double ain = acos(clampd((v[0] * fb.uin[0] + v[1] * fb.uin[1]) / norm3(v), -1, 1));
        fb_state(fb.tmax * 40, p, v);
        double aout = acos(clampd((v[0] * fb.uout[0] + v[1] * fb.uout[1]) / norm3(v), -1, 1));
        fb_state(0, p, v);
        EXPECT(ain < 0.01 && aout < 0.01 && fabs(norm3(p) - fb.rp) < 1e-3, "v_inf %.3f km/s, r_p %.2f R_J, turn %.2f deg; far legs line up with the asymptotes to %.4f / %.4f rad; periapsis %.3f km off",
               fb.vinf, fb.rp / R_JUP, fb.delta / RAD, ain, aout, fabs(norm3(p) - fb.rp));
        EXPECT(fabs(fb.dv - 2 * fb.vinf * sin(fb.delta / 2)) < 1e-9, "|dv| %.4f = 2 v_inf sin(delta/2) %.4f; helio %.2f -> %.2f km/s", fb.dv, 2 * fb.vinf * sin(fb.delta / 2), hypot(fb.vin[0], fb.vin[1]), hypot(fb.vout[0], fb.vout[1]));
    }
    printf("CR3BP:\n");
    cr_init(3);
    const double known[3] = {0.836915, 1.155682, -1.005063};
    for (int i = 0; i < 3; i++)
        EXPECT(fabs(cr.Le[i][0] - known[i]) < 2e-6, "Earth-Moon L%d x = %.6f (known %.6f), %.0f km from Earth, C = %.6f", i + 1, cr.Le[i][0], known[i], fabs(cr.Le[i][0] + MU_EM) * 384400, cr.Le[i][2]);
    EXPECT(fabs(cr.Le[3][0] - (0.5 - MU_EM)) < 1e-12 && fabs(cr.Le[3][1] - sqrt(3) / 2) < 1e-12, "L4/L5 = (%.6f, +-%.6f), C = %.6f", cr.Le[3][0], cr.Le[3][1], cr.Le[3][2]);
    {
        cr_mu = MU_EM;
        int n10 = 10 * 400;
        double *buf = malloc(sizeof(double) * 4 * (n10 + 2)), dr;
        int n = cr_run(cr.ly0, 10 * cr.lyT, cr.lyT / 400, buf, n10 + 1, 0, 0, &dr);
        EXPECT(dr < 1e-9, "L1 Lyapunov orbit: T = %.4f (%.2f d), C = %.6f; Jacobi drift over 10 periods %.2e (%d samples)", cr.lyT, cr.lyT * 4.3484, cr.lyC, dr, n);
        free(buf);
    }
    EXPECT(cr.chdrift < 1e-8, "%d L1-neck transits over %.0f TU: max Jacobi drift %.2e", cr.nch, CR_T, cr.chdrift);
    EXPECT(cr.sjdrift < 1e-8, "Sun-Jupiter horseshoe + tadpole over 600 TU (%.0f yr): max Jacobi drift %.2e", 600 * 11.862 / TAU_D, cr.sjdrift);
    printf("Decay:\n");
    printf("  185 x 800 km, B = 50 kg/m^2: entry interface (120 km) after %.2f days, %d steps of %.0f s\n", dc.life / DAY, dc.n, DC_DT);
    printf("%s\n", bad ? "SOME CHECKS FAILED" : "all checks passed");
    return bad != 0;
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
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-act") && i + 1 < argc) opt_act = atoi(argv[++i]) % NACT;
        else if (!strcmp(a, "-check")) opt_check = 1;
        else if (!strcmp(a, "-font-mono") && i + 1 < argc) opt_font_mono = argv[++i];
        else if (!strcmp(a, "-font-sans") && i + 1 < argc) opt_font_sans = argv[++i];
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    pick_role();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int pi = -1;
    for (int i = 0; i < NPAL; i++) if (!strcmp(opt_palette, PALS[i].name)) pi = i;
    if (!strcmp(opt_palette, "random")) pi = (int)(hash32(opt_seed) % NPAL);
    if (pi < 0) usage();
    pal = &PALS[pi];
    int offscreen = opt_shot || opt_bench > 0;
    shot_mode = opt_shot != NULL;

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    for (int i = 0; host[i] && i < 63; i++) mission_name[i] = (char)toupper((unsigned char)host[i]);
    dc_init();
    if (opt_check) return check(wall());
    land_init();
    fonts_init();

    /* --epoch: when the show started, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    t_wall0 = opt_epoch ? opt_epoch : wall();
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
        XStoreName(dpy, win, "Orbits");
        XClassHint chint = {"orbits", "Orbits"};
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
        target_fbo = fbo;
        if (getenv("ORBITS_RESIZE_TEST")) {
            /* the window resized again and again: caches keyed by size must not pile up */
            int ow = W, oh = H;
            for (int i = 0; i < 40; i++) {
                W = 320 + (i * 157) % (ow - 320); H = 240 + (i * 97) % (oh - 240);
                render(t_wall0 + opt_at + i * 3);
                glFinish();
            }
            printf("orbits: 40 sizes rendered, %d text styles, atlas %d rows\n", nstyles, atlas_h);
            return 0;
        }
        if (opt_shot) {
            render(t_wall0 + opt_at);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double wt0 = t_wall0 + opt_at;
        render(wt0);
        glFinish();
        GLuint q;
        glGenQueries(1, &q);
        static double samp[4096], gsamp[4096];
        double sum = 0, worst = 0;
        int ns = opt_bench < 4096 ? opt_bench : 4096;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(wt0 + i / 60.0);
            glEndQuery(GL_TIME_ELAPSED);
            double f1 = now();
            glFinish();
            GLuint64 ns_gpu = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns_gpu);
            double ms = (now() - f0) * 1000;
            if (i < 4096) { samp[i] = (f1 - f0) * 1000; gsamp[i] = ns_gpu / 1e6; }
            sum += ms;
            if (ms > worst) worst = ms;
        }
        for (int p = 1; p < ns; p++) for (int k = p; k > 0 && samp[k] < samp[k - 1]; k--) { double t = samp[k]; samp[k] = samp[k - 1]; samp[k - 1] = t; }
        for (int p = 1; p < ns; p++) for (int k = p; k > 0 && gsamp[k] < gsamp[k - 1]; k--) { double t = gsamp[k]; gsamp[k] = gsamp[k - 1]; gsamp[k - 1] = t; }
        printf("orbits: %dx%d role %d act %d, %d frames: GPU median %.3f ms (p90 %.3f), CPU build median %.3f ms, wall avg %.3f worst %.3f ms\n",
               W, H, role, act, opt_bench, gsamp[ns / 2], gsamp[ns * 9 / 10], samp[ns / 2], sum / opt_bench, worst);
        printf("orbits: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
        double t = now(), wt = wall();
        render(wt);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
