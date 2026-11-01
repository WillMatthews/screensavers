/*
 * clockwork — an opened-up mechanical clock movement, ticking, in brass.
 *
 * A skeleton movement built from real gearing: mainspring barrel (72) into
 * the centre pinion (12), centre wheel (64) into the third pinion (8), third
 * wheel (60) into the fourth pinion (8), fourth wheel (48) into the escape
 * pinion (12), so the centre wheel turns once an hour and the fourth once a
 * minute. A 15-tooth club-tooth escape wheel and a lever (pallet fork with
 * ruby pallets) are driven by a screwed balance with a breathing hairspring,
 * 1 Hz, two beats a second: the whole train steps forward with each tick.
 * Under the chapter ring the motion works (cannon pinion 12, minute wheel
 * 36/10, hour wheel 40) give 12:1, and the hands show the real local time,
 * with a small seconds hand on the fourth wheel.
 *
 * Every part is a procedural mesh, mostly extruded profiles: wheels with
 * cycloidal-style ogival teeth and radial flanks, pinion leaves, sawtooth
 * escape and ratchet teeth, capsule-shaped cocks with polished bevels. It is
 * rasterised with 4x MSAA and shaded as metal: anisotropic GGX (circular
 * graining on the wheels, perlage on the plate, Geneva stripes on the cocks),
 * a warm studio environment, a shadow map from the key light and disc
 * occlusion under the wheels (exact view factors, baked once into a 3D
 * texture since nothing moves out of place). The fast parts (balance, lever, escape
 * wheel, seconds hand) get motion blur by drawing them at four shutter times,
 * one per MSAA sample. Then depth of field, a little bloom, filmic tonemap.
 * The scene renders at about a megapixel and is upscaled (Catmull-Rom); live,
 * the scale follows a GPU-time budget, since one instance runs per monitor.
 *
 * Same plumbing as stir and flightsim: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] for one frame offscreen and
 * --bench N for timing.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The movement already keeps the wall clock, so every instance shows the
 * same tick; each gets its own camera, its path run off the shared epoch.
 * The output under the middle of the layout holds the whole movement and
 * the dial; the others, left to right, take turns at close-ups of the
 * escapement (balance, lever, escape wheel) and of the train and barrel.
 * On a portrait screen the camera backs off to keep the same width in view.
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
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

#define PI 3.14159265358979323846
#define TAU (2 * PI)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 0;                  /* camera-path time to start at */
static int off_w = 1920, off_h = 1080;
static float opt_scale = 0;                /* render scale; 0 = adapt to the frame budget */
static double opt_epoch;                   /* CLOCK_REALTIME seconds the camera path starts at */

/* ---- spanning: one instance per output, each with its own camera ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];             /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { R_ALL, R_FRONT, R_ESCAPEMENT, R_TRAIN };
static int role = R_ALL;                   /* R_ALL: one screen, the camera tours everything */

static void die(const char *what) { fprintf(stderr, "clockwork: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* local time at unix time u, in seconds since local midnight */
static double clock_at(double u) {
    struct tm tm;
    time_t s = (time_t)floor(u);
    localtime_r(&s, &tm);
    return tm.tm_hour * 3600.0 + tm.tm_min * 60.0 + tm.tm_sec + (u - floor(u));
}

static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* local wall-clock time, in seconds since local midnight */
static double clock_now(void) { return clock_at(wallclock()); }

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}

/* ---- 4x4 matrices, column-major ---- */
typedef struct { float m[16]; } M4;

static M4 m4_id(void) { M4 r = {{0}}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static M4 m4_mul(M4 a, M4 b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 4; i++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + i] * b.m[c * 4 + k];
            r.m[c * 4 + i] = s;
        }
    return r;
}
static M4 m4_trans(float x, float y, float z) { M4 r = m4_id(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static M4 m4_rotz(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }
static M4 m4_roty(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
static M4 m4_persp(float fovy, float asp, float n, float f) {
    M4 r = {{0}};
    float t = 1.0f / tanf(fovy * 0.5f);
    r.m[0] = t / asp; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1; r.m[14] = 2 * f * n / (n - f);
    return r;
}
static M4 m4_ortho(float l, float r_, float b, float t, float n, float f) {
    M4 r = m4_id();
    r.m[0] = 2 / (r_ - l); r.m[5] = 2 / (t - b); r.m[10] = -2 / (f - n);
    r.m[12] = -(r_ + l) / (r_ - l); r.m[13] = -(t + b) / (t - b); r.m[14] = -(f + n) / (f - n);
    return r;
}
static void v3norm(float *v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; } }
static M4 m4_look(const float *e, const float *at, const float *up) {
    float f[3] = {at[0] - e[0], at[1] - e[1], at[2] - e[2]};
    v3norm(f);
    float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    v3norm(s);
    float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
    M4 r = m4_id();
    r.m[0] = s[0]; r.m[4] = s[1]; r.m[8] = s[2];
    r.m[1] = u[0]; r.m[5] = u[1]; r.m[9] = u[2];
    r.m[2] = -f[0]; r.m[6] = -f[1]; r.m[10] = -f[2];
    r.m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
    r.m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
    r.m[14] = f[0] * e[0] + f[1] * e[1] + f[2] * e[2];
    return r;
}

/* ---- mesh building ----
 * One vertex buffer for everything. A vertex carries its material and, for
 * the hairspring, how far along the spring it is (the shader winds it). */
enum { M_BRASS, M_PLATE, M_BRIDGE, M_STEEL, M_BLUED, M_RUBY, M_GOLD, M_DIAL, M_LACQUER, M_BLACK };

typedef struct { float p[3], n[3], m, s; } Vtx;
static Vtx *vb;
static int nv, cv;
static M4 xf;                              /* applied to everything emitted */
static int xf_on;
static float cur_mat;

static void vtx(float x, float y, float z, float nx, float ny, float nz, float s) {
    if (nv == cv) vb = realloc(vb, (size_t)(cv = cv ? cv * 2 : 65536) * sizeof *vb);
    Vtx *v = &vb[nv++];
    if (xf_on) {
        const float *m = xf.m;
        v->p[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
        v->p[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        v->p[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        v->n[0] = m[0] * nx + m[4] * ny + m[8] * nz;
        v->n[1] = m[1] * nx + m[5] * ny + m[9] * nz;
        v->n[2] = m[2] * nx + m[6] * ny + m[10] * nz;
    } else {
        v->p[0] = x; v->p[1] = y; v->p[2] = z;
        v->n[0] = nx; v->n[1] = ny; v->n[2] = nz;
    }
    v->m = cur_mat;
    v->s = s;
}

/* a quad a b c d (as two triangles) with a normal per corner */
static void quad(const float *a, const float *b, const float *c, const float *d,
                 const float *na, const float *nb, const float *nc, const float *nd) {
    vtx(a[0], a[1], a[2], na[0], na[1], na[2], 0);
    vtx(b[0], b[1], b[2], nb[0], nb[1], nb[2], 0);
    vtx(c[0], c[1], c[2], nc[0], nc[1], nc[2], 0);
    vtx(a[0], a[1], a[2], na[0], na[1], na[2], 0);
    vtx(c[0], c[1], c[2], nc[0], nc[1], nc[2], 0);
    vtx(d[0], d[1], d[2], nd[0], nd[1], nd[2], 0);
}

#define MAXC 4096
static float cA[2 * MAXC], cB[2 * MAXC];   /* contour scratch */

/* outward segment normals of a closed CCW contour, and per-vertex normals
 * for the two ends of each segment: smoothed across gentle bends, kept
 * sharp at corners (tooth tips and roots) */
typedef struct { float sn[2 * MAXC], mn[2 * MAXC], ea[2 * MAXC], eb[2 * MAXC]; } Edges;
static Edges eo, ei;

static void edges(const float *c, int n, Edges *e) {
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        float dx = c[2 * j] - c[2 * i], dy = c[2 * j + 1] - c[2 * i + 1], l = sqrtf(dx * dx + dy * dy);
        if (l < 1e-6f) { e->sn[2 * i] = 0; e->sn[2 * i + 1] = 0; continue; }
        e->sn[2 * i] = dy / l; e->sn[2 * i + 1] = -dx / l;
    }
    /* degenerate segments borrow a neighbour's normal */
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < n; i++)
            if (e->sn[2 * i] == 0 && e->sn[2 * i + 1] == 0) {
                int p = (i + n - 1) % n;
                e->sn[2 * i] = e->sn[2 * p]; e->sn[2 * i + 1] = e->sn[2 * p + 1];
            }
    for (int i = 0; i < n; i++) {
        int p = (i + n - 1) % n;
        float mx = e->sn[2 * p] + e->sn[2 * i], my = e->sn[2 * p + 1] + e->sn[2 * i + 1], l = sqrtf(mx * mx + my * my);
        if (l < 1e-6f) { mx = e->sn[2 * i]; my = e->sn[2 * i + 1]; l = 1; }
        e->mn[2 * i] = mx / l; e->mn[2 * i + 1] = my / l;
    }
    for (int i = 0; i < n; i++) {
        int p = (i + n - 1) % n, j = (i + 1) % n;
        float d0 = e->sn[2 * p] * e->sn[2 * i] + e->sn[2 * p + 1] * e->sn[2 * i + 1];
        float d1 = e->sn[2 * i] * e->sn[2 * j] + e->sn[2 * i + 1] * e->sn[2 * j + 1];
        e->ea[2 * i] = d0 > 0.82f ? e->mn[2 * i] : e->sn[2 * i];
        e->ea[2 * i + 1] = d0 > 0.82f ? e->mn[2 * i + 1] : e->sn[2 * i + 1];
        e->eb[2 * i] = d1 > 0.82f ? e->mn[2 * j] : e->sn[2 * i];
        e->eb[2 * i + 1] = d1 > 0.82f ? e->mn[2 * j + 1] : e->sn[2 * i + 1];
    }
}

/* Extrude a closed CCW contour o (n points) from z0 to z1. With an inner
 * contour `in` (same count, point i at the same angle as o[i]) it's a ring;
 * without, the caps fan from the centroid (the shape must be star-shaped
 * about it). bev > 0 cuts a 45-degree chamfer round the top edges. */
static void extrude(const float *o, const float *in, int n, float z0, float z1, float bev, int bottom) {
    static float oi[2 * MAXC], ii[2 * MAXC];
    edges(o, n, &eo);
    if (in) edges(in, n, &ei);
    float zt = z1 - bev;
    for (int i = 0; i < n; i++) {
        float k = bev / fmaxf(eo.mn[2 * i] * eo.sn[2 * i] + eo.mn[2 * i + 1] * eo.sn[2 * i + 1], 0.5f);
        oi[2 * i] = o[2 * i] - eo.mn[2 * i] * k;
        oi[2 * i + 1] = o[2 * i + 1] - eo.mn[2 * i + 1] * k;
        if (in) {
            float k2 = bev / fmaxf(ei.mn[2 * i] * ei.sn[2 * i] + ei.mn[2 * i + 1] * ei.sn[2 * i + 1], 0.5f);
            ii[2 * i] = in[2 * i] + ei.mn[2 * i] * k2;
            ii[2 * i + 1] = in[2 * i + 1] + ei.mn[2 * i + 1] * k2;
        }
    }
    float cx = 0, cy = 0;
    for (int i = 0; i < n; i++) { cx += o[2 * i]; cy += o[2 * i + 1]; }
    cx /= n; cy /= n;
    const float up[3] = {0, 0, 1}, dn[3] = {0, 0, -1};
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        /* outer wall */
        float a0[3] = {o[2 * i], o[2 * i + 1], z0}, b0[3] = {o[2 * j], o[2 * j + 1], z0};
        float a1[3] = {o[2 * i], o[2 * i + 1], zt}, b1[3] = {o[2 * j], o[2 * j + 1], zt};
        float na[3] = {eo.ea[2 * i], eo.ea[2 * i + 1], 0}, nb[3] = {eo.eb[2 * i], eo.eb[2 * i + 1], 0};
        if (zt > z0) quad(a0, b0, b1, a1, na, nb, nb, na);
        float ta[3] = {oi[2 * i], oi[2 * i + 1], z1}, tb[3] = {oi[2 * j], oi[2 * j + 1], z1};
        if (bev > 0) {
            float ca[3] = {na[0], na[1], 1}, cb[3] = {nb[0], nb[1], 1};
            v3norm(ca); v3norm(cb);
            quad(a1, b1, tb, ta, ca, cb, cb, ca);
        }
        if (in) {
            float p0[3] = {in[2 * i], in[2 * i + 1], z0}, q0[3] = {in[2 * j], in[2 * j + 1], z0};
            float p1[3] = {in[2 * i], in[2 * i + 1], zt}, q1[3] = {in[2 * j], in[2 * j + 1], zt};
            float pa[3] = {-ei.ea[2 * i], -ei.ea[2 * i + 1], 0}, pb[3] = {-ei.eb[2 * i], -ei.eb[2 * i + 1], 0};
            if (zt > z0) quad(q0, p0, p1, q1, pb, pa, pa, pb);
            float ua[3] = {ii[2 * i], ii[2 * i + 1], z1}, ub[3] = {ii[2 * j], ii[2 * j + 1], z1};
            if (bev > 0) {
                float ca[3] = {pa[0], pa[1], 1}, cb[3] = {pb[0], pb[1], 1};
                v3norm(ca); v3norm(cb);
                quad(q1, p1, ua, ub, cb, ca, ca, cb);
            }
            quad(ua, ta, tb, ub, up, up, up, up);
            if (bottom) quad(p0, q0, b0, a0, dn, dn, dn, dn);
        } else {
            float c1[3] = {cx, cy, z1};
            vtx(c1[0], c1[1], c1[2], 0, 0, 1, 0);
            vtx(ta[0], ta[1], ta[2], 0, 0, 1, 0);
            vtx(tb[0], tb[1], tb[2], 0, 0, 1, 0);
            if (bottom) {
                vtx(cx, cy, z0, 0, 0, -1, 0);
                vtx(b0[0], b0[1], b0[2], 0, 0, -1, 0);
                vtx(a0[0], a0[1], a0[2], 0, 0, -1, 0);
            }
        }
    }
}

/* ---- contours ---- */
static int seg_for(float r) { int n = (int)(r * 12); return n < 12 ? 12 : n > 160 ? 160 : n; }

static int circle_c(float *o, float cx, float cy, float r, int n) {
    for (int i = 0; i < n; i++) {
        double a = TAU * i / n;
        o[2 * i] = cx + r * (float)cos(a);
        o[2 * i + 1] = cy + r * (float)sin(a);
    }
    return n;
}

/* an inner circle matched point for point to contour o, round (cx, cy) */
static void match_circle(const float *o, int n, float cx, float cy, float r, float *in) {
    for (int i = 0; i < n; i++) {
        float dx = o[2 * i] - cx, dy = o[2 * i + 1] - cy, l = sqrtf(dx * dx + dy * dy);
        in[2 * i] = cx + dx / l * r;
        in[2 * i + 1] = cy + dy / l * r;
    }
}

/* A wheel or pinion, tooth 0 centred on angle `rot`. Clock-style: radial
 * flanks below the pitch circle, an elliptical (ogival) addendum above,
 * small root fillets. ha / hf: addendum / dedendum in modules; thick: tooth
 * thickness at the pitch circle as a fraction of half the pitch. */
static int gear_c(float *o, int N, float m, float ha, float hf, float thick, float rot) {
    float rp = m * N * 0.5f, rf = rp - hf * m, Ht = ha * m, pa = (float)TAU / N;
    float hw = thick * (float)PI * m * 0.25f, af = hw / rp;
    float rfil = fminf(0.3f * m, (pa * 0.5f - af) * rf * 0.7f);
    int n = 0;
#define P(a, r) (o[2 * n] = (r) * cosf(a), o[2 * n + 1] = (r) * sinf(a), n++)
    for (int k = 0; k < N; k++) {
        float c = rot + k * pa;
        P(c - pa * 0.5f, rf);
        P(c - (pa * 0.5f + af + rfil / rf) * 0.5f, rf);
        for (int j = 0; j <= 2; j++) {
            float s = j * (float)PI / 4;
            P(c - af - rfil * (1 - sinf(s)) / rf, rf + rfil * (1 - cosf(s)));
        }
        P(c - af, rp);
        for (int j = 1; j <= 5; j++) {
            float s = j * (float)PI / 10, h = Ht * sinf(s), w = hw * cosf(s);
            P(c - w / (rp + h), rp + h);
        }
        for (int j = 4; j >= 1; j--) {
            float s = j * (float)PI / 10, h = Ht * sinf(s), w = hw * cosf(s);
            P(c + w / (rp + h), rp + h);
        }
        P(c + af, rp);
        for (int j = 2; j >= 0; j--) {
            float s = j * (float)PI / 4;
            P(c + af + rfil * (1 - sinf(s)) / rf, rf + rfil * (1 - cosf(s)));
        }
        P(c + (pa * 0.5f + af + rfil / rf) * 0.5f, rf);
    }
#undef P
    return n;
}

/* Sawtooth teeth (escape wheel, ratchet): per tooth a profile of (u, r)
 * with u in [0, 1] across the pitch and r from root rr (0) to tip R (1),
 * ending with a radial drop. mirror flips which way the teeth lean. */
static int saw_c(float *o, int N, float rr, float R, const float (*pf)[2], int np, int mirror, float rot) {
    int n = 0;
    float pa = (float)TAU / N;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < np; j++) {
            float a = rot + (k + pf[j][0]) * pa, r = rr + (R - rr) * pf[j][1];
            o[2 * n] = r * cosf(a); o[2 * n + 1] = r * sinf(a); n++;
        }
    if (mirror) {
        for (int i = 0; i < n; i++) o[2 * i + 1] = -o[2 * i + 1];
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            float tx = o[2 * i], ty = o[2 * i + 1];
            o[2 * i] = o[2 * j]; o[2 * i + 1] = o[2 * j + 1];
            o[2 * j] = tx; o[2 * j + 1] = ty;
        }
    }
    return n;
}

/* convex hull of two circles: a cock or a lever arm */
static int capsule_c(float *o, float ax, float ay, float ra, float bx, float by, float rb) {
    float dx = bx - ax, dy = by - ay, d = sqrtf(dx * dx + dy * dy), u = atan2f(dy, dx);
    float c = (ra - rb) / d;
    float be = acosf(c < -0.99f ? -0.99f : c > 0.99f ? 0.99f : c);
    int nb = 6 + (int)(rb * 2 * be * 4), na = 6 + (int)(ra * (2 * (float)PI - 2 * be) * 4), n = 0;
    if (nb > 200) nb = 200;
    if (na > 400) na = 400;
    for (int i = 0; i <= nb; i++) {
        float a = u - be + 2 * be * i / nb;
        o[2 * n] = bx + rb * cosf(a); o[2 * n + 1] = by + rb * sinf(a); n++;
    }
    for (int i = 0; i <= na; i++) {
        float a = u + be + (2 * (float)PI - 2 * be) * i / na;
        o[2 * n] = ax + ra * cosf(a); o[2 * n + 1] = ay + ra * sinf(a); n++;
    }
    return n;
}

/* a quadrilateral bar from (x0,y0) to (x1,y1), w0 wide at the start and w1
 * at the end */
static int bar_c(float *o, float x0, float y0, float x1, float y1, float w0, float w1) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy), px = -dy / l, py = dx / l;
    o[0] = x0 + px * w0 * 0.5f; o[1] = y0 + py * w0 * 0.5f;
    o[2] = x0 - px * w0 * 0.5f; o[3] = y0 - py * w0 * 0.5f;
    o[4] = x1 - px * w1 * 0.5f; o[5] = y1 - py * w1 * 0.5f;
    o[6] = x1 + px * w1 * 0.5f; o[7] = y1 + py * w1 * 0.5f;
    /* CCW? (o1 - o0) x (o2 - o1) > 0 */
    float cr = (o[2] - o[0]) * (o[5] - o[3]) - (o[3] - o[1]) * (o[4] - o[2]);
    if (cr < 0) {
        float t[8];
        memcpy(t, o, sizeof t);
        for (int i = 0; i < 4; i++) { o[2 * i] = t[2 * (3 - i)]; o[2 * i + 1] = t[2 * (3 - i) + 1]; }
    }
    return 4;
}

/* ---- primitive parts ---- */
static void cyl(float cx, float cy, float r, float z0, float z1, float bev) {
    int n = circle_c(cA, cx, cy, r, seg_for(r));
    extrude(cA, NULL, n, z0, z1, bev, 0);
}
static void ann(float cx, float cy, float r0, float r1, float z0, float z1, float bev) {
    int n = circle_c(cA, cx, cy, r1, seg_for(r1));
    match_circle(cA, n, cx, cy, r0, cB);
    extrude(cA, cB, n, z0, z1, bev, 0);
}
static void capsule(float ax, float ay, float ra, float bx, float by, float rb, float z0, float z1, float bev) {
    int n = capsule_c(cA, ax, ay, ra, bx, by, rb);
    extrude(cA, NULL, n, z0, z1, bev, 0);
}
static void bar(float x0, float y0, float x1, float y1, float w0, float w1, float z0, float z1) {
    bar_c(cA, x0, y0, x1, y1, w0, w1);
    extrude(cA, NULL, 4, z0, z1, 0, 0);
}

/* a slotted screw head standing on z, slot at angle a */
static void screw(float x, float y, float z, float r, float a) {
    cur_mat = M_BLUED;
    cyl(x, y, r, z, z + 0.42f * r, 0.2f * r);
    cur_mat = M_BLACK;
    float c = cosf(a) * r * 0.86f, s = sinf(a) * r * 0.86f;
    bar(x - c, y - s, x + c, y + s, 0.24f * r, 0.24f * r, z + 0.2f * r, z + 0.43f * r);
}

/* a ruby in a gold chaton, held by two little screws, on a surface at z;
 * pivot: the end of an arbor shows in the hole */
static void jewel(float x, float y, float z, float r, float a) {
    cur_mat = M_GOLD;
    ann(x, y, r * 0.62f, r, z, z + 0.32f, 0.12f);
    cur_mat = M_RUBY;
    cyl(x, y, r * 0.64f, z, z + 0.24f, 0);
    ann(x, y, r * 0.16f, r * 0.46f, z + 0.24f, z + 0.36f, 0.08f);
    cur_mat = M_STEEL;
    cyl(x, y, r * 0.16f, z + 0.2f, z + 0.38f, 0);
    for (int k = 0; k < 2; k++) {
        float b = a + k * (float)PI;
        screw(x + cosf(b) * r * 1.35f, y + sinf(b) * r * 1.35f, z, r * 0.36f, b + 0.7f + k);
    }
}

/* ---- the movement ---- */
enum { B_STATIC, B_BARREL, B_CENTER, B_THIRD, B_FOURTH, B_ESCAPE, B_FORK, B_BALANCE,
       B_MINWHEEL, B_HOURWHEEL, B_HOUR, B_MINUTE, B_SECOND, B_RATCHET, B_PLATE, NB };
/* drawn front to back, so the depth test throws away what's hidden early */
static const int draw_order[NB] = {B_MINUTE, B_HOUR, B_SECOND, B_STATIC, B_RATCHET, B_HOURWHEEL, B_MINWHEEL, B_CENTER,
                                   B_BALANCE, B_FORK, B_ESCAPE, B_THIRD, B_FOURTH, B_BARREL, B_PLATE};
typedef struct { float x, y; int first, count, blur; } Body;   /* first, count: indices once uploaded */
static Body body[NB];

/* gearing */
#define N_BARREL 72
#define N_CPIN 12
#define N_CENTER 64
#define N_TPIN 8
#define N_THIRD 60
#define N_FPIN 8
#define N_FOURTH 48
#define N_EPIN 12
#define N_ESCAPE 15
#define N_CANNON 12
#define N_MINW 36
#define N_MINP 10
#define N_HOURW 40
static const float MOD_B = 0.55f, MOD_C = 0.5f, MOD_T = 0.42f, MOD_MW = 0.4f, MOD_HW = 0.384f;

static float phi_f;                        /* direction balance -> lever pivot -> escape wheel */
static float esc_rot, esc_mirror;          /* the escape teeth's phase on their arbor, and lean */
static float bal_amp = 4.0f;               /* balance amplitude, radians */
static const float FORK_BANK = 0.17f;
static const float D_EP = 7.4f, D_PB = 8.6f, R_ESC = 6.3f;
static float spring_end;                   /* hairspring's outer end, angle in the balance frame at rest */
static int spokes_style;

#define MAXDISC 32
static float disc[MAXDISC][4], disc_s[MAXDISC];   /* occluders: centre x, y, height z, radius; strength */
static int ndisc;
static void add_disc(float x, float y, float z, float r, float s) {
    if (ndisc == MAXDISC) return;
    disc[ndisc][0] = x; disc[ndisc][1] = y; disc[ndisc][2] = z; disc[ndisc][3] = r;
    disc_s[ndisc++] = s;
}

/* The occlusion under the wheels, cocks and dial, for an upward-facing
 * point anywhere in the movement, baked once into a 3D texture: each
 * occluder is a disc, and a disc's view factor from a point facing it is
 * exact. */
#define AO_N 128
#define AO_Z 56
static const float AO_MIN[3] = {-56, -56, -3.5f}, AO_SIZE[3] = {112, 112, 28};
static unsigned char *bake_ao(void) {
    unsigned char *a = malloc((size_t)AO_N * AO_N * AO_Z);
    for (int z = 0; z < AO_Z; z++)
        for (int y = 0; y < AO_N; y++)
            for (int x = 0; x < AO_N; x++) {
                float px = AO_MIN[0] + (x + 0.5f) * AO_SIZE[0] / AO_N, py = AO_MIN[1] + (y + 0.5f) * AO_SIZE[1] / AO_N;
                float pz = AO_MIN[2] + (z + 0.5f) * AO_SIZE[2] / AO_Z;
                float occ = 0;
                for (int i = 0; i < ndisc; i++) {
                    float h = disc[i][2] - pz;
                    if (h < 0.05f) continue;
                    float qx = px - disc[i][0], qy = py - disc[i][1], rho2 = qx * qx + qy * qy, R2 = disc[i][3] * disc[i][3];
                    float X = h * h + R2 + rho2;
                    occ += disc_s[i] * 0.5f * (1 - (h * h + rho2 - R2) / sqrtf(fmaxf(X * X - 4 * R2 * rho2, 1e-6f)));
                }
                a[((size_t)z * AO_N + y) * AO_N + x] = (unsigned char)lroundf(255 * expf(-1.3f * fmaxf(occ, 0)));
            }
    return a;
}

static double mesh_angle(double th1, int n1, int n2, float x1, float y1, float x2, float y2) {
    double phi = atan2(y2 - y1, x2 - x1);
    return phi + PI + PI / n2 - (th1 - phi) * n1 / n2;
}

/* every rotating body's angle at clock time t (seconds since midnight) */
typedef struct { float ang[NB]; float alpha; } Pose;

static double ease_tick(double f) {
    if (f >= 1) { double g = f - 1; return 1 + 0.05 * exp(-g * 2.5) * sin(g * 2.6 + PI); }
    return 1 - (1 - f) * (1 - f) * (1 - f);
}

static void pose_at(double t, Pose *p) {
    /* two beats a second; the train steps at each, just as the balance
     * swings through the middle and the lever flicks over */
    const double w1 = 0.02, w2 = 0.12;
    double v = t * 2 + w1, k = floor(v), f = (v - k) / (w1 + w2);
    double beats = k - 1 + ease_tick(f);
    double ts = (beats + 0.5) * 0.5;
    double th_c = -TAU * ts / 3600;
    Body *b = body;
    double th_b = mesh_angle(th_c, N_CPIN, N_BARREL, b[B_CENTER].x, b[B_CENTER].y, b[B_BARREL].x, b[B_BARREL].y);
    double th_t = mesh_angle(th_c, N_CENTER, N_TPIN, b[B_CENTER].x, b[B_CENTER].y, b[B_THIRD].x, b[B_THIRD].y);
    double th_f = mesh_angle(th_t, N_THIRD, N_FPIN, b[B_THIRD].x, b[B_THIRD].y, b[B_FOURTH].x, b[B_FOURTH].y);
    double th_e = mesh_angle(th_f, N_FOURTH, N_EPIN, b[B_FOURTH].x, b[B_FOURTH].y, b[B_ESCAPE].x, b[B_ESCAPE].y);
    double th_mw = mesh_angle(th_c, N_CANNON, N_MINW, 0, 0, b[B_MINWHEEL].x, b[B_MINWHEEL].y);
    double th_hw = mesh_angle(th_mw, N_MINP, N_HOURW, b[B_MINWHEEL].x, b[B_MINWHEEL].y, 0, 0);
    double a[NB] = {0};
    a[B_BARREL] = th_b; a[B_CENTER] = th_c; a[B_THIRD] = th_t; a[B_FOURTH] = th_f; a[B_ESCAPE] = th_e;
    a[B_MINWHEEL] = th_mw; a[B_HOURWHEEL] = th_hw;
    a[B_HOUR] = -TAU * ts / 43200; a[B_MINUTE] = th_c; a[B_SECOND] = -TAU * ts / 60;
    double alpha = bal_amp * sin(TAU * t);
    double fk = alpha / 0.45;
    fk = fk < -1 ? -1 : fk > 1 ? 1 : fk;
    fk = fk * (1.5 - 0.5 * fk * fk);
    a[B_FORK] = phi_f + FORK_BANK * fk;
    a[B_BALANCE] = phi_f + alpha;
    for (int i = 0; i < NB; i++) p->ang[i] = (float)fmod(a[i], TAU);
    p->alpha = (float)alpha;
}

static void begin_body(int i, float x, float y) { body[i].x = x; body[i].y = y; body[i].first = nv; }
static void end_body(int i) { body[i].count = nv - body[i].first; }

/* a spoked wheel at the body origin: toothed rim, spokes, hub */
static void wheel(int N, float m, float z0, float z1, int spokes, float hub) {
    cur_mat = M_BRASS;
    float rp = m * N * 0.5f, rf = rp - 1.6f * m, rim = fmaxf(0.9f, rp * 0.09f), ri = rf - rim;
    int n = gear_c(cA, N, m, 1.4f, 1.6f, 0.95f, 0);
    match_circle(cA, n, 0, 0, ri, cB);
    extrude(cA, cB, n, z0, z1, 0, 0);
    float w = fmaxf(0.55f, ri * 0.085f);
    for (int k = 0; k < spokes; k++) {
        float a = (float)TAU * k / spokes + 0.3f, c = cosf(a), s = sinf(a);
        if (spokes_style) {   /* gently curved crossings, in segments */
            float pr = hub * 0.8f, pa = a;
            for (int j = 1; j <= 6; j++) {
                float u = j / 6.0f, r = hub * 0.8f + (ri + 0.3f - hub * 0.8f) * u;
                float b = a + 0.32f * sinf(u * (float)PI) * (ri / 14.0f) / (1 + ri / 14.0f);
                float ww0 = w * (1.25f - 0.45f * (j - 1) / 6.0f), ww1 = w * (1.25f - 0.45f * u);
                bar(pr * cosf(pa), pr * sinf(pa), r * cosf(b), r * sinf(b), ww0, ww1, z0 + 0.05f, z1 - 0.05f);
                pr = r; pa = b;
            }
        } else {
            bar(hub * 0.8f * c, hub * 0.8f * s, (ri + 0.3f) * c, (ri + 0.3f) * s, w * 1.3f, w * 0.8f, z0 + 0.05f, z1 - 0.05f);
        }
    }
    cyl(0, 0, hub, z0 - 0.25f, z1 + 0.3f, 0.18f);
}

static void pinion(int N, float m, float z0, float z1) {
    cur_mat = M_STEEL;
    int n = gear_c(cA, N, m, 0.62f, 1.7f, 0.78f, 0);
    extrude(cA, NULL, n, z0, z1, 0, 0);
}

static void arbor(float r, float z0, float z1) {
    cur_mat = M_STEEL;
    cyl(0, 0, r, z0, z1, 0);
}

/* Roman numerals in strokes, upright with their feet to the centre */
static void numeral(const char *s, float ang, float r0, float h, float z) {
    float ux = cosf(ang), uy = sinf(ang);          /* up: radially out */
    float rx = uy, ry = -ux;                       /* right: clockwise */
    float adv[3] = {1.3f, 2.7f, 2.9f};             /* I V X */
    float total = -0.35f;
    for (const char *p = s; *p; p++) total += adv[*p == 'I' ? 0 : *p == 'V' ? 1 : 2] + 0.35f;
    float x = -total * 0.5f;
    const float thick = 0.6f, thin = 0.22f, serif = 0.2f;
    float cx = ux * r0, cy = uy * r0;
#define G(gx, gy) cx + rx * (gx) + ux * (gy), cy + ry * (gx) + uy * (gy)
    for (const char *p = s; *p; p++) {
        int g = *p == 'I' ? 0 : *p == 'V' ? 1 : 2;
        float a = adv[g], m = x + a * 0.5f;
        if (g == 0) {
            bar(G(m, 0.1f), G(m, h - 0.1f), thick, thick, z, z + 0.12f);
            bar(G(m - 0.62f, serif * 0.5f), G(m + 0.62f, serif * 0.5f), serif, serif, z, z + 0.12f);
            bar(G(m - 0.62f, h - serif * 0.5f), G(m + 0.62f, h - serif * 0.5f), serif, serif, z, z + 0.12f);
        } else if (g == 1) {
            bar(G(m - a * 0.42f, h - 0.1f), G(m - 0.05f, 0.05f), thick, thick * 0.7f, z, z + 0.12f);
            bar(G(m + a * 0.42f, h - 0.1f), G(m + 0.05f, 0.05f), thin, thin, z, z + 0.12f);
            bar(G(m - a * 0.5f - 0.15f, h - serif * 0.5f), G(m - a * 0.2f, h - serif * 0.5f), serif, serif, z, z + 0.12f);
            bar(G(m + a * 0.25f, h - serif * 0.5f), G(m + a * 0.5f + 0.15f, h - serif * 0.5f), serif, serif, z, z + 0.12f);
        } else {
            bar(G(m - a * 0.38f, h - 0.1f), G(m + a * 0.38f, 0.1f), thick, thick, z, z + 0.12f);
            bar(G(m + a * 0.38f, h - 0.1f), G(m - a * 0.38f, 0.1f), thin, thin, z, z + 0.12f);
            bar(G(m - a * 0.5f - 0.1f, serif * 0.5f), G(m + a * 0.5f + 0.1f, serif * 0.5f), serif, serif, z, z + 0.12f);
            bar(G(m - a * 0.5f - 0.1f, h - serif * 0.5f), G(m + a * 0.5f + 0.1f, h - serif * 0.5f), serif, serif, z, z + 0.12f);
        }
        x += a + 0.35f;
    }
#undef G
}

/* a cock: a bar from an arbor out to a foot on its own pillar, screwed down,
 * with the arbor's jewel on top */
static void cock(float ax, float ay, float ra, float fx, float fy, float rf, int jewelled, float z0, float z1) {
    cur_mat = M_BRIDGE;
    capsule(ax, ay, ra, fx, fy, rf, z0, z1, 0.32f);
    cyl(fx, fy, rf * 0.72f, -0.01f, z0, 0.0f);
    screw(fx, fy, z1, rf * 0.42f, frand() * 3);
    if (jewelled) jewel(ax, ay, z1, fminf(ra * 0.55f, 1.6f), frand() * 3);
    float dx = fx - ax, dy = fy - ay;
    for (int k = 0; k < 3; k++) {
        float u = (k + 0.5f) / 3;
        add_disc(ax + dx * u, ay + dy * u, z0, (ra + (rf - ra) * u) * 1.05f, 0.9f);
    }
}

static void hand(float len, float ring, float ring_r, float tail, float w, float z0, float z1, int breguet) {
    cur_mat = M_BLUED;
    cyl(0, 0, w * 2.4f, z0, z1 + 0.05f, 0.08f);
    if (breguet) {
        bar(0, 0, 0, ring - ring_r * 0.9f, w * 1.5f, w * 0.9f, z0, z1);
        ann(0, ring, ring_r * 0.64f, ring_r, z0, z1, 0.05f);
        bar(0, ring + ring_r * 0.9f, 0, len, w * 0.9f, 0.02f, z0, z1);
        int n = 0;   /* a slim lozenge where the needle leaves the ring */
        float yb = ring + ring_r * 0.85f;
        cA[2 * n] = 0; cA[2 * n + 1] = yb; n++;
        cA[2 * n] = w * 1.2f; cA[2 * n + 1] = yb + 1.4f; n++;
        cA[2 * n] = 0; cA[2 * n + 1] = yb + 4.0f; n++;
        cA[2 * n] = -w * 1.2f; cA[2 * n + 1] = yb + 1.4f; n++;
        extrude(cA, NULL, n, z0, z1, 0, 0);
    } else {
        bar(0, 0, 0, len, w * 1.3f, 0.02f, z0, z1);
    }
    bar(0, 0, 0, -tail, w * 1.3f, w * 1.0f, z0, z1);
    cyl(0, -tail, w * 2.0f, z0, z1, 0.05f);
}

static void hairspring(float r0, float r1, float turns, float w, float z0, float z1, float a0) {
    cur_mat = M_BLUED;
    int n = (int)(turns * 36);
    float px[1] = {0}, py[1] = {0}, pd[2] = {0}, ps = 0;
    for (int i = 0; i <= n; i++) {
        float u = (float)i / n, ph = turns * (float)TAU * u, r = r0 + (r1 - r0) * u, a = a0 + ph;
        float dx = cosf(a), dy = sinf(a);
        float x = r * dx, y = r * dy;
        if (i) {
            float ix0 = px[0] - pd[0] * w * 0.5f, iy0 = py[0] - pd[1] * w * 0.5f, ox0 = px[0] + pd[0] * w * 0.5f, oy0 = py[0] + pd[1] * w * 0.5f;
            float ix1 = x - dx * w * 0.5f, iy1 = y - dy * w * 0.5f, ox1 = x + dx * w * 0.5f, oy1 = y + dy * w * 0.5f;
            /* top */
            vtx(ix0, iy0, z1, 0, 0, 1, ps); vtx(ox0, oy0, z1, 0, 0, 1, ps); vtx(ox1, oy1, z1, 0, 0, 1, u);
            vtx(ix0, iy0, z1, 0, 0, 1, ps); vtx(ox1, oy1, z1, 0, 0, 1, u); vtx(ix1, iy1, z1, 0, 0, 1, u);
            /* outer and inner walls */
            vtx(ox0, oy0, z0, pd[0], pd[1], 0, ps); vtx(ox1, oy1, z0, dx, dy, 0, u); vtx(ox1, oy1, z1, dx, dy, 0, u);
            vtx(ox0, oy0, z0, pd[0], pd[1], 0, ps); vtx(ox1, oy1, z1, dx, dy, 0, u); vtx(ox0, oy0, z1, pd[0], pd[1], 0, ps);
            vtx(ix1, iy1, z0, -dx, -dy, 0, u); vtx(ix0, iy0, z0, -pd[0], -pd[1], 0, ps); vtx(ix0, iy0, z1, -pd[0], -pd[1], 0, ps);
            vtx(ix1, iy1, z0, -dx, -dy, 0, u); vtx(ix0, iy0, z1, -pd[0], -pd[1], 0, ps); vtx(ix1, iy1, z1, -dx, -dy, 0, u);
        }
        px[0] = x; py[0] = y; pd[0] = dx; pd[1] = dy; ps = u;
    }
}

static float PXY[NB][2];

static void build(void) {
    spokes_style = frand() < 0.6f;
    /* where the arbors are */
    float ab = (float)(215 * PI / 180), at = (float)(25 * PI / 180), af = (float)(95 * PI / 180), ae = (float)(160 * PI / 180);
    float d_cb = MOD_B * (N_BARREL + N_CPIN) * 0.5f, d_ct = MOD_C * (N_CENTER + N_TPIN) * 0.5f;
    float d_tf = MOD_T * (N_THIRD + N_FPIN) * 0.5f, d_fe = MOD_T * (N_FOURTH + N_EPIN) * 0.5f;
    float d_mw = MOD_MW * (N_CANNON + N_MINW) * 0.5f;
    body[B_CENTER].x = 0; body[B_CENTER].y = 0;
    body[B_BARREL].x = d_cb * cosf(ab); body[B_BARREL].y = d_cb * sinf(ab);
    body[B_THIRD].x = d_ct * cosf(at); body[B_THIRD].y = d_ct * sinf(at);
    body[B_FOURTH].x = body[B_THIRD].x + d_tf * cosf(af); body[B_FOURTH].y = body[B_THIRD].y + d_tf * sinf(af);
    body[B_ESCAPE].x = body[B_FOURTH].x + d_fe * cosf(ae); body[B_ESCAPE].y = body[B_FOURTH].y + d_fe * sinf(ae);
    body[B_MINWHEEL].x = d_mw * cosf(-0.7f); body[B_MINWHEEL].y = d_mw * sinf(-0.7f);
    body[B_RATCHET].x = body[B_BARREL].x; body[B_RATCHET].y = body[B_BARREL].y;
    body[B_SECOND].x = body[B_FOURTH].x; body[B_SECOND].y = body[B_FOURTH].y;

    /* which way does the escape wheel turn, and where do its teeth rest
     * between beats? Line the lever up so a tooth sits on a pallet. */
    phi_f = (float)(5 * PI / 180);
    Pose p0, p1;
    pose_at(1000.0 + 0.25, &p0);           /* between beats */
    pose_at(1000.0 + 0.75, &p1);
    double de = remainder((double)p1.ang[B_ESCAPE] - p0.ang[B_ESCAPE], TAU);
    esc_mirror = de < 0;
    /* tooth tips are at u ~ 0.9 of each pitch (mirrored: 0.1); the entry
     * pallet sits 1.25 pitches round from the line of centres */
    {
        double pa = TAU / N_ESCAPE, tipu = esc_mirror ? -0.9 : 0.9;
        double want = phi_f + PI + 1.25 * pa;    /* from the escape wheel, towards the lever, plus a bit */
        double have = p0.ang[B_ESCAPE] + tipu * pa;
        esc_rot = (float)remainder(want - have, pa);
    }
    body[B_FORK].x = body[B_ESCAPE].x - D_EP * cosf(phi_f);
    body[B_FORK].y = body[B_ESCAPE].y - D_EP * sinf(phi_f);
    body[B_BALANCE].x = body[B_FORK].x - D_PB * cosf(phi_f);
    body[B_BALANCE].y = body[B_FORK].y - D_PB * sinf(phi_f);
    for (int i = 0; i < NB; i++) { PXY[i][0] = body[i].x; PXY[i][1] = body[i].y; }
    float bx = body[B_BALANCE].x, by = body[B_BALANCE].y;
    float cock_dir = phi_f + (float)PI + 0.42f;

    const float ZC0 = 13.0f, ZC1 = 14.6f;      /* the cocks */

    /* ---- the barrel: drum and toothed ring, turning once in six hours ---- */
    begin_body(B_BARREL, PXY[B_BARREL][0], PXY[B_BARREL][1]);
    {
        cur_mat = M_BRASS;
        int n = gear_c(cA, N_BARREL, MOD_B, 1.4f, 1.6f, 0.95f, 0);
        match_circle(cA, n, 0, 0, 16.0f, cB);
        extrude(cA, cB, n, 0.9f, 2.3f, 0, 0);
        ann(0, 0, 16.6f, 18.25f, 0.9f, 7.0f, 0.3f);
        n = circle_c(cA, 0, 0, 17.0f, 140);
        match_circle(cA, n, 0, 0, 3.2f, cB);
        extrude(cA, cB, n, 6.2f, 7.3f, 0.35f, 0);
        cur_mat = M_STEEL;
        ann(0, 0, 1.6f, 3.2f, 6.6f, 7.5f, 0.15f);
    }
    end_body(B_BARREL);
    add_disc(PXY[B_BARREL][0], PXY[B_BARREL][1], 7.3f, 18.2f, 1.0f);

    /* ---- centre wheel: once an hour; its arbor carries the minute hand ---- */
    begin_body(B_CENTER, 0, 0);
    {
        cur_mat = M_STEEL;
        int n = gear_c(cA, N_CPIN, MOD_B, 0.62f, 1.7f, 0.78f, 0);
        extrude(cA, NULL, n, 0.6f, 2.6f, 0, 0);
        arbor(0.62f, 0, 16.0f);
        wheel(N_CENTER, MOD_C, 8.0f, 9.1f, 5, 2.4f);
        /* cannon pinion over the cock */
        cur_mat = M_STEEL;
        n = gear_c(cA, N_CANNON, MOD_MW, 0.62f, 1.7f, 0.78f, 0);
        extrude(cA, NULL, n, 15.4f, 16.8f, 0, 0);
        cyl(0, 0, 0.75f, 16.8f, 23.4f, 0);
    }
    end_body(B_CENTER);
    add_disc(0, 0, 9.1f, 16.6f, 0.45f);

    begin_body(B_THIRD, PXY[B_THIRD][0], PXY[B_THIRD][1]);
    pinion(N_TPIN, MOD_C, 7.4f, 9.7f);
    arbor(0.5f, 0, ZC0);
    wheel(N_THIRD, MOD_T, 5.9f, 6.9f, 5, 1.9f);
    end_body(B_THIRD);
    add_disc(PXY[B_THIRD][0], PXY[B_THIRD][1], 6.9f, 13.1f, 0.45f);

    begin_body(B_FOURTH, PXY[B_FOURTH][0], PXY[B_FOURTH][1]);
    pinion(N_FPIN, MOD_T, 5.5f, 7.4f);
    arbor(0.45f, 0, 19.6f);
    wheel(N_FOURTH, MOD_T, 3.0f, 4.0f, 4, 1.7f);
    end_body(B_FOURTH);
    add_disc(PXY[B_FOURTH][0], PXY[B_FOURTH][1], 4.0f, 10.6f, 0.45f);

    /* ---- escape wheel: club teeth, steel ---- */
    begin_body(B_ESCAPE, PXY[B_ESCAPE][0], PXY[B_ESCAPE][1]);
    {
        pinion(N_EPIN, MOD_T, 2.6f, 4.5f);
        arbor(0.4f, 0, ZC0);
        static const float prof[][2] = {{0.0f, 0}, {0.2f, 0}, {0.42f, 0.02f}, {0.56f, 0.18f}, {0.66f, 0.42f},
                                         {0.74f, 0.7f}, {0.79f, 0.9f}, {0.83f, 0.99f}, {0.9f, 1.0f}, {0.97f, 0.96f},
                                         {1.0f, 0.9f}, {1.0f, 0.05f}};
        int np = (int)(sizeof prof / sizeof *prof);
        cur_mat = M_STEEL;
        int n = saw_c(cA, N_ESCAPE, R_ESC - 1.9f, R_ESC, prof, np, (int)esc_mirror, esc_mirror ? -esc_rot : esc_rot);
        match_circle(cA, n, 0, 0, 3.6f, cB);
        extrude(cA, cB, n, 7.4f, 8.0f, 0, 0);
        for (int k = 0; k < 4; k++) {
            float a = (float)TAU * k / 4 + 0.4f;
            bar(0.8f * cosf(a), 0.8f * sinf(a), 3.8f * cosf(a), 3.8f * sinf(a), 0.75f, 0.55f, 7.45f, 7.95f);
        }
        cyl(0, 0, 1.1f, 7.2f, 8.3f, 0.12f);
    }
    end_body(B_ESCAPE);
    add_disc(PXY[B_ESCAPE][0], PXY[B_ESCAPE][1], 8.0f, R_ESC, 0.3f);

    /* ---- the lever: pallets on the escape wheel, fork at the balance ---- */
    begin_body(B_FORK, PXY[B_FORK][0], PXY[B_FORK][1]);
    {
        cur_mat = M_STEEL;
        arbor(0.32f, 0, ZC0);
        float zf0 = 7.45f, zf1 = 8.0f;
        for (int s = -1; s <= 1; s += 2) {
            /* a pallet stone where a tooth tip meets it, 1.25 pitches from the line */
            float a = (float)PI + s * 1.25f * (float)TAU / N_ESCAPE;
            float tx = D_EP + R_ESC * cosf(a), ty = R_ESC * sinf(a);
            float ox = D_EP + (R_ESC + 1.5f) * cosf(a), oy = (R_ESC + 1.5f) * sinf(a);
            cur_mat = M_STEEL;
            capsule(0, 0, 1.0f, ox, oy, 0.75f, zf0, zf1, 0.1f);
            cur_mat = M_RUBY;
            bar(tx - 0.25f * cosf(a), ty - 0.25f * sinf(a), ox + 0.3f * cosf(a), oy + 0.3f * sinf(a), 0.55f, 0.55f, zf0 - 0.05f, zf1 + 0.05f);
        }
        cur_mat = M_STEEL;
        float tip = -D_PB + 1.25f + 0.55f;
        capsule(0, 0, 0.9f, tip + 0.4f, 0, 0.4f, zf0, zf1, 0.1f);
        capsule(tip + 0.3f, 0.75f, 0.32f, tip - 0.7f, 0.85f, 0.28f, zf0, zf1, 0.06f);
        capsule(tip + 0.3f, -0.75f, 0.32f, tip - 0.7f, -0.85f, 0.28f, zf0, zf1, 0.06f);
        cyl(0, 0, 1.15f, zf0 - 0.3f, zf1 + 0.4f, 0.15f);
    }
    end_body(B_FORK);

    /* ---- the balance: screwed bimetallic rim, three arms, roller, and the
     * hairspring (wound by the shader) ---- */
    begin_body(B_BALANCE, bx, by);
    {
        const float r_out = 9.6f, zb0 = 9.6f, zb1 = 10.5f;
        cur_mat = M_STEEL;
        cyl(0, 0, 0.36f, 0.2f, ZC0, 0);
        cyl(0, 0, 1.9f, 7.45f, 7.95f, 0.1f);                 /* roller */
        cyl(0, 0, 0.95f, 7.3f, 11.6f, 0.12f);                 /* collet / hub */
        cur_mat = M_RUBY;
        cyl(1.25f, 0, 0.26f, 7.2f, 8.3f, 0);                  /* impulse jewel */
        cur_mat = M_STEEL;
        ann(0, 0, r_out - 1.0f, r_out - 0.55f, zb0, zb1, 0.08f);
        cur_mat = M_BRASS;
        ann(0, 0, r_out - 0.56f, r_out, zb0, zb1, 0.12f);
        for (int k = 0; k < 3; k++) {
            float a = (float)TAU * k / 3 + (float)PI / 3;
            bar(0.7f * cosf(a), 0.7f * sinf(a), (r_out - 0.8f) * cosf(a), (r_out - 0.8f) * sinf(a), 1.1f, 0.8f, zb0 + 0.1f, zb1 - 0.1f);
        }
        /* timing screws round the rim, clear of the arms */
        cur_mat = M_GOLD;
        for (int k = 0; k < 18; k++) {
            float a = (float)TAU * (k + 0.5f) / 18 + (float)PI / 3;
            xf = m4_mul(m4_rotz(a), m4_mul(m4_trans(r_out - 0.05f, 0, (zb0 + zb1) * 0.5f), m4_roty((float)PI / 2)));
            xf_on = 1;
            cur_mat = M_GOLD;
            cyl(0, 0, 0.42f, 0, 0.55f, 0.12f);
            cur_mat = M_BLACK;
            bar(-0.36f, 0, 0.36f, 0, 0.1f, 0.1f, 0.42f, 0.56f);
            xf_on = 0;
        }
        /* hairspring: its outer end comes round under the cock to the stud */
        spring_end = cock_dir - phi_f;
        float turns = 11.0f;
        cur_mat = M_STEEL;
        cyl(0, 0, 1.35f, 11.0f, 11.5f, 0.1f);
        hairspring(1.35f, 5.4f, turns, 0.085f, 11.05f, 11.35f, spring_end - turns * (float)TAU);
    }
    end_body(B_BALANCE);
    add_disc(bx, by, 10.5f, 9.6f, 0.22f);

    /* ---- motion works over the cock: minute wheel and hour wheel ---- */
    begin_body(B_MINWHEEL, PXY[B_MINWHEEL][0], PXY[B_MINWHEEL][1]);
    wheel(N_MINW, MOD_MW, 15.8f, 16.5f, 4, 1.3f);
    pinion(N_MINP, MOD_HW, 16.5f, 18.1f);
    end_body(B_MINWHEEL);
    begin_body(B_HOURWHEEL, 0, 0);
    wheel(N_HOURW, MOD_HW, 17.3f, 18.0f, 4, 1.5f);
    cur_mat = M_BRASS;
    cyl(0, 0, 1.2f, 18.0f, 22.5f, 0.1f);
    end_body(B_HOURWHEEL);

    /* ---- hands ---- */
    begin_body(B_HOUR, 0, 0);
    hand(30.5f, 21.0f, 3.0f, 6.0f, 0.72f, 22.5f, 22.85f, 1);
    end_body(B_HOUR);
    begin_body(B_MINUTE, 0, 0);
    hand(48.5f, 35.0f, 2.3f, 9.0f, 0.56f, 23.3f, 23.65f, 1);
    cur_mat = M_BLUED;
    cyl(0, 0, 1.0f, 23.6f, 24.2f, 0.25f);
    end_body(B_MINUTE);
    begin_body(B_SECOND, PXY[B_SECOND][0], PXY[B_SECOND][1]);
    hand(5.8f, 0, 0, 1.8f, 0.24f, 19.9f, 20.08f, 0);
    end_body(B_SECOND);

    /* ---- ratchet wheel on the barrel bridge (still: the click holds it) ---- */
    begin_body(B_RATCHET, PXY[B_RATCHET][0], PXY[B_RATCHET][1]);
    {
        static const float prof[][2] = {{0, 0}, {0.12f, 0.06f}, {0.5f, 0.45f}, {0.85f, 0.92f}, {0.93f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.04f}};
        cur_mat = M_BRASS;
        int n = saw_c(cA, 44, 7.3f, 8.2f, prof, 7, 1, 0);
        match_circle(cA, n, 0, 0, 2.2f, cB);
        extrude(cA, cB, n, ZC1, ZC1 + 1.1f, 0, 0);
        screw(0, 0, ZC1 + 1.1f, 2.0f, 0.5f);
    }
    end_body(B_RATCHET);

    /* ---- everything that doesn't move ---- */
    begin_body(B_PLATE, 0, 0);
    cur_mat = M_PLATE;   /* the plate: perlage, chamfered edge */
    cyl(0, 0, 46.0f, -3.0f, 0.0f, 0.8f);
    end_body(B_PLATE);
    begin_body(B_STATIC, 0, 0);
    {
        /* cocks and bridges */
        float bxr = PXY[B_BARREL][0], byr = PXY[B_BARREL][1];
        /* barrel bridge: a broad round end over the barrel, foot out to the edge */
        cock(bxr, byr, 9.5f, bxr + 9.0f, byr - 20.0f, 4.6f, 0, ZC0, ZC1);
        cur_mat = M_STEEL;
        cyl(bxr, byr, 1.6f, 0.0f, ZC0, 0);
        /* the click and its spring, on the barrel bridge */
        {
            float a = -0.2f, cx = bxr + 9.6f * cosf(a), cy = byr + 9.6f * sinf(a);
            cur_mat = M_STEEL;
            capsule(cx, cy, 0.9f, bxr + 8.1f * cosf(a + 0.32f), byr + 8.1f * sinf(a + 0.32f), 0.35f, ZC1, ZC1 + 0.7f, 0.12f);
            screw(cx, cy, ZC1 + 0.7f, 0.75f, 1.0f);
            for (int k = 0; k < 8; k++) {
                float u0 = k / 8.0f, u1 = (k + 1) / 8.0f;
                float r0 = 10.6f, a0 = a - 0.1f - 0.5f * u0, a1 = a - 0.1f - 0.5f * u1;
                bar(bxr + r0 * cosf(a0), byr + r0 * sinf(a0), bxr + r0 * cosf(a1), byr + r0 * sinf(a1), 0.28f, 0.28f, ZC1, ZC1 + 0.35f);
            }
        }
        /* train bridge: centre, third and fourth, one shaped bar */
        {
            float fx = -9.5f, fy = 6.0f, ex = PXY[B_FOURTH][0] + 9.0f, ey = PXY[B_FOURTH][1] + 7.5f;
            cur_mat = M_BRIDGE;
            capsule(fx, fy, 3.6f, 0, 0, 3.6f, ZC0, ZC1, 0.32f);
            capsule(0, 0, 3.6f, PXY[B_THIRD][0], PXY[B_THIRD][1], 2.8f, ZC0, ZC1 - 0.002f, 0.32f);
            capsule(PXY[B_THIRD][0], PXY[B_THIRD][1], 2.8f, PXY[B_FOURTH][0], PXY[B_FOURTH][1], 2.7f, ZC0, ZC1 - 0.004f, 0.32f);
            capsule(PXY[B_FOURTH][0], PXY[B_FOURTH][1], 2.7f, ex, ey, 3.4f, ZC0, ZC1 - 0.006f, 0.32f);
            cyl(fx, fy, 2.5f, 0, ZC0, 0);
            cyl(ex, ey, 2.4f, 0, ZC0, 0);
            screw(fx, fy, ZC1, 1.5f, frand() * 3);
            screw(ex, ey, ZC1, 1.45f, frand() * 3);
            jewel(PXY[B_THIRD][0], PXY[B_THIRD][1], ZC1, 1.45f, 2.2f);
            jewel(PXY[B_FOURTH][0], PXY[B_FOURTH][1], ZC1, 1.45f, 0.4f);
            cur_mat = M_GOLD;
            ann(0, 0, 0.9f, 2.1f, ZC1, ZC1 + 0.35f, 0.12f);
            float pts[5][3] = {{fx, fy, 3.6f}, {0, 0, 3.6f}, {PXY[B_THIRD][0], PXY[B_THIRD][1], 2.8f},
                               {PXY[B_FOURTH][0], PXY[B_FOURTH][1], 2.7f}, {ex, ey, 3.4f}};
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 2; j++) {
                    float u = (j + 0.5f) / 2;
                    add_disc(pts[k][0] + (pts[k + 1][0] - pts[k][0]) * u, pts[k][1] + (pts[k + 1][1] - pts[k][1]) * u, ZC0,
                             pts[k][2] + (pts[k + 1][2] - pts[k][2]) * u, 0.9f);
                }
        }
        /* escape and pallet cocks */
        float ea = phi_f + 1.35f;
        cock(PXY[B_ESCAPE][0], PXY[B_ESCAPE][1], 2.3f, PXY[B_ESCAPE][0] + 8.5f * cosf(ea), PXY[B_ESCAPE][1] + 8.5f * sinf(ea), 2.8f, 1, ZC0, ZC1);
        cock(PXY[B_FORK][0], PXY[B_FORK][1], 1.9f, PXY[B_FORK][0] + 7.0f * cosf(ea + 0.25f), PXY[B_FORK][1] + 7.0f * sinf(ea + 0.25f), 2.4f, 1, ZC0, ZC1);
        /* the balance cock, with the hairspring stud under it */
        {
            float fx = bx + 15.5f * cosf(cock_dir), fy = by + 15.5f * sinf(cock_dir);
            cur_mat = M_BRIDGE;
            capsule(bx, by, 3.7f, fx, fy, 4.4f, ZC0, ZC1, 0.36f);
            cyl(fx, fy, 3.2f, 0, ZC0, 0);
            screw(fx, fy, ZC1, 1.7f, frand() * 3);
            jewel(bx, by, ZC1, 1.9f, cock_dir + 1.2f);
            cur_mat = M_STEEL;
            float sx = bx + 5.75f * cosf(cock_dir), sy = by + 5.75f * sinf(cock_dir);
            cyl(sx, sy, 0.45f, 10.9f, ZC0, 0.05f);
            /* regulator index: a slim lever over the cock, its tail on a scale */
            cur_mat = M_STEEL;
            float ia = cock_dir + 0.06f;
            ann(bx, by, 2.0f, 2.6f, ZC1, ZC1 + 0.3f, 0.06f);
            bar(bx + 2.4f * cosf(ia), by + 2.4f * sinf(ia), bx + 11.5f * cosf(ia), by + 11.5f * sinf(ia), 0.55f, 0.18f, ZC1, ZC1 + 0.3f);
            for (int k = 0; k < 3; k++) {
                float u = (k + 0.5f) / 3;
                add_disc(bx + (fx - bx) * u, by + (fy - by) * u, ZC0, 3.8f + 0.6f * u, 0.9f);
            }
        }
        /* minute wheel stud */
        cur_mat = M_STEEL;
        cyl(PXY[B_MINWHEEL][0], PXY[B_MINWHEEL][1], 0.55f, 0, 18.4f, 0.1f);
        /* seconds ring on a post from the train bridge */
        {
            float sx = PXY[B_FOURTH][0], sy = PXY[B_FOURTH][1];
            cur_mat = M_DIAL;
            ann(sx, sy, 6.3f, 7.0f, 19.4f, 19.7f, 0.05f);
            cur_mat = M_LACQUER;
            for (int k = 0; k < 60; k++) {
                float a = (float)TAU * k / 60, r1 = k % 5 ? 6.75f : 6.4f;
                bar(sx + r1 * cosf(a), sy + r1 * sinf(a), sx + 6.95f * cosf(a), sy + 6.95f * sinf(a), k % 5 ? 0.07f : 0.16f, k % 5 ? 0.07f : 0.16f, 19.7f, 19.75f);
            }
            float pa = 0.35f;
            cur_mat = M_STEEL;
            cyl(sx + 6.65f * cosf(pa), sy + 6.65f * sinf(pa), 0.35f, ZC1, 19.4f, 0);
        }
        /* chapter ring on four posts */
        cur_mat = M_DIAL;
        ann(0, 0, 43.4f, 52.0f, 20.9f, 21.6f, 0.3f);
        cur_mat = M_GOLD;
        ann(0, 0, 43.25f, 43.75f, 21.3f, 21.75f, 0.1f);
        for (int k = 0; k < 4; k++) {
            float a = (float)PI / 4 + k * (float)PI / 2;
            cur_mat = M_BRIDGE;
            cyl(45.0f * cosf(a), 45.0f * sinf(a), 1.3f, 0, 20.9f, 0.1f);
            screw(47.6f * cosf(a + 0.03f), 47.6f * sinf(a + 0.03f), 21.6f, 0.55f, a);
        }
        cur_mat = M_LACQUER;
        static const char *rom[12] = {"XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"};
        for (int h = 0; h < 12; h++) numeral(rom[h], (float)PI / 2 - h * (float)PI / 6, 44.4f, 4.1f, 21.6f);
        ann(0, 0, 49.35f, 49.5f, 21.6f, 21.7f, 0);
        ann(0, 0, 51.2f, 51.35f, 21.6f, 21.7f, 0);
        for (int k = 0; k < 60; k++) {
            float a = (float)TAU * k / 60, w = k % 5 ? 0.12f : 0.42f;
            bar(49.45f * cosf(a), 49.45f * sinf(a), 51.25f * cosf(a), 51.25f * sinf(a), w, w, 21.6f, 21.7f);
        }
        /* plate screws round the edge */
        for (int k = 0; k < 7; k++) {
            float a = (float)TAU * (k + 0.3f) / 7 + 0.2f;
            screw(42.0f * cosf(a), 42.0f * sinf(a), 0, 1.0f, frand() * 3);
        }
    }
    end_body(B_STATIC);
    add_disc(0, 0, 20.9f, 52.0f, 0.75f);
    add_disc(0, 0, 20.9f, 43.4f, -0.75f);

    body[B_ESCAPE].blur = body[B_FORK].blur = body[B_BALANCE].blur = body[B_SECOND].blur = 1;
    body[B_CENTER].blur = body[B_THIRD].blur = body[B_FOURTH].blur = 0;
}

/* ---- camera: a slow orbit that drifts between points of interest ---- */
typedef struct { float tgt[3], dist, focus_z; } Poi;
static const Poi POIS[] = {
    {{0, 0, 10}, 185, 10},         /* the whole movement */
    {{-9, 25, 9}, 66, 9},          /* escapement and balance */
    {{0, 4, 21}, 118, 23},         /* dial and hands */
    {{12, 14, 9}, 88, 8},          /* the train */
    {{-19, -12, 12}, 92, 12},      /* barrel and ratchet */
    {{-14, 25, 11}, 50, 11},       /* the balance close up */
    /* only on the side screens, spanning: */
    {{1, 26, 10}, 40, 10},         /* escape wheel and pallets */
    {{15, 15, 9}, 56, 9},          /* third and fourth wheels meshing */
    {{-19, -13, 7}, 58, 7},        /* the ratchet and click */
    {{8, 10, 9}, 64, 9},           /* the train, closer than the tour's view of it */
};
#define NPOI ((int)(sizeof POIS / sizeof *POIS))
static int poi_order[64];
static float cam_ph[6];

/* Which views each role tours: the lone screen all six of the originals;
 * spanning, the front one stays on the whole movement and the dial while
 * the others hold close-ups of the escapement and of the train and barrel. */
static void camera_init(void) {
    static const int all[] = {0, 1, 2, 3, 4, 5}, front[] = {0, 2}, esc[] = {1, 6, 5}, train[] = {9, 7, 4, 8};
    const int *pool = role == R_FRONT ? front : role == R_ESCAPEMENT ? esc : role == R_TRAIN ? train : all;
    int np = role == R_FRONT ? 2 : role == R_ESCAPEMENT ? 3 : role == R_TRAIN ? 4 : 6;
    int n = 0;
    for (int round = 0; n < 64; round++) {
        int o[NPOI];
        for (int i = 0; i < np; i++) o[i] = pool[i];
        for (int i = np - 1; i > 0; i--) { int j = (int)(frand() * (i + 1)); if (j > i) j = i; int t = o[i]; o[i] = o[j]; o[j] = t; }
        for (int i = 0; i < np && n < 64; i++) {
            if (n && o[i] == poi_order[n - 1]) continue;
            poi_order[n++] = o[i];
        }
        (void)round;
    }
    if (role == R_ALL) {
        poi_order[0] = 0;   /* open on the whole movement */
        if (poi_order[1] == 0) poi_order[1] = 2;
    } else if (role == R_FRONT) {
        for (int i = 0; i < 64; i++) poi_order[i] = i % 2 ? 2 : 0;
    }
    for (int i = 0; i < 6; i++) cam_ph[i] = frand() * (float)TAU;
}

static void camera_at(double t, float *eye, float *tgt, float *focus) {
    const double seg = 34, glide = 10;
    /* spanning, the side screens change views at other moments than the front */
    if (role == R_ESCAPEMENT) t += 11;
    if (role == R_TRAIN) t += 22;
    double s = t / seg;
    int k = (int)floor(s);
    double f = (s - k) * seg - (seg - glide);
    f = f < 0 ? 0 : f / glide;
    f = f * f * f * (f * (f * 6 - 15) + 10);
    const Poi *a = &POIS[poi_order[((k % 64) + 64) % 64]], *b = &POIS[poi_order[(((k + 1) % 64) + 64) % 64]];
    float u = (float)f;
    for (int i = 0; i < 3; i++) tgt[i] = a->tgt[i] + (b->tgt[i] - a->tgt[i]) * u;
    float dist = expf(logf(a->dist) + (logf(b->dist) - logf(a->dist)) * u);
    /* the field of view is vertical: on a portrait screen, back off so the
     * same width shows (the whole movement, not a slice of it) */
    if (W < H) dist *= (float)H / W;
    float fz = a->focus_z + (b->focus_z - a->focus_z) * u;
    float az = 0.5f * sinf((float)(t * TAU / 83) + cam_ph[0]) + 0.16f * sinf((float)(t * TAU / 31) + cam_ph[1]);
    float el = 0.14f + 0.26f * sinf((float)(t * TAU / 67) + cam_ph[2]) + 0.06f * sinf((float)(t * TAU / 23) + cam_ph[3]);
    eye[0] = tgt[0] + dist * sinf(az) * cosf(el);
    eye[1] = tgt[1] + dist * sinf(el);
    eye[2] = tgt[2] + dist * cosf(az) * cosf(el);
    float fp[3] = {tgt[0], tgt[1], fz};
    float dx = fp[0] - eye[0], dy = fp[1] - eye[1], dz = fp[2] - eye[2];
    *focus = sqrtf(dx * dx + dy * dy + dz * dz);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
#define BG_MID 0.028f, 0.020f, 0.014f, 1

static const char *VS_SCENE = "#version 330 core\n" GLSL(
    layout(location = 0) in vec3 aP;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aM;          /* material, hairspring position (0 inner .. 1 stud) */
    uniform mat4 uModel, uVP;
    uniform float uSpring;                    /* balance angle: the spring winds against it */
    out vec3 vW, vN, vO, vT;
    flat out int vMat;
    void main() {
        vec3 p = aP, n = aN;
        if (aM.y > 0.0) {
            float a = -uSpring * aM.y;
            float c = cos(a), s = sin(a);
            p.xy = mat2(c, s, -s, c) * p.xy;
            n.xy = mat2(c, s, -s, c) * n.xy;
        }
        vec4 w = uModel * vec4(p, 1.0);
        vW = w.xyz;
        vN = mat3(uModel) * n;
        vO = p;
        vec2 tg = vec2(-p.y, p.x);
        vT = mat3(uModel) * vec3(tg / max(length(tg), 1e-4), 0.0);
        vMat = int(aM.x + 0.5);
        gl_Position = uVP * w;
    });

static const char *FS_DEPTH = "#version 330 core\n" GLSL(
    void main() {});

static const char *FS_SCENE = "#version 330 core\n" GLSL(
    in vec3 vW, vN, vO, vT;
    flat in int vMat;
    uniform vec3 uCam, uL, uLcol;
    uniform sampler2DShadow uShadow;
    uniform mat4 uLVP;
    uniform float uShTexel;
    uniform sampler3D uAO;
    uniform vec3 uAOmin, uAOinv;
    layout(location = 0) out vec4 o;
    layout(location = 1) out float oDepth;     /* view distance, for the lens */

    const float PI = 3.14159265;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

    vec3 lobe(vec3 r, vec3 d, vec3 c, float w, float rough) {
        float s = w * w + rough * rough * 1.6;
        return c * (w * w / s) * exp((dot(r, d) - 1.0) / s);
    }
    /* a dark, warm studio: a big softbox up and to the left, an amber card
     * on the right, a low warm bounce, a strip light above */
    vec3 env(vec3 r, float rough) {
        vec3 c = vec3(0.022, 0.016, 0.011) * (0.8 + 0.5 * r.y);
        c += lobe(r, normalize(vec3(-0.45, 0.55, 0.70)), vec3(1.0, 0.83, 0.62) * 2.4, 0.26, rough);
        c += lobe(r, normalize(vec3(0.15, 0.05, 1.0)), vec3(1.0, 0.78, 0.52) * 0.55, 0.20, rough);
        c += lobe(r, normalize(vec3(0.80, 0.10, 0.55)), vec3(1.0, 0.58, 0.26) * 1.3, 0.18, rough);
        c += lobe(r, normalize(vec3(-0.30, -0.55, 0.75)), vec3(0.62, 0.42, 0.26) * 0.35, 0.40, rough);
        c += lobe(r, normalize(vec3(0.20, 0.97, 0.10)), vec3(1.0, 0.9, 0.75) * 0.45, 0.14, rough);
        return c;
    }

    float shadow(vec3 p, vec3 n) {
        vec4 l = uLVP * vec4(p + n * 0.06 + uL * 0.04, 1.0);
        vec3 c = l.xyz * 0.5 + 0.5;
        const vec2 k[8] = vec2[](vec2(-0.613, 0.354), vec2(0.170, -0.862), vec2(0.784, 0.373), vec2(-0.265, -0.394),
                                 vec2(0.322, 0.110), vec2(-0.880, -0.260), vec2(0.040, 0.770), vec2(0.580, -0.540));
        float s = 0.0;
        for (int i = 0; i < 8; i++) s += texture(uShadow, vec3(c.xy + k[i] * uShTexel * 2.2, c.z - 0.0004));
        return s / 8.0;
    }

    /* baked occlusion (see bake_ao), for what faces up into it */
    float discAO(vec3 p, vec3 n) {
        if (n.z < 0.1) return 1.0;
        float a = texture(uAO, (p + vec3(0.0, 0.0, 0.3) - uAOmin) * uAOinv).r;
        return mix(1.0, a, n.z);
    }

    void main() {
        vec3 N = normalize(vN);
        vec3 V = normalize(uCam - vW);
        if (dot(N, V) < 0.0 && !gl_FrontFacing) N = -N;
        vec3 T = vT - N * dot(vT, N);
        T = dot(T, T) > 1e-6 ? normalize(T) : normalize(cross(N, vec3(0.31, 0.53, 0.79)));
        vec2 uv = vO.xy;
        bool top = N.z > 0.93, bev = N.z > 0.3 && N.z <= 0.93;

        vec3 F0 = vec3(0.04), alb = vec3(0.0), emit = vec3(0.0);
        float rT = 0.3, rB = 0.3;
        int m = vMat;
        if (m == 0) {                          /* gilt brass, circular graining */
            F0 = vec3(0.98, 0.72, 0.36);
            if (top) {
                float ring = hash(vec2(floor(length(uv) * 3.0), 7.0));
                rT = 0.10; rB = 0.38 + 0.08 * ring;
                F0 *= 0.92 + 0.08 * ring;
            } else if (bev) { rT = rB = 0.10; }
            else { rT = rB = 0.2; }
        } else if (m == 1) {                   /* plate: dark brass, perlage */
            F0 = vec3(0.30, 0.225, 0.15);
            if (top) {
                float s = 1.35;
                vec2 g = uv / s;
                vec2 base = floor(g);
                /* spots overlap like scales: the latest (highest row, then
                 * column) that covers this point is on top */
                vec2 bc = base * s; float id = 0.0;
                for (int k = 0; k < 9; k++) {
                    vec2 c = base + vec2(1 - k % 3, 1 - k / 3);
                    vec2 cc = (c + 0.5 + (vec2(hash(c), hash(c + 7.7)) - 0.5) * 0.45) * s;
                    if (length(uv - cc) < 0.98 * s) { bc = cc; id = hash(c + 3.1); break; }
                }
                vec2 q = uv - bc;
                float r = length(q);
                T = normalize(vec3(-q.y, q.x, 0.0) / max(r, 1e-3) + vec3(1e-4));
                rT = 0.26; rB = 0.42;
                F0 *= 0.92 + 0.12 * id;
            } else if (bev) { rT = rB = 0.12; F0 *= 1.2; }
            else { rT = rB = 0.3; }
        } else if (m == 2) {                   /* cocks: aged brass, Geneva stripes, polished bevels */
            F0 = vec3(0.40, 0.36, 0.31);
            if (top) {
                vec2 D = normalize(vec2(0.83, 0.56)), Dp = vec2(-D.y, D.x);
                float s = dot(uv, Dp) / 2.2, f = fract(s) - 0.5;
                vec2 along = normalize(D + Dp * f * 0.9);
                T = vec3(along, 0.0);
                N = normalize(N + vec3(Dp * f * 0.12, 0.0));
                rT = 0.10; rB = 0.36;
                F0 *= 0.9 + 0.1 * hash(vec2(floor(s), 1.0));
            } else if (bev) { rT = rB = 0.05; F0 = vec3(0.70, 0.66, 0.60); }
            else { rT = rB = 0.22; }
        } else if (m == 3) {                   /* polished steel */
            F0 = vec3(0.60, 0.60, 0.62);
            if (top) { rT = 0.06; rB = 0.22; } else { rT = rB = 0.10; }
        } else if (m == 4) {                   /* blued steel */
            F0 = vec3(0.05, 0.075, 0.19);
            rT = rB = 0.2;
        } else if (m == 5) {                   /* ruby */
            F0 = vec3(0.08);
            alb = vec3(0.07, 0.002, 0.006);
            rT = rB = 0.08;
        } else if (m == 6) {                   /* polished gold */
            F0 = vec3(1.0, 0.76, 0.40);
            rT = rB = 0.09;
        } else if (m == 7) {                   /* silvered dial, cream */
            F0 = vec3(0.70, 0.62, 0.48);
            if (top) { rT = 0.34; rB = 0.55; } else { rT = rB = 0.2; }
        } else if (m == 8) {                   /* black lacquer */
            alb = vec3(0.008, 0.007, 0.006);
            F0 = vec3(0.045);
            rT = rB = 0.18;
        } else {                               /* slots: nothing */
            alb = vec3(0.004);
            F0 = vec3(0.02);
            rT = rB = 0.6;
        }
        vec3 B = cross(N, T);
        T = cross(B, N);

        float NoV = max(dot(N, V), 1e-3);
        float sh = shadow(vW, normalize(vN));
        float ao = discAO(vW, normalize(vN));

        /* key light: anisotropic GGX */
        vec3 L = uL, Hh = normalize(L + V);
        float NoL = max(dot(N, L), 0.0);
        float ax = max(rT * rT, 0.012), ay = max(rB * rB, 0.012);
        float ToH = dot(T, Hh), BoH = dot(B, Hh), NoH = dot(N, Hh);
        float d = ToH * ToH / (ax * ax) + BoH * BoH / (ay * ay) + NoH * NoH;
        float D = 1.0 / (PI * ax * ay * d * d);
        float a2 = ax * ay;
        float Vis = 0.5 / (NoL * sqrt(NoV * NoV * (1.0 - a2) + a2) + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2) + 1e-5);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(L, Hh), 0.0), 5.0);
        vec3 col = D * Vis * F * NoL * uLcol * sh;
        col += alb * NoL * uLcol * sh / PI;

        /* environment: the reflection, bent along the grain */
        float r = sqrt(rT * rB);
        vec3 aniDir = rT < rB ? B : T;
        float an = abs(rB - rT) / max(rB + rT, 1e-3);
        vec3 aT = cross(aniDir, V), aN = cross(aT, aniDir);
        vec3 Nb = normalize(mix(N, aN, clamp(an * 0.8, 0.0, 0.9)));
        vec3 R = reflect(-V, Nb);
        vec3 Fe = F0 + (max(vec3(1.0 - r), F0) - F0) * pow(1.0 - NoV, 5.0);
        float specAO = mix(1.0, ao, 0.85) * mix(0.55, 1.0, sh);
        col += env(R, r) * Fe * specAO;
        if (m >= 5 && m != 6 && m != 7) col += alb * env(N, 1.0) * ao * 0.9;
        if (m == 5) {                          /* light through the stone */
            float through = pow(1.0 - NoV, 1.5);
            col += vec3(0.30, 0.006, 0.02) * (0.05 + 0.35 * sh * pow(max(dot(-reflect(L, N), V), 0.0), 4.0) + 0.3 * through) * ao;
        }
        o = vec4(min(col + emit, vec3(60.0)), 1.0);
        oDepth = distance(uCam, vW);
    });

static const char *VS_TRI = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* depth of field, at half resolution: a gather over a golden-angle disc
 * sized by the circle of confusion */
static const char *FS_PREP = "#version 330 core\n" GLSL(
    uniform sampler2D uCol;
    uniform sampler2DMS uDep;                 /* view distance; 1000 for nothing */
    uniform vec2 uSize;                       /* scene size, px */
    uniform vec2 uLens;                       /* focus distance, aperture (px) */
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy * 2.0 / uSize;
        ivec2 p = ivec2(gl_FragCoord.xy) * 2;
        float z = min(texelFetch(uDep, p, 0).r, texelFetch(uDep, p + ivec2(1), 3).r);
        float c = z > 990.0 ? 0.0 : clamp(uLens.y * (z - uLens.x) / z, -14.0, 14.0);
        o = vec4(texture(uCol, uv).rgb, c);
    });
static const char *FS_DOF = "#version 330 core\n" GLSL(
    uniform sampler2D uPrep;                  /* half res: colour, circle of confusion (scene px) */
    uniform vec2 uHalf;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / uHalf;
        vec4 c = texture(uPrep, uv);
        float R = abs(c.a) * 0.5;             /* in half-res px */
        if (R < 0.4) { o = c; return; }
        vec3 acc = c.rgb;
        float ws = 1.0;
        for (int i = 0; i < 20; i++) {
            float fi = float(i) + 0.5, rr = sqrt(fi / 20.0) * R, a = fi * 2.39996;
            vec4 s = texture(uPrep, uv + vec2(cos(a), sin(a)) * rr / uHalf);
            float w = clamp(abs(s.a) * 0.5 - rr + 1.0, 0.0, 1.0);
            acc += s.rgb * w;
            ws += w;
        }
        o = vec4(acc / ws, c.a);
    });

static const char *FS_FINAL = "#version 330 core\n" GLSL(
    uniform sampler2D uCol, uDof, uPrep;
    uniform sampler2DMS uZ;
    uniform vec2 uOut, uSize;
    uniform float uTime, uExposure;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    /* Catmull-Rom upscale (5 taps, corners dropped), as in flightsim */
    vec3 catmull(sampler2D t, vec2 uv) {
        vec2 sz = vec2(textureSize(t, 0)), p = uv * sz - 0.5, c = floor(p) + 0.5, f = p - floor(p);
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
        vec2 w12 = w1 + w2, t0 = (c - 1.0) / sz, t3 = (c + 2.0) / sz, t12 = (c + w2 / w12) / sz;
        vec3 r = texture(t, vec2(t12.x, t0.y)).rgb * w12.x * w0.y + texture(t, vec2(t0.x, t12.y)).rgb * w0.x * w12.y +
                 texture(t, t12).rgb * w12.x * w12.y + texture(t, vec2(t3.x, t12.y)).rgb * w3.x * w12.y +
                 texture(t, vec2(t12.x, t3.y)).rgb * w12.x * w3.y;
        float ws = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
        return max(r / ws, vec3(0.0));
    }
    void main() {
        vec2 uv = gl_FragCoord.xy / uOut;
        vec3 sharp = uSize.x < uOut.x - 0.5 ? catmull(uCol, uv) : texture(uCol, uv).rgb;
        float c = texture(uPrep, uv).a;
        vec3 blur = texture(uDof, uv).rgb;
        vec3 col = mix(sharp, blur, smoothstep(0.7, 2.2, abs(c)));
        if (texelFetch(uZ, ivec2(uv * uSize), 0).r > 990.0) {      /* the warm dark behind it all */
            vec2 q = (uv - 0.5) * vec2(uOut.x / uOut.y, 1.0);
            float g = exp(-dot(q - vec2(-0.1, 0.08), q - vec2(-0.1, 0.08)) * 2.4);
            col = mix(vec3(0.012, 0.0085, 0.006), vec3(0.040, 0.028, 0.020), g);
        }
        vec3 bl = textureLod(uDof, uv, 2.0).rgb * 0.25 + textureLod(uDof, uv, 3.5).rgb * 0.4 + textureLod(uDof, uv, 5.0).rgb * 0.35;
        col += min(max(bl - 0.5, 0.0), 1.0) * 0.16;
        col = aces(col * uExposure);
        vec2 q = uv - 0.5;
        q.x *= uOut.x / uOut.y;
        col *= mix(1.0, 0.6, smoothstep(0.35, 1.0, length(q)));
        col = pow(col, vec3(1.0 / 2.2));
        col += (hash(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5) * (1.5 / 255.0);
        o = vec4(col, 1.0);
    });

static GLuint prog_scene, prog_depth, prog_prep, prog_dof, prog_final;
static GLuint vao_mesh, vbo_mesh, vao_empty;
static int nverts;
static GLuint fbo_shadow, tex_shadow;
static const int SHADOW = 1536;
static GLuint fbo_ms, rb_ms_col, tex_ms_z, rb_ms_dep, fbo_res, tex_res_col, fbo_prep, tex_prep, fbo_dof, tex_dof;
static int rs_w, rs_h, msaa = 4;
static float render_scale = 1.0f;
static struct { GLint model, vp, spring, cam, l, lcol, lvp, shtexel; } us;
static GLuint tex_ao;
static struct { GLint model, vp, spring; } ud;
static M4 light_vp;
static float light_dir[3];

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "clockwork: shader:\n%s\n", log);
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
        fprintf(stderr, "clockwork: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, GLenum filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : (GLint)filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

/* (re)make the render targets for a scene of w x h */
static void targets(int w, int h) {
    if (w == rs_w && h == rs_h) return;
    if (rs_w) {
        glDeleteFramebuffers(1, &fbo_ms); glDeleteRenderbuffers(1, &rb_ms_col); glDeleteTextures(1, &tex_ms_z);
        glDeleteRenderbuffers(1, &rb_ms_dep);
        glDeleteFramebuffers(1, &fbo_res); glDeleteTextures(1, &tex_res_col);
        glDeleteFramebuffers(1, &fbo_dof); glDeleteTextures(1, &tex_dof);
        glDeleteFramebuffers(1, &fbo_prep); glDeleteTextures(1, &tex_prep);
    }
    rs_w = w; rs_h = h;
    glGenRenderbuffers(1, &rb_ms_col);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_col);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, GL_R11F_G11F_B10F, w, h);
    /* the lens needs depth, but sampling a multisampled depth buffer is
     * slow here (it gets decompressed first): the shading writes its view
     * distance to a small second target instead, read a sample at a time */
    glGenTextures(1, &tex_ms_z);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, msaa, GL_R16F, w, h, GL_TRUE);
    glGenRenderbuffers(1, &rb_ms_dep);
    glBindRenderbuffer(GL_RENDERBUFFER, rb_ms_dep);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, GL_DEPTH_COMPONENT24, w, h);
    glGenFramebuffers(1, &fbo_ms);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_ms_col);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb_ms_dep);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("multisampled framebuffer incomplete");

    tex_res_col = mk_tex(GL_R11F_G11F_B10F, w, h, GL_RGB, GL_FLOAT, GL_LINEAR);
    glGenFramebuffers(1, &fbo_res);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_res);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_res_col, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("resolve framebuffer incomplete");

    tex_prep = mk_tex(GL_RGBA16F, (w + 1) / 2, (h + 1) / 2, GL_RGBA, GL_HALF_FLOAT, GL_LINEAR);
    glGenFramebuffers(1, &fbo_prep);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_prep);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_prep, 0);
    tex_dof = mk_tex(GL_RGBA16F, (w + 1) / 2, (h + 1) / 2, GL_RGBA, GL_HALF_FLOAT, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGenFramebuffers(1, &fbo_dof);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dof);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_dof, 0);
}

static void gl_init(void) {
    prog_scene = mk_prog(VS_SCENE, FS_SCENE);
    prog_depth = mk_prog(VS_SCENE, FS_DEPTH);
    prog_prep = mk_prog(VS_TRI, FS_PREP);
    prog_dof = mk_prog(VS_TRI, FS_DOF);
    prog_final = mk_prog(VS_TRI, FS_FINAL);
    glGenVertexArrays(1, &vao_empty);
    glEnable(GL_CULL_FACE);   /* the camera and the light are always in front */

    /* weld identical vertices: an index buffer, so shared corners are
     * transformed once */
    unsigned *idx = malloc((size_t)nv * sizeof *idx);
    int hsize = 1;
    while (hsize < nv * 2) hsize <<= 1;
    int *ht = malloc((size_t)hsize * sizeof *ht);
    memset(ht, -1, (size_t)hsize * sizeof *ht);
    int nu = 0;
    for (int i = 0; i < nv; i++) {
        uint32_t h = 2166136261u;
        const unsigned char *b = (const unsigned char *)&vb[i];
        for (size_t k = 0; k < sizeof(Vtx); k++) h = (h ^ b[k]) * 16777619u;
        int slot = (int)(h & (uint32_t)(hsize - 1));
        while (ht[slot] >= 0 && memcmp(&vb[ht[slot]], &vb[i], sizeof(Vtx))) slot = (slot + 1) & (hsize - 1);
        if (ht[slot] < 0) { vb[nu] = vb[i]; ht[slot] = nu++; }
        idx[i] = (unsigned)ht[slot];
    }
    free(ht);
    nverts = nu;
    glGenVertexArrays(1, &vao_mesh);
    glBindVertexArray(vao_mesh);
    glGenBuffers(1, &vbo_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mesh);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)nu * (GLsizeiptr)sizeof(Vtx), vb, GL_STATIC_DRAW);
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)nv * 4, idx, GL_STATIC_DRAW);
    free(idx);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)12);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)24);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    free(vb);
    vb = NULL;

    GLint maxs = 4;
    glGetIntegerv(GL_MAX_SAMPLES, &maxs);
    if (msaa > maxs) msaa = maxs;

    tex_shadow = mk_tex(GL_DEPTH_COMPONENT24, SHADOW, SHADOW, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_shadow, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    /* the key light, high on the left, a little in front */
    light_dir[0] = -0.42f; light_dir[1] = 0.58f; light_dir[2] = 0.70f;
    v3norm(light_dir);
    float le[3] = {light_dir[0] * 150, light_dir[1] * 150, 10 + light_dir[2] * 150}, lt[3] = {0, 0, 10}, up[3] = {0, 0, 1};
    light_vp = m4_mul(m4_ortho(-60, 60, -60, 60, 60, 240), m4_look(le, lt, up));

    us.model = glGetUniformLocation(prog_scene, "uModel");
    us.vp = glGetUniformLocation(prog_scene, "uVP");
    us.spring = glGetUniformLocation(prog_scene, "uSpring");
    us.cam = glGetUniformLocation(prog_scene, "uCam");
    us.l = glGetUniformLocation(prog_scene, "uL");
    us.lcol = glGetUniformLocation(prog_scene, "uLcol");
    us.lvp = glGetUniformLocation(prog_scene, "uLVP");
    us.shtexel = glGetUniformLocation(prog_scene, "uShTexel");
    ud.model = glGetUniformLocation(prog_depth, "uModel");
    ud.vp = glGetUniformLocation(prog_depth, "uVP");
    ud.spring = glGetUniformLocation(prog_depth, "uSpring");

    glUseProgram(prog_scene);
    glUniform1i(glGetUniformLocation(prog_scene, "uShadow"), 0);
    glUniform3fv(us.l, 1, light_dir);
    glUniform3f(us.lcol, 3.4f, 2.85f, 2.2f);
    glUniformMatrix4fv(us.lvp, 1, GL_FALSE, light_vp.m);
    glUniform1f(us.shtexel, 1.0f / SHADOW);
    glUniform1i(glGetUniformLocation(prog_scene, "uAO"), 1);
    glUniform3fv(glGetUniformLocation(prog_scene, "uAOmin"), 1, AO_MIN);
    glUniform3f(glGetUniformLocation(prog_scene, "uAOinv"), 1 / AO_SIZE[0], 1 / AO_SIZE[1], 1 / AO_SIZE[2]);
    unsigned char *ao = bake_ao();
    glGenTextures(1, &tex_ao);
    glBindTexture(GL_TEXTURE_3D, tex_ao);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, AO_N, AO_N, AO_Z, 0, GL_RED, GL_UNSIGNED_BYTE, ao);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    free(ao);
    glUseProgram(prog_prep);
    glUniform1i(glGetUniformLocation(prog_prep, "uCol"), 0);
    glUniform1i(glGetUniformLocation(prog_prep, "uDep"), 1);
    glUseProgram(prog_dof);
    glUniform1i(glGetUniformLocation(prog_dof, "uPrep"), 3);
    glUseProgram(prog_final);
    glUniform1i(glGetUniformLocation(prog_final, "uCol"), 0);
    glUniform1i(glGetUniformLocation(prog_final, "uDof"), 2);
    glUniform1i(glGetUniformLocation(prog_final, "uPrep"), 3);
    glUniform1i(glGetUniformLocation(prog_final, "uZ"), 4);
}

static void draw_body(int i, const Pose *p, GLint lmodel, GLint lspring) {
    M4 m = m4_mul(m4_trans(body[i].x, body[i].y, 0), m4_rotz(p->ang[i]));
    glUniformMatrix4fv(lmodel, 1, GL_FALSE, m.m);
    glUniform1f(lspring, i == B_BALANCE ? p->alpha : 0.0f);
    glDrawElements(GL_TRIANGLES, body[i].count, GL_UNSIGNED_INT, (void *)((size_t)body[i].first * 4));
}

static const float NEAR = 12.0f, FAR = 520.0f, FOV = 30.0f;

/* the scene's render scale to start with: about a megapixel (4x MSAA),
 * upscaled with Catmull-Rom; live, it then follows the frame budget */
static float default_scale(void) {
    float s = sqrtf(1.0e6f / ((float)W * H));
    return s > 1 ? 1 : s;
}

/* --bench: GPU time per pass, from timestamps between them */
enum { PR_SHADOW, PR_SCENE, PR_BLUR, PR_RESOLVE, PR_DOF, PR_FINAL, PR_N };
static const char *prof_name[PR_N] = {"shadow", "scene", "mblur", "resolve", "dof", "final"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    GLuint64 ts[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &ts[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (ts[i + 1] - ts[i]) * 1e-6;
}

/* t: clock time (s since midnight); ct: camera time; dt: frame time (for the shutter) */
static void render(double t, double ct, double dt, GLuint target) {
    int sw = (int)(W * render_scale + 0.5f), sh = (int)(H * render_scale + 0.5f);
    if (sw < 16) sw = 16;
    if (sh < 16) sh = 16;
    targets(sw, sh);
    Pose pose;
    pose_at(t, &pose);

    /* shadow map */
    prof_mark(PR_SHADOW);
    glBindVertexArray(vao_mesh);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, SHADOW, SHADOW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.5f, 3.0f);
    glUseProgram(prog_depth);
    glUniformMatrix4fv(ud.vp, 1, GL_FALSE, light_vp.m);
    for (int j = 0; j < NB; j++) draw_body(draw_order[j], &pose, ud.model, ud.spring);
    glDisable(GL_POLYGON_OFFSET_FILL);

    /* scene */
    prof_mark(PR_SCENE);
    float eye[3], tgt[3], focus;
    camera_at(ct, eye, tgt, &focus);
    float up[3] = {0.04f * sinf((float)(ct * 0.05)), 1, 0};
    M4 vp = m4_mul(m4_persp(FOV * (float)PI / 180, (float)W / H, NEAR, FAR), m4_look(eye, tgt, up));
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ms);
    glViewport(0, 0, sw, sh);
    /* the background is painted in at the end; clear to its middling tone so
     * antialiased edges blend towards about the right colour */
    static const float clear_col[4] = {BG_MID}, clear_z[4] = {1000, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 0, clear_col);
    glClearBufferfv(GL_COLOR, 1, clear_z);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glBindVertexArray(vao_mesh);
    glUseProgram(prog_scene);
    glUniformMatrix4fv(us.vp, 1, GL_FALSE, vp.m);
    glUniform3fv(us.cam, 1, eye);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, tex_ao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    /* the fast parts at four moments across a 3/4-frame shutter, one MSAA
     * sample each: resolving the samples is the motion blur. Only what
     * actually moves in that time (the lever and escape wheel rest between
     * ticks) needs drawing four times. */
    Pose sub[4];
    int moving[NB] = {0};
    for (int k = 0; k < 4; k++) pose_at(t - dt * 0.75 * (k / 3.0), &sub[k]);
    for (int i = 0; i < NB; i++)
        if (body[i].blur && msaa >= 4)
            moving[i] = fabs(remainder((double)sub[3].ang[i] - sub[0].ang[i], TAU)) > 2e-4 ||
                        (i == B_BALANCE && fabsf(sub[3].alpha - sub[0].alpha) > 2e-4f);
    for (int j = 0; j < NB; j++)
        if (!moving[draw_order[j]]) draw_body(draw_order[j], &pose, us.model, us.spring);
    prof_mark(PR_BLUR);
    if (msaa >= 4) {
        glEnable(GL_SAMPLE_MASK);
        for (int k = 0; k < 4; k++) {
            glSampleMaski(0, 1u << k);
            for (int i = 0; i < NB; i++)
                if (moving[i]) draw_body(i, &sub[k], us.model, us.spring);
        }
        glSampleMaski(0, ~0u);
        glDisable(GL_SAMPLE_MASK);
    }

    /* resolve colour and depth */
    prof_mark(PR_RESOLVE);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ms);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_res);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, sw, sh, 0, 0, sw, sh, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    /* depth of field */
    prof_mark(PR_DOF);
    float aper = 30.0f * sh / 1080.0f;
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glBindVertexArray(vao_empty);
    int hw = (sw + 1) / 2, hh = (sh + 1) / 2;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_prep);
    glViewport(0, 0, hw, hh);
    glUseProgram(prog_prep);
    glUniform2f(glGetUniformLocation(prog_prep, "uSize"), (float)sw, (float)sh);
    glUniform2f(glGetUniformLocation(prog_prep, "uLens"), focus, aper);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_res_col);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_dof);
    glUseProgram(prog_dof);
    glUniform2f(glGetUniformLocation(prog_dof, "uHalf"), (float)hw, (float)hh);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_prep);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_dof);
    glGenerateMipmap(GL_TEXTURE_2D);

    /* tonemap onto the target */
    prof_mark(PR_FINAL);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_final);
    glUniform2f(glGetUniformLocation(prog_final, "uOut"), (float)W, (float)H);
    glUniform2f(glGetUniformLocation(prog_final, "uSize"), (float)sw, (float)sh);
    glUniform1f(glGetUniformLocation(prog_final, "uTime"), (float)fmod(ct, 1000.0));
    glUniform1f(glGetUniformLocation(prog_final, "uExposure"), 0.92f);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex_ms_z);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_res_col);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_mark(PR_N);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "clockwork");
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
        "usage: clockwork [--window | --root | --layer] [--seed N] [--scale X] [--epoch UNIXTIME]\n"
        "                 [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                 [--shot FILE.ppm | --bench FRAMES] [--size WxH] [--at SECONDS]\n"
        "  --scale X   fixed render scale (default: adapts to the frame budget)\n"
        "  --at S      start the camera path S seconds in\n");
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
 * shows the whole movement; the rest, left to right, take turns at the
 * escapement and the train. */
static int pick_role(void) {
    if (span_me < 0) return R_ALL;
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
    return rank % 2 ? R_TRAIN : R_ESCAPEMENT;
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
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    role = pick_role();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed * 2654435761u | 1;
    for (int i = 0; i < 4; i++) frand();
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
        if (offscreen) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Clockwork");
        XClassHint chint = {"clockwork", "Clockwork"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    build();
    camera_init();
    gl_init();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        render_scale = opt_scale > 0 ? opt_scale : default_scale();
        /* with --epoch, the moment --at seconds after it, so that separate
         * instances' shots match */
        double t0c = opt_epoch ? clock_at(opt_epoch + opt_at) : clock_now();
        render(t0c, opt_at, 1.0 / 60, fbo);
        glFinish();
        if (opt_shot) {
            save_ppm(opt_shot);
            return 0;
        }
        double t0 = now(), worst = 0;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            render(t0c + (i + 1) / 60.0, opt_at + (i + 1) / 60.0, 1.0 / 60, fbo);
            glFinish();
            prof_collect();
            double d = now() - f0;
            if (d > worst) worst = d;
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d scale %.2f msaa %d: %.2f ms/frame avg, %.2f worst (%s)\n", W, H, render_scale, msaa, per * 1e3,
               worst * 1e3, (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < PR_N; i++) printf("  %-8s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        printf("  %d vertices, %d triangles\n", nverts, nv / 3);
        return 0;
    }

    /* GPU time per frame, from timer queries read a couple of frames late,
     * steers the render scale (unless --scale): several of us may share
     * one small GPU */
    GLuint q[3];
    glGenQueries(3, q);
    int qn = 0;
    render_scale = opt_scale > 0 ? opt_scale : default_scale();
    double avg = 0.004, last_change = 0;
    double t0 = now(), last = t0, c0 = clock_now() - t0;
    for (int frame = 0;; frame++) {
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
        if (frame % 600 == 0) c0 = clock_now() - t;   /* follow the wall clock (DST, suspend) */
        if (frame >= 3) {
            GLuint64 ns = 0;
            GLint ready = 0;
            glGetQueryObjectiv(q[qn], GL_QUERY_RESULT_AVAILABLE, &ready);
            if (ready) {
                glGetQueryObjectui64v(q[qn], GL_QUERY_RESULT, &ns);
                avg += (ns * 1e-9 - avg) * 0.1;
            }
        }
        glBeginQuery(GL_TIME_ELAPSED, q[qn]);
        /* spanning, the camera paths all run off the shared epoch */
        double ct = opt_epoch ? opt_at + (wallclock() - opt_epoch) : opt_at + (t - t0);
        render(c0 + t, ct, t - last > 0.1 ? 1.0 / 60 : t - last, 0);
        glEndQuery(GL_TIME_ELAPSED);
        qn = (qn + 1) % 3;
        last = t;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        if (opt_scale <= 0 && frame > 30 && t - last_change > 1.0) {
            float s = render_scale;
            if (avg > 0.006) s *= 0.9f;
            else if (avg < 0.0035) s *= 1.05f;
            float d = default_scale(), lo = 0.6f * d, hi = fminf(1.0f, 1.25f * d);   /* stay near the default: there may be one of us per monitor */
            s = s < lo ? lo : s > hi ? hi : s;
            if (fabsf(s - render_scale) > 0.02f) { render_scale = s; last_change = t; }
        }
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
