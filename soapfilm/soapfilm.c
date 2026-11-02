/*
 * soapfilm — the iridescence of soap films, as in macro photographs.
 *
 * Three scenes, an act each, in an order shuffled from the seed (or fixed
 * with --scene), crossfading through black:
 *
 *   film     a flat soap film in a brass wire frame filling the screen, lit
 *            by a big softbox behind the camera. It's drawn up out of the
 *            bath thick and silvery, drains (thick at the bottom, the
 *            colours stacking up from the top, thinning to the common black
 *            film that spreads down from the top edge) while buoyant thin
 *            film rises in plumes off the frame's sides and bottom and
 *            curls into eddies. When it's old it bursts from the black film:
 *            a ragged hole with a rolled-up rim races across, throwing off
 *            mist, and a new film is drawn. Two films an act.
 *   bubbles  free bubbles drifting on the air in front of black velvet,
 *            each a thin film whose colours swirl over it and thin to a
 *            black cap with age, reflecting a window and a strip light.
 *            Bubbles that touch join into clusters, their films meeting at
 *            120 degrees across a shared wall (a cap of radius r1 r2 /
 *            |r1 - r2| through the circle of contact) with a Plateau border
 *            round it. Old ones burst: the hole opens from a point, its rim
 *            shedding a fine mist. New ones are blown in from the left or
 *            drift down from above.
 *   foam     a raft of foam seen from above with a tilted plane of focus:
 *            domed cells of film, each with the window in it, divided by
 *            glassy Plateau borders that widen at the junctions. It
 *            coarsens (von Neumann: cells with fewer than six sides shrink),
 *            films between cells rupture (the border retracts to its ends,
 *            the two become one), caps burst and their neighbours close in.
 *
 * The colour is physical. Thin-film reflectance (the Airy sum for a film
 * of soapy water in air, Fresnel per polarisation, water's dispersion) is
 * integrated over the visible spectrum against a daylight-ish 5800 K source
 * and the CIE 1931 observer, once, into a table of thickness x incidence:
 * black below ~30 nm, first-order silver-white, straw, orange-red, magenta
 * at ~200 nm (where green cancels), blue, then the second order's green,
 * yellow, magenta, blue, washing out to pastel pinks and greens and grey
 * past a micron (see --lutstrip). Every film in every scene is that table
 * looked up with its thickness and view angle, times the light it reflects.
 *
 * The flat film is a 2D simulation on the GPU: an incompressible flow in
 * the film's plane (semi-Lagrangian, Jacobi pressure, vorticity
 * confinement, a faint stirring of air) carrying the thickness (MacCormack).
 * Thin film is lighter, so it's pushed up against the local mean; the
 * thickness relaxes slowly to the drainage profile of its age (thick below,
 * sqrt of the depth beneath the black front above); the edges pull out thin
 * patches (marginal regeneration) that feed the rising plumes; below 35 nm
 * the film collapses to common black. It runs at a fixed 30 steps a second
 * from the moment the film is drawn, so it's the same film whenever and
 * wherever it's rendered. Bubbles and foam are CPU simulations, likewise in
 * fixed steps from the start of their act.
 *
 * Rendered at an internal resolution (about a megapixel, --scale to set),
 * into colour + depth, then a depth-of-field gather at a third of that
 * (Gustafsson's single-pass bokeh), a faint bloom, a filmic curve, vignette
 * and grain. A small caption, bottom left: the time, and how long the
 * screen's been locked (swaylock's start, from /proc), on the front output.
 *
 * Same plumbing as marbling: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen,
 * --bench N to time N frames offscreen (SOAP_PROF=1 for GPU time a stage).
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index,
 * as for flightsim. The outputs are windows onto one wall (the layout's
 * bounding box) seen from one eye, over the front output (the one under
 * the middle of the layout): one film in one frame round the whole wall,
 * one air full of bubbles drifting across the bezels, one raft of foam.
 * Everything is a function of the seed, the time since the epoch and the
 * layout, so the instances agree without talking.
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

#define PI 3.14159265358979f

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_scene = -1, opt_caption = 1;
static uint32_t opt_seed;
static const char *opt_shot, *opt_lutstrip;
static double opt_at = 30;             /* --shot / --bench: from this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the clock starts at */
static float opt_scale;                /* internal resolution; 0: chosen from the size */
static int offline;

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, front = 1;
static float wallr[4], outr[4];        /* the whole wall's and this output's logical rects */
static float eye_x = 0.5f, eye_y = 0.5f; /* the shared eye, over the front output (wall units) */

static void die(const char *what) { fprintf(stderr, "soapfilm: %s\n", what); exit(1); }

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

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}
static float frange(float a, float b) { return a + (b - a) * frand(); }
static int irange(int a, int b) { int v = a + (int)(frand() * (b - a + 1)); return v > b ? b : v; }

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x ? x : 1;
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth01(float x) { x = clampf(x, 0, 1); return x * x * (3 - 2 * x); }

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else 0 */
static time_t locked_at;
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

/* ---- the schedule: acts of one scene each, back to back ---- */
enum { S_FILM, S_BUBBLES, S_FOAM, NSCENE };
static const char *SCENE_NAME[NSCENE] = {"film", "bubbles", "foam"};
#define NFILM 2
#define LEAD 1.6                       /* film: the empty frame before it's drawn */
#define DRAW 2.4                       /* drawing it up out of the bath */
#define POPD 1.6                       /* the hole crossing it, the mist settling */
#define TAIL 1.4                       /* the empty frame after */
#define ACT_FADE 2.5
#define SIMHZ 30

typedef struct {
    int scene;
    uint32_t seed;
    double start, dur;
    double fstart[NFILM], flife[NFILM];
} Act;

static int scene_of(int idx) {
    if (opt_scene >= 0) return opt_scene;
    int deck = idx / NSCENE, pos = idx % NSCENE, order[NSCENE], prev_last = -1;
    for (int d = deck - 1; d <= deck; d++) {
        if (d < 0) continue;
        rng = mix32(opt_seed * 2654435761U ^ (uint32_t)d * 40503U ^ 0x50a9U);
        for (int i = 0; i < NSCENE; i++) order[i] = i;
        for (int i = NSCENE - 1; i > 0; i--) { int j = irange(0, i); int t = order[i]; order[i] = order[j]; order[j] = t; }
        if (d < deck) prev_last = order[NSCENE - 1];
    }
    if (order[0] == prev_last) { int t = order[0]; order[0] = order[1]; order[1] = t; }
    return order[pos];
}

static void act_gen(int idx, Act *a) {
    memset(a, 0, sizeof *a);
    a->scene = scene_of(idx);
    a->seed = mix32(opt_seed ^ (uint32_t)idx * 0x9E3779B9U ^ 0xf11b5U);
    rng = a->seed;
    switch (a->scene) {
    case S_FILM: {
        double t = 0;
        for (int f = 0; f < NFILM; f++) {
            a->fstart[f] = t;
            a->flife[f] = frange(72, 96);
            t += LEAD + a->flife[f] + POPD + TAIL;
        }
        a->dur = t;
        break;
    }
    case S_BUBBLES: a->dur = 230; break;
    case S_FOAM: a->dur = 200; break;
    }
}

static Act *acts;
static int nacts, capacts;
static const Act *act_get(int idx) {
    while (nacts <= idx) {
        if (nacts == capacts) acts = realloc(acts, (size_t)(capacts = capacts ? capacts * 2 : 16) * sizeof *acts);
        act_gen(nacts, &acts[nacts]);
        acts[nacts].start = nacts ? acts[nacts - 1].start + acts[nacts - 1].dur : 0;
        nacts++;
    }
    return &acts[idx];
}
static int act_at(double t, double *lt) {
    if (t < 0) t = 0;
    int i = 0;
    while (act_get(i)->start + act_get(i)->dur <= t) i++;
    *lt = t - act_get(i)->start;
    return i;
}

/* ---- thin-film interference, computed once ----
 * A film of soapy water (n = 1.324 + 3046/l^2, l in nm) in air, d thick,
 * lit at incidence cos c. Both faces reflect, the first with the phase flip,
 * so with r = r12 = -r23 (Fresnel, per polarisation) and the round-trip
 * phase delta = 4 pi n d cos(t2) / l, the Airy sum gives
 *     R = 2 r^2 (1 - cos delta) / (1 - 2 r^2 cos delta + r^4),
 * averaged over s and p. That's integrated against a 5800 K light and the
 * CIE 1931 observer (Wyman, Sloan & Shirley's multi-lobe fit) every 5 nm
 * from 380 to 780 nm, taken to linear sRGB and white-balanced so that a
 * mirror would come out white; the odd colour outside sRGB is pulled in
 * towards the grey of the same luminance. The table: 1024 thicknesses
 * 0..HMAX microns across, 32 incidence cosines 0..1 up. */
#define LUT_N 1024
#define LUT_C 32
#define HMAX 2.4f

static float cmf_g(float l, float mu, float s1, float s2) {
    float t = (l - mu) / (l < mu ? s1 : s2);
    return expf(-0.5f * t * t);
}

static float *lut_build(void) {
    enum { NL = 81 };
    double S[NL], xb[NL], yb[NL], zb[NL], nn[NL], lam[NL];
    double wX = 0, wY = 0, wZ = 0;
    for (int k = 0; k < NL; k++) {
        double l = 380 + 5 * k;
        lam[k] = l;
        double lm = l * 1e-9;
        S[k] = 1.0 / (pow(lm, 5) * (exp(1.4388e-2 / (lm * 5800.0)) - 1.0));
        xb[k] = 1.056 * cmf_g((float)l, 599.8f, 37.9f, 31.0f) + 0.362 * cmf_g((float)l, 442.0f, 16.0f, 26.7f) -
                0.065 * cmf_g((float)l, 501.1f, 20.4f, 26.2f);
        yb[k] = 0.821 * cmf_g((float)l, 568.8f, 46.9f, 40.5f) + 0.286 * cmf_g((float)l, 530.9f, 16.3f, 31.1f);
        zb[k] = 1.217 * cmf_g((float)l, 437.0f, 11.8f, 36.0f) + 0.681 * cmf_g((float)l, 459.0f, 26.0f, 13.8f);
        nn[k] = 1.324 + 3046.0 / (l * l);
        wX += S[k] * xb[k]; wY += S[k] * yb[k]; wZ += S[k] * zb[k];
    }
    static const double M[9] = {3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570};
    double wr = M[0] * wX + M[1] * wY + M[2] * wZ, wg = M[3] * wX + M[4] * wY + M[5] * wZ, wb = M[6] * wX + M[7] * wY + M[8] * wZ;
    float *out = malloc(sizeof(float) * 4 * LUT_N * LUT_C);
    for (int ci = 0; ci < LUT_C; ci++) {
        double c1 = fmax((double)ci / (LUT_C - 1), 0.002), s12 = 1 - c1 * c1;
        for (int hi = 0; hi < LUT_N; hi++) {
            double d = (double)hi / (LUT_N - 1) * HMAX * 1000.0;
            double X = 0, Y = 0, Z = 0;
            for (int k = 0; k < NL; k++) {
                double n = nn[k], c2 = sqrt(1 - s12 / (n * n));
                double rs = (c1 - n * c2) / (c1 + n * c2), rp = (n * c1 - c2) / (n * c1 + c2);
                double cd = cos(4 * M_PI * n * d * c2 / lam[k]);
                double R = 0;
                for (int pol = 0; pol < 2; pol++) {
                    double r2 = pol ? rp * rp : rs * rs;
                    R += 0.5 * 2 * r2 * (1 - cd) / (1 - 2 * r2 * cd + r2 * r2);
                }
                X += R * S[k] * xb[k]; Y += R * S[k] * yb[k]; Z += R * S[k] * zb[k];
            }
            double rgb[3] = {(M[0] * X + M[1] * Y + M[2] * Z) / wr, (M[3] * X + M[4] * Y + M[5] * Z) / wg,
                             (M[6] * X + M[7] * Y + M[8] * Z) / wb};
            double lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            double mn = fmin(rgb[0], fmin(rgb[1], rgb[2]));
            if (mn < 0) {
                double t = lum > 0 ? lum / (lum - mn) : 0;
                for (int i = 0; i < 3; i++) rgb[i] = lum + (rgb[i] - lum) * t;
            }
            float *o = out + 4 * (ci * LUT_N + hi);
            for (int i = 0; i < 3; i++) o[i] = (float)fmax(rgb[i], 0);
            o[3] = 1;
        }
    }
    return out;
}

/* the normal-incidence strip, thickness 0..HMAX left to right, for checking
 * the order of the colours by eye (--lutstrip FILE.ppm) */
static void lut_strip(const char *path) {
    float *lut = lut_build();
    int w = LUT_N, h = 120;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int ci = y < 90 ? LUT_C - 1 : LUT_C / 2;
            float *o = lut + 4 * (ci * LUT_N + x);
            unsigned char px[3];
            for (int i = 0; i < 3; i++) {
                float v = o[i] * 13.0f;                /* lit by a softbox of radiance 13 */
                v = v * (2.51f * v + 0.03f) / (v * (2.43f * v + 0.59f) + 0.14f);
                v = v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1 / 2.4f) - 0.055f;
                /* a tick every 100 nm along the bottom */
                if (y > 110 && (x * 2400 / (LUT_N - 1)) % 100 < 3) v = 1;
                px[i] = (unsigned char)(clampf(v, 0, 1) * 255 + 0.5f);
            }
            fwrite(px, 1, 3, f);
        }
    fclose(f);
    free(lut);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

static GLuint compile(GLenum type, const char *a, const char *b, const char *c) {
    GLuint s = glCreateShader(type);
    const char *src[3] = {a, b ? b : "", c ? c : ""};
    glShaderSource(s, 3, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "soapfilm: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint link_prog(GLuint vs, GLuint fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, vs);
    glAttachShader(pr, fs);
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "soapfilm: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

/* a full-screen pass: fragment shader from (head, body) */
static GLuint fs_prog(const char *head, const char *body) {
    static GLuint vs;
    if (!vs) vs = compile(GL_VERTEX_SHADER, VS, NULL, NULL);
    return link_prog(vs, compile(GL_FRAGMENT_SHADER, "#version 330 core\n", head, body));
}

static GLint U(GLuint p, const char *name) { return glGetUniformLocation(p, name); }

typedef struct { GLuint tex, fbo; int w, h; } RT;

static void rt_make(RT *r, int w, int h, GLenum ifmt, GLenum filter, int mip) {
    if (r->tex) { glDeleteTextures(1, &r->tex); glDeleteFramebuffers(1, &r->fbo); }
    r->w = w; r->h = h;
    glGenTextures(1, &r->tex);
    glBindTexture(GL_TEXTURE_2D, r->tex);
    GLenum fmt = ifmt == GL_R16F || ifmt == GL_R32F ? GL_RED : ifmt == GL_RG16F || ifmt == GL_RG32F ? GL_RG : GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (mip) glGenerateMipmap(GL_TEXTURE_2D);
    glGenFramebuffers(1, &r->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r->tex, 0);
}

static void rt_bind(const RT *r) {
    glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    glViewport(0, 0, r->w, r->h);
}

static void tex_unit(int unit, GLenum target, GLuint tex) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(target, tex);
}

static GLuint vao, tex_lut, tex_noise, tex_noise3;
static void draw_fs(void) { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, 3); }

/* tiling value noise, `per` cells across `n` texels */
static float lattice2(const float *g, int per, int n, int x, int y) {
    float fx = (float)x * per / n, fy = (float)y * per / n;
    int ix = (int)fx, iy = (int)fy;
    float u = fx - ix, v = fy - iy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
#define G(i, j) g[(((iy + (j)) % per) * per + (ix + (i)) % per)]
    return (G(0, 0) * (1 - u) + G(1, 0) * u) * (1 - v) + (G(0, 1) * (1 - u) + G(1, 1) * u) * v;
#undef G
}

/* 2D: 256x256 tiling, smooth (r, g), fine (b), white (a) */
static void noise2_init(void) {
    unsigned char *px = malloc(256 * 256 * 4);
    float *acc = calloc(256 * 256 * 3, sizeof *acc), *g = malloc(256 * 256 * sizeof *g);
    static const int per[3][3] = {{4, 8, 16}, {4, 8, 16}, {16, 32, 64}};
    static const float wt[3] = {0.55f, 0.3f, 0.15f};
    rng = 0x5eed1234U;
    for (int c = 0; c < 3; c++)
        for (int o = 0; o < 3; o++) {
            int P = per[c][o];
            for (int i = 0; i < P * P; i++) g[i] = frand();
            for (int y = 0; y < 256; y++)
                for (int x = 0; x < 256; x++) acc[(y * 256 + x) * 3 + c] += wt[o] * lattice2(g, P, 256, x, y);
        }
    for (int i = 0; i < 256 * 256; i++) {
        for (int c = 0; c < 3; c++) {
            float v = 0.5f + (acc[i * 3 + c] - 0.5f) * 2.2f;
            px[i * 4 + c] = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255);
        }
        px[i * 4 + 3] = (unsigned char)(frand() * 255);
    }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    free(px); free(acc); free(g);
}

/* 3D: 64^3 tiling smooth value noise, four independent channels, 4 cells
 * across plus a half-weight octave of 8 */
#define N3 64
static void noise3_init(void) {
    float *acc = calloc((size_t)N3 * N3 * N3 * 4, sizeof *acc);
    rng = 0x3d5eedU;
    for (int c = 0; c < 4; c++)
        for (int o = 0; o < 2; o++) {
            int P = o ? 8 : 4;
            float g[512];
            for (int i = 0; i < P * P * P; i++) g[i] = frand();
            float wt = o ? 0.33f : 0.67f;
            for (int z = 0; z < N3; z++)
                for (int y = 0; y < N3; y++)
                    for (int x = 0; x < N3; x++) {
                        float fx = (float)x * P / N3, fy = (float)y * P / N3, fz = (float)z * P / N3;
                        int ix = (int)fx, iy = (int)fy, iz = (int)fz;
                        float u = fx - ix, v = fy - iy, w = fz - iz;
                        u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v); w = w * w * (3 - 2 * w);
                        float s = 0;
                        for (int k = 0; k < 8; k++) {
                            int a = k & 1, b = (k >> 1) & 1, cc = k >> 2;
                            float gv = g[(((iz + cc) % P) * P + (iy + b) % P) * P + (ix + a) % P];
                            s += gv * (a ? u : 1 - u) * (b ? v : 1 - v) * (cc ? w : 1 - w);
                        }
                        acc[(((size_t)z * N3 + y) * N3 + x) * 4 + c] += wt * s;
                    }
        }
    unsigned char *px = malloc((size_t)N3 * N3 * N3 * 4);
    for (size_t i = 0; i < (size_t)N3 * N3 * N3 * 4; i++) {
        float v = 0.5f + (acc[i] - 0.5f) * 2.0f;
        px[i] = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255);
    }
    glGenTextures(1, &tex_noise3);
    glBindTexture(GL_TEXTURE_3D, tex_noise3);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, N3, N3, N3, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    free(acc); free(px);
}

/* ================= shaders ================= */

/* shared by the scene passes: the film colour table, noise, the lights */
static const char *FS_COMMON = GLSL(
    uniform sampler2D uLut;
    uniform sampler2D uNoise;
    uniform sampler3D uN3;
    uniform vec4 uMap;          /* wall position = uMap.xy + gl_FragCoord.xy * uMap.zw */
    uniform vec3 uEye;          /* in wall units; the screen plane is z = 0 */
    uniform float uTime, uFade;
    /* the lights: a softbox or window (dir, right, up, (half w, half h, edge, radiance)),
     * a strip light, and how much window bar */
    uniform vec3 uBoxD, uBoxU, uBoxV, uStrD, uStrU, uStrV;
    uniform vec4 uBox, uStr;
    uniform float uBars, uAmb, uFill;
    layout(location = 0) out vec4 oCol;
    layout(location = 1) out vec2 oDep;

    /* reflectance of a film h microns thick at incidence cos c */
    vec3 tfilm(float h, float c) {
        return texture(uLut, vec2((clamp(h * (1.0 / 2.4), 0.0, 1.0) * 1023.0 + 0.5) / 1024.0,
                                  (clamp(c, 0.0, 1.0) * 31.0 + 0.5) / 32.0)).rgb;
    }

    vec3 rectLight(vec3 d, vec3 D, vec3 R, vec3 Up, vec4 b, float bars) {
        float k = dot(d, D);
        if (k <= 0.05) return vec3(0.0);
        vec2 uv = vec2(dot(d, R), dot(d, Up)) / k;
        vec2 q = abs(uv) - b.xy;
        float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
        float m = 1.0 - smoothstep(-b.z, b.z, sd);
        vec2 n = uv / b.xy;
        m *= 1.0 - 0.22 * dot(n, n);
        if (bars > 0.0) {
            /* a window: a mullion and two transoms */
            float e = 0.018 + b.z * 0.4;
            float bx = smoothstep(0.03, 0.03 + e, abs(n.x));
            float by = smoothstep(0.03, 0.03 + e, abs(abs(n.y) - 0.34));
            m *= mix(1.0, 0.06 + 0.94 * min(bx, by), bars);
        }
        return m * b.w * vec3(1.0, 0.975, 0.94);
    }

    vec3 env(vec3 d) {
        vec3 c = mix(vec3(0.022, 0.017, 0.012), vec3(0.008, 0.010, 0.014), smoothstep(-0.7, 0.8, d.y)) * uAmb;
        c += uFill * vec3(1.0, 0.97, 0.93) * smoothstep(0.3, -0.9, d.z) * (0.55 + 0.45 * d.y);
        c += rectLight(d, uBoxD, uBoxU, uBoxV, uBox, uBars);
        c += rectLight(d, uStrD, uStrU, uStrV, uStr, 0.0) * vec3(0.85, 0.93, 1.06);
        return c;
    }

    /* the black velvet behind everything, a little light spilling on it */
    vec3 backdrop(vec2 P) {
        vec2 g = P - vec2(uEye.x - 0.35, uEye.y + 0.25);
        return vec3(0.0062, 0.0056, 0.0052) + vec3(0.012, 0.010, 0.0085) * exp(-dot(g, g) * 1.6);
    }

    vec2 wallPos() { return uMap.xy + gl_FragCoord.xy * uMap.zw; }
);

/* ---- the film simulation: a 2D incompressible flow in the film's plane
 * (it's nearly so: Couder's soap-film turbulence), carrying the thickness.
 * Thin film is lighter than thick, so it rises (buoyancy against the local
 * mean); vorticity confinement keeps the eddies; a faint, slowly changing
 * stirring stands for the air. The thickness relaxes slowly to the drainage
 * profile (thick at the bottom, sqrt of the depth below the black front at
 * the top, thinning with age), the vertical edges and the bottom pull thin
 * patches out of the meniscus (marginal regeneration, which feeds the
 * rising plumes), and below ~35 nm the film collapses to common black. ---- */
static const char *SIM_HEAD = GLSL(
    uniform sampler2D uVel, uH, uHhat, uCurl, uDiv, uP, uNoise;
    uniform sampler3D uN3;
    uniform vec2 uRes;          /* this pass's grid */
    uniform float uDt, uAd, uDx, uTau, uSeed;
    out vec4 o;
    vec2 UV() { return gl_FragCoord.xy / uRes; }
);

static const char *SIM_CURL = GLSL(
    void main() {
        vec2 uv = UV(), t = 1.0 / uRes;
        float L = texture(uVel, uv - vec2(t.x, 0.0)).y, R = texture(uVel, uv + vec2(t.x, 0.0)).y;
        float B = texture(uVel, uv - vec2(0.0, t.y)).x, T = texture(uVel, uv + vec2(0.0, t.y)).x;
        o = vec4(((R - L) - (T - B)) / (2.0 * uDx), 0.0, 0.0, 1.0);
    });

static const char *SIM_VEL = GLSL(
    uniform float uBuoy, uVort, uDrag, uStir, uLod;
    void main() {
        vec2 uv = UV(), t = 1.0 / uRes;
        vec2 u = texture(uVel, uv).xy;
        vec2 v = texture(uVel, uv - uDt * vec2(u.x / uAd, u.y)).xy;
        float h = texture(uH, uv).r, hm = textureLod(uH, uv, uLod).r;
        float fb = uBuoy * clamp((hm - h) / (hm + 0.12), -1.5, 1.5);
        float c = texture(uCurl, uv).r;
        float cl = abs(texture(uCurl, uv - vec2(t.x, 0.0)).r), cr = abs(texture(uCurl, uv + vec2(t.x, 0.0)).r);
        float cb = abs(texture(uCurl, uv - vec2(0.0, t.y)).r), ct = abs(texture(uCurl, uv + vec2(0.0, t.y)).r);
        vec2 g = vec2(cr - cl, ct - cb);
        g /= length(g) + 1e-5;
        vec2 fv = uVort * c * vec2(g.y, -g.x) * uDx;
        vec3 q = vec3(uv.x * uAd * 0.9, uv.y * 0.9, uTau * 0.015) + uSeed;
        float e = 0.03;
        float n0 = texture(uN3, q).r, nx = texture(uN3, q + vec3(e, 0.0, 0.0)).r, ny = texture(uN3, q + vec3(0.0, e, 0.0)).r;
        vec2 fs = uStir * vec2(ny - n0, -(nx - n0)) / e;
        v += uDt * (vec2(0.0, fb) + fv + fs);
        v *= exp(-uDrag * uDt);
        vec2 f = gl_FragCoord.xy;
        if (f.x < 1.0 || f.y < 1.0 || f.x > uRes.x - 1.0 || f.y > uRes.y - 1.0) v = vec2(0.0);
        o = vec4(v, 0.0, 1.0);
    });

static const char *SIM_DIV = GLSL(
    void main() {
        vec2 uv = UV(), t = 1.0 / uRes;
        float L = texture(uVel, uv - vec2(t.x, 0.0)).x, R = texture(uVel, uv + vec2(t.x, 0.0)).x;
        float B = texture(uVel, uv - vec2(0.0, t.y)).y, T = texture(uVel, uv + vec2(0.0, t.y)).y;
        o = vec4(((R - L) + (T - B)) / (2.0 * uDx), 0.0, 0.0, 1.0);
    });

static const char *SIM_JACOBI = GLSL(
    void main() {
        ivec2 i = ivec2(gl_FragCoord.xy), m = ivec2(uRes) - 1;
        float L = texelFetch(uP, clamp(i - ivec2(1, 0), ivec2(0), m), 0).r, R = texelFetch(uP, clamp(i + ivec2(1, 0), ivec2(0), m), 0).r;
        float B = texelFetch(uP, clamp(i - ivec2(0, 1), ivec2(0), m), 0).r, T = texelFetch(uP, clamp(i + ivec2(0, 1), ivec2(0), m), 0).r;
        float d = texelFetch(uDiv, i, 0).r;
        o = vec4((L + R + B + T - uDx * uDx * d) * 0.25, 0.0, 0.0, 1.0);
    });

static const char *SIM_PROJECT = GLSL(
    void main() {
        ivec2 i = ivec2(gl_FragCoord.xy), m = ivec2(uRes) - 1;
        float L = texelFetch(uP, clamp(i - ivec2(1, 0), ivec2(0), m), 0).r, R = texelFetch(uP, clamp(i + ivec2(1, 0), ivec2(0), m), 0).r;
        float B = texelFetch(uP, clamp(i - ivec2(0, 1), ivec2(0), m), 0).r, T = texelFetch(uP, clamp(i + ivec2(0, 1), ivec2(0), m), 0).r;
        vec2 v = texelFetch(uVel, i, 0).xy - vec2(R - L, T - B) / (2.0 * uDx);
        vec2 f = gl_FragCoord.xy;
        if (f.x < 1.0 || f.y < 1.0 || f.x > uRes.x - 1.0 || f.y > uRes.y - 1.0) v = vec2(0.0);
        o = vec4(v, 0.0, 1.0);
    });

/* thickness, step 1: carried back along the flow */
static const char *SIM_HADV = GLSL(
    void main() {
        vec2 uv = UV();
        vec2 u = texture(uVel, uv).xy;
        o = vec4(texture(uH, uv - uDt * vec2(u.x / uAd, u.y)).r, 0.0, 0.0, 1.0);
    });

/* step 2: MacCormack's correction (clamped to the source cell), then the
 * drainage, the edges and the black film */
static const char *SIM_HNEW = GLSL(
    uniform float uHb, uS, uYb, uRelax;
    void main() {
        vec2 uv = UV();
        vec2 u = texture(uVel, uv).xy, du = uDt * vec2(u.x / uAd, u.y);
        float hh = texture(uHhat, uv).r, hf = texture(uHhat, uv + du).r, h0 = texture(uH, uv).r;
        float hn = hh + 0.5 * (h0 - hf);
        vec2 st = (uv - du) * uRes - 0.5;
        ivec2 i = ivec2(floor(st)), m = ivec2(uRes) - 1;
        float a = texelFetch(uH, clamp(i, ivec2(0), m), 0).r, b = texelFetch(uH, clamp(i + ivec2(1, 0), ivec2(0), m), 0).r;
        float c = texelFetch(uH, clamp(i + ivec2(0, 1), ivec2(0), m), 0).r, d = texelFetch(uH, clamp(i + ivec2(1, 1), ivec2(0), m), 0).r;
        hn = clamp(hn, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));

        float yp = 1.0 - uv.y;
        float prof = sqrt(clamp((yp - uYb) / max(1.0 - uYb, 1e-3), 0.0, 1.0));
        float hT = uHb * mix(prof, 1.0, uS);
        hn += (hT - hn) * (1.0 - exp(-uDt / uRelax));

        /* marginal regeneration: thin patches drawn out along the sides and the bottom */
        float ex = min(uv.x, 1.0 - uv.x) * uAd, ey = uv.y;
        float side = uv.x < 0.5 ? 0.0 : 0.5;
        float ps = smoothstep(0.42, 0.72, texture(uNoise, vec2(uv.y * 2.2 + uSeed, uTau * 0.035 + side)).g);
        float pb = smoothstep(0.42, 0.72, texture(uNoise, vec2(uv.x * uAd * 1.6 + uSeed, uTau * 0.035 + 0.25)).g);
        float mreg = max(exp(-ex / 0.012) * ps, exp(-ey / 0.012) * pb);
        hn += (0.4 * hT - hn) * (1.0 - exp(-uDt * 2.5 * mreg));

        /* common black film */
        if (hn < 0.035) hn = mix(hn, 0.008, 1.0 - exp(-uDt * 3.0));
        o = vec4(max(hn, 0.006), 0.0, 0.0, 1.0);
    });

/* a newly drawn film: thick, and stirred up by the drawing */
static const char *SIM_INIT = GLSL(
    uniform float uMode, uHb;
    void main() {
        vec2 uv = UV();
        vec3 q = vec3(uv.x * uAd * 1.3, uv.y * 1.3, 0.0) + uSeed;
        if (uMode < 0.5) {
            float n = texture(uN3, q * 1.7).g;
            o = vec4(uHb * (0.85 + 0.3 * n), 0.0, 0.0, 1.0);
        } else {
            float e = 0.02;
            float n0 = texture(uN3, q).b, nx = texture(uN3, q + vec3(e, 0.0, 0.0)).b, ny = texture(uN3, q + vec3(0.0, e, 0.0)).b;
            float m0 = texture(uN3, q * 2.6).a, mx = texture(uN3, q * 2.6 + vec3(e, 0.0, 0.0)).a, my = texture(uN3, q * 2.6 + vec3(0.0, e, 0.0)).a;
            vec2 v = 0.05 * vec2(ny - n0, -(nx - n0)) / e + 0.02 * vec2(my - m0, -(mx - m0)) / e;
            vec2 f = gl_FragCoord.xy;
            if (f.x < 1.0 || f.y < 1.0 || f.x > uRes.x - 1.0 || f.y > uRes.y - 1.0) v = vec2(0.0);
            o = vec4(v, 0.0, 1.0);
        }
    });

/* ---- the flat film in its wire frame ---- */
static const char *FS_FILM = GLSL(
    uniform sampler2D uH;
    uniform vec2 uHRes;
    uniform vec4 uFrame;        /* centre, half extents (wall units) */
    uniform float uFrameR, uWireR;
    uniform float uOn, uWipe;   /* film there at all; drawn up to this wall y */
    uniform vec4 uPop;          /* centre, hole radius, seed (radius < 0: none) */
    uniform float uDepth0, uTilt;

    float hTex(vec2 uv) {
        vec2 st = uv * uHRes - 0.5, i = floor(st), f = st - i;
        vec2 w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0, w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0;
        vec2 w2 = (1.0 + 3.0 * f + 3.0 * f * f - 3.0 * f * f * f) / 6.0, w3 = f * f * f / 6.0;
        vec2 s0 = w0 + w1, s1 = w2 + w3;
        vec2 p0 = (i - 1.0 + w1 / s0 + 0.5) / uHRes, p1 = (i + 1.0 + w3 / s1 + 0.5) / uHRes;
        return s0.y * (s0.x * texture(uH, p0).r + s1.x * texture(uH, vec2(p1.x, p0.y)).r) +
               s1.y * (s0.x * texture(uH, vec2(p0.x, p1.y)).r + s1.x * texture(uH, p1).r);
    }
    float sdRoundBox(vec2 p, vec2 b, float r) {
        vec2 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    void main() {
        vec2 P = wallPos();
        float px = uMap.z;
        vec3 rd = normalize(vec3(P, 0.0) - uEye);
        vec3 col = backdrop(P);
        vec2 q = P - uFrame.xy;
        float d = sdRoundBox(q, uFrame.zw, uFrameR);
        float depth = uDepth0 + uTilt * (P.y - uEye.y);

        /* the film */
        float inner = -d - uWireR;
        if (uOn > 0.5 && inner > -px) {
            vec2 uv = (q + uFrame.zw) / (2.0 * uFrame.zw);
            float h = max(hTex(uv), 0.0);
            /* the meniscus up against the wire, and at the edge being drawn up */
            h += 3.0 * exp(-max(inner, 0.0) / 0.0022);
            float dw = uWipe - P.y;
            h += 3.5 * exp(-max(dw, 0.0) / 0.0035);
            float cov = clamp(inner / px + 0.5, 0.0, 1.0) * clamp(dw / px + 0.5, 0.0, 1.0);
            /* the hole of a pop, its rim rolling up the film as it goes */
            float rim = 0.0;
            if (uPop.z > 0.0) {
                vec2 dp = P - uPop.xy;
                float rr = length(dp), ang = atan(dp.y, dp.x) / 6.2831853;
                float rag = texture(uNoise, vec2(ang, uPop.w)).r - 0.5, fine = texture(uNoise, vec2(ang * 3.0, uPop.w + 0.37)).b - 0.5;
                float re = uPop.z * (1.0 + 0.10 * rag) + 0.012 * fine * min(uPop.z * 6.0, 1.0);
                float into = re - rr;
                cov *= clamp(-into / px + 0.5, 0.0, 1.0);
                float rimw = 0.0018 + 0.0022 * min(uPop.z, 1.0);
                float k = (into + rimw * 0.6) / rimw;
                rim = exp(-k * k);
            }
            /* a flat mirror, faintly rippled */
            vec2 wob = (texture(uNoise, P * 0.35 + vec2(uTime * 0.004, 0.0)).rg - 0.5) * 0.02;
            vec3 n = normalize(vec3(wob, -1.0));
            float c = abs(dot(rd, n));
            vec3 refl = reflect(rd, n);
            vec3 L = env(refl);
            vec3 R = tfilm(h, c);
            vec3 film = col * (1.0 - R) + R * L;
            col = mix(col, film, cov);
            /* the rolled-up rim: a bead of liquid, bright where it faces the light */
            col += rim * (L * 0.12 + 0.05);
        }

        /* the wire: a brass rod, its section a cylinder */
        float s = d / uWireR;
        if (abs(s) < 1.0 + px / uWireR) {
            float e = 0.002;
            vec2 g = normalize(vec2(sdRoundBox(q + vec2(e, 0.0), uFrame.zw, uFrameR) - sdRoundBox(q - vec2(e, 0.0), uFrame.zw, uFrameR),
                                    sdRoundBox(q + vec2(0.0, e), uFrame.zw, uFrameR) - sdRoundBox(q - vec2(0.0, e), uFrame.zw, uFrameR)) + 1e-6);
            float sc = clamp(s, -1.0, 1.0);
            vec3 n = normalize(vec3(g * sc, -sqrt(1.0 - sc * sc)));
            vec3 F = vec3(0.62, 0.46, 0.26);
            float fr = pow(1.0 - abs(dot(rd, n)), 5.0);
            vec3 wire = (F + (1.0 - F) * fr) * env(reflect(rd, n)) * 0.75;
            float a = clamp((uWireR - abs(d)) / px + 0.5, 0.0, 1.0);
            col = mix(col, wire, a);
        }
        oCol = vec4(col * uFade, 1.0);
        oDep = vec2(depth, 1.0);
    });

/* ---- bubbles: one instanced quad each, ray-traced, added up ----
 * per bubble 8 texels: (centre, r) (pop dir, progress) (film h, seed,
 * black cap, spin) (draw wall k?) and up to four bubbles joined to it
 * (centre, r). A bubble's outer film is its sphere outside the ones it's
 * joined to; the wall between two is a cap of radius r1 r2 / |r1 - r2|
 * through their circle of contact, bulging into the larger. */
static const char *VS_BUB = GLSL(
    uniform sampler2D uB;
    uniform vec4 uOut;          /* this output's rect in wall units */
    uniform vec3 uEye;
    flat out int vI;
    void main() {
        int i = gl_InstanceID;
        vec4 a = texelFetch(uB, ivec2(0, i), 0);
        vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float((gl_VertexID >> 1) & 1) * 2.0 - 1.0);
        float D = -uEye.z, z = a.z + D;
        if (a.w <= 0.0 || z < a.w * 1.05) { gl_Position = vec4(2.0, 2.0, 0.0, 1.0); vI = i; return; }
        vec2 pc = uEye.xy + (a.xy - uEye.xy) * (D / z);
        float rp = a.w * D / sqrt(z * z - a.w * a.w);
        rp *= 1.08 + 0.6 * length(a.xy - uEye.xy) / z;
        vec2 P = pc + corner * rp;
        gl_Position = vec4((P - uOut.xy) / uOut.zw * 2.0 - 1.0, 0.0, 1.0);
        vI = i;
    });

static const char *FS_BUB = GLSL(
    uniform sampler2D uB;
    uniform float uFocus, uAperture;
    flat in int vI;
    vec4 bt(int k) { return texelFetch(uB, ivec2(k, vI), 0); }

    float filmH(vec3 q, vec4 F) {
        float cs = cos(F.w), sn = sin(F.w);
        vec3 s = vec3(cs * q.x + sn * q.z, q.y, -sn * q.x + cs * q.z);
        vec3 w = texture(uN3, s * 0.42 + F.y + vec3(0.0, uTime * 0.008, 0.0)).rgb - 0.5;
        vec3 s2 = s + 1.6 * w;
        float n = texture(uN3, s2 * 0.42 + F.y * 1.37).g * 0.8 + texture(uN3, s2 * 1.0 - F.y).b * 0.2;
        float h = F.x * (1.0 - 0.45 * q.y) * (0.45 + 1.1 * n);
        float top = smoothstep(1.0 - F.z, 1.0 - F.z + 0.05, q.y + 0.15 * (n - 0.5));
        return mix(h, 0.011, top * step(0.001, F.z));
    }

    /* the hole of a pop: the film gone within angle a of the pop point */
    float holeRim(vec3 q, vec4 Pp, float seed, out bool gone) {
        gone = false;
        if (Pp.w < 0.0) return 0.0;
        vec3 ax = Pp.xyz;
        vec3 t1 = normalize(cross(ax, abs(ax.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0))), t2 = cross(ax, t1);
        float phi = atan(dot(q, t2), dot(q, t1)) / 6.2831853;
        float th = acos(clamp(dot(q, ax), -1.0, 1.0));
        float rag = texture(uNoise, vec2(phi, seed)).r - 0.5, fine = texture(uNoise, vec2(phi * 2.0, seed + 0.4)).b - 0.5;
        float a = Pp.w * 3.5 * (1.0 + 0.16 * rag) + 0.08 * fine * Pp.w;
        if (th < a) { gone = true; return 0.0; }
        return Pp.w > 0.0 ? exp(-(th - a) / 0.045) : 0.0;
    }

    bool inJoined(vec3 p) {
        for (int k = 0; k < 4; k++) {
            vec4 L = bt(4 + k);
            if (L.w > 0.0 && distance(p, L.xyz) < L.w) return true;
        }
        return false;
    }
    /* how close p (on this bubble's outer film) is to a Plateau border */
    float border(vec3 p, float r) {
        float b = 0.0;
        for (int k = 0; k < 4; k++) {
            vec4 L = bt(4 + k);
            if (L.w > 0.0) b = max(b, exp(-max(distance(p, L.xyz) - L.w, 0.0) / (0.025 * min(r, L.w))));
        }
        return b;
    }

    void main() {
        vec4 A = bt(0), Pp = bt(1), F = bt(2), Wf = bt(3);
        vec3 c = A.xyz;
        float r = A.w;
        vec2 P = wallPos();
        vec3 ro = uEye, rd = normalize(vec3(P, 0.0) - uEye);
        vec3 oc = ro - c;
        float b = dot(oc, rd), dperp = sqrt(max(dot(oc, oc) - b * b, 0.0));
        float pxr = uMap.z * (-b) / (-uEye.z);
        float cov = clamp((r - dperp) / pxr + 0.5, 0.0, 1.0);
        if (cov <= 0.0) discard;
        float sq = sqrt(max(r * r - dperp * dperp, 0.0));
        float t0 = -b - sq, t1 = -b + sq;
        vec3 acc = vec3(0.0);
        float seed = fract(F.y * 7.31);

        /* the near face */
        vec3 p0 = ro + rd * t0, n0 = (p0 - c) / r;
        bool gone;
        float rim0 = holeRim(n0, Pp, seed, gone);
        float T = 1.0;
        if (!gone && !inJoined(p0)) {
            float bo = border(p0, r);
            float h = filmH(n0, F) + 2.5 * bo + 1.5 * rim0;
            vec3 R = tfilm(h, abs(dot(rd, n0)));
            vec3 Lr = env(reflect(rd, n0));
            acc += R * Lr * (1.0 + 2.0 * bo * bo) + rim0 * Lr * 0.05;
            T = 1.0 - dot(R, vec3(0.333));
        }
        /* the far face, from inside */
        vec3 p1 = ro + rd * t1, n1 = (p1 - c) / r;
        float rim1 = holeRim(n1, Pp, seed, gone);
        if (!gone && !inJoined(p1)) {
            float bo = border(p1, r);
            float h = filmH(n1, F) + 2.5 * bo + 1.5 * rim1;
            vec3 R = tfilm(h, abs(dot(rd, n1)));
            acc += T * (R * env(reflect(rd, n1)) * (1.0 + 2.0 * bo * bo) + rim1 * env(reflect(rd, n1)) * 0.05);
        }
        /* the walls shared with joined bubbles (each drawn by one of the pair) */
        for (int k = 0; k < 4; k++) {
            vec4 L = bt(4 + k);
            if (L.w <= 0.0 || Wf[k] < 0.5) continue;
            vec3 cj = L.xyz;
            float rj = L.w;
            vec3 ax = cj - c;
            float d = length(ax);
            ax /= d;
            float x0 = (d * d + r * r - rj * rj) / (2.0 * d), rho2 = r * r - x0 * x0;
            if (rho2 <= 0.0) continue;
            float dr = r - rj;
            vec3 hp;
            vec3 hn;
            bool hit = false;
            if (abs(dr) < 0.004 * r) {
                float den = dot(rd, ax);
                if (abs(den) > 1e-4) {
                    float t = (x0 - dot(ro - c, ax)) / den;
                    hp = ro + rd * t;
                    vec3 rel = hp - c - ax * x0;
                    if (t > 0.0 && dot(rel, rel) < rho2) { hit = true; hn = ax; }
                }
            } else {
                float Rw = r * rj / abs(dr), off = sqrt(max(Rw * Rw - rho2, 0.0));
                float sgn = dr < 0.0 ? 1.0 : -1.0;   /* +1: bulging into j */
                vec3 cw = c + ax * (x0 - sgn * off);
                vec3 ow = ro - cw;
                float bw = dot(ow, rd), cwq = dot(ow, ow) - Rw * Rw, dw = bw * bw - cwq;
                if (dw > 0.0) {
                    float sw = sqrt(dw);
                    for (int e = 0; e < 2; e++) {
                        float t = e == 0 ? -bw - sw : -bw + sw;
                        vec3 pp = ro + rd * t;
                        float ax0 = dot(pp - c, ax) - x0;
                        if (t > 0.0 && ax0 * sgn > 0.0) { hp = pp; hn = (pp - cw) / Rw; hit = true; break; }
                    }
                }
            }
            if (hit) {
                vec3 rel = hp - c - ax * dot(hp - c, ax);
                float edge = exp(-(sqrt(rho2) - length(rel)) / (0.03 * min(r, rj)));
                float h = F.x * 0.75 * (0.5 + texture(uN3, hp * 3.0 + F.y).g) + 2.5 * edge;
                vec3 R = tfilm(h, abs(dot(rd, hn)));
                acc += 0.93 * R * env(reflect(rd, hn)) * (1.0 + 2.0 * edge * edge);
            }
        }
        acc *= cov * uFade;
        float lum = dot(acc, vec3(0.3, 0.5, 0.2)) + 1e-4 * cov;
        oCol = vec4(acc, 0.0);
        oDep = vec2(t0 * lum, lum);
    });

/* the backdrop on its own (bubbles and mist are added over it) */
static const char *FS_BACK = GLSL(
    uniform float uFar;
    void main() {
        oCol = vec4(backdrop(wallPos()) * uFade, 1.0);
        oDep = vec2(uFar * 1e-4, 1e-4);
    });

/* ---- mist: droplets flung off a bursting film, instanced point quads ---- */
static const char *VS_MIST = GLSL(
    layout(location = 0) in vec4 aP;     /* position, radius */
    layout(location = 1) in vec4 aC;     /* brightness, -, -, - */
    uniform vec4 uOut;
    uniform vec3 uEye;
    uniform float uPx;
    out vec2 vC;
    out float vB, vD;
    void main() {
        vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float((gl_VertexID >> 1) & 1) * 2.0 - 1.0);
        float D = -uEye.z, z = aP.z + D;
        vec2 pc = uEye.xy + (aP.xy - uEye.xy) * (D / max(z, 0.05));
        float rp = aP.w * D / max(z, 0.05);
        float rs = max(rp, uPx * 1.2);
        vB = aC.x * (rp * rp) / (rs * rs);
        vD = length(vec3(aP.xy, aP.z) - uEye);
        vC = corner;
        vec2 P = pc + corner * rs * 1.5;
        gl_Position = vec4((P - uOut.xy) / uOut.zw * 2.0 - 1.0, 0.0, 1.0);
    });

static const char *FS_MIST = GLSL(
    in vec2 vC;
    in float vB, vD;
    void main() {
        float r = length(vC * 1.5);
        float a = exp(-r * r * 2.2);
        vec3 c = vec3(1.0, 0.97, 0.92) * vB * a * 0.9 * uFade;
        float lum = dot(c, vec3(0.3, 0.5, 0.2)) + 1e-5;
        oCol = vec4(c, 0.0);
        oDep = vec2(vD * lum, lum);
    });

/* ---- depth of field: a half-resolution gather (after Gustafsson's single
 * pass bokeh), circles of confusion from a thin lens ---- */
static const char *FS_DOWN = GLSL(
    uniform sampler2D uCol, uDep;
    uniform vec2 uSrc;          /* source texel */
    uniform float uFocus, uAperture, uPxW, uMaxC;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy * 3.0 * uSrc;
        vec2 e = uSrc * 0.75;
        vec3 c = 0.25 * (texture(uCol, uv + vec2(-e.x, -e.y)).rgb + texture(uCol, uv + vec2(e.x, -e.y)).rgb +
                         texture(uCol, uv + vec2(-e.x, e.y)).rgb + texture(uCol, uv + vec2(e.x, e.y)).rgb);
        vec2 d = texture(uDep, uv).rg;
        float z = d.x / max(d.y, 1e-6);
        /* signed: far +, near -, in half-res pixels */
        float coc = uAperture * (z - uFocus) / max(z, 0.05) / uPxW;
        o = vec4(c, clamp(coc, -uMaxC, uMaxC));   /* in this pass's pixels */
    });

static const char *FS_DOF = GLSL(
    uniform sampler2D uHalf;
    uniform vec2 uTx;
    uniform float uMaxC;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy * uTx;
        vec4 c0 = textureLod(uHalf, uv, 0.0);
        float cs = abs(c0.a);
        vec3 acc = c0.rgb;
        float tot = 1.0, rad = 1.0;
        /* only as far out as anything near here is blurred */
        float reach = min(uMaxC, max(cs, max(abs(textureLod(uHalf, uv, 2.0).a), abs(textureLod(uHalf, uv, 4.0).a)) * 2.5) + 1.0);
        for (float ang = 0.0; rad < reach; ang += 2.39996323) {
            vec4 s = textureLod(uHalf, uv + vec2(cos(ang), sin(ang)) * uTx * rad, 0.0);
            float ss = abs(s.a);
            if (s.a > c0.a) ss = clamp(ss, 0.0, cs * 2.0);
            float m = smoothstep(rad - 0.5, rad + 0.5, ss);
            acc += mix(acc / tot, s.rgb, m);
            tot += 1.0;
            rad += 0.9 / rad;
        }
        o = vec4(acc / tot, cs);
    });

/* ---- the print: depth of field blended in, a soft bloom, the filmic
 * curve, a little warmth, vignette and grain ---- */
static const char *FS_FINAL = GLSL(
    uniform sampler2D uCol, uDofT;
    uniform vec2 uRes;
    uniform vec4 uOutN;         /* this output in the wall, normalised by wall height: x0 y0 w h */
    uniform vec2 uWallC;        /* the wall's centre and half-diagonal for the vignette */
    uniform float uWallR, uExpo, uGrainT;
    uniform sampler2D uNoise;
    out vec4 o;
    vec3 aces(vec3 x) { return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec3 sharp = texture(uCol, uv).rgb;
        vec4 dof = texture(uDofT, uv);
        vec3 c = mix(sharp, dof.rgb, smoothstep(0.3, 0.9, dof.a));
        vec3 bl = textureLod(uDofT, uv, 2.0).rgb * 0.5 + textureLod(uDofT, uv, 4.0).rgb * 0.5;
        c += bl * 0.035;
        c *= uExpo;
        /* vignette, over the whole wall */
        vec2 P = uOutN.xy + uv * uOutN.zw;
        float v = length(P - uWallC) / uWallR;
        c *= 1.0 - 0.38 * smoothstep(0.35, 1.15, v);
        c = aces(c);
        c = pow(c, vec3(1.0 / 2.2));
        /* the faintest split tone: warm highlights, cool deep shadows */
        c = mix(c, c * vec3(1.02, 1.0, 0.97), smoothstep(0.3, 0.9, c.g));
        c += (texture(uNoise, gl_FragCoord.xy / 256.0 + vec2(uGrainT * 0.37, uGrainT * 0.61)).a - 0.5) * 0.018;
        c += (texelFetch(uNoise, ivec2(gl_FragCoord.xy) & 255, 0).a - 0.5) / 255.0;
        o = vec4(c, 1.0);
    });

/* ---- caption text ---- */
static const char *VS_TEXT = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aP;     /* x, y (pixels, y down), u, v (atlas texels) */
    layout(location = 1) in vec4 aC;
    uniform vec2 uRes, uAtlas;
    out vec2 vT;
    out vec4 vC;
    void main() {
        gl_Position = vec4(aP.x / uRes.x * 2.0 - 1.0, 1.0 - aP.y / uRes.y * 2.0, 0.0, 1.0);
        vT = aP.zw / uAtlas;
        vC = aC;
    });
static const char *FS_TEXT = "#version 330 core\n" GLSL(
    uniform sampler2D uAt;
    in vec2 vT;
    in vec4 vC;
    out vec4 o;
    void main() { o = vec4(vC.rgb, vC.a * texture(uAt, vT).r); });

/* ================= state ================= */

static GLuint p_curl, p_vel, p_div, p_jac, p_proj, p_hadv, p_hnew, p_init;
static GLuint p_film, p_back, p_bub, p_mist, p_down, p_dof, p_final, p_text;
static GLuint vbo_mist, vao_mist, vbo_text, vao_text, tex_bub, tex_atlas;

/* render targets at the internal resolution */
static int IW, IH;
static GLuint scene_fbo, scene_col, scene_dep;
static RT half_rt, dof_rt;

static void scene_targets(float budget) {
    float s = opt_scale > 0 ? opt_scale : fminf(1.0f, sqrtf(budget / ((float)W * H)));
    int iw = (int)lroundf(W * s), ih = (int)lroundf(H * s);
    if (iw == IW && ih == IH && scene_fbo) return;
    IW = iw; IH = ih;
    if (scene_fbo) { glDeleteFramebuffers(1, &scene_fbo); glDeleteTextures(1, &scene_col); glDeleteTextures(1, &scene_dep); }
    GLuint t[2];
    glGenTextures(2, t);
    scene_col = t[0]; scene_dep = t[1];
    glBindTexture(GL_TEXTURE_2D, scene_col);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, IW, IH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, scene_dep);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, IW, IH, 0, GL_RG, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &scene_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, scene_col, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, scene_dep, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    rt_make(&half_rt, (IW + 2) / 3, (IH + 2) / 3, GL_RGBA16F, GL_LINEAR, 1);
    rt_make(&dof_rt, (IW + 2) / 3, (IH + 2) / 3, GL_RGBA16F, GL_LINEAR, 1);
}

/* the view: wall units are wall heights, y up, origin at the wall's bottom left */
static float wallA;                    /* wall aspect */
static float out_w0[4];                /* this output's rect in wall units: x0, y0, w, h */
static void view_update(void) {
    if (span_me < 0) {
        wallr[0] = wallr[1] = 0; wallr[2] = (float)W; wallr[3] = (float)H;
        memcpy(outr, wallr, sizeof outr);
    }
    float wh = wallr[3];
    wallA = wallr[2] / wh;
    out_w0[0] = (outr[0] - wallr[0]) / wh;
    out_w0[1] = 1 - (outr[1] + outr[3] - wallr[1]) / wh;
    out_w0[2] = outr[2] / wh;
    out_w0[3] = outr[3] / wh;
    if (span_me < 0) { eye_x = wallA * 0.5f; eye_y = 0.5f; }
}

/* uniforms every scene pass shares */
static float cur_fade, cur_time;
static void set_common(GLuint p, float eyeD) {
    glUseProgram(p);
    glUniform1i(U(p, "uLut"), 0);
    glUniform1i(U(p, "uNoise"), 1);
    glUniform1i(U(p, "uN3"), 2);
    glUniform4f(U(p, "uMap"), out_w0[0], out_w0[1], out_w0[2] / IW, out_w0[3] / IH);
    glUniform3f(U(p, "uEye"), eye_x, eye_y, -eyeD);
    glUniform1f(U(p, "uTime"), cur_time);
    glUniform1f(U(p, "uFade"), cur_fade);
}

static void norm3(float *v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); v[0] /= l; v[1] /= l; v[2] /= l; }

/* a rect light facing back along -dir: its frame */
static void light_frame(GLuint p, const char *dn, const char *un, const char *vn, float dx, float dy, float dz) {
    float d[3] = {dx, dy, dz}, up[3] = {0, 1, 0}, u[3], v[3];
    norm3(d);
    u[0] = up[1] * d[2] - up[2] * d[1]; u[1] = up[2] * d[0] - up[0] * d[2]; u[2] = up[0] * d[1] - up[1] * d[0];
    norm3(u);
    v[0] = d[1] * u[2] - d[2] * u[1]; v[1] = d[2] * u[0] - d[0] * u[2]; v[2] = d[0] * u[1] - d[1] * u[0];
    glUniform3fv(U(p, dn), 1, d);
    glUniform3fv(U(p, un), 1, u);
    glUniform3fv(U(p, vn), 1, v);
}

/* ================= the film ================= */
static struct {
    int act, film;                     /* which film this state is */
    long steps;
    int vx, vy, hx, hy;
    float ad;
    RT vel[2], prs[2], curl, div, h[2], hhat;
    int cv, cp, ch;
    float seed;
} fm = {.act = -1};

static void film_frame(float *cx, float *cy, float *hw, float *hh) {
    float m = 0.034f;
    *cx = wallA * 0.5f; *cy = 0.5f;
    *hw = wallA * 0.5f - m; *hh = 0.5f - m;
}

static void film_alloc(void) {
    float cx, cy, hw, hh;
    film_frame(&cx, &cy, &hw, &hh);
    float ad = hw / hh;
    /* thickness texels per frame height: about one per 2 logical px of the
     * wall (the same on every output, so they all run the same film),
     * velocity a third of that */
    float pxh = wallr[3] * hh * 2;
    int hy = (int)clampf(pxh / 2.0f, 300, 900);
    int hx = (int)lroundf(hy * ad);
    if (hx == fm.hx && hy == fm.hy) return;
    fm.hx = hx; fm.hy = hy; fm.ad = ad;
    fm.vx = (hx + 2) / 3; fm.vy = (hy + 2) / 3;
    for (int i = 0; i < 2; i++) {
        rt_make(&fm.vel[i], fm.vx, fm.vy, GL_RG16F, GL_LINEAR, 0);
        rt_make(&fm.prs[i], fm.vx, fm.vy, GL_R32F, GL_NEAREST, 0);
        rt_make(&fm.h[i], hx, hy, GL_R16F, GL_LINEAR, 1);
    }
    rt_make(&fm.curl, fm.vx, fm.vy, GL_R16F, GL_LINEAR, 0);
    rt_make(&fm.div, fm.vx, fm.vy, GL_R32F, GL_NEAREST, 0);
    rt_make(&fm.hhat, hx, hy, GL_R16F, GL_LINEAR, 0);
    fm.act = -1;
}

/* the drainage profile's parameters at film age tau */
static void film_profile(float tau, float life, float *hb, float *s, float *yb) {
    *hb = 0.78f + 2.0f * expf(-tau / 6.0f);
    *s = expf(-tau / 3.5f);
    float tb = life * 0.38f, x = clampf((tau - tb) / (life - tb), 0, 1);
    *yb = 0.24f * powf(x, 1.15f);
}

static void sim_common(GLuint p, int gx, int gy, float tau) {
    glUseProgram(p);
    glUniform1i(U(p, "uVel"), 0); glUniform1i(U(p, "uH"), 1); glUniform1i(U(p, "uHhat"), 2);
    glUniform1i(U(p, "uCurl"), 3); glUniform1i(U(p, "uDiv"), 4); glUniform1i(U(p, "uP"), 5);
    glUniform1i(U(p, "uNoise"), 6); glUniform1i(U(p, "uN3"), 7);
    glUniform2f(U(p, "uRes"), (float)gx, (float)gy);
    glUniform1f(U(p, "uDt"), 1.0f / SIMHZ);
    glUniform1f(U(p, "uAd"), fm.ad);
    glUniform1f(U(p, "uDx"), 1.0f / fm.vy);
    glUniform1f(U(p, "uTau"), tau);
    glUniform1f(U(p, "uSeed"), fm.seed);
}

static void film_init(int act, int film, uint32_t seed) {
    film_alloc();
    fm.act = act; fm.film = film; fm.steps = 0;
    fm.seed = (seed % 1000) / 1000.0f * 7.0f;
    fm.cv = fm.cp = fm.ch = 0;
    glDisable(GL_BLEND);
    tex_unit(7, GL_TEXTURE_3D, tex_noise3);
    float hb, s, yb;
    film_profile(0, 80, &hb, &s, &yb);
    sim_common(p_init, fm.hx, fm.hy, 0);
    glUniform1f(U(p_init, "uMode"), 0);
    glUniform1f(U(p_init, "uHb"), hb);
    rt_bind(&fm.h[0]); draw_fs();
    glBindTexture(GL_TEXTURE_2D, fm.h[0].tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    sim_common(p_init, fm.vx, fm.vy, 0);
    glUniform1f(U(p_init, "uMode"), 1);
    rt_bind(&fm.vel[0]); draw_fs();
    for (int i = 0; i < 2; i++) { rt_bind(&fm.prs[i]); glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT); }
}

static void film_step(float life) {
    float tau = (float)fm.steps / SIMHZ;
    float hb, s, yb;
    film_profile(tau, life, &hb, &s, &yb);
    glDisable(GL_BLEND);
    tex_unit(6, GL_TEXTURE_2D, tex_noise);
    tex_unit(7, GL_TEXTURE_3D, tex_noise3);
    /* curl */
    sim_common(p_curl, fm.vx, fm.vy, tau);
    tex_unit(0, GL_TEXTURE_2D, fm.vel[fm.cv].tex);
    rt_bind(&fm.curl); draw_fs();
    /* velocity: advected, forced */
    sim_common(p_vel, fm.vx, fm.vy, tau);
    glUniform1f(U(p_vel, "uBuoy"), 0.10f);
    glUniform1f(U(p_vel, "uVort"), 2.2f);
    glUniform1f(U(p_vel, "uDrag"), 0.10f);
    glUniform1f(U(p_vel, "uStir"), 0.0035f + 0.03f * expf(-tau / 3.0f));
    glUniform1f(U(p_vel, "uLod"), 4.0f);
    tex_unit(0, GL_TEXTURE_2D, fm.vel[fm.cv].tex);
    tex_unit(1, GL_TEXTURE_2D, fm.h[fm.ch].tex);
    tex_unit(3, GL_TEXTURE_2D, fm.curl.tex);
    rt_bind(&fm.vel[fm.cv ^ 1]); draw_fs();
    fm.cv ^= 1;
    /* made divergence free */
    sim_common(p_div, fm.vx, fm.vy, tau);
    tex_unit(0, GL_TEXTURE_2D, fm.vel[fm.cv].tex);
    rt_bind(&fm.div); draw_fs();
    sim_common(p_jac, fm.vx, fm.vy, tau);
    tex_unit(4, GL_TEXTURE_2D, fm.div.tex);
    for (int i = 0; i < 24; i++) {
        tex_unit(5, GL_TEXTURE_2D, fm.prs[fm.cp].tex);
        rt_bind(&fm.prs[fm.cp ^ 1]); draw_fs();
        fm.cp ^= 1;
    }
    sim_common(p_proj, fm.vx, fm.vy, tau);
    tex_unit(0, GL_TEXTURE_2D, fm.vel[fm.cv].tex);
    tex_unit(5, GL_TEXTURE_2D, fm.prs[fm.cp].tex);
    rt_bind(&fm.vel[fm.cv ^ 1]); draw_fs();
    fm.cv ^= 1;
    /* thickness */
    sim_common(p_hadv, fm.hx, fm.hy, tau);
    tex_unit(0, GL_TEXTURE_2D, fm.vel[fm.cv].tex);
    tex_unit(1, GL_TEXTURE_2D, fm.h[fm.ch].tex);
    rt_bind(&fm.hhat); draw_fs();
    sim_common(p_hnew, fm.hx, fm.hy, tau);
    glUniform1f(U(p_hnew, "uHb"), hb);
    glUniform1f(U(p_hnew, "uS"), s);
    glUniform1f(U(p_hnew, "uYb"), yb);
    glUniform1f(U(p_hnew, "uRelax"), 6.0f);
    tex_unit(2, GL_TEXTURE_2D, fm.hhat.tex);
    rt_bind(&fm.h[fm.ch ^ 1]); draw_fs();
    fm.ch ^= 1;
    glBindTexture(GL_TEXTURE_2D, fm.h[fm.ch].tex);   /* unit 2 is still the active one */
    tex_unit(1, GL_TEXTURE_2D, fm.h[fm.ch].tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    fm.steps++;
}

/* ================= mist ================= */
#define MAXMIST 4096
static float mist_buf[MAXMIST * 8];
static int mist_n;

static void mist_add(float x, float y, float z, float r, float b) {
    if (mist_n >= MAXMIST) return;
    float *m = mist_buf + mist_n++ * 8;
    m[0] = x; m[1] = y; m[2] = z; m[3] = r; m[4] = b; m[5] = m[6] = m[7] = 0;
}

/* a droplet launched at time tl from p with velocity v, seen at time t */
static int droplet(float tl, float t, const float *p, const float *v, float life, float *out) {
    float e = t - tl;
    if (e < 0 || e > life) return 0;
    float tau = 0.3f, k = tau * (1 - expf(-e / tau));
    out[0] = p[0] + v[0] * k;
    out[1] = p[1] + v[1] * k - 0.35f * (e - k) * tau * 2.0f;
    out[2] = p[2] + v[2] * k;
    return 1;
}

/* the film's pop: droplets thrown off its rim as the hole passes */
static void mist_film(float cx, float cy, float vp, float e, uint32_t seed, float px, float py, float hw, float hh) {
    uint32_t save = rng;
    rng = seed;
    for (int i = 0; i < 1400; i++) {
        float ang = frange(0, 2 * PI), r0 = sqrtf(frand()) * vp * 1.4f;
        float x0 = cx + r0 * cosf(ang), y0 = cy + r0 * sinf(ang);
        float sp = vp * frange(0.55f, 0.95f), life = frange(0.25f, 1.1f), rad = frange(0.0006f, 0.002f);
        float zv = frange(-0.25f, 0.12f);
        float br = frange(0.4f, 2.4f);
        if (fabsf(x0 - px) > hw || fabsf(y0 - py) > hh) continue;   /* only where there was film */
        float p[3] = {x0, y0, 0}, v[3] = {sp * cosf(ang), sp * sinf(ang) + frange(-0.05f, 0.1f), zv}, o[3];
        if (droplet(r0 / vp, e, p, v, life, o)) {
            float age = (e - r0 / vp) / life;
            mist_add(o[0], o[1], o[2], rad, br * (1 - age) * (1 - age) * (1 - age) * 3.0f);
        }
    }
    rng = save;
}

static void mist_draw(void) {
    if (!mist_n) return;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mist);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(mist_n * 8 * sizeof(float)), mist_buf, GL_STREAM_DRAW);
    glBindVertexArray(vao_mist);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, mist_n);
    glDisable(GL_BLEND);
}

/* ================= bubbles ================= */
#define NB 56
#define POPB 0.42f                     /* a bubble's burst, slowed down */
typedef struct {
    int on, popping;
    float p[3], v[3], r;
    float born, popt;                  /* times (act-local) */
    float pd[3];                       /* where it bursts from */
    float h0, seed, spin, spinv, life;
    int link[4], nl;
} Bub;
typedef struct { float t, p[3], v[3], r, pd[3]; uint32_t seed; } PopEv;
#define NPOPEV 24
static struct {
    int act;
    long steps;
    uint32_t rng;
    float next_spawn;
    Bub b[NB];
    PopEv ev[NPOPEV];
    int nev;
    float wind[6][7];                  /* air: modes k (3), amplitude dir (3), phase */
} bs = {.act = -1};

#define BUB_D 1.7f                     /* the eye's distance from the screen plane */

/* the wall seen through at depth z: x0, x1, y0, y1 */
static void vis(float z, float *v) {
    float k = (BUB_D + z) / BUB_D;
    v[0] = eye_x - eye_x * k; v[1] = eye_x + (wallA - eye_x) * k;
    v[2] = eye_y - eye_y * k; v[3] = eye_y + (1 - eye_y) * k;
}

static void air(const float *p, float t, float *v) {
    v[0] = 0.022f; v[1] = 0.003f; v[2] = 0;
    for (int m = 0; m < 6; m++) {
        float *w = bs.wind[m];
        float s = sinf(w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + w[6] + t * 0.05f * (1 + m * 0.3f));
        v[0] += w[3] * s; v[1] += w[4] * s; v[2] += w[5] * s;
    }
}

static int bub_count(void) { int n = 0; for (int i = 0; i < NB; i++) n += bs.b[i].on; return n; }
static int bub_target(void) { return (int)lroundf(11.0f * wallA) + 6; }

static void bub_new(Bub *b, float t, int entering) {
    memset(b, 0, sizeof *b);
    b->on = 1;
    float u = frand();
    b->r = 0.035f + 0.16f * u * u;
    float z = frange(-0.2f, 1.3f);
    if (frand() < 0.12f) z = frange(1.6f, 3.2f);       /* some far off, small in the dark */
    b->p[2] = z;
    float v[4];
    vis(z, v);
    if (entering && frand() < 0.45f) {
        /* drifting down from above */
        b->p[0] = v[0] + (v[1] - v[0]) * frange(0.05f, 0.95f);
        b->p[1] = v[3] + b->r * 1.2f;
    } else if (entering) {
        /* blown in from the left */
        b->p[0] = v[0] - b->r * 1.2f;
        b->p[1] = v[2] + (v[3] - v[2]) * frange(0.1f, 0.9f);
    } else {
        b->p[0] = v[0] + (v[1] - v[0]) * frand();
        b->p[1] = v[2] + (v[3] - v[2]) * frange(0.08f, 0.92f);
    }
    b->v[0] = frange(0.01f, 0.04f); b->v[1] = frange(-0.01f, 0.012f);
    b->born = t - (entering ? 0 : frange(0, 30));
    b->life = frange(30, 110);
    b->popt = b->born + b->life;
    b->h0 = frange(0.3f, 0.62f);
    b->seed = frand() * 50;
    b->spin = frange(0, 6.28f);
    b->spinv = frange(-0.12f, 0.12f);
    float a = frange(0, 2 * PI), tilt = frange(0, 0.75f);
    b->pd[0] = sinf(tilt) * cosf(a); b->pd[1] = cosf(tilt); b->pd[2] = sinf(tilt) * sinf(a);
}

static void unlink_bub(int i) {
    for (int k = 0; k < bs.b[i].nl; k++) {
        Bub *o = &bs.b[bs.b[i].link[k]];
        for (int m = 0; m < o->nl; m++)
            if (o->link[m] == i) { o->link[m] = o->link[--o->nl]; break; }
    }
    bs.b[i].nl = 0;
}

static void bubbles_init(int act, uint32_t seed) {
    memset(&bs, 0, sizeof bs);
    bs.act = act;
    rng = mix32(seed ^ 0xb0bb1e5U);
    for (int m = 0; m < 6; m++) {
        float k[3] = {frange(-1, 1), frange(-1, 1), frange(-0.5f, 0.5f)}, a[3] = {frange(-1, 1), frange(-1, 1), frange(-1, 1)};
        norm3(k);
        float kk = frange(1.5f, 4.5f), d = a[0] * k[0] + a[1] * k[1] + a[2] * k[2];
        for (int i = 0; i < 3; i++) a[i] -= d * k[i];          /* across k: the field has no divergence */
        norm3(a);
        float amp = frange(0.006f, 0.014f);
        for (int i = 0; i < 3; i++) { bs.wind[m][i] = k[i] * kk; bs.wind[m][3 + i] = a[i] * amp; }
        bs.wind[m][6] = frange(0, 6.28f);
    }
    int n = bub_target();
    for (int i = 0; i < n && i < NB; i++) bub_new(&bs.b[i], 0, 0);
    bs.next_spawn = frange(1, 4);
    bs.rng = rng;
}

static void bubbles_step(void) {
    float dt = 1.0f / SIMHZ, t = (float)bs.steps / SIMHZ;
    rng = bs.rng;
    for (int i = 0; i < NB; i++) {
        Bub *b = &bs.b[i];
        if (!b->on) continue;
        float a[3];
        air(b->p, t, a);
        float k = 1 - expf(-dt / 1.6f);
        for (int j = 0; j < 3; j++) b->v[j] += (a[j] - b->v[j]) * k;
        b->v[1] -= 0.0015f * dt;
        for (int j = 0; j < 3; j++) b->p[j] += b->v[j] * dt;
        b->spin += b->spinv * dt;
        if (!b->popping && t >= b->popt) {
            b->popping = 1;
            if (getenv("SOAP_DEBUG")) fprintf(stderr, "pop %d at %.2f: (%.2f %.2f %.2f) r %.3f\n", i, t, b->p[0], b->p[1], b->p[2], b->r);
            if (bs.nev < NPOPEV) {
                PopEv *e = &bs.ev[bs.nev++];
                e->t = b->popt;
                memcpy(e->p, b->p, sizeof e->p); memcpy(e->v, b->v, sizeof e->v); memcpy(e->pd, b->pd, sizeof e->pd);
                e->r = b->r;
                e->seed = mix32((uint32_t)i * 977U ^ (uint32_t)(b->popt * 1000));
            }
        }
        if (b->popping && t >= b->popt + POPB) {
            unlink_bub(i);
            b->on = 0;
            continue;
        }
        /* drifted out of sight */
        float v[4];
        vis(b->p[2], v);
        if (b->p[0] > v[1] + b->r * 1.5f || b->p[1] > v[3] + b->r * 2 || b->p[1] < v[2] - b->r * 2 ||
            b->p[2] < -0.9f || b->p[2] > 4.0f) { unlink_bub(i); b->on = 0; }
    }
    /* forget old pops */
    while (bs.nev && t - bs.ev[0].t > POPB + 2.5f) { memmove(bs.ev, bs.ev + 1, (size_t)(--bs.nev) * sizeof *bs.ev); }
    /* touching bubbles join; joined ones hold at the distance where their
     * films meet at 120 degrees: d^2 = r1^2 + r2^2 - r1 r2 */
    for (int it = 0; it < 3; it++)
        for (int i = 0; i < NB; i++) {
            Bub *a = &bs.b[i];
            if (!a->on) continue;
            for (int j = i + 1; j < NB; j++) {
                Bub *b = &bs.b[j];
                if (!b->on) continue;
                float d[3] = {b->p[0] - a->p[0], b->p[1] - a->p[1], b->p[2] - a->p[2]};
                float dist = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 1e-6f;
                int linked = 0;
                for (int k = 0; k < a->nl; k++) linked |= a->link[k] == j;
                float target;
                if (linked) target = sqrtf(a->r * a->r + b->r * b->r - a->r * b->r);
                else if (dist < a->r + b->r) {
                    if (it == 0 && !a->popping && !b->popping && a->nl < 3 && b->nl < 3) {
                        a->link[a->nl++] = j; b->link[b->nl++] = i;
                        float ma = a->r * a->r * a->r, mb = b->r * b->r * b->r;
                        for (int k = 0; k < 3; k++) a->v[k] = b->v[k] = (a->v[k] * ma + b->v[k] * mb) / (ma + mb);
                        target = sqrtf(a->r * a->r + b->r * b->r - a->r * b->r);
                    } else target = a->r + b->r;
                } else continue;
                if (!linked && dist >= target) continue;
                float ma = a->r * a->r * a->r, mb = b->r * b->r * b->r;
                float corr = (dist - target) / dist * 0.25f;
                for (int k = 0; k < 3; k++) {
                    a->p[k] += d[k] * corr * mb / (ma + mb);
                    b->p[k] -= d[k] * corr * ma / (ma + mb);
                }
            }
        }
    /* blow new ones in from the left */
    if (t >= bs.next_spawn) {
        if (bub_count() < bub_target())
            for (int i = 0; i < NB; i++)
                if (!bs.b[i].on) { bub_new(&bs.b[i], t, 1); break; }
        bs.next_spawn = t + frange(0.5f, 2.2f) * 1.6f / fmaxf(wallA, 0.8f);
    }
    bs.rng = rng;
    bs.steps++;
}

static float bub_tex[NB * 8 * 4];
static void bubbles_upload(float t) {
    memset(bub_tex, 0, sizeof bub_tex);
    for (int i = 0; i < NB; i++) {
        Bub *b = &bs.b[i];
        if (!b->on) continue;
        float *o = bub_tex + i * 32;
        o[0] = b->p[0]; o[1] = b->p[1]; o[2] = b->p[2]; o[3] = b->r;
        o[4] = b->pd[0]; o[5] = b->pd[1]; o[6] = b->pd[2];
        o[7] = b->popping ? clampf((t - b->popt) / POPB, 0, 1) : -1;
        float age = t - b->born, x = clampf(age / b->life, 0, 1);
        o[8] = fmaxf(b->h0 * expf(-age / 120.0f), 0.16f);
        o[9] = b->seed;
        o[10] = x > 0.7f ? 0.55f * smooth01((x - 0.7f) / 0.3f) + 0.02f : 0;
        o[11] = b->spin;
        for (int k = 0; k < b->nl && k < 4; k++) {
            Bub *j = &bs.b[b->link[k]];
            o[12 + k] = i < b->link[k] ? 1 : 0;
            o[16 + 4 * k] = j->p[0]; o[17 + 4 * k] = j->p[1]; o[18 + 4 * k] = j->p[2]; o[19 + 4 * k] = j->r;
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex_bub);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, NB, GL_RGBA, GL_FLOAT, bub_tex);
}

/* droplets from the bubbles that burst lately */
static void mist_bubbles(float t) {
    uint32_t save = rng;
    for (int k = 0; k < bs.nev; k++) {
        PopEv *e = &bs.ev[k];
        rng = e->seed;
        int n = (int)(220 * e->r / 0.08f) + 40;
        for (int i = 0; i < n && i < 600; i++) {
            /* a point on the sphere the rim passes at angle th from the pop point */
            float th = acosf(1 - 2 * frand()) * 0.999f, ph = frange(0, 2 * PI);
            float t1[3], t2[3], *a = e->pd;
            float up[3] = {0, 1, 0};
            if (fabsf(a[1]) > 0.9f) { up[0] = 1; up[1] = 0; }
            t1[0] = a[1] * up[2] - a[2] * up[1]; t1[1] = a[2] * up[0] - a[0] * up[2]; t1[2] = a[0] * up[1] - a[1] * up[0];
            norm3(t1);
            t2[0] = a[1] * t1[2] - a[2] * t1[1]; t2[1] = a[2] * t1[0] - a[0] * t1[2]; t2[2] = a[0] * t1[1] - a[1] * t1[0];
            float q[3], tg[3];
            for (int j = 0; j < 3; j++) {
                float rad = cosf(ph) * t1[j] + sinf(ph) * t2[j];
                q[j] = cosf(th) * a[j] + sinf(th) * rad;
                tg[j] = -sinf(th) * a[j] + cosf(th) * rad;        /* the rim's way */
            }
            float tl = e->t + POPB * th / 3.5f;
            float sp = frange(0.15f, 0.6f) * e->r * 3.5f / POPB * 0.25f;
            float p[3], v[3], o[3];
            for (int j = 0; j < 3; j++) {
                p[j] = e->p[j] + q[j] * e->r;
                v[j] = e->v[j] + tg[j] * sp + q[j] * frange(0.0f, 0.1f) * sp;
            }
            float life = frange(0.6f, 1.6f);
            if (droplet(tl, t, p, v, life, o)) {
                float age = (t - tl) / life;
                mist_add(o[0], o[1], o[2], frange(0.0005f, 0.0016f), frange(0.5f, 2.5f) * (1 - age) * (1 - age) * 3.0f);
            }
        }
    }
    rng = save;
}

static float dof_focus, dof_aperture;
static void bind_scene_tex(void);

/* ================= foam =================
 * A raft of bubbles pressed into cells, seen from above: a power diagram
 * (Laguerre cells; centres and weights) kept close to a relaxed foam by
 * rasterising it coarsely on the CPU a few times a second (each cell moves
 * to its centroid, its weight chases its target area, neighbours are
 * counted), coarsening by von Neumann's law dA/dt = k (n - 6), with films
 * between cells rupturing (the border retracts to its ends and the two
 * merge) and caps bursting (the cell opens and its neighbours close in).
 * On the GPU each pixel finds its cell and nearest border among the cells
 * binned near it: the face is a shallow dome of film, the border a glassy
 * Plateau border, wider at the junctions. */
#define NCELL 900
#define FOAMHZ 6
#define FBIN 0.08f
#define FBCAP 48
typedef struct {
    int on;
    float x, y, w, target, area, cx, cy;
    float px, py, pw;                  /* before the last step, to glide from */
    float born, h0, seed;
    int partner;                       /* ruptured into this one (-1 none) */
    float rup_t;                       /* when the film between them broke */
    float burst_t;                     /* when its cap burst (< 0 never) */
    int nn;
} Cell;
static struct {
    int act;
    long steps;
    uint32_t rng;
    float x0, y0, x1, y1;              /* the simulated domain (wall units) */
    int n;
    float next_rup, next_burst;
    Cell c[NCELL];
} fo = {.act = -1};
#define RUPD 0.35f                     /* a border retracting */
#define BURSTD 0.45f                   /* a cap opening */

static int *fbin;                      /* the CPU raster's bins: FBCAP ids each */
static int fbx, fby, fbin_over;
#define FST 0.011f                     /* the CPU raster's spacing */
static int fo_nb[NCELL][16], fo_nnb[NCELL], fo_gw, fo_gh, fo_own_dirty;
static float *fo_own;

static void foam_bins(float grow) {
    fbx = (int)ceilf((fo.x1 - fo.x0) / FBIN); fby = (int)ceilf((fo.y1 - fo.y0) / FBIN);
    fbin = realloc(fbin, sizeof(int) * (size_t)fbx * fby * FBCAP);
    for (int i = 0; i < fbx * fby * FBCAP; i++) fbin[i] = -1;
    for (int i = 0; i < fo.n; i++) {
        Cell *c = &fo.c[i];
        if (!c->on) continue;
        float a = fmaxf(fmaxf(c->area, c->target), fminf(c->w * PI, 3 * fmaxf(c->area, c->target)));
        float r = sqrtf(fmaxf(a / PI, 1e-5f)) * grow + 0.012f;
        int bx0 = (int)floorf((c->x - r - fo.x0) / FBIN), bx1 = (int)floorf((c->x + r - fo.x0) / FBIN);
        int by0 = (int)floorf((c->y - r - fo.y0) / FBIN), by1 = (int)floorf((c->y + r - fo.y0) / FBIN);
        for (int by = by0 < 0 ? 0 : by0; by <= by1 && by < fby; by++)
            for (int bx = bx0 < 0 ? 0 : bx0; bx <= bx1 && bx < fbx; bx++) {
                /* the bin's nearest point to the centre */
                float qx = clampf(c->x, fo.x0 + bx * FBIN, fo.x0 + (bx + 1) * FBIN), qy = clampf(c->y, fo.y0 + by * FBIN, fo.y0 + (by + 1) * FBIN);
                if ((qx - c->x) * (qx - c->x) + (qy - c->y) * (qy - c->y) > r * r) continue;
                int *b = fbin + (by * fbx + bx) * FBCAP;
                int k = 0;
                while (k < FBCAP && b[k] >= 0) k++;
                if (k < FBCAP) b[k] = i;
                else fbin_over++;
            }
    }
}

static int foam_owner(float x, float y) {
    int bx = (int)((x - fo.x0) / FBIN), by = (int)((y - fo.y0) / FBIN);
    if (bx < 0 || by < 0 || bx >= fbx || by >= fby) return -1;
    int *b = fbin + (by * fbx + bx) * FBCAP, best = -1;
    float bd = 1e9f;
    for (int k = 0; k < FBCAP && b[k] >= 0; k++) {
        Cell *c = &fo.c[b[k]];
        float d = (x - c->x) * (x - c->x) + (y - c->y) * (y - c->y) - c->w;
        if (d < bd) { bd = d; best = b[k]; }
    }
    return best;
}

static float foam_rm(void) { return 0.052f; }

static void foam_init(int act, uint32_t seed) {
    memset(&fo, 0, sizeof fo);
    fo.act = act;
    rng = mix32(seed ^ 0xf0a3U);
    float m = 0.1f;
    fo.x0 = -m; fo.y0 = -m; fo.x1 = wallA + m; fo.y1 = 1 + m;
    float rm = foam_rm(), area = (fo.x1 - fo.x0) * (fo.y1 - fo.y0);
    int n = (int)(area / (PI * rm * rm * 0.92f));
    if (n > NCELL) n = NCELL;
    fo.n = n;
    for (int i = 0; i < n; i++) {
        Cell *c = &fo.c[i];
        c->on = 1;
        c->x = frange(fo.x0, fo.x1); c->y = frange(fo.y0, fo.y1);
        float s = expf(frange(-0.6f, 0.6f));             /* a spread of sizes */
        c->target = PI * rm * rm * s * s;
        c->w = c->target / PI * 0.5f;
        c->area = c->target;
        c->born = -frange(0, 60);
        c->h0 = frange(0.18f, 0.5f);
        c->seed = frand() * 40;
        c->partner = -1;
        c->burst_t = -1;
    }
    fo.next_rup = frange(1, 3);
    fo.next_burst = frange(2, 5);
    fo.rng = rng;
}

/* one relaxation and coarsening step (dt = 1 / FOAMHZ) */
static void foam_step(int prewarm) {
    float dt = 1.0f / FOAMHZ, t = (float)fo.steps / FOAMHZ;
    rng = fo.rng;
    foam_bins(2.0f);
    /* rasterise: areas, centroids, neighbours */
    static float sa[NCELL], sx[NCELL], sy[NCELL];
    int (*nb)[16] = fo_nb, *nnb = fo_nnb;
    for (int i = 0; i < fo.n; i++) { fo.c[i].px = fo.c[i].x; fo.c[i].py = fo.c[i].y; fo.c[i].pw = fo.c[i].w; }
    memset(sa, 0, sizeof sa); memset(sx, 0, sizeof sx); memset(sy, 0, sizeof sy); memset(fo_nnb, 0, sizeof fo_nnb);
    float st = FST;
    int gw = (int)((fo.x1 - fo.x0) / st), gh = (int)((fo.y1 - fo.y0) / st);
    if (gw > 1024) gw = 1024;
    fo_gw = gw; fo_gh = gh;
    fo_own = realloc(fo_own, sizeof(float) * (size_t)gw * gh);
    static int row[2][1024];
    for (int gy = 0; gy < gh; gy++) {
        int *cur = row[gy & 1], *prev = row[(gy & 1) ^ 1];
        for (int gx = 0; gx < gw && gx < 1024; gx++) {
            float x = fo.x0 + (gx + 0.5f) * st, y = fo.y0 + (gy + 0.5f) * st;
            int o = foam_owner(x, y);
            cur[gx] = o;
            fo_own[gy * gw + gx] = (float)o;
            if (o < 0) continue;
            sa[o] += 1; sx[o] += x; sy[o] += y;
            int others[2] = {gx ? cur[gx - 1] : -1, gy ? prev[gx] : -1};
            for (int k = 0; k < 2; k++) {
                int q = others[k];
                if (q < 0 || q == o) continue;
                int seen = 0;
                for (int m = 0; m < nnb[o]; m++) seen |= nb[o][m] == q;
                if (!seen && nnb[o] < 16) nb[o][nnb[o]++] = q;
                seen = 0;
                for (int m = 0; m < nnb[q]; m++) seen |= nb[q][m] == o;
                if (!seen && nnb[q] < 16) nb[q][nnb[q]++] = o;
            }
        }
    }
    float mean = 0;
    int live = 0;
    for (int i = 0; i < fo.n; i++) {
        Cell *c = &fo.c[i];
        if (!c->on) continue;
        c->nn = nnb[i];
        c->area = sa[i] * st * st;
        if (sa[i] > 0) { c->cx = sx[i] / sa[i]; c->cy = sy[i] / sa[i]; } else { c->cx = c->x; c->cy = c->y; }
        mean += c->target; live++;
    }
    mean /= live ? live : 1;
    for (int i = 0; i < fo.n; i++) {
        Cell *c = &fo.c[i];
        if (!c->on) continue;
        /* Lloyd, and the weight after the target area */
        float k = prewarm ? 0.6f : 0.25f;
        c->x += (c->cx - c->x) * k; c->y += (c->cy - c->y) * k;
        c->x = clampf(c->x, fo.x0, fo.x1); c->y = clampf(c->y, fo.y0, fo.y1);
        int dying = (c->partner >= 0 && t > c->rup_t + RUPD) || (c->burst_t >= 0 && t > c->burst_t + BURSTD);
        if (dying) c->w -= dt * (c->partner >= 0 ? 0.004f : 0.0025f) * (mean / 0.009f);
        else {
            c->w += (c->target - c->area) / PI * (prewarm ? 0.7f : 0.35f);
            if (c->w < 1e-5f) c->w = 1e-5f;
        }
        /* von Neumann */
        if (!prewarm && c->partner < 0 && c->burst_t < 0) c->target += 0.00006f * (c->nn - 6) * dt * (mean / 0.009f);
        /* merged away, burst, or shrunk to nothing */
        if (dying) c->target *= expf(-dt / 0.6f);
        if (dying && (c->area < mean * 0.03f || c->w < -mean * 2.0f)) {
            c->on = 0;
            for (int j = 0; j < fo.n; j++) if (fo.c[j].partner == i) fo.c[j].partner = -1;
        }
    }
    /* a film breaks between two neighbours: they become one */
    if (!prewarm && t >= fo.next_rup) {
        for (int tries = 0; tries < 20; tries++) {
            int i = irange(0, fo.n - 1);
            Cell *c = &fo.c[i];
            if (!c->on || c->partner >= 0 || c->burst_t >= 0 || !nnb[i]) continue;
            int j = nb[i][irange(0, nnb[i] - 1)];
            Cell *d = &fo.c[j];
            if (!d->on || d->partner >= 0 || d->burst_t >= 0) continue;
            int busy = 0;
            for (int q = 0; q < fo.n; q++) busy |= fo.c[q].on && (fo.c[q].partner == i || fo.c[q].partner == j);
            if (busy) continue;
            /* the smaller is absorbed by the larger */
            if (c->target > d->target) { Cell *s = c; c = d; d = s; int x = i; i = j; j = x; }
            c->partner = j; c->rup_t = t;
            d->target += c->target;
            break;
        }
        fo.next_rup = t + frange(1.5f, 4.0f) / fmaxf(wallA / 1.6f, 0.6f);
    }
    /* a cap bursts */
    if (!prewarm && t >= fo.next_burst) {
        for (int tries = 0; tries < 20; tries++) {
            int i = irange(0, fo.n - 1);
            Cell *c = &fo.c[i];
            if (!c->on || c->partner >= 0 || c->burst_t >= 0) continue;
            if (c->x < eye_x - wallA * 0.5f || c->x > eye_x + wallA * 0.5f) continue;
            int busy = 0;
            for (int q = 0; q < fo.n; q++) busy |= fo.c[q].on && fo.c[q].partner == i;
            if (busy) continue;
            c->burst_t = t;
            /* its share goes to the neighbours */
            for (int k = 0; k < nnb[i]; k++) fo.c[nb[i][k]].target += c->target / nnb[i];
            break;
        }
        fo.next_burst = t + frange(3.0f, 7.0f) / fmaxf(wallA / 1.6f, 0.6f);
    }
    fo.rng = rng;
    fo_own_dirty = 1;
    if (!prewarm) fo.steps++;
}

static GLuint tex_cells, tex_fbin;
static float *cell_buf;

static void foam_upload(float t, float frac) {
    if (!cell_buf) cell_buf = malloc(sizeof(float) * NCELL * 28);
    memset(cell_buf, 0, sizeof(float) * NCELL * 28);
    /* between steps the cells glide from where they were to where they're going */
    for (int i = 0; i < fo.n; i++) {
        Cell *c = &fo.c[i];
        float *o = cell_buf + i * 28;
        for (int k = 0; k < 16; k++) o[12 + k] = k < fo_nnb[i] ? (float)fo_nb[i][k] : -1;
        if (!c->on) { o[2] = -1; o[3] = -1; continue; }
        o[0] = c->px + (c->x - c->px) * frac; o[1] = c->py + (c->y - c->py) * frac; o[2] = c->pw + (c->w - c->pw) * frac;
        o[3] = (float)c->partner;
        o[4] = c->partner >= 0 ? (t - c->rup_t) / RUPD : -1;
        o[5] = c->burst_t >= 0 ? (t - c->burst_t) / BURSTD : -1;
        o[6] = fmaxf(c->h0 * expf(-(t - c->born) / 500.0f), 0.13f);
        o[7] = c->seed;
        o[8] = sqrtf(fmaxf(c->area, 1e-5f) / PI);
    }
    glBindTexture(GL_TEXTURE_2D, tex_cells);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 7, NCELL, GL_RGBA, GL_FLOAT, cell_buf);
    if (fo_own_dirty) {
        glBindTexture(GL_TEXTURE_2D, tex_fbin);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, fo_gw, fo_gh, 0, GL_RED, GL_FLOAT, fo_own);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        fo_own_dirty = 0;
    }
}

static const char *FS_FOAM = GLSL(
    uniform sampler2D uCells, uBins;
    uniform vec4 uDom;          /* domain x0, y0, bin size, bins across */
    uniform vec2 uBinN;
    uniform float uDepth0, uTilt;
    vec4 cl(int i, int k) { return texelFetch(uCells, ivec2(k, i), 0); }

    void main() {
        vec2 P = wallPos();
        float px = uMap.z;
        vec3 rd = normalize(vec3(P, 0.0) - uEye);
        ivec2 b = clamp(ivec2(floor((P - uDom.xy) / uDom.z)), ivec2(0), ivec2(uBinN) - 1);
        int o0 = int(texelFetch(uBins, b, 0).r);
        int best = -1;
        float bd = 1e9;
        vec4 ci = vec4(0.0);
        /* the cell under the raster sample, and its neighbours: who owns P */
        for (int k = 0; k < 17; k++) {
            if (o0 < 0) break;
            int id = k == 0 ? o0 : int(cl(o0, 3 + (k - 1) / 4)[(k - 1) & 3]);
            if (id < 0) break;
            vec4 c = cl(id, 0);
            float d = dot(P - c.xy, P - c.xy) - c.z;
            if (d < bd) { bd = d; best = id; ci = c; }
        }
        vec3 col = backdrop(P);
        if (best < 0) { oCol = vec4(col * uFade, 1.0); oDep = vec2(uDepth0, 1.0); return; }
        int me = best;
        vec4 c0 = ci, c1 = cl(me, 1), c2 = cl(me, 2);
        /* the nearest two borders, among its neighbours */
        float e1 = 1e9, e2 = 1e9;
        vec2 dir1 = vec2(1.0, 0.0);
        float gapHalf = -1.0;
        for (int k = 0; k < 16; k++) {
            int jd = int(cl(me, 3 + k / 4)[k & 3]);
            if (jd < 0) break;
            vec4 cj = cl(jd, 0);
            vec2 dv = cj.xy - ci.xy;
            float dl = length(dv);
            if (dl < 1e-5) continue;
            float pj = dot(P - cj.xy, P - cj.xy) - cj.z;
            float e = (pj - bd) / (2.0 * dl);
            /* a ruptured film between these two: its border retracting */
            float gh = -1.0;
            if (int(c0.w) == jd && c1.x >= 0.0) gh = c1.x;
            else if (int(cj.w) == me) gh = cl(jd, 1).x;
            if (gh >= 0.0) {
                if (gh >= 1.0) continue;
                /* along the border from its middle */
                vec2 u = dv / dl;
                vec2 m = ci.xy + u * (dl * 0.5 + (ci.z - cj.z) / (2.0 * dl));
                float along = abs(dot(P - m, vec2(-u.y, u.x)));
                float L = gh * gh * 0.16;
                if (along < L) continue;
                gapHalf = max(gapHalf, exp(-(along - L) / 0.004));
            }
            if (e < e1) { e2 = e1; e1 = e; dir1 = dv / dl; }
            else if (e < e2) e2 = e;
        }
        /* which film: a cell absorbed in a rupture shows its new owner's */
        vec4 film = c1;
        vec3 cc = ci.xyz;
        float rr = c2.x;
        if (c0.w >= 0.0 && c1.x >= 1.0) { int o = int(c0.w); film = cl(o, 1); cc = cl(o, 0).xyz; rr = cl(o, 2).x; }

        float depth = uDepth0 + uTilt * (P.y - uEye.y);
        float wb = 0.0026 + 0.035 * rr;
        float node = exp(-(e2 - e1) / (wb * 1.6));
        float wid = wb * (1.0 + 1.6 * node);

        /* the face: a shallow dome */
        vec2 d = P - cc.xy;
        float Rc = rr * 1.45;
        vec3 nrm = normalize(vec3(d, -sqrt(max(Rc * Rc - dot(d, d), Rc * Rc * 0.05))));
        float h;
        {
            float sn = sin(film.w), cs = cos(film.w);
            vec2 q = mat2(cs, sn, -sn, cs) * d / max(rr, 0.01);
            vec3 s = vec3(q * 0.55, film.w * 0.37 + uTime * 0.004);
            vec3 w = texture(uN3, s * 0.9 + vec3(0.0, 0.0, uTime * 0.006)).rgb - 0.5;
            float nz = texture(uN3, (s + 1.4 * w) * 0.8 + film.w).g;
            h = film.z * (0.6 + 0.8 * nz);
            h += 1.4 * exp(-max(e1, 0.0) / 0.006);       /* thicker against the border */
        }
        vec3 R = tfilm(h, abs(dot(rd, nrm)));
        vec3 L = env(reflect(rd, nrm));
        /* under the film: the foam going down, dim */
        vec3 under = backdrop(P) * 0.6 + vec3(0.010, 0.008, 0.007) * texture(uNoise, P * 1.7).r;
        vec3 face = under * (1.0 - R) + R * L;
        /* a burst cap: the film gone inside a ragged hole */
        if (c1.y >= 0.0) {
            vec2 hc = c0.xy + vec2(0.3, 0.2) * rr;
            float ang = atan(P.y - hc.y, P.x - hc.x) / 6.2831853;
            float hr = min(c1.y, 1.5) * rr * 1.6 * (1.0 + 0.12 * (texture(uNoise, vec2(ang, film.w)).r - 0.5));
            float into = hr - length(P - hc);
            face = mix(face, under, clamp(into / px + 0.5, 0.0, 1.0));
            if (c1.y < 1.4) face += exp(-pow((into + 0.002) / 0.0022, 2.0)) * (L * 0.1 + 0.04);
        }
        col = face;
        if (gapHalf > 0.0) col += gapHalf * (L * 0.12 + 0.03);

        /* the Plateau border: a glassy channel of liquid */
        if (e1 < wid + px) {
            float s = clamp(e1 / wid, 0.0, 1.0);
            vec3 bn = normalize(vec3(-dir1 * s, -sqrt(max(1.0 - s * s, 0.0))));
            float cosb = abs(dot(rd, bn));
            float F = 0.02 + 0.98 * pow(1.0 - cosb, 5.0);
            vec3 inside = under * 0.5 + env(refract(rd, bn, 0.75)) * 0.08;
            vec3 pb = F * env(reflect(rd, bn)) + (1.0 - F) * inside;
            col = mix(col, pb, clamp((wid - e1) / px + 0.5, 0.0, 1.0));
        }
        oCol = vec4(col * uFade, 1.0);
        oDep = vec2(depth, 1.0);
    });

static GLuint p_foam;

static void render_foam(const Act *a, int ai, double lt) {
    long want = (long)(lt * FOAMHZ);
    if (fo.act != ai || fo.steps > want) {
        foam_init(ai, a->seed);
        for (int i = 0; i < 40; i++) foam_step(1);
        for (int i = 0; i < fo.n; i++) { fo.c[i].px = fo.c[i].x; fo.c[i].py = fo.c[i].y; fo.c[i].pw = fo.c[i].w; }
    }
    long cap = offline ? want : fo.steps + (want - fo.steps > 12 ? 60 : 2);
    while (fo.steps < want && fo.steps < cap) foam_step(0);
    foam_upload((float)lt, clampf((float)(lt * FOAMHZ - fo.steps) + 1, 0, 1));
    if (getenv("SOAP_DEBUG")) fprintf(stderr, "foam: %d cells, bin overflow %d\n", fo.n, fbin_over);
    fbin_over = 0;

    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo);
    glViewport(0, 0, IW, IH);
    glDisable(GL_BLEND);
    bind_scene_tex();
    GLuint p = p_foam;
    set_common(p, BUB_D);
    tex_unit(3, GL_TEXTURE_2D, tex_cells);
    tex_unit(4, GL_TEXTURE_2D, tex_fbin);
    glUniform1i(U(p, "uCells"), 3);
    glUniform1i(U(p, "uBins"), 4);
    glUniform4f(U(p, "uDom"), fo.x0, fo.y0, FST, 0);
    glUniform2f(U(p, "uBinN"), (float)fo_gw, (float)fo_gh);
    glUniform1f(U(p, "uDepth0"), BUB_D);
    glUniform1f(U(p, "uTilt"), 0.9f);
    light_frame(p, "uBoxD", "uBoxU", "uBoxV", -0.35f, 0.4f, -1.0f);
    glUniform4f(U(p, "uBox"), 0.26f, 0.34f, 0.05f, 8.0f);
    light_frame(p, "uStrD", "uStrU", "uStrV", 0.9f, 0.2f, -0.3f);
    glUniform4f(U(p, "uStr"), 0.06f, 0.8f, 0.04f, 3.0f);
    glUniform1f(U(p, "uBars"), 1.0f);
    glUniform1f(U(p, "uAmb"), 2.0f);
    glUniform1f(U(p, "uFill"), 3.2f);
    draw_fs();
    dof_focus = BUB_D;
    dof_aperture = 0.026f;
}

/* ================= caption ================= */
typedef struct { int x, y, w, h, bx, by; float adv; } Glyph;
typedef struct { Glyph g[96]; float px; } Face;
static Face f_time, f_away;
#define ATW 1024
#define ATH 512
static unsigned char *atlas;
static int sh_x = 1, sh_y = 1, sh_h, font_for_h;

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
        if (sh_x + g->w + 3 > ATW) { sh_x = 1; sh_y += sh_h + 3; sh_h = 0; }
        if (sh_y + g->h + 3 > ATH) die("glyph atlas full");
        g->x = sh_x; g->y = sh_y;
        sh_x += g->w + 3;
        if (g->h > sh_h) sh_h = g->h;
        for (int y = 0; y < g->h; y++)
            memcpy(atlas + (size_t)(g->y + y) * ATW + g->x, s->bitmap.buffer + y * s->bitmap.pitch, (size_t)g->w);
    }
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
}

static float cap_s(void) { return fminf((float)W, (float)H) / 1080.0f; }

static void fonts_build(void) {
    if (font_for_h == W * 10000 + H) return;
    font_for_h = W * 10000 + H;
    if (!atlas) atlas = malloc(ATW * ATH);
    memset(atlas, 0, ATW * ATH);
    sh_x = sh_y = 1; sh_h = 0;
    float s = cap_s();
    load_font(&f_time, "EB Garamond:style=Regular", 34 * s);
    load_font(&f_away, "EB Garamond:style=Italic", 19 * s);
    if (!tex_atlas) glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, ATW, ATH, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

static float tv[4096 * 8];
static int tvn;
static void text_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > (int)(sizeof tv / sizeof *tv)) return;
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}
static float text_w(const Face *f, const char *s, float track) {
    float w = 0;
    for (const char *p = s; *p; p++) w += f->g[(unsigned char)*p >= 32 && (unsigned char)*p < 127 ? *p - 32 : 0].adv + (p[1] ? track : 0);
    return w;
}
static void text(const Face *f, const char *s, float x, float y, float track, const float *c) {
    float pen = roundf(x);
    y = roundf(y);
    for (const char *p = s; *p; p++) {
        int i = (unsigned char)*p == 0xB7 ? 95 : (unsigned char)*p >= 32 && (unsigned char)*p < 127 ? *p - 32 : 0;
        const Glyph *g = &f->g[i];
        if (g->w) {
            float x0 = pen + g->bx, y0 = y - g->by, x1 = x0 + g->w, y1 = y0 + g->h;
            text_v(x0, y0, g->x, g->y, c); text_v(x1, y0, g->x + g->w, g->y, c); text_v(x1, y1, g->x + g->w, g->y + g->h, c);
            text_v(x0, y0, g->x, g->y, c); text_v(x1, y1, g->x + g->w, g->y + g->h, c); text_v(x0, y1, g->x, g->y + g->h, c);
        }
        pen += g->adv + track;
    }
}

static void caption_draw(GLuint target) {
    if (!opt_caption || !front) return;
    fonts_build();
    tvn = 0;
    float s = cap_s();
    time_t tt = time(NULL);
    struct tm tm;
    localtime_r(&tt, &tm);
    char hm[16], away[64] = "";
    strftime(hm, sizeof hm, "%H:%M", &tm);
    if (locked_at) {
        long a = (long)(tt - locked_at);
        if (a < 0) a = 0;
        if (a < 60) snprintf(away, sizeof away, "away a moment");
        else if (a < 3600) snprintf(away, sizeof away, "away %ld min", a / 60);
        else snprintf(away, sizeof away, "away %ld h %02ld min", a / 3600, a / 60 % 60);
    }
    /* inside the film's wire frame, clear of its corner, with a soft shadow */
    float x = roundf(0.034f * H + 54 * s), y = roundf(H - 0.034f * H - 60 * s);
    static const float ct[4] = {0.95f, 0.92f, 0.86f, 0.86f}, ca[4] = {0.9f, 0.86f, 0.8f, 0.68f};
    static const float sh[4] = {0.0f, 0.0f, 0.0f, 0.42f};
    float o = fmaxf(1, roundf(1.5f * s));
    for (int pass = 0; pass < 2; pass++) {
        float yy = y, d = pass ? 0 : o;
        if (away[0]) {
            text(&f_away, away, x + 1 + d, yy + d, 0.6f * s, pass ? ca : sh);
            yy -= roundf(36 * s);
        }
        text(&f_time, hm, x + d, yy + d, 2.0f * s, pass ? ct : sh);
    }
    (void)text_w;
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(p_text);
    glUniform2f(U(p_text, "uRes"), (float)W, (float)H);
    glUniform2f(U(p_text, "uAtlas"), ATW, ATH);
    glUniform1i(U(p_text, "uAt"), 0);
    tex_unit(0, GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    glDisable(GL_BLEND);
}

/* ================= frames ================= */
static void gl_init(void) {
    glGenVertexArrays(1, &vao);
    float *lut = lut_build();
    glGenTextures(1, &tex_lut);
    glBindTexture(GL_TEXTURE_2D, tex_lut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, LUT_N, LUT_C, 0, GL_RGBA, GL_FLOAT, lut);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    free(lut);
    noise2_init();
    noise3_init();

    p_curl = fs_prog(SIM_HEAD, SIM_CURL);
    p_vel = fs_prog(SIM_HEAD, SIM_VEL);
    p_div = fs_prog(SIM_HEAD, SIM_DIV);
    p_jac = fs_prog(SIM_HEAD, SIM_JACOBI);
    p_proj = fs_prog(SIM_HEAD, SIM_PROJECT);
    p_hadv = fs_prog(SIM_HEAD, SIM_HADV);
    p_hnew = fs_prog(SIM_HEAD, SIM_HNEW);
    p_init = fs_prog(SIM_HEAD, SIM_INIT);
    p_film = fs_prog(FS_COMMON, FS_FILM);
    p_back = fs_prog(FS_COMMON, FS_BACK);
    p_down = fs_prog("", FS_DOWN);
    p_dof = fs_prog("", FS_DOF);
    p_final = fs_prog("", FS_FINAL);
    p_bub = link_prog(compile(GL_VERTEX_SHADER, "#version 330 core\n", VS_BUB, NULL),
                      compile(GL_FRAGMENT_SHADER, "#version 330 core\n", FS_COMMON, FS_BUB));
    p_mist = link_prog(compile(GL_VERTEX_SHADER, "#version 330 core\n", VS_MIST, NULL),
                       compile(GL_FRAGMENT_SHADER, "#version 330 core\n", FS_COMMON, FS_MIST));
    p_text = link_prog(compile(GL_VERTEX_SHADER, VS_TEXT, NULL, NULL), compile(GL_FRAGMENT_SHADER, FS_TEXT, NULL, NULL));

    p_foam = fs_prog(FS_COMMON, FS_FOAM);
    glGenTextures(1, &tex_cells);
    glBindTexture(GL_TEXTURE_2D, tex_cells);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 7, NCELL, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &tex_fbin);
    glGenTextures(1, &tex_bub);
    glBindTexture(GL_TEXTURE_2D, tex_bub);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 8, NB, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenVertexArrays(1, &vao_mist);
    glGenBuffers(1, &vbo_mist);
    glBindVertexArray(vao_mist);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mist);
    glBufferData(GL_ARRAY_BUFFER, sizeof mist_buf, NULL, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribDivisor(0, 1);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    glVertexAttribDivisor(1, 1);

    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    glBindVertexArray(vao);
}

static void bind_scene_tex(void) {
    tex_unit(0, GL_TEXTURE_2D, tex_lut);
    tex_unit(1, GL_TEXTURE_2D, tex_noise);
    tex_unit(2, GL_TEXTURE_3D, tex_noise3);
}


/* the film act at act-local time lt */
static void render_film(const Act *a, int ai, double lt) {
    int f = 0;
    while (f + 1 < NFILM && lt >= a->fstart[f + 1]) f++;
    double tau = lt - a->fstart[f] - LEAD;     /* the film's age; < 0 before it's drawn */
    float life = (float)a->flife[f];
    uint32_t fseed = mix32(a->seed ^ (uint32_t)(f + 1) * 0x51ed27U);
    float cx, cy, hw, hh;
    film_frame(&cx, &cy, &hw, &hh);
    int on = tau >= 0 && tau < life + POPD;
    if (on) {
        double ts = fmin(tau, life + 0.2);
        long want = (long)(ts * SIMHZ);
        film_alloc();
        if (fm.act != ai || fm.film != f || fm.steps > want) film_init(ai, f, fseed);
        long cap = offline ? want : fm.steps + (want - fm.steps > 40 ? 90 : 4);
        while (fm.steps < want && fm.steps < cap) film_step(life);
    }
    float pop[4] = {0, 0, -1, 0};
    rng = fseed;
    float popx = cx + frange(-0.6f, 0.6f) * hw, popy = cy + hh * frange(0.62f, 0.9f);
    float vp = 0;
    {
        float dx = fmaxf(fabsf(popx - (cx - hw)), fabsf(popx - (cx + hw))), dy = fmaxf(popy - (cy - hh), cy + hh - popy);
        vp = sqrtf(dx * dx + dy * dy) / 0.85f;
    }
    if (on && tau >= life) {
        float e = (float)(tau - life);
        pop[0] = popx; pop[1] = popy; pop[2] = vp * e; pop[3] = (fseed % 997) / 997.0f;
        if (pop[2] > 0.0f) mist_film(popx, popy, vp, e, fseed, cx, cy, hw, hh);
        if (e > 0.9f) on = 0;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo);
    glViewport(0, 0, IW, IH);
    glDisable(GL_BLEND);
    bind_scene_tex();
    tex_unit(3, GL_TEXTURE_2D, fm.h[fm.ch].tex);
    set_common(p_film, BUB_D);
    GLuint p = p_film;
    glUniform1i(U(p, "uH"), 3);
    glUniform2f(U(p, "uHRes"), (float)fm.hx, (float)fm.hy);
    glUniform4f(U(p, "uFrame"), cx, cy, hw, hh);
    glUniform1f(U(p, "uFrameR"), 0.05f);
    glUniform1f(U(p, "uWireR"), 0.0042f);
    glUniform1f(U(p, "uOn"), on ? 1.0f : 0.0f);
    float wipe = (float)(cy - hh + 2 * hh * smooth01((float)(tau / DRAW)) * 1.02f);
    glUniform1f(U(p, "uWipe"), wipe);
    glUniform4fv(U(p, "uPop"), 1, pop);
    glUniform1f(U(p, "uDepth0"), BUB_D);
    glUniform1f(U(p, "uTilt"), 0.35f);
    {
        /* a softbox behind the camera, aimed at the film's middle and big
         * enough that the whole wall reflects it, falling off to the edges */
        float dx = (cx - eye_x) / BUB_D, dy = (cy - eye_y) / BUB_D;
        float sx = hw / BUB_D, sy = hh / BUB_D;
        light_frame(p, "uBoxD", "uBoxU", "uBoxV", dx - 0.12f * sx, dy + 0.06f * sy, -1.0f);
        glUniform4f(U(p, "uBox"), 0.8f * sx, 0.75f * sy, 0.5f * fmaxf(sx, sy), 7.5f);
    }
    light_frame(p, "uStrD", "uStrU", "uStrV", 0.9f, 0.1f, -0.4f);
    glUniform4f(U(p, "uStr"), 0.08f, 0.7f, 0.06f, 3.0f);
    glUniform1f(U(p, "uBars"), 0.0f);
    glUniform1f(U(p, "uAmb"), 1.0f);
    glUniform1f(U(p, "uFill"), 0.0f);
    draw_fs();
    dof_focus = BUB_D;
    dof_aperture = 0.004f;
}

static void render_bubbles(const Act *a, int ai, double lt) {
    long want = (long)(lt * SIMHZ);
    if (bs.act != ai || bs.steps > want) bubbles_init(ai, a->seed);
    long cap = offline ? want : bs.steps + (want - bs.steps > 40 ? 400 : 4);
    while (bs.steps < want && bs.steps < cap) bubbles_step();
    float t = (float)bs.steps / SIMHZ;
    bubbles_upload(t);
    mist_bubbles(t);

    glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo);
    glViewport(0, 0, IW, IH);
    glDisable(GL_BLEND);
    bind_scene_tex();
    set_common(p_back, BUB_D);
    glUniform1f(U(p_back, "uFar"), BUB_D + 6.0f);
    draw_fs();

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    GLuint p = p_bub;
    set_common(p, BUB_D);
    tex_unit(3, GL_TEXTURE_2D, tex_bub);
    glUniform1i(U(p, "uB"), 3);
    glUniform4fv(U(p, "uOut"), 1, out_w0);
    light_frame(p, "uBoxD", "uBoxU", "uBoxV", -0.55f, 0.5f, -1.0f);
    glUniform4f(U(p, "uBox"), 0.5f, 0.62f, 0.07f, 11.0f);
    light_frame(p, "uStrD", "uStrU", "uStrV", 1.0f, 0.12f, 0.45f);
    glUniform4f(U(p, "uStr"), 0.05f, 0.9f, 0.04f, 6.0f);
    glUniform1f(U(p, "uBars"), 1.0f);
    glUniform1f(U(p, "uAmb"), 2.5f);
    glUniform1f(U(p, "uFill"), 1.6f);
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, NB);
    glDisable(GL_BLEND);
    dof_focus = BUB_D + 0.45f;
    dof_aperture = 0.03f;
}

/* GPU time per stage, for --bench with SOAP_PROF set */
static GLuint tq[4];
static double tq_sum[4];
static int tq_on = -1, tq_n;
static void tq_mark(int i) {
    if (tq_on < 0) tq_on = getenv("SOAP_PROF") != NULL;
    if (!tq_on) return;
    if (!tq[0]) glGenQueries(4, tq);
    if (i) glEndQuery(GL_TIME_ELAPSED);
    if (i < 3) glBeginQuery(GL_TIME_ELAPSED, tq[i]);
    if (i == 3) {
        for (int k = 0; k < 3; k++) { GLuint64 ns = 0; glGetQueryObjectui64v(tq[k], GL_QUERY_RESULT, &ns); tq_sum[k] += ns * 1e-6; }
        tq_n++;
    }
}

static void render(double t, GLuint target) {
    tq_mark(0);
    view_update();
    double lt;
    int ai = act_at(t, &lt);
    const Act *a = act_get(ai);
    /* about a megapixel inside; the foam, mostly out of focus, less */
    scene_targets(a->scene == S_FOAM ? 0.7e6f : 1.0e6f);
    cur_time = (float)fmod(t, 4000.0);
    cur_fade = (float)(smooth01((float)(lt / ACT_FADE)) * smooth01((float)((a->dur - lt) / ACT_FADE)));
    mist_n = 0;
    switch (a->scene) {
    case S_FILM: render_film(a, ai, lt); break;
    case S_BUBBLES: render_bubbles(a, ai, lt); break;
    default: render_foam(a, ai, lt); break;
    }
    /* mist over whichever scene */
    if (mist_n) {
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo);
        glViewport(0, 0, IW, IH);
        set_common(p_mist, BUB_D);
        glUniform4fv(U(p_mist, "uOut"), 1, out_w0);
        glUniform1f(U(p_mist, "uPx"), out_w0[3] / IH);
        mist_draw();
    }

    tq_mark(1);
    /* depth of field */
    static int nodof = -1;
    if (nodof < 0) nodof = getenv("SOAP_NODOF") != NULL;
    float pxw = out_w0[3] / IH * 3.0f;        /* wall units per DOF pixel (a third of the scene's) */
    float maxc = 7.0f;
    rt_bind(&half_rt);
    glUseProgram(p_down);
    tex_unit(0, GL_TEXTURE_2D, scene_col);
    tex_unit(1, GL_TEXTURE_2D, scene_dep);
    glUniform1i(U(p_down, "uCol"), 0);
    glUniform1i(U(p_down, "uDep"), 1);
    glUniform2f(U(p_down, "uSrc"), 1.0f / IW, 1.0f / IH);
    glUniform1f(U(p_down, "uFocus"), dof_focus);
    glUniform1f(U(p_down, "uAperture"), dof_aperture);
    glUniform1f(U(p_down, "uPxW"), pxw);
    glUniform1f(U(p_down, "uMaxC"), maxc);
    draw_fs();
    if (!nodof) {
    glBindTexture(GL_TEXTURE_2D, half_rt.tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    rt_bind(&dof_rt);
    glUseProgram(p_dof);
    tex_unit(0, GL_TEXTURE_2D, half_rt.tex);
    glUniform1i(U(p_dof, "uHalf"), 0);
    glUniform2f(U(p_dof, "uTx"), 1.0f / half_rt.w, 1.0f / half_rt.h);
    glUniform1f(U(p_dof, "uMaxC"), maxc);
    draw_fs();
    glBindTexture(GL_TEXTURE_2D, dof_rt.tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    }

    tq_mark(2);
    /* the print */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_final);
    tex_unit(0, GL_TEXTURE_2D, scene_col);
    tex_unit(1, GL_TEXTURE_2D, dof_rt.tex);
    tex_unit(2, GL_TEXTURE_2D, tex_noise);
    glUniform1i(U(p_final, "uCol"), 0);
    glUniform1i(U(p_final, "uDofT"), 1);
    glUniform1i(U(p_final, "uNoise"), 2);
    glUniform2f(U(p_final, "uRes"), (float)W, (float)H);
    glUniform4fv(U(p_final, "uOutN"), 1, out_w0);
    glUniform2f(U(p_final, "uWallC"), eye_x, eye_y);
    glUniform1f(U(p_final, "uWallR"), sqrtf(wallA * wallA * 0.25f + 0.25f));
    glUniform1f(U(p_final, "uExpo"), 1.0f);
    glUniform1f(U(p_final, "uGrainT"), (float)fmod(t * 24.0, 997.0));
    draw_fs();
    caption_draw(target);
    tq_mark(3);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "soapfilm");
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

/* spanning: one wall, one eye over the front output (the one under the
 * middle of the layout); every output is a window onto it */
static void span_setup(void) {
    if (span_me < 0) return;
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    wallr[0] = x0; wallr[1] = y0; wallr[2] = x1 - x0; wallr[3] = y1 - y0;
    memcpy(outr, span[span_me], sizeof outr);
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    front = span_me == mid;
    float wh = wallr[3];
    eye_x = (span[mid][0] + span[mid][2] / 2 - x0) / wh;
    eye_y = 1 - (span[mid][1] + span[mid][3] / 2 - y0) / wh;
}

static void usage(void) {
    fprintf(stderr,
        "usage: soapfilm [--window | --root | --layer] [--seed N] [--scene film|bubbles|foam]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--shot FILE.ppm | --bench N] [--size WxH] [--at SECONDS]\n"
        "                [--scale F] [--no-caption]\n");
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

static GLuint offscreen(void) {
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-scene") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NSCENE; k++) if (!strcmp(n, SCENE_NAME[k])) opt_scene = k;
            if (opt_scene < 0) usage();
        }
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-lutstrip") && i + 1 < argc) opt_lutstrip = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else usage();
    }
    if (opt_lutstrip) { lut_strip(opt_lutstrip); return 0; }
    if (span_me >= span_n) span_me = -1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (!opt_seed) opt_seed = 1;
    span_setup();
    locked_at = lock_start();
    if (getenv("SOAP_LOCKED_AGO")) locked_at = time(NULL) - atol(getenv("SOAP_LOCKED_AGO"));   /* for testing the caption */
    offline = opt_shot || opt_bench > 0;

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offline) wl_init();
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
        if (offline) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Soapfilm");
        XClassHint chint = {"soapfilm", "Soapfilm"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offline) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (getenv("SOAP_DEBUG"))
        for (int i = 0; i < 4; i++) {
            const Act *a = act_get(i);
            fprintf(stderr, "act %d: %s at %.1f for %.1f", i, SCENE_NAME[a->scene], a->start, a->dur);
            if (a->scene == S_FILM)
                for (int f = 0; f < NFILM; f++) fprintf(stderr, " [film %d drawn %.1f, pops %.1f]", f, a->start + a->fstart[f] + LEAD, a->start + a->fstart[f] + LEAD + a->flife[f]);
            fprintf(stderr, "\n");
        }
    if (opt_shot) {
        GLuint fbo = offscreen();
        render(opt_at, fbo);
        glFinish();
        if (getenv("SOAP_DEBUG")) fprintf(stderr, "gl error %x, fbo %x\n", glGetError(), glCheckFramebufferStatus(GL_FRAMEBUFFER));
        save_ppm(opt_shot);
        return 0;
    }
    if (opt_bench > 0) {
        GLuint fbo = offscreen();
        for (int i = 5; i > 0; i--) render(opt_at - i / 60.0, fbo);   /* warm up, and catch the sims up */
        glFinish();
        double sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double t0 = now();
            render(opt_at + i / 60.0, fbo);
            glFinish();
            double ms = (now() - t0) * 1e3;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        if (tq_n) printf("gpu: scene %.3f ms, dof %.3f ms, print %.3f ms\n", tq_sum[0] / tq_n, tq_sum[1] / tq_n, tq_sum[2] / tq_n);
        printf("soapfilm: %d frames at %dx%d (internal %dx%d) from %.1f s: avg %.3f ms, worst %.3f ms\nrenderer: %s\n",
               opt_bench, W, H, IW, IH, opt_at, sum / opt_bench, worst, (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    if (span_me >= 0 && opt_epoch) t0 -= wallclock() - opt_epoch;   /* the clock runs from the shared epoch */
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
        render(t - t0, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
