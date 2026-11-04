/*
 * papercut — a cut-paper shadow box as a lock screen: eight sheets of
 * coloured card stacked a few millimetres apart over a base sheet, each cut
 * with a scalpel along a slowly changing line, so you look down through the
 * holes into the layers below and every cut edge throws a soft shadow onto
 * the sheets beneath it.
 *
 * The cuts come from a handful of scenes, each held for a minute and a half
 * and then re-cut, front sheets first, into the next:
 *
 *   tunnel     nested portals, each sheet's hole a little smaller than the
 *              one in front of it, their rims wobbling and turning slowly
 *   hills      a layered landscape: far ridges sharp and pale, near ones
 *              rounded and dark, drifting past with parallax; a paper sun
 *   islands    stacked topography: the sheets are contour levels of a
 *              slowly drifting height field
 *   bloom      layered paper flowers, alternate sheets' petals offset and
 *              turning against each other
 *
 * Every few minutes the time (or the host name) is cut into the stack: its
 * letters rise through the sheets one at a time, back to front, each sheet's
 * letters a little bolder than the one above, so the figures stand up out of
 * the box in stepped outlines; then they sink back.
 *
 * What makes it read as paper: the sheets are lit, not flat-filled; a baked
 * fibre texture embossed by the light gives each sheet its tooth and a faint
 * mottle; cut edges carry a scalpel's tiny wobble, a bright rim where they
 * face the light and a dark one where they don't; and the shadows are cast
 * from each sheet above at its own height, so they widen and soften with the
 * depth they fall through, plus a contact shadow where sheets nearly touch.
 *
 * Rendering: a field pass at half resolution writes, for each of the eight
 * cut sheets, a signed distance (in layout pixels) to its cut line into two
 * RGBA16F targets; the composite pass reads them, walks the sheets front to
 * back and shades the one(s) on top. Light work for the CPU.
 *
 * Palettes (--palette, else picked from the seed): paper (white on white,
 * all shadow), dusk, sea, forest, kraft, ember, brass.
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch (--schedule prints the scenes). Spanning (like voronoi): run one
 * instance per monitor with the same --seed, --epoch and --layout, and
 * --output = this one's index; the stack lives in wall coordinates, so it
 * is one continuous sheet across the bezels. The "front" output (under the
 * middle of the layout) sets the scale and carries the caption and the time.
 *
 * Same plumbing as voronoi: Xlib + GLX for --window / --root, a native
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
#include "../common/hidpi.h"

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_caption = 1, opt_schedule;
static double opt_fps = 30;
static const char *opt_font = "Inter";
static const char *opt_palette;
static const char *opt_scene;
static const char *opt_text;           /* --text: what the host-name cuts spell instead */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the stack starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "papercut: %s\n", what); exit(1); }

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
static double rr(Rng *r, double a, double b) { return a + (b - a) * rnd(r); }

static double lat(int x, int y, uint32_t salt) {
    uint64_t h = mix64(((uint64_t)(uint32_t)x << 32 | (uint32_t)y) ^ ((uint64_t)salt << 17) ^ 0x5eedull);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
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
 * under the middle of the layout, and its height L sets the scale. */
static double bx0, by0, bx1, by1, bw, bh;
static double fr[4];                   /* front output: x, y, w, h */
static double L;                       /* front output height: the unit of size */

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
    bw = bx1 - bx0; bh = by1 - by0;
    /* the scale: the front screen's shorter side, so a portrait front
     * output doesn't get giant shapes */
    L = fmin(fr[2], fr[3]);
}

/* the outputs to place things on: the layout, or the one screen */
static int n_outs(void) { return span_me >= 0 ? span_n : 1; }
static void out_rect(int i, double r[4]) {
    if (span_me >= 0) for (int k = 0; k < 4; k++) r[k] = span[i][k];
    else for (int k = 0; k < 4; k++) r[k] = fr[k];
}
/* a random point well inside some output (weighted by area) */
static void visible_point(Rng *r, double *x, double *y, double inset) {
    double o[4];
    out_rect(0, o);
    if (n_outs() > 1) {
        double tot = 0, pick;
        for (int i = 0; i < span_n; i++) tot += (double)span[i][2] * span[i][3];
        pick = rnd(r) * tot;
        for (int i = 0; i < span_n; i++) {
            pick -= (double)span[i][2] * span[i][3];
            if (pick <= 0 || i == span_n - 1) { out_rect(i, o); break; }
        }
    }
    *x = o[0] + o[2] * (inset + (1 - 2 * inset) * rnd(r));
    *y = o[1] + o[3] * (inset + (1 - 2 * inset) * rnd(r));
}

/* ==== the schedule ==== */
#define NLAYER 8                       /* cut sheets over the base */
#define ACT 90.0                       /* seconds a scene is held, including its re-cut */
#define MORPH 20.0                     /* the last MORPH seconds re-cut into the next scene */
enum { S_TUNNEL, S_HILLS, S_ISLANDS, S_BLOOM, NSCENE };
static const char *scene_name[NSCENE] = {"tunnel", "hills", "islands", "bloom"};
static int forced_scene = -1;

/* the scenes come in shuffled rounds, never the same twice running */
static void bag(long g, int p[NSCENE]) {
    Rng r = rng_for((uint64_t)g, 5, 0);
    for (int i = 0; i < NSCENE; i++) p[i] = i;
    for (int i = NSCENE - 1; i > 0; i--) {
        int j = (int)(rnd(&r) * (i + 1));
        int t = p[i]; p[i] = p[j]; p[j] = t;
    }
}
static int act_kind(long a) {
    if (forced_scene >= 0) return forced_scene;
    if (a < 0) a = 0;
    long g = a / NSCENE;
    int p[NSCENE];
    bag(g, p);
    if (g > 0) {
        int q[NSCENE];
        bag(g - 1, q);
        if (p[0] == q[NSCENE - 1]) { int t = p[0]; p[0] = p[1]; p[1] = t; }
    }
    return p[a % NSCENE];
}

/* the cut text: every third act, the time or the host name */
enum { T_NONE, T_TIME, T_HOST };
#define TXT_IN (0.08 * ACT)             /* starts rising */
#define TXT_RISE 16.0
#define TXT_OUT (0.55 * ACT)            /* starts sinking */
#define TXT_GROW 0.011                  /* each outline step, in L */
static int act_text(long a) {
    if (a % 3) return T_NONE;
    return (((a / 3) + opt_seed) & 1) ? T_HOST : T_TIME;
}

/* a scene's parameters, as the shader reads them: 12 vec4 */
#define NPAR 12
typedef struct { int kind; float P[NPAR][4]; } Scene;

static void scene_params(long a, Scene *s) {
    memset(s, 0, sizeof *s);
    s->kind = act_kind(a);
    Rng r = rng_for((uint64_t)a, 17, (uint64_t)s->kind);
    float (*P)[4] = s->P;
    int no = n_outs();
    switch (s->kind) {
    case S_TUNNEL: {
        /* a main portal on every screen, and a satellite or two */
        int n = 0;
        for (int i = 0; i < no && n < 8; i++) {
            double o[4];
            out_rect(i, o);
            double R = 0.5 * hypot(o[2], o[3]) * rr(&r, 0.78, 0.9);
            P[n][0] = (float)(o[0] + o[2] * rr(&r, 0.4, 0.6));
            P[n][1] = (float)(o[1] + o[3] * rr(&r, 0.4, 0.6));
            P[n][2] = (float)R;
            P[n][3] = (float)(int)rr(&r, 3, 7.99);
            n++;
        }
        int ns = rnd(&r) < 0.5 ? 1 : 2;
        if (no > 1) ns += no - 1;
        for (int i = 0; i < ns && n < 8; i++) {
            double x, y;
            visible_point(&r, &x, &y, 0.12);
            P[n][0] = (float)x; P[n][1] = (float)y;
            P[n][2] = (float)(L * rr(&r, 0.22, 0.34));
            P[n][3] = (float)(int)rr(&r, 3, 6.99);
            n++;
        }
        P[10][0] = (float)n;
        P[10][1] = (float)rr(&r, 0, 100);
        P[10][2] = (float)rr(&r, 0.10, 0.16);         /* wobble */
        P[10][3] = (float)(rr(&r, 0.012, 0.02) * (rnd(&r) < 0.5 ? -1 : 1));   /* turning, rad/s */
        break;
    }
    case S_HILLS: {
        P[0][0] = (float)(fr[1] + fr[3] * rr(&r, 0.26, 0.34));   /* the far ridge */
        P[0][1] = (float)(fr[1] + fr[3] * 0.97);                 /* the near one */
        P[0][2] = (float)(L * rr(&r, 0.09, 0.13));               /* height of the hills */
        P[0][3] = (float)(L * rr(&r, 0.45, 0.65));               /* their width */
        P[1][0] = (float)rr(&r, 0, 100);
        P[1][1] = (float)(L * rr(&r, 0.006, 0.010) * (rnd(&r) < 0.5 ? -1 : 1));   /* drift, px/s */
        P[1][2] = (float)(fr[0] + fr[2] * rr(&r, 0.18, 0.82));   /* the sun */
        P[1][3] = (float)(fr[1] + fr[3] * rr(&r, 0.13, 0.22));
        P[2][0] = (float)(L * rr(&r, 0.055, 0.08));
        break;
    }
    case S_ISLANDS:
        P[0][0] = (float)(L * rr(&r, 0.55, 0.75));                /* feature size */
        P[0][1] = (float)rr(&r, 0, 16); P[0][2] = (float)rr(&r, 0, 16);
        {
            double a2 = rr(&r, 0, 2 * M_PI), v = L * rr(&r, 0.004, 0.007);
            P[1][0] = (float)(v * cos(a2)); P[1][1] = (float)(v * sin(a2));
        }
        P[1][2] = (float)rr(&r, 0.31, 0.35);                     /* the lowest contour */
        P[1][3] = (float)rr(&r, 0.66, 0.70);                     /* the highest */
        break;
    case S_BLOOM: {
        /* one big flower per screen, and smaller ones around it */
        int n = 0;
        for (int i = 0; i < no && n < 10; i++) {
            double o[4];
            out_rect(i, o);
            double m = fmin(o[2], o[3]);
            P[n][0] = (float)(o[0] + o[2] * rr(&r, 0.35, 0.65));
            P[n][1] = (float)(o[1] + o[3] * rr(&r, 0.38, 0.62));
            P[n][2] = (float)(m * rr(&r, 0.40, 0.48));
            P[n][3] = (float)(int)rr(&r, 5, 9.99);
            n++;
        }
        int want = n + (no > 1 ? 2 * no : 3 + (int)(rnd(&r) * 2));
        for (int tries = 0; n < want && n < 10 && tries < 200; tries++) {
            double x, y, R = L * rr(&r, 0.12, 0.2);
            visible_point(&r, &x, &y, -0.05);
            int ok = 1;
            for (int j = 0; j < n && ok; j++)
                if (hypot(x - P[j][0], y - P[j][1]) < 0.8 * (R + P[j][2])) ok = 0;
            if (!ok) continue;
            P[n][0] = (float)x; P[n][1] = (float)y; P[n][2] = (float)R;
            P[n][3] = (float)(int)rr(&r, 5, 12.99);
            n++;
        }
        P[10][0] = (float)n;
        P[10][1] = (float)rr(&r, 0, 100);
        P[10][2] = (float)(rr(&r, 0.010, 0.018) * (rnd(&r) < 0.5 ? -1 : 1));
        P[10][3] = (float)rr(&r, 0.5, 1.1);                      /* petal sharpness */
        break;
    }
    }
}

/* ==== the cut text ==== */
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

/* set a line of text at px into a canvas with a margin; returns it (w x h, 0..255) */
static unsigned char *set_text(const char *line, int px, int pad, int *ow, int *oh) {
    FT_Set_Pixel_Sizes(face_big, 0, (FT_UInt)px);
    int w = 0;
    for (const char *c = line; *c; c++)
        if (!FT_Load_Char(face_big, (unsigned char)*c, FT_LOAD_DEFAULT)) w += (int)(face_big->glyph->advance.x >> 6);
    int asc = (int)(face_big->size->metrics.ascender >> 6), dsc = (int)(-face_big->size->metrics.descender >> 6);
    int cw = w + 2 * pad, ch = asc + dsc + 2 * pad;
    unsigned char *cv = calloc((size_t)cw * ch, 1);
    int pen = pad, base = pad + asc;
    for (const char *c = line; *c; c++) {
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

/* squared Euclidean distance transform (Felzenszwalb & Huttenlocher), 1D */
static void edt1(const float *f, int n, float *d, int *v, float *z) {
    int k = 0;
    v[0] = 0; z[0] = -1e30f; z[1] = 1e30f;
    for (int q = 1; q < n; q++) {
        float s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        while (s <= z[k]) {
            k--;
            s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = 1e30f;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        d[q] = (q - v[k]) * (float)(q - v[k]) + f[v[k]];
    }
}
static void edt2(float *g, int w, int h) {
    int n = w > h ? w : h;
    float *f = calloc(n, sizeof(float)), *d = malloc(sizeof(float) * n), *z = malloc(sizeof(float) * (n + 1));
    int *v = malloc(sizeof(int) * n);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) f[y] = g[y * w + x];
        edt1(f, h, d, v, z);
        for (int y = 0; y < h; y++) g[y * w + x] = d[y];
    }
    for (int y = 0; y < h; y++) {
        edt1(g + (size_t)y * w, w, d, v, z);
        memcpy(g + (size_t)y * w, d, sizeof(float) * w);
    }
    free(f); free(d); free(z); free(v);
}

static GLuint tex_text;
static float txt_box[4], txt_hw;       /* where the SDF lies (world), the deepest inside */
static long txt_act = -1;

/* the text of act a as a signed distance field in world pixels (positive
 * outside the letters), over the box it is fitted to */
static void build_text(long a, int kind) {
    if (txt_act == a) return;
    txt_act = a;
    char buf[64];
    double box[4];
    text_box(kind == T_HOST, box);
    if (kind == T_TIME) {
        time_t tt = (time_t)(opt_epoch + a * ACT + TXT_IN);
        struct tm tm;
        localtime_r(&tt, &tm);
        strftime(buf, sizeof buf, "%H:%M", &tm);
    } else {
        snprintf(buf, sizeof buf, "%s", opt_text ? opt_text : host);
        for (char *c = buf; *c; c++) *c = (char)toupper((unsigned char)*c);
    }
    /* measure at 100 px, then fit the ink to the box */
    int cw, ch;
    unsigned char *cv = set_text(buf, 100, 0, &cw, &ch);
    int x0 = cw, y0 = ch, x1 = 0, y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    free(cv);
    if (x1 <= x0) { txt_hw = 0; return; }
    double fitw = fmin(box[2] * 0.50 / (x1 - x0 + 1), box[3] * 0.27 / (y1 - y0 + 1)) * 100;   /* world px per em */
    /* the canvas: one texel per world pixel, unless that gets too big */
    double tx = fmax(1.0, (x1 - x0) * fitw / 100 / 3000);
    int px = (int)(fitw / tx), pad = (int)(0.12 * L / tx) + 4;
    cv = set_text(buf, px, pad, &cw, &ch);
    x0 = cw; y0 = ch; x1 = 0; y1 = 0;
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            if (cv[y * cw + x] > 60) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    size_t n = (size_t)cw * ch;
    float *gi = malloc(sizeof(float) * n), *go = malloc(sizeof(float) * n), *sd = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) {
        int in = cv[i] >= 128;
        go[i] = in ? 0 : 1e20f;        /* to the nearest inside texel */
        gi[i] = in ? 1e20f : 0;        /* to the nearest outside one */
    }
    edt2(go, cw, ch);
    edt2(gi, cw, ch);
    float deepest = 0;
    for (size_t i = 0; i < n; i++) {
        float a2 = cv[i] / 255.0f, s;
        if (a2 > 0.02f && a2 < 0.98f) s = 0.5f - a2;
        else if (cv[i] >= 128) s = -(sqrtf(gi[i]) - 0.5f);
        else s = sqrtf(go[i]) - 0.5f;
        sd[i] = s * (float)tx;
        if (-sd[i] > deepest) deepest = -sd[i];
    }
    /* ink centre onto the box centre */
    double cx = box[0] + box[2] / 2, cyy = box[1] + box[3] / 2;
    txt_box[0] = (float)(cx - (x0 + x1 + 1) * 0.5 * tx);
    txt_box[1] = (float)(cyy - (y0 + y1 + 1) * 0.5 * tx);
    txt_box[2] = (float)(cw * tx); txt_box[3] = (float)(ch * tx);
    txt_hw = deepest;
    glBindTexture(GL_TEXTURE_2D, tex_text);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, cw, ch, 0, GL_RED, GL_FLOAT, sd);
    free(cv); free(gi); free(go); free(sd);
}

/* ==== GL ==== */
#define GLSL(...) #__VA_ARGS__

/* full-screen triangle */
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 c = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(c, 0.0, 1.0);
    });

/* the field pass: for each cut sheet, the signed distance (world px,
 * positive where the paper is) to its cut line, at half resolution */
static const char *FS_FIELD = "#version 330 core\n" GLSL(
    uniform sampler2D uNoise, uText;
    uniform vec4 uPA[12], uPB[12];
    uniform int uKA, uKB;
    uniform float uU, uT, uL, uM;
    uniform vec3 uView;
    uniform vec4 uTBox;
    uniform float uTT, uTGrow, uTOn, uTMoat;
    layout(location = 0) out vec4 oA;
    layout(location = 1) out vec4 oB;

    /* fbm with one lattice cell per unit (4 octaves), from the baked tile */
    float F(vec2 p) { return texture(uNoise, p * (1.0 / 16.0)).r; }

    float hillY(vec4 P[12], float k, float x) {
        float z = (k + 1.0) / 8.0;
        float sk = P[0].w * mix(0.45, 1.5, z);
        float u = x / sk + P[1].x + 11.3 * k + uT * P[1].y * (0.2 + z * z) / sk;
        float h = F(vec2(u, 3.7 * k + 0.5));
        /* far ridges sharp, near ones rounded */
        float rg = 1.0 - abs(2.0 * h - 1.0);
        h = mix(rg * 0.9 + 0.05, h, smoothstep(0.25, 0.75, z));
        float yk = mix(P[0].x, P[0].y, pow(z, 1.3));
        float ak = P[0].z * mix(1.5, 0.75, z);
        return yk - ak * (h - 0.5) * 2.0;
    }

    /* value noise, one lattice cell per unit, unsaturated */
    float V(vec2 p) { return texture(uNoise, p * (1.0 / 64.0)).g; }
    float islandH(vec4 P[12], vec2 p) {
        vec2 q = (p - P[1].xy * uT) / P[0].x + P[0].yz;
        return 0.56 * V(q) + 0.28 * V(q * 2.03 + 7.1) + 0.16 * V(q * 4.1 + 3.3);
    }

    void scene(int kind, vec4 P[12], vec2 p, out float f[8]) {
        if (kind == 0) {                         /* tunnel */
            for (int k = 0; k < 8; k++) f[k] = 1e4;
            int n = int(P[10].x);
            for (int i = 0; i < 8; i++) {
                if (i >= n) break;
                vec2 q = p - P[i].xy;
                float r = length(q), R = P[i].z;
                if (r > R * 1.5) { for (int k = 0; k < 8; k++) f[k] = min(f[k], r - R * 1.25); continue; }
                float a = atan(q.y, q.x);
                vec2 cs = vec2(cos(a), sin(a));
                for (int k = 0; k < 8; k++) {
                    float fk = float(k), z = (fk + 1.0) / 8.0;
                    float Rk = R * mix(0.12, 1.0, pow(z, 1.2));
                    float w = F(cs * (1.1 + 0.15 * fk) + vec2(P[10].y + 5.3 * fk + 17.0 * float(i), uT * 0.006));
                    float sg = (k & 1) == 1 ? 1.0 : -1.0;
                    float lb = sin(P[i].w * a + 1.9 * fk + P[10].y + uT * P[10].w * sg * (0.5 + 0.5 * z));
                    Rk *= 1.0 + P[10].z * (w - 0.5) * 2.0 + 0.035 * lb;
                    f[k] = min(f[k], r - Rk);
                }
            }
        } else if (kind == 1) {                  /* hills */
            for (int k = 0; k < 8; k++) {
                float fk = float(k);
                float y = hillY(P, fk, p.x), y2 = hillY(P, fk, p.x + 2.0);
                float sl = (y2 - y) * 0.5;
                f[k] = (p.y - y) / sqrt(1.0 + sl * sl);
            }
            f[0] = max(f[0], P[2].x - length(p - P[1].zw));
        } else if (kind == 2) {                  /* islands */
            float e = 2.0;
            float h = islandH(P, p), hx = islandH(P, p + vec2(e, 0.0)), hy = islandH(P, p + vec2(0.0, e));
            float gl = max(length(vec2(hx - h, hy - h)) / e, 1e-6);
            float br = 0.02 * sin(uT * 0.03);
            for (int k = 0; k < 8; k++) {
                float lv = mix(P[1].z, P[1].w, float(k) / 7.0) + br;
                f[k] = (h - lv) / gl;
            }
        } else {                                 /* bloom */
            for (int k = 0; k < 8; k++) f[k] = -1e4;
            int n = int(P[10].x);
            for (int i = 0; i < 10; i++) {
                if (i >= n) break;
                vec2 q = p - P[i].xy;
                float r = length(q), R = P[i].z;
                if (r > R * 2.5) { for (int k = 0; k < 8; k++) f[k] = max(f[k], R - r); continue; }
                float a = atan(q.y, q.x), np = P[i].w;
                float si = (i & 1) == 1 ? -1.0 : 1.0;
                for (int k = 0; k < 8; k++) {
                    float fk = float(k), z = fk / 7.0;
                    float Rk = R * mix(1.0, 0.14, pow(z, 0.85));
                    float off = (k & 1) == 1 ? 3.14159265 / np : 0.0;
                    float sg = (k & 1) == 1 ? -1.0 : 1.0;
                    float ph = P[10].y * float(i + 1) + off + uT * P[10].z * sg * si * (0.6 + 0.4 * z);
                    float s = pow(abs(cos(0.5 * np * (a - ph))), P[10].w);
                    float Rp = Rk * mix(0.55 + 0.45 * s, 1.0, smoothstep(0.75, 1.0, z));
                    f[k] = max(f[k], (Rp - r) * mix(1.0, 0.8, s));
                }
            }
        }
    }

    void main() {
        /* this texel's world position: texel (i, j) is device pixel 2i - M */
        vec2 dev = vec2(gl_FragCoord.x, gl_FragCoord.y) * 2.0 - uM;
        vec2 p = uView.xy + dev / uView.z;
        float a[8], b[8];
        scene(uKA, uPA, p, a);
        if (uU > 0.0) {
            scene(uKB, uPB, p, b);
            /* re-cut: the front sheets first, each spreading from a few
             * places, blended where the distances saturate so the new cut
             * line sweeps across the old */
            float s = 0.04 * uL;
            for (int k = 0; k < 8; k++) {
                float v = uU * 1.6 - 0.35 * float(7 - k) / 7.0 - 0.45 * F(p / (0.45 * uL) + vec2(3.1, 1.7) * float(k));
                float e = clamp(v / 0.08, 0.0, 1.0);
                e = e * e * e * (e * (e * 6.0 - 15.0) + 10.0);
                float g = mix(tanh(a[k] / s), tanh(b[k] / s), e);
                a[k] = s * atanh(clamp(g, -0.995, 0.995));
            }
        }
        if (uTOn > 0.0) {
            vec2 tu = (p - uTBox.xy) / uTBox.zw;
            if (tu.x > 0.0 && tu.y > 0.0 && tu.x < 1.0 && tu.y < 1.0) {
                float sd = texture(uText, tu).r;
                for (int k = 0; k < 8; k++) {
                    /* a window cut through every sheet around the letters,
                     * and the letters standing in it on three stepped
                     * outlines; a sheet the text has risen well past draws
                     * its outline back in */
                    float x = uTT - float(k);
                    float dil = x <= 3.0 ? x : 3.0 - 4.0 * (x - 3.0);
                    a[k] = max(min(a[k], sd - uTMoat), -sd + uTGrow * dil);
                }
            }
        }
        float lim = 0.15 * uL;
        oA = clamp(vec4(a[0], a[1], a[2], a[3]), -lim, lim);
        oB = clamp(vec4(a[4], a[5], a[6], a[7]), -lim, lim);
    });

/* the composite: walk the sheets front to back and light the ones on top */
static const char *FS_COMP = "#version 330 core\n" GLSL(
    uniform sampler2D uFA, uFB, uNoise, uPaper;
    uniform vec3 uView;
    uniform vec2 uRes, uFRes;
    uniform float uM, uL;
    uniform vec3 uCol[9];
    uniform vec2 uLd;
    uniform float uGap, uSoft0, uSoft1, uShadow, uEmb;
    uniform vec3 uLamp;
    out vec4 o;

    vec2 uvOf(vec2 w) { return ((w - uView.xy) * uView.z + uM) / (2.0 * uFRes); }
    float layerD(int j, vec2 w) {
        vec2 uv = uvOf(w);
        return j < 5 ? texture(uFA, uv)[j - 1] : texture(uFB, uv)[j - 5];
    }
    float vn(vec2 p) { return texture(uNoise, p * (1.0 / 64.0)).g; }
    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }

    vec3 shade(int k, vec2 p, float d[9], vec2 g, float px) {
        vec3 c = uCol[k];
        /* the paper: fibres embossed by the light, a faint mottle */
        vec2 pp = p + vec2(float(k) * 137.3, float(k) * 71.9);
        vec4 t0 = texture(uPaper, pp * (1.0 / 1024.0));
        float h1 = texture(uPaper, (pp - uLd * 1.2) * (1.0 / 1024.0)).r;
        float emb = (t0.r - h1);
        c *= (0.97 + 0.06 * t0.g) * (1.0 + uEmb * emb) * (0.985 + 0.03 * t0.r);
        /* shadows from every sheet above, cast from its own height */
        float lit = 1.0;
        for (int j = k + 1; j <= 8; j++) {
            float gj = float(j - k);
            float w = uSoft0 + uSoft1 * gj;
            float dj = layerD(j, p - uLd * uGap * gj);
            float s = smoothstep(-w, w * 0.5, dj);
            lit *= 1.0 - uShadow * s / (1.0 + 0.22 * (gj - 1.0));
            /* contact: the sheet just above nearly touches near its edge */
            lit *= 1.0 - 0.17 * exp(min(d[j], 0.0) / (0.7 * w)) / (1.0 + 0.6 * (gj - 1.0));
        }
        /* the cut edge: a bright rim facing the light, dark away from it */
        if (k > 0) {
            float e = 1.0 - smoothstep(0.2 * px, 2.0 * px, d[k]);
            float lg = dot(g / max(length(g), 1e-6), uLd);
            c += e * max(lg, 0.0) * 0.12 * (0.6 + 0.4 * c);
            c *= 1.0 - e * max(-lg, 0.0) * 0.25;
        }
        /* the lamp: a broad fall-off from the upper left */
        float lamp = 1.04 - uLamp.z * length((p - uLamp.xy) / uL);
        return c * lit * lamp;
    }

    void main() {
        vec2 dev = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 p = uView.xy + dev / uView.z;
        float px = 1.0 / uView.z;
        vec2 uv = uvOf(p);
        vec4 A = texture(uFA, uv), B = texture(uFB, uv);
        float d[9];
        d[0] = 1e4;
        d[1] = A.x; d[2] = A.y; d[3] = A.z; d[4] = A.w;
        d[5] = B.x; d[6] = B.y; d[7] = B.z; d[8] = B.w;
        /* a scalpel's line: a little wobble along every cut */
        vec2 g[9];
        g[0] = vec2(0.0);
        for (int k = 1; k <= 8; k++) {
            float fk = float(k);
            d[k] += uL * (0.0007 * (vn(p / (uL * 0.0035) + fk * 7.1) - 0.5) + 0.0022 * (vn(p / (uL * 0.025) + fk * 3.3) - 0.5));
            g[k] = vec2(dFdx(d[k]), -dFdy(d[k]));
        }
        vec3 col = vec3(0.0);
        float T = 1.0;
        for (int k = 8; k >= 0; k--) {
            float c = k == 0 ? 1.0 : smoothstep(-0.7 * px, 0.7 * px, d[k]);
            if (c <= 0.0) continue;
            col += T * c * shade(k, p, d, g[k], px);
            T *= 1.0 - c;
            if (T < 0.004) break;
        }
        col += (h21(gl_FragCoord.xy) - 0.5) / 255.0;
        o = vec4(col, 1.0);
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

static GLuint prog_field, prog_comp, prog_text;
static GLuint vao_empty, vao_text, vbo_text;
static GLuint tex_atlas, tex_noise, tex_paper;
static GLuint fbo_field, tex_fa, tex_fb;
static int fw_, fh_, f_m;               /* field size (texels) and margin (device px) */

/* a tileable 512x512 noise tile: R fbm (16 lattice cells across, 4
 * octaves), G and B value noise (64 cells) */
#define NT 512
static double tile_vn(double x, double y, int Lc, uint32_t salt) {
    double fx = floor(x), fy = floor(y);
    int ix = (int)fx, iy = (int)fy;
    double u = x - fx, v = y - fy;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v);
    int x0 = ((ix % Lc) + Lc) % Lc, y0 = ((iy % Lc) + Lc) % Lc, x1 = (x0 + 1) % Lc, y1 = (y0 + 1) % Lc;
    double a = lat(x0, y0, salt), b = lat(x1, y0, salt), c = lat(x0, y1, salt), d = lat(x1, y1, salt);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}
static void tex_params(GLint wrap, int mip) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
}
static void bake_noise(void) {
    unsigned char *t = malloc(NT * NT * 4);
    for (int y = 0; y < NT; y++)
        for (int x = 0; x < NT; x++) {
            double u = (double)x / NT, v = (double)y / NT, f = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 4; o++) {
                int Lc = 16 << o;
                f += amp * tile_vn(u * Lc, v * Lc, Lc, 900 + o);
                norm += amp; amp *= 0.5;
            }
            f /= norm;
            f = 0.5 + (f - 0.5) * 1.6;
            double g = tile_vn(u * 64, v * 64, 64, 911), b = tile_vn(u * 64, v * 64, 64, 913);
            unsigned char *o = t + 4 * (y * NT + x);
            o[0] = (unsigned char)(clamp01(f) * 255); o[1] = (unsigned char)(g * 255); o[2] = (unsigned char)(b * 255);
            o[3] = 255;
        }
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NT, NT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tex_params(GL_REPEAT, 1);
    free(t);
}

/* the paper itself, 1024x1024 and tileable, a texel per layout pixel:
 * R a height map: the cloudy "formation" of the pulp at a few pixels to a
 * few dozen, short fibres lying in it and a fine tooth; G a broad mottle */
#define PT 1024
static void bake_paper(void) {
    float *h = calloc((size_t)PT * PT, sizeof(float));
    Rng r = rng_for(0x9a9e7, 1, 2);
    for (int y = 0; y < PT; y++)
        for (int x = 0; x < PT; x++) {
            double u = (double)x / PT, v = (double)y / PT;
            double fm = 0.45 * tile_vn(u * 256, v * 256, 256, 960) + 0.30 * tile_vn(u * 128, v * 128, 128, 961) +
                        0.17 * tile_vn(u * 64, v * 64, 64, 962) + 0.08 * tile_vn(u * 32, v * 32, 32, 963);
            h[y * PT + x] = (float)(fm - 0.5) + (float)(rnd(&r) - 0.5) * 0.18f;
        }
    /* fibres: short, nearly straight, lying in the sheet */
    for (int f = 0; f < 5000; f++) {
        double x = rnd(&r) * PT, y = rnd(&r) * PT, a = rnd(&r) * 2 * M_PI;
        int steps = 6 + (int)(rnd(&r) * 22);
        float amp = (float)((rnd(&r) < 0.7 ? 1 : -1) * (0.04 + 0.08 * rnd(&r)));
        double bend = (rnd(&r) - 0.5) * 0.08;
        for (int s = 0; s < steps; s++) {
            x += 0.8 * cos(a); y += 0.8 * sin(a);
            a += bend + (rnd(&r) - 0.5) * 0.06;
            int ix = (int)floor(x), iy = (int)floor(y);
            for (int dy = -1; dy <= 2; dy++)
                for (int dx = -1; dx <= 2; dx++) {
                    double ddx = ix + dx - x, ddy = iy + dy - y;
                    float wgt = (float)exp(-(ddx * ddx + ddy * ddy) / 0.45);
                    h[(((iy + dy) % PT + PT) % PT) * PT + (((ix + dx) % PT + PT) % PT)] += amp * wgt;
                }
        }
    }
    double mean = 0, var = 0;
    for (int i = 0; i < PT * PT; i++) mean += h[i];
    mean /= PT * PT;
    for (int i = 0; i < PT * PT; i++) var += (h[i] - mean) * (h[i] - mean);
    double sd = sqrt(var / (PT * PT)) + 1e-9;
    unsigned char *t = malloc((size_t)PT * PT * 4);
    for (int y = 0; y < PT; y++)
        for (int x = 0; x < PT; x++) {
            double u = (double)x / PT, v = (double)y / PT, m = 0, amp = 0.5, norm = 0;
            for (int o = 0; o < 4; o++) {
                int Lc = 6 << o;
                m += amp * tile_vn(u * Lc, v * Lc, Lc, 940 + o);
                norm += amp; amp *= 0.5;
            }
            m /= norm;
            unsigned char *o = t + 4 * (y * PT + x);
            o[0] = (unsigned char)(clamp01(0.5 + (h[y * PT + x] - mean) / sd * 0.16) * 255);
            o[1] = (unsigned char)(clamp01(0.5 + (m - 0.5) * 2.2) * 255);
            o[2] = 128; o[3] = 255;
        }
    glGenTextures(1, &tex_paper);
    glBindTexture(GL_TEXTURE_2D, tex_paper);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PT, PT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tex_params(GL_REPEAT, 1);
    free(t); free(h);
}

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "papercut: shader:\n%s\n", log);
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

/* ---- palettes: the base sheet, then the eight cut sheets back to front ---- */
static const struct palette {
    const char *name;
    int n;
    float stop[6][3];                  /* interpolated over the nine sheets */
    float shadow, emboss;
} palettes[] = {
    {"paper",  9, {{0.0f}}, 0.30f, 0.45f},           /* filled in pal_colours: white on white */
    {"dusk",   5, {{1.00f, 0.86f, 0.64f}, {0.95f, 0.56f, 0.45f}, {0.62f, 0.31f, 0.45f}, {0.25f, 0.18f, 0.36f}, {0.10f, 0.10f, 0.20f}}, 0.40f, 0.30f},
    {"sea",    5, {{0.95f, 0.93f, 0.84f}, {0.66f, 0.86f, 0.80f}, {0.24f, 0.60f, 0.63f}, {0.07f, 0.32f, 0.40f}, {0.03f, 0.15f, 0.22f}}, 0.40f, 0.30f},
    {"forest", 5, {{0.96f, 0.92f, 0.76f}, {0.76f, 0.82f, 0.58f}, {0.40f, 0.58f, 0.38f}, {0.16f, 0.34f, 0.26f}, {0.07f, 0.17f, 0.14f}}, 0.40f, 0.30f},
    {"kraft",  4, {{0.95f, 0.91f, 0.82f}, {0.80f, 0.65f, 0.47f}, {0.58f, 0.42f, 0.28f}, {0.30f, 0.20f, 0.13f}}, 0.40f, 0.40f},
    {"ember",  5, {{1.00f, 0.90f, 0.62f}, {0.98f, 0.64f, 0.30f}, {0.86f, 0.34f, 0.20f}, {0.52f, 0.13f, 0.16f}, {0.22f, 0.06f, 0.10f}}, 0.42f, 0.30f},
    {"brass",  4, {{0.93f, 0.78f, 0.45f}, {0.76f, 0.55f, 0.24f}, {0.40f, 0.27f, 0.12f}, {0.11f, 0.08f, 0.06f}}, 0.44f, 0.35f},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float pal_col[9][3], cap_col[3], cap_halo[3];

static float lin(float c) { return powf(c, 2.2f); }
static float gam(float c) { return powf(c, 1 / 2.2f); }

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "papercut: no palette \"%s\" (paper, dusk, sea, forest, kraft, ember, brass)\n", opt_palette);
            exit(1);
        }
    } else {
        uint32_t h = opt_seed * 2654435761u;
        h ^= h >> 15;
        pal = (int)(h % NPAL);
    }
    const struct palette *P = &palettes[pal];
    for (int k = 0; k < 9; k++) {
        if (pal == 0) {
            /* white card, alternately a shade warmer and cooler */
            static const float wc[2][3] = {{0.965f, 0.955f, 0.935f}, {0.950f, 0.948f, 0.940f}};
            for (int c = 0; c < 3; c++) pal_col[k][c] = k == 0 ? 0.93f - 0.01f * c : wc[k & 1][c];
            continue;
        }
        double u = k / 8.0 * (P->n - 1);
        int i = (int)u;
        if (i >= P->n - 1) i = P->n - 2;
        float f = (float)(u - i);
        for (int c = 0; c < 3; c++) pal_col[k][c] = gam(lin(P->stop[i][c]) * (1 - f) + lin(P->stop[i + 1][c]) * f);
    }
    /* the caption: light on a dark front sheet, dark on a light one */
    float lum = 0.3f * pal_col[8][0] + 0.59f * pal_col[8][1] + 0.11f * pal_col[8][2];
    for (int c = 0; c < 3; c++) {
        cap_col[c] = lum < 0.45f ? 0.96f - 0.04f * c : 0.16f - 0.02f * c;
        cap_halo[c] = pal_col[8][c] * 0.9f;
    }
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
    float mx = (float)fmax(14, H * 0.035), by = (float)(H - fmax(14, H * 0.035));
    float y1 = by, y0 = by - cap_px[1] * 1.55f;
    float halo[4] = {cap_halo[0], cap_halo[1], cap_halo[2], 0.16f}, col[4] = {cap_col[0], cap_col[1], cap_col[2], 0.92f};
    float r = fmaxf(1.0f, cap_px[0] / 12.0f);
    for (int k = 0; k < 12; k++) {
        float a = k * 0.5236f, dx = r * cosf(a), dy = r * sinf(a);
        for (int q = 1; q <= 2; q++) {
            text_at(0, mx + dx * q, y0 + dy * q, l0, halo);
            text_at(1, mx + dx * q, y1 + dy * q, l1, halo);
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
    prog_field = mk_prog(VS_FULL, FS_FIELD);
    prog_comp = mk_prog(VS_FULL, FS_COMP);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    glGenVertexArrays(1, &vao_empty);
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
    bake_paper();
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &tex_text);
    glBindTexture(GL_TEXTURE_2D, tex_text);
    tex_params(GL_CLAMP_TO_EDGE, 0);
    {
        float big = 1e3f;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, &big);
    }
    glGenTextures(1, &tex_fa);
    glGenTextures(1, &tex_fb);
    glGenFramebuffers(1, &fbo_field);
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

/* the field targets: half resolution, with a margin for shadows cast from
 * just off the screen */
static void field_resize(void) {
    int m = (int)ceil(0.10 * L * v_sc) & ~1;
    int fw = (W + 2 * m + 1) / 2, fh = (H + 2 * m + 1) / 2;
    if (fw == fw_ && fh == fh_ && m == f_m) return;
    fw_ = fw; fh_ = fh; f_m = m;
    GLuint t[2] = {tex_fa, tex_fb};
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, t[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, fw, fh, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
        tex_params(GL_CLAMP_TO_EDGE, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_field);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_fa, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_fb, 0);
    static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("field framebuffer incomplete");
}

/* how far the act's text has risen through the stack: the sheet index its
 * plain letters have reached (below the first sheet: not there at all) */
#define TXT_END (TXT_OUT + TXT_RISE + 4)
static double text_level(long a, double ts, double *on, double *moat) {
    *on = 0;
    if (act_text(a) == T_NONE || ts < TXT_IN || ts > TXT_END || txt_hw <= 0) return -100;
    double lo = -txt_hw / (TXT_GROW * L) - 1.5, hi = NLAYER - 1;
    /* the window opens first and closes last */
    double u = ts < TXT_OUT ? ease((ts - TXT_IN - 3) / TXT_RISE) : 1 - ease((ts - TXT_OUT) / TXT_RISE);
    double m = ts < TXT_OUT ? ease((ts - TXT_IN) / 6) : 1 - ease((ts - TXT_END + 6) / 6);
    *moat = -txt_hw - 2 + (txt_hw + 2 + (3 * TXT_GROW + 0.022) * L) * m;
    *on = 1;
    return lo + (hi - lo) * u;
}

/* where the light comes from (the direction shadows fall, radians, y down):
 * the upper left, but from low down for the hills, so each ridge throws
 * its shadow up onto the one behind it */
static double scene_light(int kind) { return kind == S_HILLS ? -1.05 : 0.95; }

static void set_scene(GLuint pr, const char *ku, const char *pu, const Scene *s) {
    glUniform1i(U(pr, ku), s->kind);
    glUniform4fv(U(pr, pu), NPAR, &s->P[0][0]);
}

static void render(double t, double wall, GLuint target) {
    if (t < 0) t = 0;
    view_update();
    field_resize();
    long a = (long)floor(t / ACT);
    double ts = t - a * ACT, u = clamp01((ts - (ACT - MORPH)) / MORPH);
    Scene A, B;
    scene_params(a, &A);
    if (u > 0) scene_params(a + 1, &B);
    else B = A;
    double on = 0, tt = -100, moat = -1e3;
    if (act_text(a) != T_NONE && ts >= TXT_IN && ts <= TXT_END) {
        build_text(a, act_text(a));
        tt = text_level(a, ts, &on, &moat);
    }

    /* the field */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_field);
    glViewport(0, 0, fw_, fh_);
    glDisable(GL_BLEND);
    glUseProgram(prog_field);
    glUniform3f(U(prog_field, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform1f(U(prog_field, "uM"), (float)f_m);
    glUniform1f(U(prog_field, "uL"), (float)L);
    glUniform1f(U(prog_field, "uT"), (float)fmod(t, 20000.0));
    glUniform1f(U(prog_field, "uU"), (float)u);
    set_scene(prog_field, "uKA", "uPA", &A);
    set_scene(prog_field, "uKB", "uPB", &B);
    glUniform4f(U(prog_field, "uTBox"), txt_box[0], txt_box[1], txt_box[2], txt_box[3]);
    glUniform1f(U(prog_field, "uTT"), (float)tt);
    glUniform1f(U(prog_field, "uTGrow"), (float)(TXT_GROW * L));
    glUniform1f(U(prog_field, "uTOn"), (float)on);
    glUniform1f(U(prog_field, "uTMoat"), (float)moat);
    glUniform1i(U(prog_field, "uNoise"), 0);
    glUniform1i(U(prog_field, "uText"), 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_text);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* the composite */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_comp);
    glUniform3f(U(prog_comp, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(prog_comp, "uRes"), (float)W, (float)H);
    glUniform2f(U(prog_comp, "uFRes"), (float)fw_, (float)fh_);
    glUniform1f(U(prog_comp, "uM"), (float)f_m);
    glUniform1f(U(prog_comp, "uL"), (float)L);
    glUniform3fv(U(prog_comp, "uCol"), 9, &pal_col[0][0]);
    /* the light: from the upper left, swinging a little over ten minutes */
    double la = scene_light(A.kind) + (scene_light(B.kind) - scene_light(A.kind)) * ease(u) + 0.22 * sin(t * 2 * M_PI / 600.0);
    glUniform2f(U(prog_comp, "uLd"), (float)cos(la), (float)sin(la));
    glUniform1f(U(prog_comp, "uGap"), (float)(0.0075 * L));
    glUniform1f(U(prog_comp, "uSoft0"), (float)(0.0025 * L));
    glUniform1f(U(prog_comp, "uSoft1"), (float)(0.0045 * L));
    glUniform1f(U(prog_comp, "uShadow"), palettes[pal].shadow);
    glUniform1f(U(prog_comp, "uEmb"), palettes[pal].emboss);
    glUniform3f(U(prog_comp, "uLamp"), (float)(fr[0] + 0.1 * fr[2]), (float)(fr[1] - 0.2 * fr[3]), 0.06f);
    glUniform1i(U(prog_comp, "uFA"), 1);
    glUniform1i(U(prog_comp, "uFB"), 2);
    glUniform1i(U(prog_comp, "uNoise"), 0);
    glUniform1i(U(prog_comp, "uPaper"), 3);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_fa);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_fb);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_paper);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "papercut");
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
        "usage: papercut [--window | --root | --layer] [--seed N] [--fps N]\n"
        "                [--palette paper|dusk|sea|forest|kraft|ember|brass]\n"
        "                [--scene tunnel|hills|islands|bloom] [--no-caption]\n"
        "                [--text WORD] [--font FAMILY] [--schedule]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX] [--render-scale F]\n"
        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --palette   the card; default: picked from the seed\n"
        "  --scene     hold one scene (re-cut every %.0f s) instead of cycling\n"
        "  --text      what the host-name cuts spell instead of the host name\n"
        "  --fps       frame rate (default 30: it moves slowly)\n"
        "  --schedule  print what happens when (seconds after the epoch) and exit\n"
        "  --render-scale  with --layer, draw F (0-1] of the output's native pixels (default 1)\n", ACT);
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
    for (long a = 0; a < 12; a++) {
        int tk = act_text(a);
        printf("%7.1f s  act %2ld: %-8s%s  (re-cut into %s from %.1f s)\n", a * ACT, a, scene_name[act_kind(a)],
               tk == T_TIME ? "  + the time" : tk == T_HOST ? "  + the host name" : "",
               scene_name[act_kind(a + 1)], (a + 1) * ACT - MORPH);
    }
    printf("text: cut from %.1f s into its act, risen by %.1f s, sinks from %.1f s, gone by %.1f s\n", TXT_IN, TXT_IN + 3 + TXT_RISE, TXT_OUT, TXT_END);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-scene") && i + 1 < argc) opt_scene = argv[++i];
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
    if (opt_scene) {
        for (forced_scene = 0; forced_scene < NSCENE; forced_scene++) if (!strcmp(scene_name[forced_scene], opt_scene)) break;
        if (forced_scene == NSCENE) { fprintf(stderr, "papercut: no scene \"%s\" (tunnel, hills, islands, bloom)\n", opt_scene); exit(1); }
    }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (opt_schedule) { print_schedule(); return 0; }
    int offscreen = opt_shot || opt_bench > 0;
    pick_palette();
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    fonts_init();

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    int dw = W, dh = H;
    /* the domain is in logical px (the view scales it to the buffer), so a HiDPI output cuts the same shapes */
    if (opt_layer && !offscreen) { wl_init(); dw = hd.lw > 0 ? hd.lw : W; dh = hd.lh > 0 ? hd.lh : H; }
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
        /* a window gets its real size from the WM: lay the stack out for
         * the screen it will most likely fill */
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Papercut");
        XClassHint chint = {"papercut", "Papercut"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    domain_init(dw, dh);
    gl_init();

    /* --epoch: when the stack was first cut, shared by every instance; a
     * --shot or --bench then shows the moment --at seconds after it */
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
        long a = (long)floor(fmax(opt_at, 0) / ACT);
        if (opt_shot && !opt_bench) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            fprintf(stderr, "papercut: %s, %dx%d, %s (act %ld), first frame %.0f ms\n", palettes[pal].name, W, H,
                    scene_name[act_kind(a)], a, first);
            return 0;
        }
        GLuint q;
        glGenQueries(1, &q);
        double sum = 0, worst = 0, gsum = 0, gmin = 1e9;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 30.0;
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, opt_epoch + t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            double ms = (now() - f0) * 1000;
            sum += ms; gsum += ns / 1e6; gmin = fmin(gmin, ns / 1e6);
            if (ms > worst) worst = ms;
        }
        printf("papercut: %s, %s, %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, scene_name[act_kind(a)], W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("papercut:   GPU %.3f ms (best %.3f); field %dx%d\n", gsum / opt_bench, gmin, fw_, fh_);
        printf("papercut: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
        if (opt_shot) { glBindFramebuffer(GL_FRAMEBUFFER, fbo); save_ppm(opt_shot); }
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
