/*
 * origami — a big sheet of paper lying under a low lamp, folding itself
 * into origami tessellations and opening out again. The crease pattern is
 * pressed into the sheet; how far each part of it is folded drifts in slow
 * swells, so the facets tilt in and out of the light and the whole sheet
 * seems to breathe.
 *
 * Patterns, each held for a minute and a half:
 *
 *   miura      Miura-ori: zigzag pleats of parallelograms, a herringbone of
 *              four facet tones
 *   pleats     curved-crease pleating: parallel pleats bent along a slow
 *              wave, so the folded surface rolls like cloth
 *   waterbomb  the waterbomb tessellation: offset rows of waterbomb bases,
 *              each standing up as a little pyramid
 *   hex        a triangle grid folded into hexagonal pyramids or, with the
 *              third set of vertices half raised, tumbling blocks
 *   twist      a square twist: in every cell a square rises and turns,
 *              neighbours turning opposite ways, pleated down to the corners
 *
 * At the end of a pattern a front sweeps across the sheet: ahead of it the
 * old folds relax flat, behind it the next pattern's creases fold up.
 *
 * Every third pattern a clearing opens in the folds (its crease lines still
 * faintly there, as on any unfolded sheet) and the time (or the host name)
 * is blind-embossed into the flat paper there: raised letters
 * with a soft bevel, lit by the same raking lamp; then they sink back and
 * the folds close over.
 *
 * What makes it read as paper: it is lit, not coloured: a low lamp off to
 * one side, so a facet's tone is its angle to the light, faces turned away
 * fall into a translucent half-light, and every fold casts a soft shadow
 * into the valley beyond it (marched through a height buffer). The creases
 * are slightly rounded, catching a thin highlight on a mountain fold and a
 * dark line in a valley, and coloured papers whiten a little along their
 * creases where the dye cracks. The sheet carries a baked fibre texture
 * (formation, tooth and, for washi, long kozo fibres) pinned to the paper
 * itself, embossed by the light, so it slides with the facets as they fold.
 *
 * Rendering: the sheet is a mesh generated in the vertex shader from the
 * lattice (no vertex buffers), folded by a fold-amount field; it writes
 * height, facet normal and paper coordinates into a G-buffer, and a
 * full-screen pass lights it. Light work for the CPU.
 *
 * Palettes (--palette, else picked from the seed): paper, washi, kraft,
 * indigo, rose, sage, ink, brass.
 *
 * Deterministic: everything is a function of --seed and the time since
 * --epoch (--schedule prints the patterns). Spanning (like papercut): run
 * one instance per monitor with the same --seed, --epoch and --layout, and
 * --output = this one's index; the sheet lives in wall coordinates, so it
 * is one continuous sheet across the bezels. The "front" output (under the
 * middle of the layout) sets the scale and carries the caption and the time.
 *
 * Same plumbing as papercut: Xlib + GLX for --window / --root, a native
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
static double opt_fps = 30;
static const char *opt_font = "Inter";
static const char *opt_palette;
static const char *opt_scene;
static const char *opt_text;           /* --text: what the host-name embossing spells instead */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds after the epoch */
static int opt_bench;
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the sheet starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;

static void die(const char *what) { fprintf(stderr, "origami: %s\n", what); exit(1); }

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
#define ACT 90.0                       /* seconds a pattern is held, including the sweep into the next */
#define MORPH 24.0                     /* the last MORPH seconds sweep into the next pattern */
enum { S_MIURA, S_PLEATS, S_WATERBOMB, S_HEX, S_TWIST, NSCENE };
static const char *scene_name[NSCENE] = {"miura", "pleats", "waterbomb", "hex", "twist"};
static int forced_scene = -1;

/* the patterns come in shuffled rounds, never the same twice running */
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

/* the embossed text: every third act, the time or the host name */
enum { T_NONE, T_TIME, T_HOST };
#define TXT_IN 6.0                      /* the clearing starts to open */
#define TXT_OPEN 8.0                    /* ...taking this long */
#define TXT_RISE 5.0                    /* the letters rise once it's open */
#define TXT_OUT 50.0                    /* the letters start to sink */
#define TXT_END (TXT_OUT + TXT_RISE + TXT_OPEN)
static int act_text(long a) {
    if (a % 3) return T_NONE;
    return (((a / 3) + opt_seed) & 1) ? T_HOST : T_TIME;
}

/* a pattern, as the shaders read it */
typedef struct {
    int kind, vpc;
    float P[4][4];     /* the lattice: see the mesh shader */
    float Lv[3][4];    /* the fold-amount field */
    float Sw[4];       /* the sweep into this pattern: radial?, centre / direction, wobble */
} Scene;

static void scene_params(long a, Scene *s) {
    memset(s, 0, sizeof *s);
    s->kind = act_kind(a);
    Rng r = rng_for((uint64_t)a, 17, (uint64_t)s->kind);
    float (*P)[4] = s->P;
    double rot = 0;
    s->vpc = 6;
    switch (s->kind) {
    case S_MIURA: {
        double A = L * rr(&r, 0.075, 0.095), B = A * rr(&r, 0.62, 0.8);
        P[0][0] = (float)A; P[0][1] = (float)B;
        P[0][2] = (float)(B * rr(&r, 0.55, 0.7));             /* height at full fold */
        P[0][3] = (float)(B * rr(&r, 0.22, 0.34));             /* the zigzag, flat */
        P[1][0] = (float)(B * rr(&r, 0.10, 0.18));             /* ...and how much it sharpens folded */
        P[1][1] = 0;
        rot = rnd(&r) < 0.6 ? 0 : M_PI / 2;
        rot += rr(&r, -0.12, 0.12);
        break;
    }
    case S_PLEATS: {
        double B = L * rr(&r, 0.045, 0.06), lam = L * rr(&r, 0.6, 1.0);
        P[0][0] = (float)(L * 0.008);                          /* columns: fine, so the creases curve */
        P[0][1] = (float)B;
        P[0][2] = (float)(B * rr(&r, 0.55, 0.7));
        P[0][3] = (float)(lam * rr(&r, 0.07, 0.11));           /* the bend */
        P[1][0] = (float)(lam * rr(&r, 0.02, 0.04));
        P[1][1] = 1;
        P[1][2] = (float)(2 * M_PI / lam);
        P[1][3] = (float)rr(&r, 0, 2 * M_PI);
        P[3][0] = (float)rr(&r, 0.0, 0.35);                    /* a second, slower harmonic */
        P[3][1] = (float)rr(&r, 0, 2 * M_PI);
        rot = rnd(&r) < 0.5 ? 0 : M_PI / 2;
        rot += rr(&r, -0.2, 0.2);
        break;
    }
    case S_WATERBOMB: {
        double A = L * rr(&r, 0.11, 0.14);
        P[0][0] = (float)A; P[0][1] = (float)A;
        P[0][2] = (float)(A * rr(&r, 0.34, 0.44));
        P[3][0] = (float)rr(&r, 0.15, 0.3);                    /* the corners' height, of the centre's */
        s->vpc = 24;
        rot = rnd(&r) < 0.6 ? 0 : M_PI / 2;
        break;
    }
    case S_TWIST: {
        double A = L * rr(&r, 0.10, 0.13);
        P[0][0] = (float)A; P[0][1] = (float)A;
        P[0][2] = (float)(A * rr(&r, 0.16, 0.22));             /* the squares rise this far */
        P[3][0] = (float)rr(&r, 0.26, 0.34);                   /* their half-size, of the cell */
        P[3][1] = (float)rr(&r, 0.45, 0.7);                    /* and their twist folded, radians */
        s->vpc = 30;
        rot = rr(&r, -0.3, 0.3);
        break;
    }
    case S_HEX: {
        double A = L * rr(&r, 0.09, 0.12);
        P[0][0] = (float)A; P[0][1] = (float)A;
        P[0][2] = (float)(A * rr(&r, 0.34, 0.44));
        P[3][0] = rnd(&r) < 0.5 ? 0.0f : (float)rr(&r, 0.45, 0.55);   /* pyramids, or blocks */
        rot = rr(&r, 0, M_PI / 3);
        break;
    }
    }
    P[2][0] = (float)cos(rot); P[2][1] = (float)sin(rot);
    P[2][2] = (float)(bx0 + bw / 2); P[2][3] = (float)(by0 + bh / 2);
    /* the fold amount: a level, slow blobs drifting over it, and a swell */
    s->Lv[0][0] = (float)rr(&r, 0.55, 0.68);
    s->Lv[0][1] = (float)rr(&r, 0.4, 0.55);
    s->Lv[0][2] = (float)(L * rr(&r, 0.7, 1.1));
    s->Lv[0][3] = (float)rr(&r, 0.15, 0.24);
    {
        double da = rr(&r, 0, 2 * M_PI), v = rr(&r, 0.010, 0.018);          /* blobs per second */
        s->Lv[1][0] = (float)(v * cos(da)); s->Lv[1][1] = (float)(v * sin(da));
        double wa = rr(&r, 0, 2 * M_PI), k = 2 * M_PI / (L * rr(&r, 0.8, 1.3));
        s->Lv[1][2] = (float)(k * cos(wa)); s->Lv[1][3] = (float)(k * sin(wa));
        s->Lv[2][0] = (float)(2 * M_PI / rr(&r, 18, 28));
        s->Lv[2][1] = (float)rr(&r, 0, 16); s->Lv[2][2] = (float)rr(&r, 0, 16);
    }
    /* the sweep that brings this pattern in */
    if (rnd(&r) < 0.45) {
        double x, y;
        visible_point(&r, &x, &y, 0.2);
        s->Sw[0] = 1; s->Sw[1] = (float)x; s->Sw[2] = (float)y;
    } else {
        double da = rr(&r, 0, 2 * M_PI);
        s->Sw[0] = 0; s->Sw[1] = (float)cos(da); s->Sw[2] = (float)sin(da);
    }
    s->Sw[3] = (float)(0.25 * L);
}

/* the range of a sweep's coordinate over the wall (without its wobble) */
static void sweep_range(const Scene *s, double *lo, double *hi) {
    *lo = 1e300; *hi = -1e300;
    for (int i = 0; i <= 8; i++)
        for (int j = 0; j <= 8; j++) {
            double x = bx0 + bw * i / 8, y = by0 + bh * j / 8;
            double v = s->Sw[0] > 0.5 ? hypot(x - s->Sw[1], y - s->Sw[2]) : x * s->Sw[1] + y * s->Sw[2];
            *lo = fmin(*lo, v); *hi = fmax(*hi, v);
        }
}
/* ==== the embossed text ==== */
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
static float txt_box[4], txt_ink[4];   /* where the SDF lies (world), and the ink's x0, y0, x1, y1 */
static long txt_act = -1;
static int txt_ok;

/* the text of act a as a signed distance field in world pixels (positive
 * outside the letters), over the box it is fitted to */
static void build_text(long a, int kind) {
    if (txt_act == a) return;
    txt_act = a;
    char buf[64];
    double box[4];
    text_box(kind == T_HOST, box);
    if (kind == T_TIME) {
        time_t tt = (time_t)(opt_epoch + a * ACT + TXT_IN + TXT_OPEN);
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
    txt_ok = 0;
    if (x1 <= x0) return;
    double fitw = fmin(box[2] * 0.46 / (x1 - x0 + 1), box[3] * 0.25 / (y1 - y0 + 1)) * 100;   /* world px per em */
    /* the canvas: one texel per world pixel, unless that gets too big */
    double tx = fmax(1.0, (x1 - x0) * fitw / 100 / 3000);
    int px = (int)(fitw / tx), pad = (int)(0.03 * L / tx) + 4;
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
    for (size_t i = 0; i < n; i++) {
        float a2 = cv[i] / 255.0f, s;
        if (a2 > 0.02f && a2 < 0.98f) s = 0.5f - a2;
        else if (cv[i] >= 128) s = -(sqrtf(gi[i]) - 0.5f);
        else s = sqrtf(go[i]) - 0.5f;
        sd[i] = s * (float)tx;
    }
    /* ink centre onto the box centre */
    double cx = box[0] + box[2] / 2, cyy = box[1] + box[3] / 2;
    txt_box[0] = (float)(cx - (x0 + x1 + 1) * 0.5 * tx);
    txt_box[1] = (float)(cyy - (y0 + y1 + 1) * 0.5 * tx);
    txt_box[2] = (float)(cw * tx); txt_box[3] = (float)(ch * tx);
    txt_ink[0] = (float)(txt_box[0] + x0 * tx); txt_ink[1] = (float)(txt_box[1] + y0 * tx);
    txt_ink[2] = (float)(txt_box[0] + (x1 + 1) * tx); txt_ink[3] = (float)(txt_box[1] + (y1 + 1) * tx);
    txt_ok = 1;
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

/* the sheet: a lattice of cells generated from gl_VertexID, uVPC vertices a
 * cell. Lattice points are in half-cell units (I, J), biased positive, so a
 * point shared by several cells always comes out the same. uP:
 *   [0] cell width, cell height, height at full fold, zigzag (flat)
 *   [1] zigzag gain folded, smooth (curved) profile, its wavenumber, phase
 *   [2] cos, sin of the lattice's rotation, its origin (world)
 *   [3] pattern extras */
static const char *VS_MESH = "#version 330 core\n" GLSL(
    uniform sampler2D uNoise;
    uniform vec3 uView;
    uniform vec2 uGRes;
    uniform float uM, uT, uL;
    uniform int uKind, uVPC, uNX;
    uniform ivec2 uC0;
    uniform vec4 uP[4];
    uniform vec4 uLv[3];
    uniform vec4 uSw;
    uniform vec4 uFront;
    uniform vec4 uClr;
    uniform vec2 uClr2;
    out vec3 vW, vN, vR, vNR;
    out vec2 vPaper;
    float gF = -1.0;    /* >= 0: fold everything this far (the crease pattern's reference shape) */

    const int BIAS = 8192;
    const ivec2 QUAD[6] = ivec2[6](ivec2(0, 0), ivec2(2, 0), ivec2(2, 2), ivec2(0, 0), ivec2(2, 2), ivec2(0, 2));
    const ivec2 TRI[6] = ivec2[6](ivec2(0, 0), ivec2(2, 0), ivec2(0, 2), ivec2(2, 0), ivec2(2, 2), ivec2(0, 2));
    const int TW[30] = int[30](0, 1, 2, 0, 2, 3, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7);
    const ivec2 CORNER[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));
    const ivec2 RING[8] = ivec2[8](ivec2(0, 0), ivec2(1, 0), ivec2(2, 0), ivec2(2, 1), ivec2(2, 2), ivec2(1, 2), ivec2(0, 2), ivec2(0, 1));

    float N(vec2 p) { return textureLod(uNoise, p * (1.0 / 16.0), 0.0).r; }

    float sweepS(vec2 p) {
        float s = uSw.x < 0.5 ? dot(p, uSw.yz) : length(p - uSw.yz);
        return s + uSw.w * (N(p / (0.6 * uL) + 3.7) - 0.5) * 2.0;
    }

    /* how far folded the sheet is here, 0 (flat) to 1 */
    float level(vec2 p) {
        float b = uLv[0].x + uLv[0].y * (N(p / uLv[0].z + uLv[1].xy * uT + uLv[2].yz) - 0.5) * 3.5;
        b += uLv[0].w * sin(dot(p, uLv[1].zw) - uLv[2].x * uT);
        b = clamp(b, 0.0, 1.0);
        if (uFront.z != 0.0) b *= smoothstep(uFront.w, uFront.w + uFront.y, (sweepS(p) - uFront.x) * uFront.z);
        if (uClr2.y > 0.0) {
            /* a rounded clearing with a wandering edge */
            vec2 q = abs(p - uClr.xy) / uClr.zw;
            float d = (pow(pow(q.x, 4.0) + pow(q.y, 4.0), 0.25) - 1.0) * min(uClr.z, uClr.w);
            d += 0.08 * uL * (N(p / (0.35 * uL) + 9.1) - 0.5) * 2.0;
            b *= 1.0 - uClr2.y * (1.0 - smoothstep(0.0, uClr2.x, d));
        }
        return b;
    }

    vec2 rot(vec2 l) { return vec2(uP[2].x * l.x - uP[2].y * l.y, uP[2].y * l.x + uP[2].x * l.y); }

    /* lattice point (I, J): where it lies folded (world, z up), and flat */
    vec3 sheet(int I, int J, out vec2 pf) {
        float a = uP[0].x, b = uP[0].y, h = uP[0].z;
        vec2 lf, lz;          /* flat position, and how it moves as it folds */
        float zc;
        if (uKind <= 1) {     /* miura and pleats: rows of pleats, zigzag or curved */
            int i = I / 2, j = J / 2;
            float x = float(i - BIAS) * a;
            float g = (i & 1) == 1 ? 0.5 : -0.5;
            if (uKind == 1) g = sin(x * uP[1].z + uP[1].w) + uP[3].x * sin(0.37 * x * uP[1].z + uP[3].y);
            lf = vec2(x, float(j - BIAS) * b + uP[0].w * g);
            lz = vec2(0.0, uP[1].x * g);
            zc = float(j & 1);
        } else if (uKind == 2) {   /* waterbomb: offset rows of cells */
            lf = vec2(float(I - 2 * BIAS), float(J - 2 * BIAS)) * (0.5 * a);
            lz = vec2(0.0);
            if ((J & 1) == 0) zc = uP[3].x;
            else zc = ((I - (((J - 1) / 2) & 1)) & 1) == 1 ? 1.0 : 0.0;
        } else {                   /* hex: a triangle lattice, three sets of vertices */
            int i = I / 2 - BIAS, j = J / 2 - BIAS;
            lf = vec2((float(i) + 0.5 * float(j)) * a, float(j) * a * 0.8660254);
            lz = vec2(0.0);
            int m = (I / 2 - J / 2 + 3 * BIAS) % 3;
            zc = m == 0 ? 1.0 : m == 1 ? uP[3].x : 0.0;
        }
        pf = uP[2].zw + rot(lf);
        float f = gF >= 0.0 ? gF : level(pf);
        return vec3(pf + rot(lz) * f, f * h * zc);
    }

    void main() {
        int cid = gl_VertexID / uVPC, k = gl_VertexID - cid * uVPC;
        int ci = uC0.x + cid % uNX, cj = uC0.y + cid / uNX;
        if (uKind == 4) {
            /* twist: a square in each cell rises and turns, alternate cells
             * turning opposite ways, pleated down to the cell's corners */
            int v = TW[k];
            vec2 lc = vec2(float(ci - BIAS), float(cj - BIAS)) * uP[0].x;
            vec3 w[2];
            vec2 pf;
            for (int pass = 0; pass < 2; pass++) {
                gF = pass == 0 ? -1.0 : 1.0;
                if (v >= 4) {
                    pf = uP[2].zw + rot(lc + vec2(CORNER[v - 4]) * uP[0].x);
                    w[pass] = vec3(pf, 0.0);
                } else {
                    vec2 cc = lc + vec2(0.5 * uP[0].x);
                    float f = gF >= 0.0 ? gF : level(uP[2].zw + rot(cc));
                    vec2 q = (vec2(CORNER[v]) * 2.0 - 1.0) * uP[3].x * uP[0].x;
                    float an = uP[3].y * f * (((ci + cj) & 1) == 1 ? 1.0 : -1.0);
                    pf = uP[2].zw + rot(cc + q);
                    vec2 qr = vec2(cos(an) * q.x - sin(an) * q.y, sin(an) * q.x + cos(an) * q.y);
                    w[pass] = vec3(uP[2].zw + rot(cc + qr), f * uP[0].z);
                }
            }
            vW = w[0]; vR = w[1]; vN = vNR = vec3(0.0, 0.0, 1.0); vPaper = pf;
            vec2 dv = (w[0].xy - uView.xy) * uView.z + uM;
            gl_Position = vec4(dv / uGRes * 2.0 - 1.0, 0.0, 1.0);
            return;
        }
        ivec2 o;
        if (uVPC == 24) {
            int t = k / 3, c = k - t * 3;
            o = c == 0 ? ivec2(1, 1) : RING[(t + c - 1) % 8];
        } else o = uKind == 3 ? TRI[k] : QUAD[k];
        int I = 2 * ci + o.x, J = 2 * cj + o.y;
        if (uKind == 2) I += cj & 1;
        vec2 pf, d;
        vec3 w = sheet(I, J, pf);
        gF = 1.0;
        vR = sheet(I, J, d);
        vN = vNR = vec3(0.0, 0.0, 1.0);
        if (uKind == 1) {
            /* curved pleats: smooth along the pleat, sharp across it */
            int J0 = 2 * cj, J1 = J0 + 2;
            for (int pass = 0; pass < 2; pass++) {
                gF = pass == 0 ? -1.0 : 1.0;
                vec3 tx = sheet(I + 2, J0, d) - sheet(I - 2, J0, d) + sheet(I + 2, J1, d) - sheet(I - 2, J1, d);
                vec3 ty = sheet(I, J1, d) - sheet(I, J0, d);
                vec3 nn = normalize(cross(tx, ty));
                nn *= sign(nn.z);
                if (pass == 0) vN = nn; else vNR = nn;
            }
        }
        vec2 pw = w.xy;
        float z = w.z;
        vW = w;
        vPaper = pf;
        vec2 dev = (pw - uView.xy) * uView.z + uM;
        gl_Position = vec4(dev / uGRes * 2.0 - 1.0, 0.0, 1.0);
    });

static const char *FS_MESH = "#version 330 core\n" GLSL(
    uniform sampler2D uNoise;
    uniform vec4 uSw;
    uniform vec4 uFront;
    uniform float uL;
    uniform int uKind;
    in vec3 vW, vN, vR, vNR;
    in vec2 vPaper;
    layout(location = 0) out vec4 o0;
    layout(location = 1) out vec4 o1;
    layout(location = 2) out float o2;
    float N(vec2 p) { return texture(uNoise, p * (1.0 / 16.0)).r; }
    float sweepS(vec2 p) {
        float s = uSw.x < 0.5 ? dot(p, uSw.yz) : length(p - uSw.yz);
        return s + uSw.w * (N(p / (0.6 * uL) + 3.7) - 0.5) * 2.0;
    }
    void main() {
        if (uFront.z != 0.0 && (sweepS(vPaper) - uFront.x) * uFront.z < 0.0) discard;
        vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
        if (n.z < 0.0) n = -n;
        if (uKind == 1) n = normalize(vN);
        o0 = vec4(vW.z, n.xy, uFront.z);
        o2 = vW.z;
        vec3 nr = normalize(cross(dFdx(vR), dFdy(vR)));
        nr *= sign(nr.z);
        if (uKind == 1) nr = normalize(vNR);
        o1 = vec4(clamp((vPaper - vW.xy) / (0.12 * uL) + 0.5, 0.0, 1.0), nr.xy * 0.5 + 0.5);
    });

/* the light on the paper */
static const char *FS_COMP = "#version 330 core\n" GLSL(
    uniform sampler2D uG0, uG1, uGH, uNoise, uPaper, uText;
    uniform vec3 uView;
    uniform vec2 uRes, uGRes;
    uniform float uM, uL;
    uniform vec3 uLd;                /* towards the lamp (x, y down the screen, z up) */
    uniform vec3 uCol, uCol2;
    uniform vec4 uPal;               /* crease whitening, washi fibres, sheen, shadow */
    uniform float uEmb, uSMax, uAOR;
    uniform vec4 uTBox;
    uniform vec2 uTxt;               /* the letters' height (world px), their bevel */
    uniform vec3 uLamp;
    out vec4 o;

    float h21(vec2 p) { p = fract(p * vec2(234.34, 435.345)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
    float G(vec2 w) { return texture(uGH, ((w - uView.xy) * uView.z + uM) / uGRes).r; }
    float txt(vec2 w) {
        vec2 tu = (w - uTBox.xy) / uTBox.zw;
        if (tu.x <= 0.0 || tu.y <= 0.0 || tu.x >= 1.0 || tu.y >= 1.0) return 1e3;
        return texture(uText, tu).r;
    }
    vec3 nrm(vec4 g) { return vec3(g.yz, sqrt(max(1.0 - dot(g.yz, g.yz), 0.0))); }

    void main() {
        vec2 dev = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 p = uView.xy + dev / uView.z;
        float px = 1.0 / uView.z;
        vec2 gp = dev + uM, iv = 1.0 / uGRes;
        vec4 c0 = texelFetch(uG0, ivec2(gp), 0);
        vec2 disp = (texelFetch(uG1, ivec2(gp), 0).xy - 0.5) * (0.12 * uL);
        /* the facet normal, softened over a pixel or so: the crease is a
         * little rounded, and it antialiases the facet edges */
        vec4 ta = texture(uG0, (gp + vec2(-0.7, -0.7)) * iv), tb = texture(uG0, (gp + vec2(0.7, 0.7)) * iv);
        vec4 tc = texture(uG0, (gp + vec2(0.7, -0.7)) * iv), td = texture(uG0, (gp + vec2(-0.7, 0.7)) * iv);
        vec3 na = nrm(ta), nb = nrm(tb), nc = nrm(tc), nd = nrm(td);
        vec3 n = normalize(nrm(c0) + 0.5 * (na + nb + nc + nd));
        float crease = smoothstep(0.003, 0.03, 1.0 - min(dot(na, nb), dot(nc, nd)));
        /* the paper remembers its creases: a hairline where the pattern
         * folds, even while it lies flat */
        float mem;
        {
            vec3 ra = nrm(texture(uG1, (gp + vec2(-0.7, -0.7)) * iv).yzwx * 2.0 - 1.0), rb = nrm(texture(uG1, (gp + vec2(0.7, 0.7)) * iv).yzwx * 2.0 - 1.0);
            vec3 rc = nrm(texture(uG1, (gp + vec2(0.7, -0.7)) * iv).yzwx * 2.0 - 1.0), rd = nrm(texture(uG1, (gp + vec2(-0.7, 0.7)) * iv).yzwx * 2.0 - 1.0);
            mem = smoothstep(0.01, 0.06, 1.0 - min(dot(ra, rb), dot(rc, rd))) * (1.0 - crease);
            mem *= step(abs(ta.w - tb.w) + abs(tc.w - td.w), 0.01);   /* not where one pattern meets the next */
        }
        float lap = 0.25 * (ta.x + tb.x + tc.x + td.x) - c0.x;
        float ridge = clamp(-lap / (0.08 * px + 0.02), 0.0, 1.0), valley = clamp(lap / (0.08 * px + 0.02), 0.0, 1.0);

        /* the letters, embossed */
        float sd = txt(p);
        if (uTxt.x > 0.0 && sd < 6.0 * uTxt.y) {
            float e = px * 0.75;
            float sx = txt(p + vec2(e, 0.0)) - txt(p - vec2(e, 0.0));
            float sy = txt(p + vec2(0.0, e)) - txt(p - vec2(0.0, e));
            vec2 gd = vec2(sx, sy) / (2.0 * e);
            /* the bevel: a smooth shoulder from the edge inwards */
            float u = clamp(-sd / uTxt.y, 0.0, 1.0);
            float slope = uTxt.x / uTxt.y * 1.6 * u * (1.0 - u) * 4.0 * step(sd, 0.0);
            slope += uTxt.x / uTxt.y * 0.25 * exp(-max(sd, 0.0) / (0.7 * uTxt.y)) * step(0.0, sd);
            n = normalize(n + vec3(gd * slope, 0.0));
        }

        /* real facets are never quite flat: a faint bow in the paper */
        {
            vec2 q = (p + disp) / (0.14 * uL);
            vec2 bw = texture(uNoise, q * (1.0 / 64.0)).gb - 0.5;
            n = normalize(n + vec3(0.06 * bw, 0.0));
        }
        vec3 Ld = normalize(uLd);
        vec2 dir = normalize(Ld.xy);
        float tanE = Ld.z / length(Ld.xy);
        /* shadows: march toward the lamp through the height buffer */
        float h0 = c0.x, sh = 0.0;
        float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        for (int i = 0; i < 8; i++) {
            float t = (float(i) + jit) / 8.0;
            float s = uSMax * t * t + 1.5 * px;
            float hq = G(p + dir * s);
            sh = max(sh, clamp((hq - h0 - s * tanE - 0.25) / (0.3 * s + 1.2), 0.0, 1.0));
        }
        sh = smoothstep(0.0, 1.0, sh);
        /* ambient occlusion: lower than the paper around it */
        float hav = 0.25 * (G(p + vec2(uAOR, 0.0)) + G(p - vec2(uAOR, 0.0)) + G(p + vec2(0.0, uAOR)) + G(p - vec2(0.0, uAOR)));
        float ao = 1.0 - 0.45 * clamp((hav - h0) / (1.4 * uAOR), 0.0, 1.0);

        /* the paper itself, pinned to the sheet */
        vec2 pp = p + disp;
        vec4 t0 = texture(uPaper, pp * (1.3 / 1024.0));
        float h1 = texture(uPaper, (pp + dir * 0.9) * (1.3 / 1024.0)).r;
        float emb = (t0.r - h1);
        float dye = texture(uNoise, pp / (1.3 * uL) * (1.0 / 16.0)).r;
        vec3 c = mix(uCol, uCol2, smoothstep(0.25, 0.75, dye));
        c *= (0.965 + 0.07 * t0.g);
        c = mix(c, min(c * 1.07 + 0.025, 1.0), t0.b * uPal.y);

        /* the lamp: key from the side, a soft sky, light through the sheet */
        float dif = dot(n, Ld);
        float key = max(dif, 0.0) * (1.0 - uPal.w * sh);
        float thru = max(-dif, 0.0) * 0.18;
        float amb = 0.30 + 0.12 * n.z;
        float ref = 0.30 + 0.12 + Ld.z;
        /* the shade is lit by the paper around it: a touch of its colour */
        float lc = dot(c, vec3(0.3, 0.59, 0.11));
        vec3 bounce = mix(vec3(1.0, 0.975, 0.94), c / max(lc, 0.05), 0.22);
        vec3 col = c * ((amb + thru) * bounce * ao + key * mix(1.0, ao, 0.5)) / ref;
        col *= 1.0 + uEmb * emb * (0.5 + 0.5 * max(dif, 0.0));
        /* the creases: a mountain fold whitens a little where the dye cracks
         * and catches the light; a valley is a dark hairline */
        col = mix(col, max(col, vec3(0.9) * (amb + key) / ref), crease * ridge * uPal.x * 0.35);
        col *= 1.0 - crease * valley * 0.22;
        col *= 1.0 - crease * 0.05 - mem * 0.07;
        /* foil papers: a broad sheen */
        vec3 hv = normalize(Ld + vec3(0.0, 0.0, 1.0));
        col += uPal.z * pow(max(dot(n, hv), 0.0), 24.0) * (1.0 - uPal.w * sh) * vec3(1.0, 0.92, 0.75);
        /* the lamp's fall-off */
        col *= max(1.10 - uLamp.z * length((p - uLamp.xy) / uL), 0.80);
        /* highlights roll off toward white, keeping their hue */
        float lm = dot(col, vec3(0.3, 0.59, 0.11));
        float lt = lm > 0.8 ? 0.8 + (lm - 0.8) / (1.0 + (lm - 0.8) * 2.5) : lm;
        col *= lt / max(lm, 1e-4);
        col = mix(col, vec3(lt), clamp((lm - 0.85) * 1.5, 0.0, 0.35));
        col += (h21(gl_FragCoord.xy + 7.0) - 0.5) / 255.0;
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

static GLuint prog_mesh, prog_comp, prog_text;
static GLuint vao_empty, vao_text, vbo_text;
static GLuint tex_atlas, tex_noise, tex_paper;
static GLuint fbo_g, tex_g0, tex_g1, tex_gh;
static int gw_, gh_, g_m, g_mr;         /* G-buffer size, margin to the upper left and lower right (device px) */
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
    /* B: washi's long kozo fibres, a few hundred pixels long, wandering */
    float *fb = calloc((size_t)PT * PT, sizeof(float));
    for (int f = 0; f < 1400; f++) {
        double x = rnd(&r) * PT, y = rnd(&r) * PT, a = rnd(&r) * 2 * M_PI;
        int steps = 60 + (int)(rnd(&r) * 260);
        float amp = (float)(0.35 + 0.65 * rnd(&r)), wd = (float)(0.5 + 0.9 * rnd(&r));
        double bend = (rnd(&r) - 0.5) * 0.02;
        for (int s = 0; s < steps; s++) {
            x += 0.7 * cos(a); y += 0.7 * sin(a);
            a += bend + (rnd(&r) - 0.5) * 0.05;
            if (rnd(&r) < 0.01) bend = (rnd(&r) - 0.5) * 0.03;
            int ix = (int)floor(x), iy = (int)floor(y);
            for (int dy = -1; dy <= 2; dy++)
                for (int dx = -1; dx <= 2; dx++) {
                    double ddx = ix + dx - x, ddy = iy + dy - y;
                    float wgt = (float)exp(-(ddx * ddx + ddy * ddy) / wd);
                    float *q = &fb[(((iy + dy) % PT + PT) % PT) * PT + (((ix + dx) % PT + PT) % PT)];
                    if (amp * wgt > *q) *q = amp * wgt;
                }
        }
    }
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
            o[2] = (unsigned char)(clamp01(fb[y * PT + x]) * 255); o[3] = 255;
        }
    glGenTextures(1, &tex_paper);
    glBindTexture(GL_TEXTURE_2D, tex_paper);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PT, PT, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
    glGenerateMipmap(GL_TEXTURE_2D);
    tex_params(GL_REPEAT, 1);
    free(t); free(h); free(fb);
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
        fprintf(stderr, "origami: shader:\n%s\n", log);
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

/* ---- palettes: the paper ---- */
static const struct palette {
    const char *name;
    float col[3], col2[3];             /* the sheet, and the colour its dye drifts to */
    float whiten, fibre, sheen, shadow, emboss;
} palettes[] = {
    {"paper",  {0.930f, 0.922f, 0.900f}, {0.945f, 0.935f, 0.915f}, 0.00f, 0.10f, 0.00f, 0.55f, 0.30f},
    {"washi",  {0.905f, 0.865f, 0.770f}, {0.925f, 0.890f, 0.800f}, 0.10f, 0.40f, 0.00f, 0.50f, 0.25f},
    {"kraft",  {0.660f, 0.515f, 0.355f}, {0.610f, 0.470f, 0.320f}, 0.30f, 0.20f, 0.00f, 0.55f, 0.35f},
    {"indigo", {0.150f, 0.225f, 0.400f}, {0.200f, 0.300f, 0.500f}, 0.55f, 0.30f, 0.00f, 0.50f, 0.25f},
    {"rose",   {0.880f, 0.560f, 0.545f}, {0.920f, 0.650f, 0.570f}, 0.40f, 0.15f, 0.00f, 0.52f, 0.25f},
    {"sage",   {0.545f, 0.625f, 0.495f}, {0.600f, 0.660f, 0.520f}, 0.35f, 0.30f, 0.00f, 0.52f, 0.28f},
    {"ink",    {0.195f, 0.195f, 0.210f}, {0.225f, 0.215f, 0.215f}, 0.50f, 0.25f, 0.10f, 0.50f, 0.30f},
    {"brass",  {0.700f, 0.545f, 0.300f}, {0.600f, 0.450f, 0.240f}, 0.15f, 0.00f, 0.35f, 0.55f, 0.20f},
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;
static float cap_col[3], cap_halo[3];

static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) break;
        if (pal == NPAL) {
            fprintf(stderr, "origami: no palette \"%s\" (paper, washi, kraft, indigo, rose, sage, ink, brass)\n", opt_palette);
            exit(1);
        }
    } else {
        pal = (int)(mix64(opt_seed ^ 0x9a1e77ull) % NPAL);
    }
    const struct palette *P = &palettes[pal];
    /* the caption: light on dark paper, dark on light */
    float lum = 0.3f * P->col[0] + 0.59f * P->col[1] + 0.11f * P->col[2];
    for (int c = 0; c < 3; c++) {
        cap_col[c] = lum < 0.45f ? 0.96f - 0.04f * c : 0.16f - 0.02f * c;
        cap_halo[c] = P->col[c] * 0.9f;
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
    prog_mesh = mk_prog(VS_MESH, FS_MESH);
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
    glGenTextures(1, &tex_g0);
    glGenTextures(1, &tex_g1);
    glGenTextures(1, &tex_gh);
    glGenFramebuffers(1, &fbo_g);
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

/* the G-buffer: full resolution, with a margin for shadows cast from just
 * off the screen. The lamp is always to the upper left, so that side gets
 * room for the longest shadow and the other two just a little. G0 height,
 * normal.xy, which pattern; G1 where the paper has moved from, and the
 * normal it would have folded right up (where its creases are) */
static void gbuf_resize(int ml, int mr) {
    int gw = W + ml + mr, gh = H + ml + mr;
    if (gw == gw_ && gh == gh_ && ml == g_m) return;
    gw_ = gw; gh_ = gh; g_m = ml; g_mr = mr;
    glBindTexture(GL_TEXTURE_2D, tex_g0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, gw, gh, 0, GL_RGBA, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, 0);
    glBindTexture(GL_TEXTURE_2D, tex_g1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gw, gh, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    tex_params(GL_CLAMP_TO_EDGE, 0);
    glBindTexture(GL_TEXTURE_2D, tex_gh);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, gw, gh, 0, GL_RED, GL_HALF_FLOAT, NULL);
    tex_params(GL_CLAMP_TO_EDGE, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_g);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_g0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_g1, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, tex_gh, 0);
    static const GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("G-buffer incomplete");
}

#define BIAS 8192
/* the cells of a pattern's lattice that cover a world rectangle */
static long cell_range(const Scene *s, const double r[4], int c0[2], int *nx) {
    double cs = s->P[2][0], sn = s->P[2][1], ox = s->P[2][2], oy = s->P[2][3];
    double a = s->P[0][0], b = s->P[0][1];
    double i0 = 1e300, i1 = -1e300, j0 = 1e300, j1 = -1e300;
    for (int k = 0; k < 4; k++) {
        double x = r[(k & 1) ? 2 : 0] - ox, y = r[(k & 2) ? 3 : 1] - oy;
        double lx = cs * x + sn * y, ly = -sn * x + cs * y, i, j;
        if (s->kind == S_HEX) { j = ly / (0.8660254 * a); i = lx / a - 0.5 * j; }
        else { i = lx / a; j = ly / b; }
        i0 = fmin(i0, i); i1 = fmax(i1, i); j0 = fmin(j0, j); j1 = fmax(j1, j);
    }
    int pj = 2, pi = 2;
    if (s->kind <= S_PLEATS) pj += (int)ceil((fabs(s->P[0][3]) + fabs(s->P[1][0])) * (1 + s->P[3][0]) / b);
    c0[0] = (int)floor(i0) - pi + BIAS; c0[1] = (int)floor(j0) - pj + BIAS;
    *nx = (int)ceil(i1) - (int)floor(i0) + 2 * pi;
    int ny = (int)ceil(j1) - (int)floor(j0) + 2 * pj;
    if ((long)*nx * ny > 600000) ny = 600000 / *nx;
    return (long)*nx * ny;
}

static void draw_mesh(const Scene *s, const float sw[4], float F, float w, float side, const float clr[4], const float clr2[2],
                      double t, const double rect[4]) {
    GLuint pr = prog_mesh;
    int c0[2], nx;
    long n = cell_range(s, rect, c0, &nx);
    glUniform1i(U(pr, "uKind"), s->kind);
    glUniform1i(U(pr, "uVPC"), s->vpc);
    glUniform1i(U(pr, "uNX"), nx);
    glUniform2i(U(pr, "uC0"), c0[0], c0[1]);
    glUniform4fv(U(pr, "uP"), 4, &s->P[0][0]);
    glUniform4fv(U(pr, "uLv"), 3, &s->Lv[0][0]);
    glUniform4fv(U(pr, "uSw"), 1, sw);
    glUniform4f(U(pr, "uFront"), F, w, side, (float)(1.2 * (s->kind == S_PLEATS ? s->P[0][1] : fmax(s->P[0][0], s->P[0][1]))));
    glUniform4fv(U(pr, "uClr"), 1, clr);
    glUniform2fv(U(pr, "uClr2"), 1, clr2);
    glUniform1f(U(pr, "uT"), (float)fmod(t, 20000.0));
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(n * s->vpc));
}

static double max_height(const Scene *s) { return s->P[0][2]; }

static void render(double t, double wall, GLuint target) {
    if (t < 0) t = 0;
    view_update();
    long a = (long)floor(t / ACT);
    double ts = t - a * ACT, u = clamp01((ts - (ACT - MORPH)) / MORPH);
    Scene A, B;
    scene_params(a, &A);
    if (u > 0) scene_params(a + 1, &B);
    else B = A;

    /* the lamp: low, from the upper left, swinging a little over ten minutes */
    double az = atan2(-0.62, -0.78) + 0.40 * sin(t * 2 * M_PI / 600.0);
    double el = 0.50 + 0.06 * sin(t * 2 * M_PI / 840.0);
    double hmax = fmax(max_height(&A), max_height(&B));
    double smax = 1.1 * hmax / tan(el);
    double aor = 0.4 * (A.kind == S_PLEATS ? A.P[0][1] : fmin(A.P[0][0], A.P[0][1]));
    /* the margin: room for the longest shadow the lamp will throw (at its lowest) */
    gbuf_resize((int)ceil((hmax / tan(0.44) + 4) * v_sc), (int)ceil((aor + 4) * v_sc));

    /* the text: a clearing opens in the folds, and the letters rise in it */
    float clr[4] = {0, 0, 1, 1}, clr2[2] = {1, 0}, none2[2] = {1, 0};
    double temb = 0;
    int tk = act_text(a);
    if (tk != T_NONE && ts >= TXT_IN && ts <= TXT_END) {
        build_text(a, tk);
        if (txt_ok) {
            double open = ts < TXT_OUT + TXT_RISE ? ease((ts - TXT_IN) / TXT_OPEN) : 1 - ease((ts - TXT_OUT - TXT_RISE) / TXT_OPEN);
            temb = ts < TXT_OUT ? ease((ts - TXT_IN - 0.7 * TXT_OPEN) / TXT_RISE) : 1 - ease((ts - TXT_OUT) / TXT_RISE);
            double pad = 0.06 * L + 1.2 * (A.kind == S_PLEATS ? A.P[0][1] : fmax(A.P[0][0], A.P[0][1]));
            clr[0] = (txt_ink[0] + txt_ink[2]) / 2; clr[1] = (txt_ink[1] + txt_ink[3]) / 2;
            clr[2] = (float)((txt_ink[2] - txt_ink[0]) / 2 + pad); clr[3] = (float)((txt_ink[3] - txt_ink[1]) / 2 + pad);
            clr2[0] = (float)(0.12 * L); clr2[1] = (float)open;
        }
    }

    /* the sheet */
    double rect[4] = {v_ox - g_m / v_sc, v_oy - g_m / v_sc, v_ox + (W + g_mr) / v_sc, v_oy + (H + g_mr) / v_sc};
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_g);
    glViewport(0, 0, gw_, gh_);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog_mesh);
    glUniform3f(U(prog_mesh, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(prog_mesh, "uGRes"), (float)gw_, (float)gh_);
    glUniform1f(U(prog_mesh, "uM"), (float)g_m);
    glUniform1f(U(prog_mesh, "uL"), (float)L);
    glUniform1i(U(prog_mesh, "uNoise"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    if (u > 0) {
        /* the sweep: ahead of the front the old folds relax, behind it the
         * new ones rise; both are flat where it passes */
        double lo, hi, w = 0.22 * L, na = B.Sw[3] + 0.3 * L;
        sweep_range(&B, &lo, &hi);
        double uu = u * u * (3 - 2 * u);
        float F = (float)(lo - w - na + (hi - lo + 2 * w + 2 * na) * uu);
        draw_mesh(&A, B.Sw, F, (float)w, 1, clr, clr2, t, rect);
        draw_mesh(&B, B.Sw, F, (float)w, -1, clr, none2, t, rect);
    } else {
        draw_mesh(&A, A.Sw, 0, 1, 0, clr, clr2, t, rect);
    }

    /* the light */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    GLuint pr = prog_comp;
    glUseProgram(pr);
    const struct palette *P = &palettes[pal];
    glUniform3f(U(pr, "uView"), (float)v_ox, (float)v_oy, (float)v_sc);
    glUniform2f(U(pr, "uRes"), (float)W, (float)H);
    glUniform2f(U(pr, "uGRes"), (float)gw_, (float)gh_);
    glUniform1f(U(pr, "uM"), (float)g_m);
    glUniform1f(U(pr, "uL"), (float)L);
    glUniform3f(U(pr, "uLd"), (float)(cos(el) * cos(az)), (float)(cos(el) * sin(az)), (float)sin(el));
    glUniform3fv(U(pr, "uCol"), 1, P->col);
    glUniform3fv(U(pr, "uCol2"), 1, P->col2);
    glUniform4f(U(pr, "uPal"), P->whiten, P->fibre, P->sheen, P->shadow);
    glUniform1f(U(pr, "uEmb"), P->emboss);
    glUniform1f(U(pr, "uSMax"), (float)smax);
    glUniform1f(U(pr, "uAOR"), (float)aor);
    glUniform4f(U(pr, "uTBox"), txt_box[0], txt_box[1], txt_box[2], txt_box[3]);
    glUniform2f(U(pr, "uTxt"), (float)(0.0022 * L * temb), (float)(0.0035 * L));
    glUniform3f(U(pr, "uLamp"), (float)(fr[0] + 0.1 * fr[2]), (float)(fr[1] - 0.2 * fr[3]), 0.13f);
    glUniform1i(U(pr, "uG0"), 1);
    glUniform1i(U(pr, "uG1"), 2);
    glUniform1i(U(pr, "uNoise"), 0);
    glUniform1i(U(pr, "uPaper"), 3);
    glUniform1i(U(pr, "uText"), 4);
    glUniform1i(U(pr, "uGH"), 5);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_g0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_g1);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_paper);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_text);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_gh);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "origami");
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
        "usage: origami [--window | --root | --layer] [--seed N] [--fps N]\n"
        "               [--palette paper|washi|kraft|indigo|rose|sage|ink|brass]\n"
        "               [--scene miura|pleats|waterbomb|hex|twist] [--no-caption]\n"
        "               [--text WORD] [--font FAMILY] [--schedule]\n"
        "               [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "               [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "  --palette   the paper; default: picked from the seed\n"
        "  --scene     hold one pattern (re-folded every %.0f s) instead of cycling\n"
        "  --text      what the host-name embossing spells instead of the host name\n"
        "  --fps       frame rate (default 30: it moves slowly)\n"
        "  --schedule  print what happens when (seconds after the epoch) and exit\n", ACT);
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
        printf("%7.1f s  act %2ld: %-10s%s  (sweeps into %s from %.1f s)\n", a * ACT, a, scene_name[act_kind(a)],
               tk == T_TIME ? "  + the time" : tk == T_HOST ? "  + the host name" : "",
               scene_name[act_kind(a + 1)], (a + 1) * ACT - MORPH);
    }
    printf("text: a clearing opens from %.1f s into its act, letters risen by %.1f s, sink from %.1f s, closed by %.1f s\n",
           TXT_IN, TXT_IN + 0.7 * TXT_OPEN + TXT_RISE, TXT_OUT, TXT_END);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
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
        if (forced_scene == NSCENE) { fprintf(stderr, "origami: no scene \"%s\" (miura, pleats, waterbomb, hex, twist)\n", opt_scene); exit(1); }
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
        /* a window gets its real size from the WM: lay the sheet out for
         * the screen it will most likely fill */
        dw = W; dh = H;
        if (!opt_root && !offscreen) { dw = DisplayWidth(dpy, scr); dh = DisplayHeight(dpy, scr); }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Origami");
        XClassHint chint = {"origami", "Origami"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    domain_init(dw, dh);
    gl_init();

    /* --epoch: when the sheet was first folded, shared by every instance; a
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
            fprintf(stderr, "origami: %s, %dx%d, %s (act %ld), first frame %.0f ms\n", palettes[pal].name, W, H,
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
        printf("origami: %s, %s, %dx%d, %d frames from %.1f s: avg %.3f ms, worst %.3f ms per frame\n",
               palettes[pal].name, scene_name[act_kind(a)], W, H, opt_bench, opt_at, sum / opt_bench, worst);
        printf("origami:   GPU %.3f ms (best %.3f); G-buffer %dx%d\n", gsum / opt_bench, gmin, gw_, gh_);
        printf("origami: GL renderer: %s (first frame %.0f ms)\n", (const char *)glGetString(GL_RENDERER), first);
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
