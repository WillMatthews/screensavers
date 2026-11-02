/*
 * atc — an air traffic control radar scope, with the real traffic overhead.
 *
 * A dark, modern ATC display: range rings and a bearing scale around the
 * receiver (or the configured location), a faint graticule, airports with
 * their runways and extended approach centrelines (ticked every mile), and
 * every aircraft as a target with its history dots, a one-minute speed
 * vector, and a data block (callsign, flight level with a climb/descend
 * arrow and the selected level, ground speed, type) on a leader line. The
 * blocks arrange themselves to keep clear of each other. Colours: white and
 * grey for normal traffic, amber levels for aircraft climbing or
 * descending, cyan for the one being followed, red for 7700/7600 (7500 just
 * shows its code, quietly), gold for airships. A faint sweep turns, and
 * every so often the scope zooms in to follow something interesting: an
 * aircraft on final, an emergency, and above all an airship (lighter than
 * air, category B2), with the Goodyear ones (N1A, N2A, N3A) the star.
 * Lock info: local and UTC clocks, the time away (swaylock's start time),
 * and the hostname as the sector name.
 *
 * Where the traffic comes from, in order:
 *   1. a local ADS-B receiver: --feed URL, else readsb/dump1090's
 *      http://127.0.0.1:8080/data/aircraft.json or
 *      http://127.0.0.1/tar1090/data/aircraft.json if one answers;
 *      fetched every 2 s
 *   2. else, only if a location is configured (--lat/--lon, or "LAT LON" in
 *      ~/.config/wallpapers/location, as nightsky), the public adsb.lol API:
 *      https://api.adsb.lol/v2/point/LAT/LON/RADIUS with LAT/LON rounded to
 *      two decimals, every 15 s (backing off on errors and HTTP 429); and,
 *      every 5 minutes, /v2/hex/ for the three Goodyear airships (which
 *      sends nothing about you). Data credited on screen ("data: adsb.lol").
 *   3. else simulated traffic: arrivals, departures, overflights, the odd
 *      light aircraft and, now and then, an airship loitering over a stadium,
 *      around the airports near the location (London when there's none).
 * Fetches are curl processes in the background (never blocking a frame),
 * written to a shared cache, $XDG_CACHE_HOME/wallpapers/atc.json, under a
 * flock, so one instance per monitor still makes one request. Between fixes
 * targets are dead-reckoned and corrections eased in. --sim (or --offline)
 * never touches the network; --shot and --bench never fetch (they use the
 * simulation, or --replay FILE, a saved aircraft.json).
 *
 * Spanning: one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout is the scope; portrait outputs
 * are a flight strip board (paper strips in coloured holders: buff
 * arrivals, blue departures, pink overflights, racked in bays, with
 * handwritten cleared levels, runways and ticks; strips slide in and out as
 * aircraft come and go); other landscape outputs show the approach to the
 * nearest big airport beside an arrivals/departures board, then a wide-range
 * scope. Standalone, a portrait screen gets the scope over a bay of strips.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size.
 *
 * Keys (window mode): q / Escape quit.
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
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "proto-layer-shell.h"

extern char **environ;

#define PI 3.14159265f
#define DEG (PI / 180.0f)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;
static const char *opt_feed, *opt_replay;
static int opt_sim, opt_no_sweep, opt_no_follow, opt_role = -1;
static double opt_lat = NAN, opt_lon = NAN;
static int loc_given, have_loc;
static float opt_range = 40;           /* the scope's range ring, nm */
static float opt_radius = 80;          /* what to ask the public feed for, nm */
static const char *opt_font_mono = "JetBrains Mono,JetBrainsMono Nerd Font Mono,DejaVu Sans Mono";
static const char *opt_font_sans = "Inter,Cantarell,DejaVu Sans";
static const char *opt_font_hand = "Purisa,Comic Neue,DejaVu Sans";

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];
static int span_n, span_me = -1;
enum { R_SCOPE, R_STRIPS, R_APPROACH, R_WIDE, R_COMBO };

static void die(const char *what) { fprintf(stderr, "atc: %s\n", what); exit(1); }

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
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float wrap180(float a) { a = fmodf(a + 180, 360); if (a < 0) a += 360; return a - 180; }
static float wrap360(float a) { a = fmodf(a, 360); return a < 0 ? a + 360 : a; }

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return x;
}
/* a little generator seeded per thing, so every instance draws the same */
typedef struct { uint32_t s; } Rng;
static float rnd(Rng *r) { r->s ^= r->s << 13; r->s ^= r->s >> 17; r->s ^= r->s << 5; return (r->s >> 8) * (1.0f / 16777216.0f); }
static float rndr(Rng *r, float a, float b) { return a + (b - a) * rnd(r); }
static int rndi(Rng *r, int n) { int i = (int)(rnd(r) * n); return i < n ? i : n - 1; }

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0;
    char *b = malloc(cap + 1);
    for (;;) {
        size_t r = fread(b + n, 1, cap - n, f);
        n += r;
        if (n < cap || cap >= (64u << 20)) break;
        b = realloc(b, (cap *= 2) + 1);
    }
    fclose(f);
    b[n] = 0;
    if (len) *len = n;
    return b;
}

/* ---- where: --lat/--lon, else ~/.config/wallpapers/location (as nightsky) ---- */
static void load_location(void) {
    if (loc_given) { have_loc = 1; return; }
    char path[1024];
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xdg && *xdg) snprintf(path, sizeof path, "%s/wallpapers/location", xdg);
    else if (home) snprintf(path, sizeof path, "%s/.config/wallpapers/location", home);
    else return;
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        for (char *c = line; *c; c++) if (*c == ',') *c = ' ';
        double la, lo;
        if (line[0] != '#' && sscanf(line, "%lf %lf", &la, &lo) == 2 && fabs(la) <= 90 && fabs(lo) <= 360) {
            opt_lat = la; opt_lon = lo > 180 ? lo - 360 : lo;
            have_loc = 1;
            break;
        }
    }
    fclose(f);
}

/* ---- lock info ---- */
static char host[64], sector[64];
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
        if (p && sscanf(p + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &start) == 1) {
            time_t t = btime + (time_t)(start / (unsigned long long)hz);
            if (!best || t < best) best = t;
        }
    }
    if (d) closedir(d);
    return best;
}

/* ---- glyph atlas: printable ASCII plus a few symbols, in each text style
 * used, rasterised at its exact pixel size ---- */
#define S_DEG "\x7f"
#define S_UP "\x80"
#define S_DN "\x81"
#define S_DOT "\x82"
#define S_RARR "\x83"
#define S_LARR "\x84"
#define NGLYPH 101
static const FT_ULong extra_cp[] = {0xB0, 0x2191, 0x2193, 0xB7, 0x2192, 0x2190};
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGLYPH]; } Style;
enum { F_MONO, F_MONOB, F_SANS, F_SANSB, F_HAND, NFACE };
#define MAXSTYLE 64
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static FT_Face load_face(const char *fam, const char *weight) {
    char pat_s[400];
    snprintf(pat_s, sizeof pat_s, "%s:weight=%s:fontformat=CFF,TrueType", fam, weight);
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
    faces[F_SANS] = load_face(opt_font_sans, "medium");
    faces[F_SANSB] = load_face(opt_font_sans, "semibold");
    faces[F_HAND] = load_face(opt_font_hand, "regular");
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 7) p = 7;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) die("too many text styles");
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    styles[nstyles].g[0].adv = -1;     /* not rasterised yet */
    return nstyles++;
}

static int atlas_dirty;
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
            FT_ULong code = c >= 95 ? extra_cp[c - 95] : (FT_ULong)(32 + c);
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
    atlas_dirty = 1;
}

static int gidx(unsigned char c) { return c >= 0x7f && c < 0x7f + 6 ? 95 + (c - 0x7f) : c >= 32 && c < 127 ? c - 32 : 0; }
static const Glyph *glyph(int s, unsigned char c) { return &styles[s].g[gidx(c)]; }

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p; p++) w += glyph(s, (unsigned char)*p)->adv + (p[1] ? track : 0);
    return w;
}
static float cap_h(int s) { return glyph(s, 'H')->top; }

/* ---- the vector stream: every shape is a quad whose fragment shader works
 * out its own anti-aliased coverage from a signed distance ----
 * per vertex: x, y (px), local x, y, 4 shape params, rgba, kind */
enum { K_LINE, K_CIRC, K_TEXT, K_RECT, K_SHADOW, K_ELLIPSE, K_GLOW };
#define VF 13
static float *vb;
static int vbn, vbcap;

static void vtx(float x, float y, float lx, float ly, const float *p, const float *c, int k) {
    if (vbn + VF > vbcap) vb = realloc(vb, (size_t)(vbcap = vbcap ? vbcap * 2 : 1 << 16) * sizeof *vb);
    float *o = vb + vbn;
    o[0] = x; o[1] = y; o[2] = lx; o[3] = ly;
    o[4] = p[0]; o[5] = p[1]; o[6] = p[2]; o[7] = p[3];
    o[8] = c[0]; o[9] = c[1]; o[10] = c[2]; o[11] = c[3];
    o[12] = (float)k;
    vbn += VF;
}

/* a quad centred on (cx, cy), turned by ang (radians, clockwise on screen),
 * half extents hx, hy; local coords run -hx..hx, -hy..hy */
static void shape(float cx, float cy, float ang, float hx, float hy, const float *p, const float *c, int k) {
    float cs = cosf(ang), sn = sinf(ang);
    float L[4][2] = {{-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}};
    float P[4][2];
    for (int i = 0; i < 4; i++) {
        P[i][0] = cx + L[i][0] * cs - L[i][1] * sn;
        P[i][1] = cy + L[i][0] * sn + L[i][1] * cs;
    }
    int ix[6] = {0, 1, 2, 0, 2, 3};
    for (int i = 0; i < 6; i++) vtx(P[ix[i]][0], P[ix[i]][1], L[ix[i]][0], L[ix[i]][1], p, c, k);
}

static float alpha_mul = 1;            /* fades whole groups */
#define C4(r, g, b, a) ((const float[4]){(r), (g), (b), (a)})
static const float *ca(const float *c, float a) {
    static float buf[16][4];
    static int bi;
    float *o = buf[bi++ & 15];
    o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = c[3] * a;
    return o;
}
static const float *am(const float *c) { return alpha_mul == 1 ? c : ca(c, alpha_mul); }

/* a capsule from a to b, w px wide; dash > 0: dashes of that period, on for
 * the fraction duty */
static void line_d(float x0, float y0, float x1, float y1, float w, const float *c, float dash, float duty) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1e-4f) { dx = 1; dy = 0; } else { dx /= l; dy /= l; }
    float hw = w * 0.5f;
    float p[4] = {l * 0.5f, hw, dash, duty};
    shape((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, atan2f(dy, dx), l * 0.5f + hw + 1.5f, hw + 1.5f, p, am(c), K_LINE);
}
static void line(float x0, float y0, float x1, float y1, float w, const float *c) { line_d(x0, y0, x1, y1, w, c, 0, 0); }

static void disc(float x, float y, float r, const float *c) {
    float p[4] = {r, 0, 0, 0};
    shape(x, y, 0, r + 1.5f, r + 1.5f, p, am(c), K_CIRC);
}
static void ring(float x, float y, float r, float w, const float *c) {
    float p[4] = {r, w * 0.5f, 0, 0};
    shape(x, y, 0, r + w * 0.5f + 1.5f, r + w * 0.5f + 1.5f, p, am(c), K_CIRC);
}
/* a soft glow, additive: radius r, falling off */
static void glow(float x, float y, float r, const float *c) {
    float p[4] = {r, 0, 0, 0};
    shape(x, y, 0, r, r, p, am(c), K_GLOW);
}
/* materials for rects */
enum { M_FLAT, M_PAPER, M_HOLDER, M_BOARD, M_GLASS, M_TAPE, M_PANEL };
static void rect_r(float cx, float cy, float hw, float hh, float ang, float rad, const float *c, int mat, float seed) {
    float p[4] = {hw, hh, rad, (float)mat + fminf(fabsf(seed), 0.99f)};
    shape(cx, cy, ang, hw + 1.5f, hh + 1.5f, p, am(c), K_RECT);
}
static void rect(float x0, float y0, float x1, float y1, float rad, const float *c, int mat) {
    rect_r((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, (x1 - x0) * 0.5f, (y1 - y0) * 0.5f, 0, rad, c, mat, 0);
}
static void shadow(float x0, float y0, float x1, float y1, float rad, float blur, const float *c) {
    float hw = (x1 - x0) * 0.5f, hh = (y1 - y0) * 0.5f;
    float p[4] = {hw, hh, rad, blur};
    shape((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, 0, hw + blur * 2 + 1, hh + blur * 2 + 1, p, am(c), K_SHADOW);
}
static void frame(float x0, float y0, float x1, float y1, float w, const float *c) {
    line(x0, y0, x1, y0, w, c); line(x1, y0, x1, y1, w, c);
    line(x1, y1, x0, y1, w, c); line(x0, y1, x0, y0, w, c);
}
/* an ellipse, semi-axes a (along ang) and b; w > 0 an outline that wide */
static void ellipse(float cx, float cy, float a, float b, float ang, float w, const float *c) {
    float p[4] = {a, b, w * 0.5f, 0};
    shape(cx, cy, ang, a + w + 1.5f, b + w + 1.5f, p, am(c), K_ELLIPSE);
}

/* set str with its baseline at (x, y); align -1 left, 0 centred, 1 right;
 * ang turns the line about (x, y); jit > 0 wobbles each glyph like a hand */
static float text_ex(int s, float x, float y, int align, const char *str, const float *col, float track, float ang, uint32_t jit) {
    float w = text_width(s, str, track);
    float pen = -(align == 0 ? w * 0.5f : align > 0 ? w : 0);
    float cs = cosf(ang), sn = sinf(ang);
    if (ang == 0 && !jit) { x = roundf(x + pen) - pen; y = roundf(y); }
    const float *c = am(col);
    float px = (float)styles[s].px;
    uint32_t h = jit;
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph(s, (unsigned char)*p);
        if (g->w) {
            float dy = 0, ga = 0, gs = 1;
            if (jit) {
                h = hash32(h + 0x9e37u);
                dy = ((h & 255) / 255.0f - 0.5f) * px * 0.10f;
                ga = (((h >> 8) & 255) / 255.0f - 0.5f) * 0.12f;
                gs = 1 + (((h >> 16) & 255) / 255.0f - 0.5f) * 0.10f;
            }
            /* glyph box in the line's frame */
            float bx0 = pen + g->left, by0 = -g->top * gs + dy, bx1 = bx0 + g->w * gs, by1 = by0 + g->h * gs;
            float gc = cosf(ga), gsn = sinf(ga), mx = (bx0 + bx1) * 0.5f, my = by1;
            float L[4][2] = {{bx0, by0}, {bx1, by0}, {bx1, by1}, {bx0, by1}};
            float U[4][2] = {{g->x, g->y}, {g->x + g->w, g->y}, {g->x + g->w, g->y + g->h}, {g->x, g->y + g->h}};
            float P[4][2];
            for (int i = 0; i < 4; i++) {
                float lx = L[i][0] - mx, ly = L[i][1] - my;
                float rx = mx + lx * gc - ly * gsn, ry = my + lx * gsn + ly * gc;
                P[i][0] = x + rx * cs - ry * sn;
                P[i][1] = y + rx * sn + ry * cs;
            }
            static const float zp[4] = {0, 0, 0, 0};
            int ix[6] = {0, 1, 2, 0, 2, 3};
            for (int i = 0; i < 6; i++) vtx(P[ix[i]][0], P[ix[i]][1], U[ix[i]][0], U[ix[i]][1], zp, c, K_TEXT);
        }
        pen += g->adv + track;
    }
    return w;
}
static float text(int s, float x, float y, int align, const char *str, const float *col) {
    return text_ex(s, x, y, align, str, col, 0, 0, 0);
}
static float text_t(int s, float x, float y, int align, const char *str, const float *col, float track) {
    return text_ex(s, x, y, align, str, col, track, 0, 0);
}

/* ---- airports: a small table from memory. Each runway: its two ends'
 * names, its centre, the true heading from the first end to the second
 * (the direction of a landing on the first) and its length in metres.
 * Positions are good to a few hundred metres, plenty for a scope. ---- */
typedef struct { const char *a, *b; double lat, lon; float hdg, len; } RwyDef;
typedef struct { const char *icao, *name; double lat, lon; float elev; int rank; RwyDef r[6]; } AptDef;

static const AptDef APT[] = {
    /* UK and Ireland */
    {"EGLL", "HEATHROW", 51.4700, -0.4543, 83, 1, {{"09L", "27R", 51.4775, -0.4592, 89.7f, 3902}, {"09R", "27L", 51.4648, -0.4583, 89.7f, 3660}}},
    {"EGKK", "GATWICK", 51.1481, -0.1903, 202, 1, {{"08R", "26L", 51.1488, -0.1866, 77.7f, 3316}}},
    {"EGSS", "STANSTED", 51.8850, 0.2350, 348, 2, {{"04", "22", 51.8850, 0.2350, 42.6f, 3048}}},
    {"EGGW", "LUTON", 51.8747, -0.3683, 526, 2, {{"07", "25", 51.8747, -0.3683, 75.5f, 2162}}},
    {"EGLC", "LONDON CITY", 51.5053, 0.0553, 19, 3, {{"09", "27", 51.5049, 0.0522, 92.6f, 1508}}},
    {"EGMC", "SOUTHEND", 51.5714, 0.6956, 49, 3, {{"05", "23", 51.5714, 0.6956, 52.0f, 1856}}},
    {"EGLF", "FARNBOROUGH", 51.2758, -0.7763, 238, 3, {{"06", "24", 51.2758, -0.7763, 64.0f, 2440}}},
    {"EGCC", "MANCHESTER", 53.3537, -2.2750, 257, 1, {{"05L", "23R", 53.3567, -2.2836, 50.6f, 3048}, {"05R", "23L", 53.3497, -2.2700, 50.6f, 3050}}},
    {"EGGP", "LIVERPOOL", 53.3336, -2.8497, 80, 2, {{"09", "27", 53.3336, -2.8497, 87.0f, 2286}}},
    {"EGBB", "BIRMINGHAM", 52.4539, -1.7480, 327, 2, {{"15", "33", 52.4539, -1.7480, 150.7f, 3052}}},
    {"EGNX", "EAST MIDLANDS", 52.8311, -1.3281, 306, 2, {{"09", "27", 52.8311, -1.3281, 86.5f, 2893}}},
    {"EGNM", "LEEDS BRADFORD", 53.8659, -1.6606, 681, 3, {{"14", "32", 53.8659, -1.6606, 141.5f, 2250}}},
    {"EGNT", "NEWCASTLE", 55.0375, -1.6917, 266, 2, {{"07", "25", 55.0375, -1.6917, 69.4f, 2329}}},
    {"EGPH", "EDINBURGH", 55.9500, -3.3725, 135, 1, {{"06", "24", 55.9500, -3.3725, 64.6f, 2556}}},
    {"EGPF", "GLASGOW", 55.8719, -4.4331, 26, 2, {{"05", "23", 55.8719, -4.4331, 52.7f, 2665}}},
    {"EGPD", "ABERDEEN", 57.2019, -2.1978, 215, 2, {{"16", "34", 57.2019, -2.1978, 158.8f, 1953}}},
    {"EGPE", "INVERNESS", 57.5425, -4.0475, 31, 3, {{"05", "23", 57.5425, -4.0475, 52.0f, 1887}}},
    {"EGGD", "BRISTOL", 51.3827, -2.7191, 622, 2, {{"09", "27", 51.3827, -2.7191, 87.3f, 2011}}},
    {"EGFF", "CARDIFF", 51.3967, -3.3433, 220, 3, {{"12", "30", 51.3967, -3.3433, 121.0f, 2392}}},
    {"EGHI", "SOUTHAMPTON", 50.9503, -1.3568, 44, 3, {{"02", "20", 50.9503, -1.3568, 18.8f, 1723}}},
    {"EGHH", "BOURNEMOUTH", 50.7800, -1.8425, 38, 3, {{"08", "26", 50.7800, -1.8425, 77.0f, 2271}}},
    {"EGSH", "NORWICH", 52.6758, 1.2828, 117, 3, {{"09", "27", 52.6758, 1.2828, 89.0f, 1841}}},
    {"EGAA", "BELFAST", 54.6575, -6.2158, 268, 2, {{"07", "25", 54.6575, -6.2158, 69.5f, 2780}, {"17", "35", 54.6600, -6.2200, 167.8f, 1891}}},
    {"EIDW", "DUBLIN", 53.4213, -6.2701, 242, 1, {{"10R", "28L", 53.4213, -6.2701, 103.0f, 2637}, {"10L", "28R", 53.4300, -6.2580, 103.0f, 3110}, {"16", "34", 53.4280, -6.2620, 163.0f, 2072}}},
    /* Europe */
    {"EHAM", "SCHIPHOL", 52.3086, 4.7639, -11, 1, {{"18R", "36L", 52.3610, 4.7119, 183.0f, 3800}, {"18C", "36C", 52.3140, 4.7372, 183.0f, 3300}, {"18L", "36R", 52.3215, 4.7800, 183.0f, 3400}, {"06", "24", 52.2992, 4.7800, 57.9f, 3500}, {"09", "27", 52.3166, 4.7970, 87.0f, 3453}, {"04", "22", 52.3040, 4.7840, 41.0f, 2014}}},
    {"EBBR", "BRUSSELS", 50.9014, 4.4844, 184, 1, {{"07L", "25R", 50.9000, 4.4900, 66.0f, 3638}, {"07R", "25L", 50.8900, 4.4900, 66.0f, 3211}, {"01", "19", 50.8960, 4.4720, 15.0f, 2987}}},
    {"LFPG", "CHARLES DE GAULLE", 49.0097, 2.5479, 392, 1, {{"08L", "26R", 49.0250, 2.5300, 86.0f, 4215}, {"08R", "26L", 49.0205, 2.5400, 86.0f, 2700}, {"09L", "27R", 48.9950, 2.5600, 86.0f, 2700}, {"09R", "27L", 48.9910, 2.5650, 86.0f, 4200}}},
    {"LFPO", "ORLY", 48.7233, 2.3794, 291, 1, {{"06", "24", 48.7260, 2.3600, 61.0f, 3650}, {"08", "26", 48.7300, 2.3850, 75.0f, 3320}}},
    {"EDDF", "FRANKFURT", 50.0333, 8.5706, 364, 1, {{"07L", "25R", 50.0490, 8.5320, 70.0f, 2800}, {"07C", "25C", 50.0350, 8.5500, 70.0f, 4000}, {"07R", "25L", 50.0270, 8.5340, 70.0f, 4000}, {"18", "36", 50.0340, 8.5260, 180.0f, 4000}}},
    {"EDDM", "MUNICH", 48.3538, 11.7861, 1487, 1, {{"08L", "26R", 48.3640, 11.7720, 82.0f, 4000}, {"08R", "26L", 48.3410, 11.7900, 82.0f, 4000}}},
    {"EDDL", "DUSSELDORF", 51.2895, 6.7668, 147, 1, {{"05R", "23L", 51.2875, 6.7710, 53.0f, 3000}, {"05L", "23R", 51.2950, 6.7600, 53.0f, 2700}}},
    {"EDDK", "COLOGNE", 50.8659, 7.1427, 302, 2, {{"14L", "32R", 50.8659, 7.1427, 140.0f, 3815}, {"06", "24", 50.8700, 7.1300, 64.0f, 2459}}},
    {"EDDH", "HAMBURG", 53.6304, 9.9882, 53, 1, {{"05", "23", 53.6250, 9.9950, 53.0f, 3250}, {"15", "33", 53.6350, 9.9850, 153.0f, 3666}}},
    {"EDDB", "BERLIN", 52.3667, 13.5033, 157, 1, {{"07L", "25R", 52.3770, 13.5150, 70.0f, 3600}, {"07R", "25L", 52.3580, 13.5280, 70.0f, 4000}}},
    {"EDDS", "STUTTGART", 48.6899, 9.2220, 1276, 2, {{"07", "25", 48.6899, 9.2220, 72.0f, 3345}}},
    {"LSZH", "ZURICH", 47.4647, 8.5492, 1416, 1, {{"14", "32", 47.4750, 8.5450, 137.0f, 3300}, {"16", "34", 47.4580, 8.5600, 155.0f, 3700}, {"10", "28", 47.4580, 8.5450, 95.0f, 2500}}},
    {"LSGG", "GENEVA", 46.2381, 6.1089, 1411, 2, {{"04", "22", 46.2381, 6.1089, 47.0f, 3900}}},
    {"LOWW", "VIENNA", 48.1103, 16.5697, 600, 1, {{"11", "29", 48.1100, 16.5550, 115.0f, 3500}, {"16", "34", 48.1050, 16.5800, 165.0f, 3600}}},
    {"EKCH", "COPENHAGEN", 55.6181, 12.6561, 17, 1, {{"04L", "22R", 55.6150, 12.6450, 43.0f, 3600}, {"04R", "22L", 55.6100, 12.6650, 43.0f, 3300}, {"12", "30", 55.6250, 12.6500, 126.0f, 2800}}},
    {"ESSA", "ARLANDA", 59.6519, 17.9186, 137, 1, {{"01L", "19R", 59.6600, 17.9300, 0.0f, 3300}, {"01R", "19L", 59.6450, 17.9500, 0.0f, 2500}, {"08", "26", 59.6400, 17.9000, 80.0f, 2500}}},
    {"ENGM", "GARDERMOEN", 60.1939, 11.1004, 681, 1, {{"01L", "19R", 60.2050, 11.0850, 15.0f, 3600}, {"01R", "19L", 60.1900, 11.1100, 15.0f, 2950}}},
    {"EFHK", "HELSINKI", 60.3172, 24.9633, 179, 1, {{"04L", "22R", 60.3172, 24.9500, 43.0f, 3440}, {"04R", "22L", 60.3250, 24.9900, 43.0f, 3060}, {"15", "33", 60.3250, 24.9600, 147.0f, 2900}}},
    {"LEMD", "BARAJAS", 40.4719, -3.5626, 1998, 1, {{"14L", "32R", 40.4720, -3.5400, 143.0f, 3500}, {"14R", "32L", 40.4650, -3.5550, 143.0f, 4100}, {"18L", "36R", 40.4920, -3.5750, 181.0f, 3500}, {"18R", "36L", 40.4900, -3.5600, 181.0f, 4350}}},
    {"LEBL", "BARCELONA", 41.2971, 2.0785, 12, 1, {{"07L", "25R", 41.2950, 2.0850, 66.0f, 3352}, {"07R", "25L", 41.2880, 2.0750, 66.0f, 2660}, {"02", "20", 41.3000, 2.0950, 19.0f, 2528}}},
    {"LEPA", "PALMA", 39.5517, 2.7388, 27, 1, {{"06L", "24R", 39.5550, 2.7300, 58.0f, 3270}, {"06R", "24L", 39.5480, 2.7450, 58.0f, 3000}}},
    {"LPPT", "LISBON", 38.7813, -9.1359, 374, 1, {{"03", "21", 38.7813, -9.1359, 23.0f, 3805}}},
    {"LFMN", "NICE", 43.6584, 7.2159, 12, 2, {{"04L", "22R", 43.6600, 7.2200, 44.0f, 2960}, {"04R", "22L", 43.6550, 7.2100, 44.0f, 2570}}},
    {"LFLL", "LYON", 45.7256, 5.0811, 821, 2, {{"17L", "35R", 45.7256, 5.0900, 175.0f, 2670}, {"17R", "35L", 45.7256, 5.0750, 175.0f, 4000}}},
    {"LIRF", "FIUMICINO", 41.8003, 12.2389, 15, 1, {{"16L", "34R", 41.8250, 12.2600, 165.0f, 3900}, {"16R", "34L", 41.8100, 12.2300, 165.0f, 3900}, {"07", "25", 41.7950, 12.2400, 74.0f, 3300}}},
    {"LIMC", "MALPENSA", 45.6306, 8.7281, 768, 1, {{"17L", "35R", 45.6300, 8.7350, 172.0f, 3920}, {"17R", "35L", 45.6300, 8.7200, 172.0f, 3920}}},
    {"LGAV", "ATHENS", 37.9364, 23.9445, 308, 1, {{"03L", "21R", 37.9200, 23.9250, 36.0f, 3800}, {"03R", "21L", 37.9350, 23.9600, 36.0f, 4000}}},
    {"EPWA", "WARSAW", 52.1657, 20.9671, 362, 1, {{"11", "29", 52.1657, 20.9671, 107.0f, 2800}, {"15", "33", 52.1600, 20.9750, 147.0f, 3690}}},
    {"LKPR", "PRAGUE", 50.1008, 14.2600, 1247, 1, {{"06", "24", 50.1008, 14.2600, 65.0f, 3715}, {"12", "30", 50.1050, 14.2500, 124.0f, 3250}}},
    {"LHBP", "BUDAPEST", 47.4369, 19.2556, 495, 1, {{"13L", "31R", 47.4400, 19.2500, 133.0f, 3707}, {"13R", "31L", 47.4300, 19.2700, 133.0f, 3010}}},
    /* North America */
    {"KATL", "ATLANTA", 33.6407, -84.4277, 1026, 1, {{"08L", "26R", 33.6495, -84.4385, 90.0f, 2743}, {"08R", "26L", 33.6470, -84.4385, 90.0f, 3048}, {"09L", "27R", 33.6347, -84.4180, 90.0f, 3624}, {"09R", "27L", 33.6318, -84.4180, 90.0f, 2743}, {"10", "28", 33.6205, -84.4380, 90.0f, 2743}}},
    {"KJFK", "KENNEDY", 40.6413, -73.7781, 13, 1, {{"04L", "22R", 40.6420, -73.7900, 31.0f, 3682}, {"04R", "22L", 40.6400, -73.7600, 31.0f, 2560}, {"13L", "31R", 40.6550, -73.7750, 121.0f, 3048}, {"13R", "31L", 40.6430, -73.7930, 121.0f, 4442}}},
    {"KLGA", "LA GUARDIA", 40.7769, -73.8740, 21, 2, {{"04", "22", 40.7769, -73.8780, 44.0f, 2134}, {"13", "31", 40.7790, -73.8700, 134.0f, 2134}}},
    {"KEWR", "NEWARK", 40.6925, -74.1687, 18, 1, {{"04L", "22R", 40.6900, -74.1750, 39.0f, 3353}, {"04R", "22L", 40.6880, -74.1650, 39.0f, 3048}, {"11", "29", 40.7000, -74.1650, 107.0f, 2073}}},
    {"KBOS", "BOSTON", 42.3656, -71.0096, 20, 1, {{"04R", "22L", 42.3656, -71.0096, 35.0f, 3050}, {"15R", "33L", 42.3600, -71.0100, 149.0f, 3073}, {"09", "27", 42.3600, -71.0050, 92.0f, 2133}}},
    {"KPHL", "PHILADELPHIA", 39.8744, -75.2424, 36, 1, {{"09L", "27R", 39.8800, -75.2400, 84.0f, 2895}, {"09R", "27L", 39.8700, -75.2450, 84.0f, 3658}}},
    {"KIAD", "DULLES", 38.9531, -77.4565, 312, 1, {{"01C", "19C", 38.9531, -77.4565, 0.0f, 3505}, {"01R", "19L", 38.9400, -77.4400, 0.0f, 3505}, {"01L", "19R", 38.9500, -77.4700, 0.0f, 2865}, {"12", "30", 38.9400, -77.4600, 120.0f, 3200}}},
    {"KDCA", "REAGAN", 38.8512, -77.0402, 15, 2, {{"01", "19", 38.8512, -77.0380, 2.0f, 2094}}},
    {"KCLT", "CHARLOTTE", 35.2140, -80.9431, 748, 1, {{"18C", "36C", 35.2140, -80.9431, 180.0f, 3048}, {"18L", "36R", 35.2200, -80.9300, 180.0f, 2645}, {"18R", "36L", 35.2140, -80.9600, 180.0f, 2743}}},
    {"KMIA", "MIAMI", 25.7959, -80.2870, 8, 1, {{"08R", "26L", 25.7930, -80.2870, 90.0f, 3202}, {"08L", "26R", 25.8000, -80.2800, 90.0f, 2621}, {"09", "27", 25.7880, -80.2950, 90.0f, 2851}, {"12", "30", 25.7960, -80.2800, 120.0f, 2850}}},
    {"KFLL", "FORT LAUDERDALE", 26.0742, -80.1506, 9, 1, {{"10L", "28R", 26.0742, -80.1506, 90.0f, 2743}, {"10R", "28L", 26.0680, -80.1450, 90.0f, 2438}}},
    {"KPMP", "POMPANO BEACH", 26.2471, -80.1111, 19, 3, {{"15", "33", 26.2471, -80.1111, 150.0f, 1400}}},
    {"KMCO", "ORLANDO", 28.4312, -81.3081, 96, 1, {{"18L", "36R", 28.4312, -81.3250, 180.0f, 3659}, {"18R", "36L", 28.4312, -81.3330, 180.0f, 3048}, {"17L", "35R", 28.4312, -81.2900, 180.0f, 2743}, {"17R", "35L", 28.4312, -81.2830, 180.0f, 3659}}},
    {"KORD", "O'HARE", 41.9742, -87.9073, 672, 1, {{"10L", "28R", 41.9850, -87.9200, 90.0f, 3962}, {"10C", "28C", 41.9650, -87.9100, 90.0f, 3292}, {"10R", "28L", 41.9580, -87.9000, 90.0f, 2286}, {"09L", "27R", 41.9890, -87.8900, 90.0f, 2286}, {"09R", "27L", 41.9850, -87.8950, 90.0f, 2428}, {"04R", "22L", 41.9700, -87.8900, 41.0f, 2286}}},
    {"KMDW", "MIDWAY", 41.7868, -87.7522, 620, 2, {{"13C", "31C", 41.7868, -87.7522, 135.0f, 1988}, {"04R", "22L", 41.7868, -87.7522, 43.0f, 1964}}},
    {"KDTW", "DETROIT", 42.2162, -83.3554, 645, 1, {{"03L", "21R", 42.2162, -83.3700, 32.0f, 2591}, {"03R", "21L", 42.2162, -83.3600, 32.0f, 3048}, {"04L", "22R", 42.2162, -83.3450, 32.0f, 3659}, {"04R", "22L", 42.2162, -83.3400, 32.0f, 3048}}},
    {"KMSP", "MINNEAPOLIS", 44.8848, -93.2223, 841, 1, {{"12L", "30R", 44.8900, -93.2150, 118.0f, 3048}, {"12R", "30L", 44.8800, -93.2200, 118.0f, 3355}, {"17", "35", 44.8700, -93.2300, 173.0f, 2438}}},
    {"KCAK", "AKRON-CANTON", 40.9161, -81.4422, 1228, 3, {{"01", "19", 40.9161, -81.4422, 0.0f, 2286}, {"05", "23", 40.9161, -81.4422, 50.0f, 2000}}},
    {"KCLE", "CLEVELAND", 41.4117, -81.8498, 799, 1, {{"06L", "24R", 41.4117, -81.8550, 58.0f, 2743}, {"06R", "24L", 41.4050, -81.8450, 58.0f, 2743}}},
    {"KDFW", "DALLAS-FT WORTH", 32.8998, -97.0403, 607, 1, {{"17L", "35R", 32.8960, -97.0090, 180.0f, 2591}, {"17C", "35C", 32.9010, -97.0160, 180.0f, 4085}, {"17R", "35L", 32.9010, -97.0240, 180.0f, 4085}, {"18L", "36R", 32.9030, -97.0560, 180.0f, 4085}, {"18R", "36L", 32.9030, -97.0630, 180.0f, 4085}}},
    {"KIAH", "HOUSTON", 29.9902, -95.3368, 97, 1, {{"08L", "26R", 30.0000, -95.3300, 90.0f, 2743}, {"08R", "26L", 29.9950, -95.3300, 90.0f, 2866}, {"09", "27", 29.9800, -95.3400, 90.0f, 3048}, {"15L", "33R", 29.9900, -95.3550, 150.0f, 3658}, {"15R", "33L", 29.9900, -95.3500, 150.0f, 3048}}},
    {"KDEN", "DENVER", 39.8561, -104.6737, 5434, 1, {{"16R", "34L", 39.8800, -104.6950, 180.0f, 4877}, {"16L", "34R", 39.8800, -104.6700, 180.0f, 3658}, {"17R", "35L", 39.8350, -104.6950, 180.0f, 3658}, {"17L", "35R", 39.8450, -104.6550, 180.0f, 3658}, {"07", "25", 39.8400, -104.7000, 90.0f, 3658}, {"08", "26", 39.8630, -104.6400, 90.0f, 3658}}},
    {"KPHX", "PHOENIX", 33.4342, -112.0116, 1135, 1, {{"07L", "25R", 33.4370, -112.0050, 90.0f, 3139}, {"07R", "25L", 33.4300, -112.0050, 90.0f, 2377}, {"08", "26", 33.4250, -112.0100, 90.0f, 3502}}},
    {"KLAS", "LAS VEGAS", 36.0840, -115.1537, 2181, 1, {{"01L", "19R", 36.0800, -115.1500, 16.0f, 2739}, {"01R", "19L", 36.0800, -115.1450, 16.0f, 2979}, {"08L", "26R", 36.0870, -115.1500, 77.0f, 4423}, {"08R", "26L", 36.0800, -115.1500, 77.0f, 3208}}},
    {"KSLC", "SALT LAKE CITY", 40.7899, -111.9791, 4227, 1, {{"16L", "34R", 40.7899, -111.9700, 180.0f, 3659}, {"16R", "34L", 40.7899, -111.9850, 180.0f, 3659}, {"17", "35", 40.7800, -111.9600, 180.0f, 2925}}},
    {"KLAX", "LOS ANGELES", 33.9416, -118.4085, 128, 1, {{"06L", "24R", 33.9500, -118.4100, 83.0f, 2721}, {"06R", "24L", 33.9470, -118.4050, 83.0f, 3135}, {"07L", "25R", 33.9370, -118.4100, 83.0f, 3382}, {"07R", "25L", 33.9330, -118.4050, 83.0f, 3369}}},
    {"KSAN", "SAN DIEGO", 32.7338, -117.1933, 17, 2, {{"09", "27", 32.7338, -117.1933, 88.0f, 2865}}},
    {"KSFO", "SAN FRANCISCO", 37.6213, -122.3790, 13, 1, {{"10L", "28R", 37.6130, -122.3700, 118.0f, 3618}, {"10R", "28L", 37.6110, -122.3750, 118.0f, 3231}, {"01L", "19R", 37.6200, -122.3800, 28.0f, 2286}, {"01R", "19L", 37.6150, -122.3780, 28.0f, 2636}}},
    {"KSEA", "SEATTLE", 47.4502, -122.3088, 433, 1, {{"16L", "34R", 47.4502, -122.3020, 180.0f, 3627}, {"16C", "34C", 47.4502, -122.3110, 180.0f, 2873}, {"16R", "34L", 47.4502, -122.3180, 180.0f, 2591}}},
    {"CYYZ", "TORONTO", 43.6777, -79.6248, 569, 1, {{"05", "23", 43.6800, -79.6300, 57.0f, 3389}, {"06L", "24R", 43.6700, -79.6200, 67.0f, 2956}, {"06R", "24L", 43.6650, -79.6100, 67.0f, 2743}, {"15L", "33R", 43.6800, -79.6200, 140.0f, 3368}, {"15R", "33L", 43.6750, -79.6150, 140.0f, 2770}}},
    {"CYVR", "VANCOUVER", 49.1939, -123.1844, 14, 1, {{"08L", "26R", 49.2000, -123.1800, 100.0f, 3030}, {"08R", "26L", 49.1900, -123.1800, 100.0f, 3505}}},
    /* elsewhere */
    {"RJTT", "HANEDA", 35.5494, 139.7798, 21, 1, {{"16L", "34R", 35.5494, 139.7850, 157.0f, 3000}, {"16R", "34L", 35.5400, 139.7700, 157.0f, 3360}, {"04", "22", 35.5300, 139.7950, 37.0f, 2500}, {"05", "23", 35.5300, 139.8100, 47.0f, 2500}}},
    {"VHHH", "HONG KONG", 22.3080, 113.9185, 28, 1, {{"07L", "25R", 22.3080, 113.9050, 73.0f, 3800}, {"07R", "25L", 22.2980, 113.9200, 73.0f, 3800}}},
    {"WSSS", "CHANGI", 1.3644, 103.9915, 22, 1, {{"02L", "20R", 1.3600, 103.9850, 23.0f, 4000}, {"02C", "20C", 1.3500, 103.9900, 23.0f, 4000}}},
    {"OMDB", "DUBAI", 25.2532, 55.3657, 62, 1, {{"12L", "30R", 25.2600, 55.3600, 120.0f, 4447}, {"12R", "30L", 25.2450, 55.3650, 120.0f, 4000}}},
    {"YSSY", "SYDNEY", -33.9399, 151.1753, 21, 1, {{"16R", "34L", -33.9500, 151.1700, 169.0f, 3962}, {"16L", "34R", -33.9550, 151.1850, 169.0f, 2438}, {"07", "25", -33.9350, 151.1700, 65.0f, 2530}}},
};
#define NAPT_DEF ((int)(sizeof APT / sizeof APT[0]))

/* ---- the local plane: nm east (x) and north (y) of the centre ---- */
static double lat0 = 51.48, lon0 = -0.10, coslat0 = 0.623;

static void to_xy(double lat, double lon, double *x, double *y) {
    double dl = lon - lon0;
    while (dl > 180) dl -= 360;
    while (dl < -180) dl += 360;
    *x = dl * 60 * coslat0;
    *y = (lat - lat0) * 60;
}

/* great-circle distance (nm) and initial bearing (deg) from the centre */
static void gc_from_home(double lat, double lon, double *nm, double *brg) {
    double p1 = lat0 * DEG, p2 = lat * DEG, dl = (lon - lon0) * DEG;
    double a = sin((p2 - p1) / 2) * sin((p2 - p1) / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2);
    *nm = 2 * atan2(sqrt(a), sqrt(1 - a)) * 3440.065;
    *brg = fmod(atan2(sin(dl) * cos(p2), cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)) / DEG + 360, 360);
}

/* the airports near enough to draw, projected */
typedef struct {
    float x0, y0, x1, y1;              /* the ends: a's threshold, b's */
    float hdg, len_nm;
    const char *na, *nb;
    int act_a, act_b;                  /* in use for landings, in that direction */
    float use_a, use_b;                /* recent landings each way (decaying), from the traffic */
} Rwy;
typedef struct { const AptDef *d; float x, y, dist; int nr; Rwy r[6]; } Apt;
#define MAXAPT 48
static Apt apt[MAXAPT];
static int napt;

static void airports_init(void) {
    napt = 0;
    struct { int i; double d; } c[NAPT_DEF];
    int nc = 0;
    for (int i = 0; i < NAPT_DEF; i++) {
        double x, y;
        to_xy(APT[i].lat, APT[i].lon, &x, &y);
        double d = sqrt(x * x + y * y);
        if (d < 260) { c[nc].i = i; c[nc].d = d; nc++; }
    }
    /* nearest first */
    for (int a = 0; a < nc; a++)
        for (int b = a + 1; b < nc; b++)
            if (c[b].d < c[a].d) { __typeof__(c[0]) t = c[a]; c[a] = c[b]; c[b] = t; }
    for (int k = 0; k < nc && napt < MAXAPT; k++) {
        Apt *A = &apt[napt++];
        memset(A, 0, sizeof *A);
        A->d = &APT[c[k].i];
        double x, y;
        to_xy(A->d->lat, A->d->lon, &x, &y);
        A->x = (float)x; A->y = (float)y; A->dist = (float)c[k].d;
        for (int r = 0; r < 6 && A->d->r[r].a; r++) {
            const RwyDef *R = &A->d->r[r];
            Rwy *w = &A->r[A->nr++];
            double cx, cy;
            to_xy(R->lat, R->lon, &cx, &cy);
            float hl = R->len / 1852.0f * 0.5f, s = sinf(R->hdg * DEG), co = cosf(R->hdg * DEG);
            w->x0 = (float)cx - s * hl; w->y0 = (float)cy - co * hl;
            w->x1 = (float)cx + s * hl; w->y1 = (float)cy + co * hl;
            w->hdg = R->hdg; w->len_nm = hl * 2;
            w->na = R->a; w->nb = R->b;
        }
    }
}

/* runway end e (0: landing on a, heading hdg; 1: on b) of r: its threshold
 * and the landing direction */
static void rwy_end(const Rwy *w, int e, float *tx, float *ty, float *ux, float *uy, float *hdg) {
    float h = e ? w->hdg + 180 : w->hdg;
    *tx = e ? w->x1 : w->x0; *ty = e ? w->y1 : w->y0;
    *ux = sinf(h * DEG); *uy = cosf(h * DEG);
    if (hdg) *hdg = wrap360(h);
}

/* ---- the traffic ---- */
#define MAXAC 640
#define MAXHIST 32
enum { KD_NONE, KD_ARR, KD_DEP, KD_OVF, KD_GA, KD_SHIP };
typedef struct {
    int used, sim;
    uint32_t id;
    char hex[8], cs[10], reg[12], type[8], sqk[6];
    int cat;                           /* emitter category, e.g. 0xA3; 0xB2 lighter than air */
    int ship, goodyear;                /* an airship; one of Goodyear's */
    const char *ship_name;
    int ground, emerg;                 /* emerg: 7500/7600/7700 or 0 */
    /* the latest fix */
    double fx, fy, ft;
    float falt, fgs, ftrk, fvr, fsel;  /* fsel: selected/cleared altitude, -1 none */
    int has_vel;
    /* the display state, dead-reckoned, and the correction being eased out */
    double x, y;
    float alt, gs, trk, vr;
    double ex, ey;
    float ealt;
    double et;
    /* history dots, newest last */
    float hx[MAXHIST], hy[MAXHIST];
    int hn;
    double hlast;
    double first, last;                /* wall-clock: first seen, last fix */
    float fade;                        /* 0..1 in, and out once gone */
    int gone;
    /* the data block */
    float lang, lang_v;                /* drawn leader angle (rad), its spring */
    int slot;
    float lrad, lrad_v;                /* leader length factor, its spring */
    int compact;                       /* in a crowd: a one-line block */
    float bw, bh;                      /* block size, px */
    /* what it's doing */
    int kind, kind_n;                  /* KD_*, frames the classifier has disagreed */
    int apt, rwy, rend;                /* arrivals: the runway end it's lined up on (-1 none) */
    float dist_thr;                    /* nm to go to that threshold */
    float rfl;                         /* the level on the printed strip */
    float cfl[4];                      /* cleared levels, latest last (strip annotations) */
    int ncfl;
    double cfl_t;
    float cfl_pend;
    /* the strip board */
    float sy, syv, sx, sxv, salpha;
    int strip, bay;
} Ac;
static Ac ac[MAXAC];

static Ac *ac_find(uint32_t id, int create) {
    Ac *freeslot = NULL;
    for (int i = 0; i < MAXAC; i++) {
        if (ac[i].used && ac[i].id == id) return &ac[i];
        if (!ac[i].used && !freeslot) freeslot = &ac[i];
    }
    if (!create || !freeslot) return NULL;
    memset(freeslot, 0, sizeof *freeslot);
    freeslot->used = 1;
    freeslot->id = id;
    freeslot->fsel = -1;
    freeslot->apt = freeslot->rwy = freeslot->rend = -1;
    freeslot->slot = -1;
    return freeslot;
}

/* Goodyear's airships, by registration; their ICAO addresses follow from
 * the N-number (the FAA's block from a00001 in N-number order) */
static const struct { const char *reg, *name; } GOODYEAR[] = {
    {"N1A", "WINGFOOT ONE"}, {"N2A", "WINGFOOT TWO"}, {"N3A", "WINGFOOT THREE"},
};
#define NGY 3
static uint32_t gy_hex[NGY];

/* US N-number -> ICAO 24-bit address (the FAA's block from a00001, in
 * N-number order: N1, N1A, N1AA.. N1AZ, N1B.., N10, N10A.., N100...); 0 if
 * it isn't one */
static uint32_t n_to_icao(const char *n) {
    static const char *L = "ABCDEFGHJKLMNPQRSTUVWXYZ";
    const uint32_t sfx = 1 + 24 * 25, b4 = 1 + 24 + 10, b3 = 10 * b4 + sfx, b2 = 10 * b3 + sfx, b1 = 10 * b2 + sfx;
    const uint32_t bucket[4] = {b2, b3, b4, 0};
    if (*n == 'N' || *n == 'n') n++;
    if (*n < '1' || *n > '9') return 0;
    uint32_t out = 0xa00001 + (uint32_t)(*n++ - '1') * b1;
    for (int pos = 0; *n; pos++) {
        int c = toupper((unsigned char)*n);
        const char *li = isalpha(c) ? strchr(L, c) : NULL;
        if (pos == 3) {                /* the fifth character: one more digit or letter */
            if (n[1]) return 0;
            if (li) return out + 1 + (uint32_t)(li - L);
            return isdigit(c) ? out + 25 + (uint32_t)(c - '0') : 0;
        }
        if (li) {                      /* a suffix of one or two letters ends it */
            out += 1 + 25 * (uint32_t)(li - L);
            if (n[1]) {
                const char *l2 = isalpha((unsigned char)n[1]) ? strchr(L, toupper((unsigned char)n[1])) : NULL;
                if (!l2 || n[2]) return 0;
                out += 1 + (uint32_t)(l2 - L);
            }
            return out;
        }
        if (!isdigit(c)) return 0;
        out += sfx + (uint32_t)(c - '0') * bucket[pos];
        n++;
    }
    return out;
}

/* ---- a small tolerant JSON scanner: just enough for aircraft.json ---- */
typedef struct { const char *p, *e; } Js;
static void js_ws(Js *j) { while (j->p < j->e && (unsigned char)*j->p <= ' ') j->p++; }
static int js_str(Js *j, char *out, int n) {
    int k = 0;
    if (j->p >= j->e || *j->p != '"') return 0;
    j->p++;
    while (j->p < j->e && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\' && j->p < j->e) {
            c = *j->p++;
            if (c == 'u') { j->p += 4; if (j->p > j->e) j->p = j->e; c = '?'; }
            else if (c == 'n' || c == 't' || c == 'r') c = ' ';
        }
        if (out && k < n - 1) out[k++] = c;
    }
    if (j->p < j->e) j->p++;
    if (out) out[k] = 0;
    return 1;
}
static void js_skip(Js *j) {
    js_ws(j);
    if (j->p >= j->e) return;
    if (*j->p == '"') { js_str(j, NULL, 0); return; }
    if (*j->p == '{' || *j->p == '[') {
        int d = 0;
        while (j->p < j->e) {
            char c = *j->p;
            if (c == '"') { js_str(j, NULL, 0); continue; }
            j->p++;
            if (c == '{' || c == '[') d++;
            else if ((c == '}' || c == ']') && --d == 0) return;
        }
        return;
    }
    while (j->p < j->e && *j->p != ',' && *j->p != '}' && *j->p != ']') j->p++;
}
/* a value: number -> *num and 1; string -> str and 2; else 0 (skipped) */
static int js_val(Js *j, double *num, char *str, int n) {
    js_ws(j);
    if (j->p >= j->e) return 0;
    if (*j->p == '"') { js_str(j, str, n); return 2; }
    if (*j->p == '-' || isdigit((unsigned char)*j->p)) {
        char *end;
        *num = strtod(j->p, &end);
        if (end > j->e) end = (char *)j->e;
        j->p = end;
        return 1;
    }
    js_skip(j);
    return 0;
}

typedef struct {
    char hex[8], flight[10], reg[12], type[8], sqk[6], emerg[16];
    int cat, ground, have_pos, have_alt, have_gs, have_trk, have_vr, have_sel;
    double lat, lon, seen, seen_pos;
    float alt, gs, trk, vr, sel;
} Rec;

static int parse_cat(const char *s) {
    if (strlen(s) != 2 || !isalpha((unsigned char)s[0]) || !isdigit((unsigned char)s[1])) return 0;
    return ((toupper((unsigned char)s[0]) - 'A' + 0xA) << 4) | (s[1] - '0');
}

/* one aircraft object at j->p ('{'); fills r */
static int js_aircraft(Js *j, Rec *r) {
    memset(r, 0, sizeof *r);
    r->seen = r->seen_pos = -1;
    js_ws(j);
    if (j->p >= j->e || *j->p != '{') { js_skip(j); return 0; }
    j->p++;
    float alt_geom = 0, geom_rate = 0, true_hdg = 0;
    int have_geom = 0, have_grate = 0, have_hdg = 0;
    for (;;) {
        js_ws(j);
        if (j->p >= j->e) return 0;
        if (*j->p == '}') { j->p++; break; }
        if (*j->p == ',') { j->p++; continue; }
        char key[32];
        if (!js_str(j, key, sizeof key)) { js_skip(j); continue; }
        js_ws(j);
        if (j->p < j->e && *j->p == ':') j->p++;
        double v = 0;
        char s[32];
        int t = js_val(j, &v, s, sizeof s);
        if (!strcmp(key, "hex") && t == 2) snprintf(r->hex, sizeof r->hex, "%s", s);
        else if (!strcmp(key, "flight") && t == 2) snprintf(r->flight, sizeof r->flight, "%s", s);
        else if (!strcmp(key, "r") && t == 2) snprintf(r->reg, sizeof r->reg, "%s", s);
        else if (!strcmp(key, "t") && t == 2) snprintf(r->type, sizeof r->type, "%s", s);
        else if (!strcmp(key, "squawk") && t == 2) snprintf(r->sqk, sizeof r->sqk, "%s", s);
        else if (!strcmp(key, "emergency") && t == 2) snprintf(r->emerg, sizeof r->emerg, "%s", s);
        else if (!strcmp(key, "category") && t == 2) r->cat = parse_cat(s);
        else if (!strcmp(key, "lat") && t == 1) { r->lat = v; r->have_pos |= 1; }
        else if (!strcmp(key, "lon") && t == 1) { r->lon = v; r->have_pos |= 2; }
        else if (!strcmp(key, "alt_baro")) {
            if (t == 1) { r->alt = (float)v; r->have_alt = 1; }
            else if (t == 2 && !strcmp(s, "ground")) { r->ground = 1; r->alt = 0; r->have_alt = 1; }
        }
        else if (!strcmp(key, "alt_geom") && t == 1) { alt_geom = (float)v; have_geom = 1; }
        else if (!strcmp(key, "gs") && t == 1) { r->gs = (float)v; r->have_gs = 1; }
        else if (!strcmp(key, "track") && t == 1) { r->trk = (float)v; r->have_trk = 1; }
        else if (!strcmp(key, "true_heading") && t == 1) { true_hdg = (float)v; have_hdg = 1; }
        else if (!strcmp(key, "baro_rate") && t == 1) { r->vr = (float)v; r->have_vr = 1; }
        else if (!strcmp(key, "geom_rate") && t == 1) { geom_rate = (float)v; have_grate = 1; }
        else if ((!strcmp(key, "nav_altitude_mcp") || (!strcmp(key, "nav_altitude_fms") && !r->have_sel)) && t == 1) {
            r->sel = (float)v; r->have_sel = 1;
        }
        else if (!strcmp(key, "seen") && t == 1) r->seen = v;
        else if (!strcmp(key, "seen_pos") && t == 1) r->seen_pos = v;
    }
    if (!r->have_alt && have_geom) { r->alt = alt_geom; r->have_alt = 1; }
    if (!r->have_vr && have_grate) { r->vr = geom_rate; r->have_vr = 1; }
    if (!r->have_trk && have_hdg) { r->trk = true_hdg; r->have_trk = 1; }
    return r->hex[0] != 0;
}

typedef void (*RecFn)(const Rec *r, double tnow);

/* the whole document: "now" and the "aircraft" (readsb) or "ac" (adsb.lol)
 * array; returns how many aircraft, -1 if it isn't one of those */
static int js_parse_feed(const char *buf, size_t len, RecFn fn, double *tnow_out) {
    Js j = {buf, buf + len};
    js_ws(&j);
    if (j.p >= j.e || *j.p != '{') return -1;
    j.p++;
    double tnow = 0;
    const char *arr = NULL;
    for (;;) {
        js_ws(&j);
        if (j.p >= j.e) break;
        if (*j.p == '}') break;
        if (*j.p == ',') { j.p++; continue; }
        char key[32];
        if (!js_str(&j, key, sizeof key)) { js_skip(&j); continue; }
        js_ws(&j);
        if (j.p < j.e && *j.p == ':') j.p++;
        js_ws(&j);
        if ((!strcmp(key, "aircraft") || !strcmp(key, "ac")) && j.p < j.e && *j.p == '[') { arr = j.p; js_skip(&j); }
        else if (!strcmp(key, "now")) { double v; char s[8]; if (js_val(&j, &v, s, sizeof s) == 1) tnow = v; }
        else js_skip(&j);
    }
    if (!arr) return -1;
    if (tnow > 1e11) tnow /= 1000;     /* adsb.lol: milliseconds */
    if (tnow_out) *tnow_out = tnow;
    Js a = {arr + 1, buf + len};
    int n = 0;
    for (;;) {
        js_ws(&a);
        if (a.p >= a.e || *a.p == ']') break;
        if (*a.p == ',') { a.p++; continue; }
        Rec r;
        if (js_aircraft(&a, &r)) { if (fn) fn(&r, tnow); n++; }
    }
    return n;
}

/* ---- the feed ----
 * curl runs in the background into a temp file; a good answer is renamed
 * into the shared cache. Every instance reads the cache when it changes;
 * whoever holds the flock does the fetching. */
enum { SRC_SIM, SRC_PROBE, SRC_LOCAL, SRC_PUBLIC, SRC_REPLAY };
static int src = SRC_SIM;
static const char *probe_urls[3];
static int nprobe, probe_i;
static char feed_url[512], ship_url[256];
static char cache_dir[512], cache_path[600], src_path[600], lock_path[600], backoff_path[600];
static char ship_path[600], ship_lock_path[600];
static double live_until;              /* real data is showing until then */
static double data_t;                  /* the feed's "now" of the last load */
static int data_n;                     /* aircraft in it */

typedef struct {
    pid_t pid;
    int lockfd;
    char tmp[640], code[640];
} Job;
static Job job_main = {0, -1, "", ""}, job_ship = {0, -1, "", ""}, job_probe = {0, -1, "", ""};

static void job_release(Job *j) {
    if (j->lockfd >= 0) { flock(j->lockfd, LOCK_UN); close(j->lockfd); j->lockfd = -1; }
}

/* start curl url -> j->tmp, the HTTP status into j->code; 0 if it couldn't */
static int job_start(Job *j, const char *url, const char *dir, const char *tag, double timeout) {
    snprintf(j->tmp, sizeof j->tmp, "%s/.%s.%d.tmp", dir, tag, (int)getpid());
    snprintf(j->code, sizeof j->code, "%s/.%s.%d.code", dir, tag, (int)getpid());
    char to[16];
    snprintf(to, sizeof to, "%.1f", timeout);
    char *argv[] = {"curl", "-s", "--compressed", "-m", to, "--max-filesize", "20000000",
                    "-o", j->tmp, "-w", "%{http_code}", (char *)url, NULL};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, j->code, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    int ok = posix_spawnp(&j->pid, "curl", &fa, NULL, argv, environ) == 0;
    posix_spawn_file_actions_destroy(&fa);
    if (!ok) j->pid = 0;
    return ok;
}

/* -1 still running; else the HTTP status (0 on a failure to connect) */
static int job_poll(Job *j) {
    if (!j->pid) return 0;
    int st;
    pid_t r = waitpid(j->pid, &st, WNOHANG);
    if (r == 0) return -1;
    j->pid = 0;
    char b[16];
    slurp_line(j->code, b, sizeof b);
    unlink(j->code);
    return atoi(b);
}

static double file_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st)) return 0;
    return st.st_mtim.tv_sec + st.st_mtim.tv_nsec * 1e-9;
}

static void write_small(const char *path, const char *s) {
    char tmp[700];
    snprintf(tmp, sizeof tmp, "%s.%d", path, (int)getpid());
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    fputs(s, f);
    fclose(f);
    rename(tmp, path);
}

static void cache_init(void) {
    const char *xc = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    char parent[400];
    if (xc && *xc) snprintf(parent, sizeof parent, "%s", xc);
    else snprintf(parent, sizeof parent, "%s/.cache", home ? home : "/tmp");
    mkdir(parent, 0755);
    snprintf(cache_dir, sizeof cache_dir, "%s/wallpapers", parent);
    mkdir(cache_dir, 0755);
    snprintf(cache_path, sizeof cache_path, "%s/atc.json", cache_dir);
    snprintf(src_path, sizeof src_path, "%s/atc.src", cache_dir);
    snprintf(lock_path, sizeof lock_path, "%s/atc.lock", cache_dir);
    snprintf(backoff_path, sizeof backoff_path, "%s/atc.backoff", cache_dir);
    snprintf(ship_path, sizeof ship_path, "%s/atc-airships.json", cache_dir);
    snprintf(ship_lock_path, sizeof ship_lock_path, "%s/atc-airships.lock", cache_dir);
}

/* ---- ingesting a feed document ---- */
static double ingest_wall;             /* wall-clock time of this load */
static double ingest_skew;             /* feed clock - our clock */
static int ingest_live;

static int is_goodyear(const Rec *r, const char **name) {
    uint32_t id = (uint32_t)strtoul(r->hex, NULL, 16);
    for (int g = 0; g < NGY; g++) {
        char rg[12];
        snprintf(rg, sizeof rg, "%s", r->reg);
        if (id == gy_hex[g] || !strcasecmp(rg, GOODYEAR[g].reg)) { *name = GOODYEAR[g].name; return 1; }
    }
    return 0;
}

static void trim(char *s) {
    size_t n = strlen(s);
    while (n && s[n - 1] == ' ') s[--n] = 0;
    char *p = s;
    while (*p == ' ') p++;
    memmove(s, p, strlen(p) + 1);
}

static void on_rec(const Rec *r, double tnow) {
    if (r->have_pos != 3 || r->hex[0] == '~' && !r->flight[0]) return;    /* no position; a TIS-B blip with no name */
    double age = r->seen_pos >= 0 ? r->seen_pos : r->seen >= 0 ? r->seen : 0;
    if (age > 60) return;
    double x, y;
    to_xy(r->lat, r->lon, &x, &y);
    if (fabs(x) > 400 || fabs(y) > 400) return;
    uint32_t id = (uint32_t)strtoul(r->hex[0] == '~' ? r->hex + 1 : r->hex, NULL, 16) | (r->hex[0] == '~' ? 0x2000000u : 0);
    Ac *a = ac_find(id, 1);
    if (!a) return;
    int fresh = !a->first;
    if (fresh) { a->first = ingest_wall; a->fade = 0; }
    a->gone = 0;
    a->sim = 0;
    snprintf(a->hex, sizeof a->hex, "%s", r->hex);
    if (r->flight[0]) { snprintf(a->cs, sizeof a->cs, "%s", r->flight); trim(a->cs); }
    if (r->reg[0]) snprintf(a->reg, sizeof a->reg, "%s", r->reg);
    if (r->type[0]) snprintf(a->type, sizeof a->type, "%s", r->type);
    if (r->sqk[0]) snprintf(a->sqk, sizeof a->sqk, "%s", r->sqk);
    if (!a->cs[0]) {
        if (a->reg[0]) { snprintf(a->cs, sizeof a->cs, "%s", a->reg); for (char *c = a->cs; *c; c++) if (*c == '-') memmove(c, c + 1, strlen(c)); }
        else { snprintf(a->cs, sizeof a->cs, "%s", r->hex); for (char *c = a->cs; *c; c++) *c = (char)toupper((unsigned char)*c); }
    }
    if (r->cat) a->cat = r->cat;
    const char *nm = NULL;
    a->goodyear = is_goodyear(r, &nm);
    if (a->goodyear) a->ship_name = nm;
    a->ship = a->cat == 0xB2 || a->goodyear;
    a->ground = r->ground;
    a->emerg = 0;
    int sq = atoi(a->sqk);
    if (sq == 7500 || sq == 7600 || sq == 7700) a->emerg = sq;
    else if (!strcmp(r->emerg, "general") || !strcmp(r->emerg, "lifeguard") || !strcmp(r->emerg, "minfuel")) a->emerg = 7700;
    else if (!strcmp(r->emerg, "nordo")) a->emerg = 7600;
    else if (!strcmp(r->emerg, "unlawful")) a->emerg = 7500;
    /* the fix, on our clock */
    double tf = (tnow > 0 ? tnow - ingest_skew : ingest_wall) - age;
    float alt = r->have_alt ? r->alt : a->falt;
    if (!fresh && tf <= a->ft) return;    /* nothing new */
    /* where the display was going to be at tf: the correction eases from there */
    if (!fresh) {
        double dt = tf - a->ft;
        float v = a->fgs / 3600.0f;
        double px = a->fx + sin(a->ftrk * DEG) * v * dt, py = a->fy + cos(a->ftrk * DEG) * v * dt;
        float palt = a->falt + a->fvr / 60.0f * (float)dt;
        if (!a->has_vel) { px = a->fx; py = a->fy; palt = a->falt; }
        double ce = exp(-(ingest_wall - a->et) / 1.6);
        a->ex = (px - x) + a->ex * ce;
        a->ey = (py - y) + a->ey * ce;
        a->ealt = (palt - alt) + a->ealt * (float)ce;
        a->et = ingest_wall;
        /* a jump of miles is a new fix, not an error to ease */
        if (fabs(a->ex) > 3 || fabs(a->ey) > 3) a->ex = a->ey = 0;
        if (fabsf(a->ealt) > 2000) a->ealt = 0;
    }
    a->fx = x; a->fy = y; a->ft = tf; a->falt = alt;
    a->has_vel = r->have_gs && r->have_trk;
    if (r->have_gs) a->fgs = r->gs;
    if (r->have_trk) a->ftrk = r->trk;
    a->fvr = r->have_vr ? r->vr : 0;
    if (a->ground) { a->fvr = 0; if (a->fgs < 40) a->has_vel = a->fgs > 3 && r->have_trk; }
    a->fsel = r->have_sel ? r->sel : -1;
    if (fresh) { a->x = x; a->y = y; a->alt = alt; a->trk = a->ftrk; a->gs = a->fgs; a->vr = a->fvr; a->et = ingest_wall; }
    a->last = ingest_wall;
}

static int load_feed_file(const char *path, double wt, int is_ships) {
    size_t len;
    char *buf = slurp(path, &len);
    if (!buf) return -1;
    ingest_wall = wt;
    double tnow = 0;
    int n = js_parse_feed(buf, len, NULL, &tnow);
    if (n >= 0) {
        /* the feed's clock against ours: trust it within reason (the file
         * may be a few seconds old; positions are timed by "now" - seen) */
        double mt = file_mtime(path);
        ingest_skew = tnow > 0 && fabs(tnow - mt) < 600 ? tnow - mt : 0;
        if (opt_replay) ingest_skew = tnow > 0 ? tnow - wt : 0;
        if (!is_ships) {
            js_parse_feed(buf, len, on_rec, &tnow);
            data_t = tnow; data_n = n;
        }
    }
    free(buf);
    return n;
}

/* the airship watch: the Goodyear airships wherever they are */
typedef struct { int on; double lat, lon; float alt, gs, trk; char reg[8]; const char *name; double t; } Far;
static Far far_ship;
static void on_ship_rec(const Rec *r, double tnow) {
    (void)tnow;
    const char *nm = NULL;
    if (r->have_pos != 3 || !is_goodyear(r, &nm)) return;
    double d, b;
    gc_from_home(r->lat, r->lon, &d, &b);
    if (far_ship.on && far_ship.t == ingest_wall) {
        double d0, b0;
        gc_from_home(far_ship.lat, far_ship.lon, &d0, &b0);
        if (d0 <= d) return;           /* keep the nearest */
    }
    far_ship.on = 1;
    far_ship.lat = r->lat; far_ship.lon = r->lon;
    far_ship.alt = r->alt; far_ship.gs = r->gs; far_ship.trk = r->trk;
    far_ship.name = nm;
    far_ship.t = ingest_wall;
    for (int g = 0; g < NGY; g++) if (GOODYEAR[g].name == nm) snprintf(far_ship.reg, sizeof far_ship.reg, "%s", GOODYEAR[g].reg);
}
static void load_ships(double wt) {
    size_t len;
    char *buf = slurp(ship_path, &len);
    if (!buf) return;
    ingest_wall = wt;
    far_ship.on = 0;
    double tnow;
    js_parse_feed(buf, len, on_ship_rec, &tnow);
    free(buf);
    if (wt - file_mtime(ship_path) > 1800) far_ship.on = 0;
}

/* fetch into the shared cache if it's due and nobody else is at it */
static double backoff_s = 15, next_feed_check;
static double cache_seen, ship_seen;

static double read_backoff(void) {
    char b[32];
    slurp_line(backoff_path, b, sizeof b);
    return atof(b);
}

static void fetch_tick(Job *j, const char *url, const char *path, const char *lockp, const char *tag,
                       double interval, double timeout, double wt, int with_src) {
    if (j->pid) {
        int code = job_poll(j);
        if (code < 0) return;
        double tn;
        size_t len;
        char *buf = code == 200 ? slurp(j->tmp, &len) : NULL;
        int ok = buf && js_parse_feed(buf, len, NULL, &tn) >= 0;
        free(buf);
        if (ok) {
            if (with_src) write_small(src_path, url);
            rename(j->tmp, path);
            backoff_s = 15;
        } else {
            unlink(j->tmp);
            /* back off, for everyone: harder on "too many requests" */
            backoff_s = fmin(900, fmax(code == 429 ? 120 : 30, backoff_s * 2));
            char b[32];
            snprintf(b, sizeof b, "%.0f\n", wt + backoff_s);
            write_small(backoff_path, b);
        }
        job_release(j);
        return;
    }
    double age = wt - file_mtime(path);
    if (age < interval) return;
    if (with_src) {
        char s[512];
        slurp_line(src_path, s, sizeof s);
        if (strcmp(s, url)) age = 1e9;   /* the cache is someone else's (another source) */
    }
    if (wt < read_backoff()) return;
    j->lockfd = open(lockp, O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (j->lockfd < 0 || flock(j->lockfd, LOCK_EX | LOCK_NB)) { job_release(j); return; }
    /* another instance may have just fetched */
    if (wt - file_mtime(path) < interval) {
        char s[512];
        slurp_line(src_path, s, sizeof s);
        if (!with_src || !strcmp(s, url)) { job_release(j); return; }
    }
    if (!job_start(j, url, cache_dir, tag, timeout)) job_release(j);
}

static void feed_init(void) {
    gy_hex[0] = n_to_icao("N1A"); gy_hex[1] = n_to_icao("N2A"); gy_hex[2] = n_to_icao("N3A");
    if (opt_replay) { src = SRC_REPLAY; return; }
    if (opt_sim || opt_shot || opt_bench) { src = SRC_SIM; return; }
    cache_init();
    if (opt_feed) probe_urls[nprobe++] = opt_feed;
    else {
        probe_urls[nprobe++] = "http://127.0.0.1:8080/data/aircraft.json";
        probe_urls[nprobe++] = "http://127.0.0.1/tar1090/data/aircraft.json";
    }
    src = SRC_PROBE;
    probe_i = 0;
    if (have_loc) {
        /* nothing but a rounded position and a radius */
        snprintf(ship_url, sizeof ship_url, "https://api.adsb.lol/v2/hex/%06x,%06x,%06x", gy_hex[0], gy_hex[1], gy_hex[2]);
    }
}

static void feed_tick(double wt) {
    if (src == SRC_SIM || src == SRC_REPLAY) return;
    if (src == SRC_PROBE) {
        if (job_probe.pid) {
            int code = job_poll(&job_probe);
            if (code < 0) return;
            size_t len;
            double tn;
            char *buf = code == 200 ? slurp(job_probe.tmp, &len) : NULL;
            int ok = buf && js_parse_feed(buf, len, NULL, &tn) >= 0;
            free(buf);
            unlink(job_probe.tmp);
            if (ok) {
                snprintf(feed_url, sizeof feed_url, "%s", probe_urls[probe_i]);
                src = SRC_LOCAL;
                return;
            }
            probe_i++;
        }
        if (probe_i < nprobe) { job_start(&job_probe, probe_urls[probe_i], cache_dir, "atc-probe", 1.5); return; }
        if (have_loc) {
            snprintf(feed_url, sizeof feed_url, "https://api.adsb.lol/v2/point/%.2f/%.2f/%d",
                     opt_lat, opt_lon, (int)clampf(opt_radius, 5, 250));
            src = SRC_PUBLIC;
        } else src = SRC_SIM;
        return;
    }
    if (wt < next_feed_check && !job_main.pid && !job_ship.pid) return;
    next_feed_check = wt + 0.5;
    if (src == SRC_LOCAL) fetch_tick(&job_main, feed_url, cache_path, lock_path, "atc", 2, 1.5, wt, 1);
    else fetch_tick(&job_main, feed_url, cache_path, lock_path, "atc", 15, 12, wt, 1);
    if (ship_url[0]) fetch_tick(&job_ship, ship_url, ship_path, ship_lock_path, "atc-ships", 300, 12, wt, 0);
    /* pick up whatever's new in the caches */
    double mt = file_mtime(cache_path);
    if (mt && mt != cache_seen) {
        char s[512];
        slurp_line(src_path, s, sizeof s);
        if (!strcmp(s, feed_url)) {
            cache_seen = mt;
            if (wt - mt < 120 && load_feed_file(cache_path, wt, 0) >= 0) live_until = mt + (src == SRC_LOCAL ? 30 : 90);
        }
    }
    double ms = file_mtime(ship_path);
    if (ms && ms != ship_seen) { ship_seen = ms; load_ships(wt); }
}

/* ---- simulated traffic ----
 * Streams of arrivals and departures for the airports nearby, overflights,
 * light aircraft and the odd airship. Every aircraft is a pure function of
 * the seed and the wall clock: its spawn time comes from a hash of its slot,
 * and it's flown in fixed half-second steps from then, so every instance
 * (and every --shot) sees the same sky. */
#define MAXSIM 200
#define SIM_H 0.5
#define HIST_DT 5.0
#define MAXWP 12
typedef struct {
    int alive, kind, ai, ri, end;
    uint32_t key;
    double t0, tk, tend;
    float x, y, alt, gs, trk, vr;
    float px, py, palt, pgs, ptrk, pvr;
    float sx0, sy0;
    float wx[MAXWP], wy[MAXWP], walt[MAXWP], wspd[MAXWP];
    int nwp, wi, loop;
    float cruise, cspd, elev, turn;
    char cs[10], type[8], sqk[6], reg[12];
    int cat, emerg;
    float cfl;
    float hx[MAXHIST], hy[MAXHIST];
    int hn, hmax;
} Sim;
static Sim sim[MAXSIM];
static uint32_t dead[2048];
static int deadn;

typedef struct { int kind, ai; float period, life, pskip; uint32_t salt; } Stream;
static Stream stream[24];
static int nstream;
static float wind_from;
static int sim_arr_rwy[MAXAPT], sim_dep_rwy[MAXAPT], sim_end[MAXAPT];
static float ship_x, ship_y;

static const char *AIRLINE[] = {"QXL", "ZEV", "KQZ", "VZX", "JQX", "WZQ", "TQX", "OZQ", "EXQ", "QZA", "XLQ", "FZQ"};
#define NAIRLINE 12
static const char *JET[] = {"A320", "A20N", "A21N", "B738", "B38M", "A319", "E190", "A321", "B739", "A333", "B789", "B77W", "A359", "A388", "B744", "DH8D", "AT76", "CRJ9", "E195", "A332"};
static const char *GA_T[] = {"C172", "PA28", "SR22", "DA40", "C152", "P28A", "DA42", "BE36"};

static uint32_t slot_hash(uint32_t salt, long n) { return hash32(opt_seed * 0x9E3779B9u ^ salt ^ hash32((uint32_t)n * 0x85EBCA6Bu + 0x1234567u)); }

static int is_dead(uint32_t k) { for (int i = 0; i < deadn && i < 2048; i++) if (dead[i] == k) return 1; return 0; }

static void sim_wp(Sim *s, float x, float y, float alt, float spd) {
    if (s->nwp >= MAXWP) return;
    s->wx[s->nwp] = x; s->wy[s->nwp] = y; s->walt[s->nwp] = alt; s->wspd[s->nwp] = spd;
    s->nwp++;
}

/* a plausible landing direction for the day: mostly westerlies */
static void sim_runways(double wt) {
    uint32_t h = hash32(opt_seed * 31u + (uint32_t)(wt / 86400) * 7919u);
    Rng r = {h | 1};
    wind_from = rnd(&r) < 0.7f ? rndr(&r, 225, 290) : rndr(&r, 0, 360);
    for (int i = 0; i < napt; i++) {
        Apt *A = &apt[i];
        /* the main runway (longest), and a parallel to it for departures */
        int best = 0;
        for (int k = 1; k < A->nr; k++) if (A->r[k].len_nm > A->r[best].len_nm + 0.05f) best = k;
        int par = -1;
        for (int k = 0; k < A->nr; k++)
            if (k != best && fabsf(wrap180(A->r[k].hdg - A->r[best].hdg)) < 12 && A->r[k].len_nm > 1.2f) par = k;
        int e = fabsf(wrap180(A->r[best].hdg - wind_from)) < 90 ? 0 : 1;
        /* arrivals on one parallel, departures off the other, chosen by the airport */
        int swap = par >= 0 && (hash32((uint32_t)i * 977u + opt_seed) & 1);
        sim_arr_rwy[i] = swap ? par : best;
        sim_dep_rwy[i] = par >= 0 ? (swap ? best : par) : best;
        sim_end[i] = e;
        for (int k = 0; k < A->nr; k++) { A->r[k].act_a = A->r[k].act_b = 0; }
        if (e == 0) A->r[sim_arr_rwy[i]].act_a = 1; else A->r[sim_arr_rwy[i]].act_b = 1;
    }
}

static void sim_init(double wt) {
    nstream = 0;
    sim_runways(wt);
    int nsa = 0;
    for (int i = 0; i < napt && nsa < 4; i++) {
        const Apt *A = &apt[i];
        if (A->dist > 50 || (A->d->rank == 3 && A->dist > 25)) continue;
        float per = A->d->rank == 1 ? 135 : A->d->rank == 2 ? 320 : 720;
        stream[nstream++] = (Stream){KD_ARR, i, per, 1500, 0.08f, 0xA0000u + (uint32_t)i};
        stream[nstream++] = (Stream){KD_DEP, i, per * 1.05f, 900, 0.08f, 0xD0000u + (uint32_t)i};
        nsa++;
    }
    stream[nstream++] = (Stream){KD_OVF, -1, 150, 1500, 0.2f, 0x0F0F0u};
    stream[nstream++] = (Stream){KD_GA, -1, 520, 1700, 0.4f, 0x6A6A6u};
    stream[nstream++] = (Stream){KD_SHIP, -1, 3600, 2700, 0, 0x5A1Bu};
    Rng r = {hash32(opt_seed * 13u + 0x51ab) | 1};
    float b = rndr(&r, 0, 360), d = rndr(&r, 6, 12);
    ship_x = sinf(b * DEG) * d; ship_y = cosf(b * DEG) * d;
}

static void sim_name(Sim *s, Rng *r, int ga) {
    int us = lon0 < -30;
    if (ga) {
        static const char *A = "ABCDEFGHJKLMNPRSTUVWXYZ";
        if (us) snprintf(s->cs, sizeof s->cs, "N%d%c%c", 100 + rndi(r, 899), A[rndi(r, 23)], A[rndi(r, 23)]);
        else snprintf(s->cs, sizeof s->cs, "G%c%c%c%c", A[rndi(r, 23)], A[rndi(r, 23)], A[rndi(r, 23)], A[rndi(r, 23)]);
        snprintf(s->reg, sizeof s->reg, "%s", s->cs);
        snprintf(s->type, sizeof s->type, "%s", GA_T[rndi(r, 8)]);
        snprintf(s->sqk, sizeof s->sqk, us ? "1200" : "7000");
        s->cat = 0xA1;
        return;
    }
    int n = rnd(r) < 0.5f ? 1 + rndi(r, 99) : 100 + rndi(r, 2900);
    if (rnd(r) < 0.25f) {
        static const char *A = "ABCDEFGHJKLMNPRSTUVWXYZ";
        snprintf(s->cs, sizeof s->cs, "%s%d%c", AIRLINE[rndi(r, NAIRLINE)], n % 1000, A[rndi(r, 23)]);
    } else snprintf(s->cs, sizeof s->cs, "%s%d", AIRLINE[rndi(r, NAIRLINE)], n);
    int ti = rnd(r) < 0.72f ? rndi(r, 9) : rndi(r, 20);
    snprintf(s->type, sizeof s->type, "%s", JET[ti]);
    s->cat = ti >= 9 && ti <= 15 ? 0xA5 : ti >= 15 ? 0xA2 : 0xA3;
    if (!strcmp(s->type, "DH8D") || !strcmp(s->type, "AT76")) s->cat = 0xA2;
    for (;;) {
        snprintf(s->sqk, sizeof s->sqk, "%d%d%d%d", rndi(r, 8), rndi(r, 8), rndi(r, 8), rndi(r, 8));
        int q = atoi(s->sqk);
        if (q != 7500 && q != 7600 && q != 7700 && q != 7000 && q != 1200 && q != 2000 && q > 1000) break;
    }
    snprintf(s->reg, sizeof s->reg, "%c-%c%c%c%c", us ? 'N' : 'G', 'A' + rndi(r, 26), 'A' + rndi(r, 26), 'A' + rndi(r, 26), 'A' + rndi(r, 26));
}

static int sim_spawn(Sim *s, const Stream *st, uint32_t key, double t0) {
    memset(s, 0, sizeof *s);
    Rng r = {key | 1};
    rnd(&r); rnd(&r);
    s->key = key; s->kind = st->kind; s->ai = st->ai; s->t0 = s->tk = t0; s->tend = t0 + st->life;
    s->turn = 2.6f; s->hmax = 9; s->cfl = -1;
    if (st->kind == KD_ARR || st->kind == KD_DEP) {
        const Apt *A = &apt[st->ai];
        s->elev = A->d->elev;
        int ri = st->kind == KD_ARR ? sim_arr_rwy[st->ai] : sim_dep_rwy[st->ai], e = sim_end[st->ai];
        s->ri = ri; s->end = e;
        float tx, ty, ux, uy, hdg;
        rwy_end(&A->r[ri], e, &tx, &ty, &ux, &uy, &hdg);
        sim_name(s, &r, 0);
        /* four corridors in and out, fixed per airport */
        uint32_t ah = hash32((uint32_t)st->ai * 7717u + 99u);
        float c0 = (float)(ah % 360);
        if (st->kind == KD_ARR) {
            float b = c0 + 90 * rndi(&r, 4) + rndr(&r, -12, 12);
            float d0 = rndr(&r, 52, 64);
            float ex = A->x + sinf(b * DEG) * d0, ey = A->y + cosf(b * DEG) * d0;
            s->x = s->sx0 = ex; s->y = s->sy0 = ey;
            s->alt = (float)(100 * (int)(rndr(&r, 110, 190))) ; s->alt = roundf(s->alt / 1000) * 1000;
            s->gs = rndr(&r, 280, 320);
            /* a feeder fix ~20 nm out on the arrival's side, then a base leg
             * onto a 10-12 mile final */
            float side = (ex - tx) * uy - (ey - ty) * ux;          /* + right of the approach */
            side = side >= 0 ? 1 : -1;
            float px = -uy * side, py = ux * side;                  /* toward that side */
            float fin = rndr(&r, 9.5f, 12.5f);
            float bx = tx - ux * (fin + 3) + px * 6, by = ty - uy * (fin + 3) + py * 6;
            float ix = tx - ux * fin, iy = ty - uy * fin;
            float fx = A->x + sinf(b * DEG) * 24, fy = A->y + cosf(b * DEG) * 24;
            /* don't fly the feeder fix if it's behind the base point */
            float fdx = fx - bx, fdy = fy - by;
            if (fdx * fdx + fdy * fdy > 36) sim_wp(s, fx, fy, 10000 + s->elev, 250);
            sim_wp(s, bx, by, 5000 + s->elev, 210);
            sim_wp(s, ix, iy, 3000 + s->elev, 175);
            sim_wp(s, tx, ty, s->elev + 50, 140);
            sim_wp(s, tx + ux * A->r[ri].len_nm * 0.55f, ty + uy * A->r[ri].len_nm * 0.55f, s->elev, 25);
            s->cruise = s->alt;
            s->cfl = roundf(fminf(s->alt, 10000 + s->elev) / 1000) * 1000;
            /* a rare emergency, or a radio failure */
            if (key % 53 == 3) { snprintf(s->sqk, sizeof s->sqk, "7700"); s->emerg = 7700; }
            else if (key % 97 == 11) { snprintf(s->sqk, sizeof s->sqk, "7600"); s->emerg = 7600; }
        } else {
            /* lined up at the threshold, rolling */
            s->x = s->sx0 = tx; s->y = s->sy0 = ty;
            s->alt = s->elev; s->gs = 135; s->trk = hdg;
            float b = c0 + 45 + 90 * rndi(&r, 4) + rndr(&r, -15, 15);
            float L = A->r[ri].len_nm;
            float w1x = tx + ux * (L + 4.5f), w1y = ty + uy * (L + 4.5f);
            sim_wp(s, w1x, w1y, 4000, 220);
            sim_wp(s, A->x + sinf(b * DEG) * 22, A->y + cosf(b * DEG) * 22, 12000, 280);
            sim_wp(s, A->x + sinf(b * DEG) * 95, A->y + cosf(b * DEG) * 95, 30000, 420);
            s->cruise = (float)(1000 * (int)rndr(&r, 23, 38));
            s->cspd = rndr(&r, 400, 470);
            s->cfl = 6000;
        }
    } else if (st->kind == KD_OVF) {
        sim_name(s, &r, 0);
        if (s->cat == 0xA2) { snprintf(s->type, sizeof s->type, "A320"); s->cat = 0xA3; }
        float b = rndr(&r, 0, 360), off = rndr(&r, -55, 55);
        float dx = sinf(b * DEG), dy = cosf(b * DEG), qx = -dy, qy = dx;
        s->x = s->sx0 = -dx * 90 + qx * off; s->y = s->sy0 = -dy * 90 + qy * off;
        float bend = rndr(&r, -12, 12);
        sim_wp(s, qx * (off + bend), qy * (off + bend), 0, 0);
        sim_wp(s, dx * 90 + qx * (off + bend * 2), dy * 90 + qy * (off + bend * 2), 0, 0);
        float trk = wrap360(b);
        int fl = 290 + 20 * rndi(&r, 6) + (trk < 180 ? 0 : 10);  /* odd east, even west, roughly */
        s->alt = s->cruise = (float)fl * 100;
        s->gs = s->cspd = rndr(&r, 430, 500);
        s->trk = trk;
        for (int k = 0; k < s->nwp; k++) { s->walt[k] = s->cruise; s->wspd[k] = s->cspd; }
        s->cfl = s->cruise;
    } else if (st->kind == KD_GA) {
        sim_name(s, &r, 1);
        float b = rndr(&r, 0, 360), d = rndr(&r, 8, 30);
        s->x = s->sx0 = sinf(b * DEG) * d; s->y = s->sy0 = cosf(b * DEG) * d;
        float alt = 500.0f * (int)rndr(&r, 4, 9);
        s->alt = s->cruise = alt;
        s->gs = s->cspd = rndr(&r, 90, 125);
        for (int k = 0; k < 5; k++) {
            float bb = rndr(&r, 0, 360), dd = rndr(&r, 6, 32);
            sim_wp(s, sinf(bb * DEG) * dd, cosf(bb * DEG) * dd, alt, s->cspd);
        }
        sim_wp(s, sinf(b * DEG) * 60, cosf(b * DEG) * 60, alt, s->cspd);
        s->turn = 4; s->trk = rndr(&r, 0, 360);
    } else {
        /* an airship, loitering over the stadium */
        snprintf(s->cs, sizeof s->cs, "BLIMP1");
        snprintf(s->reg, sizeof s->reg, "SIM-LTA");
        snprintf(s->type, sizeof s->type, "LTA");
        snprintf(s->sqk, sizeof s->sqk, "%s", lon0 < -30 ? "1200" : "7000");
        s->cat = 0xB2;
        float b = rndr(&r, 0, 360);
        s->x = s->sx0 = ship_x + sinf(b * DEG) * 5; s->y = s->sy0 = ship_y + cosf(b * DEG) * 5;
        s->alt = s->cruise = rndr(&r, 1100, 1500);
        s->gs = s->cspd = rndr(&r, 28, 36);
        for (int k = 0; k < 10; k++) {
            float a = (b + 180 + k * 36) * DEG;
            sim_wp(s, ship_x + sinf(a) * 1.3f + cosf(a) * 0.4f, ship_y + cosf(a) * 0.9f, s->alt, s->cspd);
        }
        s->loop = 1; s->turn = 2.5f; s->hmax = 30;
        s->trk = wrap360(b + 180);
    }
    if (s->nwp && st->kind != KD_OVF && st->kind != KD_DEP) {
        float dx = s->wx[0] - s->x, dy = s->wy[0] - s->y;
        s->trk = wrap360(atan2f(dx, dy) / DEG);
    }
    s->px = s->x; s->py = s->y; s->palt = s->alt; s->pgs = s->gs; s->ptrk = s->trk;
    s->alive = 1;
    return 1;
}

/* distance along the route from here to waypoint k */
static float route_dist(const Sim *s, int k) {
    float d = 0, x = s->x, y = s->y;
    for (int i = s->wi; i <= k && i < s->nwp; i++) {
        float dx = s->wx[i] - x, dy = s->wy[i] - y;
        d += sqrtf(dx * dx + dy * dy);
        x = s->wx[i]; y = s->wy[i];
    }
    return d;
}

static void sim_step(Sim *s, float h) {
    s->px = s->x; s->py = s->y; s->palt = s->alt; s->pgs = s->gs; s->ptrk = s->trk; s->pvr = s->vr;
    float v = s->gs / 3600.0f;
    float want = s->trk;
    if (s->wi >= s->nwp && s->loop) s->wi = 0;
    if (s->wi < s->nwp) {
        float ax = s->wi ? s->wx[s->wi - 1] : s->sx0, ay = s->wi ? s->wy[s->wi - 1] : s->sy0;
        if (s->loop && s->wi == 0 && s->tk - s->t0 > 60) { ax = s->wx[s->nwp - 1]; ay = s->wy[s->nwp - 1]; }
        float bx = s->wx[s->wi], by = s->wy[s->wi];
        float dx = bx - ax, dy = by - ay, len = sqrtf(dx * dx + dy * dy);
        if (len < 1e-3f) { s->wi++; return; }
        dx /= len; dy /= len;
        float along = (s->x - ax) * dx + (s->y - ay) * dy;
        float la = fmaxf(0.5f, v * 22);
        float ta = fminf(along + la, len);
        float tx = ax + dx * ta, ty = ay + dy * ta;
        /* turn early onto the next leg (not onto the runway) */
        float lead = v * 3600 * (1 / (s->turn * 60)) * 0.9f;  /* ~ turn radius */
        int last = s->wi == s->nwp - 1;
        if (along > len - (last || s->kind == KD_ARR && s->wi >= s->nwp - 2 ? 0.05f : lead)) { s->wi++; }
        float ox = tx - s->x, oy = ty - s->y;
        if (ox * ox + oy * oy > 1e-6f) want = wrap360(atan2f(ox, oy) / DEG);
    } else if (s->kind != KD_SHIP) {
        s->alive = 0;
        return;
    }
    float dt = wrap180(want - s->trk), rate = s->turn * h;
    s->trk = wrap360(s->trk + clampf(dt, -rate, rate));
    /* speed */
    int wi = s->wi < s->nwp ? s->wi : s->nwp - 1;
    float vt = s->kind == KD_DEP ? (s->alt < 10000 ? fminf(250, 160 + (s->alt - s->elev) * 0.05f) : s->cspd)
             : s->kind == KD_OVF || s->kind == KD_GA || s->kind == KD_SHIP ? s->cspd
             : s->wspd[wi];
    if (s->kind == KD_ARR && s->alt < 10500 && vt > 250) vt = 250;
    s->gs += clampf(vt - s->gs, -1.3f * h, (s->kind == KD_DEP ? 3.0f : 1.5f) * h);
    /* altitude */
    float want_alt = s->cruise, vmax = 3000, vmin = -2600;
    if (s->kind == KD_ARR) {
        want_alt = s->cruise;
        for (int k = s->wi; k < s->nwp; k++) want_alt = fminf(want_alt, s->walt[k] + 310 * route_dist(s, k));
        if (s->wi >= s->nwp - 2) want_alt = fminf(want_alt, s->elev + 50 + 318 * route_dist(s, s->nwp - 2));
        vmax = 0;
    } else if (s->kind == KD_DEP) {
        want_alt = s->cruise;
        vmax = s->alt < 10000 ? 3000 : s->alt < 25000 ? 2000 : 1100;
        if (s->gs < 145) vmax = 0;
    } else if (s->kind == KD_GA || s->kind == KD_SHIP) {
        want_alt = s->walt[wi]; vmax = 600; vmin = -600;
    }
    float vt_r = clampf((want_alt - s->alt) * 60 / 22, vmin, vmax);
    s->vr += clampf(vt_r - s->vr, -350 * h, 350 * h);
    s->alt += s->vr / 60 * h;
    if (s->alt < s->elev) { s->alt = s->elev; if (s->vr < 0) s->vr = 0; }
    /* the cleared level, as a controller would step it */
    if (s->kind == KD_ARR) {
        float c = roundf(fmaxf(want_alt + 1500, 3000 + s->elev) / 1000) * 1000;
        if (c < s->cfl - 900) s->cfl = c;
    } else if (s->kind == KD_DEP) {
        float c = s->alt > 4000 ? (s->alt > 9000 ? (s->alt > 16000 ? s->cruise : 18000) : 10000) : 6000;
        if (c > s->cfl) s->cfl = fminf(c, s->cruise);
    }
    s->x += sinf(s->trk * DEG) * v * h;
    s->y += cosf(s->trk * DEG) * v * h;
    s->tk += h;
    if (floor(s->tk / HIST_DT) != floor((s->tk - h) / HIST_DT)) {
        if (s->hn == MAXHIST) { memmove(s->hx, s->hx + 1, (MAXHIST - 1) * sizeof(float)); memmove(s->hy, s->hy + 1, (MAXHIST - 1) * sizeof(float)); s->hn--; }
        s->hx[s->hn] = s->x; s->hy[s->hn] = s->y; s->hn++;
    }
    if (s->kind == KD_DEP || s->kind == KD_OVF || s->kind == KD_GA) {
        if (s->x * s->x + s->y * s->y > 100 * 100) s->alive = 0;
    }
    if (s->tk >= s->tend) s->alive = 0;
}

static void sim_kill(Sim *s) {
    s->alive = 0;
    dead[deadn++ & 2047] = s->key;
    if (deadn >= 4096) deadn = 2048;
}

static void sim_tick(double wt) {
    /* spawn whatever's due (and not flown and gone already) */
    for (int si = 0; si < nstream; si++) {
        const Stream *st = &stream[si];
        long n0 = (long)floor((wt - st->life) / st->period) - 1, n1 = (long)floor(wt / st->period);
        for (long n = n0; n <= n1; n++) {
            uint32_t key = slot_hash(st->salt, n);
            Rng r = {key | 1};
            double ts = n * (double)st->period + rnd(&r) * st->period * 0.7;
            if (rnd(&r) < st->pskip || ts > wt || ts < wt - st->life) continue;
            int found = 0;
            for (int i = 0; i < MAXSIM && !found; i++) if (sim[i].alive && sim[i].key == key) found = 1;
            if (found || is_dead(key)) continue;
            int fi = -1;
            for (int i = 0; i < MAXSIM; i++) if (!sim[i].alive) { fi = i; break; }
            if (fi < 0) break;
            if (!sim_spawn(&sim[fi], st, key, ts)) continue;
        }
    }
    /* fly each to now */
    for (int i = 0; i < MAXSIM; i++) {
        Sim *s = &sim[i];
        if (!s->alive) continue;
        while (s->alive && s->tk < wt) sim_step(s, SIM_H);
        if (!s->alive) sim_kill(s);
    }
}

/* the sim's aircraft into the traffic table */
static void sim_publish(double wt) {
    for (int i = 0; i < MAXAC; i++) if (ac[i].used && ac[i].sim) ac[i].gone = 1;
    for (int i = 0; i < MAXSIM; i++) {
        Sim *s = &sim[i];
        if (!s->alive) continue;
        uint32_t id = 0x1000000u | (s->key & 0xffffff);
        Ac *a = ac_find(id, 1);
        if (!a) continue;
        if (!a->first) {
            a->first = s->t0; a->fade = 0;
            snprintf(a->cs, sizeof a->cs, "%s", s->cs);
            snprintf(a->reg, sizeof a->reg, "%s", s->reg);
            snprintf(a->type, sizeof a->type, "%s", s->type);
            snprintf(a->hex, sizeof a->hex, "%06x", id & 0xffffff);
            a->cat = s->cat;
            a->ship = s->cat == 0xB2;
            if (s->kind != KD_SHIP) a->kind = s->kind;
            if (s->kind == KD_GA) a->kind = KD_OVF;
            if (s->kind == KD_SHIP) a->kind = KD_SHIP;
            if (s->kind == KD_ARR) { a->apt = s->ai; a->rwy = s->ri; a->rend = s->end; }
            if (s->kind == KD_DEP) { a->apt = s->ai; }
        }
        snprintf(a->sqk, sizeof a->sqk, "%s", s->sqk);
        a->emerg = s->emerg;
        a->sim = 1; a->gone = 0;
        float f = (float)clampf((float)(1 - (s->tk - wt) / SIM_H), 0, 1);
        a->x = a->fx = mixf(s->px, s->x, f);
        a->y = a->fy = mixf(s->py, s->y, f);
        a->alt = a->falt = mixf(s->palt, s->alt, f);
        a->gs = a->fgs = mixf(s->pgs, s->gs, f);
        a->trk = a->ftrk = wrap360(s->ptrk + wrap180(s->trk - s->ptrk) * f);
        a->vr = a->fvr = mixf(s->pvr, s->vr, f);
        a->ex = a->ey = 0; a->ealt = 0;
        a->ft = wt; a->last = wt; a->has_vel = 1;
        a->ground = s->kind == KD_ARR && s->alt < s->elev + 20 && s->wi >= s->nwp - 1;
        if (s->kind == KD_DEP && s->alt < s->elev + 30 && s->tk - s->t0 < 30) a->ground = 1;
        a->fsel = s->cfl;
        a->rfl = s->cruise;
        int n = s->hn < s->hmax ? s->hn : s->hmax;
        memcpy(a->hx, s->hx + s->hn - n, n * sizeof(float));
        memcpy(a->hy, s->hy + s->hn - n, n * sizeof(float));
        a->hn = n;
    }
}

/* ---- keeping the traffic up to date ---- */
static int shot_mode;

static void push_hist(Ac *a, int hmax) {
    if (a->hn >= hmax) { memmove(a->hx, a->hx + 1, (hmax - 1) * sizeof(float)); memmove(a->hy, a->hy + 1, (hmax - 1) * sizeof(float)); a->hn = hmax - 1; }
    a->hx[a->hn] = (float)a->x; a->hy[a->hn] = (float)a->y; a->hn++;
}

/* the runway end an aircraft is lined up on, if any */
static void align_rwy(Ac *a, int ai) {
    a->rwy = a->rend = -1;
    a->dist_thr = -1;
    if (ai < 0) return;
    const Apt *A = &apt[ai];
    float best = 1e9f;
    for (int r = 0; r < A->nr; r++)
        for (int e = 0; e < 2; e++) {
            float tx, ty, ux, uy, hdg;
            rwy_end(&A->r[r], e, &tx, &ty, &ux, &uy, &hdg);
            float dx = tx - (float)a->x, dy = ty - (float)a->y;
            float along = dx * ux + dy * uy, lat = fabsf(dx * uy - dy * ux);
            if (along < -0.3f || along > 30) continue;
            if (fabsf(wrap180(a->trk - hdg)) > 28 || lat > 1.2f + along * 0.07f) continue;
            if (a->alt - A->d->elev > 1500 + along * 420) continue;     /* far too high to be landing */
            if (lat < best) { best = lat; a->rwy = r; a->rend = e; a->dist_thr = along; }
        }
}

/* arrival, departure or overflight, from what it's doing near which airport */
static void classify(Ac *a, float dt) {
    if (a->ship) { a->kind = KD_SHIP; return; }
    if (a->sim) {
        if (a->kind == KD_ARR) align_rwy(a, a->apt);
        return;
    }
    int want = KD_OVF, wapt = -1;
    int light = a->cat == 0xA1 || a->cat == 0xA7 || a->cat == 0xB1 || a->cat == 0xB4;
    float best = 1e9f;
    for (int i = 0; i < napt; i++) {
        const Apt *A = &apt[i];
        if (A->dist > 120) break;
        if (A->d->rank == 3 && light) continue;
        float dx = A->x - (float)a->x, dy = A->y - (float)a->y, d = sqrtf(dx * dx + dy * dy);
        if (d > 45) continue;
        float brg = wrap360(atan2f(dx, dy) / DEG);
        float elev = A->d->elev;
        int arr = a->alt < 14000 && (a->vr < -250 || (a->alt - elev < 4000 && d < 12)) && fabsf(wrap180(a->trk - brg)) < 75;
        int dep = a->alt < 18000 && a->vr > 250 && fabsf(wrap180(a->trk - brg - 180)) < 80 && (a->alt - elev) < 1000 + d * 600;
        float score = d * (A->d->rank == 1 ? 0.7f : A->d->rank == 2 ? 1.0f : 1.6f);
        if ((arr || dep) && score < best) { best = score; want = arr ? KD_ARR : KD_DEP; wapt = i; }
    }
    if (light && want == KD_OVF) want = KD_OVF;
    if (a->kind == KD_NONE || shot_mode) { a->kind = want; a->apt = wapt; }
    else if (want != a->kind || (want != KD_OVF && wapt != a->apt)) {
        /* it has to keep it up for a few seconds to change bays */
        a->kind_n += 1;
        if (a->kind_n > (int)(4 / fmaxf(dt, 1e-3f))) { a->kind = want; a->apt = wapt; a->kind_n = 0; }
    } else a->kind_n = 0;
    if (a->kind == KD_ARR) align_rwy(a, a->apt);
    else { a->rwy = a->rend = -1; a->dist_thr = -1; }
}

/* the levels a controller would have written on the strip */
static void track_cfl(Ac *a, double wt) {
    float c = a->fsel;
    if (c < 0) {
        /* no selected altitude: the level it settles at */
        if (fabsf(a->vr) < 150) c = roundf(a->alt / 1000) * 1000;
        else return;
    }
    c = roundf(c / 100) * 100;
    if (c < 1000 && !a->ship) return;
    if (a->ncfl == 0) { a->cfl[a->ncfl++] = c; a->cfl_t = wt; return; }
    if (fabsf(c - a->cfl[a->ncfl - 1]) < 500) { a->cfl_pend = -1; return; }
    if (a->cfl_pend != c) { a->cfl_pend = c; a->cfl_t = wt; return; }
    if (wt - a->cfl_t > 4 || shot_mode) {
        if (a->ncfl == 4) { memmove(a->cfl, a->cfl + 1, 3 * sizeof(float)); a->ncfl = 3; }
        a->cfl[a->ncfl++] = c;
        a->cfl_pend = -1;
    }
}

static int strip_busy(const Ac *a);

static void traffic_update(double wt, float dt) {
    for (int i = 0; i < napt; i++)
        for (int r = 0; r < apt[i].nr; r++) {
            apt[i].r[r].use_a *= expf(-dt / 600); apt[i].r[r].use_b *= expf(-dt / 600);
        }
    double stale = src == SRC_LOCAL ? 25 : src == SRC_REPLAY ? 1e9 : 75;
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->used) continue;
        if (!a->sim) {
            double tdr = fmin(fmax(wt - a->ft, 0), 45);
            double ce = exp(-(wt - a->et) / 1.6);
            float v = a->has_vel ? a->fgs / 3600.0f : 0;
            a->x = a->fx + sin(a->ftrk * DEG) * v * tdr + a->ex * ce;
            a->y = a->fy + cos(a->ftrk * DEG) * v * tdr + a->ey * ce;
            a->alt = a->falt + (a->ground ? 0 : a->fvr / 60.0f * (float)fmin(tdr, 20)) + a->ealt * (float)ce;
            a->gs = a->fgs; a->vr = a->fvr;
            a->trk = wrap360(a->trk + wrap180(a->ftrk - a->trk) * fminf(1, dt * 2.5f));
            if (wt - a->last > stale) a->gone = 1;
            int hmax = a->ship ? 30 : 9;
            if (floor(wt / HIST_DT) != floor(a->hlast / HIST_DT)) { if (a->hlast) push_hist(a, hmax); a->hlast = wt; }
        }
        if (a->gone) {
            a->fade -= dt / 2.5f;
            if (a->fade <= 0 && !strip_busy(a)) { a->used = 0; continue; }
            if (a->fade < 0) a->fade = 0;
        } else a->fade = fminf(1, a->fade + dt / 1.5f);
        classify(a, dt);
        track_cfl(a, wt);
        if (!a->sim) {
            /* the printed level: as first seen, or the highest a departure's reached */
            float l = a->alt >= 6000 ? roundf(a->alt / 1000) * 1000 : roundf(a->alt / 100) * 100;
            if (!a->rfl || (a->kind == KD_DEP && l > a->rfl)) a->rfl = l;
        }
        if (a->kind == KD_ARR && a->apt >= 0 && a->rwy >= 0 && a->dist_thr < 12 && !a->gone) {
            Rwy *w = &apt[a->apt].r[a->rwy];
            if (a->rend) w->use_b += dt; else w->use_a += dt;
        }
    }
    /* real traffic: the runways in use are the ones being landed on */
    if (src != SRC_SIM) {
        for (int i = 0; i < napt; i++)
            for (int r = 0; r < apt[i].nr; r++) {
                Rwy *w = &apt[i].r[r];
                w->act_a = w->use_a > 20; w->act_b = w->use_b > 20;
            }
    }
}

/* ---- views ---- */
typedef struct {
    float x0, y0, x1, y1;              /* screen rect */
    double cx, cy;                     /* nm at its centre */
    float ppn;                         /* px per nm */
    double hx, hy;                     /* rings around here */
    float outer;                       /* the bearing scale's radius, nm (0 none) */
    int approach;                      /* the close-up: runway names, ground traffic */
} View;

static void vp(const View *v, double x, double y, float *sx, float *sy) {
    *sx = (v->x0 + v->x1) * 0.5f + (float)(x - v->cx) * v->ppn;
    *sy = (v->y0 + v->y1) * 0.5f - (float)(y - v->cy) * v->ppn;
}

/* the scope follows something interesting, now and then */
static uint32_t fol_id;
static long fol_slot = -1;
static double fol_lost, fol_x, fol_y;
static float fol_range;
#define FOL_P 100.0

static int ac_visible(const Ac *a) { return a->used && !a->gone && !a->ground && a->fade > 0; }

static float interest(const Ac *a, float r0) {
    float d = sqrtf((float)(a->x * a->x + a->y * a->y));
    if (d > r0 * 0.85f || a->emerg == 7500) return 0;
    if (a->ship) return a->goodyear ? 3000 : 1000;
    if (a->emerg) return 500;
    if (a->kind == KD_ARR && a->dist_thr > 2 && a->dist_thr < 14) return 60;
    if (fabsf(a->vr) > 500) return 20;
    return 8;
}

static int any_ship(float r0) {
    for (int i = 0; i < MAXAC; i++) if (ac_visible(&ac[i]) && ac[i].ship && interest(&ac[i], r0) > 0) return 1;
    return 0;
}

/* how far into following (0..1), and the view it wants */
static float follow(double wt, float r0, double *tx, double *ty, float *trange) {
    if (opt_no_follow) return 0;
    long slot = (long)floor(wt / FOL_P);
    double ph = wt - slot * FOL_P;
    int ship = any_ship(r0);
    double w0 = ship ? 20 : 55, w1 = 96;
    if (ph < w0 || ph > w1) { fol_id = 0; return 0; }
    if (slot != fol_slot) {
        /* choose: an airship or an emergency outright, else a weighted pick */
        fol_slot = slot;
        fol_id = 0; fol_lost = 0;
        float tot = 0, top = 0;
        for (int i = 0; i < MAXAC; i++) if (ac_visible(&ac[i])) { float s = interest(&ac[i], r0); tot += s; if (s > top) top = s; }
        if (tot > 0) {
            Rng r = {hash32((uint32_t)slot * 2654435761u ^ opt_seed) | 1};
            float pick = rnd(&r) * tot;
            for (int i = 0; i < MAXAC; i++) {
                if (!ac_visible(&ac[i])) continue;
                float s = interest(&ac[i], r0);
                if (top >= 500) { if (s == top) { fol_id = ac[i].id; break; } continue; }
                if ((pick -= s) <= 0 && s > 0) { fol_id = ac[i].id; break; }
            }
        }
    }
    if (!fol_id) return 0;
    Ac *a = ac_find(fol_id, 0);
    if (a && ac_visible(a)) {
        float lead = a->ship ? 0 : 15;
        fol_x = a->x + sin(a->trk * DEG) * a->gs / 3600 * lead;
        fol_y = a->y + cos(a->trk * DEG) * a->gs / 3600 * lead;
        fol_range = a->ship ? 3.2f : a->emerg ? 14 : 10;
        fol_lost = 0;
    } else if (!fol_lost) fol_lost = wt;
    *tx = fol_x; *ty = fol_y; *trange = fol_range;
    float f = smooth((float)w0, (float)w0 + 7, (float)ph) * (1 - smooth((float)w1 - 7, (float)w1, (float)ph));
    if (fol_lost) f *= 1 - smooth(0, 5, (float)(wt - fol_lost));
    return f;
}

/* ---- colours ---- */
static const float COL_TXT[4] = {0.84f, 0.88f, 0.92f, 1};
static const float COL_DIM[4] = {0.52f, 0.58f, 0.65f, 1};
static const float COL_AMB[4] = {0.96f, 0.68f, 0.30f, 1};
static const float COL_CYAN[4] = {0.36f, 0.86f, 0.95f, 1};
static const float COL_RED[4] = {1.0f, 0.33f, 0.28f, 1};
static const float COL_GOLD[4] = {0.98f, 0.79f, 0.32f, 1};
static const float COL_MAP[4] = {0.62f, 0.69f, 0.76f, 1};
static const float COL_RING[4] = {0.40f, 0.47f, 0.55f, 1};

/* ---- the data blocks ---- */
static int st_lbl, st_lblb, st_lbls, st_tiny, st_tinyb, st_bar, st_barb, st_big;
static float lbl_lh;

static void block_lines(const Ac *a, char L[3][40]) {
    if (a->ship) {
        snprintf(L[0], 40, "%s", a->goodyear ? a->reg : a->cs);
        int ft = (int)lroundf(a->alt / 10) * 10;
        snprintf(L[1], 40, "%d,%03dFT", ft / 1000, ft % 1000);
        if (ft < 1000) snprintf(L[1], 40, "%dFT", ft);
        snprintf(L[2], 40, "%dKT %s", (int)lroundf(a->gs), a->goodyear ? "GOODYEAR" : "AIRSHIP");
        return;
    }
    int fl = (int)lroundf(a->alt / 100);
    const char *arr = a->vr > 400 ? S_UP : a->vr < -400 ? S_DN : " ";
    if (a->compact) {
        snprintf(L[0], 40, "%s", a->cs);
        snprintf(L[1], 40, "%03d%s", fl, arr);
        L[2][0] = 0;
        return;
    }
    if (a->fsel >= 0 && fabsf(a->fsel - a->alt) > 250) snprintf(L[1], 40, "%03d%s%03d", fl, arr, (int)lroundf(a->fsel / 100));
    else snprintf(L[1], 40, "%03d%s", fl, arr);
    snprintf(L[0], 40, "%s", a->cs);
    if (a->emerg && a->emerg != 7500) snprintf(L[0], 40, "%s %s", a->cs, a->emerg == 7700 ? "EMG" : "RCF");
    snprintf(L[2], 40, "%02d %s", (int)lroundf(a->gs / 10), a->type[0] ? a->type : "");
    if (a->emerg == 7500) snprintf(L[2], 40, "%02d %s", (int)lroundf(a->gs / 10), "7500");
}

static void block_size(const Ac *a, float *w, float *h) {
    char L[3][40];
    block_lines(a, L);
    float m = 0;
    if (a->compact) {
        *w = text_width(st_lblb, L[0], 0) + text_width(st_lbl, L[1], 0) + lbl_lh * 0.5f + 4;
        *h = lbl_lh;
        return;
    }
    for (int k = 0; k < 3; k++) m = fmaxf(m, text_width(k ? st_lbl : st_lblb, L[k], 0));
    *w = m + 4;
    *h = lbl_lh * 3;
}

/* where the block sits for a leader at angle ang (screen radians) */
static float leader_len(void) { return lbl_lh * 2.0f; }
static void block_pos(float px, float py, float ang, float lr, float bw, float bh, float *bx, float *by, float *ex, float *ey) {
    float L = leader_len() * lr;
    float c = cosf(ang), s = sinf(ang);
    *ex = px + c * L; *ey = py + s * L;
    float fx = clampf(c * 1.6f, -1, 1), fy = clampf(s * 1.6f, -1, 1);
    *bx = *ex - bw * (0.5f - 0.5f * fx);
    *by = *ey - bh * (0.5f - 0.5f * fy);
}

static float overlap(float ax0, float ay0, float ax1, float ay1, float bx0, float by0, float bx1, float by1) {
    float w = fminf(ax1, bx1) - fmaxf(ax0, bx0), h = fminf(ay1, by1) - fmaxf(ay0, by0);
    return w > 0 && h > 0 ? w * h : 0;
}

static double lbl_next;
static int lbl_prio(const Ac *a) {
    return (a->id == fol_id ? 1000 : 0) + (a->emerg ? 500 : 0) + (a->ship ? 400 : 0) + (int)(a->alt / 1000);
}

/* choose each block's side: least overlap with blocks already placed,
 * targets and speed vectors, with a little stickiness */
static void arrange_labels(const View *v, double wt, int force) {
    if (wt < lbl_next && !force) return;
    lbl_next = wt + 0.25;
    int idx[MAXAC], n = 0;
    float sx[MAXAC], sy[MAXAC];
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->used || a->ground || a->fade <= 0) continue;
        vp(v, a->x, a->y, &sx[i], &sy[i]);
        if (sx[i] < v->x0 - 200 || sx[i] > v->x1 + 200 || sy[i] < v->y0 - 200 || sy[i] > v->y1 + 200) continue;
        idx[n++] = i;
    }
    /* in a crowd, the less important blocks shrink to a line */
    float cr = lbl_lh * 5.5f;
    for (int k = 0; k < n; k++) {
        Ac *a = &ac[idx[k]];
        int nb = 0;
        for (int j = 0; j < n; j++) {
            float dx = sx[idx[j]] - sx[idx[k]], dy = sy[idx[j]] - sy[idx[k]];
            if (j != k && dx * dx + dy * dy < cr * cr) nb++;
        }
        int vip = a->id == fol_id || (a->emerg && a->emerg != 7500) || a->ship;
        if (vip) a->compact = 0;
        else if (nb >= 4) a->compact = 1;
        else if (nb <= 2) a->compact = 0;
        block_size(a, &a->bw, &a->bh);
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (lbl_prio(&ac[idx[j]]) > lbl_prio(&ac[idx[i]])) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    static float box[MAXAC][4];
    for (int k = 0; k < n; k++) {
        Ac *a = &ac[idx[k]];
        float px = sx[idx[k]], py = sy[idx[k]];
        float best = 1e30f, bang = -PI / 4;
        int bslot = 1;
        float vx = sinf(a->trk * DEG) * a->gs / 60 * v->ppn, vy = -cosf(a->trk * DEG) * a->gs / 60 * v->ppn;
        for (int s = 0; s < 16; s++) {
            float ang = (-3 + (s & 7)) * PI / 4;   /* -135, -90, -45 (up right), 0, ... */
            float bx, by, ex, ey;
            block_pos(px, py, ang, s & 8 ? 1.9f : 1, a->bw, a->bh, &bx, &by, &ex, &ey);
            float x0 = bx - 2, y0 = by - 2, x1 = bx + a->bw + 2, y1 = by + a->bh + 2;
            float c = 0;
            for (int j = 0; j < k; j++) c += overlap(x0, y0, x1, y1, box[j][0], box[j][1], box[j][2], box[j][3]) * 3;
            for (int j = 0; j < n; j++) {
                float tx = sx[idx[j]], ty = sy[idx[j]];
                c += overlap(x0, y0, x1, y1, tx - 7, ty - 7, tx + 7, ty + 7) * (j == k ? 6 : 2);
            }
            /* its own speed vector */
            for (int q = 1; q <= 4; q++) {
                float qx = px + vx * q / 4, qy = py + vy * q / 4;
                if (qx > x0 && qx < x1 && qy > y0 && qy < y1) c += a->bw * a->bh * 0.25f;
            }
            if (x0 < v->x0 || x1 > v->x1 || y0 < v->y0 || y1 > v->y1) c += a->bw * a->bh * 0.8f;
            if (s != a->slot && a->slot >= 0) c += a->bw * a->bh * 0.12f;
            if ((s & 7) != 2) c += a->bw * a->bh * 0.02f;   /* up and right, all else equal */
            if (s & 8) c += a->bw * a->bh * 0.05f;          /* short leaders */
            if (c < best) { best = c; bang = ang; bslot = s; }
        }
        if (a->slot < 0 || force) { a->lang = bang; a->lang_v = 0; a->lrad = bslot & 8 ? 1.9f : 1; a->lrad_v = 0; }
        a->slot = bslot;
        float bx, by, ex, ey;
        block_pos(px, py, bang, bslot & 8 ? 1.9f : 1, a->bw, a->bh, &bx, &by, &ex, &ey);
        box[k][0] = bx - 2; box[k][1] = by - 2; box[k][2] = bx + a->bw + 2; box[k][3] = by + a->bh + 2;
    }
}

static void ease_labels(float dt) {
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->used || a->slot < 0) continue;
        float tgt = (-3 + (a->slot & 7)) * PI / 4;
        float d = remainderf(tgt - a->lang, 2 * PI);
        /* a critically damped spring, ~0.6 s */
        float w = 7, h = fminf(dt, 0.05f);
        a->lang_v += (w * w * d - 2 * w * a->lang_v) * h;
        a->lang += a->lang_v * h;
        float lr = a->slot & 8 ? 1.9f : 1;
        a->lrad_v += (w * w * (lr - a->lrad) - 2 * w * a->lrad_v) * h;
        a->lrad += a->lrad_v * h;
    }
}

/* the colour an aircraft is drawn in */
static const float *ac_col(const Ac *a, int *vert) {
    *vert = fabsf(a->vr) > 400 && !a->ship;
    if (a->emerg == 7700 || a->emerg == 7600) return COL_RED;
    if (a->ship) return COL_GOLD;
    if (a->id == fol_id) return COL_CYAN;
    return COL_TXT;
}

static void draw_airship(float x, float y, float ang, float s, const float *c, float lw) {
    /* an envelope, heading-aligned, with cruciform fins and a gondola */
    float ca_ = cosf(ang), sa = sinf(ang);
    ellipse(x, y, 9.5f * s, 3.9f * s, ang, lw, c);
    float tx = x - ca_ * 8.6f * s, ty = y - sa * 8.6f * s;
    float nx = -sa, ny = ca_;
    line(tx, ty, tx - ca_ * 2.6f * s + nx * 3.4f * s, ty - sa * 2.6f * s + ny * 3.4f * s, lw, c);
    line(tx, ty, tx - ca_ * 2.6f * s - nx * 3.4f * s, ty - sa * 2.6f * s - ny * 3.4f * s, lw, c);
    line(tx, ty, tx - ca_ * 3.2f * s, ty - sa * 3.2f * s, lw, c);
    rect_r(x + ca_ * 1.0f * s + nx * 4.3f * s, y + sa * 1.0f * s + ny * 4.3f * s, 2.2f * s, 0.8f * s, ang, 0.6f * s, c, M_FLAT, 0);
}

static void draw_traffic(const View *v, double wt, float sel_pulse) {
    float lw = fmaxf(1.1f, lbl_lh / 12.5f);
    for (int pass = 0; pass < 2; pass++)
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->used || a->fade <= 0) continue;
        if (a->ground && !v->approach) continue;
        float px, py;
        vp(v, a->x, a->y, &px, &py);
        if (px < v->x0 - 300 || px > v->x1 + 300 || py < v->y0 - 300 || py > v->y1 + 300) continue;
        int vert;
        const float *c = ac_col(a, &vert);
        alpha_mul = a->fade;
        if (a->ground) {
            if (pass == 0) disc(px, py, 1.8f, ca(COL_DIM, 0.6f));
            alpha_mul = 1;
            continue;
        }
        float scr = a->trk * DEG - PI / 2;
        if (pass == 0) {
            /* history dots, then the speed vector */
            for (int k = 0; k < a->hn; k++) {
                float hx, hy;
                vp(v, a->hx[k], a->hy[k], &hx, &hy);
                float f = (float)(k + 1) / (float)(a->hn + 1);
                disc(hx, hy, a->ship ? 1.9f : 1.6f, ca(a->ship ? COL_GOLD : COL_RING, (a->ship ? 0.25f : 0.18f) + 0.6f * f));
            }
            float vl = a->gs / 60 * v->ppn;
            if (a->ship) vl *= 2;
            vl = fminf(vl, fmaxf(28, 0.075f * fminf(v->x1 - v->x0, v->y1 - v->y0)));
            if (vl > 4) line(px + cosf(scr) * 5, py + sinf(scr) * 5, px + cosf(scr) * (vl + 5), py + sinf(scr) * (vl + 5), lw * 0.9f, ca(c, 0.75f));
            alpha_mul = 1;
            continue;
        }
        /* the target */
        if (a->ship) {
            float s = a->goodyear ? 1.25f : 1.05f;
            glow(px, py, 26 * s, C4(0.98f, 0.79f, 0.32f, -0.10f));
            draw_airship(px, py, scr, s * lbl_lh / 14, c, lw * 1.1f);
        } else {
            ring(px, py, 3.6f, lw * 1.1f, c);
            disc(px, py, 1.1f, c);
        }
        if (a->emerg == 7700 || a->emerg == 7600) {
            float pz = 0.5f + 0.5f * sinf((float)wt * 4);
            ring(px, py, 10 + 4 * pz, 1.4f, ca(COL_RED, 0.35f + 0.4f * pz));
        }
        if (a->id == fol_id && sel_pulse > 0) {
            float b = (a->ship ? 16 : 9) + 3 * (1 - sel_pulse), k = 4;
            const float *cc = ca(a->ship ? COL_GOLD : COL_CYAN, sel_pulse);
            for (int q = 0; q < 4; q++) {
                float sx = q & 1 ? 1 : -1, sy = q & 2 ? 1 : -1;
                line(px + sx * b, py + sy * b, px + sx * (b - k), py + sy * b, 1.3f, cc);
                line(px + sx * b, py + sy * b, px + sx * b, py + sy * (b - k), 1.3f, cc);
            }
        }
        /* the data block on its leader */
        char L[3][40];
        block_lines(a, L);
        float bw, bh;
        block_size(a, &bw, &bh);
        float bx, by, ex, ey;
        block_pos(px, py, a->lang, a->lrad > 0.5f ? a->lrad : 1, bw, bh, &bx, &by, &ex, &ey);
        float lc = cosf(a->lang), ls = sinf(a->lang);
        line(px + lc * 7, py + ls * 7, ex - lc * 2, ey - ls * 2, 1.0f, ca(c, 0.55f));
        if (a->emerg == 7700 || a->emerg == 7600) rect(bx - 3, by - 2, bx + bw + 3, by + bh + 2, 3, ca(COL_RED, 0.18f), M_FLAT);
        if (a->ship) rect(bx - 4, by - 3, bx + bw + 4, by + bh + 3, 3, C4(0.10f, 0.08f, 0.03f, 0.72f), M_FLAT);
        else rect(bx - 2, by - 1, bx + bw + 2, by + bh + 1, 2, C4(0.03f, 0.045f, 0.06f, 0.55f), M_FLAT);
        float asc = cap_h(st_lbl);
        float y0 = by + (lbl_lh + asc) * 0.5f;
        if (a->compact) {
            float w0 = text(st_lblb, bx + 2, y0, -1, L[0], ca(c, 0.85f));
            text(st_lbl, bx + 2 + w0 + lbl_lh * 0.5f, y0, -1, L[1], vert && c == COL_TXT ? COL_AMB : ca(COL_DIM, 1));
            alpha_mul = 1;
            continue;
        }
        text(st_lblb, bx + 2, y0, -1, L[0], c);
        const float *c2 = vert && c == COL_TXT ? COL_AMB : c;
        text(st_lbl, bx + 2, y0 + lbl_lh, -1, L[1], c2);
        text(st_lbl, bx + 2, y0 + lbl_lh * 2, -1, L[2], a->emerg == 7500 ? ca(COL_RED, 0.8f) : ca(c == COL_TXT ? COL_DIM : c, c == COL_TXT ? 1 : 0.75f));
        alpha_mul = 1;
    }
}

/* ---- the map under it all ---- */
static float nice_step(float range) {
    float s = range / 4;
    float p = powf(10, floorf(log10f(s)));
    float m = s / p;
    return (m < 1.5f ? 1 : m < 3.5f ? 2.5f : m < 7.5f ? 5 : 10) * p;
}

static void draw_map(const View *v, int active_only) {
    /* the graticule: every degree, or half when close */
    float span_nm = (v->x1 - v->x0) / v->ppn;
    float gstep = span_nm > 120 ? 1.0f : span_nm > 30 ? 0.5f : 0.25f;
    const float *gc = C4(0.35f, 0.42f, 0.50f, 0.09f);
    double xa = v->cx - (v->x1 - v->x0) * 0.6 / v->ppn, xb = v->cx + (v->x1 - v->x0) * 0.6 / v->ppn;
    double ya = v->cy - (v->y1 - v->y0) * 0.6 / v->ppn, yb = v->cy + (v->y1 - v->y0) * 0.6 / v->ppn;
    for (double la = floor((lat0 + ya / 60) / gstep) * gstep; la <= lat0 + yb / 60; la += gstep) {
        float sx0, sy, sx1, sy1;
        vp(v, xa, (la - lat0) * 60, &sx0, &sy);
        vp(v, xb, (la - lat0) * 60, &sx1, &sy1);
        line_d(sx0, sy, sx1, sy1, 1, gc, 9, 0.5f);
    }
    double lo_a = lon0 + xa / 60 / coslat0, lo_b = lon0 + xb / 60 / coslat0;
    for (double lo = floor(lo_a / gstep) * gstep; lo <= lo_b; lo += gstep) {
        float sx, sy0, sx1, sy1;
        vp(v, (lo - lon0) * 60 * coslat0, ya, &sx, &sy0);
        vp(v, (lo - lon0) * 60 * coslat0, yb, &sx1, &sy1);
        line_d(sx, sy0, sx1, sy1, 1, gc, 9, 0.5f);
    }
    /* airports: extended centrelines on the runways in use, then the runways */
    for (int i = 0; i < napt; i++) {
        const Apt *A = &apt[i];
        float ax, ay;
        vp(v, A->x, A->y, &ax, &ay);
        float reach = 20 * v->ppn;
        if (ax < v->x0 - reach || ax > v->x1 + reach || ay < v->y0 - reach || ay > v->y1 + reach) continue;
        int any_act = 0;
        for (int r = 0; r < A->nr; r++) any_act |= A->r[r].act_a | A->r[r].act_b;
        for (int r = 0; r < A->nr; r++) {
            const Rwy *w = &A->r[r];
            for (int e = 0; e < 2; e++) {
                int act = e ? w->act_b : w->act_a;
                if (!act && (any_act || active_only || A->d->rank > 1)) continue;
                if (A->d->rank == 3 && v->ppn < 22) continue;
                if (A->d->rank == 2 && v->ppn < 9 && A->dist > 30) continue;
                float tx, ty, ux, uy, hdg;
                rwy_end(w, e, &tx, &ty, &ux, &uy, &hdg);
                float len = act ? (v->approach ? 15 : 12) : 6;
                float a = act ? 0.42f : 0.16f;
                float sx0, sy0, sx1, sy1;
                vp(v, tx, ty, &sx0, &sy0);
                vp(v, tx - ux * len, ty - uy * len, &sx1, &sy1);
                line(sx0, sy0, sx1, sy1, 1.0f, ca(COL_MAP, a));
                /* a tick every mile, longer at 5 and 10 */
                if (v->ppn * 1 > 5) {
                    float nx = -uy, ny = ux;
                    for (int k = 1; k <= (int)len; k++) {
                        float kx, ky;
                        vp(v, tx - ux * k, ty - uy * k, &kx, &ky);
                        float tl = (k % 5 == 0 ? 7.0f : 3.5f) * fminf(1.6f, fmaxf(0.8f, v->ppn / 14));
                        line(kx + nx * tl, ky - ny * tl, kx - nx * tl, ky + ny * tl, 1.0f, ca(COL_MAP, a * (k % 5 == 0 ? 1.1f : 0.8f)));
                    }
                }
                if (v->approach && act) {
                    float lx, ly;
                    vp(v, tx - ux * (len + 1.0f), ty - uy * (len + 1.0f), &lx, &ly);
                    text(st_tinyb, lx, ly + cap_h(st_tinyb) * 0.5f, 0, e ? w->nb : w->na, ca(COL_MAP, 0.7f));
                }
            }
        }
        for (int r = 0; r < A->nr; r++) {
            const Rwy *w = &A->r[r];
            float sx0, sy0, sx1, sy1;
            vp(v, w->x0, w->y0, &sx0, &sy0);
            vp(v, w->x1, w->y1, &sx1, &sy1);
            float wd = fmaxf(A->d->rank == 1 ? 2.4f : 1.8f, 0.03f * v->ppn);
            line(sx0, sy0, sx1, sy1, wd, ca(COL_MAP, A->d->rank == 3 ? 0.55f : 0.85f));
            if (0) {
                /* the runway names at the thresholds */
                float dx = sx1 - sx0, dy = sy1 - sy0, l = sqrtf(dx * dx + dy * dy) + 1e-3f;
                dx /= l; dy /= l;
                text(st_tiny, sx0 - dx * 12, sy0 - dy * 12 + cap_h(st_tiny) * 0.5f, 0, w->na, ca(COL_MAP, 0.55f));
                text(st_tiny, sx1 + dx * 12, sy1 + dy * 12 + cap_h(st_tiny) * 0.5f, 0, w->nb, ca(COL_MAP, 0.55f));
            }
        }
        /* the name, off to the side of the main runway */
        const Rwy *m = &A->r[0];
        float nx = -(m->y1 - m->y0), ny = m->x1 - m->x0, nl = sqrtf(nx * nx + ny * ny) + 1e-4f;
        nx /= nl; ny /= nl;
        if (ny < 0) { nx = -nx; ny = -ny; }
        float off = fmaxf(m->len_nm * 0.5f * v->ppn, 6) + 12;
        int st = A->d->rank == 1 ? st_tinyb : st_tiny;
        text_t(st, ax - nx * off + 2, ay - ny * off, nx > 0.3f ? 1 : nx < -0.3f ? -1 : 0, A->d->icao, ca(COL_MAP, A->d->rank == 3 ? 0.4f : 0.6f), 1);
    }
}

/* labels for the range rings and the bearing scale (the rings themselves
 * are in the background shader) */
static void draw_ring_labels(const View *v, float step, float step_a) {
    float hx, hy;
    vp(v, v->hx, v->hy, &hx, &hy);
    float maxr = hypotf(fmaxf(hx - v->x0, v->x1 - hx), fmaxf(hy - v->y0, v->y1 - hy)) / v->ppn;
    float lb = 135 * DEG;
    for (int k = 1; k * step * 0.5f < maxr && k < 60; k++) {
        float r = k * step * 0.5f;
        float a = (k & 1) ? step_a : 1;
        if (a < 0.05f) continue;
        if (v->outer > 0 && fabsf(r - v->outer) < 0.01f) continue;
        float x = hx + sinf(lb) * r * v->ppn, y = hy - cosf(lb) * r * v->ppn;
        if (x < v->x0 + 10 || x > v->x1 - 30 || y < v->y0 + 10 || y > v->y1 - 10) continue;
        char b[16];
        snprintf(b, sizeof b, r == floorf(r) ? "%.0f" : "%.1f", r);
        rect(x - text_width(st_tiny, b, 0) * 0.5f - 3, y - cap_h(st_tiny) * 0.5f - 3, x + text_width(st_tiny, b, 0) * 0.5f + 3, y + cap_h(st_tiny) * 0.5f + 3, 2, C4(0.035f, 0.05f, 0.065f, 0.9f), M_FLAT);
        text(st_tiny, x, y + cap_h(st_tiny) * 0.5f, 0, b, ca(COL_RING, 0.85f * a));
    }
    if (v->outer > 0) {
        float R = v->outer * v->ppn;
        for (int d = 0; d < 360; d += 30) {
            float s = sinf(d * DEG), c = cosf(d * DEG);
            float x = hx + s * (R + lbl_lh * 1.1f), y = hy - c * (R + lbl_lh * 1.1f);
            if (x < v->x0 + 20 || x > v->x1 - 20 || y < v->y0 + 12 || y > v->y1 - 8) continue;
            char b[8];
            snprintf(b, sizeof b, "%03d", d ? d : 360);
            if (!d) { line(hx, hy - R - 2, hx - 5, hy - R - 11, 1.5f, ca(COL_MAP, 0.8f)); line(hx, hy - R - 2, hx + 5, hy - R - 11, 1.5f, ca(COL_MAP, 0.8f)); y -= 10; }
            text(st_tiny, x, y + cap_h(st_tiny) * 0.5f, 0, d ? b : "N", ca(COL_MAP, d ? 0.55f : 0.85f));
        }
    }
    /* a little compass rose on the receiver */
    const float *rc = ca(COL_MAP, 0.35f);
    for (int k = 0; k < 4; k++) {
        float s = sinf(k * PI / 2), c = cosf(k * PI / 2), l = k == 0 ? 16 : 10;
        line(hx + s * 4, hy - c * 4, hx + s * l, hy - c * l, 1.2f, rc);
    }
    ring(hx, hy, 2.2f, 1.2f, rc);
}

/* ---- lock info ---- */
static void fmt_away(char *b, size_t n, double wt) {
    long s = (long)(wt - (double)locked_at);
    if (s < 0) s = 0;
    if (s >= 3600) snprintf(b, n, "%ld:%02ld:%02ld", s / 3600, s / 60 % 60, s % 60);
    else snprintf(b, n, "%ld:%02ld", s / 60, s % 60);
}

static const char *src_credit(void) {
    if (src == SRC_PUBLIC && wall() < live_until) return "data: adsb.lol";
    if (src == SRC_LOCAL && wall() < live_until) return "data: local receiver";
    if (src == SRC_REPLAY) return "data: replay";
    if (src == SRC_PROBE) return "simulated traffic";
    return "simulated traffic";
}
static int data_live(double wt) { return src == SRC_REPLAY || ((src == SRC_LOCAL || src == SRC_PUBLIC) && wt < live_until); }

static int count_traffic(void) {
    int n = 0;
    for (int i = 0; i < MAXAC; i++) if (ac_visible(&ac[i])) n++;
    return n;
}

/* the bar along the top of a display: sector, clocks, time away; and on the
 * right whatever the display wants to say */
static float bar_h;
static void draw_topbar(float x0, float y0, float x1, double wt, const char *right1, const char *right2) {
    float h = bar_h;
    rect(x0, y0, x1, y0 + h, 0, C4(0.055f, 0.07f, 0.088f, 0.96f), M_PANEL);
    line(x0, y0 + h, x1, y0 + h, 1, C4(0.30f, 0.37f, 0.45f, 0.55f));
    float cy = y0 + h * 0.5f + cap_h(st_bar) * 0.5f;
    float x = x0 + h * 0.6f;
    const float *lab = C4(0.50f, 0.57f, 0.65f, 1), *val = C4(0.88f, 0.92f, 0.96f, 1);
    char b[64];
    time_t ts = (time_t)wt;
    struct tm lt, ut;
    localtime_r(&ts, &lt);
    gmtime_r(&ts, &ut);
    struct { const char *l; char v[40]; const float *c; } F[4];
    F[0].l = "SECTOR"; snprintf(F[0].v, 40, "%s", sector); F[0].c = val;
    F[1].l = "UTC"; snprintf(F[1].v, 40, "%02d:%02d:%02d", ut.tm_hour, ut.tm_min, ut.tm_sec); F[1].c = val;
    F[2].l = "LCL"; snprintf(F[2].v, 40, "%02d:%02d:%02d", lt.tm_hour, lt.tm_min, lt.tm_sec); F[2].c = val;
    F[3].l = "AWAY"; fmt_away(F[3].v, 40, wt); F[3].c = COL_AMB;
    for (int k = 0; k < 4; k++) {
        x += text_t(st_bar, x, cy, -1, F[k].l, lab, 1) + h * 0.3f;
        x += text(st_barb, x, cy, -1, F[k].v, F[k].c) + h * 0.55f;
        line(x - h * 0.28f, y0 + h * 0.25f, x - h * 0.28f, y0 + h * 0.75f, 1, C4(0.3f, 0.36f, 0.43f, 0.6f));
        x += h * 0.2f;
    }
    float xr = x1 - h * 0.6f;
    snprintf(b, sizeof b, "%s", src_credit());
    xr -= text(st_bar, xr, cy, 1, b, lab) + h * 0.55f;
    if (right1) {
        line(xr + h * 0.28f, y0 + h * 0.25f, xr + h * 0.28f, y0 + h * 0.75f, 1, C4(0.3f, 0.36f, 0.43f, 0.6f));
        xr -= text(st_barb, xr, cy, 1, right1, val) + h * 0.3f;
    }
    if (right2) {
        xr -= h * 0.25f;
        xr -= text(st_barb, xr, cy, 1, right2, COL_CYAN) + h * 0.3f;
    }
    (void)xr;
}

/* a readout of the aircraft being followed */
static void draw_track_info(float x, float y, const Ac *a, float alpha) {
    if (!a || alpha <= 0) return;
    alpha_mul = alpha;
    char L[6][48];
    int n = 0;
    const float *hc = a->ship ? COL_GOLD : a->emerg && a->emerg != 7500 ? COL_RED : COL_CYAN;
    snprintf(L[n++], 48, "%s", a->ship && a->goodyear ? a->ship_name : a->cs);
    if (a->ship) snprintf(L[n++], 48, "%s  %s", a->goodyear ? "GOODYEAR AIRSHIP" : "AIRSHIP", a->reg[0] ? a->reg : "");
    else snprintf(L[n++], 48, "%-5s %s", a->type[0] ? a->type : "----", a->reg);
    if (a->ship) snprintf(L[n++], 48, "ALT  %5d FT", (int)lroundf(a->alt));
    else snprintf(L[n++], 48, "ALT  FL%03d %s", (int)lroundf(a->alt / 100), a->vr > 300 ? S_UP : a->vr < -300 ? S_DN : "");
    snprintf(L[n++], 48, "GS   %3d KT  TRK %03d", (int)lroundf(a->gs), (int)lroundf(a->trk) % 360);
    snprintf(L[n++], 48, "V/S %+5d FPM", (int)(lroundf(a->vr / 50) * 50));
    snprintf(L[n++], 48, "SSR  %s", a->sqk[0] ? a->sqk : "----");
    float w = 0;
    for (int k = 0; k < n; k++) w = fmaxf(w, text_width(k ? st_lbl : st_lblb, L[k], 0));
    float pad = lbl_lh * 0.7f, h = lbl_lh * (n + 0.3f) + pad * 2;
    w += pad * 2;
    y -= h;
    rect(x, y, x + w, y + h, 4, C4(0.04f, 0.055f, 0.07f, 0.88f), M_FLAT);
    line(x, y, x, y + h, 2, hc);
    float ty = y + pad + cap_h(st_lbl) + lbl_lh * 0.1f;
    for (int k = 0; k < n; k++) {
        text(k ? st_lbl : st_lblb, x + pad, ty, -1, L[k], k ? (k == 1 ? COL_DIM : COL_TXT) : hc);
        ty += lbl_lh;
    }
    alpha_mul = 1;
}

/* a far airship (from the watch): which way, how far */
static const char *compass8(float b) {
    static const char *P[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    return P[(int)floorf(wrap360(b + 22.5f) / 45) & 7];
}
static void draw_far_ship(float x, float y, float r0) {
    if (!far_ship.on) return;
    double d, b;
    gc_from_home(far_ship.lat, far_ship.lon, &d, &b);
    if (d < r0 * 0.9) return;          /* it's on the scope */
    char s[96];
    snprintf(s, sizeof s, "AIRSHIP %s " S_DOT " %s " S_DOT " %.0f NM %s", far_ship.reg, far_ship.name, d, compass8((float)b));
    float w = text_width(st_lblb, s, 0), h = lbl_lh * 1.9f;
    rect(x, y - h, x + w + h * 1.6f, y, 4, C4(0.10f, 0.08f, 0.03f, 0.85f), M_FLAT);
    line(x, y - h, x, y, 2, COL_GOLD);
    float ax = x + h * 0.75f, ay = y - h * 0.5f, ang = (float)b * DEG;
    float dx = sinf(ang), dy = -cosf(ang);
    line(ax - dx * h * 0.3f, ay - dy * h * 0.3f, ax + dx * h * 0.32f, ay + dy * h * 0.32f, 2, COL_GOLD);
    line(ax + dx * h * 0.32f, ay + dy * h * 0.32f, ax + dx * h * 0.05f - dy * h * 0.2f, ay + dy * h * 0.05f + dx * h * 0.2f, 2, COL_GOLD);
    line(ax + dx * h * 0.32f, ay + dy * h * 0.32f, ax + dx * h * 0.05f + dy * h * 0.2f, ay + dy * h * 0.05f - dx * h * 0.2f, 2, COL_GOLD);
    text(st_lblb, x + h * 1.4f, y - h * 0.5f + cap_h(st_lblb) * 0.5f, -1, s, COL_GOLD);
}

/* ---- the flight strip board ---- */
enum { BAY_ARR, BAY_DEP, BAY_OVF, NBAY };
static const char *BAY_NAME[NBAY] = {"ARRIVALS", "DEPARTURES", "OVERFLIGHTS"};
static int st_cs, st_sm, st_smb, st_hand, st_handb, st_tape, st_led_lab, st_dymo;
static float strip_w, strip_h;
static int strip_cols;

static int strip_busy(const Ac *a) { return a->strip && a->salpha > 0.01f; }

static int ac_bay(const Ac *a) {
    if (a->kind == KD_ARR) return BAY_ARR;
    if (a->kind == KD_DEP) return BAY_DEP;
    return BAY_OVF;
}

static const char *wake(const Ac *a) {
    if (!strcmp(a->type, "A388")) return "J";
    switch (a->cat) {
    case 0xA1: return "L";
    case 0xA2: return "S";
    case 0xA5: return "H";
    case 0xB2: return "L";
    default: return "M";
    }
}

static void fmt_level(char *b, size_t n, float alt) {
    if (alt < 6000) snprintf(b, n, "A%03d", (int)lroundf(alt / 100));
    else snprintf(b, n, "%d", (int)lroundf(alt / 100));
}

static const float INK[4] = {0.11f, 0.17f, 0.42f, 0.92f};
static const float INK_RED[4] = {0.72f, 0.12f, 0.10f, 0.92f};
static const float PRINT[4] = {0.16f, 0.16f, 0.17f, 0.95f};
static const float RULE[4] = {0.72f, 0.52f, 0.50f, 0.55f};

/* handwriting: a little tilt and wobble, the same every time for a strip */
static float hand(float x, float y, int align, const char *s, const float *c, uint32_t seed, int big) {
    float ang = ((hash32(seed) & 255) / 255.0f - 0.5f) * 0.09f;
    return text_ex(big ? st_handb : st_hand, x, y, align, s, c, 0, ang, seed | 1);
}

/* a pen circle round something, not quite closed */
static void pen_circle(float x, float y, float rx, float ry, const float *c, uint32_t seed) {
    float a0 = ((hash32(seed) & 255) / 255.0f) * 2 * PI;
    int n = 22;
    float px = 0, py = 0;
    for (int i = 0; i <= n; i++) {
        float t = a0 + (float)i / n * 2 * PI * 1.08f;
        float wob = 1 + 0.06f * sinf(t * 3 + a0);
        float qx = x + cosf(t) * rx * wob, qy = y + sinf(t) * ry * wob;
        if (i) line(px, py, qx, qy, 1.4f, c);
        px = qx; py = qy;
    }
}

static void pen_tick(float x, float y, float s, const float *c) {
    line(x - s * 0.5f, y - s * 0.05f, x - s * 0.15f, y + s * 0.35f, 1.8f, c);
    line(x - s * 0.15f, y + s * 0.35f, x + s * 0.55f, y - s * 0.5f, 1.8f, c);
}

static void holder_col(int bay, const Ac *a, float *c) {
    static const float C[NBAY][3] = {{0.80f, 0.64f, 0.36f}, {0.27f, 0.45f, 0.74f}, {0.80f, 0.47f, 0.58f}};
    if (a->ship) { c[0] = 0.13f; c[1] = 0.25f; c[2] = 0.55f; c[3] = 1; return; }
    c[0] = C[bay][0]; c[1] = C[bay][1]; c[2] = C[bay][2]; c[3] = 1;
}

static void draw_strip(const Ac *a, float x, float y, float w, float h, double wt) {
    uint32_t seed = hash32(a->id * 2654435761u);
    float hc[4];
    holder_col(a->bay, a, hc);
    shadow(x + 2, y + 3, x + w + 2, y + h + 4, 3, h * 0.08f, C4(0, 0, 0, 0.55f));
    /* the holder, then the paper in it */
    rect(x, y, x + w, y + h, h * 0.06f, hc, M_HOLDER);
    float lip = h * 0.34f, m = h * 0.07f;
    float px0 = x + lip, py0 = y + m, px1 = x + w - m * 0.8f, py1 = y + h - m;
    float paper[4] = {0.95f, 0.93f, 0.86f, 1};
    if (a->ship) { paper[0] = 0.97f; paper[1] = 0.91f; paper[2] = 0.70f; }
    rect(px0, py0, px1, py1, 1.5f, paper, M_PAPER);
    /* the holder's lip: its letter */
    const char *tag = a->ship ? "S" : a->bay == BAY_ARR ? "A" : a->bay == BAY_DEP ? "D" : "O";
    text(st_smb, x + lip * 0.5f, y + h * 0.5f + cap_h(st_smb) * 0.5f, 0, tag, C4(1, 1, 1, 0.55f));
    float pw = px1 - px0, ph = py1 - py0;
    static const float FX[] = {0, 0.25f, 0.37f, 0.49f, 0.64f, 0.76f, 0.89f, 1};
    for (int k = 1; k < 7; k++) {
        float fx = px0 + pw * FX[k];
        line(fx, py0 + 1, fx, py1 - 1, 1, RULE);
    }
    line(px0 + pw * FX[1], py0 + ph * 0.5f, px0 + pw * FX[3], py0 + ph * 0.5f, 1, RULE);
    line(px0 + pw * FX[4], py0 + ph * 0.5f, px0 + pw * FX[5], py0 + ph * 0.5f, 1, RULE);
    float top = py0 + ph * 0.25f + cap_h(st_sm) * 0.5f, bot = py0 + ph * 0.75f + cap_h(st_sm) * 0.5f;
    float pad = ph * 0.12f;
    char b[48];
    /* callsign, and type/wake under it */
    text(st_cs, px0 + pad, py0 + ph * 0.46f + cap_h(st_cs) * 0.5f, -1, a->cs, PRINT);
    snprintf(b, sizeof b, "%s/%s", a->type[0] ? a->type : "ZZZZ", wake(a));
    text(st_sm, px0 + pad, py1 - ph * 0.1f, -1, b, ca(PRINT, 0.8f));
    /* SSR and registration */
    snprintf(b, sizeof b, "A%s", a->sqk[0] ? a->sqk : "----");
    text(st_smb, px0 + pw * FX[1] + pad, top, -1, b, a->emerg ? INK_RED : PRINT);
    text(st_sm, px0 + pw * FX[1] + pad, bot, -1, a->reg[0] ? a->reg : a->hex, ca(PRINT, 0.75f));
    if (a->emerg == 7500) line(px0 + pw * FX[1] + pad, top + 3, px0 + pw * FX[1] + pad + text_width(st_smb, b, 0), top + 3, 1.4f, INK_RED);
    /* the level when first seen, printed; the level now, small */
    fmt_level(b, sizeof b, a->rfl);
    if (a->ship) snprintf(b, sizeof b, "%d", (int)lroundf(a->alt / 10) * 10);
    text(st_smb, px0 + pw * FX[2] + pad, top, -1, a->ship ? "ALT" : "RFL", ca(PRINT, 0.6f));
    text(st_smb, px0 + pw * FX[2] + pad, bot, -1, b, PRINT);
    /* cleared levels, handwritten, the old ones struck out */
    float cx = px0 + pw * FX[3] + pad, cw = pw * (FX[4] - FX[3]) - pad * 2;
    int nshow = a->ncfl > 3 ? 3 : a->ncfl;
    for (int k = 0; k < nshow; k++) {
        int idx = a->ncfl - nshow + k;
        float v = a->cfl[idx];
        char s[16];
        if (v >= 6000 || (a->kind == KD_OVF && !a->ship)) snprintf(s, sizeof s, "%d", (int)lroundf(v / 100));
        else snprintf(s, sizeof s, "A%d", (int)lroundf(v / 1000));
        if (a->ship) snprintf(s, sizeof s, "%d", (int)lroundf(v / 100) * 100);
        float sx = cx + cw * (nshow == 1 ? 0.15f : (float)k / nshow * 0.9f);
        float sy = py0 + ph * (k == nshow - 1 ? 0.68f : 0.42f) + cap_h(st_hand) * 0.5f;
        if (k < nshow - 1) {
            float tw = hand(sx, sy, -1, s, ca(INK, 0.75f), seed + (uint32_t)idx * 31, 0);
            line(sx - 2, sy - cap_h(st_hand) * 0.35f, sx + tw + 2, sy - cap_h(st_hand) * 0.55f, 1.4f, ca(INK, 0.8f));
        } else hand(sx, sy, -1, s, INK, seed + (uint32_t)idx * 31, 1);
    }
    /* speed and track */
    snprintf(b, sizeof b, "N%04d", (int)lroundf(a->gs));
    text(st_smb, px0 + pw * FX[4] + pad, top, -1, b, PRINT);
    snprintf(b, sizeof b, "%03d" S_DEG, (int)lroundf(a->trk) % 360);
    text(st_sm, px0 + pw * FX[4] + pad, bot, -1, b, ca(PRINT, 0.75f));
    /* where to: the airport printed, the runway written and ringed */
    float dx = px0 + pw * FX[5] + pad;
    if (a->kind == KD_ARR || a->kind == KD_DEP) {
        const Apt *A = a->apt >= 0 ? &apt[a->apt] : NULL;
        text(st_smb, dx, top, -1, A ? A->d->icao : "----", PRINT);
        if (a->kind == KD_ARR && A && a->rwy >= 0) {
            const char *rn = a->rend ? A->r[a->rwy].nb : A->r[a->rwy].na;
            float tw = hand(dx + ph * 0.1f, bot + ph * 0.04f, -1, rn, INK, seed + 7, 1);
            pen_circle(dx + ph * 0.1f + tw * 0.5f, bot - cap_h(st_hand) * 0.45f, tw * 0.75f + 3, cap_h(st_hand) * 0.95f, ca(INK, 0.75f), seed + 9);
        } else if (a->kind == KD_DEP) {
            snprintf(b, sizeof b, S_RARR "%03d", (int)lroundf(a->trk / 10) * 10 % 360);
            hand(dx + ph * 0.1f, bot + ph * 0.04f, -1, b, INK, seed + 7, 0);
        }
    } else if (a->ship) {
        text(st_smb, dx, top, -1, a->goodyear && a->ship_name ? "GOODYEAR" : "LOITER", PRINT);
        hand(dx + ph * 0.05f, bot + ph * 0.04f, -1, "VMC", INK, seed + 7, 0);
    } else {
        text(st_smb, dx, top, -1, a->cat == 0xA1 ? "VFR" : "TRANSIT", PRINT);
    }
    /* time first seen, and a tick once it's been on frequency a while */
    time_t ts = (time_t)a->first;
    struct tm tm;
    gmtime_r(&ts, &tm);
    snprintf(b, sizeof b, "%02d%02d", tm.tm_hour, tm.tm_min);
    float tx = px0 + pw * FX[6] + pad;
    text(st_smb, tx, top, -1, b, PRINT);
    if (a->emerg == 7700 || a->emerg == 7600) {
        float tw = hand(tx, bot + ph * 0.05f, -1, a->emerg == 7700 ? "EMG" : "RCF", INK_RED, seed + 3, 0);
        line(tx - 1, bot + ph * 0.12f, tx + tw + 2, bot + ph * 0.10f, 1.5f, INK_RED);
    }
    else if (wt - a->first > 20 || shot_mode) pen_tick(tx + ph * 0.35f, bot - ph * 0.12f, ph * 0.32f, INK);
    /* the airship strip: a little airship drawn on it */
    if (a->ship) {
        float sx = px0 + pw * FX[1] - pad - ph * 0.42f, sy = py0 + ph * 0.27f;
        draw_airship(sx, sy, -0.12f, ph * 0.03f, ca(INK, 0.85f), 1.2f);
    }
}

static void spring2(float *x, float *v, float tgt, float dt, float w) {
    for (float t = 0; t < dt; t += 1.0f / 120) {
        float h = fminf(1.0f / 120, dt - t);
        *v += (w * w * (tgt - *x) - 2 * w * *v) * h;
        *x += *v * h;
    }
}

/* a Dymo-style label: raised white letters on black tape */
static void dymo(float x, float y, const char *s, uint32_t seed) {
    float w = text_width(st_dymo, s, styles[st_dymo].px * 0.12f) + styles[st_dymo].px * 1.2f, h = styles[st_dymo].px * 1.55f;
    float ang = ((hash32(seed) & 255) / 255.0f - 0.5f) * 0.02f;
    rect_r(x + w * 0.5f + 1, y + h * 0.5f + 2, w * 0.5f, h * 0.5f, ang, 2, C4(0, 0, 0, 0.35f), M_FLAT, 0);
    rect_r(x + w * 0.5f, y + h * 0.5f, w * 0.5f, h * 0.5f, ang, 2, C4(0.07f, 0.075f, 0.08f, 1), M_TAPE, 0.3f);
    text_ex(st_dymo, x + w * 0.5f, y + h * 0.5f + cap_h(st_dymo) * 0.5f, 0, s, C4(0.92f, 0.93f, 0.92f, 0.95f), styles[st_dymo].px * 0.12f, ang, 0);
}

/* seven segments, lit or ghosted */
static void seg7(float x, float y, float h, const char *s, const float *c) {
    static const unsigned char D[10] = {0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f};
    float w = h * 0.54f, lw = h * 0.125f, sk = h * 0.07f;
    for (const char *p = s; *p; p++) {
        if (*p == ':') {
            for (int k = 0; k < 2; k++) {
                float dy = y + h * (k ? 0.72f : 0.30f);
                disc(x + w * 0.18f, dy, lw * 0.6f, c);
                glow(x + w * 0.18f, dy, lw * 3, C4(c[0], c[1], c[2], -0.25f));
            }
            x += w * 0.42f;
            continue;
        }
        unsigned m = isdigit((unsigned char)*p) ? D[*p - '0'] : 0;
        /* segment ends: a b c d e f g */
        float X0 = x + lw, X1 = x + w - lw, Y0 = y + lw * 0.6f, Y1 = y + h * 0.5f, Y2 = y + h - lw * 0.6f;
        float S[7][4] = {{X0, Y0, X1, Y0}, {X1, Y0, X1, Y1}, {X1, Y1, X1, Y2}, {X0, Y2, X1, Y2}, {X0, Y1, X0, Y2}, {X0, Y0, X0, Y1}, {X0, Y1, X1, Y1}};
        for (int k = 0; k < 7; k++) {
            float ax = S[k][0] + sk * (1 - (S[k][1] - y) / h), bx = S[k][2] + sk * (1 - (S[k][3] - y) / h);
            float dx = bx - ax, dy = S[k][3] - S[k][1], l = sqrtf(dx * dx + dy * dy);
            float sh = lw * 0.62f / fmaxf(l, 1);
            float x0 = ax + dx * sh, y0 = S[k][1] + dy * sh, x1 = bx - dx * sh, y1 = S[k][3] - dy * sh;
            if (m >> k & 1) {
                line(x0, y0, x1, y1, lw, c);
                glow((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, l * 0.9f, C4(c[0], c[1], c[2], -0.10f));
            } else line(x0, y0, x1, y1, lw, C4(c[0], c[1], c[2], 0.06f));
        }
        x += w + h * 0.14f;
    }
}

static float seg7_w(float h, const char *s) {
    float w = 0;
    for (const char *p = s; *p; p++) w += *p == ':' ? h * 0.52f * 0.42f : h * 0.52f + h * 0.14f;
    return w;
}

/* the clocks over the board */
static void draw_board_header(float x0, float y0, float x1, float y1, double wt) {
    float h = y1 - y0;
    rect(x0, y0, x1, y1, 6, C4(0.025f, 0.025f, 0.03f, 1), M_GLASS);
    time_t ts = (time_t)wt;
    struct tm lt, ut;
    localtime_r(&ts, &lt);
    gmtime_r(&ts, &ut);
    char b[32];
    float dh = h * 0.52f;
    snprintf(b, sizeof b, "%02d:%02d:%02d", ut.tm_hour, ut.tm_min, ut.tm_sec);
    float pad = h * 0.3f;
    const float *led = C4(1.0f, 0.30f, 0.16f, 1);
    float lx = x0 + pad;
    seg7(lx, y0 + (h - dh) * 0.5f, dh, b, led);
    float wU = seg7_w(dh, b);
    text_t(st_led_lab, lx + wU + h * 0.12f, y0 + (h - dh) * 0.5f + dh, -1, "UTC", ca(led, 0.8f), 1);
    snprintf(b, sizeof b, "%02d:%02d", lt.tm_hour, lt.tm_min);
    float dh2 = dh * 0.62f;
    float x2 = x1 - pad - seg7_w(dh2, b);
    const float *led2 = C4(1.0f, 0.62f, 0.18f, 1);
    seg7(x2, y0 + (h - dh) * 0.5f + (dh - dh2), dh2, b, led2);
    text_t(st_led_lab, x2 - h * 0.1f, y0 + (h - dh) * 0.5f + dh, 1, "LOCAL", ca(led2, 0.8f), 1);
    /* the sector and time away between, quietly */
    float mx = (lx + wU + h * 0.12f + text_width(st_led_lab, "UTC", 1) + x2 - h * 0.1f - text_width(st_led_lab, "LOCAL", 1)) * 0.5f;
    text_t(st_tape, mx, y0 + h * 0.42f, 0, sector, C4(0.80f, 0.82f, 0.84f, 0.85f), 2);
    char aw[32], s2[48];
    fmt_away(aw, sizeof aw, wt);
    snprintf(s2, sizeof s2, "AWAY %s", aw);
    text_t(st_led_lab, mx, y0 + h * 0.72f, 0, s2, ca(led2, 0.75f), 1);
}

/* lay out and draw the board in a rect: bays of strips that slide in and out */
static void draw_board(float x0, float y0, float x1, float y1, double wt, float dt, int header) {
    rect(x0, y0, x1, y1, 0, C4(0.17f, 0.19f, 0.20f, 1), M_BOARD);
    float m = (x1 - x0) * 0.025f;
    float hy = y0 + m * 0.6f;
    if (header) {
        float hh = fminf((y1 - y0) * 0.075f, (x1 - x0) * 0.13f);
        draw_board_header(x0 + m, hy, x1 - m, hy + hh, wt);
        hy += hh + m * 0.8f;
    }
    /* which column each bay is in, and its share of the height */
    float colw = (x1 - x0 - m * (strip_cols + 1)) / strip_cols;
    float lab_h = styles[st_dymo].px * 2.3f;
    float gap = strip_h * 0.13f;
    int col_of[NBAY], cnt[NBAY] = {0}, want[NBAY] = {0};
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->used || a->gone || a->fade < 0.2f || a->kind == KD_NONE) continue;
        if (a->ground && a->kind != KD_DEP) continue;
        want[ac_bay(a)]++;
    }
    float bay_y0[NBAY], bay_y1[NBAY];
    if (strip_cols >= 2) {
        col_of[BAY_ARR] = 0; col_of[BAY_DEP] = 1; col_of[BAY_OVF] = 1;
        bay_y0[BAY_ARR] = hy; bay_y1[BAY_ARR] = y1 - m * 0.5f;
        /* departures and overflights share the right column, by need */
        float avail = y1 - m * 0.5f - hy - lab_h * 2 - m * 0.5f;
        float sl = strip_h + gap;
        int tot = (int)(avail / sl);
        int nd = want[BAY_DEP] + 1, no = want[BAY_OVF] + 1;
        int kd = (int)lroundf((float)tot * nd / (nd + no));
        kd = kd < 2 ? 2 : kd > tot - 2 ? tot - 2 : kd;
        bay_y0[BAY_DEP] = hy; bay_y1[BAY_DEP] = hy + lab_h + kd * sl;
        bay_y0[BAY_OVF] = bay_y1[BAY_DEP] + m * 0.5f; bay_y1[BAY_OVF] = y1 - m * 0.5f;
    } else {
        float avail = y1 - m * 0.5f - hy - lab_h * 3 - m;
        float sl = strip_h + gap;
        int tot = (int)(avail / sl);
        int share[NBAY], s = 0;
        for (int b = 0; b < NBAY; b++) { share[b] = 1; s++; }
        while (s < tot) {
            /* hand out the rest to whoever's short of room */
            int bb = 0;
            float worst = -1;
            for (int b = 0; b < NBAY; b++) { float need = (float)(want[b] + 0.5f) / share[b]; if (need > worst) { worst = need; bb = b; } }
            share[bb]++; s++;
        }
        float y = hy;
        for (int b = 0; b < NBAY; b++) {
            col_of[b] = 0;
            bay_y0[b] = y; bay_y1[b] = y + lab_h + share[b] * sl;
            y = bay_y1[b] + m * 0.5f;
        }
    }
    /* which aircraft get a strip: arrivals nearest the runway first, the
     * rest newest first; the airship always, at the top of its bay */
    int list[NBAY][MAXAC], nl[NBAY] = {0};
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        int on = a->used && !a->gone && a->fade >= 0.2f && a->kind != KD_NONE && !(a->ground && a->kind != KD_DEP);
        if (on) { int b = ac_bay(a); list[b][nl[b]++] = i; }
    }
    for (int b = 0; b < NBAY; b++) {
        for (int p = 0; p < nl[b]; p++)
            for (int q = p + 1; q < nl[b]; q++) {
                Ac *A = &ac[list[b][p]], *B = &ac[list[b][q]];
                float ka, kb;
                if (b == BAY_ARR) {
                    ka = A->dist_thr >= 0 ? A->dist_thr : 30 + hypotf((float)(A->x - (A->apt >= 0 ? apt[A->apt].x : 0)), (float)(A->y - (A->apt >= 0 ? apt[A->apt].y : 0)));
                    kb = B->dist_thr >= 0 ? B->dist_thr : 30 + hypotf((float)(B->x - (B->apt >= 0 ? apt[B->apt].x : 0)), (float)(B->y - (B->apt >= 0 ? apt[B->apt].y : 0)));
                } else { ka = -(float)A->first; kb = -(float)B->first; }
                if (A->ship) ka = -1e12f;
                if (B->ship) kb = -1e12f;
                if (A->emerg && A->emerg != 7500) ka -= 1e6f;
                if (B->emerg && B->emerg != 7500) kb -= 1e6f;
                if (kb < ka) { int t = list[b][p]; list[b][p] = list[b][q]; list[b][q] = t; }
            }
    }
    for (int i = 0; i < MAXAC; i++) ac[i].bay = ac[i].strip ? ac[i].bay : -1;
    float colx[2];
    for (int c = 0; c < strip_cols && c < 2; c++) colx[c] = x0 + m + c * (colw + m);
    int keep[MAXAC] = {0};
    for (int b = 0; b < NBAY; b++) {
        float bx = colx[col_of[b]], by0 = bay_y0[b], by1 = bay_y1[b];
        /* the bay: a recessed channel with its label */
        rect(bx - m * 0.3f, by0 + lab_h * 0.55f, bx + colw + m * 0.3f, by1, 4, C4(0.10f, 0.11f, 0.12f, 1), M_BOARD);
        shadow(bx - m * 0.3f, by0 + lab_h * 0.55f, bx + colw + m * 0.3f, by0 + lab_h * 0.55f + 6, 0, 5, C4(0, 0, 0, 0.5f));
        char lb[48];
        const char *ap = NULL;
        for (int k = 0; k < napt; k++) if (apt[k].dist < 30 && apt[k].d->rank == 1) { ap = apt[k].d->icao; break; }
        (void)ap;
        snprintf(lb, sizeof lb, "%s", BAY_NAME[b]);
        dymo(bx + m * 0.2f, by0, lb, (uint32_t)b * 77u + 5);
        int cap = (int)((by1 - by0 - lab_h) / (strip_h + gap));
        if (nl[b] > cap) {
            char more[16];
            snprintf(more, sizeof more, "+%d", nl[b] - cap + 1);
            text(st_smb, bx + colw, by0 + lab_h * 0.5f, 1, more, C4(0.75f, 0.77f, 0.8f, 0.8f));
            cap -= 1;
        }
        for (int k = 0; k < nl[b] && k < cap; k++) {
            Ac *a = &ac[list[b][k]];
            float tx = bx, ty = by0 + lab_h + k * (strip_h + gap);
            if (!a->strip || shot_mode) {
                /* slides in from the right */
                a->strip = 1;
                a->sx = shot_mode ? tx : x1 + 40; a->sxv = 0;
                a->sy = ty; a->syv = 0;
                a->salpha = 1;
            }
            a->bay = b;
            spring2(&a->sx, &a->sxv, tx, dt, 7);
            spring2(&a->sy, &a->syv, ty, dt, 8);
            keep[list[b][k]] = 1;
        }
    }
    /* strips no longer wanted slide out to the left and go */
    for (int i = 0; i < MAXAC; i++) {
        Ac *a = &ac[i];
        if (!a->strip || keep[i]) continue;
        spring2(&a->sx, &a->sxv, x0 - strip_w - 60, dt, 5);
        a->salpha -= dt / 1.4f;
        if (a->salpha <= 0 || shot_mode) { a->strip = 0; a->salpha = 0; }
    }
    /* draw, the moving ones last so they pass over */
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < MAXAC; i++) {
            Ac *a = &ac[i];
            if (!a->strip || a->bay < 0) continue;
            int moving = fabsf(a->sxv) + fabsf(a->syv) > 20 || !keep[i];
            if (moving != pass) continue;
            alpha_mul = clampf(a->salpha, 0, 1);
            draw_strip(a, a->sx, a->sy, colw, strip_h, wt);
            alpha_mul = 1;
        }
}

/* ---- the arrivals/departures board beside the approach ---- */
static void draw_lists(float x0, float y0, float x1, float y1, int ai, double wt) {
    rect(x0, y0, x1, y1, 0, C4(0.045f, 0.058f, 0.072f, 1), M_PANEL);
    line(x0, y0, x0, y1, 1, C4(0.3f, 0.37f, 0.45f, 0.5f));
    float pad = (x1 - x0) * 0.05f, y = y0 + pad;
    float rh = lbl_lh * 1.45f;
    const float *lab = C4(0.50f, 0.57f, 0.65f, 1);
    char b[64];
    for (int pass = 0; pass < 2; pass++) {
        int kind = pass ? KD_DEP : KD_ARR;
        snprintf(b, sizeof b, "%s", pass ? "DEPARTURES" : "ARRIVALS");
        text_t(st_barb, x0 + pad, y + cap_h(st_barb), -1, b, COL_TXT, 2);
        if (ai >= 0) text_t(st_bar, x1 - pad, y + cap_h(st_barb), 1, apt[ai].d->name, lab, 1);
        y += rh * 1.1f;
        float cxs[6] = {0, 0.08f, 0.40f, 0.58f, 0.74f, 1.0f};
        static const char *HA[] = {"#", "CALLSIGN", "TYPE", "RWY", "NM", "LEVEL"};
        static const char *HD[] = {"", "CALLSIGN", "TYPE", "FROM", "TRK", "LEVEL"};
        float cw = x1 - x0 - pad * 2;
        for (int c = 0; c < 6; c++)
            text_t(st_tiny, x0 + pad + cw * cxs[c], y, c == 5 ? 1 : -1, pass ? HD[c] : HA[c], lab, 1);
        y += lbl_lh * 0.5f;
        line(x0 + pad, y, x1 - pad, y, 1, C4(0.3f, 0.37f, 0.45f, 0.5f));
        y += lbl_lh * 0.35f;
        int idx[MAXAC], n = 0;
        for (int i = 0; i < MAXAC; i++) {
            Ac *a = &ac[i];
            if (!a->used || a->gone || a->kind != kind || a->fade <= 0) continue;
            if (pass == 0 && ai >= 0 && a->apt != ai) continue;
            if (a->ground && kind == KD_ARR) continue;
            idx[n++] = i;
        }
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                Ac *A = &ac[idx[p]], *B = &ac[idx[q]];
                float ka = pass ? -(float)A->first : (A->dist_thr >= 0 ? A->dist_thr : 50 + hypotf((float)A->x - apt[ai < 0 ? 0 : ai].x, (float)A->y - apt[ai < 0 ? 0 : ai].y));
                float kb = pass ? -(float)B->first : (B->dist_thr >= 0 ? B->dist_thr : 50 + hypotf((float)B->x - apt[ai < 0 ? 0 : ai].x, (float)B->y - apt[ai < 0 ? 0 : ai].y));
                if (kb < ka) { int t = idx[p]; idx[p] = idx[q]; idx[q] = t; }
            }
        int maxrows = (int)(((y1 - y0) * (pass ? 0.92f : 0.5f) - (y - y0)) / rh);
        if (pass) maxrows = (int)((y1 - pad - lbl_lh * 5 - y) / rh);
        for (int k = 0; k < n && k < maxrows; k++) {
            Ac *a = &ac[idx[k]];
            float ry = y + rh * 0.5f + cap_h(st_lbl) * 0.5f;
            alpha_mul = a->fade;
            if (k % 2 == 0) rect(x0 + pad * 0.5f, y, x1 - pad * 0.5f, y + rh, 2, C4(1, 1, 1, 0.025f), M_FLAT);
            int vert;
            const float *c = ac_col(a, &vert);
            if (!pass) { snprintf(b, sizeof b, "%d", k + 1); text(st_lbl, x0 + pad + cw * cxs[0], ry, -1, b, COL_DIM); }
            text(st_lblb, x0 + pad + cw * cxs[1], ry, -1, a->cs, c);
            text(st_lbl, x0 + pad + cw * cxs[2], ry, -1, a->type, COL_DIM);
            if (!pass) {
                const char *rn = a->apt >= 0 && a->rwy >= 0 ? (a->rend ? apt[a->apt].r[a->rwy].nb : apt[a->apt].r[a->rwy].na) : "--";
                text(st_lbl, x0 + pad + cw * cxs[3], ry, -1, rn, a->rwy >= 0 ? COL_TXT : COL_DIM);
                float d = a->dist_thr >= 0 ? a->dist_thr : (a->apt >= 0 ? hypotf((float)a->x - apt[a->apt].x, (float)a->y - apt[a->apt].y) : 0);
                snprintf(b, sizeof b, "%.1f", d);
                text(st_lbl, x0 + pad + cw * cxs[4], ry, -1, b, COL_TXT);
            } else {
                text(st_lbl, x0 + pad + cw * cxs[3], ry, -1, a->apt >= 0 ? apt[a->apt].d->icao : "----", COL_DIM);
                snprintf(b, sizeof b, "%03d", (int)lroundf(a->trk) % 360);
                text(st_lbl, x0 + pad + cw * cxs[4], ry, -1, b, COL_TXT);
            }
            snprintf(b, sizeof b, "%03d%s", (int)lroundf(a->alt / 100), a->vr > 400 ? S_UP : a->vr < -400 ? S_DN : " ");
            text(st_lbl, x0 + pad + cw * cxs[5], ry, 1, b, vert ? COL_AMB : COL_TXT);
            alpha_mul = 1;
            y += rh;
        }
        if (n == 0) { text(st_lbl, x0 + pad, y + rh * 0.6f, -1, "none", COL_DIM); y += rh; }
        y += rh * 0.9f;
    }
    /* the airship callout */
    Ac *ship = NULL;
    for (int i = 0; i < MAXAC; i++) if (ac_visible(&ac[i]) && ac[i].ship && (!ship || ac[i].goodyear)) ship = &ac[i];
    float by = y1 - pad;
    if (ship) {
        float h = lbl_lh * 4.6f;
        rect(x0 + pad, by - h, x1 - pad, by, 5, C4(0.11f, 0.09f, 0.04f, 0.95f), M_FLAT);
        frame(x0 + pad, by - h, x1 - pad, by, 1.2f, ca(COL_GOLD, 0.6f));
        draw_airship(x0 + pad + h * 0.55f, by - h * 0.5f, -0.15f, lbl_lh / 9, COL_GOLD, 1.6f);
        float tx = x0 + pad + h * 1.15f;
        snprintf(b, sizeof b, "%s", ship->goodyear ? ship->ship_name : "AIRSHIP ABOUT");
        text_t(st_barb, tx, by - h + lbl_lh * 1.25f, -1, b, COL_GOLD, 1);
        double d = hypot(ship->x, ship->y);
        float brg = wrap360(atan2f((float)ship->x, (float)ship->y) / DEG);
        snprintf(b, sizeof b, "%s  %s  %d FT  %d KT", ship->reg, ship->cs, (int)lroundf(ship->alt / 10) * 10, (int)lroundf(ship->gs));
        text(st_lbl, tx, by - h + lbl_lh * 2.55f, -1, b, COL_TXT);
        snprintf(b, sizeof b, "%.1f NM %s OF %s", d, compass8(brg), sector);
        text(st_lbl, tx, by - h + lbl_lh * 3.75f, -1, b, COL_DIM);
    } else {
        int ov = 0;
        for (int i = 0; i < MAXAC; i++) if (ac_visible(&ac[i]) && ac[i].kind == KD_OVF) ov++;
        snprintf(b, sizeof b, "%d OVERFLIGHTS " S_DOT " %d IN SECTOR", ov, count_traffic());
        text_t(st_bar, x0 + pad, by, -1, b, lab, 1);
        if (far_ship.on) draw_far_ship(x0 + pad, by - lbl_lh * 1.6f, 0);
    }
    (void)wt;
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
#define NOISE GLSL( \
    float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); } \
    float vnoise(vec2 p) { \
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); \
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); \
    } \
    float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; } \
)

static const char *VS_PRIM = "#version 330 core\n" GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aL;
    layout(location = 2) in vec4 aP;
    layout(location = 3) in vec4 aC;
    layout(location = 4) in float aK;
    uniform vec2 uRes;
    out vec2 vL;
    out vec4 vP;
    out vec4 vC;
    flat out int vK;
    void main() {
        vL = aL; vP = aP; vC = aC; vK = int(aK + 0.5);
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });

static const char *FS_PRIM = "#version 330 core\n" NOISE GLSL(
    uniform sampler2D uAtlas;
    in vec2 vL;
    in vec4 vP;
    in vec4 vC;
    flat in int vK;
    out vec4 o;
    float sdb(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
    void main() {
        vec3 c = vC.rgb;
        float a = 1.0;
        if (vK == 0) {
            float hw = vP.y, t = max(hw, 0.5);
            float d = length(vec2(max(abs(vL.x) - vP.x, 0.0), vL.y)) - t;
            a = clamp(0.5 - d, 0.0, 1.0) * min(hw * 2.0, 1.0);
            if (vP.z > 0.0) {
                float u = mod(vL.x + vP.x, vP.z), on = vP.z * vP.w;
                a *= clamp(0.5 - max(-u, u - on), 0.0, 1.0);
            }
        } else if (vK == 1) {
            float r = length(vL);
            if (vP.y > 0.0) a = clamp(0.5 - (abs(r - vP.x) - max(vP.y, 0.5)), 0.0, 1.0) * min(vP.y * 2.0, 1.0);
            else a = clamp(0.5 - (r - vP.x), 0.0, 1.0);
        } else if (vK == 2) {
            a = texture(uAtlas, vL / vec2(textureSize(uAtlas, 0))).r;
        } else if (vK == 3) {
            float d = sdb(vL, vP.xy, vP.z);
            a = clamp(0.5 - d, 0.0, 1.0);
            int m = int(vP.w);
            float sd = fract(vP.w) * 977.0;
            if (m == 1) {                       /* paper: fibres, a little tooth, darker at the edge */
                vec2 p = vL + sd;
                c *= 0.955 + 0.06 * fbm(p * 0.06) + 0.025 * vnoise(p * vec2(1.3, 0.11)) + 0.015 * (hash(floor(p * 1.5)) - 0.5);
                c *= 1.0 - 0.07 * smoothstep(-5.0, 0.0, d);
            } else if (m == 2) {                /* moulded plastic: lit from above, a bright top lip */
                float t = vL.y / vP.y;
                c *= 1.05 - 0.16 * t;
                c += vec3(0.20) * exp(-pow((vL.y + vP.y - 1.6) / 1.1, 2.0));
                c *= 1.0 - 0.22 * smoothstep(-2.5, 0.0, d);
            } else if (m == 3) {                /* powder-coated steel */
                vec2 p = gl_FragCoord.xy;
                c *= 0.95 + 0.06 * vnoise(p * 0.8) + 0.05 * fbm(p * 0.006);
            } else if (m == 4) {                /* smoked glass over the LEDs */
                float t = vL.y / vP.y * 0.5 + 0.5;
                c += vec3(0.035) * (1.0 - t) + vec3(0.02) * smoothstep(0.0, 1.0, 1.0 - abs(vL.x / vP.x));
                c += vec3(0.06) * exp(-pow((vL.y + vP.y - 2.0) / 1.5, 2.0));
            } else if (m == 5) {                /* embossing tape */
                c *= 0.88 + 0.12 * (0.5 + 0.5 * sin(vL.y * 2.2));
            } else if (m == 6) {
                c *= 0.97 + 0.03 * vnoise(gl_FragCoord.xy * 0.6);
            }
        } else if (vK == 4) {
            float d = sdb(vL, vP.xy, vP.z);
            a = 1.0 - smoothstep(-vP.w, vP.w * 1.6, d);
        } else if (vK == 5) {
            vec2 p = vL / vP.xy;
            float k = length(p), g = length(p / vP.xy);
            float d = (k - 1.0) * k / max(g, 1e-4);
            if (vP.z > 0.0) a = clamp(0.5 - (abs(d) - max(vP.z, 0.5)), 0.0, 1.0);
            else a = clamp(0.5 - d, 0.0, 1.0);
        } else {
            float r = length(vL) / vP.x;
            a = exp(-r * r * 4.0) * (1.0 - smoothstep(0.85, 1.0, r));
        }
        float al = abs(vC.a) * a;
        o = vec4(c * al, vC.a < 0.0 ? 0.0 : al);
    });

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the scope's face: a deep blue-grey, the range rings and bearing scale
 * about the receiver, and a faint sweep */
static const char *FS_BG = "#version 330 core\n" NOISE GLSL(
    uniform vec2 uRes;
    uniform vec4 uRect;
    uniform vec2 uC;
    uniform float uPPN, uStep, uStepA, uOuter, uSweep, uSweepA;
    out vec4 o;
    float ringline(float rn, float s) {
        float d = abs(rn - s * floor(rn / s + 0.5)) * uPPN;
        return clamp(1.0 - d, 0.0, 1.0);
    }
    void main() {
        vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec2 rc = (uRect.xy + uRect.zw) * 0.5, rs = (uRect.zw - uRect.xy) * 0.5;
        float rr = length((p - rc) / max(rs.x, rs.y));
        vec3 col = mix(vec3(0.050, 0.068, 0.088), vec3(0.020, 0.027, 0.036), smoothstep(0.05, 1.25, rr));
        vec2 d = p - uC;
        float r = length(d), rn = r / uPPN;
        float far = 1.0 - 0.55 * smoothstep(0.6, 1.4, rr);
        float a = 0.0;
        if (rn > 0.2) {
            a += ringline(rn, uStep) * 0.17;
            a += ringline(rn, uStep * 0.5) * 0.10 * uStepA * (1.0 - ringline(rn, uStep));
        }
        vec3 rcol = vec3(0.42, 0.50, 0.60);
        float ang = degrees(atan(d.x, -d.y));
        if (ang < 0.0) ang += 360.0;
        if (uOuter > 0.0) {
            float R = uOuter * uPPN;
            a += clamp(1.4 - abs(r - R), 0.0, 1.0) * 0.30;
            /* ticks inside the ring every 5 degrees, longer at 10 and 30 */
            float t5 = abs(ang - 5.0 * floor(ang / 5.0 + 0.5)) * 0.0174533 * r;
            float t10 = abs(ang - 10.0 * floor(ang / 10.0 + 0.5)) * 0.0174533 * r;
            float t30 = abs(ang - 30.0 * floor(ang / 30.0 + 0.5)) * 0.0174533 * r;
            float in5 = step(R - 6.0, r) * step(r, R), in10 = step(R - 11.0, r) * step(r, R), in30 = step(R - 18.0, r) * step(r, R);
            a += clamp(0.9 - t5, 0.0, 1.0) * in5 * 0.28 + clamp(0.9 - t10, 0.0, 1.0) * in10 * 0.12 + clamp(1.0 - t30, 0.0, 1.0) * in30 * 0.2;
            /* faint radials every 30 degrees, dashed */
            float dash = step(0.5, fract(rn / (uStep * 0.125)));
            a += clamp(0.8 - t30, 0.0, 1.0) * 0.045 * dash * step(uStep * 0.25, rn) * step(r, R);
        }
        col += rcol * a * far;
        /* the sweep: a bright edge, a long dim wake */
        float behind = mod(uSweep - ang + 360.0, 360.0);
        float sw = (exp(-behind / 40.0) * 0.03 + clamp(1.0 - behind * 0.0174533 * r / 1.2, 0.0, 1.0) * 0.045) * uSweepA;
        sw *= smoothstep(0.0, 30.0, r) * (uOuter > 0.0 ? 1.0 - smoothstep(uOuter * uPPN, uOuter * uPPN * 1.05, r) : 1.0);
        col += vec3(0.30, 0.78, 0.72) * sw;
        col += (hash(p + fract(uSweep)) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });

static GLuint prog_prim, prog_bg, vao_prim, vbo_prim, vao_full, tex_atlas;

static GLuint compile(GLenum type, const char *srcs) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &srcs, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "atc: shader:\n%s\n", log);
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
    int off[5] = {0, 2, 4, 8, 12}, sz[5] = {2, 2, 4, 4, 1};
    for (int i = 0; i < 5; i++) {
        glVertexAttribPointer((GLuint)i, sz[i], GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(off[i] * sizeof(float)));
        glEnableVertexAttribArray((GLuint)i);
    }
    glGenTextures(1, &tex_atlas);
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
static void flush(const float *clip) {
    if (!vbn) return;
    if (clip) {
        glEnable(GL_SCISSOR_TEST);
        glScissor((GLint)clip[0], (GLint)(H - clip[3]), (GLsizei)(clip[2] - clip[0]), (GLsizei)(clip[3] - clip[1]));
    }
    glUseProgram(prog_prim);
    glUniform2f(U(prog_prim, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_prim, "uAtlas"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_prim);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vbn * sizeof(float)), vb, GL_STREAM_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, vbn / VF);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    vbn = 0;
}

/* the scope face for a view */
static void draw_bg(const View *v, float step, float step_a, double wt) {
    float hx, hy;
    vp(v, v->hx, v->hy, &hx, &hy);
    glEnable(GL_SCISSOR_TEST);
    glScissor((GLint)v->x0, (GLint)(H - v->y1), (GLsizei)(v->x1 - v->x0), (GLsizei)(v->y1 - v->y0));
    glUseProgram(prog_bg);
    glUniform2f(U(prog_bg, "uRes"), (float)W, (float)H);
    glUniform4f(U(prog_bg, "uRect"), v->x0, v->y0, v->x1, v->y1);
    glUniform2f(U(prog_bg, "uC"), hx, hy);
    glUniform1f(U(prog_bg, "uPPN"), v->ppn);
    glUniform1f(U(prog_bg, "uStep"), step);
    glUniform1f(U(prog_bg, "uStepA"), step_a);
    glUniform1f(U(prog_bg, "uOuter"), v->outer);
    glUniform1f(U(prog_bg, "uSweep"), (float)fmod(wt * 360.0 / 12.0, 360.0));
    glUniform1f(U(prog_bg, "uSweepA"), opt_no_sweep ? 0.0f : v->approach ? 0.5f : 1.0f);
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
}

/* ---- layout: the roles, and the type sizes for this size ---- */
static int role = R_SCOPE;
static int lay_w, lay_h;

static int pick_role(void) {
    if (opt_role >= 0) return opt_role;
    int portrait = H > W * 1.1f;
    if (span_me < 0) return portrait ? R_COMBO : R_SCOPE;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    int front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    if (span_me == front) return portrait ? R_COMBO : R_SCOPE;
    if (span[span_me][3] > span[span_me][2] * 1.1f) return R_STRIPS;
    /* the other landscapes, left to right: the approach, then a wide scope */
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == front || i == span_me || span[i][3] > span[i][2] * 1.1f) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && i < span_me)) rank++;
    }
    return rank % 2 ? R_WIDE : R_APPROACH;
}

static void board_geom(float bw) {
    strip_cols = bw >= 900 ? 2 : 1;
    float m = bw * 0.025f;
    float colw = (bw - m * (strip_cols + 1)) / strip_cols;
    strip_w = colw;
    strip_h = clampf(colw / 7.4f, 34, 110);
}

static void layout(void) {
    role = pick_role();
    nstyles = 0;
    float base = fminf((float)W, (float)H);
    float fs = clampf(base / 74.0f, 11, 22);
    if (role == R_COMBO) fs = clampf(W / 80.0f, 11, 20);
    lbl_lh = roundf(fs * 1.12f);
    st_lbl = style(F_MONO, fs);
    st_lblb = style(F_MONOB, fs);
    st_lbls = style(F_MONO, fs * 0.85f);
    st_tiny = style(F_SANS, fs * 0.80f);
    st_tinyb = style(F_SANSB, fs * 0.82f);
    bar_h = roundf(fs * 2.3f);
    st_bar = style(F_SANS, fs * 0.82f);
    st_barb = style(F_MONOB, fs * 0.95f);
    st_big = style(F_SANSB, fs * 1.6f);
    float bw = (float)W;
    board_geom(bw);
    st_cs = style(F_MONOB, strip_h * 0.27f);
    st_sm = style(F_MONO, strip_h * 0.165f);
    st_smb = style(F_MONOB, strip_h * 0.17f);
    st_hand = style(F_HAND, strip_h * 0.25f);
    st_handb = style(F_HAND, strip_h * 0.30f);
    st_dymo = style(F_MONOB, fmaxf(strip_h * 0.19f, fs * 0.95f));
    float hh = fminf(H * 0.075f, W * 0.13f);
    st_tape = style(F_SANSB, hh * 0.20f);
    st_led_lab = style(F_SANSB, hh * 0.13f);
    build_atlas();
    lay_w = W; lay_h = H;
}

/* ---- the frame ---- */
static double last_wt;
static float sel_alpha;

static void scope_view(View *v, float x0, float y0, float x1, float y1, double wt, int follow_on, float R0) {
    v->x0 = x0; v->y0 = y0; v->x1 = x1; v->y1 = y1;
    v->hx = v->hy = 0;
    v->outer = R0;
    v->approach = 0;
    float ppn0 = 0.46f * fminf(x1 - x0, y1 - y0) / R0;
    double tx = 0, ty = 0;
    float tr = R0;
    float f = follow_on ? follow(wt, R0, &tx, &ty, &tr) : 0;
    float rng = expf(mixf(logf(R0), logf(tr), f));
    v->ppn = ppn0 * R0 / rng;
    /* zoom about the target: its place on screen slides to the middle */
    v->cx = tx - (1 - f) * tx * ppn0 / v->ppn;
    v->cy = ty - (1 - f) * ty * ppn0 / v->ppn;
    sel_alpha = f;
}

static void ring_steps(const View *v, float *step, float *step_a) {
    float rng = 0.46f * fminf(v->x1 - v->x0, v->y1 - v->y0) / v->ppn;
    float lvl = log2f(rng / 4 / 5);
    float fl = floorf(lvl);
    *step = 5 * exp2f(fl) * 2;
    *step_a = 1 - (lvl - fl);
    *step_a = clampf(*step_a * 1.5f, 0, 1);
}

static void scope_pass(View *v, double wt, float dt) {
    float step, step_a;
    ring_steps(v, &step, &step_a);
    draw_bg(v, step, step_a, wt);
    float clip[4] = {v->x0, v->y0, v->x1, v->y1};
    draw_map(v, 0);
    draw_ring_labels(v, step, step_a);
    flush(clip);
    arrange_labels(v, wt, shot_mode);
    ease_labels(dt);
    draw_traffic(v, wt, sel_alpha);
    flush(clip);
}

static int main_apt(void) {
    int best = -1;
    for (int i = 0; i < napt; i++) {
        if (apt[i].dist > 60) break;
        if (best < 0 || apt[i].d->rank < apt[best].d->rank) best = i;
    }
    if (best < 0 && napt) best = 0;
    return best;
}

static GLuint target_fbo;
static void render(double wt, float dt) {
    if (W != lay_w || H != lay_h) layout();
    if (atlas_dirty) upload_atlas();
    glBindFramebuffer(GL_FRAMEBUFFER, target_fbo);
    glViewport(0, 0, W, H);
    glClearColor(0.02f, 0.027f, 0.035f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    vbn = 0;
    char b1[48], b2[48];
    if (role == R_SCOPE || role == R_WIDE || role == R_COMBO) {
        View v;
        float sy1 = role == R_COMBO ? fminf((float)H * 0.62f, (float)W + bar_h) : (float)H;
        float R0 = role == R_WIDE ? fmaxf(opt_range * 2, opt_radius) : opt_range;
        scope_view(&v, 0, bar_h, (float)W, sy1, wt, role != R_WIDE, R0);
        scope_pass(&v, wt, dt);
        Ac *sel = fol_id ? ac_find(fol_id, 0) : NULL;
        snprintf(b1, sizeof b1, "RNG %.0f " S_DOT " TFC %d", 0.46f * fminf(v.x1 - v.x0, v.y1 - v.y0) / v.ppn, count_traffic());
        if (sel && sel_alpha > 0.5f) snprintf(b2, sizeof b2, "TRK %s", sel->cs);
        draw_topbar(0, 0, (float)W, wt, b1, sel && sel_alpha > 0.5f ? b2 : NULL);
        float m = lbl_lh;
        draw_track_info(m, sy1 - m, sel, sel_alpha);
        float fy = sy1 - m;
        if (sel && sel_alpha > 0) fy -= lbl_lh * 8;
        draw_far_ship(m, fy, R0);
        if (role == R_WIDE) text_t(st_tinyb, W - m, sy1 - m, 1, "WIDE " S_DOT " ALL TRAFFIC", ca(COL_DIM, 0.8f), 2);
        flush(NULL);
        if (role == R_COMBO) {
            draw_board(0, sy1, (float)W, (float)H, wt, dt, 0);
            line(0, sy1, (float)W, sy1, 2, C4(0, 0, 0, 0.6f));
            flush(NULL);
        }
    } else if (role == R_APPROACH) {
        int ai = main_apt();
        float lx = roundf(W * 0.64f);
        View v = {0};
        v.x0 = 0; v.y0 = bar_h; v.x1 = lx; v.y1 = (float)H;
        v.approach = 1;
        v.outer = 0;
        if (ai >= 0) {
            const Apt *A = &apt[ai];
            v.hx = A->x; v.hy = A->y;
            /* look down the approach to the runway in use */
            float tx = A->x, ty = A->y, ux = 0, uy = 0;
            for (int r = 0; r < A->nr; r++)
                for (int e = 0; e < 2; e++)
                    if ((e ? A->r[r].act_b : A->r[r].act_a) && !ux && !uy) rwy_end(&A->r[r], e, &tx, &ty, &ux, &uy, NULL);
            v.cx = tx - ux * 4.5f; v.cy = ty - uy * 4.5f;
            v.ppn = 0.5f * fminf(v.x1 - v.x0, v.y1 - v.y0) / 9.5f;
        } else { v.cx = v.cy = 0; v.ppn = 0.5f * fminf(v.x1 - v.x0, v.y1 - v.y0) / 15; }
        sel_alpha = 0;
        scope_pass(&v, wt, dt);
        snprintf(b1, sizeof b1, "APP %s", ai >= 0 ? apt[ai].d->icao : "----");
        draw_topbar(0, 0, (float)W, wt, b1, NULL);
        draw_lists(lx, bar_h, (float)W, (float)H, ai, wt);
        flush(NULL);
    } else {
        draw_board(0, 0, (float)W, (float)H, wt, dt, 1);
        text_t(st_tiny, W - W * 0.025f, H - W * 0.012f, 1, src_credit(), C4(0.7f, 0.72f, 0.75f, 0.5f), 1);
        flush(NULL);
    }
}

/* ---- the world, each frame ---- */
static double replay_t0;
static void advance(double wt, float dt) {
    if (src == SRC_REPLAY) {
        if (!replay_t0 || (!shot_mode && wt - replay_t0 > 120)) {
            for (int i = 0; i < MAXAC; i++) ac[i].used = 0;
            replay_t0 = wt;
            if (load_feed_file(opt_replay, wt, 0) < 0) die("--replay: not an aircraft.json");
            /* a snapshot has no past: lay down its history dots along the
             * way it came */
            for (int i = 0; i < MAXAC; i++) {
                Ac *a = &ac[i];
                if (!a->used || !a->has_vel) continue;
                int n = a->ship ? 24 : 8;
                for (int k = n; k >= 1; k--) {
                    double t = k * HIST_DT;
                    a->hx[a->hn] = (float)(a->fx - sin(a->ftrk * DEG) * a->fgs / 3600 * t);
                    a->hy[a->hn] = (float)(a->fy - cos(a->ftrk * DEG) * a->fgs / 3600 * t);
                    a->hn++;
                }
                a->hlast = wt;
            }
        }
    } else {
        feed_tick(wt);
        if (data_live(wt)) {
            for (int i = 0; i < MAXAC; i++) if (ac[i].used && ac[i].sim) ac[i].gone = 1;
        } else {
            sim_tick(wt);
            sim_publish(wt);
        }
    }
    traffic_update(wt, dt);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "atc");
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
        "usage: atc [--window | --root | --layer] [--seed N] [--fps N]\n"
        "           [--lat LAT --lon LON] [--feed URL] [--sim | --offline] [--replay FILE.json]\n"
        "           [--range NM] [--radius NM] [--no-sweep] [--no-follow]\n"
        "           [--role scope|strips|approach|wide|combo]\n"
        "           [--font-mono FAMILY] [--font-sans FAMILY] [--font-hand FAMILY]\n"
        "           [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "           [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "Traffic: a local readsb/dump1090 (--feed URL, else 127.0.0.1:8080 or /tar1090),\n"
        "else adsb.lol around --lat/--lon or ~/.config/wallpapers/location (\"LAT LON\"),\n"
        "else simulated. --sim never uses the network; --shot and --bench never fetch.\n");
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
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-lat") && i + 1 < argc) { opt_lat = strtod(argv[++i], NULL); loc_given |= 1; }
        else if (!strcmp(a, "-lon") && i + 1 < argc) { opt_lon = strtod(argv[++i], NULL); loc_given |= 2; }
        else if (!strcmp(a, "-feed") && i + 1 < argc) opt_feed = argv[++i];
        else if (!strcmp(a, "-replay") && i + 1 < argc) opt_replay = argv[++i];
        else if (!strcmp(a, "-sim") || !strcmp(a, "-offline")) opt_sim = 1;
        else if (!strcmp(a, "-range") && i + 1 < argc) opt_range = clampf(strtof(argv[++i], NULL), 3, 250);
        else if (!strcmp(a, "-radius") && i + 1 < argc) opt_radius = clampf(strtof(argv[++i], NULL), 5, 250);
        else if (!strcmp(a, "-no-sweep")) opt_no_sweep = 1;
        else if (!strcmp(a, "-no-follow")) opt_no_follow = 1;
        else if (!strcmp(a, "-font-mono") && i + 1 < argc) opt_font_mono = argv[++i];
        else if (!strcmp(a, "-font-sans") && i + 1 < argc) opt_font_sans = argv[++i];
        else if (!strcmp(a, "-font-hand") && i + 1 < argc) opt_font_hand = argv[++i];
        else if (!strcmp(a, "-role") && i + 1 < argc) {
            const char *r = argv[++i];
            opt_role = !strcmp(r, "scope") ? R_SCOPE : !strcmp(r, "strips") ? R_STRIPS : !strcmp(r, "approach") ? R_APPROACH
                     : !strcmp(r, "wide") ? R_WIDE : !strcmp(r, "combo") ? R_COMBO : -2;
            if (opt_role < 0) usage();
        }
        else usage();
    }
    if (loc_given && loc_given != 3) usage();
    if (loc_given && (fabs(opt_lat) > 90 || fabs(opt_lon) > 180)) usage();
    if (span_me >= span_n) span_me = -1;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;
    shot_mode = opt_shot != NULL;

    load_location();
    if (have_loc) { lat0 = opt_lat; lon0 = opt_lon; }
    if (opt_replay && !have_loc) {
        /* centre on the snapshot's traffic */
        size_t len;
        char *buf = slurp(opt_replay, &len);
        if (!buf) die("--replay: can't read it");
        double sla = 0, slo = 0;
        int n = 0;
        Js j = {buf, buf + len};
        const char *p = buf;
        (void)j;
        while ((p = strstr(p, "\"lat\":"))) {
            double la = atof(p + 6);
            const char *q = strstr(p, "\"lon\":");
            if (!q || q - p > 200) { p += 6; continue; }
            sla += la; slo += atof(q + 6); n++;
            p = q + 6;
        }
        free(buf);
        if (n) { lat0 = sla / n; lon0 = slo / n; }
    }
    coslat0 = cos(lat0 * DEG);
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    for (int i = 0; host[i] && i < 63; i++) sector[i] = (char)toupper((unsigned char)host[i]);
    airports_init();
    feed_init();

    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;
    sim_init(opt_shot ? t_wall0 + opt_at : wall());
    fonts_init();

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
        XStoreName(dpy, win, "ATC");
        XClassHint chint = {"atc", "Atc"};
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
        if (opt_shot) {
            double wt = t_wall0 + opt_at;
            layout();
            advance(wt, 10);
            traffic_update(wt, 10);
            render(wt, 1.0f / 60);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        layout();
        double wt0 = t_wall0 + opt_at;
        advance(wt0, 10);
        double sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double wt = wt0 + i / 60.0;
            advance(wt, 1.0f / 60);
            render(wt, 1.0f / 60);
            glFinish();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("atc: %dx%d role %d, %d frames: avg %.3f ms, worst %.3f ms per frame, %d aircraft\n",
               W, H, role, opt_bench, sum / opt_bench, worst, count_traffic());
        printf("atc: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    last_wt = wall();
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
        float dt = (float)fmin(fmax(wt - last_wt, 0), 0.25);
        last_wt = wt;
        advance(wt, dt);
        render(wt, dt);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
