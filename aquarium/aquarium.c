/*
 * aquarium — a planted freshwater aquarium seen through the front glass.
 *
 * HEADER-PLACEHOLDER
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
#include <glob.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

#define PI 3.14159265f
#define TAU 6.2831853f

/* ---- options ---- */
static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "EB Garamond";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot: the moment this many seconds after the epoch */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the tank's clock starts at */
static float opt_hour = -1;            /* --hour: force the local time of day (lighting) */
static float opt_load = -1, opt_temp = -1;
static double opt_away = -1;
static const char *opt_focus;          /* --focus KIND: a close-up camera on one animal */
static int opt_msaa = 4;
static float opt_scale = 1;            /* internal resolution of the scene */
static int opt_overlay = 1;

/* ---- spanning: one instance per output ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "aquarium: %s\n", what); exit(1); }

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

/* ---- random numbers: a sequential generator for building the scene, and
 * stateless hashes for anything indexed by time ---- */
static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}
static float frange(float a, float b) { return a + (b - a) * frand(); }
static float fnorm(void) {                 /* roughly normal, mean 0, sd 1 */
    return (frand() + frand() + frand() + frand() - 2.0f) * 1.732f;
}
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) { return hash32(a * 0x9E3779B1u ^ hash32(b + 0x632BE5ABu) ^ hash32(c * 0x85EBCA77u + 0x27d4eb2fu)); }
static float hf(uint32_t a, uint32_t b, uint32_t c) { return (hash3(a, b, c) >> 8) * (1.0f / 16777216.0f); }

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

/* smooth 1D noise in time: a sum of hashed cosines through integer knots */
static float tnoise(uint32_t key, double t) {
    double fl = floor(t);
    float f = (float)(t - fl), u = f * f * f * (f * (f * 6 - 15) + 10);
    long i = (long)fl;
    float a = hf(key, (uint32_t)i, 7) * 2 - 1, b = hf(key, (uint32_t)(i + 1), 7) * 2 - 1;
    return a + (b - a) * u;
}
static float tnoise_d(uint32_t key, double t) {   /* its derivative */
    double fl = floor(t);
    float f = (float)(t - fl), du = 30 * f * f * (f * (f - 2) + 1);
    long i = (long)fl;
    float a = hf(key, (uint32_t)i, 7) * 2 - 1, b = hf(key, (uint32_t)(i + 1), 7) * 2 - 1;
    return (b - a) * du;
}

/* 2D value noise for building geometry */
static float vnoise2(float x, float y, uint32_t salt) {
    float fx = floorf(x), fy = floorf(y), ax = x - fx, ay = y - fy;
    int ix = (int)fx, iy = (int)fy;
    ax = ax * ax * (3 - 2 * ax); ay = ay * ay * (3 - 2 * ay);
    float a = hf((uint32_t)ix, (uint32_t)iy, salt), b = hf((uint32_t)ix + 1, (uint32_t)iy, salt);
    float c = hf((uint32_t)ix, (uint32_t)iy + 1, salt), d = hf((uint32_t)ix + 1, (uint32_t)iy + 1, salt);
    return mixf(mixf(a, b, ax), mixf(c, d, ax), ay);
}
static float vnoise3(float x, float y, float z, uint32_t salt) {
    float fz = floorf(z), az = z - fz;
    az = az * az * (3 - 2 * az);
    uint32_t iz = (uint32_t)(int)fz;
    return mixf(vnoise2(x, y, salt * 31 + iz * 977), vnoise2(x, y, salt * 31 + (iz + 1) * 977), az);
}
static float fbm3(float x, float y, float z, int oct, uint32_t salt) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) {
        s += a * (vnoise3(x, y, z, salt + (uint32_t)i * 13) * 2 - 1);
        n += a;
        x = x * 2.03f + 1.7f; y = y * 2.03f - 3.1f; z = z * 2.03f + 0.6f; a *= 0.5f;
    }
    return s / n;
}

/* ---- little vectors ---- */
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { V3 r = {x, y, z}; return r; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vmul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static V3 vmad(V3 a, V3 b, float s) { return v3(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a) { float l = vlen(a); return l > 1e-12f ? vmul(a, 1 / l) : v3(0, 1, 0); }
static V3 vlerp(V3 a, V3 b, float t) { return v3(mixf(a.x, b.x, t), mixf(a.y, b.y, t), mixf(a.z, b.z, t)); }
/* rotate v about the unit axis k by angle a */
static V3 vrot(V3 v, V3 k, float a) {
    float c = cosf(a), s = sinf(a);
    return vadd(vadd(vmul(v, c), vmul(vcross(k, v), s)), vmul(k, vdot(k, v) * (1 - c)));
}
/* any unit vector perpendicular to n */
static V3 vperp(V3 n) {
    V3 a = fabsf(n.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0);
    return vnorm(vcross(n, a));
}
/* Catmull-Rom through four points */
static V3 vcatmull(V3 p0, V3 p1, V3 p2, V3 p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    V3 r = vmul(p1, 2);
    r = vmad(r, vsub(p2, p0), t);
    r = vmad(r, vadd(vsub(vmul(p0, 2), vmul(p1, 5)), vsub(vmul(p2, 4), p3)), t2);
    r = vmad(r, vadd(vsub(vmul(p1, 3), p0), vsub(p3, vmul(p2, 3))), t3);
    return vmul(r, 0.5f);
}

/* ---- sensors ---- */
static char host[64];
static time_t locked_at;
static char temp_path[300];

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

static long read_long(const char *path, long dflt) {
    char b[64];
    slurp_line(path, b, sizeof b);
    char *e;
    long v = strtol(b, &e, 10);
    return e == b ? dflt : v;
}

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

/* the CPU package temperature: the best-looking sensor on offer */
static void find_temp(void) {
    int best = 0;
    glob_t g;
    if (glob("/sys/class/hwmon/hwmon*/temp*_input", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            const char *in = g.gl_pathv[i];
            char dir[256], name[64], lpath[300], label[64];
            snprintf(dir, sizeof dir, "%.*s", (int)(strrchr(in, '/') - in), in);
            snprintf(lpath, sizeof lpath, "%s/name", dir);
            slurp_line(lpath, name, sizeof name);
            snprintf(lpath, sizeof lpath, "%.*s_label", (int)(strlen(in) - 6), in);
            slurp_line(lpath, label, sizeof label);
            int s = 0;
            if (!strcmp(name, "coretemp") && !strncmp(label, "Package", 7)) s = 100;
            else if (!strcmp(name, "k10temp") && !strcmp(label, "Tdie")) s = 96;
            else if (!strcmp(name, "k10temp") && !strcmp(label, "Tctl")) s = 95;
            else if (!strcmp(name, "zenpower") && !strcmp(label, "Tdie")) s = 94;
            else if (!strcmp(name, "cpu_thermal") || !strcmp(name, "cpu-thermal")) s = 80;
            else if (strstr(label, "CPU") || strstr(label, "Package")) s = 60;
            else if (!strcmp(name, "coretemp") || !strcmp(name, "k10temp")) s = 50;
            else if (!strcmp(name, "acpitz")) s = 20;
            if (s > best) { best = s; snprintf(temp_path, sizeof temp_path, "%s", in); }
        }
        globfree(&g);
    }
    if (glob("/sys/class/thermal/thermal_zone*/type", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            char type[64];
            slurp_line(g.gl_pathv[i], type, sizeof type);
            for (char *c = type; *c; c++) *c = (char)tolower((unsigned char)*c);
            int s = 0;
            if (!strcmp(type, "x86_pkg_temp")) s = 90;
            else if (!strcmp(type, "tcpu")) s = 75;
            else if (strstr(type, "cpu")) s = 70;
            else if (strstr(type, "soc")) s = 40;
            else if (!strcmp(type, "acpitz")) s = 15;
            if (s > best) {
                best = s;
                const char *p = g.gl_pathv[i];
                snprintf(temp_path, sizeof temp_path, "%.*s/temp", (int)(strrchr(p, '/') - p), p);
            }
        }
        globfree(&g);
    }
}

static struct {
    float temp;                        /* degrees C */
    float load;                        /* 0..1, over the last second */
    float load_prev;                   /* the reading before, for easing between them */
    double at;                         /* wall-clock second of the last reading */
    unsigned long long busy, total;
} sens;

static void read_cpu(unsigned long long *busy, unsigned long long *total) {
    char line[256];
    slurp_line("/proc/stat", line, sizeof line);
    unsigned long long v[8] = {0};
    sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    *total = 0;
    for (int i = 0; i < 8; i++) *total += v[i];
    *busy = *total - v[3] - v[4];
}

static void read_sensors(void) {
    unsigned long long b, t;
    read_cpu(&b, &t);
    sens.load_prev = sens.load;
    if (sens.total && t > sens.total) sens.load = clampf((float)(b - sens.busy) / (float)(t - sens.total), 0, 1);
    sens.busy = b; sens.total = t;
    if (temp_path[0]) sens.temp = read_long(temp_path, 45000) / 1000.0f;
    else sens.temp = 45;
    if (opt_load >= 0) sens.load = sens.load_prev = opt_load;
    if (opt_temp >= 0) sens.temp = opt_temp;
}

/* ---- text: a freetype glyph atlas (as in gauges) ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 16
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* regular, italic */
#define AW 1024
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < 2; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:slant=%s:fontformat=CFF,TrueType", opt_font, i ? "italic" : "roman");
        FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        if (FT_New_Face(ft, (const char *)file, 0, &faces[i])) die("can't load the font");
        FcPatternDestroy(m);
        FcPatternDestroy(pat);
    }
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 6) p = 6;
    for (int i = 0; i < nstyles; i++)
        if (styles[i].face == face && styles[i].px == p) return i;
    if (nstyles == MAXSTYLE) die("too many text styles");
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    return nstyles++;
}

static void build_atlas(void) {
    free(atlas);
    int cap = 2048;
    atlas = calloc((size_t)AW * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < nstyles; s++) {
        FT_Face f = faces[styles[s].face];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)styles[s].px);
        for (int c = 0; c < 96; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c == 95 ? 0xB0 : (FT_ULong)(32 + c);
            if (FT_Load_Char(f, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
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
}

static const Glyph *glyph(int s, unsigned char c) {
    int i = c == 0x7f ? 95 : c >= 32 && c < 127 ? c - 32 : 0;
    return &styles[s].g[i];
}

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p; p++) w += glyph(s, (unsigned char)*p)->adv + (p[1] ? track : 0);
    return w;
}

/* text quads: x, y, u, v, r, g, b, a per vertex */
static float *tv;
static int tvn, tvcap;

static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 8192) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x */
static void text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    pen = roundf(pen);
    y = roundf(y);
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph(s, (unsigned char)*p);
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y0, u1, v0, col); quad_v(x1, y1, u1, v1, col);
            quad_v(x0, y0, u0, v0, col); quad_v(x1, y1, u1, v1, col); quad_v(x0, y1, u0, v1, col);
        }
        pen += g->adv + track;
    }
}

/* ==================================================================== */
/* ---- the tank ----
 * World: metres; x right, y up, z away from the viewer into the tank. The
 * front glass is the plane z = 0, the tank floor y = 0, the water surface
 * y = YW, the back glass z = TD. The camera stands outside at (CX, CY, -DC)
 * looking along +z; what it sees through the glass is refracted (see
 * refr_tan in the shaders). */
static float YW = 0.5f, TD = 0.46f, X0 = -0.5f, X1 = 0.5f;
static float CX, CY = 0.25f, DC = 0.9f;
static float KPX = 2000;                /* canvas px per metre at the glass */
#define NW 1.333f

/* the canvas (logical px; y down): the layout's bounding box, or the window */
static float can_x0, can_y0, can_w, can_h;
/* the virtual camera: canvas point (cam_cx, cam_cy) is straight ahead,
 * cam_f the focal length in canvas px; this window shows view_* */
static float cam_cx, cam_cy, cam_f, view_x, view_y, view_w, view_h;
static int front_out = -1;              /* the output under the middle of the layout */

/* the tank's proportions follow the canvas: a landscape screen gets a long,
 * shallow tank, a portrait one a tall column of water */
static void design_tank(float bx, float by, float bw, float bh, float ccx, float ccy) {
    can_x0 = bx; can_y0 = by; can_w = bw; can_h = bh;
    cam_cx = ccx; cam_cy = ccy;
    YW = clampf(0.5f * sqrtf((bh / bw) / 0.625f), 0.40f, 0.86f);
    TD = 0.44f + 0.04f * (YW - 0.5f);
    float top = by + 0.034f * bh, bot = by + bh - 0.026f * bh;
    KPX = (bot - top) / YW;
    CY = YW - (ccy - top) / KPX;
    DC = 0.62f + 0.62f * YW;
    cam_f = KPX * DC;
    CX = 0;
    /* the side glass sits just inside the view at mid-depth, so only the
     * back half of each side pane shows, as it does in a photo of a tank */
    float zm = TD * 0.55f / NW;
    X0 = CX + (bx - ccx) / cam_f * (DC + zm);
    X1 = CX + (bx + bw - ccx) / cam_f * (DC + zm);
}

static float tank_w(void) { return X1 - X0; }

/* ==================================================================== */
/* ---- the aquascape ---- */
typedef struct { V3 c; V3 ax; float rot; uint32_t seed; } Stone;      /* ax: half-extents */
#define MAXSTONE 40
static Stone stones[MAXSTONE];
static int nstone;

#define MAXBR 160
#define BRPTS 10
typedef struct { V3 p[BRPTS]; int n; float r0, r1; int depth; } Branch;
static Branch br[MAXBR];
static int nbr;

typedef struct { float x, z, r, h; } Clump;   /* a plant clump fish keep out of */
#define MAXCLUMP 64
static Clump clumps[MAXCLUMP];
static int nclump;

#define MAXFOCAL 4
static float focal_x[MAXFOCAL], focal_z[MAXFOCAL];
static int nfocal;
static float bubbler_x, bubbler_z, outflow_x, outflow_z, outflow_dir, intake_x;

static float sand_h(float x, float z) {
    float zz = clampf(z / TD, 0, 1);
    float h = 0.038f + 0.07f * powf(smooth(0.0f, 1.0f, zz), 1.15f);
    for (int i = 0; i < nfocal; i++) {
        float dx = (x - focal_x[i]) / 0.16f, dz = (z - focal_z[i]) / 0.12f;
        h += 0.022f * expf(-(dx * dx + dz * dz));
    }
    h += 0.006f * fbm3(x * 5.0f, z * 5.0f, 0.5f, 3, 11);
    return h;
}

static V3 sand_n(float x, float z) {
    float e = 0.003f;
    float hx = sand_h(x + e, z) - sand_h(x - e, z), hz = sand_h(x, z + e) - sand_h(x, z - e);
    return vnorm(v3(-hx, 2 * e, -hz));
}

/* ---- the walkable ground: sand, stones and wood, seen from above ---- */
#define GCELL 0.004f
static float *ground;
static int gnx, gnz;

static float ground_at(float x, float z) {
    float gx = (x - X0) / GCELL, gz = z / GCELL;
    gx = clampf(gx, 0, gnx - 1.001f); gz = clampf(gz, 0, gnz - 1.001f);
    int ix = (int)gx, iz = (int)gz;
    float fx = gx - ix, fz = gz - iz;
    float *g = ground + iz * gnx + ix;
    return mixf(mixf(g[0], g[1], fx), mixf(g[gnx], g[gnx + 1], fx), fz);
}
static V3 ground_n(float x, float z) {
    float e = GCELL * 1.5f;
    float hx = ground_at(x + e, z) - ground_at(x - e, z), hz = ground_at(x, z + e) - ground_at(x, z - e);
    return vnorm(v3(-hx, 2 * e, -hz));
}

/* top-down rasterise a triangle into the ground map */
static void ground_tri(V3 a, V3 b, V3 c) {
    float minx = fminf(a.x, fminf(b.x, c.x)), maxx = fmaxf(a.x, fmaxf(b.x, c.x));
    float minz = fminf(a.z, fminf(b.z, c.z)), maxz = fmaxf(a.z, fmaxf(b.z, c.z));
    int ix0 = (int)floorf((minx - X0) / GCELL), ix1 = (int)ceilf((maxx - X0) / GCELL);
    int iz0 = (int)floorf(minz / GCELL), iz1 = (int)ceilf(maxz / GCELL);
    if (ix0 < 0) ix0 = 0;
    if (iz0 < 0) iz0 = 0;
    if (ix1 >= gnx) ix1 = gnx - 1;
    if (iz1 >= gnz) iz1 = gnz - 1;
    float d = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z);
    if (fabsf(d) < 1e-12f) return;
    for (int iz = iz0; iz <= iz1; iz++)
        for (int ix = ix0; ix <= ix1; ix++) {
            float px = X0 + ix * GCELL, pz = iz * GCELL;
            float l0 = ((b.z - c.z) * (px - c.x) + (c.x - b.x) * (pz - c.z)) / d;
            float l1 = ((c.z - a.z) * (px - c.x) + (a.x - c.x) * (pz - c.z)) / d;
            float l2 = 1 - l0 - l1;
            if (l0 < -0.02f || l1 < -0.02f || l2 < -0.02f) continue;
            float y = l0 * a.y + l1 * b.y + l2 * c.y;
            float *g = ground + iz * gnx + ix;
            if (y > *g) *g = y;
        }
}

/* ==================================================================== */
/* ---- meshes ----
 * The solid vertex: position, normal, uv, material, and three material
 * specific values (12 floats). */
enum { M_SAND, M_SUBSTRATE, M_BACK, M_SIDE, M_SURF, M_FRAME, M_WOOD, M_STONE, M_EEL, M_SHRIMP,
       M_AMANO, M_SEYE, M_MENISCUS, M_CABINET, M_PIPE, M_AIRSTONE, M_CORYBARBEL };
#define SV 12
typedef struct { float *v; int nv, cv; uint32_t *ix; int ni, ci; } Mesh;

static int m_vert(Mesh *m, V3 p, V3 n, float u, float v, float mat, float a0, float a1, float a2) {
    if (m->nv == m->cv) m->v = realloc(m->v, (size_t)(m->cv = m->cv ? m->cv * 2 : 4096) * SV * sizeof(float));
    float *o = m->v + (size_t)m->nv * SV;
    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z;
    o[6] = u; o[7] = v; o[8] = mat; o[9] = a0; o[10] = a1; o[11] = a2;
    return m->nv++;
}
static void m_tri(Mesh *m, int a, int b, int c) {
    if (m->ni + 3 > m->ci) m->ix = realloc(m->ix, (size_t)(m->ci = m->ci ? m->ci * 2 : 8192) * sizeof(uint32_t));
    m->ix[m->ni++] = (uint32_t)a; m->ix[m->ni++] = (uint32_t)b; m->ix[m->ni++] = (uint32_t)c;
}
static V3 m_pos(Mesh *m, int i) { float *o = m->v + (size_t)i * SV; return v3(o[0], o[1], o[2]); }
static void m_setn(Mesh *m, int i, V3 n) { float *o = m->v + (size_t)i * SV + 3; o[0] = n.x; o[1] = n.y; o[2] = n.z; }
static V3 m_nrm(Mesh *m, int i) { float *o = m->v + (size_t)i * SV + 3; return v3(o[0], o[1], o[2]); }

/* a grid of (nu+1) x (nv+1) vertices from f(u, v) */
typedef void (*GridFn)(float u, float v, V3 *p, V3 *n, float *a, void *ctx);
static void m_grid(Mesh *m, int nu, int nv, GridFn f, void *ctx, float mat, int flip) {
    int base = m->nv;
    for (int j = 0; j <= nv; j++)
        for (int i = 0; i <= nu; i++) {
            float u = (float)i / nu, v = (float)j / nv, a[3] = {0, 0, 0};
            V3 p, n;
            f(u, v, &p, &n, a, ctx);
            m_vert(m, p, n, u, v, mat, a[0], a[1], a[2]);
        }
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            int a = base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
            if (flip) { m_tri(m, a, c, b); m_tri(m, b, c, d); }
            else { m_tri(m, a, b, c); m_tri(m, b, d, c); }
        }
}

/* smooth normals from the faces, for vertices [v0, nv) and indices [i0, ni) */
static void m_normals(Mesh *m, int v0, int i0) {
    for (int i = v0; i < m->nv; i++) m_setn(m, i, v3(0, 0, 0));
    for (int k = i0; k + 2 < m->ni; k += 3) {
        int a = (int)m->ix[k], b = (int)m->ix[k + 1], c = (int)m->ix[k + 2];
        V3 pa = m_pos(m, a), fn = vcross(vsub(m_pos(m, b), pa), vsub(m_pos(m, c), pa));
        m_setn(m, a, vadd(m_nrm(m, a), fn)); m_setn(m, b, vadd(m_nrm(m, b), fn)); m_setn(m, c, vadd(m_nrm(m, c), fn));
    }
    for (int i = v0; i < m->nv; i++) m_setn(m, i, vnorm(m_nrm(m, i)));
}

static Mesh scene_mesh;                 /* everything that never moves */
static int scene_ni_opaque;             /* sand .. stones; then the glass and surface */

/* -- the tank's own surfaces -- */
static void g_sand(float u, float v, V3 *p, V3 *n, float *a, void *ctx) {
    (void)ctx; (void)a;
    float x = mixf(X0 - 0.01f, X1 + 0.01f, u), z = mixf(0.0f, TD, v * v * 0.35f + v * 0.65f);
    *p = v3(x, sand_h(x, z), z);
    *n = sand_n(x, z);
}
static void g_front(float u, float v, V3 *p, V3 *n, float *a, void *ctx) {
    (void)ctx; (void)a;
    float x = mixf(X0 - 0.01f, X1 + 0.01f, u);
    *p = v3(x, v * sand_h(x, 0), 0.0004f);
    *n = v3(0, 0, -1);
}
static void g_back(float u, float v, V3 *p, V3 *n, float *a, void *ctx) {
    (void)ctx; (void)a;
    float x = mixf(X0 - 0.01f, X1 + 0.01f, u);
    *p = v3(x, v * YW, TD);
    *n = v3(0, 0, -1);
}
static void g_surf(float u, float v, V3 *p, V3 *n, float *a, void *ctx) {
    (void)ctx; (void)a;
    *p = v3(mixf(X0 - 0.01f, X1 + 0.01f, u), YW, v * v * TD);
    *n = v3(0, -1, 0);
}
static void g_side(float u, float v, V3 *p, V3 *n, float *a, void *ctx) {
    float x = *(float *)ctx;
    (void)a;
    *p = v3(x, v * YW, u * u * TD);
    *n = v3(x < 0 ? 1.0f : -1.0f, 0, 0);
}
/* the black trim round the front glass, and the cabinet under it: drawn in
 * front of the glass, so not refracted */
static void quad_front(Mesh *m, float x0, float y0, float x1, float y1, float z, float mat) {
    int a = m_vert(m, v3(x0, y0, z), v3(0, 0, -1), 0, 0, mat, 0, 0, 0);
    int b = m_vert(m, v3(x1, y0, z), v3(0, 0, -1), 1, 0, mat, 0, 0, 0);
    int c = m_vert(m, v3(x0, y1, z), v3(0, 0, -1), 0, 1, mat, 0, 0, 0);
    int d = m_vert(m, v3(x1, y1, z), v3(0, 0, -1), 1, 1, mat, 0, 0, 0);
    m_tri(m, a, b, c); m_tri(m, b, d, c);
}

/* -- stones: displaced spheres, flattened and sunk into the sand -- */
static float stone_disp(V3 d, uint32_t s) {
    float r = 1.0f;
    float n1 = fbm3(d.x * 1.4f + s % 97, d.y * 1.4f, d.z * 1.4f, 3, s);
    float ridge = 1.0f - fabsf(fbm3(d.x * 2.6f, d.y * 2.6f + s % 89, d.z * 2.6f, 4, s + 5));
    r += 0.22f * n1 + 0.10f * ridge * ridge;
    /* strata: soft terraces across the stone */
    float st = d.y * 3.0f + 0.4f * n1;
    r += 0.025f * (st - floorf(st));
    /* facets: a few flat cuts, like split rock */
    for (int k = 0; k < 4; k++) {
        V3 pn = vnorm(v3(hf(s, (uint32_t)k, 1) * 2 - 1, hf(s, (uint32_t)k, 2) * 1.2f - 0.2f, hf(s, (uint32_t)k, 3) * 2 - 1));
        float c = vdot(d, pn), cut = 0.62f + 0.25f * hf(s, (uint32_t)k, 4);
        if (c > cut) r *= mixf(1.0f, cut / c, 0.8f);
    }
    return r;
}

static void build_stone(Mesh *m, const Stone *st) {
    /* an icosphere, subdivided four times */
    static const float ico_v[12][3] = {
        {-1, 1.618f, 0}, {1, 1.618f, 0}, {-1, -1.618f, 0}, {1, -1.618f, 0}, {0, -1, 1.618f}, {0, 1, 1.618f},
        {0, -1, -1.618f}, {0, 1, -1.618f}, {1.618f, 0, -1}, {1.618f, 0, 1}, {-1.618f, 0, -1}, {-1.618f, 0, 1}};
    static const int ico_f[20][3] = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6},
        {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    int nf = 20, cap = 20 * 256;
    V3 *tri = malloc(sizeof(V3) * 3 * (size_t)cap);
    for (int f = 0; f < 20; f++)
        for (int k = 0; k < 3; k++) tri[f * 3 + k] = vnorm(v3(ico_v[ico_f[f][k]][0], ico_v[ico_f[f][k]][1], ico_v[ico_f[f][k]][2]));
    for (int it = 0; it < 4; it++) {
        V3 *nt = malloc(sizeof(V3) * 3 * (size_t)cap);
        for (int f = 0; f < nf; f++) {
            V3 a = tri[f * 3], b = tri[f * 3 + 1], c = tri[f * 3 + 2];
            V3 ab = vnorm(vadd(a, b)), bc = vnorm(vadd(b, c)), ca = vnorm(vadd(c, a));
            V3 q[4][3] = {{a, ab, ca}, {ab, b, bc}, {ca, bc, c}, {ab, bc, ca}};
            for (int s = 0; s < 4; s++)
                for (int k = 0; k < 3; k++) nt[(f * 4 + s) * 3 + k] = q[s][k];
        }
        free(tri);
        tri = nt;
        nf *= 4;
    }
    /* weld by brute hashing on the unit-sphere position */
    int v0 = m->nv, i0 = m->ni;
    int hn = 1 << 14;
    int *ht = malloc(sizeof(int) * (size_t)hn);
    for (int i = 0; i < hn; i++) ht[i] = -1;
    int *next = malloc(sizeof(int) * (size_t)nf * 3);
    V3 *uv = malloc(sizeof(V3) * (size_t)nf * 3);
    int nu = 0;
    int *map = malloc(sizeof(int) * (size_t)nf * 3);
    float cr = cosf(st->rot), sr = sinf(st->rot);
    for (int k = 0; k < nf * 3; k++) {
        V3 d = tri[k];
        uint32_t h = hash3((uint32_t)lroundf(d.x * 4096), (uint32_t)lroundf(d.y * 4096), (uint32_t)lroundf(d.z * 4096)) & (uint32_t)(hn - 1);
        int f = -1;
        for (int e = ht[h]; e >= 0; e = next[e])
            if (vlen(vsub(uv[e], d)) < 1e-4f) { f = e; break; }
        if (f < 0) {
            f = nu++;
            uv[f] = d;
            next[f] = ht[h];
            ht[h] = f;
            float r = stone_disp(d, st->seed);
            V3 p = v3(d.x * st->ax.x * r, d.y * st->ax.y * r, d.z * st->ax.z * r);
            /* the bottom flattens where it beds into the sand */
            if (p.y < -st->ax.y * 0.35f) p.y = -st->ax.y * 0.35f + (p.y + st->ax.y * 0.35f) * 0.3f;
            V3 w = v3(st->c.x + p.x * cr - p.z * sr, st->c.y + p.y, st->c.z + p.x * sr + p.z * cr);
            m_vert(m, w, v3(0, 1, 0), 0, 0, M_STONE, d.x * 3 + st->seed % 7, d.y * 3, d.z * 3);
        }
        map[k] = v0 + f;
    }
    for (int f = 0; f < nf; f++) m_tri(m, map[f * 3], map[f * 3 + 2], map[f * 3 + 1]);
    m_normals(m, v0, i0);
    /* outward-facing? (the winding of the icosahedron table) */
    {
        V3 p = m_pos(m, v0), n = m_nrm(m, v0);
        if (vdot(vsub(p, st->c), n) < 0) {
            for (int k = i0; k + 2 < m->ni; k += 3) { uint32_t t = m->ix[k + 1]; m->ix[k + 1] = m->ix[k + 2]; m->ix[k + 2] = t; }
            for (int i = v0; i < m->nv; i++) m_setn(m, i, vmul(m_nrm(m, i), -1));
        }
    }
    free(tri); free(ht); free(next); free(uv); free(map);
}

/* -- driftwood: branching tubes with grooved bark -- */
static V3 br_eval(const Branch *b, float t, V3 *tan) {
    float f = t * (b->n - 1);
    int i = (int)floorf(f);
    if (i > b->n - 2) i = b->n - 2;
    if (i < 0) i = 0;
    float u = f - i;
    V3 p0 = b->p[i > 0 ? i - 1 : 0], p1 = b->p[i], p2 = b->p[i + 1], p3 = b->p[i + 2 < b->n ? i + 2 : b->n - 1];
    if (tan) {
        V3 a = vcatmull(p0, p1, p2, p3, fmaxf(u - 0.01f, 0)), c = vcatmull(p0, p1, p2, p3, fminf(u + 0.01f, 1));
        *tan = vnorm(vsub(c, a));
    }
    return vcatmull(p0, p1, p2, p3, u);
}
static float br_len(const Branch *b) {
    float l = 0;
    for (int i = 1; i < b->n; i++) l += vlen(vsub(b->p[i], b->p[i - 1]));
    return l;
}
static float br_radius(const Branch *b, float t) {
    return mixf(b->r0, b->r1, powf(t, 0.8f));
}

static void build_branch(Mesh *m, const Branch *b, uint32_t seed) {
    float len = br_len(b);
    int nr = (int)(len / 0.005f) + 4, ns = b->r0 > 0.01f ? 12 : 9;
    int v0 = m->nv, i0 = m->ni;
    V3 tan, prev_n = v3(0, 0, 0);
    for (int r = 0; r <= nr; r++) {
        float t = (float)r / nr;
        V3 c = br_eval(b, t, &tan);
        /* parallel transport of the ring's frame */
        V3 nn = r == 0 ? vperp(tan) : vnorm(vsub(prev_n, vmul(tan, vdot(prev_n, tan))));
        prev_n = nn;
        V3 bn = vcross(tan, nn);
        float rad = br_radius(b, t);
        float along = t * len;
        if (r == nr) rad *= 0.35f;
        for (int s = 0; s <= ns; s++) {
            float a = (float)s / ns * TAU;
            V3 d = vadd(vmul(nn, cosf(a)), vmul(bn, sinf(a)));
            /* knots and grooves */
            float gro = 0.10f * sinf(a * 5 + along * 18 + (seed % 13)) * (0.6f + 0.4f * sinf(along * 7));
            float knot = 0.35f * powf(fmaxf(0, vnoise2(along * 25, a * 1.2f, seed) - 0.62f) / 0.38f, 2);
            float rr = rad * (1 + gro + knot + 0.15f * (vnoise2(along * 60, a * 3, seed + 3) - 0.5f));
            m_vert(m, vmad(c, d, rr), d, along, a / TAU, M_WOOD, rad, (float)(seed % 1000) * 0.01f, (float)b->depth);
        }
    }
    for (int r = 0; r < nr; r++)
        for (int s = 0; s < ns; s++) {
            int a = v0 + r * (ns + 1) + s, bb = a + 1, c = a + ns + 1, d = c + 1;
            m_tri(m, a, c, bb); m_tri(m, bb, c, d);
        }
    m_normals(m, v0, i0);
    /* the tip: a little cap */
    int tipc = m_vert(m, br_eval(b, 1, &tan), tan, len, 0, M_WOOD, b->r1 * 0.3f, (float)(seed % 1000) * 0.01f, (float)b->depth);
    for (int s = 0; s < ns; s++) m_tri(m, v0 + nr * (ns + 1) + s, tipc, v0 + nr * (ns + 1) + s + 1);
}

/* -- tubes for the equipment (airline, filter pipes) -- */
static void build_tube(Mesh *m, const V3 *pts, int np, float rad, float mat, int ns) {
    int v0 = m->nv;
    V3 prev_n = v3(0, 0, 0);
    for (int i = 0; i < np; i++) {
        V3 tan = vnorm(vsub(pts[i < np - 1 ? i + 1 : i], pts[i > 0 ? i - 1 : i]));
        V3 nn = i == 0 ? vperp(tan) : vnorm(vsub(prev_n, vmul(tan, vdot(prev_n, tan))));
        prev_n = nn;
        V3 bn = vcross(tan, nn);
        for (int s = 0; s <= ns; s++) {
            float a = (float)s / ns * TAU;
            V3 d = vadd(vmul(nn, cosf(a)), vmul(bn, sinf(a)));
            m_vert(m, vmad(pts[i], d, rad), d, (float)i / (np - 1), (float)s / ns, mat, rad, 0, 0);
        }
    }
    for (int i = 0; i < np - 1; i++)
        for (int s = 0; s < ns; s++) {
            int a = v0 + i * (ns + 1) + s, b = a + 1, c = a + ns + 1, d = c + 1;
            m_tri(m, a, c, b); m_tri(m, b, c, d);
        }
}

/* ==================================================================== */
/* ---- plants: instanced leaves ----
 * Every leaf, blade, ribbon, stem and sprig is an instance of one of four
 * strip meshes, shaped and swayed in the vertex shader. 20 floats each:
 *   attach xyz, type | dir xyz, length | side xyz, width |
 *   root y, plant height, flex, seed | curl, fold, hue, height fraction */
enum { L_ROTALA, L_LUDWIGIA, L_HYGRO, L_SWORD, L_VALLIS, L_HAIR, L_MONTE, L_MOSS, L_ANUBIAS, L_FERN, L_STEM, L_LITTER, L_BUCE };
enum { LM_TINY, LM_LEAF, LM_RIBBON, LM_BLADE, LM_N };
#define LI 20
typedef struct { float *v; int n, cap; } Inst;
static Inst leaves[LM_N], leaves_s[LM_N];     /* swaying, and still (baked) */

static void leaf_add(int mesh, int type, V3 at, V3 dir, V3 side, float len, float wid,
                     float root_y, float ph, float flex, float curl, float fold, float hue, float hfrac) {
    int still = type == L_MONTE || type == L_HAIR || type == L_MOSS || type == L_LITTER || type == L_ANUBIAS || type == L_FERN;
    Inst *L = still ? &leaves_s[mesh] : &leaves[mesh];
    if (L->n == L->cap) L->v = realloc(L->v, (size_t)(L->cap = L->cap ? L->cap * 2 : 1024) * LI * sizeof(float));
    float *o = L->v + (size_t)L->n * LI;
    dir = vnorm(dir);
    side = vnorm(vsub(side, vmul(dir, vdot(side, dir))));
    o[0] = at.x; o[1] = at.y; o[2] = at.z; o[3] = (float)type;
    o[4] = dir.x; o[5] = dir.y; o[6] = dir.z; o[7] = len;
    o[8] = side.x; o[9] = side.y; o[10] = side.z; o[11] = wid;
    o[12] = root_y; o[13] = ph; o[14] = flex; o[15] = frand();
    o[16] = curl; o[17] = fold; o[18] = hue; o[19] = hfrac;
    L->n++;
}

/* a leaf whose blade faces up-ish: side is horizontal, perpendicular to dir */
static V3 leaf_side(V3 dir, float twist) {
    V3 s = vcross(v3(0, 1, 0), dir);
    if (vlen(s) < 1e-3f) s = v3(1, 0, 0);
    s = vnorm(s);
    return vrot(s, vnorm(dir), twist);
}

static void add_clump(float x, float z, float r, float h) {
    if (nclump < MAXCLUMP) clumps[nclump++] = (Clump){x, z, r, h};
}

/* a group of stem plants: Rotala, Ludwigia or Hygrophila */
static void plant_stems(float cx, float cz, float rad, float hgt, int kind, int nstems) {
    int lt = kind == 0 ? L_ROTALA : kind == 1 ? L_LUDWIGIA : L_HYGRO;
    float hue = frange(-1, 1);
    for (int s = 0; s < nstems; s++) {
        float a = frand() * TAU, r = rad * sqrtf(frand());
        float x = cx + cosf(a) * r, z = clampf(cz + sinf(a) * r * 0.7f, 0.03f, TD - 0.015f);
        float y0 = sand_h(x, z) - 0.005f;
        float h = hgt * frange(0.7f, 1.05f) * (1 - 0.25f * r / rad);
        h = fminf(h, YW - y0 - 0.015f);
        V3 lean = v3(frange(-0.12f, 0.12f), 1, frange(-0.12f, 0.12f));
        V3 dir = vnorm(lean);
        float sflex = 1.0f;
        leaf_add(LM_RIBBON, L_STEM, v3(x, y0, z), dir, v3(1, 0, 0), h, kind == 1 ? 0.0024f : 0.0017f, y0, h, sflex, 0, 0, hue, 0);
        /* leaves in pairs (opposite, each pair turned a quarter) or whorls */
        float node = kind == 0 ? 0.006f : kind == 1 ? 0.0095f : 0.012f;
        int per = kind == 0 ? 3 : 2;
        float rot = frand() * TAU;
        for (float hh = 0.012f + frand() * node; hh < h; hh += node * frange(0.85f, 1.15f)) {
            float f = hh / h;
            rot += kind == 0 ? 0.9f : PI / 2;
            for (int k = 0; k < per; k++) {
                float az = rot + k * TAU / per;
                V3 out = v3(cosf(az), 0, sinf(az));
                /* young leaves at the tip point up, old ones spread */
                float up = mixf(0.35f, 1.6f, powf(f, 3));
                V3 ldir = vnorm(vadd(out, vmul(dir, up)));
                V3 at = vmad(v3(x, y0, z), dir, hh);
                float len, wid;
                if (kind == 0) { len = 0.016f * mixf(1.0f, 0.65f, f); wid = 0.002f; }
                else if (kind == 1) { len = 0.018f * mixf(1.0f, 0.7f, f); wid = 0.0075f; }
                else { len = 0.032f * mixf(1.0f, 0.7f, f); wid = 0.0062f; }
                len *= frange(0.85f, 1.15f);
                int mesh = kind == 0 ? LM_TINY : LM_LEAF;
                leaf_add(mesh, lt, at, ldir, leaf_side(ldir, frange(-0.3f, 0.3f)), len, wid, y0, h, sflex,
                         kind == 2 ? -0.8f : -0.4f, 0.5f, hue, f);
            }
        }
    }
    add_clump(cx, cz, rad + 0.01f, hgt);
}

/* an Amazon sword: a rosette of long-stalked lance leaves */
static void plant_sword(float cx, float cz, float size) {
    float y0 = sand_h(cx, cz) - 0.004f;
    int n = 10 + (int)(frand() * 7);
    float hue = frange(-1, 1);
    for (int i = 0; i < n; i++) {
        float az = frand() * TAU, f = frand();
        float elev = mixf(1.35f, 0.55f, f * f);           /* inner leaves stand up */
        V3 dir = vnorm(v3(cosf(az) * cosf(elev), sinf(elev), sinf(az) * cosf(elev)));
        float len = size * mixf(0.75f, 1.1f, frand()) * (1.15f - 0.4f * f * f);
        leaf_add(LM_LEAF, L_SWORD, v3(cx, y0, cz), dir, leaf_side(dir, frange(-0.35f, 0.35f)), len, len * 0.16f,
                 y0, size * 0.9f, 0.6f, -mixf(0.6f, 1.6f, f), 0.6f, hue, 0.5f);
    }
    add_clump(cx, cz, size * 0.55f, size * 0.8f);
}

/* Vallisneria: ribbons rising to the surface */
static void plant_vallis(float cx, float cz, float rad, int n) {
    float hue = frange(-1, 1);
    for (int i = 0; i < n; i++) {
        float a = frand() * TAU, r = rad * sqrtf(frand());
        float x = cx + cosf(a) * r, z = clampf(cz + sinf(a) * r * 0.6f, 0.03f, TD - 0.01f);
        float y0 = sand_h(x, z) - 0.004f;
        float len = (YW - y0) * frange(0.7f, 1.25f);
        V3 dir = vnorm(v3(frange(-0.2f, 0.2f), 1, frange(-0.15f, 0.15f)));
        float tw = frand() * PI;
        leaf_add(LM_RIBBON, L_VALLIS, v3(x, y0, z), dir, vrot(v3(1, 0, 0), v3(0, 1, 0), tw), len, frange(0.006f, 0.0095f),
                 y0, len, 1.6f, -frange(0.2f, 0.8f), 0.15f, hue, 0);
    }
    add_clump(cx, cz, rad + 0.01f, YW * 0.9f);
}

/* dwarf hairgrass tufts and Monte Carlo cushions over the foreground */
static void plant_carpet(float x0, float x1, float z0, float z1, float density) {
    int n = (int)((x1 - x0) * (z1 - z0) * density);
    for (int i = 0; i < n; i++) {
        float x = frange(x0, x1), z = frange(z0, z1);
        /* patches, with sand showing between them */
        float pat = fbm3(x * 9, z * 9, 3.3f, 3, 21);
        if (pat < -0.08f) continue;
        float y0 = sand_h(x, z);
        if (pat > 0.18f || hf((uint32_t)(x * 997), (uint32_t)(z * 991), 3) < 0.35f) {
            /* Monte Carlo: tiny round leaves on creeping runners */
            V3 dir = vnorm(v3(frange(-1, 1), frange(0.3f, 1.2f), frange(-1, 1)));
            leaf_add(LM_TINY, L_MONTE, v3(x, y0 + frange(0.0f, 0.012f) * (pat + 0.3f), z), dir, leaf_side(dir, frange(-0.5f, 0.5f)),
                     0.0055f, 0.0055f, y0 - 0.01f, 0.03f, 0.2f, 0, 0.2f, frange(-1, 1), 0);
        } else {
            /* hairgrass: a tuft of fine blades */
            int nb = 5 + (int)(frand() * 6);
            float h = frange(0.025f, 0.055f);
            for (int b = 0; b < nb; b++) {
                V3 dir = vnorm(v3(frange(-0.35f, 0.35f), 1, frange(-0.35f, 0.35f)));
                float bx = x + frange(-0.003f, 0.003f), bz = z + frange(-0.003f, 0.003f);
                leaf_add(LM_BLADE, L_HAIR, v3(bx, y0 - 0.002f, bz), dir, leaf_side(dir, frand() * PI), h * frange(0.6f, 1.1f), 0.0008f,
                         y0 - 0.002f, h, 0.9f, -frange(0.2f, 1.2f), 0, frange(-1, 1), 0);
            }
        }
    }
}

/* epiphytes on the wood: moss everywhere on top, a few ferns and Anubias */
static void plant_epiphytes(void) {
    for (int bi = 0; bi < nbr; bi++) {
        const Branch *b = &br[bi];
        float len = br_len(b);
        int n = (int)(len * (b->depth == 0 ? 900 : 500));
        for (int i = 0; i < n; i++) {
            float t = frand();
            V3 tan, c = br_eval(b, t, &tan);
            float rad = br_radius(b, t);
            /* the upper side of the branch */
            V3 up = vnorm(vsub(v3(0, 1, 0), vmul(tan, tan.y)));
            float a = fnorm() * 0.9f;
            V3 side = vnorm(vcross(tan, up));
            V3 d = vnorm(vadd(vmul(up, cosf(a)), vmul(side, sinf(a))));
            if (d.y < 0.15f) continue;
            /* moss clings in patches */
            if (vnoise2(t * len * 30, (float)bi, 77) < 0.42f) continue;
            V3 at = vmad(c, d, rad * 0.9f);
            V3 dir = vnorm(vadd(d, v3(frange(-0.5f, 0.5f), frange(-0.1f, 0.4f), frange(-0.5f, 0.5f))));
            leaf_add(LM_TINY, L_MOSS, at, dir, leaf_side(dir, frand() * PI), frange(0.008f, 0.016f), frange(0.006f, 0.01f),
                     at.y - 0.02f, 0.05f, 0.25f, -frange(0, 1.2f), 0, frange(-1, 1), 0);
        }
    }
    /* Anubias and Java fern rhizomes tied to the thicker branches */
    int ne = 2 + nfocal * 2;
    for (int e = 0; e < ne && nbr; e++) {
        const Branch *b = &br[(int)(frand() * nbr)];
        if (b->r0 < 0.007f) continue;
        float t = frange(0.1f, 0.55f);
        V3 tan, c = br_eval(b, t, &tan);
        c.y += br_radius(b, t) * 0.8f;
        int fern = frand() < 0.5f;
        int n = fern ? 7 + (int)(frand() * 5) : 6 + (int)(frand() * 5);
        for (int i = 0; i < n; i++) {
            float az = frand() * TAU;
            float elev = fern ? frange(0.5f, 1.25f) : frange(0.05f, 0.7f);
            V3 dir = vnorm(v3(cosf(az) * cosf(elev), sinf(elev), sinf(az) * cosf(elev)));
            V3 at = vadd(c, vmul(tan, frange(-0.02f, 0.02f)));
            if (fern)
                leaf_add(LM_LEAF, L_FERN, at, dir, leaf_side(dir, frange(-0.4f, 0.4f)), frange(0.07f, 0.13f), 0.012f,
                         at.y - 0.03f, 0.12f, 0.45f, -frange(0.4f, 1.1f), 0.5f, frange(-1, 1), 0.5f);
            else
                leaf_add(LM_LEAF, L_ANUBIAS, at, dir, leaf_side(dir, frange(-0.2f, 0.2f)), frange(0.04f, 0.065f), 0.02f,
                         at.y - 0.03f, 0.06f, 0.15f, -frange(0.1f, 0.5f), 0.25f, frange(-1, 1), 0.5f);
        }
    }
}

/* a few dead Indian almond leaves on the sand, the blackwater keeper's touch */
static void plant_litter(void) {
    int n = (int)(tank_w() * 7);
    for (int i = 0; i < n; i++) {
        float x = frange(X0 + 0.03f, X1 - 0.03f), z = frange(0.04f, TD * 0.6f);
        float y = sand_h(x, z) + 0.0015f;
        float az = frand() * TAU;
        V3 dir = vnorm(v3(cosf(az), frange(-0.02f, 0.06f), sinf(az)));
        leaf_add(LM_LEAF, L_LITTER, v3(x, y, z), dir, leaf_side(dir, frange(-0.15f, 0.15f)), frange(0.04f, 0.075f), 0.024f,
                 y - 0.01f, 0.02f, 0.0f, frange(-0.3f, 0.6f), frange(-0.4f, 0.6f), frange(-1, 1), 0);
    }
}

/* ---- the whole scape, from the seed ---- */
static void grow_branch(V3 base, V3 dir, float len, float r0, int depth) {
    if (nbr >= MAXBR) return;
    Branch *b = &br[nbr++];
    b->depth = depth;
    b->r0 = r0;
    b->r1 = fmaxf(r0 * 0.25f, 0.0018f);
    b->n = BRPTS;
    V3 p = base, d = vnorm(dir);
    float seg = len / (BRPTS - 1);
    for (int i = 0; i < BRPTS; i++) {
        b->p[i] = p;
        /* wander, a little upward, kept inside the water and the glass */
        d = vnorm(vadd(d, v3(fnorm() * 0.22f, fnorm() * 0.18f + 0.05f, fnorm() * 0.2f)));
        if (p.y > YW - 0.06f && d.y > 0) d.y *= 0.2f;
        if (p.z < 0.08f && d.z < 0) d.z = -d.z * 0.5f;
        if (p.z > TD - 0.04f && d.z > 0) d.z = -d.z * 0.5f;
        d = vnorm(d);
        p = vmad(p, d, seg);
        p.y = fminf(p.y, YW - 0.035f);
        p.x = clampf(p.x, X0 + 0.02f, X1 - 0.02f);
    }
    Branch me = *b;
    if (depth >= 2) return;
    int nk = depth == 0 ? 2 + (int)(frand() * 2.5f) : (int)(frand() * 2.5f);
    for (int k = 0; k < nk; k++) {
        float t = frange(0.3f, 0.85f);
        V3 tan, at = br_eval(&me, t, &tan);
        V3 off = vnorm(vrot(vperp(tan), tan, frand() * TAU));
        V3 nd = vnorm(vadd(vmul(tan, 0.8f), vadd(off, v3(0, 0.35f, 0))));
        grow_branch(at, nd, len * frange(0.3f, 0.6f) * (1 - t * 0.4f), br_radius(&me, t) * 0.7f, depth + 1);
    }
}

static void build_scape(void) {
    float w = tank_w();
    /* focal points of the hardscape: a golden-section spot on a single
     * screen, more along a long tank */
    nfocal = w < 1.0f ? 1 : w < 1.7f ? 2 : 3;
    if (nfocal > MAXFOCAL) nfocal = MAXFOCAL;
    for (int i = 0; i < nfocal; i++) {
        float f = nfocal == 1 ? (frand() < 0.5f ? 0.36f : 0.64f) : (i + 0.5f) / nfocal + frange(-0.06f, 0.06f);
        focal_x[i] = X0 + w * f;
        focal_z[i] = TD * frange(0.48f, 0.62f);
    }
    /* stones round each focal point, then a few strays */
    nstone = 0;
    for (int i = 0; i < nfocal; i++) {
        int n = 3 + (int)(frand() * 3);
        for (int k = 0; k < n && nstone < MAXSTONE; k++) {
            Stone *s = &stones[nstone++];
            float big = k == 0 ? 1.0f : frange(0.4f, 0.8f);
            float r = 0.075f * big * (0.85f + 0.3f * YW);
            float a = frand() * TAU, d = k == 0 ? 0.02f : frange(0.06f, 0.15f);
            s->c = v3(focal_x[i] + cosf(a) * d * 1.3f, 0, clampf(focal_z[i] - 0.03f + sinf(a) * d * 0.8f, 0.07f, TD - 0.06f));
            s->ax = v3(r * frange(0.9f, 1.3f), r * frange(0.55f, 0.95f) * (k == 0 ? 1.25f : 1), r * frange(0.7f, 1.0f));
            s->c.y = sand_h(s->c.x, s->c.z) + s->ax.y * 0.35f;
            s->rot = frand() * TAU;
            s->seed = hash32(opt_seed + 101u * (uint32_t)nstone);
        }
    }
    for (int k = 0; k < 2 + (int)(w * 4) && nstone < MAXSTONE; k++) {
        Stone *s = &stones[nstone++];
        float r = frange(0.012f, 0.03f);
        s->c = v3(frange(X0 + 0.05f, X1 - 0.05f), 0, frange(0.05f, TD * 0.7f));
        s->ax = v3(r * frange(0.9f, 1.4f), r * frange(0.5f, 0.8f), r);
        s->c.y = sand_h(s->c.x, s->c.z) + s->ax.y * 0.25f;
        s->rot = frand() * TAU;
        s->seed = hash32(opt_seed + 101u * (uint32_t)nstone);
    }
    /* driftwood rising from behind the stones, leaning out */
    nbr = 0;
    for (int i = 0; i < nfocal; i++) {
        int n = 2 + (int)(frand() * 2);
        for (int k = 0; k < n; k++) {
            float side = (k % 2 ? 1.0f : -1.0f) * frange(0.4f, 1.1f);
            V3 base = v3(focal_x[i] + frange(-0.04f, 0.04f), 0, focal_z[i] + frange(0.0f, 0.06f));
            base.y = sand_h(base.x, base.z) - 0.02f;
            V3 dir = vnorm(v3(side, frange(0.9f, 1.5f), frange(-0.25f, 0.15f)));
            grow_branch(base, dir, YW * frange(0.55f, 0.9f), frange(0.011f, 0.017f), 0);
        }
        /* a root lying along the sand */
        V3 rb = v3(focal_x[i] + frange(-0.05f, 0.05f), 0, focal_z[i] - 0.05f);
        rb.y = sand_h(rb.x, rb.z) - 0.004f;
        grow_branch(rb, v3(frange(-1, 1), 0.05f, frange(-0.6f, -0.1f)), frange(0.12f, 0.2f), 0.009f, 1);
    }
    /* equipment: the bubbler at one end at the back, the filter's outflow at the other */
    int left = frand() < 0.5f;
    bubbler_x = left ? X0 + w * 0.08f : X1 - w * 0.08f;
    bubbler_z = TD * 0.82f;
    outflow_x = left ? X1 - w * 0.06f : X0 + w * 0.06f;
    outflow_z = TD * 0.9f;
    outflow_dir = left ? -1.0f : 1.0f;
    intake_x = outflow_x - outflow_dir * 0.07f;

    /* plants */
    nclump = 0;
    for (int i = 0; i < LM_N; i++) leaves[i].n = leaves_s[i].n = 0;
    /* the background: stem plants in groups of a kind, with Vallisneria at the ends */
    float gx = X0 + 0.02f;
    int kind = (int)(frand() * 3);
    while (gx < X1) {
        float gw = frange(0.07f, 0.14f);
        float cx = gx + gw / 2;
        float endf = fminf(cx - X0, X1 - cx) / w;
        if (endf < 0.1f) plant_vallis(cx, TD * frange(0.78f, 0.9f), gw * 0.55f, (int)(gw * 300));
        else {
            float h = (YW - 0.1f) * frange(0.55f, 0.95f);
            plant_stems(cx, TD * frange(0.78f, 0.88f), gw * 0.5f, h, kind, (int)(gw * 230));
            kind = (kind + 1 + (int)(frand() * 2)) % 3;
        }
        gx += gw * frange(0.85f, 1.05f);
    }
    /* midground: a sword by each focal point, a shorter stem group or two */
    for (int i = 0; i < nfocal; i++) {
        float sx = focal_x[i] + (frand() < 0.5f ? -1 : 1) * frange(0.12f, 0.2f);
        plant_sword(clampf(sx, X0 + 0.1f, X1 - 0.1f), TD * frange(0.6f, 0.72f), YW * frange(0.3f, 0.4f));
        float hx = focal_x[i] + (sx > focal_x[i] ? -1 : 1) * frange(0.14f, 0.24f);
        plant_stems(clampf(hx, X0 + 0.05f, X1 - 0.05f), TD * frange(0.6f, 0.7f), 0.04f, YW * frange(0.25f, 0.4f), (int)(frand() * 3), 12);
    }
    /* foreground carpet, leaving an open sand path near the front */
    plant_carpet(X0, X1, 0.02f, TD * 0.5f, 26000);
    plant_epiphytes();
    plant_litter();
}

/* ---- the static mesh, and the ground map from it ---- */
static int nscene_stone_i0, nscene_stone_i1;
static void build_scene_mesh(void) {
    Mesh *m = &scene_mesh;
    m->nv = m->ni = 0;
    float w = tank_w();
    int nu = (int)(w / 0.006f) + 8;
    m_grid(m, nu, 70, g_sand, NULL, M_SAND, 0);
    for (int i = 0; i < nstone; i++) build_stone(m, &stones[i]);
    for (int i = 0; i < nbr; i++) build_branch(m, &br[i], hash32(opt_seed + 7u * (uint32_t)i));
    /* the airline to the bubbler, and the filter's pipes */
    {
        V3 a[6];
        float y0 = sand_h(bubbler_x, bubbler_z);
        for (int i = 0; i < 6; i++) a[i] = v3(bubbler_x + 0.004f * sinf(i * 1.3f), y0 + 0.012f + (YW + 0.03f - y0) * i / 5.0f, TD - 0.012f);
        a[0] = v3(bubbler_x, y0 + 0.012f, bubbler_z);
        a[1] = v3(bubbler_x, y0 + 0.02f, mixf(bubbler_z, TD - 0.012f, 0.7f));
        build_tube(m, a, 6, 0.0025f, M_PIPE, 8);
        V3 st[2] = {v3(bubbler_x, y0 - 0.004f, bubbler_z), v3(bubbler_x, y0 + 0.013f, bubbler_z)};
        build_tube(m, st, 2, 0.013f, M_AIRSTONE, 16);
        V3 o[8];
        for (int i = 0; i < 6; i++) o[i] = v3(outflow_x, YW + 0.04f - 0.1f * i / 5.0f, outflow_z);
        o[6] = v3(outflow_x - outflow_dir * 0.012f, YW - 0.075f, outflow_z - 0.005f);
        o[7] = v3(outflow_x - outflow_dir * 0.035f, YW - 0.078f, outflow_z - 0.012f);
        for (int i = 0; i < 8; i++) o[i].x = mixf(o[i].x, o[i].x, 1);
        build_tube(m, o, 8, 0.0065f, M_PIPE, 12);
        float iy = sand_h(intake_x, outflow_z) + 0.05f;
        V3 in[2] = {v3(intake_x, YW + 0.04f, outflow_z + 0.01f), v3(intake_x, iy, outflow_z + 0.01f)};
        build_tube(m, in, 2, 0.0065f, M_PIPE, 12);
        V3 str[2] = {v3(intake_x, iy + 0.04f, outflow_z + 0.01f), v3(intake_x, iy - 0.003f, outflow_z + 0.01f)};
        build_tube(m, str, 2, 0.011f, M_PIPE, 14);
    }
    /* the ground map: the sand, raised by every stone and branch */
    gnx = (int)(w / GCELL) + 2;
    gnz = (int)(TD / GCELL) + 2;
    free(ground);
    ground = malloc(sizeof(float) * (size_t)gnx * gnz);
    for (int iz = 0; iz < gnz; iz++)
        for (int ix = 0; ix < gnx; ix++) ground[iz * gnx + ix] = sand_h(X0 + ix * GCELL, iz * GCELL);
    for (int k = 0; k + 2 < m->ni; k += 3) {
        float mat = m->v[(size_t)m->ix[k] * SV + 8];
        if (mat != M_STONE && mat != M_WOOD) continue;
        ground_tri(m_pos(m, (int)m->ix[k]), m_pos(m, (int)m->ix[k + 1]), m_pos(m, (int)m->ix[k + 2]));
    }
    scene_ni_opaque = m->ni;
    /* the substrate seen through the front glass, the back, the frame */
    m_grid(m, nu, 6, g_front, NULL, M_SUBSTRATE, 1);
    m_grid(m, nu / 2, 24, g_back, NULL, M_BACK, 1);
    float bot = -0.6f, top = YW + 0.6f;
    quad_front(m, X0 - 3, bot, X1 + 3, 0.0f, -0.0005f, M_CABINET);
    quad_front(m, X0 - 3, -0.016f, X1 + 3, 0.0005f, -0.001f, M_FRAME);
    quad_front(m, X0 - 3, YW + 0.006f, X1 + 3, top, -0.001f, M_FRAME);
    quad_front(m, X0 - 3, YW - 0.0012f, X1 + 3, YW + 0.006f, -0.0008f, M_MENISCUS);
    nscene_stone_i0 = m->ni;
    /* the mirrors: the underside of the surface, and the side panes */
    m_grid(m, nu / 2, 24, g_surf, NULL, M_SURF, 0);
    float sx0 = X0, sx1 = X1;
    m_grid(m, 24, 12, g_side, &sx0, M_SIDE, 0);
    m_grid(m, 24, 12, g_side, &sx1, M_SIDE, 1);
    nscene_stone_i1 = m->ni;
}

/* ==================================================================== */
/* ---- shaders ----
 * Everything that never moves (the tank, sand, stones, wood, the carpet,
 * moss) is drawn once, supersampled, into a G-buffer of light-independent
 * terms; each frame a fullscreen pass relights it with the moving
 * caustics and shadows. Only the swaying plants and the animals are drawn
 * every frame, lit the same way directly. uPass: 0 a frame, 1 the shadow
 * map, 2 the bake. */
#define GLSL(...) #__VA_ARGS__

static const char *COMMON = GLSL(
    uniform vec3 uCam;
    uniform vec4 uProj;
    uniform float uYW;
    uniform float uTD;
    uniform vec2 uXR;
    uniform float uT;
    uniform int uPass;
    uniform mat4 uShadowM;
    uniform vec3 uL;
    uniform vec3 uSun;
    uniform vec3 uAmbTop;
    uniform vec3 uFog;
    uniform vec3 uExt;
    uniform vec2 uCausOff;
    uniform float uNight;
    uniform float uBakeW;
    uniform sampler2D uCaus;
    uniform sampler2DShadow uShadow;
    uniform sampler2D uGround;
    uniform vec4 uGroundR;
    const float NW = 1.333;
    const float CTILE = 0.26;

    float wt(float hz) { return 6.2831853 * floor(hz * 1024.0 + 0.5) / 1024.0 * uT; }
    float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    float hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash12(i), hash12(i + vec2(1, 0)), f.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), f.x), f.y);
    }
    float vnoise3(vec3 p) {
        vec3 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y);
        float b = mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y);
        return mix(a, b, f.z);
    }
    float fbm2(vec2 p) { return 0.5 * vnoise(p) + 0.25 * vnoise(p * 2.03 + 1.7) + 0.125 * vnoise(p * 4.01 - 3.1) + 0.0625 * vnoise(p * 8.05 + 5.3); }
    float fbm3(vec3 p) { return 0.5 * vnoise3(p) + 0.25 * vnoise3(p * 2.03 + 1.7) + 0.125 * vnoise3(p * 4.01 - 3.1); }

    /* refraction through the front glass: the apparent tangent (x, y) of a
     * point P as the camera sees it, and the point G where that ray crosses
     * the glass. Snell's law in the plane of the camera axis and P, solved
     * with Newton from the paraxial guess. */
    vec2 refr_tan(vec3 P, out vec3 G) {
        vec2 d = P.xy - uCam.xy;
        float D = uCam.z;
        if (P.z <= 0.0) {
            float w = max(D + P.z, 1e-3);
            G = vec3(uCam.xy + d * D / w, 0.0);
            return d / w;
        }
        float R = length(d);
        if (R < 1e-6) { G = vec3(uCam.xy, 0.0); return vec2(0.0); }
        float z = P.z;
        float r = R * D / (D + z / NW);
        for (int i = 0; i < 3; i++) {
            float a = sqrt(r * r + D * D), b = sqrt((R - r) * (R - r) + z * z);
            float g = r / a - NW * (R - r) / b;
            float dg = D * D / (a * a * a) + NW * z * z / (b * b * b);
            r = clamp(r - g / dg, 0.0, R);
        }
        vec2 dir = d / R;
        G = vec3(uCam.xy + dir * r, 0.0);
        return dir * r / D;
    }
    vec4 place(vec3 P, out vec3 G) {
        if (uPass == 1) { G = P; return uShadowM * vec4(P, 1.0); }
        vec2 t = refr_tan(P, G);
        float w = max(uCam.z + (P.z > 0.0 ? P.z / NW : P.z), 0.005);
        vec2 ndc = t * uProj.xy + uProj.zw;
        float zn = (w - 0.004) / 4.0 * 2.0 - 1.0;
        return vec4(ndc * w, zn * w, w);
    }

    /* the lamp's light, focused by the ripples: the caustics texture
     * projected down the light from where the ray crossed the surface */
    vec3 caustic(vec3 P) {
        float depth = max(uYW - P.y, 0.0);
        vec2 S = P.xz + uL.xz / uL.y * depth;
        vec2 uv = S / CTILE + uCausOff;
        vec2 a = texture(uCaus, uv).rg, b = texture(uCaus, uv * 1.013 + 0.002).rg;
        float k = smoothstep(0.015, 0.22, depth);
        float ca = mix(a.r, a.g, k), cb = mix(b.r, b.g, k);
        return vec3(ca, mix(ca, cb, 0.5), cb);
    }
    float shadow(vec3 P) {
        vec4 s = uShadowM * vec4(P - uL * 0.003, 1.0);
        s.xyz = s.xyz * 0.5 + 0.5;
        vec2 px = vec2(1.2) / vec2(textureSize(uShadow, 0));
        return 0.5 * (texture(uShadow, vec3(s.xy + px * vec2(0.7, -0.4), s.z)) + texture(uShadow, vec3(s.xy - px * vec2(0.7, -0.4), s.z)));
    }
    /* how buried a point on the bottom is among the stones and wood */
    float ground_occ(vec3 P) {
        vec2 g = (P.xz - uGroundR.xy) * uGroundR.zw;
        float o = 0.0;
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.785 + 0.3;
            float r = (i % 2 == 0) ? 0.006 : 0.014;
            o += max(texture(uGround, g + vec2(cos(a), sin(a)) * r * uGroundR.zw).r - P.y, 0.0) * (i % 2 == 0 ? 1.0 : 0.5);
        }
        return exp(-o * 80.0);
    }
    /* the water's own glow, brighter facing up and higher in the tank */
    float amb_s(vec3 P, vec3 N) {
        float h = clamp(P.y / uYW, 0.0, 1.0);
        return mix(0.28, 1.0, clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * mix(0.55, 1.05, h);
    }
    float fog_s(float y) { return mix(0.7, 1.2, clamp(y / uYW, 0.0, 1.0)); }
    vec3 water(vec3 col, vec3 P, vec3 G) {
        vec3 tr = exp(-uExt * length(P - G));
        return col * tr + uFog * fog_s(P.y) * (1.0 - tr);
    }
);

/* the fragment side: the outputs, and how a surface is lit or baked */
static const char *COMMON_FS = GLSL(
    /* a surface's light, split into what the lamp gives it (times the
     * lamp's colour, its caustics and shadow) and what the water's glow
     * gives it; baked, or lit now */
    layout(location = 0) out vec4 o0;
    layout(location = 1) out vec4 o1;
    layout(location = 2) out vec4 o2;
    void emit(vec3 P, vec3 G, vec3 sunp, float spec, vec3 amb, float mark, bool outside) {
        if (uPass == 2) {
            float d = outside ? 0.0 : length(P - G);
            o0 = vec4(sunp, spec) * uBakeW;
            o1 = vec4(amb, fog_s(P.y) + 4.0 * mark) * uBakeW;
            o2 = vec4(P.x, outside ? uYW + 1.0 : P.y, P.z, d) * uBakeW;
            return;
        }
        vec3 col;
        if (outside) col = sunp * uSun + amb * uAmbTop;
        else {
            vec3 trd = exp(-uExt * max(uYW - P.y, 0.0));
            col = (sunp + spec) * uSun * trd * caustic(P) * shadow(P) + amb * uAmbTop;
            col = water(col, P, G);
        }
        o0 = vec4(col, 1.0);
        o1 = vec4(outside ? -1.0 : P.z, mark, 0.0, 1.0);
    }
    /* the usual split for a surface of albedo alb */
    void emit_lit(vec3 P, vec3 G, vec3 alb, vec3 N, float spec, float rough, float occ, vec3 transmit) {
        vec3 V = normalize(G - P);
        float ndl = max(dot(N, -uL), 0.0);
        vec3 H = normalize(V - uL);
        float sp = pow(max(dot(N, H), 0.0), rough) * spec * (rough + 8.0) / 8.0 * ndl;
        emit(P, G, alb * (ndl + transmit) * occ, sp * occ, alb * amb_s(P, N) * occ, 0.0, false);
    }
);

/* ---- the scene's solids: the tank, sand, stones, wood, gear, eel, shrimp ---- */
static const char *VS_SOLID = GLSL(
    layout(location = 0) in vec3 aP;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aUV;
    layout(location = 3) in float aMat;
    layout(location = 4) in vec3 aAux;
    out vec3 vP; out vec3 vN; out vec3 vG; out vec2 vUV; out vec3 vAux;
    flat out float vMat;
    void main() {
        vP = aP; vN = aN; vUV = aUV; vAux = aAux; vMat = aMat;
        gl_Position = place(aP, vG);
    });

static const char *FS_SOLID = GLSL(
    in vec3 vP; in vec3 vN; in vec3 vG; in vec2 vUV; in vec3 vAux;
    flat in float vMat;
    uniform float uSub;                 /* the bake's subsample, for dithered coverage */
    void creature(int mat, inout vec3 alb, inout vec3 n, inout float spec, inout float rough, inout vec3 tr, inout float alpha);
    void main() {
        int mat = int(vMat + 0.5);
        if (uPass == 1) return;
        vec3 N = normalize(vN);
        if (!gl_FrontFacing) N = -N;
        vec3 V = normalize(vG - vP);
        vec3 alb = vec3(0.5), n = N, tr = vec3(0.0);
        float spec = 0.05, rough = 20.0, occ = 1.0, alpha = 1.0;
        if (mat == 0) {
            /* sand: fine grains, ripples, darker detritus in the troughs */
            vec2 q = vP.xz;
            float warp = fbm2(q * 18.0);
            float ph = dot(q, vec2(0.93, 0.37)) * 6.2831 / 0.021 + warp * 5.0;
            float rip = sin(ph);
            float ramp = smoothstep(0.25, 0.75, fbm2(q * 6.0 + 4.0)) * 0.9 + 0.1;
            vec2 grad = vec2(0.93, 0.37) * cos(ph) * 0.35 * ramp;
            float fw = length(fwidth(q)) * 900.0;
            float g1 = vnoise(q * 900.0), g2 = vnoise(q * 2600.0 + 7.0);
            float gf = 1.0 - smoothstep(0.4, 1.2, fw);
            float gf2 = 1.0 - smoothstep(0.15, 0.45, fw);
            alb = vec3(0.74, 0.65, 0.50);
            alb *= 0.9 + 0.2 * fbm2(q * 40.0);
            alb = mix(alb, alb * vec3(0.55, 0.5, 0.45), smoothstep(0.62, 0.8, g1) * gf * 0.8);
            alb = mix(alb, vec3(0.92, 0.9, 0.84), smoothstep(0.7, 0.85, g2) * gf2 * 0.6);
            alb = mix(alb, vec3(0.3, 0.24, 0.16), smoothstep(0.6, 1.0, -rip) * ramp * 0.35);
            alb = mix(alb, vec3(0.26, 0.22, 0.13), smoothstep(0.55, 0.8, fbm2(q * 9.0 + 11.0)) * 0.5);
            n = normalize(N + vec3(grad.x, 0.0, grad.y) + vec3(g1 - 0.5, 0.0, g2 - 0.5) * 0.25 * gf);
            occ = mix(0.3, 1.0, ground_occ(vP));
        } else if (mat == 1) {
            /* the substrate against the front glass: dark soil under sand, in shade */
            vec2 q = vP.xy;
            float top = vUV.y;
            float soil = 1.0 - smoothstep(0.45, 0.6, top + 0.08 * fbm2(q * 60.0));
            vec2 c = q * 350.0;
            vec2 ci = floor(c), cf = fract(c) - 0.5 - 0.3 * (vec2(hash12(ci), hash12(ci + 3.1)) - 0.5);
            float ball = 1.0 - smoothstep(0.25, 0.45, length(cf));
            vec3 sandc = vec3(0.62, 0.54, 0.41) * (0.75 + 0.5 * vnoise(q * 1500.0));
            vec3 soilc = vec3(0.07, 0.05, 0.035) * (0.6 + 0.8 * ball) + vec3(0.04) * pow(ball, 6.0);
            vec3 a = mix(sandc, soilc, soil);
            float rt = abs(sin(q.x * 90.0 + 6.0 * fbm2(q * 25.0)));
            a = mix(a, vec3(0.75, 0.7, 0.55), (1.0 - smoothstep(0.0, 0.06, rt)) * smoothstep(0.2, 0.7, top) * step(0.6, vnoise(vec2(q.x * 20.0, 1.0))) * 0.6);
            float depthf = mix(0.2, 0.5, top);
            emit(vP, vG, a * 0.04 * depthf, 0.0, a * 0.8 * depthf, 0.0, true);
            return;
        } else if (mat == 2) {
            /* the back: black foil behind the glass */
            emit(vP, vG, vec3(0.0005), 0.0, vec3(0.012, 0.035, 0.035) * (0.3 + 0.7 * vP.y / uYW), 0.0, false);
            return;
        } else if (mat == 3 || mat == 4) {
            /* mirrors, finished in the composite */
            emit(vP, vG, vec3(0.0), 0.0, vec3(0.0), mat == 4 ? 1.0 : -1.0, false);
            return;
        } else if (mat == 5) {
            /* the black trim round the glass */
            float y = vP.y;
            float edge = y > uYW ? 1.0 - smoothstep(0.0, 0.01, y - uYW - 0.006) : smoothstep(-0.006, 0.0, y);
            emit(vP, vG, vec3(0.004 + 0.008 * edge), 0.0, vec3(0.04 + 0.1 * edge), 0.0, true);
            return;
        } else if (mat == 12) {
            /* the meniscus: a bright line where the water climbs the glass */
            float y = (vP.y - uYW) / 0.0072 + 0.17;
            float wob = 0.05 * sin(vP.x * 60.0) + 0.03 * sin(vP.x * 173.0);
            float band = exp(-pow((y - 0.35 - wob) * 6.0, 2.0)) + 0.4 * exp(-pow((y - 0.75) * 9.0, 2.0));
            emit(vP, vG, vec3(0.1) * band + 0.002, 0.0, vec3(0.6) * band + 0.03, 0.0, true);
            return;
        } else if (mat == 13) {
            /* the cabinet under the tank: dark walnut catching the tank's glow */
            vec2 q = vP.xy;
            float grain = fbm2(vec2(q.x * 6.0, q.y * 90.0 + 4.0 * fbm2(q * 7.0)));
            vec3 a = mix(vec3(0.05, 0.028, 0.016), vec3(0.11, 0.06, 0.03), grain);
            float glow = exp(q.y * 9.0);
            emit(vP, vG, a * (0.03 * glow + 0.006), 0.0, a * (0.4 * glow + 0.05), 0.0, true);
            return;
        } else if (mat == 6) {
            /* driftwood: grooved bark, worn smooth on top, a skin of biofilm */
            float along = vUV.x, around = vUV.y * 6.2831;
            float s = vAux.y;
            float g = fbm2(vec2(along * 55.0 + s, around * 1.6 + s * 3.0));
            float g2 = vnoise(vec2(along * 200.0, around * 4.0 + s));
            alb = mix(vec3(0.20, 0.12, 0.065), vec3(0.36, 0.25, 0.15), g);
            alb *= 0.75 + 0.35 * g2;
            alb = mix(alb, vec3(0.13, 0.075, 0.04), smoothstep(0.55, 0.75, g2) * 0.6);
            alb = mix(alb, vec3(0.22, 0.24, 0.10), smoothstep(0.3, 0.9, N.y) * 0.35 * smoothstep(0.4, 0.7, fbm2(vP.xz * 70.0)));
            n = normalize(N + 0.25 * vec3(g2 - 0.5) * vec3(1.0, 0.3, 1.0));
            spec = 0.08;
            occ = mix(0.55, 1.0, ground_occ(vP + vec3(0.0, 0.004, 0.0)));
        } else if (mat == 7) {
            /* seiryu stone: blue-grey, white calcite veins, algae on top */
            vec3 p = vAux;
            float n1 = fbm3(p * 2.0);
            float n2 = vnoise3(p * 14.0);
            alb = mix(vec3(0.24, 0.26, 0.28), vec3(0.46, 0.47, 0.48), n1);
            alb *= 0.8 + 0.35 * n2;
            float vein = 1.0 - smoothstep(0.0, 0.035, abs(fbm3(p * 1.3 + 7.0) - 0.5));
            alb = mix(alb, vec3(0.7, 0.7, 0.66), vein * 0.3 * smoothstep(0.4, 0.7, vnoise3(p * 3.0 + 2.0)));
            float crack = 1.0 - smoothstep(0.0, 0.05, abs(vnoise3(p * 5.0) - 0.5));
            alb *= 1.0 - 0.45 * crack;
            alb = mix(alb, vec3(0.20, 0.28, 0.10), smoothstep(0.55, 0.95, N.y) * smoothstep(0.45, 0.7, fbm3(p * 4.0 + 3.0)) * 0.6);
            n = normalize(N + 0.35 * (vec3(vnoise3(p * 9.0), vnoise3(p * 9.0 + 5.0), vnoise3(p * 9.0 + 9.0)) - 0.5));
            spec = 0.12; rough = 24.0;
            occ = mix(0.5, 1.0, smoothstep(-0.003, 0.01, vP.y - texture(uGround, (vP.xz - uGroundR.xy) * uGroundR.zw).r + 0.01));
        } else if (mat == 14) {
            /* clear acrylic pipes and airline: mostly the glints on them */
            float f = pow(1.0 - abs(dot(N, V)), 3.0);
            float streak = smoothstep(0.75, 0.95, abs(dot(N, normalize(vec3(-0.6, 0.0, -1.0)))));
            alpha = clamp(0.2 + f * 0.6 + streak * 0.5, 0.0, 1.0);
            if (uPass == 2 && hash12(gl_FragCoord.xy * 1.37 + uSub * 17.3) > alpha) discard;
            emit(vP, vG, vec3(0.08) * (f + streak), 0.0, vec3(0.08) + vec3(1.2) * (f * 0.6 + streak * 0.5), 0.0, false);
            return;
        } else if (mat == 15) {
            /* the airstone: grey and porous */
            float pore = vnoise3(vP * 900.0);
            alb = vec3(0.42, 0.42, 0.40) * (0.7 + 0.4 * pore);
            spec = 0.02;
        } else {
            creature(mat, alb, n, spec, rough, tr, alpha);
        }
        emit_lit(vP, vG, alb, n, spec, rough, occ, tr);
    });

/* ---- plants: instanced leaf strips, swaying ---- */
static const char *VS_LEAF = GLSL(
    layout(location = 0) in vec2 aUV;
    layout(location = 1) in vec4 iA;
    layout(location = 2) in vec4 iB;
    layout(location = 3) in vec4 iC;
    layout(location = 4) in vec4 iD;
    layout(location = 5) in vec4 iE;
    out vec3 vP; out vec3 vN; out vec3 vG; out vec2 vUV;
    flat out vec4 vInfo;
    uniform vec2 uCurrent;
    uniform float uSway;
    float leaf_w(int type, float v) {
        if (type == 0) return pow(sin(3.1416 * min(v, 0.999)), 0.5) * (1.0 - 0.3 * v);
        if (type == 1 || type == 8) return pow(max(sin(3.1416 * pow(v, 1.25)), 0.0), 0.75) * smoothstep(0.0, 0.2, v);
        if (type == 3) return v < 0.28 ? 0.06 : pow(max(sin(3.1416 * (v - 0.28) / 0.72), 0.0), 0.8);
        if (type == 2 || type == 9) return pow(max(sin(3.1416 * pow(v, 0.8)), 0.0), 0.9);
        if (type == 4) return 1.0 - smoothstep(0.93, 1.0, v) * 0.7;
        if (type == 5) return 1.0 - 0.8 * v;
        if (type == 6) return sqrt(max(1.0 - pow(2.0 * v - 1.0, 2.0), 0.0));
        if (type == 11) return pow(max(sin(3.1416 * pow(v, 0.85)), 0.0), 0.7);
        return 1.0;
    }
    vec3 sway(vec3 P, float rootY, float H, float flex, float seed) {
        float f = clamp((P.y - rootY) / max(H, 0.02), 0.0, 1.4);
        float k = flex * f * f;
        float ph = P.x * 5.0 + P.z * 3.0;
        vec2 osc = vec2(sin(wt(0.19) - ph + seed * 0.7) + 0.45 * sin(wt(0.43) - ph * 1.7 + seed * 3.0),
                        0.6 * sin(wt(0.15) - ph * 0.8 + seed * 2.0) + 0.3 * sin(wt(0.37) + seed * 5.0));
        vec2 d = (uCurrent + uSway * osc) * k;
        float hh = max(P.y - rootY, 0.01);
        return vec3(d.x, -0.5 * dot(d, d) / hh, d.y);
    }
    void main() {
        int type = int(iA.w + 0.5);
        vec3 dir = iB.xyz, side = iC.xyz;
        float len = iB.w, wid = iC.w;
        float curl = iE.x, fold = iE.y;
        vec3 nrm = normalize(cross(side, dir));
        if (nrm.y < 0.0 && type != 10) nrm = -nrm;
        float v = aUV.y;
        float ang = curl * v;
        vec3 c;
        if (abs(curl) < 1e-3) c = dir * v;
        else c = dir * (sin(ang) / curl) + nrm * ((1.0 - cos(ang)) / curl);
        vec3 tng = dir * cos(ang) + nrm * sin(ang);
        vec3 n = nrm * cos(ang) - dir * sin(ang);
        vec3 P0 = iA.xyz + c * len;
        vec3 sd = side;
        if (type == 10 || type == 4) {
            /* stems face the camera; Vallisneria ribbons twist as they rise */
            vec3 toCam = normalize(vec3(uCam.xy, -uCam.z) - P0);
            if (type == 10) sd = normalize(cross(tng, toCam));
            else sd = normalize(mix(side, normalize(cross(tng, vec3(0.0, 0.0, 1.0))), 0.4 + 0.4 * sin(v * 6.0 + iD.w * 9.0)));
            n = normalize(cross(sd, tng));
        }
        float w = wid * leaf_w(type, v);
        vec3 P = P0 + sd * aUV.x * w + n * fold * abs(aUV.x) * w * 0.35;
        vec3 N = normalize(n - sd * aUV.x * fold * 0.5);
        P += sway(P, iD.x, iD.y, iD.z, iD.w);
        if (type == 4 || type == 10) P.y = min(P.y, uYW - 0.004 - 0.002 * aUV.x);
        P.x = clamp(P.x, uXR.x + 0.002, uXR.y - 0.002);
        P.z = clamp(P.z, 0.004, uTD - 0.003);
        vP = P; vN = N; vUV = aUV;
        vInfo = vec4(float(type), iE.z, iE.w, iD.w);
        gl_Position = place(P, vG);
    });

static const char *FS_LEAF = GLSL(
    in vec3 vP; in vec3 vN; in vec3 vG; in vec2 vUV;
    flat in vec4 vInfo;
    uniform float uSub;
    void main() {
        int type = int(vInfo.x + 0.5);
        float hue = vInfo.y, hf = vInfo.z, seed = vInfo.w;
        float u = vUV.x, v = vUV.y;
        /* coverage: soft edges; moss is feathery */
        float a = 1.0 - smoothstep(0.75, 1.0, abs(u));
        if (type == 7) {
            float fr = abs(fract(u * 2.5 + v * 3.0 + seed * 7.0) - 0.5) * 2.0;
            float fr2 = abs(fract(-u * 2.5 + v * 3.0 + seed * 3.0) - 0.5) * 2.0;
            float strand = max(1.0 - smoothstep(0.1, 0.35, abs(u) * 0.2 + fr * 0.3), 1.0 - smoothstep(0.12, 0.3, fr2 * 0.4));
            a *= strand * (1.0 - smoothstep(0.7, 1.0, length(vec2(u, v * 2.0 - 1.0))));
        }
        if (a < 0.5) discard;
        if (uPass == 1) return;
        vec3 N = normalize(vN);
        if (!gl_FrontFacing) N = -N;
        vec3 alb; float spec = 0.15, rough = 24.0, trans = 0.5;
        float var = hue * 0.08;
        if (type == 0) {
            alb = mix(vec3(0.22, 0.45, 0.1), vec3(0.78, 0.36, 0.30), smoothstep(0.55, 1.0, hf));
            alb *= 1.0 + var;
        } else if (type == 1) {
            alb = mix(vec3(0.32, 0.38, 0.12), vec3(0.55, 0.16, 0.10), smoothstep(0.3, 0.95, hf));
            if (!gl_FrontFacing) alb = vec3(0.5, 0.12, 0.1);
            spec = 0.25;
        } else if (type == 2) {
            alb = vec3(0.34, 0.58, 0.14) * (1.0 + var) * mix(0.85, 1.1, hf);
        } else if (type == 3) {
            alb = vec3(0.11, 0.32, 0.07) * (1.0 + var);
            trans = 0.3;
            float rib = 1.0 - smoothstep(0.0, 0.06, abs(u));
            float vein = 1.0 - smoothstep(0.0, 0.08, abs(abs(u) - 0.55 * sin(3.1416 * clamp((v - 0.28) / 0.72, 0.0, 1.0))));
            alb = mix(alb, vec3(0.45, 0.62, 0.28), max(rib, vein * 0.5) * step(0.28, v));
            if (v < 0.28) alb = vec3(0.3, 0.45, 0.18);
            spec = 0.25;
        } else if (type == 4) {
            alb = vec3(0.26, 0.48, 0.12) * (1.0 + var) * (0.9 + 0.1 * sin(u * 30.0));
            trans = 0.5;
        } else if (type == 5) {
            alb = vec3(0.22, 0.50, 0.10) * (1.0 + var) * mix(0.7, 1.1, v);
        } else if (type == 6) {
            alb = vec3(0.30, 0.62, 0.12) * (1.0 + var);
            spec = 0.3;
        } else if (type == 7) {
            alb = vec3(0.14, 0.30, 0.07) * (1.0 + var) * mix(0.7, 1.2, v);
            trans = 0.3;
        } else if (type == 8) {
            alb = vec3(0.05, 0.16, 0.04) * (1.0 + var);
            float vein = 1.0 - smoothstep(0.0, 0.05, abs(u));
            alb = mix(alb, vec3(0.12, 0.25, 0.08), vein);
            spec = 0.9; rough = 60.0; trans = 0.1;
        } else if (type == 9) {
            alb = vec3(0.09, 0.24, 0.05) * (1.0 + var);
            float vein = 1.0 - smoothstep(0.0, 0.04, abs(abs(u) - fract(v * 9.0) * 0.8));
            alb *= 1.0 - 0.3 * vein;
            spec = 0.35; trans = 0.25;
        } else if (type == 10) {
            alb = mix(vec3(0.28, 0.42, 0.14), vec3(0.55, 0.25, 0.18), clamp(hue * 0.5 + 0.5, 0.0, 1.0) * 0.6);
            trans = 0.2;
        } else {
            alb = mix(vec3(0.30, 0.16, 0.06), vec3(0.18, 0.09, 0.04), vnoise(vUV * vec2(6.0, 9.0)));
            float vein = 1.0 - smoothstep(0.0, 0.05, abs(u));
            alb = mix(alb, vec3(0.42, 0.28, 0.14), vein * 0.6);
            trans = 0.0; spec = 0.05;
        }
        float bk = max(dot(-N, -uL), 0.0);
        emit_lit(vP, vG, alb, N, spec, rough, 1.0, bk * trans * vec3(0.8, 1.0, 0.6));
    });

/* ---- the caustics texture: the lamp's light through the rippled surface,
 * splatted from a mesh of the surface displaced by refraction ---- */
static const char *VS_CAUS = GLSL(
    layout(location = 0) in vec2 aP;
    uniform float uPh[8];
    uniform vec4 uK[8];
    uniform float uD;
    out vec2 vSrc;
    void main() {
        vec2 p = aP;
        vec2 d = vec2(0.0);
        for (int i = 0; i < 8; i++) d -= uK[i].xy * uK[i].z * sin(dot(uK[i].xy, p) * 6.2831853 / 256.0 + uPh[i]);
        int ix = gl_InstanceID % 3 - 1, iy = gl_InstanceID / 3 - 1;
        vec2 q = p + d * uD + vec2(float(ix), float(iy)) * 256.0;
        vSrc = p;
        gl_Position = vec4(q / 128.0 - 1.0, 0.0, 1.0);
    });
static const char *FS_CAUS = GLSL(
    in vec2 vSrc;
    uniform vec4 uMask;
    void main() {
        vec2 a = dFdx(vSrc), b = dFdy(vSrc);
        float I = abs(a.x * b.y - a.y * b.x);
        o0 = uMask * min(I, 12.0);
    });

/* ---- fullscreen passes ---- */
static const char *VS_FULL = GLSL(
    out vec2 vUV;
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        vUV = p;
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the bake: accumulate one subsample of the G-buffer */
static const char *FS_ACCUM = GLSL(
    in vec2 vUV;
    uniform sampler2D uG0, uG1, uG2;
    uniform float uW;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        o0 = texelFetch(uG0, p, 0) * uW;
        o1 = texelFetch(uG1, p, 0) * uW;
        o2 = texelFetch(uG2, p, 0) * uW;
    });

/* each frame: light the baked G-buffer */
static const char *FS_RELIGHT = GLSL(
    in vec2 vUV;
    uniform int uDbg;
    uniform sampler2D uG0, uG1, uG2;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        vec4 g0 = texelFetch(uG0, p, 0), g1 = texelFetch(uG1, p, 0), g2 = texelFetch(uG2, p, 0);
        vec3 P = g2.xyz;
        float mark = floor(g1.a / 4.0 + 0.5), fs = g1.a - 4.0 * mark;
        vec3 col;
        if (P.y > uYW + 0.3) {
            col = g0.rgb * uSun + g1.rgb * uAmbTop;
            o1 = vec4(-1.0, 0.0, 0.0, 1.0);
        } else {
            vec3 trd = exp(-uExt * max(uYW - P.y, 0.0));
            col = (g0.rgb + g0.a) * uSun * trd * caustic(P) * shadow(P) + g1.rgb * uAmbTop;
            vec3 tr = exp(-uExt * g2.w);
            col = col * tr + uFog * fs * (1.0 - tr);
            o1 = vec4(P.z, mark, 0.0, 1.0);
            if (uDbg == 1) col = vec3(shadow(P));
            if (uDbg == 2) col = caustic(P) * 0.3;
            if (uDbg == 3) col = fract(P * 10.0);
        }
        o0 = vec4(col, 1.0);
    });

/* temporal antialiasing: the camera never moves, so the history only
 * needs clamping where things do */
static const char *FS_TAA = GLSL(
    in vec2 vUV;
    uniform sampler2D uCur, uHist;
    uniform float uAlpha;
    uniform int uClamp;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        vec3 c = texelFetch(uCur, p, 0).rgb;
        vec3 h = texelFetch(uHist, p, 0).rgb;
        if (uClamp == 1) {
            vec3 mn = c, mx = c;
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++) {
                    if (i == 0 && j == 0) continue;
                    vec3 s = texelFetch(uCur, p + ivec2(i, j), 0).rgb;
                    mn = min(mn, s); mx = max(mx, s);
                }
            vec3 ext = (mx - mn) * 0.25;
            h = clamp(h, mn - ext, mx + ext);
        }
        o0 = vec4(mix(h, c, uAlpha), 1.0);
    });

/* the view ray through a pixel: from the glass point G into the water */
static const char *VIEWRAY = GLSL(
    uniform vec4 uView;      /* this output's canvas rect */
    uniform vec3 uCanCam;    /* cam_cx, cam_cy, cam_f */
    vec2 canvas_of(vec2 uv) { return vec2(uView.x + uv.x * uView.z, uView.y + (1.0 - uv.y) * uView.w); }
    vec2 tan_of(vec2 cv) { return vec2(cv.x - uCanCam.x, uCanCam.y - cv.y) / uCanCam.z; }
    void view_ray(vec2 uv, out vec3 G, out vec3 dW) {
        vec2 t = tan_of(canvas_of(uv));
        G = vec3(uCam.xy + t * uCam.z, 0.0);
        vec3 dA = normalize(vec3(t, 1.0));
        float eta = 1.0 / NW;
        vec2 xy = eta * dA.xy;
        dW = vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
    }
);

static const char *FS_SHAFTS = GLSL(
    in vec2 vUV;
    uniform sampler2D uAux;
    uniform float uShaftK;
    void main() {
        vec3 G, d;
        view_ray(vUV, G, d);
        float zEnd = texture(uAux, vUV).r;
        if (zEnd <= 0.0) zEnd = uTD;
        zEnd = min(zEnd, uTD);
        float sEnd = zEnd / d.z;
        if (d.y > 0.0) sEnd = min(sEnd, (uYW - G.y) / d.y);
        const int N = 10;
        float ds = sEnd / float(N);
        float j = hash12(gl_FragCoord.xy + fract(uT * 7.13) * 91.0);
        vec3 acc = vec3(0.0);
        for (int i = 0; i < N; i++) {
            float s = (float(i) + j) * ds;
            vec3 P = G + d * s;
            float depth = max(uYW - P.y, 0.0);
            vec2 S = P.xz + uL.xz / uL.y * depth;
            float beam = textureLod(uCaus, S / CTILE + uCausOff, 4.5).g;
            beam = beam * beam;
            vec4 sc = uShadowM * vec4(P, 1.0);
            sc.xyz = sc.xyz * 0.5 + 0.5;
            float sh = texture(uShadow, sc.xyz);
            vec3 tr = exp(-uExt * (s + depth));
            acc += beam * sh * tr * ds;
        }
        o0 = vec4(acc * uSun * uShaftK, 1.0);
    });

static const char *FS_COMP = GLSL(
    in vec2 vUV;
    uniform sampler2D uCol;
    uniform sampler2D uAux;
    uniform sampler2D uShafts;
    uniform vec4 uLay;       /* the whole canvas rect */
    uniform float uExpo;
    uniform float uCA;
    /* where the surface (world y = uYW) and a side pane are on the canvas at depth z */
    float surf_y(float z) { return uCanCam.y - uCanCam.z * (uYW - uCam.y) / (uCam.z + z / NW); }
    float side_x(float xw, float z) { return uCanCam.x + uCanCam.z * (xw - uCam.x) / (uCam.z + z / NW); }
    vec2 uv_of(vec2 cv) { return vec2((cv.x - uView.x) / uView.z, 1.0 - (cv.y - uView.y) / uView.w); }
    /* total internal reflection: the tank seen again, mirrored about the
     * surface (or a side pane) at the depth of what it reflects */
    vec3 mirror(vec2 uv, float zs, float kind) {
        vec2 cv = canvas_of(uv);
        float rip = sin(cv.x * 0.02 / uCanCam.z * 2000.0 + wt(0.6)) * 0.6 + sin((cv.x * 0.047 + cv.y * 0.03) / uCanCam.z * 2000.0 - wt(0.9)) * 0.4;
        vec2 q = cv;
        float z = zs;
        for (int i = 0; i < 2; i++) {
            if (kind > 0.0) q.y = 2.0 * surf_y(z) - cv.y;
            else q.x = 2.0 * side_x(cv.x < uCanCam.x ? uXR.x : uXR.y, z) - cv.x;
            float zz = texture(uAux, clamp(uv_of(q), 0.001, 0.999)).r;
            if (zz > 0.0) z = max(zz, zs);
        }
        vec2 quv = uv_of(q) + (kind > 0.0 ? vec2(rip * 0.002, rip * 0.004) : vec2(0.0, rip * 0.001));
        vec3 c = texture(uCol, clamp(quv, 0.001, 0.999)).rgb;
        vec3 tr = exp(-uExt * max(z - zs, 0.0) * 0.7);
        c = c * tr + uFog * 1.1 * (1.0 - tr);
        return c * 0.9;
    }
    vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
    void main() {
        vec2 uv = vUV;
        vec4 aux = texture(uAux, uv);
        vec3 c = texture(uCol, uv).rgb;
        float mk = aux.g;
        vec2 cv = canvas_of(uv);
        if (abs(mk) > 0.02) {
            vec3 m = mirror(uv, max(aux.r, 0.0), mk);
            if (mk > 0.0) {
                /* glints on the underside where the ripples catch the lamp */
                vec2 s = cv / uCanCam.z * 2000.0;
                float g = vnoise(vec2(s.x * 0.03 + wt(0.2) * 3.0, s.y * 0.08)) * vnoise(vec2(s.x * 0.011 - wt(0.13) * 2.0, s.y * 0.05 + 3.0));
                m += uSun * 0.12 * pow(g, 3.0) + uAmbTop * 0.04;
            }
            c = mix(c, m, clamp(abs(mk), 0.0, 1.0));
        } else if (uCA > 0.0 && aux.r > 0.0) {
            /* water and glass disperse light more the further off axis */
            vec2 t = tan_of(cv);
            float k = dot(t, t);
            if (k > 0.02) {
                vec2 off = t * k * uCA * uCanCam.z / uView.zw * 0.0015;
                off.y = -off.y;
                c.r = texture(uCol, uv + off).r;
                c.b = texture(uCol, uv - off).b;
            }
        }
        if (aux.r > 0.0 || abs(mk) > 0.02) c += texture(uShafts, uv).rgb;
        c *= uExpo;
        /* a soft vignette over the whole wall */
        vec2 lv = (cv - uLay.xy) / uLay.zw - 0.5;
        c *= 1.0 - 0.3 * smoothstep(0.3, 0.9, length(lv * vec2(1.0, uLay.w / uLay.z * 0.7 + 0.3)));
        c = aces(c);
        c = pow(c, vec3(1.0 / 2.2));
        c += (hash12(gl_FragCoord.xy + fract(uT) * 37.0) - 0.5) / 255.0;
        o0 = vec4(c, 1.0);
    });

/* ==================================================================== */
/* ---- GL plumbing ---- */
static GLuint compile(GLenum type, const char *const *src, int n) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, n, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "aquarium: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

/* a program: the vertex body after the common code; the fragment body
 * after that and the fragment side's, between an optional prelude and tail */
static GLuint mk_prog(const char *vs, const char *fs_pre, const char *fs, const char *fs_tail) {
    const char *hdr = "#version 330 core\n";
    const char *v[3] = {hdr, COMMON, vs};
    const char *f[6] = {hdr, COMMON, COMMON_FS, fs_pre ? fs_pre : "", fs, fs_tail ? fs_tail : ""};
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, v, 3));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, f, 6));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "aquarium: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

/* ---- the light of the day ----
 * The tank's lamp follows the local clock: a warm dawn, full daylight,
 * a warm, dimmer evening, then blue moonlight through the night. */
typedef struct {
    float sun[3], amb_top[3], amb_bot[3], fog[3];
    float night, act, shaft;
} Lighting;
static Lighting lit;
static float ext_c[3] = {1.3f, 0.72f, 0.78f};    /* absorption of the water, 1/m */
static V3 light_dir;

static const struct { float h; Lighting l; } light_keys[] = {
    {0.0f, {{0.09f, 0.16f, 0.36f}, {0.012f, 0.026f, 0.06f}, {0.003f, 0.007f, 0.016f}, {0.004f, 0.011f, 0.026f}, 1.0f, 0.25f, 0.5f}},
    {6.5f, {{0.09f, 0.16f, 0.36f}, {0.012f, 0.026f, 0.06f}, {0.003f, 0.007f, 0.016f}, {0.004f, 0.011f, 0.026f}, 1.0f, 0.25f, 0.5f}},
    {7.5f, {{1.10f, 0.72f, 0.42f}, {0.10f, 0.10f, 0.10f}, {0.03f, 0.035f, 0.035f}, {0.035f, 0.05f, 0.05f}, 0.35f, 0.6f, 0.9f}},
    {9.0f, {{1.7f, 1.62f, 1.48f}, {0.30f, 0.42f, 0.42f}, {0.08f, 0.12f, 0.11f}, {0.04f, 0.14f, 0.13f}, 0.0f, 1.0f, 1.0f}},
    {17.0f, {{1.7f, 1.62f, 1.48f}, {0.30f, 0.42f, 0.42f}, {0.08f, 0.12f, 0.11f}, {0.04f, 0.14f, 0.13f}, 0.0f, 1.0f, 1.0f}},
    {19.0f, {{2.2f, 1.55f, 0.95f}, {0.22f, 0.24f, 0.26f}, {0.06f, 0.07f, 0.065f}, {0.06f, 0.09f, 0.095f}, 0.05f, 0.85f, 1.2f}},
    {21.5f, {{0.75f, 0.42f, 0.30f}, {0.06f, 0.07f, 0.09f}, {0.015f, 0.02f, 0.025f}, {0.016f, 0.025f, 0.035f}, 0.6f, 0.5f, 0.9f}},
    {22.5f, {{0.09f, 0.16f, 0.36f}, {0.012f, 0.026f, 0.06f}, {0.003f, 0.007f, 0.016f}, {0.004f, 0.011f, 0.026f}, 1.0f, 0.25f, 0.5f}},
    {24.0f, {{0.09f, 0.16f, 0.36f}, {0.012f, 0.026f, 0.06f}, {0.003f, 0.007f, 0.016f}, {0.004f, 0.011f, 0.026f}, 1.0f, 0.25f, 0.5f}},
};

static float local_hour(double wt) {
    if (opt_hour >= 0) return fmodf(opt_hour, 24);
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    return tm.tm_hour + tm.tm_min / 60.0f + (tm.tm_sec + (float)(wt - floor(wt))) / 3600.0f;
}

static void lighting(double wt) {
    float h = local_hour(wt);
    int n = (int)(sizeof light_keys / sizeof light_keys[0]);
    int i = 0;
    while (i < n - 2 && light_keys[i + 1].h <= h) i++;
    float f = smooth(light_keys[i].h, light_keys[i + 1].h, h);
    const float *a = (const float *)&light_keys[i].l, *b = (const float *)&light_keys[i + 1].l;
    float *o = (float *)&lit;
    for (int k = 0; k < (int)(sizeof lit / sizeof(float)); k++) o[k] = mixf(a[k], b[k], f);
    light_dir = vnorm(v3(0.24f, -1.0f, 0.16f));
}

/* the instantaneous CPU load, eased between the once-a-second readings */
static float cpu_load(double wt) {
    float f = clampf((float)(wt - sens.at), 0, 1);
    return mixf(sens.load_prev, sens.load, f * f * (3 - 2 * f));
}

/* ---- the state every scene shader shares ---- */
static float proj[4];
static float shadowM[16];
static double t_now;                   /* seconds since the epoch */
static GLuint tex_caus, tex_shadow, tex_ground;
static float caus_off[2];
static int cur_pass;

static void set_common(GLuint p) {
    glUseProgram(p);
    glUniform3f(U(p, "uCam"), CX, CY, DC);
    glUniform4fv(U(p, "uProj"), 1, proj);
    glUniform1f(U(p, "uYW"), YW);
    glUniform1f(U(p, "uTD"), TD);
    glUniform2f(U(p, "uXR"), X0, X1);
    glUniform1f(U(p, "uT"), (float)fmod(t_now, 1024.0));
    glUniform1i(U(p, "uPass"), cur_pass);
    glUniformMatrix4fv(U(p, "uShadowM"), 1, GL_FALSE, shadowM);
    glUniform3f(U(p, "uL"), light_dir.x, light_dir.y, light_dir.z);
    glUniform3fv(U(p, "uSun"), 1, lit.sun);
    glUniform3fv(U(p, "uAmbTop"), 1, lit.amb_top);
    glUniform3fv(U(p, "uFog"), 1, lit.fog);
    glUniform3fv(U(p, "uExt"), 1, ext_c);
    glUniform2fv(U(p, "uCausOff"), 1, caus_off);
    glUniform1f(U(p, "uNight"), lit.night);
    glUniform1i(U(p, "uCaus"), 0);
    glUniform1i(U(p, "uShadow"), 1);
    glUniform1i(U(p, "uGround"), 2);
    glUniform4f(U(p, "uGroundR"), X0, 0, 1.0f / (gnx * GCELL), 1.0f / (gnz * GCELL));
}

/* a VAO for the 12-float solid vertex */
static void solid_attribs(void) {
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, SV * 4, (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, SV * 4, (void *)12);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, SV * 4, (void *)24);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, SV * 4, (void *)32);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, SV * 4, (void *)36);
    for (int i = 0; i < 5; i++) glEnableVertexAttribArray((GLuint)i);
}

static GLuint p_solid;

/* ==================================================================== */
/* ---- creatures (stub) ---- */
static void creatures_init(void) {}
static void creatures_gl(void) {}
static void creatures_update(double t, double wt) { (void)t; (void)wt; }
static void creatures_draw(int pass) { (void)pass; }
static void creatures_draw_translucent(void) {}
static int creature_focus(const char *kind, V3 *at, float *size) { (void)kind; (void)at; (void)size; return 0; }
static const char *CREATURE_FS = GLSL(
    void creature(int mat, inout vec3 alb, inout vec3 n, inout float spec, inout float rough, inout vec3 tr, inout float alpha) { alb = vec3(1.0, 0.0, 1.0); });

/* ==================================================================== */
/* ---- overlays (stub) ---- */
static void overlay_gl(void) {}
static void overlay_draw(double wt) { (void)wt; }

/* ==================================================================== */
/* ---- GL objects ---- */
static GLuint p_leaf, p_caus, p_shafts, p_comp, p_accum, p_relight, p_taa;
static GLuint vao_scene, vbo_scene, ibo_scene;
static GLuint vao_leaf[2][LM_N], vbo_leafm[LM_N], ibo_leafm[LM_N], vbo_leafi[2][LM_N];
static int leaf_ni[LM_N];
static GLuint vao_caus, vbo_caus, ibo_caus;
static int caus_ni;
static GLuint fbo_caus, fbo_shadow, fbo_shadow_s, tex_shadow_s;
static int shadow_w = 2048, shadow_h = 1024;
static GLuint fbo_main, tex_cur, tex_aux, rb_depth;
static GLuint fbo_sdepth, rb_sdepth;
static GLuint fbo_bake, tex_bake[3], rb_bake;
static GLuint fbo_gbuf, tex_g[3];
static GLuint fbo_taa[2], tex_taa[2];
static int taa_cur, taa_frames;
static GLuint fbo_sh, tex_sh;
static GLuint vao_empty;
static int sw_, sh_;                    /* the scene's internal size */
static int fb_w, fb_h;                  /* what the targets were made for */
static float jit[2];                    /* this frame's subpixel jitter, in scene pixels */
#define CAUS_N 256

static void leaf_mesh(int m, int nu, int nv) {
    float *v = malloc(sizeof(float) * 2 * (size_t)(nu + 1) * (nv + 1));
    uint32_t *ix = malloc(sizeof(uint32_t) * 6 * (size_t)nu * nv);
    int k = 0;
    for (int j = 0; j <= nv; j++)
        for (int i = 0; i <= nu; i++) { v[k++] = (float)i / nu * 2 - 1; v[k++] = (float)j / nv; }
    int n = 0;
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            uint32_t a = (uint32_t)(j * (nu + 1) + i), b = a + 1, c = a + (uint32_t)nu + 1, d = c + 1;
            ix[n++] = a; ix[n++] = b; ix[n++] = c; ix[n++] = b; ix[n++] = d; ix[n++] = c;
        }
    glGenBuffers(1, &vbo_leafm[m]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_leafm[m]);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * 2 * (nu + 1) * (nv + 1)), v, GL_STATIC_DRAW);
    /* (an element buffer binds to the VAO bound now: make it inside the leaves' own) */
    glGenVertexArrays(1, &vao_leaf[0][m]);
    glBindVertexArray(vao_leaf[0][m]);
    glGenBuffers(1, &ibo_leafm[m]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_leafm[m]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * n), ix, GL_STATIC_DRAW);
    leaf_ni[m] = n;
    for (int s = 0; s < 2; s++) {
        Inst *L = s ? &leaves_s[m] : &leaves[m];
        if (s) glGenVertexArrays(1, &vao_leaf[s][m]);
        glBindVertexArray(vao_leaf[s][m]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_leafm[m]);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_leafm[m]);
        glGenBuffers(1, &vbo_leafi[s][m]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_leafi[s][m]);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * LI * (L->n ? L->n : 1)), L->v, GL_STATIC_DRAW);
        for (int a = 0; a < 5; a++) {
            glVertexAttribPointer((GLuint)(1 + a), 4, GL_FLOAT, GL_FALSE, LI * 4, (void *)(intptr_t)(a * 16));
            glEnableVertexAttribArray((GLuint)(1 + a));
            glVertexAttribDivisor((GLuint)(1 + a), 1);
        }
    }
    free(v); free(ix);
}

static GLuint mk_tex(int w, int h, GLenum ifmt, GLenum filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, ifmt == GL_RG32F || ifmt == GL_RG16F ? GL_RG : GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static GLuint mk_rb_depth(int w, int h) {
    GLuint r;
    glGenRenderbuffers(1, &r);
    glBindRenderbuffer(GL_RENDERBUFFER, r);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    return r;
}
static GLuint mk_fbo(int n, const GLuint *tex, GLuint depth_rb) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    GLenum bufs[3];
    for (int i = 0; i < n; i++) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
        bufs[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    if (depth_rb) glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);
    if (n) glDrawBuffers(n, bufs);
    else { glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE); }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete");
    return f;
}

static void bake_static(void);

static void make_targets(void) {
    sw_ = (int)lroundf(W * opt_scale); sh_ = (int)lroundf(H * opt_scale);
    if (sw_ < 16) sw_ = 16;
    if (sh_ < 16) sh_ = 16;
    if (fb_w) {
        GLuint fb[] = {fbo_main, fbo_sdepth, fbo_bake, fbo_gbuf, fbo_taa[0], fbo_taa[1], fbo_sh};
        glDeleteFramebuffers(7, fb);
        GLuint tx[] = {tex_cur, tex_aux, tex_bake[0], tex_bake[1], tex_bake[2], tex_g[0], tex_g[1], tex_g[2], tex_taa[0], tex_taa[1], tex_sh};
        glDeleteTextures(11, tx);
        GLuint rb[] = {rb_depth, rb_sdepth, rb_bake};
        glDeleteRenderbuffers(3, rb);
    }
    tex_cur = mk_tex(sw_, sh_, GL_RGBA16F, GL_LINEAR);
    tex_aux = mk_tex(sw_, sh_, GL_RG32F, GL_NEAREST);
    rb_depth = mk_rb_depth(sw_, sh_);
    GLuint m2[2] = {tex_cur, tex_aux};
    fbo_main = mk_fbo(2, m2, rb_depth);
    rb_sdepth = mk_rb_depth(sw_, sh_);
    fbo_sdepth = mk_fbo(0, NULL, rb_sdepth);
    for (int i = 0; i < 3; i++) {
        tex_bake[i] = mk_tex(sw_, sh_, GL_RGBA16F, GL_NEAREST);
        tex_g[i] = mk_tex(sw_, sh_, GL_RGBA16F, GL_NEAREST);
    }
    rb_bake = mk_rb_depth(sw_, sh_);
    fbo_bake = mk_fbo(3, tex_bake, rb_bake);
    fbo_gbuf = mk_fbo(3, tex_g, 0);
    for (int i = 0; i < 2; i++) {
        tex_taa[i] = mk_tex(sw_, sh_, GL_RGBA16F, GL_LINEAR);
        fbo_taa[i] = mk_fbo(1, &tex_taa[i], 0);
    }
    tex_sh = mk_tex((sw_ + 3) / 4, (sh_ + 3) / 4, GL_RGBA16F, GL_LINEAR);
    fbo_sh = mk_fbo(1, &tex_sh, 0);
    fb_w = W; fb_h = H;
    taa_frames = 0;
    bake_static();
}

static void gl_init(void) {
    p_solid = mk_prog(VS_SOLID, NULL, FS_SOLID, CREATURE_FS);
    p_leaf = mk_prog(VS_LEAF, NULL, FS_LEAF, NULL);
    p_caus = mk_prog(VS_CAUS, NULL, FS_CAUS, NULL);
    p_shafts = mk_prog(VS_FULL, VIEWRAY, FS_SHAFTS, NULL);
    p_comp = mk_prog(VS_FULL, VIEWRAY, FS_COMP, NULL);
    p_accum = mk_prog(VS_FULL, NULL, FS_ACCUM, NULL);
    p_relight = mk_prog(VS_FULL, NULL, FS_RELIGHT, NULL);
    p_taa = mk_prog(VS_FULL, NULL, FS_TAA, NULL);
    glGenVertexArrays(1, &vao_empty);

    /* the static scene */
    glGenVertexArrays(1, &vao_scene);
    glBindVertexArray(vao_scene);
    glGenBuffers(1, &vbo_scene);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_scene);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * SV * scene_mesh.nv), scene_mesh.v, GL_STATIC_DRAW);
    solid_attribs();
    glGenBuffers(1, &ibo_scene);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_scene);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * scene_mesh.ni), scene_mesh.ix, GL_STATIC_DRAW);

    leaf_mesh(LM_TINY, 2, 2);
    leaf_mesh(LM_LEAF, 2, 8);
    leaf_mesh(LM_RIBBON, 1, 28);
    leaf_mesh(LM_BLADE, 1, 4);

    /* the caustics: a grid of the surface, and the texture it lights */
    {
        int n = 112;
        float *v = malloc(sizeof(float) * 2 * (n + 1) * (n + 1));
        uint32_t *ix = malloc(sizeof(uint32_t) * 6 * n * n);
        int k = 0;
        for (int j = 0; j <= n; j++)
            for (int i = 0; i <= n; i++) { v[k++] = i * (float)CAUS_N / n; v[k++] = j * (float)CAUS_N / n; }
        caus_ni = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                uint32_t a = (uint32_t)(j * (n + 1) + i), b = a + 1, c = a + (uint32_t)n + 1, d = c + 1;
                ix[caus_ni++] = a; ix[caus_ni++] = b; ix[caus_ni++] = c; ix[caus_ni++] = b; ix[caus_ni++] = d; ix[caus_ni++] = c;
            }
        glGenVertexArrays(1, &vao_caus);
        glBindVertexArray(vao_caus);
        glGenBuffers(1, &vbo_caus);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_caus);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * k), v, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
        glEnableVertexAttribArray(0);
        glGenBuffers(1, &ibo_caus);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_caus);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * caus_ni), ix, GL_STATIC_DRAW);
        free(v); free(ix);
        glGenTextures(1, &tex_caus);
        glBindTexture(GL_TEXTURE_2D, tex_caus);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CAUS_N, CAUS_N, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D);
        fbo_caus = mk_fbo(1, &tex_caus, 0);
    }
    /* the shadow maps, from the lamp straight down the light: one of what
     * never moves, made once, and one each frame with the animals added */
    for (int s = 0; s < 2; s++) {
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_w, shadow_h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
        GLuint f;
        glGenFramebuffers(1, &f);
        glBindFramebuffer(GL_FRAMEBUFFER, f);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, t, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        if (s) { tex_shadow_s = t; fbo_shadow_s = f; } else { tex_shadow = t; fbo_shadow = f; }
    }
    /* the ground map, for soft occlusion on the sand */
    glGenTextures(1, &tex_ground);
    glBindTexture(GL_TEXTURE_2D, tex_ground);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, gnx, gnz, 0, GL_RED, GL_FLOAT, ground);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    creatures_gl();
    overlay_gl();
}

/* the lamp's view: an orthographic box round the tank, along the light */
static void build_shadow_matrix(void) {
    V3 f = light_dir, r = vnorm(vcross(f, v3(0, 0, 1))), u = vcross(r, f);
    float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
    for (int i = 0; i < 8; i++) {
        V3 c = v3(i & 1 ? X1 : X0, i & 2 ? YW + 0.01f : -0.01f, i & 4 ? TD : 0);
        float q[3] = {vdot(c, r), vdot(c, u), vdot(c, f)};
        for (int k = 0; k < 3; k++) { lo[k] = fminf(lo[k], q[k]); hi[k] = fmaxf(hi[k], q[k]); }
    }
    float s[3], o[3];
    for (int k = 0; k < 3; k++) { s[k] = 2 / (hi[k] - lo[k]); o[k] = -(hi[k] + lo[k]) / (hi[k] - lo[k]); }
    V3 ax[3] = {r, u, f};
    memset(shadowM, 0, sizeof shadowM);
    for (int k = 0; k < 3; k++) {
        shadowM[0 + k] = ax[k].x * s[k];
        shadowM[4 + k] = ax[k].y * s[k];
        shadowM[8 + k] = ax[k].z * s[k];
        shadowM[12 + k] = o[k];
    }
    shadowM[15] = 1;
}

/* ---- the camera for this output ---- */
static void frame_camera(float bx, float by, float bw, float bh, float ccx, float ccy) {
    can_x0 = bx; can_y0 = by; can_w = bw; can_h = bh;
    cam_cx = ccx; cam_cy = ccy;
    float top = by + 0.034f * bh, bot = by + bh - 0.026f * bh;
    KPX = (bot - top) / YW;
    CY = YW - (ccy - top) / KPX;
    DC = 0.62f + 0.62f * YW;
    cam_f = KPX * DC;
    CX = 0;
}

static V3 focus_at;
static float focus_size;
static int focus_on;

static void setup_view(void) {
    if (span_me >= 0) {
        view_x = span[span_me][0]; view_y = span[span_me][1];
        view_w = span[span_me][2]; view_h = span[span_me][3];
    } else {
        view_x = 0; view_y = 0; view_w = (float)W; view_h = (float)H;
        frame_camera(0, 0, (float)W, (float)H, W / 2.0f, H / 2.0f);
    }
    if (focus_on) {
        /* a close-up: the camera steps up to the glass in front of the animal */
        CX = focus_at.x; CY = focus_at.y;
        DC = 0.16f;
        float app = DC + fmaxf(focus_at.z, 0) / NW;
        cam_f = view_w * app / (focus_size * 2.6f);
        cam_cx = view_x + view_w / 2; cam_cy = view_y + view_h / 2;
        KPX = cam_f / DC;
    }
    proj[0] = 2 * cam_f / view_w;
    proj[2] = 2 * (cam_cx - view_x) / view_w - 1 + 2 * jit[0] / sw_;
    proj[1] = 2 * cam_f / view_h;
    proj[3] = 1 - 2 * (cam_cy - view_y) / view_h + 2 * jit[1] / sh_;
}

/* ---- the caustics ---- */
static float caus_k[8][4];
static float caus_w[8];
static void caustics_init(void) {
    static const int kv[8][2] = {{2, 1}, {-1, 3}, {3, -2}, {4, 3}, {-5, 2}, {1, -6}, {6, 5}, {-7, -3}};
    for (int i = 0; i < 8; i++) {
        float kl = sqrtf((float)(kv[i][0] * kv[i][0] + kv[i][1] * kv[i][1]));
        caus_k[i][0] = (float)kv[i][0];
        caus_k[i][1] = (float)kv[i][1];
        caus_k[i][2] = 1.0f / kl * powf(kl, -0.35f) * (0.8f + 0.4f * hf(opt_seed, (uint32_t)i, 41));
        caus_k[i][3] = 0;
        caus_w[i] = 1.25f * sqrtf(kl) * (0.85f + 0.3f * hf(opt_seed, (uint32_t)i, 42));
    }
}

static void render_caustics(double t) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_caus);
    glViewport(0, 0, CAUS_N, CAUS_N);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p_caus);
    float ph[8];
    for (int i = 0; i < 8; i++) ph[i] = (float)fmod(caus_w[i] * t + i * 1.7, TAU);
    glUniform1fv(U(p_caus, "uPh"), 8, ph);
    glUniform4fv(U(p_caus, "uK"), 8, &caus_k[0][0]);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBindVertexArray(vao_caus);
    /* shallow: soft and broad; deep: the sharp network */
    glUniform1f(U(p_caus, "uD"), 7.0f);
    glUniform4f(U(p_caus, "uMask"), 1, 0, 0, 0);
    glDrawElementsInstanced(GL_TRIANGLES, caus_ni, GL_UNSIGNED_INT, 0, 9);
    glUniform1f(U(p_caus, "uD"), 22.0f);
    glUniform4f(U(p_caus, "uMask"), 0, 1, 0, 0);
    glDrawElementsInstanced(GL_TRIANGLES, caus_ni, GL_UNSIGNED_INT, 0, 9);
    glDisable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D, tex_caus);
    glGenerateMipmap(GL_TEXTURE_2D);
    /* the pattern drifts with the surface current */
    caus_off[0] = (float)fmod(t * 0.011, 1.0);
    caus_off[1] = (float)fmod(t * 0.004, 1.0);
}

static void draw_scene_solids(int pass) {
    set_common(p_solid);
    glBindVertexArray(vao_scene);
    int n = pass == 1 ? scene_ni_opaque : nscene_stone_i1;
    glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, 0);
}

/* the plants: still ones (s = 1, baked) or swaying ones */
static void draw_leaves(int still) {
    set_common(p_leaf);
    float ld = cpu_load(wall());
    glUniform2f(U(p_leaf, "uCurrent"), outflow_dir * (0.006f + 0.02f * ld), 0.002f);
    glUniform1f(U(p_leaf, "uSway"), 0.008f + 0.012f * ld);
    for (int m = 0; m < LM_N; m++) {
        Inst *L = still ? &leaves_s[m] : &leaves[m];
        if (!L->n) continue;
        glBindVertexArray(vao_leaf[still][m]);
        glDrawElementsInstanced(GL_TRIANGLES, leaf_ni[m], GL_UNSIGNED_INT, 0, L->n);
    }
}

static void bind_scene_textures(GLuint shadow) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_caus);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, shadow);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_ground);
    glActiveTexture(GL_TEXTURE0);
}

static void fullscreen(void) {
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* everything that never moves: the G-buffer (supersampled by accumulating
 * jittered renders), its depth, and its shadow map */
static void bake_static(void) {
    double t0 = now();
    t_now = 30;
    build_shadow_matrix();
    float savej[2] = {jit[0], jit[1]};
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
    /* the still shadow map: wood, stones, gear and every plant */
    cur_pass = 1;
    bind_scene_textures(0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow_s);
    glViewport(0, 0, shadow_w, shadow_h);
    glClear(GL_DEPTH_BUFFER_BIT);
    draw_scene_solids(1);
    draw_leaves(0);
    draw_leaves(1);
    /* the G-buffer */
    cur_pass = 2;
    bind_scene_textures(0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_gbuf);
    glViewport(0, 0, sw_, sh_);
    float z4[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; i++) glClearBufferfv(GL_COLOR, i, z4);
    const int NS = 12;
    for (int k = 0; k < NS; k++) {
        /* a spread of subsample positions (golden-angle spiral) */
        float r = sqrtf((k + 0.5f) / NS) * 0.6f, a = k * 2.39996f;
        jit[0] = r * cosf(a); jit[1] = r * sinf(a);
        setup_view();
        cur_pass = 2;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bake);
        glEnable(GL_DEPTH_TEST);
        for (int i = 0; i < 3; i++) glClearBufferfv(GL_COLOR, i, z4);
        glClear(GL_DEPTH_BUFFER_BIT);
        set_common(p_solid);
        glUniform1f(U(p_solid, "uBakeW"), 1);
        glUniform1f(U(p_solid, "uSub"), (float)k);
        glBindVertexArray(vao_scene);
        glDrawElements(GL_TRIANGLES, nscene_stone_i1, GL_UNSIGNED_INT, 0);
        set_common(p_leaf);
        glUniform1f(U(p_leaf, "uBakeW"), 1);
        glUniform1f(U(p_leaf, "uSub"), (float)k);
        draw_leaves(1);
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_gbuf);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glUseProgram(p_accum);
        glUniform1f(U(p_accum, "uW"), 1.0f / NS);
        for (int i = 0; i < 3; i++) {
            char nm[8];
            snprintf(nm, sizeof nm, "uG%d", i);
            glUniform1i(U(p_accum, nm), 3 + i);
            glActiveTexture(GL_TEXTURE3 + i);
            glBindTexture(GL_TEXTURE_2D, tex_bake[i]);
        }
        glActiveTexture(GL_TEXTURE0);
        fullscreen();
        glDisable(GL_BLEND);
    }
    /* its depth, unjittered, for the moving things to hide behind */
    jit[0] = jit[1] = 0;
    setup_view();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_sdepth);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_DEPTH_BUFFER_BIT);
    set_common(p_solid);
    glBindVertexArray(vao_scene);
    glDrawElements(GL_TRIANGLES, nscene_stone_i1, GL_UNSIGNED_INT, 0);
    draw_leaves(1);
    glDisable(GL_DEPTH_TEST);
    jit[0] = savej[0]; jit[1] = savej[1];
    setup_view();
    cur_pass = 0;
    glFinish();
    if (getenv("AQ_VERBOSE")) fprintf(stderr, "aquarium: baked the still scene at %dx%d in %.0f ms\n", sw_, sh_, (now() - t0) * 1000);
}

/* --bench: GPU time of each pass */
static GLuint tq[8];
static double tq_sum[8];
static int tq_on, tq_n;
static const char *tq_name[8] = {"caustics", "shadow", "scene", "taa", "shafts", "composite", "overlay", ""};
static void tq_begin(int i) { if (tq_on) glBeginQuery(GL_TIME_ELAPSED, tq[i]); }
static void tq_end(void) { if (tq_on) glEndQuery(GL_TIME_ELAPSED); }
static void tq_collect(void) {
    if (!tq_on) return;
    for (int i = 0; i < 7; i++) { GLuint64 ns = 0; glGetQueryObjectui64v(tq[i], GL_QUERY_RESULT, &ns); tq_sum[i] += ns * 1e-6; }
    tq_n++;
}

static float taa_alpha = 0.12f;
static int taa_clamp = 1;

static void render(double t, double wt, GLuint target) {
    t_now = t;
    lighting(wt);
    build_shadow_matrix();
    if (fb_w != W || fb_h != H) { setup_view(); make_targets(); t_now = t; }
    creatures_update(t, wt);
    if (opt_focus) {
        int was = focus_on;
        focus_on = creature_focus(opt_focus, &focus_at, &focus_size);
        if (focus_on || was) { setup_view(); if (!was) bake_static(); }
    }
    setup_view();

    tq_begin(0);
    render_caustics(t);
    tq_end();
    if (getenv("AQ_CDUMP")) {
        static float px[CAUS_N * CAUS_N * 4];
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_caus);
        glReadPixels(0, 0, CAUS_N, CAUS_N, GL_RGBA, GL_FLOAT, px);
        FILE *f = fopen(getenv("AQ_CDUMP"), "wb");
        fprintf(f, "P6\n%d %d\n255\n", CAUS_N, CAUS_N);
        double sum = 0;
        for (int i = 0; i < CAUS_N * CAUS_N; i++) {
            unsigned char c[3] = {(unsigned char)fminf(px[i * 4] * 80, 255), (unsigned char)fminf(px[i * 4 + 1] * 80, 255), 0};
            sum += px[i * 4 + 1];
            fwrite(c, 1, 3, f);
        }
        fclose(f);
        fprintf(stderr, "caustics mean %.3f\n", sum / (CAUS_N * CAUS_N));
    }

    /* the shadow map: the still one, then the animals */
    tq_begin(1);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_shadow_s);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_shadow);
    glBlitFramebuffer(0, 0, shadow_w, shadow_h, 0, 0, shadow_w, shadow_h, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    cur_pass = 1;
    bind_scene_textures(0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, shadow_w, shadow_h);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
    creatures_draw(1);
    tq_end();

    /* the scene: the baked still part relit, then everything that moves */
    tq_begin(2);
    cur_pass = 0;
    bind_scene_textures(getenv("AQ_SHS") ? tex_shadow_s : tex_shadow);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_sdepth);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_main);
    glBlitFramebuffer(0, 0, sw_, sh_, 0, 0, sw_, sh_, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);
    glViewport(0, 0, sw_, sh_);
    glDisable(GL_DEPTH_TEST);
    set_common(p_relight);
    glUniform1i(U(p_relight, "uDbg"), getenv("AQ_DBG") ? atoi(getenv("AQ_DBG")) : 0);
    for (int i = 0; i < 3; i++) {
        char nm[8];
        snprintf(nm, sizeof nm, "uG%d", i);
        glUniform1i(U(p_relight, nm), 3 + i);
        glActiveTexture(GL_TEXTURE3 + i);
        glBindTexture(GL_TEXTURE_2D, tex_g[i]);
    }
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    glEnable(GL_DEPTH_TEST);
    creatures_draw(0);
    if (getenv("AQ_FWD")) draw_scene_solids(0);
    draw_leaves(0);
    creatures_draw_translucent();
    glDisable(GL_DEPTH_TEST);
    tq_end();

    /* temporal antialiasing */
    tq_begin(3);
    int prev = taa_cur;
    taa_cur ^= 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_taa[taa_cur]);
    glUseProgram(p_taa);
    glUniform1i(U(p_taa, "uCur"), 3);
    glUniform1i(U(p_taa, "uHist"), 4);
    glUniform1f(U(p_taa, "uAlpha"), taa_frames ? taa_alpha : 1.0f);
    glUniform1i(U(p_taa, "uClamp"), taa_clamp);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_cur);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_taa[prev]);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    taa_frames++;
    tq_end();

    /* light shafts, at a quarter size */
    tq_begin(4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_sh);
    glViewport(0, 0, (sw_ + 3) / 4, (sh_ + 3) / 4);
    set_common(p_shafts);
    float vr[4] = {view_x, view_y, view_w, view_h}, cc[3] = {cam_cx, cam_cy, cam_f};
    glUniform4fv(U(p_shafts, "uView"), 1, vr);
    glUniform3fv(U(p_shafts, "uCanCam"), 1, cc);
    glUniform1i(U(p_shafts, "uAux"), 3);
    glUniform1f(U(p_shafts, "uShaftK"), (getenv("AQ_SK") ? atof(getenv("AQ_SK")) : 0.06f) * lit.shaft);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    tq_end();

    /* the composite */
    tq_begin(5);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    set_common(p_comp);
    glUniform4fv(U(p_comp, "uView"), 1, vr);
    glUniform3fv(U(p_comp, "uCanCam"), 1, cc);
    float lay[4] = {can_x0, can_y0, can_w, can_h};
    glUniform4fv(U(p_comp, "uLay"), 1, lay);
    glUniform1i(U(p_comp, "uCol"), 4);
    glUniform1i(U(p_comp, "uAux"), 3);
    glUniform1i(U(p_comp, "uShafts"), 5);
    glUniform1f(U(p_comp, "uExpo"), 1.0f);
    glUniform1f(U(p_comp, "uCA"), focus_on ? 0.0f : 1.0f);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_taa[taa_cur]);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_sh);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    tq_end();

    tq_begin(6);
    if (opt_overlay && !focus_on) overlay_draw(wt);
    tq_end();
}

/* the next subpixel jitter (Halton 2, 3) */
static void next_jitter(int i) {
    float h2 = 0, h3 = 0, f = 0.5f;
    for (int n = i + 1; n; n /= 2, f *= 0.5f) h2 += f * (n % 2);
    f = 1.0f / 3;
    for (int n = i + 1; n; n /= 3, f /= 3) h3 += f * (n % 3);
    jit[0] = h2 - 0.5f; jit[1] = h3 - 0.5f;
}

/* ==================================================================== */
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
    /* a resize remakes the targets at the next frame */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "aquarium");
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
        "usage: aquarium [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--hour H] [--load 0..1] [--temp C] [--away SECONDS] [--no-overlay]\n"
        "                [--msaa 0|2|4|8] [--scale S] [--focus tetra|angel|discus|cory|eel|shrimp|amano]\n"
        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
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

/* the design canvas: the whole layout when spanning (one tank across every
 * output, the camera centred on the one under the middle), else the screen */
static void design(int dw, int dh) {
    if (span_me >= 0) {
        float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        for (int i = 0; i < span_n; i++) {
            x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
            y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
        }
        float mx = (x0 + x1) / 2;
        front_out = 0;
        for (int i = 0; i < span_n; i++)
            if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front_out = i; break; }
        design_tank(x0, y0, x1 - x0, y1 - y0, span[front_out][0] + span[front_out][2] / 2, span[front_out][1] + span[front_out][3] / 2);
    } else {
        design_tank(0, 0, (float)dw, (float)dh, dw / 2.0f, dh / 2.0f);
    }
}

static double last_read;
static void maybe_read_sensors(double wt) {
    /* every instance reads on the same wall-clock second */
    if (floor(wt) != last_read) {
        read_sensors();
        sens.at = floor(wt);
        last_read = floor(wt);
    }
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
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-hour") && i + 1 < argc) opt_hour = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-load") && i + 1 < argc) opt_load = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-temp") && i + 1 < argc) opt_temp = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-away") && i + 1 < argc) opt_away = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-focus") && i + 1 < argc) opt_focus = argv[++i];
        else if (!strcmp(a, "-msaa") && i + 1 < argc) opt_msaa = atoi(argv[++i]);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = clampf(strtof(argv[++i], NULL), 0.25f, 2.0f);
        else if (!strcmp(a, "-no-overlay")) opt_overlay = 0;
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    find_temp();
    read_sensors();
    {   /* a first CPU-load sample over a tenth of a second */
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
        read_sensors();
        sens.load_prev = sens.load;
    }
    fonts_init();

    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;

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
        /* a window is about to go fullscreen: design for the screen */
        dw = ra.width; dh = ra.height;
        if (opt_root) { W = ra.width; H = ra.height; XSelectInput(dpy, root, StructureNotifyMask); }
        if (offscreen) { W = dw = off_w; H = dh = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Aquarium");
        XClassHint chint = {"aquarium", "Aquarium"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        int ctx_attr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                          GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None};
        typedef GLXContext (*CreateCtx)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
        CreateCtx create = (CreateCtx)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
        int nfb = 0;
        int fb_attr[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT,
                         GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_DOUBLEBUFFER, True, None};
        GLXFBConfig *fbc = glXChooseFBConfig(dpy, scr, fb_attr, &nfb);
        GLXContext ctx = NULL;
        if (create && fbc && nfb > 0) {
            for (int i = 0; i < nfb && !ctx; i++) {
                XVisualInfo *v2 = glXGetVisualFromFBConfig(dpy, fbc[i]);
                if (v2 && v2->visualid == vi->visualid) ctx = create(dpy, fbc[i], NULL, True, ctx_attr);
                if (v2) XFree(v2);
            }
            if (!ctx) ctx = create(dpy, fbc[0], NULL, True, ctx_attr);
        }
        if (!ctx) die("can't create a GL 3.3 core context");
        glXMakeCurrent(dpy, win, ctx);
    }

    /* the tank, from the seed */
    design(dw, dh);
    rng = opt_seed | 1;
    build_scape();
    build_scene_mesh();
    caustics_init();
    rng = hash32(opt_seed) | 1;
    creatures_init();
    gl_init();
    setup_view();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (opt_shot) {
            /* the moment, held still while the jitter supersamples it */
            double wt = opt_epoch ? opt_epoch + opt_at : wall();
            sens.at = floor(wt) - 1;
            taa_clamp = 0;
            for (int i = 0; i < 12; i++) {
                next_jitter(i);
                taa_alpha = 1.0f / (i + 1);
                render(opt_at, wt, fbo);
            }
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double wt0 = wall();
        next_jitter(0);
        render(30.0, wt0, fbo);
        glFinish();
        glGenQueries(8, tq);
        tq_on = 1;
        double sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = 30.0 + (i + 1) / 60.0;
            next_jitter(i + 1);
            render(t, wt0 + (i + 1) / 60.0, fbo);
            glFinish();
            tq_collect();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("aquarium: %dx%d (scene %dx%d), %d frames: avg %.3f ms, worst %.3f ms per frame\n",
               W, H, sw_, sh_, opt_bench, sum / opt_bench, worst);
        printf("aquarium: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < 7; i++) printf("  %-10s %.3f ms\n", tq_name[i], tq_sum[i] / (tq_n ? tq_n : 1));
        printf("  plants: swaying %d+%d+%d+%d, still %d+%d+%d+%d; scene %d verts %d tris\n",
               leaves[0].n, leaves[1].n, leaves[2].n, leaves[3].n, leaves_s[0].n, leaves_s[1].n, leaves_s[2].n, leaves_s[3].n,
               scene_mesh.nv, scene_mesh.ni / 3);
        return 0;
    }

    double t0 = now();
    int frame = 0;
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
        maybe_read_sensors(wt);
        next_jitter(frame++ % 8);
        render(el, wt, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
