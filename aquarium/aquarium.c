/*
 * aquarium — a planted freshwater aquarium seen through the front glass.
 *
 * A nature-style scape: sand with ripples and leaf litter, seiryu stones,
 * branching driftwood furred with moss and carrying ferns and Anubias,
 * stem plants (Rotala, Ludwigia, Hygrophila), Amazon swords, Vallisneria,
 * a hairgrass and Monte Carlo carpet. The plants sway in the filter's
 * current. An air stone sends up a bubble column; leaves pearl oxygen in
 * strong light; motes drift and catch the light.
 *
 * Who lives here:
 *   neon tetras   a school: boids (separation, alignment) anchored to
 *                 closed-form slots in a wandering school, with darts
 *   angelfish     silver, four bars running into tall swept fins, feelers
 *   discus        red with turquoise vermiculation, a fringe of fin
 *   Corydoras     pandas grazing nose-down, now and then dashing up for air
 *   a fire eel    cruising the bottom, nosing round the stones, and now and
 *                 then burrowing right into the sand to wait, head out
 *   shrimp        cherry and Amano: grazing with alternate picks of the
 *                 claws, walking, hopping, now and then flicking backwards
 * Fish are lofted from cross-sections with membrane fins on rays, bent per
 * vertex by a travelling body wave (tail beat from speed, turning, banking,
 * pectorals sculling when they hover), with iridescent, silvered, scaled
 * shading and proper eyes. The eel's body follows its head's track by
 * arclength; the shrimp are built each frame from segments and limbs.
 *
 * The water: refraction through the flat front glass (Snell, per vertex),
 * caustics splatted from a refracting mesh of the rippled surface, a shadow
 * map from the lamp, light shafts, absorption and haze with distance, the
 * underside of the surface and the side panes as mirrors (total internal
 * reflection), dispersion toward the edges, TAA.
 *
 * The lamp follows the local clock: a warm dawn, daylight, a warm, dimmer
 * evening, blue moonlight at night (the neons fade, everyone slows and
 * goes low). The machine shows quietly: CPU load stirs the fish and the
 * current; a thermometer on the glass reads the CPU package temperature;
 * a small caption names the host and how long the screen has been locked.
 *
 * Cost: everything still (tank, sand, stones, wood, carpet, moss) is baked
 * once, supersampled, into a G-buffer of light-independent terms and relit
 * each frame; only swaying plants and animals are drawn per frame, at an
 * internal resolution that follows the output's size (--scale to override).
 *
 * Deterministic: everything follows from --seed and the time since
 * --epoch. The fish run a fixed-step sim in overlapping 24 s chapters, each
 * replayed from a closed-form start and crossfaded, so any instance (or a
 * restarted one) agrees; the eel's and shrimps' lives are planned ahead
 * from the seed. Spanning: one instance per output with the same --seed,
 * --epoch and --layout, --output = its index; one tank fills the layout's
 * bounding box through one camera centred on the output under the middle,
 * so fish swim from one screen to the next. A portrait screen gets a tall
 * tank.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer, --shot FILE.ppm [--at S]
 * [--size WxH], --bench N. --hour, --load, --temp, --away fake the clock
 * and sensors; --focus KIND [--pose YAW] frames a close-up of one animal.
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
#include "../common/hidpi.h"

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
static float opt_pose = -1e9f;         /* --pose YAW: with --focus, hold it side on for a portrait */
static int opt_msaa = 4;
static float opt_scale = 0;            /* internal resolution of the scene; 0: by the output's size */
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
            FT_ULong code = c == 95 ? 0xB7 : (FT_ULong)(32 + c);
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
                int mesh = LM_TINY;
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
        leaf_add(LM_RIBBON, L_VALLIS, v3(x, y0, z), dir, vrot(v3(1, 0, 0), v3(0, 1, 0), tw), len * frange(0.75f, 1.0f), frange(0.0045f, 0.007f),
                 y0, len, 2.6f, -frange(0.4f, 1.4f), 0.15f, hue, 0);
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
    bubbler_z = TD * 0.58f;
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
        vec2 a = texture(uCaus, uv).rg, b = texture(uCaus, uv * 1.005 + 0.001).rg;
        float k = smoothstep(0.015, 0.22, depth);
        float ca = mix(a.r, a.g, k), cb = mix(b.r, b.g, k);
        /* the focused lines, without blowing out what they land on */
        vec3 c = vec3(ca, mix(ca, cb, 0.5), cb);
        return 0.3 + 0.7 * c / (1.0 + c * 0.22);
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
    float leaf_w(int type, float v) {
        if (type == 0) return pow(sin(3.1416 * min(v, 0.999)), 0.5) * (1.0 - 0.3 * v);
        if (type == 1 || type == 8) return pow(max(sin(3.1416 * pow(v, 1.25)), 0.0), 0.75) * smoothstep(0.0, 0.2, v);
        if (type == 3) return v < 0.28 ? 0.06 : pow(max(sin(3.1416 * (v - 0.28) / 0.72), 0.0), 0.8);
        if (type == 2 || type == 9) return pow(max(sin(3.1416 * pow(v, 0.8)), 0.0), 0.9);
        if (type == 4) return 1.0 - smoothstep(0.93, 1.0, v) * 0.7;
        if (type == 5) return 1.0 - 0.8 * v;
        if (type == 6) return 1.0;
        if (type == 11) return pow(max(sin(3.1416 * pow(v, 0.85)), 0.0), 0.7);
        return 1.0;
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
    float g_alpha = 1.0;
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
        o0 = vec4(col, g_alpha);
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
            g_alpha = alpha;
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
    uniform int uTiny;
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
        float w = wid * (uTiny == 1 && type != 7 ? 1.0 : leaf_w(type, v));
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
    uniform int uTiny;
    void main() {
        int type = int(vInfo.x + 0.5);
        float hue = vInfo.y, hf = vInfo.z, seed = vInfo.w;
        float u = vUV.x, v = vUV.y;
        /* coverage: soft edges; moss is feathery */
        float a = 1.0 - smoothstep(0.75, 1.0, abs(u));
        if (type == 7) {
            /* a moss sprig: a wiry stem with fine side shoots, tapering */
            float w = (1.0 - v) * 0.95;
            float stem = 1.0 - smoothstep(0.06, 0.14, abs(u));
            float br = 1.0 - smoothstep(0.1, 0.24, abs(fract(v * 6.0 - abs(u) * 1.3 + seed * 3.0) - 0.5));
            a = max(stem, br * (1.0 - smoothstep(w - 0.1, w, abs(u)))) * (1.0 - smoothstep(0.9, 1.0, v));
        }
        if (type == 6) a = 1.0 - smoothstep(0.8, 1.0, length(vec2(u, v * 2.0 - 1.0)));
        else if (uTiny == 1 && type != 7) { float lw = leaf_w(type, v); a = 1.0 - smoothstep(lw * 0.8, lw, abs(u)); }
        if (a < 0.5) discard;
        if (uPass == 1) return;
        vec3 N = normalize(vN);
        if (!gl_FrontFacing) N = -N;
        vec3 alb; float spec = 0.05, rough = 24.0, trans = 0.5;
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
            alb = vec3(0.1, 0.28, 0.06) * (1.0 + var);
            trans = 0.25; spec = 0.03;
            float rib = 1.0 - smoothstep(0.0, 0.06, abs(u));
            float vein = 1.0 - smoothstep(0.0, 0.08, abs(abs(u) - 0.55 * sin(3.1416 * clamp((v - 0.28) / 0.72, 0.0, 1.0))));
            alb = mix(alb, vec3(0.45, 0.62, 0.28), max(rib, vein * 0.5) * step(0.28, v));
            if (v < 0.28) alb = vec3(0.3, 0.45, 0.18);
            spec = 0.25;
        } else if (type == 4) {
            alb = vec3(0.2, 0.4, 0.1) * (1.0 + var) * (0.85 + 0.15 * sin(u * 30.0)) * mix(0.75, 1.1, v);
            trans = 0.35; spec = 0.02;
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
        }
        o0 = vec4(col, 1.0);
    });

/* temporal antialiasing: the camera never moves, so the history only
 * needs clamping where things do */
static const char *FS_TAA = GLSL(
    in vec2 vUV;
    uniform sampler2D uCur, uHist, uAux, uShafts;
    uniform float uAlpha;
    uniform int uClamp;
    /* the light shafts march from a fresh offset every frame, so they go
     * through the history too, and widen the clamp by their own spread */
    vec3 shaft(ivec2 p, vec2 o) {
        return texture(uShafts, (vec2(p) + 0.5) / vec2(textureSize(uCur, 0)) + o / vec2(textureSize(uShafts, 0))).rgb;
    }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        vec3 c = texelFetch(uCur, p, 0).rgb;
        vec3 h = texelFetch(uHist, p, 0).rgb;
        vec2 aux = texelFetch(uAux, p, 0).rg;
        float wet = aux.r > 0.0 || abs(aux.g) > 0.02 ? 1.0 : 0.0;
        vec3 sh = shaft(p, vec2(0.0)) * wet;
        if (uClamp == 1) {
            vec3 mn = c, mx = c, smn = sh, smx = sh;
            for (int k = 0; k < 4; k++) {
                ivec2 o = k == 0 ? ivec2(1, 0) : k == 1 ? ivec2(-1, 0) : k == 2 ? ivec2(0, 1) : ivec2(0, -1);
                vec3 s = texelFetch(uCur, p + o, 0).rgb;
                mn = min(mn, s); mx = max(mx, s);
                s = shaft(p, vec2(o)) * wet;
                smn = min(smn, s); smx = max(smx, s);
            }
            mn += smn; mx += smx;
            vec3 ext = (mx - mn) * 0.25;
            h = clamp(h, mn - ext, mx + ext);
        }
        o0 = vec4(mix(h, c + sh, uAlpha), 1.0);
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
    uniform float uJit;
    void main() {
        vec3 G, d;
        view_ray(vUV, G, d);
        float zEnd = texture(uAux, vUV).r;
        if (zEnd <= 0.0) zEnd = uTD;
        zEnd = min(zEnd, uTD);
        float sEnd = zEnd / d.z;
        if (d.y > 0.0) sEnd = min(sEnd, (uYW - G.y) / d.y);
        const int N = 6;
        float ds = sEnd / float(N);
        float j = fract(hash12(gl_FragCoord.xy) + uJit * 0.6180339);
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
    {9.0f, {{1.7f, 1.62f, 1.48f}, {0.33f, 0.42f, 0.38f}, {0.08f, 0.12f, 0.11f}, {0.04f, 0.14f, 0.13f}, 0.0f, 1.0f, 1.0f}},
    {17.0f, {{1.7f, 1.62f, 1.48f}, {0.33f, 0.42f, 0.38f}, {0.08f, 0.12f, 0.11f}, {0.04f, 0.14f, 0.13f}, 0.0f, 1.0f, 1.0f}},
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
/* ---- the fish: bodies ----
 * Each species is one mesh in body units (standard length 1: snout at
 * x = 0.4, the base of the tail at x = -0.6), lofted from a table of
 * cross-sections along the spine, with its fins as membranes stretched on
 * rays. Instances are bent in the vertex shader by a travelling wave, so
 * they swim. Parts: 0 body, 1 dorsal, 2 caudal, 3 anal, 4 pectoral,
 * 5 pelvic, 6 adipose, 7 barbel. */
enum { SP_TETRA, SP_ANGEL, SP_DISCUS, SP_CORY, SP_N };
typedef struct { float s, top, bot, w; } Prof;

static float prof_eval(const Prof *p, int n, float s, int which) {
    if (s <= p[0].s) return (&p[0].top)[which];
    if (s >= p[n - 1].s) return (&p[n - 1].top)[which];
    int i = 0;
    while (i < n - 2 && p[i + 1].s < s) i++;
    float t = (s - p[i].s) / (p[i + 1].s - p[i].s);
    float y0 = (&p[i > 0 ? i - 1 : 0].top)[which], y1 = (&p[i].top)[which], y2 = (&p[i + 1].top)[which];
    float y3 = (&p[i + 2 < n ? i + 2 : n - 1].top)[which];
    float t2 = t * t, t3 = t2 * t;
    /* monotone-safe cubic (Catmull-Rom can overshoot near the snout) */
    float h = y1 + (y2 - y1) * (t * t * (3 - 2 * t));
    float cr = 0.5f * (2 * y1 + (y2 - y0) * t + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 + (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
    return fmaxf(mixf(h, cr, 0.6f), 0);
}

typedef struct {
    const Prof *prof; int np;
    float et, eb, ez;                   /* superellipse exponents: top, belly, sides */
    float eye_s, eye_y, eye_r;
} BodySpec;

static void build_body(Mesh *m, const BodySpec *b, int ns, int na) {
    int v0 = m->nv, i0 = m->ni;
    for (int i = 0; i <= ns; i++) {
        float f = (float)i / ns;
        float s = 0.5f - 0.5f * cosf(PI * f);
        s = mixf(s, f, 0.35f);
        float top = prof_eval(b->prof, b->np, s, 0), bot = prof_eval(b->prof, b->np, s, 1), w = prof_eval(b->prof, b->np, s, 2);
        for (int j = 0; j <= na; j++) {
            float a = (float)j / na * TAU, ca = cosf(a), sa = sinf(a);
            float y = sa >= 0 ? top * powf(sa, b->et) : -bot * powf(-sa, b->eb);
            float z = w * (ca >= 0 ? 1 : -1) * powf(fabsf(ca), b->ez);
            /* the eye bulges a little from the head */
            float de = sqrtf((s - b->eye_s) * (s - b->eye_s) + (y - b->eye_y) * (y - b->eye_y)) / b->eye_r;
            if (de < 1.3f && fabsf(z) > 1e-4f) z += (z > 0 ? 1 : -1) * b->eye_r * 0.22f * fmaxf(0, 1 - de * de / 1.69f);
            float yn = y >= 0 ? (top > 1e-5f ? y / top : 0) : (bot > 1e-5f ? y / bot : 0);
            m_vert(m, v3(0.4f - s, y, z), v3(0, 0, 0), s, (float)j / na, 0, s, yn, z >= 0 ? 1.0f : -1.0f);
        }
    }
    for (int i = 0; i < ns; i++)
        for (int j = 0; j < na; j++) {
            int a = v0 + i * (na + 1) + j, bb = a + 1, c = a + na + 1, d = c + 1;
            m_tri(m, a, bb, c); m_tri(m, bb, d, c);
        }
    m_normals(m, v0, i0);
    /* make them point out: the winding above faces in for this parametrisation */
    {
        int k = v0 + (ns / 2) * (na + 1);   /* a vertex on the right side, mid-body */
        if (vdot(m_nrm(m, k), v3(0, 0, 1)) < 0)
            for (int i = v0; i < m->nv; i++) m_setn(m, i, vmul(m_nrm(m, i), -1));
    }
    /* the snout's ring is one point: give it the forward normal */
    for (int j = 0; j <= na; j++) m_setn(m, v0 + j, v3(1, 0, 0));
}

/* body-unit point from (s, y, z) */
static V3 bp(float s, float y, float z) { return v3(0.4f - s, y, z); }

/* a fin: rays from points along its base to points along its outline
 * (a polyline, sampled evenly by length); median fins lie in z = 0 */
static void build_fin(Mesh *m, int part, V3 b0, V3 b1, const V3 *outline, int no, int nu, int nv, V3 pivot, float bulge) {
    /* outline arclength */
    float L[32];
    L[0] = 0;
    for (int i = 1; i < no; i++) L[i] = L[i - 1] + vlen(vsub(outline[i], outline[i - 1]));
    int v0 = m->nv, i0 = m->ni;
    for (int j = 0; j <= nv; j++)
        for (int i = 0; i <= nu; i++) {
            float u = (float)i / nu, v = (float)j / nv;
            V3 base = vlerp(b0, b1, u);
            float tl = u * L[no - 1];
            int k = 0;
            while (k < no - 2 && L[k + 1] < tl) k++;
            float f = (tl - L[k]) / fmaxf(L[k + 1] - L[k], 1e-6f);
            V3 edge = vlerp(outline[k], outline[k + 1], f);
            V3 p = vlerp(base, edge, v);
            /* membranes billow a little between the rays */
            p.z += bulge * sinf(PI * v) * sinf(u * PI * 9) * 0.3f + bulge * sinf(PI * v * 0.5f);
            m_vert(m, p, v3(0, 0, 1), u, v, (float)part, pivot.x, pivot.y, pivot.z);
        }
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            int a = v0 + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
            m_tri(m, a, b, c); m_tri(m, b, d, c);
        }
    m_normals(m, v0, i0);
}

/* the top or bottom of the profile at s, in body units */
static float ptop(const BodySpec *b, float s) { return prof_eval(b->prof, b->np, s, 0); }
static float pbot(const BodySpec *b, float s) { return -prof_eval(b->prof, b->np, s, 1); }

/* paired fins on both sides: the outline given for the right side (z > 0) */
static void build_pair(Mesh *m, int part, V3 b0, V3 b1, const V3 *outline, int no, int nu, int nv, float bulge) {
    for (int side = -1; side <= 1; side += 2) {
        V3 o[16];
        for (int i = 0; i < no; i++) o[i] = v3(outline[i].x, outline[i].y, outline[i].z * side);
        V3 a = v3(b0.x, b0.y, b0.z * side), b = v3(b1.x, b1.y, b1.z * side);
        V3 pv = vlerp(a, b, 0.5f);
        build_fin(m, part, a, b, o, no, nu, nv, pv, bulge * side);
    }
}

/* a thin tube in body units, for barbels and the angelfish's feelers */
static void build_feeler(Mesh *m, int part, const V3 *pts, int np, float r0, float r1) {
    int v0 = m->nv;
    const int ns = 6;
    V3 prev_n = v3(0, 0, 0);
    for (int i = 0; i < np; i++) {
        V3 tan = vnorm(vsub(pts[i < np - 1 ? i + 1 : i], pts[i > 0 ? i - 1 : i]));
        V3 nn = i == 0 ? vperp(tan) : vnorm(vsub(prev_n, vmul(tan, vdot(prev_n, tan))));
        prev_n = nn;
        V3 bn = vcross(tan, nn);
        float r = mixf(r0, r1, (float)i / (np - 1));
        for (int s = 0; s <= ns; s++) {
            float a = (float)s / ns * TAU;
            V3 d = vadd(vmul(nn, cosf(a)), vmul(bn, sinf(a)));
            m_vert(m, vmad(pts[i], d, r), d, (float)i / (np - 1), (float)s / ns, (float)part, 0, 0, 0);
        }
    }
    for (int i = 0; i < np - 1; i++)
        for (int s = 0; s < ns; s++) {
            int a = v0 + i * (ns + 1) + s, b = a + 1, c = a + ns + 1, d = c + 1;
            m_tri(m, a, c, b); m_tri(m, b, c, d);
        }
}

/* -- the species -- */
static const Prof prof_tetra[] = {
    {0, 0, 0, 0}, {0.02f, 0.018f, 0.02f, 0.016f}, {0.06f, 0.045f, 0.05f, 0.032f}, {0.13f, 0.078f, 0.088f, 0.044f},
    {0.24f, 0.108f, 0.12f, 0.05f}, {0.37f, 0.124f, 0.13f, 0.049f}, {0.5f, 0.114f, 0.118f, 0.044f}, {0.62f, 0.092f, 0.094f, 0.036f},
    {0.74f, 0.068f, 0.068f, 0.026f}, {0.85f, 0.05f, 0.05f, 0.018f}, {0.94f, 0.044f, 0.044f, 0.013f}, {1.0f, 0.046f, 0.046f, 0.011f}};
static const Prof prof_angel[] = {
    {0, 0, 0, 0}, {0.035f, 0.05f, 0.035f, 0.02f}, {0.09f, 0.15f, 0.11f, 0.034f}, {0.17f, 0.28f, 0.22f, 0.048f},
    {0.27f, 0.39f, 0.34f, 0.056f}, {0.38f, 0.44f, 0.41f, 0.058f}, {0.5f, 0.41f, 0.39f, 0.053f}, {0.62f, 0.32f, 0.3f, 0.044f},
    {0.75f, 0.2f, 0.19f, 0.032f}, {0.88f, 0.105f, 0.1f, 0.021f}, {1.0f, 0.078f, 0.074f, 0.015f}};
static const Prof prof_discus[] = {
    {0, 0, 0, 0}, {0.03f, 0.075f, 0.055f, 0.022f}, {0.08f, 0.19f, 0.15f, 0.038f}, {0.15f, 0.31f, 0.27f, 0.052f},
    {0.25f, 0.41f, 0.37f, 0.064f}, {0.38f, 0.46f, 0.43f, 0.07f}, {0.5f, 0.46f, 0.43f, 0.068f}, {0.62f, 0.41f, 0.39f, 0.06f},
    {0.74f, 0.31f, 0.3f, 0.046f}, {0.86f, 0.18f, 0.17f, 0.03f}, {0.95f, 0.1f, 0.1f, 0.02f}, {1.0f, 0.08f, 0.08f, 0.015f}};
static const Prof prof_cory[] = {
    {0, 0, 0, 0}, {0.03f, 0.055f, 0.035f, 0.05f}, {0.1f, 0.15f, 0.075f, 0.105f}, {0.2f, 0.235f, 0.105f, 0.135f},
    {0.3f, 0.275f, 0.115f, 0.138f}, {0.42f, 0.265f, 0.115f, 0.125f}, {0.55f, 0.215f, 0.105f, 0.1f}, {0.68f, 0.158f, 0.088f, 0.075f},
    {0.8f, 0.11f, 0.07f, 0.05f}, {0.92f, 0.08f, 0.058f, 0.03f}, {1.0f, 0.07f, 0.055f, 0.024f}};

static BodySpec body_of[SP_N] = {
    {prof_tetra, 12, 1.0f, 1.0f, 1.0f, 0.1f, 0.022f, 0.047f},
    {prof_angel, 11, 1.0f, 1.0f, 0.8f, 0.115f, 0.075f, 0.052f},
    {prof_discus, 12, 1.0f, 1.0f, 0.8f, 0.12f, 0.075f, 0.05f},
    {prof_cory, 11, 0.95f, 0.4f, 0.9f, 0.16f, 0.115f, 0.052f},
};

static void build_species(Mesh *m, int sp, int *body_ni) {
    m->nv = m->ni = 0;
    const BodySpec *b = &body_of[sp];
    int big = sp == SP_ANGEL || sp == SP_DISCUS;
    build_body(m, b, big ? 48 : 34, big ? 32 : 22);
    *body_ni = m->ni;
    V3 z0 = v3(0, 0, 0);
    if (sp == SP_TETRA) {
        V3 d[] = {bp(0.44f, 0.115f, 0), bp(0.47f, 0.25f, 0), bp(0.55f, 0.24f, 0), bp(0.63f, 0.16f, 0), bp(0.62f, 0.095f, 0)};
        build_fin(m, 1, bp(0.44f, ptop(b, 0.44f) - 0.005f, 0), bp(0.6f, ptop(b, 0.6f) - 0.005f, 0), d, 5, 9, 5, z0, 0.002f);
        V3 ad[] = {bp(0.8f, 0.07f, 0), bp(0.83f, 0.1f, 0), bp(0.87f, 0.095f, 0), bp(0.87f, 0.06f, 0)};
        build_fin(m, 6, bp(0.8f, ptop(b, 0.8f) - 0.004f, 0), bp(0.87f, ptop(b, 0.87f) - 0.004f, 0), ad, 4, 4, 2, z0, 0);
        V3 a[] = {bp(0.55f, -0.115f, 0), bp(0.58f, -0.22f, 0), bp(0.7f, -0.17f, 0), bp(0.84f, -0.12f, 0), bp(0.86f, -0.06f, 0)};
        build_fin(m, 3, bp(0.55f, pbot(b, 0.55f) + 0.005f, 0), bp(0.86f, pbot(b, 0.86f) + 0.004f, 0), a, 5, 16, 5, z0, 0.002f);
        V3 c[] = {bp(1.0f, -0.045f, 0), bp(1.14f, -0.13f, 0), bp(1.3f, -0.2f, 0), bp(1.25f, -0.1f, 0), bp(1.14f, 0.0f, 0),
                  bp(1.25f, 0.1f, 0), bp(1.3f, 0.2f, 0), bp(1.14f, 0.13f, 0), bp(1.0f, 0.045f, 0)};
        build_fin(m, 2, bp(0.97f, -0.045f, 0), bp(0.97f, 0.045f, 0), c, 9, 18, 6, z0, 0.003f);
        V3 pc[] = {bp(0.25f, -0.06f, 0.06f), bp(0.31f, -0.09f, 0.1f), bp(0.38f, -0.1f, 0.085f), bp(0.36f, -0.07f, 0.065f)};
        build_pair(m, 4, bp(0.23f, -0.045f, 0.052f), bp(0.25f, -0.07f, 0.05f), pc, 4, 6, 4, 0.002f);
        V3 pv[] = {bp(0.47f, -0.13f, 0.025f), bp(0.53f, -0.19f, 0.04f), bp(0.58f, -0.16f, 0.03f)};
        build_pair(m, 5, bp(0.45f, -0.115f, 0.02f), bp(0.49f, -0.118f, 0.018f), pv, 3, 4, 3, 0.001f);
    } else if (sp == SP_ANGEL) {
        /* the tall swept sails, banded, with trailing tips */
        V3 d[] = {bp(0.3f, 0.44f, 0), bp(0.35f, 0.72f, 0), bp(0.44f, 0.98f, 0), bp(0.64f, 1.2f, 0), bp(1.05f, 1.36f, 0),
                  bp(0.95f, 1.02f, 0), bp(0.91f, 0.6f, 0), bp(0.94f, 0.16f, 0)};
        build_fin(m, 1, bp(0.3f, ptop(b, 0.3f) - 0.01f, 0), bp(0.9f, ptop(b, 0.9f) - 0.008f, 0), d, 8, 20, 9, z0, 0.004f);
        V3 a[] = {bp(0.42f, -0.41f, 0), bp(0.47f, -0.68f, 0), bp(0.56f, -0.92f, 0), bp(0.72f, -1.14f, 0), bp(1.05f, -1.3f, 0),
                  bp(0.95f, -0.98f, 0), bp(0.91f, -0.56f, 0), bp(0.94f, -0.14f, 0)};
        build_fin(m, 3, bp(0.4f, pbot(b, 0.4f) + 0.01f, 0), bp(0.9f, pbot(b, 0.9f) + 0.008f, 0), a, 8, 20, 9, z0, 0.004f);
        V3 c[] = {bp(1.0f, -0.075f, 0), bp(1.2f, -0.2f, 0), bp(1.5f, -0.36f, 0), bp(1.3f, -0.15f, 0), bp(1.24f, 0.0f, 0),
                  bp(1.3f, 0.15f, 0), bp(1.5f, 0.36f, 0), bp(1.2f, 0.2f, 0), bp(1.0f, 0.075f, 0)};
        build_fin(m, 2, bp(0.97f, -0.075f, 0), bp(0.97f, 0.075f, 0), c, 9, 20, 7, z0, 0.004f);
        V3 pc[] = {bp(0.28f, -0.04f, 0.07f), bp(0.36f, -0.08f, 0.11f), bp(0.44f, -0.08f, 0.09f), bp(0.42f, -0.03f, 0.07f)};
        build_pair(m, 4, bp(0.26f, -0.02f, 0.055f), bp(0.27f, -0.07f, 0.055f), pc, 4, 6, 4, 0.003f);
        /* the pelvic feelers: long threads hanging down and back */
        for (int side = -1; side <= 1; side += 2) {
            V3 th[6];
            for (int i = 0; i < 6; i++) {
                float f = i / 5.0f;
                th[i] = bp(0.3f + 0.28f * f * f, pbot(b, 0.3f) - 0.62f * f, 0.015f * side + 0.02f * f * side);
            }
            build_feeler(m, 5, th, 6, 0.006f, 0.0025f);
        }
    } else if (sp == SP_DISCUS) {
        /* the fringe of fin round the back of the disc */
        V3 d[8], a[8];
        for (int i = 0; i < 8; i++) {
            float s = mixf(0.22f, 1.04f, i / 7.0f);
            float ext = 0.1f + 0.05f * sinf(PI * i / 7.0f);
            d[i] = bp(s + 0.03f, ptop(b, fminf(s, 1)) + ext * (1 - 0.6f * smooth(0.8f, 1.05f, s)), 0);
            a[i] = bp(s + 0.03f, pbot(b, fminf(s, 1)) - ext * (1 - 0.6f * smooth(0.8f, 1.05f, s)), 0);
        }
        build_fin(m, 1, bp(0.2f, ptop(b, 0.2f) - 0.01f, 0), bp(0.97f, ptop(b, 0.97f) - 0.008f, 0), d, 8, 22, 4, z0, 0.002f);
        build_fin(m, 3, bp(0.36f, pbot(b, 0.36f) + 0.01f, 0), bp(0.97f, pbot(b, 0.97f) + 0.008f, 0), a, 8, 20, 4, z0, 0.002f);
        V3 c[7];
        for (int i = 0; i < 7; i++) {
            float an = mixf(-1.1f, 1.1f, i / 6.0f);
            c[i] = bp(1.0f + 0.2f * cosf(an), 0.17f * sinf(an), 0);
        }
        build_fin(m, 2, bp(0.97f, -0.08f, 0), bp(0.97f, 0.08f, 0), c, 7, 14, 5, z0, 0.003f);
        V3 pc[] = {bp(0.3f, 0.0f, 0.08f), bp(0.38f, -0.03f, 0.115f), bp(0.45f, -0.04f, 0.09f), bp(0.43f, 0.0f, 0.075f)};
        build_pair(m, 4, bp(0.27f, 0.02f, 0.06f), bp(0.28f, -0.04f, 0.06f), pc, 4, 6, 4, 0.003f);
        V3 pv[] = {bp(0.35f, -0.44f, 0.02f), bp(0.42f, -0.56f, 0.03f), bp(0.45f, -0.46f, 0.02f)};
        build_pair(m, 5, bp(0.33f, -0.42f, 0.015f), bp(0.37f, -0.43f, 0.015f), pv, 3, 4, 3, 0.001f);
    } else {
        /* Corydoras: a tall spined dorsal, an adipose spine, stout pectorals, barbels */
        V3 d[] = {bp(0.3f, 0.3f, 0), bp(0.34f, 0.52f, 0), bp(0.42f, 0.46f, 0), bp(0.48f, 0.3f, 0)};
        build_fin(m, 1, bp(0.3f, ptop(b, 0.3f) - 0.01f, 0), bp(0.48f, ptop(b, 0.48f) - 0.01f, 0), d, 4, 7, 5, z0, 0.002f);
        V3 ad[] = {bp(0.74f, 0.15f, 0), bp(0.77f, 0.2f, 0), bp(0.83f, 0.16f, 0), bp(0.84f, 0.1f, 0)};
        build_fin(m, 6, bp(0.74f, ptop(b, 0.74f) - 0.004f, 0), bp(0.84f, ptop(b, 0.84f) - 0.004f, 0), ad, 4, 3, 2, z0, 0);
        V3 a[] = {bp(0.66f, -0.1f, 0), bp(0.7f, -0.19f, 0), bp(0.8f, -0.15f, 0), bp(0.84f, -0.07f, 0)};
        build_fin(m, 3, bp(0.66f, pbot(b, 0.66f) + 0.004f, 0), bp(0.84f, pbot(b, 0.84f) + 0.004f, 0), a, 4, 7, 4, z0, 0.001f);
        V3 c[] = {bp(1.0f, -0.055f, 0), bp(1.18f, -0.15f, 0), bp(1.32f, -0.2f, 0), bp(1.22f, -0.07f, 0), bp(1.15f, 0.0f, 0),
                  bp(1.22f, 0.08f, 0), bp(1.32f, 0.22f, 0), bp(1.18f, 0.16f, 0), bp(1.0f, 0.065f, 0)};
        build_fin(m, 2, bp(0.97f, -0.055f, 0), bp(0.97f, 0.065f, 0), c, 9, 16, 6, z0, 0.003f);
        V3 pc[] = {bp(0.22f, -0.1f, 0.2f), bp(0.32f, -0.12f, 0.26f), bp(0.4f, -0.11f, 0.2f), bp(0.36f, -0.09f, 0.15f)};
        build_pair(m, 4, bp(0.18f, -0.07f, 0.12f), bp(0.21f, -0.1f, 0.12f), pc, 4, 6, 3, 0.003f);
        V3 pv[] = {bp(0.5f, -0.13f, 0.08f), bp(0.58f, -0.15f, 0.1f), bp(0.62f, -0.12f, 0.06f)};
        build_pair(m, 5, bp(0.47f, -0.11f, 0.05f), bp(0.52f, -0.11f, 0.05f), pv, 3, 4, 3, 0.001f);
        for (int side = -1; side <= 1; side += 2)
            for (int k = 0; k < 2; k++) {
                V3 th[4];
                for (int i = 0; i < 4; i++) {
                    float f = i / 3.0f;
                    th[i] = bp(0.02f + 0.02f * k - 0.03f * f, -0.03f - 0.08f * f - 0.02f * k, (0.02f + 0.03f * k + 0.04f * f) * side);
                }
                build_feeler(m, 7, th, 4, 0.008f, 0.004f);
            }
    }
}

/* ==================================================================== */
/* ---- who lives here ---- */
typedef struct {
    int sp;
    float size;                         /* standard length, m */
    uint32_t seed;
    float pref_y, pref_z;               /* where in the water it likes to be */
    float slot[3];                      /* a school's member: its place in the school */
} Fish;
#define MAXFISH 96
static Fish fish[MAXFISH];
static int nfish, nsp[SP_N], sp_first[SP_N];

/* the simulated state of a fish */
typedef struct {
    float p[3], v[3];
    float f[3];                         /* heading */
    float roll, bend, phase, amp, pph, pamp, pitch_bias;
    float yaw_rate;
} Agent;
typedef struct {
    int chapter;                        /* -1: free */
    double t;                           /* the time of cur */
    Agent prev[MAXFISH], cur[MAXFISH];
} Sim;
#define SIM_DT (1.0 / 30)
#define CH_T 24.0                       /* chapters: each sim runs from (k - 1) CH_T, shown from k CH_T */
#define CH_BLEND 4.0
static Sim sims[3];


/* where a fish would like to be at time t: closed-form wandering, so every
 * instance (and every chapter's sim) agrees */
static V3 school_centre(double t, float night) {
    float w = tank_w();
    double ts = t * (1 - 0.6f * night);
    float x = (X0 + X1) / 2 + w * 0.36f * (0.72f * sinf((float)fmod(ts * 0.021 + 1.3, TAU)) + 0.28f * tnoise(opt_seed + 11, ts / 17.0));
    float y = YW * (0.52f + 0.12f * sinf((float)fmod(ts * 0.047, TAU)) + 0.06f * tnoise(opt_seed + 12, ts / 9.0));
    float z = TD * (0.27f + 0.08f * sinf((float)fmod(ts * 0.033 + 2.0, TAU)) + 0.05f * tnoise(opt_seed + 13, ts / 11.0));
    y = mixf(y, YW * 0.3f, night);
    return v3(x, y, z);
}

static V3 fish_target(int i, double t, float night) {
    Fish *F = &fish[i];
    uint32_t k = F->seed;
    float w = tank_w();
    if (F->sp == SP_TETRA) {
        V3 c = school_centre(t, night);
        float spread = 1 + 0.6f * night;
        V3 o = v3(F->slot[0] * spread, F->slot[1] * spread, F->slot[2] * spread);
        o.x += 0.025f * tnoise(k, t / 3.1);
        o.y += 0.012f * tnoise(k + 1, t / 2.7);
        o.z += 0.02f * tnoise(k + 2, t / 3.7);
        return vadd(c, o);
    }
    if (F->sp == SP_CORY) {
        /* a loose group working along the front of the sand */
        double ts = t * (1 - 0.7f * night);
        float gx = (X0 + X1) / 2 + w * 0.34f * tnoise(opt_seed + 31, ts / 41.0);
        float gz = TD * (0.17f + 0.08f * tnoise(opt_seed + 32, ts / 23.0));
        float x = gx + F->slot[0] + 0.03f * tnoise(k, ts / 7.0);
        float z = clampf(gz + F->slot[2] + 0.02f * tnoise(k + 1, ts / 6.0), 0.035f, TD * 0.5f);
        float y = ground_at(x, z) + F->size * 0.12f;
        /* now and then one darts up for a gulp of air */
        double per = 230 + 140 * hf(k, 5, 5);
        double ph = fmod(t + per * hf(k, 6, 6), per);
        if (night < 0.5f && ph < 7) {
            float f = (float)(ph / 7);
            float up = sinf(PI * f);
            y = mixf(y, YW - 0.012f, powf(up, 0.6f));
        }
        return v3(x, y, z);
    }
    /* angelfish and discus: slow wandering over the midwater */
    double ts = t * (1 - 0.6f * night);
    float sc = F->sp == SP_DISCUS ? 0.7f : 1.0f;
    float x = (X0 + X1) / 2 + w * 0.42f * sc * tnoise(k, ts / 23.0) + w * 0.12f * tnoise(k + 9, ts / 7.0);
    float y = F->pref_y + YW * 0.12f * tnoise(k + 1, ts / 13.0);
    float z = F->pref_z + TD * 0.12f * tnoise(k + 2, ts / 17.0);
    y = mixf(y, YW * 0.35f, night * 0.6f);
    return v3(x, y, z);
}

typedef struct { float cruise, vmax, amax, turn, lam; } SpecMotion;
static const SpecMotion motion[SP_N] = {
    {0.055f, 0.32f, 1.6f, 6.0f, 0.95f},
    {0.035f, 0.14f, 0.45f, 1.6f, 1.2f},
    {0.03f, 0.12f, 0.35f, 1.3f, 1.2f},
    {0.03f, 0.18f, 0.9f, 3.5f, 0.95f},
};

static void sim_init(Sim *S, int chapter) {
    S->chapter = chapter;
    double t0 = (chapter - 1) * CH_T;
    S->t = t0;
    float night = lit.night;
    for (int i = 0; i < nfish; i++) {
        Agent *a = &S->cur[i];
        memset(a, 0, sizeof *a);
        V3 p = fish_target(i, t0, night), p2 = fish_target(i, t0 + 0.5, night);
        V3 v = vmul(vsub(p2, p), 2);
        a->p[0] = p.x; a->p[1] = p.y; a->p[2] = p.z;
        a->v[0] = v.x; a->v[1] = v.y; a->v[2] = v.z;
        V3 f = vlen(v) > 1e-4f ? vnorm(v) : v3(hf(fish[i].seed, 3, 3) < 0.5f ? 1 : -1, 0, 0);
        f.y *= 0.3f;
        f = vnorm(f);
        a->f[0] = f.x; a->f[1] = f.y; a->f[2] = f.z;
        a->phase = hf(fish[i].seed, 4, 4) * TAU;
        a->pph = hf(fish[i].seed, 5, 4) * TAU;
        a->amp = 0.05f;
        S->prev[i] = *a;
    }
}

static float br_box[MAXBR][6];
static void branch_boxes(void) {
    for (int i = 0; i < nbr; i++) {
        float *bx = br_box[i];
        bx[0] = bx[1] = bx[2] = 1e9f; bx[3] = bx[4] = bx[5] = -1e9f;
        for (int k = 0; k < br[i].n; k++) {
            V3 p = br[i].p[k];
            bx[0] = fminf(bx[0], p.x); bx[1] = fminf(bx[1], p.y); bx[2] = fminf(bx[2], p.z);
            bx[3] = fmaxf(bx[3], p.x); bx[4] = fmaxf(bx[4], p.y); bx[5] = fmaxf(bx[5], p.z);
        }
    }
}

/* push away from the glass, the surface, the ground, stones, wood, plants */
static V3 avoid(V3 p, float r) {
    V3 a = v3(0, 0, 0);
    float m = 0.035f + r;
    if (p.x < X0 + m) a.x += (X0 + m - p.x) * 30;
    if (p.x > X1 - m) a.x -= (p.x - X1 + m) * 30;
    if (p.z < 0.02f + r) a.z += (0.02f + r - p.z) * 40;
    if (p.z > TD - 0.03f - r) a.z -= (p.z - TD + 0.03f + r) * 40;
    if (p.y > YW - 0.02f - r) a.y -= (p.y - YW + 0.02f + r) * 40;
    float g = ground_at(p.x, p.z) + 0.012f + r * 0.5f;
    if (p.y < g) a.y += (g - p.y) * 60;
    for (int i = 0; i < nstone; i++) {
        V3 d = vsub(p, stones[i].c);
        float rr = fmaxf(stones[i].ax.x, stones[i].ax.z) * 1.2f + r + 0.01f;
        float l = vlen(d);
        if (l < rr && l > 1e-5f) a = vmad(a, d, (rr - l) / l * 25);
    }
    for (int bi = 0; bi < nbr; bi++) {
        const Branch *b = &br[bi];
        if (p.x < br_box[bi][0] - r - 0.03f || p.x > br_box[bi][3] + r + 0.03f || p.y < br_box[bi][1] - r - 0.03f ||
            p.y > br_box[bi][4] + r + 0.03f || p.z < br_box[bi][2] - r - 0.03f || p.z > br_box[bi][5] + r + 0.03f) continue;
        for (int k = 0; k < b->n - 1; k += 2) {
            V3 s0 = b->p[k], s1 = b->p[k + 2 < b->n ? k + 2 : b->n - 1];
            V3 sd = vsub(s1, s0);
            float tt = clampf(vdot(vsub(p, s0), sd) / fmaxf(vdot(sd, sd), 1e-8f), 0, 1);
            V3 d = vsub(p, vmad(s0, sd, tt));
            float rr = b->r0 + r + 0.015f, l = vlen(d);
            if (l < rr && l > 1e-5f) a = vmad(a, d, (rr - l) / l * 25);
        }
    }
    for (int i = 0; i < nclump; i++) {
        const Clump *c = &clumps[i];
        if (p.y > ground_at(c->x, c->z) + c->h) continue;
        float dx = p.x - c->x, dz = p.z - c->z, l = sqrtf(dx * dx + dz * dz), rr = c->r + r;
        if (l < rr && l > 1e-5f) { a.x += dx / l * (rr - l) * 12; a.z += dz / l * (rr - l) * 12; }
    }
    return a;
}

static void sim_step(Sim *S, double wt) {
    float dt = (float)SIM_DT;
    double t = S->t + SIM_DT;
    float night = lit.night;
    float act = 0.6f + 0.8f * cpu_load(wt);
    memcpy(S->prev, S->cur, sizeof(Agent) * (size_t)nfish);
    for (int i = 0; i < nfish; i++) {
        Fish *F = &fish[i];
        const SpecMotion *M = &motion[F->sp];
        Agent *a = &S->cur[i];
        V3 p = v3(a->p[0], a->p[1], a->p[2]), v = v3(a->v[0], a->v[1], a->v[2]);
        V3 tg = fish_target(i, t, night), tg2 = fish_target(i, t + 0.25, night);
        V3 tv = vmul(vsub(tg2, tg), 4);
        float kp = F->sp == SP_TETRA ? 1.1f : F->sp == SP_CORY ? 1.6f : 0.6f;
        V3 vd = vadd(tv, vmul(vsub(tg, p), kp));
        V3 acc = vmul(vsub(vd, v), 1.8f);
        /* schooling: keep apart, match the neighbours' heading */
        if (F->sp == SP_TETRA || F->sp == SP_CORY) {
            V3 sep = v3(0, 0, 0), ali = v3(0, 0, 0);
            int na = 0;
            float rs = F->sp == SP_TETRA ? 0.022f : 0.035f;
            for (int j = sp_first[F->sp]; j < sp_first[F->sp] + nsp[F->sp]; j++) {
                if (j == i) continue;
                Agent *b = &S->cur[j];
                V3 d = vsub(p, v3(b->p[0], b->p[1], b->p[2]));
                float l2 = vdot(d, d);
                if (l2 < rs * rs && l2 > 1e-10f) sep = vmad(sep, d, (rs - sqrtf(l2)) / sqrtf(l2) * 40);
                if (l2 < 0.07f * 0.07f) { ali = vadd(ali, v3(b->v[0], b->v[1], b->v[2])); na++; }
            }
            acc = vadd(acc, sep);
            if (na) acc = vmad(acc, vsub(vmul(ali, 1.0f / na), v), 0.8f);
        } else {
            /* the big fish give each other room */
            for (int j = 0; j < nfish; j++) {
                if (j == i || fish[j].sp == SP_TETRA || fish[j].sp == SP_CORY) continue;
                Agent *b = &S->cur[j];
                V3 d = vsub(p, v3(b->p[0], b->p[1], b->p[2]));
                float l = vlen(d), rr = (F->size + fish[j].size) * 0.9f;
                if (l < rr && l > 1e-5f) acc = vmad(acc, d, (rr - l) / l * 6);
            }
        }
        acc = vadd(acc, avoid(p, F->size * (F->sp == SP_ANGEL ? 0.6f : 0.4f)));
        /* darts: a sudden burst now and then, more when the machine is busy */
        if (F->sp == SP_TETRA) {
            double slot = floor(t / 1.7 + hf(F->seed, 7, 7));
            float r = hf(F->seed, (uint32_t)(int64_t)slot, 8);
            float ft = (float)(t / 1.7 + hf(F->seed, 7, 7) - slot);
            if (r < 0.05f * act * (1 - night) && ft < 0.22f) {
                V3 dir = vnorm(vadd(v3(a->f[0], a->f[1] * 0.3f, a->f[2]), v3(hf(F->seed, (uint32_t)slot, 9) - 0.5f, 0, hf(F->seed, (uint32_t)slot, 10) - 0.5f)));
                acc = vmad(acc, dir, 9.0f * (1 - ft / 0.22f));
            }
        }
        float al = vlen(acc), amax = M->amax * (1 + 3 * (F->sp == SP_TETRA));
        if (al > amax) acc = vmul(acc, amax / al);
        v = vmad(v, acc, dt);
        float sp = vlen(v), vmax = M->vmax * (1 - 0.5f * night);
        if (sp > vmax) { v = vmul(v, vmax / sp); sp = vmax; }
        v = vmul(v, 1 - 0.15f * dt);
        p = vmad(p, v, dt);
        /* heading: turn toward the velocity at a limited rate; hovering, keep it */
        V3 f = v3(a->f[0], a->f[1], a->f[2]);
        V3 want = sp > 0.008f ? vnorm(v) : f;
        if (F->sp == SP_CORY && night < 0.5f && sp < 0.05f) {
            /* grazing: nose down into the sand */
            want = vnorm(v3(want.x, -0.45f, want.z));
        }
        float maxp = F->sp == SP_CORY ? 0.9f : F->sp == SP_TETRA ? 0.5f : 0.22f;
        if (want.y > maxp) want = vnorm(v3(want.x, maxp, want.z));
        if (want.y < -maxp) want = vnorm(v3(want.x, -maxp, want.z));
        /* yaw: heading angle in the horizontal, turned at a limited rate */
        float psi = atan2f(f.z, f.x), psw = atan2f(want.z, want.x);
        float dpsi = fmodf(psw - psi + 3 * PI, TAU) - PI;
        float maxt = M->turn * dt * (sp > 0.02f ? 1 : 0.5f);
        float dyaw = clampf(dpsi, -maxt, maxt);
        psi += dyaw;
        float py = mixf(f.y, want.y, clampf(dt * 2.5f, 0, 1));
        float ch = sqrtf(fmaxf(1 - py * py, 0));
        f = v3(cosf(psi) * ch, py, sinf(psi) * ch);
        a->yaw_rate = mixf(a->yaw_rate, dyaw / dt, 0.2f);
        a->f[0] = f.x; a->f[1] = f.y; a->f[2] = f.z;
        a->p[0] = p.x; a->p[1] = p.y; a->p[2] = p.z;
        a->v[0] = v.x; a->v[1] = v.y; a->v[2] = v.z;
        /* the swimming: tail beat from the speed along the heading */
        float fwd = fmaxf(vdot(v, f), 0);
        float bl = fwd / F->size;
        float freq = (F->sp == SP_TETRA ? 2.5f : 1.4f) + (F->sp == SP_TETRA ? 2.2f : 1.6f) * bl;
        float ta = vlen(acc);
        if (freq > 14) freq = 14;
        a->phase += TAU * freq * dt;
        if (a->phase > 1000) a->phase -= 160 * PI;
        float amp = 0.035f + 0.05f * clampf(bl / 3, 0, 1) + 0.02f * clampf(ta / 2, 0, 1);
        a->amp = mixf(a->amp, amp, 0.15f);
        a->bend = mixf(a->bend, clampf(a->yaw_rate * 0.07f, -0.3f, 0.3f), 0.25f);
        float rmax = F->sp == SP_ANGEL || F->sp == SP_DISCUS ? 0.12f : 0.35f;
        a->roll = mixf(a->roll, clampf(a->yaw_rate * fwd * 1.5f, -rmax, rmax), 0.1f);
        /* pectorals scull when hovering, fold when cruising */
        float hover = 1 - clampf(bl / 1.5f, 0, 1);
        a->pph += TAU * (1.2f + 2.5f * hover + 0.5f * bl) * dt;
        if (a->pph > 1000) a->pph -= 160 * PI;
        a->pamp = mixf(a->pamp, 0.15f + 0.55f * hover, 0.1f);
    }
    S->t = t;
}

static void sim_advance(Sim *S, double t, double wt) {
    int n = 0;
    while (S->t + SIM_DT <= t && n < 100000) { sim_step(S, wt); n++; }
}

static Sim *sim_get(int chapter) {
    for (int i = 0; i < 3; i++) if (sims[i].chapter == chapter) return &sims[i];
    /* reuse the oldest */
    int o = 0;
    for (int i = 1; i < 3; i++) if (sims[i].chapter < sims[o].chapter) o = i;
    sim_init(&sims[o], chapter);
    return &sims[o];
}

static float ang_mix(float a, float b, float w) {
    float d = fmodf(b - a, TAU);
    if (d > PI) d -= TAU;
    if (d < -PI) d += TAU;
    return a + d * w;
}

/* the agents as shown at time t */
static Agent shown[MAXFISH];
static void agents_at(double t, double wt) {
    int k = (int)floor(t / CH_T);
    if (k < 0) k = 0;
    double into = t - k * CH_T;
    Sim *cur = sim_get(k);
    Sim *old = into < CH_BLEND && k > 0 ? sim_get(k - 1) : NULL;
    /* warm the next chapter's sim alongside */
    Sim *next = sim_get(k + 1);
    sim_advance(cur, t, wt);
    if (old) sim_advance(old, t, wt);
    sim_advance(next, t, wt);
    float w = old ? smooth(0, (float)CH_BLEND, (float)into) : 1;
    for (int i = 0; i < nfish; i++) {
        Agent A[2];
        Sim *src[2] = {cur, old};
        for (int s = 0; s < (old ? 2 : 1); s++) {
            Sim *S = src[s];
            float f = clampf((float)((t - (S->t - SIM_DT)) / SIM_DT), 0, 1);
            Agent *p = &S->prev[i], *c = &S->cur[i], *o = &A[s];
            *o = *c;
            for (int k2 = 0; k2 < 3; k2++) { o->p[k2] = mixf(p->p[k2], c->p[k2], f); o->f[k2] = mixf(p->f[k2], c->f[k2], f); }
            o->phase = mixf(p->phase, c->phase, f);
            o->pph = mixf(p->pph, c->pph, f);
        }
        Agent *o = &shown[i];
        *o = A[0];
        if (old) {
            for (int k2 = 0; k2 < 3; k2++) { o->p[k2] = mixf(A[1].p[k2], A[0].p[k2], w); o->f[k2] = mixf(A[1].f[k2], A[0].f[k2], w); }
            o->phase = ang_mix(A[1].phase, A[0].phase, w);
            o->pph = ang_mix(A[1].pph, A[0].pph, w);
            o->amp = mixf(A[1].amp, A[0].amp, w);
            o->bend = mixf(A[1].bend, A[0].bend, w);
            o->roll = mixf(A[1].roll, A[0].roll, w);
            o->pamp = mixf(A[1].pamp, A[0].pamp, w);
        }
    }
}

static void roster(void) {
    float w = tank_w();
    nfish = 0;
    int counts[SP_N];
    counts[SP_TETRA] = (int)clampf(roundf(34 * w / 0.9f), 18, 64);
    counts[SP_ANGEL] = w > 1.2f ? 4 : 3;
    counts[SP_DISCUS] = 2;
    counts[SP_CORY] = (int)clampf(roundf(6 * w / 0.9f), 5, 9);
    const float size[SP_N] = {0.03f, 0.08f, 0.11f, 0.038f};
    for (int sp = 0; sp < SP_N; sp++) {
        sp_first[sp] = nfish;
        nsp[sp] = counts[sp];
        for (int i = 0; i < counts[sp] && nfish < MAXFISH; i++) {
            Fish *F = &fish[nfish++];
            F->sp = sp;
            F->seed = hash32(opt_seed * 977u + (uint32_t)nfish * 7919u);
            F->size = size[sp] * frange(0.88f, 1.1f);
            F->pref_y = YW * frange(0.4f, 0.7f);
            F->pref_z = TD * frange(0.3f, 0.55f);
            if (sp == SP_DISCUS) { F->pref_y = YW * frange(0.35f, 0.55f); F->pref_z = TD * frange(0.35f, 0.5f); }
            /* a member's place: an ellipsoid stretched along the school */
            float r = 0.022f * cbrtf((float)counts[sp]);
            V3 o;
            do o = v3(frange(-1, 1), frange(-1, 1), frange(-1, 1)); while (vdot(o, o) > 1);
            if (sp == SP_TETRA) { F->slot[0] = o.x * r * 1.9f; F->slot[1] = o.y * r * 0.7f; F->slot[2] = o.z * r * 1.1f; }
            else { F->slot[0] = o.x * 0.09f; F->slot[1] = 0; F->slot[2] = o.z * 0.05f; }
        }
    }
    for (int i = 0; i < 3; i++) sims[i].chapter = -1000;
}

/* ---- the fish shaders ---- */
static const char *VS_FISH = GLSL(
    layout(location = 0) in vec3 aP;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aUV;
    layout(location = 3) in float aMat;
    layout(location = 4) in vec3 aAux;
    layout(location = 5) in vec4 iA;    /* position, length */
    layout(location = 6) in vec4 iB;    /* heading, roll */
    layout(location = 7) in vec4 iC;    /* tail phase, amplitude, bend, pectoral phase */
    layout(location = 8) in vec4 iD;    /* pectoral amplitude, seed, colour fade, - */
    uniform float uLam;
    uniform float uEnv0;
    uniform int uSpec;
    out vec3 vP; out vec3 vN; out vec3 vG; out vec3 vRest; out vec3 vAux; out vec2 vUV;
    flat out vec4 vInst;
    flat out float vPart;
    vec3 rot(vec3 v, vec3 k, float a) { float c = cos(a), s = sin(a); return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c); }
    float lat_at(float s, float A, float ph, float bend) {
        float env = uEnv0 + (1.0 - uEnv0) * pow(clamp((s - 0.12) / 0.88, 0.0, 1.0), 1.6);
        if (s > 1.0) env = 1.0 + (s - 1.0) * 1.3;
        return A * env * sin(ph - 6.2831853 * s / uLam) + bend * (s - 0.38) * (s - 0.38);
    }
    void main() {
        vec3 p = aP, n = aN;
        int part = int(aMat + 0.5);
        float A = iC.y, ph = iC.x, bend = iC.z;
        if (part == 4) {
            /* pectorals row about their base: forward and back, feathering */
            vec3 pv = aAux;
            float side = pv.z > 0.0 ? 1.0 : -1.0;
            float ang = iD.x * sin(iC.w + (side > 0.0 ? 0.0 : 0.5)) * 0.9 - 0.35 * (1.0 - iD.x);
            vec3 ax = normalize(vec3(0.2, 1.0, 0.3 * side));
            p = pv + rot(p - pv, ax, ang * side);
            n = rot(n, ax, ang * side);
        }
        float s = 0.4 - p.x;
        float k = 6.2831853 / uLam;
        float e0 = clamp((s - 0.12) / 0.88, 0.0, 1.0);
        float env = uEnv0 + (1.0 - uEnv0) * e0 * sqrt(e0) * sqrt(sqrt(e0)) * 1.0;
        float denv = s > 0.12 && s < 1.0 ? (1.0 - uEnv0) * 1.6 * pow(e0, 0.6) / 0.88 : 0.0;
        if (s > 1.0) { env = 1.0 + (s - 1.0) * 1.3; denv = 1.3; }
        float sn0 = sin(ph - k * s), cs0 = cos(ph - k * s);
        float lat = A * env * sn0 + bend * (s - 0.38) * (s - 0.38);
        float dl = A * (denv * sn0 - env * k * cs0) + 2.0 * bend * (s - 0.38);
        if (part == 1 || part == 3 || part == 6) lat += aUV.y * aUV.y * 0.02 * sin(ph * 0.6 - aUV.x * 7.0 + iD.y * 6.0);
        if (part == 2) lat += aUV.y * 0.012 * sin(ph - 1.5 - aUV.x * 3.0);
        if (part == 5) lat += aUV.y * aUV.y * 0.03 * sin(ph * 0.5 - 1.0);
        if (part == 5 && uSpec == 1) lat += aUV.x * aUV.x * 0.08 * sin(ph * 0.3 + iD.y * 3.0);
        float th = atan(dl);
        float c = cos(th), sn = sin(th);
        vec3 q = vec3(p.x + p.z * sn, p.y, lat + p.z * c);
        vec3 nq = vec3(n.x * c + n.z * sn, n.y, -n.x * sn + n.z * c);
        vec3 F = normalize(iB.xyz);
        vec3 R = normalize(cross(F, vec3(0.0, 1.0, 0.0)));
        vec3 Up = cross(R, F);
        float r = iB.w;
        vec3 U2 = Up * cos(r) + R * sin(r), R2 = R * cos(r) - Up * sin(r);
        vec3 P = iA.xyz + iA.w * (q.x * F + q.y * U2 + q.z * R2);
        vP = P;
        vN = normalize(nq.x * F + nq.y * U2 + nq.z * R2);
        vRest = aP; vAux = aAux; vUV = aUV;
        vInst = iD; vPart = float(part);
        gl_Position = place(P, vG);
    });

static const char *FS_FISH = GLSL(
    in vec3 vP; in vec3 vN; in vec3 vG; in vec3 vRest; in vec3 vAux; in vec2 vUV;
    flat in vec4 vInst;
    flat in float vPart;
    uniform int uSpec;
    uniform vec3 uEye;
    uniform float uFrame;
    /* structural colour: what a reflecting layer does as the angle changes */
    vec3 film(float t, vec3 a, vec3 b) { return mix(b, a, smoothstep(0.1, 0.85, t)); }
    /* overlapping scales: a darker rim at the back edge of each */
    float scales(float s, float y, float n) {
        vec2 q = vec2(s * n, y * n * 0.3);
        q.y += 0.5 * floor(q.x);
        vec2 f = fract(q) - vec2(0.2, 0.5);
        return smoothstep(0.3, 0.55, length(f * vec2(1.3, 1.0)));
    }
    void main() {
        int part = int(vPart + 0.5);
        vec3 N = normalize(vN);
        if (!gl_FrontFacing) N = -N;
        vec3 V = normalize(vG - vP);
        float s = 0.4 - vRest.x, y = vRest.y;
        float yn = vAux.y;
        float seed = vInst.y, fade = vInst.z;
        vec3 alb = vec3(0.5), irid = vec3(0.0);
        float spec = 0.3, rough = 60.0, alpha = 1.0, silver = 0.0, trans = 0.0;
        float nv = clamp(dot(N, V), 0.0, 1.0);
        float rays = 0.0;
        if (part != 0 && part != 7) {
            float nr = part == 2 ? 18.0 : part == 4 ? 7.0 : part == 5 ? 4.0 : 12.0;
            float rd = abs(fract(vUV.x * nr) - 0.5) * 2.0;
            rays = smoothstep(0.55, 0.95, rd) * (1.0 - 0.5 * vUV.y);
        }
        if (uSpec == 0) {
            /* neon tetra: olive back, the electric line, red behind, silver belly */
            if (part == 0) {
                float yc = 0.14 + 0.36 * smoothstep(0.5, 0.86, s);
                float hw = 0.2 * (1.0 - 0.5 * smoothstep(0.6, 0.86, s)) * smoothstep(0.05, 0.16, s);
                float blue = smoothstep(hw, hw * 0.5, abs(yn - yc)) * smoothstep(0.08, 0.14, s) * (1.0 - smoothstep(0.82, 0.87, s));
                float red = smoothstep(0.36, 0.5, s) * (1.0 - smoothstep(0.96, 1.0, s)) * smoothstep(yc - hw * 0.4, yc - hw * 1.1, yn) * smoothstep(-0.98, -0.8, yn);
                vec3 back = mix(vec3(0.3, 0.27, 0.15), vec3(0.11, 0.1, 0.06), smoothstep(0.6, 0.95, yn));
                vec3 belly = vec3(0.5, 0.52, 0.54);
                alb = mix(belly, back, smoothstep(yc - 0.05, yc + 0.25, yn));
                silver = (1.0 - smoothstep(yc - 0.3, yc, yn)) * 0.55;
                alb = mix(alb, vec3(0.6, 0.02, 0.03) * (1.0 - 0.65 * fade), red);
                silver *= 1.0 - red;
                irid = film(nv, vec3(0.02, 0.22, 1.0), vec3(0.04, 0.62, 0.95)) * blue * 1.6 * (1.0 - 0.8 * fade);
                alb = mix(alb, vec3(0.02, 0.04, 0.09), blue);
                alb *= 0.88 + 0.12 * scales(s, yn, 34.0);
                float gill = 1.0 - smoothstep(0.0, 0.01, abs(s - 0.19 - 0.04 * yn * yn));
                alb *= 1.0 - 0.3 * gill * step(yn, 0.4);
                /* the mouth */
                alb *= 1.0 - 0.6 * (1.0 - smoothstep(0.0, 0.06, abs(yn + 0.05))) * (1.0 - smoothstep(0.02, 0.045, s));
                trans = 0.35 * smoothstep(0.84, 1.0, s);
            } else {
                alpha = 0.1 + 0.28 * rays;
                alb = vec3(0.8, 0.82, 0.85);
                if (part == 3) { alpha += 0.45 * (1.0 - smoothstep(0.0, 0.1, vUV.x)) * smoothstep(0.2, 0.8, vUV.y); alb = vec3(0.9); }
                if (part == 6) alpha = 0.45;
                if (part == 2) { alb = mix(alb, vec3(0.75, 0.15, 0.1), 0.5 * (1.0 - smoothstep(0.0, 0.35, vUV.y))); alpha += 0.15 * (1.0 - vUV.y); }
                trans = 0.6;
            }
        } else if (uSpec == 1) {
            /* angelfish: silver, four black bars running on into the fins */
            float b1 = smoothstep(0.034, 0.022, abs(s - 0.115 - 0.04 * y));
            float b2 = smoothstep(0.055, 0.038, abs(s - 0.35 - 0.05 * y));
            float b3 = smoothstep(0.05, 0.034, abs(s - 0.63 - 0.08 * y));
            float b4 = smoothstep(0.032, 0.02, abs(s - 0.91));
            float bar = max(max(b1, b2), max(b3, b4));
            if (part == 0) {
                alb = vec3(0.6, 0.6, 0.58);
                alb = mix(alb, vec3(0.52, 0.45, 0.28), smoothstep(0.4, 0.9, yn) * (1.0 - smoothstep(0.25, 0.5, s)));
                alb = mix(alb, vec3(0.025), bar * 0.95);
                silver = 0.6 * (1.0 - bar);
                alb *= 0.92 + 0.08 * scales(s, yn, 46.0);
                spec = 0.45;
            } else if (part == 5) {
                alb = mix(vec3(0.55, 0.55, 0.5), vec3(0.85, 0.85, 0.8), vUV.x);
                alpha = 0.95;
            } else {
                float sy = abs(y);
                /* the long sails: grey, banded, darker at the trailing edge */
                alb = mix(vec3(0.34, 0.35, 0.33), vec3(0.02), bar * 0.92);
                alb = mix(alb, vec3(0.04), 0.6 * smoothstep(0.5, 0.8, vUV.y) * (1.0 - smoothstep(0.86, 0.97, vUV.y)));
                alb = mix(alb, vec3(0.03), 0.7 * (1.0 - smoothstep(0.0, 0.06, vUV.x)));
                alpha = 0.5 + 0.2 * rays + 0.4 * bar - 0.25 * smoothstep(0.85, 1.0, vUV.y);
                if (part == 4) { alpha = 0.12 + 0.2 * rays; alb = vec3(0.8); }
                if (part == 2) { alpha = 0.3 + 0.3 * rays; alb = mix(vec3(0.5), vec3(0.12), 0.5 * rays); }
                trans = 0.4;
                (void(sy));
            }
        } else if (uSpec == 2) {
            /* discus: fine turquoise vermiculation over red-brown */
            float w1 = sin(yn * 34.0 + 2.6 * sin(s * 15.0 + seed * 7.0) + 1.4 * sin(s * 33.0 - yn * 9.0 + seed));
            float line = smoothstep(0.55, 0.92, w1) * 0.85;
            float head = 1.0 - smoothstep(0.15, 0.35, s);
            line = max(line, head * smoothstep(0.0, 0.6, w1) * 0.8);
            float bars = smoothstep(0.6, 1.0, 0.5 + 0.5 * cos((s - 0.05) * 6.2831 * 4.5));
            if (part == 0) {
                alb = vec3(0.5, 0.17, 0.06) * (1.0 - 0.3 * bars * smoothstep(0.1, 0.3, s));
                irid = film(nv, vec3(0.03, 0.3, 0.38), vec3(0.1, 0.5, 0.35)) * line * 0.6;
                alb = mix(alb, vec3(0.1, 0.2, 0.2), line * 0.65);
                alb *= 0.94 + 0.06 * scales(s, yn, 60.0);
                spec = 0.45;
            } else {
                float fy = vUV.y;
                alb = mix(vec3(0.45, 0.16, 0.06), vec3(0.08, 0.25, 0.28), line * 0.6);
                irid = vec3(0.04, 0.4, 0.5) * line * 0.5 * (1.0 - fy);
                alb = mix(alb, vec3(0.03), smoothstep(0.62, 0.72, fy) * (1.0 - smoothstep(0.78, 0.86, fy)));
                alb = mix(alb, vec3(0.75, 0.15, 0.05), smoothstep(0.82, 0.9, fy));
                alpha = 0.9 - 0.3 * fy;
                if (part == 2 || part == 4) { alpha = 0.25 + 0.25 * rays; alb = vec3(0.55, 0.3, 0.15); irid = vec3(0.0); }
                if (part == 5) { alpha = 0.8; alb = vec3(0.6, 0.1, 0.05); }
                trans = 0.5;
            }
        } else {
            /* panda Corydoras: pale, a black mask, a black sail and a tail spot */
            if (part == 0) {
                alb = vec3(0.64, 0.42, 0.29);
                alb = mix(alb, vec3(0.7, 0.52, 0.42), smoothstep(-0.2, -0.7, yn));
                float mask = smoothstep(0.085, 0.06, abs(s - 0.165 + 0.15 * y)) * smoothstep(-0.07, -0.03, y);
                float spot = smoothstep(1.0, 0.75, length(vec2((s - 0.915) / 0.075, (yn - 0.05) / 0.85)));
                float dors = smoothstep(0.11, 0.06, abs(s - 0.39)) * smoothstep(0.75, 0.95, yn) * 0.8;
                alb = mix(alb, vec3(0.02, 0.02, 0.025), max(max(mask, spot), dors));
                /* the bony plates: two rows meeting along the flank */
                float seam = 1.0 - smoothstep(0.0, 0.05, abs(yn - 0.05));
                float plate = 1.0 - smoothstep(0.0, 0.07, abs(fract(s * 21.0 + yn * 0.8) - 0.5) * 2.0 - 0.88);
                alb *= 1.0 - 0.18 * max(seam, plate * smoothstep(0.25, 0.4, s) * (1.0 - smoothstep(0.85, 0.95, s)) * smoothstep(-0.8, -0.4, yn));
                silver = 0.08;
                irid = film(nv, vec3(0.2, 0.16, 0.2), vec3(0.08, 0.25, 0.18)) * 0.12;
                spec = 0.12;
            } else if (part == 7) {
                alb = vec3(0.8, 0.72, 0.66);
            } else {
                alb = vec3(0.85, 0.82, 0.78);
                alpha = 0.18 + 0.35 * rays;
                if (part == 1) { alb = vec3(0.02); alpha = 0.95 - 0.35 * vUV.y; }
                if (part == 4) alpha = 0.3 + 0.35 * rays + 0.5 * (1.0 - smoothstep(0.0, 0.15, vUV.x));
                if (part == 2) alb = mix(alb, vec3(0.1), 0.4 * rays);
                if (part == 6) { alpha = 0.6; alb = vec3(0.3); }
                trans = 0.5;
            }
        }
        /* the eye: a dark lens in a ring, with the lamp in it */
        vec3 glint = vec3(0.0);
        if (part == 0) {
            vec2 de = vec2(s - uEye.x, y - uEye.y) / uEye.z;
            float d = length(de);
            if (d < 1.05 && abs(vRest.z) > 0.004) {
                vec3 iris = uSpec == 0 ? mix(vec3(0.32, 0.33, 0.33), vec3(0.12, 0.3, 0.6), smoothstep(-0.2, 0.7, de.y)) :
                            uSpec == 1 ? mix(vec3(0.6, 0.55, 0.42), vec3(0.65, 0.12, 0.06), smoothstep(-0.2, 0.5, de.y)) :
                            uSpec == 2 ? vec3(0.65, 0.09, 0.04) : vec3(0.4, 0.33, 0.18);
                float pupil = smoothstep(0.64, 0.56, d);
                vec3 e = mix(iris, vec3(0.004), pupil);
                e = mix(e, vec3(0.03), smoothstep(0.86, 1.0, d));
                float k = 1.0 - smoothstep(0.98, 1.05, d);
                alb = mix(alb, e, k);
                irid *= 1.0 - k; silver *= 1.0 - k;
                /* the cornea: a round lens, so the highlight sits off centre */
                vec3 bn = normalize(N + 0.6 * normalize(vec3(0.0, 1.0, 0.0) + V * 0.3));
                float sp = pow(max(dot(reflect(-V, N), -uL), 0.0), 30.0);
                glint = vec3(1.0) * k * (0.25 + 1.5 * sp) * smoothstep(0.3, 0.1, length(de - vec2(-0.18, 0.28)));
                spec = mix(spec, 1.2, k); rough = mix(rough, 200.0, k);
                (void(bn));
            }
        }
        if (uPass == 1) { if (alpha < 0.5) discard; return; }
        /* light: the lamp from above, a little wrapped (they are thin and
         * translucent), the water's glow mirrored in the silver and the
         * scales, and structural colour */
        float ndl = max(dot(N, -uL), 0.0);
        float wrap = max(dot(N, -uL) * 0.5 + 0.5, 0.0);
        float bk = max(dot(N, uL), 0.0) * trans;
        vec3 H = normalize(V - uL);
        float sp = pow(max(dot(N, H), 0.0), rough) * spec * (rough + 8.0) / 8.0 * ndl;
        vec3 R = reflect(-V, N);
        float env = mix(0.3, 1.7, smoothstep(-0.7, 0.9, R.y));
        float fres = pow(1.0 - nv, 4.0);
        vec3 sunp = alb * (ndl * 0.8 + wrap * 0.2 + bk) + irid * (0.35 + 0.65 * wrap)
                  + silver * vec3(0.85, 0.92, 1.0) * 0.55 * pow(max(dot(R, -uL), 0.0), 6.0);
        if (part != 0 && part != 7) { sp *= 0.15; }
        vec3 amb = alb * amb_s(vP, N) + silver * env * 0.2 + irid * 0.9 + fres * env * 0.05 * (part == 0 ? 1.0 : 0.2);
        sunp += glint;
        g_alpha = alpha;
        emit(vP, vG, sunp, sp, amb, 0.0, false);
    });

/* ---- the fish on the GPU ---- */
static GLuint p_fish;
static GLuint vao_sp[SP_N], vbo_sp[SP_N], ibo_sp[SP_N], ivbo_sp[SP_N];
static int ni_sp[SP_N], nb_sp[SP_N];
static float fish_inst[MAXFISH * 16];
static int frame_no;

static void fish_gl(void) {
    p_fish = mk_prog(VS_FISH, NULL, FS_FISH, NULL);
    Mesh m = {0};
    for (int sp = 0; sp < SP_N; sp++) {
        int nb;
        build_species(&m, sp, &nb);
        ni_sp[sp] = m.ni;
        nb_sp[sp] = nb;
        glGenVertexArrays(1, &vao_sp[sp]);
        glBindVertexArray(vao_sp[sp]);
        glGenBuffers(1, &vbo_sp[sp]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sp[sp]);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * SV * m.nv), m.v, GL_STATIC_DRAW);
        solid_attribs();
        glGenBuffers(1, &ibo_sp[sp]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_sp[sp]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * m.ni), m.ix, GL_STATIC_DRAW);
        glGenBuffers(1, &ivbo_sp[sp]);
        glBindBuffer(GL_ARRAY_BUFFER, ivbo_sp[sp]);
        glBufferData(GL_ARRAY_BUFFER, sizeof fish_inst, NULL, GL_DYNAMIC_DRAW);
        for (int a = 0; a < 4; a++) {
            glVertexAttribPointer((GLuint)(5 + a), 4, GL_FLOAT, GL_FALSE, 64, (void *)(intptr_t)(a * 16));
            glEnableVertexAttribArray((GLuint)(5 + a));
            glVertexAttribDivisor((GLuint)(5 + a), 1);
        }
    }
    free(m.v); free(m.ix);
}

static int cmp_far(const void *a, const void *b) {
    float za = shown[*(const int *)a].p[2], zb = shown[*(const int *)b].p[2];
    return za < zb ? 1 : za > zb ? -1 : 0;
}

static void fish_upload(void) {
    for (int sp = 0; sp < SP_N; sp++) {
        float *o = fish_inst;
        /* far to near, for the fins' blending */
        int order[MAXFISH];
        for (int k = 0; k < nsp[sp]; k++) order[k] = sp_first[sp] + k;
        qsort(order, (size_t)nsp[sp], sizeof(int), cmp_far);
        for (int k = 0; k < nsp[sp]; k++) {
            int i = order[k];
            Agent *a = &shown[i];
            Fish *F = &fish[i];
            V3 f = vnorm(v3(a->f[0], a->f[1], a->f[2]));
            float fade = sp == SP_TETRA ? lit.night : 0;
            float amp = a->amp * (0.8f + 0.5f * cpu_load(wall()));
            float v[16] = {a->p[0], a->p[1], a->p[2], F->size, f.x, f.y, f.z, a->roll,
                           a->phase, amp, a->bend, a->pph, a->pamp, (F->seed % 1000) * 0.001f, fade, 0};
            memcpy(o, v, sizeof v);
            o += 16;
        }
        glBindBuffer(GL_ARRAY_BUFFER, ivbo_sp[sp]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(sizeof(float) * 16 * nsp[sp]), fish_inst);
    }
}

/* what: 0 bodies, 1 fins, 2 both */
static void fish_draw(int what) {
    set_common(p_fish);
    glUniform1f(U(p_fish, "uFrame"), (float)(frame_no % 64));
    for (int sp = 0; sp < SP_N; sp++) {
        if (!nsp[sp]) continue;
        const BodySpec *b = &body_of[sp];
        glUniform1i(U(p_fish, "uSpec"), sp);
        glUniform1f(U(p_fish, "uLam"), motion[sp].lam);
        glUniform1f(U(p_fish, "uEnv0"), sp == SP_TETRA ? 0.12f : sp == SP_CORY ? 0.15f : 0.08f);
        glUniform3f(U(p_fish, "uEye"), b->eye_s, b->eye_y, b->eye_r);
        glBindVertexArray(vao_sp[sp]);
        int i0 = what == 1 ? nb_sp[sp] : 0, i1 = what == 0 ? nb_sp[sp] : ni_sp[sp];
        glDrawElementsInstanced(GL_TRIANGLES, i1 - i0, GL_UNSIGNED_INT, (void *)(intptr_t)(i0 * 4), nsp[sp]);
    }
}


/* ==================================================================== */
/* ---- the eel ----
 * A fire eel, Mastacembelus erythrotaenia. Its head follows a path laid
 * down in advance, waypoint by waypoint, by a deterministic walker: it
 * cruises along the bottom, noses round the stones and wood, and now and
 * then burrows right into the sand and waits with only its head showing.
 * The body follows the head's track (follow-the-leader, by arclength), so
 * it slides into and out of its hole; a travelling wave rides on top. */
#define EEL_LEN 0.3f
typedef struct { double t; V3 p; } EelWP;
static EelWP *ewp;
static int newp, cewp;
/* the dense track: time, position, arclength */
typedef struct { double t; V3 p; double s; } TrackPt;
static TrackPt *etr;
static int netr, cetr;
static struct {
    double t;
    float x, z, psi;
    int mode, left;
    uint32_t r;
    V3 aim;
    float dwell_dir;
} ew;
enum { EM_CRUISE, EM_NOSE, EM_BURROW };

/* what the eel lies on: the sand, low stones; tall things (wood) it passes under or round */
static float eel_floor(float x, float z) {
    float g = ground_at(x, z), sh = sand_h(x, z);
    return g - sh < 0.05f ? g : sh;
}

static float erand(void) { ew.r = hash32(ew.r + 0x9e3779b9u); return (ew.r >> 8) * (1.0f / 16777216.0f); }

static void ewp_add(double t, V3 p) {
    if (newp == cewp) ewp = realloc(ewp, sizeof(EelWP) * (size_t)(cewp = cewp ? cewp * 2 : 1024));
    ewp[newp].t = t; ewp[newp].p = p;
    newp++;
}

/* keep the walker in the tank, away from the glass */
static void ew_clamp(void) {
    ew.x = clampf(ew.x, X0 + 0.06f, X1 - 0.06f);
    ew.z = clampf(ew.z, 0.05f, TD - 0.06f);
}

/* lay down the next stretch of path */
static void eel_plan(void) {
    if (ew.left <= 0) {
        float r = erand();
        ew.mode = r < 0.4f ? EM_CRUISE : r < 0.68f ? EM_NOSE : EM_BURROW;
        ew.left = ew.mode == EM_CRUISE ? 8 + (int)(erand() * 12) : ew.mode == EM_NOSE ? 5 + (int)(erand() * 5) : 1;
        if (ew.mode == EM_NOSE) {
            /* a spot on a stone or a branch */
            for (int k = 0; k < 40; k++) {
                float x = mixf(X0 + 0.08f, X1 - 0.08f, erand()), z = mixf(0.08f, TD - 0.08f, erand());
                x = mixf(ew.x, x, 0.4f + 0.6f * erand());
                z = mixf(ew.z, z, 0.5f + 0.5f * erand());
                float hh = ground_at(x, z) - sand_h(x, z);
                if ((hh > 0.012f && hh < 0.05f) || k == 39) { ew.aim = v3(x, ground_at(x, z), z); break; }
            }
        } else {
            ew.aim = v3(mixf(X0 + 0.1f, X1 - 0.1f, erand()), 0, mixf(0.08f, TD * 0.75f, erand()));
        }
    }
    float speed = 0.03f + 0.025f * erand();
    if (ew.mode == EM_BURROW) {
        /* find a run of open, deep sand ahead */
        float bx = ew.x, bz = ew.z, bpsi = ew.psi;
        int ok = 0;
        for (int k = 0; k < 30 && !ok; k++) {
            float psi = ew.psi + (erand() - 0.5f) * 2.5f;
            float sx = ew.x + cosf(psi) * 0.05f, sz = ew.z + sinf(psi) * 0.05f;
            ok = 1;
            for (int j = 0; j <= 8 && ok; j++) {
                float px = sx + cosf(psi) * 0.32f * j / 8, pz = sz + sinf(psi) * 0.32f * j / 8;
                if (px < X0 + 0.05f || px > X1 - 0.05f || pz < 0.05f || pz > TD - 0.05f) ok = 0;
                else if (ground_at(px, pz) > sand_h(px, pz) + 0.004f || sand_h(px, pz) < 0.05f) ok = 0;
            }
            if (ok) { bx = sx; bz = sz; bpsi = psi; }
        }
        if (!ok) { ew.mode = EM_CRUISE; ew.left = 6; return; }
        /* go in: down at a slant, then on under the sand, curving a little,
         * then up at the far end until the head is out */
        double t = ew.t;
        V3 last = ewp[newp - 1].p;
        V3 p = v3(bx, sand_h(bx, bz) + 0.012f, bz);
        t += vlen(vsub(p, last)) / speed;
        ewp_add(t, p);
        float x = bx, z = bz, psi = bpsi;
        int n = 9;
        for (int j = 1; j <= n; j++) {
            psi += (erand() - 0.5f) * 0.4f;
            x += cosf(psi) * 0.036f; z += sinf(psi) * 0.036f;
            x = clampf(x, X0 + 0.05f, X1 - 0.05f); z = clampf(z, 0.05f, TD - 0.05f);
            float sh = sand_h(x, z);
            float y = j < 2 ? sh - 0.012f : fmaxf(sh - 0.026f, 0.014f);
            if (j == n) y = sh + 0.004f;
            t += 0.036 / 0.018;
            ewp_add(t, v3(x, y, z));
        }
        /* the head comes up, pointing up and on */
        float hx = x + cosf(psi) * 0.012f, hz = z + sinf(psi) * 0.012f;
        V3 head = v3(hx, sand_h(hx, hz) + 0.022f, hz);
        t += 3.0;
        ewp_add(t, head);
        /* and waits, creeping a millimetre or so, looking about */
        double dwell = 50 + 110 * erand();
        int nd = (int)(dwell / 9);
        for (int j = 1; j <= nd; j++) {
            float f = (float)j / nd;
            float lx = hx + cosf(psi) * 0.008f * f + sinf(psi) * 0.004f * sinf(j * 1.7f);
            float lz = hz + sinf(psi) * 0.008f * f - cosf(psi) * 0.004f * sinf(j * 1.7f);
            t += dwell / nd;
            ewp_add(t, v3(lx, sand_h(lx, lz) + 0.022f + 0.004f * f + 0.003f * sinf(j * 2.3f), lz));
        }
        /* then out, and away */
        V3 l = ewp[newp - 1].p;
        float ox = l.x + cosf(psi) * 0.04f, oz = l.z + sinf(psi) * 0.04f;
        t += 3.0;
        ewp_add(t, v3(ox, eel_floor(ox, oz) + 0.03f, oz));
        ew.x = ox; ew.z = oz; ew.psi = psi; ew.t = t;
        ew_clamp();
        ew.left = 0;
        return;
    }
    /* cruising and nosing: steps along the bottom, steering for the aim */
    float step = ew.mode == EM_NOSE ? 0.022f : 0.04f;
    if (ew.mode == EM_NOSE) speed *= 0.45f;
    float want = atan2f(ew.aim.z - ew.z, ew.aim.x - ew.x);
    float d = fmodf(want - ew.psi + 3 * PI, TAU) - PI;
    ew.psi += clampf(d, -0.6f, 0.6f) * 0.7f + (erand() - 0.5f) * 0.5f;
    /* round anything tall (the wood), never up it */
    for (int k = 0; k < 8; k++) {
        float nx = ew.x + cosf(ew.psi) * step, nz = ew.z + sinf(ew.psi) * step;
        if (ground_at(nx, nz) - sand_h(nx, nz) < 0.045f) break;
        ew.psi += (k % 2 ? -1.0f : 1.0f) * 0.5f * (k + 1);
    }
    ew.x += cosf(ew.psi) * step;
    ew.z += sinf(ew.psi) * step;
    if (ew.x < X0 + 0.07f || ew.x > X1 - 0.07f || ew.z < 0.06f || ew.z > TD - 0.07f) ew.psi += PI * 0.5f;
    ew_clamp();
    float g = eel_floor(ew.x, ew.z);
    float lift = ew.mode == EM_NOSE ? 0.012f + 0.02f * erand() : 0.012f + (erand() < 0.12f ? 0.05f * erand() : 0);
    V3 p = v3(ew.x, g + lift, ew.z);
    V3 last = ewp[newp - 1].p;
    ew.t += fmaxf(vlen(vsub(p, last)), 0.004f) / speed;
    ewp_add(ew.t, p);
    ew.left--;
}

static V3 eel_wp_eval(int j, float u) {
    V3 p0 = ewp[j > 0 ? j - 1 : 0].p, p1 = ewp[j].p, p2 = ewp[j + 1].p, p3 = ewp[j + 2 < newp ? j + 2 : newp - 1].p;
    return vcatmull(p0, p1, p2, p3, u);
}

/* extend the dense track to cover time tmax */
static int etr_seg;                     /* the next waypoint segment to sample */
static void eel_extend(double tmax) {
    while (newp < 4 || ewp[newp - 3].t < tmax + 40) eel_plan();
    while (etr_seg < newp - 3 && ewp[etr_seg].t <= tmax + 30) {
        int j = etr_seg;
        const int NS = 10;
        for (int k = (j == 0 ? 0 : 1); k <= NS; k++) {
            float u = (float)k / NS;
            V3 p = eel_wp_eval(j, u);
            if (netr == cetr) etr = realloc(etr, sizeof(TrackPt) * (size_t)(cetr = cetr ? cetr * 2 : 8192));
            TrackPt *tp = &etr[netr];
            tp->t = ewp[j].t + (ewp[j + 1].t - ewp[j].t) * u;
            tp->p = p;
            tp->s = netr ? etr[netr - 1].s + vlen(vsub(p, etr[netr - 1].p)) : 0;
            netr++;
        }
        etr_seg++;
    }
}

static void eel_init(void) {
    ew.r = hash32(opt_seed ^ 0xEE1u);
    /* before the epoch: lying along the sand, so the body has a past */
    ew.x = mixf(X0, X1, 0.25f + 0.5f * erand());
    ew.z = TD * (0.25f + 0.3f * erand());
    ew.psi = erand() < 0.5f ? 0 : PI;
    ew.t = -60;
    float bx = ew.x - cosf(ew.psi) * 0.4f, bz = ew.z;
    for (int k = 0; k < 10; k++) {
        float f = k / 9.0f;
        float x = mixf(bx, ew.x, f), z = bz + 0.02f * sinf(f * 5);
        ewp_add(-60 + 40 * f, v3(x, eel_floor(x, z) + 0.012f, z));
    }
    ew.t = -20;
    ew.mode = EM_CRUISE;
    ew.left = 3;
}

static int track_find_t(double t) {
    int lo = 0, hi = netr - 1;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (etr[m].t <= t) lo = m; else hi = m; }
    return lo;
}
static V3 track_at_s(double s, int hint) {
    int lo = 0, hi = hint + 1 < netr ? hint + 1 : netr - 1;
    if (s <= 0) return etr[0].p;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (etr[m].s <= s) lo = m; else hi = m; }
    double d = etr[hi].s - etr[lo].s;
    float f = d > 1e-9 ? (float)((s - etr[lo].s) / d) : 0;
    return vlerp(etr[lo].p, etr[hi].p, clampf(f, 0, 1));
}

/* the eel's half-width and half-height along it (f: 0 snout .. 1 tail) */
static float eel_w(float f) {
    if (f < 0.1f) return 0.0012f + 0.0083f * powf(f / 0.1f, 0.62f);
    if (f < 0.16f) return 0.0095f;
    if (f < 0.62f) return 0.0095f;
    return mixf(0.0095f, 0.0022f, powf((f - 0.62f) / 0.38f, 1.2f));
}
static float eel_h(float f) {
    return eel_w(f) * mixf(0.9f, 1.3f, smooth(0.02f, 0.14f, f));
}

static Mesh dyn_mesh;                   /* the eel and the cherry shrimp, rebuilt each frame */
#define eel_mesh dyn_mesh
static V3 eel_head_p, eel_head_dir;
#define EEL_NS 100
#define EEL_NA 14
static void eel_build(double t) {
    eel_extend(t + 5);
    Mesh *m = &eel_mesh;
    m->nv = m->ni = 0;
    int ih = track_find_t(t);
    double dt_ = etr[ih + 1].t - etr[ih].t;
    float fu = dt_ > 1e-9 ? (float)((t - etr[ih].t) / dt_) : 0;
    double S = etr[ih].s + (etr[ih + 1].s - etr[ih].s) * fu;
    /* speed, for the wave */
    int ih2 = track_find_t(t - 1.0);
    float speed = (float)((S - etr[ih2].s) / 1.0);
    float amp = 0.0025f + 0.014f * clampf(speed / 0.04f, 0, 1);
    float lam = EEL_LEN * 0.55f;
    V3 sp[EEL_NS], tg[EEL_NS];
    for (int k = 0; k < EEL_NS; k++) {
        float f = (float)k / (EEL_NS - 1);
        sp[k] = track_at_s(S - f * EEL_LEN, ih + 1);
    }
    /* the wave rides on the track, sideways in the horizontal */
    for (int k = 0; k < EEL_NS; k++) tg[k] = vnorm(vsub(sp[k > 0 ? k - 1 : 0], sp[k < EEL_NS - 1 ? k + 1 : k]));
    V3 wav[EEL_NS];
    for (int k = 0; k < EEL_NS; k++) {
        float f = (float)k / (EEL_NS - 1);
        V3 side = vcross(tg[k], v3(0, 1, 0));
        if (vlen(side) < 1e-3f) side = v3(1, 0, 0);
        side = vnorm(side);
        float under = smooth(-0.004f, 0.006f, sp[k].y - sand_h(sp[k].x, sp[k].z) - eel_h(f) * 0.5f);
        float ph = TAU * (f * EEL_LEN / lam - (float)fmod(S / (lam * 0.75), 1.0)) ;
        float a = amp * (0.12f + 0.88f * powf(f, 1.3f)) * under;
        a += 0.0018f * f * under * sinf(TAU * f * 1.5f - (float)fmod(t * 0.9, TAU));
        wav[k] = vmad(sp[k], side, a * sinf(ph));
    }
    for (int k = 0; k < EEL_NS; k++) tg[k] = vnorm(vsub(wav[k > 0 ? k - 1 : 0], wav[k < EEL_NS - 1 ? k + 1 : k]));
    eel_head_p = wav[0];
    eel_head_dir = tg[0];
    /* rings, then the fin folds */
    V3 prev_up = v3(0, 1, 0);
    V3 ups[EEL_NS], sides[EEL_NS];
    for (int k = 0; k < EEL_NS; k++) {
        V3 T = tg[k];
        V3 up = vsub(prev_up, vmul(T, vdot(prev_up, T)));
        V3 wup = vsub(v3(0, 1, 0), vmul(T, T.y));
        up = vnorm(vlerp(up, wup, 0.3f));
        if (vlen(up) < 1e-3f) up = vperp(T);
        prev_up = up;
        ups[k] = up;
        sides[k] = vnorm(vcross(T, up));
    }
    for (int k = 0; k < EEL_NS; k++) {
        float f = (float)k / (EEL_NS - 1);
        float w = eel_w(f), h = eel_h(f);
        for (int j = 0; j <= EEL_NA; j++) {
            float a = (float)j / EEL_NA * TAU;
            float ca = cosf(a), sa = sinf(a);
            V3 n = vnorm(vadd(vmul(sides[k], ca / w), vmul(ups[k], sa / h)));
            V3 p = vadd(wav[k], vadd(vmul(sides[k], ca * w), vmul(ups[k], sa * h)));
            /* the snout: a fleshy, pointed proboscis tip */
            m_vert(m, p, n, f, (float)j / EEL_NA, M_EEL, f, sa, ca >= 0 ? 1.0f : -1.0f);
        }
    }
    for (int k = 0; k < EEL_NS - 1; k++)
        for (int j = 0; j < EEL_NA; j++) {
            int a = k * (EEL_NA + 1) + j, b = a + 1, c = a + EEL_NA + 1, d = c + 1;
            m_tri(m, a, c, b); m_tri(m, b, c, d);
        }
    /* snout cap */
    {
        int c = m_vert(m, vmad(wav[0], tg[0], 0.0025f), tg[0], 0, 0, M_EEL, -0.01f, 0, 0);
        for (int j = 0; j < EEL_NA; j++) m_tri(m, c, j, j + 1);
    }
    /* the dorsal and anal folds along the back half, rounding into the tail */
    for (int side = 0; side < 2; side++) {
        float f0 = side ? 0.5f : 0.42f;
        int base = m->nv;
        int n = 0;
        for (int k = 0; k < EEL_NS; k++) {
            float f = (float)k / (EEL_NS - 1);
            if (f < f0) continue;
            float hgt = eel_h(f) * (0.55f + 0.9f * smooth(f0, 0.9f, f)) * (1 - smooth(0.93f, 1.0f, f) * 0.6f) * smooth(f0, f0 + 0.06f, f);
            if (f > 0.97f) hgt += 0.004f;
            V3 dir = side ? vmul(ups[k], -1) : ups[k];
            V3 b0 = vmad(wav[k], dir, eel_h(f) * 0.85f), b1 = vmad(wav[k], dir, eel_h(f) * 0.85f + hgt);
            m_vert(m, b0, sides[k], f, 0, M_EEL, f, side ? -1.0f : 1.0f, 2);
            m_vert(m, b1, sides[k], f, 1, M_EEL, f, side ? -1.0f : 1.0f, 2);
            n++;
        }
        for (int k = 0; k < n - 1; k++) {
            int a = base + k * 2;
            m_tri(m, a, a + 2, a + 1); m_tri(m, a + 1, a + 2, a + 3);
        }
    }
    /* pectorals: two small fans behind the head */
    for (int s = -1; s <= 1; s += 2) {
        int k = (int)(0.11f * (EEL_NS - 1));
        V3 at = vadd(wav[k], vmul(sides[k], eel_w(0.11f) * 0.9f * s));
        float fl = 0.3f * sinf((float)fmod(t * 5.0, TAU) + s);
        V3 out = vnorm(vadd(vmul(sides[k], s * 0.8f), vmul(tg[k], -0.6f + fl)));
        V3 upv = ups[k];
        int a = m_vert(m, vmad(at, upv, 0.003f), sides[k], 0, 0, M_EEL, 0.11f, 0, 3);
        int b = m_vert(m, vmad(at, upv, -0.003f), sides[k], 0, 1, M_EEL, 0.11f, 0, 3);
        int c = m_vert(m, vadd(vmad(at, upv, 0.004f), vmul(out, 0.008f)), sides[k], 1, 0, M_EEL, 0.11f, 1, 3);
        int d = m_vert(m, vadd(vmad(at, upv, -0.004f), vmul(out, 0.008f)), sides[k], 1, 1, M_EEL, 0.11f, 1, 3);
        m_tri(m, a, c, b); m_tri(m, b, c, d);
    }
}

/* ==================================================================== */
/* ---- the shrimp ----
 * Cherry shrimp (Neocaridina) and a few Amano shrimp (Caridina), built
 * every frame from segments: carapace and rostrum, six abdominal segments
 * bent in the shrimp's hump, a tail fan, stalked eyes, long antennae and
 * forked antennules, five pairs of legs (the first two with little claws)
 * and the swimmerets under the tail. Each lives through a deterministic
 * diary of episodes: grazing (the claws picking at the surface in turn),
 * walking, a hop through the water, and now and then a flick backwards. */
#define MAXSHRIMP 16
enum { SE_GRAZE, SE_WALK, SE_SWIM, SE_FLICK };
typedef struct { double t0, t1; int kind; float ax, az, apsi, bx, bz, bpsi; } Episode;
typedef struct {
    int amano;
    float len;
    uint32_t seed, r;
    Episode *ep;
    int nep, cep;
    int saddle;                         /* a berried-to-be female: the yellow saddle */
} Shrimp;
static Shrimp shrimp[MAXSHRIMP];
static int nshrimp;

static float srand_(Shrimp *s) { s->r = hash32(s->r + 0x632be5abu); return (s->r >> 8) * (1.0f / 16777216.0f); }

/* somewhere a shrimp would graze: the tops of stones and wood mostly, the sand sometimes */
static void graze_spot(Shrimp *s, float nearx, float nearz, float range, float *x, float *z) {
    for (int k = 0; k < 30; k++) {
        float a = srand_(s) * TAU, r = range * sqrtf(srand_(s));
        float px = clampf(nearx + cosf(a) * r, X0 + 0.04f, X1 - 0.04f), pz = clampf(nearz + sinf(a) * r, 0.03f, TD - 0.04f);
        int decor = ground_at(px, pz) > sand_h(px, pz) + 0.008f;
        if (decor || srand_(s) < 0.25f || k == 29) { *x = px; *z = pz; return; }
    }
}

static void shrimp_plan(Shrimp *s) {
    if (s->nep == s->cep) s->ep = realloc(s->ep, sizeof(Episode) * (size_t)(s->cep = s->cep ? s->cep * 2 : 64));
    Episode *prev = s->nep ? &s->ep[s->nep - 1] : NULL;
    Episode *e = &s->ep[s->nep];
    e->t0 = prev ? prev->t1 : -20;
    e->ax = prev ? prev->bx : 0; e->az = prev ? prev->bz : 0; e->apsi = prev ? prev->bpsi : 0;
    if (!prev) {
        /* where it starts: a stone or the wood near one of the focal points */
        int f = (int)(srand_(s) * nfocal);
        graze_spot(s, focal_x[f], focal_z[f] - 0.04f, 0.18f, &e->ax, &e->az);
        e->apsi = srand_(s) * TAU;
    }
    float r = srand_(s);
    int kind = !prev || prev->kind != SE_GRAZE ? SE_GRAZE : r < 0.62f ? SE_WALK : r < 0.9f ? SE_SWIM : SE_FLICK;
    e->kind = kind;
    e->bx = e->ax; e->bz = e->az; e->bpsi = e->apsi;
    if (kind == SE_GRAZE) {
        e->t1 = e->t0 + 6 + 30 * srand_(s);
        e->bpsi = e->apsi + (srand_(s) - 0.5f) * 1.2f;
    } else if (kind == SE_WALK) {
        float d = 0.015f + 0.05f * srand_(s);
        float psi = e->apsi + (srand_(s) - 0.5f) * 1.6f;
        e->bx = clampf(e->ax + cosf(psi) * d, X0 + 0.04f, X1 - 0.04f);
        e->bz = clampf(e->az + sinf(psi) * d, 0.03f, TD - 0.04f);
        e->bpsi = atan2f(e->bz - e->az, e->bx - e->ax);
        e->apsi = e->bpsi;
        e->t1 = e->t0 + d / (0.008f + 0.006f * srand_(s));
        int steep = 0;
        for (int k = 1; k <= 6; k++) {
            float f0 = (k - 1) / 6.0f, f1 = k / 6.0f;
            if (fabsf(ground_at(mixf(e->ax, e->bx, f1), mixf(e->az, e->bz, f1)) - ground_at(mixf(e->ax, e->bx, f0), mixf(e->az, e->bz, f0))) > 0.012f) steep = 1;
        }
        if (steep) { e->kind = SE_SWIM; e->t1 = e->t0 + 1.5f + d / 0.05f; }
    } else if (kind == SE_SWIM) {
        graze_spot(s, e->ax, e->az, 0.2f, &e->bx, &e->bz);
        e->bpsi = atan2f(e->bz - e->az, e->bx - e->ax);
        float d = sqrtf((e->bx - e->ax) * (e->bx - e->ax) + (e->bz - e->az) * (e->bz - e->az));
        e->t1 = e->t0 + 1.5f + d / 0.05f;
    } else {
        /* the flick: shoots backwards, tail first */
        float d = 0.04f + 0.05f * srand_(s);
        e->bx = clampf(e->ax - cosf(e->apsi) * d, X0 + 0.04f, X1 - 0.04f);
        e->bz = clampf(e->az - sinf(e->apsi) * d, 0.03f, TD - 0.04f);
        e->bpsi = e->apsi;
        e->t1 = e->t0 + 1.6;
    }
    s->nep++;
}

static void shrimp_init(void) {
    float w = tank_w();
    int ncherry = (int)clampf(roundf(8 * w / 0.9f), 6, 13), namano = w > 1.2f ? 3 : 2;
    nshrimp = 0;
    for (int i = 0; i < ncherry + namano && nshrimp < MAXSHRIMP; i++) {
        Shrimp *s = &shrimp[nshrimp++];
        memset(s, 0, sizeof *s);
        s->amano = i >= ncherry;
        s->seed = hash32(opt_seed * 31u + (uint32_t)i * 101u + 7u);
        s->r = s->seed;
        s->len = s->amano ? frange(0.036f, 0.044f) : frange(0.019f, 0.026f);
        s->saddle = !s->amano && frand() < 0.35f;
    }
}

static Episode *shrimp_ep(Shrimp *s, double t) {
    while (!s->nep || s->ep[s->nep - 1].t1 < t + 5) shrimp_plan(s);
    int lo = 0, hi = s->nep - 1;
    while (lo < hi) { int m = (lo + hi) / 2; if (s->ep[m].t1 < t) lo = m + 1; else hi = m; }
    return &s->ep[lo];
}

/* -- building blocks, in world space -- */
static float ring_cs[17][17][2];
static void tube_ring(Mesh *m, V3 c, V3 a, V3 b, float ra, float rb, int ns, float mat, float part, float u, float v) {
    if (ring_cs[ns][1][0] == 0) for (int k = 0; k <= ns; k++) { ring_cs[ns][k][0] = cosf((float)k / ns * TAU); ring_cs[ns][k][1] = sinf((float)k / ns * TAU); }
    for (int k = 0; k <= ns; k++) {
        float ca = ring_cs[ns][k][0], sa = ring_cs[ns][k][1];
        V3 n = vnorm(vadd(vmul(a, ca / fmaxf(ra, 1e-5f)), vmul(b, sa / fmaxf(rb, 1e-5f))));
        m_vert(m, vadd(c, vadd(vmul(a, ca * ra), vmul(b, sa * rb))), n, u, (float)k / ns, mat, part, v, sa);
    }
}
static void tube_link(Mesh *m, int r0, int r1, int ns) {
    for (int k = 0; k < ns; k++) {
        int a = r0 + k, b = a + 1, c = r1 + k, d = c + 1;
        m_tri(m, a, c, b); m_tri(m, b, c, d);
    }
}
/* a thin limb through points, radius tapering r0 -> r1 */
static void limb(Mesh *m, const V3 *p, int n, float r0, float r1, float mat, float part) {
    const int ns = 4;
    V3 prev = v3(0, 0, 0);
    int last = -1;
    for (int i = 0; i < n; i++) {
        V3 t = vnorm(vsub(p[i < n - 1 ? i + 1 : i], p[i > 0 ? i - 1 : i]));
        V3 a = i == 0 ? vperp(t) : vnorm(vsub(prev, vmul(t, vdot(prev, t))));
        prev = a;
        V3 b = vcross(t, a);
        float r = mixf(r0, r1, (float)i / (n - 1));
        int base = m->nv;
        tube_ring(m, p[i], a, b, r, r, ns, mat, part, (float)i / (n - 1), 0);
        if (last >= 0) tube_link(m, last, base, ns);
        last = base;
    }
}
static void blob(Mesh *m, V3 c, float r, float mat, float part) {
    const int nu = 8, nv = 6;
    int base = m->nv;
    for (int j = 0; j <= nv; j++)
        for (int i = 0; i <= nu; i++) {
            float th = PI * j / nv, ph = TAU * i / nu;
            V3 n = v3(sinf(th) * cosf(ph), cosf(th), sinf(th) * sinf(ph));
            m_vert(m, vmad(c, n, r), n, (float)i / nu, (float)j / nv, mat, part, 0, 0);
        }
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            int a = base + j * (nu + 1) + i, b = a + 1, c2 = a + nu + 1, d = c2 + 1;
            m_tri(m, a, c2, b); m_tri(m, b, c2, d);
        }
}

static Mesh shrimp_mesh[2];             /* cherries (opaque), amanos (see-through) */
static V3 shrimp_pos[MAXSHRIMP];
static int shrimp_spec = -1;            /* AQ_SPEC: pose this one for a look */
static float shrimp_spec_x, shrimp_spec_z, shrimp_spec_yaw;

static void shrimp_build_one(Shrimp *s, int idx, double t, Mesh *m) {
    Episode *e = shrimp_ep(s, t);
    float L = s->len;
    float u = (float)clampf((float)((t - e->t0) / fmax(e->t1 - e->t0, 1e-3)), 0, 1);
    float x, z, psi, lift = 0, curl = 0, pleo = 0, walk = 0;
    float ease = u * u * (3 - 2 * u);
    float act = 1;
    if (e->kind == SE_GRAZE) {
        x = e->ax + 0.002f * sinf((float)fmod(t * 0.7, TAU) + s->seed % 7);
        z = e->az + 0.002f * sinf((float)fmod(t * 0.53, TAU) + s->seed % 5);
        psi = e->apsi + ang_mix(0, e->bpsi - e->apsi, ease) + 0.15f * sinf((float)fmod(t * 0.31, TAU) + s->seed % 3);
        walk = (float)(t * 0.4);
    } else if (e->kind == SE_WALK) {
        x = mixf(e->ax, e->bx, ease); z = mixf(e->az, e->bz, ease);
        psi = e->bpsi;
        float d = sqrtf((e->bx - e->ax) * (e->bx - e->ax) + (e->bz - e->az) * (e->bz - e->az));
        walk = ease * d / (L * 0.12f);
    } else if (e->kind == SE_SWIM) {
        x = mixf(e->ax, e->bx, ease); z = mixf(e->az, e->bz, ease);
        psi = ang_mix(e->apsi, e->bpsi, smooth(0, 0.2f, u));
        lift = sinf(PI * u) * (0.03f + 0.02f * (s->seed % 3)) * smooth(0, 0.15f, u);
        pleo = smooth(0, 0.1f, u) * (1 - smooth(0.85f, 1, u));
        walk = (float)(t * 0.4);
    } else {
        /* the flick: most of the way in the first tenth of a second */
        float k = 1 - expf(-u * 14);
        x = mixf(e->ax, e->bx, k); z = mixf(e->az, e->bz, k);
        psi = e->apsi;
        lift = 0.025f * sinf(PI * fminf(u * 1.4f, 1)) ;
        curl = (1 - smooth(0.0f, 0.35f, u)) * smooth(0, 0.02f, u);
        pleo = 0.5f * (1 - smooth(0.1f, 0.6f, u));
        walk = (float)(t * 0.4);
    }
    (void)act;
    if (idx == shrimp_spec) { x = shrimp_spec_x; z = shrimp_spec_z; psi = shrimp_spec_yaw; lift = 0; curl = 0; }
    float g = ground_at(x, z);
    V3 gn = ground_n(x, z);
    float hgt = L * 0.13f;
    V3 pos = v3(x, g + hgt + lift, z);
    shrimp_pos[idx] = pos;
    /* the body frame, following the slope under it */
    V3 F0 = v3(cosf(psi), 0, sinf(psi));
    V3 Up = vnorm(vlerp(v3(0, 1, 0), gn, lift > 0.004f ? 0.2f : 0.85f));
    V3 F = vnorm(vsub(F0, vmul(Up, vdot(F0, Up))));
    V3 R = vcross(F, Up);
    float bob = e->kind == SE_GRAZE ? 0.03f * sinf((float)fmod(t * 9.0 + idx, TAU)) * 0.2f : 0;
    F = vnorm(vadd(F, vmul(Up, bob - 0.08f)));
    Up = vnorm(vcross(R, F));
#define LP(px, py, pz) vadd(pos, vadd(vmul(F, (px) * L), vadd(vmul(Up, (py) * L), vmul(R, (pz) * L))))
    float mat = s->amano ? M_AMANO : M_SHRIMP;
    float sid = (float)(s->saddle ? 1 : 0) + idx * 0.01f;
    /* the carapace: a lofted shell from the rostrum's base back */
    {
        const int nr = 9, ns = 12;
        int last = -1;
        for (int i = 0; i <= nr; i++) {
            float f = (float)i / nr;
            float px = mixf(0.42f, 0.0f, f);
            float hh = 0.055f + 0.075f * sinf(PI * mixf(0.18f, 0.92f, f));
            float ww = 0.045f + 0.06f * sinf(PI * mixf(0.2f, 0.95f, f));
            V3 c = LP(px, 0.02f + 0.02f * f, 0);
            int base = m->nv;
            for (int k = 0; k <= ns; k++) {
                float an = (float)k / ns * TAU, ca = cosf(an), sa = sinf(an);
                float yy = sa > 0 ? sa * hh : sa * hh * 0.85f;
                V3 n = vnorm(vadd(vmul(Up, sa / hh), vmul(R, ca / ww)));
                m_vert(m, vadd(c, vadd(vmul(Up, yy * L), vmul(R, ca * ww * L))), n, f, (float)k / ns, mat, 0, sa, sid);
            }
            if (last >= 0) tube_link(m, last, base, ns);
            last = base;
        }
    }
    /* the rostrum */
    {
        V3 p[3] = {LP(0.4f, 0.06f, 0), LP(0.5f, 0.07f, 0), LP(0.6f, 0.085f, 0)};
        limb(m, p, 3, 0.014f * L, 0.003f * L, mat, 6);
    }
    /* the abdomen: six segments, arched, curling under in a flick */
    V3 joint = LP(0.02f, 0.03f, 0);
    V3 dir = vnorm(vadd(vmul(F, -1), vmul(Up, 0.18f)));
    static const float seg_len[6] = {0.095f, 0.095f, 0.1f, 0.085f, 0.075f, 0.11f};
    static const float seg_bend[6] = {0.0f, 0.08f, 0.26f, 0.12f, 0.06f, 0.04f};
    float hump = 0.95f;
    for (int sgi = 0; sgi < 6; sgi++) {
        float bend = seg_bend[sgi] * hump + curl * 0.42f + (e->kind == SE_GRAZE ? 0.02f * sinf((float)fmod(t * 1.3, TAU) + sgi) : 0);
        dir = vnorm(vrot(dir, R, bend));
        float hh = 0.105f * (1 - sgi * 0.11f), ww = 0.085f * (1 - sgi * 0.1f);
        if (sgi == 5) { hh *= 0.8f; ww *= 0.8f; }
        V3 upn = vnorm(vcross(dir, R));
        if (vdot(upn, Up) < 0) upn = vmul(upn, -1);
        V3 nxt = vmad(joint, dir, seg_len[sgi] * L);
        /* each segment's shell overlaps the next, a little proud at its front */
        int r0 = m->nv;
        tube_ring(m, vmad(joint, dir, -0.01f * L), upn, R, hh * 1.06f * L, ww * 1.06f * L, 10, mat, 1, sgi / 6.0f, (float)sgi);
        int r1 = m->nv;
        tube_ring(m, vmad(nxt, dir, 0.004f * L), upn, R, hh * 0.9f * L, ww * 0.9f * L, 10, mat, 1, (sgi + 1) / 6.0f, (float)sgi);
        tube_link(m, r0, r1, 10);
        /* the swimmerets under it, beating front to back */
        if (sgi < 5) {
            for (int sd = -1; sd <= 1; sd += 2) {
                float ph = (float)fmod(t * 9.0, TAU) - sgi * 0.9f;
                float beat = pleo * sinf(ph) * 0.7f + 0.2f;
                V3 a = vadd(vmad(joint, dir, seg_len[sgi] * L * 0.5f), vadd(vmul(upn, -hh * 0.8f * L), vmul(R, sd * ww * 0.4f * L)));
                V3 dn = vnorm(vrot(vmul(upn, -1), R, -beat));
                V3 p2[3] = {a, vmad(a, dn, 0.06f * L), vadd(vmad(a, dn, 0.12f * L), vmul(dir, 0.02f * L))};
                limb(m, p2, 3, 0.01f * L, 0.006f * L, mat, 5);
            }
        }
        joint = nxt;
        if (sgi == 5) {
            /* the tail fan: telson and four uropods */
            for (int f = -2; f <= 2; f++) {
                float spread = f * 0.32f * (1 - curl * 0.3f);
                V3 fd = vnorm(vrot(vrot(dir, upn, spread), R, 0.1f));
                V3 side = vnorm(vcross(fd, upn));
                float fl = (f == 0 ? 0.13f : 0.12f) * L;
                V3 a = joint, b = vmad(joint, fd, fl);
                float wd = 0.03f * L;
                int q0 = m_vert(m, vmad(a, side, -wd * 0.5f), upn, 0, 0, mat, 4, (float)f, 0);
                int q1 = m_vert(m, vmad(a, side, wd * 0.5f), upn, 1, 0, mat, 4, (float)f, 0);
                int q2 = m_vert(m, vmad(b, side, -wd), upn, 0, 1, mat, 4, (float)f, 1);
                int q3 = m_vert(m, vmad(b, side, wd), upn, 1, 1, mat, 4, (float)f, 1);
                m_tri(m, q0, q1, q2); m_tri(m, q1, q3, q2);
            }
        }
    }
    /* the eyes on their stalks */
    for (int sd = -1; sd <= 1; sd += 2) {
        V3 a = LP(0.4f, 0.05f, sd * 0.03f), b = LP(0.46f, 0.07f, sd * 0.075f);
        V3 p[2] = {a, b};
        limb(m, p, 2, 0.012f * L, 0.012f * L, mat, 2);
        blob(m, b, 0.03f * L, M_SEYE, 0);
    }
    /* the long antennae, sweeping back and waving, and the forked antennules */
    for (int sd = -1; sd <= 1; sd += 2) {
        V3 p[10];
        float w1 = 0.25f * sinf((float)fmod(t * 0.9, TAU) + sd + s->seed % 11);
        float w2 = 0.2f * sinf((float)fmod(t * 1.7, TAU) + sd * 2 + s->seed % 13);
        V3 d = vnorm(vadd(vadd(vmul(F, 0.7f), vmul(R, sd * 0.5f)), vmul(Up, 0.35f)));
        p[0] = LP(0.44f, 0.03f, sd * 0.03f);
        float segl = (s->amano ? 1.0f : 1.35f) * L / 9;
        for (int i = 1; i < 10; i++) {
            float f = i / 9.0f;
            /* they arc out and round to trail back along the body */
            d = vnorm(vrot(d, Up, -sd * (0.27f + w1 * 0.14f)));
            d = vnorm(vrot(d, R, -0.085f - w2 * 0.07f * f));
            p[i] = vmad(p[i - 1], d, segl);
        }
        limb(m, p, 10, 0.008f * L, 0.0025f * L, mat, 3);
        for (int k = 0; k < 2; k++) {
            float fl = 0.25f * sinf((float)fmod(t * 6.0, TAU) * (1 + k * 0.3f) + sd);
            V3 q[4];
            V3 dd = vnorm(vadd(vadd(F, vmul(R, sd * (0.25f + 0.25f * k))), vmul(Up, 0.5f + fl)));
            q[0] = LP(0.46f, 0.06f, sd * 0.02f);
            for (int i = 1; i < 4; i++) { dd = vnorm(vrot(dd, R, -0.12f)); q[i] = vmad(q[i - 1], dd, 0.1f * L); }
            limb(m, q, 4, 0.006f * L, 0.0025f * L, mat, 3);
        }
    }
    /* the legs: claws picking (in turn) at the surface in front, then the walkers */
    float step = L * 0.12f;
    for (int lg = 0; lg < 5; lg++) {
        float hx = 0.34f - lg * 0.08f;
        for (int sd = -1; sd <= 1; sd += 2) {
            V3 hip = LP(hx, -0.07f, sd * 0.05f);
            if (lg < 2) {
                /* chelipeds: down to the surface, up to the mouth */
                float rate = 2.6f + (s->seed % 5) * 0.2f;
                float ph = (float)fmod(t * rate + (sd > 0 ? 0.5 : 0) + lg * 0.25, 1.0);
                float pick = e->kind == SE_GRAZE ? smooth(0.0f, 0.35f, ph) * (1 - smooth(0.5f, 0.9f, ph)) : 0.3f;
                V3 ground_pt = LP(0.52f + 0.06f * lg, -0.2f, sd * 0.08f);
                ground_pt.y = fmaxf(ground_pt.y, ground_at(ground_pt.x, ground_pt.z) + 0.0005f);
                V3 mouth = LP(0.4f, -0.06f, sd * 0.02f);
                V3 tip = vlerp(mouth, ground_pt, 1 - pick);
                V3 elbow = vadd(vlerp(hip, tip, 0.5f), vadd(vmul(R, sd * 0.06f * L), vmul(Up, 0.03f * L)));
                V3 p[4] = {hip, elbow, vlerp(elbow, tip, 0.6f), tip};
                limb(m, p, 4, 0.012f * L, 0.007f * L, mat, 2);
                blob(m, tip, 0.012f * L, mat, 7);
            } else {
                /* walkers: feet on the ground, stepping in a wave */
                float ph = walk * 1.0f + lg * 0.33f + (sd > 0 ? 0.5f : 0);
                float cyc = ph - floorf(ph);
                float sw = cyc < 0.3f ? cyc / 0.3f : 1 - (cyc - 0.3f) / 0.7f;
                float raise = cyc < 0.3f ? sinf(PI * cyc / 0.3f) : 0;
                if (e->kind == SE_GRAZE) { sw = 0.5f; raise = 0; }
                V3 foot = LP(hx + (sw - 0.5f) * step / L - 0.04f * (lg - 2), -0.25f, sd * (0.2f + 0.02f * lg));
                float gy = ground_at(foot.x, foot.z);
                if (lift > 0.004f) foot = vmad(LP(hx - 0.1f, -0.22f, sd * 0.13f), Up, 0);
                else foot.y = gy + raise * 0.06f * L;
                V3 knee = vadd(vlerp(hip, foot, 0.4f), vadd(vmul(R, sd * 0.07f * L), vmul(Up, 0.07f * L)));
                V3 p[4] = {hip, knee, vlerp(knee, foot, 0.5f), foot};
                limb(m, p, 4, 0.011f * L, 0.004f * L, mat, 2);
            }
        }
    }
#undef LP
}

static void shrimp_build(double t) {
    shrimp_mesh[0].nv = shrimp_mesh[0].ni = 0;
    shrimp_mesh[1].nv = shrimp_mesh[1].ni = 0;
    for (int i = 0; i < nshrimp; i++) shrimp_build_one(&shrimp[i], i, t, shrimp[i].amano ? &shrimp_mesh[1] : &dyn_mesh);
}

/* ==================================================================== */
/* ---- the creatures, together ---- */
static GLuint vao_dyn[2], vbo_dyn[2], ibo_dyn[2];
static int dyn_ni[2];

static void creatures_init(void) {
    branch_boxes();
    roster();
    eel_init();
    shrimp_init();
}

static void creatures_gl(void) {
    fish_gl();
    for (int i = 0; i < 2; i++) {
        glGenVertexArrays(1, &vao_dyn[i]);
        glBindVertexArray(vao_dyn[i]);
        glGenBuffers(1, &vbo_dyn[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_dyn[i]);
        solid_attribs();
        glGenBuffers(1, &ibo_dyn[i]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_dyn[i]);
    }
}


static void dyn_upload(int i, const Mesh *m) {
    glBindVertexArray(vao_dyn[i]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_dyn[i]);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * SV * m->nv), m->v, GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_dyn[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(sizeof(uint32_t) * m->ni), m->ix, GL_STREAM_DRAW);
    dyn_ni[i] = m->ni;
}

static double cpu_ms[4];
static void creatures_update(double t, double wt) {
    double c0 = now();
    agents_at(t, wt);
    double c1 = now();
    fish_upload();
    eel_build(t);
    shrimp_build(t);
    dyn_upload(0, &dyn_mesh);
    dyn_upload(1, &shrimp_mesh[1]);
    frame_no++;
    double c2 = now();
    cpu_ms[0] += (c1 - c0) * 1000; cpu_ms[1] += (c2 - c1) * 1000; cpu_ms[2] += 1;
}

static void dyn_draw(int i) {
    if (!dyn_ni[i]) return;
    set_common(p_solid);
    glBindVertexArray(vao_dyn[i]);
    glDrawElements(GL_TRIANGLES, dyn_ni[i], GL_UNSIGNED_INT, 0);
}

static void creatures_draw(int pass) {
    fish_draw(pass == 1 ? 2 : 0);
    dyn_draw(0);
    if (pass == 1) dyn_draw(1);
}
/* fins and the other see-through parts, blended over the scene far to near */
static void creatures_draw_translucent(void) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_FALSE);
    dyn_draw(1);
    fish_draw(1);
    glDepthMask(GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_BLEND);
}

/* --focus: the best-placed animal of a kind for a close-up */
static int creature_focus(const char *kind, V3 *at, float *size) {
    if (!strcmp(kind, "eel")) {
        *at = vmad(eel_head_p, eel_head_dir, -0.04f);
        *size = 0.075f;
        return 1;
    }
    if (!strcmp(kind, "shrimp") || !strcmp(kind, "amano")) {
        int want = !strcmp(kind, "amano"), best = -1;
        float bs = 1e9f;
        for (int i = 0; i < nshrimp; i++) {
            if (shrimp[i].amano != want) continue;
            float sc = shrimp_pos[i].z;
            if (sc < bs) { bs = sc; best = i; }
        }
        if (best < 0) return 0;
        if (opt_pose > -1000) {
            /* on top of the stone nearest the glass */
            int bs2 = 0;
            for (int i = 1; i < nstone; i++) if (stones[i].c.z - stones[i].ax.y < stones[bs2].c.z - stones[bs2].ax.y && stones[i].ax.y > 0.02f) bs2 = i;
            shrimp_spec = best;
            shrimp_spec_x = stones[bs2].c.x; shrimp_spec_z = stones[bs2].c.z;
            float topy = -1;
            for (float dz = -0.04f; dz < 0.04f; dz += 0.002f) {
                float g = ground_at(stones[bs2].c.x, stones[bs2].c.z + dz);
                if (g > topy) { topy = g; shrimp_spec_z = stones[bs2].c.z + dz; }
            }
            shrimp_spec_yaw = opt_pose * PI / 180;
            eel_build(t_now);
            shrimp_build(t_now);
            dyn_upload(0, &dyn_mesh);
            dyn_upload(1, &shrimp_mesh[1]);
        }
        *at = vadd(shrimp_pos[best], v3(0, shrimp[best].len * 0.1f, 0));
        *size = shrimp[best].len * 1.0f;
        return 1;
    }
    int sp = !strcmp(kind, "tetra") ? SP_TETRA : !strcmp(kind, "angel") ? SP_ANGEL : !strcmp(kind, "discus") ? SP_DISCUS :
             !strcmp(kind, "cory") ? SP_CORY : -1;
    if (sp < 0 || !nsp[sp]) return 0;
    int best = -1;
    float bs = -1e9f;
    for (int i = sp_first[sp]; i < sp_first[sp] + nsp[sp]; i++) {
        Agent *a = &shown[i];
        float sc = fabsf(a->f[0]) * 2 - a->p[2] * 4 - fabsf(a->f[1]) * 2;
        if (sc > bs) { bs = sc; best = i; }
    }
    Agent *a = &shown[best];
    if (opt_pose > -1000) {
        /* --pose: a specimen pose for a portrait: near the glass, side on, turned by YAW degrees */
        float yaw = opt_pose * PI / 180;
        a->p[0] = (X0 + X1) / 2; a->p[1] = sp == SP_CORY ? ground_at(a->p[0], 0.035f) + 0.03f : YW * 0.5f; a->p[2] = 0.035f;
        a->f[0] = cosf(yaw); a->f[1] = sp == SP_CORY ? -0.25f : 0; a->f[2] = sinf(yaw);
        a->roll = 0; a->bend = 0;
        fish_upload();
    }
    *at = v3(a->p[0], a->p[1], a->p[2]);
    *size = fish[best].size * (sp == SP_ANGEL ? 1.6f : 1.15f);
    return 1;
}

/* the eel's and the shrimps' skins */
static const char *CREATURE_FS = GLSL(
    void creature(int mat, inout vec3 alb, inout vec3 n, inout float spec, inout float rough, inout vec3 tr, inout float alpha) {
        if (mat == 8) {
            /* fire eel: charcoal, with red-orange lines that break into dashes and spots toward the tail */
            float f = vAux.x, sa = vAux.y;
            if (vAux.z > 1.5) {
                float e = vUV.y;
                alb = mix(vec3(0.045, 0.04, 0.045), vec3(0.7, 0.13, 0.04), smoothstep(0.55, 0.92, e));
                if (vAux.z > 2.5) alb = vec3(0.35, 0.12, 0.06);
                tr = vec3(0.25, 0.05, 0.02) * max(dot(n, uL), 0.0);
                spec = 0.1;
            } else {
                alb = mix(vec3(0.15, 0.12, 0.11), vec3(0.04, 0.038, 0.045), smoothstep(-0.7, 0.3, sa));
                float wob = 0.06 * sin(f * 70.0) + 0.04 * sin(f * 23.0);
                float s1 = smoothstep(0.11, 0.05, abs(sa - 0.32 - wob));
                float s2 = smoothstep(0.09, 0.04, abs(sa + 0.3 + wob * 0.7));
                float dash = mix(1.0, smoothstep(0.3, 0.45, fract(f * 34.0 + 0.4 * sin(f * 17.0))) * smoothstep(1.0, 0.9, fract(f * 34.0 + 0.4 * sin(f * 17.0))), smoothstep(0.45, 0.65, f));
                float dash2 = mix(1.0, smoothstep(0.4, 0.55, fract(f * 27.0 + 0.3)) * smoothstep(1.0, 0.9, fract(f * 27.0 + 0.3)), smoothstep(0.3, 0.5, f));
                float spots = step(0.72, vnoise(vec2(f * 90.0, sa * 5.0 + vAux.z * 3.0))) * smoothstep(0.6, 0.8, f);
                float red = max(max(s1 * dash, s2 * dash2 * 0.9), spots) * smoothstep(0.02, 0.06, f) * (1.0 - smoothstep(0.95, 1.0, f));
                alb = mix(alb, vec3(0.78, 0.16, 0.04), red);
                /* the eye: small, dark, in a gold ring */
                float de = length(vec2((f - 0.042) / 0.012, (sa - 0.38) / 0.22));
                alb = mix(alb, mix(vec3(0.5, 0.35, 0.1), vec3(0.01), smoothstep(0.75, 0.6, de)), smoothstep(1.0, 0.9, de));
                spec = mix(0.2, 1.0, smoothstep(1.0, 0.9, de));
                rough = mix(30.0, 120.0, smoothstep(1.0, 0.9, de));
                /* the snout's fleshy tip */
                alb = mix(alb, vec3(0.3, 0.1, 0.07), smoothstep(0.02, 0.0, f));
                tr = vec3(0.06, 0.02, 0.01);
            }
            /* sand dusting where it meets the bottom */
            float g = texture(uGround, (vP.xz - uGroundR.xy) * uGroundR.zw).r;
            alb = mix(alb, vec3(0.5, 0.43, 0.32), smoothstep(0.004, -0.001, vP.y - g) * 0.6);
        } else if (mat == 9 || mat == 10) {
            int part = int(vAux.x + 0.5);
            if (mat == 9) {
                /* cherry shrimp: deep red shell, the limbs clearer */
                alb = vec3(0.62, 0.035, 0.03);
                if (part == 0) {
                    alb = mix(alb, vec3(0.42, 0.02, 0.02), smoothstep(0.2, 0.9, vAux.y));
                    if (vAux.z >= 1.0) alb = mix(alb, vec3(0.65, 0.6, 0.18), smoothstep(0.55, 0.8, vAux.y) * smoothstep(0.45, 0.7, vUV.x) * 0.9);
                }
                if (part == 1) alb *= 0.85 + 0.25 * smoothstep(0.75, 0.95, fract(vUV.x * 6.0));
                if (part == 2 || part == 3 || part == 5 || part == 7) alb = mix(vec3(0.75, 0.22, 0.18), vec3(0.6, 0.05, 0.04), 0.4);
                if (part == 4) alb = mix(vec3(0.65, 0.06, 0.04), vec3(0.85, 0.5, 0.4), vUV.y * 0.5);
                tr = vec3(0.55, 0.08, 0.05) * (0.4 + 0.6 * max(dot(n, uL), 0.0));
                spec = 0.45; rough = 45.0;
                alpha = 1.0;
            } else {
                /* Amano: glassy grey, a pale back line, rows of red-brown dashes and dots */
                alb = vec3(0.42, 0.44, 0.4);
                float dots = 0.0;
                if (part == 1 || part == 0) {
                    float row = abs(vAux.z) > 0.2 && abs(vAux.z) < 0.85 ? 1.0 : 0.0;
                    dots = step(0.62, vnoise(vec2(vUV.x * 55.0, vUV.y * 9.0))) * row;
                    alb = mix(alb, vec3(0.45, 0.18, 0.1), dots);
                    alb = mix(alb, vec3(0.85, 0.82, 0.7), smoothstep(0.85, 0.97, vAux.z) * 0.8);
                }
                tr = vec3(0.25, 0.27, 0.24);
                spec = 0.6; rough = 60.0;
                alpha = 0.42 + 0.4 * dots + (part == 2 || part == 3 ? 0.15 : 0.0);
            }
        } else if (mat == 11) {
            alb = vec3(0.01); spec = 1.4; rough = 140.0;
        }
    });


/* ==================================================================== */
/* ---- bubbles and the motes in the water ----
 * The air stone's column, the odd pearl of oxygen off a leaf in strong
 * light, and suspended particles drifting on the current: all closed-form
 * in time, so every instance agrees. Drawn as camera-facing sprites. */
static const char *VS_SPRITE = GLSL(
    layout(location = 0) in vec2 aQ;
    layout(location = 1) in vec4 iA;    /* bubbles: centre, radius */
    uniform int uKind;                  /* 0 bubbles (instanced data), 1 motes (from the instance id) */
    uniform float uSeed;
    uniform vec2 uDrift;
    out vec2 vQ;
    out vec3 vP; out vec3 vG;
    flat out float vR;
    flat out float vK;
    void main() {
        vec3 c; float r;
        if (uKind == 0) { c = iA.xyz; r = iA.w; }
        else {
            /* a mote: a hashed home in the tank, carried round by the current */
            float id = float(gl_InstanceID);
            vec3 h = vec3(hash12(vec2(id, uSeed)), hash12(vec2(id + 0.37, uSeed * 1.3)), hash12(vec2(id * 1.7, uSeed + 3.0)));
            float W = uXR.y - uXR.x;
            float sp = 0.4 + 0.6 * hash12(vec2(id, 7.0));
            /* the drift's speed is a whole number of tank widths per 1024 s */
            float dx = floor(sp * uDrift.x * 1024.0 / W + 0.5) * W / 1024.0;
            c.x = uXR.x + mod(h.x * W + dx * uT, W);
            c.y = 0.03 + h.y * (uYW - 0.05) + 0.004 * sin(wt(0.07) + id);
            c.z = 0.01 + h.z * (uTD - 0.03) + 0.005 * sin(wt(0.05) + id * 1.3);
            c.y += 0.003 * sin(wt(0.11) * (1.0 + h.x) + id * 2.1);
            r = 0.00025 + 0.0004 * hash12(vec2(id, 11.0));
        }
        vec4 cc = place(c, vG);
        /* a sprite a little bigger than the thing, in clip space */
        vec2 off = aQ * vec2(uProj.x, uProj.y) * r * 1.25;
        if (uPass == 1) off = aQ * r * 2.0;
        gl_Position = cc + vec4(off, 0.0, 0.0);
        vQ = aQ;
        vP = c;
        vR = r;
        vK = float(uKind);
    });

static const char *FS_SPRITE = GLSL(
    in vec2 vQ;
    in vec3 vP; in vec3 vG;
    flat in float vR;
    flat in float vK;
    void main() {
        float d = length(vQ);
        if (d > 1.0) discard;
        if (uPass == 1) discard;
        vec3 lightc = uSun * exp(-uExt * max(uYW - vP.y, 0.0)) * shadow(vP);
        vec3 col; float a;
        if (vK < 0.5) {
            /* a bubble: a bright rim, the lamp caught on top, a little glow under */
            float z = sqrt(max(1.0 - d * d, 0.0));
            vec3 n = vec3(vQ.x, vQ.y, -z);
            float rim = pow(1.0 - z, 2.5);
            float hl = pow(max(dot(normalize(n), normalize(vec3(-0.25, 0.75, -0.6))), 0.0), 40.0);
            float under = smoothstep(0.5, 0.0, length(vQ - vec2(0.1, -0.45))) * 0.5;
            col = (lightc * 0.35 + uAmbTop * 0.9) * rim + lightc * (hl * 2.2 + under * 0.4) + uFog * 0.3;
            a = clamp(rim * 0.85 + hl + under * 0.3 + 0.06, 0.0, 1.0);
            col = water(col, vP, vG);
            o0 = vec4(col * a, a) ;
        } else {
            /* a mote: catches the light where the light is */
            float c = caustic(vP).g;
            float b = (1.0 - smoothstep(0.0, 1.0, d)) * 0.6;
            col = (lightc * (0.3 + 0.5 * c) + uAmbTop * 0.5) * b * vec3(0.95, 0.92, 0.8) * 0.5;
            col = water(col, vP, vG) * b;
            o0 = vec4(col, 0.0);
        }
        o1 = vec4(0.0);
    });

static GLuint p_sprite, vao_sprite, vbo_quad, vbo_bub;
#define MAXBUB 256
static float bub[MAXBUB * 4];
static int nbub, nmotes;

static void sprites_gl(void) {
    p_sprite = mk_prog(VS_SPRITE, NULL, FS_SPRITE, NULL);
    static const float q[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenVertexArrays(1, &vao_sprite);
    glBindVertexArray(vao_sprite);
    glGenBuffers(1, &vbo_quad);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
    glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &vbo_bub);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bub);
    glBufferData(GL_ARRAY_BUFFER, sizeof bub, NULL, GL_STREAM_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    nmotes = (int)(tank_w() * 2200);
}

/* the bubbles now: from the air stone a steady column; off the leaves, in
 * strong light, a pearl now and then */
static void bubbles_update(double t) {
    nbub = 0;
    float y0 = sand_h(bubbler_x, bubbler_z) + 0.014f;
    const double per = 0.055;
    long id1 = (long)floor(t / per);
    for (long id = id1; id > id1 - 60 && nbub < MAXBUB - 40; id--) {
        uint32_t h = (uint32_t)id;
        double tb = (id + 0.8 * hf(h, 1, 50)) * per;
        float age = (float)(t - tb);
        if (age < 0) continue;
        float r = 0.0011f + 0.0016f * hf(h, 2, 50);
        float v = 0.17f + 0.12f * r / 0.0027f;
        float y = y0 + v * age - 0.5f * fmaxf(0, 0.25f - age) * v;
        if (y > YW - r * 0.5f) continue;
        float rise = (y - y0) / (YW - y0);
        float x = bubbler_x + (hf(h, 3, 50) - 0.5f) * 0.008f + 0.0035f * sinf(age * 13 + hf(h, 4, 50) * 6) + outflow_dir * -0.02f * rise * rise;
        float z = bubbler_z + (hf(h, 5, 50) - 0.5f) * 0.008f + 0.0035f * cosf(age * 11 + hf(h, 6, 50) * 6) - 0.01f * rise;
        float *o = bub + nbub * 4;
        o[0] = x; o[1] = y; o[2] = z; o[3] = r * (1 + 0.15f * rise);
        nbub++;
    }
    /* pearling: oxygen off the leaves, only in full light */
    if (lit.night < 0.2f && leaves[LM_TINY].n) {
        const double pp = 1.7;
        long k1 = (long)floor(t / pp);
        for (long k = k1; k > k1 - 8 && nbub < MAXBUB; k--) {
            uint32_t h = (uint32_t)k * 7u + 3u;
            double tb = (k + hf(h, 1, 60)) * pp;
            float age = (float)(t - tb);
            if (age < 0) continue;
            const float *L = leaves[LM_TINY].v + (size_t)(hash32(h) % (uint32_t)leaves[LM_TINY].n) * LI;
            float r = 0.0006f + 0.0005f * hf(h, 2, 60);
            float y = L[1] + 0.12f * age;
            if (y > YW - r) continue;
            float *o = bub + nbub * 4;
            o[0] = L[0] + 0.002f * sinf(age * 9); o[1] = y; o[2] = L[2]; o[3] = r;
            nbub++;
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bub);
    glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(sizeof(float) * 4 * nbub), bub);
}

static void sprites_draw(void) {
    set_common(p_sprite);
    glBindVertexArray(vao_sprite);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_FALSE);
    glUniform1f(U(p_sprite, "uSeed"), (float)(opt_seed % 1000));
    float ld = cpu_load(wall());
    glUniform2f(U(p_sprite, "uDrift"), outflow_dir * (0.004f + 0.01f * ld), 0);
    glUniform1i(U(p_sprite, "uKind"), 1);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, nmotes);
    glUniform1i(U(p_sprite, "uKind"), 0);
    if (nbub) glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, nbub);
    glDepthMask(GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_BLEND);
}

/* ==================================================================== */
/* ---- the overlays: a thermometer on the glass, a small caption ----
 * Only on the front output (or the only one). The thermometer reads the
 * CPU package temperature on a 30-100 scale; the caption names the host
 * and how long the screen has been locked. */
static const char *VS_OVER = GLSL(
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    layout(location = 2) in vec4 aCol;
    uniform vec2 uRes;
    out vec2 vUV;
    out vec4 vCol;
    out vec2 vPx;
    void main() {
        vUV = aUV; vCol = aCol; vPx = aPos;
        gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
    });
/* text from the atlas (aCol.a > 0), or the thermometer drawn by distance (aCol.a < 0) */
static const char *FS_OVER = GLSL(
    uniform sampler2D uAtlas;
    uniform vec4 uTh;                   /* the thermometer: x, y (top of the tube), height, width */
    uniform float uLevel;               /* 0..1 of the scale */
    uniform vec3 uLight;
    in vec2 vUV;
    in vec4 vCol;
    in vec2 vPx;
    float sdRound(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
    void main() {
        if (vCol.a >= 0.0) { o0 = vec4(vCol.rgb, vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r); return; }
        /* in thermometer units: x across (-1..1 the tube), y down from the top */
        float w = uTh.w;
        vec2 p = (vPx - uTh.xy) / w;
        float H = uTh.z / w;
        /* the backing card, the glass tube, the bulb, the suction cup */
        float card = sdRound(p - vec2(0.0, H * 0.5), vec2(1.0, H * 0.5), 0.45);
        float tube = sdRound(p - vec2(0.0, H * 0.5), vec2(0.22, H * 0.47), 0.22);
        float bulb = length(p - vec2(0.0, H * 0.93)) - 0.42;
        float glass = min(tube, bulb);
        float cup = length((p - vec2(0.0, -0.55)) * vec2(1.0, 1.6)) - 0.8;
        float stem = sdRound(p - vec2(0.0, -0.1), vec2(0.14, 0.3), 0.1);
        float aa = 1.5 / w;
        vec4 c = vec4(0.0);
        /* rubber cup, slightly see-through */
        float cupm = 1.0 - smoothstep(-aa, aa, min(cup, stem));
        c = mix(c, vec4(vec3(0.55, 0.6, 0.58) * uLight * 0.5, 0.55), cupm);
        /* the card: frosted white with printed ticks */
        float cardm = 1.0 - smoothstep(-aa, aa, card);
        vec3 cardc = vec3(0.86, 0.88, 0.86) * uLight;
        float y0 = H * 0.08, y1 = H * 0.82;
        float ty = (p.y - y0) / (y1 - y0);
        float minor = abs(fract(ty * 35.0 + 0.5) - 0.5) / 35.0 * (y1 - y0);
        float major = abs(fract(ty * 7.0 + 0.5) - 0.5) / 7.0 * (y1 - y0);
        float tick = (1.0 - smoothstep(0.02, 0.05, minor)) * step(0.45, abs(p.x)) * step(abs(p.x), 0.62);
        tick = max(tick, (1.0 - smoothstep(0.03, 0.07, major)) * step(0.4, abs(p.x)) * step(abs(p.x), 0.8));
        tick *= step(-0.01, ty) * step(ty, 1.01);
        cardc = mix(cardc, vec3(0.15, 0.17, 0.18) * uLight, tick);
        c = mix(c, vec4(cardc, 0.82), cardm);
        /* the glass and its liquid */
        float gm = 1.0 - smoothstep(-aa, aa, glass);
        float lev = y1 - (y1 - y0) * uLevel;
        float liquid = (1.0 - smoothstep(-aa, aa, sdRound(p - vec2(0.0, (lev + H * 0.93) * 0.5), vec2(0.08, (H * 0.93 - lev) * 0.5), 0.08)));
        liquid = max(liquid, 1.0 - smoothstep(-aa, aa, length(p - vec2(0.0, H * 0.93)) - 0.3));
        vec3 gc = mix(vec3(0.82, 0.86, 0.86), vec3(0.95), smoothstep(0.1, 0.0, abs(p.x + 0.1))) * uLight;
        c = mix(c, vec4(gc, 0.35 + 0.4 * smoothstep(0.15, 0.22, abs(p.x))), gm);
        c = mix(c, vec4(vec3(0.72, 0.08, 0.06) * uLight * 1.3, 0.95), liquid);
        /* a highlight running down the glass */
        float hl = smoothstep(0.05, 0.0, abs(p.x + 0.09)) * gm * step(p.y, H * 0.86);
        c.rgb += vec3(0.5) * hl * uLight;
        o0 = vec4(c.rgb, c.a);
    });

static GLuint p_over, vao_over, vbo_over, tex_atlas;
static int st_small, st_num;
static float th_rect[4];
static int over_ok;

static void overlay_gl(void) {
    p_over = mk_prog(VS_OVER, NULL, FS_OVER, NULL);
    glGenVertexArrays(1, &vao_over);
    glBindVertexArray(vao_over);
    glGenBuffers(1, &vbo_over);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_over);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void *)8);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
    for (int i = 0; i < 3; i++) glEnableVertexAttribArray((GLuint)i);
    glGenTextures(1, &tex_atlas);
    sprites_gl();
}

/* the sizes follow the front output's height */
static int over_h;
static void overlay_layout(void) {
    over_h = H;
    nstyles = 0;
    float px = H / 1200.0f;
    st_small = style(1, fmaxf(15 * px, 11));
    st_num = style(0, fmaxf(12 * px, 9));
    build_atlas();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    over_ok = 1;
}

static float temp_shown = -1;
static void overlay_draw(double wt) {
    /* only the front output carries them */
    if (span_me >= 0 && span_me != front_out) return;
    if (!over_ok || over_h != H) overlay_layout();
    tvn = 0;
    float px = H / 1200.0f;
    /* the thermometer: stuck inside the glass, near the right edge, high up */
    float tw = 13 * px, th = 230 * px;
    float tx = W - 70 * px, ty = H * 0.16f;
    th_rect[0] = tx; th_rect[1] = ty; th_rect[2] = th; th_rect[3] = tw;
    float none[4] = {0, 0, 0, -1};
    quad_v(tx - tw * 1.6f, ty - tw * 1.8f, 0, 0, none); quad_v(tx + tw * 1.6f, ty - tw * 1.8f, 0, 0, none);
    quad_v(tx + tw * 1.6f, ty + th + tw, 0, 0, none);
    quad_v(tx - tw * 1.6f, ty - tw * 1.8f, 0, 0, none); quad_v(tx + tw * 1.6f, ty + th + tw, 0, 0, none);
    quad_v(tx - tw * 1.6f, ty + th + tw, 0, 0, none);
    int nth = tvn;
    /* its numbers */
    float lv = mixf(0.16f, 1.0f, 1 - lit.night);
    float ink[4] = {0.13f * lv, 0.15f * lv, 0.16f * lv, 0.9f};
    float y0 = ty + th * 0.08f, y1 = ty + th * 0.82f;
    for (int k = 0; k <= 7; k++) {
        char b[8];
        snprintf(b, sizeof b, "%d", 30 + k * 10);
        if (k % 2) continue;
        text(st_num, tx - tw * 0.9f, y1 - (y1 - y0) * k / 7 + styles[st_num].px * 0.35f, 1, b, ink, 0);
    }
    /* the caption: the host, and how long it has been away */
    double away = opt_away >= 0 ? opt_away : wt - (double)locked_at;
    if (away < 0) away = 0;
    long a = (long)away;
    char cap[160], hostu[64];
    for (int i = 0; i < 63 && host[i]; i++) { hostu[i] = (char)tolower((unsigned char)host[i]); hostu[i + 1] = 0; }
    if (a >= 3600) snprintf(cap, sizeof cap, "%s  \x7f  away %ldh %02ldm", hostu, a / 3600, (a / 60) % 60);
    else snprintf(cap, sizeof cap, "%s  \x7f  away %ldm", hostu, a / 60);
    float capc[4] = {0.78f * lv, 0.8f * lv, 0.76f * lv, 0.55f};
    float shc[4] = {0, 0, 0, 0.45f};
    float cx = 34 * px, cy = H - 14 * px;
    text(st_small, cx + 1, cy + 1, -1, cap, shc, 1.5f * px);
    text(st_small, cx, cy, -1, cap, capc, 1.5f * px);

    float tC = sens.temp;
    if (temp_shown < 0) temp_shown = tC;
    temp_shown = mixf(temp_shown, tC, 0.05f);
    glUseProgram(p_over);
    glUniform2f(U(p_over, "uRes"), (float)W, (float)H);
    glUniform1i(U(p_over, "uAtlas"), 0);
    glUniform4fv(U(p_over, "uTh"), 1, th_rect);
    glUniform1f(U(p_over, "uLevel"), clampf((temp_shown - 30) / 70, 0, 1));
    float lc[3] = {lv, lv, lv * (1 - lit.night * 0.15f)};
    glUniform3fv(U(p_over, "uLight"), 1, lc);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_over);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_over);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, nth / 8);
    glDrawArrays(GL_TRIANGLES, nth / 8, (tvn - nth) / 8);
    glDisable(GL_BLEND);
}

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
    float sc = opt_scale > 0 ? opt_scale : clampf(sqrtf(1.3e6f / ((float)W * H)), 0.5f, 1.0f);
    sw_ = (int)lroundf(W * sc); sh_ = (int)lroundf(H * sc);
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
        glUniform1i(U(p_leaf, "uTiny"), m == LM_TINY);
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
static double bake_ms;
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
    bake_ms = (now() - t0) * 1000;
}

/* --bench: GPU time of each pass */
static GLuint tq[12];
static double tq_sum[12];
static int tq_on, tq_n;
static const char *tq_name[12] = {"caustics", "shadow", "relight", "taa", "shafts", "composite", "overlay", "animals", "plants", "see-thru", "", ""};
static void tq_begin(int i) { if (tq_on) glBeginQuery(GL_TIME_ELAPSED, tq[i]); }
static void tq_end(void) { if (tq_on) glEndQuery(GL_TIME_ELAPSED); }
static void tq_collect(void) {
    if (!tq_on) return;
    for (int i = 0; i < 10; i++) { GLuint64 ns = 0; glGetQueryObjectui64v(tq[i], GL_QUERY_RESULT, &ns); tq_sum[i] += ns * 1e-6; }
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
    bubbles_update(t);
    if (opt_focus) {
        int was = focus_on;
        focus_on = creature_focus(opt_focus, &focus_at, &focus_size);
        if (focus_on || was) { setup_view(); if (!was) bake_static(); }
    }
    setup_view();

    tq_begin(0);
    render_caustics(t);
    tq_end();

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
    bind_scene_textures(tex_shadow);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_sdepth);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_main);
    glBlitFramebuffer(0, 0, sw_, sh_, 0, 0, sw_, sh_, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);
    glViewport(0, 0, sw_, sh_);
    glDisable(GL_DEPTH_TEST);
    set_common(p_relight);
    for (int i = 0; i < 3; i++) {
        char nm[8];
        snprintf(nm, sizeof nm, "uG%d", i);
        glUniform1i(U(p_relight, nm), 3 + i);
        glActiveTexture(GL_TEXTURE3 + i);
        glBindTexture(GL_TEXTURE_2D, tex_g[i]);
    }
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    tq_end();
    glEnable(GL_DEPTH_TEST);
    tq_begin(7);
    creatures_draw(0);
    tq_end();
    tq_begin(8);
    draw_leaves(0);
    tq_end();
    tq_begin(9);
    creatures_draw_translucent();
    sprites_draw();
    tq_end();
    glDisable(GL_DEPTH_TEST);

    /* light shafts, at a quarter size, for the antialiasing to fold in */
    float vr[4] = {view_x, view_y, view_w, view_h}, cc[3] = {cam_cx, cam_cy, cam_f};
    tq_begin(4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_sh);
    glViewport(0, 0, (sw_ + 3) / 4, (sh_ + 3) / 4);
    set_common(p_shafts);
    glUniform4fv(U(p_shafts, "uView"), 1, vr);
    glUniform3fv(U(p_shafts, "uCanCam"), 1, cc);
    glUniform1i(U(p_shafts, "uAux"), 3);
    glUniform1f(U(p_shafts, "uJit"), (float)(frame_no % 61));
    glUniform1f(U(p_shafts, "uShaftK"), 0.06f * lit.shaft);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    tq_end();

    /* temporal antialiasing */
    tq_begin(3);
    int prev = taa_cur;
    taa_cur ^= 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_taa[taa_cur]);
    glViewport(0, 0, sw_, sh_);
    glUseProgram(p_taa);
    glUniform1i(U(p_taa, "uCur"), 3);
    glUniform1i(U(p_taa, "uHist"), 4);
    glUniform1f(U(p_taa, "uAlpha"), taa_frames ? taa_alpha : 1.0f);
    glUniform1i(U(p_taa, "uClamp"), taa_clamp);
    glUniform1i(U(p_taa, "uAux"), 5);
    glUniform1i(U(p_taa, "uShafts"), 6);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_cur);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_taa[prev]);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex_sh);
    glActiveTexture(GL_TEXTURE0);
    fullscreen();
    taa_frames++;
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
    glUniform1f(U(p_comp, "uExpo"), 1.0f);
    glUniform1f(U(p_comp, "uCA"), focus_on ? 0.0f : 1.0f);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_aux);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_taa[taa_cur]);
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
        "usage: aquarium [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--hour H] [--load 0..1] [--temp C] [--away SECONDS] [--no-overlay]\n"
        "                [--msaa 0|2|4|8] [--scale S] [--focus tetra|angel|discus|cory|eel|shrimp|amano [--pose YAW]]\n"
        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "                [--render-scale F]\n"
        "  --render-scale F  with --layer, draw F x the output's native pixels (0 < F <= 1; default 1)\n");
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
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
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
        else if (!strcmp(a, "-pose") && i + 1 < argc) opt_pose = strtof(argv[++i], NULL);
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
        glGenQueries(12, tq);
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
        printf("aquarium: GL renderer: %s; the still scene baked in %.0f ms\n", (const char *)glGetString(GL_RENDERER), bake_ms);
        for (int i = 0; i < 10; i++) printf("  %-10s %.3f ms\n", tq_name[i], tq_sum[i] / (tq_n ? tq_n : 1));
        printf("  cpu: sim %.3f ms, eel+shrimp+upload %.3f ms\n", cpu_ms[0] / cpu_ms[2], cpu_ms[1] / cpu_ms[2]);
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
