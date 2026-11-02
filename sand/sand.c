/*
 * sand — a falling-sand world that runs itself: a side-on diorama of a
 * country in cross-section, from the clouds down through a volcano, a range,
 * a lake, downs and forest to the sea, and below them bedded rock, caves and
 * the magma under everything.
 *
 * The world is a grid of cells (~4-5 screen px each), simulated on the GPU
 * as a cellular automaton in integer textures, 60 ticks a second:
 *
 *   MOVE   a Margolus pass: the grid in 2x2 blocks, offset by (0,0) and
 *          (1,1) on alternate ticks; each block rearranges its four cells at
 *          once (heavier falls through lighter, powders slide off piles at
 *          their own angle of repose, liquids run sideways on a direction bit
 *          so they level, gases rise and wander, flung lava flies), so no
 *          two moves can conflict and nothing is lost
 *   REACT  every cell from its 3x3 neighbourhood: temperature diffuses (in
 *          1/16 K, at each material's conductivity) and relaxes to the
 *          climate (a lapse rate up the mountains, a geotherm down to the
 *          magma, hot spots round the chamber and throat); phase changes
 *          follow it; fire, growth, rain and the rest are "pull" rules whose
 *          dice both cells of a pair can roll, so they always agree
 *   WIND   now and then the cloud deck shifts bodily a cell downwind
 *   SHIFT  now and then a stretch of upper crust is lifted or let down
 *
 * Materials: stone, basalt, obsidian, coal, dirt, grass, sand, wet sand,
 * silt, snow, ice (floats), water, oil, lava, magma, steam (vapour, visible
 * when hot), smoke, cloud, fire, embers, ash and cinders, seeds, grass
 * blades and flowers, wood and leaves, leaf litter, lightning.
 *
 * Nature runs it, all from the seed and the tick:
 *   - the year turns every 24 minutes, starting from the real season: the
 *     snowline rises and falls, leaves colour, drop in autumn and come back
 *     in spring; trees grow from seeds a cell at a time, branch, leaf out
 *   - water evaporates off the sea and lakes and rises as vapour that joins
 *     the clouds; cumulus drift on the wind; storms build wet clouds that
 *     rain, or snow where it's cold (flakes melt to rain as they fall); a
 *     canopy holds and drips its rain; lightning strikes in the thick of a
 *     storm and can set the forest burning (rain puts it out); running water
 *     picks up silt and drops it where it slows; lakes freeze over
 *   - the volcano: quiet, a rumble as the magma column rises and smoke
 *     pours out, then a fountain of lava that flies out, rains cinders and
 *     runs down the flanks, setting the trees in its way alight, and cools
 *   - the open sea beyond the edge holds the sea at its level; every two
 *     seconds a reduction pass reads back per-column counts, and the slow
 *     geology lifts the country where it's worn down and lets it down where
 *     it's piled up, so the world keeps its shape however long it runs
 *
 * Rendering, at cell resolution then up to the screen: per-grain colour,
 * strata folded through the rock, skylight propagated down through the grid
 * (slanting with the sun, shadowed by canopy, cloud and water), an even
 * cutaway light on the cross-section, the glow of hot things lighting their
 * surroundings and blooming, soft smoke and steam, water clear in the
 * shallows and inky in the depths, a lit lip on ground under open air and
 * pixel outlines on rock by the caves, rain streaks and snowflakes, the sky
 * for the real local time (sun, moon, stars, two far ranges in the haze),
 * and a small pixel-font caption of the time away (swaylock's start time).
 *
 * Machine tie-ins, standalone only: CPU load makes the storms wetter and the
 * volcano angrier; CPU temperature warms the climate (the snowline climbs).
 * They're off when spanning: the instances couldn't agree on a reading
 * without talking, and a single differing cell would split their worlds.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * Every instance simulates the whole wall's world, deterministically (integer
 * state and hashing, ticks counted off the shared epoch, controllers fed only
 * by GPU readbacks every instance makes alike) and shows its own rect of it,
 * so the world runs on continuously across the monitors. The output under
 * the middle of the layout is the front one: the world is scaled to it and
 * the caption goes on it; a portrait output gets the volcano and its plumbing
 * from sky to magma. An instance started late (a restart) fast-forwards from
 * the epoch to catch up, as many ticks as fit in 12 ms a frame.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen
 * (fast-forwarding the sim to --at); --bench N times N frames, per pass.
 * --hour H fixes the time of day, --season 0..1 the time of year (0 is
 * midwinter), --load/--temp the tie-ins; --cell PX the cell size; --when
 * lists the coming eruptions, storms and lightning strikes.
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
#define HZ 60                          /* sim ticks a second */

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 60;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the world started at */
static float opt_hour = -1;            /* force the local time of day (hours) */
static float opt_load = -1, opt_temp = -1;   /* force the machine tie-ins */
static int opt_cell;                   /* force the cell size (px) */
static int opt_caption = 1;
static float opt_season = -1;         /* force the time of year: 0 midwinter, .25 spring, .5 summer, .75 autumn */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "sand: %s\n", what); exit(1); }

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
static int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }

/* ---- hashing and noise (CPU side: world generation and the event schedule) ---- */
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static uint32_t hash3(int32_t a, int32_t b, uint32_t c) {
    return hash32((uint32_t)a * 0x9E3779B1u ^ hash32((uint32_t)b * 0x85EBCA77u ^ hash32(c * 0xC2B2AE3Du + opt_seed)));
}
static float h01(int32_t a, int32_t b, uint32_t c) { return (hash3(a, b, c) >> 8) * (1.0f / 16777216.0f); }

static float vnoise1(float x, uint32_t salt) {
    float i = floorf(x), f = x - i;
    float a = h01((int)i, 0, salt), b = h01((int)i + 1, 0, salt);
    f = f * f * (3 - 2 * f);
    return a + (b - a) * f;
}
static float vnoise2(float x, float y, uint32_t salt) {
    float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy;
    int X = (int)ix, Y = (int)iy;
    float a = h01(X, Y, salt), b = h01(X + 1, Y, salt), c = h01(X, Y + 1, salt), d = h01(X + 1, Y + 1, salt);
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    return mixf(mixf(a, b, fx), mixf(c, d, fx), fy);
}
static float fbm1(float x, int oct, uint32_t salt) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) { s += a * (vnoise1(x, salt + i * 101) * 2 - 1); n += a; x *= 2.03f; a *= 0.5f; }
    return s / n;
}
static float fbm2(float x, float y, int oct, uint32_t salt) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) { s += a * (vnoise2(x, y, salt + i * 101) * 2 - 1); n += a; x *= 2.03f; y *= 2.03f; a *= 0.5f; }
    return s / n;
}

/* ---- sensors (only steer the climate when standalone) ---- */
static char temp_path[300];
static time_t locked_at;
static float sens_temp = -1, sens_load = -1;
static unsigned long long cpu_busy0, cpu_total0;

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

static void read_cpu(unsigned long long *busy, unsigned long long *total) {
    char line[256];
    slurp_line("/proc/stat", line, sizeof line);
    unsigned long long v[8] = {0};
    sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    *total = 0;
    for (int i = 0; i < 8; i++) *total += v[i];
    *busy = *total - v[3] - v[4];
}

/* a reading, eased: the climate changes slowly whatever the machine does */
static void read_sensors(void) {
    if (temp_path[0]) {
        float t = read_long(temp_path, 0) / 1000.0f;
        if (t > 5 && t < 120) sens_temp = sens_temp < 0 ? t : sens_temp + (t - sens_temp) * 0.1f;
    }
    unsigned long long b, t;
    read_cpu(&b, &t);
    if (cpu_total0 && t > cpu_total0) {
        float l = (float)(b - cpu_busy0) / (float)(t - cpu_total0);
        sens_load = sens_load < 0 ? l : sens_load + (l - sens_load) * 0.1f;
    }
    cpu_busy0 = b; cpu_total0 = t;
}

/* ---- the world's geometry ---- */
static int CS = 4;                     /* cell size, px */
static int NX, NY;                     /* world size, cells; row 0 at the bottom */
static float wx0, wy0;                 /* layout px of the world's top-left */
static float vw_x, vw_y, vw_w, vw_h;   /* this output's rect in layout px */
static int front_me = 1;               /* this output carries the caption */
static float fr_x, fr_y, fr_w, fr_h;   /* the front output, layout px */

/* world features, cells */
static int j_sea, j_mag, cb0, cb1;     /* sea level, top of the magma floor, cloud band */
static int vent_x, vent_top, vent_bot, crater_w;   /* the volcano's throat */
static int ch_x, ch_y, ch_rx, ch_ry;   /* its magma chamber */
static int lake_x, lake_w, sea_side, sea_end;
static int mt_x0, mt_x1;               /* the main range */
static float Ufeat;                    /* feature scale, cells */
static float lapse;                    /* K per row above the sea */
static float *surf0;                   /* generated ground height per column */

static void setup_world(void) {
    float bx0 = 0, by0 = 0, bx1 = W, by1 = H;
    vw_x = 0; vw_y = 0; vw_w = W; vw_h = H;
    fr_x = 0; fr_y = 0; fr_w = W; fr_h = H;
    int tall = -1;
    if (span_me >= 0) {
        bx0 = by0 = 1e9f; bx1 = by1 = -1e9f;
        for (int i = 0; i < span_n; i++) {
            bx0 = fminf(bx0, span[i][0]); bx1 = fmaxf(bx1, span[i][0] + span[i][2]);
            by0 = fminf(by0, span[i][1]); by1 = fmaxf(by1, span[i][1] + span[i][3]);
        }
        /* the front output: the one under the middle of the layout */
        float mx = (bx0 + bx1) / 2;
        int front = 0;
        for (int i = 0; i < span_n; i++)
            if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
        front_me = span_me == front;
        fr_x = span[front][0]; fr_y = span[front][1]; fr_w = span[front][2]; fr_h = span[front][3];
        vw_x = span[span_me][0]; vw_y = span[span_me][1]; vw_w = span[span_me][2]; vw_h = span[span_me][3];
        float best = 1.25f;
        for (int i = 0; i < span_n; i++)
            if (span[i][3] / span[i][2] > best) { best = span[i][3] / span[i][2]; tall = i; }
    }
    float bw = bx1 - bx0, bh = by1 - by0;
    CS = opt_cell > 0 ? opt_cell : clampi((int)lrintf(fminf(fr_w, fr_h) / 270.0f), 3, 10);
    while (opt_cell <= 0 && ceilf(bw / CS) * ceilf(bh / CS) > 720000) CS++;
    NX = (int)ceilf(bw / CS); NY = (int)ceilf(bh / CS);
    wx0 = bx0; wy0 = by0;

    /* rows of the front output, and the scale of things in it */
    int f_bot = NY - (int)ceilf((fr_y + fr_h - by0) / CS);
    int f_h = (int)(fr_h / CS), f_w = (int)(fr_w / CS);
    int portrait = fr_h > fr_w * 1.2f;
    Ufeat = fminf(f_h, f_w * 0.9f);
    j_sea = f_bot + (int)(f_h * (portrait ? 0.46f : 0.36f));
    cb0 = clampi(f_bot + (int)(f_h * 0.79f), j_sea + 20, NY - 12);
    cb1 = clampi(f_bot + (int)(f_h * 0.93f), cb0 + 6, NY - 3);
    j_mag = 3;

    /* the volcano: on a portrait output if there is one, else to one side */
    uint32_t s = opt_seed;
    if (tall >= 0) vent_x = (int)((span[tall][0] + span[tall][2] * 0.5f - bx0) / CS);
    else vent_x = (int)(NX * ((hash32(s ^ 77) & 1) ? 0.76f : 0.24f));
    if (portrait && tall < 0) vent_x = (int)(NX * ((hash32(s ^ 77) & 1) ? 0.64f : 0.36f));
    sea_side = vent_x > NX / 2 ? -1 : 1;           /* -1: the sea's on the left */
    int sea_w = (int)fmaxf(NX * 0.12f, 30);
    sea_end = sea_side < 0 ? sea_w : NX - 1 - sea_w;
    float Hv = Ufeat * 0.27f;
    int vhalf = (int)(Hv * 1.25f);
    int toward = -sea_side * -1;                   /* direction from the volcano to the sea */
    toward = sea_side < 0 ? -1 : 1;
    int mhalf = (int)fmaxf(NX * 0.075f, 26);
    int mc = vent_x + toward * (vhalf + mhalf / 2 + (int)(NX * 0.02f));
    mt_x0 = mc - mhalf; mt_x1 = mc + mhalf;
    int medge = toward > 0 ? mt_x1 : mt_x0;
    lake_x = medge + (int)((sea_end - medge) * 0.3f);
    lake_w = clampi((int)(NX * 0.035f), 12, 46);
    crater_w = clampi((int)(Hv * 0.12f), 4, 10);
    vent_top = j_sea + (int)Hv;
    ch_rx = clampi((int)(Ufeat * 0.09f), 12, 36);
    ch_ry = clampi((int)(Ufeat * 0.05f), 7, 18);
    ch_x = vent_x;
    ch_y = clampi(j_sea - (int)(Ufeat * 0.5f), j_mag + ch_ry + 6, j_sea - ch_ry - 20);
    vent_bot = ch_y;
    lapse = 23.0f / (Ufeat * 0.4f);
}

/* ---- materials ---- */
enum {
    M_AIR, M_STONE, M_BASALT, M_OBSID, M_DIRT, M_GRASS, M_SAND, M_WETSAND, M_SNOW, M_ICE,
    M_WATER, M_OIL, M_LAVA, M_STEAM, M_SMOKE, M_CLOUD, M_FIRE, M_WOOD, M_LEAF, M_EMBER,
    M_ASH, M_SEED, M_PLANT, M_MAGMA, M_SILT, M_BOLT, M_COAL, M_LITTER
};
#define TIP 128u                       /* x bit 7: a growing tip (wood) */
#define DIRR 64u                       /* x bit 6: a liquid running right */
#define K16(k) ((int)((k) * 16))

static uint16_t *cells;                /* NX * NY * 4: material|flags|shade<<8, T (K*16), life, aux */

static void put(int x, int y, int m, float T, int life, int aux) {
    if (x < 0 || y < 0 || x >= NX || y >= NY) return;
    uint16_t *c = cells + 4 * ((size_t)y * NX + x);
    c[0] = (uint16_t)((c[0] & 0xff00) | m);
    c[1] = (uint16_t)clampi(K16(T), 0, 65535);
    c[2] = (uint16_t)life;
    c[3] = (uint16_t)aux;
}
static int mat_at(int x, int y) {
    if (x < 0 || y < 0 || x >= NX || y >= NY) return M_STONE;
    return cells[4 * ((size_t)y * NX + x)] & 63;
}

/* the climate, the same as the shaders' Tenv() (K): a lapse rate above
 * the ground, a geotherm below it, a hot spot under the volcano */
static float season_k;                 /* the current seasonal offset, K */
static float tsea = 286;
static float tenv(int x, int y) {
    float s = surf0[clampi(x, 0, NX - 1)];
    float ta = tsea + season_k - lapse * (fmaxf(y, j_sea) - j_sea);
    float t = ta;
    if (y < s) {
        float ts = tsea + season_k - lapse * (fmaxf(s, j_sea) - j_sea);
        float d = (s - y) / (float)(s - j_mag);
        t = ts + (520 - ts) * smooth(0, 1, d) + 1100 * expf(-(y - j_mag) / 7.0f);
    }
    float dx = (x - vent_x) / (float)(crater_w + 4);
    if (y < vent_top - 2) t += 620 * expf(-dx * dx) * smooth(vent_bot - 30, vent_bot, y);
    float ex = (x - ch_x) / (ch_rx + 6.0f), ey = (y - ch_y) / (ch_ry + 6.0f);
    t += 900 * expf(-2 * (ex * ex + ey * ey));
    return t;
}

/* a tree, drawn (the starting forest) */
static void tree(int x, int y, int ht, int conifer, uint32_t salt) {
    for (int j = 0; j < ht; j++) {
        int life = ht - 1 - j;
        int e = conifer ? 1 + (life * 3) / ht + (life > 2) : 0;
        put(x, y + j, M_WOOD, 290, life, e | (conifer ? 256 : 0));
    }
    if (conifer) {
        for (int j = 2; j <= ht + 1; j++) {
            int r = (int)((ht + 1 - j) * 0.38f + 1);
            for (int i = -r; i <= r; i++) {
                if (!i && j < ht) continue;
                if (h01(x + i, y + j, salt) < 0.12f && abs(i) == r) continue;
                if (mat_at(x + i, y + j) == M_AIR) put(x + i, y + j, M_LEAF, 290, 0, 1 | 256);
            }
        }
        cells[4 * ((size_t)(y + ht - 1) * NX + x) + 3] = 3 | 256;
        return;
    }
    /* branches, then leafy crowns on their ends and the top */
    int ends[8][2], ne = 0;
    ends[ne][0] = x; ends[ne][1] = y + ht - 1; ne++;
    for (int b = 0; b < 3; b++) {
        int by = y + ht / 2 + b * ht / 6 + (int)(h01(x, b, salt) * 3);
        int dir = (b + (hash3(x, y, salt) & 1)) & 1 ? 1 : -1;
        int len = 2 + (int)(h01(x, b + 7, salt) * ht / 4.0f);
        int bx = x;
        for (int k = 0; k < len; k++) {
            bx += dir;
            if (k & 1) by++;
            if (mat_at(bx, by) == M_AIR) put(bx, by, M_WOOD, 290, 0, 0);
        }
        if (ne < 8) { ends[ne][0] = bx; ends[ne][1] = by; ne++; }
    }
    for (int e = 0; e < ne; e++) {
        int r = e == 0 ? 5 + ht / 7 : 3 + ht / 10;
        cells[4 * ((size_t)ends[e][1] * NX + ends[e][0]) + 3] = (uint16_t)(r + 1);
        for (int j = -r; j <= r; j++)
            for (int i = -r - 1; i <= r + 1; i++) {
                float d = sqrtf(i * i * 0.7f + j * j * 1.1f) + vnoise2((ends[e][0] + i) * 0.5f, (ends[e][1] + j) * 0.5f, salt) * 2.2f;
                if (d > r) continue;
                if (mat_at(ends[e][0] + i, ends[e][1] + j + 1) == M_AIR)
                    put(ends[e][0] + i, ends[e][1] + j + 1, M_LEAF, 290, 0, 1);
            }
    }
}

static void worldgen(void) {
    size_t n = (size_t)NX * NY;
    cells = calloc(n * 4, sizeof *cells);
    surf0 = malloc(NX * sizeof *surf0);
    for (size_t i = 0; i < n; i++) cells[4 * i] = (uint16_t)((hash32((uint32_t)i * 2654435761u ^ opt_seed) & 0xff) << 8);

    /* the ground's profile */
    float Hv = vent_top - j_sea, Hm = fminf(Ufeat * 0.38f, (mt_x1 - mt_x0) * 1.25f);
    int ld = clampi((int)(Ufeat * 0.075f), 9, 34);
    for (int x = 0; x < NX; x++) {
        float h = j_sea + 5 + Ufeat * 0.035f * fbm1(x * 0.02f, 4, 11) + Ufeat * 0.02f * fbm1(x * 0.08f, 2, 12);
        /* rolling hills across the lowlands */
        float hill = fbm1(x * 0.0045f + 3.7f, 3, 13) + 0.25f;
        h += Ufeat * 0.10f * fmaxf(hill, 0) + Ufeat * 0.025f * fbm1(x * 0.02f, 2, 14);
        /* and a line of downs between the lake and the sea */
        {
            float xh = lake_x + 0.55f * (sea_end - lake_x), wh = fmaxf(NX * 0.1f, 30);
            float e2 = clampf(1 - fabsf(x - xh) / wh, 0, 1);
            e2 = e2 * e2 * (3 - 2 * e2);
            h += Ufeat * 0.12f * e2 * (0.6f + 0.6f * (1 - fabsf(fbm1(x * 0.03f, 3, 16))));
        }
        /* the range */
        float mc = (mt_x0 + mt_x1) * 0.5f, mh = (mt_x1 - mt_x0) * 0.5f;
        float e = clampf(1 - fabsf(x - mc) / (mh * 1.5f), 0, 1);
        e = e * e * (3 - 2 * e);
        float ridge = 1 - fabsf(fbm1(x * 0.022f, 4, 21));
        h += Hm * e * (0.35f + 0.75f * ridge * ridge);
        /* foothills rising to the range and the volcano */
        float dv = fabsf((float)(x - vent_x));
        h += Ufeat * 0.05f * clampf(1 - dv / (NX * 0.35f), 0, 1);
        /* the volcano: a cone with a crater */
        float cone = j_sea + Hv - dv * 0.78f + 2.5f * fbm1(x * 0.1f, 2, 31);
        if (dv < crater_w + 2) cone = j_sea + Hv - 7 * (1 - (dv / (crater_w + 2)) * (dv / (crater_w + 2))) + 2;
        if (cone > h) h = cone + 0 * h;
        /* the lake */
        float dl = (x - lake_x) / (float)lake_w;
        if (fabsf(dl) < 1.6f) h -= ld * expf(-dl * dl * 1.6f);
        /* the sea */
        float ds = sea_side < 0 ? x - sea_end : sea_end - x;     /* >0: inland */
        float sf = j_sea - Ufeat * 0.11f + 3 * fbm1(x * 0.05f, 2, 41);
        h = mixf(sf, h, smooth(-NX * 0.06f, 14, ds));
        surf0[x] = h;
    }
    /* the lake's level: the lower of its rims */
    float rimL = 0, rimR = 0;
    for (int x = lake_x; x >= 0 && x > lake_x - lake_w * 3; x--) if (surf0[x] > rimL) rimL = surf0[x];
    for (int x = lake_x; x < NX && x < lake_x + lake_w * 3; x++) if (surf0[x] > rimR) rimR = surf0[x];
    int lake_lv = (int)fminf(rimL, rimR) - 1;
    if (lake_lv > surf0[lake_x] + ld * 1.2f) lake_lv = (int)(surf0[lake_x] + ld * 0.9f);
    /* its outlet: the land falls from the lake's seaward shore to the sea, so
     * it spills over into a river rather than drowning the country */
    {
        int dir = sea_side;                        /* towards the sea */
        int x0 = lake_x + dir * (int)(lake_w * 1.1f);
        int n = abs(sea_end - x0);
        for (int k = 0, below = 0; k <= n; k++) {
            int x = x0 + dir * k;
            if (x < 0 || x >= NX) break;
            float f = (float)k / n;
            float prof = fmaxf(lake_lv + 0.5f - k * 0.3f, mixf(lake_lv + 0.5f, j_sea + 2, f));
            if (surf0[x] > prof) surf0[x] = prof, below = 0;
            else if (++below > 6) break;           /* the land falls away by itself from here */
        }
    }

    season_k = 0;
    for (int x = 0; x < NX; x++) {
        int h = (int)surf0[x];
        float slope = fabsf(surf0[clampi(x + 1, 0, NX - 1)] - surf0[clampi(x - 1, 0, NX - 1)]) * 0.5f;
        float dv = fabsf((float)(x - vent_x));
        int volc = dv < Hv * 1.15f && h > j_sea + 2 && h >= j_sea + Hv - dv * 0.78f - 3;
        int dirt = (int)(5 + 3 * vnoise1(x * 0.1f, 51) - slope * 2.2f);
        float alt = (h - j_sea) / Ufeat;
        if (alt > 0.30f) dirt = (int)(dirt * 0.3f);
        int beach = 0;
        float ds = sea_side < 0 ? x - sea_end : sea_end - x;
        if (ds < 12 || (abs(x - lake_x) < lake_w * 1.6f && h <= lake_lv + 2)) beach = 1;
        for (int y = 0; y < NY; y++) {
            float T = tenv(x, y);
            int m = M_AIR;
            if (y <= h) {
                m = M_STONE;
                if (y > h - dirt) m = beach ? M_SAND : M_DIRT;
                if (beach && y > h - 6) m = M_SAND;
                {   /* the volcano's own rock: under its cone, down into a bowl */
                    float cone = j_sea + Hv - dv * 0.78f + 2;
                    float q = dv / (Hv * 1.3f);
                    float base = j_sea - 4 - Hv * 0.55f * fmaxf(1 - q * q, 0) + 6 * vnoise1(x * 0.08f, 52);
                    if (y <= cone && y > base && dv < Hv * 1.3f && m != M_SAND) m = M_BASALT;
                    (void)volc;
                }
                if (m == M_DIRT && y == h && !beach && alt < 0.33f) m = M_GRASS;
                /* a coal seam, or a vein of obsidian in the old flows */
                if (m == M_STONE && fbm2(x * 0.02f, y * 0.06f, 3, 61) > 0.56f && y < h - 14) m = M_COAL;
                if (m == M_BASALT && vnoise2(x * 0.3f, y * 0.3f, 62) > 0.9f) m = M_OBSID;
            } else if ((ds < 0 || ds < 12) && y <= j_sea) {
                m = M_WATER;
            } else if (abs(x - lake_x) < lake_w * 3 && y <= lake_lv) {
                m = M_WATER;
            }
            put(x, y, m, T, 0, m == M_DIRT || m == M_GRASS ? 120 : 0);
        }
    }
    /* the lake only where it's connected to its middle */
    for (int dir = -1; dir <= 1; dir += 2)
        for (int x = lake_x + dir, gone = 0; x >= 0 && x < NX && abs(x - lake_x) < lake_w * 3; x += dir) {
            if (surf0[x] >= lake_lv) gone = 1;
            if (gone) for (int y = (int)surf0[x] + 1; y <= lake_lv; y++) if (mat_at(x, y) == M_WATER) put(x, y, M_AIR, tenv(x, y), 0, 0);
        }

    /* caves */
    for (int x = 0; x < NX; x++)
        for (int y = j_mag + 6; y < NY; y++) {
            float h = surf0[x];
            if (y > h - 16) break;
            float depth = h - y;
            float dv = fabsf((float)(x - vent_x));
            if (dv < crater_w + 10 && y > ch_y - ch_ry) continue;
            if (y > j_sea - 40 && dv < (vent_top - j_sea) * 1.3f) continue;
            /* winding galleries, and here and there a great hall */
            float n1 = fbm2(x * 0.008f, y * 0.015f, 3, 71);
            float n2 = fbm2(x * 0.0075f, y * 0.013f, 3, 72);
            float crust = smooth(16, 36, depth);
            float w = 0.055f * crust;
            int cave = fabsf(n1) < w || n2 > 0.47f - 0.08f * crust;
            if (!cave) continue;
            int m = M_AIR;
            float n3 = vnoise2(x * 0.03f, y * 0.03f, 73);
            if (n3 > 0.62f) m = M_WATER;
            else if (n3 < 0.12f && y < j_sea - 30) m = M_OIL;
            if (y < j_mag + 30 && vnoise2(x * 0.05f, y * 0.05f, 74) > 0.55f) m = M_LAVA;
            put(x, y, m, m == M_LAVA ? 1450 : tenv(x, y), 0, 0);
        }

    /* the magma floor, the chamber, the throat */
    for (int x = 0; x < NX; x++) {
        int top = j_mag + (int)(3 * vnoise1(x * 0.07f, 81));
        for (int y = 0; y <= top; y++) put(x, y, M_MAGMA, 1700, 0, 0);
    }
    for (int y = j_mag; y < vent_top + 2; y++)
        for (int x = vent_x - ch_rx - 2; x <= vent_x + ch_rx + 2; x++) {
            float ex = (x - ch_x) / (float)ch_rx, ey = (y - ch_y) / (float)ch_ry;
            float wob = 0.15f * fbm2(x * 0.1f, y * 0.1f, 2, 82);
            int in = ex * ex + ey * ey < 1 + wob;
            /* feeder pipes down to the floor */
            if (y < ch_y && fabsf(x - ch_x + 6 * sinf(y * 0.13f)) < 2.5f) in = 1;
            /* the throat, wandering a little */
            float vx = vent_x + 2.5f * fbm1(y * 0.05f, 2, 83) * smooth(vent_top - 10, vent_top - 30, y);
            int hw = 1 + (y > vent_top - 12);
            if (y > ch_y && fabsf(x - vx) <= hw) in = 1;
            if (!in) continue;
            int m = y <= vent_top - 0.25f * (vent_top - vent_bot) ? M_LAVA : M_AIR;
            put(x, y, m, m == M_LAVA ? 1500 : tenv(x, y), 0, 0);
        }

    /* snow on the heights */
    for (int x = 0; x < NX; x++) {
        int h = (int)surf0[x];
        float t = tenv(x, h + 1) - 3;
        if (t < 273 && mat_at(x, h) != M_WATER && mat_at(x, h + 1) == M_AIR) {
            int d = 1 + (int)((273 - t) / 3);
            for (int y = h + 1; y <= h + d && y < NY; y++) put(x, y, M_SNOW, fminf(t, 270), 0, 0);
        }
    }

    /* the forest: trees on the gentle grass, firs high up, grass blades */
    for (int x = 3; x < NX - 3;) {
        int h = (int)surf0[x];
        float slope = fabsf(surf0[x + 2] - surf0[x - 2]) * 0.25f;
        float alt = (h - j_sea) / Ufeat;
        float forest = vnoise1(x * 0.015f, 91) * 0.7f + 0.3f;
        int ok = mat_at(x, h) == M_GRASS && slope < 1.4f && mat_at(x, h + 1) == M_AIR;
        if (ok && h01(x, 0, 92) < forest) {
            int conifer = alt > 0.11f || (alt > 0.06f && h01(x, 4, 92) < 0.4f);
            int ht = (int)(Ufeat * (conifer ? 0.065f : 0.075f) * (0.55f + 0.8f * h01(x, 1, 93)));
            ht = clampi(ht, 6, 40);
            tree(x, h + 1, ht, conifer, 94 + x);
            x += (conifer ? 4 : 7) + (int)(h01(x, 2, 95) * (conifer ? 5 : 8));
            continue;
        }
        if (ok && h01(x, 5, 92) < 0.35f) {
            /* a bush */
            put(x, h + 1, M_WOOD, 290, 0, 3);
            for (int j = 1; j <= 3; j++)
                for (int i = -2; i <= 2; i++)
                    if (abs(i) + (j - 1) <= 2 + (h01(x + i, j, 97) < 0.4f) && mat_at(x + i, h + j) == M_AIR)
                        put(x + i, h + j, M_LEAF, 290, 0, 0);
            x += 3;
            continue;
        }
        x++;
    }
    for (int x = 0; x < NX; x++) {
        int h = (int)surf0[x];
        if (mat_at(x, h) != M_GRASS) continue;
        int b = (int)(h01(x, 3, 96) * 4.3f);
        for (int k = 1; k <= b; k++) if (mat_at(x, h + k) == M_AIR) put(x, h + k, M_PLANT, 290, b - k, 0);
    }

    /* a few clouds */
    for (int k = 0; k < 4 + NX / 200; k++) {
        float cx = h01(k, 0, 101) * NX, cy = mixf(cb0, cb1, 0.3f + 0.4f * h01(k, 1, 101));
        float rx = 10 + h01(k, 2, 101) * NX * 0.04f, ry = 3 + rx * 0.22f;
        for (int y = (int)(cy - ry * 1.4f); y <= cy + ry * 1.4f; y++)
            for (int x = (int)(cx - rx * 1.3f); x <= cx + rx * 1.3f; x++) {
                float dx = (x - cx) / rx, dy = (y - cy) / ry;
                if (dx * dx + dy * dy + 0.5f * fbm2(x * 0.15f, y * 0.15f, 3, 102) < 1 && mat_at(x, y) == M_AIR)
                    put(x, y, M_CLOUD, tenv(x, y), 0, 1 + (h01(x, y, 103) < 0.3f));
            }
    }
}

/* ======================================================================
 * shaders
 * ====================================================================== */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }
);

/* what every sim pass shares: the material tables, hashing, fetching */
#define SIM_COMMON GLSL( \
    uniform usampler2D S; \
    uniform ivec2 N; \
    uniform uint tick; \
    uniform uint seed; \
    const int AIR = 0, STONE = 1, BASALT = 2, OBSID = 3, DIRT = 4, GRASS = 5, SAND = 6, WETSAND = 7, SNOW = 8, ICE = 9, \
              WATER = 10, OIL = 11, LAVA = 12, STEAM = 13, SMOKE = 14, CLOUD = 15, FIRE = 16, WOOD = 17, LEAF = 18, EMBER = 19, \
              ASH = 20, SEED = 21, PLANT = 22, MAGMA = 23, SILT = 24, BOLT = 25, COAL = 26, LITTER = 27; \
    /* class: 0 static, 1 powder, 2 liquid, 3 gas, 4 air */ \
    const int CLS[32] = int[32](4,0,0,0,1,1,1,1,1,1, 2,2,2,3,3,3,3,0,0,0, 1,1,0,0,2,0,0,1,0,0, 0,0); \
    const int DEN[32] = int[32](50,255,255,255,170,170,160,175,120,90, 100,80,130,20,30,50,25,255,255,255, 110,90,255,255,115,255,255,90,255,255, 255,255); \
    /* falling (or rising) mobility, of 256 */ \
    const int MOB[32] = int[32](256,0,0,0,256,256,256,256,110,200, 256,256,80,200,150,40,220,0,0,0, 140,150,0,0,160,0,0,100,0,0, 0,0); \
    /* sliding off a pile, of 256: the angle of repose */ \
    const int SLD[32] = int[32](0,0,0,0,5,3,190,4,40,30, 256,256,60,160,120,30,140,0,0,0, 90,110,0,0,180,0,0,70,0,0, 0,0); \
    /* running sideways (liquids) or wandering (gases), of 256 */ \
    const int HMV[32] = int[32](0,0,0,0,0,0,0,0,0,0, 256,220,50,110,90,0,110,0,0,0, 0,0,0,0,110,0,0,0,0,0, 0,0); \
    /* thermal conductivity, of 1024 a tick */ \
    const int CND[32] = int[32](4,20,20,20,10,10,12,24,6,30, 40,16,3,12,6,4,48,6,10,48, 6,8,10,8,24,96,10,10,20,20, 20,20); \
    const uint TIP = 128u, DIRR = 64u; \
    uint hsh(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; } \
    uint rnd(ivec2 p, uint salt) { return hsh(uint(p.x) * 0x9E3779B1u ^ hsh(uint(p.y) * 0x85EBCA77u ^ hsh(tick * 0xC2B2AE3Du ^ hsh(salt + seed)))); } \
    int MAT(uvec4 c) { return int(c.x & 63u); } \
    uvec4 at(ivec2 q) { \
        if (q.x < 0 || q.y < 0 || q.x >= N.x || q.y >= N.y) return uvec4(1u, 4640u, 0u, 0u); \
        return texelFetch(S, q, 0); \
    } \
)

/* MOVE: one Margolus pass. Each fragment works out its whole 2x2 block
 * (every fragment of a block reaches the same answer) and keeps its cell. */
static const char *FS_MOVE = "#version 330 core\n" SIM_COMMON GLSL(
    uniform ivec2 off;
    uniform int wind;                  /* -256..256, + blows right */
    uniform int cb0, cb1;              /* the cloud band, rows */
    out uvec4 o;

    int cls(uvec4 c) { int m = MAT(c); return m == LAVA && c.w > 0u ? 3 : CLS[m]; }
    bool soft(uvec4 c) { int m = MAT(c); return m == LEAF || m == PLANT || m == LITTER || m == SEED; }
    int dens(uvec4 c, int j) {
        int m = MAT(c);
        if (m == LAVA && c.w > 0u) return 15;          /* fountaining: flung upwards */
        if (m == CLOUD) return j < cb0 ? 20 : j > cb1 ? 70 : 50;
        return DEN[m];
    }
    int mob(uvec4 c) { int m = MAT(c); return m == LAVA && c.w > 0u ? 256 : MOB[m]; }
    int slide(uvec4 c) {
        int m = MAT(c);
        if (m == DIRT) return 5 + int(c.w & 255u) / 4;     /* wet dirt slumps */
        if (m == LAVA && c.w > 0u) return 200;
        return SLD[m];
    }
    /* heavier above lighter: swap (a fall, or a rise) */
    bool vsw(uvec4 t, uvec4 b, int jt, uint r) {
        int ct = cls(t), cb = cls(b);
        if (ct == 0 || cb == 0) return false;
        if (t.x == b.x && t.w == b.w) return false;
        int dt = dens(t, jt), db = dens(b, jt - 1);
        if (dt <= db) return false;
        int p = min(mob(t), mob(b));
        if (ct == 1 && cb == 2) p = p / 3;
        if (ct == 2 && cb == 2) p = p / 2;
        return int(r & 255u) < p;
    }
    /* t (top) and b (bottom, diagonally): t slides down off its support,
     * or b rises past whatever is over it */
    bool dsw(uvec4 t, uvec4 b, uvec4 ut, uvec4 ob, int jt, uint r) {
        int ct = cls(t), cb = cls(b);
        if (ct == 0 || cb == 0) return false;
        if (MAT(t) == MAT(b)) return false;
        int dt = dens(t, jt), db = dens(b, jt - 1);
        if (dt <= db) return false;
        bool tf = (ct == 1 || ct == 2) && (cls(ut) == 0 || dens(ut, jt - 1) >= dt);
        bool br = cb == 3 && (cls(ob) == 0 || dens(ob, jt) <= db);
        if (tf) return int(r & 255u) < slide(t);
        if (br) return int(r & 255u) < slide(b);
        return false;
    }

    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        ivec2 bo = ((p - off) & ivec2(~1)) + off;
        int me = (p.x - bo.x) + 2 * (p.y - bo.y);
        uvec4 c[4];
        for (int k = 0; k < 4; k++) c[k] = at(bo + ivec2(k & 1, k >> 1));
        bool mv[4] = bool[4](false, false, false, false);
        uint r = rnd(bo, 7u);
        int jt = bo.y + 1;

        /* falls and rises, column by column */
        for (int col = 0; col < 2; col++) {
            uint rr = hsh(r + uint(col) * 31u);
            if (vsw(c[2 + col], c[col], jt, rr)) {
                uvec4 t = c[2 + col]; c[2 + col] = c[col]; c[col] = t;
                mv[2 + col] = true; mv[col] = true;
            }
        }
        /* slides: top-left with bottom-right, top-right with bottom-left */
        uint r2 = hsh(r ^ 0x55u);
        bool first = (r2 & 256u) != 0u;
        for (int k = 0; k < 2; k++) {
            int d = first ? k : 1 - k;
            int t = d == 0 ? 2 : 3, b = d == 0 ? 1 : 0;
            int ut = d == 0 ? 0 : 1, ob = d == 0 ? 3 : 2;
            if (mv[t] || mv[b]) continue;
            if (dsw(c[t], c[b], c[ut], c[ob], jt, hsh(r2 + uint(k) * 17u))) {
                uvec4 x = c[t]; c[t] = c[b]; c[b] = x;
                mv[t] = true; mv[b] = true;
            }
        }
        /* sideways: liquids run on their direction bit, gases wander and
         * drift with the wind, light powders blow about */
        for (int row = 0; row < 2; row++) {
            int L = 2 * row, R = L + 1;
            if (mv[L] || mv[R]) continue;
            uint rr = hsh(r + 0x99u + uint(row) * 7u);
            int r8 = int(rr & 255u), r8b = int((rr >> 8) & 255u);
            int j = bo.y + row;
            int cl = cls(c[L]), cr = cls(c[R]);
            int ml = MAT(c[L]), mr = MAT(c[R]);
            bool swp = false;
            if (cl == 2 && (c[L].x & DIRR) != 0u) {
                if ((cr != 0 && dens(c[R], j) < dens(c[L], j) || ml == WATER && soft(c[R])) && r8 < HMV[ml]) swp = true;
                else if ((cr == 0 || dens(c[R], j) >= dens(c[L], j)) && r8b < 128) c[L].x &= ~DIRR;
            } else if (cr == 2 && (c[R].x & DIRR) == 0u) {
                if ((cl != 0 && dens(c[L], j) < dens(c[R], j) || mr == WATER && soft(c[L])) && r8 < HMV[mr]) swp = true;
                else if ((cl == 0 || dens(c[L], j) >= dens(c[R], j)) && r8b < 128) c[R].x |= DIRR;
            } else if ((cl == 3 && cr == 4) || (cl == 4 && cr == 3)) {
                int g = cl == 3 ? L : R;
                int m = MAT(c[g]);
                int h = HMV[m];
                int pr = m == CLOUD ? 0 : h * (256 + wind) / 512;
                int pl = m == CLOUD ? 0 : h * (256 - wind) / 512;
                if (m == LAVA) {           /* flung lava flies out the way it was thrown */
                    bool right = (c[g].x & DIRR) != 0u;
                    pr = right ? 120 : 6; pl = right ? 6 : 120;
                }
                if (g == L ? r8 < pr : r8 < pl) swp = true;
            } else if ((cl == 1 && cr == 4) || (cl == 4 && cr == 1)) {
                int g = cl == 1 ? L : R;
                int m = MAT(c[g]);
                if (m == SNOW || m == ASH || m == SEED || m == LITTER) {
                    int pr = 12 + max(wind, 0) / 3, pl = 12 + max(-wind, 0) / 3;
                    if (g == L ? r8 < pr : r8 < pl) swp = true;
                }
            }
            /* a little randomness in the running water, so it doesn't lock */
            if (!swp && cl == 2 && r8b == 7) c[L].x ^= DIRR;
            if (swp) { uvec4 x = c[L]; c[L] = c[R]; c[R] = x; }
        }
        o = c[me];
    }
);
/* REACT: heat, phase changes, fire, growth, weather; a cell's new self from
 * its 3x3 neighbourhood */
static const char *FS_REACT = "#version 330 core\n" SIM_COMMON GLSL(
    uniform sampler2D surf;            /* the ground's height per column (rows), for the climate */
    uniform vec4 clim;                 /* sea-level temperature (K), lapse (K/row), sea row, magma row */
    uniform vec4 vent;                 /* x, bottom row, top row, crater half width */
    uniform vec4 chamber;              /* the magma chamber: centre, radii */
    uniform ivec4 erupt;               /* pump, fountain, melt (of 65536), smoke */
    uniform ivec2 magma;               /* the magma column's level in the throat (row), how fast it fills */
    uniform ivec4 water;               /* evaporation, rain, cloud dissipation (of 65536), storm spawn */
    uniform ivec2 storm;               /* x range the storm spawns clouds over */
    uniform ivec4 puff;                /* a cumulus being born: centre of its base, half width, height */
    uniform ivec4 puff2;               /* its moisture, on, salt */
    uniform ivec4 grow;                /* tree growth, leaf growth, leaf drop, seed drop (of 65536) */
    uniform ivec4 grow2;               /* sprouting, grass, magma flow, weathering */
    uniform ivec4 bolt;                /* x, top row, on, salt */
    uniform int cb0, cb1;
    uniform ivec2 seaedge;             /* the column where the open sea comes in, and its level */
    out uvec4 o;

    float tenv(ivec2 p) {
        float s = texelFetch(surf, ivec2(p.x, 0), 0).r;
        float y = float(p.y);
        float t = clim.x - clim.y * (max(y, clim.z) - clim.z);
        if (y < s) {
            float ts = clim.x - clim.y * (max(s, clim.z) - clim.z);
            float d = (s - y) / max(s - clim.w, 1.0);
            t = ts + (520.0 - ts) * smoothstep(0.0, 1.0, d) + 1000.0 * exp(-(y - clim.w) / 7.0);
        }
        float dx = (float(p.x) - vent.x) / (vent.w + 4.0);
        if (y < vent.z - 2.0) t += 620.0 * exp(-dx * dx) * smoothstep(vent.y - 30.0, vent.y, y);
        vec2 e = (vec2(p) - chamber.xy) / (chamber.zw + 6.0);
        t += 900.0 * exp(-2.0 * dot(e, e));
        return t;
    }
    /* the throat, opening into a funnel of a crater at the top */
    bool invent(ivec2 p) { return abs(float(p.x) - vent.x) <= 3.0 && float(p.y) > vent.y && float(p.y) <= vent.z + 16.0; }
    bool incrater(ivec2 p) {
        float w = 3.0 + max(float(p.y) - vent.z + 3.0, 0.0) * 0.8;
        return abs(float(p.x) - vent.x) <= w && float(p.y) > vent.z - 6.0 && float(p.y) <= vent.z + 24.0;
    }
    /* a cumulus: a few lumps on a flat base */
    bool inpuff(ivec2 p) {
        if (puff2.y == 0) return false;
        vec2 d = vec2(p - puff.xy);
        if (d.y < 0.0 || abs(d.x) > float(puff.z) * 1.3) return false;
        for (int i = 0; i < 5; i++) {
            float h = float(hsh(uint(puff2.z) + uint(i) * 77u) & 255u) / 255.0;
            float cx = (float(i) - 2.0) / 2.0 * float(puff.z) * 0.8 + (h - 0.5) * float(puff.z) * 0.3;
            float r = float(puff.w) * (0.45 + 0.55 * h) * (1.0 - abs(float(i) - 2.0) * 0.18);
            vec2 e = d - vec2(cx, r * 0.35);
            float n = float(hsh(uint(p.x) * 7919u ^ uint(p.y) * 104729u ^ uint(puff2.z)) & 255u) / 255.0;
            if (dot(e, e) < r * r * (0.85 + 0.3 * n)) return true;
        }
        return false;
    }
    bool inchamber(ivec2 p) { vec2 e = (vec2(p) - chamber.xy) / chamber.zw; return dot(e, e) < 0.8; }
    int boltx(int y) {
        float f = float(y) * 0.13;
        float i = floor(f), u = fract(f);
        float a = float(hsh(uint(i) + uint(bolt.w)) & 255u) / 255.0, b = float(hsh(uint(i) + 1u + uint(bolt.w)) & 255u) / 255.0;
        float g = float(hsh(uint(y) * 7u + uint(bolt.w)) & 3u);
        return bolt.x + int(round((mix(a, b, u) - 0.5) * 18.0 + g - 1.5));
    }
    bool onbolt(ivec2 p) {
        if (p.y > bolt.y) return false;
        int a = boltx(p.y), b = boltx(p.y + 1);
        return p.x >= min(a, b) && p.x <= max(a, b);
    }
    bool passable(uvec4 c) { return CLS[MAT(c)] >= 3; }
    /* is the bolt's path clear from its top down to (not including) row y? */
    bool boltclear(int y) {
        for (int j = bolt.y; j > y; j--) {
            int a = boltx(j), b = boltx(j + 1);
            for (int x = min(a, b); x <= max(a, b); x++) if (!passable(at(ivec2(x, j)))) return false;
        }
        return true;
    }
    /* the pull rules' shared decisions: computed by both cells of a pair */
    int rainp(uvec4 cl) {
        if (cl.z < 5u || cl.w < 2u) return 0;                /* fair-weather cloud: no rain */
        return water.y * int(min(cl.w, 4u));
    }
    bool rains(ivec2 q, uvec4 cl) { return int(rnd(q, 11u) & 65535u) < rainp(cl); }
    bool grows(ivec2 q) { return int(rnd(q, 12u) & 65535u) < grow.x; }
    int IGN(int m) {
        return m == WOOD ? 560 : m == LEAF ? 480 : m == GRASS ? 500 : m == PLANT ? 470 : m == OIL ? 480 :
               m == COAL ? 650 : m == SEED ? 480 : m == LITTER ? 450 : 0;
    }
    int FUEL(int m) {
        return m == WOOD ? 900 : m == LEAF ? 90 : m == GRASS ? 60 : m == PLANT ? 45 : m == COAL ? 4000 :
               m == SEED ? 40 : m == LITTER ? 50 : 30;
    }
    int CATCH(int m) {                 /* catching from a flame alongside, of 65536 */
        return m == LEAF ? 1400 : m == PLANT ? 1600 : m == GRASS ? 700 : m == WOOD ? 160 : m == COAL ? 30 :
               m == SEED ? 1500 : m == LITTER ? 2200 : m == OIL ? 3000 : 0;
    }

    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        uvec4 c = at(p);
        int m = MAT(c);
        uvec4 nb[8];
        nb[0] = at(p + ivec2(-1, 0)); nb[1] = at(p + ivec2(1, 0));
        nb[2] = at(p + ivec2(0, -1)); nb[3] = at(p + ivec2(0, 1));
        nb[4] = at(p + ivec2(-1, -1)); nb[5] = at(p + ivec2(1, -1));
        nb[6] = at(p + ivec2(-1, 1)); nb[7] = at(p + ivec2(1, 1));
        uvec4 nL = nb[0], nR = nb[1], nD = nb[2], nU = nb[3];
        int mD = MAT(nD), mU = MAT(nU);

        /* heat: diffusion between the four neighbours */
        int T = int(c.y);
        int k = CND[m], acc = 0;
        for (int i = 0; i < 4; i++) acc += min(k, CND[MAT(nb[i])]) * (int(nb[i].y) - T);
        if (!(m == LAVA && c.w > 0u)) T += (acc + 512) >> 10;     /* (flung lava keeps its heat) */
        float te = tenv(p);
        int TE = int(te * 16.0);

        int nWater = 0, nLava = 0, nFlame = 0, nAir = 0, nIce = 0, nCloud = 0, nMagma = 0, nTree = 0, nGrass = 0, nPlant = 0, nSteam = 0;
        for (int i = 0; i < 8; i++) {
            int q = MAT(nb[i]);
            nWater += int(q == WATER); nLava += int(q == LAVA); nFlame += int(q == FIRE || q == EMBER || q == BOLT);
            nAir += int(q == AIR); nIce += int(q == ICE || q == SNOW); nCloud += int(q == CLOUD);
            nMagma += int(q == MAGMA); nTree += int(q == WOOD || q == LEAF); nGrass += int(q == GRASS);
            nPlant += int(q == GRASS || q == PLANT || q == LEAF); nSteam += int(q == STEAM);
        }
        uint r0 = rnd(p, 1u), r1 = rnd(p, 2u);
        int R8 = int(r0 & 255u), R8b = int((r0 >> 8) & 255u), R16 = int(r0 >> 16), Q16 = int(r1 & 65535u);
        uint x = c.x;
        int life = int(c.z), aux = int(c.w);
        int nm = m;                    /* the new material */

        /* water passes through trees (they'd dam it otherwise): a drop with a
         * trunk, leaf or blade next to it hops over to the air beyond; one
         * direction a tick (down, right, left), so no air gets two drops */
        ivec2 dd = (tick & 1u) == 0u ? ivec2(1, 0) : ivec2(-1, 0);
        bool seep = false;
        /* and down through the canopy: a leaf (or a blade) holds a drop, hands
         * it to the leaf below or lets it fall out underneath */
        bool wetU = (mU == LEAF || mU == PLANT) && (nU.w & 512u) != 0u;
        bool soft = m == LEAF || m == PLANT, wet = soft && (c.w & 512u) != 0u;
        bool softD = mD == LEAF || mD == PLANT, wetD = softD && (nD.w & 512u) != 0u;
        bool dripme = (rnd(p, 16u) & 3u) != 0u, dripU = (rnd(p + ivec2(0, 1), 16u) & 3u) != 0u;
        if (m == WATER && softD && !wetD && dripme) { seep = true; nm = AIR; T = TE; life = 0; aux = 0; }
        else if (soft && !wet && mU == WATER && dripU) { seep = true; aux |= 512; }
        else if (wet && (mD == AIR || softD && !wetD) && dripme) { seep = true; aux &= ~512; }
        else if (m == AIR && wetU && dripU) { seep = true; nm = WATER; T = int(nU.y); life = 0; aux = 0; }
        else if (softD && !wetD && wetU && dripU && soft && !wet) {}
        if (!seep && soft && !wet && wetU && dripU) { seep = true; aux |= 512; }
        if (m == AIR || m == WATER) {
            /* walk away from the cell (back for air, on for water) through up
             * to three cells of tree to the water (air) on the far side */
            ivec2 w = m == AIR ? -dd : dd;
            for (int k = 1; k <= 4 && !seep; k++) {
                uvec4 o2 = at(p + k * w);
                int mm = MAT(o2);
                if (k > 1 && (m == AIR ? mm == WATER : mm == AIR)) {
                    ivec2 src = m == AIR ? p + k * w : p;
                    if ((rnd(src, 15u) & 1u) != 0u) {
                        seep = true;
                        if (m == AIR) { nm = WATER; T = int(o2.y); x = (x & 0xff00u) | (o2.x & DIRR); life = 0; aux = 0; }
                        else { nm = AIR; T = int(nb[0].y + nb[1].y) / 2; life = 0; aux = 0; }
                    }
                    break;
                }
                if (mm != WOOD && mm != LEAF && mm != PLANT && mm != LITTER && mm != SEED) break;
            }
        }
        /* the open sea beyond the edge keeps the sea at its level */
        if (seep) {}
        else
        if (p.x == seaedge.x && (m == AIR || m == WATER || CLS[m] == 3) && R8 < 64) {
            if (p.y <= seaedge.y && m != WATER && CLS[mD] != 3 && mD != AIR) { nm = WATER; T = TE; aux = 0; life = 0; }
            else if (p.y > seaedge.y && m == WATER) nm = AIR;
        }
        /* magma welling up into the chamber and the throat, through whatever
         * air or fumes are in the way */
        else if (m != LAVA && m != MAGMA && m != BOLT &&
                 ((invent(p) || incrater(p) && CLS[m] != 0) && p.y <= magma.x || CLS[m] >= 3 && (inchamber(p) || nMagma > 0)) && R16 < magma.y) {
            nm = LAVA; T = 1550 * 16; aux = 0; life = 0;
        }
        /* lightning: the path from the cloud down to the first thing it hits */
        else if (bolt.z > 0 && onbolt(p) && boltclear(p.y)) {
            if (passable(c)) { nm = BOLT; life = 8; T = 3000 * 16; }
            else T = max(T, 2600 * 16);
        }
        else if (m == AIR) {
            T += (TE - T) >> 5;
            int ne = int(mD == EMBER) * 3 + int(MAT(nb[4]) == EMBER) + int(MAT(nb[5]) == EMBER) +
                     int(MAT(nL) == EMBER) + int(MAT(nR) == EMBER);
            bool vtop = abs(float(p.x) - vent.x) < vent.w + 2.0 && float(p.y) > vent.z - 4.0 && float(p.y) < vent.z + 8.0;
            if (mD == WOOD && (nD.x & TIP) != 0u && nD.z > 0u && grows(p - ivec2(0, 1))) {
                /* a tree grows a cell taller */
                nm = WOOD; x |= TIP; life = int(nD.z) - 1;
                bool fir = (nD.w & 256u) != 0u;
                aux = fir ? 256 + 1 + min(life, 9) / 3 : 0;
                T = TE;
            } else if (ne > 0 && R8 < ne * 12) {
                nm = FIRE; life = 12 + R8b % 30; T = 1250 * 16;
            } else if (mU == CLOUD && rains(p + ivec2(0, 1), nU)) {
                nm = int(nU.y) < 272 * 16 ? SNOW : WATER; T = int(nU.y); life = 0; aux = 0;
            } else if (MAT(nb[4]) == WOOD && (nb[4].x & TIP) != 0u && nb[4].z >= 4u && (nb[4].w & 256u) == 0u &&
                       int(rnd(p, 13u) & 65535u) < grow.x / 5) {
                nm = WOOD; x |= TIP; life = min(int(nb[4].z) / 2, 6); aux = 0; T = TE;
            } else if (MAT(nb[5]) == WOOD && (nb[5].x & TIP) != 0u && nb[5].z >= 4u && (nb[5].w & 256u) == 0u &&
                       int(rnd(p, 14u) & 65535u) < grow.x / 5) {
                nm = WOOD; x |= TIP; life = min(int(nb[5].z) / 2, 6); aux = 0; T = TE;
            } else if (vtop && R16 < erupt.w) {
                nm = SMOKE; life = 0; T = 600 * 16;
            } else {
                /* leaves, from a crown, a branch end or the leaf before */
                int best = 0, sp = 0;
                int dc = 1 + int((r1 >> 20) & 1u);              /* diagonals cost more, sometimes: round crowns */
                for (int i = 0; i < 8; i++) {
                    int q = MAT(nb[i]);
                    int e = int(nb[i].w & 255u) - (i < 4 ? 0 : dc - 1);
                    if ((q == WOOD || q == LEAF) && e > best) { best = e; sp = int(nb[i].w & 256u); }
                }
                int lp = sp != 0 ? 900 : grow.y;
                if (best > 0 && Q16 < lp && TE > 266 * 16) {
                    nm = LEAF; aux = sp | (best - 1); life = 0; T = TE;
                } else if (mD == GRASS && Q16 < grow2.y) {
                    nm = PLANT; life = R8b % 3; T = TE;
                } else if (mD == PLANT && nD.z > 0u && Q16 < grow2.y * 4) {
                    nm = PLANT; life = int(nD.z) - 1; T = TE;
                } else if (mU == LEAF && (nU.w & 256u) == 0u && Q16 < grow.w) {
                    nm = SEED; life = 0; aux = 0;
                } else if (p.y > cb0 - 24 && p.y < cb0 && int(rnd(p, 17u) & 0xfffffu) < grow2.z) {
                    nm = SEED; life = 0; aux = 0;                /* blown in from elsewhere */
                } else if (inpuff(p)) {
                    nm = CLOUD; aux = puff2.x; life = 0; T = min(T, TE);
                } else if (p.y >= cb0 && p.y <= cb1 && p.x >= storm.x && p.x < storm.y && Q16 < (nCloud >= 4 ? water.w * 6 : 0)) {
                    /* storm cloud: a few seeds, then billowing out from them */
                    nm = CLOUD; aux = 1 + R8b % 3; life = 0;
                }
            }
        }
        else if (m == WATER) {
            if (nLava > 0 && R8 < 90) { nm = STEAM; T = max(T, 420 * 16); life = 0; }
            else if (T > 373 * 16 && R8 < 64) { nm = STEAM; life = 0; }
            else if (T < 272 * 16) {
                if (mD == AIR) nm = SNOW;
                else if ((nIce > 0 || mU == AIR) && R8 < 6) nm = ICE;
            } else if (mU == AIR) {
                int f = clamp((T - 273 * 16) / 24, 4, 80);
                int ev = water.x * f / 16;
                if (mD != WATER && nAir >= 4) ev *= 40;          /* a lone drop, a film: soon gone */
                if (Q16 < ev) { nm = STEAM; life = 0; }
            }
        }
        else if (m == ICE) {
            T += (TE - T) >> 8;
            if (T > 274 * 16 && R8 < (T > 282 * 16 ? 48 : 6)) nm = WATER;
            if (nLava + nFlame > 0) nm = WATER;
            else if (CLS[mU] >= 3 && Q16 < 4 + (T > 268 * 16 ? 8 : 0)) { nm = STEAM; life = 0; }   /* ablation in the open */
        }
        else if (m == SNOW) {
            T += (TE - T) >> 6;
            if (mD == AIR && T > 275 * 16 && R8 < 24) nm = WATER;
            else if (T > 275 * 16 && R8 < 2 + (T - 275 * 16) / 32) nm = WATER;
            else if (nLava + nFlame > 0 && R8 < 64) nm = WATER;
            else if (mU == SNOW && (mD == SNOW || mD == ICE) && Q16 < 3) nm = ICE;
            else if (CLS[mU] >= 3 && mD != AIR && Q16 < 3 + (T > 268 * 16 ? 6 : 0)) { nm = STEAM; life = 0; }   /* sublimation */
        }
        else if (m == STEAM) {
            T += (TE - T) >> 6;
            life = min(life + 1, 65535);
            if (p.y >= N.y - 2) nm = AIR;
            else if (p.y >= cb0 - 4 && p.y < (cb0 + cb1) / 2 && T < 292 * 16 && R8 < 24 && (nCloud >= 1 || nSteam >= 3 || R8 == 3)) { nm = CLOUD; aux = 2; life = 0; }
            else if (nCloud >= 2 && T < 330 * 16 && R8 < 64) { nm = CLOUD; aux = 1; life = 0; }
            else if (CLS[mU] == 0 && T < 340 * 16 && R8 < 2) nm = WATER;
            else if (life > 5000 && R8 < 4) nm = AIR;
        }
        else if (m == SMOKE) {
            T += (TE - T) >> 5;
            life += 1;
            if (p.y >= N.y - 2 || life > 420 + R8 * 3) nm = AIR;
        }
        else if (m == CLOUD) {
            T += (TE - T) >> 6;
            life = nCloud;                                    /* thickness, for rains() */
            if (mD == AIR && rains(p, c)) { aux -= 1; if (aux <= 0) nm = AIR; }
            if ((p.x < 2 || p.x >= N.x - 2) && R8 < 16) nm = AIR;
            else if (nCloud <= 1 && Q16 < 600) { nm = STEAM; life = 0; }
            else if (Q16 < (nCloud <= 3 ? water.z * 10 + 20 : water.z)) nm = AIR;
            else if (nLava + nFlame > 0) nm = STEAM;
        }
        else if (m == FIRE) {
            life -= 1 + int(nTree + nGrass == 0);
            if (nWater > 0 && R8 < 128) { nm = SMOKE; life = 0; }
            else if (life <= 0) { nm = R8b < 90 ? SMOKE : AIR; life = 0; }
        }
        else if (m == EMBER) {
            T = max(T, 1150 * 16);
            life -= 1;
            int orig = aux & 63;
            if (nWater + nIce > 0 && R8 < 40) { nm = orig; aux = 0; T = 340 * 16; life = 0; x &= ~TIP; }
            else if (life <= 0) {
                if (orig == WOOD) nm = R8 < 90 ? ASH : AIR;
                else if (orig == GRASS) nm = DIRT;
                else if (orig == COAL) nm = ASH;
                else nm = R8 < 30 ? ASH : AIR;
                aux = 0; life = 0;
            }
        }
        else if (m == LAVA) {
            if (aux > 0) aux -= 1;
            if (invent(p) && R16 < erupt.x) aux = max(aux, 3 + R8b % 4);
            bool launch = invent(p) && float(p.y) > vent.z - 30.0 && Q16 < erupt.y * (mU != LAVA && CLS[mU] != 0 ? 8 : 1);
            if (launch) { aux = 25 + R8b % 60; T = max(T, 1500 * 16); x = (r1 & 0x10000u) != 0u ? x | DIRR : x & ~DIRR; }
            if (TE > T) T += (TE - T) >> 4;
            if (invent(p) && p.y <= magma.x) T = max(T, 1200 * 16);      /* the magma column keeps hot */
            /* the column sinks back when the pressure's off */
            if (invent(p) && p.y > magma.x + 2 && aux == 0 && mU != LAVA && R16 < 1500) { nm = AIR; T = TE; }
            int exposed = 0;
            for (int i = 0; i < 4; i++) exposed += int(CLS[MAT(nb[i])] >= 3);
            T -= exposed * 3;
            if (nWater > 0 && R8 < 40) { nm = OBSID; aux = 0; }
            else if (T < 1000 * 16 && R8 < 24) {
                /* it sets: as rock where it lies, as cinders in the air */
                if (CLS[mD] != 0 || aux > 0 || incrater(p)) { nm = ASH; aux = 1; life = 0; }
                else { nm = BASALT; aux = 0; }
            }
        }
        else if (m == MAGMA) {
            T = 1700 * 16;
        }
        else if (m == BOLT) {
            life -= 1; T = 3000 * 16;
            if (life <= 0) { nm = AIR; T = 1500 * 16; }
        }
        else if (m == STONE || m == BASALT || m == OBSID) {
            T += (TE - T) >> 7;
            int open = 0;
            for (int i = 0; i < 8; i++) open += int(CLS[MAT(nb[i])] >= 3);
            if (open >= 6 && R8 < 16) { nm = m == STONE ? SAND : ASH; aux = m == STONE ? 0 : 1; life = 0; }
            else
            if (T > 1600 * 16 && R8 < 8) { nm = LAVA; aux = 0; }
            else if (invent(p) && R16 < erupt.z) { nm = LAVA; T = max(T, 1500 * 16); aux = 0; }
            else if (m != STONE && nAir > 0 && T < 320 * 16 && nPlant + nWater > 0 && Q16 < grow2.w) nm = DIRT;
            else if (m == STONE && mU == WATER && nWater >= 3 && Q16 < grow2.w / 4) nm = SAND;
        }
        else if (m == DIRT) {
            T += (TE - T) >> 7;
            aux = nWater > 0 ? 255 : max(aux - int(R8 < 6), 0);
            if (T > 1500 * 16 && R8 < 8) { nm = LAVA; aux = 0; }
            else if ((mU == AIR || mU == PLANT || mU == SEED) && aux > 20 && T > 276 * 16 && T < 315 * 16 &&
                     Q16 < grow2.y / 4 + nGrass * grow2.y * 2) nm = GRASS;
            else if (mU == WATER && (MAT(nL) == WATER || MAT(nR) == WATER) && Q16 < 6) nm = SILT;
        }
        else if (m == SAND) {
            T += (TE - T) >> 7;
            if (nWater > 0 && R8 < 24) nm = WETSAND;
            else if (mU == WATER && MAT(nL) == WATER && MAT(nR) == WATER && Q16 < 10) nm = SILT;
            else if (T > 1700 * 16) nm = LAVA;
        }
        else if (m == WETSAND) {
            T += (TE - T) >> 7;
            if (nWater == 0 && (Q16 < 160 || T > 330 * 16)) nm = SAND;
        }
        else if (m == SILT) {
            if (nWater == 0 && R8 < 24) nm = SAND;
            else if (CLS[mD] <= 1 && mD != AIR && R8 < 4) nm = SAND;
        }
        else if (m == ASH) {
            life = min(life + 1, 65535);
            if ((mD == DIRT || mD == GRASS) && life > 1500 && R8 < 2) nm = AIR;
            if (mD == WATER && life > 600 && R8 < 2) nm = AIR;
            if (aux != 0 && CLS[mU] >= 3 && CLS[mD] <= 1 && life > 3000 && Q16 < 3) nm = AIR;   /* cinders weather away */
        }
        else if (m == SEED) {
            life = min(life + 1, 65535);
            bool daylight = float(p.y) > texelFetch(surf, ivec2(p.x, 0), 0).r - 4.0;     /* not down a cave */
            if ((mD == DIRT || mD == GRASS) && mU == AIR && nTree == 0 && daylight && TE > 277 * 16 && TE < 306 * 16 && Q16 < grow2.x) {
                nm = WOOD; x |= TIP; life = 7 + R8 % 24;
                bool fir = TE < 282 * 16;
                aux = fir ? 256 + 1 + min(life, 9) / 3 : 0;
            } else if (life > 3000) nm = AIR;
        }
        else if (m == LITTER) {
            life = min(life + 1, 65535);
            if (CLS[mD] != 4 && life > 900 && R8 < 2) nm = AIR;
        }
        else if (m == PLANT) {
            if (TE < 268 * 16 && Q16 < 60) nm = AIR;
            else if (mD != GRASS && mD != PLANT && mD != DIRT && R8 < 8) nm = AIR;
        }
        else if (m == GRASS) {
            T += (TE - T) >> 7;
            int cu = CLS[mU];
            if (mU != PLANT && mU != LEAF && mU != WOOD && mU != SNOW && mU != SEED && mU != LITTER && mU != ASH && cu != 3 && cu != 4 && R8 < 16)
                nm = DIRT;
        }
        else if (m == WOOD) {
            if ((x & TIP) != 0u) {
                if (life > 0 && (mU == AIR || mU == LEAF) && grows(p)) x &= ~TIP;
                else if (life == 0) {
                    x &= ~TIP;
                    bool fir = (aux & 256) != 0;
                    aux = fir ? 256 + 2 : 4 + R8 % 3;
                }
            }
        }
        else if (m == LEAF) {
            if (mD == WOOD && (nD.x & TIP) != 0u && nD.z > 0u && grows(p - ivec2(0, 1))) {
                nm = WOOD; x |= TIP; life = int(nD.z) - 1;
                bool fir = (nD.w & 256u) != 0u;
                aux = fir ? 256 + 1 + min(life, 9) / 3 : 0;
            } else if ((aux & 256) == 0 && Q16 < grow.z) { nm = LITTER; life = 0; aux = 0; }
            else if ((aux & 512) != 0 && Q16 < 20) aux &= ~512;          /* a held drop dries */
        }

        /* fire: anything that burns catches when hot enough or alongside a flame */
        if (nm == m && (m == WOOD || m == LEAF || m == PLANT || m == GRASS || m == COAL || m == SEED || m == LITTER || m == OIL)) {
            bool lit = T > IGN(m) * 16 && R8 < 64;
            if (!lit && nFlame + nLava > 0 && Q16 < CATCH(m) * (nFlame + 2 * nLava)) lit = true;
            if (lit) {
                if (m == OIL) { nm = FIRE; life = 60 + R8b % 60; }
                else { nm = EMBER; aux = m; life = FUEL(m) * (128 + R8b) / 256; x &= ~TIP; }
                T = 1150 * 16;
            }
        }
        if (nm == FIRE) T = max(T, 1250 * 16);

        x = (x & 0xffc0u) | uint(nm);
        o = uvec4(x, uint(clamp(T, 0, 65535)), uint(clamp(life, 0, 65535)), uint(clamp(aux, 0, 65535)));
    }
);
/* SHIFT: the slow geology. Columns marked +1 are lifted a cell (new rock
 * welling up from below), -1 let down a cell (the bottom of the crust lost
 * to the magma), above the floor row. */
static const char *FS_SHIFT = "#version 330 core\n" SIM_COMMON GLSL(
    uniform isampler2D shift;
    uniform sampler2D surf;
    uniform int base0;
    out uvec4 o;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        int s = texelFetch(shift, ivec2(p.x, 0), 0).r;
        /* only the upper crust moves: what's deep (the chamber, the floor) stays */
        int base = max(base0, int(texelFetch(surf, ivec2(p.x, 0), 0).r) - 14);
        uvec4 c = at(p);
        if (p.y >= base && s > 0) c = p.y == base ? uvec4((c.x & 0xff00u) | uint(STONE), at(p).y, 0u, 0u) : at(p - ivec2(0, 1));
        else if (p.y >= base && s < 0) c = p.y == N.y - 1 ? uvec4(c.x & 0xff00u, c.y, 0u, 0u) : at(p + ivec2(0, 1));
        o = c;
    }
);

/* WIND: the cloud deck moves bodily with the wind, a cell at a time when
 * the wind has carried it that far, so clouds keep their shapes */
static const char *FS_WIND = "#version 330 core\n" SIM_COMMON GLSL(
    uniform int dir, y0;
    out uvec4 o;
    bool free(ivec2 q) { if (q.x < 0 || q.x >= N.x) return true; int m = MAT(at(q)); return m == AIR || m == CLOUD; }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        uvec4 c = at(p);
        o = c;
        if (p.y < y0) return;
        int m = MAT(c);
        ivec2 s = p - ivec2(dir, 0);
        uvec4 cs = at(s);
        if ((m == AIR || m == CLOUD) && s.x >= 0 && s.x < N.x && MAT(cs) == CLOUD) o = uvec4(cs.x, c.y, cs.z, cs.w);
        else if (m == CLOUD && free(p + ivec2(dir, 0))) o = uvec4(c.x & 0xff00u, c.y, 0u, 0u);
    }
);

/* STATS: per column: (land cells, water in all its forms, cloud + steam,
 * the ground's top row) in row 0; (lava, wood, fire, the top of anything
 * standing, trees included) in row 1 */
static const char *FS_STATS = "#version 330 core\n" SIM_COMMON GLSL(
    out uvec4 o;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        uvec4 a = uvec4(0u);
        int run = 0;
        for (int y = 0; y < N.y; y++) {
            int m = MAT(texelFetch(S, ivec2(p.x, y), 0));
            if (p.y == 0) {
                bool land = m == STONE || m == BASALT || m == OBSID || m == DIRT || m == GRASS || m == SAND ||
                            m == WETSAND || m == COAL || m == SILT || m == LAVA || m == ASH || m == MAGMA;
                run = land ? run + 1 : 0;
                if (land) a.x++;
                if (run >= 3) a.w = uint(y);
                if (m == WATER || m == ICE || m == SNOW) a.y++;
                if (m == CLOUD || m == STEAM) a.z++;
            } else {
                if (m == LAVA) a.x++;
                if (m == WOOD) a.y++;
                if (m == FIRE || m == EMBER) a.z++;
                if (CLS[m] == 0 && m != MAGMA) a.w = uint(y);       /* the top of anything standing */
            }
        }
        o = a;
    }
);

/* value noise and the rock's strata, for the shade and cell passes */
#define NOISE_COMMON GLSL( \
    uint nh(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; } \
    float h1(ivec2 p, uint s) { return float(nh(uint(p.x) * 0x9E3779B1u ^ nh(uint(p.y) + s * 0x85EBCA77u)) & 0xffffu) / 65535.0; } \
    float vn(vec2 x, uint s) { \
        vec2 i = floor(x), f = fract(x); \
        f = f * f * (3.0 - 2.0 * f); \
        ivec2 q = ivec2(i); \
        return mix(mix(h1(q, s), h1(q + ivec2(1, 0), s), f.x), mix(h1(q + ivec2(0, 1), s), h1(q + ivec2(1, 1), s), f.x), f.y); \
    } \
    /* bedded rock, gently folded: granite, slate, sandstone, shale, limestone, red beds */ \
    vec3 strata(vec2 p, uint salt) { \
        float w = vn(vec2(p.x * 0.011, 3.1), salt + 21u) * 7.0 + vn(vec2(p.x * 0.045, 1.7), salt + 22u) * 1.6; \
        float L = p.y * 0.07 + w + p.x * 0.002; \
        float b = floor(L); \
        float h = h1(ivec2(int(b), 0), salt + 23u); \
        vec3 c = h < 0.2 ? vec3(0.44, 0.43, 0.45) : h < 0.38 ? vec3(0.35, 0.38, 0.45) : h < 0.56 ? vec3(0.62, 0.50, 0.36) : \
                 h < 0.72 ? vec3(0.29, 0.27, 0.30) : h < 0.88 ? vec3(0.58, 0.56, 0.50) : vec3(0.56, 0.37, 0.29); \
        float f = fract(L); \
        c *= 0.93 + 0.1 * smoothstep(0.0, 0.2, f) - 0.1 * smoothstep(0.8, 1.0, f); \
        return c; \
    } \
)

/* SHADE: each cell's albedo and cover, its light, and how much skylight it
 * lets through */
static const char *FS_SHADE = "#version 330 core\n" SIM_COMMON NOISE_COMMON GLSL(
    uniform float time;
    uniform vec4 season;               /* autumn, winter, spring greening, - */
    layout(location = 0) out vec4 o_alb;
    layout(location = 1) out vec4 o_emi;
    layout(location = 2) out uint o_mat;   /* material | 64 falling | 128 open above */
    layout(location = 3) out vec4 o_gas;   /* smoke, steam: drawn soft, from the blurred densities */

    vec3 heat(float T) {               /* glow of hot matter, K */
        float k = max(T - 720.0, 0.0) / 800.0;
        return vec3(1.0, 0.28, 0.06) * k * k * 2.2 + vec3(0.4, 0.45, 0.2) * k * k * k * 1.5;
    }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        uvec4 c = at(p);
        int m = MAT(c);
        float v = float((c.x >> 8) & 255u) / 255.0;    /* the grain's own shade */
        float T = float(c.y) / 16.0;
        vec3 a = vec3(0.0), e = vec3(0.0);
        vec4 gs = vec4(0.0);
        float al = 1.0, tr = 0.42;
        int mU = MAT(at(p + ivec2(0, 1))), mD = MAT(at(p - ivec2(0, 1)));
        vec2 wp = vec2(p);
        if (m == AIR) { al = 0.0; tr = 1.0; }
        else if (m == STONE) {
            a = strata(wp, seed % 97u) * (0.88 + 0.12 * vn(wp * 0.15, 5u)) * (0.9 + 0.2 * v);
        }
        else if (m == BASALT) {
            /* old flows, one over another */
            float L = wp.y * 0.16 + vn(vec2(wp.x * 0.03, 0.5), 6u) * 4.0 - abs(wp.x - float(N.x) * 0.5) * 0.0;
            float f = fract(L), b = h1(ivec2(int(floor(L)), 3), 7u);
            a = mix(vec3(0.17, 0.16, 0.17), vec3(0.27, 0.23, 0.22), b) * (0.85 + 0.25 * v);
            a *= 0.9 + 0.12 * smoothstep(0.0, 0.25, f) - 0.15 * smoothstep(0.85, 1.0, f);
        }
        else if (m == OBSID) { a = vec3(0.13, 0.10, 0.18) * (0.8 + 0.4 * v); if (v > 0.85) a += vec3(0.30, 0.26, 0.42) * (v - 0.85) * 6.0; }
        else if (m == COAL) { a = vec3(0.14, 0.13, 0.14) * (0.8 + 0.4 * v); if (v > 0.9) a += vec3(0.22, 0.22, 0.25); }
        else if (m == DIRT) {
            a = mix(vec3(0.42, 0.28, 0.17), vec3(0.33, 0.21, 0.13), vn(wp * 0.15, 7u));
            a *= 0.85 + 0.25 * v;
            a *= 1.0 - float(c.w & 255u) / 255.0 * 0.25;
        }
        else if (m == GRASS) {
            vec3 g = mix(vec3(0.30, 0.55, 0.20), vec3(0.42, 0.62, 0.22), v);
            g = mix(g, vec3(0.62, 0.55, 0.25), season.x * 0.6);
            g = mix(g, vec3(0.45, 0.42, 0.30), season.y * 0.6);
            a = g * (0.9 + 0.2 * v);
        }
        else if (m == SAND) { a = mix(vec3(0.90, 0.78, 0.52), vec3(0.80, 0.66, 0.42), vn(wp * 0.3, 8u)) * (0.9 + 0.18 * v); }
        else if (m == WETSAND) { a = vec3(0.60, 0.48, 0.31) * (0.9 + 0.18 * v); }
        else if (m == SILT) { a = vec3(0.50, 0.42, 0.30) * (0.9 + 0.2 * v); al = 0.9; tr = 0.3; }
        else if (m == SNOW) {
            a = vec3(0.92, 0.95, 1.0) * (0.94 + 0.08 * v);
            if (mD == AIR) { al = 0.9; tr = 0.9; }
        }
        else if (m == ICE) { a = mix(vec3(0.70, 0.85, 0.95), vec3(0.86, 0.94, 1.0), v); if (v > 0.9) a = vec3(0.97, 0.99, 1.0); al = 0.92; tr = 0.94; }
        else if (m == WATER) {
            bool top = mU == AIR || CLS[mU] == 3;
            a = mix(vec3(0.10, 0.36, 0.55), vec3(0.13, 0.42, 0.60), v);
            al = 0.62; tr = 0.93;
            if (top) { a = vec3(0.55, 0.78, 0.90); al = 0.85; }
            if (mD == AIR) { a = vec3(0.55, 0.70, 0.90); al = 0.7; tr = 0.95; }       /* a falling drop */
        }
        else if (m == OIL) { a = vec3(0.10, 0.08, 0.06) + vec3(0.06, 0.03, 0.08) * v; al = 0.95; tr = 0.4; if (mU == AIR) a += vec3(0.12, 0.1, 0.16) * vn(wp * 0.3 + time * 0.2, 9u); }
        else if (m == LAVA) {
            float n = vn(wp * 0.35 + vec2(time * 0.4, time * 0.15), 10u) * 0.6 + vn(wp * 0.9 - time * 0.6, 11u) * 0.4;
            float hot = clamp((T - 1000.0) / 650.0, 0.0, 1.0);
            a = vec3(0.25, 0.06, 0.02);
            e = mix(vec3(0.75, 0.12, 0.02), vec3(1.0, 0.62, 0.16), n * hot) * (0.5 + 1.1 * hot);
            if (n < 0.3 && hot < 0.6) e *= 0.45;       /* crust */
        }
        else if (m == MAGMA) {
            float n = vn(wp * 0.25 + vec2(time * 0.2, 0.0), 12u);
            a = vec3(0.2, 0.03, 0.01);
            e = mix(vec3(0.6, 0.08, 0.02), vec3(1.0, 0.45, 0.1), n) * 1.1;
        }
        else if (m == STEAM) {
            gs.g = clamp((T - 315.0) / 160.0, 0.0, 1.0) * 0.9 + 0.05;
            al = 0.0; tr = 0.97;
        }
        else if (m == SMOKE) {
            float age = float(c.z) / 1100.0;
            gs.r = clamp(1.0 - age, 0.15, 1.0);
            al = 0.0; tr = 0.88;
        }
        else if (m == CLOUD) {
            float wet = float(min(c.w, 4u)) / 4.0;
            a = mix(vec3(0.97, 0.97, 0.98), vec3(0.55, 0.57, 0.62), wet * 0.8) * (0.95 + 0.08 * v);
            al = 0.78 + 0.15 * wet; tr = 0.955 - 0.02 * wet;
        }
        else if (m == FIRE) {
            float l = clamp(float(c.z) / 40.0, 0.0, 1.0);
            float fl = h1(p, uint(tick / 3u));
            e = mix(vec3(0.9, 0.18, 0.02), vec3(1.0, 0.8, 0.3), l) * (1.4 + 0.8 * fl);
            a = vec3(0.3, 0.05, 0.0); al = 0.5; tr = 1.0;
        }
        else if (m == EMBER) {
            int o = int(c.w & 63u);
            a = o == LEAF || o == PLANT ? vec3(0.15, 0.08, 0.03) : vec3(0.12, 0.08, 0.06);
            float fl = h1(p, uint(tick / 4u));
            e = vec3(1.0, 0.34, 0.06) * (0.7 + 0.9 * fl);
            if (o == LEAF) al = 0.9;
        }
        else if (m == BOLT) { e = vec3(1.6, 1.6, 2.4) * 2.5; a = vec3(1.0); al = 1.0; tr = 1.0; }
        else if (m == WOOD) {
            bool fir = (c.w & 256u) != 0u;
            float st = vn(vec2(wp.x * 0.9, wp.y * 0.15), 13u);
            a = (fir ? vec3(0.27, 0.18, 0.12) : vec3(0.36, 0.25, 0.15)) * (0.75 + 0.35 * st + 0.1 * v);
        }
        else if (m == LEAF) {
            bool fir = (c.w & 256u) != 0u;
            if (fir) a = mix(vec3(0.10, 0.27, 0.17), vec3(0.16, 0.36, 0.20), v);
            else {
                vec3 g = mix(vec3(0.22, 0.48, 0.16), vec3(0.40, 0.62, 0.18), v);
                vec3 au = v < 0.33 ? vec3(0.85, 0.32, 0.10) : v < 0.7 ? vec3(0.93, 0.62, 0.15) : vec3(0.70, 0.45, 0.12);
                a = mix(g, au, season.x);
                a = mix(a, vec3(0.42, 0.66, 0.22), season.z * 0.4);
            }
            a *= 0.85 + 0.2 * vn(wp * 0.4, 14u);
            al = 1.0; tr = fir ? 0.45 : 0.55;
        }
        else if (m == PLANT) {
            vec3 g = mix(vec3(0.32, 0.58, 0.20), vec3(0.45, 0.68, 0.25), v);
            g = mix(g, vec3(0.70, 0.62, 0.30), season.x * 0.7);
            g = mix(g, vec3(0.50, 0.45, 0.32), season.y * 0.6);
            a = g;
            if (mU != PLANT && v > 0.86 && season.y < 0.5) {     /* a flower */
                float f = fract(v * 37.0);
                a = f < 0.33 ? vec3(0.95, 0.85, 0.30) : f < 0.66 ? vec3(0.92, 0.50, 0.65) : vec3(0.75, 0.70, 0.95);
            }
            al = 1.0; tr = 0.8;
        }
        else if (m == ASH) { a = (c.w != 0u ? vec3(0.20, 0.15, 0.14) : vec3(0.50, 0.48, 0.46)) * (0.85 + 0.25 * v); if (mD == AIR) tr = 0.9; }
        else if (m == SEED) { a = vec3(0.50, 0.38, 0.18); if (mD == AIR) tr = 0.9; }
        else if (m == LITTER) { a = mix(vec3(0.62, 0.30, 0.10), vec3(0.75, 0.52, 0.18), v); if (mD == AIR) tr = 0.9; }
        else { a = vec3(1.0, 0.0, 1.0); }

        /* hot matter glows */
        if (m != FIRE && m != BOLT && m != LAVA && m != MAGMA && m != AIR && CLS[m] != 3) e += heat(T);
        o_alb = vec4(a, al);
        o_emi = vec4(e, tr);
        bool fall = (CLS[m] == 1 || CLS[m] == 2) && mD == AIR;
        bool open = CLS[mU] >= 3;
        o_mat = uint(m) | (fall ? 64u : 0u) | (open ? 128u : 0u);
        o_gas = gs;
    }
);

/* LIGHT: skylight, propagated a row a pass down through the grid from the
 * previous frame's (so after a change it settles in a few seconds), slanting
 * with the sun, leaking sideways a little under overhangs */
static const char *FS_LIGHT = "#version 330 core\n" GLSL(
    uniform sampler2D prev, emi;
    uniform ivec2 N;
    uniform vec3 slant;                /* weights of the up-left, up, up-right cells */
    out vec2 o;
    float lo(ivec2 q) { q.x = clamp(q.x, 0, N.x - 1); return q.y >= N.y ? 1.0 : texelFetch(prev, q, 0).r; }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        float tr = texelFetch(emi, p, 0).a;
        float lin = (slant.x * lo(p + ivec2(-1, 1)) + slant.y * lo(p + ivec2(0, 1)) + slant.z * lo(p + ivec2(1, 1))) / (slant.x + slant.y + slant.z);
        float side = 0.5 * (lo(p + ivec2(-1, 0)) + lo(p + ivec2(1, 0)));
        lin = max(lin, side * 0.86);
        o = vec2(lin * tr, lin);
    }
);

/* GLOW: the light of hot things, the emissive mip chain summed wide and
 * soft, at cell resolution */
static const char *FS_GLOW = "#version 330 core\n" GLSL(
    uniform sampler2D emi;
    uniform ivec2 N;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / vec2(N);
        vec3 g = textureLod(emi, uv, 1.0).rgb * 0.35 + textureLod(emi, uv, 2.5).rgb * 0.45 +
                 textureLod(emi, uv, 4.0).rgb * 0.55 + textureLod(emi, uv, 5.5).rgb * 0.6;
        vec3 b = textureLod(emi, uv, 3.0).rgb * 0.18 + textureLod(emi, uv, 5.0).rgb * 0.22;
        o = vec4(g * 1.6, dot(b, vec3(0.3, 0.5, 0.2)));
    }
);

/* CELL: each cell's final colour, at cell resolution: lit by the sky where it
 * reaches, by the cutaway's even light in the cross-section, by the glow of
 * hot things; over the sky (or the caves' back wall) where it isn't solid.
 * A second target carries what the screen pass draws inside the cell. */
static const char *FS_CELL = "#version 330 core\n" NOISE_COMMON GLSL(
    uniform sampler2D alb, emi, lit, back, glow, gas;
    uniform usampler2D mat;
    uniform ivec2 N;
    uniform float CS, cut;
    uniform vec3 sky_top, sky_hor, sun_col, amb, moon_col;
    uniform vec4 sun;                  /* world px x, y (from the top-left), radius, visible */
    uniform vec4 moon;
    uniform float flash;
    uniform vec4 world;                /* sea row, feature scale (rows), cloud base row, seed salt */
    uniform uint seed;
    layout(location = 0) out vec4 o_col;   /* colour; alpha: how much sky shows (for the stars) */
    layout(location = 1) out vec4 o_over;  /* rain/snow colour or the lip factor; kind bits */

    float hs(vec2 p) { uvec2 q = uvec2(ivec2(p) + 4096); uint h = q.x * 0x9E3779B1u ^ (q.y + uint(world.w)) * 0x85EBCA77u; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15; return float(h & 0xffffu) / 65535.0; }
    float vn1(float x, float s) { float i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f); return mix(hs(vec2(i, s)), hs(vec2(i + 1.0, s)), f); }
    float ridge(float x, float s) {
        float a = 0.0, w = 0.5;
        for (int i = 0; i < 4; i++) { a += w * (1.0 - abs(vn1(x, s + float(i) * 13.0) * 2.0 - 1.0)); x *= 2.1; w *= 0.5; }
        return a;
    }
    void main() {
        ivec2 ci = ivec2(gl_FragCoord.xy);
        vec2 cf = vec2(ci) + 0.5;
        vec4 A = texelFetch(alb, ci, 0);
        vec4 E = texelFetch(emi, ci, 0);
        float lin = texelFetch(lit, ci, 0).g;
        vec4 G = texelFetch(glow, ci, 0);
        vec3 glo = G.rgb;
        uint mi = texelFetch(mat, ci, 0).r;
        int m = int(mi & 63u);
        bool fall = (mi & 64u) != 0u, open = (mi & 128u) != 0u;

        /* behind: the sky, the far ranges, or the back wall of the caves */
        float bw = texelFetch(back, ivec2(ci.x, 0), 0).r;
        float sea = world.x, U = world.y;
        vec3 bg;
        float skyvis = 0.0;
        if (cf.y < bw) {
            /* the back wall: the same beds, in shadow, with a little relief */
            vec3 rock = strata(cf, seed % 97u) * vec3(0.50, 0.48, 0.48);
            float n = vn(cf * 0.35, 31u) * 0.6 + vn(cf * 0.9, 32u) * 0.4;
            rock *= 0.7 + 0.5 * n;
            /* recessed: darker near the cave's walls */
            float occ = 0.0;
            for (int i = 0; i < 8; i++) {
                ivec2 d = ivec2(i == 0 ? 2 : i == 1 ? -2 : i == 2 ? 0 : i == 3 ? 0 : i == 4 ? 1 : i == 5 ? -1 : i == 6 ? 1 : -1,
                                i == 0 ? 0 : i == 1 ? 0 : i == 2 ? 2 : i == 3 ? -2 : i == 4 ? 1 : i == 5 ? 1 : i == 6 ? -1 : -1);
                occ += texelFetch(alb, clamp(ci + d, ivec2(0), N - 1), 0).a > 0.9 ? 1.0 : 0.0;
            }
            rock *= 1.0 - occ * 0.07;
            bg = rock * (cut * 0.85 + lin * 0.6 * sun_col + glo * vec3(1.3, 1.1, 1.0));
        } else {
            float hz = 1.0 - exp(-max(cf.y - sea, 0.0) / (U * 0.55));
            bg = mix(sky_hor, sky_top, hz);
            skyvis = hz;
            /* sun and moon: pixel-art discs, with a halo */
            vec2 wc = vec2(cf.x, float(N.y) - cf.y) * CS;
            if (sun.w > 0.0) {
                float d = length(wc - sun.xy) / sun.z;
                bg += sun_col * exp(-d * 0.55) * 0.35 * sun.w;
                if (d < 1.0) { bg = mix(bg, vec3(1.0, 0.93, 0.72) * 1.25, sun.w); skyvis = 0.0; }
            }
            if (moon.w > 0.0) {
                float d = length(wc - moon.xy) / moon.z;
                float d2 = length(wc - moon.xy - vec2(moon.z * 0.45, -moon.z * 0.15)) / moon.z;
                bg += moon_col * exp(-d * 0.9) * 0.12 * moon.w;
                if (d < 1.0) { bg = mix(bg, moon_col * (d2 < 0.95 ? 0.35 : 1.0), moon.w); skyvis = 0.0; }
            }
            /* two far ranges, paler with distance */
            float x = float(ci.x);
            float r1 = sea + U * (0.12 + 0.30 * ridge(x * 0.006, world.w));
            float r2 = sea + U * (0.02 + 0.16 * ridge(x * 0.013 + 50.0, world.w + 5.0));
            vec3 haze = mix(sky_hor, sky_top, 0.25);
            if (cf.y < r1) {
                /* far peaks: bluer and paler towards their feet, in the haze */
                vec3 c1 = mix(haze, vec3(0.30, 0.34, 0.45) * (amb + sun_col * 0.4), 0.42);
                c1 = mix(c1, haze, clamp((r1 - cf.y) / (U * 0.35), 0.0, 1.0) * 0.35);
                if (cf.y > r1 - U * 0.03 && cf.y > sea + U * 0.28) c1 = mix(c1, vec3(0.9, 0.93, 1.0) * (amb + sun_col * 0.6), 0.45);
                bg = c1; skyvis = 0.0;
            }
            if (cf.y < r2) {
                vec3 c2 = mix(haze, vec3(0.20, 0.27, 0.27) * (amb + sun_col * 0.5), 0.62);
                c2 = mix(c2, haze, clamp((r2 - cf.y) / (U * 0.25), 0.0, 1.0) * 0.3);
                bg = c2; skyvis = 0.0;
            }
            bg += flash * vec3(0.5, 0.5, 0.7) * (1.0 - hz * 0.5);
        }

        /* the cell, lit */
        vec3 skyl = sun_col * lin + amb * (0.3 + 0.7 * lin);
        float inside = smoothstep(bw + 4.0, bw - 3.0, cf.y);
        vec3 cutl = vec3(1.0, 0.94, 0.86) * cut * mix(1.0, 0.72, clamp((bw - cf.y) / max(bw, 1.0), 0.0, 1.0)) * inside;
        vec3 light = max(skyl, cutl) + glo + flash * 0.6 * lin + 0.02;
        vec3 col = A.rgb * light;
        float cover = A.a;
        vec4 over = vec4(0.0);
        if (m == 10 && !fall) {
            /* water: clear in the shallows, deepening to ink, a bright skin */
            float d = clamp(lin * 1.15, 0.0, 1.0);
            vec3 deep = vec3(0.015, 0.07, 0.13);
            col = mix(deep * (light + 0.3), A.rgb * (skyl * 0.9 + 0.12 + cutl * 0.6), d) + glo * A.rgb;
            cover = mix(0.94, 0.55, d);
            if (open) over = vec4(mix(sky_hor, sky_top, 0.3) * 1.15 + sun_col * 0.25 * sun.w, 4.0);
        } else if (fall && (m == 10 || m == 8)) {
            /* rain and snow: drawn by the screen pass inside the cell */
            over = vec4(m == 10 ? vec3(0.70, 0.80, 0.95) * (max(lin, 0.35) * sun_col + amb + 0.15) + glo * 0.5
                                : vec3(0.95, 0.97, 1.0) * (max(lin, 0.5) * sun_col + amb + 0.2), m == 10 ? 1.0 : 2.0);
            cover = 0.0;
        } else if (A.a > 0.95) {
            /* a solid's exposed faces: a lit lip on top, dark outlines elsewhere */
            float e = 0.0;
            if (open) e += 1.0;
            if (texelFetch(alb, max(ci - ivec2(0, 1), ivec2(0)), 0).a < 0.5) e += 2.0;
            if (texelFetch(alb, max(ci - ivec2(1, 0), ivec2(0)), 0).a < 0.5) e += 4.0;
            if (texelFetch(alb, min(ci + ivec2(1, 0), N - 1), 0).a < 0.5) e += 8.0;
            if (e > 0.0) over = vec4(lin > 0.25 ? 1.1 + 0.4 * lin : 0.8, e, 0.0, 3.0);
        } else if (m == 15) {
            /* cloud: a few flat tones, lit from above */
            float t = clamp(lin, 0.0, 1.0);
            float tone = t > 0.85 ? 1.0 : t > 0.62 ? 0.84 : t > 0.4 ? 0.7 : 0.58;
            vec3 under = vec3(0.42, 0.45, 0.55);
            col = mix(under, A.rgb, tone) * (sun_col * 0.85 + amb * 0.7 + 0.08) + glo * 0.6 + flash * 0.8;
        } else if (m == 0) {
            /* the tail of a raindrop's streak */
            uint b1 = texelFetch(mat, max(ci - ivec2(0, 1), ivec2(0)), 0).r, b2 = texelFetch(mat, max(ci - ivec2(0, 2), ivec2(0)), 0).r;
            float k = (b1 & 127u) == 74u ? 1.3 : (b2 & 127u) == 74u ? 1.5 : 0.0;
            if (k > 0.0) over = vec4(vec3(0.70, 0.80, 0.95) * (max(lin, 0.35) * sun_col + amb + 0.15), k);
        }
        vec3 c = mix(bg, col, cover) + E.rgb;
        /* smoke and steam, soft */
        if (cover < 0.9) {
            vec2 uv = cf / vec2(N);
            vec2 g = textureLod(gas, uv, 0.0).rg * 0.4 + textureLod(gas, uv, 1.0).rg * 0.35 + textureLod(gas, uv, 2.0).rg * 0.45;
            float sm = clamp(g.r * 1.1, 0.0, 0.85), st = clamp(g.g * 1.0, 0.0, 0.75);
            vec3 lt = (sun_col * max(lin, 0.3) + amb) * (1.0 - inside) + cutl * 0.8 + glo * 0.8;
            c = mix(c, vec3(0.20, 0.19, 0.19) * lt + glo * 0.15, sm);
            c = mix(c, vec3(0.90, 0.92, 0.95) * lt, st * (1.0 - inside * 0.6));
        }
        /* bloom over everything */
        c += glo * 0.1 * (1.0 - cover * 0.5);
        o_col = vec4(c, skyvis * (1.0 - cover));
        o_over = over;
    }
);

/* COMPOSITE: the screen. Each pixel takes its cell's colour, and draws the
 * little things inside cells: rain streaks, snowflakes, lit lips on the
 * ground, the stars; then the caption */
static const char *FS_COMP = "#version 330 core\n" GLSL(
    uniform sampler2D cellc, over, cap;
    uniform ivec2 N;
    uniform float CS;
    uniform vec4 view;                 /* world px of this output's top-left, world px per screen px */
    uniform vec2 scr;
    uniform float stars, time, salt;
    uniform vec4 capr;                 /* caption: screen px x, y (bottom-left), scale, alpha */
    uniform vec2 capsz;
    out vec4 o;

    float hs(vec2 p) { uvec2 q = uvec2(ivec2(p) + 4096); uint h = q.x * 0x9E3779B1u ^ (q.y + uint(salt)) * 0x85EBCA77u; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15; return float(h & 0xffffu) / 65535.0; }
    vec3 soft(vec3 c) {                /* a soft shoulder instead of clipping */
        vec3 k = 0.78 + 0.22 * (1.0 - exp(-(c - 0.78) / 0.22));
        return mix(c, k, step(0.78, c));
    }
    void main() {
        vec2 sp = vec2(gl_FragCoord.x, scr.y - gl_FragCoord.y);        /* screen px, from the top */
        vec2 wp = view.xy + sp * view.zw;                               /* world px, from the top-left */
        vec2 cf = vec2(wp.x / CS, float(N.y) - wp.y / CS);              /* cells, y up */
        ivec2 ci = clamp(ivec2(floor(cf)), ivec2(0), N - 1);
        vec2 fr = fract(cf);
        vec4 C = texelFetch(cellc, ci, 0);
        vec4 V = texelFetch(over, ci, 0);
        vec3 col = C.rgb;
        float px = 1.0 / CS;
        if (V.a > 0.5) {
            if (V.a < 1.6) col = mix(col, V.rgb, (1.75 - V.a) * step(abs(fr.x - 0.5) * CS, 0.55));
            else if (V.a < 2.5) col = mix(col, V.rgb, 0.95 * step(length((fr - 0.5) * CS), 1.0));
            else if (V.a < 3.5) {
                int e = int(V.g + 0.5);
                float edge = px * 1.01;
                if ((e & 1) != 0 && fr.y > 1.0 - edge) col *= V.r;
                else if ((e & 2) != 0 && fr.y < edge) col *= 0.62;
                else if ((e & 4) != 0 && fr.x < edge) col *= 0.75;
                else if ((e & 8) != 0 && fr.x > 1.0 - edge) col *= 0.75;
            }
            else if (fr.y > 1.0 - px * 1.01) col = mix(col, V.rgb, 0.9);
        }
        /* stars, a pixel or two each */
        if (stars > 0.0 && C.a > 0.0) {
            vec2 sc = floor(wp / 2.0);
            float s = hs(sc);
            if (s > 0.9975) {
                float tw = 0.6 + 0.4 * sin(time * (1.0 + hs(sc + 3.0) * 3.0) + s * 40.0);
                col += vec3(0.85, 0.88, 1.0) * stars * tw * (0.4 + 0.6 * hs(sc + 9.0)) * C.a;
            }
        }
        /* the caption */
        if (capr.w > 0.0) {
            vec2 q = (gl_FragCoord.xy - capr.xy) / capr.z;
            if (q.x >= 0.0 && q.y >= 0.0 && q.x < capsz.x && q.y < capsz.y) {
                vec2 t = texelFetch(cap, ivec2(q), 0).rg;
                col = mix(col, col * 0.35, t.g * capr.w * 0.8);
                col = mix(col, vec3(0.93, 0.86, 0.72), t.r * capr.w);
            }
        }
        col = soft(col);
        col += (hs(gl_FragCoord.xy + time * 60.0) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    }
);

/* ======================================================================
 * GL
 * ====================================================================== */
static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "sand: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, VS));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

static GLuint p_move, p_react, p_shift, p_stats, p_shade, p_light, p_comp, vao;
static GLuint t_st[2], fb_st[2], t_alb, t_emi, fb_shade, t_lt[2], fb_lt[2];
static GLuint t_surf, t_back, t_shift, t_stats, fb_stats, t_cap, t_mat, p_wind;
static GLuint p_glow, p_cell, t_glow, fb_glow, t_cell, t_over, fb_cell, t_gas;
static int cur, lt_cur;
static struct { GLint S, N, tick, seed, off, wind, cb0, cb1; } um;
static struct { GLint S, N, tick, seed, surf, clim, vent, chamber, erupt, magma, water, storm, grow, grow2, bolt, cb0, cb1, seaedge, puff, puff2; } ur;

static GLuint tex2d(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLint filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint fbo_for(GLuint t0, GLuint t1, GLuint t2) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t0, 0);
    if (t1) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, t1, 0);
        if (t2) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, t2, 0);
        GLenum b[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
        glDrawBuffers(t2 ? 3 : 2, b);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("incomplete framebuffer");
    return f;
}

static float *surf_cur, *back_cur;     /* the climate's ground line and the back wall, per column, as the geology moves them */
static signed char *shift_col;

static void gl_init(void) {
    p_move = mk_prog(FS_MOVE);
    p_react = mk_prog(FS_REACT);
    p_shift = mk_prog(FS_SHIFT);
    p_stats = mk_prog(FS_STATS);
    p_shade = mk_prog(FS_SHADE);
    p_light = mk_prog(FS_LIGHT);
    p_comp = mk_prog(FS_COMP);
    p_wind = mk_prog(FS_WIND);
    p_glow = mk_prog(FS_GLOW);
    p_cell = mk_prog(FS_CELL);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    um.S = U(p_move, "S"); um.N = U(p_move, "N"); um.tick = U(p_move, "tick"); um.seed = U(p_move, "seed");
    um.off = U(p_move, "off"); um.wind = U(p_move, "wind"); um.cb0 = U(p_move, "cb0"); um.cb1 = U(p_move, "cb1");
    ur.S = U(p_react, "S"); ur.N = U(p_react, "N"); ur.tick = U(p_react, "tick"); ur.seed = U(p_react, "seed");
    ur.surf = U(p_react, "surf"); ur.clim = U(p_react, "clim"); ur.vent = U(p_react, "vent"); ur.chamber = U(p_react, "chamber"); ur.erupt = U(p_react, "erupt"); ur.magma = U(p_react, "magma");
    ur.water = U(p_react, "water"); ur.storm = U(p_react, "storm"); ur.grow = U(p_react, "grow"); ur.grow2 = U(p_react, "grow2");
    ur.bolt = U(p_react, "bolt"); ur.cb0 = U(p_react, "cb0"); ur.cb1 = U(p_react, "cb1");
    ur.seaedge = U(p_react, "seaedge"); ur.puff = U(p_react, "puff"); ur.puff2 = U(p_react, "puff2");
    t_cap = tex2d(GL_RG8, 8, 8, GL_RG, GL_UNSIGNED_BYTE, NULL, GL_NEAREST);
}

/* the world's textures (sized by it) */
static void gl_world(void) {
    for (int i = 0; i < 2; i++) {
        t_st[i] = tex2d(GL_RGBA16UI, NX, NY, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, cells, GL_NEAREST);
        fb_st[i] = fbo_for(t_st[i], 0, 0);
        t_lt[i] = tex2d(GL_RG16F, NX, NY, GL_RG, GL_FLOAT, NULL, GL_NEAREST);
        fb_lt[i] = fbo_for(t_lt[i], 0, 0);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    t_alb = tex2d(GL_RGBA8, NX, NY, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST);
    t_emi = tex2d(GL_RGBA16F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    t_mat = tex2d(GL_R8UI, NX, NY, GL_RED_INTEGER, GL_UNSIGNED_BYTE, NULL, GL_NEAREST);
    t_gas = tex2d(GL_RGBA8, NX, NY, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    fb_shade = fbo_for(t_alb, t_emi, t_mat);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, t_gas, 0);
    {
        GLenum b[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
        glDrawBuffers(4, b);
    }
    t_glow = tex2d(GL_RGBA16F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR);
    fb_glow = fbo_for(t_glow, 0, 0);
    t_cell = tex2d(GL_RGBA16F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
    t_over = tex2d(GL_RGBA16F, NX, NY, GL_RGBA, GL_FLOAT, NULL, GL_NEAREST);
    fb_cell = fbo_for(t_cell, t_over, 0);
    surf_cur = malloc(NX * sizeof *surf_cur);
    back_cur = malloc(NX * sizeof *back_cur);
    shift_col = calloc(NX, 1);
    for (int x = 0; x < NX; x++) { surf_cur[x] = surf0[x]; back_cur[x] = surf0[x] - 3; }
    t_surf = tex2d(GL_R32F, NX, 1, GL_RED, GL_FLOAT, surf_cur, GL_NEAREST);
    t_back = tex2d(GL_R32F, NX, 1, GL_RED, GL_FLOAT, back_cur, GL_NEAREST);
    t_shift = tex2d(GL_R8I, NX, 1, GL_RED_INTEGER, GL_BYTE, shift_col, GL_NEAREST);
    t_stats = tex2d(GL_RGBA32UI, NX, 2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, NULL, GL_NEAREST);
    fb_stats = fbo_for(t_stats, 0, 0);
}

static void gl_world_free(void) {
    GLuint t[] = {t_st[0], t_st[1], t_lt[0], t_lt[1], t_alb, t_emi, t_mat, t_gas, t_glow, t_cell, t_over, t_surf, t_back, t_shift, t_stats};
    GLuint f[] = {fb_st[0], fb_st[1], fb_lt[0], fb_lt[1], fb_shade, fb_glow, fb_cell, fb_stats};
    glDeleteTextures(sizeof t / sizeof *t, t);
    glDeleteFramebuffers(sizeof f / sizeof *f, f);
    free(surf_cur); free(back_cur); free(shift_col);
}

/* ======================================================================
 * nature's schedule: everything here is a function of the seed, the tick
 * and the stats read back from the GPU, so every instance agrees
 * ====================================================================== */
static uint32_t tick;                  /* sim ticks since the world was made */
#define PREROLL 300
static double season_phase0;           /* from the date the world started */
static int tied;                       /* the machine steers the climate */
static float tie_load = 0.1f, tie_temp = 45;

/* what the controllers make of the stats */
static double W0 = -1, TR0 = -1, wdef, tdef;
static float *target;                  /* the ground profile the geology keeps to */
static int *surf_meas, *top_meas;
static int shift_pending;

/* the uniforms the passes get, worked out once a tick */
static struct {
    int wind, wind_step, storm_x0, storm_x1, bolt[4];
    double wind_acc;
    float clim[4], vent[4];
    int erupt[4], water[4], grow[4], grow2[4], magma[2], puff[4], puff2[4];
    float season[4], storm_i, volc_phase, flash_tick;
} sc;
static uint32_t last_bolt = 0xffffffffu;

static float storm_intensity(double t) {
    float a = vnoise1((float)(t / 170.0), 301) * 0.7f + vnoise1((float)(t / 47.0), 302) * 0.3f;
    return smooth(0.42f, 0.72f, a);
}

static void schedule(uint32_t tk) {
    double t = (double)tk / HZ;
    /* seasons: a year every 24 minutes, starting from the real one */
    double ph = season_phase0 + 2 * PI * t / (24 * 60.0);
    double a = fmod(ph, 2 * PI);
    if (a < 0) a += 2 * PI;
    float winter = (float)cos(ph);
    season_k = -9 * winter + (tied ? clampf((tie_temp - 50) * 0.25f, -3, 9) : 0);
    float A = (float)(a / PI);                                   /* 0 midwinter, 1 midsummer, 2 */
    float autumn = A > 1.0f ? smooth(1.2f, 1.5f, A) : 1 - smooth(0.25f, 0.42f, A);
    sc.season[0] = autumn;
    sc.season[1] = smooth(-0.2f, 0.7f, winter);
    sc.season[2] = smooth(0.3f, 0.5f, A) * (1 - smooth(0.7f, 0.9f, A));
    sc.clim[0] = tsea + season_k; sc.clim[1] = lapse; sc.clim[2] = j_sea; sc.clim[3] = j_mag;

    /* wind */
    sc.wind = (int)(256 * clampf(1.5f * fbm1((float)(t / 55.0), 3, 311) + 0.25f * sinf((float)(t / 400.0)), -1, 1));
    sc.wind_acc += sc.wind / 256.0 * 2.2 / HZ;             /* up to 2.2 cells a second */
    sc.wind_step = sc.wind_acc >= 1 ? 1 : sc.wind_acc <= -1 ? -1 : 0;
    sc.wind_acc -= sc.wind_step;

    /* weather: storms come and go; a thirsty world gets more of them */
    float si = storm_intensity(t) + (float)fmax(wdef, 0) * 3 + (tied ? tie_load * 0.6f : 0);
    si = clampf(si, 0, 1);
    sc.storm_i = si;
    float cx = NX * (0.1f + 0.8f * vnoise1((float)(t / 220.0), 321));
    float hw = NX * 0.18f + 30;
    sc.storm_x0 = (int)(cx - hw); sc.storm_x1 = (int)(cx + hw);
    sc.water[0] = 60;
    sc.water[1] = (int)(8 + 260 * si);
    sc.water[2] = (int)(1 + 5 * (1 - si) + 30 * fmax(-wdef - 0.03, 0));
    sc.water[3] = (int)(si * si * 40 * (wdef > 0.08 ? 2 : 1) * (wdef < -0.05 ? 0 : 1));
    /* cumulus: now and then in fair weather, one after another in a storm */
    sc.puff2[1] = 0;
    {
        float pp = 1.0f / (HZ * 14) + si * si / (HZ * 6.0f);
        if (h01((int)tk, 3, 361) < pp) {
            int stormy = h01((int)tk, 4, 361) < si;
            float sz = (stormy ? 1.3f : 0.8f) * (0.6f + 0.8f * h01((int)tk, 5, 361)) * fmaxf(Ufeat / 300, 0.6f);
            sc.puff[0] = stormy ? sc.storm_x0 + (int)(h01((int)tk, 6, 361) * (sc.storm_x1 - sc.storm_x0))
                                : (int)(h01((int)tk, 6, 361) * NX);
            sc.puff[1] = cb0 + (int)((cb1 - cb0) * (0.1f + 0.35f * h01((int)tk, 7, 361)));
            sc.puff[2] = (int)(18 * sz) + 4;
            sc.puff[3] = (int)(9 * sz) + 3;
            sc.puff2[0] = stormy ? 2 + (h01((int)tk, 8, 361) < 0.4f) : 1;
            sc.puff2[1] = 1;
            sc.puff2[2] = (int)(hash3((int)tk, 9, 361) & 0xffff);
        }
    }
    /* lightning in the thick of it */
    sc.bolt[2] = 0;
    float lp = (si - 0.6f) * 0.0016f;
    if (lp > 0 && h01((int)tk, 0, 331) < lp) {
        /* it goes for the highest thing about: of a few places under the storm */
        int bx = 0, bh = -1;
        for (int k = 0; k < 6; k++) {
            int x = clampi(sc.storm_x0 + (int)(h01((int)tk, 10 + k, 331) * (sc.storm_x1 - sc.storm_x0)), 8, NX - 9);
            int h = top_meas ? top_meas[x] : 0;
            if (h > bh) { bh = h; bx = x; }
        }
        sc.bolt[0] = bx;
        sc.bolt[1] = cb0 + 2;
        sc.bolt[2] = 1;
        sc.bolt[3] = (int)(hash3((int)tk, 2, 331) & 0xffff);
        last_bolt = tk;
    }

    /* the volcano: quiet, a rumble and smoke, a fountain, waning */
    double vt = t + h01(0, 0, 341) * 400;
    int k = 0;
    double c0 = 0, len = 0;
    for (;; k++) {
        len = 330 + 240 * h01(k, 0, 342);
        if (vt < c0 + len) break;
        c0 += len;
    }
    float f = (float)((vt - c0) / len);
    float strength = (0.7f + 0.6f * h01(k, 1, 342)) * (tied ? 0.8f + tie_load * 0.8f : 1);
    sc.volc_phase = f;
    float build = smooth(0.6f, 0.8f, f) * (1 - smooth(0.9f, 0.96f, f));
    float erupt = smooth(0.82f, 0.84f, f) * (1 - smooth(0.88f, 0.92f, f));
    sc.erupt[0] = (int)(400 * build);
    sc.erupt[1] = (int)(1400 * erupt * strength);
    sc.erupt[2] = (int)(60 * build + 300 * erupt);
    sc.erupt[3] = (int)(6 + 150 * build * build + 900 * erupt);
    sc.grow2[2] = (A > 0.35f && A < 1.3f && tdef > 0.15) ? (int)(tdef * 5) : 0;     /* of 2^20 */
    {
        float lo = vent_top - 0.25f * (vent_top - vent_bot), hi = vent_top + 3;
        sc.magma[0] = (int)mixf(lo, hi, fmaxf(smooth(0.6f, 0.78f, f) * (1 - smooth(0.9f, 0.98f, f)), 0));
        sc.magma[1] = (int)(300 + 3000 * (build + erupt));
    }
    sc.vent[0] = vent_x; sc.vent[1] = vent_bot; sc.vent[2] = vent_top; sc.vent[3] = crater_w;

    /* life */
    float spring = (A > 0.3f && A < 1.25f) ? 1 : 0;
    sc.grow[0] = 260;
    sc.grow[1] = (int)(500 * spring * (A < 0.7f ? 1 : 0.4f));
    sc.grow[2] = (A > 1.45f && A < 1.85f) ? 12 : 0;
    sc.grow[3] = (A > 0.9f && A < 1.5f) ? (int)(2 + 6 * clampf((float)tdef * 4, 0, 1)) : 0;
    sc.grow2[0] = (int)(40 * clampf(1 + (float)tdef * 5, 0, 4));
    sc.grow2[1] = 24;
    sc.grow2[3] = 3;
}

/* every couple of seconds: read the per-column stats back, steer the water
 * budget and the forest, and pick a stretch of country to lift or let down */
static void stats_and_geology(void) {
    static unsigned *st;
    static int st_nx;
    if (st_nx != NX) { free(st); st = malloc((size_t)NX * 2 * 4 * sizeof *st); st_nx = NX; }   /* sized by the world */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_stats);
    glViewport(0, 0, NX, 2);
    glUseProgram(p_stats);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t_st[cur]);
    glUniform1i(U(p_stats, "S"), 0);
    glUniform2i(U(p_stats, "N"), NX, NY);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, NX, 2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, st);
    double wat = 0, trees = 0;
    for (int x = 0; x < NX; x++) {
        unsigned *a = st + 4 * x, *b = st + 4 * (NX + x);
        wat += a[1] + a[2];
        trees += b[1];
        surf_meas[x] = (int)a[3];
        top_meas[x] = (int)b[3];
    }
    if (W0 < 0) {
        W0 = wat; TR0 = fmax(trees, NX * 0.1);
        for (int x = 0; x < NX; x++) target[x] = surf_meas[x];
    }
    wdef = (W0 - wat) / fmax(W0, 1);
    if (getenv("SAND_FIRELOG")) {
        double fire = 0;
        for (int x = 0; x < NX; x++) fire += st[4 * (NX + x) + 2];
        if (fire > 30) fprintf(stderr, "fire %5.0f at %.0f s, storm %.2f\n", fire, (tick - PREROLL) / (double)HZ, sc.storm_i);
    }
    tdef = (TR0 - trees) / fmax(TR0, 1);

    /* the geology: where the ground has strayed furthest from its profile */
    int best = -1;
    float be = 0;
    for (int x = 4; x < NX - 4; x++) {
        if (abs(x - vent_x) < fmaxf(crater_w + 34, (vent_top - j_sea) * 1.3f)) continue;   /* the volcano builds and breaks its own cone */
        float e = 0;
        for (int i = -4; i <= 4; i++) e += surf_meas[x + i] - target[x + i];
        e /= 9;
        float s = fabsf(e) + h01(x, (int)tick, 351) * 0.1f;
        if (s > be) { be = s; best = x; }
    }
    if (getenv("SAND_GEO") && tick % (HZ * 60) == 0) {
        float ev = 0, ea = 0;
        int nv = 0, na = 0;
        for (int x = 0; x < NX; x++) {
            float e = surf_meas[x] - target[x];
            if (abs(x - vent_x) < 40) { ev += e; nv++; } else { ea += fabsf(e); na++; }
        }
        fprintf(stderr, "t %4.0f s  water %.3f trees %.3f  volcano mean err %+.1f  elsewhere mean |err| %.1f  best %d %.1f\n",
                (tick - PREROLL) / (double)HZ, wdef, tdef, ev / nv, ea / na, best, be);
    }
    if (best >= 0 && be >= 2.5f) {
        float e = 0;
        for (int i = -4; i <= 4; i++) e += surf_meas[best + i] - target[best + i];
        int s = e > 0 ? -1 : 1;
        int hw = 3 + (int)(h01(best, (int)tick, 352) * 4);
        memset(shift_col, 0, NX);
        for (int x = best - hw; x <= best + hw; x++)
            if (x >= 0 && x < NX && abs(x - vent_x) >= fmaxf(crater_w + 26, (vent_top - j_sea) * 1.3f - 8)) { shift_col[x] = (signed char)s; surf_cur[x] += s; back_cur[x] += s; }
        shift_pending = 1;
    }
}

/* --bench: GPU time per pass */
enum { PR_MOVE, PR_REACT, PR_SHADE, PR_LIGHT, PR_GLOW, PR_CELL, PR_COMP, PR_N };
static const char *prof_name[PR_N] = {"move", "react", "shade", "light", "glow", "cell", "comp"};
static GLuint prof_q[PR_N];
static double prof_ms[PR_N];
static int prof_on;
static void prof_begin(int i) { if (prof_on) glBeginQuery(GL_TIME_ELAPSED, prof_q[i]); }
static void prof_end(void) { if (prof_on) glEndQuery(GL_TIME_ELAPSED); }
static void prof_collect(void) {
    for (int i = 0; i < PR_N; i++) {
        GLuint64 ns = 0;
        glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &ns);
        prof_ms[i] += ns * 1e-6;
    }
}

static void bind_tex(int unit, GLuint t) { glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, t); }

static void sim_tick(void) {
    schedule(tick);
    glViewport(0, 0, NX, NY);
    /* MOVE */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_st[1 - cur]);
    glUseProgram(p_move);
    bind_tex(0, t_st[cur]);
    glUniform1i(um.S, 0);
    glUniform2i(um.N, NX, NY);
    glUniform1ui(um.tick, tick);
    glUniform1ui(um.seed, opt_seed);
    glUniform2i(um.off, tick & 1, tick & 1);
    glUniform1i(um.wind, sc.wind);
    glUniform1i(um.cb0, cb0); glUniform1i(um.cb1, cb1);
    prof_begin(PR_MOVE);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_end();
    /* REACT */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_st[cur]);
    glUseProgram(p_react);
    bind_tex(0, t_st[1 - cur]);
    bind_tex(1, t_surf);
    glUniform1i(ur.S, 0);
    glUniform1i(ur.surf, 1);
    glUniform2i(ur.N, NX, NY);
    glUniform1ui(ur.tick, tick);
    glUniform1ui(ur.seed, opt_seed);
    glUniform4fv(ur.clim, 1, sc.clim);
    glUniform4fv(ur.vent, 1, sc.vent);
    glUniform4f(ur.chamber, ch_x, ch_y, ch_rx, ch_ry);
    glUniform4iv(ur.erupt, 1, sc.erupt);
    glUniform2iv(ur.magma, 1, sc.magma);
    glUniform4iv(ur.water, 1, sc.water);
    glUniform2i(ur.storm, sc.storm_x0, sc.storm_x1);
    glUniform4iv(ur.puff, 1, sc.puff);
    glUniform4iv(ur.puff2, 1, sc.puff2);
    glUniform4iv(ur.grow, 1, sc.grow);
    glUniform4iv(ur.grow2, 1, sc.grow2);
    glUniform4iv(ur.bolt, 1, sc.bolt);
    glUniform1i(ur.cb0, cb0); glUniform1i(ur.cb1, cb1);
    glUniform2i(ur.seaedge, sea_side < 0 ? 0 : NX - 1, j_sea);
    prof_begin(PR_REACT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_end();
    /* the cloud deck, when the wind has carried it a cell */
    if (sc.wind_step) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb_st[1 - cur]);
        glUseProgram(p_wind);
        bind_tex(0, t_st[cur]);
        glUniform1i(U(p_wind, "S"), 0);
        glUniform2i(U(p_wind, "N"), NX, NY);
        glUniform1i(U(p_wind, "dir"), sc.wind_step);
        glUniform1i(U(p_wind, "y0"), cb0 - 16);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        cur = 1 - cur;
    }
    /* the geology, when it moves */
    if (shift_pending) {
        bind_tex(0, t_shift);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, 1, GL_RED_INTEGER, GL_BYTE, shift_col);
        bind_tex(0, t_surf);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, 1, GL_RED, GL_FLOAT, surf_cur);
        bind_tex(0, t_back);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, 1, GL_RED, GL_FLOAT, back_cur);
        glBindFramebuffer(GL_FRAMEBUFFER, fb_st[1 - cur]);
        glUseProgram(p_shift);
        bind_tex(0, t_st[cur]);
        bind_tex(1, t_shift);
        glUniform1i(U(p_shift, "S"), 0);
        glUniform1i(U(p_shift, "shift"), 1);
        glUniform2i(U(p_shift, "N"), NX, NY);
        glUniform1i(U(p_shift, "base0"), j_mag + 4);
        bind_tex(2, t_surf);
        glUniform1i(U(p_shift, "surf"), 2);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        cur = 1 - cur;
        shift_pending = 0;
    }
    tick++;
    if (tick % 120 == 0) stats_and_geology();
}

/* ======================================================================
 * the caption: a little pixel font
 * ====================================================================== */
static const struct { char c; unsigned char r[7]; } font[] = {
    {'0', {14, 17, 19, 21, 25, 17, 14}}, {'1', {4, 12, 4, 4, 4, 4, 14}}, {'2', {14, 17, 1, 2, 4, 8, 31}},
    {'3', {30, 1, 1, 14, 1, 1, 30}}, {'4', {2, 6, 10, 18, 31, 2, 2}}, {'5', {31, 16, 30, 1, 1, 17, 14}},
    {'6', {6, 8, 16, 30, 17, 17, 14}}, {'7', {31, 1, 2, 4, 8, 8, 8}}, {'8', {14, 17, 17, 14, 17, 17, 14}},
    {'9', {14, 17, 17, 15, 1, 2, 12}}, {'a', {0, 0, 14, 1, 15, 17, 15}}, {'w', {0, 0, 17, 17, 21, 21, 10}},
    {'y', {0, 0, 17, 17, 15, 1, 14}}, {'h', {16, 16, 22, 25, 17, 17, 17}}, {'m', {0, 0, 26, 21, 21, 21, 21}},
    {'d', {1, 1, 13, 19, 17, 17, 15}}, {'s', {0, 0, 15, 16, 14, 1, 30}}, {' ', {0, 0, 0, 0, 0, 0, 0}},
};
static int cap_w, cap_h;
static char cap_text[64];

static void caption(const char *s) {
    if (!strcmp(s, cap_text)) return;
    snprintf(cap_text, sizeof cap_text, "%s", s);
    int n = (int)strlen(s);
    cap_w = n * 6 + 2; cap_h = 9;
    unsigned char *px = calloc((size_t)cap_w * cap_h * 2, 1);
    for (int i = 0; i < n; i++) {
        const unsigned char *g = NULL;
        for (size_t k = 0; k < sizeof font / sizeof *font; k++) if (font[k].c == s[i]) g = font[k].r;
        if (!g) continue;
        for (int r = 0; r < 7; r++)
            for (int b = 0; b < 5; b++) {
                if (!(g[r] >> (4 - b) & 1)) continue;
                int x = 1 + i * 6 + b, y = 7 - r;      /* texture rows go up */
                px[2 * (y * cap_w + x)] = 255;
                for (int dy = -1; dy <= 0; dy++)       /* a shadow, down and right */
                    for (int dx = 0; dx <= 1; dx++) {
                        int sx = x + dx, sy = y + dy;
                        if (sx < cap_w && sy >= 0) px[2 * (sy * cap_w + sx) + 1] = 255;
                    }
            }
    }
    bind_tex(0, t_cap);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, cap_w, cap_h, 0, GL_RG, GL_UNSIGNED_BYTE, px);
    free(px);
}

static void away_text(double wt, char *buf, size_t n) {
    long s = (long)(wt - (double)locked_at);
    if (s < 0) s = 0;
    long m = s / 60, h = m / 60, d = h / 24;
    if (d > 0) snprintf(buf, n, "away %ldd %ldh", d, h % 24);
    else if (h > 0) snprintf(buf, n, "away %ldh %02ldm", h, m % 60);
    else snprintf(buf, n, "away %ldm", m);
}

/* ======================================================================
 * rendering
 * ====================================================================== */
static void mix3(float *o, const float *a, const float *b, float t) { for (int i = 0; i < 3; i++) o[i] = a[i] + (b[i] - a[i]) * t; }

static void light_passes(int n, const float *slant) {
    glUseProgram(p_light);
    glUniform1i(U(p_light, "prev"), 0);
    glUniform1i(U(p_light, "emi"), 1);
    glUniform2i(U(p_light, "N"), NX, NY);
    glUniform3fv(U(p_light, "slant"), 1, slant);
    glViewport(0, 0, NX, NY);
    bind_tex(1, t_emi);
    for (int i = 0; i < n; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fb_lt[1 - lt_cur]);
        bind_tex(0, t_lt[lt_cur]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        lt_cur = 1 - lt_cur;
    }
}

static int light_fresh = 1;            /* the skylight hasn't been propagated yet: do it all at once */
static void render(GLuint target, double wt, double secs, int settle) {
    /* the time of day */
    float hour;
    if (opt_hour >= 0) hour = opt_hour;
    else {
        time_t tt = (time_t)wt;
        struct tm lt;
        localtime_r(&tt, &lt);
        hour = lt.tm_hour + lt.tm_min / 60.0f + lt.tm_sec / 3600.0f;
    }
    float el = sinf((hour - 6.5f) / 12.0f * PI);              /* sun elevation, -1..1 */
    float day = smooth(-0.05f, 0.25f, el), dusk = expf(-el * el / 0.03f), night = 1 - smooth(-0.25f, 0.0f, el);
    float top_d[3] = {0.22f, 0.45f, 0.80f}, hor_d[3] = {0.66f, 0.81f, 0.93f};
    float top_k[3] = {0.24f, 0.22f, 0.45f}, hor_k[3] = {0.98f, 0.58f, 0.34f};
    float top_n[3] = {0.012f, 0.02f, 0.055f}, hor_n[3] = {0.05f, 0.075f, 0.14f};
    float sky_top[3], sky_hor[3], sun_col[3], amb[3], t3[3];
    mix3(sky_top, top_n, top_d, day); mix3(t3, sky_top, top_k, dusk * 0.6f); memcpy(sky_top, t3, sizeof t3);
    mix3(sky_hor, hor_n, hor_d, day); mix3(t3, sky_hor, hor_k, dusk * 0.85f); memcpy(sky_hor, t3, sizeof t3);
    float sd[3] = {1.0f, 0.95f, 0.86f}, sk[3] = {1.0f, 0.62f, 0.36f}, sn[3] = {0.16f, 0.22f, 0.36f};
    mix3(sun_col, sn, sd, day); mix3(t3, sun_col, sk, dusk * 0.7f * (1 - night)); memcpy(sun_col, t3, sizeof t3);
    float ad[3] = {0.30f, 0.36f, 0.48f}, an[3] = {0.05f, 0.07f, 0.12f};
    mix3(amb, an, ad, day);
    float flash = last_bolt != 0xffffffffu && tick >= last_bolt ? expf(-(float)(tick - last_bolt) / 7.0f) : 0;

    /* the sun and moon cross the whole wall */
    float lw = NX * CS, horizon = (NY - j_sea) * CS;
    float sx = lw * clampf((hour - 6) / 12, -0.2f, 1.2f), sy = horizon - el * horizon * 0.95f;
    float mh = fmodf(hour + 12, 24), mel = sinf((mh - 6.5f) / 12.0f * PI);
    float mx = lw * clampf((mh - 6) / 12, -0.2f, 1.2f), my = horizon - mel * horizon * 0.95f;
    /* skylight slants with the sun */
    float sdir = el > 0 ? (hour < 12 ? 1.0f : -1.0f) * (1 - el) : 0;
    float slant[3] = {1 + fmaxf(sdir, 0) * 2, 1.5f, 1 + fmaxf(-sdir, 0) * 2};

    /* SHADE */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_shade);
    glViewport(0, 0, NX, NY);
    glUseProgram(p_shade);
    bind_tex(0, t_st[cur]);
    glUniform1i(U(p_shade, "S"), 0);
    glUniform2i(U(p_shade, "N"), NX, NY);
    glUniform1ui(U(p_shade, "tick"), tick);
    glUniform1ui(U(p_shade, "seed"), opt_seed);
    glUniform1f(U(p_shade, "time"), (float)fmod(secs, 1000.0));
    glUniform4fv(U(p_shade, "season"), 1, sc.season);
    prof_begin(PR_SHADE);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    bind_tex(0, t_emi);
    glGenerateMipmap(GL_TEXTURE_2D);
    bind_tex(0, t_gas);
    glGenerateMipmap(GL_TEXTURE_2D);
    prof_end();
    /* LIGHT */
    prof_begin(PR_LIGHT);
    light_passes(settle || light_fresh ? NY + 20 : 2, slant);
    light_fresh = 0;
    prof_end();

    /* GLOW */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_glow);
    glUseProgram(p_glow);
    bind_tex(0, t_emi);
    glUniform1i(U(p_glow, "emi"), 0);
    glUniform2i(U(p_glow, "N"), NX, NY);
    prof_begin(PR_GLOW);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_end();

    /* CELL: only the cells this output shows */
    glBindFramebuffer(GL_FRAMEBUFFER, fb_cell);
    glUseProgram(p_cell);
    {
        int x0 = (int)floorf((vw_x - wx0) / CS) - 1, x1 = (int)ceilf((vw_x - wx0 + vw_w) / CS) + 1;
        int y1 = NY - (int)floorf((vw_y - wy0) / CS) + 1, y0 = NY - (int)ceilf((vw_y - wy0 + vw_h) / CS) - 1;
        glEnable(GL_SCISSOR_TEST);
        glScissor(x0, y0, x1 - x0, y1 - y0);
    }
    bind_tex(0, t_alb); bind_tex(1, t_emi); bind_tex(2, t_lt[lt_cur]); bind_tex(3, t_back); bind_tex(4, t_glow); bind_tex(5, t_mat);
    glUniform1i(U(p_cell, "alb"), 0); glUniform1i(U(p_cell, "emi"), 1); glUniform1i(U(p_cell, "lit"), 2);
    glUniform1i(U(p_cell, "back"), 3); glUniform1i(U(p_cell, "glow"), 4); glUniform1i(U(p_cell, "mat"), 5);
    bind_tex(6, t_gas);
    glUniform1i(U(p_cell, "gas"), 6);
    glUniform1f(U(p_cell, "cut"), 0.30f + 0.16f * day);
    glUniform2i(U(p_cell, "N"), NX, NY);
    glUniform1f(U(p_cell, "CS"), (float)CS);
    glUniform3fv(U(p_cell, "sky_top"), 1, sky_top);
    glUniform3fv(U(p_cell, "sky_hor"), 1, sky_hor);
    glUniform3fv(U(p_cell, "sun_col"), 1, sun_col);
    glUniform3fv(U(p_cell, "amb"), 1, amb);
    glUniform3f(U(p_cell, "moon_col"), 0.85f, 0.87f, 0.80f);
    float sr = fmaxf(Ufeat * 0.022f, 4) * CS;
    glUniform4f(U(p_cell, "sun"), sx, sy, sr, el > -0.1f ? 1.0f : 0.0f);
    glUniform4f(U(p_cell, "moon"), mx, my, sr * 0.8f, mel > -0.1f ? 1 - day * 0.7f : 0.0f);
    glUniform1f(U(p_cell, "flash"), flash);
    glUniform4f(U(p_cell, "world"), (float)j_sea, Ufeat, (float)cb0, (float)(opt_seed % 997));
    glUniform1ui(U(p_cell, "seed"), opt_seed);
    prof_begin(PR_CELL);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_end();
    glDisable(GL_SCISSOR_TEST);

    /* COMPOSITE */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_comp);
    bind_tex(0, t_cell); bind_tex(1, t_over); bind_tex(2, t_cap);
    glUniform1i(U(p_comp, "cellc"), 0); glUniform1i(U(p_comp, "over"), 1); glUniform1i(U(p_comp, "cap"), 2);
    glUniform2i(U(p_comp, "N"), NX, NY);
    glUniform1f(U(p_comp, "CS"), (float)CS);
    glUniform4f(U(p_comp, "view"), vw_x - wx0, vw_y - wy0, vw_w / W, vw_h / H);
    glUniform2f(U(p_comp, "scr"), (float)W, (float)H);
    glUniform1f(U(p_comp, "stars"), night);
    glUniform1f(U(p_comp, "time"), (float)fmod(secs, 1000.0));
    glUniform1f(U(p_comp, "salt"), (float)(opt_seed % 997));
    if (opt_caption && front_me && cap_text[0]) {
        float s = fmaxf(2, roundf(CS * 0.5f));
        float mxp = 4 * CS, myp = H - 4 * CS - cap_h * s;     /* top left, in the sky */
        glUniform4f(U(p_comp, "capr"), mxp, myp, s, 0.85f);
        glUniform2f(U(p_comp, "capsz"), (float)cap_w, (float)cap_h);
    } else glUniform4f(U(p_comp, "capr"), 0, 0, 1, 0);
    prof_begin(PR_COMP);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_end();
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
    /* a resize just rescales the view: the world keeps its size */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "sand");
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
        "usage: sand [--window | --root | --layer] [--seed N] [--fps N] [--cell PX] [--no-caption]\n"
        "            [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "            [--hour H] [--season 0..1] [--load 0..1] [--temp C]\n"
        "            [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--when]\n");
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

/* the world for the current size: laid out, generated, settled a little */
static int world_w, world_h;
static void make_world(void) {
    if (cells) {
        gl_world_free();
        free(cells); free(surf0); free(target); free(surf_meas); free(top_meas);
    }
    world_w = W; world_h = H;
    setup_world();
    worldgen();
    gl_world();
    target = malloc(NX * sizeof *target);
    surf_meas = calloc(NX, sizeof *surf_meas);
    top_meas = calloc(NX, sizeof *top_meas);
    tick = 0; cur = 0; lt_cur = 0; W0 = -1; last_bolt = 0xffffffffu;
    memset(&sc, 0, sizeof sc);
    light_fresh = 1;
    for (int i = 0; i < PREROLL; i++) sim_tick();
    W0 = -1;
}

/* catch the sim up to the tick for sim time el (seconds since the epoch),
 * spending at most budget seconds of GPU time on it */
static void catch_up(double el, double budget) {
    uint32_t want = PREROLL + (uint32_t)fmax(el * HZ, 0);
    if (want <= tick) return;
    uint32_t behind = want - tick;
    if (behind <= 4) { while (tick < want) sim_tick(); return; }
    double t0 = now();
    while (tick < want) {
        for (int i = 0; i < (budget > 1 ? 240 : 8) && tick < want; i++) sim_tick();
        glFinish();
        if (now() - t0 > budget) break;
    }
}

int main(int argc, char **argv) {
    int opt_when = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-cell") && i + 1 < argc) opt_cell = atoi(argv[++i]);
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-hour") && i + 1 < argc) opt_hour = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-season") && i + 1 < argc) opt_season = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-load") && i + 1 < argc) opt_load = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-temp") && i + 1 < argc) opt_temp = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-when")) opt_when = 1;
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    int offscreen = opt_shot || opt_bench > 0;

    /* --epoch: when the world began, shared by every instance; a --shot then
     * shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;
    {   /* the seasons start from the real one */
        time_t tt = (time_t)t_wall0;
        struct tm lt;
        localtime_r(&tt, &lt);
        season_phase0 = ((lt.tm_yday + 0.5) / 365.25 - 15 / 365.25) * 2 * PI;
        if (opt_season >= 0) season_phase0 = opt_season * 2 * PI;
    }
    find_temp();
    tied = span_me < 0 && !offscreen;
    if (opt_load >= 0) { tied = 1; tie_load = opt_load; }
    if (opt_temp >= 0) { tied = 1; tie_temp = opt_temp; }
    if (tied && opt_load < 0) { read_sensors(); }

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (offscreen || opt_when) { W = off_w; H = off_h; }
    if (opt_when) {
        setup_world();
        printf("sand: world %dx%d cells of %d px\n", NX, NY, CS);
        int was_e = 0, was_s = 0;
        for (uint32_t tk = 0; tk < PREROLL + 3600u * HZ; tk++) {
            schedule(tk);
            if (tk < PREROLL) continue;
            int e = sc.erupt[1] > 0, s = sc.storm_i > 0.6f;
            double t = (tk - PREROLL) / (double)HZ;
            if (e && !was_e) printf("  %7.1f s  eruption\n", t);
            if (s && !was_s) printf("  %7.1f s  storm\n", t);
            if (sc.bolt[2]) printf("  %7.1f s  lightning at x %d\n", t, sc.bolt[0]);
            was_e = e; was_s = s;
        }
        return 0;
    }
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
        XStoreName(dpy, win, "Sand");
        XClassHint chint = {"sand", "Sand"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
        if (!offscreen && !opt_root) {
            /* wait for the window's real size before laying out the world */
            for (int i = 0; i < 50; i++) {
                XEvent ev;
                if (XCheckTypedWindowEvent(dpy, win, ConfigureNotify, &ev)) { W = ev.xconfigure.width; H = ev.xconfigure.height; }
                struct timespec ts = {0, 10000000};
                nanosleep(&ts, NULL);
            }
        }
    }

    gl_init();
    make_world();
    char cap[64];

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (opt_shot) {
            double t0 = now();
            catch_up(opt_at, 1e9);
            double wt = t_wall0 + opt_at;
            away_text(wt, cap, sizeof cap);
            caption(cap);
            render(fbo, wt, opt_at, 1);
            glFinish();
            save_ppm(opt_shot);
            if (getenv("SAND_DUMP")) {
                /* debugging: the world's state, raw (NX NY, then 4 x u16 a cell, bottom row first) */
                uint16_t *st = malloc((size_t)NX * NY * 8);
                glBindFramebuffer(GL_FRAMEBUFFER, fb_st[cur]);
                glPixelStorei(GL_PACK_ALIGNMENT, 2);
                glReadPixels(0, 0, NX, NY, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, st);
                FILE *f = fopen(getenv("SAND_DUMP"), "wb");
                if (f) { int d[2] = {NX, NY}; fwrite(d, sizeof d, 1, f); fwrite(st, 8, (size_t)NX * NY, f); fclose(f); }
                free(st);
            }
            fprintf(stderr, "sand: %dx%d cells of %d px, %u ticks in %.1f s\n", NX, NY, CS, tick, now() - t0);
            return 0;
        }
        /* --bench: a tick and a frame each, as live */
        render(fbo, wall(), 0, 1);
        glFinish();
        double sum = 0, worst = 0, simsum = 0;
        glGenQueries(PR_N, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            sim_tick();
            glFinish();
            double s1 = now();
            render(fbo, wall(), i / 60.0, 0);
            glFinish();
            prof_collect();
            double ms = (now() - f0) * 1000;
            simsum += (s1 - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("sand: %dx%d, world %dx%d cells of %d px, %d frames: avg %.3f ms (sim %.3f), worst %.3f ms per frame\n",
               W, H, NX, NY, CS, opt_bench, sum / opt_bench, simsum / opt_bench, worst);
        for (int i = 0; i < PR_N; i++) printf("  %-6s %6.3f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        printf("sand: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double last_read = 0, t_resized = 0;
    int last_w = W, last_h = H;
    /* debugging: SAND_RESIZE_TEST=N resizes the window N times, then quits */
    int rtest = dpy && !opt_root && getenv("SAND_RESIZE_TEST") ? atoi(getenv("SAND_RESIZE_TEST")) : 0, rdone = 0;
    double rnext = now() + 1;
    for (;;) {
        if (rtest && now() > rnext) {
            if (rdone == rtest) {
                char st[256];
                long rss = 0;
                FILE *f = fopen("/proc/self/statm", "r");
                if (f) { if (fscanf(f, "%*s %ld", &rss) != 1) rss = 0; fclose(f); }
                snprintf(st, sizeof st, "sand: resize test: %d resizes, now %dx%d, world %dx%d cells, rss %ld KB\n",
                         rdone, W, H, NX, NY, rss * (sysconf(_SC_PAGESIZE) / 1024));
                fputs(st, stderr);
                return 0;
            }
            XResizeWindow(dpy, win, 320 + (int)(hash32(rdone * 7 + 1) % 1600), 240 + (int)(hash32(rdone * 13 + 5) % 1000));
            rdone++;
            rnext = now() + (rdone % 3 ? 0.25 : 0.9);     /* some settle long enough to rebuild, some don't */
        }
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
        double t = now(), wt = wall(), el = wt - t_wall0;
        /* standalone, a new size gets a world of its own once it's settled
         * (a window tiled, then fullscreened; an output's mode changed) */
        if (W != last_w || H != last_h) { last_w = W; last_h = H; t_resized = t; }
        if (span_me < 0 && (W != world_w || H != world_h) && t - t_resized > 0.5) make_world();
        if (tied && t - last_read >= 2) {
            read_sensors();
            if (opt_load < 0 && sens_load >= 0) tie_load = sens_load;
            if (opt_temp < 0 && sens_temp >= 0) tie_temp = sens_temp;
            last_read = t;
        }
        catch_up(el, 0.012);
        away_text(wt, cap, sizeof cap);
        caption(cap);
        render(0, wt, el, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
