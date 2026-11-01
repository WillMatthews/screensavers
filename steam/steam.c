/*
 * steam — the machine's vital signs on the footplate of a steam engine: the
 * backhead of a locomotive boiler, seen head-on in the warm light of a cab
 * lamp and the fire.
 *
 * The round end of the boiler, riveted iron in a green, lined cladding ring,
 * stands out of a painted cab front over worn floorboards. On it:
 *
 *   BOILER PRESSURE   CPU load, from /proc/stat deltas
 *   STEAM CHEST       the CPU package temperature, with a red line
 *   RESERVOIR         memory in use, in GB (with no battery: the fan speed,
 *                     or else the load average, as DRAUGHT)
 *   two water gauge glasses, Klinger-striped, whose level is the battery
 *   charge (with no battery: free memory); the water bobs, and a green
 *   jewel by the right-hand glass is lit while on mains
 *   the firehole: butterfly doors left just ajar, the fire glowing through
 *   the gap and the air holes and spilling onto the floor, brighter and
 *   livelier the harder the CPU works
 *   a builder's plate, raised brass on red, "<HOST> WORKS", with UPTIME and
 *   AWAY (how long the screen has been locked) on rolling drums
 *
 * Round it, on the cab front: a railway clock in a mahogany surround, a
 * Watt governor on the steam pipe whose flyballs spin faster and fly
 * higher with the load, a safety valve that now and then lifts with a gust
 * of steam (more often near the red line), a paraffin lamp, copper and
 * iron pipework with flanges, lagging and handwheels, wisps of steam
 * leaking from tired joints, and condensation dripping from the pipes.
 *
 * Everything is drawn with signed distance functions. The parts that never
 * move are drawn once into a texture; each frame adds the needles, drums,
 * water, fire, flyballs, flame, drips and steam over it, the fire and steam
 * only inside their own bounds, so a frame costs little.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size.
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

#define PI 3.14159265f
#define DEG (PI / 180.0f)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "C059", *opt_sans = "Gillius ADF";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 25;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;

static void die(const char *what) { fprintf(stderr, "steam: %s\n", what); exit(1); }

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

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }

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

/* ---- sensors ---- */
static char host[64];
static time_t locked_at;
static char temp_path[300], batt_dir[256], fan_path[300];
enum { AUX_BATT, AUX_FAN, AUX_LOADAVG };
static int aux_kind;
static int ncpu = 1;

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

static void find_aux(void) {
    glob_t g;
    if (glob("/sys/class/power_supply/BAT*/capacity", 0, NULL, &g) == 0 && g.gl_pathc) {
        const char *p = g.gl_pathv[0];
        snprintf(batt_dir, sizeof batt_dir, "%.*s", (int)(strrchr(p, '/') - p), p);
        aux_kind = AUX_BATT;
        globfree(&g);
        return;
    }
    if (glob("/sys/class/hwmon/hwmon*/fan*_input", 0, NULL, &g) == 0 && g.gl_pathc) {
        snprintf(fan_path, sizeof fan_path, "%s", g.gl_pathv[0]);
        aux_kind = AUX_FAN;
        globfree(&g);
        return;
    }
    aux_kind = AUX_LOADAVG;
}

static int on_mains(void) {
    glob_t g;
    int on = 0;
    if (glob("/sys/class/power_supply/*/type", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            char t[32], path[300];
            slurp_line(g.gl_pathv[i], t, sizeof t);
            if (strcmp(t, "Mains")) continue;
            snprintf(path, sizeof path, "%.*s/online", (int)(strrchr(g.gl_pathv[i], '/') - g.gl_pathv[i]), g.gl_pathv[i]);
            if (read_long(path, 0) == 1) on = 1;
        }
        globfree(&g);
    }
    return on;
}

/* latest readings */
static struct {
    float temp;                        /* degrees C */
    float load;                        /* 0..1 */
    float mem_gb, mem_total_gb;
    float aux;                         /* battery % / fan rpm / load average */
    int charging, mains;
    double uptime, uptime_at;          /* /proc/uptime, and now() when read */
    unsigned long long cpu_busy, cpu_total;
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
    if (temp_path[0]) sens.temp = read_long(temp_path, 0) / 1000.0f;

    unsigned long long b, t;
    read_cpu(&b, &t);
    if (sens.cpu_total && t > sens.cpu_total)
        sens.load = (float)(b - sens.cpu_busy) / (float)(t - sens.cpu_total);
    sens.cpu_busy = b; sens.cpu_total = t;

    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        long total = 0, avail = 0;
        while (fgets(line, sizeof line, f)) {
            sscanf(line, "MemTotal: %ld", &total);
            sscanf(line, "MemAvailable: %ld", &avail);
        }
        fclose(f);
        sens.mem_total_gb = total / 1048576.0f;
        sens.mem_gb = (total - avail) / 1048576.0f;
    }

    char buf[64], path[300];
    if (aux_kind == AUX_BATT) {
        snprintf(path, sizeof path, "%s/capacity", batt_dir);
        sens.aux = (float)read_long(path, 0);
        snprintf(path, sizeof path, "%s/status", batt_dir);
        slurp_line(path, buf, sizeof buf);
        sens.charging = !strcmp(buf, "Charging");
        sens.mains = on_mains() || sens.charging;
    } else if (aux_kind == AUX_FAN) {
        sens.aux = (float)read_long(fan_path, 0);
    } else {
        slurp_line("/proc/loadavg", buf, sizeof buf);
        sens.aux = strtof(buf, NULL);
    }

    slurp_line("/proc/uptime", buf, sizeof buf);
    sens.uptime = strtod(buf, NULL);
    sens.uptime_at = now();
}


/* ---- the instruments ---- */
enum { G_TEMP, G_LOAD, G_MEM, G_CLOCK, NG };
enum { K_DIAL, K_CLOCK };

typedef struct {
    int kind;
    float cx, cy, r;                   /* px: centre, face radius */
    float a0, a1;                      /* the sweep: radians clockwise from 12 o'clock */
    float vmin, vmax;
    int nmaj, nmin, numevery;          /* major intervals, minor per major, a numeral every n majors */
    float red0, red1, amb0, amb1;      /* painted zones, as fractions of the scale (< 0: none) */
    char label[24], unit[24];
    float omega, zeta;                 /* needle spring */
    float x, v, target;                /* needle state, fraction of the scale */
    float ph[3];                       /* tremble phases */
} Gauge;
static Gauge G[NG];

/* a "nice" full-scale and major step for 0..v */
static void nice_scale(float v, float *top, int *nmaj) {
    static const float steps[] = {0.5f, 1, 2, 2.5f, 4, 5, 10, 20, 25, 50, 100};
    for (size_t i = 0; i < sizeof steps / sizeof *steps; i++) {
        int n = (int)ceilf(v / steps[i] - 1e-3f);
        if (n <= 8) { *top = n * steps[i]; *nmaj = n < 4 ? 4 : n; if (n < 4) *top = 4 * steps[i]; return; }
    }
    *top = v; *nmaj = 5;
}

static void gauges_init(void) {
    Gauge *g;
    for (int i = 0; i < NG; i++) {
        G[i].a0 = -135 * DEG; G[i].a1 = 135 * DEG;
        G[i].red0 = G[i].red1 = G[i].amb0 = G[i].amb1 = -1;
        G[i].numevery = 1; G[i].nmin = 5;
        G[i].omega = 2 * PI * 0.8f; G[i].zeta = 0.42f;
        G[i].x = -0.03f;                   /* resting on the stop pin */
        for (int k = 0; k < 3; k++) G[i].ph[k] = frand() * 2 * PI;
    }
    g = &G[G_TEMP];
    g->vmin = 20; g->vmax = 100; g->nmaj = 8; g->nmin = 5; g->numevery = 2;
    g->red0 = (85 - 20) / 80.0f; g->red1 = 1;
    g->omega = 2 * PI * 0.45f; g->zeta = 0.5f;
    snprintf(g->label, sizeof g->label, "STEAM CHEST");
    snprintf(g->unit, sizeof g->unit, "CPU \x7f" "C");

    g = &G[G_LOAD];
    g->vmin = 0; g->vmax = 100; g->nmaj = 10; g->nmin = 2; g->numevery = 2;
    g->red0 = 0.9f; g->red1 = 1;
    g->omega = 2 * PI * 0.9f; g->zeta = 0.34f;
    snprintf(g->label, sizeof g->label, "BOILER PRESSURE");
    snprintf(g->unit, sizeof g->unit, "CPU LOAD %%");

    g = &G[G_MEM];
    g->vmin = 0; g->omega = 2 * PI * 0.6f; g->zeta = 0.45f;
    if (aux_kind == AUX_BATT) {
        nice_scale(sens.mem_total_gb, &g->vmax, &g->nmaj);
        g->nmin = 4; g->numevery = g->nmaj > 6 ? 2 : 1;
        g->amb0 = sens.mem_total_gb * 0.9f / g->vmax; g->amb1 = sens.mem_total_gb / g->vmax;
        snprintf(g->label, sizeof g->label, "RESERVOIR");
        snprintf(g->unit, sizeof g->unit, "MEMORY GB");
    } else if (aux_kind == AUX_FAN) {
        nice_scale(fmaxf(6000, sens.aux * 1.3f) / 1000, &g->vmax, &g->nmaj);
        g->nmin = 5;
        snprintf(g->label, sizeof g->label, "BLOWER");
        snprintf(g->unit, sizeof g->unit, "FAN RPM x1000");
    } else {
        nice_scale((float)ncpu, &g->vmax, &g->nmaj);
        g->nmin = 5;
        g->red0 = ncpu / g->vmax; g->red1 = 1;
        snprintf(g->label, sizeof g->label, "DRAUGHT");
        snprintf(g->unit, sizeof g->unit, "LOAD AVERAGE");
    }

    g = &G[G_CLOCK];
    g->kind = K_CLOCK;
    g->a0 = 0; g->a1 = 2 * PI; g->nmaj = 12; g->nmin = 5;
    g->omega = 2 * PI * 7.0f; g->zeta = 0.3f;    /* the seconds hand: a quick tick with a bounce */
}

/* the water in the gauge glasses: the battery, or else free memory */
static struct { float x, v, target; } wat = {0.0f, 0, 0};

/* the needles' targets from the latest readings */
static void set_targets(void) {
    Gauge *g = &G[G_TEMP];
    g->target = (sens.temp - g->vmin) / (g->vmax - g->vmin);
    G[G_LOAD].target = sens.load;
    g = &G[G_MEM];
    if (aux_kind == AUX_BATT) {
        g->target = sens.mem_gb / g->vmax;
        wat.target = sens.aux / 100.0f;
    } else {
        if (aux_kind == AUX_FAN) g->target = sens.aux / 1000.0f / g->vmax;
        else g->target = sens.aux / g->vmax;
        wat.target = sens.mem_total_gb > 0 ? 1 - sens.mem_gb / sens.mem_total_gb : 0.5f;
    }
    for (int i = 0; i < G_CLOCK; i++) G[i].target = clampf(G[i].target, -0.03f, 1.03f);
    wat.target = clampf(wat.target, 0, 1);
}

/* ---- the engine's moving parts ---- */
static float sec_angle, sec_vel;       /* the clock's seconds hand, degrees */
static float load_s;                   /* the load, smoothed */
static float gov_w = 0.3f, gov_th = 0.65f, gov_thv, gov_ph;   /* governor: rev/s, ball swing (rad), phase */
static float fire_s = 0.5f;            /* how hard the fire is drawing */
#define NPUFF 3
static struct { float t0, dur, amp, seed; } puff[NPUFF];   /* safety-valve lifts */
static int puff_n;
static float sv_next;                  /* the valve stays seated until at least then */
static uint32_t erng;                  /* the events' own random stream */
static float efrand(void) {
    erng ^= erng << 13; erng ^= erng >> 17; erng ^= erng << 5;
    return (erng >> 8) * (1.0f / 16777216.0f);
}

/* one step: needle springs (the stop pins bounce them), the water, the
 * governor, the fire, and now and then a lift of the safety valve.
 * t: wall-clock time (for the clock); st: seconds since start */
static void physics(double t, double st, float dt, int powered) {
    for (int i = 0; i < G_CLOCK; i++) {
        Gauge *g = &G[i];
        float tgt = powered ? g->target : -0.03f;
        float a = g->omega * g->omega * (tgt - g->x) - 2 * g->zeta * g->omega * g->v;
        g->v += a * dt;
        g->x += g->v * dt;
        if (g->x < -0.035f) { g->x = -0.035f; g->v = -0.3f * g->v; }
        if (g->x > 1.035f) { g->x = 1.035f; g->v = -0.3f * g->v; }
    }
    /* the seconds hand jumps to each new second and rings a little */
    struct tm tm;
    time_t ts = (time_t)t;
    localtime_r(&ts, &tm);
    float tgt = tm.tm_sec * 6.0f;
    float d = fmodf(tgt - sec_angle + 540.0f, 360.0f) - 180.0f;
    Gauge *c = &G[G_CLOCK];
    float a = c->omega * c->omega * d - 2 * c->zeta * c->omega * sec_vel;
    sec_vel += a * dt;
    sec_angle = fmodf(sec_angle + sec_vel * dt + 360.0f, 360.0f);

    /* the water: a slow, lightly damped slosh toward the level */
    {
        float w = 2 * PI * 0.32f, z = 0.18f;
        float wa = w * w * ((powered ? wat.target : 0.0f) - wat.x) - 2 * z * w * wat.v;
        wat.v += wa * dt;
        wat.x += wat.v * dt;
    }

    load_s += (sens.load - load_s) * dt / 2.5f;
    float ls = clampf(load_s, 0, 1);
    /* the governor: speed follows the load, the balls swing out with it on a pendulum */
    gov_w += (0.28f + 1.25f * ls - gov_w) * dt / 3.0f;
    float th_t = 0.62f + 0.45f * clampf((gov_w - 0.28f) / 1.25f, 0, 1);
    float gw = 2 * PI * 0.55f;
    gov_thv += (gw * gw * (th_t - gov_th) - 2 * 0.22f * gw * gov_thv) * dt;
    gov_th += gov_thv * dt;
    gov_ph = fmodf(gov_ph + 2 * PI * gov_w * dt, 2 * PI * 64);
    fire_s += (0.5f + 0.5f * ls - fire_s) * dt / 2.0f;

    /* the safety valve: lifts at random, far more readily near the red line */
    float pr = G[G_LOAD].x;
    if (st >= sv_next) {
        float rate = 1.0f / 40 + 0.6f * clampf((pr - 0.75f) / 0.2f, 0, 1);
        if (efrand() < rate * dt) {
            int k = puff_n++ % NPUFF;
            puff[k].t0 = (float)st;
            puff[k].dur = 1.4f + 2.4f * efrand();
            puff[k].amp = 0.75f + 0.25f * efrand();
            puff[k].seed = efrand() * 50;
            sv_next = (float)st + puff[k].dur + (pr > 0.8f ? 2.0f : 7.0f) + 8 * efrand();
        }
    }
}

/* the needle's angle as drawn, with a faint tremble */
static float needle_angle(const Gauge *g, double t) {
    float j = 0.0012f * (sinf((float)t * 7.1f + g->ph[0]) * 0.5f + sinf((float)t * 11.3f + g->ph[1]) * 0.3f +
                         sinf((float)t * 17.9f + g->ph[2]) * 0.2f);
    float x = clampf(g->x + j, -0.035f, 1.035f);
    return g->a0 + (g->a1 - g->a0) * x;
}

/* ---- glyph atlas: printable ASCII (+ the degree sign as 0x7f) in each
 * text style used, rasterised at its exact pixel size ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 32
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
enum { F_SERIF, F_SERIF_B, F_SANS, F_SANS_B };
static FT_Face faces[4];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < 4; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:weight=%s:fontformat=CFF,TrueType", i < 2 ? opt_font : opt_sans,
                 i & 1 ? "bold" : "regular");
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
    int cap = 4096;
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

/* the width of str in a face at px, before the atlas is built; track in ems */
static float measure(int face, float px, const char *str, float track) {
    FT_Set_Pixel_Sizes(faces[face], 0, (FT_UInt)lroundf(px));
    float w = 0;
    for (const char *p = str; *p; p++)
        if (!FT_Load_Char(faces[face], (unsigned char)*p, FT_LOAD_DEFAULT))
            w += (faces[face]->glyph->advance.x >> 6) + (p[1] ? track * lroundf(px) : 0);
    return w;
}

/* cap height: how far 'H' stands above the baseline */
static float cap_h(int s) { return glyph(s, 'H')->top; }

/* text quads: x, y, u, v, r, g, b, a per vertex */
static float *tv;
static int tvn, tvcap;

static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 8192) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x;
 * coloured ct at the cap height shading to cb at the baseline */
static void text2(int s, float x, float y, int align, const char *str, const float *ct, const float *cb, float track) {
    float w = text_width(s, str, track);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    pen = roundf(pen);
    y = roundf(y);
    float ch = cap_h(s);
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph(s, (unsigned char)*p);
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            float c0[4], c1[4];
            for (int k = 0; k < 4; k++) {
                float f0 = clampf((y0 - (y - ch)) / fmaxf(ch, 1), -0.3f, 1.3f), f1 = clampf((y1 - (y - ch)) / fmaxf(ch, 1), -0.3f, 1.3f);
                c0[k] = ct[k] + (cb[k] - ct[k]) * f0;
                c1[k] = ct[k] + (cb[k] - ct[k]) * f1;
            }
            quad_v(x0, y0, u0, v0, c0); quad_v(x1, y0, u1, v0, c0); quad_v(x1, y1, u1, v1, c1);
            quad_v(x0, y0, u0, v0, c0); quad_v(x1, y1, u1, v1, c1); quad_v(x0, y1, u0, v1, c1);
        }
        pen += g->adv + track;
    }
}
static void text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    text2(s, x, y, align, str, col, col, track);
}

/* centred on (x, y) by cap height */
static void text_c(int s, float x, float y, const char *str, const float *col, float track) {
    text(s, x, y + cap_h(s) * 0.5f, 0, str, col, track);
}

/* engraved into brass: a light lip under a dark fill */
static void text_engraved(int s, float x, float y, int align, const char *str, float track) {
    static const float hi[4] = {1.0f, 0.88f, 0.6f, 0.30f}, lo[4] = {0.07f, 0.05f, 0.03f, 0.88f};
    float o = fmaxf(1, roundf(styles[s].px / 28.0f));
    text(s, x, y + o, align, str, hi, track);
    text(s, x, y, align, str, lo, track);
}

/* cast in relief: brass letters standing out of a painted ground, with a
 * shadow below right and the lit bevel above left */
static void text_raised(int s, float x, float y, int align, const char *str, float track) {
    static const float sh[4] = {0.03f, 0.01f, 0.005f, 0.75f}, hl[4] = {1.0f, 0.9f, 0.62f, 0.85f};
    static const float ft[4] = {0.86f, 0.68f, 0.36f, 1.0f}, fb[4] = {0.50f, 0.34f, 0.14f, 1.0f};
    float o = fmaxf(1, roundf(styles[s].px / 18.0f));
    text(s, x + o, y + o * 1.3f, align, str, sh, track);
    text(s, x - o * 0.5f, y - o * 0.6f, align, str, hl, track);
    text2(s, x, y, align, str, ft, fb, track);
}

/* stencilled in cream paint, a bit worn */
static void text_paint(int s, float x, float y, int align, const char *str, float track) {
    static const float c[4] = {0.78f, 0.72f, 0.58f, 0.72f};
    text(s, x, y, align, str, c, track);
}

/* ---- the scene ---- */
#define MAXP 24
#define MAXF 16
#define MAXW 6
#define NLEAK 3
#define NDRIP 2
enum { M_COPPER, M_IRON, M_BRASS };

/* everything in px (y down) */
static struct {
    float s;                         /* px per design unit */
    float bh[4];                     /* the backhead: centre x, y, radius, floor y */
    float hor[4];                    /* the floor's horizon y, vanishing x, plank width per unit depth */
    float lag[4];                    /* the lagged steam main: y, radius */
    float pa[MAXP][4], pb[MAXP][4];  /* pipes: ends; radius, material */
    int np;
    float fl[MAXF][4];               /* flanges: x y angle pipe-radius */
    int nf;
    float wh[MAXW][4];               /* handwheels: x y radius */
    int nw;
    float wg[2][4];                  /* water gauge glasses: x, top, bottom, tube radius */
    float fd[4];                     /* the firehole: centre, half width, half height */
    float gov[4];                    /* governor: spindle x, pivot y, arm length, gear y */
    float govb[4];                   /* valve body y, valve radius, gear radius, side of the steam inlet */
    float sv[4];                     /* safety valve: x, seat y, size */
    float lamp[4];                   /* the lamp: flame x, y, size, side its bracket goes to */
    float ml[4];                     /* the mains jewel: x y r (r 0: none) */
    float leak[NLEAK][4];            /* leaking joints: x y size direction */
    float drip[NDRIP][4];            /* drip points: x y period phase */
    float tag[2][4];                 /* brass tags under the glasses */
    float pq[4], rule[4];            /* the builder's plate, its rule */
} SC;

#define NDRUM 17
static float drum_rect[NDRUM][4];
static float win_rect[2][4];
static int drum_spec[NDRUM];           /* which digit of the time each drum shows, see digit_of() */
static int drum_style;                 /* text style of the drum digits */

enum { D_S1, D_S10, D_M1, D_M10, D_H1, D_H10, D_D1, D_D10, D_D100 };

static int digit_of(long n, int spec) {
    switch (spec) {
    case D_S1: return (int)(n % 10);
    case D_S10: return (int)(n / 10 % 6);
    case D_M1: return (int)(n / 60 % 10);
    case D_M10: return (int)(n / 600 % 6);
    case D_H1: return (int)(n / 3600 % 24 % 10);
    case D_H10: return (int)(n / 3600 % 24 / 10);
    case D_D1: return (int)(n / 86400 % 10);
    case D_D10: return (int)(n / 864000 % 10);
    default: return (int)(n / 8640000 % 10);
    }
}
static int radix_of(int spec) { return spec == D_S10 || spec == D_M10 ? 6 : spec == D_H10 ? 3 : 10; }

typedef struct { float x, y, r; } Disc;

/* the design, in units: landscape fits a 16 x 9 box, portrait 9 x 16 */
static float BOXW, BOXH, OX, OY;
#define PX(v) (OX + (v) * SC.s)
#define PY(v) (OY + (v) * SC.s)

static void pipe_(float x0, float y0, float x1, float y1, float r, int mat) {
    if (SC.np == MAXP) return;
    float *a = SC.pa[SC.np], *b = SC.pb[SC.np];
    a[0] = PX(x0); a[1] = PY(y0); a[2] = PX(x1); a[3] = PY(y1);
    b[0] = r * SC.s; b[1] = (float)mat; b[2] = b[3] = 0;
    SC.np++;
}
static void flange(float x, float y, int vertical, float r) {
    if (SC.nf == MAXF) return;
    float *f = SC.fl[SC.nf++];
    f[0] = PX(x); f[1] = PY(y); f[2] = vertical ? PI / 2 : 0; f[3] = r * SC.s;
}
static void wheel(float x, float y, float r) {
    if (SC.nw == MAXW) return;
    float *w = SC.wh[SC.nw++];
    w[0] = PX(x); w[1] = PY(y); w[2] = r * SC.s; w[3] = 0;
}

static void layout(void) {
    int tall = W < H;
    BOXW = tall ? 9 : 16; BOXH = tall ? 9 * 16.0f / 9 : 9;
    SC.s = fminf(W / BOXW, H / BOXH);
    OX = (W - BOXW * SC.s) / 2; OY = (H - BOXH * SC.s) / 2;
    float s = SC.s;
    SC.np = SC.nf = SC.nw = 0;

    Disc d[NG];
    float pqx, pqy, pqh = 1.5f;
    float bhx, bhy, bhR, flo, hor, lagy;
    float wgx[2], wgy0, wgy1;
    float fdx, fdy;
    float gx, gpy, gL, ggy, gvy;
    float svx, svy, lpx, lpy, lside;
    if (!tall) {
        bhx = 8; bhy = 5.05f; bhR = 4.05f; flo = 8.5f; hor = 3.2f; lagy = 0.62f;
        d[G_LOAD] = (Disc){8.0f, 2.42f, 1.0f};
        d[G_TEMP] = (Disc){5.72f, 3.12f, 0.74f};
        d[G_MEM] = (Disc){10.28f, 3.12f, 0.74f};
        d[G_CLOCK] = (Disc){1.95f, 2.55f, 1.06f};
        pqx = 8; pqy = 4.62f;
        wgx[0] = 5.4f; wgx[1] = 10.6f; wgy0 = 4.5f; wgy1 = 6.7f;
        fdx = 8; fdy = 6.98f;
        gx = 14.45f; gpy = 1.6f; gL = 0.95f; ggy = 3.55f; gvy = 4.3f;
        svx = 12.4f; svy = 3.05f;
        lpx = 1.45f; lpy = 5.05f; lside = -1;

        /* left: a copper feed down the cab front into the boiler, an iron pipe with a valve */
        pipe_(3.55f, 0.6f, 3.55f, 7.3f, 0.09f, M_COPPER);
        pipe_(3.55f, 6.25f, 4.6f, 6.25f, 0.085f, M_COPPER);
        pipe_(-0.5f, 7.3f, 3.55f, 7.3f, 0.11f, M_IRON);
        flange(3.55f, 1.35f, 1, 0.09f); flange(3.55f, 4.7f, 1, 0.09f);
        flange(4.05f, 6.25f, 0, 0.085f); flange(0.75f, 7.3f, 0, 0.11f);
        wheel(2.3f, 7.3f, 0.36f);
        /* the steam fountain behind the pressure gauge, with its valves */
        pipe_(6.5f, 1.72f, 9.5f, 1.72f, 0.075f, M_COPPER);
        pipe_(8.0f, 1.72f, 8.0f, 0.7f, 0.09f, M_COPPER);
        wheel(6.5f, 1.72f, 0.24f); wheel(9.5f, 1.72f, 0.24f);
        /* right: steam main down to the governor's valve, and away to the floor */
        pipe_(13.2f, 0.6f, 13.2f, gvy, 0.1f, M_COPPER);
        pipe_(13.2f, gvy, gx, gvy, 0.1f, M_COPPER);
        pipe_(gx, gvy, 15.45f, gvy, 0.1f, M_COPPER);
        pipe_(15.45f, gvy, 15.45f, 9.6f, 0.11f, M_IRON);
        pipe_(13.2f, svy, svx, svy, 0.08f, M_COPPER);
        flange(13.2f, 1.35f, 1, 0.1f); flange(13.75f, gvy, 0, 0.1f);
        flange(15.45f, 5.6f, 1, 0.11f); flange(12.78f, svy, 0, 0.08f);
        wheel(15.45f, 7.0f, 0.36f);
        /* drains from the gauge glasses */
        pipe_(wgx[0], wgy1, wgx[0], 9.6f, 0.04f, M_COPPER);
        pipe_(wgx[1], wgy1, wgx[1], 9.6f, 0.04f, M_COPPER);

        SC.leak[0][0] = PX(4.05f); SC.leak[0][1] = PY(6.12f); SC.leak[0][2] = 0.55f * s; SC.leak[0][3] = 0.2f;
        SC.leak[1][0] = PX(gx); SC.leak[1][1] = PY(gvy - 0.42f); SC.leak[1][2] = 0.5f * s; SC.leak[1][3] = -0.15f;
        SC.leak[2][0] = PX(15.45f); SC.leak[2][1] = PY(5.5f); SC.leak[2][2] = 0.45f * s; SC.leak[2][3] = -0.3f;
        SC.drip[0][0] = PX(1.35f); SC.drip[0][1] = PY(7.3f + 0.11f); SC.drip[0][2] = 6.3f; SC.drip[0][3] = 0.4f;
        SC.drip[1][0] = PX(13.2f); SC.drip[1][1] = PY(gvy + 0.1f); SC.drip[1][2] = 8.9f; SC.drip[1][3] = 3.1f;
    } else {
        bhx = 4.5f; bhy = 8.4f; bhR = 4.1f; flo = 15.1f; hor = 8.0f; lagy = 0.7f;
        d[G_LOAD] = (Disc){4.5f, 5.78f, 0.95f};
        d[G_TEMP] = (Disc){2.35f, 7.3f, 0.72f};
        d[G_MEM] = (Disc){6.65f, 7.3f, 0.72f};
        d[G_CLOCK] = (Disc){2.3f, 2.65f, 1.12f};
        pqx = 4.5f; pqy = 8.98f;
        wgx[0] = 1.3f; wgx[1] = 7.7f; wgy0 = 9.95f; wgy1 = 12.15f;
        fdx = 4.5f; fdy = 11.3f;
        gx = 6.95f; gpy = 1.65f; gL = 0.88f; ggy = 3.45f; gvy = 4.15f;
        svx = 4.75f; svy = 3.1f;
        lpx = 8.2f; lpy = 4.95f; lside = 1;

        pipe_(5.6f, 0.7f, 5.6f, gvy, 0.1f, M_COPPER);
        pipe_(5.6f, gvy, gx, gvy, 0.1f, M_COPPER);
        pipe_(gx, gvy, 9.6f, gvy, 0.1f, M_COPPER);
        pipe_(5.6f, svy, svx, svy, 0.08f, M_COPPER);
        flange(5.6f, 1.45f, 1, 0.1f); flange(8.2f, gvy, 0, 0.1f); flange(5.2f, svy, 0, 0.08f);
        /* the fountain */
        pipe_(2.75f, 5.25f, 6.25f, 5.25f, 0.075f, M_COPPER);
        pipe_(4.5f, 5.25f, 4.5f, 4.4f, 0.09f, M_COPPER);
        wheel(2.75f, 5.25f, 0.24f); wheel(6.25f, 5.25f, 0.24f);
        /* low down: the injector feed across the backhead */
        pipe_(wgx[0], wgy1, wgx[0], 13.5f, 0.04f, M_COPPER);
        pipe_(wgx[1], wgy1, wgx[1], 13.5f, 0.04f, M_COPPER);
        pipe_(-0.5f, 13.5f, 9.5f, 13.5f, 0.1f, M_COPPER);
        pipe_(0.7f, 13.5f, 0.7f, 16.5f, 0.1f, M_IRON);
        pipe_(8.3f, 13.5f, 8.3f, 16.5f, 0.1f, M_IRON);
        flange(3.05f, 13.5f, 0, 0.1f); flange(5.95f, 13.5f, 0, 0.1f);
        flange(0.7f, 14.3f, 1, 0.1f); flange(8.3f, 14.3f, 1, 0.1f);
        wheel(2.2f, 13.5f, 0.4f); wheel(6.8f, 13.5f, 0.4f);

        SC.leak[0][0] = PX(3.05f); SC.leak[0][1] = PY(13.38f); SC.leak[0][2] = 0.55f * s; SC.leak[0][3] = 0.2f;
        SC.leak[1][0] = PX(gx); SC.leak[1][1] = PY(gvy - 0.4f); SC.leak[1][2] = 0.5f * s; SC.leak[1][3] = -0.15f;
        SC.leak[2][0] = PX(8.3f); SC.leak[2][1] = PY(14.2f); SC.leak[2][2] = 0.45f * s; SC.leak[2][3] = -0.3f;
        SC.drip[0][0] = PX(4.55f); SC.drip[0][1] = PY(13.6f); SC.drip[0][2] = 6.3f; SC.drip[0][3] = 0.4f;
        SC.drip[1][0] = PX(7.45f); SC.drip[1][1] = PY(13.6f); SC.drip[1][2] = 8.9f; SC.drip[1][3] = 3.1f;
    }
    SC.bh[0] = PX(bhx); SC.bh[1] = PY(bhy); SC.bh[2] = bhR * s; SC.bh[3] = PY(flo);
    SC.hor[0] = PY(hor); SC.hor[1] = PX(bhx); SC.hor[2] = 0.62f * s / (PY(flo) - PY(hor));
    SC.lag[0] = PY(lagy); SC.lag[1] = 0.24f * s;
    float tr = 0.1f;
    for (int i = 0; i < 2; i++) {
        SC.wg[i][0] = PX(wgx[i]); SC.wg[i][1] = PY(wgy0); SC.wg[i][2] = PY(wgy1); SC.wg[i][3] = tr * s;
        float ty = wgy1 + 0.62f;
        SC.tag[i][0] = PX(wgx[i] - 0.5f); SC.tag[i][1] = PY(ty); SC.tag[i][2] = PX(wgx[i] + 0.5f); SC.tag[i][3] = PY(ty + 0.24f);
        for (int k = 0; k < 4; k++) SC.tag[i][k] = roundf(SC.tag[i][k]);
    }
    SC.fd[0] = PX(fdx); SC.fd[1] = PY(fdy); SC.fd[2] = 1.18f * s; SC.fd[3] = 0.8f * s;
    SC.gov[0] = PX(gx); SC.gov[1] = PY(gpy); SC.gov[2] = gL * s; SC.gov[3] = PY(ggy);
    SC.govb[0] = PY(gvy); SC.govb[1] = 0.25f * s; SC.govb[2] = 0.27f * s; SC.govb[3] = -1;
    SC.sv[0] = PX(svx); SC.sv[1] = PY(svy - 0.08f); SC.sv[2] = 0.6f * s; SC.sv[3] = 0;
    SC.lamp[0] = PX(lpx); SC.lamp[1] = PY(lpy); SC.lamp[2] = 0.62f * s; SC.lamp[3] = lside;
    if (aux_kind == AUX_BATT) {
        float mx = tall ? wgx[1] + 0.55f : wgx[1] + 0.62f;
        SC.ml[0] = PX(mx); SC.ml[1] = PY(wgy0 + 0.35f); SC.ml[2] = 0.075f * s;
    } else SC.ml[2] = 0;

    for (int i = 0; i < NG; i++) {
        G[i].cx = PX(d[i].x); G[i].cy = PY(d[i].y); G[i].r = d[i].r * s;
    }

    /* the builder's plate, in plate-height tenths (k): name, rule, two windows of drums */
    float k = pqh / 10;
    float dw = 1.3f * k, dgap = 0.07f * k, ggap = 0.5f * k, wpad = 0.16f * k;
    float upw = 9 * dw + 5 * dgap + 3 * ggap + 2 * wpad;
    float pqw = 2.0f * k + 5.4f * k + 0.6f * k + upw + 2.0f * k;
    float kp = k * s;
    float pcx = PX(pqx), pcy = PY(pqy);
    float *plq = SC.pq;
    plq[0] = roundf(pcx - pqw * s / 2); plq[1] = roundf(pcy - pqh * s / 2);
    plq[2] = roundf(pcx + pqw * s / 2); plq[3] = roundf(pcy + pqh * s / 2);

    nstyles = 0;
    char hn[80];
    int n = 0;
    for (const char *p = host; *p && n < 60; p++) hn[n++] = (char)toupper((unsigned char)*p);
    hn[n] = 0;
    strcat(hn, " WORKS");
    float name_px = 2.1f * kp;
    while (name_px > 8 && measure(F_SANS_B, name_px, hn, 0.12f) > (plq[2] - plq[0]) - 5.0f * kp) name_px *= 0.94f;
    int st_name = style(F_SANS_B, name_px), st_lab = style(F_SANS_B, 1.05f * kp), st_cap = style(F_SANS, 0.8f * kp);
    float rx1 = plq[2] - 2.0f * kp;
    float wy[2] = {plq[1] + 4.6f * kp, plq[1] + 7.35f * kp}, wh = 2.05f * kp;
    int nd = 0;
    for (int w = 0; w < 2; w++) {
        int nine = w == 0;
        int per[4] = {nine ? 3 : 2, 2, 2, 2};
        int specs[4][3] = {{D_D100, D_D10, D_D1}, {D_H10, D_H1}, {D_M10, D_M1}, {D_S10, D_S1}};
        if (!nine) { specs[0][0] = D_D10; specs[0][1] = D_D1; }
        float right = rx1 - wpad * s;
        for (int gi = 3; gi >= 0; gi--) {
            for (int j = per[gi] - 1; j >= 0; j--) {
                float x1 = right, x0 = right - dw * s;
                drum_rect[nd][0] = roundf(x0); drum_rect[nd][2] = roundf(x1);
                drum_rect[nd][1] = roundf(wy[w] + wpad * s); drum_rect[nd][3] = roundf(wy[w] + wh - wpad * s);
                drum_spec[nd] = specs[gi][j];
                nd++;
                right = x0 - dgap * s;
            }
            right += dgap * s;
            if (gi) right -= ggap * s;
        }
        win_rect[w][0] = roundf(right - wpad * s); win_rect[w][1] = roundf(wy[w]);
        win_rect[w][2] = roundf(rx1); win_rect[w][3] = roundf(wy[w] + wh);
    }
    drum_style = style(F_SERIF, (drum_rect[0][3] - drum_rect[0][1]) * 0.7f);

    int st_num[NG], st_glab[NG], st_unit[NG];
    for (int i = 0; i < NG; i++) {
        float r = G[i].r;
        st_num[i] = style(i == G_CLOCK ? F_SERIF : F_SERIF_B, (i == G_CLOCK ? 0.14f : 0.15f) * r);
        st_glab[i] = style(F_SANS_B, (i == G_LOAD ? 0.075f : 0.085f) * r);
        st_unit[i] = style(F_SANS, (i == G_CLOCK ? 0.065f : 0.075f) * r);
    }
    int st_tag = style(F_SANS_B, 0.14f * s), st_mains = style(F_SANS_B, 0.11f * s);
    build_atlas();

    /* the print: numerals and labels on the faces */
    tvn = 0;
    static const float ink[4] = {0.09f, 0.075f, 0.065f, 0.94f}, red[4] = {0.6f, 0.13f, 0.09f, 0.95f};
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        float r = g->r;
        if (g->kind == K_CLOCK) {
            static const char *rom[12] = {"I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"};
            for (int h = 1; h <= 12; h++) {
                float a = h * 30 * DEG;
                text_c(st_num[i], g->cx + sinf(a) * 0.7f * r, g->cy - cosf(a) * 0.7f * r, rom[h - 1], ink, 0);
            }
            text_c(st_unit[i], g->cx, g->cy + 0.34f * r, "RAILWAY TIME", ink, 0.15f * styles[st_unit[i]].px);
            continue;
        }
        float nr = 0.655f;
        for (int k2 = 0; k2 <= g->nmaj; k2 += g->numevery) {
            char b[16];
            float v = g->vmin + (g->vmax - g->vmin) * k2 / g->nmaj;
            snprintf(b, sizeof b, "%g", v);
            float a = g->a0 + (g->a1 - g->a0) * k2 / g->nmaj;
            float f = (float)k2 / g->nmaj;
            int over = g->red0 >= 0 && f >= g->red0 - 1e-3f && f <= g->red1 + 1e-3f && i != G_MEM;
            text_c(st_num[i], g->cx + sinf(a) * nr * r, g->cy - cosf(a) * nr * r, b, over ? red : ink, 0);
        }
        float lt = 0.1f * styles[st_glab[i]].px;
        text_c(st_glab[i], g->cx, g->cy - 0.24f * r, g->label, ink, lt);
        text_c(st_unit[i], g->cx, g->cy + 0.36f * r, g->unit, ink, 0.12f * styles[st_unit[i]].px);
    }

    /* the plate: raised name, rule, captions */
    text_raised(st_name, (plq[0] + plq[2]) / 2, plq[1] + 1.75f * kp + cap_h(st_name) / 2, 0, hn, 0.12f * styles[st_name].px);
    SC.rule[0] = plq[0] + 2.0f * kp; SC.rule[2] = plq[2] - 2.0f * kp; SC.rule[1] = plq[1] + 3.25f * kp;
    SC.rule[3] = fmaxf(0.8f, 0.09f * kp);
    for (int w = 0; w < 2; w++)
        text_raised(st_lab, plq[0] + 2.0f * kp, (win_rect[w][1] + win_rect[w][3]) / 2 + cap_h(st_lab) / 2, -1,
                    w ? "AWAY" : "UPTIME", 0.1f * styles[st_lab].px);
    static const char *caps[4] = {"SEC", "MIN", "HRS", "DAYS"};
    static const int cg[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 8}};
    for (int c = 0; c < 4; c++) {
        float x = (drum_rect[cg[c][0]][2] + drum_rect[cg[c][1]][0]) / 2;
        text_raised(st_cap, x, plq[1] + 4.3f * kp, 0, caps[c], 0.14f * styles[st_cap].px);
    }

    /* tags under the gauge glasses, and the mains jewel's stencil */
    for (int i = 0; i < 2; i++) {
        const char *t = i == 0 ? "WATER" : aux_kind == AUX_BATT ? "BATTERY" : "FREE MEM";
        float tx = (SC.tag[i][0] + SC.tag[i][2]) / 2, ty = (SC.tag[i][1] + SC.tag[i][3]) / 2;
        text_engraved(st_tag, tx, ty + cap_h(st_tag) / 2, 0, t, 0.12f * styles[st_tag].px);
    }
    if (SC.ml[2] > 0)
        text_paint(st_mains, SC.ml[0], SC.ml[1] + 0.26f * s + cap_h(st_mains), 0, "MAINS", 0.15f * styles[st_mains].px);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* shared: noise, shapes, metal and paint, the firehole's openings */
#define COMMON GLSL( \
    const vec3 L = vec3(-0.4636, -0.6182, 0.6347); \
    float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); } \
    float vnoise(vec2 p) { \
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); \
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); \
    } \
    float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; } \
    float fbm3(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s / 0.875; } \
    float aa(float d) { return clamp(0.5 - d, 0.0, 1.0); } \
    float sdBox(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; } \
    float sdRect(vec2 p, vec4 R, float r) { return sdBox(p - (R.xy + R.zw) * 0.5, (R.zw - R.xy) * 0.5, r); } \
    float sdSeg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); } \
    vec3 metal(vec3 n, vec3 base) { \
        vec3 R = reflect(vec3(0.0, 0.0, -1.0), n); \
        float w = dot(R, L); \
        float env = 0.16 + 0.5 * smoothstep(-0.3, 1.0, w) + 0.55 * pow(max(w, 0.0), 6.0); \
        float dif = max(dot(n, L), 0.0); \
        return base * (0.1 + 0.32 * dif + 0.95 * env) + vec3(1.0, 0.88, 0.62) * 0.55 * pow(max(w, 0.0), 40.0); \
    } \
    vec3 paint(vec3 alb, vec3 n, float gloss) { \
        vec3 R = reflect(vec3(0.0, 0.0, -1.0), n); \
        float dif = max(dot(n, L), 0.0); \
        return alb * (0.32 + 0.8 * dif) + vec3(1.0, 0.9, 0.75) * gloss * (0.25 * pow(max(dot(R, L), 0.0), 6.0) + pow(max(dot(R, L), 0.0), 30.0)); \
    } \
    vec3 dome(vec2 sp, vec3 base) { \
        float z = sqrt(max(1.0 - dot(sp, sp), 0.0)); \
        return metal(normalize(vec3(sp * 0.9, z + 0.2)), base); \
    } \
    float fhSlots(vec2 q, vec2 hs, float u) { \
        vec2 r = vec2(abs(q.x), q.y) - vec2(hs.x * 0.52, hs.y * 0.18); \
        vec2 g = vec2(0.25 * u, 0.2 * u); \
        vec2 k = clamp(floor(r / g + 0.5), vec2(-1.0, 0.0), vec2(1.0, 1.0)); \
        return sdBox(r - k * g, vec2(0.075 * u, 0.024 * u), 0.022 * u); \
    } \
)

static const char *FS_STATIC = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform float uSeed, uS;
    uniform vec4 uBH, uHor, uLag;
    uniform vec4 uPA[24], uPB[24];
    uniform int uNP;
    uniform vec4 uFl[16];
    uniform int uNF;
    uniform vec4 uWh[6];
    uniform int uNW;
    uniform vec4 uG[4], uA[4], uZ[4];
    uniform vec4 uWG[2], uTag[2];
    uniform vec4 uFD;
    uniform vec4 uPq, uRule, uWin[2];
    uniform vec4 uGov, uGovB, uSV, uLamp, uML;
    uniform vec3 uRed, uAmber, uBrass, uCopper, uGreen;
    out vec4 o;
)
COMMON
GLSL(
    vec3 rivetP(vec2 p, vec2 c, float r, vec3 col, vec3 alb, float gloss) {
        col *= mix(1.0, 0.5, smoothstep(r * 1.7, r * 0.5, length(p - c - vec2(0.35, 0.5) * r)));
        float d = length(p - c);
        if (d < r + 1.0) {
            vec2 sp = (p - c) / r;
            vec3 n = normalize(vec3(sp * 0.9, sqrt(max(1.0 - dot(sp, sp), 0.0)) + 0.15));
            col = mix(col, paint(alb, n, gloss), aa(d - r));
        }
        return col;
    }
    vec2 rowPt(vec2 p, vec2 a, vec2 dir, float sp) { return a + dir * (floor(dot(p - a, dir) / sp + 0.5) * sp); }

    /* the cab front: green paint, worn to red primer and bare iron, rust runs, seams */
    vec3 wall(vec2 p) {
        float u = uS;
        vec2 q = p / u;
        vec3 c = uGreen * (0.72 + 0.55 * fbm(q * 0.8 + uSeed));
        c *= 0.94 + 0.08 * vnoise(vec2(q.x * 3.0, q.y * 45.0));
        float w = fbm(q * 2.1 + 17.0 + uSeed);
        c = mix(c, vec3(0.25, 0.085, 0.045), 0.75 * smoothstep(0.72, 0.75, w) * smoothstep(0.3, 0.6, vnoise(q * 9.0)));
        c = mix(c, vec3(0.07, 0.06, 0.055), smoothstep(0.77, 0.79, w));
        float run = vnoise(vec2(q.x * 11.0, q.y * 0.7 + uSeed));
        c = mix(c, vec3(0.2, 0.075, 0.035), 0.35 * smoothstep(0.72, 0.95, run) * smoothstep(0.45, 0.75, fbm(q * 1.3 + 5.0)));
        /* grime gathers low down and up toward the roof */
        c *= 1.0 - 0.3 * smoothstep(0.5, 1.0, p.y / uBH.w) - 0.35 * smoothstep(0.35, 0.0, p.y / uRes.y);
        vec3 col = paint(c, vec3(0.0, 0.0, 1.0), 0.06);
        /* plate seams either side of the boiler and across at the waist, riveted */
        float sx = uBH.z * 1.085 + 0.55 * u;
        float ax = abs(p.x - uBH.x);
        float ys = uBH.y + 0.3 * uBH.z;
        float dv = ax - sx;
        col *= 1.0 - 0.55 * aa(abs(dv + 0.5) - 0.7) + 0.25 * aa(abs(dv - 1.5) - 0.6);
        if (ax > sx) col *= 1.0 - 0.55 * aa(abs(p.y - ys) - 0.7) + 0.25 * aa(abs(p.y - ys + 1.5) - 0.6);
        float rr = 0.05 * u, sp = 0.32 * u;
        vec2 rc = vec2(uBH.x + sign(p.x - uBH.x) * (sx + 0.13 * u), 0.0);
        rc.y = floor(p.y / sp + 0.5) * sp;
        col = rivetP(p, rc, rr, col, c, 0.15);
        if (ax > sx + 0.2 * u) col = rivetP(p, vec2(floor(p.x / sp + 0.5) * sp, ys + 0.13 * u), rr, col, c, 0.15);
        return col;
    }

    float dBoiler(vec2 p) { return p.y < uBH.y ? length(p - uBH.xy) - uBH.z : abs(p.x - uBH.x) - uBH.z; }

    vec3 ironAlb(vec2 p) {
        vec2 q = p / uS;
        vec3 c = mix(vec3(0.045, 0.043, 0.042), vec3(0.1, 0.092, 0.086), fbm(q * 1.4 + uSeed + 3.0));
        c *= 0.9 + 0.14 * vnoise(q * vec2(6.0, 50.0));
        c = mix(c, vec3(0.19, 0.08, 0.04), 0.55 * smoothstep(0.64, 0.8, fbm(q * 3.1 + 41.0)));
        return c;
    }

    /* the boiler's end: green lined cladding round a riveted iron backhead */
    vec3 boiler(vec2 p, vec3 col) {
        float R = uBH.z, u = uS;
        float cw = 0.085 * R;
        float d = dBoiler(p);
        col *= mix(1.0, 0.4, smoothstep(0.3 * u, -0.05 * u, dBoiler(p - vec2(0.07, 0.1) * u) - cw));
        if (d > cw + 1.0) return col;
        vec2 dir = p.y < uBH.y ? normalize(p - uBH.xy) : vec2(sign(p.x - uBH.x), 0.0);
        vec3 c;
        if (d > 0.0) {
            float t = d / cw;
            vec3 g = uGreen * (0.9 + 0.3 * fbm(p / u * 2.0 + 9.0));
            g = mix(g, vec3(0.06, 0.05, 0.045), 0.6 * smoothstep(0.7, 0.8, fbm(p / u * 4.0 + 3.0)));
            /* lining: a black band with an orange line down its middle */
            float dl = abs(t - 0.56);
            g = mix(g, vec3(0.02), aa((dl - 0.075) * cw));
            g = mix(g, vec3(0.8, 0.4, 0.1), 0.9 * aa((dl - 0.022) * cw));
            float bul = smoothstep(0.75, 1.0, t);
            vec3 n = normalize(vec3(dir * bul * 1.4, 1.0));
            c = paint(g, n, 0.35);
            /* half-round brass beading on the inner edge */
            float v = (t - 0.09) / 0.09;
            if (abs(v) < 1.0) {
                vec3 bn = normalize(vec3(dir * v, sqrt(1.0 - v * v)));
                vec3 b = metal(bn, uBrass * (0.85 + 0.2 * vnoise(p * 0.05)));
                c = mix(c, b, aa((abs(v) - 1.0) * 0.09 * cw));
            }
            c *= 1.0 - 0.6 * aa(abs(d - cw) - 1.0);
        } else {
            vec3 alb = ironAlb(p);
            /* heat: tempered brown round the firehole, soot rising from it */
            vec2 fq = (p - uFD.xy) / u;
            float heat = exp(-length(fq * vec2(0.55, 0.75)) * 1.4);
            alb = mix(alb, vec3(0.13, 0.075, 0.07), 0.6 * heat);
            float soot = smoothstep(1.8, 0.0, abs(fq.x) - 0.6) * smoothstep(-1.5, -0.6, fq.y) * smoothstep(-3.5, -0.8, fq.y);
            alb *= 1.0 - 0.55 * soot * (0.6 + 0.4 * fbm(fq * 3.0));
            /* a satin sheen, broad, toward the top left */
            vec3 n = normalize(vec3(-(p - uBH.xy) / R * 0.12, 1.0));
            c = paint(alb, n, 0.18);
            /* the lap seam across the middle, double-riveted */
            float ys = uBH.y;
            c *= 1.0 + 0.6 * aa(abs(p.y - ys + 1.0) - 0.6) - 0.5 * aa(abs(p.y - ys - 1.0) - 1.0);
            float sp = 0.3 * u, rr = 0.052 * u;
            if (abs(p.y - ys) < 0.3 * u) {
                float row = p.y < ys ? -1.0 : 1.0;
                float off = row > 0.0 ? 0.5 * sp : 0.0;
                vec2 rc = vec2(floor((p.x - uBH.x - off) / sp + 0.5) * sp + uBH.x + off, ys + row * 0.1 * u);
                c = rivetP(p, rc, rr, c, alb * 1.2, 0.3);
            }
            /* the firebox stays below it: rows of little caulked heads */
            if (p.y > ys + 0.3 * u && d < -0.25 * u) {
                float sg = 0.42 * u;
                float ry = floor((p.y - ys - 0.45 * u) / sg + 0.5);
                float off = mod(ry, 2.0) * 0.5 * sg;
                vec2 sc = vec2(floor((p.x - uBH.x - off) / sg + 0.5) * sg + uBH.x + off, ys + 0.45 * u + ry * sg);
                if (sc.y < uBH.w - 0.2 * u) c = rivetP(p, sc, 0.045 * u, c, alb * 1.3, 0.25);
            }
            /* the ring of rivets round the edge */
            float Rr = R - 0.15 * u;
            vec2 rc;
            if (p.y < uBH.y) {
                vec2 v = p - uBH.xy;
                float a = atan(v.x, -v.y), st = 0.3 * u / Rr;
                float ka = floor(a / st + 0.5) * st;
                rc = uBH.xy + Rr * vec2(sin(ka), -cos(ka));
            } else {
                rc = vec2(uBH.x + sign(p.x - uBH.x) * Rr, uBH.y + floor((p.y - uBH.y) / sp + 0.5) * sp);
            }
            c = rivetP(p, rc, 0.058 * u, c, alb * 1.2, 0.3);
            c *= 1.0 - 0.5 * smoothstep(-0.12 * u, 0.0, d);
        }
        return mix(col, c, aa(d - cw));
    }

    /* the floorboards, running away toward the backhead */
    vec3 floorB(vec2 p, vec3 col) {
        if (p.y < uBH.w - 1.0) return col;
        float dy = p.y - uHor.x;
        float pw = dy * uHor.z;
        float wx = (p.x - uHor.y) / pw;
        float pi = floor(wx), fx = fract(wx);
        float z = 60.0 / dy * uS;
        float h = hash(vec2(pi, 3.7) + uSeed);
        float seg = floor(z / 3.0 + h * 7.0);
        float h2 = hash(vec2(pi, seg));
        vec3 c = mix(vec3(0.16, 0.105, 0.065), vec3(0.27, 0.18, 0.11), h2);
        c *= 0.8 + 0.35 * vnoise(vec2(wx * 6.0 + pi * 13.0, z * 2.2));
        c *= 0.85 + 0.2 * vnoise(vec2(wx * 30.0, z * 0.6));
        /* worn grey along the middle where the crew stand, coal dust everywhere */
        c = mix(c, vec3(0.2, 0.19, 0.17), 0.3 * smoothstep(3.5, 0.5, abs(p.x - uFD.x) / uS));
        c *= 1.0 - 0.5 * smoothstep(0.75, 0.95, fbm(vec2(wx, z * 0.4) * 1.5 + 7.0));
        c *= 1.0 - 0.6 * step(0.985, hash(floor(p / 2.0)));
        /* the gaps between boards and at their ends */
        c *= 1.0 - 0.75 * aa((min(fx, 1.0 - fx) * pw) - 0.6);
        c *= 1.0 - 0.6 * smoothstep(0.02, 0.0, abs(fract(z / 3.0 + h * 7.0) - 0.5) - 0.48);
        /* darker under the boiler, lighter toward us */
        c *= 0.75 + 0.6 * smoothstep(uBH.w, uBH.w + 1.0 * uS, p.y);
        c *= 1.0 - 0.6 * smoothstep(0.12 * uS, 0.0, p.y - uBH.w);
        return mix(col, paint(c, normalize(vec3(0.0, -0.6, 0.8)), 0.08), aa(uBH.w - p.y));
    }

    /* the steam main across the top, lagged in canvas and strapped */
    vec3 lag(vec2 p, vec3 col) {
        float y = uLag.x, r = uLag.y;
        col *= mix(1.0, 0.4, smoothstep(1.6, 0.9, (p.y - y - 0.3 * r) / r) * step(y, p.y));
        float v = (p.y - y) / r;
        if (abs(v) > 1.0 + 1.0 / r) return col;
        float vc = clamp(v, -1.0, 1.0);
        vec3 n = normalize(vec3(0.0, vc, sqrt(max(1.0 - vc * vc, 0.0))));
        vec2 q = p / uS;
        float weave = sin(p.x * 1.6) * sin(p.y * 1.6 + p.x * 0.3);
        vec3 alb = vec3(0.6, 0.57, 0.5) * (0.78 + 0.32 * fbm(q * vec2(1.5, 5.0) + 31.0)) * (0.95 + 0.05 * weave);
        alb = mix(alb, vec3(0.22, 0.18, 0.14), 0.55 * smoothstep(0.5, 0.82, fbm(q * 1.3 + 77.0)));
        alb *= mix(1.0, 0.55, smoothstep(0.0, -1.0, v));
        vec3 c = paint(alb, n, 0.02);
        float bx = mod(q.x + 0.4, 1.7) - 0.85;
        float db = (abs(bx) - 0.045) * uS;
        if (db < 1.0) {
            vec3 bn = normalize(vec3(0.0, vc * 1.03, sqrt(max(1.0 - vc * vc, 0.02))));
            c = mix(c * (1.0 - 0.5 * aa(db - 2.0)), metal(bn, vec3(0.36, 0.34, 0.32)), aa(db));
        }
        return mix(col, c, aa((abs(v) - 1.0) * r));
    }

    vec3 pipeBase(float mat, vec2 p) {
        vec2 q = p / uS;
        if (mat < 0.5) {
            vec3 b = uCopper;
            b = mix(b, b * vec3(0.5, 0.42, 0.38), 0.65 * smoothstep(0.4, 0.75, fbm(q * 3.0 + 5.0)));
            b = mix(b, vec3(0.2, 0.4, 0.32), 0.5 * smoothstep(0.72, 0.86, fbm(q * 5.0 + 50.0)));
            return b;
        }
        if (mat < 1.5) return vec3(0.06, 0.057, 0.054) * (0.8 + 0.4 * fbm(q * 3.0));
        return uBrass;
    }
    vec3 pipeShade(float mat, vec3 n, vec2 p) {
        vec3 b = pipeBase(mat, p);
        return mat > 0.5 && mat < 1.5 ? paint(b, n, 0.35) : metal(n, b);
    }

    vec3 pipes(vec2 p, vec3 col) {
        float sh = 1.0;
        for (int i = 0; i < 24; i++) {
            if (i >= uNP) break;
            float r = uPB[i].x;
            float d = sdSeg(p - vec2(0.35, 0.55) * r * 1.3, uPA[i].xy, uPA[i].zw) - r;
            sh = min(sh, mix(1.0, 0.45, smoothstep(r * 0.9, -r * 0.4, d)));
        }
        col *= sh;
        for (int i = 0; i < 24; i++) {
            if (i >= uNP) break;
            vec2 a = uPA[i].xy, b = uPA[i].zw, ba = b - a;
            float h = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
            vec2 v = p - a - ba * h;
            float r = uPB[i].x, d = length(v) - r;
            if (d > 1.0) continue;
            vec2 uu = v / r;
            vec3 n = normalize(vec3(uu, sqrt(max(1.0 - dot(uu, uu), 0.0)) * 1.1));
            col = mix(col, pipeShade(uPB[i].y, n, p), aa(d));
        }
        return col;
    }

    /* bolted flanges: a fat short cylinder across the pipe, nuts either side */
    vec3 flanges(vec2 p, vec3 col) {
        for (int i = 0; i < 16; i++) {
            if (i >= uNF) break;
            vec4 F = uFl[i];
            vec2 dir = vec2(cos(F.z), sin(F.z)), pe = vec2(-dir.y, dir.x);
            float r = F.w;
            vec2 q = p - F.xy;
            float al = dot(q, dir), ac = dot(q, pe);
            if (abs(al) > 1.6 * r || abs(ac) > 2.6 * r) continue;
            float R2 = 1.8 * r, hw = 0.2 * r;
            vec2 qs = q - vec2(0.3, 0.45) * r;
            float ds = sdBox(vec2(dot(qs, dir), dot(qs, pe)), vec2(hw + 0.3 * r, R2), 0.1 * r);
            col *= mix(1.0, 0.5, smoothstep(0.5 * r, -0.2 * r, ds));
            /* the nuts, behind and in front of the flange */
            for (int s = 0; s < 2; s++) {
                float sa = s == 0 ? -1.0 : 1.0;
                float dn = sdBox(vec2(abs(al) - hw - 0.13 * r, ac - sa * 1.35 * r), vec2(0.14 * r, 0.24 * r), 0.04 * r);
                if (dn < 1.0) {
                    float vv = clamp((ac - sa * 1.35 * r) / (0.24 * r), -1.0, 1.0);
                    vec3 n = normalize(vec3(pe * vv * 0.8, 1.0));
                    n = abs(vv) < 0.45 ? vec3(0.0, 0.0, 1.0) : n;
                    col = mix(col, metal(n, vec3(0.3, 0.29, 0.28)), aa(dn));
                }
            }
            float d = sdBox(vec2(al, ac), vec2(hw, R2), 0.06 * r);
            if (d < 1.0) {
                float v = clamp(ac / R2, -1.0, 1.0);
                vec3 n = normalize(vec3(pe * v, sqrt(1.0 - v * v)));
                vec3 base = vec3(0.5, 0.36, 0.2) * (0.8 + 0.3 * vnoise(p * 0.2));
                vec3 c = metal(n, base) * (1.0 - 0.35 * smoothstep(hw - 1.5, hw, abs(al)));
                col = mix(col, c, aa(d));
            }
        }
        return col;
    }

    /* valve handwheels, red paint worn to the iron where hands grip */
    vec3 wheels(vec2 p, vec3 col) {
        for (int i = 0; i < 6; i++) {
            if (i >= uNW) break;
            vec4 Wh = uWh[i];
            float R = Wh.z;
            vec2 q = p - Wh.xy;
            float d = length(q);
            if (d > 1.3 * R) continue;
            /* the valve bonnet behind */
            float db = d - 0.42 * R;
            col *= mix(1.0, 0.5, smoothstep(0.15 * R, -0.05 * R, length(q - vec2(0.08, 0.12) * R) - 0.42 * R));
            if (db < 1.0) col = mix(col, dome(q / (0.42 * R) * 0.75, uBrass * 0.85), aa(db));
            float ph = hash(Wh.xy) * 6.28;
            float sp = 6.28318 / 5.0;
            /* the wheel and its shadow */
            for (int pass = 0; pass < 2; pass++) {
                vec2 qq = pass == 0 ? q - vec2(0.1, 0.15) * R : q;
                float dd = length(qq);
                float a = atan(qq.y, qq.x) - ph;
                float ka = floor(a / sp + 0.5) * sp + ph;
                vec2 sd = vec2(cos(ka), sin(ka)), spp = vec2(-sd.y, sd.x);
                float al = dot(qq, sd), ac = dot(qq, spp);
                float dr = abs(dd - 0.84 * R) - 0.12 * R;
                float dsp = max(abs(ac) - 0.055 * R * (1.25 - 0.5 * al / R), abs(al - 0.5 * R) - 0.36 * R);
                float dbs = dd - 0.17 * R;
                float dw = min(dr, min(dsp, dbs));
                if (pass == 0) { col *= mix(1.0, 0.5, smoothstep(0.06 * R, -0.03 * R, dw)); continue; }
                if (dw > 1.0) continue;
                vec3 n;
                if (dr <= min(dsp, dbs)) {
                    float v = clamp((dd - 0.84 * R) / (0.12 * R), -1.0, 1.0);
                    n = normalize(vec3(qq / max(dd, 1e-3) * v, sqrt(1.0 - v * v)));
                } else if (dsp <= dbs) {
                    float v = clamp(ac / (0.06 * R), -1.0, 1.0);
                    n = normalize(vec3(spp * v * 0.8, 1.0));
                } else {
                    vec2 sp2 = qq / (0.17 * R);
                    n = normalize(vec3(sp2 * 0.8, sqrt(max(1.0 - dot(sp2, sp2), 0.0)) + 0.2));
                }
                vec3 alb = uRed * vec3(0.72, 0.6, 0.6) * (0.85 + 0.3 * vnoise(qq * 0.3 + 5.0));
                float wear = smoothstep(0.5, 0.8, vnoise(vec2(atan(qq.y, qq.x) * 2.5, 3.0) + Wh.xy)) * step(dr, min(dsp, dbs));
                vec3 c = mix(paint(alb, n, 0.3), metal(n, vec3(0.42, 0.4, 0.38)), wear * 0.85);
                if (dbs < min(dr, dsp)) c = metal(n, vec3(0.38, 0.36, 0.34));
                col = mix(col, c, aa(dw));
            }
        }
        return col;
    }
)
GLSL(
    /* the bezel's cross-section, by radius / face radius: a lip sloping
     * into the face, a rounded rim, a flat flange, a rounded outer edge */
    float bezH(float t) {
        if (t < 1.035) return 0.5 * (t - 1.0) / 0.035;
        if (t < 1.13) { float u = (t - 1.035) / 0.095; return mix(0.5, 0.32, u) + 0.55 * sin(u * 3.14159); }
        if (t < 1.195) return 0.32 - 0.6 * (t - 1.13);
        float u = clamp((t - 1.195) / 0.025, 0.0, 1.0);
        return 0.281 * cos(u * 1.5708);
    }

    vec3 screw(vec2 sp, float ang, vec3 under, float dpx) {
        vec3 c = dome(sp * 0.85, uBrass * 0.95);
        vec2 rs = vec2(cos(ang) * sp.x + sin(ang) * sp.y, -sin(ang) * sp.x + cos(ang) * sp.y);
        float slot = smoothstep(0.17, 0.11, abs(rs.y)) * step(abs(rs.x), 0.9);
        c = mix(c, c * 0.15, slot);
        c += vec3(0.25, 0.18, 0.08) * smoothstep(0.17, 0.12, abs(rs.y - 0.08)) * step(0.0, rs.y) * slot;
        c *= 1.0 - 0.4 * smoothstep(0.75, 1.0, length(sp));
        return mix(under, c, aa(dpx));
    }

    vec3 face(int i, vec2 q, float d, float r) {
        float t = d / r;
        vec2 u = q / r;
        bool clock = uG[i].w > 0.5;
        /* white enamel, a little crazed and yellowed at the edges */
        vec3 c = (clock ? vec3(0.93, 0.91, 0.86) : vec3(0.92, 0.89, 0.82)) * (1.03 - 0.1 * t * t);
        float n = fbm(u * 2.5 + float(i) * 7.13 + uSeed);
        c *= 0.95 + 0.07 * n;
        c = mix(c, vec3(0.8, 0.69, 0.5), 0.3 * smoothstep(0.55, 0.8, fbm(u * 1.7 + 19.0 + float(i))) * smoothstep(0.4, 1.0, t));
        c = mix(c, vec3(0.6, 0.5, 0.36), 0.2 * smoothstep(0.74, 0.8, fbm(u * 6.0 + 3.0 + float(i) * 2.0)));
        float craze = abs(vnoise(u * 9.0 + float(i)) - 0.5);
        c *= 1.0 - 0.08 * smoothstep(0.02, 0.0, craze) * smoothstep(0.5, 0.95, t);
        c *= 0.985 + 0.03 * hash(floor(q));

        vec3 ink = vec3(0.09, 0.075, 0.065);
        float th = atan(q.x, -q.y);
        float lw = max(0.0045 * r, 0.9);
        if (!clock) {
            vec4 A = uA[i];
            float span = A.y - A.x;
            float n2 = A.z * A.w;
            float ro = 0.88 * r;
            vec4 Z = uZ[i];
            float band = max(0.86 * r - d, d - 0.94 * r);
            for (int z = 0; z < 2; z++) {
                vec2 zz = z == 0 ? Z.xy : Z.zw;
                if (zz.x < 0.0) continue;
                float za = A.x + span * zz.x, zb = A.x + span * zz.y;
                float dz = max(band, max(za - th, th - zb) * d);
                c = mix(c, z == 0 ? uRed : uAmber, 0.92 * aa(dz));
            }
            float da = max(abs(d - ro) - lw * 0.7, max(A.x - th, th - A.y) * d - lw);
            float cov = aa(da);
            float k = clamp(floor((th - A.x) / span * n2 + 0.5), 0.0, n2);
            float tk = A.x + span * k / n2;
            vec2 dir = vec2(sin(tk), -cos(tk));
            vec2 perp = vec2(-dir.y, dir.x);
            float along = dot(q, dir), across = dot(q, perp);
            float km = mod(k, A.w);
            bool major = km < 0.5;
            bool mid = !major && mod(A.w, 2.0) < 0.5 && abs(km - A.w * 0.5) < 0.5;
            float len = (major ? 0.13 : mid ? 0.095 : 0.065) * r;
            float wid = major ? max(0.011 * r, 1.4) : max(0.005 * r, 0.85);
            float dt = max(abs(across) - wid, abs(along - (ro - len * 0.5)) - len * 0.5);
            cov = max(cov, aa(dt));
            c = mix(c, ink, cov * 0.93);
            vec2 pc = 0.8 * r * vec2(sin(A.x - 0.07), -cos(A.x - 0.07));
            float pr = 0.022 * r;
            c *= mix(1.0, 0.6, smoothstep(pr * 1.5, pr * 0.5, length(q - pc - vec2(0.4, 0.6) * pr)));
            float dpin = length(q - pc);
            if (dpin < pr + 1.0) c = mix(c, dome((q - pc) / pr, uBrass), aa(dpin - pr));
        } else {
            /* a railway minute track and bold hour bars */
            float dr = min(abs(d - 0.925 * r), abs(d - 0.855 * r)) - lw * 0.6;
            float cov = aa(max(dr, d - 0.94 * r));
            float k = floor(th / 0.10472 + 0.5);
            float tk = k * 0.10472;
            vec2 dir = vec2(sin(tk), -cos(tk));
            vec2 perp = vec2(-dir.y, dir.x);
            float along = dot(q, dir), across = dot(q, perp);
            bool hour = mod(k + 60.0, 5.0) < 0.5;
            float d1 = max(abs(across) - max(0.0045 * r, 0.85), abs(along - 0.89 * r) - 0.035 * r);
            float d2 = max(abs(across) - 0.014 * r, abs(along - 0.89 * r) - 0.035 * r);
            cov = max(cov, aa(hour ? d2 : d1));
            c = mix(c, ink, cov * 0.93);
        }
        vec2 lo = normalize(L.xy);
        float sh = smoothstep(0.93 * r, 1.04 * r, length(q + lo * 0.07 * r));
        c *= 1.0 - 0.45 * sh;
        c *= 1.0 - 0.25 * smoothstep(0.9, 1.0, t);
        return c;
    }

    vec3 bezel(int i, vec2 q, float d, float r) {
        float t = d / r;
        float e = 0.0015;
        float slope = (bezH(t + e) - bezH(t - e)) / (2.0 * e) * 0.06;
        vec2 dir = q / max(d, 1e-3);
        vec3 n = normalize(vec3(dir * -slope, 1.0));
        float br = 1.0 + 0.09 * (vnoise(vec2(d * 0.6, float(i) * 13.0)) - 0.5) + 0.06 * (vnoise(vec2(d * 2.7, 5.0 + float(i))) - 0.5);
        vec3 base = mix(uBrass, uBrass * vec3(0.68, 0.6, 0.48), 0.55 * smoothstep(0.45, 0.8, fbm(q / r * 4.0 + float(i) * 3.0 + uSeed)));
        vec3 c = metal(n, base) * br;
        c *= 1.0 - 0.45 * exp(-pow((t - 1.035) / 0.005, 2.0)) - 0.3 * exp(-pow((t - 1.13) / 0.006, 2.0));
        float sr = 0.03 * r;
        for (int s = 0; s < 4; s++) {
            float a = 0.7854 + 1.5708 * float(s);
            vec2 sc = 1.165 * r * vec2(sin(a), -cos(a));
            float ds = length(q - sc);
            c *= mix(1.0, 0.55, smoothstep(sr * 1.4, sr * 0.6, length(q - sc - vec2(0.3, 0.45) * sr)));
            if (ds < sr + 1.0) c = screw((q - sc) / sr, hash(vec2(float(i), float(s)) + uSeed) * 3.14, c, ds - sr);
        }
        return c;
    }

    /* the clock's mahogany surround, turned and french-polished */
    vec3 mahogany(vec2 q, float d, float r) {
        float t = (d - 1.22 * r) / (0.3 * r);
        float a = atan(q.y, q.x);
        float grain = sin(a * 3.0 + d / r * 40.0 + 4.0 * fbm(vec2(a * 2.0, d / r * 3.0) + uSeed));
        vec3 alb = mix(vec3(0.16, 0.05, 0.03), vec3(0.3, 0.1, 0.05), 0.5 + 0.5 * grain);
        alb *= 0.85 + 0.25 * vnoise(vec2(a * 40.0, d * 0.4));
        float prof = sin(t * 3.14159);
        vec2 dir = q / max(d, 1e-3);
        vec3 n = normalize(vec3(dir * -cos(t * 3.14159) * 0.9, 1.0));
        return paint(alb, n, 0.55) * (0.8 + 0.2 * prof);
    }

    /* the water gauges' brass: two protector rods, the cocks top and bottom with their levers */
    vec3 wgFrame(vec2 p, vec3 col, vec4 G, float side) {
        float x = G.x, y0 = G.y, y1 = G.z, tr = G.w;
        vec2 q = p - vec2(x, 0.0);
        if (abs(q.x) > 6.0 * tr || p.y < y0 - 3.0 * tr || p.y > y1 + 3.0 * tr) return col;
        /* shadow of the lot */
        vec2 qs = q - vec2(0.4, 0.6) * tr;
        float dsh = min(sdBox(vec2(qs.x, p.y - 0.6 * tr - (y0 + y1) * 0.5), vec2(1.8 * tr, (y1 - y0) * 0.5 + 1.6 * tr), 0.5 * tr),
                        sdSeg(vec2(qs.x, p.y - 0.6 * tr), vec2(side * 1.9 * tr, y0 - 0.85 * tr), vec2(side * 4.4 * tr, y0 - 0.85 * tr)) - 0.3 * tr);
        col *= mix(1.0, 0.45, smoothstep(0.8 * tr, -0.2 * tr, dsh));
        /* the dark slot behind the glass */
        float dsl = sdBox(vec2(q.x, p.y - (y0 + y1) * 0.5), vec2(1.3 * tr, (y1 - y0) * 0.5), 0.0);
        col = mix(col, vec3(0.02, 0.018, 0.016), aa(dsl));
        /* protector rods */
        float rx = abs(q.x) - 1.55 * tr;
        if (abs(rx) < 0.3 * tr + 1.0 && p.y > y0 && p.y < y1) {
            float v = clamp(rx / (0.25 * tr), -1.0, 1.0);
            vec3 n = normalize(vec3(sign(q.x) * v, 0.0, sqrt(1.0 - v * v)));
            col = mix(col, metal(n, uBrass * 0.9), aa((abs(rx) - 0.25 * tr)));
        }
        /* the two cocks */
        for (int k = 0; k < 2; k++) {
            float yc = k == 0 ? y0 - 0.85 * tr : y1 + 0.85 * tr;
            vec2 c = vec2(x, yc);
            /* lever: tapered bar outward, a ball on its end */
            vec2 la = c + vec2(side * 1.5 * tr, 0.0), lb = c + vec2(side * 4.5 * tr, 0.0);
            float dl = min(sdSeg(p, la, lb) - 0.2 * tr, length(p - lb) - 0.42 * tr);
            if (dl < 1.0) {
                float vy = clamp((p.y - yc) / (0.42 * tr), -1.0, 1.0);
                vec3 n = length(p - lb) < 0.43 * tr ? normalize(vec3((p - lb) / (0.42 * tr), 0.6)) : normalize(vec3(0.0, vy, sqrt(1.0 - vy * vy) + 0.1));
                col = mix(col, metal(n, vec3(0.32, 0.31, 0.3)), aa(dl));
            }
            float db = sdBox(p - c, vec2(1.9 * tr, 1.0 * tr), 0.4 * tr);
            if (db < 1.0) {
                vec2 lq = (p - c) / vec2(1.9 * tr, 1.0 * tr);
                vec3 n = normalize(vec3(lq * vec2(0.5, 0.9) * smoothstep(0.4, 1.0, abs(lq)), 1.0));
                vec3 c2 = metal(n, uBrass * (0.8 + 0.25 * vnoise(p * 0.3)));
                /* the gland nut where the glass enters */
                float gy = k == 0 ? y0 - 0.05 * tr : y1 + 0.05 * tr;
                float dn = sdBox(p - vec2(x, gy), vec2(1.35 * tr, 0.32 * tr), 0.08 * tr);
                c2 = mix(c2, metal(vec3(0.0, 0.0, 1.0), uBrass) * 0.9, aa(dn));
                c2 *= 1.0 - 0.5 * aa(abs(dn) - 0.6);
                col = mix(col, c2, aa(db));
            }
        }
        return col;
    }

    /* the firehole: a heavy cast ring, and the butterfly doors left just apart */
    vec3 firedoor(vec2 p, vec3 col) {
        vec2 c = uFD.xy, hs = uFD.zw;
        float u = uS;
        vec2 q = p - c;
        float ro = 0.24 * u;
        float dOut = sdBox(q, hs + ro, 0.42 * u);
        col *= mix(1.0, 0.4, smoothstep(0.25 * u, -0.05 * u, sdBox(q - vec2(0.06, 0.1) * u, hs + ro, 0.42 * u)));
        if (dOut > 1.0) return col;
        float dIn = sdBox(q, hs, 0.3 * u);
        vec2 e = vec2(1.0, 0.0);
        vec3 c2;
        if (dIn > 0.0) {
            float t = clamp(dIn / ro, 0.0, 1.0);
            vec2 g = normalize(vec2(sdBox(q + e.xy, hs, 0.3 * u) - sdBox(q - e.xy, hs, 0.3 * u),
                                    sdBox(q + e.yx, hs, 0.3 * u) - sdBox(q - e.yx, hs, 0.3 * u)) + 1e-6);
            vec3 n = normalize(vec3(g * -cos(t * 3.14159) * 1.1, 1.0));
            vec3 alb = vec3(0.075, 0.07, 0.066) * (0.75 + 0.5 * fbm(q / u * 5.0 + 3.0));
            alb = mix(alb, vec3(0.16, 0.08, 0.05), 0.4 * smoothstep(0.55, 0.8, fbm(q / u * 3.0 + 13.0)));
            /* the bottom lip polished by the shovel */
            float lip = smoothstep(0.3 * hs.y, hs.y, q.y) * smoothstep(hs.x * 0.9, hs.x * 0.3, abs(q.x)) * smoothstep(0.7, 0.2, t);
            c2 = mix(paint(alb, n, 0.25), metal(n, vec3(0.4, 0.38, 0.36)), lip * 0.85);
            /* studs round the ring */
            float a = atan(q.y / (hs.y + 0.5 * ro), q.x / (hs.x + 0.5 * ro));
            float st = 6.28318 / 14.0;
            float ka = (floor(a / st) + 0.5) * st;
            vec2 sc = vec2(cos(ka), sin(ka));
            sc = sc / max(abs(sc.x) / (hs.x + 0.5 * ro), abs(sc.y) / (hs.y + 0.5 * ro)) * 0.97;
            c2 = rivetP(q, sc, 0.06 * u, c2, vec3(0.12, 0.11, 0.1), 0.4);
        } else {
            float gap = 0.022 * u;
            float side = q.x < 0.0 ? -1.0 : 1.0;
            vec2 dc = vec2(abs(q.x) - hs.x * 0.5, q.y);
            vec3 n = normalize(vec3(side * dc.x / hs.x * 0.35, dc.y / hs.y * 0.3, 1.0));
            /* a raised rim round each leaf */
            float leaf = max(sdBox(dc, vec2(hs.x * 0.5 - gap * 0.5, hs.y), 0.25 * u), dIn);
            float rim = smoothstep(-0.09 * u, -0.06 * u, leaf);
            vec2 lg = normalize(vec2(side * dc.x, dc.y * 1.6) + 1e-4);
            n = normalize(n + vec3(side * lg.x, lg.y, 0.0) * 0.5 * rim * (1.0 - smoothstep(-0.03 * u, 0.0, leaf)));
            vec3 alb = vec3(0.12, 0.11, 0.1) * (0.75 + 0.5 * fbm(q / u * 4.0 + 7.0));
            /* heat-tempered and sooted, worst near the gap */
            alb = mix(alb, vec3(0.16, 0.09, 0.07), 0.5 * smoothstep(0.6, 0.0, abs(q.x) / hs.x));
            alb = mix(alb, vec3(0.08, 0.08, 0.12), 0.25 * smoothstep(0.45, 0.75, fbm(q / u * 2.5 + 5.0)));
            c2 = paint(alb, n, 0.3);
            /* the air holes, dark until the fire lights them */
            float dsl = fhSlots(q, hs, u);
            c2 = mix(c2, c2 * 1.5, aa(abs(dsl - 2.0) - 1.0));
            c2 = mix(c2, vec3(0.015, 0.005, 0.0), aa(dsl));
            /* hinge knuckles on the outer edges */
            for (int k = 0; k < 2; k++) {
                vec2 hc = vec2(side * (hs.x - 0.05 * u), (k == 0 ? -0.55 : 0.55) * hs.y);
                float dh = sdBox(q - hc, vec2(0.06 * u, 0.13 * u), 0.05 * u);
                if (dh < 1.0) {
                    float v = clamp((q.x - hc.x) / (0.06 * u), -1.0, 1.0);
                    c2 = mix(c2, metal(normalize(vec3(v, 0.0, sqrt(1.0 - v * v))), vec3(0.22, 0.21, 0.2)), aa(dh));
                }
            }
            /* the handles: worn bright by the crew */
            vec2 hc = vec2(side * (gap + 0.17 * u), -0.05 * hs.y);
            float dh = sdBox(q - hc, vec2(0.035 * u, 0.24 * u), 0.03 * u);
            c2 *= mix(1.0, 0.45, smoothstep(0.05 * u, -0.01 * u, sdBox(q - hc - vec2(0.04, 0.06) * u, vec2(0.035 * u, 0.24 * u), 0.03 * u)));
            if (dh < 1.0) {
                float v = clamp((q.x - hc.x) / (0.035 * u), -1.0, 1.0);
                c2 = mix(c2, metal(normalize(vec3(v, 0.0, sqrt(1.0 - v * v))), vec3(0.55, 0.52, 0.48)), aa(dh));
            }
            /* the gap between the leaves and round them: dark (the fire is added each frame) */
            float dark = max(aa(abs(q.x) - gap), aa(-(dIn + 0.03 * u)));
            c2 = mix(c2, vec3(0.02, 0.006, 0.0), dark);
        }
        return mix(col, c2, aa(dOut));
    }

    /* the builder's plate: brass, beaded, the ground painted red, the drums sunk in */
    vec3 plaque(vec2 p, vec3 col) {
        vec4 P = uPq;
        float h = P.w - P.y;
        float cr = 0.12 * h;
        float dq = sdRect(p, P, cr);
        col *= mix(1.0, 0.4, smoothstep(0.08 * h, -0.02 * h, sdRect(p - vec2(0.025, 0.035) * h, P, cr)));
        if (dq > 1.0) return col;
        vec2 e = vec2(1.0, 0.0);
        vec2 g = normalize(vec2(sdRect(p + e.xy, P, cr) - sdRect(p - e.xy, P, cr), sdRect(p + e.yx, P, cr) - sdRect(p - e.yx, P, cr)) + 1e-6);
        float br = 1.0 + 0.1 * (vnoise(vec2(p.x * 0.01, p.y * 1.7)) - 0.5);
        vec3 base = mix(uBrass, uBrass * vec3(0.72, 0.62, 0.5), 0.4 * smoothstep(0.45, 0.8, fbm(p / h * 1.3 + 5.0 + uSeed)));
        /* the ground */
        vec3 red = uRed * vec3(0.62, 0.5, 0.5) * (0.85 + 0.25 * fbm(p / h * 6.0 + 11.0));
        red *= 0.92 + 0.12 * hash(floor(p / 1.5));
        vec3 c = paint(red, vec3(0.0, 0.0, 1.0), 0.15);
        /* the beaded border */
        float bw = 0.085 * h;
        float v = (-dq - bw * 0.5) / (bw * 0.5);
        if (v < 1.2) {
            float vc = clamp(v, -1.0, 1.0);
            vec3 n = normalize(vec3(g * vc * 1.2, sqrt(1.0 - vc * vc) + 0.05));
            vec3 b = metal(n, base) * br;
            c *= 1.0 - 0.6 * aa((v - 1.0) * bw * 0.5 - 2.0) * step(1.0, v);
            c = mix(c, b, aa((v - 1.0) * bw * 0.5));
        }
        /* the raised rule */
        float dr = max(abs(p.y - uRule.y) - uRule.w, max(uRule.x - p.x, p.x - uRule.z));
        c = mix(c, c * 0.4, aa(max(abs(p.y - uRule.y - uRule.w - 1.0) - 1.0, max(uRule.x - p.x, p.x - uRule.z))));
        c = mix(c, metal(normalize(vec3(0.0, clamp((p.y - uRule.y) / uRule.w, -1.0, 1.0) * 0.8, 1.0)), base), aa(dr));
        /* the windows, sunk in with a brass bevel */
        for (int w = 0; w < 2; w++) {
            float rr = 0.12 * (uWin[w].w - uWin[w].y);
            float dw = sdRect(p, uWin[w], rr);
            float fw = 0.12 * (uWin[w].w - uWin[w].y);
            if (dw < fw) {
                vec2 gw = normalize(vec2(sdRect(p + e.xy, uWin[w], rr) - sdRect(p - e.xy, uWin[w], rr), sdRect(p + e.yx, uWin[w], rr) - sdRect(p - e.yx, uWin[w], rr)) + 1e-6);
                vec3 fn = normalize(vec3(-gw * 0.9, 1.0));
                c = mix(c, metal(fn, base * 0.9) * br, aa(-dw + fw - 0.5));
                c = mix(c, vec3(0.03, 0.022, 0.016), aa(dw));
            }
        }
        /* a bolt in each corner */
        float sr = 0.05 * h;
        for (int s = 0; s < 4; s++) {
            vec2 sc = vec2(s % 2 == 0 ? P.x + 0.042 * h + bw * 0.5 : P.z - 0.042 * h - bw * 0.5, s < 2 ? P.y + 0.042 * h + bw * 0.5 : P.w - 0.042 * h - bw * 0.5);
            float ds = length(p - sc);
            c *= mix(1.0, 0.55, smoothstep(sr * 1.4, sr * 0.6, length(p - sc - vec2(0.3, 0.45) * sr)));
            if (ds < sr + 1.0) c = mix(c, dome((p - sc) / sr, base * 0.9), aa(ds - sr));
        }
        return mix(col, c, aa(dq));
    }

    /* little brass tags, riveted on */
    vec3 tags(vec2 p, vec3 col) {
        for (int i = 0; i < 2; i++) {
            vec4 T = uTag[i];
            float h = T.w - T.y;
            float d = sdRect(p, T, 0.15 * h);
            col *= mix(1.0, 0.5, smoothstep(0.2 * h, -0.05 * h, sdRect(p - vec2(0.06, 0.1) * h, T, 0.15 * h)));
            if (d > 1.0) continue;
            vec2 e = vec2(1.0, 0.0);
            vec2 g = normalize(vec2(sdRect(p + e.xy, T, 0.15 * h) - sdRect(p - e.xy, T, 0.15 * h), sdRect(p + e.yx, T, 0.15 * h) - sdRect(p - e.yx, T, 0.15 * h)) + 1e-6);
            vec3 n = normalize(vec3(g * 0.8 * smoothstep(-0.12 * h, 0.0, d), 1.0));
            vec3 c = metal(n, uBrass * (0.75 + 0.3 * fbm(p / h * 2.0 + float(i) * 9.0)));
            for (int s = 0; s < 2; s++) {
                vec2 rc = vec2(s == 0 ? T.x + 0.28 * h : T.z - 0.28 * h, (T.y + T.w) * 0.5);
                float dr = length(p - rc) - 0.11 * h;
                if (dr < 1.0) c = mix(c, dome((p - rc) / (0.11 * h), uBrass * 0.8), aa(dr));
            }
            col = mix(col, c, aa(d));
        }
        return col;
    }

    /* the governor's mounting: the throttle valve on the steam pipe, the
     * red gearbox on it, the spindle's frame */
    vec3 govBase(vec2 p, vec3 col) {
        float gx = uGov.x, py = uGov.y, L = uGov.z, gy = uGov.w;
        float vy = uGovB.x, vr = uGovB.y, gr = uGovB.z;
        if (abs(p.x - gx) > 1.4 * L || p.y < py - 0.6 * L || p.y > vy + 1.4 * vr) return col;
        vec2 so = vec2(0.06, 0.09) * uS;
        /* the valve body: a brass globe with a neck up to the gearbox */
        float dv = sdBox(p - vec2(gx, vy), vec2(1.5 * vr, 0.95 * vr), 0.6 * vr);
        float dn = sdBox(p - vec2(gx, (gy + vy) * 0.5), vec2(0.32 * vr, (vy - gy) * 0.5), 0.0);
        col *= mix(1.0, 0.45, smoothstep(0.3 * vr, -0.1 * vr, min(sdBox(p - so - vec2(gx, vy), vec2(1.5 * vr, 0.95 * vr), 0.6 * vr), sdBox(p - so - vec2(gx, (gy + vy) * 0.5), vec2(0.32 * vr, (vy - gy) * 0.5), 0.0))));
        if (dn < 1.0) {
            float v = clamp((p.x - gx) / (0.32 * vr), -1.0, 1.0);
            col = mix(col, metal(normalize(vec3(v, 0.0, sqrt(1.0 - v * v))), uBrass * 0.85), aa(dn));
        }
        if (dv < 1.0) {
            vec2 lq = (p - vec2(gx, vy)) / vec2(1.5 * vr, 0.95 * vr);
            vec3 n = normalize(vec3(lq.x * smoothstep(0.5, 1.0, abs(lq.x)) * 1.2, lq.y * 0.9, 1.0));
            vec3 c = metal(n, uBrass * (0.78 + 0.25 * fbm(lq * 2.0 + 3.0)));
            /* the bonnet flange across its top, the joints to the pipe */
            c *= 1.0 - 0.45 * aa(abs(abs(lq.x) - 0.72) * 1.5 * vr - 1.0);
            col = mix(col, c, aa(dv));
        }
        /* the gearbox: cast iron, oxide red */
        vec2 gq = p - vec2(gx, gy);
        float dg = sdBox(gq, vec2(1.3 * gr, 1.0 * gr), 0.3 * gr);
        col *= mix(1.0, 0.45, smoothstep(0.3 * gr, -0.1 * gr, sdBox(gq - so, vec2(1.3 * gr, 1.0 * gr), 0.3 * gr)));
        if (dg < 1.0) {
            vec2 lq = gq / vec2(1.3 * gr, 1.0 * gr);
            vec3 n = normalize(vec3(lq * smoothstep(0.55, 1.0, abs(lq)) * 1.2, 1.0));
            vec3 alb = vec3(0.3, 0.085, 0.05) * (0.7 + 0.5 * fbm(gq / gr * 2.0 + 21.0));
            col = mix(col, paint(alb, n, 0.3), aa(dg));
            /* the gland on top where the spindle comes out */
            float dgl = sdBox(gq + vec2(0.0, 1.05 * gr), vec2(0.28 * gr, 0.18 * gr), 0.05 * gr);
            if (dgl < 1.0) col = mix(col, metal(normalize(vec3(clamp(gq.x / (0.28 * gr), -1.0, 1.0), 0.0, 0.8)), uBrass), aa(dgl));
        }
        return col;
    }

    /* the safety valve: a brass pop valve on its branch */
    vec3 safety(vec2 p, vec3 col) {
        float x = uSV.x, y = uSV.y, s = uSV.z;
        vec2 q = p - vec2(x, y);
        if (abs(q.x) > 0.8 * s || q.y < -1.4 * s || q.y > 0.4 * s) return col;
        vec2 so = vec2(0.05, 0.08) * uS;
        float d1 = sdBox(q - vec2(0.0, -0.06 * s), vec2(0.36 * s, 0.07 * s), 0.02 * s);
        float d2 = sdBox(q - vec2(0.0, -0.45 * s), vec2(0.22 * s, 0.34 * s), 0.05 * s);
        float d3 = sdBox(q - vec2(0.0, -0.86 * s), vec2(0.3 * s, 0.08 * s), 0.03 * s);
        float d4 = sdBox(q - vec2(0.0, -1.0 * s), vec2(0.15 * s, 0.08 * s), 0.04 * s);
        float dall = min(min(d1, d2), min(d3, d4));
        vec2 qs = q - so;
        float dsh = min(min(sdBox(qs - vec2(0.0, -0.06 * s), vec2(0.36 * s, 0.07 * s), 0.02 * s), sdBox(qs - vec2(0.0, -0.45 * s), vec2(0.22 * s, 0.34 * s), 0.05 * s)),
                        min(sdBox(qs - vec2(0.0, -0.86 * s), vec2(0.3 * s, 0.08 * s), 0.03 * s), sdBox(qs - vec2(0.0, -1.0 * s), vec2(0.15 * s, 0.08 * s), 0.04 * s)));
        col *= mix(1.0, 0.45, smoothstep(0.08 * s, -0.03 * s, dsh));
        if (dall > 1.0) return col;
        float hw = d1 == dall ? 0.36 : d2 == dall ? 0.22 : d3 == dall ? 0.3 : 0.15;
        float v = clamp(q.x / (hw * s), -1.0, 1.0);
        vec3 n = normalize(vec3(v, 0.0, sqrt(1.0 - v * v)));
        vec3 base = uBrass * (0.78 + 0.3 * fbm(q / s * 3.0 + 4.0));
        vec3 c = metal(n, base);
        /* the joints between the parts */
        c *= 1.0 - 0.5 * aa(min(min(abs(d1), abs(d2)), min(abs(d3), abs(d4))) - 0.7) * step(-2.0, -dall);
        return mix(col, c, aa(dall));
    }

    /* the paraffin lamp: a tin reflector, a brass font, a glass chimney */
    vec3 lampBody(vec2 p, vec3 col) {
        float s = uLamp.z, side = uLamp.w;
        vec2 q = p - uLamp.xy;
        if (abs(q.x) > 1.6 * s || q.y < -1.3 * s || q.y > 1.0 * s) return col;
        vec2 so = vec2(0.05, 0.08) * uS;
        /* the bracket from the wall */
        float db = min(sdBox(q - vec2(side * 0.55 * s, 0.62 * s), vec2(0.55 * s, 0.035 * s), 0.01 * s),
                       sdBox(q - vec2(side * 1.1 * s, 0.45 * s), vec2(0.06 * s, 0.32 * s), 0.03 * s));
        float dbs = min(sdBox(q - so - vec2(side * 0.55 * s, 0.62 * s), vec2(0.55 * s, 0.035 * s), 0.01 * s),
                        sdBox(q - so - vec2(side * 1.1 * s, 0.45 * s), vec2(0.06 * s, 0.32 * s), 0.03 * s));
        col *= mix(1.0, 0.5, smoothstep(0.05 * s, -0.02 * s, dbs));
        if (db < 1.0) col = mix(col, paint(vec3(0.05), normalize(vec3(0.0, clamp((q.y - 0.62 * s) / (0.035 * s), -1.0, 1.0), 1.0)), 0.4), aa(db));
        /* reflector: a dished tin disc behind */
        vec2 rq = q - vec2(0.0, -0.15 * s);
        float dr = length(rq) - 0.6 * s;
        col *= mix(1.0, 0.5, smoothstep(0.1 * s, -0.05 * s, length(rq - so) - 0.6 * s));
        if (dr < 1.0) {
            vec2 sp = rq / (0.6 * s);
            vec3 n = normalize(vec3(-sp * 0.5, 1.0));
            vec3 c = metal(n, vec3(0.5, 0.48, 0.44) * (0.7 + 0.4 * fbm(sp * 3.0 + 2.0)));
            c *= 1.0 - 0.5 * smoothstep(0.85, 1.0, length(sp));
            col = mix(col, c, aa(dr));
        }
        /* the chimney: glass, bulging round the flame */
        float cy = q.y + 0.2 * s;
        float hwc = 0.12 * s + 0.08 * s * exp(-pow(cy / (0.25 * s), 2.0));
        float dc = max(abs(q.x) - hwc, abs(q.y + 0.25 * s) - 0.5 * s);
        if (dc < 1.0) {
            float v = q.x / hwc;
            vec3 gc = col * vec3(0.85, 0.82, 0.75) + vec3(0.06, 0.055, 0.05);
            gc += vec3(1.0, 0.95, 0.85) * 0.35 * smoothstep(0.25, 0.0, abs(v + 0.55)) + vec3(1.0, 0.95, 0.85) * 0.15 * smoothstep(0.12, 0.0, abs(v - 0.7));
            gc *= 1.0 - 0.4 * smoothstep(0.75, 1.0, abs(v));
            col = mix(col, gc, aa(dc));
        }
        /* the cowl on top */
        float dw = sdBox(q - vec2(0.0, -0.82 * s), vec2(0.17 * s, 0.08 * s), 0.03 * s);
        float dw2 = sdBox(q - vec2(0.0, -0.95 * s), vec2(0.1 * s, 0.06 * s), 0.04 * s);
        if (min(dw, dw2) < 1.0) col = mix(col, metal(normalize(vec3(clamp(q.x / (0.17 * s), -1.0, 1.0), 0.0, 0.9)), vec3(0.2, 0.19, 0.18)), aa(min(dw, dw2)));
        /* the font and burner */
        float df = length((q - vec2(0.0, 0.5 * s)) / vec2(0.36, 0.2)) - s;
        float dbr = sdBox(q - vec2(0.0, 0.3 * s), vec2(0.15 * s, 0.06 * s), 0.02 * s);
        col *= mix(1.0, 0.5, smoothstep(0.08 * s, -0.02 * s, length((q - so - vec2(0.0, 0.5 * s)) / vec2(0.36, 0.2)) - s));
        if (dbr < 1.0) col = mix(col, metal(normalize(vec3(clamp(q.x / (0.15 * s), -1.0, 1.0), 0.0, 0.8)), uBrass), aa(dbr));
        if (df < 1.0) {
            vec2 sp = (q - vec2(0.0, 0.5 * s)) / vec2(0.36 * s, 0.2 * s);
            col = mix(col, dome(sp * 0.95, uBrass * (0.85 + 0.2 * vnoise(sp * 4.0))), aa(df * 0.25));
        }
        return col;
    }

    void main() {
        vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec3 col = wall(p);
        col = boiler(p, col);
        col = floorB(p, col);
        col = lag(p, col);
        col = pipes(p, col);
        col = flanges(p, col);
        col = wheels(p, col);
        col = wgFrame(p, col, uWG[0], -1.0);
        col = wgFrame(p, col, uWG[1], 1.0);
        col = tags(p, col);
        col = firedoor(p, col);
        col = plaque(p, col);
        col = govBase(p, col);
        col = safety(p, col);
        col = lampBody(p, col);
        /* the instruments' shadows, then the instruments */
        for (int i = 0; i < 4; i++) {
            float rb = (uG[i].w > 0.5 ? 1.5 : 1.22) * uG[i].z;
            float d = length(p - uG[i].xy - vec2(0.035, 0.05) * rb) - rb;
            col *= mix(1.0, 0.35, smoothstep(0.12 * rb, -0.03 * rb, d));
        }
        for (int i = 0; i < 4; i++) {
            vec2 q = p - uG[i].xy;
            float d = length(q), r = uG[i].z;
            float ro = uG[i].w > 0.5 ? 1.5 : 1.22;
            if (d > ro * r + 1.0) continue;
            vec3 b = d > 1.22 * r ? mahogany(q, d, r) : bezel(i, q, d, r);
            if (d < r + 1.0) b = mix(b, face(i, q, d, r), aa(d - r));
            col *= 1.0 - 0.6 * aa(abs(d - ro * r - 0.002 * r) - 1.0);
            col = mix(col, b, aa(d - ro * r));
        }
        /* the mains jewel's socket */
        if (uML.z > 0.0) {
            vec2 q = p - uML.xy;
            float lr = uML.z, d = length(q);
            col *= mix(1.0, 0.55, smoothstep(lr * 1.9, lr * 1.3, length(q - vec2(0.2, 0.3) * lr)));
            float u = clamp((d - lr) / (0.6 * lr), 0.0, 1.0);
            vec2 dir = q / max(d, 1e-3);
            col = mix(col, metal(normalize(vec3(dir * cos(u * 3.14159) * 0.9, 1.0)), uBrass), aa(d - 1.6 * lr));
        }
        /* the light: the lamp's warm pool, the fire's glow, gloom in the corners */
        vec2 dl = (p - uLamp.xy) / uS;
        float pool = exp(-dot(dl, dl) / 5.0);
        vec2 df = (p - uFD.xy) / uS;
        float fg = exp(-length(df * vec2(0.55, 0.8)) * 0.8);
        vec3 ill = vec3(0.92, 0.9, 0.87) + vec3(1.0, 0.68, 0.36) * 0.9 * pool + vec3(1.0, 0.45, 0.15) * 0.3 * fg;
        vec2 uv = p / uRes - 0.5;
        ill *= 1.0 - 0.45 * pow(clamp(length(uv * vec2(uRes.x / max(uRes.x, uRes.y), uRes.y / max(uRes.x, uRes.y))) * 1.5, 0.0, 1.0), 2.5);
        o = vec4(col * ill, 1.0);
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

/* every frame: the static scene, then everything that moves */
static const char *FS_DYN = "#version 330 core\n" GLSL(
    uniform sampler2D uStatic, uAtlas;
    uniform vec2 uRes;
    uniform float uS, uT;
    uniform vec4 uG[4];
    uniform vec4 uN[4];
    uniform vec4 uWin[2];
    uniform vec4 uDp[17];
    uniform vec4 uDr[17];
    uniform vec4 uDig[10];
    uniform vec4 uBH;
    uniform vec4 uWG[2];
    uniform vec2 uWL;            /* water level in each glass, 0..1 */
    uniform vec4 uFD;
    uniform vec4 uFire;          /* intensity, flicker, lamp flicker */
    uniform vec4 uGov, uGovB, uGovS;   /* governor; its state: swing, phase, gear angle */
    uniform vec4 uLamp;
    uniform vec4 uML;            /* mains jewel x y r on */
    uniform vec4 uSV;
    uniform vec4 uPuff[3];       /* age, duration, strength, seed */
    uniform vec4 uLeak[3];       /* x y size drift */
    uniform vec3 uLeakA;
    uniform vec4 uDrip[2];       /* x y radius stretch */
    uniform vec3 uBrass, uRed, uSage;
    out vec4 o;
)
COMMON
GLSL(
    float fbm4(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.07 + vec2(5.3, 1.9); a *= 0.5; } return s / 0.9375; }

    float sdUC(vec2 p, float r1, float r2, float h) {
        p.x = abs(p.x);
        float b = (r1 - r2) / h, a = sqrt(1.0 - b * b);
        float k = dot(p, vec2(-b, a));
        if (k < 0.0) return length(p) - r1;
        if (k > a * h) return length(p - vec2(0.0, h)) - r2;
        return dot(p, vec2(a, b)) - r1;
    }
    float ndot(vec2 a, vec2 b) { return a.x * b.x - a.y * b.y; }
    float sdRhombus(vec2 p, vec2 b) {
        p = abs(p);
        float h = clamp(ndot(b - 2.0 * p, b) / dot(b, b), -1.0, 1.0);
        float d = length(p - 0.5 * b * vec2(1.0 - h, 1.0 + h));
        return d * sign(p.x * b.y + p.y * b.x - b.x * b.y);
    }
    vec2 hframe(vec2 q, float a) {
        vec2 dir = vec2(sin(a), -cos(a));
        return vec2(dot(q, vec2(-dir.y, dir.x)), dot(q, dir));
    }
    /* a pressure-gauge pointer: a slim blade with a ring counterweight */
    float needle(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.18 * r), 0.018 * r, 0.004 * r, 1.06 * r);
        float ring = abs(length(h + vec2(0.0, 0.17 * r)) - 0.045 * r) - 0.016 * r;
        return min(d, ring);
    }
    /* railway clock hands: plain spades */
    float hourHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.12 * r), 0.028 * r, 0.012 * r, 0.5 * r);
        return min(d, sdRhombus(h - vec2(0.0, 0.46 * r), vec2(0.055 * r, 0.11 * r)));
    }
    float minuteHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.14 * r), 0.022 * r, 0.008 * r, 0.82 * r);
        return min(d, sdRhombus(h - vec2(0.0, 0.72 * r), vec2(0.036 * r, 0.12 * r)));
    }
    float secondHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.25 * r), 0.008 * r, 0.0035 * r, 1.12 * r);
        return min(d, length(h + vec2(0.0, 0.2 * r)) - 0.032 * r);
    }
    vec3 steelHand(vec3 base, float d, float r) { return base * (1.0 + 1.6 * smoothstep(-0.012 * r, 0.0, d)); }

    vec3 glass(vec2 q, float r, vec3 col) {
        vec2 u = q / r;
        float rr = length(u);
        col *= 1.0 - 0.12 * smoothstep(0.75, 1.0, rr);
        vec2 un = u / max(rr, 1e-3);
        float c1 = length(u - vec2(-0.03, -0.04)), c2 = length(u - vec2(0.09, 0.12));
        float cres = smoothstep(0.98, 0.92, c1) * smoothstep(0.84, 0.97, c2);
        cres *= smoothstep(-0.1, 0.9, dot(un, vec2(-0.6, -0.8)));
        col += vec3(1.0, 0.93, 0.8) * cres * 0.2;
        float sheen = smoothstep(0.3, -0.7, dot(u, vec2(0.6, 0.8))) * smoothstep(1.0, 0.6, rr);
        col += vec3(1.0, 0.9, 0.75) * sheen * 0.045;
        vec2 sp = u - vec2(-0.4, -0.5);
        col += vec3(1.0, 0.95, 0.85) * exp(-dot(sp, sp * vec2(60.0, 120.0))) * 0.35;
        float glint = smoothstep(0.92, 0.975, rr) * smoothstep(1.0, 0.98, rr) * smoothstep(0.2, 0.95, dot(un, vec2(0.6, 0.8)));
        col += vec3(1.0, 0.75, 0.5) * glint * 0.2;
        return col;
    }

    vec3 instrument(int i, vec2 q, float r, vec3 col) {
        float bl = max(0.012 * r, 1.0);
        vec2 so = vec2(0.45, 0.65);
        vec3 dark = vec3(0.04, 0.038, 0.04);
        if (uG[i].w < 0.5) {
            float a = uN[i].x;
            col *= 1.0 - 0.38 * smoothstep(bl, -bl, needle(q - so * 0.035 * r, a, r));
            float d = needle(q, a, r);
            col = mix(col, steelHand(dark, d, r), aa(d));
        } else {
            vec4 n = uN[i];
            float s1 = min(hourHand(q - so * 0.025 * r, n.x, r), minuteHand(q - so * 0.035 * r, n.y, r));
            float s2 = secondHand(q - so * 0.05 * r, n.z, r);
            col *= 1.0 - 0.36 * smoothstep(bl, -bl, s1);
            col *= 1.0 - 0.3 * smoothstep(bl, -bl, s2);
            float dh = hourHand(q, n.x, r);
            col = mix(col, steelHand(dark, dh, r), aa(dh));
            float dm = minuteHand(q, n.y, r);
            col = mix(col, steelHand(dark, dm, r), aa(dm));
            float dsec = secondHand(q, n.z, r);
            col = mix(col, uRed * (1.0 + 0.5 * smoothstep(-0.006 * r, 0.0, dsec)), aa(dsec));
        }
        float hr = 0.055 * r, dh = length(q);
        col *= 1.0 - 0.35 * smoothstep(hr * 1.4, hr * 0.8, length(q - so * 0.3 * hr));
        if (dh < hr + 1.0) col = mix(col, dome(q / hr, uG[i].w > 0.5 ? vec3(0.12) : uBrass), aa(dh - hr));
        return glass(q, r, col);
    }

    vec3 drum(int j, vec2 p, vec3 col) {
        vec4 R = uDp[j];
        vec2 c = (R.xy + R.zw) * 0.5;
        vec2 lc = p - c;
        float hh = (R.w - R.y) * 0.5;
        float v = lc.y / hh;
        float phi = asin(clamp(v * 0.42, -1.0, 1.0));
        float pitch = hh / 0.42 * 0.62832;
        vec4 D = uDr[j];
        float s = D.z + phi / 0.62832;
        float k = floor(s + 0.5);
        float rad = abs(D.w);
        float dg = k <= 0.0 ? mod(D.x + k + rad * 4.0, rad) : mod(D.y + k - 1.0, rad);
        float ly = (s - k) * pitch;
        vec4 g = uDig[int(dg + 0.5)];
        vec2 uv = g.xy + g.zw * 0.5 + vec2(lc.x, ly);
        float ink = 0.0;
        if (abs(lc.x) < g.z * 0.5 + 1.0 && abs(ly) < g.w * 0.5 + 1.0)
            ink = texture(uAtlas, uv / vec2(textureSize(uAtlas, 0))).r;
        bool ivory = D.w < 0.0;
        vec3 bg = ivory ? vec3(0.86, 0.8, 0.7) : vec3(0.06, 0.05, 0.045);
        vec3 fg = ivory ? vec3(0.5, 0.12, 0.08) : vec3(0.9, 0.85, 0.76);
        vec3 dc = mix(bg, fg, ink);
        float cphi = cos(phi);
        dc *= 0.3 + 0.75 * pow(smoothstep(0.86, 1.0, cphi), 1.2) + 0.08 * smoothstep(0.0, -0.8, v);
        float hw = (R.z - R.x) * 0.5;
        dc *= 1.0 - 0.5 * smoothstep(hw - 2.5, hw, abs(lc.x));
        return dc;
    }

    /* the fire: a flickering blackbody ramp, moving up */
    vec3 fireCol(vec2 p, float I) {
        vec2 f = (p - uFD.xy) / uS;
        float n = fbm3(vec2(f.x * 3.0, f.y * 2.2 + uT * 1.7)) * 0.65 + 0.35 * vnoise(vec2(f.x * 7.0 - uT * 0.4, f.y * 6.0 + uT * 3.1));
        float h = I * (0.25 + 1.15 * smoothstep(0.2, 0.8, n)) * (0.85 + 0.3 * uFire.y);
        vec3 c = mix(vec3(0.25, 0.02, 0.0), vec3(1.0, 0.28, 0.03), smoothstep(0.1, 0.5, h));
        c = mix(c, vec3(1.0, 0.62, 0.18), smoothstep(0.45, 0.85, h));
        c = mix(c, vec3(1.0, 0.9, 0.65), smoothstep(0.85, 1.25, h));
        return c;
    }

    vec3 firehole(vec2 p, vec3 col) {
        vec2 q = p - uFD.xy, hs = uFD.zw;
        float u = uS, I = uFire.x;
        float dIn = sdBox(q, hs, 0.3 * u);
        if (dIn < 0.5) {
            float gap = 0.022 * u;
            float mg = aa(abs(q.x) - gap) * (0.75 + 0.25 * smoothstep(-hs.y, hs.y, q.y));
            float mp = aa(-(dIn + 0.03 * u)) * aa(dIn) * (0.25 + 0.75 * smoothstep(-hs.y, hs.y, q.y));
            float ms = aa(fhSlots(q, hs, u));
            float m = max(mg, max(mp, ms * 0.9));
            vec3 f = fireCol(p, I);
            /* the leaves' edges and the holes' rims lit by the fire */
            float near = exp(-max(abs(q.x) - gap, 0.0) / (0.05 * u)) * (1.0 - smoothstep(hs.y * 0.6, hs.y, abs(q.y)));
            float rim = aa(abs(fhSlots(q, hs, u) - 1.5) - 1.5);
            col += f * (col * 2.0 + 0.05) * (near * 0.8 + rim * 0.6) * I;
            col += f * 0.12 * I * exp(-max(abs(q.x) - gap, 0.0) / (0.12 * u)) * smoothstep(hs.y * 1.1, hs.y * 0.5, abs(q.y));
            col = mix(col, f, m);
        }
        return col;
    }

    /* the fire's light on the floor: a bar thrown through the gap between the doors, and the glow off the plate */
    vec3 floorLight(vec2 p, vec3 col) {
        float dy = p.y - uBH.w;
        if (dy < 0.0) return col;
        float I = uFire.x * (0.8 + 0.4 * uFire.y);
        float w = 0.025 * uS + dy * 0.12;
        float bar = exp(-pow((p.x - uFD.x) / w, 2.0)) * smoothstep(0.0, 0.08 * uS, dy);
        float pool = exp(-length(vec2((p.x - uFD.x) / (2.2 * uS), dy / (1.0 * uS))));
        return col * (1.0 + vec3(1.0, 0.5, 0.18) * (bar * 2.2 + pool * 0.5) * I);
    }

    /* the water gauge glasses: Klinger stripes behind, which the water
     * turns the other way, a meniscus, a bubble now and then */
    vec3 waterGlass(vec2 p, vec3 col, vec4 G, float lvl, float k) {
        float x = G.x, y0 = G.y, y1 = G.z, tr = G.w;
        float u = (p.x - x) / (1.2 * tr);
        if (abs(u) > 1.0 || p.y < y0 || p.y > y1) return col;
        float ws = y1 - (y1 - y0) * lvl;
        /* the surface tilts and ripples a little */
        ws += sin(uT * 2.3 + k * 2.0 + u * 1.5) * 0.04 * tr + sin(uT * 3.7 + k) * 0.03 * tr;
        float wet = smoothstep(ws - 0.7, ws + 0.7, p.y);
        float sx = asin(clamp(u, -0.99, 0.99)) / 1.5708;
        float per = 0.95 * tr;
        float dry = sin(((sx * 1.1 * tr) + (p.y - y0)) / per * 3.14159);
        float wv = sin(((sx * 1.8 * tr) - (p.y - y0)) / per * 3.14159);
        float stripes = mix(dry, wv, wet);
        vec3 back = mix(vec3(0.05, 0.045, 0.04), vec3(0.72, 0.7, 0.64), smoothstep(-0.3, 0.3, stripes));
        back = mix(back, back * vec3(0.8, 0.95, 0.95) + vec3(0.0, 0.02, 0.02), wet);
        vec3 c = back * (0.65 + 0.35 * sqrt(max(1.0 - u * u, 0.0)));
        /* the meniscus: a dark curve with a bright line over it */
        float dm = p.y - ws + 0.25 * tr * (u * u);
        c = mix(c, vec3(0.02), 0.85 * exp(-pow(dm / (0.09 * tr), 2.0)));
        c += vec3(0.9, 0.95, 1.0) * 0.4 * exp(-pow((dm + 0.14 * tr) / (0.05 * tr), 2.0));
        /* a bubble rising */
        float bt = fract(uT * 0.11 + k * 0.37);
        vec2 bc = vec2(x + 0.3 * tr * sin(uT * 3.0 + k), mix(y1, ws, bt));
        float db = length(p - bc) - 0.12 * tr;
        c = mix(c, c * 0.4 + vec3(0.5), 0.6 * aa(abs(db) - 0.6) * wet * step(ws + 0.2 * tr, bc.y));
        /* the glass: lit edges and a long highlight */
        c *= 1.0 - 0.5 * smoothstep(0.7, 1.0, abs(u));
        c += vec3(1.0, 0.97, 0.9) * (0.35 * exp(-pow((u + 0.45) / 0.12, 2.0)) + 0.12 * exp(-pow((u - 0.6) / 0.08, 2.0)));
        return mix(col, c, aa((abs(u) - 1.0) * 1.2 * tr));
    }

    /* the governor: spindle, the two flyball arms with their links, the sleeve, the bevel gear */
    vec3 govArm(vec2 p, vec3 col, float ang, float th) {
        float gx = uGov.x, py = uGov.y, L = uGov.z;
        float z = sin(ang), cs = cos(ang);
        vec2 piv = vec2(gx + 0.05 * L * cs, py);
        vec2 ball = vec2(gx + L * sin(th) * cs, py + L * cos(th));
        float ys = py + 0.95 * L * cos(th);
        vec2 mid = mix(piv, ball, 0.5);
        float rb = 0.19 * L * (1.0 + 0.06 * z);
        vec2 so = vec2(0.1, 0.15) * L;
        float darm = min(sdSeg(p, piv, ball) - 0.024 * L, sdSeg(p, mid, vec2(gx + 0.06 * L * cs, ys)) - 0.016 * L);
        float dball = length(p - ball) - rb;
        col *= mix(1.0, 0.6, smoothstep(0.03 * L, -0.02 * L, min(sdSeg(p - so, piv, ball) - 0.024 * L, length(p - so - ball) - rb)));
        float shade = 0.8 + 0.2 * z;
        if (darm < 1.0) {
            vec2 dir = normalize(ball - piv);
            float v = clamp(dot(p - piv, vec2(-dir.y, dir.x)) / (0.024 * L), -1.0, 1.0);
            col = mix(col, metal(normalize(vec3(vec2(-dir.y, dir.x) * v, 1.0)), vec3(0.42, 0.4, 0.38)) * shade, aa(darm));
        }
        if (dball < 1.0) col = mix(col, dome((p - ball) / rb, uBrass) * shade, aa(dball));
        return col;
    }
    vec3 governor(vec2 p, vec3 col) {
        float gx = uGov.x, py = uGov.y, L = uGov.z, gy = uGov.w;
        if (abs(p.x - gx) > 1.05 * L || p.y < py - 0.3 * L || p.y > gy + 1.2 * uGovB.z) return col;
        float th = uGovS.x, ph = uGovS.y;
        bool firstBack = sin(ph) < 0.0;
        float a0 = firstBack ? ph : ph + 3.14159, a1 = firstBack ? ph + 3.14159 : ph;
        col = govArm(p, col, a0, th);
        /* spindle, sleeve and the top cap */
        float ys = py + 0.95 * L * cos(th);
        float dsp = sdBox(p - vec2(gx, (py - 0.12 * L + gy) * 0.5), vec2(0.03 * L, (gy - py + 0.12 * L) * 0.5), 0.0);
        float dsl = sdBox(p - vec2(gx, ys), vec2(0.075 * L, 0.07 * L), 0.02 * L);
        float dcap = length(p - vec2(gx, py - 0.12 * L)) - 0.07 * L;
        float dhub = sdBox(p - vec2(gx, py), vec2(0.09 * L, 0.05 * L), 0.02 * L);
        float dd = min(min(dsp, dsl), min(dcap, dhub));
        if (dd < 1.0) {
            float hw = dsp == dd ? 0.03 : dsl == dd ? 0.075 : dcap == dd ? 0.07 : 0.09;
            float v = clamp((p.x - gx) / (hw * L), -1.0, 1.0);
            vec3 n = normalize(vec3(v, 0.0, sqrt(1.0 - v * v)));
            vec3 base = dsp == dd ? vec3(0.45, 0.44, 0.42) : uBrass;
            col = mix(col, metal(n, base), aa(dd));
        }
        col = govArm(p, col, a1, th);
        /* the bevel gear on the gearbox face, turning */
        vec2 gq = p - vec2(gx, gy);
        float gr = uGovB.z * 0.78, d = length(gq);
        if (d < gr * 1.15) {
            float a = atan(gq.y, gq.x) + uGovS.z;
            float teeth = 0.5 + 0.5 * cos(a * 22.0);
            float dg = d - gr * (1.0 + 0.07 * smoothstep(0.2, 0.8, teeth));
            float sp = 6.28318 / 4.0;
            float ka = floor(a / sp + 0.5) * sp;
            float web = abs(sin(a - ka)) * d - 0.11 * gr;
            float hole = max(-web, max(d - 0.72 * gr, 0.28 * gr - d));
            float dgear = max(dg, -hole);
            col *= mix(1.0, 0.5, smoothstep(0.05 * gr, -0.05 * gr, length(gq - vec2(0.05, 0.08) * gr) - gr));
            if (dgear < 1.0) {
                vec3 n = normalize(vec3(gq / gr * 0.5 * smoothstep(0.75, 1.0, d / gr), 1.0));
                vec3 c = metal(n, uBrass * 0.85) * (1.0 - 0.3 * smoothstep(0.85, 1.0, d / gr) * (1.0 - teeth));
                c = mix(c, dome(gq / (0.2 * gr), vec3(0.4, 0.38, 0.36)), aa(d - 0.2 * gr));
                col = mix(col, c, aa(dgear));
            }
        }
        return col;
    }

    /* the lamp's flame, and its glow in the chimney */
    vec3 lampFlame(vec2 p, vec3 col) {
        float s = uLamp.z;
        vec2 q = (p - uLamp.xy) / s;
        if (abs(q.x) > 1.0 || q.y < -1.2 || q.y > 0.6) return col;
        float f = uFire.z;
        float sw = 0.012 * sin(uT * 7.0) + 0.01 * sin(uT * 11.3);
        vec2 fq = vec2(q.x - sw * (0.2 - q.y), q.y);
        float hgt = 0.2 * (1.0 + 0.06 * f);
        float wid = 0.055 * (1.0 + 0.6 * smoothstep(-hgt, 0.08, fq.y));
        float d = length(vec2(fq.x / wid, (fq.y + 0.0) / hgt * (fq.y < 0.0 ? 1.0 : 2.2)));
        float core = smoothstep(1.0, 0.3, d);
        vec3 fc = mix(vec3(1.0, 0.55, 0.15), vec3(1.0, 0.95, 0.75), smoothstep(0.4, 0.0, d - 0.1 * fq.y / hgt));
        fc = mix(fc, vec3(0.3, 0.4, 0.9), smoothstep(0.03, 0.09, fq.y) * smoothstep(0.6, 1.0, d) * 0.4);
        float inGlass = step(abs(q.x), 0.2) * step(-0.75, q.y) * step(q.y, 0.25);
        col += vec3(1.0, 0.72, 0.35) * (0.35 + 0.05 * f) * exp(-length(q * vec2(1.0, 0.7)) * 5.0) * inGlass;
        col += vec3(1.0, 0.7, 0.35) * 0.1 * exp(-length(q) * 2.2);
        return mix(col, fc * 1.2, core);
    }

    /* one plume of steam from src: billowing up and spreading, the part
     * between heights lo and hi (the stretch emitted while the valve was open) */
    float plume(vec2 p, vec2 src, float sz, float spd, float seed, float lo, float hi, float drift) {
        vec2 q = (p - src) / sz;
        float up = -q.y;
        if (up < -0.3 || up > 4.6 || up < lo - 0.6 || up > hi + 0.8) return 0.0;
        float x = q.x - drift * up * up * 0.2;
        float w = 0.1 + 0.3 * up;
        if (abs(x) > 2.2 * w + 0.15) return 0.0;
        float env = exp(-x * x / (w * w)) * smoothstep(-0.15, 0.2, up) * exp(-up * 0.5) * smoothstep(4.6, 3.6, up);
        env *= smoothstep(lo - 0.5, lo + 0.2, up) * smoothstep(hi + 0.7, hi - 0.1, up);
        if (env < 0.004) return 0.0;
        vec2 nq = vec2(x / (0.3 + 0.25 * up) + seed, up * 1.1 - uT * spd);
        float wv = vnoise(nq * 1.4 + 3.7);
        float f = fbm4(nq * vec2(1.8, 1.5) + vec2(wv * 1.4, wv * 0.6));
        return env * smoothstep(0.3, 0.75, f + 0.15 * env);
    }

    vec3 steamOver(vec2 p, vec3 col, float a) {
        vec2 dl = (p - uLamp.xy) / uS;
        vec2 df = (p - uFD.xy) / uS;
        vec3 lit = vec3(0.74, 0.74, 0.74) + vec3(1.0, 0.75, 0.45) * 0.35 * exp(-dot(dl, dl) / 8.0)
                 + vec3(1.0, 0.45, 0.15) * 0.4 * uFire.x * exp(-length(df) * 0.7);
        return mix(col, vec3(0.93, 0.92, 0.9) * lit, clamp(a, 0.0, 1.0));
    }

    /* a water drop: hanging, swelling, then falling */
    vec3 drip(vec2 p, vec3 col, vec4 D) {
        if (D.z <= 0.0) return col;
        vec2 q = p - D.xy;
        q.y /= D.w;
        float d = length(q) - D.z;
        if (d > 2.0) return col;
        vec2 sp = q / D.z;
        vec3 c = col * 0.45 + vec3(0.05, 0.06, 0.07);
        c += vec3(1.0, 0.95, 0.85) * 0.8 * exp(-dot(sp - vec2(-0.35, -0.35), sp - vec2(-0.35, -0.35)) * 10.0);
        c += vec3(1.0, 0.6, 0.3) * 0.25 * smoothstep(0.4, 1.0, sp.y) * smoothstep(1.0, 0.7, length(sp));
        return mix(col, c, aa(d * min(D.w, 1.0)));
    }

    void main() {
        vec2 fc = gl_FragCoord.xy;
        vec2 p = vec2(fc.x, uRes.y - fc.y);
        vec3 col = texelFetch(uStatic, ivec2(fc), 0).rgb;
        /* the fire's and the lamp's light, flickering over the baked average */
        vec2 df = (p - uFD.xy) / uS;
        float fg = exp(-length(df * vec2(0.55, 0.8)) * 0.8);
        vec2 dl = (p - uLamp.xy) / uS;
        float pool = exp(-dot(dl, dl) / 5.0);
        col *= 1.0 + vec3(1.0, 0.45, 0.15) * fg * ((uFire.x - 0.5) * 0.35 + uFire.y * 0.12 * uFire.x)
                   + vec3(1.0, 0.7, 0.4) * pool * 0.08 * uFire.z;
        col = floorLight(p, col);
        for (int i = 0; i < 4; i++) {
            vec2 q = p - uG[i].xy;
            float r = uG[i].z;
            if (dot(q, q) < (r + 1.0) * (r + 1.0)) col = mix(col, instrument(i, q, r, col), aa(length(q) - r));
        }
        for (int w = 0; w < 2; w++) {
            vec4 R = uWin[w];
            if (p.x < R.x || p.x > R.z || p.y < R.y || p.y > R.w) continue;
            for (int j = 0; j < 17; j++) {
                vec4 D = uDp[j];
                if (p.x >= D.x && p.x < D.z && p.y >= D.y && p.y < D.w) col = drum(j, p, col);
            }
            float t = (p.y - R.y) / (R.w - R.y);
            col += vec3(1.0, 0.94, 0.82) * 0.07 * smoothstep(0.1, 0.2, t) * smoothstep(0.42, 0.25, t);
            col *= 1.0 - 0.5 * smoothstep(0.2, 0.0, t) - 0.35 * smoothstep(0.8, 1.0, t);
        }
        col = waterGlass(p, col, uWG[0], uWL.x, 0.0);
        col = waterGlass(p, col, uWG[1], uWL.y, 1.0);
        col = firehole(p, col);
        col = governor(p, col);
        col = lampFlame(p, col);
        if (uML.z > 0.0) {
            vec2 q = p - uML.xy;
            float lr = uML.z, on = uML.w, d = length(q);
            if (d < lr * 6.0) {
                col += uSage * on * 0.35 * exp(-max(d - lr, 0.0) / (lr * 1.1)) * step(lr * 1.6, d);
                if (d < lr + 1.0) {
                    vec2 sp = q / lr;
                    float z = sqrt(max(1.0 - dot(sp, sp), 0.0));
                    float fac = 0.85 + 0.15 * cos(floor(atan(sp.y, sp.x) / 0.7854) * 2.1 + floor(length(sp) * 3.0));
                    vec3 j = uSage * (0.13 + on * (0.5 + 1.1 * z)) * fac;
                    j += vec3(1.0, 0.95, 0.9) * exp(-dot(sp - vec2(-0.35, -0.4), sp - vec2(-0.35, -0.4)) * 25.0) * 0.55;
                    col = mix(col, j, aa(d - lr));
                }
            }
        }
        col = drip(p, col, uDrip[0]);
        col = drip(p, col, uDrip[1]);
        /* steam: the leaking joints, then the safety valve */
        float a = 0.0;
        for (int i = 0; i < 3; i++)
            a += uLeakA[i] * 1.5 * plume(p, uLeak[i].xy, uLeak[i].z, 0.55, float(i) * 17.0, -1.0, 99.0, uLeak[i].w);
        vec2 sv = uSV.xy - vec2(0.0, 1.05 * uSV.z);
        for (int i = 0; i < 3; i++) {
            vec4 P = uPuff[i];
            if (P.z <= 0.0) continue;
            float spd = 1.5;
            float hi = P.x * spd * 0.9, lo = max(P.x - P.y, 0.0) * spd * 0.9;
            float fade = exp(-max(P.x - P.y, 0.0) * 0.45);
            float jet = P.x < P.y ? smoothstep(0.0, 0.25, P.x) * smoothstep(0.0, 0.4, P.y - P.x) : 0.0;
            float pl = plume(p, sv, 1.3 * uSV.z * 1.6, spd, P.w, lo, hi, 0.15);
            a += P.z * fade * pl * 1.6;
            /* the hard white jet right at the valve while it blows */
            vec2 jq = (p - sv) / uSV.z;
            a += jet * P.z * exp(-pow(jq.x / (0.12 + 0.25 * max(-jq.y, 0.0)), 2.0)) * smoothstep(0.0, 0.15, -jq.y) * exp(jq.y * 0.9) * 0.9
                 * (0.7 + 0.3 * vnoise(vec2(jq.x * 4.0, jq.y * 3.0 + uT * 9.0)));
        }
        if (a > 0.002) col = steamOver(p, col, a);
        o = vec4(col, 1.0);
    });
static double t_start_wall;
static double sim_t, sim_t_now;

static GLuint prog_static, prog_text, prog_dyn, vao, vao_text, vbo_text;
static GLuint tex_atlas, tex_static, fbo_static;
static int static_w, static_h;
static float c_brass[3], c_red[3], c_amber[3], c_sage[3], c_copper[3], c_green[3];

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "steam: shader:\n%s\n", log);
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

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

#define U(p, n) glGetUniformLocation(p, n)

static void gl_init(void) {
    prog_static = mk_prog(VS, FS_STATIC);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    prog_dyn = mk_prog(VS, FS_DYN);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_text);
    glGenBuffers(1, &vbo_text);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(4 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glGenTextures(1, &tex_atlas);

    /* Aged Brass, and the engine's own paint */
    hex("#A53C2C", c_red); hex("#E9994A", c_amber); hex("#8FA672", c_sage);
    c_brass[0] = 0.70f; c_brass[1] = 0.52f; c_brass[2] = 0.25f;
    c_copper[0] = 0.72f; c_copper[1] = 0.36f; c_copper[2] = 0.22f;
    c_green[0] = 0.07f; c_green[1] = 0.15f; c_green[2] = 0.1f;
}

/* (re)build everything that only changes with the size: layout, atlas,
 * print, and the static scene texture */
static void build_static(void) {
    layout();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (static_w != W || static_h != H) {
        if (static_w) { glDeleteTextures(1, &tex_static); glDeleteFramebuffers(1, &fbo_static); }
        glGenTextures(1, &tex_static);
        glBindTexture(GL_TEXTURE_2D, tex_static);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glGenFramebuffers(1, &fbo_static);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_static);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_static, 0);
        static_w = W; static_h = H;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_static);
    glViewport(0, 0, W, H);

    GLuint p = prog_static;
    glUseProgram(p);
    float gv[NG][4], av[NG][4], zv[NG][4];
    for (int i = 0; i < NG; i++) {
        gv[i][0] = G[i].cx; gv[i][1] = G[i].cy; gv[i][2] = G[i].r; gv[i][3] = (float)G[i].kind;
        av[i][0] = G[i].a0; av[i][1] = G[i].a1; av[i][2] = (float)G[i].nmaj; av[i][3] = (float)G[i].nmin;
        zv[i][0] = G[i].red0; zv[i][1] = G[i].red1; zv[i][2] = G[i].amb0; zv[i][3] = G[i].amb1;
    }
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform1f(U(p, "uSeed"), (opt_seed % 1000) * 0.137f);
    glUniform1f(U(p, "uS"), SC.s);
    glUniform4fv(U(p, "uBH"), 1, SC.bh);
    glUniform4fv(U(p, "uHor"), 1, SC.hor);
    glUniform4fv(U(p, "uLag"), 1, SC.lag);
    glUniform4fv(U(p, "uPA"), MAXP, &SC.pa[0][0]);
    glUniform4fv(U(p, "uPB"), MAXP, &SC.pb[0][0]);
    glUniform1i(U(p, "uNP"), SC.np);
    glUniform4fv(U(p, "uFl"), MAXF, &SC.fl[0][0]);
    glUniform1i(U(p, "uNF"), SC.nf);
    glUniform4fv(U(p, "uWh"), MAXW, &SC.wh[0][0]);
    glUniform1i(U(p, "uNW"), SC.nw);
    glUniform4fv(U(p, "uG"), NG, &gv[0][0]);
    glUniform4fv(U(p, "uA"), NG, &av[0][0]);
    glUniform4fv(U(p, "uZ"), NG, &zv[0][0]);
    glUniform4fv(U(p, "uWG"), 2, &SC.wg[0][0]);
    glUniform4fv(U(p, "uTag"), 2, &SC.tag[0][0]);
    glUniform4fv(U(p, "uFD"), 1, SC.fd);
    glUniform4fv(U(p, "uPq"), 1, SC.pq);
    glUniform4fv(U(p, "uRule"), 1, SC.rule);
    glUniform4fv(U(p, "uWin"), 2, &win_rect[0][0]);
    glUniform4fv(U(p, "uGov"), 1, SC.gov);
    glUniform4fv(U(p, "uGovB"), 1, SC.govb);
    glUniform4fv(U(p, "uSV"), 1, SC.sv);
    glUniform4fv(U(p, "uLamp"), 1, SC.lamp);
    glUniform4fv(U(p, "uML"), 1, SC.ml);
    glUniform3fv(U(p, "uRed"), 1, c_red);
    float amb[3];
    for (int i = 0; i < 3; i++) amb[i] = c_amber[i] * 0.9f;
    glUniform3fv(U(p, "uAmber"), 1, amb);
    glUniform3fv(U(p, "uBrass"), 1, c_brass);
    glUniform3fv(U(p, "uCopper"), 1, c_copper);
    glUniform3fv(U(p, "uGreen"), 1, c_green);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* the print */
    glUseProgram(prog_text);
    glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_text, "uAtlas"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STATIC_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    glDisable(GL_BLEND);

    /* the dynamic pass's fixed uniforms */
    p = prog_dyn;
    glUseProgram(p);
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    glUniform1f(U(p, "uS"), SC.s);
    glUniform1i(U(p, "uStatic"), 0);
    glUniform1i(U(p, "uAtlas"), 1);
    glUniform4fv(U(p, "uG"), NG, &gv[0][0]);
    glUniform4fv(U(p, "uWin"), 2, &win_rect[0][0]);
    glUniform4fv(U(p, "uDp"), NDRUM, &drum_rect[0][0]);
    float dig[10][4];
    for (int d = 0; d < 10; d++) {
        const Glyph *g = glyph(drum_style, (unsigned char)('0' + d));
        dig[d][0] = g->x; dig[d][1] = g->y; dig[d][2] = g->w; dig[d][3] = g->h;
    }
    glUniform4fv(U(p, "uDig"), 10, &dig[0][0]);
    glUniform4fv(U(p, "uBH"), 1, SC.bh);
    glUniform4fv(U(p, "uWG"), 2, &SC.wg[0][0]);
    glUniform4fv(U(p, "uFD"), 1, SC.fd);
    glUniform4fv(U(p, "uGov"), 1, SC.gov);
    glUniform4fv(U(p, "uGovB"), 1, SC.govb);
    glUniform4fv(U(p, "uLamp"), 1, SC.lamp);
    glUniform4fv(U(p, "uSV"), 1, SC.sv);
    glUniform4fv(U(p, "uLeak"), NLEAK, &SC.leak[0][0]);
    glUniform3fv(U(p, "uBrass"), 1, c_brass);
    glUniform3fv(U(p, "uRed"), 1, c_red);
    glUniform3fv(U(p, "uSage"), 1, c_sage);
}

/* smooth value noise in time, -1..1 */
static float tnoise(float t, float seed) {
    float i = floorf(t), f = t - i;
    float u = f * f * (3 - 2 * f);
    uint32_t a = (uint32_t)(int32_t)i * 2654435761u ^ (uint32_t)(seed * 977), b = (uint32_t)(int32_t)(i + 1) * 2654435761u ^ (uint32_t)(seed * 977);
    a ^= a >> 15; a *= 2246822519u; a ^= a >> 13;
    b ^= b >> 15; b *= 2246822519u; b ^= b >> 13;
    float va = (a & 0xffff) / 32767.5f - 1, vb = (b & 0xffff) / 32767.5f - 1;
    return va + (vb - va) * u;
}

/* t: seconds since start; wt: wall-clock time */
static void render(double t, double wt, GLuint target) {
    if (static_w != W || static_h != H) build_static();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    GLuint p = prog_dyn;
    glUseProgram(p);
    float ft = (float)fmod(t, 10000.0);
    glUniform1f(U(p, "uT"), ft);

    float nv[NG][4] = {{0}};
    for (int i = 0; i < G_CLOCK; i++) nv[i][0] = needle_angle(&G[i], t);
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    float secs = tm.tm_sec + (float)(wt - floor(wt));
    float mins = tm.tm_min + secs / 60;
    nv[G_CLOCK][0] = ((tm.tm_hour % 12) + mins / 60) * 30 * DEG;
    nv[G_CLOCK][1] = mins * 6 * DEG;
    nv[G_CLOCK][2] = sec_angle * DEG;
    glUniform4fv(U(p, "uN"), NG, &nv[0][0]);

    /* drums: digit now, digit next, and the roll between over the last 0.3 s */
    double up = sens.uptime + (now() - sens.uptime_at);
    double away = wt - (double)locked_at;
    if (away < 0) away = 0;
    float dr[NDRUM][4];
    for (int j = 0; j < NDRUM; j++) {
        double v = j < 9 ? up : away;
        long n = (long)v;
        double fr = v - n;
        int a = digit_of(n, drum_spec[j]), b = digit_of(n + 1, drum_spec[j]);
        float f = 0;
        if (a != b && fr > 0.7) {
            float x = (float)((fr - 0.7) / 0.3);
            float c1 = 1.2f, c3 = c1 + 1, xm = x - 1;
            f = 1 + c3 * xm * xm * xm + c1 * xm * xm;
        }
        int ivory = drum_spec[j] == D_S1 || drum_spec[j] == D_S10;
        dr[j][0] = (float)a; dr[j][1] = (float)b; dr[j][2] = f;
        dr[j][3] = (float)radix_of(drum_spec[j]) * (ivory ? -1 : 1);
    }
    glUniform4fv(U(p, "uDr"), NDRUM, &dr[0][0]);

    /* the water, the second glass a beat behind the first */
    float wl0 = clampf(wat.x + 0.004f * sinf(ft * 1.3f), 0.02f, 0.98f);
    float wl1 = clampf(wat.x + 0.004f * sinf(ft * 1.3f - 0.9f) - 0.002f, 0.02f, 0.98f);
    glUniform2f(U(p, "uWL"), wl0, wl1);

    /* the fire: its draw, a quick flicker, the lamp's slower one */
    float fl = 0.5f * tnoise(ft * 3.1f, 1) + 0.3f * tnoise(ft * 7.3f, 2) + 0.2f * tnoise(ft * 13.0f, 3);
    float lf = 0.6f * tnoise(ft * 2.3f, 4) + 0.4f * tnoise(ft * 6.1f, 5);
    glUniform4f(U(p, "uFire"), fire_s, fl, lf, 0);
    glUniform4f(U(p, "uGovS"), gov_th, gov_ph, -gov_ph * 1.6f, 0);

    float ml[4] = {SC.ml[0], SC.ml[1], SC.ml[2],
                   sens.mains ? (sens.charging ? 0.7f + 0.3f * sinf(ft * 2.0f) : 1.0f) : 0.0f};
    glUniform4fv(U(p, "uML"), 1, ml);

    /* safety-valve lifts: age, how long it blew, strength (0: none), seed */
    float pv[NPUFF][4] = {{0}};
    for (int i = 0; i < NPUFF; i++) {
        float age = (float)sim_t_now - puff[i].t0;
        if (puff[i].amp > 0 && age >= 0 && age < puff[i].dur + 7) {
            pv[i][0] = age; pv[i][1] = puff[i].dur; pv[i][2] = puff[i].amp; pv[i][3] = puff[i].seed;
        }
    }
    glUniform4fv(U(p, "uPuff"), NPUFF, &pv[0][0]);
    /* the leaks breathe in and out, slowly */
    float la[3];
    for (int i = 0; i < 3; i++) la[i] = clampf(0.6f + 0.35f * tnoise(ft * 0.12f + i * 7.0f, 10 + i), 0, 1) * (i == 2 ? 0.7f : 1.0f);
    glUniform3fv(U(p, "uLeakA"), 1, la);

    /* drips: swell under the pipe, let go, fall */
    float dv[NDRIP][4] = {{0}};
    for (int i = 0; i < NDRIP; i++) {
        float per = SC.drip[i][2], ph = fmodf(ft + SC.drip[i][3], per) / per;
        float s = SC.s, r0 = 0.05f * s;
        dv[i][0] = SC.drip[i][0];
        if (ph < 0.88f) {
            float g = ph / 0.88f;
            float r = r0 * (0.35f + 0.65f * sqrtf(g));
            dv[i][1] = SC.drip[i][1] + r * (0.6f + 0.5f * g);
            dv[i][2] = r; dv[i][3] = 1.0f + 0.35f * g;
        } else {
            float tf = (ph - 0.88f) * per;
            float y = 0.5f * 40.0f * tf * tf * s;
            dv[i][1] = SC.drip[i][1] + r0 + y;
            dv[i][2] = y > 2.5f * s ? 0 : r0 * 0.85f; dv[i][3] = 1.0f + clampf(tf * 6.0f, 0, 1.5f);
        }
    }
    glUniform4fv(U(p, "uDrip"), NDRIP, &dv[0][0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_static);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* advance the springs to time t (fixed sub-steps) and read the sensors once a second */
static double last_read;
static void advance(double t, double wt) {
    if (t - last_read >= 1.0) {
        read_sensors();
        set_targets();
        last_read = t;
    }
    const float h = 1.0f / 240;
    while (sim_t + h <= t) {
        sim_t += h;
        physics(wt - (t - sim_t), sim_t, h, sim_t > 0.6);
    }
    sim_t_now = t;
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
    /* a resize rebuilds the panel at the next frame */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "steam");
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
        "usage: steam [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY] [--sans FAMILY]\n"
        "             [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
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
        else if (!strcmp(a, "-sans") && i + 1 < argc) opt_sans = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    int offscreen = opt_shot || opt_bench > 0;

    /* sensors first: the scales depend on them */
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
    find_temp();
    find_aux();
    read_sensors();
    {   /* a first CPU-load sample over a tenth of a second */
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
        read_sensors();
    }
    erng = rng ^ 0x9e3779b9u;
    if (!erng) erng = 1;
    sv_next = 5 + 10 * efrand();
    gauges_init();
    set_targets();
    fonts_init();

    double t_wall0 = wall();
    if (opt_shot) t_wall0 -= opt_at;
    t_start_wall = t_wall0;
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
        XStoreName(dpy, win, "Steam");
        XClassHint chint = {"steam", "Steam"};
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
        last_read = 0;
        if (opt_shot) {
            /* let the springs run to --at (the readings held still) */
            double wt = wall();
            const float h = 1.0f / 240;
            sec_angle = (float)(((time_t)wt % 60) * 6);
            for (sim_t = 0; sim_t < opt_at; sim_t += h) physics(wt - (opt_at - sim_t), sim_t, h, sim_t > 0.6);
            sim_t_now = opt_at;
            render(opt_at, wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double t0 = now();
        build_static();
        glFinish();
        double tb = now() - t0, sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = 1.0 + i / 60.0;
            advance(t, wall());
            render(t, wall(), fbo);
            glFinish();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("steam: %dx%d, %d frames: avg %.3f ms, worst %.3f ms per frame (static scene built once in %.1f ms)\n",
               W, H, opt_bench, sum / opt_bench, worst, tb * 1000);
        printf("steam: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    sim_t = 0;
    last_read = 0;
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
        double t = now(), el = t - t0, wt = wall();
        advance(el, wt);
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
