/*
 * gauges — the machine's vital signs on a brass instrument panel, like the
 * gauge cluster of an old locomotive or a ship's engine room.
 *
 * A blackened-steel plate, riveted into a dark wood surround, carries:
 *
 *   CPU TEMP   the package sensor, with a terracotta red-line and a lamp
 *              that lights when the needle is in it
 *   CPU LOAD   from /proc/stat deltas
 *   CLOCK      local time, hour and minute hands sweeping, seconds ticking
 *   MEMORY     in use (MemTotal - MemAvailable), in GB
 *   BATTERY    a fuel gauge, E to F, with a lamp lit while on mains power;
 *              with no battery, the fan speed or else the load average
 *   a brass plaque with the hostname engraved over two odometers:
 *   UPTIME     days, hours, minutes, seconds on rolling drums
 *   AWAY       how long the screen has been locked (swaylock's start time)
 *
 * Every instrument is drawn with signed distance functions, so it is crisp
 * at any size: a domed, brushed brass bezel with screws, an aged ivory face
 * with printed ticks, a needle with a counterweight and its shadow, and a
 * domed glass with its reflections. Numerals and labels come from a
 * freetype glyph atlas. The parts that never move (the panel, bezels, faces
 * and print) are drawn once into a texture; each frame only adds needles,
 * drums, lamps and glass over it, so a frame costs little.
 *
 * Sensors are read once a second; needles follow on springs (a little
 * overshoot, then settle) with a faint tremble at rest.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout shows the cluster above; the
 * others, left to right, take turns at
 *   CORE BANK    the load averages (1, 5, 15 minutes, an OVERLOAD lamp)
 *                over a grid of per-CPU load dials (SMT siblings paired
 *                past 24)
 *   AUXILIARIES  a chronograph of the time away, network RX/TX and disk
 *                read/write on twin-needle log dials (with a LINK lamp),
 *                the turning fans, and a rack of thermometers (CPU, GPU,
 *                NVMe, Wi-Fi, board... whatever hwmon offers)
 * each on its own stretch of the same steel and wood, with a title plaque.
 * Every instance reads its sensors on the same wall-clock second and the
 * springs run from the shared epoch, so the wall powers up and moves as one.
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
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
#include <sys/utsname.h>
#include "proto-layer-shell.h"

#define PI 3.14159265f
#define DEG (PI / 180.0f)

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "C059";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 25;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the instruments powered up at */

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;
enum { R_FRONT, R_BANK, R_AUX };
static int role = R_FRONT;

static void die(const char *what) { fprintf(stderr, "gauges: %s\n", what); exit(1); }

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

/* The side panels' sensors: discovered once, read with the rest */
#define MAXCPU 256
#define MAXTH 6
#define MAXFAN 2
static int nth, nfan;
static char th_path[MAXTH][300], th_name[MAXTH][16];
static char fanx_path[MAXFAN][300], fanx_name[MAXFAN][24];
static float fanx_top[MAXFAN];

/* a few thermometers: one per interesting hwmon device, best label first */
static void find_thermometers(void) {
    glob_t g;
    if (glob("/sys/class/hwmon/hwmon*/temp*_input", 0, NULL, &g)) return;
    struct { char path[300], name[16]; int pri, dev; } c[64];
    int nc = 0;
    for (size_t i = 0; i < g.gl_pathc && nc < 64; i++) {
        const char *in = g.gl_pathv[i];
        char dir[256], name[64], lpath[300], label[64];
        snprintf(dir, sizeof dir, "%.*s", (int)(strrchr(in, '/') - in), in);
        snprintf(lpath, sizeof lpath, "%s/name", dir);
        slurp_line(lpath, name, sizeof name);
        snprintf(lpath, sizeof lpath, "%.*s_label", (int)(strlen(in) - 6), in);
        slurp_line(lpath, label, sizeof label);
        long v = read_long(in, -1);
        if (v <= 0 || v > 150000) continue;
        const char *nm = NULL;
        int pri = 0;
        if (!strcmp(name, "coretemp") || !strcmp(name, "k10temp") || !strcmp(name, "zenpower")) {
            nm = "CPU"; pri = 100;
            if (!strncmp(label, "Package", 7) || !strcmp(label, "Tctl") || !strcmp(label, "Tdie")) pri = 101;
        } else if (!strcmp(name, "amdgpu") || !strcmp(name, "nouveau") || !strcmp(name, "radeon") || !strcmp(name, "i915") || !strcmp(name, "xe")) {
            nm = "GPU"; pri = 90 + (!label[0] || !strcmp(label, "edge"));
        } else if (!strcmp(name, "nvme")) {
            nm = "NVME"; pri = 80 + (!label[0] || !strcmp(label, "Composite"));
        } else if (!strcmp(name, "drivetemp")) { nm = "DISK"; pri = 75; }
        else if (!strncmp(name, "spd5118", 7) || !strcmp(name, "jc42")) { nm = "DIMM"; pri = 60; }
        else if (!strncmp(name, "iwlwifi", 7) || !strncmp(name, "mt79", 4) || !strncmp(name, "ath1", 4)) { nm = "WIFI"; pri = 55; }
        else if (!strncmp(name, "pch_", 4)) { nm = "CHIPSET"; pri = 50; }
        else if (!strncmp(name, "acpitz", 6)) { nm = "BOARD"; pri = 30; }
        else if (!strncmp(name, "nct", 3) || !strncmp(name, "it87", 4)) { nm = "BOARD"; pri = 35 + !strcmp(label, "SYSTIN"); }
        if (!nm) continue;
        /* one per device: keep its best-labelled sensor */
        int dev = atoi(dir + strlen("/sys/class/hwmon/hwmon")), k;
        for (k = 0; k < nc && c[k].dev != dev; k++) {}
        if (k < nc && c[k].pri >= pri) continue;
        if (k == nc) nc++;
        snprintf(c[k].path, sizeof c[k].path, "%s", in);
        snprintf(c[k].name, sizeof c[k].name, "%s", nm);
        c[k].pri = pri; c[k].dev = dev;
    }
    globfree(&g);
    /* best first, at most one board, then numbered where names repeat */
    for (int n = 0; n < MAXTH; n++) {
        int best = -1;
        for (int k = 0; k < nc; k++)
            if (c[k].pri >= 0 && (best < 0 || c[k].pri > c[best].pri)) best = k;
        if (best < 0) break;
        int same = 0;
        for (int j = 0; j < nth; j++) if (!strncmp(th_name[j], c[best].name, strlen(c[best].name))) same++;
        if (!(same && !strcmp(c[best].name, "BOARD"))) {
            snprintf(th_path[nth], sizeof th_path[nth], "%s", c[best].path);
            if (same) snprintf(th_name[nth], sizeof th_name[nth], "%s %d", c[best].name, same + 1);
            else snprintf(th_name[nth], sizeof th_name[nth], "%s", c[best].name);
            nth++;
        }
        c[best].pri = -1;
        if (nth == MAXTH) break;
    }
}

/* fans that are turning now, up to two */
static void find_fans(void) {
    glob_t g;
    if (glob("/sys/class/hwmon/hwmon*/fan*_input", 0, NULL, &g)) return;
    for (size_t i = 0; i < g.gl_pathc && nfan < MAXFAN; i++) {
        long v = read_long(g.gl_pathv[i], 0);
        if (v <= 0) continue;
        char lpath[300], label[64];
        snprintf(lpath, sizeof lpath, "%.*s_label", (int)(strlen(g.gl_pathv[i]) - 6), g.gl_pathv[i]);
        slurp_line(lpath, label, sizeof label);
        snprintf(fanx_path[nfan], sizeof fanx_path[nfan], "%s", g.gl_pathv[i]);
        if (label[0] && strlen(label) < 12) {
            for (char *p = label; *p; p++) *p = (char)toupper((unsigned char)*p);
            snprintf(fanx_name[nfan], sizeof fanx_name[nfan], "%s", label);
        } else snprintf(fanx_name[nfan], sizeof fanx_name[nfan], "FAN %d", nfan + 1);
        fanx_top[nfan] = fmaxf(3000, v * 1.6f);
        nfan++;
    }
    globfree(&g);
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
    /* the side panels' */
    float lavg[3];
    float core[MAXCPU];
    unsigned long long core_busy[MAXCPU], core_total[MAXCPU];
    double net_rx, net_tx, disk_r, disk_w;   /* bytes a second */
    unsigned long long net_c[2], disk_c[2];  /* the counters, at io_at */
    double io_at;
    int link;
    float th[MAXTH], fan[MAXFAN];
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

/* per-core load from the cpuN lines of /proc/stat */
static void read_cores(void) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        int n;
        unsigned long long v[8] = {0};
        if (strncmp(line, "cpu", 3) || !isdigit((unsigned char)line[3])) continue;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &n, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) < 5 ||
            n < 0 || n >= MAXCPU) continue;
        unsigned long long t = 0;
        for (int i = 0; i < 8; i++) t += v[i];
        unsigned long long b = t - v[3] - v[4];
        if (sens.core_total[n] && t > sens.core_total[n])
            sens.core[n] = (float)(b - sens.core_busy[n]) / (float)(t - sens.core_total[n]);
        sens.core_busy[n] = b; sens.core_total[n] = t;
    }
    fclose(f);
}

/* is this a real device (not lo, a bridge, a tunnel, a container's veth)? */
static int physical_if(const char *name) {
    char p[300];
    snprintf(p, sizeof p, "/sys/class/net/%.64s/device", name);
    return access(p, F_OK) == 0;
}

static void read_io(void) {
    unsigned long long rx = 0, tx = 0, rd = 0, wr = 0;
    int link = 0;
    char line[512];
    FILE *f = fopen("/proc/net/dev", "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            char *c = strchr(line, ':');
            if (!c) continue;
            *c = 0;
            char *name = line;
            while (*name == ' ') name++;
            if (!physical_if(name)) continue;
            unsigned long long v[9];
            if (sscanf(c + 1, "%llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]) == 9) {
                rx += v[0]; tx += v[8];
            }
            char p[300], st[32];
            snprintf(p, sizeof p, "/sys/class/net/%.64s/operstate", name);
            slurp_line(p, st, sizeof st);
            if (!strcmp(st, "up")) link = 1;
        }
        fclose(f);
    }
    if ((f = fopen("/proc/diskstats", "r"))) {
        while (fgets(line, sizeof line, f)) {
            char name[64], p[300];
            unsigned long long v[7];
            if (sscanf(line, "%*u %*u %63s %llu %llu %llu %llu %llu %llu %llu", name, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) != 8) continue;
            snprintf(p, sizeof p, "/sys/block/%s/device", name);
            if (access(p, F_OK)) continue;          /* partitions, loops, device-mapper */
            rd += v[2] * 512; wr += v[6] * 512;
        }
        fclose(f);
    }
    double t = now(), dt = t - sens.io_at;
    if (sens.io_at > 0 && dt > 0.05) {
        sens.net_rx = (rx - sens.net_c[0]) / dt; sens.net_tx = (tx - sens.net_c[1]) / dt;
        sens.disk_r = (rd - sens.disk_c[0]) / dt; sens.disk_w = (wr - sens.disk_c[1]) / dt;
    }
    sens.net_c[0] = rx; sens.net_c[1] = tx; sens.disk_c[0] = rd; sens.disk_c[1] = wr;
    sens.io_at = t;
    sens.link = link;
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

    if (role == R_BANK) {
        read_cores();
        slurp_line("/proc/loadavg", buf, sizeof buf);
        sscanf(buf, "%f %f %f", &sens.lavg[0], &sens.lavg[1], &sens.lavg[2]);
    } else if (role == R_AUX) {
        read_io();
        for (int i = 0; i < nth; i++) sens.th[i] = read_long(th_path[i], 0) / 1000.0f;
        for (int i = 0; i < nfan; i++) sens.fan[i] = (float)read_long(fanx_path[i], 0);
    }
}

/* ---- the instruments ---- */
enum { G_TEMP, G_LOAD, G_MEM, G_AUX, G_CLOCK, NG_FRONT };
#define MAXG 32
/* kinds: a dial with a needle (or two), the clock, a chronograph of the
 * time away, a thermometer (r is its mount's half-height) */
enum { K_DIAL, K_CLOCK, K_CHRONO, K_THERMO };
/* where a needle's reading comes from */
enum { S_TEMP, S_LOAD, S_MEM, S_AUX, S_CORE, S_LAVG, S_NET, S_DISK, S_FAN, S_THERM, S_NONE };
/* how the scale is numbered */
enum { SC_LIN, SC_FUEL, SC_LOG };

typedef struct {
    int kind, src, idx, scale;
    int dual;                          /* a second (red) needle */
    int hot;                           /* numerals in the red zone printed red */
    float txt;                         /* the label's size, relative */
    float cx, cy, r;                   /* px: centre, face radius */
    float a0, a1;                      /* the sweep: radians clockwise from 12 o'clock */
    float vmin, vmax;
    int nmaj, nmin, numevery;          /* major intervals, minor per major, a numeral every n majors */
    float red0, red1, amb0, amb1;      /* painted zones, as fractions of the scale (< 0: none) */
    char label[24], unit[24];
    float omega, zeta;                 /* needle spring */
    float x, v, target;                /* needle state, fraction of the scale */
    float x2, v2, target2;             /* the second needle */
    float ph[3];                       /* tremble phases */
} Gauge;
static Gauge G[MAXG];
static int NG = NG_FRONT;

/* a "nice" full-scale and major step for 0..v */
static void nice_scale(float v, float *top, int *nmaj) {
    static const float steps[] = {0.5f, 1, 2, 2.5f, 4, 5, 10, 20, 25, 50, 100};
    for (size_t i = 0; i < sizeof steps / sizeof *steps; i++) {
        int n = (int)ceilf(v / steps[i] - 1e-3f);
        if (n <= 8) { *top = n * steps[i]; *nmaj = n < 4 ? 4 : n; if (n < 4) *top = 4 * steps[i]; return; }
    }
    *top = v; *nmaj = 5;
}

static Gauge *add_gauge(int kind, int src, int idx, const char *label, const char *unit) {
    Gauge *g = &G[NG++];
    g->kind = kind; g->src = src; g->idx = idx;
    snprintf(g->label, sizeof g->label, "%s", label);
    snprintf(g->unit, sizeof g->unit, "%s", unit);
    return g;
}

/* load-average and fan dials, shared by the panels */
static void lavg_scale(Gauge *g) {
    nice_scale((float)ncpu, &g->vmax, &g->nmaj);
    g->nmin = 5;
    g->red0 = ncpu / g->vmax; g->red1 = 1;
}

/* bytes a second on decades from 100 B/s to 10^top */
static void log_scale(Gauge *g, int top) {
    g->scale = SC_LOG; g->vmin = 2; g->vmax = (float)top;
    g->nmaj = top - 2; g->nmin = 2; g->numevery = 1;
    g->dual = 1;
    g->omega = 2 * PI * 0.9f; g->zeta = 0.4f;
}

static int core_groups;                /* core dials on the bank; each shows ncpu / core_groups CPUs */

static void gauges_init(void) {
    Gauge *g;
    NG = 0;
    if (role == R_FRONT) {
        add_gauge(K_DIAL, S_TEMP, 0, "CPU TEMP", "\x7f" "C");
        add_gauge(K_DIAL, S_LOAD, 0, "CPU LOAD", "PER CENT");
        add_gauge(K_DIAL, S_MEM, 0, "MEMORY", "GB");
        add_gauge(K_DIAL, S_AUX, 0, "", "");
        add_gauge(K_CLOCK, S_NONE, 0, "", "");
    } else if (role == R_BANK) {
        static const char *u[3] = {"1 MINUTE", "5 MINUTES", "15 MINUTES"};
        for (int i = 0; i < 3; i++) add_gauge(K_DIAL, S_LAVG, i, "LOAD AVG", u[i]);
        core_groups = ncpu;
        while (core_groups > 24) core_groups = (core_groups + 1) / 2;
        for (int i = 0; i < core_groups; i++) {
            char b[24];
            snprintf(b, sizeof b, "%s %d", core_groups == ncpu ? "CPU" : "CORE", i);
            add_gauge(K_DIAL, S_CORE, i, b, "");
        }
    } else {
        add_gauge(K_CHRONO, S_NONE, 0, "CHRONOGRAPH", "TIME AWAY");
        add_gauge(K_DIAL, S_NET, 0, "NETWORK", "BYTES / SEC");
        add_gauge(K_DIAL, S_DISK, 0, "DISKS", "BYTES / SEC");
        for (int i = 0; i < nfan; i++) add_gauge(K_DIAL, S_FAN, i, fanx_name[i], "RPM x 1000");
        for (int i = 0; i < nth; i++) add_gauge(K_THERMO, S_THERM, i, th_name[i], "\x7f" "C");
    }
    for (int i = 0; i < NG; i++) {
        G[i].a0 = -135 * DEG; G[i].a1 = 135 * DEG;
        G[i].red0 = G[i].red1 = G[i].amb0 = G[i].amb1 = -1;
        G[i].numevery = 1; G[i].nmin = 5; G[i].txt = 1;
        G[i].omega = 2 * PI * 0.8f; G[i].zeta = 0.42f;
        G[i].x = G[i].x2 = -0.03f;         /* resting on the stop pin */
        for (int k = 0; k < 3; k++) G[i].ph[k] = frand() * 2 * PI;
    }
    for (int i = 0; i < NG; i++) {
        g = &G[i];
        switch (g->src) {
        case S_TEMP:
            g->vmin = 20; g->vmax = 100; g->nmaj = 8; g->nmin = 5; g->numevery = 1;
            g->red0 = (85 - 20) / 80.0f; g->red1 = 1; g->hot = 1;
            g->omega = 2 * PI * 0.45f; g->zeta = 0.5f;
            break;
        case S_LOAD:
            g->vmin = 0; g->vmax = 100; g->nmaj = 10; g->nmin = 2; g->numevery = 2;
            g->amb0 = 0.8f; g->amb1 = 1;
            g->omega = 2 * PI * 1.0f; g->zeta = 0.34f;
            break;
        case S_MEM:
            nice_scale(sens.mem_total_gb, &g->vmax, &g->nmaj);
            g->vmin = 0; g->nmin = 4; g->numevery = g->nmaj > 6 ? 2 : 1;
            g->amb0 = sens.mem_total_gb * 0.9f / g->vmax; g->amb1 = sens.mem_total_gb / g->vmax;
            g->omega = 2 * PI * 0.6f; g->zeta = 0.45f;
            break;
        case S_AUX:
            g->vmin = 0; g->omega = 2 * PI * 0.5f; g->zeta = 0.5f;
            if (aux_kind == AUX_BATT) {
                /* a fuel gauge: E to F over the top */
                g->a0 = -60 * DEG; g->a1 = 60 * DEG;
                g->vmax = 100; g->nmaj = 4; g->nmin = 2; g->numevery = 2;
                g->red0 = 0; g->red1 = 0.15f; g->scale = SC_FUEL;
                snprintf(g->label, sizeof g->label, "BATTERY");
                snprintf(g->unit, sizeof g->unit, "CHARGE");
            } else if (aux_kind == AUX_FAN) {
                nice_scale(fmaxf(6000, sens.aux * 1.3f) / 1000, &g->vmax, &g->nmaj);
                g->nmin = 5;
                snprintf(g->label, sizeof g->label, "FAN");
                snprintf(g->unit, sizeof g->unit, "RPM x 1000");
            } else {
                lavg_scale(g);
                snprintf(g->label, sizeof g->label, "LOAD AVG");
                snprintf(g->unit, sizeof g->unit, "1 MINUTE");
            }
            break;
        case S_LAVG:
            lavg_scale(g);
            g->omega = 2 * PI * (0.35f - 0.08f * g->idx); g->zeta = 0.6f;
            break;
        case S_CORE:
            g->vmin = 0; g->vmax = 100; g->nmaj = 4; g->nmin = 5; g->numevery = 1;
            g->amb0 = 0.8f; g->amb1 = 1; g->txt = 1.55f;
            g->omega = 2 * PI * (1.0f + 0.15f * frand()); g->zeta = 0.3f;
            break;
        case S_NET: log_scale(g, 9); break;
        case S_DISK: log_scale(g, 10); break;
        case S_FAN:
            nice_scale(fanx_top[g->idx] / 1000, &g->vmax, &g->nmaj);
            g->vmin = 0; g->nmin = 5;
            g->omega = 2 * PI * 0.5f; g->zeta = 0.5f;
            break;
        case S_THERM:
            g->vmin = 0; g->vmax = 100; g->nmaj = 10; g->nmin = 2; g->numevery = 2;
            g->red0 = 0.85f; g->red1 = 1;
            g->omega = 2 * PI * 0.3f; g->zeta = 0.8f;      /* mercury: slow, no bounce */
            break;
        }
        if (g->kind == K_CLOCK || g->kind == K_CHRONO) {
            g->a0 = 0; g->a1 = 2 * PI; g->nmaj = 12; g->nmin = 5;
            g->omega = 2 * PI * 7.0f; g->zeta = 0.3f;    /* the seconds hand: a quick tick with a bounce */
        }
    }
}

static float logx(double rate, const Gauge *g) {
    return ((float)log10(rate > 1 ? rate : 1) - g->vmin) / (g->vmax - g->vmin);
}

/* the needles' targets from the latest readings */
static void set_targets(void) {
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        switch (g->src) {
        case S_TEMP: case S_THERM:
            g->target = ((g->src == S_TEMP ? sens.temp : sens.th[g->idx]) - g->vmin) / (g->vmax - g->vmin);
            break;
        case S_LOAD: g->target = sens.load; break;
        case S_MEM: g->target = sens.mem_gb / g->vmax; break;
        case S_AUX:
            if (aux_kind == AUX_BATT) g->target = sens.aux / 100.0f;
            else if (aux_kind == AUX_FAN) g->target = sens.aux / 1000.0f / g->vmax;
            else g->target = sens.aux / g->vmax;
            break;
        case S_LAVG: g->target = sens.lavg[g->idx] / g->vmax; break;
        case S_CORE: {
            float s = 0;
            int n = 0;
            for (int c = g->idx; c < ncpu && c < MAXCPU; c += core_groups) { s += sens.core[c]; n++; }
            g->target = n ? s / n : 0;
            break;
        }
        case S_NET: g->target = logx(sens.net_rx, g); g->target2 = logx(sens.net_tx, g); break;
        case S_DISK: g->target = logx(sens.disk_r, g); g->target2 = logx(sens.disk_w, g); break;
        case S_FAN: g->target = sens.fan[g->idx] / 1000.0f / g->vmax; break;
        default: break;
        }
        g->target = clampf(g->target, -0.03f, 1.03f);
        g->target2 = clampf(g->target2, -0.03f, 1.03f);
    }
}

static void spring(Gauge *g, float *x, float *v, float tgt, float dt) {
    float a = g->omega * g->omega * (tgt - *x) - 2 * g->zeta * g->omega * *v;
    *v += a * dt;
    *x += *v * dt;
    float lo = g->kind == K_THERMO ? -0.06f : -0.035f;
    if (*x < lo) { *x = lo; *v = -0.3f * *v; }
    if (*x > 1.035f) { *x = 1.035f; *v = -0.3f * *v; }
}

/* one step of the needle springs; the stop pins at either end bounce them */
static float sec_angle, sec_vel;     /* the clock's seconds hand, degrees */
static void physics(double t, float dt, int powered) {
    int clock = -1;
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        if (g->kind == K_CLOCK) { clock = i; continue; }
        if (g->kind == K_CHRONO) continue;
        /* unpowered, the thermometers sit at the room's temperature */
        float rest = g->kind == K_THERMO ? 0.2f : -0.03f;
        spring(g, &g->x, &g->v, powered ? g->target : rest, dt);
        if (g->dual) spring(g, &g->x2, &g->v2, powered ? g->target2 : -0.03f, dt);
    }
    if (clock < 0) return;
    /* the seconds hand jumps to each new second and rings a little */
    double tt = t;
    struct tm tm;
    time_t ts = (time_t)tt;
    localtime_r(&ts, &tm);
    float tgt = tm.tm_sec * 6.0f;
    float d = fmodf(tgt - sec_angle + 540.0f, 360.0f) - 180.0f;
    Gauge *c = &G[clock];
    float a = c->omega * c->omega * d - 2 * c->zeta * c->omega * sec_vel;
    sec_vel += a * dt;
    sec_angle = fmodf(sec_angle + sec_vel * dt + 360.0f, 360.0f);
}

/* the needle's angle as drawn, with a faint tremble */
static float needle_angle(const Gauge *g, float x, double t) {
    float j = 0.0012f * (sinf((float)t * 7.1f + g->ph[0]) * 0.5f + sinf((float)t * 11.3f + g->ph[1]) * 0.3f +
                         sinf((float)t * 17.9f + g->ph[2]) * 0.2f);
    x = clampf(x + j, -0.035f, 1.035f);
    return g->a0 + (g->a1 - g->a0) * x;
}

/* ---- glyph atlas: printable ASCII (+ the degree sign as 0x7f) in each
 * text style used, rasterised at its exact pixel size ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 64
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* regular, bold */
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < 2; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:weight=%s:fontformat=CFF,TrueType", opt_font, i ? "bold" : "regular");
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

/* ---- layout ---- */
static float plate[4];                 /* the steel plate: x0 y0 x1 y1 */
static float plq[4];                   /* the brass plaque */
static float rule_l[4];                /* the engraved rule under the hostname: x0 y x1 half-thickness */
#define NDRUM 17
static float drum_rect[NDRUM][4];
static float win_rect[2][4];
static int drum_spec[NDRUM];           /* which digit of the time each drum shows, see digit_of() */
static int drum_style;                 /* text style of the drum digits */
static int nlamp;

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

/* an instrument's outline (bezel or mount) in units of its r */
static float ext_x(int i) { return G[i].kind == K_THERMO ? 0.26f : 1.22f; }
static float ext_y(int i) { return G[i].kind == K_THERMO ? 1.0f : 1.22f; }

/* The side panels are laid out as rows of instruments, each row centred;
 * an item is a gauge index or -1 for the plaque. */
#define MAXROW 30
#define MAXITEM 16
typedef struct { int n, it[MAXITEM]; } Row;
typedef struct { int n; Row r[MAXROW]; } Rows;

static void row_add(Rows *R, int item) {
    Row *w = &R->r[R->n - 1];
    if (w->n < MAXITEM) w->it[w->n++] = item;
}
static void row_new(Rows *R) { if (R->n < MAXROW) R->r[R->n++].n = 0; }

/* place the rows (centred on 0, 0) into d / the plaque; returns the extent */
static void stack_rows(const Rows *R, Disc *d, float pqw, float pqh, float *pqx, float *pqy, float *bw, float *bh) {
    const float gx = 0.32f, gy = 0.3f;
    float hw[MAXROW], hh[MAXROW], tot_h = 0, max_w = 0;
    for (int r = 0; r < R->n; r++) {
        const Row *w = &R->r[r];
        hw[r] = 0; hh[r] = 0;
        for (int k = 0; k < w->n; k++) {
            int i = w->it[k];
            float ex = i < 0 ? pqw / 2 : ext_x(i) * d[i].r, ey = i < 0 ? pqh / 2 : ext_y(i) * d[i].r;
            hw[r] += 2 * ex + (k ? gx : 0);
            hh[r] = fmaxf(hh[r], ey);
        }
        tot_h += 2 * hh[r] + (r ? gy : 0);
        max_w = fmaxf(max_w, hw[r]);
    }
    float y = -tot_h / 2;
    for (int r = 0; r < R->n; r++) {
        const Row *w = &R->r[r];
        float x = -hw[r] / 2, cy = y + hh[r];
        for (int k = 0; k < w->n; k++) {
            int i = w->it[k];
            float ex = i < 0 ? pqw / 2 : ext_x(i) * d[i].r;
            if (i < 0) { *pqx = x + ex; *pqy = cy; }
            else { d[i].x = x + ex; d[i].y = cy; }
            x += 2 * ex + gx;
        }
        y += 2 * hh[r] + gy;
    }
    *bw = max_w; *bh = tot_h;
}

/* the largest scale that fits these rows into aw x ah */
static float rows_fit(const Rows *R, Disc *d, float pqw, float pqh, float aw, float ah) {
    float px, py, bw, bh;
    stack_rows(R, d, pqw, pqh, &px, &py, &bw, &bh);
    return fminf(aw / bw, ah / bh);
}

/* the side panels' arrangement: try a few, keep the one whose instruments come out biggest */
static void layout_rows(Disc *d, float pqw, float pqh, float aw, float ah, float *pqx, float *pqy) {
    Rows best = {0}, R;
    float bs = -1, rc = 1, best_rc = 1, pen = 1;
    /* the bank: as big a core dial as fits, up to a little under the load
     * dials, and rather a full grid than a ragged last row */
#define TRY() do { float s_ = rows_fit(&R, d, pqw, pqh, aw, ah) * fminf(rc, 1.2f) * pen; if (s_ > bs * 1.01f) { bs = s_; best = R; best_rc = rc; } } while (0)
    if (role == R_BANK) {
        int n = NG - 3;
        for (rc = 0.8f; rc < 1.21f; rc += 0.1f)
        for (int cols = 1; cols <= n; cols++) {
            for (int i = 3; i < NG; i++) d[i].r = rc;
            pen = n % cols ? 0.88f : 1;
            for (int v = 0; v < 2; v++) {
                R.n = 0;
                row_new(&R);
                for (int i = 0; i < 3; i++) row_add(&R, i);
                if (v) row_add(&R, -1);
                for (int c = 0; c < n; c++) {
                    if (c % cols == 0) row_new(&R);
                    row_add(&R, 3 + c);
                }
                if (!v) { row_new(&R); row_add(&R, -1); }
                TRY();
            }
        }
    } else {
        /* 0 chronograph, 1 network, 2 disks, then fans, then thermometers */
        int f0 = 3, t0 = 3 + nfan;
        /* the plaque among the thermometers, a fan at either end
         * (tried first: it wins a tie) */
        for (int v = 0; v < 2; v++) {
            R.n = 0;
            row_new(&R);
            if (!v) { row_add(&R, 1); row_add(&R, 0); row_add(&R, 2); }
            else { row_add(&R, 0); row_new(&R); row_add(&R, 1); row_add(&R, 2); }
            row_new(&R);
            if (nfan) row_add(&R, f0);
            for (int i = 0; i < nth / 2; i++) row_add(&R, t0 + i);
            row_add(&R, -1);
            for (int i = nth / 2; i < nth; i++) row_add(&R, t0 + i);
            if (nfan > 1) row_add(&R, f0 + 1);
            TRY();
        }
        for (int v = 0; v < 4; v++) {
            R.n = 0;
            row_new(&R);
            if (v < 2) { row_add(&R, 1); row_add(&R, 0); row_add(&R, 2); }
            else { row_add(&R, 0); row_new(&R); row_add(&R, 1); row_add(&R, 2); }
            row_new(&R);
            if (nfan) row_add(&R, f0);
            for (int i = 0; i < nth; i++) row_add(&R, t0 + i);
            if (v & 1) { if (nfan > 1) row_add(&R, f0 + 1); row_add(&R, -1); }
            else { if (nfan > 1) row_add(&R, f0 + 1); row_new(&R); row_add(&R, -1); }
            if (R.r[1].n == 0 && v < 2) continue;
            TRY();
        }
    }
#undef TRY
    for (int i = 3; role == R_BANK && i < NG; i++) d[i].r = best_rc;
    float bw, bh;
    stack_rows(&best, d, pqw, pqh, pqx, pqy, &bw, &bh);
}

static int pq_title;                   /* the side panels' plaque: just a title, no drums */
static float lamp_pos[4][3];           /* x, y, r */
static int lamp_kind[4], lamp_g[4];
enum { L_HOT, L_MAINS, L_OVER, L_LINK };

static void add_lamp(int kind, int g, float dy, float rf) {
    if (nlamp == 4) return;
    lamp_kind[nlamp] = kind; lamp_g[nlamp] = g;
    lamp_pos[nlamp][0] = G[g].cx; lamp_pos[nlamp][1] = G[g].cy + dy * G[g].r;
    lamp_pos[nlamp][2] = rf * G[g].r;
    nlamp++;
}

/* thermometer geometry, in units of its half-height r: the scale from
 * vmin at TH_Y0 up to vmax at TH_Y1 */
#define TH_Y0 0.56f
#define TH_Y1 (-0.66f)
#define TH_TX (-0.035f)                /* the tube's centre line */

static void upper(char *dst, const char *src, size_t n) {
    size_t i = 0;
    for (; src[i] && i + 1 < n; i++) dst[i] = (char)toupper((unsigned char)src[i]);
    dst[i] = 0;
}

static void layout(void) {
    float m = (float)(W < H ? W : H);
    float frame = roundf(0.026f * m);
    plate[0] = frame; plate[1] = frame; plate[2] = W - frame; plate[3] = H - frame;
    float inset = 0.05f * m;
    float ax0 = plate[0] + inset, ay0 = plate[1] + inset, ax1 = plate[2] - inset, ay1 = plate[3] - inset;

    /* design positions, in units; the plaque height in the same units */
    int tall = W < H * 1.15f;
    Disc d[MAXG];
    float pqx = 0, pqy = 0, pqh = 1.95f, pqw;
    float k = pqh / 10;
    float dw = 1.3f * k, dgap = 0.07f * k, ggap = 0.5f * k, wpad = 0.16f * k;
    pq_title = role != R_FRONT;
    if (role == R_FRONT) {
        if (!tall) {
            d[G_CLOCK] = (Disc){8.0f, 3.6f, 2.3f};
            d[G_TEMP] = (Disc){2.95f, 3.35f, 1.72f};
            d[G_LOAD] = (Disc){13.05f, 3.35f, 1.72f};
            d[G_MEM] = (Disc){0, 7.45f, 1.12f};
            d[G_AUX] = (Disc){0, 7.45f, 1.12f};
            pqx = 8.0f; pqy = 7.6f;
        } else {
            d[G_CLOCK] = (Disc){4.5f, 3.2f, 2.4f};
            d[G_TEMP] = (Disc){2.25f, 7.85f, 1.7f};
            d[G_LOAD] = (Disc){6.75f, 7.85f, 1.7f};
            d[G_MEM] = (Disc){2.25f, 11.45f, 1.15f};
            d[G_AUX] = (Disc){6.75f, 11.45f, 1.15f};
            pqx = 4.5f; pqy = 14.3f;
        }
        /* the plaque, in plaque-height tenths (k): label column, window of drums */
        float upw = 9 * dw + 5 * dgap + 3 * ggap + 2 * wpad;
        pqw = 2.0f * k + 5.4f * k + 0.6f * k + upw + 2.0f * k;
        if (!tall) {
            float gap = 0.3f;
            d[G_MEM].x = pqx - pqw / 2 - gap - 1.22f * d[G_MEM].r;
            d[G_AUX].x = pqx + pqw / 2 + gap + 1.22f * d[G_AUX].r;
        }
    } else {
        pqh = 1.45f; pqw = 6.2f;
        for (int i = 0; i < NG; i++) {
            Gauge *g = &G[i];
            d[i].r = g->kind == K_CHRONO ? 2.15f : g->kind == K_THERMO ? 1.75f : g->src == S_CORE ? 1.0f
                   : g->src == S_LAVG ? 1.3f : g->src == S_FAN ? 1.1f : 1.5f;
        }
        layout_rows(d, pqw, pqh, ax1 - ax0, ay1 - ay0, &pqx, &pqy);
    }

    /* fit the bounding box into the plate, spreading a little if there's room */
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    for (int i = 0; i < NG; i++) {
        float rx = d[i].r * ext_x(i), ry = d[i].r * ext_y(i);
        bx0 = fminf(bx0, d[i].x - rx); bx1 = fmaxf(bx1, d[i].x + rx);
        by0 = fminf(by0, d[i].y - ry); by1 = fmaxf(by1, d[i].y + ry);
    }
    bx0 = fminf(bx0, pqx - pqw / 2); bx1 = fmaxf(bx1, pqx + pqw / 2);
    by0 = fminf(by0, pqy - pqh / 2); by1 = fmaxf(by1, pqy + pqh / 2);
    float bw = bx1 - bx0, bh = by1 - by0;
    float s = fminf((ax1 - ax0) / bw, (ay1 - ay0) / bh);
    float sx = fminf(1.25f, (ax1 - ax0) / (bw * s)), sy = fminf(1.3f, (ay1 - ay0) / (bh * s));
    float mx = (bx0 + bx1) / 2, my = (by0 + by1) / 2;
    float ox = (ax0 + ax1) / 2, oy = (ay0 + ay1) / 2;
#define PX(v) (ox + ((v) - mx) * s * sx)
#define PY(v) (oy + ((v) - my) * s * sy)
    for (int i = 0; i < NG; i++) {
        G[i].cx = PX(d[i].x); G[i].cy = PY(d[i].y); G[i].r = d[i].r * s;
    }
    float kp = k * s;                  /* plaque tenth, px */
    float pcx = PX(pqx), pcy = PY(pqy);
    plq[0] = roundf(pcx - pqw * s / 2); plq[1] = roundf(pcy - pqh * s / 2);
    plq[2] = roundf(pcx + pqw * s / 2); plq[3] = roundf(pcy + pqh * s / 2);

    /* inside the plaque: hostname, rule, captions, two windows of drums */
    nstyles = 0;
    char hn[64];
    upper(hn, host, sizeof hn);
    int st_name, st_lab = 0, st_cap = 0, st_sub = 0;
    char title[64], sub[96];
    if (!pq_title) {
        /* the hostname as large as fits between the corner screws */
        float name_px = 2.0f * kp;
        while (name_px > 8 && measure(1, name_px, hn, 0.22f) > (plq[2] - plq[0]) - 5.0f * kp) name_px *= 0.94f;
        st_name = style(1, name_px); st_lab = style(1, 1.0f * kp); st_cap = style(0, 0.78f * kp);
        float rx1 = plq[2] - 2.0f * kp;   /* the windows' right edge */
        float wy[2] = {plq[1] + 4.6f * kp, plq[1] + 7.35f * kp}, wh = 2.05f * kp;
        int counts[2] = {9, 8};
        int nd = 0;
        for (int w = 0; w < 2; w++) {
            float x = rx1 - wpad * s;
            int groups = 4;
            int per[4] = {counts[w] == 9 ? 3 : 2, 2, 2, 2};
            int specs[4][3] = {{D_D100, D_D10, D_D1}, {D_H10, D_H1}, {D_M10, D_M1}, {D_S10, D_S1}};
            if (counts[w] == 8) { specs[0][0] = D_D10; specs[0][1] = D_D1; }
            /* lay out right to left */
            float right = x;
            for (int gidx = groups - 1; gidx >= 0; gidx--) {
                for (int j = per[gidx] - 1; j >= 0; j--) {
                    float x1 = right, x0 = right - dw * s;
                    drum_rect[nd][0] = roundf(x0); drum_rect[nd][2] = roundf(x1);
                    drum_rect[nd][1] = roundf(wy[w] + wpad * s); drum_rect[nd][3] = roundf(wy[w] + wh - wpad * s);
                    drum_spec[nd] = specs[gidx][j];
                    nd++;
                    right = x0 - dgap * s;
                }
                right += dgap * s;
                if (gidx) right -= ggap * s;
            }
            win_rect[w][0] = roundf(right - wpad * s); win_rect[w][1] = roundf(wy[w]);
            win_rect[w][2] = roundf(rx1); win_rect[w][3] = roundf(wy[w] + wh);
        }
    } else {
        /* a title, a rule, and the host and what this panel is under it */
        if (role == R_BANK) {
            snprintf(title, sizeof title, "CORE BANK");
            snprintf(sub, sizeof sub, "%s  -  %d THREADS", hn, ncpu);
        } else {
            struct utsname u;
            char rel[32] = "";
            if (uname(&u) == 0) {
                snprintf(rel, sizeof rel, "%.31s", u.release);
                rel[strspn(rel, "0123456789.")] = 0;
                size_t l = strlen(rel);
                if (l && rel[l - 1] == '.') rel[l - 1] = 0;
            }
            snprintf(title, sizeof title, "AUXILIARIES");
            snprintf(sub, sizeof sub, "%s  -  LINUX %s", hn, rel);
        }
        float tk = pqh * s / 10;       /* this plaque's tenth */
        float name_px = 3.4f * tk;
        while (name_px > 8 && measure(1, name_px, title, 0.22f) > (plq[2] - plq[0]) - 2.6f * tk) name_px *= 0.94f;
        st_name = style(1, name_px);
        float sub_px = 1.45f * tk;
        while (sub_px > 7 && measure(0, sub_px, sub, 0.18f) > (plq[2] - plq[0]) - 2.6f * tk) sub_px *= 0.94f;
        st_sub = style(0, sub_px);
        for (int j = 0; j < NDRUM; j++) {
            drum_rect[j][0] = drum_rect[j][1] = -100; drum_rect[j][2] = drum_rect[j][3] = -90;
            drum_spec[j] = D_S1;
        }
        for (int w = 0; w < 2; w++) { win_rect[w][0] = win_rect[w][1] = -100; win_rect[w][2] = win_rect[w][3] = -90; }
    }
    drum_style = style(0, (drum_rect[0][3] - drum_rect[0][1]) * 0.7f);

    /* text styles for the instruments */
    int st_num[MAXG], st_glab[MAXG], st_unit[MAXG], st_mark[MAXG];
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        float r = g->r;
        if (g->kind == K_THERMO) {
            st_num[i] = style(0, 0.062f * r);
            st_glab[i] = style(1, 0.07f * r);
            st_unit[i] = st_mark[i] = style(0, 0.05f * r);
            continue;
        }
        st_num[i] = style(0, (g->kind != K_DIAL ? 0.17f : g->scale == SC_FUEL ? 0.19f : g->src == S_CORE ? 0.16f
                              : g->scale == SC_LOG ? 0.115f : 0.145f) * r);
        st_glab[i] = style(1, 0.092f * g->txt * r);
        st_unit[i] = style(0, 0.082f * r);
        st_mark[i] = style(0, 0.058f * r);
    }
    build_atlas();

    /* lamps */
    nlamp = 0;
    for (int i = 0; i < NG; i++) {
        if (G[i].src == S_TEMP) add_lamp(L_HOT, i, 0.55f, 0.058f);
        if (G[i].src == S_AUX && aux_kind == AUX_BATT) add_lamp(L_MAINS, i, 0.5f, 0.075f);
        if (G[i].src == S_LAVG && G[i].idx == 0) add_lamp(L_OVER, i, 0.62f, 0.06f);
        if (G[i].src == S_NET) add_lamp(L_LINK, i, 0.6f, 0.05f);
    }

    /* the print: numerals and labels on the faces, engraving on the plaque */
    tvn = 0;
    static const float ink[4] = {0.11f, 0.085f, 0.065f, 0.93f}, red[4] = {0.55f, 0.18f, 0.13f, 0.95f};
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        float r = g->r;
        if (g->kind == K_CLOCK || g->kind == K_CHRONO) {
            int chrono = g->kind == K_CHRONO;
            for (int h = 1; h <= 12; h++) {
                char b[8];
                snprintf(b, sizeof b, "%d", chrono ? h * 5 : h);
                float a = h * 30 * DEG;
                text_c(st_num[i], g->cx + sinf(a) * 0.68f * r, g->cy - cosf(a) * 0.68f * r, b, ink, 0);
            }
            text_c(st_mark[i], g->cx, g->cy - 0.33f * r, chrono ? "CHRONOGRAPH" : "AGED BRASS & CO.", ink, 0.18f * styles[st_mark[i]].px);
            text_c(st_unit[i], g->cx, g->cy + 0.34f * r, chrono ? "TIME AWAY" : "LOCAL TIME", ink, 0.2f * styles[st_unit[i]].px);
            continue;
        }
        if (g->kind == K_THERMO) {
            /* numerals right of the tube, the name at the top, the unit at the bottom */
            for (int k2 = 0; k2 <= g->nmaj; k2 += g->numevery) {
                char b[16];
                snprintf(b, sizeof b, "%g", g->vmin + (g->vmax - g->vmin) * k2 / g->nmaj);
                float y = g->cy + (TH_Y0 + (TH_Y1 - TH_Y0) * k2 / g->nmaj) * r;
                text(st_num[i], g->cx + 0.035f * r, y + cap_h(st_num[i]) * 0.5f, -1, b,
                     (float)k2 / g->nmaj >= g->red0 - 1e-3f ? red : ink, 0);
            }
            text_c(st_glab[i], g->cx, g->cy - 0.785f * r, g->label, ink, 0.12f * styles[st_glab[i]].px);
            text_c(st_unit[i], g->cx + 0.09f * r, g->cy + 0.72f * r, g->unit, ink, 0);
            continue;
        }
        int fuel = g->scale == SC_FUEL;
        float nr = fuel ? 0.62f : 0.655f;
        for (int k2 = 0; k2 <= g->nmaj; k2 += g->numevery) {
            char b[16];
            float v = g->vmin + (g->vmax - g->vmin) * k2 / g->nmaj;
            if (fuel) snprintf(b, sizeof b, "%s", k2 == 0 ? "E" : k2 == g->nmaj ? "F" : k2 * 2 == g->nmaj ? "1/2" : "");
            else if (g->scale == SC_LOG) {
                static const char *dec[] = {"1", "10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1G", "10G", "100G"};
                snprintf(b, sizeof b, "%s", dec[(int)lroundf(v)]);
            } else snprintf(b, sizeof b, "%g", v);
            if (!b[0]) continue;
            float a = g->a0 + (g->a1 - g->a0) * k2 / g->nmaj;
            int over = g->hot && g->red0 >= 0 && (float)k2 / g->nmaj >= g->red0 - 1e-3f && (float)k2 / g->nmaj <= g->red1 + 1e-3f;
            float rr = g->scale == SC_LOG ? 0.67f : nr;
            text_c(st_num[i], g->cx + sinf(a) * rr * r, g->cy - cosf(a) * rr * r, b, over ? red : ink, 0);
        }
        float lt = 0.14f * styles[st_glab[i]].px;
        if (fuel) {
            text_c(st_glab[i], g->cx, g->cy + 0.27f * r, g->label, ink, lt);
            text_c(st_mark[i], g->cx, g->cy + 0.69f * r, "MAINS", ink, 0.2f * styles[st_mark[i]].px);
        } else if (g->src == S_CORE) {
            text_c(st_glab[i], g->cx, g->cy - 0.3f * r, g->label, ink, lt);
        } else {
            text_c(st_glab[i], g->cx, g->cy - 0.33f * r, g->label, ink, lt);
            text_c(st_unit[i], g->cx, g->cy + (g->dual ? 0.34f : 0.3f) * r, g->unit, ink, 0.1f * styles[st_unit[i]].px);
            if (g->hot) text_c(st_mark[i], g->cx, g->cy + 0.73f * r, "HOT", ink, 0.2f * styles[st_mark[i]].px);
            if (g->src == S_LAVG && g->idx == 0) text_c(st_mark[i], g->cx, g->cy + 0.79f * r, "OVERLOAD", ink, 0.2f * styles[st_mark[i]].px);
            if (g->src == S_NET) text_c(st_mark[i], g->cx, g->cy + 0.73f * r, "LINK", ink, 0.2f * styles[st_mark[i]].px);
            if (g->dual) {
                /* which needle is which */
                const char *l1 = g->src == S_NET ? "RX" : "READ", *l2 = g->src == S_NET ? "TX" : "WRITE";
                float y = g->cy + 0.19f * r + cap_h(st_unit[i]) * 0.5f, tr = 0.14f * styles[st_unit[i]].px;
                text(st_unit[i], g->cx - 0.07f * r, y, 1, l1, ink, tr);
                text(st_unit[i], g->cx + 0.07f * r, y, -1, l2, red, tr);
            }
        }
    }
    if (!pq_title) {
        text_engraved(st_name, (plq[0] + plq[2]) / 2, plq[1] + 1.7f * kp + cap_h(st_name) / 2, 0, hn, 0.22f * styles[st_name].px);
        rule_l[0] = plq[0] + 2.0f * kp; rule_l[2] = plq[2] - 2.0f * kp; rule_l[1] = plq[1] + 3.2f * kp;
        rule_l[3] = fmaxf(0.6f, 0.06f * kp);
        for (int w = 0; w < 2; w++)
            text_engraved(st_lab, plq[0] + 2.0f * kp, (win_rect[w][1] + win_rect[w][3]) / 2 + cap_h(st_lab) / 2, -1,
                          w ? "AWAY" : "UPTIME", 0.16f * styles[st_lab].px);
        /* group captions over the first window: drums 0-1 seconds, 2-3 minutes, 4-5 hours, 6-8 days */
        static const char *caps[4] = {"SEC", "MIN", "HRS", "DAYS"};
        static const int cg[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 8}};
        for (int c = 0; c < 4; c++) {
            float x = (drum_rect[cg[c][0]][2] + drum_rect[cg[c][1]][0]) / 2;
            text_engraved(st_cap, x, plq[1] + 4.25f * kp, 0, caps[c], 0.18f * styles[st_cap].px);
        }
    } else {
        float tk = (plq[3] - plq[1]) / 10, cx = (plq[0] + plq[2]) / 2;
        text_engraved(st_name, cx, plq[1] + 3.7f * tk + cap_h(st_name) / 2, 0, title, 0.22f * styles[st_name].px);
        rule_l[0] = plq[0] + 1.6f * tk; rule_l[2] = plq[2] - 1.6f * tk; rule_l[1] = plq[1] + 6.2f * tk;
        rule_l[3] = fmaxf(0.6f, 0.08f * tk);
        text_engraved(st_sub, cx, plq[1] + 7.9f * tk + cap_h(st_sub) / 2, 0, sub, 0.18f * styles[st_sub].px);
    }
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* shared: noise, shapes and the brass */
#define COMMON GLSL( \
    const vec3 L = vec3(-0.4636, -0.6182, 0.6347); \
    float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); } \
    float vnoise(vec2 p) { \
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); \
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); \
    } \
    float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; } \
    float sdBox(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; } \
    float sdRect(vec2 p, vec4 R, float r) { return sdBox(p - (R.xy + R.zw) * 0.5, (R.zw - R.xy) * 0.5, r); } \
    vec3 metal(vec3 n, vec3 base) { \
        vec3 R = reflect(vec3(0.0, 0.0, -1.0), n); \
        float w = dot(R, L); \
        float env = 0.16 + 0.5 * smoothstep(-0.3, 1.0, w) + 0.55 * pow(max(w, 0.0), 6.0); \
        float dif = max(dot(n, L), 0.0); \
        return base * (0.1 + 0.32 * dif + 0.95 * env) + vec3(1.0, 0.88, 0.62) * 0.55 * pow(max(w, 0.0), 40.0); \
    } \
    vec3 dome(vec2 sp, vec3 base) { \
        float z = sqrt(max(1.0 - dot(sp, sp), 0.0)); \
        return metal(normalize(vec3(sp * 0.9, z + 0.2)), base); \
    } \
)

static const char *FS_STATIC = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform float uSeed;
    uniform vec4 uPlate, uPq, uRule;
    uniform vec4 uG[32];       /* cx cy r kind */
    uniform vec4 uA[32];       /* a0 a1 nmaj nmin */
    uniform vec4 uZ[32];       /* red0 red1 amb0 amb1 */
    uniform int uNG;
    uniform vec4 uWin[2];
    uniform vec4 uLs[4];       /* lamp sockets: x y r */
    uniform int uNL;
    uniform vec3 uRed, uAmber, uBrass;
    out vec4 o;
)
COMMON
GLSL(
    float aa(float d) { return clamp(0.5 - d, 0.0, 1.0); }

    /* the bezel's cross-section, by radius / face radius: a lip sloping
     * into the face, a rounded rim, a flat flange, a rounded outer edge */
    float bezH(float t) {
        if (t < 1.035) return 0.5 * (t - 1.0) / 0.035;
        if (t < 1.13) { float u = (t - 1.035) / 0.095; return mix(0.5, 0.32, u) + 0.55 * sin(u * 3.14159); }
        if (t < 1.195) return 0.32 - 0.6 * (t - 1.13);
        float u = clamp((t - 1.195) / 0.025, 0.0, 1.0);
        return 0.281 * cos(u * 1.5708);
    }

    vec3 wood(vec2 p) {
        float m = min(uRes.x, uRes.y);
        vec2 q = p / m;
        float warp = fbm(q * vec2(1.5, 7.0) + uSeed) * 3.0;
        float grain = sin((q.y * 55.0 + warp * 6.0) ) * 0.5 + 0.5;
        float fine = vnoise(vec2(q.x * 30.0, q.y * 900.0 + warp * 40.0));
        vec3 c = mix(vec3(0.065, 0.045, 0.032), vec3(0.13, 0.088, 0.058), grain * 0.6 + fine * 0.4);
        return c * (0.85 + 0.3 * fbm(q * 4.0 + 3.0));
    }

    vec3 steel(vec2 p) {
        float m = min(uRes.x, uRes.y);
        vec2 q = p / m;
        float mott = fbm(q * 3.0 + 11.0 + uSeed);
        vec3 c = mix(vec3(0.075, 0.064, 0.056), vec3(0.13, 0.108, 0.09), mott);
        /* faint horizontal brushing, and a few warm rust-brown stains */
        c *= 0.93 + 0.1 * vnoise(vec2(p.x * 0.004, p.y * 0.9)) + 0.04 * hash(floor(p));
        c = mix(c, vec3(0.16, 0.09, 0.05), 0.35 * smoothstep(0.62, 0.8, fbm(q * 2.0 + 37.0)));
        /* lit from the upper left, falling off into the corners */
        vec2 uv = p / uRes;
        c *= 1.25 - 0.45 * length(uv - vec2(0.3, 0.2)) - 0.12 * uv.y;
        return c;
    }

    vec3 rivet(vec2 sp, vec3 under, float dpx) {
        vec3 c = dome(sp, vec3(0.42, 0.36, 0.3));
        return mix(under, c, aa(dpx));
    }

    vec3 panel(vec2 p) {
        float m = min(uRes.x, uRes.y);
        float dp = sdRect(p, uPlate, 0.012 * m);
        vec3 col = wood(p);
        /* the plate's shadow on the wood */
        col *= mix(1.0, 0.35, smoothstep(0.02 * m, -0.004 * m, sdRect(p - vec2(0.004, 0.006) * m, uPlate, 0.012 * m)));
        if (dp < 1.0) {
            vec3 s = steel(p);
            /* a bevelled edge catching the light at the top left */
            float bw = 0.006 * m;
            if (dp > -bw) {
                vec2 e = vec2(1.0, 0.0);
                vec2 g = normalize(vec2(sdRect(p + e.xy, uPlate, 0.012 * m) - sdRect(p - e.xy, uPlate, 0.012 * m),
                                        sdRect(p + e.yx, uPlate, 0.012 * m) - sdRect(p - e.yx, uPlate, 0.012 * m)) + 1e-6);
                vec3 n = normalize(vec3(g * 0.8, 1.0));
                s *= 0.55 + 1.1 * max(dot(n, L), 0.0);
            }
            col = mix(col, s, aa(dp));
            /* rivets in rows round the edge */
            float ri = 0.02 * m, rr = 0.0058 * m;
            vec4 R = uPlate + vec4(ri, ri, -ri, -ri);
            float best = 1e9; vec2 bc = vec2(0.0);
            for (int e = 0; e < 4; e++) {
                bool horiz = e < 2;
                float len = horiz ? R.z - R.x : R.w - R.y;
                float n = max(1.0, floor(len / (0.075 * m) + 0.5));
                float sp = len / n;
                float t = horiz ? p.x - R.x : p.y - R.y;
                float k = clamp(floor(t / sp + 0.5), 0.0, n);
                vec2 c = horiz ? vec2(R.x + k * sp, e == 0 ? R.y : R.w) : vec2(e == 2 ? R.x : R.z, R.y + k * sp);
                float d = length(p - c);
                if (d < best) { best = d; bc = c; }
            }
            /* shadow, then the rivet */
            col *= mix(1.0, 0.5, smoothstep(rr * 1.6, rr * 0.6, length(p - bc - vec2(0.3, 0.45) * rr)));
            if (best < rr + 1.0) col = rivet((p - bc) / rr, col, best - rr);
        }
        return col;
    }

    vec3 screw(vec2 sp, float ang, vec3 under, float dpx) {
        vec3 c = dome(sp * 0.85, uBrass * 0.95);
        vec2 rs = vec2(cos(ang) * sp.x + sin(ang) * sp.y, -sin(ang) * sp.x + cos(ang) * sp.y);
        float slot = smoothstep(0.17, 0.11, abs(rs.y)) * step(abs(rs.x), 0.9);
        c = mix(c, c * 0.15, slot);
        /* the slot's lit lower wall */
        c += vec3(0.25, 0.18, 0.08) * smoothstep(0.17, 0.12, abs(rs.y - 0.08)) * step(0.0, rs.y) * slot;
        c *= 1.0 - 0.4 * smoothstep(0.75, 1.0, length(sp));
        return mix(under, c, aa(dpx));
    }

    vec3 face(int i, vec2 q, float d, float r) {
        float t = d / r;
        vec2 u = q / r;
        vec3 c = vec3(0.90, 0.85, 0.76) * (1.03 - 0.12 * t * t);
        /* age: yellowed blotches, foxing, paper grain */
        float n = fbm(u * 2.5 + float(i) * 7.13 + uSeed);
        c *= 0.95 + 0.08 * n;
        c = mix(c, vec3(0.78, 0.66, 0.47), 0.3 * smoothstep(0.5, 0.78, fbm(u * 1.7 + 19.0 + float(i))));
        c = mix(c, vec3(0.62, 0.48, 0.32), 0.25 * smoothstep(0.72, 0.8, fbm(u * 6.0 + 3.0 + float(i) * 2.0)));
        c *= 0.985 + 0.03 * hash(floor(q));

        vec3 ink = vec3(0.11, 0.085, 0.065);
        float th = atan(q.x, -q.y);              /* clockwise from 12 o'clock */
        float lw = max(0.0045 * r, 0.9);          /* line half-widths, never thinner than a pixel */
        if (uG[i].w < 0.5) {
            vec4 A = uA[i];
            float span = A.y - A.x;
            float n = A.z * A.w;
            float ro = 0.88 * r;
            /* painted zones just outside the scale line */
            vec4 Z = uZ[i];
            float band = max(0.88 * r - d, d - 0.94 * r);
            for (int z = 0; z < 2; z++) {
                vec2 zz = z == 0 ? Z.xy : Z.zw;
                if (zz.x < 0.0) continue;
                float za = A.x + span * zz.x, zb = A.x + span * zz.y;
                float dz = max(band, max(za - th, th - zb) * d);
                c = mix(c, z == 0 ? uRed : uAmber, 0.9 * aa(dz));
            }
            /* the scale line */
            float da = max(abs(d - ro) - lw * 0.7, max(A.x - th, th - A.y) * d - lw);
            float cov = aa(da);
            /* ticks */
            float k = clamp(floor((th - A.x) / span * n + 0.5), 0.0, n);
            float tk = A.x + span * k / n;
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
            /* the stop pin, just below zero */
            vec2 pc = 0.8 * r * vec2(sin(A.x - 0.07), -cos(A.x - 0.07));
            float pr = 0.022 * r;
            c *= mix(1.0, 0.6, smoothstep(pr * 1.5, pr * 0.5, length(q - pc - vec2(0.4, 0.6) * pr)));
            float dpin = length(q - pc);
            if (dpin < pr + 1.0) c = mix(c, dome((q - pc) / pr, uBrass), aa(dpin - pr));
        } else {
            /* a railway-style minute track and bold hour bars */
            float dr = min(abs(d - 0.925 * r), abs(d - 0.855 * r)) - lw * 0.6;
            float cov = aa(max(dr, d - 0.94 * r));
            float k = floor(th / 0.10472 + 0.5);
            float tk = k * 0.10472;
            vec2 dir = vec2(sin(tk), -cos(tk));
            vec2 perp = vec2(-dir.y, dir.x);
            float along = dot(q, dir), across = dot(q, perp);
            bool hour = mod(k + 60.0, 5.0) < 0.5;
            float d1 = max(abs(across) - max(0.0045 * r, 0.85), abs(along - 0.89 * r) - 0.035 * r);
            float d2 = max(abs(across) - 0.016 * r, abs(along - 0.86 * r) - 0.065 * r);
            cov = max(cov, aa(hour ? d2 : d1));
            c = mix(c, ink, cov * 0.93);
        }
        /* the bezel lip shades the side of the face toward the light */
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
        /* concentric turning marks, and a little tarnish */
        float br = 1.0 + 0.09 * (vnoise(vec2(d * 0.6, float(i) * 13.0)) - 0.5) + 0.06 * (vnoise(vec2(d * 2.7, 5.0 + float(i))) - 0.5);
        vec3 base = mix(uBrass, uBrass * vec3(0.7, 0.62, 0.5), 0.5 * smoothstep(0.45, 0.8, fbm(q / r * 4.0 + float(i) * 3.0 + uSeed)));
        vec3 c = metal(n, base) * br;
        /* dirt in the crevices */
        c *= 1.0 - 0.45 * exp(-pow((t - 1.035) / 0.005, 2.0)) - 0.3 * exp(-pow((t - 1.13) / 0.006, 2.0));
        /* four screws in the flange */
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

    /* a thermometer: a brass mount, an ivory scale, a glass tube and bulb
     * (the mercury is drawn every frame); r is the mount's half-height */
    vec3 thermo(int i, vec2 q, float r, vec3 col) {
        float hw = 0.26 * r, cr = 0.07 * r;
        float dq = sdBox(q, vec2(hw, r), cr);
        if (dq > 1.0) return col;
        vec2 e = vec2(1.0, 0.0);
        vec3 n = vec3(0.0, 0.0, 1.0);
        float bw = 0.05 * r;
        if (dq > -bw) {
            vec2 g = normalize(vec2(sdBox(q + e.xy, vec2(hw, r), cr) - sdBox(q - e.xy, vec2(hw, r), cr),
                                    sdBox(q + e.yx, vec2(hw, r), cr) - sdBox(q - e.yx, vec2(hw, r), cr)) + 1e-6);
            float u = (dq + bw) / bw;
            n = normalize(vec3(g * (0.25 + 1.4 * u * u), 1.0));
        }
        float br = 1.0 + 0.08 * (vnoise(vec2(q.x * 0.05, q.y * 0.6 + float(i) * 9.0)) - 0.5);
        vec3 base = mix(uBrass, uBrass * vec3(0.72, 0.62, 0.5), 0.45 * smoothstep(0.45, 0.8, fbm(q / r * 2.0 + float(i) * 3.0 + uSeed)));
        vec3 c = metal(n, base) * br;
        /* the ivory scale plate, sunk into the mount */
        vec2 ph = vec2(0.19 * r, 0.86 * r);
        float dp = sdBox(q, ph, 0.03 * r);
        c *= 1.0 - 0.5 * aa(abs(dp) - 1.2);
        if (dp < 1.0) {
            vec2 u = q / r;
            vec3 f = vec3(0.90, 0.85, 0.76) * (1.0 - 0.08 * u.y * u.y);
            f *= 0.95 + 0.08 * fbm(u * 3.0 + float(i) * 7.13 + uSeed);
            f = mix(f, vec3(0.78, 0.66, 0.47), 0.3 * smoothstep(0.5, 0.78, fbm(u * 2.0 + 19.0 + float(i))));
            f *= 0.985 + 0.03 * hash(floor(q));
            /* inner shadow along the top and left edge */
            f *= 1.0 - 0.35 * smoothstep(0.02 * r, 0.0, -dp) * smoothstep(-0.5, 0.5, -(q.x + q.y * 0.3) / r);
            vec3 ink = vec3(0.11, 0.085, 0.065);
            vec4 A = uA[i];
            float y0 = 0.56 * r, y1 = -0.66 * r, tx = -0.035 * r;
            float nt = A.z * A.w;
            float k = clamp(floor((q.y - y0) / (y1 - y0) * nt + 0.5), 0.0, nt);
            float ty = y0 + (y1 - y0) * k / nt;
            bool major = mod(k, A.w) < 0.5;
            float len = (major ? 0.075 : 0.045) * r;
            float x0 = tx - 0.04 * r;
            float lw = max(0.0045 * r, 0.8);
            float dt = max(abs(q.y - ty) - (major ? lw * 1.3 : lw * 0.8), abs(q.x - (x0 - len * 0.5)) - len * 0.5);
            /* and the scale's spine */
            float ds = max(abs(q.x - (x0 - 0.002 * r)) - lw * 0.7, max(y1 - q.y, q.y - y0) - lw);
            f = mix(f, ink, aa(min(dt, ds)) * 0.93);
            /* the red zone, a thin strip outside the ticks */
            vec4 Z = uZ[i];
            if (Z.x >= 0.0) {
                float za = y0 + (y1 - y0) * Z.x, zb = y0 + (y1 - y0) * Z.y;
                float dz = max(abs(q.x - (x0 - 0.1 * r)) - 0.012 * r, max(zb - q.y, q.y - za));
                f = mix(f, uRed, 0.9 * aa(dz));
            }
            /* the tube's glass channel and bulb, empty */
            float tw = 0.022 * r, bc = 0.69 * r, brd = 0.062 * r;
            float dtube = min(sdBox(q - vec2(tx, (y1 - 0.06 * r + bc) * 0.5), vec2(tw, (bc - y1 + 0.06 * r) * 0.5), tw), length(q - vec2(tx, bc)) - brd);
            vec3 gl = mix(vec3(0.80, 0.80, 0.76), vec3(0.95, 0.95, 0.92), smoothstep(-tw, tw, -(q.x - tx)));
            f = mix(f, f * 0.55, aa(dtube - 1.6));
            f = mix(f, gl, aa(dtube));
            c = mix(c, f, aa(dp));
        }
        /* a screw top and bottom */
        float sr = 0.04 * r;
        for (int s = 0; s < 2; s++) {
            vec2 sc = vec2(0.0, s == 0 ? -0.93 * r : 0.93 * r);
            float d = length(q - sc);
            c *= mix(1.0, 0.55, smoothstep(sr * 1.4, sr * 0.6, length(q - sc - vec2(0.3, 0.45) * sr)));
            if (d < sr + 1.0) c = screw((q - sc) / sr, hash(vec2(float(i), float(s) + 4.0) + uSeed) * 3.14, c, d - sr);
        }
        /* a dark gap where it meets the plate */
        col *= 1.0 - 0.6 * aa(abs(dq) - 1.0);
        return mix(col, c, aa(dq));
    }

    vec3 plaque(vec2 p, vec3 col) {
        vec4 P = uPq;
        float h = P.w - P.y;
        float cr = 0.07 * h;
        float dq = sdRect(p, P, cr);
        /* its shadow */
        col *= mix(1.0, 0.4, smoothstep(0.08 * h, -0.02 * h, sdRect(p - vec2(0.025, 0.035) * h, P, cr)));
        if (dq > 1.0) return col;
        vec2 e = vec2(1.0, 0.0);
        float bw = 0.045 * h;
        vec3 n = vec3(0.0, 0.0, 1.0);
        if (dq > -bw) {
            vec2 g = normalize(vec2(sdRect(p + e.xy, P, cr) - sdRect(p - e.xy, P, cr), sdRect(p + e.yx, P, cr) - sdRect(p - e.yx, P, cr)) + 1e-6);
            float u = (dq + bw) / bw;
            n = normalize(vec3(g * (0.25 + 1.4 * u * u), 1.0));
        }
        float br = 1.0 + 0.1 * (vnoise(vec2(p.x * 0.01, p.y * 1.7)) - 0.5) + 0.05 * (vnoise(vec2(p.x * 0.05, p.y * 4.3)) - 0.5);
        vec3 base = mix(uBrass, uBrass * vec3(0.72, 0.62, 0.5), 0.45 * smoothstep(0.45, 0.8, fbm(p / h * 1.3 + 5.0 + uSeed)));
        vec3 c = metal(n, base) * br;
        /* the engraved rule */
        float dr = max(abs(p.y - uRule.y) - uRule.w, max(uRule.x - p.x, p.x - uRule.z));
        c = mix(c, c * 0.25, aa(dr));
        c = mix(c, c * 1.35, aa(max(abs(p.y - uRule.y - uRule.w - 0.6) - 0.5, max(uRule.x - p.x, p.x - uRule.z))));
        /* the odometer windows, sunk in with a bevel */
        for (int w = 0; w < 2; w++) {
            float rr = 0.12 * (uWin[w].w - uWin[w].y);
            float dw = sdRect(p, uWin[w], rr);
            float fw = 0.09 * (uWin[w].w - uWin[w].y);
            if (dw < fw) {
                vec2 g = normalize(vec2(sdRect(p + e.xy, uWin[w], rr) - sdRect(p - e.xy, uWin[w], rr), sdRect(p + e.yx, uWin[w], rr) - sdRect(p - e.yx, uWin[w], rr)) + 1e-6);
                vec3 fn = normalize(vec3(-g * 0.9, 1.0));
                vec3 fc = metal(fn, base * 0.9) * br;
                c = mix(c, fc, aa(-dw + fw - 0.5));
                vec3 hole = vec3(0.03, 0.022, 0.016);
                c = mix(c, hole, aa(dw));
            }
        }
        /* a screw in each corner */
        float sr = 0.052 * h;
        for (int s = 0; s < 4; s++) {
            vec2 sc = vec2(s % 2 == 0 ? P.x + 0.09 * h : P.z - 0.09 * h, s < 2 ? P.y + 0.09 * h : P.w - 0.09 * h);
            float ds = length(p - sc);
            c *= mix(1.0, 0.55, smoothstep(sr * 1.4, sr * 0.6, length(p - sc - vec2(0.3, 0.45) * sr)));
            if (ds < sr + 1.0) c = screw((p - sc) / sr, hash(vec2(float(s) + 9.0, 3.0) + uSeed) * 3.14, c, ds - sr);
        }
        return mix(col, c, aa(dq));
    }

    void main() {
        vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
        vec3 col = panel(p);
        /* the instruments' shadows on the plate */
        for (int i = 0; i < uNG; i++) {
            float rb = 1.22 * uG[i].z;
            if (uG[i].w > 2.5) {
                float r = uG[i].z;
                float d = sdBox(p - uG[i].xy - vec2(0.035, 0.05) * 0.5 * r, vec2(0.26 * r, r), 0.07 * r);
                col *= mix(1.0, 0.35, smoothstep(0.09 * r, -0.02 * r, d));
                continue;
            }
            float d = length(p - uG[i].xy - vec2(0.035, 0.05) * rb) - rb;
            col *= mix(1.0, 0.35, smoothstep(0.12 * rb, -0.03 * rb, d));
        }
        col = plaque(p, col);
        for (int i = 0; i < uNG; i++) {
            vec2 q = p - uG[i].xy;
            float d = length(q), r = uG[i].z;
            if (uG[i].w > 2.5) {
                if (abs(q.x) < 0.3 * r && abs(q.y) < 1.05 * r) col = thermo(i, q, r, col);
                continue;
            }
            if (d > 1.22 * r + 1.0) continue;
            vec3 b = bezel(i, q, d, r);
            if (d < r + 1.0) b = mix(b, face(i, q, d, r), aa(d - r));
            /* a dark gap where the bezel meets the plate */
            col *= 1.0 - 0.6 * aa(abs(d - 1.222 * r) - 1.0);
            col = mix(col, b, aa(d - 1.22 * r));
        }
        /* lamp sockets: a little domed brass ring round a dark jewel */
        for (int l = 0; l < 4; l++) {
            if (l >= uNL) break;
            vec2 q = p - uLs[l].xy;
            float lr = uLs[l].z, d = length(q);
            if (d > lr * 1.7 + 1.0) continue;
            col *= mix(1.0, 0.55, smoothstep(lr * 1.9, lr * 1.3, length(q - vec2(0.2, 0.3) * lr)));
            float u = clamp((d - lr) / (0.6 * lr), 0.0, 1.0);
            vec2 dir = q / max(d, 1e-3);
            vec3 n = normalize(vec3(dir * cos(u * 3.14159) * 0.9, 1.0));
            vec3 ring = metal(n, uBrass);
            col = mix(col, ring, aa(d - 1.6 * lr));
        }
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

/* every frame: the static panel, then needles and hands with their shadows,
 * the glass, the odometer drums and the lamps */
static const char *FS_DYN = "#version 330 core\n" GLSL(
    uniform sampler2D uStatic, uAtlas;
    uniform vec2 uRes;
    uniform vec4 uG[32];
    uniform vec4 uN[32];         /* needle angle, second needle's, has a second; the clocks: hour, minute, second; a thermometer: its level */
    uniform int uNG;
    uniform vec4 uWin[2];
    uniform vec4 uDp[17];        /* drum rects */
    uniform vec4 uDr[17];        /* digit, next digit, roll 0..1, radix (< 0: an ivory drum) */
    uniform vec4 uDig[10];       /* atlas rects of the digits */
    uniform vec4 uLamp[4];       /* x y r on */
    uniform vec3 uLampC[4];
    uniform int uNL;
    uniform vec3 uBrass, uRed;
    out vec4 o;
)
COMMON
GLSL(
    float aa(float d) { return clamp(0.5 - d, 0.0, 1.0); }

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
    /* q in the frame of a hand at angle a: x across, y out along it */
    vec2 hframe(vec2 q, float a) {
        vec2 dir = vec2(sin(a), -cos(a));
        return vec2(dot(q, vec2(-dir.y, dir.x)), dot(q, dir));
    }
    float needle(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float tail = 0.2 * r;
        float d = sdUC(h + vec2(0.0, tail), 0.016 * r, 0.0035 * r, 1.08 * r);
        d = min(d, length(h + vec2(0.0, tail * 0.9)) - 0.055 * r);
        return d;
    }
    float hourHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.12 * r), 0.026 * r, 0.008 * r, 0.64 * r);
        d = min(d, sdRhombus(h - vec2(0.0, 0.34 * r), vec2(0.06 * r, 0.12 * r)));
        return max(d, -(length(h - vec2(0.0, 0.34 * r)) - 0.026 * r));
    }
    float minuteHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.14 * r), 0.02 * r, 0.005 * r, 0.96 * r);
        d = min(d, sdRhombus(h - vec2(0.0, 0.5 * r), vec2(0.04 * r, 0.1 * r)));
        return max(d, -(length(h - vec2(0.0, 0.5 * r)) - 0.017 * r));
    }
    float secondHand(vec2 q, float a, float r) {
        vec2 h = hframe(q, a);
        float d = sdUC(h + vec2(0.0, 0.25 * r), 0.008 * r, 0.0035 * r, 1.12 * r);
        float cw = length(h + vec2(0.0, 0.2 * r));
        d = min(d, cw - 0.034 * r);
        return d;
    }

    /* a dark blued-steel hand, lit along its left flank */
    vec3 steelHand(vec3 base, float d, float r) {
        return base * (1.0 + 1.6 * smoothstep(-0.012 * r, 0.0, d));
    }

    vec3 glass(vec2 q, float r, vec3 col) {
        vec2 u = q / r;
        float rr = length(u);
        col *= 1.0 - 0.12 * smoothstep(0.75, 1.0, rr);
        vec2 un = u / max(rr, 1e-3);
        /* a crescent of window light on the upper left of the dome */
        float c1 = length(u - vec2(-0.03, -0.04)), c2 = length(u - vec2(0.09, 0.12));
        float cres = smoothstep(0.98, 0.92, c1) * smoothstep(0.84, 0.97, c2);
        cres *= smoothstep(-0.1, 0.9, dot(un, vec2(-0.6, -0.8)));
        col += vec3(1.0, 0.95, 0.86) * cres * 0.2;
        /* a broad soft sheen over that half, and a bright spot */
        float sheen = smoothstep(0.3, -0.7, dot(u, vec2(0.6, 0.8))) * smoothstep(1.0, 0.6, rr);
        col += vec3(1.0, 0.93, 0.8) * sheen * 0.045;
        vec2 sp = u - vec2(-0.4, -0.5);
        col += vec3(1.0, 0.97, 0.9) * exp(-dot(sp, sp * vec2(60.0, 120.0))) * 0.35;
        /* and a thin glint round the lower right rim */
        float glint = smoothstep(0.92, 0.975, rr) * smoothstep(1.0, 0.98, rr) * smoothstep(0.2, 0.95, dot(un, vec2(0.6, 0.8)));
        col += vec3(1.0, 0.92, 0.78) * glint * 0.16;
        return col;
    }

    vec3 instrument(int i, vec2 q, float r, vec3 col) {
        float bl = max(0.012 * r, 1.0);       /* shadow blur */
        vec2 so = vec2(0.45, 0.65);
        vec3 dark = vec3(0.045, 0.04, 0.042);
        if (uG[i].w < 0.5) {
            float a = uN[i].x;
            if (uN[i].z > 0.5) {
                /* the second needle, red, under the first */
                float a2 = uN[i].y;
                float ds2 = needle(q - so * 0.03 * r, a2, r * 0.93);
                col *= 1.0 - 0.34 * smoothstep(bl, -bl, ds2);
                float d2 = needle(q, a2, r * 0.93);
                col = mix(col, uRed * (0.75 + 0.6 * smoothstep(-0.012 * r, 0.0, d2)), aa(d2));
            }
            float ds = needle(q - so * 0.035 * r, a, r);
            col *= 1.0 - 0.38 * smoothstep(bl, -bl, ds);
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
        /* the brass cap over the arbor */
        float hr = 0.06 * r, dh = length(q);
        col *= 1.0 - 0.35 * smoothstep(hr * 1.4, hr * 0.8, length(q - so * 0.3 * hr));
        if (dh < hr + 1.0) col = mix(col, dome(q / hr, uBrass), aa(dh - hr));
        if (uG[i].w > 0.5 && dh < 0.4 * hr + 1.0) col = mix(col, uRed * (1.1 - 0.5 * length(q / (0.4 * hr) + 0.4)), aa(dh - 0.4 * hr));
        return glass(q, r, col);
    }

    /* the mercury, and the tube's glass over it */
    vec3 mercury(int i, vec2 q, float r, vec3 col) {
        float tx = -0.035 * r, tw = 0.022 * r, bc = 0.69 * r, brd = 0.062 * r;
        float y0 = 0.56 * r, y1 = -0.66 * r;
        float top = y0 + (y1 - y0) * uN[i].x;
        float iw = tw * 0.62;
        float dm = min(sdBox(q - vec2(tx, (top + bc) * 0.5), vec2(iw, max((bc - top) * 0.5, 0.0)), iw * 0.5), length(q - vec2(tx, bc)) - brd * 0.8);
        float lx = (q.x - tx) / iw;
        vec3 m = uRed * (0.75 + 0.55 * exp(-pow(lx + 0.35, 2.0) * 4.0));
        float bl = length(q - vec2(tx, bc) + vec2(0.3, 0.35) * brd) / brd;
        m += vec3(1.0, 0.85, 0.75) * 0.5 * exp(-bl * bl * 20.0);
        col = mix(col, m, aa(dm));
        /* a highlight running down the glass */
        float dg = sdBox(q - vec2(tx - 0.45 * tw, (y1 + bc) * 0.5 - 0.04 * r), vec2(0.18 * tw, (bc - y1) * 0.5 - 0.04 * r), 0.18 * tw);
        col += vec3(1.0, 0.97, 0.9) * 0.28 * aa(dg + 0.3);
        return col;
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
        vec3 fg = ivory ? vec3(0.5, 0.14, 0.1) : vec3(0.9, 0.85, 0.76);
        vec3 dc = mix(bg, fg, ink);
        /* the drum's curve: lit across its middle, dark top and bottom */
        float cphi = cos(phi);
        dc *= 0.3 + 0.75 * pow(smoothstep(0.86, 1.0, cphi), 1.2) + 0.08 * smoothstep(0.0, -0.8, v);
        /* worn edges between drums */
        float hw = (R.z - R.x) * 0.5;
        dc *= 1.0 - 0.5 * smoothstep(hw - 2.5, hw, abs(lc.x));
        return dc;
    }

    void main() {
        vec2 fc = gl_FragCoord.xy;
        vec2 p = vec2(fc.x, uRes.y - fc.y);
        vec3 col = texelFetch(uStatic, ivec2(fc), 0).rgb;
        for (int i = 0; i < uNG; i++) {
            vec2 q = p - uG[i].xy;
            float r = uG[i].z;
            if (uG[i].w > 2.5) {
                if (abs(q.x + 0.035 * r) < 0.03 * r && q.y > -0.75 * r && q.y < 0.76 * r) col = mercury(i, q, r, col);
                continue;
            }
            if (dot(q, q) < (r + 1.0) * (r + 1.0)) col = mix(col, instrument(i, q, r, col), aa(length(q) - r));
        }
        for (int w = 0; w < 2; w++) {
            vec4 R = uWin[w];
            if (p.x < R.x || p.x > R.z || p.y < R.y || p.y > R.w) continue;
            for (int j = 0; j < 17; j++) {
                vec4 D = uDp[j];
                if (p.x >= D.x && p.x < D.z && p.y >= D.y && p.y < D.w) col = drum(j, p, col);
            }
            /* the window's glass: a soft band of light across its top */
            float t = (p.y - R.y) / (R.w - R.y);
            col += vec3(1.0, 0.94, 0.82) * 0.07 * smoothstep(0.1, 0.2, t) * smoothstep(0.42, 0.25, t);
            col *= 1.0 - 0.5 * smoothstep(0.2, 0.0, t) - 0.35 * smoothstep(0.8, 1.0, t);
        }
        /* lamps: a jewel, glowing when lit, its light spilling round it */
        for (int l = 0; l < 4; l++) {
            if (l >= uNL) break;
            vec2 q = p - uLamp[l].xy;
            float lr = uLamp[l].z, on = uLamp[l].w, d = length(q);
            if (d > lr * 6.0) continue;
            vec3 lc = uLampC[l];
            col += lc * on * 0.35 * exp(-max(d - lr, 0.0) / (lr * 1.1)) * step(lr * 1.6, d);
            if (d < lr + 1.0) {
                vec2 sp = q / lr;
                float z = sqrt(max(1.0 - dot(sp, sp), 0.0));
                /* facets: a few flat-ish rings */
                float fac = 0.85 + 0.15 * cos(floor(atan(sp.y, sp.x) / 0.7854) * 2.1 + floor(length(sp) * 3.0));
                vec3 j = lc * (0.13 + on * (0.5 + 1.1 * z)) * fac;
                j += vec3(1.0, 0.95, 0.9) * exp(-dot(sp - vec2(-0.35, -0.4), sp - vec2(-0.35, -0.4)) * 25.0) * 0.55;
                col = mix(col, j, aa(d - lr));
            }
        }
        o = vec4(col, 1.0);
    });

static GLuint prog_static, prog_text, prog_dyn, vao, vao_text, vbo_text;
static GLuint tex_atlas, tex_static, fbo_static;
static int static_w, static_h;
static float c_brass[3], c_red[3], c_amber[3], c_sage[3];

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "gauges: shader:\n%s\n", log);
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

    /* Aged Brass */
    hex("#A53C2C", c_red); hex("#E9994A", c_amber); hex("#8FA672", c_sage);
    c_brass[0] = 0.70f; c_brass[1] = 0.52f; c_brass[2] = 0.25f;
}

/* (re)build everything that only changes with the size: layout, atlas,
 * print, and the static panel texture */
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
    float gv[MAXG][4], av[MAXG][4], zv[MAXG][4], ls[4][4] = {{0}};
    for (int i = 0; i < NG; i++) {
        gv[i][0] = G[i].cx; gv[i][1] = G[i].cy; gv[i][2] = G[i].r; gv[i][3] = (float)G[i].kind;
        av[i][0] = G[i].a0; av[i][1] = G[i].a1; av[i][2] = (float)G[i].nmaj; av[i][3] = (float)G[i].nmin;
        zv[i][0] = G[i].red0; zv[i][1] = G[i].red1; zv[i][2] = G[i].amb0; zv[i][3] = G[i].amb1;
    }
    for (int l = 0; l < nlamp; l++) { ls[l][0] = lamp_pos[l][0]; ls[l][1] = lamp_pos[l][1]; ls[l][2] = lamp_pos[l][2]; }
    glUniform2f(U(p, "uRes"), (float)W, (float)H);
    /* each output its own stretch of wood and steel */
    glUniform1f(U(p, "uSeed"), ((opt_seed + (span_me > 0 ? 131u * (uint32_t)span_me : 0)) % 1000) * 0.137f);
    glUniform1i(U(p, "uNG"), NG);
    glUniform4fv(U(p, "uPlate"), 1, plate);
    glUniform4fv(U(p, "uPq"), 1, plq);
    glUniform4fv(U(p, "uRule"), 1, rule_l);
    glUniform4fv(U(p, "uG"), NG, &gv[0][0]);
    glUniform4fv(U(p, "uA"), NG, &av[0][0]);
    glUniform4fv(U(p, "uZ"), NG, &zv[0][0]);
    glUniform4fv(U(p, "uWin"), 2, &win_rect[0][0]);
    glUniform4fv(U(p, "uLs"), 4, &ls[0][0]);
    glUniform1i(U(p, "uNL"), nlamp);
    glUniform3fv(U(p, "uRed"), 1, c_red);
    float amb[3];
    for (int i = 0; i < 3; i++) amb[i] = c_amber[i] * 0.9f;
    glUniform3fv(U(p, "uAmber"), 1, amb);
    glUniform3fv(U(p, "uBrass"), 1, c_brass);
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
    glUniform1i(U(p, "uStatic"), 0);
    glUniform1i(U(p, "uAtlas"), 1);
    glUniform4fv(U(p, "uG"), NG, &gv[0][0]);
    glUniform1i(U(p, "uNG"), NG);
    glUniform4fv(U(p, "uWin"), 2, &win_rect[0][0]);
    glUniform4fv(U(p, "uDp"), NDRUM, &drum_rect[0][0]);
    float dig[10][4];
    for (int d = 0; d < 10; d++) {
        const Glyph *g = glyph(drum_style, (unsigned char)('0' + d));
        dig[d][0] = g->x; dig[d][1] = g->y; dig[d][2] = g->w; dig[d][3] = g->h;
    }
    glUniform4fv(U(p, "uDig"), 10, &dig[0][0]);
    glUniform1i(U(p, "uNL"), nlamp);
    glUniform3fv(U(p, "uBrass"), 1, c_brass);
    glUniform3fv(U(p, "uRed"), 1, c_red);
}

/* t: seconds since start; wt: wall-clock time */
static double t_start_wall;
static void render(double t, double wt, GLuint target) {
    if (static_w != W || static_h != H) build_static();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    GLuint p = prog_dyn;
    glUseProgram(p);

    float nv[MAXG][4] = {{0}};
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    float secs = tm.tm_sec + (float)(wt - floor(wt));
    float mins = tm.tm_min + secs / 60;
    double away = wt - (double)locked_at;
    if (away < 0) away = 0;
    for (int i = 0; i < NG; i++) {
        Gauge *g = &G[i];
        if (g->kind == K_CLOCK) {
            nv[i][0] = ((tm.tm_hour % 12) + mins / 60) * 30 * DEG;
            nv[i][1] = mins * 6 * DEG;
            nv[i][2] = sec_angle * DEG;
        } else if (g->kind == K_CHRONO) {
            /* the time away: a sweeping seconds hand, minutes, and hours */
            nv[i][0] = (float)fmod(away / 3600, 12) * 30 * DEG;
            nv[i][1] = (float)fmod(away / 60, 60) * 6 * DEG;
            nv[i][2] = (float)fmod(away, 60) * 6 * DEG;
        } else if (g->kind == K_THERMO) {
            nv[i][0] = g->x;
        } else {
            nv[i][0] = needle_angle(g, g->x, t);
            nv[i][1] = needle_angle(g, g->x2, t + 3.7);
            nv[i][2] = (float)g->dual;
        }
    }
    glUniform4fv(U(p, "uN"), NG, &nv[0][0]);

    /* drums: digit now, digit next, and the roll between over the last 0.3 s */
    double up = sens.uptime + (now() - sens.uptime_at);
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
            f = 1 + c3 * xm * xm * xm + c1 * xm * xm;         /* ease out with a little overshoot */
        }
        int ivory = drum_spec[j] == D_S1 || drum_spec[j] == D_S10;
        dr[j][0] = (float)a; dr[j][1] = (float)b; dr[j][2] = f;
        dr[j][3] = (float)radix_of(drum_spec[j]) * (ivory ? -1 : 1);
    }
    glUniform4fv(U(p, "uDr"), NDRUM, &dr[0][0]);

    /* lamps: HOT when the temperature is in the red; MAINS on the battery */
    /* (and OVERLOAD past one minute's load per CPU; LINK while a network is up) */
    float lv[4][4] = {{0}}, lc[4][3] = {{0}};
    for (int l = 0; l < nlamp; l++) {
        Gauge *g = &G[lamp_g[l]];
        lv[l][0] = lamp_pos[l][0]; lv[l][1] = lamp_pos[l][1]; lv[l][2] = lamp_pos[l][2];
        if (lamp_kind[l] == L_HOT || lamp_kind[l] == L_OVER) {
            float on = g->x >= g->red0 ? 1.0f : 0.0f;
            lv[l][3] = on * (0.85f + 0.15f * sinf((float)t * 6.0f));
            for (int i = 0; i < 3; i++) lc[l][i] = c_red[i] * 1.5f;
        } else if (lamp_kind[l] == L_MAINS) {
            /* steady on mains; breathing slowly while charging */
            lv[l][3] = sens.mains ? (sens.charging ? 0.7f + 0.3f * sinf((float)t * 2.0f) : 1.0f) : 0.0f;
            memcpy(lc[l], c_sage, sizeof lc[l]);
        } else {
            lv[l][3] = sens.link ? 1.0f : 0.0f;
            memcpy(lc[l], c_sage, sizeof lc[l]);
        }
    }
    glUniform4fv(U(p, "uLamp"), 4, &lv[0][0]);
    glUniform3fv(U(p, "uLampC"), 4, &lc[0][0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_static);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* advance the springs to time t (fixed sub-steps) and read the sensors once a second */
static double sim_t, last_read;
static void advance(double t, double wt) {
    /* spanning, every instance reads on the same wall-clock second, so
     * their needles move together */
    if (span_me >= 0 ? floor(wt) != last_read : t - last_read >= 1.0) {
        read_sensors();
        set_targets();
        last_read = span_me >= 0 ? floor(wt) : t;
    }
    const float h = 1.0f / 240;
    if (t - sim_t > 10) sim_t = t - 10;     /* long after the epoch: the springs have settled anyway */
    while (sim_t + h <= t) {
        sim_t += h;
        physics(wt - (t - sim_t), h, sim_t > 0.6);
    }
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "gauges");
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
        "usage: gauges [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY]\n"
        "              [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "              [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
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
 * gets the main cluster; the rest, left to right, take the core bank and the
 * auxiliaries in turn. */
static int pick_role(void) {
    if (span_me < 0) return R_FRONT;
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
    return rank % 2 ? R_AUX : R_BANK;
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
    if (role == R_AUX) { find_thermometers(); find_fans(); }
    read_sensors();
    {   /* a first CPU-load sample over a tenth of a second */
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
        read_sensors();
    }
    gauges_init();
    set_targets();
    fonts_init();

    /* --epoch: when the panel powered up, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
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
        XStoreName(dpy, win, "Gauges");
        XClassHint chint = {"gauges", "Gauges"};
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
            double wt = opt_epoch ? opt_epoch + opt_at : wall();
            const float h = 1.0f / 240;
            sec_angle = (float)(((time_t)wt % 60) * 6);
            for (sim_t = 0; sim_t < opt_at; sim_t += h) physics(wt - (opt_at - sim_t), h, sim_t > 0.6);
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
        printf("gauges: %dx%d, %d frames: avg %.3f ms, worst %.3f ms per frame (static panel built once in %.1f ms)\n",
               W, H, opt_bench, sum / opt_bench, worst, tb * 1000);
        printf("gauges: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
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
        double t = now(), wt = wall(), el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
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
