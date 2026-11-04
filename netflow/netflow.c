/*
 * netflow — the machine's own networking, live, drawn as a systems diagram.
 *
 * A layered network stack, read bottom to top the way a packet arrives:
 *
 *   LINK       one box per interface (Wi-Fi with its signal rings, Ethernet,
 *              WWAN, tunnels, bridges, the veths folded into one, loopback),
 *              each with its state, speed or signal, and its rx/tx rates
 *   NETWORK    the IP bus (v4 + v6): packets in and out, discards, reassembly
 *   TRANSPORT  TCP, UDP and ICMP, with their segment/datagram/message rates,
 *              retransmits, errors and unreachables
 *   SOCKETS    a matrix of pips, one per socket: TCP by connection state
 *              (established lit, listening hollow, TIME_WAIT dim, half-closed
 *              amber), UDP, and raw
 *
 * Packets are glowing particles flowing up the conduits (received) and down
 * (sent), at rates that follow the real per-interface packet rates (log
 * scaled), coloured by protocol and sized by the average packet size.
 * Retransmits are red flicks that bounce back off the IP layer; drops and
 * errors fall off the side of their interface. When nothing much is moving,
 * keepalive trickles, the scope beams and a slow carrier shimmer on the bus
 * keep it breathing.
 *
 * Beside the stack: an oscilloscope per interface (rx and tx throughput on
 * a log scale, the last two minutes), TCP connection-state dials, and a
 * stylised congestion-window sawtooth (an AIMD sketch, not a measurement:
 * its loss rate comes from RetransSegs / OutSegs, so it saws faster and
 * lower as retransmits rise).
 *
 * Data, strictly aggregate and read-only (no capture, no root, no addresses
 * or ports, no SSIDs ever read or shown): /proc/net/dev, /proc/net/snmp,
 * /proc/net/snmp6, /proc/net/netstat, /proc/net/sockstat{,6}, the state
 * column only of /proc/net/tcp{,6}, /proc/net/wireless, and
 * /sys/class/net/<if>/{type,operstate,flags,speed,uevent,...}. Sampled once
 * a second and smoothed. --shot and --bench use a deterministic, seeded
 * synthetic traffic model instead (--traffic busy | idle), unless
 * --traffic live is given.
 *
 * Palettes (--palette, else from the seed): oscilloscope (green phosphor),
 * datacentre (cool blues), paper (a light schematic), amber.
 *
 * Every shape is a signed-distance primitive (capsules, rounded boxes, arcs,
 * glowing dots) on an instanced quad, so it's crisp and anti-aliased at any
 * size; text comes from a freetype glyph atlas. The parts that never move
 * are drawn once into a texture.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The output under the middle of the layout shows the stack; the others,
 * left to right, take turns at the per-interface scopes and the protocol
 * counters with the connection states. Every instance samples on the same
 * wall-clock second.
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
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include "proto-layer-shell.h"
#include "../common/hidpi.h"

#define PI 3.14159265f

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "Inter";
static const char *opt_mono = "JetBrains Mono,JetBrainsMono Nerd Font Mono,DejaVu Sans Mono";
static const char *opt_palette;
static const char *opt_traffic;        /* live | busy | idle */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 60;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;
enum { T_LIVE, T_BUSY, T_IDLE };
static int traffic = T_LIVE;

/* ---- spanning: one instance per output, each with a role ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];
static int span_n, span_me = -1;
enum { R_FRONT, R_SCOPES, R_COUNTERS };
static int role = R_FRONT;

static void die(const char *what) { fprintf(stderr, "netflow: %s\n", what); exit(1); }

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

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
/* a hash of (channel, integer time) in [0, 1) */
static float hrand(uint32_t ch, long k) {
    return (hash32(hash32((uint32_t)k * 0x9e3779b9u ^ ch * 0x85ebca6bu) ^ opt_seed) >> 8) * (1.0f / 16777216.0f);
}
/* smooth value noise in [0, 1] over time x */
static float vnoise(uint32_t ch, double x) {
    double f = floor(x);
    float t = (float)(x - f), a = hrand(ch, (long)f), b = hrand(ch, (long)f + 1);
    t = t * t * (3 - 2 * t);
    return a + (b - a) * t;
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

/* a whole (small) /proc file into buf */
static size_t slurp(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t r = fread(buf, 1, n - 1, f);
    buf[r] = 0;
    fclose(f);
    return r;
}

static int exists(const char *path) { struct stat st; return stat(path, &st) == 0; }

static char host[64];
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
        /* fields after the command: state is #3, starttime #22 */
        if (p && sscanf(p + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &start) == 1) {
            time_t t = btime + (time_t)(start / (unsigned long long)hz);
            if (!best || t < best) best = t;
        }
    }
    if (d) closedir(d);
    return best;
}

/* ---- palettes ---- */
typedef struct {
    const char *name;
    const char *bg0, *bg1, *grid, *ink, *dim, *line, *tcp, *udp, *icmp, *rx, *tx, *red, *warn, *scr;
    int light;
} PalDef;
static const PalDef pals[] = {
    {"oscilloscope", "#020906", "#05150b", "#1f7a45", "#c4ffda", "#5a9c72", "#2f8a55",
     "#4dff8a", "#c8ff52", "#52f5e6", "#4dff8a", "#b2ffe9", "#ff5544", "#ffb43c", "#010503", 0},
    {"datacentre", "#040812", "#0a1628", "#2c5690", "#dbe9ff", "#7290b8", "#3f72b4",
     "#47b9ff", "#a191ff", "#3fe3c6", "#47b9ff", "#c2e6ff", "#ff5a6e", "#ffb347", "#02060d", 0},
    {"paper", "#f5f1e7", "#ebe4d4", "#5f84b8", "#1b2333", "#68707e", "#33507c",
     "#1b5eb0", "#b9512a", "#23875b", "#1b5eb0", "#c27b12", "#c62e29", "#c98a12", "#fbf9f3", 1},
    {"amber", "#0b0602", "#180c03", "#8f4e14", "#ffdcaa", "#a87840", "#a0601e",
     "#ffad3f", "#ffe08e", "#ff7a2e", "#ffad3f", "#ffe6bd", "#ff4632", "#ffd040", "#060301", 0},
};
#define NPAL (int)(sizeof pals / sizeof pals[0])
static int pal_i;
static float c_bg0[3], c_bg1[3], c_grid[3], c_ink[3], c_dim[3], c_line[3], c_proto[3][3], c_rx[3], c_tx[3],
    c_red[3], c_warn[3], c_scr[3];
static int light;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

static void set_palette(void) {
    const PalDef *p = &pals[pal_i];
    hex(p->bg0, c_bg0); hex(p->bg1, c_bg1); hex(p->grid, c_grid); hex(p->ink, c_ink); hex(p->dim, c_dim);
    hex(p->line, c_line); hex(p->tcp, c_proto[0]); hex(p->udp, c_proto[1]); hex(p->icmp, c_proto[2]);
    hex(p->rx, c_rx); hex(p->tx, c_tx); hex(p->red, c_red); hex(p->warn, c_warn); hex(p->scr, c_scr);
    light = p->light;
}

/* ---- the data ---- */
#define MAXIF 7
#define HIST 128                       /* seconds of history kept; the scopes show 120 */
#define SCOPE_S 120
enum { IK_WIFI, IK_ETH, IK_WWAN, IK_TUN, IK_BRIDGE, IK_VETH, IK_OTHER, IK_LOOP };
static const char *ik_name[] = {"WI-FI", "ETHERNET", "WWAN", "TUNNEL", "BRIDGE", "VIRTUAL", "LINK", "LOOPBACK"};
enum { RXB, RXP, RXE, RXD, TXB, TXP, TXE, TXD, NIC };

typedef struct {
    char name[20];
    int kind, up, speed, nagg;          /* nagg: how many veths are folded into this one */
    double c[NIC];                      /* cumulative counters (live) */
    int have_c;
    float r[NIC];                       /* the latest per-second rates */
    float d[NIC];                       /* displayed: smoothed toward r */
    float hrx[HIST], htx[HIST];         /* bytes/s by second, a ring indexed by second % HIST */
    unsigned char hev[HIST];            /* errors + drops in that second (capped) */
    long hfrom;                         /* the first second with history */
    float sig_q, sig_dbm, sig_qd;       /* Wi-Fi: link quality 0..1 (and displayed), level dBm; sig_q < 0: none */
    float act;                          /* smoothed activity 0..1 for the box's glow */
} Iface;
static Iface ifs[MAXIF];
static int nif;

/* protocol counters (cumulative in live mode) and their rates */
enum { PR_TCP_IN, PR_TCP_OUT, PR_RETX, PR_AOPEN, PR_POPEN, PR_TIMEOUT, PR_LDROP,
       PR_UDP_IN, PR_UDP_OUT, PR_UDP_ERR, PR_ICMP_IN, PR_ICMP_OUT, PR_ICMP_UNR,
       PR_IP_IN, PR_IP_OUT, PR_IP_DISC, PR_REASM, NPR };
static double pcum[NPR];
static int have_pcum;
static float pr[NPR], pd[NPR], hpr[NPR][HIST];
static long pr_from;

/* levels */
enum { LV_ESTAB, LV_TCP_INUSE, LV_ORPHAN, LV_TW, LV_UDP_INUSE, LV_RAW_INUSE, LV_SOCK_USED, NLV };
static float lv[NLV], lvd[NLV];
/* TCP connection states, by the kernel's numbering (1 ESTABLISHED .. 12 NEW_SYN_RECV) */
#define NST 13
static float st[NST], std_[NST];
static double last_sample_t;

/* sort key: physical first, loopback last */
static int if_order(const Iface *a) { return a->kind * 2 + !a->up; }

static int classify(const char *name) {
    char p[300], ue[1024];
    if (!strcmp(name, "lo")) return IK_LOOP;
    snprintf(p, sizeof p, "/sys/class/net/%s/type", name);
    long type = read_long(p, 1);
    if (type == 772) return IK_LOOP;
    snprintf(p, sizeof p, "/sys/class/net/%s/uevent", name);
    slurp(p, ue, sizeof ue);
    if (strstr(ue, "DEVTYPE=wlan")) return IK_WIFI;
    snprintf(p, sizeof p, "/sys/class/net/%s/wireless", name);
    if (exists(p)) return IK_WIFI;
    if (strstr(ue, "DEVTYPE=wwan") || type == 519) return IK_WWAN;
    snprintf(p, sizeof p, "/sys/class/net/%s/bridge", name);
    if (strstr(ue, "DEVTYPE=bridge") || exists(p)) return IK_BRIDGE;
    snprintf(p, sizeof p, "/sys/class/net/%s/tun_flags", name);
    if (exists(p) || type == 65534 || strstr(ue, "DEVTYPE=wireguard")) return IK_TUN;
    snprintf(p, sizeof p, "/sys/class/net/%s/device", name);
    if (exists(p)) return type == 1 ? IK_ETH : IK_OTHER;
    return IK_VETH;
}

static int if_up(const char *name) {
    char p[300], op[32];
    snprintf(p, sizeof p, "/sys/class/net/%s/operstate", name);
    slurp_line(p, op, sizeof op);
    if (!strcmp(op, "up")) return 1;
    if (strcmp(op, "unknown")) return 0;
    snprintf(p, sizeof p, "/sys/class/net/%s/flags", name);
    char fl[32];
    slurp_line(p, fl, sizeof fl);
    return (strtoul(fl, NULL, 16) & 1) != 0;
}

static int layout_dirty = 1;

/* adopt a new interface list, keeping each one's history by name */
static void set_ifaces(Iface *nw, int n) {
    int changed = n != nif;
    for (int i = 0; i < n; i++) {
        int found = 0;
        for (int j = 0; j < nif; j++) {
            if (strcmp(ifs[j].name, nw[i].name)) continue;
            Iface keep = nw[i];
            nw[i] = ifs[j];
            nw[i].kind = keep.kind; nw[i].up = keep.up; nw[i].speed = keep.speed; nw[i].nagg = keep.nagg;
            if (keep.have_c) {
                /* fresh counters; rates from the old ones */
                memcpy(nw[i].r, keep.r, sizeof keep.r);
                memcpy(nw[i].c, keep.c, sizeof keep.c);
                nw[i].have_c = 1;
            }
            nw[i].sig_q = keep.sig_q; nw[i].sig_dbm = keep.sig_dbm;
            if (i != j || ifs[j].up != keep.up) changed = 1;
            found = 1;
            break;
        }
        if (!found) changed = 1;
    }
    memcpy(ifs, nw, sizeof(Iface) * (size_t)n);
    nif = n;
    if (changed) layout_dirty = 1;
}

static int cmp_if(const void *a, const void *b) {
    const Iface *x = a, *y = b;
    int d = if_order(x) - if_order(y);
    return d ? d : strcmp(x->name, y->name);
}

/* /proc/net/dev + /sys/class/net: interface counters, kinds and states */
static void read_ifaces_live(double dt, long k) {
    static char buf[16384];
    Iface nw[32];
    int n = 0;
    memset(nw, 0, sizeof nw);
    slurp("/proc/net/dev", buf, sizeof buf);
    char wl[4096];
    slurp("/proc/net/wireless", wl, sizeof wl);
    Iface veth = {0};
    strcpy(veth.name, "veth*");
    veth.kind = IK_VETH;
    for (char *line = strtok(buf, "\n"); line; line = strtok(NULL, "\n")) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = 0;
        char *nm = line;
        while (*nm == ' ') nm++;
        double c[16] = {0};
        if (sscanf(colon + 1, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &c[0], &c[1], &c[2], &c[3], &c[4],
                   &c[5], &c[6], &c[7], &c[8], &c[9], &c[10], &c[11]) != 12) continue;
        Iface f = {0};
        snprintf(f.name, sizeof f.name, "%s", nm);
        f.kind = classify(nm);
        f.up = if_up(nm);
        f.c[RXB] = c[0]; f.c[RXP] = c[1]; f.c[RXE] = c[2]; f.c[RXD] = c[3];
        f.c[TXB] = c[8]; f.c[TXP] = c[9]; f.c[TXE] = c[10]; f.c[TXD] = c[11];
        f.have_c = 1;
        f.sig_q = -1;
        if (f.kind == IK_ETH) {
            char p[300];
            snprintf(p, sizeof p, "/sys/class/net/%s/speed", nm);
            long s = read_long(p, -1);
            f.speed = s > 0 && s < 1000000 ? (int)s : 0;
        }
        if (f.kind == IK_WIFI) {
            /* "name: status quality. level. noise. ..." */
            char key[40];
            snprintf(key, sizeof key, "%s:", nm);
            char *w = strstr(wl, key);
            float q, l;
            if (w && sscanf(w + strlen(key), " %*x %f %f", &q, &l) == 2) {
                f.sig_q = clampf(q / 70.0f, 0, 1);
                f.sig_dbm = l;
            }
        }
        if (f.kind == IK_VETH) {
            for (int i = 0; i < NIC; i++) veth.c[i] += f.c[i];
            veth.up |= f.up;
            veth.have_c = 1;
            if (!veth.nagg) snprintf(veth.name, sizeof veth.name, "%s", f.name);
            veth.nagg++;
            continue;
        }
        if (n < 31) nw[n++] = f;
    }
    if (veth.nagg) {
        if (veth.nagg > 1) strcpy(veth.name, "veth*");
        veth.sig_q = -1;
        nw[n++] = veth;
    }
    qsort(nw, (size_t)n, sizeof nw[0], cmp_if);
    /* too many: drop the down ones (not loopback) first, then the tail */
    while (n > MAXIF) {
        int drop = -1;
        for (int i = n - 1; i >= 0 && drop < 0; i--)
            if (!nw[i].up && nw[i].kind != IK_LOOP) drop = i;
        if (drop < 0) drop = nw[n - 1].kind == IK_LOOP ? n - 2 : n - 1;
        memmove(nw + drop, nw + drop + 1, sizeof nw[0] * (size_t)(n - drop - 1));
        n--;
    }
    /* rates from the previous counters, matched by name */
    for (int i = 0; i < n; i++) {
        memset(nw[i].r, 0, sizeof nw[i].r);
        for (int j = 0; j < nif; j++) {
            if (strcmp(ifs[j].name, nw[i].name) || !ifs[j].have_c || dt <= 0) continue;
            for (int c = 0; c < NIC; c++) nw[i].r[c] = (float)fmax(0, (nw[i].c[c] - ifs[j].c[c]) / dt);
        }
    }
    int had = nif;
    set_ifaces(nw, n);
    for (int i = 0; i < nif; i++)
        if (!had || ifs[i].hfrom == 0) ifs[i].hfrom = k;
}

/* "Sect: k1 k2 ...\nSect: v1 v2 ..." pairs (snmp, netstat) */
static double mib(const char *buf, const char *sect, const char *key) {
    size_t sl = strlen(sect);
    for (const char *p = buf; p && *p;) {
        if (!strncmp(p, sect, sl) && p[sl] == ':') {
            const char *v = strchr(p, '\n');
            if (!v) return 0;
            v++;
            if (strncmp(v, sect, sl) || v[sl] != ':') return 0;
            const char *kp = p + sl + 1, *vp = v + sl + 1;
            size_t kl = strlen(key);
            for (;;) {
                while (*kp == ' ') kp++;
                while (*vp == ' ') vp++;
                if (*kp == '\n' || !*kp || *vp == '\n' || !*vp) return 0;
                const char *ke = kp;
                while (*ke && *ke != ' ' && *ke != '\n') ke++;
                if ((size_t)(ke - kp) == kl && !strncmp(kp, key, kl)) return strtod(vp, NULL);
                kp = ke;
                while (*vp && *vp != ' ' && *vp != '\n') vp++;
            }
        }
        p = strchr(p, '\n');
        if (p) p++;
    }
    return 0;
}

/* "Key value" lines (snmp6) */
static double kv(const char *buf, const char *key) {
    size_t kl = strlen(key);
    for (const char *p = buf; p && *p;) {
        if (!strncmp(p, key, kl) && (p[kl] == ' ' || p[kl] == '\t')) return strtod(p + kl, NULL);
        p = strchr(p, '\n');
        if (p) p++;
    }
    return 0;
}

/* only the 4th column (the state) of each line; nothing else is kept */
static void count_states(const char *path, float *out) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return; }
    while (fgets(line, sizeof line, f)) {
        unsigned s;
        if (sscanf(line, "%*s %*s %*s %x", &s) == 1 && s < NST) out[s] += 1;
    }
    fclose(f);
}

static void read_protos_live(double dt) {
    static char snmp[8192], six[16384], net[16384], ss[1024], ss6[1024];
    slurp("/proc/net/snmp", snmp, sizeof snmp);
    slurp("/proc/net/snmp6", six, sizeof six);
    slurp("/proc/net/netstat", net, sizeof net);
    slurp("/proc/net/sockstat", ss, sizeof ss);
    slurp("/proc/net/sockstat6", ss6, sizeof ss6);
    double c[NPR];
    c[PR_TCP_IN] = mib(snmp, "Tcp", "InSegs");
    c[PR_TCP_OUT] = mib(snmp, "Tcp", "OutSegs");
    c[PR_RETX] = mib(snmp, "Tcp", "RetransSegs");
    c[PR_AOPEN] = mib(snmp, "Tcp", "ActiveOpens");
    c[PR_POPEN] = mib(snmp, "Tcp", "PassiveOpens");
    c[PR_TIMEOUT] = mib(net, "TcpExt", "TCPTimeouts");
    c[PR_LDROP] = mib(net, "TcpExt", "ListenDrops");
    c[PR_UDP_IN] = mib(snmp, "Udp", "InDatagrams") + kv(six, "Udp6InDatagrams");
    c[PR_UDP_OUT] = mib(snmp, "Udp", "OutDatagrams") + kv(six, "Udp6OutDatagrams");
    c[PR_UDP_ERR] = mib(snmp, "Udp", "InErrors") + kv(six, "Udp6InErrors");
    c[PR_ICMP_IN] = mib(snmp, "Icmp", "InMsgs") + kv(six, "Icmp6InMsgs");
    c[PR_ICMP_OUT] = mib(snmp, "Icmp", "OutMsgs") + kv(six, "Icmp6OutMsgs");
    c[PR_ICMP_UNR] = mib(snmp, "Icmp", "InDestUnreachs") + kv(six, "Icmp6InDestUnreachs");
    c[PR_IP_IN] = mib(snmp, "Ip", "InReceives") + kv(six, "Ip6InReceives");
    c[PR_IP_OUT] = mib(snmp, "Ip", "OutRequests") + kv(six, "Ip6OutRequests");
    c[PR_IP_DISC] = mib(snmp, "Ip", "InDiscards") + kv(six, "Ip6InDiscards");
    c[PR_REASM] = mib(snmp, "Ip", "ReasmReqds") + kv(six, "Ip6ReasmReqds");
    for (int i = 0; i < NPR; i++) {
        pr[i] = have_pcum && dt > 0 ? (float)fmax(0, (c[i] - pcum[i]) / dt) : 0;
        pcum[i] = c[i];
    }
    have_pcum = 1;
    lv[LV_ESTAB] = (float)mib(snmp, "Tcp", "CurrEstab");
    int a = 0, b = 0, o = 0, t = 0;
    const char *p;
    lv[LV_TCP_INUSE] = lv[LV_ORPHAN] = lv[LV_TW] = lv[LV_UDP_INUSE] = lv[LV_RAW_INUSE] = 0;
    if ((p = strstr(ss, "TCP: inuse")) && sscanf(p, "TCP: inuse %d orphan %d tw %d", &a, &o, &t) == 3) {
        lv[LV_TCP_INUSE] = (float)a; lv[LV_ORPHAN] = (float)o; lv[LV_TW] = (float)t;
    }
    if ((p = strstr(ss, "UDP: inuse")) && sscanf(p, "UDP: inuse %d", &b) == 1) lv[LV_UDP_INUSE] = (float)b;
    if ((p = strstr(ss, "RAW: inuse")) && sscanf(p, "RAW: inuse %d", &b) == 1) lv[LV_RAW_INUSE] = (float)b;
    if ((p = strstr(ss, "sockets: used")) && sscanf(p, "sockets: used %d", &b) == 1) lv[LV_SOCK_USED] = (float)b;
    if ((p = strstr(ss6, "TCP6: inuse")) && sscanf(p, "TCP6: inuse %d", &b) == 1) lv[LV_TCP_INUSE] += (float)b;
    if ((p = strstr(ss6, "UDP6: inuse")) && sscanf(p, "UDP6: inuse %d", &b) == 1) lv[LV_UDP_INUSE] += (float)b;
    if ((p = strstr(ss6, "RAW6: inuse")) && sscanf(p, "RAW6: inuse %d", &b) == 1) lv[LV_RAW_INUSE] += (float)b;
    memset(st, 0, sizeof st);
    count_states("/proc/net/tcp", st);
    count_states("/proc/net/tcp6", st);
}

/* ---- the synthetic traffic model (--shot, --bench): a pure function of
 * the seed and the second, so any moment can be rendered exactly ---- */
static void synth_ifaces(void) {
    static const struct { const char *name; int kind, up, speed, nagg; } def[MAXIF] = {
        {"wlan0", IK_WIFI, 1, 0, 0}, {"eth0", IK_ETH, 1, 1000, 0}, {"wwan0", IK_WWAN, 0, 0, 0},
        {"wg0", IK_TUN, 1, 0, 0}, {"docker0", IK_BRIDGE, 1, 0, 0}, {"veth*", IK_VETH, 1, 0, 3},
        {"lo", IK_LOOP, 1, 0, 0},
    };
    Iface nw[MAXIF];
    memset(nw, 0, sizeof nw);
    for (int i = 0; i < MAXIF; i++) {
        snprintf(nw[i].name, sizeof nw[i].name, "%s", def[i].name);
        nw[i].kind = def[i].kind; nw[i].up = def[i].up; nw[i].speed = def[i].speed; nw[i].nagg = def[i].nagg;
        nw[i].sig_q = def[i].kind == IK_WIFI ? 0.7f : -1;
        nw[i].hfrom = LONG_MIN / 2;
    }
    set_ifaces(nw, MAXIF);
}

/* an episode: on for dur seconds out of every period, at a seeded phase; 0..1 with soft edges */
static float episode(uint32_t ch, long k, int period, int dur) {
    long w = (long)floor((double)k / period);
    float out = 0;
    for (long ww = w - 1; ww <= w; ww++) {
        if (hrand(ch + 7, ww) < 0.25f) continue;         /* some windows are quiet */
        long s = ww * period + (long)(hrand(ch, ww) * (period - dur));
        long d = dur / 2 + (long)(hrand(ch + 3, ww) * dur / 2);
        float x = (float)(k - s);
        if (x >= 0 && x < d) out = fmaxf(out, fminf(1, fminf(x / 2.5f + 0.2f, (d - x) / 2.0f)));
    }
    return out;
}

static void synth(long k) {
    int idle = traffic == T_IDLE;
    double x = (double)k;
    float rb[MAXIF], tb[MAXIF], rsz[MAXIF], tsz[MAXIF];
    for (int i = 0; i < MAXIF; i++) { rb[i] = tb[i] = 0; rsz[i] = tsz[i] = 600; }
    float dl = idle ? 0 : episode(11, k, 47, 26), ul = idle ? 0 : episode(12, k, 71, 14);
    float build = idle ? 0 : episode(13, k, 59, 18);
    /* wlan0 */
    rb[0] = powf(10, (idle ? 2.1f : 3.4f) + (idle ? 0.7f : 1.3f) * vnoise(20, x / 6)) + dl * powf(10, 6.2f + 0.6f * vnoise(21, x / 3));
    tb[0] = powf(10, (idle ? 1.6f : 3.0f) + 1.1f * vnoise(22, x / 5)) + rb[0] * 0.035f + ul * powf(10, 5.6f + 0.4f * vnoise(23, x / 2));
    rsz[0] = 160 + 1250 * clampf(dl * 1.4f, 0, 1); tsz[0] = 90 + 1300 * ul;
    /* eth0 */
    rb[1] = powf(10, (idle ? 1.4f : 3.9f) + (idle ? 1.0f : 1.6f) * vnoise(30, x / 9));
    tb[1] = powf(10, (idle ? 1.2f : 3.6f) + (idle ? 0.8f : 1.5f) * vnoise(31, x / 8));
    rsz[1] = 400; tsz[1] = 300;
    /* wg0 */
    float ka = hrand(40, k) < (idle ? 0.06f : 0.0f) ? 1 : 0;      /* keepalive blips */
    rb[3] = idle ? 30 * ka + 2 : powf(10, 2.6f + 1.4f * vnoise(41, x / 7));
    tb[3] = idle ? 30 * ka + 2 : powf(10, 2.5f + 1.3f * vnoise(42, x / 7));
    rsz[3] = tsz[3] = 180;
    /* docker0 and its veths */
    rb[4] = idle ? 0 : powf(10, 1.5f + vnoise(50, x / 4)) + build * powf(10, 5.7f + 0.5f * vnoise(51, x));
    tb[4] = idle ? 0 : powf(10, 1.4f + vnoise(52, x / 4)) + build * powf(10, 4.4f + 0.5f * vnoise(53, x));
    rsz[4] = 1100; tsz[4] = 200;
    rb[5] = rb[4] * 0.97f; tb[5] = tb[4] * 1.02f; rsz[5] = 1100; tsz[5] = 200;
    /* lo */
    rb[6] = tb[6] = powf(10, (idle ? 2.0f : 3.6f) + (idle ? 0.8f : 1.5f) * vnoise(60, x / 5));
    rsz[6] = tsz[6] = 380;
    double pin = 0, pout = 0;
    for (int i = 0; i < nif && i < MAXIF; i++) {
        Iface *f = &ifs[i];
        if (!f->up) { rb[i] = tb[i] = 0; }
        f->r[RXB] = rb[i]; f->r[TXB] = tb[i];
        f->r[RXP] = rb[i] / rsz[i]; f->r[TXP] = tb[i] / tsz[i];
        /* now and then a dropped or errored frame */
        f->r[RXD] = f->r[TXD] = f->r[RXE] = f->r[TXE] = 0;
        if (f->up && f->kind != IK_LOOP) {
            float h = hrand(70 + (uint32_t)i, k);
            float pr_ = idle ? 0.004f : (f->kind == IK_WIFI ? 0.05f + 0.25f * dl : 0.012f);
            if (h < pr_) f->r[(h < pr_ * 0.5f) ? TXD : RXD] = 1 + (float)(int)(hrand(80 + (uint32_t)i, k) * 3);
            if (f->kind == IK_WIFI && hrand(90, k) < pr_ * 0.2f) f->r[RXE] = 1;
        }
        if (f->kind == IK_WIFI) {
            f->sig_q = 0.6f + 0.18f * vnoise(24, x / 13);
            f->sig_dbm = -64 + 14 * (f->sig_q - 0.6f) / 0.18f;
        }
        pin += f->r[RXP]; pout += f->r[TXP];
    }
    float ping = idle ? 0 : episode(14, k, 83, 12);
    pr[PR_TCP_IN] = (float)(pin * 0.83); pr[PR_TCP_OUT] = (float)(pout * 0.80);
    pr[PR_UDP_IN] = (float)(pin * 0.15); pr[PR_UDP_OUT] = (float)(pout * 0.18);
    pr[PR_ICMP_IN] = (float)(pin * 0.002) + ping * 1.0f + (hrand(15, k) < 0.1f ? 1 : 0);
    pr[PR_ICMP_OUT] = (float)(pout * 0.002) + ping * 1.0f;
    pr[PR_ICMP_UNR] = hrand(16, k) < (idle ? 0.02f : 0.12f) ? 1 : 0;
    float loss = idle ? 0.0008f : powf(10, -3.5f + 1.1f * episode(17, k, 37, 10) + 0.5f * vnoise(18, x / 11));
    pr[PR_RETX] = pr[PR_TCP_OUT] * loss;
    if (idle) pr[PR_RETX] = hrand(19, k) < 0.008f ? 1 : 0;
    pr[PR_AOPEN] = idle ? (hrand(25, k) < 0.08f ? 1 : 0) : 0.4f + 6 * vnoise(26, x / 4) * (0.3f + build);
    pr[PR_POPEN] = idle ? 0 : 0.6f * vnoise(27, x / 6) + (hrand(28, k) < 0.2f ? 1 : 0);
    pr[PR_TIMEOUT] = pr[PR_RETX] * 0.08f;
    pr[PR_LDROP] = 0;
    pr[PR_UDP_ERR] = hrand(29, k) < (idle ? 0.01f : 0.05f) ? 1 : 0;
    pr[PR_IP_IN] = (float)pin; pr[PR_IP_OUT] = (float)pout;
    pr[PR_IP_DISC] = 0; pr[PR_REASM] = hrand(33, k) < 0.03f ? 2 : 0;
    /* connection states */
    memset(st, 0, sizeof st);
    st[1] = roundf(idle ? 7 + 3 * vnoise(34, x / 40) : 22 + 26 * vnoise(34, x / 25) + 18 * build);
    st[6] = roundf(idle ? (hrand(35, k / 30) < 0.4f ? 1 : 0) : 4 + 30 * vnoise(35, x / 15) * (0.4f + build));
    st[10] = 14;
    st[8] = idle ? 1 : roundf(1 + 2 * vnoise(36, x / 20));
    st[2] = idle ? 0 : (hrand(37, k) < 0.25f ? roundf(1 + 2 * hrand(38, k)) : 0);
    st[3] = idle ? 0 : (hrand(39, k) < 0.08f ? 1 : 0);
    st[4] = idle ? 0 : (hrand(43, k) < 0.15f ? 1 : 0);
    st[5] = idle ? 0 : roundf(2 * vnoise(44, x / 9));
    st[9] = idle ? 0 : (hrand(45, k) < 0.1f ? 1 : 0);
    lv[LV_ESTAB] = st[1] + st[8];
    lv[LV_TW] = st[6];
    lv[LV_TCP_INUSE] = st[1] + st[2] + st[3] + st[4] + st[5] + st[8] + st[9] + st[10];
    lv[LV_ORPHAN] = st[4] + st[9];
    lv[LV_UDP_INUSE] = roundf(18 + 6 * vnoise(46, x / 60));
    lv[LV_RAW_INUSE] = 1 + (ping > 0);
    lv[LV_SOCK_USED] = roundf(1300 + 160 * vnoise(47, x / 90));
}

/* one second's sample into the rings */
static long last_k = LONG_MIN;
static void push_hist(long k) {
    for (int i = 0; i < nif; i++) {
        Iface *f = &ifs[i];
        int j = (int)(((k % HIST) + HIST) % HIST);
        f->hrx[j] = f->r[RXB]; f->htx[j] = f->r[TXB];
        float ev = f->r[RXD] + f->r[TXD] + f->r[RXE] + f->r[TXE];
        f->hev[j] = (unsigned char)fminf(ev, 255);
    }
    int j = (int)(((k % HIST) + HIST) % HIST);
    for (int i = 0; i < NPR; i++) hpr[i][j] = pr[i];
}

static void sample(long k) {
    if (traffic == T_LIVE) {
        double t = now(), dt = last_sample_t ? t - last_sample_t : 0;
        last_sample_t = t;
        read_ifaces_live(dt, k);
        read_protos_live(dt);
        if (dt <= 0) {
            /* the first read only primes the counters: history starts next second */
            for (int i = 0; i < nif; i++) ifs[i].hfrom = k + 1;
            pr_from = k + 1;
        }
    } else {
        synth(k);
    }
    push_hist(k);
}

/* ---- particles: packets along the conduits ---- */
enum { PK_PKT, PK_RETX, PK_BOUNCE, PK_FALL };
typedef struct {
    float s, v, size, age, life, x, y, vx, vy, bx, by;
    short ifi, proto;
    char dir, kind, err;
} Part;
#define MAXP 4096
static Part parts[MAXP];
static int np;

/* a conduit: a polyline bottom (the wire) to top (the sockets); gap
 * segments (inside a box or the bus) have no length */
#define NPT 40
typedef struct {
    float x[NPT], y[NPT], s[NPT], run0[NPT];
    unsigned char gap[NPT];
    int n;
    float len, s_ifb, s_ift, s_ipt;
} Path;
static Path paths[MAXIF][3][2];        /* [iface][proto][0 rx lane, 1 tx lane] */
static float acc[MAXIF][3][2], acc_retx, acc_drop[MAXIF][2];
static float travel_t = 3.4f;          /* seconds wire to sockets */

static void path_pos(const Path *p, float s, float *x, float *y, float *run0) {
    if (s <= 0) { *x = p->x[0]; *y = p->y[0]; if (run0) *run0 = 0; return; }
    for (int i = 0; i + 1 < p->n; i++) {
        if (p->gap[i] || s > p->s[i + 1]) continue;
        float L = p->s[i + 1] - p->s[i];
        float f = L > 1e-4f ? (s - p->s[i]) / L : 0;
        *x = p->x[i] + (p->x[i + 1] - p->x[i]) * f;
        *y = p->y[i] + (p->y[i + 1] - p->y[i]) * f;
        if (run0) *run0 = p->run0[i];
        return;
    }
    *x = p->x[p->n - 1]; *y = p->y[p->n - 1];
    if (run0) *run0 = p->run0[p->n - 2];
}

/* the end of the visible run containing s */
static float run_end(const Path *p, float s) {
    for (int i = 0; i + 1 < p->n; i++) {
        if (p->gap[i] || s > p->s[i + 1]) continue;
        int j = i;
        while (j + 1 < p->n && !p->gap[j]) j++;
        return p->s[j];
    }
    return p->len;
}

/* how many particles a second for a packet rate: log-scaled */
static float emit_rate(float pps) { return pps <= 0 ? 0 : 1.45f * powf(log1pf(pps), 1.3f); }
static float rare_rate(float x) { return x <= 0 ? 0 : x < 3 ? x : 3 + 2 * logf(x / 3); }

static float u = 1;                    /* the layout unit, px */
static float dpr = 1;                  /* buffer px per logical px (--layer at a HiDPI scale) */

static void spawn(int ifi, int p, int dir, int kind, float sz) {
    if (np >= MAXP) return;
    Part *q = &parts[np++];
    memset(q, 0, sizeof *q);
    const Path *pa = &paths[ifi][p][dir];
    q->ifi = (short)ifi; q->proto = (short)p; q->dir = (char)dir; q->kind = (char)kind;
    q->v = pa->len / (travel_t * (0.85f + 0.3f * frand()));
    q->s = dir == 0 ? frand() * 4 * u : pa->len - frand() * 4 * u;
    q->size = sz;
}

static void step_particles(float dt) {
    if (!nif) return;
    /* protocol shares, by direction */
    float in3[3] = {pd[PR_TCP_IN], pd[PR_UDP_IN], pd[PR_ICMP_IN]}, out3[3] = {pd[PR_TCP_OUT], pd[PR_UDP_OUT], pd[PR_ICMP_OUT]};
    float sin_ = in3[0] + in3[1] + in3[2], sout = out3[0] + out3[1] + out3[2];
    float wtx = 0;
    for (int i = 0; i < nif; i++) if (ifs[i].kind != IK_LOOP) wtx += ifs[i].d[TXP];
    for (int i = 0; i < nif; i++) {
        Iface *f = &ifs[i];
        if (!f->up) continue;
        for (int dir = 0; dir < 2; dir++) {
            float pps = f->d[dir ? TXP : RXP], bytes = f->d[dir ? TXB : RXB];
            float avg = pps > 0.01f ? bytes / pps : 200;
            float sz = (1.5f + 1.9f * sqrtf(clampf(avg / 1500, 0, 1))) * u;
            float *sh = dir ? out3 : in3, tot = dir ? sout : sin_;
            for (int p = 0; p < 3; p++) {
                float share = tot > 0 ? sh[p] / tot : (p == 0);
                float e = emit_rate(pps * share);
                if (p == 0 && e < 0.07f) e = 0.07f;          /* keepalive trickle */
                acc[i][p][dir] -= e * dt;
                while (acc[i][p][dir] <= 0) {
                    spawn(i, p, dir, PK_PKT, sz * (0.8f + 0.4f * frand()));
                    acc[i][p][dir] += -logf(fmaxf(frand(), 1e-4f));
                }
            }
            /* drops and errors, rare: they fall off the side */
            float de = rare_rate(f->d[dir ? TXD : RXD] + f->d[dir ? TXE : RXE]);
            acc_drop[i][dir] -= de * dt;
            while (acc_drop[i][dir] <= 0) {
                if (de > 0) {
                    spawn(i, (int)(frand() * 1.3f) % 3, dir, PK_PKT, sz);
                    if (np) { parts[np - 1].kind = PK_PKT; parts[np - 1].err = 1 + (f->d[dir ? TXE : RXE] > f->d[dir ? TXD : RXD]); }
                }
                acc_drop[i][dir] += -logf(fmaxf(frand(), 1e-4f)) * (de > 0 ? 1 : 1e6f);
            }
        }
    }
    /* retransmits, down the TCP tx lanes, weighted by each interface's tx */
    float re = rare_rate(pd[PR_RETX]);
    acc_retx -= re * dt;
    while (acc_retx <= 0) {
        if (re > 0 && wtx > 0) {
            float pick = frand() * wtx;
            int ii = 0;
            for (int i = 0; i < nif; i++) {
                if (ifs[i].kind == IK_LOOP) continue;
                ii = i;
                if ((pick -= ifs[i].d[TXP]) <= 0) break;
            }
            if (ifs[ii].up) spawn(ii, 0, 1, PK_RETX, 2.6f * u);
        }
        acc_retx += -logf(fmaxf(frand(), 1e-4f)) * (re > 0 ? 1 : 1e6f);
    }
    /* move */
    float g = 520 * u;
    for (int i = 0; i < np;) {
        Part *q = &parts[i];
        const Path *pa = &paths[q->ifi][q->proto][(int)q->dir];
        int dead = 0;
        q->age += dt;
        if (q->kind == PK_FALL) {
            q->vy += g * dt;
            q->x += q->vx * dt; q->y += q->vy * dt;
            dead = q->age > q->life;
        } else if (q->kind == PK_BOUNCE) {
            q->s += q->v * dt;
            dead = q->age > q->life;
        } else {
            q->s += (q->dir == 0 ? 1 : -1) * q->v * dt;
            if (q->err) {
                /* rx: falls off where it reaches its interface; tx: as it leaves it */
                int fall = q->dir == 0 ? q->s >= pa->s_ifb : q->s <= pa->s_ift;
                if (fall) {
                    path_pos(pa, q->dir == 0 ? pa->s_ifb : pa->s_ift + 0.01f, &q->x, &q->y, NULL);
                    q->kind = PK_FALL;
                    q->age = 0; q->life = 1.5f;
                    float side = frand() < 0.5f ? -1 : 1;
                    q->vx = side * (50 + 90 * frand()) * u;
                    q->vy = -(60 + 80 * frand()) * u;
                }
            } else if (q->kind == PK_RETX && q->s <= pa->s_ipt) {
                q->kind = PK_BOUNCE;
                q->s = pa->s_ipt + 0.01f;
                path_pos(pa, q->s, &q->bx, &q->by, NULL);
                q->age = 0; q->life = 0.75f; q->v *= 0.75f;
            }
            dead = q->s > pa->len || q->s < 0;
        }
        if (dead) parts[i] = parts[--np];
        else i++;
    }
}

/* ---- the congestion-window sketch: AIMD, illustrative ---- */
#define CWN 360                        /* 36 s at 10 Hz */
static float cw = 20, cw_target = 40, cwh[CWN], cw_loss[CWN];
static long cw_tick = LONG_MIN;
static float cw_p;                     /* smoothed loss fraction */

static void step_cwnd(double t, float dt) {
    /* a trickle of segments makes the ratio noise: count at least 50 a second */
    float p = pd[PR_RETX] / fmaxf(pd[PR_TCP_OUT], 50);
    cw_p += (p - cw_p) * (1 - expf(-dt / 2.0f));
    /* Mathis: W ~ 1.22 / sqrt(p) segments */
    float wt = clampf(1.22f / sqrtf(fmaxf(cw_p, 1.2e-4f)), 6, 100);
    float T = 2.2f + wt / 28;              /* seconds per sawtooth tooth */
    cw += (wt * 0.5f / T) * dt;
    int loss = 0;
    if (cw >= cw_target) { cw *= 0.5f; loss = 1; cw_target = wt * (0.88f + 0.24f * frand()); }
    if (cw_target > wt * 1.15f) cw_target = wt * (0.9f + 0.2f * frand());
    long tick = (long)floor(t * 10);
    if (cw_tick == LONG_MIN) {
        cw_tick = tick - 1;
        for (int i = 0; i < CWN; i++) cwh[i] = -1;
    }
    static int pending_loss;
    pending_loss |= loss;
    while (cw_tick < tick) {
        cw_tick++;
        int j = (int)(((cw_tick % CWN) + CWN) % CWN);
        cwh[j] = cw;
        cw_loss[j] = (float)pending_loss;
        pending_loss = 0;
    }
}

/* ease the displayed values toward the latest sample */
static void smooth(float dt) {
    float a = 1 - expf(-dt / 0.55f), b = 1 - expf(-dt / 1.2f);
    for (int i = 0; i < nif; i++) {
        Iface *f = &ifs[i];
        for (int c = 0; c < NIC; c++) f->d[c] += (f->r[c] - f->d[c]) * a;
        float act = clampf(log10f(1 + f->d[RXB] + f->d[TXB]) / 7.5f, 0, 1);
        f->act += (act - f->act) * a;
        if (f->sig_q >= 0) f->sig_qd += (f->sig_q - f->sig_qd) * b;
    }
    for (int i = 0; i < NPR; i++) pd[i] += (pr[i] - pd[i]) * a;
    for (int i = 0; i < NLV; i++) lvd[i] += (lv[i] - lvd[i]) * b;
    for (int i = 0; i < NST; i++) std_[i] += (st[i] - std_[i]) * b;
}

static void snap_smooth(void) {
    for (int i = 0; i < nif; i++) {
        memcpy(ifs[i].d, ifs[i].r, sizeof ifs[i].r);
        ifs[i].sig_qd = ifs[i].sig_q;
        ifs[i].act = clampf(log10f(1 + ifs[i].d[RXB] + ifs[i].d[TXB]) / 7.5f, 0, 1);
    }
    memcpy(pd, pr, sizeof pr);
    memcpy(lvd, lv, sizeof lv);
    memcpy(std_, st, sizeof st);
}

static double sim_t;
static void sim_step(double t, float h) {
    long k = (long)floor(t);
    if (k != last_k) {
        sample(k);
        if (last_k == LONG_MIN) snap_smooth();
        last_k = k;
    }
    smooth(h);
    step_particles(h);
    step_cwnd(t, h);
}

static void advance(double t, int catchup) {
    const float h = 1.0f / 60;
    if (!catchup && t - sim_t > 1.0) sim_t = t - 1.0;
    while (sim_t + h <= t) {
        sim_t += h;
        sim_step(sim_t, h);
    }
}

/* ---- glyph atlas: printable ASCII and a few extras, in each text style
 * used, rasterised at its exact pixel size ---- */
#define NGL 104
static const unsigned extra_cp[] = {0x2191, 0x2193, 0xB7, 0xD7, 0x2013, 0x2009, 0x2192, 0x25CF};
#define UP "\x01"
#define DN "\x02"
#define MID "\x03"
#define TIMES "\x04"
#define NDASH "\x05"
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NGL]; } Style;
#define MAXSTYLE 48
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
enum { F_REG, F_SEMI, F_LIGHT, F_MONO, NFACE };
static FT_Face faces[NFACE];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        char pat_s[300];
        static const char *wt[] = {"regular", "semibold", "light", "regular"};
        snprintf(pat_s, sizeof pat_s, "%s:weight=%s:fontformat=CFF,TrueType", i == F_MONO ? opt_mono : opt_font, wt[i]);
        FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
        if (i == F_MONO) FcPatternAddInteger(pat, FC_SPACING, FC_MONO);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        int index = 0;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        FcPatternGetInteger(m, FC_INDEX, 0, &index);
        if (FT_New_Face(ft, (const char *)file, index, &faces[i])) die("can't load the font");
        FcPatternDestroy(m);
        FcPatternDestroy(pat);
    }
}

static int style(int face, float px) {
    int p = (int)lroundf(px);
    if (p < 7) p = 7;
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
        for (int c = 0; c < NGL; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_ULong code = c < 95 ? (FT_ULong)(32 + c) : c == 95 ? 0xB0 : extra_cp[c - 96];
            FT_UInt gi = FT_Get_Char_Index(f, code);
            if (!gi && c >= 96) gi = FT_Get_Char_Index(f, c == 98 ? '.' : c == 99 ? 'x' : c == 100 ? '-' : ' ');
            if (FT_Load_Glyph(f, gi, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
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
    int i = c == 0x7f ? 95 : c >= 32 && c < 127 ? c - 32 : c >= 1 && c <= 8 ? 95 + c : 0;
    return &styles[s].g[i];
}

static float text_width(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p; p++) w += glyph(s, (unsigned char)*p)->adv + (p[1] ? track : 0);
    return w;
}

static float cap_h(int s) { return glyph(s, 'H')->top; }

/* text quads: x, y, u, v, r, g, b, a per vertex */
static float *tv;
static int tvn, tvcap;

static void quad_v(float x, float y, float uu, float v, const float *c, float a) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 16384) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = uu; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = a;
    tvn += 8;
}

/* set str with its baseline at y; align -1 left, 0 centred, 1 right of x */
static float text(int s, float x, float y, int align, const char *str, const float *col, float a, float track) {
    float w = text_width(s, str, track);
    float pen = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    pen = roundf(pen);
    y = roundf(y);
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph(s, (unsigned char)*p);
        if (g->w) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            quad_v(x0, y0, u0, v0, col, a); quad_v(x1, y0, u1, v0, col, a); quad_v(x1, y1, u1, v1, col, a);
            quad_v(x0, y0, u0, v0, col, a); quad_v(x1, y1, u1, v1, col, a); quad_v(x0, y1, u0, v1, col, a);
        }
        pen += g->adv + track;
    }
    return w;
}

/* ---- shapes: one instanced quad each, drawn as a signed distance ---- */
enum { SH_SEG, SH_BOX, SH_RING, SH_DOT };
typedef struct { float a[4], b[4], c[4], e[4]; } Prim;
static Prim *pv;
static int pn, pcap;

static Prim *prim(int kind, const float *col, float alpha, float glow, float gs) {
    if (pn == pcap) pv = realloc(pv, (size_t)(pcap = pcap ? pcap * 2 : 8192) * sizeof *pv);
    Prim *p = &pv[pn++];
    memset(p, 0, sizeof *p);
    p->b[2] = (float)kind;
    p->b[3] = glow;
    p->c[0] = col[0]; p->c[1] = col[1]; p->c[2] = col[2]; p->c[3] = alpha;
    p->e[3] = light ? gs * 0.35f : gs;
    return p;
}
/* a line w px wide */
static void seg(float x0, float y0, float x1, float y1, float w, const float *col, float a, float glow, float gs) {
    Prim *p = prim(SH_SEG, col, a, glow, gs);
    p->a[0] = x0; p->a[1] = y0; p->a[2] = x1; p->a[3] = y1; p->b[0] = w * 0.5f;
}
static void dash(float x0, float y0, float x1, float y1, float w, float period, float duty, const float *col, float a) {
    Prim *p = prim(SH_SEG, col, a, 0, 0);
    p->a[0] = x0; p->a[1] = y0; p->a[2] = x1; p->a[3] = y1; p->b[0] = w * 0.5f;
    p->e[0] = period; p->e[1] = duty;
}
/* a rounded box (top-left x, y; size w, h); stroke 0 fills it */
static void box(float x, float y, float w, float h, float r, float stroke, const float *col, float a, float glow, float gs) {
    Prim *p = prim(SH_BOX, col, a, glow, gs);
    p->a[0] = x + w / 2; p->a[1] = y + h / 2; p->a[2] = w / 2; p->a[3] = h / 2;
    p->b[0] = fminf(r, fminf(w, h) / 2); p->b[1] = stroke;
}
/* a ring (or with a1 > a0, an arc from a0 to a1, radians clockwise from +x on screen) */
static void ring(float cx, float cy, float r, float stroke, float a0, float a1, const float *col, float a, float glow, float gs) {
    Prim *p = prim(SH_RING, col, a, glow, gs);
    p->a[0] = cx; p->a[1] = cy; p->b[0] = r; p->b[1] = stroke; p->e[0] = a0; p->e[1] = a1;
}
static void disc(float cx, float cy, float r, const float *col, float a, float glow, float gs) {
    Prim *p = prim(SH_RING, col, a, glow, gs);
    p->a[0] = cx; p->a[1] = cy; p->b[0] = r;
}
/* a glowing particle with a hot core */
static void dot(float cx, float cy, float r, const float *col, float a, float glow, float gs) {
    Prim *p = prim(SH_DOT, col, a, glow, gs);
    p->a[0] = cx; p->a[1] = cy; p->b[0] = r;
}

/* ---- number formatting ---- */
static void fmt_bytes(char *b, size_t n, double v) {
    static const char *un[] = {"B/s", "kB/s", "MB/s", "GB/s"};
    int i = 0;
    while (v >= 999.5 && i < 3) { v /= 1000; i++; }
    if (i == 0) snprintf(b, n, "%.0f %s", v, un[i]);
    else snprintf(b, n, v < 9.995 ? "%.2f %s" : v < 99.95 ? "%.1f %s" : "%.0f %s", v, un[i]);
}
static void fmt_rate(char *b, size_t n, double v) {
    if (v < 0.05) snprintf(b, n, "0");
    else if (v < 9.95) snprintf(b, n, "%.1f", v);
    else if (v < 999.5) snprintf(b, n, "%.0f", v);
    else if (v < 9995) snprintf(b, n, "%.2fk", v / 1e3);
    else if (v < 99950) snprintf(b, n, "%.1fk", v / 1e3);
    else if (v < 999500) snprintf(b, n, "%.0fk", v / 1e3);
    else snprintf(b, n, "%.2fM", v / 1e6);
}

/* ---- layout ---- */
typedef struct { float x, y, w, h; } Rect;
static Rect r_head, r_stack, r_scopes, r_states, r_counters;
static int st_host, st_sub, st_clock, st_away, st_lnum, st_lname, st_tiny, st_name, st_mono, st_monos, st_proto,
    st_big, st_title, st_iname, st_clock_s;

static struct {
    float gx, cx0, cx1;
    float ys0, ys1, yt0, yt1, yi0, yi1, yf0, yf1, yw;
    Rect ifb[MAXIF], trb[3], skb[3];
} SG;

static int sc_if[MAXIF], sc_n;

static void build_paths(void) {
    float cw_ = SG.cx1 - SG.cx0;
    (void)cw_;
    for (int i = 0; i < nif; i++) {
        const Rect *b = &SG.ifb[i];
        float bcx = b->x + b->w / 2;
        for (int p = 0; p < 3; p++)
            for (int d = 0; d < 2; d++) {
                Path *pa = &paths[i][p][d];
                int n = 0;
                float xa = bcx + (p - 1) * 0.25f * b->w + (d ? 1 : -1) * 0.045f * b->w;
                const Rect *t = &SG.trb[p];
                float xb = t->x + t->w / 2 + (((2 * i + d) + 0.5f) / (2.0f * nif) - 0.5f) * 0.78f * t->w;
                /* S0: from the wire, fanning out to the lanes */
                float wx = bcx;
                for (int j = 0; j < 6; j++) {
                    float f = j / 5.0f, e = f * f * (3 - 2 * f);
                    pa->x[n] = wx + (xa - wx) * e;
                    pa->y[n] = SG.yw + (SG.yf1 - SG.yw) * f;
                    pa->gap[n++] = 0;
                }
                pa->gap[n - 1] = 1;
                /* S1: interface to the bus */
                pa->x[n] = xa; pa->y[n] = SG.yf0; pa->gap[n++] = 0;
                pa->x[n] = xa; pa->y[n] = SG.yi1; pa->gap[n++] = 1;
                /* S2: the bus to the transport box, an S-curve */
                int nb = 22;
                for (int j = 0; j < nb; j++) {
                    float f = j / (float)(nb - 1);
                    float y0 = SG.yi0, y1 = SG.yt1, ym = (y0 + y1) / 2;
                    float m = 1 - f;
                    /* cubic: (xa, y0) (xa, ym) (xb, ym) (xb, y1) */
                    float bx = m * m * m * xa + 3 * m * m * f * xa + 3 * m * f * f * xb + f * f * f * xb;
                    float by = m * m * m * y0 + 3 * m * m * f * ym + 3 * m * f * f * ym + f * f * f * y1;
                    pa->x[n] = bx; pa->y[n] = by; pa->gap[n++] = 0;
                }
                pa->gap[n - 1] = 1;
                /* S3: up into the sockets */
                pa->x[n] = xb; pa->y[n] = SG.yt0; pa->gap[n++] = 0;
                pa->x[n] = xb; pa->y[n] = SG.ys1; pa->gap[n++] = 0;
                pa->n = n;
                float s = 0, r0 = 0;
                for (int j = 0; j < n; j++) {
                    pa->s[j] = s;
                    if (j + 1 < n) {
                        pa->run0[j] = r0;
                        if (!pa->gap[j]) s += hypotf(pa->x[j + 1] - pa->x[j], pa->y[j + 1] - pa->y[j]);
                        else r0 = s;
                    }
                }
                pa->len = s;
                pa->s_ifb = pa->s[5];
                pa->s_ift = pa->s[6];
                pa->s_ipt = pa->s[8];
            }
    }
}

static void layout_stack(Rect r) {
    float gut = fmaxf(84 * u, fminf(r.w * 0.11f, 150 * u));
    SG.gx = r.x;
    SG.cx0 = r.x + gut;
    SG.cx1 = r.x + r.w;
    float y = r.y, h = r.h;
    SG.ys0 = y;
    SG.ys1 = y + h * 0.16f;
    SG.yt0 = y + h * 0.235f;
    SG.yt1 = y + h * 0.40f;
    SG.yi0 = y + h * 0.585f;
    SG.yi1 = y + h * 0.645f;
    SG.yf0 = y + h * 0.745f;
    SG.yf1 = y + h * 0.935f;
    SG.yw = y + h;
    float cw_ = SG.cx1 - SG.cx0;
    for (int p = 0; p < 3; p++) {
        float bw = cw_ * 0.29f, cx = SG.cx0 + cw_ * (1 + 2 * p) / 6.0f;
        SG.trb[p] = (Rect){cx - bw / 2, SG.yt0, bw, SG.yt1 - SG.yt0};
        SG.skb[p] = (Rect){cx - bw / 2, SG.ys0, bw, SG.ys1 - SG.ys0};
    }
    int n = nif > 0 ? nif : 1;
    float slot = cw_ / n, bw = fminf(slot * 0.86f, 250 * u);
    for (int i = 0; i < nif; i++) {
        float cx = SG.cx0 + slot * (i + 0.5f);
        SG.ifb[i] = (Rect){cx - bw / 2, SG.yf0, bw, SG.yf1 - SG.yf0};
    }
    build_paths();
}

/* the scopes: which interfaces fit, up ones first */
static void pick_scopes(Rect r) {
    float cell_min = 108 * u;
    int maxn = (int)(r.h / cell_min);
    if (maxn < 1) maxn = 1;
    sc_n = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < nif && sc_n < maxn; i++)
            if ((pass == 0) == (ifs[i].up != 0)) sc_if[sc_n++] = i;
    /* keep them in the stack's order */
    for (int a = 0; a < sc_n; a++)
        for (int b = a + 1; b < sc_n; b++)
            if (sc_if[b] < sc_if[a]) { int t = sc_if[a]; sc_if[a] = sc_if[b]; sc_if[b] = t; }
}

static void layout(void) {
    nstyles = 0;
    u = dpr * clampf(fminf(W, H) / dpr / 1080.0f, 0.5f, 3.0f);   /* the clamp is in logical px */
    st_host = style(F_SEMI, 21 * u);
    st_sub = style(F_REG, 12.5f * u);
    st_clock = style(F_LIGHT, 50 * u);
    st_clock_s = style(F_LIGHT, 30 * u);
    st_away = style(F_MONO, 12.5f * u);
    st_lnum = style(F_LIGHT, 26 * u);
    st_lname = style(F_SEMI, 10.5f * u);
    st_tiny = style(F_REG, 10 * u);
    st_name = style(F_SEMI, 14 * u);
    st_iname = style(F_SEMI, 14 * u);
    st_mono = style(F_MONO, 12.5f * u);
    st_monos = style(F_MONO, 10.5f * u);
    st_proto = style(F_SEMI, 24 * u);
    st_big = style(F_MONO, 24 * u);
    st_title = style(F_SEMI, 11.5f * u);

    float M = 34 * u, hh = (role == R_FRONT ? 78 : 58) * u;
    r_stack = r_scopes = r_states = r_counters = (Rect){0, 0, 0, 0};
    r_head = (Rect){M, M * 0.9f, W - 2 * M, hh};
    float top = r_head.y + hh + 22 * u, avail = H - top - M;
    if (role == R_FRONT && span_me < 0) {
        if (W >= H * 1.15f) {
            float sw = (W - 2 * M) * 0.63f;
            r_stack = (Rect){M, top + 14 * u, sw, avail - 14 * u};
            float xr = M + sw + 40 * u, wr = W - M - xr;
            r_scopes = (Rect){xr, top, wr, avail * 0.5f};
            r_states = (Rect){xr, top + avail * 0.5f + 22 * u, wr, avail * 0.5f - 22 * u};
        } else {
            float sh = avail * 0.57f;
            r_stack = (Rect){M, top + 14 * u, W - 2 * M, sh - 14 * u};
            float yb = top + sh + 36 * u, hb = H - M - yb, wl = (W - 2 * M) * 0.54f;
            r_scopes = (Rect){M, yb, wl, hb};
            r_states = (Rect){M + wl + 36 * u, yb, W - 2 * M - wl - 36 * u, hb};
        }
    } else if (role == R_FRONT) {
        r_stack = (Rect){M, top + 14 * u, W - 2 * M, avail - 14 * u};
    } else if (role == R_SCOPES) {
        r_scopes = (Rect){M, top, W - 2 * M, avail};
    } else {
        if (W >= H) {
            float wl = (W - 2 * M) * 0.47f;
            r_states = (Rect){M, top, wl, avail};
            r_counters = (Rect){M + wl + 48 * u, top, W - 2 * M - wl - 48 * u, avail};
        } else {
            r_states = (Rect){M, top, W - 2 * M, avail * 0.56f};
            r_counters = (Rect){M, top + avail * 0.56f + 30 * u, W - 2 * M, avail * 0.44f - 30 * u};
        }
    }
    if (r_stack.w > 0) layout_stack(r_stack);
    if (r_scopes.w > 0) pick_scopes(r_scopes);
    build_atlas();
    np = 0;
}

/* ---- drawing: each panel draws its static parts (dyn 0, into the cached
 * texture) or its moving parts (dyn 1, every frame) ---- */
static double T_now;                   /* sim seconds */
static double WT_now;                  /* wall-clock */

static void icon(int kind, float x, float y, float s, const float *col, float a, float q, int dyn) {
    /* (x, y): the icon's centre; s: its size */
    float w = 1.3f * u;
    switch (kind) {
    case IK_WIFI: {
        /* a dot and three arcs, lit by link quality */
        float base_y = y + s * 0.38f;
        if (!dyn) {
            for (int i = 0; i < 3; i++)
                ring(x, base_y, s * (0.32f + 0.3f * i), w, -PI * 0.75f, -PI * 0.25f, col, a * 0.25f, 0, 0);
            disc(x, base_y, 1.6f * u, col, a * 0.5f, 0, 0);
        } else if (q >= 0) {
            int lit = q > 0.75f ? 3 : q > 0.45f ? 2 : q > 0.15f ? 1 : 0;
            for (int i = 0; i < lit; i++)
                ring(x, base_y, s * (0.32f + 0.3f * i), w, -PI * 0.75f, -PI * 0.25f, col, a, 3 * u, 0.25f);
            disc(x, base_y, 1.6f * u, col, a, 3 * u, 0.3f);
            /* a beacon ripple every few seconds */
            float ph = (float)fmod(T_now / 3.0, 1.0);
            ring(x, base_y, s * (0.25f + 1.0f * ph), w * 0.8f, -PI * 0.75f, -PI * 0.25f, col, a * 0.5f * (1 - ph), 0, 0);
        }
        break;
    }
    case IK_ETH: case IK_OTHER:
        if (!dyn) {
            box(x - s * 0.45f, y - s * 0.32f, s * 0.9f, s * 0.64f, 1.5f * u, w, col, a * 0.6f, 0, 0);
            box(x - s * 0.22f, y + s * 0.02f, s * 0.44f, s * 0.3f, 0.5f * u, w, col, a * 0.6f, 0, 0);
            for (int i = -2; i <= 2; i++) seg(x + i * s * 0.12f, y - s * 0.2f, x + i * s * 0.12f, y - s * 0.08f, w, col, a * 0.6f, 0, 0);
        }
        break;
    case IK_WWAN:
        if (!dyn) {
            seg(x, y - s * 0.1f, x - s * 0.25f, y + s * 0.42f, w, col, a * 0.6f, 0, 0);
            seg(x, y - s * 0.1f, x + s * 0.25f, y + s * 0.42f, w, col, a * 0.6f, 0, 0);
            ring(x, y - s * 0.12f, s * 0.22f, w, -PI * 0.8f, -PI * 0.2f, col, a * 0.5f, 0, 0);
            ring(x, y - s * 0.12f, s * 0.4f, w, -PI * 0.8f, -PI * 0.2f, col, a * 0.35f, 0, 0);
        }
        break;
    case IK_TUN:
        if (!dyn) {
            box(x - s * 0.48f, y - s * 0.28f, s * 0.96f, s * 0.56f, s * 0.28f, w, col, a * 0.6f, 0, 0);
            box(x - s * 0.3f, y - s * 0.12f, s * 0.6f, s * 0.24f, s * 0.12f, w, col, a * 0.6f, 0, 0);
        }
        break;
    case IK_BRIDGE:
        if (!dyn) {
            seg(x - s * 0.45f, y + s * 0.2f, x + s * 0.45f, y + s * 0.2f, w, col, a * 0.6f, 0, 0);
            ring(x, y + s * 0.2f, s * 0.38f, w, PI, 2 * PI, col, a * 0.6f, 0, 0);
            for (int i = -1; i <= 1; i++) seg(x + i * s * 0.22f, y + s * 0.2f, x + i * s * 0.22f, y + s * 0.2f - s * (i ? 0.3f : 0.38f), w, col, a * 0.4f, 0, 0);
        }
        break;
    case IK_VETH:
        if (!dyn) {
            ring(x - s * 0.17f, y, s * 0.26f, w, 0, 0, col, a * 0.6f, 0, 0);
            ring(x + s * 0.17f, y, s * 0.26f, w, 0, 0, col, a * 0.6f, 0, 0);
        }
        break;
    case IK_LOOP:
        if (!dyn) {
            ring(x, y, s * 0.34f, w, -PI * 0.35f, PI * 1.45f, col, a * 0.6f, 0, 0);
            float ax = x + s * 0.34f * cosf(-PI * 0.35f), ay = y + s * 0.34f * sinf(-PI * 0.35f);
            seg(ax, ay, ax - s * 0.16f, ay - s * 0.02f, w, col, a * 0.6f, 0, 0);
            seg(ax, ay, ax + s * 0.02f, ay + s * 0.16f, w, col, a * 0.6f, 0, 0);
        }
        break;
    }
}

static void draw_head(int dyn) {
    Rect r = r_head;
    char b[192];
    float y1 = r.y + r.h * 0.42f, y2 = r.y + r.h * 0.78f;
    if (!dyn) {
        char up[64];
        int i = 0;
        for (; host[i] && i < 63; i++) up[i] = (char)toupper((unsigned char)host[i]);
        up[i] = 0;
        if (role == R_FRONT) {
            text(st_host, r.x, y1, -1, up, c_ink, 1, 3.2f * u);
            snprintf(b, sizeof b, "network stack " MID " %d interface%s " MID " %s", nif, nif == 1 ? "" : "s",
                     traffic == T_LIVE ? "live from /proc" : traffic == T_BUSY ? "synthetic traffic (busy)" : "synthetic traffic (idle)");
            text(st_sub, r.x, y2, -1, b, c_dim, 1, 0.3f * u);
        } else {
            text(st_host, r.x, y1, -1, role == R_SCOPES ? "INTERFACES" : "PROTOCOLS", c_ink, 1, 3.2f * u);
            snprintf(b, sizeof b, "%s " MID " %s", host,
                     role == R_SCOPES ? "throughput, rx and tx, log scale, last two minutes"
                                      : "TCP connection states " MID " counters per second");
            text(st_sub, r.x, y2, -1, b, c_dim, 1, 0.3f * u);
        }
        seg(r.x, r.y + r.h + 6 * u, r.x + r.w, r.y + r.h + 6 * u, 1 * u, c_line, 0.35f, 0, 0);
        seg(r.x, r.y + r.h + 6 * u, r.x + 60 * u, r.y + r.h + 6 * u, 2 * u, c_proto[0], 0.9f, 0, 0);
        return;
    }
    time_t ts = (time_t)WT_now;
    struct tm tm;
    localtime_r(&ts, &tm);
    snprintf(b, sizeof b, "%02d:%02d", tm.tm_hour, tm.tm_min);
    long away = (long)(WT_now - (double)locked_at);
    if (away < 0) away = 0;
    char a[64];
    snprintf(a, sizeof a, "away %ld:%02ld:%02ld", away / 3600, away / 60 % 60, away % 60);
    if (role == R_FRONT) {
        text(st_clock, r.x + r.w, y1 + cap_h(st_clock) * 0.35f, 1, b, c_ink, 1, 0);
        text(st_away, r.x + r.w, y2 + 6 * u, 1, a, c_dim, 1, 0);
    } else {
        text(st_clock_s, r.x + r.w, y1 + cap_h(st_clock_s) * 0.2f, 1, b, c_ink, 0.85f, 0);
        text(st_away, r.x + r.w, y2 + 4 * u, 1, a, c_dim, 1, 0);
    }
}

static const char *proto_name[3] = {"TCP", "UDP", "ICMP"};
static const char *proto_desc[3] = {"transmission control", "user datagrams", "control messages"};

/* a labelled value in a little grid cell */
static void stat_cell(float x, float y, const char *label, const char *val, int dyn, const float *vc) {
    if (!dyn) text(st_tiny, x, y, -1, label, c_dim, 1, 0.8f * u);
    else text(st_mono, x, y + 16 * u, -1, val, vc, 1, 0);
}

static void draw_particles(void) {
    for (int i = 0; i < np; i++) {
        Part *q = &parts[i];
        const Path *pa = &paths[q->ifi][q->proto][(int)q->dir];
        const float *col = q->err ? (q->err == 2 ? c_red : c_warn) : c_proto[q->proto];
        float x, y, r0;
        if (q->kind == PK_FALL) {
            float f = 1 - q->age / q->life;
            dot(q->x, q->y, q->size, q->err == 2 ? c_red : c_warn, f, 6 * u, 0.6f);
            seg(q->x, q->y, q->x - q->vx * 0.05f, q->y - q->vy * 0.05f, q->size, q->err == 2 ? c_red : c_warn, f * 0.35f, 0, 0);
            continue;
        }
        if (q->kind == PK_BOUNCE) {
            float f = q->age / q->life;
            path_pos(pa, q->s, &x, &y, &r0);
            float sb = fmaxf(q->s - 26 * u, pa->s_ipt + 0.01f), tx, ty;
            path_pos(pa, sb, &tx, &ty, NULL);
            seg(tx, ty, x, y, 2.2f * u, c_red, (1 - f) * 0.6f, 5 * u, 0.4f);
            dot(x, y, 2.6f * u, c_red, 1 - f, 7 * u, 0.7f);
            ring(q->bx, q->by, (4 + 22 * f) * u, 1.4f * u, PI * 1.05f, PI * 1.95f, c_red, (1 - f) * 0.8f, 4 * u, 0.4f);
            continue;
        }
        path_pos(pa, q->s, &x, &y, &r0);
        /* fade in from the wire, out into the sockets */
        float a = 1;
        float fs = q->s / pa->len;
        if (q->dir == 0) a *= clampf(fs / 0.03f, 0, 1) * clampf((1 - fs) / 0.06f, 0, 1);
        else a *= clampf((1 - fs) / 0.03f, 0, 1) * clampf(fs / 0.05f, 0, 1);
        /* a trail behind it, within this visible run */
        float tl = q->v * 0.07f, ts_, tx, ty;
        if (q->dir == 0) ts_ = fmaxf(q->s - tl, r0 + 0.01f);
        else ts_ = fminf(q->s + tl, run_end(pa, q->s) - 0.01f);
        path_pos(pa, ts_, &tx, &ty, NULL);
        const float *cc = q->kind == PK_RETX ? c_red : col;
        if (fabsf(tx - x) + fabsf(ty - y) > 0.5f) seg(tx, ty, x, y, q->size * 1.1f, cc, a * 0.28f, 0, 0);
        dot(x, y, q->size, cc, a, (q->kind == PK_RETX ? 7 : 5) * u, q->kind == PK_RETX ? 0.7f : 0.45f);
    }
}

/* the rx/tx level meters in an interface box: log scale, 1 B/s .. 1 GB/s */
static void meter_geom(const Rect *bx, float *x0, float *x1, float *y0, float *y1) {
    float ip = 10 * u;
    float sy = bx->y + ip + cap_h(st_iname) + 32 * u, ly = bx->y + bx->h - ip - 18 * u - cap_h(st_mono);
    float mid = (sy + ly) / 2 + 2 * u;
    *x0 = bx->x + ip; *x1 = bx->x + bx->w - ip;
    *y0 = mid - 3.5f * u; *y1 = mid + 3.5f * u;
}

static void draw_stack(int dyn) {
    char b[64];
    float cw_ = SG.cx1 - SG.cx0;
    float pad = 12 * u;
    /* the layer labels in the gutter */
    static const char *ln[5] = {"", "L4", "L3", "L2", ""};
    static const char *lname[5] = {"SOCKETS", "TRANSPORT", "NETWORK", "LINK", ""};
    static const char *lsrc[5] = {"sockstat " MID " tcp states", "/proc/net/snmp", "IPv4 + IPv6", "/proc/net/dev", ""};
    float lys[4] = {(SG.ys0 + SG.ys1) / 2, (SG.yt0 + SG.yt1) / 2, (SG.yi0 + SG.yi1) / 2, (SG.yf0 + SG.yf1) / 2};
    if (!dyn) {
        for (int l = 0; l < 4; l++) {
            float y = lys[l];
            if (ln[l][0]) text(st_lnum, SG.gx, y - 4 * u, -1, ln[l], c_dim, 0.9f, 0);
            text(st_lname, SG.gx, y + (ln[l][0] ? 14 : 0) * u, -1, lname[l], c_ink, 0.9f, 1.6f * u);
            text(st_tiny, SG.gx, y + (ln[l][0] ? 29 : 15) * u, -1, lsrc[l], c_dim, 0.8f, 0);
        }
        /* conduits */
        for (int i = 0; i < nif; i++)
            for (int p = 0; p < 3; p++)
                for (int d = 0; d < 2; d++) {
                    const Path *pa = &paths[i][p][d];
                    float a = ifs[i].up ? 0.13f : 0.045f;
                    if (light) a *= 1.3f;
                    for (int j = 0; j + 1 < pa->n; j++) {
                        if (pa->gap[j]) continue;
                        if (j < 5 && ifs[i].kind == IK_WIFI)
                            dash(pa->x[j], pa->y[j], pa->x[j + 1], pa->y[j + 1], 1.0f * u, 5 * u, 0.5f, c_proto[p], a * 1.3f);
                        else
                            seg(pa->x[j], pa->y[j], pa->x[j + 1], pa->y[j + 1], 1.0f * u, c_proto[p], a, 0, 0);
                    }
                }
        /* the sockets shelf */
        box(SG.cx0, SG.ys0, cw_, SG.ys1 - SG.ys0, 8 * u, 0, c_scr, light ? 0.7f : 0.55f, 0, 0);
        box(SG.cx0, SG.ys0, cw_, SG.ys1 - SG.ys0, 8 * u, 1 * u, c_line, 0.45f, 0, 0);
        for (int p = 0; p < 3; p++) {
            Rect k = SG.skb[p];
            static const char *sl[3] = {"TCP SOCKETS", "UDP SOCKETS", "RAW SOCKETS"};
            text(st_tiny, k.x + pad * 0.3f, k.y + 18 * u, -1, sl[p], c_proto[p], 0.9f, 1.2f * u);
        }
        /* transport boxes */
        for (int p = 0; p < 3; p++) {
            Rect t = SG.trb[p];
            box(t.x, t.y, t.w, t.h, 9 * u, 0, c_scr, light ? 0.85f : 0.7f, 0, 0);
            box(t.x, t.y, t.w, t.h, 9 * u, 1.2f * u, c_proto[p], 0.55f, 0, 0);
            text(st_proto, t.x + pad, t.y + pad + cap_h(st_proto), -1, proto_name[p], c_proto[p], 1, 1 * u);
            text(st_tiny, t.x + pad, t.y + pad + cap_h(st_proto) + 15 * u, -1, proto_desc[p], c_dim, 1, 0.4f * u);
            float gy = t.y + t.h * 0.52f, cx2 = t.x + t.w * 0.52f;
            static const char *lab[3][4] = {{"SEGS IN /s", "SEGS OUT /s", "RETRANS /s", "OPENS /s"},
                                            {"DGRAMS IN /s", "DGRAMS OUT /s", "ERRORS /s", "SOCKETS"},
                                            {"MSGS IN /s", "MSGS OUT /s", "UNREACH /s", "RAW SOCKETS"}};
            float rowh = (t.y + t.h - pad - gy) / 2;
            for (int c = 0; c < 4; c++)
                stat_cell(c % 2 ? cx2 : t.x + pad, gy + (c / 2) * rowh, lab[p][c], NULL, 0, NULL);
        }
        /* the IP bus */
        box(SG.cx0, SG.yi0, cw_, SG.yi1 - SG.yi0, 6 * u, 0, c_scr, light ? 0.85f : 0.7f, 0, 0);
        box(SG.cx0, SG.yi0, cw_, SG.yi1 - SG.yi0, 6 * u, 1.2f * u, c_line, 0.8f, 0, 0);
        float ym = (SG.yi0 + SG.yi1) / 2;
        float w1 = text(st_proto, SG.cx0 + pad, ym + cap_h(st_proto) / 2, -1, "IP", c_ink, 1, 1 * u);
        text(st_tiny, SG.cx0 + pad * 1.6f + w1, ym + cap_h(st_tiny) / 2, -1, "v4 + v6", c_dim, 1, 0.5f * u);
        /* interface boxes */
        for (int i = 0; i < nif; i++) {
            Rect bx = SG.ifb[i];
            Iface *f = &ifs[i];
            float a = f->up ? 1 : 0.45f;
            box(bx.x, bx.y, bx.w, bx.h, 9 * u, 0, c_scr, light ? 0.85f : 0.7f, 0, 0);
            box(bx.x, bx.y, bx.w, bx.h, 9 * u, 1.1f * u, c_line, 0.7f * a, 0, 0);
            float ip = 10 * u;
            text(st_iname, bx.x + ip, bx.y + ip + cap_h(st_iname), -1, f->name, c_ink, a, 0);
            char kd[32];
            if (f->kind == IK_VETH && f->nagg > 1) snprintf(kd, sizeof kd, "VIRTUAL " TIMES "%d", f->nagg);
            else snprintf(kd, sizeof kd, "%s", ik_name[f->kind]);
            text(st_tiny, bx.x + ip, bx.y + ip + cap_h(st_iname) + 15 * u, -1, kd, c_dim, a, 1.1f * u);
            icon(f->kind, bx.x + bx.w - ip - 9 * u, bx.y + ip + 7 * u, 18 * u, c_ink, a, -1, 0);
            float ly = bx.y + bx.h - ip - 18 * u;
            text(st_mono, bx.x + ip, ly, -1, DN, c_rx, a, 0);
            text(st_mono, bx.x + ip, ly + 16 * u, -1, UP, c_tx, a, 0);
            float mx0, mx1, my0, my1;
            meter_geom(&bx, &mx0, &mx1, &my0, &my1);
            if (bx.h > 120 * u) {
                seg(mx0, my0, mx1, my0, 3 * u, c_grid, 0.22f * a, 0, 0);
                seg(mx0, my1, mx1, my1, 3 * u, c_grid, 0.22f * a, 0, 0);
                for (int d = 0; d <= 9; d += 3) {
                    float x = mx0 + (mx1 - mx0) * d / 9.0f;
                    seg(x, my0 - 4 * u, x, my0 - 6.5f * u, 0.9f * u, c_dim, 0.6f * a, 0, 0);
                }
            }
            /* the wire under it */
            float wx = bx.x + bx.w / 2;
            if (f->kind == IK_WIFI) {
                for (int k = 0; k < 2; k++)
                    ring(wx, SG.yw, (5 + 5 * k) * u, 1 * u, PI * 0.2f, PI * 0.8f, c_line, 0.5f, 0, 0);
            } else if (f->kind != IK_LOOP) {
                seg(wx - 7 * u, SG.yw, wx + 7 * u, SG.yw, 1.4f * u, c_line, 0.6f * a, 0, 0);
            } else {
                ring(wx, SG.yw, 4 * u, 1 * u, 0, 0, c_line, 0.5f, 0, 0);
            }
        }
        return;
    }

    /* ---- moving parts ---- */
    /* activity glow on the interface boxes */
    for (int i = 0; i < nif; i++) {
        Rect bx = SG.ifb[i];
        Iface *f = &ifs[i];
        float a = f->up ? 1 : 0.45f, ip = 10 * u;
        if (f->up) box(bx.x, bx.y, bx.w, bx.h, 9 * u, 1.3f * u, c_rx, 0.15f + 0.6f * f->act, 8 * u, 0.12f + 0.25f * f->act);
        /* state + speed/signal */
        float sy = bx.y + ip + cap_h(st_iname) + 32 * u;
        float pulse = 0.75f + 0.25f * sinf((float)T_now * 2.1f + i);
        const float *sc = f->up ? c_proto[0] : c_warn;
        disc(bx.x + ip + 3 * u, sy - 4 * u, 3 * u, sc, f->up ? pulse : 0.8f, 5 * u, 0.4f);
        char sv[48];
        if (!f->up) snprintf(sv, sizeof sv, "DOWN");
        else if (f->kind == IK_WIFI && f->sig_q >= 0) snprintf(sv, sizeof sv, "UP " MID " %.0f dBm", f->sig_dbm);
        else if (f->speed >= 1000) snprintf(sv, sizeof sv, "UP " MID " %g Gb/s", f->speed / 1000.0);
        else if (f->speed > 0) snprintf(sv, sizeof sv, "UP " MID " %d Mb/s", f->speed);
        else snprintf(sv, sizeof sv, "UP");
        text(st_monos, bx.x + ip + 11 * u, sy, -1, sv, f->up ? c_ink : c_warn, a * 0.9f, 0);
        if (f->kind == IK_WIFI) icon(IK_WIFI, bx.x + bx.w - ip - 9 * u, bx.y + ip + 7 * u, 18 * u, c_proto[0], a, f->sig_qd, 1);
        float ly = bx.y + bx.h - ip - 18 * u;
        float mx0, mx1, my0, my1;
        meter_geom(&bx, &mx0, &mx1, &my0, &my1);
        if (bx.h > 120 * u && f->up) {
            float fr = clampf(log10f(fmaxf(f->d[RXB], 1)) / 9, 0, 1), ft = clampf(log10f(fmaxf(f->d[TXB], 1)) / 9, 0, 1);
            if (fr > 0.003f) seg(mx0, my0, mx0 + (mx1 - mx0) * fr, my0, 3 * u, c_rx, 0.95f, 5 * u, 0.3f);
            if (ft > 0.003f) seg(mx0, my1, mx0 + (mx1 - mx0) * ft, my1, 3 * u, c_tx, 0.8f, 5 * u, 0.25f);
        }
        fmt_bytes(b, sizeof b, f->d[RXB]);
        text(st_mono, bx.x + ip + 13 * u, ly, -1, b, c_ink, a, 0);
        fmt_bytes(b, sizeof b, f->d[TXB]);
        text(st_mono, bx.x + ip + 13 * u, ly + 16 * u, -1, b, c_ink, a * 0.8f, 0);
    }
    /* the bus: a carrier shimmer running along it, brighter with traffic */
    {
        float ipa = clampf(log10f(1 + pd[PR_IP_IN] + pd[PR_IP_OUT]) / 5, 0, 1);
        box(SG.cx0, SG.yi0, cw_, SG.yi1 - SG.yi0, 6 * u, 1.3f * u, c_proto[0], 0.12f + 0.4f * ipa, 10 * u, 0.1f + 0.2f * ipa);
        float ph = (float)fmod(T_now / 7.0, 1.0);
        float sx = SG.cx0 + 6 * u + (cw_ - 12 * u) * ph;
        float sa = sinf(ph * PI);
        seg(fmaxf(sx - 60 * u, SG.cx0 + 6 * u), SG.yi0, sx, SG.yi0, 1.6f * u, c_proto[0], 0.5f * sa, 6 * u, 0.3f * sa);
        float sx2 = SG.cx1 - 6 * u - (cw_ - 12 * u) * ph;
        seg(sx2, SG.yi1, fminf(sx2 + 60 * u, SG.cx1 - 6 * u), SG.yi1, 1.6f * u, c_proto[0], 0.35f * sa, 6 * u, 0.2f * sa);
        float ym = (SG.yi0 + SG.yi1) / 2 + cap_h(st_mono) / 2;
        char l[160], v1[16], v2[16], v3[16], v4[16];
        fmt_rate(v1, sizeof v1, pd[PR_IP_IN]); fmt_rate(v2, sizeof v2, pd[PR_IP_OUT]);
        fmt_rate(v3, sizeof v3, pd[PR_IP_DISC]); fmt_rate(v4, sizeof v4, pd[PR_REASM]);
        float x = SG.cx0 + cw_ * 0.2f;
        snprintf(l, sizeof l, "in %s pkt/s", v1);
        float step = cw_ * 0.2f;
        text(st_mono, x, ym, -1, l, c_ink, 1, 0);
        snprintf(l, sizeof l, "out %s pkt/s", v2);
        text(st_mono, x + step, ym, -1, l, c_ink, 1, 0);
        snprintf(l, sizeof l, "discards %s/s", v3);
        text(st_mono, x + step * 2, ym, -1, l, pd[PR_IP_DISC] > 0.05f ? c_warn : c_dim, 1, 0);
        snprintf(l, sizeof l, "reassembly %s/s", v4);
        if (x + step * 3 + 120 * u < SG.cx1) text(st_mono, x + step * 3, ym, -1, l, c_dim, 1, 0);
    }
    /* transport values */
    for (int p = 0; p < 3; p++) {
        Rect t = SG.trb[p];
        float act = clampf(log10f(1 + (p == 0 ? pd[PR_TCP_IN] + pd[PR_TCP_OUT] : p == 1 ? pd[PR_UDP_IN] + pd[PR_UDP_OUT] : pd[PR_ICMP_IN] + pd[PR_ICMP_OUT])) / 4.5f, 0, 1);
        box(t.x, t.y, t.w, t.h, 9 * u, 1.3f * u, c_proto[p], 0.1f + 0.5f * act, 9 * u, 0.08f + 0.25f * act);
        float gy = t.y + t.h * 0.52f, cx2 = t.x + t.w * 0.52f, rowh = (t.y + t.h - pad - gy) / 2;
        float v[4];
        int warn2 = 0;
        if (p == 0) { v[0] = pd[PR_TCP_IN]; v[1] = pd[PR_TCP_OUT]; v[2] = pd[PR_RETX]; v[3] = pd[PR_AOPEN] + pd[PR_POPEN]; warn2 = pd[PR_RETX] > 0.05f; }
        else if (p == 1) { v[0] = pd[PR_UDP_IN]; v[1] = pd[PR_UDP_OUT]; v[2] = pd[PR_UDP_ERR]; v[3] = lvd[LV_UDP_INUSE]; warn2 = pd[PR_UDP_ERR] > 0.05f; }
        else { v[0] = pd[PR_ICMP_IN]; v[1] = pd[PR_ICMP_OUT]; v[2] = pd[PR_ICMP_UNR]; v[3] = lvd[LV_RAW_INUSE]; warn2 = pd[PR_ICMP_UNR] > 0.05f; }
        for (int c = 0; c < 4; c++) {
            if (c == 3 && p > 0) snprintf(b, sizeof b, "%.0f", v[c]);
            else fmt_rate(b, sizeof b, v[c]);
            stat_cell(c % 2 ? cx2 : t.x + pad, gy + (c / 2) * rowh, NULL, b, 1, c == 2 && warn2 ? (p == 0 ? c_red : c_warn) : c_ink);
        }
        /* the headline: total per second, top right */
        fmt_rate(b, sizeof b, v[0] + v[1]);
        float hw = text(st_big, t.x + t.w - pad, t.y + pad + cap_h(st_proto), 1, b, c_proto[p], 1, 0);
        (void)hw;
        text(st_tiny, t.x + t.w - pad, t.y + pad + cap_h(st_proto) + 15 * u, 1, p == 0 ? "segments/s" : p == 1 ? "datagrams/s" : "messages/s", c_dim, 1, 0.4f * u);
    }
    /* socket pips */
    for (int p = 0; p < 3; p++) {
        Rect k = SG.skb[p];
        float top = k.y + 28 * u;
        float gw = k.w - pad * 0.6f, gh = k.y + k.h - 8 * u - top;
        /* categories in order: (count, colour, style) */
        float cnt[12];
        const float *cc[12];
        int sty[12], nc = 0;
        if (p == 0) {
            static const int order[11] = {1, 2, 3, 12, 4, 5, 8, 9, 11, 7, 6};
            for (int j = 0; j < 11; j++) {
                int s = order[j];
                cnt[nc] = std_[s];
                cc[nc] = s == 1 || s == 2 || s == 3 || s == 12 ? c_proto[0] : s == 8 || s == 9 || s == 11 ? c_warn : c_dim;
                sty[nc++] = s == 6 ? 2 : s == 2 || s == 3 || s == 12 ? 3 : s == 1 ? 0 : 4;
            }
            cnt[nc] = std_[10]; cc[nc] = c_proto[0]; sty[nc++] = 1;   /* LISTEN: hollow */
            (void)0;
        } else {
            cnt[nc] = p == 1 ? lvd[LV_UDP_INUSE] : lvd[LV_RAW_INUSE];
            cc[nc] = c_proto[p]; sty[nc++] = 0;
        }
        int idx = 0;
        float total = 0;
        for (int c = 0; c < nc; c++) total += cnt[c];
        /* pips as big as fit: room for at least 48, then shrinking to 7 px */
        float room = fmaxf(48, ceilf(total / 16) * 16 * 1.1f);
        float sp = clampf(sqrtf(gw * gh / room), 7 * u, 19 * u);
        int cols = (int)(gw / sp), rows = (int)(gh / sp);
        if (cols < 1 || rows < 1) continue;
        int cap = cols * rows;
        for (int c = 0; c < nc && idx < cap; c++) {
            float n = cnt[c];
            int whole = (int)ceilf(n - 0.02f);
            for (int j = 0; j < whole && idx < cap; j++, idx++) {
                float fa = j + 1 <= n ? 1 : n - j;
                float px = k.x + pad * 0.3f + sp * (idx % cols) + sp / 2, py = top + sp * (idx / cols) + sp / 2;
                float tw = 0.85f + 0.15f * sinf((float)T_now * 0.9f + idx * 2.17f);
                switch (sty[c]) {
                case 0: dot(px, py, sp * 0.26f, cc[c], fa * tw, 3.5f * u, 0.3f); break;
                case 1: ring(px, py, sp * 0.26f, 1.2f * u, 0, 0, cc[c], fa * 0.9f, 0, 0); break;
                case 2: disc(px, py, sp * 0.17f, cc[c], fa * 0.55f, 0, 0); break;
                case 3: dot(px, py, sp * 0.26f, cc[c], fa * (0.5f + 0.5f * sinf((float)T_now * 8 + idx)), 4 * u, 0.4f); break;
                default: disc(px, py, sp * 0.24f, cc[c], fa * 0.85f, 3 * u, 0.2f); break;
                }
            }
        }
        if (total > cap + 0.5f) {
            snprintf(b, sizeof b, "+%.0f", total - cap);
            text(st_monos, k.x + k.w - pad * 0.3f, k.y + k.h - 4 * u, 1, b, c_dim, 1, 0);
        }
        /* the counts, top right of each cluster */
        if (p == 0) snprintf(b, sizeof b, "%.0f open " MID " %.0f listen " MID " %.0f tw", lvd[LV_ESTAB], std_[10], std_[6]);
        else snprintf(b, sizeof b, "%.0f", cnt[0]);
        text(st_monos, k.x + k.w - pad * 0.3f, k.y + 18 * u, 1, b, c_ink, 0.85f, 0);
    }
    if (!(role == R_FRONT && span_me < 0 && W < H * 1.15f)) {
        snprintf(b, sizeof b, "%.0f sockets in use", lvd[LV_SOCK_USED]);
        text(st_tiny, SG.gx, lys[0] + 42 * u, -1, b, c_dim, 0.8f, 0);
    }
    draw_particles();
}

/* a log-scale throughput axis: 1 B/s .. 1 GB/s */
static float scope_y(float v, float y0, float h) { return y0 + h - h * clampf(log10f(fmaxf(v, 1)) / 9, 0, 1); }

static void draw_scopes(int dyn) {
    Rect r = r_scopes;
    if (sc_n < 1) return;
    float gap = 16 * u, ch = (r.h - gap * (sc_n - 1)) / sc_n;
    char b[64], c2[64];
    for (int s = 0; s < sc_n; s++) {
        Iface *f = &ifs[sc_if[s]];
        float y = r.y + s * (ch + gap);
        float hh = 24 * u, ax = 34 * u, bl = 16 * u;
        Rect sc = {r.x + ax, y + hh + 4 * u, r.w - ax, ch - hh - 4 * u - bl};
        float a = f->up ? 1 : 0.5f;
        if (!dyn) {
            float w = text(st_name, r.x, y + cap_h(st_name) + 3 * u, -1, f->name, c_ink, a, 0);
            char kd[32];
            if (f->kind == IK_VETH && f->nagg > 1) snprintf(kd, sizeof kd, "VIRTUAL " TIMES "%d", f->nagg);
            else snprintf(kd, sizeof kd, "%s", ik_name[f->kind]);
            text(st_tiny, r.x + w + 10 * u, y + cap_h(st_name) + 3 * u, -1, kd, c_dim, a, 1.1f * u);
            box(sc.x, sc.y, sc.w, sc.h, 5 * u, 0, c_scr, light ? 0.9f : 0.85f, 0, 0);
            /* graticule: 10 s divisions, a line per decade */
            for (int i = 1; i < 12; i++) {
                float x = sc.x + sc.w * i / 12.0f;
                seg(x, sc.y + 2 * u, x, sc.y + sc.h - 2 * u, (i == 6 ? 1.0f : 0.8f) * u, c_grid, i == 6 ? 0.3f : 0.16f, 0, 0);
            }
            for (int d = 1; d < 9; d++) {
                float yy = sc.y + sc.h - sc.h * d / 9.0f;
                int major = d % 3 == 0;
                if (major) seg(sc.x + 2 * u, yy, sc.x + sc.w - 2 * u, yy, 0.9f * u, c_grid, 0.3f, 0, 0);
                else dash(sc.x + 2 * u, yy, sc.x + sc.w - 2 * u, yy, 0.8f * u, 4 * u, 0.4f, c_grid, 0.3f);
            }
            box(sc.x, sc.y, sc.w, sc.h, 5 * u, 1 * u, c_line, 0.6f, 0, 0);
            static const char *dl[4] = {"1", "1k", "1M", "1G"};
            for (int d = 0; d <= 3; d++) {
                float yy = sc.y + sc.h - sc.h * d / 3.0f;
                text(st_monos, sc.x - 6 * u, yy + cap_h(st_monos) * (d == 0 ? 0 : d == 3 ? 1 : 0.5f), 1, dl[d], c_dim, 0.9f, 0);
            }
            float by = sc.y + sc.h + bl - 3 * u;
            text(st_tiny, sc.x, by, -1, "-120 s", c_dim, 0.8f, 0);
            text(st_tiny, sc.x + sc.w / 2, by, 0, "-60 s", c_dim, 0.8f, 0);
            text(st_tiny, sc.x + sc.w, by, 1, "now", c_dim, 0.8f, 0);
            text(st_tiny, sc.x - 6 * u, by, 1, "B/s", c_dim, 0.8f, 0);
            continue;
        }
        /* header values */
        fmt_bytes(b, sizeof b, f->d[RXB]);
        fmt_bytes(c2, sizeof c2, f->d[TXB]);
        char l[160];
        snprintf(l, sizeof l, "%s", c2);
        float yh = y + cap_h(st_name) + 3 * u;
        float w2 = text(st_mono, r.x + r.w, yh, 1, l, c_tx, a, 0);
        text(st_mono, r.x + r.w - w2 - 4 * u, yh, 1, UP, c_tx, a, 0);
        float x3 = r.x + r.w - w2 - 22 * u;
        float w1 = text(st_mono, x3, yh, 1, b, c_rx, a, 0);
        text(st_mono, x3 - w1 - 4 * u, yh, 1, DN, c_rx, a, 0);
        if (!f->up) text(st_monos, x3 - w1 - 26 * u, yh, 1, "DOWN", c_warn, 1, 1 * u);
        else if (f->kind == IK_WIFI && f->sig_q >= 0) {
            snprintf(l, sizeof l, "%.0f dBm", f->sig_dbm);
            text(st_monos, x3 - w1 - 26 * u, yh, 1, l, c_dim, 1, 0);
        }
        /* traces: sample j is known at time j + 1; the leading edge is the smoothed value now */
        double tn = T_now;
        long kn = (long)floor(tn) - 1;
        float dx = sc.w / SCOPE_S;
        float right = sc.x + sc.w - 1.5f * u;
        for (int tr = 0; tr < 2; tr++) {
            const float *hc = tr ? f->htx : f->hrx;
            const float *col = tr ? c_tx : c_rx;
            float px = right, py = scope_y(f->d[tr ? TXB : RXB], sc.y + 2 * u, sc.h - 4 * u);
            float x0 = py, lead_y = py;
            (void)x0;
            for (long j = kn; j >= kn - SCOPE_S - 1 && j >= f->hfrom; j--) {
                int ji = (int)(((j % HIST) + HIST) % HIST);
                float x = right - (float)(tn - (double)(j + 1)) * dx;
                float yv = scope_y(hc[ji], sc.y + 2 * u, sc.h - 4 * u);
                float cx = x, cy = yv;
                if (x < sc.x + 1.5f * u) {
                    float t = (sc.x + 1.5f * u - x) / fmaxf(px - x, 1e-3f);
                    cx = sc.x + 1.5f * u; cy = yv + (py - yv) * t;
                }
                if (tr == 0) {
                    float fl = sc.y + sc.h - 2 * u, ft = fmaxf(cy, py);
                    if (fl > ft && px > cx) box(cx, ft, px - cx, fl - ft, 0, 0, col, light ? 0.07f : 0.06f, 0, 0);
                }
                seg(cx, cy, px, py, 1.5f * u, col, 0.95f * a, 4 * u, 0.35f);
                px = cx; py = cy;
                if (x < sc.x + 1.5f * u) break;
                /* drops and errors that second: a red tick on the floor */
                if (tr == 0 && f->hev[ji]) {
                    seg(x, sc.y + sc.h - 2 * u, x, sc.y + sc.h - 10 * u, 1.6f * u, c_red, 0.9f, 4 * u, 0.4f);
                }
            }
            dot(right, lead_y, 2.6f * u, col, a, 9 * u, 0.6f + 0.15f * sinf((float)T_now * 3 + tr));
        }
    }
}

/* a dial for one connection state: log scale 0 .. 1000 */
static void dial(float cx, float cy, float R, const char *label, float n, const float *col, int dyn) {
    float a0 = PI * 0.75f, sweep = PI * 1.5f;
    if (!dyn) {
        ring(cx, cy, R, 3 * u, a0, a0 + sweep, c_grid, light ? 0.3f : 0.28f, 0, 0);
        static const float tk[] = {0, 1, 10, 100, 1000};
        static const char *tl[] = {"0", "1", "10", "100", "1k"};
        for (int i = 0; i < 5; i++) {
            float f = log10f(1 + tk[i]) / log10f(1001), a = a0 + sweep * f;
            float c = cosf(a), s = sinf(a);
            seg(cx + c * (R + 5 * u), cy + s * (R + 5 * u), cx + c * (R + 10 * u), cy + s * (R + 10 * u), 1.2f * u, c_dim, 0.8f, 0, 0);
            text(st_monos, cx + c * (R + 20 * u), cy + s * (R + 20 * u) + cap_h(st_monos) / 2, 0, tl[i], c_dim, 0.8f, 0);
        }
        for (int i = 1; i < 30; i++) {
            float a = a0 + sweep * i / 30.0f;
            float c = cosf(a), s = sinf(a);
            seg(cx + c * (R + 5 * u), cy + s * (R + 5 * u), cx + c * (R + 7 * u), cy + s * (R + 7 * u), 0.8f * u, c_dim, 0.4f, 0, 0);
        }
        text(st_lname, cx, cy + R + 22 * u, 0, label, c_ink, 0.85f, 1.4f * u);
        return;
    }
    float f = clampf(log10f(1 + fmaxf(n, 0)) / log10f(1001), 0, 1);
    if (f > 0.002f) ring(cx, cy, R, 4 * u, a0, a0 + sweep * f, col, 1, 8 * u, 0.4f);
    float a = a0 + sweep * f;
    dot(cx + cosf(a) * R, cy + sinf(a) * R, 4 * u, col, 1, 9 * u, 0.6f);
    char b[16];
    snprintf(b, sizeof b, "%.0f", n);
    text(st_big, cx, cy + cap_h(st_big) * 0.45f, 0, b, c_ink, 1, 0);
}

static void draw_states(int dyn) {
    Rect r = r_states;
    char b[96];
    float y = r.y;
    if (!dyn) text(st_title, r.x, y + cap_h(st_title), -1, "TCP CONNECTION STATES", c_ink, 1, 1.8f * u);
    y += 26 * u;
    /* three dials */
    float dh = fminf(r.w / 3 * 0.9f, r.h * 0.38f);
    float R = dh * 0.33f;
    static const char *dn[3] = {"ESTABLISHED", "TIME_WAIT", "LISTEN"};
    static const int ds[3] = {1, 6, 10};
    for (int i = 0; i < 3; i++) {
        float cx = r.x + r.w * (i + 0.5f) / 3, cy = y + dh * 0.45f;
        dial(cx, cy, R, dn[i], std_[ds[i]], i == 1 ? c_dim : c_proto[0], dyn);
    }
    y += dh + 6 * u;
    /* the other states, bars in two columns */
    static const char *on[8] = {"SYN_SENT", "SYN_RECV", "FIN_WAIT1", "FIN_WAIT2", "CLOSE_WAIT", "LAST_ACK", "CLOSING", "CLOSE"};
    static const int os[8] = {2, 3, 4, 5, 8, 9, 11, 7};
    float rh = 19 * u, colw = (r.w - 30 * u) / 2;
    for (int i = 0; i < 8; i++) {
        float x = r.x + (i / 4) * (colw + 30 * u), yy = y + (i % 4) * rh + rh * 0.7f;
        float bx0 = x + 86 * u, bx1 = x + colw - 34 * u;
        float n = std_[os[i]] + (os[i] == 3 ? std_[12] : 0);
        if (!dyn) {
            text(st_tiny, x, yy, -1, on[i], c_dim, 1, 0.6f * u);
            seg(bx0, yy - 3.5f * u, bx1, yy - 3.5f * u, 3 * u, c_grid, 0.25f, 0, 0);
            continue;
        }
        float f = clampf(log10f(1 + n) / 2, 0, 1);
        const float *col = os[i] == 8 || os[i] == 9 || os[i] == 11 ? c_warn : c_proto[0];
        if (f > 0.003f) seg(bx0, yy - 3.5f * u, bx0 + (bx1 - bx0) * f, yy - 3.5f * u, 3 * u, col, 0.95f, 5 * u, 0.3f);
        snprintf(b, sizeof b, "%.0f", n);
        text(st_monos, x + colw, yy, 1, b, n >= 0.5f ? c_ink : c_dim, 1, 0);
    }
    y += rh * 4 + 6 * u;
    if (dyn) {
        char v1[16], v2[16];
        fmt_rate(v1, sizeof v1, pd[PR_AOPEN]); fmt_rate(v2, sizeof v2, pd[PR_POPEN]);
        snprintf(b, sizeof b, "opens/s: %s active " MID " %s passive " MID " orphans %.0f", v1, v2, lvd[LV_ORPHAN]);
        text(st_monos, r.x, y + 8 * u, -1, b, c_dim, 1, 0);
    }
    y += 26 * u;
    /* the congestion-window sketch */
    Rect g = {r.x, y + 30 * u, r.w, r.y + r.h - y - 30 * u};
    if (g.h < 50 * u) return;
    if (!dyn) {
        text(st_title, r.x, y + cap_h(st_title), -1, "CONGESTION WINDOW", c_ink, 1, 1.8f * u);
        float w = text_width(st_title, "CONGESTION WINDOW", 1.8f * u);
        text(st_tiny, r.x + w + 10 * u, y + cap_h(st_title), -1, NDASH " illustrative AIMD sketch, not measured", c_dim, 1, 0.3f * u);
        box(g.x, g.y, g.w, g.h, 5 * u, 0, c_scr, light ? 0.9f : 0.8f, 0, 0);
        for (int i = 1; i < 6; i++) {
            float x = g.x + g.w * i / 6.0f;
            dash(x, g.y + 2 * u, x, g.y + g.h - 2 * u, 0.8f * u, 4 * u, 0.4f, c_grid, 0.3f);
        }
        for (int i = 1; i < 4; i++) {
            float yy = g.y + g.h * i / 4.0f;
            dash(g.x + 2 * u, yy, g.x + g.w - 2 * u, yy, 0.8f * u, 4 * u, 0.4f, c_grid, 0.3f);
        }
        box(g.x, g.y, g.w, g.h, 5 * u, 1 * u, c_line, 0.6f, 0, 0);
        return;
    }
    float p = cw_p * 100;
    snprintf(b, sizeof b, "loss %.2f%% (RetransSegs / OutSegs) " MID " cwnd %.0f seg", p, cw);
    text(st_monos, g.x + 10 * u, g.y + 10 * u + cap_h(st_monos), -1, b, c_dim, 1, 0);
    /* 36 s at 10 Hz, scaled to the tallest tooth on show */
    float top = 20;
    for (int i = 0; i < CWN; i++) top = fmaxf(top, cwh[i]);
    top *= 1.3f;
    float px = 0, py = 0;
    long tick = cw_tick;
    for (int i = 0; i < CWN - 1; i++) {
        long tk = tick - i;
        int j = (int)(((tk % CWN) + CWN) % CWN);
        float x = g.x + g.w - 3 * u - (g.w - 6 * u) * i / (float)(CWN - 2);
        if (cwh[j] < 0) break;
        float yy = g.y + g.h - 3 * u - (g.h - 8 * u) * clampf(cwh[j] / top, 0, 1);
        if (i) {
            seg(x, yy, px, py, 1.6f * u, c_proto[0], 0.95f, 4 * u, 0.3f);
            if (cw_loss[(int)((((tk + 1) % CWN) + CWN) % CWN)] > 0) {
                seg(px, g.y + g.h - 3 * u, px, py, 1 * u, c_red, 0.35f, 0, 0);
                dot(x, yy, 2.4f * u, c_red, 0.9f, 6 * u, 0.5f);
            }
        }
        px = x; py = yy;
    }
}

/* the counters, each with a sparkline of the last minute */
static void draw_counters(int dyn) {
    Rect r = r_counters;
    static const struct { int grp; int pr; const char *label; } rows[] = {
        {0, PR_TCP_IN, "segments in"}, {0, PR_TCP_OUT, "segments out"}, {0, PR_RETX, "retransmits"},
        {0, PR_AOPEN, "active opens"}, {0, PR_POPEN, "passive opens"}, {0, PR_TIMEOUT, "timeouts"},
        {0, PR_LDROP, "listen drops"}, {1, PR_UDP_IN, "datagrams in"}, {1, PR_UDP_OUT, "datagrams out"},
        {1, PR_UDP_ERR, "receive errors"}, {2, PR_ICMP_IN, "messages in"}, {2, PR_ICMP_OUT, "messages out"},
        {2, PR_ICMP_UNR, "unreachables in"}, {3, PR_IP_IN, "packets in"}, {3, PR_IP_OUT, "packets out"},
        {3, PR_IP_DISC, "discards"}, {3, PR_REASM, "reassemblies"},
    };
    static const char *gn[4] = {"TCP", "UDP", "ICMP", "IP"};
    int nr = (int)(sizeof rows / sizeof rows[0]);
    float lines = nr + 4 * 1.6f;
    float rh = fminf(r.h / lines, 44 * u);
    float y = r.y + (r.h - rh * lines) / 2;
    int g = -1;
    char b[32];
    for (int i = 0; i < nr; i++) {
        if (rows[i].grp != g) {
            g = rows[i].grp;
            y += rh * 0.6f;
            if (!dyn) {
                const float *col = g < 3 ? c_proto[g] : c_ink;
                text(st_title, r.x, y + rh * 0.55f, -1, gn[g], col, 1, 2 * u);
                seg(r.x + 46 * u, y + rh * 0.55f - cap_h(st_title) / 2, r.x + r.w, y + rh * 0.55f - cap_h(st_title) / 2, 1 * u, col, 0.3f, 0, 0);
            }
            y += rh;
        }
        float yb = y + rh * 0.62f;
        float sx0 = r.x + r.w * 0.36f, sx1 = r.x + r.w * 0.78f;
        const float *col = g < 3 ? c_proto[g] : c_ink;
        if (!dyn) {
            text(st_sub, r.x + 12 * u, yb, -1, rows[i].label, c_dim, 1, 0);
            seg(sx0, yb, sx1, yb, 0.8f * u, c_grid, 0.3f, 0, 0);
            text(st_tiny, r.x + r.w, yb, 1, "/s", c_dim, 0.8f, 0);
        } else {
            int bad = rows[i].pr == PR_RETX || rows[i].pr == PR_LDROP || rows[i].pr == PR_UDP_ERR || rows[i].pr == PR_IP_DISC || rows[i].pr == PR_TIMEOUT;
            float v = pd[rows[i].pr];
            fmt_rate(b, sizeof b, v);
            text(st_mono, r.x + r.w - 16 * u, yb, 1, b, bad && v > 0.05f ? (rows[i].pr == PR_RETX ? c_red : c_warn) : c_ink, 1, 0);
            /* sparkline, 60 s, scaled to its own peak */
            long kn = last_k;
            float mx = 1;
            for (int s = 0; s < 60; s++) {
                long j = kn - s;
                if (j < pr_from) break;
                mx = fmaxf(mx, hpr[rows[i].pr][((j % HIST) + HIST) % HIST]);
            }
            float sh = rh * 0.6f, px = 0, py = 0;
            for (int s = 0; s < 60; s++) {
                long j = kn - s;
                if (j < pr_from) break;
                float vv = hpr[rows[i].pr][((j % HIST) + HIST) % HIST];
                float x = sx1 - (sx1 - sx0) * (s + (float)(T_now - floor(T_now))) / 60.0f;
                float yy = yb - sh * log1pf(vv) / log1pf(mx);
                if (x < sx0) break;
                if (s) seg(x, yy, px, py, 1.2f * u, bad && vv > 0.05f ? c_red : col, 0.85f, 3 * u, 0.2f);
                px = x; py = yy;
            }
        }
        y += rh;
    }
}

static void draw_all(int dyn) {
    if (r_head.w > 0) draw_head(dyn);
    if (r_stack.w > 0) draw_stack(dyn);
    if (r_scopes.w > 0) draw_scopes(dyn);
    if (r_states.w > 0) draw_states(dyn);
    if (r_counters.w > 0) draw_counters(dyn);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the ground: a gradient, a dot grid like engineering paper, a vignette */
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform vec3 uBg0, uBg1, uGrid;
    uniform float uU, uLight;
    out vec4 o;
    float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
    void main() {
        vec2 f = gl_FragCoord.xy, uv = f / uRes;
        vec3 c = mix(uBg1, uBg0, smoothstep(0.0, 1.0, uv.y));
        vec2 q = (uv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
        float v = dot(q, q);
        c *= mix(1.0 - 0.55 * v, 1.0 - 0.12 * v, uLight);
        float s = 24.0 * uU;
        vec2 g = (fract(f / s) - 0.5) * s;
        float dotm = 1.0 - smoothstep(0.55 * uU, 0.55 * uU + 1.0, length(g));
        vec2 G = abs((fract(f / (s * 5.0)) - 0.5) * s * 5.0);
        float cross = (1.0 - smoothstep(0.35 * uU, 0.35 * uU + 1.0, min(G.x, G.y))) * step(max(G.x, G.y), 4.0 * uU);
        c = mix(c, uGrid, (dotm * 0.22 + cross * 0.35) * mix(0.55, 0.7, uLight));
        c += (hash(f) - 0.5) / 255.0;
        o = vec4(c, 1.0);
    });

static const char *VS_PRIM = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 iA;
    layout(location = 1) in vec4 iB;
    layout(location = 2) in vec4 iC;
    layout(location = 3) in vec4 iE;
    uniform vec2 uRes;
    out vec2 vP;
    flat out vec4 vA;
    flat out vec4 vB;
    flat out vec4 vC;
    flat out vec4 vE;
    void main() {
        vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
        int k = int(iB.z + 0.5);
        float pad = iB.w * 3.0 + 1.5;
        vec2 p;
        if (k == 0) {
            vec2 d = iA.zw - iA.xy;
            float L = length(d);
            vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0), n = vec2(-t.y, t.x);
            float e = iB.x + pad;
            p = (iA.xy + iA.zw) * 0.5 + t * c.x * (L * 0.5 + e) + n * c.y * e;
        } else if (k == 1) {
            p = iA.xy + c * (iA.zw + pad);
        } else {
            p = iA.xy + c * (iB.x + iB.y * 0.5 + pad);
        }
        vP = p; vA = iA; vB = iB; vC = iC; vE = iE;
        gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
    });

static const char *FS_PRIM = "#version 330 core\n" GLSL(
    uniform float uLight;
    in vec2 vP;
    flat in vec4 vA;
    flat in vec4 vB;
    flat in vec4 vC;
    flat in vec4 vE;
    out vec4 o;
    void main() {
        int k = int(vB.z + 0.5);
        float d, core = 0.0;
        if (k == 0) {
            vec2 pa = vP - vA.xy, ba = vA.zw - vA.xy;
            float L2 = max(dot(ba, ba), 1e-6);
            float h = clamp(dot(pa, ba) / L2, 0.0, 1.0);
            d = length(pa - ba * h) - vB.x;
            if (vE.x > 0.0) {
                float s = dot(pa, ba) / sqrt(L2);
                float m = mod(s, vE.x) - vE.x * vE.y * 0.5;
                d = max(d, abs(m) - vE.x * vE.y * 0.5);
            }
        } else if (k == 1) {
            vec2 q = abs(vP - vA.xy) - (vA.zw - vB.x);
            d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - vB.x;
            if (vB.y > 0.0) d = abs(d + vB.y * 0.5) - vB.y * 0.5;
        } else {
            vec2 pc = vP - vA.xy;
            float l = length(pc);
            if (vB.y > 0.0) {
                d = abs(l - vB.x) - vB.y * 0.5;
                if (vE.y > vE.x) {
                    float a = atan(pc.y, pc.x);
                    float rel = mod(a - vE.x, 6.2831853);
                    if (rel > vE.y - vE.x) {
                        vec2 e0 = vB.x * vec2(cos(vE.x), sin(vE.x)), e1 = vB.x * vec2(cos(vE.y), sin(vE.y));
                        d = min(length(pc - e0), length(pc - e1)) - vB.y * 0.5;
                    }
                }
            } else d = l - vB.x;
            if (k == 3) core = clamp(1.0 - l / max(vB.x, 0.5), 0.0, 1.0);
        }
        float cov = clamp(0.5 - d, 0.0, 1.0);
        float g = 0.0;
        if (vB.w > 0.0) {
            float dd = max(d, 0.0);
            g = vE.w * exp(-dd / vB.w) * clamp(1.0 - dd / (3.0 * vB.w), 0.0, 1.0);
        }
        vec3 col = mix(vC.rgb, vec3(1.0), core * core * 0.65 * (1.0 - uLight));
        float a = vC.a * cov;
        float ga = vC.a * g;
        if (uLight > 0.5) {
            float t = clamp(a + ga * (1.0 - a), 0.0, 1.0);
            o = vec4(col * t, t);
        } else {
            o = vec4(col * (a + ga), a);
        }
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
    void main() {
        float a = vCol.a * texelFetch(uAtlas, ivec2(vUV), 0).r;
        o = vec4(vCol.rgb * a, a);
    });

static const char *FS_BLIT = "#version 330 core\n" GLSL(
    uniform sampler2D uTex;
    out vec4 o;
    void main() { o = vec4(texelFetch(uTex, ivec2(gl_FragCoord.xy), 0).rgb, 1.0); });

static GLuint prog_bg, prog_prim, prog_text, prog_blit, vao, vao_prim, vbo_prim, vao_text, vbo_text, tex_atlas,
    tex_static, fbo_static;
static int static_w, static_h, frame_prims, frame_glyphs;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "netflow: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *vs, const char *fs) {
    GLuint pr_ = glCreateProgram();
    glAttachShader(pr_, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(pr_, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr_);
    GLint ok;
    glGetProgramiv(pr_, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr_;
}

#define UL(p, n) glGetUniformLocation(p, n)

static void gl_init(void) {
    prog_bg = mk_prog(VS_FULL, FS_BG);
    prog_prim = mk_prog(VS_PRIM, FS_PRIM);
    prog_text = mk_prog(VS_TEXT, FS_TEXT);
    prog_blit = mk_prog(VS_FULL, FS_BLIT);
    glGenVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao_prim);
    glGenBuffers(1, &vbo_prim);
    glBindVertexArray(vao_prim);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
    for (int i = 0; i < 4; i++) {
        glVertexAttribPointer((GLuint)i, 4, GL_FLOAT, GL_FALSE, sizeof(Prim), (void *)(i * 4 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)i);
        glVertexAttribDivisor((GLuint)i, 1);
    }
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
}

/* the shapes then the text collected so far */
static void flush(void) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    if (pn) {
        glUseProgram(prog_prim);
        glUniform2f(UL(prog_prim, "uRes"), (float)W, (float)H);
        glUniform1f(UL(prog_prim, "uLight"), (float)light);
        glBindVertexArray(vao_prim);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_prim);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)pn * sizeof(Prim)), pv, GL_STREAM_DRAW);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pn);
    }
    if (tvn) {
        glUseProgram(prog_text);
        glUniform2f(UL(prog_text, "uRes"), (float)W, (float)H);
        glUniform1i(UL(prog_text, "uAtlas"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)tvn * sizeof(float)), tv, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
    }
    glDisable(GL_BLEND);
    pn = 0;
    tvn = 0;
}

static void build_static(void) {
    layout();
    layout_dirty = 0;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
    glUseProgram(prog_bg);
    glUniform2f(UL(prog_bg, "uRes"), (float)W, (float)H);
    glUniform3fv(UL(prog_bg, "uBg0"), 1, c_bg0);
    glUniform3fv(UL(prog_bg, "uBg1"), 1, c_bg1);
    glUniform3fv(UL(prog_bg, "uGrid"), 1, c_grid);
    glUniform1f(UL(prog_bg, "uU"), u);
    glUniform1f(UL(prog_bg, "uLight"), (float)light);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    pn = 0; tvn = 0;
    draw_all(0);
    flush();
}

static void render(GLuint target) {
    if (layout_dirty || static_w != W || static_h != H) build_static();
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(prog_blit);
    glUniform1i(UL(prog_blit, "uTex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_static);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    pn = 0; tvn = 0;
    draw_all(1);
    frame_prims = pn; frame_glyphs = tvn / 48;
    flush();
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
    dpr = (float)hidpi_factor(&hd, w);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "netflow");
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
        "usage: netflow [--window | --root | --layer] [--seed N] [--fps N] [--palette NAME]\n"
        "               [--traffic live|busy|idle] [--font FAMILY] [--mono FAMILY]\n"
        "               [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "               [--render-scale F (--layer: draw F x the output's native pixels, 0 < F <= 1; default 1)]\n"
        "               [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n"
        "palettes: oscilloscope, datacentre, paper, amber\n");
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

/* Spanning, the output under the middle of the layout (flightsim's front
 * rule) gets the stack; the rest, left to right, take the scopes and the
 * protocol counters in turn. */
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
    return rank % 2 ? R_COUNTERS : R_SCOPES;
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

/* bring a synthetic run up to time t: history straight from the model, then
 * the last stretch simulated (particles, smoothing, the window sketch) */
static void synth_warmup(double t) {
    long k1 = (long)floor(t) - 40;
    for (long k = k1 - HIST; k < k1; k++) { synth(k); push_hist(k); }
    pr_from = k1 - HIST;
    for (int i = 0; i < nif; i++) ifs[i].hfrom = k1 - HIST;
    synth(k1 - 1);
    snap_smooth();
    last_k = k1 - 1;
    sim_t = (double)k1;
    advance(t, 1);
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
        else if (!strcmp(a, "-mono") && i + 1 < argc) opt_mono = argv[++i];
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-traffic") && i + 1 < argc) opt_traffic = argv[++i];
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
    traffic = offscreen ? T_BUSY : T_LIVE;
    if (opt_traffic) {
        if (!strcmp(opt_traffic, "live")) traffic = T_LIVE;
        else if (!strcmp(opt_traffic, "busy")) traffic = T_BUSY;
        else if (!strcmp(opt_traffic, "idle")) traffic = T_IDLE;
        else usage();
    }
    pal_i = (int)(hash32(opt_seed * 7919u + 13) % NPAL);
    if (opt_palette) {
        int k;
        for (k = 0; k < NPAL; k++) if (!strcmp(pals[k].name, opt_palette)) break;
        if (k == NPAL) {
            fprintf(stderr, "netflow: no palette \"%s\" (", opt_palette);
            for (int j = 0; j < NPAL; j++) fprintf(stderr, "%s%s", pals[j].name, j + 1 < NPAL ? ", " : ")\n");
            exit(1);
        }
        pal_i = k;
    }
    set_palette();

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    fonts_init();

    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;

    if (traffic != T_LIVE) synth_ifaces();

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
        XStoreName(dpy, win, "Netflow");
        XClassHint chint = {"netflow", "Netflow"};
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
        double t = opt_at;
        if (traffic == T_LIVE) {
            /* a few real seconds of samples, then the particles run on them */
            int secs = (int)fmin(fmax(opt_at, 2), 6);
            for (int s = 0; s <= secs; s++) {
                sample(s);
                if (s == 0) { /* primes */ }
                if (s < secs) { struct timespec ts = {1, 0}; nanosleep(&ts, NULL); }
            }
            last_k = secs;
            snap_smooth();
            sim_t = secs;
            t = secs + 0.999;
            double save = sim_t;
            /* run the motion without sampling again */
            const float h = 1.0f / 60;
            for (double tt = save - 8; tt < save; tt += h) { smooth(h); step_particles(h); step_cwnd(tt, h); }
            T_now = t;
        } else {
            synth_warmup(t);
            T_now = t;
        }
        WT_now = t_wall0 + t;
        if (opt_shot) {
            if (layout_dirty) build_static();
            /* particles were spawned before the layout existed: rerun them on it */
            if (traffic != T_LIVE) {
                np = 0;
                const float h = 1.0f / 60;
                for (double tt = t - 6; tt < t; tt += h) step_particles(h);
            } else {
                np = 0;
                const float h = 1.0f / 60;
                for (int i = 0; i < 360; i++) step_particles(h);
            }
            render(fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        double t0 = now();
        build_static();
        glFinish();
        double tb = now() - t0, sum = 0, worst = 0, gsum = 0;
        np = 0;
        for (int i = 0; i < 240; i++) advance(t + i / 60.0, 0);
        t += 4;
        GLuint q;
        glGenQueries(1, &q);
        clock_t c0 = clock();
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            T_now = t + i / 60.0;
            WT_now = t_wall0 + T_now;
            advance(T_now, 0);
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            gsum += ns / 1e6;
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        double cpu = (double)(clock() - c0) / CLOCKS_PER_SEC * 1000 / opt_bench;
        printf("netflow: %dx%d, %d frames: avg %.3f ms wall, worst %.3f ms; GPU %.3f ms/frame; CPU %.3f ms/frame; "
               "static layer built in %.1f ms; %d particles, %d shapes, %d glyphs\n",
               W, H, opt_bench, sum / opt_bench, worst, gsum / opt_bench, cpu, tb * 1000, np, frame_prims, frame_glyphs);
        printf("netflow: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    sim_t = 0;
    if (traffic != T_LIVE) {
        double el0 = opt_epoch ? fmax(wall() - opt_epoch, 0) : 0;
        synth_warmup(el0 + 0.001);
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
        double t = now(), wt = wall(), el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        if (layout_dirty && nif) build_static();
        advance(el, 0);
        T_now = el;
        WT_now = wt;
        render(0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
