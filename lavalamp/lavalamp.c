/*
 * lavalamp — a lava lamp glowing on a dark sideboard in a dim room.
 *
 * A classic rocket lamp, brass, chrome or black enamel cone and cap round a
 * tapered glass bottle, stands on a walnut sideboard against a plaster
 * wall. Its bulb lights the wax from below; the lamp is the room's light,
 * throwing a coloured glow up the wall and across the varnish, with a faint
 * cool window somewhere behind you for the glints.
 *
 * The wax is simulated (see "the wax" below): a pool heated by the bulb,
 * plumes that swell into buds on thick necks, necks that stretch, drain and
 * pinch off, blobs that rise at their Stokes speed, cool by their surface,
 * flatten under the cap, sink back as rounder drops and are swallowed by the
 * pool; touching blobs merge; in a hot spell a long column climbs the lamp.
 * Switched on (the --epoch, i.e. the lock), the wax is a cold lump: it
 * softens over a minute or so in a cloudy liquid that clears as it warms,
 * then the first tower rises.
 *
 * Rendering: each frame the blobs, necks and pool become one smoothly
 * blended signed distance field in a 3D texture, and a second 3D texture
 * holds how much wax lies between every point and the bulb. The scene shader
 * finds the glass (a surface of revolution), refracts into the liquid,
 * marches through it gathering the bulb's light scattered by the liquid (and
 * shadowed by the wax), and shades the wax it meets: lit from below,
 * scattering light inside so the underside glows and the crown falls dark,
 * with a soft sheen. Rays that miss the wax refract out the back of the
 * bottle to the wall. Glass and metal take Fresnel reflections of the room;
 * the metal also sees the glowing glass. Then bloom, a filmic curve, a warm
 * grade, vignette and grain. The room is drawn into a cache about once a
 * second (it only changes as the lamps warm up); each frame redraws just the
 * lamp's box. About 2.5 ms of GPU a frame at 1920x1080 on an Iris Xe.
 *
 * Palettes (--palette, else from the seed): classic (red wax in yellow),
 * blue (in clear), purple (in pink), green (in blue), white (in cyan) and
 * glitter (pink wax in violet, gold flakes carried round by the convection).
 *
 * A small caption in the corner: the time, and how long the screen has been
 * locked (swaylock's start time, as gauges reads it).
 *
 * Spanning: run one instance per output with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * The sideboard and wall run on across all of them, each output with its own
 * lamp (a taller one on a portrait screen) seen through its own camera in a
 * parallel rig, so the shelf edge and the light on the wall are continuous
 * from screen to screen; each lamp lights its neighbours' walls too. The
 * output under the middle of the layout gets the --palette (or the seed's)
 * and the caption; the others, left to right, complementary colours. Each
 * lamp's wax is its own (the seed and its output), so instances need not
 * talk; the light they throw on each other depends only on the epoch.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen
 * (--at counts from switch-on; 0-120 s is the warm-up). --bench N times N
 * frames offscreen. --cpu lets the CPU load turn the heater up (livelier
 * wax under load; the wax then no longer repeats exactly). --stats prints
 * the wax's state.
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

#define PI 3.14159265f

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_cpu, opt_stats, opt_caption = 1;
static float opt_fps = 60, opt_scale = 0;  /* 0: the default render scale */
static const char *opt_font = "EB Garamond";
static const char *opt_palette;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 240;            /* --shot: the frame this many seconds after switch-on */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the lamps were switched on */

/* ---- spanning: one instance per output ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "lavalamp: %s\n", what); exit(1); }

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

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}


static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float a, float b, float x) { x = clampf((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x); }

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

/* --cpu: the CPU load (0..1) from /proc/stat deltas, read once a second */
static float cpu_load;
static void read_cpu(void) {
    static unsigned long long pb, pt;
    char line[512];
    slurp_line("/proc/stat", line, sizeof line);
    unsigned long long v[8] = {0};
    if (sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) < 4)
        return;
    unsigned long long tot = 0;
    for (int i = 0; i < 8; i++) tot += v[i];
    unsigned long long busy = tot - v[3] - v[4];
    if (pt && tot > pt) cpu_load = clampf((float)(busy - pb) / (float)(tot - pt), 0, 1);
    pb = busy; pt = tot;
}

/* ---- palettes: wax, liquid, and the metal of the base ---- */
enum { M_BRASS, M_CHROME, M_BLACK };
typedef struct {
    const char *name;
    float wax[3];      /* the wax's colour (linear) */
    float ext[3];      /* how fast light dies in the wax, per channel, per unit */
    float abs[3];      /* the liquid's absorption per unit */
    float sc[3];       /* the liquid's scattering (its glow) */
    int metal;
    int glitter;
    float glow[3];     /* the light the lamp throws on the room */
} Palette;
static const Palette pals[] = {
    {"classic", {0.85f, 0.07f, 0.02f}, {1.0f, 4.0f, 9.0f}, {0.03f, 0.12f, 1.1f}, {1.00f, 0.80f, 0.22f}, M_BRASS, 0,
     {1.00f, 0.52f, 0.16f}},
    {"blue", {0.05f, 0.22f, 0.95f}, {9.0f, 3.6f, 0.9f}, {0.05f, 0.04f, 0.035f}, {0.30f, 0.33f, 0.38f}, M_CHROME, 0,
     {0.42f, 0.52f, 1.00f}},
    {"purple", {0.50f, 0.06f, 0.75f}, {3.0f, 10.0f, 1.8f}, {0.06f, 0.55f, 0.22f}, {0.95f, 0.42f, 0.62f}, M_BLACK, 0,
     {0.95f, 0.38f, 0.72f}},
    {"green", {0.22f, 0.90f, 0.06f}, {5.0f, 1.0f, 10.0f}, {1.10f, 0.32f, 0.06f}, {0.22f, 0.52f, 0.95f}, M_CHROME, 0,
     {0.32f, 0.72f, 0.95f}},
    {"white", {0.92f, 0.90f, 0.86f}, {1.9f, 2.0f, 2.2f}, {1.30f, 0.12f, 0.08f}, {0.25f, 0.85f, 0.90f}, M_CHROME, 0,
     {0.45f, 0.88f, 0.95f}},
    {"glitter", {0.95f, 0.16f, 0.40f}, {1.4f, 8.0f, 3.5f}, {0.55f, 0.75f, 0.06f}, {0.42f, 0.36f, 0.95f}, M_BLACK, 1,
     {0.62f, 0.45f, 1.00f}},
};
#define NPAL ((int)(sizeof pals / sizeof pals[0]))
/* the lamps beside a palette, for the other outputs: complementary colours */
static const int comp[NPAL][2] = {{1, 3}, {0, 2}, {3, 4}, {2, 0}, {2, 0}, {4, 0}};

/* ---- the lamp (lamp units: the glass's bottom at y = 0; the foot of the
 * base at LAMP_FOOT; the cap's top at LAMP_TOP) ---- */
#define LAMP_FOOT (-1.75f)
#define LAMP_TOP 3.91f
#define LAMP_H (LAMP_TOP - LAMP_FOOT)
#define HG 3.22f                       /* the liquid's top, under the cap */
#define GLASS_T 0.022f
static float rg(float y) { return 0.635f - 0.095f * y - 0.16f * expf(-4.5f * y); }   /* the glass's outer radius */

/* the field box: where the wax can be */
static const float box_min[3] = {-0.58f, -0.02f, -0.58f}, box_size[3] = {1.16f, 3.28f, 1.16f};
#define FX 40
#define FY 112
#define FZ 40



/* ---- the wax ----
 * Modelled the way a lava lamp actually works, at the level of its blobs:
 *
 *   the pool on the bottom is heated by the bulb through the glass and the
 *   coil, and loses heat to the liquid; while it's colder than the liquid
 *   can float it lies there, a heavy puddle
 *   once it's hot enough, a plume over the bulb feeds a bud: hot wax wells
 *   up into a swelling dome that lifts on a thick neck
 *   when the plume stops feeding it, the neck stretches as the bud rises and
 *   drains into the bud and back into the pool, thinner and thinner, until
 *   it pinches off; the stubs snap back into the blob and the pool
 *   a free blob rises at its Stokes speed (bigger blobs faster), pulled
 *   into a teardrop by the drag; it cools at a rate set by its surface for
 *   its volume (small ones cool sooner), fastest near the top, where it
 *   flattens against the glass, drifts, and gets squeezed by the taper
 *   cold, it sinks back as a rounder drop, and is swallowed by the pool
 *   (a ripple, and its cold wax cools the pool a little)
 *   blobs that touch are drawn together and merge
 *   in a hot spell the plume keeps feeding while the bud rises: a long
 *   column that stretches up the lamp before it lets go
 *
 * Switched on cold, the wax is a solid lump: it takes a minute or two to
 * melt (it is drawn paler and more opaque while solid), and the first plume
 * pushes up a tall tower.
 *
 * Everything is a fixed-step function of the seed and the time since
 * switch-on, so it's the same every run. Rendering turns the blobs, necks
 * and pool into one smoothly blended signed distance field. */
#define MAXB 14
#define DT (1.0f / 60)
#define T_ROOM 0.06f
#define T_NEUTRAL 0.47f                /* wax at this temperature neither rises nor sinks */
#define T_MELT 0.30f
#define WAX_V 0.36f                   /* all the wax there is (~11% of the lamp) */
#define Y_TOP (HG - 0.03f)

typedef struct {
    float x[3], v[3];
    float V, T;                        /* volume, temperature */
    float sy;                          /* vertical stretch (>1 a teardrop, <1 flattened) */
    float tail;                        /* the pinched neck's stub hanging under it, retracting */
    float ph;                          /* its own wobble */
    float absorb;                      /* 0..1 while the pool swallows it */
    int alive;
} Blob;

typedef struct {
    int on, feeding;
    float base[2];                     /* where the neck meets the pool */
    float x[3], vy, V, T, Vn;          /* the bud's centre, rise speed, volume, temperature; the neck's volume */
    double feed_end;
    float flux;                        /* volume per second the plume pours in */
} Bud;

typedef struct {
    Blob b[MAXB];
    Bud bud;
    float Vpool, Tpool;
    float stub, stub_r, stub_x, stub_z; /* the neck's stub on the pool after a pinch */
    float rip[4][4];                   /* ripples on the pool: x, z, age, strength */
    long steps;
    double t0;
    double ev_next;
    float ph[6];
    uint32_t rng;
} Sim;
static Sim sim, sim_prev;              /* sim_prev: a step ago, to interpolate */

static float srand1(void) {
    uint32_t r = sim.rng;
    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
    sim.rng = r;
    return (r >> 8) * (1.0f / 16777216.0f);
}

static float ri(float y) { return rg(y) - GLASS_T; }

/* the pool's depth for a volume: the glass's bottom is curved, so tabulated */
#define NPV 256
static float pv_tab[NPV + 1];          /* volume below y = i * (1.5 / NPV) */
static void pool_table(void) {
    pv_tab[0] = 0;
    for (int i = 0; i < NPV; i++) {
        float y = (i + 0.5f) * 1.5f / NPV, r = ri(y);
        pv_tab[i + 1] = pv_tab[i] + PI * r * r * 1.5f / NPV;
    }
}
static float pool_h(float V) {
    for (int i = 0; i < NPV; i++)
        if (pv_tab[i + 1] >= V) return (i + (V - pv_tab[i]) / (pv_tab[i + 1] - pv_tab[i])) * 1.5f / NPV;
    return 1.5f;
}

static float radius(float V) { return cbrtf(V * (3 / (4 * PI))); }

/* the liquid warms up over the first minutes after switch-on */
static float warmth(double t) { return 1 - expf(-(float)t / 100.0f); }

/* the liquid's temperature: warm near the bulb, cool at the top; always a
 * little colder than the wax needs to float, so only heated wax rises */
static float t_liquid(float y, double t) {
    float eq = 0.44f - 0.24f * y / HG;
    return T_ROOM + (eq - T_ROOM) * warmth(t);
}

/* the heater's power: a slow wander, and now and then a hot spell (when the
 * lamp throws up long columns) */
static float spell(double t) { return powf(fmaxf(0, sinf((float)t / 151.0f + sim.ph[3])), 10.0f); }
static float heater(double t) {
    float s = (float)t;
    float p = 1 + 0.10f * sinf(s / 47.0f + sim.ph[0]) + 0.08f * sinf(s / 113.0f + sim.ph[1]) + 0.05f * sinf(s / 23.0f + sim.ph[2]);
    p += 0.35f * spell(t);
    if (opt_cpu) p *= 0.9f + 0.3f * cpu_load;
    return p;
}

static void sim_init(uint32_t seed, double t0) {
    memset(&sim, 0, sizeof sim);
    pool_table();
    sim.rng = hash32(seed ^ 0x1a7a1a7au) | 1;
    for (int i = 0; i < 6; i++) sim.ph[i] = srand1() * 2 * PI;
    sim.t0 = t0;
    sim.Vpool = WAX_V;
    sim.Tpool = t0 > 1 ? 0.6f : T_ROOM;  /* starting late: the lamp's been on a while */
    sim.ev_next = t0 + 2;
    sim_prev = sim;
}

static Blob *new_blob(void) {
    for (int i = 0; i < MAXB; i++)
        if (!sim.b[i].alive) { memset(&sim.b[i], 0, sizeof sim.b[i]); sim.b[i].alive = 1; return &sim.b[i]; }
    return NULL;
}

static void ripple(float x, float z, float s) {
    int k = 0;
    for (int i = 1; i < 4; i++) if (sim.rip[i][2] > sim.rip[k][2]) k = i;
    sim.rip[k][0] = x; sim.rip[k][1] = z; sim.rip[k][2] = 0; sim.rip[k][3] = s;
}

/* the speed a blob of radius r at temperature T settles to: Stokes, buoyancy
 * against drag, so it goes as r squared */
static float stokes(float r, float T) {
    float v = 0.95f * (r / 0.16f) * (r / 0.16f) * (T - T_NEUTRAL);
    return clampf(v, -0.16f, 0.2f);
}

static void sim_step(void) {
    sim_prev = sim;
    double t = sim.t0 + sim.steps * DT;
    float warm = warmth(t), hp = heater(t);
    float h = pool_h(sim.Vpool);

    /* the pool: heated through the bottom, losing heat to the liquid */
    sim.Tpool += DT * (0.014f * hp * (1.0f - sim.Tpool) - 0.016f * (sim.Tpool - t_liquid(h * 0.5f, t)));
    for (int k = 0; k < 4; k++) sim.rip[k][2] += DT;
    if (sim.stub > 0) sim.stub = fmaxf(0, sim.stub - DT * (0.12f + 1.6f * sim.stub));

    /* a plume starts a bud, once the pool is molten and hot enough to float */
    Bud *bd = &sim.bud;
    if (!bd->on && t >= sim.ev_next && sim.Tpool > T_NEUTRAL + 0.02f && sim.Vpool > WAX_V * 0.45f) {
        float a = srand1() * 2 * PI, r = 0.13f * sqrtf(srand1());
        float sp = spell(t);
        int tower = sim.steps * DT + sim.t0 < 240 && sim.ev_next < sim.t0 + 3;  /* the first, warming up */
        memset(bd, 0, sizeof *bd);
        bd->on = bd->feeding = 1;
        bd->base[0] = r * cosf(a); bd->base[1] = r * sinf(a);
        bd->V = 0.0015f;
        bd->T = sim.Tpool + 0.1f;
        bd->x[0] = bd->base[0]; bd->x[2] = bd->base[1];
        bd->x[1] = h - 0.02f;
        float dur = (3.5f + 4.5f * srand1()) * (1 + 2.2f * sp) * (tower ? 2.6f : 1);
        float vol = fminf(0.34f * sim.Vpool, 0.014f + 0.032f * srand1())
 * (1 + 1.4f * sp) * (tower ? 2.0f : 1);
        bd->flux = vol / dur;
        bd->feed_end = t + dur;
    }
    if (bd->on) {
        if (bd->feeding && (t >= bd->feed_end || sim.Vpool < WAX_V * 0.3f)) {
            bd->feeding = 0;
            sim.ev_next = t + (4.0f + 14.0f * srand1()) / hp;
        }
        float rb = radius(bd->V);
        if (bd->feeding) {
            float dv = bd->flux * DT;
            bd->V += dv;
            sim.Vpool -= dv;
            bd->T += (sim.Tpool + 0.12f - bd->T) * DT * 0.5f;
        }
        /* buoyancy lifts the bud; the neck holds it back while it's thick */
        float L = fmaxf(0, bd->x[1] - rb * 0.75f - h);
        float rn = L > 0.005f ? sqrtf(bd->Vn / (PI * L)) : rb;
        float hold = clampf(rn / (rb + 1e-3f), 0, 1);
        float vt = stokes(rb, bd->T) * (1 - 0.55f * hold);
        if (bd->x[1] < h + rb * 0.4f) vt = fmaxf(vt, 0.03f);   /* still welling up out of the pool */
        bd->vy += (vt - bd->vy) * DT * 1.5f;
        bd->x[1] += bd->vy * DT;
        /* lean with the liquid's circulation a little */
        bd->x[0] += (bd->base[0] * 0.6f - bd->x[0]) * DT * 0.1f;
        bd->x[2] += (bd->base[1] * 0.6f - bd->x[2]) * DT * 0.1f;
        if (bd->x[1] + rb > Y_TOP) bd->x[1] = Y_TOP - rb;
        bd->T += DT * (-0.004f / fmaxf(rb, 0.04f) * (bd->T - t_liquid(bd->x[1], t)));
        /* the neck: fed thick while the plume runs; then it stretches and
         * drains, into the bud above and the pool below */
        L = fmaxf(0, bd->x[1] - rb * 0.75f - h);
        if (bd->feeding) {
            float want = fminf(PI * (0.62f * rb) * (0.62f * rb) * L, 0.3f * sim.Vpool);
            float dn = (want - bd->Vn) * DT * 2.0f;
            bd->Vn += dn;
            sim.Vpool -= dn;
        } else {
            float d = bd->Vn * DT * 0.45f;
            bd->Vn -= d;
            bd->V += d * 0.6f;
            sim.Vpool += d * 0.4f;
        }
        rn = L > 0.005f ? sqrtf(bd->Vn / (PI * L)) : rb;
        if (!bd->feeding && L > 0.05f && (rn < 0.022f || rn < 0.1f * rb)) {
            /* pinch-off */
            Blob *b = new_blob();
            if (b) {
                memcpy(b->x, bd->x, sizeof b->x);
                b->v[1] = bd->vy;
                b->V = bd->V + bd->Vn * 0.5f;
                b->T = bd->T;
                b->sy = 1.25f;
                b->tail = fminf(L * 0.45f, 0.35f);
                b->ph = srand1() * 50;
            } else sim.Vpool += bd->V;
            sim.Vpool += bd->Vn * 0.5f;
            sim.stub = fminf(L * 0.4f, 0.3f);
            sim.stub_r = fmaxf(rn, 0.03f);
            sim.stub_x = bd->base[0]; sim.stub_z = bd->base[1];
            ripple(bd->base[0], bd->base[1], 0.6f);
            bd->on = 0;
        } else if (!bd->feeding && bd->vy < 0 && bd->x[1] < h + rb * 0.5f) {
            /* it never got away: back into the pool */
            sim.Vpool += bd->V + bd->Vn;
            sim.Tpool += (bd->T - sim.Tpool) * bd->V / sim.Vpool;
            bd->on = 0;
        }
    }

    /* free blobs */
    for (int i = 0; i < MAXB; i++) {
        Blob *b = &sim.b[i];
        if (!b->alive) continue;
        float r = radius(b->V);
        float topc = smooth(HG * 0.62f, HG * 0.97f, b->x[1] + r);
        b->T += DT * (-(0.0045f / fmaxf(r, 0.04f)) * (1 + 5.0f * topc) * (b->T - t_liquid(b->x[1], t)));
        float vt = stokes(r, b->T);
        /* the liquid's slow circulation: up the middle, down the sides */
        float R = ri(b->x[1]), rr = sqrtf(b->x[0] * b->x[0] + b->x[2] * b->x[2]), xi = rr / R;
        float U = 0.025f * warm, fy = U * (1 - 2 * xi * xi) * sinf(PI * b->x[1] / HG);
        float wx = 0.012f * sinf((float)t * 0.11f + b->ph), wz = 0.012f * cosf((float)t * 0.083f + b->ph * 1.3f);
        float fr = -U * 0.6f * xi * cosf(PI * b->x[1] / HG);
        float tx = (rr > 1e-3f ? fr * b->x[0] / rr : 0) + wx, tz = (rr > 1e-3f ? fr * b->x[2] / rr : 0) + wz;
        if (b->absorb > 0) { vt = -0.05f; tx = tz = 0; }
        b->v[0] += (tx - b->v[0]) * DT * 0.8f;
        b->v[2] += (tz - b->v[2]) * DT * 0.8f;
        b->v[1] += (vt + fy - b->v[1]) * DT * 1.2f;
        for (int k = 0; k < 3; k++) b->x[k] += b->v[k] * DT;

        /* shape: drawn out by moving, flattened under the top, squeezed by the taper */
        float rx = r / sqrtf(b->sy), ry = r * b->sy;
        int top = b->x[1] + ry >= Y_TOP - 0.005f;
        float want = 1 + 1.5f * fabsf(b->v[1]);
        if (b->v[1] < 0) want = 1 - 0.8f * fabsf(b->v[1]) + 0.05f;   /* falling drops round, a little flat below */
        if (top) want = 0.62f;
        float rxmax = ri(b->x[1]) * 0.93f;
        if (r > rxmax) want = fmaxf(want, (r / rxmax) * (r / rxmax));
        b->sy += (want - b->sy) * DT * 0.9f;
        rx = r / sqrtf(b->sy); ry = r * b->sy;
        if (b->x[1] + ry > Y_TOP) { b->x[1] = Y_TOP - ry; if (b->v[1] > 0) b->v[1] = 0; }
        float lim = fmaxf(0, ri(b->x[1]) - rx * 0.97f);
        rr = sqrtf(b->x[0] * b->x[0] + b->x[2] * b->x[2]);
        if (rr > lim) { b->x[0] *= lim / rr; b->x[2] *= lim / rr; }
        if (b->tail > 0) b->tail = fmaxf(0, b->tail - DT * (0.08f + 1.2f * b->tail));

        /* the pool takes it back */
        if (b->x[1] - ry < h + 0.01f && b->v[1] <= 0.01f) {
            if (b->absorb == 0) ripple(b->x[0], b->x[2], 0.8f + 4 * r);
            b->absorb += DT / 2.5f;
            float dv = fminf(b->V, b->V * DT * 1.2f + 0.0004f * DT);
            b->V -= dv;
            sim.Tpool = (sim.Tpool * sim.Vpool + b->T * dv) / (sim.Vpool + dv);
            sim.Vpool += dv;
            if (b->V < 0.0006f || b->absorb >= 1) {
                sim.Tpool = (sim.Tpool * sim.Vpool + b->T * b->V) / (sim.Vpool + b->V);
                sim.Vpool += b->V;
                b->alive = 0;
            }
        }
    }

    /* touching blobs draw together and merge */
    for (int i = 0; i < MAXB; i++) {
        Blob *a = &sim.b[i];
        if (!a->alive || a->absorb > 0) continue;
        for (int j = i + 1; j < MAXB; j++) {
            Blob *c = &sim.b[j];
            if (!c->alive || c->absorb > 0) continue;
            float d[3] = {c->x[0] - a->x[0], c->x[1] - a->x[1], c->x[2] - a->x[2]};
            float ra = radius(a->V), rc = radius(c->V);
            float sy = 0.5f * (a->sy + c->sy);
            float dd = sqrtf(d[0] * d[0] + d[1] * d[1] / (sy * sy) + d[2] * d[2]);
            float reach = ra + rc;
            if (dd > reach * 1.08f) continue;
            if (dd < reach * 0.45f) {
                float V = a->V + c->V;
                for (int k = 0; k < 3; k++) {
                    a->x[k] = (a->x[k] * a->V + c->x[k] * c->V) / V;
                    a->v[k] = (a->v[k] * a->V + c->v[k] * c->V) / V;
                }
                a->T = (a->T * a->V + c->T * c->V) / V;
                a->sy = (a->sy * a->V + c->sy * c->V) / V;
                a->V = V;
                c->alive = 0;
                continue;
            }
            /* surface tension pulls the bridge shut */
            float pull = 0.06f * (1.08f - dd / reach) / fmaxf(dd, 1e-3f);
            float wa = c->V / (a->V + c->V), wc = 1 - wa;
            for (int k = 0; k < 3; k++) { a->x[k] += d[k] * pull * wa * DT; c->x[k] -= d[k] * pull * wc * DT; }
        }
        /* and a blob that drifts into the bud joins it */
        if (bd->on) {
            float d[3] = {bd->x[0] - a->x[0], bd->x[1] - a->x[1], bd->x[2] - a->x[2]};
            float dd = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (dd < (radius(a->V) + radius(bd->V)) * 0.5f) {
                bd->T = (bd->T * bd->V + a->T * a->V) / (bd->V + a->V);
                bd->V += a->V;
                a->alive = 0;
            }
        }
    }
    sim.steps++;
}

/* run the wax up to time t (seconds since switch-on) */
static void sim_advance(double t, int max_steps) {
    int n = 0;
    while (sim.t0 + (sim.steps + 1) * DT <= t) {
        sim_step();
        if (max_steps && ++n >= max_steps) { sim.t0 = t - sim.steps * DT; break; }
    }
}

/* --stats */
static void sim_stats(double t) {
    printf("t=%6.1f pool V=%.3f T=%.2f h=%.2f bud=%s", t, sim.Vpool, sim.Tpool, pool_h(sim.Vpool),
           sim.bud.on ? (sim.bud.feeding ? "feed" : "neck") : "-");
    if (sim.bud.on) printf("(y %.2f r %.2f T %.2f)", sim.bud.x[1], radius(sim.bud.V), sim.bud.T);
    printf(" blobs:");
    for (int i = 0; i < MAXB; i++)
        if (sim.b[i].alive) printf(" r%.2f@%.2f/T%.2f", radius(sim.b[i].V), sim.b[i].x[1], sim.b[i].T);
    printf("\n");
    fflush(stdout);
}

/* ---- the scene: one sideboard along the wall, a lamp on it for each
 * output ----
 * World units are lamp units (a standard lamp is LAMP_H tall); the
 * sideboard's top is y = 0, the wall is z = ZW, its front edge z = ZE. Each
 * output looks straight at the wall through its own camera, all of them at
 * the same height and distance, side by side: a parallel rig, so the wall,
 * the sideboard's edge and the light on them run on unbroken from one
 * output to the next (flightsim's single camera would stretch the side lamps
 * out of shape). The camera of output i sits as far along x as its centre
 * is from the front output's, scaled to the wall. */
#define DC 21.0f                       /* camera to the lamps */
#define YC 3.35f                       /* camera height */
#define ZW 1.9f                        /* the wall */
#define ZE (-2.4f)                     /* the sideboard's front edge */
static int me, front;                  /* this output; the front one */
static float out_f[MAX_OUT], out_y0[MAX_OUT];   /* focal length and horizon row (layout px) */
static float out_cx[MAX_OUT];          /* camera x (world) */
static float out_s[MAX_OUT];           /* lamp scale */
static int out_pal[MAX_OUT];
static int pal_front = -1;

static void setup_layout(void) {
    if (span_me < 0) {
        span_n = 1;
        span[0][0] = span[0][1] = 0; span[0][2] = (float)W; span[0][3] = (float)H;
        me = front = 0;
    } else {
        me = span_me;
        float x0 = 1e9f, x1 = -1e9f;
        for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
        float mx = (x0 + x1) / 2;
        front = 0;
        for (int i = 0; i < span_n; i++)
            if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { front = i; break; }
    }
    /* the front output: the lamp ~70% of the height, a little above centre */
    const float *fr = span[front];
    float F = 0.70f * fr[3] * DC / LAMP_H;
    float lamp_w = 2 * 0.97f * F / DC;
    if (lamp_w > 0.62f * fr[2]) F *= 0.62f * fr[2] / lamp_w;
    float Y0 = fr[1] + 0.46f * fr[3] + F * (LAMP_H / 2 - YC) / DC;
    float foot = Y0 + F * YC / DC;     /* the row the lamps stand on */
    float fxc = fr[0] + fr[2] / 2;
    /* other outputs: the same rig if the sideboard crosses them sensibly,
     * with a lamp as tall as fits (a grande on a portrait screen) */
    for (int i = 0; i < span_n; i++) {
        const float *r = span[i];
        out_f[i] = F; out_y0[i] = Y0;
        out_cx[i] = (r[0] + r[2] / 2 - fxc) * (DC + ZW) / F;
        if (i == front) { out_s[i] = 1; continue; }
        if (foot > r[1] + 0.35f * r[3] && foot < r[1] + 0.97f * r[3]) {
            float s = (foot - (r[1] + 0.09f * r[3])) * DC / (F * LAMP_H);
            out_s[i] = clampf(s, 0.55f, 1.7f);
        } else {
            /* framed on its own */
            float f = 0.70f * r[3] * DC / LAMP_H;
            out_f[i] = f;
            out_y0[i] = r[1] + 0.46f * r[3] + f * (LAMP_H / 2 - YC) / DC;
            out_s[i] = 1;
        }
    }
    /* palettes: the front's from --palette or the seed; the others, left
     * to right, complementary to it */
    if (pal_front < 0) {
        pal_front = (int)(hash32(opt_seed * 7919u + 13) % NPAL);
        if (opt_palette) {
            int k;
            for (k = 0; k < NPAL; k++) if (!strcmp(pals[k].name, opt_palette)) break;
            if (k == NPAL) {
                fprintf(stderr, "lavalamp: no palette \"%s\" (", opt_palette);
                for (int j = 0; j < NPAL; j++) fprintf(stderr, "%s%s", pals[j].name, j + 1 < NPAL ? ", " : ")\n");
                exit(1);
            }
            pal_front = k;
        }
    }
    for (int i = 0; i < span_n; i++) {
        if (i == front) { out_pal[i] = pal_front; continue; }
        int rk = 0;
        for (int j = 0; j < span_n; j++)
            if (j != front && j != i && (span[j][0] < span[i][0] || (span[j][0] == span[i][0] && j < i))) rk++;
        out_pal[i] = comp[pal_front][rk % 2];
    }
}

/* ---- the caption: the time, and how long the screen's been locked ---- */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[96]; } Style;
#define MAXSTYLE 8
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* regular, italic */
#define AW 1024
static unsigned char *atlas;
static int atlas_h, atlas_dirty;

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
    if (nstyles == MAXSTYLE) nstyles = 0;  /* sizes changed: start over */
    styles[nstyles].face = face;
    styles[nstyles].px = p;
    atlas_dirty = 1;
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
            FT_ULong code = c == 95 ? 0xB7 : (FT_ULong)(32 + c);   /* 0x7f: a middle dot */
            if (FT_Load_Char(f, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot g = f->glyph;
            int w = (int)g->bitmap.width, h = (int)g->bitmap.rows;
            if (x + w + 2 > AW) { x = 1; y += shelf + 2; shelf = 0; }
            if (y + h + 2 > cap) break;
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

static float *tv;
static int tvn, tvcap;
static void quad_v(float x, float y, float u, float v, const float *c) {
    if (tvn + 8 > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 4096) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3];
    tvn += 8;
}

/* str with its baseline at y; align -1 left, 0 centred, 1 right of x */
static void text(int s, float x, float y, int align, const char *str, const float *col, float track) {
    float w = text_width(s, str, track);
    float pen = roundf(x - (align == 0 ? w * 0.5f : align > 0 ? w : 0));
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

/* ---- shaders ---- */
#define GLSL(...) #__VA_ARGS__

/* a layer-per-instance fullscreen pass over a 3D texture: one point per
 * layer, the geometry shader makes it a triangle on that layer */
static const char *VS_LAYER = "#version 330 core\n" GLSL(
flat out int vL;
void main() { vL = gl_VertexID; gl_Position = vec4(0.0); }
);
static const char *GS_LAYER = "#version 330 core\n" GLSL(
layout(points) in;
layout(triangle_strip, max_vertices = 3) out;
flat in int vL[];
flat out int gL;
void main() {
    vec2 c[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    for (int i = 0; i < 3; i++) { gl_Layer = vL[0]; gL = vL[0]; gl_Position = vec4(c[i], 0.0, 1.0); EmitVertex(); }
    EndPrimitive();
}
);

#define FIELD_COMMON GLSL( \
uniform vec3 uBoxMin; uniform vec3 uBoxSize; uniform ivec3 uN; \
const float HG = 3.22; \
const vec3 BULB = vec3(0.0, -0.30, 0.0); \
float rg(float y) { return 0.635 - 0.095 * y - 0.16 * exp(-4.5 * y); } \
float rgd(float y) { return -0.095 + 0.72 * exp(-4.5 * y); } \
float ri(float y) { return rg(y) - 0.022; } \
vec3 boxuv(vec3 p) { return (p - uBoxMin) / uBoxSize; } \
)

/* the wax as a signed distance (R) and temperature (G), blobs, buds, necks
 * and pool blended smoothly */
static const char *FS_FIELD = "#version 330 core\n" FIELD_COMMON GLSL(
flat in int gL;
out vec2 o;
uniform vec4 uPool;            /* height, temperature, how solid, time */
uniform vec4 uRip[4];          /* x, z, age, strength */
uniform vec4 uStub;            /* x, z, length, radius */
uniform int uNB;
uniform vec4 uB[14];           /* x, y, z, r */
uniform vec4 uB2[14];          /* stretch, temperature, tail, wobble phase */
uniform float uBAbs[14];
uniform vec4 uBud;             /* x, y, z, r (r 0: none) */
uniform vec4 uBud2;            /* temperature, base x, base z, neck radius */

float smin(float a, float b, float k) { float h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }
float sdEll(vec3 p, vec3 r) { float k0 = length(p / r); float k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / max(k1, 1e-5); }
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a; float l2 = dot(ba, ba); float rr = r1 - r2; float a2 = l2 - rr * rr;
    if (a2 <= 1e-6) return min(length(p - a) - r1, length(p - b) - r2);   /* one end swallows the other */
    float il2 = 1.0 / l2;
    vec3 pa = p - a; float y = dot(pa, ba); float z = y - l2; vec3 xv = pa * l2 - ba * y; float x2 = dot(xv, xv);
    float y2 = y * y * l2; float z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

void main() {
    vec3 vox = uBoxSize / vec3(uN);
    vec3 p = uBoxMin + vec3(gl_FragCoord.xy, float(gL) + 0.5) * vox;
    float t = uPool.w;
    /* the pool: its surface heaves a little, ripples where things land, and
     * while solid it's a lumpy crust */
    float sol = uPool.z;
    float surf = uPool.x;
    surf += (0.012 + 0.03 * sol) * sin(p.x * 9.0 + t * 0.21) * sin(p.z * 8.0 - t * 0.17);
    surf += 0.02 * sol * sin(p.x * 23.0 + p.z * 5.0) * sin(p.z * 19.0 - p.x * 7.0);
    for (int k = 0; k < 4; k++) {
        float rr = length(p.xz - uRip[k].xy), age = uRip[k].z;
        surf += uRip[k].w * 0.035 * sin(rr * 26.0 - age * 5.0) * exp(-age * 0.9) * exp(-rr * 4.0) * step(age, 6.0);
    }
    /* domed, slumping at the glass */
    float R0 = ri(max(uPool.x, 0.05));
    surf += 0.30 * uPool.x * (0.6 - dot(p.xz, p.xz) / (R0 * R0));
    float d = p.y - surf;
    float tw = exp(-max(d, 0.0) * 25.0), tsum = tw * uPool.y;
    if (uStub.z > 0.0) {
        float ds = sdRoundCone(p, vec3(uStub.x, surf - 0.05, uStub.y), vec3(uStub.x, uPool.x + uStub.z, uStub.y), uStub.w * 2.2 + 0.04, uStub.w);
        d = smin(d, ds, 0.08);
    }
    if (uBud.w > 0.0) {
        vec3 c = uBud.xyz; float r = uBud.w, rn = uBud2.w;
        float db = sdEll(p - c, vec3(r * 0.97, r * 1.06, r * 0.97));
        vec3 a = vec3(uBud2.y, uPool.x - 0.05, uBud2.z), top = c - vec3(0.0, r * 0.6, 0.0);
        vec3 m = mix(a, top, 0.55);
        float dn = min(sdRoundCone(p, a, m, max(rn * 1.8, rn + 0.06), rn), sdRoundCone(p, m, top, rn, min(r * 0.7, rn * 1.9 + 0.01)));
        db = smin(db, dn, 0.06);
        float w = exp(-max(db, 0.0) * 25.0); tsum += w * uBud2.x; tw += w;
        d = smin(d, db, 0.07);
    }
    for (int i = 0; i < uNB; i++) {
        vec3 c = uB[i].xyz; float r = uB[i].w, sy = uB2[i].x;
        vec3 R = vec3(r / sqrt(sy), r * sy, r / sqrt(sy));
        vec3 q = p - c;
        /* a teardrop: the lower half drawn out a touch further when stretched */
        if (q.y < 0.0) R.y *= 1.0 + 0.35 * max(sy - 1.0, 0.0);
        float db = sdEll(q, R);
        float ph = uB2[i].w;
        db += 0.012 * (r / 0.16) * sin(q.x * 11.0 + ph + t * 0.37) * sin(q.y * 9.0 + ph * 1.7) * sin(q.z * 10.0 - t * 0.29);
        if (uB2[i].z > 0.0) {
            float L = uB2[i].z;
            db = smin(db, sdRoundCone(p, c - vec3(0.0, R.y * 0.5, 0.0), c - vec3(0.0, R.y + L, 0.0), min(r * 0.45, 0.06), 0.012 + 0.03 * L), 0.05);
        }
        float k = 0.06 + 0.12 * uBAbs[i];
        float w = exp(-max(db, 0.0) * 25.0); tsum += w * uB2[i].y; tw += w;
        d = smin(d, db, k);
    }
    o = vec2(clamp(d, -0.3, 0.3), tsum / max(tw, 1e-4));
}
);

/* how much wax lies between each voxel and the bulb */
static const char *FS_LIGHT = "#version 330 core\n" FIELD_COMMON GLSL(
flat in int gL;
out float o;
uniform sampler3D uField;
void main() {
    vec3 vox = uBoxSize / vec3(uN);
    vec3 p = uBoxMin + vec3(gl_FragCoord.xy, float(gL) + 0.5) * vox;
    vec3 d = BULB - p; float L = length(d); vec3 dir = d / L;
    float tmax = min(L, max(p.y - 0.0, 0.0) / max(-dir.y, 1e-3));
    float ds = tmax / 14.0, acc = 0.0;
    for (int i = 0; i < 14; i++) {
        vec3 q = p + dir * (float(i) + 0.5) * ds;
        acc += smoothstep(0.02, -0.02, texture(uField, boxuv(q)).r);
    }
    o = acc * ds;
}
);

static const char *VS_FULL = "#version 330 core\n" GLSL(
void main() {
    vec2 c[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(c[gl_VertexID], 0.0, 1.0);
}
);

/* the scene, in linear HDR at the render scale */
static const char *FS_SCENE = "#version 330 core\n" FIELD_COMMON GLSL(
out vec4 o;
uniform vec2 uRes;
uniform vec4 uView;            /* this output's rect (layout px) */
uniform vec3 uPP;              /* principal point x, y (layout px), focal length */
uniform vec3 uCam;
uniform int uNL;
uniform vec4 uLP[8];           /* each lamp's origin (world) and scale */
uniform vec3 uLC[8];           /* the light each lamp throws */
uniform int uMe;               /* which of them is ours */
uniform sampler3D uField;
uniform sampler3D uLight;
uniform vec3 uWax, uExt, uAbs, uSc, uBulb, uMetal, uGlow;
uniform int uMetalKind;
uniform float uGlitter, uTime, uSeed, uWarm;
uniform int uMode;             /* 1: just the room, for the cached background */
uniform float uPix;            /* a render pixel's size at the lamp, in lamp units */
const float ZW = 1.9, ZE = -2.4, FOOT = -1.75;

float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec3 hash33(vec3 p3) { p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yxx) * p3.zyx); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), f.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) { return 0.5 * vnoise(p) + 0.3 * vnoise(p * 2.1 + 3.1) + 0.2 * vnoise(p * 4.3 + 7.7); }

/* ---- the room ---- */
/* a dim window off to the left behind the camera: the cool highlight on the glass and metal */
const vec3 WIN = vec3(0.012, 0.016, 0.026);
vec3 windowLight(vec3 rd) {
    vec3 wd = normalize(vec3(-0.62, 0.30, -0.72));
    vec3 u = normalize(cross(wd, vec3(0.0, 1.0, 0.0))), v = cross(u, wd);
    float c = dot(rd, wd);
    if (c <= 0.0) return vec3(0.0);
    float a = dot(rd, u) / c, b = dot(rd, v) / c;
    float m = smoothstep(0.55, 0.15, abs(a)) * smoothstep(0.75, 0.25, abs(b));
    return WIN * 1.4 * m;
}

/* the light the lamps throw: each glass a line of light up its axis; the
 * base shades what's below its shoulder */
vec3 lampLight(vec3 p, vec3 n) {
    vec3 e = vec3(0.0);
    for (int l = 0; l < uNL; l++) {
        vec4 L = uLP[l]; float S = L.w;
        vec3 pl = (p - L.xyz) / S;
        vec3 acc = vec3(0.0);
        for (int k = 0; k < 5; k++) {
            float yl = 0.25 + (float(k) + 0.5) / 5.0 * 2.8;
            vec3 dl = vec3(0.0, yl, 0.0) - pl;
            float r2 = dot(dl, dl); vec3 dir = dl * inversesqrt(r2);
            float c = max(dot(n, dir), 0.0);
            float vis = 1.0;
            if (pl.y < 0.1) {
                float u1 = (0.1 - pl.y) / dl.y; vec3 q1 = pl + dl * u1;
                vis *= smoothstep(0.42, 0.75, length(q1.xz));
                if (pl.y < -1.45) { float u2 = (-1.45 - pl.y) / dl.y; vec3 q2 = pl + dl * u2; vis *= smoothstep(0.75, 1.15, length(q2.xz)); }
            }
            float prof = 1.35 - 0.45 * (yl / 3.0);
            acc += vec3(c * vis * prof / (r2 + 0.25));
        }
        e += uLC[l] * acc * (S * S / 5.0) / (S * S);
    }
    return e;
}

/* the rest of the room, behind the camera: a little of the lamps' light comes back off it */
vec3 roomBounce() { vec3 c = vec3(0.0); for (int l = 0; l < uNL; l++) c += uLC[l]; return c * 0.012; }

vec3 wallShade(vec3 p) {
    float g = fbm(p.xy * vec2(2.2, 2.6)) * 0.6 + fbm(p.xy * 11.0) * 0.4;
    vec3 alb = vec3(0.30, 0.26, 0.22) * (0.82 + 0.3 * g);
    float ao = 1.0 - 0.55 * exp(-p.y / 0.35);
    vec3 irr = lampLight(p, vec3(0.0, 0.0, -1.0)) + WIN * (0.15 + 0.15 * smoothstep(8.0, -8.0, p.x)) + vec3(0.0006, 0.0005, 0.0005);
    return alb * irr * ao;
}

/* how the lamps look in the varnish (rough: just their silhouettes, the glass glowing, the base dark) */
vec3 lampsInMirror(vec3 ro, vec3 rd, out float hit) {
    hit = 0.0;
    vec3 c = vec3(0.0);
    for (int l = 0; l < uNL; l++) {
        vec4 L = uLP[l]; float S = L.w;
        vec3 o = (ro - L.xyz) / S;
        vec2 oc = o.xz; vec2 d = rd.xz;
        float a = dot(d, d), b = dot(oc, d), cc = dot(oc, oc) - 0.55 * 0.55;
        float disc = b * b - a * cc;
        if (disc < 0.0) continue;
        float t = (-b - sqrt(disc)) / a;
        if (t < 0.0) continue;
        float y = o.y + rd.y * t;
        if (y > FOOT && y < 3.9) {
            hit = 1.0;
            float glass = smoothstep(0.05, 0.25, y) * smoothstep(3.25, 3.1, y);
            c = uLC[l] * 0.11 * glass * (1.3 - 0.3 * y / 3.0) + uLC[l] * 0.004;
        }
    }
    return c;
}

vec3 tableShade(vec3 p, vec3 rd) {
    /* walnut: long grain, figure, the odd darker streak */
    float w = p.x * 0.7 + 0.25 * fbm(vec2(p.x * 0.3, p.z * 2.0));
    float grain = fbm(vec2(p.x * 0.35, p.z * 9.0 + 2.0 * fbm(vec2(p.x * 0.5, p.z * 1.3))));
    float rings = 0.5 + 0.5 * sin(p.z * 26.0 + grain * 9.0 + w);
    vec3 alb = mix(vec3(0.085, 0.045, 0.024), vec3(0.19, 0.105, 0.056), grain * 0.7 + rings * 0.3);
    vec3 n = vec3(0.0, 1.0, 0.0);
    float ao = 1.0;
    for (int l = 0; l < uNL; l++) {
        float dd = length((p.xz - uLP[l].xz) / uLP[l].w);
        ao *= mix(0.25, 1.0, smoothstep(0.93, 1.6, dd));
    }
    ao *= 1.0 - 0.5 * exp(-(ZW - p.z) / 0.3);
    vec3 col = alb * (lampLight(p, n) + WIN * 0.6 + vec3(0.0015)) * ao;
    /* the varnish: a glossy reflection of the wall and the lamps */
    vec3 r = reflect(rd, n);
    float c = max(dot(-rd, n), 0.0);
    float F = 0.04 + 0.96 * pow(1.0 - c, 5.0);
    vec3 refl;
    float hit;
    vec3 lm = lampsInMirror(p, r, hit);
    if (hit > 0.0) refl = mix(wallShade(p + r * max((ZW - p.z) / max(r.z, 1e-3), 0.0)), lm, 0.7);
    else if (r.z > 1e-4) { vec3 q = p + r * ((ZW - p.z) / r.z); refl = wallShade(q); }
    else refl = windowLight(r);
    float rough = 0.6 + 0.4 * grain;
    return col + refl * F * rough * 0.9 * ao;
}

vec3 bg(vec3 ro, vec3 rd) {
    float t = 1e9; int m = 0;
    if (rd.z > 1e-5) { float tw = (ZW - ro.z) / rd.z; vec3 q = ro + rd * tw; if (tw > 0.0 && q.y >= 0.0) { t = tw; m = 1; } }
    if (rd.y < -1e-5 && ro.y > 0.0) { float tt = -ro.y / rd.y; vec3 q = ro + rd * tt; if (tt > 0.0 && tt < t && q.z >= ZE && q.z <= ZW) { t = tt; m = 2; } }
    if (rd.z > 1e-5 && ro.z < ZE) { float tf = (ZE - ro.z) / rd.z; vec3 q = ro + rd * tf; if (tf > 0.0 && tf < t && q.y < 0.0) { t = tf; m = 3; } }
    vec3 q = ro + rd * t;
    if (m == 1) return wallShade(q);
    if (m == 2) {
        vec3 c = tableShade(q, rd);
        /* the rounded front edge catches the light */
        c += lampLight(q, normalize(vec3(0.0, 1.0, -1.0))) * vec3(0.10, 0.06, 0.035) * smoothstep(0.06, 0.0, q.z - ZE);
        return c;
    }
    if (m == 3) {
        /* the sideboard's front: dark wood in shadow, a highlight along its lip */
        float grain = fbm(vec2(q.x * 0.35, q.y * 9.0));
        vec3 alb = mix(vec3(0.035, 0.018, 0.010), vec3(0.075, 0.040, 0.022), grain);
        vec3 irr = lampLight(vec3(q.x, 0.4, q.z), normalize(vec3(0.0, 0.35, -1.0))) * 0.06 + WIN * 0.8 + roomBounce() * 0.5;
        float lip = smoothstep(-0.12, -0.02, q.y) * smoothstep(0.0, -0.02, q.y);
        return alb * irr + lip * lampLight(q, vec3(0.0, 1.0, 0.0)) * 0.012 + alb * WIN * 0.5 * smoothstep(-6.0, -0.5, q.y);
    }
    return windowLight(rd) + roomBounce() + vec3(0.0004, 0.0004, 0.0005);
}

/* ---- the lamp (in its own units) ---- */
float sdPoly(vec2 p, vec2 v[10], int n) {
    float d = dot(p - v[0], p - v[0]); float s = 1.0;
    for (int i = 0, j = n - 1; i < n; j = i, i++) {
        vec2 e = v[j] - v[i]; vec2 w = p - v[i];
        vec2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));
        bvec3 c = bvec3(p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) s *= -1.0;
    }
    return s * sqrt(d);
}
const vec2 BASE[10] = vec2[10](vec2(0.0, -1.75), vec2(0.955, -1.75), vec2(0.975, -1.73), vec2(0.975, -1.665), vec2(0.945, -1.625),
                               vec2(0.535, -0.06), vec2(0.548, -0.035), vec2(0.548, 0.07), vec2(0.515, 0.10), vec2(0.0, 0.10));
const vec2 CAP[10] = vec2[10](vec2(0.0, 3.17), vec2(0.343, 3.17), vec2(0.352, 3.19), vec2(0.352, 3.245), vec2(0.335, 3.27),
                              vec2(0.205, 3.83), vec2(0.17, 3.885), vec2(0.11, 3.91), vec2(0.0, 3.915), vec2(0.0, 3.915));
float sdBase(vec3 p) { return sdPoly(vec2(length(p.xz), p.y), BASE, 10) - 0.004; }
float sdCap(vec3 p) { return sdPoly(vec2(length(p.xz), p.y), CAP, 9) - 0.004; }
float sdGlass(vec3 p) { return max((length(p.xz) - rg(p.y)) * 0.8, max(0.0 - p.y, p.y - 3.25)); }
float sdLamp(vec3 p, out int id) {
    float a = sdBase(p), b = sdCap(p), c = sdGlass(p);
    id = 0; float d = a;
    if (b < d) { d = b; id = 1; }
    if (c < d) { d = c; id = 2; }
    return d;
}
float sdMetal(vec3 p) { return min(sdBase(p), sdCap(p)); }
vec3 metalNormal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0); const float e = 0.0015;
    return normalize(k.xyy * sdMetal(p + k.xyy * e) + k.yyx * sdMetal(p + k.yyx * e) + k.yxy * sdMetal(p + k.yxy * e) + k.xxx * sdMetal(p + k.xxx * e));
}
vec3 glassNormal(vec3 p) { float r = max(length(p.xz), 1e-4); return normalize(vec3(p.x / r, -rgd(p.y), p.z / r)); }

vec3 Lp;  float Ls;            /* our lamp's origin and scale */
vec3 toWorld(vec3 p) { return Lp + p * Ls; }

/* the bulb's light at q in the liquid, before the wax shades it */
vec3 bulbE(vec3 q) {
    vec3 d = q - BULB; float r2 = dot(d, d); float r = sqrt(r2);
    float up = d.y / r;
    float cone = 0.25 + 0.75 * smoothstep(0.2, 0.95, up);
    return uBulb * cone / (r2 + 0.15) * exp(-uAbs * r);
}
vec3 E(vec3 q) { float wl = texture(uLight, boxuv(q)).r; return bulbE(q) * (0.3 + 0.7 * exp(-uExt * wl * 1.6)); }
vec2 fieldAt(vec3 q) { return texture(uField, boxuv(q)).rg; }
vec3 fieldGrad(vec3 q) { const float e = 0.012; return vec3(fieldAt(q + vec3(e, 0, 0)).r - fieldAt(q - vec3(e, 0, 0)).r, fieldAt(q + vec3(0, e, 0)).r - fieldAt(q - vec3(0, e, 0)).r, fieldAt(q + vec3(0, 0, e)).r - fieldAt(q - vec3(0, 0, e)).r); }

/* glitter: flakes carried round by the convection, flashing as they turn */
vec3 glint(vec3 q, vec3 d, float ds, vec3 e) {
    float r = length(q.xz);
    float vy = 0.05 * (1.0 - 2.0 * r * r / 0.24);
    float a = uTime * 0.03;
    vec3 g = vec3(q.x * cos(a) - q.z * sin(a), q.y - vy * uTime, q.x * sin(a) + q.z * cos(a));
    const float cs = 0.045;
    vec3 c = floor(g / cs);
    vec3 h = hash33(c + uSeed);
    vec3 fp = (c + 0.2 + 0.6 * h) * cs;
    fp.y += vy * uTime;
    fp = vec3(fp.x * cos(a) + fp.z * sin(a), fp.y, -fp.x * sin(a) + fp.z * cos(a));
    vec3 w = fp - q; float tc = dot(w, d);
    if (abs(tc) > ds * 0.5) return vec3(0.0);
    float d2 = dot(w, w) - tc * tc;
    float sz = 0.0035;
    vec3 nf = normalize(hash33(c * 1.7 + 4.0) * 2.0 - 1.0 + vec3(sin(uTime * (0.5 + h.x) + h.y * 6.0), cos(uTime * (0.4 + h.z)), 0.0) * 0.4);
    vec3 hv = normalize(normalize(BULB - fp) - d);
    float sp = pow(abs(dot(nf, hv)), 60.0);
    return e * (0.15 + 6.0 * sp) * exp(-d2 / (sz * sz)) * vec3(1.0, 0.86, 0.6) * 0.6;
}

vec3 shadeWax(vec3 p, vec3 d, float T) {
    const vec2 k = vec2(1.0, -1.0); const float e = 0.012;
    vec3 n = normalize(k.xyy * fieldAt(p + k.xyy * e).r + k.yyx * fieldAt(p + k.yyx * e).r + k.yxy * fieldAt(p + k.yxy * e).r + k.xxx * fieldAt(p + k.xxx * e).r);
    float solid = 1.0 - smoothstep(0.24, 0.34, T);
    vec3 ext = uExt * (1.0 + 2.0 * solid);
    vec3 l = normalize(BULB - p);
    float wrap = pow(max(0.0, dot(n, l) * 0.6 + 0.4), 2.2);
    vec3 e0 = E(p);
    /* light scattered inside the wax: march in a little way */
    vec3 sss = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        float s = 0.015 + float(i) * 0.04;
        vec3 q = p + d * s;
        float occ = smoothstep(0.02, -0.02, fieldAt(q).r);
        sss += E(q) * exp(-ext * s * 0.5) * occ * 0.04;
    }
    /* lit from below through the liquid (shaded by any wax between), wrapped round a little */
    vec3 eo = E(p + n * 0.035);
    vec3 diff = eo * wrap;
    /* the glowing liquid all round lights it softly from every side */
    vec3 amb = uSc * bulbE(p) * 0.035 + uSc * uBulb * 0.008;
    /* thin edges let the light through */
    float rim = pow(1.0 - max(dot(-d, n), 0.0), 2.0);
    /* light from the bulb shines up through the wax: the underside glows, the crown falls dark */
    float under = smoothstep(0.35, -0.85, n.y);
    vec3 col = uWax * (diff * 2.0 + sss * ext * 0.45 * (0.5 + 1.3 * under) + amb * 1.0 + rim * eo * 0.1);

    col *= (0.6 + 0.4 * pow(max(dot(-d, n), 0.0), 0.6)) * 2.6;


    /* hotter wax glows a touch more */
    col *= 0.85 + 0.4 * smoothstep(0.4, 0.75, T);
    /* solid wax: paler, chalky, more of its light from the surface */
    col = mix(col, mix(uWax, vec3(0.9), 0.35) * (e0 * wrap * 0.25 + amb * 2.0), solid * 0.6);
    /* a soft sheen: the wax is smooth, the bright liquid below reflects in it */
    vec3 rr = reflect(d, n);
    float fr = 0.03 + 0.3 * pow(1.0 - max(dot(-d, n), 0.0), 5.0);
    col += fr * uSc * bulbE(p) * (0.15 + 0.6 * smoothstep(0.0, -0.8, rr.y)) * (1.0 - solid * 0.7);
    return col;
}

/* along a refracted ray inside the liquid, from p, to whatever it meets */
vec3 interior(vec3 p, vec3 d, float jit) {
    /* where the ray leaves the liquid */
    float t = 0.0;
    for (int i = 0; i < 48; i++) {
        vec3 q = p + d * t;
        float g = min(ri(q.y) - length(q.xz), min(q.y - 0.02, HG - q.y));
        if (g < 0.0015 && i > 0) break;
        t += max(g * 0.8, 0.004);
    }
    float tex = t;
    vec3 tr = vec3(1.0), acc = vec3(0.0);
    /* cold, the liquid is cloudy; it clears as it warms */
    float cloud = 0.10 * (1.0 + 2.5 * (1.0 - uWarm) * (1.0 - uWarm));
    float tt = 0.0;
    float last = 1.0;
    /* the nearest miss, to soften the wax's silhouette over a pixel */
    float dmin = 1e9, tmin = 0.0; vec3 accm = vec3(0.0), trm = vec3(1.0);
    for (int i = 0; i < 160; i++) {
        if (tt >= tex) break;
        vec3 q = p + d * tt;
        vec2 f = fieldAt(q);
        if (f.r < 0.0) {
            /* refine between the last two steps */
            float a = tt - last, b = tt;
            for (int k = 0; k < 5; k++) { float m = 0.5 * (a + b); if (fieldAt(p + d * m).r < 0.0) b = m; else a = m; }
            vec3 h = p + d * b;
            return acc + tr * shadeWax(h, d, fieldAt(h).g);
        }
        if (f.r < dmin) { dmin = f.r; tmin = tt; accm = acc; trm = tr; }
        float ds = clamp(f.r * 0.8, 0.008, 0.03);
        ds = min(ds, tex - tt + 1e-3);
        if (i == 0) ds *= jit;
        vec3 e = E(q);
        acc += tr * e * uSc * cloud * ds;
        if (uGlitter > 0.0) acc += tr * glint(q, d, ds, e);
        tr *= exp(-(uAbs + vec3(1.8 * (1.0 - uWarm) * (1.0 - uWarm))) * ds);

        last = ds;
        tt += ds;
    }
    vec3 q = p + d * tex;
    vec3 miss;
    if (q.y < 0.05) miss = acc + tr * uBulb * vec3(1.0, 0.85, 0.6) * 0.06;
    else if (q.y > HG - 0.03) miss = acc + tr * uSc * uBulb * 0.002;
    else {
        vec3 n = glassNormal(q);
        vec3 d2 = refract(d, -n, 1.36);
        if (dot(d2, d2) < 0.5) miss = acc + tr * uSc * bulbE(q) * 0.02;
        else miss = acc + tr * bg(toWorld(q + n * 0.03), d2) * 0.96;
    }
    if (dmin < uPix) {
        vec3 h = p + d * tmin;
        h -= normalize(vec3(0.001) + fieldGrad(h)) * dmin;
        vec3 w = accm + trm * shadeWax(h, d, fieldAt(h).g);
        miss = mix(miss, w, 0.5 * (1.0 - dmin / uPix));
    }
    return miss;
}

vec3 metalShade(vec3 p, vec3 rd, int id) {
    vec3 n = metalNormal(p);
    vec3 r = reflect(rd, n);
    float c = max(dot(-rd, n), 0.0);
    /* brushed round the axis: streaks in the reflection */
    float ang = atan(p.z, p.x);
    float brush = 0.85 + 0.15 * vnoise(vec2(ang * 40.0, p.y * 900.0));
    vec3 env = bg(toWorld(p + n * 0.01), r);
    env += windowLight(r) * 0.5;
    /* the glowing glass seen in the metal: the base's shoulder looks up into it, the cap's skirt down */
    vec2 oc = p.xz; vec2 dd = r.xz;
    float a = dot(dd, dd), b = dot(oc, dd), cc = dot(oc, oc) - 0.5 * 0.5;
    float disc = b * b - a * cc;
    if (disc > 0.0 && a > 1e-5) {
        float t = (-b - sqrt(disc)) / a;
        float y = p.y + r.y * t;
        if (t > 0.0 && y > 0.0 && y < 3.2) env = uGlow * 1.6 * (1.25 - 0.3 * y / 3.2) * smoothstep(0.0, 0.25, y) * smoothstep(3.2, 2.9, y);
    }
    vec3 col;
    if (uMetalKind == 2) {
        /* black enamel */
        float F = 0.04 + 0.96 * pow(1.0 - c, 5.0);
        vec3 diff = vec3(0.012) * (lampLight(toWorld(p), n) + WIN);
        col = diff + env * F;
    } else {
        vec3 F = uMetal + (1.0 - uMetal) * pow(1.0 - c, 5.0);
        col = env * F * brush + uMetal * lampLight(toWorld(p), n) * 0.22;
    }
    return col;
}

vec3 shadeLamp(vec3 ro, vec3 rd, vec3 bgc, float jit) {
    /* the bounding cylinder */
    vec2 oc = ro.xz; vec2 d = rd.xz;
    float a = dot(d, d), b = dot(oc, d), c = dot(oc, oc) - 1.0;
    float disc = b * b - a * c;
    if (disc < 0.0) return bgc;
    float t0 = (-b - sqrt(disc)) / a, t1 = (-b + sqrt(disc)) / a;
    t0 = max(t0, 0.0);
    float t = t0; int id = -1;
    for (int i = 0; i < 96; i++) {
        vec3 p = ro + rd * t;
        int k; float dd = sdLamp(p, k);
        if (dd < 0.0005) { id = k; break; }
        t += dd;
        if (t > t1) break;
    }
    if (id < 0) return bgc;
    vec3 p = ro + rd * t;
    if (id != 2) return metalShade(p, rd, id);
    /* the glass: a reflection of the room over what the liquid shows */
    vec3 n = glassNormal(p);
    float cs = max(dot(-rd, n), 0.0);
    float F = 0.04 + 0.96 * pow(1.0 - cs, 5.0);
    F = min(1.0, F * 1.6);           /* both faces of the glass */
    vec3 r = reflect(rd, n);
    vec3 refl = bg(toWorld(p + n * 0.01), r) + windowLight(r);
    vec3 d1 = refract(rd, n, 1.0 / 1.36);
    vec3 pin = p + d1 * (0.022 / max(dot(d1, -n), 0.2));
    vec3 inside = interior(pin, d1, jit);
    /* the glass's own thickness at the rim, a faint dark-green edge */
    float edge = smoothstep(0.25, 0.0, cs);
    inside *= mix(vec3(1.0), vec3(0.55, 0.7, 0.62), edge * 0.6);
    /* at grazing angles the glass wall carries the liquid's glow along it: a bright rim */
    float yy = clamp(p.y / 3.2, 0.0, 1.0);
    inside += uSc * uBulb * 0.05 * (1.2 - yy) * smoothstep(0.32, 0.06, cs) * smoothstep(0.0, 0.15, p.y) * smoothstep(3.25, 3.1, p.y);

    return mix(inside, refl, F);
}

void main() {
    vec2 f = gl_FragCoord.xy / uRes;
    vec2 lp = uView.xy + vec2(f.x, 1.0 - f.y) * uView.zw;
    vec3 rd = normalize(vec3(lp.x - uPP.x, uPP.y - lp.y, uPP.z));
    vec3 ro = uCam;
    vec3 col = bg(ro, rd);
    if (uMode == 1) { o = vec4(col, 1.0); return; }
    Lp = uLP[uMe].xyz; Ls = uLP[uMe].w;
    float jit = 0.5 + 0.5 * hash12(gl_FragCoord.xy + fract(uTime * 7.31) * 100.0);
    col = shadeLamp((ro - Lp) / Ls, rd, col, jit);
    if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);
    o = vec4(col, 1.0);
}
);

/* bloom: a chain of half-size copies, then added back up */
static const char *FS_DOWN = "#version 330 core\n" GLSL(
out vec4 o;
uniform sampler2D uSrc;
uniform vec2 uTexel;           /* of the source */
uniform float uStep;           /* source texels per target texel */
void main() {
    vec2 uv = gl_FragCoord.xy * uStep * uTexel;

    vec3 c = texture(uSrc, uv + uTexel * vec2(-1.0, -1.0)).rgb + texture(uSrc, uv + uTexel * vec2(1.0, -1.0)).rgb
           + texture(uSrc, uv + uTexel * vec2(-1.0, 1.0)).rgb + texture(uSrc, uv + uTexel * vec2(1.0, 1.0)).rgb;
    o = vec4(c * 0.25, 1.0);
}
);
static const char *FS_UP = "#version 330 core\n" GLSL(
out vec4 o;
uniform sampler2D uSrc;
uniform vec2 uTexel;           /* of the source */
uniform vec2 uDst;             /* the target's size */
void main() {
    vec2 uv = gl_FragCoord.xy / uDst;
    vec3 c = vec3(0.0);
    c += texture(uSrc, uv + uTexel * vec2(-1.0, -1.0)).rgb + texture(uSrc, uv + uTexel * vec2(1.0, -1.0)).rgb;
    c += texture(uSrc, uv + uTexel * vec2(-1.0, 1.0)).rgb + texture(uSrc, uv + uTexel * vec2(1.0, 1.0)).rgb;
    c += 2.0 * (texture(uSrc, uv + uTexel * vec2(-1.0, 0.0)).rgb + texture(uSrc, uv + uTexel * vec2(1.0, 0.0)).rgb
              + texture(uSrc, uv + uTexel * vec2(0.0, -1.0)).rgb + texture(uSrc, uv + uTexel * vec2(0.0, 1.0)).rgb);
    c += 4.0 * texture(uSrc, uv).rgb;
    o = vec4(c / 16.0, 1.0);
}
);

/* the final picture: exposure, bloom, a soft filmic curve, a warm grade,
 * vignette and grain */
static const char *FS_FINAL = "#version 330 core\n" GLSL(
out vec4 o;
uniform sampler2D uHDR, uBloom;
uniform vec2 uOut;
uniform float uTime;
uniform vec4 uVig;             /* this output within the whole wall: x, y, w, h (0..1), and */
uniform float uVigAspect;      /* the wall's aspect */
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec3 filmic(vec3 x) {
    /* Hable's curve, toe lifted a hair */
    const float A = 0.22, B = 0.30, C = 0.10, D = 0.20, E = 0.0, F = 0.30;
    vec3 y = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    const float Wt = 11.2;
    float w = ((Wt * (A * Wt + C * B) + D * E) / (Wt * (A * Wt + B) + D * F)) - E / F;
    return y / w;
}
void main() {
    vec2 uv = gl_FragCoord.xy / uOut;
    vec3 c = texture(uHDR, uv).rgb;
    vec3 b = texture(uBloom, uv).rgb;
    c = c + b * 0.035;
    c *= 7.5;

    vec2 gw = uVig.xy + vec2(uv.x, 1.0 - uv.y) * uVig.zw;
    vec2 v = gw - 0.5; v.x *= uVigAspect;
    c *= 1.0 - 0.32 * smoothstep(0.35, 1.05, length(v) * min(1.0, 1.6 / max(uVigAspect, 1.0)));
    c = filmic(c);
    /* grade: shadows a little warm, saturation held */
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, 1.06);
    c += vec3(0.004, 0.0025, 0.001) * (1.0 - smoothstep(0.0, 0.3, l));
    c = pow(max(c, 0.0), vec3(1.0 / 2.2));
    /* grain, a touch stronger in the mids, and a dither */
    float g = hash12(gl_FragCoord.xy + fract(uTime * 13.7) * 811.0) + hash12(gl_FragCoord.xy * 1.37 + fract(uTime * 7.1) * 433.0) - 1.0;
    c += g * 0.022 * (0.35 + smoothstep(0.0, 0.4, l) * (1.0 - smoothstep(0.5, 1.0, l)));
    o = vec4(c, 1.0);
}
);

static const char *VS_TEXT = "#version 330 core\n" GLSL(
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aCol;
uniform vec2 uRes, uAtlasSize;
out vec2 vUV; out vec4 vCol;
void main() {
    vUV = aUV / uAtlasSize; vCol = aCol;
    gl_Position = vec4(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0, 0.0, 1.0);
}
);
static const char *FS_TEXT = "#version 330 core\n" GLSL(
in vec2 vUV; in vec4 vCol; out vec4 o;
uniform sampler2D uAtlas;
void main() { o = vec4(vCol.rgb, vCol.a * texture(uAtlas, vUV).r); }
);

/* ---- GL ---- */
static GLuint p_field, p_light, p_scene, p_down, p_up, p_final, p_text;
static GLuint vao, vao_text, vbo_text, tex_atlas;
static GLuint tex_field, tex_light, fbo_field, fbo_light;
static GLuint tex_hdr, fbo_hdr, tex_room, fbo_room;
static double room_at = -1e9;          /* when the cached room was drawn */
#define NBLOOM 5
static GLuint tex_bl[NBLOOM], fbo_bl[NBLOOM];
static int bl_w[NBLOOM], bl_h[NBLOOM];
static int rt_w, rt_h;
static float render_scale = 1;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "lavalamp: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *vs, const char *gs, const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs));
    if (gs) glAttachShader(pr, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "lavalamp: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

static GLuint mk_tex2d(int w, int h, GLenum ifmt, GLenum filter) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static GLuint mk_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}
static GLuint mk_tex3d(GLenum ifmt, GLenum fmt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_3D, t);
    glTexImage3D(GL_TEXTURE_3D, 0, ifmt, FX, FY, FZ, 0, fmt, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return t;
}

static void gl_init(void) {
    p_field = mk_prog(VS_LAYER, GS_LAYER, FS_FIELD);
    p_light = mk_prog(VS_LAYER, GS_LAYER, FS_LIGHT);
    p_scene = mk_prog(VS_FULL, NULL, FS_SCENE);
    p_down = mk_prog(VS_FULL, NULL, FS_DOWN);
    p_up = mk_prog(VS_FULL, NULL, FS_UP);
    p_final = mk_prog(VS_FULL, NULL, FS_FINAL);
    p_text = mk_prog(VS_TEXT, NULL, FS_TEXT);
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

    tex_field = mk_tex3d(GL_RG16F, GL_RG);
    tex_light = mk_tex3d(GL_R16F, GL_RED);
    glGenFramebuffers(1, &fbo_field);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_field);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_field, 0);
    glGenFramebuffers(1, &fbo_light);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_light);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_light, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("layered framebuffer incomplete");
    /* the fixed uniforms of the 3D passes */
    GLuint ps[3] = {p_field, p_light, p_scene};
    for (int i = 0; i < 3; i++) {
        glUseProgram(ps[i]);
        glUniform3fv(U(ps[i], "uBoxMin"), 1, box_min);
        glUniform3fv(U(ps[i], "uBoxSize"), 1, box_size);
        glUniform3i(U(ps[i], "uN"), FX, FY, FZ);
    }
}

/* the render targets: the scene at the render scale, and the bloom chain */
static void ensure_targets(void) {
    /* portrait: the lamp fills more of the screen, so a little coarser */
    if (opt_scale <= 0) render_scale = H > W * 1.2f ? 0.6f : 0.7f;
    int w = (int)(W * render_scale + 0.5f), h = (int)(H * render_scale + 0.5f);
    if (w < 16) w = 16;
    if (h < 16) h = 16;
    if (w == rt_w && h == rt_h) return;
    if (rt_w) {
        glDeleteTextures(1, &tex_hdr); glDeleteFramebuffers(1, &fbo_hdr);
        glDeleteTextures(1, &tex_room); glDeleteFramebuffers(1, &fbo_room);
        glDeleteTextures(NBLOOM, tex_bl); glDeleteFramebuffers(NBLOOM, fbo_bl);
    }
    rt_w = w; rt_h = h;
    tex_hdr = mk_tex2d(w, h, GL_R11F_G11F_B10F, GL_LINEAR);
    fbo_hdr = mk_fbo(tex_hdr);
    tex_room = mk_tex2d(w, h, GL_R11F_G11F_B10F, GL_NEAREST);
    fbo_room = mk_fbo(tex_room);
    room_at = -1e9;
    for (int i = 0; i < NBLOOM; i++) {
        w = (w + (i ? 1 : 3)) / (i ? 2 : 4); h = (h + (i ? 1 : 3)) / (i ? 2 : 4);
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        bl_w[i] = w; bl_h[i] = h;
        tex_bl[i] = mk_tex2d(w, h, GL_R11F_G11F_B10F, GL_LINEAR);
        fbo_bl[i] = mk_fbo(tex_bl[i]);
    }
}

static void lin3(const float *srgb, float *out, float k) { for (int i = 0; i < 3; i++) out[i] = srgb[i] * k; }

/* the light a lamp throws on the room, by palette and how warmed up */
static void lamp_glow(int pal, double t, float *out) {
    float w = 0.55f + 0.45f * warmth(t);
    lin3(pals[pal].glow, out, 0.3f * w);
}

/* the wax's state, a step ago blended with now */
static void upload_wax(float a) {
    GLuint p = p_field;
    glUseProgram(p);
    const Sim *s0 = &sim_prev, *s1 = &sim;
    double t = s1->t0 + (s1->steps - 1 + a) * DT;
    float h = pool_h(s0->Vpool + (s1->Vpool - s0->Vpool) * a);
    float Tp = s0->Tpool + (s1->Tpool - s0->Tpool) * a;
    glUniform4f(U(p, "uPool"), h, Tp, 1 - smooth(0.2f, T_MELT + 0.04f, Tp), (float)fmod(t, 10000.0));
    float rip[4][4];
    for (int k = 0; k < 4; k++) { memcpy(rip[k], s1->rip[k], sizeof rip[k]); if (s1->rip[k][3] == 0) rip[k][2] = 99; }
    glUniform4fv(U(p, "uRip"), 4, &rip[0][0]);
    glUniform4f(U(p, "uStub"), s1->stub_x, s1->stub_z, s0->stub + (s1->stub - s0->stub) * a, s1->stub_r);
    float B[MAXB][4], B2[MAXB][4], BA[MAXB];
    int n = 0;
    for (int i = 0; i < MAXB; i++) {
        const Blob *b = &s1->b[i], *o = &s0->b[i];
        if (!b->alive) continue;
        int old = o->alive && fabsf(o->x[1] - b->x[1]) < 0.2f;
        for (int k = 0; k < 3; k++) B[n][k] = old ? o->x[k] + (b->x[k] - o->x[k]) * a : b->x[k];
        float V = old ? o->V + (b->V - o->V) * a : b->V;
        B[n][3] = radius(V);
        B2[n][0] = old ? o->sy + (b->sy - o->sy) * a : b->sy;
        B2[n][1] = b->T;
        B2[n][2] = old ? o->tail + (b->tail - o->tail) * a : b->tail;
        B2[n][3] = b->ph;
        BA[n] = b->absorb;
        n++;
    }
    glUniform1i(U(p, "uNB"), n);
    if (n) {
        glUniform4fv(U(p, "uB"), n, &B[0][0]);
        glUniform4fv(U(p, "uB2"), n, &B2[0][0]);
        glUniform1fv(U(p, "uBAbs"), n, BA);
    }
    const Bud *b = &s1->bud, *o = &s0->bud;
    if (b->on) {
        int old = o->on;
        float x[3];
        for (int k = 0; k < 3; k++) x[k] = old ? o->x[k] + (b->x[k] - o->x[k]) * a : b->x[k];
        float V = old ? o->V + (b->V - o->V) * a : b->V, Vn = old ? o->Vn + (b->Vn - o->Vn) * a : b->Vn;
        float rb = radius(V), L = fmaxf(0.005f, x[1] - rb * 0.75f - h);
        float rn = fminf(sqrtf(Vn / (PI * L)), rb);
        glUniform4f(U(p, "uBud"), x[0], x[1], x[2], rb);
        glUniform4f(U(p, "uBud2"), b->T, b->base[0], b->base[1], fmaxf(rn, 0.012f));
    } else glUniform4f(U(p, "uBud"), 0, 0, 0, 0);
}

static double t_now;                   /* seconds since switch-on, for the shaders */

/* --bench: GPU time per pass */
enum { PR_FIELD, PR_LIGHT, PR_SCENE, PR_BLOOM, PR_FINAL, PR_N };
static const char *prof_name[PR_N] = {"field", "light", "scene", "bloom", "final"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N];
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (t[i + 1] - t[i]) * 1e-6;
}

static void caption(double wt) {
    tvn = 0;
    if (!opt_caption || me != front) return;
    const float *r = span[me];
    float sc = (float)H / r[3];        /* layout px to ours */
    float u = fminf(r[2], r[3]) * sc;
    int s1 = style(0, u * 0.034f), s2 = style(1, u * 0.019f);
    if (atlas_dirty) {
        build_atlas();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        atlas_dirty = 0;
    }
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    char tbuf[32], abuf[64];
    snprintf(tbuf, sizeof tbuf, "%d:%02d", tm.tm_hour, tm.tm_min);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    if (away < 60) snprintf(abuf, sizeof abuf, "away a moment");
    else if (away < 3600) snprintf(abuf, sizeof abuf, "away %ld min", away / 60);
    else snprintf(abuf, sizeof abuf, "away %ld h %02ld min", away / 3600, (away / 60) % 60);
    const float c1[4] = {0.86f, 0.78f, 0.66f, 0.62f}, c2[4] = {0.80f, 0.72f, 0.60f, 0.48f};
    float m = u * 0.055f;
    float x = W - m, y = H - m;
    text(s2, x, y, 1, abuf, c2, u * 0.0016f);
    text(s1, x, y - u * 0.034f, 1, tbuf, c1, u * 0.001f);
}

/* t: seconds since switch-on; a: how far between the last two steps; wt: wall time */
static void render(double t, float a, double wt, GLuint target) {
    setup_layout();
    ensure_targets();
    t_now = t;
    const Palette *P = &pals[out_pal[me]];

    /* the wax, and the light through it */
    prof_mark(PR_FIELD);
    glDisable(GL_BLEND);
    glBindVertexArray(vao);
    glViewport(0, 0, FX, FY);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_field);
    upload_wax(a);
    glDrawArrays(GL_POINTS, 0, FZ);
    prof_mark(PR_LIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_light);
    glUseProgram(p_light);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex_field);
    glUniform1i(U(p_light, "uField"), 0);
    glDrawArrays(GL_POINTS, 0, FZ);

    /* the scene: the room is cached (it changes only as the lamps warm up)
     * and copied in; only the lamp's own box is drawn every frame */
    prof_mark(PR_SCENE);
    GLuint p = p_scene;
    glUseProgram(p);
    glUniform2f(U(p, "uRes"), (float)rt_w, (float)rt_h);
    glUniform4fv(U(p, "uView"), 1, span[me]);
    const float *r = span[me];
    glUniform3f(U(p, "uPP"), r[0] + r[2] / 2, out_y0[me], out_f[me]);
    glUniform3f(U(p, "uCam"), out_cx[me], YC, -DC);
    float LP[MAX_OUT][4], LC[MAX_OUT][3];
    for (int i = 0; i < span_n; i++) {
        LP[i][0] = out_cx[i]; LP[i][1] = -LAMP_FOOT * out_s[i]; LP[i][2] = 0; LP[i][3] = out_s[i];
        lamp_glow(out_pal[i], t, LC[i]);
    }
    glUniform1i(U(p, "uNL"), span_n);
    glUniform4fv(U(p, "uLP"), span_n, &LP[0][0]);
    glUniform3fv(U(p, "uLC"), span_n, &LC[0][0]);
    glUniform1i(U(p, "uMe"), me);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex_field);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, tex_light);
    glUniform1i(U(p, "uField"), 0);
    glUniform1i(U(p, "uLight"), 1);
    glUniform3fv(U(p, "uWax"), 1, P->wax);
    glUniform3fv(U(p, "uExt"), 1, P->ext);
    glUniform3fv(U(p, "uAbs"), 1, P->abs);
    glUniform3fv(U(p, "uSc"), 1, P->sc);
    float bulb = 0.35f + 0.65f * fminf(1, (float)t / 4);   /* the bulb comes up as the filament warms */
    glUniform3f(U(p, "uBulb"), 1.0f * bulb, 0.86f * bulb, 0.64f * bulb);
    static const float metals[3][3] = {{0.80f, 0.60f, 0.32f}, {0.82f, 0.82f, 0.84f}, {0.04f, 0.04f, 0.04f}};
    glUniform3fv(U(p, "uMetal"), 1, metals[P->metal]);
    glUniform1i(U(p, "uMetalKind"), P->metal);
    float gl[3];
    lamp_glow(out_pal[me], t, gl);
    for (int i = 0; i < 3; i++) gl[i] *= 1.6f;
    glUniform3fv(U(p, "uGlow"), 1, gl);
    glUniform1f(U(p, "uGlitter"), (float)P->glitter);
    glUniform1f(U(p, "uTime"), (float)fmod(t, 10000.0));
    glUniform1f(U(p, "uSeed"), (float)(opt_seed % 1000));
    glUniform1f(U(p, "uWarm"), warmth(t));
    if (fabs(t - room_at) > 1.0) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_room);
        glViewport(0, 0, rt_w, rt_h);
        glUniform1i(U(p, "uMode"), 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        room_at = t;
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_room);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_hdr);
    glBlitFramebuffer(0, 0, rt_w, rt_h, 0, 0, rt_w, rt_h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_hdr);
    glViewport(0, 0, rt_w, rt_h);
    glUniform1i(U(p, "uMode"), 0);
    glUniform1f(U(p, "uPix"), 1.3f * DC / (out_f[me] * rt_h / r[3]) / out_s[me]);
    {   /* the lamp's box on screen: its corners through our camera */
        float S = out_s[me], F = out_f[me], cx = r[0] + r[2] / 2, x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        for (int k = 0; k < 8; k++) {
            float X = ((k & 1) ? 1.0f : -1.0f) * 1.0f * S, Y = (k & 2) ? (LAMP_H + 0.05f) * S : -0.05f, Z = ((k & 4) ? 1.0f : -1.0f) * S;
            float lx = cx + F * X / (Z + DC), ly = out_y0[me] - F * (Y - YC) / (Z + DC);
            float px = (lx - r[0]) / r[2] * rt_w, py = (1 - (ly - r[1]) / r[3]) * rt_h;
            x0 = fminf(x0, px); x1 = fmaxf(x1, px); y0 = fminf(y0, py); y1 = fmaxf(y1, py);
        }
        int sx0 = (int)clampf(x0 - 4, 0, rt_w), sx1 = (int)clampf(x1 + 4, 0, rt_w);
        int sy0 = (int)clampf(y0 - 4, 0, rt_h), sy1 = (int)clampf(y1 + 4, 0, rt_h);
        if (sx1 > sx0 && sy1 > sy0) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(sx0, sy0, sx1 - sx0, sy1 - sy0);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glDisable(GL_SCISSOR_TEST);
        }
    }

    /* bloom */
    prof_mark(PR_BLOOM);
    glUseProgram(p_down);
    glUniform1i(U(p_down, "uSrc"), 0);
    glActiveTexture(GL_TEXTURE0);
    for (int i = 0; i < NBLOOM; i++) {

        int sw = i ? bl_w[i - 1] : rt_w, sh = i ? bl_h[i - 1] : rt_h;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bl[i]);
        glViewport(0, 0, bl_w[i], bl_h[i]);
        glBindTexture(GL_TEXTURE_2D, i ? tex_bl[i - 1] : tex_hdr);
        glUniform2f(U(p_down, "uTexel"), 1.0f / sw, 1.0f / sh);
        glUniform1f(U(p_down, "uStep"), i ? 2.0f : 4.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glUseProgram(p_up);
    glUniform1i(U(p_up, "uSrc"), 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    for (int i = NBLOOM - 1; i > 0; i--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bl[i - 1]);
        glViewport(0, 0, bl_w[i - 1], bl_h[i - 1]);
        glBindTexture(GL_TEXTURE_2D, tex_bl[i]);
        glUniform2f(U(p_up, "uTexel"), 1.0f / bl_w[i], 1.0f / bl_h[i]);
        glUniform2f(U(p_up, "uDst"), (float)bl_w[i - 1], (float)bl_h[i - 1]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glDisable(GL_BLEND);

    /* the final picture */
    prof_mark(PR_FINAL);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_final);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_hdr);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_bl[0]);
    glUniform1i(U(p_final, "uHDR"), 0);
    glUniform1i(U(p_final, "uBloom"), 1);
    glUniform2f(U(p_final, "uOut"), (float)W, (float)H);
    glUniform1f(U(p_final, "uTime"), (float)fmod(t, 1000.0));
    {
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int i = 0; i < span_n; i++) {
            x0 = fminf(x0, span[i][0]); y0 = fminf(y0, span[i][1]);
            x1 = fmaxf(x1, span[i][0] + span[i][2]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
        }
        glUniform4f(U(p_final, "uVig"), (r[0] - x0) / (x1 - x0), (r[1] - y0) / (y1 - y0), r[2] / (x1 - x0), r[3] / (y1 - y0));
        glUniform1f(U(p_final, "uVigAspect"), (x1 - x0) / (y1 - y0));
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);

    caption(wt);
    if (tvn) {
        glUseProgram(p_text);
        glUniform2f(U(p_text, "uRes"), (float)W, (float)H);
        glUniform2f(U(p_text, "uAtlasSize"), (float)AW, (float)atlas_h);
        glUniform1i(U(p_text, "uAtlas"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glBindVertexArray(vao_text);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STREAM_DRAW);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES, 0, tvn / 8);
        glDisable(GL_BLEND);
        glBindVertexArray(vao);
    }
    prof_mark(PR_N);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "lavalamp");
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
        "usage: lavalamp [--window | --root | --layer] [--seed N] [--palette NAME] [--fps N]\n"
        "                [--scale X] [--font FAMILY] [--no-caption] [--cpu]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--stats]\n"
        "  --palette  classic, blue, purple, green, white or glitter (default: from the seed)\n"
        "  --scale    render scale for the scene (default 0.7, 0.6 portrait; the caption and grain are always full size)\n"
        "  --cpu      the bulb runs hotter with the CPU load (the wax then no longer repeats run to run)\n"
        "  --at       --shot: that many seconds after switch-on (default 240; 0-120 is the warm-up)\n");
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

/* each output's lamp is its own: its wax from the seed and its place */
static uint32_t lamp_seed(void) { return opt_seed * 2654435761u + (uint32_t)(span_me >= 0 ? span_me : 0) * 40503u + 1; }

/* starting long after switch-on: begin with a warm lamp a few minutes back,
 * rather than simulate hours */
#define CATCH_UP 300.0
static void sim_start(double t) {
    double t0 = t > 2 * CATCH_UP ? t - CATCH_UP : 0;
    sim_init(lamp_seed(), t0);
    sim_advance(t, 0);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-cpu")) opt_cpu = 1;
        else if (!strcmp(a, "-stats")) opt_stats = 1;
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
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    render_scale = opt_scale > 0 ? clampf(opt_scale, 0.25f, 1) : 0.7f;
    int offscreen = opt_shot || opt_bench > 0;
    fonts_init();

    /* --epoch: when the lamps were switched on, shared by every instance */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
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
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Lava Lamp");
        XClassHint chint = {"lavalamp", "Lavalamp"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (offscreen) {
        GLuint tex = mk_tex2d(W, H, GL_RGBA8, GL_NEAREST), fbo = mk_fbo(tex);
        double at = opt_shot ? opt_at : 200;
        double t0 = now();
        sim_start(at);
        double tsim = now() - t0;
        if (opt_stats) sim_stats(at);
        double wt = t_wall0 + at;
        if (opt_shot) {
            render(at, 1, wt, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        render(at, 1, wt, fbo);
        glFinish();
        double sum = 0, worst = 0;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = at + (i + 1) / 60.0;
            sim_advance(t, 0);
            render(t, 1, wt + (i + 1) / 60.0, fbo);
            glFinish();
            prof_collect();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("lavalamp: %dx%d scale %.2f, %d frames: avg %.3f ms, worst %.3f ms per frame (wax run up in %.0f ms)\n",
               W, H, render_scale, opt_bench, sum / opt_bench, worst, tsim * 1000);
        printf("lavalamp: CPU %.3f ms per frame\n", (double)clock() / CLOCKS_PER_SEC * 1000 / opt_bench);

        printf("lavalamp: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        for (int i = 0; i < PR_N; i++) printf("  %-6s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);

        return 0;
    }

    double el0 = opt_epoch ? fmax(wall() - opt_epoch, 0) : 0;
    sim_start(el0);
    double t0 = now() - el0, last_cpu = 0, last_stats = 0;
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
        double el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        if (opt_cpu && t - last_cpu >= 1) { read_cpu(); last_cpu = t; }
        sim_advance(el, 30);
        /* drawn a step behind, blending the last two steps, so it's smooth at any frame rate */
        float a = (float)((el - (sim.t0 + sim.steps * DT)) / DT);
        render(el, clampf(a, 0, 1), wt, 0);
        if (opt_stats && el - last_stats >= 5) { sim_stats(el); last_stats = el; }
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
