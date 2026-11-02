/*
 * library — a round stone library high on a sea cliff, where nobody is home.
 *
 * Curved oak shelves of leather-bound books run round a stone wall pierced
 * by tall arched windows onto a grey-green sea; a low fire burns in the
 * hearth; dust drifts in the shafts of light. On a reading desk, by a brass
 * lamp, a cup of tea going cold and a chair pushed back, a great book lies
 * open. Its right-hand page carries an inset image that is a live window
 * into another world (an "age"):
 *
 *   SALTWATCH   a sea fortress on a rock, in a storm, lightning and rain
 *   LANTERNWOOD a forest of giant glowing mushrooms, spores drifting
 *   THE MERE    a glass greenhouse alight on a frozen lake, snow falling
 *   THE DRIFT   floating islands of rock over cloud, at sunset
 *   VERDIGRIS   a desert of copper towers in the heat haze
 *
 * Each is a small raymarched scene rendered into a texture at a low rate
 * and shown on the page like an old hand-tinted photograph. Every few
 * minutes the book turns its pages (they bend as they go) to another age,
 * and the camera, drifting slowly round the room, dollies in towards the
 * image as if about to step into it, then eases back.
 *
 * The left-hand page is a keeper's journal of the machine's day, written
 * in ink that appears as you watch: the host, its user, when they left the
 * desk (swaylock's start time), how long the house has been awake, the CPU
 * temperature and load. The real local time sets the light outside: sun
 * shafts by day (the sun's real elevation for the date), a long low gold at
 * dusk, moonlight (with the real phase) at night, when the fire and the
 * lamp carry the room.
 *
 * The room is built once as meshes (stone, oak, leather, brass, paper) and
 * lit forward with a shadowed sun/moon, the fire and the lamp (their shadows
 * rendered once; nothing they light moves), sky light from the windows and
 * a voxel ambient occlusion. Light shafts are marched through the sun's
 * shadow map at quarter resolution; bloom and a filmic grade finish it.
 *
 * Spanning: run one instance per monitor with the same --seed, --epoch and
 * --layout (every output's logical rect), and --output = this one's index.
 * As in flightsim, one virtual camera is centred on the output under the
 * middle of the layout and each instance draws its own slice of the view;
 * everything is a function of the seed and the time since the epoch.
 *
 * Same plumbing as gauges: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen.
 * --bench N times N frames offscreen at --size (LIBRARY_PROF=1: per stage).
 *
 * Also: --hour H pretends it's that local time of day (the light), --age N
 * keeps one world in the book (0..4), --name NAME is who left the desk,
 * --still holds the camera on the desk, --scale S fixes the render scale
 * (else it tracks the frame budget), --fov DEG the horizontal angle across
 * a spanned layout (default 100).
 *
 * The timeline: every 180 s the leaves turn (three of them, bending as
 * they go) to the next world; the camera eases in towards the picture
 * until it nearly fills the middle screen, holds, eases back, then drifts
 * to look at two other corners of the room before coming back to the desk.
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
#include <pwd.h>
#include <stdarg.h>
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
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the timeline starts at */
static float opt_hour = -1;            /* --hour: pretend it's this local time of day */
static int opt_age = -1;               /* --age: always this world in the book */
static float opt_scale;                /* render scale; 0 = dynamic */
static float opt_fov = 100;            /* spanning: horizontal degrees across the whole layout */
static const char *opt_name;           /* --name: who the journal says left the desk */
static int opt_still;                  /* --still: no camera moves */

/* ---- spanning: one virtual camera, a slice of its view per output ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

static void die(const char *what) { fprintf(stderr, "library: %s\n", what); exit(1); }

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

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float smoothf(float a, float b, float x) {
    float t = clampf((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

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

/* ---- the machine ---- */
static char host[64], user[64];
static time_t locked_at;               /* swaylock's start, or 0 */
static char temp_path[300];

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
    float temp;                        /* degrees C, 0 if unknown */
    float lavg[3];
    double uptime;                     /* seconds */
} sens;

static void read_sensors(void) {
    char buf[128];
    sens.temp = temp_path[0] ? read_long(temp_path, 0) / 1000.0f : 0;
    slurp_line("/proc/loadavg", buf, sizeof buf);
    sscanf(buf, "%f %f %f", &sens.lavg[0], &sens.lavg[1], &sens.lavg[2]);
    slurp_line("/proc/uptime", buf, sizeof buf);
    sens.uptime = strtod(buf, NULL);
}

/* who keeps this desk: --name, else the first word of the account's full
 * name, else the login, capitalised */
static void find_user(void) {
    if (opt_name) { snprintf(user, sizeof user, "%s", opt_name); return; }
    struct passwd *pw = getpwuid(getuid());
    user[0] = 0;
    if (pw && pw->pw_gecos && pw->pw_gecos[0] && pw->pw_gecos[0] != ',') {
        size_t n = strcspn(pw->pw_gecos, " ,");
        if (n > 0 && n < sizeof user) snprintf(user, sizeof user, "%.*s", (int)n, pw->pw_gecos);
    }
    if (!user[0] && pw && pw->pw_name) snprintf(user, sizeof user, "%s", pw->pw_name);
    if (!user[0]) snprintf(user, sizeof user, "the keeper");
    user[0] = (char)toupper((unsigned char)user[0]);
}

/* ---- the worlds in the book ---- */
#define NAGE 5
static const char *age_name[NAGE] = {"Saltwatch", "Lanternwood", "The Mere", "The Drift", "Verdigris"};
static const char *age_note[NAGE] = {
    "A keep upon a rock, and the sea forever at its door. The lamp in the high window has not gone out.",
    "Under the caps the light is green and slow. Spores rise like embers that have forgotten to fall.",
    "A glass house upon the ice, warm within. Someone tends it, though I have never seen them.",
    "Islands that have let go of the ground, and the evening that never quite ends.",
    "Copper spires in the heat, singing faintly when the wind is right. The sand remembers rain.",
};

/* the timeline: every PERIOD seconds the book turns to another world */
#define PERIOD 180.0
#define TURN_T 3.2                     /* the page turn, seconds (three leaves, staggered) */
#define PHASE0 22.0                    /* the timeline starts this far into a period */

static int age_of(long period) {
    if (opt_age >= 0) return opt_age % NAGE;
    /* a shuffled order per cycle of five, so neighbours differ */
    long cyc = period >= 0 ? period / NAGE : (period - NAGE + 1) / NAGE;
    int k = (int)(period - cyc * NAGE);
    int ord[NAGE] = {0, 1, 2, 3, 4};
    uint32_t h = hash32(opt_seed * 977u + (uint32_t)cyc * 7919u);
    for (int i = NAGE - 1; i > 0; i--) {
        int j = (int)(h % (uint32_t)(i + 1));
        h = hash32(h);
        int t = ord[i]; ord[i] = ord[j]; ord[j] = t;
    }
    return ord[k];
}

/* ---- the journal: the machine's day in the keeper's hand ---- */
#define JMAX 2048
static const char *ordinal_word[32] = {"", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
    "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth",
    "thirtieth", "thirty-first"};
static const char *month_name[12] = {"January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"};
static const char *day_name[7] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
static const char *num_word[24] = {"no", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"};

static void cat(char *dst, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void cat(char *dst, const char *fmt, ...) {
    size_t n = strlen(dst);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst + n, JMAX - n, fmt, ap);
    va_end(ap);
}

static const char *pick(uint32_t *h, const char *const *opts, int n) {
    *h = hash32(*h + 0x9e3779b9u);
    return opts[*h % (uint32_t)n];
}

static void say_span(char *dst, double secs) {
    long m = (long)(secs / 60);
    if (m < 2) { cat(dst, "a minute or two"); return; }
    if (m < 60) { cat(dst, "%ld minutes", m); return; }
    long h = m / 60;
    m %= 60;
    if (h < 24) {
        cat(dst, "%s hour%s", num_word[h], h == 1 ? "" : "s");
        if (m >= 5) cat(dst, " and %ld minutes", m);
        return;
    }
    long d = h / 24;
    h %= 24;
    if (d <= 23) cat(dst, "%s day%s", num_word[d], d == 1 ? "" : "s");
    else cat(dst, "%ld days", d);
    if (h) cat(dst, " and %s hour%s", num_word[h], h == 1 ? "" : "s");
}

/* The heading (date) goes in head; the entry's paragraphs in body, split
 * by '\n'. wt: the wall-clock time it's written; period: which page. */
static void compose_journal(char *head, char *body, double wt, long period, int age) {
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    head[0] = body[0] = 0;
    float hour = opt_hour >= 0 ? opt_hour : tm.tm_hour + tm.tm_min / 60.0f;
    int hh = (int)hour, mm = (int)((hour - hh) * 60);
    cat(head, "%s, the %s of %s", day_name[tm.tm_wday], ordinal_word[tm.tm_mday], month_name[tm.tm_mon]);
    uint32_t h = hash32(opt_seed ^ (uint32_t)period * 2654435761u);
    char Host[64];
    snprintf(Host, sizeof Host, "%s", host);
    Host[0] = (char)toupper((unsigned char)Host[0]);

    /* the hour, the weather */
    cat(body, "%02d:%02d. ", hh, mm);
    if (hour < 5 || hour >= 22) {
        static const char *o[] = {"Late. The sea is loud below the cliff and the fire is low.",
                                  "Past midnight by the clock. Only the sea and the fire are awake.",
                                  "Night. Moonlight lies across the floor like a rug nobody laid."};
        cat(body, "%s", hour >= 22 ? o[hash32(h) % 2 == 0 ? 0 : 2] : pick(&h, o, 3));
    } else if (hour < 9) {
        static const char *o[] = {"Morning. Grey light on the water; the gulls are up.",
                                  "Early. A mist on the sea, lifting."};
        cat(body, "%s", pick(&h, o, 2));
    } else if (hour < 15) {
        static const char *o[] = {"The light has come round to the south windows.",
                                  "Midday. The dust stands still in the sun.",
                                  "A bright hour. Cloud shadows walk across the sea."};
        cat(body, "%s", pick(&h, o, 3));
    } else if (hour < 19) {
        static const char *o[] = {"Evening coming on. I have lit the lamp.",
                                  "The sun is low and gold on the shelves."};
        cat(body, "%s", pick(&h, o, 2));
    } else {
        static const char *o[] = {"Dark now. The lamp is lit and the fire made up.",
                                  "Evening. The windows have gone blue, then black."};
        cat(body, "%s", pick(&h, o, 2));
    }
    cat(body, "\n");

    /* the absent keeper */
    if (locked_at) {
        struct tm lt;
        localtime_r(&locked_at, &lt);
        double away = wt - (double)locked_at;
        cat(body, "%s left the desk at %02d:%02d and has not returned", user, lt.tm_hour, lt.tm_min);
        if (away > 600) { cat(body, " \xe2\x80\x94 "); say_span(body, away); cat(body, " now."); }
        else cat(body, "; the chair is still warm.");
    } else {
        static const char *o[] = {"%s was here a moment ago. The tea is still warm.",
                                  "%s has stepped out. The pen is still wet."};
        cat(body, pick(&h, o, 2), user);
    }
    cat(body, "\n");

    /* the engines */
    float t = sens.temp;
    if (t > 0) {
        if (t >= 75) cat(body, "The engines run hot below; I can feel them through the floor (%.0f\xc2\xb0).", t);
        else if (t >= 55) cat(body, "The engines run warm (%.0f\xc2\xb0).", t);
        else cat(body, "The engines are cool and quiet, %.0f\xc2\xb0.", t);
        cat(body, " ");
    }
    float l1 = sens.lavg[0], l15 = sens.lavg[2];
    if (l1 < 0.6f) cat(body, "Little work in hand: the load stands at %.2f", l1);
    else if (l1 < 3) cat(body, "Steady work below, the load at %.2f", l1);
    else cat(body, "Heavy labour below: the load stands at %.1f", l1);
    if (l1 > l15 * 1.3f + 0.1f) cat(body, " and rising.");
    else if (l1 < l15 * 0.7f - 0.1f) cat(body, " and falling.");
    else cat(body, ".");
    cat(body, "\n");

    /* the house */
    cat(body, "%s has kept its watch ", Host);
    say_span(body, sens.uptime);
    cat(body, " without sleep.\n");

    /* the book */
    {
        static const char *o[] = {"The book has opened itself at %s again.",
                                  "Turned to %s. The picture moves still; I watched it an hour.",
                                  "%s, the page says. I could almost step through."};
        cat(body, pick(&h, o, 3), age_name[age]);
    }
    static const char *close[] = {"Wind from the south-west. Tea gone cold.",
                                  "Something knocked twice in the cistern.",
                                  "Three ships on the horizon at noon; none since.",
                                  "The ladder has moved again.",
                                  "I have mended the clasp on the blue folio.",
                                  "The tide turned while I wrote this."};
    cat(body, " %s", pick(&h, close, 6));
}

/* ---- vectors and matrices (column-major) ---- */
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { return (V3){x, y, z}; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vmul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a) { float l = vlen(a); return l > 0 ? vmul(a, 1 / l) : a; }
static V3 vlerp(V3 a, V3 b, float t) { return vadd(a, vmul(vsub(b, a), t)); }

typedef struct { float m[16]; } M4;
static M4 m4_id(void) { M4 r = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}}; return r; }
static M4 m4_mul(M4 a, M4 b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static M4 m4_trans(float x, float y, float z) { M4 r = m4_id(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static M4 m4_roty(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
static M4 m4_rotx(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
static M4 m4_rotz(float a) { M4 r = m4_id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }
static V3 m4_pt(const M4 *a, V3 p) {
    return v3(a->m[0] * p.x + a->m[4] * p.y + a->m[8] * p.z + a->m[12],
              a->m[1] * p.x + a->m[5] * p.y + a->m[9] * p.z + a->m[13],
              a->m[2] * p.x + a->m[6] * p.y + a->m[10] * p.z + a->m[14]);
}
static V3 m4_dir(const M4 *a, V3 p) {
    return v3(a->m[0] * p.x + a->m[4] * p.y + a->m[8] * p.z,
              a->m[1] * p.x + a->m[5] * p.y + a->m[9] * p.z,
              a->m[2] * p.x + a->m[6] * p.y + a->m[10] * p.z);
}
/* a camera at eye looking along fwd */
static M4 m4_look(V3 eye, V3 fwd, V3 up) {
    V3 f = vnorm(fwd), s = vnorm(vcross(f, up)), u = vcross(s, f);
    M4 r = m4_id();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -vdot(s, eye); r.m[13] = -vdot(u, eye); r.m[14] = vdot(f, eye);
    return r;
}
static M4 m4_frustum(float l, float r, float b, float t, float n, float f) {
    M4 m = {{0}};
    m.m[0] = 2 * n / (r - l); m.m[5] = 2 * n / (t - b);
    m.m[8] = (r + l) / (r - l); m.m[9] = (t + b) / (t - b);
    m.m[10] = -(f + n) / (f - n); m.m[11] = -1;
    m.m[14] = -2 * f * n / (f - n);
    return m;
}
static M4 m4_ortho(float l, float r, float b, float t, float n, float f) {
    M4 m = m4_id();
    m.m[0] = 2 / (r - l); m.m[5] = 2 / (t - b); m.m[10] = -2 / (f - n);
    m.m[12] = -(r + l) / (r - l); m.m[13] = -(t + b) / (t - b); m.m[14] = -(f + n) / (f - n);
    return m;
}

/* ---- fonts: a hand for the journal, a book face for print ---- */
#define NCP 104
static const uint32_t extra_cp[NCP - 95] = {0xB0, 0x2014, 0x2019, 0x2018, 0x201C, 0x201D, 0x2013, 0xE9, 0xB7};
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
typedef struct { int face, px; Glyph g[NCP]; } Style;
#define MAXSTYLE 8
static Style styles[MAXSTYLE];
static int nstyles;
static FT_Library ft;
static FT_Face faces[2];               /* 0: handwriting, 1: print */
static char hand_family[128];
#define AW 2048
static unsigned char *atlas;
static int atlas_h;

static int cp_index(uint32_t cp) {
    if (cp >= 32 && cp < 127) return (int)cp - 32;
    for (int i = 0; i < NCP - 95; i++) if (extra_cp[i] == cp) return 95 + i;
    return 0;
}
static uint32_t cp_of(int i) { return i < 95 ? (uint32_t)(32 + i) : extra_cp[i - 95]; }

/* the next code point of a UTF-8 string */
static uint32_t utf8_next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    uint32_t c = *p++;
    if (c >= 0xF0) { c = (c & 7) << 18 | (p[0] & 63u) << 12 | (p[1] & 63u) << 6 | (p[2] & 63u); p += 3; }
    else if (c >= 0xE0) { c = (c & 15) << 12 | (p[0] & 63u) << 6 | (p[1] & 63u); p += 2; }
    else if (c >= 0xC0) { c = (c & 31) << 6 | (p[0] & 63u); p += 1; }
    *s = (const char *)p;
    return c;
}

/* load the first of these families fontconfig really has (not a stand-in;
 * the last is taken whatever it matches) */
static int load_face(const char *const *fams, int n, FT_Face *out, char *got, size_t gotn) {
    for (int i = 0; i < n; i++) {
        FcPattern *pat = FcNameParse((const FcChar8 *)fams[i]);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL, *fam = NULL;
        int ok = 0;
        if (m && FcPatternGetString(m, FC_FILE, 0, &file) == FcResultMatch &&
            FcPatternGetString(m, FC_FAMILY, 0, &fam) == FcResultMatch) {
            char want[128];
            snprintf(want, sizeof want, "%s", fams[i]);
            want[strcspn(want, ":")] = 0;
            if ((i == n - 1 || !strncasecmp((const char *)fam, want, strlen(want))) && !FT_New_Face(ft, (const char *)file, 0, out)) {
                ok = 1;
                if (got) snprintf(got, gotn, "%s", (const char *)fam);
            }
        }
        if (m) FcPatternDestroy(m);
        FcPatternDestroy(pat);
        if (ok) return 1;
    }
    return 0;
}

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    /* a hand if there is one, else a calligraphic italic, else any serif italic */
    static const char *hands[] = {"Caveat", "Kalam", "Homemade Apple", "Dancing Script", "Satisfy", "Parisienne",
                                  "Tangerine", "Z003", "URW Chancery L", "TeX Gyre Chorus", "EB Garamond:italic",
                                  "C059:italic", "serif:italic"};
    if (!load_face(hands, (int)(sizeof hands / sizeof *hands), &faces[0], hand_family, sizeof hand_family))
        die("no font for the handwriting");
    static const char *print[] = {"EB Garamond", "C059", "Gentium Book Plus", "DejaVu Serif", "serif"};
    if (!load_face(print, (int)(sizeof print / sizeof *print), &faces[1], NULL, 0)) die("no font for print");
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
        for (int c = 0; c < NCP; c++) {
            Glyph *gl = &styles[s].g[c];
            memset(gl, 0, sizeof *gl);
            FT_UInt gi = FT_Get_Char_Index(f, cp_of(c));
            if (!gi && c >= 95) {          /* no such glyph: a plain stand-in */
                static const char sub[] = "o--''\"\"-e.";
                gi = FT_Get_Char_Index(f, (FT_ULong)(unsigned char)sub[c - 95]);
            }
            if (FT_Load_Glyph(f, gi, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot g = f->glyph;
            int w = (int)g->bitmap.width, h = (int)g->bitmap.rows;
            if (x + w + 3 > AW) { x = 1; y += shelf + 3; shelf = 0; }
            if (y + h + 3 > cap) die("glyph atlas full");
            for (int yy = 0; yy < h; yy++)
                memcpy(atlas + (size_t)(y + yy) * AW + x, g->bitmap.buffer + yy * g->bitmap.pitch, (size_t)w);
            *gl = (Glyph){(short)x, (short)y, (short)w, (short)h, (short)g->bitmap_left, (short)g->bitmap_top,
                          (short)(g->advance.x >> 6)};
            x += w + 3;
            if (h > shelf) shelf = h;
        }
    }
    atlas_h = y + shelf + 3;
}

static const Glyph *glyph(int s, uint32_t cp) { return &styles[s].g[cp_index(cp)]; }

static float text_width(int s, const char *str) {
    float w = 0;
    for (const char *p = str; *p;) w += glyph(s, utf8_next(&p))->adv;
    return w;
}

/* text quads, in page pixels: x, y, u, v, t0 (when the pen reaches the
 * glyph), dur (how long it takes), fx (0..1 across it), ink (strength) */
#define TVF 8
static float *tv;
static int tvn, tvcap;

static void tvert(float x, float y, float u, float v, float t0, float dur, float fx, float ink) {
    if (tvn + TVF > tvcap) tv = realloc(tv, (size_t)(tvcap = tvcap ? tvcap * 2 : 16384) * sizeof *tv);
    float *o = tv + tvn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = t0; o[5] = dur; o[6] = fx; o[7] = ink;
    tvn += TVF;
}

/* Set str with its baseline at y from x, a little unsteadily (the hand
 * wanders off the line and back); t is when the pen starts, cps how fast it
 * writes. Returns the time the pen lifts. */
static float hand_line(int s, float x, float y, const char *str, float t, float cps, float slope, uint32_t seed, float ink) {
    float pen = x;
    int i = 0;
    for (const char *p = str; *p; i++) {
        uint32_t cp = utf8_next(&p);
        const Glyph *g = glyph(s, cp);
        float wob = sinf(i * 0.37f + (float)(seed % 97)) * styles[s].px * 0.03f;
        float yy = y + wob + (pen - x) * slope;
        float dur = fmaxf(g->adv, 1) / (cps * styles[s].px * 0.5f);
        if (g->w) {
            float x0 = pen + g->left, y0 = yy - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            float k = ink * (0.82f + 0.18f * ((hash32(seed + (uint32_t)i) & 255) / 255.0f));
            tvert(x0, y0, u0, v0, t, dur, 0, k); tvert(x1, y0, u1, v0, t, dur, 1, k); tvert(x1, y1, u1, v1, t, dur, 1, k);
            tvert(x0, y0, u0, v0, t, dur, 0, k); tvert(x1, y1, u1, v1, t, dur, 1, k); tvert(x0, y1, u0, v1, t, dur, 0, k);
        }
        pen += g->adv;
        t += dur;
        if (cp == ' ') t += 0.05f;
        if (cp == '.' || cp == ',' || cp == ';') t += 0.25f;
    }
    return t;
}

/* Wrap a paragraph to width w; returns the time the pen lifts. *y advances. */
static float hand_para(int s, float x, float *y, float w, float lead, const char *str, float t, float cps,
                       uint32_t seed, float ink, int indent) {
    char line[512];
    const char *p = str;
    float first = indent ? styles[s].px * 1.2f : 0;
    while (*p) {
        /* the longest run of words that fits */
        const char *q = p, *brk = NULL;
        int n = 0;
        for (;;) {
            const char *e = q;
            while (*e && *e != ' ') e++;
            int len = (int)(e - p);
            if (len >= 500) break;
            memcpy(line, p, (size_t)len);
            line[len] = 0;
            if (text_width(s, line) > w - first && brk) break;
            brk = e;
            n = len;
            if (!*e) break;
            q = e + 1;
        }
        if (!brk) break;
        memcpy(line, p, (size_t)n);
        line[n] = 0;
        seed = hash32(seed);
        float slope = ((seed & 1023) / 1023.0f - 0.5f) * 0.012f;
        t = hand_line(s, x + first, *y, line, t, cps, slope, seed, ink) + 0.35f;
        first = 0;
        *y += lead;
        p = brk;
        while (*p == ' ') p++;
    }
    return t;
}

/* ---- meshes: everything static goes into one vertex buffer ---- */
enum {
    MT_STONE, MT_FLOOR, MT_OAK, MT_SHELF, MT_BOOK, MT_BOOKTOP, MT_BRASS, MT_PAPER, MT_PAGE, MT_PAGE_R,
    MT_RUG, MT_IRON, MT_SOOT, MT_EMBER, MT_CERAMIC, MT_LAMPGLASS, MT_CLOTH, MT_TIMBER, MT_PLASTER, MT_SKY,
    MT_LEATHER, MT_TEA, MT_INK, MT_CANDLE, MT_WALLOUT,
};
#define VF 13                          /* pos3 nrm3 uv2 col3 mat var */
typedef struct { float *v; uint32_t *ix; int nv, ni, cv, ci; } Mesh;
static Mesh mesh;                      /* the room */
static M4 xf;                          /* the current transform for the builders */
static float cur_col[3] = {1, 1, 1}, cur_var;
static int cur_mat;
static Mesh *cur = &mesh;

static void col3(float r, float g, float b) { cur_col[0] = r; cur_col[1] = g; cur_col[2] = b; }
static void mat(int m, float r, float g, float b) { cur_mat = m; col3(r, g, b); cur_var = 0; }

static uint32_t vert(V3 p, V3 n, float u, float v) {
    Mesh *m = cur;
    if (m->nv == m->cv) m->v = realloc(m->v, (size_t)(m->cv = m->cv ? m->cv * 2 : 65536) * VF * sizeof(float));
    p = m4_pt(&xf, p);
    n = vnorm(m4_dir(&xf, n));
    if (vlen(n) < 0.5f) n = v3(0, 1, 0);
    float *o = m->v + (size_t)m->nv * VF;
    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z; o[6] = u; o[7] = v;
    o[8] = cur_col[0]; o[9] = cur_col[1]; o[10] = cur_col[2]; o[11] = (float)cur_mat; o[12] = cur_var;
    return (uint32_t)m->nv++;
}
static void tri(uint32_t a, uint32_t b, uint32_t c) {
    Mesh *m = cur;
    if (m->ni + 3 > m->ci) m->ix = realloc(m->ix, (size_t)(m->ci = m->ci ? m->ci * 2 : 196608) * sizeof(uint32_t));
    m->ix[m->ni++] = a; m->ix[m->ni++] = b; m->ix[m->ni++] = c;
}
/* a quad p0 p1 p2 p3 counter-clockwise seen from the front, uv per corner */
static void quad(V3 p0, V3 p1, V3 p2, V3 p3, float u0, float v0, float u1, float v1) {
    /* Newell's normal: sound even when two corners meet */
    V3 q[4] = {p0, p1, p2, p3}, n = v3(0, 0, 0);
    for (int i = 0; i < 4; i++) {
        V3 a = q[i], b = q[(i + 1) & 3];
        n.x += (a.y - b.y) * (a.z + b.z); n.y += (a.z - b.z) * (a.x + b.x); n.z += (a.x - b.x) * (a.y + b.y);
    }
    n = vnorm(n);
    if (vlen(n) < 0.5f) n = v3(0, 1, 0);
    uint32_t a = vert(p0, n, u0, v0), b = vert(p1, n, u1, v0), c = vert(p2, n, u1, v1), d = vert(p3, n, u0, v1);
    tri(a, b, c); tri(a, c, d);
}
/* a box centred at c with half-extents h; uv in metres per face. skip:
 * bitmask of faces to leave out (1 -x, 2 +x, 4 -y, 8 +y, 16 -z, 32 +z) */
static void box(V3 c, V3 h, int skip) {
    V3 p[8];
    for (int i = 0; i < 8; i++) p[i] = v3(c.x + (i & 1 ? h.x : -h.x), c.y + (i & 2 ? h.y : -h.y), c.z + (i & 4 ? h.z : -h.z));
    float X = 2 * h.x, Y = 2 * h.y, Z = 2 * h.z;
    if (!(skip & 1)) quad(p[0], p[4], p[6], p[2], 0, 0, Z, Y);
    if (!(skip & 2)) quad(p[5], p[1], p[3], p[7], 0, 0, Z, Y);
    if (!(skip & 4)) quad(p[0], p[1], p[5], p[4], 0, 0, X, Z);
    if (!(skip & 8)) quad(p[6], p[7], p[3], p[2], 0, 0, X, Z);
    if (!(skip & 16)) quad(p[1], p[0], p[2], p[3], 0, 0, X, Y);
    if (!(skip & 32)) quad(p[4], p[5], p[7], p[6], 0, 0, X, Y);
}
/* a surface of revolution about the local y axis: profile (r[i], y[i]) */
static void lathe(const float *r, const float *y, int n, int seg, float a0, float a1) {
    uint32_t base = (uint32_t)cur->nv;
    for (int i = 0; i < n; i++) {
        /* the profile's normal from its neighbours */
        int i0 = i > 0 ? i - 1 : i, i1 = i < n - 1 ? i + 1 : i;
        float dr = r[i1] - r[i0], dy = y[i1] - y[i0];
        float nr = dy, ny = -dr, l = sqrtf(nr * nr + ny * ny);
        if (l > 0) { nr /= l; ny /= l; }
        for (int s = 0; s <= seg; s++) {
            float a = a0 + (a1 - a0) * s / seg, ca = cosf(a), sa = sinf(a);
            vert(v3(r[i] * sa, y[i], r[i] * ca), v3(nr * sa, ny, nr * ca), (float)s / seg, y[i]);
        }
    }
    for (int i = 0; i < n - 1; i++)
        for (int s = 0; s < seg; s++) {
            uint32_t a = base + (uint32_t)(i * (seg + 1) + s), b = a + 1, c = a + (uint32_t)(seg + 1), d = c + 1;
            tri(a, b, d); tri(a, d, c);
        }
}
static void cylinder(float r, float y0, float y1, int seg, int caps) {
    float R[4] = {0, r, r, 0}, Y[4] = {y0, y0, y1, y1};
    if (caps) lathe(R, Y, 4, seg, 0, 2 * PI);
    else lathe(R + 1, Y + 1, 2, seg, 0, 2 * PI);
}
/* a cylinder of radius r from a to b (logs, rods, legs) */
static void rod(V3 a, V3 b, float r, int seg) {
    M4 save = xf;
    V3 d = vsub(b, a);
    float len = vlen(d);
    V3 y = vmul(d, 1 / len), x = vnorm(vcross(fabsf(y.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0), y)), z = vcross(x, y);
    M4 m = {{x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, z.x, z.y, z.z, 0, a.x, a.y, a.z, 1}};
    xf = m4_mul(save, m);
    cylinder(r, 0, len, seg, 1);
    xf = save;
}

/* Room coordinates: metres, y up, the floor at y = 0 and the room's axis at
 * x = z = 0. Azimuth a is measured from -z (straight ahead of the usual
 * camera) towards +x. */
static V3 at_az(float a, float r, float y) { return v3(r * sinf(a), y, -r * cosf(a)); }
static V3 inward(float a) { return v3(-sinf(a), 0, cosf(a)); }

/* part of a vertical cylinder, facing in (or out); uv: arc metres, height */
static void cyl_wall(float r, float a0, float a1, float y0, float y1, int facing_in, float du) {
    int seg = (int)ceilf((a1 - a0) * r / du);
    if (seg < 1) seg = 1;
    int rows = (int)ceilf((y1 - y0) / 0.5f);
    if (rows < 1) rows = 1;
    uint32_t base = (uint32_t)cur->nv;
    for (int j = 0; j <= rows; j++)
        for (int s = 0; s <= seg; s++) {
            float a = a0 + (a1 - a0) * s / seg, y = y0 + (y1 - y0) * j / rows;
            V3 n = facing_in ? inward(a) : vmul(inward(a), -1);
            vert(at_az(a, r, y), n, a * r, y);
        }
    for (int j = 0; j < rows; j++)
        for (int s = 0; s < seg; s++) {
            uint32_t a = base + (uint32_t)(j * (seg + 1) + s), b = a + 1, c = a + (uint32_t)(seg + 1), d = c + 1;
            if (facing_in) { tri(a, b, d); tri(a, d, c); } else { tri(a, d, b); tri(a, c, d); }
        }
}
/* a flat ring sector at height y, facing up (or down); uv: x, z in metres */
static void ring(float r0, float r1, float a0, float a1, float y, int up) {
    int seg = (int)ceilf((a1 - a0) * r1 / 0.25f);
    if (seg < 1) seg = 1;
    for (int s = 0; s < seg; s++) {
        float b0 = a0 + (a1 - a0) * s / seg, b1 = a0 + (a1 - a0) * (s + 1) / seg;
        V3 p0 = at_az(b0, r0, y), p1 = at_az(b1, r0, y), p2 = at_az(b1, r1, y), p3 = at_az(b0, r1, y);
        if (up) quad(p0, p1, p2, p3, b0 * r1, r0, b1 * r1, r1);
        else quad(p0, p3, p2, p1, b0 * r1, r0, b1 * r1, r1);
    }
}
/* a curved slab: a ring sector with thickness, and its front (inner) edge */
static void arc_slab(float r0, float r1, float a0, float a1, float y0, float y1) {
    ring(r0, r1, a0, a1, y1, 1);
    ring(r0, r1, a0, a1, y0, 0);
    cyl_wall(r0, a0, a1, y0, y1, 1, 0.25f);
}

/* ---- the room ---- */
#define RIN 4.6f                       /* the wall's inner radius */
#define RWT 0.85f                      /* its thickness */
#define HWALL 6.4f                     /* the eaves */
#define HAPEX 9.6f                     /* the roof's apex */
#define SHELF_D 0.40f                  /* shelf depth */
#define SHELF_B 0.14f                  /* the plinth */
#define SHELF_P 0.385f                 /* shelf pitch */
#define SHELF_N 10                     /* rows of books */
#define FIRE_AZ (-40 * DEG)
#define DOOR_AZ (180 * DEG)

/* the windows: azimuth, sill, springing of the arch, half-width inside
 * (splayed) and at the glass */
typedef struct { float a, y0, y1, wi, wo; } Opening;
#define NWIN 3
static Opening win[NWIN] = {
    {30 * DEG, 0.62f, 3.05f, 0.70f, 0.48f},
    {62 * DEG, 0.48f, 3.05f, 0.70f, 0.48f},
    {94 * DEG, 0.62f, 3.05f, 0.70f, 0.48f},
};
static int seat_win = 1;               /* the window with the seat */

/* the desk, the book on it */
static V3 desk_c = {0.05f, 0, -1.15f};
#define DESK_H 0.78f
#define DESK_HW 0.86f                  /* half width (x) */
#define DESK_HD 0.46f                  /* half depth (z) */
#define BOOK_TILT (13 * DEG)           /* the reading stand */
#define PAGE_W 0.30f
#define PAGE_H 0.42f
#define PAGE_T 0.022f                  /* each half's block of pages */
static M4 book_m;                      /* book space -> room: x across, y up off the page, z towards the reader */
static V3 lamp_pos, fire_pos, cup_pos;

/* a frame on the wall at azimuth a, radius r: x along the wall (towards
 * increasing azimuth), y up, z in towards the axis */
static M4 frame_at(float a, float r, float y) {
    V3 p = at_az(a, r, y);
    M4 m = {{cosf(a), 0, sinf(a), 0, 0, 1, 0, 0, -sinf(a), 0, cosf(a), 0, p.x, p.y, p.z, 1}};
    return m;
}

/* the curve round an opening, s across (metres), y up: u in 0..1 runs up
 * the left side, over the arch and down the right */
static void opening_pt(const Opening *o, float w, float u, float *s, float *y) {
    float side = o->y1 - o->y0, arc = PI * w, tot = 2 * side + arc, d = u * tot;
    if (d < side) { *s = -w; *y = o->y0 + d; }
    else if (d < side + arc) {
        float th = PI - (d - side) / w;
        *s = w * cosf(th); *y = o->y1 + w * sinf(th);
    } else { *s = w; *y = o->y1 + (tot - d); }
}

static float hue_rand(uint32_t *h) { *h = hash32(*h); return (*h & 0xffff) / 65535.0f; }

/* leather */
static void book_colour(uint32_t *h, float *c) {
    static const float pal[][3] = {
        {0.32f, 0.055f, 0.045f}, {0.40f, 0.10f, 0.06f}, {0.07f, 0.16f, 0.10f}, {0.06f, 0.09f, 0.18f},
        {0.42f, 0.25f, 0.12f}, {0.22f, 0.12f, 0.06f}, {0.05f, 0.045f, 0.04f}, {0.48f, 0.33f, 0.12f},
        {0.60f, 0.52f, 0.38f}, {0.16f, 0.20f, 0.17f}, {0.30f, 0.17f, 0.20f}, {0.14f, 0.08f, 0.05f},
    };
    int k = (int)(hue_rand(h) * 12) % 12;
    float b = 0.7f + 0.5f * hue_rand(h);
    for (int i = 0; i < 3; i++) c[i] = pal[k][i] * b;
}

/* one book standing in a frame: its spine on the frame's z = 0 plane facing
 * +z, its bottom at y = 0, centred on x = 0 */
static void book_box(float th, float h, float d, float style) {
    V3 c = v3(0, h / 2, -d / 2), e = v3(th / 2, h / 2, d / 2);
    V3 p[8];
    for (int i = 0; i < 8; i++) p[i] = v3(c.x + (i & 1 ? e.x : -e.x), c.y + (i & 2 ? e.y : -e.y), c.z + (i & 4 ? e.z : -e.z));
    float sv = cur_var;
    cur_mat = MT_BOOK;
    cur_var = floorf(style * 8) + clampf(h, 0.05f, 0.95f);
    quad(p[4], p[5], p[7], p[6], 0, 0, 1, 1);                 /* spine */
    cur_var = -1;
    quad(p[0], p[4], p[6], p[2], 0, 0, d, h);                 /* the sides */
    quad(p[5], p[1], p[3], p[7], 0, 0, d, h);
    cur_mat = MT_BOOKTOP;
    cur_var = th;
    quad(p[6], p[7], p[3], p[2], 0, 1, 1, 0);                 /* top: the page edges */
    cur_var = sv;
}

/* fill the arc from azimuth a0 to a1 at radius r (the spines' line), on a
 * shelf at height y with clearance hmax */
static void fill_books(float a0, float a1, float r, float y, float hmax, uint32_t *h) {
    float s = a0 * r, s1 = a1 * r;
    float col[3];
    int set_left = 0;
    float set_col[3] = {0}, set_h = 0, set_th = 0, set_style = 0;
    book_colour(h, col);
    while (s < s1 - 0.02f) {
        float u = hue_rand(h);
        /* now and then a gap, sometimes with a book leaning into it */
        if (u < 0.035f && s > a0 * r + 0.1f) {
            float g = 0.06f + 0.12f * hue_rand(h);
            if (s + g > s1) break;
            if (hue_rand(h) < 0.6f) {
                float th = 0.03f + 0.02f * hue_rand(h), bh = fminf(hmax - 0.04f, 0.2f + 0.1f * hue_rand(h));
                float d = 0.18f + 0.08f * hue_rand(h);
                float sn = clampf((g - th) / bh, 0, 0.6f), ang = asinf(sn);
                float az = (s + g) / r;
                book_colour(h, col);
                for (int i = 0; i < 3; i++) cur_col[i] = col[i];
                M4 save = xf;
                /* pivot on its bottom far corner, leaning back onto the last book */
                xf = m4_mul(m4_mul(frame_at(az, r + 0.01f, y), m4_rotz(ang)), m4_trans(-th / 2, 0, 0));
                book_box(th, bh, d, hue_rand(h));
                xf = save;
            }
            s += g;
            continue;
        }
        /* or a little stack lying flat */
        if (u < 0.05f && s1 - s > 0.36f) {
            int n = 2 + (int)(hue_rand(h) * 3);
            float bw = 0.22f + 0.08f * hue_rand(h), yy = y;
            float az = (s + bw / 2 + 0.01f) / r;
            for (int i = 0; i < n && yy < y + hmax - 0.08f; i++) {
                float th = 0.025f + 0.03f * hue_rand(h), d = 0.2f + 0.06f * hue_rand(h);
                book_colour(h, col);
                for (int k = 0; k < 3; k++) cur_col[k] = col[k];
                M4 save = xf;
                xf = m4_mul(m4_mul(frame_at(az + (hue_rand(h) - 0.5f) * 0.004f, r, yy + th / 2), m4_rotz(PI / 2)),
                            m4_trans(0, -bw / 2, 0));
                book_box(th, bw, d, 0);
                xf = save;
                yy += th;
            }
            s += bw + 0.03f;
            continue;
        }
        /* a matched set, or a single */
        if (set_left <= 0) {
            if (hue_rand(h) < 0.3f) {
                set_left = 3 + (int)(hue_rand(h) * 9);
                book_colour(h, set_col);
                set_h = fminf(hmax - 0.03f, 0.22f + 0.1f * hue_rand(h));
                set_th = 0.03f + 0.025f * hue_rand(h);
                set_style = hue_rand(h);
            } else set_left = 0;
        }
        float th, bh, st;
        if (set_left > 0) {
            set_left--;
            th = set_th * (0.92f + 0.16f * hue_rand(h));
            bh = set_h;
            st = set_style;
            for (int i = 0; i < 3; i++) col[i] = set_col[i] * (0.9f + 0.2f * hue_rand(h));
        } else {
            th = 0.02f + 0.05f * hue_rand(h) * hue_rand(h) + 0.01f;
            float tall = hue_rand(h);
            bh = fminf(hmax - 0.025f, 0.19f + 0.14f * tall * tall + 0.03f * hue_rand(h));
            st = hue_rand(h);
            book_colour(h, col);
        }
        if (s + th > s1) break;
        float d = 0.17f + 0.1f * hue_rand(h);
        for (int i = 0; i < 3; i++) cur_col[i] = col[i];
        M4 save = xf;
        xf = frame_at((s + th / 2) / r, r + 0.012f * hue_rand(h), y);
        book_box(th, bh, d, st);
        xf = save;
        s += th + 0.0015f;
    }
}

/* a run of shelving round the wall from azimuth a0 to a1 */
static void shelf_run(float a0, float a1, uint32_t seed) {
    uint32_t h = seed;
    float rf = RIN - SHELF_D, top = SHELF_B + SHELF_N * SHELF_P;
    float span_m = (a1 - a0) * rf;
    int nb = (int)lroundf(span_m / 0.92f);
    if (nb < 1) nb = 1;
    float ut = 0.05f / rf;                     /* an upright's half thickness, as an angle */
    xf = m4_id();
    /* back boards, plinth, cornice */
    mat(MT_SHELF, 0.30f, 0.17f, 0.08f);
    cyl_wall(RIN - 0.02f, a0, a1, 0, top + 0.12f, 1, 0.3f);
    mat(MT_SHELF, 0.36f, 0.21f, 0.10f);
    arc_slab(rf, RIN, a0, a1, 0, SHELF_B);
    arc_slab(rf - 0.06f, RIN, a0 - 0.01f, a1 + 0.01f, top, top + 0.07f);
    arc_slab(rf - 0.10f, RIN, a0 - 0.015f, a1 + 0.015f, top + 0.07f, top + 0.16f);
    arc_slab(rf - 0.04f, RIN, a0 - 0.008f, a1 + 0.008f, top + 0.16f, top + 0.22f);
    for (int k = 1; k < SHELF_N; k++) {
        float y = SHELF_B + k * SHELF_P;
        arc_slab(rf, RIN, a0, a1, y - 0.032f, y);
    }
    /* uprights */
    for (int b = 0; b <= nb; b++) {
        float a = a0 + (a1 - a0) * b / nb;
        xf = frame_at(a, RIN - SHELF_D / 2 - 0.01f, 0);
        mat(MT_SHELF, 0.38f, 0.22f, 0.11f);
        box(v3(0, (top + 0.05f) / 2, 0), v3(0.05f * (b == 0 || b == nb ? 1.4f : 1), (top + 0.05f) / 2, SHELF_D / 2 + 0.025f), 0);
        /* carved capital on the cornice */
        box(v3(0, top + 0.11f, 0.03f), v3(0.07f, 0.11f, SHELF_D / 2 + 0.04f), 4);
    }
    xf = m4_id();
    /* the books */
    for (int b = 0; b < nb; b++) {
        float ba = a0 + (a1 - a0) * b / nb + ut * 1.15f, bb = a0 + (a1 - a0) * (b + 1) / nb - ut * 1.15f;
        for (int k = 0; k < SHELF_N; k++) {
            float y = SHELF_B + k * SHELF_P;
            /* the odd shelf left half empty, or holding something else */
            h = hash32(h + (uint32_t)(b * 31 + k));
            float clear = SHELF_P - 0.035f;
            if ((h & 63) == 7) {
                fill_books(ba, ba + (bb - ba) * 0.45f, rf + 0.015f, y, clear, &h);
                continue;
            }
            fill_books(ba, bb, rf + 0.015f, y, clear, &h);
        }
    }
}

/* the window: its embrasure (jambs, soffit, sill), the glazing bars, and a
 * window seat in one of them */
static void window_build(int wi) {
    const Opening *o = &win[wi];
    int n = 40;
    xf = m4_id();
    mat(MT_STONE, 0.62f, 0.58f, 0.52f);
    cur_var = 2;                               /* embrasure stone: never cut */
    for (int i = 0; i < n; i++) {
        float s0, y0, s1, y1, t0, z0, t1, z1;
        opening_pt(o, o->wi, (float)i / n, &s0, &y0);
        opening_pt(o, o->wi, (float)(i + 1) / n, &s1, &y1);
        opening_pt(o, o->wo, (float)i / n, &t0, &z0);
        opening_pt(o, o->wo, (float)(i + 1) / n, &t1, &z1);
        float ro = RIN + RWT;
        V3 a = at_az(o->a + s0 / RIN, RIN, y0), b = at_az(o->a + s1 / RIN, RIN, y1);
        V3 c = at_az(o->a + t1 / ro, ro, z1), d = at_az(o->a + t0 / ro, ro, z0);
        V3 nn = vnorm(vcross(vsub(d, a), vsub(b, a)));
        uint32_t ia = vert(a, nn, (float)i / n * 6, 0), ib = vert(b, nn, (float)(i + 1) / n * 6, 0);
        uint32_t ic = vert(c, nn, (float)(i + 1) / n * 6, RWT), id = vert(d, nn, (float)i / n * 6, RWT);
        tri(ia, ib, ic); tri(ia, ic, id);
    }
    /* the sill */
    {
        float ro = RIN + RWT;
        V3 a = at_az(o->a - o->wi / RIN, RIN, o->y0), b = at_az(o->a + o->wi / RIN, RIN, o->y0);
        V3 c = at_az(o->a + o->wo / ro, ro, o->y0), d = at_az(o->a - o->wo / ro, ro, o->y0);
        quad(a, b, c, d, 0, 0, 1.4f, RWT);
        /* its nosing */
        xf = frame_at(o->a, RIN, o->y0);
        box(v3(0, -0.04f, 0.03f), v3(o->wi + 0.06f, 0.04f, 0.05f), 0);
    }
    /* the glazing: a stone mullion, iron transoms and lead cames */
    float zg = -(RWT - 0.16f);                 /* the glass plane in the frame */
    xf = frame_at(o->a, RIN, 0);
    float w = o->wo;
    mat(MT_STONE, 0.6f, 0.57f, 0.5f);
    cur_var = 2;
    box(v3(0, (o->y0 + o->y1 + w) / 2, zg), v3(0.045f, (o->y1 + w - o->y0) / 2, 0.09f), 0);
    mat(MT_IRON, 0.05f, 0.05f, 0.05f);
    for (int k = 1; k < 4; k++) {
        float y = o->y0 + (o->y1 - o->y0) * k / 4;
        box(v3(0, y, zg), v3(w, 0.014f, 0.02f), 0);
    }
    for (int side = -1; side <= 1; side += 2)
        for (int k = 1; k < 3; k++) {
            float s = side * w * k / 3;
            float top = o->y1 + sqrtf(fmaxf(w * w - s * s, 0));
            box(v3(s, (o->y0 + top) / 2, zg), v3(0.006f, (top - o->y0) / 2, 0.006f), 0);
        }
    for (float y = o->y0 + 0.2f; y < o->y1 + w - 0.05f; y += 0.2f) {
        float hw = y <= o->y1 ? w : sqrtf(fmaxf(w * w - (y - o->y1) * (y - o->y1), 0));
        box(v3(0, y, zg), v3(hw, 0.006f, 0.006f), 0);
    }
    /* the seat: a cushion and a book left open on it */
    if (wi == seat_win) {
        xf = frame_at(o->a, RIN, o->y0);
        mat(MT_CLOTH, 0.13f, 0.22f, 0.18f);
        box(v3(0, 0.05f, -0.3f), v3(o->wi * 0.82f, 0.05f, 0.3f), 0);
        box(v3(-o->wi * 0.52f, 0.2f, -0.42f), v3(0.18f, 0.14f, 0.07f), 0);
        mat(MT_CLOTH, 0.42f, 0.16f, 0.1f);
        box(v3(o->wi * 0.55f, 0.18f, -0.45f), v3(0.15f, 0.12f, 0.06f), 0);
        /* a small book, face down, tented */
        M4 b = m4_mul(xf, m4_mul(m4_trans(0.12f, 0.11f, -0.18f), m4_roty(0.5f)));
        mat(MT_LEATHER, 0.3f, 0.07f, 0.05f);
        for (int sd = -1; sd <= 1; sd += 2) {
            xf = m4_mul(b, m4_rotz(sd * 0.32f));
            box(v3(sd * 0.075f, 0, 0), v3(0.075f, 0.004f, 0.105f), 0);
        }
        xf = m4_id();
    }
}

/* the fireplace, built in a frame on the wall at FIRE_AZ */
static void fireplace_build(void) {
    M4 F = frame_at(FIRE_AZ, RIN, 0);
    xf = F;
    float hw = 1.15f, dep = 0.55f, ow = 0.62f, oh = 1.02f, back = 0.03f;
    mat(MT_STONE, 0.64f, 0.6f, 0.54f);
    cur_var = 3;                                               /* dressed stone */
    box(v3(-(hw + ow) / 2, oh / 2, (dep - 0.45f) / 2), v3((hw - ow) / 2, oh / 2, (dep + 0.45f) / 2), 0);
    box(v3((hw + ow) / 2, oh / 2, (dep - 0.45f) / 2), v3((hw - ow) / 2, oh / 2, (dep + 0.45f) / 2), 0);
    box(v3(0, (oh + HWALL) / 2, (dep - 0.5f) / 2), v3(hw, (HWALL - oh) / 2, (dep + 0.5f) / 2), 0);
    /* a moulded lintel and jambs proud of the breast */
    box(v3(0, oh + 0.11f, dep + 0.04f), v3(ow + 0.2f, 0.11f, 0.04f), 0);
    box(v3(-ow - 0.1f, oh / 2, dep + 0.03f), v3(0.1f, oh / 2, 0.03f), 0);
    box(v3(ow + 0.1f, oh / 2, dep + 0.03f), v3(0.1f, oh / 2, 0.03f), 0);
    /* the mantel shelf */
    mat(MT_OAK, 0.25f, 0.14f, 0.07f);
    box(v3(0, oh + 0.3f, dep + 0.08f), v3(hw + 0.12f, 0.05f, 0.14f), 0);
    /* the firebox: sooty back and splayed cheeks */
    mat(MT_SOOT, 0.05f, 0.045f, 0.04f);
    quad(v3(-ow * 0.75f, 0, back), v3(ow * 0.75f, 0, back), v3(ow * 0.75f, oh, back), v3(-ow * 0.75f, oh, back), 0, 0, 1.2f, oh);
    quad(v3(-ow, 0, dep), v3(-ow * 0.75f, 0, back), v3(-ow * 0.75f, oh, back), v3(-ow, oh, dep), 0, 0, 1, oh);
    quad(v3(ow * 0.75f, 0, back), v3(ow, 0, dep), v3(ow, oh, dep), v3(ow * 0.75f, oh, back), 0, 0, 1, oh);
    quad(v3(-ow, oh, dep), v3(-ow * 0.75f, oh, back), v3(ow * 0.75f, oh, back), v3(ow, oh, dep), 0, 0, 1, 1);
    /* the hearth */
    mat(MT_STONE, 0.5f, 0.47f, 0.43f);
    cur_var = 3;
    box(v3(0, 0.025f, (dep + 1.25f) / 2 - 0.1f), v3(hw + 0.15f, 0.025f, (1.25f - dep) / 2 + 0.1f), 0);
    mat(MT_SOOT, 0.07f, 0.06f, 0.05f);
    box(v3(0, 0.012f, (dep + back) / 2), v3(ow, 0.012f, (dep - back) / 2), 0);
    /* firedogs and a grate */
    mat(MT_IRON, 0.06f, 0.055f, 0.05f);
    for (int sd = -1; sd <= 1; sd += 2) {
        box(v3(sd * 0.32f, 0.12f, 0.02f), v3(0.018f, 0.1f, 0.3f), 0);
        box(v3(sd * 0.32f, 0.3f, 0.3f), v3(0.02f, 0.08f, 0.02f), 0);
        xf = m4_mul(F, m4_trans(sd * 0.32f, 0.4f, 0.3f));
        float r[] = {0, 0.035f, 0.03f, 0}, y[] = {0, 0.02f, 0.06f, 0.08f};
        lathe(r, y, 4, 10, 0, 2 * PI);
        xf = F;
    }
    for (int k = 0; k < 6; k++) box(v3(0, 0.1f, -0.12f + k * 0.055f), v3(0.3f, 0.008f, 0.008f), 0);
    /* logs, burning at their undersides */
    mat(MT_EMBER, 0.18f, 0.12f, 0.08f);
    rod(v3(-0.33f, 0.16f, -0.02f), v3(0.31f, 0.15f, 0.06f), 0.06f, 10);
    rod(v3(-0.26f, 0.15f, 0.16f), v3(0.28f, 0.16f, 0.12f), 0.055f, 10);
    rod(v3(-0.1f, 0.25f, 0.13f), v3(0.18f, 0.29f, -0.02f), 0.05f, 10);
    mat(MT_EMBER, 0.2f, 0.1f, 0.05f);
    cur_var = 1;                                               /* a bed of embers */
    box(v3(0, 0.11f, 0.05f), v3(0.28f, 0.015f, 0.17f), 0);
    fire_pos = m4_pt(&F, v3(0, 0.38f, 0.12f));
    /* a coal scuttle and a poker */
    mat(MT_BRASS, 0.55f, 0.36f, 0.16f);
    xf = m4_mul(F, m4_trans(hw + 0.3f, 0, 0.75f));
    {
        float r[] = {0, 0.12f, 0.14f, 0.15f, 0.14f, 0}, y[] = {0, 0, 0.05f, 0.2f, 0.3f, 0.3f};
        lathe(r, y, 5, 16, 0, 2 * PI);
    }
    mat(MT_IRON, 0.05f, 0.05f, 0.05f);
    xf = F;
    rod(v3(-hw - 0.2f, 0.02f, 0.75f), v3(-hw - 0.08f, 0.72f, 0.6f), 0.01f, 6);
    /* candlesticks on the mantel */
    for (int sd = -1; sd <= 1; sd += 2) {
        xf = m4_mul(F, m4_trans(sd * 0.85f, oh + 0.35f, dep + 0.1f));
        mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
        float r[] = {0, 0.06f, 0.055f, 0.016f, 0.014f, 0.03f, 0.025f, 0.012f, 0},
              y[] = {0, 0, 0.02f, 0.04f, 0.2f, 0.22f, 0.24f, 0.24f, 0.24f};
        lathe(r, y, 9, 14, 0, 2 * PI);
        mat(MT_CANDLE, 0.85f, 0.8f, 0.68f);
        cylinder(0.011f, 0.24f, 0.24f + 0.1f + 0.06f * (sd > 0), 10, 1);
    }
    /* a few books and a small brass-bound box */
    xf = m4_mul(F, m4_trans(-0.3f, oh + 0.35f, dep + 0.06f));
    uint32_t h = 991;
    for (int i = 0; i < 4; i++) {
        float c[3];
        book_colour(&h, c);
        col3(c[0], c[1], c[2]);
        M4 s = xf;
        xf = m4_mul(xf, m4_mul(m4_trans(i * 0.045f, 0, 0.1f), m4_roty(PI)));
        book_box(0.04f, 0.2f + 0.03f * (i & 1), 0.15f, 0.3f);
        xf = s;
    }
    xf = m4_mul(F, m4_trans(0.32f, oh + 0.35f, dep + 0.08f));
    mat(MT_OAK, 0.2f, 0.1f, 0.05f);
    box(v3(0, 0.05f, 0), v3(0.12f, 0.05f, 0.07f), 0);
    mat(MT_BRASS, 0.6f, 0.42f, 0.2f);
    box(v3(0, 0.1f, 0), v3(0.125f, 0.006f, 0.075f), 4);
    xf = m4_id();
}

/* a turned leg: bun foot, column, block at the top */
static void leg(V3 p, float h) {
    M4 s = xf;
    xf = m4_mul(s, m4_trans(p.x, p.y, p.z));
    float r[] = {0, 0.035f, 0.045f, 0.03f, 0.03f, 0.04f, 0.028f, 0.028f, 0.036f, 0.036f, 0},
          y[] = {0, 0, 0.04f, 0.09f, 0.3f, 0.34f, 0.38f, h - 0.16f, h - 0.14f, h, h};
    lathe(r, y, 11, 12, 0, 2 * PI);
    xf = s;
}

/* the reading desk, the chair pushed back from it, and what's on it */
static void desk_build(void) {
    M4 D = m4_trans(desk_c.x, 0, desk_c.z);
    xf = D;
    mat(MT_OAK, 0.32f, 0.18f, 0.09f);
    box(v3(0, DESK_H - 0.025f, 0), v3(DESK_HW, 0.025f, DESK_HD), 0);
    mat(MT_OAK, 0.28f, 0.15f, 0.07f);
    box(v3(0, DESK_H - 0.11f, 0), v3(DESK_HW - 0.06f, 0.06f, DESK_HD - 0.06f), 0);
    for (int i = 0; i < 4; i++)
        leg(v3((i & 1 ? 1 : -1) * (DESK_HW - 0.08f), 0, (i & 2 ? 1 : -1) * (DESK_HD - 0.08f)), DESK_H - 0.05f);
    /* stretchers */
    for (int sd = -1; sd <= 1; sd += 2) box(v3(sd * (DESK_HW - 0.08f), 0.12f, 0), v3(0.02f, 0.02f, DESK_HD - 0.1f), 0);
    box(v3(0, 0.12f, 0), v3(DESK_HW - 0.1f, 0.02f, 0.02f), 0);
    /* drawer fronts and their brass pulls, on the reader's side */
    for (int sd = -1; sd <= 1; sd += 2) {
        mat(MT_OAK, 0.3f, 0.16f, 0.08f);
        box(v3(sd * 0.42f, DESK_H - 0.11f, DESK_HD - 0.055f), v3(0.3f, 0.045f, 0.01f), 0);
        mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
        box(v3(sd * 0.42f, DESK_H - 0.11f, DESK_HD - 0.04f), v3(0.04f, 0.008f, 0.008f), 0);
    }

    /* the reading stand and the book on it */
    float top = DESK_H;
    M4 B = m4_mul(D, m4_mul(m4_trans(0, top + 0.065f, -0.02f), m4_rotx(BOOK_TILT)));
    book_m = B;
    xf = m4_mul(D, m4_trans(0, top, -0.02f));
    mat(MT_OAK, 0.22f, 0.12f, 0.06f);
    {
        /* a wedge: low at the reader's side */
        float zz = 0.24f, hh = 0.12f;
        V3 a = v3(-0.32f, 0, zz), b = v3(0.32f, 0, zz), c = v3(0.32f, 0, -zz), d = v3(-0.32f, 0, -zz);
        V3 e = v3(-0.32f, 0.01f, zz), f = v3(0.32f, 0.01f, zz), g = v3(0.32f, hh, -zz), k = v3(-0.32f, hh, -zz);
        quad(e, f, g, k, 0, 0, 0.64f, 0.5f);
        quad(a, b, f, e, 0, 0, 0.64f, 0.02f);
        quad(c, d, k, g, 0, 0, 0.64f, hh);
        quad(b, c, g, f, 0, 0, 0.5f, hh);
        quad(d, a, e, k, 0, 0, 0.5f, hh);
    }
    xf = B;
    /* the covers, lying open, and the spine's curve under the gutter */
    mat(MT_LEATHER, 0.28f, 0.09f, 0.05f);
    cur_var = 1;
    box(v3(-PAGE_W / 2 - 0.006f, -0.004f, 0), v3(PAGE_W / 2 + 0.012f, 0.004f, PAGE_H / 2 + 0.012f), 0);
    box(v3(PAGE_W / 2 + 0.006f, -0.004f, 0), v3(PAGE_W / 2 + 0.012f, 0.004f, PAGE_H / 2 + 0.012f), 0);
    /* the page blocks' edges (their tops are drawn with the page textures) */
    mat(MT_PAPER, 0.8f, 0.74f, 0.6f);
    cur_var = 1;
    for (int sd = -1; sd <= 1; sd += 2) {
        int n = 16;
        for (int i = 0; i < n; i++) {
            float x0 = (float)i / n * PAGE_W, x1 = (float)(i + 1) / n * PAGE_W;
            float h0 = PAGE_T * sinf(fminf(x0 / 0.07f, 1) * PI / 2), h1 = PAGE_T * sinf(fminf(x1 / 0.07f, 1) * PI / 2);
            for (int e = -1; e <= 1; e += 2) {
                float z = e * PAGE_H / 2;
                V3 a = v3(sd * x0, 0, z), b = v3(sd * x1, 0, z), c = v3(sd * x1, h1, z), d = v3(sd * x0, h0, z);
                if ((sd > 0) == (e > 0)) quad(a, b, c, d, 0, 0, 1, h1 / PAGE_T);
                else quad(b, a, d, c, 0, 0, 1, h1 / PAGE_T);
            }
        }
        float hx = PAGE_T * 0.96f;
        V3 a = v3(sd * PAGE_W, 0, -PAGE_H / 2), b = v3(sd * PAGE_W, 0, PAGE_H / 2), c = v3(sd * PAGE_W, hx, PAGE_H / 2),
           d = v3(sd * PAGE_W, hx, -PAGE_H / 2);
        if (sd > 0) quad(a, b, c, d, 0, 0, 1, 1);
        else quad(b, a, d, c, 0, 0, 1, 1);
    }
    /* a ribbon marker trailing from the gutter */
    mat(MT_CLOTH, 0.45f, 0.06f, 0.05f);
    xf = B;
    quad(v3(0.004f, 0.001f, PAGE_H / 2 - 0.01f), v3(0.016f, 0.001f, PAGE_H / 2 - 0.01f),
         v3(0.03f, -0.07f, PAGE_H / 2 + 0.08f), v3(0.018f, -0.07f, PAGE_H / 2 + 0.08f), 0, 0, 1, 1);

    /* the lamp: brass font and column, a glass chimney, an etched shade */
    V3 lp = v3(-0.58f, top, -0.2f);
    xf = m4_mul(D, m4_trans(lp.x, lp.y, lp.z));
    mat(MT_BRASS, 0.72f, 0.52f, 0.25f);
    {
        float r[] = {0, 0.09f, 0.09f, 0.075f, 0.022f, 0.02f, 0.03f, 0.03f, 0.075f, 0.08f, 0.06f, 0.03f, 0.03f, 0.04f, 0.04f, 0},
              y[] = {0, 0, 0.015f, 0.03f, 0.06f, 0.24f, 0.26f, 0.28f, 0.33f, 0.37f, 0.41f, 0.42f, 0.44f, 0.45f, 0.47f, 0.47f};
        lathe(r, y, 16, 24, 0, 2 * PI);
    }
    mat(MT_LAMPGLASS, 0.9f, 0.85f, 0.7f);
    cur_var = 0;                               /* the chimney */
    {
        float r[] = {0.022f, 0.03f, 0.028f, 0.016f, 0.016f}, y[] = {0.47f, 0.5f, 0.55f, 0.6f, 0.7f};
        lathe(r, y, 5, 16, 0, 2 * PI);
    }
    cur_var = 1;                               /* the shade, an open bell of cream glass */
    {
        float r[] = {0.045f, 0.075f, 0.105f, 0.118f, 0.112f, 0.09f, 0.06f, 0.05f},
              y[] = {0.47f, 0.49f, 0.53f, 0.58f, 0.63f, 0.68f, 0.71f, 0.725f};
        lathe(r, y, 8, 28, 0, 2 * PI);
    }
    mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
    {
        float r[] = {0.044f, 0.05f, 0.05f, 0.044f}, y[] = {0.462f, 0.462f, 0.475f, 0.475f};
        lathe(r, y, 4, 24, 0, 2 * PI);
    }
    lamp_pos = m4_pt(&D, v3(lp.x, lp.y + 0.62f, lp.z));

    /* tea, going cold */
    V3 cp = v3(0.6f, top, 0.16f);
    xf = m4_mul(D, m4_trans(cp.x, cp.y, cp.z));
    mat(MT_CERAMIC, 0.85f, 0.83f, 0.78f);
    {
        float r[] = {0, 0.07f, 0.075f, 0.07f, 0.03f, 0.028f, 0}, y[] = {0, 0, 0.004f, 0.008f, 0.012f, 0.004f, 0.004f};
        lathe(r, y, 7, 24, 0, 2 * PI);
        float r2[] = {0, 0.022f, 0.025f, 0.038f, 0.044f, 0.046f, 0.043f, 0.04f, 0.036f, 0.02f, 0},
              y2[] = {0.004f, 0.004f, 0.01f, 0.03f, 0.06f, 0.075f, 0.075f, 0.06f, 0.02f, 0.016f, 0.016f};
        lathe(r2, y2, 11, 24, 0, 2 * PI);
    }
    /* its handle */
    for (int i = 0; i < 6; i++) {
        float a0 = -1.2f + i * 0.45f, a1 = a0 + 0.45f;
        rod(v3(0.045f + 0.018f * cosf(a0), 0.045f + 0.018f * sinf(a0), 0), v3(0.045f + 0.018f * cosf(a1), 0.045f + 0.018f * sinf(a1), 0), 0.004f, 6);
    }
    mat(MT_TEA, 0.2f, 0.09f, 0.03f);
    {
        float r[] = {0, 0.04f}, y[] = {0.062f, 0.062f};
        lathe(r, y, 2, 24, 0, 2 * PI);
    }
    cup_pos = m4_pt(&D, v3(cp.x, cp.y + 0.07f, cp.z));

    /* an inkwell, and the pen laid across a sheet of notes */
    xf = m4_mul(D, m4_trans(0.5f, top, -0.26f));
    mat(MT_INK, 0.03f, 0.04f, 0.06f);
    {
        float r[] = {0, 0.04f, 0.042f, 0.04f, 0.02f, 0.018f, 0.022f, 0.022f, 0},
              y[] = {0, 0, 0.01f, 0.05f, 0.06f, 0.07f, 0.072f, 0.078f, 0.078f};
        lathe(r, y, 9, 18, 0, 2 * PI);
    }
    mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
    {
        float r[] = {0.025f, 0.025f}, y[] = {0.071f, 0.079f};
        lathe(r, y, 2, 18, 0, 2 * PI);
    }
    /* loose notes, slightly askew */
    xf = m4_mul(D, m4_mul(m4_trans(0.5f, top + 0.002f, 0.02f), m4_roty(0.22f)));
    mat(MT_PAGE, 0.86f, 0.8f, 0.66f);
    cur_var = 4;                               /* the filler page texture */
    quad(v3(-0.105f, 0, 0.148f), v3(0.105f, 0, 0.148f), v3(0.105f, 0, -0.148f), v3(-0.105f, 0, -0.148f), 0, 1, 1, 0);
    xf = m4_mul(D, m4_mul(m4_trans(0.47f, top + 0.004f, 0.05f), m4_roty(-0.15f)));
    cur_var = 5;
    quad(v3(-0.105f, 0, 0.148f), v3(0.105f, 0, 0.148f), v3(0.105f, 0, -0.148f), v3(-0.105f, 0, -0.148f), 0, 1, 1, 0);
    /* the pen: a dark barrel and a brass nib */
    xf = D;
    mat(MT_INK, 0.05f, 0.03f, 0.02f);
    rod(v3(0.36f, top + 0.012f, 0.1f), v3(0.52f, top + 0.012f, -0.04f), 0.0055f, 8);
    mat(MT_BRASS, 0.75f, 0.55f, 0.28f);
    rod(v3(0.34f, top + 0.012f, 0.118f), v3(0.36f, top + 0.012f, 0.1f), 0.004f, 8);

    /* a stack of books at the left front */
    uint32_t h = 4242 + opt_seed;
    float y = top;
    for (int i = 0; i < 3; i++) {
        float c[3], th = 0.04f + 0.02f * hue_rand(&h), bw = 0.26f - i * 0.025f;
        book_colour(&h, c);
        col3(c[0], c[1], c[2]);
        xf = m4_mul(D, m4_mul(m4_mul(m4_trans(-0.6f, y + th / 2, 0.22f), m4_roty(0.3f * (hue_rand(&h) - 0.5f) + 0.1f)),
                              m4_mul(m4_rotz(PI / 2), m4_trans(0, -bw / 2, 0.09f))));
        book_box(th, bw, 0.18f, 0.5f);
        y += th;
    }

    /* the chair, pushed back and turned as someone got up */
    M4 C = m4_mul(m4_trans(desk_c.x - 0.28f, 0, desk_c.z + 0.98f), m4_roty(0.42f));
    xf = C;
    mat(MT_OAK, 0.3f, 0.17f, 0.085f);
    box(v3(0, 0.46f, 0), v3(0.23f, 0.022f, 0.22f), 0);
    for (int i = 0; i < 4; i++) {
        float lx = (i & 1 ? 1 : -1) * 0.19f, lz = (i & 2 ? 1 : -1) * 0.18f;
        box(v3(lx, 0.22f, lz), v3(0.018f, 0.22f, 0.018f), 0);
    }
    /* the back: two stiles, a crest rail, spindles (towards the desk, -z) */
    for (int sd = -1; sd <= 1; sd += 2) box(v3(sd * 0.19f, 0.74f, 0.19f), v3(0.02f, 0.3f, 0.02f), 0);
    box(v3(0, 0.98f, 0.19f), v3(0.22f, 0.05f, 0.02f), 0);
    for (int k = -2; k <= 2; k++) rod(v3(k * 0.06f, 0.48f, 0.19f), v3(k * 0.06f, 0.93f, 0.19f), 0.009f, 8);
    mat(MT_CLOTH, 0.32f, 0.08f, 0.06f);
    box(v3(0, 0.495f, 0.01f), v3(0.2f, 0.014f, 0.19f), 0);
    xf = m4_id();
}

/* the roof: a cone of boards on radial rafters, a ring beam, and an iron
 * chandelier hanging from the boss */
static void roof_build(void) {
    xf = m4_id();
    mat(MT_TIMBER, 0.3f, 0.2f, 0.12f);
    {
        int n = 12;
        float r[13], y[13];
        for (int i = 0; i <= n; i++) {
            float t = (float)i / n;
            r[i] = (RIN + 0.1f) * t;
            y[i] = HAPEX - (HAPEX - HWALL) * t;
        }
        lathe(r, y, n + 1, 96, 0, 2 * PI);
    }
    mat(MT_TIMBER, 0.24f, 0.15f, 0.08f);
    float slope = atan2f(HAPEX - HWALL, RIN);
    for (int k = 0; k < 16; k++) {
        float a = k * 2 * PI / 16;
        xf = m4_mul(m4_roty(-a), m4_mul(m4_trans(0, HAPEX - 0.06f, 0), m4_rotx(-slope)));
        box(v3(0, -0.1f, -RIN / cosf(slope) / 2), v3(0.07f, 0.1f, RIN / cosf(slope) / 2), 0);
    }
    xf = m4_id();
    arc_slab(RIN - 0.25f, RIN, 0, 2 * PI, HWALL - 0.3f, HWALL);
    mat(MT_STONE, 0.6f, 0.56f, 0.5f);
    cur_var = 3;
    arc_slab(RIN - 0.12f, RIN, 0, 2 * PI, HWALL - 0.42f, HWALL - 0.3f);
    /* the boss */
    xf = m4_trans(0, HAPEX - 0.25f, 0);
    mat(MT_TIMBER, 0.22f, 0.13f, 0.07f);
    {
        float r[] = {0, 0.2f, 0.24f, 0.2f, 0}, y[] = {-0.2f, -0.12f, 0, 0.1f, 0.1f};
        lathe(r, y, 5, 16, 0, 2 * PI);
    }
    /* the chandelier: an iron hoop on three chains, with candles */
    float cy = 4.55f, cr = 0.75f;
    xf = m4_id();
    mat(MT_IRON, 0.05f, 0.05f, 0.05f);
    for (int k = 0; k < 3; k++) {
        float a = k * 2 * PI / 3;
        rod(v3(cr * sinf(a), cy, cr * cosf(a)), v3(0, HAPEX - 0.5f, 0), 0.008f, 5);
    }
    for (int k = 0; k < 24; k++) {
        float a0 = k * 2 * PI / 24, a1 = (k + 1) * 2 * PI / 24;
        rod(v3(cr * sinf(a0), cy, cr * cosf(a0)), v3(cr * sinf(a1), cy, cr * cosf(a1)), 0.022f, 6);
    }
    for (int k = 0; k < 8; k++) {
        float a = (k + 0.5f) * 2 * PI / 8;
        xf = m4_trans(cr * sinf(a), cy + 0.02f, cr * cosf(a));
        mat(MT_IRON, 0.05f, 0.05f, 0.05f);
        float r[] = {0, 0.04f, 0.035f, 0.012f, 0}, y[] = {0, 0, 0.03f, 0.03f, 0.03f};
        lathe(r, y, 5, 8, 0, 2 * PI);
        mat(MT_CANDLE, 0.85f, 0.8f, 0.68f);
        cylinder(0.012f, 0.03f, 0.1f + 0.05f * (k % 3), 8, 1);
    }
    xf = m4_id();
}

/* a stout door at the back, in a shallow arched recess */
static void door_build(void) {
    M4 F = frame_at(DOOR_AZ, RIN, 0);
    xf = F;
    mat(MT_OAK, 0.2f, 0.11f, 0.05f);
    for (int k = -4; k <= 4; k++) box(v3(k * 0.11f, 1.1f, 0.03f), v3(0.054f, 1.1f, 0.03f), 0);
    mat(MT_IRON, 0.04f, 0.04f, 0.04f);
    for (int k = 0; k < 3; k++) box(v3(-0.1f, 0.35f + k * 0.75f, 0.07f), v3(0.4f, 0.025f, 0.01f), 0);
    xf = m4_id();
}

/* a brass telescope on a tripod by the third window, looking out to sea */
static void telescope_build(void) {
    const Opening *o = &win[2];
    M4 F = m4_mul(frame_at(o->a, RIN - 1.05f, 0), m4_roty(PI));
    xf = F;
    mat(MT_OAK, 0.25f, 0.14f, 0.07f);
    for (int k = 0; k < 3; k++) {
        float a = k * 2 * PI / 3 + 0.3f;
        rod(v3(0.38f * sinf(a), 0, 0.38f * cosf(a)), v3(0, 1.12f, 0), 0.014f, 6);
    }
    mat(MT_BRASS, 0.75f, 0.55f, 0.28f);
    xf = m4_mul(F, m4_mul(m4_trans(0, 1.16f, 0), m4_rotx(PI / 2 - 0.12f)));
    {
        float r[] = {0, 0.028f, 0.028f, 0.034f, 0.034f, 0.04f, 0.04f, 0.046f, 0.046f, 0.04f},
              y[] = {-0.35f, -0.35f, -0.1f, -0.1f, 0.15f, 0.15f, 0.4f, 0.42f, 0.5f, 0.5f};
        lathe(r, y, 10, 16, 0, 2 * PI);
    }
    xf = m4_id();
}

static void room_build(void) {
    xf = m4_id();
    cur = &mesh;
    /* the walls (the windows are cut by the shader) */
    mat(MT_STONE, 0.6f, 0.56f, 0.5f);
    cur_var = 1;
    cyl_wall(RIN, 0, 2 * PI, 0, HWALL, 1, 0.2f);
    for (int i = 0; i < NWIN; i++) window_build(i);
    xf = m4_id();
    /* the floor: stone flags */
    mat(MT_FLOOR, 0.5f, 0.47f, 0.42f);
    for (int k = 0; k < 10; k++) ring(RIN * k / 10, RIN * (k + 1) / 10 + 0.001f, 0, 2 * PI, 0, 1);
    /* a round rug under the desk */
    xf = m4_trans(desk_c.x, 0.006f, desk_c.z + 0.4f);
    mat(MT_RUG, 0.4f, 0.12f, 0.08f);
    {
        int seg = 64;
        float rr = 1.75f;
        uint32_t c = vert(v3(0, 0, 0), v3(0, 1, 0), 0, 0);
        for (int s = 0; s < seg; s++) {
            float a0 = s * 2 * PI / seg, a1 = (s + 1) * 2 * PI / seg;
            uint32_t p = vert(v3(rr * sinf(a0), 0, rr * cosf(a0)), v3(0, 1, 0), rr * sinf(a0), rr * cosf(a0));
            uint32_t q = vert(v3(rr * sinf(a1), 0, rr * cosf(a1)), v3(0, 1, 0), rr * sinf(a1), rr * cosf(a1));
            tri(c, p, q);
        }
    }
    xf = m4_id();
    /* shelving between the windows, the fireplace and the door */
    shelf_run(-22.5f * DEG, 19.4f * DEG, 1001 + opt_seed * 7);
    shelf_run(104.6f * DEG, 172 * DEG, 2002 + opt_seed * 7);
    shelf_run(-172 * DEG, -57.5f * DEG, 3003 + opt_seed * 7);
    /* narrow presses between the windows */
    shelf_run(39.6f * DEG, 52.4f * DEG, 4004 + opt_seed * 7);
    shelf_run(71.6f * DEG, 84.4f * DEG, 5005 + opt_seed * 7);
    /* a brass rail along the shelves behind the desk, for the ladder */
    mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
    for (int k = 0; k < 40; k++) {
        float a0 = (-22.5f + 41.9f * k / 40) * DEG, a1 = (-22.5f + 41.9f * (k + 1) / 40) * DEG;
        rod(at_az(a0, RIN - SHELF_D - 0.07f, 3.75f), at_az(a1, RIN - SHELF_D - 0.07f, 3.75f), 0.014f, 6);
    }
    /* the library ladder, hooked on the rail */
    {
        float a = -11 * DEG;
        M4 F = frame_at(a, RIN - SHELF_D - 0.07f, 0);
        xf = F;
        mat(MT_OAK, 0.3f, 0.17f, 0.09f);
        float foot = 1.15f;
        for (int sd = -1; sd <= 1; sd += 2) rod(v3(sd * 0.22f, 0, foot), v3(sd * 0.2f, 3.8f, 0), 0.025f, 6);
        for (int k = 1; k < 13; k++) {
            float t = k / 13.0f;
            rod(v3(-0.21f, 3.8f * t, foot * (1 - t)), v3(0.21f, 3.8f * t, foot * (1 - t)), 0.016f, 6);
        }
        mat(MT_BRASS, 0.7f, 0.5f, 0.24f);
        for (int sd = -1; sd <= 1; sd += 2) box(v3(sd * 0.2f, 3.78f, 0.0f), v3(0.012f, 0.04f, 0.03f), 0);
        xf = m4_id();
    }
    fireplace_build();
    desk_build();
    roof_build();
    door_build();
    telescope_build();
    /* the sky, far outside */
    xf = m4_id();
    mat(MT_SKY, 1, 1, 1);
    {
        int n = 16;
        float r[17], y[17];
        for (int i = 0; i <= n; i++) {
            float th = PI * i / n;
            r[i] = 60 * sinf(th);
            y[i] = -60 * cosf(th);
        }
        lathe(r, y, n + 1, 32, 0, 2 * PI);
    }
}

/* the open pages: each a grid over its block of paper, rising from the
 * gutter; u across the page (outer edge 0 on the left page), v down it */
static float page_h(float x) { return PAGE_T * sinf(fminf(x / 0.07f, 1) * PI / 2) - 0.0025f * (x / PAGE_W) * (x / PAGE_W); }

static void pages_build(void) {
    xf = book_m;
    for (int sd = -1; sd <= 1; sd += 2) {
        mat(MT_PAGE, 1, 1, 1);
        cur_var = sd < 0 ? 0 : 1 + 8;      /* the journal; the plate with the live image */
        int nx = 36, nz = 6;
        uint32_t base = (uint32_t)cur->nv;
        for (int j = 0; j <= nz; j++)
            for (int i = 0; i <= nx; i++) {
                float x = (float)i / nx * PAGE_W, z = -PAGE_H / 2 + PAGE_H * j / nz;
                float y = page_h(x) + 0.0004f, dy = (page_h(x + 0.001f) - page_h(x)) / 0.001f;
                V3 n = vnorm(v3(-sd * dy, 1, 0));
                float u = sd < 0 ? 1 - (float)i / nx : (float)i / nx;
                vert(v3(sd * x, y, z), n, u, (float)j / nz);
            }
        for (int j = 0; j < nz; j++)
            for (int i = 0; i < nx; i++) {
                uint32_t a = base + (uint32_t)(j * (nx + 1) + i), b = a + 1, c = a + (uint32_t)(nx + 1), d = c + 1;
                if (sd > 0) { tri(a, d, b); tri(a, c, d); } else { tri(a, b, d); tri(a, d, c); }
            }
    }
    xf = m4_id();
}

/* ---- shaders ---- */
#define GLSL(...) #__VA_ARGS__

static const char *VS_SCENE = "#version 330 core\n" GLSL(
layout(location = 0) in vec3 aP;
layout(location = 1) in vec3 aN;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec3 aC;
layout(location = 4) in vec2 aM;
uniform mat4 uVP;
out vec3 vP; out vec3 vN; out vec2 vUV; out vec3 vC;
flat out int vMat; flat out float vVar;
invariant gl_Position;
void main() {
    vP = aP; vN = aN; vUV = aUV; vC = aC;
    vMat = int(aM.x + 0.5); vVar = aM.y;
    gl_Position = uVP * vec4(aP, 1.0);
}
);

/* the window and fireplace cut-outs, shared by the shadow and scene passes */
#define CUTS GLSL( \
uniform vec4 uWin[3];   /* azimuth, sill, springing, half-width */ \
const float RIN = 4.6; \
bool in_window(vec3 p) { \
    float a = atan(p.x, -p.z); \
    for (int i = 0; i < 3; i++) { \
        float da = a - uWin[i].x; \
        da = mod(da + 3.14159265, 6.2831853) - 3.14159265; \
        float s = da * RIN, w = uWin[i].w; \
        if (abs(s) < w && p.y > uWin[i].y && (p.y < uWin[i].z || s * s + (p.y - uWin[i].z) * (p.y - uWin[i].z) < w * w)) return true; \
    } \
    return false; \
} \
)

static const char *FS_SHADOW = "#version 330 core\n" CUTS GLSL(
in vec3 vP; in vec3 vN; in vec2 vUV; in vec3 vC;
flat in int vMat; flat in float vVar;
uniform int uSkip;      /* a material this light ignores (its own lamp glass, its fire) */
void main() {
    if (vMat == 19 || vMat == uSkip) discard;
    if (vMat == 0 && vVar > 0.5 && vVar < 1.5 && in_window(vP)) discard;
}
);

/* noise from a tiling texture: r, g, b smooth at 16, 32, 64 cells a tile */
#define NOISE GLSL( \
uniform sampler2D uNoise; \
float n1(vec2 p) { return texture(uNoise, p).r; } \
float n2(vec2 p) { return texture(uNoise, p).g; } \
float n3(vec2 p) { return texture(uNoise, p).b; } \
float fbm(vec2 p) { return 0.5 * n1(p) + 0.25 * n1(p * 2.03 + 0.17) + 0.125 * n1(p * 4.11 + 0.41) + 0.0625 * n1(p * 8.3 + 0.7); } \
float h12(vec2 p) { p = fract(p * vec2(443.897, 441.423)); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); } \
float h13(vec3 p) { p = fract(p * vec3(443.897, 441.423, 437.195)); p += dot(p, p.yzx + 19.19); return fract((p.x + p.y) * p.z); } \
)

static const char *FS_SCENE = "#version 330 core\n" CUTS NOISE GLSL(
in vec3 vP; in vec3 vN; in vec2 vUV; in vec3 vC;
flat in int vMat; flat in float vVar;
out vec4 oCol;
uniform vec3 uCam;
uniform float uTime;
uniform vec3 uKeyDir, uKeyCol;          /* the sun or the moon, as it lights the room */
uniform mat4 uKeyVP;
uniform sampler2DShadow uKeySh;
uniform vec3 uFirePos, uFireCol, uLampPos, uLampCol;
uniform samplerCubeShadow uFireSh, uLampSh;
uniform sampler3D uAO;
uniform vec3 uSkyAmb, uGndAmb, uWinL, uBounce;
uniform vec4 uWinC[3];                  /* the windows' centres (xyz) and areas */
/* the outside */
uniform vec3 uSunDir, uMoonDir, uSkyZen, uSkyHor, uSunCol;
uniform float uNight, uMoonPhase, uStorm;
/* the book */
uniform sampler2D uPage0, uPage1, uPage2, uPage3, uPage4, uPage5, uAgeLive, uAgeOld;
uniform vec4 uPanel;                    /* the plate on the right page: u0 v0 u1 v1 */
uniform vec2 uAgeScale;                 /* how much of the age textures holds the picture */
uniform float uPanelGlow, uLampOn;
uniform int uDebug;

const float PI = 3.14159265;

struct S { vec3 alb; float rough, metal; vec3 emis; vec3 n; float spec, wrap; };

float D_ggx(float nh, float a) { float a2 = a * a, d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
vec3 brdf(S s, vec3 L, vec3 V, vec3 rad) {
    vec3 n = s.n;
    float nl = dot(n, L);
    float diff = max((nl + s.wrap) / (1.0 + s.wrap), 0.0);
    if (diff <= 0.0) return vec3(0);
    nl = max(nl, 0.0);
    vec3 H = normalize(L + V);
    float nh = max(dot(n, H), 0.0), nv = max(dot(n, V), 1e-3), vh = max(dot(V, H), 0.0);
    float a = max(s.rough * s.rough, 0.002);
    vec3 F0 = mix(vec3(s.spec), s.alb, s.metal);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
    float k = a * 0.5;
    float G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
    vec3 sp = D_ggx(nh, a) * F * G / max(4.0 * nl * nv, 1e-3);
    return rad * (s.alb * (1.0 - s.metal) * diff / PI * (1.0 - F0.r) + sp * nl);
}

float cube_ref(vec3 v) {
    float z = max(abs(v.x), max(abs(v.y), abs(v.z)));
    float n = 0.04, f = 14.0;
    return ((f + n) / (f - n) - 2.0 * f * n / ((f - n) * z)) * 0.5 + 0.5;
}
float key_shadow(vec3 p, vec3 n) {
    vec4 q = uKeyVP * vec4(p + n * 0.012 + uKeyDir * 0.004, 1.0);
    vec3 c = q.xyz / q.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) return 1.0;
    float s = 0.0;
    vec2 t = vec2(1.0 / 2048.0);
    s += texture(uKeySh, vec3(c.xy + t * vec2(-0.6, -0.3), c.z));
    s += texture(uKeySh, vec3(c.xy + t * vec2(0.6, 0.3), c.z));
    return s * 0.5;
}
float pt_shadow1(samplerCubeShadow sh, vec3 lp, vec3 p, vec3 n) {
    vec3 v = p + n * 0.01 - lp;
    return texture(sh, vec4(v, cube_ref(v) - 0.0015 - 0.002 * length(v)));
}
float pt_shadow(samplerCubeShadow sh, vec3 lp, vec3 p, vec3 n) {
    vec3 v = p + n * 0.01 - lp;
    float d = length(v), bias = 0.003 + 0.003 * d;
    vec3 t1 = normalize(cross(v, abs(v.y) < 0.9 * d ? vec3(0, 1, 0) : vec3(1, 0, 0))), t2 = normalize(cross(v, t1));
    float r = 0.01 * d, s = 0.0, ref = cube_ref(v) - bias;
    s += texture(sh, vec4(v + (t1 + t2 * 0.5) * r, ref));
    s += texture(sh, vec4(v - (t1 + t2 * 0.5) * r, ref));
    return s * 0.5;
}

/* the world outside the windows */
vec3 outside(vec3 d) {
    float el = d.y;
    vec3 sky = mix(uSkyHor, uSkyZen, pow(clamp(el, 0.0, 1.0), 0.55));
    float sd = max(dot(d, uSunDir), 0.0);
    sky += uSunCol * (0.05 * pow(sd, 6.0) + 0.25 * pow(sd, 60.0));
    vec3 col;
    if (el > -0.002) {
        /* stars and the moon by night */
        if (uNight > 0.05) {
            vec3 q = d * 300.0;
            vec3 c = floor(q);
            float h = h13(c);
            float st = step(0.985, h) * smoothstep(0.5, 0.0, length(fract(q) - 0.5)) * (h - 0.985) * 60.0;
            sky += vec3(0.8, 0.85, 1.0) * st * uNight * smoothstep(0.0, 0.15, el);
            float md = dot(d, uMoonDir);
            float disc = smoothstep(0.99975, 0.9999, md);
            sky += vec3(0.9, 0.92, 1.0) * (disc * (0.25 + 1.5 * uMoonPhase) + 0.02 * uMoonPhase * pow(max(md, 0.0), 300.0)) * uNight;
        }
        /* a deck of drifting cloud */
        vec2 cp = d.xz / max(el + 0.06, 0.02) * 0.06 + vec2(uTime * 0.0012, uTime * 0.0004);
        float c = fbm(cp) + 0.35 * uStorm;
        float cov = smoothstep(0.42, 0.75, c) * smoothstep(-0.0, 0.25, el);
        float lit = 0.6 + 0.6 * pow(sd, 3.0);
        vec3 cc = mix(uSkyHor * 0.75, uSkyHor * 1.15 + uSunCol * 0.12 * lit, smoothstep(0.4, 0.9, fbm(cp * 1.7 + 3.1)));
        cc = mix(cc, uSkyZen * 0.6, uNight * 0.6);
        col = mix(sky, cc, cov * 0.9);
        /* far headlands on the horizon */
        float az = atan(d.x, -d.z);
        float land = 0.012 * smoothstep(0.35, 0.8, n1(vec2(az * 0.09, 0.3))) + 0.006 * n2(vec2(az * 0.5, 0.7)) - 0.004;
        if (el < land) col = mix(uSkyHor * 0.55, uSkyHor, 0.55 + 0.4 * smoothstep(0.0, 0.012, el));
    } else {
        /* the sea, far below the cliff */
        float t = 60.0 / -el;
        vec2 sp = d.xz * t;
        vec2 w = sp * 0.02 + vec2(uTime * 0.01, uTime * 0.006);
        vec2 g = vec2(n2(w) - n2(w + vec2(0.01, 0)), n2(w) - n2(w + vec2(0, 0.01))) * 25.0
               + vec2(n3(w * 3.1 + 0.3) - n3(w * 3.1 + vec2(0.31, 0.3)), n3(w * 3.1 + 0.3) - n3(w * 3.1 + vec2(0.3, 0.31))) * 18.0;
        vec3 n = normalize(vec3(g.x, 1.0, g.y));
        vec3 r = reflect(d, n);
        r.y = abs(r.y);
        float fr = 0.03 + 0.97 * pow(1.0 - max(dot(-d, n), 0.0), 5.0);
        vec3 refl = mix(uSkyHor, uSkyZen, pow(clamp(r.y, 0.0, 1.0), 0.55));
        vec3 deep = vec3(0.025, 0.05, 0.045) * (uSkyHor + uSunCol * 0.2) * 3.0;
        col = mix(deep, refl, fr);
        col += uSunCol * pow(max(dot(r, uSunDir), 0.0), 400.0) * 3.0;
        col += vec3(0.6, 0.65, 0.8) * pow(max(dot(r, uMoonDir), 0.0), 300.0) * uMoonPhase * uNight * 0.6;
        /* white water here and there */
        float foam = smoothstep(0.72, 0.8, n1(sp * 0.004 + uTime * 0.003) * 0.6 + n3(sp * 0.03) * 0.5);
        col = mix(col, uSkyHor * 0.9, foam * 0.25 * smoothstep(0.0, -0.1, el));
        col = mix(col, uSkyHor, exp(-(-el) * 40.0) * 0.8);
    }
    return col;
}

/* stone walls in courses */
void ashlar(inout S s, vec2 uv, vec3 T) {
    float ch = 0.33;
    float cy = floor(uv.y / ch);
    float ofs = h12(vec2(cy, 3.7)) * 3.0, L = 0.48 + 0.3 * h12(vec2(cy, 9.1));
    float x = uv.x + ofs;
    float bx = floor(x / L);
    vec2 f = vec2(fract(x / L) * L, fract(uv.y / ch) * ch);
    vec2 e = min(f, vec2(L, ch) - f);
    float de = min(e.x, e.y);
    float id = h12(vec2(bx, cy));
    float mort = 1.0 - smoothstep(0.004, 0.012, de + (n3(uv * 0.9) - 0.5) * 0.006);
    float mott = fbm(uv * 0.35 + id) ;
    s.alb = vC * (0.82 + 0.3 * id) * (0.8 + 0.4 * mott);
    s.alb = mix(s.alb, vC * vec3(0.55, 0.53, 0.5), mort);
    s.rough = 0.85;
    /* bevelled block edges catch the light */
    float bev = 1.0 - smoothstep(0.0, 0.03, de);
    vec2 dir = vec2(e.x < e.y ? (f.x < L * 0.5 ? -1.0 : 1.0) : 0.0, e.x < e.y ? 0.0 : (f.y < ch * 0.5 ? -1.0 : 1.0));
    s.n = normalize(s.n + (T * dir.x + vec3(0, 1, 0) * dir.y) * bev * 0.6 * (1.0 - mort)
                    + (T * (n2(uv * 0.7) - 0.5) + vec3(0, 1, 0) * (n2(uv * 0.7 + 0.5) - 0.5)) * 0.18
                    + (T * (n3(uv * 1.3) - 0.5) + vec3(0, 1, 0) * (n3(uv * 1.3 + 0.5) - 0.5)) * 0.06);
    /* pitting and lichen-dark weathering in the old stone */
    s.alb *= 0.92 + 0.12 * n3(uv * 1.1 + id);
}

vec3 leather(vec3 c, vec2 p) {
    float m = fbm(p * 1.3);
    return c * (0.75 + 0.5 * m) * (0.9 + 0.2 * n3(p * 6.0));
}

/* the page textures and the living plate */
vec4 page_tex(int k, vec2 uv) {
    if (k == 0) return texture(uPage0, uv);
    if (k == 1) return texture(uPage1, uv);
    if (k == 2) return texture(uPage2, uv);
    if (k == 3) return texture(uPage3, uv);
    if (k == 4) return texture(uPage4, uv);
    return texture(uPage5, uv);
}

void main() {
    vec3 p = vP, V = normalize(uCam - p);
    vec3 n = normalize(vN);
    S s;
    s.alb = vC; s.rough = 0.6; s.metal = 0.0; s.emis = vec3(0); s.n = n; s.spec = 0.04; s.wrap = 0.0;
    float ao_mul = 1.0, clearcoat = 0.0;
    int m = vMat;
    if (uDebug == 9) { oCol = vec4(vC * max(dot(n, vec3(0.3, 0.8, 0.5)), 0.1), 1.0); return; }
    if (m == 19) {                                         /* the sky */
        vec3 d = normalize(p - uCam);
        vec3 c = outside(d);
        /* old glass: a faint green cast and a little of the room by night */
        c = c * vec3(0.9, 0.95, 0.92) + vec3(0.02, 0.012, 0.006) * uNight * uLampOn;
        oCol = vec4(c, 1.0);
        return;
    }
    if (m == 0) {                                          /* stone */
        if (vVar > 0.5 && vVar < 1.5) {
            if (in_window(p)) discard;
            float a = atan(p.x, -p.z);
            ashlar(s, vec2(a * RIN, p.y), vec3(cos(a), 0, sin(a)));
        } else {
            /* dressed stone: courses on whatever way the face looks */
            vec3 T = abs(n.y) > 0.7 ? vec3(1, 0, 0) : normalize(vec3(-n.z, 0.0, n.x));
            vec2 uv = abs(n.y) > 0.7 ? p.xz : vec2(dot(p, T), p.y);
            ashlar(s, uv * (vVar > 2.5 ? 0.8 : 1.0) + 7.3, T);
        }
        s.alb *= vec3(1.02, 0.97, 0.9);
    } else if (m == 1) {    } else if (m == 1) {                                   /* floor flags */
        vec2 q = p.xz;
        float row = floor(q.y / 0.55);
        float x = q.x + h12(vec2(row, 1.3)) * 2.0;
        float L = 0.7 + 0.25 * h12(vec2(row, 7.7));
        float bx = floor(x / L);
        vec2 f = vec2(fract(x / L) * L, fract(q.y / 0.55) * 0.55);
        vec2 e = min(f, vec2(L, 0.55) - f);
        float de = min(e.x, e.y);
        float id = h12(vec2(bx, row));
        float mort = 1.0 - smoothstep(0.003, 0.009, de);
        float mott = fbm(q * 0.5 + id * 7.0);
        s.alb = vC * (0.75 + 0.35 * id) * (0.8 + 0.4 * mott);
        s.alb = mix(s.alb, vC * 0.4, mort);
        /* worn smooth where people walk */
        float wear = smoothstep(3.8, 1.0, length(q - vec2(0.0, 0.0)));
        s.rough = mix(0.8, 0.42, wear * (0.6 + 0.4 * n2(q * 0.3))) + mort * 0.2;
        s.n = normalize(n + vec3(n3(q * 1.7) - 0.5, 0, n3(q * 1.7 + 0.5) - 0.5) * 0.2);
    } else if (m == 2 || m == 3 || m == 17) {              /* oak, shelving, roof timber */
        vec2 w = vUV;                                      /* along the grain, across it (metres) */
        float warp = n1(w * vec2(0.15, 0.6)) * 0.8 + n2(w * vec2(0.4, 1.5)) * 0.25;
        float ring = fract((w.y + warp * 0.12) * 38.0);
        float late = smoothstep(0.55, 0.9, ring) * (1.0 - smoothstep(0.9, 1.0, ring));
        float fig = n2(vec2(w.x * 0.08, w.y * 1.2 + warp));
        float pores = n3(vec2(w.x * 0.9, w.y * 60.0));
        vec3 c = vC * vec3(0.78, 0.76, 0.74) * (0.8 + 0.35 * fig);
        c = mix(c, c * 0.6, late * 0.65);
        c *= 0.88 + 0.18 * pores;
        s.alb = mix(vec3(dot(c, vec3(0.333))), c, 0.85);
        s.rough = m == 2 ? 0.42 : m == 3 ? 0.55 : 0.75;
        clearcoat = m == 2 ? 0.5 : m == 3 ? 0.2 : 0.0;
        s.n = normalize(n + (vec3(pores) - 0.5) * 0.04);
    } else if (m == 4) {                                   /* a book's spine or covers */
        if (vVar < -0.5) {
            s.alb = leather(vC, p.xy * 7.0 + p.zx * 5.0) * 0.85;
            s.rough = 0.65;
        } else {
            float style = floor(vVar), h = fract(vVar);
            float u = vUV.x, v = vUV.y, ym = v * h;      /* height up the spine, metres */
            vec2 bp = vec2(u * 0.03, ym) * 20.0 + vC.rg * 10.0;
            vec3 c = leather(vC, bp);
            /* the round of the spine */
            float a = (u - 0.5) * 2.0;
            vec3 T = normalize(cross(vec3(0, 1, 0), n));
            s.n = normalize(n + T * a * 0.9);
            /* raised bands and gilt rules, depending on the style */
            float gilt = 0.0, band = 0.0;
            float top = h - ym;
            if (style >= 2.0) {
                for (int i = 1; i <= 4; i++) {
                    float by = h * (0.12 + 0.19 * float(i));
                    float d = abs(ym - by);
                    band = max(band, 1.0 - smoothstep(0.004, 0.008, d));
                    gilt = max(gilt, (1.0 - smoothstep(0.0012, 0.0025, abs(d - 0.0095))) * step(4.0, style));
                }
            }
            gilt = max(gilt, 1.0 - smoothstep(0.001, 0.0022, abs(top - 0.018)));
            gilt = max(gilt, 1.0 - smoothstep(0.001, 0.0022, abs(ym - 0.018)));
            /* a title label */
            if (mod(style, 3.0) < 1.5) {
                float ly = h * 0.74;
                if (abs(ym - ly) < 0.022 && abs(u - 0.5) < 0.36) {
                    c = mix(c, vec3(0.08, 0.05, 0.035), 0.75);
                    float txt = step(0.55, n3(vec2(u * 3.0, ym * 22.0) + style)) * step(abs(ym - ly), 0.008) * step(abs(u - 0.5), 0.28);
                    gilt = max(gilt, txt * 0.8);
                }
            } else {
                float ly = h * 0.78;
                float txt = step(0.5, n3(vec2(u * 2.5, ym * 25.0) + style * 3.1)) * step(abs(ym - ly), 0.007) * step(abs(u - 0.5), 0.3);
                gilt = max(gilt, txt * 0.7);
            }
            c *= 1.0 - 0.35 * band;
            s.n = normalize(s.n + vec3(0, 1, 0) * band * sign(0.5 - fract(ym * 50.0)) * 0.3);
            s.alb = mix(c, vec3(0.75, 0.55, 0.25), gilt);
            s.metal = gilt;
            s.rough = mix(0.6, 0.3, gilt);
            /* worn at the head and tail */
            s.alb *= 0.85 + 0.25 * smoothstep(0.0, 0.03, min(ym, top));
        }
    } else if (m == 5) {                                   /* the tops of books: page edges, dust */
        float th = vVar, cv = 0.003 / max(th, 0.004);
        float u = vUV.x;
        bool cover = u < cv || u > 1.0 - cv || vUV.y > 0.985;
        vec3 paper = vec3(0.62, 0.55, 0.42) * (0.85 + 0.15 * n3(vec2(u * 3.0, vUV.y * 20.0)));
        s.alb = cover ? vC * 0.8 : paper;
        s.alb = mix(s.alb, vec3(0.4, 0.38, 0.35), 0.25 * n2(p.xz * 3.0));
        s.rough = 0.85;
    } else if (m == 6) {                                   /* brass */
        float tar = fbm(p.xy * 6.0 + p.zx * 4.0);
        s.alb = vC * (0.85 + 0.3 * tar);
        s.metal = 1.0;
        s.rough = 0.25 + 0.2 * smoothstep(0.4, 0.7, tar);
    } else if (m == 7) {                                   /* paper edges */
        float st = 0.85 + 0.15 * n3(vec2(vUV.x * 0.5, vUV.y * 60.0));
        s.alb = vC * st;
        s.rough = 0.9;
        s.wrap = 0.3;
    } else if (m == 8) {                                   /* a page */
        int layer = int(mod(vVar, 8.0) + 0.5);
        int sel = int(vVar / 8.0 + 0.01);
        if (!gl_FrontFacing) {                             /* a turning leaf's other side */
            layer = int(fract(vVar) * 100.0 + 0.5);
            sel = 0;
            n = -n;
            s.n = n;
        }
        vec2 uv = vUV;
        vec4 pg = page_tex(layer, vec2(uv.x, 1.0 - uv.y));
        s.alb = pg.rgb;
        s.rough = mix(0.85, 0.22, pg.a);                   /* wet ink shines */
        s.wrap = 0.35;
        if (sel > 0) {
            vec2 q = (uv - uPanel.xy) / (uPanel.zw - uPanel.xy);
            if (q.x > 0.0 && q.x < 1.0 && q.y > 0.0 && q.y < 1.0) {
                vec2 aq = vec2(q.x, 1.0 - q.y) * uAgeScale;
                vec3 c = sel == 1 ? texture(uAgeLive, aq).rgb : texture(uAgeOld, aq).rgb;
                /* like an old hand-tinted plate: warm, a little faded, grainy */
                float l = dot(c, vec3(0.3, 0.55, 0.15));
                c = mix(c, vec3(l) * vec3(1.08, 0.98, 0.82), 0.12);
                c = smoothstep(0.0, 1.0, c * 1.08) * 0.92 + vec3(0.02, 0.016, 0.01);
                float gr = h12(floor(q * vec2(420.0, 350.0)) + floor(uTime * 12.0)) - 0.5;
                c *= 1.0 + gr * 0.07;
                vec2 e = min(q, 1.0 - q);
                float vig = smoothstep(0.0, 0.16, e.x) * smoothstep(0.0, 0.16, e.y);
                vig = mix(0.35, 1.0, vig) * (1.0 - 0.25 * length(q - 0.5));
                c *= vig;
                s.alb = mix(s.alb, c * pg.rgb * 1.2, 0.95);
                s.emis = c * uPanelGlow * vig;
                s.rough = 0.6;
            }
        }
    } else if (m == 10) {                                  /* the rug */
        vec2 q = vUV;
        float r = length(q), a = atan(q.y, q.x);
        vec3 mad = vec3(0.30, 0.065, 0.045), ind = vec3(0.045, 0.06, 0.12), ivo = vec3(0.56, 0.5, 0.38),
             gold = vec3(0.42, 0.27, 0.09), grn = vec3(0.1, 0.15, 0.09);
        vec3 c = mad;
        /* the field: a fine lattice of lozenges, each holding a flower */
        vec2 lq = mat2(0.7071, 0.7071, -0.7071, 0.7071) * q * 9.0;
        vec2 lf = abs(fract(lq) - 0.5);
        float lz = lf.x + lf.y;
        c = mix(c, ind, (1.0 - smoothstep(0.035, 0.06, abs(lz - 0.47))) * 0.9);
        float fl = length(fract(lq) - 0.5);
        c = mix(c, mix(gold, ivo, 0.4), (1.0 - smoothstep(0.08, 0.11, fl)) * 0.8);
        c = mix(c, ind * 1.5, (1.0 - smoothstep(0.03, 0.05, abs(fl - 0.2))) * 0.5);
        /* the medallion: an eight-pointed star */
        float star = 0.48 + 0.07 * cos(a * 8.0);
        if (r < star + 0.02) c = ivo * 0.8;
        if (r < star) c = ind;
        float petals = 0.3 + 0.05 * cos(a * 16.0);
        if (r < star - 0.06) c = mix(ind, grn, 0.5);
        if (r < petals) c = mad * 1.1;
        if (abs(r - petals) < 0.012) c = gold;
        if (r < 0.1) c = gold * 0.8;
        if (r < 0.06) c = ind;
        /* borders */
        if (r > 1.36) c = gold;
        if (r > 1.4) {
            c = ind;
            float k = a / 6.2831853 * 40.0;
            vec2 m2 = vec2(fract(k) - 0.5, (r - 1.56) / 0.14);
            float d = abs(m2.x) * 2.0 + abs(m2.y);
            if (d < 0.75) c = mad;
            if (d < 0.35) c = ivo;
            if (abs(m2.y) > 0.9) c = grn;
        }
        if (abs(r - 1.4) < 0.012 || abs(r - 1.72) < 0.012) c = ivo;
        float fib = n3(q * 30.0) * 0.5 + n3(q * 70.0) * 0.5;
        s.alb = c * (0.8 + 0.35 * fib) * (0.85 + 0.2 * fbm(q * 0.8));
        /* worn under the desk and where the chair goes */
        float wear = smoothstep(0.9, 0.0, length(q - vec2(-0.25, 0.55)));
        s.alb = mix(s.alb, vec3(dot(s.alb, vec3(0.33))) * 1.2, wear * 0.35);
        s.rough = 0.95;
        s.wrap = 0.2;
    } else if (m == 11) {                                  /* iron */
        s.alb = vC * (0.8 + 0.4 * n3(p.xy * 8.0 + p.z));
        s.metal = 0.6;
        s.rough = 0.55;
    } else if (m == 12) {                                  /* soot */
        s.alb = vC * (0.6 + 0.6 * fbm(p.xy * 2.0 + p.zx));
        s.rough = 0.95;
        float glow = smoothstep(0.7, 0.0, distance(p, uFirePos));
        s.emis = uFireCol * 0.025 * glow * glow;
    } else if (m == 13) {                                  /* the fire's wood and embers */
        float cr = fbm(p.xz * 9.0 + p.y * 7.0);
        float crack = smoothstep(0.55, 0.62, cr) * (1.0 - smoothstep(0.62, 0.75, cr));
        float under = clamp(0.55 - n.y, 0.0, 1.0);
        float fl = 0.7 + 0.3 * n1(vec2(uTime * 0.2, p.x * 2.0));
        s.alb = vC * (0.4 + 0.6 * cr);
        s.rough = 0.9;
        vec3 ember = vec3(1.0, 0.32, 0.06);
        if (vVar > 0.5) s.emis = ember * (0.6 + 2.2 * n1(p.xz * 6.0 + uTime * 0.03)) * fl * 1.5;
        else s.emis = ember * (crack * 3.0 + under * under * 1.6) * fl;
    } else if (m == 14) {                                  /* glazed china */
        s.alb = vC;
        float band = smoothstep(0.003, 0.001, abs(vUV.y - 0.066)) + smoothstep(0.002, 0.0, abs(vUV.y - 0.052)) * 0.6;
        s.alb = mix(s.alb, vec3(0.08, 0.14, 0.38), band);
        s.rough = 0.12;
        s.spec = 0.05;
    } else if (m == 15) {                                  /* the lamp's glass */
        vec3 warm = uLampCol;
        if (vVar < 0.5) { s.alb = vec3(0.1); s.emis = warm * 0.9 * uLampOn; s.rough = 0.05; }
        else {
            s.alb = vC * 0.9; s.rough = 0.3; s.wrap = 0.6;
            float facing = abs(dot(n, V));
            float lowr = smoothstep(0.24, 0.0, abs(p.y - uLampPos.y + 0.02));
            s.emis = warm * vec3(1.0, 0.85, 0.7) * (0.05 + 0.25 * facing * facing + 0.25 * lowr) * uLampOn;
        }
    } else if (m == 16) {                                  /* cloth */
        float wv = n3(p.xy * 40.0 + p.z * 30.0);
        s.alb = vC * (0.8 + 0.3 * wv);
        s.rough = 0.95;
        s.wrap = 0.25;
    } else if (m == 20) {                                  /* the great book's leather */
        s.alb = leather(vC, p.xz * 30.0);
        float ex = 1.0 - smoothstep(0.0, 0.02, min(abs(vUV.x), 1.0));
        s.rough = 0.55;
        s.alb *= 1.0 - 0.2 * ex;
    } else if (m == 21) {                                  /* tea */
        s.alb = vC;
        s.rough = 0.04;
        s.spec = 0.03;
    } else if (m == 22) {                                  /* black glass, the pen */
        s.alb = vC;
        s.rough = 0.12;
        s.spec = 0.05;
    } else if (m == 23) {                                  /* wax */
        s.alb = vC;
        s.rough = 0.5;
        s.wrap = 0.5;
    }

    /* ambient occlusion from the voxelised room */
    vec3 aq = (p + s.n * 0.16 + vec3(5.8, 0.0, 5.8)) / vec3(11.6, 10.0, 11.6);
    float occ = texture(uAO, aq).r;
    float ao = clamp(1.0 / (1.0 + 9.0 * max(occ - 0.03, 0.0)), 0.08, 1.0) * ao_mul;

    vec3 col = vec3(0);
    /* the sun or the moon, through the windows */
    if (dot(uKeyCol, vec3(1)) > 0.0) {
        float sh = key_shadow(p, s.n);
        if (sh > 0.0) col += brdf(s, uKeyDir, V, uKeyCol * sh);
    }
    /* the fire */
    {
        vec3 l = uFirePos - p;
        float d2 = dot(l, l);
        vec3 L = l * inversesqrt(d2);
        float sh = pt_shadow(uFireSh, uFirePos, p, s.n);
        col += brdf(s, L, V, uFireCol * sh / (d2 + 0.08));
    }
    /* the lamp */
    if (uLampOn > 0.0) {
        vec3 l = uLampPos - p;
        float d2 = dot(l, l);
        vec3 L = l * inversesqrt(d2);
        float sh = pt_shadow(uLampSh, uLampPos, p, s.n);
        col += brdf(s, L, V, uLampCol * uLampOn * sh / (d2 + 0.02));
    }
    /* sky light from each window, and ambient */
    vec3 diffc = s.alb * (1.0 - s.metal);
    vec3 E = vec3(0);
    for (int i = 0; i < 3; i++) {
        vec3 l = uWinC[i].xyz - p;
        float d2 = dot(l, l);
        vec3 L = l * inversesqrt(d2);
        vec3 wn = normalize(vec3(-uWinC[i].x, 0.0, -uWinC[i].z));
        float cs = max((dot(s.n, L) + 0.25) / 1.25, 0.0), cw = max(dot(wn, -L), 0.0);
        E += uWinL * uWinC[i].w * cs * cw / (PI * (d2 + uWinC[i].w * 0.3));
    }
    vec3 hemi = mix(uGndAmb, uSkyAmb, s.n.y * 0.5 + 0.5) + uBounce
              + uFireCol * 0.012 + uLampCol * uLampOn * 0.02;
    col += diffc * (E + hemi) * ao;
    /* a little sheen of the window light on metal and varnish */
    vec3 R = reflect(-V, s.n);
    vec3 F0 = mix(vec3(s.spec), s.alb, s.metal);
    float fres = pow(1.0 - max(dot(s.n, V), 0.0), 5.0);
    if (s.metal > 0.3 || s.rough < 0.3 || clearcoat > 0.0) {
        float env = 0.0;
        for (int i = 0; i < 3; i++) {
            vec3 l = normalize(uWinC[i].xyz - p);
            env += pow(max(dot(R, l), 0.0), mix(400.0, 8.0, s.rough)) * mix(60.0, 1.0, s.rough);
        }
        /* the room itself, warm and dim, in everything polished */
        vec3 room = mix(uGndAmb * 6.0 + uFireCol * 0.02 + uLampCol * uLampOn * 0.03, uSkyAmb * 3.0, R.y * 0.5 + 0.5);
        col += (F0 + (1.0 - F0) * fres * (1.0 - s.rough)) * (uWinL * 0.04 * env + room) * ao;
    } else col += F0 * fres * hemi * 0.6 * ao;
    if (clearcoat > 0.0) {
        float cc = 0.04 + 0.96 * fres;
        float g = 0.0;
        for (int i = 0; i < 3; i++) g += pow(max(dot(R, normalize(uWinC[i].xyz - p)), 0.0), 120.0) * 8.0;
        vec3 lf = normalize(uFirePos - p), ll = normalize(uLampPos - p);
        col += cc * clearcoat * ao * (uWinL * 0.04 * g + uFireCol * pow(max(dot(R, lf), 0.0), 200.0) * 0.15
                                      + uLampCol * uLampOn * pow(max(dot(R, ll), 0.0), 200.0) * 0.4);
    }
    col += s.emis;
    if (uDebug == 1) col = vec3(float(m % 4), float((m / 4) % 4), float(m / 16)) / 3.0 * 0.3 + (any(isnan(col)) ? vec3(10, 0, 10) : vec3(0));
    if (uDebug == 2) col = vec3(ao) * 0.3;
    oCol = vec4(col, 1.0);
}
);

static const char *VS_FULL = "#version 330 core\n" GLSL(
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
);

/* flames, steam: camera-facing quads built each frame */
static const char *VS_FX = "#version 330 core\n" GLSL(
layout(location = 0) in vec3 aP;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec2 aK;
uniform mat4 uVP;
out vec2 vUV; out vec2 vK; out vec3 vP;
void main() { vUV = aUV; vK = aK; vP = aP; gl_Position = uVP * vec4(aP, 1.0); }
);

static const char *FS_FLAME = "#version 330 core\n" NOISE GLSL(
in vec2 vUV; in vec2 vK; in vec3 vP;
out vec4 oCol;
uniform float uTime;
void main() {
    vec2 q = vUV;
    float sd = vK.x, k = vK.y;
    float t = uTime;
    float n = fbm(vec2(q.x * 0.35 + sd, q.y * 0.22 - t * 0.55 + sd * 3.0));
    float n2v = n1(vec2(q.x * 0.6 + sd * 2.0, q.y * 0.5 - t * 0.9));
    float x = q.x - 0.5 + (n - 0.5) * 0.55 * q.y + (n2v - 0.5) * 0.2 * q.y;
    float w = 0.42 * pow(1.0 - q.y, 0.65) * (0.6 + 0.4 * smoothstep(0.0, 0.25, q.y));
    float f = smoothstep(w, w * 0.15, abs(x)) * smoothstep(1.0, 0.45, q.y + (n - 0.5) * 0.5);
    f *= smoothstep(0.0, 0.08, q.y);
    float core = smoothstep(0.3, 0.9, f) * (1.0 - q.y);
    vec3 c = mix(vec3(1.0, 0.22, 0.03), vec3(1.0, 0.62, 0.22), core);
    c = mix(c, vec3(1.0, 0.85, 0.55), core * core * 0.6);
    oCol = vec4(c * f * k * 2.4, 1.0);
}
);

static const char *FS_STEAM = "#version 330 core\n" NOISE GLSL(
in vec2 vUV; in vec2 vK; in vec3 vP;
out vec4 oCol;
uniform float uTime;
uniform vec3 uTint;
void main() {
    vec2 q = vUV;
    float t = uTime;
    float sway = (n1(vec2(q.y * 0.08 - t * 0.02, 0.3)) - 0.5) * 0.7 * q.y;
    float x = q.x - 0.5 - sway;
    float wisp = n1(vec2(x * 0.12 + 0.2, q.y * 0.08 - t * 0.025)) * n1(vec2(x * 0.2 + 0.7, q.y * 0.12 - t * 0.04));
    float a = smoothstep(0.2, 0.5, wisp) * smoothstep(0.4, 0.05, abs(x)) * smoothstep(0.0, 0.2, q.y) * smoothstep(1.0, 0.2, q.y);
    oCol = vec4(uTint * a * 0.12, 1.0);
}
);

/* dust in the air: points, bright where the sun reaches them */
static const char *VS_DUST = "#version 330 core\n" GLSL(
uniform mat4 uVP, uKeyVP;
uniform float uTime, uPx;
uniform vec3 uKeyDir, uKeyCol, uFirePos, uFireCol, uLampPos, uLampCol, uCam;
uniform vec4 uWinC[3];
uniform sampler2DShadow uKeySh;
out vec3 vCol;
out float vA;
vec3 h3(float i) { return fract(sin(vec3(i * 12.9898, i * 78.233, i * 37.719)) * 43758.5453); }
void main() {
    float i = float(gl_VertexID);
    vec3 h = h3(i + 0.5), g = h3(i * 1.37 + 7.1);
    vec3 p;
    float life = 40.0 + 30.0 * g.z, ph = fract(uTime / life + h.z);
    if (gl_VertexID % 3 != 0 && dot(uKeyCol, vec3(1.0)) > 0.0) {
        /* in the shafts: from a point on a window, back along the light */
        int w = gl_VertexID % 3 == 1 ? int(g.x * 3.0) : 1 + int(g.y * 2.0);
        vec3 c = uWinC[w].xyz;
        vec3 wn = normalize(vec3(-c.x, 0.0, -c.z)), wt = vec3(-wn.z, 0.0, wn.x);
        p = c + wt * (h.x - 0.5) * 1.1 + vec3(0.0, (h.y - 0.5) * 2.6, 0.0) - uKeyDir * (0.3 + g.y * 6.0);
    } else {
        float r = sqrt(h.x) * 4.2, a = h.y * 6.2831;
        p = vec3(r * sin(a), 0.3 + g.x * 3.6, -r * cos(a));
    }
    /* drifting, rising a little, turning over */
    p += vec3(sin(uTime * 0.05 + h.x * 40.0), 0.0, cos(uTime * 0.043 + h.y * 40.0)) * 0.25;
    p.y += (ph - 0.5) * 0.5 + sin(uTime * 0.11 + g.x * 30.0) * 0.08;
    vec4 q = uKeyVP * vec4(p, 1.0);
    vec3 sc = q.xyz / q.w * 0.5 + 0.5;
    float vis = (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0) ? texture(uKeySh, sc) : 0.0;
    if (length(p.xz) > 4.4) vis = 0.0;
    float df = distance(p, uFirePos), dl = distance(p, uLampPos);
    vCol = uKeyCol * vis * 0.05 + uFireCol * 0.002 / (df * df + 0.3) + uLampCol * 0.004 / (dl * dl + 0.1);
    float fade = sin(ph * 3.14159);
    gl_Position = uVP * vec4(p, 1.0);
    float sz = uPx * 0.004 / max(gl_Position.w, 0.05);
    gl_PointSize = clamp(sz, 1.0, 9.0);
    vA = fade * clamp(sz, 0.15, 1.0) * clamp(1.6 / max(sz, 1.0), 0.15, 1.0);
}
);

static const char *FS_DUST = "#version 330 core\n" GLSL(
in vec3 vCol; in float vA;
out vec4 oCol;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float a = smoothstep(0.5, 0.0, length(d));
    oCol = vec4(vCol * a * vA, 1.0);
}
);

/* light shafts: march the sun's shadow map through the room's air */
static const char *FS_VOL = "#version 330 core\n" GLSL(
in vec2 vUV;
out vec4 oCol;
uniform sampler2D uDepth;
uniform sampler3D uNoise3;
uniform mat4 uInvVP, uKeyVP;
uniform vec3 uCam, uKeyDir, uKeyCol, uFirePos, uFireCol, uLampPos, uLampCol;
uniform sampler2DShadow uKeySh;
uniform float uTime, uDens;
uniform int uFrame;
void main() {
    float z = texture(uDepth, vUV).r;
    vec4 w = uInvVP * vec4(vUV * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);
    vec3 pe = w.xyz / w.w, rd = pe - uCam;
    float tmax = length(rd);
    rd /= tmax;
    /* only the room's air */
    float a = dot(rd.xz, rd.xz), b = dot(uCam.xz, rd.xz), c = dot(uCam.xz, uCam.xz) - 4.5 * 4.5;
    float disc = b * b - a * c;
    if (disc > 0.0 && a > 1e-5) tmax = min(tmax, (-b + sqrt(disc)) / a);
    if (rd.y > 0.0) tmax = min(tmax, (5.5 - uCam.y) / rd.y);
    tmax = min(tmax, 12.0);
    const int N = 22;
    float dt = tmax / float(N);
    float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))) + float(uFrame % 16) * 0.618034);
    float cth = dot(rd, uKeyDir), g = 0.68;
    float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cth, 1.5) * 0.2 + 0.025;
    vec3 L = vec3(0);
    bool key = dot(uKeyCol, vec3(1.0)) > 0.0;
    for (int i = 0; i < N; i++) {
        float t = (float(i) + j) * dt;
        vec3 p = uCam + rd * t;
        float dn = texture(uNoise3, p * 0.23 + vec3(uTime * 0.004, uTime * 0.002, 0.0)).r;
        float d2 = texture(uNoise3, p * 0.9 + vec3(0.0, uTime * 0.01, 0.3)).r;
        dn = 0.15 + 2.2 * dn * dn * dn + 0.6 * d2 * d2;
        vec3 s = vec3(0);
        if (key) {
            vec4 q = uKeyVP * vec4(p, 1.0);
            vec3 sc = q.xyz / q.w * 0.5 + 0.5;
            s += uKeyCol * texture(uKeySh, sc) * hg;
        }
        vec3 lf = uFirePos - p, ll = uLampPos - p;
        s += uFireCol * 0.04 / (dot(lf, lf) + 0.4) + uLampCol * 0.05 / (dot(ll, ll) + 0.15);
        L += s * dn;
    }
    oCol = vec4(L * dt * uDens, 1.0);
}
);

/* bloom: a chain of halvings, then back up */
static const char *FS_DOWN = "#version 330 core\n" GLSL(
in vec2 vUV;
out vec4 oCol;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uThresh;
void main() {
    vec3 c = texture(uSrc, vUV + uTexel * vec2(-1, -1)).rgb + texture(uSrc, vUV + uTexel * vec2(1, -1)).rgb +
             texture(uSrc, vUV + uTexel * vec2(-1, 1)).rgb + texture(uSrc, vUV + uTexel * vec2(1, 1)).rgb;
    c *= 0.25;
    if (uThresh > 0.0) {
        float l = max(c.r, max(c.g, c.b));
        c *= max(l - uThresh, 0.0) / max(l, 1e-4);
    }
    oCol = vec4(c, 1.0);
}
);
static const char *FS_UP = "#version 330 core\n" GLSL(
in vec2 vUV;
out vec4 oCol;
uniform sampler2D uSrc;
uniform vec2 uTexel;
void main() {
    vec3 c = texture(uSrc, vUV).rgb * 4.0;
    c += (texture(uSrc, vUV + uTexel * vec2(1, 0)).rgb + texture(uSrc, vUV - uTexel * vec2(1, 0)).rgb +
          texture(uSrc, vUV + uTexel * vec2(0, 1)).rgb + texture(uSrc, vUV - uTexel * vec2(0, 1)).rgb) * 2.0;
    c += texture(uSrc, vUV + uTexel).rgb + texture(uSrc, vUV - uTexel).rgb +
         texture(uSrc, vUV + uTexel * vec2(1, -1)).rgb + texture(uSrc, vUV + uTexel * vec2(-1, 1)).rgb;
    oCol = vec4(c / 16.0, 1.0);
}
);

/* the finish: air, bloom, exposure, a filmic curve and a grade */
static const char *FS_COMP = "#version 330 core\n" GLSL(
in vec2 vUV;
out vec4 oCol;
uniform sampler2D uScene, uVol, uBloom;
uniform vec2 uTexel, uVolTexel;
uniform float uExposure, uTime, uSharp;
uniform vec4 uVig;                       /* this view in the layout, normalised: x0 y0 w h */
uniform int uDebug;
float h12(vec2 p) { p = fract(p * vec2(443.897, 441.423)); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
void main() {
    vec3 c = texture(uScene, vUV).rgb;
    if (uSharp > 0.0) {
        vec3 b = texture(uScene, vUV + uTexel * vec2(1, 0)).rgb + texture(uScene, vUV - uTexel * vec2(1, 0)).rgb +
                 texture(uScene, vUV + uTexel * vec2(0, 1)).rgb + texture(uScene, vUV - uTexel * vec2(0, 1)).rgb;
        c = max(c + (c - b * 0.25) * uSharp, 0.0);
    }
    vec3 v = texture(uVol, vUV).rgb * 0.4 + (texture(uVol, vUV + uVolTexel * vec2(1, 0.5)).rgb + texture(uVol, vUV - uVolTexel * vec2(1, 0.5)).rgb +
             texture(uVol, vUV + uVolTexel * vec2(-0.5, 1)).rgb + texture(uVol, vUV - uVolTexel * vec2(-0.5, 1)).rgb) * 0.15;
    if (uDebug == 3) c = v * 4.0;
    else c += v;
    c += texture(uBloom, vUV).rgb * 0.06;
    c *= uExposure;
    /* grade before the curve: cool the shadows, warm the lights */
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(c * vec3(0.94, 1.0, 1.04), c * vec3(1.05, 1.0, 0.92), smoothstep(0.02, 0.6, l));
    c = aces(c);
    float l2 = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l2), c, 0.92);
    c = pow(c, vec3(1.0 / 2.2));
    /* a vignette over the whole wall, and fine grain */
    vec2 q = uVig.xy + vUV * uVig.zw;
    q.y = 1.0 - q.y;
    vec2 d = (q - 0.5) * vec2(1.0, 0.9);
    c *= 1.0 - 0.38 * smoothstep(0.25, 0.85, length(d));
    c += (h12(gl_FragCoord.xy + fract(uTime) * 91.0) - 0.5) * 0.018;
    oCol = vec4(c, 1.0);
}
);

/* ---- the pages ---- */
static const char *FS_PAPER = "#version 330 core\n" NOISE GLSL(
in vec2 vUV;
out vec4 oCol;
uniform int uKind;                       /* 0 journal, 1 a plate, 2 a blank journal page, 3/4 older pages */
uniform vec4 uPanel;
uniform vec2 uRule;                      /* first ruled line, spacing (in 0..1 of the page height) */
uniform float uSeed;
uniform vec2 uSize;
void main() {
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);   /* v down the page */
    vec2 px = uv * uSize;
    vec2 sp = uv * vec2(0.7, 1.0) + uSeed;
    float fib = n3(sp * 6.0) * 0.6 + n3(sp * 14.0 + 0.3) * 0.4;
    vec3 c = vec3(0.87, 0.81, 0.67) * (0.95 + 0.06 * fib);
    c *= 0.94 + 0.08 * fbm(sp * 0.7);
    /* toned towards the edges, and foxed */
    vec2 e = min(uv, 1.0 - uv);
    float edge = min(e.x * 1.4, e.y);
    c *= mix(0.8, 1.0, smoothstep(0.0, 0.08, edge + (n1(sp * 2.0) - 0.5) * 0.03));
    c = mix(c, c * vec3(0.92, 0.82, 0.66), smoothstep(0.06, 0.0, edge) * 0.7);
    for (int i = 0; i < 9; i++) {
        vec2 fp = vec2(h12(vec2(float(i), uSeed)), h12(vec2(uSeed, float(i) + 3.1)));
        float r = 0.004 + 0.012 * h12(vec2(float(i) * 7.0, uSeed + 1.0));
        float f = smoothstep(r, r * 0.3, length((uv - fp) * vec2(0.71, 1.0)) * (0.8 + 0.4 * n2(uv * 8.0)));
        c = mix(c, c * vec3(0.85, 0.7, 0.5), f * 0.5);
    }
    /* a big old stain on some pages */
    float st = smoothstep(0.62, 0.66, n1(sp * 0.45 + 1.7)) - smoothstep(0.66, 0.75, n1(sp * 0.45 + 1.7)) * 0.6;
    c = mix(c, c * vec3(0.9, 0.8, 0.65), st * 0.25 * step(2.5, float(uKind)));
    vec3 ink = vec3(0.16, 0.1, 0.06);
    if (uKind == 0 || uKind == 2 || uKind == 3) {
        /* faint ruling and a margin */
        float y = (uv.y - uRule.x) / uRule.y;
        if (y > -0.5 && uv.y < 0.95) {
            float d = abs(fract(y + 0.5) - 0.5) * uRule.y * uSize.y;
            c = mix(c, c * vec3(0.78, 0.84, 0.92), (1.0 - smoothstep(0.6, 1.6, d)) * 0.35);
        }
        if (abs(px.x - uSize.x * 0.1) < 1.2 && uv.y > 0.04 && uv.y < 0.96) c = mix(c, c * vec3(1.0, 0.75, 0.7), 0.4);
    }
    if (uKind == 1) {
        /* the plate's printed frame: a double rule with corner pieces */
        vec2 a = uPanel.xy * uSize, b = uPanel.zw * uSize;
        for (int k = 0; k < 2; k++) {
            float o = k == 0 ? 7.0 : 15.0, wdt = k == 0 ? 1.4 : 3.2;
            vec2 lo = a - o, hi = b + o;
            vec2 dd = max(lo - px, px - hi);
            float dbox = max(dd.x, dd.y);
            float rim = abs(dbox);
            c = mix(c, ink, (1.0 - smoothstep(wdt * 0.5, wdt * 0.5 + 1.0, rim)) * 0.85);
        }
        for (int k = 0; k < 4; k++) {
            vec2 cc = vec2(k % 2 == 0 ? a.x - 26.0 : b.x + 26.0, k < 2 ? a.y - 26.0 : b.y + 26.0);
            vec2 d = abs(px - cc);
            float dia = d.x + d.y;
            c = mix(c, ink, (1.0 - smoothstep(9.0, 10.5, dia)) * 0.8);
            c = mix(c, vec3(0.86, 0.8, 0.66), (1.0 - smoothstep(4.0, 5.5, dia)) * 0.9);
        }
        /* a rule under the caption */
        float ry = b.y + 150.0;
        if (abs(px.y - ry) < 1.2 && abs(px.x - uSize.x * 0.5) < uSize.x * 0.18) c = mix(c, ink, 0.7);
    }
    oCol = vec4(c, 0.0);
}
);

static const char *VS_INK = "#version 330 core\n" GLSL(
layout(location = 0) in vec2 aP;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aK;         /* t0, duration, fx, ink */
uniform vec2 uSize;
out vec2 vUV; out vec4 vK;
void main() {
    vUV = aUV; vK = aK;
    gl_Position = vec4(aP.x / uSize.x * 2.0 - 1.0, 1.0 - aP.y / uSize.y * 2.0, 0.0, 1.0);
}
);
static const char *FS_INK = "#version 330 core\n" GLSL(
in vec2 vUV; in vec4 vK;
out vec4 oCol;
uniform sampler2D uAtlas;
uniform vec2 uAtlasSize;
uniform float uT;
uniform int uWet;                        /* 1: write the wetness instead */
void main() {
    float cov = texture(uAtlas, vUV / uAtlasSize).r;
    float prog = (uT - vK.x) / max(vK.y, 1e-3);
    float rev = smoothstep(vK.z - 0.12, vK.z + 0.02, prog * 1.12);
    float wet = 1.0 - smoothstep(vK.x + vK.y, vK.x + vK.y + 7.0, uT);
    float a = cov * rev * vK.w;
    if (uWet == 1) { oCol = vec4(0, 0, 0, a * wet); return; }
    vec3 dry = vec3(0.09, 0.055, 0.03), fresh = vec3(0.02, 0.025, 0.06);
    oCol = vec4(mix(dry, fresh, wet), min(a * 1.15, 1.0));
}
);

/* ---- the worlds in the book: one small raymarched scene each ---- */
#define AGE_COMMON GLSL( \
in vec2 vUV; \
out vec4 oCol; \
uniform float uT; \
uniform vec2 uRes; \
const float PI = 3.14159265; \
vec3 cam_ray(vec2 uv, vec3 ro, vec3 ta, float fl) { \
    vec3 f = normalize(ta - ro), r = normalize(cross(f, vec3(0, 1, 0))), u = cross(r, f); \
    vec2 q = (uv - 0.5) * vec2(uRes.x / uRes.y, 1.0); \
    return normalize(f * fl + r * q.x + u * q.y); \
} \
vec3 tone(vec3 c) { c = max(c, 0.0); return c / (1.0 + c * 0.9); } \
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); } \
float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); \
    return mix(mix(h12(i), h12(i + vec2(1, 0)), f.x), mix(h12(i + vec2(0, 1)), h12(i + vec2(1, 1)), f.x), f.y); } \
float sdCyl(vec3 p, float r, float h) { vec2 d = vec2(length(p.xz) - r, abs(p.y) - h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); } \
float sdCone(vec3 p, float r, float h) { float q = length(p.xz); return max(dot(vec2(h, r) / length(vec2(h, r)), vec2(q, p.y - h)), -p.y); } \
)

/* SALTWATCH: a keep on a rock in a storm */
static const char *FS_AGE0 = "#version 330 core\n" NOISE AGE_COMMON GLSL(
float flash;
float keep(vec3 p) {
    vec3 q = p - vec3(0, 0, -70);
    float n = fbm(q.xz * 0.012 + q.y * 0.01) * 7.0;
    float rock = (length(q.xz * vec2(1.0, 1.4)) - (24.0 - q.y * 0.8) + n) * 0.6;
    rock = max(rock, q.y - 9.0 - n * 0.4);
    float d = rock;
    vec3 w = q - vec3(0, 14, 0);
    d = min(d, sdBox(w, vec3(11, 5.5, 7)));
    vec3 cw = w - vec3(0, 6.0, 0);
    float cren = max(sdBox(cw, vec3(11.4, 0.8, 7.4)), -sdBox(vec3(mod(cw.x + 0.9, 1.8) - 0.9, cw.y - 0.6, cw.z), vec3(0.45, 0.6, 8)));
    d = min(d, cren);
    for (int i = 0; i < 2; i++) {
        vec3 t = w - vec3(i == 0 ? -11.0 : 11.0, 2.0, i == 0 ? -7.0 : 7.0);
        d = min(d, sdCyl(t, 3.2, 9.0));
        d = min(d, sdCone(t - vec3(0, 9.0, 0), 4.0, 6.0));
    }
    vec3 t = w - vec3(4, 10, -2);
    d = min(d, sdCyl(t, 3.8, 14.0));
    d = min(d, sdCone(t - vec3(0, 14.0, 0), 4.6, 8.0));
    return d;
}
vec3 nrm(vec3 p) {
    vec2 e = vec2(0.05, 0);
    return normalize(vec3(keep(p + e.xyy) - keep(p - e.xyy), keep(p + e.yxy) - keep(p - e.yxy), keep(p + e.yyx) - keep(p - e.yyx)));
}
vec3 sky(vec3 d) {
    vec2 cp = d.xz / max(d.y + 0.08, 0.03) * 0.05 + vec2(uT * 0.018, uT * 0.007);
    float c = fbm(cp) * 0.7 + fbm(cp * 2.3 + 1.0) * 0.3;
    vec3 col = mix(vec3(0.16, 0.19, 0.21), vec3(0.05, 0.06, 0.07), smoothstep(0.35, 0.75, c));
    col *= 0.8 + 0.4 * smoothstep(0.0, 0.3, d.y);
    col += vec3(0.5, 0.55, 0.7) * flash * (0.5 + c);
    return col;
}
void main() {
    float tf = uT / 7.3;
    float fh = h12(vec2(floor(tf), 4.2));
    flash = step(0.4, fh) * exp(-fract(tf) * 7.3 * 6.0) * 2.5 + step(0.4, fh) * exp(-max(fract(tf) * 7.3 - 0.18, 0.0) * 9.0) * step(0.18, fract(tf) * 7.3) * 1.5;
    vec3 ro = vec3(sin(uT * 0.03) * 6.0, 5.0 + sin(uT * 0.4) * 0.6, 20.0);
    vec3 rd = cam_ray(vUV, ro, vec3(0, 16, -70), 1.5);
    rd.y += sin(uT * 0.5) * 0.004;
    vec3 col = sky(rd);
    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 90; i++) {
        vec3 p = ro + rd * t;
        float d = keep(p);
        if (d < 0.02 * t * 0.02 + 0.01) { hit = t; break; }
        t += d;
        if (t > 200.0) break;
    }
    float ts = rd.y < 0.0 ? -ro.y / rd.y : 1e9;
    vec3 L = normalize(vec3(-0.5, 0.6, 0.3));
    if (hit > 0.0 && hit < ts) {
        vec3 p = ro + rd * hit, n = nrm(p);
        float stone = 0.6 + 0.4 * fbm(p.xy * 0.2 + p.z * 0.1);
        vec3 a = vec3(0.22, 0.23, 0.22) * stone;
        float dif = max(dot(n, L), 0.0) * 0.5 + 0.35 * (n.y * 0.5 + 0.5);
        col = a * (dif * vec3(0.5, 0.55, 0.6) + flash * 1.4 * max(dot(n, normalize(vec3(0.3, 1.0, 0.6))), 0.0));
        /* the lamp in the high window, and a lit slit below */
        vec3 q = p - vec3(4, 34, -72);
        float lw = smoothstep(1.4, 0.4, length(q.xy * vec2(1.2, 0.7)));
        col += vec3(1.0, 0.7, 0.3) * lw * 3.0 * step(0.0, p.z + 72.0 + 4.0);
        col = mix(col, sky(rd) * 0.9, 1.0 - exp(-hit * 0.006));
    } else if (rd.y < 0.0) {
        vec3 p = ro + rd * ts;
        vec2 w = p.xz * 0.025 + vec2(uT * 0.05, uT * 0.08);
        vec2 g = vec2(n1(w) - n1(w + vec2(0.004, 0)), n1(w) - n1(w + vec2(0, 0.004))) * 90.0;
        g += vec2(n3(w * 4.0 + uT * 0.02) - n3(w * 4.0 + vec2(0.004, 0) + uT * 0.02), 0.0) * 40.0;
        vec3 n = normalize(vec3(g.x, 1.0, g.y));
        float fr = 0.04 + 0.96 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
        vec3 r = reflect(rd, n);
        col = mix(vec3(0.03, 0.06, 0.06), sky(vec3(r.x, abs(r.y), r.z)), fr);
        /* foam: on the crests, and where the sea breaks on the rock */
        float crest = smoothstep(0.62, 0.8, n1(w * 1.5) * 0.7 + n3(w * 6.0) * 0.4);
        float dr = keep(vec3(p.x, 1.0, p.z));
        float surf = smoothstep(8.0, 0.0, dr) * (0.5 + 0.5 * n3(p.xz * 0.2 + uT * 0.3));
        col = mix(col, vec3(0.55, 0.6, 0.62) * (0.6 + flash), clamp(crest * 0.5 + surf, 0.0, 1.0));
        col = mix(col, sky(vec3(rd.x, 0.02, rd.z)), 1.0 - exp(-ts * 0.004));
    }
    /* the rain */
    vec2 ru = vec2(vUV.x * uRes.x / uRes.y + vUV.y * 0.25, vUV.y);
    float rn = h12(vec2(floor(ru.x * 160.0), 1.0));
    float streak = step(0.9, fract(ru.y * 1.6 + uT * 2.6 + rn * 13.0) * (0.6 + 0.4 * rn));
    col += vec3(0.5, 0.55, 0.6) * streak * 0.12 * (0.5 + flash);
    col = tone(col * 1.6);
    oCol = vec4(col, 1.0);
}
);

/* LANTERNWOOD: a forest of giant mushrooms that give their own light */
static const char *FS_AGE1 = "#version 330 core\n" NOISE AGE_COMMON GLSL(
vec3 glow_acc;
float cell_shroom(vec3 p, vec2 cid, out float hue, out float capd) {
    float h = h12(cid), h2 = h12(cid + 7.3);
    vec2 c = (cid + 0.5 + (vec2(h, h2) - 0.5) * 0.5) * 14.0;
    float H = 5.0 + 9.0 * h2, R = 2.5 + 4.0 * h;
    vec3 q = p - vec3(c.x, 0, c.y);
    q.x += sin(q.y * 0.25 + h * 6.0) * 0.6;
    float stem = sdCyl(q - vec3(0, H * 0.5, 0), 0.4 + 0.25 * R * 0.25 + 0.2 * smoothstep(H * 0.4, 0.0, q.y), H * 0.5);
    vec3 cq = q - vec3(0, H, 0);
    float cap = (length(cq / vec3(R, R * 0.45, R)) - 1.0) * R * 0.4;
    cap = max(cap, -(cq.y + 0.15 * R * (1.0 - length(cq.xz) / R)));
    hue = h;
    capd = cap;
    return min(stem, cap);
}
float map(vec3 p, out float hue, out float capd) {
    vec2 b = floor(p.xz / 14.0 - 0.5);
    float d = 1e9;
    hue = 0.0; capd = 1e9;
    for (int i = 0; i < 4; i++) {
        vec2 cid = b + vec2(i % 2, i / 2);
        float hh, cd;
        float dd = cell_shroom(p, cid, hh, cd);
        if (dd < d) { d = dd; hue = hh; }
        capd = min(capd, cd);
    }
    float g = p.y - (fbm(p.xz * 0.02) - 0.5) * 2.0;
    d = min(d, g);
    return d;
}
vec3 hue_col(float h) { return h < 0.33 ? vec3(0.2, 0.9, 0.8) : h < 0.66 ? vec3(0.5, 0.4, 1.0) : vec3(0.5, 1.0, 0.45); }
void main() {
    vec3 ro = vec3(uT * 0.4, 2.2 + sin(uT * 0.2) * 0.3, -uT * 0.9);
    vec3 ta = ro + vec3(sin(uT * 0.05) * 3.0, 1.8, -10.0);
    vec3 rd = cam_ray(vUV, ro, ta, 1.3);
    vec3 fogc = vec3(0.02, 0.05, 0.06);
    vec3 col = mix(vec3(0.01, 0.02, 0.04), vec3(0.03, 0.08, 0.1), smoothstep(0.4, -0.1, rd.y));
    float t = 0.2, hue = 0.0, capd = 1e9;
    vec3 glow = vec3(0);
    bool hit = false;
    for (int i = 0; i < 100; i++) {
        vec3 p = ro + rd * t;
        float hh, cd;
        float d = map(p, hh, cd);
        glow += hue_col(hh) * 0.03 / (1.0 + cd * cd * 0.6) * min(d, 1.0);
        if (d < 0.01 * t) { hit = true; hue = hh; capd = cd; break; }
        t += d * 0.8;
        if (t > 90.0) break;
    }
    if (hit) {
        vec3 p = ro + rd * t;
        float hh, cd;
        vec2 e = vec2(0.03, 0);
        vec3 n = normalize(vec3(map(p + e.xyy, hh, cd) - map(p - e.xyy, hh, cd), map(p + e.yxy, hh, cd) - map(p - e.yxy, hh, cd),
                                map(p + e.yyx, hh, cd) - map(p - e.yyx, hh, cd)));
        vec3 gc = hue_col(hue);
        vec3 c;
        if (capd < 0.05) {
            /* caps: dark tops spotted with light, glowing gills beneath */
            float spots = smoothstep(0.62, 0.7, n3(p.xz * 0.15 + p.y * 0.1));
            c = vec3(0.04, 0.03, 0.05) * (0.5 + 0.5 * n.y) + gc * spots * 1.5;
            if (n.y < 0.2) {
                float gill = 0.5 + 0.5 * sin(atan(p.z, p.x) * 140.0);
                c = gc * (0.9 + 0.8 * gill);
            }
        } else if (p.y < 0.6 + abs(n.x) * 0.2 && n.y > 0.6) {
            float moss = n3(p.xz * 0.5);
            c = vec3(0.02, 0.05, 0.03) * (0.5 + moss) + glow * 0.4;
        } else {
            c = vec3(0.12, 0.13, 0.1) * (0.3 + 0.2 * n3(vec2(p.y * 2.0, atan(p.z, p.x)))) * (glow * 2.0 + 0.05);
        }
        col = mix(c, fogc, 1.0 - exp(-t * 0.035));
    }
    col += glow;
    /* spores, rising */
    for (int k = 0; k < 2; k++) {
        vec2 g = vUV * vec2(uRes.x / uRes.y, 1.0) * (20.0 + 14.0 * float(k)) + vec2(0.0, -uT * (0.25 + 0.1 * float(k)));
        vec2 id = floor(g), f = fract(g) - 0.5;
        float h = h12(id + float(k) * 3.0);
        vec2 off = vec2(sin(uT * 0.5 + h * 9.0), cos(uT * 0.4 + h * 7.0)) * 0.25;
        float sp = smoothstep(0.09, 0.0, length(f - off)) * step(0.82, h);
        col += hue_col(fract(h * 3.0)) * sp * 0.8;
    }
    col = tone(col * 1.4);
    oCol = vec4(col, 1.0);
}
);

/* THE MERE: a glass house alight on a frozen lake, snow falling at dusk */
static const char *FS_AGE2 = "#version 330 core\n" NOISE AGE_COMMON GLSL(
float house(vec3 p) {
    vec3 q = p - vec3(0, 0, 0);
    float body = sdBox(q - vec3(0, 1.4, 0), vec3(3.2, 1.4, 2.0));
    vec3 r = q - vec3(0, 2.8, 0);
    float roof = max(max(abs(r.z) * 0.8 + r.y * 1.0 - 1.6, -r.y), abs(r.x) - 3.2);
    float d = min(body, roof * 0.7);
    /* a lantern on a post by the door */
    d = min(d, sdCyl(p - vec3(3.9, 0.8, 2.6), 0.05, 0.8));
    return d;
}
vec3 sky(vec3 d) {
    vec3 c = mix(vec3(0.85, 0.6, 0.6), vec3(0.18, 0.22, 0.42), pow(clamp(d.y * 2.5, 0.0, 1.0), 0.6));
    float sd = max(dot(d, normalize(vec3(-0.6, 0.05, -1.0))), 0.0);
    c += vec3(1.0, 0.6, 0.4) * pow(sd, 12.0) * 0.6;
    float cl = fbm(d.xz / max(d.y + 0.1, 0.05) * 0.08 + uT * 0.002);
    c = mix(c, c * vec3(0.8, 0.75, 0.85), smoothstep(0.5, 0.8, cl) * 0.7);
    return c;
}
vec3 interior(vec3 p, vec3 rd) {
    /* warm light, the shapes of plants against it */
    float plant = smoothstep(0.45, 0.6, fbm(p.xy * 0.8 + p.z * 0.3)) * smoothstep(2.4, 0.6, p.y);
    vec3 warm = vec3(1.0, 0.68, 0.32) * 2.2;
    return mix(warm, vec3(0.05, 0.09, 0.04), plant * 0.85);
}
vec3 shade_house(vec3 p, vec3 rd) {
    vec2 e = vec2(0.01, 0);
    vec3 n = normalize(vec3(house(p + e.xyy) - house(p - e.xyy), house(p + e.yxy) - house(p - e.yxy), house(p + e.yyx) - house(p - e.yyx)));
    if (length(p.xz - vec2(3.9, 2.6)) < 0.2) return vec3(0.03);
    /* the iron frame: a grid over each face */
    vec3 a = abs(n);
    vec2 uv = a.x > 0.5 ? p.zy : a.z > 0.5 ? p.xy : p.xz;
    vec2 g = abs(fract(uv / vec2(0.55, 0.5)) - 0.5);
    float frame = 1.0 - smoothstep(0.03, 0.06, min(g.x, g.y) * 0.55);
    vec3 e3 = abs(abs(p - vec3(0, 1.4, 0)) - vec3(3.2, 1.4, 2.0));
    float two = e3.x + e3.y + e3.z - max(e3.x, max(e3.y, e3.z));
    frame = max(frame, (1.0 - smoothstep(0.06, 0.12, two)) * step(p.y, 2.85));
    vec3 glass = interior(p + rd * 0.8, rd);
    vec3 r = reflect(rd, n);
    float fr = 0.06 + 0.9 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
    glass = mix(glass, sky(r), fr);
    /* snow on the roof */
    float snow = smoothstep(0.4, 0.9, n.y) * smoothstep(0.4, 0.6, n3(p.xz * 2.0) + 0.3);
    vec3 c = mix(glass, vec3(0.06, 0.07, 0.08), frame);
    return mix(c, vec3(0.8, 0.8, 0.9) * 0.7, snow);
}
void main() {
    float a = sin(uT * 0.02) * 0.3;
    vec3 ro = vec3(sin(a) * 15.0, 1.7, cos(a) * 15.0);
    vec3 rd = cam_ray(vUV, ro, vec3(0, 1.8, 0), 1.6);
    vec3 col = sky(rd);
    /* the far shore: dark pines on snowy hills */
    float az = atan(rd.x, -rd.z);
    float hill = 0.035 + 0.03 * n1(vec2(az * 0.3, 1.0));
    float pines = hill + 0.02 * pow(abs(sin(az * 90.0 + n2(vec2(az, 0.0)) * 30.0)), 3.0) * smoothstep(0.4, 0.6, n2(vec2(az * 0.6, 2.0)));
    if (rd.y < pines && rd.y > 0.0) col = mix(vec3(0.07, 0.08, 0.12), vec3(0.5, 0.48, 0.58), smoothstep(pines - 0.01, hill - 0.015, rd.y) * 0.7);
    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 70; i++) {
        float d = house(ro + rd * t);
        if (d < 0.002) { hit = t; break; }
        t += d;
        if (t > 60.0) break;
    }
    float ti = rd.y < 0.0 ? -ro.y / rd.y : 1e9;
    if (hit > 0.0 && hit < ti) col = shade_house(ro + rd * hit, rd);
    else if (rd.y < 0.0) {
        vec3 p = ro + rd * ti;
        /* ice: cracked, dusted with snow, holding the house's light */
        float cr = abs(n1(p.xz * 0.05) - 0.5);
        float crack = 1.0 - smoothstep(0.0, 0.012, cr);
        float snow = smoothstep(0.5, 0.75, fbm(p.xz * 0.12));
        vec3 r = reflect(rd, vec3(0, 1, 0));
        vec3 ice = mix(vec3(0.16, 0.2, 0.28), sky(r) * 0.8, 0.5);
        float tr = 0.0, hr = -1.0;
        vec3 rp = p + r * 0.01;
        for (int i = 0; i < 40; i++) {
            float d = house(rp + r * tr);
            if (d < 0.01) { hr = tr; break; }
            tr += d;
            if (tr > 40.0) break;
        }
        if (hr > 0.0) ice = mix(ice, shade_house(rp + r * hr, r) * 0.6, 0.85);
        ice = mix(ice, vec3(0.75, 0.75, 0.85), snow * 0.7);
        ice = mix(ice, vec3(0.05, 0.06, 0.1), crack * 0.6);
        col = mix(ice, sky(vec3(rd.x, 0.0, rd.z)), 1.0 - exp(-ti * 0.02));
    }
    /* the lantern */
    vec3 lp = vec3(3.9, 1.7, 2.6);
    vec3 lv = lp - ro;
    float lt = dot(lv, rd);
    if (lt > 0.0 && (hit < 0.0 || lt < hit + 0.3)) col += vec3(1.0, 0.7, 0.35) * 0.05 / (0.002 + dot(lv - rd * lt, lv - rd * lt));
    /* snow falling */
    for (int k = 0; k < 3; k++) {
        float sc = 16.0 + 12.0 * float(k);
        vec2 g = vUV * vec2(uRes.x / uRes.y, 1.0) * sc + vec2(sin(uT * 0.3 + float(k)) * 0.5, uT * (0.4 + 0.25 * float(k)));
        vec2 id = floor(g), f = fract(g) - 0.5;
        float h = h12(id + float(k) * 11.0);
        vec2 off = vec2(sin(uT + h * 20.0) * 0.25, (h - 0.5) * 0.5);
        col += vec3(0.85, 0.85, 0.95) * smoothstep(0.06 + 0.02 * float(k), 0.0, length(f - off)) * step(0.5, h) * 0.6;
    }
    col = tone(col * 1.25);
    oCol = vec4(col, 1.0);
}
);

/* THE DRIFT: islands of rock afloat over a sea of cloud, at sunset */
static const char *FS_AGE3 = "#version 330 core\n" NOISE AGE_COMMON GLSL(
const vec3 SUN = normalize(vec3(0.0, 0.06, -1.0));
vec4 isl[7];
float island(vec3 p, vec4 c) {
    vec3 q = p - c.xyz;
    float R = c.w, r = length(q.xz);
    float n = fbm(q.xz * 0.06 / (R * 0.1) + c.x) * R * 0.35;
    float top = q.y - (0.1 * R + n * 0.2);
    float under = -q.y - (R - r) * 1.7 - n;
    float side = r - R - n * 0.5;
    float d = max(max(top, under), side) * 0.55;
    /* trees on top */
    for (int i = 0; i < 3; i++) {
        vec2 o = vec2(cos(c.w * 3.0 + float(i) * 2.1), sin(c.w * 3.0 + float(i) * 2.1)) * R * 0.45;
        vec3 tq = q - vec3(o.x, 0.1 * R + R * 0.12, o.y);
        d = min(d, length(tq * vec3(1.0, 0.7, 1.0)) - R * 0.09);
    }
    return d;
}
float map(vec3 p) {
    float d = 1e9;
    for (int i = 0; i < 7; i++) d = min(d, island(p, isl[i]));
    return d;
}
vec3 sky(vec3 d) {
    float y = d.y;
    vec3 c = mix(vec3(1.0, 0.5, 0.25), vec3(0.25, 0.22, 0.45), smoothstep(-0.05, 0.45, y));
    c = mix(c, vec3(0.08, 0.08, 0.2), smoothstep(0.4, 1.0, y));
    float s = max(dot(d, SUN), 0.0);
    c += vec3(1.0, 0.7, 0.4) * (pow(s, 8.0) * 0.6 + pow(s, 300.0) * 6.0);
    float band = fbm(vec2(atan(d.x, -d.z) * 2.0, y * 14.0) + uT * 0.002);
    c = mix(c, c * vec3(1.15, 0.8, 0.8), smoothstep(0.55, 0.7, band) * smoothstep(0.3, 0.05, abs(y - 0.12)));
    return c;
}
void main() {
    for (int i = 0; i < 7; i++) {
        float h = float(i);
        isl[i] = vec4(sin(h * 2.4) * (20.0 + h * 9.0), sin(uT * 0.25 + h) * 0.6 + (h - 3.0) * 2.0 + 2.0,
                      -30.0 - h * 22.0, 4.0 + 4.0 * fract(h * 0.618 + 0.3));
    }
    isl[0] = vec4(-6.0, -4.0 + sin(uT * 0.3) * 0.4, -18.0, 7.0);
    vec3 ro = vec3(sin(uT * 0.04) * 4.0, 3.0, 8.0);
    vec3 rd = cam_ray(vUV, ro, vec3(0, -2.0, -60.0), 1.5);
    vec3 col = sky(rd);
    /* the sea of cloud below */
    float tc = rd.y < 0.0 ? (-28.0 - ro.y) / rd.y : 1e9;
    if (tc < 1e8) {
        vec3 p = ro + rd * tc;
        float c = fbm(p.xz * 0.012 + vec2(uT * 0.004, 0.0));
        vec3 cc = mix(vec3(0.35, 0.25, 0.4), vec3(1.0, 0.62, 0.4), smoothstep(0.35, 0.75, c));
        col = mix(cc, sky(vec3(rd.x, 0.0, rd.z)), 1.0 - exp(-tc * 0.004));
    }
    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 90; i++) {
        float d = map(ro + rd * t);
        if (d < 0.003 * t) { hit = t; break; }
        t += d;
        if (t > 260.0) break;
    }
    if (hit > 0.0 && hit < tc) {
        vec3 p = ro + rd * hit;
        vec2 e = vec2(0.02 * hit * 0.1 + 0.01, 0);
        vec3 n = normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
        float grass = smoothstep(0.5, 0.8, n.y);
        vec3 a = mix(vec3(0.3, 0.22, 0.18) * (0.7 + 0.5 * n3(p.xy * 0.4)), vec3(0.18, 0.28, 0.1), grass);
        float dif = max(dot(n, normalize(SUN + vec3(0, 0.25, 0))), 0.0);
        vec3 c = a * (dif * vec3(1.4, 0.85, 0.55) + (0.5 + 0.5 * n.y) * vec3(0.25, 0.2, 0.35) + max(-n.y, 0.0) * vec3(0.3, 0.2, 0.25));
        col = mix(c, sky(rd) * 0.95, 1.0 - exp(-hit * 0.008));
    }
    col = tone(col * 1.15);
    oCol = vec4(col, 1.0);
}
);

/* VERDIGRIS: copper towers in a desert of dunes, under a white sun */
static const char *FS_AGE4 = "#version 330 core\n" NOISE AGE_COMMON GLSL(
const vec3 SUN = normalize(vec3(0.6, 0.5, -0.6));
float dune(vec2 p) {
    float a = p.x * 0.045 + sin(p.y * 0.03) * 1.6 + vn(p * 0.05) * 4.0;
    float sa = sin(a);
    float d = 1.0 - sqrt(sa * sa + 0.03);
    return d * d * 6.0 + vn(p * 0.02) * 6.0 - 2.0;
}
float tower(vec3 p, out float id) {
    vec2 c = floor(p.xz / 50.0);
    id = 0.0;
    float d = 1e9;
    for (int i = 0; i < 4; i++) {
        vec2 cc = c + vec2(i % 2, i / 2) - 0.5;
        float h = h12(cc);
        if (h < 0.35) continue;
        vec2 o = (cc + 0.5 + (vec2(h, h12(cc + 3.0)) - 0.5) * 0.6) * 50.0;
        float H = 20.0 + 40.0 * h12(cc + 9.0), R = 1.4 + 1.6 * h12(cc + 5.0);
        vec3 q = p - vec3(o.x, 0, o.y);
        float r = R * (1.0 - 0.35 * clamp(q.y / H, 0.0, 1.0));
        float body = sdCyl(q - vec3(0, H * 0.5, 0), r, H * 0.5);
        float spire = sdCone(q - vec3(0, H, 0), r * 0.7, H * 0.25);
        float rings = sdCyl(vec3(q.x, mod(q.y, 6.0) - 3.0, q.z), r + 0.2, 0.25);
        rings = max(rings, q.y - H);
        float dd = min(min(body, spire), rings);
        if (dd < d) { d = dd; id = h; }
    }
    return d;
}
vec3 sky(vec3 d) {
    vec3 c = mix(vec3(0.92, 0.82, 0.66), vec3(0.45, 0.62, 0.8), pow(clamp(d.y, 0.0, 1.0), 0.5));
    float s = max(dot(d, SUN), 0.0);
    c += vec3(1.0, 0.95, 0.8) * (pow(s, 6.0) * 0.4 + pow(s, 400.0) * 4.0);
    return c;
}
void main() {
    vec2 uv = vUV;
    /* heat shimmer near the ground line */
    uv.x += (n3(vec2(uv.y * 6.0, uT * 0.6)) - 0.5) * 0.006 * smoothstep(0.65, 0.3, uv.y);
    vec3 ro = vec3(uT * 2.0, 6.0, 0.0);
    ro.y = max(dune(ro.xz), 0.0) + 9.0;
    vec3 rd = cam_ray(uv, ro, ro + vec3(-3.0, 0.4, -20.0), 1.4);
    vec3 col = sky(rd);
    float t = 0.5, hit = -1.0, kind = 0.0, id = 0.0;
    for (int i = 0; i < 110; i++) {
        vec3 p = ro + rd * t;
        float tid;
        float dt = tower(p, tid);
        float dg = (p.y - dune(p.xz)) * 0.35;
        float d = min(dt, dg);
        if (d < 0.003 * t) { hit = t; kind = dt < dg ? 1.0 : 0.0; id = tid; break; }
        t += d;
        if (t > 500.0) break;
    }
    if (hit > 0.0) {
        vec3 p = ro + rd * hit;
        vec3 n;
        vec3 c;
        if (kind > 0.5) {
            vec2 e = vec2(0.02, 0);
            float q1, q2, q3, q4, q5, q6;
            n = normalize(vec3(tower(p + e.xyy, q1) - tower(p - e.xyy, q2), tower(p + e.yxy, q3) - tower(p - e.yxy, q4),
                               tower(p + e.yyx, q5) - tower(p - e.yyx, q6)));
            float streak = smoothstep(0.45, 0.7, n2(vec2(atan(n.z, n.x) * 3.0 + id * 10.0, p.y * 0.02)) + 0.2 * n3(p.xz + p.y * 0.1));
            vec3 copper = vec3(0.8, 0.36, 0.14), verd = vec3(0.2, 0.62, 0.5);
            vec3 a = mix(copper, verd, streak);
            float dif = max(dot(n, SUN), 0.0);
            vec3 r = reflect(rd, n);
            float spec = pow(max(dot(r, SUN), 0.0), 30.0) * (1.0 - streak);
            c = a * (dif * 1.5 + 0.3) + copper * spec * 3.0 + sky(r) * 0.1 * (1.0 - streak);
        } else {
            vec2 e = vec2(0.1, 0);
            n = normalize(vec3(dune(p.xz - e.xy) - dune(p.xz + e.xy), 2.0 * e.x, dune(p.xz - e.yx) - dune(p.xz + e.yx)));
            float rip = sin(p.x * 3.0 + vn(p.xz * 0.15) * 10.0) * 0.5 + 0.5;
            vec3 a = vec3(0.86, 0.62, 0.4) * (0.94 + 0.08 * rip * exp(-hit * 0.02));
            float dif = max(dot(n, SUN), 0.0);
            c = a * (dif * 1.5 + 0.25 * vec3(0.6, 0.7, 1.0));
        }
        col = mix(c, sky(vec3(rd.x, 0.05, rd.z)), 1.0 - exp(-hit * (kind > 0.5 ? 0.0028 : 0.005)));
    }
    /* blown sand */
    float sand = fbm(vec2(uv.x * 3.0 - uT * 0.12, uv.y * 14.0)) * smoothstep(0.5, 0.15, uv.y);
    col = mix(col, vec3(0.9, 0.75, 0.55), sand * 0.25);
    col = tone(col * 1.1);
    oCol = vec4(col, 1.0);
}
);

/* ---- GL plumbing ---- */
static GLuint compile(GLenum type, const char *src, const char *name) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "library: shader %s:\n%s\n", name, log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *vs, const char *fs, const char *name) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, vs, name));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs, name));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "library: link %s:\n%s\n", name, log);
        exit(1);
    }
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum filt, GLenum wrap) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)wrap);
    return t;
}
static GLuint mk_fbo(GLuint col, GLuint depth) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    if (col) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
    else { glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE); }
    if (depth) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete");
    return f;
}

static GLuint p_scene, p_shadow, p_flame, p_steam, p_dust, p_vol, p_down, p_up, p_comp, p_paper, p_ink, p_age[NAGE];
static GLuint vao_room, vbo_room, ibo_room, vao_leaf, vbo_leaf, vao_fx, vbo_fx, vao_empty, vao_ink;
static GLuint tex_noise, tex_noise3, tex_ao, tex_atlas;
#define KEY_SM 2048
static GLuint tex_keysh, fbo_keysh;
#define FIRE_SM 512
#define LAMP_SM 1024
static GLuint tex_firesh, tex_lampsh;
#define PW 1024
#define PH 1434
#define NPAGE 6
static GLuint tex_page[NPAGE], fbo_page[NPAGE];
#define AGE_W 640
#define AGE_H 534
static GLuint tex_age[2], fbo_age[2];  /* live, and the one being turned away */
static GLuint tex_scene, tex_depth, fbo_scene, tex_vol, fbo_vol;
#define NBLOOM 5
static GLuint tex_bloom[NBLOOM], fbo_bloom[NBLOOM];
static int bloom_w[NBLOOM], bloom_h[NBLOOM];
static int scene_w, scene_h, vol_w, vol_h;
static float render_scale = 1.0f;
static const float panel_uv[4] = {0.13f, 0.09f, 0.87f, 0.53f};

/* tiling smooth value noise, cells a tile */
static float *vnoise_tile(int n, int cells, uint32_t salt) {
    float *g = malloc(sizeof(float) * cells * cells), *o = malloc(sizeof(float) * n * n);
    for (int i = 0; i < cells * cells; i++) g[i] = (hash32((uint32_t)i * 2654435761u ^ salt) & 0xffff) / 65535.0f;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float fx = (float)x * cells / n, fy = (float)y * cells / n;
            int ix = (int)fx, iy = (int)fy;
            float tx = fx - ix, ty = fy - iy;
            tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
            int x1 = (ix + 1) % cells, y1 = (iy + 1) % cells;
            float a = g[iy * cells + ix], b = g[iy * cells + x1], c = g[y1 * cells + ix], d = g[y1 * cells + x1];
            o[y * n + x] = mixf(mixf(a, b, tx), mixf(c, d, tx), ty);
        }
    free(g);
    return o;
}

static void noise_init(void) {
    int n = 256;
    float *a = vnoise_tile(n, 16, 11), *b = vnoise_tile(n, 32, 22), *c = vnoise_tile(n, 64, 33);
    unsigned char *px = malloc((size_t)n * n * 4);
    for (int i = 0; i < n * n; i++) {
        px[i * 4 + 0] = (unsigned char)(a[i] * 255);
        px[i * 4 + 1] = (unsigned char)(b[i] * 255);
        px[i * 4 + 2] = (unsigned char)(c[i] * 255);
        px[i * 4 + 3] = (unsigned char)(hash32((uint32_t)i) & 255);
    }
    tex_noise = mk_tex(GL_RGBA8, n, n, GL_RGBA, GL_UNSIGNED_BYTE, px, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    free(a); free(b); free(c); free(px);
    /* a 3D one for the air */
    int m = 32, cells = 8;
    float *g = malloc(sizeof(float) * cells * cells * cells);
    for (int i = 0; i < cells * cells * cells; i++) g[i] = (hash32((uint32_t)i * 747796405u + 7) & 0xffff) / 65535.0f;
    unsigned char *v = malloc((size_t)m * m * m);
    for (int z = 0; z < m; z++)
        for (int y = 0; y < m; y++)
            for (int x = 0; x < m; x++) {
                float f[3] = {(float)x * cells / m, (float)y * cells / m, (float)z * cells / m};
                int i0[3], i1[3];
                float t[3];
                for (int k = 0; k < 3; k++) {
                    i0[k] = (int)f[k]; i1[k] = (i0[k] + 1) % cells;
                    t[k] = f[k] - i0[k]; t[k] = t[k] * t[k] * (3 - 2 * t[k]);
                }
#define G3(a, b, c) g[((c) * cells + (b)) * cells + (a)]
                float c00 = mixf(G3(i0[0], i0[1], i0[2]), G3(i1[0], i0[1], i0[2]), t[0]);
                float c10 = mixf(G3(i0[0], i1[1], i0[2]), G3(i1[0], i1[1], i0[2]), t[0]);
                float c01 = mixf(G3(i0[0], i0[1], i1[2]), G3(i1[0], i0[1], i1[2]), t[0]);
                float c11 = mixf(G3(i0[0], i1[1], i1[2]), G3(i1[0], i1[1], i1[2]), t[0]);
#undef G3
                float val = mixf(mixf(c00, c10, t[1]), mixf(c01, c11, t[1]), t[2]);
                v[((size_t)z * m + y) * m + x] = (unsigned char)(val * 255);
            }
    glGenTextures(1, &tex_noise3);
    glBindTexture(GL_TEXTURE_3D, tex_noise3);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m, m, m, 0, GL_RED, GL_UNSIGNED_BYTE, v);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    free(g); free(v);
}

/* Ambient occlusion: the room's surfaces scattered into a voxel grid and
 * blurred; the shaders read how crowded the air is just off a surface. */
#define AOX 116
#define AOY 100
#define AOZ 116
static void ao_build(void) {
    float *v = calloc((size_t)AOX * AOY * AOZ, sizeof(float));
    uint32_t h = 12345;
    for (int t = 0; t < mesh.ni; t += 3) {
        const float *a = mesh.v + (size_t)mesh.ix[t] * VF, *b = mesh.v + (size_t)mesh.ix[t + 1] * VF, *c = mesh.v + (size_t)mesh.ix[t + 2] * VF;
        int m = (int)(a[11] + 0.5f);
        if (m == MT_SKY || m == MT_LAMPGLASS) continue;
        V3 A = v3(a[0], a[1], a[2]), B = v3(b[0], b[1], b[2]), C = v3(c[0], c[1], c[2]);
        float area = 0.5f * vlen(vcross(vsub(B, A), vsub(C, A)));
        int n = (int)(area / (0.04f * 0.04f)) + 2;
        if (n > 200000) n = 200000;
        for (int i = 0; i < n; i++) {
            float r1 = hue_rand(&h), r2 = hue_rand(&h);
            if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
            V3 p = vadd(A, vadd(vmul(vsub(B, A), r1), vmul(vsub(C, A), r2)));
            /* window holes are cut in the shader: skip them here too */
            if (m == MT_STONE && a[12] > 0.5f && a[12] < 1.5f) {
                float az = atan2f(p.x, -p.z);
                int hole = 0;
                for (int w = 0; w < NWIN; w++) {
                    float da = remainderf(az - win[w].a, 2 * PI), s = da * RIN;
                    if (fabsf(s) < win[w].wi && p.y > win[w].y0 &&
                        (p.y < win[w].y1 || s * s + (p.y - win[w].y1) * (p.y - win[w].y1) < win[w].wi * win[w].wi)) hole = 1;
                }
                if (hole) continue;
            }
            int x = (int)((p.x + 5.8f) / 11.6f * AOX), y = (int)(p.y / 10.0f * AOY), z = (int)((p.z + 5.8f) / 11.6f * AOZ);
            if (x < 0 || y < 0 || z < 0 || x >= AOX || y >= AOY || z >= AOZ) continue;
            v[((size_t)z * AOY + y) * AOX + x] = 1;
        }
    }
    /* three box blurs along each axis */
    float *tmp = malloc(sizeof(float) * AOX * AOY * AOZ);
    int dims[3] = {AOX, AOY, AOZ};
    size_t stride[3] = {1, AOX, (size_t)AOX * AOY};
    for (int pass = 0; pass < 3; pass++)
        for (int ax = 0; ax < 3; ax++) {
            int R = 2, n = dims[ax];
            memcpy(tmp, v, sizeof(float) * AOX * AOY * AOZ);
            for (int z = 0; z < AOZ; z++)
                for (int y = 0; y < AOY; y++)
                    for (int x = 0; x < AOX; x++) {
                        int c[3] = {x, y, z};
                        if (c[ax] != 0) continue;
                        size_t base = (size_t)z * stride[2] + (size_t)y * stride[1] + (size_t)x;
                        float acc = 0;
                        for (int k = -R; k <= R; k++) acc += tmp[base + (size_t)clampf((float)k, 0, (float)(n - 1)) * stride[ax]];
                        for (int i = 0; i < n; i++) {
                            v[base + (size_t)i * stride[ax]] = acc / (2 * R + 1);
                            int ia = i - R < 0 ? 0 : i - R, ib = i + R + 1 > n - 1 ? n - 1 : i + R + 1;
                            acc += tmp[base + (size_t)ib * stride[ax]] - tmp[base + (size_t)ia * stride[ax]];
                        }
                    }
        }
    free(tmp);
    glGenTextures(1, &tex_ao);
    glBindTexture(GL_TEXTURE_3D, tex_ao);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, AOX, AOY, AOZ, 0, GL_RED, GL_FLOAT, v);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    free(v);
}

static void set_wins(GLuint p) {
    float w[NWIN][4];
    for (int i = 0; i < NWIN; i++) { w[i][0] = win[i].a; w[i][1] = win[i].y0; w[i][2] = win[i].y1; w[i][3] = win[i].wi; }
    glUniform4fv(U(p, "uWin"), NWIN, &w[0][0]);
}

static void draw_room(void) {
    glBindVertexArray(vao_room);
    glDrawElements(GL_TRIANGLES, mesh.ni, GL_UNSIGNED_INT, NULL);
}

/* the point lights' shadows: their surroundings never move, so once */
static GLuint cube_shadow(V3 pos, int size, int skip) {
    GLuint t, f;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_CUBE_MAP, t);
    for (int i = 0; i < 6; i++)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLenum)i, 0, GL_DEPTH_COMPONENT24, size, size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    static const V3 dirs[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const V3 ups[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    M4 P = m4_frustum(-0.04f, 0.04f, -0.04f, 0.04f, 0.04f, 14.0f);
    glUseProgram(p_shadow);
    set_wins(p_shadow);
    glUniform1i(U(p_shadow, "uSkip"), skip);
    glViewport(0, 0, size, size);
    glEnable(GL_DEPTH_TEST);
    for (int i = 0; i < 6; i++) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLenum)i, t, 0);
        glClear(GL_DEPTH_BUFFER_BIT);
        M4 vp = m4_mul(P, m4_look(pos, dirs[i], ups[i]));
        glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, vp.m);
        draw_room();
    }
    glDeleteFramebuffers(1, &f);
    return t;
}

/* ---- the light of the day ---- */
static struct {
    V3 key_dir, key_col, sun_dir, moon_dir, sun_col, sky_zen, sky_hor, win_l, sky_amb, gnd_amb, bounce;
    float sun_el, night, moon_phase, exposure, lamp_on;
} lt;
static M4 key_vp;
static V3 key_dir_drawn;
static int key_drawn;

/* the windows face the sea, south; the middle one due south */
#define SOUTH_AZ (62 * DEG)
#define LATITUDE (52.0f * DEG)

static V3 compass_dir(float el, float az_compass) {
    float a = az_compass - PI + SOUTH_AZ;   /* compass bearing -> room azimuth */
    return v3(cosf(el) * sinf(a), sinf(el), -cosf(el) * cosf(a));
}

static void sky_light(double wt) {
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    float hour = opt_hour >= 0 ? opt_hour : tm.tm_hour + tm.tm_min / 60.0f + tm.tm_sec / 3600.0f;
    float solar = hour - (tm.tm_isdst > 0 ? 1 : 0);
    float decl = -23.44f * DEG * cosf(2 * PI / 365 * (tm.tm_yday + 10));
    float H = (solar - 12) * 15 * DEG;
    float sel = sinf(LATITUDE) * sinf(decl) + cosf(LATITUDE) * cosf(decl) * cosf(H);
    float el = asinf(sel);
    float az = atan2f(sinf(H), cosf(H) * sinf(LATITUDE) - tanf(decl) * cosf(LATITUDE)) + PI;
    lt.sun_el = el;
    lt.sun_dir = compass_dir(el, az);
    /* the moon: its phase from a known new moon, lagging the sun by it */
    double days = (wt - 947182440.0) / 86400.0;
    float ph = (float)fmod(days / 29.530589, 1.0);
    if (ph < 0) ph += 1;
    lt.moon_phase = 0.5f - 0.5f * cosf(ph * 2 * PI);
    float Hm = H - ph * 2 * PI;
    float dm = decl * cosf(ph * 2 * PI) * -1;
    float mel = asinf(sinf(LATITUDE) * sinf(dm) + cosf(LATITUDE) * cosf(dm) * cosf(Hm));
    float maz = atan2f(sinf(Hm), cosf(Hm) * sinf(LATITUDE) - tanf(dm) * cosf(LATITUDE)) + PI;
    lt.moon_dir = compass_dir(mel, maz);

    float s = sel;
    float day = smoothf(0.0f, 0.3f, s), gold = smoothf(-0.08f, 0.02f, s) * (1 - day);
    float twi = smoothf(-0.22f, -0.04f, s);
    lt.night = 1 - twi;
    V3 zen_d = v3(0.20f, 0.33f, 0.55f), hor_d = v3(0.62f, 0.68f, 0.72f);
    V3 zen_g = v3(0.17f, 0.2f, 0.38f), hor_g = v3(0.95f, 0.52f, 0.3f);
    V3 zen_n = v3(0.0035f, 0.005f, 0.012f), hor_n = v3(0.01f, 0.014f, 0.028f);
    float moonlit = 1 + 3 * lt.moon_phase * smoothf(-0.05f, 0.2f, sinf(mel));
    zen_n = vmul(zen_n, moonlit); hor_n = vmul(hor_n, moonlit);
    V3 zen = vlerp(zen_n, vlerp(zen_g, zen_d, day), twi), hor = vlerp(hor_n, vlerp(hor_g, hor_d, day), twi);
    if (gold > 0 && day < 1) { zen = vlerp(zen, zen_g, gold * 0.5f); hor = vlerp(hor, hor_g, gold * 0.8f); }
    float bright = 1.6f * mixf(0.08f, 1.0f, smoothf(-0.1f, 0.15f, s));
    lt.sky_zen = vmul(zen, bright);
    lt.sky_hor = vmul(hor, bright);
    /* the sun reddens towards the horizon */
    float hi = smoothf(0.0f, 0.4f, s);
    lt.sun_col = vmul(vlerp(v3(1.0f, 0.42f, 0.16f), v3(1.0f, 0.92f, 0.8f), hi), 7.0f * smoothf(-0.03f, 0.05f, s));
    if (s > -0.03f) { lt.key_dir = lt.sun_dir; lt.key_col = lt.sun_col; }
    else if (mel > 0.03f) {
        lt.key_dir = lt.moon_dir;
        lt.key_col = vmul(v3(0.42f, 0.52f, 0.85f), 0.3f * lt.moon_phase * smoothf(0.03f, 0.2f, sinf(mel)) * (1 - twi * 0.8f));
    } else { lt.key_dir = v3(0, 1, 0); lt.key_col = v3(0, 0, 0); }
    /* light through the windows and about the room */
    V3 sky = vlerp(lt.sky_hor, lt.sky_zen, 0.35f);
    lt.win_l = vmul(sky, 1.6f);
    lt.sky_amb = vmul(sky, 0.05f);
    lt.gnd_amb = vadd(vmul(sky, 0.03f), v3(0.004f, 0.0025f, 0.0012f));
    /* sun on the floor lights the whole room a little, warm */
    lt.bounce = vmul(vmul(lt.key_col, 0.018f), s > 0 ? 1.0f : 0.4f);
    lt.bounce.z *= 0.8f;
    float lum = 0.3f * lt.win_l.x + 0.55f * lt.win_l.y + 0.15f * lt.win_l.z;
    lt.exposure = 1.5f / (0.14f + lum);
    if (lt.exposure > 3.4f) lt.exposure = 3.4f;
    lt.lamp_on = 1 - smoothf(0.05f, 0.2f, s);
}

static float fire_flicker(double t) {
    float f = 0.82f + 0.07f * sinf((float)t * 7.3f) + 0.05f * sinf((float)t * 13.1f + 1.3f) + 0.04f * sinf((float)t * 23.7f + 0.4f);
    f += 0.06f * sinf((float)t * 1.7f) * sinf((float)t * 0.37f);
    return f;
}

static void key_shadow_render(void) {
    V3 d = lt.key_dir;
    V3 c = v3(0, 3.0f, 0);
    V3 eye = vadd(c, vmul(d, 25));
    V3 up = fabsf(d.y) > 0.95f ? v3(1, 0, 0) : v3(0, 1, 0);
    M4 view = m4_look(eye, vmul(d, -1), up);
    M4 proj = m4_ortho(-6.6f, 6.6f, -6.6f, 6.6f, 10.0f, 42.0f);
    key_vp = m4_mul(proj, view);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_keysh);
    glViewport(0, 0, KEY_SM, KEY_SM);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.5f, 2.0f);
    glUseProgram(p_shadow);
    set_wins(p_shadow);
    glUniform1i(U(p_shadow, "uSkip"), MT_LAMPGLASS);
    glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, key_vp.m);
    draw_room();
    glDisable(GL_POLYGON_OFFSET_FILL);
    key_dir_drawn = d;
    key_drawn = 1;
}

/* ---- the pages ---- */
static int st_head, st_body, st_cap, st_note, st_num;
typedef struct { GLuint vbo; int n; } TextBuf;
static TextBuf txt_jcur, txt_jprev, txt_cap[2], txt_fill[2];
static long txt_period = -999999;

static void textbuf_set(TextBuf *b) {
    if (!b->vbo) glGenBuffers(1, &b->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(tvn * sizeof(float)), tv, GL_STATIC_DRAW);
    b->n = tvn / TVF;
    tvn = 0;
}

/* lay out a journal entry; the pen starts at t0 (timeline seconds) */
static void layout_journal(TextBuf *b, double wt, long period, float t0) {
    char head[JMAX], body[JMAX];
    compose_journal(head, body, wt, period, age_of(period));
    tvn = 0;
    uint32_t sd = hash32((uint32_t)period * 31u + opt_seed);
    float x0 = PW * 0.13f, w = PW * 0.8f, lead = styles[st_body].px * 1.38f;
    float y = PH * 0.075f + styles[st_head].px;
    char Host[64];
    snprintf(Host, sizeof Host, "%s", host);
    Host[0] = (char)toupper((unsigned char)Host[0]);
    float t = hand_line(st_head, x0, y, Host, t0, 13, 0, sd, 1.0f) + 0.4f;
    float hw = text_width(st_head, head);
    t = hand_line(st_head, x0 + w - hw - 10, y + styles[st_head].px * 1.15f, head, t, 13, -0.004f, sd + 7, 1.0f) + 0.8f;
    y = PH * 0.2f;
    const char *p = body;
    while (*p) {
        char para[JMAX];
        size_t n = strcspn(p, "\n");
        memcpy(para, p, n);
        para[n] = 0;
        t = hand_para(st_body, x0, &y, w, lead, para, t, 13, sd = hash32(sd), 1.0f, 1) + 0.6f;
        y += lead * 0.35f;
        p += n;
        if (*p == '\n') p++;
    }
    textbuf_set(b);
}

static void layout_plate(TextBuf *b, int age) {
    tvn = 0;
    const char *nm = age_name[age];
    float y = panel_uv[3] * PH + 105;
    char up[64];
    int k = 0;
    for (const char *p = nm; *p && k < 60; p++) {
        up[k++] = (char)toupper((unsigned char)*p);
        if (p[1]) up[k++] = ' ';
    }
    up[k] = 0;
    float w = text_width(st_cap, up);
    hand_line(st_cap, PW * 0.5f - w / 2, y, up, -1000, 50, 0, 3, 1.0f);
    float yy = y + 115;
    hand_para(st_note, PW * 0.17f, &yy, PW * 0.66f, styles[st_note].px * 1.4f, age_note[age], -1000, 50, 77 + (uint32_t)age, 0.95f, 0);
    char num[16];
    snprintf(num, sizeof num, "%d", 117 + age * 23 + (int)(opt_seed % 7));
    hand_line(st_num, PW * 0.88f - text_width(st_num, num), PH * 0.95f, num, -1000, 50, 0, 1, 0.8f);
    textbuf_set(b);
}

static void layout_filler(TextBuf *b, int which) {
    static const char *a[] = {
        "\xe2\x80\xa6 and the glass fell all morning. By noon the sea had come up white along the whole of the point, and the gulls sat it out on the roof.",
        "Mended the clasp on the blue folio. Found a pressed flower in it I do not remember pressing.",
        "The ladder had moved again in the night. I have stopped asking.",
        "Copied the tide tables for the month. Wind south-west, freshening.",
    };
    static const char *b2[] = {
        "A light out on the water after dark, steady, too low for a ship. Gone by morning.",
        "The picture in the book was different today. I am sure of it. I have marked the page with the red ribbon.",
        "Swept the stair. Wound the clock. Read until the candle failed.",
        "Rain at four. The ink will not dry.",
    };
    const char **s = which ? b2 : a;
    tvn = 0;
    float y = PH * 0.1f, x0 = PW * 0.13f, w = PW * 0.8f, lead = styles[st_body].px * 1.38f;
    uint32_t sd = 555 + (uint32_t)which;
    for (int i = 0; i < 4 && y < PH * 0.9f; i++) {
        hand_para(st_body, x0, &y, w, lead, s[i], -1000, 50, sd = hash32(sd), 0.8f, 1);
        y += lead * 0.6f;
    }
    textbuf_set(b);
}

static void draw_text(TextBuf *b, float t) {
    if (!b->n) return;
    glBindVertexArray(vao_ink);
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, TVF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, TVF * sizeof(float), (void *)(2 * sizeof(float)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, TVF * sizeof(float), (void *)(4 * sizeof(float)));
    glUseProgram(p_ink);
    glUniform2f(U(p_ink, "uSize"), PW, PH);
    glUniform1i(U(p_ink, "uAtlas"), 0);
    glUniform2f(U(p_ink, "uAtlasSize"), AW, (float)atlas_h);
    glUniform1f(U(p_ink, "uT"), t);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColorMask(1, 1, 1, 0);
    glUniform1i(U(p_ink, "uWet"), 0);
    glDrawArrays(GL_TRIANGLES, 0, b->n);
    glColorMask(0, 0, 0, 1);
    glBlendFunc(GL_ONE, GL_ONE);
    glUniform1i(U(p_ink, "uWet"), 1);
    glDrawArrays(GL_TRIANGLES, 0, b->n);
    glColorMask(1, 1, 1, 1);
    glDisable(GL_BLEND);
}

/* kind: 0 journal, 1 plate, 2 blank journal, 3/4 older pages */
static void render_page(int layer, int kind, float seed, TextBuf *b, float t) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_page[layer]);
    glViewport(0, 0, PW, PH);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(p_paper);
    glUniform1i(U(p_paper, "uKind"), kind);
    glUniform4fv(U(p_paper, "uPanel"), 1, panel_uv);
    float lead = styles[st_body].px * 1.38f;
    glUniform2f(U(p_paper, "uRule"), (PH * 0.2f + 6) / PH + (kind == 3 ? -0.1f : 0), lead / PH);
    glUniform1f(U(p_paper, "uSeed"), seed);
    glUniform2f(U(p_paper, "uSize"), PW, PH);
    glUniform1i(U(p_paper, "uNoise"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    if (b) draw_text(b, t);
    glBindTexture(GL_TEXTURE_2D, tex_page[layer]);
    glGenerateMipmap(GL_TEXTURE_2D);
}

/* ---- the timeline ---- */
typedef struct { long period; double u; } Tl;
static Tl timeline(double t) {
    double tl = t + PHASE0;
    Tl r;
    r.period = (long)floor(tl / PERIOD);
    r.u = tl - r.period * PERIOD;
    return r;
}

/* keep the pages' ink up to date with the timeline */
static float last_ink_u = -1;
static int ink_done;
static void pages_update(Tl tl, double wt) {
    if (tl.period != txt_period) {
        read_sensors();
        double wt0 = wt - tl.u;                         /* when this page was turned */
        layout_journal(&txt_jcur, wt0 + TURN_T + 1.0, tl.period, (float)TURN_T + 1.0f);
        layout_journal(&txt_jprev, wt0 - PERIOD + TURN_T + 1.0, tl.period - 1, -10000);
        layout_plate(&txt_cap[0], age_of(tl.period));
        layout_plate(&txt_cap[1], age_of(tl.period - 1));
        render_page(1, 1, 2.0f, &txt_cap[0], 0);
        render_page(3, 1, 2.0f, &txt_cap[1], 0);
        txt_period = tl.period;
        last_ink_u = -1;
        ink_done = 0;
    }
    /* the left page: last time's entry until the leaves have turned, then
     * this time's, written as we watch */
    float u = (float)tl.u;
    if (u < TURN_T) {
        if (last_ink_u < 0 || last_ink_u >= TURN_T) render_page(0, 0, 1.0f, &txt_jprev, 1e6f);
        last_ink_u = u;
        return;
    }
    if (ink_done && last_ink_u >= TURN_T) return;
    if (last_ink_u >= TURN_T && u - last_ink_u < 1.0f / 20 && u > last_ink_u) return;
    render_page(0, 0, 1.0f, &txt_jcur, u);
    last_ink_u = u;
    /* finished, and dry? */
    if (txt_jcur.n) {
        float tmax = 0;
        float *vb = malloc(sizeof(float) * (size_t)txt_jcur.n * TVF);
        glBindBuffer(GL_ARRAY_BUFFER, txt_jcur.vbo);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(sizeof(float) * (size_t)txt_jcur.n * TVF), vb);
        for (int i = 0; i < txt_jcur.n; i++) tmax = fmaxf(tmax, vb[i * TVF + 4] + vb[i * TVF + 5]);
        free(vb);
        if (u > tmax + 8) ink_done = 1;
    }
}

/* ---- the turning leaves ---- */
static float *leaf_v;
static int leaf_nv;
#define LNX 28
#define LNZ 8
static void leaves_build(float u) {
    leaf_nv = 0;
    if (u >= TURN_T) return;
    if (!leaf_v) leaf_v = malloc(sizeof(float) * VF * 3 * LNX * LNZ * 6);
    static const int front[3] = {3 + 16, 4, 5}, back[3] = {5, 4, 2};
    for (int k = 0; k < 3; k++) {
        float tau = clampf((u - 0.15f - k * 0.5f) / 2.0f, 0, 1);
        if (tau <= 0 || tau >= 1) continue;
        tau = tau * tau * (3 - 2 * tau);
        V3 P[LNZ + 1][LNX + 1];
        for (int j = 0; j <= LNZ; j++) {
            float z = -PAGE_H / 2 + PAGE_H * j / LNZ;
            /* the corner nearest the reader goes first */
            float tr = clampf(tau + 0.12f * (z / PAGE_H + 0.5f) * sinf(PI * tau), 0, 1);
            float x = 0, y = 0, ds = PAGE_W / LNX, lag = 0.55f;
            for (int i = 0; i <= LNX; i++) {
                float s = i * ds;
                float off = mixf((3 - k) * 0.0007f, (k + 1) * 0.0007f, tr);
                P[j][i] = v3(x, y + off, z);
                float rest = atanf((page_h(s + ds) - page_h(s)) / ds);
                float w = clampf(tr * (1 + lag) - lag * powf(s / PAGE_W, 0.8f), 0, 1);
                w = w * w * (3 - 2 * w);
                float phi = mixf(rest, PI - rest, w);
                x += cosf(phi) * ds;
                y += sinf(phi) * ds;
            }
        }
        for (int j = 0; j < LNZ; j++)
            for (int i = 0; i < LNX; i++) {
                V3 q[4] = {P[j][i], P[j][i + 1], P[j + 1][i + 1], P[j + 1][i]};
                float uu[4] = {(float)i / LNX, (float)(i + 1) / LNX, (float)(i + 1) / LNX, (float)i / LNX};
                float vv[4] = {(float)j / LNZ, (float)j / LNZ, (float)(j + 1) / LNZ, (float)(j + 1) / LNZ};
                static const int order[6] = {0, 2, 1, 0, 3, 2};
                for (int e = 0; e < 6; e++) {
                    int c = order[e];
                    int ii = i + (c == 1 || c == 2), jj = j + (c >= 2);
                    /* the normal from the grid's neighbours */
                    V3 dx = vsub(P[jj][ii < LNX ? ii + 1 : ii], P[jj][ii > 0 ? ii - 1 : ii]);
                    V3 dz = vsub(P[jj < LNZ ? jj + 1 : jj][ii], P[jj > 0 ? jj - 1 : jj][ii]);
                    V3 n = vnorm(vcross(dz, dx));
                    V3 p = m4_pt(&book_m, q[c]);
                    n = vnorm(m4_dir(&book_m, n));
                    float *o = leaf_v + (size_t)leaf_nv * VF;
                    o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = n.x; o[4] = n.y; o[5] = n.z;
                    o[6] = uu[c]; o[7] = vv[c]; o[8] = o[9] = o[10] = 1;
                    o[11] = MT_PAGE; o[12] = (float)front[k] + back[k] / 100.0f;
                    leaf_nv++;
                }
            }
    }
}

/* ---- the camera ---- */
static float cam_cx, cam_cy, cam_f, view_x, view_y, view_w, view_h, front_w, front_h;
static float lay_x0, lay_y0, lay_w, lay_h;

/* as flightsim: one virtual camera centred on the output under the middle
 * of the layout, each instance showing its own rect of the canvas */
static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = (float)W; view_h = (float)H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f;
        /* fit at least 64 degrees across and 44 up, whichever binds */
        cam_f = fminf(W / 2.0f / tanf(32 * DEG), H / 2.0f / tanf(22 * DEG));
        front_w = (float)W; front_h = (float)H;
        lay_x0 = lay_y0 = 0; lay_w = (float)W; lay_h = (float)H;
        return;
    }
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    lay_x0 = x0; lay_y0 = y0; lay_w = x1 - x0; lay_h = y1 - y0;
    float mx = (x0 + x1) / 2;
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { mid = i; break; }
    cam_cx = span[mid][0] + span[mid][2] / 2;
    cam_cy = span[mid][1] + span[mid][3] / 2;
    front_w = span[mid][2]; front_h = span[mid][3];
    float halfw = fmaxf(cam_cx - x0, x1 - cam_cx);
    cam_f = halfw / tanf(opt_fov / 2 * DEG);
    view_x = span[span_me][0]; view_y = span[span_me][1];
    view_w = span[span_me][2]; view_h = span[span_me][3];
}

typedef struct { V3 eye, at; } Shot;
static V3 panel_c, panel_n;

static Shot shot_home(void) {
    Shot s = {v3(desk_c.x + 0.42f, 1.52f, desk_c.z + 2.05f), v3(desk_c.x - 0.02f, DESK_H + 0.12f, desk_c.z - 0.05f)};
    return s;
}
static Shot shot_close(int deep) {
    /* near enough that the plate fills most of the front view */
    float th = front_h / 2 / cam_f, tw = front_w / 2 / cam_f;
    float ph = PAGE_H * (panel_uv[3] - panel_uv[1]), pw = PAGE_W * (panel_uv[2] - panel_uv[0]);
    float d = fmaxf(ph / (2 * th * (deep ? 0.86f : 0.7f)), pw / (2 * tw * (deep ? 0.9f : 0.75f)));
    V3 toward = vnorm(vadd(panel_n, v3(0.05f, 0, 0.35f)));
    Shot s = {vadd(panel_c, vmul(toward, d)), panel_c};
    return s;
}
static const Shot wander[] = {
    {{1.1f, 1.45f, 0.9f}, {3.6f, 1.25f, -1.7f}},       /* the window seat */
    {{-0.2f, 1.55f, 1.0f}, {-2.7f, 0.9f, -2.9f}},      /* the fire */
    {{0.4f, 1.35f, 1.3f}, {-0.6f, 3.3f, -4.2f}},       /* the high shelves, the ladder */
    {{-1.3f, 1.6f, 0.5f}, {1.9f, 1.3f, -3.3f}},        /* across the desk to the first window */
    {{1.4f, 1.15f, 1.2f}, {-0.4f, 1.0f, -2.2f}},       /* low, the desk against the shelves */
    {{-0.6f, 1.7f, 1.6f}, {2.6f, 1.9f, 0.2f}},         /* the windows, the telescope */
};
#define NWANDER 6

static float ease(float x) { x = clampf(x, 0, 1); return x * x * (3 - 2 * x); }
static float ease_soft(float x) { x = clampf(x, 0, 1); return mixf(x, x * x * (3 - 2 * x), 0.6f); }

static Shot shot_lerp(Shot a, Shot b, float t) {
    Shot s = {vlerp(a.eye, b.eye, t), vlerp(a.at, b.at, t)};
    return s;
}

static Shot camera_at(double t) {
    Tl tl = timeline(t);
    float u = (float)tl.u;
    Shot home = shot_home(), close = shot_close(0), deep = shot_close(1);
    if (opt_still) return home;
    uint32_t h = hash32((uint32_t)tl.period * 2246822519u + opt_seed);
    int wa = (int)(h % NWANDER), wb = (int)((h >> 8) % (NWANDER - 1));
    if (wb >= wa) wb++;
    /* a second, nearer home for variety */
    Shot home2 = home;
    home2.eye = vadd(home.eye, v3(-0.55f, 0.08f, -0.35f));
    /* the leaves turn while we stand at the desk's edge */
    Shot turn = shot_lerp(home, close, 0.5f);
    turn.at = vlerp(home.at, panel_c, 0.6f);
    turn.at.x -= 0.1f;
    Shot s;
    if (u < 5) s = turn;
    else if (u < 26) s = shot_lerp(turn, close, ease((u - 5) / 21));
    else if (u < 38) s = shot_lerp(close, deep, ease((u - 26) / 12));
    else if (u < 60) s = shot_lerp(deep, home2, ease((u - 38) / 22));
    else if (u < 92) s = shot_lerp(home2, wander[wa], ease_soft((u - 60) / 32));
    else if (u < 104) s = wander[wa];
    else if (u < 134) s = shot_lerp(wander[wa], wander[wb], ease_soft((u - 104) / 30));
    else if (u < 144) s = wander[wb];
    else if (u < 164) s = shot_lerp(wander[wb], home, ease((u - 144) / 20));
    else s = shot_lerp(home, turn, ease((u - 164) / 14));
    /* breathing: the slightest drift */
    float k = 1.0f - 0.85f * (u > 20 && u < 44 ? 1.0f : 0.0f);
    s.eye = vadd(s.eye, vmul(v3(sinf((float)t * 0.13f), sinf((float)t * 0.09f + 1) * 0.6f, sinf((float)t * 0.07f + 2)), 0.025f * k));
    s.at = vadd(s.at, vmul(v3(sinf((float)t * 0.05f + 3), sinf((float)t * 0.061f), 0), 0.03f * k));
    return s;
}

/* ---- per-frame state ---- */
static M4 view_m, proj_m, vp_m;
static V3 cam_eye;
static int frame_no;
static double last_age_t = -1;
static int age_px_w, age_px_h;          /* the picture's size in the age textures */

static void scene_targets(void) {
    int sw = (int)lroundf(W * render_scale), sh = (int)lroundf(H * render_scale);
    if (sw < 16) sw = 16;
    if (sh < 16) sh = 16;
    if (sw == scene_w && sh == scene_h) return;
    if (scene_w) {
        glDeleteTextures(1, &tex_scene); glDeleteTextures(1, &tex_depth); glDeleteFramebuffers(1, &fbo_scene);
        glDeleteTextures(1, &tex_vol); glDeleteFramebuffers(1, &fbo_vol);
        glDeleteTextures(NBLOOM, tex_bloom); glDeleteFramebuffers(NBLOOM, fbo_bloom);
    }
    scene_w = sw; scene_h = sh;
    tex_scene = mk_tex(GL_RGBA16F, sw, sh, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    tex_depth = mk_tex(GL_DEPTH_COMPONENT24, sw, sh, GL_DEPTH_COMPONENT, GL_FLOAT, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    fbo_scene = mk_fbo(tex_scene, tex_depth);
    vol_w = (sw + 3) / 4; vol_h = (sh + 3) / 4;
    tex_vol = mk_tex(GL_RGBA16F, vol_w, vol_h, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    fbo_vol = mk_fbo(tex_vol, 0);
    int bw = sw / 2, bh = sh / 2;
    for (int i = 0; i < NBLOOM; i++) {
        bloom_w[i] = bw > 1 ? bw : 1; bloom_h[i] = bh > 1 ? bh : 1;
        tex_bloom[i] = mk_tex(GL_RGBA16F, bloom_w[i], bloom_h[i], GL_RGBA, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_bloom[i] = mk_fbo(tex_bloom[i], 0);
        bw /= 2; bh /= 2;
    }
}

static void gl_init(void) {
    p_scene = mk_prog(VS_SCENE, FS_SCENE, "scene");
    p_shadow = mk_prog(VS_SCENE, FS_SHADOW, "shadow");
    p_flame = mk_prog(VS_FX, FS_FLAME, "flame");
    p_steam = mk_prog(VS_FX, FS_STEAM, "steam");
    p_dust = mk_prog(VS_DUST, FS_DUST, "dust");
    p_vol = mk_prog(VS_FULL, FS_VOL, "vol");
    p_down = mk_prog(VS_FULL, FS_DOWN, "down");
    p_up = mk_prog(VS_FULL, FS_UP, "up");
    p_comp = mk_prog(VS_FULL, FS_COMP, "comp");
    p_paper = mk_prog(VS_FULL, FS_PAPER, "paper");
    p_ink = mk_prog(VS_INK, FS_INK, "ink");
    const char *ages[NAGE] = {FS_AGE0, FS_AGE1, FS_AGE2, FS_AGE3, FS_AGE4};
    for (int i = 0; i < NAGE; i++) {
        char nm[16];
        snprintf(nm, sizeof nm, "age%d", i);
        p_age[i] = mk_prog(VS_FULL, ages[i], nm);
    }
    glGenVertexArrays(1, &vao_empty);
    noise_init();

    /* the room */
    glGenVertexArrays(1, &vao_room);
    glBindVertexArray(vao_room);
    glGenBuffers(1, &vbo_room);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_room);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)mesh.nv * VF * sizeof(float)), mesh.v, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo_room);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_room);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)((size_t)mesh.ni * sizeof(uint32_t)), mesh.ix, GL_STATIC_DRAW);
    const int off[5] = {0, 3, 6, 8, 11}, cnt[5] = {3, 3, 2, 3, 2};
    for (int a = 0; a < 5; a++) {
        glVertexAttribPointer((GLuint)a, cnt[a], GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(off[a] * sizeof(float)));
        glEnableVertexAttribArray((GLuint)a);
    }
    glGenVertexArrays(1, &vao_leaf);
    glBindVertexArray(vao_leaf);
    glGenBuffers(1, &vbo_leaf);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_leaf);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * VF * 3 * LNX * LNZ * 6), NULL, GL_DYNAMIC_DRAW);
    for (int a = 0; a < 5; a++) {
        glVertexAttribPointer((GLuint)a, cnt[a], GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(off[a] * sizeof(float)));
        glEnableVertexAttribArray((GLuint)a);
    }
    glGenVertexArrays(1, &vao_fx);
    glBindVertexArray(vao_fx);
    glGenBuffers(1, &vbo_fx);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_fx);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * 6 * 16, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(5 * sizeof(float)));
    for (int a = 0; a < 3; a++) glEnableVertexAttribArray((GLuint)a);
    glGenVertexArrays(1, &vao_ink);
    glBindVertexArray(vao_ink);
    for (int a = 0; a < 3; a++) glEnableVertexAttribArray((GLuint)a);

    /* shadows */
    tex_keysh = mk_tex(GL_DEPTH_COMPONENT24, KEY_SM, KEY_SM, GL_DEPTH_COMPONENT, GL_FLOAT, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    fbo_keysh = mk_fbo(0, tex_keysh);
    glEnable(GL_DEPTH_TEST);
    tex_firesh = cube_shadow(fire_pos, FIRE_SM, MT_EMBER);
    tex_lampsh = cube_shadow(lamp_pos, LAMP_SM, MT_LAMPGLASS);
    ao_build();

    /* pages and type */
    st_head = style(0, 50);
    st_body = style(0, 40);
    st_cap = style(1, 44);
    st_note = style(0, 36);
    st_num = style(1, 30);
    build_atlas();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    tex_atlas = mk_tex(GL_R8, AW, atlas_h, GL_RED, GL_UNSIGNED_BYTE, atlas, GL_LINEAR, GL_CLAMP_TO_EDGE);
    for (int i = 0; i < NPAGE; i++) {
        tex_page[i] = mk_tex(GL_RGBA8, PW, PH, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_LINEAR_MIPMAP_LINEAR, GL_CLAMP_TO_EDGE);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameterf(GL_TEXTURE_2D, 0x84FE /* GL_TEXTURE_MAX_ANISOTROPY */, 8.0f);
        fbo_page[i] = mk_fbo(tex_page[i], 0);
    }
    layout_filler(&txt_fill[0], 0);
    layout_filler(&txt_fill[1], 1);
    render_page(2, 2, 1.0f, NULL, 0);
    render_page(4, 3, 3.0f, &txt_fill[0], 0);
    render_page(5, 4, 4.0f, &txt_fill[1], 0);
    for (int i = 0; i < 2; i++) {
        tex_age[i] = mk_tex(GL_RGBA8, AGE_W, AGE_H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_LINEAR, GL_CLAMP_TO_EDGE);
        fbo_age[i] = mk_fbo(tex_age[i], 0);
    }
    glGetError();
    if (getenv("LIBRARY_DEBUG")) fprintf(stderr, "library: room %d vertices, %d triangles\n", mesh.nv, mesh.ni / 3);
}

/* the picture in the book, sized to how big it is on screen */
static void age_render(int which, int age, double t) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_age[which]);
    glViewport(0, 0, age_px_w, age_px_h);
    glDisable(GL_DEPTH_TEST);
    GLuint p = p_age[age];
    glUseProgram(p);
    glUniform1f(U(p, "uT"), (float)fmod(t + age * 37.0, 3600.0));
    glUniform2f(U(p, "uRes"), (float)age_px_w, (float)age_px_h);
    glUniform1i(U(p, "uNoise"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void fx_quad(float *o, V3 c, V3 right, float w, float h, float seed, float k) {
    V3 up = v3(0, 1, 0);
    V3 p[4] = {vsub(c, vmul(right, w / 2)), vadd(c, vmul(right, w / 2)), vadd(vadd(c, vmul(right, w / 2)), vmul(up, h)),
               vadd(vsub(c, vmul(right, w / 2)), vmul(up, h))};
    float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    static const int ord[6] = {0, 1, 2, 0, 2, 3};
    for (int e = 0; e < 6; e++) {
        int i = ord[e];
        float *q = o + e * 7;
        q[0] = p[i].x; q[1] = p[i].y; q[2] = p[i].z; q[3] = uv[i][0]; q[4] = uv[i][1]; q[5] = seed; q[6] = k;
    }
}

static void set_scene_uniforms(GLuint p, double t, float flick) {
    glUniformMatrix4fv(U(p, "uVP"), 1, GL_FALSE, vp_m.m);
    glUniform3f(U(p, "uCam"), cam_eye.x, cam_eye.y, cam_eye.z);
    glUniform1f(U(p, "uTime"), (float)fmod(t, 3600.0));
    glUniform3f(U(p, "uKeyDir"), lt.key_dir.x, lt.key_dir.y, lt.key_dir.z);
    glUniform3f(U(p, "uKeyCol"), lt.key_col.x, lt.key_col.y, lt.key_col.z);
    glUniformMatrix4fv(U(p, "uKeyVP"), 1, GL_FALSE, key_vp.m);
    V3 fc = vmul(v3(1.0f, 0.4f, 0.13f), 1.5f * flick);
    glUniform3f(U(p, "uFirePos"), fire_pos.x, fire_pos.y + 0.02f * sinf((float)t * 5.1f), fire_pos.z);
    glUniform3f(U(p, "uFireCol"), fc.x, fc.y, fc.z);
    float lf = 1.0f + 0.02f * sinf((float)t * 9.0f) * sinf((float)t * 2.3f);
    V3 lc = vmul(v3(1.0f, 0.66f, 0.34f), 0.42f * lf);
    glUniform3f(U(p, "uLampPos"), lamp_pos.x, lamp_pos.y, lamp_pos.z);
    glUniform3f(U(p, "uLampCol"), lc.x, lc.y, lc.z);
    float wc[NWIN][4];
    for (int i = 0; i < NWIN; i++) {
        V3 c = at_az(win[i].a, RIN + 0.15f, (win[i].y0 + win[i].y1 + win[i].wo) / 2);
        wc[i][0] = c.x; wc[i][1] = c.y; wc[i][2] = c.z;
        wc[i][3] = 2 * win[i].wo * (win[i].y1 - win[i].y0 + win[i].wo * 0.8f);
    }
    glUniform4fv(U(p, "uWinC"), NWIN, &wc[0][0]);
}

/* --bench with LIBRARY_PROF set: where the time goes */
static int prof_on, prof_cur = -1;
static double prof_ms[8];
static GLuint prof_q[8];
static const char *prof_name[8] = {"setup", "age", "pages", "room", "fx", "shafts", "bloom", "finish"};
/* GPU time of each stage, from timer queries: stage i starts now */
static void prof(int i) {
    if (!prof_on) return;
    if (!prof_q[0]) glGenQueries(8, prof_q);
    if (prof_cur >= 0) glEndQuery(GL_TIME_ELAPSED);
    if (i == 7 || i < 0) {
        /* collect the frame's */
        for (int k = 0; k < 8 && i == 7; k++) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(prof_q[k], GL_QUERY_RESULT, &ns);
            prof_ms[k] += ns / 1e6;
        }
        prof_cur = i < 0 ? 0 : -1;
        if (i < 0) glBeginQuery(GL_TIME_ELAPSED, prof_q[0]);
        return;
    }
    prof_cur = i + 1;
    glBeginQuery(GL_TIME_ELAPSED, prof_q[i + 1]);
}

static void render(double t, double wt, GLuint target) {
    prof(-1);
    setup_view();
    scene_targets();
    sky_light(wt);
    Tl tl = timeline(t);
    float u = (float)tl.u;
    int age = age_of(tl.period), age_prev = age_of(tl.period - 1);

    /* the camera */
    Shot s = camera_at(t);
    /* spanning, the wall is wide and short: look a little higher, so the
     * windows and the shelves' upper rows come into the side screens */
    if (span_me >= 0) { s.at.y += 0.12f; s.eye.y += 0.05f; }
    /* a tall screen: less rug, more shelves */
    else if (H > W) s.at.y += 0.45f * (1.0f - (float)W / H);
    if (getenv("LIBRARY_CAM")) sscanf(getenv("LIBRARY_CAM"), "%f,%f,%f,%f,%f,%f", &s.eye.x, &s.eye.y, &s.eye.z, &s.at.x, &s.at.y, &s.at.z);
    cam_eye = s.eye;
    float l = (view_x - cam_cx) / cam_f, r = (view_x + view_w - cam_cx) / cam_f;
    float b = (cam_cy - view_y - view_h) / cam_f, tp = (cam_cy - view_y) / cam_f;
    float nr = 0.02f;
    proj_m = m4_frustum(l * nr, r * nr, b * nr, tp * nr, nr, 150);
    view_m = m4_look(s.eye, vsub(s.at, s.eye), v3(0, 1, 0));
    vp_m = m4_mul(proj_m, view_m);

    /* shadows of the sun or moon: again whenever it has moved a little */
    if (!key_drawn || vdot(key_dir_drawn, lt.key_dir) < cosf(0.15f * DEG)) key_shadow_render();

    prof(0);
    /* the picture: as sharp as it is big on screen, at a third of the frames */
    {
        float dist = vlen(vsub(panel_c, cam_eye));
        float px = PAGE_H * (panel_uv[3] - panel_uv[1]) / dist * cam_f * render_scale;
        int ah = (int)clampf(px * 0.8f, 200, AGE_H);
        int aw = (int)(ah * (float)AGE_W / AGE_H);
        int changed = abs(ah - age_px_h) > 24;
        if (changed || !age_px_h) { age_px_w = aw; age_px_h = ah; }
        int due = opt_shot ? 1 : (frame_no % 3 == 0 || changed || last_age_t < 0);
        if (due) {
            age_render(0, age, t);
            if (u < TURN_T) age_render(1, age_prev, t);
            last_age_t = t;
        }
    }
    prof(1);
    pages_update(tl, wt);
    leaves_build(u);
    prof(2);

    /* the room */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    float flick = fire_flicker(t);
    /* depth first, so the expensive shading runs once a pixel */
    glUseProgram(p_shadow);
    set_wins(p_shadow);
    glUniform1i(U(p_shadow, "uSkip"), -1);
    glUniformMatrix4fv(U(p_shadow, "uVP"), 1, GL_FALSE, vp_m.m);
    glColorMask(0, 0, 0, 0);
    draw_room();
    glColorMask(1, 1, 1, 1);
    GLuint p = p_scene;
    glUseProgram(p);
    set_scene_uniforms(p, t, flick);
    set_wins(p);
    glUniform3f(U(p, "uSkyAmb"), lt.sky_amb.x, lt.sky_amb.y, lt.sky_amb.z);
    glUniform3f(U(p, "uGndAmb"), lt.gnd_amb.x, lt.gnd_amb.y, lt.gnd_amb.z);
    glUniform3f(U(p, "uWinL"), lt.win_l.x, lt.win_l.y, lt.win_l.z);
    glUniform3f(U(p, "uBounce"), lt.bounce.x, lt.bounce.y, lt.bounce.z);
    glUniform3f(U(p, "uSunDir"), lt.sun_dir.x, lt.sun_dir.y, lt.sun_dir.z);
    glUniform3f(U(p, "uMoonDir"), lt.moon_dir.x, lt.moon_dir.y, lt.moon_dir.z);
    glUniform3f(U(p, "uSkyZen"), lt.sky_zen.x, lt.sky_zen.y, lt.sky_zen.z);
    glUniform3f(U(p, "uSkyHor"), lt.sky_hor.x, lt.sky_hor.y, lt.sky_hor.z);
    glUniform3f(U(p, "uSunCol"), lt.sun_col.x, lt.sun_col.y, lt.sun_col.z);
    glUniform1f(U(p, "uNight"), lt.night);
    glUniform1f(U(p, "uMoonPhase"), lt.moon_phase);
    glUniform1f(U(p, "uStorm"), 0.15f);
    glUniform4fv(U(p, "uPanel"), 1, panel_uv);
    glUniform2f(U(p, "uAgeScale"), (float)age_px_w / AGE_W, (float)age_px_h / AGE_H);
    glUniform1f(U(p, "uPanelGlow"), 0.1f + 0.25f * lt.night);
    glUniform1f(U(p, "uLampOn"), lt.lamp_on);
    glUniform1i(U(p, "uDebug"), getenv("LIBRARY_DEBUG") ? atoi(getenv("LIBRARY_DEBUG")) : 0);
    static const char *pg[NPAGE] = {"uPage0", "uPage1", "uPage2", "uPage3", "uPage4", "uPage5"};
    int unit = 0;
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex_noise); glUniform1i(U(p, "uNoise"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex_keysh); glUniform1i(U(p, "uKeySh"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_CUBE_MAP, tex_firesh); glUniform1i(U(p, "uFireSh"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_CUBE_MAP, tex_lampsh); glUniform1i(U(p, "uLampSh"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_3D, tex_ao); glUniform1i(U(p, "uAO"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex_age[0]); glUniform1i(U(p, "uAgeLive"), unit++);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex_age[1]); glUniform1i(U(p, "uAgeOld"), unit++);
    for (int i = 0; i < NPAGE; i++) {
        glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex_page[i]); glUniform1i(U(p, pg[i]), unit++);
    }
    draw_room();
    if (leaf_nv) {
        glBindVertexArray(vao_leaf);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_leaf);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(sizeof(float) * VF * (size_t)leaf_nv), leaf_v);
        glDrawArrays(GL_TRIANGLES, 0, leaf_nv);
    }

    prof(3);
    /* fire, steam, dust: added light */
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);
    {
        float fx[7 * 6 * 16];
        int nq = 0;
        V3 fwd = vsub(fire_pos, cam_eye);
        V3 right = vnorm(v3(-fwd.z, 0, fwd.x));
        V3 fr = vnorm(v3(cosf(FIRE_AZ), 0, sinf(FIRE_AZ))), fin = inward(FIRE_AZ);
        for (int i = 0; i < 6; i++) {
            float o = (i - 2.5f) / 2.5f;
            V3 c = vadd(vadd(fire_pos, vmul(fr, o * 0.22f)), vadd(vmul(fin, -0.08f + 0.06f * (i & 1)), v3(0, -0.2f, 0)));
            float hh = 0.32f + 0.18f * (1 - fabsf(o)) + 0.05f * sinf((float)t * 1.3f + i);
            fx_quad(fx + nq++ * 42, c, right, 0.2f + 0.08f * (1 - fabsf(o)), hh, i * 1.37f, 0.8f + 0.25f * flick);
        }
        glUseProgram(p_flame);
        glUniformMatrix4fv(U(p_flame, "uVP"), 1, GL_FALSE, vp_m.m);
        glUniform1f(U(p_flame, "uTime"), (float)fmod(t, 3600.0));
        glUniform1i(U(p_flame, "uNoise"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_noise);
        glBindVertexArray(vao_fx);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_fx);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 42 * (size_t)nq, fx);
        glDrawArrays(GL_TRIANGLES, 0, nq * 6);
        /* the tea's last steam */
        V3 cf = vsub(cup_pos, cam_eye);
        V3 cr = vnorm(v3(-cf.z, 0, cf.x));
        fx_quad(fx, vadd(cup_pos, v3(0, -0.005f, 0)), cr, 0.2f, 0.34f, 0, 1);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 42, fx);
        glUseProgram(p_steam);
        glUniformMatrix4fv(U(p_steam, "uVP"), 1, GL_FALSE, vp_m.m);
        glUniform1f(U(p_steam, "uTime"), (float)fmod(t, 3600.0));
        glUniform1i(U(p_steam, "uNoise"), 0);
        V3 tint = vadd(vadd(vmul(lt.win_l, 0.05f), vmul(v3(1.0f, 0.66f, 0.34f), 0.12f * lt.lamp_on)), v3(0.01f, 0.006f, 0.003f));
        glUniform3f(U(p_steam, "uTint"), tint.x, tint.y, tint.z);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    {
        GLuint q = p_dust;
        glUseProgram(q);
        set_scene_uniforms(q, t, flick);
        glUniform1f(U(q, "uPx"), cam_f * render_scale * (scene_h / (float)scene_h));
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_keysh);
        glUniform1i(U(q, "uKeySh"), 1);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glBindVertexArray(vao_empty);
        glDrawArrays(GL_POINTS, 0, 1800);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    prof(4);
    /* light shafts */
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_vol);
        glViewport(0, 0, vol_w, vol_h);
        GLuint q = p_vol;
        glUseProgram(q);
        set_scene_uniforms(q, t, flick);
        M4 inv;
        {
            /* the inverse of the view-projection, by Gauss-Jordan */
            double a[4][8];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) { a[i][j] = vp_m.m[j * 4 + i]; a[i][j + 4] = i == j; }
            for (int c = 0; c < 4; c++) {
                int piv = c;
                for (int r2 = c + 1; r2 < 4; r2++) if (fabs(a[r2][c]) > fabs(a[piv][c])) piv = r2;
                for (int j = 0; j < 8; j++) { double tmp = a[c][j]; a[c][j] = a[piv][j]; a[piv][j] = tmp; }
                double d = a[c][c];
                for (int j = 0; j < 8; j++) a[c][j] /= d;
                for (int r2 = 0; r2 < 4; r2++)
                    if (r2 != c) {
                        double f = a[r2][c];
                        for (int j = 0; j < 8; j++) a[r2][j] -= f * a[c][j];
                    }
            }
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) inv.m[j * 4 + i] = (float)a[i][j + 4];
        }
        glUniformMatrix4fv(U(q, "uInvVP"), 1, GL_FALSE, inv.m);
        glUniform1f(U(q, "uDens"), 0.035f);
        glUniform1i(U(q, "uFrame"), frame_no);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_depth); glUniform1i(U(q, "uDepth"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_keysh); glUniform1i(U(q, "uKeySh"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, tex_noise3); glUniform1i(U(q, "uNoise3"), 2);
        glBindVertexArray(vao_empty);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    prof(5);
    /* bloom */
    {
        glUseProgram(p_down);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(U(p_down, "uSrc"), 0);
        for (int i = 0; i < NBLOOM; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i]);
            glViewport(0, 0, bloom_w[i], bloom_h[i]);
            int sw = i ? bloom_w[i - 1] : scene_w, sh = i ? bloom_h[i - 1] : scene_h;
            glBindTexture(GL_TEXTURE_2D, i ? tex_bloom[i - 1] : tex_scene);
            glUniform2f(U(p_down, "uTexel"), 0.5f / sw, 0.5f / sh);
            glUniform1f(U(p_down, "uThresh"), i == 0 ? 0.6f / lt.exposure : 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glUseProgram(p_up);
        glUniform1i(U(p_up, "uSrc"), 0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        for (int i = NBLOOM - 1; i > 0; i--) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i - 1]);
            glViewport(0, 0, bloom_w[i - 1], bloom_h[i - 1]);
            glBindTexture(GL_TEXTURE_2D, tex_bloom[i]);
            glUniform2f(U(p_up, "uTexel"), 1.0f / bloom_w[i], 1.0f / bloom_h[i]);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glDisable(GL_BLEND);
    }
    prof(6);
    /* the finish */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_comp);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_scene); glUniform1i(U(p_comp, "uScene"), 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_vol); glUniform1i(U(p_comp, "uVol"), 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_bloom[0]); glUniform1i(U(p_comp, "uBloom"), 2);
    glUniform2f(U(p_comp, "uTexel"), 1.0f / scene_w, 1.0f / scene_h);
    glUniform2f(U(p_comp, "uVolTexel"), 1.0f / vol_w, 1.0f / vol_h);
    glUniform1f(U(p_comp, "uExposure"), lt.exposure);
    glUniform1i(U(p_comp, "uDebug"), getenv("LIBRARY_DEBUG") ? atoi(getenv("LIBRARY_DEBUG")) : 0);
    glUniform1f(U(p_comp, "uTime"), (float)fmod(t, 1000.0));
    glUniform1f(U(p_comp, "uSharp"), render_scale < 0.95f ? 0.35f : 0.12f);
    glUniform4f(U(p_comp, "uVig"), (view_x - lay_x0) / lay_w, (view_y - lay_y0) / lay_h, view_w / lay_w, view_h / lay_h);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof(7);
    frame_no++;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "library");
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
        "usage: library [--window | --root | --layer] [--seed N] [--fps N] [--scale S]\n"
        "               [--hour H] [--age N] [--name NAME] [--still] [--fov DEG]\n"
        "               [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "               [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
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

static void save_ppm(const char *path, GLuint fbo) {
    unsigned char *px = malloc((size_t)W * H * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
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
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-hour") && i + 1 < argc) opt_hour = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-age") && i + 1 < argc) opt_age = atoi(argv[++i]);
        else if (!strcmp(a, "-name") && i + 1 < argc) opt_name = argv[++i];
        else if (!strcmp(a, "-still")) opt_still = 1;
        else if (!strcmp(a, "-fov") && i + 1 < argc) opt_fov = strtof(argv[++i], NULL);
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
    if (opt_fov < 20) opt_fov = 20;
    if (opt_fov > 150) opt_fov = 150;
    if (opt_hour >= 24) opt_hour = fmodf(opt_hour, 24);
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    int offscreen = opt_shot || opt_bench > 0;

    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    find_user();
    find_temp();
    read_sensors();
    fonts_init();
    room_build();
    pages_build();
    panel_c = m4_pt(&book_m, v3(PAGE_W * (panel_uv[0] + panel_uv[2]) / 2, page_h(PAGE_W * 0.5f),
                                -PAGE_H / 2 + PAGE_H * (panel_uv[1] + panel_uv[3]) / 2));
    panel_n = vnorm(m4_dir(&book_m, v3(0, 1, 0)));

    /* --epoch: when the timeline started, shared by every instance; a --shot
     * then shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();

    Display *dpy = NULL;
    Window root = 0, win_x = 0;
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
        win_x = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput, vi->visual,
                              CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win_x, "Library");
        XClassHint chint = {"library", "Library"};
        XSetClassHint(dpy, win_x, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win_x, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win_x);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win_x, ctx);
    }
    gl_init();

    if (offscreen) {
        GLuint tex = mk_tex(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
        GLuint fbo = mk_fbo(tex, 0);
        render_scale = opt_scale > 0 ? opt_scale : 1.0f;
        if (opt_shot) {
            double wt = t_wall0 + opt_at;
            render(opt_at, wt, fbo);
            glFinish();
            save_ppm(opt_shot, fbo);
            return 0;
        }
        double wt0 = wall();
        render(opt_at, wt0, fbo);
        glFinish();
        double sum = 0, worst = 0;
        prof_on = getenv("LIBRARY_PROF") != NULL;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            double t = opt_at + (i + 1) / 60.0;
            render(t, wt0 + (t - opt_at), fbo);
            glFinish();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("library: %dx%d scale %.2f, %d frames: avg %.3f ms, worst %.3f ms per frame\n", W, H, render_scale, opt_bench,
               sum / opt_bench, worst);
        printf("library: GL renderer: %s; hand: %s\n", (const char *)glGetString(GL_RENDERER), hand_family);
        if (prof_on)
            for (int i = 0; i < 8; i++) printf("  %-8s %6.2f ms\n", prof_name[i], prof_ms[i] / opt_bench);
        return 0;
    }

    double t0 = now();
    render_scale = opt_scale > 0 ? opt_scale : 0.7f;
    double avg = 1.0 / 60, last_change = 0;
    for (;;) {
        if (opt_layer && !wl_poll()) return 0;
        while (dpy && XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win_x, ev.xconfigure.width, ev.xconfigure.height);
                else { W = ev.xconfigure.width; H = ev.xconfigure.height; }
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        double t = now(), wt = wall(), el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        render(el, wt, 0);
        glFinish();
        double took = now() - t;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win_x);
        /* dynamic resolution: keep well inside the frame, leaving room for
         * the other instances sharing the GPU */
        if (frame_no > 10) avg += (took - avg) * 0.1;
        if (opt_scale <= 0 && frame_no > 30 && t - last_change > 0.5) {
            float s = render_scale;
            if (avg > 0.0055) s *= 0.92f;
            else if (avg < 0.0035) s *= 1.04f;
            s = clampf(s, 0.5f, 1.0f);
            if (fabsf(s - render_scale) > 0.02f) { render_scale = s; last_change = t; avg = 1.0 / 150; }
        }
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
