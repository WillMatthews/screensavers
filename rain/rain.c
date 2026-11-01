/*
 * rain — looking out of a window at night through rain-covered glass.
 *
 * Behind the glass, out of focus, a street running away from the window:
 * facades with lit windows and shopfronts, street lamps, cars coming and
 * going with their reflections on the wet road, all blurred into warm
 * bokeh. On the glass, water: small beads that sit, larger drops that
 * grow as rain lands on them, then let go and run down in wobbly paths,
 * leaving trails of tiny beads and swallowing whatever they meet. Each drop
 * is a little lens: it shows a sharp, upside-down view of the street, with
 * a dark rim and a glint. The glass between them is fogged, so the street
 * is blurrier there, and a running drop wipes a clear track through it.
 *
 * The weather is real when it can be: with a location (--lat/--lon, or
 * "LAT LON" in ~/.config/wallpapers/location) it asks Open-Meteo for the
 * current conditions every 15 minutes, in the background, cached in
 * $XDG_CACHE_HOME/wallpapers/rain-weather.json (shared between instances).
 * Rain rate, drizzle or showers, thunder (lightning flashes), wind (slants
 * the runs), cloud, temperature (sleet and snowflakes on the glass, frost
 * creeping in from the edges) and the sun's height (night, dusk, a grey
 * overcast day) all feed in, easing over a minute or so. Without a location
 * there is no network at all: a slow made-up weather where showers come
 * and go. --weather forces one condition. Shots and --bench never fetch.
 *
 * Plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), --shot
 * FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen, and
 * --bench N to time N offscreen frames.
 *
 * Keys (window mode): q / Escape quit.
 *
 * RAIN_DEBUG=1 in the environment prints the weather as loaded, and splits
 * --bench into CPU / GPU passes.
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
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

extern char **environ;

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;
static int off_w = 1920, off_h = 1080;
static double opt_lat = NAN, opt_lon = NAN;
static int opt_weather = -1;           /* forced condition, index into WX_NAMES */
static double opt_hour = -1;           /* --hour: pretend it's this local time of day */
static double opt_fps = 60;

static void die(const char *what) { fprintf(stderr, "rain: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* the drops' own random numbers (xorshift) */
static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}

static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }

/* The street is laid out from an integer hash of the scene seed, the same
 * function here and in the shaders, so the bokeh drawn from here lines up
 * with the facades drawn there. */
static uint32_t scene_seed;
static uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu ^ scene_seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
static float h01(uint32_t a, uint32_t b, uint32_t c) { return (hash3(a, b, c) >> 8) * (1.0f / 16777216.0f); }

/* ---- weather ----
 * rain: 0..1 how hard; size: drop size (drizzle 0.45 .. storm 1.3); snow:
 * fraction falling as snow; storm: thunder; wind: -1..1 sideways; cloud
 * 0..1; temp in C; day / dusk from the sun. */
typedef struct { float rain, size, snow, storm, wind, cloud, temp; } Wx;
static const char *WX_NAMES[] = {"rain", "drizzle", "storm", "clear", "snow"};
static const Wx WX_FORCED[] = {
    {0.55f, 1.0f, 0, 0, 0.15f, 0.9f, 9},
    {0.28f, 0.45f, 0, 0, 0.05f, 1.0f, 10},
    {0.95f, 1.3f, 0, 1, 0.6f, 1.0f, 13},
    {0.0f, 1.0f, 0, 0, 0.1f, 0.35f, 8},
    {0.5f, 1.0f, 1, 0, 0.25f, 1.0f, -2},
};
static Wx wx_cur, wx_tgt;
static float sky_day, sky_dusk;        /* from the sun */
static int have_loc, have_real;
static time_t real_at;                 /* when the real weather was observed */

/* smooth 1D value noise in [0, 1] */
static float vnoise(double t, uint32_t k) {
    double fl = floor(t);
    float f = (float)(t - fl);
    uint32_t i = (uint32_t)(int64_t)fl;
    float a = (hash3(i, k, 0x51u) >> 8) / 16777216.0f, b = (hash3(i + 1, k, 0x51u) >> 8) / 16777216.0f;
    return mixf(a, b, f * f * (3 - 2 * f));
}

/* made-up weather, from the wall clock alone (so every monitor agrees):
 * showers coming and going over half-hours, the odd storm, and a season */
static Wx wx_simulated(time_t T) {
    uint32_t keep = scene_seed;
    scene_seed = 0x5eedu;
    double t = (double)T;
    Wx w;
    float n = 0.6f * vnoise(t / 2400, 1) + 0.4f * vnoise(t / 700, 2);
    w.rain = smooth(0.48f, 0.74f, n) * (0.3f + 0.7f * vnoise(t / 3000, 3));
    w.size = 0.45f + 0.8f * vnoise(t / 5000, 4);
    w.storm = w.rain > 0.6f && vnoise(t / 4000, 5) > 0.72f ? 1.0f : 0.0f;
    w.wind = (vnoise(t / 1800, 6) - 0.5f) * 1.4f;
    w.cloud = 0.4f + 0.6f * smooth(0.25f, 0.55f, n);
    struct tm tm;
    localtime_r(&T, &tm);
    w.temp = 10.5f - 7.5f * cosf(6.2831853f * (tm.tm_yday - 15) / 365.0f) + (vnoise(t / 20000, 7) - 0.5f) * 8;
    w.snow = smooth(1.5f, -0.5f, w.temp);
    scene_seed = keep;
    return w;
}

/* the sun's altitude in degrees at unix time T */
static double sun_alt(double T, double lat, double lon) {
    double d = T / 86400.0 - 10957.5, rad = M_PI / 180;
    double g = (357.529 + 0.98560028 * d) * rad, q = 280.459 + 0.98564736 * d;
    double L = (q + 1.915 * sin(g) + 0.020 * sin(2 * g)) * rad, e = (23.439 - 0.00000036 * d) * rad;
    double ra = atan2(cos(e) * sin(L), cos(L)), dec = asin(sin(e) * sin(L));
    double gmst = fmod(18.697374558 + 24.06570982441908 * d, 24.0);
    double lha = gmst * 15 * rad + lon * rad - ra;
    return asin(sin(lat * rad) * sin(dec) + cos(lat * rad) * cos(dec) * cos(lha)) / rad;
}

/* day and dusk from the sun; without a location, a mid latitude and the
 * longitude the timezone implies */
static void sky_update(time_t T) {
    struct tm tm;
    localtime_r(&T, &tm);
    double t = (double)T;
    if (opt_hour >= 0) t += (opt_hour - (tm.tm_hour + tm.tm_min / 60.0 + tm.tm_sec / 3600.0)) * 3600;
    double lat = have_loc ? opt_lat : 50, lon = have_loc ? opt_lon : tm.tm_gmtoff / 240.0;
    float alt = (float)sun_alt(t, lat, lon);
    sky_day = smooth(-5, 7, alt);
    sky_dusk = expf(-(alt - 1) * (alt - 1) / 40.0f);
}

/* ---- Open-Meteo: only this one endpoint, only with lat/lon ---- */
static char cache_path[512], lock_path[512], tmp_path[560];
static pid_t fetch_pid;
static int lock_fd = -1;
static time_t next_check, backoff_until, loaded_mtime;

static int jnum(const char *s, const char *e, const char *key, double *out) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    size_t n = strlen(pat);
    for (const char *p = s; p + n <= e; p++)
        if (!memcmp(p, pat, n)) {
            char *end;
            double v = strtod(p + n, &end);
            if (end == p + n) return 0;
            *out = v;
            return 1;
        }
    return 0;
}

/* parse a forecast response into a Wx; 0 if it isn't one, or is for
 * somewhere else */
static int wx_parse(const char *js, Wx *w) {
    double lat, lon;
    if (!jnum(js, js + strlen(js), "latitude", &lat) || !jnum(js, js + strlen(js), "longitude", &lon)) return 0;
    if (fabs(lat - opt_lat) > 0.5 || fabs(lon - opt_lon) > 0.5) return 0;
    const char *s = strstr(js, "\"current\":");
    if (!s) return 0;
    const char *e = strchr(s, '}');
    if (!e) return 0;
    double precip, temp, rain = 0, showers = 0, snowfall = 0, cloud = 80, ws = 0, wd = 0, code = 0;
    if (!jnum(s, e, "precipitation", &precip) || !jnum(s, e, "temperature_2m", &temp)) return 0;
    jnum(s, e, "rain", &rain); jnum(s, e, "showers", &showers); jnum(s, e, "snowfall", &snowfall);
    jnum(s, e, "cloud_cover", &cloud); jnum(s, e, "wind_speed_10m", &ws); jnum(s, e, "wind_direction_10m", &wd);
    jnum(s, e, "weather_code", &code);
    int c = (int)code;
    Wx x = {0, 1.0f, 0, 0, 0, (float)cloud / 100, (float)temp};
    if (precip > 0) x.rain = clampf(1 - expf(-(float)precip / 2.0f), 0.12f, 1);
    if (c >= 51 && c <= 57) { x.rain = fmaxf(x.rain, 0.2f + (c - 51) * 0.03f); x.size = 0.45f; }
    if ((c >= 61 && c <= 67) || (c >= 80 && c <= 82)) {
        int k = c >= 80 ? c - 80 : ((c - 61) / 2) % 3;
        x.rain = fmaxf(x.rain, 0.3f + 0.25f * k);
    }
    if ((c >= 71 && c <= 77) || c == 85 || c == 86) { x.rain = fmaxf(x.rain, 0.35f); x.snow = 1; }
    if (c >= 95) { x.storm = 1; x.rain = fmaxf(x.rain, 0.8f); x.size = 1.3f; }
    if (snowfall > 0) x.snow = fmaxf(x.snow, clampf((float)(snowfall * 7 / (snowfall * 7 + rain + showers + 1e-3)), 0, 1));
    if (x.rain > 0 && temp < 2.5) x.snow = fmaxf(x.snow, smooth(2.5f, 0.0f, (float)temp));
    /* wind_direction is where it blows from; only the sideways part slants
     * the drops */
    x.wind = clampf(-(float)(sin(wd * M_PI / 180) * ws / 45), -1, 1);
    *w = x;
    return 1;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = malloc(65536);
    size_t n = fread(buf, 1, 65535, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

static int load_cache(void) {
    char *js = slurp(cache_path);
    Wx w;
    int ok = js && wx_parse(js, &w);
    free(js);
    struct stat st;
    if (ok && !stat(cache_path, &st)) {
        if (getenv("RAIN_DEBUG"))
            fprintf(stderr, "rain: weather rain %.2f size %.2f snow %.2f storm %.0f wind %.2f cloud %.2f temp %.1f (age %lds)\n",
                    w.rain, w.size, w.snow, w.storm, w.wind, w.cloud, w.temp, (long)(time(NULL) - st.st_mtime));
        wx_tgt = w;
        have_real = 1;
        real_at = st.st_mtime;
    }
    return ok;
}

static void release_lock(void) {
    if (lock_fd >= 0) { flock(lock_fd, LOCK_UN); close(lock_fd); lock_fd = -1; }
}

static void start_fetch(time_t t) {
    lock_fd = open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB)) {   /* another instance is at it */
        release_lock();
        next_check = t + 10;
        return;
    }
    struct stat st;
    if (!stat(cache_path, &st) && t - st.st_mtime < 900 && st.st_mtime != loaded_mtime) {   /* just fetched by another */
        release_lock();
        next_check = t;
        return;
    }
    char url[400];
    snprintf(url, sizeof url,
             "https://api.open-meteo.com/v1/forecast?latitude=%.3f&longitude=%.3f"
             "&current=precipitation,rain,showers,snowfall,cloud_cover,wind_speed_10m,"
             "wind_direction_10m,temperature_2m,is_day,weather_code",
             opt_lat, opt_lon);
    snprintf(tmp_path, sizeof tmp_path, "%s.%d.tmp", cache_path, (int)getpid());
    char *argv[] = {"curl", "-sf", "-m", "20", "--max-filesize", "60000", "-o", tmp_path, url, NULL};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    if (posix_spawnp(&fetch_pid, "curl", &fa, NULL, argv, environ)) {
        fetch_pid = 0;
        release_lock();
        backoff_until = t + 300;
    }
    posix_spawn_file_actions_destroy(&fa);
}

/* called every frame; never blocks */
static void weather_tick(void) {
    if (!have_loc || opt_weather >= 0) return;
    time_t t = time(NULL);
    if (fetch_pid) {
        int status;
        pid_t r = waitpid(fetch_pid, &status, WNOHANG);
        if (r == 0) return;
        fetch_pid = 0;
        char *js = r > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0 ? slurp(tmp_path) : NULL;
        Wx w;
        if (js && wx_parse(js, &w) && !rename(tmp_path, cache_path)) {
            next_check = 0;                  /* picked up just below */
        } else {
            unlink(tmp_path);
            backoff_until = t + 300;
        }
        free(js);
        release_lock();
    }
    if (t < next_check) return;
    next_check = t + 30;
    struct stat st;
    int have = !stat(cache_path, &st);
    static int cache_ok;
    if (have && st.st_mtime != loaded_mtime) {
        loaded_mtime = st.st_mtime;
        cache_ok = t - st.st_mtime < 3 * 3600 && load_cache();   /* a cache for elsewhere doesn't count */
    }
    if ((!have || !cache_ok || t - st.st_mtime >= 900) && t >= backoff_until) start_fetch(t);
    if (have_real && t - real_at > 3 * 3600) have_real = 0;   /* stale: back to make-believe */
}

static void location_init(void) {
    if (isnan(opt_lat) || isnan(opt_lon)) {
        const char *home = getenv("HOME");
        char path[512];
        snprintf(path, sizeof path, "%s/.config/wallpapers/location", home ? home : ".");
        FILE *f = fopen(path, "r");
        if (f) {
            double a, b;
            if (fscanf(f, "%lf%*[ ,\t]%lf", &a, &b) == 2) { opt_lat = a; opt_lon = b; }
            fclose(f);
        }
    }
    have_loc = !isnan(opt_lat) && !isnan(opt_lon) && fabs(opt_lat) <= 90 && fabs(opt_lon) <= 180;
    if (!have_loc) return;
    const char *xc = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    char dir[400];
    if (xc && *xc) snprintf(dir, sizeof dir, "%s/wallpapers", xc);
    else snprintf(dir, sizeof dir, "%s/.cache/wallpapers", home ? home : "/tmp");
    char parent[400];
    snprintf(parent, sizeof parent, "%.*s", (int)(strrchr(dir, '/') - dir), dir);
    mkdir(parent, 0755);
    mkdir(dir, 0755);
    snprintf(cache_path, sizeof cache_path, "%s/rain-weather.json", dir);
    snprintf(lock_path, sizeof lock_path, "%s/rain-weather.lock", dir);
}

/* the weather to aim for now */
static void wx_target(time_t t) {
    if (opt_weather >= 0) wx_tgt = WX_FORCED[opt_weather];
    else if (!have_real) wx_tgt = wx_simulated(t);
}

static void wx_ease(float dt) {
    float k = 1 - expf(-dt / 45.0f);
    float *c = (float *)&wx_cur, *g = (float *)&wx_tgt;
    for (size_t i = 0; i < sizeof(Wx) / sizeof(float); i++) c[i] += (g[i] - c[i]) * k;
}

/* ---- the water on the glass ----
 * Positions in screen pixels, y down. sc scales everything from 1080p. */
#define MAXD 9000
typedef struct {
    float x, y, r;          /* centre, radius */
    float vy, wx;           /* running: speed down, sideways wander */
    float stall, trail;     /* seconds stuck; distance since the last bead */
    float runr;             /* the radius at which this one lets go */
    float seed, melt;       /* flakes: 0 crystal .. 1 water */
    uint8_t type, run;      /* type 0 water, 1 snowflake */
} Drop;
static Drop drops[MAXD];
static int nd, cap_d;
#define DROP_SCALE 1.8f   /* drop sizes are in 1080p px x this */
static float sc = 1, maxr;
static int *ghead, gnx, gny;
static int gnext[MAXD];
static float gcs;

static void grid_alloc(void) {
    gcs = 24 * sc;
    gnx = (int)(W / gcs) + 1; gny = (int)(H / gcs) + 1;
    free(ghead);
    ghead = malloc(sizeof(int) * (size_t)gnx * gny);
    cap_d = (int)(2600.0f * W * H / (1920.0f * 1080.0f));
    if (cap_d > MAXD - 64) cap_d = MAXD - 64;
    if (cap_d < 600) cap_d = 600;
}

static void grid_add(int i) {
    int cx = (int)(drops[i].x / gcs), cy = (int)(drops[i].y / gcs);
    cx = cx < 0 ? 0 : cx >= gnx ? gnx - 1 : cx;
    cy = cy < 0 ? 0 : cy >= gny ? gny - 1 : cy;
    gnext[i] = ghead[cy * gnx + cx];
    ghead[cy * gnx + cx] = i;
}

static void grid_build(void) {
    for (int i = 0; i < gnx * gny; i++) ghead[i] = -1;
    maxr = 0;
    for (int i = 0; i < nd; i++)
        if (drops[i].r > 0) { grid_add(i); if (drops[i].r > maxr) maxr = drops[i].r; }
}

/* the largest drop overlapping a circle, or -1 */
static int overlap(float x, float y, float r, int skip, float k) {
    float R = r + maxr;
    int x0 = (int)((x - R) / gcs), x1 = (int)((x + R) / gcs), y0 = (int)((y - R) / gcs), y1 = (int)((y + R) / gcs);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= gnx) x1 = gnx - 1;
    if (y1 >= gny) y1 = gny - 1;
    int best = -1;
    for (int cy = y0; cy <= y1; cy++)
        for (int cx = x0; cx <= x1; cx++)
            for (int j = ghead[cy * gnx + cx]; j >= 0; j = gnext[j]) {
                Drop *o = &drops[j];
                if (j == skip || o->r <= 0) continue;
                float dx = o->x - x, dy = o->y - y, rr = (r + o->r) * k;
                if (dx * dx + dy * dy < rr * rr && (best < 0 || o->r > drops[best].r)) best = j;
            }
    return best;
}

static float new_runr(void) { return sc * (4.8f + 2.4f * frand()); }

static int add_drop(float x, float y, float r, int type) {
    int i, fresh = nd < cap_d;
    if (fresh) i = nd++;
    else {                                   /* full: an old small bead makes way */
        i = -1;
        for (int k = 0; k < 8; k++) {
            int j = (int)(frand() * nd);
            if (j >= nd) j = nd - 1;
            if (!drops[j].run && (i < 0 || drops[j].r < drops[i].r)) i = j;
        }
        if (i < 0) return -1;
    }
    Drop *d = &drops[i];
    memset(d, 0, sizeof *d);
    d->x = x; d->y = y; d->r = r; d->type = (uint8_t)type;
    d->runr = new_runr();
    d->seed = frand();
    if (fresh) grid_add(i);     /* a recycled slot is already listed (in its old cell) */
    if (r > maxr) maxr = r;
    return i;
}

static void absorb(Drop *a, Drop *b) {
    float ma = a->r * a->r * a->r, mb = b->r * b->r * b->r;
    float wa = a->run ? ma * 3 : ma;          /* a runner keeps its line */
    a->x = (a->x * wa + b->x * mb) / (wa + mb);
    if (!a->run) a->y = (a->y * wa + b->y * mb) / (wa + mb);
    a->r = cbrtf(ma + mb);
    if (b->run && b->vy > a->vy) a->vy = b->vy;
    b->r = 0;
}

/* rain (or snow) landing on the glass */
static void impact(float snow, float size) {
    float x = frand() * W, y = frand() * H;
    if (frand() < snow) {
        if (frand() > 0.3f) return;          /* fewer flakes than drops reach the glass */
        float r = sc * (3.0f + 5.0f * powf(frand(), 2.0f));
        int j = overlap(x, y, r * 0.6f, -1, 0.7f);
        if (j >= 0 && drops[j].type == 0) {   /* onto water: melts in */
            Drop tmp = {0};
            tmp.x = x; tmp.y = y; tmp.r = r * 0.45f;
            absorb(&drops[j], &tmp);
            return;
        }
        add_drop(x, y, r, 1);
        return;
    }
    float u = frand();
    float r = sc * (0.9f + 3.6f * size * powf(u, 2.2f));
    if (frand() < 0.08f * size * size) r += sc * (2.5f + 4.0f * frand()) * size;
    int j = overlap(x, y, r, -1, 0.9f);
    if (j >= 0) {
        Drop tmp = {0};
        tmp.x = x; tmp.y = y; tmp.r = r;
        if (drops[j].type == 1) { drops[j].melt = fminf(1, drops[j].melt + 0.25f); return; }
        absorb(&drops[j], &tmp);
        return;
    }
    add_drop(x, y, r, 0);
}

static float impact_acc;

static void sim_step(float dt) {
    const Wx *w = &wx_cur;
    float area = (float)W * H / (1920.0f * 1080.0f);
    grid_build();

    /* impacts per second over the whole screen */
    float rate = w->rain > 0.01f ? area * (20 + 300 * w->rain) * (w->size < 0.7f ? 1.6f : 1.0f) : 0;
    impact_acc += rate * dt;
    while (impact_acc >= 1) { impact_acc -= 1; impact(w->snow, w->size); }

    float wet = smooth(0.0f, 0.15f, w->rain);
    /* evaporation: r^2 shrinks steadily, so the smallest go first */
    float evap = sc * sc * (0.004f + 0.05f * (1 - wet)) * (w->temp > 0 ? 1.0f : 0.3f);
    /* the room warms the glass: even below freezing flakes go, slowly */
    float melt_rate = 0.006f + 0.02f * smooth(-4, 1, w->temp) + 0.02f * fmaxf(0, w->temp);

    for (int i = 0; i < nd; i++) {
        Drop *d = &drops[i];
        if (d->r <= 0) continue;
        if (d->type == 1) {
            d->melt = fminf(1, d->melt + melt_rate * dt * (0.6f + 0.8f * d->seed));
            if (d->melt >= 1) { d->type = 0; d->r *= 0.6f; d->melt = 0; }
            
            continue;
        }
        if (!d->run) {
            float r2 = d->r * d->r - evap * dt;
            d->r = r2 > 0 ? sqrtf(r2) : 0;
            if (d->r < 0.45f * sc) d->r = 0;
            if (d->r > d->runr) { d->run = 1; d->vy = 0; d->stall = 0.1f + 0.4f * frand(); }
            continue;
        }
        /* running: a bigger drop runs faster; it sticks and lurches */
        float rr = d->r / sc;
        float vt = sc * 9.0f * powf(fmaxf(rr - 3.6f, 0.4f), 1.5f);
        if (vt > sc * 650) vt = sc * 650;
        if (d->stall > 0) {
            d->stall -= dt;
            d->vy *= expf(-dt * 14);
        } else {
            d->vy += (vt - d->vy) * fminf(1, dt * 3.5f);
            float p = 1.4f * expf(-(rr - d->runr / sc) * 0.5f);
            if (frand() < p * dt) d->stall = 0.08f + 0.6f * frand() * frand();
        }
        d->wx += ((frand() - 0.5f) * sc * 900 * (0.4f + 0.6f * (1 - fminf(1, d->vy / (sc * 300)))) - d->wx * 3.0f) * dt;
        float dy = d->vy * dt;
        d->x += (d->wx * fminf(1, d->vy / (sc * 40)) + w->wind * d->vy * 0.55f) * dt;
        d->y += dy;
        d->trail += dy;
        if (d->trail > d->r * (0.45f + 0.7f * frand())) {
            d->trail = 0;
            float rb = d->r * (0.14f + 0.2f * frand());
            if (frand() < 0.15f) rb *= 1.5f;
            float m = d->r * d->r * d->r - rb * rb * rb;
            if (rb > 0.5f * sc && m > 0) {
                d->r = cbrtf(m);
                add_drop(d->x + (frand() - 0.5f) * d->r * 0.6f, d->y - d->r * (0.8f + 0.4f * frand()), rb, 0);
            }
        }
        if (d->r < d->runr * 0.62f) { d->run = 0; d->runr = d->r * 1.25f + sc * 1.5f; }
        if (d->y - d->r * 3 > H || d->x < -40 * sc || d->x > W + 40 * sc) { d->r = 0; continue; }
        /* swallow what it runs into */
        for (int guard = 0; guard < 4; guard++) {
            int j = overlap(d->x, d->y + d->r * 0.25f, d->r, i, 0.82f);
            if (j < 0) break;
            Drop *o = &drops[j];
            if (o->y < d->y - d->r * 0.55f && !o->run) break;   /* its own trail, behind */
            absorb(d, o);
        }
    }
    /* compact */
    int k = 0;
    for (int i = 0; i < nd; i++)
        if (drops[i].r > 0) drops[k++] = drops[i];
    nd = k;
}

/* ---- the street: lights for the bokeh and the sharp pass ----
 * World: metres, x across (camera at 0), height up, z away. The camera is a
 * first-floor window looking down the street towards a vanishing point. */
#define CAM_H 6.0f
#define FOC 0.9f
#define VPX 0.64f      /* vanishing point, x as a fraction of the width */
#define VPY 0.53f      /* and y as a fraction of the height (down) */
#define LAT_L -11.0f   /* the facades */
#define LAT_R 9.5f
static float lights[4096 * 8];
static int nl;
static float bokeh_r;                  /* px */

static void add_light(float x, float y, float rb, float rs, float r, float g, float b, float shape) {
    if (nl >= 4096) return;
    float m = rb * 3 + 4;
    if (x < -m || x > W + m || y < -m * 3 || y > H + m * 3) return;
    float *L = &lights[nl++ * 8];
    L[0] = x; L[1] = y; L[2] = rb; L[3] = rs; L[4] = r; L[5] = g; L[6] = b; L[7] = shape;
}

/* world to screen px */
static void proj(float lat, float hgt, float z, float *x, float *y) {
    *x = (VPX * W / H + FOC * lat / z) * H;
    *y = (VPY + FOC * (CAM_H - hgt) / z) * H;
}

static float bh(uint32_t side, float z) {
    float seg = floorf(z / 14);
    return 7 + 22 * h01(side, (uint32_t)(seg < 0 ? 0 : seg), 7);
}

/* is window (column ci, floor fi) on this side lit at scene time t; its colour */
static int win_lit(uint32_t side, uint32_t ci, uint32_t fi, float t, float litp, float *col) {
    uint32_t e = (uint32_t)floorf(t / 600 + h01(side, ci, fi + 500));
    float p = fi == 0 ? litp * 1.5f : litp;
    if (h01(side + 16 * e, ci, fi) >= p) return 0;
    float k = h01(side, ci, fi + 2000), b = 0.45f + 0.55f * h01(side, ci, fi + 3000);
    static const float pal[5][3] = {{0.94f, 0.89f, 0.81f}, {0.91f, 0.60f, 0.29f}, {0.95f, 0.74f, 0.33f},
                                    {0.56f, 0.65f, 0.45f}, {0.80f, 0.30f, 0.18f}};
    if (fi == 0) {
        int i = (int)(k * 5);
        if (i > 4) i = 4;
        for (int c = 0; c < 3; c++) col[c] = pal[i][c] * (0.8f + 0.4f * b);
    } else {
        for (int c = 0; c < 3; c++) col[c] = mixf(pal[1][c], pal[0][c], k) * b;
        if (k > 0.93f) for (int c = 0; c < 3; c++) col[c] = pal[3][c] * 0.6f * b;
    }
    return 1;
}

static void scene_lights(float t, float night, float wet, float snow, float rainy) {
    nl = 0;
    float lamps = smooth(0.25f, 0.85f, night);
    bokeh_r = H * 0.036f * mixf(0.72f, 1.0f, rainy);
    float x, y, x2, y2;
    float rs_min = 2.5f * H / 1080.0f;
    /* street lamps, both sides, staggered */
    if (lamps > 0.01f)
        for (int side = 0; side < 2; side++)
            for (int k = 0; k < 16; k++) {
                float z = (side ? 9.0f : 21.0f) + 24.0f * k, lat = side ? 0.8f : -8.4f, hgt = 6.4f;
                proj(lat, hgt, z, &x, &y);
                int led = h01(side, k, 77) < 0.35f;
                float c[3] = {0.91f, 0.56f, 0.24f};
                if (led) { c[0] = 1.0f; c[1] = 0.80f; c[2] = 0.52f; }
                float pw = lamps * 1.5f * powf(10 / z, 1.05f);
                float near = smooth(200, 10, z);
                float rs = fmaxf(rs_min, 0.32f * FOC / z * H);
                add_light(x, y, bokeh_r * (0.9f + 0.25f * near), rs, c[0] * pw, c[1] * pw, c[2] * pw, 0);
                if (wet > 0.01f) {
                    proj(lat, -hgt, z, &x2, &y2);
                    float pr = pw * 0.35f * wet;
                    add_light(x2, y2, bokeh_r * 0.9f, rs * 1.3f, c[0] * pr, c[1] * pr, c[2] * pr, 1);
                }
            }
    /* windows and shopfronts */
    float litp = 0.06f + 0.3f * smooth(0.2f, 0.9f, night);
    for (uint32_t side = 0; side < 2; side++) {
        float lat = side ? LAT_R : LAT_L;
        for (uint32_t ci = 2; ci < 40; ci++) {
            float z = (ci + 0.5f) * 3.5f;
            float top = bh(side, z);
            for (uint32_t fi = 0; fi < 9; fi++) {
                float hgt = fi == 0 ? 1.5f : (fi + 0.55f) * 3.2f;
                if (hgt > top - 0.5f) break;
                float c[3];
                if (!win_lit(side, ci, fi, t, litp, c)) continue;
                proj(lat, hgt, z, &x, &y);
                float pw = (fi == 0 ? 0.5f : 0.26f) * powf(12 / z, 0.9f) * (0.25f + 0.75f * night);
                float rs = fmaxf(rs_min, (fi == 0 ? 1.2f : 0.7f) * FOC / z * H);
                add_light(x, y, bokeh_r * (0.85f + 0.2f * h01(side, ci, fi + 4000)), rs * 0.6f, c[0] * pw, c[1] * pw, c[2] * pw, 0);
                if (fi == 0 && wet > 0.01f) {
                    proj(lat * 0.93f, -hgt, z, &x2, &y2);
                    float pr = pw * 0.3f * wet;
                    add_light(x2, y2, bokeh_r * 0.85f, rs, c[0] * pr, c[1] * pr, c[2] * pr, 1);
                }
            }
        }
    }
    /* a traffic light */
    {
        float ph = fmodf(t + 17, 60);
        float c[3] = {0.56f, 0.72f, 0.42f};
        if (ph >= 26 && ph < 29) { c[0] = 1.0f; c[1] = 0.62f; c[2] = 0.15f; }
        else if (ph >= 29) { c[0] = 0.86f; c[1] = 0.22f; c[2] = 0.10f; }
        proj(1.2f, 4.2f, 46, &x, &y);
        float pw = 0.7f * (0.4f + 0.6f * night);
        add_light(x, y, bokeh_r * 0.8f, rs_min * 1.2f, c[0] * pw, c[1] * pw, c[2] * pw, 0);
        if (wet > 0.01f) {
            proj(1.2f, -4.2f, 46, &x2, &y2);
            add_light(x2, y2, bokeh_r * 0.8f, rs_min * 1.5f, c[0] * pw * 0.3f * wet, c[1] * pw * 0.3f * wet, c[2] * pw * 0.3f * wet, 1);
        }
    }
    /* cars: headlights coming, tail lights going */
    float density = mixf(0.35f, 0.6f, 1 - night) + 0.1f;
    for (uint32_t lane = 0; lane < 2; lane++)
        for (uint32_t k = 0; k < 6; k++) {
            float v = 8 + 6 * h01(lane, k, 11);
            float len = 257 / v, P = len + 6 + 40 * h01(lane, k, 12);
            float tt = t + P * h01(lane, k, 13) * 7;
            float cyc = floorf(tt / P), ph = tt - cyc * P;
            if (ph > len || h01(lane, k, 1000 + (uint32_t)cyc) > density) continue;
            float z = lane == 0 ? 260 - v * ph : 3 + v * ph;
            float lat = lane == 0 ? -5.6f : -2.0f;
            float on = smooth(3, 7, z) * smooth(260, 230, z);
            float c[3];
            float pw;
            if (lane == 0) { c[0] = 1.0f; c[1] = 0.92f; c[2] = 0.78f; pw = 2.4f * powf(10 / z, 1.2f) * (0.5f + 0.5f * night); }
            else { c[0] = 0.85f; c[1] = 0.17f; c[2] = 0.08f; pw = 1.1f * powf(10 / z, 1.1f) * (0.45f + 0.55f * night); }
            pw = fminf(pw, 3.0f) * on;
            for (int s = -1; s <= 1; s += 2) {
                proj(lat + s * 0.72f, 0.7f, z, &x, &y);
                float rs = fmaxf(rs_min, 0.22f * FOC / z * H);
                add_light(x, y, bokeh_r * (0.9f + 0.3f * smooth(80, 5, z)), rs, c[0] * pw, c[1] * pw, c[2] * pw, 0);
                if (wet > 0.01f) {
                    proj(lat + s * 0.72f, -0.7f, z, &x2, &y2);
                    float pr = pw * 0.45f * wet;
                    add_light(x2, y2 + rs, bokeh_r * 0.9f, rs * 1.4f, c[0] * pr, c[1] * pr, c[2] * pr, 1);
                }
            }
        }
    /* snow falling between us and the street */
    if (snow > 0.02f)
        for (uint32_t k = 0; k < 160; k++) {
            float sp = H * (0.05f + 0.08f * h01(k, 1, 21));
            float span = H * 1.2f;
            float yy = fmodf(h01(k, 2, 21) * span + sp * t, span) - H * 0.1f;
            float xx = fmodf(h01(k, 3, 21) * W * 1.3f + wx_cur.wind * sp * 0.8f * t + H * 0.02f * sinf(t * 0.7f + k), W * 1.3f);
            if (xx < 0) xx += W * 1.3f;
            xx -= W * 0.15f;
            float b = snow * (0.03f + 0.05f * h01(k, 4, 21)) * (0.5f + 0.5f * night + 0.6f);
            float rb = bokeh_r * (0.3f + 0.5f * h01(k, 5, 21));
            add_light(xx, yy, rb, rs_min * 1.2f, b * 0.95f, b * 0.92f, b * 0.86f, 2);
        }
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* the street, sharp, at half resolution: facades with windows, the wet road
 * mirroring them, the sky; the lights are added on top as points */
static const char *FS_STREET = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform float uAsp, uT, uDay, uDusk, uCloud, uFlash, uWet, uLitP, uNight, uSnowGround;
    uniform uint uSeed;
    out vec4 o;
    const float CAMH = 6.0, FOC = 0.9, VPY = 0.53, LL = 11.0, LR = 9.5;

    uint hash3(uint a, uint b, uint c) {
        uint h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu ^ uSeed;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
        return h;
    }
    float h01(uint a, uint b, uint c) { return float(hash3(a, b, c) >> 8) * (1.0 / 16777216.0); }
    float bh(uint side, float z) { return 7.0 + 22.0 * h01(side, uint(max(floor(z / 14.0), 0.0)), 7u); }

    vec3 sky(float Y) {
        float t = clamp((VPY - Y) / VPY, 0.0, 1.0);
        vec3 night = mix(vec3(0.16, 0.095, 0.05) * (0.45 + 0.6 * uCloud), vec3(0.03, 0.022, 0.018), pow(t, 0.55));
        vec3 day = mix(vec3(0.68, 0.65, 0.60), vec3(0.52, 0.50, 0.48), t) * mix(1.12, 0.92, uCloud);
        vec3 c = mix(night, day, uDay);
        c += uDusk * vec3(0.52, 0.25, 0.09) * exp(-t * 3.5) * (1.0 - 0.5 * uCloud);
        c += uFlash * vec3(0.80, 0.77, 0.72) * (0.35 + 0.65 * t);
        return c;
    }

    vec3 window(uint side, float ci, float fi, out float lit) {
        uint c = uint(ci), f = uint(fi);
        uint e = uint(floor(uT / 600.0 + h01(side, c, f + 500u)));
        float p = f == 0u ? uLitP * 1.5 : uLitP;
        lit = h01(side + 16u * e, c, f) < p ? 1.0 : 0.0;
        float k = h01(side, c, f + 2000u), b = 0.45 + 0.55 * h01(side, c, f + 3000u);
        vec3 pal[5] = vec3[5](vec3(0.94, 0.89, 0.81), vec3(0.91, 0.60, 0.29), vec3(0.95, 0.74, 0.33),
                              vec3(0.56, 0.65, 0.45), vec3(0.80, 0.30, 0.18));
        if (f == 0u) return pal[min(int(k * 5.0), 4)] * (0.45 + 0.3 * b);
        if (k > 0.93) return pal[3] * 0.6 * b;
        return mix(pal[1], pal[0], k) * b;
    }

    vec3 facade(uint side, float z, float hgt) {
        float tone = h01(side, uint(floor(z / 14.0)), 3u);
        vec3 wallN = vec3(0.034, 0.027, 0.022) * (0.7 + 0.6 * tone) + vec3(0.10, 0.055, 0.02) * exp(-hgt / 3.5) * 0.7 * uNight;
        vec3 wallD = mix(vec3(0.33, 0.31, 0.29), vec3(0.40, 0.33, 0.27), tone);
        vec3 c = mix(wallN, wallD, uDay);
        float cz = z / 3.5, fl = hgt < 3.6 ? 0.0 : (hgt - 0.0) / 3.2;
        float ci = floor(cz), fi = floor(fl);
        vec2 f = vec2(fract(cz), hgt < 3.6 ? hgt / 3.6 : fract(fl));
        bool shop = hgt < 3.6;
        vec2 lo = shop ? vec2(0.06, 0.05) : vec2(0.22, 0.3), hi = shop ? vec2(0.94, 0.72) : vec2(0.78, 0.82);
        vec2 fw = max(fwidth(vec2(cz, shop ? hgt / 3.6 : fl)), vec2(1e-4));
        vec2 w2 = smoothstep(lo - fw, lo + fw, f) * (1.0 - smoothstep(hi - fw, hi + fw, f));
        float win = w2.x * w2.y;
        if (win > 0.0) {
            float lit;
            vec3 wc = window(side, ci, fi, lit);
            vec3 dark = mix(vec3(0.012, 0.010, 0.009), vec3(0.24, 0.235, 0.23), uDay);
            c = mix(c, lit > 0.5 ? wc * mix(0.85, 0.25, uDay) * (shop ? 0.45 + 0.55 * smoothstep(6.0, 30.0, z) : 1.0) + dark : dark, win);
        }
        return mix(c, sky(VPY - 0.02), 1.0 - exp(-z / 230.0));
    }

    /* what a ray through (X, Y) meets above the ground: facade or sky */
    vec3 upper(float X, float Y) {
        float dx = X - 0.64 * uAsp, dy = Y - VPY;
        float zl = dx < -1e-4 ? FOC * LL / -dx : 1e9, zr = dx > 1e-4 ? FOC * LR / dx : 1e9;
        uint side = zl < zr ? 0u : 1u;
        float z = min(zl, zr);
        float hgt = CAMH - dy * z / FOC;
        if (z < 1e8 && hgt >= 0.0 && hgt <= bh(side, z)) return facade(side, z, hgt);
        vec3 c = sky(Y);
        /* the far end of the street, a low silhouette */
        float s = 0.012 + 0.045 * pow(h01(uint(floor(X * 38.0 + 400.0)), 9u, 9u), 2.0);
        if (VPY - Y < s && dy < 0.0) c = mix(c, mix(vec3(0.022, 0.018, 0.015), vec3(0.43, 0.42, 0.40), uDay), 0.85);
        return c;
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        float X = uv.x * uAsp, Y = 1.0 - uv.y;
        float dx = X - 0.64 * uAsp, dy = Y - VPY;
        vec3 c;
        float zl = dx < -1e-4 ? FOC * LL / -dx : 1e9, zr = dx > 1e-4 ? FOC * LR / dx : 1e9;
        float zf = min(zl, zr), zg = dy > 1e-4 ? FOC * CAMH / dy : 1e9;
        if (zg < zf) {
            /* the ground */
            float lat = dx * zg / FOC;
            bool road = lat > -7.6 && lat < 0.0;
            vec3 g = road ? mix(vec3(0.016, 0.014, 0.012), vec3(0.16, 0.155, 0.15), uDay)
                          : mix(vec3(0.032, 0.026, 0.021), vec3(0.25, 0.24, 0.23), uDay);
            g = mix(g, mix(vec3(0.09, 0.08, 0.075), vec3(0.62, 0.61, 0.60), uDay), uSnowGround * (road ? 0.5 : 1.0));
            float fwl = max(fwidth(lat), 1e-3);
            float dash = (1.0 - smoothstep(0.08, 0.08 + fwl, abs(lat + 3.8))) * step(fract(zg / 9.0), 0.45);
            g += dash * mix(vec3(0.05, 0.045, 0.04), vec3(0.18), uDay) * (1.0 - uSnowGround);
            /* pools of lamplight */
            for (int s = 0; s < 2; s++) {
                float z0 = s == 0 ? 9.0 : 21.0, la = s == 0 ? 0.8 : -8.4;
                float zz = z0 + 24.0 * clamp(floor((zg - z0) / 24.0 + 0.5), 0.0, 15.0);
                float d2 = (zg - zz) * (zg - zz) + (lat - la) * (lat - la);
                g += vec3(0.30, 0.17, 0.07) * exp(-d2 / 30.0) * uNight * 0.5;
            }
            /* wet: the street mirrored in it, rippled */
            float ripple = sin(Y * 420.0 + sin(X * 37.0) * 3.0) * 0.0025 * uWet;
            float fres = mix(0.18, 0.6, exp(-dy * 6.0));
            c = g + upper(X + ripple, 2.0 * VPY - Y) * fres * uWet;
            c = mix(c, sky(VPY - 0.02), 1.0 - exp(-zg / 260.0));
        } else {
            c = upper(X, Y);
        }
        o = vec4(c, 1.0);
    });

/* lights, instanced: a bokeh disc or a sharp point, or a wet-road streak */
static const char *VS_LIGHT = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 a0;   /* x, y (px, y down), bokeh r, sharp r */
    layout(location = 1) in vec4 a1;   /* rgb, shape */
    uniform vec2 uScreen;
    uniform float uBokeh;
    out vec2 vP;
    out vec3 vCol;
    out float vShape;
    out vec2 vC;
    void main() {
        vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
        float r = uBokeh > 0.5 ? a0.z : a0.w;
        vec2 ext = vec2(r, a1.w > 0.5 && a1.w < 1.5 ? r * (uBokeh > 0.5 ? 2.2 : 4.0) : r);
        if (a1.w > 0.5 && a1.w < 1.5) ext.x *= 0.8;
        vec2 p = a0.xy + c * ext + (a1.w > 0.5 && a1.w < 1.5 ? vec2(0.0, ext.y * 0.6) : vec2(0.0));
        gl_Position = vec4(p.x / uScreen.x * 2.0 - 1.0, 1.0 - p.y / uScreen.y * 2.0, 0.0, 1.0);
        vP = c;
        vShape = a1.w;
        vC = a0.xy / uScreen * 2.0 - 1.0;
        /* the same light spread over a bigger disc is dimmer */
        vCol = a1.rgb * (uBokeh > 0.5 ? 1.0 : 2.6);
    });
static const char *FS_LIGHT = "#version 330 core\n" GLSL(
    in vec2 vP;
    in vec3 vCol;
    in float vShape;
    in vec2 vC;
    uniform float uBokeh, uEdge;
    out vec4 o;
    void main() {
        float d = length(vP), v;
        if (vShape < 0.5) {
            if (uBokeh > 0.5) {
                /* a disc with a brighter rim, clipped into a cat's eye
                 * towards the corners */
                v = 1.0 - smoothstep(1.0 - uEdge, 1.0, d);
                vec2 q = vP + vC * vec2(1.0, -1.0) * 0.45;
                v *= 1.0 - smoothstep(1.22 - uEdge, 1.22, length(q));
                v *= 0.8 + 0.3 * smoothstep(0.55, 0.97, d) + 0.03 * sin(d * 23.0);
            } else {
                v = exp(-d * d * 7.0) + 0.12 * exp(-d * d * 1.5);
            }
        } else if (vShape < 1.5) {
            /* reflection: a soft vertical streak, broken by ripples */
            vec2 p = vP;
            v = exp(-p.x * p.x * 4.0 - (p.y + 0.2) * (p.y + 0.2) * 1.6) * (0.75 + 0.25 * sin(p.y * 9.0 + p.x * 3.0));
        } else {
            v = exp(-d * d * 3.5);
        }
        if (v < 0.002) discard;
        o = vec4(vCol * v, 1.0);
    });

/* separable blur, 5 linear taps */
static const char *FS_BLUR = "#version 330 core\n" GLSL(
    uniform sampler2D uSrc;
    uniform vec2 uRes, uStep;
    uniform float uLod;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec3 c = textureLod(uSrc, uv, uLod).rgb * 0.2270270;
        c += (textureLod(uSrc, uv + uStep * 1.3846154, uLod).rgb + textureLod(uSrc, uv - uStep * 1.3846154, uLod).rgb) * 0.3162162;
        c += (textureLod(uSrc, uv + uStep * 3.2307692, uLod).rgb + textureLod(uSrc, uv - uStep * 3.2307692, uLod).rgb) * 0.0702703;
        o = vec4(c, 1.0);
    });

/* a flat value (fog regrowing, by constant-alpha blending) */
static const char *FS_FILL = "#version 330 core\n" GLSL(
    uniform vec4 uVal;
    out vec4 o;
    void main() { o = uVal; });

/* drops, instanced: one quad each */
static const char *VS_DROP = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 a0;   /* x, y (px, y down), r, stretch (flakes: melt) */
    layout(location = 1) in vec4 a1;   /* direction of travel (y down), type, seed */
    uniform vec2 uScreen;
    uniform float uRScale;
    out vec2 vP;
    out float vR, vS, vType, vSeed;
    out vec2 vDir;
    void main() {
        vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
        float r = a0.z * uRScale;
        float ext = (a1.z > 0.5 ? r : r * (1.0 + a0.w) * 1.15) + 1.5;
        vec2 p = a0.xy + c * ext;
        gl_Position = vec4(p.x / uScreen.x * 2.0 - 1.0, 1.0 - p.y / uScreen.y * 2.0, 0.0, 1.0);
        vP = c * ext;
        vR = r;
        vS = a0.w;
        vType = a1.z;
        vSeed = a1.w;
        vDir = a1.xy;
    });

/* the drop's surface: its normal (xy, y up) and thickness, and coverage;
 * the highest surface wins through the depth test. Flakes write ice. */
static const char *FS_DROP = "#version 330 core\n" GLSL(
    in vec2 vP;
    in float vR, vS, vType, vSeed;
    in vec2 vDir;
    layout(location = 0) out vec4 oN;
    layout(location = 1) out vec4 oIce;
    void main() {
        vec2 p = vP / vR;
        if (vType < 0.5) {
            vec2 pd = vec2(-vDir.y, vDir.x);
            float al = dot(p, vDir), pe = dot(p, pd);
            if (al < 0.0) { al /= 1.0 + vS; pe *= 1.0 + 0.7 * vS * min(1.0, -al); }
            else al *= 1.0 + 0.12 * vS;
            float a = atan(p.y, p.x);
            float wob = 1.0 + (0.035 * sin(3.0 * a + vSeed * 6.28) + 0.02 * sin(5.0 * a + vSeed * 19.0) +
                               0.03 * sin(2.0 * a + vSeed * 41.0)) / (1.0 + 4.0 * vS);
            /* beads sag a little: fuller below */
            float d = length(vec2(al, pe)) * wob * (1.0 - 0.06 * clamp(p.y, -1.0, 1.0) * (1.0 - min(vS, 1.0)));
            if (d >= 1.0) discard;
            float h = sqrt(1.0 - d * d);
            vec2 n = al * vDir + pe * pd;
            n = n / max(length(n), 1e-4) * d;
            float aa = 1.0 - smoothstep(1.0 - 1.4 / max(vR, 0.7), 1.0, d);
            oN = vec4(n.x * 0.5 + 0.5, -n.y * 0.5 + 0.5, h, aa);
            oIce = vec4(0.0);
            gl_FragDepth = 1.0 - 0.9 * h;
        } else {
            float s = sin(vSeed * 6.28), k = cos(vSeed * 6.28);
            vec2 q = mat2(k, s, -s, k) * p;
            float a = atan(q.y, q.x), d = length(q);
            float arm = pow(abs(cos(3.0 * a)), 3.0);
            float lump = 0.42 + 0.14 * sin(5.0 * a + vSeed * 13.0) + 0.1 * sin(2.0 * a + vSeed * 7.0) + 0.06 * sin(11.0 * a + vSeed * 3.0);
            float R = mix(lump + 0.4 * arm * (0.5 + 0.5 * sin(vSeed * 29.0)), 0.5, vS);
            float ice = (1.0 - smoothstep(R * 0.35, R, d));
            ice *= 0.55 + 0.45 * sin(d * 11.0 + a * 2.0 + vSeed * 9.0) * sin(a * 7.0 + vSeed * 5.0);
            ice = clamp(ice, 0.0, 1.0) * 0.8;
            if (ice < 0.02) discard;
            vec2 n = normalize(p + 1e-4) * min(d / 0.6, 1.0) * 0.8;
            oN = vec4(n.x * 0.5 + 0.5, -n.y * 0.5 + 0.5, 0.3, clamp(vS * 1.3 * ice, 0.0, 1.0));
            oIce = vec4(ice * (1.0 - 0.85 * vS));
            gl_FragDepth = 0.02;
        }
    });

/* running drops wipe the fog off */
static const char *FS_WIPE = "#version 330 core\n" GLSL(
    in vec2 vP;
    in float vR, vS, vType, vSeed;
    in vec2 vDir;
    out vec4 o;
    void main() {
        float d = length(vP / vR);
        o = vec4(0.0, 0.0, 0.0, (1.0 - smoothstep(0.5, 1.0, d)) * 0.9);
    });

/* everything onto the screen */
static const char *FS_COMP = "#version 330 core\n" GLSL(
    uniform sampler2D uD, uIce, uS, uBl, uB, uF, uN;
    uniform vec2 uRes;
    uniform float uK, uBlur, uBokehGain, uFrost, uFlash, uExpo, uTime, uFogVeil;
    uniform vec3 uHaze;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        float asp = uRes.x / uRes.y;
        ivec2 ip = ivec2(gl_FragCoord.xy);
        vec4 d = texelFetch(uD, ip, 0);
        float ice = texelFetch(uIce, ip, 0).r;
        vec2 nuv = uv * vec2(asp, 1.0);
        vec4 nz = texture(uN, nuv * 1.3);
        float fog = texture(uF, uv).r * (0.55 + 0.9 * nz.r);
        /* frost, feathering in from the edges */
        float e = min(min(uv.x, 1.0 - uv.x) * asp, min(uv.y, 1.0 - uv.y));
        float fr = uFrost * (1.0 - smoothstep(0.0, 0.03 + 0.17 * uFrost, e + (nz.r - 0.5) * 0.14));
        float cry = texture(uN, nuv * 7.0).b;
        float frost = clamp(fr * fr * (0.55 + 0.9 * cry) * (0.8 + 0.4 * nz.g), 0.0, 1.0);
        float veil = clamp(fog * uFogVeil + frost * 0.6, 0.0, 1.0);

        vec3 bg = textureLod(uBl, uv, uBlur + veil * 1.6).rgb + textureLod(uB, uv, veil * 2.6).rgb * uBokehGain * (1.0 - 0.3 * veil);
        vec3 far = textureLod(uBl, uv, 5.5).rgb;
        vec3 haze = uHaze + far * 1.3;
        vec3 col = mix(bg, haze, veil * 0.55);
        col = mix(col, haze * 1.4 + far * 0.6 + vec3(0.035, 0.032, 0.028), frost * 0.75);

        if (d.a > 0.002) {
            vec2 n = d.xy * 2.0 - 1.0;
            float h = d.z, m = length(n);
            /* each drop a lens: a small, upside-down view of the street */
            vec2 uvR = uv - n * uK * vec2(1.0 / asp, 1.0);
            vec2 gx = dFdx(uvR), gy = dFdy(uvR);
            vec3 thru = textureGrad(uS, uvR, gx * 1.6, gy * 1.6).rgb * 1.2 + bg * 0.18;
            float rim = smoothstep(0.62, 1.0, m);
            vec3 dc = thru * (1.0 - 0.8 * rim * rim);
            /* a glint, up and to the left */
            vec3 n3 = normalize(vec3(n * 1.3, max(h, 0.06)));
            float sp = pow(max(dot(n3, normalize(vec3(-0.35, 0.55, 1.6))), 0.0), 70.0);
            dc += sp * (haze * 1.5 + far * 1.2 + vec3(0.05, 0.045, 0.04));
            col = mix(col, dc, d.a);
        }
        /* snow on the glass catches the light */
        vec3 ic = haze * 2.2 + bg * 0.6 + far * 1.2 + vec3(0.11, 0.10, 0.09);
        col = mix(col, ic, clamp(ice, 0.0, 1.0) * 0.9);

        col += uFlash * vec3(0.28, 0.27, 0.25) * (0.25 + veil);
        col *= uExpo;
        vec2 q = (uv - 0.5) * vec2(asp, 1.0);
        col *= mix(0.62, 1.0, smoothstep(1.25, 0.35, length(q)));
        col = 1.0 - exp(-col * 1.08);
        col += (hash(gl_FragCoord.xy + fract(uTime) * 91.7) - 0.5) * (2.0 / 255.0);
        o = vec4(col, 1.0);
    });

static GLuint p_street, p_light, p_blur, p_fill, p_drop, p_wipe, p_comp;
static GLuint vao_empty, vao_light, vbo_light, vao_drop, vbo_drop, tex_noise;
/* render targets: S street (half, mips), B bokeh (half), T/Bl blur (quarter,
 * Bl with mips), F fog (quarter), D drops (full: normal + ice + depth) */
static GLuint tS, fS, tB, fB, tT, fT, tBl, fBl, tF, fF, tD, tI, rbD, fD;
static int tw_, th_;
static float drop_inst[MAXD * 8];

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "rain: shader:\n%s\n", log);
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

static GLuint mk_tex(int w, int h, GLenum ifmt, GLenum fmt, GLenum type, int mips) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mips ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (mips) glGenerateMipmap(GL_TEXTURE_2D);
    return t;
}

static GLuint mk_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}

static int hw(void) { return W / 2 > 1 ? W / 2 : 1; }
static int hh(void) { return H / 2 > 1 ? H / 2 : 1; }
static int qw(void) { return W / 4 > 1 ? W / 4 : 1; }
static int qh(void) { return H / 4 > 1 ? H / 4 : 1; }

static void fog_clear(float v) {
    glBindFramebuffer(GL_FRAMEBUFFER, fF);
    glClearColor(v, v, v, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

/* (re)make the targets at the window's size */
static void targets(void) {
    if (tw_ == W && th_ == H) return;
    if (tw_) {
        GLuint t[] = {tS, tB, tT, tBl, tF, tD, tI}, f[] = {fS, fB, fT, fBl, fF, fD};
        glDeleteTextures(7, t);
        glDeleteFramebuffers(6, f);
        glDeleteRenderbuffers(1, &rbD);
    }
    tw_ = W; th_ = H;
    sc = H / 1080.0f * DROP_SCALE;
    int k = 0;                               /* resized: drop what's now off the glass */
    for (int i = 0; i < nd; i++)
        if (drops[i].x < W && drops[i].y < H) drops[k++] = drops[i];
    nd = k;
    grid_alloc();
    tS = mk_tex(hw(), hh(), GL_RGBA16F, GL_RGBA, GL_FLOAT, 1); fS = mk_fbo(tS);
    tB = mk_tex(hw(), hh(), GL_RGBA16F, GL_RGBA, GL_FLOAT, 1); fB = mk_fbo(tB);
    tT = mk_tex(qw(), qh(), GL_RGBA16F, GL_RGBA, GL_FLOAT, 0); fT = mk_fbo(tT);
    tBl = mk_tex(qw(), qh(), GL_RGBA16F, GL_RGBA, GL_FLOAT, 1); fBl = mk_fbo(tBl);
    tF = mk_tex(qw(), qh(), GL_R16F, GL_RED, GL_FLOAT, 0); fF = mk_fbo(tF);
    tD = mk_tex(W, H, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    tI = mk_tex(W, H, GL_R8, GL_RED, GL_UNSIGNED_BYTE, 0);
    fD = mk_fbo(tD);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tI, 0);
    glGenRenderbuffers(1, &rbD);
    glBindRenderbuffer(GL_RENDERBUFFER, rbD);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbD);
    GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("incomplete framebuffer");
    fog_clear(0.3f);
}

/* tileable value noise: r smooth fbm, g ridged, b fine crystals */
static void make_noise(void) {
    enum { N = 256 };
    static unsigned char px[N * N * 4];
    uint32_t keep = scene_seed;
    scene_seed = 0xf0f0u;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float v[3] = {0, 0, 0};
            for (int ch = 0; ch < 3; ch++) {
                float amp = 0.5f, sum = 0;
                for (int o = 0; o < 5; o++) {
                    int per = (ch == 2 ? 16 : 4) << o;
                    float fx = (float)x * per / N, fy = (float)y * per / N;
                    int ix = (int)fx, iy = (int)fy;
                    float tx = fx - ix, ty = fy - iy;
                    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
                    float c00 = h01(ix % per, iy % per, o + 10 * ch), c10 = h01((ix + 1) % per, iy % per, o + 10 * ch);
                    float c01 = h01(ix % per, (iy + 1) % per, o + 10 * ch), c11 = h01((ix + 1) % per, (iy + 1) % per, o + 10 * ch);
                    float n = mixf(mixf(c00, c10, tx), mixf(c01, c11, tx), ty);
                    if (ch == 1) n = 1 - fabsf(n * 2 - 1);
                    sum += n * amp;
                    v[ch] += amp;
                    amp *= 0.5f;
                }
                v[ch] = sum / v[ch];
            }
            v[1] = powf(v[1], 3.0f);
            for (int c = 0; c < 3; c++) px[(y * N + x) * 4 + c] = (unsigned char)(clampf(v[c], 0, 1) * 255);
            px[(y * N + x) * 4 + 3] = 255;
        }
    scene_seed = keep;
    glGenTextures(1, &tex_noise);
    glBindTexture(GL_TEXTURE_2D, tex_noise);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
}

static void inst_vao(GLuint *vao, GLuint *vbo) {
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    for (int i = 0; i < 2; i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, 32, (void *)(uintptr_t)(i * 16));
        glVertexAttribDivisor(i, 1);
    }
}

static void gl_init(void) {
    p_street = mk_prog(VS, FS_STREET);
    p_light = mk_prog(VS_LIGHT, FS_LIGHT);
    p_blur = mk_prog(VS, FS_BLUR);
    p_fill = mk_prog(VS, FS_FILL);
    p_drop = mk_prog(VS_DROP, FS_DROP);
    p_wipe = mk_prog(VS_DROP, FS_WIPE);
    p_comp = mk_prog(VS, FS_COMP);
    glGenVertexArrays(1, &vao_empty);
    inst_vao(&vao_light, &vbo_light);
    inst_vao(&vao_drop, &vbo_drop);
    make_noise();
    glUseProgram(p_comp);
    const char *names[] = {"uD", "uIce", "uS", "uBl", "uB", "uF", "uN"};
    for (int i = 0; i < 7; i++) glUniform1i(U(p_comp, names[i]), i);
}

/* ---- weather-driven state per frame ---- */
static double scene_t, dbg_sim, dbg_fog, dbg_ren;                 /* street time, s */
static float flash, flash_next = 6;
static float flash_pulses[3], flash_at;

static void lightning(float dt) {
    flash_next -= dt;
    if (wx_cur.storm > 0.5f && flash_next <= 0) {
        flash_at = 0;
        for (int i = 0; i < 3; i++) flash_pulses[i] = i == 0 ? 0 : flash_pulses[i - 1] + 0.06f + 0.18f * frand();
        flash_next = 7 + 20 * frand();
    }
    flash_at += dt;
    flash = 0;
    for (int i = 0; i < 3; i++) {
        float a = flash_at - flash_pulses[i];
        if (a >= 0 && a < 1.5f) flash += (i == 0 ? 0.9f : 0.55f) * expf(-a / 0.07f);
    }
    flash = fminf(flash, 1.0f) * smooth(0.5f, 1.0f, wx_cur.storm);
}

/* fog on the glass: regrow towards the weather's level, then let the
 * running drops wipe their tracks */
static void fog_step(float dt) {
    const Wx *w = &wx_cur;
    float cold = smooth(10, -2, w->temp);
    float target = clampf(0.18f + 0.45f * smooth(0, 0.6f, w->rain) + 0.3f * cold, 0, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, fF);
    glViewport(0, 0, qw(), qh());
    glBindVertexArray(vao_empty);
    glEnable(GL_BLEND);
    glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA);
    glBlendColor(0, 0, 0, 1 - expf(-dt / 22.0f));
    glUseProgram(p_fill);
    glUniform4f(U(p_fill, "uVal"), target, 0, 0, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    int n = 0;
    for (int i = 0; i < nd; i++) {
        Drop *d = &drops[i];
        if (!d->run || d->vy < 5 * sc) continue;
        float *p = &drop_inst[n++ * 8];
        p[0] = d->x; p[1] = d->y; p[2] = d->r; p[3] = 0; p[4] = 0; p[5] = 1; p[6] = 0; p[7] = 0;
    }
    if (n) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(vao_drop);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_drop);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)n * 32, drop_inst, GL_STREAM_DRAW);
        glUseProgram(p_wipe);
        glUniform2f(U(p_wipe, "uScreen"), (float)W, (float)H);
        glUniform1f(U(p_wipe, "uRScale"), 1.5f);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, n);
    }
    glDisable(GL_BLEND);
}

/* one simulation tick: weather, drops, fog */
static void step(float dt) {
    if (!opt_shot && !opt_bench) {
        weather_tick();
        wx_target(time(NULL));
        wx_ease(dt);
    }
    scene_t += dt;
    lightning(dt);
    sim_step(dt);
    fog_step(dt);
}

static void blur_pass(GLuint dst, GLuint src, float lod, float sx, float sy) {
    glBindFramebuffer(GL_FRAMEBUFFER, dst);
    glViewport(0, 0, qw(), qh());
    glUseProgram(p_blur);
    glBindTexture(GL_TEXTURE_2D, src);
    glUniform2f(U(p_blur, "uRes"), (float)qw(), (float)qh());
    glUniform2f(U(p_blur, "uStep"), sx, sy);
    glUniform1f(U(p_blur, "uLod"), lod);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static GLuint dbg_q[8];
static double dbg_pass[8];
static int dbg_on;
static void tq(int i) {
    if (!dbg_on) return;
    if (!dbg_q[0]) glGenQueries(8, dbg_q);
    if (i > 0) glEndQuery(GL_TIME_ELAPSED);
    if (i < 6) glBeginQuery(GL_TIME_ELAPSED, dbg_q[i]);
    if (i == 6) for (int k = 0; k < 6; k++) { GLuint64 ns; glGetQueryObjectui64v(dbg_q[k], GL_QUERY_RESULT, &ns); dbg_pass[k] += ns * 1e-6; }
}

static void render(GLuint target) {
    const Wx *w = &wx_cur;
    float day = sky_day, night = 1 - day;
    float rainy = smooth(0.0f, 0.35f, w->rain);
    float wet = clampf(0.45f + 0.55f * rainy, 0, 1);
    float snowg = smooth(0.5f, -1.5f, w->temp) * smooth(0.1f, 0.6f, w->snow);
    float t = (float)scene_t;

    /* the street, sharp */
    tq(0);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, fS);
    glViewport(0, 0, hw(), hh());
    glBindVertexArray(vao_empty);
    glUseProgram(p_street);
    glUniform2f(U(p_street, "uRes"), (float)hw(), (float)hh());
    glUniform1f(U(p_street, "uAsp"), (float)W / H);
    glUniform1f(U(p_street, "uT"), t);
    glUniform1f(U(p_street, "uDay"), day);
    glUniform1f(U(p_street, "uNight"), night);
    glUniform1f(U(p_street, "uDusk"), sky_dusk);
    glUniform1f(U(p_street, "uCloud"), w->cloud);
    glUniform1f(U(p_street, "uFlash"), flash);
    glUniform1f(U(p_street, "uWet"), wet);
    glUniform1f(U(p_street, "uSnowGround"), snowg);
    glUniform1f(U(p_street, "uLitP"), 0.06f + 0.3f * smooth(0.2f, 0.9f, night));
    glUniform1ui(U(p_street, "uSeed"), scene_seed);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* its lights, as points there and as bokeh discs */
    tq(1);
    scene_lights(t, night, wet, w->snow * smooth(0.05f, 0.2f, w->rain), rainy);
    glBindVertexArray(vao_light);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_light);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)nl * 32, lights, GL_STREAM_DRAW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glUseProgram(p_light);
    glUniform2f(U(p_light, "uScreen"), (float)W, (float)H);
    glUniform1f(U(p_light, "uBokeh"), 0);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, nl);
    glBindFramebuffer(GL_FRAMEBUFFER, fB);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUniform1f(U(p_light, "uBokeh"), 1);
    glUniform1f(U(p_light, "uEdge"), 2.5f / (bokeh_r * 0.5f));
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, nl);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tB);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tS);
    glGenerateMipmap(GL_TEXTURE_2D);

    /* the out-of-focus street */
    tq(2);
    glBindVertexArray(vao_empty);
    blur_pass(fT, tS, 1, 2.2f / qw(), 0);
    blur_pass(fBl, tT, 0, 0, 2.2f / qh());
    glBindTexture(GL_TEXTURE_2D, tBl);
    glGenerateMipmap(GL_TEXTURE_2D);

    /* the water */
    tq(3);
    int n = 0;
    for (int i = 0; i < nd; i++) {
        Drop *d = &drops[i];
        float *p = &drop_inst[n++ * 8];
        float vx = d->run ? d->wx * fminf(1, d->vy / (sc * 40)) + w->wind * d->vy * 0.55f : 0, vy = d->run ? d->vy : 1;
        float l = sqrtf(vx * vx + vy * vy);
        p[0] = d->x; p[1] = d->y; p[2] = d->r;
        p[3] = d->type ? d->melt : (d->run ? clampf(d->vy / (sc * 220), 0, 1.3f) : 0);
        p[4] = l > 0 ? vx / l : 0; p[5] = l > 0 ? vy / l : 1;
        p[6] = d->type; p[7] = d->seed;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fD);
    glViewport(0, 0, W, H);
    float cn[4] = {0.5f, 0.5f, 0, 0}, ci[4] = {0, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 0, cn);
    glClearBufferfv(GL_COLOR, 1, ci);
    glClearDepth(1);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    if (n) {
        glBindVertexArray(vao_drop);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_drop);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)n * 32, drop_inst, GL_STREAM_DRAW);
        glUseProgram(p_drop);
        glUniform2f(U(p_drop, "uScreen"), (float)W, (float)H);
        glUniform1f(U(p_drop, "uRScale"), 1);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, n);
    }
    glDisable(GL_DEPTH_TEST);

    /* onto the screen */
    tq(4);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glBindVertexArray(vao_empty);
    GLuint tx[] = {tD, tI, tS, tBl, tB, tF, tex_noise};
    for (int i = 6; i >= 0; i--) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, tx[i]);
    }
    glUseProgram(p_comp);
    glUniform2f(U(p_comp, "uRes"), (float)W, (float)H);
    glUniform1f(U(p_comp, "uK"), 0.2f);
    glUniform1f(U(p_comp, "uBlur"), mixf(0.3f, 1.3f, rainy));
    glUniform1f(U(p_comp, "uBokehGain"), 1.0f);
    glUniform1f(U(p_comp, "uFrost"), smooth(1.5f, -4.0f, w->temp));
    glUniform1f(U(p_comp, "uFlash"), flash);
    glUniform1f(U(p_comp, "uExpo"), mixf(1.1f, 1.0f, day));
    glUniform1f(U(p_comp, "uTime"), t);
    glUniform1f(U(p_comp, "uFogVeil"), 1.0f);
    glUniform3f(U(p_comp, "uHaze"), mixf(0.04f, 0.08f, day), mixf(0.033f, 0.077f, day), mixf(0.027f, 0.072f, day));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    tq(5);
    tq(6);
}

/* start with glass that has been out in this weather a while: half a
 * minute of rain, then ten seconds of what it's doing now */
static void warm_up(void) {
    Wx real = wx_cur, wet = real;
    if (wet.rain < 0.35f) { wet.rain = 0.35f; wet.size = 1.0f; }
    wx_cur = wet;
    fog_clear(0.3f);
    for (int i = 0; i < 30 * 30; i++) sim_step(1.0f / 30);
    wx_cur = real;
    for (int i = 0; i < 10 * 30; i++) step(1.0f / 30);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "rain");
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
        "usage: rain [--window | --root | --layer] [--seed N] [--lat LAT --lon LON]\n"
        "            [--weather rain|drizzle|storm|clear|snow] [--hour H] [--fps N]\n"
        "            [--shot FILE.ppm [--at SECONDS] [--size WxH]] [--bench N [--size WxH]]\n");
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
        else if (!strcmp(a, "-lat") && i + 1 < argc) opt_lat = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-lon") && i + 1 < argc) opt_lon = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-hour") && i + 1 < argc) opt_hour = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-weather") && i + 1 < argc) {
            const char *w = argv[++i];
            for (int k = 0; k < 5; k++)
                if (!strcmp(w, WX_NAMES[k])) opt_weather = k;
            if (opt_weather < 0) usage();
        }
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (opt_fps < 5) opt_fps = 5;
    if (opt_hour >= 24) usage();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    scene_seed = opt_seed;
    /* every monitor gets the same street but its own water */
    uint32_t oh = 2166136261u;
    for (const char *p = getenv("SWAYLOCK_PLUGIN_OUTPUT_NAME"); p && *p; p++) oh = (oh ^ (unsigned char)*p) * 16777619u;
    rng = (opt_seed * 2654435761u) ^ oh;
    if (!rng) rng = 1;
    int offscreen = opt_shot || opt_bench > 0;
    if (offscreen) {
        opt_layer = 0;
        opt_lat = opt_lon = NAN;            /* no network for shots and benches */
        have_loc = 0;
    } else location_init();
    if (have_loc && opt_weather < 0) {
        load_cache();
        if (have_real && time(NULL) - real_at > 3 * 3600) have_real = 0;
    }
    time_t t0w = time(NULL);
    wx_target(t0w);
    wx_cur = wx_tgt;
    sky_update(t0w);

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer) wl_init();
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
        XStoreName(dpy, win, "Rain");
        XClassHint chint = {"rain", "Rain"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();
    targets();
    scene_t = 200 + (opt_seed % 1000);
    warm_up();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (opt_bench > 0) {
            double sum = 0, worst = 0;
            dbg_on = getenv("RAIN_DEBUG") != NULL;
            render(fbo);
            glFinish();
            for (int i = 0; i < opt_bench; i++) {
                double a = now();
                if (getenv("RAIN_DEBUG")) { glFinish(); double b = now(); sim_step(1.0f / 60); dbg_sim += now() - b; b = now(); fog_step(1.0f / 60); glFinish(); dbg_fog += now() - b; b = now(); render(fbo); glFinish(); dbg_ren += now() - b; }
                else {
                step(1.0f / 60);
                render(fbo);
                glFinish();
                }
                double ms = (now() - a) * 1000;
                sum += ms;
                if (ms > worst) worst = ms;
            }
            printf("%dx%d, %d frames: avg %.2f ms, worst %.2f ms  (%s; %d drops)\n", W, H, opt_bench, sum / opt_bench,
                   worst, (const char *)glGetString(GL_RENDERER), nd);
            if (getenv("RAIN_DEBUG")) { printf("sim %.2f fog %.2f render %.2f ms; gpu street %.2f lights %.2f blur %.2f drops %.2f comp %.2f\n", dbg_sim * 1000 / opt_bench, dbg_fog * 1000 / opt_bench, dbg_ren * 1000 / opt_bench,
                dbg_pass[0] / opt_bench, dbg_pass[1] / opt_bench, dbg_pass[2] / opt_bench, dbg_pass[3] / opt_bench, dbg_pass[4] / opt_bench); }
            return 0;
        }
        int steps = (int)(opt_at * 60);
        for (int i = 0; i < steps; i++) step(1.0f / 60);
        render(fbo);
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }

    double last = now(), last_sky = 0;
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
        double t = now();
        float dt = (float)(t - last);
        if (dt > 0.1f) dt = 0.1f;
        last = t;
        if (t - last_sky > 30) { sky_update(time(NULL)); last_sky = t; }
        targets();
        step(dt);
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
