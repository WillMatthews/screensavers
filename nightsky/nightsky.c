/*
 * nightsky — the real sky above you right now, over a dark mountain lake.
 *
 * The ~800 brightest stars sit where they really are (stars.h, from the
 * Yale Bright Star Catalogue: complete to magnitude 3.5, nearly so to 4,
 * most to 4.5), with thousands of fainter made-up ones crowding round the
 * Milky Way the way the real ones do. The
 * Sun, Moon and the five naked-eye planets are worked out for the moment:
 * low-precision formulas (Meeus for the Moon, JPL's mean elements for the
 * planets), good to a few arcminutes, which is far finer than a pixel.
 * Positions are precessed from J2000 to the date, turned to the local sky
 * with the sidereal time, lifted by refraction and dimmed and reddened by
 * the air near the horizon, where stars also twinkle hardest.
 *
 * The sky colour follows the real Sun: deep warm night, then astronomical,
 * nautical and civil twilight glowing on the side the Sun is, then a pale
 * day sky where only the Moon shows. The Moon is drawn lit from the Sun's
 * real direction, so its phase and the tilt of the crescent are right, with
 * its seas where they are; it is drawn three times its true size (--moon).
 * Over it all: a faint Milky Way with its dust lanes, the odd meteor, thin
 * drifting cloud (--clouds 0 for none) and the constellation figures in dim
 * gold (--no-lines to leave them out), named like an old star atlas: the
 * constellations in spaced capitals, the planets and brightest stars in
 * italic beside them (--no-labels; --font NAME, EB Garamond by default).
 * Names only come out once it's dark, and stand back from the land. Below: made-up mountain ridges, rim-lit when the Moon
 * or the twilight is behind them, and a still lake mirroring the lot.
 *
 * Where: --lat LAT --lon LON (degrees, north and east positive), else the
 * first "LAT LON" line of ~/.config/wallpapers/location (e.g. "55.95 -3.19";
 * $XDG_CONFIG_HOME is honoured; lines starting # are skipped), else Greenwich
 * (51.48, 0.0). It never looks your location up and never shows it.
 * When: now, or --time "YYYY-MM-DD HH:MM" (local time; TZ=UTC for UT).
 * The sky turns at its real rate, too slowly to see, so the twinkling,
 * meteors and clouds are what move; --timelapse X runs the sky X times fast.
 * The view faces the equator (south in the north), nudged so the Moon is
 * in the frame when it's up; --az DEG pins it (0 = north, 90 = east),
 * --pitch DEG tilts it up (18) and --fov DEG sets the width (90; e.g.
 * --fov 4 --az A --pitch ALT to look closely at the Moon). --info prints the
 * Sun, Moon, planets and a few stars for the moment, to check against an
 * almanac. --seed N picks the mountains, clouds, meteors and the distant
 * town's glow; the stars are the same in every run.
 *
 * Spanning: run one instance per monitor with the same --seed, --epoch and
 * --layout (every output's logical rect), and --output = this one's index.
 * Then the sky clock runs from the shared epoch and the view is one
 * panorama across the wall: a cylinder round the eye, so the horizon is one
 * straight line through every screen and the mountains and lake run on
 * across the bezels, centred on --az (or the chosen bearing) in the middle
 * of the "front" output, the one under the middle of the layout. --fov is
 * then twice the angle from there to the far edge of the wall (default 280),

 * and the horizon sits 80% of the way down the front output unless --pitch
 * is given. The figures and names run across every output.
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), --shot
 * FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen and
 * --bench N to time N offscreen frames.
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
#include "stars.h"

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_lines = 1, opt_labels = 1, opt_bench, opt_info;
static const char *opt_font = "EB Garamond";
static double opt_lat = 51.48, opt_lon = 0.0;   /* Greenwich */
static int loc_given;
static const char *opt_time;
static double opt_timelapse = 1;
static double opt_az = NAN, opt_pitch = 18, opt_fov = 90;
static int opt_pitch_set, opt_fov_set;
static double opt_epoch;                         /* CLOCK_REALTIME seconds the sky's clock starts at */

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];                   /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;
static float opt_clouds = 0.32f;                 /* how much of the sky they cover, about */
static float opt_moon = 3;                       /* the Moon drawn this many times its size */
static int opt_fps = 30;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 25;
static int off_w = 1920, off_h = 1080;

#define PI 3.14159265358979323846
#define D2R (PI / 180)
#define R2D (180 / PI)

static void die(const char *what) { fprintf(stderr, "nightsky: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0f / 16777216.0f);
}
static uint32_t hash32(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x8da6b343u ^ b * 0xd8163841u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h ? h : 1;
}
static double clampd(double x, double a, double b) { return x < a ? a : x > b ? b : x; }
static double smoothd(double a, double b, double x) {
    double t = clampd((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

/* ---- noise (from flightsim) for the ridges ---- */
static float rand01(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h & 0xffffff) / (float)0x1000000;
}
static float smooth(float t) { return t * t * (3 - 2 * t); }
static float vnoise(float x, float z, uint32_t salt) {
    float fx = floorf(x), fz = floorf(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    float tx = smooth(x - fx), tz = smooth(z - fz);
    float a = rand01(ix, iz, salt), b = rand01(ix + 1, iz, salt);
    float c = rand01(ix, iz + 1, salt), d = rand01(ix + 1, iz + 1, salt);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}
static float fbm(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        s += vnoise(x, z, salt + i) * amp;
        norm += amp; amp *= 0.5f; x *= 2.03f; z *= 2.03f;
    }
    return s / norm;
}
static float ridged(float x, float z, int oct, uint32_t salt) {
    float s = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < oct; i++) {
        float n = 1 - fabsf(vnoise(x, z, salt + i) * 2 - 1);
        s += n * n * amp;
        norm += amp; amp *= 0.5f; x *= 2.07f; z *= 2.07f;
    }
    return s / norm;
}

/* ================= astronomy ================= */
typedef struct { double x, y, z; } v3;
typedef struct { double m[3][3]; } m3;

static v3 vec(double x, double y, double z) { return (v3){x, y, z}; }
static double dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 add(v3 a, v3 b) { return vec(a.x + b.x, a.y + b.y, a.z + b.z); }
static v3 sub(v3 a, v3 b) { return vec(a.x - b.x, a.y - b.y, a.z - b.z); }
static v3 scale(v3 a, double s) { return vec(a.x * s, a.y * s, a.z * s); }
static v3 cross(v3 a, v3 b) { return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static double len(v3 a) { return sqrt(dot(a, a)); }
static v3 norm(v3 a) { double l = len(a); return l > 0 ? scale(a, 1 / l) : a; }
static v3 radec(double ra, double dec) { return vec(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)); }

static m3 mmul(m3 a, m3 b) {
    m3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return r;
}
static v3 mvec(m3 a, v3 v) {
    return vec(a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
               a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
               a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}
static m3 mT(m3 a) {
    m3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) r.m[i][j] = a.m[j][i];
    return r;
}
/* rotations of the coordinate frame (passive), as in Meeus */
static m3 R1(double a) { double c = cos(a), s = sin(a); return (m3){{{1, 0, 0}, {0, c, s}, {0, -s, c}}}; }
static m3 R2(double a) { double c = cos(a), s = sin(a); return (m3){{{c, 0, -s}, {0, 1, 0}, {s, 0, c}}}; }
static m3 R3(double a) { double c = cos(a), s = sin(a); return (m3){{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}}; }

/* equatorial J2000 -> galactic (rows: towards the centre, l = 90, the north pole) */
static const m3 EQ2GAL = {{{-0.0548755604162154, -0.8734370902348850, -0.4838350155487132},
                           {+0.4941094278755837, -0.4448296299600112, +0.7469822444972189},
                           {-0.8676661490190047, -0.1980763734312015, +0.4559837761750669}}};

/* precession J2000 -> mean equator and equinox of date (IAU 1976) */
static m3 precession(double T) {
    double as = D2R / 3600;
    double zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) * as;
    double z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) * as;
    double th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T * T * T) * as;
    return mmul(R3(-z), mmul(R2(th), R3(-zeta)));
}

/* JPL approximate Keplerian elements, 1800-2050 (Standish): a, e, I, L,
 * long. of perihelion, long. of node, then their rates per century */
static const double ELEM[6][12] = {
    {0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
     0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081},
    {0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
     0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418},
    {1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
     0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0},
    {1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
     0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343},
    {5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
     -0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106},
    {9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448,
     -0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794},
};
enum { MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN };

/* heliocentric ecliptic J2000 position, au */
static v3 helio(int p, double T) {
    const double *e = ELEM[p];
    double a = e[0] + e[6] * T, ec = e[1] + e[7] * T, I = (e[2] + e[8] * T) * D2R;
    double L = e[3] + e[9] * T, pi = e[4] + e[10] * T, node = (e[5] + e[11] * T) * D2R;
    double M = fmod((L - pi) * D2R, 2 * PI), w = pi * D2R - node;
    double E = M;
    for (int i = 0; i < 12; i++) E = E - (E - ec * sin(E) - M) / (1 - ec * cos(E));
    double xp = a * (cos(E) - ec), yp = a * sqrt(1 - ec * ec) * sin(E);
    double cw = cos(w), sw = sin(w), cn = cos(node), sn = sin(node), ci = cos(I), si = sin(I);
    return vec((cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp,
               (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp,
               (sw * si) * xp + (cw * si) * yp);
}

static v3 ecl2eq(v3 v, double eps) { return mvec(R1(-eps), v); }

/* the Moon, geocentric, equatorial of date, km (Meeus ch. 47, the larger terms) */
static v3 moon_geo(double T) {
    double Lp = 218.3164477 + 481267.88123421 * T;
    double D = (297.8501921 + 445267.1114034 * T) * D2R;
    double M = (357.5291092 + 35999.0502909 * T) * D2R;
    double Mp = (134.9633964 + 477198.8675055 * T) * D2R;
    double F = (93.2720950 + 483202.0175233 * T) * D2R;
    double E = 1 - 0.002516 * T;
    static const struct { signed char d, m, mp, f; int l, r; } LR[] = {
        {0, 0, 1, 0, 6288774, -20905355}, {2, 0, -1, 0, 1274027, -3699111}, {2, 0, 0, 0, 658314, -2955968},
        {0, 0, 2, 0, 213618, -569925}, {0, 1, 0, 0, -185116, 48888}, {0, 0, 0, 2, -114332, -3149},
        {2, 0, -2, 0, 58793, 246158}, {2, -1, -1, 0, 57066, -152138}, {2, 0, 1, 0, 53322, -170733},
        {2, -1, 0, 0, 45758, -204586}, {0, 1, -1, 0, -40923, -129620}, {1, 0, 0, 0, -34720, 108743},
        {0, 1, 1, 0, -30383, 104755}, {2, 0, 0, -2, 15327, 10321}, {0, 0, 1, 2, -12528, 0},
        {0, 0, 1, -2, 10980, 79661}, {4, 0, -1, 0, 10675, -34782}, {0, 0, 3, 0, 10034, -23210},
        {4, 0, -2, 0, 8548, -21636}, {2, 1, -1, 0, -7888, 24208}, {2, 1, 0, 0, -6766, 30824},
        {1, 0, -1, 0, -5163, -8379}, {1, 1, 0, 0, 4987, -16675}, {2, -1, 1, 0, 4036, -12831},
        {2, 0, 2, 0, 3994, -10445}, {4, 0, 0, 0, 3861, -11650}, {2, 0, -3, 0, 3665, 14403},
        {0, 1, -2, 0, -2689, -7003}, {2, 0, -1, 2, -2602, 0}, {2, -1, -2, 0, 2390, 10056},
        {1, 0, 1, 0, -2348, 6322}, {2, -2, 0, 0, 2236, -9884},
    };
    static const struct { signed char d, m, mp, f; int b; } B[] = {
        {0, 0, 0, 1, 5128122}, {0, 0, 1, 1, 280602}, {0, 0, 1, -1, 277693}, {2, 0, 0, -1, 173237},
        {2, 0, -1, 1, 55413}, {2, 0, -1, -1, 46271}, {2, 0, 0, 1, 32573}, {0, 0, 2, 1, 17198},
        {2, 0, 1, -1, 9266}, {0, 0, 2, -1, 8822}, {2, -1, 0, -1, 8216}, {2, 0, -2, -1, 4324},
        {2, 0, 1, 1, 4200}, {2, 1, 0, -1, -3359}, {2, -1, -1, 1, 2463}, {2, -1, 0, 1, 2211},
    };
    double sl = 0, sr = 0, sb = 0;
    for (size_t i = 0; i < sizeof LR / sizeof *LR; i++) {
        double arg = LR[i].d * D + LR[i].m * M + LR[i].mp * Mp + LR[i].f * F;
        double k = abs(LR[i].m) == 1 ? E : abs(LR[i].m) == 2 ? E * E : 1;
        sl += LR[i].l * k * sin(arg);
        sr += LR[i].r * k * cos(arg);
    }
    for (size_t i = 0; i < sizeof B / sizeof *B; i++) {
        double arg = B[i].d * D + B[i].m * M + B[i].mp * Mp + B[i].f * F;
        double k = abs(B[i].m) == 1 ? E : abs(B[i].m) == 2 ? E * E : 1;
        sb += B[i].b * k * sin(arg);
    }
    double lam = (Lp + sl / 1e6) * D2R, bet = sb / 1e6 * D2R, dist = 385000.56 + sr / 1000;
    double eps = (23.439291 - 0.0130042 * T) * D2R;
    return scale(ecl2eq(radec(lam, bet), eps), dist);
}

/* the sky at one moment, for the renderer */
static struct {
    m3 eq2hor;                  /* equatorial J2000 -> local east, north, up */
    m3 hor2gal;
    v3 sun, moon;               /* apparent directions, local */
    v3 moon_sun;                /* from the Moon towards the Sun, local */
    v3 moon_w, moon_a;          /* the Moon's disc: towards sky west, and celestial north */
    double sun_alt, moon_alt;   /* degrees, apparent */
    double moon_k;              /* illuminated fraction */
    double moon_sd;             /* semi-diameter, radians */
    float planet[5][8];         /* vertex data, as the stars' */
} sky;

static double refraction_deg(double alt) {
    if (alt < -1.5) return 0;
    return 1.02 / tan((alt + 10.3 / (alt + 5.11)) * D2R) / 60;
}
static v3 refract(v3 d) {
    double alt = asin(clampd(d.z, -1, 1)) * R2D, alt2 = alt + refraction_deg(alt);
    double h = hypot(d.x, d.y);
    if (h < 1e-9) return d;
    double c = cos(alt2 * D2R);
    return vec(d.x / h * c, d.y / h * c, sin(alt2 * D2R));
}

/* B-V to a display colour (Ballesteros' temperature, Helland's blackbody fit) */
static void bv_rgb(double bv, float *rgb) {
    double t = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62)) / 100;
    double r = t <= 66 ? 255 : 329.698727446 * pow(t - 60, -0.1332047592);
    double g = t <= 66 ? 99.4708025861 * log(t) - 161.1195681661 : 288.1221695283 * pow(t - 60, -0.0755148492);
    double b = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * log(t - 10) - 305.0447927307;
    r = clampd(r, 0, 255); g = clampd(g, 0, 255); b = clampd(b, 0, 255);
    double m = fmax(r, fmax(g, b));
    rgb[0] = (float)(r / m); rgb[1] = (float)(g / m); rgb[2] = (float)(b / m);
}

static void sky_compute(double unix_t) {
    double jd = unix_t / 86400.0 + 2440587.5;
    double T = (jd - 2451545.0) / 36525;                       /* UT */
    double Tt = T + 69.2 / 86400 / 36525;                      /* TT, near enough */
    double gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T;
    double lst = fmod(gmst + opt_lon, 360) * D2R;
    double lat = opt_lat * D2R;
    m3 L = {{{0, 1, 0}, {-sin(lat), 0, cos(lat)}, {cos(lat), 0, sin(lat)}}};
    m3 P = precession(Tt);
    m3 date2hor = mmul(L, R3(lst));
    sky.eq2hor = mmul(date2hor, P);
    sky.hor2gal = mmul(EQ2GAL, mT(sky.eq2hor));

    /* Sun and planets, from the Earth, equatorial J2000 */
    double eps0 = 23.43928 * D2R;
    v3 earth = helio(EARTH, Tt);
    v3 sun_eq = ecl2eq(scale(earth, -1), eps0);
    static const int pl[5] = {MERCURY, VENUS, MARS, JUPITER, SATURN};
    static const double bvs[5] = {0.93, 0.82, 1.36, 0.83, 1.04};
    for (int i = 0; i < 5; i++) {
        v3 h = helio(pl[i], Tt), g = sub(h, earth);
        double r = len(h), dl = len(g);
        double ph = acos(clampd(dot(h, g) / (r * dl), -1, 1)) * R2D;  /* phase angle */
        double m = 5 * log10(r * dl);
        switch (pl[i]) {
        case MERCURY: m += -0.42 + 0.038 * ph - 0.000273 * ph * ph + 0.000002 * ph * ph * ph; break;
        case VENUS:   m += -4.40 + 0.0009 * ph + 0.000239 * ph * ph - 0.00000065 * ph * ph * ph; break;
        case MARS:    m += -1.52 + 0.016 * ph; break;
        case JUPITER: m += -9.40 + 0.005 * ph; break;
        default:      m += -8.88 + 0.044 * ph; break;
        }
        v3 e = norm(ecl2eq(g, eps0));
        float *o = sky.planet[i];
        o[0] = (float)e.x; o[1] = (float)e.y; o[2] = (float)e.z;
        bv_rgb(bvs[i], o + 3);
        o[6] = (float)m; o[7] = -1;                            /* -1: a planet, steady */
    }
    v3 sun_hor = norm(mvec(sky.eq2hor, sun_eq));
    sky.sun = refract(sun_hor);
    sky.sun_alt = asin(sky.sun.z) * R2D;

    /* the Moon: geocentric of date, then from where you stand */
    v3 mg = moon_geo(Tt);
    double phi = lat;
    v3 obs = scale(vec(cos(phi) * cos(lst), cos(phi) * sin(lst), sin(phi)), 6378.14);
    v3 mt = sub(mg, obs);
    v3 m_j2000 = mvec(mT(P), mt);
    sky.moon = refract(norm(mvec(sky.eq2hor, m_j2000)));
    sky.moon_alt = asin(sky.moon.z) * R2D;
    sky.moon_sd = asin(1737.4 / len(mt));
    v3 sun_km = scale(sun_eq, 1.495978707e8);
    sky.moon_sun = norm(mvec(sky.eq2hor, sub(sun_km, m_j2000)));
    sky.moon_k = (1 - dot(norm(sun_eq), norm(mvec(mT(P), mg)))) / 2;
    v3 np = vec(0, cos(lat), sin(lat));                      /* the celestial pole, local */
    v3 east = norm(cross(np, sky.moon));
    sky.moon_w = scale(east, -1);
    sky.moon_a = cross(sky.moon, east);
}

/* ================= the catalogue, as vertices ================= */
/* per star: J2000 unit vector, rgb, V, twinkle seed (-1 for planets) */
static float *starv;
static int nstars;
#define NFILL 9000

static int find_star(const char *bayer, const char *con) {
    char want[32];
    snprintf(want, sizeof want, "%s %s", bayer, con);
    for (int i = 0; i < NBRIGHT; i++)
        if (!strcmp(BRIGHT[i].name, want)) return i;
    return -1;
}

static void put_star(float *o, v3 e, double v, double bv, float tw) {
    o[0] = (float)e.x; o[1] = (float)e.y; o[2] = (float)e.z;
    bv_rgb(bv, o + 3);
    o[6] = (float)v; o[7] = tw;
}

static void build_stars(void) {
    nstars = NBRIGHT + NFILL;
    starv = malloc(sizeof(float) * 8 * nstars);
    for (int i = 0; i < NBRIGHT; i++)
        put_star(starv + 8 * i, radec(BRIGHT[i].ra * 15 * D2R, BRIGHT[i].dec * D2R), BRIGHT[i].v, BRIGHT[i].bv,
                 (float)((hash32(i, 77) & 0xffff) / 65536.0));
    /* the faint ones: counts rising ~x3 a magnitude, crowding the galactic
     * plane more the fainter they are; always the same sky */
    uint32_t keep = rng;
    rng = 0x5eed1234;
    m3 gal2eq = mT(EQ2GAL);
    for (int i = 0; i < NFILL; i++) {
        double lo = pow(10, 0.5 * 4.6), hi = pow(10, 0.5 * 7.0);
        double m = log10(lo + frand() * (hi - lo)) / 0.5;
        double l = frand() * 2 * PI, b;
        double plane = 0.25 + 0.2 * (m - 4.6) / 2.4;
        if (frand() < plane) {
            double u = frand() * 0.999 + 0.0005;
            b = (u < 0.5 ? log(2 * u) : -log(2 - 2 * u)) * 10 * D2R;   /* Laplace, scale 10 deg */
            if (frand() < 0.5) l = (frand() - 0.5) * 2 * 1.1;         /* more towards the centre */
            if (fabs(b) > PI / 2) b = copysign(PI / 2 - 0.01, b);
        } else
            b = asin(frand() * 2 - 1);
        v3 g = vec(cos(b) * cos(l), cos(b) * sin(l), sin(b));
        double bv = 0.1 + 1.2 * frand() * frand() + 0.5 * frand();
        put_star(starv + 8 * (NBRIGHT + i), mvec(gal2eq, g), m, bv, frand());
    }
    rng = keep;
}

/* constellation figures as pairs of star indices, and where to name each */
static int *seg, nseg;
static struct { v3 c; const char *name; } fig[NFIGURES];

static const char *const CON_NAME[][2] = {
    {"And", "Andromeda"}, {"Aql", "Aquila"}, {"Aqr", "Aquarius"}, {"Ara", "Ara"}, {"Ari", "Aries"},
    {"Aur", "Auriga"}, {"Boo", "Bootes"}, {"Cnc", "Cancer"}, {"CVn", "Canes Venatici"},
    {"CMa", "Canis Major"}, {"CMi", "Canis Minor"}, {"Cap", "Capricornus"}, {"Car", "Carina"},
    {"Cas", "Cassiopeia"}, {"Cen", "Centaurus"}, {"Cep", "Cepheus"}, {"Cet", "Cetus"},
    {"Col", "Columba"}, {"Com", "Coma Berenices"}, {"CrA", "Corona Australis"},
    {"CrB", "Corona Borealis"}, {"Crv", "Corvus"}, {"Crt", "Crater"}, {"Cru", "Crux"},
    {"Cyg", "Cygnus"}, {"Del", "Delphinus"}, {"Dra", "Draco"}, {"Eri", "Eridanus"},
    {"Gem", "Gemini"}, {"Gru", "Grus"}, {"Her", "Hercules"}, {"Hya", "Hydra"}, {"Hyi", "Hydrus"},
    {"Ind", "Indus"}, {"Lac", "Lacerta"}, {"Leo", "Leo"}, {"LMi", "Leo Minor"}, {"Lep", "Lepus"},
    {"Lib", "Libra"}, {"Lup", "Lupus"}, {"Lyn", "Lynx"}, {"Lyr", "Lyra"}, {"Mon", "Monoceros"},
    {"Mus", "Musca"}, {"Oph", "Ophiuchus"}, {"Ori", "Orion"}, {"Pav", "Pavo"}, {"Peg", "Pegasus"},
    {"Per", "Perseus"}, {"Phe", "Phoenix"}, {"Psc", "Pisces"}, {"PsA", "Piscis Austrinus"},
    {"Pup", "Puppis"}, {"Sge", "Sagitta"}, {"Sgr", "Sagittarius"}, {"Sco", "Scorpius"},
    {"Ser", "Serpens"}, {"Tau", "Taurus"}, {"Tri", "Triangulum"}, {"TrA", "Triangulum Australe"},
    {"Tuc", "Tucana"}, {"UMa", "Ursa Major"}, {"UMi", "Ursa Minor"}, {"Vel", "Vela"}, {"Vir", "Virgo"},
};

/* the brightest stars, by their proper names */
static const char *const STAR_NAME[][2] = {
    {"alp CMa", "Sirius"}, {"alp Car", "Canopus"}, {"alp Boo", "Arcturus"}, {"alp Lyr", "Vega"},
    {"alp Aur", "Capella"}, {"bet Ori", "Rigel"}, {"alp CMi", "Procyon"}, {"alp Eri", "Achernar"},
    {"alp Ori", "Betelgeuse"}, {"bet Cen", "Hadar"}, {"alp Aql", "Altair"}, {"alp Tau", "Aldebaran"},
    {"alp Sco", "Antares"}, {"alp Vir", "Spica"}, {"bet Gem", "Pollux"}, {"alp PsA", "Fomalhaut"},
    {"alp Cyg", "Deneb"}, {"alp Leo", "Regulus"}, {"alp Gem", "Castor"}, {"alp UMi", "Polaris"},
    {"gam Ori", "Bellatrix"}, {"alp UMa", "Dubhe"}, {"alp Per", "Mirfak"}, {"alp And", "Alpheratz"},
    {"bet Per", "Algol"}, {"alp Ari", "Hamal"}, {"eps Ori", "Alnilam"}, {"eta UMa", "Alkaid"},
};
#define NSTARNAME ((int)(sizeof STAR_NAME / sizeof *STAR_NAME))
static int star_name_idx[NSTARNAME];

static void build_figures(void) {
    seg = malloc(sizeof(int) * 2 * 1024);
    for (int i = 0; i < NSTARNAME; i++) {
        star_name_idx[i] = -1;
        for (int j = 0; j < NBRIGHT; j++)
            if (!strcmp(BRIGHT[j].name, STAR_NAME[i][0])) { star_name_idx[i] = j; break; }
    }
    for (int f = 0; f < NFIGURES; f++) {
        char con[8], tok[16];
        const char *p = FIGURES[f];
        int n = 0;
        sscanf(p, "%7s%n", con, &n);
        p += n;
        fig[f].name = NULL;
        for (int i = 0; i < (int)(sizeof CON_NAME / sizeof *CON_NAME); i++)
            if (!strcmp(CON_NAME[i][0], con)) fig[f].name = CON_NAME[i][1];
        int seen[96], nseen = 0;
        v3 sum = vec(0, 0, 0);
        int prev = -1;
        while (*p) {
            while (*p == ' ') p++;
            if (*p == ';') { prev = -1; p++; continue; }
            if (*p == '-') { p++; continue; }
            int k = 0;
            while (*p && *p != '-' && *p != ';' && *p != ' ' && k < 15) tok[k++] = *p++;
            tok[k] = 0;
            if (!k) break;
            char *dot = strchr(tok, '.');
            int s;
            if (dot) { *dot = 0; s = find_star(tok, dot + 1); }
            else s = find_star(tok, con);
            if (s < 0) fprintf(stderr, "nightsky: figure %s: no star %s\n", con, tok);
            if (s >= 0) {
                int dup = 0;
                for (int i = 0; i < nseen; i++) dup |= seen[i] == s;
                if (!dup && nseen < 96) {
                    seen[nseen++] = s;
                    sum = add(sum, vec(starv[8 * s], starv[8 * s + 1], starv[8 * s + 2]));
                }
            }
            if (s >= 0 && prev >= 0 && nseg < 1024) { seg[2 * nseg] = prev; seg[2 * nseg + 1] = s; nseg++; }
            prev = s;
        }
        fig[f].c = nseen ? norm(sum) : vec(0, 0, 1);
        if (!nseen) fig[f].name = NULL;
    }
}

/* ================= the land ================= */
#define RIDGE_N 16384
static float ridge[RIDGE_N * 4];   /* far, mid, near ridge tops; the near shore (all degrees) */

static void bake_ridge(uint32_t seed) {
    uint32_t salt = seed * 131u;
    for (int i = 0; i < RIDGE_N; i++) {
        double az = 2 * PI * i / RIDGE_N;
        float c = (float)cos(az), s = (float)sin(az);
        /* warp the bearing a little, so slopes come out lopsided */
        float wz = (fbm(7 + c * 1.7f, 3 + s * 1.7f, 3, salt + 3) - 0.5f) * 0.6f;
        float cw = (float)cos(az + wz), sw = (float)sin(az + wz);
        float mass = fbm(60 + cw * 2.2f, 40 + sw * 2.2f, 4, salt + 1);          /* broad massifs */
        float rid = ridged(20 + cw * 7.0f, 30 + sw * 7.0f, 6, salt + 2);       /* their ridgelines */
        float m0 = fminf(fmaxf((mass - 0.30f) / 0.40f, 0), 1);
        float h0 = 0.9f + 16.0f * powf(m0 * m0 * (3 - 2 * m0), 1.3f) * (0.4f + 0.6f * rid);
        float mass1 = fbm(10 + cw * 3.5f, 90 + sw * 3.5f, 4, salt + 21);
        float rid1 = ridged(70 + cw * 11.0f, 10 + sw * 11.0f, 6, salt + 23);
        float m1 = fminf(fmaxf((mass1 - 0.30f) / 0.42f, 0), 1);
        float h1 = 0.5f + 8.0f * powf(m1 * m1 * (3 - 2 * m1), 1.2f) * (0.35f + 0.65f * rid1);
        float h2 = 0.25f + 1.5f * powf(fbm(80 + c * 9, 10 + s * 9, 5, salt + 41), 1.6f);
        float sh = -(6.6f + 1.6f * fbm(15 + c * 7, 90 + s * 7, 4, salt + 51));
        ridge[4 * i] = h0; ridge[4 * i + 1] = h1; ridge[4 * i + 2] = h2; ridge[4 * i + 3] = sh;
    }
    /* conifers along the near ridge, where the forest noise allows */
    double per = 360.0 / RIDGE_N;
    uint32_t keep = rng;
    rng = seed * 2654435761u | 1;
    for (double az = 0; az < 360;) {
        double a = az * D2R;
        float c = (float)cos(a), s = (float)sin(a);
        float wood = fbm(200 + c * 6, 300 + s * 6, 4, salt + 61);
        double step = 0.09 + 0.25 * frand();
        if (wood > 0.47f) {
            int ci = (int)(az / per) % RIDGE_N;
            double base = ridge[4 * ci + 2];
            double th = (0.18 + 0.45 * frand()) * (0.6 + 1.2 * (wood - 0.47));
            double hw = th * (0.22 + 0.08 * frand());
            int span = (int)(hw / per) + 1;
            for (int k = -span; k <= span; k++) {
                int j = ((ci + k) % RIDGE_N + RIDGE_N) % RIDGE_N;
                double x = fabs(k * per) / hw;
                if (x >= 1) continue;
                /* tiers of branches: a ragged spire */
                double tier = fmod((1 - x) * 5.0 + frand() * 0.2, 1.0);
                double v = base + th * (1 - x) - th * 0.08 * tier;
                if (v > ridge[4 * j + 2]) ridge[4 * j + 2] = (float)v;
            }
            step *= 0.6;
        }
        az += step;
    }
    /* reeds in clumps along the near shore */
    for (double az = 0; az < 360;) {
        double a = az * D2R;
        float c = (float)cos(a), s = (float)sin(a);
        float clump = fbm(400 + c * 14, 100 + s * 14, 3, salt + 71);
        if (clump > 0.58f) {
            int ci = (int)(az / per) % RIDGE_N;
            double tall = (0.3 + 1.1 * frand()) * (clump - 0.5) * 3;
            int w = 1 + (frand() < 0.3);
            for (int k = 0; k < w; k++) {
                int j = (ci + k) % RIDGE_N;
                float v = ridge[4 * j + 3] + (float)tall;
                if (v > ridge[4 * j + 3]) ridge[4 * j + 3] = v;
            }
            az += 0.04 + 0.12 * frand();
        } else az += 0.3;
    }
    rng = keep;
}

/* ================= GL ================= */
#define GLSL(...) #__VA_ARGS__
static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* one-off bakes: 0 = noise (RGBA, tiling), 1 = the Milky Way (galactic l, b) */
static const char *FS_BAKE = "#version 330 core\n" GLSL(
    uniform int uMode;
    uniform vec2 uSize;
    uniform int uSalt;
    out vec4 o;
    const float PI = 3.14159265;
    uint hashu(uvec3 p) {
        uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ p.z * 0xcb1ab31fu;
        h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;
        return h;
    }
    float h01(ivec3 p) { return float(hashu(uvec3(p)) & 0xffffffu) / 16777216.0; }
    float vn2(vec2 x, int per, int salt) {
        vec2 i = floor(x), f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        ivec2 a = ivec2(mod(i, float(per))), b = ivec2(mod(i + 1.0, float(per)));
        float v00 = h01(ivec3(a.x, a.y, salt)), v10 = h01(ivec3(b.x, a.y, salt));
        float v01 = h01(ivec3(a.x, b.y, salt)), v11 = h01(ivec3(b.x, b.y, salt));
        return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
    }
    float fbm2(vec2 uv, int per, int oct, int salt) {
        float s = 0.0, amp = 0.5, nrm = 0.0;
        for (int k = 0; k < oct; k++) {
            s += vn2(uv * float(per), per, salt + k) * amp;
            nrm += amp; amp *= 0.5; per *= 2;
        }
        return s / nrm;
    }
    float vn3(vec3 x, int salt) {
        vec3 i = floor(x), f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        ivec3 c = ivec3(i);
        float a = mix(h01(c + ivec3(0, 0, salt * 7919)), h01(c + ivec3(1, 0, salt * 7919)), f.x);
        float b = mix(h01(c + ivec3(0, 1, salt * 7919)), h01(c + ivec3(1, 1, salt * 7919)), f.x);
        float d = mix(h01(c + ivec3(0, 0, 1 + salt * 7919)), h01(c + ivec3(1, 0, 1 + salt * 7919)), f.x);
        float e = mix(h01(c + ivec3(0, 1, 1 + salt * 7919)), h01(c + ivec3(1, 1, 1 + salt * 7919)), f.x);
        return mix(mix(a, b, f.y), mix(d, e, f.y), f.z);
    }
    float fbm3(vec3 p, int oct, int salt) {
        float s = 0.0, amp = 0.5, nrm = 0.0;
        for (int k = 0; k < oct; k++) {
            s += vn3(p, salt + k) * amp;
            nrm += amp; amp *= 0.5; p *= 2.03;
        }
        return s / nrm;
    }
    float blob(vec3 g, float l, float b, float r) {
        vec3 c = vec3(cos(radians(b)) * cos(radians(l)), cos(radians(b)) * sin(radians(l)), sin(radians(b)));
        float a = degrees(acos(clamp(dot(g, c), -1.0, 1.0)));
        return exp(-a * a / (2.0 * r * r));
    }
    void main() {
        vec2 uv = gl_FragCoord.xy / uSize;
        if (uMode == 0) {
            o = vec4(fbm2(uv, 4, 6, uSalt), fbm2(uv, 8, 5, uSalt + 10),
                     fbm2(uv, 16, 4, uSalt + 20), fbm2(uv, 16, 4, uSalt + 30));
            return;
        }
        float l = (uv.x - 0.5) * 2.0 * PI, b = (uv.y - 0.5) * radians(96.0);
        vec3 g = vec3(cos(b) * cos(l), cos(b) * sin(l), sin(b));
        float ld = degrees(l), bd = degrees(b);
        float wn = fbm3(g * 3.0 + 7.0, 3, 4);
        float wid = (2.8 + 5.5 * exp(-pow(ld / 40.0, 2.0))) * (0.7 + 0.7 * wn);
        float along = 0.22 + 0.55 * exp(-pow(ld / 50.0, 2.0)) + 0.35 * exp(-pow((ld - 75.0) / 18.0, 2.0))
                    + 0.30 * exp(-pow((ld + 72.0) / 24.0, 2.0)) + 0.10 * exp(-pow((ld - 135.0) / 30.0, 2.0));
        float bc = 1.5 * sin(radians(ld * 2.3)) * (fbm3(g * 2.0, 2, 5) - 0.3);
        float disk = along * (exp(-pow((bd - bc) / wid, 2.0)) + 0.30 * exp(-abs(bd - bc) / (wid * 2.4)));
        float bulge = 0.75 * exp(-(ld * ld + bd * bd * 1.6) / (2.0 * 9.0 * 9.0));
        float clump = fbm3(g * 7.0, 5, 1), mid = fbm3(g * 18.0, 4, 6), fine = fbm3(g * 70.0, 3, 2);
        float cl = clump * clump * 2.2;
        float st = (disk * (0.15 + cl) + bulge * (0.5 + cl)) * (0.55 + 0.9 * mid) * (0.75 + 0.5 * fine);
        /* the bright star clouds: Sagittarius, Scutum, Cygnus, Carina */
        st += (0.45 * blob(g, 2.0, -4.5, 3.0) + 0.30 * blob(g, 27.0, -2.0, 2.0) + 0.25 * blob(g, 76.0, 1.0, 3.5)
             + 0.22 * blob(g, 287.0, -0.8, 3.0)) * (0.5 + mid) * (0.7 + 0.6 * fine);
        /* dust: the Great Rift from Cygnus to Ophiuchus, the Pipe, filaments, the Coalsack */
        float dn = fbm3(g * 9.0 + 3.1, 5, 3), dn2 = fbm3(g * 25.0 + 1.7, 3, 7);
        float rc = 2.5 + 2.0 * sin(radians(ld * 3.0)) + 3.0 * (dn - 0.5);
        float rw = 1.6 + 1.3 * dn;
        float rift = exp(-pow((bd - rc) / rw, 2.0)) * smoothstep(-25.0, -5.0, ld) * (1.0 - smoothstep(60.0, 92.0, ld));
        float dust = clamp(rift * (0.3 + 1.3 * dn2), 0.0, 1.0) * 0.9;
        dust = max(dust, smoothstep(0.55, 0.72, dn) * exp(-pow(bd / (wid * 1.1), 2.0)) * (0.4 + 0.6 * dn2));
        dust = max(dust, 0.8 * blob(g, 0.5, 5.0, 2.5) * smoothstep(0.35, 0.6, dn2));
        dust = max(dust, 0.6 * blob(g, 353.0, 16.0, 4.0) * smoothstep(0.4, 0.65, dn2));
        dust = max(dust, 0.8 * blob(g, 301.0, -1.0, 2.2));
        float v = st * (1.0 - dust);
        /* the Magellanic Clouds, Andromeda, the Orion Nebula */
        v += 0.55 * blob(g, 280.5, -32.9, 2.6) * (0.6 + 0.8 * fine) + 0.30 * blob(g, 302.8, -44.3, 1.5);
        v += 0.30 * blob(g, 121.2, -21.6, 0.8) + 0.20 * blob(g, 209.0, -19.4, 0.5);
        float warm = clamp(bulge * 1.4 + dust * 0.7 + 0.25 * blob(g, 2.0, -4.5, 6.0), 0.0, 1.0);
        o = vec4(clamp(v * 0.8, 0.0, 1.0), warm, 0.0, 1.0);
    });

/* shared by the sky, stars, lines and meteors */
static const char *COMMON = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform vec3 uCF, uCR, uCU;     /* the camera, local east-north-up */
    uniform vec2 uTan;
    uniform int uCyl;               /* spanning: a cylindrical panorama instead of the pinhole */
    uniform vec4 uSpan;             /* uCyl: this window's bearing (from uAz0) at left, right, tan(alt) at bottom, top */
    uniform float uAz0;
    uniform float uShoreK;
    uniform sampler2D uRidge;       /* by azimuth: far, mid, near ridge tops; near shore (degrees) */
    uniform sampler2D uNoise;
    uniform float uCloudCov, uTime, uPxRad;
    uniform vec2 uWind;
    const float PI = 3.14159265;
    vec3 dirAt(vec2 frag) {
        vec2 n = frag / uRes;
        if (uCyl == 1) {
            float az = uAz0 + mix(uSpan.x, uSpan.y, n.x);
            return normalize(vec3(sin(az), cos(az), mix(uSpan.z, uSpan.w, n.y)));
        }
        n = n * 2.0 - 1.0;
        return normalize(uCF + n.x * uTan.x * uCR + n.y * uTan.y * uCU);
    }
    /* the other way: where direction d lands, 0..1 across the window; false
     * if it's behind (nearer the eye than zmin, for the pinhole) */
    bool toUV(vec3 d, float zmin, out vec2 uv) {
        if (uCyl == 1) {
            float u = mod(atan(d.x, d.y) - uAz0 + PI, 2.0 * PI) - PI, h = length(d.xy);
            uv = vec2((u - uSpan.x) / (uSpan.y - uSpan.x), (d.z / max(h, 1e-6) - uSpan.z) / (uSpan.w - uSpan.z));
            return h > 1e-4;
        }
        vec3 c = vec3(dot(d, uCR), dot(d, uCU), dot(d, uCF));
        uv = c.xy / (max(c.z, 1e-3) * uTan) * 0.5 + 0.5;
        return c.z > zmin;
    }
    vec4 ridgeAt(vec3 d) {
        vec4 r = textureLod(uRidge, vec2(atan(d.x, d.y) / (2.0 * PI), 0.5), 0.0);
        r.a *= uShoreK;
        return r;
    }
    float altDeg(vec3 d) { return degrees(asin(clamp(d.z, -1.0, 1.0))); }
    float airmass(float h) {
        h = max(h, 0.0);
        return 1.0 / (sin(radians(h)) + 0.50572 * pow(h + 6.07995, -1.6364));
    }
    vec3 extinction(float h) { return pow(vec3(10.0), -0.4 * vec3(0.12, 0.19, 0.32) * (airmass(h) - 1.0)); }
    /* thin cloud on a flat layer overhead, drifting with the wind */
    float cloudAt(vec3 d) {
        if (uCloudCov <= 0.0 || d.z <= 0.0) return 0.0;
        float k = 1.0 / (d.z + 0.035);
        vec2 p = d.xy * k * 0.16 + uWind;
        float lod = log2(max(uPxRad * k * k * 0.16 * 512.0, 1e-4));
        float n = textureLod(uNoise, p * vec2(0.55, 1.0), lod).r * 0.62
                + textureLod(uNoise, p * vec2(1.4, 2.6) + 0.37, lod + 1.0).g * 0.38;
        float th = 0.66 - 0.34 * uCloudCov;
        float c = smoothstep(th, th + 0.13, n);
        return c * smoothstep(0.0, 0.12, d.z) * 0.92;
    }
    );

static const char *FS_SKY = GLSL(
    uniform mat3 uHor2Gal;
    uniform sampler2D uMW;
    uniform vec3 uSun, uMoon, uMoonW, uMoonA, uMoonSun;
    uniform float uMoonR, uSunR, uMWAmt, uTownAz, uDither, uMoonRpx;
    uniform vec2 uMoonPx, uMoonEW, uMoonEA;
    uniform vec3 uZen, uHorS, uHorA, uSunGlow, uMoonGlow, uMoonCol, uEarthshine, uSunDisc;
    uniform vec3 uCloudAmb, uSunLit, uLand, uRimMoon, uRimSun, uTown, uWaterBody, uMWCol, uMWWarm;
    uniform int uPass;              /* 0: the sky alone, at half size; 1: everything, from it */
    uniform float uExpo;            /* overall exposure */
    uniform sampler2D uBg;          /* pass 0's result */
    out vec4 o;

    float hash12(vec2 p) {
        vec3 p3 = fract(vec3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    /* the air: night glow, twilight, day, and the light round Sun and Moon */
    vec3 atmos(vec3 d) {
        float h = max(d.z, 0.0);
        vec2 dh = normalize(d.xy + vec2(1e-5, 0.0));
        vec2 sh = normalize(uSun.xy + vec2(1e-5, 0.0));
        float tow = dot(dh, sh) * 0.5 + 0.5;
        vec3 c = mix(uZen, uHorA, exp(-h * 3.4));
        c += (uHorS - uHorA) * exp(-h * 4.2) * tow * tow * tow;
        float cs = dot(d, uSun);
        c += uSunGlow * (0.6 * exp((cs - 1.0) * 4000.0) + 0.3 * exp((cs - 1.0) * 150.0) + 0.2 * exp((cs - 1.0) * 12.0) + 0.1 * exp((cs - 1.0) * 2.0));
        float cm = dot(d, uMoon);
        c += uMoonGlow * (0.45 * exp((cm - 1.0) * 2500.0) + 0.22 * exp((cm - 1.0) * 150.0) + 0.10 * exp((cm - 1.0) * 12.0));
        float town = max(dot(dh, vec2(sin(uTownAz), cos(uTownAz))), 0.0);
        town *= town; town *= town;
        c += uTown * exp(-h * 9.0) * town;
        return c;
    }

    /* the seas, roughly where they are: lat, lon, radius (degrees), depth */
    const vec4 MARIA[20] = vec4[](
        vec4(25.0, -60.0, 17.0, 0.8), vec4(5.0, -56.0, 14.0, 0.8), vec4(-6.0, -45.0, 10.0, 0.7),   /* Procellarum */
        vec4(42.0, -45.0, 9.0, 0.6), vec4(34.0, -16.0, 15.0, 1.0),                                 /* Imbrium */
        vec4(56.0, -22.0, 6.0, 0.6), vec4(57.0, 0.0, 6.0, 0.6), vec4(57.0, 22.0, 6.0, 0.55),       /* Frigoris */
        vec4(28.0, 17.5, 10.0, 1.0), vec4(8.0, 30.0, 11.0, 1.0), vec4(15.0, 24.0, 6.0, 0.9),       /* Serenitatis, Tranquillitatis */
        vec4(17.0, 59.0, 8.0, 1.0), vec4(-6.0, 51.0, 10.0, 0.9), vec4(-15.0, 35.0, 6.0, 0.85),     /* Crisium, Fecunditatis, Nectaris */
        vec4(13.0, 4.0, 5.0, 0.8), vec4(7.0, -30.0, 7.0, 0.6), vec4(-10.0, -23.0, 6.0, 0.7),       /* Vaporum, Insularum, Cognitum */
        vec4(-21.0, -17.0, 10.0, 0.75), vec4(-24.0, -39.0, 6.0, 0.85), vec4(-2.0, -12.0, 4.0, 0.6)); /* Nubium, Humorum, Medii */
    float moonAlbedo(vec3 p, vec2 xy) {
        float n = textureLod(uNoise, xy * 0.3 + vec2(0.3, 0.6), 0.0).g;
        float n2 = textureLod(uNoise, xy * 1.1 + vec2(0.1, 0.2), 0.0).r;
        float warp = (n - 0.5) * 14.0 + (n2 - 0.5) * 6.0;      /* ragged shores, in degrees */
        float dark = 0.0;
        for (int i = 0; i < 20; i++) {
            vec4 m = MARIA[i];
            float la = radians(m.x), lo = radians(m.y);
            vec3 q = vec3(cos(la) * sin(lo), sin(la), cos(la) * cos(lo));
            float a = degrees(acos(clamp(dot(p, q), -1.0, 1.0))) + warp;
            dark = max(dark, m.w * (1.0 - smoothstep(m.z * 0.6, m.z * 1.35, a)));
        }
        vec3 ty = vec3(cos(radians(-43.0)) * sin(radians(-11.0)), sin(radians(-43.0)), cos(radians(-43.0)) * cos(radians(-11.0)));
        float tycho = exp(-pow(degrees(acos(clamp(dot(p, ty), -1.0, 1.0))) / 2.5, 2.0));
        return (1.0 - 0.36 * dark) * (0.93 + 0.14 * n2) + 0.2 * tycho;
    }
    /* the Moon's disc at (x, y) across it (x towards sky west, y celestial
     * north, radius 1), px = a pixel in those units: colour, coverage */
    vec4 moonXY(float x, float y, float px) {
        float r = length(vec2(x, y));
        float a = 1.0 - smoothstep(1.0 - px, 1.0 + px, r);
        if (a <= 0.0) return vec4(0.0);
        float z = sqrt(max(1.0 - r * r, 0.0));
        vec3 n = x * uMoonW + y * uMoonA - z * uMoon;
        float mu0 = dot(n, uMoonSun);
        float lit = max(mu0, 0.0);
        lit = mix(lit, 2.0 * lit / (lit + z + 0.06), 0.65) * smoothstep(-0.03, 0.05, mu0);
        vec3 p = vec3(x, y, z);         /* x: towards Mare Crisium, y: lunar north */
        float alb = moonAlbedo(p, vec2(x, y));
        vec3 col = uMoonCol * alb * lit + uEarthshine * alb * (1.0 - smoothstep(-0.05, 0.1, mu0));
        return vec4(col, a);
    }
    /* seen in direction d (in the lake) */
    vec4 moonDisc(vec3 d) {
        float c = dot(d, uMoon);
        if (c < cos(uMoonR * 1.3)) return vec4(0.0);
        vec3 q = d - uMoon * c;
        float sr = sin(uMoonR);
        return moonXY(dot(q, uMoonW) / sr, dot(q, uMoonA) / sr, uPxRad / sr);
    }
    /* seen directly: a true circle on the screen, however wide the lens */
    vec4 moonScreen() {
        vec2 off = gl_FragCoord.xy - uMoonPx;
        if (uMoonRpx <= 0.0 || dot(off, off) > uMoonRpx * uMoonRpx * 1.7) return vec4(0.0);
        return moonXY(dot(off, uMoonEW) / uMoonRpx, dot(off, uMoonEA) / uMoonRpx, 1.0 / uMoonRpx);
    }

    vec3 cloudCol(vec3 d, float cl) {
        float fm = max(dot(d, uMoon), 0.0), fs = max(dot(d, uSun), 0.0);
        vec3 c = uCloudAmb * (1.0 - 0.3 * cl);
        c += uMoonGlow * (0.5 + 2.0 * pow(fm, 30.0) + 5.0 * pow(fm, 800.0));
        c += uSunLit * (0.6 + 1.4 * pow(fs, 6.0));
        return c;
    }

    /* the sky in direction d: air, Milky Way, Sun, clouds (not the Moon's disc or the land) */
    vec3 skyBg(vec3 d) {
        vec3 c = atmos(d);
        float h = altDeg(d);
        vec3 ext = extinction(h);
        vec3 bg = vec3(0.0);
        if (uMWAmt > 0.0) {
            vec3 g = uHor2Gal * d;
            float b = asin(clamp(g.z, -1.0, 1.0));
            if (abs(b) < radians(47.5)) {
                vec2 uv = vec2(atan(g.y, g.x) / (2.0 * PI) + 0.5, b / radians(96.0) + 0.5);
                vec2 mw = textureLod(uMW, uv, 0.0).rg;
                bg = mix(uMWCol, uMWWarm, mw.g) * mw.r * uMWAmt * ext;
            }
        }
        if (dot(d, uSun) > cos(uSunR)) bg += uSunDisc * pow(ext, vec3(0.5));
        c += bg;
        float cl = cloudAt(d);
        if (cl > 0.0) c = mix(c, cloudCol(d, cl), cl);
        return c;
    }

    /* one ridge: dark land fading into the haze with distance, rim-lit by
     * whatever light is behind it */
    vec3 land(float top, float haze, float a, vec3 d, vec3 hz) {
        float az = atan(d.x, d.y);
        float gully = textureLod(uNoise, vec2(az * 9.0, a * 0.05), 0.0).b * 0.6
                    + textureLod(uNoise, vec2(az * 31.0, a * 0.2), 0.0).a * 0.4;
        vec3 base = uLand * (0.65 + 0.7 * gully) * (1.0 - 0.25 * haze);
        float mist = clamp(haze * (0.75 + 0.6 * exp(-max(a, 0.0) * 1.1)), 0.0, 0.92);
        vec3 c = mix(base, hz, mist);
        vec2 dh = normalize(d.xy + vec2(1e-5, 0.0));
        float rm = max(dot(dh, normalize(uMoon.xy + vec2(1e-5, 0.0))), 0.0);
        float rs = max(dot(dh, normalize(uSun.xy + vec2(1e-5, 0.0))), 0.0);
        rm *= rm; rm *= rm * rm; rm *= rm;     /* ^12 */
        rs *= rs; rs *= rs * rs;               /* ^6 */
        float e = max(top - a, 0.0);
        c += (uRimMoon * rm + uRimSun * rs) * exp(-e / 0.12) * (1.0 - 0.6 * haze);
        return c;
    }

    /* the sky from pass 0 where it's on screen (always, but at the edges) */
    vec3 bgAt(vec3 d) {
        vec2 uv;
        if (toUV(d, 0.05, uv) && all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0))))
            return textureLod(uBg, uv, 0.0).rgb;
        return skyBg(d);
    }
    vec3 withMoon(vec3 c, vec3 d, bool direct) {
        vec4 m = direct ? moonScreen() : moonDisc(d);
        if (m.a > 0.0) c += m.rgb * pow(extinction(altDeg(d)), vec3(0.3)) * m.a * (1.0 - cloudAt(d));
        return c;
    }

    vec3 scene(vec3 d, float pxdeg, bool direct) {
        vec4 R = ridgeAt(d);
        float a = altDeg(d);
        float aF = clamp((R.r - a) / pxdeg + 0.5, 0.0, 1.0);
        float aM = clamp((R.g - a) / pxdeg + 0.5, 0.0, 1.0);
        float aN = clamp((R.b - a) / pxdeg + 0.5, 0.0, 1.0);
        if (aN >= 1.0) return land(R.b, 0.06, a, d, vec3(0.0)) ;
        vec3 hz = bgAt(normalize(vec3(d.xy, 0.03)));
        vec3 c = vec3(0.0);
        if (max(aF, aM) < 1.0) c = withMoon(bgAt(d), d, direct);
        if (aF > 0.0) c = mix(c, land(R.r, 0.55, a, d, hz), aF);
        if (aM > 0.0) c = mix(c, land(R.g, 0.30, a, d, hz), aM);
        if (aN > 0.0) c = mix(c, land(R.b, 0.06, a, d, hz), aN);
        return c;
    }

    vec3 water(vec3 d) {
        float D = 1.0 / max(-d.z, 1e-4);             /* eye one unit above the lake */
        vec2 P = d.xy * D;
        float lod = log2(max(D * D * uPxRad * 0.8 * 512.0, 1e-4));
        vec2 g = textureLod(uNoise, P * vec2(0.35, 0.8) + vec2(uTime * 0.010, uTime * 0.021), lod).ba
               + 0.7 * textureLod(uNoise, P * vec2(0.9, 2.1) + vec2(-uTime * 0.017, uTime * 0.033), lod + 1.4).ba - 0.85;
        float calm = 1.0 / (1.0 + D * 0.02);
        vec3 n = normalize(vec3(g * vec2(0.06, 0.20) * calm, 1.0));
        vec3 r = reflect(d, n);
        r.z = max(r.z, 0.0004);
        r = normalize(r);
        float ci = clamp(dot(-d, n), 0.0, 1.0);
        float F = 0.02 + 0.98 * pow(1.0 - ci, 5.0);
        F = 0.3 + 0.62 * F;
        return scene(r, degrees(uPxRad) * 1.5, false) * F + uWaterBody * (1.0 - F);
    }

    void main() {
        vec3 d = dirAt(gl_FragCoord.xy);
        if (uPass == 0) {
            o = vec4(d.z > -0.02 ? skyBg(d) : atmos(vec3(d.xy, 0.0)), 1.0);
            return;
        }
        float pxdeg = degrees(uPxRad);
        vec3 c;
        if (d.z >= 0.0) c = scene(d, pxdeg, true);
        else {
            float a = altDeg(d);
            float sh = ridgeAt(d).a;
            float aS = clamp((sh - a) / pxdeg + 0.5, 0.0, 1.0);
            c = aS < 1.0 ? water(d) : vec3(0.0);
            vec3 bank = uLand * 0.55;
            c = mix(c, bank, aS);
        }
        c = c * uExpo + (hash12(gl_FragCoord.xy + uDither) - 0.5) / 255.0;
        o = vec4(c, 1.0);
    });

static const char *VS_STAR = GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aCol;
    layout(location = 2) in vec2 aMag;      /* V, twinkle seed (-1: planet) */
    uniform mat3 uEq2Hor;
    uniform float uLim, uGain, uPx, uMirror, uMoonR, uMoonWash, uMaxI;
    uniform vec3 uMoon;
    out vec3 vCol;
    out float vSig, vHalf, vHalo;
    void cull() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vCol = vec3(0.0); vSig = 1.0; vHalf = 1.0; vHalo = 0.0; }
    void main() {
        vec3 d = uEq2Hor * aPos;
        float alt = altDeg(d);
        if (alt < -1.0) { cull(); return; }
        alt += 1.02 / tan(radians(alt + 10.3 / (alt + 5.11))) / 60.0;
        vec2 hz = normalize(d.xy + vec2(1e-7, 0.0)) * cos(radians(alt));
        d = vec3(hz, sin(radians(alt)));
        vec4 R = ridgeAt(d);
        if (alt < max(max(R.r, R.g), R.b)) { cull(); return; }
        if (uMirror > 0.5 && -alt < R.a + 0.15) { cull(); return; }
        if (dot(d, uMoon) > cos(uMoonR)) { cull(); return; }
        float m = aMag.x + 0.2 * (airmass(alt) - 1.0);
        float vis = 1.0 - smoothstep(uLim - 0.8, uLim + 0.4, m);
        vis *= 1.0 - uMoonWash * exp((dot(d, uMoon) - 1.0) * 30.0);
        vis *= 1.0 - cloudAt(d);
        if (vis <= 0.004) { cull(); return; }

        float planet = aMag.y < 0.0 ? 1.0 : 0.0;
        float seed = max(aMag.y, 0.0);
        float X = airmass(alt);
        float sc = clamp((X - 1.0) / 5.0, 0.0, 1.0);
        float amp = (0.10 + 0.55 * sc) * (1.0 - 0.8 * planet);
        float t = uTime + seed * 100.0;
        float tw = 0.6 * sin(t * (1.7 + seed * 2.3) + seed * 40.0) + 0.4 * sin(t * (4.1 + fract(seed * 7.0) * 3.0));
        float f = pow(10.0, -0.4 * (m - 1.0));
        float I = uGain * min(pow(f, 0.5), uMaxI) * vis * (1.0 + amp * tw);
        vec3 col = mix(vec3(1.0), aCol, 0.85) * extinction(alt);
        col *= 1.0 + vec3(0.25, 0.0, -0.25) * amp * sin(t * 3.3 + seed * 9.0) * sc;
        vCol = col * I;
        vSig = uPx * (0.78 + 0.42 * pow(f, 0.25));
        vHalo = 0.05 * (1.0 - smoothstep(-1.0, 2.6, m)) * vis * min(I, 6.0);
        vHalf = vSig * 3.2 + (vHalo > 0.0 ? 18.0 * uPx : 0.0);
        if (uMirror > 0.5) {
            vCol *= 0.5 * (0.8 + 0.2 * sin(uTime * 2.3 + seed * 60.0));
            vHalf *= 2.0;
            d.z = -d.z;
        }
        vec2 uv;
        if (!toUV(d, 0.02, uv)) { cull(); return; }
        gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
        gl_PointSize = min(2.0 * vHalf, 96.0);
        vHalf = gl_PointSize * 0.5;
    });

static const char *FS_STAR = GLSL(
    in vec3 vCol;
    in float vSig, vHalf, vHalo;
    uniform float uMirror;
    out vec4 o;
    void main() {
        vec2 p = (gl_PointCoord * 2.0 - 1.0) * vHalf;
        if (uMirror > 0.5) p.y *= 0.4;
        float r2 = dot(p, p);
        float core = exp(-r2 / (2.0 * vSig * vSig));
        float halo = vHalo * exp(-sqrt(r2) / (vSig * 3.0));
        float mx = max(max(vCol.r, vCol.g), max(vCol.b, 1e-4));
        o = vec4(vCol * core + vCol / mx * halo, 1.0);
    });

/* screen-space strips: constellation lines and meteors */
static const char *VS_LINE = GLSL(
    layout(location = 0) in vec4 aV;        /* x, y (NDC), across (-1..1), alpha */
    out float vAcross, vAlpha;
    void main() { vAcross = aV.z; vAlpha = aV.w; gl_Position = vec4(aV.xy, 0.0, 1.0); });

static const char *FS_LINE = GLSL(
    in float vAcross, vAlpha;
    uniform vec3 uColor;
    out vec4 o;
    void main() {
        vec3 d = dirAt(gl_FragCoord.xy);
        vec4 R = ridgeAt(d);
        if (d.z <= 0.0 || altDeg(d) < max(max(R.r, R.g), R.b) + 0.03) discard;
        float w = 1.0 - smoothstep(0.3, 1.0, abs(vAcross));
        o = vec4(uColor * vAlpha * w * (1.0 - cloudAt(d)), 1.0);
    });

/* labels: glyphs from the atlas, hidden by the land and dimmed by cloud like the lines */
static const char *VS_TEXT = GLSL(
    layout(location = 0) in vec4 aP;        /* x, y (NDC), u, v (atlas) */
    layout(location = 1) in vec3 aC;        /* colour, alpha already in */
    out vec2 vUV;
    out vec3 vC;
    void main() { vUV = aP.zw; vC = aC; gl_Position = vec4(aP.xy, 0.0, 1.0); });

static const char *FS_TEXT = GLSL(
    in vec2 vUV;
    in vec3 vC;
    uniform sampler2D uAtlas;
    out vec4 o;
    void main() {
        vec3 d = dirAt(gl_FragCoord.xy);
        vec4 R = ridgeAt(d);
        if (d.z <= 0.0 || altDeg(d) < max(max(R.r, R.g), R.b) + 0.03) discard;
        float a = texture(uAtlas, vUV).r;
        o = vec4(vC * a * (1.0 - 0.8 * cloudAt(d)), 1.0);
    });

static GLuint prog_sky, prog_star, prog_line, prog_text, prog_bake, vao_full;
static GLuint tex_atlas, vao_text, vbo_text;
static GLuint tex_ridge, tex_noise, tex_mw, tex_bg, fbo_bg;
static int bg_w, bg_h;
static GLuint vao_star, vbo_star, vao_planet, vbo_planet, vao_strip, vbo_strip;

static GLuint compile(GLenum type, const char *const *src, int n) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, n, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "nightsky: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}
static GLuint mk_prog(const char *vs, const char *fs, int common) {
    const char *v[2] = {COMMON, vs}, *f[2] = {COMMON, fs};
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, common ? v : v + 1, common ? 2 : 1));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, common ? f : f + 1, common ? 2 : 1));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "nightsky: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

#define U(p, n) glGetUniformLocation(p, n)
static void u1f(GLuint p, const char *n, double v) { glUniform1f(U(p, n), (float)v); }
static void u2f(GLuint p, const char *n, double a, double b) { glUniform2f(U(p, n), (float)a, (float)b); }
static void u3v(GLuint p, const char *n, v3 v) { glUniform3f(U(p, n), (float)v.x, (float)v.y, (float)v.z); }
static void u3c(GLuint p, const char *n, const float *c) { glUniform3fv(U(p, n), 1, c); }
static void umat(GLuint p, const char *n, m3 m) {
    float f[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) f[i * 3 + j] = (float)m.m[i][j];
    glUniformMatrix3fv(U(p, n), 1, GL_TRUE, f);
}

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum wrap_s, GLenum wrap_t) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
    return t;
}

/* render a bake into a fresh texture */
static GLuint bake(int mode, GLenum ifmt, int w, int h, GLenum wrap_t, int salt) {
    GLuint t = mk_tex(ifmt, w, h, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_REPEAT, wrap_t);
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    glViewport(0, 0, w, h);
    glUseProgram(prog_bake);
    glUniform1i(U(prog_bake, "uMode"), mode);
    glUniform1i(U(prog_bake, "uSalt"), salt);
    u2f(prog_bake, "uSize", w, h);
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &f);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    return t;
}

static void vertex_layout(void) {
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, (void *)12);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, (void *)24);
}

static void gl_init(void) {
    prog_bake = mk_prog(VS_FULL, FS_BAKE, 0);
    const char *full_body = VS_FULL + strlen("#version 330 core\n");
    prog_sky = mk_prog(full_body, FS_SKY, 1);
    prog_star = mk_prog(VS_STAR, FS_STAR, 1);
    prog_line = mk_prog(VS_LINE, FS_LINE, 1);
    prog_text = mk_prog(VS_TEXT, FS_TEXT, 1);
    glGenVertexArrays(1, &vao_full);

    tex_ridge = mk_tex(GL_RGBA32F, RIDGE_N, 1, GL_RGBA, GL_FLOAT, ridge, GL_REPEAT, GL_CLAMP_TO_EDGE);
    tex_noise = bake(0, GL_RGBA8, 512, 512, GL_REPEAT, (int)(opt_seed % 1000));
    tex_mw = bake(1, GL_RG8, 4096, 1024, GL_CLAMP_TO_EDGE, 0);

    glGenVertexArrays(1, &vao_star);
    glBindVertexArray(vao_star);
    glGenBuffers(1, &vbo_star);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_star);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * nstars, starv, GL_STATIC_DRAW);
    vertex_layout();
    glGenVertexArrays(1, &vao_planet);
    glBindVertexArray(vao_planet);
    glGenBuffers(1, &vbo_planet);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_planet);
    glBufferData(GL_ARRAY_BUFFER, sizeof sky.planet, NULL, GL_DYNAMIC_DRAW);
    vertex_layout();
    glGenVertexArrays(1, &vao_strip);
    glBindVertexArray(vao_strip);
    glGenBuffers(1, &vbo_strip);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *)0);
    glGenVertexArrays(1, &vao_text);
    glBindVertexArray(vao_text);
    glGenBuffers(1, &vbo_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 28, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 28, (void *)16);
    glBindVertexArray(0);

    GLuint progs[4] = {prog_sky, prog_star, prog_line, prog_text};
    for (int i = 0; i < 4; i++) {
        glUseProgram(progs[i]);
        glUniform1i(U(progs[i], "uRidge"), 0);
        glUniform1i(U(progs[i], "uNoise"), 1);
        glUniform1i(U(progs[i], "uMW"), 2);
        glUniform1i(U(progs[i], "uBg"), 3);
        glUniform1i(U(progs[i], "uAtlas"), 4);
    }
    glEnable(GL_PROGRAM_POINT_SIZE);
}

/* ================= the view ================= */
static struct { v3 f, r, u; double tx, ty; } cam;
static double view_az, hfov_deg;
/* spanning: the panorama. This window's bearings left and right of view_az
 * and tan(altitude) at its bottom and top; the whole wall's bearings; the
 * front output's sides and top (radians, tan); and logical px -> window px */
static double cyl_u0, cyl_u1, cyl_v0, cyl_v1, wall_u0, wall_u1, front_u, front_top, span_sc = 1, front_px = 1;
static double shore_k = 1;         /* the near shore's depth below the horizon, x this */

/* the output under the middle of the layout */
static int front_output(void) {
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
    }
    float mx = (x0 + x1) / 2;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) return i;
    return 0;
}

static void camera_span(void) {
    double x0 = 1e9, x1 = -1e9;
    for (int i = 0; i < span_n; i++) {
        x0 = fmin(x0, span[i][0]); x1 = fmax(x1, span[i][0] + span[i][2]);
    }
    const float *f = span[span_front], *me = span[span_me];
    double cx = f[0] + f[2] / 2.0, cy = f[1] + f[3] / 2.0;
    double fov = opt_fov_set ? opt_fov : 280;
    double fl = fmax(cx - x0, x1 - cx) / (fov / 2 * D2R);   /* logical px per radian */
    if ((x1 - x0) / fl > 1.9 * PI) fl = (x1 - x0) / (1.9 * PI);
    /* the front output's centre looks out at tan(alt) = tp */
    double tp = opt_pitch_set ? tan(opt_pitch * D2R) : 0.30 * f[3] / fl;
    cyl_u0 = (me[0] - cx) / fl; cyl_u1 = (me[0] + me[2] - cx) / fl;
    cyl_v0 = tp - (me[1] + me[3] - cy) / fl; cyl_v1 = tp - (me[1] - cy) / fl;
    wall_u0 = (x0 - cx) / fl; wall_u1 = (x1 - cx) / fl;
    front_u = f[2] / 2.0 / fl; front_top = tp + f[3] / 2.0 / fl;
    span_sc = W / me[2];
    front_px = fmax(1.0, fmin(f[2] / 1920.0, f[3] / 1080.0));
    hfov_deg = (wall_u1 - wall_u0) * R2D;
    /* a tall output reaching well below the others would be mostly near
     * bank: push the shore down so the lake comes to near its foot */
    double low = 0;
    for (int i = 0; i < span_n; i++) low = fmax(low, -(tp - (span[i][1] + span[i][3] - cy) / fl));
    shore_k = fmax(1.0, 0.92 * atan(low) * R2D / 7.4);
    /* the pinhole's vectors, for anything that still asks: facing the front */
    double A = view_az * D2R;
    cam.f = vec(sin(A), cos(A), 0); cam.r = vec(cos(A), -sin(A), 0); cam.u = vec(0, 0, 1);
    cam.tx = cam.ty = 1;
}

static void camera(void) {
    if (span_me >= 0) { camera_span(); return; }
    double A = view_az * D2R, p = opt_pitch * D2R;
    cam.f = vec(sin(A) * cos(p), cos(A) * cos(p), sin(p));
    cam.r = vec(cos(A), -sin(A), 0);
    cam.u = cross(cam.r, cam.f);
    double t = tan(opt_fov / 2 * D2R);
    if (W >= H) { cam.tx = t; cam.ty = t * H / W; }
    else {
        cam.ty = t; cam.tx = t * W / H;
        /* portrait: tilt up more, so the sky fills the height (horizon a
         * quarter of the way up rather than a third) */
        if (!opt_pitch_set) {
            p = atan(0.5 * cam.ty);
            cam.f = vec(sin(A) * cos(p), cos(A) * cos(p), sin(p));
            cam.u = cross(cam.r, cam.f);
        }
        /* and bring the lake down to near the foot, rather than a band of bank */
        v3 foot = norm(sub(cam.f, scale(cam.u, cam.ty)));
        shore_k = fmax(1.0, -0.92 * asin(foot.z) * R2D / 7.4);

    }
    hfov_deg = 2 * atan(cam.tx) * R2D;
}

/* the view's pitch, as camera() has it */
static double view_pitch(void) { return asin(cam.f.z) * R2D; }

/* a local direction to pixels; 0 if behind */
static int project(v3 d, double *x, double *y) {
    if (span_me >= 0) {
        double h = hypot(d.x, d.y);
        if (h < 1e-6) return 0;
        double u = fmod(atan2(d.x, d.y) - view_az * D2R + 3 * PI, 2 * PI) - PI;
        if (fabs(u) > 0.97 * PI) return 0;
        *x = (u - cyl_u0) / (cyl_u1 - cyl_u0) * W;
        *y = (d.z / h - cyl_v0) / (cyl_v1 - cyl_v0) * H;
        return 1;
    }
    double cz = dot(d, cam.f);
    if (cz < 0.02) return 0;
    *x = (dot(d, cam.r) / cz / cam.tx + 1) * 0.5 * W;
    *y = (dot(d, cam.u) / cz / cam.ty + 1) * 0.5 * H;
    return 1;
}

/* ================= colours of the sky, by the Sun's altitude ================= */
struct key { float s, zen[3], hs[3], ha[3]; };
static const struct key KEYS[] = {
    {-90, {0.021f, 0.0175f, 0.0155f}, {0.125f, 0.095f, 0.071f}, {0.125f, 0.095f, 0.071f}},
    {-18, {0.021f, 0.0175f, 0.0155f}, {0.125f, 0.095f, 0.071f}, {0.125f, 0.095f, 0.071f}},
    {-14, {0.023f, 0.023f, 0.031f}, {0.175f, 0.108f, 0.064f}, {0.118f, 0.092f, 0.076f}},
    {-10, {0.026f, 0.031f, 0.052f}, {0.300f, 0.150f, 0.065f}, {0.085f, 0.070f, 0.074f}},
    {-6, {0.062f, 0.080f, 0.130f}, {0.700f, 0.340f, 0.120f}, {0.185f, 0.150f, 0.165f}},
    {-3, {0.130f, 0.175f, 0.270f}, {0.920f, 0.520f, 0.220f}, {0.400f, 0.300f, 0.320f}},
    {0, {0.240f, 0.330f, 0.460f}, {0.990f, 0.680f, 0.380f}, {0.600f, 0.500f, 0.480f}},
    {4, {0.360f, 0.470f, 0.610f}, {1.000f, 0.820f, 0.580f}, {0.760f, 0.700f, 0.650f}},
    {15, {0.500f, 0.610f, 0.730f}, {0.950f, 0.900f, 0.800f}, {0.900f, 0.870f, 0.800f}},
    {90, {0.530f, 0.640f, 0.760f}, {0.940f, 0.910f, 0.840f}, {0.910f, 0.890f, 0.830f}},
};
static void sky_keys(double s, float *zen, float *hs, float *ha) {
    int n = sizeof KEYS / sizeof *KEYS, i = 0;
    while (i < n - 2 && s > KEYS[i + 1].s) i++;
    double t = clampd((s - KEYS[i].s) / (KEYS[i + 1].s - KEYS[i].s), 0, 1);
    for (int c = 0; c < 3; c++) {
        zen[c] = (float)(KEYS[i].zen[c] + (KEYS[i + 1].zen[c] - KEYS[i].zen[c]) * t);
        hs[c] = (float)(KEYS[i].hs[c] + (KEYS[i + 1].hs[c] - KEYS[i].hs[c]) * t);
        ha[c] = (float)(KEYS[i].ha[c] + (KEYS[i + 1].ha[c] - KEYS[i].ha[c]) * t);
    }
}
/* the faintest star you'd see, by the Sun's altitude */
static double limit_mag(double s) {
    static const double K[][2] = {{-90, 6.5}, {-18, 6.5}, {-15, 6.0}, {-12, 5.0}, {-9, 3.8}, {-6, 2.2},
                                  {-4, 1.0}, {-2, -0.6}, {0, -2.0}, {5, -3.6}, {90, -3.9}};
    int i = 0;
    while (i < 9 && s > K[i + 1][0]) i++;
    double t = clampd((s - K[i][0]) / (K[i + 1][0] - K[i][0]), 0, 1);
    return K[i][1] + (K[i + 1][1] - K[i][1]) * t;
}

/* ================= meteors ================= */
#define NMET 4096
static double met_t[NMET];
static int met_n;
struct meteor { double t0, dur, len, bright; v3 p0, axis; };
static struct meteor meteor_k(int k) {
    struct meteor m;
    uint32_t keep = rng;
    rng = hash32(opt_seed, (uint32_t)k + 1000);
    m.t0 = met_t[k];
    m.dur = 0.5 + 0.7 * frand();
    m.len = (9 + 18 * frand()) * D2R;
    m.bright = 0.7 + 1.1 * frand() * frand();
    double az = view_az + (frand() - 0.5) * hfov_deg * 0.95, alt = 16 + 50 * frand();
    if (span_me >= 0) az = view_az + (wall_u0 + (wall_u1 - wall_u0) * (0.025 + 0.95 * frand())) * R2D;
    m.p0 = vec(sin(az * D2R) * cos(alt * D2R), cos(az * D2R) * cos(alt * D2R), sin(alt * D2R));
    v3 te = norm(cross(vec(0, 0, 1), m.p0)), tn = cross(m.p0, te);
    double ang = (200 + 140 * frand()) * D2R;
    if (frand() < 0.5) ang = PI - ang;          /* left or right, mostly falling */
    v3 v = add(scale(te, cos(ang)), scale(tn, sin(ang)));
    m.axis = norm(cross(m.p0, v));
    rng = keep;
    return m;
}
static v3 rotate(v3 p, v3 k, double a) {           /* Rodrigues */
    return add(add(scale(p, cos(a)), scale(cross(k, p), sin(a))), scale(k, dot(k, p) * (1 - cos(a))));
}

/* ================= a frame ================= */
static double t_base;              /* the sky's clock at el = 0, unix seconds */
static float strip[6 * 4 * 2048];
static int nstrip;

static void strip_seg(double x0, double y0, double x1, double y1, double w0, double w1, double a0, double a1) {
    if (nstrip + 6 > 6 * 2048) return;
    double dx = x1 - x0, dy = y1 - y0, l = hypot(dx, dy);
    if (l < 0.5) return;
    double nx = -dy / l, ny = dx / l;
    double P[4][4] = {{x0 + nx * w0, y0 + ny * w0, 1, a0}, {x0 - nx * w0, y0 - ny * w0, -1, a0},
                      {x1 + nx * w1, y1 + ny * w1, 1, a1}, {x1 - nx * w1, y1 - ny * w1, -1, a1}};
    static const int idx[6] = {0, 1, 2, 2, 1, 3};
    for (int i = 0; i < 6; i++) {
        float *o = strip + 4 * nstrip++;
        o[0] = (float)(P[idx[i]][0] / W * 2 - 1);
        o[1] = (float)(P[idx[i]][1] / H * 2 - 1);
        o[2] = (float)P[idx[i]][2];
        o[3] = (float)P[idx[i]][3];
    }
}

/* ================= labels ================= */
/* printable ASCII in three styles: constellations (spaced roman capitals),
 * stars and planets (italic), as on an old star atlas */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
enum { ST_CON, ST_STAR, ST_PLANET, NST };
static Glyph glyphs[NST][95];
static int atlas_px;                    /* the size it was built for, 0: not yet */
static FT_Library ft;
static FT_Face faces[2];                /* regular, italic */
#define AW 1024
#define AH 512

static void fonts_init(void) {
    if (!FcInit() || FT_Init_FreeType(&ft)) die("font init failed");
    for (int i = 0; i < 2; i++) {
        char pat_s[256];
        snprintf(pat_s, sizeof pat_s, "%s:%s", opt_font, i ? "italic" : "weight=regular");
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

static double style_px(int st, double px) {
    static const double base[NST] = {14, 14.5, 16.5};
    return fmax(9, base[st] * px);
}

static void build_atlas(double px) {
    int key = (int)lround(px * 100);
    if (key == atlas_px) return;
    atlas_px = key;
    unsigned char *a = calloc((size_t)AW * AH, 1);
    int x = 1, y = 1, shelf = 0;
    for (int st = 0; st < NST; st++) {
        FT_Face f = faces[st == ST_CON ? 0 : 1];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)lround(style_px(st, px)));
        for (int c = 0; c < 95; c++) {
            Glyph *g = &glyphs[st][c];
            memset(g, 0, sizeof *g);
            if (FT_Load_Char(f, (FT_ULong)(32 + c), FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot s = f->glyph;
            int w = (int)s->bitmap.width, h = (int)s->bitmap.rows;
            if (x + w + 2 > AW) { x = 1; y += shelf + 2; shelf = 0; }
            if (y + h + 2 > AH) break;          /* absurd sizes: drop the rest */
            for (int yy = 0; yy < h; yy++)
                memcpy(a + (size_t)(y + yy) * AW + x, s->bitmap.buffer + yy * s->bitmap.pitch, (size_t)w);
            *g = (Glyph){(short)x, (short)y, (short)w, (short)h, (short)s->bitmap_left, (short)s->bitmap_top,
                         (short)(s->advance.x >> 6)};
            x += w + 2;
            if (h > shelf) shelf = h;
        }
    }
    if (!tex_atlas) {
        glGenTextures(1, &tex_atlas);
        glBindTexture(GL_TEXTURE_2D, tex_atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, AW, AH, 0, GL_RED, GL_UNSIGNED_BYTE, a);
    free(a);
}

static const Glyph *glyph_of(int st, char c) {
    return &glyphs[st][(unsigned char)c >= 32 && (unsigned char)c < 127 ? c - 32 : 0];
}
static double text_w(int st, const char *str, double track) {
    double w = 0;
    for (const char *p = str; *p; p++) w += glyph_of(st, *p)->adv + (p[1] ? track : 0);
    return w;
}

#define TEXT_MAX (6 * 4096)
static float textv[TEXT_MAX * 7];
static int ntext;
static float placed[256][4];            /* label boxes so far this frame */
static int nplaced;

static void text_vert(double x, double y, double u, double v, const float *c) {
    float *o = textv + 7 * ntext++;
    o[0] = (float)(x / W * 2 - 1); o[1] = (float)(y / H * 2 - 1);
    o[2] = (float)(u / AW); o[3] = (float)(v / AH);
    o[4] = c[0]; o[5] = c[1]; o[6] = c[2];
}

/* set str with its baseline at y (pixels, y up) from x, unless it would land
 * on a label already placed; returns whether it did */
static int label(int st, double x, double y, const char *str, double track, const float *col) {
    double w = text_w(st, str, track), cap = glyph_of(st, 'H')->top, pad = 0.35 * cap;
    double bx0 = x - pad, by0 = y - 0.3 * cap - pad, bx1 = x + w + pad, by1 = y + cap + pad;
    if (bx1 < 0 || bx0 > W || by1 < 0 || by0 > H) return 0;
    for (int i = 0; i < nplaced; i++)
        if (bx0 < placed[i][2] && bx1 > placed[i][0] && by0 < placed[i][3] && by1 > placed[i][1]) return 0;
    if (nplaced < 256) { float *b = placed[nplaced++]; b[0] = (float)bx0; b[1] = (float)by0; b[2] = (float)bx1; b[3] = (float)by1; }
    double pen = round(x);
    y = round(y);
    for (const char *p = str; *p; p++) {
        const Glyph *g = glyph_of(st, *p);
        if (g->w && ntext + 6 <= TEXT_MAX) {
            double x0 = pen + g->left, x1 = x0 + g->w, y1 = y + g->top, y0 = y1 - g->h;
            double u0 = g->x, u1 = g->x + g->w, v0 = g->y, v1 = g->y + g->h;   /* v0: the glyph's top row */
            text_vert(x0, y1, u0, v0, col); text_vert(x1, y1, u1, v0, col); text_vert(x1, y0, u1, v1, col);
            text_vert(x0, y1, u0, v0, col); text_vert(x1, y0, u1, v1, col); text_vert(x0, y0, u0, v1, col);
        }
        pen += g->adv + track;
    }
    return 1;
}

/* the land's highest point (degrees) between two bearings (radians) */
static double ridge_max(double az0, double az1) {
    double top = -90;
    int i0 = (int)floor(az0 / (2 * PI) * RIDGE_N), i1 = (int)ceil(az1 / (2 * PI) * RIDGE_N);
    for (int i = i0; i <= i1; i++) {
        const float *r = ridge + 4 * (((i % RIDGE_N) + RIDGE_N) % RIDGE_N);
        top = fmax(top, fmax(r[0], fmax(r[1], r[2])));
    }
    return top;
}

static double label_pxrad;              /* radians per pixel, near enough */

/* where a local direction lands, if a label from dx0 to dx1 pixels across it
 * and dy below it would stand clear of the land */
static int sky_spot(v3 d, double min_alt, double dx0, double dx1, double dy, double *x, double *y) {
    if (d.z < sin(min_alt * D2R)) return 0;
    double h = hypot(d.x, d.y), az = atan2(d.x, d.y), alt = asin(d.z) * R2D;
    double k = label_pxrad / fmax(h, 0.05);         /* bearing per pixel, wider towards the zenith */
    if (alt - dy * label_pxrad * R2D < ridge_max(az + dx0 * k, az + dx1 * k) + 0.6) return 0;
    return project(d, x, y);
}

static v3 star_dir(int i) { return refract(norm(mvec(sky.eq2hor, vec(starv[8 * i], starv[8 * i + 1], starv[8 * i + 2])))); }

static void labels(double px, double pxrad, double fade, double moon_amt, double lim) {
    ntext = nplaced = 0;
    if (fade <= 0) return;
    build_atlas(px);
    label_pxrad = pxrad;
    double dim = fade * (1 - 0.35 * moon_amt);
    double x, y, gap = 7 * px + 2;
    char buf[64];
    /* the planets first: they're what people look for */
    static const char *const PNAME[5] = {"Mercury", "Venus", "Mars", "Jupiter", "Saturn"};
    float pc[3] = {(float)(0.95 * 0.55 * dim), (float)(0.86 * 0.55 * dim), (float)(0.64 * 0.55 * dim)};
    for (int i = 0; i < 5; i++) {
        const float *o = sky.planet[i];
        if (o[6] > lim) continue;
        v3 d = refract(norm(mvec(sky.eq2hor, vec(o[0], o[1], o[2]))));
        int st = ST_PLANET;
        if (!sky_spot(d, 1, -gap, gap + text_w(st, PNAME[i], 0), glyph_of(st, 'x')->top, &x, &y)) continue;
        label(st, x + gap, y - glyph_of(st, 'x')->top * 0.5, PNAME[i], 0, pc);
    }
    /* the Moon's own place is plain enough; the bright stars by name */
    float sc[3] = {(float)(0.80 * 0.40 * dim), (float)(0.77 * 0.40 * dim), (float)(0.70 * 0.40 * dim)};
    for (int i = 0; i < NSTARNAME; i++) {
        int k = star_name_idx[i];
        if (k < 0 || BRIGHT[k].v > lim - 0.5) continue;
        int st = ST_STAR;
        if (!sky_spot(star_dir(k), 2, -gap, gap + text_w(st, STAR_NAME[i][1], 0), glyph_of(st, 'x')->top, &x, &y)) continue;
        label(st, x + gap, y - glyph_of(st, 'x')->top * 0.5, STAR_NAME[i][1], 0, sc);
    }
    /* constellations, in spaced capitals at the middle of the figure */
    float cc[3] = {(float)(0.84 * 0.50 * dim), (float)(0.63 * 0.50 * dim), (float)(0.17 * 0.50 * dim)};
    for (int f = 0; f < NFIGURES; f++) {
        if (!fig[f].name) continue;
        v3 d = refract(norm(mvec(sky.eq2hor, fig[f].c)));
        int n = 0;
        for (const char *p = fig[f].name; *p && n < 63; p++) buf[n++] = (char)toupper((unsigned char)*p);
        buf[n] = 0;
        int st = ST_CON;
        double track = 0.22 * style_px(st, px);
        double w = text_w(st, buf, track);
        if (!sky_spot(d, 4, -w / 2, w / 2, glyph_of(st, 'H')->top, &x, &y)) continue;
        label(st, x - w / 2, y - glyph_of(st, 'H')->top * 0.5, buf, track, cc);
    }
}

static void render(double el, GLuint target) {
    double t = t_base + el * opt_timelapse;
    sky_compute(t);
    camera();
    double s = sky.sun_alt;
    double px = span_me >= 0 ? front_px * span_sc : fmax(1.0, fmin(W / 1920.0, H / 1080.0));
    double pxrad = span_me >= 0 ? (cyl_v1 - cyl_v0) / H : 2 * cam.ty / H;
    double night = 1 - smoothd(-10, 1, s);
    double moon_up = smoothd(-2, 8, sky.moon_alt);
    double moon_amt = pow(sky.moon_k, 1.4) * moon_up;

    float zen[3], hs[3], ha[3];
    sky_keys(s, zen, hs, ha);
    /* moonlight lifts the night sky towards a soft grey */
    for (int c = 0; c < 3; c++) {
        float lift = (float)(moon_amt * night * (c == 2 ? 0.092 : c == 1 ? 0.095 : 0.099));
        zen[c] += lift * 0.75f; hs[c] += lift; ha[c] += lift;
    }

    /* the half-size sky's target */
    int hw = (W + 1) / 2, hh = (H + 1) / 2;
    if (hw != bg_w || hh != bg_h) {
        if (bg_w) { glDeleteTextures(1, &tex_bg); glDeleteFramebuffers(1, &fbo_bg); }
        bg_w = hw; bg_h = hh;
        tex_bg = mk_tex(GL_RGBA16F, hw, hh, GL_RGBA, GL_FLOAT, NULL, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1, &fbo_bg);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bg);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bg, 0);
    }
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_ridge);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_noise);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_mw);
    glActiveTexture(GL_TEXTURE0);

    double wind_t = el * sqrt(fmax(opt_timelapse, 1));
    double wa = (hash32(opt_seed, 5) % 360) * D2R;
    GLuint progs[4] = {prog_sky, prog_star, prog_line, prog_text};
    for (int i = 0; i < 4; i++) {
        GLuint p = progs[i];
        glUseProgram(p);
        u2f(p, "uRes", W, H);
        u3v(p, "uCF", cam.f); u3v(p, "uCR", cam.r); u3v(p, "uCU", cam.u);
        u2f(p, "uTan", cam.tx, cam.ty);
        glUniform1i(U(p, "uCyl"), span_me >= 0);
        glUniform4f(U(p, "uSpan"), (float)cyl_u0, (float)cyl_u1, (float)cyl_v0, (float)cyl_v1);
        u1f(p, "uAz0", view_az * D2R);
        u1f(p, "uShoreK", shore_k);
        u1f(p, "uCloudCov", opt_clouds);
        u1f(p, "uTime", el);
        u1f(p, "uPxRad", pxrad);
        u2f(p, "uWind", cos(wa) * 0.0022 * wind_t + 0.37 * (opt_seed % 97), sin(wa) * 0.0022 * wind_t + 0.11 * (opt_seed % 89));
    }

    /* ---- the sky, the land and the lake ---- */
    GLuint p = prog_sky;
    glUseProgram(p);
    umat(p, "uHor2Gal", sky.hor2gal);
    u3v(p, "uSun", sky.sun); u3v(p, "uMoon", sky.moon);
    u3v(p, "uMoonW", sky.moon_w); u3v(p, "uMoonA", sky.moon_a); u3v(p, "uMoonSun", sky.moon_sun);
    u1f(p, "uMoonR", sky.moon_sd * opt_moon);
    {
        double x0, y0, xw, yw, xa, ya, e = 1e-3;
        float rpx = 0, ew[2] = {1, 0}, ea[2] = {0, 1}, mp[2] = {0, 0};
        if (project(sky.moon, &x0, &y0) && project(norm(add(sky.moon, scale(sky.moon_w, e))), &xw, &yw) &&
            project(norm(add(sky.moon, scale(sky.moon_a, e))), &xa, &ya)) {
            double lw = hypot(xw - x0, yw - y0), la = hypot(xa - x0, ya - y0);
            rpx = (float)(sky.moon_sd * opt_moon * (lw + la) / 2 / e);
            ew[0] = (float)((xw - x0) / lw); ew[1] = (float)((yw - y0) / lw);
            double sgn = -ew[1] * (xa - x0) + ew[0] * (ya - y0) >= 0 ? 1 : -1;
            ea[0] = (float)(-ew[1] * sgn); ea[1] = (float)(ew[0] * sgn);
            mp[0] = (float)x0; mp[1] = (float)y0;
        }
        u1f(p, "uMoonRpx", rpx);
        glUniform2fv(U(p, "uMoonPx"), 1, mp);
        glUniform2fv(U(p, "uMoonEW"), 1, ew);
        glUniform2fv(U(p, "uMoonEA"), 1, ea);
    }
    u1f(p, "uSunR", 0.00465 * 2);
    u1f(p, "uMWAmt", 0.50 * (1 - smoothd(-18, -12, s)) * (1 - 0.8 * moon_amt));
    /* a lock screen is seen in a lit room: lift the night a little */
    u1f(p, "uExpo", 1.0 + 0.6 * night);
    u1f(p, "uTownAz", (hash32(opt_seed, 9) % 360) * D2R);
    u1f(p, "uDither", fmod(el * 61.7, 97.0));
    u3c(p, "uZen", zen); u3c(p, "uHorS", hs); u3c(p, "uHorA", ha);
    double sg = smoothd(-9, 1, s), day = smoothd(-4, 10, s);
    double low = 1 - smoothd(5, 25, s);
    u3v(p, "uSunGlow", scale(vec(0.55 + 0.45 * low, 0.48 + 0.12 * low, 0.36 - 0.12 * low), sg * (0.5 + 0.5 * day)));
    double mg = moon_up * sky.moon_k * (0.25 + 0.75 * night) * (s > 0 ? 0.25 : 1);
    u3v(p, "uMoonGlow", scale(vec(0.30, 0.285, 0.26), mg));
    double mc = 0.95 - 0.6 * day;
    u3v(p, "uMoonCol", vec(1.00 * mc, 0.955 * mc, 0.87 * mc));
    u3v(p, "uEarthshine", scale(vec(0.034, 0.035, 0.040), (1 - sky.moon_k) * (1 - sky.moon_k) * night * night));
    u3v(p, "uSunDisc", vec(3.0, 2.7, 2.2));
    float amb[3];
    for (int c = 0; c < 3; c++) amb[c] = ha[c] * (float)(1.0 + 0.25 * day) + (float)(night * (c == 0 ? 0.020 : c == 1 ? 0.014 : 0.009));
    u3c(p, "uCloudAmb", amb);
    double lit = smoothd(-11, -3, s) * (1 - smoothd(2, 12, s));
    u3v(p, "uSunLit", add(scale(vec(0.95, 0.45, 0.22), lit), scale(vec(0.12, 0.11, 0.10), day)));
    v3 landn = vec(0.017, 0.014, 0.0115), landd = vec(0.25, 0.235, 0.18);
    v3 land = add(scale(landn, 1 - day), scale(landd, day));
    land = add(land, scale(vec(0.030, 0.029, 0.029), moon_amt * night));
    u3v(p, "uLand", land);
    u3v(p, "uWaterBody", scale(land, 0.7));
    u3v(p, "uRimMoon", scale(vec(0.30, 0.28, 0.25), moon_amt * night * smoothd(-1, 6, sky.moon_alt) * (1 - smoothd(25, 50, sky.moon_alt))));
    u3v(p, "uRimSun", scale(vec(0.80, 0.42, 0.16), smoothd(-9, -2, s) * (1 - smoothd(3, 15, s))));
    u3v(p, "uTown", scale(vec(0.040, 0.024, 0.011), night));
    u3v(p, "uMWCol", vec(0.86, 0.82, 0.74));
    u3v(p, "uMWWarm", vec(1.00, 0.82, 0.58));
    glBindVertexArray(vao_full);
    /* pass 0: the sky at half size */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_bg);
    glViewport(0, 0, hw, hh);
    u2f(p, "uRes", hw, hh);
    glUniform1i(U(p, "uPass"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    /* pass 1: land, lake, Moon, over it */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    u2f(p, "uRes", W, H);
    glUniform1i(U(p, "uPass"), 1);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_bg);
    glActiveTexture(GL_TEXTURE0);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    /* ---- constellation figures ---- */
    double la = opt_lines ? 0.20 * smoothd(-3, -10, s) * (1 - 0.4 * moon_amt) : 0;
    nstrip = 0;
    if (la > 0) {
        double lw = 0.75 * px + 0.6, gap = 6 * px;
        for (int i = 0; i < nseg; i++) {
            v3 a = refract(norm(mvec(sky.eq2hor, vec(starv[8 * seg[2 * i]], starv[8 * seg[2 * i] + 1], starv[8 * seg[2 * i] + 2]))));
            v3 b = refract(norm(mvec(sky.eq2hor, vec(starv[8 * seg[2 * i + 1]], starv[8 * seg[2 * i + 1] + 1], starv[8 * seg[2 * i + 1] + 2]))));
            if (a.z < -0.05 && b.z < -0.05) continue;
            /* clip to just in front of the eye */
            double za = dot(a, cam.f), zb = dot(b, cam.f), zmin = 0.05;
            if (za < zmin && zb < zmin) continue;
            if (za < zmin) a = norm(add(a, scale(sub(b, a), (zmin - za) / (zb - za))));
            if (zb < zmin) b = norm(add(b, scale(sub(a, b), (zmin - zb) / (za - zb))));
            double x0, y0, x1, y1;
            if (!project(a, &x0, &y0) || !project(b, &x1, &y1)) continue;
            double l = hypot(x1 - x0, y1 - y0);
            if (l < 2 * gap + 3) continue;
            double ux = (x1 - x0) / l, uy = (y1 - y0) / l;
            strip_seg(x0 + ux * gap, y0 + uy * gap, x1 - ux * gap, y1 - uy * gap, lw, lw, la, la);
        }
        if (nstrip) {
            glUseProgram(prog_line);
            glUniform3f(U(prog_line, "uColor"), 0.84f, 0.63f, 0.17f);
            glBindVertexArray(vao_strip);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * nstrip, strip, GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, nstrip);
        }
    }

    /* ---- stars and planets, then their reflections ---- */
    double moon_wash = 0.85 * moon_amt * night;
    glUseProgram(prog_star);
    umat(prog_star, "uEq2Hor", sky.eq2hor);
    u1f(prog_star, "uLim", limit_mag(s) + 0.4 * night - 1.1 * moon_amt * night);
    u1f(prog_star, "uGain", (2.2 + 1.5 * night) * (1 - 0.8 * day));
    u1f(prog_star, "uMaxI", 30 - 28 * day);          /* by day Venus is a pale point at most */
    u1f(prog_star, "uPx", px);
    u1f(prog_star, "uMoonR", sky.moon_sd * opt_moon * 1.05);
    u1f(prog_star, "uMoonWash", moon_wash);
    u3v(prog_star, "uMoon", sky.moon);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_planet);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof sky.planet, sky.planet);
    for (int mirror = 0; mirror < 2; mirror++) {
        u1f(prog_star, "uMirror", mirror);
        glBindVertexArray(vao_star);
        glDrawArrays(GL_POINTS, 0, nstars);
        glBindVertexArray(vao_planet);
        glDrawArrays(GL_POINTS, 0, 5);
    }

    /* ---- names: planets, bright stars, constellations ---- */
    if (opt_labels) {
        labels(px, pxrad, smoothd(-3, -10, s), moon_amt, limit_mag(s) + 0.4 * night - 1.1 * moon_amt * night);
        if (ntext) {
            glUseProgram(prog_text);
            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_atlas);
            glActiveTexture(GL_TEXTURE0);
            glBindVertexArray(vao_text);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * ntext, textv, GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, ntext);
        }
    }

    /* ---- meteors ---- */
    if (s < -12 && opt_timelapse <= 60) {
        if (!met_n) {
            uint32_t keep = rng;
            rng = hash32(opt_seed, 3);
            double tt = 4 + 10 * frand();
            for (met_n = 0; met_n < NMET; met_n++) {
                met_t[met_n] = tt;
                tt += 6 - 34 * log(frand() + 1e-6);    /* about one every 40 s */
            }
            rng = keep;
        }
        nstrip = 0;
        for (int k = 0; k < NMET && met_t[k] <= el; k++) {
            if (el > met_t[k] + 1.6) continue;
            struct meteor m = meteor_k(k);
            double u = (el - m.t0) / m.dur;
            if (u < 0 || u > 1) continue;
            double head = m.len * u, tail = fmax(0, head - m.len * 0.5);
            double env = pow(sin(PI * u), 0.6) * m.bright;
            int nsub = 8;
            for (int j = 0; j < nsub; j++) {
                double a0 = tail + (head - tail) * j / nsub, a1 = tail + (head - tail) * (j + 1) / nsub;
                v3 p0 = rotate(m.p0, m.axis, a0), p1 = rotate(m.p0, m.axis, a1);
                double x0, y0, x1, y1;
                if (!project(p0, &x0, &y0) || !project(p1, &x1, &y1)) continue;
                double f0 = (double)j / nsub, f1 = (double)(j + 1) / nsub;
                strip_seg(x0, y0, x1, y1, (0.3 + 1.0 * f0) * px + 0.5, (0.3 + 1.0 * f1) * px + 0.5,
                          env * f0 * f0, env * f1 * f1);
            }
        }
        if (nstrip) {
            glUseProgram(prog_line);
            glUniform3f(U(prog_line, "uColor"), 1.0f, 0.95f, 0.82f);
            glBindVertexArray(vao_strip);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * nstrip, strip, GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, nstrip);
        }
    }
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

/* ================= --layer: a native Wayland surface (wlr-layer-shell + EGL) ================= */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "nightsky");
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

/* --info: the numbers behind the picture, for checking against an almanac */
static void hor_print(const char *name, v3 d, double mag) {
    double alt = asin(clampd(d.z, -1, 1)) * R2D, az = fmod(atan2(d.x, d.y) * R2D + 360, 360);
    printf("  %-10s alt %6.2f  az %6.2f", name, alt, az);
    if (!isnan(mag)) printf("  mag %5.2f", mag);
    printf("\n");
}
static void eq_print(const char *name, v3 e) {
    double ra = fmod(atan2(e.y, e.x) * R2D + 360, 360) / 15, dec = asin(e.z) * R2D;
    printf("  %-10s RA %2dh%05.2fm  Dec %+6.2f (J2000)\n", name, (int)ra, fmod(ra, 1) * 60, dec);
}
static void info(void) {
    sky_compute(t_base);
    time_t tt = (time_t)t_base;
    char buf[64];
    strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S %Z", localtime(&tt));
    printf("%s  (unix %.0f)  lat %.3f lon %.3f\n", buf, t_base, opt_lat, opt_lon);
    v3 sun_eq = mvec(mT(sky.eq2hor), sky.sun);
    printf("Sun and Moon (apparent, refracted; Moon topocentric):\n");
    hor_print("Sun", sky.sun, NAN);
    hor_print("Moon", sky.moon, NAN);
    double el = acos(clampd(dot(sky.sun, sky.moon), -1, 1)) * R2D;
    printf("  Moon lit %.1f%%, elongation %.1f deg, %s; semi-diameter %.1f'\n", sky.moon_k * 100, el,
           dot(cross(sky.sun, sky.moon), vec(0, cos(opt_lat * D2R), sin(opt_lat * D2R))) > 0 ? "waxing" : "waning",
           sky.moon_sd * R2D * 60);
    eq_print("Sun", sun_eq);
    eq_print("Moon", mvec(mT(sky.eq2hor), sky.moon));
    static const char *pn[5] = {"Mercury", "Venus", "Mars", "Jupiter", "Saturn"};
    printf("Planets:\n");
    for (int i = 0; i < 5; i++) {
        v3 e = vec(sky.planet[i][0], sky.planet[i][1], sky.planet[i][2]);
        eq_print(pn[i], e);
        hor_print("", mvec(sky.eq2hor, e), sky.planet[i][6]);
    }
    printf("Stars:\n");
    static const char *st[] = {"alp UMi", "alp Ori", "bet Ori", "alp CMa", "alp Lyr", "alp UMa", "alp Cru"};
    for (size_t i = 0; i < sizeof st / sizeof *st; i++) {
        int k = -1;
        for (int j = 0; j < NBRIGHT; j++) if (!strcmp(BRIGHT[j].name, st[i])) k = j;
        if (k >= 0) hor_print(st[i], refract(mvec(sky.eq2hor, radec(BRIGHT[k].ra * 15 * D2R, BRIGHT[k].dec * D2R))), BRIGHT[k].v);
    }
}

/* ================= setup ================= */
static void usage(void) {
    fprintf(stderr,
        "usage: nightsky [--window | --root | --layer] [--lat LAT --lon LON] [--time \"YYYY-MM-DD HH:MM\"]\n"
        "                [--timelapse X] [--az DEG] [--pitch DEG] [--fov DEG] [--no-lines] [--no-labels] [--font NAME]\n"
        "                [--clouds X | --no-clouds]\n"
        "                [--moon SCALE] [--fps N] [--seed N] [--epoch UNIXTIME]\n"
        "                [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX] [--render-scale F]\n"

        "                [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH] [--info]\n"
        "Location: --lat/--lon, else ~/.config/wallpapers/location (\"LAT LON\", north and east positive),\n"
        "else Greenwich. Times are local (set TZ=UTC for UT).\n");
    exit(1);
}

static void load_location(void) {
    if (loc_given) return;
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
            opt_lat = la; opt_lon = lo;
            break;
        }
    }
    fclose(f);
}

static double parse_time(const char *s) {
    struct tm tm = {0};
    int Y, M, D, h = 0, m = 0, sec = 0;
    if (sscanf(s, "%d-%d-%d%*[ T]%d:%d:%d", &Y, &M, &D, &h, &m, &sec) < 3) usage();
    tm.tm_year = Y - 1900; tm.tm_mon = M - 1; tm.tm_mday = D;
    tm.tm_hour = h; tm.tm_min = m; tm.tm_sec = sec; tm.tm_isdst = -1;
    return (double)mktime(&tm);
}

/* face the equator; if the Moon is up but out of frame, turn just enough to bring it in */
static void choose_view(void) {
    if (!isnan(opt_az)) { view_az = opt_az; return; }
    view_az = opt_lat >= 0 ? 180 : 0;
    sky_compute(t_base);
    camera();
    double top = span_me >= 0 ? atan(front_top) * R2D : view_pitch() + atan(cam.ty) * R2D;
    v3 m = sky.moon;
    double malt = sky.moon_alt, maz = atan2(m.x, m.y) * R2D;
    if (malt > 3 && malt < top - 4 && sky.moon_k > 0.08) {
        double diff = fmod(maz - view_az + 540, 360) - 180, half = hfov_deg / 2 - 22;
        /* spanning: anywhere on the front output will do */
        if (span_me >= 0) half = fmax(front_u * R2D - 12, 0);
        if (fabs(diff) > half && fabs(diff) < 115) view_az += diff - copysign(half, diff);
    }
}

static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static GLuint offscreen(void) {
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-lat") && i + 1 < argc) { opt_lat = strtod(argv[++i], NULL); loc_given = 1; }
        else if (!strcmp(a, "-lon") && i + 1 < argc) { opt_lon = strtod(argv[++i], NULL); loc_given = 1; }
        else if (!strcmp(a, "-time") && i + 1 < argc) opt_time = argv[++i];
        else if (!strcmp(a, "-timelapse") && i + 1 < argc) opt_timelapse = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-az") && i + 1 < argc) opt_az = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-pitch") && i + 1 < argc) { opt_pitch = strtod(argv[++i], NULL); opt_pitch_set = 1; }
        else if (!strcmp(a, "-fov") && i + 1 < argc) { opt_fov = strtod(argv[++i], NULL); opt_fov_set = 1; }
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-lines")) opt_lines = 1;
        else if (!strcmp(a, "-no-lines")) opt_lines = 0;
        else if (!strcmp(a, "-labels")) opt_labels = 1;
        else if (!strcmp(a, "-no-labels")) opt_labels = 0;
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-clouds") && i + 1 < argc) opt_clouds = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-no-clouds")) opt_clouds = 0;
        else if (!strcmp(a, "-moon") && i + 1 < argc) opt_moon = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-info")) opt_info = 1;
        else usage();
    }
    if (span_me >= span_n) span_me = -1;
    if (span_me >= 0) span_front = front_output();
    if (fabs(opt_lat) > 90 || opt_fov < 1 || opt_fov > (span_me >= 0 ? 340 : 170) || opt_fps < 1 || opt_moon <= 0) usage();
    if (opt_clouds < 0) opt_clouds = 0;
    if (opt_clouds > 1) opt_clouds = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    load_location();
    /* spanning: every instance's sky clock starts at the shared epoch */
    double start = wallclock();
    if (span_me >= 0 && opt_epoch) start = opt_epoch;
    t_base = opt_time ? parse_time(opt_time) : start;
    if (opt_info) { info(); return 0; }
    int offscr = opt_shot || opt_bench;

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offscr) wl_init();
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
        if (offscr) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Nightsky");
        XClassHint chint = {"nightsky", "Nightsky"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscr) XMapRaised(dpy, win);
        /* a core 3.3 context where GLX offers one */
        typedef GLXContext (*ctxattr_fn)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
        ctxattr_fn mk = (ctxattr_fn)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
        GLXContext ctx = NULL;
        int nfb = 0;
        int fbattr[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT,
                        GLX_VISUAL_ID, (int)vi->visualid, GLX_DOUBLEBUFFER, True, None};
        GLXFBConfig *fbc = mk ? glXChooseFBConfig(dpy, scr, fbattr, &nfb) : NULL;
        if (fbc && nfb > 0) {
            int cattr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                           GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None};
            ctx = mk(dpy, fbc[0], NULL, True, cattr);
            XFree(fbc);
        }
        if (!ctx) ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    build_stars();
    build_figures();
    if (opt_labels) fonts_init();
    bake_ridge(opt_seed);
    gl_init();
    camera();
    choose_view();

    if (opt_shot) {
        GLuint fbo = offscreen();
        render(opt_at, fbo);
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }
    if (opt_bench) {
        GLuint fbo = offscreen();
        render(0, fbo);
        glFinish();
        double sum = 0, worst = 0, gsum = 0;
        double *g = malloc(sizeof(double) * opt_bench);
        GLuint q;
        glGenQueries(1, &q);
        for (int i = 0; i < opt_bench; i++) {
            double t0 = now();
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(opt_at + i / 30.0, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            double ms = (now() - t0) * 1000;
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            g[i] = ns / 1e6;
            gsum += g[i];
            sum += ms;
            if (ms > worst) worst = ms;
        }
        qsort(g, opt_bench, sizeof *g, cmpd);
        printf("nightsky: %d frames at %dx%d: avg %.2f ms, worst %.2f ms; GPU avg %.2f ms, median %.2f ms  (%s)\n",
               opt_bench, W, H, sum / opt_bench, worst, gsum / opt_bench, g[opt_bench / 2],
               (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    if (span_me >= 0) t0 -= wallclock() - start;   /* el counts from the epoch */
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
        render(t - t0, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / opt_fps - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
