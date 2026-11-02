/*
 * steam — a night express: a big steam engine, its tender and a rake of
 * lit carriages running through moonlit country, filmed.
 *
 * The machine's vitals are part of the scene rather than read off gauges:
 *
 *   the CPU load      is how hard she's worked: the speed, the exhaust beat,
 *                     the valve gear's cut-off, the smoke
 *   the CPU heat      is the fire: the glow under the engine and from the cab,
 *                     the plume's lit underside, the sparks
 *   memory in use     is the train's length (carriages are added or taken off
 *                     out of shot)
 *   the host's name   is the engine's name, on the brass plates over her
 *                     driving wheels and the headboard; and the stations'
 *   the time          is the station clocks', and a small caption with the
 *                     time away from the machine (the screen locked)
 *
 * The world: the line wanders and climbs through farmland and hills; every
 * few kilometres come a station with its lamps, a level crossing with a car
 * waiting, a viaduct over a misty valley, a tunnel; signals clear ahead of
 * the train and go back to danger behind it; telegraph wires dip between
 * their poles. The moon has its real phase for the date and roughly its real
 * bearing (lifted to be seen); the seed picks the weather.
 *
 * The camera: long, slow takes with dips to black between them, chosen by
 * what's coming up: alongside the driving wheels with the rods whirling, low
 * at the lineside as she bears down and roars past, a crane shot as she
 * crosses the viaduct, out of the cab along the boiler, on a platform as she
 * runs through, from behind as she dives into a tunnel, high and wide.
 *
 * Drawn with procedural meshes (built at start), terrain tiles streamed
 * round the camera, moon shadows, smoke and steam as lit particles, bloom
 * and a filmic grade.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * Every instance runs the same simulation off the shared epoch (reading
 * the sensors on the same wall-clock seconds) and films it through one
 * virtual camera centred on the middle monitor; each draws just its own
 * slice of that camera's view, so the wall is one picture.
 *
 * Same plumbing as the other savers: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame
 * offscreen, --bench N to time N frames.
 *
 * --scale F sets the scene's render scale (by default it keeps to about 1200p's worth
 * of pixels); --weather 0-3 (clear, rain, snow, mist) overrides the seed's;
 * --view N holds one kind of shot and --fire starts a round of firing at
 * --at, for testing; so do the environment variables STEAM_LIST=SECONDS
 * (print the director's shots), STEAM_LOAD=0..1 (a fixed CPU load),
 * STEAM_FULLSIM (a --shot simulates from the start), STEAM_BLOW (the safety
 * valves and whistle blowing) and STEAM_FIRE_AGE (where --fire starts).
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
#include <pthread.h>
#include "proto-layer-shell.h"

static int W = 1280, H = 720;
static int opt_root, opt_layer, opt_bench;
static float opt_fps = 60;
static const char *opt_font = "C059", *opt_sans = "Gillius ADF";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the run started at (spanning) */
static float opt_scale;                /* the scene's render scale; 0: by the output's size */
static int opt_view = -1;              /* --view N: hold one kind of shot (for testing) */
static int opt_weather = -1;           /* --weather N: 0 clear, 1 rain, 2 snow, 3 mist */
static int opt_caption = 1;

/* ---- spanning: one instance per output, one camera ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1;

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
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float sstepf(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
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

#include "gfx.h"
#include "sensors.h"
#include "world.h"
#include "shaders.h"
#include "loco.h"

static void smoke_step(double dt);
static void smoke_draw(void);
static void smoke_init(void);
static void smoke_targets(int sw, int sh);
static void smoke_targets_free(void);
static void add_light(const double *wp, float rad, float r, float g, float b);
static void set_common(GLuint p);

/* ---- the train ---- */
#define MAXCOACH 9
static struct {
    double z;                  /* world z of the engine's origin */
    double v, vt;              /* speed and where it's heading, m/s */
    double wa[4];              /* wheel angles: drivers, bogie, trailing, tender/carriage */
    double beats;              /* exhausts so far (4 per turn of the drivers) */
    float fire;                /* the fire's brightness, 0..1 */
    float cutoff;              /* the valve gear's setting, 0.15 .. 0.75 */
    int ncoach, ncoach_want;
    float lit;                 /* fraction of the compartments lit */
    float load_s, temp_s;
} tr;

/* where everything is: a vehicle on the line, centred at zc, its bogie
 * pivots a ahead and b behind */
typedef struct { double o[3]; V3 X, Y, Z; } Frame;
static Frame vframe(double zc, double a, double b) {
    Frame f;
    double zf = zc + a, zr = zc - b;
    double pf[3] = {track_x(zf), track_y(zf), zf}, pr[3] = {track_x(zr), track_y(zr), zr};
    V3 Z = vnorm(v3((float)(pf[0] - pr[0]), (float)(pf[1] - pr[1]), (float)(pf[2] - pr[2])));
    f.o[0] = (pf[0] * b + pr[0] * a) / (a + b);
    f.o[1] = (pf[1] * b + pr[1] * a) / (a + b);
    f.o[2] = zc;
    /* canted into the curves */
    float roll = clampf((float)(track_ddx(zc) * 90), -0.1f, 0.1f);
    V3 X = vnorm(vcross(v3(0, 1, 0), Z));
    V3 Y = vcross(Z, X);
    V3 X2 = vadd(vmul(X, cosf(roll)), vmul(Y, sinf(roll)));
    f.X = X2; f.Y = vcross(Z, X2); f.Z = Z;
    return f;
}
/* the train's vehicles: 0 the engine, 1 the tender, 2.. the carriages */
#define TENDER_DZ 10.05
static double veh_z(int i) {
    if (i == 0) return tr.z;
    if (i == 1) return tr.z - TENDER_DZ;
    return tr.z - TENDER_DZ - 3.9 - 0.45 - COACH_L / 2 - (i - 2) * (COACH_L + 0.9);
}
static Frame veh_frame(int i) {
    double z = veh_z(i);
    if (i == 0) return vframe(z, 5.6, 2.2);
    if (i == 1) return vframe(z, 2.1, 2.1);
    return vframe(z, COACH_L / 2 - 3, COACH_L / 2 - 3);
}
static double train_tail_z(void) { return veh_z(1 + tr.ncoach) - COACH_L / 2; }

/* ---- the camera ---- */
static struct {
    double p[3];
    V3 R, U, F;                /* right, up, forward */
    float hfov;                /* across a 16:9 frame, degrees */
    float focus, fstop;        /* depth of field: focus distance (m), blur scale */
} cam;
static void cam_look(const double *p, const double *at, float roll) {
    for (int k = 0; k < 3; k++) cam.p[k] = p[k];
    V3 F = vnorm(v3((float)(at[0] - p[0]), (float)(at[1] - p[1]), (float)(at[2] - p[2])));
    V3 R = vnorm(vcross(F, v3(0, 1, 0)));
    V3 U = vcross(R, F);
    cam.R = vnorm(vadd(vmul(R, cosf(roll)), vmul(U, sinf(roll))));
    cam.U = vcross(cam.R, F);
    cam.F = F;
    cam.focus = (float)sqrt((at[0] - p[0]) * (at[0] - p[0]) + (at[1] - p[1]) * (at[1] - p[1]) + (at[2] - p[2]) * (at[2] - p[2]));
}
/* a point in a vehicle's frame, in the world */
static void fpt(const Frame *f, float x, float y, float z, double *out) {
    out[0] = f->o[0] + f->X.x * x + f->Y.x * y + f->Z.x * z;
    out[1] = f->o[1] + f->X.y * x + f->Y.y * y + f->Z.y * z;
    out[2] = f->o[2] + f->X.z * x + f->Y.z * y + f->Z.z * z;
}

/* ---- the sky ---- */
static struct {
    V3 moon, moon_up;          /* toward the moon; toward the sun across its face */
    float phase;               /* the moon's elongation from the sun, radians: 0 new, pi full */
    float lit;                 /* its illuminated fraction */
    int weather;               /* 0 clear, 1 rain, 2 snow, 3 mist */
    float cloud, towns[3];
} sky;

static double t_now;
static double sim_t;
static float door_now;
static double wall_now;        /* the wall clock, for the station clocks */
static float fire_door(double t);
static int fire_round(double t, double *age, double *len);
static double next_round(double t);
static float flick;            /* the fire's flicker, this frame */
#include "lineside.h"
#include "director.h"
#include "platform.h"

/* ---- lettering: the signs' and plates' texture, and the caption ---- */
static FT_Library ft;
static FT_Face face_serif, face_sans, face_sans_b;
static FT_Face load_face(const char *fam, const char *style) {
    char pat_s[256];
    snprintf(pat_s, sizeof pat_s, "%s:%s:fontformat=CFF,TrueType", fam, style);
    FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    FT_Face f;
    if (FT_New_Face(ft, (const char *)file, 0, &f)) die("can't load the font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
    return f;
}
static float text_w(FT_Face f, int px, const char *s, float track) {
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
    float w = 0;
    for (const char *p = s; *p; p++)
        if (!FT_Load_Char(f, (unsigned char)*p, FT_LOAD_DEFAULT)) w += (f->glyph->advance.x >> 6) + (p[1] ? track * px : 0);
    return w;
}
/* draw s from x (left) at baseline y into an 8-bit buffer bw x bh */
static void text_draw(unsigned char *buf, int bw, int bh, FT_Face f, int px, const char *s, float x, float base, float track) {
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
    for (const char *p = s; *p; p++) {
        if (FT_Load_Char(f, (unsigned char)*p, FT_LOAD_RENDER)) continue;
        FT_GlyphSlot g = f->glyph;
        int x0 = (int)lroundf(x) + g->bitmap_left, y0 = (int)lroundf(base) - g->bitmap_top;
        for (unsigned j = 0; j < g->bitmap.rows; j++)
            for (unsigned i = 0; i < g->bitmap.width; i++) {
                int X = x0 + (int)i, Y = y0 + (int)j;
                if (X < 0 || Y < 0 || X >= bw || Y >= bh) continue;
                unsigned v = g->bitmap.buffer[j * g->bitmap.pitch + i];
                unsigned char *d = buf + Y * bw + X;
                if (v > *d) *d = (unsigned char)v;
            }
        x += (g->advance.x >> 6) + track * px;
    }
}
/* the sign texture: rows of lettering, each centred across the full width */
#define SA_W 1024
#define SA_RH 128
#define SA_ROWS 8
static unsigned char sign_px[SA_W * SA_RH * SA_ROWS];
static GLuint tex_sign;
static void sign_row(int row, FT_Face f, const char *s, float track, float fill) {
    unsigned char *b = sign_px + row * SA_W * SA_RH;
    memset(b, 0, SA_W * SA_RH);
    int px = (int)(SA_RH * 0.62f);
    while (px > 8 && text_w(f, px, s, track) > SA_W * fill) px--;
    FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
    FT_Load_Char(f, 'H', FT_LOAD_DEFAULT);
    float cap = (float)(f->glyph->metrics.horiBearingY >> 6);
    text_draw(b, SA_W, SA_RH, f, px, s, SA_W / 2.0f - text_w(f, px, s, track) / 2, SA_RH / 2.0f + cap / 2, track);
}
static char host_uc[64];
static void signs_build(void) {
    int n = 0;
    for (const char *p = host; *p && n < 40; p++) host_uc[n++] = (char)toupper((unsigned char)*p);
    host_uc[n] = 0;
    char num[16];
    uint32_t h = 2166136261u;
    for (const char *p = host; *p; p++) h = (h ^ (unsigned char)*p) * 16777619u;
    snprintf(num, sizeof num, "%u", 4000 + h % 5999);
    sign_row(0, face_sans_b, num, 0.12f, 0.7f);
    sign_row(1, face_sans_b, host_uc, 0.14f, 0.9f);
    sign_row(2, face_serif, host_uc, 0.1f, 0.8f);
    sign_row(3, face_sans_b, host_uc, 0.16f, 0.9f);
    glGenTextures(1, &tex_sign);
    glBindTexture(GL_TEXTURE_2D, tex_sign);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, SA_W, SA_RH * SA_ROWS, 0, GL_RED, GL_UNSIGNED_BYTE, sign_px);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
/* row 4: the lineside board with the time away, redrawn when it changes */
static char away_str[32];
static void away_board(double wt) {
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    char s[32];
    snprintf(s, sizeof s, "AWAY %ld:%02ld", away / 3600, away / 60 % 60);
    if (!strcmp(s, away_str)) return;
    snprintf(away_str, sizeof away_str, "%s", s);
    sign_row(4, face_sans_b, s, 0.14f, 0.86f);
    glBindTexture(GL_TEXTURE_2D, tex_sign);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 4 * SA_RH, SA_W, SA_RH, GL_RED, GL_UNSIGNED_BYTE, sign_px + 4 * SA_W * SA_RH);
    glGenerateMipmap(GL_TEXTURE_2D);
}
/* the caption: the time, and how long the screen has been locked */
#define CAP_W 1024
#define CAP_H 192
static unsigned char cap_px[CAP_W * CAP_H];
static GLuint tex_cap;
static char cap_str[160];
static void caption_update(double wt) {
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    long away = (long)(wt - (double)locked_at);
    if (away < 0) away = 0;
    char t1[32], t2[128];
    strftime(t1, sizeof t1, "%H:%M", &tm);
    if (away < 60) snprintf(t2, sizeof t2, "%s", host_uc);
    else if (away < 3600) snprintf(t2, sizeof t2, "%s   AWAY %ld MIN", host_uc, away / 60);
    else snprintf(t2, sizeof t2, "%s   AWAY %ldH %02ldM", host_uc, away / 3600, away / 60 % 60);
    char all[160];
    snprintf(all, sizeof all, "%s|%s", t1, t2);
    if (!strcmp(all, cap_str)) return;
    snprintf(cap_str, sizeof cap_str, "%s", all);
    memset(cap_px, 0, sizeof cap_px);
    text_draw(cap_px, CAP_W, CAP_H, face_serif, 112, t1, 6, 112, 0.0f);
    text_draw(cap_px, CAP_W, CAP_H, face_sans, 30, t2, 10, 168, 0.22f);
    if (!tex_cap) {
        glGenTextures(1, &tex_cap);
        glBindTexture(GL_TEXTURE_2D, tex_cap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, tex_cap);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, CAP_W, CAP_H, 0, GL_RED, GL_UNSIGNED_BYTE, cap_px);
    glGenerateMipmap(GL_TEXTURE_2D);
}

/* a low-precision moon and sun (after Meeus) for the date, seen from about
 * latitude 52 N and the longitude the time zone suggests */
static void sky_init(double wt) {
    double d = wt / 86400.0 + 2440587.5 - 2451545.0;
    double R = M_PI / 180;
    double L = 218.316 + 13.176396 * d, Mm = 134.963 + 13.064993 * d, F = 93.272 + 13.229350 * d;
    double lon = (L + 6.289 * sin(Mm * R)) * R, lat = 5.128 * sin(F * R) * R;
    double Ms = 357.529 + 0.98560028 * d, Ls = 280.459 + 0.98564736 * d;
    double slon = (Ls + 1.915 * sin(Ms * R) + 0.020 * sin(2 * Ms * R)) * R;
    double e = 23.439 * R;
    double E = acos(cos(lat) * cos(lon - slon));
    sky.phase = (float)E;
    sky.lit = (float)((1 - cos(E)) / 2);
    time_t ts = (time_t)wt;
    struct tm tm;
    localtime_r(&ts, &tm);
    double olon = tm.tm_gmtoff / 3600.0 * 15 * R, olat = 52 * R;
    double gmst = (280.46061837 + 360.98564736629 * d) * R;
    V3 dirs[2];
    for (int k = 0; k < 2; k++) {
        double bl = k ? 0 : lat, ln = k ? slon : lon;
        double ra = atan2(sin(ln) * cos(e) - tan(bl) * sin(e), cos(ln));
        double dec = asin(sin(bl) * cos(e) + cos(bl) * sin(e) * sin(ln));
        double ha = gmst + olon - ra;
        double alt = asin(sin(olat) * sin(dec) + cos(olat) * cos(dec) * cos(ha));
        double az = atan2(-sin(ha), tan(dec) * cos(olat) - sin(olat) * cos(ha));
        if (k == 0) alt = fmax(fmin(alt, 50 * R), 22 * R);   /* lifted to be seen */
        /* the line runs north along +z, east is -x */
        dirs[k] = v3((float)(-cos(alt) * sin(az)), (float)sin(alt), (float)(cos(alt) * cos(az)));
    }
    sky.moon = vnorm(dirs[0]);
    V3 toward = vsub(dirs[1], vmul(sky.moon, vdot(dirs[1], sky.moon)));
    sky.moon_up = vlen(toward) > 1e-4f ? vnorm(toward) : v3(0, -1, 0);
    uint32_t h = opt_seed * 2654435761u;
    h ^= h >> 15;
    sky.weather = opt_weather >= 0 ? opt_weather : (h % 100 < 55 ? 0 : h % 100 < 75 ? 1 : h % 100 < 87 ? 3 : 2);
    sky.cloud = sky.weather == 0 ? 0.2f + 0.3f * ((h >> 8) % 100) / 100.0f : sky.weather == 3 ? 0.45f : 0.92f;
    sky.towns[0] = (float)((h >> 12) % 628) / 100.0f;
    sky.towns[1] = sky.towns[0] + 2.0f + (float)((h >> 20) % 200) / 100.0f;
    sky.towns[2] = 1.0f;
}

/* ---- the simulation ---- */
static double last_read;

/* the fireman fires a round now and then (more often when she's worked
 * hard): the firehole doors open for a few seconds and the cab fills with
 * the fire's light. A function of time alone, so every output agrees. */
static uint32_t thash(long n, int salt) {
    uint32_t h = (uint32_t)n * 0x9e3779b9u ^ (uint32_t)salt * 0x85ebca6bu ^ opt_seed * 0xc2b2ae35u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return h;
}
/* the round in progress at t: its age (s) and length, or 0 */
static double force_round = -1;        /* --fire: a round starting then (for testing) */
static int fire_round(double t, double *age, double *len) {
    if (force_round >= 0 && t >= force_round && t <= force_round + 7) { *age = t - force_round; *len = 7; return 1; }
    for (long k = (long)floor(t / 28) - 1; k <= (long)floor(t / 28); k++) {
        float a = (thash(k, 31) & 0xffff) / 65536.0f, b = (thash(k, 32) & 0xffff) / 65536.0f;
        if (b > 0.45f + 0.5f * tr.load_s) continue;
        double t0 = k * 28.0 + a * 20, l = 4.5 + 3 * b;
        if (t >= t0 && t <= t0 + l) { *age = t - t0; *len = l; return 1; }
    }
    return 0;
}
/* when the next round starts after t (within a minute), or 0 */
static double next_round(double t) {
    for (long k = (long)floor(t / 28); k <= (long)floor(t / 28) + 2; k++) {
        float a = (thash(k, 31) & 0xffff) / 65536.0f, b = (thash(k, 32) & 0xffff) / 65536.0f;
        if (b > 0.45f + 0.5f * tr.load_s) continue;
        double t0 = k * 28.0 + a * 20;
        if (t0 > t) return t0;
    }
    return 0;
}
static float fire_door(double t) {
    float best = 0;
    if (force_round >= 0 && t >= force_round && t <= force_round + 7) {
        float u = (float)(t - force_round);
        best = sstepf(0, 0.5f, u) * sstepf(7, 6.4f, u) * (0.75f + 0.25f * cosf(u * 2 * (float)M_PI / 1.3f));
    }
    for (long k = (long)floor(t / 28) - 1; k <= (long)floor(t / 28); k++) {
        float a = (thash(k, 31) & 0xffff) / 65536.0f, b = (thash(k, 32) & 0xffff) / 65536.0f;
        if (b > 0.45f + 0.5f * tr.load_s) continue;
        double t0 = k * 28.0 + a * 20, len = 4.5 + 3 * b;
        double u = t - t0;
        if (u < 0 || u > len) continue;
        /* open, a shovelful at a time (the doors close a little between), then shut */
        float o = sstepf(0, 0.5f, (float)u) * sstepf((float)len, (float)len - 0.6f, (float)u);
        o *= 0.75f + 0.25f * cosf((float)u * 2 * (float)M_PI / 1.3f);
        if (o > best) best = o;
    }
    return best;
}
static int sim_started;
static void physics(double dt) {
    float load = clampf(getenv("STEAM_LOAD") ? (float)atof(getenv("STEAM_LOAD")) : sens.load, 0, 1);
    tr.load_s += (load - tr.load_s) * (float)(dt / 4.0);
    tr.temp_s += (sens.temp - tr.temp_s) * (float)(dt / 6.0);
    /* she runs between 34 and 90 mph by how hard the CPU works */
    tr.vt = 15 + 25 * tr.load_s;
    double acc = clampf((float)(tr.vt - tr.v) / 6.0f, -0.6f, 0.45f);
    tr.v += acc * dt;
    tr.z += tr.v * dt;
    tr.wa[0] += tr.v / DRV_R * dt;
    tr.wa[1] += tr.v / 0.48 * dt;
    tr.wa[2] += tr.v / 0.56 * dt;
    tr.wa[3] += tr.v / 0.63 * dt;
    tr.beats += tr.v / (2 * M_PI * DRV_R) * 4 * dt;
    /* the driver notches up as she gathers speed, and opens out climbing */
    float ct = clampf(0.62f - (float)(tr.v - 15) / 25 * 0.4f + 0.25f * (float)(acc > 0.1 ? acc : 0) + (float)track_dy(tr.z) * 8, 0.15f, 0.75f);
    tr.cutoff += (ct - tr.cutoff) * (float)(dt / 3.0);
    float ft = clampf((tr.temp_s - 35) / 50, 0, 1);
    tr.fire += (0.45f + 0.55f * ft - tr.fire) * (float)(dt / 5.0);
    float mf = sens.mem_total_gb > 0 ? sens.mem_gb / sens.mem_total_gb : 0.4f;
    tr.ncoach_want = 3 + (int)lroundf(clampf(mf, 0, 1) * 6);
    if (tr.ncoach_want > MAXCOACH) tr.ncoach_want = MAXCOACH;
    smoke_step(dt);
}
static void sim_init(void) {
    tr.load_s = clampf(sens.load, 0, 1);
    tr.temp_s = sens.temp > 0 ? sens.temp : 50;
    tr.v = tr.vt = 15 + 25 * tr.load_s;
    tr.cutoff = 0.3f;
    tr.fire = 0.7f;
    float mf = sens.mem_total_gb > 0 ? sens.mem_gb / sens.mem_total_gb : 0.4f;
    tr.ncoach = tr.ncoach_want = 3 + (int)lroundf(clampf(mf, 0, 1) * 6);
    if (tr.ncoach > MAXCOACH) tr.ncoach = tr.ncoach_want = MAXCOACH;
    tr.lit = 0.85f;
}
/* advance to t seconds after the epoch; read the sensors once a second (on
 * the same wall-clock second as the other instances, when spanning) */
static void advance(double t, double wt) {
    if (span_me >= 0 ? floor(wt) != last_read : t - last_read >= 1.0 || !sim_started) {
        read_sensors();
        last_read = span_me >= 0 ? floor(wt) : t;
    }
    wall_now = wt;
    const double h = 1.0 / 120;
    if (!sim_started) {
        /* the train is where a steady run from the epoch would have put it,
         * with the last half minute run for real (so the smoke's there) */
        sim_started = 1;
        double t0 = getenv("STEAM_FULLSIM") ? 0 : floor(fmax(t - 30, 0) / 10) * 10;   /* on a 10 s grid, so outputs started together agree */
        sim_t = t0;
        tr.z = 600 + 20 * t0;
        director_init(t0);
    }
    if (t - sim_t > 30) {
        tr.z += 20 * (t - 30 - sim_t);
        sim_t = t - 30;
    }
    while (sim_t + h <= t) {
        sim_t += h;
        physics(h);
        director_update(sim_t);
    }
}

/* ---- drawing ---- */
static GLuint p_obj, p_ter, p_tree, p_sky, p_down, p_up, p_post, p_shadow, p_tree_pre, p_ter_pre, p_wire, p_rain;
static float cam_vel[3];
static GLuint vao_full;

/* the canvas (see setup_view), and this output's slice of it */
static float cam_cx, cam_cy, cam_f, view_x, view_y, view_w, view_h, lay_x0, lay_y0, lay_w, lay_h, front_w, front_h;
static void setup_view(void) {
    if (span_me < 0) {
        view_x = view_y = 0; view_w = (float)W; view_h = (float)H;
        cam_cx = W / 2.0f; cam_cy = H / 2.0f;
        lay_x0 = lay_y0 = 0; lay_w = (float)W; lay_h = (float)H;
        front_w = (float)W; front_h = (float)H;
    } else {
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
        view_x = span[span_me][0]; view_y = span[span_me][1];
        view_w = span[span_me][2]; view_h = span[span_me][3];
    }
    /* the lens is set for a 16:9 frame as wide as the front output allows */
    float fw = fminf(front_w, front_h * 16 / 9.0f);
    cam_f = fw / 2 / tanf(cam.hfov / 2 * (float)M_PI / 180);
}

static M4 VP;
static float view_tan[4], rot3[9];
static void setup_matrices(void) {
    float l = (view_x - cam_cx) / cam_f, r = (view_x + view_w - cam_cx) / cam_f;
    float b = (cam_cy - view_y - view_h) / cam_f, t = (cam_cy - view_y) / cam_f;
    view_tan[0] = l; view_tan[1] = b; view_tan[2] = r - l; view_tan[3] = t - b;
    M4 V = m4id();
    V3 B = vmul(cam.F, -1);
    V.m[0] = cam.R.x; V.m[4] = cam.R.y; V.m[8] = cam.R.z;
    V.m[1] = cam.U.x; V.m[5] = cam.U.y; V.m[9] = cam.U.z;
    V.m[2] = B.x; V.m[6] = B.y; V.m[10] = B.z;
    /* camera -> world, column-major, for the sky */
    rot3[0] = cam.R.x; rot3[1] = cam.R.y; rot3[2] = cam.R.z;
    rot3[3] = cam.U.x; rot3[4] = cam.U.y; rot3[5] = cam.U.z;
    rot3[6] = B.x; rot3[7] = B.y; rot3[8] = B.z;
    float n = 0.2f, fa = 16000;
    M4 P = {{0}};
    P.m[0] = 2 / (r - l); P.m[8] = (r + l) / (r - l);
    P.m[5] = 2 / (t - b); P.m[9] = (t + b) / (t - b);
    P.m[10] = -(fa + n) / (fa - n); P.m[11] = -1; P.m[14] = -2 * fa * n / (fa - n);
    VP = m4mul(P, V);
}

/* a frame's model matrix, camera-relative */
static M4 frame_model(const Frame *f) {
    return m4basis(f->X, f->Y, f->Z, v3((float)(f->o[0] - cam.p[0]), (float)(f->o[1] - cam.p[1]), (float)(f->o[2] - cam.p[2])));
}

/* the point lights for this frame */
#define MAXPL 16
static float pl[MAXPL][4], plc[MAXPL][4];
static int npl;
static void add_light(const double *wp, float rad, float r, float g, float b) {
    if (npl >= MAXPL) return;
    float x = (float)(wp[0] - cam.p[0]), y = (float)(wp[1] - cam.p[1]), z = (float)(wp[2] - cam.p[2]);
    if (x * x + y * y + z * z > 600.0f * 600.0f) return;
    pl[npl][0] = x; pl[npl][1] = y; pl[npl][2] = z;
    pl[npl][3] = rad;
    plc[npl][0] = r; plc[npl][1] = g; plc[npl][2] = b; plc[npl][3] = 0;
    npl++;
}

static float car_u[12][4], car_d[12][4], car_box[4];
/* the point lights that reach a sphere, into the uniforms (per draw) */
static float pl_sel[MAXPL][4], plc_sel[MAXPL][4];
static void lights_for(GLuint p, float x, float y, float z, float r) {
    int n = 0;
    for (int i = 0; i < npl; i++) {
        float dx = pl[i][0] - x, dy = pl[i][1] - y, dz = pl[i][2] - z;
        float reach = pl[i][3] * 8 + r;
        if (dx * dx + dy * dy + dz * dz > reach * reach) continue;
        memcpy(pl_sel[n], pl[i], sizeof pl[i]);
        memcpy(plc_sel[n], plc[i], sizeof plc[i]);
        n++;
    }
    glUniform4fv(UL(p, "uPL"), MAXPL, &pl_sel[0][0]);
    glUniform4fv(UL(p, "uPLC"), MAXPL, &plc_sel[0][0]);
    glUniform1i(UL(p, "uNPL"), n);
}
static int ncar_u;
static M4 shadowM;
static float fog_col[3], fog_den, mist_y, moon_col[3], sky_col[3], gnd_col[3];

static void set_common(GLuint p) {
    glUseProgram(p);
    glUniform3f(UL(p, "uMoonDir"), sky.moon.x, sky.moon.y, sky.moon.z);
    glUniform3fv(UL(p, "uMoonCol"), 1, moon_col);
    glUniform3fv(UL(p, "uSkyCol"), 1, sky_col);
    glUniform3fv(UL(p, "uGndCol"), 1, gnd_col);
    glUniform3fv(UL(p, "uFogCol"), 1, fog_col);
    glUniform1f(UL(p, "uFogDen"), fog_den);
    glUniform1f(UL(p, "uMistY"), mist_y);
    glUniform1f(UL(p, "uMistDen"), sky.weather == 3 ? 0.006f : 0.0018f);
    glUniform1f(UL(p, "uTime"), (float)fmod(t_now, 1000.0));
    glUniform3f(UL(p, "uWOff"), (float)fmod(cam.p[0], 4096.0), (float)cam.p[1], (float)fmod(cam.p[2], 4096.0));
    glUniform4fv(UL(p, "uPL"), MAXPL, &pl[0][0]);
    glUniform4fv(UL(p, "uPLC"), MAXPL, &plc[0][0]);
    glUniform1i(UL(p, "uNPL"), npl);
    glUniform4fv(UL(p, "uCar"), 12, &car_u[0][0]);
    glUniform4fv(UL(p, "uCarD"), 12, &car_d[0][0]);
    glUniform1i(UL(p, "uNCar"), ncar_u);
    glUniform4fv(UL(p, "uCarBox"), 1, car_box);
    glUniform2f(UL(p, "uWeather"), sky.weather == 1 ? 1.0f : 0.0f, sky.weather == 2 ? 1.0f : 0.0f);
    glUniform1i(UL(p, "uShadow"), 4);
    glUniformMatrix4fv(UL(p, "uShadowM"), 1, GL_FALSE, shadowM.m);
    glUniformMatrix4fv(UL(p, "uVP"), 1, GL_FALSE, VP.m);
}

/* ---- render targets ---- */
static int scene_w, scene_h;
static GLuint fbo_scene, tex_scene, tex_depth;
#define NBLOOM 6
static GLuint fbo_bloom[NBLOOM], tex_bloom[NBLOOM];
static int bloom_w[NBLOOM], bloom_h[NBLOOM];
static GLuint fbo_shadow, tex_shadow;
#define SHADOW_N 2048
static void targets(int sw, int sh) {
    if (sw == scene_w && sh == scene_h) return;
    if (scene_w) {
        glDeleteFramebuffers(1, &fbo_scene); glDeleteTextures(1, &tex_scene); glDeleteTextures(1, &tex_depth);
        glDeleteFramebuffers(NBLOOM, fbo_bloom); glDeleteTextures(NBLOOM, tex_bloom);
        smoke_targets_free();
    }
    scene_w = sw; scene_h = sh;
    tex_scene = mk_tex(sw, sh, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_LINEAR);
    tex_depth = mk_tex(sw, sh, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_NEAREST);
    fbo_scene = mk_fbo(tex_scene, tex_depth);
    int bw = sw, bh = sh;
    for (int i = 0; i < NBLOOM; i++) {
        bw = (bw + 1) / 2; bh = (bh + 1) / 2;
        bloom_w[i] = bw; bloom_h[i] = bh;
        tex_bloom[i] = mk_tex(bw, bh, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_LINEAR);
        fbo_bloom[i] = mk_fbo(tex_bloom[i], 0);
    }
    smoke_targets(sw, sh);
}

static void fullscreen(void) {
    glBindVertexArray(vao_full);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
static void fullscreen_tri(void) { fullscreen(); }
static char fs_buf[1 << 17];
#define FSRC(a, b) (snprintf(fs_buf, sizeof fs_buf, "%s%s%s%s", HDR, GL_COMMON, a, b), fs_buf)
#include "smoke.h"

static int tile_visible(const Tile *t) {
    double size = TROOT / (1 << t->level);
    float hx = (float)(size / 2), hy = (t->hmax - t->hmin) / 2 + 15;
    return visible((float)(t->ox + size / 2 - cam.p[0]), (float)((t->hmax + t->hmin) / 2 - cam.p[1]), (float)(t->oz + size / 2 - cam.p[2]), sqrtf(2 * hx * hx + hy * hy));
}
#include "trees.h"

/* the valve gear and rods for one side (s = +1 / -1), the engine's frame f */
static void draw_motion(GLuint p, const Frame *f, int s) {
    M4 base = frame_model(f);
    /* this side's crank leads the other's by a quarter turn */
    float th = (float)tr.wa[0] + (s > 0 ? 0 : (float)M_PI / 2);
    float x = s * 0.93f;
    float cz = CRANK_R * cosf(th), cy = DRV_R + CRANK_R * sinf(th);
    /* the crosshead, driven by the connecting rod from the middle pin */
    float dy = CYL_Y - cy;
    float xz = cz + sqrtf(CON_L * CON_L - dy * dy);
    /* the return crank and eccentric rod, the expansion link, radius rod, combination lever */
    float ea = th - (float)M_PI / 2;
    float ez = 0.21f * cosf(ea), ey = DRV_R + 0.21f * sinf(ea);
    float kz = 1.6f, ky = 1.62f;
    float lo = -0.8f, hi = 0.8f;
    for (int i = 0; i < 24; i++) {
        float a = (lo + hi) / 2;
        float fz = kz + 0.3f * sinf(a), fy = ky - 0.3f * cosf(a);
        float d = sqrtf((fz - ez) * (fz - ez) + (fy - ey) * (fy - ey));
        if (d > 1.55f) hi = a; else lo = a;
    }
    float la = (lo + hi) / 2;
    float fz = kz + 0.3f * sinf(la), fy = ky - 0.3f * cosf(la);
    float die = 0.06f + (tr.cutoff - 0.15f) / 0.6f * 0.2f;
    float dz = kz - die * sinf(la), dyy = ky + die * cosf(la);
    float tz = dz + sqrtf(2.3f * 2.3f - (1.62f - dyy) * (1.62f - dyy)), ty = 1.62f;
    float uz = xz + 0.35f, uy = 0.78f;
    struct { float az, ay, bz, by, xo; GMesh *m; } parts[] = {
        {cz + drv_z[0], cy, cz + drv_z[2], cy, x + s * 0.05f, &g_rod},
        {cz, cy, xz, CYL_Y, x + s * 0.12f, &g_rod},
        {ez, ey, fz, fy, x + s * 0.2f, &g_link},
        {kz + 0.3f * sinf(la), ky - 0.3f * cosf(la), kz - 0.32f * sinf(la), ky + 0.32f * cosf(la), x + s * 0.16f, &g_rod},
        {dz, dyy, tz, ty, x + s * 0.14f, &g_link},
        {tz, ty + 0.05f, uz, uy, x + s * 0.1f, &g_link},
        {uz, uy, xz, CYL_Y - 0.1f, x + s * 0.1f, &g_link},
    };
    for (size_t i = 0; i < sizeof parts / sizeof *parts; i++) {
        V3 a = v3(parts[i].xo, parts[i].ay, parts[i].az), b = v3(parts[i].xo, parts[i].by, parts[i].bz);
        V3 Z = vsub(b, a);
        float L = vlen(Z);
        Z = vmul(Z, 1 / L);
        V3 X = v3(1, 0, 0), Y = vcross(Z, X);
        M4 m = m4mul(base, m4basis(X, Y, vmul(Z, L), a));
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
        gm_draw(parts[i].m);
    }
    M4 m = m4mul(base, m4trans(x + s * 0.12f, CYL_Y, xz));
    glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
    gm_draw(&g_xhead);
    for (int i = 0; i < 3; i++) {
        M4 pm = m4mul(base, m4mul(m4trans(s * 0.8f, cy, cz + drv_z[i]), m4scale((float)s, 1, 1)));
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, pm.m);
        gm_draw(&g_pin);
    }
}

/* the footplate's moving parts: the firehole doors, the needles, the water in the glasses, the shovel */
static void draw_part(GLuint p, M4 base, M4 local, GMesh *g) {
    M4 m = m4mul(base, local);
    glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
    gm_draw(g);
}
static void draw_footplate(GLuint p, const Frame *f) {
    M4 base = frame_model(f);
    float zb = -3.27f;
    /* the doors slide apart */
    float o = door_now;
    for (int s = -1; s <= 1; s += 2)
        draw_part(p, base, m4trans(s * (0.15f + 0.26f * o), 2.45f, zb - 0.1f), &g_door);
    /* boiler pressure: the CPU's load, trembling; the steam chest: its heat (the dials are seen mirrored from the cab) */
    float tremble = 0.015f * sinf((float)t_now * 23) + 0.01f * sinf((float)t_now * 37);
    float a1 = (-2.356f + 4.712f * clampf(tr.load_s * 0.9f + 0.05f + tremble, 0, 1));
    float a2 = (-2.356f + 4.712f * clampf((tr.temp_s - 20) / 80, 0, 1));
    draw_part(p, base, m4mul(m4mul(m4trans(0.0f, 3.32f, zb - 0.055f), m4rotz(a1)), m4scale(0.012f / 0.04f * 0.4f, 0.11f, 1)), &g_needle);
    draw_part(p, base, m4mul(m4mul(m4trans(-0.4f, 3.22f, zb - 0.055f), m4rotz(a2)), m4scale(0.3f, 0.075f, 1)), &g_needle);
    /* the water in the glasses: the battery's charge (or a comfortable half) */
    float lvl = aux_kind == AUX_BATT ? clampf(sens.aux / 100, 0.04f, 1) : 0.6f;
    lvl += 0.01f * sinf((float)t_now * 5.3f);
    for (int s = -1; s <= 1; s += 2)
        draw_part(p, base, m4mul(m4trans(s * 0.45f, 2.755f, zb - 0.06f), m4scale(1, 0.33f * lvl, 1)), &g_water);
    /* the shovel: a round in progress swings coal from the tender into the fire; between rounds it leans on the tender */
    double age, len;
    V3 blade, aim;
    if (fire_round(t_now, &age, &len)) {
        float c = (float)fmod(age + 0.3, 1.3) / 1.3f;
        V3 mouth = v3(0, 2.42f, -3.45f), in = v3(0, 2.42f, -3.05f), coal = v3(0.15f, 2.05f, -6.15f), lift = v3(-0.15f, 2.55f, -4.7f);
        if (c < 0.15f) blade = vlerp(mouth, in, sstepf(0, 0.15f, c));
        else if (c < 0.3f) blade = vlerp(in, mouth, sstepf(0.15f, 0.3f, c));
        else if (c < 0.55f) blade = vlerp(mouth, lift, sstepf(0.3f, 0.55f, c));
        else if (c < 0.65f) blade = vlerp(lift, coal, sstepf(0.55f, 0.65f, c));
        else if (c < 0.8f) blade = vlerp(coal, lift, sstepf(0.65f, 0.8f, c));
        else blade = vlerp(lift, mouth, sstepf(0.8f, 1.0f, c));
        if (age > len - 0.8) blade = vlerp(blade, v3(-0.5f, 2.6f, -5.6f), sstepf((float)len - 0.8f, (float)len, (float)age));
        /* the fireman's hands follow behind it */
        aim = vnorm(vsub(blade, v3(-0.35f, 3.0f, -5.25f)));
    } else {
        blade = v3(-0.55f, 1.95f, -6.2f);
        aim = vnorm(v3(0.05f, -0.9f, -0.35f));
    }
    V3 Z = vmul(aim, -1), X = vnorm(vcross(v3(0, 1, 0), Z)), Y = vcross(Z, X);
    draw_part(p, base, m4basis(X, Y, Z, blade), &g_shovel);
}

static void draw_train(GLuint p, int wheels) {
    glUniform1i(UL(p, "uTex"), 3);
    float wa[4], ww[4];
    static const float rr[4] = {DRV_R, 0.48f, 0.56f, 0.63f};
    for (int i = 0; i < 4; i++) {
        wa[i] = (float)fmod(tr.wa[i], 2 * M_PI);
        ww[i] = (float)(tr.v / rr[i]);
    }
    glUniform4fv(UL(p, "uWheelA4"), 1, wa);
    glUniform4fv(UL(p, "uWheelW4"), 1, ww);
    int nveh = 2 + tr.ncoach;
    for (int i = 0; i < nveh; i++) {
        Frame f = veh_frame(i);
        double d2 = (f.o[0] - cam.p[0]) * (f.o[0] - cam.p[0]) + (f.o[2] - cam.p[2]) * (f.o[2] - cam.p[2]);
        if (d2 > 3000.0 * 3000.0) continue;
        double c[3];
        fpt(&f, 0, 2.0f, i == 0 ? 1.0f : 0.0f, c);
        if (!visible((float)(c[0] - cam.p[0]), (float)(c[1] - cam.p[1]), (float)(c[2] - cam.p[2]), i == 0 ? 8.0f : i == 1 ? 5.0f : 11.0f)) continue;
        M4 m = frame_model(&f);
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
        glUniform4f(UL(p, "uObj"), 1.0f, tr.lit, (float)i * 1.37f, tr.fire * (0.85f + 0.3f * flick) * (0.4f + 1.6f * door_now));
        if (wheels) { gm_draw(i == 0 ? &g_loco_w : i == 1 ? &g_tender_w : &g_coach_w); continue; }
        gm_draw(i == 0 ? &g_loco : i == 1 ? &g_tender : &g_coach);
        if (i == 0 && d2 < 400.0 * 400.0) { draw_motion(p, &f, 1); draw_motion(p, &f, -1); }
        if (i == 0 && d2 < 40.0 * 40.0) draw_footplate(p, &f);
    }
}

/* the moon's shadow map, round what the camera's looking at */
static void render_shadow(void) {
    float fd = fminf(cam.focus, 45);
    V3 c = vmul(cam.F, fd);
    float R = 55;
    V3 Z = sky.moon;
    V3 X = vnorm(vcross(v3(0, 1, 0), Z));
    V3 Y = vcross(Z, X);
    /* snap the centre to the map's texels so edges don't crawl */
    float texel = 2 * R / SHADOW_N;
    V3 wc = vadd(c, v3((float)fmod(cam.p[0], 1024.0), (float)cam.p[1], (float)fmod(cam.p[2], 1024.0)));
    float cx = vdot(c, X), cy = vdot(c, Y), cz = vdot(c, Z);
    float wx = vdot(wc, X), wy = vdot(wc, Y);
    cx -= wx - floorf(wx / texel) * texel;
    cy -= wy - floorf(wy / texel) * texel;
    M4 V = m4id();
    V.m[0] = X.x; V.m[4] = X.y; V.m[8] = X.z;
    V.m[1] = Y.x; V.m[5] = Y.y; V.m[9] = Y.z;
    V.m[2] = Z.x; V.m[6] = Z.y; V.m[10] = Z.z;
    V.m[12] = -cx; V.m[13] = -cy; V.m[14] = -cz;
    float d = 200;
    M4 P = m4id();
    P.m[0] = 1 / R; P.m[5] = 1 / R; P.m[10] = -1 / d;
    shadowM = m4mul(P, V);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, SHADOW_N, SHADOW_N);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(p_shadow);
    glUniformMatrix4fv(UL(p_shadow, "uVP"), 1, GL_FALSE, shadowM.m);
    glUniform4fv(UL(p_shadow, "uPal2"), 64, &pal2[0][0]);
    glUniform1f(UL(p_shadow, "uPre"), 0);
    cull_on = 0;
    draw_train(p_shadow, 0);
    draw_train(p_shadow, 1);
    lineside_draw(p_shadow, 1);
    glUseProgram(p_tree_pre);
    glUniformMatrix4fv(UL(p_tree_pre, "uVP"), 1, GL_FALSE, shadowM.m);
    glUniform1f(UL(p_tree_pre, "uTime"), (float)fmod(t_now, 1000.0));
    trees_near_only = 1;
    trees_focus[0] = c.x; trees_focus[1] = c.y; trees_focus[2] = c.z;
    draw_trees(p_tree_pre);
    trees_near_only = 0;
    cull_on = 1;
}

static float render_scale = 1;
static int frame_no;
/* --bench: GPU time per pass */
enum { PR_SHADOW, PR_PRE, PR_OBJ, PR_LINE, PR_TREE, PR_TER, PR_SKY, PR_RESOLVE, PR_SMOKE, PR_BLOOM, PR_POST, PR_N };
static const char *prof_name[PR_N] = {"shadow", "prepass", "train", "lineside", "trees", "terrain", "sky", "resolve", "smoke", "bloom", "post"};
static GLuint prof_q[PR_N + 1];
static double prof_ms[PR_N], prof_cpu;
static int prof_on;
static void prof_mark(int i) { if (prof_on) glQueryCounter(prof_q[i], GL_TIMESTAMP); }
static void prof_collect(void) {
    if (!prof_on) return;
    GLuint64 t[PR_N + 1];
    for (int i = 0; i <= PR_N; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < PR_N; i++) prof_ms[i] += (t[i + 1] - t[i]) / 1e6;
}
static void render(GLuint target) {
    t_now = sim_t;
    {
        /* the camera's velocity, for the rain's streaks */
        director_camera(sim_t - 1.0 / 60);
        double p0[3] = {cam.p[0], cam.p[1], cam.p[2]};
        director_camera(sim_t);
        for (int k = 0; k < 3; k++) cam_vel[k] = (float)((cam.p[k] - p0[k]) * 60);
    }
    setup_view();
    setup_matrices();
    frustum_from(&VP);
    int sw = (int)(W * render_scale + 0.5f), sh = (int)(H * render_scale + 0.5f);
    targets(sw, sh);
    tiles_update(cam.p[0], cam.p[2], 9000, opt_shot || frame_no < 2);
    lineside_update();
    frame_no++;
    if (tr.ncoach != tr.ncoach_want && director_offscreen_tail()) tr.ncoach = tr.ncoach_want;

    /* light and air */
    float ml = 0.3f + 0.7f * sky.lit;
    float cl = sky.weather == 0 ? 1.0f : 0.5f;
    moon_col[0] = 0.42f * ml * cl; moon_col[1] = 0.48f * ml * cl; moon_col[2] = 0.62f * ml * cl;
    sky_col[0] = 0.005f + 0.006f * ml; sky_col[1] = 0.007f + 0.008f * ml; sky_col[2] = 0.012f + 0.016f * ml;
    gnd_col[0] = 0.002f; gnd_col[1] = 0.0022f; gnd_col[2] = 0.0025f;
    fog_col[0] = 0.006f + 0.008f * ml; fog_col[1] = 0.008f + 0.010f * ml; fog_col[2] = 0.013f + 0.017f * ml;
    fog_den = sky.weather == 3 ? 1 / 700.0f : sky.weather == 1 ? 1 / 1300.0f : sky.weather == 2 ? 1 / 1000.0f : 1 / 3000.0f;
    mist_y = (float)(-22 - cam.p[1]);
    if (lineside_in_tunnel(cam.p[2]) && fabs(cam.p[0] - track_x(cam.p[2])) < 6 && cam.p[1] < track_y(cam.p[2]) + 6) {
        /* underground: no moon, no sky; only what the train brings */
        for (int k = 0; k < 3; k++) { moon_col[k] *= 0.01f; sky_col[k] *= 0.08f; fog_col[k] *= 0.1f; }
    }
    flick = 0.6f * sinf((float)t_now * 7.3f) * sinf((float)t_now * 3.1f + 1) + 0.4f * sinf((float)t_now * 13.7f + 2);

    /* lights: the fire under the engine and in the cab, the lamps; the carriages' windows */
    npl = 0;
    Frame f0 = veh_frame(0);
    double wp[3];
    float fi = tr.fire * (0.85f + 0.15f * flick);
    fpt(&f0, 0, 0.45f, -3.3f, wp); add_light(wp, 1.0f, 2.6f * fi, 0.8f * fi, 0.18f * fi);
    float door = fire_door(t_now);
    door_now = door;
    float ci = fi * (0.6f + 3.2f * door * (0.85f + 0.3f * flick));
    fpt(&f0, 0, 2.6f, -3.7f, wp); add_light(wp, 0.9f + 1.0f * door, 2.2f * ci, 0.7f * ci, 0.15f * ci);
    for (int s = -1; s <= 1; s += 2) { fpt(&f0, s * 0.87f, 1.48f, 7.7f, wp); add_light(wp, 1.0f, 0.8f, 0.68f, 0.45f); }
    /* the gauge lamp in the cab */
    fpt(&f0, 0.38f, 3.6f, -3.42f, wp); add_light(wp, 0.35f, 0.5f, 0.32f, 0.15f);
    {
        Frame ft = veh_frame(1 + tr.ncoach);
        fpt(&ft, 0.6f, 1.6f, -COACH_L / 2 - 0.4f, wp); add_light(wp, 0.6f, 0.6f, 0.03f, 0.01f);
    }
    lineside_lights();
    ncar_u = 0;
    {
        Frame fa = veh_frame(2), fb = veh_frame(1 + tr.ncoach);
        car_box[0] = (float)((fa.o[0] + fb.o[0]) / 2 - cam.p[0]); car_box[1] = (float)((fa.o[1] + fb.o[1]) / 2 - cam.p[1]);
        car_box[2] = (float)((fa.o[2] + fb.o[2]) / 2 - cam.p[2]);
        car_box[3] = (float)(fabs(fa.o[2] - fb.o[2]) / 2 + COACH_L / 2 + 20);
    }
    for (int i = 2; i < 2 + tr.ncoach && ncar_u < 12; i++) {
        Frame f = veh_frame(i);
        car_u[ncar_u][0] = (float)(f.o[0] - cam.p[0]); car_u[ncar_u][1] = (float)(f.o[1] - cam.p[1] + 2.5);
        car_u[ncar_u][2] = (float)(f.o[2] - cam.p[2]); car_u[ncar_u][3] = COACH_L / 2;
        V3 fz = vnorm(v3(f.Z.x, 0, f.Z.z));
        car_d[ncar_u][0] = fz.x; car_d[ncar_u][1] = fz.z; car_d[ncar_u][2] = tr.lit * (lineside_in_tunnel(f.o[2]) ? 0.3f : 1.0f); car_d[ncar_u][3] = 0;
        ncar_u++;
    }

    prof_mark(PR_SHADOW);
    render_shadow();
    prof_mark(PR_PRE);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_sign);
    glActiveTexture(GL_TEXTURE0);

    /* depth first, cheaply, so the costly shading runs once a pixel (the
     * wheels, blurred and partly see-through, are left to the main pass) */
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glUseProgram(p_shadow);
    glUniformMatrix4fv(UL(p_shadow, "uVP"), 1, GL_FALSE, VP.m);
    glUniform1f(UL(p_shadow, "uPre"), 1);
    draw_train(p_shadow, 0);
    lineside_draw(p_shadow, 0);
    glUseProgram(p_tree_pre);
    glUniformMatrix4fv(UL(p_tree_pre, "uVP"), 1, GL_FALSE, VP.m);
    glUniform1f(UL(p_tree_pre, "uTime"), (float)fmod(t_now, 1000.0));
    draw_trees(p_tree_pre);
    glUseProgram(p_ter_pre);
    glUniformMatrix4fv(UL(p_ter_pre, "uVP"), 1, GL_FALSE, VP.m);
    for (int i = 0; i < ndraw; i++) {
        Tile *t = drawlist[i];
        if (!tile_visible(t)) continue;
        glUniform3f(UL(p_ter_pre, "uTileO"), (float)(t->ox - cam.p[0]), (float)(-cam.p[1]), (float)(t->oz - cam.p[2]));
        gm_draw(&t->mesh);
    }
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    /* the train and the lineside */
    prof_mark(PR_OBJ);
    set_common(p_obj);
    glUniform4fv(UL(p_obj, "uPal"), 64, &pal[0][0]);
    glUniform4fv(UL(p_obj, "uPal2"), 64, &pal2[0][0]);
    draw_train(p_obj, 0);
    prof_mark(PR_LINE);
    lineside_draw(p_obj, 0);
    /* the wheels' spoked centres, blurred by their turning: blended over */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    draw_train(p_obj, 1);
    set_common(p_wire);
    glUniform1f(UL(p_wire, "uPixAng"), 1.0f / (cam_f * render_scale));
    lineside_wires(p_wire);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    prof_mark(PR_TREE);
    set_common(p_tree);
    draw_trees(p_tree);

    prof_mark(PR_TER);
    set_common(p_ter);
    {
        double tz = cam.p[2];
        float t4[4] = {(float)(track_x(tz) - cam.p[0]), 0, (float)track_dx(tz), (float)(track_y(tz) - cam.p[1])};
        Feat cr = feat_near(tz, F_CROSSING);
        Feat vd = feat_near(tz, F_VIADUCT);
        float t5[4] = {(float)track_ddx(tz), (float)track_dy(tz), cr.on ? (float)((cr.z0 + cr.z1) / 2 - tz) : 1e6f, 0};
        double vm = (vd.z0 + vd.z1) / 2;
        float rv[4] = {(float)(vm - tz), (float)((vd.z1 - vd.z0) / 2), (float)(track_y(vm) - 41.5 - cam.p[1]), 0};
        glUniform4fv(UL(p_ter, "uTrack"), 1, t4);
        glUniform4fv(UL(p_ter, "uTrack2"), 1, t5);
        glUniform4fv(UL(p_ter, "uRiver"), 1, rv);
    }
    for (int i = 0; i < ndraw; i++) {
        Tile *t = drawlist[i];
        if (!tile_visible(t)) continue;
        {
            double size = TROOT / (1 << t->level);
            lights_for(p_ter, (float)(t->ox + size / 2 - cam.p[0]), (float)((t->hmin + t->hmax) / 2 - cam.p[1]), (float)(t->oz + size / 2 - cam.p[2]),
                       (float)(size * 0.71 + (t->hmax - t->hmin) / 2));
        }
        glUniform3f(UL(p_ter, "uTileO"), (float)(t->ox - cam.p[0]), (float)(-cam.p[1]), (float)(t->oz - cam.p[2]));
        glUniform3f(UL(p_ter, "uTileW"), (float)fmod(t->ox, 4096.0), 0, (float)fmod(t->oz, 4096.0));
        gm_draw(&t->mesh);
    }

    /* the sky behind it all */
    prof_mark(PR_SKY);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    set_common(p_sky);
    glUniformMatrix3fv(UL(p_sky, "uCamRot"), 1, GL_FALSE, rot3);
    glUniform4fv(UL(p_sky, "uViewTan"), 1, view_tan);
    glUniform1f(UL(p_sky, "uPhase"), sky.phase);
    glUniform3f(UL(p_sky, "uMoonUp"), sky.moon_up.x, sky.moon_up.y, sky.moon_up.z);
    glUniform1f(UL(p_sky, "uCloud"), sky.cloud);
    glUniform1f(UL(p_sky, "uStars"), sky.weather == 0 ? 1.0f : 0.15f);
    glUniform4f(UL(p_sky, "uTowns"), sky.towns[0], sky.towns[1], sky.towns[2], 0);
    fullscreen();
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    /* rain or snow, if it's that kind of night */
    if (sky.weather == 1 || sky.weather == 2) {
        int snow = sky.weather == 2;
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        set_common(p_rain);
        float box = snow ? 24.0f : 30.0f;
        glUniform3f(UL(p_rain, "uCamW"), (float)fmod(cam.p[0], (double)box), (float)fmod(cam.p[1], (double)box), (float)fmod(cam.p[2], (double)box));
        glUniform3fv(UL(p_rain, "uCamV"), 1, cam_vel);
        glUniform1f(UL(p_rain, "uSnowing"), (float)snow);
        glUniform1f(UL(p_rain, "uShutter"), snow ? 0.012f : 0.02f);
        glUniform1f(UL(p_rain, "uTime"), (float)fmod(t_now, 600.0));
        glBindVertexArray(vao_full);
        if (snow) glDrawArrays(GL_LINES, 0, 2 * 9000);
        else glDrawArrays(GL_LINES, 0, 2 * 14000);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }

    prof_mark(PR_RESOLVE);
    glDisable(GL_DEPTH_TEST);

    /* smoke and steam, at half resolution, over it */
    prof_mark(PR_SMOKE);
    smoke_draw();
    prof_mark(PR_BLOOM);

    /* bloom */
    glUseProgram(p_down);
    glActiveTexture(GL_TEXTURE5);
    for (int i = 0; i < NBLOOM; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i]);
        glViewport(0, 0, bloom_w[i], bloom_h[i]);
        int sw2 = i ? bloom_w[i - 1] : scene_w, sh2 = i ? bloom_h[i - 1] : scene_h;
        glBindTexture(GL_TEXTURE_2D, i ? tex_bloom[i - 1] : tex_scene);
        glUniform1i(UL(p_down, "uSrc"), 5);
        glUniform2f(UL(p_down, "uTexel"), 1.0f / sw2, 1.0f / sh2);
        glUniform1f(UL(p_down, "uPrefilter"), i ? 0.0f : 0.15f);
        fullscreen();
    }
    glUseProgram(p_up);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    for (int i = NBLOOM - 1; i > 0; i--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom[i - 1]);
        glViewport(0, 0, bloom_w[i - 1], bloom_h[i - 1]);
        glBindTexture(GL_TEXTURE_2D, tex_bloom[i]);
        glUniform1i(UL(p_up, "uSrc"), 5);
        glUniform2f(UL(p_up, "uTexel"), 1.0f / bloom_w[i], 1.0f / bloom_h[i]);
        fullscreen();
    }
    glDisable(GL_BLEND);

    prof_mark(PR_POST);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    set_common(p_post);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex_bloom[0]);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, tex_cap);
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, tex_depth);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(UL(p_post, "uScene"), 5);
    glUniform1i(UL(p_post, "uBloom"), 6);
    glUniform1i(UL(p_post, "uCap"), 7);
    glUniform1i(UL(p_post, "uDepth"), 8);
    glUniform1f(UL(p_post, "uExposure"), 3.5f);
    glUniform1f(UL(p_post, "uBloomAmt"), 0.05f);
    glUniform1f(UL(p_post, "uFade"), director_fade(sim_t));
    glUniform4f(UL(p_post, "uVig"), (view_x - lay_x0) / lay_w, (lay_y0 + lay_h - view_y - view_h) / lay_h, view_w / lay_w, view_h / lay_h);
    {
        /* the caption sits in the front output's bottom left corner */
        int show = opt_caption && (span_me < 0 || (view_w == front_w && fabsf(view_x + view_w / 2 - cam_cx) < 1));
        float ch = front_h * 0.075f, cw = ch * CAP_W / CAP_H;
        float x0 = front_h * 0.05f, y0 = front_h * 0.045f;
        glUniform4f(UL(p_post, "uCapRect"), x0 / W, y0 / H, cw / W, ch / H);
        glUniform1f(UL(p_post, "uCapOn"), show ? 1.0f : 0.0f);
    }
    fullscreen();
    prof_mark(PR_N);
}

static void gl_init(void) {
    p_obj = mk_prog("obj", VS_OBJ, FSRC(FS_OBJ_HEAD, FS_OBJ));
    p_shadow = mk_prog("shadow", VS_OBJ, FSRC(FS_OBJ_HEAD, FS_SHADOW));
    p_ter = mk_prog("terrain", VS_TER, FSRC(FS_TER, ""));
    p_tree = mk_prog("tree", VS_TREE, FSRC(FS_TREE, ""));
    p_tree_pre = mk_prog("tree_pre", VS_TREE, FSRC(FS_NONE, ""));
    p_wire = mk_prog("wire", VS_OBJ, FSRC(FS_WIRE, ""));
    {
        static char vs[1 << 15];
        snprintf(vs, sizeof vs, "%s%s%s", HDR, GL_COMMON, VS_RAIN + strlen(HDR));
        p_rain = mk_prog("rain", vs, FSRC(FS_RAIN, ""));
    }
    p_ter_pre = mk_prog("ter_pre", VS_TER, FSRC(FS_NONE, ""));
    p_sky = mk_prog("sky", VS_FULL, FSRC(FS_SKY, ""));
    p_down = mk_prog("down", VS_FULL, FSRC(FS_DOWN, ""));
    p_up = mk_prog("up", VS_FULL, FSRC(FS_UP, ""));
    p_post = mk_prog("post", VS_FULL, FSRC(FS_POST, ""));
    p_smoke = mk_prog("smoke", VS_SMOKE, FSRC(FS_SMOKE, ""));
    p_smokecomp = mk_prog("smokecomp", VS_FULL, FSRC(FS_SMOKECOMP, ""));
    glGenVertexArrays(1, &vao_full);
    tex_shadow = mk_tex(SHADOW_N, SHADOW_N, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_LINEAR);
    fbo_shadow = mk_fbo(0, tex_shadow);
    loco_meshes_init();
    trees_init();
    lineside_init();
    smoke_init();
    signs_build();
}

static void usage(void) {
    fprintf(stderr,
        "usage: steam [--window | --root | --layer] [--seed N] [--fps N] [--font FAMILY] [--sans FAMILY]\n"
        "             [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "             [--scale F] [--weather 0-3] [--view N] [--fire] [--no-caption]\n"
        "             [--shot FILE.ppm [--at SECONDS]] [--bench N] [--size WxH]\n");
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

/* keep about a 1200p scene's worth of pixels */
static float auto_scale(void) {
    if (opt_scale > 0) return opt_scale;
    float px = (float)W * H;
    return px > 1920.0f * 1200 ? sqrtf(1920.0f * 1200 / px) : 1.0f;
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
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-view") && i + 1 < argc) opt_view = atoi(argv[++i]);
        else if (!strcmp(a, "-weather") && i + 1 < argc) opt_weather = atoi(argv[++i]);
        else if (!strcmp(a, "-no-caption")) opt_caption = 0;
        else if (!strcmp(a, "-fire")) force_round = 1e9;     /* set from --at below */
        else usage();
    }
    if (force_round > 1e8) force_round = opt_at - (getenv("STEAM_FIRE_AGE") ? atof(getenv("STEAM_FIRE_AGE")) : 2.0);
    if (span_me >= span_n) span_me = -1;
    if (opt_fps < 1) opt_fps = 1;
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    wseed = opt_seed * 0x9e3779b9u;
    for (int i = 0; i < 6; i++) tph[i] = hrand(i, 0, 77) * 2 * M_PI;
    int offscreen = opt_shot || opt_bench > 0;

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
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    face_serif = load_face(opt_font, "weight=regular");
    face_sans = load_face(opt_sans, "weight=regular");
    face_sans_b = load_face(opt_sans, "weight=bold");

    /* --epoch: when the run began, shared by every instance; a --shot then
     * shows the moment --at seconds after it */
    double t_wall0 = opt_epoch ? opt_epoch : wall();
    if (opt_shot && !opt_epoch) t_wall0 -= opt_at;
    locked_at = lock_start();
    if (!locked_at) locked_at = (time_t)t_wall0;
    sky_init(t_wall0);
    if (!opt_shot) tiles_start_worker();
    sim_init();

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
        win = XCreateWindow(dpy, root, 0, 0, (unsigned)W, (unsigned)H, 0, vi->depth, InputOutput, vi->visual,
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
        GLuint tex = mk_tex(W, H, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        GLuint fbo = mk_fbo(tex, 0);
        render_scale = auto_scale();
        if (getenv("STEAM_LIST")) {
            /* testing: list the shots the director takes over the first N seconds */
            static const char *nm[NSHOT] = {"wheels", "front", "passby", "aerial", "cab", "side", "moon", "fire", "viaduct", "station", "tunnel", "crossing"};
            double T = atof(getenv("STEAM_LIST"));
            int last = -1;
            for (double t = 0; t < T; t += 0.25) {
                advance(t, t_wall0 + t);
                if (shot.n != last) {
                    last = shot.n;
                    printf("%7.1f  %-9s  until %7.1f  (z %.0f, v %.1f)\n", shot.t0, nm[shot.kind], shot.t1, tr.z, tr.v);
                }
            }
            return 0;
        }
        if (opt_shot) {
            double wt = t_wall0 + opt_at;
            advance(opt_at, wt);
            caption_update(wt);
            away_board(wt);
            render(fbo);
            glFinish();
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            save_ppm(opt_shot);
            return 0;
        }
        double wt0 = wall();
        advance(opt_at, wt0);
        caption_update(wt0);
        away_board(wt0);
        render(fbo);
        glFinish();
        double sum = 0, worst = 0;
        glGenQueries(PR_N + 1, prof_q);
        prof_on = 1;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            advance(opt_at + (i + 1) / 60.0, wt0 + (i + 1) / 60.0);
            double c0 = now();
            render(fbo);
            prof_cpu += now() - c0;
            glFinish();
            prof_collect();
            double ms = (now() - f0) * 1000;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("steam: %dx%d (scene x%.2f), %d frames: avg %.3f ms, worst %.3f ms per frame\n", W, H, render_scale, opt_bench, sum / opt_bench, worst);
        printf("steam: verts loco %d tender %d coach %d chunk0 %d tree %d\n", g_loco.n, g_tender.n, g_coach.n, chunks[0].mesh.n, g_tree.n);
        printf("steam: GPU ms/frame:");
        for (int i = 0; i < PR_N; i++) printf(" %s %.2f", prof_name[i], prof_ms[i] / opt_bench);
        printf("; CPU in render %.2f\n", prof_cpu * 1000 / opt_bench);
        printf("steam: GL renderer: %s\n", (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    for (;;) {
        if (opt_layer && !wl_poll()) return 0;
        while (dpy && XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win, (unsigned)ev.xconfigure.width, (unsigned)ev.xconfigure.height);
                else { W = ev.xconfigure.width; H = ev.xconfigure.height; }
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        double t = now(), wt = wall(), el = opt_epoch ? fmax(wt - opt_epoch, 0) : t - t0;
        render_scale = auto_scale();
        advance(el, wt);
        caption_update(wt);
        away_board(wt);
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
