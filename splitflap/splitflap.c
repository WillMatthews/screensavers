/*
 * splitflap — a lock-screen away message on an old railway departure board.
 *
 * Rows of split-flap modules: dark flaps on hinges, letters cut through
 * the middle by the split, an axle pin either side. Every 10-15 s the board
 * changes: the top departure leaves and the rows below move up, or a row or
 * two are replaced, and each module turns its flaps over one at a time until
 * it reaches the new letter — so a letter further round the drum takes longer,
 * and a change ripples across the board left to right, row after row.
 *
 *     TIME   DESTINATION   STATUS     REMARKS
 *     19:07  WILL          AWAY       SINCE 19:07
 *     19:31  OVERTIME      LOCKED     24 MIN
 *     19:35  THE KETTLE    BOARDING   TEA, PROBABLY
 *     ...
 *
 * The first two rows are live (who's away and since when, how long the host
 * has been locked); the rest are drawn from ENTRIES below, and a two-line
 * notice under the board rotates through NOTICES. A big flap clock sits in
 * the header. Live details (the time, how long the screen has been locked —
 * from swaylock's start time —, battery, uptime) come from stir's engine and
 * refresh every minute. $N, $H, $S, $D and *starred* words are set in the keyword colour (gold, in brass).
 *
 * Each module is two instanced quads: the module at rest (top half of the
 * letter coming, bottom half of the letter going) and, while it turns, the
 * flap falling over the hinge, in perspective, front face lit from above,
 * back face (the bottom of the new letter) landing over the old one. Nothing
 * moving, nothing drawn: between changes the board sleeps.
 *
 * Palettes: brass (Aged Brass: cream on brown-black, gold keywords), classic
 * (white on charcoal, yellow keywords, orange-red trouble), blue (white on
 * navy, warm yellow keywords), led (amber on black). One is picked from the
 * seed, so each lock looks different; --palette NAME picks it outright.
 *
 * Same plumbing as stir: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame offscreen,
 * --bench N to time N frames offscreen.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "proto-layer-shell.h"

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench;
static const char *opt_font = "JetBrainsMono Nerd Font Mono,monospace:style=Bold";   /* flap lettering is heavy */
static const char *opt_name = "Will";
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = -1;             /* --shot / --bench: from this many seconds in */
static int off_w = 1920, off_h = 1080;

/* ---- what the board says ----
 * Tokens: $N name, $H host, $T time now, $S locked since, $D how long locked,
 * $DATE today, $B battery, $UP uptime. $N, $H, $S and $D come out gold, as
 * does anything between *stars*. Board text is upper-cased; anything the
 * flaps don't carry (see FLAPS) turns into a blank. */

/* departures: destination (12), status (9), remarks (18). The status is
 * coloured by what it says, see status_colour(). */
static const char *ENTRIES[][3] = {
    {"THE KETTLE", "BOARDING", "TEA, PROBABLY"},
    {"LUNCH", "DELAYED", "BACK SOON"},
    {"A MEETING", "ON TIME", "COULD BE AN EMAIL"},
    {"A WALK", "DEPARTED", "FRESH AIR"},
    {"COFFEE", "BOARDING", "FLAT WHITE"},
    {"EMAIL", "CANCELLED", "TRY SLACK"},
    {"KEYBOARD", "IDLE", "SINCE $S"},
    {"THIS SCREEN", "LOCKED", "KEYS WON'T HELP"},
    {"THE FANS", "SPINNING", "ALL IS WELL"},
    {"$N", "EXPECTED", "SHORTLY-ISH"},
    {"THE BUILD", "RUNNING", "PROBABLY"},
    {"ALL TABS", "UNCHANGED", "AS $N LEFT THEM"},
    {"BATTERY", "$B", "HOLDING UP"},
    {"$H", "UP $UP", "IN NO HURRY"},
    {"NOTES", "ON PAPER", "PAPER STILL WORKS"},
    {"THE CHAIR", "VACANT", "NEGOTIABLE"},
    {"MESSAGES", "QUEUED", "WHEN NOTICED"},
    {"$H", "GUARDING", "THE DESK"},
    {"FISHING", "DEPARTED", "METAPHORICALLY"},
    {"THE CURSOR", "RESTING", "SO IS $N"},
    {"FOCUS", "CHARGING", "BACK SHORTLY"},
    {"THE DESK", "ATTENDED", "BY A PLANT"},
    {"TEA", "BREWING", "4 MIN STEEP"},
    {"A SANDWICH", "ON TIME", "PLATFORM 2"},
    {"DEEP THOUGHT", "ONGOING", "ABOUT LUNCH"},
    {"RIGHT BACK", "DELAYED", "DEFINE 'RIGHT'"},
    {"PING $N", "NO REPLY", "TRY LATER"},
    {"LAST SEEN", "$S", "NEAR A KETTLE"},
    {"THE BUILDING", "LEFT", "BUILDING IS FINE"},
    {"THE PRINTER", "JAMMED", "NOT $N'S FAULT"},
    {"THE FRIDGE", "BOARDING", "SNACK RUN"},
    {"STAND-UP", "ON TIME", "SITTING DOWN"},
    {"A BIT", "EXPECTED", "1 MIN TO 1 HR"},
    {"UNSAVED WORK", "SAFE", "NONE HARMED"},
    {"THE LIFT", "DELAYED", "STAIRS ADVISED"},
    {"FRESH AIR", "BOARDING", "WINDOW SEAT"},
    {"BISCUITS", "CANCELLED", "TIN IS EMPTY"},
    {"LOCKED SINCE", "$S", "STILL LOCKED"},
    {"THE PLANT", "WATERED", "PROBABLY"},
    {"INBOX", "HOLDING", "WAITING ON $N"},
};
#define NENTRY ((int)(sizeof ENTRIES / sizeof *ENTRIES))

/* the notice under the board, up to two lines */
static const char *NOTICES[] = {
    "This is $N's computer. $N is *away*.",
    "$H is locked. Pressing keys won't help, but it won't hurt either.",
    "Passengers are reminded that this computer is locked.",
    "$N has been *away* since $S. Normal service will resume shortly.",
    "Please do not feed the laptop.",
    "If it's urgent, try Slack.",
    "Leave a note on paper. Paper still works.",
    "$N is most likely by the kettle.",
    "Out of office, but only by a few metres.",
    "The tea is probably going cold.",
    "All the tabs are exactly where $N left them.",
    "Unattended computer. Attended desk plant.",
    "$N will be right back. Define 'right'.",
    "Please mind the gap between the chair and the desk.",
    "Delays are expected. $N apologises for any inconvenience.",
    "This service is locked. Please stand clear of the keyboard.",
    "Lost property: one $N. If found, please return to $H.",
    "Nobody is typing. That is the point.",
    "Gone fishing. Metaphorically.",
    "Back in a bit. A bit is somewhere between a minute and an hour.",
    "It is $T. $N was last seen at $S.",
    "$H has been up for $UP and is in no hurry.",
    "Battery at $B and holding.",
    "Messages will be answered in the order they are noticed.",
    "The keyboard misses $N. The keyboard is like that.",
    "Knock knock. Who's there? Not $N.",
    "This screen will unlock when $N returns. Not before.",
    "$N is out. $H is in.",
    "*Away*, but not far.",
    "Locked for $D and counting.",
    "See it, say it, leave it for $N.",
    "Today is $DATE. $H is guarding the desk.",
};
#define NNOTICE ((int)(sizeof NOTICES / sizeof *NOTICES))

static void die(const char *what) { fprintf(stderr, "splitflap: %s\n", what); exit(1); }

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

/* ---- live details (stir's, in board-sized words) ---- */
static char host[64], name_uc[64];
static time_t locked_at;
static char d_now[16], d_since[32], d_dur[48], d_date[48], d_batt[48], d_up[48], d_dow[16];
static struct tm now_tm;

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else now */
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
    return best ? best : time(NULL);
}

/* a span the way a board would put it: 12 MIN, 1 HR 5 MIN, 2 DAYS 3 HR */
static void fmt_span(char *out, size_t n, long s) {
    long m = s / 60, h = m / 60, dd = h / 24;
    if (dd >= 1) snprintf(out, n, "%ld DAY%s %ld HR", dd, dd == 1 ? "" : "S", h % 24);
    else if (h >= 1 && m % 60) snprintf(out, n, "%ld HR %ld MIN", h, m % 60);
    else if (h >= 1) snprintf(out, n, "%ld HR", h);
    else if (m >= 1) snprintf(out, n, "%ld MIN", m);
    else snprintf(out, n, "UNDER 1 MIN");
}

static void live_init(void) {
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    for (char *p = host; *p; p++) *p = (char)toupper((unsigned char)*p);
    snprintf(name_uc, sizeof name_uc, "%s", opt_name);
    for (char *p = name_uc; *p; p++) *p = (char)toupper((unsigned char)*p);
    locked_at = lock_start();
}

static void live_update(time_t t) {
    struct tm since_tm;
    localtime_r(&t, &now_tm);
    localtime_r(&locked_at, &since_tm);
    strftime(d_now, sizeof d_now, "%H:%M", &now_tm);
    if (since_tm.tm_yday == now_tm.tm_yday && since_tm.tm_year == now_tm.tm_year)
        strftime(d_since, sizeof d_since, "%H:%M", &since_tm);
    else
        strftime(d_since, sizeof d_since, "%H:%M %a", &since_tm);
    fmt_span(d_dur, sizeof d_dur, (long)(t - locked_at));
    strftime(d_date, sizeof d_date, "%A %-d %B", &now_tm);
    strftime(d_dow, sizeof d_dow, "%a %d %b", &now_tm);

    char buf[64];
    double up = 0;
    slurp_line("/proc/uptime", buf, sizeof buf);
    sscanf(buf, "%lf", &up);
    fmt_span(d_up, sizeof d_up, (long)up);

    d_batt[0] = 0;
    glob_t g;
    if (glob("/sys/class/power_supply/BAT*/capacity", 0, NULL, &g) == 0 && g.gl_pathc) {
        char cap[16], st[32], path[300];
        slurp_line(g.gl_pathv[0], cap, sizeof cap);
        snprintf(path, sizeof path, "%.*s/status", (int)(strrchr(g.gl_pathv[0], '/') - g.gl_pathv[0]), g.gl_pathv[0]);
        slurp_line(path, st, sizeof st);
        if (cap[0]) snprintf(d_batt, sizeof d_batt, "%s%%%s", cap, !strcmp(st, "Charging") ? " CHARGING" : "");
    }
    globfree(&g);
}

/* ---- the flaps: the drum of every module, in the order it turns ---- */
static const char FLAPS[] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-/&!?%+()@";
#define NF ((int)sizeof FLAPS - 1)
static const char DIGITS[] = " 0123456789";   /* the clock's drums carry only digits */
static const char TIMES[] = " 0123456789:-";  /* and the time column's, digits and a colon */
#define G_DOT (NF)                          /* extra glyphs for the painted signs */
#define G_TRAIN (NF + 1)
#define NGLYPH (NF + 2)

static int flap_index(const char *seq, int c) {
    c = toupper(c);
    const char *p = c ? strchr(seq, c) : NULL;
    return p ? (int)(p - seq) : 0;
}

/* ink colours, uInk[] in the shader */
enum { INK_CREAM, INK_GOLD, INK_AMBER, INK_SAGE, INK_DIM, INK_BRASS, INK_TRAY, INK_N };

/* expand tokens and *gold* into upper-case text and per-letter colours; 0 if
 * it needs a detail this machine doesn't have (no battery) */
static int expand(const char *m, char *out, unsigned char *col, int cap) {
    int n = 0, gold = 0;
    for (const char *p = m; *p && n < cap - 1;) {
        if (*p == '*') { gold = !gold; p++; continue; }
        if (*p == '$') {
            static const struct { const char *tok; int gold; } toks[] = {
                {"DATE", 0}, {"UP", 0}, {"N", 1}, {"H", 1}, {"T", 0}, {"S", 1}, {"D", 1}, {"B", 0}};
            const char *val = NULL;
            int g = 0;
            for (size_t k = 0; k < sizeof toks / sizeof *toks && !val; k++) {
                size_t l = strlen(toks[k].tok);
                if (strncmp(p + 1, toks[k].tok, l)) continue;
                const char *v[] = {d_date, d_up, name_uc, host, d_now, d_since, d_dur, d_batt};
                val = v[k]; g = toks[k].gold;
                p += 1 + l;
            }
            if (val) {
                if (!*val) return 0;
                for (; *val && n < cap - 1; val++) {
                    out[n] = (char)toupper((unsigned char)*val);
                    col[n++] = (gold || g) && *val != ' ' ? INK_GOLD : INK_CREAM;
                }
                continue;
            }
        }
        out[n] = (char)toupper((unsigned char)*p++);
        col[n] = gold && out[n] != ' ' ? INK_GOLD : INK_CREAM;
        n++;
    }
    out[n] = 0;
    return 1;
}

static int status_colour(const char *s) {
    static const char *sage[] = {"ON TIME", "BOARDING", "DEPARTED", "SAFE", "WATERED"};
    static const char *amber[] = {"DELAYED", "CANCELLED", "JAMMED", "NO REPLY"};
    for (size_t i = 0; i < sizeof sage / sizeof *sage; i++) if (!strcmp(s, sage[i])) return INK_SAGE;
    for (size_t i = 0; i < sizeof amber / sizeof *amber; i++) if (!strcmp(s, amber[i])) return INK_AMBER;
    if (!strcmp(s, "AWAY") || !strcmp(s, "LOCKED")) return INK_GOLD;
    return -1;
}

/* ---- glyph atlas: every glyph at each size the board uses, in one R8
 * texture. A slot is a size: cells of cw x ch, the capitals centred in the
 * cell so the split runs through their middle. ---- */
enum { SLOT_FLAP, SLOT_CLOCK, SLOT_TITLE, SLOT_LABEL, SLOT_N };
static struct { int x, y, cw, ch; } slot[SLOT_N];
static unsigned char *atlas;
static int atlas_w, atlas_h;
static FT_Library ft;
static FT_Face face;
static float cap_ratio, adv_ratio;     /* cap height and advance per pixel of font size */

static void font_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    FcPattern *pat = FcNameParse((const FcChar8 *)opt_font);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    if (FT_Init_FreeType(&ft) || FT_New_Face(ft, (const char *)file, 0, &face)) die("can't load the font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
    FT_Set_Pixel_Sizes(face, 0, 1000);
    FT_Load_Char(face, 'H', FT_LOAD_NO_BITMAP);
    cap_ratio = face->glyph->metrics.height / 64.0f / 1000.0f;
    FT_Load_Char(face, 'M', FT_LOAD_NO_BITMAP);
    adv_ratio = face->glyph->advance.x / 64.0f / 1000.0f;
    if (cap_ratio <= 0 || adv_ratio <= 0) die("the font has no capitals");
}

static unsigned glyph_code(int g) {
    return g < NF ? (unsigned char)FLAPS[g] : g == G_DOT ? 0xB7 : 0xF238;
}

/* render the glyphs of one slot: capitals cap px tall in cw x ch cells */
static void atlas_slot(int s, float cap) {
    int cw = slot[s].cw, ch = slot[s].ch;
    FT_Set_Pixel_Sizes(face, 0, (FT_UInt)lroundf(cap / cap_ratio));
    int base = (int)lroundf(ch * 0.5f + cap * 0.5f);   /* baseline: capitals centred */
    for (int g = 1; g < NGLYPH; g++) {
        unsigned code = glyph_code(g);
        if (!FT_Get_Char_Index(face, code)) continue;
        if (FT_Load_Char(face, code, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
        FT_GlyphSlot gs = face->glyph;
        int adv = (int)(gs->advance.x >> 6);
        int x0 = slot[s].x + g * cw + (cw - adv) / 2 + gs->bitmap_left;
        int y0 = slot[s].y + base - gs->bitmap_top;
        if (g == G_TRAIN) y0 = slot[s].y + (ch - (int)gs->bitmap.rows) / 2;   /* icons: centred */
        for (unsigned y = 0; y < gs->bitmap.rows; y++)
            for (unsigned x = 0; x < gs->bitmap.width; x++) {
                int ax = x0 + (int)x, ay = y0 + (int)y;
                if (ax < slot[s].x + g * cw || ax >= slot[s].x + (g + 1) * cw || ay < slot[s].y || ay >= slot[s].y + ch) continue;
                unsigned char v = gs->bitmap.buffer[y * gs->bitmap.pitch + x];
                unsigned char *d = &atlas[(size_t)ay * atlas_w + ax];
                if (v > *d) *d = v;
            }
    }
}

/* ---- the board: modules, grouped into fields ---- */
typedef struct {
    short x, y;                 /* top left, px */
    unsigned char sl;           /* atlas slot */
    unsigned char ns;           /* drum size */
    const char *seq;            /* the drum */
    unsigned char cur, ccol;    /* the flap showing, before this run; its ink */
    unsigned char tgt, tcol;    /* where this run ends up */
    unsigned short n;           /* flaps to turn in this run (0: at rest) */
    double t0;                  /* when the run starts */
    float step;                 /* seconds per flap, a little different for each module */
} Mod;
static Mod *mods;
static int nmod, capmod;

typedef struct { int first, len; } Field;

enum { TCOL_TIME, TCOL_DEST, TCOL_STATUS, TCOL_REM, NTCOL };
static const int TCOL_W[NTCOL] = {5, 12, 9, 18};
static const char *TCOL_NAME[NTCOL] = {"TIME", "DESTINATION", "STATUS", "REMARKS"};
#define MAXROWS 14

static Field f_clock[2], f_date, f_row[MAXROWS][NTCOL], f_notice[2];
static int nrows;                       /* table rows, sized to the screen */
static int fw, fh, fgx, rowp;           /* module size, gap between modules, row pitch */
static int cwid, chgt;                  /* clock module size */

/* painted things: signs (text) and solid rounded rects (trays, rules) */
typedef struct { float x, y, w, h; int kind, a, b, c; float r; } Paint;
static Paint paint[1024];
static int npaint;

static void add_rect(float x, float y, float w, float h, int ink, float radius) {
    if (npaint < (int)(sizeof paint / sizeof *paint))
        paint[npaint++] = (Paint){x, y, w, h, 3, ink, 0, 0, radius};
}

/* painted text in a slot; returns its width */
static float add_text(float x, float y, int sl, int ink, const char *s) {
    float x0 = x;
    for (; *s; s++, x += slot[sl].cw) {
        int g = *s == 1 ? G_DOT : *s == 2 ? G_TRAIN : flap_index(FLAPS, (unsigned char)*s);
        if (g && npaint < (int)(sizeof paint / sizeof *paint))
            paint[npaint++] = (Paint){x, y, (float)slot[sl].cw, (float)slot[sl].ch, 2, g, ink, sl, 0};
    }
    return x - x0;
}

static Field add_field(int x, int y, int len, int sl, int w, const char *seq) {
    Field f = {nmod, len};
    if (nmod + len > capmod) mods = realloc(mods, (size_t)(capmod = (nmod + len) * 2) * sizeof *mods);
    for (int i = 0; i < len; i++) {
        Mod *m = &mods[nmod++];
        memset(m, 0, sizeof *m);
        m->x = (short)(x + i * (w + fgx)); m->y = (short)y;
        m->sl = (unsigned char)sl;
        m->seq = seq; m->ns = (unsigned char)strlen(seq);
        m->step = 0.052f * (0.88f + 0.24f * frand());
    }
    return f;
}

static int field_w(int len, int w) { return len * w + (len - 1) * fgx; }

/* the tray a field of modules sits in */
static void add_tray(Field f, int w, int h) {
    float pad = fgx * 0.9f + 2;
    add_rect(mods[f.first].x - pad, mods[f.first].y - pad, field_w(f.len, w) + 2 * pad, h + 2 * pad, INK_TRAY, pad * 1.3f);
}

/* lay the board out for W x H: modules sized to fit the table across ~90% of
 * the width, as many rows as the height allows */
static void layout(void) {
    uint32_t keep = rng;
    rng = opt_seed * 2654435761u | 1;
    nmod = 0; npaint = 0;
    int tw = 0;
    for (int c = 0; c < NTCOL; c++) tw += TCOL_W[c];
    float colgap = 1.1f;                                 /* between columns, in module pitches */
    float pitch = 0.9f * W / (tw + (NTCOL - 1) * colgap);
    /* height: header ~2.4 rows, labels, at least 6 rows, notice 2 + gaps */
    float rp_max = 0.92f * H / (2.6f + 0.9f + 6 + 0.7f + 2.2f);
    if (pitch * 0.87f * 1.45f * 1.24f > rp_max) pitch = rp_max / (0.87f * 1.45f * 1.24f);
    fw = (int)lroundf(pitch * 0.87f);
    fgx = (int)lroundf(pitch) - fw;
    if (fgx < 2) fgx = 2;
    fh = (int)lroundf(fw * 1.45f);
    rowp = fh + (int)lroundf(fh * 0.24f);
    chgt = (int)lroundf(fh * 1.9f);
    cwid = (int)lroundf(chgt * 0.66f);

    /* the atlas for these sizes */
    slot[SLOT_FLAP].cw = fw; slot[SLOT_FLAP].ch = fh;
    slot[SLOT_CLOCK].cw = cwid; slot[SLOT_CLOCK].ch = chgt;
    float cap_title = chgt * 0.36f, cap_label = fh * 0.27f;
    slot[SLOT_TITLE].cw = (int)lroundf(cap_title / cap_ratio * adv_ratio * 1.08f); slot[SLOT_TITLE].ch = (int)lroundf(cap_title * 1.8f);
    slot[SLOT_LABEL].cw = (int)lroundf(cap_label / cap_ratio * adv_ratio * 1.25f); slot[SLOT_LABEL].ch = (int)lroundf(cap_label * 2.2f);
    atlas_w = atlas_h = 0;
    for (int s = 0; s < SLOT_N; s++) {
        slot[s].x = 0; slot[s].y = atlas_h;
        atlas_h += slot[s].ch + 1;
        if (slot[s].cw * NGLYPH > atlas_w) atlas_w = slot[s].cw * NGLYPH;
    }
    free(atlas);
    atlas = calloc((size_t)atlas_w * atlas_h, 1);
    atlas_slot(SLOT_FLAP, fh * 0.55f);
    atlas_slot(SLOT_CLOCK, chgt * 0.55f);
    atlas_slot(SLOT_TITLE, cap_title);
    atlas_slot(SLOT_LABEL, cap_label);

    /* columns */
    int colx[NTCOL], bw;
    int x = 0;
    for (int c = 0; c < NTCOL; c++) {
        colx[c] = x;
        x += field_w(TCOL_W[c], fw) + (int)lroundf(colgap * (fw + fgx)) + fgx;
    }
    bw = colx[NTCOL - 1] + field_w(TCOL_W[NTCOL - 1], fw);
    int x0 = (W - bw) / 2;

    /* vertical: header, labels, rows, notice; rows fill what's left */
    int head = chgt, lab = (int)lroundf(fh * 0.95f), gap1 = (int)lroundf(fh * 0.55f), gap2 = (int)lroundf(fh * 0.7f);
    int notice = 2 * rowp - (rowp - fh);
    int avail = (int)(0.9f * H) - head - gap1 - lab - gap2 - notice;
    nrows = (avail + (rowp - fh)) / rowp;
    if (nrows > 10) nrows = 10;
    if (nrows < 4) nrows = 4;
    int total = head + gap1 + lab + nrows * rowp - (rowp - fh) + gap2 + notice;
    int y = (H - total) / 2;

    /* header: the sign on the left, date and clock on the right */
    int cx = x0 + bw - (4 * cwid + 2 * fgx + 2 * (int)lroundf(cwid * 0.55f));
    f_clock[0] = add_field(cx, y, 2, SLOT_CLOCK, cwid, DIGITS);
    int cx2 = cx + 2 * cwid + fgx + 2 * (int)lroundf(cwid * 0.55f);
    f_clock[1] = add_field(cx2, y, 2, SLOT_CLOCK, cwid, DIGITS);
    add_tray(f_clock[0], cwid, chgt);
    add_tray(f_clock[1], cwid, chgt);
    add_text(cx + 2 * cwid + fgx + (2 * (int)lroundf(cwid * 0.55f) - cwid) * 0.5f - fgx * 0.5f, (float)y, SLOT_CLOCK, INK_GOLD, ":");
    int dlen = 10;
    int dx = cx - (int)lroundf(1.4f * (fw + fgx)) - field_w(dlen, fw);
    f_date = add_field(dx, y + chgt - fh, dlen, SLOT_FLAP, fw, FLAPS);
    add_tray(f_date, fw, fh);

    char sub[96];
    snprintf(sub, sizeof sub, "%s \1 AWAY SERVICES", host);
    float tx = (float)x0;
    float ty = y + chgt * 0.5f - slot[SLOT_TITLE].ch * 0.62f;
    tx += add_text(tx, ty, SLOT_TITLE, INK_GOLD, "\2") + slot[SLOT_TITLE].cw * 0.35f;
    float title_x = tx;
    add_text(tx, ty, SLOT_TITLE, INK_CREAM, "DEPARTURES");
    add_text(title_x, y + chgt - slot[SLOT_LABEL].ch * 0.75f, SLOT_LABEL, INK_DIM, sub);
    y += head + gap1;

    /* column labels and a brass rule under them */
    for (int c = 0; c < NTCOL; c++)
        add_text((float)x0 + colx[c] - (slot[SLOT_LABEL].cw - fw * 0.0f) * 0.0f, y + lab * 0.5f - slot[SLOT_LABEL].ch * 0.5f - fh * 0.08f,
                 SLOT_LABEL, INK_DIM, TCOL_NAME[c]);
    add_rect((float)x0, (float)(y + lab - 1), (float)bw, fmaxf(1.0f, roundf(fh * 0.03f)), INK_BRASS, 0);
    y += lab + (int)lroundf(fh * 0.35f);

    /* the departures */
    for (int r = 0; r < nrows; r++, y += rowp)
        for (int c = 0; c < NTCOL; c++) {
            f_row[r][c] = add_field(x0 + colx[c], y, TCOL_W[c], SLOT_FLAP, fw, c == TCOL_TIME ? TIMES : FLAPS);
            add_tray(f_row[r][c], fw, fh);
        }
    y += gap2 - (rowp - fh);

    /* the notice: two full-width lines */
    add_rect((float)x0, (float)(y - gap2 / 2), (float)bw, fmaxf(1.0f, roundf(fh * 0.03f)), INK_BRASS, 0);
    y += (int)lroundf(fh * 0.2f);
    int nlen = (bw + fgx) / (fw + fgx);
    int nx = x0 + (bw - field_w(nlen, fw)) / 2;
    for (int i = 0; i < 2; i++) {
        f_notice[i] = add_field(nx, y + i * rowp, nlen, SLOT_FLAP, fw, FLAPS);
        add_tray(f_notice[i], fw, fh);
    }
    rng = keep;
}

/* ---- turning the flaps ---- */
static void mod_settle(Mod *m, double t) {
    if (m->n && t >= m->t0 + m->n * (double)m->step) { m->cur = m->tgt; m->ccol = m->tcol; m->n = 0; }
}

/* send a module to flap idx (ink col), starting at time start; snap: at once */
static void mod_target(Mod *m, int idx, int col, double t, double start, int snap) {
    mod_settle(m, t);
    if (m->n) {
        if (t >= m->t0) {                           /* mid-run: finish this flap, then carry on */
            int k = (int)((t - m->t0) / m->step);
            m->cur = (unsigned char)((m->cur + k + 1) % m->ns);
            m->ccol = k + 1 == m->n ? m->tcol : INK_CREAM;
            m->t0 += (k + 1) * (double)m->step;
            start = m->t0;
        }
        m->n = 0;                                   /* (not started yet: never mind) */
    }
    m->tgt = (unsigned char)idx; m->tcol = (unsigned char)col;
    if (snap) { m->cur = m->tgt; m->ccol = m->tcol; return; }
    int n = (idx - m->cur + m->ns) % m->ns;
    if (!n && col != m->ccol) {
        if (idx) n = m->ns;                         /* same letter, other ink: once round */
        else m->ccol = (unsigned char)col;
    }
    m->n = (unsigned short)n;
    m->t0 = start;
}

/* set a field's text, rippling left to right from `start` */
static void set_field(Field f, const char *txt, const unsigned char *col, int ink, double t, double start, int snap) {
    int l = txt ? (int)strlen(txt) : 0;
    for (int i = 0; i < f.len; i++) {
        Mod *m = &mods[f.first + i];
        int c = i < l ? (unsigned char)txt[i] : ' ';
        int k = ink >= 0 ? ink : col ? col[i < l ? i : 0] : INK_CREAM;
        mod_target(m, flap_index(m->seq, c), k, t, start + i * 0.028 + 0.03 * frand(), snap);
    }
}

static int board_busy(double t) {
    int busy = 0;
    for (int i = 0; i < nmod; i++) { mod_settle(&mods[i], t); busy |= mods[i].n != 0; }
    return busy;
}

/* ---- what's on the board ---- */
static int row_entry[MAXROWS];          /* rows 2..: an ENTRIES index; 0 and 1 are live */
static int notice_cur = -1;
static int deck[NENTRY], ndeck, ndeal;
static int ndeck_n, deck_n[NNOTICE], ndeal_n;

static int entry_ok(int e) {
    char buf[128];
    unsigned char col[128];
    for (int k = 0; k < 3; k++)
        if (!expand(ENTRIES[e][k], buf, col, sizeof buf) || (int)strlen(buf) > TCOL_W[k + 1]) return 0;
    return 1;
}

/* the notice wrapped onto two lines; 0 if it doesn't fit */
static int notice_wrap(int i, char out[2][256], unsigned char cols[2][256]) {
    char buf[512];
    unsigned char col[512];
    int len = f_notice[0].len;
    if (!expand(NOTICES[i], buf, col, sizeof buf)) return 0;
    memset(out, 0, 2 * 256);
    int n = (int)strlen(buf), p = 0;
    for (int line = 0; line < 2; line++) {
        while (buf[p] == ' ') p++;
        int e = p + len < n ? p + len : n;
        if (e < n) {                                /* break at a space */
            int b = e;
            while (b > p && buf[b] != ' ') b--;
            if (b == p) return 0;
            e = b;
        }
        memcpy(out[line], buf + p, (size_t)(e - p));
        memcpy(cols[line], col + p, (size_t)(e - p));
        p = e;
    }
    while (buf[p] == ' ') p++;
    return p >= n;
}

static int deal_entry(void) {
    for (int tries = 0; tries < 2 * NENTRY; tries++) {
        if (ndeal >= ndeck) {                       /* reshuffle */
            for (int i = ndeck - 1; i > 0; i--) { int j = (int)(frand() * (i + 1)); if (j > i) j = i; int s = deck[i]; deck[i] = deck[j]; deck[j] = s; }
            ndeal = 0;
        }
        int e = deck[ndeal++], on = 0;
        for (int r = 2; r < nrows; r++) on |= row_entry[r] == e;
        if (!on) return e;
    }
    return deck[0];
}

static int deal_notice(void) {
    if (ndeal_n >= ndeck_n) {
        for (int i = ndeck_n - 1; i > 0; i--) { int j = (int)(frand() * (i + 1)); if (j > i) j = i; int s = deck_n[i]; deck_n[i] = deck_n[j]; deck_n[j] = s; }
        ndeal_n = 0;
        if (ndeck_n > 1 && deck_n[0] == notice_cur) { int s = deck_n[0]; deck_n[0] = deck_n[1]; deck_n[1] = s; }
    }
    return deck_n[ndeal_n++];
}

static void decks_init(void) {
    char a[2][256];
    unsigned char c[2][256];
    ndeck = ndeck_n = 0;
    for (int e = 0; e < NENTRY; e++) if (entry_ok(e)) deck[ndeck++] = e;
    for (int i = 0; i < NNOTICE; i++) if (notice_wrap(i, a, c)) deck_n[ndeck_n++] = i;
    if (!ndeck || !ndeck_n) die("nothing fits on the board");
    ndeal = ndeck; ndeal_n = ndeck_n;
}

/* show row r (start: when its first module goes) */
static void row_apply(int r, double t, double start, int snap) {
    char buf[NTCOL][128];
    unsigned char col[NTCOL][128];
    memset(col, 0, sizeof col);
    if (r == 0) {
        snprintf(buf[0], 128, "%.5s", d_since);
        expand("$N", buf[1], col[1], 128);
        snprintf(buf[2], 128, "AWAY");
        expand("SINCE $S", buf[3], col[3], 128);
    } else if (r == 1) {
        snprintf(buf[0], 128, "%s", d_now);
        expand("$H", buf[1], col[1], 128);
        snprintf(buf[2], 128, "LOCKED");
        expand("$D", buf[3], col[3], 128);
    } else {
        /* the next departures, five minutes apart */
        int m = now_tm.tm_hour * 60 + now_tm.tm_min;
        m = ((m + 4) / 5 * 5 + 5 * (r - 1)) % (24 * 60);
        snprintf(buf[0], 128, "%02d:%02d", m / 60, m % 60);
        for (int k = 0; k < 3; k++)
            if (!expand(ENTRIES[row_entry[r]][k], buf[k + 1], col[k + 1], 128)) buf[k + 1][0] = 0;
    }
    int sc = status_colour(buf[2]);
    for (int c = 0; c < NTCOL; c++) {
        double s = start + (c ? (mods[f_row[r][c].first].x - mods[f_row[r][0].first].x) / (double)(fw + fgx) * 0.028 : 0);
        set_field(f_row[r][c], buf[c], c == TCOL_TIME ? NULL : col[c], c == TCOL_TIME ? INK_CREAM : c == TCOL_STATUS && sc >= 0 ? sc : -1, t, s, snap);
    }
}

static void notice_apply(double t, double start, int snap) {
    char l[2][256];
    unsigned char c[2][256];
    if (notice_cur < 0 || !notice_wrap(notice_cur, l, c)) return;
    for (int i = 0; i < 2; i++) set_field(f_notice[i], l[i], c[i], -1, t, start + i * 0.12, snap);
}

static void clock_apply(double t, double start, int snap) {
    char hh[3] = {d_now[0], d_now[1], 0}, mm[3] = {d_now[3], d_now[4], 0};
    set_field(f_clock[0], hh, NULL, INK_GOLD, t, start, snap);
    set_field(f_clock[1], mm, NULL, INK_GOLD, t, start + 0.05, snap);
    set_field(f_date, d_dow, NULL, INK_CREAM, t, start + 0.1, snap);
}

static void board_apply(double t, double start, double row_gap, int snap) {
    clock_apply(t, start, snap);
    for (int r = 0; r < nrows; r++) row_apply(r, t, start + 0.2 + r * row_gap, snap);
    notice_apply(t, start + 0.3 + nrows * row_gap, snap);
}

/* the timetable: the board comes up at T_START, then every 10-15 s a change */
#define T_START 0.6
static double next_change, next_notice;
static int started, last_min = -1;

static void board_init(void) {
    decks_init();
    for (int r = 2; r < MAXROWS; r++) row_entry[r] = -1;
    for (int r = 2; r < MAXROWS; r++) row_entry[r] = deal_entry();
    notice_cur = deal_notice();
    started = 0;
    last_min = -1;
}

static void board_update(double t, int snap) {
    time_t wall = time(NULL);
    if (wall / 60 != last_min) {                    /* the minute turned: live details */
        int first = last_min < 0;
        last_min = (int)(wall / 60);
        live_update(wall);
        if (started && !first) board_apply(t, t, 0.09, 0);
    }
    if (!started && (t >= T_START || snap)) {
        started = 1;
        board_apply(t, snap ? t : T_START, 0.16, snap);
        next_change = t + 9 + 3 * frand();
        next_notice = t + 15 + 4 * frand();
    }
    if (!started) return;
    if (t >= next_change) {
        float r = frand();
        if (r < 0.45f) {                            /* the top one leaves: the rest move up */
            for (int i = 2; i + 1 < nrows; i++) row_entry[i] = row_entry[i + 1];
            row_entry[nrows - 1] = -1;
            row_entry[nrows - 1] = deal_entry();
            for (int i = 2; i < nrows; i++) row_apply(i, t, t + (i - 2) * 0.14, 0);
        } else {                                    /* one or two change */
            int k = r < 0.8f ? 1 : 2;
            for (int j = 0; j < k; j++) {
                int i = 2 + (int)(frand() * (nrows - 2));
                if (i >= nrows) i = nrows - 1;
                row_entry[i] = -1;
                row_entry[i] = deal_entry();
                row_apply(i, t, t + j * 0.3, 0);
            }
        }
        next_change = t + 10 + 5 * frand();
    }
    if (t >= next_notice) {
        notice_cur = deal_notice();
        notice_apply(t, t, 0);
        next_notice = t + 18 + 8 * frand();
    }
}

/* after a resize: lay out again and show everything as it stands */
static void board_relayout(double t) {
    layout();
    if (started) board_apply(t, t, 0, 1);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
/* instances: rect (x, y, w, h px, y down), what (g0, g1, angle, kind),
 * more (ink0, ink1, slot, radius). kind 0: a module at rest (top half g1,
 * bottom half g0, `angle` of the flap falling over it for its shading);
 * 1: the falling flap, front g0 top half, back g1 bottom half; 2: a painted
 * glyph g0; 3: a solid rounded rect */
static const char *VS = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aRect;
    layout(location = 1) in vec4 aWhat;
    layout(location = 2) in vec4 aMore;
    uniform vec2 uRes;
    out vec2 vL;
    flat out vec4 vRect, vWhat, vMore;
    void main() {
        vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);
        vRect = aRect; vWhat = aWhat; vMore = aMore;
        int kind = int(aWhat.w);
        vec2 p;
        float w = 1.0;
        if (kind == 1) {
            /* the flap: hinged across the middle, free edge v = 1; turned by
             * angle (0: up, showing its front; pi: down, showing its back),
             * seen in perspective from in front of the module */
            float hh = aRect.w * 0.5, th = aWhat.z;
            float u = mix(-1.0, aRect.z + 1.0, c.x), v = c.y * (hh + 1.0);
            float dy = -v * cos(th), z = v * sin(th);
            float s = 3.2 * aRect.w / (3.2 * aRect.w - z);
            p = vec2(aRect.x + aRect.z * 0.5 + (u - aRect.z * 0.5) * s, aRect.y + hh + dy * s);
            vL = vec2(u, v);
            w = 1.0 / s;
        } else {
            float m = kind == 0 ? ceil(aRect.z * 0.18) + 2.0 : 1.0;   /* room for the shadow and pins */
            p = aRect.xy - m + c * (aRect.zw + 2.0 * m);
            vL = p - aRect.xy;
        }
        vec2 ndc = vec2(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0);
        gl_Position = vec4(ndc * w, 0.0, w);
    });

static const char *FS = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    uniform vec2 uAtlasSize;
    uniform vec4 uSlot[4];          /* x, y, cw, ch */
    uniform vec3 uInk[7];
    uniform vec3 uFlap[4];          /* card: top outer, top at the split, bottom outer, bottom at the split */
    uniform vec3 uPin[2];           /* axle pin: shaded, lit */
    in vec2 vL;
    flat in vec4 vRect, vWhat, vMore;
    out vec4 o;

    float box(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }

    float glyph(int g, vec2 l) {
        vec4 s = uSlot[int(vMore.z)];
        l = clamp(l, vec2(0.5), s.zw - 0.5);
        return texture(uAtlas, (s.xy + vec2(float(g) * s.z, 0.0) + l) / uAtlasSize).r;
    }

    /* one half of a module's card at l (module px, y down), premultiplied */
    vec4 card(vec2 l, bool top, int g, int ink, float lit) {
        vec2 sz = vRect.zw;
        float hh = sz.y * 0.5, gap = max(1.0, floor(sz.y * 0.035 + 0.5));
        float hl = hh - gap * 0.5;                          /* height of a half */
        vec2 c = vec2(sz.x * 0.5, top ? hl * 0.5 : sz.y - hl * 0.5);
        bool outer = top ? l.y < c.y : l.y > c.y;
        float d = box(l - c, vec2(sz.x * 0.5, hl * 0.5), outer ? sz.x * 0.085 : 1.0);
        float a = clamp(0.5 - d, 0.0, 1.0);
        float e = top ? l.y / hl : (sz.y - l.y) / hl;       /* 0 at the outer edge, 1 at the split */
        vec3 col = top ? mix(uFlap[0], uFlap[1], e) : mix(uFlap[2], uFlap[3], e * e);
        /* no two flaps quite alike: a little wear and tint each */
        float wear = fract(sin(dot(vRect.xy, vec2(12.9898, 78.233))) * 43758.5453);
        col *= 0.94 + 0.12 * wear;
        /* gloss: a soft sheen across the upper card, a fainter one below */
        col += (top ? 0.030 : 0.014) * exp(-pow((e - 0.28) * 4.5, 2.0));
        /* bevel: a lit rim along the outer edge, the sides a touch darker */
        float rim = 1.0 - smoothstep(0.0, 1.6, -d);
        col += (outer && (top ? l.y < 2.5 : false) ? 0.07 : 0.0) * rim;
        col *= 1.0 - 0.25 * rim;
        /* the hinge: the lower card sits in the upper one's shadow */
        if (!top) col *= 1.0 - 0.45 * exp(-(l.y - hh) / (sz.y * 0.035));
        float ga = glyph(g, l);
        vec3 inkc = uInk[ink] * (top ? 1.0 : 0.9) * (1.0 - 0.06 * e);
        col = mix(col, inkc, ga);
        return vec4(col * lit * a, a);
    }

    vec4 over(vec4 top, vec4 under) { return top + under * (1.0 - top.a); }

    void main() {
        int kind = int(vWhat.w);
        vec2 l = vL, sz = vRect.zw;
        if (kind == 3) {                                    /* solid */
            float d = box(l - sz * 0.5, sz * 0.5, vMore.w);
            float a = clamp(0.5 - d, 0.0, 1.0) * (int(vWhat.x) == 6 ? 1.0 : 0.75);
            o = vec4(uInk[int(vWhat.x)] * a, a);
            return;
        }
        if (kind == 2) {                                    /* painted */
            float a = glyph(int(vWhat.x), l);
            o = vec4(uInk[int(vMore.x)] * a, a);
            return;
        }
        float th = vWhat.z, hh = sz.y * 0.5;
        if (kind == 1) {                                    /* the falling flap */
            bool front = th < 1.5707963;
            vec2 ml = vec2(l.x, front ? hh - l.y : hh + l.y);
            /* lit from above and in front */
            vec3 L = normalize(vec3(0.0, -0.55, 0.85));
            vec3 n = vec3(0.0, sin(th), cos(th)) * (front ? 1.0 : -1.0);
            float lit = (0.3 + 0.7 * max(dot(n, L), 0.0)) / (0.3 + 0.7 * L.z);
            o = card(ml, front, int(front ? vWhat.x : vWhat.y), int(front ? vMore.x : vMore.y), lit);
            return;
        }
        /* a module: its shadow, the axle pins, the two halves */
        vec4 res = vec4(0.0);
        float ds = box(l - vec2(sz.x * 0.5, hh + sz.y * 0.035), sz * 0.5, sz.x * 0.1);
        res.a = 0.6 * (1.0 - smoothstep(-sz.x * 0.04, sz.x * 0.12, ds));
        float pw = sz.x * 0.045, ph = max(2.0, sz.y * 0.045);
        float dp = box(vec2(abs(l.x - sz.x * 0.5) - sz.x * 0.5, l.y - hh), vec2(pw, ph * 0.5), ph * 0.5);
        float pa = clamp(0.5 - dp, 0.0, 1.0);
        vec3 pin = mix(uPin[0], uPin[1], smoothstep(ph * 0.5, -ph * 0.5, l.y - hh));
        res = over(vec4(pin * pa, pa), res);
        bool top = l.y < hh;
        float sh = 1.0;
        if (th > 0.0) {
            if (top) sh = 1.0 - 0.5 * (1.0 - th / 3.1415927);       /* just uncovered */
            else sh = 1.0 - 0.45 * smoothstep(0.9, 3.0, th);       /* under the falling flap */
        }
        vec4 cd = card(l, top, int(top ? vWhat.y : vWhat.x), int(top ? vMore.y : vMore.x), sh);
        o = over(cd, res);
    });

/* the board's face behind it all: dark painted metal, a faint grain, the
 * light falling off towards the corners */
static const char *VS_BG = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });
static const char *FS_BG = "#version 330 core\n" GLSL(
    uniform vec2 uRes;
    uniform vec3 uBg;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 q = gl_FragCoord.xy / uRes - 0.5;
        float v = 1.0 - 0.55 * dot(q * vec2(1.1, 1.5), q * vec2(1.1, 1.5));
        float grain = (hash(floor(gl_FragCoord.xy)) - 0.5) * 0.012 + (hash(vec2(floor(gl_FragCoord.y * 0.5), 7.0)) - 0.5) * 0.006;
        o = vec4(uBg * v * (1.0 + 0.1 * q.y) + grain, 1.0);
    });

static GLuint prog, prog_bg, vao, vbo, tex_atlas;
static int tex_w, tex_h;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "splitflap: shader:\n%s\n", log);
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

/* the board's colours: inks (letters, keywords, bad news, good news, labels,
 * rules, trays), the face behind, the flap cards, the axle pins */
typedef struct {
    const char *name;
    const char *ink[INK_N];
    float age;                  /* letters dimmed a touch */
    const char *bg;
    float flap[4][3], pin[2][3];
} Palette;
static const Palette PALETTES[] = {
    /* Aged Brass: cream letters on near-black flaps, gold for the keywords */
    {"brass", {"#EFE3CE", "#FFC332", "#E9994A", "#8FA672", "#A89880", "#D6A12A", "#0D0A08"}, 0.95f, "#1B1511",
     {{0.200f, 0.163f, 0.135f}, {0.158f, 0.128f, 0.106f}, {0.170f, 0.138f, 0.114f}, {0.112f, 0.090f, 0.075f}},
     {{0.10f, 0.085f, 0.07f}, {0.30f, 0.25f, 0.19f}}},
    /* the classic station board: white on charcoal, yellow keywords, orange-red trouble */
    {"classic", {"#F4F4F0", "#FFD400", "#FF5A2C", "#7FD36E", "#8E8E8C", "#5C5C5A", "#060606"}, 1.0f, "#121212",
     {{0.185f, 0.185f, 0.185f}, {0.145f, 0.145f, 0.145f}, {0.155f, 0.155f, 0.155f}, {0.098f, 0.098f, 0.098f}},
     {{0.07f, 0.07f, 0.07f}, {0.32f, 0.32f, 0.32f}}},
    /* white on deep navy, warm yellow-orange keywords */
    {"blue", {"#F2F5FA", "#FFB627", "#FF6B5A", "#7FD8B0", "#8EA2C8", "#3E5E9E", "#030817"}, 1.0f, "#0B1631",
     {{0.085f, 0.130f, 0.255f}, {0.066f, 0.103f, 0.205f}, {0.074f, 0.113f, 0.222f}, {0.046f, 0.072f, 0.148f}},
     {{0.04f, 0.06f, 0.12f}, {0.22f, 0.28f, 0.42f}}},
    /* amber LED: one colour on black, keywords burning brighter */
    {"led", {"#C8601A", "#FFC45C", "#FF4A12", "#E8862A", "#7A4718", "#4A2A0C", "#000000"}, 1.0f, "#050403",
     {{0.075f, 0.070f, 0.066f}, {0.058f, 0.054f, 0.051f}, {0.062f, 0.058f, 0.055f}, {0.036f, 0.034f, 0.032f}},
     {{0.03f, 0.028f, 0.026f}, {0.16f, 0.14f, 0.12f}}},
};
enum { NPAL = sizeof PALETTES / sizeof *PALETTES };
static int opt_pal = -1;        /* -1: chosen from the seed */

static void gl_init(void) {
    prog = mk_prog(VS, FS);
    prog_bg = mk_prog(VS_BG, FS_BG);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    for (int i = 0; i < 3; i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void *)(i * 4 * sizeof(float)));
        glVertexAttribDivisor(i, 1);
    }
    glGenTextures(1, &tex_atlas);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    const Palette *pl = &PALETTES[opt_pal];
    float ink[INK_N * 3], bg[3];
    for (int i = 0; i < INK_N; i++) hex(pl->ink[i], ink + 3 * i);
    for (int i = 0; i < 3; i++) ink[i] *= pl->age;
    glUseProgram(prog);
    glUniform3fv(glGetUniformLocation(prog, "uInk"), INK_N, ink);
    glUniform3fv(glGetUniformLocation(prog, "uFlap"), 4, &pl->flap[0][0]);
    glUniform3fv(glGetUniformLocation(prog, "uPin"), 2, &pl->pin[0][0]);
    glUniform1i(glGetUniformLocation(prog, "uAtlas"), 0);
    hex(pl->bg, bg);
    glUseProgram(prog_bg);
    glUniform3fv(glGetUniformLocation(prog_bg, "uBg"), 1, bg);
}

static void atlas_upload(void) {
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlas_w, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
    tex_w = atlas_w; tex_h = atlas_h;
    float s[SLOT_N * 4];
    for (int i = 0; i < SLOT_N; i++) { s[4 * i] = (float)slot[i].x; s[4 * i + 1] = (float)slot[i].y; s[4 * i + 2] = (float)slot[i].cw; s[4 * i + 3] = (float)slot[i].ch; }
    glUseProgram(prog);
    glUniform4fv(glGetUniformLocation(prog, "uSlot"), SLOT_N, s);
    glUniform2f(glGetUniformLocation(prog, "uAtlasSize"), (float)atlas_w, (float)atlas_h);
}

static float *inst;
static int ninst, capinst, nflap;

static void push(float x, float y, float w, float h, float g0, float g1, float th, float kind, float c0, float c1, float sl, float r) {
    if (ninst >= capinst) inst = realloc(inst, (size_t)(capinst = capinst ? capinst * 2 : 2048) * 12 * sizeof(float));
    float *p = inst + (size_t)ninst++ * 12;
    p[0] = x; p[1] = y; p[2] = w; p[3] = h; p[4] = g0; p[5] = g1; p[6] = th; p[7] = kind;
    p[8] = c0; p[9] = c1; p[10] = sl; p[11] = r;
}

static void render(double t, GLuint target) {
    /* gather: paint, modules at rest, then the falling flaps over them */
    ninst = 0;
    for (int i = 0; i < npaint; i++) {
        Paint *p = &paint[i];
        if (p->kind == 3) push(p->x, p->y, p->w, p->h, (float)p->a, 0, 0, 3, 0, 0, 0, p->r);
        else push(p->x, p->y, p->w, p->h, (float)p->a, 0, 0, 2, (float)p->b, 0, (float)p->c, 0);
    }
    int first_flap = -1;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < nmod; i++) {
            Mod *m = &mods[i];
            mod_settle(m, t);
            int from = m->cur, to = m->cur, c0 = m->ccol, c1 = m->ccol;
            float th = 0;
            if (m->n && t >= m->t0) {
                double f = (t - m->t0) / m->step;
                int k = (int)f;
                float p = (float)(f - k);
                from = (m->cur + k) % m->ns; to = (from + 1) % m->ns;
                c0 = k == 0 ? m->ccol : INK_CREAM;
                c1 = k + 1 == m->n ? m->tcol : INK_CREAM;
                th = 3.1415927f * p * (0.25f + 0.75f * p);   /* falls, faster as it goes */
                if (th < 1e-4f) th = 1e-4f;
            }
            if (pass == 1 && th == 0) continue;
            int g0 = flap_index(FLAPS, (unsigned char)m->seq[from]), g1 = flap_index(FLAPS, (unsigned char)m->seq[to]);
            float w = (float)slot[m->sl].cw, h = (float)slot[m->sl].ch;
            if (pass == 1 && first_flap < 0) first_flap = ninst;
            push(m->x, m->y, w, h, (float)g0, (float)g1, th, (float)pass, (float)c0, (float)c1, (float)m->sl, 0);
        }
    nflap = first_flap < 0 ? 0 : ninst - first_flap;

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glBindVertexArray(vao);
    glUseProgram(prog_bg);
    glUniform2f(glGetUniformLocation(prog_bg, "uRes"), (float)W, (float)H);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    if (tex_w != atlas_w || tex_h != atlas_h || !tex_w) atlas_upload();
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "uRes"), (float)W, (float)H);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)ninst * 12 * sizeof(float), inst, GL_STREAM_DRAW);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, ninst);
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
    int configured, closed, dirty;
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
    wl.dirty = 1;                       /* draw again, even if nothing's moving */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "splitflap");
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

/* dispatch what's waiting, waiting up to timeout_ms for something to come */
static int wl_poll(int timeout_ms) {
    while (wl_display_prepare_read(wl.d) != 0) wl_display_dispatch_pending(wl.d);
    wl_display_flush(wl.d);
    struct pollfd pf = {wl_display_get_fd(wl.d), POLLIN, 0};
    if (poll(&pf, 1, timeout_ms) > 0) wl_display_read_events(wl.d);
    else wl_display_cancel_read(wl.d);
    if (wl_display_dispatch_pending(wl.d) < 0) return 0;
    return !wl.closed;
}

static void usage(void) {
    fprintf(stderr,
        "usage: splitflap [--window | --root | --layer] [--seed N] [--name NAME] [--font FONTCONFIG-PATTERN]\n"
        "                 [--palette brass|classic|blue|led]\n"
        "                 [--shot FILE.ppm | --bench FRAMES] [--size WxH] [--at SECONDS]\n");
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
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-name") && i + 1 < argc) opt_name = argv[++i];
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2 || off_w < 64 || off_h < 64) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            for (opt_pal = 0; opt_pal < NPAL && strcmp(n, PALETTES[opt_pal].name); opt_pal++) {}
            if (opt_pal == NPAL) usage();
        }
        else usage();
    }
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    /* the palette: from the seed (its own hash, so the timetable is the same
     * whichever palette it's drawn in) unless asked for */
    if (opt_pal < 0) opt_pal = (int)(((opt_seed * 0x9E3779B1u) >> 16) % NPAL);
    int offscreen = opt_shot || opt_bench > 0;
    /* a shot: well into the board's life; a bench: the opening, when every
     * module is turning */
    if (opt_at < 0) opt_at = opt_shot ? 25 : 1.5;

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
        XStoreName(dpy, win, "Splitflap");
        XClassHint chint = {"splitflap", "Splitflap"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        if (!ctx) die("can't create a GL context");
        glXMakeCurrent(dpy, win, ctx);
    }
    const char *ver = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
    if (!ver || atof(ver) < 3.3) die("needs OpenGL 3.3");

    font_init();
    live_init();
    live_update(time(NULL));
    gl_init();
    int lw = W, lh = H;
    layout();
    board_init();

    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        /* run the timetable up to --at */
        for (double t = 0; t < opt_at; t += 1.0 / 60) board_update(t, 0);
        board_update(opt_at, 0);
        if (opt_shot) {
            render(opt_at, fbo);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        render(opt_at, fbo);                         /* warm-up */
        glFinish();
        double t0 = now(), worst = 0, gpu = 0;
        long flaps = 0;
        GLuint q;
        glGenQueries(1, &q);
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now(), t = opt_at + (i + 1) / 60.0;
            board_update(t, 0);
            glBeginQuery(GL_TIME_ELAPSED, q);
            render(t, fbo);
            glEndQuery(GL_TIME_ELAPSED);
            glFinish();
            GLuint64 ns = 0;
            glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            gpu += ns * 1e-9;
            flaps += nflap;
            worst = fmax(worst, now() - f0);
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d: %.3f ms/frame avg, %.3f worst (GPU %.3f ms avg), %d modules, %.0f flaps falling on average\n  (%s)\n",
               W, H, per * 1e3, worst * 1e3, gpu / opt_bench * 1e3, nmod, (double)flaps / opt_bench,
               (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    int dirty = 2;
    for (;;) {
        if (opt_layer) {
            if (!wl_poll(0)) return 0;
            if (wl.dirty) { wl.dirty = 0; dirty = 2; }
        }
        while (dpy && XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.window == root) XResizeWindow(dpy, win, ev.xconfigure.width, ev.xconfigure.height);
                else { W = ev.xconfigure.width; H = ev.xconfigure.height; dirty = 2; }
            } else if (ev.type == Expose) {
                dirty = 2;
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape) return 0;
            } else if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == wm_delete) {
                return 0;
            }
        }
        double t = now(), el = t - t0;
        if (W != lw || H != lh) { lw = W; lh = H; board_relayout(el); }
        board_update(el, 0);
        /* nothing turning and nothing to redraw: sleep, waking for events */
        if (!board_busy(el) && !dirty && started) {
            if (opt_layer) { if (!wl_poll(100)) return 0; }
            else {
                struct pollfd pf = {ConnectionNumber(dpy), POLLIN, 0};
                poll(&pf, 1, 100);
            }
            continue;
        }
        if (dirty) dirty--;
        render(el, 0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
