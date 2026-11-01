/*
 * stir — a lock-screen away message, twisted round the middle of the screen
 * after the Midjourney home page, with a clock standing still in the eye.
 *
 * The screen is a grid of monospace cells over a page of away messages
 * ("This is Will's computer. overtime is locked. Away since 14:32. ..."),
 * set solid and word-wrapped, with live details (how long the screen has been
 * locked, the time, battery, uptime) refreshed every minute. The name, the
 * host and the word "away" are set in the accent. --text FILE replaces the
 * messages with your own (same $tokens and *gold* markup, see MSGS below).
 *
 * Each frame every cell looks up the letter it shows by turning its own
 * position, in [-1, 1] across the grid, about the centre by an angle that
 * falls off as 1 / radius:
 *
 *     angle = twist / max(0.1, r)
 *
 * so the middle spins fastest and the edges barely move, and the rows wind
 * into a spiral. Sampling a whole-letter grid along that turned, sheared
 * grid is what drags words out into "ooorrd" smears and gaps; the gold words
 * become gold streaks. Midjourney's twist grows as 0.1 rad/s forever; here
 * it starts at half that (--speed 1 for theirs) and swings between winding one way and the other
 * (--twist MAX radians), so the page comes back to readable now and then.
 *
 * On top: glowing type, warming to the accent towards the fast-spinning
 * middle and sinking into the dark at the edges, phosphor trails, and a
 * curved monitor face with scanlines (--plain for none of that).
 *
 * Colours come from one of a few palettes, picked from the seed so lock
 * sessions differ, or named with --palette: brass (cream and gold on brown),
 * paper (soft grey and white on neutral black), phosphor (green P1 CRT),
 * amber (P3 amber terminal) or ice (blue-white and cyan on blue-black).
 *
 * Over the middle, not twisted, a big HH:MM made of its own digits (like an
 * old banner printout) fades in a few seconds after start, its letters
 * rolling into place, with the date and host beneath.
 *
 * Same plumbing as flightsim: Xlib + GLX for --window / --root, a native
 * wlr-layer-shell surface over EGL for --layer (swaylock-plugin), and
 * --shot FILE.ppm [--at SECONDS] to render one frame offscreen.
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
static int opt_root, opt_layer, opt_clock = 1, opt_plain;
static float opt_cell;                 /* row height in px; 0 = from the screen height */
static float opt_speed = 0.5f;          /* x Midjourney's 0.1 rad/s: half as fast, calmer for a lock screen */
static float opt_twist = 5.0f;         /* the twist swings between +-this (radians at r = 1) */
static const char *opt_font = "JetBrainsMono Nerd Font Mono,monospace:style=Medium";
static const char *opt_name = "Will";
static const char *opt_text;          /* file of messages; default MSGS */
static const char *opt_palette;       /* --palette NAME; default from the seed */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 25;             /* --shot: the frame this many seconds in */
static int off_w = 1920, off_h = 1080;

/* Away messages. Tokens: $N name, $H host, $U login, $T time now, $S locked
 * since, $D how long locked, $DATE today, $B battery, $UP uptime. $N, $H,
 * $S and $D come out gold, as does anything between *stars*. The first
 * three always open the page; the rest are shuffled to fill it. */
static const char *MSGS[] = {
    "This is $N's computer.",
    "$N is *away* from this computer.",
    "$H is locked.",
    "$N has been *away* since $S.",
    "Locked for $D and counting.",
    "Back soon, probably.",
    "Pressing keys won't help, but it won't hurt either.",
    "$N is most likely by the kettle.",
    "If it's urgent, try Slack.",
    "Nothing to see here. Just letters going round.",
    "It is $T. $N was last seen at $S.",
    "Today is $DATE.",
    "$H is guarding the desk.",
    "Please do not feed the laptop.",
    "$N will be right back. Define 'right'.",
    "The cursor is resting. So is $N.",
    "Gone for a walk. Or a coffee. Or a meeting that should have been an email.",
    "Leave a note on paper. Paper still works.",
    "$H has been up for $UP and is in no hurry.",
    "All the tabs are exactly where $N left them.",
    "Unattended computer. Attended desk plant.",
    "Out of office, but only by a few metres.",
    "The tea is probably going cold.",
    "Do not disturb: the letters are busy.",
    "$N is thinking. Possibly about lunch.",
    "No, you can't check your email on this one.",
    "Screensaver on duty since $S.",
    "Please hold. Your call is important to $H.",
    "$N stepped *away* at $S. It is now $T.",
    "The build is still running. Probably.",
    "*Away* from keyboard. Near a kettle.",
    "This computer belongs to $N. The chair is negotiable.",
    "Status: *away*. Mood: optimistic.",
    "Locked at $S. Still locked.",
    "$N will return shortly. Shortly is a relative term.",
    "The fans are spinning. The letters are spinning. $N is elsewhere.",
    "Somewhere, $N has a perfectly good reason for being *away*.",
    "Nobody is typing. That is the point.",
    "Come back later. $H will still be here.",
    "$H has been on its own for $D.",
    "Battery at $B and holding.",
    "The desk is unoccupied. The computer is not impressed.",
    "Messages will be answered in the order they are noticed.",
    "Gone fishing. Metaphorically.",
    "Back in a bit. A bit is somewhere between a minute and an hour.",
    "$N has left the building. The building is fine.",
    "If found, please return $N to $H.",
    "No unsaved work was harmed in the making of this screensaver.",
    "$N is AFK.",
    "Still *away*. Still locked. Still $H.",
    "Last keystroke: $S.",
    "Time *away*: $D.",
    "Every letter here will find its way home.",
    "Do not adjust your set. $N has merely stepped out.",
    "$H is keeping $N's seat warm. Literally: check the fans.",
    "There is nobody at this desk. There is a computer, and it is locked.",
    "The keyboard misses $N. The keyboard is like that.",
    "$N is in a meeting, at lunch, or lost in thought. Possibly all three.",
    "Knock knock. Who's there? Not $N.",
    "This screen will self-unlock when $N returns. Not before.",
    "Idle since $S. Idle is a strong word.",
    "$N is out. $H is in.",
    "Recalculating the whereabouts of $N.",
    "*Away*, but not far.",
    "$U@$H:~$ away",
    "$U@$H:~$ whoami  # not here",
    "$U@$H:~$ uptime  # up $UP",
    "$U@$H:~$ date  # $DATE, $T",
    "$U@$H:~$ last $U  # *away* since $S",
    "$U@$H:~$ who  # nobody, right now",
    "$U@$H:~$ ping $N  # no reply, try again later",
    "$U@$H:~$ cat status  # *away*",
    "$U@$H:~$ upower -b  # $B",
    "$U@$H:~$ finger $N  # on holiday from the keyboard",
    "$U@$H:~$ sleep 60  # $N will be back after this",
    "$U@$H:~$ fg  # nothing happened",
};
#define NMSG ((int)(sizeof MSGS / sizeof *MSGS))

static void die(const char *what) { fprintf(stderr, "stir: %s\n", what); exit(1); }

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

/* ---- live details ---- */
static char host[64], login[64];
static time_t locked_at;
static char d_now[16], d_since[32], d_dur[48], d_date[48], d_batt[48], d_up[48];

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

static void fmt_span(char *out, size_t n, long s) {
    long m = s / 60, h = m / 60, dd = h / 24;
    if (dd >= 1) snprintf(out, n, "%ld day%s, %ld hour%s", dd, dd == 1 ? "" : "s", h % 24, h % 24 == 1 ? "" : "s");
    else if (h >= 1 && m % 60) snprintf(out, n, "%ld hour%s %ld minute%s", h, h == 1 ? "" : "s", m % 60, m % 60 == 1 ? "" : "s");
    else if (h >= 1) snprintf(out, n, "%ld hour%s", h, h == 1 ? "" : "s");
    else if (m >= 1) snprintf(out, n, "%ld minute%s", m, m == 1 ? "" : "s");
    else snprintf(out, n, "less than a minute");
}

static void live_init(void) {
    gethostname(host, sizeof host - 1);
    host[strcspn(host, ".")] = 0;
    const char *u = getenv("USER");
    struct passwd *pw = u ? NULL : getpwuid(getuid());
    snprintf(login, sizeof login, "%s", u ? u : pw ? pw->pw_name : "user");
    locked_at = lock_start();
}

static void live_update(time_t t) {
    struct tm now_tm, since_tm;
    localtime_r(&t, &now_tm);
    localtime_r(&locked_at, &since_tm);
    strftime(d_now, sizeof d_now, "%H:%M", &now_tm);
    if (since_tm.tm_yday == now_tm.tm_yday && since_tm.tm_year == now_tm.tm_year)
        strftime(d_since, sizeof d_since, "%H:%M", &since_tm);
    else
        strftime(d_since, sizeof d_since, "%H:%M on %A", &since_tm);
    fmt_span(d_dur, sizeof d_dur, (long)(t - locked_at));
    strftime(d_date, sizeof d_date, "%A %-d %B", &now_tm);

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
        if (cap[0]) snprintf(d_batt, sizeof d_batt, "%s%%%s", cap, !strcmp(st, "Charging") ? ", charging" : "");
    }
    globfree(&g);
}

/* ---- the text: messages, tokens expanded, as a stream of cells ----
 * Each byte is a printable ASCII char, | 0x80 for gold; '\n' separates
 * messages. */
static unsigned char *stream;
static size_t slen, scap;

static void put(unsigned char c) {
    if (slen + 1 >= scap) stream = realloc(stream, scap = scap ? scap * 2 : 4096);
    stream[slen++] = c;
}

/* append one message, expanding $tokens and *gold*; 0 if it needs a detail
 * this machine doesn't have (no battery) */
static int put_msg(const char *m) {
    size_t mark = slen;
    int gold = 0;
    for (const char *p = m; *p;) {
        if (*p == '*') { gold ^= 0x80; p++; continue; }
        if (*p == '$') {
            static const struct { const char *tok; int gold; } toks[] = {
                {"DATE", 0}, {"UP", 0}, {"N", 1}, {"H", 1}, {"U", 0}, {"T", 0}, {"S", 1}, {"D", 1}, {"B", 0}};
            const char *val = NULL;
            int g = 0;
            for (size_t k = 0; k < sizeof toks / sizeof *toks && !val; k++) {
                size_t n = strlen(toks[k].tok);
                if (strncmp(p + 1, toks[k].tok, n)) continue;
                const char *v[] = {d_date, d_up, opt_name, host, login, d_now, d_since, d_dur, d_batt};
                val = v[k]; g = toks[k].gold ? 0x80 : 0;
                p += 1 + n;
            }
            if (val) {
                if (!*val) { slen = mark; return 0; }
                for (; *val; val++) put((unsigned char)((*val >= 32 && *val < 127 ? *val : '?') | (gold | g)));
                continue;
            }
        }
        unsigned char c = (unsigned char)*p++;
        if (c == '\n' || c == '\t' || c == '\r') c = ' ';
        put((unsigned char)((c >= 32 && c < 127 ? c : '?') | (c == ' ' ? 0 : gold)));
    }
    put('\n');
    return 1;
}

static char *file_text;                /* --text, split into messages on blank lines and sentence ends */

static void load_text_file(void) {
    FILE *f = fopen(opt_text, "rb");
    if (!f) { perror(opt_text); exit(1); }
    size_t cap = 4096, n = 0;
    char *buf = malloc(cap);
    for (size_t r; (r = fread(buf + n, 1, cap - n - 1, f)) > 0;)
        if ((n += r) == cap - 1) buf = realloc(buf, cap *= 2);
    fclose(f);
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        char c = buf[i];
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            int para = c == '\n' && i + 1 < n && buf[i + 1] == '\n';
            int stop = o && strchr(".!?", buf[o - 1]);
            if (para || stop) { if (o && buf[o - 1] != '\n') buf[o++] = '\n'; }
            else if (o && buf[o - 1] != ' ' && buf[o - 1] != '\n') buf[o++] = ' ';
        } else buf[o++] = c;
    }
    while (o && (buf[o - 1] == ' ' || buf[o - 1] == '\n')) o--;
    if (!o) die("the text is empty");
    buf[o] = 0;
    file_text = buf;
}

/* the whole stream, enough to fill the page: the opening messages, then
 * shuffled rounds of the rest. Same seed, same order, so a refresh only
 * changes the live details. */
static void build_stream(size_t need) {
    slen = 0;
    uint32_t keep = rng;
    rng = opt_seed | 1;
    if (file_text) {
        size_t at_wrap = 0;
        for (char *p = file_text; slen < need;) {
            char *e = strchr(p, '\n');
            char msg[4096];
            size_t n = e ? (size_t)(e - p) : strlen(p);
            if (n >= sizeof msg) n = sizeof msg - 1;
            memcpy(msg, p, n); msg[n] = 0;
            put_msg(msg);
            if (!e) {
                if (slen == at_wrap) die("no line of the text can be shown here");
                at_wrap = slen;
            }
            p = e ? e + 1 : file_text;
        }
    } else {
        int order[NMSG];
        for (int i = 0; i < 3; i++) put_msg(MSGS[i]);
        while (slen < need) {
            for (int i = 3; i < NMSG; i++) order[i] = i;
            for (int i = NMSG - 1; i > 3; i--) { int j = 3 + (int)(frand() * (i - 2)); if (j > i) j = i; int t = order[i]; order[i] = order[j]; order[j] = t; }
            for (int i = 3; i < NMSG; i++) put_msg(MSGS[order[i]]);
        }
    }
    rng = keep;
}

/* ---- glyph atlas: printable ASCII rendered once into an R8 strip ---- */
static int cw, ch, nglyph;             /* cell size in px */
static unsigned char *atlas;

static void build_atlas(int cell_h) {
    if (!FcInit()) die("fontconfig init failed");
    FcPattern *pat = FcNameParse((const FcChar8 *)opt_font);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");

    FT_Library ft;
    FT_Face face;
    if (FT_Init_FreeType(&ft) || FT_New_Face(ft, (const char *)file, 0, &face)) die("can't load the font");
    /* set solid, like the original: rows a font-size apart, letters an advance apart */
    FT_Set_Pixel_Sizes(face, 0, (FT_UInt)cell_h);
    FT_Load_Char(face, 'M', FT_LOAD_DEFAULT);
    cw = (int)(face->glyph->advance.x >> 6);
    ch = cell_h;
    int asc = (int)(face->size->metrics.ascender >> 6), desc = (int)(-face->size->metrics.descender >> 6);
    int base = (int)lroundf(ch * (float)asc / (asc + desc));   /* baseline, from the top of the cell */

    nglyph = 95;                                        /* printable ASCII, from ' ' */
    atlas = calloc((size_t)nglyph * cw * ch, 1);
    for (int g = 1; g < nglyph; g++) {
        if (FT_Load_Char(face, (FT_ULong)(32 + g), FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
        FT_GlyphSlot s = face->glyph;
        int x0 = g * cw + s->bitmap_left, y0 = base - s->bitmap_top;
        for (unsigned y = 0; y < s->bitmap.rows; y++)
            for (unsigned x = 0; x < s->bitmap.width; x++) {
                int ax = x0 + (int)x, ay = y0 + (int)y;
                if (ax < g * cw || ax >= (g + 1) * cw || ay < 0 || ay >= ch) continue;
                atlas[(size_t)ay * nglyph * cw + ax] = s->bitmap.buffer[y * s->bitmap.pitch + x];
            }
    }
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
}

/* ---- the page: the stream word-wrapped over the grid, messages two spaces
 * apart; a cell is the glyph index, | 0x80 for gold ---- */
static int cols, rows;
static unsigned char *page;

static void page_build(void) {
    build_stream((size_t)cols * rows * 3 / 2);
    memset(page, 0, (size_t)cols * rows);
    size_t pos = 0;
#define CH(i) (stream[(i) % slen] & 0x7f)
    for (int r = 0; r < rows; r++) {
        int c = 1;                                      /* a one-cell margin */
        for (;;) {
            if (CH(pos) == '\n') { pos++; if (c > 1) c += 2; continue; }   /* messages two spaces apart */
            while (CH(pos) == ' ') pos++;
            size_t w = 0;
            while (CH(pos + w) != ' ' && CH(pos + w) != '\n' && w < slen) w++;
            if (c + (int)w > cols - 1) {
                if (c > 1) break;                       /* wrap onto the next row */
                w = (size_t)(cols - 2);                 /* a word longer than the row: chop it */
            }
            for (size_t k = 0; k < w; k++, pos++) {
                unsigned char s = stream[pos % slen];
                page[r * cols + c++] = (unsigned char)(((s & 0x7f) - 32) | (s & 0x80));
            }
            if (CH(pos) == ' ') c++;
        }
    }
#undef CH
}

/* ---- the clock: HH:MM in 5x9 banner digits, each lit cell its own digit,
 * centred and untwisted, with a caption beneath. A cell of the overlay is
 * 0 = none, 1 = kept clear, 2 + glyph = clock, 0x80 | (2 + glyph) = caption ---- */
static const char *DIGITS[11][9] = {
    {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."},
    {"..#..", ".##..", "#.#..", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"},
    {".###.", "#...#", "....#", "....#", "...#.", "..#..", ".#...", "#....", "#####"},
    {".###.", "#...#", "....#", "....#", "..##.", "....#", "....#", "#...#", ".###."},
    {"...#.", "..##.", ".#.#.", "#..#.", "#..#.", "#####", "...#.", "...#.", "...#."},
    {"#####", "#....", "#....", "####.", "....#", "....#", "....#", "#...#", ".###."},
    {".###.", "#...#", "#....", "#....", "####.", "#...#", "#...#", "#...#", ".###."},
    {"#####", "....#", "....#", "...#.", "..#..", "..#..", ".#...", ".#...", ".#..."},
    {".###.", "#...#", "#...#", "#...#", ".###.", "#...#", "#...#", "#...#", ".###."},
    {".###.", "#...#", "#...#", "#...#", ".####", "....#", "....#", "#...#", ".###."},
    {".", ".", "#", ".", ".", ".", "#", ".", "."},              /* : */
};
static unsigned char *clockov;

static void clock_build(void) {
    memset(clockov, 0, (size_t)cols * rows);
    if (!opt_clock) return;
    const char *hm = d_now;                             /* "HH:MM" */
    /* scale: digits ~18% of the rows tall, square-ish "pixels" */
    int sy = (int)lroundf(rows * 0.18f / 9);
    if (sy < 1) sy = 1;
    int sx, width;
    for (;; sy--) {
        sx = (int)lroundf(sy * (float)ch / cw);
        if (sx < 1) sx = 1;
        width = (4 * 5 + 1 + 4) * sx;                   /* 4 digits, a colon, 4 one-pixel gaps */
        if (width <= cols - 8 || sy == 1) break;
    }
    int height = 9 * sy;
    int x0 = (cols - width) / 2, y0 = (rows - height) / 2 - 1;
    int x = x0;
    for (const char *p = hm; *p; p++) {
        int d = *p == ':' ? 10 : *p - '0';
        if (d < 0 || d > 10) continue;
        int w = (int)strlen(DIGITS[d][0]);
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < w; c++) {
                if (DIGITS[d][r][c] != '#') continue;
                for (int yy = 0; yy < sy; yy++)
                    for (int xx = 0; xx < sx; xx++) {
                        int cx = x + c * sx + xx, cy = y0 + r * sy + yy;
                        if (cx >= 0 && cy >= 0 && cx < cols && cy < rows) clockov[cy * cols + cx] = (unsigned char)(2 + *p - 32);
                    }
            }
        x += (w + 1) * sx;
    }
    /* caption, a blank row under the digits */
    char cap[160];
    snprintf(cap, sizeof cap, "%s  -  %s  -  locked %s", host, d_date, d_since);
    int cl = (int)strlen(cap), cy = y0 + height + 1, cx0 = (cols - cl) / 2;
    if (cy < rows)
        for (int i = 0; i < cl; i++)
            if (cx0 + i >= 0 && cx0 + i < cols && cap[i] != ' ')
                clockov[cy * cols + cx0 + i] = (unsigned char)(0x80 | (2 + cap[i] - 32));
    /* clear a panel round it all, corners rounded off, so it reads */
    int bx0 = cols, by0 = rows, bx1 = -1, by1 = -1;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (clockov[r * cols + c]) {
                if (c < bx0) bx0 = c;
                if (c > bx1) bx1 = c;
                if (r < by0) by0 = r;
                if (r > by1) by1 = r;
            }
    int padx = 2 + 2 * sx, pady = 1 + sy / 2;
    bx0 -= padx; bx1 += padx; by0 -= pady; by1 += pady;
    for (int r = by0; r <= by1; r++)
        for (int c = bx0; c <= bx1; c++) {
            if (r < 0 || c < 0 || r >= rows || c >= cols || clockov[r * cols + c]) continue;
            int ex = c - bx0 < 2 ? 2 - (c - bx0) : bx1 - c < 2 ? 2 - (bx1 - c) : 0;
            int ey = r == by0 || r == by1;
            if (ex && ey) continue;                     /* the corners */
            clockov[r * cols + c] = 1;
        }
}

/* the twist at time t: 0.1 rad/s x --speed to start with, then swinging
 * between +-opt_twist so it never winds itself into noise */
static float twist_at(double t) {
    return opt_twist * (float)sin(0.1 * t / opt_twist);
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });
static const char *FS = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;   /* 95 printable ASCII glyphs, each cw x ch */
    uniform sampler2D uPage;    /* cols x rows: glyph index | 128 for gold, / 255 */
    uniform sampler2D uClock;   /* cols x rows: the clock overlay, see clock_build() */
    uniform vec2 uRes, uCell, uOrigin;
    uniform ivec2 uGrid;
    uniform float uTwist, uClockA, uDepth;
    uniform vec3 uBg, uFg, uGold, uClockCol, uCaption;
    out vec4 o;

    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    float glyph(int gi, ivec2 l) { return texelFetch(uAtlas, ivec2(gi * int(uCell.x) + l.x, l.y), 0).r; }

    void main() {
        vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) - uOrigin;
        ivec2 cell = ivec2(floor(px / uCell));
        o = vec4(uBg, 1.0);
        if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, uGrid))) return;
        ivec2 l = ivec2(px - vec2(cell) * uCell);
        vec3 col = uBg;

        /* the swirl: this cell in [-1, 1] (y up), turned by twist / r */
        vec2 g = vec2(uGrid);
        float p = (float(cell.x) + 0.5) * 2.0 / g.x - 1.0;   /* cell centres, so the */
        float c = 1.0 - (float(cell.y) + 0.5) * 2.0 / g.y;   /* untwisted page maps 1:1 */
        float f = uTwist / max(0.1, length(vec2(p, c)));
        float s = sin(f), k = cos(f);
        float w = p * k + c * s, b = p * s - c * k;
        int pcol = int(floor((w + 1.0) * 0.5 * g.x));
        int prow = int(floor((b + 1.0) * 0.5 * g.y));
        int ov = int(texelFetch(uClock, cell, 0).r * 255.0 + 0.5);
        if (pcol >= 0 && pcol < uGrid.x) {                   /* off the ends of the lines: blank */
            prow = ((prow % uGrid.y) + uGrid.y) % uGrid.y;   /* the rows wrap */
            int v = int(texelFetch(uPage, ivec2(pcol, prow), 0).r * 255.0 + 0.5);
            float a = glyph(v & 127, l);
            if (ov != 0) a *= 1.0 - uClockA;                 /* make way for the clock */
            /* depth: towards the fast-spinning middle the type warms to
             * the accent and brightens, out at the edges it sinks into the dark */
            float rr = length(vec2(p, c));
            float lift = uDepth * (1.0 - smoothstep(0.0, 0.8, rr));
            float fade = mix(1.0, 0.42, uDepth * smoothstep(0.3, 1.35, rr));
            vec3 fg = mix(uFg, uGold, 0.5 * lift) * (1.0 + 0.3 * lift) * fade;
            vec3 gd = uGold * (1.0 + 0.2 * lift) * fade;
            col = mix(col, v >= 128 ? gd : fg, a);
        }

        /* the clock, its letters rolling through the alphabet into place */
        if (ov >= 2) {
            int code = (ov & 127) - 2 + 32;
            float roll = (1.0 - uClockA) * (1.0 - uClockA) * 40.0 * hash(vec2(cell));
            code = 33 + ((code - 33 + int(roll)) % 94);
            float a = glyph(code - 32, l) * smoothstep(0.0, 0.4, uClockA);
            col = mix(col, ov >= 128 ? uCaption : uClockCol, a);
        }
        o = vec4(col, 1.0);
    });

/* phosphor: what's lit stays lit a moment, fading */
static const char *FS_TRAIL = "#version 330 core\n" GLSL(
    uniform sampler2D uScene, uPrev;
    uniform float uDecay;
    out vec4 o;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        o = max(texelFetch(uScene, p, 0), texelFetch(uPrev, p, 0) * uDecay);
    });

/* the monitor: Midjourney's screen curve (easing in over 3 s), a touch of
 * colour fringing towards the edges, glow, scanlines and a vignette */
static const char *FS_CRT = "#version 330 core\n" GLSL(
    uniform sampler2D uAcc;     /* the phosphor, with mips for the glow */
    uniform vec2 uRes;
    uniform float uCurve;
    uniform vec3 uBg;
    out vec4 o;
    void main() {
        vec2 q = gl_FragCoord.xy / uRes * 2.0 - 1.0;
        q *= 1.0 + 0.1 * uCurve;
        q *= 1.0 - 0.085 * uCurve + 0.05 * uCurve * pow(abs(q.yx), vec2(2.0));
        vec2 uv = q * 0.5 + 0.5;
        vec2 px = 1.5 / uRes;                              /* soft glass edge */
        vec2 e = smoothstep(vec2(0.0), px, uv) * smoothstep(vec2(0.0), px, 1.0 - uv);
        if (e.x * e.y <= 0.0) { o = vec4(0.0, 0.0, 0.0, 1.0); return; }

        vec2 d = (uv - 0.5) * 0.0025 * dot(q, q);
        vec3 col = vec3(texture(uAcc, uv + d).r, texture(uAcc, uv).g, texture(uAcc, uv - d).b);
        vec3 glow = textureLod(uAcc, uv, 2.5).rgb * 0.45 + textureLod(uAcc, uv, 4.0).rgb * 0.35 +
                    textureLod(uAcc, uv, 5.5).rgb * 0.3;
        col += max(glow - uBg, 0.0) * 0.55;
        float line = 0.5 + 0.5 * cos(uv.y * uRes.y * 6.2831853 / 3.0);
        col *= 0.84 + 0.16 * line;
        col *= mix(0.5, 1.0, smoothstep(1.75, 0.55, length(q * vec2(0.85, 1.0))));
        o = vec4(col * e.x * e.y, 1.0);
    });

static GLuint prog, prog_trail, prog_crt, tex_atlas, tex_page, tex_clock, vao;
static GLuint tex_scene, fbo_scene, tex_acc[2], fbo_acc[2];
static int post_w, post_h, acc_cur;

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "stir: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

static void grid_init(void) {
    int cell_h = opt_cell > 0 ? (int)opt_cell : (int)(H / 50.0f + 0.5f);
    if (cell_h < 10) cell_h = 10;
    build_atlas(cell_h);
    cols = W / cw + 1; rows = H / ch + 1;   /* overhang; centred below */
    page = calloc((size_t)cols * rows, 1);
    clockov = calloc((size_t)cols * rows, 1);
    if (opt_text) load_text_file();
    live_init();
}

static GLuint mk_tex(int w, int h, const void *data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static GLuint mk_prog(const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, VS));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr);
    return pr;
}

static GLuint mk_target(GLuint *tex, int mips) {
    GLuint f;
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mips ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (mips) glGenerateMipmap(GL_TEXTURE_2D);
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    return f;
}

/* (re)make the offscreen buffers at the window's size */
static void post_size(void) {
    if (post_w == W && post_h == H) return;
    if (post_w) {
        glDeleteTextures(1, &tex_scene); glDeleteFramebuffers(1, &fbo_scene);
        glDeleteTextures(2, tex_acc); glDeleteFramebuffers(2, fbo_acc);
    }
    post_w = W; post_h = H;
    fbo_scene = mk_target(&tex_scene, 0);
    for (int i = 0; i < 2; i++) fbo_acc[i] = mk_target(&tex_acc[i], 1);
}

/* bg, ink (the type, set at 62% over bg), accent (the keywords at 85%, and
 * what the middle warms towards), clock, caption */
static const struct palette { const char *name, *bg, *ink, *accent, *clock, *caption; } palettes[] = {
    {"brass",    "#17120E", "#EFE3CE", "#FFC332", "#FFC332", "#EFE3CE"},   /* Aged Brass */
    {"paper",    "#0F0F10", "#B4B4B2", "#FFFFFF", "#FFFFFF", "#9C9C9A"},   /* Midjourney-ish, neutral */
    {"phosphor", "#07110A", "#33C85E", "#C2FFD0", "#9CFFB4", "#3FCF6A"},   /* P1 green CRT */
    {"amber",    "#120A03", "#D47A10", "#FFD48C", "#FFB040", "#D98A20"},   /* P3 amber terminal */
    {"ice",      "#0A0F18", "#C8D8EC", "#5FE0FF", "#7FE8FF", "#B0C4DC"},   /* blue-white and cyan */
};
#define NPAL ((int)(sizeof palettes / sizeof palettes[0]))
static int pal;

/* --palette NAME, else from the seed (hashed, so the message order the
 * seed also drives isn't tied to the palette) */
static void pick_palette(void) {
    if (opt_palette) {
        for (pal = 0; pal < NPAL; pal++) if (!strcmp(palettes[pal].name, opt_palette)) return;
        fprintf(stderr, "stir: no palette \"%s\" (brass, paper, phosphor, amber, ice)\n", opt_palette);
        exit(1);
    }
    uint32_t h = opt_seed * 2654435761u;
    h ^= h >> 15;
    pal = (int)(h % NPAL);
}

static void gl_init(void) {
    prog = mk_prog(FS);
    prog_trail = mk_prog(FS_TRAIL);
    prog_crt = mk_prog(FS_CRT);
    glGenVertexArrays(1, &vao);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    tex_atlas = mk_tex(nglyph * cw, ch, atlas);
    tex_page = mk_tex(cols, rows, NULL);
    tex_clock = mk_tex(cols, rows, NULL);

    /* the palette: type on the background, the keywords (uGold, also what
     * the middle warms towards), the clock, and the caption under it */
    const struct palette *pl = &palettes[pal];
    float bg[3], ink[3], acc[3], clk[3], cap[3], c[3];
    hex(pl->bg, bg); hex(pl->ink, ink); hex(pl->accent, acc); hex(pl->clock, clk); hex(pl->caption, cap);
    glUseProgram(prog);
    glUniform3fv(glGetUniformLocation(prog, "uBg"), 1, bg);
    for (int i = 0; i < 3; i++) c[i] = ink[i] * 0.62f + bg[i] * 0.38f;
    glUniform3fv(glGetUniformLocation(prog, "uFg"), 1, c);
    for (int i = 0; i < 3; i++) c[i] = acc[i] * 0.85f + bg[i] * 0.15f;
    glUniform3fv(glGetUniformLocation(prog, "uGold"), 1, c);
    glUniform3fv(glGetUniformLocation(prog, "uClockCol"), 1, clk);
    glUniform3fv(glGetUniformLocation(prog, "uCaption"), 1, cap);
    glUniform1i(glGetUniformLocation(prog, "uAtlas"), 0);
    glUniform1i(glGetUniformLocation(prog, "uPage"), 1);
    glUniform1i(glGetUniformLocation(prog, "uClock"), 2);
    glUseProgram(prog_trail);
    glUniform1i(glGetUniformLocation(prog_trail, "uScene"), 0);
    glUniform1i(glGetUniformLocation(prog_trail, "uPrev"), 1);
    glUseProgram(prog_crt);
    glUniform1i(glGetUniformLocation(prog_crt, "uAcc"), 0);
    glUniform3fv(glGetUniformLocation(prog_crt, "uBg"), 1, bg);
}

/* rebuild the page and clock when the minute turns */
static void refresh(time_t t) {
    static time_t last_min = -1;
    if (t / 60 == last_min) return;
    last_min = t / 60;
    live_update(t);
    page_build();
    clock_build();
    glBindTexture(GL_TEXTURE_2D, tex_page);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cols, rows, GL_RED, GL_UNSIGNED_BYTE, page);
    glBindTexture(GL_TEXTURE_2D, tex_clock);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cols, rows, GL_RED, GL_UNSIGNED_BYTE, clockov);
}

/* tw: twist time (scaled by --speed); el: real seconds since start; dt:
 * since the last frame; target: the framebuffer to end up in */
static void render(double tw, double el, double dt, GLuint target) {
    refresh(time(NULL));
    if (!opt_plain) post_size();
    glBindFramebuffer(GL_FRAMEBUFFER, opt_plain ? target : fbo_scene);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_atlas);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_page);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_clock);
    glActiveTexture(GL_TEXTURE0);
    glViewport(0, 0, W, H);
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "uRes"), (float)W, (float)H);
    glUniform2f(glGetUniformLocation(prog, "uCell"), (float)cw, (float)ch);
    glUniform2f(glGetUniformLocation(prog, "uOrigin"), floorf((W - cols * cw) * 0.5f), floorf((H - rows * ch) * 0.5f));
    glUniform2i(glGetUniformLocation(prog, "uGrid"), cols, rows);
    glUniform1f(glGetUniformLocation(prog, "uTwist"), twist_at(tw));
    /* the clock fades in from 2 s to 5 s */
    double a = (el - 2.0) / 3.0;
    a = a < 0 ? 0 : a > 1 ? 1 : a;
    glUniform1f(glGetUniformLocation(prog, "uClockA"), (float)(a * a * (3 - 2 * a)));
    glUniform1f(glGetUniformLocation(prog, "uDepth"), opt_plain ? 0.0f : 1.0f);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    if (opt_plain) return;

    /* phosphor trails: a ghost at half strength, gone in ~0.15 s */
    int prev = acc_cur;
    acc_cur ^= 1;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc[acc_cur]);
    glUseProgram(prog_trail);
    glUniform1f(glGetUniformLocation(prog_trail, "uDecay"), 0.5f * (float)exp(-dt / 0.15));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_scene);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_acc[prev]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_acc[acc_cur]);
    glGenerateMipmap(GL_TEXTURE_2D);

    /* onto the glass */
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glUseProgram(prog_crt);
    glUniform2f(glGetUniformLocation(prog_crt, "uRes"), (float)W, (float)H);
    double cu = el / 3.0 > 1 ? 1 : el / 3.0;
    glUniform1f(glGetUniformLocation(prog_crt, "uCurve"), (float)(cu * (2 - cu)));
    glDrawArrays(GL_TRIANGLES, 0, 3);
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
    /* the grid is sized once; a later resize just recentres it */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "stir");
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
        "usage: stir [--window | --root | --layer] [--cell PX] [--speed X] [--twist RAD] [--seed N]\n"
        "            [--name NAME] [--text FILE] [--no-clock] [--plain] [--font FONTCONFIG-PATTERN]\n"
        "            [--palette brass|paper|phosphor|amber|ice] [--shot FILE.ppm [--size WxH] [--at SECONDS]]\n"
        "  --palette  colours; default: picked from the seed, so each session differs\n");
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
        else if (!strcmp(a, "-cell") && i + 1 < argc) opt_cell = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-twist") && i + 1 < argc) opt_twist = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-text") && i + 1 < argc) opt_text = argv[++i];
        else if (!strcmp(a, "-name") && i + 1 < argc) opt_name = argv[++i];
        else if (!strcmp(a, "-palette") && i + 1 < argc) opt_palette = argv[++i];
        else if (!strcmp(a, "-no-clock")) opt_clock = 0;
        else if (!strcmp(a, "-plain")) opt_plain = 1;
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (opt_twist <= 0) usage();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    pick_palette();

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !opt_shot) wl_init();
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
        if (opt_shot) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Stir");
        XClassHint chint = {"stir", "Stir"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!opt_shot) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    /* a window gets its real size from the WM; size the grid for the
     * largest it's likely to be (the screen), and recentre on resize */
    int gw = W, gh = H;
    if (dpy && !opt_root && !opt_shot) {
        gw = DisplayWidth(dpy, DefaultScreen(dpy));
        gh = DisplayHeight(dpy, DefaultScreen(dpy));
    }
    int sw = W, sh = H;
    W = gw; H = gh;
    grid_init();
    W = sw; H = sh;
    gl_init();

    if (opt_shot) {
        GLuint tex, fbo;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        for (int i = 60; i >= 0; i--) {                 /* the last second, so the trails are real */
            double t = opt_at - i / 60.0;
            render(t * opt_speed, t, 1.0 / 60, fbo);
        }
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }

    double t0 = now(), last = t0;
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
        double t = now(), el = t - t0;
        render(el * opt_speed, el, t - last, 0);
        last = t;
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
