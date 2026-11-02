/*
 * ink — ink dropped into a glass tank of water, as in the macro photographs.
 *
 * A drop falls into the tank, splashes (a small crown and jet on the
 * surface, ripples running off along the meniscus), and goes in as a slug
 * of heavy ink with a burst of downward momentum. That rolls up into a
 * vortex ring, the mushroom cap with its two curled lobes, trailing a stem
 * back to the surface; being denser than the water the ink in the curls
 * sinks on its own, so the ring breaks into smaller rings and fingers that
 * hang under it, and those into smaller ones (Rayleigh-Taylor). Every so
 * often another drop of another ink comes in and folds round what's there.
 * Old ink slowly fades and sinks out of the bottom of the frame, so the
 * water never gets muddy; after a couple of minutes it all clears and the
 * tank starts again.
 *
 * It is simulated: a 2D incompressible fluid on the GPU (semi-Lagrangian
 * velocity, vorticity confinement, Jacobi pressure projection, Boussinesq
 * buoyancy from the ink), with the inks (four, one per channel) carried at
 * three times the velocity's resolution by MacCormack advection so the
 * folds and filaments stay crisp. There are two such slices: a sharp one
 * near the glass and a softer one further back, smaller and out of focus,
 * for depth. The sim steps 30 times a second on a fixed clock (so it is
 * deterministic from the seed and the start time) and is cross-faded
 * between steps; a frame only ever renders.
 *
 * Rendering treats the ink as column density: on a white ground (indigo,
 * sumi, botanical) Beer-Lambert absorption per ink, so colours mix
 * subtractively and thick ink goes deep; on a black ground (neon, sunset)
 * light scattered out of the ink from a softbox above, self-shadowed by
 * marching the blurred density toward the light, with a forward-scattered
 * glow through the thin parts. The glass walls, the meniscus with the
 * underside of the surface mirrored in it, and the strip of air above the
 * water are drawn over it; a small caption gives the time and the time
 * away (since swaylock started, from /proc).
 *
 * Same plumbing as marbling and gauges: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame
 * offscreen, --bench N to time N frames offscreen.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * with --layout (every output's logical rect) and --output = this one's
 * index, as for flightsim. All the outputs look into one wide tank, the
 * layout's bounding box: each instance runs the same simulation of the
 * whole tank (same grid, same steps from the epoch, so they agree without
 * talking) and draws its own part of it. The caption goes on the "front"
 * output, the one under the middle of the layout. --split gives each output
 * a tank of its own instead, the drops staggered between them.
 *
 * Inks come from a palette (indigo, sumi, botanical, neon, sunset), picked
 * from the seed for the session or fixed with --palette NAME.
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

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_split, opt_nocap;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 40;             /* --shot / --bench: from this many seconds in */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the clock starts at */
static const char *opt_font = "EB Garamond";

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;
static int scene_shift;                /* --split: this output plays scenes from this one on */
static double span_toff;               /* and this far into them */

static void die(const char *what) { fprintf(stderr, "ink: %s\n", what); exit(1); }

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
static float frange(float a, float b) { return a + (b - a) * frand(); }

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x ? x : 1;
}

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* ---- palettes ----
 * Four inks each, one per channel of the dye. On a white ground an ink's
 * colour is what a unit of it lets through (its absorption is -ln of it);
 * on a black ground it is the colour it scatters. `w` weights how often
 * each is dropped, so an accent stays an accent. */
typedef struct {
    const char *name;
    int dark;
    const char *bg;                    /* the ground at its brightest */
    const char *ink[4];
    float w[4];
} Palette;

static const Palette PALETTES[] = {
    {"indigo", 0, "#F4F3EF", {"#5468B8", "#3F7FB4", "#59607A", "#8C79C0"}, {1.0f, 0.8f, 0.6f, 0.35f}},
    {"sumi", 0, "#F1ECE1", {"#5A5856", "#68615A", "#5C6170", "#D0604E"}, {1.0f, 0.7f, 0.5f, 0.16f}},
    {"botanical", 0, "#F3F4EE", {"#6FA052", "#3E9A93", "#4F8A6A", "#B8A65A"}, {1.0f, 0.9f, 0.7f, 0.3f}},
    {"neon", 1, "#0A0B10", {"#19D2FF", "#FF2BB8", "#FFE02A", "#8A4DFF"}, {1.0f, 1.0f, 0.7f, 0.45f}},
    {"sunset", 1, "#0C0908", {"#F0203A", "#FF6A12", "#FFC23A", "#C2185B"}, {1.0f, 0.9f, 0.45f, 0.5f}},
};
#define NPAL ((int)(sizeof PALETTES / sizeof *PALETTES))
static const Palette *P;
static int opt_palette = -1;

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

/* ---- the tank, in canvas heights from the bottom left ---- */
#define AIR 0.05f                      /* strip of air above the water */
#define INSET 0.012f                   /* glass walls this far in from the sides */
#define BELOW 0.3f                     /* the water goes on this far below the frame */
static float aspect = 1.6f;            /* canvas width / height */
static float surf_v, wall_l, wall_r;   /* the water surface, the walls */
static float dom[4];                   /* sim domain: x0 y0 x1 y1 */

/* ---- scenes: a timed list of drops, then clear water ---- */
#define SIM_HZ 30.0
#define SCENE 150.0                    /* seconds from clear water to clear water */
#define FADE 12.0                      /* the last of it clearing */
#define MAXDROP 24
typedef struct {
    float t, x, r, speed;              /* impact time, place, radius (canvas heights), entry speed */
    int ink, layer;
    float wob;                         /* how far off straight down it goes in */
} Drop;
typedef struct { int n; Drop d[MAXDROP]; } Scene;

/* the inks come out of a shuffled bag holding each in proportion to its
 * weight, so every scene has its main colours (not, by chance, all accent),
 * and never the same twice running */
static int bag[32], nbag, bagi;
static int pick_ink(int avoid) {
    if (bagi >= nbag) {
        nbag = bagi = 0;
        for (int i = 0; i < 4; i++)
            for (int k = (int)lroundf(P->w[i] * 4); k > 0 && nbag < 32; k--) bag[nbag++] = i;
        for (int i = nbag - 1; i > 0; i--) {
            int j = (int)(frand() * (i + 1)) % (i + 1), t = bag[i];
            bag[i] = bag[j]; bag[j] = t;
        }
    }
    for (int i = bagi; i < nbag; i++)
        if (bag[i] != avoid) { int t = bag[i]; bag[i] = bag[bagi]; bag[bagi] = t; break; }
    return bag[bagi++];
}

static void gen_scene(int idx, Scene *s) {
    idx += scene_shift;
    rng = mix32(opt_seed * 2654435761U ^ (uint32_t)idx * 0x9E3779B9U ^ 0x1c4U);
    memset(s, 0, sizeof *s);
    float tw = wall_r - wall_l, mg = fmaxf(0.08f, (tw < 1.0f ? 0.2f : 0.14f) * tw), t = 1.6f, lo = wall_l + mg, hi = wall_r - mg;
    int last = -1;
    nbag = bagi = 0;
    float lastx = (lo + hi) / 2;
    while (t < SCENE - FADE - 28 && s->n < MAXDROP) {
        int pair = s->n > 0 && frand() < 0.22f;
        for (int k = 0; k <= pair && s->n < MAXDROP; k++) {
            Drop *d = &s->d[s->n++];
            d->t = t + k * frange(0.4f, 1.6f);
            d->ink = s->n == 1 ? (frand() < 0.6f ? 0 : 1) : pick_ink(last);   /* open on a main ink */
            last = d->ink;
            /* the first near the middle; then somewhere not where the last one went in */
            float x = (lo + hi) / 2 + frange(-0.12f, 0.12f) * (hi - lo);
            if (s->n > 1)
                for (int c = 0; c < 6; c++) {
                    x = frange(lo, hi);
                    if (fabsf(x - lastx) > 0.18f * (hi - lo)) break;
                }
            if (k == 1) x = lastx + (frand() < 0.5f ? -1 : 1) * frange(0.05f, 0.12f);
            x = fminf(fmaxf(x, lo), hi);
            d->x = lastx = x;
            d->layer = s->n > 1 && frand() < 0.3f;
            /* a narrow (portrait) tank takes smaller drops, or one fills it */
            d->r = frange(0.018f, 0.03f) * (d->layer ? 0.85f : 1.0f) * fminf(1.0f, fmaxf(0.65f, tw / 1.5f));
            d->speed = frange(0.26f, 0.36f);
            d->wob = frange(-0.18f, 0.18f);
        }
        /* a wider tank (a wall of outputs) gets them more often */
        t += frange(7.0f, 17.0f) * (s->n < 3 ? 0.8f : 1.0f) * fminf(1.0f, 1.6f / tw) * fmaxf(1.0f, 1.1f / tw);
    }
}

/* ---- GL ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

static const char *FS_HEAD = "#version 330 core\n" GLSL(
    uniform vec2 uN;            /* this pass's target size in texels */
    uniform float uDt;
    ivec2 P() { return ivec2(gl_FragCoord.xy); }
    vec2 UV() { return gl_FragCoord.xy / uN; }
    vec4 at(sampler2D s, ivec2 p) { return texelFetch(s, clamp(p, ivec2(0), ivec2(uN) - 1), 0); }
    float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
);

/* vorticity, for confinement */
static const char *FS_CURL = GLSL(
    uniform sampler2D uVel;
    out float o;
    void main() {
        ivec2 p = P();
        o = 0.5 * ((at(uVel, p + ivec2(1, 0)).y - at(uVel, p - ivec2(1, 0)).y) -
                   (at(uVel, p + ivec2(0, 1)).x - at(uVel, p - ivec2(0, 1)).x));
    });

/* advect the velocity and add the forces: vorticity confinement, the ink's
 * weight, the drops' impulses, a slow drift in the tank */
static const char *FS_VEL = GLSL(
    uniform sampler2D uVel, uCurl, uDye;
    uniform float uEps, uBuoy, uDyeLod, uDamp, uSurfDamp, uT;
    uniform vec4 uDrift;        /* amplitude, scale, phases */
    uniform int uNImp;
    uniform vec4 uImp[8];       /* x, y (cells), radius (cells), accel (cells/s^2) */
    uniform vec2 uImpDir[8];
    out vec2 o;
    void main() {
        ivec2 p = P();
        vec2 uv = UV();
        vec2 v = texture(uVel, uv).xy;
        vec2 nv = texture(uVel, uv - uDt * v / uN).xy;
        float c = at(uCurl, p).x;
        vec2 g = 0.5 * vec2(abs(at(uCurl, p + ivec2(1, 0)).x) - abs(at(uCurl, p - ivec2(1, 0)).x),
                            abs(at(uCurl, p + ivec2(0, 1)).x) - abs(at(uCurl, p - ivec2(0, 1)).x));
        g /= length(g) + 1e-5;
        nv += uDt * uEps * vec2(g.y, -g.x) * c;
        float d = dot(textureLod(uDye, uv, uDyeLod), vec4(1.0));
        nv.y -= uDt * uBuoy * d;
        for (int i = 0; i < uNImp; i++) {
            vec2 q = (gl_FragCoord.xy - uImp[i].xy) / uImp[i].z;
            nv += uDt * uImp[i].w * uImpDir[i] * exp(-dot(q, q));
        }
        /* slow convection in the tank: the curl of a few drifting waves */
        vec2 x = gl_FragCoord.xy * uDrift.y;
        float t = uT;
        vec2 f = vec2(cos(x.y * 1.3 + uDrift.z + t * 0.05) * sin(x.x * 0.7 + t * 0.03),
                      -sin(x.y * 1.3 + uDrift.z + t * 0.05) * cos(x.x * 0.7 + t * 0.03) * 0.7 / 1.3)
                + 0.6 * vec2(cos(x.y * 2.1 + uDrift.w - t * 0.04) * sin(x.x * 1.9 + 1.0),
                             -sin(x.y * 2.1 + uDrift.w - t * 0.04) * cos(x.x * 1.9 + 1.0) * 1.9 / 2.1);
        nv += uDt * uDrift.x * f;
        /* the drag of the surface: a vortex near it would run along it */
        float top = (uN.y - gl_FragCoord.y) / uN.y;
        nv *= exp(-uDt * (uDamp + uSurfDamp * smoothstep(0.035, 0.0, top)));
        if (p.x == 0 || p.x == int(uN.x) - 1) nv.x = 0.0;
        if (p.y == 0 || p.y == int(uN.y) - 1) nv.y = 0.0;
        o = nv;
    });

static const char *FS_DIV = GLSL(
    uniform sampler2D uVel;
    out float o;
    void main() {
        ivec2 p = P();
        o = 0.5 * (at(uVel, p + ivec2(1, 0)).x - at(uVel, p - ivec2(1, 0)).x +
                   at(uVel, p + ivec2(0, 1)).y - at(uVel, p - ivec2(0, 1)).y);
    });

static const char *FS_JACOBI = GLSL(
    uniform sampler2D uP, uDiv;
    out float o;
    void main() {
        ivec2 p = P();
        o = 0.25 * (at(uP, p + ivec2(1, 0)).x + at(uP, p - ivec2(1, 0)).x +
                    at(uP, p + ivec2(0, 1)).x + at(uP, p - ivec2(0, 1)).x - at(uDiv, p).x);
    });

/* multigrid: what the pressure leaves of the divergence, summed 2x2 into
 * the next level down, and that level's correction brought back up */
static const char *FS_RESID = GLSL(
    uniform sampler2D uP, uDiv;
    out float o;
    void main() {
        ivec2 p = P();
        o = at(uDiv, p).x - (at(uP, p + ivec2(1, 0)).x + at(uP, p - ivec2(1, 0)).x +
                             at(uP, p + ivec2(0, 1)).x + at(uP, p - ivec2(0, 1)).x - 4.0 * at(uP, p).x);
    });
static const char *FS_RESTRICT = GLSL(
    uniform sampler2D uSrc;
    out float o;
    void main() {
        ivec2 b = 2 * P(), n = textureSize(uSrc, 0);
        float r = 0.0;
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
                ivec2 q = b + ivec2(i, j);
                if (q.x < n.x && q.y < n.y) r += texelFetch(uSrc, q, 0).x;
            }
        o = r;
    });
static const char *FS_PROLONG = GLSL(
    uniform sampler2D uP, uE;
    out float o;
    void main() { o = at(uP, P()).x + texture(uE, gl_FragCoord.xy * 0.5 / vec2(textureSize(uE, 0))).x; });

static const char *FS_GRAD = GLSL(
    uniform sampler2D uVel, uP;
    out vec2 o;
    void main() {
        ivec2 p = P();
        vec2 v = at(uVel, p).xy - 0.5 * vec2(at(uP, p + ivec2(1, 0)).x - at(uP, p - ivec2(1, 0)).x,
                                             at(uP, p + ivec2(0, 1)).x - at(uP, p - ivec2(0, 1)).x);
        if (p.x == 0 || p.x == int(uN.x) - 1) v.x = 0.0;
        if (p.y == 0 || p.y == int(uN.y) - 1) v.y = 0.0;
        o = v;
    });

/* the ink: semi-Lagrangian there (uSign 1) or back (-1) */
static const char *FS_ADV = GLSL(
    uniform sampler2D uSrc, uVel;
    uniform vec2 uVN;           /* velocity grid size */
    uniform float uSign;
    out vec4 o;
    void main() {
        vec2 uv = UV();
        vec2 v = texture(uVel, uv).xy / uVN;
        o = texture(uSrc, uv - uSign * uDt * v);
    });

/* MacCormack's correction, clamped to the neighbourhood it came from, then
 * the drops' ink, and fading */
static const char *FS_DYE = GLSL(
    uniform sampler2D uSrc, uHat, uBack, uVel;
    uniform vec2 uVN;
    uniform float uDecay, uSink, uSinkY, uSeed;
    uniform int uNInj;
    uniform vec4 uInj[8];       /* x, y (texels), radius (texels), amount/s */
    uniform vec4 uInjMask[8];
    out vec4 o;
    void main() {
        vec2 uv = UV();
        vec2 v = texture(uVel, uv).xy / uVN;
        vec2 bp = (uv - uDt * v) * uN - 0.5;
        ivec2 i = ivec2(floor(bp));
        vec4 a = at(uSrc, i), b = at(uSrc, i + ivec2(1, 0)), c = at(uSrc, i + ivec2(0, 1)), d = at(uSrc, i + ivec2(1, 1));
        vec4 lo = min(min(a, b), min(c, d)), hi = max(max(a, b), max(c, d));
        vec4 r = texture(uHat, uv) + 0.5 * (at(uSrc, P()) - texture(uBack, uv));
        r = clamp(r, lo, hi);
        for (int k = 0; k < uNInj; k++) {
            vec2 q = (gl_FragCoord.xy - uInj[k].xy) / uInj[k].z;
            float n = 0.9 + 0.2 * sin(q.x * 2.3 + q.y * 1.7 + uSeed);
            r += uDt * uInj[k].w * uInjMask[k] * smoothstep(1.0, 0.55, length(q)) * n;
        }
        float y = gl_FragCoord.y / uN.y;
        r *= exp(-uDt * (uDecay + uSink * smoothstep(uSinkY, 0.0, y)));
        o = max(r, 0.0);
    });

/* this frame's ink in a slice: the last two steps cross-faded */
static const char *FS_NOW = GLSL(
    uniform sampler2D uSrc0, uSrc1;
    uniform float uMix;
    out vec4 o;
    void main() { o = mix(at(uSrc0, P()), at(uSrc1, P()), uMix); });

/* ---- the picture ----
 * In two passes: the water and the ink, at a reduced scale (FINAL not
 * defined), then at full size that scaled up with the glass, the surface,
 * the drops and the splashes drawn crisply over it (FINAL).
 *
 * A slice is taken as a sheet of ink curved round the axis of the drop
 * nearest: within uZ0 of it, a pixel looks through the sheet revolved about
 * that axis (so the curls of a ring are seen as a torus, a cap as a dome,
 * the fingers under it as a chandelier), tilted as if from a little above;
 * further out, and towards halfway to the next drop, it flattens back into
 * the plain sheet. */
static const char *FS_SHOW = GLSL(
    uniform sampler2D uA, uB;   /* near and far slices: now (no mips) */
    uniform sampler2D uA1, uB1; /* and the latest step (with mips) */
    uniform sampler2D uLow;     /* FINAL: the first pass */
    uniform vec4 uDomA, uDomB;
    uniform float uAxA[8], uAxB[8];
    uniform int uNAxA, uNAxB;
    uniform float uZ0, uTilt;
    uniform float uFade;
    uniform vec4 uMap;          /* canvas position = (x + y fc.x, z + w fc.y) */
    uniform vec3 uTank;         /* wall_l, wall_r, surface */
    uniform vec2 uCam;          /* the eye's axis */
    uniform float uZoom;        /* the far slice is this much smaller */
    uniform int uDark;
    uniform vec3 uBg;
    uniform vec3 uSig[4], uCol[4];
    uniform vec2 uDyeN;
    uniform int uNVis;
    uniform vec4 uVis[8];       /* drops: x, time since impact, radius, ink */
    uniform float uPx;          /* an output pixel, in canvas heights */
    uniform float uAspect;
    uniform vec2 uLowScale;     /* FINAL: the first pass's size / ours */
    out vec4 o;

    float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }

    vec4 tap(int k, vec2 q, float lod) {
        vec4 dm = k == 0 ? uDomA : uDomB;
        vec2 uv = (q - dm.xy) / (dm.zw - dm.xy);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
        if (lod < 0.5) return k == 0 ? textureLod(uA, uv, 0.0) : textureLod(uB, uv, 0.0);
        return k == 0 ? textureLod(uA1, uv, lod) : textureLod(uB1, uv, lod);
    }

    /* the ink seen through slice k at q */
    vec4 column(int k, vec2 q, float lod, int nz, float jit) {
        vec4 pl = tap(k, q, lod);
        /* a touch of unsharp mask: the folds as crisp as the photographs' */
        if (lod < 0.5) pl = max(pl + 0.7 * (pl - tap(k, q, 1.3)), 0.0);
        int n = k == 0 ? uNAxA : uNAxB;
        float d1 = 1e9, d2 = 1e9, a1 = q.x;
        for (int i = 0; i < 8; i++) {
            if (i >= n) break;
            float ax = k == 0 ? uAxA[i] : uAxB[i];
            float d = abs(q.x - ax);
            if (d < d1) { d2 = d1; d1 = d; a1 = ax; } else if (d < d2) d2 = d;
        }
        float zm = uZ0 * smoothstep(0.0, 1.0, min(0.5 * (d2 - d1), 4.0 * uZ0 - d1) / uZ0);
        if (zm < 0.003) return pl;
        float dx = q.x - a1;
        vec4 acc = vec4(0.0);
        for (int i = 0; i < 8; i++) {
            if (i >= nz) break;
            float z = ((float(i) + jit) / float(nz) * 2.0 - 1.0) * zm;
            float r = sqrt(dx * dx + z * z);
            float y = q.y + z * uTilt;
            acc += mix(tap(k, vec2(a1 + r, y), lod + 0.7), tap(k, vec2(a1 - r, y), lod + 0.7), 0.5 - 0.5 * dx / max(r, 1e-6));
        }
        float w = zm / uZ0;
        return mix(pl, 0.55 * pl + acc / float(nz) * 1.1, w);
    }

    /* one slice over what's behind it */
    vec3 slice(int k, vec2 q, float lod, float dim, vec3 col, float jit) {
        vec4 c = column(k, q, lod, k == 0 ? 8 : 5, jit) * uFade;
        /* the far slice's own surface is out of sight: let its ink go before it */
        if (k == 1) c *= smoothstep(uTank.z, uTank.z - 0.03, q.y);
        float tot = dot(c, vec4(1.0));
        if (tot < 1e-4) return col;
        vec4 dm = k == 0 ? uDomA : uDomB;
        vec3 tau = c.x * uSig[0] + c.y * uSig[1] + c.z * uSig[2] + c.w * uSig[3];
        /* light from a softbox up and to the left: how much ink lies between */
        vec2 L = normalize(vec2(-0.35, 1.0));
        float sh = 0.0;
        for (int i = 1; i <= 4; i++) sh += dot(tap(k, q + L * 0.008 * float(i * i), 3.0), vec4(1.0)) * float(i);
        float lit = exp(-sh * 0.07 * uFade);
        /* the cloud's surface, from the blurred density's slope */
        vec2 e = vec2(2.0 / uDyeN.y, 0.0) * (dm.w - dm.y);
        float dl = 0.0, dr = 0.0, dd = 0.0, du = 0.0;
        if (k == 0) {           /* the far slice is too soft to show it */
            dl = log(1.0 + dot(tap(k, q - e.xy, 1.5), vec4(1))); dr = log(1.0 + dot(tap(k, q + e.xy, 1.5), vec4(1)));
            dd = log(1.0 + dot(tap(k, q - e.yx, 1.5), vec4(1))); du = log(1.0 + dot(tap(k, q + e.yx, 1.5), vec4(1)));
        }
        vec3 n = normalize(vec3(dl - dr, dd - du, 0.35));
        float lam = clamp(dot(n, normalize(vec3(-0.35, 1.0, 0.6))), 0.0, 1.0);
        if (uDark == 0) {
            vec3 T = exp(-tau);
            float a = 1.0 - exp(-tot * 0.6);
            vec3 ink = (c.x * exp(-uSig[0]) + c.y * exp(-uSig[1]) + c.z * exp(-uSig[2]) + c.w * exp(-uSig[3])) / tot;
            return col * T + a * ink * (0.10 * lit * lam) * dim;
        }
        vec3 sc = (c.x * uCol[0] + c.y * uCol[1] + c.z * uCol[2] + c.w * uCol[3]) / tot;
        /* faint old ink sinks back into the dark; dense new ink stands out */
        float a = 1.0 - exp(-1.4 * tot * tot / (tot + 0.2));
        float lt2 = exp(-sh * 0.02 * uFade);
        float thin = tot * exp(-tot * 1.2);
        float edge = clamp(length(vec2(dl - dr, dd - du)) * 1.5, 0.0, 1.0);
        vec3 l = sc * a * (0.1 + 2.6 * lt2 * (0.25 + 0.75 * lam)) + sc * thin * 0.6 * lt2 + sc * edge * a * 0.5 * lt2;
        /* a glow round it, light scattered on through the water */
        vec4 g = tap(k, q, 4.5) * uFade;
        vec3 glow = (g.x * uCol[0] + g.y * uCol[1] + g.z * uCol[2] + g.w * uCol[3]) * 0.05;
        return col * exp(-tau * 0.8) + (l + glow) * dim;
    }

    float surfAt(float x) {
        float s = uTank.z;
        /* meniscus: the water climbs the glass */
        s += 0.003 * (exp(-(x - uTank.x) / 0.004) + exp(-(uTank.y - x) / 0.004));
        for (int i = 0; i < uNVis; i++) {
            float ts = uVis[i].y;
            if (ts < 0.0 || ts > 6.0) continue;
            float d = abs(x - uVis[i].x);
            s += 0.0035 * exp(-ts * 0.9) * sin(140.0 * d - 9.0 * ts) * exp(-d * 9.0) * smoothstep(0.0, 0.02, 0.12 * ts + 0.01 - d);
        }
        return s;
    }

    vec3 ground(vec2 q) {
        /* a softbox, brightest a little above the middle */
        vec2 g = (q - vec2(uAspect * 0.5, 0.62)) / vec2(max(uAspect, 1.0), 1.0);
        float fall = exp(-dot(g, g) * 1.6);
        return uDark == 1 ? uBg * (0.35 + 1.6 * fall) : uBg * (0.93 + 0.07 * fall);
    }

    void main() {
        vec2 fc = gl_FragCoord.xy;
        vec2 q = vec2(uMap.x + uMap.y * fc.x, uMap.z + uMap.w * fc.y);
        float s = surfAt(q.x);
) "\n#ifndef FINAL\n" GLSL(
        vec3 bg = ground(q), col = bg;
        bool water = q.y < s && q.x > uTank.x && q.x < uTank.y;
        if (water) {
            float jit = 0.5;
            /* water takes a little light */
            col *= uDark == 1 ? vec3(0.92, 0.97, 1.0) : vec3(0.975, 0.99, 0.995);
            vec2 qb = uCam + (q - uCam) / uZoom;
            col = slice(1, qb, 1.0, 0.8, col, jit);
            /* the far slice sits in more water */
            col = mix(col, bg * (uDark == 1 ? vec3(0.9, 0.95, 1.0) : vec3(0.97, 0.985, 0.99)), 0.12);
            col = slice(0, q, 0.0, 1.0, col, jit);
            /* the underside of the surface, a mirror for a moment below it */
            float bd = (s - q.y) / 0.014;
            if (bd < 1.0) {
                vec3 m = slice(0, vec2(q.x, s + (s - q.y) * 2.5 - 0.03), 1.0, 1.0, bg, jit);
                col = mix(col, m * (uDark == 1 ? 0.8 : 0.95), 0.45 * (1.0 - bd) * (1.0 - bd));
            }
        }
        o = vec4(col, 1.0);
) "\n#else\n" GLSL(
        vec3 col = texture(uLow, fc * uLowScale / vec2(textureSize(uLow, 0))).rgb;
        /* the surface line, seen nearly edge on */
        float ld = abs(q.y - s) / uPx;
        if (q.x > uTank.x && q.x < uTank.y) {
            float line = exp(-ld * ld * 0.25);
            col = uDark == 1 ? col + vec3(0.10, 0.11, 0.12) * line : col * (1.0 - 0.28 * line);
            float glint = exp(-pow((q.y - s - 1.6 * uPx) / uPx, 2.0) * 0.5);
            col = uDark == 1 ? col + vec3(0.05) * glint : mix(col, vec3(1.0), 0.35 * glint);
        }
        /* the glass walls, seen edge on: a thick green-grey band with bright faces */
        for (int k = 0; k < 2; k++) {
            float wx = k == 0 ? uTank.x : uTank.y;
            float d = (k == 0 ? wx - q.x : q.x - wx) / 0.006;       /* 0 at the inner face, 1 outer */
            if (d > -0.3 && d < 1.3) {
                float band = smoothstep(-0.05, 0.05, d) * smoothstep(1.05, 0.95, d);
                vec3 glass = uDark == 1 ? vec3(0.05, 0.075, 0.07) : vec3(0.80, 0.86, 0.83);
                col = mix(col, col * glass * (uDark == 1 ? 12.0 : 1.0), 0.2 * band);
                float face = exp(-pow(d * 0.006 / uPx, 2.0) * 0.4) + 0.6 * exp(-pow((d - 1.0) * 0.006 / uPx, 2.0) * 0.4);
                col = uDark == 1 ? col + vec3(0.04, 0.05, 0.048) * face : col * (1.0 - 0.1 * face);
            }
        }
        /* drops falling in, and their splashes */
        for (int i = 0; i < uNVis; i++) {
            float ts = uVis[i].y, r = uVis[i].z;
            int k = int(uVis[i].w);
            vec3 ic = uDark == 1 ? uCol[k] * 0.9 : exp(-uSig[k] * 3.0);
            if (ts < 0.0) {
                /* falling: slow motion, the drop a little stretched */
                float rd = 0.0045 + r * 0.12;
                vec2 c = vec2(uVis[i].x, s + rd - ts * 0.16);
                vec2 dq = (q - c) / vec2(rd, rd * 1.25);
                float m = smoothstep(1.0, 1.0 - 2.0 * uPx / rd, length(dq));
                float hl = exp(-dot(dq - vec2(-0.35, 0.4), dq - vec2(-0.35, 0.4)) * 18.0);
                col = mix(col, ic, m) + m * hl * 0.25;
            } else if (ts < 1.4) {
                /* the jet thrown back up, and a drop off its tip */
                float hgt = 0.022 * sin(3.14159 * clamp((ts - 0.05) / 1.1, 0.0, 1.0));
                float x = q.x - uVis[i].x, y = q.y - s;
                float wj = 0.0016 + 0.001 * (1.0 - y / max(hgt, 1e-4));
                float m = step(0.0, y) * step(y, hgt) * smoothstep(wj, wj - 1.5 * uPx, abs(x));
                vec2 tip = vec2(x, y - hgt - 0.006 * smoothstep(0.3, 1.0, ts));
                m = max(m, smoothstep(0.0022, 0.0022 - 1.5 * uPx, length(tip)) * step(0.25, ts));
                vec3 wc = uDark == 1 ? col + 0.12 : col * 0.86;
                col = mix(col, wc, m * 0.8);
            }
        }
        /* the film's shoulder: bright ink rolls off rather than clipping */
        if (uDark == 1) col = mix(col, 0.75 + 0.25 * (1.0 - exp(-(col - 0.75) / 0.25)), step(0.75, col));
        /* a little lens: vignette and grain */
        vec2 vq = (q - vec2(uAspect * 0.5, 0.5)) / vec2(max(uAspect, 1.0), 1.0);
        col *= 1.0 - (uDark == 1 ? 0.25 : 0.12) * dot(vq, vq);
        col += (hash(fc) - 0.5) * (uDark == 1 ? 0.012 : 0.02);
        o = vec4(col, 1.0);
) "\n#endif\n}\n";

static GLuint compile(GLenum type, const char *a, const char *b) {
    GLuint s = glCreateShader(type);
    const char *src[2] = {a, b ? b : ""};
    glShaderSource(s, 2, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "ink: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_link(GLuint vs, GLuint fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, vs);
    glAttachShader(pr, fs);
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}

static GLuint vs_full;
static GLuint mk_sim(const char *fs) { return mk_link(vs_full, compile(GL_FRAGMENT_SHADER, FS_HEAD, fs)); }
#define U(p, n) glGetUniformLocation(p, n)

/* ---- the simulation: a slice of the tank ---- */
#define NLEV 3
typedef struct {
    int nx, ny, f, dx, dy;             /* velocity grid, ink resolution factor, ink grid */
    float cph;                         /* velocity cells per canvas height */
    float dom[4];                      /* its domain, in canvas heights */
    GLuint vel[2], curl, dye[3], tmp[2], now;
    GLuint fvel[2], fcurl, fdye[3], ftmp[2], fnow;
    struct {                           /* the pressure, at each multigrid level */
        int w, h, c;
        GLuint p[2], rhs, res, fp[2], frhs, fres;
    } lv[NLEV];
    int v, prev, cur;                  /* which of each is current */
} Slice;
static Slice sl[2];                    /* near, far */

static GLuint prog_resid, prog_restrict, prog_prolong;
static GLuint prog_final, prog_now, prog_curl, prog_vel, prog_div, prog_jac, prog_grad, prog_adv, prog_dye, prog_show;
static GLuint vao;

static void tex_fbo(GLuint *tex, GLuint *fbo, int w, int h, GLenum ifmt, GLenum fmt, int mips) {
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    int levels = 1;
    if (mips) for (int m = w > h ? w : h; m > 1; m >>= 1) levels++;
    for (int l = 0, lw = w, lh = h; l < levels; l++, lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1)
        glTexImage2D(GL_TEXTURE_2D, l, ifmt, lw, lh, 0, fmt, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels - 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mips ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("float render targets unsupported");
}

static void slice_free(Slice *s) {
    if (!s->nx) return;
    glDeleteTextures(9, &s->vel[0]);
    glDeleteFramebuffers(9, &s->fvel[0]);
    for (int l = 0; l < NLEV; l++) {
        glDeleteTextures(4, &s->lv[l].p[0]);
        glDeleteFramebuffers(4, &s->lv[l].fp[0]);
    }
    s->nx = 0;
}

static void slice_init(Slice *s, float cells, int f) {
    slice_free(s);
    float dw = s->dom[2] - s->dom[0], dh = s->dom[3] - s->dom[1];
    float h = sqrtf(dw * dh / cells);
    s->nx = (int)lroundf(dw / h); s->ny = (int)lroundf(dh / h);
    s->cph = s->ny / dh;
    s->f = f; s->dx = s->nx * f; s->dy = s->ny * f;
    for (int i = 0; i < 2; i++) tex_fbo(&s->vel[i], &s->fvel[i], s->nx, s->ny, GL_RG16F, GL_RG, 0);
    tex_fbo(&s->curl, &s->fcurl, s->nx, s->ny, GL_R16F, GL_RED, 0);
    for (int l = 0, w = s->nx, h = s->ny; l < NLEV; l++, w = (w + 1) / 2, h = (h + 1) / 2) {
        s->lv[l].w = w; s->lv[l].h = h;
        for (int i = 0; i < 2; i++) tex_fbo(&s->lv[l].p[i], &s->lv[l].fp[i], w, h, GL_R32F, GL_RED, 0);
        tex_fbo(&s->lv[l].rhs, &s->lv[l].frhs, w, h, GL_R32F, GL_RED, 0);
        tex_fbo(&s->lv[l].res, &s->lv[l].fres, w, h, GL_R32F, GL_RED, 0);
    }
    for (int i = 0; i < 3; i++) tex_fbo(&s->dye[i], &s->fdye[i], s->dx, s->dy, GL_RGBA16F, GL_RGBA, 1);
    for (int i = 0; i < 2; i++) tex_fbo(&s->tmp[i], &s->ftmp[i], s->dx, s->dy, GL_RGBA16F, GL_RGBA, 0);
    tex_fbo(&s->now, &s->fnow, s->dx, s->dy, GL_RGBA16F, GL_RGBA, 0);
}

static void slice_clear(Slice *s) {
    static const float z[4] = {0, 0, 0, 0};
    GLuint *f = &s->fvel[0];
    for (int i = 0; i < 9; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, f[i]);
        glClearBufferfv(GL_COLOR, 0, z);
    }
    for (int l = 0; l < NLEV; l++)
        for (int i = 0; i < 4; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, (&s->lv[l].fp[0])[i]);
            glClearBufferfv(GL_COLOR, 0, z);
        }
    for (int i = 0; i < 3; i++) {
        glBindTexture(GL_TEXTURE_2D, s->dye[i]);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    for (int l = 0; l < NLEV; l++) s->lv[l].c = 0;
    s->v = 0; s->prev = 0; s->cur = 1;
}

static void bind(int unit, GLuint tex) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, tex);
}

static void pass(GLuint prog, GLuint fbo, int w, int h) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    glUniform2f(U(prog, "uN"), (float)w, (float)h);
    glUniform1f(U(prog, "uDt"), (float)(1.0 / SIM_HZ));
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* ---- the scene's clock ---- */
static Scene scene;
static int scene_idx = -1;
static long steps;                     /* done in this scene */
static float drift[4];

#define ZOOM 0.86f                     /* the far slice looks this much smaller */
#define CAMX (aspect * 0.5f)
#define CAMY 0.8f
/* where a drop goes in, in its slice's own coordinates: a far one is placed
 * so it lands where the near one would have on screen */
static float drop_x(const Drop *d) { return d->layer ? CAMX + (d->x - CAMX) / ZOOM : d->x; }

#define IMPULSE 0.30f                  /* the slug of ink going in, seconds */
#define TAIL 2.2f
#define RING 2.6f                      /* the ring helped along this long */                      /* the rest draining off the surface */

static void jacobi(Slice *s, int l, int n) {
    glUseProgram(prog_jac);
    bind(1, s->lv[l].rhs);
    for (int i = 0; i < n; i++) {
        bind(0, s->lv[l].p[s->lv[l].c]);
        pass(prog_jac, s->lv[l].fp[1 - s->lv[l].c], s->lv[l].w, s->lv[l].h);
        s->lv[l].c ^= 1;
    }
}

/* the pressure, by a V-cycle from last step's: smooth, hand what's left
 * down a level to be solved there (where long waves take few sweeps), add
 * that back, smooth again */
static void vcycle(Slice *s, int l) {
    static const int pre[NLEV] = {4, 4, 0}, post[NLEV] = {6, 6, 40};
    jacobi(s, l, pre[l]);
    if (l + 1 < NLEV) {
        glUseProgram(prog_resid);
        bind(0, s->lv[l].p[s->lv[l].c]); bind(1, s->lv[l].rhs);
        pass(prog_resid, s->lv[l].fres, s->lv[l].w, s->lv[l].h);
        glUseProgram(prog_restrict);
        bind(0, s->lv[l].res);
        pass(prog_restrict, s->lv[l + 1].frhs, s->lv[l + 1].w, s->lv[l + 1].h);
        static const float z[4] = {0, 0, 0, 0};
        glBindFramebuffer(GL_FRAMEBUFFER, s->lv[l + 1].fp[s->lv[l + 1].c]);
        glClearBufferfv(GL_COLOR, 0, z);
        vcycle(s, l + 1);
        glUseProgram(prog_prolong);
        bind(0, s->lv[l].p[s->lv[l].c]); bind(1, s->lv[l + 1].p[s->lv[l + 1].c]);
        pass(prog_prolong, s->lv[l].fp[1 - s->lv[l].c], s->lv[l].w, s->lv[l].h);
        s->lv[l].c ^= 1;
    }
    jacobi(s, l, post[l]);
}

static void step_slice(Slice *s, int layer, double t) {
    float dt = (float)(1.0 / SIM_HZ);
    /* the drops going in now */
    float imp[8][4], dir[8][2], inj[8][4], msk[8][4];
    int ni = 0, nj = 0;
    for (int i = 0; i < scene.n; i++) {
        Drop *d = &scene.d[i];
        if (d->layer != layer) continue;
        float ts = (float)t - d->t;
        if (ts < 0 || ts > RING) continue;
        float cx = (drop_x(d) - s->dom[0]) * s->cph, top = (surf_v - s->dom[1]) * s->cph;
        float rc = d->r * s->cph;
        if (ts < RING && ni < 8) {
            /* the slug's momentum, then (standing in for the third
             * dimension, which keeps a ring going far longer than a 2D
             * pair) a dying push that follows the ring down */
            float u = d->speed * s->cph, a = d->wob * 0.25f;
            float fall = u * 0.55f * 2.5f * (1.0f - expf(-ts / 2.5f));
            imp[ni][0] = cx + d->wob * rc * 0.5f + sinf(a) * fall; imp[ni][1] = top - rc * 1.5f - cosf(a) * fall;
            imp[ni][2] = rc * (1.1f + 0.25f * ts);
            imp[ni][3] = ts < IMPULSE ? u / IMPULSE * 1.4f * (1.0f - ts / IMPULSE * 0.6f)
                                      : u * 0.9f * expf(-(ts - IMPULSE) / 0.9f);
            dir[ni][0] = sinf(a); dir[ni][1] = -cosf(a);
            ni++;
        }
        if (nj < 8 && ts < TAIL) {
            float fd = (float)s->f;
            float amt, rad, y;
            if (ts < IMPULSE) { amt = 8.0f / IMPULSE; rad = rc; y = top - rc * 1.3f - d->speed * s->cph * ts * 0.6f; }
            else { amt = 0.45f * expf(-(ts - IMPULSE) * 2.2f); rad = rc * 0.35f; y = top - rc * 0.4f; }
            inj[nj][0] = (cx + d->wob * rc * 0.3f) * fd; inj[nj][1] = y * fd; inj[nj][2] = rad * fd; inj[nj][3] = amt;
            for (int c = 0; c < 4; c++) msk[nj][c] = c == d->ink;
            nj++;
        }
    }
    glBindVertexArray(vao);

    glUseProgram(prog_curl);
    bind(0, s->vel[s->v]);
    pass(prog_curl, s->fcurl, s->nx, s->ny);

    glUseProgram(prog_vel);
    bind(0, s->vel[s->v]); bind(1, s->curl); bind(2, s->dye[s->cur]);
    glUniform1f(U(prog_vel, "uEps"), 2.2f);
    glUniform1f(U(prog_vel, "uBuoy"), 0.16f * s->cph);
    glUniform1f(U(prog_vel, "uDyeLod"), log2f((float)s->f));
    glUniform1f(U(prog_vel, "uDamp"), 0.08f);
    glUniform1f(U(prog_vel, "uSurfDamp"), 1.2f);
    glUniform1f(U(prog_vel, "uT"), (float)t);
    glUniform4f(U(prog_vel, "uDrift"), 0.004f * s->cph, 6.0f / s->cph, drift[layer * 2], drift[layer * 2 + 1]);
    glUniform1i(U(prog_vel, "uNImp"), ni);
    if (ni) { glUniform4fv(U(prog_vel, "uImp"), ni, &imp[0][0]); glUniform2fv(U(prog_vel, "uImpDir"), ni, &dir[0][0]); }
    pass(prog_vel, s->fvel[1 - s->v], s->nx, s->ny);
    s->v = 1 - s->v;

    glUseProgram(prog_div);
    bind(0, s->vel[s->v]);
    pass(prog_div, s->lv[0].frhs, s->nx, s->ny);
    vcycle(s, 0);

    glUseProgram(prog_grad);
    bind(0, s->vel[s->v]); bind(1, s->lv[0].p[s->lv[0].c]);
    pass(prog_grad, s->fvel[1 - s->v], s->nx, s->ny);
    s->v = 1 - s->v;

    /* the ink: there, back, corrected */
    glUseProgram(prog_adv);
    glUniform2f(U(prog_adv, "uVN"), (float)s->nx, (float)s->ny);
    bind(1, s->vel[s->v]);
    bind(0, s->dye[s->cur]);
    glUniform1f(U(prog_adv, "uSign"), 1.0f);
    pass(prog_adv, s->ftmp[0], s->dx, s->dy);
    bind(0, s->tmp[0]);
    glUniform1f(U(prog_adv, "uSign"), -1.0f);
    pass(prog_adv, s->ftmp[1], s->dx, s->dy);

    int nxt = 3 - s->prev - s->cur;
    glUseProgram(prog_dye);
    bind(0, s->dye[s->cur]); bind(1, s->tmp[0]); bind(2, s->tmp[1]); bind(3, s->vel[s->v]);
    glUniform2f(U(prog_dye, "uVN"), (float)s->nx, (float)s->ny);
    double ls = t;
    float decay = 1.0f / 50.0f;
    if (ls > SCENE - FADE) decay += 0.15f;
    glUniform1f(U(prog_dye, "uDecay"), decay);
    glUniform1f(U(prog_dye, "uSink"), 1.2f);
    glUniform1f(U(prog_dye, "uSinkY"), (0.02f - s->dom[1]) / (s->dom[3] - s->dom[1]));
    glUniform1f(U(prog_dye, "uSeed"), (float)(steps % 977));
    glUniform1i(U(prog_dye, "uNInj"), nj);
    if (nj) { glUniform4fv(U(prog_dye, "uInj"), nj, &inj[0][0]); glUniform4fv(U(prog_dye, "uInjMask"), nj, &msk[0][0]); }
    pass(prog_dye, s->fdye[nxt], s->dx, s->dy);
    glBindTexture(GL_TEXTURE_2D, s->dye[nxt]);
    glGenerateMipmap(GL_TEXTURE_2D);
    s->prev = s->cur; s->cur = nxt;
    (void)dt;
}

/* the tank's geometry and the grids, for a canvas this shape */
static float sim_for_aspect;
static float cells_budget = 50000;
static void tank_setup(float a, float cells) {
    if (sim_for_aspect == a && sl[0].nx) return;
    sim_for_aspect = aspect = a;
    surf_v = 1.0f - AIR;
    wall_l = INSET; wall_r = a - INSET;
    dom[0] = wall_l; dom[1] = -BELOW; dom[2] = wall_r; dom[3] = surf_v;
    memcpy(sl[0].dom, dom, sizeof dom);
    /* the far slice: as far down, but wider, so its walls are out of sight */
    sl[1].dom[0] = CAMX + (dom[0] - 0.01f - CAMX) / ZOOM; sl[1].dom[2] = CAMX + (dom[2] + 0.01f - CAMX) / ZOOM;
    sl[1].dom[1] = dom[1]; sl[1].dom[3] = dom[3];
    /* the near ink at four times the velocity's resolution where that stays affordable */
    slice_init(&sl[0], cells, cells * 16 <= 1.0e6f ? 4 : 3);
    slice_init(&sl[1], cells * 0.45f, 3);
    scene_idx = -1;
}

/* bring the simulation to local time t, returning the cross-fade between
 * the last two steps */
static float advance(double t) {
    if (t < 0) t = 0;
    int idx = (int)floor(t / SCENE);
    double lt = t - idx * SCENE;
    if (idx != scene_idx) {
        scene_idx = idx;
        gen_scene(idx, &scene);
        rng = mix32(opt_seed ^ (uint32_t)(idx + scene_shift) * 0x85ebca6bU);
        for (int i = 0; i < 4; i++) drift[i] = frange(0, 6.2831853f);
        for (int k = 0; k < 2; k++) slice_clear(&sl[k]);
        steps = 0;
    }
    long want = (long)floor(lt * SIM_HZ) + 1;
    /* live, a late start (say an output restarted mid-scene) catches up a
     * few steps a frame, in fast motion, rather than freezing for seconds */
    int budget = opt_shot || opt_bench ? 1 << 30 : 12;
    while (steps < want && budget-- > 0) {
        double st = steps / SIM_HZ;
        for (int k = 0; k < 2; k++) step_slice(&sl[k], k, st);
        steps++;
    }
    if (steps < want) return 1.0f;
    return (float)(lt * SIM_HZ - (want - 1));
}

/* ---- caption: the time, and the time away ---- */
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

/* a glyph atlas of printable ASCII in two sizes */
typedef struct { short x, y, w, h, left, top, adv; } Glyph;
static Glyph glyphs[2][96];
static int font_px[2], atlas_w = 1024, atlas_h;
static GLuint atlas_tex, prog_text, vao_text, vbo_text;
static int font_for_h;
static FT_Library ft;
static FT_Face face;

static const char *VS_TEXT = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 a;     /* x, y (px from top left), u, v (texels) */
    uniform vec2 uRes;
    out vec2 vUV;
    void main() {
        vUV = a.zw;
        gl_Position = vec4(a.x / uRes.x * 2.0 - 1.0, 1.0 - a.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_TEXT = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    uniform vec4 uColor;
    in vec2 vUV;
    out vec4 o;
    void main() { o = vec4(uColor.rgb, uColor.a * texelFetch(uAtlas, ivec2(vUV), 0).r); });

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    char pat_s[256];
    snprintf(pat_s, sizeof pat_s, "%s:style=Italic", opt_font);
    FcPattern *pat = FcNameParse((const FcChar8 *)pat_s);
    FcConfigSubstitute(NULL, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);
    FcResult res;
    FcPattern *m = FcFontMatch(NULL, pat, &res);
    FcChar8 *file = NULL;
    if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
    if (FT_New_Face(ft, (const char *)file, 0, &face)) die("can't load the font");
    FcPatternDestroy(m);
    FcPatternDestroy(pat);
}

static void build_atlas(int h) {
    font_for_h = h;
    font_px[0] = (int)lroundf(h * 0.03f); font_px[1] = (int)lroundf(h * 0.017f);
    if (font_px[0] < 12) font_px[0] = 12;
    if (font_px[1] < 9) font_px[1] = 9;
    int cap = 1024;
    unsigned char *px = calloc((size_t)atlas_w * cap, 1);
    int x = 1, y = 1, shelf = 0;
    for (int s = 0; s < 2; s++) {
        FT_Set_Pixel_Sizes(face, 0, (FT_UInt)font_px[s]);
        for (int c = 0; c < 96; c++) {
            Glyph *gl = &glyphs[s][c];
            memset(gl, 0, sizeof *gl);
            if (FT_Load_Char(face, (FT_ULong)(32 + c), FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) continue;
            FT_GlyphSlot g = face->glyph;
            int w = (int)g->bitmap.width, gh = (int)g->bitmap.rows;
            if (x + w + 2 > atlas_w) { x = 1; y += shelf + 2; shelf = 0; }
            if (y + gh + 2 > cap) break;
            for (int yy = 0; yy < gh; yy++)
                memcpy(px + (size_t)(y + yy) * atlas_w + x, g->bitmap.buffer + yy * g->bitmap.pitch, (size_t)w);
            *gl = (Glyph){(short)x, (short)y, (short)w, (short)gh, (short)g->bitmap_left, (short)g->bitmap_top,
                          (short)(g->advance.x >> 6)};
            x += w + 2;
            if (gh > shelf) shelf = gh;
        }
    }
    atlas_h = y + shelf + 2;
    if (!atlas_tex) glGenTextures(1, &atlas_tex);
    glBindTexture(GL_TEXTURE_2D, atlas_tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlas_w, atlas_h, 0, GL_RED, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    free(px);
}

static float tv[4096];
static int tvn;

static float text_w(int s, const char *str, float track) {
    float w = 0;
    for (const char *p = str; *p; p++) {
        int c = (unsigned char)*p;
        w += glyphs[s][c >= 32 && c < 127 ? c - 32 : 0].adv + (p[1] ? track : 0);
    }
    return w;
}

/* str with its baseline at y (px from the top), right-aligned to x */
static void text(int s, float x, float y, const char *str, float track) {
    float pen = roundf(x - text_w(s, str, track));
    y = roundf(y);
    for (const char *p = str; *p; p++) {
        int c = (unsigned char)*p;
        const Glyph *g = &glyphs[s][c >= 32 && c < 127 ? c - 32 : 0];
        if (g->w && tvn + 24 <= (int)(sizeof tv / sizeof *tv)) {
            float x0 = pen + g->left, y0 = y - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float u0 = g->x, v0 = g->y, u1 = g->x + g->w, v1 = g->y + g->h;
            float q[24] = {x0, y0, u0, v0, x1, y0, u1, v0, x1, y1, u1, v1, x0, y0, u0, v0, x1, y1, u1, v1, x0, y1, u0, v1};
            memcpy(tv + tvn, q, sizeof q);
            tvn += 24;
        }
        pen += g->adv + track;
    }
}

static void draw_caption(GLuint target) {
    if (opt_nocap || (span_me >= 0 && !opt_split && span_me != span_front)) return;
    if (!prog_text) {
        fonts_init();
        prog_text = mk_link(compile(GL_VERTEX_SHADER, VS_TEXT, NULL), compile(GL_FRAGMENT_SHADER, FS_TEXT, NULL));
        glGenVertexArrays(1, &vao_text);
        glGenBuffers(1, &vbo_text);
        glBindVertexArray(vao_text);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);
    }
    int ref = W < H ? W : H;
    if (font_for_h != ref) build_atlas(ref);
    time_t tnow = time(NULL);
    struct tm lt;
    localtime_r(&tnow, &lt);
    char tbuf[32], abuf[64] = "";
    snprintf(tbuf, sizeof tbuf, "%d:%02d", lt.tm_hour, lt.tm_min);
    if (locked_at && tnow >= locked_at) {
        long a = (long)(tnow - locked_at);
        if (a < 3600) snprintf(abuf, sizeof abuf, "away %ld:%02ld", a / 60, a % 60);
        else snprintf(abuf, sizeof abuf, "away %ldh %02ldm", a / 3600, a / 60 % 60);
    }
    tvn = 0;
    float m = ref * 0.05f, x = W - m, y = H - m;
    if (abuf[0]) { text(1, x, y, abuf, font_px[1] * 0.06f); y -= font_px[1] * 1.55f; }
    text(0, x, y, tbuf, font_px[0] * 0.04f);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog_text);
    glUniform2f(U(prog_text, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_text, "uAtlas"), 0);
    if (P->dark) glUniform4f(U(prog_text, "uColor"), 0.78f, 0.80f, 0.84f, 0.55f);
    else glUniform4f(U(prog_text, "uColor"), 0.16f, 0.17f, 0.2f, 0.55f);
    bind(0, atlas_tex);
    glBindVertexArray(vao_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glBufferData(GL_ARRAY_BUFFER, tvn * sizeof(float), tv, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, tvn / 4);
    glDisable(GL_BLEND);
}

/* ---- a frame ---- */
static float canvas[4];                /* the canvas (the tank's frame): x, y (top), w, h, logical */

static void gl_init(void) {
    vs_full = compile(GL_VERTEX_SHADER, VS, NULL);
    prog_curl = mk_sim(FS_CURL);
    prog_vel = mk_sim(FS_VEL);
    prog_div = mk_sim(FS_DIV);
    prog_jac = mk_sim(FS_JACOBI);
    prog_grad = mk_sim(FS_GRAD);
    prog_adv = mk_sim(FS_ADV);
    prog_dye = mk_sim(FS_DYE);
    prog_now = mk_sim(FS_NOW);
    prog_resid = mk_sim(FS_RESID);
    prog_restrict = mk_sim(FS_RESTRICT);
    prog_prolong = mk_sim(FS_PROLONG);
    prog_show = mk_link(vs_full, compile(GL_FRAGMENT_SHADER, "#version 330 core\n", FS_SHOW));
    prog_final = mk_link(vs_full, compile(GL_FRAGMENT_SHADER, "#version 330 core\n#define FINAL 1\n", FS_SHOW));
    glGenVertexArrays(1, &vao);
    GLuint p;
    glUseProgram(p = prog_vel);
    glUniform1i(U(p, "uVel"), 0); glUniform1i(U(p, "uCurl"), 1); glUniform1i(U(p, "uDye"), 2);
    glUseProgram(p = prog_curl); glUniform1i(U(p, "uVel"), 0);
    glUseProgram(p = prog_div); glUniform1i(U(p, "uVel"), 0);
    glUseProgram(p = prog_jac); glUniform1i(U(p, "uP"), 0); glUniform1i(U(p, "uDiv"), 1);
    glUseProgram(p = prog_resid); glUniform1i(U(p, "uP"), 0); glUniform1i(U(p, "uDiv"), 1);
    glUseProgram(p = prog_restrict); glUniform1i(U(p, "uSrc"), 0);
    glUseProgram(p = prog_prolong); glUniform1i(U(p, "uP"), 0); glUniform1i(U(p, "uE"), 1);
    glUseProgram(p = prog_grad); glUniform1i(U(p, "uVel"), 0); glUniform1i(U(p, "uP"), 1);
    glUseProgram(p = prog_adv); glUniform1i(U(p, "uSrc"), 0); glUniform1i(U(p, "uVel"), 1);
    glUseProgram(p = prog_dye);
    glUniform1i(U(p, "uSrc"), 0); glUniform1i(U(p, "uHat"), 1); glUniform1i(U(p, "uBack"), 2); glUniform1i(U(p, "uVel"), 3);
    for (int k = 0; k < 2; k++) {
        glUseProgram(p = k ? prog_final : prog_show);
        glUniform1i(U(p, "uA"), 0); glUniform1i(U(p, "uB"), 1); glUniform1i(U(p, "uA1"), 2); glUniform1i(U(p, "uB1"), 3);
        glUniform1i(U(p, "uLow"), 4);
    }
    glUseProgram(p = prog_now); glUniform1i(U(p, "uSrc0"), 0); glUniform1i(U(p, "uSrc1"), 1);
}

/* --bench: GPU time of the sim, the cross-fade and the picture */
static GLuint prof_q[4];
static double prof_ms[3];
static int prof_n;
static void prof_mark(int i) { if (opt_bench > 0) { if (!prof_q[0]) glGenQueries(4, prof_q); glQueryCounter(prof_q[i], GL_TIMESTAMP); } }
static void prof_read(void) {
    GLuint64 v[4];
    for (int i = 0; i < 4; i++) glGetQueryObjectui64v(prof_q[i], GL_QUERY_RESULT, &v[i]);
    for (int i = 0; i < 3; i++) prof_ms[i] += (v[i + 1] - v[i]) * 1e-6;
    prof_n++;
}

/* the first pass's target, at --scale of the output */
static GLuint low_tex, low_fbo;
static int low_w, low_h;
static float opt_scale;                /* 0: 0.7, less on a big output */

static void set_uniforms(GLuint p, double lt, float fade) {
    glUseProgram(p);
    glUniform1f(U(p, "uFade"), fade);
    glUniform4fv(U(p, "uDomA"), 1, sl[0].dom);
    glUniform4fv(U(p, "uDomB"), 1, sl[1].dom);
    glUniform3f(U(p, "uTank"), wall_l, wall_r, surf_v);
    glUniform2f(U(p, "uCam"), CAMX, CAMY);
    glUniform1f(U(p, "uZoom"), ZOOM);
    glUniform1f(U(p, "uZ0"), 0.09f);
    glUniform1f(U(p, "uTilt"), 0.08f);
    /* the axes of the drops in so far, the latest first */
    for (int k = 0; k < 2; k++) {
        float ax[8];
        int n = 0;
        for (int i = scene.n - 1; i >= 0 && n < 8; i--)
            if (scene.d[i].layer == k && scene.d[i].t <= lt) ax[n++] = drop_x(&scene.d[i]);
        glUniform1i(U(p, k ? "uNAxB" : "uNAxA"), n);
        if (n) glUniform1fv(U(p, k ? "uAxB" : "uAxA"), n, ax);
    }
    glUniform1f(U(p, "uAspect"), aspect);
    glUniform1i(U(p, "uDark"), P->dark);
    float bg[3], sig[4][3], col[4][3];
    hex(P->bg, bg);
    glUniform3fv(U(p, "uBg"), 1, bg);
    for (int i = 0; i < 4; i++) {
        hex(P->ink[i], col[i]);
        for (int c = 0; c < 3; c++)
            sig[i][c] = P->dark ? 1.2f * (1.0f - col[i][c]) + 0.15f : -logf(fmaxf(col[i][c], 0.02f));
    }
    glUniform3fv(U(p, "uSig"), 4, &sig[0][0]);
    glUniform3fv(U(p, "uCol"), 4, &col[0][0]);
    glUniform2f(U(p, "uDyeN"), (float)sl[0].dx, (float)sl[0].dy);
    float vis[8][4];
    int nv = 0;
    for (int i = 0; i < scene.n && nv < 8; i++) {
        float ts = (float)lt - scene.d[i].t;
        if (ts < -1.0f || ts > 6.0f || scene.d[i].layer) continue;
        vis[nv][0] = scene.d[i].x; vis[nv][1] = ts; vis[nv][2] = scene.d[i].r; vis[nv][3] = (float)scene.d[i].ink;
        nv++;
    }
    glUniform1i(U(p, "uNVis"), nv);
    if (nv) glUniform4fv(U(p, "uVis"), nv, &vis[0][0]);
}

static void render(double t, GLuint target) {
    t += span_toff;
    /* the canvas: the layout's bounding box, or just this output */
    float rx = 0, ry = 0, rw = (float)W, rh = (float)H;
    if (span_me >= 0 && !opt_split) { rx = span[span_me][0]; ry = span[span_me][1]; rw = span[span_me][2]; rh = span[span_me][3]; }
    else { canvas[0] = 0; canvas[1] = 0; canvas[2] = (float)W; canvas[3] = (float)H; }
    float ch = canvas[3];
    tank_setup(canvas[2] / ch, cells_budget * (span_me >= 0 && !opt_split ? 1.8f : 1.0f));
    prof_mark(0);
    float mixf = advance(t);
    prof_mark(1);
    double lt = t - floor(t / SCENE) * SCENE;
    float fade = lt > SCENE - FADE ? (float)((SCENE - lt) / FADE) : 1.0f;
    fade = fade * fade * (3 - 2 * fade);

    /* this frame's ink, between the last two steps */
    glUseProgram(prog_now);
    glUniform1f(U(prog_now, "uMix"), mixf);
    glBindVertexArray(vao);
    for (int k = 0; k < 2; k++) {
        bind(0, sl[k].dye[sl[k].prev]); bind(1, sl[k].dye[sl[k].cur]);
        pass(prog_now, sl[k].fnow, sl[k].dx, sl[k].dy);
    }
    prof_mark(2);

    /* the water and the ink, scaled down */
    float sc = opt_scale > 0 ? opt_scale : fminf(0.7f, sqrtf(1.1e6f / ((float)W * H)));
    int lw = (int)lroundf(W * sc), lh = (int)lroundf(H * sc);
    if (lw < 1) lw = 1;
    if (lh < 1) lh = 1;
    if (lw != low_w || lh != low_h) {
        if (low_w) { glDeleteTextures(1, &low_tex); glDeleteFramebuffers(1, &low_fbo); }
        low_w = lw; low_h = lh;
        tex_fbo(&low_tex, &low_fbo, lw, lh, GL_RGBA16F, GL_RGBA, 0);
    }
    float sx = rw / W / ch, sy = rh / H / ch;
    float mx = (rx - canvas[0]) / ch, my = (ch - (ry - canvas[1]) - rh) / ch;
    set_uniforms(prog_show, lt, fade);
    glUniform4f(U(prog_show, "uMap"), mx, sx * W / lw, my, sy * H / lh);
    glUniform1f(U(prog_show, "uPx"), sy * H / lh);
    bind(0, sl[0].now); bind(1, sl[1].now);
    bind(2, sl[0].dye[sl[0].cur]); bind(3, sl[1].dye[sl[1].cur]);
    glBindFramebuffer(GL_FRAMEBUFFER, low_fbo);
    glViewport(0, 0, lw, lh);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* and at full size, the glass and the surface over it */
    set_uniforms(prog_final, lt, fade);
    glUniform4f(U(prog_final, "uMap"), mx, sx, my, sy);
    glUniform1f(U(prog_final, "uPx"), sy);
    glUniform2f(U(prog_final, "uLowScale"), (float)lw / W, (float)lh / H);
    bind(4, low_tex);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    prof_mark(3);
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "ink");
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

/* spanning: the canvas is the layout's bounding box; the front output is
 * the one under its middle. --split: each output its own tank, a scene
 * ahead per step round from the front one and that far into it */
static void span_setup(void) {
    if (span_me < 0) return;
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]);
        y0 = fminf(y0, span[i][1]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    canvas[0] = x0; canvas[1] = y0; canvas[2] = x1 - x0; canvas[3] = y1 - y0;
    float mx = (x0 + x1) / 2;
    span_front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { span_front = i; break; }
    if (opt_split) {
        int k = (span_me - span_front + span_n) % span_n;
        scene_shift = k;
        span_toff = SCENE * 0.37 * k;
    }
}

static void usage(void) {
    fprintf(stderr,
        "usage: ink [--window | --root | --layer] [--seed N] [--palette NAME] [--font FAMILY] [--no-caption]\n"
        "           [--scale F (of the output the ink is drawn at; default 0.7, less above 1.1 Mpx)]\n"
        "           [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX] [--split]\n"
        "           [--shot FILE.ppm | --bench N] [--size WxH] [--at SECONDS]\n"
        "palettes: indigo sumi botanical neon sunset (default: from the seed)\n");
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
        else if (!strcmp(a, "-split")) opt_split = 1;
        else if (!strcmp(a, "-no-caption")) opt_nocap = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NPAL; k++) if (!strcmp(n, PALETTES[k].name)) opt_palette = k;
            if (opt_palette < 0) usage();
        }
        else if (!strcmp(a, "-font") && i + 1 < argc) opt_font = argv[++i];
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = fminf(fmaxf((float)atof(argv[++i]), 0.25f), 1.0f);
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
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (!opt_seed) opt_seed = 1;
    P = &PALETTES[opt_palette >= 0 ? opt_palette : (int)(mix32(opt_seed * 2246822519U ^ 0x1c4e7U) % NPAL)];
    span_setup();
    int offline = opt_shot || opt_bench > 0;
    locked_at = lock_start();

    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    if (opt_layer && !offline) wl_init();
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
        if (offline) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Ink");
        XClassHint chint = {"ink", "Ink"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offline) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (opt_shot) {
        GLuint fbo = offscreen();
        render(opt_at, fbo);
        draw_caption(fbo);
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }
    if (opt_bench > 0) {
        GLuint fbo = offscreen();
        double tw = now();
        render(opt_at - 5 / 60.0, fbo);    /* the catch-up from clear water */
        glFinish();
        double catchup = (now() - tw) * 1e3;
        for (int i = 4; i > 0; i--) render(opt_at - i / 60.0, fbo);
        glFinish();
        double sum = 0, worst = 0;
        for (int i = 0; i < opt_bench; i++) {
            double t0 = now();
            render(opt_at + i / 60.0, fbo);
            glFinish();
            prof_read();
            double ms = (now() - t0) * 1e3;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("ink: %d frames at %dx%d from %.1f s: avg %.3f ms, worst %.3f ms (catch-up %.0f ms)\n"
               "GPU per frame: sim %.3f ms, cross-fade %.3f ms, picture %.3f ms\n"
               "sim: %dx%d (ink %dx%d) + %dx%d (ink %dx%d)\nrenderer: %s\n",
               opt_bench, W, H, opt_at, sum / opt_bench, worst, catchup,
               prof_ms[0] / prof_n, prof_ms[1] / prof_n, prof_ms[2] / prof_n,
               sl[0].nx, sl[0].ny, sl[0].dx, sl[0].dy, sl[1].nx, sl[1].ny, sl[1].dx, sl[1].dy,
               (const char *)glGetString(GL_RENDERER));
        return 0;
    }

    double t0 = now();
    if (span_me >= 0 && opt_epoch) t0 -= wallclock() - opt_epoch;   /* the clock runs from the shared epoch */
    double cap_t = 0;
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
        if (t - cap_t > 30) { locked_at = lock_start(); cap_t = t; }
        render(t - t0, 0);
        draw_caption(0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}
