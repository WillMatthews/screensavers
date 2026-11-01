/*
 * coral — reaction–diffusion growth, cast in brass.
 *
 * A Gray–Scott system (two chemicals, u feeding v, v eating u and dying at a
 * kill rate) runs on the GPU at a fraction of the screen's resolution, a few
 * steps a frame, ping-ponging between two float textures. It starts from a
 * handful of seeds and creeps outwards like coral or lichen.
 *
 * The feed and kill rates are not one pair but a smooth field across the
 * screen, drifting slowly: a low-frequency noise picks a point along a path
 * through the (feed, kill) plane that passes dividing spots, mazes, holes,
 * branching coral and brain-coral worms, so several morphologies grow side by
 * side and change as the field moves over them.
 *
 * Every frame v is turned into a height field (rounded ridges, flat-topped
 * where they are broad, like an embossed plate), upsampled with a B-spline
 * and lit as metal: hammered brass ridges that mostly reflect a soft lamp
 * wandering low over the plate, so a pool of gold sheen and cream glints
 * drifts across it; dark oxidised valleys on a near-black brown ground.
 *
 * When the plate has filled and gone still (or after a few minutes anyway) a
 * tide of high kill rate sweeps slowly across it with a ragged edge and
 * dissolves the old growth (its ridges glowing faintly amber as they go);
 * spores sprout in its wake and seeds fall behind it, so new growth follows
 * it in and the plate never sticks and never goes blank. If the growth ever
 * dies out on its own it is reseeded.
 *
 * Same plumbing as stir and flightsim: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
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
static int opt_root, opt_layer, opt_bench;
static float opt_scale = 3.0f;          /* screen px per simulation cell */
static float opt_speed = 1.0f;          /* growth speed */
static float opt_morph = -1;            /* pin the morphology (0..1); <0 = a drifting field */
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 60;              /* --shot / --bench: this many seconds in */
static int off_w = 1920, off_h = 1080;
static int debug;                       /* CORAL_DEBUG=1: the life cycle on stderr */

#define STEPS_PER_SEC 200.0             /* simulation steps per second at --speed 1 */
#define MAX_STEPS 16                    /* per frame, however long the frame took */

/* the tide that clears the plate for regrowth */
#define TIDE_SPEED (1.0 / 40)           /* noise units per second: ~65 s for the whole sweep */
#define TIDE_BAND 0.2                   /* its width, in noise units (~8 s to cross a point) */
#define TIDE_KILL 0.02f                 /* extra kill rate inside it */
#define SPORES 0.004                    /* chance a 6x6 block in its wake sprouts, per second */

static void die(const char *what) { fprintf(stderr, "coral: %s\n", what); exit(1); }

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

/* ---- shaders ---- */
#define GLSL(...) #__VA_ARGS__
static const char *VS = GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });

/* shared noise, prepended to the shaders that want it */
static const char *NOISE = GLSL(
    float hash(vec2 p) {
        vec3 q = fract(vec3(p.xyx) * 0.1031);
        q += dot(q, q.yzx + 33.33);
        return fract((q.x + q.y) * q.z);
    }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
    }
    float fbm(vec2 p) {
        float s = 0.0, a = 0.5;
        for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p + 7.3; a *= 0.5; }
        return s / 0.9375;
    }
);

/* The parameter field, at a sixteenth of the sim's area:
 *   r = morphology, 0..1 along the (feed, kill) path
 *   g = extra kill rate (the eroding tide)
 *   b = 1 where fresh seeds may take (behind the tide, or anywhere between tides)
 *   a = the tide's wake, just behind it, where spores sprout */
static const char *FS_PARAM = GLSL(
    uniform vec2 uRes;          /* this texture's size */
    uniform float uAspect, uMorph, uTide, uBand, uKill, uEroding;
    uniform vec2 uDrift, uDir, uOff;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec2 p = vec2(uv.x * uAspect, uv.y);
        float m = uMorph;
        if (m < 0.0) {
            /* two slow layers, so it changes shape rather than just sliding */
            float a = fbm(p * 1.25 + uDrift + uOff);
            float b = fbm(p * 0.7 - uDrift.yx * 0.6 + uOff.yx + 4.0);
            m = clamp((0.6 * a + 0.4 * b - 0.5) * 2.1 + 0.5, 0.0, 1.0);
        }
        /* the tide: mostly a straight sweep, with a ragged noisy front */
        float lin = dot(uv - 0.5, uDir) / (0.5 * (abs(uDir.x) + abs(uDir.y)));
        float n = 0.5 + 0.36 * lin + 0.42 * (fbm(p * 2.2 + uOff * 1.7 + 11.0) - 0.5);
        float band = smoothstep(uTide - uBand, uTide - uBand + 0.03, n) * (1.0 - smoothstep(uTide - 0.02, uTide, n));
        float behind = 1.0 - smoothstep(uTide - uBand - 0.02, uTide - uBand, n);
        float wake = behind * smoothstep(uTide - uBand - 0.09, uTide - uBand - 0.03, n);
        band *= uEroding;
        o = vec4(m, uKill * band, mix(1.0, behind, uEroding), wake * uEroding);
    });

/* One Gray–Scott step (Karl Sims' units: Du 1, Dv 0.5, dt 1, 3x3 Laplacian),
 * dropping seeds in as asked. */
static const char *FS_STEP = GLSL(
    uniform sampler2D uSim, uParam;
    uniform ivec2 uSize;
    uniform vec2 uFK[6];        /* the (feed, kill) path the morphology runs along */
    uniform vec4 uSeeds[8];     /* x, y, radius (cells), wobble phase */
    uniform int uSeedN;
    uniform float uSpore;       /* chance for a 6x6 block in the tide's wake to sprout, this step */
    uniform vec2 uRand;
    out vec4 o;
    vec2 S(ivec2 p) { return texelFetch(uSim, clamp(p, ivec2(0), uSize - 1), 0).rg; }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        vec2 c = S(p);
        vec2 lap = 0.2 * (S(p + ivec2(1, 0)) + S(p - ivec2(1, 0)) + S(p + ivec2(0, 1)) + S(p - ivec2(0, 1))) +
                   0.05 * (S(p + ivec2(1, 1)) + S(p + ivec2(-1, 1)) + S(p + ivec2(1, -1)) + S(p + ivec2(-1, -1))) - c;
        vec4 pr = texture(uParam, (vec2(p) + 0.5) / vec2(uSize));
        float m = clamp(pr.r, 0.0, 1.0) * 5.0;
        int i = min(int(m), 4);
        vec2 fk = mix(uFK[i], uFK[i + 1], m - float(i));
        float f = fk.x, k = fk.y + pr.g;
        float u = c.r, v = c.g, uvv = u * v * v;
        u += lap.r - uvv + f * (1.0 - u);
        v += 0.5 * lap.g + uvv - (f + k) * v;
        for (int s = 0; s < uSeedN; s++) {
            vec2 d = vec2(p) - uSeeds[s].xy;
            float a = atan(d.y, d.x);
            float r = uSeeds[s].z * (1.0 + 0.3 * sin(3.0 * a + uSeeds[s].w) + 0.15 * sin(5.0 * a - 2.0 * uSeeds[s].w));
            if (length(d) < r && pr.b > 0.5) { u = 0.5; v = 0.25 + 0.1 * hash(vec2(p) + uSeeds[s].w); }
        }
        /* spores in the tide's wake, so new growth follows it in */
        if (uSpore > 0.0) {
            vec2 b = floor(vec2(p) / 6.0);
            if (hash(b + uRand) < uSpore * pr.a && length(vec2(p) - b * 6.0 - 2.5) < 2.6) { u = 0.5; v = 0.3; }
        }
        o = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0);
    });

/* The relief, at sim resolution: r = height (ridges, flat-topped where broad),
 * gb = its gradient per cell, a = how hard the tide is eroding here */
static const char *FS_PREP = GLSL(
    uniform sampler2D uSim, uParam;
    uniform ivec2 uSize;
    uniform float uKill;
    out vec4 o;
    float V(ivec2 p) { return texelFetch(uSim, clamp(p, ivec2(0), uSize - 1), 0).g; }
    float Hgt(ivec2 p) {
        float v = V(p);
        return 0.8 * smoothstep(0.1, 0.32, v) + 0.2 * smoothstep(0.28, 0.45, v);
    }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        float h00 = Hgt(p + ivec2(-1, -1)), h10 = Hgt(p + ivec2(0, -1)), h20 = Hgt(p + ivec2(1, -1));
        float h01 = Hgt(p + ivec2(-1, 0)), h11 = Hgt(p), h21 = Hgt(p + ivec2(1, 0));
        float h02 = Hgt(p + ivec2(-1, 1)), h12 = Hgt(p + ivec2(0, 1)), h22 = Hgt(p + ivec2(1, 1));
        float gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) / 8.0;
        float gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) / 8.0;
        float hb = (4.0 * h11 + 2.0 * (h01 + h21 + h10 + h12) + h00 + h20 + h02 + h22) / 16.0;
        float glow = texture(uParam, (vec2(p) + 0.5) / vec2(uSize)).g / uKill;
        o = vec4(hb, gx, gy, glow);
    });

/* Lit metal, at screen resolution */
static const char *FS_SHOW = GLSL(
    uniform sampler2D uRelief, uHammer, uPlate;
    uniform vec2 uRes, uSimRes;
    uniform vec3 uLight;        /* the lamp, over the plate (units of screen height) */
    uniform float uBump, uFade;
    uniform vec3 uBg, uBgAlt, uCream, uGold, uBrass, uAmber, uSage;
    out vec4 o;

    /* B-spline bicubic from four bilinear taps: smooth relief from coarse cells */
    vec4 cubic(vec2 uv, float lod) {
        vec2 res = uSimRes / exp2(lod);
        vec2 p = uv * res - 0.5, f = fract(p);
        p -= f;
        vec2 f2 = f * f, f3 = f2 * f;
        vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
        vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
        vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
        vec2 w3 = f3 / 6.0;
        vec2 s0 = w0 + w1, s1 = w2 + w3;
        vec2 c0 = (p - 0.5 + w1 / s0) / res, c1 = (p + 1.5 + w3 / s1) / res;
        return mix(mix(textureLod(uRelief, vec2(c0.x, c0.y), lod), textureLod(uRelief, vec2(c1.x, c0.y), lod), s1.x),
                   mix(textureLod(uRelief, vec2(c0.x, c1.y), lod), textureLod(uRelief, vec2(c1.x, c1.y), lod), s1.x), s1.y);
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec4 r = cubic(uv, 0.0);
        float h = r.r;
        float metal = smoothstep(0.18, 0.6, h);

        /* the cast relief, then the hammer marks over it (bolder on the metal) */
        vec3 n = normalize(vec3(-r.gb * uBump, 1.0));
        vec2 dimple = texture(uHammer, gl_FragCoord.xy / 256.0).rg * 2.0 - 1.0;
        n = normalize(n + vec3(dimple * mix(0.03, 0.075, metal), 0.0));

        /* cavities: the relief well below its surroundings sits in shadow */
        float around = cubic(uv, 2.0).r;
        float cav = clamp((around - h) * 2.0, 0.0, 1.0);
        float ao = 1.0 - 0.6 * cav;

        /* a small studio over the plate: one soft lamp, low and close,
         * drifting round, a dim warm bounce opposite, dark elsewhere. Metal
         * is mostly what it reflects of that, so the sheen pools where the
         * lamp's reflection is and the pool wanders with it. */
        vec3 P = vec3(gl_FragCoord.xy / uRes.y, 0.0);
        vec3 E = vec3(0.5 * uRes.x / uRes.y, 0.5, 2.2);
        vec3 V = normalize(E - P), toL = uLight - P;
        vec3 L = normalize(toL);
        float att = 1.6 / (1.0 + 1.2 * dot(toL, toL));
        vec3 R = reflect(-V, n);
        float kr = max(dot(R, L), 0.0);
        vec3 Lb = normalize(vec3(-L.xy, 0.35));
        float br = max(dot(R, Lb), 0.0);
        vec3 env = vec3(0.03) + uCream * att * (0.6 * pow(kr, 5.0) + 0.9 * pow(kr, 30.0)) + uAmber * 0.16 * pow(br, 2.0);
        float dl = max(dot(n, L), 0.0) * att;

        /* the ground: dark warm brown, mottled, a little oxidised beside the ridges */
        vec2 plate = texture(uPlate, uv).rg;
        float mott = plate.r;
        vec3 ground = mix(uBg, uBgAlt, 0.5 * mott);
        float ox = smoothstep(0.02, 0.3, around) * (1.0 - metal);
        ground = mix(ground, uBg * 0.55 + uSage * 0.012, 0.7 * ox);
        vec3 gcol = ground * (0.5 + 0.7 * dl) * ao + uAmber * 0.03 * pow(kr, 30.0);

        /* brass: reflections tinted by the alloy, with a little of its own
         * colour so the shadowed side never goes dead */
        float pat = plate.g;
        vec3 alloy = mix(uBrass, uGold, 0.35 * pat);
        vec3 mcol = alloy * env * 1.25 + alloy * 0.05 * (0.4 + dl);
        mcol += uCream * 1.1 * att * pow(kr, 140.0);                    /* the glints go cream */
        mcol *= ao;
        mcol = mix(mcol, mcol * vec3(0.7, 0.78, 0.62), 0.5 * cav);   /* patina in the crevices */

        vec3 col = mix(gcol, mcol, metal);

        /* the tide: what it is eating glows faintly amber */
        col += uAmber * 0.12 * r.a * metal;

        /* vignette, fade in, and dither against banding in the dark */
        vec2 q = uv * 2.0 - 1.0;
        col *= mix(0.62, 1.0, smoothstep(1.6, 0.45, length(q * vec2(0.9, 1.0))));
        col = pow(max(col, 0.0) * uFade, vec3(1.0 / 2.2));    /* the palette is linear; the screen isn't */
        col += (hash(gl_FragCoord.xy + fract(uFade * 7.0)) - 0.5) / 255.0;
        o = vec4(col, 1.0);
    });

/* the plate's own slow mottling (r) and alloy variation (g), drawn once at
 * a quarter of the screen's resolution */
static const char *FS_PLATE = GLSL(
    out vec4 o;
    void main() {
        vec2 px = gl_FragCoord.xy * 4.0;
        o = vec4(fbm(px / 300.0 + 5.0), vnoise(px / 140.0 + 3.0), 0.0, 1.0);
    });

/* for the life cycle: v at 64x36, read back once a second */
static const char *FS_PROBE = GLSL(
    uniform sampler2D uSim;
    uniform vec2 uRes;
    out vec4 o;
    void main() { o = vec4(texture(uSim, gl_FragCoord.xy / uRes).g * 2.5, 0.0, 0.0, 1.0); });

/* ---- GL state ---- */
static GLuint p_param, p_step, p_prep, p_show, p_probe, vao, tex_plate;
static GLuint tex_sim[2], fbo_sim[2], tex_param, fbo_param, tex_relief, fbo_relief, tex_probe, fbo_probe, tex_ham;
static int sim_w, sim_h, par_w, par_h, cur;
#define PW 64
#define PH 36

static GLuint compile(GLenum type, const char *src, const char *pre) {
    GLuint s = glCreateShader(type);
    const char *srcs[3] = {"#version 330 core\n", pre ? pre : "", src};
    glShaderSource(s, 3, srcs, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "coral: shader:\n%s\n", log);
        exit(1);
    }
    return s;
}

static GLuint mk_prog(const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile(GL_VERTEX_SHADER, VS, NULL));
    glAttachShader(pr, compile(GL_FRAGMENT_SHADER, fs, NOISE));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) die("shader link failed");
    return pr;
}

#define UL(p, n) glGetUniformLocation(p, n)

static GLuint mk_tex(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void *data, GLenum filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint mk_fbo(GLuint tex) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer incomplete (no float render targets?)");
    return f;
}

/* Hammer marks: a tiling field of shallow overlapping bowls (Worley F1 squared),
 * stored as its normal in rg */
#define HAM 256
#define HAM_CELLS 13
static void build_hammer(void) {
    float px[HAM_CELLS][HAM_CELLS], py[HAM_CELLS][HAM_CELLS];
    float cs = (float)HAM / HAM_CELLS;
    for (int j = 0; j < HAM_CELLS; j++)
        for (int i = 0; i < HAM_CELLS; i++) {
            px[j][i] = (i + 0.15f + 0.7f * frand()) * cs;
            py[j][i] = (j + 0.15f + 0.7f * frand()) * cs;
        }
    float *hgt = malloc(sizeof(float) * HAM * HAM);
    for (int y = 0; y < HAM; y++)
        for (int x = 0; x < HAM; x++) {
            int ci = (int)(x / cs), cj = (int)(y / cs);
            float best = 1e9f;
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int i = ci + di, j = cj + dj;
                    float ox = 0, oy = 0;
                    if (i < 0) { i += HAM_CELLS; ox = -HAM; } else if (i >= HAM_CELLS) { i -= HAM_CELLS; ox = HAM; }
                    if (j < 0) { j += HAM_CELLS; oy = -HAM; } else if (j >= HAM_CELLS) { j -= HAM_CELLS; oy = HAM; }
                    float dx = x + 0.5f - (px[j][i] + ox), dy = y + 0.5f - (py[j][i] + oy);
                    float d = dx * dx + dy * dy;
                    if (d < best) best = d;
                }
            hgt[y * HAM + x] = best / (cs * cs);
        }
    /* soften the creases between the blows a little: two box blurs */
    float *tmp = malloc(sizeof(float) * HAM * HAM);
    for (int pass = 0; pass < 2; pass++) {
        for (int y = 0; y < HAM; y++)
            for (int x = 0; x < HAM; x++) {
                float s = 0;
                for (int k = -2; k <= 2; k++) s += hgt[y * HAM + (x + k + HAM) % HAM];
                tmp[y * HAM + x] = s / 5;
            }
        for (int y = 0; y < HAM; y++)
            for (int x = 0; x < HAM; x++) {
                float s = 0;
                for (int k = -2; k <= 2; k++) s += tmp[((y + k + HAM) % HAM) * HAM + x];
                hgt[y * HAM + x] = s / 5;
            }
    }
    free(tmp);
    unsigned char *img = malloc((size_t)HAM * HAM * 4);
    for (int y = 0; y < HAM; y++)
        for (int x = 0; x < HAM; x++) {
            float gx = hgt[y * HAM + (x + 1) % HAM] - hgt[y * HAM + (x + HAM - 1) % HAM];
            float gy = hgt[((y + 1) % HAM) * HAM + x] - hgt[((y + HAM - 1) % HAM) * HAM + x];
            float s = 6.0f;
            float nx = -gx * s, ny = -gy * s;
            nx = nx < -1 ? -1 : nx > 1 ? 1 : nx;
            ny = ny < -1 ? -1 : ny > 1 ? 1 : ny;
            unsigned char *p = img + ((size_t)y * HAM + x) * 4;
            p[0] = (unsigned char)lroundf((nx * 0.5f + 0.5f) * 255);
            p[1] = (unsigned char)lroundf((ny * 0.5f + 0.5f) * 255);
            p[2] = (unsigned char)lroundf(fminf(hgt[y * HAM + x], 1.0f) * 255);
            p[3] = 255;
        }
    tex_ham = mk_tex(GL_RGBA8, HAM, HAM, GL_RGBA, GL_UNSIGNED_BYTE, img, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    free(hgt);
    free(img);
}

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) {
        float c = ((v >> (16 - 8 * i)) & 255) / 255.0f;
        rgb[i] = c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);   /* to linear */
    }
}

/* the (feed, kill) path: mitosis, worms, fingerprints, mazes, coral, worms */
static const float FK[6][2] = {
    {0.0367f, 0.0649f}, {0.0340f, 0.0630f}, {0.0370f, 0.0600f},
    {0.0420f, 0.0590f}, {0.0545f, 0.0620f}, {0.0500f, 0.0630f},
};

static void gl_init(int gw, int gh) {
    p_param = mk_prog(FS_PARAM);
    p_step = mk_prog(FS_STEP);
    p_prep = mk_prog(FS_PREP);
    p_show = mk_prog(FS_SHOW);
    p_probe = mk_prog(FS_PROBE);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    sim_w = (int)lroundf(gw / opt_scale); sim_h = (int)lroundf(gh / opt_scale);
    if (sim_w < 16) sim_w = 16;
    if (sim_h < 16) sim_h = 16;
    par_w = (sim_w + 3) / 4; par_h = (sim_h + 3) / 4;
    for (int i = 0; i < 2; i++) {
        tex_sim[i] = mk_tex(GL_RG32F, sim_w, sim_h, GL_RG, GL_FLOAT, NULL, GL_LINEAR);
        fbo_sim[i] = mk_fbo(tex_sim[i]);
        glClearColor(1, 0, 0, 0);                       /* all food, no growth */
        glClear(GL_COLOR_BUFFER_BIT);
    }
    tex_param = mk_tex(GL_RGBA16F, par_w, par_h, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR);
    fbo_param = mk_fbo(tex_param);
    tex_relief = mk_tex(GL_RGBA16F, sim_w, sim_h, GL_RGBA, GL_FLOAT, NULL, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    fbo_relief = mk_fbo(tex_relief);
    tex_probe = mk_tex(GL_RGBA8, PW, PH, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST);
    fbo_probe = mk_fbo(tex_probe);
    build_hammer();
    int pw = (gw + 3) / 4, ph = (gh + 3) / 4;
    tex_plate = mk_tex(GL_RG8, pw, ph, GL_RG, GL_UNSIGNED_BYTE, NULL, GL_LINEAR);
    GLuint fbo_plate = mk_fbo(tex_plate), p_plate = mk_prog(FS_PLATE);
    glViewport(0, 0, pw, ph);
    glUseProgram(p_plate);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDeleteFramebuffers(1, &fbo_plate);
    glDeleteProgram(p_plate);

    glUseProgram(p_step);
    glUniform1i(UL(p_step, "uSim"), 0);
    glUniform1i(UL(p_step, "uParam"), 1);
    glUniform2i(UL(p_step, "uSize"), sim_w, sim_h);
    glUniform2fv(UL(p_step, "uFK"), 6, &FK[0][0]);
    glUseProgram(p_prep);
    glUniform1i(UL(p_prep, "uSim"), 0);
    glUniform1i(UL(p_prep, "uParam"), 1);
    glUniform2i(UL(p_prep, "uSize"), sim_w, sim_h);
    glUniform1f(UL(p_prep, "uKill"), TIDE_KILL);
    glUseProgram(p_probe);
    glUniform1i(UL(p_probe, "uSim"), 0);
    glUniform2f(UL(p_probe, "uRes"), PW, PH);
    glUseProgram(p_param);
    glUniform2f(UL(p_param, "uRes"), (float)par_w, (float)par_h);
    glUniform1f(UL(p_param, "uAspect"), (float)sim_w / sim_h);
    glUniform1f(UL(p_param, "uMorph"), opt_morph);
    glUniform2f(UL(p_param, "uOff"), frand() * 100, frand() * 100);

    /* Aged Brass */
    float c[3];
    glUseProgram(p_show);
    glUniform1i(UL(p_show, "uRelief"), 0);
    glUniform1i(UL(p_show, "uHammer"), 1);
    glUniform1i(UL(p_show, "uPlate"), 2);
    glUniform2f(UL(p_show, "uSimRes"), (float)sim_w, (float)sim_h);
    hex("#17120E", c); glUniform3fv(UL(p_show, "uBg"), 1, c);
    hex("#3A2E25", c); glUniform3fv(UL(p_show, "uBgAlt"), 1, c);
    hex("#EFE3CE", c); glUniform3fv(UL(p_show, "uCream"), 1, c);
    hex("#FFC332", c); glUniform3fv(UL(p_show, "uGold"), 1, c);
    hex("#D6A12A", c); glUniform3fv(UL(p_show, "uBrass"), 1, c);
    hex("#E9994A", c); glUniform3fv(UL(p_show, "uAmber"), 1, c);
    hex("#8FA672", c); glUniform3fv(UL(p_show, "uSage"), 1, c);
}

/* ---- the life cycle ---- */
enum { GROW, ERODE };
static struct {
    double t;                   /* simulated seconds */
    double steps;               /* owed fractional steps */
    int phase;
    double age;                 /* in this phase */
    double tide, next_drop, next_probe;
    double still;               /* how long it's barely changed */
    float dir[2];
    int cycles;
    float seeds[8][4];
    int nseeds;
    unsigned char prev[PW * PH];
    int have_prev;
    float coverage, change;
} lc;

static void drop_seeds(int n, float rmin, float rmax) {
    for (int i = 0; i < n && lc.nseeds < 8; i++) {
        float *s = lc.seeds[lc.nseeds++];
        s[0] = (0.06f + 0.88f * frand()) * sim_w;
        s[1] = (0.06f + 0.88f * frand()) * sim_h;
        s[2] = rmin + (rmax - rmin) * frand();
        s[3] = frand() * 6.2832f;
    }
}

static void start_tide(void) {
    lc.phase = ERODE; lc.age = 0;
    lc.tide = -0.25; lc.next_drop = 3.0;
    float a = frand() * 6.2832f;
    lc.dir[0] = cosf(a); lc.dir[1] = sinf(a);
    glUseProgram(p_param);
    glUniform2f(UL(p_param, "uOff"), frand() * 100, frand() * 100);
    if (debug) fprintf(stderr, "coral: %6.1fs tide (coverage %.3f, change %.4f)\n", lc.t, lc.coverage, lc.change);
}

/* coverage and change over the last second, from a coarse readback */
static void probe(void) {
    unsigned char px[PW * PH * 4];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_probe);
    glViewport(0, 0, PW, PH);
    glUseProgram(p_probe);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_sim[cur]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, PW, PH, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int on = 0;
    long diff = 0;
    for (int i = 0; i < PW * PH; i++) {
        unsigned char v = px[i * 4];
        on += v > 64;
        diff += abs((int)v - (int)lc.prev[i]);
        lc.prev[i] = v;
    }
    lc.coverage = (float)on / (PW * PH);
    lc.change = lc.have_prev ? diff / (255.0f * PW * PH) : 1.0f;
    lc.have_prev = 1;
}

static void lifecycle(double dt) {
    lc.age += dt;
    if (lc.t >= lc.next_probe) {
        lc.next_probe = lc.t + 1.0;
        probe();
        lc.still = lc.change < 0.008f ? lc.still + 1.0 : 0.0;
        if (debug > 1) fprintf(stderr, "coral: %6.1fs %s cov %.3f change %.4f\n", lc.t, lc.phase ? "erode" : "grow ", lc.coverage, lc.change);
        if (lc.phase == GROW) {
            if (lc.coverage < 0.003f && lc.age > 4) {          /* died out: start again */
                if (debug) fprintf(stderr, "coral: %6.1fs reseed\n", lc.t);
                drop_seeds(4 + (int)(frand() * 3), 4, 8);
                lc.age = 0;
            } else if ((lc.age > 90 && lc.still >= 10) || lc.age > 180) {
                start_tide();
            }
        }
    }
    if (lc.phase == ERODE) {
        lc.tide += dt * TIDE_SPEED;
        /* seeds fall all over; only those behind the tide take */
        if (lc.age >= lc.next_drop) {
            lc.next_drop = lc.age + 1.0;
            drop_seeds(3, 5, 9);
        }
        if (lc.tide > 1.25 + TIDE_BAND) {
            lc.phase = GROW; lc.age = 0; lc.still = 0; lc.cycles++;
            if (debug) fprintf(stderr, "coral: %6.1fs tide out\n", lc.t);
        }
    }
}

static void sim_param(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_param);
    glViewport(0, 0, par_w, par_h);
    glUseProgram(p_param);
    /* the morphology field drifts about a screen-width every ten minutes */
    float d = (float)lc.t * 0.004f;
    glUniform2f(UL(p_param, "uDrift"), d * 0.8f + 0.3f * sinf(d * 0.7f), d * 0.45f + 0.3f * cosf(d * 0.5f));
    glUniform2f(UL(p_param, "uDir"), lc.dir[0], lc.dir[1]);
    glUniform1f(UL(p_param, "uTide"), (float)lc.tide);
    glUniform1f(UL(p_param, "uBand"), (float)TIDE_BAND);
    glUniform1f(UL(p_param, "uKill"), TIDE_KILL);
    glUniform1f(UL(p_param, "uEroding"), lc.phase == ERODE ? 1.0f : 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* advance the simulation by dt seconds */
static void advance(double dt) {
    lc.steps += dt * STEPS_PER_SEC * opt_speed;
    int n = (int)lc.steps;
    lc.steps -= n;
    if (n > MAX_STEPS) n = MAX_STEPS;
    lc.t += dt;
    sim_param();
    glViewport(0, 0, sim_w, sim_h);
    glUseProgram(p_step);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_param);
    glActiveTexture(GL_TEXTURE0);
    for (int i = 0; i < n; i++) {
        if (i == 0 && lc.nseeds) {
            glUniform4fv(UL(p_step, "uSeeds"), lc.nseeds, &lc.seeds[0][0]);
            glUniform1i(UL(p_step, "uSeedN"), lc.nseeds);
        } else if (i == 1 || (i == 0 && !lc.nseeds)) {
            glUniform1i(UL(p_step, "uSeedN"), 0);
        }
        /* spores, on the first step of a frame */
        glUniform1f(UL(p_step, "uSpore"), i == 0 && lc.phase == ERODE ? (float)(SPORES * dt) : 0.0f);
        if (i == 0) glUniform2f(UL(p_step, "uRand"), frand() * 512, frand() * 512);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_sim[cur ^ 1]);
        glBindTexture(GL_TEXTURE_2D, tex_sim[cur]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        cur ^= 1;
    }
    if (n) lc.nseeds = 0;
    lifecycle(dt);
}

static void render(GLuint target, double el) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_relief);
    glViewport(0, 0, sim_w, sim_h);
    glUseProgram(p_prep);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_param);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_sim[cur]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindTexture(GL_TEXTURE_2D, tex_relief);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glUseProgram(p_show);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_ham);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_plate);
    glActiveTexture(GL_TEXTURE0);
    glUniform2f(UL(p_show, "uRes"), (float)W, (float)H);
    /* the lamp wanders round an ellipse over the plate every two and a half
     * minutes, rising and dipping a little */
    double a = lc.t * 6.2831853 / 150.0 + 2.2, asp = (double)W / H;
    glUniform3f(UL(p_show, "uLight"), (float)(0.5 * asp + 0.42 * asp * cos(a)), (float)(0.5 + 0.38 * sin(a)),
                (float)(0.8 + 0.15 * sin(lc.t * 6.2831853 / 47.0)));
    glUniform1f(UL(p_show, "uBump"), 2.4f * 3.0f / opt_scale);
    double f = el / 1.5;
    glUniform1f(UL(p_show, "uFade"), (float)(f > 1 ? 1 : f * f * (3 - 2 * f)));
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
    /* the simulation is sized once; a later resize just stretches it */
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "coral");
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
        "usage: coral [--window | --root | --layer] [--seed N] [--speed X] [--scale PX] [--morph 0..1]\n"
        "             [--shot FILE.ppm | --bench FRAMES] [--size WxH] [--at SECONDS]\n");
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
        else if (!strcmp(a, "-speed") && i + 1 < argc) opt_speed = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-scale") && i + 1 < argc) opt_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-morph") && i + 1 < argc) opt_morph = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) { if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2) usage(); }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else usage();
    }
    if (opt_speed <= 0 || opt_scale < 1 || opt_scale > 8 || opt_morph > 1 || off_w < 16 || off_h < 16) usage();
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    rng = opt_seed | 1;
    for (int i = 0; i < 8; i++) frand();
    const char *dbg = getenv("CORAL_DEBUG");
    debug = dbg ? atoi(dbg) : 0;
    int offscreen = opt_shot || opt_bench;

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
        XStoreName(dpy, win, "Coral");
        XClassHint chint = {"coral", "Coral"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }

    /* a window gets its real size from the WM; size the simulation for the
     * largest it's likely to be (the screen), and stretch on resize */
    int gw = W, gh = H;
    if (dpy && !opt_root && !offscreen) {
        gw = DisplayWidth(dpy, DefaultScreen(dpy));
        gh = DisplayHeight(dpy, DefaultScreen(dpy));
    }
    gl_init(gw, gh);
    drop_seeds(5 + (int)(frand() * 3), 4, 8);           /* the first growth */

    if (offscreen) {
        GLuint tex = mk_tex(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_NEAREST);
        GLuint fbo = mk_fbo(tex);
        for (double t = 0; t < opt_at - 1e-9; t += 1.0 / 60) advance(1.0 / 60);
        if (opt_shot) {
            render(fbo, opt_at);
            glFinish();
            save_ppm(opt_shot);
            return 0;
        }
        render(fbo, opt_at);
        glFinish();
        double t0 = now(), worst = 0, t_sim = 0;
        for (int i = 0; i < opt_bench; i++) {
            double f0 = now();
            advance(1.0 / 60);
            glFinish();
            double f1 = now();
            render(fbo, opt_at + i / 60.0);
            glFinish();
            t_sim += f1 - f0;
            worst = fmax(worst, now() - f0);
        }
        double per = (now() - t0) / opt_bench;
        printf("%dx%d, sim %dx%d, %.0f steps/s: %.2f ms/frame avg (simulation %.2f, drawing %.2f), %.2f worst (%s)\n",
               W, H, sim_w, sim_h, STEPS_PER_SEC * opt_speed, per * 1e3, t_sim / opt_bench * 1e3,
               (per - t_sim / opt_bench) * 1e3, worst * 1e3, (const char *)glGetString(GL_RENDERER));
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
        double t = now(), dt = t - last;
        if (dt > 0.1) dt = 0.1;                         /* after a stall, don't lurch */
        advance(dt);
        render(0, t - t0);
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
