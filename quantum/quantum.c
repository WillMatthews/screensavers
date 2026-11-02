/*
 * quantum — the time-dependent Schrödinger equation, solved live.
 *
 *     i hbar dpsi/dt = -hbar^2/2m Laplacian psi + V psi      (hbar = m = 1)
 *
 * on a 2D grid (64k to 128k points), by the split-step Fourier method:
 * Strang splitting, half a potential step, a whole kinetic step in k-space,
 * half a potential step. The FFT is a Stockham autosort, radix 4 with a
 * closing radix-2 stage, one fragment pass per stage on RG32F textures; the
 * potential and kinetic multipliers ride in the first and last passes. An
 * absorbing layer (a complex potential) lines the edges where waves leave.
 * Plain float32 loses ~1e-7 of the norm a step, always the same way (the
 * state changes slowly, so the rounding errors line up), so every complex
 * product is compensated (fma for the products' errors, TwoSum for the
 * sums', each multiplier a float pair plus its residual): without
 * absorbers the norm then holds to ~5e-7 over 8000 steps.
 *
 * The experiments, in a shuffled cycle, each opened by a small title card
 * with its governing equations:
 *   tunnelling   a packet meets a square barrier: R(t), T(t) against T(k)
 *                averaged over the packet; the kx distribution against
 *                T(k) |phi0(k)|^2
 *   double slit  the pattern on a screen, and particles landing on it one
 *                at a time, drawn from |psi|^2 there (the flux through the
 *                screen), with the histogram filling in
 *   coherent     a coherent state circling (or tracing a Lissajous figure)
 *                in a harmonic well, its classical orbit, phase space
 *   squeezed     a squeezed state sloshing and breathing at 2 omega, with
 *                its uncertainty ellipse turning in phase space
 *   revival      a packet in the infinite square well, the odd extension
 *                of the box on the periodic grid, where free evolution is
 *                exact: it falls apart and comes back at T = 4mL^2/(pi hbar),
 *                beside the 1D quantum carpet woven row by row
 *   Bragg        a wave through a 2D crystal turned to the Bragg angle: the
 *                Ewald circle in momentum space, a polar diffraction plot
 *   stadium      a packet launched along the unstable bouncing orbit of a
 *                Bunimovich stadium (or, alternately, into a ring of atoms,
 *                a quantum corral), with the long exposure where scars show
 * Pictures are domain coloured (hue the phase, brightness |psi|^2, a soft
 * glow), the potential a translucent relief with the classical turning
 * line V = E, beside the momentum picture |phi(k)|^2 and live <x>, <p>,
 * norm and energy. --check verifies the physics (norm and energy, the
 * coherent state's orbit, the squeezed width, T for three barriers, the
 * fringe spacing, the revival) and times a step.
 *
 * Palettes (--palette, else from the seed): phase (a hue wheel on black),
 * mono, paper (light), ember. A small caption gives the time and the time
 * away (since swaylock started, from /proc).
 *
 * Same plumbing as gauges and ink: Xlib + GLX for --window / --root, a
 * native wlr-layer-shell surface over EGL for --layer (swaylock-plugin),
 * --shot FILE.ppm [--at SECONDS] [--size WxH] to render one frame
 * offscreen, --bench N to time N frames offscreen. --act NAME plays one
 * experiment on a loop.
 *
 * Spanning: run one instance per monitor with the same --seed and --epoch,
 * --layout (every output's logical rect) and --output = this one's index.
 * Every instance runs the same simulation from the epoch (same grid, same
 * steps, so they agree without talking); the output under the middle of the
 * layout shows the experiment, the others, left to right, the act's own
 * picture (the screen, the carpet, phase space...) and momentum space.
 * A late start catches up a few steps a frame, in fast motion.
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

#define PI 3.14159265358979323846

static int W = 1280, H = 800;
static int opt_root, opt_layer, opt_bench, opt_check;
static uint32_t opt_seed;
static const char *opt_shot;
static double opt_at = 30;             /* --shot / --bench: this many seconds into the act */
static int off_w = 1920, off_h = 1080;
static double opt_epoch;               /* CLOCK_REALTIME seconds the performance starts at */
static int opt_palette = -1, opt_act = -1, opt_nocap, opt_resize_test;

/* ---- spanning ---- */
#define MAX_OUT 8
static float span[MAX_OUT][4];         /* x, y, w, h of each output (logical) */
static int span_n, span_me = -1, span_front;
enum { R_FRONT, R_MOMENTUM, R_AUX };
static int role = R_FRONT;

static void die(const char *what) { fprintf(stderr, "quantum: %s\n", what); exit(1); }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wallclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng;
static double drand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng >> 8) * (1.0 / 16777216.0);
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x ? x : 1;
}

static double clampd(double x, double a, double b) { return x < a ? a : x > b ? b : x; }
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

static void slurp_line(const char *path, char *buf, size_t n) {
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
    fclose(f);
}

/* ---- palettes ---- */
typedef struct {
    const char *name;
    int mode;                          /* 0 phase wheel on black, 1 mono, 2 paper, 3 ember */
    const char *bg, *ink, *dim, *accent, *relief;
} Palette;
static const Palette PALETTES[] = {
    {"phase", 0, "#06070A", "#E4E7EE", "#7D8496", "#F0C674", "#8C9AB8"},
    {"mono", 1, "#040404", "#EBEBEB", "#808080", "#FFFFFF", "#9A9A9A"},
    {"paper", 2, "#F2F0EA", "#23252B", "#7B7D85", "#B0432C", "#8C8576"},
    {"ember", 3, "#0A0605", "#F1E3CD", "#94806A", "#E9994A", "#B07A40"},
};
#define NPAL ((int)(sizeof PALETTES / sizeof *PALETTES))
static const Palette *P;
static float c_bg[3], c_ink[3], c_dim[3], c_acc[3], c_rel[3];

static void hex(const char *h, float *rgb) {
    unsigned v = (unsigned)strtoul(h + 1, NULL, 16);
    for (int i = 0; i < 3; i++) rgb[i] = ((v >> (16 - 8 * i)) & 255) / 255.0f;
}

/* ==================================================================== */
/* ---- GL plumbing ---- */
#define GLSL(...) #__VA_ARGS__
#define U(p, n) glGetUniformLocation(p, n)
static int have_exact;                 /* GL_ARB_gpu_shader5: fma() and precise */

static GLuint compile(GLenum type, const char *a, const char *b) {
    GLuint s = glCreateShader(type);
    const char *src[2] = {a, b ? b : ""};
    glShaderSource(s, 2, src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "quantum: shader:\n%s\n", log);
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
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "quantum: link:\n%s\n", log);
        exit(1);
    }
    return pr;
}

static const char *VS_FULL = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    });
static GLuint vs_full, vao;
static char fs_head[256];
static GLuint mk_fs(const char *fs) { return mk_link(vs_full, compile(GL_FRAGMENT_SHADER, fs_head, fs)); }

static void bind(int unit, GLuint tex) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, tex);
}

static GLuint mk_tex(int w, int h, GLenum ifmt, GLenum fmt, const float *data, int linear, int repeat) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint mk_fbo(GLuint t0, GLuint t1) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t0, 0);
    if (t1) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, t1, 0);
        static const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, bufs);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("float render targets unsupported");
    return f;
}

typedef struct { GLuint t[2], f[2]; int w, h; } Pair;
static void pair_free(Pair *p) {
    if (!p->w) return;
    glDeleteTextures(2, p->t);
    glDeleteFramebuffers(2, p->f);
    p->w = p->h = 0;
}
static void pair_init(Pair *p, int w, int h, GLenum ifmt, GLenum fmt, int linear, int repeat) {
    if (p->w == w && p->h == h) return;
    pair_free(p);
    for (int i = 0; i < 2; i++) {
        p->t[i] = mk_tex(w, h, ifmt, fmt, NULL, linear, repeat);
        p->f[i] = mk_fbo(p->t[i], 0);
    }
    p->w = w; p->h = h;
}

/* ==================================================================== */
/* ---- the solver: split-step Fourier on the GPU ----
 *
 * psi(t + dt) = Vh . F^-1 [ K . F [ Vh . psi(t) ] ],  Vh = exp(-i V dt/2 - W dt/2),
 * K = exp(-i k^2 dt / 2) / (nx ny)  (hbar = m = 1, W the absorbing layer).
 * F is a Stockham autosort FFT, radix 4 (and a closing radix-2 stage for
 * odd powers of two), one fragment pass per stage, x then y, on RG32F.
 * The multipliers are fused into the first and last passes, so a step is
 * just the 2 (log4 nx + log4 ny) passes.
 *
 * Every complex product (twiddles, K, Vh) is carried out compensated: the
 * products' rounding errors recovered with fma, the sums' with TwoSum, and
 * each multiplier stored as a float pair plus its residual. A plain float
 * butterfly loses ~1e-7 of the norm per step, systematically (the state
 * changes slowly, so the rounding errors line up); compensated, the norm
 * holds to ~1e-10 a step. */
static const char *FS_FFT = GLSL(
    uniform sampler2D uSrc, uPre, uPost;
    layout(std140) uniform Tw { vec4 uTw[1024]; };
    uniform int uAxis, uN, uNs, uRadix, uPreOn, uPostOn;
    uniform float uSign;
    out vec2 o;
    \n#ifdef EXACT\n
    vec2 cmul(vec2 a, vec4 m) {
        precise float p1 = a.x * m.x, p2 = a.y * m.y, p3 = a.x * m.y, p4 = a.y * m.x;
        precise float e1 = fma(a.x, m.x, -p1), e2 = fma(a.y, m.y, -p2), e3 = fma(a.x, m.y, -p3), e4 = fma(a.y, m.x, -p4);
        precise float c1 = a.x * m.z - a.y * m.w, c2 = a.x * m.w + a.y * m.z;
        precise float s1 = p1 - p2, b1 = s1 - p1, r1 = (p1 - (s1 - b1)) + (-p2 - b1);
        precise float s2 = p3 + p4, b2 = s2 - p3, r2 = (p3 - (s2 - b2)) + (p4 - b2);
        precise vec2 res = vec2(s1 + (((e1 - e2) + r1) + c1), s2 + (((e3 + e4) + r2) + c2));
        return res;
    }
    \n#else\n
    vec2 cmul(vec2 a, vec4 m) { return vec2(a.x * m.x - a.y * m.y, a.x * m.y + a.y * m.x); }
    \n#endif\n
    vec2 fetch(ivec2 p, int k) {
        if (uAxis == 0) p.x = k; else p.y = k;
        vec2 v = texelFetch(uSrc, p, 0).rg;
        if (uPreOn == 1) v = cmul(v, texelFetch(uPre, p, 0));
        return v;
    }
    vec4 tw(int m) { vec4 w = uTw[m]; w.yw *= uSign; return w; }
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        int i = uAxis == 0 ? p.x : p.y;
        vec2 v;
        if (uRadix == 4) {
            int q = i / (4 * uNs), rem = i - q * 4 * uNs;
            int r = rem / uNs, jm = rem - r * uNs;
            int j = q * uNs + jm, st = jm * (1024 / (4 * uNs)), n4 = uN / 4;
            vec2 v0 = fetch(p, j);
            vec2 v1 = cmul(fetch(p, j + n4), tw(st));
            vec2 v2 = cmul(fetch(p, j + 2 * n4), tw(2 * st));
            vec2 v3 = cmul(fetch(p, j + 3 * n4), tw(3 * st));
            vec2 s02 = v0 + v2, d02 = v0 - v2, s13 = v1 + v3, d13 = v1 - v3;
            vec2 jd = vec2(d13.y, -d13.x) * uSign;
            v = r == 0 ? s02 + s13 : r == 1 ? d02 + jd : r == 2 ? s02 - s13 : d02 - jd;
        } else {
            int q = i / (2 * uNs), rem = i - q * 2 * uNs;
            int r = rem >= uNs ? 1 : 0, jm = rem - r * uNs;
            int j = q * uNs + jm;
            vec2 a = fetch(p, j), b = cmul(fetch(p, j + uN / 2), tw(jm * (1024 / (2 * uNs))));
            v = r == 0 ? a + b : a - b;
        }
        if (uPostOn == 1) v = cmul(v, texelFetch(uPost, p, 0));
        o = v;
    });

/* moments, in 8x8 blocks: position space (|psi|^2 weighted 1, x, y, V; the
 * part right of uSplit, x^2, y^2, the peak) or momentum space (1, kx, ky,
 * k^2/2; kx^2, ky^2, -, the peak) */
static const char *FS_RED1 = GLSL(
    uniform sampler2D uSrc, uV;
    uniform int uK;
    uniform ivec2 uN, uWin;
    uniform vec4 uG;
    uniform float uSplit;
    layout(location = 0) out vec4 o0;
    layout(location = 1) out vec4 o1;
    void main() {
        ivec2 b = ivec2(gl_FragCoord.xy) * 8;
        vec4 s0 = vec4(0.0), s1 = vec4(0.0);
        float mx = 0.0;
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) {
                ivec2 p = b + ivec2(i, j);
                vec2 z = texelFetch(uSrc, p, 0).rg;
                float r = dot(z, z);
                if (uK == 0) {
                    if (p.x >= uWin.x || p.y >= uWin.y) continue;
                    float x = uG.x + float(p.x) * uG.z, y = uG.y + float(p.y) * uG.w;
                    float V = texelFetch(uV, p, 0).r;
                    s0 += r * vec4(1.0, x, y, V);
                    s1.xyz += r * vec3(x > uSplit ? 1.0 : 0.0, x * x, y * y);
                } else {
                    float kx = float(p.x < uN.x / 2 ? p.x : p.x - uN.x) * uG.z;
                    float ky = float(p.y < uN.y / 2 ? p.y : p.y - uN.y) * uG.w;
                    s0 += r * vec4(1.0, kx, ky, 0.5 * (kx * kx + ky * ky));
                    s1.xy += r * vec2(kx * kx, ky * ky);
                }
                mx = max(mx, r);
            }
        o0 = s0;
        o1 = vec4(s1.xyz, mx);
    });
static const char *FS_RED2 = GLSL(
    uniform sampler2D uA, uB;
    layout(location = 0) out vec4 o0;
    layout(location = 1) out vec4 o1;
    void main() {
        ivec2 b = ivec2(gl_FragCoord.xy) * 8;
        vec4 s0 = vec4(0.0), s1 = vec4(0.0);
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) {
                s0 += texelFetch(uA, b + ivec2(i, j), 0);
                vec4 t = texelFetch(uB, b + ivec2(i, j), 0);
                s1 = vec4(s1.xyz + t.xyz, max(s1.w, t.w));
            }
        o0 = s0;
        o1 = s1;
    });

/* |psi|^2 into an accumulator (additive blending), from a window of the grid */
static const char *FS_RHO = GLSL(
    uniform sampler2D uSrc;
    uniform ivec2 uOff;
    uniform int uTr;
    uniform float uScale;
    out vec4 o;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        if (uTr == 1) p = p.yx;
        vec2 z = texelFetch(uSrc, p + uOff, 0).rg;
        o = vec4(dot(z, z) * uScale);
    });

/* a separable blur (and, first, a 2x downsample of |psi|^2) for the glow */
static const char *FS_GLOW = GLSL(
    uniform sampler2D uSrc;
    uniform int uPass;
    uniform vec2 uTexel;
    out vec4 o;
    void main() {
        vec2 uv = gl_FragCoord.xy * uTexel;
        if (uPass == 0) {
            vec2 a = texture(uSrc, uv + uTexel * vec2(-0.25, -0.25)).rg, b = texture(uSrc, uv + uTexel * vec2(0.25, -0.25)).rg;
            vec2 c = texture(uSrc, uv + uTexel * vec2(-0.25, 0.25)).rg, d = texture(uSrc, uv + uTexel * vec2(0.25, 0.25)).rg;
            o = vec4(0.25 * (dot(a, a) + dot(b, b) + dot(c, c) + dot(d, d)));
            return;
        }
        vec2 dir = uPass == 1 ? vec2(uTexel.x, 0.0) : vec2(0.0, uTexel.y);
        float s = 0.0, w = 0.0;
        for (int i = -6; i <= 6; i++) {
            float g = exp(-float(i * i) / 12.0);
            s += g * texture(uSrc, uv + dir * float(i)).r;
            w += g;
        }
        o = vec4(s / w);
    });

/* the momentum distribution along kx: |phi|^2 summed over ky */
static const char *FS_MARG = GLSL(
    uniform sampler2D uSrc;
    uniform int uNy;
    out vec4 o;
    void main() {
        int i = int(gl_FragCoord.x);
        float s = 0.0;
        for (int j = 0; j < uNy; j++) { vec2 z = texelFetch(uSrc, ivec2(i, j), 0).rg; s += dot(z, z); }
        o = vec4(s);
    });

/* the revival: phi0 . exp(-i k^2 t / 2) with the phase 2 pi (m^2 s mod 2^20) / 2^20
 * in exact integer arithmetic (t = s/2^20 of the revival time), so any t is exact */
static const char *FS_PHASE = GLSL(
    uniform sampler2D uSrc;
    uniform ivec2 uN;
    uniform uint uS;
    uniform float uScale;
    out vec2 o;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        int mx = p.x < uN.x / 2 ? p.x : p.x - uN.x, my = p.y < uN.y / 2 ? p.y : p.y - uN.y;
        uint m2 = uint(mx * mx + my * my);
        uint ph = (m2 * uS) & 1048575u;
        float a = -6.283185307 * float(ph) / 1048576.0;
        vec2 z = texelFetch(uSrc, p, 0).rg, e = vec2(cos(a), sin(a));
        o = vec2(z.x * e.x - z.y * e.y, z.x * e.y + z.y * e.x) * uScale;
    });

static GLuint prog_fft, prog_red1, prog_red2, prog_rho, prog_glow, prog_phase, prog_marg;
static GLint fu_axis, fu_n, fu_ns, fu_radix, fu_pre, fu_post, fu_sign;
static GLuint ubo_tw;

/* the multiplier exp(i t) * scale as a float pair and its residual */
static void mult(double t, double scale, float *o) {
    double c = cos(t) * scale, s = sin(t) * scale;
    o[0] = (float)c; o[1] = (float)s;
    o[2] = (float)(c - o[0]); o[3] = (float)(s - o[1]);
}

static int log2i(int n) { int l = 0; while ((1 << l) < n) l++; return l; }

static int pass_k;                      /* passes so far in this transform: picks the ping-pong buffer */
static void fft_pass(GLuint src, Pair *b, int axis, int n, int ns, int radix, float sign, GLuint pre, GLuint post) {
    bind(0, src);
    glBindFramebuffer(GL_FRAMEBUFFER, b->f[pass_k & 1]);
    glUniform1i(fu_axis, axis); glUniform1i(fu_n, n); glUniform1i(fu_ns, ns); glUniform1i(fu_radix, radix);
    glUniform1f(fu_sign, sign);
    glUniform1i(fu_pre, pre ? 1 : 0);
    glUniform1i(fu_post, post ? 1 : 0);
    if (pre) bind(1, pre);
    if (post) bind(2, post);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    pass_k++;
}

/* transform along one axis: src (or, after the first pass, the buffer last
 * written) -> b; pre multiplies the first pass's input, post the last's output.
 * Returns the texture holding the result. */
static GLuint fft_axis(GLuint src, Pair *b, int axis, int n, float sign, GLuint pre, GLuint post) {
    int ns = 1, l = log2i(n), last = l / 2 + l % 2, k = 0;
    for (; ns * 4 <= n; ns *= 4, k++) {
        fft_pass(src, b, axis, n, ns, 4, sign, pre, k == last - 1 ? post : 0);
        src = b->t[(pass_k - 1) & 1]; pre = 0;
    }
    for (; ns < n; ns *= 2, k++) {
        fft_pass(src, b, axis, n, ns, 2, sign, pre, k == last - 1 ? post : 0);
        src = b->t[(pass_k - 1) & 1]; pre = 0;
    }
    return src;
}

static void fft_begin(int w, int h) {
    glUseProgram(prog_fft);
    glViewport(0, 0, w, h);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
}

/* ---- the grid ---- */
typedef struct {
    int nx, ny;
    double x0, y0, dx, dy, dt;
    Pair psi;                          /* the state lives in psi.t[0] between steps */
    Pair meas;                         /* the momentum-space picture */
    GLuint vh, kin, vtex;              /* RGBA32F multipliers; R32F V for the picture and <V> */
    GLuint glow[2], fglow[2];          /* half-res |psi|^2, blurred */
    GLuint red1[2], fred1, red2[2], fred2;
    int gw, gh;
    GLuint kshow;                      /* the texture holding the momentum picture */
    int kstale;                        /* steps taken since it was made */
    double vmax;                       /* V at which the relief saturates */
    double norm0, rmax0;               /* the norm uploaded (sum |psi|^2 dx dy), and the peak of |psi|^2 */
    int win_x, win_y;                  /* moments only over [0, win) (the revival's quadrant) */
    double split;                      /* moments: the probability right of this x */
} Grid;
static Grid G;

static void grid_alloc(int nx, int ny) {
    if (G.nx == nx && G.ny == ny) return;
    if (G.nx) {
        glDeleteTextures(3, &G.vh);
        glDeleteTextures(2, G.glow); glDeleteFramebuffers(2, G.fglow);
        glDeleteTextures(2, G.red1); glDeleteFramebuffers(1, &G.fred1);
        glDeleteTextures(2, G.red2); glDeleteFramebuffers(1, &G.fred2);
    }
    pair_free(&G.psi); pair_free(&G.meas);
    G.nx = nx; G.ny = ny;
    pair_init(&G.psi, nx, ny, GL_RG32F, GL_RG, 1, 1);
    pair_init(&G.meas, nx, ny, GL_RG32F, GL_RG, 1, 1);
    G.vh = mk_tex(nx, ny, GL_RGBA32F, GL_RGBA, NULL, 0, 1);
    G.kin = mk_tex(nx, ny, GL_RGBA32F, GL_RGBA, NULL, 0, 1);
    G.vtex = mk_tex(nx, ny, GL_R32F, GL_RED, NULL, 1, 1);
    G.gw = nx / 2; G.gh = ny / 2;
    for (int i = 0; i < 2; i++) {
        G.glow[i] = mk_tex(G.gw, G.gh, GL_R16F, GL_RED, NULL, 1, 1);
        G.fglow[i] = mk_fbo(G.glow[i], 0);
    }
    for (int i = 0; i < 2; i++) G.red1[i] = mk_tex(nx / 8, ny / 8, GL_RGBA32F, GL_RGBA, NULL, 0, 0);
    G.fred1 = mk_fbo(G.red1[0], G.red1[1]);
    for (int i = 0; i < 2; i++) G.red2[i] = mk_tex(nx / 64, ny / 64, GL_RGBA32F, GL_RGBA, NULL, 0, 0);
    G.fred2 = mk_fbo(G.red2[0], G.red2[1]);
}

/* upload a state and its world: psi (complex, 2 per point), V and the
 * absorbing layer W (both may be NULL). psi is normalised to 1 over dx dy. */
static void grid_upload(const double *psi, const double *V, const double *Wab) {
    int n = G.nx * G.ny;
    float *buf = malloc(sizeof(float) * 4 * (size_t)n);
    double s = 0;
    for (int i = 0; i < n; i++) s += psi[2 * i] * psi[2 * i] + psi[2 * i + 1] * psi[2 * i + 1];
    double sc = s > 0 ? 1 / sqrt(s * G.dx * G.dy) : 1;
    double s2 = 0;
    G.rmax0 = 0;
    for (int i = 0; i < 2 * n; i++) { buf[i] = (float)(psi[i] * sc); s2 += (double)buf[i] * buf[i]; }
    for (int i = 0; i < n; i++) G.rmax0 = fmax(G.rmax0, (double)buf[2 * i] * buf[2 * i] + (double)buf[2 * i + 1] * buf[2 * i + 1]);
    G.norm0 = s2 * G.dx * G.dy;
    glBindTexture(GL_TEXTURE_2D, G.psi.t[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.nx, G.ny, GL_RG, GL_FLOAT, buf);
    for (int i = 0; i < n; i++) {
        double v = V ? V[i] : 0, w = Wab ? Wab[i] : 0;
        mult(-v * G.dt / 2, exp(-w * G.dt / 2), buf + 4 * i);
    }
    glBindTexture(GL_TEXTURE_2D, G.vh);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.nx, G.ny, GL_RGBA, GL_FLOAT, buf);
    double sx = 2 * PI / (G.nx * G.dx), sy = 2 * PI / (G.ny * G.dy);
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double kx = (i < G.nx / 2 ? i : i - G.nx) * sx, ky = (j < G.ny / 2 ? j : j - G.ny) * sy;
            mult(-(kx * kx + ky * ky) / 2 * G.dt, 1.0 / ((double)G.nx * G.ny), buf + 4 * (j * G.nx + i));
        }
    glBindTexture(GL_TEXTURE_2D, G.kin);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.nx, G.ny, GL_RGBA, GL_FLOAT, buf);
    for (int i = 0; i < n; i++) buf[i] = V ? (float)V[i] : 0;
    glBindTexture(GL_TEXTURE_2D, G.vtex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.nx, G.ny, GL_RED, GL_FLOAT, buf);
    free(buf);
    G.win_x = G.nx; G.win_y = G.ny;
    G.split = 1e30;
    G.kshow = 0;
}

/* one Strang step: psi <- Vh F^-1 K F Vh psi, back into psi.t[0] */
static void grid_step(void) {
    fft_begin(G.nx, G.ny);
    pass_k = 1;                        /* the first pass reads psi.t[0], so writes psi.t[1]; an even count ends in t[0] */
    GLuint t = fft_axis(G.psi.t[0], &G.psi, 0, G.nx, 1, G.vh, 0);
    t = fft_axis(t, &G.psi, 1, G.ny, 1, 0, G.kin);
    t = fft_axis(t, &G.psi, 0, G.nx, -1, 0, 0);
    fft_axis(t, &G.psi, 1, G.ny, -1, 0, G.vh);
}

/* the momentum-space picture of the current state (unnormalised) */
static void grid_measure_fft(void) {
    fft_begin(G.nx, G.ny);
    pass_k = 0;
    GLuint t = fft_axis(G.psi.t[0], &G.meas, 0, G.nx, 1, 0, 0);
    G.kshow = fft_axis(t, &G.meas, 1, G.ny, 1, 0, 0);
    G.kstale = 0;
}

/* moments */
typedef struct {
    double n, x, y, V, right, xx, yy, rmax;    /* position space, n = norm */
    double kn, kx, ky, T, kxx, kyy, kmax;      /* momentum space (kn: Parseval's norm) */
    int have_k;
} Mom;
static Mom M;

/* live, the numbers come back a frame late through pixel-buffer objects, so
 * the CPU never waits on the GPU in the middle of a frame; offline (shots,
 * --check) they are read at once */
static int async_rd;
typedef struct { GLuint pbo; int bytes, pending, n; } Rd;
static Rd rd[5];
static void rd_issue(int k, int w, int h, GLenum fmt, int comps, GLenum att, int n) {
    Rd *r = &rd[k];
    int bytes = w * h * comps * 4;
    if (!r->pbo) glGenBuffers(1, &r->pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, r->pbo);
    if (r->bytes < bytes) { glBufferData(GL_PIXEL_PACK_BUFFER, bytes, NULL, GL_STREAM_READ); r->bytes = bytes; }
    glReadBuffer(att);
    glReadPixels(0, 0, w, h, fmt, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    r->pending = 1; r->n = n;
}
static int rd_fetch(int k, void *dst, int bytes) {
    Rd *r = &rd[k];
    if (!r->pending) return 0;
    r->pending = 0;
    glBindBuffer(GL_PIXEL_PACK_BUFFER, r->pbo);
    void *m = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT);
    if (m) memcpy(dst, m, (size_t)bytes);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    return m != NULL;
}
static void rd_drop(void) { for (int k = 0; k < 5; k++) rd[k].pending = 0; }

static void reduce(GLuint src, int kmode, float *o) {
    glUseProgram(prog_red1);
    glBindFramebuffer(GL_FRAMEBUFFER, G.fred1);
    glViewport(0, 0, G.nx / 8, G.ny / 8);
    bind(0, src);
    bind(1, G.vtex);
    glUniform1i(U(prog_red1, "uK"), kmode);
    glUniform2i(U(prog_red1, "uN"), G.nx, G.ny);
    glUniform2i(U(prog_red1, "uWin"), G.win_x, G.win_y);
    if (kmode) glUniform4f(U(prog_red1, "uG"), 0, 0, (float)(2 * PI / (G.nx * G.dx)), (float)(2 * PI / (G.ny * G.dy)));
    else glUniform4f(U(prog_red1, "uG"), (float)G.x0, (float)G.y0, (float)G.dx, (float)G.dy);
    glUniform1f(U(prog_red1, "uSplit"), (float)G.split);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(prog_red2);
    glBindFramebuffer(GL_FRAMEBUFFER, G.fred2);
    glViewport(0, 0, G.nx / 64, G.ny / 64);
    bind(0, G.red1[0]);
    bind(1, G.red1[1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    int n = (G.nx / 64) * (G.ny / 64);
    if (async_rd) {
        rd_issue(2 * kmode, G.nx / 64, G.ny / 64, GL_RGBA, 4, GL_COLOR_ATTACHMENT0, n);
        rd_issue(2 * kmode + 1, G.nx / 64, G.ny / 64, GL_RGBA, 4, GL_COLOR_ATTACHMENT1, n);
        return;
    }
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, G.nx / 64, G.ny / 64, GL_RGBA, GL_FLOAT, o);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, G.nx / 64, G.ny / 64, GL_RGBA, GL_FLOAT, o + 4 * n);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
}

static void mom_x(const float *buf, int n);
static void mom_k(const float *buf, int n);
static long meas_frame;
static void measure(int with_k) {
    static float buf[2 * 4 * 256];
    int n = (G.nx / 64) * (G.ny / 64);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
    if (async_rd) {
        /* last frame's numbers, then this frame's sent off */
        int pn = rd[0].n;
        if (rd[0].pending && rd_fetch(0, buf, pn * 16) && rd_fetch(1, buf + 4 * pn, pn * 16)) mom_x(buf, pn);
        pn = rd[2].n;
        if (rd[2].pending && rd_fetch(2, buf, pn * 16) && rd_fetch(3, buf + 4 * pn, pn * 16)) mom_k(buf, pn);
        reduce(G.psi.t[0], 0, buf);
        if (!with_k) return;
        /* the momentum picture: every frame on small grids, every other on big ones */
        meas_frame++;
        if (!G.kshow || (G.kstale && (G.nx * G.ny <= 65536 || (meas_frame & 1)))) {
            grid_measure_fft();
            reduce(G.kshow, 1, buf);
        }
        return;
    }
    reduce(G.psi.t[0], 0, buf);
    mom_x(buf, n);
    if (!with_k) return;
    if (!G.kshow || G.kstale) grid_measure_fft();
    reduce(G.kshow, 1, buf);
    mom_k(buf, n);
}
static void mom_x(const float *buf, int n) {
    double s[8] = {0}, mx = 0;
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 4; c++) s[c] += buf[4 * i + c];
        for (int c = 0; c < 3; c++) s[4 + c] += buf[4 * n + 4 * i + c];
        if (buf[4 * n + 4 * i + 3] > mx) mx = buf[4 * n + 4 * i + 3];
    }
    double dA = G.dx * G.dy, nn = s[0] > 0 ? s[0] : 1;
    M.n = s[0] * dA; M.x = s[1] / nn; M.y = s[2] / nn; M.V = s[3] / nn;
    M.right = s[4] / nn; M.xx = s[5] / nn; M.yy = s[6] / nn; M.rmax = mx;
}
static void mom_k(const float *buf, int n) {
    double k[8] = {0}, mx = 0;
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 4; c++) k[c] += buf[4 * i + c];
        for (int c = 0; c < 3; c++) k[4 + c] += buf[4 * n + 4 * i + c];
        if (buf[4 * n + 4 * i + 3] > mx) mx = buf[4 * n + 4 * i + 3];
    }
    double kn = k[0] > 0 ? k[0] : 1;
    M.kn = k[0]; M.kx = k[1] / kn; M.ky = k[2] / kn; M.T = k[3] / kn; M.kxx = k[4] / kn; M.kyy = k[5] / kn; M.kmax = mx;
    M.have_k = 1;
}

static GLuint marg_tex, marg_fbo;
static float kmarg[1024];
static int have_marg;
static void marginal(void) {
    if (!G.kshow) return;
    if (!marg_tex) { marg_tex = mk_tex(1024, 1, GL_R32F, GL_RED, NULL, 0, 0); marg_fbo = mk_fbo(marg_tex, 0); }
    glUseProgram(prog_marg);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, marg_fbo);
    glViewport(0, 0, G.nx, 1);
    glUniform1i(U(prog_marg, "uNy"), G.ny);
    bind(0, G.kshow);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    if (async_rd) {
        if (rd[4].pending && rd_fetch(4, kmarg, rd[4].n * 4)) have_marg = 1;
        rd_issue(4, G.nx, 1, GL_RED, 1, GL_COLOR_ATTACHMENT0, G.nx);
        return;
    }
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, G.nx, 1, GL_RED, GL_FLOAT, kmarg);
    have_marg = 1;
}

/* the whole state, read back (for --check) */
static float *grid_read(void) {
    float *b = malloc(sizeof(float) * 2 * (size_t)G.nx * G.ny);
    glBindTexture(GL_TEXTURE_2D, G.psi.t[0]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, b);
    return b;
}

/* the glow: |psi|^2 at half resolution, blurred both ways */
static void make_glow(GLuint src) {
    glUseProgram(prog_glow);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
    glViewport(0, 0, G.gw, G.gh);
    glUniform2f(U(prog_glow, "uTexel"), 1.0f / G.gw, 1.0f / G.gh);
    glUniform1i(U(prog_glow, "uPass"), 0);
    bind(0, src);
    glBindFramebuffer(GL_FRAMEBUFFER, G.fglow[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUniform1i(U(prog_glow, "uPass"), 1);
    bind(0, G.glow[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, G.fglow[1]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUniform1i(U(prog_glow, "uPass"), 2);
    bind(0, G.glow[1]);
    glBindFramebuffer(GL_FRAMEBUFFER, G.fglow[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* add |psi|^2 (times scale) over a w x h window at (ox, oy) into fbo at (dx, dy) */
static void accumulate(GLuint src, GLuint fbo, int ox, int oy, int dx, int dy, int w, int h, float scale, int tr) {
    glUseProgram(prog_rho);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(dx, dy, w, h);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glUniform2i(U(prog_rho, "uOff"), ox - dx, oy - dy);
    glUniform1i(U(prog_rho, "uTr"), tr);
    glUniform1f(U(prog_rho, "uScale"), scale);
    bind(0, src);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

/* ==================================================================== */
/* ---- text: a glyph cache over freetype, any codepoint, any size ---- */
enum { F_MATH, F_SANS, F_LIGHT, F_FALLBACK, NFACE };
static const char *face_pat[NFACE] = {"Latin Modern Math", "Inter:style=Regular", "Inter:style=Light", "DejaVu Math TeX Gyre"};
static FT_Library ft;
static FT_Face faces[NFACE];
typedef struct { int face, px; uint32_t cp; short x, y, w, h, left, top; float adv; int used; } GEnt;
#define GCACHE 8192
static GEnt gcache[GCACHE];
#define ATLAS 2048
static GLuint atlas_tex;
static int at_x = 1, at_y = 1, at_shelf, atlas_full, glyphs_used;

static void fonts_init(void) {
    if (!FcInit()) die("fontconfig init failed");
    if (FT_Init_FreeType(&ft)) die("freetype init failed");
    for (int i = 0; i < NFACE; i++) {
        FcPattern *pat = FcNameParse((const FcChar8 *)face_pat[i]);
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        FcResult res;
        FcPattern *m = FcFontMatch(NULL, pat, &res);
        FcChar8 *file = NULL;
        if (!m || FcPatternGetString(m, FC_FILE, 0, &file) != FcResultMatch) die("no font matched");
        if (FT_New_Face(ft, (const char *)file, 0, &faces[i])) die("can't load a font");
        FcPatternDestroy(m);
        FcPatternDestroy(pat);
    }
}

static void atlas_reset(void) {
    memset(gcache, 0, sizeof gcache);
    at_x = 1; at_y = 1; at_shelf = 0; atlas_full = 0; glyphs_used = 0;
    if (!atlas_tex) {
        glGenTextures(1, &atlas_tex);
        glBindTexture(GL_TEXTURE_2D, atlas_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glBindTexture(GL_TEXTURE_2D, atlas_tex);
    unsigned char *z = calloc(ATLAS * ATLAS, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, ATLAS, ATLAS, 0, GL_RED, GL_UNSIGNED_BYTE, z);
    free(z);
}

static const GEnt *glyph(int face, int px, uint32_t cp) {
    uint32_t h = (cp * 2654435761u) ^ ((uint32_t)px * 40503u) ^ ((uint32_t)face * 97u);
    for (int k = 0; k < GCACHE; k++) {
        GEnt *e = &gcache[(h + (uint32_t)k) % GCACHE];
        if (e->used && e->face == face && e->px == px && e->cp == cp) return e;
        if (e->used) continue;
        if (++glyphs_used > GCACHE * 3 / 4) atlas_full = 1;   /* sizes come and go: start over next frame */
        FT_Face f = faces[face];
        /* what a face lacks (Latin Modern has no hbar), from the fallback */
        if (!FT_Get_Char_Index(f, cp) && FT_Get_Char_Index(faces[F_FALLBACK], cp)) f = faces[F_FALLBACK];
        FT_Set_Pixel_Sizes(f, 0, (FT_UInt)px);
        memset(e, 0, sizeof *e);
        e->used = 1; e->face = face; e->px = px; e->cp = cp;
        if (FT_Load_Char(f, cp, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT)) return e;
        FT_GlyphSlot g = f->glyph;
        int w = (int)g->bitmap.width, gh = (int)g->bitmap.rows;
        e->adv = g->advance.x / 64.0f;
        e->left = (short)g->bitmap_left; e->top = (short)g->bitmap_top;
        if (!w || !gh) return e;
        if (at_x + w + 2 > ATLAS) { at_x = 1; at_y += at_shelf + 2; at_shelf = 0; }
        if (at_y + gh + 2 > ATLAS) { atlas_full = 1; return e; }   /* full: blank until the next frame resets it */
        glBindTexture(GL_TEXTURE_2D, atlas_tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, g->bitmap.pitch);
        glTexSubImage2D(GL_TEXTURE_2D, 0, at_x, at_y, w, gh, GL_RED, GL_UNSIGNED_BYTE, g->bitmap.buffer);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        e->x = (short)at_x; e->y = (short)at_y; e->w = (short)w; e->h = (short)gh;
        at_x += w + 2;
        if (gh > at_shelf) at_shelf = gh;
        return e;
    }
    return &gcache[0];
}

static uint32_t utf8(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    uint32_t c = *p++;
    if (c >= 0xF0) { c = (c & 7) << 18 | (p[0] & 63) << 12 | (p[1] & 63) << 6 | (p[2] & 63); p += 3; }
    else if (c >= 0xE0) { c = (c & 15) << 12 | (p[0] & 63) << 6 | (p[1] & 63); p += 2; }
    else if (c >= 0xC0) { c = (c & 31) << 6 | (p[0] & 63); p += 1; }
    *s = (const char *)p;
    return c;
}

/* ---- the overlay: text, lines, dots, fills, in one vertex stream ----
 * x, y (px, from the top left), u, v, r, g, b, a, kind, param */
#define OVF 10
static float *ov;
static int ovn, ovcap;
static void ov_v(float x, float y, float u, float v, const float *c, float kind, float prm) {
    if (ovn + OVF > ovcap) ov = realloc(ov, (size_t)(ovcap = ovcap ? ovcap * 2 : 65536) * sizeof *ov);
    float *o = ov + ovn;
    o[0] = x; o[1] = y; o[2] = u; o[3] = v; o[4] = c[0]; o[5] = c[1]; o[6] = c[2]; o[7] = c[3]; o[8] = kind; o[9] = prm;
    ovn += OVF;
}
static void ov_quad(const float q[4][4], const float *c, float kind, float prm) {
    ov_v(q[0][0], q[0][1], q[0][2], q[0][3], c, kind, prm); ov_v(q[1][0], q[1][1], q[1][2], q[1][3], c, kind, prm);
    ov_v(q[2][0], q[2][1], q[2][2], q[2][3], c, kind, prm); ov_v(q[0][0], q[0][1], q[0][2], q[0][3], c, kind, prm);
    ov_v(q[2][0], q[2][1], q[2][2], q[2][3], c, kind, prm); ov_v(q[3][0], q[3][1], q[3][2], q[3][3], c, kind, prm);
}
static void rgba(float *o, const float *c, float a) { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = a; }

static void fill(float x0, float y0, float x1, float y1, const float *c) {
    float q[4][4] = {{x0, y0, 0, 0}, {x1, y0, 0, 0}, {x1, y1, 0, 0}, {x0, y1, 0, 0}};
    ov_quad(q, c, 2, 0);
}
/* an antialiased line w px wide */
static void line(float x0, float y0, float x1, float y1, float w, const float *c) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    if (l < 1e-4f) return;
    dx /= l; dy /= l;
    float hw = w * 0.5f + 1.0f, nx = -dy * hw, ny = dx * hw, ex = dx * w * 0.5f, ey = dy * w * 0.5f;
    float q[4][4] = {{x0 - ex + nx, y0 - ey + ny, hw, 0}, {x1 + ex + nx, y1 + ey + ny, hw, 0},
                     {x1 + ex - nx, y1 + ey - ny, -hw, 0}, {x0 - ex - nx, y0 - ey - ny, -hw, 0}};
    ov_quad(q, c, 1, w * 0.5f);
}
static void dashed(float x0, float y0, float x1, float y1, float w, float on, float off, const float *c) {
    float dx = x1 - x0, dy = y1 - y0, l = sqrtf(dx * dx + dy * dy);
    for (float s = 0; s < l; s += on + off) {
        float e = fminf(s + on, l);
        line(x0 + dx * s / l, y0 + dy * s / l, x0 + dx * e / l, y0 + dy * e / l, w, c);
    }
}
static void dot(float x, float y, float r, const float *c) {
    float R = r + 1;
    float q[4][4] = {{x - R, y - R, -R, -R}, {x + R, y - R, R, -R}, {x + R, y + R, R, R}, {x - R, y + R, -R, R}};
    ov_quad(q, c, 3, r);
}
static void ring(float x, float y, float r, float w, const float *c, int dash) {
    int n = (int)clampf(r * 0.8f, 24, 256);
    for (int i = 0; i < n; i++) {
        if (dash && (i & 1)) continue;
        float a0 = 2 * (float)PI * i / n, a1 = 2 * (float)PI * (i + 1) / n;
        line(x + r * cosf(a0), y + r * sinf(a0), x + r * cosf(a1), y + r * sinf(a1), w, c);
    }
}

/* plain text: face at px, baseline y, align -1 left 0 centre 1 right;
 * track in px; tab: digits on a fixed advance */
static float text_run(int face, int px, float x, float y, const char *s, const float *c, float track, int tab, int emit) {
    float pen = x, dw = tab ? glyph(face, px, '0')->adv : 0;
    while (*s) {
        uint32_t cp = utf8(&s);
        const GEnt *g = glyph(face, px, cp);
        float adv = g->adv, off = 0;
        if (tab && cp >= '0' && cp <= '9') { off = (dw - adv) * 0.5f; adv = dw; }
        if (emit && g->w) {
            float x0 = roundf(pen + off) + g->left, y0 = roundf(y) - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float q[4][4] = {{x0, y0, g->x, g->y}, {x1, y0, g->x + g->w, g->y}, {x1, y1, g->x + g->w, g->y + g->h},
                             {x0, y1, g->x, g->y + g->h}};
            ov_quad(q, c, 0, 0);
        }
        pen += adv + (*s ? track : 0);
    }
    return pen - x;
}
static float text(int face, float px, float x, float y, int align, const char *s, const float *c, float track, int tab) {
    int p = (int)lroundf(px);
    if (p < 6) p = 6;
    float w = text_run(face, p, 0, 0, s, c, track, tab, 0);
    float x0 = x - (align == 0 ? w * 0.5f : align > 0 ? w : 0);
    text_run(face, p, x0, y, s, c, track, tab, 1);
    return w;
}

/* ---- maths: a little typesetter ----
 * letters are set in math italic; \f{a}{b} a fraction, ^{..} _{..} scripts,
 * \r{..} upright, \t{..} in the text face, \, \; \q spaces. */
typedef struct { float w, asc, desc; } Box;
static uint32_t math_italic(uint32_t c) {
    if (c == 'h') return 0x210E;
    if (c == 0x127) return 0x210F;
    if (c >= 'a' && c <= 'z') return 0x1D44E + (c - 'a');
    if (c >= 'A' && c <= 'Z') return 0x1D434 + (c - 'A');
    if (c >= 0x3B1 && c <= 0x3C9) return 0x1D6FC + (c - 0x3B1);
    return c;
}
static const float *m_col;
static Box m_seq(const char **s, float px, float x, float y, int emit, int upright, int face);
static Box m_group(const char **s, float px, float x, float y, int emit, int upright, int face) {
    if (**s == '{') { (*s)++; Box b = m_seq(s, px, x, y, emit, upright, face); if (**s == '}') (*s)++; return b; }
    char one[8] = {0};
    const char *p = *s;
    utf8(s);
    memcpy(one, p, (size_t)(*s - p));
    const char *q = one;
    return m_seq(&q, px, x, y, emit, upright, face);
}
static Box m_seq(const char **s, float px, float x, float y, int emit, int upright, int face) {
    Box b = {0, 0, 0};
    float last_w = 0;                  /* the width of the last atom, for its scripts */
    int ip = (int)lroundf(px);
    if (ip < 6) ip = 6;
    while (**s && **s != '}') {
        const char *c = *s;
        if (c[0] == '\\') {
            char cmd = c[1];
            *s += 2;
            if (cmd == ',') { b.w += px * 0.17f; continue; }
            if (cmd == ';') { b.w += px * 0.28f; continue; }
            if (cmd == 'q') { b.w += px * 1.0f; continue; }
            if (cmd == 'o') {
                Box g = m_group(s, px, x + b.w, y, emit, upright, face);
                float th = fmaxf(1.0f, px * 0.045f), yl = y - g.asc - px * 0.12f;
                if (emit) fill(x + b.w + px * 0.04f, yl - th * 0.5f, x + b.w + g.w - px * 0.02f, yl + th * 0.5f, m_col);
                b.w += g.w; b.asc = fmaxf(b.asc, g.asc + px * 0.18f); b.desc = fmaxf(b.desc, g.desc); last_w = g.w;
                continue;
            }
            if (cmd == 'r' || cmd == 't') {
                Box g = m_group(s, px, x + b.w, y, emit, 1, cmd == 't' ? F_SANS : face);
                b.w += g.w; b.asc = fmaxf(b.asc, g.asc); b.desc = fmaxf(b.desc, g.desc); last_w = g.w;
                continue;
            }
            if (cmd == 'f') {
                float sp = px * 0.86f, axis = px * 0.25f, gap = px * 0.16f, th = fmaxf(1.0f, px * 0.045f);
                const char *s0 = *s;
                Box n = m_group(s, sp, 0, 0, 0, upright, face);
                Box d = m_group(s, sp, 0, 0, 0, upright, face);
                float wd = fmaxf(n.w, d.w) + px * 0.24f;
                if (emit) {
                    *s = s0;
                    m_group(s, sp, x + b.w + (wd - n.w) * 0.5f, y - axis - gap - n.desc, 1, upright, face);
                    m_group(s, sp, x + b.w + (wd - d.w) * 0.5f, y - axis + gap + d.asc, 1, upright, face);
                    fill(x + b.w + px * 0.06f, y - axis - th * 0.5f, x + b.w + wd - px * 0.06f, y - axis + th * 0.5f, m_col);
                }
                b.asc = fmaxf(b.asc, axis + gap + n.desc + n.asc);
                b.desc = fmaxf(b.desc, gap + d.asc + d.desc - axis);
                b.w += wd; last_w = wd;
                continue;
            }
            continue;
        }
        if (c[0] == '^' || c[0] == '_') {
            /* scripts on the last atom: one or both */
            float sp = px * 0.68f, xs = x + b.w;
            float wsup = 0, wsub = 0;
            int both = 0;
            for (int k = 0; k < 2 && (**s == '^' || **s == '_'); k++) {
                int sup = **s == '^';
                (*s)++;
                if (k == 0) {
                    const char *t = *s;
                    Box tmp = m_group(&t, sp, 0, 0, 0, upright, face);
                    (void)tmp;
                    both = (*t == '^' || *t == '_');
                }
                float dy = sup ? -px * (both ? 0.44f : 0.40f) : px * (both ? 0.26f : 0.18f);
                Box g = m_group(s, sp, xs + (sup ? px * 0.03f : 0), y + dy, emit, upright, face);
                if (sup) { wsup = g.w; b.asc = fmaxf(b.asc, -dy + g.asc); }
                else { wsub = g.w; b.desc = fmaxf(b.desc, dy + g.desc); }
            }
            b.w += fmaxf(wsup, wsub) + px * 0.04f;
            (void)last_w;
            continue;
        }
        uint32_t cp = utf8(s);
        if (cp == ' ') { b.w += px * 0.22f; continue; }
        if (cp == '-') cp = 0x2212;
        if (cp == '\'') cp = 0x2032;
        if (face == F_MATH) cp = upright ? (cp == 0x127 ? 0x210F : cp) : math_italic(cp);
        const GEnt *g = glyph(face, ip, cp);
        if (emit && g->w) {
            float x0 = roundf(x + b.w) + g->left, y0 = roundf(y) - g->top, x1 = x0 + g->w, y1 = y0 + g->h;
            float q[4][4] = {{x0, y0, g->x, g->y}, {x1, y0, g->x + g->w, g->y}, {x1, y1, g->x + g->w, g->y + g->h},
                             {x0, y1, g->x, g->y + g->h}};
            ov_quad(q, m_col, 0, 0);
        }
        b.asc = fmaxf(b.asc, g->top);
        b.desc = fmaxf(b.desc, g->h - g->top);
        b.w += g->adv; last_w = g->adv;
    }
    return b;
}
static Box math(float px, float x, float y, int align, const char *src, const float *c) {
    const char *s = src;
    m_col = c;
    Box b = m_seq(&s, px, 0, 0, 0, 0, F_MATH);
    float x0 = x - (align == 0 ? b.w * 0.5f : align > 0 ? b.w : 0);
    s = src;
    m_seq(&s, px, x0, y, 1, 0, F_MATH);
    return b;
}
static Box math_measure(float px, const char *src) {
    const char *s = src;
    float c[4] = {0};
    m_col = c;
    return m_seq(&s, px, 0, 0, 0, 0, F_MATH);
}

/* ==================================================================== */
/* ---- the acts ---- */
enum { A_TUNNEL, A_SLIT, A_COHERENT, A_SQUEEZED, A_REVIVAL, A_BRAGG, A_STADIUM, NACT };
static const char *act_key[NACT] = {"tunnel", "slit", "coherent", "squeezed", "revival", "bragg", "stadium"};
static const double act_len[NACT] = {49, 47, 52, 52, 66, 38, 64};
#define INTRO 2.5                      /* seconds of title before the sim starts */
#define FADE 1.6

typedef struct {
    int id, variant, num;              /* which act, its variant, its place in the cycle (1..) */
    const char *title, *sub, *eq[2];
    double sps;                        /* steps per second of the act's clock */
    long max_steps;
    double vx0, vy0, vx1, vy1;         /* the region shown */
    double kview, kcx;                 /* the momentum panel's half-width and centre (kx) */
    float expo_k;                      /* exposure of the momentum picture */
    /* tunnelling */
    double k0, V0, a, T_th, x_start, sigx;
    /* double slit */
    double d, w, xs, xwall, lam, Lscr;
    /* oscillators */
    double wx, wy, x0, y0, px0, py0, s0;
    /* revival */
    double Lbox, trev;
    /* bragg */
    double latt, lrot;
    /* stadium */
    double R, alen;
    int corral;
} Act;
static Act A;
static long steps;                     /* done in this act */
static int act_cur = -1, act_cycle = -1;
static double act_t;                   /* local time in the act */
static double sim_t;                   /* the act's physical time (steps * dt) */

/* history for the plots: (t, a, b, c) per frame */
#define HMAX 4096
enum { H_T, H_X, H_Y, H_PX, H_A, H_B, HCOL };
static float hist[HMAX][HCOL];
static int hn;
static void hist_add(float t, float x, float y, float px, float a, float b) {
    if (hn && t <= hist[hn - 1][0]) return;
    if (hn == HMAX) { memmove(hist, hist + 1, sizeof(hist[0]) * (HMAX - 1)); hn--; }
    float *h = hist[hn++];
    h[H_T] = t; h[H_X] = x; h[H_Y] = y; h[H_PX] = px; h[H_A] = a; h[H_B] = b;
}

/* square barrier transmission, E in units hbar = m = 1 */
static double T_square(double E, double V0, double a) {
    if (E <= 0) return 0;
    if (fabs(E - V0) < 1e-9) return 1 / (1 + V0 * a * a / 2);
    if (E < V0) {
        double kap = sqrt(2 * (V0 - E)), s = sinh(kap * a);
        return 1 / (1 + V0 * V0 * s * s / (4 * E * (V0 - E)));
    }
    double k2 = sqrt(2 * (E - V0)), s = sin(k2 * a);
    return 1 / (1 + V0 * V0 * s * s / (4 * E * (E - V0)));
}

/* ---- per act: the grid, potential and initial state ---- */
static double *g_psi, *g_V, *g_W;
static void scratch(int n) {
    free(g_psi); free(g_V); free(g_W);
    g_psi = calloc((size_t)n * 2, sizeof(double));
    g_V = calloc((size_t)n, sizeof(double));
    g_W = calloc((size_t)n, sizeof(double));
}
static void geom(int nx, int ny, double dx, double dy, double cx, double cy, double dt) {
    grid_alloc(nx, ny);
    G.dx = dx; G.dy = dy; G.dt = dt;
    G.x0 = cx - nx * dx / 2; G.y0 = cy - ny * dy / 2;
    scratch(nx * ny);
}
#define XI(i) (G.x0 + (i) * G.dx)
#define YJ(j) (G.y0 + (j) * G.dy)
/* a Gaussian packet: |psi|^2 has standard deviations sx, sy */
static void packet(double x0, double y0, double sx, double sy, double kx, double ky, double amp) {
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i) - x0, y = YJ(j) - y0;
            double a = amp * exp(-x * x / (4 * sx * sx) - y * y / (4 * sy * sy)), ph = kx * x + ky * y;
            g_psi[2 * (j * G.nx + i)] += a * cos(ph);
            g_psi[2 * (j * G.nx + i) + 1] += a * sin(ph);
        }
}
/* absorbing layers (a complex potential -iW), bx wide on the left and
 * right edges and by on the top and bottom (0: none), quadratic ramps.
 * A wave crossing one at speed v comes out e^-4.5 the amplitude, so what
 * wraps round the periodic box has crossed two: e^-9. Any stronger and the
 * ramp itself starts to reflect (which shows, in the slits' fringes). */
static void absorber(double bx, double by, double v) {
    double x0 = G.x0, x1 = G.x0 + G.nx * G.dx, y0 = G.y0, y1 = G.y0 + G.ny * G.dy;
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i), y = YJ(j), w = 0;
            double dx_ = fmin(x - x0, x1 - x), dy_ = fmin(y - y0, y1 - y);
            if (bx > 0 && dx_ < bx) { double s = (bx - dx_) / bx; w = fmax(w, 13.5 * v / bx * s * s); }
            if (by > 0 && dy_ < by) { double s = (by - dy_) / by; w = fmax(w, 13.5 * v / by * s * s); }
            g_W[j * G.nx + i] += w;
        }
}

/* the 1D quantum carpet for the revival act */
static Pair carpet;
static GLuint carpet_in, carpet_tex;
#define CAR_N 1024                     /* the doubled well: 512 points inside */
#define CAR_R 1024                     /* rows: t from 0 to T_rev */
static GLuint rv_phi0;                 /* the revival's initial state in momentum space */
static Pair rv;

/* the double slit's screen: |psi|^2 at the screen accumulated, read every
 * frame; clicks sampled from it at fixed step counts */
static GLuint scr_tex, scr_fbo;
#define SNAP 8                         /* steps between screen samples */
#define NCLICK 6000
static float clicks[NCLICK][2];        /* y, arrival time */
static int nclick;
static double scr_acc[1024], scr_total, scr_prev[1024];
static long scr_read_step;

/* the stadium's long exposure */
static GLuint avg_tex, avg_fbo;
static long avg_n;

static void act_title(void) {
    switch (A.id) {
    case A_TUNNEL:
        A.title = "Tunnelling"; A.sub = "a wavepacket meets a square barrier";
        A.eq[0] = "iħ \\f{∂ψ}{∂t} = -\\f{ħ^{2}}{2m}∇^{2}ψ + V(x)ψ";
        A.eq[1] = "T = \\f{1}{1 + V_{0}^{2}\\r{sinh}^{2}(κa) / 4E(V_{0} - E)}";
        break;
    case A_SLIT:
        A.title = "The double slit"; A.sub = "one particle at a time";
        A.eq[0] = "iħ \\f{∂ψ}{∂t} = -\\f{ħ^{2}}{2m}∇^{2}ψ + V(x,y)ψ";
        A.eq[1] = "Δy ≈ \\f{λL}{d},\\q λ = \\f{2πħ}{p}";
        break;
    case A_COHERENT:
        A.title = "A coherent state"; A.sub = A.wy == A.wx ? "the harmonic oscillator's most classical state" : "a Lissajous orbit in an anisotropic well";
        A.eq[0] = "V = \\f{1}{2}m(ω_{x}^{2}x^{2} + ω_{y}^{2}y^{2})";
        A.eq[1] = "\\f{d}{dt}⟨x⟩ = \\f{⟨p⟩}{m},\\q \\f{d}{dt}⟨p⟩ = -⟨∇V⟩";
        break;
    case A_SQUEEZED:
        A.title = "A squeezed state"; A.sub = "breathing at twice the oscillator frequency";
        A.eq[0] = "V = \\f{1}{2}mω^{2}r^{2}";
        A.eq[1] = "σ_{x}^{2}(t) = σ_{0}^{2}\\r{cos}^{2}ωt + \\f{ħ^{2}}{4m^{2}ω^{2}σ_{0}^{2}}\\r{sin}^{2}ωt";
        break;
    case A_REVIVAL:
        A.title = "Quantum revival"; A.sub = "the infinite square well and its carpet";
        A.eq[0] = "E_{n} = \\f{n^{2}π^{2}ħ^{2}}{2mL^{2}}";
        A.eq[1] = "T_{\\r{rev}} = \\f{4mL^{2}}{πħ}";
        break;
    case A_BRAGG:
        A.title = "Bragg scattering"; A.sub = "a wave diffracted by a crystal";
        A.eq[0] = "2d \\r{sin} θ = nλ";
        A.eq[1] = "k' = k + G,\\q |k'| = |k|";
        break;
    case A_STADIUM:
        A.title = A.corral ? "A quantum corral" : "The stadium"; A.sub = A.corral ? "a ring of atoms, like the STM corrals" : "quantum chaos, and the scars of periodic orbits";
        A.eq[0] = "iħ \\f{∂ψ}{∂t} = -\\f{ħ^{2}}{2m}∇^{2}ψ + V(r)ψ";
        A.eq[1] = "\\o{ρ}(r) = \\f{1}{T}∫_{0}^{T}|ψ(r,t)|^{2} dt";
        break;
    }
}

static void setup_tunnel(int v) {
    /* (k0, V0, width in cells): E/V0 below 1, T from ~0.1 to ~0.5 */
    static const double sets[3][3] = {{4.0, 10.0, 4}, {4.0, 9.0, 5}, {4.2, 10.5, 3}};
    const double *s = sets[v % 3];
    /* the barrier's sharp edges want a fine grid along x: 1024 x 64 */
    double dx = 80.0 / 1024;
    geom(1024, 64, dx, 20.0 / 64, 0, 0, 0.005);
    A.k0 = s[0]; A.V0 = s[1]; A.a = s[2] * 2 * dx; A.sigx = 2.0; A.x_start = -18;
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i);
            if (x >= -1e-9 && x < A.a - 1e-9) g_V[j * G.nx + i] = A.V0;        }
    packet(A.x_start, 0, A.sigx, 2.4, A.k0, 0, 1);
    absorber(4, 0, A.k0);
    grid_upload(g_psi, g_V, g_W);
    G.split = A.a * 0.5;
    G.vmax = A.V0 * 1.5;
    /* the packet's own T: T(kx) averaged over its momentum distribution */
    double sk = 1 / (2 * A.sigx), num = 0, den = 0;
    for (int i = -400; i <= 400; i++) {
        double k = A.k0 + i * sk * 0.015, wgt = exp(-(k - A.k0) * (k - A.k0) / (2 * sk * sk));
        if (k <= 0) continue;
        num += wgt * T_square(k * k / 2, A.V0, A.a);
        den += wgt;
    }
    A.T_th = num / den;
    A.sps = 48; A.max_steps = (long)(10.0 / G.dt);
    A.vx0 = -36; A.vx1 = 36; A.vy0 = -10; A.vy1 = 10;
    A.kview = 8; A.expo_k = 1;
}

static void setup_slit(int v) {
    static const double sets[2][2] = {{8.0, 1.25}, {6.5, 1.1}};
    const double *s = sets[v % 2];
    double dx = 80.0 / 512;
    geom(512, 256, dx, 48.0 / 256, 0, 0, 0.01);
    A.k0 = 5.0; A.d = s[0]; A.w = s[1]; A.xwall = -20; A.xs = 26;
    A.lam = 2 * PI / A.k0; A.Lscr = A.xs - A.xwall;
    /* a wall a unit thick, edges softened over half a cell (a step's sharp
     * phase kicks spurious high momenta into the split-step); at V0 = 60
     * the chance of tunnelling through it is ~e^-19 */
    double V0 = 60, e = 0.08, th = 1.0;
#define WIN(u, a, b) (0.25 * (1 + tanh(((u) - (a)) / e)) * (1 + tanh(((b) - (u)) / e)))
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i), y = YJ(j);
            double open = fmax(WIN(y, A.d / 2 - A.w / 2, A.d / 2 + A.w / 2), WIN(y, -A.d / 2 - A.w / 2, -A.d / 2 + A.w / 2));
            g_V[j * G.nx + i] = V0 * WIN(x, A.xwall, A.xwall + th) * (1 - open);
        }
#undef WIN
    packet(-29, 0, 3.5, 6.5, A.k0, 0, 1);
    absorber(5, 6, A.k0);
    grid_upload(g_psi, g_V, g_W);
    G.vmax = V0;
    A.sps = 40; A.max_steps = (long)(16.0 / G.dt);
    A.vx0 = -35; A.vx1 = 35; A.vy0 = -17.5; A.vy1 = 17.5;
    A.kview = 4.5; A.kcx = 4; A.expo_k = 1;
    /* the screen: one column of |psi|^2 per snapshot */
    if (!scr_tex) {
        scr_tex = mk_tex(256, 1, GL_R32F, GL_RED, NULL, 0, 0);
        scr_fbo = mk_fbo(scr_tex, 0);
    }
    static const float z[4] = {0};
    glBindFramebuffer(GL_FRAMEBUFFER, scr_fbo);
    glClearBufferfv(GL_COLOR, 0, z);
    memset(scr_acc, 0, sizeof scr_acc); memset(scr_prev, 0, sizeof scr_prev);
    scr_total = 0; nclick = 0; scr_read_step = 0;
}

static void setup_osc(int squeezed, int v) {
    double L = 24, dx = L / 256;
    geom(256, 256, dx, dx, 0, 0, 0.01);
    A.wx = 1; A.wy = 1;
    if (!squeezed && v % 2) A.wy = 1.5;
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i), y = YJ(j);
            g_V[j * G.nx + i] = 0.5 * (A.wx * A.wx * x * x + A.wy * A.wy * y * y);
        }
    if (squeezed) {
        A.s0 = 0.42;                    /* the width parameter of psi ~ exp(-x^2 / 2 s^2) */
        A.x0 = 5.5; A.y0 = 0; A.px0 = 0; A.py0 = 0;
        packet(A.x0, A.y0, A.s0 / sqrt(2), 1 / sqrt(2 * A.wy), A.px0, A.py0, 1);
    } else {
        A.x0 = 6.5; A.y0 = 0; A.px0 = 0; A.py0 = A.wy == A.wx ? 6.5 : 5.0;
        packet(A.x0, A.y0, 1 / sqrt(2 * A.wx), 1 / sqrt(2 * A.wy), A.px0, A.py0, 1);
    }
    grid_upload(g_psi, g_V, NULL);
    G.vmax = 60;
    A.sps = 90; A.max_steps = 1L << 40;
    A.vx0 = -10.5; A.vx1 = 10.5; A.vy0 = -10.5; A.vy1 = 10.5;
    A.kview = 10; A.expo_k = 1;
}

/* the infinite well [0, L]^2 as the odd extension onto [-L, L)^2: free
 * evolution there is exact on the grid, phi(k, t) = phi(k, 0) exp(-i k^2 t / 2),
 * with k = pi m / L, so the revival at T = 4 L^2 / pi is exact too */
static void setup_revival(void) {
    int n = 256;
    A.Lbox = 12;
    double dx = 2 * A.Lbox / n;
    geom(n, n, dx, dx, 0, 0, 0);
    G.x0 = -A.Lbox; G.y0 = -A.Lbox;
    A.trev = 4 * A.Lbox * A.Lbox / PI;
    double xc = 0.5 * A.Lbox, yc = 0.3 * A.Lbox, sg = A.Lbox / 22, kx = 0, ky = 22 / A.Lbox;
    A.x0 = xc; A.y0 = yc; A.px0 = kx; A.py0 = ky; A.s0 = sg;
    /* odd in x and y: four images */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double x = XI(i), y = YJ(j), re = 0, im = 0;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2) {
                    double X = sx * x, Y = sy * y, ax = X - xc, ay = Y - yc;
                    double a = sx * sy * exp(-(ax * ax + ay * ay) / (4 * sg * sg)), ph = kx * X + ky * Y;
                    re += a * cos(ph); im += a * sin(ph);
                }
            g_psi[2 * (j * n + i)] = re; g_psi[2 * (j * n + i) + 1] = im;
        }
    grid_upload(g_psi, NULL, NULL);
    G.win_x = n / 2 + 1; G.win_y = n / 2 + 1;
    /* phi0 = F psi0 */
    pair_init(&rv, n, n, GL_RG32F, GL_RG, 1, 1);
    if (!rv_phi0) rv_phi0 = mk_tex(n, n, GL_RG32F, GL_RG, NULL, 0, 1);
    fft_begin(n, n);
    pass_k = 0;
    GLuint t = fft_axis(G.psi.t[0], &rv, 0, n, 1, 0, 0);
    t = fft_axis(t, &rv, 1, n, 1, 0, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, rv.f[t == rv.t[0] ? 0 : 1]);
    glBindTexture(GL_TEXTURE_2D, rv_phi0);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, n, n);
    G.vmax = 1;
    A.sps = 0; A.max_steps = 0;
    A.vx0 = 0; A.vx1 = A.Lbox; A.vy0 = 0; A.vy1 = A.Lbox;
    A.kview = 6.5; A.expo_k = 1;

    /* the carpet: the 1D well's psi(x, t) for t from 0 to T_rev, one row
     * each, from the exact phases (in double) and one batched inverse FFT */
    pair_init(&carpet, CAR_N, CAR_R, GL_RG32F, GL_RG, 1, 0);
    if (!carpet_in) carpet_in = mk_tex(CAR_N, CAR_R, GL_RG32F, GL_RG, NULL, 0, 0);
    double L = A.Lbox, cdx = 2 * L / CAR_N;
    double *f = calloc(2 * CAR_N, sizeof(double));
    for (int i = 0; i < CAR_N; i++) {
        double x = -L + i * cdx;
        for (int sx = -1; sx <= 1; sx += 2) {
            double X = sx * x, ax = X - xc, a = sx * exp(-ax * ax / (4 * sg * sg)), ph = kx * X;
            f[2 * i] += a * cos(ph); f[2 * i + 1] += a * sin(ph);
        }
    }
    /* its DFT, in double */
    double *F = calloc(2 * CAR_N, sizeof(double));
    for (int m = 0; m < CAR_N; m++) {
        double re = 0, im = 0;
        for (int i = 0; i < CAR_N; i++) {
            double a = -2 * PI * (double)m * i / CAR_N, c = cos(a), s = sin(a);
            re += f[2 * i] * c - f[2 * i + 1] * s; im += f[2 * i] * s + f[2 * i + 1] * c;
        }
        F[2 * m] = re; F[2 * m + 1] = im;
    }
    double nrm = 0;
    for (int i = 0; i < CAR_N; i++) nrm += f[2 * i] * f[2 * i] + f[2 * i + 1] * f[2 * i + 1];
    nrm = 1 / sqrt(nrm * cdx / 2) / CAR_N;                /* normalised over the well [0, L] */
    float *buf = malloc(sizeof(float) * 2 * CAR_N * CAR_R);
    for (int r = 0; r < CAR_R; r++) {
        double tau = (double)r / (CAR_R - 1);
        for (int m = 0; m < CAR_N; m++) {
            int mm = m < CAR_N / 2 ? m : m - CAR_N;
            /* exp(-i k^2 t / 2) = exp(-2 pi i m^2 tau) with k = pi m / L, t = tau T_rev */
            double ph = -2 * PI * fmod((double)mm * mm * tau, 1.0), c = cos(ph), s = sin(ph);
            buf[2 * (r * CAR_N + m)] = (float)((F[2 * m] * c - F[2 * m + 1] * s) * nrm);
            buf[2 * (r * CAR_N + m) + 1] = (float)((F[2 * m] * s + F[2 * m + 1] * c) * nrm);
        }
    }
    glBindTexture(GL_TEXTURE_2D, carpet_in);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, CAR_N, CAR_R, GL_RG, GL_FLOAT, buf);
    free(buf); free(F); free(f);
    fft_begin(CAR_N, CAR_R);
    pass_k = 0;
    carpet_tex = fft_axis(carpet_in, &carpet, 0, CAR_N, -1, 0, 0);
}

/* the revival's state at time t: phi0 phased exactly, transformed back */
static void revival_at(double t) {
    int n = G.nx;
    uint32_t s = (uint32_t)llround(fmod(t / A.trev, 1.0) * 1048576.0) & 1048575u;
    glUseProgram(prog_phase);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
    glViewport(0, 0, n, n);
    glUniform2i(U(prog_phase, "uN"), n, n);
    glUniform1ui(U(prog_phase, "uS"), s);
    glUniform1f(U(prog_phase, "uScale"), 1.0f / ((float)n * n));
    bind(0, rv_phi0);
    glBindFramebuffer(GL_FRAMEBUFFER, G.meas.f[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    /* the momentum picture is |phi0|^2 (and stays so); psi from the inverse */
    fft_begin(n, n);
    pass_k = 0;
    GLuint tt = fft_axis(G.meas.t[0], &G.psi, 0, n, -1, 0, 0);
    tt = fft_axis(tt, &G.psi, 1, n, -1, 0, 0);
    if (tt != G.psi.t[0]) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, G.psi.f[1]);
        glBindTexture(GL_TEXTURE_2D, G.psi.t[0]);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, n, n);
    }
    G.kshow = rv_phi0;
}

static void setup_bragg(int v) {
    double L = 48, dx = L / 256;
    geom(256, 256, dx, dx, 0, 0, 0.02);
    A.latt = 2.0;
    double lam = v % 2 ? 1.25 : 1.4;
    A.k0 = 2 * PI / lam;
    /* rotate the lattice so the incoming beam (along +x) meets the (1,0)
     * planes at the Bragg angle: k . G^ = -|G|/2 */
    double G0 = 2 * PI / A.latt, c = G0 / (2 * A.k0);
    A.lrot = acos(c);                  /* the angle between k and -G */
    double gx = -cos(A.lrot), gy = sin(A.lrot);   /* G^ */
    double ux = -gy, uy = gx;          /* along the planes */
    for (int j = 0; j < G.ny; j++)
        for (int i = 0; i < G.nx; i++) {
            double x = XI(i), y = YJ(j), V = 0;
            /* lattice coordinates */
            double a1 = (x * gx + y * gy) / A.latt, a2 = (x * ux + y * uy) / A.latt;
            if (fabs(a1) > 5.6 || fabs(a2) > 5.6) { g_V[j * G.nx + i] = 0; continue; }
            for (int p = (int)floor(a1) - 1; p <= (int)floor(a1) + 2; p++)
                for (int q = (int)floor(a2) - 1; q <= (int)floor(a2) + 2; q++) {
                    if (abs(p) > 5 || abs(q) > 5) continue;
                    double cx = (p * gx + q * ux) * A.latt, cy = (p * gy + q * uy) * A.latt;
                    double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    V += 5.0 * exp(-r2 / (2 * 0.28 * 0.28));
                }
            g_V[j * G.nx + i] = V;
        }
    packet(-15, 0, 3.0, 4.5, A.k0, 0, 1);
    absorber(4, 4, A.k0);
    grid_upload(g_psi, g_V, g_W);
    G.vmax = 6;
    A.sps = 22; A.max_steps = (long)(13.5 / G.dt);
    A.vx0 = -20.5; A.vx1 = 20.5; A.vy0 = -20.5; A.vy1 = 20.5;
    A.kview = 7; A.expo_k = 1;
}

static void setup_stadium(int v) {
    A.corral = v % 2;
    if (A.corral) {
        double L = 28, dx = L / 256;
        geom(256, 256, dx, dx, 0, 0, 0.01);
        A.R = 9;
        int na = 44;
        for (int j = 0; j < G.ny; j++)
            for (int i = 0; i < G.nx; i++) {
                double x = XI(i), y = YJ(j), V = 0;
                for (int k = 0; k < na; k++) {
                    double a = 2 * PI * k / na, cx = A.R * cos(a), cy = A.R * sin(a);
                    double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    if (r2 < 4) V += 40 * exp(-r2 / (2 * 0.32 * 0.32));
                }
                g_V[j * G.nx + i] = V;
            }
        A.k0 = 5.0;
        packet(-3, -2, 1.4, 1.4, A.k0 * 0.8, A.k0 * 0.6, 1);
        absorber(3.5, 3.5, A.k0);
        grid_upload(g_psi, g_V, g_W);
        G.vmax = 30;
        A.sps = 70; A.max_steps = 1L << 40;
        A.vx0 = -12; A.vx1 = 12; A.vy0 = -12; A.vy1 = 12;
        A.kview = 9;
    } else {
        double dx = 40.0 / 512;
        geom(512, 256, dx, dx, 0, 0, 0.01);
        A.R = 8.5; A.alen = 8.5;        /* semicircles of radius R, straight sides 2 alen */
        A.k0 = 6.0;
        double V0 = 3 * A.k0 * A.k0 / 2;      /* walls a cell or two soft, V dt < 1: a sharp one at a big V dt pumps energy into high k */
        for (int j = 0; j < G.ny; j++)
            for (int i = 0; i < G.nx; i++) {
                double x = XI(i), y = YJ(j), d;
                double ax = fabs(x) - A.alen;
                if (ax <= 0) d = fabs(y) - A.R;
                else d = sqrt(ax * ax + y * y) - A.R;
                g_V[j * G.nx + i] = V0 * 0.5 * (1 + tanh(d / 0.12));
            }
        /* launched along the unstable bouncing orbit through the long axis,
         * a little off it */
        packet(-2, 0.6, 1.3, 1.3, A.k0, 0.4, 1);
        grid_upload(g_psi, g_V, NULL);
        G.vmax = V0;
        A.sps = 60; A.max_steps = 1L << 40;
        A.vx0 = -18.4; A.vx1 = 18.4; A.vy0 = -9.4; A.vy1 = 9.4;
        A.kview = 10;
    }
    A.expo_k = 1;
    if (!avg_tex || G.nx != 0) {
        if (avg_tex) { glDeleteTextures(1, &avg_tex); glDeleteFramebuffers(1, &avg_fbo); }
        avg_tex = mk_tex(G.nx, G.ny, GL_R32F, GL_RED, NULL, 1, 1);
        avg_fbo = mk_fbo(avg_tex, 0);
    }
    static const float z[4] = {0};
    glBindFramebuffer(GL_FRAMEBUFFER, avg_fbo);
    glClearBufferfv(GL_COLOR, 0, z);
    avg_n = 0;
}

static void act_setup(int id, int variant, int num) {
    memset(&A, 0, sizeof A);
    A.id = id; A.variant = variant; A.num = num;
    switch (id) {
    case A_TUNNEL: setup_tunnel(variant); break;
    case A_SLIT: setup_slit(variant); break;
    case A_COHERENT: setup_osc(0, variant); break;
    case A_SQUEEZED: setup_osc(1, variant); break;
    case A_REVIVAL: setup_revival(); break;
    case A_BRAGG: setup_bragg(variant); break;
    case A_STADIUM: setup_stadium(variant); break;
    }
    act_title();
    rd_drop();
    have_marg = 0;
    steps = 0; sim_t = 0; hn = 0;
    memset(&M, 0, sizeof M);
}

/* new arrivals at the screen since the last look: particles, one at a
 * time, each where the wave says with the odds |psi|^2 (the flux there
 * since the last sample), as many as the probability that came through */
#define CLICKS_PER_P 26000.0
static void screen_clicks(void) {
    static float row[256];
    glBindFramebuffer(GL_FRAMEBUFFER, scr_fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, G.ny, 1, GL_RED, GL_FLOAT, row);
    double inc[256], sum = 0;
    for (int j = 0; j < G.ny; j++) {
        inc[j] = fmax(row[j] - scr_prev[j], 0);
        scr_prev[j] = row[j];
        scr_acc[j] = row[j];
        sum += inc[j];
    }
    /* probability through the screen: |psi|^2 v dt dy */
    double before = scr_total;
    scr_total += sum * A.k0 * G.dy;
    long want = (long)floor(scr_total * CLICKS_PER_P), have = (long)floor(before * CLICKS_PER_P);
    for (long c = have; c < want && nclick < NCLICK; c++) {
        rng = mix32((uint32_t)c * 0x9E3779B9U ^ opt_seed ^ 0x51u);
        double u = drand() * sum, acc = 0;
        int j = 0;
        for (; j < G.ny - 1 && acc + inc[j] < u; j++) acc += inc[j];
        double f = inc[j] > 0 ? (u - acc) / inc[j] : 0.5;
        clicks[nclick][0] = (float)(YJ(j) + (f - 0.5) * G.dy);
        clicks[nclick][1] = (float)(steps * G.dt + drand() * SNAP * G.dt);
        nclick++;
    }
}

/* the hooks after each step */
static void after_step(void) {
    steps++;
    if (A.id == A_SLIT && steps % SNAP == 0) {
        /* the screen column's |psi|^2 dt, summed, as a row of scr_tex */
        int is = (int)lround((A.xs - G.x0) / G.dx);
        accumulate(G.psi.t[0], scr_fbo, is, 0, 0, 0, G.ny, 1, (float)(G.dt * SNAP), 1);
        screen_clicks();
    }
    if (A.id == A_STADIUM && steps % 2 == 0) {
        accumulate(G.psi.t[0], avg_fbo, 0, 0, 0, 0, G.nx, G.ny, 1.0f, 0);
        avg_n++;
    }
}

/* ==================================================================== */
/* ---- the act clock ---- */
static double cycle_len(void) {
    double s = 0;
    for (int i = 0; i < NACT; i++) s += act_len[i];
    return s;
}
/* the order of the acts in cycle c: shuffled from the seed, never the same
 * act twice running across a cycle boundary */
static void cycle_order(int c, int *ord) {
    int prev_last = -1;
    if (c > 0) {
        int p[NACT];
        cycle_order(c - 1 > 0 ? -(c - 1) - 1 : -1, p);   /* the previous cycle, without its own fix-up */
        prev_last = p[NACT - 1];
    }
    int cc = c < 0 ? -c - 1 : c;
    for (int i = 0; i < NACT; i++) ord[i] = i;
    rng = mix32(opt_seed * 2654435761U ^ (uint32_t)cc * 0x85EBCA6BU ^ 0x9u);
    for (int i = NACT - 1; i > 0; i--) {
        int j = (int)(drand() * (i + 1)) % (i + 1), t = ord[i];
        ord[i] = ord[j]; ord[j] = t;
    }
    if (c > 0 && ord[0] == prev_last) { int t = ord[0]; ord[0] = ord[1]; ord[1] = t; }
}

static void act_at(double t, int *id, int *num, int *variant, int *cyc, double *lt) {
    if (t < 0) t = 0;
    if (opt_act >= 0) {
        double L = act_len[opt_act];
        int c = (int)floor(t / L);
        *id = opt_act; *num = 1; *variant = c + (int)(opt_seed % 7); *cyc = c; *lt = t - c * L;
        return;
    }
    double cl = cycle_len();
    int c = (int)floor(t / cl);
    double r = t - c * cl;
    int ord[NACT];
    cycle_order(c, ord);
    for (int i = 0; i < NACT; i++) {
        double L = act_len[ord[i]];
        if (r < L || i == NACT - 1) {
            *id = ord[i]; *num = i + 1; *variant = c + (int)(opt_seed % 5); *cyc = c; *lt = r;
            return;
        }
        r -= L;
    }
}

static int offline;
static double expo_r;                  /* the smoothed peak of |psi|^2, for the exposure */
static double E0;                      /* the energy at the first measurement */
static int e0_set;

/* the revival's clock: tau = t / T_rev runs 0 -> 1 over the act, easing
 * through the fractional revivals so they can be seen */
static double revival_tau(double lt) {
    double run = act_len[A_REVIVAL] - INTRO - FADE - 3.0;
    double u = clampd((lt - INTRO) / run, 0, 1);
    return u * 1.0;
}

static void advance(double t) {
    int id, num, var, cyc;
    double lt;
    act_at(t, &id, &num, &var, &cyc, &lt);
    if (id != act_cur || cyc != act_cycle) {
        act_cur = id; act_cycle = cyc;
        act_setup(id, var, num);
        expo_r = 0; e0_set = 0;
    }
    act_t = lt;
    if (A.id == A_REVIVAL) {
        double tau = revival_tau(lt);
        revival_at(tau * A.trev);
        sim_t = tau * A.trev;
        return;
    }
    long want = (long)floor((lt - INTRO) * A.sps);
    if (want < 0) want = 0;
    if (want > A.max_steps) want = A.max_steps;
    /* live, a late start catches up a few steps a frame (in fast motion) */
    int budget = offline ? 1 << 30 : (int)ceil(A.sps / 60.0) + 6;
    int did = 0;
    while (steps < want && budget-- > 0) { grid_step(); after_step(); did = 1; }
    if (did) G.kstale = 1;
    sim_t = steps * G.dt;
}

/* the frame's numbers, and the exposure */
static void frame_measure(void) {
    measure(1);
    if (A.id == A_REVIVAL) {
        /* the quadrant holds a quarter of the doubled state's norm */
        M.n *= 4;
    }
    /* the exposure follows the peak of |psi|^2 (so what came through the
     * slits shows once the reflection has gone), but never brightens more
     * than 40x on the start's, or it would dig up the absorbers' leavings */


    double target = A.id == A_REVIVAL ? M.rmax : fmax(M.rmax, G.rmax0 / 40);
    if (expo_r <= 0) expo_r = target;
    else expo_r += (target - expo_r) * (offline ? 0.15 : 0.06);
    double E = M.T + M.V;
    if (!e0_set && M.have_k) { E0 = E; e0_set = 1; }
    float ha = 0, hb = 0;
    if (A.id == A_TUNNEL) { ha = (float)(M.n - M.right * M.n); hb = (float)(M.right * M.n); }
    else { ha = (float)sqrt(fmax(M.xx - M.x * M.x, 0)); hb = (float)sqrt(fmax(M.kxx - M.kx * M.kx, 0)); }
    hist_add((float)sim_t, (float)M.x, (float)M.y, (float)M.kx, ha, hb);
    if (A.id == A_TUNNEL) marginal();
}

/* ==================================================================== */
/* ---- drawing ---- */
typedef struct { float x0, y0, x1, y1; } Rect;
static float rw(Rect r) { return r.x1 - r.x0; }
static float rh(Rect r) { return r.y1 - r.y0; }
static Rect fit(Rect a, float aspect, float vy) {    /* vy: 0 top, 0.5 centre, 1 bottom */
    float w = rw(a), h = rh(a);
    if (w / h > aspect) w = h * aspect; else h = w / aspect;
    Rect r;
    r.x0 = a.x0 + (rw(a) - w) * 0.5f; r.x1 = r.x0 + w;
    r.y0 = a.y0 + (rh(a) - h) * vy; r.y1 = r.y0 + h;
    return r;
}

static float u_;                       /* one unit: a thousandth of the output's height (or width, portrait) */
static float fade_a;                   /* the act's fade in and out */

static const char *FS_BG = GLSL(
    uniform vec3 uBg;
    uniform vec2 uRes;
    uniform int uMode;
    out vec4 o;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
        vec2 q = gl_FragCoord.xy / uRes - 0.5;
        q.x *= uRes.x / uRes.y;
        float v = dot(q, q);
        vec3 c = uBg;
        if (uMode == 2) c *= 1.0 - 0.06 * v;
        else c = c * (1.08 - 0.5 * v) + vec3(0.004, 0.006, 0.012) * (1.0 - v);
        c += (hash(gl_FragCoord.xy) - 0.5) * (uMode == 2 ? 0.012 : 0.008);
        o = vec4(c, 1.0);
    });

/* the picture of a field in a panel: kind 0 psi (domain colouring, the
 * potential in relief, a glow), 1 momentum density, 2 the carpet, 3 a
 * density (the long exposure) */
static const char *FS_PANEL = GLSL(
    uniform sampler2D uPsi, uV, uGlow;
    uniform int uKind, uMode;
    uniform vec4 uRect;      /* GL window px: x0 y0 x1 y1 */
    uniform vec4 uView;      /* world shown: x0 y0 x1 y1 */
    uniform vec4 uGrid;      /* world -> uv: u = (x - gx0) / gLx ... */
    uniform vec2 uN;         /* texels of uPsi */
    uniform float uExpo, uVscale, uFade, uReveal, uGlowAmt, uE, uK0, uKW;
    uniform int uRelMode;
    uniform vec2 uKR;        /* the polar plot: the k-range of uPsi */
    uniform vec3 uBg, uRel, uInk;
    out vec4 o;
    vec3 hue(float t) {
        return 0.52 + 0.42 * cos(6.2831853 * (t + vec3(0.0, 0.6667, 0.3333)));
    }
    vec3 ember(float e) {
        e = clamp(e, 0.0, 1.0);
        vec3 a = vec3(0.30, 0.04, 0.02), b = vec3(0.65, 0.20, 0.10), c = vec3(0.93, 0.58, 0.25), d = vec3(1.0, 0.92, 0.72);
        vec3 r = mix(vec3(0.0), a, smoothstep(0.0, 0.25, e));
        r = mix(r, b, smoothstep(0.2, 0.5, e));
        r = mix(r, c, smoothstep(0.45, 0.8, e));
        return mix(r, d, smoothstep(0.75, 1.0, e));
    }
    vec2 catrom(sampler2D t, vec2 uv) {
        vec2 sp = uv * uN, tp = floor(sp - 0.5) + 0.5, f = sp - tp;
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
        vec2 w12 = w1 + w2, o12 = w2 / w12;
        vec2 t0 = (tp - 1.0) / uN, t3 = (tp + 2.0) / uN, t12 = (tp + o12) / uN;
        vec2 r = vec2(0.0);
        r += texture(t, vec2(t0.x, t0.y)).rg * w0.x * w0.y;
        r += texture(t, vec2(t12.x, t0.y)).rg * w12.x * w0.y;
        r += texture(t, vec2(t3.x, t0.y)).rg * w3.x * w0.y;
        r += texture(t, vec2(t0.x, t12.y)).rg * w0.x * w12.y;
        r += texture(t, vec2(t12.x, t12.y)).rg * w12.x * w12.y;
        r += texture(t, vec2(t3.x, t12.y)).rg * w3.x * w12.y;
        r += texture(t, vec2(t0.x, t3.y)).rg * w0.x * w3.y;
        r += texture(t, vec2(t12.x, t3.y)).rg * w12.x * w3.y;
        r += texture(t, vec2(t3.x, t3.y)).rg * w3.x * w3.y;
        return r;
    }
    /* |z|^2 interpolated (not z: a momentum picture's phase winds fast) */
    float catrom_rho(sampler2D t, vec2 uv) {
        vec2 sp = uv * uN - 0.5, tp = floor(sp), f = sp - tp;
        vec4 wx = vec4(f.x * (-0.5 + f.x * (1.0 - 0.5 * f.x)), 1.0 + f.x * f.x * (-2.5 + 1.5 * f.x),
                       f.x * (0.5 + f.x * (2.0 - 1.5 * f.x)), f.x * f.x * (-0.5 + 0.5 * f.x));
        vec4 wy = vec4(f.y * (-0.5 + f.y * (1.0 - 0.5 * f.y)), 1.0 + f.y * f.y * (-2.5 + 1.5 * f.y),
                       f.y * (0.5 + f.y * (2.0 - 1.5 * f.y)), f.y * f.y * (-0.5 + 0.5 * f.y));
        ivec2 n = ivec2(uN), b = ivec2(tp) - 1;
        float r = 0.0;
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++) {
                ivec2 q = ivec2((b.x + i + n.x * 4) % n.x, (b.y + j + n.y * 4) % n.y);
                vec2 z = texelFetch(t, q, 0).rg;
                r += dot(z, z) * wx[i] * wy[j];
            }
        return max(r, 0.0);
    }
    /* domain colouring: hue the phase, brightness |psi|^2 */
    vec3 shade(vec2 z, float glow) {
        float r = dot(z, z), b = 1.0 - exp(-r * uExpo), t = atan(z.y, z.x) / 6.2831853;
        float g = 1.0 - exp(-glow * uExpo * 0.9);
        if (uMode == 0) {
            vec3 h = hue(t);
            vec3 c = uBg + h * b * 1.02 + vec3(1.0) * pow(b, 6.0) * 0.30;
            return c + mix(h, vec3(0.6, 0.7, 1.0), 0.5) * g * uGlowAmt * 0.45;
        }
        if (uMode == 1) return uBg + vec3(b * 0.97) + vec3(g * uGlowAmt * 0.32);
        if (uMode == 2) {
            vec3 h = hue(t) * 0.82 + 0.06;
            vec3 c = uBg * mix(vec3(1.0), h * h * 0.95, b);
            return c * (1.0 - g * uGlowAmt * 0.10);
        }
        float e = b * (0.80 + 0.20 * cos(6.2831853 * t));
        return uBg + ember(e) + ember(g * 0.55) * uGlowAmt * 0.45;
    }
    /* a density alone, in the palette */
    vec3 density(float r) {
        float b = 1.0 - exp(-r * uExpo);
        if (uMode == 0) return uBg + mix(vec3(0.20, 0.42, 0.95), vec3(0.95, 0.97, 1.0), smoothstep(0.35, 1.0, b)) * b;
        if (uMode == 1) return uBg + vec3(b * 0.97);
        if (uMode == 2) return uBg * mix(vec3(1.0), vec3(0.16, 0.20, 0.36), b);
        return uBg + ember(b);
    }
    void main() {
        vec2 f = (gl_FragCoord.xy - uRect.xy) / (uRect.zw - uRect.xy);
        vec2 w = mix(uView.xy, uView.zw, f);
        vec2 uv = (w - uGrid.xy) / uGrid.zw + 0.5 / uN;
        vec3 c;
        if (uKind == 0) {
            vec2 z = catrom(uPsi, uv);
            float glow = texture(uGlow, uv).r;
            c = shade(z, glow);
            /* the potential: a translucent relief, lit from the upper left */
            vec2 e = 1.0 / uN;
            float Vr = texture(uV, uv).r;
            if (Vr > 0.002 * uVscale) {
                float h = clamp(Vr / uVscale, 0.0, 1.0);
                float hx = clamp(texture(uV, uv + vec2(e.x, 0.0)).r / uVscale, 0.0, 1.0) - clamp(texture(uV, uv - vec2(e.x, 0.0)).r / uVscale, 0.0, 1.0);
                float hy = clamp(texture(uV, uv + vec2(0.0, e.y)).r / uVscale, 0.0, 1.0) - clamp(texture(uV, uv - vec2(0.0, e.y)).r / uVscale, 0.0, 1.0);
                vec3 n = normalize(vec3(-hx * 3.0, -hy * 3.0, 1.0));
                float dif = clamp(dot(n, normalize(vec3(-0.5, 0.6, 0.65))), 0.0, 1.0);
                vec3 rc = uRel * (0.45 + 0.75 * dif);
                /* walls and wells: faint where the particle may go, a solid
                 * relief where it classically can't (V > E), and the
                 * turning line V = E between; bumps (atoms): by height */
                float hE = Vr / max(uE, 1e-6), a, rim;
                if (uRelMode == 1) {
                    a = smoothstep(0.03, 0.6, h) * 0.42;
                    rim = 0.0;
                } else if (uRelMode == 2) {
                    /* a smooth well: equipotentials at quarters of E, the
                     * turning line bolder, a light veil beyond it */
                    a = 0.13 * smoothstep(1.0, 3.0, hE);
                    float q = hE * 4.0, fq = abs(q - floor(q + 0.5)) / max(fwidth(q), 1e-5);
                    rim = (1.0 - smoothstep(0.0, 1.2, fq)) * (hE < 1.1 && hE > 0.2 ? 0.10 : 0.0);
                    rim = max(rim, (1.0 - smoothstep(0.0, 1.4, abs(hE - 1.0) / max(fwidth(hE), 1e-5))) * 0.40);
                } else {
                    a = 0.10 * smoothstep(0.02, 0.6, hE) + 0.30 * smoothstep(1.0, 2.6, hE);
                    rim = (1.0 - smoothstep(0.0, 1.4, abs(hE - 1.0) / max(fwidth(hE), 1e-5))) * 0.45;
                }
                c = mix(c, rc, a);
                c = mix(c, uRel * 1.15, rim);
            }
        } else if (uKind == 1) {
            c = density(catrom_rho(uPsi, uv));
        } else if (uKind == 2) {
            /* the carpet: x across, t down; rows past now not yet woven */
            c = density(catrom_rho(uPsi, uv));
            float row = 1.0 - f.y;
            c = mix(uBg, c, smoothstep(uReveal + 0.0015, uReveal - 0.0015, row));
        } else if (uKind == 4) {
            /* a polar plot of |phi|^2 about the ring |k| = k0, by direction */
            float r = length(w), th = atan(w.y, w.x);
            vec2 d = vec2(cos(th), sin(th));
            float I = 0.0;
            for (int i = 0; i < 9; i++) {
                vec2 k = d * (uK0 + (float(i) / 8.0 - 0.5) * uKW);
                I += catrom_rho(uPsi, k / uKR + 0.5 / uN);
            }
            float v = 1.0 - exp(-I * uExpo);
            float R0 = 0.42, R = R0 + (0.98 - R0) * sqrt(v);
            float px = fwidth(r);
            float inside = smoothstep(R + px, R - px, r) * smoothstep(R0 - px, R0 + px, r);
            float edge = (1.0 - smoothstep(0.0, 1.6 * px, abs(r - R))) * step(R0, r);
            c = uBg;
            c = mix(c, density(0.6 / max(uExpo, 1e-12) * (0.3 + v)), inside * 0.85);
            c = mix(c, uInk, edge * 0.85);
        } else {
            float r = texture(uPsi, uv).r;
            c = density(r);
        }
        o = vec4(mix(uBg, c, uFade), 1.0);
    });

static const char *VS_OV = "#version 330 core\n" GLSL(
    layout(location = 0) in vec4 aP;     /* x, y, u, v */
    layout(location = 1) in vec4 aC;
    layout(location = 2) in vec2 aK;     /* kind, param */
    uniform vec2 uRes;
    out vec2 vUV;
    out vec4 vC;
    flat out vec2 vK;
    void main() {
        vUV = aP.zw; vC = aC; vK = aK;
        gl_Position = vec4(aP.x / uRes.x * 2.0 - 1.0, 1.0 - aP.y / uRes.y * 2.0, 0.0, 1.0);
    });
static const char *FS_OV = "#version 330 core\n" GLSL(
    uniform sampler2D uAtlas;
    in vec2 vUV;
    in vec4 vC;
    flat in vec2 vK;
    out vec4 o;
    void main() {
        float a = 1.0;
        if (vK.x < 0.5) a = texelFetch(uAtlas, ivec2(vUV), 0).r;
        else if (vK.x < 1.5) a = clamp(vK.y + 0.5 - abs(vUV.x), 0.0, 1.0);
        else if (vK.x < 2.5) a = 1.0;
        else a = clamp(vK.y + 0.5 - length(vUV), 0.0, 1.0);
        o = vec4(vC.rgb, vC.a * a);
    });

static GLuint prog_bg, prog_panel, prog_ov, vao_ov, vbo_ov;

static void ov_flush(GLuint target) {
    if (!ovn) return;
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(prog_ov);
    glUniform2f(U(prog_ov, "uRes"), (float)W, (float)H);
    bind(0, atlas_tex);
    glBindVertexArray(vao_ov);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ov);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(ovn * sizeof(float)), ov, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, ovn / OVF);
    glDisable(GL_BLEND);
    ovn = 0;
}

/* a picture panel; r in px from the top left; view the world rect shown */
static GLuint draw_target;
static float pan_E = 1, pan_k0, pan_kw, pan_kr[2];
static int pan_relmode;
static void panel(Rect r, int kind, GLuint tex, int tw, int th, const double grid[4], const double view[4],
                  float expo, GLuint vtex, float vscale, GLuint glow, float glow_amt, float reveal) {
    glBindFramebuffer(GL_FRAMEBUFFER, draw_target);
    glEnable(GL_SCISSOR_TEST);
    int x0 = (int)floorf(r.x0), x1 = (int)ceilf(r.x1), y0 = H - (int)ceilf(r.y1), y1 = H - (int)floorf(r.y0);
    glScissor(x0, y0, x1 - x0, y1 - y0);
    glViewport(0, 0, W, H);
    glUseProgram(prog_panel);
    glUniform4f(U(prog_panel, "uRect"), r.x0, H - r.y1, r.x1, H - r.y0);
    glUniform4f(U(prog_panel, "uView"), (float)view[0], (float)view[1], (float)view[2], (float)view[3]);
    glUniform4f(U(prog_panel, "uGrid"), (float)grid[0], (float)grid[1], (float)grid[2], (float)grid[3]);
    glUniform2f(U(prog_panel, "uN"), (float)tw, (float)th);
    glUniform1i(U(prog_panel, "uKind"), kind);
    glUniform1i(U(prog_panel, "uMode"), P->mode);
    glUniform1f(U(prog_panel, "uExpo"), expo);
    glUniform1f(U(prog_panel, "uVscale"), vscale);
    glUniform1f(U(prog_panel, "uFade"), fade_a);
    glUniform1f(U(prog_panel, "uReveal"), reveal);
    glUniform1f(U(prog_panel, "uGlowAmt"), glow_amt);
    glUniform3fv(U(prog_panel, "uBg"), 1, c_bg);
    glUniform3fv(U(prog_panel, "uRel"), 1, c_rel);
    glUniform3fv(U(prog_panel, "uInk"), 1, c_ink);
    glUniform1f(U(prog_panel, "uE"), pan_E);
    glUniform1i(U(prog_panel, "uRelMode"), pan_relmode);
    glUniform1f(U(prog_panel, "uK0"), pan_k0);
    glUniform1f(U(prog_panel, "uKW"), pan_kw);
    glUniform2f(U(prog_panel, "uKR"), pan_kr[0], pan_kr[1]);
    bind(0, tex);
    bind(1, vtex ? vtex : tex);
    bind(2, glow ? glow : tex);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
}

/* colours with alpha, faded with the act */
static float *col(const float *c, float a) {
    static float ring_[32][4];
    static int k;
    float *o = ring_[k++ & 31];
    rgba(o, c, a * fade_a);
    return o;
}
static float *colk(const float *c, float a) {    /* not faded (the clock) */
    static float ring_[16][4];
    static int k;
    float *o = ring_[k++ & 15];
    rgba(o, c, a);
    return o;
}

/* world <-> panel px */
typedef struct { Rect r; double v[4]; } Map;
static float mx_(const Map *m, double x) { return m->r.x0 + (float)((x - m->v[0]) / (m->v[2] - m->v[0])) * rw(m->r); }
static float my_(const Map *m, double y) { return m->r.y1 - (float)((y - m->v[1]) / (m->v[3] - m->v[1])) * rh(m->r); }

/* a hairline frame with ticks along the bottom and left, in world units */
static void frame(const Map *m, double step, const char *xl, const char *yl, int labels) {
    float lw = fmaxf(1.0f, u_ * 0.9f), tk = u_ * 7;
    float *fc = col(c_dim, 0.45f), *tc = col(c_dim, 0.8f);
    Rect r = m->r;
    line(r.x0, r.y0, r.x1, r.y0, lw, fc); line(r.x0, r.y1, r.x1, r.y1, lw, fc);
    line(r.x0, r.y0, r.x0, r.y1, lw, fc); line(r.x1, r.y0, r.x1, r.y1, lw, fc);
    float px = u_ * 13;
    for (double x = ceil(m->v[0] / step) * step; x <= m->v[2] + 1e-9; x += step) {
        float X = mx_(m, x);
        if (X < r.x0 + 2 || X > r.x1 - 2) continue;
        line(X, r.y1, X, r.y1 - tk, lw, fc);
        line(X, r.y0, X, r.y0 + tk, lw, fc);
        if (labels) {
            char b[32];
            snprintf(b, sizeof b, "%g", fabs(x) < 1e-9 ? 0.0 : x);
            text(F_SANS, px, X, r.y1 + px * 1.45f, 0, b, tc, 0, 0);
        }
    }
    for (double y = ceil(m->v[1] / step) * step; y <= m->v[3] + 1e-9; y += step) {
        float Y = my_(m, y);
        if (Y < r.y0 + 2 || Y > r.y1 - 2) continue;
        line(r.x0, Y, r.x0 + tk, Y, lw, fc);
        line(r.x1, Y, r.x1 - tk, Y, lw, fc);
        if (labels) {
            char b[32];
            snprintf(b, sizeof b, "%g", fabs(y) < 1e-9 ? 0.0 : y);
            text(F_SANS, px, r.x0 - px * 0.6f, Y + px * 0.36f, 1, b, tc, 0, 0);
        }
    }
    if (labels && xl) math(px * 1.25f, r.x1, r.y1 + px * 2.9f, 1, xl, tc);
    if (labels && yl) math(px * 1.25f, r.x0 - px * 0.6f, r.y0 - px * 0.7f, 1, yl, tc);
}

/* a small caption over a panel: SMALL CAPS label, then maths */
static void caption(Rect r, const char *label, const char *m) {
    float px = u_ * 12.5f;
    float w = text(F_SANS, px, r.x0, r.y0 - px * 0.95f, -1, label, col(c_dim, 0.95f), px * 0.14f, 0);
    if (m) math(px * 1.3f, r.x0 + w + px * 0.9f, r.y0 - px * 0.95f, -1, m, col(c_ink, 0.85f));
}

static const char *roman[] = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"};

/* the title card: the experiment's number, its name and a line about it */
static float title_head(float x, float y, float width, float scale) {
    float s = u_ * scale;
    char hdr[64];
    snprintf(hdr, sizeof hdr, "%s", roman[(A.num - 1) % 10]);
    /* intro: the card is a little brighter at first */
    float lift = 1.0f - smooth(INTRO + 2, INTRO + 6, (float)act_t) * 0.15f;
    text(F_SANS, 13 * s, x, y, -1, "EXPERIMENT", col(c_acc, 0.9f * lift), 13 * s * 0.22f, 0);
    float wexp = text_run(F_SANS, (int)lroundf(13 * s), 0, 0, "EXPERIMENT", c_acc, 13 * s * 0.22f, 0, 0);
    text(F_SANS, 13 * s, x + wexp + 13 * s * 0.6f, y, -1, hdr, col(c_acc, 0.9f * lift), 13 * s * 0.22f, 0);
    y += 40 * s;
    text(F_LIGHT, 34 * s, x, y, -1, A.title, col(c_ink, 0.96f * lift), 0, 0);
    y += 26 * s;
    text(F_LIGHT, 15 * s, x, y, -1, A.sub, col(c_dim, 0.95f * lift), 0, 0);
    y += 22 * s;
    line(x, y, x + fminf(width, 60 * s), y, fmaxf(1, s * 0.9f), col(c_dim, 0.5f));
    return y;
}
/* and the equations under it; y is the top of the first */
static float title_eqs(float x, float y, float width, float scale) {
    float s = u_ * scale;
    for (int i = 0; i < 2; i++) {
        if (!A.eq[i]) continue;
        float px = 21 * s;
        Box b = math_measure(px, A.eq[i]);
        if (b.w > width) px *= width / b.w;
        b = math_measure(px, A.eq[i]);
        y += fmaxf(b.asc - px * 0.7f, 0);
        math(px, x, y, -1, A.eq[i], col(c_ink, 0.92f));
        y += b.desc + 30 * s;
    }
    return y;
}

/* readouts: a maths label and a tabular value per row */
static float ro_y, ro_x, ro_w, ro_s;
static void ro_row(const char *label, const char *val, const float *vc) {
    float px = 15.5f * ro_s;
    math(px * 1.08f, ro_x, ro_y, -1, label, col(c_dim, 0.95f));
    text(F_SANS, px, ro_x + ro_w, ro_y, 1, val, vc ? vc : col(c_ink, 0.95f), 0, 1);
    ro_y += px * 1.75f;
}
static void readouts(float x, float y, float width, float scale) {
    ro_x = x; ro_y = y; ro_w = width; ro_s = u_ * scale;
    char b[64];
    float *acc = col(c_acc, 0.95f);
    double t = sim_t;
    snprintf(b, sizeof b, "%.2f", t);
    ro_row("t", b, NULL);
    switch (A.id) {
    case A_TUNNEL: {
        double R = M.n * (1 - M.right), T = M.n * M.right;
        snprintf(b, sizeof b, "%.4f", R); ro_row("R", b, NULL);
        snprintf(b, sizeof b, "%.4f", T); ro_row("T", b, acc);
        snprintf(b, sizeof b, "%.4f", A.T_th); ro_row("T_{\\r{theory}}", b, NULL);
        snprintf(b, sizeof b, "%.3f", A.k0 * A.k0 / 2 / A.V0); ro_row("E / V_{0}", b, NULL);
        break;
    }
    case A_SLIT:
        snprintf(b, sizeof b, "%d", nclick); ro_row("\\t{particles}", b, acc);
        snprintf(b, sizeof b, "%.2f", A.lam * A.Lscr / A.d); ro_row("λL/d", b, NULL);
        snprintf(b, sizeof b, "%.3f", A.lam); ro_row("λ", b, NULL);
        break;
    case A_COHERENT: case A_SQUEEZED: {
        snprintf(b, sizeof b, "%+.3f", M.x); ro_row("⟨x⟩", b, NULL);
        snprintf(b, sizeof b, "%+.3f", M.kx); ro_row("⟨p_{x}⟩", b, NULL);
        if (A.id == A_SQUEEZED) {
            double wt = sim_t, s2 = A.s0 * A.s0 / 2 * cos(wt) * cos(wt) + 1 / (2 * A.s0 * A.s0) * sin(wt) * sin(wt);
            snprintf(b, sizeof b, "%.4f", sqrt(fmax(M.xx - M.x * M.x, 0))); ro_row("σ_{x}", b, acc);
            snprintf(b, sizeof b, "%.4f", sqrt(s2)); ro_row("σ_{x}^{\\r{theory}}", b, NULL);
        } else {
            double xc = A.x0 * cos(A.wx * t) + A.px0 / A.wx * sin(A.wx * t), yc = A.y0 * cos(A.wy * t) + A.py0 / A.wy * sin(A.wy * t);
            snprintf(b, sizeof b, "%+.3f", M.y); ro_row("⟨y⟩", b, NULL);
            snprintf(b, sizeof b, "%.1e", hypot(M.x - xc, M.y - yc)); ro_row("|⟨r⟩ - r_{\\r{cl}}|", b, NULL);
        }
        break;
    }
    case A_REVIVAL:
        snprintf(b, sizeof b, "%.4f", sim_t / A.trev); ro_row("t / T_{\\r{rev}}", b, acc);
        snprintf(b, sizeof b, "%.2f", A.trev); ro_row("T_{\\r{rev}}", b, NULL);
        break;
    case A_BRAGG:
        snprintf(b, sizeof b, "%.3f", A.k0); ro_row("|k|", b, NULL);
        snprintf(b, sizeof b, "%.1f°", 90 - A.lrot * 180 / PI); ro_row("θ_{B}", b, acc);
        snprintf(b, sizeof b, "%+.3f, %+.3f", M.kx, M.ky); ro_row("⟨p⟩", b, NULL);
        break;
    case A_STADIUM:
        snprintf(b, sizeof b, "%+.3f, %+.3f", M.x, M.y); ro_row("⟨r⟩", b, NULL);
        break;
    }
    /* conservation */
    snprintf(b, sizeof b, "%.6f", M.n);
    ro_row("‖ψ‖^{2}", b, NULL);
    if (A.id != A_REVIVAL) {
        double E = M.T + M.V;
        snprintf(b, sizeof b, "%.4f", E);
        ro_row("⟨H⟩", b, NULL);
        int absorbing = A.id == A_TUNNEL || A.id == A_SLIT || A.id == A_BRAGG || (A.id == A_STADIUM && A.corral);
        if (!absorbing && e0_set) {
            snprintf(b, sizeof b, "%.1e", fabs(E - E0) / fmax(fabs(E0), 1e-9));
            ro_row("|ΔE| / E", b, NULL);
        }
    }
}

/* the time, and the time away */
static time_t locked_at;
static void clock_caption(float xr, float yb, float scale) {
    float s = u_ * scale;
    time_t tnow = time(NULL);
    struct tm lt;
    localtime_r(&tnow, &lt);
    char tb[32], ab[64] = "";
    snprintf(tb, sizeof tb, "%d:%02d", lt.tm_hour, lt.tm_min);
    if (locked_at && tnow >= locked_at) {
        long a = (long)(tnow - locked_at);
        if (a < 3600) snprintf(ab, sizeof ab, "away %ld:%02ld", a / 60, a % 60);
        else snprintf(ab, sizeof ab, "away %ldh %02ldm", a / 3600, a / 60 % 60);
    }
    if (ab[0]) { text(F_SANS, 13 * s, xr, yb, 1, ab, colk(c_dim, 0.9f), 13 * s * 0.08f, 1); yb -= 13 * s * 1.9f; }
    text(F_LIGHT, 36 * s, xr, yb, 1, tb, colk(c_ink, 0.8f), 0, 1);
}

/* ---- the panels ---- */
static void view_of(double v[4]) { v[0] = A.vx0; v[1] = A.vy0; v[2] = A.vx1; v[3] = A.vy1; }
static double view_aspect(void) { return (A.vx1 - A.vx0) / (A.vy1 - A.vy0); }

static void sim_panel(Rect r, int labels) {
    double grid[4] = {G.x0, G.y0, G.nx * G.dx, G.ny * G.dy}, v[4];
    view_of(v);
    double ref = fmax(expo_r, 1e-12);
    float expo = (float)(1.0 / (0.32 * ref));
    make_glow(G.psi.t[0]);
    pan_relmode = A.id == A_BRAGG || (A.id == A_STADIUM && A.corral) ? 1 : A.id == A_COHERENT || A.id == A_SQUEEZED ? 2 : 0;
    pan_E = (float)(e0_set ? E0 : 1);
    panel(r, 0, G.psi.t[0], G.nx, G.ny, grid, v, expo, G.vtex, (float)G.vmax, G.glow[0], 1.0f, 0);
    Map m = {r, {v[0], v[1], v[2], v[3]}};
    double span_ = v[2] - v[0];
    double step = span_ > 50 ? 10 : span_ > 20 ? 5 : span_ > 8 ? 2 : 1;
    frame(&m, step, "x", "y", labels);
    float lw = fmaxf(1, u_ * 1.1f);
    switch (A.id) {
    case A_SLIT: {
        /* the screen */
        float X = mx_(&m, A.xs);
        dashed(X, r.y0, X, r.y1, lw, u_ * 4, u_ * 6, col(c_dim, 0.6f));
        /* the latest arrivals flash on the screen */
        for (int i = 0; i < nclick; i++) {
            float age = (float)(sim_t - clicks[i][1]);
            if (age < 0 || age > 1.2f) continue;
            float Y = my_(&m, clicks[i][0]);
            if (Y < r.y0 || Y > r.y1) continue;
            dot(X, Y, u_ * 1.6f, col(c_ink, 0.9f * expf(-age * 3)));
        }
        break;
    }
    case A_COHERENT: case A_SQUEEZED: {
        /* the classical orbit (dashed) and the centroid's trail */
        int n = 240;
        double T = A.wy != A.wx ? 4 * PI : 2 * PI;
        float px = 0, py = 0;
        for (int i = 0; i <= n; i++) {
            double t = T * i / n;
            double x = A.x0 * cos(A.wx * t) + A.px0 / A.wx * sin(A.wx * t), y = A.y0 * cos(A.wy * t) + A.py0 / A.wy * sin(A.wy * t);
            float X = mx_(&m, x), Y = my_(&m, y);
            if (i && (i & 1)) line(px, py, X, Y, lw, col(c_ink, 0.22f));
            px = X; py = Y;
        }
        for (int i = 1; i < hn; i++) {
            if (sim_t - hist[i][0] > 2.5) continue;
            float a = (float)(1 - (sim_t - hist[i][0]) / 2.5);
            line(mx_(&m, hist[i - 1][H_X]), my_(&m, hist[i - 1][H_Y]), mx_(&m, hist[i][H_X]), my_(&m, hist[i][H_Y]), lw * 1.3f, col(c_acc, 0.7f * a));
        }
        if (hn) {
            dot(mx_(&m, M.x), my_(&m, M.y), u_ * 3.2f, col(c_acc, 0.95f));
        }
        break;
    }
    case A_BRAGG: {
        /* the incoming beam and the Bragg-reflected one */
        double kx = A.k0, gx = -cos(A.lrot) * 2 * PI / A.latt, gy = sin(A.lrot) * 2 * PI / A.latt;
        double ox = kx + gx, oy = gy, l = hypot(ox, oy);
        float cx = mx_(&m, 0), cy = my_(&m, 0), L = rw(r) * 0.42f;
        dashed(cx - L, cy, cx, cy, lw, u_ * 5, u_ * 6, col(c_ink, 0.25f));
        dashed(cx, cy, cx + (float)(ox / l) * L, cy - (float)(oy / l) * L, lw, u_ * 5, u_ * 6, col(c_acc, 0.45f));
        break;
    }
    case A_TUNNEL: {
        float X = mx_(&m, A.a * 0.5);
        math(u_ * 15, X, r.y0 - u_ * 9, 0, "V_{0}", col(c_dim, 0.9f));
        break;
    }
    default: break;
    }
}

/* tunnelling's momentum picture is one-dimensional: the distribution of
 * kx, against the packet's at the start, and what T(k) says comes through */
static void k_plot_tunnel(Rect r, int labels) {
    float lw = fmaxf(1, u_ * 1.3f), px = u_ * 12.5f;
    caption(r, "MOMENTUM", "|φ(k_{x})|^{2}");
    Rect p = {r.x0, r.y0 + px * 0.6f, r.x1, r.y1 - px * 2.2f};
    double k0 = -6.5, k1 = 6.5, sk = 1 / (2 * A.sigx), p0max = 1 / (sqrt(2 * PI) * sk);
    float *fc = col(c_dim, 0.5f), *tc = col(c_dim, 0.85f);
    line(p.x0, p.y1, p.x1, p.y1, fmaxf(1, u_ * 0.9f), fc);
    for (int k = -6; k <= 6; k += 2) {
        float X = p.x0 + (float)((k - k0) / (k1 - k0)) * rw(p);
        line(X, p.y1, X, p.y1 + u_ * 5, fmaxf(1, u_ * 0.9f), fc);
        char b[16];
        snprintf(b, sizeof b, "%d", k);
        text(F_SANS, px, X, p.y1 + px * 1.6f, 0, b, tc, 0, 0);
    }
    math(px * 1.2f, p.x1, p.y1 + px * 3.2f, 1, "k_{x}", tc);
    (void)labels;
#define KX(k) (p.x0 + (float)(((k) - k0) / (k1 - k0)) * rw(p))
#define KY(v) (p.y1 - (float)((v) / (1.15 * p0max)) * rh(p))
    /* the packet as it set out, and T(k) times it */
    float la = 0, lb = 0, ta = 0, tb = 0;
    for (int i = 0; i <= 200; i++) {
        double k = k0 + (k1 - k0) * i / 200, p0 = exp(-(k - A.k0) * (k - A.k0) / (2 * sk * sk)) / (sqrt(2 * PI) * sk);
        double pt = k > 0 ? p0 * T_square(k * k / 2, A.V0, A.a) : 0;
        float X = KX(k), Y = KY(p0), Yt = KY(pt);
        if (i) {
            line(la, lb, X, Y, lw * 0.8f, col(c_dim, 0.45f));
            line(ta, tb, X, Yt, lw, col(c_acc, 0.85f));
        }
        la = X; lb = Y; ta = X; tb = Yt;
    }
    /* now */
    if (have_marg && M.kn > 0) {
        double dk = 2 * PI / (G.nx * G.dx);
        float lx = 0, ly = 0;
        int first = 1;
        for (int i = 0; i < G.nx; i++) {
            int m = (i + G.nx / 2) % G.nx;       /* in order of k */
            int mm = m < G.nx / 2 ? m : m - G.nx;
            double k = mm * dk;
            if (k < k0 || k > k1) continue;
            double v = kmarg[m] / (M.kn * dk);
            float X = KX(k), Y = KY(v);
            fill(X - (float)(dk / (k1 - k0) * rw(p)) * 0.5f, Y, X + (float)(dk / (k1 - k0) * rw(p)) * 0.5f, p.y1, col(c_ink, 0.10f));
            if (!first) line(lx, ly, X, Y, lw, col(c_ink, 0.9f));
            lx = X; ly = Y; first = 0;
        }
    }
    float yk = p.y0 + px * 0.2f;
    math(px * 1.1f, KX(A.k0) + px * 2.2f, yk + px * 1.0f, -1, "T(k)|φ_{0}|^{2}", col(c_acc, 0.9f));
#undef KX
#undef KY
}

static void k_panel(Rect r, int labels) {
    if (A.id == A_TUNNEL) { k_plot_tunnel(r, labels); return; }
    if (!G.kshow) return;
    double nx = G.nx, ny = G.ny, dkx = 2 * PI / (G.nx * G.dx), dky = 2 * PI / (G.ny * G.dy);
    double grid[4] = {-0.5 * dkx, -0.5 * dky, nx * dkx, ny * dky};
    double kv = A.kview, asp = rw(r) / rh(r), kc = A.kcx;
    double v[4] = {kc - kv * fmax(asp, 1), -kv * fmax(1 / asp, 1), kc + kv * fmax(asp, 1), kv * fmax(1 / asp, 1)};
    float expo = (float)(1.0 / (0.25 * fmax(M.kmax, 1e-20)));
    panel(r, 1, G.kshow, G.nx, G.ny, grid, v, expo, 0, 1, 0, 0, 0);
    (void)ny; (void)nx;
    Map m = {r, {v[0], v[1], v[2], v[3]}};
    if (!labels) caption(r, "MOMENTUM", "|φ(k)|^{2}");
    frame(&m, kv > 8 ? 5 : 2, "k_{x}", "k_{y}", labels);
    float lw = fmaxf(1, u_ * 1.0f);
    float cx = mx_(&m, 0), cy = my_(&m, 0);
    line(cx - u_ * 5, cy, cx + u_ * 5, cy, lw, col(c_dim, 0.6f));
    line(cx, cy - u_ * 5, cx, cy + u_ * 5, lw, col(c_dim, 0.6f));
    if (A.id == A_BRAGG) {
        /* the Ewald circle |k'| = |k| and the lattice points k + G */
        float R = mx_(&m, A.k0) - cx;
        ring(cx, cy, R, lw, col(c_acc, 0.5f), 1);
        double G0 = 2 * PI / A.latt, gx = -cos(A.lrot), gy = sin(A.lrot), ux = -gy, uy = gx;
        for (int p = -3; p <= 3; p++)
            for (int q = -3; q <= 3; q++) {
                double kx = A.k0 + G0 * (p * gx + q * ux), ky = G0 * (p * gy + q * uy);
                float X = mx_(&m, kx), Y = my_(&m, ky);
                if (X < r.x0 || X > r.x1 || Y < r.y0 || Y > r.y1) continue;
                ring(X, Y, u_ * 4, lw, col(c_ink, 0.35f), 0);
            }
    }
    if (A.id == A_COHERENT || A.id == A_SQUEEZED || A.id == A_STADIUM) {
        dot(mx_(&m, M.kx), my_(&m, M.ky), u_ * 2.6f, col(c_acc, 0.9f));
    }
}

/* the plots and pictures that go with each act */
static void plot_axes(Rect r, double t1, double y0, double y1, const char *xl, const char *yl) {
    float lw = fmaxf(1, u_ * 0.9f);
    float *fc = col(c_dim, 0.5f), *tc = col(c_dim, 0.85f);
    line(r.x0, r.y1, r.x1, r.y1, lw, fc);
    line(r.x0, r.y0, r.x0, r.y1, lw, fc);
    float px = u_ * 12;
    char b[32];
    snprintf(b, sizeof b, "%g", y1);
    text(F_SANS, px, r.x0 - px * 0.5f, r.y0 + px * 0.4f, 1, b, tc, 0, 0);
    snprintf(b, sizeof b, "%g", y0);
    text(F_SANS, px, r.x0 - px * 0.5f, r.y1 + px * 0.4f, 1, b, tc, 0, 0);
    snprintf(b, sizeof b, "%g", t1);
    text(F_SANS, px, r.x1, r.y1 + px * 1.5f, 1, b, tc, 0, 0);
    if (xl) math(px * 1.2f, r.x1 - px * 3.2f, r.y1 + px * 1.5f, 1, xl, tc);
    if (yl) math(px * 1.2f, r.x0 + px * 0.6f, r.y0 + px * 1.0f, -1, yl, tc);
}
static void curve(Rect r, double t1, double y0, double y1, int idx, float lw, const float *c) {
    for (int i = 1; i < hn; i++) {
        float xa = r.x0 + (float)(hist[i - 1][0] / t1) * rw(r), xb = r.x0 + (float)(hist[i][0] / t1) * rw(r);
        float ya = r.y1 - (float)((hist[i - 1][idx] - y0) / (y1 - y0)) * rh(r), yb = r.y1 - (float)((hist[i][idx] - y0) / (y1 - y0)) * rh(r);
        if (xb > r.x1) break;
        line(xa, clampf(ya, r.y0, r.y1), xb, clampf(yb, r.y0, r.y1), lw, c);
    }
}

static void aux_panel(Rect r, int big) {
    float lw = fmaxf(1, u_ * 1.4f);
    float *acc = col(c_acc, 0.95f), *ink = col(c_ink, 0.85f), *dim = col(c_dim, 0.7f);
    float px = u_ * 12.5f;
    switch (A.id) {
    case A_TUNNEL: {
        caption(r, "PROBABILITY", "R(t),\\; T(t)");
        Rect p = {r.x0 + px * 2.5f, r.y0 + px * 0.5f, r.x1, r.y1 - px * 2.0f};
        double t1 = A.max_steps * G.dt;
        plot_axes(p, t1, 0, 1, "t", NULL);
        float Y = p.y1 - (float)A.T_th * rh(p);
        dashed(p.x0, Y, p.x1, Y, lw * 0.8f, u_ * 6, u_ * 5, col(c_acc, 0.5f));
        math(px * 1.15f, p.x1 - px * 0.3f, Y - px * 0.5f, 1, "T_{\\r{theory}}", col(c_acc, 0.8f));
        curve(p, t1, 0, 1, H_A, lw, ink);
        curve(p, t1, 0, 1, H_B, lw, acc);
        if (hn) {
            float x = p.x0 + (float)(hist[hn - 1][0] / t1) * rw(p);
            math(px * 1.2f, x + px * 0.5f, p.y1 - hist[hn - 1][H_A] * rh(p) + px * 0.4f, -1, "R", ink);
            math(px * 1.2f, x + px * 0.5f, p.y1 - hist[hn - 1][H_B] * rh(p) + px * 0.4f, -1, "T", acc);
        }
        break;
    }
    case A_SLIT: {
        /* the screen: the particles as they land, their histogram, and the
         * wave's |psi|^2 there; y runs along the panel's long side */
        caption(r, "SCREEN", "\\t{x} = L");
        int vertical = rh(r) > rw(r);
        double y0 = A.vy0, y1 = A.vy1;
        int nb = 72;
        static float hb[256];
        memset(hb, 0, sizeof hb);
        int nin = 0;
        for (int i = 0; i < nclick; i++) {
            if (clicks[i][1] > sim_t) continue;
            int b = (int)((clicks[i][0] - y0) / (y1 - y0) * nb);
            if (b >= 0 && b < nb) { hb[b] += 1; nin++; }
        }
        /* both as densities over the screen shown, on one scale: the wave's
         * |psi|^2 and the particles' histogram */
        double wsum = 0, wpk = 1e-30;
        for (int j = 0; j < G.ny; j++)
            if (YJ(j) >= y0 && YJ(j) <= y1) { wsum += scr_acc[j] * G.dy; wpk = fmax(wpk, scr_acc[j]); }
        double wmax = wpk;
        float hmax = nin ? (float)(wpk / fmax(wsum, 1e-30) * nin * (y1 - y0) / nb) * 1.08f : 1;
        wmax *= 1.08;
        /* along: y; across: the histogram height */
        Rect p = r;
        float strip = (vertical ? rw(p) : rh(p)) * 0.30f;
        for (int b = 0; b < nb; b++) {
            float a0 = (float)b / nb, a1 = (float)(b + 1) / nb, hgt = fminf(hb[b] / hmax, 1.0f);
            if (vertical) fill(p.x0 + strip * 1.1f, p.y1 - a1 * rh(p) + 0.5f, p.x0 + strip * 1.1f + hgt * (rw(p) - strip * 1.1f), p.y1 - a0 * rh(p) - 0.5f, col(c_acc, 0.30f * fminf(1.0f, nin / 500.0f)));
            else fill(p.x0 + a0 * rw(p) + 0.5f, p.y1 - hgt * (rh(p) - strip * 1.1f), p.x0 + a1 * rw(p) - 0.5f, p.y1, col(c_acc, 0.30f * fminf(1.0f, nin / 500.0f)));
        }
        /* the wave's prediction */
        float lx = 0, ly = 0;
        int first = 1;
        for (int j = 0; j < G.ny; j++) {
            double y = YJ(j);
            if (y < y0 || y > y1) continue;
            float a = (float)((y - y0) / (y1 - y0)), h = (float)(scr_acc[j] / wmax);
            float X, Yv;
            if (vertical) { X = p.x0 + strip * 1.1f + h * (rw(p) - strip * 1.1f); Yv = p.y1 - a * rh(p); }
            else { X = p.x0 + a * rw(p); Yv = p.y1 - h * (rh(p) - strip * 1.1f); }
            if (!first) line(lx, ly, X, Yv, lw, ink);
            lx = X; ly = Yv; first = 0;
        }
        /* the particles, scattered across a strip */
        for (int i = 0; i < nclick; i++) {
            float age = (float)(sim_t - clicks[i][1]);
            if (age < 0) continue;
            float a = (float)((clicks[i][0] - y0) / (y1 - y0));
            if (a < 0 || a > 1) continue;
            rng = mix32((uint32_t)i * 747796405u + 1);
            float j = (float)drand();
            float fr = 0.35f + 0.6f * expf(-age * 1.5f);
            if (vertical) dot(p.x0 + j * strip, p.y1 - a * rh(p), u_ * (big ? 1.6f : 1.2f), col(c_ink, fr));
            else dot(p.x0 + a * rw(p), p.y0 + j * strip, u_ * (big ? 1.6f : 1.2f), col(c_ink, fr));
        }
        float fl = fmaxf(1, u_ * 0.9f);
        line(r.x0, r.y0, r.x1, r.y0, fl, dim); line(r.x0, r.y1, r.x1, r.y1, fl, dim);
        line(r.x0, r.y0, r.x0, r.y1, fl, dim); line(r.x1, r.y0, r.x1, r.y1, fl, dim);
        /* the fringe spacing lambda L / d, marked about the middle */
        double dy = A.lam * A.Lscr / A.d;
        for (int k = -3; k <= 3; k++) {
            double y = k * dy;
            if (y < y0 || y > y1) continue;
            float a = (float)((y - y0) / (y1 - y0));
            if (vertical) line(r.x1 - u_ * 8, r.y1 - a * rh(r), r.x1, r.y1 - a * rh(r), fl, col(c_acc, 0.8f));
            else line(r.x0 + a * rw(r), r.y0, r.x0 + a * rw(r), r.y0 + u_ * 8, fl, col(c_acc, 0.8f));
        }
        break;
    }
    case A_COHERENT: case A_SQUEEZED: {
        /* phase space (x, p_x): the classical circle and the state's
         * uncertainty ellipse, moving round it */
        caption(r, "PHASE SPACE", "(x, p_{x})");
        Rect q = fit(r, 1, 0.5f);
        double ext = 8.5;
        Map m = {q, {-ext, -ext, ext, ext}};
        frame(&m, 2, "x", "p_{x}", big);
        float cx = mx_(&m, 0), cy = my_(&m, 0), R = mx_(&m, A.x0) - cx;
        ring(cx, cy, R, lw * 0.8f, col(c_dim, 0.5f), 1);
        line(q.x0, cy, q.x1, cy, fmaxf(1, u_ * 0.7f), col(c_dim, 0.25f));
        line(cx, q.y0, cx, q.y1, fmaxf(1, u_ * 0.7f), col(c_dim, 0.25f));
        /* the trail in phase space */
        for (int i = 1; i < hn; i++) {
            if (sim_t - hist[i][0] > 3) continue;
            float a = (float)(1 - (sim_t - hist[i][0]) / 3);
            line(mx_(&m, hist[i - 1][H_X]), my_(&m, hist[i - 1][H_PX]), mx_(&m, hist[i][H_X]), my_(&m, hist[i][H_PX]), lw * 1.2f, col(c_acc, 0.6f * a));
        }
        /* the analytic ellipse (a Gaussian's 1-sigma contour in phase space), rotating with omega t */
        double t = sim_t, s0 = A.id == A_SQUEEZED ? A.s0 : 1;
        double sx0 = s0 / sqrt(2), sp0 = 1 / (s0 * sqrt(2));
        double xc = A.x0 * cos(t) + A.px0 * sin(t), pc = -A.x0 * sin(t) + A.px0 * cos(t);
        int n = 72;
        float lx = 0, ly = 0;
        for (int i = 0; i <= n; i++) {
            double a = 2 * PI * i / n, X0 = sx0 * cos(a), P0 = sp0 * sin(a);
            double X = X0 * cos(t) + P0 * sin(t), Pp = -X0 * sin(t) + P0 * cos(t);
            float Xs = mx_(&m, xc + X), Ys = my_(&m, pc + Pp);
            if (i) line(lx, ly, Xs, Ys, lw, col(c_acc, 0.9f));
            lx = Xs; ly = Ys;
        }
        /* the measured centroid and spreads */
        if (M.have_k) {
            double sx = sqrt(fmax(M.xx - M.x * M.x, 0)), sp = sqrt(fmax(M.kxx - M.kx * M.kx, 0));
            float X = mx_(&m, M.x), Y = my_(&m, M.kx);
            line(mx_(&m, M.x - sx), Y, mx_(&m, M.x + sx), Y, lw * 0.8f, ink);
            line(X, my_(&m, M.kx - sp), X, my_(&m, M.kx + sp), lw * 0.8f, ink);
            dot(X, Y, u_ * 2.6f, ink);
        }
        break;
    }
    case A_REVIVAL: {
        caption(r, "QUANTUM CARPET", "|ψ(x,t)|^{2}");
        double grid[4] = {-A.Lbox, 0, 2 * A.Lbox, 1}, v[4] = {0, 1, A.Lbox, 0};   /* t = 0 at the top */
        Rect q = r;
        double tau = sim_t / A.trev;
        float expo = (float)(1.0 / (0.22 * (1.0 / (sqrt(2 * PI) * A.s0))));
        panel(q, 2, carpet_tex, CAR_N, CAR_R, grid, v, expo, 0, 1, 0, 0, (float)tau);
        Map m = {q, {0, 0, A.Lbox, 1}};
        float fl = fmaxf(1, u_ * 0.9f);
        line(q.x0, q.y0, q.x1, q.y0, fl, dim); line(q.x0, q.y1, q.x1, q.y1, fl, dim);
        line(q.x0, q.y0, q.x0, q.y1, fl, dim); line(q.x1, q.y0, q.x1, q.y1, fl, dim);
        /* the fractional revivals, marked down the side */
        static const int fr[][2] = {{1, 4}, {1, 2}, {3, 4}, {1, 1}, {1, 3}, {2, 3}};
        for (int i = 0; i < (big ? 6 : 4); i++) {
            double tt = (double)fr[i][0] / fr[i][1];
            float Y = q.y0 + (float)tt * rh(q);
            line(q.x1, Y, q.x1 + u_ * 6, Y, fl, col(c_acc, 0.8f));
            char b[16];
            if (fr[i][1] == 1) snprintf(b, sizeof b, "T_{\\r{rev}}");
            else snprintf(b, sizeof b, "\\f{%d}{%d}", fr[i][0], fr[i][1]);
            math(px * 1.0f, q.x1 + u_ * 9, Y + px * 0.35f, -1, b, col(c_acc, 0.85f));
        }
        float Y = q.y0 + (float)fmin(tau, 1) * rh(q);
        line(q.x0, Y, q.x1, Y, fl, col(c_ink, 0.6f));
        (void)m;
        break;
    }
    case A_BRAGG: {
        /* the far field: |phi|^2 around the Ewald ring, by angle */
        caption(r, "DIFFRACTION", "|φ(k)|^{2},\\; |k| ≈ k_{0}");
        Rect q = fit(r, 1, 0.5f);
        if (G.kshow) {
            double dkx = 2 * PI / (G.nx * G.dx), dky = 2 * PI / (G.ny * G.dy);
            double grid[4] = {0, 0, 1, 1}, v[4] = {-1, -1, 1, 1};
            pan_k0 = (float)A.k0; pan_kw = 1.4f; pan_kr[0] = (float)(G.nx * dkx); pan_kr[1] = (float)(G.ny * dky);
            panel(q, 4, G.kshow, G.nx, G.ny, grid, v, (float)(1.0 / (0.5 * 9 * fmax(M.kmax, 1e-20))), 0, 1, 0, 0, 0);
        }
        float cx = (q.x0 + q.x1) / 2, cy = (q.y0 + q.y1) / 2, R = rw(q) * 0.5f * 0.98f;
        ring(cx, cy, rw(q) * 0.5f * 0.42f, fmaxf(1, u_ * 0.8f), col(c_dim, 0.4f), 0);
        /* the incoming direction and the Bragg direction */
        double gx = -cos(A.lrot) * 2 * PI / A.latt, gy = sin(A.lrot) * 2 * PI / A.latt;
        double bx = A.k0 + gx, by = gy, l = hypot(bx, by);
        line(cx, cy, cx + R * 1.15f, cy, fmaxf(1, u_ * 0.8f), col(c_dim, 0.6f));
        line(cx, cy, cx + R * 1.15f * (float)(bx / l), cy - R * 1.15f * (float)(by / l), fmaxf(1, u_ * 0.8f), col(c_acc, 0.7f));
        math(px * 1.2f, cx + R * 1.22f * (float)(bx / l), cy - R * 1.22f * (float)(by / l), 0, "2θ_{B}", col(c_acc, 0.9f));
        break;
    }
    case A_STADIUM: {
        caption(r, "LONG EXPOSURE", "\\o{|ψ|^{2}}");
        double grid[4] = {G.x0, G.y0, G.nx * G.dx, G.ny * G.dy}, v[4];
        view_of(v);
        Rect q = fit(r, (float)view_aspect(), 0.5f);
        float expo = (float)(1.0 / (0.30 * fmax(expo_r, 1e-12) * fmax((double)avg_n, 1) * 0.35));
        panel(q, 3, avg_tex, G.nx, G.ny, grid, v, expo, 0, 1, 0, 0, 0);
        float fl = fmaxf(1, u_ * 0.9f);
        line(q.x0, q.y0, q.x1, q.y0, fl, dim); line(q.x0, q.y1, q.x1, q.y1, fl, dim);
        line(q.x0, q.y0, q.x0, q.y1, fl, dim); line(q.x1, q.y0, q.x1, q.y1, fl, dim);
        break;
    }
    }
}

/* ---- the layouts ----
 * The front output carries the experiment, its card and its numbers; on
 * its own it also carries the momentum picture and the act's own panel.
 * How they're arranged goes by the shape of the experiment: a band (the
 * barrier), a wide box (the slits, the stadium) or a square. */
#define CLOCK_W (210 * u_)
static void render_front(void) {
    float m = 54 * u_;
    double asp = view_aspect();
    int insets = span_me < 0;
    Rect sim, kr = {0, 0, 0, 0}, ar = {0, 0, 0, 0};
    float gap = 34 * u_;
    if (W >= H * 1.1f && asp >= 2.4) {
        /* the band: card, equations and momentum across the top, the
         * experiment across the middle, its plot and the numbers below */
        float top = m + 250 * u_;
        float y0 = top + 40 * u_;
        float bottom = insets ? H - m - 300 * u_ : H - m - 60 * u_;
        sim = fit((Rect){m * 1.6f, y0, W - m, bottom}, (float)asp, insets ? 0.0f : 0.5f);
        float ty = m + 24 * u_;
        float hy = title_head(m * 1.6f, ty, 400 * u_, 1.0f);
        (void)hy;
        float ex = m * 1.6f + 470 * u_;
        float ew = (insets ? W - m - (top - m) - 60 * u_ : W - m) - ex;
        title_eqs(ex, ty + 18 * u_, fminf(ew, 620 * u_), 1.0f);
        if (insets) {
            float s = top - m;
            kr = (Rect){W - m - s, m + 6 * u_, W - m, m + 6 * u_ + s};
            ar = (Rect){sim.x0 + 30 * u_, sim.y1 + 92 * u_, sim.x0 + (sim.x1 - sim.x0) * 0.58f, H - m - 10 * u_};
            readouts(ar.x1 + 90 * u_, ar.y0 + 6 * u_, fminf(300 * u_, W - m - CLOCK_W - ar.x1 - 120 * u_), 1.0f);
        } else readouts(W - m - 300 * u_, m + 30 * u_, 300 * u_, 1.0f);   /* where the momentum would be */
    } else if (W >= H * 1.1f) {
        /* a column on the right for the card and the numbers */
        float colw = fmaxf(W * 0.26f, 380 * u_);
        float cx = W - colw - m * 0.2f;
        Rect left = {m * 1.6f, m, cx - m * 1.2f, H - m * 1.2f};
        if (asp >= 1.45) {
            if (insets) {
                sim = fit((Rect){left.x0, left.y0, left.x1, left.y0 + rh(left) * 0.62f}, (float)asp, 0);
                float bt = sim.y1 + 78 * u_, s = H - m - bt;
                kr = (Rect){sim.x1 - s, bt, sim.x1, bt + s};
                ar = (Rect){sim.x0 + 20 * u_, bt, kr.x0 - gap * 2.2f, bt + s};
            } else sim = fit(left, (float)asp, 0.4f);
        } else {
            sim = fit(left, (float)asp, 0.5f);
            sim.x1 -= sim.x0 - left.x0; sim.x0 = left.x0;
            cx = sim.x1 + m * 1.5f; colw = W - m - cx;
            if (insets) {
                float s = fminf((colw - gap) / 2, rh(left) * 0.30f);
                float by = H - m - 70 * u_;
                kr = (Rect){cx, by - s, cx + s, by};
                ar = (Rect){cx + s + gap, by - s, cx + 2 * s + gap, by};
            }
        }
        float y = title_head(cx, m + 30 * u_, colw - m, 1.0f);
        y = title_eqs(cx, y + 44 * u_, colw - m * 0.5f, 1.0f);
        readouts(cx, y + 6 * u_, fminf(colw - m, 320 * u_), 1.0f);
    } else {
        /* portrait: the card and numbers across the top, the experiment,
         * then the two panels side by side */
        float x0 = m * 1.5f, x1 = W - m;
        float y = title_head(x0, m + 30 * u_, W * 0.5f, 1.0f);
        readouts(W * 0.60f, m + 34 * u_, x1 - W * 0.60f, 1.0f);
        y = title_eqs(x0, y + 44 * u_, W * 0.52f, 1.0f);
        float sy = fmaxf(y + 40 * u_, H * 0.25f);
        if (insets) {
            float s = (x1 - x0 - gap * 2) / 2;
            float room = H - m - 90 * u_ - sy;          /* for the sim, a gap and the insets */
            float sh = room - s - 110 * u_;
            if (sh < (x1 - x0) / (float)asp * 0.6f) { s *= 0.8f; sh = room - s - 110 * u_; }
            sim = fit((Rect){x0, sy, x1, sy + sh}, (float)asp, 0.0f);
            /* the sim and the insets as one block, centred in the room */
            float blk = rh(sim) + 100 * u_ + s, sh0 = (room - blk) * 0.5f;
            if (sh0 > 0) { sim.y0 += sh0; sim.y1 += sh0; }
            float iy = sim.y1 + 100 * u_;
            kr = (Rect){x0, iy, x0 + s, iy + s};
            ar = (Rect){x1 - s, iy, x1, iy + s};
        } else sim = fit((Rect){x0, sy, x1, H - m - 100 * u_}, (float)asp, 0.35f);
    }
    sim_panel(sim, 1);
    if (insets && rw(kr) > 0) {
        k_panel(kr, 0);
        aux_panel(ar, 0);
    }
}

static void render_momentum(void) {
    float m = 60 * u_;
    Rect area = {m * 1.5f, m * 2.2f, W - m, H - m * 1.6f};
    Rect r = fit(area, 1, 0.5f);
    k_panel(r, 1);
    if (A.id != A_TUNNEL) caption(r, "MOMENTUM SPACE", "|φ(k_{x}, k_{y})|^{2}");   /* (tunnelling's plot has its own) */
    float x = W > H ? fminf(r.x1 + m, W - 300 * u_) : m;
    if (W > H && x + 200 * u_ < W) {
        ro_x = x; ro_y = r.y0 + 20 * u_; ro_w = fminf(260 * u_, W - m - x); ro_s = u_;
        char b[64];
        snprintf(b, sizeof b, "%+.3f", M.kx); ro_row("⟨p_{x}⟩", b, NULL);
        snprintf(b, sizeof b, "%+.3f", M.ky); ro_row("⟨p_{y}⟩", b, NULL);
        snprintf(b, sizeof b, "%.4f", M.T); ro_row("⟨p^{2}⟩ / 2m", b, NULL);
    }
}

static void render_aux(void) {
    float m = 60 * u_;
    Rect area = {m * 1.6f, m * 2.4f, W - m * 1.6f, H - m * 1.8f};
    if (A.id == A_REVIVAL) {
        /* the carpet: x across, t down */
        Rect r = fit(area, rw(area) / rh(area) > 0.8f ? 0.8f : rw(area) / rh(area), 0.5f);
        aux_panel(r, 1);
    } else if (A.id == A_TUNNEL) {
        Rect r = fit(area, 1.6f, 0.5f);
        aux_panel(r, 1);
    } else aux_panel(area, 1);
}

static void draw_scene(GLuint target) {
    /* every text size follows the output's: on a resize the glyphs start over */
    static int atlas_w, atlas_h;
    if (atlas_full || atlas_w != W || atlas_h != H) { atlas_reset(); atlas_w = W; atlas_h = H; }
    u_ = fminf(W, H) / 1000.0f;
    if (W < H) u_ = W / 1000.0f * 1.15f;
    double L = act_len[A.id];
    fade_a = smooth(0.2f, 0.2f + (float)FADE, (float)act_t) * (1 - smooth((float)(L - FADE), (float)L - 0.1f, (float)act_t));
    draw_target = target;
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glUseProgram(prog_bg);
    glUniform3fv(U(prog_bg, "uBg"), 1, c_bg);
    glUniform2f(U(prog_bg, "uRes"), (float)W, (float)H);
    glUniform1i(U(prog_bg, "uMode"), P->mode);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ovn = 0;
    if (role == R_MOMENTUM) render_momentum();
    else if (role == R_AUX) render_aux();
    else render_front();
    if (role == R_FRONT && !opt_nocap) clock_caption(W - 52 * u_, H - 46 * u_, 1.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, target);
    ov_flush(target);
}

static void gl_init(void) {
    char ext_ok = 0;
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (int i = 0; i < n; i++)
        if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i), "GL_ARB_gpu_shader5")) ext_ok = 1;
    have_exact = ext_ok;
    snprintf(fs_head, sizeof fs_head, "#version 330 core\n%s", have_exact ? "#extension GL_ARB_gpu_shader5 : require\n#define EXACT 1\n" : "");
    vs_full = compile(GL_VERTEX_SHADER, VS_FULL, NULL);
    glGenVertexArrays(1, &vao);
    prog_fft = mk_fs(FS_FFT);
    prog_red1 = mk_fs(FS_RED1);
    prog_red2 = mk_fs(FS_RED2);
    prog_rho = mk_fs(FS_RHO);
    prog_glow = mk_fs(FS_GLOW);
    prog_phase = mk_fs(FS_PHASE);
    prog_marg = mk_fs(FS_MARG);
    prog_bg = mk_fs(FS_BG);
    prog_panel = mk_fs(FS_PANEL);
    prog_ov = mk_link(compile(GL_VERTEX_SHADER, VS_OV, NULL), compile(GL_FRAGMENT_SHADER, FS_OV, NULL));
    glUseProgram(prog_fft);
    glUniform1i(U(prog_fft, "uSrc"), 0); glUniform1i(U(prog_fft, "uPre"), 1); glUniform1i(U(prog_fft, "uPost"), 2);
    fu_axis = U(prog_fft, "uAxis"); fu_n = U(prog_fft, "uN"); fu_ns = U(prog_fft, "uNs"); fu_radix = U(prog_fft, "uRadix");
    fu_pre = U(prog_fft, "uPreOn"); fu_post = U(prog_fft, "uPostOn"); fu_sign = U(prog_fft, "uSign");
    float tw[1024 * 4];
    for (int m = 0; m < 1024; m++) mult(-2 * PI * m / 1024, 1, tw + 4 * m);
    glGenBuffers(1, &ubo_tw);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_tw);
    glBufferData(GL_UNIFORM_BUFFER, sizeof tw, tw, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo_tw);
    glUniformBlockBinding(prog_fft, glGetUniformBlockIndex(prog_fft, "Tw"), 0);
    glUseProgram(prog_red1); glUniform1i(U(prog_red1, "uSrc"), 0); glUniform1i(U(prog_red1, "uV"), 1);
    glUseProgram(prog_red2); glUniform1i(U(prog_red2, "uA"), 0); glUniform1i(U(prog_red2, "uB"), 1);
    glUseProgram(prog_panel);
    glUniform1i(U(prog_panel, "uPsi"), 0); glUniform1i(U(prog_panel, "uV"), 1); glUniform1i(U(prog_panel, "uGlow"), 2);
    glUseProgram(prog_ov); glUniform1i(U(prog_ov, "uAtlas"), 0);
    glGenVertexArrays(1, &vao_ov);
    glGenBuffers(1, &vbo_ov);
    glBindVertexArray(vao_ov);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ov);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, OVF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, OVF * sizeof(float), (void *)(4 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, OVF * sizeof(float), (void *)(8 * sizeof(float)));
    glEnableVertexAttribArray(0); glEnableVertexAttribArray(1); glEnableVertexAttribArray(2);
    hex(P->bg, c_bg); hex(P->ink, c_ink); hex(P->dim, c_dim); hex(P->accent, c_acc); hex(P->relief, c_rel);
    fonts_init();
    atlas_reset();
}

/* ==================================================================== */
/* ---- when the screen was locked ---- */
static time_t lock_start(void) {
    time_t best = 0, btime = 0;
    char line_[256];
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        while (fgets(line_, sizeof line_, f))
            if (sscanf(line_, "btime %ld", &btime) == 1) break;
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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "quantum");
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

/* Spanning, the output under the middle of the layout is the front one and
 * gets the experiment; the others, left to right, take the act's own
 * picture (the carpet, the screen, phase space...) and momentum space in turn. */
static int pick_role(void) {
    if (span_me < 0) return R_FRONT;
    float x0 = 1e9f, x1 = -1e9f;
    for (int i = 0; i < span_n; i++) { x0 = fminf(x0, span[i][0]); x1 = fmaxf(x1, span[i][0] + span[i][2]); }
    float mx = (x0 + x1) / 2;
    span_front = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= mx && mx < span[i][0] + span[i][2]) { span_front = i; break; }
    if (span_me == span_front) return R_FRONT;
    int rank = 0;
    for (int i = 0; i < span_n; i++) {
        if (i == span_front || i == span_me) continue;
        if (span[i][0] < span[span_me][0] || (span[i][0] == span[span_me][0] && (span[i][1] < span[span_me][1] ||
                                               (span[i][1] == span[span_me][1] && i < span_me)))) rank++;
    }
    return rank % 2 ? R_MOMENTUM : R_AUX;
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

static GLuint offscreen_fbo(void) {
    GLuint tex = mk_tex(W, H, GL_RGBA8, GL_RGBA, NULL, 0, 0);
    return mk_fbo(tex, 0);
}

/* run the act's clock from its start to t, measuring as live frames would */
static void run_to(double t) {
    int id, num, var, cyc;
    double lt;
    act_at(t, &id, &num, &var, &cyc, &lt);
    double t0 = t - lt;
    for (double tt = t0; tt < t - 1e-9; tt += 1.0 / 30) {
        advance(tt);
        frame_measure();
    }
    advance(t);
    frame_measure();
}

static void usage(void) {
    fprintf(stderr,
        "usage: quantum [--window | --root | --layer] [--seed N] [--palette NAME] [--act NAME] [--no-caption]\n"
        "               [--epoch UNIXTIME] [--layout 'x,y,w,h;x,y,w,h;...' --output INDEX]\n"
        "               [--shot FILE.ppm | --bench N] [--size WxH] [--at SECONDS] [--check]\n"
        "palettes: phase mono paper ember (default: from the seed)\n"
        "acts: tunnel slit coherent squeezed revival bragg stadium (default: all, in turn)\n");
    exit(1);
}

static int run_check(void);

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-check")) opt_check = 1;
        else if (!strcmp(a, "-no-caption")) opt_nocap = 1;
        else if (!strcmp(a, "-resize-test") && i + 1 < argc) opt_resize_test = atoi(argv[++i]);
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-palette") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NPAL; k++) if (!strcmp(n, PALETTES[k].name)) opt_palette = k;
            if (opt_palette < 0) usage();
        }
        else if (!strcmp(a, "-act") && i + 1 < argc) {
            const char *n = argv[++i];
            for (int k = 0; k < NACT; k++) if (!strcmp(n, act_key[k])) opt_act = k;
            if (opt_act < 0) usage();
        }
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
    if (!opt_seed) opt_seed = (uint32_t)time(NULL) ^ (uint32_t)(now() * 1e6);
    if (!opt_seed) opt_seed = 1;
    P = &PALETTES[opt_palette >= 0 ? opt_palette : (int)(mix32(opt_seed * 2246822519U ^ 0x7a3U) % NPAL)];
    offline = opt_shot || opt_bench > 0 || opt_check;
    async_rd = !opt_shot && !opt_check;
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
        XStoreName(dpy, win, "Quantum");
        XClassHint chint = {"quantum", "Quantum"};
        XSetClassHint(dpy, win, &chint);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offline) XMapRaised(dpy, win);
        GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
        glXMakeCurrent(dpy, win, ctx);
    }
    gl_init();

    if (opt_check) return run_check();
    if (opt_shot) {
        GLuint fbo = offscreen_fbo();
        run_to(opt_at);
        draw_scene(fbo);
        glFinish();
        save_ppm(opt_shot);
        return 0;
    }
    if (opt_bench > 0) {
        GLuint fbo = offscreen_fbo();
        double tw = now();
        run_to(opt_at);
        glFinish();
        double catchup = (now() - tw) * 1e3;
        GLuint q[3];
        glGenQueries(3, q);
        double sum = 0, worst = 0, gsim = 0, gdraw = 0;
        long st0 = steps;
        for (int i = 0; i < opt_bench; i++) {
            double t0 = now();
            glQueryCounter(q[0], GL_TIMESTAMP);
            advance(opt_at + (i + 1) / 60.0);
            frame_measure();
            glQueryCounter(q[1], GL_TIMESTAMP);
            draw_scene(fbo);
            glQueryCounter(q[2], GL_TIMESTAMP);
            glFinish();
            GLuint64 a, b, c;
            glGetQueryObjectui64v(q[0], GL_QUERY_RESULT, &a);
            glGetQueryObjectui64v(q[1], GL_QUERY_RESULT, &b);
            glGetQueryObjectui64v(q[2], GL_QUERY_RESULT, &c);
            gsim += (b - a) * 1e-6; gdraw += (c - b) * 1e-6;
            double ms = (now() - t0) * 1e3;
            sum += ms;
            if (ms > worst) worst = ms;
        }
        printf("quantum: %s, %d frames at %dx%d from %.1f s: avg %.3f ms, worst %.3f ms (catch-up %.0f ms)\n"
               "  GPU per frame: sim + measure %.3f ms (%.2f steps/frame on %dx%d), picture %.3f ms\n  renderer: %s%s\n",
               act_key[A.id], opt_bench, W, H, opt_at, sum / opt_bench, worst, catchup, gsim / opt_bench,
               (double)(steps - st0) / opt_bench, G.nx, G.ny, gdraw / opt_bench,
               (const char *)glGetString(GL_RENDERER), have_exact ? "" : " (no gpu_shader5: plain float products)");
        return 0;
    }

    double t0 = now();
    if (opt_epoch) t0 -= wallclock() - opt_epoch;   /* the clock runs from the shared epoch */
    double cap_t = 0;
    for (long frame = 0;; frame++) {
        if (opt_resize_test && dpy && frame % 12 == 0) {
            /* --resize-test N: N window sizes in turn, then report and quit */
            int k = (int)(frame / 12);
            if (k >= opt_resize_test) {
                char st[256];
                slurp_line("/proc/self/statm", st, sizeof st);
                printf("quantum: %d sizes, now %dx%d, statm %s, %d glyphs cached\n", k, W, H, st, glyphs_used);
                return 0;
            }
            XResizeWindow(dpy, win, 320 + (k * 397) % 1500, 240 + (k * 263) % 1000);
        }
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
        advance(t - t0);
        frame_measure();
        draw_scene(0);
        if (opt_layer) eglSwapBuffers(wl.ed, wl.es);
        else glXSwapBuffers(dpy, win);
        double spare = 1.0 / 60 - (now() - t);
        if (spare > 0) {
            struct timespec ts = {0, (long)(spare * 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}

/* ==================================================================== */
/* ---- --check: the physics, verified ---- */
static double cpu_norm(const float *b) {
    double s = 0;
    for (long i = 0; i < (long)G.nx * G.ny; i++) s += (double)b[2 * i] * b[2 * i] + (double)b[2 * i + 1] * b[2 * i + 1];
    return s * G.dx * G.dy;
}

static int run_check(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("quantum --check (%s; %s)\n", (const char *)glGetString(GL_RENDERER),
           have_exact ? "compensated products" : "plain float products");

    const char *only = getenv("QONLY");
#define SECTION(name) if (!only || strstr(only, name))
    /* 1. norm and energy, no absorbers: the coherent state, 8000 steps */
    SECTION("norm") {
    act_setup(A_COHERENT, 0, 1);
    measure(1);
    double E0c = M.T + M.V, n0 = G.norm0, worst_n = 0, worst_e = 0, worst_x = 0;
    double Ecl = 0.5 * (A.px0 * A.px0 + A.py0 * A.py0) + 0.5 * (A.wx * A.wx * A.x0 * A.x0) + 0.5 * (A.wx + A.wy);
    double tw = now();
    for (int k = 1; k <= 8; k++) {
        for (int s = 0; s < 1000; s++) grid_step();
        steps += 1000;
        float *b = grid_read();
        double n = cpu_norm(b);
        free(b);
        G.kshow = 0;
        measure(1);
        double t = steps * G.dt, E = M.T + M.V;
        double xc = A.x0 * cos(t) + A.px0 * sin(t), yc = A.y0 * cos(A.wy * t) + A.py0 / A.wy * sin(A.wy * t);
        worst_n = fmax(worst_n, fabs(n / n0 - 1));
        worst_e = fmax(worst_e, fabs(E / E0c - 1));
        worst_x = fmax(worst_x, hypot(M.x - xc, M.y - yc));
        printf("  coherent  t = %5.1f  |norm - 1| = %.2e  E = %.6f (exact %.6f)  |E/E0 - 1| = %.1e  |<r> - r_cl| = %.1e\n",
               t, fabs(n / n0 - 1), E, Ecl, fabs(E / E0c - 1), hypot(M.x - xc, M.y - yc));
    }
    double ms = (now() - tw) * 1e3 / 8000;
    printf("  => over 8000 steps (dt %.3g, %d^2): max |norm - 1| %.2e, max |dE/E| %.2e, max centroid error %.2e; %.3f ms/step (incl. reads)\n",
           G.dt, G.nx, worst_n, worst_e, worst_x, ms);

    }
    /* 2. squeezed: sigma_x(t) against the exact breathing */
    SECTION("squeeze") {
    act_setup(A_SQUEEZED, 0, 1);
    double worst_s = 0;
    for (int k = 1; k <= 16; k++) {
        for (int s = 0; s < 40; s++) grid_step();
        steps += 40;
        G.kshow = 0;
        measure(1);
        double t = steps * G.dt, s2 = A.s0 * A.s0 / 2 * cos(t) * cos(t) + 1 / (2 * A.s0 * A.s0) * sin(t) * sin(t);
        double sx = sqrt(fmax(M.xx - M.x * M.x, 0));
        worst_s = fmax(worst_s, fabs(sx / sqrt(s2) - 1));
    }
    printf("  squeezed: sigma_x(t) vs sigma_0^2 cos^2 t + sin^2 t / (4 sigma_0^2): max relative error %.2e over t = 0..%.1f\n",
           worst_s, steps * G.dt);

    }
    /* 3. tunnelling */
    SECTION("tunnel")
    for (int v = 0; v < 3; v++) {
        act_setup(A_TUNNEL, v, 1);
        for (long s = 0; s < A.max_steps; s++) grid_step();
        steps = A.max_steps;
        measure(0);
        double T = M.right * M.n, R = (1 - M.right) * M.n, Tk = T_square(A.k0 * A.k0 / 2, A.V0, A.a);
        printf("  tunnel %d: k0 %.2f V0 %.1f a %.3f (E/V0 %.2f): T = %.4f, R = %.4f, T+R = %.6f | theory, packet-averaged T = %.4f (plane wave %.4f): %+.1f%%\n",
               v, A.k0, A.V0, A.a, A.k0 * A.k0 / 2 / A.V0, T, R, T + R, A.T_th, Tk, (T / A.T_th - 1) * 100);
    }

    /* 4. the double slit: fringe spacing on the screen */
    SECTION("slit")
    for (int v = 0; v < 2; v++) {
        act_setup(A_SLIT, v, 1);
        for (long s = 0; s < A.max_steps; s++) { grid_step(); after_step(); }
        /* the maxima of the accumulated screen intensity, near the middle */
        double pk[64];
        int np = 0;
        for (int j = 2; j < G.ny - 2 && np < 64; j++)
            if (scr_acc[j] > scr_acc[j - 1] && scr_acc[j] >= scr_acc[j + 1] && scr_acc[j] > 0.05 * scr_acc[G.ny / 2]) {
                /* parabolic peak */
                double a = scr_acc[j - 1], b = scr_acc[j], c = scr_acc[j + 1], off = 0.5 * (a - c) / (a - 2 * b + c);
                pk[np++] = YJ(j) + off * G.dy;
            }
        double sp = 0;
        int ns = 0;
        for (int i = 1; i < np; i++)
            if (fabs(pk[i]) < 15 && fabs(pk[i - 1]) < 15) { sp += pk[i] - pk[i - 1]; ns++; }
        sp = ns ? sp / ns : 0;
        double th = A.lam * A.Lscr / A.d;
        /* exactly, for two point sources: the m-th maximum where the paths differ by m lambda */
        double y1 = 0;
        for (double y = 0; y < 30; y += 1e-4) {
            double d1 = hypot(A.Lscr, y + A.d / 2) - hypot(A.Lscr, y - A.d / 2);
            if (d1 >= A.lam) { y1 = y; break; }
        }
        /* and by Huygens-Fresnel for the finite slits (2D Green's function
         * e^ikr / sqrt r, obliquity (1 + cos)/2): first maximum and minimum */
        double hf_max = 0, hf_min = 0, prev = -1, pprev = -1;
        for (double y = 0; y < 20 && !hf_max; y += 0.005) {
            double re = 0, im = 0;
            for (int sl = -1; sl <= 1; sl += 2)
                for (int q = 0; q < 160; q++) {
                    double yy = sl * A.d / 2 - A.w / 2 + A.w * (q + 0.5) / 160, r = hypot(A.Lscr, y - yy);
                    double amp = (1 + A.Lscr / r) / 2 / sqrt(r);
                    re += amp * cos(A.k0 * r); im += amp * sin(A.k0 * r);
                }
            double I = re * re + im * im;
            if (pprev >= 0 && prev < pprev && prev <= I && !hf_min) hf_min = y - 0.005;
            if (pprev >= 0 && prev > pprev && prev >= I && hf_min) hf_max = y - 0.005;
            pprev = prev; prev = I;
        }
        /* the simulated first minimum: the least of the screen intensity between 0 and the first maximum */
        double first_max = 0, first_min = 0, lo = 1e30;
        for (int i = 0; i < np; i++) if (pk[i] > 0.5) { first_max = pk[i]; break; }
        for (int j = 0; j < G.ny; j++)
            if (YJ(j) > 0 && YJ(j) < first_max && scr_acc[j] < lo) { lo = scr_acc[j]; first_min = YJ(j); }
        printf("  slit %d: d %.2f, w %.2f, L %.1f, lambda %.4f: first maximum %.3f, first minimum %.2f | lambda L / d = %.3f (%+.1f%%);"
               " Huygens-Fresnel, finite slits: maximum %.3f (%+.1f%%), minimum %.2f; two point sources: maximum %.3f; mean spacing %.3f; %d particles\n",
               v, A.d, A.w, A.Lscr, A.lam, first_max, first_min, th, (first_max / th - 1) * 100, hf_max, (first_max / hf_max - 1) * 100, hf_min, y1, sp, nclick);
        printf("    peaks:");
        for (int i = 0; i < np; i++) printf(" %.2f", pk[i]);
        printf("\n");
        if (getenv("QPROF"))
            for (int j = 0; j < G.ny; j += 2) printf("    prof %d %.3f %.6g\n", v, YJ(j), scr_acc[j] / scr_acc[G.ny / 2]);
    }

    /* 5. the revival: |<psi0|psi(t)>|^2 about T_rev, from the exact propagator,
     * and the same by stepping the split-step solver (V = 0) */
    SECTION("revival") {
    act_setup(A_REVIVAL, 0, 1);
    float *p0 = grid_read();
    int n = G.nx;
    double fr[] = {0.25, 0.5, 0.97, 0.99, 1.0, 1.01};
    for (int i = 0; i < 6; i++) {
        revival_at(fr[i] * A.trev);
        float *b = grid_read();
        double re = 0, im = 0, nn = 0, n0_ = 0, rm = 0, imm = 0;
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                long a = (long)j * n + k;
                /* the mirror image x -> L - x of psi0 in the quadrant, i.e. -x - L + 2L... use psi0(-x) = -psi0(x) and the 2L period */
                int km = (n - k + n / 2) % n;       /* x -> L - x on the doubled grid */
                int jm = (n - j + n / 2) % n;
                long am = (long)jm * n + km;
                re += (double)p0[2 * a] * b[2 * a] + (double)p0[2 * a + 1] * b[2 * a + 1];
                im += (double)p0[2 * a] * b[2 * a + 1] - (double)p0[2 * a + 1] * b[2 * a];
                rm += (double)p0[2 * am] * b[2 * a] + (double)p0[2 * am + 1] * b[2 * a + 1];
                imm += (double)p0[2 * am] * b[2 * a + 1] - (double)p0[2 * am + 1] * b[2 * a];
                nn += (double)b[2 * a] * b[2 * a] + (double)b[2 * a + 1] * b[2 * a + 1];
                n0_ += (double)p0[2 * a] * p0[2 * a] + (double)p0[2 * a + 1] * p0[2 * a + 1];
            }
        free(b);
        printf("  revival t = %.2f T_rev (T_rev = 4L^2/pi = %.4f): |<psi0|psi>|^2 = %.6f, |<mirror psi0|psi>|^2 = %.6f, norm %.6f\n",
               fr[i], A.trev, (re * re + im * im) / (nn * n0_), (rm * rm + imm * imm) / (nn * n0_), nn / n0_);
    }
    /* by stepping: V = 0, dt = T_rev / 2000 */
    {
        G.dt = A.trev / 2000;
        double *z = calloc((size_t)2 * n * n, sizeof(double));
        for (long a = 0; a < (long)n * n * 2; a++) z[a] = p0[a];
        grid_upload(z, NULL, NULL);
        free(z);
        for (int s = 0; s < 2000; s++) grid_step();
        float *b = grid_read();
        double re = 0, im = 0, nn = 0, n0_ = 0;
        for (long a = 0; a < (long)n * n; a++) {
            re += (double)p0[2 * a] * b[2 * a] + (double)p0[2 * a + 1] * b[2 * a + 1];
            im += (double)p0[2 * a] * b[2 * a + 1] - (double)p0[2 * a + 1] * b[2 * a];
            nn += (double)b[2 * a] * b[2 * a] + (double)b[2 * a + 1] * b[2 * a + 1];
            n0_ += (double)p0[2 * a] * p0[2 * a] + (double)p0[2 * a + 1] * p0[2 * a + 1];
        }
        printf("  revival by 2000 split steps to T_rev: |<psi0|psi>|^2 = %.6f, |norm - 1| = %.1e\n", (re * re + im * im) / (nn * n0_), fabs(nn / n0_ - 1));
        free(b);
    }
    free(p0);
    }

    /* 6. cost per step */
    SECTION("cost") {
    int sizes[][2] = {{256, 256}, {512, 128}, {512, 256}};
    for (int i = 0; i < 3; i++) {
        G.nx = 0;
        geom(sizes[i][0], sizes[i][1], 0.1, 0.1, 0, 0, 0.01);
        packet(0, 0, 2, 2, 1, 0, 1);
        grid_upload(g_psi, NULL, NULL);
        for (int s = 0; s < 50; s++) grid_step();
        glFinish();
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double t0 = now();
            for (int s = 0; s < 200; s++) grid_step();
            glFinish();
            best = fmin(best, (now() - t0) * 1e3 / 200);
        }
        printf("  cost: %dx%d, %.3f ms per step (best of 5 x 200)\n", sizes[i][0], sizes[i][1], best);
    }
    }
    return 0;
}
