/*
 * currents — a moving study of the chrome sphere, violet streamlines and
 * vermilion wake of Tame Impala's Currents cover, designed by Robert Beatty.
 * Original procedural artwork: no cover image, textures or external assets.
 *
 * Analytic cylinder flow bends a fine striped sheet around the sphere.
 * Alternating advecting vortices fold the downstream sheet into a wake.
 * The chrome is a shaded sphere reflecting procedural studio lights and
 * the coloured sheet. Seed + elapsed time fully determine every frame.
 *
 * --layer is native Wayland; --window/--root use GLX. --epoch, --layout and
 * --output draw one continuous picture across the monitor wall. --shot and
 * --bench use an offscreen framebuffer. q / Escape close window mode.
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
#include "proto-layer-shell.h"
#include "../common/hidpi.h"

#define MAX_OUT 8
static int W = 1280, H = 800, opt_root, opt_layer, opt_bench;
static uint32_t opt_seed;
static float opt_fps = 60;
static double opt_epoch, opt_at = 30;
static const char *opt_shot;
static int off_w = 1920, off_h = 1080;
static float span[MAX_OUT][4];
static int span_n, span_me = -1;
static float view[4], wall_size[2], focus[4];
static GLuint prog, vao;

static void die(const char *s) { fprintf(stderr, "currents: %s\n", s); exit(1); }
static double seconds(clockid_t clock) {
    struct timespec t;
    clock_gettime(clock, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void parse_layout(const char *s) {
    for (span_n = 0; span_n < MAX_OUT; span_n++) {
        float *r = span[span_n];
        int used = 0;
        if (sscanf(s, "%f,%f,%f,%f%n", &r[0], &r[1], &r[2], &r[3], &used) != 4 || r[2] <= 0 || r[3] <= 0) break;
        s += used;
        if (!*s) { span_n++; break; }
        if (*s++ != ';') die("invalid layout");
    }
}

static void layout(void) {
    if (span_me < 0) {
        view[0] = view[1] = 0; view[2] = W; view[3] = H;
        wall_size[0] = W; wall_size[1] = H;
        memcpy(focus, view, sizeof focus);
        return;
    }
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (int i = 0; i < span_n; i++) {
        x0 = fminf(x0, span[i][0]); y0 = fminf(y0, span[i][1]);
        x1 = fmaxf(x1, span[i][0] + span[i][2]); y1 = fmaxf(y1, span[i][1] + span[i][3]);
    }
    int mid = 0;
    for (int i = 0; i < span_n; i++)
        if (span[i][0] <= (x0 + x1) * 0.5f && span[i][0] + span[i][2] > (x0 + x1) * 0.5f) { mid = i; break; }
    memcpy(view, span[span_me], sizeof view); view[0] -= x0; view[1] -= y0;
    memcpy(focus, span[mid], sizeof focus); focus[0] -= x0; focus[1] -= y0;
    wall_size[0] = x1 - x0; wall_size[1] = y1 - y0;
}

#define GLSL(...) #__VA_ARGS__
static const char *vs = "#version 330 core\n" GLSL(
    void main() {
        vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
        gl_Position = vec4(p, 0.0, 1.0);
    }
);

static const char *fs = "#version 330 core\n" GLSL(
    uniform vec2 uRes, uWall;
    uniform vec4 uView, uFocus;
    uniform float uTime, uSeed;
    out vec4 o;
    const float TAU = 6.28318530718;
    mat2 rotate(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
    float grain(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

    vec3 chrome(vec3 n, float t) {
        vec3 r = reflect(vec3(0.0, 0.0, -1.0), n);
        vec3 c = mix(vec3(0.06, 0.075, 0.12), vec3(0.37, 0.41, 0.49), smoothstep(-0.6, 0.8, r.y));
        float box = exp(-pow(abs((r.x + 0.34) / 0.52), 6.0) - pow(abs((r.y - 0.48) / 0.29), 6.0));
        float strip = exp(-pow(abs((r.x - 0.65) / 0.12), 4.0) - pow(abs((r.y - 0.2) / 0.75), 6.0));
        float soft = pow(max(dot(r, normalize(vec3(-0.7, 0.75, 0.65))), 0.0), 18.0);
        c += vec3(0.82, 0.87, 0.97) * box + vec3(0.52, 0.57, 0.68) * strip + vec3(0.7) * soft;
        float horizon = exp(-pow((r.y + 0.06) / 0.11, 2.0));
        c *= 1.0 - 0.7 * horizon;
        float reflected = 0.5 + 0.5 * sin(r.x * 60.0 + r.y * 27.0 + t * 0.16);
        c += vec3(0.17, 0.075, 0.24) * reflected * (1.0 - smoothstep(-0.7, 0.0, r.y));
        c += vec3(0.45, 0.045, 0.055) * exp(-pow((r.x + 0.55) / 0.19, 2.0)) * (1.0 - smoothstep(-0.8, 0.4, r.y));
        float fresnel = pow(1.0 - n.z, 4.0);
        return mix(c, vec3(0.56, 0.53, 0.66), fresnel * 0.65);
    }

    void main() {
        vec2 cp = uView.xy + vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y) * uView.zw;
        float t = uTime + uSeed * 23.0;
        float h = uFocus.w;
        vec2 centre = uFocus.xy + uFocus.zw * vec2(0.56, 0.57);
        centre += h * vec2(0.065 * sin(t * 0.033), 0.025 * sin(t * 0.047));
        vec2 p = (cp - centre) / h;
        /* On a wide wall, steer the wake along the screens instead of
         * sending it straight into the gap below the laptop. */
        float angle = (0.82 + 0.055 * sin(t * 0.023)) / max(1.0, uWall.x / h);
        vec2 direction = vec2(cos(angle), sin(angle));
        vec2 crossflow = vec2(-direction.y, direction.x);
        vec2 q = vec2(dot(p, direction), dot(p, crossflow));
        float radius = 0.125;

        /* Inverse vortex maps preserve an unbroken sheet of fine lines.
         * A vortex enters gently at the body and drifts into a wider wake. */
        vec2 z = q;
        float travel = t * 0.055, spacing = 0.30;
        float step = floor(travel / spacing), drift = fract(travel / spacing) * spacing;
        for (int i = 15; i >= 0; i--) {
            float s = 0.13 + float(i) * spacing + drift;
            float side = mod(float(i) - step, 2.0) * 2.0 - 1.0;
            float width = 0.14 + 0.055 * sqrt(s);
            vec2 v = vec2(s, side * (0.08 + 0.045 * sqrt(s)) + 0.025 * sin(s * 3.8 - t * 0.14));
            vec2 d = z - v;
            float strength = 5.8 * smoothstep(0.13, 0.4, s) * exp(-s * 0.14);
            float a = side * strength * exp(-dot(d, d) / (width * width));
            z = v + rotate(a) * d;
        }
        float r2 = max(dot(q, q), radius * radius);
        float stream = z.y * (1.0 - radius * radius / r2);
        stream += 0.008 * sin(z.x * 4.0 + t * 0.13) + 0.003 * sin(z.x * 12.0 - t * 0.09);
        float phase = stream * TAU * 98.0 + t * 0.10;
        float wave = cos(phase), aa = max(fwidth(phase) * 0.48, 0.025);
        float ridge = smoothstep(0.15 - aa, 0.15 + aa, wave);
        float light = clamp(0.45 + cp.x / uWall.x * 0.52 - cp.y / uWall.y * 0.22, 0.0, 1.0);
        vec3 dark = mix(vec3(0.035, 0.026, 0.07), vec3(0.12, 0.11, 0.20), light);
        vec3 lavender = mix(vec3(0.48, 0.32, 0.65), vec3(0.82, 0.73, 0.86), light);
        vec3 c = mix(dark, lavender, ridge);
        c += vec3(0.11, 0.08, 0.14) * pow(max(wave, 0.0), 12.0);

        /* The fine scarlet ribbon upstream broadens into warm pools in
         * the folded downstream sheet, rather than painting over it. */
        float ribbon = 1.0 - smoothstep(0.008, 0.014, abs(stream));
        ribbon *= 1.0 - smoothstep(-0.03, 0.15, q.x);
        vec3 red = mix(vec3(0.77, 0.16, 0.24), vec3(1.0, 0.40, 0.43), ridge * 0.65);
        c = mix(c, red, ribbon);
        float wake = smoothstep(0.05, 0.55, q.x) * exp(-q.x * 0.22) * exp(-pow(z.y / 0.095, 4.0));
        float pool = smoothstep(0.0, 0.55, sin(z.x * 13.0 - t * 0.18)) * wake;
        vec3 warm = mix(vec3(0.87, 0.20, 0.30), vec3(0.95, 0.57, 0.24), 0.5 + 0.5 * sin(z.x * 3.3 + t * 0.05));
        c = mix(c, warm, pool * 0.8);

        vec2 ball = p / radius;
        float d2 = dot(ball, ball);
        float shadow = exp(-dot((p - vec2(0.018, 0.025)) / (radius * 1.12), (p - vec2(0.018, 0.025)) / (radius * 1.12)) * 2.8);
        c *= 1.0 - shadow * 0.48;
        if (d2 < 1.02) {
            vec3 n = normalize(vec3(ball.x, -ball.y, sqrt(max(1.0 - d2, 0.0))));
            float edge = 1.0 - smoothstep(1.0 - fwidth(d2), 1.0 + fwidth(d2), d2);
            c = mix(c, chrome(n, t), edge);
        }
        float vignette = 1.0 - 0.12 * smoothstep(0.3, 1.1, length((cp - uWall * 0.5) / max(uWall.x, uWall.y)));
        c *= vignette;
        c += (grain(cp * 1.7) - 0.5) * 0.023;
        o = vec4(clamp(c, 0.0, 1.0), 1.0);
    }
);

static GLuint shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL); glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof log, NULL, log); die(log); }
    return s;
}

static void gl_init(void) {
    GLuint v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs);
    prog = glCreateProgram(); glAttachShader(prog, v); glAttachShader(prog, f); glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(prog, sizeof log, NULL, log); die(log); }
    glDeleteShader(v); glDeleteShader(f);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
}

static void render(double t, GLuint target) {
    layout();
    glBindFramebuffer(GL_FRAMEBUFFER, target); glViewport(0, 0, W, H); glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "uRes"), W, H);
    glUniform2fv(glGetUniformLocation(prog, "uWall"), 1, wall_size);
    glUniform4fv(glGetUniformLocation(prog, "uView"), 1, view);
    glUniform4fv(glGetUniformLocation(prog, "uFocus"), 1, focus);
    glUniform1f(glGetUniformLocation(prog, "uTime"), (float)t);
    glUniform1f(glGetUniformLocation(prog, "uSeed"), (opt_seed % 65521) / 65521.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

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
    wl.ls = zwlr_layer_shell_v1_get_layer_surface(wl.shell, wl.surf, NULL, ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, "currents");
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



static void usage(int status) {
    fprintf(status ? stderr : stdout,
        "usage: currents [--window | --root | --layer] [--seed N] [--fps N]\n"
        "                [--epoch UNIXTIME] [--layout 'x,y,w,h;...' --output N] [--render-scale F]\n"
        "                [--shot FILE.ppm | --bench N] [--size WxH] [--at SECONDS]\n");
    exit(status);
}

static void save_ppm(const char *path) {
    unsigned char *px = malloc((size_t)W * H * 3);
    if (!px) die("out of memory");
    glPixelStorei(GL_PACK_ALIGNMENT, 1); glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; y--) fwrite(px + (size_t)y * W * 3, 1, (size_t)W * 3, f);
    fclose(f); free(px);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        if (!strcmp(a, "-root")) opt_root = 1;
        else if (!strcmp(a, "-window")) opt_root = opt_layer = 0;
        else if (!strcmp(a, "-layer")) opt_layer = 1;
        else if (!strcmp(a, "-seed") && i + 1 < argc) opt_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "-render-scale") && i + 1 < argc) opt_render_scale = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-fps") && i + 1 < argc) opt_fps = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-epoch") && i + 1 < argc) opt_epoch = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-layout") && i + 1 < argc) parse_layout(argv[++i]);
        else if (!strcmp(a, "-output") && i + 1 < argc) span_me = atoi(argv[++i]);
        else if (!strcmp(a, "-shot") && i + 1 < argc) opt_shot = argv[++i];
        else if (!strcmp(a, "-bench") && i + 1 < argc) opt_bench = atoi(argv[++i]);
        else if (!strcmp(a, "-size") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &off_w, &off_h) != 2 || off_w < 1 || off_h < 1) usage(1);
        }
        else if (!strcmp(a, "-at") && i + 1 < argc) opt_at = strtod(argv[++i], NULL);
        else if (!strcmp(a, "-help")) usage(0);
        else usage(1);
    }
    if (span_me >= span_n || span_me < -1) span_me = -1;
    if (!isfinite(opt_fps) || opt_fps < 1) opt_fps = 60;
    if (!isfinite(opt_at) || !isfinite(opt_epoch) || opt_bench < 0) usage(1);
    if (!opt_seed) opt_seed = (uint32_t)time(NULL);
    int offscreen = opt_shot || opt_bench > 0;
    Display *dpy = NULL;
    Window root = 0, win = 0;
    Atom wm_delete = 0;
    GLXContext ctx = NULL;
    if (opt_layer && !offscreen) wl_init();
    else {
        dpy = XOpenDisplay(NULL);
        if (!dpy) die("can't open display");
        int scr = DefaultScreen(dpy);
        root = RootWindow(dpy, scr);
        int attrs[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
        XVisualInfo *vi = glXChooseVisual(dpy, scr, attrs);
        if (!vi) die("no suitable GLX visual");
        XSetWindowAttributes swa = {0};
        swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
        swa.event_mask = StructureNotifyMask | KeyPressMask | ExposureMask;
        swa.override_redirect = opt_root;
        if (opt_root) {
            XWindowAttributes ra; XGetWindowAttributes(dpy, root, &ra);
            W = ra.width; H = ra.height; XSelectInput(dpy, root, StructureNotifyMask);
        }
        if (offscreen) { W = off_w; H = off_h; }
        win = XCreateWindow(dpy, root, 0, 0, offscreen ? 64 : W, offscreen ? 64 : H, 0, vi->depth, InputOutput,
                            vi->visual, CWColormap | CWEventMask | CWOverrideRedirect, &swa);
        XStoreName(dpy, win, "Currents");
        XClassHint ch = {"currents", "Currents"}; XSetClassHint(dpy, win, &ch);
        wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False); XSetWMProtocols(dpy, win, &wm_delete, 1);
        if (!offscreen) XMapRaised(dpy, win);
        ctx = glXCreateContext(dpy, vi, NULL, True); XFree(vi);
        if (!ctx || !glXMakeCurrent(dpy, win, ctx)) die("can't create GLX context");
    }
    gl_init();
    if (offscreen) {
        GLuint tex, fbo;
        glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("incomplete framebuffer");
        render(opt_at, fbo); glFinish();
        if (opt_shot) save_ppm(opt_shot);
        else {
            double start = seconds(CLOCK_MONOTONIC);
            for (int i = 0; i < opt_bench; i++) { render(opt_at + i / 60.0, fbo); glFinish(); }
            double ms = (seconds(CLOCK_MONOTONIC) - start) * 1000 / opt_bench;
            printf("currents: %dx%d, %d frames: %.3f ms/frame, %.1f fps; %s\n", W, H, opt_bench, ms, 1000 / ms, glGetString(GL_RENDERER));
        }
        glDeleteFramebuffers(1, &fbo); glDeleteTextures(1, &tex);
    } else {
        double start = seconds(CLOCK_MONOTONIC);
        int running = 1;
        while (running) {
            if (opt_layer && !wl_poll()) break;
            while (dpy && XPending(dpy)) {
                XEvent e; XNextEvent(dpy, &e);
                if (e.type == ConfigureNotify) {
                    if (e.xconfigure.window == root) XResizeWindow(dpy, win, e.xconfigure.width, e.xconfigure.height);
                    else { W = e.xconfigure.width; H = e.xconfigure.height; }
                } else if (e.type == KeyPress) {
                    KeySym k = XLookupKeysym(&e.xkey, 0);
                    if (k == XK_q || k == XK_Escape) running = 0;
                } else if (e.type == ClientMessage && (Atom)e.xclient.data.l[0] == wm_delete) running = 0;
            }
            double frame = seconds(CLOCK_MONOTONIC);
            double t = opt_epoch ? seconds(CLOCK_REALTIME) - opt_epoch : frame - start;
            render(fmax(0, t), 0);
            if (opt_layer) eglSwapBuffers(wl.ed, wl.es); else glXSwapBuffers(dpy, win);
            double spare = 1.0 / opt_fps - (seconds(CLOCK_MONOTONIC) - frame);
            if (spare > 0) { struct timespec nap = {0, (long)(spare * 1e9)}; nanosleep(&nap, NULL); }
        }
    }
    glDeleteProgram(prog); glDeleteVertexArrays(1, &vao);
    if (dpy) { glXMakeCurrent(dpy, None, NULL); glXDestroyContext(dpy, ctx); XDestroyWindow(dpy, win); XCloseDisplay(dpy); }
    return 0;
}
