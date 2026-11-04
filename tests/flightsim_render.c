/* Real GL regression checks; the Python wrapper builds and runs this in Xvfb. */
#define main flightsim_program_main
#include "../flightsim/flightsim.c"
#undef main
#include <assert.h>

static void check_history_target(void) {
    GLint sampler, history, attachment;
    glGetUniformiv(p_taa, UL(p_taa, "uHistory"), &sampler);
    glActiveTexture(GL_TEXTURE0 + sampler);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &history);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                         GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &attachment);
    assert(history != attachment);  /* sampling the draw target is undefined */
    glActiveTexture(GL_TEXTURE7);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

static void stripes(int invert) {
    float *px = malloc(sizeof(float) * scene_w * scene_h * 3);
    for (int y = 0; y < scene_h; y++)
        for (int x = 0; x < scene_w; x++)
            for (int k = 0; k < 3; k++)
                px[(y * scene_w + x) * 3 + k] = ((x & 1) ^ invert) ? 0.9f : 0.1f;
    glTextureSubImage2D(tex_scene, 0, 0, 0, scene_w, scene_h, GL_RGB, GL_FLOAT, px);
    free(px);
    float depth = 0;
    glClearTexImage(tex_depth, 0, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
}

static void expect_pixel(float expected) {
    float px[3];
    glReadPixels(32, 24, 1, 1, GL_RGB, GL_FLOAT, px);
    for (int k = 0; k < 3; k++) {
        if (fabsf(px[k] - expected) >= 0.003f)
            fprintf(stderr, "history pixel: got %g, expected %g\n", px[k], expected);
        assert(fabsf(px[k] - expected) < 0.003f);
    }
}

static void check_temporal_history(void) {
    fl.x = fl.z = 0; fl.y = 300;
    fl.yaw = fl.pitch = fl.roll = 0;
    setup_view();
    setup_matrices(0, 0, scene_w, scene_h);
    glDisable(GL_DEPTH_TEST);
    taa_hist = 0;
    stripes(0);
    taa_resolve();
    check_history_target();
    expect_pixel(0.1f);
    GLuint previous = tex_taa[taa_cur];

    stripes(1);
    taa_resolve();
    check_history_target();
    assert(previous == tex_taa[!taa_cur]);
    expect_pixel(0.1f * 0.88f + 0.9f * 0.12f);

    /* Resizing must invalidate history, including on the frame after it. */
    for (int i = 0; i < 4; i++) {
        int w = i & 1 ? 64 : 48, h = i & 1 ? 48 : 32;
        resize_scene(w, h);
        setup_matrices(0, 0, w, h);
        stripes(0);
        taa_resolve();
        check_history_target();
        float valid;
        glGetUniformfv(p_taa, UL(p_taa, "uHistOK"), &valid);
        assert(valid == 0);
        expect_pixel(0.1f);
        taa_resolve();
        check_history_target();
        expect_pixel(0.1f);
    }
}

static void check_stationary_world_stays_stationary(void) {
    /* Sample a fixed world-space ramp through each jittered camera. The
     * resolved image must stay still, even when the history starts fresh. */
    GLuint p = program("stationary world", VS_FULL, GLSL(
        in vec2 vUV;
        uniform vec4 uViewTan;
        uniform vec2 uBase;
        out vec4 oCol;
        void main() {
            float x = (uViewTan.x + vUV.x * uViewTan.z - uBase.x) / uBase.y;
            oCol = vec4(vec3(0.1 + 0.8 * x), 1.0);
        }
    ));
    setup_matrices(0, 0, scene_w, scene_h);
    float base[2] = {view_tan[0], view_tan[2]};
    for (int i = 1; i <= 16; i++) {
        setup_matrices(halton(i % 8 + 1, 2) - 0.5f,
                       halton(i % 8 + 1, 3) - 0.5f, scene_w, scene_h);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
        glViewport(0, 0, scene_w, scene_h);
        glDisable(GL_DEPTH_TEST);
        set_common(p);
        glUniform4fv(UL(p, "uViewTan"), 1, view_tan);
        glUniform2fv(UL(p, "uBase"), 1, base);
        fullscreen();
        if (i == 1 || i == 9) taa_hist = 0;
        taa_resolve();
        float px[3], expected = 0.1f + 0.8f * 32.5f / scene_w;
        glReadPixels(32, 24, 1, 1, GL_RGB, GL_FLOAT, px);
        if (fabsf(px[0] - expected) >= 0.0007f)
            fprintf(stderr, "stationary image moved: got %g, expected %g\n", px[0], expected);
        assert(fabsf(px[0] - expected) < 0.0007f);
    }
    glDeleteProgram(p);
}

static void check_cloud_stability(void) {
    opt_clouds = 1;
    fl.x = fl.z = 0; fl.y = 1200;
    fl.yaw = fl.roll = 0; fl.pitch = 0.55f;
    sim_time = 0;
    float cover = pal.cover;
    pal.cover = 0.7f;
    setup_matrices(0, 0, scene_w, scene_h);
    pix_ang = 1 / cam_f;
    int n = cloud_w * cloud_h * 4;
    float *a = malloc(sizeof(float) * n), *b = malloc(sizeof(float) * n);
    draw_clouds();
    glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, a);
    taa_frame++;
    draw_clouds();
    glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, b);
    double opacity = 0;
    for (int i = 0; i < n; i++) {
        assert(isfinite(a[i]) && fabsf(a[i] - b[i]) < 1e-6f);
        if (i % 4 == 3) opacity += 1 - a[i];
    }
    assert(opacity > 1); /* actually exercised clouds, not empty sky */

    /* Crossing the wrapped world origin must not swap the billow pattern. */
    fl.x = 16384 - 0.01;
    draw_clouds();
    glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, a);
    fl.x = 16384 + 0.01;
    draw_clouds();
    glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, b);
    double error = 0;
    for (int i = 0; i < n; i++) error += fabsf(a[i] - b[i]);
    assert(error / n < 0.001);
    free(a); free(b);
    pal.cover = cover;
    opt_clouds = 0;
}

static void check_camera_between_ticks(void) {
    advance(30);
    double t = steps * STEP;
    render_at(t, fbo_final);
    Flight state = fl;
    render_at(t + STEP * 0.25, fbo_final);
    float delta[3];
    glGetUniformfv(p_taa, UL(p_taa, "uCamDelta"), delta);
    assert(fabsf(delta[0] - (state.x - previous_fl.x) * 0.25) < 1e-5);
    assert(fabsf(delta[2] - (state.z - previous_fl.z) * 0.25) < 1e-5);
    assert(hypotf(delta[0], delta[2]) > 0.1f);
    assert(fl.x == state.x && fl.y == state.y && fl.z == state.z);
    assert(fl.yaw == state.yaw && fl.pitch == state.pitch && fl.roll == state.roll);
}

static void check_cloud_sampling_quality(void) {
    /* The old 48-sample march was repeatable but visibly striped. Compare
     * the production pass with dense quadrature, including a half cloud-pixel
     * camera movement that exposes flicker from under-sampled layers. */
    int old_w = W, old_h = H;
    W = 640; H = 480;
    resize_scene(W, H);
    opt_time = "golden";
    choose_palette();
    opt_clouds = 1;
    fl.x = fl.z = 0; fl.y = 700;
    fl.yaw = 100 / DEG; fl.roll = 0;
    sim_time = 0;
    setup_view();
    pix_ang = 1 / cam_f;
    GLuint ref = program("dense cloud reference", VS_FULL, GLSL(
        in vec2 vUV;
        uniform vec4 uViewTan; uniform mat3 uCamRot;
        out vec4 oCol;
        void main() {
            vec3 rd = normalize(uCamRot * vec3(uViewTan.xy + vUV * uViewTan.zw, -1.0));
            oCol = clouds(rd, 1e9, 4.0);
        }
    ));
    int n = cloud_w * cloud_h * 4;
    float *actual = malloc(sizeof(float) * n), *dense = malloc(sizeof(float) * n);
    float *previous_error = calloc(n, sizeof(float));
    for (int frame = 0; frame < 3; frame++) {
        fl.pitch = 0.15f + frame / cam_f;
        setup_matrices(0, 0, W, H);
        glDisable(GL_DEPTH_TEST);
        draw_clouds();
        glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, actual);
        set_common(ref);
        glUniform4fv(UL(ref, "uViewTan"), 1, view_tan);
        glUniformMatrix3fv(UL(ref, "uCamRot"), 1, GL_FALSE, rot3);
        glUniform1f(UL(ref, "uPixAng"), pix_ang * 2);
        fullscreen();
        glReadPixels(0, 0, cloud_w, cloud_h, GL_RGBA, GL_FLOAT, dense);
        double error = 0, shimmer = 0, opacity = 0;
        for (int i = 0; i < n; i++) {
            float e = actual[i] - dense[i];
            error += fabsf(e);
            shimmer += fabsf(e - previous_error[i]);
            previous_error[i] = e;
            if (i % 4 == 3) opacity += 1 - dense[i];
        }
        error /= n; shimmer /= n;
        if (error >= 0.002 || (frame && shimmer >= 0.002))
            fprintf(stderr, "cloud sampling: error %g, shimmer %g\n", error, shimmer);
        assert(opacity > cloud_w * cloud_h * 0.05);
        assert(error < 0.002);
        assert(!frame || shimmer < 0.002);
    }
    free(actual); free(dense); free(previous_error);
    glDeleteProgram(ref);
    opt_clouds = 0;
    W = old_w; H = old_h;
    resize_scene(W, H);
    setup_view();
}

static void check_water_depth(void) {
    float underwater[2] = {-50, 0};
    for (int i = 0; i < 2; i++) {
        glClearTexImage(hm[i].tex, 0, GL_RG, GL_FLOAT, underwater);
        glGenerateTextureMipmap(hm[i].tex);
    }
    fl.y = 50; fl.pitch = -0.3f;
    setup_matrices(0, 0, scene_w, scene_h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glClearDepth(0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    draw_water();
    float depth;
    glReadPixels(32, 8, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    /* Intersect this pixel's camera ray with y=0; reversed-Z is near / z. */
    float vx = view_tan[0] + (32.5f / scene_w) * view_tan[2];
    float vy = view_tan[1] + (8.5f / scene_h) * view_tan[3];
    float ray_y = rot3[1] * vx + rot3[4] * vy - rot3[7];
    float expected = 1.5f * -ray_y / fl.y;
    assert(expected > 0 && fabsf(depth - expected) < 1e-5f);
}

static void check_deep_mist(void) {
    /* Deep valleys used to overflow the mist antiderivative's exponential,
     * producing NaNs even though that branch should contribute zero. */
    GLuint p = program("mist regression", VS_FULL, GLSL(
        out vec4 oCol;
        void main() { oCol = vec4(vec3(mistDepth(vec3(0.0, -1.0, 0.0), 2000.0)), 1.0); }
    ));
    set_common(p);
    glUniform1f(UL(p, "uMistTop0"), 2500);
    glUniform1f(UL(p, "uMist"), 0.001f);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glDisable(GL_DEPTH_TEST);
    fullscreen();
    float px[3];
    glReadPixels(32, 24, 1, 1, GL_RGB, GL_FLOAT, px);
    for (int k = 0; k < 3; k++) assert(isfinite(px[k]) && px[k] >= 0);
    glDeleteProgram(p);
}

static void check_flight_and_resolution_changes(void) {
    opt_trees = opt_clouds = 1;
    advance(30);
    for (int frame = 0; frame < 16; frame++) {
        render_scale = frame < 4 || frame >= 12 ? 1.0f : frame < 8 ? 0.5f : 0.75f;
        advance(30 + frame / 60.0);
        render(fbo_final);
        float *px = malloc(sizeof(float) * scene_w * scene_h * 3);
        glGetTextureImage(tex_scene, 0, GL_RGB, GL_FLOAT,
                          sizeof(float) * scene_w * scene_h * 3, px);
        for (int i = 0; i < scene_w * scene_h * 3; i++)
            assert(isfinite(px[i]) && px[i] >= 0);
        free(px);
        assert(glGetError() == GL_NO_ERROR);
    }
    opt_trees = opt_clouds = 0;
    glDisable(GL_DEPTH_TEST);
}

static void check_portrait_hud(void) {
    W = 270; H = 480;
    fl.pitch = 20 / DEG; fl.roll = 0;
    setup_view();
    GLuint tex = mk_tex2d(GL_RGBA8, W, H, GL_RGBA, GL_UNSIGNED_BYTE,
                          NULL, GL_NEAREST, GL_CLAMP_TO_EDGE);
    GLuint fbo = mk_fbo(tex);
    glViewport(0, 0, W, H);
    glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
    float black[4] = {0};
    glClearBufferfv(GL_COLOR, 0, black);
    draw_hud();
    unsigned char pixels[270 * 17 * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, H / 2 - 8, W, 17, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    int left = 0, right = 0;
    for (int y = 0; y < 17; y++)
        for (int x = 0; x < W; x++) {
            const unsigned char *p = pixels + (y * W + x) * 3;
            if (p[1] > 128 && p[0] < 128) {
                left += x < W / 4;
                right += x > W * 3 / 4;
            }
        }
    assert(left > 20 && right > 20); /* both readout boxes actually appear */
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

int main(void) {
    char *args[] = {"flightsim-test", "--bench", "1", "--size", "64x48",
                    "--seed", "42", "--time", "day", "--no-hud",
                    "--no-trees", "--no-clouds"};
    assert(flightsim_program_main(sizeof args / sizeof args[0], args) == 0);
    prof_on = 0;
    check_camera_between_ticks();
    check_flight_and_resolution_changes();
    check_temporal_history();
    check_stationary_world_stays_stationary();
    check_cloud_stability();
    check_cloud_sampling_quality();
    check_water_depth();
    check_deep_mist();
    check_portrait_hud();
    assert(glGetError() == GL_NO_ERROR);
    puts("Flightsim: motion, blending, jitter, resizing, water depth, mist and portrait HUD pass");
    return 0;
}
