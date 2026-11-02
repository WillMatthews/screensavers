/* steam: smoke, steam and sparks, as lit particles. Included once, by steam.c.
 *
 * Each exhaust beat throws a puff out of the chimney at the speed of the
 * blast; the air drags it to rest and the train runs out from under it, so
 * the plume streams back over the train and hangs in the air behind. Hot
 * smoke rises; steam (from the safety valves when she blows off, the
 * cylinder drain cocks when she's working hard, the whistle) is whiter and
 * thins faster; sparks fly up from the chimney and fall. Lit on the CPU by
 * the point lights near each (the fire's glow underneath, station lamps),
 * shaded on the GPU as a soft lumpy ball under the moon; drawn at half
 * resolution, softened against the scene's depth, then laid over it. */

#define MAXPUFF 1400
typedef struct {
    double p[3];
    float v[3];
    float age, life, r0, r1, a0, seed, dark;
    int kind;                  /* 0 smoke, 1 steam, 2 spark */
} Puff;
static Puff puffs[MAXPUFF];
static int npuff;
static double beats_done;
static uint32_t srng = 12345;
static float srand01(void) { srng ^= srng << 13; srng ^= srng >> 17; srng ^= srng << 5; return (srng >> 8) * (1.0f / 16777216.0f); }

static struct {
    double blow_until, blow_next;     /* the safety valves */
    double whistle_until, whistle_next;
    double cocks_until;
    float spark_acc, steam_acc;
} sm;

static Puff *puff_new(void) {
    if (npuff < MAXPUFF) return &puffs[npuff++];
    /* full: replace the oldest-looking */
    int best = 0;
    float ba = -1;
    for (int i = 0; i < MAXPUFF; i += 7)
        if (puffs[i].age / puffs[i].life > ba) { ba = puffs[i].age / puffs[i].life; best = i; }
    return &puffs[best];
}

static void emit(const Frame *f, float x, float y, float z, V3 vloc, float spread, int kind, float r0, float r1, float life, float a0) {
    Puff *q = puff_new();
    fpt(f, x, y, z, q->p);
    V3 vw = vadd(vadd(vmul(f->X, vloc.x), vmul(f->Y, vloc.y)), vmul(f->Z, vloc.z));
    vw = vadd(vw, vmul(f->Z, (float)tr.v));
    vw = vadd(vw, v3((srand01() - 0.5f) * spread, (srand01() - 0.5f) * spread, (srand01() - 0.5f) * spread));
    q->v[0] = vw.x; q->v[1] = vw.y; q->v[2] = vw.z;
    q->age = 0; q->life = life * (0.8f + 0.4f * srand01());
    q->r0 = r0; q->r1 = r1 * (0.8f + 0.4f * srand01());
    q->a0 = a0; q->seed = srand01() * 100; q->kind = kind;
    q->dark = clampf((tr.temp_s - 40) / 50, 0, 1);
}

/* whistle: called by the director and the lineside, a long or short blast */
static void whistle(double t, double len) { if (t > sm.whistle_until + 3) sm.whistle_until = t + len; }

static void smoke_step(double dt) {
    double t = sim_t;
    Frame f = veh_frame(0);
    /* the beats */
    while (beats_done + 1 <= tr.beats) {
        beats_done += 1;
        float work = clampf(0.4f + tr.cutoff + 0.5f * tr.load_s, 0.3f, 1.4f);
        int n = tr.v > 12 ? 1 : 2;
        for (int i = 0; i < n; i++)
            emit(&f, 0, 4.15f, 6.15f, v3(0, 7 + 7 * work, 0), 1.6f, 0, 0.4f, 5.5f + 3.0f * work, 9.0f + 4 * work, 0.5f + 0.2f * work);
    }
    if (beats_done < tr.beats - 50) beats_done = floor(tr.beats);
    /* sparks, more when she's worked hard and the fire's hot */
    sm.spark_acc += (float)dt * (2 + 30 * tr.load_s * tr.load_s) * tr.fire;
    while (sm.spark_acc >= 1) {
        sm.spark_acc -= 1;
        emit(&f, (srand01() - 0.5f) * 0.3f, 4.2f, 6.15f, v3(0, 7 + 6 * srand01(), 0), 3.0f, 2, 0.03f, 0.03f, 0.7f + 0.8f * srand01(), 1);
    }
    /* the safety valves blow off now and then, and when the regulator's eased (the load fell) */
    if (t > sm.blow_next || (tr.vt < tr.v - 4 && t > sm.blow_until + 20)) {
        sm.blow_until = t + 4 + 5 * srand01();
        sm.blow_next = t + 70 + 120 * srand01();
    }
    /* the drain cocks open when she's pulling away hard */
    if (tr.vt > tr.v + 3 && t > sm.cocks_until + 10) sm.cocks_until = t + 5;
    /* the whistle now and then */
    if (t > sm.whistle_next) {
        whistle(t, 0.8 + 1.5 * srand01());
        sm.whistle_next = t + 60 + 90 * srand01();
    }
    sm.steam_acc += (float)dt * 40;
    while (sm.steam_acc >= 1) {
        sm.steam_acc -= 1;
        if (t < sm.blow_until)
            for (int s = -1; s <= 1; s += 2)
                emit(&f, s * 0.18f, 4.12f, -1.6f, v3(0, 16, 0), 1.2f, 1, 0.12f, 2.6f, 2.6f, 0.5f);
        if (t < sm.whistle_until)
            emit(&f, 0.3f, 4.12f, -2.95f, v3(0, 10, 2), 1.0f, 1, 0.1f, 1.8f, 1.8f, 0.45f);
        if (t < sm.cocks_until && srand01() < 0.6f)
            for (int s = -1; s <= 1; s += 2)
                emit(&f, s * 1.1f, 0.4f, 5.4f, v3(s * 5.0f, -0.5f, 3), 1.0f, 1, 0.12f, 1.6f, 1.5f, 0.5f);
    }
    /* move and age them */
    float wind[3] = {1.2f, 0, 0.4f};
    for (int i = 0; i < npuff; i++) {
        Puff *q = &puffs[i];
        q->age += (float)dt;
        if (q->age >= q->life) { *q = puffs[--npuff]; i--; continue; }
        float tau = q->kind == 2 ? 1.6f : q->kind == 1 ? 0.45f : 0.3f + 0.6f * q->age;
        float k = (float)dt / tau;
        if (k > 1) k = 1;
        for (int c = 0; c < 3; c++) q->v[c] += (wind[c] - q->v[c]) * k;
        q->v[1] += (q->kind == 2 ? -5.0f : q->kind == 0 ? 1.4f : 0.8f) * (float)dt;
        for (int c = 0; c < 3; c++) q->p[c] += q->v[c] * dt;
        /* don't sink into the ground */
        if (q->kind != 2 && q->age > 0.5f) {
            double gy = track_y(q->p[2]) + 0.5;
            if (q->p[1] < gy) q->p[1] = gy;
        }
    }
}

/* ---- drawing ---- */
static GLuint p_smoke, p_smokecomp, vao_smoke, vbo_smoke;
static GLuint fbo_smoke, tex_smoke;
static int smoke_w, smoke_h;
static float *sbuf;
static int *sorder;
static float *sdist;

static void smoke_init(void) {
    glGenVertexArrays(1, &vao_smoke);
    glGenBuffers(1, &vbo_smoke);
    glBindVertexArray(vao_smoke);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_smoke);
    for (int i = 0; i < 3; i++) {
        glVertexAttribPointer((GLuint)i, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void *)(i * 4 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)i);
        glVertexAttribDivisor((GLuint)i, 1);
    }
    sbuf = malloc(sizeof(float) * 12 * MAXPUFF);
    sorder = malloc(sizeof(int) * MAXPUFF);
    sdist = malloc(sizeof(float) * MAXPUFF);
    srng = opt_seed | 1;
    sm.blow_next = 20 + 40 * srand01();
    sm.whistle_next = 8 + 20 * srand01();
    if (getenv("STEAM_BLOW")) {   /* testing: the safety valves and the whistle going from the start */
        sm.blow_until = 1e9; sm.whistle_until = 1e9; sm.blow_next = 1e9;
    }
}
static void smoke_targets(int sw, int sh) {
    int div = sw >= 1600 ? 3 : 2;      /* soft stuff: a third of the resolution is plenty on a big screen */
    smoke_w = (sw + div - 1) / div; smoke_h = (sh + div - 1) / div;
    tex_smoke = mk_tex(smoke_w, smoke_h, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, GL_LINEAR);
    fbo_smoke = mk_fbo(tex_smoke, 0);
}
static void smoke_targets_free(void) {
    glDeleteFramebuffers(1, &fbo_smoke);
    glDeleteTextures(1, &tex_smoke);
}
static int sort_far(const void *a, const void *b) {
    float da = sdist[*(const int *)a], db = sdist[*(const int *)b];
    return da < db ? 1 : da > db ? -1 : 0;
}

static void smoke_draw(void) {
    int n = 0;
    for (int i = 0; i < npuff; i++) {
        Puff *q = &puffs[i];
        float x = (float)(q->p[0] - cam.p[0]), y = (float)(q->p[1] - cam.p[1]), z = (float)(q->p[2] - cam.p[2]);
        float d = vdot(v3(x, y, z), cam.F);
        float u = q->age / q->life;
        float r = q->r0 + (q->r1 - q->r0) * sqrtf(u);
        if (d < -r || x * x + y * y + z * z > 1500.0f * 1500.0f) continue;
        sdist[i] = d;
        sorder[n++] = i;
    }
    qsort(sorder, (size_t)n, sizeof *sorder, sort_far);
    for (int j = 0; j < n; j++) {
        Puff *q = &puffs[sorder[j]];
        float *o = sbuf + j * 12;
        float x = (float)(q->p[0] - cam.p[0]), y = (float)(q->p[1] - cam.p[1]), z = (float)(q->p[2] - cam.p[2]);
        float u = q->age / q->life;
        float r = q->r0 + (q->r1 - q->r0) * sqrtf(u);
        float a = q->a0 * powf(1 - u, 1.6f) * sstepf(0.0f, 0.08f, q->age);
        if (q->kind == 1) a *= sstepf(0.0f, 0.12f, q->age);        /* steam's clear for its first inches */
        /* light from the point lights, soft over the puff's size */
        float lr = 0, lg = 0, lb = 0;
        for (int k = 0; k < npl; k++) {
            float dx = pl[k][0] - x, dy = pl[k][1] - y, dz = pl[k][2] - z;
            float d2 = dx * dx + dy * dy + dz * dz;
            float rad = pl[k][3] * 1.4f + r * 0.4f;
            float att = 1 / (1 + d2 / (rad * rad));
            att *= att;
            lr += plc[k][0] * att; lg += plc[k][1] * att; lb += plc[k][2] * att;
        }
        /* the carriages' windows, on steam beside the train */
        for (int k = 0; k < ncar_u; k++) {
            float dx = x - car_u[k][0], dz = z - car_u[k][2];
            float lz = dx * car_d[k][0] + dz * car_d[k][1];
            float lx = dx * car_d[k][1] - dz * car_d[k][0];
            if (fabsf(lz) > car_u[k][3] + r || fabsf(lx) > 8 + r) continue;
            float side = fmaxf(fabsf(lx) - 1.4f, 0);
            float g = car_d[k][2] * 0.35f / (1 + side * side * 0.3f);
            lr += 1.0f * g; lg += 0.6f * g; lb += 0.3f * g;
        }
        o[0] = x; o[1] = y; o[2] = z; o[3] = r;
        o[4] = a; o[5] = (float)q->kind; o[6] = q->seed; o[7] = q->dark;
        o[8] = lr; o[9] = lg; o[10] = lb; o[11] = u;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_smoke);
    glViewport(0, 0, smoke_w, smoke_h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    if (n) {
        glBindVertexArray(vao_smoke);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_smoke);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12 * (size_t)n, sbuf, GL_STREAM_DRAW);
        set_common(p_smoke);
        glUniform3f(UL(p_smoke, "uCamR"), cam.R.x, cam.R.y, cam.R.z);
        glUniform3f(UL(p_smoke, "uCamU"), cam.U.x, cam.U.y, cam.U.z);
        glUniform3f(UL(p_smoke, "uCamF"), cam.F.x, cam.F.y, cam.F.z);
        glUniform1i(UL(p_smoke, "uDepth"), 8);
        glUniform2f(UL(p_smoke, "uDepthSize"), (float)scene_w, (float)scene_h);
        glUniform2f(UL(p_smoke, "uSmokeSize"), (float)smoke_w, (float)smoke_h);
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D, tex_depth);
        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n);
        glDisable(GL_BLEND);
    }
    /* lay it over the scene */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_scene);
    glViewport(0, 0, scene_w, scene_h);
    glUseProgram(p_smokecomp);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_smoke);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(UL(p_smokecomp, "uSrc"), 5);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    fullscreen_tri();
    glDisable(GL_BLEND);
}
