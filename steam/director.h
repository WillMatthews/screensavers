/* steam: the camera's shots. Included once, by steam.c.
 *
 * Long takes, chosen by what the train is about to run through: a viaduct,
 * a station, a tunnel or a level crossing coming up gets its own set-up,
 * placed in the world for the train to arrive into; otherwise a rotation
 * of shots that ride with the train. Dips to black between them. All of it
 * a function of the simulation, so spanned instances agree. */

enum { SH_WHEELS, SH_FRONT, SH_PASSBY, SH_AERIAL, SH_CAB, SH_SIDE, SH_VIADUCT, SH_STATION, SH_TUNNEL, SH_CROSSING, NSHOT };
static const char *shot_name[NSHOT] = {"wheels", "front", "passby", "aerial", "cab", "side", "viaduct", "station", "tunnel", "crossing"};
static struct {
    int kind, n;
    double t0, t1;
    float side;
    double a[3];               /* a world anchor (where a fixed camera stands) */
    double fz0, fz1;           /* the set piece's extent */
    double look[3], lookv[3];  /* the look-at point, on a spring */
    int look_init;
} shot;
#define FADE 0.9

static uint32_t dhash(int n, int salt) {
    uint32_t h = (uint32_t)n * 0x9e3779b9u ^ (uint32_t)salt * 0x85ebca6bu ^ opt_seed * 0xc2b2ae35u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return h;
}
static float dfrand(int n, int salt) { return (dhash(n, salt) & 0xffffff) / (float)0x1000000; }

/* the front of the engine, and the tail of the train, along the line */
static double head_z(void) { return tr.z + 7.4; }

/* the look-at point a shot wants now */
static void shot_target(double t, double *out) {
    Frame f0 = veh_frame(0);
    switch (shot.kind) {
    case SH_PASSBY:
    case SH_STATION:
    case SH_CROSSING: {
        /* the engine's front until she's past, then along the train, then her tail going away */
        double cz = shot.a[2];
        double hz = head_z();
        if (hz < cz + 6) { fpt(&f0, 0, 2.6f, 6.0f, out); break; }
        double tz = fmax(train_tail_z(), cz - 30 - (hz - cz) * 0.05);
        if (tz > hz) tz = hz;
        line_pt(tz, 0, 2.4, out);
        break;
    }
    case SH_VIADUCT: {
        double hz = head_z();
        double mz = fmin(fmax(hz - 30, shot.fz0 - 40), shot.fz1 + 60);
        line_pt(mz, 0, 2.0, out);
        break;
    }
    case SH_TUNNEL:
        fpt(&f0, 0, 3.0f, -10.0f, out);
        if (head_z() > shot.fz0) line_pt(shot.fz0 + 4, 0, 3.0, out);
        break;
    default:
        fpt(&f0, 0, 2.5f, 0, out);
    }
    (void)t;
}

static double shot_len(int kind, double t, int n) {
    (void)t;
    switch (kind) {
    case SH_VIADUCT: case SH_STATION: case SH_TUNNEL: case SH_CROSSING: return 34;
    default: return 22 + 12 * dfrand(n, 3);
    }
}

/* choose the next shot at time t */
static void shot_pick(double t) {
    int n = shot.n + 1;
    int prev = shot.kind;
    double v = fmax(tr.v, 8);
    int kind = -1;
    /* a set piece coming up? */
    struct { int f, k; double lo, hi; } want[] = {
        {F_VIADUCT, SH_VIADUCT, 8, 24}, {F_STATION, SH_STATION, 7, 20}, {F_TUNNEL, SH_TUNNEL, 9, 22}, {F_CROSSING, SH_CROSSING, 7, 18},
    };
    for (size_t i = 0; i < sizeof want / sizeof *want && kind < 0; i++) {
        Feat f = feat_near(head_z() + v * 15, want[i].f);
        double dt = (f.z0 - head_z()) / v;
        if (f.on && dt > want[i].lo && dt < want[i].hi && want[i].k != prev) {
            kind = want[i].k;
            shot.fz0 = f.z0; shot.fz1 = f.z1;
        }
    }
    if (kind < 0) {
        static const int rot[] = {SH_WHEELS, SH_FRONT, SH_AERIAL, SH_PASSBY, SH_CAB, SH_SIDE, SH_FRONT, SH_WHEELS, SH_PASSBY, SH_AERIAL};
        int r = (int)(dhash(n, 1) % (sizeof rot / sizeof *rot));
        kind = rot[r];
        if (kind == prev) kind = rot[(r + 1) % (sizeof rot / sizeof *rot)];
    }
    if (opt_view >= 0 && opt_view < NSHOT) {
        kind = opt_view;
        if (kind >= SH_VIADUCT) {
            static const int fk[] = {F_VIADUCT, F_STATION, F_TUNNEL, F_CROSSING};
            Feat f = feat_near(head_z() + v * 14, fk[kind - SH_VIADUCT]);
            shot.fz0 = f.z0; shot.fz1 = f.z1;
        }
    }
    shot.kind = kind;
    shot.n = n;
    shot.t0 = t;
    shot.t1 = t + shot_len(kind, t, n);
    shot.side = dfrand(n, 2) < 0.5f ? -1.0f : 1.0f;
    shot.look_init = 0;
    /* where fixed cameras stand */
    switch (kind) {
    case SH_PASSBY: {
        double z = head_z() + v * (8 + 4 * dfrand(n, 4));
        line_pt(z, shot.side * (4.2 + 2 * dfrand(n, 5)), 0.6 + 1.2 * dfrand(n, 6), shot.a);
        break;
    }
    case SH_STATION:
        line_pt(shot.fz1 - 40, -pole_side * 3.4, 0.92 + 1.65, shot.a);
        break;
    case SH_CROSSING:
        line_pt((shot.fz0 + shot.fz1) / 2 - 3.5, pole_side * 15.5, 1.4, shot.a);
        break;
    case SH_VIADUCT: {
        double mz = (shot.fz0 + shot.fz1) / 2 + (dfrand(n, 7) - 0.5) * 80;
        line_pt(mz, shot.side * (95 + 40 * dfrand(n, 8)), -22, shot.a);
        break;
    }
    default: break;
    }
}

static void director_init(double t) {
    shot.n = (int)(dhash(0, 9) % 1000);
    shot.kind = -1;
    shot_pick(t);
    /* a run started mid-way begins its first shot in the clear */
    shot.t0 = t - FADE;
}

static void director_update(double t) {
    if (t >= shot.t1) shot_pick(t);
    /* fixed-camera shots end early once the train has gone */
    if ((shot.kind == SH_PASSBY || shot.kind == SH_STATION || shot.kind == SH_CROSSING) && t < shot.t1 - FADE &&
        train_tail_z() > shot.a[2] + 60 && t - shot.t0 > 8)
        shot.t1 = t + FADE;
    if (shot.kind == SH_VIADUCT && train_tail_z() > shot.fz1 + 40 && t < shot.t1 - FADE) shot.t1 = t + FADE;
    if (shot.kind == SH_TUNNEL && train_tail_z() > shot.fz0 + 30 && t < shot.t1 - FADE) shot.t1 = t + FADE + 1.5;
    /* the look-at point follows its target on a critically damped spring */
    double tg[3];
    shot_target(t, tg);
    if (!shot.look_init) {
        for (int k = 0; k < 3; k++) { shot.look[k] = tg[k]; shot.lookv[k] = 0; }
        shot.look_init = 1;
    }
    double w = 2.2, h = 1.0 / 120;
    for (int k = 0; k < 3; k++) {
        double a = w * w * (tg[k] - shot.look[k]) - 2 * w * shot.lookv[k];
        shot.lookv[k] += a * h;
        shot.look[k] += shot.lookv[k] * h;
    }
}

static float director_fade(double t) {
    float a = clampf((float)((t - shot.t0) / FADE), 0, 1), b = clampf((float)((shot.t1 - t) / FADE), 0, 1);
    a = a * a * (3 - 2 * a); b = b * b * (3 - 2 * b);
    return fminf(a, b);
}
/* may the train change length now? only while the screen is dark */
static int director_offscreen_tail(void) { return director_fade(sim_t) < 0.02f; }

/* a slow wander, for a hand on the camera */
static float wander(double t, float f, int salt) {
    return sinf((float)(t * f) + salt * 1.7f) * 0.6f + sinf((float)(t * f * 2.31) + salt * 4.1f) * 0.4f;
}

static void director_camera(double t) {
    double u = clampf((float)((t - shot.t0) / (shot.t1 - shot.t0)), 0, 1);
    float s = shot.side;
    Frame f0 = veh_frame(0);
    double p[3], at[3];
    float roll = 0;
    cam.fstop = 0;
    switch (shot.kind) {
    case SH_WHEELS: {
        float z = (float)(4.5 - 6.5 * u);
        fpt(&f0, s * (3.3f + 0.05f * wander(t, 0.7f, 1)), 1.05f + 0.04f * wander(t, 0.9f, 2), z, p);
        fpt(&f0, s * 0.9f, 1.2f, z - 3.5f, at);
        cam.hfov = 62; cam.fstop = 1.2f;
        roll = 0.01f * wander(t, 0.5f, 3);
        break;
    }
    case SH_FRONT: {
        float d = (float)(14 - 3 * u);
        fpt(&f0, s * (2.0f + 1.4f * (float)u), 0.55f + 0.5f * (float)u, d, p);
        fpt(&f0, s * 0.3f, 2.5f, 3.5f, at);
        cam.hfov = 48; cam.fstop = 0.8f;
        roll = -s * 0.03f;
        break;
    }
    case SH_AERIAL: {
        fpt(&f0, s * (40 + 10 * (float)u), 26 + 6 * (float)u, 35 - 70 * (float)u, p);
        fpt(&f0, 0, 3, -20 - 10 * (float)u, at);
        cam.hfov = 42;
        break;
    }
    case SH_CAB: {
        fpt(&f0, s * 1.78f, 3.45f + 0.015f * wander(t, 3.1f, 4), -3.9f, p);
        fpt(&f0, s * 1.15f, 3.0f, 40, at);
        cam.hfov = 64; cam.fstop = 0.6f;
        roll = 0.012f * wander(t, 1.3f, 5);
        break;
    }
    case SH_SIDE: {
        float z = (float)(-14 + 22 * u);
        fpt(&f0, s * 26, 3.0f, z, p);
        fpt(&f0, 0, 2.0f, z + 1, at);
        cam.hfov = 30;
        break;
    }
    case SH_PASSBY:
    case SH_STATION:
    case SH_CROSSING:
        for (int k = 0; k < 3; k++) { p[k] = shot.a[k]; at[k] = shot.look[k]; }
        p[1] += 0.02 * wander(t, 0.6f, 6);
        cam.hfov = shot.kind == SH_CROSSING ? 52 : 50; cam.fstop = 1.0f;
        break;
    case SH_VIADUCT:
        for (int k = 0; k < 3; k++) { p[k] = shot.a[k]; at[k] = shot.look[k]; }
        p[1] += 30 * u;                /* the crane rises */
        cam.hfov = 46;
        break;
    case SH_TUNNEL: {
        /* chase behind the train until near the portal, then stop */
        double cz = fmin(train_tail_z() - 26, shot.fz0 - 38);
        line_pt(cz, s * 3.2, 6.5, p);
        for (int k = 0; k < 3; k++) at[k] = shot.look[k];
        cam.hfov = 48;
        break;
    }
    default:
        fpt(&f0, 10, 3, 0, p);
        fpt(&f0, 0, 2, 0, at);
        cam.hfov = 50;
    }
    /* never below the ground */
    double g = terrain_h(p[0], p[2]);
    if (shot.kind != SH_WHEELS && shot.kind != SH_FRONT && shot.kind != SH_CAB && p[1] < g + 0.5) p[1] = g + 0.5;
    cam_look(p, at, roll);
    if (cam.fstop == 0) cam.focus = 1e4f;
}
