/* steam: the train, built from lathe-turned shapes, boxes and rods.
 * Included once, by steam.c.
 *
 * An express engine of no railway in particular: a 4-6-2 with 2 m driving
 * wheels, outside cylinders and Walschaerts valve gear, smoke deflectors,
 * a lined deep green with brass and copper, a red buffer beam; a six-wheeled
 * tender heaped with coal; and varnished teak carriages with their windows
 * lit. Local frames: x right (facing forward), y up from the rail top, z
 * forward; the engine's origin is under its middle driving axle. */

/* the palette: the shader's materials, by index */
enum {
    PM_TERRAIN, PM_GREEN, PM_BLACK, PM_RED, PM_BRASS, PM_COPPER, PM_STEEL, PM_LINE,
    PM_WHEEL_D, PM_TYRE, PM_CABIN, PM_GLASS, PM_WINDOW, PM_COAL, PM_TEAK, PM_ROOF,
    PM_RAIL, PM_SLEEPER, PM_BALLAST, PM_STONE, PM_COPING, PM_LAMP, PM_SIG_R, PM_SIG_G,
    PM_WHITE, PM_SIGN, PM_POLE, PM_IRON, PM_FIRE, PM_CARPAINT, PM_HEAD, PM_TAIL,
    PM_BRICK, PM_SLATE, PM_BWIN, PM_PLATE, PM_WHEEL_B, PM_WHEEL_T, PM_WHEEL_X, PM_SMOKEBOX,
    PM_HEADBOARD, PM_GRIME, PM_SIGN2, PM_FLAME, PM_NAME, NPM
};
static float pal[48][4], pal2[48][4];
static void pal_set(int i, float r, float g, float b, float rough, float metal, float emis, float kind, float w) {
    pal[i][0] = r; pal[i][1] = g; pal[i][2] = b; pal[i][3] = rough;
    pal2[i][0] = metal; pal2[i][1] = emis; pal2[i][2] = kind; pal2[i][3] = w;
}
static void palette_init(void) {
    pal_set(PM_TERRAIN, 0.1f, 0.1f, 0.1f, 0.9f, 0, 0, 0, 0);
    pal_set(PM_GREEN, 0.035f, 0.11f, 0.06f, 0.32f, 0, 0, 0, 0);
    pal_set(PM_BLACK, 0.025f, 0.025f, 0.025f, 0.55f, 0, 0, 0, 0);
    pal_set(PM_RED, 0.2f, 0.022f, 0.016f, 0.4f, 0, 0, 0, 0);
    pal_set(PM_BRASS, 0.78f, 0.56f, 0.24f, 0.22f, 1, 0, 0, 0);
    pal_set(PM_COPPER, 0.8f, 0.42f, 0.27f, 0.25f, 1, 0, 0, 0);
    pal_set(PM_STEEL, 0.6f, 0.6f, 0.62f, 0.22f, 1, 0, 7, 0);
    pal_set(PM_LINE, 0.55f, 0.28f, 0.06f, 0.4f, 0, 0, 0, 0);
    pal_set(PM_WHEEL_D, 0.035f, 0.11f, 0.06f, 0.4f, 0, 0, 1, 20);
    pal_set(PM_WHEEL_B, 0.035f, 0.11f, 0.06f, 0.4f, 0, 0, 1, 112);
    pal_set(PM_WHEEL_T, 0.035f, 0.11f, 0.06f, 0.4f, 0, 0, 1, 214);
    pal_set(PM_WHEEL_X, 0.03f, 0.03f, 0.03f, 0.5f, 0, 0, 1, 310);
    pal_set(PM_TYRE, 0.5f, 0.5f, 0.5f, 0.3f, 1, 0, 7, 0);
    pal_set(PM_CABIN, 0.1f, 0.085f, 0.06f, 0.6f, 0, 0, 0, 0);
    pal_set(PM_GLASS, 0.01f, 0.012f, 0.015f, 0.04f, 0, 0, 0, 0);
    pal_set(PM_WINDOW, 1, 1, 1, 0.05f, 0, 0.7f, 2, 0);
    pal_set(PM_COAL, 0.03f, 0.03f, 0.03f, 0.55f, 0, 0, 5, 0);
    pal_set(PM_TEAK, 0.2f, 0.085f, 0.035f, 0.35f, 0, 0, 0, 0);
    pal_set(PM_ROOF, 0.12f, 0.12f, 0.12f, 0.7f, 0, 0, 0, 0);
    pal_set(PM_RAIL, 0.45f, 0.42f, 0.4f, 0.25f, 1, 0, 7, 0);
    pal_set(PM_SLEEPER, 0.05f, 0.04f, 0.03f, 0.9f, 0, 0, 0, 0);
    pal_set(PM_BALLAST, 0.13f, 0.12f, 0.11f, 0.95f, 0, 0, 0, 0);
    pal_set(PM_STONE, 0.22f, 0.2f, 0.17f, 0.85f, 0, 0, 6, 0);
    pal_set(PM_COPING, 0.32f, 0.3f, 0.27f, 0.8f, 0, 0, 0, 0);
    pal_set(PM_LAMP, 1.0f, 0.72f, 0.4f, 0.2f, 0, 30.0f, 4, 0);
    pal_set(PM_SIG_R, 1.0f, 0.08f, 0.03f, 0.2f, 0, 40.0f, 4, 0);
    pal_set(PM_SIG_G, 0.25f, 1.0f, 0.55f, 0.2f, 0, 40.0f, 4, 0);
    pal_set(PM_WHITE, 0.6f, 0.6f, 0.58f, 0.5f, 0, 0, 0, 0);
    pal_set(PM_SIGN, 0.03f, 0.04f, 0.07f, 0.5f, 0, 0, 3, 3);
    pal_set(PM_SIGN2, 0.02f, 0.035f, 0.08f, 0.5f, 0, 0, 3, 4);
    pal_set(PM_POLE, 0.07f, 0.055f, 0.04f, 0.9f, 0, 0, 0, 0);
    pal_set(PM_IRON, 0.04f, 0.04f, 0.045f, 0.5f, 0.6f, 0, 0, 0);
    pal_set(PM_FIRE, 1.0f, 0.45f, 0.12f, 0.5f, 0, 25.0f, 4, 0);
    pal_set(PM_CARPAINT, 0.12f, 0.02f, 0.02f, 0.25f, 0, 0, 0, 0);
    pal_set(PM_HEAD, 1.0f, 0.8f, 0.5f, 0.1f, 0, 3.0f, 4, 0);
    pal_set(PM_TAIL, 1.0f, 0.05f, 0.02f, 0.1f, 0, 30.0f, 4, 0);
    pal_set(PM_BRICK, 0.12f, 0.06f, 0.04f, 0.9f, 0, 0, 6, 0);
    pal_set(PM_SLATE, 0.05f, 0.055f, 0.06f, 0.6f, 0, 0, 0, 0);
    pal_set(PM_BWIN, 1.0f, 0.7f, 0.38f, 0.2f, 0, 6.0f, 4, 0);
    pal_set(PM_PLATE, 0.02f, 0.02f, 0.02f, 0.4f, 0, 0, 3, 10);
    pal_set(PM_NAME, 0.22f, 0.02f, 0.015f, 0.4f, 0, 0, 3, 11);
    pal_set(PM_SMOKEBOX, 0.02f, 0.02f, 0.021f, 0.45f, 0, 0, 0, 0);
    pal_set(PM_HEADBOARD, 0.015f, 0.03f, 0.07f, 0.4f, 0, 0, 3, 2);
    pal_set(PM_GRIME, 0.03f, 0.028f, 0.025f, 0.8f, 0, 0, 0, 0);
    pal_set(PM_FLAME, 1.0f, 0.55f, 0.2f, 0.5f, 0, 8.0f, 4, 0);
}

static void mat(int m) { mb->mat = (float)m; }
static void push(M4 m) { mb->xf = m4mul(mb->xf, m); }
/* a lathe about the z axis placed with its axis along a direction */
static void lathe_at(V3 o, V3 axis, const float *rz, int np, int n) {
    V3 Z = vnorm(axis);
    V3 X = fabsf(Z.y) < 0.9f ? vnorm(vcross(v3(0, 1, 0), Z)) : vnorm(vcross(v3(1, 0, 0), Z));
    V3 Y = vcross(Z, X);
    M4 s = mb->xf;
    push(m4basis(X, Y, Z, o));
    mb_lathe(rz, np, n, 0, 2 * (float)M_PI);
    mb->xf = s;
}
static void cyl_at(V3 o, V3 axis, float r, float len, int n) {
    float pr[12] = {0, 0, r, 0, r, 0, r, len, r, len, 0, len};
    lathe_at(o, axis, pr, 6, n);
}
static void boxc(float cx, float cy, float cz, float hx, float hy, float hz) { mb_box(v3(cx - hx, cy - hy, cz - hz), v3(cx + hx, cy + hy, cz + hz)); }
/* a flat disc facing +x (or -x) with uv in units of its radius, for the wheel shader */
static void disc_x(float x, float cy, float cz, float R, float facing, int n) {
    V3 nn = v3(facing, 0, 0);
    for (int i = 0; i < n; i++) {
        float a0 = 2 * (float)M_PI * i / n, a1 = 2 * (float)M_PI * (i + 1) / n;
        V3 c = v3(x, cy, cz), A = v3(x, cy + R * sinf(a0), cz + R * cosf(a0)), B = v3(x, cy + R * sinf(a1), cz + R * cosf(a1));
        if (facing > 0) { mb_vert(c, nn, 0, 0); mb_vert(A, nn, cosf(a0), sinf(a0)); mb_vert(B, nn, cosf(a1), sinf(a1)); }
        else { mb_vert(c, nn, 0, 0); mb_vert(B, nn, cosf(a1), sinf(a1)); mb_vert(A, nn, cosf(a0), sinf(a0)); }
    }
}
/* a wheel at (x side, axle height R, z): tyre with flange, spoked centre, boss */
static void wheel(float side, float R, float z, int discmat) {
    float xo = side * 0.80f;           /* the outer face */
    float x_in = side * 0.66f;
    /* tyre: a lathe about x */
    float pr[] = {R - 0.09f, 0, R, 0, R, 0, R, 0.13f, R + 0.035f, 0.13f, R + 0.035f, 0.15f, R - 0.09f, 0.15f, R - 0.09f, 0};
    mat(PM_TYRE);
    lathe_at(v3(xo, R, z), v3(-side, 0, 0), pr, 8, 40);
    /* the spoked centre, both faces */
    mat(discmat);
    disc_x(xo - side * 0.005f, R, z, R - 0.06f, side, 32);
    disc_x(x_in, R, z, R - 0.06f, -side, 24);
    /* axle */
    mat(PM_BLACK);
    cyl_at(v3(x_in, R, z), v3(-side, 0, 0), 0.11f, 0.66f, 12);
}
static void handrail(V3 a, V3 b, int posts) {
    mat(PM_STEEL);
    mb_rod(a, b, 0.016f, 6, 0);
    for (int i = 0; i <= posts; i++) {
        V3 p = vlerp(a, b, (float)i / posts);
        V3 dir = vnorm(v3(p.x, 0, 0));
        mb_rod(p, vsub(p, vmul(dir, 0.08f)), 0.012f, 5, 0);
    }
}

static Mesh m_loco, m_tender, m_coach, m_rod, m_link, m_xhead, m_pin;
static GMesh g_loco, g_tender, g_coach, g_rod, g_link, g_xhead, g_pin;
static GMesh g_loco_w, g_tender_w, g_coach_w;      /* their wheels' spoked centres, drawn blended */
static int is_wheel(int m) { return pal2[m][2] > 0.5f && pal2[m][2] < 1.5f; }

#define BOILER_Y 2.95f
#define DRV_R 1.0f
#define CRANK_R 0.36f
#define CYL_Y 1.0f
#define CON_L 3.2f
static const float drv_z[3] = {2.15f, 0.0f, -2.15f};

static void build_loco(void) {
    mb_begin(&m_loco);
    /* frames and the stretcher at the front */
    mat(PM_BLACK);
    for (int s = -1; s <= 1; s += 2) boxc(s * 0.6f, 1.15f, 1.5f, 0.025f, 0.45f, 5.9f);
    boxc(0, 1.3f, 6.2f, 0.6f, 0.25f, 0.3f);
    /* the running plate, high over the wheels, dropping to the buffer beam at the front */
    mat(PM_BLACK);
    boxc(0, 2.12f, 1.9f, 1.36f, 0.02f, 4.55f);
    for (int s = -1; s <= 1; s += 2) {
        mat(PM_GREEN);
        boxc(s * 1.34f, 2.0f, 1.9f, 0.02f, 0.1f, 4.55f);       /* valance */
        mat(PM_LINE);
        boxc(s * 1.362f, 1.99f, 1.9f, 0.003f, 0.012f, 4.5f);
        /* the drop: a quarter curve down to the beam */
        mat(PM_GREEN);
        for (int i = 0; i < 8; i++) {
            float a0 = (float)M_PI / 2 * i / 8, a1 = (float)M_PI / 2 * (i + 1) / 8;
            float z0 = 6.45f + 0.75f * sinf(a0), y0 = 1.32f + 0.8f * cosf(a0);
            float z1 = 6.45f + 0.75f * sinf(a1), y1 = 1.32f + 0.8f * cosf(a1);
            mb_quad(v3(s * 1.36f, y0, z0), v3(s * 1.36f, y1, z1), v3(s * 1.36f, y1 - 0.2f, z1), v3(s * 1.36f, y0 - 0.2f, z0));
            if (s > 0) {
                mat(PM_BLACK);
                mb_quad(v3(-1.36f, y0 + 0.02f, z0), v3(1.36f, y0 + 0.02f, z0), v3(1.36f, y1 + 0.02f, z1), v3(-1.36f, y1 + 0.02f, z1));
                mat(PM_GREEN);
            }
        }
    }
    /* buffer beam, buffers, coupling, lamp irons and the headlamps */
    mat(PM_RED);
    boxc(0, 1.08f, 7.25f, 1.36f, 0.25f, 0.06f);
    for (int s = -1; s <= 1; s += 2) {
        mat(PM_BLACK);
        cyl_at(v3(s * 0.87f, 1.06f, 7.3f), v3(0, 0, 1), 0.12f, 0.32f, 16);
        mat(PM_STEEL);
        float bh[] = {0, 0.32f, 0.21f, 0.32f, 0.22f, 0.36f, 0.2f, 0.4f, 0, 0.41f};
        lathe_at(v3(s * 0.87f, 1.06f, 7.3f), v3(0, 0, 1), bh, 5, 24);
    }
    mat(PM_IRON);
    boxc(0, 1.0f, 7.42f, 0.04f, 0.1f, 0.12f);
    /* two lamps over the buffers: express headcode */
    for (int s = -1; s <= 1; s += 2) {
        mat(PM_WHITE);
        boxc(s * 0.87f, 1.5f, 7.28f, 0.13f, 0.17f, 0.1f);
        mat(PM_BLACK);
        boxc(s * 0.87f, 1.7f, 7.28f, 0.09f, 0.03f, 0.08f);
        mat(PM_HEAD);
        float lp[] = {0, 0, 0.085f, 0, 0.085f, 0.012f, 0, 0.02f};
        lathe_at(v3(s * 0.87f, 1.48f, 7.38f), v3(0, 0, 1), lp, 4, 16);
    }
    /* cylinders, valve chests, slide bars */
    for (int s = -1; s <= 1; s += 2) {
        float cx = s * 1.06f;
        mat(PM_BLACK);
        float cy[] = {0, 3.95f, 0.3f, 3.95f, 0.36f, 3.98f, 0.36f, 4.06f, 0.33f, 4.06f, 0.33f, 5.14f, 0.36f, 5.14f, 0.36f, 5.22f, 0.3f, 5.25f, 0, 5.25f};
        lathe_at(v3(cx, CYL_Y, 0), v3(0, 0, 1), cy, 10, 28);
        float vc[] = {0, 3.9f, 0.18f, 3.9f, 0.21f, 3.93f, 0.21f, 5.3f, 0.18f, 5.33f, 0, 5.33f};
        lathe_at(v3(cx, 1.55f, 0), v3(0, 0, 1), vc, 6, 20);
        boxc(cx, 1.28f, 4.6f, 0.2f, 0.18f, 0.5f);
        /* cylinder covers, polished */
        mat(PM_STEEL);
        float cc[] = {0, 5.25f, 0.28f, 5.25f, 0.3f, 5.21f, 0.3f, 5.2f};
        lathe_at(v3(cx, CYL_Y, 0), v3(0, 0, 1), cc, 4, 28);
        float cc2[] = {0, 5.33f, 0.16f, 5.33f, 0.18f, 5.31f};
        lathe_at(v3(cx, 1.55f, 0), v3(0, 0, 1), cc2, 3, 20);
        /* slide bars and their bracket */
        mat(PM_STEEL);
        boxc(cx, CYL_Y + 0.15f, 3.2f, 0.05f, 0.025f, 0.8f);
        boxc(cx, CYL_Y - 0.15f, 3.2f, 0.05f, 0.025f, 0.8f);
        mat(PM_BLACK);
        boxc(s * 0.85f, 1.4f, 2.45f, 0.24f, 0.55f, 0.04f);       /* motion bracket */
        boxc(cx - s * 0.05f, 1.55f, 1.55f, 0.06f, 0.12f, 0.12f); /* expansion link bracket */
        /* drain cock pipes under the cylinders */
        mat(PM_COPPER);
        mb_rod(v3(cx, CYL_Y - 0.3f, 4.2f), v3(cx + s * 0.05f, 0.45f, 4.6f), 0.02f, 6, 0);
        mb_rod(v3(cx, CYL_Y - 0.3f, 5.1f), v3(cx + s * 0.05f, 0.45f, 5.4f), 0.02f, 6, 0);
    }
    /* wheels: bogie, drivers, trailing */
    for (int s = -1; s <= 1; s += 2) {
        wheel(s, 0.48f, 6.3f, PM_WHEEL_B);
        wheel(s, 0.48f, 4.9f, PM_WHEEL_B);
        for (int i = 0; i < 3; i++) wheel(s, DRV_R, drv_z[i], PM_WHEEL_D);
        wheel(s, 0.56f, -4.15f, PM_WHEEL_T);
        /* bogie frame, axleboxes, the trailing truck's */
        mat(PM_BLACK);
        boxc(s * 0.98f, 0.55f, 5.6f, 0.04f, 0.17f, 1.0f);
        boxc(s * 1.0f, 0.5f, 6.3f, 0.06f, 0.12f, 0.14f);
        boxc(s * 1.0f, 0.5f, 4.9f, 0.06f, 0.12f, 0.14f);
        boxc(s * 0.98f, 0.62f, -4.15f, 0.05f, 0.18f, 0.25f);
        /* splasher-less: the driving wheels' axleboxes and springs behind them */
        for (int i = 0; i < 3; i++) {
            boxc(s * 0.62f, 1.0f, drv_z[i], 0.06f, 0.16f, 0.15f);
            mat(PM_IRON);
            boxc(s * 0.62f, 0.72f, drv_z[i], 0.05f, 0.05f, 0.5f);
            mat(PM_BLACK);
        }
        /* sandboxes, steps, guard irons */
        boxc(s * 1.0f, 0.3f, 7.0f, 0.03f, 0.25f, 0.06f);
        mat(PM_STEEL);
        boxc(s * 1.3f, 0.7f, -3.2f, 0.12f, 0.015f, 0.12f);
        boxc(s * 1.3f, 1.2f, -3.2f, 0.12f, 0.015f, 0.12f);
    }
    /* the boiler: smokebox, lagged barrel with bands, firebox */
    mat(PM_SMOKEBOX);
    float sb[] = {0, 6.95f, 0.75f, 6.95f, 0.8f, 6.9f, 0.84f, 6.85f, 0.9f, 6.8f, 0.9f, 5.3f, 0.86f, 5.3f};
    lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), sb, 7, 48);
    /* the door: dished, with its hinges and dart handles */
    float dr[] = {0, 7.06f, 0.3f, 7.04f, 0.6f, 6.99f, 0.76f, 6.96f, 0.8f, 6.93f};
    lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), dr, 5, 48);
    /* its rim, the hinge straps across it, the dart: a small handwheel and a handle */
    mat(PM_SMOKEBOX);
    {
        M4 sv = mb->xf;
        push(m4trans(0, BOILER_Y, 6.95f));
        mb_torus(0.79f, 0.035f, 48, 8);
        mb->xf = sv;
    }
    mat(PM_STEEL);
    for (int i = -1; i <= 1; i += 2) boxc(-0.25f, BOILER_Y + i * 0.42f, 6.98f + 0.02f * (1 - i * i), 0.5f, 0.028f, 0.018f);
    for (int i = -1; i <= 1; i += 2) cyl_at(v3(-0.78f, BOILER_Y + i * 0.42f, 6.95f), v3(0, 1, 0), 0.035f, 0.1f, 8);
    cyl_at(v3(0, BOILER_Y, 7.03f), v3(0, 0, 1), 0.04f, 0.1f, 12);
    {
        M4 sv = mb->xf;
        push(m4trans(0, BOILER_Y, 7.12f));
        mb_torus(0.09f, 0.012f, 18, 5);
        mb->xf = sv;
    }
    mb_rod(v3(0, BOILER_Y, 7.13f), v3(0.2f, BOILER_Y - 0.06f, 7.15f), 0.012f, 6, 1);
    /* the smokebox saddle under it, and the top lamp iron */
    mat(PM_SMOKEBOX);
    boxc(0, BOILER_Y - 0.75f, 5.95f, 0.62f, 0.35f, 0.6f);
    /* the number plate on the door, and the headboard on the top lamp iron */
    mat(PM_PLATE);
    mb_quad_uv(v3(-0.32f, BOILER_Y + 0.5f, 7.0f), v3(0.32f, BOILER_Y + 0.5f, 7.0f), v3(0.32f, BOILER_Y + 0.64f, 6.985f), v3(-0.32f, BOILER_Y + 0.64f, 6.985f), 0, 1, 1, 0);
    mat(PM_HEADBOARD);
    {
        /* a curved-topped board, held out on the lamp iron */
        float y0 = 1.95f, y1 = 2.42f, z = 7.12f;
        int n = 16;
        for (int i = 0; i < n; i++) {
            float u0 = (float)i / n, u1 = (float)(i + 1) / n;
            float x0 = -0.75f + 1.5f * u0, x1 = -0.75f + 1.5f * u1;
            float t0 = y1 + 0.08f * sinf(u0 * (float)M_PI), t1 = y1 + 0.08f * sinf(u1 * (float)M_PI);
            mb_quad_uv(v3(x0, y0, z), v3(x1, y0, z), v3(x1, t1, z), v3(x0, t0, z), u0, 1, u1, (y1 + 0.08f - t0) / (y1 + 0.08f - y0) * 0 + 0);
        }
        /* fix the uvs: v from 1 at the bottom to 0 at the top of the arch */
        float *v = m_loco.v + m_loco.n - n * 6 * VF;
        for (int i = 0; i < n * 6; i++) {
            float yy = v[i * VF + 1];
            v[i * VF + 7] = 1 - (yy - y0) / (y1 + 0.08f - y0);
        }
        mat(PM_BLACK);
        boxc(0, (y0 + y1) / 2 + 0.03f, z - 0.03f, 0.77f, (y1 - y0) / 2 + 0.07f, 0.02f);
    }
    /* the barrel and its bands */
    mat(PM_GREEN);
    float br[] = {0.86f, 5.3f, 0.84f, 5.3f, 0.84f, -0.7f};
    lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), br, 3, 56);
    for (int i = 0; i < 5; i++) {
        float z = 5.0f - i * 1.35f;
        mat(PM_BLACK);
        float bd[] = {0.84f, z - 0.05f, 0.852f, z - 0.05f, 0.852f, z + 0.05f, 0.84f, z + 0.05f};
        lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), bd, 4, 56);
        mat(PM_LINE);
        float bl[] = {0.84f, z - 0.065f, 0.846f, z - 0.065f, 0.846f, z - 0.055f, 0.84f, z - 0.055f};
        lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), bl, 4, 56);
        float bl2[] = {0.84f, z + 0.055f, 0.846f, z + 0.055f, 0.846f, z + 0.065f, 0.84f, z + 0.065f};
        lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), bl2, 4, 56);
    }
    /* the firebox: a round top over straight sides down between the frames */
    mat(PM_GREEN);
    float fb[] = {0.9f, -0.7f, 0.9f, -3.25f};
    lathe_at(v3(0, BOILER_Y, 0), v3(0, 0, 1), fb, 2, 56);
    for (int s = -1; s <= 1; s += 2) {
        mb_quad(v3(s * 0.9f, BOILER_Y, -3.25f * (s > 0 ? 1 : 1)), v3(s * 0.9f, BOILER_Y, -0.7f), v3(s * 0.9f, 2.12f, -0.7f), v3(s * 0.9f, 2.12f, -3.25f));
    }
    /* chimney, dome, safety valves, whistle */
    mat(PM_BLACK);
    float ch[] = {0.42f, 0.0f, 0.36f, 0.08f, 0.33f, 0.2f, 0.32f, 0.6f, 0.34f, 0.72f, 0.4f, 0.76f, 0.4f, 0.8f, 0.3f, 0.82f, 0.28f, 0.7f};
    lathe_at(v3(0, BOILER_Y + 0.55f, 6.15f), v3(0, 1, 0), ch, 9, 32);
    mat(PM_SMOKEBOX);
    float chb[] = {0.62f, -0.15f, 0.5f, 0.0f, 0.42f, 0.05f};
    lathe_at(v3(0, BOILER_Y + 0.55f, 6.15f), v3(0, 1, 0), chb, 3, 32);
    mat(PM_BRASS);
    float dm[] = {0.55f, -0.2f, 0.5f, 0.05f, 0.42f, 0.12f, 0.4f, 0.4f, 0.36f, 0.5f, 0.24f, 0.58f, 0.0f, 0.62f};
    lathe_at(v3(0, BOILER_Y + 0.62f, 2.4f), v3(0, 1, 0), dm, 7, 36);
    float sv[] = {0.12f, -0.1f, 0.12f, 0.05f, 0.1f, 0.08f, 0.1f, 0.22f, 0.13f, 0.25f, 0.13f, 0.28f, 0.05f, 0.32f, 0, 0.33f};
    lathe_at(v3(-0.18f, BOILER_Y + 0.83f, -1.6f), v3(0, 1, 0), sv, 8, 20);
    lathe_at(v3(0.18f, BOILER_Y + 0.83f, -1.6f), v3(0, 1, 0), sv, 8, 20);
    float wh[] = {0.02f, 0, 0.02f, 0.15f, 0.05f, 0.16f, 0.055f, 0.32f, 0.04f, 0.36f, 0.0f, 0.37f};
    lathe_at(v3(0.3f, BOILER_Y + 0.8f, -2.95f), v3(0, 1, 0), wh, 6, 16);
    /* handrails along the boiler, and the smoke deflectors */
    for (int s = -1; s <= 1; s += 2) {
        handrail(v3(s * 0.95f, BOILER_Y + 0.35f, -0.8f), v3(s * 0.95f, BOILER_Y + 0.35f, 5.2f), 5);
        mat(PM_GREEN);
        float x = s * 1.28f;
        float pts[][2] = {{4.9f, 2.14f}, {7.0f, 2.14f}, {6.95f, 3.4f}, {5.3f, 3.62f}, {4.9f, 3.4f}};
        for (int k = 1; k + 1 < 5; k++)
            mb_quad(v3(x, pts[0][1], pts[0][0]), v3(x, pts[k][1], pts[k][0]), v3(x, pts[k + 1][1], pts[k + 1][0]), v3(x, pts[k + 1][1], pts[k + 1][0]));
        /* with a thickness: the inner face */
        for (int k = 1; k + 1 < 5; k++)
            mb_quad(v3(x - s * 0.02f, pts[0][1], pts[0][0]), v3(x - s * 0.02f, pts[k + 1][1], pts[k + 1][0]), v3(x - s * 0.02f, pts[k][1], pts[k][0]), v3(x - s * 0.02f, pts[k][1], pts[k][0]));
        mat(PM_LINE);
        mb_rod(v3(x + s * 0.004f, 3.3f, 5.1f), v3(x + s * 0.004f, 3.3f, 6.7f), 0.006f, 4, 0);
        /* the name plate over the middle driver: brass, the host's name */
        mat(PM_NAME);
        mb_quad_uv(v3(s * 1.37f, 2.18f, -1.0f * s + 0.0f), v3(s * 1.37f, 2.18f, 1.0f * s), v3(s * 1.37f, 2.42f, 1.0f * s), v3(s * 1.37f, 2.42f, -1.0f * s), 1, 1, 0, 0);
    }
    /* pipes: the ejector along the boiler, the injectors' feeds under the cab */
    mat(PM_COPPER);
    mb_rod(v3(1.0f, BOILER_Y + 0.1f, -2.9f), v3(1.0f, BOILER_Y + 0.1f, 5.4f), 0.04f, 8, 0);
    mb_rod(v3(-0.95f, 2.3f, -2.9f), v3(-0.95f, 2.3f, 5.3f), 0.025f, 6, 0);
    /* the cab */
    float cz0 = -3.25f, cz1 = -5.55f, cy0 = 2.12f, cyw = 3.25f, cyr = 3.55f, cw = 1.36f;
    for (int s = -1; s <= 1; s += 2) {
        float x = s * cw;
        mat(PM_GREEN);
        /* the side sheet below the opening, and the pillars, with the opening's cut-out */
        mb_quad(v3(x, cy0 - 0.05f, cz0), v3(x, cy0 - 0.05f, cz1), v3(x, 2.8f, cz1), v3(x, 2.8f, cz0));
        /* the front of the side: a window between two pillars; the rear pillar */
        boxc(x, (2.8f + cyr) / 2, cz0 - 0.07f, 0.025f, (cyr - 2.8f) / 2, 0.07f);
        boxc(x, (2.8f + cyr) / 2, cz0 - 0.75f, 0.025f, (cyr - 2.8f) / 2, 0.06f);
        boxc(x, 2.84f, cz0 - 0.41f, 0.025f, 0.04f, 0.3f);
        boxc(x, cyr - 0.03f, cz0 - 0.41f, 0.025f, 0.05f, 0.3f);
        boxc(x, (2.8f + cyr) / 2, cz1 + 0.1f, 0.025f, (cyr - 2.8f) / 2, 0.1f);
        mat(PM_GLASS);
        boxc(x, (2.88f + cyr - 0.08f) / 2, cz0 - 0.41f, 0.006f, (cyr - 0.08f - 2.88f) / 2, 0.28f);
        mat(PM_GREEN);
        /* the cut-out's rounded edge: a beading along the top of the side sheet */
        mat(PM_BLACK);
        mb_rod(v3(x + s * 0.01f, 2.8f, cz0 - 0.82f), v3(x + s * 0.01f, 2.8f, cz1 + 0.2f), 0.025f, 6, 1);
        mat(PM_GREEN);
        /* the cab-side window, glazed, in the front pillar's upper part */
        mat(PM_LINE);
        boxc(x + s * 0.003f, 2.78f, (cz0 + cz1) / 2, 0.003f, 0.012f, (cz0 - cz1) / 2 - 0.03f);
        /* inside: the side sheet lined in cream */
        mat(PM_CABIN);
        boxc(x - s * 0.03f, (cy0 + 2.8f) / 2, (cz0 + cz1) / 2, 0.01f, (2.8f - cy0) / 2, (cz0 - cz1) / 2);
        /* cab floor */
        mat(PM_POLE);
        boxc(0, cy0 - 0.25f, (cz0 + cz1) / 2, cw, 0.02f, (cz0 - cz1) / 2);
    }
    /* the front of the cab: the spectacle plate round the firebox */
    mat(PM_GREEN);
    {
        int n = 24;
        for (int i = 0; i < n; i++) {
            float a0 = (float)M_PI * i / n, a1 = (float)M_PI * (i + 1) / n;
            /* between the firebox's top (r 0.9) and the roof line */
            V3 A = v3(0.92f * cosf(a0), BOILER_Y + 0.92f * sinf(a0), cz0), B = v3(0.92f * cosf(a1), BOILER_Y + 0.92f * sinf(a1), cz0);
            float rx0 = cw * cosf(a0), rx1 = cw * cosf(a1);
            float ry0 = cyr + 0.38f * (1 - (rx0 / cw) * (rx0 / cw)), ry1 = cyr + 0.38f * (1 - (rx1 / cw) * (rx1 / cw));
            V3 C = v3(rx1, ry1, cz0), D = v3(rx0, ry0, cz0);
            mb_quad(B, A, D, C);
        }
        for (int s = -1; s <= 1; s += 2) mb_quad(v3(s * 0.92f, cy0, cz0), v3(s * cw, cy0, cz0), v3(s * cw, BOILER_Y, cz0), v3(s * 0.92f, BOILER_Y, cz0));
        /* the spectacles: brass-ringed round windows */
        for (int s = -1; s <= 1; s += 2) {
            mat(PM_BRASS);
            V3 c = v3(s * 1.08f, BOILER_Y + 0.62f, cz0 + 0.02f);
            M4 sv2 = mb->xf;
            push(m4trans(c.x, c.y, c.z));
            mb_torus(0.17f, 0.02f, 24, 6);
            mb->xf = sv2;
            mat(PM_GLASS);
            {
                M4 s3 = mb->xf;
                push(m4trans(c.x, c.y, c.z + 0.005f));
                float gl[] = {0, 0, 0.17f, 0};
                mb_lathe(gl, 2, 24, 0, 2 * (float)M_PI);
                mb->xf = s3;
            }
        }
    }
    /* the roof, curved, overhanging */
    mat(PM_GREEN);
    {
        int n = 16;
        for (int i = 0; i < n; i++) {
            float x0 = -cw - 0.05f + (2 * cw + 0.1f) * i / n, x1 = -cw - 0.05f + (2 * cw + 0.1f) * (i + 1) / n;
            float y0 = cyr + 0.4f * (1 - (x0 / cw) * (x0 / cw)), y1 = cyr + 0.4f * (1 - (x1 / cw) * (x1 / cw));
            mb_quad(v3(x0, y0, cz1 - 0.3f), v3(x1, y1, cz1 - 0.3f), v3(x1, y1, cz0 + 0.06f), v3(x0, y0, cz0 + 0.06f));
            mat(PM_CABIN);
            mb_quad(v3(x1, y1 - 0.02f, cz1 - 0.3f), v3(x0, y0 - 0.02f, cz1 - 0.3f), v3(x0, y0 - 0.02f, cz0 + 0.06f), v3(x1, y1 - 0.02f, cz0 + 0.06f));
            mat(PM_GREEN);
        }
        /* the roof's rain strips and ventilator */
        mat(PM_BLACK);
        boxc(0, cyr + 0.43f, -4.3f, 0.35f, 0.04f, 0.4f);
    }
    /* inside the cab: the backhead (glowing round the firehole), gauges, the regulator */
    mat(PM_GRIME);
    mb_quad(v3(-0.9f, cy0 - 0.25f, cz0 - 0.02f), v3(0.9f, cy0 - 0.25f, cz0 - 0.02f), v3(0.9f, BOILER_Y + 0.5f, cz0 - 0.02f), v3(-0.9f, BOILER_Y + 0.5f, cz0 - 0.02f));
    mat(PM_FIRE);
    boxc(0, 2.45f, cz0 - 0.03f, 0.22f, 0.16f, 0.005f);
    mat(PM_BRASS);
    for (int s = -1; s <= 1; s += 2) {
        M4 sv2 = mb->xf;
        push(m4trans(s * 0.35f, 3.35f, cz0 - 0.05f));
        mb_torus(0.12f, 0.02f, 20, 6);
        mb->xf = sv2;
    }
    mat(PM_WHITE);
    for (int s = -1; s <= 1; s += 2) {
        M4 sv2 = mb->xf;
        push(m4trans(s * 0.35f, 3.35f, cz0 - 0.045f));
        push(m4roty((float)M_PI));
        float gl[] = {0, 0, 0.11f, 0};
        mb_lathe(gl, 2, 20, 0, 2 * (float)M_PI);
        mb->xf = sv2;
    }
    mat(PM_COPPER);
    mb_rod(v3(-0.7f, 2.2f, cz0 - 0.06f), v3(-0.7f, 3.6f, cz0 - 0.06f), 0.03f, 6, 0);
    mb_rod(v3(0.7f, 2.2f, cz0 - 0.06f), v3(0.7f, 3.6f, cz0 - 0.06f), 0.03f, 6, 0);
    mat(PM_STEEL);
    mb_rod(v3(0, 3.65f, cz0 - 0.08f), v3(0.55f, 3.25f, cz0 - 0.1f), 0.025f, 6, 1);
}

static void build_tender(void) {
    mb_begin(&m_tender);
    /* origin under the middle axle; 7.6 m over the buffer beams */
    float zf = 3.6f, zr = -3.9f, w = 1.36f, y0 = 1.25f, y1 = 3.15f;
    mat(PM_GREEN);
    for (int s = -1; s <= 1; s += 2) {
        mb_quad(v3(s * w, y0, zr), v3(s * w, y0, zf), v3(s * w, y1, zf), v3(s * w, y1, zr));
        /* flared coping along the top */
        mb_quad(v3(s * w, y1, zr), v3(s * w, y1, zf), v3(s * (w + 0.06f), y1 + 0.18f, zf), v3(s * (w + 0.06f), y1 + 0.18f, zr));
        mat(PM_LINE);
        boxc(s * (w + 0.003f), y0 + 0.12f, (zf + zr) / 2, 0.003f, 0.012f, (zf - zr) / 2 - 0.12f);
        boxc(s * (w + 0.003f), y1 - 0.12f, (zf + zr) / 2, 0.003f, 0.012f, (zf - zr) / 2 - 0.12f);
        mat(PM_GREEN);
    }
    mb_quad(v3(-w, y0, zr), v3(w, y0, zr), v3(w, y1, zr), v3(-w, y1, zr));
    mb_quad(v3(w, y0, zr), v3(-w, y0, zr), v3(-w, y1, zr), v3(w, y1, zr));
    mat(PM_GRIME);
    mb_quad(v3(-w, y0, zf), v3(-w, y1, zf), v3(w, y1, zf), v3(w, y0, zf));
    mat(PM_GREEN);
    /* the coal space: a heap at the front, the tank top behind */
    mat(PM_BLACK);
    mb_quad(v3(-w, y1 - 0.1f, zr), v3(w, y1 - 0.1f, zr), v3(w, y1 - 0.1f, zr + 2.0f), v3(-w, y1 - 0.1f, zr + 2.0f));
    mat(PM_COAL);
    {
        int nx = 28, nz = 36;
        for (int j = 0; j < nz; j++)
            for (int i = 0; i < nx; i++) {
                V3 P[4];
                for (int k = 0; k < 4; k++) {
                    int a = i + (k == 1 || k == 2), b = j + (k >= 2);
                    float x = -w + 2 * w * a / nx, z = zr + 2.0f + (zf - 0.3f - zr - 2.0f) * b / nz;
                    float u = x / w, v = (float)b / nz;
                    float h = y1 + 0.45f * (1 - u * u) * sinf(v * (float)M_PI) + 0.05f * sinf(x * 9 + z * 7) * sinf(z * 5.3f + x * 3)
                              + 0.025f * sinf(x * 23 + z * 17) * sinf(z * 19 - x * 11);
                    if (a == 0 || a == nx) h = y1 + 0.05f;
                    P[k] = v3(x, h, z);
                }
                mb_quad(P[0], P[3], P[2], P[1]);
            }
    }
    /* coal rails above the sides of the coal space */
    mat(PM_GREEN);
    for (int s = -1; s <= 1; s += 2)
        for (int k = 0; k < 3; k++) mb_rod(v3(s * (w + 0.04f), y1 + 0.28f + k * 0.14f, zr + 2.1f), v3(s * (w + 0.04f), y1 + 0.28f + k * 0.14f, zf - 0.1f), 0.018f, 5, 1);
    for (int s = -1; s <= 1; s += 2)
        for (int k = 0; k < 5; k++) { float z = zr + 2.1f + k * (zf - zr - 2.2f) / 4; mb_rod(v3(s * (w + 0.04f), y1 + 0.15f, z), v3(s * (w + 0.04f), y1 + 0.58f, z), 0.02f, 5, 1); }
    /* the front: the coal hole and the shovelling plate */
    mat(PM_GRIME);
    boxc(0, y0 + 0.9f, zf + 0.02f, w, 0.9f, 0.02f);
    mat(PM_COAL);
    boxc(0, y0 + 0.5f, zf + 0.05f, 0.4f, 0.3f, 0.02f);
    /* frames, axleboxes, springs, wheels */
    for (int s = -1; s <= 1; s += 2) {
        mat(PM_BLACK);
        boxc(s * 1.0f, 0.95f, -0.1f, 0.03f, 0.35f, 3.6f);
        for (int i = -1; i <= 1; i++) {
            float z = i * 2.1f;
            boxc(s * 1.05f, 0.62f, z, 0.08f, 0.16f, 0.16f);
            mat(PM_IRON);
            for (int k = 0; k < 4; k++) boxc(s * 1.05f, 0.88f + k * 0.035f, z, 0.07f, 0.014f, 0.45f - k * 0.06f);
            mat(PM_BLACK);
            wheel(s, 0.63f, z, PM_WHEEL_X);
        }
    }
    mat(PM_RED);
    boxc(0, 1.08f, zr - 0.06f, w, 0.22f, 0.06f);
    for (int s = -1; s <= 1; s += 2) {
        mat(PM_BLACK);
        cyl_at(v3(s * 0.87f, 1.06f, zr - 0.1f), v3(0, 0, -1), 0.12f, 0.3f, 14);
        mat(PM_STEEL);
        float bh[] = {0, 0.3f, 0.21f, 0.3f, 0.22f, 0.34f, 0, 0.36f};
        lathe_at(v3(s * 0.87f, 1.06f, zr - 0.1f), v3(0, 0, -1), bh, 4, 20);
    }
    /* the water filler and the tool boxes */
    mat(PM_BLACK);
    cyl_at(v3(0, y1 - 0.1f, zr + 0.8f), v3(0, 1, 0), 0.3f, 0.25f, 16);
}

/* a corridor carriage: 20 m over the body, teak, lit windows, two bogies */
#define COACH_L 20.0f
static void build_coach(void) {
    mb_begin(&m_coach);
    float hl = COACH_L / 2, w = 1.4f, y0 = 1.15f, y1 = 3.45f;
    int nwin = 8;
    for (int s = -1; s <= 1; s += 2) {
        float x = s * w;
        /* the side: panelled teak with the window band cut by the windows */
        mat(PM_TEAK);
        mb_quad(v3(x, y0, -hl * s), v3(x, y0, hl * s), v3(x, 2.2f, hl * s), v3(x, 2.2f, -hl * s));
        mb_quad(v3(x, 2.95f, -hl * s), v3(x, 2.95f, hl * s), v3(x, y1, hl * s), v3(x, y1, -hl * s));
        /* window band: pillars and windows (uv.x counts windows, for which are lit) */
        float wl = 1.45f, pitch = (COACH_L - 1.2f) / nwin;
        for (int i = 0; i < nwin; i++) {
            float zc = -hl + 0.6f + pitch * (i + 0.5f);
            float za = zc - wl / 2, zb = zc + wl / 2;
            mat(PM_WINDOW);
            float u0 = (float)i + (s > 0 ? 0 : 20), u1 = u0 + 0.999f;
            if (s > 0) mb_quad_uv(v3(x, 2.22f, zb), v3(x, 2.22f, za), v3(x, 2.93f, za), v3(x, 2.93f, zb), u0, 0, u1, 1);
            else mb_quad_uv(v3(x, 2.22f, za), v3(x, 2.22f, zb), v3(x, 2.93f, zb), v3(x, 2.93f, za), u0, 0, u1, 1);
            mat(PM_TEAK);
            float pa = zb, pb = zc + pitch - wl / 2;
            if (i == nwin - 1) pb = hl;
            if (s > 0) mb_quad(v3(x, 2.2f, pb), v3(x, 2.2f, pa), v3(x, 2.95f, pa), v3(x, 2.95f, pb));
            else mb_quad(v3(x, 2.2f, pa), v3(x, 2.2f, pb), v3(x, 2.95f, pb), v3(x, 2.95f, pa));
            if (i == 0) {
                if (s > 0) mb_quad(v3(x, 2.2f, za), v3(x, 2.2f, -hl), v3(x, 2.95f, -hl), v3(x, 2.95f, za));
                else mb_quad(v3(x, 2.2f, -hl), v3(x, 2.2f, za), v3(x, 2.95f, za), v3(x, 2.95f, -hl));
            }
            /* a frame of lighter beading round each */
            mat(PM_LINE);
            boxc(x + s * 0.004f, 2.205f, zc, 0.004f, 0.012f, wl / 2 + 0.02f);
            boxc(x + s * 0.004f, 2.945f, zc, 0.004f, 0.012f, wl / 2 + 0.02f);
        }
        /* panel beading on the lower side */
        mat(PM_LINE);
        boxc(x + s * 0.004f, 1.3f, 0, 0.004f, 0.01f, hl - 0.1f);
        for (int i = 0; i <= 10; i++) boxc(x + s * 0.004f, 1.7f, -hl + 0.1f + i * (COACH_L - 0.2f) / 10, 0.004f, 0.48f, 0.01f);
    }
    /* ends, with gangways */
    mat(PM_TEAK);
    for (int e = -1; e <= 1; e += 2) {
        float z = e * hl;
        if (e > 0) mb_quad(v3(-w, y0, z), v3(w, y0, z), v3(w, y1, z), v3(-w, y1, z));
        else mb_quad(v3(w, y0, z), v3(-w, y0, z), v3(-w, y1, z), v3(w, y1, z));
        mat(PM_BLACK);
        boxc(0, 2.1f, z + e * 0.25f, 0.55f, 1.0f, 0.25f);
        mat(PM_TEAK);
    }
    /* the roof: an elliptical arch, grey */
    mat(PM_ROOF);
    {
        int n = 14;
        for (int i = 0; i < n; i++) {
            float a0 = (float)M_PI * i / n, a1 = (float)M_PI * (i + 1) / n;
            V3 A = v3(w * 1.02f * cosf(a0), y1 + 0.5f * sinf(a0), -hl), B = v3(w * 1.02f * cosf(a1), y1 + 0.5f * sinf(a1), -hl);
            V3 C = v3(B.x, B.y, hl), D = v3(A.x, A.y, hl);
            mb_quad(A, D, C, B);
        }
        /* roof vents */
        mat(PM_BLACK);
        for (int i = 0; i < 6; i++) cyl_at(v3(0, y1 + 0.47f, -hl + 2 + i * 3.2f), v3(0, 1, 0), 0.12f, 0.12f, 10);
    }
    /* underframe: solebar, battery boxes, bogies */
    mat(PM_BLACK);
    boxc(0, 1.05f, 0, 1.3f, 0.1f, hl - 0.2f);
    boxc(0, 0.8f, 0, 0.5f, 0.2f, 2.5f);
    for (int b = -1; b <= 1; b += 2) {
        float zb = b * (hl - 3.0f);
        for (int s = -1; s <= 1; s += 2) {
            mat(PM_BLACK);
            boxc(s * 1.0f, 0.62f, zb, 0.04f, 0.18f, 1.6f);
            for (int a = -1; a <= 1; a += 2) {
                boxc(s * 1.04f, 0.5f, zb + a * 1.3f, 0.06f, 0.12f, 0.14f);
                wheel(s, 0.48f, zb + a * 1.3f, PM_WHEEL_X);
            }
            mat(PM_IRON);
            for (int k = 0; k < 3; k++) boxc(s * 1.04f, 0.78f + k * 0.03f, zb, 0.06f, 0.012f, 0.6f - k * 0.12f);
        }
    }
    /* buffers */
    for (int e = -1; e <= 1; e += 2)
        for (int s = -1; s <= 1; s += 2) {
            mat(PM_BLACK);
            cyl_at(v3(s * 0.87f, 1.06f, e * hl), v3(0, 0, (float)e), 0.11f, 0.3f, 12);
            mat(PM_STEEL);
            float bh[] = {0, 0.3f, 0.2f, 0.3f, 0.21f, 0.33f, 0, 0.35f};
            lathe_at(v3(s * 0.87f, 1.06f, e * hl), v3(0, 0, (float)e), bh, 4, 16);
        }
}

/* the moving parts: unit rod (along z, length 1), a crosshead, a link, a crank pin and boss */
static void build_parts(void) {
    mb_begin(&m_rod);
    mat(PM_STEEL);
    /* a fluted rod: I-section-ish box, with bosses at each end */
    boxc(0, 0, 0.5f, 0.035f, 0.055f, 0.5f);
    mb_begin(&m_link);
    mat(PM_STEEL);
    boxc(0, 0, 0.5f, 0.025f, 0.03f, 0.5f);
    mb_begin(&m_xhead);
    mat(PM_STEEL);
    boxc(0, 0, 0, 0.09f, 0.12f, 0.16f);
    mat(PM_STEEL);
    cyl_at(v3(0, 0, 0.1f), v3(0, 0, 1), 0.05f, 1.0f, 10);   /* the piston rod */
    mb_begin(&m_pin);
    mat(PM_STEEL);
    cyl_at(v3(0, 0, 0), v3(1, 0, 0), 0.075f, 0.16f, 16);
    mat(PM_BLACK);
    cyl_at(v3(0, 0, 0), v3(1, 0, 0), 0.13f, 0.04f, 20);
}

static void loco_meshes_init(void) {
    palette_init();
    static Mesh mw;
    build_loco(); mesh_split(&m_loco, &mw, is_wheel); gm_upload(&g_loco, &m_loco); gm_upload(&g_loco_w, &mw);
    build_tender(); mesh_split(&m_tender, &mw, is_wheel); gm_upload(&g_tender, &m_tender); gm_upload(&g_tender_w, &mw);
    build_coach(); mesh_split(&m_coach, &mw, is_wheel); gm_upload(&g_coach, &m_coach); gm_upload(&g_coach_w, &mw);
    build_parts();
    mb_begin(&m_rod); mat(PM_STEEL); boxc(0, 0, 0.5f, 0.035f, 0.06f, 0.5f);
    gm_upload(&g_rod, &m_rod);
    mb_begin(&m_link); mat(PM_STEEL); boxc(0, 0, 0.5f, 0.022f, 0.03f, 0.5f);
    gm_upload(&g_link, &m_link);
    mb_begin(&m_xhead); mat(PM_STEEL); boxc(0, 0, 0, 0.09f, 0.11f, 0.16f); cyl_at(v3(0, 0, 0.12f), v3(0, 0, 1), 0.05f, 1.0f, 10);
    gm_upload(&g_xhead, &m_xhead);
    mb_begin(&m_pin); mat(PM_STEEL); cyl_at(v3(0, 0, 0), v3(1, 0, 0), 0.07f, 0.17f, 14);
    gm_upload(&g_pin, &m_pin);
}
