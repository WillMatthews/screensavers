/* steam: the railway itself and what stands beside it, and the smoke.
 * Included once, by steam.c.
 *
 * The track (rails, sleepers, ballast), the telegraph poles and their
 * wires are built in 64 m chunks along the line as the camera comes near;
 * the set pieces (station, crossing, viaduct, tunnel portals) as meshes of
 * their own when one is within reach. Signals are drawn each frame. */

/* ---- chunks of track and poles ---- */
#define CHUNK 64.0
#define MAXCHUNK 72
typedef struct { long k; int live, lod; double o[3]; GMesh mesh; unsigned used; } Chunk;
static Chunk chunks[MAXCHUNK];
static int pole_side = 1;
#define SA_ROWS_F 8.0f
static double pole_off = 4.8;

/* a point on the line at z, offset x across it and y up (a local frame that follows the line) */
static void line_pt(double z, double x, double y, double *o) {
    double dx = track_dx(z);
    double l = sqrt(1 + dx * dx);
    /* across: perpendicular to the line in plan */
    double ax = 1 / l, az = -dx / l;
    o[0] = track_x(z) + ax * x;
    o[1] = track_y(z) + y;
    o[2] = z + az * x;
}
static V3 lp(const double *org, double z, double x, double y) {
    double o[3];
    line_pt(z, x, y, o);
    return v3((float)(o[0] - org[0]), (float)(o[1] - org[1]), (float)(o[2] - org[2]));
}

static int in_tunnel_z(double z) {
    Feat t = feat_near(z, F_TUNNEL);
    return t.on && z > t.z0 && z < t.z1;
}
static int lineside_in_tunnel(double z) { return in_tunnel_z(z); }
static int on_viaduct(double z) {
    Feat v = feat_near(z, F_VIADUCT);
    return v.on && z > v.z0 && z < v.z1;
}

static void chunk_build(Chunk *c) {
    static Mesh m;
    mb_begin(&m);
    double z0 = c->k * CHUNK;
    line_pt(z0, 0, 0, c->o);
    const double *O = c->o;
    int n = 16;
    /* ballast: a flat top and shoulders (on the viaduct, a deck between parapets) */
    for (int i = 0; i < n; i++) {
        double za = z0 + CHUNK * i / n, zb = z0 + CHUNK * (i + 1) / n;
        mat(PM_BALLAST);
        static const float prof[][2] = {{-3.0f, -0.62f}, {-1.95f, -0.2f}, {1.95f, -0.2f}, {3.0f, -0.62f}};
        for (int k = 0; k < 3; k++) {
            V3 a = lp(O, za, prof[k][0], prof[k][1]), b = lp(O, za, prof[k + 1][0], prof[k + 1][1]);
            V3 cc = lp(O, zb, prof[k + 1][0], prof[k + 1][1]), d = lp(O, zb, prof[k][0], prof[k][1]);
            mb_quad(a, d, cc, b);
        }
    }
    if (c->lod == 0) {
        /* sleepers */
        mat(PM_SLEEPER);
        for (double z = ceil(z0 / 0.72) * 0.72; z < z0 + CHUNK; z += 0.72) {
            V3 a = lp(O, z - 0.13, -1.3, -0.2), b = lp(O, z + 0.13, 1.3, -0.06);
            mb_box(v3(fminf(a.x, b.x), a.y, fminf(a.z, b.z)), v3(fmaxf(a.x, b.x), b.y, fmaxf(a.z, b.z)));
        }
    }
    /* rails: head bright, web dark */
    for (int r = -1; r <= 1; r += 2) {
        double x = r * 0.7525;
        for (int i = 0; i < n; i++) {
            double za = z0 + CHUNK * i / n, zb = z0 + CHUNK * (i + 1) / n;
            mat(PM_RAIL);
            V3 a = lp(O, za, x - 0.036, 0), b = lp(O, za, x + 0.036, 0), cc = lp(O, zb, x + 0.036, 0), d = lp(O, zb, x - 0.036, 0);
            mb_quad(a, d, cc, b);
            mat(PM_IRON);
            for (int sd = -1; sd <= 1; sd += 2) {
                V3 t0 = lp(O, za, x + sd * 0.036, 0), t1 = lp(O, zb, x + sd * 0.036, 0);
                V3 b0 = lp(O, za, x + sd * 0.036, -0.14), b1 = lp(O, zb, x + sd * 0.036, -0.14);
                if (sd < 0) mb_quad(t0, b0, b1, t1); else mb_quad(t0, t1, b1, b0);
            }
        }
    }
    /* telegraph poles every 50 m, and the wires sagging between them */
    if (!in_tunnel_z(z0) && !on_viaduct(z0)) {
        double side = pole_side * pole_off;
        for (double z = ceil(z0 / 50) * 50; z < z0 + CHUNK; z += 50) {
            if (in_tunnel_z(z) || on_viaduct(z) || in_feat(z, F_STATION, 20)) continue;
            double base[3];
            line_pt(z, side, 0, base);
            double gy = terrain_h(base[0], base[2]);
            V3 b = v3((float)(base[0] - O[0]), (float)(gy - O[1] - 0.3), (float)(base[2] - O[2]));
            float top = (float)(track_y(z) + 7.6 - gy);
            if (top < 6) top = 6;
            mat(PM_POLE);
            mb_rod(b, vadd(b, v3(0, top + 0.4f, 0)), 0.12f, 7, 1);
            for (int a = 0; a < 2; a++) {
                float y = top - a * 0.6f;
                mb_box(vadd(b, v3(-0.9f, y - 0.06f, -0.05f)), vadd(b, v3(0.9f, y + 0.06f, 0.05f)));
                mat(PM_WHITE);
                for (int k = 0; k < 4; k++) {
                    float x = -0.75f + k * 0.5f;
                    mb_rod(vadd(b, v3(x, y + 0.06f, 0)), vadd(b, v3(x, y + 0.2f, 0)), 0.035f, 5, 1);
                }
                mat(PM_POLE);
            }
            /* wires to the next pole */
            double z2 = z + 50;
            double b2[3];
            line_pt(z2, side, 0, b2);
            double gy2 = terrain_h(b2[0], b2[2]);
            float top2 = (float)(track_y(z2) + 7.6 - gy2);
            if (top2 < 6) top2 = 6;
            V3 e = v3((float)(b2[0] - O[0]), (float)(gy2 - O[1] - 0.3), (float)(b2[2] - O[2]));
            mat(PM_IRON);
            for (int a = 0; a < 2; a++)
                for (int k = 0; k < 4; k++) {
                    float x = -0.75f + k * 0.5f;
                    V3 p0 = vadd(b, v3(x, top - a * 0.6f + 0.2f, 0)), p1 = vadd(e, v3(x, top2 - a * 0.6f + 0.2f, 0));
                    int segs = 10;
                    for (int s = 0; s < segs; s++) {
                        float u0 = (float)s / segs, u1 = (float)(s + 1) / segs;
                        V3 q0 = vlerp(p0, p1, u0), q1 = vlerp(p0, p1, u1);
                        q0.y -= 0.55f * 4 * u0 * (1 - u0);
                        q1.y -= 0.55f * 4 * u1 * (1 - u1);
                        float w = 0.007f;
                        mb_quad(vadd(q0, v3(0, -w, 0)), vadd(q1, v3(0, -w, 0)), vadd(q1, v3(0, w, 0)), vadd(q0, v3(0, w, 0)));
                        mb_quad(vadd(q0, v3(-w, 0, 0)), vadd(q1, v3(-w, 0, 0)), vadd(q1, v3(w, 0, 0)), vadd(q0, v3(w, 0, 0)));
                    }
                }
        }
    }
    gm_upload(&c->mesh, &m);
}

static Chunk *chunk_get(long k, int lod) {
    Chunk *fr = NULL, *old = NULL;
    for (int i = 0; i < MAXCHUNK; i++) {
        Chunk *c = &chunks[i];
        if (c->live && c->k == k && c->lod == lod) { c->used = tile_clock; return c; }
        if (!c->live) { if (!fr) fr = c; }
        else if (!old || c->used < old->used) old = c;
    }
    Chunk *c = fr ? fr : old;
    c->live = 1; c->k = k; c->lod = lod; c->used = tile_clock;
    chunk_build(c);
    return c;
}

/* ---- set pieces ---- */
typedef struct { long seg; int kind, live; double o[3]; GMesh mesh; } Piece;
#define MAXPIECE 8
static Piece pieces[MAXPIECE];
static struct { double o[3]; } lamps_at[32];
static int nlamps;

/* the station: two platforms, lamps, a canopy and building, nameboards */
static void build_station(Piece *p, Feat f) {
    static Mesh m;
    mb_begin(&m);
    line_pt(f.z0, 0, 0, p->o);
    const double *O = p->o;
    int n = 21;
    for (int s = -1; s <= 1; s += 2) {
        for (int i = 0; i < n; i++) {
            double za = f.z0 + (f.z1 - f.z0) * i / n, zb = f.z0 + (f.z1 - f.z0) * (i + 1) / n;
            mat(PM_COPING);
            V3 a = lp(O, za, s * 1.62, 0.92), b = lp(O, za, s * 2.25, 0.92), c = lp(O, zb, s * 2.25, 0.92), d = lp(O, zb, s * 1.62, 0.92);
            if (s > 0) mb_quad(a, d, c, b); else mb_quad(a, b, c, d);
            V3 a2 = lp(O, za, s * 1.62, -0.6), d2 = lp(O, zb, s * 1.62, -0.6);
            if (s > 0) mb_quad(a2, d2, d, a); else mb_quad(a2, a, d, d2);
            mat(PM_BALLAST);
            V3 e = lp(O, za, s * 7.5, 0.9), g = lp(O, zb, s * 7.5, 0.9);
            if (s > 0) mb_quad(b, c, g, e); else mb_quad(b, e, g, c);
        }
        /* lamps every 21 m */
        for (int i = 0; i <= 10; i++) {
            double z = f.z0 + 5 + i * 20;
            V3 b = lp(O, z, s * 4.2, 0.92);
            mat(PM_IRON);
            mb_rod(b, vadd(b, v3(0, 3.4f, 0)), 0.05f, 6, 1);
            mb_box(vadd(b, v3(-0.17f, 3.35f, -0.17f)), vadd(b, v3(0.17f, 3.4f, 0.17f)));
            mat(PM_LAMP);
            mb_box(vadd(b, v3(-0.14f, 3.4f, -0.14f)), vadd(b, v3(0.14f, 3.75f, 0.14f)));
            mat(PM_IRON);
            float roofp[] = {0.22f, 0, 0.0f, 0.18f};
            M4 sv = mb->xf;
            mb->xf = m4mul(sv, m4mul(m4trans(b.x, b.y + 3.75f, b.z), m4rotx(-(float)M_PI / 2)));
            mb_lathe(roofp, 2, 4, (float)M_PI / 4, (float)M_PI / 4 + 2 * (float)M_PI);
            mb->xf = sv;
        }
        /* nameboards at each end */
        for (int e = 0; e < 2; e++) {
            double z = e ? f.z1 - 25 : f.z0 + 25;
            V3 b = lp(O, z, s * 5.6, 0.92);
            mat(PM_IRON);
            mb_rod(vadd(b, v3(0, 0, -1.7f)), vadd(b, v3(0, 2.4f, -1.7f)), 0.05f, 6, 1);
            mb_rod(vadd(b, v3(0, 0, 1.7f)), vadd(b, v3(0, 2.4f, 1.7f)), 0.05f, 6, 1);
            mat(PM_SIGN);
            /* the face toward the line, reading left to right from the track */
            V3 off = v3(-s * 0.03f, 0, 0);
            V3 p0 = vadd(vadd(b, v3(0, 1.75f, -1.9f)), off), p1 = vadd(vadd(b, v3(0, 1.75f, 1.9f)), off);
            V3 p2 = vadd(vadd(b, v3(0, 2.45f, 1.9f)), off), p3 = vadd(vadd(b, v3(0, 2.45f, -1.9f)), off);
            if (s > 0) mb_quad_uv(p0, p1, p2, p3, 0, 1, 1, 0);
            else mb_quad_uv(p1, p0, p3, p2, 0, 1, 1, 0);
            mat(PM_WHITE);
            mb_box(vadd(b, v3(-0.02f, 1.7f, -1.95f)), vadd(b, v3(0.02f, 2.5f, 1.95f)));
        }
    }
    /* the canopy and building on the down side */
    int s = -pole_side;
    double cz0 = f.z0 + 60, cz1 = f.z0 + 150;
    for (int i = 0; i <= 9; i++) {
        double z = cz0 + (cz1 - cz0) * i / 9;
        V3 b = lp(O, z, s * 3.4, 0.92);
        mat(PM_IRON);
        mb_rod(b, vadd(b, v3(0, 3.6f, 0)), 0.09f, 8, 1);
    }
    mat(PM_SLATE);
    for (int i = 0; i < 12; i++) {
        double za = cz0 - 3 + (cz1 - cz0 + 6) * i / 12, zb = cz0 - 3 + (cz1 - cz0 + 6) * (i + 1) / 12;
        V3 a = lp(O, za, s * 2.3, 0.92 + 3.75), b = lp(O, za, s * 7.0, 0.92 + 4.6), c = lp(O, zb, s * 7.0, 0.92 + 4.6), d = lp(O, zb, s * 2.3, 0.92 + 3.75);
        mb_quad(a, b, c, d);
        mat(PM_WHITE);
        V3 a2 = lp(O, za, s * 2.3, 0.92 + 3.45), d2 = lp(O, zb, s * 2.3, 0.92 + 3.45);
        mb_quad(a2, a, d, d2);
        mat(PM_SLATE);
    }
    /* the building: brick, lit windows, a door */
    {
        double bz0 = cz0 + 8, bz1 = cz1 - 8;
        V3 c0 = lp(O, bz0, s * 7.0, 0.92), c1 = lp(O, bz1, s * 7.0, 0.92);
        V3 c2 = lp(O, bz1, s * 13.0, 0.92), c3 = lp(O, bz0, s * 13.0, 0.92);
        V3 up = v3(0, 4.4f, 0);
        mat(PM_BRICK);
        mb_quad(c0, c1, vadd(c1, up), vadd(c0, up));
        mb_quad(c1, c2, vadd(c2, up), vadd(c1, up));
        mb_quad(c3, c0, vadd(c0, up), vadd(c3, up));
        mb_quad(c2, c3, vadd(c3, up), vadd(c2, up));
        mat(PM_SLATE);
        V3 r = v3(0, 6.3f, 0);
        V3 m0 = vlerp(c0, c3, 0.5f), m1 = vlerp(c1, c2, 0.5f);
        mb_quad(vadd(c0, up), vadd(c1, up), vadd(m1, r), vadd(m0, r));
        mb_quad(vadd(c2, up), vadd(c3, up), vadd(m0, r), vadd(m1, r));
        mb_tri(vadd(c3, up), vadd(c0, up), vadd(m0, r));
        mb_tri(vadd(c1, up), vadd(c2, up), vadd(m1, r));
        for (int i = 0; i < 9; i++) {
            float u = (i + 0.5f) / 9;
            V3 w = vlerp(c0, c1, u);
            V3 d = vnorm(vsub(c3, c0));
            mat(i % 3 == 1 ? PM_BLACK : PM_BWIN);
            float h0 = i % 3 == 1 ? 0.0f : 1.0f, h1 = i % 3 == 1 ? 2.3f : 2.6f;
            V3 a = vadd(w, vmul(d, -0.02f));
            V3 along = vnorm(vsub(c1, c0));
            mb_quad(vadd(vadd(a, vmul(along, -0.5f)), v3(0, h0, 0)), vadd(vadd(a, vmul(along, 0.5f)), v3(0, h0, 0)),
                    vadd(vadd(a, vmul(along, 0.5f)), v3(0, h1, 0)), vadd(vadd(a, vmul(along, -0.5f)), v3(0, h1, 0)));
        }
    }
    gm_upload(&p->mesh, &m);
}

/* the level crossing: gates closed across the road, a keeper's lodge, a car waiting */
static void build_crossing(Piece *p, Feat f) {
    static Mesh m;
    mb_begin(&m);
    double zc = (f.z0 + f.z1) / 2;
    line_pt(zc, 0, 0, p->o);
    const double *O = p->o;
    for (int s = -1; s <= 1; s += 2) {
        /* the gates: white, a red disc, across the road on each side */
        for (int e = -1; e <= 1; e += 2) {
            V3 post = lp(O, zc + e * 3.4, s * 3.6, -0.5);
            mat(PM_WHITE);
            mb_box(vadd(post, v3(-0.12f, 0, -0.12f)), vadd(post, v3(0.12f, 2.1f, 0.12f)));
            V3 tip = lp(O, zc, s * 3.6, -0.5);
            V3 dir = vsub(tip, post);
            float L = vlen(dir);
            V3 Z = vmul(dir, 1 / L), X = vnorm(vcross(v3(0, 1, 0), Z)), Y = v3(0, 1, 0);
            M4 sv = mb->xf;
            mb->xf = m4mul(sv, m4basis(X, Y, Z, post));
            boxc(0, 1.3f, L / 2, 0.04f, 0.06f, L / 2);
            boxc(0, 0.7f, L / 2, 0.04f, 0.05f, L / 2);
            for (int k = 0; k < 6; k++) boxc(0, 1.0f, 0.3f + k * L / 6.3f, 0.03f, 0.3f, 0.025f);
            mat(PM_SIG_R);
            M4 s2 = mb->xf;
            push(m4mul(m4trans(s * -0.05f, 1.0f, L * 0.55f), m4roty((float)M_PI / 2)));
            float disc[] = {0, 0, 0.22f, 0};
            mb_lathe(disc, 2, 14, 0, 2 * (float)M_PI);
            push(m4roty((float)M_PI));
            mb_lathe(disc, 2, 14, 0, 2 * (float)M_PI);
            mb->xf = s2;
            mb->xf = sv;
        }
    }
    /* the keeper's lodge */
    {
        V3 c = lp(O, zc + 9, -pole_side * 9.0, -0.5);
        mat(PM_BRICK);
        mb_box(vadd(c, v3(-2.5f, 0, -2.0f)), vadd(c, v3(2.5f, 3.0f, 2.0f)));
        mat(PM_SLATE);
        mb_quad(vadd(c, v3(-2.7f, 3.0f, -2.2f)), vadd(c, v3(2.7f, 3.0f, -2.2f)), vadd(c, v3(2.7f, 4.6f, 0)), vadd(c, v3(-2.7f, 4.6f, 0)));
        mb_quad(vadd(c, v3(2.7f, 3.0f, 2.2f)), vadd(c, v3(-2.7f, 3.0f, 2.2f)), vadd(c, v3(-2.7f, 4.6f, 0)), vadd(c, v3(2.7f, 4.6f, 0)));
        mat(PM_BWIN);
        mb_box(vadd(c, v3(-1.6f, 1.0f, -2.02f)), vadd(c, v3(-0.6f, 2.0f, -2.0f)));
        mb_box(vadd(c, v3(0.6f, 1.0f, -2.02f)), vadd(c, v3(1.6f, 2.0f, -2.0f)));
    }
    /* the car: a dark saloon waiting on the road, headlamps on */
    {
        int s = pole_side;
        V3 c = lp(O, zc - 1.2, s * 9.5, -0.5);
        V3 X = vnorm(vsub(lp(O, zc - 1.2, s * 10.5, -0.5), c));     /* away from the line */
        V3 Y = v3(0, 1, 0), Z = vcross(X, Y);
        M4 sv = mb->xf;
        mb->xf = m4mul(sv, m4basis(Z, Y, X, c));
        /* local: z away from the line, so its nose (toward the line) is -z */
        mat(PM_CARPAINT);
        boxc(0, 0.62f, 0, 0.85f, 0.3f, 2.2f);
        boxc(0, 1.15f, 0.25f, 0.78f, 0.25f, 1.1f);
        mat(PM_GLASS);
        boxc(0, 1.15f, 0.25f, 0.8f, 0.2f, 1.0f);
        mat(PM_BLACK);
        for (int a = -1; a <= 1; a += 2)
            for (int b = -1; b <= 1; b += 2) {
                M4 s2 = mb->xf;
                push(m4mul(m4trans(a * 0.8f, 0.34f, b * 1.4f), m4roty((float)M_PI / 2)));
                mb_cyl(0.34f, -0.1f, 0.1f, 14, 1);
                mb->xf = s2;
            }
        mat(PM_HEAD);
        for (int a = -1; a <= 1; a += 2) {
            M4 s2 = mb->xf;
            push(m4mul(m4trans(a * 0.58f, 0.7f, -2.21f), m4roty((float)M_PI)));
            float disc[] = {0, 0, 0.13f, 0};
            mb_lathe(disc, 2, 14, 0, 2 * (float)M_PI);
            mb->xf = s2;
        }
        mat(PM_TAIL);
        for (int a = -1; a <= 1; a += 2) boxc(a * 0.7f, 0.75f, 2.21f, 0.09f, 0.05f, 0.01f);
        mb->xf = sv;
    }
    gm_upload(&p->mesh, &m);
}

/* the viaduct: stone arches over the valley, parapets */
static void build_viaduct(Piece *p, Feat f) {
    static Mesh m;
    mb_begin(&m);
    line_pt(f.z0, 0, 0, p->o);
    const double *O = p->o;
    double len = f.z1 - f.z0;
    int nspan = (int)(len / 22);
    if (nspan < 3) nspan = 3;
    double sp = len / nspan;
    double hw = 2.7;
    mat(PM_STONE);
    for (int i = 0; i < nspan; i++) {
        double za = f.z0 + i * sp, zb = za + sp;
        double pier = 2.6;
        double r = (sp - pier) / 2, mid = (za + zb) / 2;
        double crown = -1.6;              /* the arch crown, below the rail */
        double spring = crown - r;
        int segs = 14;
        for (int k = 0; k < segs; k++) {
            double u0 = (double)k / segs, u1 = (double)(k + 1) / segs;
            double z0 = za + pier / 2 + u0 * 2 * r, z1 = za + pier / 2 + u1 * 2 * r;
            double y0 = spring + sqrt(fmax(r * r - (z0 - mid) * (z0 - mid), 0)), y1 = spring + sqrt(fmax(r * r - (z1 - mid) * (z1 - mid), 0));
            for (int s = -1; s <= 1; s += 2) {
                V3 a = lp(O, z0, s * hw, y0), b = lp(O, z1, s * hw, y1), c = lp(O, z1, s * hw, -0.6), d = lp(O, z0, s * hw, -0.6);
                if (s > 0) mb_quad(a, b, c, d); else mb_quad(a, d, c, b);
            }
            /* the soffit */
            V3 a = lp(O, z0, -hw, y0), b = lp(O, z0, hw, y0), c = lp(O, z1, hw, y1), d = lp(O, z1, -hw, y1);
            mb_quad(a, b, c, d);
        }
        /* the pier at za down to the valley floor */
        double pz = za;
        double ground = terrain_h(track_x(pz), pz) - track_y(pz) - 2;
        if (i > 0) {
            for (int s = -1; s <= 1; s += 2) {
                V3 a = lp(O, pz - pier / 2, s * hw, spring), b = lp(O, pz + pier / 2, s * hw, spring);
                V3 c = lp(O, pz + pier / 2 + 0.5, s * (hw + 0.6), ground), d = lp(O, pz - pier / 2 - 0.5, s * (hw + 0.6), ground);
                if (s > 0) mb_quad(d, c, b, a); else mb_quad(a, b, c, d);
                /* the spandrel over the pier */
                V3 e = lp(O, pz - pier / 2, s * hw, -0.6), g = lp(O, pz + pier / 2, s * hw, -0.6);
                if (s > 0) mb_quad(a, b, g, e); else mb_quad(a, e, g, b);
            }
            V3 a = lp(O, pz - pier / 2, -hw, spring), b = lp(O, pz - pier / 2, hw, spring);
            V3 c = lp(O, pz - pier / 2 - 0.5, hw + 0.6, ground), d = lp(O, pz - pier / 2 - 0.5, -hw - 0.6, ground);
            mb_quad(a, b, c, d);
            a = lp(O, pz + pier / 2, hw, spring); b = lp(O, pz + pier / 2, -hw, spring);
            c = lp(O, pz + pier / 2 + 0.5, -hw - 0.6, ground); d = lp(O, pz + pier / 2 + 0.5, hw + 0.6, ground);
            mb_quad(a, b, c, d);
        }
    }
    /* parapets with coping, the whole length */
    int n = 24;
    for (int i = 0; i < n; i++) {
        double za = f.z0 - 8 + (len + 16) * i / n, zb = f.z0 - 8 + (len + 16) * (i + 1) / n;
        for (int s = -1; s <= 1; s += 2) {
            mat(PM_STONE);
            V3 a = lp(O, za, s * hw, -0.6), b = lp(O, zb, s * hw, -0.6), c = lp(O, zb, s * hw, 1.0), d = lp(O, za, s * hw, 1.0);
            if (s > 0) mb_quad(a, b, c, d); else mb_quad(a, d, c, b);
            V3 a2 = lp(O, za, s * (hw - 0.4), 1.0), b2 = lp(O, zb, s * (hw - 0.4), 1.0);
            V3 a3 = lp(O, za, s * (hw - 0.4), -0.3), b3 = lp(O, zb, s * (hw - 0.4), -0.3);
            if (s > 0) mb_quad(a3, a2, b2, b3); else mb_quad(a3, b3, b2, a2);
            mat(PM_COPING);
            V3 e = lp(O, za, s * (hw + 0.08), 1.12), g = lp(O, zb, s * (hw + 0.08), 1.12), h = lp(O, zb, s * (hw - 0.48), 1.12), k = lp(O, za, s * (hw - 0.48), 1.12);
            mb_quad(e, g, h, k);
        }
    }
    gm_upload(&p->mesh, &m);
}

/* a tunnel: portals of dressed stone, a brick bore */
static void build_tunnel(Piece *p, Feat f) {
    static Mesh m;
    mb_begin(&m);
    line_pt(f.z0, 0, 0, p->o);
    const double *O = p->o;
    double R = 4.0, spring = 2.2;
    /* the bore: a horseshoe of brick along the line */
    int na = 14;
    double len = f.z1 - f.z0;
    int nz = (int)(len / 8);
    mat(PM_BRICK);
    for (int i = 0; i < nz; i++) {
        double za = f.z0 + len * i / nz, zb = f.z0 + len * (i + 1) / nz;
        for (int k = 0; k < na; k++) {
            double a0 = M_PI * k / na, a1 = M_PI * (k + 1) / na;
            V3 A = lp(O, za, R * cos(a0), spring + R * sin(a0)), B = lp(O, za, R * cos(a1), spring + R * sin(a1));
            V3 C = lp(O, zb, R * cos(a1), spring + R * sin(a1)), D = lp(O, zb, R * cos(a0), spring + R * sin(a0));
            mb_quad(A, B, C, D);
        }
        for (int s = -1; s <= 1; s += 2) {
            V3 A = lp(O, za, s * R, -0.6), B = lp(O, za, s * R, spring), C = lp(O, zb, s * R, spring), D = lp(O, zb, s * R, -0.6);
            if (s > 0) mb_quad(A, D, C, B); else mb_quad(A, B, C, D);
        }
    }
    /* the portals at each end: a face wall with the arch, a parapet */
    for (int e = 0; e < 2; e++) {
        double z = e ? f.z1 : f.z0;
        mat(PM_STONE);
        int nn = 16;
        double W2 = 13, Ht = 11;
        for (int k = 0; k < nn; k++) {
            double a0 = M_PI * k / nn, a1 = M_PI * (k + 1) / nn;
            double ox0 = (R + 0.01) * cos(a0), oy0 = spring + (R + 0.01) * sin(a0), ox1 = (R + 0.01) * cos(a1), oy1 = spring + (R + 0.01) * sin(a1);
            /* from the arch out to the wall's edge, as a fan to the top */
            double tx0 = W2 * cos(a0) * 1.2, tx1 = W2 * cos(a1) * 1.2;
            if (tx0 > W2) tx0 = W2;
            if (tx0 < -W2) tx0 = -W2;
            if (tx1 > W2) tx1 = W2;
            if (tx1 < -W2) tx1 = -W2;
            V3 A = lp(O, z, ox0, oy0), B = lp(O, z, ox1, oy1), C = lp(O, z, tx1, Ht), D = lp(O, z, tx0, Ht);
            if (e == 0) mb_quad(A, D, C, B); else mb_quad(A, B, C, D);
        }
        for (int s = -1; s <= 1; s += 2) {
            V3 A = lp(O, z, s * R, -1.5), B = lp(O, z, s * W2, -1.5), C = lp(O, z, s * W2, Ht), D = lp(O, z, s * R, spring);
            if ((s > 0) == (e == 0)) mb_quad(A, B, C, D); else mb_quad(A, D, C, B);
        }
        /* the arch ring of voussoirs, standing proud */
        mat(PM_COPING);
        for (int k = 0; k < nn; k++) {
            double a0 = M_PI * k / nn, a1 = M_PI * (k + 1) / nn;
            double dz = e ? 0.25 : -0.25;
            V3 A = lp(O, z + dz, R * cos(a0), spring + R * sin(a0)), B = lp(O, z + dz, R * cos(a1), spring + R * sin(a1));
            V3 C = lp(O, z + dz, (R + 0.9) * cos(a1), spring + (R + 0.9) * sin(a1)), D = lp(O, z + dz, (R + 0.9) * cos(a0), spring + (R + 0.9) * sin(a0));
            if (e == 0) mb_quad(A, D, C, B); else mb_quad(A, B, C, D);
        }
        V3 A = lp(O, z, -W2 - 0.3, Ht), B = lp(O, z, W2 + 0.3, Ht);
        mb_box(v3(fminf(A.x, B.x), A.y, fminf(A.z, B.z) - 0.4f), v3(fmaxf(A.x, B.x), A.y + 0.5f, fmaxf(A.z, B.z) + 0.4f));
    }
    gm_upload(&p->mesh, &m);
}

static void piece_get(long seg, int kind) {
    for (int i = 0; i < MAXPIECE; i++)
        if (pieces[i].live && pieces[i].seg == seg && pieces[i].kind == kind) return;
    Feat f = feat(seg, kind);
    if (!f.on) return;
    Piece *p = NULL;
    for (int i = 0; i < MAXPIECE; i++)
        if (!pieces[i].live) { p = &pieces[i]; break; }
    if (!p) {
        /* evict the one furthest from the camera */
        double worst = -1;
        for (int i = 0; i < MAXPIECE; i++) {
            double d = fabs(pieces[i].o[2] - cam.p[2]);
            if (d > worst) { worst = d; p = &pieces[i]; }
        }
    }
    p->live = 1; p->seg = seg; p->kind = kind;
    if (kind == F_STATION) build_station(p, f);
    else if (kind == F_CROSSING) build_crossing(p, f);
    else if (kind == F_VIADUCT) build_viaduct(p, f);
    else build_tunnel(p, f);
}

/* ---- signals: a post, an arm that drops to clear, a lamp behind its spectacle ---- */
static GMesh g_sigpost, g_sigarm, g_board, g_lamp_r, g_lamp_g;
static float sig_arm[64];              /* the arms' angles, by (segment * NSIG_SEG + i) mod 64 */
static void lineside_init(void) {
    static Mesh m;
    pole_side = hrand(1, 2, 3) < 0.5 ? 1 : -1;
    mb_begin(&m);
    mat(PM_POLE);
    boxc(0, 3.6f, 0, 0.1f, 3.6f, 0.1f);
    mat(PM_WHITE);
    float fin[] = {0, 7.2f, 0.1f, 7.2f, 0.0f, 7.6f};
    lathe_at(v3(0, 0, 0), v3(0, 1, 0), fin, 3, 6);
    mat(PM_IRON);
    boxc(0, 1.8f, 0.4f, 0.03f, 1.8f, 0.03f);    /* the ladder's rail */
    boxc(-0.45f, 5.6f, 0, 0.3f, 0.02f, 0.25f); /* the lamp's platform */
    /* the lamp case behind the arm's spectacle */
    mat(PM_BLACK);
    boxc(-0.25f, 6.4f, 0.12f, 0.12f, 0.15f, 0.12f);
    gm_upload(&g_sigpost, &m);
    mb_begin(&m);
    /* the arm, pivoted at the origin: red with a white stripe, pointing -x, the spectacle at its root */
    mat(PM_RED);
    boxc(-0.7f, 0, -0.03f, 0.65f, 0.12f, 0.02f);
    mat(PM_WHITE);
    boxc(-1.05f, 0, -0.055f, 0.06f, 0.12f, 0.005f);
    mat(PM_IRON);
    boxc(0.15f, -0.2f, -0.03f, 0.18f, 0.28f, 0.015f);
    gm_upload(&g_sigarm, &m);
    /* the lineside board with the time away: white on blue, on two legs */
    mb_begin(&m);
    mat(PM_IRON);
    boxc(-0.7f, 0.9f, 0, 0.04f, 0.9f, 0.04f);
    boxc(0.7f, 0.9f, 0, 0.04f, 0.9f, 0.04f);
    mat(PM_SIGN2);
    mb_quad_uv(v3(1.0f, 1.7f, -0.03f), v3(-1.0f, 1.7f, -0.03f), v3(-1.0f, 2.25f, -0.03f), v3(1.0f, 2.25f, -0.03f), 0, 1, 1, 0);
    mat(PM_BLACK);
    boxc(0, 1.975f, 0.0f, 1.03f, 0.3f, 0.02f);
    gm_upload(&g_board, &m);
    /* the signal lamps: a lens facing down the line toward the train */
    for (int c = 0; c < 2; c++) {
        mb_begin(&m);
        mat(c ? PM_SIG_G : PM_SIG_R);
        M4 s2 = mb->xf;
        push(m4roty((float)M_PI));
        float disc[] = {0, 0, 0.09f, 0, 0.09f, 0.02f};
        mb_lathe(disc, 3, 14, 0, 2 * (float)M_PI);
        mb->xf = s2;
        gm_upload(c ? &g_lamp_g : &g_lamp_r, &m);
    }
}

/* is the signal at z showing clear? It clears when the train is within
 * 1.5 km of it, and goes back to danger once the engine has passed */
static int signal_clear(double z) { return tr.z > z - 1500 && tr.z < z + 2; }

/* the lamps for this frame */
static void lineside_lights(void) {
    /* the nearest set-piece lamps: station lamps warm, the car's headlamps cold */
    for (int k = 0; k < nlamps; k++) {
        double *o = lamps_at[k].o;
        double d2 = (o[0] - cam.p[0]) * (o[0] - cam.p[0]) + (o[2] - cam.p[2]) * (o[2] - cam.p[2]);
        if (d2 > 350.0 * 350.0) continue;
        add_light(o, 2.4f, 2.6f, 1.6f, 0.75f);
    }
}

static void lineside_update(void) {
    /* which pieces are near: build them, and gather their lamps */
    long seg = (long)floor(cam.p[2] / SEG);
    nlamps = 0;
    for (long s = seg - 1; s <= seg + 1; s++)
        for (int k = 0; k < NFEAT; k++) {
            Feat f = feat(s, k);
            if (!f.on) continue;
            double d = fmax(fmax(f.z0 - cam.p[2], cam.p[2] - f.z1), 0);
            if (d < 2500) piece_get(s, k);
        }
    /* collect the lamps of the station and the crossing nearest the camera */
    for (int i = 0; i < MAXPIECE; i++) {
        Piece *p = &pieces[i];
        if (!p->live || (p->kind != F_STATION && p->kind != F_CROSSING)) continue;
        Feat f = feat(p->seg, p->kind);
        if (fmax(fmax(f.z0 - cam.p[2], cam.p[2] - f.z1), 0) > 500) continue;
        if (p->kind == F_STATION) {
            for (int s = -1; s <= 1; s += 2)
                for (int j = 0; j <= 10 && nlamps < 32; j++) {
                    line_pt(f.z0 + 5 + j * 20, s * 4.2, 0.92 + 3.55, lamps_at[nlamps].o);
                    nlamps++;
                }
        } else {
            double zc = (f.z0 + f.z1) / 2;
            for (int a = -1; a <= 1; a += 2) {
                if (nlamps >= 32) break;
                line_pt(zc - 1.2 + a * 0.58, pole_side * 6.6, 0.25, lamps_at[nlamps].o);
                nlamps++;
            }
        }
    }
    /* signals' arms ease toward clear or danger */
    long k0 = (long)floor(cam.p[2] / SEG);
    for (long k = k0 - 1; k <= k0 + 1; k++)
        for (int i = 0; i < NSIG_SEG; i++) {
            int idx = (int)(((k * NSIG_SEG + i) % 64 + 64) % 64);
            float want = signal_clear(signal_z(k, i)) ? 0.7f : 0.0f;
            sig_arm[idx] += (want - sig_arm[idx]) * 0.05f;
        }
}

static void lineside_draw(GLuint p, int shadow) {
    /* track chunks: near ones with sleepers */
    long kc = (long)floor(cam.p[2] / CHUNK);
    int reach = shadow ? 2 : 26;
    for (long k = kc - reach; k <= kc + reach; k++) {
        double dz = fabs((k + 0.5) * CHUNK - cam.p[2]);
        int lod = dz > 500 ? 1 : 0;
        double zc = (k + 0.5) * CHUNK;
        if (!visible((float)(track_x(zc) - cam.p[0]), (float)(track_y(zc) + 3 - cam.p[1]), (float)(zc - cam.p[2]), 42)) continue;
        Chunk *c = chunk_get(k, lod);
        M4 m = m4trans((float)(c->o[0] - cam.p[0]), (float)(c->o[1] - cam.p[1]), (float)(c->o[2] - cam.p[2]));
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
        glUniform4f(UL(p, "uObj"), 1, 1, 0, 0);
        gm_draw(&c->mesh);
    }
    for (int i = 0; i < MAXPIECE; i++) {
        Piece *pc = &pieces[i];
        if (!pc->live || fabs(pc->o[2] - cam.p[2]) > 3000) continue;
        M4 m = m4trans((float)(pc->o[0] - cam.p[0]), (float)(pc->o[1] - cam.p[1]), (float)(pc->o[2] - cam.p[2]));
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, m.m);
        glUniform4f(UL(p, "uObj"), 1, 1, 0, 0);
        gm_draw(&pc->mesh);
    }
    /* signals near the camera */
    long k0 = (long)floor(cam.p[2] / SEG);
    for (long k = k0 - 1; k <= k0 + 1; k++)
        for (int i = 0; i < NSIG_SEG; i++) {
            double z = signal_z(k, i);
            if (fabs(z - cam.p[2]) > 1500 || in_tunnel_z(z) || on_viaduct(z)) continue;
            int idx = (int)(((k * NSIG_SEG + i) % 64 + 64) % 64);
            double b[3];
            int side = -pole_side;
            line_pt(z, side * 2.9, -0.5, b);
            double dx = track_dx(z);
            float yaw = (float)atan(dx);
            M4 base = m4mul(m4trans((float)(b[0] - cam.p[0]), (float)(b[1] - cam.p[1]), (float)(b[2] - cam.p[2])), m4roty(yaw));
            /* mirror so the arm points over the line */
            if (side < 0) base = m4mul(base, m4scale(-1, 1, 1));
            glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, base.m);
            glUniform4f(UL(p, "uObj"), 1, 1, 0, 0);
            gm_draw(&g_sigpost);
            M4 arm = m4mul(base, m4mul(m4trans(0.12f, 6.4f, -0.12f), m4rotz(sig_arm[idx])));
            glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, arm.m);
            gm_draw(&g_sigarm);
            if (!shadow) {
                /* the lamp, seen through the red or green glass */
                int clear = sig_arm[idx] > 0.35f;
                M4 lm = m4mul(base, m4trans(-0.25f, 6.4f, -0.02f));
                glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, lm.m);
                glUniform4f(UL(p, "uObj"), 1, 1, 0, 0);
                gm_draw(clear ? &g_lamp_g : &g_lamp_r);
            }
        }
    /* the board with the time away, a few hundred metres before each station */
    for (long k = k0 - 1; k <= k0 + 1; k++) {
        Feat f = feat(k, F_STATION);
        if (!f.on) continue;
        double z = f.z0 - 420;
        if (fabs(z - cam.p[2]) > 800) continue;
        double b[3];
        line_pt(z, pole_side * 3.3, -0.5, b);
        float yaw = (float)atan(track_dx(z));
        M4 base = m4mul(m4trans((float)(b[0] - cam.p[0]), (float)(b[1] - cam.p[1]), (float)(b[2] - cam.p[2])), m4roty(yaw));
        glUniformMatrix4fv(UL(p, "uModel"), 1, GL_FALSE, base.m);
        glUniform4f(UL(p, "uObj"), 1, 1, 0, 0);
        gm_draw(&g_board);
    }
}
