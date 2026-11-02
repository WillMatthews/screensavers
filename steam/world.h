/* steam: the world the train runs through. Included once, by steam.c.
 *
 * The line runs roughly along +z. Its centre wanders in x and climbs and
 * falls gently in y, both as sums of sines of z, so any point of it can be
 * had directly. Every SEG metres the route repeats its pattern of set
 * pieces (a station, a level crossing, a viaduct over a misty valley, a
 * tunnel through a ridge, signals), each segment varied by the seed.
 * The terrain is natural hills, cut and embanked to carry the line, with
 * the valleys and ridges the set pieces need. Everything here is in double
 * precision; the renderer draws relative to a floating origin. */

static uint32_t wseed;

static uint32_t ihash(int32_t x, int32_t z, uint32_t salt) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)z * 0xd8163841u ^ (salt + wseed) * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
static double hrand(int32_t x, int32_t z, uint32_t salt) { return (ihash(x, z, salt) & 0xffffff) / (double)0x1000000; }
static double smooth_d(double t) { return t * t * (3 - 2 * t); }
static double sstep(double a, double b, double x) {
    double t = (x - a) / (b - a);
    return smooth_d(t < 0 ? 0 : t > 1 ? 1 : t);
}
static double vnoise_d(double x, double z, uint32_t salt) {
    double fx = floor(x), fz = floor(z);
    int32_t ix = (int32_t)fx, iz = (int32_t)fz;
    double tx = smooth_d(x - fx), tz = smooth_d(z - fz);
    double a = hrand(ix, iz, salt), b = hrand(ix + 1, iz, salt), c = hrand(ix, iz + 1, salt), d = hrand(ix + 1, iz + 1, salt);
    return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
}
static double fbm_d(double x, double z, int oct, uint32_t salt) {
    double s = 0, amp = 0.5, norm = 0;
    for (int i = 0; i < oct; i++) {
        s += vnoise_d(x, z, salt + (uint32_t)i) * amp;
        norm += amp; amp *= 0.5; x = x * 2.03 + 3.1; z = z * 2.03 + 1.7;
    }
    return s / norm;
}
static double ridged_d(double x, double z, int oct, uint32_t salt) {
    double s = 0, amp = 0.5, norm = 0;
    for (int i = 0; i < oct; i++) {
        double n = 1 - fabs(vnoise_d(x, z, salt + (uint32_t)i) * 2 - 1);
        s += n * n * amp;
        norm += amp; amp *= 0.5; x = x * 2.07 + 5.3; z = z * 2.07 + 2.9;
    }
    return s / norm;
}

/* ---- the line ---- */
static double tph[6];
static double track_x(double z) {
    return 70 * sin(z / 830 + tph[0]) + 34 * sin(z / 377 + tph[1]) + 9 * sin(z / 171 + tph[2]);
}
static double track_dx(double z) {
    return 70 / 830.0 * cos(z / 830 + tph[0]) + 34 / 377.0 * cos(z / 377 + tph[1]) + 9 / 171.0 * cos(z / 171 + tph[2]);
}
static double track_ddx(double z) {
    return -70 / (830.0 * 830) * sin(z / 830 + tph[0]) - 34 / (377.0 * 377) * sin(z / 377 + tph[1]) - 9 / (171.0 * 171) * sin(z / 171 + tph[2]);
}
static double track_y(double z) { return 14 * sin(z / 1900 + tph[3]) + 5 * sin(z / 640 + tph[4]); }
static double track_dy(double z) { return 14 / 1900.0 * cos(z / 1900 + tph[3]) + 5 / 640.0 * cos(z / 640 + tph[4]); }

/* ---- the set pieces ---- */
#define SEG 5600.0
enum { F_STATION, F_CROSSING, F_VIADUCT, F_TUNNEL, NFEAT };
typedef struct { double z0, z1; int on; } Feat;
/* where each set piece of segment k lies (on = 0: this segment does without) */
static Feat feat(long k, int kind) {
    double b = k * SEG;
    Feat f = {0, 0, 1};
    switch (kind) {
    case F_STATION: f.z0 = b + 700; f.z1 = f.z0 + 210; f.on = hrand((int32_t)k, 1, 901) < 0.85; break;
    case F_CROSSING: f.z0 = b + 1750; f.z1 = f.z0 + 8; f.on = hrand((int32_t)k, 2, 902) < 0.8; break;
    case F_VIADUCT: f.z0 = b + 2700 + 200 * hrand((int32_t)k, 3, 903); f.z1 = f.z0 + 240 + 120 * hrand((int32_t)k, 4, 904); break;
    default: f.z0 = b + 4100 + 300 * hrand((int32_t)k, 5, 905); f.z1 = f.z0 + 380 + 260 * hrand((int32_t)k, 6, 906);
             f.on = hrand((int32_t)k, 7, 907) < 0.85; break;
    }
    if (k < 0) f.on = 0;
    return f;
}
/* the set piece of a kind nearest to z (its segment, or the next), and its weight */
static Feat feat_near(double z, int kind) {
    long k = (long)floor(z / SEG);
    Feat a = feat(k, kind), b = feat(k + 1, kind), c = feat(k - 1, kind);
    double da = fmin(fabs(z - a.z0), fabs(z - a.z1)), db = fmin(fabs(z - b.z0), fabs(z - b.z1)), dc = fmin(fabs(z - c.z0), fabs(z - c.z1));
    if (!a.on) da = 1e18;
    if (!b.on) db = 1e18;
    if (!c.on) dc = 1e18;
    return da <= db && da <= dc ? a : db <= dc ? b : c;
}
static int in_feat(double z, int kind, double margin) {
    Feat f = feat_near(z, kind);
    return f.on && z > f.z0 - margin && z < f.z1 + margin;
}
/* signals: one before each station and tunnel, and every 1300 m between */
#define NSIG_SEG 6
static double signal_z(long k, int i) {
    double b = k * SEG;
    static const double at[NSIG_SEG] = {80, 560, 1400, 2400, 3500, 3950};
    return b + at[i];
}

/* ---- the land ---- */
static double land(double x, double z) {
    double d = fabs(x - track_x(z));
    double ty = track_y(z);
    /* rolling farmland near the line, hills behind, and moors and peaks far off */
    double n = fbm_d(x / 700, z / 700, 5, 11);
    double h = ty - 6 + (n - 0.45) * 60 * sstep(20, 500, d);
    h += (fbm_d(x / 140, z / 140, 3, 12) - 0.5) * 9;
    h += 210 * ridged_d(x / 2600, z / 2600, 5, 13) * sstep(500, 2600, d);
    /* valleys under the viaducts, ridges over the tunnels: across the line */
    Feat v = feat_near(z, F_VIADUCT);
    if (v.on) {
        double mid = (v.z0 + v.z1) / 2, half = (v.z1 - v.z0) / 2;
        double mz = z + 30 * sin(x / 160);              /* the river meanders */
        double w = exp(-pow((mz - mid) / (half * 0.85), 2));
        h = h * (1 - w) + (ty - 34 + 6 * sstep(0, 600, d) - 8) * w + 0.25 * (h - ty) * w * sstep(150, 900, d);
    }
    Feat t = feat_near(z, F_TUNNEL);
    if (t.on) {
        double mid = (t.z0 + t.z1) / 2, half = (t.z1 - t.z0) / 2;
        double w = sstep(half + 160, half - 40, fabs(z - mid));
        h += (ty + 34 + 30 * fbm_d(x / 300, z / 300, 3, 14) - h) * w * sstep(900, 100, d);
    }
    return h;
}
/* the formation: cut and embanked to the line's level, flat and wider at stations */
static double terrain_h(double x, double z) {
    double h = land(x, z);
    double d = fabs(x - track_x(z));
    double form = track_y(z) - 0.55;
    double w = 4.0;
    if (in_feat(z, F_STATION, 60)) {
        Feat s = feat_near(z, F_STATION);
        w += 14 * sstep(s.z0 - 60, s.z0, z) * sstep(s.z1 + 60, s.z1, z);
    }
    if (in_feat(z, F_CROSSING, 40)) w += 18;
    Feat t = feat_near(z, F_TUNNEL);
    if (t.on && z > t.z0 && z < t.z1) return fmax(h, track_y(z) + 8.5);
    Feat v = feat_near(z, F_VIADUCT);
    if (v.on && z > v.z0 + 6 && z < v.z1 - 6 && d < 40) return fmin(h, form - 6);
    if (d < w) return form;
    double up = form + (d - w) * 0.75, dn = form - (d - w) * 0.55;      /* cuttings steep, banks gentler */
    return h > up ? up : h < dn ? dn : h;
}

/* ---- terrain tiles: a quadtree of 32 x 32 cell patches round the camera ---- */
#define TCELLS 32
#define TROOT 4096.0
#define TLEAF 128.0
#define MAXTILE 400
typedef struct {
    int level; long ix, iz;     /* tile size TROOT / 2^level; index in that grid */
    double ox, oz;              /* world origin (corner) */
    GMesh mesh;
    GLuint tree_vbo; int ntree;
    float hmin, hmax;
    unsigned used;
    int live;
} Tile;
static Tile tiles[MAXTILE];
static unsigned tile_clock;

static float tbuf_h[(TCELLS + 3) * (TCELLS + 3)];
static void tile_build(Tile *t) {
    double size = TROOT / (1 << t->level), cs = size / TCELLS;
    int N = TCELLS + 1;
    /* heights with a one-cell border for the normals */
    for (int j = -1; j <= N; j++)
        for (int i = -1; i <= N; i++)
            tbuf_h[(j + 1) * (N + 2) + (i + 1)] = (float)terrain_h(t->ox + i * cs, t->oz + j * cs);
    static Mesh m;
    static unsigned short idx[TCELLS * TCELLS * 6 + 4 * TCELLS * 6];
    mb_begin(&m);
    m.mat = 0;
    float skirt = (float)(cs * 1.5 + 2);
    t->hmin = 1e9f; t->hmax = -1e9f;
#define TH(i, j) tbuf_h[((j) + 1) * (N + 2) + (i) + 1]
    /* the grid's vertices, then the skirt's (each edge vertex again, lowered) */
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            float h = TH(i, j);
            if (h < t->hmin) t->hmin = h;
            if (h > t->hmax) t->hmax = h;
            mb_vert(v3((float)(i * cs), h, (float)(j * cs)), vnorm(v3(TH(i - 1, j) - TH(i + 1, j), (float)(2 * cs), TH(i, j - 1) - TH(i, j + 1))), 0, 0);
        }
    int sk0 = N * N;
    for (int e = 0; e < 4; e++)
        for (int k = 0; k < N; k++) {
            int i = e == 0 ? k : e == 1 ? TCELLS : e == 2 ? k : 0;
            int j = e == 0 ? 0 : e == 1 ? k : e == 2 ? TCELLS : k;
            mb_vert(v3((float)(i * cs), TH(i, j) - skirt, (float)(j * cs)), v3(0, 1, 0), 0, 0);
        }
    int ni = 0;
    for (int j = 0; j < TCELLS; j++)
        for (int i = 0; i < TCELLS; i++) {
            unsigned short a0 = (unsigned short)(j * N + i), a1 = (unsigned short)(a0 + 1), a2 = (unsigned short)(a0 + N + 1), a3 = (unsigned short)(a0 + N);
            idx[ni++] = a0; idx[ni++] = a2; idx[ni++] = a1;
            idx[ni++] = a0; idx[ni++] = a3; idx[ni++] = a2;
        }
    for (int e = 0; e < 4; e++)
        for (int k = 0; k < TCELLS; k++) {
            int i0 = e == 0 ? k : e == 1 ? TCELLS : e == 2 ? k : 0, j0 = e == 0 ? 0 : e == 1 ? k : e == 2 ? TCELLS : k;
            int i1 = e == 0 ? k + 1 : e == 1 ? TCELLS : e == 2 ? k + 1 : 0, j1 = e == 0 ? 0 : e == 1 ? k + 1 : e == 2 ? TCELLS : k + 1;
            unsigned short A = (unsigned short)(j0 * N + i0), B = (unsigned short)(j1 * N + i1);
            unsigned short A2 = (unsigned short)(sk0 + e * N + k), B2 = (unsigned short)(sk0 + e * N + k + 1);
            idx[ni++] = A; idx[ni++] = B; idx[ni++] = B2;
            idx[ni++] = A; idx[ni++] = B2; idx[ni++] = A2;
        }
#undef TH
    gm_upload_indexed(&t->mesh, &m, idx, ni);

    /* trees: on a jittered grid, in copses and hedgerows, clear of the line, the roads and the water */
    t->ntree = 0;
    if (t->level >= 3) {
        static float tv[8 * 4096];
        double g = t->level >= 5 ? 9 : t->level == 4 ? 14 : 22;
        int n = 0;
        int cells = (int)(size / g);
        for (int j = 0; j < cells && n < 4096; j++)
            for (int i = 0; i < cells && n < 4096; i++) {
                double wx = t->ox + (i + hrand((int32_t)(t->ox / g) + i, (int32_t)(t->oz / g) + j, 21)) * g;
                double wz = t->oz + (j + hrand((int32_t)(t->ox / g) + i, (int32_t)(t->oz / g) + j, 22)) * g;
                double dens = fbm_d(wx / 260, wz / 260, 3, 23);
                double hedge = fabs(fmod(wx / 97 + 1000.5 + 0.6 * sin(wz / 140), 1.0) - 0.5) < 0.015 ||
                               fabs(fmod(wz / 131 + 1000.5 + 0.6 * sin(wx / 170), 1.0) - 0.5) < 0.012;
                double r = hrand((int32_t)(wx * 3), (int32_t)(wz * 3), 24);
                if (!(dens > 0.56 + 0.25 * r || (hedge && r < 0.7))) continue;
                double d = fabs(wx - track_x(wz));
                if (d < 9 || (in_feat(wz, F_STATION, 80) && d < 30) || (in_feat(wz, F_CROSSING, 30))) continue;
                double h = terrain_h(wx, wz);
                if (h < track_y(wz) - 30) continue;   /* the river */
                /* thin out: only what this level's cell size can afford */
                if (t->level < 5 && r > (t->level == 4 ? 0.7 : 0.45)) continue;
                double s = 0.7 + 0.6 * hrand((int32_t)(wx * 7), (int32_t)(wz * 7), 25);
                double kind = hrand((int32_t)(wx * 5), (int32_t)(wz * 5), 26) < 0.3 ? 1 : 0;
                float *o = tv + n * 8;
                o[0] = (float)(wx - t->ox); o[1] = (float)h; o[2] = (float)(wz - t->oz);
                o[3] = (float)((kind ? 11 : 9) * s); o[4] = (float)kind; o[5] = (float)(r * 6.28);
                o[6] = (float)hrand((int32_t)wx, (int32_t)wz, 27); o[7] = 0;
                n++;
            }
        if (!t->tree_vbo) glGenBuffers(1, &t->tree_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, t->tree_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * (size_t)(n ? n : 1), tv, GL_STATIC_DRAW);
        t->ntree = n;
    }
}

static Tile *tile_get(int level, long ix, long iz) {
    Tile *free_t = NULL, *oldest = NULL;
    for (int i = 0; i < MAXTILE; i++) {
        Tile *t = &tiles[i];
        if (t->live && t->level == level && t->ix == ix && t->iz == iz) { t->used = tile_clock; return t; }
        if (!t->live) { if (!free_t) free_t = t; }
        else if (!oldest || t->used < oldest->used) oldest = t;
    }
    Tile *t = free_t ? free_t : oldest;
    t->live = 1; t->level = level; t->ix = ix; t->iz = iz;
    double size = TROOT / (1 << level);
    t->ox = ix * size; t->oz = iz * size;
    t->used = tile_clock;
    tile_build(t);
    return t;
}

/* which tiles to draw from the camera at (cx, cz): a tile is split while
 * the camera is within 1.6 of its size of its centre */
#define MAXDRAW 400
static Tile *drawlist[MAXDRAW];
static int ndraw, nbuilt, build_budget;
static void tiles_visit(int level, long ix, long iz, double cx, double cz, double far) {
    double size = TROOT / (1 << level);
    double mx = (ix + 0.5) * size, mz = (iz + 0.5) * size;
    double dx = fmax(fabs(cx - mx) - size / 2, 0), dz = fmax(fabs(cz - mz) - size / 2, 0);
    double d = sqrt(dx * dx + dz * dz);
    if (d > far) return;
    if (size > TLEAF && d < size * 1.6) {
        /* only split once every child exists or can be made this frame */
        int ready = 1;
        for (int k = 0; k < 4 && ready; k++) {
            int l = level + 1;
            long cx2 = ix * 2 + (k & 1), cz2 = iz * 2 + (k >> 1);
            int found = 0;
            for (int i = 0; i < MAXTILE; i++)
                if (tiles[i].live && tiles[i].level == l && tiles[i].ix == cx2 && tiles[i].iz == cz2) { found = 1; break; }
            if (!found) ready = 0;
        }
        if (ready || nbuilt + 4 <= build_budget) {
            for (int k = 0; k < 4; k++) tiles_visit(level + 1, ix * 2 + (k & 1), iz * 2 + (k >> 1), cx, cz, far);
            return;
        }
    }
    int existed = 0;
    for (int i = 0; i < MAXTILE; i++)
        if (tiles[i].live && tiles[i].level == level && tiles[i].ix == ix && tiles[i].iz == iz) { existed = 1; break; }
    if (!existed) nbuilt++;
    if (ndraw < MAXDRAW) drawlist[ndraw++] = tile_get(level, ix, iz);
}
static void tiles_update(double cx, double cz, double far, int budget) {
    tile_clock++;
    ndraw = 0; nbuilt = 0; build_budget = budget;
    long r = (long)ceil(far / TROOT) + 1;
    long bx = (long)floor(cx / TROOT), bz = (long)floor(cz / TROOT);
    for (long j = bz - r; j <= bz + r; j++)
        for (long i = bx - r; i <= bx + r; i++) tiles_visit(0, i, j, cx, cz, far);
}
