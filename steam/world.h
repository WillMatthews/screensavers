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

/* ---- trees: at most one to a 9 m cell, in copses and hedgerows, clear of
 * the line, the roads and the water ---- */
#define TREE_G 9.0
typedef struct { double x, y, z; float h, r; int kind; } Tree;
static int tree_cell(long ci, long cj, Tree *t) {
    double wx = (ci + hrand((int32_t)ci, (int32_t)cj, 21)) * TREE_G;
    double wz = (cj + hrand((int32_t)ci, (int32_t)cj, 22)) * TREE_G;
    double dens = fbm_d(wx / 260, wz / 260, 3, 23);
    int hedge = fabs(fmod(wx / 97 + 1000.5 + 0.6 * sin(wz / 140), 1.0) - 0.5) < 0.015 ||
                fabs(fmod(wz / 131 + 1000.5 + 0.6 * sin(wx / 170), 1.0) - 0.5) < 0.012;
    double r = hrand((int32_t)ci, (int32_t)cj, 24);
    double d = fabs(wx - track_x(wz));
    /* fewer near the line: a cleared strip, thinning out */
    double near = sstep(14, 60, d);
    if (!(dens > 0.56 + 0.25 * r + 0.2 * (1 - near) || (hedge && r < 0.7 * near + 0.1))) return 0;
    if (d < 14 || (in_feat(wz, F_STATION, 80) && d < 30) || in_feat(wz, F_CROSSING, 30)) return 0;
    double h = terrain_h(wx, wz);
    if (h < track_y(wz) - 30) return 0;
    double s = 0.75 + 0.5 * hrand((int32_t)ci, (int32_t)cj, 25);
    double q = hrand((int32_t)ci, (int32_t)cj, 26);
    int kind;
    if (hedge && dens <= 0.56 + 0.25 * r) kind = q < 0.7 ? 4 : 2;
    else if (fbm_d(wx / 600, wz / 600, 2, 28) > 0.55) kind = q < 0.8 ? 0 : 1;
    else kind = q < 0.55 ? 2 : 3;
    static const float hgt[5] = {18, 15, 14, 18, 4.5f};
    t->x = wx; t->z = wz; t->y = h; t->h = (float)(hgt[kind] * s); t->r = (float)r; t->kind = kind;
    return 1;
}
/* is the view from a to b blocked by a tree or the ground (ignoring its last 6 m)? */
static int sight_blocked(const double *a, const double *b) {
    double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (double s = 1.5; s < L - 6; s += 2.5) {
        double p[3] = {a[0] + d[0] * s / L, a[1] + d[1] * s / L, a[2] + d[2] * s / L};
        if (p[1] < terrain_h(p[0], p[2]) + 0.2) return 1;
        long ci = (long)floor(p[0] / TREE_G), cj = (long)floor(p[2] / TREE_G);
        for (long j = cj - 1; j <= cj + 1; j++)
            for (long i = ci - 1; i <= ci + 1; i++) {
                Tree t;
                if (!tree_cell(i, j, &t)) continue;
                double dx = p[0] - t.x, dz = p[2] - t.z;
                float cr = t.kind == 4 ? 0.4f * t.h : 0.3f * t.h;
                if (dx * dx + dz * dz < (cr + 0.8) * (cr + 0.8) && p[1] < t.y + t.h + 0.5) return 1;
            }
    }
    return 0;
}

/* ---- terrain tiles: a quadtree of 32 x 32 cell patches round the camera ----
 * Built by a worker thread (a tile and its trees take a few milliseconds);
 * while a finer tile is on its way the coarser one stands in. */
#define TCELLS 32
#define TN (TCELLS + 1)
#define TROOT 4096.0
#define TLEAF 128.0
#define MAXTILE 400
#define MAXTREE 4096
typedef struct {
    int level; long ix, iz;     /* tile size TROOT / 2^level; index in that grid */
    double ox, oz;              /* world origin (corner) */
    GMesh mesh;
    GLuint tree_vbo; int ntree, tcount[5], toff[5];
    float hmin, hmax;
    unsigned used;
    int live;
} Tile;
static Tile tiles[MAXTILE];
static unsigned tile_clock;

/* what a tile is made of, on the CPU */
typedef struct {
    float h[(TN + 2) * (TN + 2)];
    float v[(TN * TN + 4 * TN) * VF];
    unsigned short idx[TCELLS * TCELLS * 6 + 4 * TCELLS * 6];
    int nv, ni;
    float trees[8 * MAXTREE];
    int ntree, tcount[5], toff[5];
    float hmin, hmax;
} TileData;

static void tile_compute(int level, long ix, long iz, TileData *d) {
    double size = TROOT / (1 << level), cs = size / TCELLS;
    double ox = ix * size, oz = iz * size;
    int N = TN;
#define TH(i, j) d->h[((j) + 1) * (N + 2) + (i) + 1]
    /* heights with a one-cell border for the normals */
    for (int j = -1; j <= N; j++)
        for (int i = -1; i <= N; i++) TH(i, j) = (float)terrain_h(ox + i * cs, oz + j * cs);
    float skirt = (float)(cs * 1.5 + 2);
    d->hmin = 1e9f; d->hmax = -1e9f;
    d->nv = 0;
    /* the grid's vertices, then the skirt's (each edge vertex again, lowered) */
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            float h = TH(i, j);
            if (h < d->hmin) d->hmin = h;
            if (h > d->hmax) d->hmax = h;
            V3 n = vnorm(v3(TH(i - 1, j) - TH(i + 1, j), (float)(2 * cs), TH(i, j - 1) - TH(i, j + 1)));
            float *o = d->v + d->nv++ * VF;
            o[0] = (float)(i * cs); o[1] = h; o[2] = (float)(j * cs); o[3] = n.x; o[4] = n.y; o[5] = n.z; o[6] = o[7] = o[8] = 0;
        }
    int sk0 = N * N;
    for (int e = 0; e < 4; e++)
        for (int k = 0; k < N; k++) {
            int i = e == 0 ? k : e == 1 ? TCELLS : e == 2 ? k : 0;
            int j = e == 0 ? 0 : e == 1 ? k : e == 2 ? TCELLS : k;
            float *o = d->v + d->nv++ * VF;
            o[0] = (float)(i * cs); o[1] = TH(i, j) - skirt; o[2] = (float)(j * cs); o[3] = 0; o[4] = 1; o[5] = 0; o[6] = o[7] = o[8] = 0;
        }
    int ni = 0;
    for (int j = 0; j < TCELLS; j++)
        for (int i = 0; i < TCELLS; i++) {
            unsigned short a0 = (unsigned short)(j * N + i), a1 = (unsigned short)(a0 + 1), a2 = (unsigned short)(a0 + N + 1), a3 = (unsigned short)(a0 + N);
            d->idx[ni++] = a0; d->idx[ni++] = a2; d->idx[ni++] = a1;
            d->idx[ni++] = a0; d->idx[ni++] = a3; d->idx[ni++] = a2;
        }
    for (int e = 0; e < 4; e++)
        for (int k = 0; k < TCELLS; k++) {
            int i0 = e == 0 ? k : e == 1 ? TCELLS : e == 2 ? k : 0, j0 = e == 0 ? 0 : e == 1 ? k : e == 2 ? TCELLS : k;
            int i1 = e == 0 ? k + 1 : e == 1 ? TCELLS : e == 2 ? k + 1 : 0, j1 = e == 0 ? 0 : e == 1 ? k + 1 : e == 2 ? TCELLS : k + 1;
            unsigned short A = (unsigned short)(j0 * N + i0), B = (unsigned short)(j1 * N + i1);
            unsigned short A2 = (unsigned short)(sk0 + e * N + k), B2 = (unsigned short)(sk0 + e * N + k + 1);
            d->idx[ni++] = A; d->idx[ni++] = B; d->idx[ni++] = B2;
            d->idx[ni++] = A; d->idx[ni++] = B2; d->idx[ni++] = A2;
        }
    d->ni = ni;

    /* trees: one to a cell of the global grid (so they keep their places as
     * tiles change detail; coarser tiles keep fewer of them), stood on this
     * tile's mesh, which at a coarse level isn't quite the true ground */
    d->ntree = 0;
    for (int k = 0; k < 5; k++) d->tcount[k] = d->toff[k] = 0;
    if (level < 3) return;
    static __thread float tv[8 * MAXTREE];
    int n = 0;
    int cells = (int)(size / TREE_G);
    float keep = level >= 5 ? 1.0f : level == 4 ? 0.4f : 0.12f;
    long ci0 = (long)floor(ox / TREE_G), cj0 = (long)floor(oz / TREE_G);
    for (int j = 0; j < cells && n < MAXTREE; j++)
        for (int i = 0; i < cells && n < MAXTREE; i++) {
            Tree t0;
            if (!tree_cell(ci0 + i, cj0 + j, &t0) || t0.r > keep) continue;
            float fx = (float)((t0.x - ox) / cs), fz = (float)((t0.z - oz) / cs);
            int gx = (int)fx, gz = (int)fz;
            if (gx < 0 || gz < 0 || gx >= TCELLS || gz >= TCELLS) continue;
            fx -= gx; fz -= gz;
            float h00 = TH(gx, gz), h10 = TH(gx + 1, gz), h01 = TH(gx, gz + 1), h11 = TH(gx + 1, gz + 1);
            float gy = fx > fz ? h00 + (h10 - h00) * fx + (h11 - h10) * fz : h00 + (h11 - h01) * fx + (h01 - h00) * fz;
            float *o = tv + n * 8;
            o[0] = (float)(t0.x - ox); o[1] = gy - 0.3f; o[2] = (float)(t0.z - oz);
            o[3] = t0.h; o[4] = (float)t0.kind; o[5] = t0.r * 6.28f;
            o[6] = (float)hrand((int32_t)t0.x, (int32_t)t0.z, 27); o[7] = 0;
            n++;
        }
#undef TH
    /* grouped by kind, for one draw a kind */
    int m = 0;
    for (int k = 0; k < 5; k++) {
        d->toff[k] = m;
        for (int i = 0; i < n; i++)
            if ((int)tv[i * 8 + 4] == k) { memcpy(d->trees + m * 8, tv + i * 8, sizeof(float) * 8); m++; }
        d->tcount[k] = m - d->toff[k];
    }
    d->ntree = n;
}

static Tile *tile_find(int level, long ix, long iz) {
    for (int i = 0; i < MAXTILE; i++)
        if (tiles[i].live && tiles[i].level == level && tiles[i].ix == ix && tiles[i].iz == iz) return &tiles[i];
    return NULL;
}
static Tile *tile_upload(int level, long ix, long iz, const TileData *d) {
    Tile *free_t = NULL, *oldest = NULL;
    for (int i = 0; i < MAXTILE; i++) {
        Tile *t = &tiles[i];
        if (!t->live) { if (!free_t) free_t = t; }
        else if (t->used != tile_clock && (!oldest || t->used < oldest->used)) oldest = t;
    }
    Tile *t = free_t ? free_t : oldest;
    if (!t) return NULL;
    t->live = 1; t->level = level; t->ix = ix; t->iz = iz;
    double size = TROOT / (1 << level);
    t->ox = ix * size; t->oz = iz * size;
    t->used = tile_clock;
    t->hmin = d->hmin; t->hmax = d->hmax;
    Mesh m = {(float *)d->v, d->nv * VF, d->nv * VF, m4id(), 0};
    gm_upload_indexed(&t->mesh, &m, d->idx, d->ni);
    if (!t->tree_vbo) glGenBuffers(1, &t->tree_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, t->tree_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * (size_t)(d->ntree ? d->ntree : 1), d->trees, GL_STATIC_DRAW);
    t->ntree = d->ntree;
    for (int k = 0; k < 5; k++) { t->tcount[k] = d->tcount[k]; t->toff[k] = d->toff[k]; }
    return t;
}

/* the worker and its queue */
#define MAXJOB 48
typedef struct { int level; long ix, iz; int state; unsigned asked; TileData *d; } Job;   /* state: 0 free, 1 queued, 2 working, 3 done */
static Job jobs[MAXJOB];
static pthread_mutex_t jmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jcv = PTHREAD_COND_INITIALIZER;
static int tiles_threaded;
static void *tile_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&jmu);
        Job *j = NULL;
        for (;;) {
            /* the most recently wanted, finest first */
            for (int i = 0; i < MAXJOB; i++)
                if (jobs[i].state == 1 && (!j || jobs[i].asked > j->asked || (jobs[i].asked == j->asked && jobs[i].level > j->level))) j = &jobs[i];
            if (j) break;
            pthread_cond_wait(&jcv, &jmu);
        }
        j->state = 2;
        int level = j->level;
        long ix = j->ix, iz = j->iz;
        TileData *d = j->d;
        pthread_mutex_unlock(&jmu);
        tile_compute(level, ix, iz, d);
        pthread_mutex_lock(&jmu);
        j->state = 3;
        pthread_mutex_unlock(&jmu);
    }
    return NULL;
}
static void tiles_start_worker(void) {
    for (int i = 0; i < MAXJOB; i++) jobs[i].d = malloc(sizeof(TileData));
    pthread_t th;
    if (pthread_create(&th, NULL, tile_worker, NULL) == 0) { pthread_detach(th); tiles_threaded = 1; }
}

/* the tile, if it's ready; sync: build it now if it isn't */
static Tile *tile_want(int level, long ix, long iz, int sync) {
    Tile *t = tile_find(level, ix, iz);
    if (t) { t->used = tile_clock; return t; }
    if (sync || !tiles_threaded) {
        static TileData *d;
        if (!d) d = malloc(sizeof(TileData));
        tile_compute(level, ix, iz, d);
        return tile_upload(level, ix, iz, d);
    }
    pthread_mutex_lock(&jmu);
    Job *j = NULL, *spare = NULL;
    for (int i = 0; i < MAXJOB; i++) {
        Job *q = &jobs[i];
        if (q->state && q->level == level && q->ix == ix && q->iz == iz) { j = q; break; }
        if (!q->state) { if (!spare || spare->state) spare = q; }
        else if (q->state == 1 && (!spare || (spare->state == 1 && q->asked < spare->asked))) spare = q;
    }
    if (j && j->state == 3) {
        pthread_mutex_unlock(&jmu);
        t = tile_upload(level, ix, iz, j->d);
        pthread_mutex_lock(&jmu);
        j->state = 0;
        pthread_mutex_unlock(&jmu);
        return t;
    }
    if (j) j->asked = tile_clock;
    else if (spare && (spare->state == 0 || spare->asked < tile_clock)) {
        spare->level = level; spare->ix = ix; spare->iz = iz; spare->asked = tile_clock; spare->state = 1;
        pthread_cond_signal(&jcv);
    }
    pthread_mutex_unlock(&jmu);
    return NULL;
}

/* which tiles to draw from the camera at (cx, cz): a tile is split while
 * the camera is within 1.6 of its size of its centre, once its children are ready */
#define MAXDRAW 400
static Tile *drawlist[MAXDRAW];
static int ndraw, tiles_sync;
static void tiles_visit(int level, long ix, long iz, double cx, double cz, double far) {
    double size = TROOT / (1 << level);
    double mx = (ix + 0.5) * size, mz = (iz + 0.5) * size;
    double dx = fmax(fabs(cx - mx) - size / 2, 0), dz = fmax(fabs(cz - mz) - size / 2, 0);
    double d = sqrt(dx * dx + dz * dz);
    if (d > far) return;
    if (size > TLEAF && d < size * 1.6) {
        int ready = 1;
        for (int k = 0; k < 4; k++)
            if (!tile_want(level + 1, ix * 2 + (k & 1), iz * 2 + (k >> 1), tiles_sync)) ready = 0;
        if (ready) {
            for (int k = 0; k < 4; k++) tiles_visit(level + 1, ix * 2 + (k & 1), iz * 2 + (k >> 1), cx, cz, far);
            return;
        }
    }
    Tile *t = tile_want(level, ix, iz, tiles_sync || level == 0);
    if (t && ndraw < MAXDRAW) drawlist[ndraw++] = t;
}
static void tiles_update(double cx, double cz, double far, int sync) {
    tile_clock++;
    ndraw = 0;
    tiles_sync = sync;
    long r = (long)ceil(far / TROOT) + 1;
    long bx = (long)floor(cx / TROOT), bz = (long)floor(cz / TROOT);
    for (long j = bz - r; j <= bz + r; j++)
        for (long i = bx - r; i <= bx + r; i++) tiles_visit(0, i, j, cx, cz, far);
}
