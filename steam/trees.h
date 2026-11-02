/* steam: trees, a few kinds, unit height, instanced per terrain tile.
 * Included once, by steam.c.
 *
 *   0  a spruce: tiers of ragged drooping skirts
 *   1  a pine: a bare trunk under a flat-topped crown
 *   2  an oak: a broad crown of lumps
 *   3  an ash: taller, looser
 *   4  a hawthorn or bush, for the hedgerows
 *
 * A crown's lumps take their normals half from the crown as a whole, so the
 * moon lights it as one soft mass with a broken edge. */

#define NTREEKIND 5
static int tree_first[NTREEKIND], tree_count[NTREEKIND];
static GMesh g_tree;
static GLuint vao_tree;

static float tnoise3(V3 p, int salt) {
    return (float)(vnoise_d(p.x * 1.7 + p.z * 0.6, p.y * 1.7 - p.z * 1.3, (uint32_t)salt) * 0.65 +
                   vnoise_d(p.x * 4.1 - p.y * 0.7, p.z * 4.1 + p.y * 1.1, (uint32_t)salt + 7) * 0.35);
}

/* a lumpy ball: a twice-subdivided octahedron, displaced; normals bent toward the crown's centre cc */
static void blob(V3 c, V3 rad, V3 cc, float bend, int salt) {
    static const V3 oct[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const int f[8][3] = {{0, 2, 4}, {4, 2, 1}, {1, 2, 5}, {5, 2, 0}, {0, 4, 3}, {4, 1, 3}, {1, 5, 3}, {5, 0, 3}};
    for (int k = 0; k < 8; k++) {
        V3 A = oct[f[k][0]], B = oct[f[k][1]], C = oct[f[k][2]];
        int n = 4;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n - i; j++) {
                /* barycentric sub-triangles */
                for (int up = 0; up < 2; up++) {
                    if (up && j == n - i - 1) continue;
                    float a[3][2];
                    if (!up) { a[0][0] = (float)i; a[0][1] = (float)j; a[1][0] = (float)i + 1; a[1][1] = (float)j; a[2][0] = (float)i; a[2][1] = (float)j + 1; }
                    else { a[0][0] = (float)i + 1; a[0][1] = (float)j; a[1][0] = (float)i + 1; a[1][1] = (float)j + 1; a[2][0] = (float)i; a[2][1] = (float)j + 1; }
                    V3 P[3], N[3];
                    for (int q = 0; q < 3; q++) {
                        float u = a[q][0] / n, v = a[q][1] / n;
                        V3 d = vnorm(vadd(vadd(vmul(A, 1 - u - v), vmul(B, u)), vmul(C, v)));
                        float disp = 0.78f + 0.42f * tnoise3(vadd(vmul(d, 2.2f), v3((float)salt, 0, 0)), salt);
                        P[q] = vadd(c, v3(d.x * rad.x * disp, d.y * rad.y * disp, d.z * rad.z * disp));
                        N[q] = vnorm(vadd(vmul(d, 1 - bend), vmul(vnorm(vsub(P[q], cc)), bend)));
                    }
                    mb_vert(P[0], N[0], 0, 0); mb_vert(P[1], N[1], 0, 0); mb_vert(P[2], N[2], 0, 0);
                }
            }
    }
}
static void trunk(float r0, float r1, float h, int n) {
    for (int i = 0; i < n; i++) {
        float a0 = 2 * (float)M_PI * i / n, a1 = 2 * (float)M_PI * (i + 1) / n;
        V3 A = v3(r0 * cosf(a0), 0, r0 * sinf(a0)), B = v3(r0 * cosf(a1), 0, r0 * sinf(a1));
        V3 C = v3(r1 * cosf(a1), h, r1 * sinf(a1)), D = v3(r1 * cosf(a0), h, r1 * sinf(a0));
        V3 na = v3(cosf(a0), 0, sinf(a0)), nb = v3(cosf(a1), 0, sinf(a1));
        mb_vert(A, na, 0, 0); mb_vert(C, nb, 0, 0); mb_vert(B, nb, 0, 0);
        mb_vert(A, na, 0, 0); mb_vert(D, na, 0, 0); mb_vert(C, nb, 0, 0);
    }
}
/* a drooping skirt of branches: a ragged cone from (0, ytop) out to radius r at ybot */
static void skirt(float ytop, float ybot, float r, int n, int salt) {
    for (int i = 0; i < n; i++) {
        float a0 = 2 * (float)M_PI * i / n, a1 = 2 * (float)M_PI * (i + 1) / n;
        float r0 = r * (0.7f + 0.5f * (float)hrand(i, salt, 61)), r1 = r * (0.7f + 0.5f * (float)hrand((i + 1) % n, salt, 61));
        float d0 = (ybot - ytop) * (0.85f + 0.3f * (float)hrand(i, salt, 62)), d1 = (ybot - ytop) * (0.85f + 0.3f * (float)hrand((i + 1) % n, salt, 62));
        V3 T = v3(0, ytop, 0);
        V3 A = v3(r0 * cosf(a0), ytop + d0, r0 * sinf(a0)), B = v3(r1 * cosf(a1), ytop + d1, r1 * sinf(a1));
        /* the tip droops: a second, short band hanging from the rim */
        V3 A2 = v3(r0 * 0.92f * cosf(a0), ytop + d0 - 0.035f, r0 * 0.92f * sinf(a0)), B2 = v3(r1 * 0.92f * cosf(a1), ytop + d1 - 0.035f, r1 * 0.92f * sinf(a1));
        V3 nT = v3(0, 1, 0), nA = vnorm(v3(cosf(a0), 0.9f, sinf(a0))), nB = vnorm(v3(cosf(a1), 0.9f, sinf(a1)));
        mb_vert(T, nT, 0, 0); mb_vert(B, nB, 0, 0); mb_vert(A, nA, 0, 0);
        V3 nA2 = vnorm(v3(cosf(a0), -0.2f, sinf(a0))), nB2 = vnorm(v3(cosf(a1), -0.2f, sinf(a1)));
        mb_vert(A, nA, 0, 0); mb_vert(B, nB, 0, 0); mb_vert(B2, nB2, 0, 0);
        mb_vert(A, nA, 0, 0); mb_vert(B2, nB2, 0, 0); mb_vert(A2, nA2, 0, 0);
    }
}

static void trees_init(void) {
    static Mesh m;
    mb_begin(&m);
    for (int k = 0; k < NTREEKIND; k++) {
        tree_first[k] = m.n / VF;
        mb->mat = (float)k;
        switch (k) {
        case 0:   /* spruce */
            trunk(0.025f, 0.01f, 0.95f, 6);
            for (int t = 0; t < 7; t++) {
                float y = 0.16f + t * 0.12f;
                skirt(y + 0.16f, y, 0.27f * (1 - t / 7.5f) + 0.03f, 11, t + 1);
            }
            break;
        case 1:   /* pine: bare trunk, a high flat crown */
            trunk(0.03f, 0.018f, 0.75f, 6);
            blob(v3(0.0f, 0.78f, 0.0f), v3(0.24f, 0.11f, 0.22f), v3(0, 0.7f, 0), 0.5f, 11);
            blob(v3(0.12f, 0.7f, 0.06f), v3(0.14f, 0.08f, 0.13f), v3(0, 0.7f, 0), 0.5f, 12);
            blob(v3(-0.1f, 0.86f, -0.05f), v3(0.15f, 0.08f, 0.14f), v3(0, 0.7f, 0), 0.5f, 13);
            break;
        case 2:   /* oak: broad */
        case 3: { /* ash: taller, looser */
            trunk(0.04f, 0.022f, 0.45f, 7);
            V3 cc = v3(0, k == 2 ? 0.6f : 0.65f, 0);
            int nl = k == 2 ? 8 : 7;
            for (int l = 0; l < nl; l++) {
                float a = l * 2.4f + k, h = (float)hrand(l, k, 63);
                float rr = (k == 2 ? 0.26f : 0.18f) * (0.5f + 0.5f * h);
                V3 c = v3(rr * cosf(a), cc.y + (h - 0.4f) * (k == 2 ? 0.25f : 0.38f), rr * sinf(a));
                float s = (k == 2 ? 0.2f : 0.15f) * (0.8f + 0.4f * (float)hrand(l, k, 64));
                blob(c, v3(s, s * 0.85f, s), cc, 0.55f, l + k * 10);
            }
            break;
        }
        default:  /* a bush */
            trunk(0.04f, 0.03f, 0.15f, 5);
            for (int l = 0; l < 4; l++) {
                float a = l * 1.9f;
                blob(v3(0.15f * cosf(a), 0.32f + 0.08f * (l & 1), 0.15f * sinf(a)), v3(0.24f, 0.2f, 0.24f), v3(0, 0.3f, 0), 0.6f, 40 + l);
            }
        }
        tree_count[k] = m.n / VF - tree_first[k];
    }
    gm_upload(&g_tree, &m);
    glGenVertexArrays(1, &vao_tree);
    glBindVertexArray(vao_tree);
    glBindBuffer(GL_ARRAY_BUFFER, g_tree.vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(6 * sizeof(float)));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(8 * sizeof(float)));
    for (int i = 0; i < 6; i++) glEnableVertexAttribArray((GLuint)i);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
}

static int tile_visible(const Tile *t);
static void draw_trees(GLuint p) {
    glBindVertexArray(vao_tree);
    for (int i = 0; i < ndraw; i++) {
        Tile *t = drawlist[i];
        if (!t->ntree || !tile_visible(t)) continue;
        glUniform3f(UL(p, "uTileO"), (float)(t->ox - cam.p[0]), (float)(-cam.p[1]), (float)(t->oz - cam.p[2]));
        glBindBuffer(GL_ARRAY_BUFFER, t->tree_vbo);
        for (int k = 0; k < NTREEKIND; k++) {
            if (!t->tcount[k]) continue;
            size_t off = sizeof(float) * 8 * (size_t)t->toff[k];
            glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)off);
            glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(off + 4 * sizeof(float)));
            glDrawArraysInstanced(GL_TRIANGLES, tree_first[k], tree_count[k], t->tcount[k]);
        }
    }
}
