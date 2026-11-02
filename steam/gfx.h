/* steam: small 3D maths, GL helpers and a procedural mesh builder.
 * Included once, by steam.c. */

typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { return (V3){x, y, z}; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vmul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a) { float l = vlen(a); return l > 1e-9f ? vmul(a, 1 / l) : v3(0, 1, 0); }
static V3 vlerp(V3 a, V3 b, float t) { return vadd(a, vmul(vsub(b, a), t)); }

/* column-major 4x4 */
typedef struct { float m[16]; } M4;
static M4 m4id(void) { M4 r = {{0}}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
static M4 m4mul(M4 a, M4 b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static M4 m4trans(float x, float y, float z) { M4 r = m4id(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
static M4 m4scale(float x, float y, float z) { M4 r = m4id(); r.m[0] = x; r.m[5] = y; r.m[10] = z; return r; }
static M4 m4rotx(float a) { M4 r = m4id(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
static M4 m4roty(float a) { M4 r = m4id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
static M4 m4rotz(float a) { M4 r = m4id(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }
/* a frame from basis vectors (columns) and an origin */
static M4 m4basis(V3 X, V3 Y, V3 Z, V3 o) {
    M4 r = m4id();
    r.m[0] = X.x; r.m[1] = X.y; r.m[2] = X.z;
    r.m[4] = Y.x; r.m[5] = Y.y; r.m[6] = Y.z;
    r.m[8] = Z.x; r.m[9] = Z.y; r.m[10] = Z.z;
    r.m[12] = o.x; r.m[13] = o.y; r.m[14] = o.z;
    return r;
}
static V3 m4pt(M4 a, V3 p) {
    return v3(a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12], a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13],
              a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14]);
}
static V3 m4dir(M4 a, V3 p) {
    return v3(a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z, a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z,
              a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z);
}

/* ---- GL ---- */
static GLuint compile_sh(GLenum type, const char *src, const char *name) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "steam: shader %s:\n%s\n", name, log);
        exit(1);
    }
    return s;
}
static GLuint mk_prog(const char *name, const char *vs, const char *fs) {
    GLuint pr = glCreateProgram();
    glAttachShader(pr, compile_sh(GL_VERTEX_SHADER, vs, name));
    glAttachShader(pr, compile_sh(GL_FRAGMENT_SHADER, fs, name));
    glLinkProgram(pr);
    GLint ok;
    glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(pr, sizeof log, NULL, log);
        fprintf(stderr, "steam: link %s:\n%s\n", name, log);
        exit(1);
    }
    return pr;
}
#define UL(p, n) glGetUniformLocation(p, n)

static GLuint mk_tex(int w, int h, GLenum ifmt, GLenum fmt, GLenum type, GLenum filt) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)ifmt, w, h, 0, fmt, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static GLuint mk_fbo(GLuint color, GLuint depth) {
    GLuint f;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    if (color) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    if (depth) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    if (!color) { glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE); }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "steam: incomplete framebuffer\n"); exit(1); }
    return f;
}

/* ---- meshes ----
 * A vertex: position, normal, uv, material (an index into the shader's
 * palette). The builder applies its current transform and material. */
#define VF 9
typedef struct {
    float *v;
    int n, cap;                 /* floats */
    M4 xf;
    float mat;
} Mesh;
static Mesh *mb;                /* the mesh being built */

static void mb_begin(Mesh *m) { m->n = 0; m->xf = m4id(); m->mat = 0; mb = m; }
static void mb_vert(V3 p, V3 n, float u, float v) {
    if (mb->n + VF > mb->cap) { mb->cap = mb->cap ? mb->cap * 2 : 65536; mb->v = realloc(mb->v, sizeof(float) * (size_t)mb->cap); }
    V3 P = m4pt(mb->xf, p), N = vnorm(m4dir(mb->xf, n));
    float *o = mb->v + mb->n;
    o[0] = P.x; o[1] = P.y; o[2] = P.z; o[3] = N.x; o[4] = N.y; o[5] = N.z; o[6] = u; o[7] = v; o[8] = mb->mat;
    mb->n += VF;
}
static void mb_tri(V3 a, V3 b, V3 c) {
    V3 n = vnorm(vcross(vsub(b, a), vsub(c, a)));
    mb_vert(a, n, 0, 0); mb_vert(b, n, 1, 0); mb_vert(c, n, 1, 1);
}
/* a quad a b c d (counter-clockwise seen from the front), flat */
static void mb_quad(V3 a, V3 b, V3 c, V3 d) {
    V3 n = vnorm(vcross(vsub(b, a), vsub(d, a)));
    mb_vert(a, n, 0, 0); mb_vert(b, n, 1, 0); mb_vert(c, n, 1, 1);
    mb_vert(a, n, 0, 0); mb_vert(c, n, 1, 1); mb_vert(d, n, 0, 1);
}
static void mb_quad_uv(V3 a, V3 b, V3 c, V3 d, float u0, float v0, float u1, float v1) {
    V3 n = vnorm(vcross(vsub(b, a), vsub(d, a)));
    mb_vert(a, n, u0, v0); mb_vert(b, n, u1, v0); mb_vert(c, n, u1, v1);
    mb_vert(a, n, u0, v0); mb_vert(c, n, u1, v1); mb_vert(d, n, u0, v1);
}
/* an axis-aligned box from lo to hi */
static void mb_box(V3 lo, V3 hi) {
    V3 p[8];
    for (int i = 0; i < 8; i++) p[i] = v3(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y, i & 4 ? hi.z : lo.z);
    mb_quad(p[0], p[2], p[3], p[1]);   /* -z */
    mb_quad(p[4], p[5], p[7], p[6]);   /* +z */
    mb_quad(p[0], p[4], p[6], p[2]);   /* -x */
    mb_quad(p[1], p[3], p[7], p[5]);   /* +x */
    mb_quad(p[0], p[1], p[5], p[4]);   /* -y */
    mb_quad(p[2], p[6], p[7], p[3]);   /* +y */
}
/* A surface of revolution about the local z axis: profile points (r, z),
 * n segments round, smooth normals from the profile's slope (a sharp
 * corner where the profile repeats a point). */
static void mb_lathe(const float *rz, int np, int n, float a0, float a1) {
    for (int i = 0; i + 1 < np; i++) {
        float r0 = rz[2 * i], z0 = rz[2 * i + 1], r1 = rz[2 * i + 2], z1 = rz[2 * i + 3];
        if (r0 == r1 && z0 == z1) continue;
        float dr = r1 - r0, dz = z1 - z0, l = sqrtf(dr * dr + dz * dz);
        /* outward normal of the segment (r, z): (dz, -dr) */
        float nr = dz / l, nz = -dr / l;
        for (int k = 0; k < n; k++) {
            float t0 = a0 + (a1 - a0) * k / n, t1 = a0 + (a1 - a0) * (k + 1) / n;
            float c0 = cosf(t0), s0 = sinf(t0), c1 = cosf(t1), s1 = sinf(t1);
            V3 A = v3(r0 * c0, r0 * s0, z0), B = v3(r0 * c1, r0 * s1, z0), C = v3(r1 * c1, r1 * s1, z1), D = v3(r1 * c0, r1 * s0, z1);
            V3 nA = v3(nr * c0, nr * s0, nz), nB = v3(nr * c1, nr * s1, nz);
            float u0 = (float)k / n, u1 = (float)(k + 1) / n, va = (float)i / (np - 1), vb = (float)(i + 1) / (np - 1);
            mb_vert(A, nA, u0, va); mb_vert(B, nB, u1, va); mb_vert(C, nB, u1, vb);
            mb_vert(A, nA, u0, va); mb_vert(C, nB, u1, vb); mb_vert(D, nA, u0, vb);
        }
    }
}
/* a cylinder about local z from z0 to z1, capped */
static void mb_cyl(float r, float z0, float z1, int n, int caps) {
    float pr[12] = {0, z0, r, z0, r, z0, r, z1, r, z1, 0, z1};
    if (caps) mb_lathe(pr, 6, n, 0, 2 * (float)M_PI);
    else mb_lathe(pr + 2, 4, n, 0, 2 * (float)M_PI);
}
/* the same along an arbitrary segment a->b */
static void mb_rod(V3 a, V3 b, float r, int n, int caps) {
    V3 Z = vsub(b, a);
    float l = vlen(Z);
    Z = vmul(Z, 1 / l);
    V3 X = fabsf(Z.y) < 0.9f ? vnorm(vcross(v3(0, 1, 0), Z)) : vnorm(vcross(v3(1, 0, 0), Z));
    V3 Y = vcross(Z, X);
    M4 save = mb->xf;
    mb->xf = m4mul(save, m4basis(X, Y, Z, a));
    mb_cyl(r, 0, l, n, caps);
    mb->xf = save;
}
/* a torus about local z: ring radius R, tube radius r */
static void mb_torus(float R, float r, int n, int m) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            float a0 = 2 * (float)M_PI * i / n, a1 = 2 * (float)M_PI * (i + 1) / n;
            float b0 = 2 * (float)M_PI * j / m, b1 = 2 * (float)M_PI * (j + 1) / m;
            V3 P[4], N[4];
            float as[4] = {a0, a1, a1, a0}, bs[4] = {b0, b0, b1, b1};
            for (int k = 0; k < 4; k++) {
                float ca = cosf(as[k]), sa = sinf(as[k]), cb = cosf(bs[k]), sb = sinf(bs[k]);
                N[k] = v3(ca * cb, sa * cb, sb);
                P[k] = v3(ca * (R + r * cb), sa * (R + r * cb), r * sb);
            }
            mb_vert(P[0], N[0], 0, 0); mb_vert(P[1], N[1], 1, 0); mb_vert(P[2], N[2], 1, 1);
            mb_vert(P[0], N[0], 0, 0); mb_vert(P[2], N[2], 1, 1); mb_vert(P[3], N[3], 0, 1);
        }
}
/* A prism: a closed 2D outline (x, y) extruded along local z from z0 to z1,
 * with flat caps (the outline must be convex, or star-shaped about its first point). */
static void mb_prism(const float *xy, int np, float z0, float z1, int capped) {
    for (int i = 0; i < np; i++) {
        int j = (i + 1) % np;
        V3 a = v3(xy[2 * i], xy[2 * i + 1], z0), b = v3(xy[2 * j], xy[2 * j + 1], z0);
        V3 c = v3(xy[2 * j], xy[2 * j + 1], z1), d = v3(xy[2 * i], xy[2 * i + 1], z1);
        mb_quad(a, b, c, d);
    }
    if (!capped) return;
    for (int i = 1; i + 1 < np; i++) {
        mb_tri(v3(xy[0], xy[1], z1), v3(xy[2 * i], xy[2 * i + 1], z1), v3(xy[2 * i + 2], xy[2 * i + 3], z1));
        mb_tri(v3(xy[0], xy[1], z0), v3(xy[2 * i + 2], xy[2 * i + 3], z0), v3(xy[2 * i], xy[2 * i + 1], z0));
    }
}

/* a mesh on the GPU (ni > 0: drawn indexed) */
typedef struct { GLuint vao, vbo, ebo; int n, ni; } GMesh;
static void gm_upload(GMesh *g, const Mesh *m) {
    if (!g->vao) { glGenVertexArrays(1, &g->vao); glGenBuffers(1, &g->vbo); }
    glBindVertexArray(g->vao);
    glBindBuffer(GL_ARRAY_BUFFER, g->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (size_t)m->n, m->v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(3 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(6 * sizeof(float)));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, VF * sizeof(float), (void *)(8 * sizeof(float)));
    for (int i = 0; i < 4; i++) glEnableVertexAttribArray((GLuint)i);
    g->n = m->n / VF;
}
static void gm_draw(const GMesh *g) {
    if (!g->n) return;
    glBindVertexArray(g->vao);
    if (g->ni) glDrawElements(GL_TRIANGLES, g->ni, GL_UNSIGNED_SHORT, 0);
    else glDrawArrays(GL_TRIANGLES, 0, g->n);
}
static void gm_upload_indexed(GMesh *g, const Mesh *m, const unsigned short *idx, int ni) {
    gm_upload(g, m);
    if (!g->ebo) glGenBuffers(1, &g->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * (size_t)ni, idx, GL_STATIC_DRAW);
    g->ni = ni;
}

/* move the triangles whose material passes keep() from m into out */
static void mesh_split(Mesh *m, Mesh *out, int (*keep)(int mat)) {
    int w = 0;
    out->n = 0;
    for (int i = 0; i < m->n; i += 3 * VF) {
        float *t = m->v + i;
        if (keep((int)(t[8] + 0.5f))) {
            if (out->n + 3 * VF > out->cap) { out->cap = out->cap ? out->cap * 2 : 65536; out->v = realloc(out->v, sizeof(float) * (size_t)out->cap); }
            memcpy(out->v + out->n, t, sizeof(float) * 3 * VF);
            out->n += 3 * VF;
        } else {
            memmove(m->v + w, t, sizeof(float) * 3 * VF);
            w += 3 * VF;
        }
    }
    m->n = w;
}

/* the view frustum's planes, for culling spheres */
static float frustum[6][4];
static void frustum_from(const M4 *vp) {
    const float *m = vp->m;
    for (int i = 0; i < 6; i++) {
        int r = i / 2;
        float sg = i % 2 ? -1.0f : 1.0f;
        for (int k = 0; k < 4; k++) frustum[i][k] = m[k * 4 + 3] + sg * m[k * 4 + r];
        float l = sqrtf(frustum[i][0] * frustum[i][0] + frustum[i][1] * frustum[i][1] + frustum[i][2] * frustum[i][2]);
        for (int k = 0; k < 4; k++) frustum[i][k] /= l;
    }
}
static int cull_on = 1;
static int visible(float x, float y, float z, float r) {
    if (!cull_on) return 1;
    for (int i = 0; i < 6; i++)
        if (frustum[i][0] * x + frustum[i][1] * y + frustum[i][2] * z + frustum[i][3] < -r) return 0;
    return 1;
}
