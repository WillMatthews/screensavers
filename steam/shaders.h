/* steam: the GLSL. Included once, by steam.c.
 *
 * Everything is drawn relative to the camera (world - camera, in metres, y
 * up); uWOff is the camera's world position modulo 4096, for the textures'
 * noise, which repeats every 4096 m. Light: the moon (with its phase), the
 * starlit sky, a few point lights (the firebox glow, the engine's lamps,
 * station lamps, a car's headlights...), and the carriages' lit windows
 * thrown on the ground beside the train; then height fog and valley mist. */
#define GLSL(...) #__VA_ARGS__
#define HDR "#version 330 core\n"

static const char *GL_COMMON = GLSL(
    uniform vec3 uMoonDir, uMoonCol, uSkyCol, uGndCol, uFogCol;
    uniform float uFogDen, uMistY, uMistDen, uTime;
    uniform vec3 uWOff;
    uniform vec4 uPL[16], uPLC[16];
    uniform int uNPL;
    uniform vec4 uCar[12];       /* the carriages: centre xyz, half length */
    uniform vec4 uCarD[12];      /* forward xz, lit fraction, kind */
    uniform int uNCar;
    uniform vec4 uCarBox;        /* a sphere round all the carriages: centre, radius */
    uniform sampler2D uShadow;
    uniform mat4 uShadowM;

    float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    float hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash12(i), hash12(i + vec2(1, 0)), f.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), f.x), f.y);
    }
    float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.02 + vec2(1.7, 9.2); a *= 0.5; } return s / 0.9375; }
    float vnoise3(vec3 p) {
        vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
                   mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
    }

    float moonShadow(vec3 P, vec3 N) {
        vec4 s = uShadowM * vec4(P + N * 0.08, 1.0);
        vec3 q = s.xyz / s.w * 0.5 + 0.5;
        if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0 || q.z > 1.0) return 1.0;
        vec2 ts = 1.0 / vec2(textureSize(uShadow, 0));
        float sh = 0.0;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++)
                sh += step(q.z - 0.0015, texture(uShadow, q.xy + vec2(i, j) * ts * 1.2).r);
        return sh / 9.0;
    }

    /* the lit windows of the carriages, thrown on whatever is beside them */
    vec3 trainGlow(vec3 P, vec3 N) {
        vec3 acc = vec3(0.0);
        vec3 dc = P - uCarBox.xyz;
        if (dot(dc, dc) > uCarBox.w * uCarBox.w) return acc;
        for (int i = 0; i < 12; i++) {
            if (i >= uNCar) break;
            vec4 C = uCar[i], D = uCarD[i];
            if (D.z <= 0.0) continue;
            vec2 f = D.xy, s = vec2(f.y, -f.x);
            vec3 d = P - C.xyz;
            float lz = dot(d.xz, f), lx = dot(d.xz, s);
            float side = abs(lx) - 1.45;
            if (side < 0.0 || side > 14.0 || abs(lz) > C.w + 10.0) continue;
            float ly = d.y + 0.4;
            /* window bays every 2.3 m, blurred by the distance */
            float blur = 0.15 + side * 0.18;
            float bays = smoothstep(C.w + blur * 3.0, C.w - 1.0, abs(lz));
            float pat = 0.6 + 0.4 * cos(lz * 6.2832 / 2.3) * exp(-blur * 2.0);
            vec3 dir = normalize(vec3(s * sign(lx), 0.0).xzy * vec3(1.0, 0.0, 1.0) + vec3(0.0, 0.25, 0.0));
            float facing = max(dot(N, -normalize(vec3(sign(lx) * s.x, -0.35 + ly * 0.0, sign(lx) * s.y)) * -1.0), 0.0);
            float fallo = 1.0 / (1.0 + side * side * 0.35) * smoothstep(-1.5, 0.5, ly) * smoothstep(3.5, 1.0, ly + side * 0.15);
            acc += vec3(1.0, 0.62, 0.3) * D.z * bays * pat * fallo * (0.35 + 0.65 * facing);
        }
        return acc * 0.9;
    }

    /* everything but the sky: albedo, normal, roughness, metal, at camera-relative P */
    vec3 lightAll(vec3 P, vec3 N, vec3 alb, float rough, float metal, float shadow) {
        vec3 V = normalize(-P);
        vec3 F0 = mix(vec3(0.04), alb, metal);
        float a = rough * rough;
        vec3 col = vec3(0.0);
        /* the moon */
        {
            vec3 L = uMoonDir, H = normalize(L + V);
            float nl = max(dot(N, L), 0.0), nh = max(dot(N, H), 0.0), nv = max(dot(N, V), 0.001);
            float dd = nh * nh * (a * a - 1.0) + 1.0;
            float D = a * a / (3.14159 * dd * dd);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
            float G = 0.5 / mix(2.0 * nl * nv, nl + nv, a);
            col += uMoonCol * shadow * nl * ((1.0 - metal) * alb / 3.14159 * (1.0 - F) + D * F * G);
        }
        /* the sky and the ground */
        vec3 amb = mix(uGndCol, uSkyCol, N.y * 0.5 + 0.5);
        vec3 R = reflect(-V, N);
        vec3 envc = mix(uGndCol * 0.6, uSkyCol * 1.6, smoothstep(-0.2, 0.6, R.y));
        vec3 Fe = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        col += (1.0 - metal) * alb * amb + envc * Fe * (1.0 - rough * 0.7);
        /* point lights */
        for (int i = 0; i < 16; i++) {
            if (i >= uNPL) break;
            vec3 d = uPL[i].xyz - P;
            float r2 = dot(d, d);
            float rad = uPL[i].w;
            if (r2 > rad * rad * 64.0) continue;
            vec3 L = d * inversesqrt(r2);
            vec3 C = uPLC[i].rgb;
            float kind = uPLC[i].w;
            /* kind > 0: a spot along -z of the light (a car's headlamps): w is the cone's direction (packed) */
            float att = 1.0 / (1.0 + r2 / (rad * rad));
            att *= smoothstep(rad * 8.0, rad * 4.0, sqrt(r2));
            vec3 H = normalize(L + V);
            float nl = max(dot(N, L), 0.0), nh = max(dot(N, H), 0.0);
            float dd = nh * nh * (a * a - 1.0) + 1.0;
            float D = a * a / (3.14159 * dd * dd);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
            col += C * att * nl * ((1.0 - metal) * alb / 3.14159 + D * F * 0.25);
        }
        col += (1.0 - metal) * alb * trainGlow(P, N);
        return col;
    }

    /* fog: a general haze, and mist lying in the valleys below uMistY */
    vec3 fogApply(vec3 col, vec3 P) {
        float dist = length(P);
        vec3 V = P / max(dist, 1e-3);
        float haze = 1.0 - exp(-dist * uFogDen);
        /* the mist's depth along the ray: density falls off above uMistY */
        float y0 = 0.0, y1 = P.y;
        float k = 0.12;
        float m0 = exp(-k * (y0 - uMistY)), m1 = exp(-k * (y1 - uMistY));
        float mist = abs(V.y) > 1e-3 ? (m0 - m1) / (k * V.y) * dist / dist : m0 * dist;
        mist = abs(V.y) > 1e-3 ? (m0 - m1) / (k * V.y) : m0 * dist;
        float mt = 1.0 - exp(-max(mist, 0.0) * uMistDen);
        float sc = pow(max(dot(V, uMoonDir), 0.0), 8.0);
        vec3 fc = uFogCol * (1.0 + 2.0 * sc);
        col = mix(col, fc, haze);
        col = mix(col, fc * 1.25, mt);
        return col;
    }
);

/* ---- objects: the train, the lineside, the buildings ---- */
static const char *VS_OBJ = HDR GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aUV;
    layout(location = 3) in float aMat;
    uniform mat4 uModel, uVP;
    out vec3 vP, vN, vL;
    out vec2 vUV;
    flat out int vMat;
    void main() {
        vec4 w = uModel * vec4(aPos, 1.0);
        vP = w.xyz;
        vN = mat3(uModel) * aN;
        vL = aPos;
        vUV = aUV;
        vMat = int(aMat + 0.5);
        gl_Position = uVP * w;
    });

static const char *FS_OBJ_HEAD = GLSL(
    in vec3 vP, vN, vL;
    in vec2 vUV;
    flat in int vMat;
    uniform vec4 uPal[48], uPal2[48];
    uniform vec4 uWheelW4;       /* the wheels' angular speeds (rad/s), for the blur: drivers, bogie, trailing, others */
    uniform vec4 uWheelA4;       /* and their angles */
    uniform vec4 uObj;           /* per draw: emissive scale, lit fraction, seed, - */
    uniform sampler2D uTex;
    out vec4 o;
);
static const char *FS_OBJ = GLSL(
    void main() {
        vec3 N = normalize(vN);
        if (!gl_FrontFacing) N = -N;
        vec4 A = uPal[vMat], B = uPal2[vMat];
        vec3 alb = A.rgb;
        float rough = A.a, metal = B.x, emis = B.y;
        int kind = int(B.z + 0.5);
        vec3 E = vec3(0.0);
        vec3 wp = vP + uWOff;
        float cover = 1.0;
        if (kind == 1) {
            /* a spoked wheel, drawn on its disc (uv: local x y over the radius), blurred by its turning */
            vec2 q = vUV;
            float r = length(q);
            int cls = int(B.w / 100.0);
            float nsp = mod(B.w, 100.0);
            float uWheelA = cls == 0 ? uWheelA4.x : cls == 1 ? uWheelA4.y : cls == 2 ? uWheelA4.z : uWheelA4.w;
            float uWheelW = cls == 0 ? uWheelW4.x : cls == 1 ? uWheelW4.y : cls == 2 ? uWheelW4.z : uWheelW4.w;
            float ang = atan(q.y, q.x) - uWheelA;
            float blur = clamp(abs(uWheelW) * 0.009 * nsp / 6.2832, 0.0, 1.0);
            float sp = abs(fract(ang * nsp / 6.2832) - 0.5) * 2.0;
            float wsp = 0.16 / max(r, 0.2);
            float spoke = smoothstep(wsp + 0.03 + blur, wsp - 0.03, 1.0 - sp);
            spoke = mix(spoke, 0.62, blur);
            float hub = smoothstep(0.3, 0.27, r);
            float rim = smoothstep(0.8, 0.83, r);
            float web = max(max(spoke, hub), rim);
            /* the balance weight, a crescent between the spokes */
            float bw = smoothstep(0.02, -0.02, length(q - 0.42 * vec2(cos(uWheelA + 3.14159), sin(uWheelA + 3.14159))) - 0.36) * smoothstep(0.32, 0.36, r);
            web = max(web, mix(bw, bw * 0.5, blur));
            cover = web;
            if (cover < 0.02) discard;
        } else if (kind == 2) {
            /* a lit window: a warm interior, blinds and silhouettes, a frame */
            float lit = step(hash12(floor(vec2(vUV.x * 0.999, 0.0)) + uObj.z), uObj.y);
            vec2 f = fract(vUV);
            float blind = smoothstep(0.55, 0.6, f.y) * step(0.5, hash12(floor(vUV) + uObj.z + 7.0));
            vec3 warm = vec3(1.0, 0.66, 0.34) * (0.9 + 0.3 * vnoise(vUV * vec2(3.0, 1.0) + uObj.z));
            warm *= 1.0 - 0.55 * smoothstep(0.35, 0.0, abs(f.x - 0.3 - 0.4 * hash12(floor(vUV) + 3.0)) - 0.05) * smoothstep(0.75, 0.3, f.y) * step(0.4, hash12(floor(vUV) + 9.0));
            warm *= 1.0 - 0.4 * blind;
            E = warm * vec3(1.0, 0.8, 0.6) * lit * emis * uObj.x;
            alb = vec3(0.02);
            rough = 0.05;
        } else if (kind == 3) {
            /* a painted sign: the texture's red is the lettering */
            float row = mod(B.w, 10.0);
            float t = texture(uTex, vec2(vUV.x, (row + clamp(vUV.y, 0.0, 1.0)) / 8.0)).r;
            if (B.w >= 10.0) {
                /* a cast plate: polished brass letters on a painted ground */
                alb = mix(alb, vec3(0.85, 0.62, 0.28), t);
                metal = t;
                rough = mix(rough, 0.2, t);
            } else alb = mix(alb, vec3(0.85, 0.83, 0.76), t);
        } else if (kind == 4) {
            E = alb * emis * uObj.x * (vMat == 28 || vMat == 43 ? uObj.w : 1.0);
        } else if (kind == 5) {
            /* coal: lumps catching the light */
            float n = vnoise3(vL * 6.0) * 0.6 + vnoise3(vL * 17.0) * 0.4;
            alb = vec3(0.03) * (0.6 + n);
            N = normalize(N + 0.6 * (vec3(vnoise3(vL * 9.0), vnoise3(vL * 9.0 + 3.1), vnoise3(vL * 9.0 + 7.7)) - 0.5));
            rough = 0.35;
        } else if (kind == 6) {
            /* masonry: courses of stone */
            vec2 c = vec2(dot(wp.xz, vec2(0.7071)) + wp.x * 0.0, wp.y);
            vec2 g = vec2(wp.x + wp.z, wp.y) / vec2(0.9, 0.35);
            g.x += floor(g.y) * 0.5;
            vec2 fg = fract(g);
            float mort = smoothstep(0.06, 0.0, min(min(fg.x, 1.0 - fg.x) * 0.9 * 2.5, min(fg.y, 1.0 - fg.y)));
            alb *= (0.75 + 0.5 * hash12(floor(g))) * (1.0 - 0.45 * mort) * (0.8 + 0.4 * vnoise(wp.xz * 0.7 + wp.y));
        } else if (kind == 7) {
            /* polished: the rails' heads, the motion's bright steel */
            rough = 0.18 + 0.15 * vnoise(wp.xz * 3.0);
        }
        /* grime and wear on everything painted */
        if (kind == 0 && metal < 0.5) {
            float g = vnoise(vL.xz * 3.0 + vL.y * 2.0);
            alb *= 0.85 + 0.25 * g;
            rough = clamp(rough + 0.15 * (g - 0.5), 0.05, 1.0);
        }
        float sh = moonShadow(vP, N);
        vec3 col = lightAll(vP, N, alb, rough, metal, sh) + E;
        o = vec4(fogApply(col, vP), cover);
    });

/* ---- the terrain ---- */
static const char *VS_TER = HDR GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aN;
    uniform vec3 uTileO;        /* the tile's corner, camera-relative */
    uniform vec3 uTileW;        /* and in world, mod 4096 */
    uniform mat4 uVP;
    out vec3 vP, vN, vW;
    void main() {
        vP = aPos + uTileO;
        vW = aPos + uTileW;
        vN = aN;
        gl_Position = uVP * vec4(vP, 1.0);
    });
static const char *FS_TER = GLSL(
    in vec3 vP, vN, vW;
    uniform vec4 uTrack;        /* the line near the camera: x and z of a point, dx/dz, rail-top y */
    uniform vec4 uTrack2;       /* d2x/dz2, dy/dz, crossing z (camera-relative), station half-width */
    uniform vec4 uRiver;        /* viaduct valley: z centre, half length (camera-relative), water y */
    out vec4 o;
    void main() {
        vec3 N = normalize(vN);
        vec3 P = vP;
        /* how far from the line: a quadratic model of it about the camera */
        float dz = P.z - uTrack.y;
        float tx = uTrack.x + uTrack.z * dz + 0.5 * uTrack2.x * dz * dz;
        float ty = uTrack.w + uTrack2.y * dz;
        float d = abs(P.x - tx);
        vec2 w2 = vW.xz;
        float n = vnoise(w2 * 0.05) * 0.6 + vnoise(w2 * 0.13) * 0.4, n2 = vnoise(w2 * 0.6), n3 = vnoise(w2 * 3.0);
        /* fields: pasture, plough and stubble in patches, darker in the hollows */
        float field = hash12(floor(w2 / vec2(97.0, 131.0) + vec2(0.6 * sin(vW.z / 140.0), 0.6 * sin(vW.x / 170.0))));
        vec3 grass = mix(vec3(0.075, 0.095, 0.05), vec3(0.11, 0.12, 0.07), n);
        grass = mix(grass, vec3(0.1, 0.08, 0.055) * (0.8 + 0.3 * sin(dot(w2, vec2(0.6, 0.8)) * 3.0)), step(0.72, field));
        grass = mix(grass, vec3(0.15, 0.13, 0.08), step(0.86, field) * 0.7);
        vec3 alb = grass * (0.8 + 0.35 * n2) * (0.9 + 0.2 * n3);
        float rough = 0.9;
        /* rock where it is steep */
        float steep = smoothstep(0.78, 0.6, N.y);
        alb = mix(alb, vec3(0.13, 0.125, 0.12) * (0.7 + 0.5 * n2), steep);
        /* the cess and ballast shoulder by the line */
        float bal = smoothstep(4.2, 3.2, d) * smoothstep(1.5, 0.3, abs(P.y - ty + 0.55));
        alb = mix(alb, vec3(0.16, 0.15, 0.14) * (0.6 + 0.6 * n3), bal);
        /* the road over the level crossing */
        float road = smoothstep(3.4, 3.0, abs(P.z - uTrack2.z));
        alb = mix(alb, vec3(0.05, 0.05, 0.055) * (0.8 + 0.3 * n3), road);
        alb = mix(alb, vec3(0.5), road * smoothstep(0.08, 0.0, abs(abs(P.z - uTrack2.z) - 2.9)) * step(4.0, d));
        if (road > 0.5) rough = 0.35;
        /* the river in the viaduct's valley */
        float wet = 0.0;
        if (abs(P.z - uRiver.x) < uRiver.y * 1.4 && P.y < uRiver.z + 0.6) {
            wet = smoothstep(uRiver.z + 0.6, uRiver.z, P.y);
            alb = mix(alb, vec3(0.01, 0.012, 0.015), wet);
            rough = mix(rough, 0.04, wet);
            N = normalize(mix(N, vec3(0.0, 1.0, 0.0), wet) + wet * 0.03 * vec3(vnoise(w2 * 1.3 + uTime * 0.4) - 0.5, 0.0, vnoise(w2.yx * 1.3 - uTime * 0.3) - 0.5));
        }
        float sh = dot(P, P) < 90.0 * 90.0 ? moonShadow(P, N) : 1.0;
        vec3 col = lightAll(P, N, alb, rough, wet, sh);
        o = vec4(fogApply(col, P), 1.0);
    });

/* ---- trees, instanced: a cone stack (conifer) or a lumpy crown (broadleaf) ---- */
static const char *VS_TREE = HDR GLSL(
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aN;
    layout(location = 2) in vec2 aUV;
    layout(location = 3) in float aMat;
    layout(location = 4) in vec4 iPos;      /* x y z, height */
    layout(location = 5) in vec4 iKind;     /* kind, angle, tint, - */
    uniform vec3 uTileO;
    uniform mat4 uVP;
    uniform float uTime;
    out vec3 vP, vN, vL;
    flat out float vKind, vTint;
    void main() {
        float c = cos(iKind.y), s = sin(iKind.y);
        vec3 p = aPos * iPos.w;
        p.xz = mat2(c, -s, s, c) * p.xz;
        p.xz += sin(uTime * 0.7 + iPos.x * 0.3 + iPos.z * 0.2) * 0.05 * aPos.y * aPos.y * iPos.w;
        vec3 n = aN;
        n.xz = mat2(c, -s, s, c) * n.xz;
        vP = p + iPos.xyz + uTileO;
        vN = n;
        vL = aPos;
        vKind = aMat; vTint = iKind.z;
        gl_Position = uVP * vec4(vP, 1.0);
    });
static const char *FS_TREE = GLSL(
    in vec3 vP, vN, vL;
    flat in float vKind, vTint;
    out vec4 o;
    void main() {
        vec3 N = normalize(vN);
        float n = vnoise3(vL * 9.0 + vTint * 10.0);
        N = normalize(N + 0.7 * (vec3(vnoise3(vL * 7.0), vnoise3(vL * 7.0 + 3.0), vnoise3(vL * 7.0 + 6.0)) - 0.5));
        vec3 alb = vKind < 1.5 ? vec3(0.022, 0.04, 0.03) : vKind > 3.5 ? vec3(0.04, 0.05, 0.025) : vec3(0.035, 0.05, 0.026);
        alb *= 0.55 + 0.9 * n * (0.7 + 0.6 * vTint);
        alb *= 0.5 + 0.5 * clamp(vL.y * 1.4, 0.0, 1.0);
        if (length(vL.xz) < 0.045 && vL.y < 0.75) alb = vec3(0.03, 0.025, 0.02);
        float sh = moonShadow(vP, N);
        vec3 col = lightAll(vP, N, alb, 0.8, 0.0, sh);
        o = vec4(fogApply(col, vP), 1.0);
    });

/* ---- the sky: moon with its phase, stars, clouds, the glow of towns ---- */
static const char *VS_FULL = HDR GLSL(
    out vec2 vUV;
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        vUV = p;
        gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
    });
static const char *FS_SKY = GLSL(
    in vec2 vUV;
    uniform mat3 uCamRot;
    uniform vec4 uViewTan;      /* left, bottom, width, height of the view at unit distance */
    uniform float uPhase;       /* the moon's lit fraction's angle: 0 new, pi full */
    uniform vec3 uMoonUp;       /* toward the sun across the moon's face */
    uniform float uCloud, uStars;
    uniform vec4 uTowns;        /* azimuths of two glows, their strength */
    out vec4 o;
    vec3 skyCol(vec3 V) {
        float h = V.y;
        vec3 c = mix(vec3(0.016, 0.022, 0.04), vec3(0.002, 0.004, 0.012), smoothstep(-0.05, 0.6, h));
        c += uFogCol * 0.6 * exp(-max(h, 0.0) * 12.0);
        /* the moon's glow in the haze */
        float md = dot(V, uMoonDir);
        c += uMoonCol * (0.012 * pow(max(md, 0.0), 6.0) + 0.06 * pow(max(md, 0.0), 120.0));
        /* towns beyond the hills */
        float az = atan(V.x, V.z);
        for (int k = 0; k < 2; k++) {
            float a = k == 0 ? uTowns.x : uTowns.y;
            float da = abs(mod(az - a + 3.14159, 6.28318) - 3.14159);
            c += vec3(0.05, 0.028, 0.012) * uTowns.z * exp(-da * da * 30.0) * exp(-max(h, 0.0) * 25.0);
        }
        return c;
    }
    void main() {
        vec2 t = uViewTan.xy + vUV * uViewTan.zw;
        vec3 V = normalize(uCamRot * vec3(t, -1.0));
        vec3 c = skyCol(V);
        /* stars, twinkling, thinning toward the haze, and a faint milky band */
        if (V.y > -0.02) {
            vec3 sv = V * 420.0;
            vec3 cell = floor(sv);
            float h = hash13(cell);
            vec3 sp = cell + 0.5 + 0.35 * (vec3(hash13(cell + 1.3), hash13(cell + 2.7), hash13(cell + 4.1)) - 0.5);
            float d = length(sv - sp);
            float mag = pow(hash13(cell + 7.0), 18.0);
            float tw = 0.75 + 0.25 * sin(uTime * (3.0 + 6.0 * h) + h * 40.0);
            float band = exp(-pow(dot(V, normalize(vec3(0.5, 0.35, -0.8))) / 0.22, 2.0));
            float st = smoothstep(0.3, 0.0, d) * (mag * 6.0 + step(0.99 - band * 0.02, h) * 0.12) * tw;
            st *= smoothstep(-0.02, 0.25, V.y) * uStars;
            c += vec3(0.9, 0.93, 1.0) * st;
            c += vec3(0.015, 0.016, 0.022) * band * vnoise(V.xz * 30.0 / (V.y + 1.0)) * uStars * smoothstep(0.05, 0.4, V.y);
        }
        /* the moon: a disc, lit across its face toward the sun */
        float md = dot(V, uMoonDir);
        float mr = 0.0085;
        if (md > cos(mr * 1.5)) {
            vec3 ax = normalize(cross(uMoonDir, vec3(0.0, 1.0, 0.0)));
            vec3 ay = cross(ax, uMoonDir);
            vec2 q = vec2(dot(V, ax), dot(V, ay)) / mr;
            float r = length(q);
            if (r < 1.0) {
                vec3 sn = vec3(q, sqrt(1.0 - r * r));
                vec2 su = vec2(dot(uMoonUp, ax), dot(uMoonUp, ay));
                vec3 sun = normalize(vec3(su * sin(uPhase), -cos(uPhase)));
                float lit = smoothstep(-0.05, 0.08, dot(sn, sun));
                float maria = 0.75 + 0.25 * fbm(q * 2.3 + 4.0);
                vec3 mc = vec3(1.0, 0.97, 0.9) * 3.2 * maria * lit + vec3(0.02, 0.025, 0.035);
                c = mix(c, mc, smoothstep(1.0, 0.97, r));
            }
        }
        /* clouds: a deck of stratocumulus, silvered toward the moon */
        if (uCloud > 0.0 && V.y > 0.0) {
            vec2 cp = V.xz / (V.y + 0.12) * 1.6 + vec2(uTime * 0.004, uTime * 0.002);
            float cl = fbm(cp * 0.9) * 0.75 + vnoise(cp * 5.3 + 5.0) * 0.25;
            float cov = smoothstep(0.62 - uCloud * 0.35, 0.95 - uCloud * 0.3, cl);
            float sil = pow(max(md, 0.0), 10.0);
            vec3 cc = vec3(0.012, 0.014, 0.02) + uMoonCol * (0.02 + 0.25 * sil * (1.0 - cov * 0.7));
            c = mix(c, cc, cov * smoothstep(0.0, 0.08, V.y));
        }
        o = vec4(c, 1.0);
    });

/* ---- post: bloom, the grade, the vignette, a little grain ---- */
static const char *FS_DOWN = GLSL(
    in vec2 vUV;
    uniform sampler2D uSrc;
    uniform vec2 uTexel;
    uniform float uPrefilter;
    out vec4 o;
    void main() {
        vec3 c = vec3(0.0);
        c += texture(uSrc, vUV + uTexel * vec2(-1.0, -1.0)).rgb;
        c += texture(uSrc, vUV + uTexel * vec2(1.0, -1.0)).rgb;
        c += texture(uSrc, vUV + uTexel * vec2(-1.0, 1.0)).rgb;
        c += texture(uSrc, vUV + uTexel * vec2(1.0, 1.0)).rgb;
        c *= 0.25;
        if (uPrefilter > 0.0) {
            float l = max(max(c.r, c.g), c.b);
            c *= max(l - uPrefilter, 0.0) / max(l, 1e-4);
        }
        o = vec4(c, 1.0);
    });
static const char *FS_UP = GLSL(
    in vec2 vUV;
    uniform sampler2D uSrc;
    uniform vec2 uTexel;
    out vec4 o;
    void main() {
        vec3 c = texture(uSrc, vUV).rgb * 4.0;
        c += texture(uSrc, vUV + vec2(uTexel.x, 0.0)).rgb * 2.0 + texture(uSrc, vUV - vec2(uTexel.x, 0.0)).rgb * 2.0;
        c += texture(uSrc, vUV + vec2(0.0, uTexel.y)).rgb * 2.0 + texture(uSrc, vUV - vec2(0.0, uTexel.y)).rgb * 2.0;
        c += texture(uSrc, vUV + uTexel).rgb + texture(uSrc, vUV - uTexel).rgb;
        c += texture(uSrc, vUV + vec2(uTexel.x, -uTexel.y)).rgb + texture(uSrc, vUV - vec2(uTexel.x, -uTexel.y)).rgb;
        o = vec4(c / 16.0, 1.0);
    });
static const char *FS_POST = GLSL(
    in vec2 vUV;
    uniform sampler2D uScene, uBloom, uCap, uDepth;
    uniform float uExposure, uBloomAmt, uFade, uCapOn;
    uniform vec4 uVig;          /* this output's rect in the layout, 0..1 */
    uniform vec4 uCapRect;      /* the caption's rect on this output, 0..1 from the bottom left */
    out vec4 o;
    float lumi(vec3 c) { float l = dot(c, vec3(0.299, 0.587, 0.114)) * uExposure; return l / (1.0 + l); }
    /* FXAA, after Lottes: blend along the edge where the contrast says there is one */
    vec3 fxaa(vec2 uv) {
        vec2 px = 1.0 / vec2(textureSize(uScene, 0));
        vec3 cM = texture(uScene, uv).rgb;
        float lM = lumi(cM);
        float lNW = lumi(texture(uScene, uv + vec2(-1.0, -1.0) * px).rgb), lNE = lumi(texture(uScene, uv + vec2(1.0, -1.0) * px).rgb);
        float lSW = lumi(texture(uScene, uv + vec2(-1.0, 1.0) * px).rgb), lSE = lumi(texture(uScene, uv + vec2(1.0, 1.0) * px).rgb);
        float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE))), lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
        if (lMax - lMin < max(0.04, lMax * 0.15)) return cM;
        vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
        float red = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);
        dir = clamp(dir / (min(abs(dir.x), abs(dir.y)) + red), -8.0, 8.0) * px;
        vec3 a = 0.5 * (texture(uScene, uv + dir * (1.0 / 3.0 - 0.5)).rgb + texture(uScene, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
        vec3 b = a * 0.5 + 0.25 * (texture(uScene, uv - dir * 0.5).rgb + texture(uScene, uv + dir * 0.5).rgb);
        float lB = lumi(b);
        return lB < lMin || lB > lMax ? a : b;
    }
    vec3 aces(vec3 x) {
        const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }
    void main() {
        vec3 c = fxaa(vUV);
        c += texture(uBloom, vUV).rgb * uBloomAmt;
        c *= uExposure;
        /* the grade: cool, deep shadows; warm, soft highlights */
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(c, c * vec3(0.86, 0.98, 1.18), smoothstep(0.25, 0.0, l) * 0.6);
        c = aces(c);
        c = pow(c, vec3(1.0 / 2.2));
        vec2 lp = uVig.xy + vUV * uVig.zw;
        vec2 v = lp - 0.5;
        c *= 1.0 - 0.55 * pow(clamp(length(v * vec2(1.0, 0.8)) * 1.35, 0.0, 1.0), 2.6);
        float g = hash12(gl_FragCoord.xy + fract(uTime * 7.31) * 300.0) - 0.5;
        c += g * 0.018;
        c *= uFade;
        if (uCapOn > 0.5) {
            vec2 cq = (vUV - uCapRect.xy) / uCapRect.zw;
            if (cq.x > -0.02 && cq.x < 1.02 && cq.y > -0.1 && cq.y < 1.1) {
                vec2 tc = vec2(cq.x, 1.0 - cq.y);
                float t = texture(uCap, tc).r;
                vec2 px = 1.0 / vec2(textureSize(uCap, 0));
                float sh = texture(uCap, tc + vec2(-2.0, -3.0) * px).r;
                sh = max(sh, texture(uCap, tc + vec2(0.0, -4.0) * px).r);
                c *= 1.0 - 0.45 * sh;
                c = mix(c, vec3(0.93, 0.88, 0.78), t * 0.85);
            }
        }
        o = vec4(c, 1.0);
    });

/* the shadow pass: only the wheels' spokes need cutting out */
static const char *FS_SHADOW = GLSL(
    uniform float uPre;
    void main() {
        if (int(uPal2[vMat].z + 0.5) == 1) {
            if (uPre > 0.5) discard;
            vec2 q = vUV;
            float r = length(q);
            int cls = int(uPal2[vMat].w / 100.0);
            float nsp = mod(uPal2[vMat].w, 100.0);
            float a = cls == 0 ? uWheelA4.x : cls == 1 ? uWheelA4.y : cls == 2 ? uWheelA4.z : uWheelA4.w;
            float sp = abs(fract((atan(q.y, q.x) - a) * nsp / 6.2832) - 0.5) * 2.0;
            if (r > 0.3 && r < 0.8 && sp < 0.75) discard;
        }
        o = vec4(0.0);
    });

/* ---- smoke and steam: soft lumpy balls, lit ---- */
static const char *VS_SMOKE = HDR GLSL(
    layout(location = 0) in vec4 iPos;      /* camera-relative centre, radius */
    layout(location = 1) in vec4 iA;        /* alpha, kind, seed, darkness */
    layout(location = 2) in vec4 iL;        /* light from the lamps (rgb), age fraction */
    uniform mat4 uVP;
    uniform vec3 uCamR, uCamU, uCamF;
    out vec2 vQ;
    out vec4 vA, vL;
    out vec3 vC;
    out float vDepth;
    void main() {
        vec2 c[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
        vec2 q = c[gl_VertexID];
        float r = iPos.w * (iA.y > 1.5 ? 1.0 : 1.0);
        vec3 p = iPos.xyz;
        if (iA.y > 1.5) {
            /* a spark: a small streak, drawn a few pixels wide whatever the distance */
            r = max(r, length(p) * 0.0016);
        }
        float rot = iA.z + iL.w * 1.5;
        mat2 m = mat2(cos(rot), sin(rot), -sin(rot), cos(rot));
        vec2 qr = m * q;
        vec3 w = p + (uCamR * qr.x + uCamU * qr.y) * r;
        vQ = q;
        vA = iA; vL = iL; vC = p;
        vDepth = dot(p, uCamF);
        gl_Position = uVP * vec4(w, 1.0);
    });
static const char *FS_SMOKE = GLSL(
    in vec2 vQ;
    in vec4 vA, vL;
    in vec3 vC;
    in float vDepth;
    uniform sampler2D uDepth;
    uniform vec2 uDepthSize;
    uniform vec3 uCamR, uCamU, uCamF;
    out vec4 o;
    float linz(float d) { float n = 0.2, f = 16000.0; float z = d * 2.0 - 1.0; return 2.0 * n * f / (f + n - z * (f - n)); }
    void main() {
        float d = length(vQ);
        if (d > 1.0) discard;
        if (vA.y > 1.5) {
            /* a spark */
            float g = smoothstep(1.0, 0.0, d);
            float life = 1.0 - vL.w;
            vec3 c = mix(vec3(1.0, 0.25, 0.04), vec3(1.0, 0.7, 0.3), life) * 6.0 * life;
            o = vec4(c * g * g, 0.0);
            return;
        }
        float seed = vA.z;
        vec2 q = vQ * 1.6 + seed;
        float n = vnoise(q * 1.3 + vL.w * 0.6) * 0.55 + vnoise(q * 2.9 - vL.w * 0.9) * 0.3 + vnoise(q * 6.1) * 0.15;
        float dens = smoothstep(1.0, 0.15, d + (n - 0.5) * 0.9);
        float a = vA.x * dens;
        /* soften where it meets solid things */
        float sz = linz(texture(uDepth, gl_FragCoord.xy * 2.0 / uDepthSize).r);
        float r = 1.0;
        a *= clamp((sz - vDepth) / max(0.5, length(vC) * 0.002 + 1.0), 0.0, 1.0);
        if (a < 0.002) discard;
        /* a ball's normal, made lumpy */
        vec3 nn = normalize(vec3(vQ + (n - 0.5) * 0.8, sqrt(max(1.0 - d * d, 0.05))));
        vec3 N = normalize(uCamR * nn.x + uCamU * nn.y - uCamF * nn.z);
        vec3 V = normalize(vC);
        float steam = vA.y > 0.5 ? 1.0 : mix(0.75, 0.32, vA.w);
        vec3 alb = vec3(steam) * mix(vec3(1.0), vec3(1.0, 0.97, 0.93), 1.0 - steam);
        /* the moon: wrapped, and bright through thin edges looking toward it */
        float ml = max(dot(N, uMoonDir) * 0.75 + 0.25, 0.0);
        float fwd = pow(max(dot(V, uMoonDir), 0.0), 6.0) * (1.0 - dens * 0.7);
        vec3 c = alb * (uMoonCol * (ml * 0.8 + fwd * 1.5) * mix(1.0, 0.5, dens) + uSkyCol * 1.2);
        /* the lamps and the fire, stronger on the underside */
        c += alb * vL.rgb * (0.3 + 0.7 * max(-N.y, 0.0)) * 0.35;
        c = fogApply(c, vC);
        o = vec4(c * a, a);
    });
static const char *FS_SMOKECOMP = GLSL(
    in vec2 vUV;
    uniform sampler2D uSrc;
    out vec4 o;
    void main() { o = texture(uSrc, vUV); });

/* a telegraph wire: a line a pixel wide, as opaque as the wire is wide */
static const char *FS_WIRE = GLSL(
    in vec3 vP, vN, vL;
    in vec2 vUV;
    flat in int vMat;
    uniform float uPixAng;
    out vec4 o;
    void main() {
        float d = length(vP);
        float a = clamp(0.006 / (d * uPixAng), 0.04, 1.0);
        vec3 V = vP / d;
        vec3 c = vec3(0.004) + uMoonCol * 0.08 * pow(max(dot(-V, uMoonDir), 0.0), 4.0);
        o = vec4(fogApply(c, vP), a);
    });
static const char *FS_NONE = GLSL(
    out vec4 o;
    void main() { o = vec4(0.0); });
