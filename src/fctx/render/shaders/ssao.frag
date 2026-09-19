#version 430 core
#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_depth;
uniform mat4 u_proj;
uniform mat4 u_inv_proj;
uniform vec2 u_noise_scale;
uniform float u_radius;
uniform float u_strength;
uniform float u_bias;
uniform int u_samples;

const int MAX_SAMPLES = 32;

// Golden-angle spiral in the hemisphere.  A fixed random kernel uploaded as a
// uniform array costs a uniform upload per resize and, at 16 samples, clumps
// badly; the spiral is stratified by construction.
vec3 hemi_sample(int i, int n, float jitter) {
    float fi = (float(i) + 0.5) / float(n);
    float ang = (float(i) + jitter) * 2.39996323;
    float r = sqrt(fi);
    float z = sqrt(max(1.0 - fi, 0.0));
    // Bias the sample lengths towards the origin so nearby geometry dominates.
    float scale = mix(0.18, 1.0, fi * fi);
    return vec3(cos(ang) * r, sin(ang) * r, z) * scale;
}

void main() {
    float depth = texture(u_depth, v_uv).r;
    if (depth >= 1.0) { f_color = vec4(1.0); return; }

    vec3 p = view_from_depth(v_uv, depth, u_inv_proj);

    // Derivative normals: the main pass is forward, so there is no normal
    // buffer to read.  Cross-derivatives are exact on flat surfaces and wrong
    // only on the one-pixel silhouette, which the blur then hides.
    vec3 n = normalize(cross(dFdx(p), dFdy(p)));
    if (n.z < 0.0) n = -n;

    vec3 rnd = vec3(ign(gl_FragCoord.xy * u_noise_scale), 0.0, 0.0);
    vec3 t = normalize(vec3(cos(rnd.x * 6.2831853), sin(rnd.x * 6.2831853), 0.0));
    t = normalize(t - n * dot(t, n));
    mat3 tbn = mat3(t, cross(n, t), n);

    int n_samples = clamp(u_samples, 1, MAX_SAMPLES);
    float occl = 0.0;
    for (int i = 0; i < MAX_SAMPLES; ++i) {
        if (i >= n_samples) break;
        vec3 sp = p + tbn * hemi_sample(i, n_samples, rnd.x) * u_radius;
        vec4 clip = u_proj * vec4(sp, 1.0);
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;

        float sd = texture(u_depth, uv).r;
        if (sd >= 1.0) continue;
        float sz = view_from_depth(uv, sd, u_inv_proj).z;

        // Range check stops a foreground object from darkening a wall metres
        // behind it, which is the classic SSAO halo.
        float range = saturate(u_radius / max(abs(p.z - sz), 1e-4));
        occl += (sz >= sp.z + u_bias ? 1.0 : 0.0) * range;
    }

    float ao = 1.0 - (occl / float(n_samples)) * u_strength;
    f_color = vec4(saturate(ao));
}
