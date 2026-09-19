const float PI = 3.14159265358979;
const float INV_PI = 0.31830988618379;

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate3(vec3 v) { return clamp(v, vec3(0.0), vec3(1.0)); }

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Interleaved gradient noise (Jimenez 2014).  Cheaper than a texture lookup and
// it decorrelates across neighbouring pixels far better than fract(sin(dot(..)))
// hashes, which band visibly once the result is used to rotate a sample kernel.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Reconstruct a view-space position from a hardware depth sample.
vec3 view_from_depth(vec2 uv, float depth, mat4 inv_proj) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 v = inv_proj * ndc;
    return v.xyz / v.w;
}
