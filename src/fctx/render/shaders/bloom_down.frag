#version 430 core
#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_src;
uniform vec2 u_texel;
uniform int u_prefilter;
uniform float u_threshold;
uniform float u_knee;

// Jimenez's 13-tap downsample.  A plain bilinear halving loses every other
// texel once the chain is six levels deep and the bloom starts to flicker on
// small moving highlights.
vec3 fetch(vec2 uv) { return texture(u_src, uv).rgb; }

float karis_weight(vec3 c) { return 1.0 / (1.0 + luminance(c)); }

vec3 karis_group(vec3 a, vec3 b, vec3 c, vec3 d) {
    float wa = karis_weight(a), wb = karis_weight(b);
    float wc = karis_weight(c), wd = karis_weight(d);
    return (a * wa + b * wb + c * wc + d * wd) / max(wa + wb + wc + wd, 1e-5);
}

void main() {
    vec2 t = u_texel;
    vec3 a = fetch(v_uv + vec2(-2.0, 2.0) * t);
    vec3 b = fetch(v_uv + vec2( 0.0, 2.0) * t);
    vec3 c = fetch(v_uv + vec2( 2.0, 2.0) * t);
    vec3 d = fetch(v_uv + vec2(-2.0, 0.0) * t);
    vec3 e = fetch(v_uv);
    vec3 f = fetch(v_uv + vec2( 2.0, 0.0) * t);
    vec3 g = fetch(v_uv + vec2(-2.0,-2.0) * t);
    vec3 h = fetch(v_uv + vec2( 0.0,-2.0) * t);
    vec3 i = fetch(v_uv + vec2( 2.0,-2.0) * t);
    vec3 j = fetch(v_uv + vec2(-1.0, 1.0) * t);
    vec3 k = fetch(v_uv + vec2( 1.0, 1.0) * t);
    vec3 l = fetch(v_uv + vec2(-1.0,-1.0) * t);
    vec3 m = fetch(v_uv + vec2( 1.0,-1.0) * t);

    vec3 result;
    if (u_prefilter == 1) {
        // The Karis average belongs on the first downsample only: weighting by
        // 1/(1+luma) there is what stops a single blown-out pixel from turning
        // into a persistent flickering star once the chain blurs it.
        result  = karis_group(j, k, l, m) * 0.5;
        result += karis_group(a, b, d, e) * 0.125;
        result += karis_group(b, c, e, f) * 0.125;
        result += karis_group(d, e, g, h) * 0.125;
        result += karis_group(e, f, h, i) * 0.125;

        float lum = luminance(result);
        float soft = clamp(lum - u_threshold + u_knee, 0.0, 2.0 * u_knee);
        soft = soft * soft / max(4.0 * u_knee, 1e-4);
        float contrib = max(soft, lum - u_threshold) / max(lum, 1e-5);
        result *= contrib;
    } else {
        result  = e * 0.125;
        result += (a + c + g + i) * 0.03125;
        result += (b + d + f + h) * 0.0625;
        result += (j + k + l + m) * 0.125;
    }
    f_color = vec4(max(result, vec3(0.0)), 1.0);
}
