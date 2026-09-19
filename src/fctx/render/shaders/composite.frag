#version 430 core
#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_strength;
uniform float u_vignette;
uniform float u_aberration;
uniform int u_aces;

// ACES fitted to a 3x3 in/out pair (Narkowicz/Hill).  The one-line Narkowicz
// curve alone desaturates bright reds badly, which is exactly the colour the
// soft end of the hardness dial sits on.
const mat3 ACES_IN = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777);
const mat3 ACES_OUT = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602);

vec3 aces_fitted(vec3 c) {
    c = ACES_IN * c;
    vec3 a = c * (c + 0.0245786) - 0.000090537;
    vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    return saturate3(ACES_OUT * (a / b));
}

vec3 reinhard(vec3 c) { return c / (c + vec3(1.0)); }

// 4x4 ordered Bayer.  A float framebuffer holds a smooth gradient perfectly
// and then the 8-bit write quantises it into visible bands across the
// backdrop; one LSB of ordered noise before the write removes them.
float bayer4(vec2 p) {
    ivec2 i = ivec2(mod(p, 4.0));
    const int M[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return float(M[i.y * 4 + i.x]) / 16.0;
}

void main() {
    vec2 centred = v_uv - 0.5;
    float r2 = dot(centred, centred);

    vec3 scene;
    if (u_aberration > 0.0) {
        // Lateral chromatic aberration only: the offset grows with radius, so
        // the centre of frame -- where the matter is -- stays sharp.
        vec2 off = centred * u_aberration * r2 * 4.0;
        scene.r = texture(u_scene, v_uv + off).r;
        scene.g = texture(u_scene, v_uv).g;
        scene.b = texture(u_scene, v_uv - off).b;
    } else {
        scene = texture(u_scene, v_uv).rgb;
    }

    scene += texture(u_bloom, v_uv).rgb * u_bloom_strength;
    scene *= u_exposure;

    vec3 mapped = u_aces == 1 ? aces_fitted(scene) : saturate3(reinhard(scene));

    float vig = 1.0 - u_vignette * saturate(r2 * 2.1);
    mapped *= vig * vig;

    mapped = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    mapped += (bayer4(gl_FragCoord.xy) - 0.5) / 255.0;

    f_color = vec4(mapped, 1.0);
}
