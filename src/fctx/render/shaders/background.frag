#version 430 core
#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform vec3 u_top;
uniform vec3 u_bottom;
uniform vec3 u_glow;
uniform vec2 u_glow_center;
uniform float u_glow_radius;

void main() {
    vec3 c = mix(u_bottom, u_top, pow(saturate(v_uv.y), 0.85));
    float d = length((v_uv - u_glow_center) * vec2(1.0, 0.72));
    c += u_glow * exp(-d * d / max(u_glow_radius * u_glow_radius, 1e-5));
    f_color = vec4(c, 1.0);
}
