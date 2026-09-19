#version 430 core
#include "common.glsl"

in vec3 v_world;
in vec3 v_nrm;
in vec4 v_color;
out vec4 f_color;

uniform vec3 u_cam_pos;
uniform vec3 u_key_color;
uniform vec3 u_key_dir;
uniform float u_emissive;

void main() {
    vec3 n = normalize(v_nrm);
    vec3 v = normalize(u_cam_pos - v_world);
    float nol = max(dot(n, normalize(-u_key_dir)), 0.0);
    float fres = pow(1.0 - saturate(dot(n, v)), 2.5);

    // Hands are a UI element standing in for the user's body, not matter, so
    // they are lit as an emissive volume: readable against any background and
    // never swallowed by the scene's own shadows.
    vec3 base = v_color.rgb;
    vec3 color = base * (u_emissive + 0.35 * nol)
               + u_key_color * 0.12 * nol
               + base * fres * 1.9;

    // Tracking confidence dims the hand rather than making it translucent.
    // Alpha blending a capsule mesh against itself shows every internal
    // surface, and a hand that goes see-through as it is predicted forward
    // looks like a rendering fault rather than like uncertainty.
    f_color = vec4(color * v_color.a, 1.0);
}
