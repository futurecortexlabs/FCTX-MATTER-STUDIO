#version 430 core
#include "particle_common.glsl"

in vec3 v_view_center;
uniform mat4 u_proj;
uniform float u_radius;

void main() {
    Impostor imp;
    if (!impostor_resolve(v_view_center, u_radius, u_proj, imp)) discard;
    gl_FragDepth = imp.depth;
}
