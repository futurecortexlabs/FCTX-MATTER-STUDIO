#version 430 core
#include "common.glsl"
#include "pbr.glsl"
#include "shadow.glsl"
#include "particle_common.glsl"

in vec3 v_view_center;
out vec4 f_color;

uniform mat4 u_proj;
uniform mat4 u_inv_view;
uniform float u_radius;

uniform vec3 u_cam_pos;
uniform vec3 u_base_color;
uniform float u_roughness;
uniform float u_metallic;
uniform float u_translucency;

uniform vec3 u_key_dir;
uniform vec3 u_key_color;
uniform vec3 u_rim_dir;
uniform vec3 u_rim_color;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_spec_gain;


void main() {
    Impostor imp;
    if (!impostor_resolve(v_view_center, u_radius, u_proj, imp)) discard;
    gl_FragDepth = imp.depth;

    vec3 world = (u_inv_view * vec4(imp.view_pos, 1.0)).xyz;
    vec3 n = normalize(mat3(u_inv_view) * imp.view_nrm);
    vec3 v = normalize(u_cam_pos - world);

    Surface s;
    s.n = n;
    s.v = v;
    s.albedo = u_base_color;
    s.rough = clamp(u_roughness, 0.045, 1.0);
    s.metal = clamp(u_metallic, 0.0, 1.0);
    s.translucency = clamp(u_translucency, 0.0, 1.0);

    vec3 key_l = normalize(-u_key_dir);
    // Grains go into the shadow map as point impostors with culling off, so
    // the map holds their near surface: the same slope bias a sheet needs,
    // and no more, or a grain stops being shadowed by the one beside it.
    float shade = sample_shadow(world, n, key_l, SHADOW_NEAR);

    vec3 color = direct_light(s, key_l, u_key_color) * shade;
    color += direct_light(s, normalize(-u_rim_dir), u_rim_color);

    // Grains sit in a pile, so the strongest occlusion cue is simply how far
    // the fragment is from the sprite centre; SSAO at half resolution cannot
    // resolve gaps this small.
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float cavity = mix(0.45, 1.0, sqrt(max(1.0 - dot(pc, pc), 0.0)));
    color += ambient_light(s, u_sky_color, u_ground_color, u_spec_gain) * cavity;

    f_color = vec4(color, 1.0);
}
