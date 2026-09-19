#version 430 core
#include "common.glsl"
#include "pbr.glsl"
#include "shadow.glsl"

in vec3 v_world;
in vec3 v_nrm;
out vec4 f_color;

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


uniform sampler2D u_ao_tex;
uniform vec2 u_resolution;
uniform int u_use_ao;

uniform int u_double_sided;
uniform float u_emissive;
uniform float u_spec_gain;

void main() {
    vec3 n = normalize(v_nrm);
    if (dot(n, n) < 1e-8) n = vec3(0.0, 1.0, 0.0);

    // Cloth is one particle thick, so half its triangles face away from the
    // camera.  Without this flip the underside of a fold shades with an
    // inverted normal and comes out flat black, which reads as a hole.
    if (u_double_sided == 1 && !gl_FrontFacing) n = -n;

    vec3 v = normalize(u_cam_pos - v_world);

    Surface s;
    s.n = n;
    s.v = v;
    s.albedo = u_base_color;
    s.rough = clamp(u_roughness, 0.045, 1.0);
    s.metal = clamp(u_metallic, 0.0, 1.0);
    s.translucency = clamp(u_translucency, 0.0, 1.0);

    vec3 key_l = normalize(-u_key_dir);
    float shade = sample_shadow(v_world, n, key_l,
                                u_double_sided == 1 ? SHADOW_SHEET : SHADOW_CLOSED);

    // Translucent matter must not be fully extinguished by its own shadow or
    // the jelly look dies the moment anything occludes the key light.
    shade = mix(shade, mix(shade, 1.0, 0.45), s.translucency);

    vec3 color = direct_light(s, key_l, u_key_color) * shade;
    color += direct_light(s, normalize(-u_rim_dir), u_rim_color);

    float ao = 1.0;
    if (u_use_ao == 1) ao = texture(u_ao_tex, gl_FragCoord.xy / u_resolution).r;

    color += ambient_light(s, u_sky_color, u_ground_color, u_spec_gain) * ao;
    color += s.albedo * u_emissive;

    f_color = vec4(color, 1.0);
}
