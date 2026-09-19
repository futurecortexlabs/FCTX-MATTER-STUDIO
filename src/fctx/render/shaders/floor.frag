#version 430 core
#include "common.glsl"
#include "shadow.glsl"

in vec3 v_world;
out vec4 f_color;

uniform vec3 u_center;
uniform float u_fade_radius;
uniform float u_grid_spacing;
uniform vec3 u_base_color;
uniform vec3 u_grid_color;
uniform vec3 u_key_color;
uniform vec3 u_key_dir;
uniform vec3 u_sky_color;


uniform sampler2D u_ao_tex;
uniform vec2 u_resolution;
uniform int u_use_ao;

// Analytically antialiased grid: width the line in *screen* space using the
// derivative of the world coordinate, so distant lines fade to a uniform tint
// instead of aliasing into a moire pattern.
float grid_line(vec2 p, float spacing) {
    vec2 q = p / spacing;
    vec2 w = fwidth(q);
    vec2 d = abs(fract(q - 0.5) - 0.5) / max(w, vec2(1e-5));
    return 1.0 - min(min(d.x, d.y), 1.0);
}

void main() {
    vec3 n = vec3(0.0, 1.0, 0.0);
    vec2 p = v_world.xz - u_center.xz;
    float r = length(p);

    float fade = 1.0 - saturate(r / u_fade_radius);
    fade = fade * fade;
    if (fade < 0.002) discard;

    float g = grid_line(p, u_grid_spacing) * 0.55
            + grid_line(p, u_grid_spacing * 4.0) * 0.45;

    vec3 albedo = mix(u_base_color, u_grid_color, saturate(g));

    vec3 key_l = normalize(-u_key_dir);
    // The floor is never drawn into the shadow map, so it has no self-shadow
    // to escape: any bias beyond the base slope term only walks the contact
    // shadow away from the body casting it.
    float shade = sample_shadow(v_world, n, key_l, SHADOW_NEAR);

    float ao = 1.0;
    if (u_use_ao == 1) ao = texture(u_ao_tex, gl_FragCoord.xy / u_resolution).r;

    float nol = max(dot(n, key_l), 0.0);
    vec3 color = albedo * (u_sky_color * 0.9 * ao + u_key_color * nol * shade);

    // A soft pool of light under the stage anchors the body to the ground even
    // when the shadow map is off.  It is modulated by the albedo rather than
    // added flat, so the grid keeps reading through the bright patch instead
    // of being washed out by it.
    color += albedo * u_key_color * 2.1 * fade * fade * ao;

    f_color = vec4(color, fade);
}
