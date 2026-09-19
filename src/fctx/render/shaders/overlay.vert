#version 430 core
// Every 2D element -- panels, glyphs, the webcam inset, skeleton lines, the
// hardness dial -- is one instance of this quad.  Keeping them in a single
// batch means the whole HUD is one draw call and the order is exactly the
// order they were appended in.
in vec2 in_corner;

in vec4 i_rect;    // x, y, w, h in pixels, origin top-left
in vec4 i_uv;      // u0, v0, u1, v1
in vec4 i_color;
in vec4 i_color2;
in vec4 i_params;  // corner radius px, mode, rotation rad, softness px

uniform vec2 u_resolution;

out vec2 v_uv;
out vec2 v_local;
out vec2 v_half;
out vec4 v_color;
out vec4 v_color2;
flat out int v_mode;
out float v_radius;
out float v_soft;
out float v_t;

void main() {
    vec2 half_size = i_rect.zw * 0.5;
    vec2 centre = i_rect.xy + half_size;
    vec2 local = (in_corner - 0.5) * i_rect.zw;

    float a = i_params.z;
    if (a != 0.0) {
        float c = cos(a), s = sin(a);
        local = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
    }

    vec2 px = centre + local;
    vec2 ndc = vec2(px.x / u_resolution.x * 2.0 - 1.0,
                    1.0 - px.y / u_resolution.y * 2.0);

    v_uv = mix(i_uv.xy, i_uv.zw, in_corner);
    v_local = (in_corner - 0.5) * i_rect.zw;
    v_half = half_size;
    v_color = i_color;
    v_color2 = i_color2;
    v_mode = int(i_params.y + 0.5);
    v_radius = i_params.x;
    v_soft = max(i_params.w, 0.75);
    v_t = in_corner.x;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
