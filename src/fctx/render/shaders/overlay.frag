#version 430 core

in vec2 v_uv;
in vec2 v_local;
in vec2 v_half;
in vec4 v_color;
in vec4 v_color2;
flat in int v_mode;
in float v_radius;
in float v_soft;
in float v_t;

out vec4 f_color;

uniform sampler2D u_font;
uniform sampler2D u_image;

const int MODE_RECT = 0;
const int MODE_IMAGE = 1;
const int MODE_GLYPH = 2;
const int MODE_GRADIENT = 3;

float rounded_box(vec2 p, vec2 b, float r) {
    r = min(r, min(b.x, b.y));
    vec2 q = abs(p) - b + vec2(r);
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec4 c;
    if (v_mode == MODE_GLYPH) {
        // The atlas is a single-channel coverage mask; multiplying it into
        // alpha keeps glyph colour independent of the bake.
        c = vec4(v_color.rgb, v_color.a * texture(u_font, v_uv).r);
    } else if (v_mode == MODE_IMAGE) {
        c = texture(u_image, v_uv) * v_color;
    } else if (v_mode == MODE_GRADIENT) {
        c = mix(v_color, v_color2, clamp(v_t, 0.0, 1.0));
    } else {
        c = v_color;
    }

    if (v_mode != MODE_GLYPH) {
        float d = rounded_box(v_local, v_half, v_radius);
        c.a *= 1.0 - smoothstep(-v_soft, 0.0, d);
    }

    if (c.a <= 0.0) discard;
    f_color = c;
}
