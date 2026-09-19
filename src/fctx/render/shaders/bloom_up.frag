#version 430 core
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_scatter;

void main() {
    vec2 t = u_texel * u_scatter;
    vec3 s  = texture(u_src, v_uv + vec2(-1.0,  1.0) * t).rgb;
    s += texture(u_src, v_uv + vec2( 0.0,  1.0) * t).rgb * 2.0;
    s += texture(u_src, v_uv + vec2( 1.0,  1.0) * t).rgb;
    s += texture(u_src, v_uv + vec2(-1.0,  0.0) * t).rgb * 2.0;
    s += texture(u_src, v_uv).rgb * 4.0;
    s += texture(u_src, v_uv + vec2( 1.0,  0.0) * t).rgb * 2.0;
    s += texture(u_src, v_uv + vec2(-1.0, -1.0) * t).rgb;
    s += texture(u_src, v_uv + vec2( 0.0, -1.0) * t).rgb * 2.0;
    s += texture(u_src, v_uv + vec2( 1.0, -1.0) * t).rgb;
    f_color = vec4(s * 0.0625, 1.0);
}
