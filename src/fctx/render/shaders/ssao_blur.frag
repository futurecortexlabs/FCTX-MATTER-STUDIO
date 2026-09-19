#version 430 core
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_ao;
uniform vec2 u_texel;
void main() {
    float sum = 0.0;
    for (int y = -2; y <= 1; ++y)
        for (int x = -2; x <= 1; ++x)
            sum += texture(u_ao, v_uv + vec2(float(x) + 0.5, float(y) + 0.5) * u_texel).r;
    f_color = vec4(sum * 0.0625);
}
