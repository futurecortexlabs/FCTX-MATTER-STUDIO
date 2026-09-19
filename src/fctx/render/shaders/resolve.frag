#version 430 core
#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_resolution;

#if SAMPLE_COUNT > 1
uniform sampler2DMS u_src;

// Tonemap-weighted resolve (Karis).  Averaging HDR samples linearly lets one
// very bright subsample dominate a partly covered edge pixel, so the MSAA
// edge stays jagged exactly where the contrast is highest -- specular
// highlights on the matter.  Weighting by 1/(1+luma) resolves in a
// perceptually flat space and then undoes the weight.
void main() {
    ivec2 p = ivec2(v_uv * u_resolution);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        vec3 c = texelFetch(u_src, p, i).rgb;
        float w = 1.0 / (1.0 + luminance(c));
        sum += c * w;
        wsum += w;
    }
    vec3 c = sum / max(wsum, 1e-5);
    f_color = vec4(c, 1.0);
}
#else
uniform sampler2D u_src;
void main() { f_color = vec4(texture(u_src, v_uv).rgb, 1.0); }
#endif
