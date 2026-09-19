#version 430 core
// One oversized triangle, not a quad: a quad's diagonal seam makes the GPU
// shade the pixels along it twice and can show as a hairline in the composite.
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
