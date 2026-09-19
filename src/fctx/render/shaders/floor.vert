#version 430 core
in vec3 in_pos;
uniform mat4 u_view_proj;
out vec3 v_world;
void main() {
    v_world = in_pos;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
