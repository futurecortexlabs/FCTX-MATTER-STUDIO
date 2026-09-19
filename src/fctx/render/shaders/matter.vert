#version 430 core
in vec3 in_pos;
in vec3 in_nrm;

uniform mat4 u_view_proj;

out vec3 v_world;
out vec3 v_nrm;

void main() {
    v_world = in_pos;
    v_nrm = in_nrm;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
