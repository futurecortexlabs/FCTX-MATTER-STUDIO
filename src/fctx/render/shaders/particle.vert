#version 430 core
in vec3 in_pos;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_radius;
uniform float u_point_scale;
uniform int u_ortho;

out vec3 v_view_center;

void main() {
    vec4 vp = u_view * vec4(in_pos, 1.0);
    v_view_center = vp.xyz;
    gl_Position = u_proj * vp;
    // Under perspective the sprite must shrink with distance; under the light's
    // orthographic projection it must not, or grains near the far plane of the
    // shadow frustum cast pinhole shadows.
    gl_PointSize = u_ortho == 1
        ? 2.0 * u_radius * u_point_scale
        : 2.0 * u_radius * u_point_scale / max(-vp.z, 1e-4);
}
