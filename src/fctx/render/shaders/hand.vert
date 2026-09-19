#version 430 core
// A capsule is drawn as one sphere shell whose two halves are pulled apart
// onto the segment endpoints.  in_side selects the endpoint, so the equator
// ring (duplicated on the CPU) becomes the cylindrical waist for free and no
// separate cylinder geometry or per-bone mesh rebuild is needed.
in vec3 in_dir;
in float in_side;

in vec3 i_a;
in vec3 i_b;
in float i_r;
in vec4 i_color;

uniform mat4 u_view_proj;

out vec3 v_world;
out vec3 v_nrm;
out vec4 v_color;

void main() {
    vec3 axis = i_b - i_a;
    float len = length(axis);
    vec3 ay = len > 1e-7 ? axis / len : vec3(0.0, 1.0, 0.0);
    // Pick the reference vector least parallel to the axis, otherwise the
    // cross product collapses for bones that happen to point straight up.
    vec3 ref = abs(ay.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 ax = normalize(cross(ref, ay));
    vec3 az = cross(ax, ay);

    vec3 dir = ax * in_dir.x + ay * in_dir.y + az * in_dir.z;
    vec3 base = mix(i_a, i_b, in_side);

    v_world = base + dir * i_r;
    v_nrm = dir;
    v_color = i_color;
    gl_Position = u_view_proj * vec4(v_world, 1.0);
}
