// Turn a point sprite into a sphere: analytic normal from the sprite
// coordinate and a corrected fragment depth.  Without the depth write the
// grains are flat discs that slice through the floor plane along a hard line.
struct Impostor {
    vec3 view_pos;
    vec3 view_nrm;
    float depth;
};

bool impostor_resolve(vec3 view_center, float radius, mat4 proj, out Impostor imp) {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    p.y = -p.y;
    float r2 = dot(p, p);
    if (r2 > 1.0) return false;
    imp.view_nrm = vec3(p, sqrt(1.0 - r2));
    imp.view_pos = view_center + imp.view_nrm * radius;
    vec4 clip = proj * vec4(imp.view_pos, 1.0);
    imp.depth = (clip.z / clip.w) * 0.5 + 0.5;
    return true;
}
