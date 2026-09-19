// 16-tap Poisson disk.  A regular grid of the same tap count aliases into
// visible stair-steps along a slowly rotating shadow edge; the irregular disk
// turns that into noise, which the per-pixel rotation then averages away.
const vec2 POISSON16[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// How this receiver's own geometry sits in the shadow map.  Two independent
// facts used to ride on one "two_sided" flag -- whether the normal may be
// flipped towards the light, and how much slope bias the surface can absorb
// -- and everything that was neither cloth nor a closed body silently took
// the closed-body branch.
const int SHADOW_SHEET  = 0;  // one particle thick: the map holds whichever face the light saw
const int SHADOW_CLOSED = 1;  // solid drawn back-face only: the map holds its *far* side
const int SHADOW_NEAR   = 2;  // the map holds its near side, or it never writes to the map

struct ShadowParams {
    mat4 light_vp;
    float texel_world;   // metres covered by one shadow texel
    float texel_uv;      // 1 / shadow_size
    float depth_range;   // metres spanned by depth 0..1 in the light frustum
    float softness;
    int receiver;        // one of SHADOW_SHEET / SHADOW_CLOSED / SHADOW_NEAR
};

float shadow_pcf(sampler2D map, vec3 world_pos, vec3 n, vec3 l, ShadowParams sp) {
    // Cloth has no inside: whichever face the light sees is the one that
    // should be offset away from it.  Without this the offset on the back of
    // a fold pushes the lookup *into* the occluder and the fold shadows
    // itself in a band of noise.
    float ndl = dot(n, l);
    if (sp.receiver == SHADOW_SHEET && ndl < 0.0) { n = -n; ndl = -ndl; }

    // Slope-scaled, in metres.  A normal offset alone is enough for a closed
    // body rendered back-face-only, but a single-layer sheet stores its own
    // depth, so the bias has to grow like tan(theta) as the surface turns
    // edge-on to the light or the acne comes straight back.
    float cos_t = max(ndl, 0.05);
    float tan_t = min(sqrt(max(1.0 - cos_t * cos_t, 0.0)) / cos_t, 6.0);
    float offset_m = sp.texel_world * (1.0 + 2.2 * tan_t);

    // A closed body stores its far surface, so most of it needs no bias at
    // all -- but near the silhouette the two surfaces converge to within a
    // texel of each other and it acnes in a band around the rim.  A solid can
    // afford a far larger offset than a sheet can, because the shadow it
    // receives from itself is never a contact shadow: the near surface it
    // would have to escape is a whole body thickness away.  Nothing else can
    // afford it.  Grains store their near surface and are 9.6 mm across, and
    // at this multiplier the lookup lands several grains away -- which erases
    // exactly the crevice shadows that make a pile read as touching spheres.
    if (sp.receiver == SHADOW_CLOSED) offset_m *= 1.0 + 7.0 * tan_t;

    // Along the normal to escape the occluder sideways, and along the light
    // to escape it in depth; neither alone covers every incidence angle.
    vec3 p = world_pos + n * offset_m + l * (offset_m * 0.6);

    vec4 lp = sp.light_vp * vec4(p, 1.0);
    vec3 proj = lp.xyz / lp.w * 0.5 + 0.5;
    if (proj.z >= 1.0 || proj.z <= 0.0) return 1.0;
    if (any(lessThan(proj.xy, vec2(0.0))) || any(greaterThan(proj.xy, vec2(1.0))))
        return 1.0;

    float bias = offset_m / max(sp.depth_range, 1e-4);
    float radius = sp.texel_uv * sp.softness * 1.6;
    float angle = ign(gl_FragCoord.xy) * 6.2831853;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    float sum = 0.0;
    for (int i = 0; i < 16; ++i) {
        float d = texture(map, proj.xy + rot * POISSON16[i] * radius).r;
        sum += (proj.z - bias <= d) ? 1.0 : 0.0;
    }
    return sum * 0.0625;
}

// Shared uniform block for every pass that receives shadow.  Keeping the
// declarations here means the three fragment shaders cannot drift out of sync
// with what the renderer uploads.
uniform sampler2D u_shadow_map;
uniform mat4 u_light_vp;
uniform float u_shadow_texel;
uniform float u_shadow_texel_uv;
uniform float u_shadow_depth_range;
uniform float u_shadow_softness;
uniform int u_use_shadow;

float sample_shadow(vec3 world_pos, vec3 n, vec3 l, int receiver) {
    if (u_use_shadow == 0) return 1.0;
    ShadowParams sp;
    sp.light_vp = u_light_vp;
    sp.texel_world = u_shadow_texel;
    sp.texel_uv = u_shadow_texel_uv;
    sp.depth_range = u_shadow_depth_range;
    sp.softness = u_shadow_softness;
    sp.receiver = receiver;
    return shadow_pcf(u_shadow_map, world_pos, n, l, sp);
}
