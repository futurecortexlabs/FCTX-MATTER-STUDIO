// Cook-Torrance terms.  v_smith is the height-correlated Smith visibility from
// Heitz 2014, which already folds in the 1/(4 NoL NoV) denominator -- do not
// divide again or specular comes out four times too dark at grazing angles.

float d_ggx(float noh, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float v_smith(float nov, float nol, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float gv = nol * sqrt(nov * nov * (1.0 - a2) + a2);
    float gl = nov * sqrt(nol * nol * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

vec3 f_schlick(vec3 f0, float u) {
    float f = pow(1.0 - u, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

// Wrapped diffuse.  w = 0 is Lambert; larger w drags the terminator around the
// back of the object, which is what makes soft matter read as if light is
// creeping through it instead of stopping dead at the silhouette.
float wrapped_diffuse(float nol, float w) {
    return max((nol + w) / ((1.0 + w) * (1.0 + w)), 0.0);
}

struct Surface {
    vec3 n;
    vec3 v;
    vec3 albedo;
    float rough;
    float metal;
    float translucency;
};

vec3 direct_light(Surface s, vec3 l, vec3 radiance) {
    vec3 h = normalize(l + s.v);
    float nol = dot(s.n, l);
    float nov = max(dot(s.n, s.v), 1e-4);
    float noh = max(dot(s.n, h), 0.0);
    float voh = max(dot(s.v, h), 0.0);

    vec3 f0 = mix(vec3(0.04), s.albedo, s.metal);
    vec3 f = f_schlick(f0, voh);
    float spec = d_ggx(noh, s.rough) * v_smith(nov, max(nol, 1e-4), s.rough);

    vec3 kd = (vec3(1.0) - f) * (1.0 - s.metal);
    float diff = wrapped_diffuse(nol, s.translucency * 0.85);

    // Back-translucency: light that entered the far side and scattered out
    // towards the eye.  Keyed off -NoL so it only appears where the surface
    // faces away from the lamp, and off VoL so it peaks when looking into it.
    float back = pow(saturate(dot(s.v, -l)), 3.0) * saturate(-nol) * s.translucency;

    return radiance * (kd * s.albedo * INV_PI * diff
                     + f * spec * max(nol, 0.0)
                     + s.albedo * back * 0.55);
}

vec3 hemisphere_ambient(vec3 n, vec3 sky, vec3 ground) {
    return mix(ground, sky, n.y * 0.5 + 0.5);
}

// Lazarov's analytic fit to the split-sum environment BRDF.  The alternative
// is a precomputed LUT texture, which would be one more resource to bake and
// bind for a term whose error here is well under a quantisation step.
vec3 env_brdf(vec3 f0, float rough, float nov) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Ambient from a two-colour hemisphere probe, diffuse plus specular.
// Without the specular half a metallic surface has nothing to reflect except
// the two lamps and renders as a black ball with two dots on it -- which is
// exactly the hard end of the hardness dial.
vec3 ambient_light(Surface s, vec3 sky, vec3 ground, float spec_gain) {
    float nov = max(dot(s.n, s.v), 1e-4);
    vec3 f0 = mix(vec3(0.04), s.albedo, s.metal);

    vec3 diffuse = hemisphere_ambient(s.n, sky, ground) * s.albedo
                 * (1.0 - s.metal);
    vec3 refl = reflect(-s.v, s.n);
    vec3 probe = hemisphere_ambient(refl, sky * spec_gain, ground);
    // Rough surfaces gather from the whole hemisphere, so they see the average
    // of the probe rather than the direction the mirror would pick.
    probe = mix(probe, hemisphere_ambient(s.n, sky * spec_gain, ground),
                s.rough * s.rough);
    return diffuse + probe * env_brdf(f0, s.rough, nov);
}
