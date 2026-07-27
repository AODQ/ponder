#ifndef UTIL_ROUGHNESS_GLSL
#define UTIL_ROUGHNESS_GLSL

#ifndef f32
#define i32v2 ivec2
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

// openpbr spec, specular_roughness_anisotropy: stretches alpha = roughness
// (engine convention: no perceptual remap) into tangent/bitangent slopes;
// matches cull's own shader-side formula 1:1.
//
// NOTE: the spec's own written formula squares roughness, and cull's
// CPU-side debug visualization code (main.cpp) independently squares
// specularRoughness when it needs the actual GGX alpha for the same
// value -- so cull's shader and the spec/cull's-own-CPU-code disagree
// with each other. a prior version of this function broke from cull to
// match the spec instead; reverted back to cull's shader-side (linear
// roughness) convention for consistency with the rest of the port. if a
// caller genuinely needs alpha = roughness^2, square it explicitly at the
// call site (name it alpha2, not roughness) rather than baking it in here.
/*
	\alpha_t = \alpha \sqrt{\frac{2}{1 + (1 - a)^2}}, \quad
	\alpha_b = (1 - a) \alpha_t
*/
// a = 0 gives (\alpha, \alpha); a = 1 gives (\alpha \sqrt{2}, 0), floored
// so the degenerate axis stays sampleable
f32v2 openPbrRoughnessAlpha(const f32 roughness, const f32 anisotropy) {
	const f32 oneMinusA = 1.0f - anisotropy;
	const f32 alphaT = (
		roughness * sqrt(2.0f / (1.0f + oneMinusA * oneMinusA))
	);
	return max(f32v2(alphaT, oneMinusA * alphaT), f32v2(1e-5f));
}

// openpbr spec, coat roughening: a rough coat effectively roughens the
// base substrate's microfacet lobes (metal and dielectric), modeled as
// the convolution of slope-space gaussians, with the coat variance
// counted twice since the reflection crosses the coat boundary twice
/*
	r'_B = \mathrm{lerp}\Bigl(
		r_B, \bigl(\mathrm{min}(1, r_B^4 + 2 r_C^4)\bigr)^{1/4}, C
	\Bigr)
*/
f32 openPbrCoatRoughenedRoughness(
	const f32 baseRoughness,
	const f32 coatRoughness,
	const f32 coatWeight
) {
	const f32 baseRoughnessSq = baseRoughness * baseRoughness;
	const f32 coatRoughnessSq = coatRoughness * coatRoughness;
	const f32 baseRoughness4 = baseRoughnessSq * baseRoughnessSq;
	const f32 coatRoughness4 = coatRoughnessSq * coatRoughnessSq;
	const f32 roughenedRoughness = (
		pow(min(1.0f, baseRoughness4 + 2.0f * coatRoughness4), 0.25f)
	);
	return mix(baseRoughness, roughenedRoughness, coatWeight);
}

#endif // UTIL_ROUGHNESS_GLSL
