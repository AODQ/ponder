module "util-roughness";
__exported import "util-types";

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
public func openPbrRoughnessAlpha(roughness: f32, anisotropy: f32) -> f32v2 {
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
public func openPbrCoatRoughenedRoughness(
	baseRoughness: f32,
	coatRoughness: f32,
	coatWeight: f32,
) -> f32 {
	const f32 baseRoughnessSq = baseRoughness * baseRoughness;
	const f32 coatRoughnessSq = coatRoughness * coatRoughness;
	const f32 baseRoughness4 = baseRoughnessSq * baseRoughnessSq;
	const f32 coatRoughness4 = coatRoughnessSq * coatRoughnessSq;
	const f32 roughenedRoughness = (
		pow(min(1.0f, baseRoughness4 + 2.0f * coatRoughness4), 0.25f)
	);
	return lerp(baseRoughness, roughenedRoughness, coatWeight);
}
