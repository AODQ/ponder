implementing bsdf;

import types;
import material;

namespace vkof {

// fujii oren-nayar constants
public static const f32 skFon1 = 0.5f - 2.0f / (3.0f * π);
public static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * π);

// -----------------------------------------------------------------------------
// -- roughness
// -----------------------------------------------------------------------------

// openpbr spec, specular_roughness_anisotropy stretches alpha = roughness
// into tangent/bitangent slopes
//
// TODO NOTE: roughness=α or roughness=α² are inconsistent throughout
// the entire codebase, probably need to standardize
/*

	\alpha_t = \alpha \sqrt{\frac{2}{1 + (1 - a)^2}}, \quad
	\alpha_b = (1 - a) \alpha_t
*/
// a = 0 as (\alpha, \alpha); a = 1 as (\alpha \sqrt{2}, 0)
public func bsdfOpenPbrRoughnessAlpha(roughness: f32, anisotropy: f32) -> f32v2 {
	let oneMinusA: f32 = 1.0f - anisotropy;
	let αT: f32 = (
		roughness * sqrt(2.0f / (1.0f + oneMinusA * oneMinusA))
	);
	return max(f32v2(αT, oneMinusA * αT), f32v2(1e-5f));
}

// roughens the base substrate's microfacet lobes
/*
	r'_B = \mathrm{lerp}\Bigl(
		r_B, \bigl(\mathrm{min}(1, r_B^4 + 2 r_C^4)\bigr)^{1/4}, C
	\Bigr)
*/
public func bsdfOpenPbrCoatRoughenedRoughness(
	baseRoughness: f32,
	coatRoughness: f32,
	coatWeight: f32,
) -> f32 {
	let baseRoughnessSq: f32 = baseRoughness * baseRoughness;
	let coatRoughnessSq: f32 = coatRoughness * coatRoughness;
	let baseRoughness4: f32 = baseRoughnessSq * baseRoughnessSq;
	let coatRoughness4: f32 = coatRoughnessSq * coatRoughnessSq;
	let roughenedRoughness: f32 = (
		pow(min(1.0f, baseRoughness4 + 2.0f * coatRoughness4), 0.25f)
	);
	return lerp(baseRoughness, roughenedRoughness, coatWeight);
}

// -----------------------------------------------------------------------------
// -- index of refraction
// -----------------------------------------------------------------------------

// openpbr spec, coat:
// base dielectric immersed in the coat
// η ratio relative to the coat
/*
	\eta_s = \mathrm{lerp}(n_b/n_a, n_b/n_c, C)
		\tag{openpbr spec, specular_ior_ratio}\\
	% kutz 2021: inverting the ratio when n_c > n_b kills the spurious
	% grazing tir ring a lobe-mixture model cannot otherwise avoid
	\eta_{bc} \rightarrow n_c/n_b \quad \mathrm{if} \; n_c > n_b
		\tag{openpbr spec, specular_ior_ratio_with_tir_fix}
*/
public func bsdfOpenPbrCoatAffectedIor(
	specularIor: f32,
	coatIor: f32,
	coatWeight: f32,
) -> f32 {
	if (coatWeight <= 0.0f) {
		return specularIor;
	}
	let ratio: f32 = specularIor / max(coatIor, 1e-5f);
	let tirFixed: f32 = ratio > 1.0f ? ratio : 1.0f / max(ratio, 1e-5f);
	return lerp(specularIor, tirFixed, coatWeight);
}

// specular weight scales f0 via an effective ior
public func bsdfOpenPbrEffectiveIor(
	ior: f32,
	weight: f32,
) -> f32 {
	let r: f32 = (ior - 1.0f) / (ior + 1.0f);
	let f0: f32 = clamp(weight * r * r, 0.0f, 0.999f);
	// \epsilon = \mathrm{sgn}(\eta_s - 1)\sqrt{\xi_s F_s}
	let s: f32 = sign(ior - 1.0f) * sqrt(f0);
	return (1.0f + s) / (1.0f - s);
}

// the base dielectric interface's ior
public func bsdfOpenPbrSpecularEffectiveIor(
	mat: MaterialBackend_OpenPbr.Params,
) -> f32 {
	return (
		bsdfOpenPbrEffectiveIor(
			bsdfOpenPbrCoatAffectedIor(
				mat.specularIor, mat.coatIor, mat.coatWeight
			),
			mat.specularWeight
		)
	);
}

// inverse of schlick η->f0 relationship per channel
// to recover an effective real η from a tinted metal's f0
public func bsdfOpenPbrIorFromF0(
	f0: vkof::Spectrum,
) -> vkof::Spectrum {
	let s: vkof::Spectrum = sqrt(clamp(f0, 0.0f, 0.9999f));
	return (1.0f + s) / (1.0f - s);
}

} // namespace vkof
