implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

// clear coat, rough dielectric layer over the base substrate
/*
	\begin{align*}
	&\textbf{(1) coat specular} \\
	% same microfacet dielectric as the base specular lobe, with the coat
	% ior / roughness / anisotropy and turquin compensation
	&f_{coat} = F \, D \, V \, k_{ms}
		\tag{openpbr spec, coat}\\
	&\textbf{(2) transmission through the coat} \\
	% single directional albedo at the camera-side direction; the spec's
	% albedo-scaling is deliberately non-reciprocal
	&T = (1 - E_{coat}(\omega_o)) \, C_{coat}
		\tag{openpbr spec, non-reciprocal-albedo-scaling-with-T}\\
	&\textbf{(3) view-dependent absorption} \\
	&\mu_i^t = \sqrt{1 - (1 - \mu_i^2) / \eta_c^2}
		\tag{openpbr spec, coat, eq. 72}\\
	&T_\mathrm{coat}^{1/\mu_i^t + 1/\mu_o^t}
		\tag{openpbr spec, coat, eq. 71}\\
	&\textbf{(4) coat darkening} \\
	% \delta = coat_darkening: physically-correct darkening at \delta=1;
	% at \delta=0 the base albedo is boosted to counteract the
	% darkening entirely, so the coated base albedo matches the
	% uncoated base color (eq. 61, eq. 62); the practical implementation
	% and approximation openpbr suggests is eq 70.
	&E_c = F_0 + E_b(1 - F_0) = \mathrm{lerp}(F_0, 1, E_b)
		\tag{openpbr spec, coat darkening, eq. 61}\\
	&B(\delta) = \mathrm{lerp}(B_0, 1, \delta)
		\tag{openpbr spec, coat darkening, eq. 62}\\
	% interfaced-lambertian derivation for context, but only eq 70 at the
	% bottom is implemented
	&E_c(\omega_o) = F(\omega_o, \eta_c)
		+ E_b \bigl(1 - F(\omega_o, \eta_c)\bigr) \Delta(E_b, \eta_c)
		\tag{openpbr spec, coat darkening, eq. 63}\\
	&\Delta(E_b, \eta_c) = \frac{1 - K}{1 - E_b K}
		\tag{openpbr spec, coat darkening, eq. 64}\\
	% K_r: lambertian-base limit; E_F(\eta_c) is the hemispherical coat
	% fresnel albedo, approximated with kulla-conty closed form for F_{avg}
	&K_r = 1 - \bigl(1 - E_F(\eta_c)\bigr) / \eta_c^2
		\tag{openpbr spec, coat darkening, eq. 65}\\
	% K_s: smooth-base limit
	&K_s = F(\omega_o, \eta_c)
		\tag{openpbr spec, coat darkening, eq. 66}\\
	&K = \mathrm{lerp}(K_s, K_r, r_b)
		\tag{openpbr spec, coat darkening, eq. 67}\\
	% r_b: effective base roughness, dielectric/metal blended by
	% base_metalness M; r_m = specular_roughness directly
	&r_b = \mathrm{lerp}(r_d, r_m, M)
		\tag{openpbr spec, coat darkening, eq. 68}\\
	% r_d: fully rough base pulled toward the microfacet roughness by
	% the base dielectric fresnel F_s and specular weight \xi_s
	&r_d = \mathrm{lerp}(1, r, \xi_s F_s)
		\tag{openpbr spec, coat darkening, eq. 69}\\
	% modulated darkening factor: applied uniformly to the whole base
	% term (both the coated and uncoated fraction), taking into account
	% coat_weight C and coat_darkening \delta
	&\mathrm{lerp}(1, \Delta, C \delta)
		\tag{openpbr spec, coat darkening, eq. 70}\\
	&\textbf{(5) layering} \\
	&f = C f_{coat}
		+ \mathrm{lerp}(1, \Delta, C \delta)
		\, \text{mix}(f_{base}, T f_{base}, C)
		\tag{openpbr spec, figure 96 + eq. 70}\\
	\end{align*}
*/
public func bsdfOpenPbrCoatEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	coatFrame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	baseSubstrate: vkof::Spectrum,
	isInsideMedium: bool,
	λ: Wavelengths,
) -> vkof::Spectrum {
	// coat specular reflection
	let iorRatio: f32 = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	let f0: f32 = iorRatio * iorRatio;
	if (f0 == 0.0f || mat.coatWeight <= 0.0f) {
		// no coat layer
		return baseSubstrate;
	}
	// per-axis slopes from the coat roughness + anisotropy
	let α: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
	);
	let ωiLocal: f32v3 = (
		f32v3(
			dot(ωi, coatFrame.tanX), dot(ωi, coatFrame.tanY), dot(ωi, coatFrame.nor)
		)
	);
	let ωoLocal: f32v3 = (
		f32v3(
			dot(ωo, coatFrame.tanX), dot(ωo, coatFrame.tanY), dot(ωo, coatFrame.nor)
		)
	);
	var hLocal: f32v3;
	let hValid: bool = bsdfHalfVectorLocal(ωiLocal, ωoLocal, hLocal);
	let dotHWi: f32 = dot(hLocal, ωiLocal);
	// a non-positive dotHWi is a back-facing microfacet, which has no reflection
	let hFacing: bool = hValid && dotHWi > 0.0f;
	// relative ior
	let coatηRel: f32 = isInsideMedium ? (1.0f / mat.coatIor) : mat.coatIor;
	let coatFresnel: f32 = (
		hFacing ? bsdfFresnelDielectric(dotHWi, coatηRel) : 0.0f
	);
	let coatDistribution: f32 = (
		bsdfMicrofacetGgxDistributionAniso(hLocal, α)
	);
	let coatVisibility: f32 = (
		bsdfMicrofacetSmithGgxVisibilityAniso(ωiLocal, ωoLocal, α)
	);
	// turquin 2019 compensation, consistent with the base dielectric lobe
	let coatEnergyCompensation: f32 = (
		bsdfMicrofacetDielectricEnergyCompensate(
			ωiLocal.z, sqrt(α.x * α.y), f0
		)
	);
	// (1) f_{coat}, coat weight folded into the layering here
	let fCoat: f32 = (
		mat.coatWeight
		// F D V
		* coatFresnel * coatDistribution * coatVisibility
		// k_{ms}
		* coatEnergyCompensation
	);
	// (2) transmission through the coat, non-reciprocal albedo scaling
	let coatAlbedoWi: f32 = (
		// E(\mu_o) in spec notation
		bsdfMicrofacetDielectricAlbedo(
			ωiLocal.z, mat.coatRoughness, f0
		)
	);
	// (3) view-dependent absorption
	let μItSq: f32 = (
		max(
			// 1 - (1 - \mu_i^2) / \eta_c^2
			1.0f - (1.0f - ωiLocal.z * ωiLocal.z) / (coatηRel * coatηRel),
			1e-4f
		)
	);
	let ωoZClamped: f32 = max(ωoLocal.z, 0.0f);
	let μOtSq: f32 = (
		max(
			// 1 - (1 - \mu_o^2) / \eta_c^2
			1.0f - (1.0f - ωoZClamped * ωoZClamped) / (coatηRel * coatηRel),
			1e-4f
		)
	);
	// eq 71
	let absorptionExponent: f32 = (
		// 1 / \mu_i^t + 1 / \mu_o^t
		0.5f * (1.0f / sqrt(μItSq) + 1.0f / sqrt(μOtSq))
	);
	let coatAbsorption: vkof::Spectrum = (
		pow(
			clamp(mat.coatColor, 1e-6f, 1.0f),
			absorptionExponent
		)
	);
	let transmittedBase: vkof::Spectrum = (
		// 1 - E_{coat}(\omega_o)
		(1.0f - coatAlbedoWi)
		* coatAbsorption
		* baseSubstrate
	);
	// (4) coat darkening; fresnel terms use the ior as coat lobe
	let kSmooth: f32 = (
		bsdfFresnelDielectric(max(ωoLocal.z, 0.0f), coatηRel)
	);
	// hemispherical average coat fresnel, kulla & conty closed form;
	// K_r clamps at zero when seen from inside the medium (eta < 1)
	let coatFresnelAvg: f32 = f0 + (1.0f - f0) * (1.0f / 21.0f);
	let kRough: f32 = (
		clamp(
			1.0f - (1.0f - coatFresnelAvg) / (coatηRel * coatηRel),
			0.0f,
			1.0f
		)
	);
	// r_d (eq 69): F_s is the unmodulated base dielectric fresnel
	let baseFresnelWo: f32 = (
		bsdfFresnelDielectric(max(ωoLocal.z, 0.0f), mat.specularIor)
	);
	let roughnessDielectric: f32 = (
		// lerp(1, r, \xi_s F_s)
		lerp(
			1.0f,
			mat.specularRoughness,
			clamp(mat.specularWeight * baseFresnelWo, 0.0f, 1.0f)
		)
	);
	// r_b (eq 68)
	let roughnessBase: f32 = (
		// lerp(r_d, r_m, M)
		lerp(roughnessDielectric, mat.specularRoughness, mat.baseMetalness)
	);
	// capped below one so the delta denominator stays positive at grazing
	let kCoat: f32 = min(lerp(kSmooth, kRough, roughnessBase), 0.9999f);
	// E_b (eq 96): normal-incidence slab albedos blended by their mix weights
	let baseAlbedo: vkof::Spectrum = (
		mat.baseWeight * mat.baseColor
	);
	let albedoOpaqueBase: vkof::Spectrum = (
		lerp(
			baseAlbedo,
			mat.subsurfaceColor,
			mat.subsurfaceWeight
		)
	);
	let albedoDielectricBase: vkof::Spectrum = (
		lerp(
			albedoOpaqueBase,
			mat.transmissionColor,
			mat.transmissionWeight
		)
	);
	let albedoBase: vkof::Spectrum = (
		clamp(
			lerp(
				albedoDielectricBase,
				baseAlbedo,
				mat.baseMetalness
			),
			0.0f,
			1.0f
		)
	);
	// (5) layering with the modulated darkening
	let darkeningDelta: vkof::Spectrum = (
		// (1 - K) / (1 - E_b K)
		(1.0f - kCoat) / (1.0f - albedoBase * kCoat)
	);
	let darkening: vkof::Spectrum = (
		// lerp(1, \Delta, C \delta)
		lerp(
			vkof::Spectrum.one(), darkeningDelta, mat.coatWeight * mat.coatDarkening
		)
	);
	return (
		vkof::Spectrum.one() * fCoat
		+ darkening * lerp(baseSubstrate, transmittedBase, mat.coatWeight)
	);
}

} // namespace vkof
