#ifndef UTIL_MATERIAL_OPENPBR_COAT_GLSL
#define UTIL_MATERIAL_OPENPBR_COAT_GLSL

// clear coat, rough dielectric layer over the base substrate
// TODO need to verify below
/*
	\begin{align*}
	&\textbf{(1) coat specular} \\
	% same microfacet dielectric as the base specular lobe, with the coat
	% ior / roughness / anisotropy and turquin compensation
	&f_{coat} = F \, D \, V \, k_{ms}
		\tag{openpbr spec, coat}\\
	&\textbf{(2) transmission through the coat} \\
	% the interface albedo E(\mu) is removed once on the way in and once on
	% the way out; coat\_color is the observed two-pass tint (mor-shared
	% defines the field as the square of the one-pass transmittance), so it
	% is applied once, not squared again
	&T = (1 - E(\mu_i)) (1 - E(\mu_o)) \, C_{coat}
		\tag{openpbr spec, coat, eq ??}\\
	&\textbf{(3) view-dependent absorption} \\
	% grazing-angle path length through an absorbing coat is longer than
	% the normal-incidence pass baked into coat\_color; \mu_i^t / \mu_o^t
	% are the cosines of the ray refracted into the coat at entry/exit
	% (eq. 72), clamped so a coat-side TIR condition (\eta_c < 1, seen
	% from inside the medium) doesn't blow up the exponent. eq. 72 reduces
	% to T_{coat}^2 at normal incidence, which mor-shared defines to equal
	% coat\_color, so the coat\_color exponent here carries an extra 1/2
	% relative to the spec's T_{coat} form of eq. 71
	&\mu_i^t = \sqrt{1 - (1 - \mu_i^2) / \eta_c^2}
		\tag{openpbr spec, coat, eq. 72}\\
	&T_\mathrm{coat}^{1/\mu_i^t + 1/\mu_o^t}
		\tag{openpbr spec, coat, eq. 71}\\
	&\textbf{(4) coat darkening} \\
	% \delta = coat_darkening: physically-correct darkening at \delta=1;
	% at \delta=0 the base albedo is instead boosted to counteract the
	% darkening entirely, so the coated base albedo matches the
	% uncoated base color (eq. 61, eq. 62); we take the practical route
	% of eq. 70 rather than computing an explicit boost factor B_0
	&E_c = F_0 + E_b(1 - F_0) = \mathrm{lerp}(F_0, 1, E_b)
		\tag{openpbr spec, coat darkening, eq. 61}\\
	&B(\delta) = \mathrm{lerp}(B_0, 1, \delta)
		\tag{openpbr spec, coat darkening, eq. 62}\\
	% interfaced-lambertian derivation (context only, not evaluated
	% directly -- eq. 70 below is the applied form)
	&E_c(\omega_o) = F(\omega_o, \eta_c)
		+ E_b \bigl(1 - F(\omega_o, \eta_c)\bigr) \Delta(E_b, \eta_c)
		\tag{openpbr spec, coat darkening, eq. 63}\\
	&\Delta(E_b, \eta_c) = \frac{1 - K}{1 - E_b K}
		\tag{openpbr spec, coat darkening, eq. 64}\\
	% K_r: lambertian-base limit; E_F(\eta_c) is the hemispherical coat
	% fresnel albedo, approximated with the same kulla-conty closed form
	% used elsewhere in this codebase for F_{avg}
	&K_r = 1 - \bigl(1 - E_F(\eta_c)\bigr) / \eta_c^2
		\tag{openpbr spec, coat darkening, eq. 65}\\
	% K_s: smooth-base limit, exact for a perfectly smooth metal/dielectric
	&K_s = F(\omega_o, \eta_c)
		\tag{openpbr spec, coat darkening, eq. 66}\\
	&K = \mathrm{lerp}(K_s, K_r, r_b)
		\tag{openpbr spec, coat darkening, eq. 67}\\
	% r_b: effective base roughness, dielectric/metal blended by
	% base_metalness M; r_m = specular_roughness directly
	&r_b = \mathrm{lerp}(r_d, r_m, M)
		\tag{openpbr spec, coat darkening, eq. 68}\\
	% r_d: an assumed fully rough underlying base (1) pulled toward the
	% microfacet specular_roughness r by the unmodulated base dielectric
	% fresnel F_s scaled by specular_weight \xi_s
	&r_d = \mathrm{lerp}(1, r, \xi_s F_s)
		\tag{openpbr spec, coat darkening, eq. 69}\\
	% modulated darkening factor: applied uniformly to the whole base
	% term (both the coated and uncoated fraction), taking into account
	% coat_weight C and coat_darkening \delta; we apply this factor
	% directly rather than computing B(\delta) explicitly, since it is
	% algebraically equal to (1-C) + C B(\delta) \Delta
	&\mathrm{lerp}(1, \Delta, C \delta)
		\tag{openpbr spec, coat darkening, eq. 70}\\
	&\textbf{(5) layering} \\
	&f = C f_{coat}
		+ \mathrm{lerp}(1, \Delta, C \delta)
		\, \text{mix}(f_{base}, T f_{base}, C)
		\tag{openpbr spec, figure 96 + eq. 70}\\
	\end{align*}
*/
vec3 openPbrCoatEvaluateF(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	const vec3 baseSubstrate,
	const bool isInsideMedium
) {
	// coat specular reflection
	const float iorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	const float f0 = iorRatio * iorRatio;
	// coatWeight is the true "is there a coat" signal; f0 == 0 alone isn't
	// enough (e.g. a caller that never set coatIor leaves it at 0, which
	// makes coatEtaRel collapse to 0 below and NaN out the whole layer even
	// though coatWeight correctly says there's no coat)
	if (f0 == 0.0f || mat.coatWeight <= 0.0f) {
		// no coat layer
		return baseSubstrate;
	}
	// per-axis slopes from the coat roughness + anisotropy
	const vec2 alpha = (
		openPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
	);
	// coat has its own shading normal (khr_materials_clearcoat's
	// clearcoatNormalTexture), sampled into the same mesh tangent basis as
	// the base normal, so wi/wo project into coatFrame here, not frame
	const vec3 wiLocal = (
		vec3(
			dot(wi, coatFrame.tanX), dot(wi, coatFrame.tanY), dot(wi, coatFrame.nor)
		)
	);
	const vec3 woLocal = (
		vec3(
			dot(wo, coatFrame.tanX), dot(wo, coatFrame.tanY), dot(wo, coatFrame.nor)
		)
	);
	const vec3 hLocal = normalize(wiLocal + woLocal);
	// relative ior: inverted when the interface is seen from inside the medium
	const float coatEtaRel = isInsideMedium ? (1.0f / mat.coatIor) : mat.coatIor;
	const float coatFresnel = (
		utilMicrofacetFresnelDielectric(dot(hLocal, wiLocal), coatEtaRel)
	);
	const float coatDistribution = (
		utilMicrofacetGgxDistributionAniso(hLocal, alpha)
	);
	const float coatVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// turquin 2019 compensation, consistent with the base dielectric lobe;
	// the scalar fit uses the effective roughness \sqrt{\alpha_x \alpha_y}
	const float coatEnergyCompensation = (
		utilMicrofacetDielectricEnergyCompensate(
			wiLocal.z, sqrt(alpha.x * alpha.y), f0
		)
	);
	// (1) f_{coat}, coat weight folded into the layering here
	const float fCoat = (
		mat.coatWeight
		* coatFresnel * coatDistribution * coatVisibility
		* coatEnergyCompensation
	);
	// (2) T: the compensated interface albedo leaves on the way in at
	// \mu_i and again on the way out at \mu_o; coat roughness and coat f0,
	// not the base specular's
	const float coatAlbedoWi = (
		utilMicrofacetDielectricAlbedo(
			wiLocal.z, mat.coatRoughness, f0
		)
	);
	const float coatAlbedoWo = (
		utilMicrofacetDielectricAlbedo(
			max(woLocal.z, 0.0f), mat.coatRoughness, f0
		)
	);
	// view-dependent absorption (eq. 72): refracted cosines into the coat
	// at entry/exit, clamped so a coat-side TIR condition (\eta_c < 1,
	// seen from inside the medium) doesn't blow up the eq. 71 exponent
	const float muItSq = (
		max(
			1.0f - (1.0f - wiLocal.z * wiLocal.z) / (coatEtaRel * coatEtaRel),
			1e-4f
		)
	);
	const float woZClamped = max(woLocal.z, 0.0f);
	const float muOtSq = (
		max(
			1.0f - (1.0f - woZClamped * woZClamped) / (coatEtaRel * coatEtaRel),
			1e-4f
		)
	);
	// eq. 71 as T_{coat}^{1/\mu_i^t + 1/\mu_o^t}; coat_color = T_{coat}^2
	// (mor-shared), so the exponent on coat_color carries an extra 1/2
	const float absorptionExponent = (
		0.5f * (1.0f / sqrt(muItSq) + 1.0f / sqrt(muOtSq))
	);
	const vec3 coatAbsorption = (
		pow(clamp(mat.coatColor, 1e-6f, 1.0f), vec3(absorptionExponent))
	);
	const vec3 transmittedBase = (
		(1.0f - coatAlbedoWi)
		* (1.0f - coatAlbedoWo)
		* coatAbsorption
		* baseSubstrate
	);
	// (4) coat darkening; fresnel terms use the same relative ior as the
	// coat lobe so the inside-medium case stays consistent
	const float kSmooth = (
		utilMicrofacetFresnelDielectric(max(woLocal.z, 0.0f), coatEtaRel)
	);
	// hemispherical average coat fresnel, kulla & conty closed form;
	// K_r clamps at zero when seen from inside the medium (eta < 1)
	const float coatFresnelAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	const float kRough = (
		clamp(
			1.0f - (1.0f - coatFresnelAvg) / (coatEtaRel * coatEtaRel),
			0.0f,
			1.0f
		)
	);
	// r_d (eq. 69): F_s is the unmodulated base dielectric fresnel (raw
	// specular_ior, not the specular_weight-modulated effective ior), per
	// the explicit \xi_s F_s product in the spec formula
	const float baseFresnelWo = (
		utilMicrofacetFresnelDielectric(max(woLocal.z, 0.0f), mat.specularIor)
	);
	const float roughnessDielectric = (
		mix(
			1.0f,
			mat.specularRoughness,
			clamp(mat.specularWeight * baseFresnelWo, 0.0f, 1.0f)
		)
	);
	// r_b (eq. 68): r_m = specular_roughness directly
	const float roughnessBase = (
		mix(roughnessDielectric, mat.specularRoughness, mat.baseMetalness)
	);
	// capped below one so the delta denominator stays positive at grazing
	const float kCoat = min(mix(kSmooth, kRough, roughnessBase), 0.9999f);
	// E_b: normal-incidence slab albedos blended by their mix weights,
	// mirroring the figure 96 layering
	const vec3 albedoOpaqueBase = (
		mix(
			mat.baseWeight * mat.baseColor,
			mat.subsurfaceColor,
			mat.subsurfaceWeight
		)
	);
	const vec3 albedoDielectricBase = (
		mix(albedoOpaqueBase, mat.transmissionColor, mat.transmissionWeight)
	);
	const vec3 albedoBase = (
		clamp(
			mix(
				albedoDielectricBase,
				mat.baseWeight * mat.baseColor,
				mat.baseMetalness
			),
			0.0f,
			1.0f
		)
	);
	const vec3 darkeningDelta = (
		(1.0f - kCoat) / (1.0f - albedoBase * kCoat)
	);
	const vec3 darkening = (
		mix(vec3(1.0f), darkeningDelta, mat.coatWeight * mat.coatDarkening)
	);
	// (5) layering with the modulated darkening on the base term
	return (
		fCoat + darkening * mix(baseSubstrate, transmittedBase, mat.coatWeight)
	);
}

#endif // UTIL_MATERIAL_OPENPBR_COAT_GLSL
