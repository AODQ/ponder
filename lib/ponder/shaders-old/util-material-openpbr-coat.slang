module "util-material-openpbr-coat";
__exported import "util-types";
__exported import "util-material-openpbr-microfacet";


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
public func openPbrCoatEvaluateF(
	tables: MaterialTableHandles,
	mat: OpenPbrMaterial,
	coatFrame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
	baseSubstrate: f32v3,
	isInsideMedium: bool,
) -> f32v3 {
	// coat specular reflection
	const f32 iorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	const f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f || mat.coatWeight <= 0.0f) {
		// no coat layer
		return baseSubstrate;
	}
	// per-axis slopes from the coat roughness + anisotropy
	const f32v2 alpha = (
		openPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
	);
	// coat has its own shading normal (khr_materials_clearcoat's
	// clearcoatNormalTexture), sampled into the same mesh tangent basis as
	// the base normal, so wi/wo project into coatFrame
	const f32v3 wiLocal = (
		f32v3(
			dot(wi, coatFrame.tanX), dot(wi, coatFrame.tanY), dot(wi, coatFrame.nor)
		)
	);
	const f32v3 woLocal = (
		f32v3(
			dot(wo, coatFrame.tanX), dot(wo, coatFrame.tanY), dot(wo, coatFrame.nor)
		)
	);
	f32v3 hLocal;
	const bool hValid = utilHalfVectorLocal(wiLocal, woLocal, hLocal);
	const f32 dotHWi = dot(hLocal, wiLocal);
	// a non-positive dotHWi is a back-facing microfacet, which has no reflection
	const bool hFacing = hValid && dotHWi > 0.0f;
	// relative ior; inverted when the interface is seen from inside the medium
	const f32 coatEtaRel = isInsideMedium ? (1.0f / mat.coatIor) : mat.coatIor;
	const f32 coatFresnel = (
		hFacing ? utilMicrofacetFresnelDielectric(dotHWi, coatEtaRel) : 0.0f
	);
	const f32 coatDistribution = (
		utilMicrofacetGgxDistributionAniso(hLocal, alpha)
	);
	const f32 coatVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// turquin 2019 compensation, consistent with the base dielectric lobe;
	// the scalar fit uses the effective roughness \sqrt{\alpha_x \alpha_y}
	const f32 coatEnergyCompensation = (
		utilMicrofacetDielectricEnergyCompensate(
			wiLocal.z, sqrt(alpha.x * alpha.y), f0
		)
	);
	// (1) f_{coat}, coat weight folded into the layering here
	const f32 fCoat = (
		mat.coatWeight
		// F D V
		* coatFresnel * coatDistribution * coatVisibility
		// k_{ms}
		* coatEnergyCompensation
	);
	// (TODO REVIEW)
	// (2) T: openpbr's albedo-scaling blocks the base with a *single*
	// directional albedo at the camera-side direction (spec's \omega_o,
	// this file's wi), which is what makes it factor out of the light
	// integral and keeps the white furnace test exact. squaring it here
	// with a second \omega_i term double-charged the base
	const f32 coatAlbedoWi = (
		// E(\mu_o) in spec notation
		utilMicrofacetDielectricAlbedo(
			wiLocal.z, mat.coatRoughness, f0
		)
	);
	// (TODO REVIEW)
	// (3) view-dependent absorption: refracted cosines into the coat
	// at entry/exit, clamped for coat-side TIR
	const f32 muItSq = (
		max(
			// 1 - (1 - \mu_i^2) / \eta_c^2
			1.0f - (1.0f - wiLocal.z * wiLocal.z) / (coatEtaRel * coatEtaRel),
			1e-4f
		)
	);
	const f32 woZClamped = max(woLocal.z, 0.0f);
	const f32 muOtSq = (
		max(
			// 1 - (1 - \mu_o^2) / \eta_c^2
			1.0f - (1.0f - woZClamped * woZClamped) / (coatEtaRel * coatEtaRel),
			1e-4f
		)
	);
	// eq 71
	const f32 absorptionExponent = (
		// 1 / \mu_i^t + 1 / \mu_o^t
		0.5f * (1.0f / sqrt(muItSq) + 1.0f / sqrt(muOtSq))
	);
	const f32v3 coatAbsorption = (
		pow(clamp(mat.coatColor, 1e-6f, 1.0f), f32v3(absorptionExponent))
	);
	const f32v3 transmittedBase = (
		// (TODO REVIEW)
		// 1 - E_{coat}(\omega_o)
		(1.0f - coatAlbedoWi)
		// (TODO REVIEW)
		* coatAbsorption
		* baseSubstrate
	);
	// (4) coat darkening; fresnel terms use the ior as coat lobe
	const f32 kSmooth = (
		utilMicrofacetFresnelDielectric(max(woLocal.z, 0.0f), coatEtaRel)
	);
	// hemispherical average coat fresnel, kulla & conty closed form;
	// K_r clamps at zero when seen from inside the medium (eta < 1)
	const f32 coatFresnelAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	const f32 kRough = (
		clamp(
			1.0f - (1.0f - coatFresnelAvg) / (coatEtaRel * coatEtaRel),
			0.0f,
			1.0f
		)
	);
	// r_d (eq 69): F_s is the unmodulated base dielectric fresnel
	const f32 baseFresnelWo = (
		utilMicrofacetFresnelDielectric(max(woLocal.z, 0.0f), mat.specularIor)
	);
	const f32 roughnessDielectric = (
		// lerp(1, r, \xi_s F_s)
		lerp(
			1.0f,
			mat.specularRoughness,
			clamp(mat.specularWeight * baseFresnelWo, 0.0f, 1.0f)
		)
	);
	// r_b (eq 68)
	const f32 roughnessBase = (
		// lerp(r_d, r_m, M)
		lerp(roughnessDielectric, mat.specularRoughness, mat.baseMetalness)
	);
	// capped below one so the delta denominator stays positive at grazing
	const f32 kCoat = min(lerp(kSmooth, kRough, roughnessBase), 0.9999f);
	// E_b (eq 96): normal-incidence slab albedos blended by their mix weights
	const f32v3 albedoOpaqueBase = (
		lerp(
			mat.baseWeight * mat.baseColor,
			mat.subsurfaceColor,
			mat.subsurfaceWeight
		)
	);
	const f32v3 albedoDielectricBase = (
		lerp(albedoOpaqueBase, mat.transmissionColor, mat.transmissionWeight)
	);
	const f32v3 albedoBase = (
		clamp(
			lerp(
				albedoDielectricBase,
				mat.baseWeight * mat.baseColor,
				mat.baseMetalness
			),
			0.0f,
			1.0f
		)
	);
	// (5) layering with the modulated darkening
	const f32v3 darkeningDelta = (
		// (1 - K) / (1 - E_b K)
		(1.0f - kCoat) / (1.0f - albedoBase * kCoat)
	);
	const f32v3 darkening = (
		// lerp(1, \Delta, C \delta)
		lerp(f32v3(1.0f), darkeningDelta, mat.coatWeight * mat.coatDarkening)
	);
	return (
		fCoat + darkening * lerp(baseSubstrate, transmittedBase, mat.coatWeight)
	);
}

