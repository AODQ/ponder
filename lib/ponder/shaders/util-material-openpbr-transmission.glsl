#ifndef UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL
#define UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL

#include "util-material-openpbr-microfacet.glsl"

// classical two-term cauchy dispersion equation (cauchy 1836),
// reparametrized via the abbe number through scale and abbe number as
// described by the openpbr spec and gltf KHR_materials_dispersion.
/*
	\begin{align*}
	&\textbf{(1) cauchy's equation} \\
	&n(\lambda) = A + B / \lambda^2
		\tag{cauchy 1836, openpbr eq 55}\\
	&\textbf{(2) coefficients from ior + abbe number} \\
	&B = \frac{n_d - 1}{V_d (\lambda_F^{-2} - \lambda_C^{-2})}
		\tag{openpbr spec, eq 56}\\
	&A = n_d - B / \lambda_d^2
		\tag{openpbr spec, eq 56}\\
	&\textbf{(3) artist-friendly abbe number} \\
	&V_d =
		\mathrm{transmission\_dispersion\_abbe\_number}
		/ \mathrm{transmission\_dispersion\_scale}
		\tag{openpbr spec, transmission dispersion}\\
	\end{align*}
*/
// -----------------------------------------------------------------------------
// -- openPbrTransmissionIor
// -----------------------------------------------------------------------------

// (TODO REVIEW)
// the ior the base dielectric refracts and fresnels at. openpbr requires the
// reflection and transmission sides of the same interface to share \eta'_s,
// so specular_weight's ior modulation belongs here too, not just on the
// reflection lobe (openpbr spec, specular: "the fresnel transmission factor
// and refraction into and out of the base dielectric should also be
// consistent with the IOR ratio")
float openPbrTransmissionIor(const OpenPbrMaterial mat) {
	return openPbrSpecularEffectiveIor(mat);
}
// (TODO REVIEW)

// pick one of the three fraunhofer lines discretized/stochastically per-path
vec3 openPbrDispersionIorRgb(const OpenPbrMaterial mat) {
	// (TODO REVIEW)
	// n_d is the modulated ior, per the spec's "specular_ior (including any
	// modulation via specular_weight) defines n(\lambda_d)"
	const float etaD = openPbrTransmissionIor(mat);
	// (TODO REVIEW)
	if (mat.transmissionDispersionScale <= 0.0f) {
		return vec3(etaD);
	}
	// \lambda_F, \lambda_d, \lambda_C, in micrometers
	const float lambdaF = 0.4861f;
	const float lambdaD = 0.5876f;
	const float lambdaC = 0.6563f;

	// V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale;
	const float vd = (
		mat.transmissionDispersionAbbeNumber
		/ mat.transmissionDispersionScale
	);
	const float vdSafe = vd != 0.0f ? vd : 1e-4f;
	// B = (n_d - 1) / (V_d (\lambda_F^{-2} - \lambda_C^{-2}))
	const float b = (
		(etaD - 1.0f)
		/ (vdSafe * (1.0f / (lambdaF * lambdaF) - 1.0f / (lambdaC * lambdaC)))
	);
	// A = n_d - B / \lambda_d^2
	const float a = etaD - b / (lambdaD * lambdaD);

	// n(\lambda) = A + B / \lambda^2, at the C/F lines;
	// n(\lambda_d) = etaD
	return vec3(
		a + b / (lambdaC * lambdaC),
		etaD,
		a + b / (lambdaF * lambdaF)
	);
}

// -----------------------------------------------------------------------------
// -- openPbrTransmissionEvaluateF
// -----------------------------------------------------------------------------

vec3 openPbrTransmissionEvaluateF(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	const float dispersedIor
) {
	/*
		\begin{align*}
		&\textbf{microfacet BTDF (Walter et al. 2007):}\\
		&h_t =
			-(\eta_i \omega_i + \eta_t \omega_o)
			/ |\eta_i \omega_i + \eta_t \omega_o|
		\\
		&f_t(\omega_i, \omega_o) =
			\frac {
				(1 - F(\omega_i \cdot h_t, \eta_t/\eta_i))
				\cdot D(h_t) \cdot G_2(\omega_i, \omega_o)
				\cdot |\omega_i \cdot h_t| \cdot |\omega_o \cdot h_t|
				\cdot \eta_t^2
			} {
				|\omega_i \cdot n| \cdot |\omega_o \cdot n|
				\cdot (
					\eta_i |\omega_i \cdot h_t| + \eta_t |\omega_o \cdot h_t|
				)^2
			}
		\\
		&V = \frac{G_2}{(4 |\omega_i \cdot n| |\omega_o \cdot n|)} \\
		&\text{simplifies to:}\\
		&f_t =
			(1 - F) \cdot D \cdot V \cdot
			\frac{
				4 |\omega_i \cdot h_t|
				\cdot |\omega_o \cdot h_t|
				\cdot \eta_t^2
			} {
				(\eta_i |\omega_i \cdot h_t| + \eta_t |\omega_o \cdot h_t|)^2
			}
		\\
		\end{align*}
	*/
	const vec3 nor = frame.nor;
	// entering: etaI=1 (air), etaT=dispersedIor
	// exiting:  etaI=dispersedIor, etaT=1 (air)
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	// khr_materials_transmission without khr_materials_volume:
	// a thin shell with no second surface to refract back out of,
	// so have to handle delta lobe as special case.
	// etaI == etaT behaves the same
	if (mat.geometryThinWalled > 0.0f || etaI == etaT) {
		const float dotNorWi = max(dot(nor, wi), 1e-5f);
		const float dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
		const float F = utilMicrofacetFresnelDielectric(dotNorWi, etaT / etaI);
		const vec3 tint = (
			(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : vec3(1.0f)
		);
		return tint * (1.0f - F) / dotNorWo;
	}
	// h_t = -normalize(\eta_i \omega_i + \eta_t \omega_o)
	const vec3 htUnnorm = -(etaI * wi + etaT * wo);
	const vec3 ht = normalize(htUnnorm);
	const vec3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(wi, htOriented) * dot(wo, htOriented) > 0.0f) {
		return vec3(0.0f);
	}

	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const float dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
	// |\omega_i \cdot h_t|, |\omega_o \cdot h_t|
	const float dotHtWi = max(dot(htOriented, wi), 0.0f);
	const float dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	const float dotNorHt = max(dot(nor, htOriented), 0.0f);

	// refraction shares the dielectric interface with the specular reflection
	// lobe, so the same slopes and (gltf-rotated) tangent frame drive both
	const vec2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const vec3 htLocal = (
		vec3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	const vec3 wiLocal = (
		vec3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	// A_\omega is even in z, so the transmitted \omega_o enters with the
	// already-abs'd dotNorWo exactly as the isotropic form takes abs()
	const vec3 woLocal = (
		vec3(dot(wo, specFrame.tanX), dot(wo, specFrame.tanY), dotNorWo)
	);
	// F(\omega_i \cdot h_t, \eta_t / \eta_i);
	const float F = utilMicrofacetFresnelDielectric(dotHtWi, etaT / etaI);
	// D(h_t)
	const float D = utilMicrofacetGgxDistributionAniso(htLocal, alpha);
	// V = G_2 / (4 |\omega_i \cdot n| |\omega_o \cdot n|)
	const float V = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// |\eta_i \omega_i + \eta_t \omega_o|
	const float denom = length(htUnnorm);
	const float btdf = (
		(1.0f - F) * D * V
		* 4.0f * dotHtWi * dotHtWo * etaT * etaT
		/ max(denom * denom, 1e-6f)
	);
	// transmissionDepth = 0, constant fresnel tint
	const vec3 tint = (
		(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : vec3(1.0f)
	);
	return tint * btdf;
}

// -----------------------------------------------------------------------------
// -- openPbrTransmissionPdf
// -----------------------------------------------------------------------------

// pdf of VNDF-sampled refraction direction (walter et al. 2007)
float openPbrTransmissionPdf(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// must match the dispersedIor openPbrTransmissionEvaluateF was called
	const float dispersedIor
) {
	/*
		pdf(\omega_o) =
			G_1(\omega_i) \cdot D(h_t)
			\cdot \frac{|\omega_i \cdot h_t|}{|\omega_i \cdot n|}
			\cdot
			\frac{
				\eta_t^2 \cdot |\omega_o \cdot h_t|
			} {
				(\eta_i |\omega_i \cdot h_t| + \eta_t |\omega_o \cdot h_t|)^2
			}
	*/
	const vec3 nor = frame.nor;
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	// thin shell is delta dirac distribution
	if (mat.geometryThinWalled > 0.0f || etaI == etaT) {
		return 1.0f;
	}
	const vec3 htUnnorm = -(etaI * wi + etaT * wo);
	const vec3 ht = normalize(htUnnorm);
	const vec3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(wi, htOriented) * dot(wo, htOriented) > 0.0f) {
		return 0.0f;
	}
	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const float dotNorHt = max(dot(nor, htOriented), 0.0f);
	const float dotHtWi = max(dot(htOriented, wi), 0.0f);
	const float dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	// slopes and frame must match the vndf sampling in openPbrSampleWo;
	// openPbrRoughnessAlpha floors both axes at 1e-5
	const vec2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const vec3 wiLocal = (
		vec3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	const vec3 htLocal = (
		vec3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	const float G1 = utilMicrofacetSmithG1Aniso(wiLocal, alpha);
	const float D = utilMicrofacetGgxDistributionAniso(htLocal, alpha);
	// see openPbrTransmissionEvaluateF's denom comment: length of the
	// unnormalized half-vector sum, not a sum of clamped cosines
	const float denom = length(htUnnorm);
	return (
		G1 * D * dotHtWi / dotNorWi
		* etaT * etaT * dotHtWo
		/ max(denom * denom, 1e-6f)
	);
}

// -----------------------------------------------------------------------------
// -- openPbrTransmissionTirReflectPdf
// -----------------------------------------------------------------------------

// pdf of the transmission lobe's TIR fallback
float openPbrTransmissionTirReflectPdf(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// must match the dispersedIor openPbrSampleWo
	const float dispersedIor
) {
	const vec3 nor = frame.nor;
	vec3 h;
	if (!utilHalfVectorLocal(wi, wo, h)) {
		return 0.0f;
	}
	const float dotNorH = dot(nor, h);
	const float dotWiH = dot(wi, h);
	if (dotNorH <= 0.0f || dotWiH <= 0.0f) {
		return 0.0f;
	}
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	const vec3 refracted = refract(-wi, h, etaI / etaT);
	if (length(refracted) > 0.5f) {
		return 0.0f;
	}
	// same slopes and frame as the refraction branch it falls back from
	const vec2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const vec3 hLocal = (
		vec3(dot(h, specFrame.tanX), dot(h, specFrame.tanY), dotNorH)
	);
	const vec3 wiLocal = (
		vec3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	const float D = utilMicrofacetGgxDistributionAniso(hLocal, alpha);
	const float G1 = utilMicrofacetSmithG1Aniso(wiLocal, alpha);
	return (D * G1 * dotWiH) / (dotNorWi * max(4.0f * dotWiH, 1e-6f));
}

#endif // UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL
