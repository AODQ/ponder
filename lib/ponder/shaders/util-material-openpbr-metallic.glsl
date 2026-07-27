#ifndef UTIL_MATERIAL_OPENPBR_METALLIC_GLSL
#define UTIL_MATERIAL_OPENPBR_METALLIC_GLSL

#include "util-material-openpbr-microfacet.glsl"

// openpbr conductor: f82-tint fresnel over the shared specular ggx lobe;
// f82-tint model from kutz, hasan & edmondson 2021
/*
	\begin{align*}
	&\textbf{(1) f82-tint conductor fresnel} \\
	&f_0 = base\_weight \cdot base\_color
		\tag{openpbr spec, metal}
	\\
	% schlick fresnel toward white at grazing
	&F_s(\mu) = f_0 + (1 - f_0)(1 - \mu)^5
		\tag{schlick 1994}
	\\
	% the edge tint is authored at \bar{\mu} = 1/7 (\theta \approx 82 deg)
	&\bar{\mu} = \tfrac{1}{7}\\
	&F_c = F_s(\bar{\mu}) (1 - S_w S_c)\\
	&F_d = \bar{\mu} (1 - \bar{\mu})^6\\
	% the correction term is zero at \mu = 0 and \mu = 1 and peaks at
	% \bar{\mu}, pulling the fresnel toward the authored target
	&F_{82}(\mu) = F_s(\mu) - \frac{\mu (1 - \mu)^6}{F_d} F_c\\
	\end{align*}
*/
vec3 openPbrFresnelMetallicEvaluateF(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo
) {
	if (mat.baseMetalness <= 0.0f) {
		return vec3(0.0f);
	}

	const vec3 nor = frame.nor;
	const float dotNorWi = max(dot(nor, wi), 1e-5f);

	// f_0 = base\_weight \cdot base\_color
	const vec3 f0 = mat.baseWeight * mat.baseColor;

	// F_s(\bar{\mu}); (1 - \frac{1}{7})^5 = 0.462664
	const vec3 fresnelConstS = f0 + (vec3(1.0f) - f0) * 0.462664f;
	// F_c = F_s(\bar{\mu}) (1 - S_w S_c)
	const vec3 fresnelConst = (
		fresnelConstS
		* (vec3(1.0f) - mat.specularWeight * mat.specularColor)
	);
	// F_d = \frac{1}{7} (1 - \frac{1}{7})^6 = 0.056653
	const float fresnelDenom = 0.056653f;

	// F_s(\mu)
	const vec3 fs = (
		f0 + (vec3(1.0f) - f0) * pow(1.0f - dotNorWi, 5.0f)
	);
	// F_{82}(\mu) = F_s(\mu) - \frac{\mu (1 - \mu)^6}{F_d} F_c
	const vec3 f82 = (
		max(
			fs
			- (
				(dotNorWi * pow(1.0f - dotNorWi, 6.0f) / fresnelDenom)
				* fresnelConst
			),
			vec3(0.0f)
		)
	);

	// conductor shares the base specular roughness + anisotropy, widened
	// by a rough coat (openpbr spec, coat roughening)
	const float roughenedRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	const vec2 alpha = (
		openPbrRoughnessAlpha(
			roughenedRoughness, mat.specularRoughnessAnisotropy
		)
	);
	// gltf interop anisotropy rotation; openpbr does not have this I believe
	// but hacking it in for gltf interop :/
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const vec3 wiLocal = (
		vec3(
			dot(wi, specFrame.tanX),
			dot(wi, specFrame.tanY),
			dot(wi, specFrame.nor)
		)
	);
	const vec3 woLocal = (
		vec3(
			dot(wo, specFrame.tanX),
			dot(wo, specFrame.tanY),
			dot(wo, specFrame.nor)
		)
	);
	vec3 hLocal;
	const bool hValid = utilHalfVectorLocal(wiLocal, woLocal, hLocal);
	const float D = (
		hValid ? utilMicrofacetGgxDistributionAniso(hLocal, alpha) : 0.0f
	);
	const float V = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);

	// thin-film interference over the metal f0 as per-channel ior
	// per-channel real-valued base ior; the f82-tint model has no real
	// n/k data so \kappa remains zero
	const float dotHWi = dot(hLocal, wiLocal);
	const vec3 etaBase = openPbrIorFromF0(f0);
	const vec3 mfFresnel = (
		mat.thinFilmWeight > 0.0f
		? mix(
			f82,
			openPbrThinfilmFresnel(mat, dotHWi, etaBase, vec3(0.0f)),
			mat.thinFilmWeight
		)
		: f82
	);

	// turquin 2019 conductor energy compensation, per channel with the
	// colored f_0; isotropic using effective roughness \sqrt{\alpha_x \alpha_y}
	const vec3 fAvg = f0 + (vec3(1.0f) - f0) * (1.0f / 21.0f);
	const float ess = (
		max(
			utilMicrofacetGgxDirectionalAlbedo(
				dotNorWi, sqrt(alpha.x * alpha.y), 1.0f, 1.0f
			),
			1e-4f
		)
	);
	const vec3 energyCompensation = (
		vec3(1.0f) + fAvg * (1.0f - ess) / ess
	);

	return mfFresnel * D * V * energyCompensation;
}

#endif // UTIL_MATERIAL_OPENPBR_METALLIC_GLSL
