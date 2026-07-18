#ifndef UTIL_MATERIAL_OPENPBR_METALLIC_GLSL
#define UTIL_MATERIAL_OPENPBR_METALLIC_GLSL

// openpbr conductor: f82-tint fresnel over the shared specular ggx lobe;
// f82-tint model from kutz, hasan & edmondson 2021 "novel aspects of the
// adobe standard material", per the openpbr spec "Metal" section
// TODO fill the eq/fig placeholders from the paper / spec copies
/*
	\begin{align*}
	&\textbf{(1) f82-tint conductor fresnel} \\
	&f_0 = base\_weight \cdot base\_color
		\tag{openpbr spec, metal}\\
	% schlick fresnel toward white at grazing
	&F_s(\mu) = f_0 + (1 - f_0)(1 - \mu)^5
		\tag{schlick 1994}\\
	% the edge tint is authored at \bar{\mu} = 1/7 (\theta \approx 82 deg);
	% the target there is S_w S_c F_s(\bar{\mu}), so specular weight and
	% color shape the edge dip, they do not scale the whole lobe
	&\bar{\mu} = \tfrac{1}{7}
		\tag{kutz-hasan-edmondson 2021, eq ??}\\
	&F_c = F_s(\bar{\mu}) (1 - S_w S_c)
		\tag{kutz-hasan-edmondson 2021, eq ??}\\
	&F_d = \bar{\mu} (1 - \bar{\mu})^6
		\tag{kutz-hasan-edmondson 2021, eq ??}\\
	% the correction term is zero at \mu = 0 and \mu = 1 and peaks at
	% \bar{\mu}, pulling the fresnel toward the authored edge target
	&F_{82}(\mu) = F_s(\mu) - \frac{\mu (1 - \mu)^6}{F_d} F_c
		\tag{kutz-hasan-edmondson 2021, eq ??}\\
	\end{align*}
*/
vec3 openPbrFresnelMetallicEvaluateF(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo
) {
	// caller mixes this in by mat.baseMetalness; skip the fresnel/ggx/
	// energy-compensation work entirely when it's going to be discarded
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
	// clamped at zero: the raw f82 correction dips negative for dark f0
	// combined with low specularWeight*specularColor (the model assumes
	// metal-like f0). deliberate deviation from cull (which returns the
	// raw value), user decision 2026-07-14; keep in sync with
	// utilFresnelMetallic in util-fresnel.glsl
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
	// gltf interop anisotropy rotation (not part of openpbr)
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
	const vec3 hLocal = normalize(wiLocal + woLocal);
	const float D = utilMicrofacetGgxDistributionAniso(hLocal, alpha);
	const float V = utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha);

	// thin-film interference over the metal's own f0 as an effective,
	// per-channel real-valued base ior; the f82-tint model has no real
	// n/k data to give thin-film a true complex conductor base, so kappa
	// is zero here (see openPbrIorFromF0)
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
	// colored f_0; the isotropic albedo fit is indexed with the effective
	// roughness \sqrt{\alpha_x \alpha_y}
	/*
		\begin{align*}
		&F_{avg} = f_0 + \frac{1 - f_0}{21}
			\tag{kulla \& conty 2017}\\
		&f = f_{ss} (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
			\tag{turquin 2019, fig ??}\\
		\end{align*}
	*/
	const vec3 fAvg = f0 + (vec3(1.0f) - f0) * (1.0f / 21.0f);
	// floored away from 0; see the identical guard + explanation on
	// utilMicrofacetDielectricEnergyCompensate in util-material-openpbr-
	// microfacet.glsl -- same rational-fit-hits-its-own-clamp failure mode
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
