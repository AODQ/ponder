#ifndef UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL
#define UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL

// classical two-term cauchy dispersion equation (cauchy 1836; not paper-
// specific, cf. hecht, optics), reparametrized via the abbe number exactly
// the way the openpbr spec and gltf KHR_materials_dispersion both expose
// it: an artist-friendly (scale, abbe_number) pair rather than raw cauchy
// coefficients. lib/mor/mor.cpp:material_load_dispersion already converts
// KHR_materials_dispersion's `dispersion` field into this same
// (transmissionDispersionScale, transmissionDispersionAbbeNumber) pair, so
// this is the one formula both material sources share
/*
	\begin{align*}
	&\textbf{(1) cauchy's equation} \\
	&n(\lambda) = A + B / \lambda^2
		\tag{cauchy 1836}\\
	&\textbf{(2) coefficients from ior + abbe number} \\
	% n_d, \lambda_d: refractive index and wavelength at the fraunhofer
	% d-line (587.6nm); \lambda_C, \lambda_F: fraunhofer C/F lines
	% (656.3nm, 486.1nm) -- the same calibration points the abbe number
	% v_d itself is defined against
	&B = \frac{n_d - 1}{V_d (\lambda_F^{-2} - \lambda_C^{-2})}
		\tag{openpbr spec, transmission dispersion}\\
	&A = n_d - B / \lambda_d^2
		\tag{openpbr spec, transmission dispersion}\\
	&\textbf{(3) artist-friendly abbe number} \\
	% scale = 0 gives v_d = \infty (no dispersion); scale = 1 gives the
	% literal authored abbe number
	&V_d =
		\mathrm{transmission\_dispersion\_abbe\_number}
		/ \mathrm{transmission\_dispersion\_scale}
		\tag{openpbr spec, transmission dispersion}\\
	\end{align*}
*/
// this renderer is otherwise plain rgb (no spectral upsampling of textures
// or lights), so there is no continuous spectrum for n(\lambda) to
// integrate over here; it is instead evaluated at exactly the three
// fraunhofer lines that calibrate v_d itself, and util-raytrace.glsl picks
// one of the three stochastically per path (a discretized hero-wavelength
// scheme; see the dispersion comment in utilIrradiance)
vec3 openPbrDispersionIorRgb(const OpenPbrMaterial mat) {
	if (mat.transmissionDispersionScale <= 0.0f) {
		return vec3(mat.specularIor);
	}
	// \lambda_F, \lambda_d, \lambda_C, in micrometers (matches this file's
	// existing mat.thinFilmThickness micrometer convention)
	const float lambdaF = 0.4861f;
	const float lambdaD = 0.5876f;
	const float lambdaC = 0.6563f;

	// V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale;
	// abbe number floored away from 0 so a scale > 0 paired with an
	// unset/zero abbe number (e.g. a future material-loading path that
	// forgets to set both together) can't send vd, and therefore b, to 0
	// and +inf below -- a + b/lambdaC^2 would then subtract inf - inf into
	// a NaN rather than a merely-wrong ior
	const float vd = (
		mat.transmissionDispersionAbbeNumber
		/ mat.transmissionDispersionScale
	);
	const float vdSafe = vd != 0.0f ? vd : 1e-4f;
	// B = (n_d - 1) / (V_d (\lambda_F^{-2} - \lambda_C^{-2}))
	const float b = (
		(mat.specularIor - 1.0f)
		/ (vdSafe * (1.0f / (lambdaF * lambdaF) - 1.0f / (lambdaC * lambdaC)))
	);
	// A = n_d - B / \lambda_d^2
	const float a = mat.specularIor - b / (lambdaD * lambdaD);

	// n(\lambda) = A + B / \lambda^2, at the C/F lines; a and b were solved
	// to make n(\lambda_d) equal mat.specularIor exactly, so the green
	// channel needs no separate evaluation
	return vec3(
		a + b / (lambdaC * lambdaC),
		mat.specularIor,
		a + b / (lambdaF * lambdaF)
	);
}

vec3 openPbrTransmissionEvaluateF(
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// this path's hero-wavelength ior (openPbrDispersionIorRgb, resolved
	// per-bounce in utilIrradiance); equals mat.specularIor whenever
	// transmission_dispersion_scale is 0 or no wavelength is locked yet.
	// only the transmitted ray bends per-wavelength -- the front-face
	// fresnel reflection elsewhere in this file's siblings stays on the
	// flat mat.specularIor, matching both openpbr and gltf's stated scope
	const float dispersedIor
) {
	/*
		microfacet BTDF (Walter et al. 2007):\\
		h_t =
			-(\eta_i \omega_i + \eta_t \omega_o)
			/ |\eta_i \omega_i + \eta_t \omega_o|
		\\
		f_t(\omega_i, \omega_o) =
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
		V = G_2 / (4 |\omega_i \cdot n| |\omega_o \cdot n|) \\
		simplifies to:\\
		f_t =
			(1 - F) \cdot D \cdot V \cdot
			\frac{
				4 |\omega_i \cdot h_t|
				\cdot |\omega_o \cdot h_t|
				\cdot \eta_t^2
			} {
				(\eta_i |\omega_i \cdot h_t| + \eta_t |\omega_o \cdot h_t|)^2
			}
		\\
		when transmissionDepth = 0: transmissionColor constant Fresnel factor
		when transmissionDepth > 0: Beer-Lambert
	*/
	// entering: etaI=1 (air), etaT=dispersedIor (glass)
	// exiting:  etaI=dispersedIor (glass), etaT=1 (air)
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	// khr_materials_transmission without khr_materials_volume: an
	// infinitely thin shell has no second surface to refract back out of,
	// so it is sampled in openPbrSampleWo as a delta lobe at wo = -wi
	// (undeviated) rather than through the continuous vndf btdf below.
	// the caller multiplies this by |dot(nor, wo)| and divides by
	// openPbrTransmissionPdf, so dividing by dotNorWo here cancels that
	// cosine, leaving the plain (1 - F) transmittance
	if (mat.geometryThinWalled > 0.0f) {
		const float dotNorWi = max(dot(nor, wi), 1e-5f);
		const float dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
		const float F = utilMicrofacetFresnelDielectric(dotNorWi, etaT / etaI);
		const vec3 tint = (
			(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : vec3(1.0f)
		);
		return tint * (1.0f - F) / dotNorWo;
	}
	// h_t = -normalize(\eta_i \omega_i + \eta_t \omega_o)
	const vec3 ht = normalize(-(etaI * wi + etaT * wo));
	const vec3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;

	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const float dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
	// |\omega_i \cdot h_t|, |\omega_o \cdot h_t|
	const float dotHtWi = max(dot(htOriented, wi), 0.0f);
	const float dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	const float dotNorHt = max(dot(nor, htOriented), 0.0f);

	// D and V square the slope internally; isotropic refraction for now
	const float alpha = mat.specularRoughness;
	// F(\omega_i \cdot h_t, \eta_t / \eta_i);
	const float F = utilMicrofacetFresnelDielectric(dotHtWi, etaT / etaI);
	// D(h_t)
	const float D = utilMicrofacetGgxDistribution(dotNorHt, alpha);
	// V = G_2 / (4 |\omega_i \cdot n| |\omega_o \cdot n|)
	const float V = utilMicrofacetSmithGgxVisibility(dotNorWi, dotNorWo, alpha);
	// \eta_i |\omega_i \cdot h_t| + \eta_t |\omega_o \cdot h_t|
	const float denom = etaI * dotHtWi + etaT * dotHtWo;
	const float btdf = (
		(1.0f - F) * D * V
		* 4.0f * dotHtWi * dotHtWo * etaT * etaT
		/ max(denom * denom, 1e-6f)
	);
	// transmissionDepth = 0:
	// transmissionColor is a constant Fresnel tint
	const vec3 tint = (
		(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : vec3(1.0f)
	);
	return tint * btdf;
}

// pdf of a VNDF-sampled refraction direction (walter et al. 2007)
float openPbrTransmissionPdf(
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// must match the dispersedIor openPbrTransmissionEvaluateF was called
	// with for this same wo, or the sampler and pdf disagree
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
	// infinitely thin shell: the whole density sits at the single
	// wo = -wi direction openPbrSampleWo produces; the mixture pdf's
	// caller already weights this by sel.probabilityTransmission, so this
	// just contributes the lobe's full weight (1) to that sum
	if (mat.geometryThinWalled > 0.0f) {
		return 1.0f;
	}
	// entering: etaI=1 (air), etaT=dispersedIor (glass)
	// exiting:  etaI=dispersedIor (glass), etaT=1 (air)
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	const vec3 ht = normalize(-(etaI * wi + etaT * wo));
	const vec3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const float dotNorHt = max(dot(nor, htOriented), 0.0f);
	const float dotHtWi = max(dot(htOriented, wi), 0.0f);
	const float dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	// roughness clamp must match the vndf sampling in openPbrSampleWo
	// G1 and D square the slope internally; isotropic refraction for now
	const float alpha = max(mat.specularRoughness, 1e-5f);
	const float G1 = utilMicrofacetSmithG1(dotNorWi, alpha);
	const float D = utilMicrofacetGgxDistribution(dotNorHt, alpha);
	const float denom = etaI * dotHtWi + etaT * dotHtWo;
	return (
		G1 * D * dotHtWi / dotNorWi
		* etaT * etaT * dotHtWo
		/ max(denom * denom, 1e-6f)
	);
}

// pdf contribution of the transmission lobe's total-internal-reflection
// fallback: when the vndf-sampled half vector h fails to refract,
// openPbrSampleWo reflects wi about h instead, so the transmission lobe
// also has non-zero density over reflected wo directions. no closed form
// restricts this to just the h's that actually undergo TIR, so this uses
// the full (unrestricted) vndf reflect pdf as a standard approximation --
// it slightly overweights the transmission lobe's reflection-side pdf,
// which is the safe direction (avoids the fireflies an underweighted
// pdf would cause)
/*
	p(h) = D(h) \, G_1(\omega_i) \, \max(\omega_i \cdot h, 0) / (\omega_i \cdot n)
	\\
	pdf(\omega_o) = p(h) / (4 (\omega_i \cdot h))
*/
float openPbrTransmissionTirReflectPdf(
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo
) {
	const float alpha = max(mat.specularRoughness, 1e-5f);
	const vec3 h = normalize(wi + wo);
	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	const float dotNorH = max(dot(nor, h), 0.0f);
	const float dotWiH = max(dot(wi, h), 0.0f);
	const float D = utilMicrofacetGgxDistribution(dotNorH, alpha);
	const float G1 = utilMicrofacetSmithG1(dotNorWi, alpha);
	return (D * G1 * dotWiH) / (dotNorWi * max(4.0f * dotWiH, 1e-6f));
}

#endif // UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL
