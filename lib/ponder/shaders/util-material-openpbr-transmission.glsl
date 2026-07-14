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
	//
	// etaI == etaT (matched ior, e.g. dispersedIor == 1) shares this exact
	// delta lobe: an index-matched interface never bends light (refract
	// with eta=1 returns wo=-wi for every microfacet) and reflects nothing
	// at any angle (F is identically 0 when eta=1), so it is physically
	// the same undeviated full-transmission case as thin-walled rather
	// than the continuous rough btdf below -- this also sidesteps the
	// htUnnorm = -(etaI*wi + etaT*wo) = 0 degenerate normalize() that
	// otherwise fires when wo lands exactly at -wi
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
	// |\eta_i \omega_i + \eta_t \omega_o|; the length of the *unnormalized*
	// half-vector sum, not the sum of two independently-clamped cosines --
	// wi and wo sit on opposite sides of the interface so the signed dot
	// products partially cancel; abs-then-add discards that cancellation
	// and badly inflates this term off the normal-incidence axis. cf.
	// blender cycles bsdf_microfacet.h, which computes this identically
	// via 1/inv_len_H (see transmission verification report)
	const float denom = length(htUnnorm);
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
	// entering: etaI=1 (air), etaT=dispersedIor (glass)
	// exiting:  etaI=dispersedIor (glass), etaT=1 (air)
	const float etaI = isInsideMedium ? dispersedIor : 1.0f;
	const float etaT = isInsideMedium ? 1.0f : dispersedIor;
	// infinitely thin shell: the whole density sits at the single
	// wo = -wi direction openPbrSampleWo produces; the mixture pdf's
	// caller already weights this by sel.probabilityTransmission, so this
	// just contributes the lobe's full weight (1) to that sum.
	// etaI == etaT (matched ior) shares this exact delta lobe -- see
	// openPbrTransmissionEvaluateF's matching comment
	if (mat.geometryThinWalled > 0.0f || etaI == etaT) {
		return 1.0f;
	}
	const vec3 htUnnorm = -(etaI * wi + etaT * wo);
	const vec3 ht = normalize(htUnnorm);
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
	// see openPbrTransmissionEvaluateF's denom comment: length of the
	// unnormalized half-vector sum, not a sum of clamped cosines
	const float denom = length(htUnnorm);
	return (
		G1 * D * dotHtWi / dotNorWi
		* etaT * etaT * dotHtWo
		/ max(denom * denom, 1e-6f)
	);
}

// pdf contribution of the transmission lobe's total-internal-reflection
// fallback: when the vndf-sampled half vector h fails to refract,
// openPbrSampleWo reflects wi about h instead, so the transmission lobe
// also has non-zero density over reflected wo directions. the pdf below is
// the unbounded VNDF-half-vector-to-reflected-wo density -- the correct
// match for this fallback's actual sampler, which draws h via the plain
// utilMicrofacetSampleGgxVndf (not the bounded wo-space sampler specular's
// own lobe uses), then reflects wi about it.
//
// restricted to h's that actually underwent TIR (reconstruct the same
// half vector openPbrSampleWo's vndf path would have sampled, then run
// the identical refract() check that function uses to decide TIR vs
// refraction). originally unrestricted -- claiming density at every h
// regardless of whether it truly TIRs -- which is documented in cull as
// "the safe direction (avoids the fireflies an underweighted pdf would
// cause)", true only when transmission is evaluated alone: its own
// sampler never draws from the non-TIR subset, so the excess claim is
// never actually divided against a real sample. found broken by the
// mixture-lobe-selection milestone's transmission+specular chi-square: a
// separately-normalized reflecting lobe (specular) legitimately covers
// those same non-TIR-restricted directions with its own pdf, so summing
// the two in openPbrEvaluatePdfSelected double-counted density there --
// measured ~10x the typical local pdf at what first looked like a
// grazing-retroreflection config, but was actually just an entering
// (isInsideMedium=false) ray, where eta = etaI/etaT < 1 makes TIR
// physically impossible -- this restriction alone zeroes the term for
// every entering config, which is what the mixture tests exercise. this
// restriction is the same one the pre-existing test harness
// (sampling_expected.comp's transmission modes) already applied on the
// test side via a hand-derived refraction discriminant; moving it into
// the production function is the actual fix, not a duplicate.
//
// tried switching to the bounded reflect pdf (utilMicrofacetGgxBoundedReflectPdfAniso,
// cf. the reflection-side specular lobe) instead of the classical
// unbounded p(h) = D(h) G_1(wi) |wi.h| / (wi.n) form below, on the theory
// that the unbounded formula's grazing blowup was the double-counting
// cause. reverted: that theory was wrong (the restriction above already
// zeroes the entering case where the blowup was actually observed,
// independent of which formula is used), and the bounded variant is
// calibrated for a *different* sampler (the bounded wo-space one) --
// swapping it in undercounted the genuine-TIR (exiting) case by
// 10%-40% at high roughness, breaking the pre-existing transmission-alone
// "restricted + tir-restricted integral is 1" test. the unbounded form
// here is the one that actually matches this fallback's sampler.
float openPbrTransmissionTirReflectPdf(
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// must match the dispersedIor openPbrSampleWo/openPbrTransmissionPdf
	// used for this same wo, or the TIR restriction below can disagree
	// with which h the sampler actually walked
	const float dispersedIor
) {
	// wo landing exactly at -wi (a grazing retroreflection) makes wi + wo
	// the zero vector, whose normalize() is a NaN that then slips past the
	// dotNorH/dotWiH checks below -- any comparison against NaN is false,
	// so the guard would fall through instead of early-returning. caught
	// via the furnace model-render test's DragonAttenuation probe. zero
	// pdf for the exact zero vector, the pbrt convention
	// (`if (wh == Vector3f(0)) return 0;`) -- a substitute direction was
	// tried and reverted, see utilMicrofacetGgxBoundedReflectPdfAniso's
	// matching guard comment for why
	const vec3 hUnnorm = wi + wo;
	if (dot(hUnnorm, hUnnorm) == 0.0f) {
		return 0.0f;
	}
	const vec3 h = normalize(hUnnorm);
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
	// a tir-fallback wo is reflective (dot(nor,wo) >= 0), so openPbrEvaluateF
	// routes it through the normal reflection stack --
	// openPbrDielectricSpecularEvaluateF, which evaluates D via the
	// anisotropic distribution. this pdf must reconstruct that same D, or
	// the two disagree on how much density this wo actually carries. the
	// isotropic and anisotropic distribution formulas are the same function
	// reparametrized (equal wherever neither floor triggers), but their
	// floors guard different terms at different absolute thresholds (1e-5
	// on the isotropic d vs 1e-7 on the anisotropic d, which is
	// d_iso / alpha^2) -- at ordinary roughness both floors are inactive
	// and the two formulas agree exactly, but at very low roughness (a
	// smooth, highly-transmissive dielectric) the isotropic floor clamps
	// its peak far below the anisotropic one, so a bounce that eval scores
	// against the (much taller) anisotropic peak was being paired with a
	// pdf computed against the (much shorter) isotropic one. individually
	// small per-bounce, this compounded across many internal tir bounces
	// into an unbounded firefly (found via the furnace model-render test's
	// DragonAttenuation case).
	//
	// deliberately NOT coat-roughened to match openPbrDielectricSpecular-
	// EvaluateF's openPbrCoatRoughenedRoughness step exactly: several
	// existing unit-test shaders (e.g. transmission_integrate.comp) build a
	// bare OpenPbrMaterial without ever touching coatWeight/coatRoughness,
	// which are then read uninitialized -- pulling those fields in here
	// regressed the "restricted + tir-restricted integral is 1" normalization
	// test on exactly that garbage. plain specularRoughness matches what
	// this function always used, and reproduces the DragonAttenuation fix
	// exactly since that material has coatWeight=0 (coat-roughening is a
	// no-op there); a coated transmissive material is already documented
	// as uncovered scope (see openPbrEvaluateF's "no furnace-test coverage
	// exists for coat/fuzz stacked on a transmissive base" comment)
	//
	// anisotropy itself is assumed 0 here (alphaX = alphaY): the tir
	// sampler this pdf backs (utilMicrofacetSampleGgxVndf, called from
	// openPbrSampleWo's transmission branch) is isotropic-only, "for now"
	// like the rest of this file, so there is no anisotropic sample
	// distribution to match in the first place -- this exactly matches
	// eval for an isotropic material, and is a strictly closer
	// approximation than the old mismatched-floor isotropic pdf for an
	// anisotropic one
	const float alpha = max(mat.specularRoughness, 1e-5f);
	const float dotNorWi = max(dot(nor, wi), 1e-5f);
	vec3 tanX, tanY;
	utilCalculateXy(nor, tanX, tanY);
	const vec3 hLocal = vec3(dot(h, tanX), dot(h, tanY), dotNorH);
	const float D = (
		utilMicrofacetGgxDistributionAniso(hLocal, vec2(alpha, alpha))
	);
	const float G1 = utilMicrofacetSmithG1(dotNorWi, alpha);
	return (D * G1 * dotWiH) / (dotNorWi * max(4.0f * dotWiH, 1e-6f));
}

#endif // UTIL_MATERIAL_OPENPBR_TRANSMISSION_GLSL
