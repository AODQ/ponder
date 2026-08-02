implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

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
// (TODO REVIEW)
public func bsdfOpenPbrTransmissionIor(
	mat: MaterialBackend_OpenPbr.Params,
) -> f32 {
	return bsdfOpenPbrSpecularEffectiveIor(mat);
}

// n(\lambda) at one wavelength in micrometers
public func bsdfOpenPbrDispersionIor(
	mat: MaterialBackend_OpenPbr.Params,
	λMicrometers: f32,
) -> f32 {
	// n_d is the modulated ior
	let ηD: f32 = bsdfOpenPbrTransmissionIor(mat);
	if (mat.transmissionDispersionScale <= 0.0f) {
		return ηD;
	}
	// \lambda_F, \lambda_d, \lambda_C, in micrometers
	let λF: f32 = 0.4861f;
	let λD: f32 = 0.5876f;
	let λC: f32 = 0.6563f;

	// V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale;
	let vd: f32 = (
		mat.transmissionDispersionAbbeNumber
		/ mat.transmissionDispersionScale
	);
	let vdSafe: f32 = vd != 0.0f ? vd : 1e-4f;
	// B = (n_d - 1) / (V_d (\lambda_F^{-2} - \lambda_C^{-2}))
	let b: f32 = (
		(ηD - 1.0f)
		/ (vdSafe * (1.0f / (λF * λF) - 1.0f / (λC * λC)))
	);
	// A = n_d - B / \lambda_d^2
	let a: f32 = ηD - b / (λD * λD);

	// n(\lambda) = A + B / \lambda^2
	return a + b / (λMicrometers * λMicrometers);
}

// rgb build: the C, d, F fraunhofer lines per channel
public func bsdfOpenPbrDispersionIorRgb(
	mat: MaterialBackend_OpenPbr.Params,
) -> f32v3 {
	return f32v3(
		bsdfOpenPbrDispersionIor(mat, 0.6563f),
		bsdfOpenPbrTransmissionIor(mat),
		bsdfOpenPbrDispersionIor(mat, 0.4861f)
	);
}

public func bsdfOpenPbrTransmissionEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
	λ: Wavelengths,
) -> vkof::Spectrum {
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
	let nor: f32v3 = frame.nor;
	// entering: etaI=1 (air), etaT=dispersedIor
	// exiting:  etaI=dispersedIor, etaT=1 (air)
	let ηI: f32 = isInsideMedium ? dispersedIor : 1.0f;
	let ηT: f32 = isInsideMedium ? 1.0f : dispersedIor;
	// transmissionDepth = 0, constant fresnel tint
	let tint: vkof::Spectrum = (
		(mat.transmissionDepth <= 0.0f)
		? mat.transmissionColor
		: vkof::Spectrum.one()
	);
	// khr_materials_transmission without khr_materials_volume:
	// a thin shell with no second surface to refract back out of,
	// so have to handle delta lobe as special case.
	// etaI == etaT behaves the same
	if (mat.geometryThinWalled > 0.0f || ηI == ηT) {
		// dirac delta along wo == -wi
		if (dot(ωo, -ωi) < 1.0f - 1e-5f) {
			return vkof::Spectrum.zero();
		}
		let dotNorWi: f32 = max(dot(nor, ωi), 1e-5f);
		let dotNorWo: f32 = max(abs(dot(nor, ωo)), 1e-5f);
		let F: f32 = bsdfFresnelDielectric(dotNorWi, ηT / ηI);
		return tint * ((1.0f - F) / dotNorWo);
	}
	// h_t = -normalize(\eta_i \omega_i + \eta_t \omega_o)
	let htUnnorm: f32v3 = -(ηI * ωi + ηT * ωo);
	let ht: f32v3 = normalize(htUnnorm);
	let htOriented: f32v3 = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(ωi, htOriented) * dot(ωo, htOriented) > 0.0f) {
		return vkof::Spectrum.zero();
	}

	let dotNorWi: f32 = max(dot(nor, ωi), 1e-5f);
	let dotNorWo: f32 = max(abs(dot(nor, ωo)), 1e-5f);
	// |\omega_i \cdot h_t|, |\omega_o \cdot h_t|
	let dotHtWi: f32 = max(dot(htOriented, ωi), 0.0f);
	let dotHtWo: f32 = max(abs(dot(htOriented, ωo)), 0.0f);
	let dotNorHt: f32 = max(dot(nor, htOriented), 0.0f);

	// -- calculate microfacet terms
	let α: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	// refraction shares dielectric interface with specular lobe
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	let htLocal: f32v3 = (
		f32v3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	let ωiLocal: f32v3 = (
		f32v3(dot(ωi, specFrame.tanX), dot(ωi, specFrame.tanY), dotNorWi)
	);
	let ωoLocal: f32v3 = (
		f32v3(dot(ωo, specFrame.tanX), dot(ωo, specFrame.tanY), dotNorWo)
	);
	// F(\omega_i \cdot h_t, \eta_t / \eta_i);
	let F: f32 = bsdfFresnelDielectric(dotHtWi, ηT / ηI);
	// D(h_t)
	let D: f32 = bsdfMicrofacetGgxDistributionAniso(htLocal, α);
	// V = G_2 / (4 |\omega_i \cdot n| |\omega_o \cdot n|)
	let V: f32 = (
		bsdfMicrofacetSmithGgxVisibilityAniso(ωiLocal, ωoLocal, α)
	);
	// |\eta_i \omega_i + \eta_t \omega_o|
	let denom: f32 = length(htUnnorm);
	let btdf: f32 = (
		(1.0f - F) * D * V
		* 4.0f * dotHtWi * dotHtWo * ηT * ηT
		/ max(denom * denom, 1e-6f)
	);
	return tint * btdf;
}

// pdf of VNDF-sampled refraction direction (walter et al. 2007)
public func bsdfOpenPbrTransmissionPdf(
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32 {
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
	let nor: f32v3 = frame.nor;
	let ηI: f32 = isInsideMedium ? dispersedIor : 1.0f;
	let ηT: f32 = isInsideMedium ? 1.0f : dispersedIor;
	// thin shell is delta dirac distribution
	if (mat.geometryThinWalled > 0.0f || ηI == ηT) {
		// dirac delta along wo == -wi
		return dot(ωo, -ωi) < 1.0f - 1e-5f ? 0.0f : 1.0f;
	}
	let htUnnorm: f32v3 = -(ηI * ωi + ηT * ωo);
	let ht: f32v3 = normalize(htUnnorm);
	let htOriented: f32v3 = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(ωi, htOriented) * dot(ωo, htOriented) > 0.0f) {
		return 0.0f;
	}
	let dotNorWi: f32 = max(dot(nor, ωi), 1e-5f);
	let dotNorHt: f32 = max(dot(nor, htOriented), 0.0f);
	let dotHtWi: f32 = max(dot(htOriented, ωi), 0.0f);
	let dotHtWo: f32 = max(abs(dot(htOriented, ωo)), 0.0f);
	// -- calculate microfacet terms
	// slopes and frame must match the vndf sampling in bsdfOpenPbrSampleWo
	let α: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	let ωiLocal: f32v3 = (
		f32v3(dot(ωi, specFrame.tanX), dot(ωi, specFrame.tanY), dotNorWi)
	);
	let htLocal: f32v3 = (
		f32v3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	let G1: f32 = bsdfMicrofacetSmithG1Aniso(ωiLocal, α);
	let D: f32 = bsdfMicrofacetGgxDistributionAniso(htLocal, α);
	let denom: f32 = length(htUnnorm);
	return (
		G1 * D * dotHtWi / dotNorWi
		* ηT * ηT * dotHtWo
		/ max(denom * denom, 1e-6f)
	);
}

// pdf of the transmission lobe's TIR fallback
public func bsdfOpenPbrTransmissionTirReflectPdf(
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32 {
	let nor: f32v3 = frame.nor;
	var h: f32v3;
	if (!bsdfHalfVectorLocal(ωi, ωo, h)) {
		return 0.0f;
	}
	let dotNorH: f32 = dot(nor, h);
	let dotWiH: f32 = dot(ωi, h);
	if (dotNorH <= 0.0f || dotWiH <= 0.0f) {
		return 0.0f;
	}
	let ηI: f32 = isInsideMedium ? dispersedIor : 1.0f;
	let ηT: f32 = isInsideMedium ? 1.0f : dispersedIor;
	let refracted: f32v3 = refract(-ωi, h, ηI / ηT);
	if (length(refracted) > 0.5f) {
		return 0.0f;
	}
	// -- calculate microfacet terms
	// same slopes and frame as bsdfOpenPbrTransmissionPdf
	let α: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	let dotNorWi: f32 = max(dot(nor, ωi), 1e-5f);
	let hLocal: f32v3 = (
		f32v3(dot(h, specFrame.tanX), dot(h, specFrame.tanY), dotNorH)
	);
	let ωiLocal: f32v3 = (
		f32v3(dot(ωi, specFrame.tanX), dot(ωi, specFrame.tanY), dotNorWi)
	);
	let D: f32 = bsdfMicrofacetGgxDistributionAniso(hLocal, α);
	let G1: f32 = bsdfMicrofacetSmithG1Aniso(ωiLocal, α);
	return (D * G1 * dotWiH) / (dotNorWi * max(4.0f * dotWiH, 1e-6f));
}

} // namespace vkof
