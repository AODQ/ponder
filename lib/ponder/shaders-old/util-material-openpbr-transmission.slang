module "util-material-openpbr-transmission";
__exported import "util-types";
__exported import "util-material-openpbr-microfacet";


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
// the ior the base dielectric refracts and fresnels at. openpbr requires the
// reflection and transmission sides of the same interface to share \eta'_s,
// so specular_weight's ior modulation belongs here too, not just on the
// reflection lobe (openpbr spec, specular: "the fresnel transmission factor
// and refraction into and out of the base dielectric should also be
// consistent with the IOR ratio")
public func openPbrTransmissionIor(
	mat: OpenPbrMaterial,
) -> f32 {
	return openPbrSpecularEffectiveIor(mat);
}
// (TODO REVIEW)

// pick one of the three fraunhofer lines discretized/stochastically per-path
public func openPbrDispersionIorRgb(
	mat: OpenPbrMaterial,
) -> f32v3 {
	// (TODO REVIEW)
	// n_d is the modulated ior, per the spec's "specular_ior (including any
	// modulation via specular_weight) defines n(\lambda_d)"
	const f32 etaD = openPbrTransmissionIor(mat);
	// (TODO REVIEW)
	if (mat.transmissionDispersionScale <= 0.0f) {
		return f32v3(etaD);
	}
	// \lambda_F, \lambda_d, \lambda_C, in micrometers
	const f32 lambdaF = 0.4861f;
	const f32 lambdaD = 0.5876f;
	const f32 lambdaC = 0.6563f;

	// V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale;
	const f32 vd = (
		mat.transmissionDispersionAbbeNumber
		/ mat.transmissionDispersionScale
	);
	const f32 vdSafe = vd != 0.0f ? vd : 1e-4f;
	// B = (n_d - 1) / (V_d (\lambda_F^{-2} - \lambda_C^{-2}))
	const f32 b = (
		(etaD - 1.0f)
		/ (vdSafe * (1.0f / (lambdaF * lambdaF) - 1.0f / (lambdaC * lambdaC)))
	);
	// A = n_d - B / \lambda_d^2
	const f32 a = etaD - b / (lambdaD * lambdaD);

	// n(\lambda) = A + B / \lambda^2, at the C/F lines;
	// n(\lambda_d) = etaD
	return f32v3(
		a + b / (lambdaC * lambdaC),
		etaD,
		a + b / (lambdaF * lambdaF)
	);
}

public func openPbrTransmissionEvaluateF(
	mat: OpenPbrMaterial,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32v3 {
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
	const f32v3 nor = frame.nor;
	// entering: etaI=1 (air), etaT=dispersedIor
	// exiting:  etaI=dispersedIor, etaT=1 (air)
	const f32 etaI = isInsideMedium ? dispersedIor : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : dispersedIor;
	// khr_materials_transmission without khr_materials_volume:
	// a thin shell with no second surface to refract back out of,
	// so have to handle delta lobe as special case.
	// etaI == etaT behaves the same
	if (mat.geometryThinWalled > 0.0f || etaI == etaT) {
		// (TODO REVIEW)
		// this branch is a dirac delta along wo == -wi -- openPbrSampleWo's
		// thin-walled case sets exactly that, and refract() at eta 1 returns
		// its incident vector unchanged. every other direction has density 0,
		// so returning the finite value there lets an nee connection collect
		// light through an interface that cannot actually scatter into it
		if (dot(wo, -wi) < 1.0f - 1e-5f) {
			return f32v3(0.0f);
		}
		// (TODO REVIEW)
		const f32 dotNorWi = max(dot(nor, wi), 1e-5f);
		const f32 dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
		const f32 F = utilMicrofacetFresnelDielectric(dotNorWi, etaT / etaI);
		const f32v3 tint = (
			(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : f32v3(1.0f)
		);
		return tint * (1.0f - F) / dotNorWo;
	}
	// h_t = -normalize(\eta_i \omega_i + \eta_t \omega_o)
	const f32v3 htUnnorm = -(etaI * wi + etaT * wo);
	const f32v3 ht = normalize(htUnnorm);
	const f32v3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(wi, htOriented) * dot(wo, htOriented) > 0.0f) {
		return f32v3(0.0f);
	}

	const f32 dotNorWi = max(dot(nor, wi), 1e-5f);
	const f32 dotNorWo = max(abs(dot(nor, wo)), 1e-5f);
	// |\omega_i \cdot h_t|, |\omega_o \cdot h_t|
	const f32 dotHtWi = max(dot(htOriented, wi), 0.0f);
	const f32 dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	const f32 dotNorHt = max(dot(nor, htOriented), 0.0f);

	// refraction shares the dielectric interface with the specular reflection
	// lobe, so the same slopes and (gltf-rotated) tangent frame drive both
	const f32v2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const f32v3 htLocal = (
		f32v3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	const f32v3 wiLocal = (
		f32v3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	// A_\omega is even in z, so the transmitted \omega_o enters with the
	// already-abs'd dotNorWo exactly as the isotropic form takes abs()
	const f32v3 woLocal = (
		f32v3(dot(wo, specFrame.tanX), dot(wo, specFrame.tanY), dotNorWo)
	);
	// F(\omega_i \cdot h_t, \eta_t / \eta_i);
	const f32 F = utilMicrofacetFresnelDielectric(dotHtWi, etaT / etaI);
	// D(h_t)
	const f32 D = utilMicrofacetGgxDistributionAniso(htLocal, alpha);
	// V = G_2 / (4 |\omega_i \cdot n| |\omega_o \cdot n|)
	const f32 V = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// |\eta_i \omega_i + \eta_t \omega_o|
	const f32 denom = length(htUnnorm);
	const f32 btdf = (
		(1.0f - F) * D * V
		* 4.0f * dotHtWi * dotHtWo * etaT * etaT
		/ max(denom * denom, 1e-6f)
	);
	// transmissionDepth = 0, constant fresnel tint
	const f32v3 tint = (
		(mat.transmissionDepth <= 0.0f) ? mat.transmissionColor : f32v3(1.0f)
	);
	return tint * btdf;
}

// pdf of VNDF-sampled refraction direction (walter et al. 2007)
public func openPbrTransmissionPdf(
	mat: OpenPbrMaterial,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
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
	const f32v3 nor = frame.nor;
	const f32 etaI = isInsideMedium ? dispersedIor : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : dispersedIor;
	// thin shell is delta dirac distribution
	if (mat.geometryThinWalled > 0.0f || etaI == etaT) {
		// (TODO REVIEW)
		// zero off the delta direction, matching openPbrTransmissionEvaluateF:
		// an nee query lands here for arbitrary wo, and reporting density 1
		// there both invents light and mis-weights the real delta bounce
		return dot(wo, -wi) < 1.0f - 1e-5f ? 0.0f : 1.0f;
		// (TODO REVIEW)
	}
	const f32v3 htUnnorm = -(etaI * wi + etaT * wo);
	const f32v3 ht = normalize(htUnnorm);
	const f32v3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
	// physical refraction requires wi and wo on opposite sides of h_t
	if (dot(wi, htOriented) * dot(wo, htOriented) > 0.0f) {
		return 0.0f;
	}
	const f32 dotNorWi = max(dot(nor, wi), 1e-5f);
	const f32 dotNorHt = max(dot(nor, htOriented), 0.0f);
	const f32 dotHtWi = max(dot(htOriented, wi), 0.0f);
	const f32 dotHtWo = max(abs(dot(htOriented, wo)), 0.0f);
	// slopes and frame must match the vndf sampling in openPbrSampleWo;
	// openPbrRoughnessAlpha floors both axes at 1e-5
	const f32v2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const f32v3 wiLocal = (
		f32v3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	const f32v3 htLocal = (
		f32v3(
			dot(htOriented, specFrame.tanX),
			dot(htOriented, specFrame.tanY),
			dotNorHt
		)
	);
	const f32 G1 = utilMicrofacetSmithG1Aniso(wiLocal, alpha);
	const f32 D = utilMicrofacetGgxDistributionAniso(htLocal, alpha);
	// see openPbrTransmissionEvaluateF's denom comment: length of the
	// unnormalized half-vector sum, not a sum of clamped cosines
	const f32 denom = length(htUnnorm);
	return (
		G1 * D * dotHtWi / dotNorWi
		* etaT * etaT * dotHtWo
		/ max(denom * denom, 1e-6f)
	);
}

// pdf of the transmission lobe's TIR fallback
public func openPbrTransmissionTirReflectPdf(
	mat: OpenPbrMaterial,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32 {
	const f32v3 nor = frame.nor;
	f32v3 h;
	if (!utilHalfVectorLocal(wi, wo, h)) {
		return 0.0f;
	}
	const f32 dotNorH = dot(nor, h);
	const f32 dotWiH = dot(wi, h);
	if (dotNorH <= 0.0f || dotWiH <= 0.0f) {
		return 0.0f;
	}
	const f32 etaI = isInsideMedium ? dispersedIor : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : dispersedIor;
	const f32v3 refracted = refract(-wi, h, etaI / etaT);
	if (length(refracted) > 0.5f) {
		return 0.0f;
	}
	// same slopes and frame as the refraction branch it falls back from
	const f32v2 alpha = (
		openPbrRoughnessAlpha(
			mat.specularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	const f32 dotNorWi = max(dot(nor, wi), 1e-5f);
	const f32v3 hLocal = (
		f32v3(dot(h, specFrame.tanX), dot(h, specFrame.tanY), dotNorH)
	);
	const f32v3 wiLocal = (
		f32v3(dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dotNorWi)
	);
	const f32 D = utilMicrofacetGgxDistributionAniso(hLocal, alpha);
	const f32 G1 = utilMicrofacetSmithG1Aniso(wiLocal, alpha);
	return (D * G1 * dotWiH) / (dotNorWi * max(4.0f * dotWiH, 1e-6f));
}

