implementing bsdf;

import types;
import material;

// belcour & barla 2017 integrates the interference over the spectrum in rgb;
// the spectral build evaluates it per wavelength instead

namespace vkof {

/*
	dielectric interface fresnel
$$
	\begin{align*}
	&r_s =
		\frac
		{n_2 \cos\theta_1 - n_1 \cos\theta_2}
		{n_2 \cos\theta_1 + n_1 \cos\theta_2}
	\\
	&r_p =
		\frac
		{n_1 \cos\theta_1 - n_2 \cos\theta_2}
		{n_1 \cos\theta_1 + n_2 \cos\theta_2}
	\\
	&\phi_s = (r_s < 0)\ ?\ \pi : 0\\
	&\phi_p = (r_p < 0)\ ?\ \pi : 0\\
	\end{align*}
$$
$$
	\left(\frac{n_1}{n_2}\right)^2\sin^2\theta_1 > 1
$$
	phase shift on TIR is amplitude-dependent, but set to 0 since
	$T_{12} = 1 - R_{12}$ and $R_{12} = 1$
*/
public func bsdfFresnelDielectricPhase(
	cosθi: f32,
	cosθt: f32,
	ηi: f32,
	ηt: f32,
	out Rp: f32,
	out Rs: f32,
	out φp: f32,
	out φs: f32,
) {
	// \sin^2\theta_1
	let sinθiSq: f32 = 1.0f - cosθi * cosθi;
	// n_1 / n_2
	let nr: f32 = ηi / ηt;

	// (n_1/n_2)^2 \sin^2\theta_1 > 1
	if (nr * nr * sinθiSq > 1.0f) {
		Rp = 1.0f;
		Rs = 1.0f;
		φp = 0.0f;
		φs = 0.0f;
		return;
	}

	// r_s
	let rs: f32 = (
		// n_2 \cos\theta_1 - n_1 \cos\theta_2
		(ηt * cosθi - ηi * cosθt)
		// n_2 \cos\theta_1 + n_1 \cos\theta_2
		/ (ηt * cosθi + ηi * cosθt)
	);
	// r_p
	let rp: f32 = (
		// n_1 \cos\theta_1 - n_2 \cos\theta_2
		(ηi * cosθi - ηt * cosθt)
		// n_1 \cos\theta_1 + n_2 \cos\theta_2
		/ (ηi * cosθi + ηt * cosθt)
	);
	Rs = rs * rs;
	Rp = rp * rp;

	// -- phase shifts
	φs = (rs < 0.0f) ? π : 0.0f;
	φp = (rp < 0.0f) ? π : 0.0f;
}

/*
	generalized Fresnel equations for reflection at
	dielectric -> conductor interface.
	added to support metallic base beneath thin film
$$
	\begin{align*}
	&A = n_3^2(1 - \kappa_3^2) - n_2^2 \sin^2 \theta_2\\
	&B = \sqrt{A^2 + 4 n_3^4 \kappa_3^2}\\
	&U=\sqrt{\tfrac{A+B}{2}}\\
	&V=\sqrt{\tfrac{B-A}{2}}\\
	&R_p =
		\frac
		{(n_2 \cos\theta_2 - U)^2 + V^2}
		{(n_2 \cos\theta_2 + U)^2 + V^2}
	\\
	&\phi_p =
		\text{atan2}(2n_2 V \cos\theta_2,\ U^2 + V^2 - n_2^2 \cos^2 \theta_2)
		+ \pi
	\\
	&R_s =
		\frac
		{
			\left(n_3^2(1-\kappa_3^2)\cos\theta_2 - n_2 U\right)^2
			+ \left(2n_3^2\kappa_3\cos\theta_2 - n_2 V\right)^2
		}
		{
			\left(n_3^2(1-\kappa_3^2)\cos\theta_2 + n_2 U\right)^2
			+ \left(2n_3^2\kappa_3\cos\theta_2 + n_2 V\right)^2
		}
	\\
	&\phi_s =
		\text{atan2}\left(
			2n_2 n_3^2 \cos\theta_2 \left(2\kappa_3 U - (1-\kappa_3^2)V\right),\
			n_3^4(1+\kappa_3^2)^2\cos^2\theta_2 - n_2^2(U^2+V^2)
		\right)
	\\
	\end{align*}
$$
*/
public func bsdfFresnelConductorPhase(
	cosθi: f32,
	cosθt: f32,
	ηi: f32,
	ηt: f32,
	κt: f32,
	out Rp: f32,
	out Rs: f32,
	out φp: f32,
	out φs: f32,
) {
	// n_2 \cos\theta_2
	let n2cosθ2: f32 = ηi * cosθi;
	// \sin\theta_2
	let n2Sinθ2Sq: f32 = ηi * ηi - n2cosθ2 * n2cosθ2;
	// n_3
	let n3: f32 = ηt;
	// \kappa_3
	let k3: f32 = κt;

	if (k3 == 0.0f) {
		// if \kappa_3 = 0, then the next medium is a dielectric
		bsdfFresnelDielectricPhase(
			cosθi, cosθt, ηi, ηt, Rp, Rs, φp, φs
		);
		return;
	}

	// n_3^2(1-\kappa_3^2)
	let n3SqRe: f32 = n3 * n3 * (1.0f - k3 * k3);
	// 2n_3^2\kappa_3
	let n3SqIm: f32 = 2.0f * n3 * n3 * k3;
	// A = n_3^2(1- \kappa_3 ^ 2) - n_2^2\sin^2\theta_2
	let A: f32 = (n3SqRe - n2Sinθ2Sq);
	// B = sqrt(A^2 + 4n_3^4\kappa_3^2)
	let B: f32 = sqrt(A * A + n3SqIm * n3SqIm);
	// U = sqrt((A + B) / 2)
	let U: f32 = sqrt((A + B) / 2.0f);
	// V = sqrt((B - A) / 2)
	let V: f32 = sqrt((B - A) / 2.0f);

	// R_p
	Rp = (
		// (n_2 \cos\theta_2 - U)^2 + V^2
		((n2cosθ2 - U) * (n2cosθ2 - U) + V * V)
		// (n_2 \cos\theta_2 + U)^2 + V^2
		/ ((n2cosθ2 + U) * (n2cosθ2 + U) + V * V)
	);

	// phi_p
	φp = (
		// atan2(2n_2 V \cos\theta_2,\ U^2 + V^2 - n_2^2 \cos^2 \theta_2)
		atan2(2.0f * n2cosθ2 * V, U * U + V * V - n2cosθ2 * n2cosθ2)
		// + \pi
		+ π
	);

	// n_3^2(1-\kappa_3^2)\cos\theta_2 - n_2 U
	let sReMinus: f32 = n3SqRe * cosθi - ηi * U;
	// n_3^2(1-\kappa_3^2)\cos\theta_2 + n_2 U
	let sRePlus: f32 = n3SqRe * cosθi + ηi * U;
	// 2n_3^2\kappa_3\cos\theta_2 - n_2 V
	let sImMinus: f32 = n3SqIm * cosθi - ηi * V;
	// 2n_3^2\kappa_3\cos\theta_2 + n_2 V
	let sImPlus: f32 = n3SqIm * cosθi + ηi * V;

	// R_s
	Rs = (
		(sReMinus * sReMinus + sImMinus * sImMinus)
		/ (sRePlus * sRePlus + sImPlus * sImPlus)
	);

	// n_3^2(1+\kappa_3^2)\cos\theta_2
	let n3SqAbsCos: f32 = n3 * n3 * (1.0f + k3 * k3) * cosθi;
	// phi_s
	φs = (
		atan2(
			2.0f * ηi * n3 * n3 * cosθi * (2.0f * k3 * U - (1.0f - k3 * k3) * V),
			n3SqAbsCos * n3SqAbsCos - ηi * ηi * (U * U + V * V)
		)
	);
}

// per-channel overload for tinted base substrate,
public func bsdfFresnelConductorPhase(
	cosθi: f32,
	cosθt: f32v3,
	ηi: f32,
	ηt: f32v3,
	κt: f32v3,
	out Rp: f32v3,
	out Rs: f32v3,
	out φp: f32v3,
	out φs: f32v3,
) {
	bsdfFresnelConductorPhase(
		cosθi, cosθt.r, ηi, ηt.r, κt.r,
		Rp.r, Rs.r, φp.r, φs.r
	);
	bsdfFresnelConductorPhase(
		cosθi, cosθt.g, ηi, ηt.g, κt.g,
		Rp.g, Rs.g, φp.g, φs.g
	);
	bsdfFresnelConductorPhase(
		cosθi, cosθt.b, ηi, ηt.b, κt.b,
		Rp.b, Rs.b, φp.b, φs.b
	);
}

/*
	evaluates XYZ sensitivity curves in fourier space; a gaussian fit of
	the CIE 1931 XYZ color matching functions in the fourier domain
	(belcour & barla 2017, eq. 10 and its supplemental gaussian-fit
	coefficients)

	\mathrm{OPD} = 2 n_2 d \cos\theta_2

	% OPD = optical path difference, d = thin-film thickness

	this implementation comes from belcour 2017 bsdf code. the 1.0e-6
	scale below (rather than the paper code's 1.0e-9) is because
	mat.thinFilmThickness is stored in micrometers
*/
public func bsdfEvaluateXyzSensitivity(
	opd: f32,
	shift: f32,
) -> f32v3 {
	// gaussian fits
	let phase: f32 = 2*π * opd * 1.0e-6;
	let val: f32v3 = f32v3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
	let pos: f32v3 = f32v3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
	let variance: f32v3 = f32v3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
	var xyz: f32v3 = (
		val * sqrt(2.0f * π * variance)
		* cos(pos * phase + shift)
		* exp(-variance * phase * phase)
	);
	xyz.x += (
		9.7470e-14f * sqrt(2.0f * π * 4.5282e+09f)
		* cos(2.2399e+06f * phase + shift)
		* exp(-4.5282e+09f * phase * phase)
	);
	return xyz / 1.0685e-7f;
}

// per-channel version of the above for a tinted base: each channel only
// needs its own matching Gaussian (pos.x/y/z ~ red/green/blue)
public func bsdfEvaluateXyzSensitivityTinted(
	opd: f32,
	shift: f32v3,
) -> f32v3 {
	return f32v3(
		bsdfEvaluateXyzSensitivity(opd, shift.r).x,
		bsdfEvaluateXyzSensitivity(opd, shift.g).y,
		bsdfEvaluateXyzSensitivity(opd, shift.b).z
	);
}

/*
	thin-film interference layer over the base substrate, balcour & barla 2017

	this replaces the $F(h \cdot \omega_i)$ term in the original microfacet model
	with a thin-film interference $R_i(h \cdot \omega_i)$ term.

	At the top of the thin-film interface, light that transmits into it
	can be bounced multiple times. This is phase-shifted by
$$
	\phi_2 = \phi_{21} + \phi_{23} = (\pi - \phi_{12}) + \phi_{23}
$$

	where generally `_{ij}` describes light going from medium `i` to medium `j`.

	this can be written as a geometric series as shown in belcour 2017, eq. 10
$$
	\begin{align*}
	&R_j = C_0 + 2 \: \sum^{+\infty}_{m=n} C_m
		\binom{cos(m\phi_2)}{sin(m \phi_2)}^T
		\binom{\mathcal{R}_j(m\mathcal{D})}{\mathcal{J}_j(m\mathcal{D})}
		\tag{belcour \& barla 2017, eq. 10}
	\\
$$

	where $C_m$ is the relative reflectivity and the rest is the phase difference,
	given per-polarization $j \in \{p, s\}$ by (belcour \& barla 2017, eq. 10)
$$
	\begin{align*}
	&T_{12} = 1 - R_{12}\\
	&R_{123} = R_{12} R_{23}\\
	&\rho_{123} = \sqrt{R_{123}}\\
	&R = \frac{T_{12}^2 R_{23}}{1 - R_{123}}\\
	&C_0 = R_{12} + R\\
	&C_m = (R - T_{12}) \: \rho_{123}^{m}, \quad m \ge 1
	\end{align*}
$$

	unpolarized reflectance is the average of the two polarizations,
	$I = \tfrac{1}{2}(I_p + I_s)$, and the optical path difference for
	one round trip through the film of ior $n_2$ and thickness $d$ is
$$
	\mathrm{OPD} = 2 n_2 d \cos\theta_2
$$
*/
public func bsdfOpenPbrThinfilmFresnel(
	mat: MaterialBackend_OpenPbr.Params,
	cosθi: f32,
	ηBase: f32v3,
	κBase: f32v3,
) -> f32v3 {
	// xyz to cie 1931 rgb color space, neutral white point
	let xyzToRgb: f32m3x3 = f32m3x3(
		2.3706743f, -0.5138850f, 0.0052982f,
		-0.9000405f, 1.4253036f, -0.0146949f,
		-0.4706338f, 0.0885814f, 1.0093968f
	);

	let η2: f32 = (
		lerp(1.0f, mat.thinFilmIor, smoothstep(0.0f, 0.03f, mat.thinFilmThickness))
	);
	let η2Inv: f32 = 1.0f / η2;

	let cosθt: f32 = (
		sqrt(1.0f - (η2Inv * η2Inv) * (1.0f - cosθi * cosθi))
	);
	// OPD = 2 n_2 d \cos\theta_2
	let opticalPathDifference: f32 = (
		2.0f * η2 * mat.thinFilmThickness * cosθt
	);

	// first interface
	var R12s: f32;
	var R12p: f32;
	var φ12s: f32;
	var φ12p: f32;
	bsdfFresnelDielectricPhase(
		/*cosθi=*/cosθi,
		/*cosθt=*/cosθt,
		/*ηi=*/1.0f,
		/*ηt=*/η2,
		/*Rp=*/R12p,
		/*Rs=*/R12s,
		/*φp=*/φ12p,
		/*φs=*/φ12s
	);

	// cosine of the transmission angle into the base medium, per channel
	let cosθBase: f32v3 = (
		sqrt(
			max(
				f32v3(0.0f),
				f32v3(1.0f) - (η2 / ηBase) * (η2 / ηBase)
				* (1.0f - cosθt * cosθt)
			)
		)
	);

	// second interface
	var R23s: f32v3;
	var R23p: f32v3;
	var φ23s: f32v3;
	var φ23p: f32v3;
	bsdfFresnelConductorPhase(
		/*cosθi=*/cosθt,
		/*cosθt=*/cosθBase,
		/*ηi=*/η2,
		/*ηt=*/ηBase,
		/*κt=*/κBase,
		/*Rp=*/R23p,
		/*Rs=*/R23s,
		/*φp=*/φ23p,
		/*φs=*/φ23s
	);

	// \phi_2 = \phi_{21} + \phi_{23} = (\pi - \phi_{12}) + \phi_{23}
	let φ2p: f32v3 = (π - φ12p) + φ23p;
	let φ2s: f32v3 = (π - φ12s) + φ23s;

	// R_{123} = R_{12} R_{23}; clamped away from 1 so 1 - R_{123} below
	// never divides by (near) zero at grazing incidence
	let R123sSqr: f32v3 = clamp(R12s * R23s, 1e-5f, 0.9999f);
	let R123pSqr: f32v3 = clamp(R12p * R23p, 1e-5f, 0.9999f);
	// \rho_{123} = \sqrt{R_{123}}
	let R123s: f32v3 = sqrt(R123sSqr);
	let R123p: f32v3 = sqrt(R123pSqr);

	// T_{12} = 1 - R_{12}
	let T12s: f32 = 1.0f - R12s;
	let T12p: f32 = 1.0f - R12p;
	// T_{12}^2
	let T121s: f32 = T12s * T12s;
	let T121p: f32 = T12p * T12p;

	// R = T_{12}^2 R_{23} / (1 - R_{123})
	let Rss: f32v3 = T121s * R23s / (1.0f - R123sSqr);
	let Rsp: f32v3 = T121p * R23p / (1.0f - R123pSqr);

	// reflectance for first interface, C_0 = R_{12} + R
	let C0s: f32v3 = R12s + Rss;
	let C0p: f32v3 = R12p + Rsp;
	let S0: f32v3 = bsdfEvaluateXyzSensitivity(0.0f, 0.0f);
	var I: f32v3 = 0.5f * (C0s * S0 + C0p * S0);

	// reflectance for interfaces m>0; the loop below folds in one more
	// \rho_{123} factor per order, giving C_m = (R - T_{12})\rho_{123}^m
	var Cms: f32v3 = Rss - T12s;
	var Cmp: f32v3 = Rsp - T12p;
	// \sum_{m=1}^{3}
	for (var m: i32 = 1; m <= 3; ++m) {
		// \rho_{123}^{m-1} -> \rho_{123}^m
		Cms *= R123s;
		Cmp *= R123p;
		// phase-modulated sensitivity for this order
		let Sms: f32v3 = (
			2.0f * bsdfEvaluateXyzSensitivityTinted(
				f32(m) * opticalPathDifference, f32(m) * φ2s
			)
		);
		let Smp: f32v3 = (
			2.0f * bsdfEvaluateXyzSensitivityTinted(
				f32(m) * opticalPathDifference, f32(m) * φ2p
			)
		);
		// accumulate the contribution of the m-th bounce; depolarizes light
		I += 0.5f * (Cms * Sms + Cmp * Smp);
	}

	// convert to rgb color space
	return clamp(mul(I, xyzToRgb), f32v3(0.0f), f32v3(1.0f));
}

/*
	airy reflectance for polarized light
	closed-form sum of the belcour & barla series
	geometric series in pe^{iΦ}
$$
	I = C_0 + 2 (R - T_{12})
		\frac{\rho_{123}\cos\Phi - \rho_{123}^2}
		{1 - 2\rho_{123}\cos\Phi + \rho_{123}^2},
	\quad \Phi = 2\pi\,\mathrm{OPD}/\lambda + \phi_2
$$
*/
func implThinfilmAiryPolarized(
	R12: f32,
	R23: f32,
	φ2: f32,
	δ: f32,
) -> f32 {
	// only the upper clamp: a lower one biases a vanishing film by ~0.006
	let R123Sqr: f32 = min(R12 * R23, 0.9999f);
	let R123: f32 = sqrt(R123Sqr);
	let T12: f32 = 1.0f - R12;
	// R = T_{12}^2 R_{23} / (1 - R_{123})
	let R: f32 = T12 * T12 * R23 / (1.0f - R123Sqr);
	let cosΦ: f32 = cos(δ + φ2);
	return (
		R12 + R
		+ 2.0f * (R - T12) * (R123 * cosΦ - R123Sqr)
		/ (1.0f - 2.0f * R123 * cosΦ + R123Sqr)
	);
}

// thin-film fresnel per wavelength; rgb keeps belcour & barla, since
// point-sampling three wavelengths aliases the interference
public func bsdfOpenPbrThinfilmFresnel(
	mat: MaterialBackend_OpenPbr.Params,
	cosθi: f32,
	ηBase: vkof::Spectrum,
	κBase: vkof::Spectrum,
	λ: Wavelengths,
) -> vkof::Spectrum {
#if defined(PONDER_SPECTRUM_RGB)
	return (
		vkof::Spectrum.fromRgb(
			bsdfOpenPbrThinfilmFresnel(
				mat, cosθi, ηBase.asRgb(λ), κBase.asRgb(λ)
			),
			λ
		)
	);
#else
	let η2: f32 = (
		lerp(1.0f, mat.thinFilmIor, smoothstep(0.0f, 0.03f, mat.thinFilmThickness))
	);
	let η2Inv: f32 = 1.0f / η2;
	let cosθt: f32 = (
		sqrt(1.0f - (η2Inv * η2Inv) * (1.0f - cosθi * cosθi))
	);
	// OPD = 2 n_2 d \cos\theta_2, in micrometers
	let opticalPathDifference: f32 = (
		2.0f * η2 * mat.thinFilmThickness * cosθt
	);

	// -- first interface, wavelength independent
	var R12s: f32;
	var R12p: f32;
	var φ12s: f32;
	var φ12p: f32;
	bsdfFresnelDielectricPhase(
		/*cosθi=*/cosθi,
		/*cosθt=*/cosθt,
		/*ηi=*/1.0f,
		/*ηt=*/η2,
		/*Rp=*/R12p,
		/*Rs=*/R12s,
		/*φp=*/φ12p,
		/*φs=*/φ12s
	);

	// -- second interface and interference, per wavelength
	var reflectance: vkof::Spectrum = vkof::Spectrum.zero();
	[ForceUnroll]
	for (u32 i = 0u; i < skWavelengthSet; ++i) {
		let ηB: f32 = ηBase.channel(i);
		let cosθBase: f32 = (
			sqrt(max(0.0f, 1.0f - (η2 / ηB) * (η2 / ηB) * (1.0f - cosθt * cosθt)))
		);
		var R23s: f32;
		var R23p: f32;
		var φ23s: f32;
		var φ23p: f32;
		bsdfFresnelConductorPhase(
			/*cosθi=*/cosθt,
			/*cosθt=*/cosθBase,
			/*ηi=*/η2,
			/*ηt=*/ηB,
			/*κt=*/κBase.channel(i),
			/*Rp=*/R23p,
			/*Rs=*/R23s,
			/*φp=*/φ23p,
			/*φs=*/φ23s
		);
		// round-trip phase, nm -> micrometers
		let δ: f32 = 2.0f * π * opticalPathDifference / (λ.lambdas[i] * 1e-3f);
		let Is: f32 = (
			implThinfilmAiryPolarized(R12s, R23s, (π - φ12s) + φ23s, δ)
		);
		let Ip: f32 = (
			implThinfilmAiryPolarized(R12p, R23p, (π - φ12p) + φ23p, δ)
		);
		reflectance.amounts[i] = clamp(0.5f * (Is + Ip), 0.0f, 1.0f);
	}
	return reflectance;
#endif
}

} // namespace vkof
