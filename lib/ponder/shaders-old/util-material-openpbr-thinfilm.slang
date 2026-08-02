module "util-material-openpbr-thinfilm";
__exported import "util-types";
__exported import "util-material-openpbr-shared";

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
public func utilFresnelDielectricPhase(
	cosThetaI: f32,
	cosThetaT: f32,
	etaI: f32,
	etaT: f32,
	out R_p: f32,
	out R_s: f32,
	out phi_p: f32,
	out phi_s: f32,
) {
	// \sin^2\theta_1
	const f32 sinThetaISq = 1.0f - cosThetaI * cosThetaI;
	// n_1 / n_2
	const f32 nr = etaI / etaT;

	// (n_1/n_2)^2 \sin^2\theta_1 > 1
	if (nr * nr * sinThetaISq > 1.0f) {
		R_p = 1.0f;
		R_s = 1.0f;
		phi_p = 0.0f;
		phi_s = 0.0f;
		return;
	}

	// r_s
	const f32 r_s = (
		// n_2 \cos\theta_1 - n_1 \cos\theta_2
		(etaT * cosThetaI - etaI * cosThetaT)
		// n_2 \cos\theta_1 + n_1 \cos\theta_2
		/ (etaT * cosThetaI + etaI * cosThetaT)
	);
	// r_p
	const f32 r_p = (
		// n_1 \cos\theta_1 - n_2 \cos\theta_2
		(etaI * cosThetaI - etaT * cosThetaT)
		// n_1 \cos\theta_1 + n_2 \cos\theta_2
		/ (etaI * cosThetaI + etaT * cosThetaT)
	);
	R_s = r_s * r_s;
	R_p = r_p * r_p;

	// -- phase shifts
	phi_s = (r_s < 0.0f) ? PI : 0.0f;
	phi_p = (r_p < 0.0f) ? PI : 0.0f;
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
public func utilFresnelConductorPhase(
	cosThetaI: f32,
	cosThetaT: f32,
	etaI: f32,
	etaT: f32,
	kappaT: f32,
	out R_p: f32,
	out R_s: f32,
	out phi_p: f32,
	out phi_s: f32,
) {
	// n_2 \cos\theta_2
	const f32 n2cosTheta2 = etaI * cosThetaI;
	// \sin\theta_2
	const f32 n2SinTheta2Sq = etaI * etaI - n2cosTheta2 * n2cosTheta2;
	// n_3
	const f32 n3 = etaT;
	// \kappa_3
	const f32 k3 = kappaT;

	if (k3 == 0.0f) {
		// if \kappa_3 = 0, then the next medium is a dielectric
		utilFresnelDielectricPhase(
			cosThetaI, cosThetaT, etaI, etaT, R_p, R_s, phi_p, phi_s
		);
		return;
	}

	// n_3^2(1-\kappa_3^2)
	const f32 n3SqRe = n3 * n3 * (1.0f - k3 * k3);
	// 2n_3^2\kappa_3
	const f32 n3SqIm = 2.0f * n3 * n3 * k3;
	// A = n_3^2(1- \kappa_3 ^ 2) - n_2^2\sin^2\theta_2
	const f32 A = (n3SqRe - n2SinTheta2Sq);
	// B = sqrt(A^2 + 4n_3^4\kappa_3^2)
	const f32 B = sqrt(A * A + n3SqIm * n3SqIm);
	// U = sqrt((A + B) / 2)
	const f32 U = sqrt((A + B) / 2.0f);
	// V = sqrt((B - A) / 2)
	const f32 V = sqrt((B - A) / 2.0f);

	// R_p
	R_p = (
		// (n_2 \cos\theta_2 - U)^2 + V^2
		((n2cosTheta2 - U) * (n2cosTheta2 - U) + V * V)
		// (n_2 \cos\theta_2 + U)^2 + V^2
		/ ((n2cosTheta2 + U) * (n2cosTheta2 + U) + V * V)
	);

	// phi_p
	phi_p = (
		// atan2(2n_2 V \cos\theta_2,\ U^2 + V^2 - n_2^2 \cos^2 \theta_2)
		atan2(2.0f * n2cosTheta2 * V, U * U + V * V - n2cosTheta2 * n2cosTheta2)
		// + \pi
		+ PI
	);

	// n_3^2(1-\kappa_3^2)\cos\theta_2 - n_2 U
	const f32 sReMinus = n3SqRe * cosThetaI - etaI * U;
	// n_3^2(1-\kappa_3^2)\cos\theta_2 + n_2 U
	const f32 sRePlus = n3SqRe * cosThetaI + etaI * U;
	// 2n_3^2\kappa_3\cos\theta_2 - n_2 V
	const f32 sImMinus = n3SqIm * cosThetaI - etaI * V;
	// 2n_3^2\kappa_3\cos\theta_2 + n_2 V
	const f32 sImPlus = n3SqIm * cosThetaI + etaI * V;

	// R_s
	R_s = (
		(sReMinus * sReMinus + sImMinus * sImMinus)
		/ (sRePlus * sRePlus + sImPlus * sImPlus)
	);

	// n_3^2(1+\kappa_3^2)\cos\theta_2
	const f32 n3SqAbsCos = n3 * n3 * (1.0f + k3 * k3) * cosThetaI;
	// phi_s
	phi_s = (
		atan2(
			2.0f * etaI * n3 * n3 * cosThetaI * (2.0f * k3 * U - (1.0f - k3 * k3) * V),
			n3SqAbsCos * n3SqAbsCos - etaI * etaI * (U * U + V * V)
		)
	);
}

// per-channel overload for tinted base substrate,
public func utilFresnelConductorPhase(
	cosThetaI: f32,
	cosThetaT: f32v3,
	etaI: f32,
	etaT: f32v3,
	kappaT: f32v3,
	out R_p: f32v3,
	out R_s: f32v3,
	out phi_p: f32v3,
	out phi_s: f32v3,
) {
	utilFresnelConductorPhase(
		cosThetaI, cosThetaT.r, etaI, etaT.r, kappaT.r,
		R_p.r, R_s.r, phi_p.r, phi_s.r
	);
	utilFresnelConductorPhase(
		cosThetaI, cosThetaT.g, etaI, etaT.g, kappaT.g,
		R_p.g, R_s.g, phi_p.g, phi_s.g
	);
	utilFresnelConductorPhase(
		cosThetaI, cosThetaT.b, etaI, etaT.b, kappaT.b,
		R_p.b, R_s.b, phi_p.b, phi_s.b
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
public func utilEvaluateXyzSensitivity(
	opd: f32,
	shift: f32,
) -> f32v3 {
	// gaussian fits
	const f32 phase = 2*PI * opd * 1.0e-6;
	const f32v3 val = f32v3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
	const f32v3 pos = f32v3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
	const f32v3 var = f32v3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
	f32v3 xyz = (
		val * sqrt(2.0f * PI * var)
		* cos(pos * phase + shift)
		* exp(-var * phase * phase)
	);
	xyz.x += (
		9.7470e-14f * sqrt(2.0f * PI * 4.5282e+09f)
		* cos(2.2399e+06f * phase + shift)
		* exp(-4.5282e+09f * phase * phase)
	);
	return xyz / 1.0685e-7f;
}

// per-channel version of the above for a tinted base: each channel only
// needs its own matching Gaussian (pos.x/y/z ~ red/green/blue)
public func utilEvaluateXyzSensitivityTinted(
	opd: f32,
	shift: f32v3,
) -> f32v3 {
	return f32v3(
		utilEvaluateXyzSensitivity(opd, shift.r).x,
		utilEvaluateXyzSensitivity(opd, shift.g).y,
		utilEvaluateXyzSensitivity(opd, shift.b).z
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
	&C_m = (R - T_{12}^2) \: \rho_{123}^{m}, \quad m \ge 1
	\end{align*}
$$

	unpolarized reflectance is the average of the two polarizations,
	$I = \tfrac{1}{2}(I_p + I_s)$, and the optical path difference for
	one round trip through the film of ior $n_2$ and thickness $d$ is
$$
	\mathrm{OPD} = 2 n_2 d \cos\theta_2
$$
*/
public func openPbrThinfilmFresnel(
	mat: OpenPbrMaterial,
	cosThetaI: f32,
	etaBase: f32v3,
	kappaBase: f32v3,
) -> f32v3 {
	// xyz to cie 1931 rgb color space, neutral white point
	const f32m3x3 xyzToRgb = f32m3x3(
		2.3706743f, -0.5138850f, 0.0052982f,
		-0.9000405f, 1.4253036f, -0.0146949f,
		-0.4706338f, 0.0885814f, 1.0093968f
	);

	const f32 eta2 = (
		lerp(1.0f, mat.thinFilmIor, smoothstep(0.0f, 0.03f, mat.thinFilmThickness))
	);
	const f32 eta2Inv = 1.0f / eta2;

	const f32 cosThetaT = (
		sqrt(1.0f - (eta2Inv * eta2Inv) * (1.0f - cosThetaI * cosThetaI))
	);
	// OPD = 2 n_2 d \cos\theta_2
	const f32 opticalPathDifference = (
		2.0f * eta2 * mat.thinFilmThickness * cosThetaT
	);

	// first interface
	f32 R12s;
	f32 R12p;
	f32 phi12s;
	f32 phi12p;
	utilFresnelDielectricPhase(
		/*cosThetaI=*/cosThetaI,
		/*cosThetaT=*/cosThetaT,
		/*etaI=*/1.0f,
		/*etaT=*/eta2,
		/*R_p=*/R12p,
		/*R_s=*/R12s,
		/*phi_p=*/phi12p,
		/*phi_s=*/phi12s
	);

	// cosine of the transmission angle into the base medium, per channel
	// (a tinted base has a different effective ior per channel); only
	// used by the \kappa_3 = 0 dielectric fallback inside
	// utilFresnelConductorPhase, which short-circuits before using it
	// whenever this would be NaN under total internal reflection
	const f32v3 cosThetaBase = (
		sqrt(
			max(
				f32v3(0.0f),
				f32v3(1.0f) - (eta2 / etaBase) * (eta2 / etaBase)
				* (1.0f - cosThetaT * cosThetaT)
			)
		)
	);

	// second interface
	f32v3 R23s;
	f32v3 R23p;
	f32v3 phi23s;
	f32v3 phi23p;
	utilFresnelConductorPhase(
		/*cosThetaI=*/cosThetaT,
		/*cosThetaT=*/cosThetaBase,
		/*etaI=*/eta2,
		/*etaT=*/etaBase,
		/*kappaT=*/kappaBase,
		/*R_p=*/R23p,
		/*R_s=*/R23s,
		/*phi_p=*/phi23p,
		/*phi_s=*/phi23s
	);

	// \phi_2 = \phi_{21} + \phi_{23} = (\pi - \phi_{12}) + \phi_{23}
	const f32v3 phi2p = (PI - phi12p) + phi23p;
	const f32v3 phi2s = (PI - phi12s) + phi23s;

	// R_{123} = R_{12} R_{23}; clamped away from 1 so 1 - R_{123} below
	// never divides by (near) zero at grazing incidence
	const f32v3 R123sSqr = clamp(R12s * R23s, 1e-5f, 0.9999f);
	const f32v3 R123pSqr = clamp(R12p * R23p, 1e-5f, 0.9999f);
	// \rho_{123} = \sqrt{R_{123}}
	const f32v3 R123s = sqrt(R123sSqr);
	const f32v3 R123p = sqrt(R123pSqr);

	// T_{12} = 1 - R_{12}
	const f32 T12s = 1.0f - R12s;
	const f32 T12p = 1.0f - R12p;
	// T_{12}^2
	const f32 T121s = T12s * T12s;
	const f32 T121p = T12p * T12p;

	// R = T_{12}^2 R_{23} / (1 - R_{123})
	const f32v3 Rss = T121s * R23s / (1.0f - R123sSqr);
	const f32v3 Rsp = T121p * R23p / (1.0f - R123pSqr);

	// reflectance for first interface, C_0 = R_{12} + R
	const f32v3 C0s = R12s + Rss;
	const f32v3 C0p = R12p + Rsp;
	const f32v3 S0 = utilEvaluateXyzSensitivity(0.0f, 0.0f);
	f32v3 I = 0.5f * (C0s * S0 + C0p * S0);

	// reflectance for interfaces m>0; the loop below folds in one more
	// \rho_{123} factor per order, giving C_m = (R - T_{12}^2)\rho_{123}^m
	f32v3 Cms = Rss - T121s;
	f32v3 Cmp = Rsp - T121p;
	// \sum_{m=1}^{3}
	for (i32 m = 1; m <= 3; ++m) {
		// \rho_{123}^{m-1} -> \rho_{123}^m
		Cms *= R123s;
		Cmp *= R123p;
		// phase-modulated sensitivity for this order
		const f32v3 Sms = (
			2.0f * utilEvaluateXyzSensitivityTinted(
				f32(m) * opticalPathDifference, f32(m) * phi2s
			)
		);
		const f32v3 Smp = (
			2.0f * utilEvaluateXyzSensitivityTinted(
				f32(m) * opticalPathDifference, f32(m) * phi2p
			)
		);
		// accumulate the contribution of the m-th bounce; depolarizes light
		I += 0.5f * (Cms * Sms + Cmp * Smp);
	}

	// convert to rgb color space
	return clamp(mul(I, xyzToRgb), f32v3(0.0f), f32v3(1.0f));
}

