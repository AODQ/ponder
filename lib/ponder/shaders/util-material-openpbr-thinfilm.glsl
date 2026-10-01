#ifndef UTIL_MATERIAL_OPENPBR_THINFILM_GLSL
#define UTIL_MATERIAL_OPENPBR_THINFILM_GLSL

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
// -----------------------------------------------------------------------------
// -- utilFresnelDielectricPhase
// -----------------------------------------------------------------------------

void utilFresnelDielectricPhase(
	const float cosThetaI,
	const float cosThetaT,
	const float etaI,
	const float etaT,
	out float R_p,
	out float R_s,
	out float phi_p,
	out float phi_s
) {
	// \sin^2\theta_1
	const float sinThetaISq = 1.0f - cosThetaI * cosThetaI;
	// n_1 / n_2
	const float nr = etaI / etaT;

	// (n_1/n_2)^2 \sin^2\theta_1 > 1
	if (nr * nr * sinThetaISq > 1.0f) {
		R_p = 1.0f;
		R_s = 1.0f;
		phi_p = 0.0f;
		phi_s = 0.0f;
		return;
	}

	// r_s
	const float r_s = (
		// n_2 \cos\theta_1 - n_1 \cos\theta_2
		(etaT * cosThetaI - etaI * cosThetaT)
		// n_2 \cos\theta_1 + n_1 \cos\theta_2
		/ (etaT * cosThetaI + etaI * cosThetaT)
	);
	// r_p
	const float r_p = (
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
// -----------------------------------------------------------------------------
// -- utilFresnelConductorPhase
// -----------------------------------------------------------------------------

void utilFresnelConductorPhase(
	const float cosThetaI, // n_2, current medium
	const float cosThetaT, // n_3, next medium
	const float etaI, // n_2, current medium
	const float etaT, // n_3, next medium
	const float kappaT, // \kappa_3, next medium
	out float R_p, // reflectance for p-polarized light
	out float R_s, // reflectance for s-polarized light
	out float phi_p, // phase shift for p-polarized light
	out float phi_s // phase shift for s-polarized light
) {
	// n_2 \cos\theta_2
	const float n2cosTheta2 = etaI * cosThetaI;
	// \sin\theta_2
	const float n2SinTheta2Sq = etaI * etaI - n2cosTheta2 * n2cosTheta2;
	// n_3
	const float n3 = etaT;
	// \kappa_3
	const float k3 = kappaT;

	if (k3 == 0.0f) {
		// if \kappa_3 = 0, then the next medium is a dielectric
		utilFresnelDielectricPhase(
			cosThetaI, cosThetaT, etaI, etaT, R_p, R_s, phi_p, phi_s
		);
		return;
	}

	// n_3^2(1-\kappa_3^2)
	const float n3SqRe = n3 * n3 * (1.0f - k3 * k3);
	// 2n_3^2\kappa_3
	const float n3SqIm = 2.0f * n3 * n3 * k3;
	// A = n_3^2(1- \kappa_3 ^ 2) - n_2^2\sin^2\theta_2
	const float A = (n3SqRe - n2SinTheta2Sq);
	// B = sqrt(A^2 + 4n_3^4\kappa_3^2)
	const float B = sqrt(A * A + n3SqIm * n3SqIm);
	// U = sqrt((A + B) / 2)
	const float U = sqrt((A + B) / 2.0f);
	// V = sqrt((B - A) / 2)
	const float V = sqrt((B - A) / 2.0f);

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
		atan(2.0f * n2cosTheta2 * V, U * U + V * V - n2cosTheta2 * n2cosTheta2)
		// + \pi
		+ PI
	);

	// n_3^2(1-\kappa_3^2)\cos\theta_2 - n_2 U
	const float sReMinus = n3SqRe * cosThetaI - etaI * U;
	// n_3^2(1-\kappa_3^2)\cos\theta_2 + n_2 U
	const float sRePlus = n3SqRe * cosThetaI + etaI * U;
	// 2n_3^2\kappa_3\cos\theta_2 - n_2 V
	const float sImMinus = n3SqIm * cosThetaI - etaI * V;
	// 2n_3^2\kappa_3\cos\theta_2 + n_2 V
	const float sImPlus = n3SqIm * cosThetaI + etaI * V;

	// R_s
	R_s = (
		(sReMinus * sReMinus + sImMinus * sImMinus)
		/ (sRePlus * sRePlus + sImPlus * sImPlus)
	);

	// n_3^2(1+\kappa_3^2)\cos\theta_2
	const float n3SqAbsCos = n3 * n3 * (1.0f + k3 * k3) * cosThetaI;
	// phi_s
	phi_s = (
		atan(
			2.0f * etaI * n3 * n3 * cosThetaI * (2.0f * k3 * U - (1.0f - k3 * k3) * V),
			n3SqAbsCos * n3SqAbsCos - etaI * etaI * (U * U + V * V)
		)
	);
}

// per-channel overload for tinted base substrate,
void utilFresnelConductorPhase(
	const float cosThetaI,
	const vec3 cosThetaT,
	const float etaI,
	const vec3 etaT,
	const vec3 kappaT,
	out vec3 R_p,
	out vec3 R_s,
	out vec3 phi_p,
	out vec3 phi_s
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
// -----------------------------------------------------------------------------
// -- utilEvaluateXyzSensitivity
// -----------------------------------------------------------------------------

vec3 utilEvaluateXyzSensitivity(const float opd, const float shift) {
	// gaussian fits
	const float phase = 2*PI * opd * 1.0e-6;
	const vec3 val = vec3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
	const vec3 pos = vec3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
	const vec3 var = vec3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
	vec3 xyz = (
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
vec3 utilEvaluateXyzSensitivityTinted(const float opd, const vec3 shift) {
	return vec3(
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
vec3 openPbrThinfilmFresnel(
	const OpenPbrMaterial mat,
	const float cosThetaI, // dot(h, \w_i)
	const vec3 etaBase, // n_3 per channel; splat one value for an achromatic base
	const vec3 kappaBase // \kappa_3 per channel; zero for a real (non-absorbing) base
) {
	// xyz to cie 1931 rgb color space, neutral white point
	const mat3 xyzToRgb = mat3(
		2.3706743f, -0.5138850f, 0.0052982f,
		-0.9000405f, 1.4253036f, -0.0146949f,
		-0.4706338f, 0.0885814f, 1.0093968f
	);

	const float eta2 = (
		mix(1.0f, mat.thinFilmIor, smoothstep(0.0f, 0.03f, mat.thinFilmThickness))
	);
	const float eta2Inv = 1.0f / eta2;

	const float cosThetaT = (
		sqrt(1.0f - (eta2Inv * eta2Inv) * (1.0f - cosThetaI * cosThetaI))
	);
	// OPD = 2 n_2 d \cos\theta_2
	const float opticalPathDifference = (
		2.0f * eta2 * mat.thinFilmThickness * cosThetaT
	);

	// first interface
	float R12s;
	float R12p;
	float phi12s;
	float phi12p;
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
	const vec3 cosThetaBase = (
		sqrt(
			max(
				vec3(0.0f),
				vec3(1.0f) - (eta2 / etaBase) * (eta2 / etaBase)
				* (1.0f - cosThetaT * cosThetaT)
			)
		)
	);

	// second interface
	vec3 R23s;
	vec3 R23p;
	vec3 phi23s;
	vec3 phi23p;
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
	const vec3 phi2p = (PI - phi12p) + phi23p;
	const vec3 phi2s = (PI - phi12s) + phi23s;

	// R_{123} = R_{12} R_{23}; clamped away from 1 so 1 - R_{123} below
	// never divides by (near) zero at grazing incidence
	const vec3 R123sSqr = clamp(R12s * R23s, 1e-5f, 0.9999f);
	const vec3 R123pSqr = clamp(R12p * R23p, 1e-5f, 0.9999f);
	// \rho_{123} = \sqrt{R_{123}}
	const vec3 R123s = sqrt(R123sSqr);
	const vec3 R123p = sqrt(R123pSqr);

	// T_{12} = 1 - R_{12}
	const float T12s = 1.0f - R12s;
	const float T12p = 1.0f - R12p;
	// T_{12}^2
	const float T121s = T12s * T12s;
	const float T121p = T12p * T12p;

	// R = T_{12}^2 R_{23} / (1 - R_{123})
	const vec3 Rss = T121s * R23s / (1.0f - R123sSqr);
	const vec3 Rsp = T121p * R23p / (1.0f - R123pSqr);

	// reflectance for first interface, C_0 = R_{12} + R
	const vec3 C0s = R12s + Rss;
	const vec3 C0p = R12p + Rsp;
	const vec3 S0 = utilEvaluateXyzSensitivity(0.0f, 0.0f);
	vec3 I = 0.5f * (C0s * S0 + C0p * S0);

	// reflectance for interfaces m>0; the loop below folds in one more
	// \rho_{123} factor per order, giving C_m = (R - T_{12}^2)\rho_{123}^m
	vec3 Cms = Rss - T121s;
	vec3 Cmp = Rsp - T121p;
	// \sum_{m=1}^{3}
	for (int m = 1; m <= 3; ++m) {
		// \rho_{123}^{m-1} -> \rho_{123}^m
		Cms *= R123s;
		Cmp *= R123p;
		// phase-modulated sensitivity for this order
		const vec3 Sms = (
			2.0f * utilEvaluateXyzSensitivityTinted(
				float(m) * opticalPathDifference, float(m) * phi2s
			)
		);
		const vec3 Smp = (
			2.0f * utilEvaluateXyzSensitivityTinted(
				float(m) * opticalPathDifference, float(m) * phi2p
			)
		);
		// accumulate the contribution of the m-th bounce; depolarizes light
		I += 0.5f * (Cms * Sms + Cmp * Smp);
	}

	// convert to rgb color space
	return clamp(xyzToRgb * I, vec3(0.0f), vec3(1.0f));
}

#endif // UTIL_MATERIAL_OPENPBR_THINFILM_GLSL
