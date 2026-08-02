module "util-material-openpbr-microfacet";
__exported import "util-types";
__exported import "util-shading-frame";
__exported import "util-material-openpbr-shared";
__exported import "util-material-openpbr-tables";
__exported import "util-roughness";
__exported import "util-energy-compensation";
__exported import "util-material-openpbr-thinfilm";


// -----------------------------------------------------------------------------
// -- microfacet half-vector helpers
// -----------------------------------------------------------------------------

// reconstructs the normalized half vector from \omega_i / \omega_o
public func utilHalfVectorLocal(
	wiLocal: f32v3,
	woLocal: f32v3,
	out hLocal: f32v3,
) -> bool {
	const f32v3 hLocalUnnorm = wiLocal + woLocal;
	const bool valid = dot(hLocalUnnorm, hLocalUnnorm) > 0.0f;
	hLocal = valid ? normalize(hLocalUnnorm) : f32v3(0.0f, 0.0f, 1.0f);
	return valid;
}

// -----------------------------------------------------------------------------
// -- microfacet Fresnel functions
// -----------------------------------------------------------------------------

// dielectric fresnel;
// eta is the specularIor for reflection and 1.0/specularIor for refraction
public func utilMicrofacetFresnelDielectric(
	cosTheta: f32,
	eta: f32,
) -> f32 {
	const f32 cosThetAlpha = cosTheta * cosTheta;
	const f32 sinThetAlpha = 1.0f - cosThetAlpha;
	const f32 etAlpha = eta * eta;
	const f32 sinThetaT2 = sinThetAlpha / etAlpha;
	if (sinThetaT2 > 1.0f) {
		return 1.0f; // total internal reflection
	}
	const f32 cosThetaT = sqrt(1.0f - sinThetaT2);
	const f32 rParallel = (
		(eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT)
	);
	const f32 rPerspective = (
		(cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT)
	);
	return 0.5f * (rParallel*rParallel + rPerspective*rPerspective);
}

// -----------------------------------------------------------------------------
// -- microfacet F_s functions
// -----------------------------------------------------------------------------

// ggx / trowbridge-reitz normal distribution function;
// trowbridge & reitz 1975, walter et al. 2007 eq 33,
// karis 2013 eq 3,
/*
	D(h) = \frac{\alpha^2}{\pi ((n \cdot h)^2 (\alpha^2 - 1) + 1)^2}
*/
public func utilMicrofacetGgxDistribution(
	dotNorH: f32,
	alpha: f32,
) -> f32 {
	const f32 alpha2 = alpha * alpha;
	// (n \cdot h)^2 (\alpha2 - 1) + 1
	const f32 d = max(dotNorH * dotNorH * (alpha2 - 1.0f) + 1.0f, 1e-5f);
	// \frac{\alpha2}{\pi d^2}
	return alpha2 / (PI * d * d);
}

// anisotropic ggx ndf in the local tangent frame (t, b, n), with mLocal the
// microfacet normal and alpha = (\alpha_x, \alpha_y)
/*
	\begin{align*}
	&D(h, \alpha_x, \alpha_y) =
		\frac{1}
		{
			\pi \alpha_x \alpha_y
			(\frac{h_x^2}{\alpha_x^2} + \frac{h_y^2}{\alpha_y^2} + h_z^2)^2
		}
		\tag{heitz14 eq 80}
	\\
	\end{align*}
*/
// isotropic \alpha_x = \alpha_y reduces to utilMicrofacetGgxDistribution:
public func utilMicrofacetGgxDistributionAniso(
	localH: f32v3,
	alpha: f32v2,
) -> f32 {
	const f32v2 alpha2 = alpha * alpha;
	// \frac{h_x^2}{A_x} + \frac{h_y^2}{A_y} + h_z^2
	const f32 d = (
		max(
			(
				localH.x * localH.x / alpha2.x
				+ localH.y * localH.y / alpha2.y
				+ localH.z * localH.z
			),
			1e-7f
		)
	);
	// 1 / ( \pi \sqrt{A_x A_y} d^2 )
	return 1.0f / (PI * alpha.x * alpha.y * d * d);
}

// heitz 2014 height-correlated smith visibility
/*
	\begin{align*}
	&\boxed {
		\textbf{height-correlated masking and shadowing}
	}
	\\
	&G_2(\omega_o, \omega_i, \omega_h) =
		\frac{
			\chi^+(\omega_o \cdot \omega_h)
			\chi^+(\omega_i \cdot \omega_h)
		} {
			1
			+ \Lambda(\omega_o)
			+ \Lambda(\omega_i)
		}
		\tag{heitz14 eq 99}
	\\
	&\Lambda(\omega) = \frac{-1 + \sqrt{1 + \frac{1}{\alpha_t^2}}} {2}
		\tag{heitz14 eq 72}
	\\
	&\chi^+(\omega) = \text{clamp}(\omega, 0, 1)\\
	&\alpha_t = \frac{1}{\alpha_o tan\theta_o}
		\tag{heitz14 eq 86}
	\\
	&\alpha_o = \sqrt{cos^2\phi_o \alpha_x^2 + sin^2\phi_o \alpha_y^2}
		\tag{heitz14 eq 80}\\
	\\
	&\phi_o = \arctan(\omega_y, \omega_x)\\
	\end{align*}
*/ /*
	\begin{align*}
	&\boxed{
		\textbf{expansion of the height-correlated smith visibility}\\
	}\\
	&\textbf{constants cancel;}~1 - 0.5 - 0.5 = 0\\
	&1 + \Lambda(\omega_o) + \Lambda(\omega_i) =
		\frac{1}{2} (
			\sqrt{1 + \frac{1}{a_t(\omega_i)^2}}
			+ \sqrt{1 + \frac{1}{a_t(\omega_o)^2}}
		)
	\\
	&1/a_t^2 = \alpha_o^2 tan^2\theta \\
	&\alpha_o^2 = cos^2\phi \alpha_x^2 + sin^2\phi \alpha_y^2 \\
	&tan^2\theta = \frac{sin^2\theta}{cos^2\theta} \\

	&\textbf{multiply and push}~sin^2\theta~{inside the parens}\\
	&\alpha_o^2 tan^2\theta =
		\frac{
			\alpha_x^2 (sin\theta cos\phi)^2 + \alpha_y^2 (sin\theta sin\phi)^2
		}{
			cos^2\theta
		}
	\\
	
	&\textbf{emplace directional components; applies typical tangent framing}\\
	&\alpha_o^2 tan^2\theta =
		\frac{
			\alpha_x^2 \omega_x^2 + \alpha_y^2 \omega_y^2
		} {
			\omega_z^2
		}
	\\
	&\textbf{rewrite}~\Lambda~{in the typical tangent frame}\\
	&\omega_x = sin\theta cos\phi \\
	&\omega_y = sin\theta sin\phi \\
	&\omega_z = cos\theta \\
	&\alpha_o^2 tan^2\theta =
	  (\alpha_x^2 \omega_x^2 + \alpha_y^2 \omega_y^2) / \omega_z^2
	\\
	&A_\omega = |(\alpha_x \omega_x, \alpha_y \omega_y, \omega_z)| \\
	&\sqrt{1 + \frac{1}{a_t(\omega)^2}} = \frac{A_\omega}{\omega_z} \\
	&\boxed {
		\Lambda(\omega) = \frac{-1 + \frac{A_\omega}{\omega_z}}{2}
	}\\
	\end{align*}
*/ /*
	\begin{align*}
	&\boxed {
		\textbf{final height-correlated smith visibility}
	}\\
	&V = \frac{G_2}{4 \mu_o \mu_i}\\

	&\textbf{expand}~G_2\\
	&V = \frac{ \frac{1}{1 + \Lambda_o + \Lambda_i} }{ 4 \mu_o \mu_i } \\

	&\textbf{apply}~(a/b)/c = a/(b * c)\\
	&V = \frac{1}{4 \mu_o \mu_i (1 + \Lambda(\omega_o) + \Lambda(\omega_i))} \\

	&\textbf{expand}~\Lambda\\
	&= \frac{1}{
		4 \mu_o \mu_i \cdot 0.5 (\frac{A_o}{\mu_o} + \frac{A_i}{\mu_i})
	}\\

	&\textbf{fold in}~4*0.5\\
	&= \frac{0.5}{\mu_o \mu_i (\frac{A_o}{\mu_o} + \frac{A_i}{\mu_i})} \\

	&\textbf{distribute}~\mu_o~\textbf{and}~\mu_i
		~\textbf{into the sum; each term cancels out}\\
	&\mu_o \mu_i \cdot A_o / \mu_o = \mu_i A_o\\
	&\mu_o \mu_i \cdot A_i / \mu_i = \mu_o A_i\\
	&\boxed{
		= \frac{0.5}{\mu_i A_o + \mu_o A_i}
	}
	\end{align*}
*/
// important note which has fucked me over before; as explained above this
// equation folds the \frac{1}{4 G_1(\omega_i) G_1(\omega_o)} term into
// the final distribution's denominator, so the final microfacet BSDF does not
// need (\omega_i \cdot \omega_h) (\omega_o \cdot \omega_h) in its denominator.
public func utilMicrofacetSmithGgxVisibilityAniso(
	wiLocal: f32v3,
	woLocal: f32v3,
	alpha: f32v2,
) -> f32 {
	if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f) {
		return 0.0f;
	}

	// \mu_i
	const f32 muI = wiLocal.z;
	// \mu_o
	const f32 muO = woLocal.z;

	// A_o
	const f32 A_o = length(f32v3(alpha.x * woLocal.x, alpha.y * woLocal.y, muO));
	// A_i
	const f32 A_i = length(f32v3(alpha.x * wiLocal.x, alpha.y * wiLocal.y, muI));

	// \frac{0.5}{\mu_i A_o + \mu_o A_i}
	return 0.5f / max(muI * A_o + muO * A_i, 1e-7f);
}

// isotropic height-correlated smith visibility; the \alpha_x = \alpha_y
// special case of utilMicrofacetSmithGgxVisibilityAniso for lobes without a
// tangent frame (A collapses to \sqrt{\mu^2 (1 - \alpha^2) + \alpha^2})
// alpha is the linear roughness
public func utilMicrofacetSmithGgxVisibility(
	dotNorWi: f32,
	dotNorWo: f32,
	alpha: f32,
) -> f32 {
	if (dotNorWi <= 0.0f || dotNorWo <= 0.0f) {
		return 0.0f;
	}
	const f32 alpha2 = alpha * alpha;
	// crossed cosines: each lambda's own 1/\mu cancels against the
	// 1/(4 \mu_i \mu_o) fold, leaving the other direction's cosine
	const f32 lambdaWi = (
		dotNorWo
		* sqrt(dotNorWi * dotNorWi * (1.0f - alpha2) + alpha2)
	);
	const f32 lambdaWo = (
		dotNorWi
		* sqrt(dotNorWo * dotNorWo * (1.0f - alpha2) + alpha2)
	);
	return 0.5f / max(lambdaWi + lambdaWo, 1e-7f);
}

// one-sided smith masking term, G_1 = \frac{1}{1 + \Lambda} with the same
// lambda as the visibility above;
// alpha is the linear roughness
public func utilMicrofacetSmithG1(
	dotNorWi: f32,
	alpha: f32,
) -> f32 {
	const f32 alpha2 = alpha * alpha;
	return (
		2.0f
		* dotNorWi
		/ (
			dotNorWi
			+ sqrt(alpha2 + (1.0f - alpha2) * dotNorWi * dotNorWi)
		)
	);
}

// anisotropic G_1 = \frac{2 \omega_z}{\omega_z + A_\omega}, with the same
// A_\omega = |(\alpha_x \omega_x, \alpha_y \omega_y, \omega_z)| derived for
// utilMicrofacetSmithGgxVisibilityAniso above
public func utilMicrofacetSmithG1Aniso(
	wLocal: f32v3,
	alpha: f32v2,
) -> f32 {
	if (wLocal.z <= 0.0f) {
		return 0.0f;
	}
	// A_\omega
	const f32 A = (
		length(f32v3(alpha.x * wLocal.x, alpha.y * wLocal.y, wLocal.z))
	);
	return 2.0f * wLocal.z / max(wLocal.z + A, 1e-7f);
}

// -----------------------------------------------------------------------------
// -- microfacet sampling functions
// -----------------------------------------------------------------------------


// heitz & d'Eon 2014 VNDF; weights each normal to reject backfacing microfacets
// ggx vndf sampling from heitz 2018, anisotropic VNDF
// eto & tokuyoshi 2023, bounded VNDF lobes to prevent below-horizon reflections
// It seems to still emit below-horizon reflections.
// scripts/bounded-vndf-bound-check.py measures the 'leaks'
// do not use for refraction sampling; instead use utilMicrofacetSampleGgxVndf,
// which does not use a bounded cap
/*
	\begin{align*}
	&\textbf{(1) target vndf distribution} \\
	&D_{\omega_i}(h) =
		\frac{
			G_1(\omega_i, h) \max(0, \omega_i \cdot h) D(h)
		}{
			\omega_i \cdot n
		}
		\tag{heitz 2018 eq 3}
	\\
	% everything below is constructed from target distribution

	&\textbf{(2) anisotropic GGX VNDF sampling} \\
	&\acute{i} = \text{normalize}(\alpha_x i_x, \; \alpha_y i_y, \; i_z)\\
	&\textbf{(3) spherical cap sampling} \\
	&\acute{h} \propto \acute{i} + \acute{o}
		\tag{dupuy & benyoub 2023}
	\\
	&\phi = 2 \pi \xi_1\\
	% linear map \xi_2 in [0, 1] to the spherical cap's z range [-b, 1];
	% check: \xi_2 = 0 \rightarrow z = 1, \; \xi_2 = 1 \rightarrow z = -b
	&\acute{o}_z = (1 - \xi_2)(1 + b) - b\\
	&\acute{o} = (
		\sqrt{1 - \acute{o}_z^2} \cos\phi,
		\sqrt{1 - \acute{o}_z^2} \sin\phi,
		\acute{o}_z
	)\\
	&\textbf{(4) bounded anisotropic VNDF sampling} \\
	&min_\phi \; \acute{o}_z(\tfrac{\pi}{2}, \phi) = -k \, \acute{i}_z
		\tag{eto & tokuyoshi 2023 eq 5}
	\\
	% note k and s use original ithe cap bound b uses \acute{i}_z
	&k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2}
		\tag{eto & tokuyoshi 2023 eq 5}
	\\
	&s = 1 + |i_{xy}| \tag{eto & tokuyoshi 2023 eq 5}\\
	% lowercase a, distinct from the roughnesses \alpha_x, \alpha_y
	&a = \text{saturate}(\min(\alpha_x, \alpha_y))
		\tag{eto & tokuyoshi 2023 eq 6}
	\\
	&b = \begin{cases}
		k \, \acute{i}_z & i_z > 0\\
		\acute{i}_z & i_z \le 0
	\end{cases}
	\\
	&\textbf{(5) unstretch and reflect}\\
	&h =
		\text{normalize}(
			\alpha_x \acute{h}_x, \; \alpha_y \acute{h}_y, \; \acute{h}_z
		)
	\\
	&\omega_o = 2 (i \cdot h) h - i
	\end{align*}
*/
public func utilMicrofacetSampleGgxBoundedWoAniso(
	tanX: f32v3,
	tanY: f32v3,
	nor: f32v3,
	wi: f32v3,
	alpha: f32v2,
	xi: f32v2,
) -> f32v3 {
	// -- step 1:  project wi into the local tangent frame

	// i = (i_x, i_y, i_z) = (wi \cdot t, wi \cdot b, wi \cdot n)
	// (unaccented: not yet stretched)
	const f32v3 i = (
		f32v3(dot(wi, tanX), dot(wi, tanY), dot(wi, nor))
	);

	// -- step 2:  stretch \acute{i}

	// \acute{i}
	const f32v3 iStd = (
		// \text{normalize}(\alpha_x i_x, \alpha_y i_y, i_z)
		normalize(f32v3(i.x * alpha.x, i.y * alpha.y, i.z))
	);

	// -- step 3: cap azimuth
	// \phi = 2 \pi \xi_1
	const f32 phi = TAU * xi.x;

	// -- step 4: cap bounds
	// a = \text{saturate}(\min(\alpha_x, \alpha_y))
	const f32 a = clamp(min(alpha.x, alpha.y), 0.0f, 1.0f);

	// s = 1 + |i_{xy}|; original i, not \acute{i}
	const f32 s = 1.0f + length(f32v2(i.x, i.y));

	const f32 a2 = a * a;
	const f32 s2 = s * s;

	// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2}; original i_z, not \acute{i}_z
	const f32 k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);

	// -- step 5

	// b = \begin{cases} ... \end{cases}
	const f32 b = i.z > 0.0f ? k * iStd.z : iStd.z;
	// \acute{o}_z = (1 - \xi_2)(1 + b) - b
	const f32 z = (1.0f - xi.y) * (1.0f + b) - b;
	// \sqrt{1 - \acute{o}_z^2}
	const f32 sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
	// \acute{o}
	const f32v3 oStd = (
		f32v3(
			// \sqrt{1 - \acute{o}_z^2} \cos\phi,
			sinTheta * cos(phi),
			// \sqrt{1 - \acute{o}_z^2} \sin\phi,
			sinTheta * sin(phi),
			// \acute{o}_z
			z
		)
	);

	// \acute{h} \propto \acute{i} + \acute{o};
	// normalization deferred
	const f32v3 hStd = iStd + oStd;
	// wi antiparallel to nor collapses the cap to its apex for every xi
	// (b = \acute{i}_z = -1, so \acute{o} = (0,0,1) unconditionally),
	// making hStd exactly zero and its normalize() a NaN. reflection lobes
	// are only ever sampled with wi already above the shading horizon, so
	// this is not known to be reachable in practice; falls back to the
	// macro normal, a finite and well-defined (if arbitrary) reflection
	if (dot(hStd, hStd) < 1e-12f) {
		return nor;
	}
	// unstretch \acute{h} to get the microfacet normal m
	const f32v3 h = (
		normalize(f32v3(hStd.x * alpha.x, hStd.y * alpha.y, hStd.z))
	);
	// reflect in local frame back to world
	const f32v3 o = 2.0f * dot(i, h) * h - i;
	return o.x*tanX + o.y*tanY + o.z*nor;
}

// isotropic wrapper over the aniso core with a frisvad frame (rotationally
// arbitrary, fine because \alpha_x = \alpha_y)
public func utilMicrofacetSampleGgxBoundedWo(
	nor: f32v3,
	wi: f32v3,
	roughness: f32,
	xi: f32v2,
) -> f32v3 {
	f32v3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	return (
		utilMicrofacetSampleGgxBoundedWoAniso(
			bitangent, binormal, nor, wi, f32v2(roughness), xi
		)
	);
}

// heitz & d'Eon 2014 VNDF; weights each normal to reject backfacing microfacets
// anisotropic form of the heitz 2018 reference listing; returns the microfacet
// normal h, not a direction. this is used for refraction sampling; reflection
// lobes use the bounded utilMicrofacetSampleGgxBoundedWoAniso sampler instead
// (its cap bound is derived for the reflection hemisphere only)
public func utilMicrofacetSampleGgxVndfAniso(
	tanX: f32v3,
	tanY: f32v3,
	N: f32v3,
	wi: f32v3,
	alpha: f32v2,
	xi: f32v2,
) -> f32v3 {
	// i = (wi \cdot t, wi \cdot b, wi \cdot n)
	const f32v3 wiLocal = (
		f32v3(dot(wi, tanX), dot(wi, tanY), dot(wi, N))
	);
	// \acute{i} = \text{normalize}(\alpha_x i_x, \; \alpha_y i_y, \; i_z)
	const f32v3 wiH = (
		normalize(f32v3(alpha.x * wiLocal.x, alpha.y * wiLocal.y, wiLocal.z))
	);
	// orthonormal basis around \acute{i};
	// T_1 = \text{normalize}((-\acute{i}_y, \acute{i}_x, 0)),
	// with a fallback for the degenerate \acute{i} = +z view
	const f32 lensq = wiH.x * wiH.x + wiH.y * wiH.y;
	const f32v3 T1 = (
		lensq > 1e-7f
		? f32v3(-wiH.y, wiH.x, 0.0f) * (1.0f / sqrt(lensq))
		: f32v3(1.0f, 0.0f, 0.0f)
	);
	// T_2 = \acute{i} * T_1
	const f32v3 T2 = cross(wiH, T1);
	// uniform polar disk sample: r = \sqrt{\xi_1}, \phi = 2\pi\xi_2
	const f32 r = sqrt(xi.x);
	const f32 phi = TAU * xi.y;
	const f32 t1 = r * cos(phi);
	f32 t2 = r * sin(phi);
	// the visible hemisphere projects onto the disk as a half-disk plus a
	// half-disk squashed by \acute{i}_z;
	// makes (t_1, t_2) uniform
	// s = \frac{1 + \acute{i}_z}{2},
	// t_2 \leftarrow (1 - s)\sqrt{1 - t_1^2} + s \, t_2
	const f32 s = 0.5f * (1.0f + wiH.z);
	t2 = (1.0f - s) * sqrt(max(1.0f - t1 * t1, 0.0f)) + s * t2;
	// lift onto the hemisphere along \acute{i};
	// \acute{h} = t_1 T_1 + t_2 T_2 + \sqrt{1 - t_1^2 - t_2^2} \, \acute{i}
	const f32v3 Nh = (
		t1 * T1 + t2 * T2
		+ sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wiH
	);
	// unstretch; \max(0, \acute{h}_z) is the \chi^+ guard;
	// h = \text{normalize}(
	//     \alpha_x \acute{h}_x, \; \alpha_y \acute{h}_y, \; \max(0, \acute{h}_z))
	const f32v3 Hlocal = (
		normalize(f32v3(alpha.x * Nh.x, alpha.y * Nh.y, max(0.0f, Nh.z)))
	);
	return Hlocal.x * tanX + Hlocal.y * tanY + Hlocal.z * N;
}

// isotropic wrapper over the aniso core with a frisvad frame (rotationally
// arbitrary, fine because \alpha_x = \alpha_y)
public func utilMicrofacetSampleGgxVndf(
	N: f32v3,
	wi: f32v3,
	alpha: f32,
	xi: f32v2,
) -> f32v3 {
	f32v3 binormal, bitangent;
	utilCalculateXy(N, binormal, bitangent);
	return (
		utilMicrofacetSampleGgxVndfAniso(
			bitangent, binormal, N, wi, f32v2(alpha), xi
		)
	);
}

// -----------------------------------------------------------------------------
// -- microfacet pdf functions
// -----------------------------------------------------------------------------

// pdf of utilMicrofacetSampleGgxBoundedWoAniso for the given reflected
// direction wo; must stay in lockstep with that sampler (same k, same
// frame) or sampler and pdf disagree in the chi-square test
/*
	\begin{align*}
	&\textbf{(1) vndf pdf of the microfacet normal} \\
	% pdf of the unbounded spherical-cap sampler; the sampler falls back to
	% the unbounded cap when i_z \le 0, so this pdf applies there
	&p(h) = \frac{
		2 D(h) \max(i \cdot h, 0)
	}{
		i_z + \sqrt{\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2 + i_z^2}
	}
	\tag{eto & tokuyoshi 2023 eq 7}\\
	&t = \sqrt{\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2 + i_z^2}\\
	&\textbf{(2) bounded pdf of the microfacet normal} \\
	% shrinking the cap shrinks the normalization area: i_z \to k i_z,
	% with k and a exactly as in the sampler (eq 5, eq 6)
	&p(h) = \frac{2 D(h) \max(i \cdot h, 0)}{k i_z + t}
	\tag{eto & tokuyoshi 2023 eq 8}\\
	&\textbf{(3) change of variable}~h \to \omega_o \\
	% reflection jacobian (walter 2007 eq 14); the \max(i \cdot h, 0)
	% cancels against the jacobian's (i \cdot h)
	&||\frac{dh}{d\omega_o}|| = \frac{1}{4 (i \cdot h)}\\
	&pdf(\omega_o) = p(h) \, ||\frac{dh}{d\omega_o}||
		= \frac{D(h)}{2 (k i_z + t)}\\
	&\textbf{(4) numerically stable form for}~i_z < 0 \\
	% multiply (1) by \frac{t - i_z}{t - i_z};
	% (i_z + t)(t - i_z) = t^2 - i_z^2 = \alpha_x^2 i_x^2 + \alpha_y^2 i_y^2,
	% avoiding cancellation in the denominator when t \approx -i_z
	&pdf(\omega_o) = \frac{
		D(h) (t - i_z)
	}{
		2 (\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2)
	}
	\end{align*}
*/
public func utilMicrofacetGgxBoundedReflectPdfAniso(
	tanX: f32v3,
	tanY: f32v3,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
	alpha: f32v2,
) -> f32 {
	// tolerance on the cap-cut tests, error scales with 1/\alpha
	const f32 kCapCutEpsilon = (
		1e-5f / clamp(min(alpha.x, alpha.y), 0.001f, 1.0f)
	);

	// same tangent frame as utilMicrofacetSampleGgxBoundedWoAniso
	const f32v3 i = (
		f32v3(dot(wi, tanX), dot(wi, tanY), dot(wi, nor))
	);
	const f32v3 o = (
		f32v3(dot(wo, tanX), dot(wo, tanY), dot(wo, nor))
	);
	f32v3 h;
	if (!utilHalfVectorLocal(i, o, h)) {
		return 0.0f;
	}
	// D(h)
	const f32 ndf = utilMicrofacetGgxDistributionAniso(h, alpha);
	// (\alpha_x i_x, \alpha_y i_y); its squared length is t^2 - i_z^2,
	// reused by the stable form in (4)
	const f32v2 ai = f32v2(i.x * alpha.x, i.y * alpha.y);
	const f32 len2 = dot(ai, ai);
	// t = \sqrt{\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2 + i_z^2}
	const f32 t = sqrt(len2 + i.z * i.z);
	if (i.z >= 0.0f) {
		// a = \text{saturate}(\min(\alpha_x, \alpha_y)) (eq 6)
		const f32 a = clamp(min(alpha.x, alpha.y), 0.0f, 1.0f);
		// s = 1 + |i_{xy}|; original i, not \acute{i}
		const f32 s = 1.0f + length(f32v2(i.x, i.y));
		const f32 a2 = a * a;
		const f32 s2 = s * s;
		// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2} (eq 5)
		const f32 k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);
		// below-horizon wo is only reachable while its std-space cap
		// coordinate stays above the eq 5 cut; outside the truncated cap's
		// image the sampler never emits, so the density there is zero.
		// above-horizon wo is always inside the image (the cut only removes
		// the band occluded at every azimuth), so no test is needed there
		if (o.z < 0.0f) {
			const f32v3 iStd = (
				normalize(f32v3(i.x * alpha.x, i.y * alpha.y, i.z))
			);
			// normals unstretch inversely to directions:
			// \acute{h} = \text{normalize}(h_x / \alpha_x, h_y / \alpha_y, h_z)
			const f32v3 hStd = (
				normalize(f32v3(h.x / alpha.x, h.y / alpha.y, h.z))
			);
			// \acute{o} = 2 (\acute{i} \cdot \acute{h}) \acute{h} - \acute{i}
			const f32v3 oStd = (
				2.0f * dot(iStd, hStd) * hStd - iStd
			);
			// cap lower bound: \acute{o}_z \ge -k \acute{i}_z
			if (oStd.z < -k * iStd.z - kCapCutEpsilon) {
				return 0.0f;
			}
		}
		// (3): pdf(\omega_o) = \frac{D(h)}{2 (k i_z + t)}
		return ndf / (2.0f * (k * i.z + t));
	}
	// the sampler's i_z < 0 fallback has no k factor (b = \acute{i}_z), so
	// its cap's reachable range is [-\acute{i}_z, 1]
	{
		const f32v3 iStd = (
			normalize(f32v3(i.x * alpha.x, i.y * alpha.y, i.z))
		);
		const f32v3 hStd = (
			normalize(f32v3(h.x / alpha.x, h.y / alpha.y, h.z))
		);
		const f32v3 oStd = (
			2.0f * dot(iStd, hStd) * hStd - iStd
		);
		// cap lower bound: \acute{o}_z \ge -\acute{i}_z (b = \acute{i}_z here)
		if (oStd.z < -iStd.z - kCapCutEpsilon) {
			return 0.0f;
		}
	}
	// (4): pdf(\omega_o) =
	//     \frac{D(h) (t - i_z)}{2 (\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2)}
	return ndf * (t - i.z) / (2.0f * max(len2, 1e-12f));
}

// isotropic wrapper over the aniso pdf with a frisvad frame
public func utilMicrofacetGgxBoundedReflectPdf(
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
	roughness: f32,
) -> f32 {
	f32v3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	return (
		utilMicrofacetGgxBoundedReflectPdfAniso(
			bitangent, binormal, nor, wi, wo, f32v2(roughness)
		)
	);
}

// rational quadratic fit to monte carla data for ggx directional albedo;
// from materialx
func utilMicrofacetGgxDirectionalAlbedo(
	dotNorWi: f32,
	alpha: f32,
	f0: f32,
	f90: f32,
) -> f32 {
	const f32 x = dotNorWi;
	const f32 y = alpha;
	const f32 x2 = x * x;
	const f32 y2 = y * y;
	const f32v4 r = (
		f32v4(0.1003, 0.9345, 1.0, 1.0)
		+ f32v4(-0.6303, -2.323, -1.765, 0.2281) * x
		+ f32v4(9.748, 2.229, 8.263, 15.94) * y
		+ f32v4(-2.038, -3.748, 11.53, -55.83) * x * y
		+ f32v4(29.34, 1.424, 28.96, 13.08) * x2
		+ f32v4(-8.245, -0.7684, -7.507, 41.26) * y2
		+ f32v4(-26.44, 1.436, -36.11, 54.9) * x2 * y
		+ f32v4(19.99, 0.2913, 15.86, 300.2) * x * y2
		+ f32v4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2
	);
	const f32v2 ab = clamp(r.xy/r.zw, 0.0f, 1.0f);
	return f0 * ab.x + f90 * ab.y;
}

// -----------------------------------------------------------------------------
// -- microfacet public interface
// -----------------------------------------------------------------------------

// returns the fractional energy to mix diffuse and specular
// uses analytic turquin 2019 energy compensation.
// The energy being compensated is energy lost due to multiple scattering on
// microfacet surface
/*
	\begin{align*}
	&\textbf{(1) compensation factor} \\
	% microfacet single-scatter discards paths that bounce more than once on
	% the microsurface; scale the lobe so the lost energy returns:
	&f = f_{ss} (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\tag{turquin 2019}\\
	% view-only dependence: drops \omega_i, mildly breaks reciprocity
	&\textbf{(2) average fresnel} \\
	% schlick fresnel averaged over the cosine-weighted hemisphere;
	% 2 \int_0^1 \mu (1 - \mu)^5 d\mu = 2 B(2, 6) = \frac{1}{21}
	&F_{avg} = 2 \int_0^1 F(\mu) \mu \, d\mu
		= F_0 + \frac{F_{90} - F_0}{21}
		\tag{kulla \& conty 2017}\\
	&\textbf{(3) single-scatter directional albedo} \\
	% E_{ss}(\mu, \alpha) with perfect fresnel (f0 = f90 = 1), from the
	% materialx rational fit in utilMicrofacetGgxDirectionalAlbedo
	\end{align*}
*/
func utilMicrofacetDielectricEnergyCompensate(
	mu: f32,
	roughness: f32,
	f0: f32,
) -> f32 {
	// (2) F_{avg} = F_0 + (F_{90} - F_0) / 21, with F_{90} = 1
	const f32 fAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	// (3) E_{ss}(\mu, \alpha)
	const f32 ess = (
		max(utilMicrofacetGgxDirectionalAlbedo(mu, roughness, 1.0f, 1.0f), 1e-4f)
	);
	// (1) multiplier for the single-scatter lobe
	return 1.0f + fAvg * (1.0f - ess) / ess;
}

// compensated directional albedo of a dielectric ggx interface: the
// fresnel-weighted single-scatter albedo times the turquin multiplier.
// this is the quantity for layer energy budgets and lobe selection
// probabilities
/*
	E(\mu) = (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\cdot E_{ss}^F(\mu, \alpha, f_0)
*/
func utilMicrofacetDielectricAlbedo(
	mu: f32,
	roughness: f32,
	f0: f32,
) -> f32 {
	// E_{ss}^F: single-scatter albedo with the actual fresnel
	const f32 essF = (
		utilMicrofacetGgxDirectionalAlbedo(mu, roughness, f0, 1.0f)
	);
	// clamped: consumers treat this as an energy fraction / probability
	return (
		clamp(
			utilMicrofacetDielectricEnergyCompensate(mu, roughness, f0) * essF,
			0.0f,
			1.0f
		)
	);
}

// the kulla conty energy table as reference
// not used anywhere, using turquin 2019 instead
func utilMicrofacetDielectricEnergyCompensationKullaConty(
	tables: MaterialTableHandles,
	mu: f32,
	roughness: f32,
) -> f32 {
	// mu axis: (i+0.5f)/16.0f (grazing->normal)
	// roughness axis: (i/16.0f)^2 (smooth->rough)
	const f32 ai = clamp(roughness * 15.0f, 0.0f, 14.999f);
	const f32 mi = clamp(mu*16.0f - 0.5f, 0.0f, 14.999f);
	const i32 ai0 = i32(ai);
	const i32 ai1 = min(ai0 + 1, 15);
	const i32 mi0 = i32(mi);
	const i32 mi1 = min(mi0 + 1, 15);
	const f32 ta = frac(ai);
	const f32 tm = frac(mi);
	return (
		lerp(
			lerp(
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], i32v2(mi0, ai0), 0
				).r,
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], i32v2(mi1, ai0), 0
				).r,
				tm
			),
			lerp(
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], i32v2(mi0, ai1), 0
				).r,
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], i32v2(mi1, ai1), 0
				).r,
				tm
			),
			ta
		)
	);
}

// openpbr spec, specular_roughness_anisotropy: stretches alpha = roughness
// into tangent/bitangent slopes;
/*
	\textbf{openpbr anisotropic roughness} \\
	\alpha_t = \alpha \sqrt{\frac{2}{1 + (1 - a)^2}}, \quad
	\alpha_b = (1 - a) \alpha_t
*/
func openPbrRoughnessAlpha(
	roughness: f32,
	anisotropy: f32,
) -> f32v2 {
	const f32 oneMinusA = 1.0f - anisotropy;
	const f32 alphaT = (
		roughness * sqrt(2.0f / (1.0f + oneMinusA * oneMinusA))
	);
	return max(f32v2(alphaT, oneMinusA * alphaT), f32v2(1e-5f));
}

/*
	\textbf{openpbr coat roughening} \\
	r'_B = \mathrm{lerp}\Bigl(
		r_B, \bigl(\mathrm{min}(1, r_B^4 + 2 r_C^4)\bigr)^{1/4}, C
	\Bigr) \tag{openpbr, eq 73}
*/
func openPbrCoatRoughenedRoughness(
	baseRoughness: f32,
	coatRoughness: f32,
	coatWeight: f32,
) -> f32 {
	const f32 baseRoughnessSq = baseRoughness * baseRoughness;
	const f32 coatRoughnessSq = coatRoughness * coatRoughness;
	const f32 baseRoughness4 = baseRoughnessSq * baseRoughnessSq;
	const f32 coatRoughness4 = coatRoughnessSq * coatRoughnessSq;
	const f32 roughenedRoughness = (
		pow(min(1.0f, baseRoughness4 + 2.0f * coatRoughness4), 0.25f)
	);
	return lerp(baseRoughness, roughenedRoughness, coatWeight);
}

// microfacet dielectric interface reflection, shared by the opaque base and
// the translucent base
public func openPbrDielectricSpecularEvaluateF(
	mat: OpenPbrMaterial,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
	etaRel: f32,
) -> f32v3 {
	/*
	   typical microfacet;
	   f_{dielectric} =
	     mf_f * mf_g * mf_v
	     / (4 * (\omega_wi \cdot n) * (\omega_wo \cdot n))
	   \\
	   where,
	     \eta = relative ior with the specular weight folded in
	*/
	const f32 roughenedRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	const f32v2 alpha = (
		openPbrRoughnessAlpha(
			roughenedRoughness, mat.specularRoughnessAnisotropy
		)
	);
	// gltf interop anisotropy rotation (not part of openpbr)
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	// local-frame directions for the anisotropic D and V
	const f32v3 wiLocal = (
		f32v3(
			dot(wi, specFrame.tanX),
			dot(wi, specFrame.tanY),
			dot(wi, specFrame.nor)
		)
	);
	const f32v3 woLocal = (
		f32v3(
			dot(wo, specFrame.tanX),
			dot(wo, specFrame.tanY),
			dot(wo, specFrame.nor)
		)
	);
	f32v3 hLocal;
	if (!utilHalfVectorLocal(wiLocal, woLocal, hLocal)) {
		return f32v3(0.0f);
	}
	const f32 dotHWi = dot(hLocal, wiLocal);

	// negative dotHWi is a back-facing microfacet
	if (dotHWi <= 0.0f) {
		return f32v3(0.0f);
	}

	// thin-film interference interface substitutes for dielectric fresnel
	// interface. metallic lobe gets its own tinted thin-film fresnel
	// elsewhere
	const f32v3 dielectricFresnel = (
		f32v3(utilMicrofacetFresnelDielectric(dotHWi, etaRel))
	);
	// gltf KHR_materials_iridescence: iridescenceFactor is a lerp sadly
	const f32v3 mfFresnel = (
		mat.thinFilmWeight > 0.0f
		? lerp(
			dielectricFresnel,
			openPbrThinfilmFresnel(mat, dotHWi, f32v3(etaRel), f32v3(0.0f)),
			mat.thinFilmWeight
		)
		: dielectricFresnel
	);
	const f32 mfDistribution = (
		utilMicrofacetGgxDistributionAniso(hLocal, alpha)
	);
	const f32 mfVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// turquin 2019 multiplicative energy compensation; f0 from the relative
	// ior, which is symmetric in \eta vs 1/\eta. the factor uses the view
	// cosine only. Isotropic so uses effective roughenss
	// \sqrt{\alpha_x \alpha_y}
	const f32 iorRatio = (etaRel - 1.0f) / (etaRel + 1.0f);
	const f32 f0 = iorRatio * iorRatio;
	const f32 energyCompensation = (
		utilMicrofacetDielectricEnergyCompensate(
			wiLocal.z, sqrt(alpha.x * alpha.y), f0
		)
	);
	// the 1/(4 (\omega_wi \cdot n) (\omega_wo \cdot n)) term is already
	// folded into the smith visibility term
	// TODO multiply by specularweight right?
	return mfFresnel * mfDistribution * mfVisibility * energyCompensation;
}

// (TODO REVIEW)
// openpbr spec, coat: the base dielectric is immersed in the coat, so its
// ior ratio is relative to the coat where the coat covers it
/*
	\eta_s = \mathrm{lerp}(n_b/n_a, n_b/n_c, C)
		\tag{openpbr spec, specular_ior_ratio}\\
	% kutz 2021: inverting the ratio when n_c > n_b kills the spurious
	% grazing tir ring a lobe-mixture model cannot otherwise avoid
	\eta_{bc} \rightarrow n_c/n_b \quad \mathrm{if} \; n_c > n_b
		\tag{openpbr spec, specular_ior_ratio_with_tir_fix}
*/
public func openPbrCoatAffectedIor(
	specularIor: f32,
	coatIor: f32,
	coatWeight: f32,
) -> f32 {
	if (coatWeight <= 0.0f) {
		return specularIor;
	}
	const f32 ratio = specularIor / max(coatIor, 1e-5f);
	const f32 tirFixed = ratio > 1.0f ? ratio : 1.0f / max(ratio, 1e-5f);
	return lerp(specularIor, tirFixed, coatWeight);
}
// (TODO REVIEW)

// specular weight scales f0 via an effective ior
public func openPbrEffectiveIor(
	ior: f32,
	weight: f32,
) -> f32 {
	const f32 r = (ior - 1.0f) / (ior + 1.0f);
	const f32 f0 = clamp(weight * r * r, 0.0f, 0.999f);
	// (TODO REVIEW)
	// \epsilon = \mathrm{sgn}(\eta_s - 1)\sqrt{\xi_s F_s}; without the sign a
	// sub-unity ratio (a base less dense than its surroundings) comes back
	// inverted
	const f32 s = sign(ior - 1.0f) * sqrt(f0);
	// (TODO REVIEW)
	return (1.0f + s) / (1.0f - s);
}

// (TODO REVIEW)
// the base dielectric interface's ior: coat ratio first, then specular
// weight's f0 modulation. every dielectric lobe of the base -- reflection,
// transmission and the albedos that layer over them -- must share this
public func openPbrSpecularEffectiveIor(
	mat: OpenPbrMaterial,
) -> f32 {
	return (
		openPbrEffectiveIor(
			openPbrCoatAffectedIor(
				mat.specularIor, mat.coatIor, mat.coatWeight
			),
			mat.specularWeight
		)
	);
}
// (TODO REVIEW)

// inverse of the schlick ior->f0 relationship above, per channel; used to
// recover an effective real ior from a tinted metal's f0 so thin-film
// interference has a base substrate to reflect off of, since the f82-tint
// conductor model only carries a schlick f0 tint, not real per-channel n/k
public func openPbrIorFromF0(
	f0: f32v3,
) -> f32v3 {
	const f32v3 s = sqrt(clamp(f0, 0.0f, 0.9999f));
	return (f32v3(1.0f) + s) / (f32v3(1.0f) - s);
}

