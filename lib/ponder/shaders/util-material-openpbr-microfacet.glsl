#ifndef UTIL_MATERIAL_OPENPBR_MICROFACET_GLSL
#define UTIL_MATERIAL_OPENPBR_MICROFACET_GLSL

#include "util-shading-frame.glsl"

// -----------------------------------------------------------------------------
// -- microfacet Fresnel functions
// -----------------------------------------------------------------------------

// dielectric fresnel;
// eta is the specularIor for reflection and 1.0/specularIor for refraction
float utilMicrofacetFresnelDielectric(const float cosTheta, const float eta) {
	const float cosThetAlpha = cosTheta * cosTheta;
	const float sinThetAlpha = 1.0f - cosThetAlpha;
	const float etAlpha = eta * eta;
	const float sinThetaT2 = sinThetAlpha / etAlpha;
	if (sinThetaT2 > 1.0f) {
		return 1.0f; // total internal reflection
	}
	const float cosThetaT = sqrt(1.0f - sinThetaT2);
	const float rParallel = (
		(eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT)
	);
	const float rPerspective = (
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
float utilMicrofacetGgxDistribution(const float dotNorH, const float alpha) {
	const float alpha2 = alpha * alpha;
	// (n \cdot h)^2 (\alpha2 - 1) + 1
	const float d = max(dotNorH * dotNorH * (alpha2 - 1.0f) + 1.0f, 1e-5f);
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
float utilMicrofacetGgxDistributionAniso(
	const vec3 localH,
	const vec2 alpha
) {
	const vec2 alpha2 = alpha * alpha;
	// \frac{h_x^2}{A_x} + \frac{h_y^2}{A_y} + h_z^2
	const float d = (
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
float utilMicrofacetSmithGgxVisibilityAniso(
	const vec3 wiLocal,
	const vec3 woLocal,
	const vec2 alpha
) {
	if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f) {
		return 0.0f;
	}

	// \mu_i
	const float muI = wiLocal.z;
	// \mu_o
	const float muO = woLocal.z;

	// A_o
	const float A_o = length(vec3(alpha.x * woLocal.x, alpha.y * woLocal.y, muO));
	// A_i
	const float A_i = length(vec3(alpha.x * wiLocal.x, alpha.y * wiLocal.y, muI));

	// \frac{0.5}{\mu_i A_o + \mu_o A_i}
	return 0.5f / max(muI * A_o + muO * A_i, 1e-7f);
}

// isotropic height-correlated smith visibility; the \alpha_x = \alpha_y
// special case of utilMicrofacetSmithGgxVisibilityAniso for lobes without a
// tangent frame (A collapses to \sqrt{\mu^2 (1 - \alpha^2) + \alpha^2})
// alpha is the ggx slope scale, roughness (squared internally)
float utilMicrofacetSmithGgxVisibility(
	const float dotNorWi,
	const float dotNorWo,
	const float alpha
) {
	if (dotNorWi <= 0.0f || dotNorWo <= 0.0f) {
		return 0.0f;
	}
	const float alpha2 = alpha * alpha;
	// crossed cosines: each lambda's own 1/\mu cancels against the
	// 1/(4 \mu_i \mu_o) fold, leaving the other direction's cosine
	const float lambdaWi = (
		dotNorWo
		* sqrt(dotNorWi * dotNorWi * (1.0f - alpha2) + alpha2)
	);
	const float lambdaWo = (
		dotNorWi
		* sqrt(dotNorWo * dotNorWo * (1.0f - alpha2) + alpha2)
	);
	return 0.5f / max(lambdaWi + lambdaWo, 1e-7f);
}

// one-sided smith masking term, G_1 = \frac{1}{1 + \Lambda} with the same
// lambda as the visibility above;
// alpha is the ggx slope scale, roughness (squared internally)
float utilMicrofacetSmithG1(const float dotNorWi, const float alpha) {
	const float alpha2 = alpha * alpha;
	return (
		2.0f
		* dotNorWi
		/ (
			dotNorWi
			+ sqrt(alpha2 + (1.0f - alpha2) * dotNorWi * dotNorWi)
		)
	);
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
vec3 utilMicrofacetSampleGgxBoundedWoAniso(
	const vec3 tanX,
	const vec3 tanY,
	const vec3 nor,
	const vec3 wi,
	const vec2 alpha,
	const vec2 xi
) {
	// -- step 1:  project wi into the local tangent frame

	// i = (i_x, i_y, i_z) = (wi \cdot t, wi \cdot b, wi \cdot n)
	// (unaccented: not yet stretched)
	const vec3 i = (
		vec3(dot(wi, tanX), dot(wi, tanY), dot(wi, nor))
	);

	// -- step 2:  stretch \acute{i}

	// \acute{i}
	const vec3 iStd = (
		// \text{normalize}(\alpha_x i_x, \alpha_y i_y, i_z)
		normalize(vec3(i.x * alpha.x, i.y * alpha.y, i.z))
	);

	// -- step 3: cap azimuth
	// \phi = 2 \pi \xi_1
	const float phi = TAU * xi.x;

	// -- step 4: cap bounds
	// a = \text{saturate}(\min(\alpha_x, \alpha_y))
	const float a = clamp(min(alpha.x, alpha.y), 0.0f, 1.0f);

	// s = 1 + |i_{xy}|; original i, not \acute{i}
	const float s = 1.0f + length(vec2(i.x, i.y));

	const float a2 = a * a;
	const float s2 = s * s;

	// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2}; original i_z, not \acute{i}_z
	const float k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);

	// -- step 5

	// b = \begin{cases} ... \end{cases}
	const float b = i.z > 0.0f ? k * iStd.z : iStd.z;
	// \acute{o}_z = (1 - \xi_2)(1 + b) - b
	const float z = (1.0f - xi.y) * (1.0f + b) - b;
	// \sqrt{1 - \acute{o}_z^2}
	const float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
	// \acute{o}
	const vec3 oStd = (
		vec3(
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
	const vec3 hStd = iStd + oStd;
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
	const vec3 h = (
		normalize(vec3(hStd.x * alpha.x, hStd.y * alpha.y, hStd.z))
	);
	// reflect in local frame back to world
	const vec3 o = 2.0f * dot(i, h) * h - i;
	return o.x*tanX + o.y*tanY + o.z*nor;
}

// isotropic wrapper over the aniso core with a frisvad frame (rotationally
// arbitrary, fine because \alpha_x = \alpha_y)
vec3 utilMicrofacetSampleGgxBoundedWo(
	const vec3 nor,
	const vec3 wi,
	const float roughness,
	const vec2 xi
) {
	vec3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	return (
		utilMicrofacetSampleGgxBoundedWoAniso(
			bitangent, binormal, nor, wi, vec2(roughness), xi
		)
	);
}

// heitz & d'Eon 2014 VNDF; weights each normal to reject backfacing microfacets
// isotropic specialization (\alpha_x = \alpha_y = \alpha) of the heitz 2018
// reference listing; returns the microfacet normal h, not a direction.
// this is used for refraction sampling; reflection lobes use
// the bounded utilMicrofacetSampleGgxBoundedWoAniso sampler instead
vec3 utilMicrofacetSampleGgxVndf(
	const vec3 N,
	const vec3 wi,
	const float alpha,
	const vec2 xi
) {
	vec3 binormal, bitangent;
	utilCalculateXy(N, binormal, bitangent);
	// i = (wi \cdot t, wi \cdot b, wi \cdot n)
	const vec3 wiLocal = (
		vec3(dot(wi, bitangent), dot(wi, binormal), dot(wi, N))
	);
	// \acute{i} = \text{normalize}(\alpha i_x, \; \alpha i_y, \; i_z)
	const vec3 wiH = (
		normalize(vec3(alpha * wiLocal.x, alpha * wiLocal.y, wiLocal.z))
	);
	// orthonormal basis around \acute{i};
	// T_1 = \text{normalize}((-\acute{i}_y, \acute{i}_x, 0)),
	// with a fallback for the degenerate \acute{i} = +z view
	const float lensq = wiH.x * wiH.x + wiH.y * wiH.y;
	const vec3 T1 = (
		lensq > 1e-7f
		? vec3(-wiH.y, wiH.x, 0.0f) * (1.0f / sqrt(lensq))
		: vec3(1.0f, 0.0f, 0.0f)
	);
	// T_2 = \acute{i} * T_1
	const vec3 T2 = cross(wiH, T1);
	// uniform polar disk sample: r = \sqrt{\xi_1}, \phi = 2\pi\xi_2
	const float r = sqrt(xi.x);
	const float phi = TAU * xi.y;
	const float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	// the visible hemisphere projects onto the disk as a half-disk plus a
	// half-disk squashed by \acute{i}_z;
	// makes (t_1, t_2) uniform
	// s = \frac{1 + \acute{i}_z}{2},
	// t_2 \leftarrow (1 - s)\sqrt{1 - t_1^2} + s \, t_2
	const float s = 0.5f * (1.0f + wiH.z);
	t2 = (1.0f - s) * sqrt(max(1.0f - t1 * t1, 0.0f)) + s * t2;
	// lift onto the hemisphere along \acute{i};
	// \acute{h} = t_1 T_1 + t_2 T_2 + \sqrt{1 - t_1^2 - t_2^2} \, \acute{i}
	const vec3 Nh = (
		t1 * T1 + t2 * T2
		+ sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wiH
	);
	// unstretch; \max(0, \acute{h}_z) is the \chi^+ guard;
	// h = \text{normalize}(
	//     \alpha \acute{h}_x, \; \alpha \acute{h}_y, \; \max(0, \acute{h}_z))
	const vec3 Hlocal = (
		normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)))
	);
	return Hlocal.x * bitangent + Hlocal.y * binormal + Hlocal.z * N;
}

// -----------------------------------------------------------------------------
// -- microfacet pdf functions
// -----------------------------------------------------------------------------

// pdf of utilMicrofacetSampleGgxBoundedWoAniso for the given reflected
// direction wo; must stay in lockstep with that sampler (same k, same
// frame) or sampler and pdf disagree in the chi-square harness
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
float utilMicrofacetGgxBoundedReflectPdfAniso(
	const vec3 tanX,
	const vec3 tanY,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const vec2 alpha
) {
	// tolerance on the reconstructed std-space cap-cut tests below: the
	// forward sampler and this backward reconstruction reach the same
	// \acute{o} via different floating-point paths (an unstretch/
	// re-stretch round trip through division by alpha), which can round
	// differently right at the cut and otherwise return pdf = 0 for a
	// direction the sampler just emitted -- a division-by-zero hazard for
	// mis weights. the round trip's error scales with 1 / alpha, so the
	// tolerance does too; clamped so it stays a negligible bias at
	// ordinary roughness and only grows bounded at the alpha floor, where
	// the lobe is a near-delta anyway and not expected to resolve exactly
	const float kCapCutEpsilon = (
		1e-5f / clamp(min(alpha.x, alpha.y), 0.001f, 1.0f)
	);

	// same tangent frame as utilMicrofacetSampleGgxBoundedWoAniso
	const vec3 i = (
		vec3(dot(wi, tanX), dot(wi, tanY), dot(wi, nor))
	);
	const vec3 o = (
		vec3(dot(wo, tanX), dot(wo, tanY), dot(wo, nor))
	);
	// reconstruct the half vector; for a sampled pair, i + o = 2 dot(i,h) h
	// exactly (the reflection identity, frame-independent), so normalize()
	// recovers h exactly for any dot(i,h) != 0, however small -- normalize()
	// only fails on the literal zero vector (dot(v,v) == 0.0), there is no
	// fuzzy near-zero precision-loss band around it. i + o == 0 exactly
	// means wo is an exact retroreflection (o == -i): a genuinely
	// unreconstructable direction (any h with dot(i,h) == 0 maps here, not
	// just one), the same convention pbrt's MicrofacetReflection uses
	// (`if (wh == Vector3f(0)) return 0;` before normalizing) -- zero
	// density, not a substitute direction. a substitute direction was tried
	// first and reverted: it fed a specific-but-arbitrary h into the same
	// cap-cut reconstruction below, which for the i.z < 0 branch could
	// disagree with whatever this configuration's actual (possibly also
	// degenerate) sampler behavior was, trading one mismatch for another
	const vec3 hUnnorm = i + o;
	if (dot(hUnnorm, hUnnorm) == 0.0f) {
		return 0.0f;
	}
	const vec3 h = normalize(hUnnorm);
	// D(h)
	const float ndf = utilMicrofacetGgxDistributionAniso(h, alpha);
	// (\alpha_x i_x, \alpha_y i_y); its squared length is t^2 - i_z^2,
	// reused by the stable form in (4)
	const vec2 ai = vec2(i.x * alpha.x, i.y * alpha.y);
	const float len2 = dot(ai, ai);
	// t = \sqrt{\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2 + i_z^2}
	const float t = sqrt(len2 + i.z * i.z);
	if (i.z >= 0.0f) {
		// a = \text{saturate}(\min(\alpha_x, \alpha_y)) (eq 6)
		const float a = clamp(min(alpha.x, alpha.y), 0.0f, 1.0f);
		// s = 1 + |i_{xy}|; original i, not \acute{i}
		const float s = 1.0f + length(vec2(i.x, i.y));
		const float a2 = a * a;
		const float s2 = s * s;
		// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2} (eq 5)
		const float k = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);
		// below-horizon wo is only reachable while its std-space cap
		// coordinate stays above the eq 5 cut; outside the truncated cap's
		// image the sampler never emits, so the density there is zero.
		// above-horizon wo is always inside the image (the cut only removes
		// the band occluded at every azimuth), so no test is needed there
		if (o.z < 0.0f) {
			const vec3 iStd = (
				normalize(vec3(i.x * alpha.x, i.y * alpha.y, i.z))
			);
			// normals unstretch inversely to directions:
			// \acute{h} = \text{normalize}(h_x / \alpha_x, h_y / \alpha_y, h_z)
			const vec3 hStd = (
				normalize(vec3(h.x / alpha.x, h.y / alpha.y, h.z))
			);
			// \acute{o} = 2 (\acute{i} \cdot \acute{h}) \acute{h} - \acute{i}
			const vec3 oStd = (
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
	// its cap's reachable range is [-\acute{i}_z, 1] rather than the
	// k-shrunk [-k \acute{i}_z, 1] used above. unlike the i_z >= 0 case, no
	// argument establishes that an above-horizon wo is always inside that
	// image here, so the reconstruction and cut test always run
	{
		const vec3 iStd = (
			normalize(vec3(i.x * alpha.x, i.y * alpha.y, i.z))
		);
		const vec3 hStd = (
			normalize(vec3(h.x / alpha.x, h.y / alpha.y, h.z))
		);
		const vec3 oStd = (
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
float utilMicrofacetGgxBoundedReflectPdf(
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const float roughness
) {
	vec3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	return (
		utilMicrofacetGgxBoundedReflectPdfAniso(
			bitangent, binormal, nor, wi, wo, vec2(roughness)
		)
	);
}

// rational quadratic fit to monte carla data for ggx directional albedo;
// from materialx
float utilMicrofacetGgxDirectionalAlbedo(
	const float dotNorWi,
	const float alpha,
	const float f0,
	const float f90
) {
	const float x = dotNorWi;
	const float y = alpha;
	const float x2 = x * x;
	const float y2 = y * y;
	const vec4 r = (
		vec4(0.1003, 0.9345, 1.0, 1.0)
		+ vec4(-0.6303, -2.323, -1.765, 0.2281) * x
		+ vec4(9.748, 2.229, 8.263, 15.94) * y
		+ vec4(-2.038, -3.748, 11.53, -55.83) * x * y
		+ vec4(29.34, 1.424, 28.96, 13.08) * x2
		+ vec4(-8.245, -0.7684, -7.507, 41.26) * y2
		+ vec4(-26.44, 1.436, -36.11, 54.9) * x2 * y
		+ vec4(19.99, 0.2913, 15.86, 300.2) * x * y2
		+ vec4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2
	);
	const vec2 ab = clamp(r.xy/r.zw, 0.0f, 1.0f);
	return f0 * ab.x + f90 * ab.y;
}

// -----------------------------------------------------------------------------
// -- microfacet public interface
// -----------------------------------------------------------------------------

// returns the fractional energy to mix diffuse and specular
// uses analytic turquin 2019 energy compensation instead of kulla conty tables
// energy compensation factor for the single-scatter ggx dielectric lobe 
// (kulla & conty 2017 compensate additively with tables
// instead; see utilMicrofacetDielectricEnergyCompensationKullaConty)
// the energy being compensated is energy lost due to multiple scattering on
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
float utilMicrofacetDielectricEnergyCompensate(
	const float mu,
	const float roughness,
	const float f0
) {
	// (2) F_{avg} = F_0 + (F_{90} - F_0) / 21, with F_{90} = 1
	const float fAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	// (3) E_{ss}(\mu, \alpha), perfect fresnel; the materialx rational fit
	// is only trained inside its (mu, alpha) domain and its own internal
	// clamp(..., 0, 1) legitimately lands on exactly 0 at grazing mu / high
	// roughness corners outside that domain, so this is floored away from
	// 0 -- otherwise (1 - ess) / ess is a live division by zero (+inf),
	// which every caller (dielectric specular, coat, conductor) later
	// multiplies against a D * V that also vanishes at those same grazing
	// angles, and 0 * inf = NaN poisons the whole lobe
	const float ess = (
		max(utilMicrofacetGgxDirectionalAlbedo(mu, roughness, 1.0f, 1.0f), 1e-4f)
	);
	// (1) multiplier for the single-scatter lobe
	return 1.0f + fAvg * (1.0f - ess) / ess;
}

// compensated directional albedo of a dielectric ggx interface: the
// fresnel-weighted single-scatter albedo times the turquin multiplier.
// this is the quantity for layer energy budgets and lobe selection
// probabilities; the bare multiplier scales the lobe itself in
// openPbrDielectricSpecularEvaluateF
/*
	E(\mu) = (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\cdot E_{ss}^F(\mu, \alpha, f_0)
*/
float utilMicrofacetDielectricAlbedo(
	const float mu,
	const float roughness,
	const float f0
) {
	// E_{ss}^F: single-scatter albedo with the actual fresnel
	const float essF = (
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
float utilMicrofacetDielectricEnergyCompensationKullaConty(
	const MaterialTableHandles tables,
	const float mu,
	const float roughness
) {
	// mu axis: (i+0.5f)/16.0f (grazing->normal)
	// roughness axis: (i/16.0f)^2 (smooth->rough)
	const float ai = clamp(roughness * 15.0f, 0.0f, 14.999f);
	const float mi = clamp(mu*16.0f - 0.5f, 0.0f, 14.999f);
	const int ai0 = int(ai);
	const int ai1 = min(ai0 + 1, 15);
	const int mi0 = int(mi);
	const int mi1 = min(mi0 + 1, 15);
	const float ta = fract(ai);
	const float tm = fract(mi);
	return (
		mix(
			mix(
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], ivec2(mi0, ai0), 0
				).r,
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], ivec2(mi1, ai0), 0
				).r,
				tm
			),
			mix(
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], ivec2(mi0, ai1), 0
				).r,
				texelFetch(
					vkofTextures[tables.kullaContyEnergy], ivec2(mi1, ai1), 0
				).r,
				tm
			),
			ta
		)
	);
}

// openpbr spec, specular_roughness_anisotropy: stretches alpha = roughness
// (engine convention: no perceptual remap) into tangent/bitangent slopes;
// verify against the spec section before trusting the exact constants
/*
	\alpha_t = \alpha \sqrt{\frac{2}{1 + (1 - a)^2}}, \quad
	\alpha_b = (1 - a) \alpha_t
*/
// a = 0 gives (\alpha, \alpha); a = 1 gives (\alpha \sqrt{2}, 0), floored
// so the degenerate axis stays sampleable
vec2 openPbrRoughnessAlpha(const float roughness, const float anisotropy) {
	const float oneMinusA = 1.0f - anisotropy;
	const float alphaT = (
		roughness * sqrt(2.0f / (1.0f + oneMinusA * oneMinusA))
	);
	return max(vec2(alphaT, oneMinusA * alphaT), vec2(1e-5f));
}

// openpbr spec, coat roughening: a rough coat effectively roughens the
// base substrate's microfacet lobes (metal and dielectric), modeled as
// the convolution of slope-space gaussians, with the coat variance
// counted twice since the reflection crosses the coat boundary twice
/*
	r'_B = \mathrm{lerp}\Bigl(
		r_B, \bigl(\mathrm{min}(1, r_B^4 + 2 r_C^4)\bigr)^{1/4}, C
	\Bigr)
*/
float openPbrCoatRoughenedRoughness(
	const float baseRoughness,
	const float coatRoughness,
	const float coatWeight
) {
	const float baseRoughnessSq = baseRoughness * baseRoughness;
	const float coatRoughnessSq = coatRoughness * coatRoughness;
	const float baseRoughness4 = baseRoughnessSq * baseRoughnessSq;
	const float coatRoughness4 = coatRoughnessSq * coatRoughnessSq;
	const float roughenedRoughness = (
		pow(min(1.0f, baseRoughness4 + 2.0f * coatRoughness4), 0.25f)
	);
	return mix(baseRoughness, roughenedRoughness, coatWeight);
}

// microfacet dielectric interface reflection, shared by the opaque base and
// the translucent base
vec3 openPbrDielectricSpecularEvaluateF(
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const vec3 wi,
	const vec3 wo,
	const float etaRel
) {
	/*
	   typical microfacet;
	   f_{dielectric} =
	     mf_f * mf_g * mf_v
	     / (4 * (\omega_wi \cdot n) * (\omega_wo \cdot n))
	   \\
	   where,
	     \eta = relative ior with the specular weight folded in
	*/
	// per-axis slopes from the openpbr anisotropy mapping; the base
	// roughness is widened by a rough coat (openpbr spec, coat roughening)
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
	// local-frame directions for the anisotropic D and V
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
	// wo landing exactly at -wi (a retroreflection) makes wiLocal + woLocal
	// the zero vector, whose normalize() is a NaN that then poisons every
	// term below -- same degenerate config as
	// openPbrTransmissionTirReflectPdf's hUnnorm guard, reachable here too
	// (found via the furnace model-render test's DragonAttenuation probe,
	// right after the TIR-fallback bounces that function's pdf covers).
	// zero reflectance for the exact zero vector, the same pbrt convention
	// (`if (wh == Vector3f(0)) return 0;`) the matching pdf guard above
	// uses -- a substitute direction was tried and reverted, see that
	// guard's comment
	const vec3 hLocalUnnorm = wiLocal + woLocal;
	if (dot(hLocalUnnorm, hLocalUnnorm) == 0.0f) {
		return vec3(0.0f);
	}
	const vec3 hLocal = normalize(hLocalUnnorm);
	// dot(h, wi) is frame-invariant
	const float dotHWi = dot(hLocal, wiLocal);
	// a non-positive dotHWi is a back-facing (physically invalid) microfacet
	// for this wi -- zero reflectance, the same guard
	// openPbrTransmissionTirReflectPdf already applies to its dotWiH.
	// missing this let a degenerate hLocal reach utilMicrofacetFresnelDielectric
	// with cosTheta <= 0; at eta == 1.0 exactly (a KHR_materials_dispersion
	// grid's matched-ior column) that function's rParallel/rPerspective
	// denominators (eta*cosTheta + cosThetaT, and its mirror) are exactly
	// eta*cosTheta + |cosTheta| = 0 for any cosTheta < 0, an exact
	// division by zero -> Inf that later multiplies against a zero
	// elsewhere in the term product and comes out NaN (found via the
	// furnace model-render test's DispersionTest probe)
	if (dotHWi <= 0.0f) {
		return vec3(0.0f);
	}
	// thin-film interference substitutes for the plain dielectric Fresnel
	// term at this same interface; the substrate beneath the film (n_3) is
	// this interface's own relative ior (achromatic, splatted); the
	// metallic (fConductor) lobe gets its own tinted thin-film call in
	// openPbrFresnelMetallicEvaluateF
	const vec3 dielectricFresnel = (
		vec3(utilMicrofacetFresnelDielectric(dotHWi, etaRel))
	);
	// gltf KHR_materials_iridescence: iridescenceFactor (and its optional
	// texture mask) lerps between the plain and iridescent fresnel, it is
	// not an on/off switch
	const vec3 mfFresnel = (
		mat.thinFilmWeight > 0.0f
		? mix(
			dielectricFresnel,
			openPbrThinfilmFresnel(mat, dotHWi, vec3(etaRel), vec3(0.0f)),
			mat.thinFilmWeight
		)
		: dielectricFresnel
	);
	const float mfDistribution = (
		utilMicrofacetGgxDistributionAniso(hLocal, alpha)
	);
	const float mfVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// turquin 2019 multiplicative energy compensation; f0 from the relative
	// ior, which is symmetric in \eta vs 1/\eta. the factor uses the view
	// cosine only. the fit is isotropic, so it is indexed with the
	// area-preserving effective roughness \sqrt{\alpha_x \alpha_y}
	const float iorRatio = (etaRel - 1.0f) / (etaRel + 1.0f);
	const float f0 = iorRatio * iorRatio;
	const float energyCompensation = (
		utilMicrofacetDielectricEnergyCompensate(
			wiLocal.z, sqrt(alpha.x * alpha.y), f0
		)
	);
	// the 1/(4 (\omega_wi \cdot n) (\omega_wo \cdot n)) term is already
	// folded into the smith visibility term
	// TODO multiply by specularweight right?
	return mfFresnel * mfDistribution * mfVisibility * energyCompensation;
}

// specular weight scales f0 via an effective ior
float openPbrEffectiveIor(const float ior, const float weight) {
	const float r = (ior - 1.0f) / (ior + 1.0f);
	const float f0 = clamp(weight * r * r, 0.0f, 0.999f);
	const float s = sqrt(f0);
	return (1.0f + s) / (1.0f - s);
}

// inverse of the schlick ior->f0 relationship above, per channel; used to
// recover an effective real ior from a tinted metal's f0 so thin-film
// interference has a base substrate to reflect off of, since the f82-tint
// conductor model only carries a schlick f0 tint, not real per-channel n/k
vec3 openPbrIorFromF0(const vec3 f0) {
	const vec3 s = sqrt(clamp(f0, 0.0f, 0.9999f));
	return (vec3(1.0f) + s) / (vec3(1.0f) - s);
}

#endif // UTIL_MATERIAL_OPENPBR_MICROFACET_GLSL
