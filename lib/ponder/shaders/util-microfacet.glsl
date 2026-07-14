#ifndef UTIL_MICROFACET_GLSL
#define UTIL_MICROFACET_GLSL

#ifndef f32
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

#ifndef PI
#define PI  3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// -- microfacet F_s functions
// -----------------------------------------------------------------------------

// ggx / trowbridge-reitz normal distribution function;
// trowbridge & reitz 1975, walter et al. 2007 eq 33,
// karis 2013 eq 3,
/*
	D(h) = \frac{\alpha^2}{\pi ((n \cdot h)^2 (\alpha^2 - 1) + 1)^2}
*/
f32 utilMicrofacetGgxDistribution(const f32 dotNorH, const f32 alpha) {
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
f32 utilMicrofacetGgxDistributionAniso(
	const f32v3 localH,
	const f32v2 alpha
) {
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
f32 utilMicrofacetSmithGgxVisibilityAniso(
	const f32v3 wiLocal,
	const f32v3 woLocal,
	const f32v2 alpha
) {
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
// alpha is the ggx slope scale, roughness (squared internally)
f32 utilMicrofacetSmithGgxVisibility(
	const f32 dotNorWi,
	const f32 dotNorWo,
	const f32 alpha
) {
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
// alpha is the ggx slope scale, roughness (squared internally)
f32 utilMicrofacetSmithG1(const f32 dotNorWi, const f32 alpha) {
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

#endif // UTIL_MICROFACET_GLSL
