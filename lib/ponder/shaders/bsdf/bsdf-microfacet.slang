implementing bsdf;

import types;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry
// TODO should flip this convention to match the rest but it was originally
// mistakenly written with the opposite direction

namespace vkof {

// -----------------------------------------------------------------------------
// -- microfacet half-vector helpers
// -----------------------------------------------------------------------------

// reconstructs the normalized half vector from \omega_i / \omega_o
public func bsdfHalfVectorLocal(
	ωiLocal: f32v3,
	ωoLocal: f32v3,
	out hLocal: f32v3,
) -> bool {
	let hLocalUnnorm: f32v3 = ωiLocal + ωoLocal;
	let valid: bool = dot(hLocalUnnorm, hLocalUnnorm) > 0.0f;
	hLocal = valid ? normalize(hLocalUnnorm) : f32v3(0.0f, 0.0f, 1.0f);
	return valid;
}

// -----------------------------------------------------------------------------
// -- microfacet F_s functions
// -----------------------------------------------------------------------------

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
// isotropic \alpha_x = \alpha_y reduces to bsdfMicrofacetGgxDistribution:
public func bsdfMicrofacetGgxDistributionAniso(
	localH: f32v3,
	α: f32v2,
) -> f32 {
	let α2: f32v2 = α * α;
	// \frac{h_x^2}{A_x} + \frac{h_y^2}{A_y} + h_z^2
	let d: f32 = (
		max(
			(
				localH.x * localH.x / α2.x
				+ localH.y * localH.y / α2.y
				+ localH.z * localH.z
			),
			1e-7f
		)
	);
	// 1 / ( \pi \sqrt{A_x A_y} d^2 )
	return 1.0f / (π * α.x * α.y * d * d);
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
// the equation folds the \frac{1}{4 G_1(\omega_i) G_1(\omega_o)} term into
// the final distribution's denominator, so the final microfacet BSDF need not
// divide by (\omega_i \cdot \omega_h) (\omega_o \cdot \omega_h)
public func bsdfMicrofacetSmithGgxVisibilityAniso(
	ωiLocal: f32v3,
	ωoLocal: f32v3,
	α: f32v2,
) -> f32 {
	if (ωiLocal.z <= 0.0f || ωoLocal.z <= 0.0f) {
		return 0.0f;
	}

	// \mu_i
	let μi: f32 = ωiLocal.z;
	// \mu_o
	let μo: f32 = ωoLocal.z;

	// A_o
	let Ao: f32 = length(f32v3(α.x * ωoLocal.x, α.y * ωoLocal.y, μo));
	// A_i
	let Ai: f32 = length(f32v3(α.x * ωiLocal.x, α.y * ωiLocal.y, μi));

	// \frac{0.5}{\mu_i A_o + \mu_o A_i}
	return 0.5f / max(μi * Ao + μo * Ai, 1e-7f);
}

// anisotropic G_1 = \frac{2 \omega_z}{\omega_z + A_\omega}, with the same
// A_\omega = |(\alpha_x \omega_x, \alpha_y \omega_y, \omega_z)| derived for
// bsdfMicrofacetSmithGgxVisibilityAniso above
public func bsdfMicrofacetSmithG1Aniso(
	ωLocal: f32v3,
	α: f32v2,
) -> f32 {
	if (ωLocal.z <= 0.0f) {
		return 0.0f;
	}
	// A_\omega
	let A: f32 = (
		length(f32v3(α.x * ωLocal.x, α.y * ωLocal.y, ωLocal.z))
	);
	return 2.0f * ωLocal.z / max(ωLocal.z + A, 1e-7f);
}

// -----------------------------------------------------------------------------
// -- microfacet sampling functions
// -----------------------------------------------------------------------------


// heitz & d'Eon 2014 VNDF;
// heitz 2018; rejects backfacing microfacets
// anisotropic VNDF eto & tokuyoshi 2023; bounded VNDF lobes to prevent
// below-horizon reflections
// It seems to still emit below-horizon reflections.
// scripts/bounded-vndf-bound-check.py measures the 'leaks'
// do not use for refraction sampling
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
public func bsdfMicrofacetSampleGgxBoundedWoAniso(
	tanX: f32v3,
	tanY: f32v3,
	nor: f32v3,
	ωi: f32v3,
	α: f32v2,
	ξ: f32v2,
) -> f32v3 {
	// -- step 1:  project wi into the local tangent frame

	// i = (i_x, i_y, i_z) = (wi \cdot t, wi \cdot b, wi \cdot n)
	// (unaccented: not yet stretched)
	let i: f32v3 = (
		f32v3(dot(ωi, tanX), dot(ωi, tanY), dot(ωi, nor))
	);

	// -- step 2:  stretch \acute{i}

	// \acute{i}
	let iStd: f32v3 = (
		// \text{normalize}(\alpha_x i_x, \alpha_y i_y, i_z)
		normalize(f32v3(i.x * α.x, i.y * α.y, i.z))
	);

	// -- step 3: cap azimuth
	// \phi = 2 \pi \xi_1
	let φ: f32 = 2.0f * π * ξ.x;

	// -- step 4: cap bounds
	// a = \text{saturate}(\min(\alpha_x, \alpha_y))
	let a: f32 = clamp(min(α.x, α.y), 0.0f, 1.0f);

	// s = 1 + |i_{xy}|; original i, not \acute{i}
	let s: f32 = 1.0f + length(f32v2(i.x, i.y));

	let a2: f32 = a * a;
	let s2: f32 = s * s;

	// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2}; original i_z, not \acute{i}_z
	let k: f32 = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);

	// -- step 5

	// b = \begin{cases} ... \end{cases}
	let b: f32 = i.z > 0.0f ? k * iStd.z : iStd.z;
	// \acute{o}_z = (1 - \xi_2)(1 + b) - b
	let z: f32 = (1.0f - ξ.y) * (1.0f + b) - b;
	// \sqrt{1 - \acute{o}_z^2}
	let sinθ: f32 = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
	// \acute{o}
	let oStd: f32v3 = (
		f32v3(
			// \sqrt{1 - \acute{o}_z^2} \cos\phi,
			sinθ * cos(φ),
			// \sqrt{1 - \acute{o}_z^2} \sin\phi,
			sinθ * sin(φ),
			// \acute{o}_z
			z
		)
	);

	// \acute{h} \propto \acute{i} + \acute{o};
	let hStd: f32v3 = iStd + oStd;
	// wi antiparallel to nor will cause hStd to store zero
	if (dot(hStd, hStd) < 1e-12f) {
		return nor;
	}
	// unstretch \acute{h} to get the microfacet normal m
	let h: f32v3 = (
		normalize(f32v3(hStd.x * α.x, hStd.y * α.y, hStd.z))
	);
	// reflect in local frame back to world
	let o: f32v3 = 2.0f * dot(i, h) * h - i;
	return o.x*tanX + o.y*tanY + o.z*nor;
}

// heitz & d'Eon 2014 VNDF; weights each normal to reject backfacing microfacets
// anisotropic. Returns the microfacet normal h.
// should only be used for refraction
public func bsdfMicrofacetSampleGgxVndfAniso(
	tanX: f32v3,
	tanY: f32v3,
	N: f32v3,
	ωi: f32v3,
	α: f32v2,
	ξ: f32v2,
) -> f32v3 {
	// i = (wi \cdot t, wi \cdot b, wi \cdot n)
	let ωiLocal: f32v3 = (
		f32v3(dot(ωi, tanX), dot(ωi, tanY), dot(ωi, N))
	);
	// \acute{i} = \text{normalize}(\alpha_x i_x, \; \alpha_y i_y, \; i_z)
	let ωiH: f32v3 = (
		normalize(f32v3(α.x * ωiLocal.x, α.y * ωiLocal.y, ωiLocal.z))
	);
	// orthonormal basis around \acute{i};
	// T_1 = \text{normalize}((-\acute{i}_y, \acute{i}_x, 0)),
	// with a fallback for the degenerate \acute{i} = +z view
	let lensq: f32 = ωiH.x * ωiH.x + ωiH.y * ωiH.y;
	let T1: f32v3 = (
		lensq > 1e-7f
		? f32v3(-ωiH.y, ωiH.x, 0.0f) * (1.0f / sqrt(lensq))
		: f32v3(1.0f, 0.0f, 0.0f)
	);
	// T_2 = \acute{i} * T_1
	let T2: f32v3 = cross(ωiH, T1);
	// uniform polar disk sample: r = \sqrt{\xi_1}, \phi = 2\pi\xi_2
	let r: f32 = sqrt(ξ.x);
	let φ: f32 = 2.0f * π * ξ.y;
	let t1: f32 = r * cos(φ);
	var t2: f32 = r * sin(φ);
	// the visible hemisphere projects onto the disk as a half-disk plus a
	// half-disk squashed by \acute{i}_z;
	// makes (t_1, t_2) uniform
	// s = \frac{1 + \acute{i}_z}{2},
	// t_2 \leftarrow (1 - s)\sqrt{1 - t_1^2} + s \, t_2
	let s: f32 = 0.5f * (1.0f + ωiH.z);
	t2 = (1.0f - s) * sqrt(max(1.0f - t1 * t1, 0.0f)) + s * t2;
	// lift onto the hemisphere along \acute{i};
	// \acute{h} = t_1 T_1 + t_2 T_2 + \sqrt{1 - t_1^2 - t_2^2} \, \acute{i}
	let Nh: f32v3 = (
		t1 * T1 + t2 * T2
		+ sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * ωiH
	);
	// unstretch; \max(0, \acute{h}_z) is the \chi^+ guard;
	// h = \text{normalize}(
	//     \alpha_x \acute{h}_x, \; \alpha_y \acute{h}_y, \; \max(0, \acute{h}_z))
	let Hlocal: f32v3 = (
		normalize(f32v3(α.x * Nh.x, α.y * Nh.y, max(0.0f, Nh.z)))
	);
	return Hlocal.x * tanX + Hlocal.y * tanY + Hlocal.z * N;
}

// -----------------------------------------------------------------------------
// -- microfacet pdf functions
// -----------------------------------------------------------------------------

// pdf of bsdfMicrofacetSampleGgxBoundedWoAniso for the given reflected
// direction wo
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
public func bsdfMicrofacetGgxBoundedReflectPdfAniso(
	tanX: f32v3,
	tanY: f32v3,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
	α: f32v2,
) -> f32 {
	// tolerance on the cap-cut tests, error scales with 1/\alpha
	let kCapCutEpsilon: f32 = (
		1e-5f / clamp(min(α.x, α.y), 0.001f, 1.0f)
	);

	// same tangent frame as bsdfMicrofacetSampleGgxBoundedWoAniso
	let i: f32v3 = (
		f32v3(dot(ωi, tanX), dot(ωi, tanY), dot(ωi, nor))
	);
	let o: f32v3 = (
		f32v3(dot(ωo, tanX), dot(ωo, tanY), dot(ωo, nor))
	);
	var h: f32v3;
	if (!bsdfHalfVectorLocal(i, o, h)) {
		return 0.0f;
	}
	// D(h)
	let ndf: f32 = bsdfMicrofacetGgxDistributionAniso(h, α);
	// (\alpha_x i_x, \alpha_y i_y); its squared length is t^2 - i_z^2,
	// reused by the stable form in (4)
	let ai: f32v2 = f32v2(i.x * α.x, i.y * α.y);
	let len2: f32 = dot(ai, ai);
	// t = \sqrt{\alpha_x^2 i_x^2 + \alpha_y^2 i_y^2 + i_z^2}
	let t: f32 = sqrt(len2 + i.z * i.z);
	if (i.z >= 0.0f) {
		// a = \text{saturate}(\min(\alpha_x, \alpha_y)) (eq 6)
		let a: f32 = clamp(min(α.x, α.y), 0.0f, 1.0f);
		// s = 1 + |i_{xy}|; original i, not \acute{i}
		let s: f32 = 1.0f + length(f32v2(i.x, i.y));
		let a2: f32 = a * a;
		let s2: f32 = s * s;
		// k = \frac{(1 - a^2) s^2}{s^2 + a^2 i_z^2} (eq 5)
		let k: f32 = (1.0f - a2) * s2 / (s2 + a2 * i.z * i.z);
		// below-horizon ωo is only reachable if its coordinate stays above eq5
		if (o.z < 0.0f) {
			let iStd: f32v3 = (
				normalize(f32v3(i.x * α.x, i.y * α.y, i.z))
			);
			// normals unstretch inversely to directions:
			// \acute{h} = \text{normalize}(h_x / \alpha_x, h_y / \alpha_y, h_z)
			let hStd: f32v3 = (
				normalize(f32v3(h.x / α.x, h.y / α.y, h.z))
			);
			// \acute{o} = 2 (\acute{i} \cdot \acute{h}) \acute{h} - \acute{i}
			let oStd: f32v3 = (
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
		let iStd: f32v3 = (
			normalize(f32v3(i.x * α.x, i.y * α.y, i.z))
		);
		let hStd: f32v3 = (
			normalize(f32v3(h.x / α.x, h.y / α.y, h.z))
		);
		let oStd: f32v3 = (
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

} // namespace vkof
