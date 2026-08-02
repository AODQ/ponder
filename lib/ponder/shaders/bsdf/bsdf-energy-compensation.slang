implementing bsdf;

import types;

namespace vkof {

// rational quadratic fit to monte carla data for ggx directional albedo;
// from materialx
public func bsdfMicrofacetGgxDirectionalAlbedo(
	dotNorWi: f32,
	α: f32,
	f0: f32,
	f90: f32,
) -> f32 {
	let x: f32 = dotNorWi;
	let y: f32 = α;
	let x2: f32 = x * x;
	let y2: f32 = y * y;
	let r: f32v4 = (
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
	let ab: f32v2 = clamp(r.xy/r.zw, 0.0f, 1.0f);
	return f0 * ab.x + f90 * ab.y;
}

// turquin 2019 multiplicative energy compensation for the single-scatter
// ggx dielectric lobe (kulla & conty 2017 compensate additively with
// tables instead)
/*
	\begin{align*}
	&\textbf{(1) compensation factor} \\
	&f = f_{ss} (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\tag{turquin 2019}\\
	&\textbf{(2) average fresnel} \\
	&F_{avg} = 2 \int_0^1 F(\mu) \mu \, d\mu
		= F_0 + \frac{F_{90} - F_0}{21}
		\tag{kulla \& conty 2017}\\
	\end{align*}
*/
public func bsdfMicrofacetDielectricEnergyCompensate(
	μ: f32,
	roughness: f32,
	f0: f32,
) -> f32 {
	// (2) F_{avg} = F_0 + (F_{90} - F_0) / 21, with F_{90} = 1
	let fAvg: f32 = f0 + (1.0f - f0) * (1.0f / 21.0f);
	// (3) E_{ss}(\mu, \alpha)
	let ess: f32 = (
		max(bsdfMicrofacetGgxDirectionalAlbedo(μ, roughness, 1.0f, 1.0f), 1e-4f)
	);
	// (1) multiplier for the single-scatter lobe
	return 1.0f + fAvg * (1.0f - ess) / ess;
}

// compensated directional albedo of a dielectric ggx interface: the
// fresnel-weighted single-scatter albedo times the turquin multiplier
/*
	E(\mu) = (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\cdot E_{ss}^F(\mu, \alpha, f_0)
*/
public func bsdfMicrofacetDielectricAlbedo(
	μ: f32,
	roughness: f32,
	f0: f32,
) -> f32 {
	// E_{ss}^F: single-scatter albedo with the actual fresnel
	let essF: f32 = (
		bsdfMicrofacetGgxDirectionalAlbedo(μ, roughness, f0, 1.0f)
	);
	return (
		clamp(
			bsdfMicrofacetDielectricEnergyCompensate(μ, roughness, f0) * essF,
			0.0f,
			1.0f
		)
	);
}

} // namespace vkof
