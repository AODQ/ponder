implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

// -----------------------------------------------------------------------------
// -- albedo/radius -> extinction inversion
// -----------------------------------------------------------------------------
// converts openpbr's user-facing subsurface parameterization into the
// (sigmaAbsorption, sigmaScattering) pair for random walk integration

// floors anisotropy
public static const f32 skMaxSubsurfaceAnisotropy = 0.95f;

// mean free path floor
public static const f32 skMinSubsurfaceRadius = 1e-5f;

/*
	christensen & burley 2015's grosjean diffusion-reflectance
	approximation (searchlight configuration):

	R_d(\alpha') = \frac{\alpha'}{2}
		\left(1 + e^{-\frac{4}{3}\sqrt{3(1-\alpha')}}\right)
		e^{-\sqrt{3(1-\alpha')}}
*/
public func bsdfOpenPbrSubsurfaceDiffusionReflectance(
	αReduced: f32,
) -> f32 {
	let s: f32 = sqrt(3.0f * (1.0f - αReduced));
	return (
		0.5f * αReduced
		* (1.0f + exp(-4.0f / 3.0f * s))
		* exp(-s)
	);
}

// inverts bsdfOpenPbrSubsurfaceDiffusionReflectance via bisection
public func bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(
	targetAlbedo: f32,
) -> f32 {
	var lo: f32 = 0.0f;
	var hi: f32 = 1.0f;
	for (var i: i32 = 0; i < 20; ++i) {
		let mid: f32 = 0.5f * (lo + hi);
		if (bsdfOpenPbrSubsurfaceDiffusionReflectance(mid) < targetAlbedo) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5f * (lo + hi);
}

// per wavelength; λ is only needed to build the rgb build's spectrum
public func bsdfOpenPbrSubsurfaceAlbedoToSigma(
	albedo: vkof::Spectrum,
	radius: vkof::Spectrum,
	anisotropy: f32,
	λ: Wavelengths,
	out σAbsorption: vkof::Spectrum,
	out σScattering: vkof::Spectrum,
) {
	let g: f32 = min(anisotropy, skMaxSubsurfaceAnisotropy);
	// -- reduced albedo, inverted per channel
#if defined(PONDER_SPECTRUM_RGB)
	let αReduced: vkof::Spectrum = (
		vkof::Spectrum.fromRgb(
			f32v3(
				bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(
					clamp(albedo.channel(0u), 0.0f, 1.0f)
				),
				bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(
					clamp(albedo.channel(1u), 0.0f, 1.0f)
				),
				bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(
					clamp(albedo.channel(2u), 0.0f, 1.0f)
				)
			),
			λ
		)
	);
#else
	var αReduced: vkof::Spectrum = vkof::Spectrum.zero();
	[ForceUnroll]
	for (u32 i = 0u; i < skWavelengthSet; ++i) {
		αReduced.amounts[i] = (
			bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(
				clamp(albedo.channel(i), 0.0f, 1.0f)
			)
		);
	}
#endif
	// -- extinction from the mean free path
	let safeRadius: vkof::Spectrum = max(radius, skMinSubsurfaceRadius);
	let σTReduced: vkof::Spectrum = vkof::Spectrum.one() / safeRadius;
	let σSReduced: vkof::Spectrum = αReduced * σTReduced;
	σAbsorption = σTReduced - σSReduced;
	σScattering = σSReduced / (1.0f - g);
}

// -----------------------------------------------------------------------------
// -- henyey-greenstein phase
// -----------------------------------------------------------------------------

public func bsdfOpenPbrSubsurfacePhaseEvalHg(
	cosθ: f32,
	g: f32,
) -> f32 {
	let g2: f32 = g * g;
	let denom: f32 = max(1.0f + g2 - 2.0f * g * cosθ, 1e-7f);
	// (1 - g^2) / (4 \pi (1 + g^2 - 2 g \cos\theta)^{3/2})
	return (1.0f - g2) / (2.0f * 2.0f * π * denom * sqrt(denom));
}

public func bsdfOpenPbrSubsurfacePhaseSampleHgCos(
	u: f32,
	g: f32,
) -> f32 {
	// isotropic limit
	if (abs(g) < 1e-3f) {
		return 1.0f - 2.0f * u;
	}
	// standard HG inversion
	let s: f32 = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
	return (1.0f + g * g - s * s) / (2.0f * g);
}

public func bsdfOpenPbrSubsurfacePhaseSampleWo(
	ωi: f32v3,
	anisotropy: f32,
	ξ: f32v2,
) -> f32v3 {
	let cosθ: f32 = (
		bsdfOpenPbrSubsurfacePhaseSampleHgCos(ξ.x, anisotropy)
	);
	let φ: f32 = 2.0f * π * ξ.y;
	return vkof::directionAboutAxis(ωi, cosθ, φ);
}

// -----------------------------------------------------------------------------
// -- dwivedi diffusion support
// -----------------------------------------------------------------------------
// dwivaldi meng 2016 , d'eon 2020 zero variance chapter
// implements 'importance-sampling'-esque guiding of the random walk's
// free-flight direction along, sampling a ray towards where the walk
// is likely to exit the medium.

// diffusion length:
/*
	\begin{align}
	&L(\alpha) =
		1 / \sqrt{1 - \alpha^(2.44294 - 0.0215813 * \alpha + 0.578637/\alpha)}
		\tag{D'eon 2020, eq 67
	\\
	&
		p(cos \theta = \frac{1}{((L - cos\theta) * \text{phase_log})}
		\tag{Meng eq 9}
	\\
	&
		cos(\theta) = L - (L+1) * exp(-Xi * \text{phase_log})
		\text{Meng eq 10}
	\\
	&
		\sigma_t' = \sigma_t * (1 - cos\theta/L)
		\sigma_t' = \sigma_t * (1 + cos\theta/L)
	\end{align}
	where L is the diffusion length,
	\alpha is the maximum albedo of the three channels,
	phase_log is the log of the phase function,
*/

public func bsdfOpenPbrDwivediDiffusionLength(
	albedoMax: f32,
) -> f32 {
	let a: f32 = clamp(albedoMax, 0.0f, 0.99999f);
	return (
		// 1 / \sqrt{1 - \alpha^(2.44294 - 0.0215813 * \alpha + 0.578637/\alpha)}
		1.0f / sqrt(1.0f - pow(a, 2.44294f - 0.0215813f * a + 0.578637f / a))
	);
}

public func bsdfOpenPbrDwivediPhaseEval(
	diffusionLength: f32,
	phaseLog: f32,
	cosθ: f32,
) -> f32 {
	return (
		// p(cos \theta) = \frac{1}{((L - cos\theta) * \text{phase_log})}
		1.0f / ((diffusionLength - cosθ) * phaseLog)
	);
}

public func bsdfOpenPbrDwivediPhaseSampleCos(
	diffusionLength: f32,
	phaseLog: f32,
	ξ: f32,
) -> f32 {
	// cos(\theta) = L - (L+1) * exp(-Xi * \text{phase_log})
	return diffusionLength - (diffusionLength + 1.0f) * exp(-ξ * phaseLog);
}

// -----------------------------------------------------------------------------
// -- subsurface lobe interface
// -----------------------------------------------------------------------------

// unlike every other lobe, subsurface has no instantaneous local value:
// real light transport through the medium only happens by running the
// walk, entered via bsdfOpenPbrSubsurfaceEntrySampleWo below.
public func bsdfOpenPbrSubsurfaceEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
) -> vkof::Spectrum {
	return vkof::Spectrum.zero();
}

public func bsdfOpenPbrSubsurfaceEntrySampleWo(
	nor: f32v3,
	ξ: f32v2,
) -> f32v3 {
	return vkof::cosineHemisphereSampleWo(-nor, ξ);
}

// mirror of bsdfOpenPbrGlossyDiffusePdf
public func bsdfOpenPbrSubsurfaceEntryPdf(
	dotNorWo: f32,
) -> f32 {
	return max(-dotNorWo, 0.0f) * (1.0f / π);
}

} // namespace vkof
