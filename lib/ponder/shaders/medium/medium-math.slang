implementing medium;

#include "include/differentiable.slang"

import types;
import util;

// -----------------------------------------------------------------------------
// -- references
// -----------------------------------------------------------------------------
// [HG41] henyey & greenstein 1941, "diffuse radiation in the galaxy",
//        astrophysical journal vol 93
// [JD23] jendersie & d'eon 2023, "an approximate mie scattering function for
//        fog and cloud rendering", siggraph talks
//        https://research.nvidia.com/labs/rtr/approximate-mie/
// [PBRT4] pharr, jakob & humphreys 2023, "physically based rendering" 4th ed,
//        ch 11 and 14
// [WND14] wilkie, nawaz, droske, weidlich & hanika 2014, "hero wavelength
//        spectral sampling", egsr
// [Q10]  quilez 2010, "better fog" (analytic exponential height fog)
//        https://iquilezles.org/articles/fog/
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// -- phase function evaluation
// -----------------------------------------------------------------------------
// methods wrap these; slang cannot differentiate methods across modules

// [HG41]
// (1 - g²) / (4π (1 + g² - 2g cosθ)^(3/2))
PonderDifferentiable
func implPhaseHenyeyGreensteinEvaluate(
	PonderNoDiff g: f32,
	ωi: f32v3,
	ωo: f32v3,
) -> f32 {
	let g2: f32 = g*g;
	let cosθ: f32 = dot(ωo, ωi);
	let denom: f32 = max(1.0f + g2 - 2.0f*g*cosθ, 1e-7f);
	return (1.0f - g2) / (4.0f*π * denom * sqrt(denom));
}

// [JD23] eq 2
PonderDifferentiable
func implPhaseDraineEvaluate(
	PonderNoDiff g: f32,
	PonderNoDiff α: f32,
	ωi: f32v3,
	ωo: f32v3,
) -> f32 {
	let g2: f32 = g*g;
	let cosθ: f32 = dot(ωo, ωi);
	let denom: f32 = max(1.0f + g2 - 2.0f*g*cosθ, 1e-7f);
	// -- henyey-greenstein term
	let hg: f32 = (1.0f - g2) / (denom * sqrt(denom));
	// -- draine term
	let draine: f32 = (
		(1.0f + α*cosθ*cosθ) / (1.0f + α*(1.0f + 2.0f*g2) / 3.0f)
	);
	return hg * draine / (4.0f*π);
}

// [JD23] eq 3
PonderDifferentiable
func implPhaseFogEvaluate(
	PonderNoDiff gHenyeyGreenstein: f32,
	PonderNoDiff gDraine: f32,
	PonderNoDiff α: f32,
	PonderNoDiff draineWeight: f32,
	ωi: f32v3,
	ωo: f32v3,
) -> f32 {
	return lerp(
		implPhaseHenyeyGreensteinEvaluate(gHenyeyGreenstein, ωi, ωo),
		implPhaseDraineEvaluate(gDraine, α, ωi, ωo),
		draineWeight
	);
}

// fog lobe parameters from the droplet diameter in µm, [JD23] eq 4-7
// fitted for droplet diameters of 5 to 50 µm
func implPhaseFogParameters(
	dropletDiameter: f32,
	out gHenyeyGreenstein: f32,
	out gDraine: f32,
	out α: f32,
	out draineWeight: f32,
) -> void {
	let d: f32 = dropletDiameter;
	gHenyeyGreenstein = exp(-0.0990567f / (d - 1.67154f));
	gDraine = exp(-2.20679f / (d + 3.91029f) - 0.428934f);
	α = exp(3.62489f - 8.29288f / (d + 5.52825f));
	draineWeight = exp(-0.599085f / (d - 0.641583f) - 0.665888f);
}

// -----------------------------------------------------------------------------
// -- phase function sampling
// -----------------------------------------------------------------------------

// henyey-greenstein cdf inversion [HG41]
// cosθ = (1 + g² - ((1 - g²) / (1 - g + 2gu))²) / 2g
func implPhaseHenyeyGreensteinSampleCos(u: f32, g: f32) -> f32 {
	// -- isotropic limit
	if (abs(g) < 1e-3f) {
		return 1.0f - 2.0f*u;
	}
	// -- standard HG inverse
	let s: f32 = (1.0f - g*g) / (1.0f - g + 2.0f*g*u);
	return (1.0f + g*g - s*s) / (2.0f*g);
}

// draine cdf inversion [JD23]
// https://research.nvidia.com/labs/rtr/approximate-mie/assets/draine.hlsl
// ported from utilVdbPhaseSampleDraineCos (shaders-old)
func implPhaseDraineSampleCos(u: f32, gRaw: f32, a: f32) -> f32 {
	// -- keep g away from zero
	// the solve divides by 2g and is unstable up to |g| ~ 0.042
	let g: f32 = (
		abs(gRaw) < 0.05f
			? (gRaw >= 0.0f ? 0.05f : -0.05f)
			: gRaw
	);
	let g2: f32 = g*g;
	let g3: f32 = g*g2;
	let g4: f32 = g2*g2;
	let g6: f32 = g2*g4;
	let pgp1_2: f32 = (1.0f + g2) * (1.0f + g2);
	let T1a: f32 = -a + a*g4;
	let T1a3: f32 = T1a*T1a*T1a;
	let T2: f32 = (
		-1296.0f * (-1.0f + g2) * (a - a*g2) * T1a * (4.0f*g2 + a*pgp1_2)
	);
	let T3: f32 = (
		3.0f*g2 * (1.0f + g*(-1.0f + 2.0f*u))
		+ a * (2.0f + g2 + g3*(1.0f + 2.0f*g2)*(-1.0f + 2.0f*u))
	);
	let T4a: f32 = 432.0f*T1a3 + T2 + 432.0f*(a - a*g2)*T3*T3;
	let T4b: f32 = -144.0f*a*g2 + 288.0f*a*g4 - 144.0f*a*g6;
	let T4b3: f32 = T4b*T4b*T4b;
	let T4: f32 = T4a + sqrt(-4.0f*T4b3 + T4a*T4a);
	let T4p3: f32 = pow(T4, 1.0f / 3.0f);
	let cbrt2: f32 = pow(2.0f, 1.0f / 3.0f);
	let T6: f32 = (
		(
			2.0f*T1a
			+ (48.0f*cbrt2*(-(a*g2) + 2.0f*a*g4 - a*g6)) / T4p3
			+ T4p3 / (3.0f*cbrt2)
		)
		/ (a - a*g2)
	);
	let T5: f32 = 6.0f*(1.0f + g2) + T6;
	let root: f32 = (
		-0.5f*sqrt(T5)
		+ sqrt(
			6.0f*(1.0f + g2) - (8.0f*T3) / (a*(-1.0f + g2)*sqrt(T5)) - T6
		) / 2.0f
	);
	return (1.0f + g2 - root*root) / (2.0f*g);
}

// -----------------------------------------------------------------------------
// -- free-flight sampling
// -----------------------------------------------------------------------------
// chromatic media sample the distance from one channel, picked uniformly, and
// weight by the average density over all channels (single-sample mis)
// [PBRT4] ch 14.1, [WND14]

// uniform pick of the channel that samples the distance
func implMediumChannelPick(ξ: f32) -> u32 {
	let channelCount: u32 = vkof::Spectrum::channelCount();
	return min(u32(ξ * f32(channelCount)), channelCount - 1u);
}

// weight of a sample that passes the segment
// Tr over the average probability of passing
func implMediumWeightPass(Tr: vkof::Spectrum) -> vkof::Spectrum {
	return Tr * (1.0f / max(Tr.average(), 1e-8f));
}

// weight of a sample that scatters inside the segment
// σs Tr over the average free-flight density σt Tr
func implMediumWeightScatter(
	σs: vkof::Spectrum,
	σt: vkof::Spectrum,
	Tr: vkof::Spectrum,
) -> vkof::Spectrum {
	let density: vkof::Spectrum = σt * Tr;
	return σs * Tr * (1.0f / max(density.average(), 1e-8f));
}

// -----------------------------------------------------------------------------
// -- exponential height fog
// -----------------------------------------------------------------------------
// density p(y) = exp(-(y - origin) / falloff); along a ray y(t) = y0 + ωy·t,
// so with a = ωy / falloff and p0 = p(y0) the density is p0·e^(-a·t)
// closed forms as in [Q10]

// F(t) = ∫₀ᵗ p(y(s)) ds = p0 · (1 - e^(-a·t)) / a
// optical depth is τ(t) = σt0 · F(t)
func implHeightFogDensityIntegral(p0: f32, a: f32, t: f32) -> f32 {
	return (
		abs(a*t) < 1e-3f
		// if a is too small, use taylor expansion of (1 - e^(-a*t)) / a
		? p0*t*(1.0f - 0.5f*a*t)
		// closed form of integral solution
		: p0*(1.0f - exp(-a*t))/a
	);
}

// solves F(t) = u for t; f32.maxValue when the ray never accumulates u
// accounts for three cases:
// 1) a == 0, ray travels horizontally in the homogeneous medium
// 2) a > 0, ray travels upward in the medium
// 3) a < 0, ray travels downward in the medium
func implHeightFogDensityIntegralInverse(p0: f32, a: f32, u: f32) -> f32 {
	let x: f32 = u*a/p0;
	// if x is too small, use taylor expansion of -log(1-x)/a
	if (abs(x) < 1e-3f) { return (u/p0)*(1.0f + 0.5f*x); }
	return x >= 1.0f ? f32.maxValue : -log(1.0f - x)/a;
}
