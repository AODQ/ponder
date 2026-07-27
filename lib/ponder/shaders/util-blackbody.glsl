#ifndef UTIL_BLACKBODY_GLSL
#define UTIL_BLACKBODY_GLSL

#ifndef f32
#define i32v2 ivec2
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

// blackbody emission from a temperature grid; see lib/ponder/VDB_REFERENCES.md
// (pbrt's GridMedium blackbody emission, Fong/Wrenninge/Kulla/Habel 2017
// "Production Volume Rendering") for why a temperature-driven emission term
// is the standard way volume renderers add self-illumination (fire), and
// Wyman/Sloan/Shirley 2013 for the CIE fit below.

// planck's law: spectral radiance of a blackbody at temperature T (kelvin),
// evaluated at wavelength lambdaNm (nanometers). SI units throughout
// (wavelength converted to meters internally) -- returns W.sr^-1.m^-2.m^-1,
// a per-meter spectral density; the caller's wavelength-domain integration
// step must also be in meters for the units to work out consistently
f32 utilPlanckRadiance(const f32 lambdaNm, const f32 temperatureKelvin) {
	// SI 2019 redefinition exact values
	const f32 h = 6.62607015e-34f;
	const f32 c = 2.99792458e8f;
	const f32 kB = 1.380649e-23f;
	const f32 lambdaM = lambdaNm * 1e-9f;
	const f32 lambda5 = (
		lambdaM * lambdaM * lambdaM * lambdaM * lambdaM
	);
	const f32 numer = 2.0f * h * c * c;
	const f32 expArg = (
		(h * c) / (lambdaM * kB * max(temperatureKelvin, 1.0f))
	);
	// clamped purely to avoid an exp() overflow to Inf in a degenerate
	// caller (near-zero T); correctly represents negligible emission at
	// that wavelength/temperature either way, not a source of bias
	return numer / (lambda5 * (exp(min(expArg, 80.0f)) - 1.0f));
}

// wyman, sloan & shirley 2013 (JCGT 2(2)), "Simple Analytic Approximations
// to the CIE XYZ Color Matching Functions" -- closed-form multi-lobe
// gaussian fit to the CIE 1931 2-degree standard observer, verified
// against the paper's own supplemental source (curves/multiLobeFit1931.cpp,
// JournalOfComputerGraphicsTechniques/TEST-0002-02-01-Wyman-Sloan-Shirley
// on github). lambdaNm in nanometers
f32v3 utilCieXyz1931(const f32 lambdaNm) {
	const f32 t1x = (
		(lambdaNm - 442.0f) * (lambdaNm < 442.0f ? 0.0624f : 0.0374f)
	);
	const f32 t2x = (
		(lambdaNm - 599.8f) * (lambdaNm < 599.8f ? 0.0264f : 0.0323f)
	);
	const f32 t3x = (
		(lambdaNm - 501.1f) * (lambdaNm < 501.1f ? 0.0490f : 0.0382f)
	);
	const f32 x = (
		0.362f * exp(-0.5f * t1x * t1x)
		+ 1.056f * exp(-0.5f * t2x * t2x)
		- 0.065f * exp(-0.5f * t3x * t3x)
	);

	const f32 t1y = (
		(lambdaNm - 568.8f) * (lambdaNm < 568.8f ? 0.0213f : 0.0247f)
	);
	const f32 t2y = (
		(lambdaNm - 530.9f) * (lambdaNm < 530.9f ? 0.0613f : 0.0322f)
	);
	const f32 y = (
		0.821f * exp(-0.5f * t1y * t1y)
		+ 0.286f * exp(-0.5f * t2y * t2y)
	);

	const f32 t1z = (
		(lambdaNm - 437.0f) * (lambdaNm < 437.0f ? 0.0845f : 0.0278f)
	);
	const f32 t2z = (
		(lambdaNm - 459.0f) * (lambdaNm < 459.0f ? 0.0385f : 0.0725f)
	);
	const f32 z = (
		1.217f * exp(-0.5f * t1z * t1z)
		+ 0.681f * exp(-0.5f * t2z * t2z)
	);

	return f32v3(x, y, z);
}

// blackbody radiance at temperatureKelvin, converted to linear sRGB.
// numerically integrates planck's law against the CIE fit above over the
// visible range (380-780nm) via a 32-sample midpoint riemann sum -- no
// LUT/texture needed, matching this codebase's preference for a small
// closed-form evaluation over a big precomputed table where the underlying
// function is cheap (cf. openPbrDispersionIorRgb's 3-line evaluation).
// output is *relative* radiance: the stefan-boltzmann T^4 brightness
// scaling between different temperatures is physically correct and
// preserved, but -- like every other light source in this otherwise
// non-physically-calibrated renderer (envIntensity etc.) -- it is not
// normalized to absolute SI units; the caller's own artist-facing emission
// scale brings it into a displayable range
f32v3 utilBlackbodyRadiance(const f32 temperatureKelvin) {
	const f32 lambdaMin = 380.0f;
	const f32 lambdaMax = 780.0f;
	const u32 sampleCount = 32u;
	const f32 dLambdaNm = (lambdaMax - lambdaMin) / f32(sampleCount);
	// meters, for planck's law's own wavelength-domain integration measure
	const f32 dLambdaM = dLambdaNm * 1e-9f;

	f32v3 xyz = f32v3(0.0f);
	for (u32 i = 0u; i < sampleCount; ++i) {
		const f32 lambdaNm = lambdaMin + (f32(i) + 0.5f) * dLambdaNm;
		const f32 radiance = utilPlanckRadiance(lambdaNm, temperatureKelvin);
		xyz += utilCieXyz1931(lambdaNm) * radiance * dLambdaM;
	}

	// CIE XYZ -> linear sRGB, D65 white point (IEC 61966-2-1)
	const f32v3 rgb = f32v3(
		 3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z,
		-0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z,
		 0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z
	);
	// out-of-gamut temperatures (very cool/very hot ends of the locus) can
	// produce a slightly negative primary; clamp rather than let it poison
	// a downstream throughput multiply -- standard practice for blackbody
	// locus colors near the gamut boundary
	return max(rgb, f32v3(0.0f));
}

#endif // UTIL_BLACKBODY_GLSL
