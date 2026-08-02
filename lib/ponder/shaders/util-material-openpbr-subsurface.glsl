#ifndef UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL
#define UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL

#include "util-random.glsl"
#include "util-shading-frame.glsl"

// -----------------------------------------------------------------------------
// -- albedo/radius -> extinction inversion
// -----------------------------------------------------------------------------
// converts openpbr's user-facing subsurface parameterization into the
// (sigmaAbsorption, sigmaScattering) pair a random walk integrates with
// (see util-vdb.glsl's utilVdbSigmaT / utilVdbAlbedo, same representation)

// floors anisotropy away from the g -> 1 similarity-relation singularity
// below (sigmaScattering = sigmaScatteringReduced / (1 - g) blows up as g
// approaches 1); openpbr's subsurface_scatter_anisotropy domain is [-1, 1]
const float skMaxSubsurfaceAnisotropy = 0.95f;

// mean free path floor: a channel radius of exactly 0 would send
// sigmaTReduced = 1/radius to infinity; flooring keeps that channel
// finite (effectively opaque -- absorbed at a very short distance)
// instead of propagating an inf into the similarity-relation divide below
const float skMinSubsurfaceRadius = 1e-5f;

/*
	christensen & burley 2015's grosjean diffusion-reflectance
	approximation (searchlight configuration):

	R_d(\alpha') = \frac{\alpha'}{2}
		\left(1 + e^{-\frac{4}{3}\sqrt{3(1-\alpha')}}\right)
		e^{-\sqrt{3(1-\alpha')}}

	where \alpha' is the reduced single-scattering albedo. the formula's
	usual A = (1+F_{dr})/(1-F_{dr}) internal-reflection correction is
	omitted (equivalent to A=1): the dielectric-interface fresnel term is
	already owned by the shared fSpecular / fSpecularDirectionalAlbedo
	terms in openPbrEvaluateF (see openPbrSubsurfaceEvaluateF below), so
	including it here would double-count that boundary
*/
vec3 openPbrSubsurfaceDiffusionReflectance(const vec3 alphaReduced) {
	const vec3 s = sqrt(3.0f * (1.0f - alphaReduced));
	return (
		0.5f * alphaReduced
		* (1.0f + exp(-4.0f / 3.0f * s))
		* exp(-s)
	);
}

// inverts openPbrSubsurfaceDiffusionReflectance per channel via bisection
// (R_d is monotonic increasing on [0, 1]: R_d(0) = 0, R_d(1) = 1); 20
// iterations halves the bracket to ~1e-6, well past f32 precision for an
// already-approximate fit
vec3 openPbrSubsurfaceInvertDiffusionReflectance(const vec3 targetAlbedo) {
	vec3 lo = vec3(0.0f);
	vec3 hi = vec3(1.0f);
	for (int i = 0; i < 20; ++i) {
		const vec3 mid = 0.5f * (lo + hi);
		const bvec3 raise = (
			lessThan(openPbrSubsurfaceDiffusionReflectance(mid), targetAlbedo)
		);
		lo = mix(lo, mid, raise);
		hi = mix(hi, mid, not(raise));
	}
	return 0.5f * (lo + hi);
}

/*
	1. solve the *reduced* single-scattering albedo \alpha' per channel
	   from the target diffuse reflectance (subsurfaceColor) via the
	   diffusion approximation above
	2. \sigma_t' = 1 / radius, \sigma_s' = \alpha' \sigma_t',
	   \sigma_a = \sigma_t' - \sigma_s'
	3. undo the similarity-theory reduction with the real HG anisotropy g
	   (subsurfaceScatterAnisotropy) so the walk's own phase function --
	   not an isotropic stand-in -- sees the correct mean free path:
	   \sigma_s = \sigma_s' / (1 - g), \sigma_a unchanged by the reduction
*/
void openPbrSubsurfaceAlbedoToSigma(
	const vec3 albedo,
	const vec3 radius,
	const float anisotropy,
	out vec3 sigmaAbsorption,
	out vec3 sigmaScattering
) {
	const vec3 safeRadius = max(radius, vec3(skMinSubsurfaceRadius));
	const float g = min(anisotropy, skMaxSubsurfaceAnisotropy);
	const vec3 alphaReduced = (
		openPbrSubsurfaceInvertDiffusionReflectance(clamp(albedo, 0.0f, 1.0f))
	);
	const vec3 sigmaTReduced = 1.0f / safeRadius;
	const vec3 sigmaSReduced = alphaReduced * sigmaTReduced;
	sigmaAbsorption = sigmaTReduced - sigmaSReduced;
	sigmaScattering = sigmaSReduced / (1.0f - g);
}

// -----------------------------------------------------------------------------
// -- bounded homogeneous free-flight walk step
// -----------------------------------------------------------------------------
// the actual walk loop (repeatedly re-intersecting the mesh boundary to
// bound each segment's distanceMax) needs the scene's ray-tracing
// acceleration structure, which lives app-side (utilTraceRay,
// app/viewer/shaders/util-raytrace.glsl) -- lib/ponder/shaders stays
// standalone. this file only owns the per-segment sampling math: given a
// candidate max distance, decide whether the path exits the medium
// unscattered, scatters, or is absorbed.

// henyey-greenstein cosine-angle sampler; the same formula as
// utilVdbPhaseSampleHgCos (util-vdb.glsl), duplicated rather than shared
// via #include: pulling in util-vdb.glsl here would drag its nanovdb/
// bindless-buffer bindings into every consumer of util-material-openpbr.glsl,
// almost none of which touch vdb grids at all
// (TODO REVIEW)
// hg phase value for the guided walk's mis denominator. classical-only
// sampling never needed it: hg sampling is exact, so phase/pdf cancels
float openPbrSubsurfacePhaseEvalHg(const float cosTheta, const float g) {
	const float g2 = g * g;
	const float denom = max(1.0f + g2 - 2.0f * g * cosTheta, 1e-7f);
	// (1 - g^2) / (4 \pi (1 + g^2 - 2 g \cos\theta)^{3/2})
	return (1.0f - g2) / (2.0f * TAU * denom * sqrt(denom));
}
// (TODO REVIEW)

float openPbrSubsurfacePhaseSampleHgCos(const float u, const float g) {
	// isotropic limit
	if (abs(g) < 1e-3f) {
		return 1.0f - 2.0f * u;
	}
	// standard HG inversion
	const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
	return (1.0f + g * g - s * s) / (2.0f * g);
}

// samples a re-aim direction around wi from the HG phase function
// (subsurfaceScatterAnisotropy as g); frisvad basis, same construction as
// utilVdbPhaseSampleMixture
vec3 openPbrSubsurfacePhaseSampleWo(
	const vec3 wi,
	const float anisotropy,
	const vec2 xi
) {
	const float cosTheta = (
		openPbrSubsurfacePhaseSampleHgCos(xi.x, anisotropy)
	);
	const float phi = TAU * xi.y;
	return utilDirectionAboutAxis(wi, cosTheta, phi);
}

// one free-flight step landed on the medium boundary unscattered (t >=
// distanceMax): the caller's traced segment reaches the mesh exit point
#define OPENPBR_SUBSURFACE_WALK_EXIT 0u
// a real collision inside the medium; outWo/outDistance are both valid.
// absorption is not a separate outcome here -- see this function's header
// comment for why
#define OPENPBR_SUBSURFACE_WALK_SCATTER 1u

/*
	adaptive-probability hero channel, ported 1:1 from blender cycles'
	random-walk subsurface scattering (intern/cycles/kernel/closure/
	volume.h's volume_sample_channel/volume_sample_channel_pdf,
	intern/cycles/kernel/integrator/subsurface_random_walk.h's per-bounce
	channel resample and throughput update). four self-derived chromatic
	schemes were tried before reaching for this reference and every one
	broke for exactly the case subsurface needs most -- strongly
	chromatic sigma_t (skin's red mean free path several times longer
	than blue's): a single hero locked for the whole path collapsed
	throughput onto one channel at a fixed 3x (heavy per-sample color
	noise); weighted delta tracking compounds its correction
	multiplicatively across null-collision retries and blows up for a
	channel far from the majorant; single-sample ratio tracking against
	a plain achromatic mean has unbounded variance for a channel far
	below the mean; one-sample mis with a *uniformly* chosen hero (1/3
	each) locked per walk bounds each step's weight to <=3, but that
	bound is also the *typical* value for the hero channel at every one
	of its own steps, and (3 * albedo) easily exceeds 1 for any
	realistic albedo, compounding into a genuine exponential blowup over
	a many-bounce walk (observed: nan pixels after enough spp).

	the fix in all four cases turned out to be the same one thing: the
	hero-selection probability was fixed (either always 1.0 for a locked
	channel, or a flat 1/3), instead of *adaptive*. cycles resamples the
	hero every bounce like the failed uniform-per-step attempt above,
	but weights the draw by how much each channel currently matters --
	proportional to |throughput * albedo| -- so a channel that's already
	grown large keeps getting the sampling precision it needs (self-
	stabilizing, not runaway), and a channel that's already small or
	mostly-absorptive stops needing to be hero as often. this is the one
	piece none of the four self-derived attempts had.
*/

// cycles' volume_sample_channel_pdf: hero-selection probability per
// channel, proportional to how much throughput is currently carried on
// it times how much of that is even scatterable (albedo) -- not a flat
// 1/3. falls back to uniform only when there's nothing left to weight by
vec3 openPbrSubsurfaceChannelPdf(const vec3 throughput, const vec3 albedo) {
	const vec3 weights = abs(throughput * albedo);
	const float sumWeights = weights.x + weights.y + weights.z;
	if (sumWeights > 1e-8f) {
		return weights / sumWeights;
	}
	return vec3(1.0f / 3.0f);
}

// cycles' volume_sample_channel: discrete inverse-cdf draw from
// channelPdf (must already sum to 1)
uint openPbrSubsurfaceSampleChannel(const vec3 channelPdf, inout u64 state) {
	const float u = fnSampleUniform(state);
	if (u < channelPdf.x) {
		return 0u;
	}
	if (u < channelPdf.x + channelPdf.y) {
		return 1u;
	}
	return 2u;
}

// -----------------------------------------------------------------------------
// -- dwivedi diffusion support
// -----------------------------------------------------------------------------
// dwivaldi meng 2016 , d'eon 2020 zero variance chapter
// implements 'importance-sampling'-esque guiding of the random walk's
// free-flight direction along, sampling a ray towards where the walk
// is likely to exit the medium.
// The guide steers both direction and distance
// It's only an approximation; it starts to fail when the medium is thin;
// that's why it needs to be mixed and importance-sampled against the
// classical sampling.

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

float openPbrDwivediDiffusionLength(const float albedoMax) {
	const float a = clamp(albedoMax, 0.0f, 0.99999f);
	return (
		// 1 / \sqrt{1 - \alpha^(2.44294 - 0.0215813 * \alpha + 0.578637/\alpha)}
		1.0f / sqrt(1.0f - pow(a, 2.44294f - 0.0215813f * a + 0.578637f / a))
	);
}

float openPbrDwivediPhaseEval(
	const float diffusionLength,
	const float phaseLog,
	const float cosTheta
) {
	return (
		// p(cos \theta) = \frac{1}{((L - cos\theta) * \text{phase_log})}
		1.0f / ((diffusionLength - cosTheta) * phaseLog)
	);
}

float openPbrDwivediPhaseSampleCos(
	const float diffusionLength,
	const float phaseLog,
	const float xi
) {
	// cos(\theta) = L - (L+1) * exp(-Xi * \text{phase_log})
	return diffusionLength - (diffusionLength + 1.0f) * exp(-xi * phaseLog);
}

// -----------------------------------------------------------------------------
// -- subsurface lobe interface
// -----------------------------------------------------------------------------

/*
	one free-flight step, hero channel drawn fresh this call from
	channelPdf (computed by the caller from the CURRENT throughput, since
	it changes every step -- see openPbrSubsurfaceChannelPdf). outWeight
	is the complete per-channel multiplier for the caller to fold
	directly into throughput -- no separate albedo multiply needed, it's
	already included (mirrors cycles' `sigma_s * transmittance` /
	`dot(channel_pdf, pdf)` for a collision, `transmittance /
	dot(channel_pdf, pdf)` for reaching the boundary): the mixture
	denominator dot(channelPdf, pdf) uses the *sampling* density
	(sigma_t-based, matching what actually produced this t), while the
	numerator is the *physical* quantity (sigma_s-weighted for a
	collision) -- these are only the same thing when picking a channel
	uniformly, which is exactly what made the earlier uniform attempts
	both easier to derive and wrong.
*/
uint openPbrSubsurfaceWalkStep(
	const vec3 sigmaAbsorption,
	const vec3 sigmaScattering,
	const float distanceMax,
	const vec3 wi,
	const float anisotropy,
	const vec3 throughput,
	// (TODO REVIEW)
	// dwivedi guiding axis: the outward normal at the walk's entry point,
	// so cos -> 1 means heading back out. a zero vector disables guiding
	// and leaves every result identical to the classical walk
	const vec3 guidingAxis,
	// (TODO REVIEW)
	out float outDistance,
	out vec3 outWo,
	out vec3 outWeight,
	inout u64 state
) {
	outWo = vec3(0.0f);
	const vec3 sigmaT = sigmaAbsorption + sigmaScattering;
	if (sigmaT.x <= 0.0f && sigmaT.y <= 0.0f && sigmaT.z <= 0.0f) {
		outWeight = vec3(1.0f);
		outDistance = distanceMax;
		return OPENPBR_SUBSURFACE_WALK_EXIT;
	}
	const vec3 albedo = sigmaScattering / max(sigmaT, vec3(1e-8f));
	const vec3 channelPdf = openPbrSubsurfaceChannelPdf(throughput, albedo);
	const uint channel = openPbrSubsurfaceSampleChannel(channelPdf, state);
	// same clamp openPbrSubsurfaceAlbedoToSigma applied when building sigma
	// from this g -- the walk's phase function and its medium must agree on g
	const float g = min(anisotropy, skMaxSubsurfaceAnisotropy);

	// (TODO REVIEW)
	// -- dwivedi guiding. every quantity below collapses to the classical
	// one when guidedFraction is 0, so the disabled path stays exact
	const float diffusionLength = (
		openPbrDwivediDiffusionLength(max(albedo.x, max(albedo.y, albedo.z)))
	);
	const bool guidingEnabled = (
		dot(guidingAxis, guidingAxis) > 0.5f && diffusionLength > 1.0f
	);
	const float phaseLog = (
		guidingEnabled
		? log((diffusionLength + 1.0f) / (diffusionLength - 1.0f))
		: 0.0f
	);
	// the zero-variance solution assumes isotropic scattering, so fade the
	// guided strategy out as the phase function sharpens
	const float guidedFraction = (
		guidingEnabled ? 1.0f - max(0.5f, pow(abs(g), 0.125f)) : 0.0f
	);
	// already heading for the exit: stretch the free path. heading deeper
	// shortens it, so the walk can turn around sooner
	const float cosTravel = guidingEnabled ? dot(wi, guidingAxis) : 0.0f;
	const vec3 sigmaTGuided = (
		max(
			sigmaT * (1.0f - cosTravel / max(diffusionLength, 1.0f)),
			vec3(1e-8f)
		)
	);
	// strategy pick must precede the distance draw it modifies. the draw is
	// independent of the xi pair the direction uses
	const bool guided = (
		guidedFraction > 0.0f && fnSampleUniform(state) < guidedFraction
	);
	const vec3 sigmaTUsed = guided ? sigmaTGuided : sigmaT;
	// (TODO REVIEW)

	const float sigmaTChannel = max(sigmaTUsed[channel], 1e-8f);
	const float t = -log(1.0f - fnSampleUniform(state)) / sigmaTChannel;
	if (t >= distanceMax) {
		// (TODO REVIEW)
		// both strategies could have reached the boundary, so the mis
		// denominator mixes both survival probabilities; the numerator stays
		// the physical transmittance
		const vec3 survival = exp(-sigmaT * distanceMax);
		const vec3 survivalGuided = exp(-sigmaTGuided * distanceMax);
		const float denom = (
			dot(channelPdf, mix(survival, survivalGuided, guidedFraction))
		);
		// (TODO REVIEW)
		outWeight = survival / max(denom, 1e-8f);
		outDistance = distanceMax;
		return OPENPBR_SUBSURFACE_WALK_EXIT;
	}

	const vec3 transmittance = exp(-sigmaT * t);
	// (TODO REVIEW)
	// mixture over both strategies regardless of which one drew t --
	// evaluating only the taken branch is what biases a guided walk
	const vec3 density = sigmaT * transmittance;
	const vec3 densityGuided = sigmaTGuided * exp(-sigmaTGuided * t);
	const float denom = (
		dot(channelPdf, mix(density, densityGuided, guidedFraction))
	);
	// (TODO REVIEW)
	outWeight = (sigmaScattering * transmittance) / max(denom, 1e-8f);
	outDistance = t;
	const vec2 xi = vec2(fnSampleUniform(state), fnSampleUniform(state));
	// (TODO REVIEW)
	if (guided) {
		const float cosTheta = (
			openPbrDwivediPhaseSampleCos(diffusionLength, phaseLog, xi.x)
		);
		outWo = utilDirectionAboutAxis(guidingAxis, cosTheta, TAU * xi.y);
	} else {
		outWo = openPbrSubsurfacePhaseSampleWo(wi, g, xi);
	}
	// hg sampling cancels against its own pdf, so the classical walk needed
	// no directional factor. once the guided strategy can produce the
	// direction, divide the physical phase value by the mixture that did
	if (guidedFraction > 0.0f) {
		const float phaseHg = openPbrSubsurfacePhaseEvalHg(dot(wi, outWo), g);
		const float pdfGuided = (
			openPbrDwivediPhaseEval(
				diffusionLength, phaseLog, dot(outWo, guidingAxis)
			) / TAU
		);
		outWeight *= (
			phaseHg / max(mix(phaseHg, pdfGuided, guidedFraction), 1e-8f)
		);
	}
	// (TODO REVIEW)
	return OPENPBR_SUBSURFACE_WALK_SCATTER;
}

/*
	unlike every other lobe in this file, subsurface has no instantaneous
	local value: real light transport through the medium only happens by
	actually running the bounded walk (openPbrSubsurfaceWalkStep), entered
	via openPbrSubsurfaceEntrySampleWo below. a direct nee query at the
	entry vertex ("what does this lobe reflect toward light direction wo")
	has no corresponding single-scattering event to answer with -- the
	energy that would show up here instead arrives later, at the walk's
	exit vertex, as ordinary nee on the exit-point's own dielectric
	interface. returning 0 here is what makes
	mix(fDiffuse, fSubsurface, mat.subsurfaceWeight) in openPbrEvaluateF
	correctly fade the local diffuse response to zero as subsurfaceWeight
	rises to 1, without needing to touch that call site
*/
vec3 openPbrSubsurfaceEvaluateF(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo
) {
	return vec3(0.0f);
}

// entry direction: cosine-weighted hemisphere mirrored to point into the
// surface. shares glossy-diffuse's E_spec top layer (see
// openPbrSubsurfaceEvaluateF), so like diffuse it has no fresnel/ior
// bending of its own -- the walk itself is where scattering direction
// actually matters
vec3 openPbrSubsurfaceEntrySampleWo(const vec3 nor, const vec2 xi) {
	return utilCosineHemisphereSampleWo(-nor, xi);
}

// mirror of openPbrGlossyDiffusePdf onto the lower hemisphere
float openPbrSubsurfaceEntryPdf(const float dotNorWo) {
	return max(-dotNorWo, 0.0f) * IPI;
}

#endif // UTIL_MATERIAL_OPENPBR_SUBSURFACE_GLSL
