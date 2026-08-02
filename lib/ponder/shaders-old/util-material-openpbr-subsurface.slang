module "util-material-openpbr-subsurface";
__exported import "util-types";
__exported import "util-random";
__exported import "util-shading-frame";
__exported import "util-material-openpbr-shared";
__exported import "util-material-openpbr-tables";


// -----------------------------------------------------------------------------
// -- albedo/radius -> extinction inversion
// -----------------------------------------------------------------------------
// converts openpbr's user-facing subsurface parameterization into the
// (sigmaAbsorption, sigmaScattering) pair a random walk integrates with
// (see util-vdb.glsl's utilVdbSigmaT / utilVdbAlbedo, same representation)

// floors anisotropy away from the g -> 1 similarity-relation singularity
// below (sigmaScattering = sigmaScatteringReduced / (1 - g) blows up as g
// approaches 1); openpbr's subsurface_scatter_anisotropy domain is [-1, 1]
public static const f32 skMaxSubsurfaceAnisotropy = 0.95f;

// mean free path floor: a channel radius of exactly 0 would send
// sigmaTReduced = 1/radius to infinity; flooring keeps that channel
// finite (effectively opaque -- absorbed at a very short distance)
// instead of propagating an inf into the similarity-relation divide below
public static const f32 skMinSubsurfaceRadius = 1e-5f;

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
public func openPbrSubsurfaceDiffusionReflectance(
	alphaReduced: f32v3,
) -> f32v3 {
	const f32v3 s = sqrt(3.0f * (1.0f - alphaReduced));
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
public func openPbrSubsurfaceInvertDiffusionReflectance(
	targetAlbedo: f32v3,
) -> f32v3 {
	f32v3 lo = f32v3(0.0f);
	f32v3 hi = f32v3(1.0f);
	for (i32 i = 0; i < 20; ++i) {
		const f32v3 mid = 0.5f * (lo + hi);
		const bool3 raise = (
			(openPbrSubsurfaceDiffusionReflectance(mid) < targetAlbedo)
		);
		lo = select(raise, mid, lo);
		hi = select((!(raise)), mid, hi);
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
public func openPbrSubsurfaceAlbedoToSigma(
	albedo: f32v3,
	radius: f32v3,
	anisotropy: f32,
	out sigmaAbsorption: f32v3,
	out sigmaScattering: f32v3,
) {
	const f32v3 safeRadius = max(radius, f32v3(skMinSubsurfaceRadius));
	const f32 g = min(anisotropy, skMaxSubsurfaceAnisotropy);
	const f32v3 alphaReduced = (
		openPbrSubsurfaceInvertDiffusionReflectance(clamp(albedo, 0.0f, 1.0f))
	);
	const f32v3 sigmaTReduced = 1.0f / safeRadius;
	const f32v3 sigmaSReduced = alphaReduced * sigmaTReduced;
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
public func openPbrSubsurfacePhaseEvalHg(
	cosTheta: f32,
	g: f32,
) -> f32 {
	const f32 g2 = g * g;
	const f32 denom = max(1.0f + g2 - 2.0f * g * cosTheta, 1e-7f);
	// (1 - g^2) / (4 \pi (1 + g^2 - 2 g \cos\theta)^{3/2})
	return (1.0f - g2) / (2.0f * TAU * denom * sqrt(denom));
}
// (TODO REVIEW)

public func openPbrSubsurfacePhaseSampleHgCos(
	u: f32,
	g: f32,
) -> f32 {
	// isotropic limit
	if (abs(g) < 1e-3f) {
		return 1.0f - 2.0f * u;
	}
	// standard HG inversion
	const f32 s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
	return (1.0f + g * g - s * s) / (2.0f * g);
}

// samples a re-aim direction around wi from the HG phase function
// (subsurfaceScatterAnisotropy as g); frisvad basis, same construction as
// utilVdbPhaseSampleMixture
public func openPbrSubsurfacePhaseSampleWo(
	wi: f32v3,
	anisotropy: f32,
	xi: f32v2,
) -> f32v3 {
	const f32 cosTheta = (
		openPbrSubsurfacePhaseSampleHgCos(xi.x, anisotropy)
	);
	const f32 phi = TAU * xi.y;
	return utilDirectionAboutAxis(wi, cosTheta, phi);
}

// one free-flight step landed on the medium boundary unscattered (t >=
// distanceMax): the caller's traced segment reaches the mesh exit point
public static const u32 OPENPBR_SUBSURFACE_WALK_EXIT = 0u;
// a real collision inside the medium; outWo/outDistance are both valid.
// absorption is not a separate outcome here -- see this function's header
// comment for why
public static const u32 OPENPBR_SUBSURFACE_WALK_SCATTER = 1u;
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
public func openPbrSubsurfaceChannelPdf(
	throughput: f32v3,
	albedo: f32v3,
) -> f32v3 {
	const f32v3 weights = abs(throughput * albedo);
	const f32 sumWeights = weights.x + weights.y + weights.z;
	if (sumWeights > 1e-8f) {
		return weights / sumWeights;
	}
	return f32v3(1.0f / 3.0f);
}

// cycles' volume_sample_channel: discrete inverse-cdf draw from
// channelPdf (must already sum to 1)
public func openPbrSubsurfaceSampleChannel(
	channelPdf: f32v3,
	inout state: u64,
) -> u32 {
	const f32 u = fnSampleUniform(state);
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

public func openPbrDwivediDiffusionLength(
	albedoMax: f32,
) -> f32 {
	const f32 a = clamp(albedoMax, 0.0f, 0.99999f);
	return (
		// 1 / \sqrt{1 - \alpha^(2.44294 - 0.0215813 * \alpha + 0.578637/\alpha)}
		1.0f / sqrt(1.0f - pow(a, 2.44294f - 0.0215813f * a + 0.578637f / a))
	);
}

public func openPbrDwivediPhaseEval(
	diffusionLength: f32,
	phaseLog: f32,
	cosTheta: f32,
) -> f32 {
	return (
		// p(cos \theta) = \frac{1}{((L - cos\theta) * \text{phase_log})}
		1.0f / ((diffusionLength - cosTheta) * phaseLog)
	);
}

public func openPbrDwivediPhaseSampleCos(
	diffusionLength: f32,
	phaseLog: f32,
	xi: f32,
) -> f32 {
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
public func openPbrSubsurfaceWalkStep(
	sigmaAbsorption: f32v3,
	sigmaScattering: f32v3,
	distanceMax: f32,
	wi: f32v3,
	anisotropy: f32,
	throughput: f32v3,
	guidingAxis: f32v3,
	out outDistance: f32,
	out outWo: f32v3,
	out outWeight: f32v3,
	inout state: u64,
) -> u32 {
	outWo = f32v3(0.0f);
	const f32v3 sigmaT = sigmaAbsorption + sigmaScattering;
	if (sigmaT.x <= 0.0f && sigmaT.y <= 0.0f && sigmaT.z <= 0.0f) {
		outWeight = f32v3(1.0f);
		outDistance = distanceMax;
		return OPENPBR_SUBSURFACE_WALK_EXIT;
	}
	const f32v3 albedo = sigmaScattering / max(sigmaT, f32v3(1e-8f));
	const f32v3 channelPdf = openPbrSubsurfaceChannelPdf(throughput, albedo);
	const u32 channel = openPbrSubsurfaceSampleChannel(channelPdf, state);
	// same clamp openPbrSubsurfaceAlbedoToSigma applied when building sigma
	// from this g -- the walk's phase function and its medium must agree on g
	const f32 g = min(anisotropy, skMaxSubsurfaceAnisotropy);

	// (TODO REVIEW)
	// -- dwivedi guiding. every quantity below collapses to the classical
	// one when guidedFraction is 0, so the disabled path stays exact
	const f32 diffusionLength = (
		openPbrDwivediDiffusionLength(max(albedo.x, max(albedo.y, albedo.z)))
	);
	const bool guidingEnabled = (
		dot(guidingAxis, guidingAxis) > 0.5f && diffusionLength > 1.0f
	);
	const f32 phaseLog = (
		guidingEnabled
		? log((diffusionLength + 1.0f) / (diffusionLength - 1.0f))
		: 0.0f
	);
	// the zero-variance solution assumes isotropic scattering, so fade the
	// guided strategy out as the phase function sharpens
	const f32 guidedFraction = (
		guidingEnabled ? 1.0f - max(0.5f, pow(abs(g), 0.125f)) : 0.0f
	);
	// already heading for the exit: stretch the free path. heading deeper
	// shortens it, so the walk can turn around sooner
	const f32 cosTravel = guidingEnabled ? dot(wi, guidingAxis) : 0.0f;
	const f32v3 sigmaTGuided = (
		max(
			sigmaT * (1.0f - cosTravel / max(diffusionLength, 1.0f)),
			f32v3(1e-8f)
		)
	);
	// strategy pick must precede the distance draw it modifies. the draw is
	// independent of the xi pair the direction uses
	const bool guided = (
		guidedFraction > 0.0f && fnSampleUniform(state) < guidedFraction
	);
	const f32v3 sigmaTUsed = guided ? sigmaTGuided : sigmaT;
	// (TODO REVIEW)

	const f32 sigmaTChannel = max(sigmaTUsed[channel], 1e-8f);
	const f32 t = -log(1.0f - fnSampleUniform(state)) / sigmaTChannel;
	if (t >= distanceMax) {
		// (TODO REVIEW)
		// both strategies could have reached the boundary, so the mis
		// denominator mixes both survival probabilities; the numerator stays
		// the physical transmittance
		const f32v3 survival = exp(-sigmaT * distanceMax);
		const f32v3 survivalGuided = exp(-sigmaTGuided * distanceMax);
		const f32 denom = (
			dot(channelPdf, lerp(survival, survivalGuided, guidedFraction))
		);
		// (TODO REVIEW)
		outWeight = survival / max(denom, 1e-8f);
		outDistance = distanceMax;
		return OPENPBR_SUBSURFACE_WALK_EXIT;
	}

	const f32v3 transmittance = exp(-sigmaT * t);
	// (TODO REVIEW)
	// mixture over both strategies regardless of which one drew t --
	// evaluating only the taken branch is what biases a guided walk
	const f32v3 density = sigmaT * transmittance;
	const f32v3 densityGuided = sigmaTGuided * exp(-sigmaTGuided * t);
	const f32 denom = (
		dot(channelPdf, lerp(density, densityGuided, guidedFraction))
	);
	// (TODO REVIEW)
	outWeight = (sigmaScattering * transmittance) / max(denom, 1e-8f);
	outDistance = t;
	const f32v2 xi = f32v2(fnSampleUniform(state), fnSampleUniform(state));
	// (TODO REVIEW)
	if (guided) {
		const f32 cosTheta = (
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
		const f32 phaseHg = openPbrSubsurfacePhaseEvalHg(dot(wi, outWo), g);
		const f32 pdfGuided = (
			openPbrDwivediPhaseEval(
				diffusionLength, phaseLog, dot(outWo, guidingAxis)
			) / TAU
		);
		outWeight *= (
			phaseHg / max(lerp(phaseHg, pdfGuided, guidedFraction), 1e-8f)
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
public func openPbrSubsurfaceEvaluateF(
	tables: MaterialTableHandles,
	mat: OpenPbrMaterial,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
	return f32v3(0.0f);
}

// entry direction: cosine-weighted hemisphere mirrored to point into the
// surface. shares glossy-diffuse's E_spec top layer (see
// openPbrSubsurfaceEvaluateF), so like diffuse it has no fresnel/ior
// bending of its own -- the walk itself is where scattering direction
// actually matters
public func openPbrSubsurfaceEntrySampleWo(
	nor: f32v3,
	xi: f32v2,
) -> f32v3 {
	return utilCosineHemisphereSampleWo(-nor, xi);
}

// mirror of openPbrGlossyDiffusePdf onto the lower hemisphere
public func openPbrSubsurfaceEntryPdf(
	dotNorWo: f32,
) -> f32 {
	return max(-dotNorWo, 0.0f) * IPI;
}

