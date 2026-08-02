module "util-regir-fill";
__exported import "util-types";
__exported import "util-emitter";
__exported import "util-regir";
// regir fill pass: the candidate loop that turns a cell into reservoirs.
//
// split out of util-regir.glsl because this needs the emitter sampler, and so
// transitively mor-shared.h and resolve_pc.h. the includer must already have
// pulled those in, exactly as util-emitter.glsl requires.
//
// -----------------------------------------------------------------------------
// -- review notes (verbose on purpose; trim once familiar)
// -----------------------------------------------------------------------------
//
// per cell, per reservoir: draw skRegirCandidateCount light samples from the
// existing flat power cdf, score each with the target function \hat{h}, and
// stream them through weighted reservoir sampling. the survivor plus its W is
// what shading later reads.
//
// the target function is *not* required to be correct, normalized, or even
// sensible -- any strictly positive function gives an unbiased estimator,
// because W divides \hat{h} back out. a better \hat{h} only means lower
// variance. that is worth internalizing: nothing here can be "wrong" in a way
// that biases the image, only in a way that makes it noisy. the things that
// *can* bias are elsewhere (a frozen grid, a light \hat{h} zeroes out that
// nothing else can reach).



// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- the target function
// -----------------------------------------------------------------------------

// rec709 luma; \hat{h} is a scalar score, so the emitter's rgb radiance has to
// collapse to one number
public func utilRegirLuminance(
	rgb: f32v3,
) -> f32 {
	return dot(rgb, f32v3(0.2126f, 0.7152f, 0.0722f));
}

// \hat{h}_N(x) = L_e * |cos(theta_L)| * cos(theta_s) / d^2
//
// evaluated at the cell's representative point, never at a shading point --
// that is the whole trade regir makes.
//
//  - L_e comes from the light list's precomputed factors rather than the
//    emitter's texture. cheap, and it cannot cost correctness: \hat{h} is a
//    steering function. the *shading* stage still evaluates the real material
//    at the sampled point, so a textured emitter stays consistent between the
//    two mis techniques.
//  - |cos(theta_L)| is absolute because emission is two-sided here, matching
//    the decision in plan-nee.md.
//  - cos(theta_s) is clamped, not absolute: a surface does not receive light
//    from behind it.
//  - in a medium there is no receiver normal, and cos(theta_s) degrades to 1.
//    it must NOT degrade to a dot product against a zero vector: that is a
//    silent zero, and a light that \hat{h} zeroes out everywhere simply
//    vanishes from the image with no error and no nan. plan-many-lights.md
//    trap 21
public func utilRegirTargetNonCanonical(
	lightRadiance: f32v3,
	lightNormal: f32v3,
	lightPosition: f32v3,
	cellPosition: f32v3,
	cellNormal: f32v3,
	hasNormal: bool,
) -> f32 {
	const f32v3 toLight = lightPosition - cellPosition;
	const f32 dist2 = dot(toLight, toLight);
	if (!(dist2 > 0.0f)) {
		return 0.0f;
	}
	const f32v3 direction = toLight * rsqrt(dist2);

	const f32 cosAtLight = abs(dot(lightNormal, -direction));
	const f32 cosAtCell = (
		hasNormal ? max(dot(cellNormal, direction), 0.0f) : 1.0f
	);

	return (
		utilRegirLuminance(lightRadiance) * cosAtLight * cosAtCell / dist2
	);
}

// -----------------------------------------------------------------------------
// -- candidate weight
// -----------------------------------------------------------------------------

// w = \hat{h}(X) / p_A(X), where p_A is the *area-measure* density the emitter
// sampler drew from.
//
// EmitterSample.pdf is a solid-angle density, and the conversion is
// p_A = p_omega * cos(theta_L) / d^2. substituting:
//
//     w = [ L_e * cos(theta_L) * cos(theta_s) / d^2 ]
//         / [ p_omega * cos(theta_L) / d^2 ]
//       = L_e * cos(theta_s) / p_omega
//
// cos(theta_L) and d^2 cancel **exactly**, in both sampling branches. that is
// worth forming directly rather than computing \hat{h} and p_A separately and
// dividing: the cancelled form has no grazing-angle singularity, which is
// precisely where the naive version is worst conditioned.
//
// a consequence worth knowing before measuring: because the hybrid sampler's
// p_omega is already solid-angle proportional, it is already distance-aware.
// so what ris adds over the current sampler here is mostly (a) the receiver
// cosine and (b) the emitter *pick* being reweighted by real geometry instead
// of power alone. the distance term is not new value. expect the visibility
// term (a later phase) to be where the bistro win actually comes from
public func utilRegirCandidateWeight(
	lightRadiance: f32v3,
	direction: f32v3,
	cellNormal: f32v3,
	hasNormal: bool,
	solidAnglePdf: f32,
) -> f32 {
	if (!(solidAnglePdf > 0.0f)) {
		return 0.0f;
	}
	const f32 cosAtCell = (
		hasNormal ? max(dot(cellNormal, direction), 0.0f) : 1.0f
	);
	return utilRegirLuminance(lightRadiance) * cosAtCell / solidAnglePdf;
}

// (TODO REVIEW)
// RECONSTRUCTED from the call shape -- was called by three shaders and
// defined nowhere. one stream per (slot, reservoir) per frame
public func utilRegirFillSeed(
	slot: u32,
	reservoirIndex: u32,
	frameIndex: u32,
) -> u64 {
	const u32 h0 = fnPcgHash(frameIndex * 0x9E3779B9u);
	const u32 h1 = fnPcgHash(slot + h0);
	const u32 h2 = fnPcgHash(reservoirIndex + h1);
	const u32 h3 = fnPcgHash(h2);
	return (u64(h2) << 32) | u64(h3);
}
// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- the candidate loop
// -----------------------------------------------------------------------------

public struct RegirFillResult {
	public GpuRegirReservoir reservoir;
	// (1/M) * sum(\hat{h}(X_i)/p(X_i)): this reservoir's monte carlo estimate
	// of \int \hat{h} dA. the caller reduces these across the cell's reservoirs
	// into GpuRegirCell.normalizationNonCanonical, which is what turns \hat{h}
	// into a pdf for the backward mis weight later.
	//
	// it comes free -- it is the same running sum W's numerator already needs
	public f32 normalizationEstimate;
};

// builds one reservoir for one cell.
//
// `hasNormal` is false for the medium bucket. `rngState` must already be
// seeded with the frame index folded in: without it every refill redraws the
// identical candidates and the grid never re-rolls the fill randomness that W's
// unbiasedness is defined over -- the pass would still run, the timings would
// still change, and the image would quietly stop converging
public func utilRegirFillReservoir(
	lights: GpuEmitterLight *,
	lightCount: u32,
	models: GpuResolveModelIndirect *,
	cellPosition: f32v3,
	cellNormal: f32v3,
	hasNormal: bool,
	inout rngState: u64,
) -> RegirFillResult {
	RegirReservoirState state = utilRegirReservoirInit();
	f32 normalizationAccum = 0.0f;

	for (u32 i = 0u; i < skRegirCandidateCount; ++i) {
		// all five drawn unconditionally, before any early-out, so the stream
		// length does not depend on whether a candidate happened to be valid.
		// a data-dependent rng stream is reproducible but not comparable
		// between two runs that differ only in scene content, which makes
		// a/b testing the target function impossible
		const f32 xiPick = fnSampleUniform(rngState);
		const f32 xiTri = fnSampleUniform(rngState);
		const f32 xiU = fnSampleUniform(rngState);
		const f32 xiV = fnSampleUniform(rngState);
		const f32 xiAccept = fnSampleUniform(rngState);

		const EmitterPick pick = utilEmitterPick(lights, lightCount, xiPick);
		// the environment entry lives in the same list but has no surface, so
		// it can never be a regir candidate; it keeps its own top-level
		// power-weighted branch in the caller
		if (!pick.valid || lights[pick.lightIndex].kind != EMITTER_KIND_MESH) {
			state.sampleCount += 1u;
			continue;
		}

		const EmitterSample es = (
			utilEmitterSampleFromPick(
				lights, pick, models, cellPosition, f32v3(xiTri, xiU, xiV)
			)
		);
		if (!es.valid) {
			state.sampleCount += 1u;
			continue;
		}

		const f32v3 radiance = lights[pick.lightIndex].radiance;
		const f32 weight = (
			utilRegirCandidateWeight(
				radiance, es.direction, cellNormal, hasNormal, es.pdf
			)
		);
		// \hat{h}(X) itself, needed because the survivor has to remember its
		// own target value for the finalize. computed explicitly rather than
		// recovered from `weight` so the two forms cannot drift apart
		const f32 target = (
			utilRegirTargetNonCanonical(
				radiance, es.normal, es.position,
				cellPosition, cellNormal, hasNormal
			)
		);

		normalizationAccum += weight;
		utilRegirReservoirUpdate(
			state,
			es.lightIndex,
			es.primitiveIndex,
			f32v2(es.barycentric.y, es.barycentric.z),
			weight,
			target,
			xiAccept
		);
	}

	RegirFillResult result;
	result.reservoir = utilRegirReservoirFinalize(state);
	result.normalizationEstimate = (
		state.sampleCount > 0u
			? normalizationAccum / f32(state.sampleCount)
			: 0.0f
	);
	return result;
}

// (TODO REVIEW)

