module "util-regir";
__exported import "util-types";
__exported import "util-random";
__exported import "util-regir-shared";
// regir world-space light-selection grid: key derivation, the open-addressed
// cell table, the reservoir primitives, and the shading-time selection.
//
// the candidate loop that actually fills reservoirs lives in
// util-regir-fill.glsl, because it needs the emitter sampler (and therefore
// mor-shared.h / resolve_pc.h) and this file must stay includable without
// them -- the tier-9 grid tests include it on its own.
//
// -----------------------------------------------------------------------------
// -- what this is, in one place (review notes; trim once it is familiar)
// -----------------------------------------------------------------------------
//
// ordinary nee picks a light with probability p and divides by p:
//
//     contribution = f * L_e * G / p
//
// the division is what makes the estimator unbiased. today p is the flat power
// cdf, which knows an emitter's power and nothing about where the shading point
// is standing.
//
// resampled importance sampling (ris) samples from a *better* distribution
// without ever having to normalize it. draw M candidates from the cheap p,
// score each with a target function \hat{h} (the smart one: radiance, distance,
// both cosines, eventually visibility), then keep one candidate with
// probability proportional to its score. the survivor is distributed roughly
// like \hat{h}.
//
// the catch: you no longer know the survivor's exact probability, so you cannot
// divide by it. ris instead hands back a weight W that does the same job:
//
//     W = (1/M) * sum_i(\hat{h}(X_i)/p(X_i)) / \hat{h}(Y)      E[W] = 1/p_true(Y)
//
// so `contribution = f * L_e * G * W`. **W is the 1/p you could not compute.**
// anywhere ordinary nee divides by a pdf, this multiplies by W. there is no
// division by a light pdf anywhere in the regir path -- a stray `/ pdf` is a
// good thing to grep for when reviewing.
//
// a "reservoir" is just the streaming trick for "keep one proportional to
// score" without storing all M candidates: hold one survivor plus a running
// weight sum, and let each new candidate replace the survivor with probability
// w_i / runningSum.
//
// regir's own contribution is only this: do the M-candidate work *once per
// grid cell* rather than once per shading point, store the survivors, and let
// every path vertex that lands in that cell reuse them. one cell holds
// skRegirReservoirCount independent survivors, each with its own W.
//
// the one thing that is genuinely subtle: W's unbiasedness is an expectation
// **over grid realizations**, not over shading draws. averaging many shading
// samples against a *frozen* grid converges to the wrong answer and does not
// improve with spp. that is why the fill pass must be re-seeded per frame.
// see plan-many-lights.md section 6.1.



// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- key derivation
// -----------------------------------------------------------------------------

// floor, not a truncating cast: (int)(x / cellSize) rounds toward zero, which
// merges [-1,0) with [0,1) and leaves one double-width cell straddling the
// origin on every axis
public func utilRegirGridCoord(
	worldPosition: f32v3,
	cellSize: f32,
) -> i32v3 {
	return i32v3(floor(worldPosition / cellSize));
}

// encodes a spherical normal into a [0, 1]^2 octahedron map. only ever used
// for bucketing, so there is no matching decode
public func utilRegirOctahedronNormalEncode(
	normal: f32v3,
) -> f32v2 {
	const f32v3 nv = (
		normal / (abs(normal.x) + abs(normal.y) + abs(normal.z))
	);
	f32v2 oct = nv.xy;
	if (nv.z < 0.0f) {
		oct = (
			(1.0f - abs(oct.yx))
			* f32v2(
				nv.x >= 0.0f ? 1.0f : -1.0f,
				nv.y >= 0.0f ? 1.0f : -1.0f
			)
		);
	}
	return oct * 0.5f + 0.5f;
}

public func utilRegirNormalBucket(
	normal: f32v3,
) -> u32 {
	const f32v2 oct = utilRegirOctahedronNormalEncode(normal);
	const u32 x = (
		min(
			u32(oct.x * f32(skRegirNormalResolution)),
			skRegirNormalResolution - 1u
		)
	);
	const u32 y = (
		min(
			u32(oct.y * f32(skRegirNormalResolution)),
			skRegirNormalResolution - 1u
		)
	);
	return y * skRegirNormalResolution + x;
}

public func utilRegirCellHashKey(
	key: GpuRegirCellKey,
) -> u32 {
	u32 h = fnPcgHash(u32(key.gridCoord.x));
	h = fnPcgHash(h ^ u32(key.gridCoord.y));
	h = fnPcgHash(h ^ u32(key.gridCoord.z));
	h = fnPcgHash(h ^ key.normalBucket);
	// 0 is reserved as the empty-slot marker, so the hash must never be 0
	return h | 1u;
}

public func utilRegirKeyEqual(
	a: GpuRegirCellKey,
	b: GpuRegirCellKey,
) -> bool {
	return all(a.gridCoord == b.gridCoord) && a.normalBucket == b.normalBucket;
}

// builds the key a shading point looks up with. a medium scatter has no
// surface normal, so it takes the reserved bucket and shares the grid with
// surface vertices without ever colliding into one of their cells
public func utilRegirKeyFor(
	worldPosition: f32v3,
	normal: f32v3,
	hasNormal: bool,
	cellSize: f32,
) -> GpuRegirCellKey {
	GpuRegirCellKey key;
	key.gridCoord = utilRegirGridCoord(worldPosition, cellSize);
	key.normalBucket = (
		hasNormal ? utilRegirNormalBucket(normal) : skRegirNormalBucketNone
	);
	return key;
}

// -----------------------------------------------------------------------------
// -- open addressing
// -----------------------------------------------------------------------------

// cellCount is a power of two so the probe walk masks instead of dividing;
// a non-constant integer modulo in this loop is a real stall
public func utilRegirSlotAt(
	hashKey: u32,
	probe: u32,
	cellCount: u32,
) -> u32 {
	return (hashKey + probe) & (cellCount - 1u);
}

// claims a slot for `key`, or returns the slot another thread already claimed
// for the same key. skRegirSlotInvalid when the probe budget is exhausted,
// which the caller answers with the flat-cdf fallback rather than by growing
// the table
public func utilRegirInsert(
	guards: u32 *,
	cells: GpuRegirCell *,
	cellCount: u32,
	key: GpuRegirCellKey,
	out probes: u32,
) -> u32 {
	// glsl parameters are mutable copies; slang's are not
	var guards = guards;
	// glsl parameters are mutable copies; slang's are not
	var cells = cells;
	const u32 hashKey = utilRegirCellHashKey(key);
	for (probes = 0u; probes < skRegirMaxProbesPerCell; ++probes) {
		const u32 slot = utilRegirSlotAt(hashKey, probes, cellCount);
		const u32 prev = atomicCompSwap(&guards[slot], 0u, hashKey);
		if (prev == 0u) {
			cells[slot].key = key;
			return slot;
		}
		// a matching guard is only a hash match; the exact key decides, so a
		// collision can never silently hand back another cell's reservoirs
		if (prev == hashKey && utilRegirKeyEqual(cells[slot].key, key)) {
			return slot;
		}
	}
	return skRegirSlotInvalid;
}

// finds `key` without claiming anything. stops at the first empty guard:
// linear probing never skips, so an empty slot proves the key was never
// inserted
public func utilRegirLookup(
	guards: u32 *,
	cells: GpuRegirCell *,
	cellCount: u32,
	key: GpuRegirCellKey,
	out probes: u32,
) -> u32 {
	// glsl parameters are mutable copies; slang's are not
	var guards = guards;
	// glsl parameters are mutable copies; slang's are not
	var cells = cells;
	const u32 hashKey = utilRegirCellHashKey(key);
	for (probes = 0u; probes < skRegirMaxProbesPerCell; ++probes) {
		const u32 slot = utilRegirSlotAt(hashKey, probes, cellCount);
		const u32 guard = guards[slot];
		if (guard == 0u) {
			return skRegirSlotInvalid;
		}
		if (guard == hashKey && utilRegirKeyEqual(cells[slot].key, key)) {
			return slot;
		}
	}
	return skRegirSlotInvalid;
}

// -----------------------------------------------------------------------------
// -- reservoir (weighted reservoir sampling)
// -----------------------------------------------------------------------------

// the in-flight form. GpuRegirReservoir is the *stored* form: weightSum and
// sampleCount are consumed by the finalize step and never written out, because
// nothing downstream needs them once unbiasedWeight exists
public struct RegirReservoirState {
	public u32 lightIndex;
	public u32 primitiveIndex;
	// (b1, b2); b0 is implied
	public f32v2 barycentric;
	// sum of every candidate weight seen so far. this is the numerator of W,
	// and separately it is the monte carlo estimate of \int \hat{h} dA that the
	// cell's normalization factor needs -- the same sum, for free
	public f32 weightSum;
	// \hat{h} of the *current survivor*, not of the last candidate
	public f32 targetValue;
	// M: how many candidates were streamed, including rejected ones
	public u32 sampleCount;
};

public func utilRegirReservoirInit(

) -> RegirReservoirState {
	RegirReservoirState res;
	res.lightIndex = 0u;
	res.primitiveIndex = 0u;
	res.barycentric = f32v2(0.0f);
	res.weightSum = 0.0f;
	res.targetValue = 0.0f;
	res.sampleCount = 0u;
	return res;
}

// streams one candidate. `weight` is \hat{h}(X)/p(X) and `target` is
// \hat{h}(X); both are needed because the survivor has to remember its own
// \hat{h} for the finalize step.
//
// the acceptance test is written `xi * weightSum < weight` rather than
// `xi < weight / weightSum`: identical in exact arithmetic, but it avoids a
// division and behaves when weightSum is denormal
public func utilRegirReservoirUpdate(
	inout res: RegirReservoirState,
	lightIndex: u32,
	primitiveIndex: u32,
	barycentric: f32v2,
	weight: f32,
	target: f32,
	xi: f32,
) {
	res.sampleCount += 1u;
	if (!(weight > 0.0f)) {
		// a zero-weight candidate still counts toward M -- dropping it from the
		// count would bias the 1/M in the finalize
		return;
	}
	res.weightSum += weight;
	if (xi * res.weightSum < weight) {
		res.lightIndex = lightIndex;
		res.primitiveIndex = primitiveIndex;
		res.barycentric = barycentric;
		res.targetValue = target;
	}
}

// W = weightSum / (M * \hat{h}(Y)).
//
// the 1/M is folded in here rather than into each candidate weight so the
// streaming loop never needs to know M up front. the targetValue guard is
// load-bearing: a survivor with \hat{h} == 0 must produce W == 0, not a nan
// that propagates into the image and is then very hard to trace back here
public func utilRegirReservoirFinalize(
	res: RegirReservoirState,
) -> GpuRegirReservoir {
	GpuRegirReservoir out_;
	out_.lightIndex = res.lightIndex;
	out_.primitiveIndex = res.primitiveIndex;
	out_.barycentric = res.barycentric;
	out_.targetValue = res.targetValue;
	out_.unbiasedWeight = (
		(res.targetValue > 0.0f && res.sampleCount > 0u)
			? res.weightSum / (f32(res.sampleCount) * res.targetValue)
			: 0.0f
	);
	return out_;
}

// -----------------------------------------------------------------------------
// -- shading-time selection
// -----------------------------------------------------------------------------

public struct RegirSelection {
	public bool valid;
	public u32 lightIndex;
	public u32 primitiveIndex;
	// (b0, b1, b2), matching EmitterSample's convention rather than the
	// stored two-component form
	public f32v3 barycentric;
	// multiply the contribution by this. there is no pdf to divide by
	public f32 unbiasedWeight;
};

public func utilRegirSelectionInvalid(

) -> RegirSelection {
	RegirSelection sel;
	sel.valid = false;
	sel.lightIndex = 0u;
	sel.primitiveIndex = 0u;
	sel.barycentric = f32v3(0.0f);
	sel.unbiasedWeight = 0.0f;
	return sel;
}

// picks one of the cell's stored light samples for a shading point.
//
// this is the simplest correct version and is deliberately so. the cell holds
// skRegirReservoirCount independent ris estimates, each already carrying a
// valid W, so choosing one uniformly and using its W is **unbiased** --
// E[f * W] is the integral we want, as long as the cell's proposal covers every
// light that can contribute here (which is what the canonical reservoirs in a
// later phase are for).
//
// what this does *not* do, on purpose: resample several reservoirs against the
// real integrand at this exact point. that is a variance reduction, not a
// correctness requirement, and layering it on before the mean is proven makes
// a wrong mean much harder to attribute.
//
// note the reservoirs were built for the cell's *representative* point, not for
// this one. that costs variance and not correctness: any proposal with the
// right support gives an unbiased estimate, it is just a worse proposal the
// further this point is from the representative.
//
// `slot` comes out so the caller can put it in path state. the backward mis
// weight at the next vertex has to price the same cell this vertex sampled
// from, and re-deriving the key there is not guaranteed to reproduce it: the
// lookup is jittered, and gpu division is not correctly rounded so a position
// on a cell boundary can quantize either way (measured -- see the tier 9
// coordinate test). plan-many-lights.md trap 20
public func utilRegirSelect(
	guards: u32 *,
	cells: GpuRegirCell *,
	reservoirs: GpuRegirReservoir *,
	cellCount: u32,
	cellSize: f32,
	shadingPosition: f32v3,
	shadingNormal: f32v3,
	hasNormal: bool,
	inout rngState: u64,
	out slot: u32,
) -> RegirSelection {
	// glsl parameters are mutable copies; slang's are not
	var guards = guards;
	// glsl parameters are mutable copies; slang's are not
	var cells = cells;
	// glsl parameters are mutable copies; slang's are not
	var reservoirs = reservoirs;
	RegirSelection sel = utilRegirSelectionInvalid();
	slot = skRegirSlotInvalid;

	const GpuRegirCellKey key = (
		utilRegirKeyFor(shadingPosition, shadingNormal, hasNormal, cellSize)
	);

	u32 probes = 0u;
	slot = utilRegirLookup(guards, cells, cellCount, key, probes);
	if (slot == skRegirSlotInvalid) {
		// no cell here (cold, evicted, or probe budget exhausted). the caller
		// must fall back to the flat power cdf -- and that vertex's mis weight
		// then has to use *that* density, not regir's. two selection strategies
		// in one renderer means two densities, and using the wrong one is a
		// silent bias. plan-many-lights.md trap 22
		return sel;
	}

	// one draw, uniform over the cell's reservoirs. the pick probability is
	// 1/skRegirReservoirCount and it does *not* appear in W: each reservoir is
	// already an unbiased estimate on its own, so picking among them uniformly
	// averages unbiased estimates rather than compounding a selection density
	const u32 pick = (
		min(
			u32(fnSampleUniform(rngState) * f32(skRegirReservoirCount)),
			skRegirReservoirCount - 1u
		)
	);
	const GpuRegirReservoir res = (
		reservoirs[slot * skRegirReservoirCount + pick]
	);

	// a reservoir that never took a valid candidate finalizes to weight 0.
	// written as !(x > 0) rather than (x <= 0) so a nan is caught here rather
	// than downstream
	if (!(res.unbiasedWeight > 0.0f)) {
		return sel;
	}

	sel.valid = true;
	sel.lightIndex = res.lightIndex;
	sel.primitiveIndex = res.primitiveIndex;
	sel.barycentric = f32v3(
		1.0f - res.barycentric.x - res.barycentric.y,
		res.barycentric.x,
		res.barycentric.y
	);
	sel.unbiasedWeight = res.unbiasedWeight;
	return sel;
}

// (TODO REVIEW)

