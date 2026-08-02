implementing light;

import types;
import model;
import util;

// implements ReGIR (Reservoir-based Grid Illumination with Reprojection)
// The idea is that, for NEE with a light sample, the contribution of a sample
// is:
// (f · L · G) / pdf(sample),
// where,
// f is Bsdf, L is radiance and G is the geometrical term
//
// When picking a light to sample from, the simple solution is to weight light
// selection based off its radiance, however the better light selection would be
// ones with a probability density proportional to the contribution of the light
// sample.
//
// The selection of light samples is cached in a spatial data structure, each
// sample stored in a reservoir with an associated weight and target function
// values.
//
// Resampled importance sampling:
///  1) sample M candidates with a cheap probability function 'p'
//   2) each candidate computes a target function 'ĥ' which is an unnormalized
//      estimate of its contribution
//   3) one candidate is selected, chosen by its probability proportional to
//      ĥ / p
//   4) finally the weight is computed as:
//      W = (1/M) · Σ ĥ(Xᵢ)/p(Xᵢ) / ĥ(Y)
// where,
//
// - p
//  cheap probability function as an area measure, for a mesh:
//  power CDF, a random triangle, then a random point on the triangle
//
// - ĥ
//  The target function, an unnormalized estimate of the contribution of a
//  light sample at a cell. There are two variants, the non-canonical:
//     ĥ = luma(L) · |cos(θ_L)| · cos(θ_s) / d²
//  It might not have any contribution at the representative point, but
//  neighboring shading surfels could have contribution. Then the canonical
//  target function:
//     ĥC = brightness / d²
//  Which will always have a contribution. The canonical function can simply be
//  the existing power CDF, which will be mentioned later.
//
//  Apparently the term 'canonical' comes from GRIS: a sampling technique that
//  covers an entire domain. Anyways, both target functions are combined with
//  MIS weights.
//
// - M
//  The number of candidates streamed into a reservoir
//
// - W
//  The the weight of the selected candidate, replaces 1/pdf(sample)
//
// for MIS, another important property is the normalization factor,
//   Norm = (1/M) · Σ ĥ(Xᵢ)/p(Xᵢ), which is the monte carlo estimate of ∫ĥ,
//   the overall contribution of light samples in the cell.
//   ĥ(sample) / Norm is the approximate pdf.
//   So when a BSDF ray hits a light, this provides the density NEE would have
//   had for that point, mirroring typical NEE probability density
//
// The MIS weight is then computed as:
//
//  with probability α, pick a Canonical reservoir
//  with probability 1-α, pick a Non-Canonical reservoir
//
// however one known shortcut is that the canonical function can be computed
// on-the-fly using the power CDF, which gives an exact PDF. So reservoirs
// might only need to store the non-canonical version, though for my own
// implementation I'm unsure yet which I'll land on, apparently this shortcut
// produces higher variance.
//
// -- density approximations
//  p̂_N(x) = ĥ_N(x) / Norm_N
//  p̂_C(x) = ĥ_C(x) / Norm_C
// -- balance heuristic; π_N = 1-α, π_C = α
//  mₜ(x) = πₜ · p̂ₜ(x) / ( (1-α)·p̂_N(x) + α·p̂_C(x) )
// -- contribution
//  contribution = f(Y) · L · G · W(Y) · mₜ(Y) / πₜ
// where Y is the selected candidate
//
// the technique, despite being an approximation, is unbiased as the two weights
// sum to 1, and each technique itself is unbiased.
//
// ReGIR uses reservoirs to store a slice of light samples in the cell rather
// than all of them, the survivors chosen from RIS. A reservoir is owned by
// a cell which contains the meta-data of the reservoir.
//
// In a regirCell at a slot, the following is stored:
// - cell key (xyz + normal bucket)
// - representative position and normal (where ĥ was evaluated)
// - Normalization factor (canonical and non-canonical)
// - last fill index (used to determine if the cell needs to be refilled)
//
// The regirCell owns the regirReservoirs, which contain a list of candidates:
// the light sample:
//   - light index
//   - primitive index
//   - parameters on the surface (barycentrics for triangle)
// and the weights:
//   - weight of the sample W
//   - target function ĥ(sample)
//
// regir-fill is the shader that fills the cells and reservoirs,
// meanwhile this file contains the data structures and functions to
// manage and query the regir cells and reservoirs, as well as the math to
// compute the RIS and target function.
//
// the grid is filled with keys by the wavefront-integrate, such that
// future regir-fill calls will populate the requested reservoirs with
// light samples. Essentially wavefront-integrate will claim a slot for a cell,
// then regir-fill will fill the claimed cell with light samples.
//
// claimed slots will be staled out over time such that they have a sentinel
// value which indicates they can be refilled.

namespace vkof {

// -----------------------------------------------------------------------------
// -- regir key
// -----------------------------------------------------------------------------

// returns the regir grid coordinate of a position in world space
public func regirGridCoord(position: f32v3, cellSize: f32) -> i32v3 {
	return i32v3(floor(position / cellSize));
}

// octahedral normal bucket index
public func regirNormalBucket(normal: f32v3) -> u32 {
	// -- normal projected onto the octahedron
	let octahedralNormal: f32v3 = (
		normal / (abs(normal.x) + abs(normal.y) + abs(normal.z))
	);
	var oct: f32v2 = octahedralNormal.xy;
	// -- fold the lower hemisphere into the upper
	if (octahedralNormal.z < 0.0f) {
		oct = (
			(1.0f - abs(oct.yx))
			* f32v2(vkof::signNonZero(oct.x), vkof::signNonZero(oct.y))
		);
	}
	let uv: f32v2 = (oct*f32v2(0.5f)) + f32v2(0.5f);
	let resolution: f32 = f32(vkof::skRegirNormalResolution);
	let x: u32 = min(u32(uv.x * resolution), vkof::skRegirNormalResolution - 1u);
	let y: u32 = min(u32(uv.y * resolution), vkof::skRegirNormalResolution - 1u);
	return (y * vkof::skRegirNormalResolution) + x;
}

// offsets position by up to half a cell per axis
public func regirJitteredPosition(position: f32v3) -> f32v3 {
	if (vkof::global.regirJitter == 0u) { return position; }
	let bits: u32v3 = asuint(position);
	var state: u32 = (
		implRegirPcgHash(
			bits.x
			^ implRegirPcgHash(
				bits.y
				^ implRegirPcgHash(
					bits.z ^ implRegirPcgHash(vkof::global.frameIndex)
				)
			)
		)
	);
	let ξx: f32 = implRegirNextF32(state);
	let ξy: f32 = implRegirNextF32(state);
	let ξz: f32 = implRegirNextF32(state);
	return (
		position + (f32v3(ξx, ξy, ξz) - 0.5f) * vkof::global.regirCellSize
	);
}

// returns the regir cell key for a position and normal
public func regirKeyFor(
	position: f32v3,
	normal: Optional<f32v3>,
) -> vkof::RegirCellKey
{
	var key: vkof::RegirCellKey;
	key.gridCoord = regirGridCoord(position, vkof::global.regirCellSize);
	key.normalBucket = (
		normal.hasValue
		? regirNormalBucket(normal.value)
		: vkof::skRegirNormalBucketNone
	);
	return key;
}

// returns the hash of a regir cell key
public func regirHashKey(key: vkof::RegirCellKey) -> u32 {
	var h: u32 = implRegirPcgHash(u32(key.gridCoord.x));
	h = implRegirPcgHash(h ^ u32(key.gridCoord.y));
	h = implRegirPcgHash(h ^ u32(key.gridCoord.z));
	h = implRegirPcgHash(h ^ key.normalBucket);
	// 0 is reserved for empty-guard marker
	return h | 1u;
}

// returns true if two regir cell keys are equal
public func regirKeyEqual(
	a: vkof::RegirCellKey,
	b: vkof::RegirCellKey,
) -> bool
{
	return (
		all(a.gridCoord == b.gridCoord)
		&& a.normalBucket == b.normalBucket
	);
}

// -----------------------------------------------------------------------------
// -- regir addressing
// -----------------------------------------------------------------------------

public func regirLookup(key: vkof::RegirCellKey) -> u32 {
	let hash: u32 = regirHashKey(key);
	// -- linear probe for the key in the array
	for (u32 probe = 0; probe < vkof::skRegirMaxProbesPerCell; ++probe) {
		let slot: u32 = implRegirSlotAt(hash, probe);
		// -- check if the slot is empty
		let guard: u32 = vkof::global.regirGuards[slot];
		if (guard == 0u) { return vkof::skRegirSlotInvalid; }
		// -- check if the slot matches the key
		if (
			guard == hash
			&& regirKeyEqual(vkof::global.regirCells[slot].key, key)
		) {
			implRegirMarkUsed(slot);
			return slot;
		}
	}
	// -- no matching slot found
	return vkof::skRegirSlotInvalid;
}

// finds and inserts a regir cell into the global array, returns the slot index
public func regirFindOrInsert(
	key: vkof::RegirCellKey,
	position: f32v3,
	normal: Optional<f32v3>,
) -> u32 {
	let hash: u32 = regirHashKey(key);
	// -- linear probe for the key
	for (u32 probe = 0u; probe < vkof::skRegirMaxProbesPerCell; ++probe) {
		let slot: u32 = implRegirSlotAt(hash, probe);
		var guard: u32 = vkof::global.regirGuards[slot];
		// -- claim the first empty or evicted slot
		if (guard == 0u || guard == vkof::skRegirSlotEvicted) {
			if (implRegirClaim(slot, guard, hash, key, position, normal)) {
				return slot;
			}
			// claim failed from another thread
			guard = vkof::global.regirGuards[slot];
		}
		// -- check if the slot matches the key
		if (
			guard == hash
			&& regirKeyEqual(vkof::global.regirCells[slot].key, key)
		) {
			implRegirMarkUsed(slot);
			return slot;
		}
	}
	// -- all slots are full
	return vkof::skRegirSlotInvalid;
}

// replaces the cell's next representative with the given position and normal
public func regirRecordShadingPoint(
	slot: u32,
	position: f32v3,
	normal: f32v3,
) {
	// a moving camera or paused timeline would flicker
	if (vkof::global.regirRefresh == 0u) { return; }
	vkof::global.regirCells[slot].pendingOri = (
		position + normal * vkof::rayEpsilon(position)
	);
	vkof::global.regirCells[slot].pendingNor = normal;
}

// true if the cell is older than regirMaxAge and should be evicted
public func regirCellExpired(cell: vkof::RegirCell) -> bool {
	return (
		vkof::global.regirMaxAge > 0u
		&& (
			vkof::global.frameIndex - cell.frameIndexLastUsed
			> vkof::global.regirMaxAge
		)
	);
}

// -----------------------------------------------------------------------------
// -- weighted reservoir sampling
// -----------------------------------------------------------------------------

// an in-flight state of a reservoir
public struct RegirReservoirState
{
	public u32 lightIndex;
	public u32 primitiveIndex;
	// parameters on surface, e.g. barycentrics for triangle
	public f32v2 parameters;
	// sum of candidate weights
	public f32 weightSum;
	// ĥ of the current surviving candidate
	public f32 targetValue;
	// M including rejected candidates
	public u32 sampleCount;
	// position
	public f32v3 position;
};

public func regirReservoirStateInit() -> RegirReservoirState {
	var reservoir: vkof::RegirReservoirState;
	reservoir.lightIndex = 0u;
	reservoir.primitiveIndex = 0u;
	reservoir.parameters = f32v2(0.0f);
	reservoir.weightSum = 0.0f;
	reservoir.targetValue = 0.0f;
	reservoir.sampleCount = 0u;
	reservoir.position = f32v3(0.0f);
	return reservoir;
}

// updates a reservoir with a new candidate sample
public func regirReservoirUpdate(
	inout reservoir: vkof::RegirReservoirState,
	position: f32v3,
	lightIndex: u32,
	primitiveIndex: u32,
	parameters: f32v2,
	weight: f32,
	target: f32,
	ξ: f32,
) {
	reservoir.sampleCount += 1u;
	// -- ignore zero-weight candidates
	if (!(weight > 0.0f)) { return; }
	// -- update the sum of weights
	reservoir.weightSum += weight;
	// -- compute the probability of selecting this candidate
	if (ξ * reservoir.weightSum < weight) {
		reservoir.position = position;
		reservoir.lightIndex = lightIndex;
		reservoir.primitiveIndex = primitiveIndex;
		reservoir.parameters = parameters;
		reservoir.targetValue = target;
	}
}


// finalizes a reservoir and writes it to the global array
public func regirReservoirFinalize(
	reservoir: vkof::RegirReservoirState,
) -> vkof::RegirReservoir
{
	var result: vkof::RegirReservoir;
	result.lightIndex = reservoir.lightIndex;
	result.primitiveIndex = reservoir.primitiveIndex;
	result.parameters = reservoir.parameters;
	result.targetValue = reservoir.targetValue;
	// W = weightSum / (M * ĥ(Y))
	// if ĥ(Y) = 0, then W = 0
	result.weight = (
		reservoir.targetValue > 0.0f && reservoir.sampleCount > 0u
		? reservoir.weightSum / (f32(reservoir.sampleCount)*reservoir.targetValue)
		: 0.0f
	);
	return result;
}

// -----------------------------------------------------------------------------
// -- target function
// -----------------------------------------------------------------------------

public func regirTargetNonCanonical(
	light: vkof::SceneEmitterLight,
	lightPosition: f32v3,
	lightNor: f32v3,
	position: f32v3,
	normal: Optional<f32v3>,
) -> f32
{
	// ĥ_N = luma(L_e) · |cos(θ_L)| · cos(θs) / d²
	let toLight: f32v3 = lightPosition - position;
	let d2: f32 = dot(toLight, toLight);
	if (d2 <= 0.0f) { return 0.0f; }
	// -- compute angles
	let direction: f32v3 = toLight * rsqrt(d2);
	// -- emission is two-sided
	let cosLight: f32 = abs(dot(lightNor, -direction));
	let cosCell: f32 = (
		normal.hasValue
		? max(dot(normal.value, direction), 0.0f)
		: 1.0f
	);
	// -- compute the luma of the emitter's radiance
	// TODO move to function
	let luma: f32 = dot(light.radiance, f32v3(0.2126f, 0.7152f, 0.0722f));
	return luma * cosLight * cosCell / d2;
}

// target non-canonical function for regir
// the returned value is proportional to the contribution of light sampled
// on the cell at the given position and normal
public func regirTargetNonCanonicalWithCell(
	light: vkof::SceneEmitterLight,
	lightPosition: f32v3,
	lightNor: f32v3,
	cell: vkof::RegirCell,
) -> f32
{
	let normal: Optional<f32v3> = (
		(cell.key.normalBucket == vkof::skRegirNormalBucketNone)
		? Optional<f32v3>(none)
		: Optional<f32v3>(cell.representativeNor)
	);
	return (
		regirTargetNonCanonical(
			/*light=*/ light,
			/*lightPosition=*/ lightPosition,
			/*lightNor=*/ lightNor,
			/*position=*/ cell.representativeOri,
			/*normal=*/ normal,
		)
	);
}

// -----------------------------------------------------------------------------
// -- fill
// -----------------------------------------------------------------------------

public struct RegirFillResult {
	public vkof::RegirReservoir reservoir;
	// estimate of the normalization factor, (1/M) * sum(ĥ / p)
	public f32 normalizationEstimate;
	public f32v3 position;
};

// fills a regir cell with a reservoir of light samples
public func regirFillReservoir(
	cell: vkof::RegirCell,
	slot: u32,
	reservoirIndex: u32,
) -> RegirFillResult
{
	// -- remove environment from light selection
	// environment is the last entry so slicing its probability off will avoid it
	let envPdf: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
	let meshMass: f32 = 1.0f - envPdf;
	// -- initialize the reservoir state
	var rng: u32 = implRegirFillSeed(slot, reservoirIndex);
	var state: vkof::RegirReservoirState = vkof::regirReservoirStateInit();
	var normalizationSum: f32 = 0.0f;
	// -- sample candidates
	for (u32 i = 0u; i < vkof::skRegirCandidateCount; ++i) {
		// -- initialize random vars
		let ξSelect: f32 = implRegirNextF32(rng);
		let ξPoint: f32v3 = (
			f32v3(
				implRegirNextF32(rng),
				implRegirNextF32(rng),
				implRegirNextF32(rng),
			)
		);
		let ξAccept: f32 = implRegirNextF32(rng);
		// -- sample a light from the scene emitter list
		let selection: vkof::EmitterSelection = (
			vkof::emitterSelection(ξSelect * meshMass)
		);
		let sample: vkof::EmitterSample = (
			vkof::emitterSampleFromSelection(selection, ξPoint)
		);
		// -- check if sample is valid
		if (
			sample.kind != vkof::EmitterKind::SceneEmitter
			|| sample.pdf <= 0.0f
		) {
			state.sampleCount += 1u;
			continue;
		}

		// -- compute the target function for this sample
		let light: vkof::SceneEmitterLight = (
			vkof::global.sceneEmitterLights[selection.lightIndex]
		);
		let target: f32 = (
			regirTargetNonCanonicalWithCell(
				/*light=*/ light,
				/*lightPosition=*/ sample.position,
				/*lightNor=*/ sample.normal,
				/*cell=*/ cell
			)
		);
		// -- compute area densities
		let pdfArea: f32 = sample.pdf / meshMass;
		let weight: f32 = target / pdfArea;
		normalizationSum += weight;
		regirReservoirUpdate(
			/*reservoir=*/ state,
			/*position=*/ sample.position,
			/*lightIndex=*/ selection.lightIndex,
			/*primitiveIndex=*/ sample.hit.element,
			/*parameters=*/ sample.hit.parameters,
			/*weight=*/ weight,
			/*target=*/ target,
			/*ξ=*/ ξAccept
		);
	}
	// -- finalize the reservoir
	var result: vkof::RegirFillResult;
	result.reservoir = vkof::regirReservoirFinalize(state);
	result.normalizationEstimate = (
		normalizationSum / f32(vkof::skRegirCandidateCount)
	);
	result.position = state.position;
	return result;
}

// -----------------------------------------------------------------------------
// -- shading
// -----------------------------------------------------------------------------

// returns true if a regir slot is usable for shading
public func regirSlotUsable(slot: u32) -> bool {
	return (
		slot != vkof::skRegirSlotInvalid
		&& (
			vkof::global.regirCells[slot].lastFillIndex
			== vkof::global.regirFillIndex
		)
	);
}

// approximate area density of the regir technique
// ĥ / Norm
public func regirPdfArea(
	slot: u32,
	lightIndex: u32,
	lightOri: f32v3,
	lightNor: f32v3,
) -> f32
{
	let cell: vkof::RegirCell = vkof::global.regirCells[slot];
	if (cell.normalizationNoncanonical <= 0.0f) { return 0.0f; }
	let light: vkof::SceneEmitterLight = (
		vkof::global.sceneEmitterLights[lightIndex]
	);
	// -- calculate non-canonical target function for the light sample
	let target: f32 = (
		regirTargetNonCanonicalWithCell(
			/*light=*/ light,
			/*lightPosition=*/ lightOri,
			/*lightNor=*/ lightNor,
			/*cell=*/ cell
		)
	);
	// slice off environment from normalization factor
	let envPdf: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
	return (1.0f - envPdf) * target / cell.normalizationNoncanonical;
}

public struct RegirSelection {
	public bool valid;
	public u32 reservoirIndex;
	public u32 lightIndex;
	public vkof::SurfaceHit hit;
	public vkof::SurfaceGeometry geometry;
	public f32 weight;
};

// picks a reservoir of a slot uniformly
public func regirSelectReservoir(
	slot: u32,
	ξ: f32,
) -> vkof::RegirSelection
{
	// -- pick a reservoir uniformly
	var selection: vkof::RegirSelection;
	selection.valid = false;
	let index: u32 = (
		min(
			u32(ξ * f32(vkof::skRegirReservoirCount)),
			vkof::skRegirReservoirCount - 1u
		)
	);
	selection.reservoirIndex = index;
	let reservoir: vkof::RegirReservoir = (
		vkof::global.regirReservoirs[slot * vkof::skRegirReservoirCount + index]
	);
	// -- check if the reservoir is valid
	if (reservoir.weight <= 0.0f) { return selection; }
	// -- rebuild the surface hit and geometry from the reservoir
	let light: vkof::SceneEmitterLight = (
		vkof::global.sceneEmitterLights[reservoir.lightIndex]
	);
	selection.valid = true;
	selection.lightIndex = reservoir.lightIndex;
	selection.hit.kind = vkof::GeometryKind.Triangle;
	selection.hit.instance = light.modelDrawIndex;
	selection.hit.element = reservoir.primitiveIndex;
	selection.hit.parameters = reservoir.parameters;
	selection.geometry = vkof::surfaceGeometryFromParameters(selection.hit);
	selection.weight = reservoir.weight;
	return selection;
}

// selects a light from the scene emitter list, returns the light index and pdf
// slightly more expensive as it resamples weights to pick a better
// reservoir candidate
public func regirSelectReservoirWithResampling(
	slot: u32,
	position: f32v3,
	normal: Optional<f32v3>,
	ξ: f32v2,
) -> vkof::RegirSelection
{
	let count: u32 = vkof::skRegirShadeCandidateCount;
	let stride: u32 = vkof::skRegirReservoirCount / count;
	let start: u32 = (
		min(
			u32(ξ.x * f32(vkof::skRegirReservoirCount)),
			vkof::skRegirReservoirCount - 1u
		)
	);
	// -- resample weights \hat{p} · W / S for every reservoir candidate
	// the idea is to compute the target function for each reservoir candidate,
	// store that as a weight (ĥ / p), then pick one of the candidates
	// proportional to its weight
	var weights: f32[vkof::skRegirShadeCandidateCount];
	var targets: f32[vkof::skRegirShadeCandidateCount];
	var weightSum: f32 = 0.0f;
	for (u32 it = 0u; it < count; ++it) {
		weights[it] = 0.0f;
		targets[it] = 0.0f;
		// -- pick a reservoir uniformly
		let index: u32 = (start + it*stride) % vkof::skRegirReservoirCount;
		// TODO get better way to pick candidate
		let candidate: vkof::RegirSelection = (
			regirSelectReservoir(
				slot,
				(f32(index) + 0.5f) / f32(vkof::skRegirReservoirCount)
			)
		);
		if (!candidate.valid) { continue; }
		// -- store the target function for this candidate
		let light: vkof::SceneEmitterLight = (
			vkof::global.sceneEmitterLights[candidate.lightIndex]
		);
		targets[it] = (
			regirTargetNonCanonical(
				/*light=*/ light,
				/*lightPosition=*/ candidate.geometry.position,
				/*lightNor=*/ candidate.geometry.geometricNormal,
				/*position=*/ position,
				/*normal=*/ normal,
			)
		);
		weights[it] = targets[it] * candidate.weight / f32(count);
		weightSum += weights[it];
	}
	// -- pick a candidate proportional to its weight
	var selection: vkof::RegirSelection;
	selection.valid = false;
	if (weightSum <= 0.0f) { return selection; }
	var chosen: u32 = count - 1u;
	var cumulative: f32 = 0.0f;
	// accumulate weights, using ξ to select based on the range
	for (u32 it = 0u; it < count; ++it) {
		cumulative += weights[it];
		// if the random number is in cumulative range weight
		if (weights[it] > 0.0f && ξ.y * weightSum < cumulative) {
			chosen = it;
			break;
		}
	}
	// -- read the chosen reservoir
	let selectedIndex: u32 = (
		(start + chosen*stride) % vkof::skRegirReservoirCount
	);
	selection = (
		regirSelectReservoir(
			slot,
			// the ξ will land exactly on the chosen reservoir
			// TODO probably a helper function is far more readable/clean
			(f32(selectedIndex) + 0.5f) / f32(vkof::skRegirReservoirCount)
		)
	);
	// since the reservoir was chosen proportional to its weight, the actual
	// weight can be computed as the sum of all weights divided by
	// target function of the chosen reservoir
	selection.weight = weightSum / targets[chosen];
	return selection;
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// pcg hash [JO20]
func implRegirPcgHash(x: u32) -> u32 {
	var state: u32 = x * 747796405u + 2891336453u;
	var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

// returns the slot index for a hash and probe count
func implRegirSlotAt(hash: u32, probe: u32) -> u32 {
	return (hash + probe) & (vkof::global.regirCellCount - 1u);
}

func implRegirMarkUsed(slot: u32) {
	if (
		vkof::global.regirCells[slot].frameIndexLastUsed
		!= vkof::global.frameIndex
	) {
		vkof::global.regirCells[slot].frameIndexLastUsed = vkof::global.frameIndex;
	}
}

// compare-and-swaps the guard from expected to hash and initializes the cell
func implRegirClaim(
	slot: u32,
	expected: u32,
	hash: u32,
	key: vkof::RegirCellKey,
	position: f32v3,
	normal: Optional<f32v3>,
) -> bool
{
	var previous: u32;
	InterlockedCompareExchange(
		vkof::global.regirGuards[slot], expected, hash, previous
	);
	if (previous != expected) { return false; }
	// push the origin slightly toward normal to avoid self-intersection
	let positionOffset: f32v3 = (
		position
		+ (
			normal.hasValue
			? normal.value*vkof::rayEpsilon(position) : f32v3(0.0f)
		)
	);
	var cell: vkof::RegirCell;
	cell.key = key;
	cell.representativeOri = positionOffset;
	cell.representativeNor = normal.hasValue ? normal.value : f32v3(0.0f);
	cell.normalizationNoncanonical = 0.0f;
	cell.normalizationCanonical = 0.0f;
	cell.lastFillIndex = vkof::skRegirSlotInvalid;
	cell.frameIndexLastUsed = vkof::global.frameIndex;
	cell.pendingOri = cell.representativeOri;
	cell.pendingNor = cell.representativeNor;
	vkof::global.regirCells[slot] = cell;
	return true;
}

func implRegirFillSeed(slot: u32, reservoirIndex: u32,) -> u32 {
	return (
		implRegirPcgHash(
			slot
			^ implRegirPcgHash(
				reservoirIndex ^ implRegirPcgHash(vkof::global.regirFillIndex)
			)
		)
	);
}

func implRegirNextF32(inout state: u32) -> f32 {
	state = implRegirPcgHash(state);
	// convert to [0, 1) float
	return asfloat((state >> 9u) | 0x3f800000u) - 1.0f;
}
