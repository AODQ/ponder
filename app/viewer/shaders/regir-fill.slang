import "util-random";
import "util-regir";
import "util-regir-fill";
import "util-mor-shared";
import "util-resolve-pc";
import "util-regir-pc";
import "util-global-pc";

struct PC {
	GpuGlobalPc global;
	GpuResolvePC resolve;
};

[[vk::push_constant]] PC pc;

// regir pass C: build one cell's reservoirs.
//
// one workgroup per table slot, one thread per reservoir. each thread streams
// skRegirCandidateCount light samples through weighted reservoir sampling
// against the target function -- see util-regir-fill.glsl for the math and
// util-regir.glsl for what a reservoir and W actually are.
//
// (review note) this dispatches over *every* slot and early-outs on an empty
// guard rather than walking a compacted live-cell list. at the 60% target load
// that wastes about 40% of the workgroups, which is nothing next to getting
// compaction and indirect dispatch right, and it can be added later without
// touching any of the math.

// RESOLVE_NO_PC: pulls GpuResolveModelIndirect out of resolve_pc.h without its
// push_constant block, since regir-pc.h declares this pass's own
#define RESOLVE_NO_PC

// one entry per reservoir, reduced by thread 0 into the cell's normalization
// factor. sized to the workgroup, which is sized to skRegirReservoirCount
groupshared f32 sRegirWeightSums[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_LocalInvocationID : SV_GroupThreadID,
	const u32v3 gl_WorkGroupID : SV_GroupID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const u32 slot = gl_WorkGroupID.x;
	const u32 reservoirIndex = gl_LocalInvocationID.x;

	if (
		unf.regirEnabled == 0u
		|| slot >= unf.regirCellCount
		|| unf.emitterLightCount == 0u
	) {
		return;
	}

	u32 * guards = (u32 *)(unf.regirGuards);
	GpuRegirCell * cells = (GpuRegirCell *)(unf.regirCells);
	GpuRegirReservoir * reservoirs = (
		(GpuRegirReservoir *)(unf.regirReservoirs)
	);

	// an unclaimed slot has no representative point, so there is nothing to
	// build a target function around
	if (guards[slot] == 0u) {
		return;
	}

	const GpuRegirCell cell = cells[slot];
	const bool hasNormal = cell.key.normalBucket != skRegirNormalBucketNone;

	u64 rngState = (
		utilRegirFillSeed(slot, reservoirIndex, unf.regirFrameIndex)
	);

	const RegirFillResult fill = (
		utilRegirFillReservoir(
			(GpuEmitterLight *)(unf.emitterLights),
			unf.emitterLightCount,
			(GpuResolveModelIndirect *)(pc.global.models),
			cell.representativePosition,
			cell.representativeNormal,
			hasNormal,
			rngState
		)
	);
	const GpuRegirReservoir res = fill.reservoir;
	const f32 weightSum = fill.normalizationEstimate;
	reservoirs[slot * skRegirReservoirCount + reservoirIndex] = res;

	// Norm = (1/M) * sum(\hat{h}/p) across every candidate the cell drew, which
	// is the monte carlo estimate of \int \hat{h} dA. it is the sum the fill
	// loop already produced, so this costs one reduction and no extra sampling.
	// stored rather than recomputed later: both halves of every mis weight have
	// to read this same number, noise and all, or they price the same direction
	// differently
	sRegirWeightSums[reservoirIndex] = weightSum;
	GroupMemoryBarrierWithGroupSync();
	if (reservoirIndex != 0u) {
		return;
	}
	f32 total = 0.0f;
	for (u32 i = 0u; i < skRegirReservoirCount; ++i) {
		total += sRegirWeightSums[i];
	}
	const f32 candidateTotal = (
		f32(skRegirReservoirCount) * f32(skRegirCandidateCount)
	);
	cells[slot].normalizationNonCanonical = total / candidateTotal;
	// the canonical target function is a later phase; nothing reads this yet
	cells[slot].normalizationCanonical = 0.0f;
	cells[slot].lastFillIndex = unf.regirFrameIndex;
}
