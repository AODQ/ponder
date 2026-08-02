import "util-random";
import "util-regir";
import "util-regir-fill";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u64 lightsVa;
	u64 modelsVa;
	f32v3 representativePosition;
	f32v3 representativeNormal;
	u32 lightCount;
	u32 realizationCount;
	u32 seedSalt;
	u32 hasNormal;
};

[[vk::push_constant]] PC pc;

// tier 10 of plan-many-lights.md: the ris / reservoir math, with no grid table
// involved at all.
//
// utilRegirFillReservoir takes a representative point and normal directly
// rather than a cell, so the fill math is testable without the hash grid, the
// scout pass, or a clear pass existing. one thread here is one independent
// *realization* of a whole cell-fill: it streams skRegirCandidateCount
// candidates through weighted reservoir sampling and reports the survivor.
//
// the assertion the c++ side builds on all of these is
//
//     E[f(Y) * W_Y] == integral f(x) dA(x)
//
// which is the operational definition of ris unbiasedness. with f == 1 it says
// E[W] is the emitter's total surface area -- an exactly computable number --
// and with f == \hat{h} it says E[targetValue * W] is \int \hat{h} dA, which is
// the same quantity the cell's normalization factor estimates by a completely
// different route.

#define RESOLVE_NO_PC

// per realization: unbiasedWeight, targetValue, weightSum, lightIndex,
// primitiveIndex, barycentric.y, barycentric.z, valid
#define REGIR_FILL_PROBE_STRIDE 8u

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 index = gl_GlobalInvocationID.x;
	if (index >= pc.realizationCount) {
		return;
	}

	// each realization is an independent fill of the same cell. the seed varies
	// only in the "reservoir index" slot, which is exactly how the real fill
	// pass decorrelates the reservoirs within one cell
	u64 rngState = utilRegirFillSeed(0u, index, pc.seedSalt);

	const RegirFillResult fill = (
		utilRegirFillReservoir(
			(GpuEmitterLight *)(pc.lightsVa),
			pc.lightCount,
			(GpuResolveModelIndirect *)(pc.modelsVa),
			pc.representativePosition,
			pc.representativeNormal,
			pc.hasNormal != 0u,
			rngState
		)
	);

	const u32 base = index * REGIR_FILL_PROBE_STRIDE;
	f32 * outBuf = (f32 *)(pc.outVa);
	const GpuRegirReservoir res = fill.reservoir;
	const f32 weightSum = fill.normalizationEstimate;
	outBuf[base + 0u] = res.unbiasedWeight;
	outBuf[base + 1u] = res.targetValue;
	outBuf[base + 2u] = weightSum;
	outBuf[base + 3u] = f32(res.lightIndex);
	outBuf[base + 4u] = f32(res.primitiveIndex);
	outBuf[base + 5u] = res.barycentric.x;
	outBuf[base + 6u] = res.barycentric.y;
	outBuf[base + 7u] = res.unbiasedWeight > 0.0f ? 1.0f : 0.0f;
}
