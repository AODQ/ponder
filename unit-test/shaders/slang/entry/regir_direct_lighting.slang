import "util-random";
import "util-regir";
import "util-regir-fill";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u64 reservoirsVa;
	u64 lightsVa;
	u64 modelsVa;
	// the point the reservoirs are built for
	f32v3 representativePosition;
	f32v3 representativeNormal;
	// the point actually being shaded; equal to the representative in the
	// control configuration, offset from it in the ones that matter
	f32v3 shadingPosition;
	f32v3 shadingNormal;
	u32 lightCount;
	u32 sampleCount;
	u32 seedSalt;
	u32 mode;
};

[[vk::push_constant]] PC pc;

// tier 11 of plan-many-lights.md, minus the renderer: fill a cell at its
// representative point, then shade at a *different* point, and check the mean
// against a closed form evaluated at the shading point.
//
// this is the question regir actually rests on. the reservoirs were built for
// P_c; every shading point in the cell reuses them. that is only legitimate if
// reusing a proposal built somewhere else costs variance and not bias -- which
// is the claim, and this is where it gets measured rather than assumed.
//
// no hash table here on purpose. tier 9 covers the lookup; wiring it in would
// only add a way for this test to fail for a reason that has nothing to do with
// the estimator. so the fill pass writes to a plain reservoir array and the
// shade pass indexes it directly, which is exactly what utilRegirSelect does
// once its lookup has succeeded.

#define RESOLVE_NO_PC

#define REGIR_DL_MODE_FILL 0u
#define REGIR_DL_MODE_SHADE 1u

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 index = gl_GlobalInvocationID.x;
	GpuRegirReservoir * reservoirs = (
		(GpuRegirReservoir *)(pc.reservoirsVa)
	);

	if (pc.mode == REGIR_DL_MODE_FILL) {
		if (index >= skRegirReservoirCount) {
			return;
		}
		u64 rngState = utilRegirFillSeed(0u, index, pc.seedSalt);
		reservoirs[index] = (
			utilRegirFillReservoir(
				(GpuEmitterLight *)(pc.lightsVa),
				pc.lightCount,
				(GpuResolveModelIndirect *)(pc.modelsVa),
				pc.representativePosition,
				pc.representativeNormal,
				/*hasNormal=*/true,
				rngState
			)
		).reservoir;
		return;
	}

	if (index >= pc.sampleCount) {
		return;
	}

	// one draw, uniform over the cell's reservoirs -- the same choice
	// utilRegirSelect makes. the 1/skRegirReservoirCount does not appear in the
	// estimator: each reservoir is independently unbiased, so picking among
	// them averages unbiased estimates rather than composing a density
	u64 rngState = utilRegirFillSeed(1u, index, pc.seedSalt);
	const u32 pick = (
		min(
			u32(fnSampleUniform(rngState) * f32(skRegirReservoirCount)),
			skRegirReservoirCount - 1u
		)
	);
	const GpuRegirReservoir res = reservoirs[pick];

	f32 * outBuf = (f32 *)(pc.outVa);
	if (!(res.unbiasedWeight > 0.0f)) {
		outBuf[index] = 0.0f;
		return;
	}

	// rebuild the sampled point exactly the way the fill pass did: from
	// (primitiveIndex, barycentric) through the same triangle fetch. storing a
	// position instead would be a second source of truth for where the sample
	// is, which is the shape of plan-nee.md trap 4
	const GpuResolveModelIndirect model = (
		((GpuResolveModelIndirect *)(pc.modelsVa))[
			((GpuEmitterLight *)(pc.lightsVa))[res.lightIndex].modelDrawIndex
		]
	);
	const EmitterTriangle tri = (
		utilEmitterTriangle(model, res.primitiveIndex)
	);
	const f32v3 bary = f32v3(
		1.0f - res.barycentric.x - res.barycentric.y,
		res.barycentric.x,
		res.barycentric.y
	);
	const f32v3 lightPos = (
		tri.p0 * bary.x + tri.p1 * bary.y + tri.p2 * bary.z
	);
	const EmitterTriangleGeometry geo = utilEmitterTriangleGeometry(tri);

	// the estimator, in area measure and evaluated at the *shading* point --
	// not at the representative one. the reservoir only supplied which point on
	// which light to connect to, and W
	const f32v3 toLight = lightPos - pc.shadingPosition;
	const f32 distanceSq = dot(toLight, toLight);
	if (!(distanceSq > 0.0f)) {
		outBuf[index] = 0.0f;
		return;
	}
	const f32 distance = sqrt(distanceSq);
	const f32v3 direction = toLight / distance;

	const f32 cosAtLight = abs(dot(geo.normal, -direction));
	const f32 cosAtSurface = max(dot(pc.shadingNormal, direction), 0.0f);
	const f32 emitted = (
		utilRegirLuminance(
			((GpuEmitterLight *)(pc.lightsVa))[res.lightIndex].radiance
		)
	);

	// no visibility term: these scenes have a single unoccluded emitter, so
	// the closed form on the cpu side is the exact answer rather than an
	// upper bound
	const f32 geometry = cosAtLight * cosAtSurface / distanceSq;
	outBuf[index] = emitted * geometry * res.unbiasedWeight;
}
