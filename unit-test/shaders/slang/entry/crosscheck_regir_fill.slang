import "util-emitter";
import "util-regir";
import "util-regir-fill";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
	u64 modelsVa;
	u64 lightsVa;
};

[[vk::push_constant]] PC pc;

#define RESOLVE_NO_PC

// util-regir-fill end to end: the target function, the candidate weight, and
// the streaming candidate loop that draws from the emitter sampler. the loop
// consumes the rng, so an identical stream is part of what this proves

// in:  lightCount, cellPosition(3), cellNormal(3), hasNormal, seed
// out: luminance, target, candidateWeight, reservoir(6), normalization,
//      rngState(2)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 9u;
	const u32 o = thread * 12u;

	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);

	const u32 lightCount = u32(inBuf[i + 0u]);
	const f32v3 cellPosition = f32v3(
		inBuf[i + 1u], inBuf[i + 2u], inBuf[i + 3u]
	);
	const f32v3 cellNormal = normalize(
		f32v3(inBuf[i + 4u], inBuf[i + 5u], inBuf[i + 6u])
	);
	const bool hasNormal = inBuf[i + 7u] > 0.5f;
	u64 rngState = u64(asuint(inBuf[i + 8u])) | (u64(thread) << 32);

	const f32v3 lightRadiance = lights[0].radiance;
	const f32v3 lightNormal = normalize(f32v3(0.0f, 1.0f, 0.25f));
	const f32v3 lightPosition = cellPosition + f32v3(0.05f, 0.1f, -0.02f);

	outBuf[o + 0u] = utilRegirLuminance(lightRadiance);
	outBuf[o + 1u] = utilRegirTargetNonCanonical(
		lightRadiance, lightNormal, lightPosition, cellPosition, cellNormal,
		hasNormal
	);
	outBuf[o + 2u] = utilRegirCandidateWeight(
		lightRadiance, normalize(lightPosition - cellPosition), cellNormal,
		hasNormal, 0.25f
	);

	const RegirFillResult fill = utilRegirFillReservoir(
		lights, lightCount, models, cellPosition, cellNormal, hasNormal,
		rngState
	);
	outBuf[o + 3u] = f32(fill.reservoir.lightIndex);
	outBuf[o + 4u] = f32(fill.reservoir.primitiveIndex);
	outBuf[o + 5u] = fill.reservoir.barycentric.x;
	outBuf[o + 6u] = fill.reservoir.barycentric.y;
	outBuf[o + 7u] = fill.reservoir.unbiasedWeight;
	outBuf[o + 8u] = fill.reservoir.targetValue;
	outBuf[o + 9u] = fill.normalizationEstimate;
	// the rng has to land on the same state, or the candidate stream diverged
	outBuf[o + 10u] = f32(u32(rngState & 0xFFFFu));
	outBuf[o + 11u] = f32(u32((rngState >> 16) & 0xFFFFu));
}
