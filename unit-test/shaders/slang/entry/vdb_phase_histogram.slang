import "util-vdb";
#include "../util-sampling-bins.slang"

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 anisotropyHG;
	f32 anisotropyDraine;
	f32 draineAlpha;
	f32 draineWeight;
};

[[vk::push_constant]] PC pc;

// observed side of the fog phase function's sampler-vs-eval chi-square:
// draws pc.sampleCount directions from utilVdbPhaseSampleMixture (the real
// production HG+Draine mixture sampler) and bins them over the full sphere
// around pc.wi, same convention as mixture_histogram.comp -- a phase
// function is the volumetric analog of a bsdf lobe, azimuthally symmetric
// about the incoming direction rather than a surface normal, so pc.wi plays
// the role mixture_histogram.comp's shading normal does.

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	UtilVdbPhaseParams params;
	params.anisotropyHG = pc.anisotropyHG;
	params.anisotropyDraine = pc.anisotropyDraine;
	params.draineAlpha = pc.draineAlpha;
	params.draineWeight = pc.draineWeight;

	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	const f32v3 dir = utilVdbPhaseSampleMixture(params, pc.wi, state);

	const bool finite = (
		!isnan(dir.x) && !isnan(dir.y) && !isnan(dir.z)
		&& !isinf(dir.x) && !isinf(dir.y) && !isinf(dir.z)
	);
	const bool unit = finite && abs(length(dir) - 1.0f) < 1e-2f;
	const u32 bin = (
		unit
		? bsdfVerifyDirectionToBin(pc.wi, dir)
		: BSDF_VERIFY_BIN_COUNT
	);

	atomicAdd(&((u32 *)(pc.histogramVa))[bin], 1u);
}
