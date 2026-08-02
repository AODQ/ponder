import "util-material-openpbr-tables";
import "util-material-openpbr-subsurface";
#include "../util-sampling-bins.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 nor;
};

[[vk::push_constant]] PC pc;

// observed side of openPbrSubsurfaceEntrySampleWo vs
// openPbrSubsurfaceEntryPdf's sampler-vs-eval chi-square: draws
// pc.sampleCount directions and bins them over the full sphere around
// pc.nor, same convention as vdb_phase_histogram.comp / mixture_histogram
// .comp.

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	const f32v2 xi = fnSampleUniform2(state);
	const f32v3 dir = openPbrSubsurfaceEntrySampleWo(pc.nor, xi);

	const bool finite = (
		!isnan(dir.x) && !isnan(dir.y) && !isnan(dir.z)
		&& !isinf(dir.x) && !isinf(dir.y) && !isinf(dir.z)
	);
	const bool unit = finite && abs(length(dir) - 1.0f) < 1e-2f;
	const u32 bin = (
		unit
		? bsdfVerifyDirectionToBin(pc.nor, dir)
		: BSDF_VERIFY_BIN_COUNT
	);

	atomicAdd(&((u32 *)(pc.histogramVa))[bin], 1u);
}
