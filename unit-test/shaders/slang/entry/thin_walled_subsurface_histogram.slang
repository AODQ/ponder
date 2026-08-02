import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
#include "../util-sampling-bins.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 nor;
	f32 anisotropy;
};

[[vk::push_constant]] PC pc;

// observed side of openPbrThinWalledSubsurfaceSampleWo vs
// openPbrThinWalledSubsurfacePdf's chi-square: draws pc.sampleCount
// directions and bins them over the full sphere around pc.nor, same
// convention as subsurface_entry_histogram.comp.

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

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

	// same draw order as openPbrSampleWo's thin-walled branch
	const f32v2 xi = fnSampleUniform2(state);
	const f32 uSide = fnSampleUniform(state);
	const f32v3 dir = (
		openPbrThinWalledSubsurfaceSampleWo(pc.nor, pc.anisotropy, xi, uSide)
	);

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
