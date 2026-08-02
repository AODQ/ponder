import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
#include "../util-sampling-bins.slang"
import "util-environment-map";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	EnvironmentMapHandles handles;
	u32 sampleCount;
	u32 seedSalt;
	f32 rotation;
};

[[vk::push_constant]] PC pc;

// observed side of the environment map's sampler-vs-pdf chi-square check
// (same technique as sampling_histogram.comp, retargeted): draws
// pc.sampleCount directions from environmentMapImportanceSample and bins
// them over the full sphere. unlike the bsdf-lobe sampling tests there's no
// material/wi involved -- environmentMapImportanceSample is a pure function
// of (handles, xi) -- so world-up stands in for the bin grid's reference
// axis, which conveniently matches the equirect map's own polar axis.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
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

	// same seeding convention as sampling_histogram.comp: a pure pcg chain
	// rather than blue noise, which is anti-correlated across pixels
	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	const u32 h2 = fnPcgHash(h1);
	const f32v2 xi = f32v2(fnUintToUniform(h1), fnUintToUniform(h2));

	f32 pdf;
	const f32v3 dir = (
		environmentMapImportanceSample(pc.handles, xi, pc.rotation, pdf)
	);

	const bool finite = (
		!isnan(dir.x) && !isnan(dir.y) && !isnan(dir.z)
		&& !isinf(dir.x) && !isinf(dir.y) && !isinf(dir.z)
		&& !isnan(pdf) && !isinf(pdf)
	);
	const bool unit = finite && abs(length(dir) - 1.0f) < 1e-2f;
	const f32v3 nor = f32v3(0.0f, 1.0f, 0.0f);
	const u32 bin = (
		unit
		? bsdfVerifyDirectionToBin(nor, dir)
		: BSDF_VERIFY_BIN_COUNT
	);

	atomicAdd(&((u32 *)(pc.histogramVa))[bin], 1u);
}
