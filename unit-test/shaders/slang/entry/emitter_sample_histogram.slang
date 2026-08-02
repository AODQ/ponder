#include "../util-emitter-test.slang"
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
#include "../util-sampling-bins.slang"
import "util-material-openpbr-shared";


// observed side of the emitter sampler's chi-square: draws directions from
// utilEmitterSample and bins them over the full sphere. world-up stands in
// for the bin grid's reference axis -- the sampler has no shading frame.

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
	const u32 h2 = fnPcgHash(h1);
	const u32 h3 = fnPcgHash(h2);
	const u32 h4 = fnPcgHash(h3);
	const f32v4 xi = f32v4(
		fnUintToUniform(h1), fnUintToUniform(h2),
		fnUintToUniform(h3), fnUintToUniform(h4)
	);

	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	const EmitterSample s = (
		utilEmitterSample(lights, pc.lightCount, models, pc.from, xi)
	);

	const bool finite = (
		s.valid
		&& !isnan(s.direction.x) && !isnan(s.direction.y) && !isnan(s.direction.z)
		&& !isinf(s.direction.x) && !isinf(s.direction.y) && !isinf(s.direction.z)
		&& !isnan(s.pdf) && !isinf(s.pdf)
	);
	const bool unit = finite && abs(length(s.direction) - 1.0f) < 1e-2f;
	const f32v3 nor = f32v3(0.0f, 1.0f, 0.0f);
	// the overflow bin doubles as the invalid-sample counter
	const u32 bin = (
		unit ? bsdfVerifyDirectionToBin(nor, s.direction) : BSDF_VERIFY_BIN_COUNT
	);

	atomicAdd(&((u32 *)(pc.outVa))[bin], 1u);
}
