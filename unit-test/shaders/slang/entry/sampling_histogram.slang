import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-fuzz";
#include "../util-sampling-bins.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	u32 mode;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32v2 alpha;
	f32v2 aniso;
};

[[vk::push_constant]] PC pc;

// observed side of the sampler-vs-pdf chi-square comparison (modeled on
// cull app/shaders/bsdf-verify-histogram.comp, retargeted from the full
// material to the individual ported samplers): draws pc.sampleCount
// directions from the selected sampler and bins them over the sphere.
// bin BSDF_VERIFY_BIN_COUNT is an overflow slot for non-finite or
// non-unit directions, so a NaN can never alias into a real bin (uint(NaN)
// is undefined in GLSL) and the tests can assert it stays empty.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

#define SAMPLING_MODE_GGX_BOUNDED_WO 0u
#define SAMPLING_MODE_GGX_VNDF_H 1u
#define SAMPLING_MODE_DIFFUSE_WO 2u
#define SAMPLING_MODE_FUZZ_WO 3u
#define SAMPLING_MODE_TRANSMISSION_ENTER 4u
#define SAMPLING_MODE_TRANSMISSION_EXIT 5u

// canonical local frame throughout: tanX=(1,0,0), tanY=(0,1,0),
// nor=(0,0,1); pc.wi lives directly in it. alpha.y is ignored by the
// vndf mode (isotropic sampler) and both alpha components by diffuse.
// the fuzz mode repurposes alpha.xy as (fuzzA, fuzzB) rather than adding a
// push-constant field; the transmission modes likewise repurpose alpha.xy
// as (specularRoughness, ior), with the medium side encoded in the mode
// (ENTER: etaI=1 outside going in; EXIT: etaI=ior inside going out).
// aniso is (specularRoughnessAnisotropy, rotation), read by the
// transmission modes only.

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

	// need to not use blue-noise which is anti-correlated across pixels.
	// just generate seeds over a pure pcg chain; hashes are chained so
	// adjacent sample indices never share a stream value
	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	const u32 h2 = fnPcgHash(h1);
	const f32v2 xi = f32v2(fnUintToUniform(h1), fnUintToUniform(h2));

	f32v3 dir;
	if (pc.mode == SAMPLING_MODE_GGX_BOUNDED_WO) {
		dir = (
			utilMicrofacetSampleGgxBoundedWoAniso(
				tanX, tanY, nor, pc.wi, pc.alpha, xi
			)
		);
	} else if (pc.mode == SAMPLING_MODE_GGX_VNDF_H) {
		dir = utilMicrofacetSampleGgxVndf(nor, pc.wi, pc.alpha.x, xi);
	} else if (pc.mode == SAMPLING_MODE_DIFFUSE_WO) {
		// the production draw path: openPbrGlossyDiffuseSampleWo takes the
		// uniform pair directly, exactly like the material shaders feed it
		dir = openPbrGlossyDiffuseSampleWo(nor, xi);
	} else if (
		pc.mode == SAMPLING_MODE_TRANSMISSION_ENTER
		|| pc.mode == SAMPLING_MODE_TRANSMISSION_EXIT
	) {
		// the production transmission draw path from openPbrSampleWo
		// (util-material-openpbr.glsl), non-thin-walled: vndf half vector,
		// refract, and the total-internal-reflection fallback that
		// reflects wi about the sampled half vector instead
		const bool isInsideMedium = pc.mode == SAMPLING_MODE_TRANSMISSION_EXIT;
		const f32 ior = pc.alpha.y;
		const f32v2 transmissionAlpha = (
			openPbrRoughnessAlpha(pc.alpha.x, pc.aniso.x)
		);
		ShadingFrame frame;
		frame.tanX = tanX;
		frame.tanY = tanY;
		frame.nor = nor;
		const ShadingFrame specFrame = shadingFrameRotate(frame, pc.aniso.y);
		const f32v3 h = (
			utilMicrofacetSampleGgxVndfAniso(
				specFrame.tanX, specFrame.tanY, specFrame.nor,
				pc.wi, transmissionAlpha, xi
			)
		);
		const f32 etaI = isInsideMedium ? ior : 1.0f;
		const f32 etaT = isInsideMedium ? 1.0f : ior;
		const f32v3 refracted = refract(-pc.wi, h, etaI / etaT);
		if (length(refracted) > 0.5f) {
			dir = refracted;
		} else {
			dir = normalize(2.0f * dot(pc.wi, h) * h - pc.wi);
		}
	} else {
		// same production draw path convention as diffuse, above
		dir = openPbrFuzzSampleWo(nor, pc.wi, pc.alpha.x, pc.alpha.y, xi);
	}

	const bool finite = (
		!isnan(dir.x) && !isnan(dir.y) && !isnan(dir.z)
		&& !isinf(dir.x) && !isinf(dir.y) && !isinf(dir.z)
	);
	const bool unit = finite && abs(length(dir) - 1.0f) < 1e-2f;
	const u32 bin = (
		unit
		? bsdfVerifyDirectionToBin(nor, dir)
		: BSDF_VERIFY_BIN_COUNT
	);

	atomicAdd(&((u32 *)(pc.histogramVa))[bin], 1u);
}
