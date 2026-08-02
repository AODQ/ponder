import "util-random";
import "util-material-openpbr";
#include "../util-sampling-bins.slang"

struct PC {
	u32v4 rootPad[8];
	u64 histogramVa;
	u64 pickCountVa;
	u32 zeltnerLtcParamHandle;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatRoughnessAnisotropy;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularRoughnessAnisotropyRotation;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 geometryThinWalled;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};

[[vk::push_constant]] PC pc;

// observed side of the multi-lobe mixture sampler-vs-pdf chi-square: draws
// pc.sampleCount directions from the real production openPbrSampleWo (all
// five lobes live, weighted by the material's own probabilities) and bins
// them over the sphere, same convention as sampling_histogram.comp. unlike
// that file this one goes through the full util-material-openpbr.glsl
// chain rather than the per-lobe files, since openPbrSampleWo/
// openPbrLobeSelection only exist there -- see mixture_selection_crosscheck
// .comp for why that rules out reusing sampling_histogram.comp's prelude
// (duplicate IPI/PI/TAU/skFon1/skFon2 symbols). also tallies which lobe
// openPbrSampleWo picked per draw, so the caller can chi-square the
// empirical pick fraction against sel.probabilityX / sel.probabilityTotal
// directly -- this is "the mixture invariant itself" (see project brief):
// sampler and pdf must agree on the same fractions.

// canonical local frame throughout, same as sampling_histogram.comp:
// tanX=(1,0,0), tanY=(0,1,0), nor=(0,0,1); pc.wi lives directly in it.
// coatFrame == frame (no coat normal map / no rotation source for coat in
// this synthetic material, matching openPbrEvaluatePdfSelected's own
// comment that coat has no rotation source)

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
	ShadingFrame frame;
	frame.tanX = tanX;
	frame.tanY = tanY;
	frame.nor = nor;
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	OpenPbrMaterial mat;
	mat.coatWeight = pc.coatWeight;
	mat.coatRoughness = pc.coatRoughness;
	mat.coatRoughnessAnisotropy = pc.coatRoughnessAnisotropy;
	mat.coatIor = pc.coatIor;
	mat.specularWeight = pc.specularWeight;
	mat.specularIor = pc.specularIor;
	mat.specularRoughness = pc.specularRoughness;
	mat.specularRoughnessAnisotropy = pc.specularRoughnessAnisotropy;
	mat.specularRoughnessAnisotropyRotation = (
		pc.specularRoughnessAnisotropyRotation
	);
	mat.baseMetalness = pc.baseMetalness;
	mat.transmissionWeight = pc.transmissionWeight;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	// openPbrLobeSelection now reads this to split the diffuse budget; left
	// unset it's uninitialized gpu memory, not a guaranteed 0
	mat.subsurfaceWeight = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.geometryThinWalled = pc.geometryThinWalled;
	mat.fuzzWeight = pc.fuzzWeight;
	mat.fuzzRoughness = pc.fuzzRoughness;

	// independent pcg32 stream per sample, same pattern as
	// sampling_histogram.comp / cornell_openpbr_render.comp: the stream
	// feeds the lobe-selection draw then the chosen lobe's directional draw
	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	f32 pdf;
	bool sampledTransmission;
	bool sampledSubsurfaceEntry;
	u32 pickedLobe;
	const f32v3 dir = (
		openPbrSampleWo(
			tables, frame, coatFrame, pc.wi, mat,
			pdf, sampledTransmission, sampledSubsurfaceEntry, pickedLobe,
			state,
			/*isInsideMedium=*/false, /*dispersedIor=*/mat.specularIor
		)
	);

	atomicAdd(&((u32 *)(pc.pickCountVa))[pickedLobe], 1u);

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
