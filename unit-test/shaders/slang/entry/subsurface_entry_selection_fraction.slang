import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 pickCountVa;
	u32 sampleCount;
	u32 seedSalt;
	f32 subsurfaceWeight;
};

[[vk::push_constant]] PC pc;

// discrete lobe-selection side: dispatches the full openPbrSampleWo with a
// material that only has diffuse + subsurface budget (coat/specular/fuzz/
// transmission/metalness all zero), and records which OPENPBR_LOBE_* got
// picked. the fraction landing on OPENPBR_LOBE_SUBSURFACE_ENTRY must match
// pc.subsurfaceWeight -- the same discrete stochastic-split identity every
// other pair of lobes in openPbrLobeSelection already relies on.

// util-material-openpbr.glsl defines IPI/PI/TAU/skFon1/skFon2/skMinRoughness
// itself -- pulling it in directly rather than the piecewise subsurface.glsl
// include used by the other subsurface_*.comp test shaders, since this one
// needs the full lobe-selection/openPbrSampleWo machinery

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

	OpenPbrMaterial mat;
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(0.8f);
	mat.baseMetalness = 0.0f;
	mat.baseDiffuseRoughness = 0.0f;
	mat.specularWeight = 0.0f;
	mat.specularColor = f32v3(1.0f);
	mat.specularRoughness = 0.3f;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = 1.5f;
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.subsurfaceWeight = pc.subsurfaceWeight;
	mat.subsurfaceColor = f32v3(0.8f);
	mat.subsurfaceRadius = 1.0f;
	mat.subsurfaceRadiusScale = f32v3(1.0f);
	mat.subsurfaceScatterAnisotropy = 0.0f;
	mat.coatWeight = 0.0f;
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = 0.0f;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.6f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = 0.0f;
	mat.fuzzColor = f32v3(1.0f);
	mat.fuzzRoughness = 0.3f;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.3f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = 0.0f;
	mat.alphaCutoff = 0.0f;

	const ShadingFrame frame = shadingFrameFromNormal(f32v3(0.0f, 0.0f, 1.0f));
	const f32v3 wi = f32v3(0.0f, 0.0f, 1.0f);
	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = 0u;

	f32 pdf;
	bool sampledTransmission;
	bool sampledSubsurfaceEntry;
	u32 pickedLobe;
	openPbrSampleWo(
		tables, frame, frame, wi, mat,
		pdf, sampledTransmission, sampledSubsurfaceEntry, pickedLobe,
		state,
		/*isInsideMedium=*/false, /*dispersedIor=*/mat.specularIor
	);

	atomicAdd(&((u32 *)(pc.pickCountVa))[pickedLobe], 1u);
}
