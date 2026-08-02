import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// the openpbr headers expect their includer to supply all of these
#ifndef f32
#define i32 i32
#define i32v2 i32v2
#define f32 f32
#define f32v2 f32v2
#define f32v3 f32v3
#define f32v4 f32v4
#define f32m44 f32m4x4
#define u32 u32
#define u32v2 u32v2
#define u32v3 u32v3
#define u32v4 u32v4
#define u64 u64
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 16u;
	const u32 o = thread * 13u;

	const f32v3 nor = normalize(f32v3(
		inBuf[i + 0u], inBuf[i + 1u], abs(inBuf[i + 2u]) + 0.1f
	));
	const f32v3 wi = normalize(f32v3(
		inBuf[i + 3u], inBuf[i + 4u], abs(inBuf[i + 5u]) + 0.1f
	));
	const f32 woSign = inBuf[i + 15u] < 0.5f ? 1.0f : -1.0f;
	const f32v3 wo = normalize(f32v3(
		inBuf[i + 6u],
		inBuf[i + 7u],
		woSign * (abs(inBuf[i + 8u]) + 0.1f)
	));
	const ShadingFrame frame = shadingFrameFromNormal(nor);

	OpenPbrMaterial mat;
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(0.8f, 0.5f, 0.2f);
	mat.baseMetalness = inBuf[i + 9u];
	mat.baseDiffuseRoughness = inBuf[i + 10u];
	mat.specularWeight = 1.0f;
	mat.specularColor = f32v3(1.0f, 0.95f, 0.9f);
	mat.specularRoughness = inBuf[i + 11u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = 1.5f;
	mat.transmissionWeight = inBuf[i + 12u];
	mat.transmissionColor = f32v3(0.95f, 0.98f, 1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 40.0f;
	mat.subsurfaceWeight = inBuf[i + 13u];
	mat.subsurfaceColor = f32v3(0.9f, 0.6f, 0.4f);
	mat.subsurfaceRadius = 0.5f;
	mat.subsurfaceRadiusScale = f32v3(1.0f, 0.7f, 0.5f);
	mat.subsurfaceScatterAnisotropy = 0.0f;
	mat.coatWeight = inBuf[i + 14u];
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = 0.15f;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.6f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = inBuf[i + 14u] * 0.5f;
	mat.fuzzColor = f32v3(0.9f, 0.9f, 1.0f);
	mat.fuzzRoughness = 0.3f;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.4f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = 0.0f;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = 0u;

	const f32v3 f = openPbrEvaluateF(
		tables, frame, frame, wi, wo, mat, false, mat.specularIor
	);
	outBuf[o + 0u] = f.r;
	outBuf[o + 1u] = f.g;
	outBuf[o + 2u] = f.b;

	const OpenPbrLobeSelection sel = openPbrLobeSelection(
		tables, mat, dot(nor, wi)
	);
	outBuf[o + 3u] = sel.probabilityCoat;
	outBuf[o + 4u] = sel.probabilitySpecular;
	outBuf[o + 5u] = sel.probabilityDiffuse;
	outBuf[o + 6u] = sel.probabilityFuzz;
	outBuf[o + 7u] = sel.probabilityTransmission;
	outBuf[o + 8u] = sel.probabilitySubsurfaceEntry;
	outBuf[o + 9u] = sel.probabilityTotal;
	outBuf[o + 10u] = sel.fuzzA;
	outBuf[o + 11u] = sel.fuzzB;

	outBuf[o + 12u] = openPbrEvaluatePdfSelected(
		sel, mat, frame, frame, wi, wo, false, mat.specularIor
	);
}
