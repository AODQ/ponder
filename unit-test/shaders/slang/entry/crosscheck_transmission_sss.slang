import "util-material-openpbr-tables";
import "util-random";
import "util-shading-frame";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-transmission";
import "util-material-openpbr-subsurface";
import "util-material-openpbr-shared";

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
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 12u;
	const u32 o = thread * 14u;

	const f32v3 nor = normalize(f32v3(
		inBuf[i + 0u], inBuf[i + 1u], abs(inBuf[i + 2u]) + 0.1f
	));
	const f32v3 wi = normalize(f32v3(
		inBuf[i + 3u], inBuf[i + 4u], abs(inBuf[i + 5u]) + 0.1f
	));
	// wo on the far side, so the btdf is actually exercised
	const f32v3 wo = normalize(f32v3(
		inBuf[i + 6u], inBuf[i + 7u], -abs(inBuf[i + 8u]) - 0.1f
	));
	const ShadingFrame frame = shadingFrameFromNormal(nor);

	OpenPbrMaterial mat;
	mat.specularRoughness = inBuf[i + 9u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = inBuf[i + 10u];
	mat.specularWeight = 1.0f;
	mat.specularColor = f32v3(1.0f);
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.1f;
	mat.coatIor = 1.6f;
	mat.transmissionWeight = 1.0f;
	mat.transmissionColor = f32v3(0.9f, 0.95f, 1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 40.0f;
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.4f;
	mat.geometryThinWalled = 0.0f;
	mat.subsurfaceColor = f32v3(0.9f, 0.6f, 0.4f);
	mat.subsurfaceRadius = inBuf[i + 11u];
	mat.subsurfaceRadiusScale = f32v3(1.0f, 0.7f, 0.5f);
	mat.subsurfaceScatterAnisotropy = inBuf[i + 9u] * 1.8f - 0.9f;

	const f32v3 btdf = openPbrTransmissionEvaluateF(
		mat, frame, wi, wo, false, mat.specularIor
	);
	outBuf[o + 0u] = btdf.r;
	outBuf[o + 1u] = btdf.g;
	outBuf[o + 2u] = btdf.b;
	outBuf[o + 3u] = openPbrTransmissionPdf(
		mat, frame, wi, wo, false, mat.specularIor
	);
	outBuf[o + 4u] = openPbrTransmissionIor(mat);

	const f32v3 refl = openPbrSubsurfaceDiffusionReflectance(
		f32v3(inBuf[i + 9u], inBuf[i + 10u] - 1.0f, inBuf[i + 11u])
	);
	outBuf[o + 5u] = refl.r;
	outBuf[o + 6u] = refl.g;
	outBuf[o + 7u] = refl.b;

	const f32v3 inverted = openPbrSubsurfaceInvertDiffusionReflectance(
		clamp(mat.subsurfaceColor, f32v3(0.0f), f32v3(1.0f))
	);
	outBuf[o + 8u] = inverted.r;
	outBuf[o + 9u] = inverted.g;

	f32v3 sigmaA;
	f32v3 sigmaS;
	openPbrSubsurfaceAlbedoToSigma(
		mat.subsurfaceColor,
		mat.subsurfaceRadius * mat.subsurfaceRadiusScale,
		mat.subsurfaceScatterAnisotropy,
		sigmaA, sigmaS
	);
	outBuf[o + 10u] = sigmaA.r;
	outBuf[o + 11u] = sigmaS.g;

	outBuf[o + 12u] = openPbrSubsurfacePhaseEvalHg(
		inBuf[i + 9u] * 2.0f - 1.0f, mat.subsurfaceScatterAnisotropy
	);
	outBuf[o + 13u] = openPbrDwivediDiffusionLength(inBuf[i + 9u]);
}
