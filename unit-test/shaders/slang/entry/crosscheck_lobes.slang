import "util-material-openpbr-tables";
import "util-shading-frame";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-metallic";
import "util-material-openpbr-coat";
import "util-material-openpbr-fuzz";
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
	const u32 i = thread * 14u;
	const u32 o = thread * 12u;

	const f32v3 nor = normalize(f32v3(
		inBuf[i + 0u], inBuf[i + 1u], abs(inBuf[i + 2u]) + 0.1f
	));
	const f32v3 wi = normalize(f32v3(
		inBuf[i + 3u], inBuf[i + 4u], abs(inBuf[i + 5u]) + 0.1f
	));
	const f32v3 wo = normalize(f32v3(
		inBuf[i + 6u], inBuf[i + 7u], abs(inBuf[i + 8u]) + 0.1f
	));
	const ShadingFrame frame = shadingFrameFromNormal(nor);

	OpenPbrMaterial mat;
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(0.8f, 0.5f, 0.2f);
	mat.baseMetalness = inBuf[i + 9u];
	mat.baseDiffuseRoughness = inBuf[i + 10u];
	mat.specularWeight = 1.0f;
	mat.specularColor = f32v3(1.0f, 0.9f, 0.8f);
	mat.specularRoughness = inBuf[i + 11u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = 1.5f;
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.4f;
	mat.coatWeight = inBuf[i + 12u];
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = inBuf[i + 13u];
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.6f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = inBuf[i + 12u];
	mat.fuzzColor = f32v3(0.9f, 0.9f, 1.0f);
	mat.fuzzRoughness = inBuf[i + 13u];
	mat.geometryThinWalled = 0.0f;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = 0u;

	const f32v3 diffuse = openPbrOrenNayerEvaluateF(
		mat.baseColor, mat.baseWeight, mat.baseDiffuseRoughness, nor, wi, wo
	);
	outBuf[o + 0u] = diffuse.r;
	outBuf[o + 1u] = diffuse.g;
	outBuf[o + 2u] = diffuse.b;

	const f32v3 metal = openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo);
	outBuf[o + 3u] = metal.r;
	outBuf[o + 4u] = metal.g;
	outBuf[o + 5u] = metal.b;

	const f32v3 coat = openPbrCoatEvaluateF(
		tables, mat, frame, wi, wo, diffuse, false
	);
	outBuf[o + 6u] = coat.r;
	outBuf[o + 7u] = coat.g;
	outBuf[o + 8u] = coat.b;

	const f32v3 fuzz = openPbrFuzzEvaluateF(tables, mat, coat, nor, wi, wo);
	outBuf[o + 9u] = fuzz.r;
	outBuf[o + 10u] = fuzz.g;
	outBuf[o + 11u] = fuzz.b;
}
