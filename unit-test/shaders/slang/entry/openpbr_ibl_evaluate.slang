import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-ibl";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 zeltnerLtcParamHandle;
	u32 count;
};

[[vk::push_constant]] PC pc;

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// 20-float sample: muI, dotCoatNorWi, irradianceScalar,
// prefilteredSpecularBaseScalar, prefilteredSpecularCoatScalar,
// baseWeight, baseColorScalar, baseMetalness, specularWeight,
// specularColorScalar, specularRoughness, specularIor, coatWeight,
// coatRoughness, coatIor, coatDarkening, coatColorScalar, fuzzWeight,
// fuzzColorScalar, fuzzRoughness

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 20u;
	const f32 muI = inBuf[base+0u];
	const f32 dotCoatNorWi = inBuf[base+1u];
	const f32v3 irradiance = f32v3(inBuf[base+2u]);
	const f32v3 prefilteredSpecularBase = f32v3(inBuf[base+3u]);
	const f32v3 prefilteredSpecularCoat = f32v3(inBuf[base+4u]);

	OpenPbrMaterial mat;
	mat.baseWeight = inBuf[base+5u];
	mat.baseColor = f32v3(inBuf[base+6u]);
	mat.baseMetalness = inBuf[base+7u];
	mat.specularWeight = inBuf[base+8u];
	mat.specularColor = f32v3(inBuf[base+9u]);
	mat.specularRoughness = inBuf[base+10u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularIor = inBuf[base+11u];
	mat.coatWeight = inBuf[base+12u];
	mat.coatRoughness = inBuf[base+13u];
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = inBuf[base+14u];
	mat.coatDarkening = inBuf[base+15u];
	mat.coatColor = f32v3(inBuf[base+16u]);
	mat.fuzzWeight = inBuf[base+17u];
	mat.fuzzColor = f32v3(inBuf[base+18u]);
	mat.fuzzRoughness = inBuf[base+19u];
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(0.0f);
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;

	const MaterialTableHandles tables = MaterialTableHandles(0u, pc.zeltnerLtcParamHandle);

	((f32v3 *)(pc.outVa))[i] = (
		openPbrIblEvaluateF(
			tables, mat, muI, dotCoatNorWi, irradiance,
			prefilteredSpecularBase, prefilteredSpecularCoat,
			/*isInsideMedium=*/ false
		)
	);
}
