import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-ibl";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// same 11-float layout as coat_ibl_furnace_integrate.comp

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 11u;
	const f32 muI = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.coatWeight = inBuf[base+1u];
	mat.coatRoughness = inBuf[base+2u];
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = inBuf[base+3u];
	mat.coatDarkening = inBuf[base+4u];
	mat.coatColor = f32v3(inBuf[base+5u]);
	mat.specularRoughness = inBuf[base+6u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularWeight = inBuf[base+7u];
	mat.specularIor = inBuf[base+8u];
	mat.baseMetalness = inBuf[base+9u];
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(inBuf[base+10u]);
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(0.0f);
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(0.0f);

	const OpenPbrCoatIblResult result = (
		openPbrCoatIblEvaluate(mat, muI, f32v3(1.0f), /*isInsideMedium=*/ false)
	);
	((f32v3 *)(pc.outVa))[i] = (
		result.ownReflectance + result.attenuation * f32v3(1.0f)
	);
}
