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

// same 10-float layout as thinfilm_metallic_furnace_integrate.comp

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 10u;
	const f32 muI = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.specularRoughness = inBuf[base+1u];
	mat.specularRoughnessAnisotropy = inBuf[base+2u];
	mat.baseWeight = inBuf[base+3u];
	mat.baseColor = f32v3(inBuf[base+4u]);
	mat.specularWeight = inBuf[base+5u];
	mat.specularColor = f32v3(inBuf[base+6u]);
	mat.thinFilmWeight = inBuf[base+7u];
	mat.thinFilmThickness = inBuf[base+8u];
	mat.thinFilmIor = inBuf[base+9u];
	mat.baseMetalness = 1.0f;
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;

	const f32 roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	((f32v3 *)(pc.outVa))[i] = (
		openPbrConductorIblAlbedo(mat, muI, roughenedSpecularRoughness)
	);
}
