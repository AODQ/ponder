import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-shared";
import bsdf;
import material;

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// direct evaluation of the lib's vkof::bsdfOpenPbrThinfilmFresnel (rgb),
// bypassing the metallic/specular callers entirely. per sample, 9 floats:
//   cosThetaI, thinFilmThickness, thinFilmIor, etaBase.rgb, kappaBase.rgb

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 9u;
	const f32 cosThetaI = inBuf[base+0u];

	var params: vkof::MaterialBackend_OpenPbr.Params;
	params.thinFilmThickness = inBuf[base+1u];
	params.thinFilmIor = inBuf[base+2u];

	const f32v3 etaBase = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32v3 kappaBase = (
		f32v3(inBuf[base+6u], inBuf[base+7u], inBuf[base+8u])
	);

	((f32v3 *)(pc.outVa))[i] = (
		vkof::bsdfOpenPbrThinfilmFresnel(params, cosThetaI, etaBase, kappaBase)
	);
}
