import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-metallic";
import "util-material-openpbr-shared";

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

// per sample, 14 floats: wi.xyz, wo.xyz, baseWeight, baseColor.rgb,
// specularWeight, specularColor (single scalar, splatted to rgb),
// specularRoughness, specularRoughnessAnisotropy. anisotropy rotation is
// fixed at 0, coat off, thin-film off. surface normal fixed to (0,0,1),
// wi/wo supplied directly in that local space.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 14u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);

	OpenPbrMaterial mat;
	mat.baseWeight = inBuf[base+6u];
	mat.baseColor = (
		f32v3(inBuf[base+7u], inBuf[base+8u], inBuf[base+9u])
	);
	mat.specularWeight = inBuf[base+10u];
	mat.specularColor = f32v3(inBuf[base+11u]);
	mat.specularRoughness = inBuf[base+12u];
	mat.specularRoughnessAnisotropy = inBuf[base+13u];
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;
	mat.thinFilmWeight = 0.0f;

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 f = openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo);
	((f32v3 *)(pc.outVa))[i] = f;
}
