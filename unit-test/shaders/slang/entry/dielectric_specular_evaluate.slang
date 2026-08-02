import "util-shading-frame";
import "util-fresnel";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-dielectric-specular";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 12 floats: wi.xyz, wo.xyz, specularRoughness,
// specularRoughnessAnisotropy, specularRoughnessAnisotropyRotation,
// coatRoughness, coatWeight, etaRel. surface normal is fixed to (0,0,1),
// wi/wo are supplied directly in that local space.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 12u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32 specularRoughness = inBuf[base+6u];
	const f32 specularRoughnessAnisotropy = inBuf[base+7u];
	const f32 specularRoughnessAnisotropyRotation = inBuf[base+8u];
	const f32 coatRoughness = inBuf[base+9u];
	const f32 coatWeight = inBuf[base+10u];
	const f32 etaRel = inBuf[base+11u];

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 f = unittest_openPbrDielectricSpecularEvaluateF(
		specularRoughness, specularRoughnessAnisotropy,
		specularRoughnessAnisotropyRotation, coatRoughness, coatWeight,
		frame, wi, wo, etaRel
	);
	((f32v3 *)(pc.outVa))[i] = f;
}
