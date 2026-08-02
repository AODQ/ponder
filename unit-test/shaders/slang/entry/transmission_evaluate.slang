import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-transmission";
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

// per sample, 14 floats in: wi.xyz, wo.xyz, ior, specularRoughness,
// geometryThinWalled, isInsideMedium (0/1), transmissionColor.rgb,
// transmissionDepth. output 6 floats: f.rgb (openPbrTransmissionEvaluateF),
// pdf (openPbrTransmissionPdf), tirReflectPdf
// (openPbrTransmissionTirReflectPdf). dispersedIor is passed as the flat
// ior (dispersion has its own dedicated shader). canonical local frame:
// nor = (0,0,1), wi/wo live directly in it.

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
	const f32 ior = inBuf[base+6u];

	OpenPbrMaterial mat;
	mat.specularIor = ior;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularRoughness = inBuf[base+7u];
	mat.geometryThinWalled = inBuf[base+8u];
	const bool isInsideMedium = inBuf[base+9u] > 0.5f;
	mat.transmissionColor = (
		f32v3(inBuf[base+10u], inBuf[base+11u], inBuf[base+12u])
	);
	mat.transmissionDepth = inBuf[base+13u];
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 0.0f;

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// frisvad basis, the frame the isotropic transmission path used internally
	const ShadingFrame frame = shadingFrameFromNormal(nor);

	const f32v3 f = (
		openPbrTransmissionEvaluateF(mat, frame, wi, wo, isInsideMedium, ior)
	);
	const f32 pdf = (
		openPbrTransmissionPdf(mat, frame, wi, wo, isInsideMedium, ior)
	);
	const f32 tirPdf = (
		openPbrTransmissionTirReflectPdf(mat, frame, wi, wo, isInsideMedium, ior)
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*6u+0u] = f.x;
	outBuf[i*6u+1u] = f.y;
	outBuf[i*6u+2u] = f.z;
	outBuf[i*6u+3u] = pdf;
	outBuf[i*6u+4u] = tirPdf;
	outBuf[i*6u+5u] = 0.0f;
}
