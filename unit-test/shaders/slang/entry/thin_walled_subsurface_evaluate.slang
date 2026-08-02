import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
#include "../util-eon-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// openPbrThinWalledSubsurfaceEvaluateF / openPbrThinWalledSubsurfacePdf
// (util-material-openpbr-diffuse.glsl) at explicit (nor, wi, wo) triples,
// plus the materialx oren-nayar reference the reflection side must match.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 14 floats in: nor.xyz, wi.xyz, wo.xyz,
// baseDiffuseRoughness, subsurfaceScatterAnisotropy, subsurfaceColor.rgb.
// 8 floats out: f.rgb, pdf, materialx reference f.rgb (the albedo-1 lobe
// shape only, unscaled by S or the split), unused

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) {
		return;
	}
	f32 * inBuf = (f32 *)(pc.inVa);
	const u32 base = i * 14u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wi = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+6u], inBuf[base+7u], inBuf[base+8u])
	);

	OpenPbrMaterial mat;
	mat.baseDiffuseRoughness = inBuf[base+9u];
	mat.subsurfaceScatterAnisotropy = inBuf[base+10u];
	mat.subsurfaceColor = (
		f32v3(inBuf[base+11u], inBuf[base+12u], inBuf[base+13u])
	);

	const f32v3 f = openPbrThinWalledSubsurfaceEvaluateF(mat, nor, wi, wo);
	const f32 pdf = (
		openPbrThinWalledSubsurfacePdf(
			dot(nor, wo), mat.subsurfaceScatterAnisotropy
		)
	);

	// the reference lobe is evaluated on the same mirrored wo the lib uses,
	// so it covers the transmission side too
	const f32 dotNorWo = dot(nor, wo);
	const f32v3 woLobe = dotNorWo < 0.0f ? wo - 2.0f * dotNorWo * nor : wo;
	const f32v3 reference = (
		mxOrenNayarCompensatedDiffuse(
			max(dot(nor, wi), 0.0f), max(dot(nor, woLobe), 0.0f),
			dot(wi, woLobe), mat.baseDiffuseRoughness, f32v3(1.0f)
		)
		* IPI
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 outBase = i * 8u;
	outBuf[outBase+0u] = f.x;
	outBuf[outBase+1u] = f.y;
	outBuf[outBase+2u] = f.z;
	outBuf[outBase+3u] = pdf;
	outBuf[outBase+4u] = reference.x;
	outBuf[outBase+5u] = reference.y;
	outBuf[outBase+6u] = reference.z;
	outBuf[outBase+7u] = 0.0f;
}
