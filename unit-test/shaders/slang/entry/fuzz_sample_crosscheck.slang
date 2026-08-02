import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-fuzz";
#include "../util-microfacet-materialx-reference.slang"
#include "../util-fuzz-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// cross-check of the ported zeltner ltc fuzz sampler (openPbrFuzzSampleWo)
// against materialx's mx_zeltner_sheen_importance_sample under matched xi.
// openPbrFuzzSampleWo takes xi directly; materialx's core cosine mapping
// takes (Xi.x -> phi, Xi.y -> cosTheta) while ponder's takes
// (xi.x -> cosTheta, xi.y -> phi), so materialx is fed the swizzled xi.yx
// (same documented convention as diffuse_sample_crosscheck.comp).
// materialx's aInv/bInv are fed straight into openPbrFuzzSampleWo as
// (fuzzA, fuzzB), same rationale as fuzz_pdf_crosscheck.comp.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 6 floats in: wi.xyz, roughness, xi.xy (canonical local
// frame, nor = (0,0,1)). output 8 floats: mineWo.xyz, refWo.xyz, minePdf,
// refPdf.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 6u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32 roughness = inBuf[base+3u];
	const f32v2 xi = f32v2(inBuf[base+4u], inBuf[base+5u]);

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	const f32 ndotV = clamp(dot(nor, wi), 0.0f, 1.0f);
	const f32 roughnessClamped = clamp(roughness, 0.01f, 1.0f);

	const f32 aInv = mx_zeltner_sheen_ltc_aInv(ndotV, roughnessClamped);
	const f32 bInv = mx_zeltner_sheen_ltc_bInv(ndotV, roughnessClamped);

	const f32v3 mineWo = openPbrFuzzSampleWo(nor, wi, aInv, bInv, xi);
	const f32 minePdf = openPbrFuzzPdf(nor, wi, mineWo, aInv, bInv);

	f32 refPdf;
	const f32v3 refWo = (
		mx_zeltner_sheen_importance_sample(
			xi.yx, wi, nor, roughnessClamped, refPdf
		)
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*8u+0u] = mineWo.x;
	outBuf[i*8u+1u] = mineWo.y;
	outBuf[i*8u+2u] = mineWo.z;
	outBuf[i*8u+3u] = refWo.x;
	outBuf[i*8u+4u] = refWo.y;
	outBuf[i*8u+5u] = refWo.z;
	outBuf[i*8u+6u] = minePdf;
	outBuf[i*8u+7u] = refPdf;
}
