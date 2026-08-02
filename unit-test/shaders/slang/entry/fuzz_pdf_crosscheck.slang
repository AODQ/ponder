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

// algebraic cross-check of the ported zeltner ltc fuzz pdf
// (openPbrFuzzPdf) against materialx's mx_zeltner_sheen_brdf, independent
// of sampling/xi: both are the same LTC-pushed-cosine density derivation
// (heitz et al. 2016 + zeltner et al. 2022), so feeding materialx's own
// aInv/bInv coefficients straight into openPbrFuzzPdf as (fuzzA, fuzzB)
// must reproduce mx_zeltner_sheen_brdf's return value to fp precision at
// every (wi, wo, roughness), not just at sampler-reachable wo. this is the
// strongest available check since it pins the port's t/lenT2 formula
// directly against an independently-authored implementation, with no RNG
// or sampler-support subtleties involved.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 7 floats in: wi.xyz, wo.xyz, roughness (canonical local frame,
// nor = (0,0,1)). output 2 floats: minePdf, refPdf.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 7u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32 roughness = inBuf[base+6u];

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// mx_sheen_bsdf clamps both at the call site before reaching
	// mx_zeltner_sheen_brdf / the ltc coefficient fits
	const f32 ndotV = clamp(dot(nor, wi), 0.0f, 1.0f);
	const f32 roughnessClamped = clamp(roughness, 0.01f, 1.0f);

	const f32 aInv = mx_zeltner_sheen_ltc_aInv(ndotV, roughnessClamped);
	const f32 bInv = mx_zeltner_sheen_ltc_bInv(ndotV, roughnessClamped);

	const f32 minePdf = openPbrFuzzPdf(nor, wi, wo, aInv, bInv);
	const f32 refPdf = (
		mx_zeltner_sheen_brdf(wo, wi, nor, ndotV, roughnessClamped)
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*2u+0u] = minePdf;
	outBuf[i*2u+1u] = refPdf;
}
