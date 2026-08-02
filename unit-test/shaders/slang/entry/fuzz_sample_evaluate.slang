import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-fuzz";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// evaluate shader for the ported zeltner ltc fuzz sampler + its own pdf at
// the sampled direction (openPbrFuzzSampleWo / openPbrFuzzPdf).
// openPbrFuzzSampleWo takes its uniform pair xi directly, same as the ggx
// samplers, giving direct per-sample control over xi (used for golden/
// degenerate/sweep cases; the real stream-drawn path is covered separately
// by the chi-square harness in sampling_histogram.comp).

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 10 floats in: nor.xyz, wi.xyz, fuzzA, fuzzB, xi.xy. output 4
// floats: wo.xyz, pdf(wo).

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 10u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wi = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32 fuzzA = inBuf[base+6u];
	const f32 fuzzB = inBuf[base+7u];
	const f32v2 xi = f32v2(inBuf[base+8u], inBuf[base+9u]);

	const f32v3 wo = openPbrFuzzSampleWo(nor, wi, fuzzA, fuzzB, xi);
	const f32 pdf = openPbrFuzzPdf(nor, wi, wo, fuzzA, fuzzB);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*4u+0u] = wo.x;
	outBuf[i*4u+1u] = wo.y;
	outBuf[i*4u+2u] = wo.z;
	outBuf[i*4u+3u] = pdf;
}
