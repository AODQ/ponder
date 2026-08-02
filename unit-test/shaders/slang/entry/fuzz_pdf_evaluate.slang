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

// evaluate shader for the ported zeltner ltc fuzz pdf (openPbrFuzzPdf) at an
// arbitrary (nor, wi, wo), independent of the sampler. used for the
// pdf-integrates-to-1 quadrature, rotation-invariance checks (nor is a free
// input, unlike the canonical-frame-only sampler evaluate shader) and
// degenerate/NaN sweeps.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 11 floats in: nor.xyz, wi.xyz, wo.xyz, fuzzA, fuzzB. output 1
// float: openPbrFuzzPdf(nor, wi, wo, fuzzA, fuzzB).

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 11u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wi = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+6u], inBuf[base+7u], inBuf[base+8u])
	);
	const f32 fuzzA = inBuf[base+9u];
	const f32 fuzzB = inBuf[base+10u];

	((f32 *)(pc.outVa))[i] = openPbrFuzzPdf(nor, wi, wo, fuzzA, fuzzB);
}
