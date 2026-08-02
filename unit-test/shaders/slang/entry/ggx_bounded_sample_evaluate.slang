import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
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

// per sample, 7 floats in: wi.xyz, alpha.xy, xi.xy. output 4 floats:
// wo.xyz, pdf at (wi -> wo). tangent frame is the canonical local frame
// tanX=(1,0,0), tanY=(0,1,0), nor=(0,0,1); wi/wo live directly in it.

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
	const f32v2 alpha = f32v2(inBuf[base+3u], inBuf[base+4u]);
	const f32v2 xi = f32v2(inBuf[base+5u], inBuf[base+6u]);

	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 wo = (
		utilMicrofacetSampleGgxBoundedWoAniso(tanX, tanY, nor, wi, alpha, xi)
	);
	const f32 pdf = (
		utilMicrofacetGgxBoundedReflectPdfAniso(tanX, tanY, nor, wi, wo, alpha)
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*4u+0u] = wo.x;
	outBuf[i*4u+1u] = wo.y;
	outBuf[i*4u+2u] = wo.z;
	outBuf[i*4u+3u] = pdf;
}
