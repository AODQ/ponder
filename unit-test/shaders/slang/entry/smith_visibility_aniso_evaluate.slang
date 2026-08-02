#include "../util-microfacet-prelude.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 8 floats: wiLocal.xyz, woLocal.xyz, alpha.xy

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32v3 wiLocal = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 woLocal = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32v2 alpha = f32v2(inBuf[base+6u], inBuf[base+7u]);
	((f32 *)(pc.outVa))[i] = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
}
