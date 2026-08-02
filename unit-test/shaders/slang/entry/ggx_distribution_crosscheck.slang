#include "../util-microfacet-prelude.slang"
#include "../util-microfacet-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 5 floats: localH.xyz, alpha.xy. output: f32v2(mine, materialx)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 5u;
	const f32v3 localH = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v2 alpha = f32v2(inBuf[base+3u], inBuf[base+4u]);
	const f32 mine = utilMicrofacetGgxDistributionAniso(localH, alpha);
	const f32 ref = mx_ggx_NDF(localH, alpha);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
