#include "../util-microfacet-prelude.slang"
#include "../util-microfacet-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v2(dotNorWi, alpha). output: f32v2(mine, materialx)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 s = ((f32v2 *)(pc.inVa))[i];
	const f32 mine = utilMicrofacetSmithG1(s.x, s.y);
	const f32 ref = mx_ggx_smith_G1(s.x, s.y);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
