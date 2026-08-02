#include "../util-microfacet-prelude.slang"
#include "../util-microfacet-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v3(dotNorWi, dotNorWo, alpha). output: f32v2(mine, materialx)
// materialx ships raw G2, not the 1/(4*NdotL*NdotV)-folded visibility form
// cull uses, so the fold is applied here (not inside the ported reference)
// to compare the same physical quantity.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v3 s = ((f32v3 *)(pc.inVa))[i];
	const f32 mine = utilMicrofacetSmithGgxVisibility(s.x, s.y, s.z);
	const f32 ref = mx_ggx_smith_G2(s.x, s.y, s.z) / (4.0f * s.x * s.y);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
