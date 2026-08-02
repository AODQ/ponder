#include "../util-microfacet-prelude.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v3(dotNorWi, dotNorWo, alpha)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v3 s = ((f32v3 *)(pc.inVa))[i];
	((f32 *)(pc.outVa))[i] = utilMicrofacetSmithGgxVisibility(s.x, s.y, s.z);
}
