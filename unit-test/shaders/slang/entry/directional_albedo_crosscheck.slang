import "util-energy-compensation";
#include "../util-microfacet-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v4(dotNorWi, alpha, f0, f90). output: f32v2(mine, materialx)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v4 s = ((f32v4 *)(pc.inVa))[i];
	const f32 mine = utilMicrofacetGgxDirectionalAlbedo(s.x, s.y, s.z, s.w);
	const f32 ref = mx_ggx_dir_albedo_analytic(s.x, s.y, s.z, s.w);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
