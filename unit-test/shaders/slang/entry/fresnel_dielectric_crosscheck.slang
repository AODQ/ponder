import "util-fresnel";
#include "../util-fresnel-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v2(cosTheta, eta). output: f32v2(mine, materialx)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 ce = ((f32v2 *)(pc.inVa))[i];
	const f32 mine = utilFresnelDielectric(ce.x, ce.y);
	const f32 ref = mxFresnelDielectric(ce.x, ce.y);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
