import "util-roughness";
#include "../util-roughness-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v2(roughness, anisotropy).
// output: f32v4(mine.alphaT, mine.alphaB, spec.alphaT, spec.alphaB)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 s = ((f32v2 *)(pc.inVa))[i];
	const f32v2 mine = openPbrRoughnessAlpha(s.x, s.y);
	const f32v2 spec = openPbrRoughnessAlphaSpec(s.x, s.y);
	((f32v4 *)(pc.outVa))[i] = f32v4(mine.x, mine.y, spec.x, spec.y);
}
