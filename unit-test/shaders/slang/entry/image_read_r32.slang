import "util-types";

struct PC {
	u32v4 rootPad[8];
	u64 dstVa;
	u32 imgHandle;
	u32 width;
	u32 height;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 total = pc.width * pc.height;
	u32 idx = gl_GlobalInvocationID.x;
	if (idx >= total) { return; }
	i32v2 coord = i32v2(idx % pc.width, idx / pc.width);
	f32 val = imageLoad(vkofImages[NonUniformResourceIndex(pc.imgHandle)], coord).r;
	((f32 *)(pc.dstVa))[idx] = val;
}
