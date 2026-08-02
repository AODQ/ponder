import "util-types";

struct PC {
	u32v4 rootPad[8];
	u32 imgHandle;
	u32 W;
	u32 H;
	u32 D;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 total = pc.W * pc.H * pc.D;
	u32 idx = gl_GlobalInvocationID.x;
	if (idx >= total) { return; }
	u32 x = idx % pc.W;
	u32 y = (idx / pc.W) % pc.H;
	u32 z = idx / (pc.W * pc.H);
	f32 val = f32(idx) / f32(total);
	imageStore(
		vkofImages3D[NonUniformResourceIndex(pc.imgHandle)],
		i32v3(x, y, z),
		f32v4(val, 0.0, 0.0, 1.0)
	);
}
