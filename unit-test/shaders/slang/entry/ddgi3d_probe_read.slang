import "util-types";

struct PC {
	u32v4 rootPad[8];
	u64 dstVa;
	u32 imgHandle;
	u32 countX;
	u32 countY;
	u32 countZ;
};

[[vk::push_constant]] PC pc;

// one invocation per probe; reads the center texel of each 8x8 probe tile

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 idx = gl_GlobalInvocationID.x;
	u32 total = pc.countX * pc.countY * pc.countZ;
	if (idx >= total) { return; }
	u32 px = idx % pc.countX;
	u32 py = (idx / pc.countX) % pc.countY;
	u32 pz = idx / (pc.countX * pc.countY);
	f32 val = imageLoad(
		vkofImages3D[NonUniformResourceIndex(pc.imgHandle)],
		i32v3(i32(px) * 8 + 4, i32(py) * 8 + 4, i32(pz))
	).r;
	((f32 *)(pc.dstVa))[idx] = val;
}
