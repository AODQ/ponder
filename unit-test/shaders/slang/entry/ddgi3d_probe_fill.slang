import "util-types";

struct PC {
	u32v4 rootPad[8];
	u32 imgHandle;
	u32 countX;
	u32 countY;
	u32 countZ;
};

[[vk::push_constant]] PC pc;

// workgroup = one probe tile; dispatch (countX, countY, countZ)

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_LocalInvocationID : SV_GroupThreadID,
	const u32v3 gl_WorkGroupID : SV_GroupID,
) {
	u32 px = gl_WorkGroupID.x;
	u32 py = gl_WorkGroupID.y;
	u32 pz = gl_WorkGroupID.z;
	u32 lx = gl_LocalInvocationID.x;
	u32 ly = gl_LocalInvocationID.y;
	// encode (pz, px, py) so all three axes produce distinct values
	f32 val = f32(pz) * 1000.0 + f32(px) * 10.0 + f32(py);
	imageStore(
		vkofImages3D[NonUniformResourceIndex(pc.imgHandle)],
		i32v3(i32(px) * 8 + i32(lx), i32(py) * 8 + i32(ly), i32(pz)),
		f32v4(val, 0.0, 0.0, 1.0)
	);
}
