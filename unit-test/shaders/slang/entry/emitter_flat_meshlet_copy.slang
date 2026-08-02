import "util-types";

struct PushConstants {
	u32v4 rootPad[8];
	u64 srcVa;
	u64 dstVa;
	u32 count;
};

[[vk::push_constant]] PushConstants pc;

// copies a u32 array reachable only by device address (mor's flatMeshlets,
// which has no vkof::Buffer handle on the cpu side) into a test-owned buffer
// so it can be read back and cross-checked

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 idx = gl_GlobalInvocationID.x;
	if (idx >= pc.count) { return; }
	((u32 *)(pc.dstVa))[idx] = ((u32 *)(pc.srcVa))[idx];
}
