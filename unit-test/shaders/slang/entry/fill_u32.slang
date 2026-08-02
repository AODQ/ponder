import "util-types";

struct PushConstants {
	u32v4 rootPad[8];
	u64 dstVa;
	u32 value;
	u32 count;
};

[[vk::push_constant]] PushConstants pc;

// Va pointer to a u32 array

// Per-dispatch push constants start at byte 128 (root constants occupy [0,128))

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 idx = gl_GlobalInvocationID.x;
	if (idx >= pc.count) { return; }
	((u32 *)(pc.dstVa))[idx] = pc.value;
}
