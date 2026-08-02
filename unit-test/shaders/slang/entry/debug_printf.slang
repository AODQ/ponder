import "util-types";

struct PushConstants {
	u32v4 rootPad[8];
	u32 value;
};

[[vk::push_constant]] PushConstants pc;

// one invocation prints a known value so the host can check it arrived
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	if (gl_GlobalInvocationID.x != 0u) { return; }
	printf("debug-printf-probe %u\n", pc.value);
}
