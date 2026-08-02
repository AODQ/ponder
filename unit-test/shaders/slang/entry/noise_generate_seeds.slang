import "util-random";

struct PC {
	u32v4 rootPad[8];
	u64 seedOutVa;
	u64 seed2OutVa;
	u32 bluenoiseHandle;
	u32 frameIndex;
	u32 salt;
	u32 width;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 x = gl_GlobalInvocationID.x;
	u32 y = gl_GlobalInvocationID.y;
	u32 idx = y * pc.width + x;

	u64 state, state2;
	f32 seed;
	f32v2 seed2;
	utilGenerateSeeds(
		state, state2, seed, seed2, u32v2(gl_GlobalInvocationID.xy),
		pc.frameIndex, pc.salt, pc.bluenoiseHandle
	);

	((f32 *)(pc.seedOutVa))[idx] = seed;
	((f32 *)(pc.seed2OutVa))[idx * 2u + 0u] = seed2.x;
	((f32 *)(pc.seed2OutVa))[idx * 2u + 1u] = seed2.y;
}
