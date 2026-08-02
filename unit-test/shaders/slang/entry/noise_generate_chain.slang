import "util-random";

struct PC {
	u32v4 rootPad[8];
	u64 dstVa;
	u32 bluenoiseHandle;
	u32 count;
};

[[vk::push_constant]] PC pc;

// single invocation (pixel 0,0): writes `count` sequential pcg32 samples
// from one stream. tests temporal/chain uniformity for one pixel, as
// opposed to noise_generate_seeds.comp's spatial (one-sample-per-pixel-
// across-many-pixels) uniformity.
[shader("compute")]
[numthreads(1, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u64 state, state2;
	f32 seed;
	f32v2 seed2;
	utilGenerateSeeds(
		state, state2, seed, seed2, u32v2(gl_GlobalInvocationID.xy),
		0u, 0u, pc.bluenoiseHandle
	);

	f32 * buf = (f32 *)(pc.dstVa);
	for (u32 i = 0u; i < pc.count; ++i) {
		buf[i] = seed;
		seed = fnSampleUniform(state);
	}
}
