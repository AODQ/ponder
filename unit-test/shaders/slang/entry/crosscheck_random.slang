import "util-random";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// in:  seedLo, seedHi (bit patterns as floats), u0, u1
// out: hash (as float), 4 consecutive uniform draws, sphere.xyz
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 4u;
	const u32 o = thread * 8u;

	const u32 lo = asuint(inBuf[i + 0u]);
	const u32 hi = asuint(inBuf[i + 1u]);
	u64 state = (u64(hi) << 32) | u64(lo);

	outBuf[o + 0u] = asfloat(fnPcgHash(lo));
	outBuf[o + 1u] = fnSampleUniform(state);
	outBuf[o + 2u] = fnSampleUniform(state);
	outBuf[o + 3u] = fnSampleUniform(state);
	outBuf[o + 4u] = fnSampleUniform(state);
	const f32v3 dir = fnSampleUniformSphere(
		f32v2(inBuf[i + 2u], inBuf[i + 3u])
	);
	outBuf[o + 5u] = dir.x;
	outBuf[o + 6u] = dir.y;
	outBuf[o + 7u] = dir.z;
}
