import "util-energy-compensation";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// in:  mu, alpha, f0, f90
// out: ggxAlbedo, compensate, dielectricAlbedo
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
	const u32 o = thread * 3u;
	const f32 mu = inBuf[i + 0u];
	const f32 alpha = inBuf[i + 1u];
	const f32 f0 = inBuf[i + 2u];
	const f32 f90 = inBuf[i + 3u];
	outBuf[o + 0u] = utilMicrofacetGgxDirectionalAlbedo(mu, alpha, f0, f90);
	outBuf[o + 1u] = utilMicrofacetDielectricEnergyCompensate(mu, alpha, f0);
	outBuf[o + 2u] = utilMicrofacetDielectricAlbedo(mu, alpha, f0);
}
