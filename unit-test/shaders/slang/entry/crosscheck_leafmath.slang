import "util-roughness";
import "util-mis";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// in:  roughness, anisotropy, coatRoughness, coatWeight, pdfA, pdfB
// out: alpha.x, alpha.y, roughenedRoughness, misWeight
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 6u;
	const u32 o = thread * 4u;

	const f32v2 alpha = openPbrRoughnessAlpha(inBuf[i + 0u], inBuf[i + 1u]);
	outBuf[o + 0u] = alpha.x;
	outBuf[o + 1u] = alpha.y;
	outBuf[o + 2u] = openPbrCoatRoughenedRoughness(
		inBuf[i + 0u], inBuf[i + 2u], inBuf[i + 3u]
	);
	outBuf[o + 3u] = misBalanceWeight(inBuf[i + 4u], inBuf[i + 5u]);
}
