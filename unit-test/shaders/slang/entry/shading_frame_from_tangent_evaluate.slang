import "util-shading-frame";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 6 floats: nor.xyz, tangent.xyz. output, 9 floats: tanX.xyz,
// tanY.xyz, nor.xyz (passthrough)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 inBase = i * 6u;
	const f32v3 nor = (
		f32v3(inBuf[inBase+0u], inBuf[inBase+1u], inBuf[inBase+2u])
	);
	const f32v3 tangent = (
		f32v3(inBuf[inBase+3u], inBuf[inBase+4u], inBuf[inBase+5u])
	);

	ShadingFrame frame = shadingFrameFromTangent(nor, tangent);

	f32 * outBuf = (f32 *)(pc.outVa);
	u32 outBase = i * 9u;
	outBuf[outBase+0u] = frame.tanX.x; outBuf[outBase+1u] = frame.tanX.y;
	outBuf[outBase+2u] = frame.tanX.z;
	outBuf[outBase+3u] = frame.tanY.x; outBuf[outBase+4u] = frame.tanY.y;
	outBuf[outBase+5u] = frame.tanY.z;
	outBuf[outBase+6u] = frame.nor.x; outBuf[outBase+7u] = frame.nor.y;
	outBuf[outBase+8u] = frame.nor.z;
}
