import "util-shading-frame";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 10 floats: tanX.xyz, tanY.xyz, nor.xyz, rotation. output, 9
// floats: rotatedTanX.xyz, rotatedTanY.xyz, rotatedNor.xyz

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 inBase = i * 10u;
	ShadingFrame frame;
	frame.tanX = (
		f32v3(inBuf[inBase+0u], inBuf[inBase+1u], inBuf[inBase+2u])
	);
	frame.tanY = (
		f32v3(inBuf[inBase+3u], inBuf[inBase+4u], inBuf[inBase+5u])
	);
	frame.nor = (
		f32v3(inBuf[inBase+6u], inBuf[inBase+7u], inBuf[inBase+8u])
	);
	const f32 rotation = inBuf[inBase+9u];

	ShadingFrame rotated = shadingFrameRotate(frame, rotation);

	f32 * outBuf = (f32 *)(pc.outVa);
	u32 outBase = i * 9u;
	outBuf[outBase+0u] = rotated.tanX.x; outBuf[outBase+1u] = rotated.tanX.y;
	outBuf[outBase+2u] = rotated.tanX.z;
	outBuf[outBase+3u] = rotated.tanY.x; outBuf[outBase+4u] = rotated.tanY.y;
	outBuf[outBase+5u] = rotated.tanY.z;
	outBuf[outBase+6u] = rotated.nor.x; outBuf[outBase+7u] = rotated.nor.y;
	outBuf[outBase+8u] = rotated.nor.z;
}
