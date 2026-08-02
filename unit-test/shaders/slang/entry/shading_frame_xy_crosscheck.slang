import "util-shading-frame";
#include "../util-shading-frame-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v3(nor). output, 12 floats: mine.binormal(3),
// mine.bitangent(3), ref.b1(3), ref.b2(3)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v3 nor = ((f32v3 *)(pc.inVa))[i];

	f32v3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);

	f32v3 b1, b2;
	branchlessONB(nor, b1, b2);

	f32 * outBuf = (f32 *)(pc.outVa);
	u32 base = i * 12u;
	outBuf[base+0u] = binormal.x; outBuf[base+1u] = binormal.y;
	outBuf[base+2u] = binormal.z;
	outBuf[base+3u] = bitangent.x; outBuf[base+4u] = bitangent.y;
	outBuf[base+5u] = bitangent.z;
	outBuf[base+6u] = b1.x; outBuf[base+7u] = b1.y;
	outBuf[base+8u] = b1.z;
	outBuf[base+9u] = b2.x; outBuf[base+10u] = b2.y;
	outBuf[base+11u] = b2.z;
}
