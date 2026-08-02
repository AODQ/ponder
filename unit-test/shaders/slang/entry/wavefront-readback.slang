// adds one wavefront integrate output image into a per-pixel f32v4 sum, so a
// test can average many frames without the viewer's accumulate node

[[vk::binding(2, 0)]] RWTexture2D<float4> vkofImages[];

struct ReadbackPc {
	uint64_t sumVa;
	uint imageHandle;
	uint width;
	uint height;
	uint pad0;
};

struct PC {
	[[vk::offset(128)]] ReadbackPc stage;
};

[[vk::push_constant]] ConstantBuffer<PC> pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(const uint3 threadId : SV_DispatchThreadID) {
	if (threadId.x >= pc.stage.width || threadId.y >= pc.stage.height) {
		return;
	}
	let sums: float4 * = (float4 *)(pc.stage.sumVa);
	let pixel: uint = threadId.y * pc.stage.width + threadId.x;
	let value: float4 = (
		vkofImages[NonUniformResourceIndex(pc.stage.imageHandle)][threadId.xy]
	);
	sums[pixel] = sums[pixel] + value;
}
