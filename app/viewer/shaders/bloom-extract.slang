import "util-global-pc";
import "util-bloom-extract-pc";

struct PC {
	GpuGlobalPc global;
	GpuBloomExtractPC bloomExtract;
};

[[vk::push_constant]] PC pc;

// bright-pass extract, ported from ~/repo/demo/shaders/bloom_extract.comp

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	const i32v2 screensize = i32v2(unf.renderWidth, unf.renderHeight);
	if (coord.x >= screensize.x || coord.y >= screensize.y) { return; }

	f32v3 col = (
		imageLoad(vkofImages[pc.bloomExtract.inputHandle], coord).rgb
		* pc.global.exposure
	);
	const f32 brightness = max(col.r, max(col.g, col.b));
	f32 soft = clamp(brightness - pc.bloomExtract.threshold, 0.0f, 1.0f);
	// normalize so dim pixels fade out softly instead of a hard cutoff
	soft /= max(brightness, 1e-4f);
	col *= soft;

	imageStore(
		vkofImages[pc.bloomExtract.outputHandle], coord, f32v4(col, 1.0f)
	);
}
