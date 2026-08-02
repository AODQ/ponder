import "util-global-pc";
import "util-bloom-composite-pc";

struct PC {
	GpuGlobalPc global;
	GpuBloomCompositePC bloomComposite;
};

[[vk::push_constant]] PC pc;

// additive bloom composite onto the already-tonemapped display image

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

	const f32v3 color = (
		imageLoad(vkofImages[pc.bloomComposite.colorHandle], coord).rgb
	);
	const f32v3 bloom = (
		imageLoad(vkofImages[pc.bloomComposite.bloomHandle], coord).rgb
	);
	const f32v3 outColor = (
		clamp(color + bloom * pc.bloomComposite.intensity, 0.0f, 1.0f)
	);

	imageStore(
		vkofImages[pc.bloomComposite.colorHandle], coord, f32v4(outColor, 1.0f)
	);
}
