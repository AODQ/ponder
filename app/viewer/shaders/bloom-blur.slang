import "util-global-pc";
import "util-bloom-blur-pc";

struct PC {
	GpuGlobalPc global;
	GpuBloomBlurPC bloomBlur;
};

[[vk::push_constant]] PC pc;

// kawase-style widening 4-tap blur, ported from
// ~/repo/demo/shaders/bloom_blur.comp. bilinear sampling at a fractional
// texel offset (passIndex + 0.5) is the whole trick -- it's what makes 4
// taps look like a much wider soft blur, so this reads via vkofTextures[]
// (sampler2D) rather than imageLoad

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

	const f32v2 uv = (f32v2(coord) + 0.5f) / f32v2(screensize);
	const f32 offset = f32(pc.bloomBlur.passIndex) + 0.5f;
	const f32v2 texel = 1.0f / f32v2(screensize);
	const u32 h = pc.bloomBlur.inputHandle;

	f32v3 col = texture(vkofTextures[h], uv + f32v2( offset,  offset) * texel).rgb;
	col      += texture(vkofTextures[h], uv + f32v2(-offset,  offset) * texel).rgb;
	col      += texture(vkofTextures[h], uv + f32v2( offset, -offset) * texel).rgb;
	col      += texture(vkofTextures[h], uv + f32v2(-offset, -offset) * texel).rgb;
	col *= 0.25f;

	imageStore(
		vkofImages[pc.bloomBlur.outputHandle], coord, f32v4(col, 1.0f)
	);
}
