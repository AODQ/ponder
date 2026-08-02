import "util-global-pc";
import "util-display-dither-pc";

struct PC {
	GpuGlobalPc global;
	GpuDisplayDitherPC displayDither;
};

[[vk::push_constant]] PC pc;

// triangular pdf dither (±1 lsb) for the 8-bit srgb swapchain. the blit
// encodes with the piecewise srgb curve and then quantizes, so the noise is
// added in that encoded space and decoded back

func srgbPiecewiseOetf(c: f32v3) -> f32v3 {
	return (
		select(
			c <= 0.0031308f,
			c * 12.92f,
			1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f
		)
	);
}

func srgbPiecewiseEotf(c: f32v3) -> f32v3 {
	return (
		select(
			c <= 0.04045f,
			c / 12.92f,
			pow((c + 0.055f) / 1.055f, 2.4f)
		)
	);
}

func pcg3d(v: u32v3) -> u32v3 {
	var w: u32v3 = v * 1664525u + 1013904223u;
	w.x += w.y * w.z;
	w.y += w.z * w.x;
	w.z += w.x * w.y;
	w ^= w >> 16u;
	w.x += w.y * w.z;
	w.y += w.z * w.x;
	w.z += w.x * w.y;
	return w;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const u32v2 coord = gl_GlobalInvocationID.xy;
	if (coord.x >= unf.renderWidth || coord.y >= unf.renderHeight) { return; }

	const f32v3 color = (
		saturate(
			imageLoad(vkofImages[pc.displayDither.inputHandle], i32v2(coord)).rgb
		)
	);
	// -- two uniforms per channel; their difference is triangular on (-1, 1)
	const u32v3 h0 = pcg3d(u32v3(coord, pc.displayDither.frameIndex * 2u));
	const u32v3 h1 = pcg3d(u32v3(coord, pc.displayDither.frameIndex * 2u + 1u));
	const f32v3 u0 = f32v3(h0 >> 8u) * (1.0f / 16777216.0f);
	const f32v3 u1 = f32v3(h1 >> 8u) * (1.0f / 16777216.0f);
	const f32v3 encoded = (
		saturate(srgbPiecewiseOetf(color) + (u0 - u1) * (1.0f / 255.0f))
	);
	// exact black and white have no step to hide
	const f32v3 dithered = (
		select(
			color <= 0.0f || color >= 1.0f, color, srgbPiecewiseEotf(encoded)
		)
	);

	imageStore(
		vkofImages[pc.displayDither.outputHandle],
		i32v2(coord),
		f32v4(dithered, 1.0f)
	);
}
