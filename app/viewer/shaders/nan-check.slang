import "util-global-pc";
import "util-nan-check-pc";

struct PC {
	GpuGlobalPc global;
	GpuNanCheckPC nanCheck;
};

[[vk::push_constant]] PC pc;

// scans the raw per-frame radiance image for broken pixels and tallies
// them into a host-readable counter buffer. two things count as broken:
//   * a true NaN/Inf bit pattern in any channel -- these reach the image
//     when a path escapes utilIrradiancePropagationEntry's NaN guard (a
//     NaN accumulator with no recorded hit is returned as-is)
//   * the guard's own sentinel: NaN accumulators WITH a hit are rewritten
//     to exactly (9999, 0, 0) so they show as pure red instead of
//     poisoning the accumulation buffer; the write is a constant, so an
//     exact float compare recovers it here

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const i32v2 coord = (
		i32v2(gl_GlobalInvocationID.xy)
		+ i32v2(pc.nanCheck.tileOriginX, pc.nanCheck.tileOriginY)
	);
	if (coord.x >= i32(pc.nanCheck.width) || coord.y >= i32(pc.nanCheck.height)) {
		return;
	}
	const f32v4 v = imageLoad(vkofImages[pc.nanCheck.inputHandle], coord);

	u32 * counts = (u32 *)(pc.nanCheck.countsVa);
	const bool isSentinel = (
		v.r == 9999.0f && v.g == 0.0f && v.b == 0.0f
	);
	if (
		isSentinel
		|| isnan(v.r) || isnan(v.g) || isnan(v.b) || isnan(v.a)
	) {
		atomicAdd(&counts[0], 1u);
		return;
	}
	if (isinf(v.r) || isinf(v.g) || isinf(v.b) || isinf(v.a)) {
		atomicAdd(&counts[1], 1u);
	}
}
