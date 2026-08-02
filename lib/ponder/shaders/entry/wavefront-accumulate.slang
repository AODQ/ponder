import types;
import util;
import wavefront;

#include "include/inline-expression.slang"

// -----------------------------------------------------------------------------
// accumulate a sample into a running mean, then resolve to display

// mirrors WavefrontAccumulatePc in lib/ponder/src/wavefront.cpp
struct StagePushConstants {
	u32 sampleHandle;
	u32 accumXyzHandle;
	u32 resolvedHandle;
	u32 outputHandle;
	u32 addSample;
	// the pass's first resetCount paths overwrite instead of accumulate
	u32 resetCount;
	f32 exposure;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

// folds a weighted sample into a running mean; w carries the sample count.
// a mean (not a sum) keeps float precision at high sample counts
func accumulateMean(previous: f32v4, current: f32v4) -> f32v4 {
	let count: f32 = previous.w + current.w;
	if (count <= 0.0f) { return previous; }
	let mean: f32v3 = (
		previous.xyz + (current.xyz - previous.xyz) * (current.w / count)
	);
	return f32v4(mean, count);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	if (
		threadId.x >= vkof::global.renderWidth
		|| threadId.y >= vkof::global.renderHeight
	) {
		return;
	}
	let coord: u32v2 = threadId.xy;
	let stage: StagePushConstants = pc.stage;

	// -- the pass sampled only the pixels it had paths for
	var pathOrder: Optional<u32> = none;
	if (stage.addSample != 0u) {
		pathOrder = vkof::wavefrontPixelPathOrder(coord);
	}

	// -- xyz mean, w sample count
	let previous: f32v4 = (
		pathOrder.hasValue && pathOrder.value < stage.resetCount
		? f32v4(0.0f)
		: vkofImages[stage.accumXyzHandle][coord]
	);
	// -- accumulate the current sample into the running mean
	let accum: f32v4 = InlineExpr {
		if (!pathOrder.hasValue) { return previous; }
		let current: f32v4 = vkofImages[stage.sampleHandle][coord];
		// one non-finite sample would poison the pixel
		if (!all(isfinite(current))) { return previous; }
		return accumulateMean(previous, current);
	};
	vkofImages[stage.accumXyzHandle][coord] = accum;

	// -- resolve: linear rec.709 mean for screenshots + denoise, then display
	let linearRec709: f32v3 = vkof::xyzToLinearRec709(accum.xyz);
	vkofImages[stage.resolvedHandle][coord] = f32v4(linearRec709, accum.w);
	// negative (out-of-srgb-gamut) channels are clipped only for display
	let display: f32v3 = (
		vkof::aces(max(linearRec709, 0.0f) * stage.exposure)
	);
	vkofImages[stage.outputHandle][coord] = f32v4(display, 1.0f);
}
