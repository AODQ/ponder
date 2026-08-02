import types;
import util;

// -----------------------------------------------------------------------------
// resolves the DLSS ray reconstruction output like the accumulate entry:
// linear rec.709 for screenshots and bloom, then display

// mirrors DlssRrResolvePc in lib/ponder/src/dlss-rr.cpp
struct StagePushConstants {
	u32 rrOutputHandle;
	u32 resolvedHandle;
	u32 outputHandle;
	f32 exposure;
	u32 outputWidth;
	u32 outputHeight;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	let stage: StagePushConstants = pc.stage;
	// rr upscales; the output can be larger than the traced render size
	if (threadId.x >= stage.outputWidth || threadId.y >= stage.outputHeight) {
		return;
	}
	let coord: u32v2 = threadId.xy;
	let linearRec709: f32v3 = vkofImages[stage.rrOutputHandle][coord].rgb;
	// w = 1 keeps the resolved image's sample-count convention
	vkofImages[stage.resolvedHandle][coord] = f32v4(linearRec709, 1.0f);
	let display: f32v3 = (
		vkof::aces(max(linearRec709, 0.0f) * stage.exposure)
	);
	vkofImages[stage.outputHandle][coord] = f32v4(display, 1.0f);
}
