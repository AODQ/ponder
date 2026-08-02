import path;
import types;
import util;
import wavefront;

// writes every path's radiance to the output image

// mirrors WavefrontOutputPc in lib/ponder/src/wavefront.cpp
struct StagePushConstants {
	u32 outputHandle;
	// <= 0 disables; caps a sample's luminance, keeping its chromaticity
	f32 fireflyClampLuminance;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	// -- bind the global state
	vkof::globalBind(pc.global);

	// -- load the path
	let slot: Optional<vkof::WavefrontPath> = vkof::wavefrontPath(threadId.x);
	if (!slot.hasValue) { return; }
	let path: vkof::PathState = vkof::pathStateLoad(slot.value.index);
	let state: vkof::WavefrontTransportState = (
		vkof::wavefrontTransportStateLoad(slot.value.index)
	);

	// -- store the radiance
	var xyz: f32v3 = path.consumer.Li.asXyz(state.λ);
	let clamp: f32 = pc.stage.fireflyClampLuminance;
	if (clamp > 0.0f && xyz.y > clamp) {
		xyz *= clamp / xyz.y;
	}
	vkofImages[pc.stage.outputHandle][slot.value.pixel] = f32v4(xyz, 1.0f);
}
