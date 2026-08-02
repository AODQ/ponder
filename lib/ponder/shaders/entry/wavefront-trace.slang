import light;
import path;
import sample;
import types;
import util;
import wavefront;

// traces one bounce of every live path, for wavefront-integrate to shade

// mirrors WavefrontTracePc in lib/ponder/src/wavefront.cpp
struct StagePushConstants {
	u32 bounce;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	let slot: Optional<vkof::WavefrontPath> = vkof::wavefrontPath(threadId.x);
	if (!slot.hasValue) { return; }
	let pathIndex: u32 = slot.value.index;
	let pixel: u32v2 = slot.value.pixel;
	let path: vkof::PathState = vkof::pathStateLoad(pathIndex);
	if (!path.alive) { return; }
	vkof::debugPathResume(pixel);

	let state: vkof::WavefrontTransportState = (
		vkof::wavefrontTransportStateLoad(pathIndex)
	);
	let curves: bool = (
		vkof::transportSchwarzschildCurves(state, vkof::skEnvironmentDistance)
	);
#if defined(PONDER_TRACE_CURVED)
	if (!curves) { return; }
	vkof::I_TraceTransport transport = (
		vkof::TraceTransport_TensorSchwarzschild(state)
	);
#else
	if (curves) { return; }
	vkof::I_TraceTransport transport = vkof::TraceTransport_Bend(state);
#endif

	let sampler: vkof::Sampler = (
		vkof::Sampler(pixel, slot.value.sampleIndex*4)
	);
	let record: vkof::PathHitRecord = (
		vkof::pathTrace(transport, sampler, pc.stage.bounce)
	);
	vkof::wavefrontTransportStateStore(pathIndex, transport.state());
	vkof::pathHitRecordStore(pathIndex, record);
}
