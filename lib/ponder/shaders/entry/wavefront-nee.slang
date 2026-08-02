import path;
import sample;
import types;
import util;
import wavefront;

// connects nee sample to the path's last vertex

// mirrors WavefrontNeePc in lib/ponder/src/wavefront.cpp
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
	// -- bind the global state
	vkof::globalBind(pc.global);

	// -- compute the path index and load the nee record
	let slot: Optional<vkof::WavefrontPath> = vkof::wavefrontPath(threadId.x);
	if (!slot.hasValue) { return; }
	let pathIndex: u32 = slot.value.index;
	let neeRecord: vkof::PathNeeRecord = (
		vkof::pathNeeRecordLoad(pathIndex)
	);
	if (neeRecord.bounce != pc.stage.bounce) { return; }
	vkof::debugPathResume(slot.value.pixel);

	// -- load the path state and transport state
	var path: vkof::PathState = vkof::pathStateLoad(pathIndex);
	let state: vkof::WavefrontTransportState = (
		vkof::wavefrontTransportStateLoad(pathIndex)
	);
	let transport: vkof::I_TraceTransport = vkof::TraceTransport_Bend(state);

	// -- connect and evaluate
#if defined(PONDER_DEBUG_PATH)
	typealias Consumer =
		vkof::PathConsumer_DebugRecord<vkof::PathConsumer_Evaluate>;
	var consumer: Consumer = Consumer(path.consumer);
#else
	var consumer: vkof::PathConsumer_Evaluate = path.consumer;
#endif
	vkof::pathNee(transport, neeRecord, consumer);
#if defined(PONDER_DEBUG_PATH)
	path.consumer = consumer.innerConsumer;
#else
	path.consumer = consumer;
#endif

	// -- store the path state
	vkof::pathStateStore(pathIndex, path);
}
