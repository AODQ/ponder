implementing path;

import types;
import util;

namespace vkof {

public static const u32 skPathBounceMax = 8u;

// the state of a path to be stored between wavefront kernel bounces
public struct PathState {
	public PathConsumer_Evaluate consumer;
	ImplPathScatterState previousScatter;
	int surfaceBounce;
	public bool alive;

	public __init(vkof::Wavelengths λ) {
		this.consumer = PathConsumer_Evaluate(λ);
		this.previousScatter = implPathScatterStateCamera();
		this.surfaceBounce = 0;
		this.alive = true;
	}

	// consumer is passed separately so it can be wrapped, e.g. by a recorder
	[mutating] public func shade<C : I_PathConsumer>(
		inout vkof::I_TraceTransport transport,
		record: PathHitRecord,
		sampler: vkof::Sampler,
		bounce: u32,
		cameraTime: f32,
		inout C consumer,
	) -> PathNeeRecord {
		var neeRecord: PathNeeRecord;
		this.alive = (
			samplePathShade(
				/*transport=*/ transport,
				/*hit=*/ record.hit,
				/*stream=*/ record.stream,
				/*sampler=*/ sampler,
				/*bounce=*/ bounce,
				/*cameraTime=*/ cameraTime,
				/*consumer=*/ consumer,
				/*previousScatter=*/ this.previousScatter,
				/*surfaceBounce=*/ this.surfaceBounce,
				/*neeRecord=*/ neeRecord
			)
		);
		return neeRecord;
	}
};

public func pathNee<C : I_PathConsumer>(
	transport: vkof::I_TraceTransport,
	neeRecord: PathNeeRecord,
	inout C consumer,
) {
	samplePathNee(transport, neeRecord, consumer);
}

// what the trace kernel hands to the shade kernel
public struct PathHitRecord {
	public TraceTransportHit hit;
	// the trace's sample stream, which nee continues
	vkof::SampleStream stream;
};

public func pathTrace(
	inout vkof::I_TraceTransport transport,
	sampler: vkof::Sampler,
	bounce: u32,
) -> PathHitRecord {
	var record: PathHitRecord;
	record.hit = samplePathTrace(transport, sampler, bounce, record.stream);
	return record;
}

public func pathStateLoad(pathIndex: u32) -> PathState {
	let states: PathState * = (PathState *)(vkof::global.wavefrontPathStates);
	return states[pathIndex];
}

public func pathStateStore(pathIndex: u32, state: PathState) {
	let states: PathState * = (PathState *)(vkof::global.wavefrontPathStates);
	states[pathIndex] = state;
}

public func pathHitRecordLoad(pathIndex: u32) -> PathHitRecord {
	let records: PathHitRecord * = (
		(PathHitRecord *)(vkof::global.wavefrontHitRecords)
	);
	return records[pathIndex];
}

public func pathHitRecordStore(pathIndex: u32, record: PathHitRecord) {
	let records: PathHitRecord * = (
		(PathHitRecord *)(vkof::global.wavefrontHitRecords)
	);
	records[pathIndex] = record;
}

public func pathNeeRecordLoad(pathIndex: u32) -> PathNeeRecord {
	let records: PathNeeRecord * = (
		(PathNeeRecord *)(vkof::global.wavefrontNeeRecords)
	);
	return records[pathIndex];
}

public func pathNeeRecordStore(pathIndex: u32, record: PathNeeRecord) {
	let records: PathNeeRecord * = (
		(PathNeeRecord *)(vkof::global.wavefrontNeeRecords)
	);
	records[pathIndex] = record;
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implPathScatterStateCamera() -> ImplPathScatterState {
	var previousScatter: ImplPathScatterState;
	previousScatter.position = f32v3(0.0f);
	previousScatter.pdfBsdf = 0.0f;
	previousScatter.direction = f32v3(0.0f);
	// the camera does no nee, so directly visible emitters keep full weight
	previousScatter.neeAvailable = false;
	previousScatter.regirSlot = vkof::skRegirSlotInvalid;
	return previousScatter;
}
