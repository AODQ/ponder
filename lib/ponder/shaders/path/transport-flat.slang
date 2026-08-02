implementing path;

import types;
import util;
import light; // TODO only for regirVisible
import model;

namespace vkof {

// -----------------------------------------------------------------------------
// flat metric tensor will produce a single ray per path.
// will still support media stack events
public struct TraceTransport_TensorMinkowski
	: I_TraceTransport
{
	// the medium stack belongs to the medium transport
	public __init(vkof::WavefrontTransportState state) {
		this.ori = state.ori;
		this.dir = state.dir;
		this.λ = state.λ;
		this.cameraTime = state.cameraTime;
		this.pathLength = state.pathLength;
	}

	public func state() -> vkof::WavefrontTransportState {
		var state: vkof::WavefrontTransportState;
		state.ori = this.ori;
		state.dir = this.dir;
		state.λ = this.λ;
		state.cameraTime = this.cameraTime;
		state.pathLength = this.pathLength;
		state.mediumStack = vkof::MediumStack();
		return state;
	}

	[mutating] public func traceTransport(
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit
	{
		// -- perform a single ray query
		ImplTraceStepHit stepHit = (
			implTraceStep(
				/*ori=*/ this.ori,
				/*dir=*/ this.dir,
				/*distMin=*/ rayEpsilon(this.ori),
				/*distMax=*/ maxDistance,
				/*alphaU=*/ stream.nextF32()
			)
		);

		let hit: TraceTransportHit = (
			implTransportHitFromStep(
				/*stepHit=*/ stepHit,
				/*chordOri=*/ this.ori,
				/*chordDirection=*/ this.dir,
				/*elapsedBefore=*/ 0.0f
			)
		);
		// update the transport state to the new position
		this.ori = hit.position;
		this.pathLength += hit.elapsedTime;
		return hit;
	}

	// visibility query that returns the surface that was hit
	func visibilityQueryWithHit(
		origin: f32v3,
		direction: f32v3,
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit
	{
		// -- offset to avoid self-intersection
		let target: f32v3 = origin + direction * maxDistance;
		let distMin: f32 = rayEpsilon(origin);
		let distMax: f32 = (
			max(maxDistance - 2.0f * rayEpsilon(target), distMin)
		);
		// -- perform a single ray query
		ImplTraceStepHit stepHit = (
			implTraceStep(
				/*ori=*/ origin,
				/*dir=*/ direction,
				/*distMin=*/ distMin,
				/*distMax=*/ distMax,
				/*alphaU=*/ stream.nextF32()
			)
		);
		return implTransportHitFromStep(stepHit, origin, direction, 0.0f);
	}

	public func visibilityQuery(
		ori: f32v3,
		target: f32v3,
		inout vkof::SampleStream stream,
	) -> vkof::TraceTransportConnection
	{
		let direction: f32v3 = normalize(target - ori);
		let maxDistance: f32 = length(target - ori);
		let hit: TraceTransportHit = (
			this.visibilityQueryWithHit(
				/*origin=*/ ori,
				/*direction=*/ direction,
				/*maxDistance=*/ maxDistance,
				/*stream=*/ stream
			)
		);
		if (hit.event == vkof::TransportEvent::Surface) {
			// connection is blocked by a surface
			return vkof::traceTransportConnectionNull();
		} else {
			// connection is clear, in flat metric space the connection is
			// trivial
			return TraceTransportConnection(
				/*visible=*/ true,
				/*directionDeparture=*/ direction,
				/*directionArrival=*/ direction,
				/*transmittance=*/ vkof::Spectrum::one(),
				/*spectralShift=*/ 1.0f,
				// for straight ray, equivalent to solid angle subtended by distance
				/*areaJacobian=*/ 1.0f / (maxDistance * maxDistance)
			);
		}
	}

	[mutating] public func newDirection(dir: f32v3) -> void {
		this.dir = dir;
	}

	[mutating] public func terminateSecondaryWavelengths() -> void {
		this.λ = vkof::wavelengthsTerminateSecondary(this.λ);
	}

	public func wavelengths() -> Wavelengths {
		return this.λ;
	}

	// since the ray is travelling backwards from camera to light, then the
	// actual time during its path is reversed and before the current camera time
	public func emitTime() -> f32 {
		return this.cameraTime - this.pathLength;
	}

	public f32v3 ori;
	public f32v3 dir;
	Wavelengths λ;
	// shutter time the light reaches the camera, fixed for the path
	f32 cameraTime;
	// meters travelled from the camera
	public f32 pathLength;
};

// -----------------------------------------------------------------------------

// cheap standlone binary visibility query
public func traceTransportVisibilityQuery(
	origin: f32v3,
	target: f32v3,
) -> bool
{
	// TODO this is code shared from above
	let maxDistance: f32 = length(target - origin);
	let direction: f32v3 = normalize(target - origin);
	// -- offset to avoid self-intersection
	let target: f32v3 = origin + direction * maxDistance;
	let distMin: f32 = rayEpsilon(origin);
	let distMax: f32 = (
		max(maxDistance - 2.0f * rayEpsilon(target), distMin)
	);
	// -- perform a single ray query
	ImplTraceStepHit stepHit = (
		implTraceStep(
			/*ori=*/ origin,
			/*dir=*/ direction,
			/*distMin=*/ distMin,
			/*distMax=*/ distMax
		)
	);
	return !stepHit.hit();
}

} // namespace vkof
