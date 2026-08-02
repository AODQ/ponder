implementing path;

import medium;
import material;

namespace vkof {

static const u32 skBendStepMax = 256u;
// the distance to traverse in each step while index an index field
static const f32 skBendStepLength = 0.25f;

// supports fractive-index fields that bend light rays
// not the same as gravitational tensors TraceTransport_Schwarzchild
public struct TraceTransport_Bend : I_TraceTransport
{
	TraceTransport_ParticipatingMedium tensor;
	vkof::I_IndexField indexField;

	// the index field is rebuilt from globals, so it carries no state
	public __init(vkof::WavefrontTransportState state) {
		this.tensor = TraceTransport_ParticipatingMedium(state);

		this.indexField = (
			vkof::IndexField_GroundHeating(
				/*groundHeight=*/ vkof::global.groundHeatingGroundHeight,
				/*hazeHeight=*/ vkof::global.groundHeatingHazeHeight,
				/*gradient=*/ vkof::global.groundHeatingGradient,
				/*noiseAmplitude=*/ vkof::global.groundHeatingNoiseAmplitude,
				/*noiseScale=*/ vkof::global.groundHeatingNoiseScale
			)
		);
	}

	[mutating] public func traceTransport(
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit {
		if (vkof::global.groundHeatingEnabled == 0u) {
			return tensor.traceTransport(maxDistance, stream);
		}
		let bounds: f32v3x2 = indexField.bounds();
		let span: f32v2 = (
			vkof::intersectRayAabb(
				/*origin=*/ tensor.flat.ori,
				/*direction=*/ tensor.flat.dir,
				/*bounds=*/ bounds
			)
		);
		let entry: f32 = max(span.x, 0.0f);
		// -- if ray does not hit the index field bounds, ignore
		if (span.y < entry || entry >= maxDistance) {
			return tensor.traceTransport(maxDistance, stream);
		}

		// -- trace until hitting the index field
		var elapsed: f32 = 0.0f;
		if (entry > 0.0f) {
			var hit: TraceTransportHit = tensor.traceTransport(entry, stream);
			if (hit.event != vkof::TransportEvent::Escape) { return hit; }
			elapsed = entry;
		}

		// -- trace bent steps through the field
		var dir: f32v3 = tensor.flat.dir;
		for (u32 step = 0u; step < skBendStepMax; ++step) {
			let x: f32v3 = tensor.flat.ori;
			// -- check if out of bounds or exceeded max distance
			if (
				elapsed >= maxDistance
				|| !vkof::intersectPointAabb(x, bounds)
			) {
				break;
			}
			// -- sample index field
			let field: vkof::IndexFieldSample = indexField.evaluate(x);
			let ds: f32 = min(skBendStepLength, maxDistance - elapsed);
			// -- ray equation of the gradient
			let bend: f32v3 = (
				(field.gradient - dot(field.gradient, dir) * dir)
				/ field.refractiveIndex
			);
			dir = normalize(dir + bend * ds);
			tensor.newDirection(dir);
			var hit: TraceTransportHit = tensor.traceTransport(ds, stream);
			vkof::debugPathPoint(hit.position, vkof::skDebugPathPointChord);
			if (hit.event != vkof::TransportEvent::Escape) {
				hit.elapsedTime += elapsed;
				// sadly for now need to invalidate the area jacobian
				hit.areaJacobian = 0.0f;
				return hit;
			}
			elapsed += ds;
		}
		// -- cast final ray in flat
		var hit: TraceTransportHit = (
			tensor.traceTransport(maxDistance - elapsed, stream)
		);
		hit.elapsedTime += elapsed;
		hit.areaJacobian = 0.0f;
		return hit;
	}

	public func visibilityQuery(
		origin: f32v3,
		target: f32v3,
		inout vkof::SampleStream stream,
	) -> vkof::TraceTransportConnection {
		if (vkof::global.groundHeatingEnabled == 0u) {
			return tensor.visibilityQuery(origin, target, stream);
		}
		let bounds: f32v3x2 = indexField.bounds();
		let span: f32v2 = (
			vkof::intersectRayAabb(
				/*origin=*/ origin,
				/*direction=*/ normalize(target - origin),
				/*bounds=*/ bounds
			)
		);
		let entry: f32 = max(span.x, 0.0f);
		// -- if ray does not hit the index field bounds, no connection
		if (span.y >= entry && entry < length(target - origin)) {
			return traceTransportConnectionNull();
		}
		return tensor.visibilityQuery(origin, target, stream);
	}

	[mutating] public func newDirection(dir: f32v3) -> void {
		tensor.newDirection(dir);
	}

	[mutating] public func terminateSecondaryWavelengths() -> void {
		tensor.terminateSecondaryWavelengths();
	}

	public func state() -> vkof::WavefrontTransportState {
		return tensor.state();
	}

	public func wavelengths() -> Wavelengths {
		return tensor.wavelengths();
	}

	public func emitTime() -> f32 {
		return tensor.emitTime();
	}
}

} // namespace vkof
