implementing path;

import medium;
import material;

namespace vkof {

static const u32 skMediumCrossingMax = 16u;

public struct TraceTransport_ParticipatingMedium
	: I_TraceTransport
{
	TraceTransport_TensorMinkowski flat;
	vkof::MediumStack mediumStack;

	public __init(vkof::WavefrontTransportState state) {
		this.flat = TraceTransport_TensorMinkowski(state);
		this.mediumStack = state.mediumStack;
	}

	public func state() -> vkof::WavefrontTransportState {
		var state: vkof::WavefrontTransportState = this.flat.state();
		state.mediumStack = this.mediumStack;
		return state;
	}

	[mutating] public func traceTransport(
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit {
		let λ: Wavelengths = flat.wavelengths();
		var weight: Spectrum = Spectrum::one();
		var elapsed: f32 = 0.0f;
		// -- iterate through medium crossings
		for (u32 crossing = 0u; crossing < skMediumCrossingMax; ++crossing) {
			let origin: f32v3 = flat.ori;
			let dir: f32v3 = flat.dir;
			let pathStart: f32 = flat.pathLength;
			var hit: TraceTransportHit = (
				flat.traceTransport(maxDistance - elapsed, stream)
			);

			// -- return if path was lost or escaped
			if (hit.event != vkof::TransportEvent::Surface) {
				hit.spectralWeight *= weight;
				hit.elapsedTime += elapsed;
				return hit;
			}

			// -- sample the current medium along the segment
			let medium: vkof::Medium = (
				vkof::mediumLoad(mediumStack.current(), λ)
			);
			let segment: vkof::MediumSegmentSample = (
				medium.sampleSegment(
					/*x0=*/ origin,
					/*x1=*/ hit.position,
					/*ξ=*/ stream.nextF32v2(),
					/*λ=*/ λ,
				)
			);
			weight *= segment.weight;

			// -- scatters inside the medium
			if (segment.scattered) {
				hit.event = vkof::TransportEvent::MediumScatter;
				hit.position = origin + dir * segment.distance;
				hit.medium = segment.point;
				hit.elapsedTime = elapsed + segment.distance;
				hit.spectralWeight = weight;
				hit.areaJacobian = 1.0f / (segment.distance * segment.distance);
				// rewind flat to the scattering point
				flat.ori = hit.position;
				flat.pathLength = pathStart + segment.distance;
				return hit;
			}

			// -- check if the surface is a medium boundary
			let boundary: Optional<vkof::MediumHandle> = (
				vkof::materialMediumHandle(
					hit.surface.hit.instance,
					vkof::materialIndexFromHit(hit.surface.hit)
				)
			);
			if (!boundary.hasValue) {
				hit.spectralWeight *= weight;
				hit.elapsedTime += elapsed;
				return hit;
			}

			// -- record the boundary event and update the medium stack
			if (hit.surface.frontFace) {
				mediumStack.enter(boundary.value);
			} else {
				mediumStack.exit(boundary.value);
			}

			// -- update the path
			vkof::debugPathPoint(hit.position, vkof::skDebugPathPointChord);
			elapsed += hit.elapsedTime;
			hit.areaJacobian = 1.0f / (segment.distance * segment.distance);
			flat.newDirection(dir);
		}

		// -- no more available crossings, return lost
		return implTransportHitLost(flat.ori, flat.dir, elapsed);
	}

	public func visibilityQuery(
		origin: f32v3,
		target: f32v3,
		inout vkof::SampleStream stream,
	) -> vkof::TraceTransportConnection {
		let λ: Wavelengths = flat.wavelengths();
		let direction: f32v3 = normalize(target - origin);
		let maxDistance: f32 = length(target - origin);
		// -- copy the medium stack and flat transport to trace visibility query
		var stack: vkof::MediumStack = mediumStack;
		var remaining: f32 = maxDistance;
		var transmittance: vkof::Spectrum = vkof::Spectrum::one();
		var ori: f32v3 = origin;

		// -- iterate through medium crossings
		for (u32 crossing = 0u; crossing < skMediumCrossingMax; ++crossing) {
			// -- trace the next segment of the ray
			let segmentStart: f32v3 = ori;
			let hit: TraceTransportHit = (
				flat.visibilityQueryWithHit(
					/*origin=*/ ori,
					/*direction=*/ direction,
					/*maxDistance=*/ remaining,
					/*stream=*/ stream
				)
			);
			// -- load the current medium
			let medium: vkof::Medium = (
				vkof::mediumLoad(stack.current(), λ)
			);
			// -- if nothing was hit, attenuate the ray
			if (hit.event != vkof::TransportEvent::Surface) {
				return vkof::TraceTransportConnection(
					/*visible=*/ true,
					/*directionDeparture=*/ direction,
					/*directionArrival=*/ direction,
					/*transmittance=*/ (
						transmittance * medium.transmittance(segmentStart, target, λ)
					),
					/*spectralShift=*/ 1.0f,
					/*areaJacobian=*/ 1.0f / (maxDistance * maxDistance),
				);
			}
			// -- opaque surface hit, calculate the transmittance to surface
			let boundary: Optional<vkof::MediumHandle> = (
				vkof::materialMediumHandle(
					hit.surface.hit.instance,
					vkof::materialIndexFromHit(hit.surface.hit)
				)
			);
			if (!boundary.hasValue) {
				return vkof::traceTransportConnectionNull();
			}
			// -- medium boundary hit, attenuate and cross the boundary
			transmittance *= medium.transmittance(segmentStart, hit.position, λ);
			if (hit.surface.frontFace) {
				stack.enter(boundary.value);
			} else {
				stack.exit(boundary.value);
			}
			ori = hit.position;
			remaining = length(target - ori);
		}
		return vkof::traceTransportConnectionNull();
	}

	[mutating] public func newDirection(dir: f32v3) -> void {
		flat.newDirection(dir);
	}

	[mutating] public func terminateSecondaryWavelengths() -> void {
		flat.terminateSecondaryWavelengths();
	}

	public func wavelengths() -> Wavelengths {
		return flat.wavelengths();
	}

	public func emitTime() -> f32 {
		return flat.emitTime();
	}
}

} // namespace vkof
