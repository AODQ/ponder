implementing path;

import types;
import util;
import model;
import sample;
import wavefront;
import medium;

namespace vkof {

// -----------------------------------------------------------------------------

public enum TransportEvent {
	// the path hit a surface
	Surface,
	// the path hit a media
	MediumScatter,
	// the path escaped the scene
	Escape,
	// the path is still in flight
	Pending,
	// the path is trapped or dead
	Lost,
	// the path hit an emitter
	Emission,
}

public struct TraceTransportHitSurface
{
	public f32v3 position;
	// resolves geometry and material on demand
	public SurfaceHit hit;
	public bool frontFace;
}

public struct TraceTransportHitEmission
{
	public Spectrum emission;
	// fraction of light passing through the emitter; 0 for an opaque emitter
	public f32 transmittance;
};

public struct TraceTransportHit
{
	public TransportEvent event;
	public f32v3 position;
	// direction of the ray at the intersection
	public f32v3 incomingDirection;
	// travel distance since beginning of the path
	public f32 elapsedTime;
	// lambda/g of frequency shift
	public f32 frequencyShift;
	public Spectrum spectralWeight;
	// probability density of the event being sampled at this distance
	public f32 distancePdf;
	// surface data when event==Surface
	public TraceTransportHitSurface surface;
	// emission data when event==Emission
	public TraceTransportHitEmission emission;
	// medium data when event==MediumScatter
	public vkof::MediumPoint medium;
	public f32 areaJacobian;
}

// specifies a connection between two points in a transport path
// e.g. the connection between the two vertices could be a curved path
public struct TraceTransportConnection
{
	public bool visible;
	public f32v3 directionDeparture;
	public f32v3 directionArrival;
	public Spectrum transmittance;
	public f32 spectralShift;
	public f32 areaJacobian;
};

public func traceTransportConnectionNull() -> TraceTransportConnection {
	var connection: TraceTransportConnection;
	connection.visible = false;
	connection.directionDeparture = f32v3(0.0f);
	connection.directionArrival = f32v3(0.0f);
	connection.transmittance = vkof::Spectrum::zero();
	connection.spectralShift = 1.0f;
	connection.areaJacobian = 1.0f;
	return connection;
}

// specifies the type of transport metric to use for the path
public interface I_TraceTransport
{
	public __init(vkof::WavefrontTransportState state);
	[mutating] public func traceTransport(
		maxDistance: f32,
		inout vkof::SampleStream stream,
	) -> TraceTransportHit;

	// returns the connection between two points in a transport path,
	// this travels through the transport mechanism, so the path may be curved
	public func visibilityQuery(
		origin: f32v3,
		target: f32v3,
		inout vkof::SampleStream stream,
	) -> vkof::TraceTransportConnection;
	[mutating] public func newDirection(dir: f32v3) -> void;
	[mutating] public func terminateSecondaryWavelengths() -> void;
	// fetch the wavelengths associated with this transport metric
	public func wavelengths() -> Wavelengths;
	// time the light was at the current point, in meters of light travel
	public func emitTime() -> f32;

	// returns the current wavefront transport state,
	// which can be used to resume the transport later
	public func state() -> vkof::WavefrontTransportState;
};

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implTransportHitFromStep(
	stepHit: ImplTraceStepHit,
	chordOri: f32v3,
	chordDirection: f32v3,
	elapsedBefore: f32,
) -> vkof::TraceTransportHit {
	var hit: vkof::TraceTransportHit;
	hit.position = chordOri + chordDirection * stepHit.distance;
	hit.incomingDirection = chordDirection;
	hit.elapsedTime = elapsedBefore + stepHit.distance;
	hit.frequencyShift = 1.0f;
	hit.distancePdf = 1.0f;
	hit.event = (
		stepHit.hit()
		? vkof::TransportEvent::Surface
		: vkof::TransportEvent::Escape
	);
	hit.surface.position = hit.position;
	hit.surface.hit.kind = vkof::GeometryKind.Triangle;
	hit.surface.hit.instance = stepHit.instanceId;
	hit.surface.hit.element = stepHit.primitiveId;
	hit.surface.hit.parameters = stepHit.barycentrics;
	hit.surface.frontFace = stepHit.frontFace;
	hit.spectralWeight = vkof::Spectrum::one();
	hit.areaJacobian = 1.0f / (stepHit.distance * stepHit.distance);
	return hit;
}

func implTransportHitLost(
	position: f32v3,
	dir: f32v3,
	elapsed: f32,
) -> vkof::TraceTransportHit {
	var hit: vkof::TraceTransportHit;
	hit.event = vkof::TransportEvent::Lost;
	hit.position = position;
	hit.incomingDirection = dir;
	hit.elapsedTime = elapsed;
	hit.frequencyShift = 1.0f;
	hit.distancePdf = 1.0f;
	hit.spectralWeight = vkof::Spectrum::one();
	return hit;
}

