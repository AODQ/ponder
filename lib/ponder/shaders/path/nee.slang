implementing path;

import light;
import material;
import model;
import sample;
import types;
import util;

namespace vkof {

// returns if a regir cell is visible from a position
// this allows both BSDF rays and regir-wavefront to agree on relative
// visibility, such that the BSDF can avoid using ReGIR weighting for occluded
// reservoirs.
public func regirVisible(
	cell: vkof::RegirCell,
	lightPosition: f32v3,
) -> bool
{
	return (
		vkof::traceTransportVisibilityQuery(
			/*origin=*/ cell.representativeOri,
			/*target=*/ lightPosition
		)
	);
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// connection between an emitter sample and a path vertex
struct ImplNeeConnection {
	// direction toward the emitter sample
	f32v3 ωo;
	// pdf of sampling the emitter with the bsdf technique
	// 0 when there is no usable sample
	f32 pdfEmitFromNee;
	// for NEE, emission * transmittance / pdfEmitFromBsdf
	// for regir, emission * transmittance * weightSolidAngle
	vkof::Spectrum radiance;
	// regir cell slot that was queried for this connection,
	// skRegirSlotInvalid when query failed
	u32 regirSlot;
	vkof::PathVertexNeeDebug debug;
};

// NEE scattering function evaluated at a vertex
enum ImplNeeVertexKind {
	None,
	Surface,
	Medium,
}

enum ImplNeeStrategy {
	// no usable sample; implNeeConnect returns the connection as is
	None,
	PowerCdf,
	Regir,
}

// a request to sample an emitter allowing different strategies
struct ImplNeeRequest {
	// pdf 0 until implNeeConnect fills it
	ImplNeeConnection connection;
	ImplNeeStrategy strategy;
	f32v3 position;
	f32v3 target;
	// ImplNeeStrategy::PowerCdf
	vkof::EmitterSample emitterSample;
	// ImplNeeStrategy::Regir
	vkof::RegirSelection selection;
	f32 envPdf;
};

// selects an emitter sample
func implNeeSelect(
	position: f32v3,
	// can be none for medium vertex
	normal: Optional<f32v3>,
	bounceSampler: vkof::SamplerBounceView,
	bounceCount: u32,
) -> ImplNeeRequest
{
	var request: ImplNeeRequest;
	request.strategy = ImplNeeStrategy::None;
	request.position = position;
	var connection: ImplNeeConnection;
	connection.regirSlot = vkof::skRegirSlotInvalid;
	connection.debug = vkof::pathVertexNeeDebugNone();
	// -- find a regir cell
	if (vkof::global.regirEnabled != 0u && normal.hasValue) {
		let key: vkof::RegirCellKey = vkof::regirKeyFor(position, normal);
		var slot: u32 = vkof::skRegirSlotInvalid;
		// -- jitter
		let jitterKey: vkof::RegirCellKey = (
			vkof::regirKeyFor(vkof::regirJitteredPosition(position), normal)
		);
		if (!vkof::regirKeyEqual(jitterKey, key)) {
			slot = vkof::regirLookup(jitterKey);
		}
		if (slot == vkof::skRegirSlotInvalid) {
			slot = vkof::regirLookup(key);
			// -- insert a new cell if the lookup failed
			// only do this if this is the first or second bounce, to lower
			// regir grid pressure
			if (bounceCount <= 2 && slot == vkof::skRegirSlotInvalid) {
				slot = vkof::regirFindOrInsert(key, position, normal.value);
			}
			if (slot != vkof::skRegirSlotInvalid) {
				vkof::regirRecordShadingPoint(slot, position, normal.value);
			}
		}
		connection.regirSlot = slot;
	}
	// -- sample an emitter
	connection.pdfEmitFromNee = 0.0f;
	let ξSelect: f32 = (
		bounceSampler.nextF32(vkof::BounceDimension.EmitterSelect)
	);
	let ξElement: f32 = (
		bounceSampler.nextF32(vkof::BounceDimension.EmitterElement)
	);
	let ξPosition: f32v2 = (
		bounceSampler.nextF32v2(vkof::BounceDimension.EmitterPosition)
	);
	// -- check if the regir cell has a reservoir for this emitter
	if (!vkof::regirSlotUsable(connection.regirSlot)) {
		connection.regirSlot = vkof::skRegirSlotInvalid;
	}
	connection.debug.regirSlot = connection.regirSlot;

	// -- slice off the environment pdf from the total pdf
	let envPdf: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
	request.envPdf = envPdf;
	if (
		ξSelect < 1.0f - envPdf
		&& connection.regirSlot != vkof::skRegirSlotInvalid
	) {
		request.connection = connection;
		if (vkof::global.meshNeeEnabled == 0u) { return request; }
		// -- sample a reservoir from the regir cell
		var streamReservoir: vkof::SampleStream = (
			bounceSampler.stream(vkof::BounceDimension.RegirShade)
		);
		let ξReservoir: f32v2 = streamReservoir.nextF32v2();
		request.selection = (
			vkof::regirSelectReservoirWithResampling(
				/*slot=*/ connection.regirSlot,
				/*position=*/ position,
				/*normal=*/ normal,
				/*ξ=*/ ξReservoir,
			)
		);
		if (!request.selection.valid) { return request; }
		request.strategy = ImplNeeStrategy::Regir;
		request.target = request.selection.geometry.position;
		return request;
	}

	let emitterSample: vkof::EmitterSample = (
		vkof::emitterSample(/*ξ=*/ f32v4(ξSelect, ξElement, ξPosition),)
	);
	let isEnvironment: bool = (
		emitterSample.kind == vkof::EmitterKind.Environment
	);
	let useNee: bool = (
		isEnvironment
		? vkof::global.envNeeEnabled != 0u
		: vkof::global.meshNeeEnabled != 0u
	);
	connection.pdfEmitFromNee = 0.0f;
	request.connection = connection;
	if (!useNee || emitterSample.pdf <= 0.0f) { return request; }
	request.strategy = ImplNeeStrategy::PowerCdf;
	request.emitterSample = emitterSample;
	request.target = (
		isEnvironment
		? position + emitterSample.ωo * vkof::skEnvironmentDistance
		: emitterSample.position
	);
	return request;
}

// traces the request's visibility, then its pdf and radiance
func implNeeConnect(
	request: ImplNeeRequest,
	transport: vkof::I_TraceTransport,
	inout stream: vkof::SampleStream,
	λ: vkof::Wavelengths,
) -> ImplNeeConnection
{
	var connection: ImplNeeConnection = request.connection;
	if (request.strategy == ImplNeeStrategy::None) { return connection; }
	let visibility: vkof::TraceTransportConnection = (
		transport.visibilityQuery(
			/*origin=*/ request.position,
			/*target=*/ request.target,
			/*stream=*/ stream,
		)
	);
	connection.debug.position = request.target;
	connection.debug.state = (
		visibility.visible
		? vkof::skDebugPathNeeVisible
		: vkof::skDebugPathNeeBlocked
	);
	if (request.strategy == ImplNeeStrategy::Regir) {
		connection.debug.strategy = vkof::skDebugPathNeeStrategyRegir;
		connection.debug.reservoir = request.selection.reservoirIndex;
		return implNeeConnectRegir(request, connection, visibility, λ);
	}
	connection.debug.strategy = vkof::skDebugPathNeeStrategyPowerCdf;
	if (!visibility.visible) { return connection; }

	// -- pdf and radiance
	// TODO move these to impl functions?
	let emitterSample: vkof::EmitterSample = request.emitterSample;
	let isEnvironment: bool = (
		emitterSample.kind == vkof::EmitterKind.Environment
	);
	var pdf: f32 = emitterSample.pdf;
	var emission: vkof::Spectrum;
	if (isEnvironment) {
		// probability already per solid angle
		emission = (
			vkof::environmentRadiance(emitterSample.ωo, λ)
		);
	} else { // mesh
		let mtrl: vkof::MaterialRef = (
			vkof::materialRefFromHit(emitterSample.hit)
		);
		let geometry: vkof::SurfaceGeometry = (
			vkof::surfaceGeometryFromParameters(emitterSample.hit)
		);
		// -- convert probably per-area to per-solid-angle
		let cosθ: f32 = (
			abs(dot(emitterSample.normal, -visibility.directionArrival))
		);
		if (cosθ <= 0.0f) { return connection; }
		let radiance: vkof::Spectrum = (
			vkof::materialEmission(
				/*mtrl=*/ mtrl, /*uv=*/ geometry.uv, /*lod=*/ 0.0f
			)
			.toSpectrum(λ)
		);
		pdf = emitterSample.pdf / (visibility.areaJacobian * cosθ);
		// -- compute the emission
		emission = radiance;
	}

	// -- return the connection
	connection.ωo = visibility.directionDeparture;
	connection.pdfEmitFromNee = pdf;
	connection.radiance = emission * visibility.transmittance / pdf;
	return connection;
}

// -----------------------------------------------------------------------------

func implNeeConnectRegir(
	request: ImplNeeRequest,
	connection: ImplNeeConnection,
	visibility: vkof::TraceTransportConnection,
	λ: vkof::Wavelengths,
) -> ImplNeeConnection
{
	var result: ImplNeeConnection = connection;
	let selection: vkof::RegirSelection = request.selection;
	let envPdf: f32 = request.envPdf;
	if (!visibility.visible) { return result; }
	if (visibility.areaJacobian <= 0.0f) { return result; }
	// compute emission
	let cosθ: f32 = (
		abs(dot(selection.geometry.geometricNormal, -visibility.directionArrival))
	);
	if (cosθ <= 0.0f) { return result; }
	let mtrl: vkof::MaterialRef = (
		vkof::materialRefFromHit(selection.hit)
	);
	let emission: vkof::Spectrum = (
		vkof::materialEmission(
			/*mtrl=*/ mtrl,
			/*uv=*/ selection.geometry.uv,
			/*lod=*/ 0.0f
		)
		.toSpectrum(λ)
	);
	// -- compute weight
	// W is 1/pdfArea, so convert to solid angle
	let weightSolidAngle: f32 = (
		selection.weight * visibility.areaJacobian * cosθ / (1.0f - envPdf)
	);
	result.ωo = visibility.directionDeparture;
	result.pdfEmitFromNee = (
		vkof::regirPdfArea(
			/*slot=*/ connection.regirSlot,
			/*lightIndex=*/ selection.lightIndex,
			/*lightOri=*/ selection.geometry.position,
			/*lightNor=*/ selection.geometry.geometricNormal,
		)
		/ (visibility.areaJacobian * cosθ)
	);
	result.radiance = emission*visibility.transmittance*weightSolidAngle;
	return result;
}
