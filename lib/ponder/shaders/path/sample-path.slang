implementing path;

import bsdf;
import light;
import material;
import medium;
import model;
import sample;
import types;
import util;

#include "include/inline-expression.slang"

namespace vkof {

// a path vertex's nee sample and scattering state
public struct PathNeeRecord {
	public u32 bounce;
	ImplNeeVertexKind kind;
	ImplNeeRequest request;
	vkof::SampleStream stream;
	vkof::Spectrum β;
	vkof::Spectrum spectralWeight;
	f32v3 ωi;
	// -- surface
	vkof::Bsdf_OpenPbr bsdf;
	// -- medium
	vkof::PhaseFunction phase;
};

// the trace half of a bounce; stream continues into samplePathShade
func samplePathTrace(
	inout vkof::I_TraceTransport transport,
	sampler: vkof::Sampler,
	bounce: u32,
	out vkof::SampleStream stream,
) -> vkof::TraceTransportHit
{
	stream = (
		sampler.bounceView(bounce).stream(vkof::BounceDimension.MediumDistance)
	);
	return transport.traceTransport(
		/*maxDistance=*/ vkof::skEnvironmentDistance,
		/*stream=*/ stream
	);
}

// shades the vertex at hit, without its nee connection
// returns false if the path is finished
func samplePathShade<C : I_PathConsumer>(
	inout vkof::I_TraceTransport transport,
	hit: vkof::TraceTransportHit,
	stream: vkof::SampleStream,
	sampler: vkof::Sampler,
	bounce: u32,
	cameraTime: f32,
	inout C consumer,
	inout ImplPathScatterState previousScatter,
	inout int surfaceBounce,
	out PathNeeRecord neeRecord,
) -> bool
{
	let bounceSampler: vkof::SamplerBounceView = (
		sampler.bounceView(u32(bounce))
	);
	var λ: vkof::Wavelengths = transport.wavelengths();

	// -- nee record
	neeRecord.bounce = bounce;
	neeRecord.kind = ImplNeeVertexKind::None;
	neeRecord.stream = stream;
	neeRecord.β = consumer.throughput();
	neeRecord.spectralWeight = hit.spectralWeight;

	// -- vertex
	var vertex: PathVertex;
	vertex.bounce = u32(bounce);
	vertex.hit = hit;
	vertex.weight = vkof::Spectrum::one();
	vertex.pdfEmitFromHit = 0.0f;
	vertex.pdfBsdfFromHit = previousScatter.pdfBsdf;
	vertex.isDeltaDirac = false;
	vertex.emitBsdf = vkof::Spectrum::zero();
	vertex.transmittance = 1.0f;
	vertex.emitTime = transport.emitTime();
	vertex.pathLength = cameraTime - vertex.emitTime;
	// the vertex is still consumed when the path ends here
	var pathEnds: bool = false;

	switch (hit.event) {
		case vkof::TransportEvent::Surface: {
			implSamplePathHitEventSurface(
				/*transport=*/ transport,
				/*hit=*/ hit,
				/*bounceSampler=*/ bounceSampler,
				/*surfaceBounce=*/ surfaceBounce,
				/*previousScatter=*/ previousScatter,
				/*vertex=*/ vertex,
				/*neeRecord=*/ neeRecord,
				/*λ=*/ λ
			);
		} break;
		case vkof::TransportEvent::MediumScatter: {
			// -- sample new direction from the phase function
			let phaseSample: vkof::PhaseSample = (
				hit.medium.phase.sample(
					/*ωi=*/ hit.incomingDirection,
					/*ξ=*/ bounceSampler.nextF32v2(
						vkof::BounceDimension.Direction
					),
				)
			);
			vertex.weight = (
				hit.spectralWeight * (phaseSample.f / phaseSample.pdf)
			);
			transport.newDirection(phaseSample.ωo);
			let request: ImplNeeRequest = (
				implNeeSelect(
					/*position=*/ hit.position,
					/*normal=*/ none,
					/*bounceSampler=*/ bounceSampler,
					/*bounceCount=*/ u32(surfaceBounce)
				)
			);
			let useNee: bool = (vkof::global.envNeeEnabled != 0u);
			if (useNee) {
				neeRecord.kind = ImplNeeVertexKind::Medium;
				neeRecord.request = request;
				neeRecord.phase = hit.medium.phase;
				neeRecord.ωi = hit.incomingDirection;
				previousScatter.neeAvailable = true;
			} else {
				previousScatter.neeAvailable = false;
			}
			previousScatter.position = hit.position;
			previousScatter.pdfBsdf = phaseSample.pdf;
			previousScatter.direction = phaseSample.ωo;
			previousScatter.regirSlot = request.connection.regirSlot;
		} break;
		case vkof::TransportEvent::Lost: {
			// -- path is trapped or dead
			pathEnds = true;
		} break;
		case vkof::TransportEvent::Emission: {
			// -- a transparent emitter can be hit multiple times
			vertex.emitBsdf  = (
				hit.emission.emission * hit.spectralWeight
			);
			vertex.transmittance = hit.emission.transmittance;
			pathEnds = hit.emission.transmittance <= 0.0f;
		} break;
		case vkof::TransportEvent::Escape:
		default: {
			// -- escape to the environment
			vertex.emitBsdf = (
				vkof::environmentRadiance(hit.incomingDirection, λ)
				* hit.spectralWeight
			);
			// a camera ray escaping is the background
			if (bounce == 0u) {
				vertex.emitBsdf = (
					vertex.emitBsdf * vkof::global.environmentBackgroundIntensity
				);
			}
			// -- pdf of sampling the environment from the previous vertex
			if (
				vkof::global.envNeeEnabled != 0u
				&& previousScatter.neeAvailable
				&& hit.areaJacobian > 0.0f
			) {
				vertex.pdfEmitFromHit = (
					vkof::emitterPdfEnvironment(previousScatter.direction)
				);
			}
			pathEnds = true;
		} break;
	}

	consumer.vertex(vertex, λ);
	if (pathEnds) { return false; }

	// -- russian roulette
	if (surfaceBounce > 2) {
		// -- maximum channel of throughput
		let β: vkof::Spectrum = consumer.throughput();
#if defined(PONDER_SPECTRUM_RGB)
		let asRgb: f32v3 = β.asRgb(λ);
		let rrChance: f32 = max(max(asRgb.x, asRgb.y), asRgb.z);
#else
		let rrChance: f32 = InlineExpr {
			var maximum: f32 = 0.0f;
			for (u32 i = 0u; i < vkof::skWavelengthSet; ++i) {
				maximum = max(maximum, β.amounts[i]);
			}
			return maximum;
		};
#endif
		let rrSample: f32 = (
			bounceSampler.nextF32(vkof::BounceDimension.RussianRoulette)
		);
		let survived: bool = rrSample <= rrChance;
		consumer.russianRoulette(rrChance, survived);
		if (!survived) {
			return false;
		}
	}
	return true;
}

// connects a vertex's nee sample and evaluates its contribution
func samplePathNee<C : I_PathConsumer>(
	transport: vkof::I_TraceTransport,
	neeRecord: PathNeeRecord,
	inout C consumer,
)
{
	if (neeRecord.kind == ImplNeeVertexKind::None) { return; }
	let λ: vkof::Wavelengths = transport.wavelengths();

	// -- connect
	var stream: vkof::SampleStream = neeRecord.stream;
	let connection: ImplNeeConnection = (
		implNeeConnect(
			/*request=*/ neeRecord.request,
			/*transport=*/ transport,
			/*stream=*/ stream,
			/*λ=*/ λ
		)
	);

	// -- prepare the vertex's nee sample
	var nee: PathVertexNee;
	nee.β = neeRecord.β;
	nee.emitNee = vkof::Spectrum::zero();
	nee.pdfEmitFromNee = connection.pdfEmitFromNee;
	nee.pdfBsdfFromNee = 0.0f;
	nee.debug = connection.debug;

	// -- evaluate the scattering function along the connection
	if (connection.pdfEmitFromNee > 0.0f) {
		if (neeRecord.kind == ImplNeeVertexKind::Surface) {
			// -- compute bsdf
			nee.pdfBsdfFromNee = (
				neeRecord.bsdf.pdf(/*ωi=*/ neeRecord.ωi, /*ωo=*/ connection.ωo)
			);
			nee.emitNee = (
				neeRecord.bsdf.evaluate(
					/*ωi=*/ neeRecord.ωi, /*ωo=*/ connection.ωo
				)
				* connection.radiance * neeRecord.spectralWeight
			);
		} else {
			// -- compute phase function
			let phase: f32 = (
				neeRecord.phase.evaluate(
					/*ωi=*/ neeRecord.ωi,
					/*ωo=*/ connection.ωo
				)
			);
			nee.pdfBsdfFromNee = phase;
			nee.emitNee = (
				connection.radiance * phase * neeRecord.spectralWeight
			);
		}
	}
	// -- forward to the consumer
	consumer.vertexNee(nee, λ);
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implSamplePathHitEventSurface(
	inout transport: vkof::I_TraceTransport,
	hit: vkof::TraceTransportHit,
	bounceSampler: vkof::SamplerBounceView,
	inout surfaceBounce: int,
	inout previousScatter: ImplPathScatterState,
	inout vertex: vkof::PathVertex,
	inout neeRecord: vkof::PathNeeRecord,
	inout λ: vkof::Wavelengths,
) -> void
{
	// -- fetch surface geometry and material
	var geometry: vkof::SurfaceGeometry = (
		vkof::surfaceGeometryFromHit(
			/*hit=*/ hit.surface.hit,
			/*worldPosition=*/ hit.surface.position
		)
	);
	// -- flip normals if the ray is inside the surface
	let isInsideMedium: bool = (
		dot(geometry.geometricNormal, hit.incomingDirection) > 0.0f
	);
	if (isInsideMedium) {
		geometry.geometricNormal = -geometry.geometricNormal;
		geometry.shadingNormal = -geometry.shadingNormal;
	}
	let mtrl: vkof::MaterialRef = vkof::materialRefFromHit(hit.surface.hit);
	// -- bsdf and emission
	let params: vkof::MaterialBackend_OpenPbr.Params = (
		vkof::MaterialBackend_OpenPbr.resolve(
			/*mtrl=*/ mtrl, /*geometry=*/ geometry, /*λ=*/ λ
		)
	);
	// -- shading frames
	let facing: f32 = isInsideMedium ? -1.0f : 1.0f;
	let frame: vkof::ShadingFrame = (
		vkof::shadingFrameFromNormalMap(
			/*meshNormal=*/ geometry.shadingNormal * facing,
			/*meshTangent=*/ geometry.tangent,
			/*facing=*/ facing,
			/*geometricNormal=*/ geometry.geometricNormal,
			/*mappedNormal=*/ params.geometryNormal,
			/*ωo=*/ -hit.incomingDirection
		)
	);
	let coatFrame: vkof::ShadingFrame = (
		vkof::shadingFrameFromNormalMap(
			/*meshNormal=*/ geometry.shadingNormal * facing,
			/*meshTangent=*/ geometry.tangent,
			/*facing=*/ facing,
			/*geometricNormal=*/ geometry.geometricNormal,
			/*mappedNormal=*/ params.geometryCoatNormal,
			/*ωo=*/ -hit.incomingDirection
		)
	);

#if !defined(PONDER_SPECTRUM_RGB)
	// -- dispersion
	// only hero wavelength continues
	if (
		params.transmissionDispersionScale > 0.0f
		&& params.transmissionWeight > 0.0f
	) {
		transport.terminateSecondaryWavelengths();
		λ = transport.wavelengths();
	}
#endif

	// -- ior at the hero wavelength
#if defined(PONDER_SPECTRUM_RGB)
	let dispersedIor: f32 = vkof::bsdfOpenPbrTransmissionIor(params);
#else
	let dispersedIor: f32 = (
		vkof::bsdfOpenPbrDispersionIor(params, λ.lambdas[0] * 1e-3f)
	);
#endif
	let bsdf: vkof::Bsdf_OpenPbr = (
		vkof::Bsdf_OpenPbr(
			/*frame=*/ frame,
			/*coatFrame=*/ coatFrame,
			/*λ=*/ λ,
			/*params=*/ params,
			/*isInsideMedium=*/ isInsideMedium,
			/*dispersedIor=*/ dispersedIor
		)
	);
	vertex.emitBsdf = (
		vkof::materialEmission(
			/*mtrl=*/ mtrl, /*uv=*/ geometry.uv, /*lod=*/ 0.0f
		)
		.toSpectrum(λ)
		* hit.spectralWeight
	);
	// -- pdf from previous vertex
	let useMeshNee: bool = vkof::global.meshNeeEnabled != 0u;
	if (
		useMeshNee
		&& previousScatter.neeAvailable
		&& mtrl.model.emitterLightCount > 0u
	) {
		// -- compute the pdf by converting from per-area to solid-angle
		let cosθ: f32 = (
			abs(dot(geometry.geometricNormal, -previousScatter.direction))
		);
		if (hit.areaJacobian > 0.0f && cosθ > 0.0f) {
			var pdfArea: f32 = 0.0f;
			let lightIndex: Optional<u32> = (
				vkof::emitterIndexFromHit(hit.surface.hit)
			);
			// calculate the pdf of sampling this hit from the previous vertex
			// either using the nee light sample pdf, or the regir reservoir
			// area pdf if the previous vertex was a regir sample
			if (
				previousScatter.regirSlot != vkof::skRegirSlotInvalid
				&& lightIndex.hasValue
			) {
				let cell: vkof::RegirCell = (
					vkof::global.regirCells[previousScatter.regirSlot]
				);
				pdfArea = (
					vkof::regirVisible(cell, geometry.position)
					? (
						vkof::regirPdfArea(
							/*slot=*/ previousScatter.regirSlot,
							/*lightIndex=*/ lightIndex.value,
							/*lightOri=*/ geometry.position,
							/*lightNor=*/ geometry.geometricNormal
						)
					)
					: 0.0f
				);
			} else {
				pdfArea = vkof::emitterPdf(hit.surface.hit, geometry);
			}
			vertex.pdfEmitFromHit = (
				pdfArea / (hit.areaJacobian * cosθ)
			);
		}
	}
	// -- select an emitter sample
	let request: ImplNeeRequest = (
		implNeeSelect(
			/*position=*/ hit.position,
			/*normal=*/ Optional<f32v3>(geometry.geometricNormal),
			/*bounceSampler=*/ bounceSampler,
			/*bounceCount=*/ u32(surfaceBounce)
		)
	);
	neeRecord.kind = ImplNeeVertexKind::Surface;
	neeRecord.request = request;
	neeRecord.ωi = hit.incomingDirection;
	neeRecord.bsdf = bsdf;
	// -- sample new direction
	let sample: vkof::BsdfSample = (
		bsdf.sample(
			/*ωi=*/ hit.incomingDirection,
			/*ξ=*/ f32v3(
				bounceSampler.nextF32v2(vkof::BounceDimension.Direction),
				bounceSampler.nextF32(vkof::BounceDimension.LobeSelect)
			)
		)
	);
	vertex.weight = sample.weight * hit.spectralWeight;
	vertex.isDeltaDirac = sample.isDeltaDirac;
	transport.newDirection(sample.ωo);
	previousScatter.position = hit.position;
	previousScatter.pdfBsdf = sample.pdf;
	previousScatter.direction = sample.ωo;
	previousScatter.neeAvailable = true;
	previousScatter.regirSlot = request.connection.regirSlot;
	++surfaceBounce;
}
