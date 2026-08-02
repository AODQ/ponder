implementing path;

import material;
import types;
import util;

namespace vkof {

// produces output from the sampled vertices of a path
public interface I_PathConsumer {
	[mutating] public func vertex(vertex: PathVertex, λ: vkof::Wavelengths);
	// the nee sample of the last vertex, once its visibility is traced
	[mutating] public func vertexNee(nee: PathVertexNee, λ: vkof::Wavelengths);
	public func throughput() -> vkof::Spectrum;
	// russian roulette after the last vertex, kept with probability rrChance
	[mutating] public func russianRoulette(rrChance: f32, survived: bool);
};

// evaluates the radiance along the path
public struct PathConsumer_Evaluate : I_PathConsumer {
	vkof::Spectrum β;
	public vkof::Spectrum Li;

	public __init(vkof::Wavelengths λ) {
		this.β = vkof::Spectrum.fromRgb(f32v3(1.0f), λ);
		this.Li = vkof::Spectrum.fromRgb(f32v3(0.0f), λ);
	}

	[mutating] public func vertex(vertex: PathVertex, λ: vkof::Wavelengths) {
		// -- add emitted radiance and update throughput
		// Li needs to compute before β to avoid multiplying the surface's
		// light with its own albedo
		Li += β * vertex.emitBsdf * implPathVertexWeightHit(vertex);
		β *= vertex.weight * vertex.transmittance;
	}

	[mutating] public func vertexNee(nee: PathVertexNee, λ: vkof::Wavelengths) {
		// -- add direct nee sample to Li
		Li += nee.β * nee.emitNee * implPathVertexWeightNee(nee);
	}

	public func throughput() -> vkof::Spectrum {
		return β;
	}

	[mutating] public func russianRoulette(rrChance: f32, survived: bool) {
		if (survived) { β *= 1.0f / rrChance; }
	}
};

// records the path to the debug path, then forwards to another consumer
public struct PathConsumer_DebugRecord<InnerConsumer : I_PathConsumer>
	: I_PathConsumer
{
	public InnerConsumer innerConsumer;

	public __init(InnerConsumer innerConsumer) {
		this.innerConsumer = innerConsumer;
	}

	[mutating] public func vertex(vertex: PathVertex, λ: vkof::Wavelengths) {
		implDebugPathRecord(vertex, innerConsumer.throughput(), λ);
		innerConsumer.vertex(vertex, λ);
	}

	[mutating] public func vertexNee(nee: PathVertexNee, λ: vkof::Wavelengths) {
		let direct: vkof::Spectrum = (
			nee.β * nee.emitNee * implPathVertexWeightNee(nee)
		);
		vkof::debugPathVertexNee(
			/*directRgb=*/ direct.asRgb(λ),
			/*pdfEmitFromNee=*/ nee.pdfEmitFromNee,
			/*pdfBsdfFromNee=*/ nee.pdfBsdfFromNee,
			/*position=*/ nee.debug.position,
			/*state=*/ nee.debug.state,
			/*regirSlot=*/ nee.debug.regirSlot,
			/*strategy=*/ nee.debug.strategy,
			/*reservoir=*/ nee.debug.reservoir,
		);
		innerConsumer.vertexNee(nee, λ);
	}

	public func throughput() -> vkof::Spectrum {
		return innerConsumer.throughput();
	}

	[mutating] public func russianRoulette(rrChance: f32, survived: bool) {
		vkof::debugPathVertexRrChance(rrChance);
		innerConsumer.russianRoulette(rrChance, survived);
	}
};

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implDebugPathRecord(
	vertex: vkof::PathVertex,
	β: vkof::Spectrum,
	λ: vkof::Wavelengths,
) {
	let hit: vkof::TraceTransportHit = vertex.hit;
	let isEmission: bool = hit.event == vkof::TransportEvent::Emission;
	var record: vkof::DebugPathVertex;
	record.event = u32(hit.event);
	record.bounce = vertex.bounce;
	record.transmittance = isEmission ? hit.emission.transmittance : 1.0f;
	record.frequencyShift = hit.frequencyShift;
	record.position = hit.position;
	record.emitTime = vertex.emitTime;
	record.direction = hit.incomingDirection;
	record.pathLength = vertex.pathLength;
	record.throughputRgb = β.asRgb(λ);
#if defined(PONDER_SPECTRUM_RGB)
	record.throughputAmounts = f32v4(β.asRgb(λ), 0.0f);
#else
	record.throughputAmounts = (
		f32v4(β.amounts[0], β.amounts[1], β.amounts[2], β.amounts[3])
	);
#endif
	record.rrChance = 1.0f;
	record.emissionRgb = vertex.emitBsdf.asRgb(λ);
	let isSurface: bool = hit.event == vkof::TransportEvent::Surface;
	record.surfaceInstance = isSurface ? hit.surface.hit.instance : 0xFFFFFFFFu;
	record.surfaceMaterial = (
		isSurface ? vkof::materialIndexFromHit(hit.surface.hit) : 0xFFFFFFFFu
	);
	record.wavelengths = (
		f32v4(λ.lambdas[0], λ.lambdas[1], λ.lambdas[2], λ.lambdas[3])
	);
	// -- debug path point
	record.directRgb = f32v3(0.0f);
	record.contributionRgb = (
		(β * vertex.emitBsdf * implPathVertexWeightHit(vertex)).asRgb(λ)
	);
	record.pdfBsdfFromHit = vertex.pdfBsdfFromHit;
	record.pdfEmitFromHit = vertex.pdfEmitFromHit;
	record.pdfEmitFromNee = 0.0f;
	record.pdfBsdfFromNee = 0.0f;
	record.isDeltaDirac = vertex.isDeltaDirac ? 1u : 0u;
	let neeDebug: vkof::PathVertexNeeDebug = vkof::pathVertexNeeDebugNone();
	record.neePosition = neeDebug.position;
	record.neeState = neeDebug.state;
	record.regirSlot = neeDebug.regirSlot;
	record.neeStrategy = neeDebug.strategy;
	record.neeReservoir = neeDebug.reservoir;
	vkof::debugPathVertex(record);
}
