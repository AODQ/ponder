implementing path;

import bsdf;
import types;
import util;

namespace vkof {

public struct PathVertexNeeDebug {
	f32v3 position;
	u32 state;
	u32 strategy;
	u32 regirSlot;
	u32 reservoir;
};

public func pathVertexNeeDebugNone() -> PathVertexNeeDebug {
	var debug: PathVertexNeeDebug;
	debug.position = f32v3(0.0f);
	debug.state = vkof::skDebugPathNeeNone;
	debug.strategy = vkof::skDebugPathNeeStrategyNone;
	debug.regirSlot = vkof::skRegirSlotInvalid;
	debug.reservoir = 0u;
	return debug;
}

// a sampled event on a path
public struct PathVertex {
	u32 bounce;
	vkof::TraceTransportHit hit;
	// scattering weight of the vertex, f·cos/pdf with the transport weight
	vkof::Spectrum weight;
	// if the path hit an emitter, the probability density of having hit it
	// using the nee technique instead
	f32 pdfEmitFromHit;
	// the probability density of sampling towards the vertex using nee
	f32 pdfBsdfFromHit;
	// true only if direction is in a dirac delta lobe
	bool isDeltaDirac;
	// emitted radiance the path ran into, before its mis weight
	vkof::Spectrum emitBsdf;

	// fraction of light that passes through the vertex,
	// e.g. a transparent emitter
	f32 transmittance;
	f32 emitTime;
	f32 pathLength;
};

// a vertex on a path, with its nee sample
public struct PathVertexNee {
	// throughput at the vertex, before its weight
	vkof::Spectrum β;
	// a direct nee sample from an emitter in the scene, before its mis weight
	vkof::Spectrum emitNee;
	// the probability density of sampling towards an emitter from nee technique
	f32 pdfEmitFromNee;
	// the probability density of sampling towards an emitter from bsdf technique
	f32 pdfBsdfFromNee;
	PathVertexNeeDebug debug;
};

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// mis weight of PathVertexNee.emitNee
func implPathVertexWeightNee(nee: vkof::PathVertexNee) -> f32 {
	return vkof::balanceHeuristic(nee.pdfEmitFromNee, nee.pdfBsdfFromNee);
}

// mis weight of PathVertex.emitBsdf; 1 when nee could not have sampled the hit
func implPathVertexWeightHit(vertex: vkof::PathVertex) -> f32 {
	if (vertex.pdfEmitFromHit <= 0.0f) { return 1.0f; }
	return (
		vkof::balanceHeuristic(
			vertex.pdfBsdfFromHit, vertex.pdfEmitFromHit
		)
	);
}

// the last vertex a direction was sampled at
struct ImplPathScatterState {
	f32v3 position;
	// the probability density of sampling towards the vertex using bsdf
	f32 pdfBsdf;
	// false when nee can't sample the next hit, such as cameras or delta lobes
	bool neeAvailable;
	// direction
	f32v3 direction;
	// regir cell slot queried for this vertex, if failed then skRegirSlotInvalid
	u32 regirSlot;
};
