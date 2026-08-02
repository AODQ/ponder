implementing path;

import types;
import util;
import model;
import material;

namespace vkof {

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

struct ImplTraceStepHit
{
	__init(
		f32 distance,
		u32 instanceId,
		u32 primitiveId,
		f32v2 barycentrics,
		bool frontFace,
	) {
		this.distance = distance;
		this.instanceId = instanceId;
		this.primitiveId = primitiveId;
		this.barycentrics = barycentrics;
		this.frontFace = frontFace;
	}

	func hit() -> bool{
		return instanceId != 0xFFFFFFFFu;
	}

	f32 distance;
	u32 instanceId;
	u32 primitiveId;
	f32v2 barycentrics;
	bool frontFace;
};

// rotates the ray's alpha sample per triangle, so stacked blend layers
// pass or block independently instead of all on the same draw
func implAlphaUTriangle(u: f32, instance: u32, element: u32) -> f32 {
	var h: u32 = instance * 0x9E3779B9u ^ element;
	h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
	h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
	h ^= h >> 16u;
	return frac(u + f32(h >> 8u) * 0x1p-24f);
}

// trace a single step of a path returning closest hit scene data
func implTraceStep(
	origin: f32v3,
	dir: f32v3,
	distMin: f32,
	distMax: f32,
	// selects pass-through for alpha blend surfaces
	alphaU: f32 = 0.5f,
)
	-> ImplTraceStepHit
{
	// -- prepare ray query
	RayQuery<RAY_FLAG_NONE> query;
	let flags: u32 = 0;
	RayDesc queryDesc;
	queryDesc.Origin = origin;
	queryDesc.TMin = distMin;
	queryDesc.TMax = distMax;
	queryDesc.Direction = dir;
	query.TraceRayInline(
		/*accelerationStructure=*/ vkofTlas,
		/*rayFlags=*/ flags,
		/*instanceInclusionMask=*/ 0xFF,
		/*ray=*/ queryDesc
	);

	// -- process candidate triangles
	while (query.Proceed()) {
		switch (query.CandidateType()) {
			case CANDIDATE_NON_OPAQUE_TRIANGLE: {
				var candidate: vkof::SurfaceHit;
				candidate.kind = vkof::GeometryKind.Triangle;
				candidate.instance = query.CandidateInstanceID();
				candidate.element = query.CandidatePrimitiveIndex();
				candidate.parameters = query.CandidateTriangleBarycentrics();
				let alphaUTriangle: f32 = (
					implAlphaUTriangle(alphaU, candidate.instance, candidate.element)
				);
				if (vkof::materialAlphaCoverage(candidate, alphaUTriangle)) {
					query.CommitNonOpaqueTriangleHit();
				}
			} break;
			case CANDIDATE_PROCEDURAL_PRIMITIVE:
				// unsupported for now
			break;
		}
	}

	// -- return closest hit
	if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
		return ImplTraceStepHit(
			/*distance=*/ query.CommittedRayT(),
			/*instanceId=*/ query.CommittedInstanceID(),
			/*primitiveId=*/ query.CommittedPrimitiveIndex(),
			/*barycentrics=*/ query.CommittedTriangleBarycentrics(),
			/*frontFace=*/ query.CommittedTriangleFrontFace()
		);
	}
	return ImplTraceStepHit(
		/*distance=*/ distMax,
		/*instanceId=*/ 0xFFFFFFFFu,
		/*primitiveId=*/ 0xFFFFFFFFu,
		/*barycentrics=*/ f32v2(0.0f, 0.0f),
		/*frontFace=*/ false
	);
}
