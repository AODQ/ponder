implementing bsdf;

import types;

#include "include/differentiable.slang"

namespace vkof {

// cosine-weighted hemisphere sampling
PonderDifferentiable
public func cosineHemisphereSampleWo(nor: f32v3, ξ: f32v2) -> f32v3 {
	return (
		vkof::reorientHemisphere(
			normalize(vkof::toCartesian(sqrt(ξ.x), 2.0f*π*ξ.y)),
			nor
		)
	);
}

// solid-angle pdf of cosineHemisphereSampleWo
public func cosineHemispherePdf(nor: f32v3, ωo: f32v3) -> f32 {
	return max(dot(nor, ωo), 0.0f) / f32(π);
}

}
