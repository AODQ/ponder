implementing bsdf;

import types;

#include "include/differentiable.slang"

// included only in the autodiff check build, see entry/autodiff-check.slang

namespace vkof {

// differentiates every PonderDifferentiable function of the module
public func bsdfAutodiffCheck() -> f32 {
	let n: f32v3 = normalize(f32v3(0.2f, 0.3f, 1.0f));
	let dn: f32v3 = f32v3(1.0f, 0.0f, 0.0f);
	let ξ: f32v2 = f32v2(0.3f, 0.7f);
	let dξ: f32v2 = f32v2(1.0f, 0.0f);
	var sum: f32 = 0.0f;

	// -- forward mode
	sum += (
		fwd_diff(cosineHemisphereSampleWo)(diffPair(n, dn), diffPair(ξ, dξ)).d.x
	);
	sum += fwd_diff(directionAboutAxis)(
		diffPair(n, dn), diffPair(0.5f, 1.0f), diffPair(1.0f, 0.0f)
	).d.x;
	sum += fwd_diff(shadingFrameFromNormal)(diffPair(n, dn)).d.tanX.x;

	// -- reverse mode
	var nPair: DifferentialPair<f32v3> = diffPair(n);
	var ξPair: DifferentialPair<f32v2> = diffPair(ξ);
	bwd_diff(cosineHemisphereSampleWo)(nPair, ξPair, f32v3(1.0f));
	sum += nPair.d.x + ξPair.d.y;
	var cosθPair: DifferentialPair<f32> = diffPair(0.5f);
	var φPair: DifferentialPair<f32> = diffPair(1.0f);
	bwd_diff(directionAboutAxis)(nPair, cosθPair, φPair, f32v3(1.0f));
	sum += cosθPair.d + φPair.d;
	return sum;
}

} // namespace vkof
