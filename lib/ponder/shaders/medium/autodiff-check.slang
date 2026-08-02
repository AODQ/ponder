implementing medium;

import types;

#include "include/differentiable.slang"

// included only in the autodiff check build, see entry/autodiff-check.slang

namespace vkof {

// differentiates every PonderDifferentiable function of the module
public func mediumAutodiffCheck() -> f32 {
	let ωi: f32v3 = f32v3(0.0f, 0.0f, 1.0f);
	let ωo: f32v3 = normalize(f32v3(0.0f, 1.0f, 1.0f));
	let dωi: f32v3 = f32v3(1.0f, 0.0f, 0.0f);
	var sum: f32 = 0.0f;

	// -- forward mode
	sum += fwd_diff(implPhaseHenyeyGreensteinEvaluate)(
		0.5f, diffPair(ωi, dωi), diffPair(ωo, f32v3(0.0f))
	).d;
	sum += fwd_diff(implPhaseDraineEvaluate)(
		0.5f, 1.0f, diffPair(ωi, dωi), diffPair(ωo, f32v3(0.0f))
	).d;
	sum += fwd_diff(implPhaseFogEvaluate)(
		0.8f, 0.4f, 20.0f, 0.3f, diffPair(ωi, dωi), diffPair(ωo, f32v3(0.0f))
	).d;

	// -- reverse mode
	var ωiPair: DifferentialPair<f32v3> = diffPair(ωi);
	var ωoPair: DifferentialPair<f32v3> = diffPair(ωo);
	bwd_diff(implPhaseHenyeyGreensteinEvaluate)(0.5f, ωiPair, ωoPair, 1.0f);
	sum += ωiPair.d.x + ωoPair.d.y;
	bwd_diff(implPhaseDraineEvaluate)(0.5f, 1.0f, ωiPair, ωoPair, 1.0f);
	sum += ωiPair.d.x + ωoPair.d.y;
	bwd_diff(implPhaseFogEvaluate)(
		0.8f,
		0.4f,
		20.0f,
		0.3f,
		ωiPair,
		ωoPair,
		1.0f
	);
	sum += ωiPair.d.x + ωoPair.d.y;
	return sum;
}

} // namespace vkof
