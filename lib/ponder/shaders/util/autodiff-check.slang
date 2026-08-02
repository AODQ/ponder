implementing util;

import types;

#include "include/differentiable.slang"

// included only in the autodiff check build, see entry/autodiff-check.slang

namespace vkof {

// differentiates every PonderDifferentiable function of the module
public func utilAutodiffCheck() -> f32 {
	let λ: Wavelengths = wavelengthSampleUniform(0.3f);
	let a: Spectrum = Spectrum::one() * 0.5f;
	let b: Spectrum = Spectrum::one() * 2.0f;
	let da: Spectrum.Differential = (
		reinterpret<Spectrum.Differential>(Spectrum::one())
	);
	var sum: f32 = 0.0f;

	// -- forward mode
	sum += spectrumAverage(fwd_diff(spectrumExp)(diffPair(a, da)).d);
	sum += fwd_diff(spectrumAverage)(diffPair(a, da)).d;
	sum += fwd_diff(spectrumAsXyz)(diffPair(a, da), λ).d.y;
	sum += spectrumAverage(
		fwd_diff(operator*)(diffPair(a, da), diffPair(b, da)).d
	);
	sum += spectrumAverage(
		fwd_diff(operator+)(diffPair(a, da), diffPair(b, da)).d
	);

	// -- reverse mode
	var aPair: DifferentialPair<Spectrum> = diffPair(a);
	bwd_diff(spectrumAsXyz)(aPair, λ, f32v3(0.0f, 1.0f, 0.0f));
	sum += spectrumAverage(aPair.d);
	bwd_diff(spectrumAverage)(aPair, 1.0f);
	sum += spectrumAverage(aPair.d);
	bwd_diff(spectrumExp)(
		aPair,
		reinterpret<Spectrum.Differential>(Spectrum::one())
	);
	sum += spectrumAverage(aPair.d);
	return sum;
}

} // namespace vkof
