implementing medium;

#include "include/inline-expression.slang"
#include "include/differentiable.slang"

import types;
import bsdf;

namespace vkof {

// phase function sample result
// contains the sample of a phase and the probability of having sampled it
public struct PhaseSample {
	public f32v3 ωo;
	public f32 f;
	public f32 pdf;
};

// phase function interface
public interface I_PhaseFunction {
	public func evaluate(ωi: f32v3, ωo: f32v3,) -> f32;
	public func sample(ωi: f32v3, ξ: f32v2,) -> vkof::PhaseSample;
};

public struct PhaseFunction_HenyeyGreenstein : vkof::I_PhaseFunction {
	f32 g;

	public __init(f32 g) {
		this.g = g;
	}

	public func evaluate(ωi: f32v3, ωo: f32v3,) -> f32 {
		return implPhaseHenyeyGreensteinEvaluate(g, ωi, ωo);
	}

	public func sample(ωi: f32v3, ξ: f32v2,) -> vkof::PhaseSample {
		let cosθ: f32 = implPhaseHenyeyGreensteinSampleCos(ξ.x, g);
		// -- sample azimuth uniformly
		let φ: f32 = 2.0f*π*ξ.y;
		// -- rotate into local frame
		let ωo: f32v3 = vkof::directionAboutAxis(ωi, cosθ, φ);
		// -- return sample
		var sample: vkof::PhaseSample;
		sample.ωo = ωo;
		sample.f = evaluate(ωi, ωo);
		sample.pdf = sample.f;
		return sample;
	}
}

// draine phase function, jendersie & d'eon 2023 eq 2
// reduces to henyey-greenstein at α = 0, rayleigh at g = 0 α = 1,
// cornette-shanks at α = 1
// ported from utilVdbPhaseDraine (shaders-old)
public struct PhaseFunction_Draine : vkof::I_PhaseFunction {
	f32 g;
	f32 α;

	public __init(f32 g, f32 α) {
		this.g = g;
		this.α = α;
	}

	public func evaluate(ωi: f32v3, ωo: f32v3,) -> f32 {
		return implPhaseDraineEvaluate(g, α, ωi, ωo);
	}

	public func sample(ωi: f32v3, ξ: f32v2,) -> vkof::PhaseSample {
		let cosθ: f32 = implPhaseDraineSampleCos(ξ.x, g, α);
		let φ: f32 = 2.0f*π*ξ.y;
		let ωo: f32v3 = vkof::directionAboutAxis(ωi, cosθ, φ);
		var sample: vkof::PhaseSample;
		sample.ωo = ωo;
		sample.f = evaluate(ωi, ωo);
		sample.pdf = sample.f;
		return sample;
	}
}

// water-droplet fog: mix of henyey-greenstein and draine lobes
// jendersie & d'eon 2023 eq 3, parameters from the droplet diameter (eq 4-7)
// ported from utilVdbPhaseFog (shaders-old)
public struct PhaseFunction_Fog : vkof::I_PhaseFunction {
	f32 gHenyeyGreenstein;
	f32 gDraine;
	f32 α;
	f32 draineWeight;

	public __init(f32 dropletDiameter) {
		implPhaseFogParameters(
			dropletDiameter,
			this.gHenyeyGreenstein,
			this.gDraine,
			this.α,
			this.draineWeight
		);
	}

	public func evaluate(ωi: f32v3, ωo: f32v3,) -> f32 {
		return implPhaseFogEvaluate(
			gHenyeyGreenstein, gDraine, α, draineWeight, ωi, ωo
		);
	}

	public func sample(ωi: f32v3, ξ: f32v2,) -> vkof::PhaseSample {
		// -- pick the lobe
		// ξ.x is reused for the lobe's own sample after rescaling
		let pickDraine: bool = ξ.x < draineWeight;
		let ξLobe: f32v2 = f32v2(
			pickDraine
				? ξ.x / draineWeight
				: (ξ.x - draineWeight) / (1.0f - draineWeight),
			ξ.y
		);
		let ωo: f32v3 = (
			pickDraine
				? vkof::PhaseFunction_Draine(gDraine, α).sample(ωi, ξLobe).ωo
				: (
					vkof::PhaseFunction_HenyeyGreenstein(gHenyeyGreenstein)
						.sample(ωi, ξLobe).ωo
				)
		);
		// -- the pdf is the mixture, not the picked lobe
		var sample: vkof::PhaseSample;
		sample.ωo = ωo;
		sample.f = evaluate(ωi, ωo);
		sample.pdf = sample.f;
		return sample;
	}
}

public func phaseHenyeyGreenstein(g: f32) -> vkof::PhaseFunction {
	var phase: vkof::PhaseFunction;
	phase.type = vkof::PhaseFunctionType.HenyeyGreenstein;
	phase.g = g;
	phase.α = 0.0f;
	phase.dropletDiameter = 0.0f;
	return phase;
}

public func phaseDraine(g: f32, α: f32) -> vkof::PhaseFunction {
	var phase: vkof::PhaseFunction;
	phase.type = vkof::PhaseFunctionType.Draine;
	phase.g = g;
	phase.α = α;
	phase.dropletDiameter = 0.0f;
	return phase;
}

public func phaseFog(dropletDiameter: f32) -> vkof::PhaseFunction {
	var phase: vkof::PhaseFunction;
	phase.type = vkof::PhaseFunctionType.Fog;
	phase.g = 0.0f;
	phase.α = 0.0f;
	phase.dropletDiameter = dropletDiameter;
	return phase;
}

public enum PhaseFunctionType {
	HenyeyGreenstein,
	Draine,
	Fog,
};

public struct PhaseFunction : I_PhaseFunction {
	public PhaseFunctionType type;
	public f32 g;
	public f32 α;
	public f32 dropletDiameter;

	public func evaluate(ωi: f32v3, ωo: f32v3,) -> f32 {
		switch (type) {
			case PhaseFunctionType.HenyeyGreenstein:
				return (
					vkof::PhaseFunction_HenyeyGreenstein(g).evaluate(ωi, ωo)
				);
			case PhaseFunctionType.Draine:
				return vkof::PhaseFunction_Draine(g, α).evaluate(ωi, ωo);
			case PhaseFunctionType.Fog:
				return (
					vkof::PhaseFunction_Fog(dropletDiameter).evaluate(ωi, ωo)
				);
			default:
				return 0.0f;
		}
	}

	public func sample(ωi: f32v3, ξ: f32v2,) -> vkof::PhaseSample {
		switch (type) {
			case PhaseFunctionType.HenyeyGreenstein:
				return (
					vkof::PhaseFunction_HenyeyGreenstein(g).sample(ωi, ξ)
				);
			case PhaseFunctionType.Draine:
				return vkof::PhaseFunction_Draine(g, α).sample(ωi, ξ);
			case PhaseFunctionType.Fog:
				return (
					vkof::PhaseFunction_Fog(dropletDiameter).sample(ωi, ξ)
				);
			default:
				var sample: vkof::PhaseSample;
				sample.ωo = f32v3(0.0f);
				sample.f = 0.0f;
				sample.pdf = 0.0f;
				return sample;
		}
	}
}

} // namespace vkof
