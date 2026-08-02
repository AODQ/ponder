implementing bsdf;

import types;
import util;

namespace vkof {

// what a sampled direction does at the surface
public enum BsdfSampleKind {
	Reflection,
	// refracts into or out of the medium behind the surface
	Transmission,
	// crosses a thin wall, which has no medium behind it
	ThinTransmission,
	// enters the subsurface medium; weight excludes f · |cosθ| / pdf
	SubsurfaceEntry,
};

public struct BsdfSample {
	public f32v3 ωo;
	// f · |cosθ| / pdf
	public vkof::Spectrum weight;
	// probability density of the sample in solid angle
	public f32 pdf;
	// true if the sample is a delta dirac
	public bool isDeltaDirac;
	public BsdfSampleKind kind;
};

// bidirectional scattering distribution function interface
// ωi travels from camera towards light, pointing inwards to surface
// TODO although some internal implementations flip ωi (and are inconsistent
// between how α=roughness) since I was careless in consistency between
// implementations from various sources. NEed to standardize it still
public interface I_Bsdf {
	// f · |cosθ|
	public func evaluate(ωi: f32v3, ωo: f32v3,) -> vkof::Spectrum;
	// probability density of sampling ωo given ωi
	public func pdf(ωi: f32v3, ωo: f32v3) -> f32;
	// sample a direction ωo given an incoming direction ωi with a uniform
	// random sample u ∈ [0, 1)³
	// ξ.xy should come from BounceDimension.Direction, ξ.z from LobeSelect
	public func sample(ωi: f32v3, ξ: f32v3) -> BsdfSample;
};

// -----------------------------------------------------------------------------
// -- lambertian diffuse BSDF
// -----------------------------------------------------------------------------
// simple diffuse bsdf for testing/debugging

public struct Bsdf_Lambertian : I_Bsdf {
	public f32v3 nor;
	public var albedo: vkof::Spectrum;

	public __init(f32v3 nor, vkof::Spectrum albedo) {
		this.nor = nor;
		this.albedo = albedo;
	}

	public func evaluate(ωi: f32v3, ωo: f32v3) -> vkof::Spectrum {
		return albedo * max(0.0f, dot(nor, ωo)) * (1.0f / π);
	}

	public func pdf(ωi: f32v3, ωo: f32v3) -> f32 {
		return vkof::cosineHemispherePdf(nor, ωo);
	}

	public func sample(ωi: f32v3, ξ: f32v3) -> BsdfSample {
		var sample: BsdfSample;
		sample.ωo = vkof::cosineHemisphereSampleWo(nor, ξ.xy);
		sample.pdf = pdf(ωi, sample.ωo);
		sample.weight = albedo;
		sample.isDeltaDirac = false;
		sample.kind = BsdfSampleKind::Reflection;
		return sample;
	}
};

} // namespace vkof
