implementing medium;

// sources:
// pbrt-v4 ch 11 and 14
// novak et al 2018 "monte carlo methods for volumetric light transport"
// font et al 2017 "production volume rendering"
// chiang et al 2016 "practical and controllable subsurface scattering"

#include "include/inline-expression.slang"

import types;
import util;

namespace vkof {

// sample of a point in a participating medium
public struct MediumPoint {
	public vkof::Spectrum σa;
	public vkof::Spectrum σs;
	// emissive component, zero for non-emissive media
	public vkof::Spectrum emission;
	public vkof::PhaseFunction phase;
};

// sample of a segment in a participating medium
public struct MediumSegmentSample {
	// if false, the sample is at the end of the segment
	public bool scattered;
	public f32 distance;
	// multiplies β
	public vkof::Spectrum weight;
	// if scattered, the medium at the scatter point
	public MediumPoint point;
}

public interface I_Medium {
	// evaluate the medium at a point
	public func at(x: f32v3, λ: vkof::Wavelengths) -> vkof::MediumPoint;
	// sample a segment in the medium
	// ξ.x samples distance, ξ.y the sampled channel
	public func sampleSegment(
		x0: f32v3,
		x1: f32v3,
		ξ: f32v2,
		λ: vkof::Wavelengths,
	) -> vkof::MediumSegmentSample;

	// majorant of the medium's extinction coefficient,
	// enables delta tracking and null-collision sampling
	public func majorant(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum;

	// returns the transmittance between two points in the medium
	public func transmittance(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum;
};

// basic homogeneous medium with constant properties
// ported from openPbrSubsurfaceWalkStep (shaders-old), without guiding
public struct Medium_Homogeneous : vkof::I_Medium {
	vkof::Spectrum σa;
	vkof::Spectrum σs;
	vkof::PhaseFunction phase;

	public __init(
		vkof::Spectrum σa,
		vkof::Spectrum σs,
		vkof::PhaseFunction phase,
	) {
		this.σa = σa;
		this.σs = σs;
		this.phase = phase;
	}

	public func at(x: f32v3, λ: vkof::Wavelengths) -> vkof::MediumPoint {
		var point: vkof::MediumPoint;
		point.σa = σa;
		point.σs = σs;
		point.emission = vkof::Spectrum::zero();
		point.phase = phase;
		return point;
	}

	public func sampleSegment(
		x0: f32v3,
		x1: f32v3,
		ξ: f32v2,
		λ: vkof::Wavelengths,
	) -> vkof::MediumSegmentSample {
		let segment: f32v3 = x1 - x0;
		let len: f32 = length(segment);
		let σt: vkof::Spectrum = σa + σs;

		// -- pick the sampling channel
		// every channel is equally likely, so the mis denominator is the average
		let σtChannel: f32 = σt.channel(implMediumChannelPick(ξ.y));

		// -- sample the free-flight distance
		let t: f32 = (
			σtChannel > 0.0f ? -log(1.0f - ξ.x) / σtChannel : f32.maxValue
		);

		var sample: vkof::MediumSegmentSample;
		// -- passes the segment
		// transmittance over the average probability of passing
		if (t >= len) {
			let transmittance: vkof::Spectrum = (σt * -len).exp();
			sample.scattered = false;
			sample.distance = len;
			sample.weight = implMediumWeightPass(transmittance);
			sample.point = at(x1, λ);
			return sample;
		}

		// -- scatters at t
		// σs T(t) over the average free-flight density at t
		let transmittance: vkof::Spectrum = (σt * -t).exp();
		sample.scattered = true;
		sample.distance = t;
		sample.weight = implMediumWeightScatter(σs, σt, transmittance);
		sample.point = at(x0 + segment * (t / len), λ);
		return sample;
	}

	public func majorant(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		let σt: vkof::Spectrum = σa + σs;
		return σt;
	}

	public func transmittance(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		let segment: f32v3 = x1 - x0;
		let len: f32 = length(segment);
		if (len <= 0.0f) { return vkof::Spectrum::one(); }
		let σt: vkof::Spectrum = majorant(x0, x1, λ);
		return (σt * -len).exp();
	}
}

// height-fog
public struct Medium_HeightFog : vkof::I_Medium {
	// absorption spectral base coefficient
	vkof::Spectrum σa0;
	// scattering spectral base coefficient
	vkof::Spectrum σs0;
	vkof::PhaseFunction phase;
	f32 origin;
	f32 falloff;

	// for height fog,
	// p(y) = exp(-(y - h0) / hfalloff) [implDensity]
	// σa(x) = σa0 · p(y)
	// σs(x) = σs0 · p(y)
	// σt(x) = (σa0 + σs0) · p(y)
	// along a ray x0 .. x1,
	// the height is y(t) = y0 + ωy · t
	// a = ωy / hfalloff

	public __init(
		vkof::Spectrum σa0,
		vkof::Spectrum σs0,
		vkof::PhaseFunction phase,
		f32 origin,
		f32 falloff,
	) {
		this.σa0 = σa0;
		this.σs0 = σs0;
		this.phase = phase;
		this.origin = origin;
		this.falloff = falloff;
	}

	public func at(
		x: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::MediumPoint {
		let y: f32 = x.y - origin;
		let σa: vkof::Spectrum = σa0 * implDensity(y);
		let σs: vkof::Spectrum = σs0 * implDensity(y);
		var point: vkof::MediumPoint;
		point.σa = σa;
		point.σs = σs;
		point.emission = vkof::Spectrum::zero();
		point.phase = vkof::phaseHenyeyGreenstein(phase.g);
		return point;
	}

	public func sampleSegment(
		x0: f32v3,
		x1: f32v3,
		ξ: f32v2,
		λ: vkof::Wavelengths,
	) -> vkof::MediumSegmentSample {
		let segment: f32v3 = x1 - x0;
		let len: f32 = length(segment);
		let σt0: vkof::Spectrum = σa0 + σs0;

		if (len <= 0.0f) {
			var sample: vkof::MediumSegmentSample;
			sample.scattered = false;
			sample.distance = 0.0f;
			sample.weight = vkof::Spectrum::one();
			sample.point = at(x1, λ);
			return sample;
		}

		// -- pick the sampling channel
		let σt0Channel: f32 = σt0.channel(implMediumChannelPick(ξ.y));

		// -- sample the free-flight distance
		let u: f32 = -log(1.0f - ξ.x) / σt0Channel;
		let a: f32 = (segment.y / len) / falloff;
		let p0: f32 = implDensity(x0.y - origin);
		// -- solve for t in F(t) = u
		// τ(t) = σt0 · F(t)
		// Tr(t) = exp(-σt0 · F(t))
		let t: f32 = implHeightFogDensityIntegralInverse(p0, a, u);

		// -- sample is beyond the segment
		// return the transmittance over the average probability of passing
		if (t >= len) {
			let Tr: vkof::Spectrum = (
				(σt0 * -implHeightFogDensityIntegral(p0, a, len)).exp()
			);
			var sample: vkof::MediumSegmentSample;
			sample.scattered = false;
			sample.distance = len;
			sample.weight = implMediumWeightPass(Tr);
			sample.point = at(x1, λ);
			return sample;
		}

		// -- sample is within the segment
		// return σs T(t) over the average free-flight density at t
		let Tr: vkof::Spectrum = (
			(σt0 * -implHeightFogDensityIntegral(p0, a, t)).exp()
		);
		var sample: vkof::MediumSegmentSample;
		sample.scattered = true;
		sample.distance = t;
		sample.weight = implMediumWeightScatter(σs0, σt0, Tr);
		sample.point = at(x0 + segment * (t / len), λ);
		return sample;
	}

	public func majorant(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		// density highest at lowest point of segmant;
		// majorant = σt0 · p(min(y0, y1))
		let y: f32 = min(x0.y, x1.y) - origin;
		let p: f32 = implDensity(y);
		let σt0: vkof::Spectrum = σa0 + σs0;
		return σt0 * p;
	}

	func implDensity(y: f32) -> f32 {
		let p0: f32 = exp(min(-y / falloff, 80.0f));
		return p0;
	}

	public func transmittance(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		// -- transmittance is exp(-σt0 · F(t))
		let segment: f32v3 = x1 - x0;
		let len: f32 = length(segment);
		let a: f32 = (segment.y / len) / falloff;
		let p0: f32 = implDensity(x0.y - origin);
		let σt0: vkof::Spectrum = σa0 + σs0;
		return (σt0 * -implHeightFogDensityIntegral(p0, a, len)).exp();
	}
};

// -----------------------------------------------------------------------------

public enum MediumType {
	None,
	Homogeneous,
	HeightFog,
}

public struct Medium : I_Medium {
	public MediumType type;
	// -- base coefficients
	public vkof::Spectrum σa;
	public vkof::Spectrum σs;
	public PhaseFunction phase;
	// -- height fog parameters
	public f32 origin;
	public f32 falloff;

	public func at(x: f32v3, λ: vkof::Wavelengths) -> vkof::MediumPoint {
		switch (type) {
			default:
			case MediumType.None:
				var point: vkof::MediumPoint;
				point.σa = vkof::Spectrum::zero();
				point.σs = vkof::Spectrum::zero();
				point.emission = vkof::Spectrum::zero();
				point.phase = vkof::phaseHenyeyGreenstein(0.0f);
				return point;
			case MediumType.Homogeneous: {
				return Medium_Homogeneous(σa, σs, phase).at(x, λ);
			}
			case MediumType.HeightFog: {
				return Medium_HeightFog(σa, σs, phase, origin, falloff).at(x, λ);
			}
		}
	}

	public func sampleSegment(
		x0: f32v3,
		x1: f32v3,
		ξ: f32v2,
		λ: vkof::Wavelengths,
	) -> vkof::MediumSegmentSample {
		switch (type) {
			default:
			case MediumType.None: {
				var sample: vkof::MediumSegmentSample;
				sample.scattered = false;
				sample.distance = length(x1 - x0);
				sample.weight = vkof::Spectrum::one();
				return sample;
			}
			case MediumType.Homogeneous: {
				return (
					Medium_Homogeneous(σa, σs, phase)
					.sampleSegment(x0, x1, ξ, λ)
				);
			}
			case MediumType.HeightFog: {
				return (
					Medium_HeightFog(σa, σs, phase, origin, falloff)
					.sampleSegment(x0, x1, ξ, λ)
				);
			}
		}
	}

	public func majorant(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		switch (type) {
			default:
			case MediumType.None: return vkof::Spectrum::zero();
			case MediumType.Homogeneous: {
				return Medium_Homogeneous(σa, σs, phase).majorant(x0, x1, λ);
			}
			case MediumType.HeightFog: {
				return (
					Medium_HeightFog(σa, σs, phase, origin, falloff)
					.majorant(x0, x1, λ)
				);
			}
		}
	}

	public func transmittance(
		x0: f32v3,
		x1: f32v3,
		λ: vkof::Wavelengths,
	) -> vkof::Spectrum {
		switch (type) {
			default:
			case MediumType.None: return vkof::Spectrum::one();
			case MediumType.Homogeneous: {
				return (
					Medium_Homogeneous(σa, σs, phase).transmittance(x0, x1, λ)
				);
			}
			case MediumType.HeightFog: {
				return (
					Medium_HeightFog(σa, σs, phase, origin, falloff)
					.transmittance(x0, x1, λ)
				);
			}
		}
	}
};

// returns the ambient medium for the scene
public func mediumAmbient(λ: Wavelengths) -> vkof::Medium {
	let σa: vkof::Spectrum = (
		Spectrum.fromRgbUnbounded(vkof::global.ambientMediumSigmaAbsorption, λ)
	);
	let σs: vkof::Spectrum = (
		Spectrum.fromRgbUnbounded(vkof::global.ambientMediumSigmaScattering, λ)
	);

	let phaseFunction: vkof::PhaseFunction = (
		vkof::PhaseFunction(
			/*type=*/ vkof::PhaseFunctionType(vkof::global.ambientMediumPhase),
			/*g=*/ vkof::global.ambientMediumAnisotropy,
			/*α=*/ vkof::global.ambientMediumDraineAlpha,
			/*dropletDiameter=*/ vkof::global.ambientMediumDropletDiameter,
		)
	);
	return Medium(
		/*type=*/ vkof::MediumType(vkof::global.ambientMediumType),
		/*σa=*/ σa,
		/*σs=*/ σs,
		/*phase=*/ phaseFunction,
		/*origin=*/ vkof::global.ambientMediumHeightOrigin,
		/*falloff=*/ vkof::global.ambientMediumHeightFalloff
	);
}

// -----------------------------------------------------------------------------
// -- material media
// -----------------------------------------------------------------------------

// a medium the path can be inside
public struct MediumHandle {
	// 0 is the ambient medium, n + 1 is material medium record n
	u32 value;

	__init(u32 value) {
		this.value = value;
	}

	public static func ambient() -> MediumHandle {
		return MediumHandle(0u);
	}

	public func isAmbient() -> bool {
		return value == 0u;
	}

	public func recordIndex() -> u32 {
		return value - 1u;
	}

	public func equals(other: MediumHandle) -> bool {
		return value == other.value;
	}
};

// the medium a material's mesh bounds, or none for a regular surface
public func materialMediumHandle(
	instance: u32,
	materialIndex: u32,
) -> Optional<MediumHandle> {
	// -- no material is a medium
	if (vkof::global.materialMediumTable == nullptr) { return none; }
	if (instance >= vkof::global.materialMediumModelCount) { return none; }
	// -- the model's records
	let entry: u32v2 = vkof::global.materialMediumTable[instance];
	if (materialIndex >= entry.y) { return none; }
	// -- a regular surface
	let index: u32 = entry.x + materialIndex;
	let record: MaterialMediumRecord = vkof::global.materialMediumRecords[index];
	if (record.type == u32(MediumType.None)) { return none; }
	return MediumHandle(index + 1u);
}

// load the medium from the handle
public func mediumLoad(handle: MediumHandle, λ: Wavelengths) -> Medium {
	if (handle.isAmbient()) { return mediumAmbient(λ); }
	let record: MaterialMediumRecord = (
		vkof::global.materialMediumRecords[handle.recordIndex()]
	);
	return Medium(
		/*type=*/ MediumType(record.type),
		/*σa=*/ Spectrum.fromRgbUnbounded(record.σa, λ),
		/*σs=*/ Spectrum.fromRgbUnbounded(record.σs, λ),
		/*phase=*/ PhaseFunction(
			/*type=*/ PhaseFunctionType(record.phase),
			/*g=*/ record.anisotropy,
			/*α=*/ record.draineAlpha,
			/*dropletDiameter=*/ record.dropletDiameter,
		),
		/*origin=*/ record.heightOrigin,
		/*falloff=*/ record.heightFalloff
	);
}

// the priority of the medium at the handle
public func mediumPriority(handle: MediumHandle) -> u32 {
	if (handle.isAmbient()) { return 0u; }
	return vkof::global.materialMediumRecords[handle.recordIndex()].priority;
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------
