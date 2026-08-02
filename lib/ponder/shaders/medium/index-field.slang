implementing medium;

import types;
import util;

namespace vkof {

// a sample of a refractive-index field at a given point
public struct IndexFieldSample
{
	public f32 refractiveIndex;
	public f32v3 gradient;
};

// a 3D field of refractive indices, used to bend light rays
public interface I_IndexField {
	public func evaluate(x: f32v3) -> IndexFieldSample;
	public func bounds() -> f32v3x2;
};

// index field of haze over a groud plane
public struct IndexField_GroundHeating : I_IndexField
{
	public f32 groundHeight;
	public f32 hazeHeight;
	public f32 refractiveIndexGradient;
	// noise parameters
	public f32 noiseAmplitude;
	public f32 noiseScale;

	public func evaluate(x: f32v3) -> vkof::IndexFieldSample {
		let y0: f32 = groundHeight;
		let y1: f32 = groundHeight + hazeHeight;
		if (x.y < y0 || x.y > y1) {
			return (
				vkof::IndexFieldSample(
					/*refractiveIndex=*/ 1.0f,
					/*gradient=*/ f32v3(0.0f)
				)
			);
		}
		// add some noise
		let height: f32 = x.y - y0;
		let scale: f32 = max(noiseScale, 1e-3f);
		let noise: vkof::NoiseSample = (
			vkof::noiseValueGradient(x / scale)
		);
		let fade: f32 = 1.0f - height/hazeHeight;
		let fadeGradient: f32v3 = f32v3(0.0f, -1.0f/hazeHeight, 0.0f);
		let noiseValue: f32 = noise.value * noiseAmplitude * fade;
		let noiseGradient: f32v3 = (
			noiseAmplitude * (noise.gradient/scale*fade + noise.value*fadeGradient)
		);
		return (
			vkof::IndexFieldSample(
				/*refractiveIndex=*/ (
					1.0f + refractiveIndexGradient*height + noiseValue
				),
				/*gradient=*/ (
					f32v3(0.0f, refractiveIndexGradient, 0.0f) + noiseGradient
				)
			)
		);
	}

	public func bounds() -> f32v3x2 {
		return f32v3x2(
			f32v3(-1.0e6f, groundHeight, -1.0e6f),
			f32v3(1.0e6f, groundHeight + hazeHeight, 1.0e6f)
		);
	}
}

} // namespace vkof
