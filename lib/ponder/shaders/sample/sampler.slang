implementing sample;

import types;
import util;

// -----------------------------------------------------------------------------

namespace vkof {

// specifies which dimension of a sample to generate, after BounceStart
// the dimension is unnamed and unbounded
public enum SampleDimension : u32 {
	PixelJitter,
	PixelJitterX = PixelJitter,
	PixelJitterY,
	CameraShutter,
	Wavelength,
	CameraLens,
	CameraLensX = CameraLens,
	CameraLensY,
	// the first dimension of the bounce, each iteration is BounceStart+i
	BounceStart,
};

// the dimension within a specific bounce
public enum BounceDimension : u32 {
	LobeSelect = 0,
	Direction = 1, // 2 wide
	EmitterSelect = 3,
	EmitterElement = 4,
	EmitterPosition = 5, // 2 wide
	RussianRoulette = 7,
	MediumDistance = 8,
	RegirShade = 9,
	Count = 10,
};

public struct SamplerBounceView {
	Sampler sampler;
	u32 offset;

	public __init(Sampler sampler, u32 offset) {
		this.sampler = sampler;
		this.offset = offset;
	}

	public func nextF32(dimension: BounceDimension) -> f32 {
		return sampler.nextF32(offset + u32(dimension));
	}
	public func nextF32v2(dimension: BounceDimension) -> f32v2 {
		return sampler.nextF32v2(offset + u32(dimension));
	}
	public func stream(dimension: BounceDimension) -> SampleStream {
		return SampleStream(sampler, offset + u32(dimension));
	}
};

// stateless sample generator to allow consistent reconstruction of rays
// for debugging. The major downside to this is that each type of sample
// needs to be tagged so that the code doesn't have to adhere to a strict
// ordering.
public struct Sampler {
	u32v3 seed;

	public __init(u32v2 pixel, u32 sampleIndex) {
		this.seed = u32v3(pixel, sampleIndex);
	}

	public __init(u32v3 seed) {
		this.seed = seed;
	}

	public func getSeed() -> u32v3 {
		return seed;
	}

	func hashForDimension(dimension: u32) -> u32v4 {
		return implPcg4d(u32v4(seed, dimension));
	}

	public func bounceView(bounce: u32) -> SamplerBounceView {
		return SamplerBounceView(
			/*sampler=*/ this,
			/*offset=*/ implBounceDimensionOffset(bounce, BounceDimension(0))
		);
	}

	// sample a single dimension, the integer dimension is picked from either
	// a value of SampleDimension or for bounce dimension,
	// implBounceDimensionOffset
	public func nextF32(dimension: u32) -> f32 {
		vkof::debugPathDimension(dimension);
		return implUintToUniform(hashForDimension(dimension).x);
	}
	public func nextF32(dimension: SampleDimension) -> f32 {
		return nextF32(u32(dimension));
	}
	public func nextF32v2(dimension: u32) -> f32v2 {
		return f32v2(nextF32(dimension), nextF32(dimension + 1u));
	}
	public func nextF32v2(dimension: SampleDimension) -> f32v2 {
		return nextF32v2(u32(dimension));
	}
	// sampling with index won't be individually tracked by the debug path, but
	// the path should still be reproducible from the initial where index=0
	public func nextF32WithIndex(dimension: u32, index: u32) -> f32 {
		if (index == 0u) { return nextF32(dimension); }
		// -- rehash
		let hash: u32v4 = hashForDimension(dimension);
		return (
			implUintToUniform(implPcg4d(u32v4(hash.y, hash.z, hash.w, index)).x)
		);
	}
};

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// generates an offset into BounceDimension
func implBounceDimensionOffset(
	bounce: u32,
	offset: vkof::BounceDimension,
) -> u32 {
	return (
		u32(vkof::SampleDimension.BounceStart)
		+ bounce * u32(vkof::BounceDimension.Count)
		+ u32(offset)
	);
}
