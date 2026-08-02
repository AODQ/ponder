implementing sample;

import types;
import util;

// -----------------------------------------------------------------------------

namespace vkof {

// a random stream of samples, to stream unknown multiple numbers at a single
// dimension
public struct SampleStream {
	vkof::Sampler sampler;
	u32 dimension;
	u32v4 hash;

	public __init(vkof::Sampler sampler, u32 dimension) {
		this.sampler = sampler;
		this.dimension = dimension;
		this.hash = sampler.hashForDimension(dimension);
	}

	[mutating]
	public func nextF32() -> f32 {
		let hashNext: u32v4 = implPcg4d(this.hash);
		let value: f32 = implUintToUniform(hashNext.x);
		this.hash = hashNext;
		return value;
	}

	[mutating]
	public func nextF32v2() -> f32v2 {
		return f32v2(nextF32(), nextF32());
	}
};


} // namespace vkof
