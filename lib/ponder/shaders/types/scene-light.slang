implementing types;

namespace vkof {

public enum EmitterKind : u32 {
	// a mesh in the scene with emission properties
	SceneEmitter = 0u,
	// the environment at infinity, with no geometry
	Environment = 1u,
	// a directional light source, with no geometry
	Directional = 2u,
};

// this is a slice of an existing mesh in the scene, which has emitter
// properties such that it can be sampled as a light source
public struct SceneEmitterLight {
	public EmitterKind kind;
	public u32 modelDrawIndex;
	// flat primitive index of the emitter's first triangle
	public u32 primitiveBaseIndex;
	// number of triangles in the emitter's mesh
	public u32 primitiveCount;
	// world area * luma(emissionColor) * emissionLuminance
	public f32 power;
	// cumulative distribution function ; cumulative power / totalPower
	public f32 cdf;
	// precomputed emissionColor * emissionLuminance
	public f32v3 radiance;
};

} // vkof
