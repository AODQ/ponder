implementing util;

namespace vkof {

// narkowicz 2015 fit of the aces filmic curve; linear in, display [0, 1] out
public func aces(x: f32v3) -> f32v3 {
	let a: f32 = 2.51f;
	let b: f32 = 0.03f;
	let c: f32 = 2.43f;
	let d: f32 = 0.59f;
	let e: f32 = 0.14f;
	return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}


// TODO move these to math.slang or something

// 256 ulp as a relative factor, wächter & binder, ray tracing gems ch. 6
static const f32 skRayEpsilonRelative = 3e-5f;
static const f32 skRayEpsilonMinimum = 1e-5f;

public func rayEpsilon(p: f32v3) -> f32 {
	return (
		max(
			vkof::skRayEpsilonMinimum,
			vkof::skRayEpsilonRelative * max(max(abs(p.x), abs(p.y)), abs(p.z))
		)
	);
}

} // namespace vkof
