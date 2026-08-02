module "util-tonemap";
__exported import "util-types";
// util for tonemapping linear HDR radiance to display-ready color


public func fnAces(
	x: f32v3,
) -> f32v3 {
	const f32 a = 2.51f;
	const f32 b = 0.03f;
	const f32 c = 2.43f;
	const f32 d = 0.59f;
	const f32 e = 0.14f;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

