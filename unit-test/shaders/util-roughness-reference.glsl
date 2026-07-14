#ifndef UTIL_ROUGHNESS_REFERENCE_GLSL
#define UTIL_ROUGHNESS_REFERENCE_GLSL

#ifndef f32
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#endif

// independent cross-check only, not used by the real ponder implementation.
//
// this is the OpenPBR spec's actual formula (fetched directly from
// academysoftwarefoundation.github.io/OpenPBR/index.html, "specular
// roughness and anisotropy" section):
//
//   alpha_t = r^2 * sqrt(2 / (1 + (1-a)^2))
//   alpha_b = (1-a) * alpha_t
//
// note the r^2 (roughness SQUARED) -- ponder's openPbrRoughnessAlpha
// (both the production and standalone copies) uses r (linear, unsquared)
// instead, matching cull's shader-side formula 1:1. this is a known,
// deliberate mismatch against the spec text (see
// unit-test/src/test-roughness.cpp for the test that documents it).
f32v2 openPbrRoughnessAlphaSpec(const f32 roughness, const f32 anisotropy) {
	const f32 oneMinusA = 1.0f - anisotropy;
	const f32 alphaT = (
		roughness * roughness * sqrt(2.0f / (1.0f + oneMinusA * oneMinusA))
	);
	return f32v2(alphaT, oneMinusA * alphaT);
}

#endif
