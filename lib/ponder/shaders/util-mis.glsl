#ifndef UTIL_MIS_GLSL
#define UTIL_MIS_GLSL

#ifndef f32
#define i32v2 ivec2
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

// balance heuristic weight for combining separate sampling techniques with
// different probability densities. Veach's MIS estimator.
f32 misBalanceWeight(const f32 pdfA, const f32 pdfB) {
	const f32 sum = pdfA + pdfB;
	return sum > 0.0f ? pdfA / sum : 0.0f;
}

#endif // UTIL_MIS_GLSL
