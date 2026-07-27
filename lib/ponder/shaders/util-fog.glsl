#ifndef UTIL_FOG_GLSL
#define UTIL_FOG_GLSL

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

#include "util-random.glsl"

// -----------------------------------------------------------------------------
// -- homogeneous fog
// -----------------------------------------------------------------------------

#define skUtilFogPhaseIsotropic (1.0f / (4.0f * skPi))

// sigma is extinction per world unit (GpuGlobalExtended.fogThickness);
// caller guards sigma > 0. free-flight distance through constant sigma is
// exactly exponential and transmittance is closed-form beer-lambert -- no
// majorant/rejection loop needed, unlike the heterogeneous vdb case
f32 utilFogSampleDistance(const f32 sigma, inout u64 state) {
	return -log(1.0f - fnSampleUniform(state)) / sigma;
}

f32 utilFogTransmittance(const f32 sigma, const f32 dist) {
	return exp(-sigma * dist);
}

#endif
