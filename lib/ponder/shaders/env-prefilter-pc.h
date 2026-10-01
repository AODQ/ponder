#ifdef __cplusplus
#pragma once
#endif

#ifndef ENV_PREFILTER_PC_H
#define ENV_PREFILTER_PC_H

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define i32 int
#define i32v2 ivec2
#define i32v3 ivec3
#define i32v4 ivec4
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
#endif

// -----------------------------------------------------------------------------
// -- GpuEnvPrefilterPc
// -----------------------------------------------------------------------------

struct GpuEnvPrefilterPc {
	u32 srcRadianceHandle;
	u32 srcRadianceMipCount;
	u32 dstStorageHandle;
	u32 dstWidth;
	u32 dstHeight;
	f32 roughness;
	u32 sampleCount;
};

#ifndef __cplusplus
// vkof's per-dispatch pushconstant window is always [128,256)
layout(push_constant, scalar) uniform PC {
	layout(offset = 128) GpuEnvPrefilterPc prefilter;
} pc;
#endif

#endif // ENV_PREFILTER_PC_H
