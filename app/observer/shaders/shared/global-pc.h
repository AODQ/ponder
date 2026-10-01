#ifdef __cplusplus
#pragma once
#endif

#ifndef GLOBAL_PC_H
#define GLOBAL_PC_H

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else // !defined(__cplusplus)
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
#endif // f32
#endif // __cplusplus

#define VA(Type) u64

// -----------------------------------------------------------------------------
// -- GpuGlobalExtended
// -----------------------------------------------------------------------------

// per-frame values that are too large for the 128-byte root pushconstant
struct GpuGlobalExtended {
	u32 renderWidth;
	u32 renderHeight;
};

// root pushconstant, must be exactly 128 bytes so that each node has their
// own 128-byte pushconstant range at offset 128
struct GpuGlobalPc {
	f32 time;
	f32v3 cameraPos;
	f32 exposure;
	f32 pad0;
	f32m44 viewProj;
	// VA(GpuGlobalExtended) extended;
	VA(GpuResolveModelIndirectBuffer) models;
	u32 renderWidth;
	u32 renderHeight;
	u64 pad1;
	u64 pad2;
	u64 pad3;
};

#ifdef __cplusplus
// the below can not be changed, the size must always be 128 to match
// the shader-side push-constants
static_assert(sizeof(GpuGlobalPc) == 128, "GpuGlobalPC must be 128 bytes");
#else
layout(buffer_reference, scalar) buffer GpuGlobalExtendedBuffer {
	GpuGlobalExtended data;
};
#endif


#endif // GLOBAL_PC_H
