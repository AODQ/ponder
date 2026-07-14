#ifdef __cplusplus
#pragma once
#endif

#ifndef GLOBAL_PC_H
#define GLOBAL_PC_H

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define i32 int
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u64 uint64_t
#endif
#endif

#define VA(Type) u64

// per-frame values that change too often (or are too large) for the 128-byte
// root pushconstant; lives in a host-writable buffer referenced by
// GpuGlobalPC.debug
struct GpuDebugPC {
	f32 envIntensity;
	u32 renderWidth;
	u32 renderHeight;
	// 0=furnace, 1=checkerboard, 2=black (ENV_MODE_* in environment.glsl)
	i32 envMode;
	// subpixel jitter on the primary camera ray (antialiasing); off traces
	// every sample through the exact pixel center
	u32 antialias;
	// single u32 atomic counter, zeroed by the cpu every frame; caps
	// util-nan-probe.glsl's debugPrintfEXT firings to
	// NAN_PROBE_LIMIT_PER_FRAME (see resolve.comp's nanProbeInit call)
	u64 nanProbeCounterVa;
	// material inspector: pixel coordinate to probe this frame, set by the
	// cpu on right-click. (-1,-1) means no probe requested. resolve.comp
	// debugPrintfEXT's the hit's material index when gl_GlobalInvocationID
	// matches, which the cpu parses back out of vkof::probe_message next
	// frame; one-shot (the cpu resets this to (-1,-1) right after upload)
	i32v2 probePixel;
};

// root pushconstant, shared by every node in the frame's render graph;
// pinned to exactly vkof's 128-byte root pushconstant range so each node's
// own pushconstant struct starts at offset 128 in the shader-side PC block
struct GpuGlobalPC {
	f32 time;
	f32v3 cameraPos;
	f32 exposure;
	f32 pad0;
	f32m44 viewProj;
	VA(GpuDebugPC) debug;
	VA(GpuResolveModelIndirectBuffer) models;
	u64 pad1;
	u64 pad2;
	u64 pad3;
};

#ifdef __cplusplus
static_assert(sizeof(GpuGlobalPC) == 128, "GpuGlobalPC must be 128 bytes");
#else
layout(buffer_reference, scalar) buffer GpuDebugPCBuffer {
	GpuDebugPC data;
};
#endif

#endif // GLOBAL_PC_H
