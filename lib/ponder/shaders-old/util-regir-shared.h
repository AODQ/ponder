#ifdef __cplusplus
#pragma once
#endif

#ifndef UTIL_REGIR_SHARED_H
#define UTIL_REGIR_SHARED_H

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
// several headers claim the `#ifndef f32` guard while defining only a subset
#ifndef i32v3
#define i32v3 ivec3
#endif
#endif

// (TODO REVIEW)
// structs, constants and buffer references only. every function lives in
// util-regir.glsl, because this header is included from c++ -- scene-state.cpp
// sizes the buffers from these structs -- where glsl's vector builtins and
// util-random.glsl do not exist

// 16 direction buckets, octahedral. more buckets tighten the cos(theta_s)
// targeting within a cell and multiply the cell count, which is what decides
// how many shading points amortize each fill
#define skRegirNormalResolution 4u
#define skRegirNormalBucketCount \
	(skRegirNormalResolution * skRegirNormalResolution)
// reserved for a vdb medium scatter, which has no surface normal. the target
// function's cos(theta) term must degrade to 1 there, not to a dot product
// against a zero vector -- a silent zero in a target function deletes lights
#define skRegirNormalBucketNone skRegirNormalBucketCount

#define skRegirMaxProbesPerCell 32u
// 0 is a valid slot index, so the miss marker cannot be 0
#define skRegirSlotInvalid 0xffffffffu

// reservoir r of slot s lives at s * skRegirReservoirCount + r
#define skRegirReservoirCount 64u
// candidates streamed into each reservoir during the fill pass
#define skRegirCandidateCount 32u

struct GpuRegirCellKey {
	// floor(worldPosition / cellSize)
	i32v3 gridCoord;
	// quantized direction bucket, or skRegirNormalBucketNone in a medium
	u32 normalBucket;
};

// the stored sample is a point on a light: the target function carries
// cos(theta_L)/d^2, which needs a point rather than an emitter index.
// position and normal are recomputed from (lightIndex, primitiveIndex,
// barycentric) rather than stored, so the fill pass and the shading pass
// cannot disagree about where the sample is
struct GpuRegirReservoir {
	u32 lightIndex;
	u32 primitiveIndex;
	// (b1, b2) in the ray-query convention; b0 is 1 - b1 - b2
	f32v2 barycentric;
	// multiply by this, never divide by a pdf. 0 when the survivor's target
	// value was 0, which must not be allowed to become a nan
	f32 unbiasedWeight;
	// \hat{h}(Y) at fill time, cached so the shading resample and the backward
	// pdf cannot recompute it inconsistently
	f32 targetValue;
};

struct GpuRegirCell {
	GpuRegirCellKey key;
	// the point and normal the target functions were evaluated at, written by
	// the scout pass. racy by design: any shading point that landed in this
	// cell is a legitimate representative
	f32v3 representativePosition;
	f32v3 representativeNormal;
	// (1/M) * sum(\hat{h}(X_i)/p(X_i)) over the cell's candidates -- the monte
	// carlo estimate of \int \hat{h} dA that turns \hat{h} into a pdf. stored
	// rather than recomputed: both halves of every mis weight must read this
	// same number or they price the same direction differently
	f32 normalizationNonCanonical;
	f32 normalizationCanonical;
	// frame index this cell was last filled; drives the refresh policy
	u32 lastFillIndex;
};

#ifndef __cplusplus
// the compare-and-swap target, 0 == empty. its own buffer rather than a field
// on GpuRegirCell: atomics reached through a buffer_reference member are the
// kind of thing that works on one driver and not another, and this keeps the
// single concurrent word away from the bulk payload written after it
layout(buffer_reference, scalar) buffer GpuRegirGuardBuffer {
	u32 data[];
};
layout(buffer_reference, scalar) buffer GpuRegirCellBuffer {
	GpuRegirCell data[];
};
layout(buffer_reference, scalar) buffer GpuRegirReservoirBuffer {
	GpuRegirReservoir data[];
};
#endif
// (TODO REVIEW)

#endif // UTIL_REGIR_SHARED_H
