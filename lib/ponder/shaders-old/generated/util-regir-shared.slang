// autogen from util-regir-shared.h by
// lib/tools/shared-header-to-slang.py

module "util-regir-shared";
__exported import "util-types";

// several headers claim the `#ifndef f32` guard while defining only a subset
#ifndef i32v3
#define i32v3 i32v3
#endif

// (TODO REVIEW)
// structs, constants and buffer references only. every function lives in
// util-regir.glsl, because this header is included from c++ -- scene-state.cpp
// sizes the buffers from these structs -- where glsl's vector builtins and
// util-random.glsl do not exist

// 16 direction buckets, octahedral. more buckets tighten the cos(theta_s)
// targeting within a cell and multiply the cell count, which is what decides
// how many shading points amortize each fill
public static const u32 skRegirNormalResolution = 4u;
public static const u32 skRegirNormalBucketCount = (skRegirNormalResolution * skRegirNormalResolution);
// reserved for a vdb medium scatter, which has no surface normal. the target
// function's cos(theta) term must degrade to 1 there, not to a dot product
// against a zero vector -- a silent zero in a target function deletes lights
public static const u32 skRegirNormalBucketNone = skRegirNormalBucketCount;

public static const u32 skRegirMaxProbesPerCell = 32u;
// 0 is a valid slot index, so the miss marker cannot be 0
public static const u32 skRegirSlotInvalid = 0xffffffffu;

// reservoir r of slot s lives at s * skRegirReservoirCount + r
public static const u32 skRegirReservoirCount = 64u;
// candidates streamed into each reservoir during the fill pass
public static const u32 skRegirCandidateCount = 32u;

public struct GpuRegirCellKey {
	// floor(worldPosition / cellSize)
	public i32v3 gridCoord;
	// quantized direction bucket, or skRegirNormalBucketNone in a medium
	public u32 normalBucket;
};

// the stored sample is a point on a light: the target function carries
// cos(theta_L)/d^2, which needs a point rather than an emitter index.
// position and normal are recomputed from (lightIndex, primitiveIndex,
// barycentric) rather than stored, so the fill pass and the shading pass
// cannot disagree about where the sample is
public struct GpuRegirReservoir {
	public u32 lightIndex;
	public u32 primitiveIndex;
	// (b1, b2) in the ray-query convention; b0 is 1 - b1 - b2
	public f32v2 barycentric;
	// multiply by this, never divide by a pdf. 0 when the survivor's target
	// value was 0, which must not be allowed to become a nan
	public f32 unbiasedWeight;
	// \hat{h}(Y) at fill time, cached so the shading resample and the backward
	// pdf cannot recompute it inconsistently
	public f32 targetValue;
};

public struct GpuRegirCell {
	public GpuRegirCellKey key;
	// the point and normal the target functions were evaluated at, written by
	// the scout pass. racy by design: any shading point that landed in this
	// cell is a legitimate representative
	public f32v3 representativePosition;
	public f32v3 representativeNormal;
	// (1/M) * sum(\hat{h}(X_i)/p(X_i)) over the cell's candidates -- the monte
	// carlo estimate of \i32 \hat{h} dA that turns \hat{h} into a pdf. stored
	// rather than recomputed: both halves of every mis weight must read this
	// same number or they price the same direction differently
	public f32 normalizationNonCanonical;
	public f32 normalizationCanonical;
	// frame index this cell was last filled; drives the refresh policy
	public u32 lastFillIndex;
};

// the compare-and-swap target, 0 == empty. its own buffer rather than a field
// on GpuRegirCell: atomics reached through a buffer_reference member are the
// kind of thing that works on one driver and not another, and this keeps the
// single concurrent word away from the bulk payload written after it

// (TODO REVIEW)
