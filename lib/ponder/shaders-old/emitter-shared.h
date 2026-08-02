#ifdef __cplusplus
#pragma once
#endif

#ifndef EMITTER_SHARED_H
#define EMITTER_SHARED_H

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

// (TODO REVIEW)
// emitter kinds. a non-mesh entry (kind != mesh) carries triangleCount == 0,
// sits outside every model's emitterLightOffset/Count range, and so can never
// be matched by the backward hit lookup -- it exists only to take its share
// of the power-weighted pick
#define EMITTER_KIND_MESH 0u
#define EMITTER_KIND_ENVIRONMENT 1u

// one entry per emissive meshlet on a tlas instance; rebuilt every frame
// alongside GpuResolveModelIndirect so material-override edits stay live.
// triangles are addressed the way ray hits report them: flat indices into
// GpuResolveModelIndirect.flatIndices/flatMeshlets, NOT the meshlet-local
// triangleOffset/meshletTris numbering the raster path uses
struct GpuEmitterLight {
	// one of EMITTER_KIND_*
	u32 kind;
	u32 modelDrawIndex;
	// flat primitiveIndex of this emitter's first triangle; triangles
	// [primitiveIndexBase, primitiveIndexBase + triangleCount) are this emitter
	u32 primitiveIndexBase;
	u32 triangleCount;
	// world area * luma(emissionColor) * emissionLuminance. steers the pick
	// only -- the pdf recomputes exact world area per triangle in-shader, so
	// an approximate power costs variance and never correctness
	f32 power;
	// cumulative power / totalPower across the whole list, monotonic
	// increasing; the last entry is 1.0
	f32 cdf;
	// emissionColor * emissionLuminance, precomputed. a fallback only: the
	// sampler evaluates the sampled point's material so textured emitters
	// stay consistent with the bsdf-hit branch
	f32v3 radiance;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuEmitterLightBuffer {
	GpuEmitterLight data[];
};
#endif
// (TODO REVIEW)

#endif // EMITTER_SHARED_H
