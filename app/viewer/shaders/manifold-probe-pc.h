#pragma once

// c++ mirror of manifold-probe.slang's ProbePush.
// no solution yet for glsl<->slang<->c++ translation

#include <srat/core-types.hpp>
#include <srat/core-math.hpp>

// mirrors manifold.MeshData in util-manifold-mesh.slang
struct GpuManifoldMeshData {
	u64 positionVa;
	u64 attributeVa;
	u64 indexVa;
	f32m44 modelMatrix;
	u32 triangleCount;
	u32 pad;
};
static_assert(sizeof(GpuManifoldMeshData) == 96u);

// u32[4] counters,
// [0] seeds attempted
// [1] seeds that reached a bend point
// [2] seeds whose first reprojection left the caster
// mirrors BendPoint in manifold-probe.slang
struct GpuManifoldBendPoint {
	f32v3 point;
	u32 converged;
	f32v3 normal;
	u32 pad;
};
static_assert(sizeof(GpuManifoldBendPoint) == 32u);

struct GpuManifoldProbePC {
	u64 casterVa;
	f32v3 pointWo;
	u32 gbufferPosHandle;
	u32 gbufferNorHandle;
	u32 width;
	u32 height;
	u32 frameIndex;
	u64 countsVa;
	// GpuManifoldBendPoint[width * height]
	u64 bendPointsVa;
};
