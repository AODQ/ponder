#ifdef __cplusplus
#pragma once
#include <mor/mor-shared.h>
#endif

#include "shared/global-pc.h"

struct GpuResolveModelIndirect {
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorMaterialBuffer) materials;
	VA(GpuMorUvTransformBuffer) uvTransforms;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorVertexAttributeBuffer) attributes;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
using GpuResolveModelIndirectBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuResolveModelIndirectBuffer {
	GpuResolveModelIndirect data[];
};
#endif

// material inspector: what a right-click probe writes into (see
// GpuResolvePc.probeResultVa below); modelDrawIndex -1 means the probed
// pixel missed all geometry
struct GpuProbeResult {
	i32 modelDrawIndex;
	i32 materialIndex;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuProbeResultBuffer {
	GpuProbeResult data;
};
#endif

struct GpuResolvePc {
	u32 visibilityImageHandle;
	u32 outputImageHandle;
	u32 frameIndex;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	// (-1,-1) means no probe requested this frame; see GpuProbeResult
	i32v2 probePixel;
	u64 probeResultVa;
	// 0 if no environment map is loaded
	u32 envRadianceHandle;
	u32 envIblSpecularHandle;
	u32 envIblIrradianceHandle;
	// EnvironmentIbl::skSpecularMipCount - 1
	f32 envIblSpecularMaxLod;
};

#ifndef __cplusplus
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuResolvePc resolve;
} pc;
#endif
