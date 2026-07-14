#ifdef __cplusplus
#pragma once
#include <mor/mor-shared.h>
#endif

#include "shared/global_pc.h"

#ifdef __cplusplus
using GpuFlatIndexBuffer = u64;
using GpuFlatMeshletBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuFlatIndexBuffer {
	u32 data[];
};
layout(buffer_reference, scalar) buffer GpuFlatMeshletBuffer {
	u32 data[];
};
#endif

struct GpuResolveModelIndirect {
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorMaterialBuffer) materials;
	VA(GpuMorUvTransformBuffer) uvTransforms;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorVertexAttributeBuffer) attributes;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	VA(GpuFlatIndexBuffer) flatIndices;
	VA(GpuFlatMeshletBuffer) flatMeshlets;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
using GpuResolveModelIndirectBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuResolveModelIndirectBuffer {
	GpuResolveModelIndirect data[];
};
#endif

struct GpuResolvePC {
	u64 bluenoiseVa;
	u32 outputImageHandle;
	u32 frameIndex;
	u32 bluenoiseCount;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 pad0;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuBluenoiseHandleBuffer {
	u32 data[];
};
#ifndef RESOLVE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuResolvePC resolve;
} pc;
#endif
#endif
