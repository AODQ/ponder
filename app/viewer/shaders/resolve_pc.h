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

// -----------------------------------------------------------------------------
// -- GpuResolveModelIndirect
// -----------------------------------------------------------------------------

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
	// pixel offset of this dispatch within the render target; (0, 0) with a
	// full-screen invocation count for an untiled dispatch. origins are
	// multiples of the tile size (see kPtTileSize in pt-accumulate-pc.h)
	u32 tileOriginX;
	u32 tileOriginY;
	// pt-denoise.comp's g-buffer: world hit position, geometric normal
	u32 gbufferPosHandle;
	u32 gbufferNorHandle;
	// base color at the hit, 1.0 on a miss -- used to demodulate before
	// filtering
	u32 gbufferAlbedoHandle;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuBluenoiseHandleBuffer {
	u32 data[];
};
#ifndef RESOLVE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuResolvePC resolve;
} pc;
#endif
#endif
