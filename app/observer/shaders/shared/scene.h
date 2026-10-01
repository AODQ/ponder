#ifdef __cplusplus
#pragma once
#endif

#ifndef SCENE_SHARED_H
#define SCENE_SHARED_H

#ifdef __cplusplus
#include <mor/mor-shared.h>
#else
#include "mor/mor-shared.h"
#include "shared/global-pc.h"
#endif

// -----------------------------------------------------------------------------
// -- GpuSceneDrawPc
// -----------------------------------------------------------------------------

struct GpuSceneDrawPc {
	u32 modelId;
	VA(GpuMorMeshletBuffer) meshlets;
	VA(GpuMorPositionBuffer) positions;
	VA(GpuMorInstanceBuffer) instances;
	VA(GpuMorMeshletTriBuffer) meshletTris;
	VA(GpuMorMeshletVertBuffer) meshletVerts;
	f32m44 modelMatrix;
};

#ifdef __cplusplus
static_assert(sizeof(GpuSceneDrawPc) <= 128, "GpuSceneDrawPc must be <= 128b");
#else
layout(push_constant, scalar) uniform GpuSceneDrawPcPushConstant {
	GpuGlobalPc global;
	GpuSceneDrawPc draw;
} pc;
#endif

#endif // SCENE_SHARED_H
