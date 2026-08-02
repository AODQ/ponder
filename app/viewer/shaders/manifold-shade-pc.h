#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#include <srat/core-math.hpp>
#endif

struct GpuManifoldShadePC {
	// BendPoint[width * height], written by manifold-probe.slang
	u64 bendPointsVa;
	u32 colorHandle;
	u32 width;
	u32 height;
	u32 pad0;
	// added to every pixel that has a bend point
	f32v3 debugColor;
	u32 pad1;
};

#ifndef __cplusplus
#ifndef MANIFOLD_SHADE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuManifoldShadePC manifoldShade;
} pc;
#endif
#endif
