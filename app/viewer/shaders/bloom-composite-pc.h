#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct GpuBloomCompositePC {
	// read-modify-write: the already-tonemapped display image
	u32 colorHandle;
	u32 bloomHandle;
	f32 intensity;
};

#ifndef __cplusplus
#ifndef BLOOM_COMPOSITE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuBloomCompositePC bloomComposite;
} pc;
#endif
#endif
