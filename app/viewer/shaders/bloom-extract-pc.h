#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct GpuBloomExtractPC {
	u32 inputHandle;
	u32 outputHandle;
	// soft-knee bright-pass threshold, in exposure-scaled linear radiance
	f32 threshold;
};

#ifndef __cplusplus
#ifndef BLOOM_EXTRACT_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuBloomExtractPC bloomExtract;
} pc;
#endif
#endif
