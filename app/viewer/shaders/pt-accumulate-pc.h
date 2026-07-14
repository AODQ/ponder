#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#endif

struct GpuPtAccumulatePC {
	u32 inputHandle;
	u32 accumHandle;
	u32 outputHandle;
	u32 reset;
};

#ifndef __cplusplus
#ifndef PT_ACCUMULATE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuPtAccumulatePC ptAccumulate;
} pc;
#endif
#endif
