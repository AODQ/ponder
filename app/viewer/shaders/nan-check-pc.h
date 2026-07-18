#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#endif

struct GpuNanCheckPC {
	// u32[2] counters: [0] NaN pixels (true NaN or the propagation entry's
	// red 9999 sentinel), [1] Inf pixels; accumulated atomically across
	// frames, zeroed by the cpu on accumulation reset
	u64 countsVa;
	u32 inputHandle;
	u32 width;
	u32 height;
	// pixel offset of this dispatch within the render target; (0, 0) with a
	// full-screen invocation count for an untiled dispatch
	u32 tileOriginX;
	u32 tileOriginY;
	u32 pad0;
};

#ifndef __cplusplus
#ifndef NAN_CHECK_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPC global;
	GpuNanCheckPC nanCheck;
} pc;
#endif
#endif
