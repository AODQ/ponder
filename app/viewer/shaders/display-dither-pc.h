#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct GpuDisplayDitherPC {
	// display image, already tonemapped
	u32 inputHandle;
	// presented image, linear; the srgb swapchain blit quantizes it
	u32 outputHandle;
	// decorrelates the dither between frames
	u32 frameIndex;
};

#ifndef __cplusplus
#ifndef DISPLAY_DITHER_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuDisplayDitherPC displayDither;
} pc;
#endif
#endif
