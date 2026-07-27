#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct GpuDenoisePC {
	u32 inputHandle;
	u32 outputHandle;
	// always ptAccumImage -- the sample-count/variance source, independent
	// of which buffer inputHandle points at this pass
	u32 accumHandle;
	u32 gbufferPosHandle;
	u32 gbufferNorHandle;
	u32 gbufferAlbedoHandle;
	// a-trous step in pixels (1, 2, 4, 8, ...)
	u32 stepSize;
	// position edge-stop distance, scaled to the scene's actual size
	f32 positionSigma;
	// nonzero on pass 0: divide input by albedo before filtering
	u32 isFirstPass;
	// nonzero on the last pass: remultiply albedo, tonemap, write display image
	u32 isFinalPass;
};

#ifndef __cplusplus
#ifndef PT_DENOISE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuDenoisePC denoise;
} pc;
#endif
#endif
