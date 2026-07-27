#ifdef __cplusplus
#pragma once
#endif

#include "shared/global_pc.h"

struct GpuBloomBlurPC {
	// sampler handle (vkofTextures[]), not a storage handle
	u32 inputHandle;
	u32 outputHandle;
	// widening kawase offset: this pass samples at (passIndex + 0.5) texels
	u32 passIndex;
};

#ifndef __cplusplus
#ifndef BLOOM_BLUR_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuBloomBlurPC bloomBlur;
} pc;
#endif
#endif
