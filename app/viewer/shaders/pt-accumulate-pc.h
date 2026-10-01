#ifdef __cplusplus
#pragma once
#include <srat/core-types.hpp>
#endif

#include "shared/global_pc.h"

// dispatch tile edge in pixels for the load-balanced resolve window: a
// multiple of the 16x16 threadgroup so interior tiles never overshoot into
// their neighbors, and equal to the 128px bluenoise period so a tile-local
// invocation id samples the same bluenoise texel a full-screen dispatch
// would (util-random.glsl's fnSampleBluenoise wraps coord % 128)
#ifdef __cplusplus
constexpr u32 kPtTileSize = 128u;
#else
const u32 kPtTileSize = 128u;
#endif

// -----------------------------------------------------------------------------
// -- GpuPtAccumulatePC
// -----------------------------------------------------------------------------

struct GpuPtAccumulatePC {
	u32 inputHandle;
	u32 accumHandle;
	u32 outputHandle;
	u32 reset;
	// single u32 counting COMPLETED PIXELS: each pixel atomicAdd's it exactly
	// once, on the frame its sample count crosses maxSpp (so the target for a
	// full render is renderWidth * renderHeight, not maxSpp). the cpu zeroes
	// it on accumulation reset and reads it live for the imgui progress
	// display
	VA(GpuSppCounterBuffer) sppCounterVa;
	// nonzero once the max-spp cap is reached: keep writing the output image
	// (exposure/tonemap stay live) but stop accumulating samples
	u32 freeze;
	// per-pixel sample target feeding the completion counter above; 0 means
	// no target (accumulate forever, counter never fires)
	u32 maxSpp;
	// the resolve tile window dispatched this frame: tileCount tiles of a
	// row-major kPtTileSize grid starting at tileFirst, wrapping modulo
	// tileTotal. the grid spans the active REGION -- normally the whole
	// screen (regionTileX/Y = 0, tileGridW/tileTotal = the screen's grid),
	// but a focus rect's tile-snapped sub-grid when one is set, offset by
	// regionTileX/Y tiles from the screen origin. pixels outside the
	// region, or inside it but outside the window, have a stale input
	// image and must re-output their existing average instead of folding
	// it in. tileCount >= tileTotal means the whole region is covered
	u32 tileFirst;
	u32 tileCount;
	u32 tileGridW;
	u32 tileTotal;
	u32 regionTileX;
	u32 regionTileY;
};

#ifndef __cplusplus
layout(buffer_reference, scalar) buffer GpuSppCounterBuffer {
	u32 data[];
};
#ifndef PT_ACCUMULATE_NO_PC
layout(push_constant, scalar) uniform PC {
	GpuGlobalPc global;
	GpuPtAccumulatePC ptAccumulate;
} pc;
#endif
#endif
