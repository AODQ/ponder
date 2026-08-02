// autogen from pt-accumulate-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-pt-accumulate-pc";
__exported import "util-types";

// dispatch tile edge in pixels for the load-balanced resolve window: a
// multiple of the 16x16 threadgroup so interior tiles never overshoot into
// their neighbors, and equal to the 128px bluenoise period so a tile-local
// invocation id samples the same bluenoise texel a full-screen dispatch
// would (util-random.glsl's fnSampleBluenoise wraps coord % 128)
public static const u32 kPtTileSize = 128u;

public struct GpuPtAccumulatePC {
	public u32 inputHandle;
	public u32 accumHandle;
	public u32 outputHandle;
	public u32 reset;
	// single u32 counting COMPLETED PIXELS: each pixel atomicAdd's it exactly
	// once, on the frame its sample count crosses maxSpp (so the target for a
	// full render is renderWidth * renderHeight, not maxSpp). the cpu zeroes
	// it on accumulation reset and reads it live for the imgui progress
	// display
	public u64 sppCounterVa; // VA(GpuSppCounterBuffer)
	// nonzero once the max-spp cap is reached: keep writing the output image
	// (exposure/tonemap stay live) but stop accumulating samples
	public u32 freeze;
	// per-pixel sample target feeding the completion counter above; 0 means
	// no target (accumulate forever, counter never fires)
	public u32 maxSpp;
	// the resolve tile window dispatched this frame: tileCount tiles of a
	// row-major kPtTileSize grid starting at tileFirst, wrapping modulo
	// tileTotal. the grid spans the active REGION -- normally the whole
	// screen (regionTileX/Y = 0, tileGridW/tileTotal = the screen's grid),
	// but a focus rect's tile-snapped sub-grid when one is set, offset by
	// regionTileX/Y tiles from the screen origin. pixels outside the
	// region, or inside it but outside the window, have a stale input
	// image and must re-output their existing average instead of folding
	// it in. tileCount >= tileTotal means the whole region is covered
	public u32 tileFirst;
	public u32 tileCount;
	public u32 tileGridW;
	public u32 tileTotal;
	public u32 regionTileX;
	public u32 regionTileY;
};
