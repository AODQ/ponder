// autogen from nan-check-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-nan-check-pc";
__exported import "util-types";

public struct GpuNanCheckPC {
	// u32[2] counters: [0] NaN pixels (true NaN or the propagation entry's
	// red 9999 sentinel), [1] Inf pixels; accumulated atomically across
	// frames, zeroed by the cpu on accumulation reset
	public u64 countsVa;
	public u32 inputHandle;
	public u32 width;
	public u32 height;
	// pixel offset of this dispatch within the render target; (0, 0) with a
	// full-screen invocation count for an untiled dispatch
	public u32 tileOriginX;
	public u32 tileOriginY;
	public u32 pad0;
};
