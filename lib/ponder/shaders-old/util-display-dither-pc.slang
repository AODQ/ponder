// autogen from display-dither-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-display-dither-pc";
__exported import "util-types";

public struct GpuDisplayDitherPC {
	// display image, already tonemapped
	public u32 inputHandle;
	// presented image, linear; the srgb swapchain blit quantizes it
	public u32 outputHandle;
	// decorrelates the dither between frames
	public u32 frameIndex;
};
