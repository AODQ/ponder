// autogen from bloom-extract-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-bloom-extract-pc";
__exported import "util-types";

public struct GpuBloomExtractPC {
	public u32 inputHandle;
	public u32 outputHandle;
	// soft-knee bright-pass threshold, in exposure-scaled linear radiance
	public f32 threshold;
};
