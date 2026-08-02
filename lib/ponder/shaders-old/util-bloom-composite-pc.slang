// autogen from bloom-composite-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-bloom-composite-pc";
__exported import "util-types";

public struct GpuBloomCompositePC {
	// read-modify-write: the already-tonemapped display image
	public u32 colorHandle;
	public u32 bloomHandle;
	public f32 intensity;
};
