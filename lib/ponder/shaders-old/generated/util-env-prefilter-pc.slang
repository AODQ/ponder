// autogen from env-prefilter-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-env-prefilter-pc";
__exported import "util-types";

public struct GpuEnvPrefilterPc {
	public u32 srcRadianceHandle;
	public u32 srcRadianceMipCount;
	public u32 dstStorageHandle;
	public u32 dstWidth;
	public u32 dstHeight;
	public f32 roughness;
	public u32 sampleCount;
};

// vkof's per-dispatch pushconstant window is always [128,256)
