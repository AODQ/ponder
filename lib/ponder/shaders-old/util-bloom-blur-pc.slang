// autogen from bloom-blur-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-bloom-blur-pc";
__exported import "util-types";

public struct GpuBloomBlurPC {
	// sampler handle (vkofTextures[]), not a storage handle
	public u32 inputHandle;
	public u32 outputHandle;
	// widening kawase offset: this pass samples at (passIndex + 0.5) texels
	public u32 passIndex;
};
