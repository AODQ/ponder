// autogen from pt-denoise-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-pt-denoise-pc";
__exported import "util-types";

public struct GpuDenoisePC {
	public u32 inputHandle;
	public u32 outputHandle;
	// always ptAccumImage -- the sample-count/variance source, independent
	// of which buffer inputHandle points at this pass
	public u32 accumHandle;
	public u32 gbufferPosHandle;
	public u32 gbufferNorHandle;
	public u32 gbufferAlbedoHandle;
	// a-trous step in pixels (1, 2, 4, 8, ...)
	public u32 stepSize;
	// position edge-stop distance, scaled to the scene's actual size
	public f32 positionSigma;
	// nonzero on pass 0: divide input by albedo before filtering
	public u32 isFirstPass;
	// nonzero on the last pass: remultiply albedo, tonemap, write display image
	public u32 isFinalPass;
};
