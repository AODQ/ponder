// autogen from scene-visibility-resolve-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-scene-visibility-resolve-pc";
__exported import "util-types";

public struct GpuResolveModelIndirect {
	public u64 meshlets; // VA(GpuMorMeshletBuffer)
	public u64 materials; // VA(GpuMorMaterialBuffer)
	public u64 uvTransforms; // VA(GpuMorUvTransformBuffer)
	public u64 positions; // VA(GpuMorPositionBuffer)
	public u64 instances; // VA(GpuMorInstanceBuffer)
	public u64 attributes; // VA(GpuMorVertexAttributeBuffer)
	public u64 meshletVerts; // VA(GpuMorMeshletVertBuffer)
	public u64 meshletTris; // VA(GpuMorMeshletTriBuffer)
	public f32m44 modelMatrix;
};

// material inspector: what a right-click probe writes into (see
// GpuResolvePc.probeResultVa below); modelDrawIndex -1 means the probed
// pixel missed all geometry
public struct GpuProbeResult {
	public i32 modelDrawIndex;
	public i32 materialIndex;
};

public struct GpuResolvePc {
	public u32 visibilityImageHandle;
	public u32 outputImageHandle;
	public u32 frameIndex;
	public u32 kullaContyEnergyHandle;
	public u32 zeltnerLtcParamHandle;
	// (-1,-1) means no probe requested this frame; see GpuProbeResult
	public i32v2 probePixel;
	public u64 probeResultVa;
	// 0 if no environment map is loaded
	public u32 envRadianceHandle;
	public u32 envIblSpecularHandle;
	public u32 envIblIrradianceHandle;
	// EnvironmentIbl::skSpecularMipCount - 1
	public f32 envIblSpecularMaxLod;
};
