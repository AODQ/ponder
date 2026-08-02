// autogen from resolve_pc.h by
// lib/tools/shared-header-to-slang.py

module "util-resolve-pc";
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
	public u64 flatIndices; // VA(GpuFlatIndexBuffer)
	public u64 flatMeshlets; // VA(GpuFlatMeshletBuffer)
	public f32m44 modelMatrix;
	// (TODO REVIEW)
	// this model's slice of the frame's emitter list; the backward (bsdf-hit)
	// mis lookup scans only its own model's emitters
	public u32 emitterLightOffset;
	public u32 emitterLightCount;
	// (TODO REVIEW)
};

public struct GpuResolvePC {
	public u64 bluenoiseVa;
	public u32 outputImageHandle;
	public u32 frameIndex;
	public u32 bluenoiseCount;
	public u32 kullaContyEnergyHandle;
	public u32 zeltnerLtcParamHandle;
	// pixel offset of this dispatch within the render target; (0, 0) with a
	// full-screen invocation count for an untiled dispatch. origins are
	// multiples of the tile size (see kPtTileSize in pt-accumulate-pc.h)
	public u32 tileOriginX;
	public u32 tileOriginY;
	// pt-denoise.comp's g-buffer: world hit position, geometric normal
	public u32 gbufferPosHandle;
	public u32 gbufferNorHandle;
	// base color at the hit, 1.0 on a miss -- used to demodulate before
	// filtering
	public u32 gbufferAlbedoHandle;
};
