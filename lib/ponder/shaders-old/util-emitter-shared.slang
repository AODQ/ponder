// autogen from emitter-shared.h by
// lib/tools/shared-header-to-slang.py

module "util-emitter-shared";
__exported import "util-types";

// (TODO REVIEW)
// emitter kinds. a non-mesh entry (kind != mesh) carries triangleCount == 0,
// sits outside every model's emitterLightOffset/Count range, and so can never
// be matched by the backward hit lookup -- it exists only to take its share
// of the power-weighted pick
public static const u32 EMITTER_KIND_MESH = 0u;
public static const u32 EMITTER_KIND_ENVIRONMENT = 1u;

// one entry per emissive meshlet on a tlas instance; rebuilt every frame
// alongside GpuResolveModelIndirect so material-override edits stay live.
// triangles are addressed the way ray hits report them: flat indices into
// GpuResolveModelIndirect.flatIndices/flatMeshlets, NOT the meshlet-local
// triangleOffset/meshletTris numbering the raster path uses
public struct GpuEmitterLight {
	// one of EMITTER_KIND_*
	public u32 kind;
	public u32 modelDrawIndex;
	// flat primitiveIndex of this emitter's first triangle; triangles
	// [primitiveIndexBase, primitiveIndexBase + triangleCount) are this emitter
	public u32 primitiveIndexBase;
	public u32 triangleCount;
	// world area * luma(emissionColor) * emissionLuminance. steers the pick
	// only -- the pdf recomputes exact world area per triangle in-shader, so
	// an approximate power costs variance and never correctness
	public f32 power;
	// cumulative power / totalPower across the whole list, monotonic
	// increasing; the last entry is 1.0
	public f32 cdf;
	// emissionColor * emissionLuminance, precomputed. a fallback only: the
	// sampler evaluates the sampled point's material so textured emitters
	// stay consistent with the bsdf-hit branch
	public f32v3 radiance;
};

// (TODO REVIEW)
