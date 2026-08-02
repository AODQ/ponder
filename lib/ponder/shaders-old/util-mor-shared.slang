// autogen from mor-shared.h by
// lib/tools/shared-header-to-slang.py

module "util-mor-shared";
__exported import "util-types";

public static const u32 MOR_DEBUG_VIEW_MESHLET_ID = 1u;
public static const u32 MOR_DEBUG_VIEW_MODEL_ID = 2u;
public static const u32 MOR_DEBUG_VIEW_TRIANGLE_ID = 3u;
public static const u32 MOR_DEBUG_VIEW_MIP_HEATMAP = 4u;
public static const u32 MOR_DEBUG_VIEW_ALBEDO = 5u;
public static const u32 MOR_DEBUG_VIEW_WORLD_NORMAL = 6u;
public static const u32 MOR_DEBUG_VIEW_ROUGHNESS = 7u;
public static const u32 MOR_DEBUG_VIEW_METALLIC = 8u;
public static const u32 MOR_DEBUG_VIEW_EMISSIVE = 9u;
public static const u32 MOR_DEBUG_VIEW_UV = 10u;

public struct GpuMorVertexAttribute {
	public f32v3 normal;
	public f32v2 uv;
	public f32v4 tangent;
};

public struct GpuMorMeshlet {
	public u32 vertexOffset;
	public u32 vertexCount;
	public u32 triangleOffset;
	public u32 triangleCount;
	public u32 instanceIndex;
	public u32 materialIndex;
};

// gltf KHR_texture_transform, precomputed into a 2x2 rotate-scale matrix
// (basisU, basisV columns) plus an offset so the shader never pays for
// sin/cos per sample; index 0 is reserved for the identity transform
public struct GpuMorUvTransform {
	public f32v2 basisU;
	public f32v2 basisV;
	public f32v2 offset;
};

public struct GpuMorMaterialComponent3 {
	public f32v3 rgb;
	public u32 texture;
	// index into GpuMorUvTransformBuffer; 0 is identity
	public u32 uvTransform;
};

// GpuMorMaterialComponent1.swizzle: bits 0..1 select the sampled texture
// channel; the invert bit flips the resolved value to 1 - x (after the
// factor multiply), used for gltf spec-gloss glossiness -> roughness
public static const u32 MOR_MATERIAL_SWIZZLE_CHANNEL_MASK = 3u;
public static const u32 MOR_MATERIAL_SWIZZLE_INVERT = 4u;

public struct GpuMorMaterialComponent1 {
	public f32 r;
	public u32 texture;
	public i32 swizzle;
	// index into GpuMorUvTransformBuffer; 0 is identity
	public u32 uvTransform;
};

// (TODO REVIEW)
// gltf alpha_mode. opaque forces geometryOpacity to 1, mask hard-thresholds
// it against alphaCutoff, blend keeps it as fractional coverage
public static const u32 MOR_ALPHA_MODE_OPAQUE = 0u;
public static const u32 MOR_ALPHA_MODE_MASK = 1u;
public static const u32 MOR_ALPHA_MODE_BLEND = 2u;
// (TODO REVIEW)

public struct GpuMorMaterial {
	// based off OpenPBR however every single parameter can have a texture
	// plugin. texture is assumed to be 1.0f in all channels if not set.

	// -- base ------------------------------------------------------------------
	public GpuMorMaterialComponent1 baseWeight;
	// base reflection albedo
	public GpuMorMaterialComponent3 baseColor;
	// fraction of the base reflection that is metallic
	public GpuMorMaterialComponent1 baseMetalness;
	// roughness of the diffuse lobe
	public GpuMorMaterialComponent1 baseDiffuseRoughness;

	// -- specular --------------------------------------------------------------
	public GpuMorMaterialComponent1 specularWeight;

	// tints the dielectric fresnel factor
	public GpuMorMaterialComponent3 specularColor;
	// roughness of the NDF dielectric lobe
	public GpuMorMaterialComponent1 specularRoughness;
	// anisotropy of the NDF dielectric lobe
	public GpuMorMaterialComponent1 specularRoughnessAnisotropy;
	// not an openpbr parameter: openpbr expresses the anisotropy direction
	// purely via the geometry tangent. this is a gltf interop field
	// (khr_materials_anisotropy anisotropyRotation): radians,
	// counter-clockwise from the tangent toward the bitangent
	public GpuMorMaterialComponent1 specularRoughnessAnisotropyRotation;
	// refractive index of the dielectric lobe
	public GpuMorMaterialComponent1 specularIor;

	// -- transmission ----------------------------------------------------------
	public GpuMorMaterialComponent1 transmissionWeight;

	// extinction of the dielectric medium
	public GpuMorMaterialComponent3 transmissionColor;
	// depth into the volume that transmission color is realized.
	// If zero acts as a constant transmission tint
	public GpuMorMaterialComponent1 transmissionDepth;
	// scattering coefficient of the interior medium
	public GpuMorMaterialComponent3 transmissionScatter;
	// anisotropy of the henyey-greenstein phase function
	public GpuMorMaterialComponent1 transmissionScatterAnisotropy;
	// linear scale of the dispersion effect
	public GpuMorMaterialComponent1 transmissionDispersionScale;
	// physical abbe number of the base dielectric medium
	public GpuMorMaterialComponent1 transmissionDispersionAbbeNumber;

	// -- subsurface
	public GpuMorMaterialComponent1 subsurfaceWeight;
	// observed reflection color of the subsurface scattering effect
	public GpuMorMaterialComponent3 subsurfaceColor;
	// length scale of the mean free path
	public GpuMorMaterialComponent1 subsurfaceRadius;
	// rgb multiplier of the subsurface radius; per-channel mean free paths
	public GpuMorMaterialComponent3 subsurfaceRadiusScale;
	// anistropy of the henyey-greenstein phase function for subsurface scatter
	public GpuMorMaterialComponent1 subsurfaceScatterAnisotropy;

	// -- coat
	public GpuMorMaterialComponent1 coatWeight;
	// square of normal-incidence transmittance of the coat layer
	public GpuMorMaterialComponent3 coatColor;
	// roughness of the NDF coat lobe
	public GpuMorMaterialComponent1 coatRoughness;
	// anisotropy of the NDF coat lobe
	public GpuMorMaterialComponent1 coatRoughnessAnisotropy;
	// refractive index of the coat layer
	public GpuMorMaterialComponent1 coatIor;
	// modulates the coat layer's darkening effect
	public GpuMorMaterialComponent1 coatDarkening;

	// -- fuzz
	public GpuMorMaterialComponent1 fuzzWeight;
	// reflection albedo of the fuzz layer
	public GpuMorMaterialComponent3 fuzzColor;
	// roughness of the NDF fuzz lobe
	public GpuMorMaterialComponent1 fuzzRoughness;

	// -- emission
	// emission luminance in nits
	public GpuMorMaterialComponent1 emissionLuminance;
	// emission color multiplier
	public GpuMorMaterialComponent3 emissionColor;

	// -- thin film
	public GpuMorMaterialComponent1 thinFilmWeight;
	// thickness of the thin film layer in micrometers
	public GpuMorMaterialComponent1 thinFilmThickness;
	// refractive index of the thin film layer
	public GpuMorMaterialComponent1 thinFilmIor;

	// -- geometry
	// opacity of the geometry surface
	public GpuMorMaterialComponent1 geometryOpacity;
	// if enabled makes the slab structure mirrored
	public GpuMorMaterialComponent1 geometryThinWalled;

	public u32 geometryNormalTexture;
	public u32 geometryNormalUvTransform;
	public u32 geometryTangentTexture;
	public u32 geometryTangentUvTransform;
	public u32 geometryCoatNormalTexture;
	public u32 geometryCoatNormalUvTransform;
	public u32 geometryCoatTangentTexture;
	public u32 geometryCoatTangentUvTransform;

	// -- gltf extras, keep minimal
	public f32 alphaCutoff;
	// (TODO REVIEW)
	// MOR_ALPHA_MODE_*, selects how geometryOpacity is resolved
	public u32 alphaMode;
	// (TODO REVIEW)
};

// sometimes the above is a bit repetitive, just use X macros

// (TODO REVIEW)
// third arg is what the map decodes to with no texture bound: +z for a
// normal, +x for a tangent, so tbn * identity reproduces the mesh basis
// (TODO REVIEW)

// the plain u32 texture fields not covered by MOR_MATERIAL_ALL_PARAMS

public struct GpuMorInstance {
	public f32m44 transform;
	public u32 meshletOffset;
	public u32 meshletCount;
};
