#ifdef __cplusplus
#pragma once
#endif

#ifndef MOR_SHARED_H
#define MOR_SHARED_H

#define MOR_DEBUG_VIEW_MESHLET_ID 1
#define MOR_DEBUG_VIEW_MODEL_ID 2
#define MOR_DEBUG_VIEW_TRIANGLE_ID 3
#define MOR_DEBUG_VIEW_MIP_HEATMAP 4
#define MOR_DEBUG_VIEW_ALBEDO 5
#define MOR_DEBUG_VIEW_WORLD_NORMAL 6
#define MOR_DEBUG_VIEW_ROUGHNESS 7
#define MOR_DEBUG_VIEW_METALLIC 8
#define MOR_DEBUG_VIEW_EMISSIVE 9
#define MOR_DEBUG_VIEW_UV 10

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define i32 int
#define i32v2 ivec2
#define f32   float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif
#endif

struct GpuMorVertexAttribute {
	f32v3 normal;
	f32v2 uv;
	f32v4 tangent;
};

struct GpuMorMeshlet {
	u32 vertexOffset;
	u32 vertexCount;
	u32 triangleOffset;
	u32 triangleCount;
	u32 instanceIndex;
	u32 materialIndex;
};

// gltf KHR_texture_transform, precomputed into a 2x2 rotate-scale matrix
// (basisU, basisV columns) plus an offset so the shader never pays for
// sin/cos per sample; index 0 is reserved for the identity transform
struct GpuMorUvTransform {
	f32v2 basisU;
	f32v2 basisV;
	f32v2 offset;
};

struct GpuMorMaterialComponent3 {
	f32v3 rgb;
	u32 texture;
	// index into GpuMorUvTransformBuffer; 0 is identity
	u32 uvTransform;
};

struct GpuMorMaterialComponent1 {
	f32 r;
	u32 texture;
	i32 swizzle;
	// index into GpuMorUvTransformBuffer; 0 is identity
	u32 uvTransform;
};

struct GpuMorMaterial {
	// based off OpenPBR however every single parameter can have a texture
	// plugin. texture is assumed to be 1.0f in all channels if not set.

	// -- base ------------------------------------------------------------------
	GpuMorMaterialComponent1 baseWeight;
	// base reflection albedo
	GpuMorMaterialComponent3 baseColor;
	// fraction of the base reflection that is metallic
	GpuMorMaterialComponent1 baseMetalness;
	// roughness of the diffuse lobe
	GpuMorMaterialComponent1 baseDiffuseRoughness;

	// -- specular --------------------------------------------------------------
	GpuMorMaterialComponent1 specularWeight;

	// tints the dielectric fresnel factor
	GpuMorMaterialComponent3 specularColor;
	// roughness of the NDF dielectric lobe
	GpuMorMaterialComponent1 specularRoughness;
	// anisotropy of the NDF dielectric lobe
	GpuMorMaterialComponent1 specularRoughnessAnisotropy;
	// not an openpbr parameter: openpbr expresses the anisotropy direction
	// purely via the geometry tangent. this is a gltf interop field
	// (khr_materials_anisotropy anisotropyRotation): radians,
	// counter-clockwise from the tangent toward the bitangent
	GpuMorMaterialComponent1 specularRoughnessAnisotropyRotation;
	// refractive index of the dielectric lobe
	GpuMorMaterialComponent1 specularIor;

	// -- transmission ----------------------------------------------------------
	GpuMorMaterialComponent1 transmissionWeight;

	// extinction of the dielectric medium
	GpuMorMaterialComponent3 transmissionColor;
	// depth into the volume that transmission color is realized.
	// If zero acts as a constant transmission tint
	GpuMorMaterialComponent1 transmissionDepth;
	// scattering coefficient of the interior medium
	GpuMorMaterialComponent3 transmissionScatter;
	// anisotropy of the henyey-greenstein phase function
	GpuMorMaterialComponent1 transmissionScatterAnisotropy;
	// linear scale of the dispersion effect
	GpuMorMaterialComponent1 transmissionDispersionScale;
	// physical abbe number of the base dielectric medium
	GpuMorMaterialComponent1 transmissionDispersionAbbeNumber;

	// -- subsurface
	GpuMorMaterialComponent1 subsurfaceWeight;
	// observed reflection color of the subsurface scattering effect
	GpuMorMaterialComponent3 subsurfaceColor;
	// length scale of the mean free path
	GpuMorMaterialComponent1 subsurfaceRadius;
	// rgb multiplier of the subsurface radius; per-channel mean free paths
	GpuMorMaterialComponent3 subsurfaceRadiusScale;
	// anistropy of the henyey-greenstein phase function for subsurface scatter
	GpuMorMaterialComponent1 subsurfaceScatterAnisotropy;

	// -- coat
	GpuMorMaterialComponent1 coatWeight;
	// square of normal-incidence transmittance of the coat layer
	GpuMorMaterialComponent3 coatColor;
	// roughness of the NDF coat lobe
	GpuMorMaterialComponent1 coatRoughness;
	// anisotropy of the NDF coat lobe
	GpuMorMaterialComponent1 coatRoughnessAnisotropy;
	// refractive index of the coat layer
	GpuMorMaterialComponent1 coatIor;
	// modulates the coat layer's darkening effect
	GpuMorMaterialComponent1 coatDarkening;

	// -- fuzz
	GpuMorMaterialComponent1 fuzzWeight;
	// reflection albedo of the fuzz layer
	GpuMorMaterialComponent3 fuzzColor;
	// roughness of the NDF fuzz lobe
	GpuMorMaterialComponent1 fuzzRoughness;

	// -- emission
	// emission luminance in nits
	GpuMorMaterialComponent1 emissionLuminance;
	// emission color multiplier
	GpuMorMaterialComponent3 emissionColor;

	// -- thin film
	GpuMorMaterialComponent1 thinFilmWeight;
	// thickness of the thin film layer in micrometers
	GpuMorMaterialComponent1 thinFilmThickness;
	// refractive index of the thin film layer
	GpuMorMaterialComponent1 thinFilmIor;

	// -- geometry
	// opacity of the geometry surface
	GpuMorMaterialComponent1 geometryOpacity;
	// if enabled makes the slab structure mirrored
	GpuMorMaterialComponent1 geometryThinWalled;

	u32 geometryNormalTexture;
	u32 geometryNormalUvTransform;
	u32 geometryTangentTexture;
	u32 geometryTangentUvTransform;
	u32 geometryCoatNormalTexture;
	u32 geometryCoatNormalUvTransform;
	u32 geometryCoatTangentTexture;
	u32 geometryCoatTangentUvTransform;

	// -- gltf extras, keep minimal
	f32 alphaCutoff;
};

// sometimes the above is a bit repetitive, just use X macros

#define MOR_MATERIAL_ALL_PARAMS_WEIGHTS(X) \
	X(GpuMorMaterialComponent1, baseWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, specularWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, transmissionWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, subsurfaceWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, coatWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, fuzzWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, thinFilmWeight, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, emissionLuminance, 0.0f, 10000.0f); \

#define MOR_MATERIAL_ALL_PARAMS_BASE(X) \
	X(GpuMorMaterialComponent3, baseColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, baseMetalness, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, baseDiffuseRoughness, 0.0f, 1.0f); \

#define MOR_MATERIAL_ALL_PARAMS_SPECULAR(X) \
	X(GpuMorMaterialComponent3, specularColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, specularRoughness, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, specularRoughnessAnisotropy, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, specularRoughnessAnisotropyRotation, 0.0f, 6.28319f); \
	X(GpuMorMaterialComponent1, specularIor, 1.0f, 3.0f); \

#define MOR_MATERIAL_ALL_PARAMS_TRANSMISSION(X) \
	X(GpuMorMaterialComponent3, transmissionColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, transmissionDepth, 0.0f, 10.0f); \
	X(GpuMorMaterialComponent3, transmissionScatter, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, transmissionScatterAnisotropy, -1.0f, 1.0f); \
	X(GpuMorMaterialComponent1, transmissionDispersionScale, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, transmissionDispersionAbbeNumber, 9.0f, 91.0f); \

#define MOR_MATERIAL_ALL_PARAMS_SUBSURFACE(X) \
	X(GpuMorMaterialComponent3, subsurfaceColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, subsurfaceRadius, 0.0f, 10.0f); \
	X(GpuMorMaterialComponent3, subsurfaceRadiusScale, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, subsurfaceScatterAnisotropy, -1.0f, 1.0f); \

#define MOR_MATERIAL_ALL_PARAMS_COAT(X) \
	X(GpuMorMaterialComponent3, coatColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, coatRoughness, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, coatRoughnessAnisotropy, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, coatIor, 1.0f, 3.0f); \
	X(GpuMorMaterialComponent1, coatDarkening, 0.0f, 1.0f); \

#define MOR_MATERIAL_ALL_PARAMS_FUZZ(X) \
	X(GpuMorMaterialComponent3, fuzzColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, fuzzRoughness, 0.0f, 1.0f); \

#define MOR_MATERIAL_ALL_PARAMS_EMISSION(X) \
	X(GpuMorMaterialComponent3, emissionColor, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, emissionLuminance, 0.0f, 10000.0f); \

#define MOR_MATERIAL_ALL_PARAMS_THIN_FILM(X) \
	X(GpuMorMaterialComponent1, thinFilmThickness, 0.0f, 2.0f); \
	X(GpuMorMaterialComponent1, thinFilmIor, 1.0f, 3.0f); \

#define MOR_MATERIAL_ALL_PARAMS_GEOMETRY(X) \
	X(GpuMorMaterialComponent1, geometryOpacity, 0.0f, 1.0f); \
	X(GpuMorMaterialComponent1, geometryThinWalled, 0.0f, 1.0f);

#define MOR_MATERIAL_ALL_PARAMS(X) \
	MOR_MATERIAL_ALL_PARAMS_WEIGHTS(X) \
	MOR_MATERIAL_ALL_PARAMS_BASE(X) \
	MOR_MATERIAL_ALL_PARAMS_SPECULAR(X) \
	MOR_MATERIAL_ALL_PARAMS_TRANSMISSION(X) \
	MOR_MATERIAL_ALL_PARAMS_SUBSURFACE(X) \
	MOR_MATERIAL_ALL_PARAMS_COAT(X) \
	MOR_MATERIAL_ALL_PARAMS_FUZZ(X) \
	MOR_MATERIAL_ALL_PARAMS_EMISSION(X) \
	MOR_MATERIAL_ALL_PARAMS_THIN_FILM(X) \
	MOR_MATERIAL_ALL_PARAMS_GEOMETRY(X)

#define MOR_MATERIAL_ALL_PARAMS_NORMAL(X) \
	X(geometryNormal, rgb); \
	X(geometryTangent, rgba); \
	X(geometryCoatNormal, rgb); \
	X(geometryCoatTangent, rgba);

// the plain u32 texture fields not covered by MOR_MATERIAL_ALL_PARAMS
#define MOR_MATERIAL_ALL_TEXTURES_NORMAL(X) \
	X(geometryNormalTexture); \
	X(geometryTangentTexture); \
	X(geometryCoatNormalTexture); \
	X(geometryCoatTangentTexture);

#define MOR_MATERIAL_ALL_PARAMS_1(X) \
	X(baseWeight); \
	X(baseMetalness); \
	X(baseDiffuseRoughness); \
	X(specularWeight); \
	X(specularRoughness); \
	X(specularRoughnessAnisotropy); \
	X(specularRoughnessAnisotropyRotation); \
	X(specularIor); \
	X(transmissionWeight); \
	X(transmissionDepth); \
	X(transmissionScatterAnisotropy); \
	X(transmissionDispersionScale); \
	X(transmissionDispersionAbbeNumber); \
	X(subsurfaceWeight); \
	X(subsurfaceRadius); \
	X(subsurfaceScatterAnisotropy); \
	X(coatWeight); \
	X(coatRoughness); \
	X(coatRoughnessAnisotropy); \
	X(coatIor); \
	X(coatDarkening); \
	X(fuzzWeight); \
	X(fuzzRoughness); \
	X(emissionLuminance); \
	X(thinFilmWeight); \
	X(thinFilmThickness); \
	X(thinFilmIor); \
	X(geometryOpacity); \
	X(geometryThinWalled);

#define MOR_MATERIAL_ALL_PARAMS_3(X) \
	X(baseColor); \
	X(specularColor); \
	X(transmissionColor); \
	X(transmissionScatter); \
	X(subsurfaceColor); \
	X(subsurfaceRadiusScale); \
	X(coatColor); \
	X(fuzzColor); \
	X(emissionColor); \

struct GpuMorInstance {
	f32m44 transform;
	u32 meshletOffset;
	u32 meshletCount;
};

#ifdef __cplusplus
using GpuMorMeshletBuffer = u64;
using GpuMorInstanceBuffer = u64;
using GpuMorPositionBuffer = u64;
using GpuMorVertexAttributeBuffer = u64;
using GpuMorMeshletVertBuffer = u64;
using GpuMorMeshletTriBuffer = u64;
using GpuMorMaterialBuffer = u64;
using GpuMorUvTransformBuffer = u64;
#else
layout(buffer_reference, scalar) buffer GpuMorMeshletBuffer {
	GpuMorMeshlet data[];
};
layout(buffer_reference, scalar) buffer GpuMorInstanceBuffer {
	GpuMorInstance data[];
};
layout(buffer_reference, scalar) buffer GpuMorPositionBuffer {
	f32v3 data[];
};
layout(buffer_reference, scalar) buffer GpuMorVertexAttributeBuffer {
	GpuMorVertexAttribute data[];
};
layout(buffer_reference, scalar) buffer GpuMorMeshletVertBuffer {
	u32 data[];
};
layout(buffer_reference, scalar) buffer GpuMorMeshletTriBuffer {
	uint8_t data[];
};
layout(buffer_reference, scalar) buffer GpuMorMaterialBuffer {
	GpuMorMaterial data[];
};
layout(buffer_reference, scalar) buffer GpuMorUvTransformBuffer {
	GpuMorUvTransform data[];
};
#endif

#endif // MOR_SHARED_H
