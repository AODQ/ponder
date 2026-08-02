#ifndef UTIL_MATERIAL_OPENPBR_LOAD_GLSL
#define UTIL_MATERIAL_OPENPBR_LOAD_GLSL

// -----------------------------------------------------------------------------
// -- material loading
// -----------------------------------------------------------------------------
// expands a GpuMorMaterial (mor's gltf-loaded, per-parameter-texturable
// material record) into a flat OpenPbrMaterial by sampling every bound
// texture.
//
// requires util-material-openpbr.glsl (OpenPbrMaterial, skMinRoughness)
// and the vkofTextures bindless sampler array to be declared before this
// file is included.

#include "mor/mor-shared.h"

vec4 fnSampleTextureWithDeriv(
	const uint handle,
	const vec2 uv,
	const vec2 uvDx,
	const vec2 uvDy
) {
	return textureGrad(vkofTextures[nonuniformEXT(handle)], uv, uvDx, uvDy);
}

vec4 fnSampleTextureWithLod(
	const uint handle,
	const vec2 uv,
	const uint lod
) {
	return textureLod(vkofTextures[nonuniformEXT(handle)], uv, lod);
}

// gltf KHR_texture_transform, resolved per texture slot; index 0 is the
// reserved identity and is short-circuited so a scene with no transformed
// textures at all never dereferences the uvTransforms buffer
vec2 openPbrTransformUv(
	const GpuMorUvTransformBuffer uvTransforms,
	const uint index,
	const vec2 uv
) {
	if (index == 0u) { return uv; }
	const GpuMorUvTransform xf = uvTransforms.data[index];
	return uv.x * xf.basisU + uv.y * xf.basisV + xf.offset;
}

// same transform applied to a uv screen-space derivative; the affine
// transform's translation term has no effect on a derivative, only its
// rotate-scale linear part does
vec2 openPbrTransformUvDeriv(
	const GpuMorUvTransformBuffer uvTransforms,
	const uint index,
	const vec2 duv
) {
	if (index == 0u) { return duv; }
	const GpuMorUvTransform xf = uvTransforms.data[index];
	return duv.x * xf.basisU + duv.y * xf.basisV;
}

// loads up material parameters from the specified OpenPBR material,
// sampling the textures with uv + derivatives
OpenPbrMaterial openPbrLoadMaterialDeriv(
	GpuMorMaterialBuffer buf,
	const uint idx,
	const GpuMorUvTransformBuffer uvTransforms,
	const vec2 uv,
	const vec2 uvDx,
	const vec2 uvDy,
	out vec3 geometryNormal,
	out vec4 geometryTangent,
	out vec3 geometryCoatNormal,
	out vec4 geometryCoatTangent
) {
	OpenPbrMaterial matOut;
	#define LOAD_COMPONENT1(name) \
		if (buf.data[idx]. name .texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name .uvTransform, uv) \
			); \
			const vec2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name .uvTransform, uvDx) \
			); \
			const vec2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name .uvTransform, uvDy) \
			); \
			matOut. name .r = (\
				buf.data[idx]. name .r \
				* fnSampleTextureWithDeriv(buf.data[idx]. name .texture, xfUv, xfUvDx, xfUvDy)[ \
					buf.data[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK \
				] \
			); \
		} else { \
			matOut. name .r = buf.data[idx]. name .r; \
		} \
		if ((buf.data[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_INVERT) != 0) { \
			matOut. name .r = 1.0f - matOut. name .r; \
		}
	MOR_MATERIAL_ALL_PARAMS_1(LOAD_COMPONENT1)
	#undef LOAD_COMPONENT1

	matOut.specularRoughness.r = max(matOut.specularRoughness.r, skMinRoughness);
	matOut.coatRoughness.r = max(matOut.coatRoughness.r, skMinRoughness);
	matOut.fuzzRoughness.r = max(matOut.fuzzRoughness.r, skMinRoughness);

	#define LOAD_COMPONENT_NORMAL(name, swizzle, identity) \
		if (buf.data[idx]. name##Texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name##UvTransform, uv) \
			); \
			const vec2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name##UvTransform, uvDx) \
			); \
			const vec2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name##UvTransform, uvDy) \
			); \
			name = ( \
				( \
					2.0f \
					* ( \
						fnSampleTextureWithDeriv( \
							buf.data[idx].name##Texture, xfUv, xfUvDx, xfUvDy \
						). swizzle \
					) \
				) \
				- vec4(1.0f). swizzle \
			); \
		} else { \
			name = (identity) .swizzle; \
		}
	MOR_MATERIAL_ALL_PARAMS_NORMAL(LOAD_COMPONENT_NORMAL)
	#undef LOAD_COMPONENT_NORMAL

	#define LOAD_COMPONENT3(name) \
		if (buf.data[idx]. name .texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name .uvTransform, uv) \
			); \
			const vec2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name .uvTransform, uvDx) \
			); \
			const vec2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf.data[idx]. name .uvTransform, uvDy) \
			); \
			matOut. name .rgb = ( \
				buf.data[idx]. name .rgb \
				* fnSampleTextureWithDeriv(buf.data[idx]. name .texture, xfUv, xfUvDx, xfUvDy).rgb \
			); \
		} else { \
			matOut. name .rgb = buf.data[idx]. name .rgb; \
		}
	MOR_MATERIAL_ALL_PARAMS_3(LOAD_COMPONENT3)
	#undef LOAD_COMPONENT3

	matOut.alphaCutoff = buf.data[idx].alphaCutoff;
	// (TODO REVIEW)
	matOut.alphaMode = buf.data[idx].alphaMode;
	// (TODO REVIEW)

	return matOut;
}

// this loads up all the material parameters from the specified OpenPBR material
// samples textures using lod
//
// see openPbrLoadMaterialDeriv above for why this takes (buf, idx) instead
// of an already-materialized GpuMorMaterial value
OpenPbrMaterial openPbrLoadMaterialLod(
	GpuMorMaterialBuffer buf,
	const uint idx,
	const GpuMorUvTransformBuffer uvTransforms,
	const vec2 uv,
	const uint lod,
	out vec3 geometryNormal,
	out vec4 geometryTangent,
	out vec3 geometryCoatNormal,
	out vec4 geometryCoatTangent
) {
	OpenPbrMaterial matOut;
	#define LOAD_COMPONENT1(name) \
		if (buf.data[idx]. name .texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name .uvTransform, uv) \
			); \
			matOut. name .r = (\
				buf.data[idx]. name .r \
				* fnSampleTextureWithLod(buf.data[idx]. name .texture, xfUv, lod)[ \
					buf.data[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK \
				] \
			); \
		} else { \
			matOut. name .r = buf.data[idx]. name .r; \
		} \
		if ((buf.data[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_INVERT) != 0) { \
			matOut. name .r = 1.0f - matOut. name .r; \
		}
	MOR_MATERIAL_ALL_PARAMS_1(LOAD_COMPONENT1)
	#undef LOAD_COMPONENT1

	matOut.specularRoughness.r = max(matOut.specularRoughness.r, skMinRoughness);
	matOut.coatRoughness.r = max(matOut.coatRoughness.r, skMinRoughness);
	matOut.fuzzRoughness.r = max(matOut.fuzzRoughness.r, skMinRoughness);

	#define LOAD_COMPONENT_NORMAL(name, swizzle, identity) \
		if (buf.data[idx]. name##Texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name##UvTransform, uv) \
			); \
			name = ( \
				( \
					2.0f \
					* ( \
						fnSampleTextureWithLod( \
							buf.data[idx].name##Texture, xfUv, lod \
						). swizzle \
					) \
				) \
				- vec4(1.0f). swizzle \
			); \
		} else { \
			name = (identity) .swizzle; \
		}
	MOR_MATERIAL_ALL_PARAMS_NORMAL(LOAD_COMPONENT_NORMAL)
	#undef LOAD_COMPONENT_NORMAL

	#define LOAD_COMPONENT3(name) \
		if (buf.data[idx]. name .texture != 0) { \
			const vec2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf.data[idx]. name .uvTransform, uv) \
			); \
			matOut. name .rgb = ( \
				buf.data[idx]. name .rgb \
				* fnSampleTextureWithLod(buf.data[idx]. name .texture, xfUv, lod).rgb \
			); \
		} else { \
			matOut. name .rgb = buf.data[idx]. name .rgb; \
		}
	MOR_MATERIAL_ALL_PARAMS_3(LOAD_COMPONENT3)
	#undef LOAD_COMPONENT3
	matOut.alphaCutoff = buf.data[idx].alphaCutoff;
	// (TODO REVIEW)
	matOut.alphaMode = buf.data[idx].alphaMode;
	// (TODO REVIEW)
	return matOut;
}

#endif // UTIL_MATERIAL_OPENPBR_LOAD_GLSL
