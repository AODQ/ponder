module "util-material-openpbr-load";
__exported import "util-types";
__exported import "util-mor-shared";
#include "util-mor-shared-macros.slang"
__exported import "util-material-openpbr-shared";
__exported import "util-material-openpbr";

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


public func fnSampleTextureWithDeriv(
	handle: u32,
	uv: f32v2,
	uvDx: f32v2,
	uvDy: f32v2,
) -> f32v4 {
	return textureGrad(vkofTextures[NonUniformResourceIndex(handle)], uv, uvDx, uvDy);
}

public func fnSampleTextureWithLod(
	handle: u32,
	uv: f32v2,
	lod: u32,
) -> f32v4 {
	return textureLod(vkofTextures[NonUniformResourceIndex(handle)], uv, lod);
}

// gltf KHR_texture_transform, resolved per texture slot; index 0 is the
// reserved identity and is short-circuited so a scene with no transformed
// textures at all never dereferences the uvTransforms buffer
public func openPbrTransformUv(
	uvTransforms: GpuMorUvTransform *,
	index: u32,
	uv: f32v2,
) -> f32v2 {
	if (index == 0u) { return uv; }
	const GpuMorUvTransform xf = uvTransforms[index];
	return uv.x * xf.basisU + uv.y * xf.basisV + xf.offset;
}

// same transform applied to a uv screen-space derivative; the affine
// transform's translation term has no effect on a derivative, only its
// rotate-scale linear part does
public func openPbrTransformUvDeriv(
	uvTransforms: GpuMorUvTransform *,
	index: u32,
	duv: f32v2,
) -> f32v2 {
	if (index == 0u) { return duv; }
	const GpuMorUvTransform xf = uvTransforms[index];
	return duv.x * xf.basisU + duv.y * xf.basisV;
}

// loads up material parameters from the specified OpenPBR material,
// sampling the textures with uv + derivatives
public func openPbrLoadMaterialDeriv(
	buf: GpuMorMaterial *,
	idx: u32,
	uvTransforms: GpuMorUvTransform *,
	uv: f32v2,
	uvDx: f32v2,
	uvDy: f32v2,
	out geometryNormal: f32v3,
	out geometryTangent: f32v4,
	out geometryCoatNormal: f32v3,
	out geometryCoatTangent: f32v4,
) -> OpenPbrMaterial {
	// glsl parameters are mutable copies; slang's are not
	var buf = buf;
	OpenPbrMaterial matOut;
	#define LOAD_COMPONENT1(name) \
		if (buf[idx]. name .texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name .uvTransform, uv) \
			); \
			const f32v2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name .uvTransform, uvDx) \
			); \
			const f32v2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name .uvTransform, uvDy) \
			); \
			matOut. name .r = (\
				buf[idx]. name .r \
				* fnSampleTextureWithDeriv(buf[idx]. name .texture, xfUv, xfUvDx, xfUvDy)[ \
					buf[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK \
				] \
			); \
		} else { \
			matOut. name .r = buf[idx]. name .r; \
		} \
		if ((buf[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_INVERT) != 0) { \
			matOut. name .r = 1.0f - matOut. name .r; \
		}
	MOR_MATERIAL_ALL_PARAMS_1(LOAD_COMPONENT1)
	#undef LOAD_COMPONENT1

	matOut.specularRoughness.r = max(matOut.specularRoughness.r, skMinRoughness);
	matOut.coatRoughness.r = max(matOut.coatRoughness.r, skMinRoughness);
	matOut.fuzzRoughness.r = max(matOut.fuzzRoughness.r, skMinRoughness);

	#define LOAD_COMPONENT_NORMAL(name, swizzle, identity) \
		if (buf[idx]. name##Texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name##UvTransform, uv) \
			); \
			const f32v2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name##UvTransform, uvDx) \
			); \
			const f32v2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name##UvTransform, uvDy) \
			); \
			name = ( \
				( \
					2.0f \
					* ( \
						fnSampleTextureWithDeriv( \
							buf[idx].name##Texture, xfUv, xfUvDx, xfUvDy \
						). swizzle \
					) \
				) \
				- f32v4(1.0f). swizzle \
			); \
		} else { \
			name = (identity) .swizzle; \
		}
	MOR_MATERIAL_ALL_PARAMS_NORMAL(LOAD_COMPONENT_NORMAL)
	#undef LOAD_COMPONENT_NORMAL

	#define LOAD_COMPONENT3(name) \
		if (buf[idx]. name .texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name .uvTransform, uv) \
			); \
			const f32v2 xfUvDx = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name .uvTransform, uvDx) \
			); \
			const f32v2 xfUvDy = ( \
				openPbrTransformUvDeriv(uvTransforms, buf[idx]. name .uvTransform, uvDy) \
			); \
			matOut. name .rgb = ( \
				buf[idx]. name .rgb \
				* fnSampleTextureWithDeriv(buf[idx]. name .texture, xfUv, xfUvDx, xfUvDy).rgb \
			); \
		} else { \
			matOut. name .rgb = buf[idx]. name .rgb; \
		}
	MOR_MATERIAL_ALL_PARAMS_3(LOAD_COMPONENT3)
	#undef LOAD_COMPONENT3

	matOut.alphaCutoff = buf[idx].alphaCutoff;
	// (TODO REVIEW)
	matOut.alphaMode = buf[idx].alphaMode;
	// (TODO REVIEW)

	return matOut;
}

// this loads up all the material parameters from the specified OpenPBR material
// samples textures using lod
//
// see openPbrLoadMaterialDeriv above for why this takes (buf, idx) instead
// of an already-materialized GpuMorMaterial value
public func openPbrLoadMaterialLod(
	buf: GpuMorMaterial *,
	idx: u32,
	uvTransforms: GpuMorUvTransform *,
	uv: f32v2,
	lod: u32,
	out geometryNormal: f32v3,
	out geometryTangent: f32v4,
	out geometryCoatNormal: f32v3,
	out geometryCoatTangent: f32v4,
) -> OpenPbrMaterial {
	// glsl parameters are mutable copies; slang's are not
	var buf = buf;
	OpenPbrMaterial matOut;
	#define LOAD_COMPONENT1(name) \
		if (buf[idx]. name .texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name .uvTransform, uv) \
			); \
			matOut. name .r = (\
				buf[idx]. name .r \
				* fnSampleTextureWithLod(buf[idx]. name .texture, xfUv, lod)[ \
					buf[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK \
				] \
			); \
		} else { \
			matOut. name .r = buf[idx]. name .r; \
		} \
		if ((buf[idx]. name .swizzle & MOR_MATERIAL_SWIZZLE_INVERT) != 0) { \
			matOut. name .r = 1.0f - matOut. name .r; \
		}
	MOR_MATERIAL_ALL_PARAMS_1(LOAD_COMPONENT1)
	#undef LOAD_COMPONENT1

	matOut.specularRoughness.r = max(matOut.specularRoughness.r, skMinRoughness);
	matOut.coatRoughness.r = max(matOut.coatRoughness.r, skMinRoughness);
	matOut.fuzzRoughness.r = max(matOut.fuzzRoughness.r, skMinRoughness);

	#define LOAD_COMPONENT_NORMAL(name, swizzle, identity) \
		if (buf[idx]. name##Texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name##UvTransform, uv) \
			); \
			name = ( \
				( \
					2.0f \
					* ( \
						fnSampleTextureWithLod( \
							buf[idx].name##Texture, xfUv, lod \
						). swizzle \
					) \
				) \
				- f32v4(1.0f). swizzle \
			); \
		} else { \
			name = (identity) .swizzle; \
		}
	MOR_MATERIAL_ALL_PARAMS_NORMAL(LOAD_COMPONENT_NORMAL)
	#undef LOAD_COMPONENT_NORMAL

	#define LOAD_COMPONENT3(name) \
		if (buf[idx]. name .texture != 0) { \
			const f32v2 xfUv = ( \
				openPbrTransformUv(uvTransforms, buf[idx]. name .uvTransform, uv) \
			); \
			matOut. name .rgb = ( \
				buf[idx]. name .rgb \
				* fnSampleTextureWithLod(buf[idx]. name .texture, xfUv, lod).rgb \
			); \
		} else { \
			matOut. name .rgb = buf[idx]. name .rgb; \
		}
	MOR_MATERIAL_ALL_PARAMS_3(LOAD_COMPONENT3)
	#undef LOAD_COMPONENT3
	matOut.alphaCutoff = buf[idx].alphaCutoff;
	// (TODO REVIEW)
	matOut.alphaMode = buf[idx].alphaMode;
	// (TODO REVIEW)
	return matOut;
}

