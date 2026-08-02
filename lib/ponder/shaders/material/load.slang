implementing material;

import types;
import util;
import model;

#include "include/inline-expression.slang"

namespace vkof {

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

public struct MaterialRef {
	public GpuMorMaterial * material;
	public GpuResolveModelIndirect * model;
	public u32 primitive;
};

// index into the model's materials
public func materialIndexFromHit(hit: SurfaceHit) -> u32 {
	let model: GpuResolveModelIndirect * = modelFromHit(hit);
	let meshletId: u32 = ((u32 *)(model.flatMeshlets))[hit.element];
	return ((GpuMorMeshlet *)(model.meshlets))[meshletId].materialIndex;
}

public func materialRefFromHit(hit: SurfaceHit) -> MaterialRef {
	let model: GpuResolveModelIndirect * = modelFromHit(hit);
	var mtrl: MaterialRef;
	mtrl.material = (
		&((GpuMorMaterial *)(model.materials))[materialIndexFromHit(hit)]
	);
	mtrl.model = model;
	mtrl.primitive = hit.element;
	return mtrl;
}

// evaluate a material component, fetching texture and applying constant
public func materialComponentEvaluate(
	mtrl: vkof::MaterialRef,
	component: GpuMorMaterialComponent3 *,
	uv: f32v2,
	lod: f32,
) -> f32v3 {
	let value: f32v3 = component.rgb;
	if (component.texture == 0u) { return value; }
	return (
		value
		* (
			implMaterialSample(
				/*mtrl=*/ mtrl,
				/*texture=*/ component.texture,
				/*uvTransform=*/ component.uvTransform,
				/*uv=*/ uv,
				/*lod=*/ lod,
			).rgb
		)
	);
}

// evaluate a material component, fetching texture and applying constant
public func materialComponentEvaluate(
	mtrl: vkof::MaterialRef,
	component: GpuMorMaterialComponent1 *,
	uv: f32v2,
	lod: f32,
) -> f32 {
	// -- fetch texel from material
	let value: f32 = InlineExpr {
		if (component.texture == 0u) { return component.r; }
		let texel: f32v4 = (
			implMaterialSample(
				/*mtrl=*/ mtrl,
				/*texture=*/ component.texture,
				/*uvTransform=*/ component.uvTransform,
				/*uv=*/ uv,
				/*lod=*/ lod
			)
		);
		// -- apply channel selection
		return (
			component.r
			* texel[u32(component.swizzle) & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK]
		);
	};

	// -- apply swizzle/invert
	let invert: bool = (
		(u32(component.swizzle) & MOR_MATERIAL_SWIZZLE_INVERT) != 0u
	);
	return invert ? 1.0f - value : value;
}

// tangent-space normal from a normal map, unnormalized; +z when none is bound
public func materialNormalEvaluate(
	mtrl: vkof::MaterialRef,
	texture: u32,
	uvTransform: u32,
	uv: f32v2,
	lod: f32,
) -> f32v3 {
	if (texture == 0u) { return f32v3(0.0f, 0.0f, 1.0f); }
	let texel: f32v4 = (
		implMaterialSample(
			/*mtrl=*/ mtrl,
			/*texture=*/ texture,
			/*uvTransform=*/ uvTransform,
			/*uv=*/ uv,
			/*lod=*/ lod
		)
	);
	return texel.xyz * 2.0f - 1.0f;
}

// return true if material is opaque with given uv and stochastic sample u
public func materialAlphaCoverage(hit: SurfaceHit, u: f32) -> bool {
	// -- fetch material from hit
	let mtrl: MaterialRef = materialRefFromHit(hit);
	// -- check alpha mode
	let alphaMode: u32 = mtrl.material.alphaMode;
	if (alphaMode == MOR_ALPHA_MODE_OPAQUE) { return true; }
	// -- fetch uv for hit
	let uv: f32v2 = modelHitUv(mtrl.model, hit);
	// -- evaluate opacity and compare to stochastic sample
	let opacity: f32 = (
		materialComponentEvaluate(
			/*mtrl=*/ mtrl,
			/*component=*/ &mtrl.material.geometryOpacity,
			/*uv=*/ uv,
			/*lod=*/ 0.0f
		)
	);
	// -- alpha test
	if (alphaMode == MOR_ALPHA_MODE_MASK) {
		return opacity >= mtrl.material.alphaCutoff;
	}
	return u < opacity;
}

public func materialEmission(
	mtrl: vkof::MaterialRef,
	uv: f32v2,
	lod: f32,
) -> vkof::ColorEmission {
	// -- evaluate emission component
	let emissionColor: f32v3 = (
		materialComponentEvaluate(
			/*mtrl=*/ mtrl,
			/*component=*/ &mtrl.material.emissionColor,
			/*uv=*/ uv,
			/*lod=*/ lod
		)
	);
	let emissionLuminance: f32 = (
		materialComponentEvaluate(
			/*mtrl=*/ mtrl,
			/*component=*/ &mtrl.material.emissionLuminance,
			/*uv=*/ uv,
			/*lod=*/ lod
		)
	);
	return (
		vkof::ColorEmission(
			/*rgb=*/ emissionColor,
			/*luminance=*/ emissionLuminance
		)
	);
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private api
// -----------------------------------------------------------------------------

// supports KHR_texture_transform
func implMaterialTransformUv(
	mtrl: vkof::MaterialRef,
	uvTransform: u32,
	uv: f32v2,
) -> f32v2 {
	if (uvTransform == 0u) { return uv; }
	let transform: GpuMorUvTransform = (
		((GpuMorUvTransform *)(mtrl.model.uvTransforms))[uvTransform]
	);
	return uv.x * transform.basisU + uv.y * transform.basisV + transform.offset;
}

// fetch a texel from a material texture, applying the uv transform
func implMaterialSample(
	mtrl: vkof::MaterialRef,
	texture: u32,
	uvTransform: u32,
	uv: f32v2,
	lod: f32,
) -> f32v4 {
	let transformedUv: f32v2 = implMaterialTransformUv(mtrl, uvTransform, uv);
	return (
		vkofTextures[NonUniformResourceIndex(texture)]
		.SampleLevel(transformedUv, lod)
	);
}
