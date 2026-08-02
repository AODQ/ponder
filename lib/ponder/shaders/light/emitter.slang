implementing light;

import types;
import util;

namespace vkof {

// -----------------------------------------------------------------------------
// -- emitter selection (power cdf binary search)
// -----------------------------------------------------------------------------

public struct EmitterSelection {
	public bool valid;
	public u32 lightIndex;
	// probability density of emitter selection
	public f32 pickPdf;
};

// selects a light from the scene emitter lights based on power cdf
public func emitterSelection(
	ξ: f32,
) -> EmitterSelection
{
	let lightCount: u32 = vkof::global.sceneEmitterLightCount;
	let lights: vkof::SceneEmitterLight * = (
		vkof::global.sceneEmitterLights
	);

	// -- check if there are any lights in the scene
	if (lightCount == 0u) {
		return {
			/*valid=*/ false,
			/*lightIndex=*/ 0u,
			/*pickPdf=*/ 0.0f,
		};
	}

	// -- prepare return result
	EmitterSelection selection = {
		/*valid=*/ false,
		/*lightIndex=*/ 0u,
		/*pickPdf=*/ 0.0f,
	};

	// -- binary search for the light index
	u32 lo = 0u;
	u32 hi = lightCount - 1u;
	while (lo < hi) {
		let mid: u32 = (lo + hi) / 2u;
		if (lights[mid].cdf < ξ) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}

	// -- store the result
	let prevCdf: f32 = (lo == 0u) ? 0.0f : lights[lo - 1u].cdf;
	selection.lightIndex = lo;
	selection.pickPdf = lights[lo].cdf - prevCdf;
	selection.valid = selection.pickPdf > 0.0f;
	return selection;
}

// probability density of a light being sampled
public func emitterPickPdf(
	lightIndex: u32,
) -> f32 {
	let lightCount: u32 = vkof::global.sceneEmitterLightCount;
	let lights: vkof::SceneEmitterLight * = (
		vkof::global.sceneEmitterLights
	);

	// -- check if the light index is valid
	if (lightIndex >= lightCount) {
		return 0.0f;
	}

	// -- compute the pdf
	let prevCdf: f32 = (lightIndex == 0u) ? 0.0f : lights[lightIndex - 1u].cdf;
	return lights[lightIndex].cdf - prevCdf;
}

// index of the light
public func emitterIndexFromHit(hit: vkof::SurfaceHit) -> Optional<u32>
{
	let model: GpuResolveModelIndirect * = vkof::modelFromHit(hit);
	if (model.emitterLightCount == 0u) {
		return none;
	}
	let lights: vkof::SceneEmitterLight * = (
		vkof::global.sceneEmitterLights
	);

	// -- binary search for the light index
	var lo: u32 = model.emitterLightOffset;
	var hi: u32 = model.emitterLightOffset + model.emitterLightCount - 1u;
	while (lo < hi) {
		let mid: u32 = (lo + hi + 1u) / 2u;
		if (lights[mid].primitiveBaseIndex <= hit.element) {
			lo = mid;
		} else {
			hi = mid - 1u;
		}
	}

	// -- return light if within primitive range
	let light: vkof::SceneEmitterLight = lights[lo];
	if (
		light.kind == vkof::EmitterKind.SceneEmitter
		&& hit.element >= light.primitiveBaseIndex
		&& hit.element < light.primitiveBaseIndex + light.primitiveCount
	) {
		return Optional<u32>(lo);
	}
	return none;
}

} // vkof
