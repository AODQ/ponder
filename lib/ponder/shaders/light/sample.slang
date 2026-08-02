implementing light;

import types;
import model;
import util;

namespace vkof {

// emitter sample result
// contains the sample of an emitter and the probability of having sampled it
public struct EmitterSample {
	public vkof::SurfaceHit hit;

	// position only valid for non-environment emitters
	public f32v3 position;
	// normal only valid for non-environment emitters
	public f32v3 normal;
	// ωo only valid for environment emitters
	public f32v3 ωo;
	// if this hits a triangle mesh, the probability density is per area
	// if it hits an environment, the probability density is per solid angle
	public f32 pdf;
	public bool isDeltaDirac;
	public EmitterKind kind;
};

// sample an emitter from the scene emitter list,
// returning the sample and its probability
public func emitterSampleFromSelection(
	selection: vkof::EmitterSelection,
	ξ: f32v3,
) -> EmitterSample
{
	// ξ.x: random number for triangle selection
	// ξ.y, ξ.z: random numbers for barycentric coordinates
	if (!selection.valid) { return implEmitterSampleInvalid(); }
	let light: vkof::SceneEmitterLight = (
		vkof::global.sceneEmitterLights[selection.lightIndex]
	);
	if (light.kind == EmitterKind.Environment) {
		let environment: EnvironmentSample = (
			environmentSample(f32v2(ξ.y, ξ.z))
		);
		return {
			/*hit=*/ vkof::SurfaceHit(),
			/*position=*/ f32v3(0.0f),
			/*normal=*/ f32v3(0.0f),
			/*ωo=*/ environment.ωo,
			/*pdf=*/ selection.pickPdf * environment.pdf,
			/*isDeltaDirac=*/ false,
			/*kind=*/ EmitterKind.Environment,
		};
	}
	if (light.kind != EmitterKind.SceneEmitter || light.primitiveCount == 0u) {
		return implEmitterSampleInvalid();
	}

	// -- select a triangle from the emitter's mesh
	let primitiveLocal: u32 = (
		min(u32(ξ.x * f32(light.primitiveCount)), light.primitiveCount - 1u)
	);

	// -- select a point on the triangle
	// using sqrt will give a uniform distribution over the triangle's area
	let s: f32 = sqrt(ξ.y);
	let u: f32 = 1.0f - s;
	let v: f32 = ξ.z * s;

	// -- store the surface hit
	var hit: vkof::SurfaceHit;
	hit.kind = vkof::GeometryKind.Triangle;
	hit.instance = light.modelDrawIndex;
	hit.element = light.primitiveBaseIndex + primitiveLocal;
	hit.parameters = f32v2(u, v);

	// -- compute the surface geometry information
	let geometry: vkof::SurfaceGeometry = (
		vkof::surfaceGeometryFromParameters(hit)
	);

	// -- compute the probability density of sampling this light
	let pdf: f32 = (
		// the probability of selecting this light
		selection.pickPdf
		// the probability of selecting this triangle
		/ f32(light.primitiveCount)
		// the probability of selecting a point on the triangle
		/ geometry.elementArea
	);

	// -- return the light sample
	return {
		/*hit=*/ hit,
		/*position=*/ geometry.position,
		/*normal=*/ geometry.geometricNormal,
		/*ωo=*/ f32v3(0.0f),
		/*pdf=*/ pdf,
		/*isDeltaDirac=*/ false,
		/*kind=*/ EmitterKind.SceneEmitter,
	};
}

// sample a point on an emitter from the scene emitter list
public func emitterSample(
	ξ: f32v4,
) -> EmitterSample
{
	return (
		emitterSampleFromSelection(
			/*selection=*/ emitterSelection(ξ.x),
			/*ξ=*/ f32v3(ξ.y, ξ.z, ξ.w)
		)
	);
}

// solid-angle pdf of sampling this direction from environment map
public func emitterPdfEnvironment(dir: f32v3) -> f32 {
	return (
		emitterPickPdf(vkof::global.environmentEmitterIndex)
		* environmentPdf(dir)
	);
}

// probability density per square meter that emitterSample has of producing
// this hit.
public func emitterPdf(
	hit: vkof::SurfaceHit,
	geometry: vkof::SurfaceGeometry,
) -> f32
{
	// -- fetch the light index from the hit
	let optLightIndex: Optional<u32> = vkof::emitterIndexFromHit(hit);
	if (!optLightIndex.hasValue) { return 0.0f; }
	let lightIndex: u32 = optLightIndex.value;
	let light: vkof::SceneEmitterLight = (
		vkof::global.sceneEmitterLights[lightIndex]
	);

	// -- calculate probability density per square meter
	return (
		emitterPickPdf(lightIndex)
		/ f32(light.primitiveCount)
		/ geometry.elementArea
	);
}

} // vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implEmitterSampleInvalid() -> vkof::EmitterSample
{
	return {
		/*hit=*/ vkof::SurfaceHit(),
		/*position=*/ f32v3(0.0f),
		/*normal=*/ f32v3(0.0f),
		/*ωo=*/ f32v3(0.0f),
		/*pdf=*/ 0.0f,
		/*isDeltaDirac=*/ false,
		/*kind=*/ vkof::EmitterKind.SceneEmitter,
	};
}
