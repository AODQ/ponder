#pragma once
__exported import "util-types";
__exported import "util-random";
__exported import "util-emitter";
__exported import "util-mor-shared";

struct PC {
	u32v4 rootPad[8];
	uint64_t outVa;
	uint64_t lightsVa;
	uint64_t modelsVa;
	f32v3 from;
	uint lightCount;
	uint sampleCount;
	uint seedSalt;
	// expected-side integration resolution; a triangle silhouette cutting
	// across a bin is a discontinuity inside the integrand, so scenes with
	// small emitters need a finer subgrid than large ones
	uint subGrid;
	// -- direct-lighting estimator fields; glsl allows exactly one
	// push_constant block, so every emitter test shares this one layout
	uint samplesPerThread;
	uint threadCount;
	uint kullaContyHandle;
	f32v3 shadingNormal;
	f32 baseWeight;
	f32 specularWeight;
	f32 specularRoughness;
	f32 metalness;
};

[[vk::push_constant]] PC pc;

// shared prelude for the emitter sampler's tier-1 shaders. keeps the include
// order and the push-constant layout identical across the roundtrip,
// histogram and expected dispatches so one C++ push struct drives all three.


// RESOLVE_NO_PC: pulls GpuResolveModelIndirect + u32 * out of
// resolve_pc.h without its push_constant block, since these shaders declare
// their own
#define RESOLVE_NO_PC





// moller-trumbore, two-sided; the expected-side integration needs to know
// which emitter triangles a given direction passes through
public func emitterTestRayTriangle(
	origin: f32v3,
	dir: f32v3,
	tri: EmitterTriangle,
	out outDistance: f32,
) -> bool {
	outDistance = 0.0f;
	const f32v3 e1 = tri.p1 - tri.p0;
	const f32v3 e2 = tri.p2 - tri.p0;
	const f32v3 p = cross(dir, e2);
	const f32 det = dot(e1, p);
	if (abs(det) < 1e-12f) {
		return false;
	}
	const f32 invDet = 1.0f / det;
	const f32v3 t = origin - tri.p0;
	const f32 u = dot(t, p) * invDet;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}
	const f32v3 q = cross(t, e1);
	const f32 v = dot(dir, q) * invDet;
	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}
	const f32 dist = dot(e2, q) * invDet;
	if (dist <= 0.0f) {
		return false;
	}
	outDistance = dist;
	return true;
}

// total emitter-strategy density for a direction, summed over every emitter
// triangle the ray passes through -- two triangles behind one another are two
// distinct sample points, so their densities add
public func emitterTestDirectionPdf(
	from: f32v3,
	dir: f32v3,
) -> f32 {
	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	f32 total = 0.0f;
	for (u32 li = 0u; li < pc.lightCount; ++li) {
		const GpuEmitterLight light = lights[li];
		const GpuResolveModelIndirect model = models[light.modelDrawIndex];
		for (u32 t = 0u; t < light.triangleCount; ++t) {
			const u32 primitiveIndex = light.primitiveIndexBase + t;
			const EmitterTriangle tri = utilEmitterTriangle(model, primitiveIndex);
			f32 dist;
			if (!emitterTestRayTriangle(from, dir, tri, dist)) {
				continue;
			}
			total += (
				utilEmitterPdf(
					lights, li, model, primitiveIndex, from, dir, dist
				)
			);
		}
	}
	return total;
}

// nearest emitter triangle along a ray; the bsdf-sampled estimator needs to
// know which triangle it landed on to look up the competing density
public func emitterTestNearestHit(
	origin: f32v3,
	dir: f32v3,
	out outLightIndex: u32,
	out outPrimitiveIndex: u32,
	out outDistance: f32,
) -> bool {
	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	outLightIndex = 0u;
	outPrimitiveIndex = 0u;
	outDistance = 0.0f;
	bool found = false;
	f32 nearest = 1e30f;
	for (u32 li = 0u; li < pc.lightCount; ++li) {
		const GpuEmitterLight light = lights[li];
		const GpuResolveModelIndirect model = models[light.modelDrawIndex];
		for (u32 t = 0u; t < light.triangleCount; ++t) {
			const u32 primitiveIndex = light.primitiveIndexBase + t;
			const EmitterTriangle tri = utilEmitterTriangle(model, primitiveIndex);
			f32 dist;
			if (!emitterTestRayTriangle(origin, dir, tri, dist)) {
				continue;
			}
			if (dist < nearest) {
				nearest = dist;
				outLightIndex = li;
				outPrimitiveIndex = primitiveIndex;
				outDistance = dist;
				found = true;
			}
		}
	}
	return found;
}

