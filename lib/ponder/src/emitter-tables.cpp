#include <ponder/emitter-tables.hpp>

#include <cmath>

// (TODO REVIEW)

namespace {

f32 luma(f32v3 const & c) {
	return f32v3_dot(c, f32v3 { 0.2126f, 0.7152f, 0.0722f });
}

} // namespace

f32 ponder::emitter_transform_area_scale(f32m44 const & transform) {
	// column-major upper-left 3x3
	f32v3 const c0 { transform.m[0], transform.m[1], transform.m[2] };
	f32v3 const c1 { transform.m[4], transform.m[5], transform.m[6] };
	f32v3 const c2 { transform.m[8], transform.m[9], transform.m[10] };
	f32 const det = std::abs(f32v3_dot(c0, f32v3_cross(c1, c2)));
	if (det <= 0.0f) {
		return 0.0f;
	}
	return std::cbrt(det * det);
}

void ponder::emitter_meshlet_inputs_from_scene(
	mor::Scene const & scene,
	mor::GpuMaterials const & materialsOverride,
	srat::slice<mor::MeshletAreaInfo const> const meshletAreas,
	std::vector<ponder::EmitterMeshletInput> & outInputs
) {
	outInputs.clear();
	// flatIndices/flatMeshlets (what ray hits index) concatenate each
	// meshlet's triangles in meshlet order, so this running sum reconstructs
	// the same flat primitiveIndex base without a second precomputed table
	u32 primitiveIndexBase = 0u;
	for (usize i = 0u; i < meshletAreas.size(); ++i) {
		mor::MeshletAreaInfo const & info = meshletAreas[i];
		u32 const thisBase = primitiveIndexBase;
		primitiveIndexBase += info.triangleCount;
		GpuMorMaterial const mat = (
			materialsOverride.id != 0u
				? mor::scene_gpu_materials_get(materialsOverride, info.materialIndex)
				: mor::scene_material_get(scene, info.materialIndex)
		);
		if (mat.emissionLuminance.r <= 0.0f) {
			continue;
		}
		outInputs.emplace_back(ponder::EmitterMeshletInput {
			.primitiveIndexBase = thisBase,
			.triangleCount = info.triangleCount,
			.objectArea = info.area,
			.emissionColor = mat.emissionColor.rgb,
			.emissionLuminance = mat.emissionLuminance.r,
		});
	}
}

ponder::EmitterRange ponder::emitter_lights_append(
	std::vector<GpuEmitterLight> & lights,
	srat::slice<ponder::EmitterMeshletInput const> const inputs,
	f32m44 const & transform,
	u32 const modelDrawIndex,
	u32 const maxLights
) {
	ponder::EmitterRange range { (u32)lights.size(), 0u, 0u };
	f32 const areaScale = ponder::emitter_transform_area_scale(transform);
	for (usize i = 0u; i < inputs.size(); ++i) {
		ponder::EmitterMeshletInput const & in = inputs[i];
		if (in.triangleCount == 0u) {
			continue;
		}
		f32 const power = (
			in.objectArea * areaScale * luma(in.emissionColor) * in.emissionLuminance
		);
		// a non-finite power would poison the cdf for every other emitter
		if (!(power > 0.0f)) {
			continue;
		}
		if (lights.size() >= maxLights) {
			// keep counting rather than breaking, so the caller can report how
			// much of the scene the list is actually missing
			++range.clipped;
			continue;
		}
		lights.emplace_back(GpuEmitterLight {
			.kind = EMITTER_KIND_MESH,
			.modelDrawIndex = modelDrawIndex,
			.primitiveIndexBase = in.primitiveIndexBase,
			.triangleCount = in.triangleCount,
			.power = power,
			.cdf = 0.0f,
			.radiance = in.emissionColor * in.emissionLuminance,
		});
	}
	range.count = (u32)lights.size() - range.offset;
	return range;
}

u32 ponder::emitter_lights_append_environment(
	std::vector<GpuEmitterLight> & lights,
	f32 const power,
	u32 const maxLights
) {
	// !(power > 0) rather than power <= 0: a NaN integral would otherwise
	// enter the cdf and make every pick past it collapse
	if (!(power > 0.0f) || lights.size() >= maxLights) {
		return (u32)lights.size();
	}
	u32 const index = (u32)lights.size();
	lights.emplace_back(GpuEmitterLight {
		.kind = EMITTER_KIND_ENVIRONMENT,
		.modelDrawIndex = 0u,
		.primitiveIndexBase = 0u,
		// the sentinel the backward hit lookup and the triangle sampler both
		// reject on
		.triangleCount = 0u,
		.power = power,
		.cdf = 0.0f,
		// unused: the environment's radiance comes from the map itself
		.radiance = f32v3 { 0.0f, 0.0f, 0.0f },
	});
	return index;
}

f32 ponder::emitter_lights_finalize(std::vector<GpuEmitterLight> & lights) {
	f32 totalPower = 0.0f;
	for (GpuEmitterLight const & l : lights) {
		totalPower += l.power;
	}
	f32 accum = 0.0f;
	for (GpuEmitterLight & l : lights) {
		accum += l.power;
		l.cdf = totalPower > 0.0f ? accum / totalPower : 0.0f;
	}
	// float accumulation can land the last entry a hair under 1.0, which
	// would leave a sliver of the [0,1) pick range past the end of the cdf
	if (!lights.empty() && totalPower > 0.0f) {
		lights.back().cdf = 1.0f;
	}
	return totalPower;
}

// (TODO REVIEW)
