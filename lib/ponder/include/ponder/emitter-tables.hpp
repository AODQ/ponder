#pragma once

#include <mor/mor.hpp>
#include <srat/core-array.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include "emitter-shared.h"

#include <vector>

namespace ponder {

// (TODO REVIEW)

// per-meshlet input to the emitter list build, decoupled from mor so the
// weighting/cdf math is testable without a loaded gltf
struct EmitterMeshletInput {
	// flat primitiveIndex of this meshlet's first triangle
	u32 primitiveIndexBase;
	u32 triangleCount;
	// object-space surface area, before the instance transform
	f32 objectArea;
	f32v3 emissionColor;
	f32 emissionLuminance;
};

// where one instance's emitters landed in the shared list; stamped onto that
// instance's GpuResolveModelIndirect entry so the backward (bsdf-hit) lookup
// only scans its own model's emitters
struct EmitterRange {
	u32 offset;
	u32 count;
	// emissive meshlets dropped because the list was already at maxLights. a
	// clipped emitter is still unbiased -- it just falls to the bsdf technique
	// alone -- so this is a variance warning, not an error
	u32 clipped;
};

// collects the emissive meshlets of one loaded scene. meshletAreas must be
// the scene's own mor::scene_meshlet_area_info (cached by the caller --
// recomputing it walks every triangle); materialsOverride.id == 0 falls back
// to the scene's unedited materials
void emitter_meshlet_inputs_from_scene(
	mor::Scene const & scene,
	mor::GpuMaterials const & materialsOverride,
	srat::slice<mor::MeshletAreaInfo const> const meshletAreas,
	std::vector<EmitterMeshletInput> & outInputs
);

// appends one GpuEmitterLight per input whose power is positive, stopping at
// maxLights. cdf is left zeroed until emitter_lights_finalize
[[nodiscard]] EmitterRange emitter_lights_append(
	std::vector<GpuEmitterLight> & lights,
	srat::slice<EmitterMeshletInput const> const inputs,
	f32m44 const & transform,
	u32 const modelDrawIndex,
	u32 const maxLights
);

// appends the environment map as a non-mesh entry so one power-weighted pick
// covers both techniques. power is EnvironmentTables::luminanceIntegral times
// the frame's envIntensity -- an irradiance-scale number against a mesh
// emitter's area*radiance, which is a heuristic and steers the pick only.
// returns the entry's index, or `lights.size()` unchanged when the power is
// not positive (an all-black map contributes nothing, so sampling it is pure
// waste)
[[nodiscard]] u32 emitter_lights_append_environment(
	std::vector<GpuEmitterLight> & lights,
	f32 const power,
	u32 const maxLights
);

// normalizes the list into a monotonic cdf; returns the total power. a list
// with no power leaves every cdf at 0, which fnPickEmitterLight reads as
// "no emitters" via the same totalPower <= 0 check
f32 emitter_lights_finalize(std::vector<GpuEmitterLight> & lights);

// the factor an instance transform scales object-space area by. exact for a
// uniform scale; |det|^(2/3) is an isotropic estimate otherwise, which is
// fine because power only steers the pick
[[nodiscard]] f32 emitter_transform_area_scale(f32m44 const & transform);

// (TODO REVIEW)

} // namespace ponder
