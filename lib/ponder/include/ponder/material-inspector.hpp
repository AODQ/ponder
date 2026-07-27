#pragma once

#include <srat/core-types.hpp>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>

namespace ponder {

// materialsOverride starts as an exact copy of the scene's materials (see
// mor::scene_gpu_materials_create), so editing it never touches the
// gltf-sourced mor::Scene -- the scene's own material is only used to label
// texture slots against the material's pre-edit state
bool material_inspector_draw(
	mor::Scene const & scene,
	mor::GpuMaterials const & materialsOverride,
	i32 const materialIndex
);

} // namespace ponder
