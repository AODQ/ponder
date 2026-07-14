#pragma once

#include <mor/mor-shared.h>
#include <srat/core-types.hpp>
#include <string>
#include <vector>

namespace mor {

struct Scene { u64 id; };
struct GpuScene { u64 id; };
struct GpuMaterials { u64 id; };

struct Buffers {
	// TODO these are VAs so maybe they shold be VA(...) types
	u64 meshlets;
	u64 materials;
	u64 uvTransforms;
	u64 textures;
	u64 instances;
	u64 positions;
	u64 attributes;
	u64 meshletVerts;
	u64 meshletTris;
	u64 flatIndices;
	u64 flatMeshlets;
	u32 vertexCount;
	u32 triangleCount;
};

Scene scene_create();
void scene_destroy(Scene const & scene);
void scene_load_gltf(Scene const & scene, char const * const path);

u32 scene_instance_count(Scene const & scene);
u32 scene_meshlet_count(Scene const & scene);
u32 scene_vertex_count(Scene const & scene);
void scene_bounds(Scene const & scene, f32v3 & outMin, f32v3 & outMax);
// false if any material in the scene uses a gltf blend mode other than
// opaque (mask or blend); the blas can only be marked opaque otherwise
[[nodiscard]] bool scene_is_fully_opaque(Scene const & scene);

void scene_imgui_debug(Scene const & scene);
void scene_imgui_textures(Scene const & scene);

GpuScene scene_gpu_upload(Scene const & scene);
void scene_gpu_destroy(GpuScene const & scene);

Buffers scene_gpu_buffers(GpuScene const & scene);
u32 scene_gpu_meshlet_count(GpuScene const & scene);

void scene_set_anisotropy(
	Scene const & scene,
	GpuScene const & gpuScene,
	f32 const & anisotropy
);

[[nodiscard]] u32 scene_material_count(Scene const & scene);
[[nodiscard]] std::string scene_material_name(Scene const & scene, u32 index);
[[nodiscard]] GpuMorMaterial scene_material_get(Scene const & scene, u32 index);

struct MeshletAreaInfo {
	u32 materialIndex;
	// object-space surface area, before any instance transform is applied
	f32 area;
	u32 triangleCount;
};
// one entry per meshlet, indexed the same as the meshlet buffer
[[nodiscard]] std::vector<MeshletAreaInfo> scene_meshlet_area_info(
	Scene const & scene
);

[[nodiscard]] GpuMaterials scene_gpu_materials_create(Scene const & scene);
void scene_gpu_materials_destroy(GpuMaterials const & mats);
[[nodiscard]] u64 scene_gpu_materials_va(GpuMaterials const & mats);
[[nodiscard]] u32 scene_gpu_materials_count(GpuMaterials const & mats);
// reads back the current cpu mirror, including any override applied via
// scene_material_override_scalars
[[nodiscard]] GpuMorMaterial scene_gpu_materials_get(
	GpuMaterials const & mats, u32 index
);

// updates the cpu mirror only; call scene_gpu_materials_upload to flush
void scene_material_override_scalars(
	GpuMaterials const & mats, u32 index, GpuMorMaterial const & scalars
);
// updates the cpu mirror only; call scene_gpu_materials_upload to flush
void scene_gpu_materials_sync_textures(
	Scene const & scene, GpuMaterials const & mats
);
void scene_gpu_materials_upload(GpuMaterials const & mats);
// rewrites texture handles invalidated by the last scene_set_anisotropy call
void scene_remap_material_textures(Scene const & scene, GpuMorMaterial & mat);

void sampler_cache_destroy();

} // namespace mor
