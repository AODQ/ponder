#pragma once

#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <stage/stage.hpp>

#include "shaders/resolve_pc.h"

#include <vector>

struct SceneState {
	mor::Scene scene {};
	mor::GpuScene gpuScene {};
	mor::Buffers bufs {};
	// live-editable copy of the scene's materials, rendered from instead of
	// bufs.materials so the material inspector's edits are visible without
	// touching the gltf-sourced original (see mor::scene_material_get,
	// used to label texture slots against the unedited material)
	mor::GpuMaterials materialsOverride {};
	vkof::AccelerationStructureBlas blas {};
	vkof::AccelerationStructureTlas tlas {};
	f32v3 boundsCenter {};
	f32 boundsExtent = 0.01f;
	vkof::Buffer debugPcBuffer {};
	vkof::Buffer modelsIndirectBuffer {};
};

// (re)builds s.tlas from scratch with the given single-instance transform;
// a zero blas (the --vdb-only empty scene) builds a zero-instance tlas
// instead, so every camera ray misses straight to the environment. safe to
// call repeatedly on the same tlas handle -- vkof::tlas_build always issues
// a fresh BUILD (no srcAccelerationStructure), never an in-place update
void scene_state_rebuild_tlas(SceneState & s, f32m44 const & transform);

// loads a gltf/glb, uploads it, builds its blas + single-instance tlas, and
// binds the tlas; blas.id == 0 on failure (partial state already destroyed).
// a null path builds an empty scene instead: no blas, a zero-instance tlas
// (every camera ray misses straight to the environment), and zeroed model
// buffers nothing dereferences on a miss -- used by --vdb without a model so
// the volume renders alone
SceneState scene_state_create(char const * const path);

void scene_state_destroy(SceneState & s);

// mirrors SceneState, generalized to n gltf instances. vdb rendering still
// binds only the stage's first vdb instance's first blob --
// GpuGlobalExtended.vdb is a single global handle, there is no per-instance
// volume slot in the shader yet, so stage mode can author multiple vdb
// instances but only the first one actually renders until the pt volume
// integrator grows multi-volume support (separate, larger task)
struct StageRenderState {
	// one blas per stage.gltfMeshes entry (shared geometry), NOT per
	// instance -- every GltfInstance whose model references the same mesh
	// reuses the same blas, matching a real multi-instance tlas: many tlas
	// instances, each pointing at a (possibly shared) blas
	std::vector<vkof::AccelerationStructureBlas> blases;
	vkof::AccelerationStructureTlas tlas {};
	vkof::Buffer modelsIndirectBuffer {};
	// entry capacity modelsIndirectBuffer is currently sized for; grows
	// (reallocating, which needs a device_wait_idle) but never shrinks, so a
	// plain transform/visibility edit -- same instance count, every frame of
	// a gizmo drag -- just re-uploads into the existing buffer instead of
	// destroying and recreating it every single frame
	u32 modelsIndirectCapacity = 0u;
	vkof::Buffer debugPcBuffer {};
	f32v3 boundsCenter {};
	f32 boundsExtent = 1.0f;
	// modelDrawIndexToInstance[i] is the index into stage.gltfInstances
	// that tlas instance i (== instanceCustomIndex == the modelsIndirect
	// slot) actually came from -- invisible/unloaded instances are skipped
	// when building the tlas, so this isn't the identity mapping in
	// general. lets the right-click material probe (which only knows the
	// hit's modelDrawIndex) resolve back to a real stage::GltfInstance
	std::vector<usize> modelDrawIndexToInstance;
};

// (re)builds one blas per gltf MESH (shared across every instance whose
// model references it -- see StageRenderState::blases) and a tlas sized for
// every instance; call whenever the mesh/model/instance lists themselves
// change (stage load/new/unload, add-instance, add/duplicate/remove model)
// -- not on a plain transform edit, which only needs
// stage_render_rebuild_tlas below
void stage_render_rebuild_blases(
	StageRenderState & r, stage::Stage const & stage
);

// (re)builds the tlas instance list + the models-indirect buffer from the
// current per-instance transforms/visibility; cheap enough to call on every
// gizmo-drag or outliner edit frame (mirrors scene_state_rebuild_tlas's
// per-transform-change cadence for the single-model path). also recomputes
// the combined world bounds (union of every visible instance's local bounds
// under its own transform) for camera framing
void stage_render_rebuild_tlas(StageRenderState & r, stage::Stage const & stage);

void stage_render_destroy(StageRenderState & r);
