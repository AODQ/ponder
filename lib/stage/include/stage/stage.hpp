#pragma once

#include <mor/mor.hpp>
#include <ponder/vdb.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <string>
#include <vector>

namespace stage {

struct Stage;

struct Transform {
	f32v3 position { 0.0f, 0.0f, 0.0f };
	f32quat rotation = f32quat_identity();
	f32v3 scale { 1.0f, 1.0f, 1.0f };
};

[[nodiscard]] f32m44 transform_to_m44(Transform const & t);

// inverse of transform_to_m44(t); computed analytically from the TRS
// components (inverse-scale * inverse-rotation * inverse-translate) rather
// than a general 4x4 inverse, since the composition is known. zero scale on
// an axis maps to a zero inverse-scale on that axis rather than dividing by
// zero
[[nodiscard]] f32m44 transform_to_inverse_m44(Transform const & t);

// -----------------------------------------------------------------------------
// -- material overrides
// -----------------------------------------------------------------------------

// scalar/color only; texturePath is reserved for when mor grows a public
// external-texture-load api, unused today
struct MaterialFieldOverride {
	// must name a MOR_MATERIAL_ALL_PARAMS field, e.g. "baseColor",
	// "specularRoughness"
	std::string field;
	// used when field is a GpuMorMaterialComponent3
	f32v3 colorValue {};
	// used when field is a GpuMorMaterialComponent1
	f32 scalarValue = 0.0f;
	// reserved, unused
	std::string texturePath;
};

struct MaterialOverride {
	// resolved against mor::scene_material_name
	std::string material;
	std::vector<MaterialFieldOverride> fields;
};

// -----------------------------------------------------------------------------
// -- gltf mesh / model / instance
//
// three levels, mirroring how a real editor separates "geometry" from
// "look" from "placement":
//   GltfMesh: raw geometry loaded from one gltf file. shared -- loading the
//     same path twice (via stage_add_gltf_model) reuses the same GltfMesh
//     rather than loading + uploading the geometry again, so instances of
//     the same file share one BLAS.
//   GltfModel: a mesh plus its OWN independent, editable material set.
//     stage_duplicate_gltf_model makes a second model referencing the same
//     meshIndex (same geometry/BLAS) with an independent copy of the
//     materials to diverge from -- this is how you get two differently-
//     colored variants of one mesh without duplicating geometry.
//   GltfInstance: one placement of a model in the scene (transform +
//     visibility only). every instance referencing the same modelIndex
//     shares that model's material by construction; there is no per-
//     instance material override -- fork a new model (duplicate) instead.
// -----------------------------------------------------------------------------

struct GltfMesh {
	// resolved absolute filesystem path; stage_load rebases the json-authored
	// (possibly relative) string against Stage::sourceDirectory once at load
	// time, and stage_save re-relativizes it against the save directory
	std::string path;

	// populated by stage_load / stage_add_gltf_model; app owns any further
	// gpu/accel-structure work (one BLAS per GltfMesh, shared by every
	// GltfInstance whose model references it)
	mor::Scene scene {};
	mor::GpuScene gpuScene {};
};

struct GltfModel {
	std::string name;
	// indexes Stage::gltfMeshes
	u32 meshIndex = 0;
	std::vector<MaterialOverride> materialOverrides;

	// populated by stage_load / stage_add_gltf_model /
	// stage_duplicate_gltf_model: mor::scene_gpu_materials_create(mesh.scene)
	// with materialOverrides applied + uploaded. independent per model, even
	// when meshIndex is shared with another model
	mor::GpuMaterials materials {};
};

struct GltfInstance {
	std::string name;
	// indexes Stage::gltfModels
	u32 modelIndex = 0;
	Transform transform;
	bool visible = true;
};

// world-space aabb of instance's mesh under its current transform (mor's
// object-space bounds composed with transform_to_m44). resolves
// instance.modelIndex -> gltfModels -> meshIndex -> gltfMeshes internally.
// false (out params untouched) if modelIndex/meshIndex don't resolve to a
// loaded mesh
[[nodiscard]] bool gltf_instance_world_bounds(
	Stage const & stage,
	GltfInstance const & instance,
	f32v3 & outMin,
	f32v3 & outMax
);

// -----------------------------------------------------------------------------
// -- vdb instance
// -----------------------------------------------------------------------------

// a "blob" is one named grid in the vdb file
struct VdbBlob {
	// e.g. "density", "temperature"
	std::string gridName;
	// rgb absorption/scattering coefficients (pbrt's HomogeneousMedium/
	// GridMedium sigma_a/sigma_s convention -- see lib/ponder/VDB_REFERENCES.md);
	// mirrors GpuGlobalExtended.vdbSigmaAbsorption/vdbSigmaScattering.
	// sigma_t = sigmaAbsorption + sigmaScattering multiplies vdb.densityMax
	// for the delta/ratio-tracking majorant; single-scattering albedo is
	// sigmaScattering/sigma_t, derived rather than stored separately.
	// defaults (absorption 0, scattering 1) reproduce the old scalar
	// sigma=1/albedo=1 pure-scattering default exactly
	f32v3 sigmaAbsorption { 0.0f, 0.0f, 0.0f };
	f32v3 sigmaScattering { 1.0f, 1.0f, 1.0f };
	// water droplet diameter in micrometers, feeds the fog phase function
	// (util-vdb.glsl's utilVdbPhaseFogDefaultParams); meaningful range is
	// roughly 5-50. mirrors GpuGlobalExtended.vdbDropletDiameter
	f32 dropletDiameter = 25.0f;
	// manual Henyey-Greenstein anisotropy override for non-water media
	// (smoke soot, dust), bypassing the droplet-diameter fog fit above;
	// -2 (skUtilVdbPhaseAnisotropyOverrideOff) means disabled. mirrors
	// GpuGlobalExtended.vdbPhaseAnisotropyOverride
	f32 phaseAnisotropyOverride = -2.0f;
	// only meaningful when gridName == "temperature": scales the raw grid
	// value into kelvin before blackbody evaluation (see
	// GpuGlobalExtended.vdbTemperatureScale for why -- raw grid values are
	// essentially never literal kelvin). mirrors
	// GpuGlobalExtended.vdbTemperatureScale
	f32 temperatureScale = 1.0f;
	// only meaningful when gridName == "temperature": artist multiplier on
	// utilBlackbodyRadiance's output (see lib/ponder/shaders/util-blackbody.glsl
	// for why blackbody radiance isn't normalized to absolute SI units, and
	// so needs one). mirrors GpuGlobalExtended.vdbEmissionScale. 0 disables
	// emission entirely, matching every other "off" sentinel in this struct
	// being a value the field can legitimately never need in practice
	f32 emissionScale = 1.0f;

	// populated by stage_load
	ponder::Vdb vdb {};
};

struct VdbInstance {
	std::string name;
	// resolved absolute filesystem path; see GltfInstance::path
	std::string path;
	Transform transform;
	bool visible = true;
	std::vector<VdbBlob> blobs;
};

// world-space aabb union of every blob's grid (nanovdb worldBBox) under the
// instance's current transform. false (out params untouched) if no blob has
// a loaded grid
[[nodiscard]] bool vdb_instance_world_bounds(
	VdbInstance const & instance, f32v3 & outMin, f32v3 & outMax
);

// -----------------------------------------------------------------------------
// -- alembic instance
// -----------------------------------------------------------------------------

// format reserved; not loaded until an alembic loader exists
struct AlembicInstance {
	std::string name;
	// resolved absolute filesystem path; see GltfInstance::path
	std::string path;
	Transform transform;
	bool visible = true;
	// path within the archive; empty means the whole archive
	std::string objectPath;
	// sample time, for future animation support
	f32 time = 0.0f;
};

// -----------------------------------------------------------------------------
// -- nested stage instance (composition)
// -----------------------------------------------------------------------------

// references another stage json as a sub-scene, placed under this transform.
// the referenced Stage's own instances/lights are loaded recursively by
// stage_load; the parent's transform composes on top of each of the child's
// instance transforms (see transform_to_m44 usage in stage_load)
struct StageInstance {
	std::string name;
	// resolved absolute filesystem path; see GltfInstance::path
	std::string path;
	Transform transform;
	bool visible = true;

	// populated by stage_load; owns everything the referenced file loaded
	Stage * child = nullptr;
};

// -----------------------------------------------------------------------------
// -- camera / lights / environment
// -----------------------------------------------------------------------------

struct Camera {
	std::string name;
	f32v3 position { 0.0f, 1.0f, 3.0f };
	f32v3 target {};
	f32v3 up { 0.0f, 1.0f, 0.0f };
	f32 fovYDegrees = 45.0f;
	f32 near = 0.01f;
	f32 far = 1000.0f;
};

struct LightInstanceQuad {
	std::string name;
	// the quad lies in the local xy plane, facing +z
	Transform transform;
	f32 width = 1.0f;
	f32 height = 1.0f;
	f32v3 radiance { 1.0f, 1.0f, 1.0f };
	bool twoSided = false;
};

// mirrors ViewerArgs::envMode 1:1 (furnace/checkerboard/black/hdrmap)
enum class EnvironmentMode { Furnace, Checkerboard, Black, Hdri };

struct Environment {
	EnvironmentMode mode = EnvironmentMode::Black;
	// hdri only; resolved absolute filesystem path, see GltfInstance::path
	std::string path;
	// radians -- matches the renderer's own env-rotation convention
	// directly (app/viewer/main.cpp's sEnvRotation slider), so syncing
	// between the two needs no unit conversion
	f32 rotationRadians = 0.0f;
	f32 intensity = 1.0f;
};

// -----------------------------------------------------------------------------
// -- stage
// -----------------------------------------------------------------------------

// current schema version this library reads/writes; stage_load rejects (with
// an error print, returning an empty Stage) a file whose "version" is newer.
// bumped to 2 for the gltfMeshes/gltfModels/gltfInstances split (2026-07-24)
inline constexpr u32 kStageFormatVersion = 2;

struct Stage {
	u32 formatVersion = kStageFormatVersion;
	std::string name;
	std::string description;

	f32 metersPerUnit = 1.0f;
	std::string activeCamera;
	std::vector<Camera> cameras;
	Environment environment;
	std::vector<LightInstanceQuad> lights;
	std::vector<GltfMesh> gltfMeshes;
	std::vector<GltfModel> gltfModels;
	std::vector<GltfInstance> gltfInstances;
	std::vector<VdbInstance> vdbInstances;
	std::vector<AlembicInstance> alembicInstances;
	std::vector<StageInstance> stageInstances;

	// directory the stage was loaded/saved from; instance paths are resolved
	// relative to this at load, and rewritten relative to it on save
	std::string sourceDirectory;

	// gpu resources queued for teardown by stage_remove_gltf_model/
	// stage_remove_gltf_mesh, not yet destroyed. removal itself only ever
	// erases the cpu-side entry (cheap, safe to call from inside an open
	// ImGui window, same as stage_remove_gltf_instance always was); actually
	// destroying gpu resources needs vkof::device_wait_idle() first, which
	// must never run while an ImGui::Begin() window is still open (an
	// exception or abort partway through unwinds past that window's
	// ImGui::End(), corrupting ImGui's window stack -- see
	// app/viewer/main.cpp's own comment on the identical instance-removal
	// bug this mirrors). call stage_flush_pending_gpu_destroys once per
	// frame from outside any open window to actually process this
	std::vector<mor::GpuMaterials> pendingGpuMaterialsDestroy;
	struct PendingMeshDestroy { mor::GpuScene gpuScene; mor::Scene scene; };
	std::vector<PendingMeshDestroy> pendingGltfMeshDestroy;
};

// parses the json, resolves paths relative to its directory, and loads every
// gltf mesh/model/vdb instance (mor::scene_load_gltf + ponder::vdb_load,
// applying gltf material overrides and uploading) plus every stageInstance
// recursively. alembicInstances are parsed but never loaded -- no alembic
// loader exists yet. on failure (missing file, parse error, newer-than-
// supported version) prints an error and returns a default Stage with an
// empty sourceDirectory, which a successfully loaded stage never has --
// check that to detect failure
[[nodiscard]] Stage stage_load(char const * const path);

// tears down every mor::Scene/GpuScene/GpuMaterials and ponder::Vdb buffer,
// and recursively destroys + frees every stageInstance's child
void stage_destroy(Stage & stage);

// destroys every gpu resource queued by stage_remove_gltf_model/
// stage_remove_gltf_mesh (via one vkof::device_wait_idle() covering all of
// them), recursing into every stageInstance's child too. call once per
// frame from OUTSIDE any open ImGui window -- see Stage::pendingGpuMaterialsDestroy's
// declaration comment for why. cheap no-op when nothing is queued
void stage_flush_pending_gpu_destroys(Stage & stage);

// finds an existing GltfMesh for path (byte-identical resolved path, no
// path-equivalence canonicalization beyond what resolve_path already did at
// call time) and returns its index, or loads path as a new GltfMesh and
// appends it. returns -1 if the file fails to load. relative paths resolve
// against stage.sourceDirectory, or the current working directory if the
// stage has none yet (e.g. a brand new, never-saved stage)
i32 stage_find_or_add_gltf_mesh(Stage & stage, char const * const path);

// creates a new GltfModel referencing meshIndex (or loading/reusing path's
// mesh if meshIndex is -1, the common case -- see stage_find_or_add_gltf_mesh)
// with no material overrides (the mesh's own gltf-authored materials, as-
// loaded), appended to stage.gltfModels. returns -1 if meshIndex is out of
// range or path fails to load
i32 stage_add_gltf_model(
	Stage & stage,
	char const * const path,
	char const * const name = nullptr,
	i32 const meshIndex = -1
);

// creates a new GltfModel referencing the SAME meshIndex as
// stage.gltfModels[modelIndex] (shares geometry/BLAS, no re-upload), with an
// independent copy of its materialOverrides + a fresh mor::GpuMaterials to
// diverge from -- the "make a differently-colored variant of this mesh
// without duplicating geometry" operation. returns -1 if modelIndex is out
// of range
i32 stage_duplicate_gltf_model(
	Stage & stage, usize const modelIndex, char const * const name = nullptr
);

// appends a new GltfInstance referencing stage.gltfModels[modelIndex] at
// the given transform (default = origin/identity). returns false (instance
// not added) if modelIndex is out of range
bool stage_add_gltf_instance(
	Stage & stage,
	usize const modelIndex,
	char const * const name = nullptr,
	Transform const & transform = {}
);

// convenience: stage_find_or_add_gltf_mesh + stage_add_gltf_model +
// stage_add_gltf_instance in one call -- the common "pick a new file, place
// it in the scene" action. returns false if the file fails to load
bool stage_add_gltf(
	Stage & stage, char const * const path, char const * const name = nullptr
);

// loads path as a new vdb instance with one blob for gridName, appended to
// stage.vdbInstances. returns false (instance not added) if the file/grid
// fails to load
bool stage_add_vdb_instance(
	Stage & stage,
	char const * const path,
	char const * const gridName = "density",
	char const * const name = nullptr
);

// re-applies a gltf model's materialOverrides onto its mor::GpuMaterials and
// uploads; call after editing overrides in place (e.g. from imgui). every
// instance referencing this model picks up the change, since they all read
// the same model.materials -- mesh is the model's own gltfMeshes[meshIndex],
// needed to resolve override material names against the source scene
void stage_gltf_model_apply_overrides(GltfModel & model, GltfMesh const & mesh);

// tears down index's gpu resources and erases it from stage.gltfInstances.
// no-op if index is out of range. does not touch the referenced model/mesh
// (other instances may still reference them) -- see
// stage_remove_gltf_model/stage_remove_gltf_mesh for those
void stage_remove_gltf_instance(Stage & stage, usize index);

// erases stage.gltfModels[index] and tears down its GpuMaterials. refuses
// (prints a warning, no-op) if any instance still references it -- remove or
// repoint those instances first. does not touch the referenced mesh (other
// models may still reference it)
void stage_remove_gltf_model(Stage & stage, usize index);

// erases stage.gltfMeshes[index] and tears down its GpuScene/Scene. refuses
// (prints a warning, no-op) if any model still references it
void stage_remove_gltf_mesh(Stage & stage, usize index);

void stage_remove_vdb_instance(Stage & stage, usize index);

// serializes to path; every instance/environment path is rewritten relative
// to path's directory so the saved file + its assets stay relocatable
[[nodiscard]] bool stage_save(Stage const & stage, char const * const path);

// one-shot signal from a gltf instance's "focus" button (stage_imgui_edit)
// to whatever owns the camera; requested is false again after the frame it
// fires on -- the caller doesn't need to clear it itself
struct FocusRequest {
	bool requested = false;
	f32v3 center {};
	f32 extent = 1.0f;
};

// outliner + per-item inspector; editable in place. returns true if anything
// changed this frame -- drive an "unsaved changes" indicator from this.
// outFocus, if non-null, is filled when a gltf instance's "focus" button is
// clicked this frame (see FocusRequest) -- the caller owns the camera, so
// this is the only way stage_imgui_edit can ask for it to move.
// showSaveButton draws an inline save-path input + save button at the
// bottom; defaults off since most callers (app/viewer included) have their
// own save ui wired to a specific file already -- pass true for a stage
// that has no other save path of its own, e.g. a nested stageInstance's
// child, which stage_imgui_edit recurses into with this set
bool stage_imgui_edit(
	Stage & stage,
	FocusRequest * outFocus = nullptr,
	bool showSaveButton = false
);

// 3d viewport transform gizmo (imguizmo-based) for whichever instance is
// currently selected in stage_imgui_edit's outliner. call once per frame,
// after your own 3d scene draw, with the active camera's view/projection
// and the viewport rect (x, y, width, height) in screen pixels. requires
// stage_imgui_edit to have been called on this stage first (or in the same
// frame) so the outliner selection exists; a no-op (returns false) if
// nothing selected, or if the current selection is a camera (cameras aren't
// gizmo-manipulable, only cameras themselves get dragged as instances of
// gltf/vdb/light -- point the camera itself with draw_camera's fields)
[[nodiscard]] bool stage_imgui_gizmo_edit(
	Stage & stage,
	f32m44 const & view,
	f32m44 const & projection,
	f32v4 const & viewportRect
);

// draws the material-editor fields for gltfInstances[instanceIndex]'s
// model's material at materialIndex (resolved via instance.modelIndex --
// editing here edits the shared GltfModel, so every other instance of the
// same model picks up the change too), inside whatever popup/window the
// caller has already opened (e.g. app/viewer's right-click "material
// inspector" popup, driven by a gpu probe rather than stage_imgui_edit's
// own selection). edits persist into the model's materialOverrides and are
// applied + uploaded immediately, same as any other material edit. false
// (nothing drawn) if instanceIndex/materialIndex don't resolve to a loaded
// material. also draws "select"/"focus" buttons (updating stage_imgui_edit's
// own outliner selection, and outFocus same as stage_imgui_edit's) since a
// probe-driven popup like this is otherwise disconnected from the outliner
// entirely -- there'd be no other way to point the transform window/gizmo
// at whatever was right-clicked
bool stage_imgui_material_inspector(
	Stage & stage,
	usize const instanceIndex,
	u32 const materialIndex,
	FocusRequest * outFocus = nullptr
);

// clears stage_imgui_edit's outliner selection (and therefore whatever the
// transform window / gizmo were targeting). safe to call even if nothing
// is currently selected
void stage_clear_selection(Stage & stage);

// if a gltf-model drag payload (started from stage_imgui_edit's "gltf
// models" panel) is being dropped on the current ImGui drag-drop target
// this frame, adds a new instance of that model at the given transform and
// returns true. false (no-op) if nothing is being dropped. the caller owns
// setting up the actual drop-target zone (ImGui::BeginDragDropTarget()
// after positioning whatever widget/window should catch the drop -- e.g.
// an invisible window covering the 3d viewport) and should call this right
// after that BeginDragDropTarget() succeeds; the payload's string tag is
// intentionally not exposed here, so callers can't accidentally typo a
// mismatched AcceptDragDropPayload call themselves
[[nodiscard]] bool stage_imgui_accept_gltf_model_drop(
	Stage & stage, Transform const & at = {}
);

} // namespace stage
