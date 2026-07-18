// imgui outliner + inspector editor for stage::Stage, plus an imguizmo-based
// 3d transform gizmo driven by the same outliner selection

#include <stage/stage.hpp>

#include <mor/mor.hpp>
#include <ponder/vdb.hpp>

#include <imgui.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <unordered_map>

enum class SelectionKind { None, Camera, Light, Gltf, Vdb };

struct Selection {
	SelectionKind kind = SelectionKind::None;
	usize index = 0;
};

struct EditorState {
	Selection selection;
	char savePathBuffer[512] = {};
};

// keyed by Stage address so nested stageInstances (each a distinct heap
// Stage owned via StageInstance::child) get independent selection/save
// state from their parent and from each other -- a plain function-local
// static would alias every level of the recursion onto the same state
static std::unordered_map<stage::Stage const *, EditorState> gEditorStates;

static EditorState & editor_state_for(stage::Stage const & s) {
	return gEditorStates[&s];
}

// shared by stage_imgui_gizmo_edit and the transform window below --
// whichever instance type the outliner has selected, if any (cameras have
// no stage::Transform of their own, so a camera selection resolves to null)
static stage::Transform * selected_transform(
	stage::Stage & s, Selection const & selection
) {
	switch (selection.kind) {
		case SelectionKind::Light:
			if (selection.index < s.lights.size()) {
				return &s.lights[selection.index].transform;
			}
			break;
		case SelectionKind::Gltf:
			if (selection.index < s.gltfInstances.size()) {
				return &s.gltfInstances[selection.index].transform;
			}
			break;
		case SelectionKind::Vdb:
			if (selection.index < s.vdbInstances.size()) {
				return &s.vdbInstances[selection.index].transform;
			}
			break;
		default:
			break;
	}
	return nullptr;
}

// shared by the gltf/vdb outliner rows' "focus" button and the material
// popup's -- fills outFocus (a no-op if null, e.g. the caller didn't wire
// one up) from a world-space aabb
static void request_focus(
	stage::FocusRequest * const outFocus,
	f32v3 const & worldMin,
	f32v3 const & worldMax
) {
	if (outFocus == nullptr) { return; }
	outFocus->requested = true;
	outFocus->center = (worldMin + worldMax) * 0.5f;
	outFocus->extent = std::max(
		{
			worldMax.x - worldMin.x,
			worldMax.y - worldMin.y,
			worldMax.z - worldMin.z,
			0.01f,
		}
	);
}

// -----------------------------------------------------------------------------
// -- small widgets
// -----------------------------------------------------------------------------

static bool draw_transform_fields(stage::Transform & t) {
	bool changed = false;
	changed |= ImGui::DragFloat3("position", &t.position.x, 0.01f);
	// quaternions aren't hand-editable; decompose fresh from the quaternion
	// every frame and reconstruct on edit, rather than caching a euler value
	// (a cache keyed by address would go stale if the owning vector
	// reallocates)
	f32v3 euler = f32quat_to_euler_xyz_degrees(t.rotation);
	if (ImGui::DragFloat3("rotation (deg)", &euler.x, 1.0f)) {
		t.rotation = f32quat_from_euler_xyz_degrees(euler);
		changed = true;
	}
	changed |= ImGui::DragFloat3("scale", &t.scale.x, 0.01f, 0.0001f, 1000.0f);
	// uniform-scale: a multiply-by-drag control, not a stored value.
	// resetting the local to 1 every call turns imgui's own per-widget-id
	// drag continuation (which persists across frames on its own) into
	// "this frame's incremental multiplier" rather than an absolute value --
	// unlabeled and format="" so no number ever prints, since 1.0 is the
	// only value that would ever show
	ImGui::Text("uniform scale");
	ImGui::SameLine();
	f32 uniformScale = 1.0f;
	if (
		ImGui::DragFloat("###uniformScale", &uniformScale, 0.01f, 0.01f, 100.0f, "")
	) {
		t.scale.x *= uniformScale;
		t.scale.y *= uniformScale;
		t.scale.z *= uniformScale;
		changed = true;
	}
	return changed;
}

// separate window (rather than inline in the main outliner/inspector
// panel) so it can be moved/docked independently while dragging values.
// window id is scoped per-Stage (###transform_<address>) so a nested
// stage's own transform window doesn't collide with its parent's when both
// have something selected at once
static bool draw_transform_window(stage::Stage & s, Selection const & selection) {
	stage::Transform * const target = selected_transform(s, selection);
	if (target == nullptr) { return false; }
	char windowId[64];
	snprintf(
		windowId, sizeof(windowId), "transform###transform_%p", (void const *)&s
	);
	bool changed = false;
	if (ImGui::Begin(windowId)) {
		changed = draw_transform_fields(*target);
	}
	ImGui::End();
	return changed;
}

template <typename Component>
static bool draw_material_field_slot(
	Component & value,
	char const * const fieldName,
	f32 const lo,
	f32 const hi,
	stage::MaterialOverride & mo
) {
	bool changed = false;
	if constexpr (std::is_same_v<Component, GpuMorMaterialComponent3>) {
		changed = ImGui::SliderFloat3(fieldName, &value.rgb.x, lo, hi);
	} else {
		changed = ImGui::SliderFloat(fieldName, &value.r, lo, hi);
	}
	if (!changed) { return false; }
	stage::MaterialFieldOverride * entry = nullptr;
	for (stage::MaterialFieldOverride & f : mo.fields) {
		if (f.field != fieldName) { continue; }
		entry = &f;
		break;
	}
	if (entry == nullptr) {
		stage::MaterialFieldOverride fresh;
		fresh.field = fieldName;
		mo.fields.emplace_back(std::move(fresh));
		entry = &mo.fields.back();
	}
	if constexpr (std::is_same_v<Component, GpuMorMaterialComponent3>) {
		entry->colorValue = value.rgb;
	} else {
		entry->scalarValue = value.r;
	}
	return true;
}

static stage::MaterialOverride & find_or_create_override(
	stage::GltfInstance & inst, std::string const & materialName
) {
	for (stage::MaterialOverride & mo : inst.materialOverrides) {
		if (mo.material != materialName) { continue; }
		return mo;
	}
	stage::MaterialOverride fresh;
	fresh.material = materialName;
	inst.materialOverrides.emplace_back(std::move(fresh));
	return inst.materialOverrides.back();
}

// mirrors app/viewer/main.cpp's fnImguiMaterial grouping (weight-gated
// sections), self-contained here since that function is app-local. drawn
// inside whatever popup/window the caller opened -- see
// stage::stage_imgui_material_inspector and app/viewer's right-click
// "material inspector" popup, the only place this is invoked from
static bool draw_material_inspector(
	mor::GpuMaterials const & materials,
	stage::MaterialOverride & mo,
	u32 const materialIndex
) {
	GpuMorMaterial mat = mor::scene_gpu_materials_get(materials, materialIndex);
	bool changed = false;

	ImGui::SeparatorText("weights");
#define X(Type, field, lo, hi) \
	changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
	MOR_MATERIAL_ALL_PARAMS_WEIGHTS(X)
#undef X

	ImGui::SeparatorText("base");
#define X(Type, field, lo, hi) \
	changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
	MOR_MATERIAL_ALL_PARAMS_BASE(X)
#undef X

	ImGui::SeparatorText("specular");
#define X(Type, field, lo, hi) \
	changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
	MOR_MATERIAL_ALL_PARAMS_SPECULAR(X)
#undef X

	if (mat.transmissionWeight.r > 0.0f) {
		ImGui::SeparatorText("transmission");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_TRANSMISSION(X)
#undef X
	}
	if (mat.subsurfaceWeight.r > 0.0f) {
		ImGui::SeparatorText("subsurface");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_SUBSURFACE(X)
#undef X
	}
	if (mat.coatWeight.r > 0.0f) {
		ImGui::SeparatorText("coat");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_COAT(X)
#undef X
	}
	if (mat.fuzzWeight.r > 0.0f) {
		ImGui::SeparatorText("fuzz");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_FUZZ(X)
#undef X
	}
	if (mat.emissionLuminance.r > 0.0f) {
		ImGui::SeparatorText("emission");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_EMISSION(X)
#undef X
	}
	if (mat.thinFilmWeight.r > 0.0f) {
		ImGui::SeparatorText("thin film");
#define X(Type, field, lo, hi) \
		changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
		MOR_MATERIAL_ALL_PARAMS_THIN_FILM(X)
#undef X
	}

	ImGui::SeparatorText("geometry");
#define X(Type, field, lo, hi) \
	changed |= draw_material_field_slot(mat.field, #field, lo, hi, mo);
	MOR_MATERIAL_ALL_PARAMS_GEOMETRY(X)
#undef X

	if (changed) {
		mor::scene_material_override_scalars(materials, materialIndex, mat);
		mor::scene_gpu_materials_upload(materials);
	}
	return changed;
}

bool stage::stage_imgui_material_inspector(
	Stage & s,
	usize const instanceIndex,
	u32 const materialIndex,
	FocusRequest * const outFocus
) {
	if (instanceIndex >= s.gltfInstances.size()) { return false; }
	GltfInstance & inst = s.gltfInstances[instanceIndex];
	if (inst.scene.id == 0u || inst.materials.id == 0u) { return false; }
	if (materialIndex >= mor::scene_material_count(inst.scene)) { return false; }
	std::string const name = mor::scene_material_name(inst.scene, materialIndex);
	ImGui::Text("#%u %s", materialIndex, name.c_str());
	if (ImGui::SmallButton("select")) {
		editor_state_for(s).selection = { SelectionKind::Gltf, instanceIndex };
	}
	ImGui::SameLine();
	if (ImGui::SmallButton("focus")) {
		f32v3 worldMin, worldMax;
		if (stage::gltf_instance_world_bounds(inst, worldMin, worldMax)) {
			request_focus(outFocus, worldMin, worldMax);
		}
	}
	ImGui::Separator();
	MaterialOverride & mo = find_or_create_override(inst, name);
	return draw_material_inspector(inst.materials, mo, materialIndex);
}

static bool draw_gltf_instance(stage::GltfInstance & inst) {
	bool changed = false;
	ImGui::TextWrapped("path: %s", inst.path.c_str());
	if (inst.scene.id != 0u) {
		// materials are edited via the right-click "material inspector"
		// popup (app/viewer/main.cpp), same as file view -- not inlined
		// here, since a full material panel per instance in this list
		// would be unwieldy with more than a couple of instances loaded
		ImGui::TextDisabled(
			"%u material(s); right-click a rendered pixel to edit",
			mor::scene_material_count(inst.scene)
		);
	}
	return changed;
}

static bool draw_vdb_instance(stage::VdbInstance & inst) {
	bool changed = false;
	ImGui::TextWrapped("path: %s", inst.path.c_str());
	ImGui::SeparatorText("blobs");
	for (stage::VdbBlob & blob : inst.blobs) {
		ImGui::PushID(blob.gridName.c_str());
		if (ImGui::TreeNode(blob.gridName.c_str())) {
			// same widgets/ranges as file view's global vdb sliders --
			// these are the only vdb params the shader actually consumes
			// (GpuGlobalExtended.vdbSigmaScale/vdbDropletDiameter/vdbAlbedo)
			changed |= ImGui::SliderFloat(
				"sigma scale", &blob.sigmaScale, 0.0f, 100.0f, "%.3f",
				ImGuiSliderFlags_Logarithmic
			);
			changed |= ImGui::SliderFloat(
				"droplet diameter", &blob.dropletDiameter, 5.0f, 50.0f
			);
			changed |= ImGui::ColorEdit3("albedo", &blob.albedo.x);
			ImGui::Text("majorant density: %.4f", blob.vdb.densityMax);
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	return changed;
}

static bool draw_light(stage::LightInstanceQuad & light) {
	bool changed = false;
	changed |= ImGui::DragFloat("width", &light.width, 0.01f, 0.001f, 1000.0f);
	changed |= (
		ImGui::DragFloat("height", &light.height, 0.01f, 0.001f, 1000.0f)
	);
	changed |= ImGui::ColorEdit3(
		"radiance",
		&light.radiance.x,
		ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR
	);
	changed |= ImGui::Checkbox("two sided", &light.twoSided);
	return changed;
}

static bool draw_camera(stage::Camera & cam) {
	bool changed = false;
	changed |= ImGui::DragFloat3("position", &cam.position.x, 0.01f);
	changed |= ImGui::DragFloat3("target", &cam.target.x, 0.01f);
	changed |= ImGui::DragFloat3("up", &cam.up.x, 0.01f);
	changed |= (
		ImGui::DragFloat("fov y (deg)", &cam.fovYDegrees, 0.1f, 1.0f, 179.0f)
	);
	changed |= ImGui::DragFloat("near", &cam.near, 0.001f, 0.0001f, cam.far);
	changed |= ImGui::DragFloat("far", &cam.far, 1.0f, cam.near, 1000000.0f);
	return changed;
}

// -----------------------------------------------------------------------------
// -- stage_imgui_edit
// -----------------------------------------------------------------------------

bool stage::stage_imgui_edit(
	Stage & s, FocusRequest * outFocus, bool showSaveButton
) {
	EditorState & state = editor_state_for(s);
	if (state.savePathBuffer[0] == '\0' && !s.sourceDirectory.empty()) {
		std::string const def = s.sourceDirectory + "/scene.json";
		strncpy(state.savePathBuffer, def.c_str(), sizeof(state.savePathBuffer) - 1);
	}
	bool changed = false;

	ImGui::PushID(&s);
	if (!s.name.empty()) { ImGui::SeparatorText(s.name.c_str()); }

	if (ImGui::TreeNodeEx("cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (usize i = 0u; i < s.cameras.size(); ++i) {
			bool const isSelected = (
				state.selection.kind == SelectionKind::Camera
				&& state.selection.index == i
			);
			if (ImGui::Selectable(s.cameras[i].name.c_str(), isSelected)) {
				state.selection = { SelectionKind::Camera, i };
			}
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (usize i = 0u; i < s.lights.size(); ++i) {
			bool const isSelected = (
				state.selection.kind == SelectionKind::Light
				&& state.selection.index == i
			);
			if (ImGui::Selectable(s.lights[i].name.c_str(), isSelected)) {
				state.selection = { SelectionKind::Light, i };
			}
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("gltf instances", ImGuiTreeNodeFlags_DefaultOpen)) {
		usize eraseIndex = (usize)-1;
		for (usize i = 0u; i < s.gltfInstances.size(); ++i) {
			stage::GltfInstance & inst = s.gltfInstances[i];
			ImGui::PushID((i32)i);
			changed |= ImGui::Checkbox("##visible", &inst.visible);
			ImGui::SameLine();
			bool const isSelected = (
				state.selection.kind == SelectionKind::Gltf
				&& state.selection.index == i
			);
			if (
				ImGui::Selectable(
					inst.name.c_str(), isSelected, 0, ImVec2(120.0f, 0.0f)
				)
			) {
				state.selection = { SelectionKind::Gltf, i };
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("focus")) {
				f32v3 worldMin, worldMax;
				if (stage::gltf_instance_world_bounds(inst, worldMin, worldMax)) {
					request_focus(outFocus, worldMin, worldMax);
				}
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("delete")) {
				eraseIndex = i;
			}
			ImGui::PopID();
		}
		if (eraseIndex != (usize)-1) {
			stage::stage_remove_gltf_instance(s, eraseIndex);
			if (state.selection.kind == SelectionKind::Gltf) {
				if (state.selection.index == eraseIndex) {
					state.selection = {};
				} else if (state.selection.index > eraseIndex) {
					state.selection.index -= 1u;
				}
			}
			changed = true;
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx("vdb instances", ImGuiTreeNodeFlags_DefaultOpen)) {
		usize eraseIndex = (usize)-1;
		for (usize i = 0u; i < s.vdbInstances.size(); ++i) {
			stage::VdbInstance & inst = s.vdbInstances[i];
			ImGui::PushID((i32)i);
			changed |= ImGui::Checkbox("##visible", &inst.visible);
			ImGui::SameLine();
			bool const isSelected = (
				state.selection.kind == SelectionKind::Vdb
				&& state.selection.index == i
			);
			if (
				ImGui::Selectable(
					inst.name.c_str(), isSelected, 0, ImVec2(120.0f, 0.0f)
				)
			) {
				state.selection = { SelectionKind::Vdb, i };
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("focus")) {
				f32v3 worldMin, worldMax;
				if (stage::vdb_instance_world_bounds(inst, worldMin, worldMax)) {
					request_focus(outFocus, worldMin, worldMax);
				}
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("delete")) {
				eraseIndex = i;
			}
			ImGui::PopID();
		}
		if (eraseIndex != (usize)-1) {
			stage::stage_remove_vdb_instance(s, eraseIndex);
			if (state.selection.kind == SelectionKind::Vdb) {
				if (state.selection.index == eraseIndex) {
					state.selection = {};
				} else if (state.selection.index > eraseIndex) {
					state.selection.index -= 1u;
				}
			}
			changed = true;
		}
		ImGui::TreePop();
	}

	if (!s.alembicInstances.empty() && ImGui::TreeNode("alembic instances")) {
		for (stage::AlembicInstance const & inst : s.alembicInstances) {
			ImGui::TextDisabled("%s (not loaded)", inst.name.c_str());
		}
		ImGui::TreePop();
	}

	if (!s.stageInstances.empty()) {
		bool const open = (
			ImGui::TreeNodeEx("nested stages", ImGuiTreeNodeFlags_DefaultOpen)
		);
		if (open) {
			for (stage::StageInstance & inst : s.stageInstances) {
				ImGui::PushID(&inst);
				changed |= ImGui::Checkbox("##visible", &inst.visible);
				ImGui::SameLine();
				if (ImGui::TreeNode(inst.name.c_str())) {
					changed |= draw_transform_fields(inst.transform);
					if (inst.child != nullptr) {
						// no other save ui reaches a nested child stage, so
						// it gets its own inline one unlike the top-level call
						changed |= (
							stage_imgui_edit(*inst.child, outFocus, true)
						);
					}
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			ImGui::TreePop();
		}
	}

	ImGui::Separator();
	switch (state.selection.kind) {
		case SelectionKind::Camera:
			if (state.selection.index < s.cameras.size()) {
				changed |= draw_camera(s.cameras[state.selection.index]);
			}
			break;
		case SelectionKind::Light:
			if (state.selection.index < s.lights.size()) {
				changed |= draw_light(s.lights[state.selection.index]);
			}
			break;
		case SelectionKind::Gltf:
			if (state.selection.index < s.gltfInstances.size()) {
				changed |= (
					draw_gltf_instance(s.gltfInstances[state.selection.index])
				);
			}
			break;
		case SelectionKind::Vdb:
			if (state.selection.index < s.vdbInstances.size()) {
				changed |= (
					draw_vdb_instance(s.vdbInstances[state.selection.index])
				);
			}
			break;
		default:
			break;
	}
	changed |= draw_transform_window(s, state.selection);

	if (showSaveButton) {
		ImGui::Separator();
		ImGui::InputText(
			"##savepath", state.savePathBuffer, sizeof(state.savePathBuffer)
		);
		ImGui::SameLine();
		if (ImGui::Button("save")) {
			if (!stage_save(s, state.savePathBuffer)) {
				printf("stage: save to '%s' failed\n", state.savePathBuffer);
			}
		}
	}

	ImGui::PopID();
	return changed;
}

void stage::stage_clear_selection(Stage & s) {
	editor_state_for(s).selection = Selection {};
}

// -----------------------------------------------------------------------------
// -- stage_imgui_gizmo_edit
// -----------------------------------------------------------------------------

bool stage::stage_imgui_gizmo_edit(
	Stage & s,
	f32m44 const & view,
	f32m44 const & projection,
	f32v4 const & viewportRect
) {
	EditorState & state = editor_state_for(s);
	Transform * const target = selected_transform(s, state.selection);
	if (target == nullptr) { return false; }

	ImGuizmo::SetRect(
		viewportRect.x, viewportRect.y, viewportRect.z, viewportRect.w
	);
	ImGuizmo::SetOrthographic(false);
	ImGuizmo::BeginFrame();

	f32m44 matrix = stage::transform_to_m44(*target);
	// scale intentionally excluded -- the uniform-scale drag in the
	// transform window covers it, and non-uniform gizmo scale handles are
	// easy to bump by accident while translating/rotating
	bool const manipulated = ImGuizmo::Manipulate(
		&view.m[0],
		&projection.m[0],
		ImGuizmo::OPERATION(ImGuizmo::TRANSLATE | ImGuizmo::ROTATE),
		ImGuizmo::LOCAL,
		&matrix.m[0]
	);
	if (!manipulated) { return false; }

	f32m44_decompose(matrix, target->position, target->rotation, target->scale);
	return true;
}
