// json scene file: parsing, loading (via mor/ponder), override application,
// and save-back serialization

#include <stage/stage.hpp>

#include <mor/mor.hpp>
#include <ponder/vdb.hpp>
#include <vkof/vkof.hpp>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <filesystem>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;
using JsonWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

// -----------------------------------------------------------------------------
// -- json read helpers
// -----------------------------------------------------------------------------

static f32 json_f32(
	rapidjson::Value const & obj, char const * const key, f32 const fallback
) {
	if (!obj.IsObject() || !obj.HasMember(key) || !obj[key].IsNumber()) {
		return fallback;
	}
	return obj[key].GetFloat();
}

static bool json_bool(
	rapidjson::Value const & obj, char const * const key, bool const fallback
) {
	if (!obj.IsObject() || !obj.HasMember(key) || !obj[key].IsBool()) {
		return fallback;
	}
	return obj[key].GetBool();
}

static std::string json_string(
	rapidjson::Value const & obj,
	char const * const key,
	std::string const & fallback
) {
	if (!obj.IsObject() || !obj.HasMember(key) || !obj[key].IsString()) {
		return fallback;
	}
	return obj[key].GetString();
}

static f32v3 json_f32v3(
	rapidjson::Value const & obj, char const * const key, f32v3 const fallback
) {
	if (
		!obj.IsObject() || !obj.HasMember(key)
		|| !obj[key].IsArray() || obj[key].Size() != 3
	) {
		return fallback;
	}
	auto const & arr = obj[key];
	return {
		arr[0].IsNumber() ? arr[0].GetFloat() : fallback.x,
		arr[1].IsNumber() ? arr[1].GetFloat() : fallback.y,
		arr[2].IsNumber() ? arr[2].GetFloat() : fallback.z,
	};
}

static stage::Transform parse_transform(rapidjson::Value const & obj) {
	stage::Transform t;
	if (
		!obj.IsObject()
		|| !obj.HasMember("transform") || !obj["transform"].IsObject()
	) {
		return t;
	}
	rapidjson::Value const & tr = obj["transform"];
	t.position = json_f32v3(tr, "position", t.position);
	t.scale = json_f32v3(tr, "scale", t.scale);
	if (
		tr.HasMember("rotation")
		&& tr["rotation"].IsArray() && tr["rotation"].Size() == 4
	) {
		auto const & r = tr["rotation"];
		t.rotation = f32quat_normalize({
			r[0].GetFloat(), r[1].GetFloat(), r[2].GetFloat(), r[3].GetFloat(),
		});
	}
	return t;
}

// json-authored path (raw, possibly relative) resolved against dir into an
// absolute, normalized path -- see GltfInstance::path
static std::string resolve_path(fs::path const & dir, std::string const & raw) {
	if (raw.empty()) { return raw; }
	fs::path p(raw);
	if (p.is_relative()) { p = dir / p; }
	std::error_code ec;
	fs::path const canon = fs::weakly_canonical(p, ec);
	return (ec ? p : canon).string();
}

// inverse of resolve_path, for stage_save: an absolute path rewritten
// relative to saveDir, forward-slashed for portability
static std::string rebase_path(
	fs::path const & saveDir, std::string const & absPath
) {
	if (absPath.empty()) { return absPath; }
	std::error_code ec;
	fs::path const rel = fs::relative(fs::path(absPath), saveDir, ec);
	if (ec || rel.empty()) { return absPath; }
	return rel.generic_string();
}

// -----------------------------------------------------------------------------
// -- material field overrides (MOR_MATERIAL_ALL_PARAMS-driven)
// -----------------------------------------------------------------------------

static bool material_field_is_color(std::string const & name) {
#define X(Type, field, lo, hi) \
	if (name == #field) { \
		return std::is_same_v<Type, GpuMorMaterialComponent3>; \
	}
	MOR_MATERIAL_ALL_PARAMS(X)
#undef X
	return false;
}

// Component is a genuine deduced template parameter here (unlike a macro-
// substituted type name), so if constexpr actually discards the untaken
// branch instead of hard-compiling both against a single concrete type
template <typename Component>
static bool material_field_apply_one(
	Component & value,
	char const * const name,
	stage::MaterialFieldOverride const & fieldOverride
) {
	if (fieldOverride.field != name) { return false; }
	if constexpr (std::is_same_v<Component, GpuMorMaterialComponent3>) {
		value.rgb = fieldOverride.colorValue;
	} else {
		value.r = fieldOverride.scalarValue;
	}
	return true;
}

static bool material_field_apply(
	GpuMorMaterial & mat, stage::MaterialFieldOverride const & fieldOverride
) {
#define X(Type, field, lo, hi) \
	if (material_field_apply_one(mat.field, #field, fieldOverride)) { \
		return true; \
	}
	MOR_MATERIAL_ALL_PARAMS(X)
#undef X
	return false;
}

void stage::stage_gltf_model_apply_overrides(
	GltfModel & model, GltfMesh const & mesh
) {
	if (model.materials.id == 0u) { return; }
	u32 const materialCount = mor::scene_material_count(mesh.scene);
	for (MaterialOverride const & mo : model.materialOverrides) {
		i32 materialIndex = -1;
		for (u32 i = 0u; i < materialCount; ++i) {
			if (mor::scene_material_name(mesh.scene, i) == mo.material) {
				materialIndex = (i32)i;
				break;
			}
		}
		if (materialIndex < 0) {
			printf(
				"stage: material override references unknown material '%s'\n",
				mo.material.c_str()
			);
			continue;
		}
		GpuMorMaterial mat = (
			mor::scene_material_get(mesh.scene, (u32)materialIndex)
		);
		for (MaterialFieldOverride const & field : mo.fields) {
			if (material_field_apply(mat, field)) { continue; }
			printf(
				"stage: material override references unknown field '%s'\n",
				field.field.c_str()
			);
		}
		mor::scene_material_override_scalars(
			model.materials, (u32)materialIndex, mat
		);
	}
	mor::scene_gpu_materials_upload(model.materials);
}

static std::vector<stage::MaterialOverride> parse_material_overrides(
	rapidjson::Value const & obj
) {
	std::vector<stage::MaterialOverride> overrides;
	if (!obj.IsObject() || !obj.HasMember("materialOverrides")) {
		return overrides;
	}
	rapidjson::Value const & arr = obj["materialOverrides"];
	if (!arr.IsArray()) { return overrides; }
	for (auto const & entry : arr.GetArray()) {
		if (!entry.IsObject()) { continue; }
		stage::MaterialOverride mo;
		mo.material = json_string(entry, "material", "");
		if (entry.HasMember("fields") && entry["fields"].IsObject()) {
			for (auto const & member : entry["fields"].GetObject()) {
				stage::MaterialFieldOverride field;
				field.field = member.name.GetString();
				rapidjson::Value const & fieldObj = member.value;
				if (fieldObj.IsObject() && fieldObj.HasMember("value")) {
					rapidjson::Value const & v = fieldObj["value"];
					if (v.IsArray() && v.Size() == 3) {
						field.colorValue = {
							v[0].GetFloat(), v[1].GetFloat(), v[2].GetFloat(),
						};
					} else if (v.IsNumber()) {
						field.scalarValue = v.GetFloat();
					}
				}
				field.texturePath = json_string(fieldObj, "texturePath", "");
				mo.fields.emplace_back(std::move(field));
			}
		}
		overrides.emplace_back(std::move(mo));
	}
	return overrides;
}

// -----------------------------------------------------------------------------
// -- instance parsing/loading
// -----------------------------------------------------------------------------

static void load_gltf_mesh(
	rapidjson::Value const & obj, fs::path const & dir, stage::Stage & out
) {
	stage::GltfMesh mesh;
	mesh.path = resolve_path(dir, json_string(obj, "path", ""));
	if (mesh.path.empty()) {
		printf("stage: gltf mesh entry has no path\n");
		out.gltfMeshes.emplace_back(stage::GltfMesh {});
		return;
	}
	mesh.scene = mor::scene_create();
	mor::scene_load_gltf(mesh.scene, mesh.path.c_str());
	mesh.gpuScene = mor::scene_gpu_upload(mesh.scene);
	out.gltfMeshes.emplace_back(std::move(mesh));
}

static void load_gltf_model(
	rapidjson::Value const & obj, stage::Stage & out
) {
	stage::GltfModel model;
	model.name = json_string(obj, "name", "");
	model.meshIndex = (u32)json_f32(obj, "meshIndex", 0.0f);
	model.materialOverrides = parse_material_overrides(obj);
	if (model.meshIndex >= out.gltfMeshes.size()) {
		printf(
			"stage: gltf model '%s' references out-of-range meshIndex %u\n",
			model.name.c_str(), model.meshIndex
		);
		out.gltfModels.emplace_back(std::move(model));
		return;
	}
	stage::GltfMesh const & mesh = out.gltfMeshes[model.meshIndex];
	if (mesh.scene.id != 0u) {
		model.materials = mor::scene_gpu_materials_create(mesh.scene);
		stage::stage_gltf_model_apply_overrides(model, mesh);
	}
	out.gltfModels.emplace_back(std::move(model));
}

static void load_gltf_instance(
	rapidjson::Value const & obj, fs::path const & dir, stage::Stage & out
) {
	// pre-version-2 files had "path"/"materialOverrides" directly on the
	// instance (no gltfMeshes/gltfModels split) -- fall back to loading a
	// fresh mesh+model for it, same as stage_add_gltf_model, so old files
	// still work
	if (obj.HasMember("path")) {
		std::string const path = (
			resolve_path(dir, json_string(obj, "path", ""))
		);
		if (path.empty()) {
			printf(
				"stage: gltf instance '%s' has no path\n",
				json_string(obj, "name", "").c_str()
			);
			return;
		}
		i32 const meshIndex = stage::stage_find_or_add_gltf_mesh(out, path.c_str());
		if (meshIndex < 0) { return; }
		stage::GltfModel model;
		model.name = json_string(obj, "name", "");
		model.meshIndex = (u32)meshIndex;
		model.materialOverrides = parse_material_overrides(obj);
		stage::GltfMesh const & mesh = out.gltfMeshes[meshIndex];
		if (mesh.scene.id != 0u) {
			model.materials = mor::scene_gpu_materials_create(mesh.scene);
			stage::stage_gltf_model_apply_overrides(model, mesh);
		}
		out.gltfModels.emplace_back(std::move(model));

		stage::GltfInstance inst;
		inst.name = json_string(obj, "name", "");
		inst.modelIndex = (u32)(out.gltfModels.size() - 1u);
		inst.transform = parse_transform(obj);
		inst.visible = json_bool(obj, "visible", true);
		out.gltfInstances.emplace_back(std::move(inst));
		return;
	}
	stage::GltfInstance inst;
	inst.name = json_string(obj, "name", "");
	inst.modelIndex = (u32)json_f32(obj, "modelIndex", 0.0f);
	inst.transform = parse_transform(obj);
	inst.visible = json_bool(obj, "visible", true);
	if (inst.modelIndex >= out.gltfModels.size()) {
		printf(
			"stage: gltf instance '%s' references out-of-range modelIndex %u\n",
			inst.name.c_str(), inst.modelIndex
		);
	}
	out.gltfInstances.emplace_back(std::move(inst));
}

static void load_vdb_instance(
	rapidjson::Value const & obj, fs::path const & dir, stage::Stage & out
) {
	stage::VdbInstance inst;
	inst.name = json_string(obj, "name", "");
	inst.path = resolve_path(dir, json_string(obj, "path", ""));
	inst.transform = parse_transform(obj);
	inst.visible = json_bool(obj, "visible", true);
	if (inst.path.empty()) {
		printf("stage: vdb instance '%s' has no path\n", inst.name.c_str());
		return;
	}
	if (obj.HasMember("blobs") && obj["blobs"].IsArray()) {
		for (auto const & blobVal : obj["blobs"].GetArray()) {
			if (!blobVal.IsObject()) { continue; }
			stage::VdbBlob blob;
			blob.gridName = json_string(blobVal, "grid", "density");
			blob.sigmaAbsorption = (
				json_f32v3(blobVal, "sigmaAbsorption", blob.sigmaAbsorption)
			);
			blob.sigmaScattering = (
				json_f32v3(blobVal, "sigmaScattering", blob.sigmaScattering)
			);
			blob.dropletDiameter = (
				json_f32(blobVal, "dropletDiameter", blob.dropletDiameter)
			);
			blob.phaseAnisotropyOverride = (
				json_f32(
					blobVal, "phaseAnisotropyOverride", blob.phaseAnisotropyOverride
				)
			);
			blob.temperatureScale = (
				json_f32(blobVal, "temperatureScale", blob.temperatureScale)
			);
			blob.emissionScale = (
				json_f32(blobVal, "emissionScale", blob.emissionScale)
			);
			blob.vdb = (
				ponder::vdb_load(inst.path.c_str(), blob.gridName.c_str())
			);
			inst.blobs.emplace_back(std::move(blob));
		}
	}
	out.vdbInstances.emplace_back(std::move(inst));
}

static void parse_alembic_instance(
	rapidjson::Value const & obj, fs::path const & dir, stage::Stage & out
) {
	stage::AlembicInstance inst;
	inst.name = json_string(obj, "name", "");
	inst.path = resolve_path(dir, json_string(obj, "path", ""));
	inst.transform = parse_transform(obj);
	inst.visible = json_bool(obj, "visible", true);
	inst.objectPath = json_string(obj, "objectPath", "");
	inst.time = json_f32(obj, "time", 0.0f);
	out.alembicInstances.emplace_back(std::move(inst));
}

static void load_stage_instance(
	rapidjson::Value const & obj, fs::path const & dir, stage::Stage & out
) {
	stage::StageInstance inst;
	inst.name = json_string(obj, "name", "");
	inst.path = resolve_path(dir, json_string(obj, "path", ""));
	inst.transform = parse_transform(obj);
	inst.visible = json_bool(obj, "visible", true);
	if (inst.path.empty()) {
		printf("stage: stage instance '%s' has no path\n", inst.name.c_str());
		return;
	}
	inst.child = new stage::Stage(stage::stage_load(inst.path.c_str()));
	out.stageInstances.emplace_back(std::move(inst));
}

static stage::Camera parse_camera(rapidjson::Value const & obj) {
	stage::Camera cam;
	cam.name = json_string(obj, "name", "camera");
	cam.position = json_f32v3(obj, "position", cam.position);
	cam.target = json_f32v3(obj, "target", cam.target);
	cam.up = json_f32v3(obj, "up", cam.up);
	cam.fovYDegrees = json_f32(obj, "fovYDegrees", cam.fovYDegrees);
	cam.near = json_f32(obj, "near", cam.near);
	cam.far = json_f32(obj, "far", cam.far);
	return cam;
}

static stage::LightInstanceQuad parse_light(rapidjson::Value const & obj) {
	stage::LightInstanceQuad light;
	light.name = json_string(obj, "name", "light");
	light.transform = parse_transform(obj);
	light.width = json_f32(obj, "width", light.width);
	light.height = json_f32(obj, "height", light.height);
	light.radiance = json_f32v3(obj, "radiance", light.radiance);
	light.twoSided = json_bool(obj, "twoSided", light.twoSided);
	return light;
}

static stage::Environment parse_environment(
	rapidjson::Value const & obj, fs::path const & dir
) {
	stage::Environment env;
	if (!obj.IsObject()) { return env; }
	std::string const modeStr = json_string(obj, "mode", "black");
	if (modeStr == "furnace") {
		env.mode = stage::EnvironmentMode::Furnace;
	} else if (modeStr == "checkerboard") {
		env.mode = stage::EnvironmentMode::Checkerboard;
	} else if (modeStr == "hdri") {
		env.mode = stage::EnvironmentMode::Hdri;
	} else {
		env.mode = stage::EnvironmentMode::Black;
	}
	env.path = resolve_path(dir, json_string(obj, "path", ""));
	env.rotationRadians = json_f32(obj, "rotationRadians", 0.0f);
	env.intensity = json_f32(obj, "intensity", 1.0f);
	return env;
}

// -----------------------------------------------------------------------------
// -- transform_to_m44 / stage_load / stage_destroy
// -----------------------------------------------------------------------------

f32m44 stage::transform_to_m44(Transform const & t) {
	return f32m44_compose(t.position, t.rotation, t.scale);
}

f32m44 stage::transform_to_inverse_m44(Transform const & t) {
	f32quat const invRotation = f32quat_normalize({
		-t.rotation.x, -t.rotation.y, -t.rotation.z, t.rotation.w,
	});
	f32v3 const invScale = {
		t.scale.x != 0.0f ? 1.0f / t.scale.x : 0.0f,
		t.scale.y != 0.0f ? 1.0f / t.scale.y : 0.0f,
		t.scale.z != 0.0f ? 1.0f / t.scale.z : 0.0f,
	};
	return (
		f32m44_scale(invScale.x, invScale.y, invScale.z)
		* f32m44_rotate_quat(invRotation)
		* f32m44_translate(-t.position.x, -t.position.y, -t.position.z)
	);
}

// world-space aabb of a local-space aabb under transform; shared by
// gltf_instance_world_bounds and vdb_instance_world_bounds below
static void transform_aabb(
	f32m44 const & transform,
	f32v3 const & localMin,
	f32v3 const & localMax,
	f32v3 & outMin,
	f32v3 & outMax
) {
	f32v4 const corners[8] = {
		transform * f32v4 { localMin.x, localMin.y, localMin.z, 1.0f },
		transform * f32v4 { localMax.x, localMin.y, localMin.z, 1.0f },
		transform * f32v4 { localMin.x, localMax.y, localMin.z, 1.0f },
		transform * f32v4 { localMax.x, localMax.y, localMin.z, 1.0f },
		transform * f32v4 { localMin.x, localMin.y, localMax.z, 1.0f },
		transform * f32v4 { localMax.x, localMin.y, localMax.z, 1.0f },
		transform * f32v4 { localMin.x, localMax.y, localMax.z, 1.0f },
		transform * f32v4 { localMax.x, localMax.y, localMax.z, 1.0f },
	};
	outMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	outMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (f32v4 const & c : corners) {
		outMin.x = std::min(outMin.x, c.x);
		outMin.y = std::min(outMin.y, c.y);
		outMin.z = std::min(outMin.z, c.z);
		outMax.x = std::max(outMax.x, c.x);
		outMax.y = std::max(outMax.y, c.y);
		outMax.z = std::max(outMax.z, c.z);
	}
}

bool stage::gltf_instance_world_bounds(
	Stage const & stage,
	GltfInstance const & instance,
	f32v3 & outMin,
	f32v3 & outMax
) {
	if (instance.modelIndex >= stage.gltfModels.size()) { return false; }
	GltfModel const & model = stage.gltfModels[instance.modelIndex];
	if (model.meshIndex >= stage.gltfMeshes.size()) { return false; }
	GltfMesh const & mesh = stage.gltfMeshes[model.meshIndex];
	if (mesh.scene.id == 0u) { return false; }
	f32v3 localMin, localMax;
	mor::scene_bounds(mesh.scene, localMin, localMax);
	transform_aabb(
		transform_to_m44(instance.transform), localMin, localMax, outMin, outMax
	);
	return true;
}

bool stage::vdb_instance_world_bounds(
	VdbInstance const & instance, f32v3 & outMin, f32v3 & outMax
) {
	f32v3 localMin { FLT_MAX, FLT_MAX, FLT_MAX };
	f32v3 localMax { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	bool any = false;
	for (VdbBlob const & blob : instance.blobs) {
		f32v3 blobMin, blobMax;
		if (!ponder::vdb_world_bounds(blob.vdb, blobMin, blobMax)) { continue; }
		localMin.x = std::min(localMin.x, blobMin.x);
		localMin.y = std::min(localMin.y, blobMin.y);
		localMin.z = std::min(localMin.z, blobMin.z);
		localMax.x = std::max(localMax.x, blobMax.x);
		localMax.y = std::max(localMax.y, blobMax.y);
		localMax.z = std::max(localMax.z, blobMax.z);
		any = true;
	}
	if (!any) { return false; }
	transform_aabb(
		transform_to_m44(instance.transform), localMin, localMax, outMin, outMax
	);
	return true;
}

stage::Stage stage::stage_load(char const * const path) {
	Stage out;
	FILE * const file = fopen(path, "rb");
	if (!file) {
		printf("stage: failed to open '%s'\n", path);
		return out;
	}
	char buffer[16384];
	rapidjson::FileReadStream stream(file, buffer, sizeof(buffer));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(file);
	if (doc.HasParseError() || !doc.IsObject()) {
		printf("stage: failed to parse '%s'\n", path);
		return out;
	}
	u32 const version = (u32)json_f32(doc, "version", (f32)kStageFormatVersion);
	if (version > kStageFormatVersion) {
		printf(
			"stage: '%s' has version %u, newer than supported version %u\n",
			path, version, kStageFormatVersion
		);
		return out;
	}
	out.formatVersion = version;
	fs::path const dir = fs::path(path).parent_path();
	out.sourceDirectory = dir.empty() ? "." : dir.string();
	out.name = json_string(doc, "name", "");
	out.description = json_string(doc, "description", "");
	out.metersPerUnit = json_f32(doc, "metersPerUnit", 1.0f);
	out.activeCamera = json_string(doc, "activeCamera", "");
	if (doc.HasMember("cameras") && doc["cameras"].IsArray()) {
		for (auto const & camVal : doc["cameras"].GetArray()) {
			if (!camVal.IsObject()) { continue; }
			out.cameras.emplace_back(parse_camera(camVal));
		}
	}
	if (doc.HasMember("environment")) {
		out.environment = parse_environment(doc["environment"], dir);
	}
	if (doc.HasMember("lights") && doc["lights"].IsArray()) {
		for (auto const & lightVal : doc["lights"].GetArray()) {
			if (!lightVal.IsObject()) { continue; }
			out.lights.emplace_back(parse_light(lightVal));
		}
	}
	// meshes and models must load before instances -- instances reference
	// them by index
	if (doc.HasMember("gltfMeshes") && doc["gltfMeshes"].IsArray()) {
		for (auto const & meshVal : doc["gltfMeshes"].GetArray()) {
			if (!meshVal.IsObject()) { continue; }
			load_gltf_mesh(meshVal, dir, out);
		}
	}
	if (doc.HasMember("gltfModels") && doc["gltfModels"].IsArray()) {
		for (auto const & modelVal : doc["gltfModels"].GetArray()) {
			if (!modelVal.IsObject()) { continue; }
			load_gltf_model(modelVal, out);
		}
	}
	if (doc.HasMember("instances") && doc["instances"].IsArray()) {
		for (auto const & entry : doc["instances"].GetArray()) {
			if (!entry.IsObject()) { continue; }
			std::string const type = json_string(entry, "type", "");
			if (type == "gltf") {
				load_gltf_instance(entry, dir, out);
			} else if (type == "vdb") {
				load_vdb_instance(entry, dir, out);
			} else if (type == "alembic") {
				parse_alembic_instance(entry, dir, out);
			} else if (type == "stage") {
				load_stage_instance(entry, dir, out);
			} else {
				printf("stage: unknown instance type '%s'\n", type.c_str());
			}
		}
	}
	return out;
}

void stage::stage_destroy(Stage & s) {
	for (mor::GpuMaterials const & m : s.pendingGpuMaterialsDestroy) {
		mor::scene_gpu_materials_destroy(m);
	}
	for (Stage::PendingMeshDestroy const & m : s.pendingGltfMeshDestroy) {
		if (m.gpuScene.id != 0u) { mor::scene_gpu_destroy(m.gpuScene); }
		if (m.scene.id != 0u) { mor::scene_destroy(m.scene); }
	}
	for (GltfModel & model : s.gltfModels) {
		if (model.materials.id != 0u) {
			mor::scene_gpu_materials_destroy(model.materials);
		}
	}
	for (GltfMesh & mesh : s.gltfMeshes) {
		if (mesh.gpuScene.id != 0u) {
			mor::scene_gpu_destroy(mesh.gpuScene);
		}
		if (mesh.scene.id != 0u) {
			mor::scene_destroy(mesh.scene);
		}
	}
	for (VdbInstance & inst : s.vdbInstances) {
		for (VdbBlob & blob : inst.blobs) {
			if (blob.vdb.gpuBuffer.id != 0u) {
				vkof::buffer_destroy(blob.vdb.gpuBuffer);
			}
			if (blob.vdb.handleBuffer.id != 0u) {
				vkof::buffer_destroy(blob.vdb.handleBuffer);
			}
		}
	}
	for (StageInstance & inst : s.stageInstances) {
		if (inst.child == nullptr) { continue; }
		stage_destroy(*inst.child);
		delete inst.child;
		inst.child = nullptr;
	}
	s = {};
}

void stage::stage_remove_gltf_instance(Stage & s, usize const index) {
	if (index >= s.gltfInstances.size()) { return; }
	// an instance owns no gpu resources of its own (model/mesh do, shared --
	// see stage_remove_gltf_model/stage_remove_gltf_mesh), so no
	// device_wait_idle/teardown needed here, just the erase
	s.gltfInstances.erase(s.gltfInstances.begin() + index);
}

void stage::stage_remove_gltf_model(Stage & s, usize const index) {
	if (index >= s.gltfModels.size()) { return; }
	for (GltfInstance const & inst : s.gltfInstances) {
		if (inst.modelIndex == (u32)index) {
			printf(
				"stage: refusing to remove gltf model '%s' -- still referenced "
				"by instance '%s'\n",
				s.gltfModels[index].name.c_str(), inst.name.c_str()
			);
			return;
		}
	}
	GltfModel & model = s.gltfModels[index];
	if (model.materials.id != 0u) {
		s.pendingGpuMaterialsDestroy.emplace_back(model.materials);
	}
	s.gltfModels.erase(s.gltfModels.begin() + index);
	// every remaining instance whose modelIndex pointed past the removed
	// entry now needs to shift down by one to keep pointing at the same
	// model (a plain vector erase, so every entry past index shifted)
	for (GltfInstance & inst : s.gltfInstances) {
		if (inst.modelIndex > (u32)index) { inst.modelIndex--; }
	}
}

void stage::stage_remove_gltf_mesh(Stage & s, usize const index) {
	if (index >= s.gltfMeshes.size()) { return; }
	for (GltfModel const & model : s.gltfModels) {
		if (model.meshIndex == (u32)index) {
			printf(
				"stage: refusing to remove gltf mesh '%s' -- still referenced "
				"by model '%s'\n",
				s.gltfMeshes[index].path.c_str(), model.name.c_str()
			);
			return;
		}
	}
	GltfMesh & mesh = s.gltfMeshes[index];
	s.pendingGltfMeshDestroy.emplace_back(
		Stage::PendingMeshDestroy { mesh.gpuScene, mesh.scene }
	);
	s.gltfMeshes.erase(s.gltfMeshes.begin() + index);
	for (GltfModel & model : s.gltfModels) {
		if (model.meshIndex > (u32)index) { model.meshIndex--; }
	}
}

void stage::stage_flush_pending_gpu_destroys(Stage & s) {
	bool const any = (
		!s.pendingGpuMaterialsDestroy.empty() || !s.pendingGltfMeshDestroy.empty()
	);
	if (any) {
		vkof::device_wait_idle();
		for (mor::GpuMaterials const & m : s.pendingGpuMaterialsDestroy) {
			mor::scene_gpu_materials_destroy(m);
		}
		s.pendingGpuMaterialsDestroy.clear();
		for (Stage::PendingMeshDestroy const & m : s.pendingGltfMeshDestroy) {
			if (m.gpuScene.id != 0u) { mor::scene_gpu_destroy(m.gpuScene); }
			if (m.scene.id != 0u) { mor::scene_destroy(m.scene); }
		}
		s.pendingGltfMeshDestroy.clear();
	}
	for (StageInstance & inst : s.stageInstances) {
		if (inst.child != nullptr) { stage_flush_pending_gpu_destroys(*inst.child); }
	}
}

void stage::stage_remove_vdb_instance(Stage & s, usize const index) {
	if (index >= s.vdbInstances.size()) { return; }
	vkof::device_wait_idle();
	for (VdbBlob & blob : s.vdbInstances[index].blobs) {
		if (blob.vdb.gpuBuffer.id != 0u) {
			vkof::buffer_destroy(blob.vdb.gpuBuffer);
		}
		if (blob.vdb.handleBuffer.id != 0u) {
			vkof::buffer_destroy(blob.vdb.handleBuffer);
		}
	}
	s.vdbInstances.erase(s.vdbInstances.begin() + index);
}

i32 stage::stage_find_or_add_gltf_mesh(Stage & stage, char const * const path) {
	fs::path const dir = (
		stage.sourceDirectory.empty() ? fs::current_path()
			: fs::path(stage.sourceDirectory)
	);
	std::string const resolved = resolve_path(dir, path);
	if (resolved.empty()) { return -1; }
	for (usize i = 0u; i < stage.gltfMeshes.size(); ++i) {
		if (stage.gltfMeshes[i].path == resolved) { return (i32)i; }
	}
	GltfMesh mesh;
	mesh.path = resolved;
	mesh.scene = mor::scene_create();
	mor::scene_load_gltf(mesh.scene, mesh.path.c_str());
	mesh.gpuScene = mor::scene_gpu_upload(mesh.scene);
	if (mesh.gpuScene.id == 0u) {
		mor::scene_destroy(mesh.scene);
		return -1;
	}
	stage.gltfMeshes.emplace_back(std::move(mesh));
	return (i32)(stage.gltfMeshes.size() - 1u);
}

i32 stage::stage_add_gltf_model(
	Stage & stage,
	char const * const path,
	char const * const name,
	i32 const meshIndexIn
) {
	i32 const meshIndex = (
		meshIndexIn >= 0 ? meshIndexIn : stage_find_or_add_gltf_mesh(stage, path)
	);
	if (meshIndex < 0 || (usize)meshIndex >= stage.gltfMeshes.size()) {
		return -1;
	}
	GltfModel model;
	model.name = (
		name != nullptr ? name : fs::path(stage.gltfMeshes[meshIndex].path).stem().string()
	);
	model.meshIndex = (u32)meshIndex;
	GltfMesh const & mesh = stage.gltfMeshes[meshIndex];
	if (mesh.scene.id != 0u) {
		model.materials = mor::scene_gpu_materials_create(mesh.scene);
	}
	stage.gltfModels.emplace_back(std::move(model));
	return (i32)(stage.gltfModels.size() - 1u);
}

i32 stage::stage_duplicate_gltf_model(
	Stage & stage, usize const modelIndex, char const * const name
) {
	if (modelIndex >= stage.gltfModels.size()) { return -1; }
	GltfModel const & source = stage.gltfModels[modelIndex];
	if (source.meshIndex >= stage.gltfMeshes.size()) { return -1; }
	GltfMesh const & mesh = stage.gltfMeshes[source.meshIndex];
	GltfModel copy;
	copy.name = name != nullptr ? name : (source.name + " copy");
	copy.meshIndex = source.meshIndex;
	copy.materialOverrides = source.materialOverrides;
	if (mesh.scene.id != 0u) {
		copy.materials = mor::scene_gpu_materials_create(mesh.scene);
		stage_gltf_model_apply_overrides(copy, mesh);
	}
	stage.gltfModels.emplace_back(std::move(copy));
	return (i32)(stage.gltfModels.size() - 1u);
}

bool stage::stage_add_gltf_instance(
	Stage & stage,
	usize const modelIndex,
	char const * const name,
	Transform const & transform
) {
	if (modelIndex >= stage.gltfModels.size()) { return false; }
	GltfInstance inst;
	inst.name = name != nullptr ? name : stage.gltfModels[modelIndex].name;
	inst.modelIndex = (u32)modelIndex;
	inst.transform = transform;
	stage.gltfInstances.emplace_back(std::move(inst));
	return true;
}

bool stage::stage_add_gltf(
	Stage & stage, char const * const path, char const * const name
) {
	i32 const modelIndex = stage_add_gltf_model(stage, path, name);
	if (modelIndex < 0) { return false; }
	return stage_add_gltf_instance(stage, (usize)modelIndex, name);
}

bool stage::stage_add_vdb_instance(
	Stage & stage,
	char const * const path,
	char const * const gridName,
	char const * const name
) {
	fs::path const dir = (
		stage.sourceDirectory.empty() ? fs::current_path()
			: fs::path(stage.sourceDirectory)
	);
	VdbInstance inst;
	inst.name = name != nullptr ? name : fs::path(path).stem().string();
	inst.path = resolve_path(dir, path);
	if (inst.path.empty()) { return false; }
	VdbBlob blob;
	blob.gridName = gridName;
	blob.vdb = ponder::vdb_load(inst.path.c_str(), gridName);
	if (blob.vdb.gpuBuffer.id == 0u) { return false; }
	inst.blobs.emplace_back(std::move(blob));
	stage.vdbInstances.emplace_back(std::move(inst));
	return true;
}

// -----------------------------------------------------------------------------
// -- stage_save
// -----------------------------------------------------------------------------

static void write_f32v3(JsonWriter & w, f32v3 const & v) {
	w.StartArray();
	w.Double((double)v.x);
	w.Double((double)v.y);
	w.Double((double)v.z);
	w.EndArray();
}

static void write_transform(JsonWriter & w, stage::Transform const & t) {
	w.Key("transform");
	w.StartObject();
	w.Key("position");
	write_f32v3(w, t.position);
	w.Key("rotation");
	w.StartArray();
	w.Double((double)t.rotation.x);
	w.Double((double)t.rotation.y);
	w.Double((double)t.rotation.z);
	w.Double((double)t.rotation.w);
	w.EndArray();
	w.Key("scale");
	write_f32v3(w, t.scale);
	w.EndObject();
}

static void write_material_overrides(
	JsonWriter & w, std::vector<stage::MaterialOverride> const & overrides
) {
	w.Key("materialOverrides");
	w.StartArray();
	for (stage::MaterialOverride const & mo : overrides) {
		w.StartObject();
		w.Key("material");
		w.String(mo.material.c_str());
		w.Key("fields");
		w.StartObject();
		for (stage::MaterialFieldOverride const & f : mo.fields) {
			w.Key(f.field.c_str());
			w.StartObject();
			w.Key("value");
			if (material_field_is_color(f.field)) {
				write_f32v3(w, f.colorValue);
			} else {
				w.Double((double)f.scalarValue);
			}
			w.Key("texturePath");
			w.String(f.texturePath.c_str());
			w.EndObject();
		}
		w.EndObject();
		w.EndObject();
	}
	w.EndArray();
}

static void write_gltf_mesh(
	JsonWriter & w, stage::GltfMesh const & mesh, fs::path const & saveDir
) {
	w.StartObject();
	w.Key("path");
	w.String(rebase_path(saveDir, mesh.path).c_str());
	w.EndObject();
}

static void write_gltf_model(JsonWriter & w, stage::GltfModel const & model) {
	w.StartObject();
	w.Key("name");
	w.String(model.name.c_str());
	w.Key("meshIndex");
	w.Uint(model.meshIndex);
	write_material_overrides(w, model.materialOverrides);
	w.EndObject();
}

static void write_gltf_instance(
	JsonWriter & w, stage::GltfInstance const & inst
) {
	w.StartObject();
	w.Key("type");
	w.String("gltf");
	w.Key("name");
	w.String(inst.name.c_str());
	w.Key("modelIndex");
	w.Uint(inst.modelIndex);
	w.Key("visible");
	w.Bool(inst.visible);
	write_transform(w, inst.transform);
	w.EndObject();
}

static void write_vdb_instance(
	JsonWriter & w, stage::VdbInstance const & inst, fs::path const & saveDir
) {
	w.StartObject();
	w.Key("type");
	w.String("vdb");
	w.Key("name");
	w.String(inst.name.c_str());
	w.Key("path");
	w.String(rebase_path(saveDir, inst.path).c_str());
	w.Key("visible");
	w.Bool(inst.visible);
	write_transform(w, inst.transform);
	w.Key("blobs");
	w.StartArray();
	for (stage::VdbBlob const & blob : inst.blobs) {
		w.StartObject();
		w.Key("grid");
		w.String(blob.gridName.c_str());
		w.Key("sigmaAbsorption");
		write_f32v3(w, blob.sigmaAbsorption);
		w.Key("sigmaScattering");
		write_f32v3(w, blob.sigmaScattering);
		w.Key("dropletDiameter");
		w.Double((double)blob.dropletDiameter);
		w.Key("phaseAnisotropyOverride");
		w.Double((double)blob.phaseAnisotropyOverride);
		w.Key("temperatureScale");
		w.Double((double)blob.temperatureScale);
		w.Key("emissionScale");
		w.Double((double)blob.emissionScale);
		w.EndObject();
	}
	w.EndArray();
	w.EndObject();
}

static void write_alembic_instance(
	JsonWriter & w,
	stage::AlembicInstance const & inst,
	fs::path const & saveDir
) {
	w.StartObject();
	w.Key("type");
	w.String("alembic");
	w.Key("name");
	w.String(inst.name.c_str());
	w.Key("path");
	w.String(rebase_path(saveDir, inst.path).c_str());
	w.Key("visible");
	w.Bool(inst.visible);
	write_transform(w, inst.transform);
	w.Key("objectPath");
	w.String(inst.objectPath.c_str());
	w.Key("time");
	w.Double((double)inst.time);
	w.EndObject();
}

static void write_stage_instance(
	JsonWriter & w, stage::StageInstance const & inst, fs::path const & saveDir
) {
	w.StartObject();
	w.Key("type");
	w.String("stage");
	w.Key("name");
	w.String(inst.name.c_str());
	w.Key("path");
	w.String(rebase_path(saveDir, inst.path).c_str());
	w.Key("visible");
	w.Bool(inst.visible);
	write_transform(w, inst.transform);
	w.EndObject();
}

static void write_camera(JsonWriter & w, stage::Camera const & cam) {
	w.StartObject();
	w.Key("name");
	w.String(cam.name.c_str());
	w.Key("position");
	write_f32v3(w, cam.position);
	w.Key("target");
	write_f32v3(w, cam.target);
	w.Key("up");
	write_f32v3(w, cam.up);
	w.Key("fovYDegrees");
	w.Double((double)cam.fovYDegrees);
	w.Key("near");
	w.Double((double)cam.near);
	w.Key("far");
	w.Double((double)cam.far);
	w.EndObject();
}

static void write_light(
	JsonWriter & w, stage::LightInstanceQuad const & light
) {
	w.StartObject();
	w.Key("name");
	w.String(light.name.c_str());
	write_transform(w, light.transform);
	w.Key("width");
	w.Double((double)light.width);
	w.Key("height");
	w.Double((double)light.height);
	w.Key("radiance");
	write_f32v3(w, light.radiance);
	w.Key("twoSided");
	w.Bool(light.twoSided);
	w.EndObject();
}

static char const * environment_mode_string(stage::EnvironmentMode const mode) {
	switch (mode) {
		case stage::EnvironmentMode::Furnace: return "furnace";
		case stage::EnvironmentMode::Checkerboard: return "checkerboard";
		case stage::EnvironmentMode::Hdri: return "hdri";
		case stage::EnvironmentMode::Black: return "black";
	}
	return "black";
}

static void write_environment(
	JsonWriter & w, stage::Environment const & env, fs::path const & saveDir
) {
	w.Key("environment");
	w.StartObject();
	w.Key("mode");
	w.String(environment_mode_string(env.mode));
	w.Key("path");
	w.String(rebase_path(saveDir, env.path).c_str());
	w.Key("rotationRadians");
	w.Double((double)env.rotationRadians);
	w.Key("intensity");
	w.Double((double)env.intensity);
	w.EndObject();
}

// note: single-level only -- a stageInstance's referenced file is
// re-referenced by (rebased) path, never re-saved itself. saving an edited
// child stage requires calling stage_save on that child directly
bool stage::stage_save(Stage const & s, char const * const path) {
	fs::path const saveDir = fs::path(path).parent_path();
	rapidjson::StringBuffer buffer;
	JsonWriter writer(buffer);
	writer.StartObject();
	writer.Key("version");
	writer.Uint(s.formatVersion);
	writer.Key("name");
	writer.String(s.name.c_str());
	writer.Key("description");
	writer.String(s.description.c_str());
	writer.Key("metersPerUnit");
	writer.Double((double)s.metersPerUnit);
	writer.Key("activeCamera");
	writer.String(s.activeCamera.c_str());
	writer.Key("cameras");
	writer.StartArray();
	for (Camera const & cam : s.cameras) { write_camera(writer, cam); }
	writer.EndArray();
	write_environment(writer, s.environment, saveDir);
	writer.Key("lights");
	writer.StartArray();
	for (LightInstanceQuad const & light : s.lights) {
		write_light(writer, light);
	}
	writer.EndArray();
	writer.Key("gltfMeshes");
	writer.StartArray();
	for (GltfMesh const & mesh : s.gltfMeshes) {
		write_gltf_mesh(writer, mesh, saveDir);
	}
	writer.EndArray();
	writer.Key("gltfModels");
	writer.StartArray();
	for (GltfModel const & model : s.gltfModels) {
		write_gltf_model(writer, model);
	}
	writer.EndArray();
	writer.Key("instances");
	writer.StartArray();
	for (GltfInstance const & inst : s.gltfInstances) {
		write_gltf_instance(writer, inst);
	}
	for (VdbInstance const & inst : s.vdbInstances) {
		write_vdb_instance(writer, inst, saveDir);
	}
	for (AlembicInstance const & inst : s.alembicInstances) {
		write_alembic_instance(writer, inst, saveDir);
	}
	for (StageInstance const & inst : s.stageInstances) {
		write_stage_instance(writer, inst, saveDir);
	}
	writer.EndArray();
	writer.EndObject();
	FILE * const file = fopen(path, "wb");
	if (!file) {
		printf("stage: failed to open '%s' for write\n", path);
		return false;
	}
	size_t const written = (
		fwrite(buffer.GetString(), 1, buffer.GetSize(), file)
	);
	fclose(file);
	return written == buffer.GetSize();
}
