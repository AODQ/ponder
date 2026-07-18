// ponder viewer: opens a gltf/glb file via mor and renders it with the
// lib/ponder OpenPBR path tracer (app/shaders/resolve.comp +
// pt-accumulate.comp). windowed by default; --screenshot renders headless
// for --spp frames and writes an hdr exr instead.

#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>

#include <ponder/bluenoise.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/vdb.hpp>
#include <ponder/zeltner-tables.hpp>

#include <stage/stage.hpp>

// cpu-side reads of the loaded grid blob (world bbox for camera framing)
#include <nanovdb/NanoVDB.h>

#include "shaders/resolve_pc.h"
#include "shaders/pt-accumulate-pc.h"
#include "shaders/nan-check-pc.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// scroll wheel arrives via callback, consumed once per frame; chained into
// imgui's own handler since installing this callback replaces the one
// imgui's glfw backend registered at init
f32 sScrollDelta = 0.0f;
void scroll_callback(GLFWwindow * const w, double const x, double const y) {
	ImGui_ImplGlfw_ScrollCallback(w, x, y);
	sScrollDelta += (f32)y;
}

// mirrors util-environment-map.glsl's EnvironmentMapHandles layout exactly;
// GpuGlobalExtended.envMap (global_pc.h) is a VA into a buffer holding one
// of these, uploaded once alongside ponder::EnvironmentTables
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 conditionalCdf;
	u32 marginalCdf;
};

// internal render resolution presets for the imgui dropdown; the scene
// renders at this size regardless of the window size (the finalImage blit
// scales to the swapchain)
struct RenderResPreset {
	char const * label;
	u32 w;
	u32 h;
};
constexpr RenderResPreset skRenderResPresets[] = {
	{ "480p", 854u, 480u },
	{ "720p", 1280u, 720u },
	{ "1080p", 1920u, 1080u },
	{ "1440p", 2560u, 1440u },
	{ "2160p", 3840u, 2160u },
	{ "4320p", 7680u, 4320u },
};

// material inspector popup: lets a texture slot be detached ("[none]") or
// swapped for any other texture already used elsewhere on this material --
// not a full asset browser, just cross-wiring between handles the material
// already references, which is all a gltf-sourced material ever has loaded
bool fnMaterialTexturePicker(
	u32 & textureHandle,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles
) {
	std::string texName = "[none]";
	{
		auto const it = (
			std::find(textureHandles.begin(), textureHandles.end(), textureHandle)
		);
		if (it != textureHandles.end()) {
			size_t const idx = std::distance(textureHandles.begin(), it);
			if (idx < textureNames.size()) {
				texName = textureNames[idx];
			}
		}
	}

	bool changed = false;
	if (ImGui::BeginCombo("##texture", texName.c_str())) {
		if (ImGui::Selectable("[none]", textureHandle == 0u)) {
			if (textureHandle != 0u) {
				textureHandle = 0u;
				changed = true;
			}
		}
		for (size_t i = 0u; i < textureNames.size(); ++i) {
			bool const selected = (textureHandles[i] == textureHandle);
			if (ImGui::Selectable(textureNames[i].c_str(), selected) && !selected) {
				textureHandle = textureHandles[i];
				changed = true;
			}
			if (selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	return changed;
}

// one editable row: rgb color plus its texture slot. minVal/maxVal are
// unused here -- they exist so the MOR_MATERIAL_ALL_PARAMS_* x-macros
// (mor-shared.h) can pass every field's (lo, hi) uniformly and overload
// resolution alone picks this or the scalar overload below
bool fnDrawMaterialSlot(
	GpuMorMaterialComponent3 & slot,
	char const * const label,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles,
	[[maybe_unused]] f32 const minVal = 0.0f,
	[[maybe_unused]] f32 const maxVal = 1.0f
) {
	bool changed = false;
	ImGui::PushID(label);
	ImGui::Text("%s", label);
	changed |= ImGui::ColorEdit3("##value", &slot.rgb.x);
	changed |= (
		fnMaterialTexturePicker(slot.texture, textureNames, textureHandles)
	);
	ImGui::PopID();
	return changed;
}

// one editable row: scalar slider plus its texture slot and, once a texture
// is attached, which channel of it to read
bool fnDrawMaterialSlot(
	GpuMorMaterialComponent1 & slot,
	char const * const label,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles,
	f32 const minVal,
	f32 const maxVal
) {
	bool changed = false;
	ImGui::PushID(label);
	ImGui::Text("%s", label);
	changed |= ImGui::SliderFloat("##value", &slot.r, minVal, maxVal);
	changed |= (
		fnMaterialTexturePicker(slot.texture, textureNames, textureHandles)
	);
	if (slot.texture != 0u) {
		i32 const prevSwizzle = slot.swizzle;
		ImGui::RadioButton("R", &slot.swizzle, 0);
		ImGui::SameLine();
		ImGui::RadioButton("G", &slot.swizzle, 1);
		ImGui::SameLine();
		ImGui::RadioButton("B", &slot.swizzle, 2);
		ImGui::SameLine();
		ImGui::RadioButton("A", &slot.swizzle, 3);
		changed |= (slot.swizzle != prevSwizzle);
	}
	ImGui::PopID();
	return changed;
}

// every texture handle the material's slots reference, deduplicated and
// named after the slot that first uses it -- the picker's dropdown
// contents. built from the original, unedited material so slot names stay
// meaningful even after edits reassign a texture to a different slot
void fnMaterialTextureList(
	std::vector<std::string> & outNames,
	std::vector<u32> & outHandles,
	GpuMorMaterial const & origMat
) {
	auto const fnAddSlot = [&](char const * const name, u32 const handle) {
		if (handle == 0u) {
			return;
		}
		auto const it = (
			std::find(outHandles.begin(), outHandles.end(), handle)
		);
		if (it != outHandles.end()) {
			return;
		}
		outNames.emplace_back(name);
		outHandles.emplace_back(handle);
	};
#define X(Type, field, lo, hi) fnAddSlot(#field, origMat.field.texture);
	MOR_MATERIAL_ALL_PARAMS(X)
#undef X
}

// editable material panel: one collapsible section per OpenPBR layer, only
// shown once that layer's weight is nonzero (or texture-driven, so a
// zero-constant weight with a texture plugged in still exposes its
// parameters). every scalar/rgb parameter comes from mor-shared.h's
// MOR_MATERIAL_ALL_PARAMS_* x-macros, so a new material parameter shows up
// automatically rather than needing a second hand-maintained field list
bool fnImguiMaterial(GpuMorMaterial & p, GpuMorMaterial const & origMat) {
	std::vector<std::string> textureNames;
	std::vector<u32> textureHandles;
	fnMaterialTextureList(textureNames, textureHandles, origMat);

	bool changed = false;
	ImGui::SeparatorText("weights");
#define X(Type, field, lo, hi) \
	changed |= ( \
		fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
	);
	MOR_MATERIAL_ALL_PARAMS_WEIGHTS(X)
#undef X
	if (p.baseWeight.r > 0.0f || p.baseWeight.texture != 0u) {
		ImGui::SeparatorText("base");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_BASE(X)
#undef X
	}
	if (p.specularWeight.r > 0.0f || p.specularWeight.texture != 0u) {
		ImGui::SeparatorText("specular");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_SPECULAR(X)
#undef X
	}
	if (p.transmissionWeight.r > 0.0f || p.transmissionWeight.texture != 0u) {
		ImGui::SeparatorText("transmission");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_TRANSMISSION(X)
#undef X
	}
	if (p.subsurfaceWeight.r > 0.0f || p.subsurfaceWeight.texture != 0u) {
		ImGui::SeparatorText("subsurface");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_SUBSURFACE(X)
#undef X
	}
	if (p.coatWeight.r > 0.0f || p.coatWeight.texture != 0u) {
		ImGui::SeparatorText("coat");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_COAT(X)
#undef X
	}
	if (p.fuzzWeight.r > 0.0f || p.fuzzWeight.texture != 0u) {
		ImGui::SeparatorText("fuzz");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_FUZZ(X)
#undef X
	}
	if (p.emissionLuminance.r > 0.0f || p.emissionLuminance.texture != 0u) {
		ImGui::SeparatorText("emission");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_EMISSION(X)
#undef X
	}
	if (p.thinFilmWeight.r > 0.0f || p.thinFilmWeight.texture != 0u) {
		ImGui::SeparatorText("thin film");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
		);
		MOR_MATERIAL_ALL_PARAMS_THIN_FILM(X)
#undef X
	}
	ImGui::SeparatorText("geometry");
#define X(Type, field, lo, hi) \
	changed |= ( \
		fnDrawMaterialSlot(p.field, #field, textureNames, textureHandles, lo, hi) \
	);
	MOR_MATERIAL_ALL_PARAMS_GEOMETRY(X)
#undef X
	ImGui::Text("alphaCutoff");
	changed |= ImGui::SliderFloat("##acutoff", &p.alphaCutoff, 0.0f, 1.0f);
	return changed;
}

// materialsOverride starts as an exact copy of the scene's materials (see
// scene_gpu_materials_create), so editing it never touches the gltf-sourced
// mor::Scene -- origMat (looked up there) is only used to label texture
// slots against the material's pre-edit state
bool fnDrawMaterialInspector(
	mor::Scene const & scene,
	mor::GpuMaterials const & materialsOverride,
	i32 const materialIndex
) {
	if (
		materialIndex < 0
		|| (u32)materialIndex >= mor::scene_material_count(scene)
	) {
		ImGui::TextDisabled("no material at that pixel");
		return false;
	}
	std::string const name = (
		mor::scene_material_name(scene, (u32)materialIndex)
	);
	ImGui::Text("#%d %s", materialIndex, name.c_str());
	ImGui::Separator();
	GpuMorMaterial mat = (
		mor::scene_gpu_materials_get(materialsOverride, (u32)materialIndex)
	);
	GpuMorMaterial const origMat = (
		mor::scene_material_get(scene, (u32)materialIndex)
	);
	bool const changed = fnImguiMaterial(mat, origMat);
	if (changed) {
		mor::scene_material_override_scalars(
			materialsOverride, (u32)materialIndex, mat
		);
	}
	return changed;
}

struct ViewerArgs {
	char const * gltfPath = nullptr;
	char const * screenshotPath = nullptr;
	char const * vdbPath = nullptr;
	u32 width = 1280u;
	u32 height = 720u;
	// accumulation frames rendered before a --screenshot is taken
	u32 spp = 256u;
	bool cameraOverride = false;
	f32 cameraAzimuth = 0.0f;
	f32 cameraElevation = 0.3f;
	f32 cameraDistance = 0.0f;
	f32 cameraFovY = 1.0f;
	f32 envIntensity = 1.0f;
	// ENV_MODE_* in environment.glsl: 0=furnace, 1=checkerboard, 2=black,
	// 3=hdrmap
	i32 envMode = 0;
	// true once --env-mode is explicitly passed, so --env-exr's
	// auto-select-hdrmap default knows not to override it
	bool envModeExplicit = false;
	char const * envExrPath = nullptr;
	// env-map next-event estimation default; see --env-mis
	bool envMis = true;
	// <= 0 disables; see --firefly-clamp
	f32 fireflyClampLuminance = 20.0f;
	// > 0 starts with the frame budget enabled at this target; mainly for
	// exercising the tiled dispatch path headless, where there is no imgui
	// toggle (a budgeted --screenshot accumulates fewer samples per pixel
	// than --spp names, since each frame only refreshes a tile window)
	f32 budgetMs = 0.0f;
	// vulkan validation layers (debug_printf capture, validation_message --
	// see vkof::init) off by default: meaningful chunk of windowed startup
	// time most interactive use doesn't need. headless (--screenshot)
	// always has it on regardless, since the numeric tests depend on it.
	// the material inspector probe no longer needs this -- see
	// GpuProbeResult, a plain host-readable buffer instead of debug_printf
	bool debug = false;
	// starts in stage edit mode with this file loaded, bypassing the
	// gltf/vdb-required startup gate entirely; mutually exclusive with
	// newStage below (parse_args rejects both)
	char const * stagePath = nullptr;
	// starts in stage edit mode with a fresh, empty, unsaved stage
	bool newStage = false;
};

void print_usage() {
	printf(
		"usage: viewer [path.gltf|.glb]\n"
		"  [--screenshot <out.png>]  render headless, write png, exit\n"
		"  [--camera <azimuth,elevation,distance,fovY>]  orbit camera\n"
		"      (radians / world units; distance 0 = auto-fit)\n"
		"  [--resolution <w>x<h>]  render resolution (default 1280x720)\n"
		"  [--spp <n>]  accumulation frames before --screenshot (default 256)\n"
		"  [--vdb <path.vdb>]  load a density grid\n"
		"  [--env-intensity <f>]  environment radiance (default 1.0)\n"
		"  [--env-mode <furnace|checkerboard|black|hdrmap>]  (default furnace)\n"
		"  [--env-exr <path.exr>]  load an exr environment map; implies\n"
		"      --env-mode hdrmap unless --env-mode is also given\n"
		"  [--env-mis <0|1>]  env map next-event estimation (default 1)\n"
		"  [--firefly-clamp <f>]  cap a sample's luminance before it's\n"
		"      written out; <= 0 disables (default 20.0)\n"
		"  [--budget-ms <f>]  start with the resolve frame budget enabled\n"
		"      at this gpu-ms target; <= 0 disables (default off)\n"
		"  [--debug]  enable vulkan validation layers (slower startup;\n"
		"      needed for NaN probes / validation messages -- always on\n"
		"      with --screenshot)\n"
		"  [--stage <path.json>]  start in stage edit mode with this file\n"
		"      loaded (skips the gltf/vdb-required startup check)\n"
		"  [--new-stage]  start in stage edit mode with a fresh, empty,\n"
		"      unsaved stage\n"
	);
}

bool parse_args(i32 const argc, char const * const * const argv, ViewerArgs & out) {
	for (i32 i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc) {
			out.screenshotPath = argv[++i];
		} else if (strcmp(argv[i], "--camera") == 0 && i + 1 < argc) {
			f32 az, el, dist, fov;
			if (
				sscanf(argv[++i], "%f,%f,%f,%f", &az, &el, &dist, &fov) != 4
			) {
				printf("--camera expects azimuth,elevation,distance,fovY\n");
				return false;
			}
			out.cameraOverride = true;
			out.cameraAzimuth = az;
			out.cameraElevation = el;
			out.cameraDistance = dist;
			out.cameraFovY = fov;
		} else if (strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
			u32 w, h;
			if (sscanf(argv[++i], "%ux%u", &w, &h) != 2 || w == 0u || h == 0u) {
				printf("--resolution expects <w>x<h>\n");
				return false;
			}
			out.width = w;
			out.height = h;
		} else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
			out.spp = (u32)atoi(argv[++i]);
		} else if (strcmp(argv[i], "--env-intensity") == 0 && i + 1 < argc) {
			out.envIntensity = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--vdb") == 0 && i+1 < argc) {
			out.vdbPath = argv[++i];
		} else if (strcmp(argv[i], "--env-mode") == 0 && i + 1 < argc) {
			char const * const mode = argv[++i];
			out.envModeExplicit = true;
			if (strcmp(mode, "furnace") == 0) {
				out.envMode = 0;
			} else if (strcmp(mode, "checkerboard") == 0) {
				out.envMode = 1;
			} else if (strcmp(mode, "black") == 0) {
				out.envMode = 2;
			} else if (strcmp(mode, "hdrmap") == 0) {
				out.envMode = 3;
			} else {
				printf(
					"--env-mode expects furnace, checkerboard, black, or hdrmap\n"
				);
				return false;
			}
		} else if (strcmp(argv[i], "--env-exr") == 0 && i + 1 < argc) {
			out.envExrPath = argv[++i];
		} else if (strcmp(argv[i], "--env-mis") == 0 && i + 1 < argc) {
			out.envMis = atoi(argv[++i]) != 0;
		} else if (strcmp(argv[i], "--firefly-clamp") == 0 && i + 1 < argc) {
			out.fireflyClampLuminance = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--budget-ms") == 0 && i + 1 < argc) {
			out.budgetMs = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--debug") == 0) {
			out.debug = true;
		} else if (strcmp(argv[i], "--stage") == 0 && i + 1 < argc) {
			out.stagePath = argv[++i];
		} else if (strcmp(argv[i], "--new-stage") == 0) {
			out.newStage = true;
		} else if (argv[i][0] == '-') {
			printf("unknown option '%s'\n", argv[i]);
			return false;
		} else {
			out.gltfPath = argv[i];
		}
	}
	if (out.stagePath != nullptr && out.newStage) {
		printf("--stage and --new-stage are mutually exclusive\n");
		return false;
	}
	// windowed mode can fall back to the settings.json model library, but a
	// headless screenshot needs to know exactly what to render -- a model,
	// a vdb volume alone, or both. --stage/--new-stage sidestep this
	// entirely (headless --screenshot doesn't support stage mode -- there's
	// nothing to interactively edit without a window)
	if (
		out.screenshotPath != nullptr
		&& out.gltfPath == nullptr
		&& out.vdbPath == nullptr
	) {
		return false;
	}
	if (out.envExrPath != nullptr && !out.envModeExplicit) {
		out.envMode = 3;
	}
	return true;
}

// everything derived from one loaded model, torn down wholesale on a model
// switch -- nothing scene-related survives a load (per-frame staging
// buffers included), so no stale state can leak between models
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

ponder::Vdb sVdb {};

// (re)builds s.tlas from scratch with the given single-instance transform;
// a zero blas (the --vdb-only empty scene) builds a zero-instance tlas
// instead, so every camera ray misses straight to the environment. safe to
// call repeatedly on the same tlas handle -- vkof::tlas_build always issues
// a fresh BUILD (no srcAccelerationStructure), never an in-place update
void scene_state_rebuild_tlas(SceneState & s, f32m44 const & transform) {
	bool const hasInstance = s.blas.id != 0u;
	vkof::TlasInstance const instance = {
		.blas = s.blas,
		.transform = transform,
		.instanceCustomIndex = 0u,
		.rayMask = 0xFFu,
	};
	vkof::RenderNode const tlasNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = tlasNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd,
				s.tlas,
				hasInstance
					? srat::slice<vkof::TlasInstance const>(&instance, 1u)
					: srat::slice<vkof::TlasInstance const>(nullptr, 0u)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&tlasNode, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(tlasNode);
}

// loads a gltf/glb, uploads it, builds its blas + single-instance tlas, and
// binds the tlas; blas.id == 0 on failure (partial state already destroyed).
// a null path builds an empty scene instead: no blas, a zero-instance tlas
// (every camera ray misses straight to the environment), and zeroed model
// buffers nothing dereferences on a miss -- used by --vdb without a model so
// the volume renders alone
SceneState scene_state_create(char const * const path) {
	SceneState s;
	if (path == nullptr) {
		s.boundsCenter = f32v3 { 0.0f, 0.0f, 0.0f };
		s.boundsExtent = 1.0f;
		s.tlas = vkof::tlas_create({ .maxInstances = 1u });
		scene_state_rebuild_tlas(s, f32m44_identity());
		vkof::acceleration_structure_set_tlas(s.tlas);
		s.debugPcBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuGlobalExtended),
			.memory = vkof::BufferMemory::HostWritable,
		});
		s.modelsIndirectBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuResolveModelIndirect),
			.memory = vkof::BufferMemory::HostWritable,
		});
		return s;
	}
	s.scene = mor::scene_create();
	mor::scene_load_gltf(s.scene, path);
	s.gpuScene = mor::scene_gpu_upload(s.scene);
	s.bufs = mor::scene_gpu_buffers(s.gpuScene);
	s.materialsOverride = mor::scene_gpu_materials_create(s.scene);

	f32v3 boundsMin, boundsMax;
	mor::scene_bounds(s.scene, boundsMin, boundsMax);
	s.boundsCenter = (boundsMin + boundsMax) * 0.5f;
	s.boundsExtent = std::max(
		{
			boundsMax.x - boundsMin.x,
			boundsMax.y - boundsMin.y,
			boundsMax.z - boundsMin.z,
			0.01f,
		}
	);

	s.blas = vkof::blas_create({
		.positionVa = s.bufs.positions,
		.vertexCount = s.bufs.vertexCount,
		.indexVa = s.bufs.flatIndices,
		.triangleCount = s.bufs.triangleCount,
		.isOpaque = mor::scene_is_fully_opaque(s.scene),
	});
	if (s.blas.id == 0u) {
		printf("failed to build blas for '%s'\n", path);
		mor::scene_gpu_destroy(s.gpuScene);
		mor::scene_destroy(s.scene);
		return s;
	}
	s.tlas = vkof::tlas_create({ .maxInstances = 1u });
	scene_state_rebuild_tlas(s, f32m44_identity());
	vkof::acceleration_structure_set_tlas(s.tlas);

	s.debugPcBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuGlobalExtended),
		.memory = vkof::BufferMemory::HostWritable,
	});
	s.modelsIndirectBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect),
		.memory = vkof::BufferMemory::HostWritable,
	});
	return s;
}

void scene_state_destroy(SceneState & s) {
	vkof::device_wait_idle();
	if (s.modelsIndirectBuffer.id != 0u) {
		vkof::buffer_destroy(s.modelsIndirectBuffer);
	}
	if (s.debugPcBuffer.id != 0u) {
		vkof::buffer_destroy(s.debugPcBuffer);
	}
	if (s.tlas.id != 0u) {
		vkof::tlas_destroy(s.tlas);
	}
	if (s.blas.id != 0u) {
		vkof::blas_destroy(s.blas);
	}
	if (s.materialsOverride.id != 0u) {
		mor::scene_gpu_materials_destroy(s.materialsOverride);
	}
	// the empty (--vdb-only) scene never creates these; mor's destroys
	// dereference the handle unguarded
	if (s.gpuScene.id != 0u) {
		mor::scene_gpu_destroy(s.gpuScene);
	}
	if (s.scene.id != 0u) {
		mor::scene_destroy(s.scene);
	}
	s = {};
}

// -----------------------------------------------------------------------------
// -- stage mode: multi-instance rendering backed by a stage::Stage
// -----------------------------------------------------------------------------

// the viewer is either looking at a single loaded file (today's behavior,
// no editing) or authoring/viewing a stage (multiple instances, imgui
// outliner + gizmo, save-back) -- entered only via new/open stage, exited
// via unload stage. see project discussion: editing is gated to stage mode
// entirely, not just saving
enum class ViewerMode { FileView, StageEdit };

// mirrors GpuGlobalExtended.frameMode. camera drags render View (cheap
// unlit preview) every moved frame; after skCameraIdleSeconds of no
// movement, one Clear frame hard-zeros accumulation, then PathTrace resumes
enum class FrameMode : i32 { View = 0, Clear = 1, PathTrace = 2 };

constexpr f32 skCameraIdleSeconds = 0.2f;

// mirrors SceneState, generalized to n gltf instances. vdb rendering still
// binds only the stage's first vdb instance's first blob --
// GpuGlobalExtended.vdb is a single global handle, there is no per-instance
// volume slot in the shader yet, so stage mode can author multiple vdb
// instances but only the first one actually renders until the pt volume
// integrator grows multi-volume support (separate, larger task)
struct StageRenderState {
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

// (re)builds one blas per visible gltf instance and a tlas sized to match;
// call whenever the instance list itself changes (stage load/new/unload,
// add-instance) -- not on a plain transform edit, which only needs
// stage_render_rebuild_tlas below
void stage_render_rebuild_blases(
	StageRenderState & r, stage::Stage const & stage
) {
	vkof::device_wait_idle();
	for (vkof::AccelerationStructureBlas & b : r.blases) {
		if (b.id != 0u) { vkof::blas_destroy(b); }
	}
	r.blases.clear();
	r.blases.reserve(stage.gltfInstances.size());
	for (stage::GltfInstance const & inst : stage.gltfInstances) {
		if (inst.gpuScene.id == 0u) {
			r.blases.emplace_back(vkof::AccelerationStructureBlas {});
			continue;
		}
		mor::Buffers const bufs = mor::scene_gpu_buffers(inst.gpuScene);
		r.blases.emplace_back(vkof::blas_create({
			.positionVa = bufs.positions,
			.vertexCount = bufs.vertexCount,
			.indexVa = bufs.flatIndices,
			.triangleCount = bufs.triangleCount,
			.isOpaque = mor::scene_is_fully_opaque(inst.scene),
		}));
	}
	if (r.tlas.id != 0u) {
		vkof::tlas_destroy(r.tlas);
	}
	r.tlas = vkof::tlas_create({
		.maxInstances = (u32)std::max<size_t>(stage.gltfInstances.size(), 1u),
	});
	if (r.debugPcBuffer.id == 0u) {
		r.debugPcBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuGlobalExtended),
			.memory = vkof::BufferMemory::HostWritable,
		});
	}
	vkof::acceleration_structure_set_tlas(r.tlas);
}

// (re)builds the tlas instance list + the models-indirect buffer from the
// current per-instance transforms/visibility; cheap enough to call on every
// gizmo-drag or outliner edit frame (mirrors scene_state_rebuild_tlas's
// per-transform-change cadence for the single-model path). also recomputes
// the combined world bounds (union of every visible instance's local bounds
// under its own transform) for camera framing
void stage_render_rebuild_tlas(StageRenderState & r, stage::Stage const & stage) {
	std::vector<vkof::TlasInstance> tlasInstances;
	std::vector<GpuResolveModelIndirect> models;
	r.modelDrawIndexToInstance.clear();
	f32v3 boundsMin { FLT_MAX, FLT_MAX, FLT_MAX };
	f32v3 boundsMax { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	bool anyBounds = false;
	for (size_t i = 0u; i < stage.gltfInstances.size(); ++i) {
		stage::GltfInstance const & inst = stage.gltfInstances[i];
		if (
			!inst.visible || i >= r.blases.size() || r.blases[i].id == 0u
		) {
			continue;
		}
		f32m44 const transform = stage::transform_to_m44(inst.transform);
		tlasInstances.emplace_back(vkof::TlasInstance {
			.blas = r.blases[i],
			.transform = transform,
			.instanceCustomIndex = (u32)models.size(),
			.rayMask = 0xFFu,
		});
		r.modelDrawIndexToInstance.emplace_back(i);
		mor::Buffers const bufs = mor::scene_gpu_buffers(inst.gpuScene);
		u64 const materialsVa = (
			inst.materials.id != 0u
				? mor::scene_gpu_materials_va(inst.materials)
				: bufs.materials
		);
		models.emplace_back(GpuResolveModelIndirect {
			.meshlets = bufs.meshlets,
			.materials = materialsVa,
			.uvTransforms = bufs.uvTransforms,
			.positions = bufs.positions,
			.instances = bufs.instances,
			.attributes = bufs.attributes,
			.meshletVerts = bufs.meshletVerts,
			.meshletTris = bufs.meshletTris,
			.flatIndices = bufs.flatIndices,
			.flatMeshlets = bufs.flatMeshlets,
			.modelMatrix = transform,
		});
		f32v3 instWorldMin, instWorldMax;
		if (stage::gltf_instance_world_bounds(inst, instWorldMin, instWorldMax)) {
			boundsMin.x = std::min(boundsMin.x, instWorldMin.x);
			boundsMin.y = std::min(boundsMin.y, instWorldMin.y);
			boundsMin.z = std::min(boundsMin.z, instWorldMin.z);
			boundsMax.x = std::max(boundsMax.x, instWorldMax.x);
			boundsMax.y = std::max(boundsMax.y, instWorldMax.y);
			boundsMax.z = std::max(boundsMax.z, instWorldMax.z);
			anyBounds = true;
		}
	}
	vkof::RenderNode const tlasNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = tlasNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd,
				r.tlas,
				srat::slice<vkof::TlasInstance const>(
					tlasInstances.data(), tlasInstances.size()
				)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&tlasNode, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(tlasNode);

	if (models.size() > r.modelsIndirectCapacity) {
		if (r.modelsIndirectBuffer.id != 0u) {
			vkof::device_wait_idle();
			vkof::buffer_destroy(r.modelsIndirectBuffer);
		}
		r.modelsIndirectCapacity = (u32)models.size();
		r.modelsIndirectBuffer = vkof::buffer_create({
			.byteCount = (
				sizeof(GpuResolveModelIndirect)
				* std::max<u32>(r.modelsIndirectCapacity, 1u)
			),
			.memory = vkof::BufferMemory::HostWritable,
		});
	}
	if (!models.empty()) {
		vkof::buffer_upload({
			.buffer = r.modelsIndirectBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(models.data()),
				models.size() * sizeof(GpuResolveModelIndirect)
			),
		});
	}

	if (anyBounds) {
		r.boundsCenter = (boundsMin + boundsMax) * 0.5f;
		r.boundsExtent = std::max(
			{
				boundsMax.x - boundsMin.x,
				boundsMax.y - boundsMin.y,
				boundsMax.z - boundsMin.z,
				0.01f,
			}
		);
	} else {
		r.boundsCenter = { 0.0f, 0.0f, 0.0f };
		r.boundsExtent = 1.0f;
	}
}

void stage_render_destroy(StageRenderState & r) {
	vkof::device_wait_idle();
	for (vkof::AccelerationStructureBlas & b : r.blases) {
		if (b.id != 0u) { vkof::blas_destroy(b); }
	}
	r.blases.clear();
	if (r.tlas.id != 0u) {
		vkof::tlas_destroy(r.tlas);
	}
	if (r.modelsIndirectBuffer.id != 0u) {
		vkof::buffer_destroy(r.modelsIndirectBuffer);
	}
	if (r.debugPcBuffer.id != 0u) {
		vkof::buffer_destroy(r.debugPcBuffer);
	}
	r = {};
}

// a scanned asset file, tagged with which configured root directory it was
// found under so the imgui model list (below) can group by root instead of
// showing one flat list across every configured library
struct ModelEntry {
	std::filesystem::path root;
	std::filesystem::path path;
};

// settings.json at the repo root selects where assets come from:
//   { "modelPaths": ["assets/models", "/absolute/path/too"] }
// each listed directory (relative paths resolve against the settings
// file's own directory) is searched recursively for files matching one of
// `extensions`. the first key in `keys` whose value is an array wins --
// multiple keys only exist to accept spelling variants
// (environmentMapPaths / environmentMaps)
std::vector<ModelEntry> scan_asset_paths(
	std::filesystem::path const & settingsPath,
	std::initializer_list<char const *> const keys,
	std::initializer_list<char const *> const extensions,
	bool const warnMissingKey
) {
	std::vector<ModelEntry> out;
	FILE * const file = fopen(settingsPath.string().c_str(), "rb");
	if (!file) {
		return out;
	}
	char buffer[16384];
	rapidjson::FileReadStream stream(file, buffer, sizeof(buffer));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(file);
	if (doc.HasParseError() || !doc.IsObject()) {
		printf("failed to parse %s\n", settingsPath.string().c_str());
		return out;
	}
	auto pathsIt = doc.MemberEnd();
	for (char const * const key : keys) {
		pathsIt = doc.FindMember(key);
		if (pathsIt != doc.MemberEnd() && pathsIt->value.IsArray()) {
			break;
		}
	}
	if (pathsIt == doc.MemberEnd() || !pathsIt->value.IsArray()) {
		if (warnMissingKey) {
			printf(
				"%s has no \"%s\" array\n",
				settingsPath.string().c_str(), *keys.begin()
			);
		}
		return out;
	}
	for (auto const & entry : pathsIt->value.GetArray()) {
		if (!entry.IsString()) {
			continue;
		}
		std::filesystem::path dir(entry.GetString());
		if (dir.is_relative()) {
			dir = settingsPath.parent_path() / dir;
		}
		std::error_code ec;
		if (!std::filesystem::is_directory(dir, ec)) {
			printf(
				"\"%s\" entry is not a directory: %s\n",
				*keys.begin(), dir.string().c_str()
			);
			continue;
		}
		// sorted per-root rather than globally, so the imgui grouping below
		// gets contiguous, alphabetized runs within each root's collapsible
		// section instead of needing to re-sort at display time
		std::vector<std::filesystem::path> found;
		for (
			std::filesystem::recursive_directory_iterator it(
				dir,
				std::filesystem::directory_options::skip_permission_denied,
				ec
			);
			it != std::filesystem::recursive_directory_iterator();
			it.increment(ec)
		) {
			if (ec) {
				break;
			}
			if (!it->is_regular_file(ec)) {
				continue;
			}
			std::filesystem::path const & p = it->path();
			for (char const * const ext : extensions) {
				if (p.extension() == ext) {
					found.push_back(p);
					break;
				}
			}
		}
		std::sort(found.begin(), found.end());
		for (std::filesystem::path & p : found) {
			out.push_back({ dir, std::move(p) });
		}
	}
	return out;
}

// draws one grouped asset list (a collapsing header per configured root
// directory, parent-dir/filename labels) inside a fixed-height child; shared
// by the models and vdb sections of the viewer window. returns the clicked
// path, empty when nothing was picked this frame
std::string asset_list_draw(
	std::vector<ModelEntry> const & list,
	std::string const & currentPath,
	char const * const childId,
	f32 const height
) {
	std::string picked;
	ImGui::BeginChild(childId, ImVec2(0.0f, height), ImGuiChildFlags_None);
	std::filesystem::path const * openRoot = nullptr;
	bool rootOpen = false;
	for (ModelEntry const & entry : list) {
		if (!openRoot || *openRoot != entry.root) {
			openRoot = &entry.root;
			ImGui::PushID(entry.root.string().c_str());
			rootOpen = ImGui::CollapsingHeader(
				entry.root.string().c_str(),
				ImGuiTreeNodeFlags_DefaultOpen
			);
			ImGui::PopID();
		}
		if (!rootOpen) {
			continue;
		}
		// parent-dir/filename: distinguishes the gltf/gltf-binary variants
		// sample-asset directories usually keep side by side under the same
		// model folder
		std::string const label = (
			entry.path.parent_path().filename().string()
			+ "/" + entry.path.filename().string()
		);
		bool const isCurrent = entry.path.string() == currentPath;
		// full path as id: labels repeat across roots (two libraries can
		// both have an "Avocado" folder)
		ImGui::PushID(entry.path.string().c_str());
		if (ImGui::Selectable(label.c_str(), isCurrent) && !isCurrent) {
			picked = entry.path.string();
		}
		ImGui::PopID();
	}
	ImGui::EndChild();
	return picked;
}

// world-bbox center/extent of a loaded grid, for camera framing when there
// is no mesh to derive bounds from
void vdb_bounds(
	ponder::Vdb const & vdb, f32v3 & outCenter, f32 & outExtent
) {
	auto const * grid = (
		reinterpret_cast<nanovdb::FloatGrid const *>(vdb.blob.data())
	);
	nanovdb::Vec3dBBox const & bbox = grid->worldBBox();
	outCenter = f32v3 {
		(f32)((bbox.min()[0] + bbox.max()[0]) * 0.5),
		(f32)((bbox.min()[1] + bbox.max()[1]) * 0.5),
		(f32)((bbox.min()[2] + bbox.max()[2]) * 0.5),
	};
	outExtent = std::max(
		{
			(f32)(bbox.max()[0] - bbox.min()[0]),
			(f32)(bbox.max()[1] - bbox.min()[1]),
			(f32)(bbox.max()[2] - bbox.min()[2]),
			0.01f,
		}
	);
}

} // namespace

int32_t main(int32_t const argc, char const * const * const argv) {
	ViewerArgs args;
	if (!parse_args(argc, argv, args)) {
		print_usage();
		return 1;
	}
	bool const headless = args.screenshotPath != nullptr;

	if (headless) {
		vkof::init_headless(args.width, args.height);
	} else {
		vkof::init(args.debug);
		vkof::render_extent_set(args.width, args.height);
	}

	ponder::vdb_initialize();

	std::filesystem::path const appDir = (
		std::filesystem::canonical(
			std::filesystem::path(__FILE__).parent_path()
		)
	);
	std::string const appShaderDir = (appDir / "shaders").string() + "/";

	// -- long-lived gpu tables
	ponder::EnergyTables const energyTables = ponder::energy_tables_create();
	ponder::ZeltnerTables const zeltnerTables = ponder::zeltner_tables_create();
	ponder::Bluenoise const bluenoise = ponder::bluenoise_create(PONDER_ASSET_DIR);

	// -- environment map (optional). envMapHandlesBuffer holds one
	// GpuEnvironmentMapHandles, re-uploaded on every load (--env-exr at
	// startup, or the imgui exr picker at runtime) --
	// GpuGlobalExtended.envMap is its VA whenever a map is resident
	ponder::EnvironmentTables sEnvTables {};
	std::string sCurrentEnvMapPath;
	vkof::Buffer const envMapHandlesBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuEnvironmentMapHandles),
		.memory = vkof::BufferMemory::HostWritable,
	});
	u64 const envMapHandlesVa = (
		vkof::buffer_virtual_address(envMapHandlesBuffer)
	);
	auto const fnLoadEnvMap = [&](std::string const & path) -> bool {
		// in-flight frames may still reference the outgoing tables' images
		vkof::device_wait_idle();
		if (sEnvTables.width != 0u) {
			ponder::environment_tables_destroy(sEnvTables);
			sEnvTables = {};
		}
		sCurrentEnvMapPath.clear();
		sEnvTables = ponder::environment_tables_create(path);
		if (sEnvTables.width == 0u) {
			// load failed (already warned by environment_tables_create)
			return false;
		}
		GpuEnvironmentMapHandles const handles = {
			.radiance = sEnvTables.radianceHandle,
			.pdf = sEnvTables.pdfHandle,
			.conditionalCdf = sEnvTables.conditionalCdfHandle,
			.marginalCdf = sEnvTables.marginalCdfHandle,
		};
		vkof::buffer_upload({
			.buffer = envMapHandlesBuffer,
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(handles),
		});
		sCurrentEnvMapPath = path;
		return true;
	};
	if (
		args.envExrPath != nullptr
		&& !fnLoadEnvMap(args.envExrPath)
		&& args.envMode == 3
	) {
		// fall back rather than rendering hdrmap with no data behind it
		args.envMode = 0;
	}

	// -- pipelines
	char const * const includePaths[] = {
		PONDER_SHADER_DIR,
		MOR_INCLUDE_DIR,
	};
	vkof::Pipeline const resolvePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (appDir / "shaders/resolve.comp").string().c_str(),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const ptAccumulatePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/pt-accumulate.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const nanCheckPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/nan-check.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);

	// -- broken-pixel counters fed by nan-check.comp; host-visible so the
	// imgui panel reads them straight off the mapping (a frame stale, and
	// the reset below can race in-flight atomics -- both fine for a debug
	// display). [0] NaN pixels, [1] Inf pixels, cumulative since the last
	// accumulation reset
	vkof::Buffer const nanCountsBuffer = vkof::buffer_create({
		.byteCount = 2u * sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnZeroNanCounts = [&]() {
		u32 const zero[2] = { 0u, 0u };
		vkof::buffer_upload({
			.buffer = nanCountsBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(zero), sizeof(zero)
			),
		});
	};
	fnZeroNanCounts();

	// -- single u32 atomic counter backing util-nan-probe.glsl's
	// NAN_PROBE_LIMIT_PER_FRAME gate; zeroed every frame (unlike
	// nanCountsBuffer above, which accumulates across an accumulation run)
	// so each frame gets its own fresh printf budget
	vkof::Buffer const nanProbeCounterBuffer = vkof::buffer_create({
		.byteCount = sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnZeroNanProbeCounter = [&]() {
		u32 const zero = 0u;
		vkof::buffer_upload({
			.buffer = nanProbeCounterBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&zero), sizeof(zero)
			),
		});
	};
	fnZeroNanProbeCounter();

	// -- material inspector probe result: resolve.comp writes the hit's
	// modelDrawIndex/materialIndex here (see GpuGlobalExtended.probeResult)
	// when a right-click sets probePixel; a plain host-readable buffer
	// rather than debug_printf, so it works without --debug/validation
	// layers enabled. reset to the sentinel before every probe request so a
	// stale leftover value from a previous click can't be mistaken for a
	// fresh one
	vkof::Buffer const probeResultBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuProbeResult),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnResetProbeResult = [&]() {
		GpuProbeResult const sentinel { -1, -1 };
		vkof::buffer_upload({
			.buffer = probeResultBuffer,
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(sentinel),
		});
	};
	fnResetProbeResult();

	// -- single u32 tallying COMPLETED PIXELS: pt-accumulate.comp atomicAdd's
	// it once per pixel, on the frame that pixel's sample count crosses the
	// max-spp target (so a finished render reads renderWidth * renderHeight).
	// host-visible so the imgui progress bar reads it straight off the
	// mapping (a frame or so stale, fine for a progress display). zeroed on
	// every accumulation reset; lowering the target below what's already
	// accumulated under-reads until the next reset (pixels only bump on the
	// crossing frame)
	vkof::Buffer const sppCounterBuffer = vkof::buffer_create({
		.byteCount = sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnZeroSppCounter = [&]() {
		u32 const zero = 0u;
		vkof::buffer_upload({
			.buffer = sppCounterBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&zero), sizeof(zero)
			),
		});
	};
	fnZeroSppCounter();
	auto const fnCompletedPixels = [&]() {
		return *reinterpret_cast<u32 const *>(
			vkof::buffer_host_address(sppCounterBuffer).ptr()
		);
	};

	// -- settings.json model + environment-map libraries at the repo root;
	// rescanned on every model load so newly added files show up without a
	// restart
	std::filesystem::path const settingsPath = (
		appDir.parent_path().parent_path() / "settings.json"
	);
	std::vector<ModelEntry> sModelList = scan_asset_paths(
		settingsPath, { "modelPaths" }, { ".gltf", ".glb" },
		/*warnMissingKey=*/true
	);
	std::vector<ModelEntry> sEnvMapList = scan_asset_paths(
		settingsPath, { "environmentMapPaths", "environmentMaps" },
		{ ".exr" }, /*warnMissingKey=*/false
	);
	std::vector<ModelEntry> sVdbList = scan_asset_paths(
		settingsPath, { "vdbPaths" }, { ".vdb" },
		/*warnMissingKey=*/false
	);

	// -- vdb
	std::string sCurrentVdbPath;
	if (args.vdbPath != nullptr) {
		sVdb = ponder::vdb_load(args.vdbPath);
		if (sVdb.gpuBuffer.id != 0u) {
			sCurrentVdbPath = args.vdbPath;
		}
	}

	// -- scene; an explicit positional path wins, then the library's first
	// model -- unless --vdb was given without any model, which deliberately
	// skips the library fallback so the volume renders alone in an empty
	// scene (zero-instance tlas, every ray reaches the environment)
	std::string sCurrentModelPath;
	bool const startInStage = args.newStage || args.stagePath != nullptr;
	if (args.gltfPath != nullptr) {
		sCurrentModelPath = args.gltfPath;
	} else if (args.vdbPath == nullptr && !startInStage && !sModelList.empty()) {
		sCurrentModelPath = sModelList[0].path.string();
	}
	// --stage/--new-stage bypass this entirely: file view starts out empty
	// (a valid, zero-instance scene) and stage mode takes over below
	if (
		sCurrentModelPath.empty() && args.vdbPath == nullptr && !startInStage
	) {
		printf(
			"no model to load: pass a .gltf/.glb path, list modelPaths in %s,"
			" or pass --stage/--new-stage\n",
			settingsPath.string().c_str()
		);
		return 1;
	}
	SceneState sScene = scene_state_create(
		sCurrentModelPath.empty() ? nullptr : sCurrentModelPath.c_str()
	);
	if (!sCurrentModelPath.empty() && sScene.blas.id == 0u) {
		return 1;
	}
	// vdb-only: frame the camera to the grid's world bbox, since there is no
	// mesh to derive bounds from
	if (sCurrentModelPath.empty() && !sVdb.blob.empty()) {
		vdb_bounds(sVdb, sScene.boundsCenter, sScene.boundsExtent);
	}

	// -- viewer mode: file view (today's single-model/single-vdb viewing, no
	// editing) vs stage edit (multi-instance authoring backed by a
	// stage::Stage, entered only via new/open stage). per-instance transform
	// editing in stage mode goes through stage_imgui_gizmo_edit (imguizmo)
	// instead of the drag-float sliders file view used to have -- those are
	// gone entirely, not just hidden, since file view no longer edits
	// anything
	ViewerMode sMode = ViewerMode::FileView;
	stage::Stage sStage {};
	StageRenderState sStageRender {};
	// unsaved until the user explicitly saves; new stage starts empty here
	char sStagePathBuffer[512] = {};

	auto const fnActiveBoundsCenter = [&]() {
		return (
			sMode == ViewerMode::StageEdit
				? sStageRender.boundsCenter : sScene.boundsCenter
		);
	};
	auto const fnActiveBoundsExtent = [&]() {
		return (
			sMode == ViewerMode::StageEdit
				? sStageRender.boundsExtent : sScene.boundsExtent
		);
	};
	// re-frames the orbit camera to whichever mode is active; called on
	// every mode switch and stage new/open/unload, mirroring what the old
	// model-load/unload paths did inline against sScene. callers still need
	// to follow up with fnSyncFpFromOrbit() themselves (declared later,
	// alongside cam/camFp) to keep first-person in sync too
	auto const fnReframeCamera = [&](srat::CameraOrbit & orbitCam) {
		orbitCam.target = fnActiveBoundsCenter();
		orbitCam.distance = fnActiveBoundsExtent() * 1.5f;
		orbitCam.near = fnActiveBoundsExtent() * 0.001f;
		orbitCam.far = fnActiveBoundsExtent() * 100.0f;
	};

	// -- render targets
	vkof::TransientImage const colorTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r32g32b32a32_sfloat,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1u,
		.isDoubleBuffered = false,
	});
	vkof::TransientImage const ptOutputTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1u,
		.isDoubleBuffered = false,
	});
	// internal render resolution; the imgui dropdown changes this at runtime
	// (headless renders stay at the --resolution size)
	u32 sRenderW = args.width;
	u32 sRenderH = args.height;

	// the accumulation buffer is a persistent image sized to the internal
	// render resolution, so a resolution change recreates it
	vkof::Image ptAccumImage {};
	u32 ptAccumStorageHandle = 0u;
	auto const fnCreateAccumImage = [&]() {
		ptAccumImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		ptAccumStorageHandle = vkof::image_storage_handle({
			.image = ptAccumImage,
			.mipLevel = 0u,
		});
	};
	fnCreateAccumImage();

	// -- environment settings (imgui-editable in windowed mode)
	f32 sEnvIntensity = args.envIntensity;
	// backdrop-only scale on primary-ray misses (resolve.comp); the historic
	// hard-coded 0.001 stays the default so the environment reads as distant
	// from the scene geometry until deliberately raised
	f32 sEnvBackgroundIntensity = 0.001f;
	i32 sEnvMode = args.envMode;
	// next-event estimation toward the env map, mis-combined with the
	// existing bsdf-sampled technique; strictly lower variance at equal
	// sample count once correct, and a no-op for every env mode other than
	// hdrmap, so on-by-default is safe
	bool sEnvNeeEnabled = args.envMis;
	f32 sEnvRotation = 0.0f;
	// applies sStage.environment onto the live (file-view-style) env state
	// above -- called on new/open stage and --stage/--new-stage startup.
	// there's no dedicated stage-side environment editor (see
	// stage::Environment's comment): the "mode"/"exr map"/"env intensity"/
	// "env rotation" controls further down are the only environment ui,
	// shared by both modes, and get synced back into sStage.environment
	// every frame while in stage mode (see the stageInstancesChanged/
	// focusRequest handling below) so save picks up whatever's live
	auto const fnApplyStageEnvironment = [&]() {
		sEnvMode = (i32)sStage.environment.mode;
		sEnvRotation = sStage.environment.rotationRadians;
		sEnvIntensity = sStage.environment.intensity;
		if (sEnvMode == 3 && !sStage.environment.path.empty()) {
			fnLoadEnvMap(sStage.environment.path);
		}
	};
	// caps a sample's luminance before it's written out; not physically
	// correct (loses a little energy from the clamped samples) but keeps
	// the rare, very-high-variance paths ordinary nee/mis can't reach
	// (e.g. a diffuse bounce that happens to hit a specular/glass object
	// which then refracts to a bright light -- nee only helps at the
	// vertex it's evaluated from, so it can't shortcut a caustic like
	// that) from blowing out the image as fireflies
	bool sFireflyClampEnabled = args.fireflyClampLuminance > 0.0f;
	f32 sFireflyClampLuminance = (
		args.fireflyClampLuminance > 0.0f ? args.fireflyClampLuminance : 20.0f
	);
	f32 sExposure = 1.0f;
	bool sAntialias = true;

	// -- vdb volume params (imgui-editable, "vdb" section); TODO should be
	// per vdb-blob (see GpuGlobalExtended), global for now
	f32 sVdbSigmaScale = 1.0f;
	f32 sVdbDropletDiameter = 25.0f;
	f32v3 sVdbAlbedo = { 1.0f, 1.0f, 1.0f };

	// -- homogeneous fog params (imgui-editable, "fog" section); see
	// GpuGlobalExtended.fogThickness/fogAlbedo and
	// util-vdb.glsl's homogeneous-fog section
	bool sFogEnabled = false;
	f32 sFogSigma = 0.1f;
	f32v3 sFogAlbedo = { 1.0f, 1.0f, 1.0f };
	// GpuGlobalExtended.fogDistanceMax: the fog's extent, so nee rays toward
	// the env map aren't fully extinguished by an otherwise-unbounded
	// medium; also caps the random-walk free-flight distance
	f32 sFogDistanceMax = 1000.0f;
	// 0 accumulates forever; > 0 freezes accumulation once every pixel holds
	// that many samples (see fnRenderFrame's freeze path). raising it later
	// resumes the same accumulation rather than restarting it
	i32 sMaxSpp = 0;
	// set by the imgui screenshot button, consumed right after the frame
	// renders so the capture matches what's on screen
	bool sScreenshotPending = false;
	std::string sLastScreenshotPath;

	// -- material inspector: right-click sets sProbePixel for the next
	// upload/render; resolve.comp writes the hit's modelDrawIndex/
	// materialIndex into probeResultBuffer (a plain buffer, not
	// debug_printf -- see GpuProbeResult). (-1,-1) means no probe pending;
	// fnUploadFrameState resets sProbePixel right after upload, so a probe
	// only ever fires for the one frame right after a click
	i32v2 sProbePixel { -1, -1 };
	i32 sPickedMaterialIndex = -1;
	// which stage instance the probe hit (file view is always 0, its only
	// "instance"); resolves to a real stage::GltfInstance via
	// sStageRender.modelDrawIndexToInstance
	i32 sPickedModelDrawIndex = -1;
	// the gpu is pipelined deeply enough that a given frame's probe-result
	// write can surface well after the render_graph_execute call that
	// submitted it returns -- reading probeResultBuffer live from the imgui
	// block (which always runs before this iteration's own fnRenderFrame)
	// would miss it essentially every time. copied out (into
	// sPickedModelDrawIndex/sPickedMaterialIndex above, and this pending
	// flag) immediately after fnRenderFrame instead, which gives the
	// readback the largest possible window before the next probe request
	// would overwrite it; the imgui block at the top of the next iteration
	// is what actually opens the popup
	bool sMaterialPopupPending = false;
	std::vector<std::string> sProbeMessages;

	// -- camera
	srat::CameraOrbit cam = {
		.target = sScene.boundsCenter,
		.distance = (
			args.cameraOverride && args.cameraDistance > 0.0f
				? args.cameraDistance
				: sScene.boundsExtent * 1.5f
		),
		.azimuth = args.cameraAzimuth,
		.elevation = args.cameraElevation,
		.fovY = args.cameraFovY,
		.aspect = (f32)sRenderW / (f32)sRenderH,
		.near = sScene.boundsExtent * 0.001f,
		.far = sScene.boundsExtent * 100.0f,
	};
	// tab (windowed mode) swaps between orbit and first-person controls.
	// the orbit eye looks back at its target, so yaw/pitch are exactly the
	// negated azimuth/elevation -- both syncs preserve the eye position and
	// view direction, which keeps the image (and the pt accumulation) from
	// jumping on toggle
	bool sFirstPerson = false;
	srat::CameraFirstPerson camFp = {};
	auto const fnSyncFpFromOrbit = [&]() {
		camFp.position = srat::camera_orbit_eye(cam);
		camFp.yaw = -cam.azimuth;
		camFp.pitch = -cam.elevation;
		camFp.fovY = cam.fovY;
		camFp.aspect = cam.aspect;
		camFp.near = cam.near;
		camFp.far = cam.far;
	};
	auto const fnSyncOrbitFromFp = [&]() {
		// the previous orbit distance is reused, so only the pivot point is
		// re-derived
		cam.target = (
			camFp.position + srat::camera_fp_forward(camFp) * cam.distance
		);
		cam.azimuth = -camFp.yaw;
		cam.elevation = -camFp.pitch;
		cam.fovY = camFp.fovY;
	};
	fnSyncFpFromOrbit();

	// -- --stage / --new-stage: enter stage mode before the first frame,
	// instead of requiring the user to click new/open in the imgui panel
	// (which they'd otherwise never reach, since the gltf/vdb-required gate
	// above is what's bypassed to get here)
	if (startInStage) {
		sMode = ViewerMode::StageEdit;
		if (args.stagePath != nullptr) {
			sStage = stage::stage_load(args.stagePath);
			if (sStage.sourceDirectory.empty()) {
				printf("failed to open stage '%s'\n", args.stagePath);
			}
			std::strncpy(
				sStagePathBuffer, args.stagePath, sizeof(sStagePathBuffer) - 1
			);
		}
		stage_render_rebuild_blases(sStageRender, sStage);
		stage_render_rebuild_tlas(sStageRender, sStage);
		fnApplyStageEnvironment();
		fnReframeCamera(cam);
		fnSyncFpFromOrbit();
	}

	// headless stays at PathTrace the whole run
	FrameMode sFrameMode = FrameMode::PathTrace;
	f32 sViewIdleSeconds = 0.0f;

	auto const fnUploadFrameState = [&]() {
		// each frame gets a fresh NAN_PROBE_LIMIT_PER_FRAME budget
		fnZeroNanProbeCounter();
		// stage mode binds only the stage's first vdb instance's first blob
		// -- see StageRenderState's comment on why -- and applies that
		// instance's transform; file view's vdb has no instance transform of
		// its own, so it always renders at the grid's native placement
		stage::VdbBlob const * const activeVdbBlob = (
			sMode == ViewerMode::StageEdit
				&& !sStage.vdbInstances.empty()
				&& !sStage.vdbInstances[0].blobs.empty()
					? &sStage.vdbInstances[0].blobs[0]
					: nullptr
		);
		u64 const activeVdbHandleVa = (
			sMode == ViewerMode::StageEdit
				? (
					activeVdbBlob != nullptr
					&& activeVdbBlob->vdb.handleBuffer.id != 0u
				)
					? vkof::buffer_virtual_address(activeVdbBlob->vdb.handleBuffer)
					: 0u
				: (sVdb.handleBuffer.id != 0u)
					? vkof::buffer_virtual_address(sVdb.handleBuffer)
					: 0u
		);
		f32m44 const activeVdbWorldToLocal = (
			sMode == ViewerMode::StageEdit && !sStage.vdbInstances.empty()
				? stage::transform_to_inverse_m44(
					sStage.vdbInstances[0].transform
				)
				: f32m44_identity()
		);
		GpuGlobalExtended const extended = {
			.envIntensity = sEnvIntensity,
			.renderWidth = sRenderW,
			.renderHeight = sRenderH,
			.envMode = sEnvMode,
			.antialias = sAntialias ? 1u : 0u,
			.frameMode = (i32)sFrameMode,
			.nanProbeCounterVa = (
				vkof::buffer_virtual_address(nanProbeCounterBuffer)
			),
			.probePixel = sProbePixel,
			.probeResult = vkof::buffer_virtual_address(probeResultBuffer),
			.envMap = sEnvTables.width != 0u ? envMapHandlesVa : 0u,
			.envRotation = sEnvRotation,
			.envNeeEnabled = sEnvNeeEnabled ? 1u : 0u,
			.fireflyClampLuminance = (
				sFireflyClampEnabled ? sFireflyClampLuminance : 0.0f
			),
			.envBackgroundIntensity = sEnvBackgroundIntensity,
			.vdb = activeVdbHandleVa,
			.vdbWorldToLocal = activeVdbWorldToLocal,
			// stage mode's blob owns these directly (edited in
			// stage_imgui_edit's blob inspector); file view keeps using its
			// own global sliders
			.vdbSigmaScale = (
				activeVdbBlob != nullptr ? activeVdbBlob->sigmaScale : sVdbSigmaScale
			),
			.vdbDropletDiameter = (
				activeVdbBlob != nullptr
					? activeVdbBlob->dropletDiameter : sVdbDropletDiameter
			),
			.vdbAlbedo = (
				activeVdbBlob != nullptr ? activeVdbBlob->albedo : sVdbAlbedo
			),
			.fogThickness = sFogEnabled ? sFogSigma : 0.0f,
			.fogAlbedo = sFogAlbedo,
			.fogDistanceMax = sFogDistanceMax,
		};
		// one-shot: consumed by this upload, so it doesn't keep firing
		// resolve.comp's debugPrintfEXT on every subsequent frame
		sProbePixel = { -1, -1 };
		vkof::buffer_upload({
			.buffer = (
				sMode == ViewerMode::StageEdit
					? sStageRender.debugPcBuffer : sScene.debugPcBuffer
			),
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(extended),
		});
		// stage mode's modelsIndirectBuffer is n entries, kept current by
		// stage_render_rebuild_tlas (called whenever an instance/transform
		// changes, not every frame) -- only file view's single-entry buffer
		// needs a per-frame upload here, and only because it always did;
		// the model is always at identity now (file view has no transform
		// editing left), so this is really just re-asserting the same
		// unchanging descriptor every frame
		if (sMode == ViewerMode::FileView) {
			// falls back to the original, un-editable buffer for the (rare)
			// zero-material model, since scene_gpu_materials_create never
			// allocates a backing buffer in that case (va stays 0); the
			// empty --vdb-only scene has no materialsOverride at all, and
			// the va getter dereferences the handle unguarded
			u64 const materialsVa = (
				sScene.materialsOverride.id != 0u
					? mor::scene_gpu_materials_va(sScene.materialsOverride)
					: 0u
			);
			GpuResolveModelIndirect const modelDesc = {
				.meshlets = sScene.bufs.meshlets,
				.materials = (
					materialsVa != 0u ? materialsVa : sScene.bufs.materials
				),
				.uvTransforms = sScene.bufs.uvTransforms,
				.positions = sScene.bufs.positions,
				.instances = sScene.bufs.instances,
				.attributes = sScene.bufs.attributes,
				.meshletVerts = sScene.bufs.meshletVerts,
				.meshletTris = sScene.bufs.meshletTris,
				.flatIndices = sScene.bufs.flatIndices,
				.flatMeshlets = sScene.bufs.flatMeshlets,
				.modelMatrix = f32m44_identity(),
			};
			vkof::buffer_upload({
				.buffer = sScene.modelsIndirectBuffer,
				.byteOffset = 0u,
				.data = srat::slice_as_bytes(modelDesc),
			});
		}
	};

	u32 sFrameIndex = 0u;
	u32 sAccumFrames = 0u;

	// -- frame budget: instead of tracing the whole screen every frame,
	// resolve dispatches a sliding window of kPtTileSize tiles sized so its
	// measured gpu time tracks sBudgetTargetMs. accumulation is
	// rate-independent (per-pixel mean + count), so tiles refreshing at
	// different cadences converge correctly; the accumulate pass gates its
	// mean-update on the window (see pt-accumulate.comp)
	bool sBudgetEnabled = true; // always enabled
	f32 sBudgetTargetMs = args.budgetMs > 0.0f ? args.budgetMs : 8.0f;
	// smoothed gpu cost of one tile; <= 0 until the first measurement lands
	double sMsPerTileEma = 0.0;
	u32 sTileCursor = 0u;
	// smoothed window size in tiles (<= 0 until the controller runs);
	// forced full-coverage frames (reset/probe/budget-off) bypass it so
	// they don't yank the smoothed size around
	double sTileKSmooth = 0.0;
	// current window's tile count + how many frames it's been held for;
	// the cursor only advances once sTileWindowFrames reaches sTileWindowSpp
	u32 sTileWindowCount = 0u;
	u32 sTileWindowFrames = 0u;
	i32 sTileWindowSpp = 32;
	// display-only: last dispatched window + its measured cost
	u32 sTileKLast = 0u;
	double sResolveMsLast = 0.0;
	struct TileFrameHistory {
		bool resolveRan;
		u32 tileCount;
	};
	// per vkof frame slot: what resolve dispatched the last time this slot
	// was used. vkof's timestamp readback (node_gpu_ms) is for the previous
	// use of the slot, i.e. kFramesInFlight (2) frames latent, so each
	// measurement must pair with the window size from that frame, not the
	// current one
	TileFrameHistory sTileHistory[2] = {};

	// -- focus region: when active, the budget window above slides over
	// just this sub-rect of the screen instead of the full grid, so all
	// resolve work concentrates on a hard-to-converge area; pixels outside
	// it stop accumulating and hold their current average (see
	// pt-accumulate.comp's coverage test). stored normalized to [0,1] so it
	// survives render-resolution switches, snapped outward to the
	// kPtTileSize grid at use. set by right-click drag while sFocusMode is
	// on (which repurposes the material-inspector probe button)
	bool sFocusMode = false;
	bool sFocusActive = false;
	f32v2 sFocusMin { 0.0f, 0.0f };
	f32v2 sFocusMax { 1.0f, 1.0f };
	struct FocusTileBounds { u32 x0, y0, x1, y1; };
	// tile-snapped inclusive bounds of the focus rect at the current render
	// res; shared by the dispatch region and the ui overlay so what is
	// outlined is exactly what gets dispatched. min == max collapses to one
	// tile; the clamps keep a stale rect valid across resolution switches
	auto const fnFocusTileBounds = [&](
		u32 const gridW, u32 const gridH
	) -> FocusTileBounds {
		return FocusTileBounds {
			.x0 = std::min(
				(u32)(sFocusMin.x * (f32)sRenderW) / kPtTileSize, gridW - 1u
			),
			.y0 = std::min(
				(u32)(sFocusMin.y * (f32)sRenderH) / kPtTileSize, gridH - 1u
			),
			.x1 = std::min(
				(u32)(sFocusMax.x * (f32)sRenderW) / kPtTileSize, gridW - 1u
			),
			.y1 = std::min(
				(u32)(sFocusMax.y * (f32)sRenderH) / kPtTileSize, gridH - 1u
			),
		};
	};

	auto const fnRenderFrame = [&](
		FrameMode const frameMode,
		// only meaningful when frameMode == PathTrace -- View/Clear have
		// their own dedicated full-screen behavior regardless of this
		bool const pathTraceReset,
		bool const freezeAccum,
		bool const probeRequested
	) {
		// max-spp freeze: accumulation is capped, so the resolve and
		// nan-check dispatches are skipped and the gpu goes mostly idle --
		// the accumulate pass alone keeps re-outputting the frozen average,
		// which keeps exposure/tonemap live. a pending material-inspector
		// probe still needs one resolve dispatch to answer (the probe write
		// lives in resolve.comp); its sample goes nowhere since accumulation
		// stays frozen. View mode always resolves (that IS its whole job);
		// Clear mode never does (pt-accumulate.comp hard-zeros without it)
		bool const runResolve = (
			frameMode == FrameMode::View
			|| (
				frameMode == FrameMode::PathTrace && (!freezeAccum || probeRequested)
			)
		);
		GpuGlobalPC const globalPC = {
			.time = 0.0f,
			.cameraPos = (
				sFirstPerson
					? camFp.position
					: srat::camera_orbit_eye(cam)
			),
			.exposure = sExposure,
			.pad0 = 0.0f,
			.viewProj = (
				sFirstPerson
					? srat::camera_fp_proj(camFp) * srat::camera_fp_view(camFp)
					: srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam)
			),
			.extended = vkof::buffer_virtual_address(
				sMode == ViewerMode::StageEdit
					? sStageRender.debugPcBuffer : sScene.debugPcBuffer
			),
			.models = vkof::buffer_virtual_address(
				sMode == ViewerMode::StageEdit
					? sStageRender.modelsIndirectBuffer
					: sScene.modelsIndirectBuffer
			),
			.pad1 = 0u,
			.pad2 = 0u,
			.pad3 = 0u,
		};

		u32 const colorHandle = vkof::transient_image_storage_handle({
			.image = colorTarget,
			.mipLevel = 0u,
		});
		u32 const ptOutputHandle = vkof::transient_image_storage_handle({
			.image = ptOutputTarget,
			.mipLevel = 0u,
		});

		// -- frame-budget window: which tiles resolve covers this frame,
		// sliding over the active region -- the full screen, or the focus
		// rect's tile-snapped sub-grid. a probe's pixel may be anywhere, so
		// it forces the full screen; budget-off and a missing cost estimate
		// merely force the whole region, which is exactly the focus payoff:
		// a small region redispatched in full every frame. View/Clear are
		// always full-screen by construction (see FrameMode's comment);
		// pathTraceReset also forces it -- a partial window can't correctly
		// reset pixels outside it (they'd silently keep blending onto stale
		// data forever, not just until the window sweeps back), and unlike
		// camera movement this is a one-off event, so paying for one
		// full-screen frame here is cheap enough
		u32 const tileGridW = (sRenderW + kPtTileSize - 1u) / kPtTileSize;
		u32 const tileGridH = (sRenderH + kPtTileSize - 1u) / kPtTileSize;
		bool const fullScreen = !sFocusActive || probeRequested;
		u32 regionTileX = 0u;
		u32 regionTileY = 0u;
		u32 regionGridW = tileGridW;
		u32 regionGridH = tileGridH;
		if (!fullScreen) {
			FocusTileBounds const b = fnFocusTileBounds(tileGridW, tileGridH);
			regionTileX = b.x0;
			regionTileY = b.y0;
			regionGridW = b.x1 - b.x0 + 1u;
			regionGridH = b.y1 - b.y0 + 1u;
		}
		u32 const tileTotal = regionGridW * regionGridH;
		u32 tileFirst = 0u;
		u32 tileCount = tileTotal;
		// frozen frames dispatch no resolve at all, so they must not slide
		// the window either (the full-coverage default is inert under freeze)
		bool const forceFullCoverage = (
			frameMode != FrameMode::PathTrace || pathTraceReset
			|| probeRequested || !runResolve
			|| !sBudgetEnabled || sMsPerTileEma <= 0.0
		);
		if (!forceFullCoverage) {
			// window size is only (re)computed when starting a new hold --
			// held steady at sTileWindowCount for sTileWindowSpp frames in a
			// row so the same region actually accumulates that many samples
			// before the cursor moves on
			if (sTileWindowFrames == 0u) {
				u32 const want = std::clamp(
					(u32)((double)sBudgetTargetMs * 0.9 / sMsPerTileEma),
					1u, tileTotal
				);
				// smooth the window size itself: per-tile cost genuinely varies
				// as the window slides across cheap/expensive screen regions, so
				// the raw quotient flaps; the ema turns that into a gradual
				// drift, and does so asymmetrically -- climbing back toward a
				// higher tile count (more perf headroom) only 3% of the way per
				// frame, so a lucky cheap frame can't talk it into overcommitting,
				// but dropping toward a lower tile count (a cost spike eating
				// into the budget) 40% of the way per frame, so a frame that's
				// gone over budget recovers in ~2 frames instead of ~7
				sTileKSmooth = (
					sTileKSmooth <= 0.0 ? (double)want
						: (double)want > sTileKSmooth
							? sTileKSmooth * 0.97 + (double)want * 0.03
							: sTileKSmooth * 0.6 + (double)want * 0.4
				);
				sTileKSmooth = std::min(sTileKSmooth, (double)tileTotal);
				sTileWindowCount = (
					std::clamp((u32)(sTileKSmooth + 0.5), 1u, tileTotal)
				);
			}
			tileFirst = sTileCursor % tileTotal;
			// the region can shrink mid-hold (a focus rect much smaller than
			// the last full-screen window) -- clamp so the cursor never wraps
			// duplicate dispatches over the same tiles
			tileCount = std::min(sTileWindowCount, tileTotal);
			if (++sTileWindowFrames >= (u32)std::max(sTileWindowSpp, 1)) {
				sTileCursor = (tileFirst + tileCount) % tileTotal;
				sTileWindowFrames = 0u;
			}
		}
		sTileKLast = runResolve ? tileCount : 0u;

		vkof::RenderNode nodes[3];
		u32 nodeCount = 0u;

		// -- resolve node: trace + shade into colorTarget
		vkof::RenderNode resolveNode {};
		if (runResolve) {
			resolveNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_image({
				.node = resolveNode,
				.image = colorTarget,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_callback({
				.node = resolveNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					// full screen coverage collapses to a single screen-sized
					// dispatch (and, via a zero salt, seeds identically to
					// the pre-budget path); a partial window -- or any focus
					// region, even fully covered -- goes one dispatch per
					// tile with a pixel origin
					bool const full = fullScreen && tileCount >= tileTotal;
					u32 const dispatchCount = full ? 1u : tileCount;
					for (u32 i = 0u; i < dispatchCount; ++i) {
						u32 const tile = (tileFirst + i) % tileTotal;
						u32 const originX = (
							full ? 0u
								: (regionTileX + tile % regionGridW)
									* kPtTileSize
						);
						u32 const originY = (
							full ? 0u
								: (regionTileY + tile / regionGridW)
									* kPtTileSize
						);
						u32 const extentX = (
							full ? sRenderW
								: std::min(kPtTileSize, sRenderW - originX)
						);
						u32 const extentY = (
							full ? sRenderH
								: std::min(kPtTileSize, sRenderH - originY)
						);
						GpuResolvePC const resolvePC = {
							.bluenoiseVa = (
								vkof::buffer_virtual_address(
									bluenoise.handleBuffer
								)
							),
							.outputImageHandle = colorHandle,
							.frameIndex = sFrameIndex,
							.bluenoiseCount = ponder::Bluenoise::kCount,
							.kullaContyEnergyHandle = (
								energyTables.kullaContyEnergyHandle
							),
							.zeltnerLtcParamHandle = (
								zeltnerTables.zeltnerLtcParamHandle
							),
							.tileOriginX = originX,
							.tileOriginY = originY,
							.pad0 = 0u,
							.pad1 = 0u,
						};
						vkof::cmd_dispatch_pushconst(
							vkof::CmdDispatchPushconst {
								.cmd = cmd,
								.pipeline = resolvePipeline,
								.push = resolvePC,
								.threadgroupSize = u32v3 { 16u, 16u, 1u },
								.invocationCount = (
									u32v3 { extentX, extentY, 1u }
								),
							}
						);
					}
				},
			});
			nodes[nodeCount++] = resolveNode;
		}

		// -- accumulate node: temporal average + exposure + tonemap
		vkof::RenderNode const accumNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::graphics })
		);
		if (!freezeAccum && frameMode != FrameMode::Clear) {
			// frozen frames never read the (possibly undispatched, possibly
			// probe-only) resolve output, so the dependency is dropped too.
			// Clear mode doesn't read it either -- resolve.comp doesn't run
			// that frame at all
			vkof::render_node_add_image({
				.node = accumNode,
				.image = colorTarget,
				.access = vkof::RenderNodeAccess::read,
			});
		}
		vkof::render_node_add_persistent_image({
			.node = accumNode,
			.image = ptAccumImage,
			.access = vkof::RenderNodeAccess::readWrite,
		});
		vkof::render_node_add_image({
			.node = accumNode,
			.image = ptOutputTarget,
			.access = vkof::RenderNodeAccess::write,
		});
		vkof::render_node_callback({
			.node = accumNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				GpuPtAccumulatePC const accumPC = {
					.inputHandle = colorHandle,
					.accumHandle = ptAccumStorageHandle,
					.outputHandle = ptOutputHandle,
					.reset = (
						frameMode == FrameMode::PathTrace && pathTraceReset
					) ? 1u : 0u,
					.sppCounterVa = (
						vkof::buffer_virtual_address(sppCounterBuffer)
					),
					.freeze = freezeAccum ? 1u : 0u,
					.maxSpp = sMaxSpp > 0 ? (u32)sMaxSpp : 0u,
					.tileFirst = tileFirst,
					.tileCount = tileCount,
					.tileGridW = regionGridW,
					.tileTotal = tileTotal,
					.regionTileX = regionTileX,
					.regionTileY = regionTileY,
				};
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = ptAccumulatePipeline,
					.push = accumPC,
					.threadgroupSize = u32v3 { 16u, 16u, 1u },
					.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
				});
			},
		});

		// -- nan-check node: tallies broken pixels in the raw radiance image;
		// skipped on frozen frames (nothing new is accumulated, and a
		// probe-only resolve's discarded sample shouldn't count), and on
		// View/Clear (View's output isn't real path-traced radiance to
		// sanity-check; Clear produces no new data at all)
		vkof::RenderNode nanCheckNode {};
		if (!freezeAccum && frameMode == FrameMode::PathTrace) {
			nanCheckNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_image({
				.node = nanCheckNode,
				.image = colorTarget,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_callback({
				.node = nanCheckNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					// mirrors the resolve dispatch window exactly: only the
					// tiles traced this frame hold fresh radiance, the rest
					// of colorTarget is stale and would recount old pixels
					bool const full = fullScreen && tileCount >= tileTotal;
					u32 const dispatchCount = full ? 1u : tileCount;
					for (u32 i = 0u; i < dispatchCount; ++i) {
						u32 const tile = (tileFirst + i) % tileTotal;
						u32 const originX = (
							full ? 0u
								: (regionTileX + tile % regionGridW)
									* kPtTileSize
						);
						u32 const originY = (
							full ? 0u
								: (regionTileY + tile / regionGridW)
									* kPtTileSize
						);
						u32 const extentX = (
							full ? sRenderW
								: std::min(kPtTileSize, sRenderW - originX)
						);
						u32 const extentY = (
							full ? sRenderH
								: std::min(kPtTileSize, sRenderH - originY)
						);
						GpuNanCheckPC const nanCheckPC = {
							.countsVa = (
								vkof::buffer_virtual_address(nanCountsBuffer)
							),
							.inputHandle = colorHandle,
							.width = sRenderW,
							.height = sRenderH,
							.tileOriginX = originX,
							.tileOriginY = originY,
							.pad0 = 0u,
						};
						vkof::cmd_dispatch_pushconst(
							vkof::CmdDispatchPushconst {
								.cmd = cmd,
								.pipeline = nanCheckPipeline,
								.push = nanCheckPC,
								.threadgroupSize = u32v3 { 16u, 16u, 1u },
								.invocationCount = (
									u32v3 { extentX, extentY, 1u }
								),
							}
						);
					}
				},
			});
			nodes[nodeCount++] = nanCheckNode;
		}
		nodes[nodeCount++] = accumNode;

		if (pathTraceReset || frameMode == FrameMode::Clear) {
			fnZeroNanCounts();
			fnZeroSppCounter();
		}

		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(nodes, nodeCount),
			.rootPushconstant = srat::slice_as_bytes(globalPC),
			.finalImage = headless ? vkof::TransientImage { 0 } : ptOutputTarget,
		});
		if (runResolve) {
			vkof::render_node_destroy(resolveNode);
		}
		if (!freezeAccum) {
			vkof::render_node_destroy(nanCheckNode);
		}
		vkof::render_node_destroy(accumNode);

		// -- frame-budget measurement: the timestamp readback inside
		// render_graph_execute above is for the previous use of this frame
		// slot, so node_gpu_ms(0) -- resolve is always declared first when
		// it runs -- pairs with the window recorded there two frames ago
		{
			u32 const slot = sFrameIndex % 2u;
			TileFrameHistory const prev = sTileHistory[slot];
			if (prev.resolveRan && prev.tileCount > 0u) {
				double const gpuMs = vkof::node_gpu_ms(0u);
				if (gpuMs > 0.0) {
					sResolveMsLast = gpuMs;
					double const perTile = gpuMs / (double)prev.tileCount;
					sMsPerTileEma = (
						sMsPerTileEma <= 0.0
							? perTile
							: sMsPerTileEma * 0.8 + perTile * 0.2
					);
				}
			}
			sTileHistory[slot] = TileFrameHistory {
				.resolveRan = runResolve,
				.tileCount = runResolve ? tileCount : 0u,
			};
		}

		++sFrameIndex;
		sAccumFrames = (
			(pathTraceReset || frameMode == FrameMode::Clear)
				? 1u
				: (freezeAccum ? sAccumFrames : sAccumFrames + 1u)
		);
	};

	if (headless) {
		fnUploadFrameState();
		for (u32 frame = 0u; frame < std::max(args.spp, 1u); ++frame) {
			fnRenderFrame(
				FrameMode::PathTrace, /*pathTraceReset=*/frame == 0u, false, false
			);
		}
		vkof::screenshot(ptAccumImage, args.screenshotPath, sExposure);
		u32 const validationCount = vkof::validation_message_count();
		if (validationCount > 0u) {
			printf("%u validation message(s) during render:\n", validationCount);
			for (u32 i = 0u; i < validationCount; ++i) {
				printf("  %s\n", vkof::validation_message(i));
			}
		}
		{
			// screenshot above waited for device idle, so the counters are
			// final here
			u32 const * const nanCounts = (
				reinterpret_cast<u32 const *>(
					vkof::buffer_host_address(nanCountsBuffer).ptr()
				)
			);
			if (nanCounts[0] > 0u || nanCounts[1] > 0u) {
				printf(
					"ERROR: %u NaN / %u Inf pixels in the framebuffer\n",
					nanCounts[0], nanCounts[1]
				);
			}
		}
		printf(
			"wrote %s (%ux%u, %u spp)\n",
			args.screenshotPath, args.width, args.height, args.spp
		);
	} else {
		GLFWwindow * const window = vkof::window();
		glfwSetScrollCallback(window, scroll_callback);
		f64 prevMouseX = 0.0;
		f64 prevMouseY = 0.0;
		glfwGetCursorPos(window, &prevMouseX, &prevMouseY);
		bool rightMouseWasDown = false;
		bool tabWasDown = false;
		// in-flight focus-region drag (right button held in focus mode);
		// anchor is in window coordinates, same space as glfwGetCursorPos
		bool focusDragActive = false;
		ImVec2 focusDragAnchor { 0.0f, 0.0f };
		// scroll adjusts this in first-person mode instead of zooming; scaled
		// by boundsExtent at use so 1.0 crosses the scene in ~2s
		f32 fpSpeedMul = 0.5f;

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}

			vkof::imgui_begin();

			f64 curMouseX, curMouseY;
			glfwGetCursorPos(window, &curMouseX, &curMouseY);
			f32 const dx = (f32)(curMouseX - prevMouseX);
			f32 const dy = (f32)(curMouseY - prevMouseY);
			prevMouseX = curMouseX;
			prevMouseY = curMouseY;

			bool cameraMoved = false;
			// set below by the imguizmo drag; shares cameraMoved's View-mode
			// treatment in the frame-mode state machine further down
			bool gizmoUsed = false;
			bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
			bool const keysFree = !ImGui::GetIO().WantCaptureKeyboard;
			// imgui uses tab for widget focus navigation, so the toggle only
			// fires when the ui isn't capturing the keyboard
			bool const tabDown = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
			if (keysFree && tabDown && !tabWasDown) {
				sFirstPerson = !sFirstPerson;
				if (sFirstPerson) {
					fnSyncFpFromOrbit();
				} else {
					fnSyncOrbitFromFp();
				}
			}
			tabWasDown = tabDown;
			if (
				mouseFree
				&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)
					== GLFW_PRESS
				&& (dx != 0.0f || dy != 0.0f)
			) {
				// signs chosen so a drag turns the view the same way in both
				// modes (yaw/pitch mirror azimuth/elevation, see the sync
				// lambdas)
				if (sFirstPerson) {
					srat::camera_fp_look(camFp, dx * 0.005f, -dy * 0.005f);
				} else {
					srat::camera_orbit_rotate(cam, -dx * 0.005f, dy * 0.005f);
				}
				cameraMoved = true;
			}
			if (
				!sFirstPerson
				&& mouseFree
				&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)
					== GLFW_PRESS
				&& (dx != 0.0f || dy != 0.0f)
			) {
				srat::camera_orbit_pan(
					cam,
					-dx * 0.002f * cam.distance,
					dy * 0.002f * cam.distance
				);
				cameraMoved = true;
			}
			if (sFirstPerson && keysFree) {
				f32v3 move = { 0.0f, 0.0f, 0.0f };
				if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { move.z += 1.0f; }
				if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { move.z -= 1.0f; }
				if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { move.x += 1.0f; }
				if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { move.x -= 1.0f; }
				if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { move.y += 1.0f; }
				if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { move.y -= 1.0f; }
				if (move.x != 0.0f || move.y != 0.0f || move.z != 0.0f) {
					f32 const speed = (
						fnActiveBoundsExtent() * fpSpeedMul
						* ImGui::GetIO().DeltaTime
					);
					srat::camera_fp_move(camFp, move * speed);
					cameraMoved = true;
				}
			}
			// right-click is moded: normally it probes the pixel under the
			// cursor (press edge only, not held) for its material index; in
			// focus mode it drags out the focus region instead, so the probe
			// is unavailable while that checkbox is on. glfwGetCursorPos is
			// in window/screen coordinates, the same space as
			// glfwGetWindowSize, not the internal render resolution
			// (sRenderW/sRenderH) -- the finalImage blit scales between the
			// two (see skRenderResPresets' comment), so both need rescaling
			// to land on the right texels
			bool const rightMouseDown = (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
			);
			if (sFocusMode) {
				// only the press edge needs mouseFree; a drag that wanders
				// over an imgui window mid-hold still finishes cleanly
				if (mouseFree && rightMouseDown && !rightMouseWasDown) {
					focusDragActive = true;
					focusDragAnchor = ImVec2((f32)curMouseX, (f32)curMouseY);
				}
				if (focusDragActive && rightMouseDown) {
					ImGui::GetForegroundDrawList()->AddRect(
						focusDragAnchor,
						ImVec2((f32)curMouseX, (f32)curMouseY),
						IM_COL32(255, 200, 0, 255)
					);
				}
				if (focusDragActive && !rightMouseDown) {
					focusDragActive = false;
					i32 windowW, windowH;
					glfwGetWindowSize(window, &windowW, &windowH);
					f32 const dragW = (
						std::abs((f32)curMouseX - focusDragAnchor.x)
					);
					f32 const dragH = (
						std::abs((f32)curMouseY - focusDragAnchor.y)
					);
					if (dragW < 4.0f || dragH < 4.0f) {
						// a click (or degenerate sliver) clears back to the
						// full screen -- the natural "never mind" gesture
						// since the probe is off in this mode
						sFocusActive = false;
					} else if (windowW > 0 && windowH > 0) {
						f32 const x0 = (
							std::min(focusDragAnchor.x, (f32)curMouseX)
							/ (f32)windowW
						);
						f32 const y0 = (
							std::min(focusDragAnchor.y, (f32)curMouseY)
							/ (f32)windowH
						);
						f32 const x1 = (
							std::max(focusDragAnchor.x, (f32)curMouseX)
							/ (f32)windowW
						);
						f32 const y1 = (
							std::max(focusDragAnchor.y, (f32)curMouseY)
							/ (f32)windowH
						);
						sFocusMin = {
							std::clamp(x0, 0.0f, 1.0f),
							std::clamp(y0, 0.0f, 1.0f),
						};
						sFocusMax = {
							std::clamp(x1, 0.0f, 1.0f),
							std::clamp(y1, 0.0f, 1.0f),
						};
						sFocusActive = true;
					}
				}
			} else if (mouseFree && rightMouseDown && !rightMouseWasDown) {
				i32 windowW, windowH;
				glfwGetWindowSize(window, &windowW, &windowH);
				if (windowW > 0 && windowH > 0) {
					sProbePixel = {
						std::clamp(
							(i32)(curMouseX * (f64)sRenderW / (f64)windowW),
							0, (i32)sRenderW - 1
						),
						std::clamp(
							(i32)(curMouseY * (f64)sRenderH / (f64)windowH),
							0, (i32)sRenderH - 1
						),
					};
					fnResetProbeResult();
				}
			}
			rightMouseWasDown = rightMouseDown;
			// committed focus region: outline the tile-snapped rect that
			// actually dispatches (fnFocusTileBounds, shared with
			// fnRenderFrame) and dim everything outside it, so the frozen
			// surroundings are unmistakable
			if (sFocusActive) {
				i32 windowW, windowH;
				glfwGetWindowSize(window, &windowW, &windowH);
				if (windowW > 0 && windowH > 0) {
					u32 const gridW = (
						(sRenderW + kPtTileSize - 1u) / kPtTileSize
					);
					u32 const gridH = (
						(sRenderH + kPtTileSize - 1u) / kPtTileSize
					);
					FocusTileBounds const b = fnFocusTileBounds(gridW, gridH);
					f32 const sx = (f32)windowW / (f32)sRenderW;
					f32 const sy = (f32)windowH / (f32)sRenderH;
					ImVec2 const p0 {
						(f32)(b.x0 * kPtTileSize) * sx,
						(f32)(b.y0 * kPtTileSize) * sy,
					};
					ImVec2 const p1 {
						(f32)std::min((b.x1 + 1u) * kPtTileSize, sRenderW)
							* sx,
						(f32)std::min((b.y1 + 1u) * kPtTileSize, sRenderH)
							* sy,
					};
					ImDrawList * const dl = ImGui::GetForegroundDrawList();
					u32 const dim = IM_COL32(0, 0, 0, 96);
					f32 const winW = (f32)windowW;
					f32 const winH = (f32)windowH;
					dl->AddRectFilled(
						ImVec2(0.0f, 0.0f), ImVec2(winW, p0.y), dim
					);
					dl->AddRectFilled(
						ImVec2(0.0f, p1.y), ImVec2(winW, winH), dim
					);
					dl->AddRectFilled(
						ImVec2(0.0f, p0.y), ImVec2(p0.x, p1.y), dim
					);
					dl->AddRectFilled(
						ImVec2(p1.x, p0.y), ImVec2(winW, p1.y), dim
					);
					dl->AddRect(p0, p1, IM_COL32(255, 200, 0, 255));
				}
			}
			if (mouseFree && sScrollDelta != 0.0f) {
				if (sFirstPerson) {
					fpSpeedMul *= expf(sScrollDelta * 0.1f);
				} else {
					srat::camera_orbit_zoom(
						cam, -sScrollDelta * 0.1f * cam.distance
					);
					cameraMoved = true;
				}
			}
			sScrollDelta = 0.0f;

			bool settingsChanged = false;
			// set by clicking a model in the list below; the load happens
			// after the imgui window is finished, outside its scope
			std::string pendingLoadPath;
			// same deferral for the exr environment-map picker
			std::string pendingEnvMapPath;
			// and for the vdb picker / the two unload buttons (file view only)
			std::string pendingVdbPath;
			bool pendingUnloadModel = false;
			bool pendingUnloadVdb = false;
			// stage mode: new/open/save/unload buttons and the add-instance
			// pickers all defer the same way, for the same reason (loads
			// wait for device idle and do their own upload work)
			bool pendingNewStage = false;
			std::string pendingOpenStagePath;
			bool pendingSaveStage = false;
			bool pendingUnloadStage = false;
			std::string pendingAddGltfPath;
			std::string pendingAddVdbPath;
			bool stageInstancesChanged = false;
			stage::FocusRequest focusRequest;
			{
				ImGui::Begin("viewer");
				if (sMode == ViewerMode::StageEdit) {
					ImGui::Text(
						"stage: %s",
						sStage.name.empty() ? "(untitled)" : sStage.name.c_str()
					);
				} else {
					ImGui::Text(
						"%s",
						sCurrentModelPath.empty()
							? "(no file loaded)" : sCurrentModelPath.c_str()
					);
				}
				ImGui::Text(
					"%u spp | frame %u | %.1f fps",
					sAccumFrames, sFrameIndex, ImGui::GetIO().Framerate
				);
				ImGui::SeparatorText("stage");
				if (sMode == ViewerMode::FileView) {
					if (ImGui::Button("new stage")) {
						pendingNewStage = true;
					}
					ImGui::SameLine();
					ImGui::InputText(
						"##openstagepath", sStagePathBuffer,
						sizeof(sStagePathBuffer)
					);
					ImGui::SameLine();
					if (ImGui::Button("open")) {
						pendingOpenStagePath = sStagePathBuffer;
					}
				} else {
					if (ImGui::Button("unload stage")) {
						pendingUnloadStage = true;
					}
					if (ImGui::Button("save stage")) {
						pendingSaveStage = true;
					}
					ImGui::SameLine();
					ImGui::InputText(
						"##savestagepath", sStagePathBuffer,
						sizeof(sStagePathBuffer)
					);
					if (ImGui::Button("unselect")) {
						stage::stage_clear_selection(sStage);
					}
				}

				if (sMode == ViewerMode::StageEdit) {
					// editing (transforms, material overrides, per-blob
					// sigma, cameras, lights, environment) all lives here --
					// see lib/stage/src/stage-imgui.cpp
					usize const gltfCountBefore = sStage.gltfInstances.size();
					stageInstancesChanged |= (
						stage::stage_imgui_edit(sStage, &focusRequest)
					);
					// the outliner's own "delete" button removes an instance
					// directly -- a gltf removal needs the full blas rebuild
					// (not just tlas), same as add-instance below
					if (sStage.gltfInstances.size() != gltfCountBefore) {
						stage_render_rebuild_blases(sStageRender, sStage);
						stage_render_rebuild_tlas(sStageRender, sStage);
						stageInstancesChanged = false;
						settingsChanged = true;
					}

					ImGui::SeparatorText("add gltf instance");
					if (sModelList.empty()) {
						ImGui::TextDisabled(
							"none found; list modelPaths in %s",
							settingsPath.string().c_str()
						);
					} else {
						pendingAddGltfPath = asset_list_draw(
							sModelList, "", "add model list", 120.0f
						);
					}
					ImGui::SeparatorText("add vdb instance");
					if (sVdbList.empty()) {
						ImGui::TextDisabled(
							"none found; list vdbPaths in %s",
							settingsPath.string().c_str()
						);
					} else {
						pendingAddVdbPath = asset_list_draw(
							sVdbList, "", "add vdb list", 100.0f
						);
					}
				} else {
					ImGui::SeparatorText("models");
					if (sCurrentModelPath.empty()) {
						ImGui::TextDisabled("no model loaded");
					} else if (ImGui::Button("unload model")) {
						pendingUnloadModel = true;
					}
					if (sModelList.empty()) {
						ImGui::TextDisabled(
							"none found; list modelPaths in %s",
							settingsPath.string().c_str()
						);
					} else {
						// grouped by modelPaths root (scan_asset_paths keeps
						// each root's entries contiguous and sorted), one
						// collapsing header per configured library directory
						pendingLoadPath = asset_list_draw(
							sModelList, sCurrentModelPath, "model list", 160.0f
						);
					}
					ImGui::SeparatorText("vdb");
					if (sCurrentVdbPath.empty()) {
						ImGui::TextDisabled("no vdb loaded");
					} else if (ImGui::Button("unload vdb")) {
						pendingUnloadVdb = true;
					}
					if (sVdbList.empty()) {
						ImGui::TextDisabled(
							"none found; list vdbPaths in %s",
							settingsPath.string().c_str()
						);
					} else {
						pendingVdbPath = asset_list_draw(
							sVdbList, sCurrentVdbPath, "vdb list", 100.0f
						);
					}
				}
				// file view only: stage mode edits the same knobs per-blob,
				// in stage_imgui_edit's own vdb instance inspector instead
				if (sMode == ViewerMode::FileView && sVdb.handleBuffer.id != 0u) {
					settingsChanged |= ImGui::SliderFloat(
						"sigma##vdb", &sVdbSigmaScale, 0.0f, 100.0f, "%.3f",
						ImGuiSliderFlags_Logarithmic
					);
					settingsChanged |= ImGui::SliderFloat(
						"droplet##vdb", &sVdbDropletDiameter, 5.0f, 50.0f
					);
					settingsChanged |= ImGui::ColorEdit3(
						"albedo##vdb", &sVdbAlbedo.x
					);
				}
				ImGui::SeparatorText("fog");
				settingsChanged |= (
					ImGui::Checkbox("fog enabled", &sFogEnabled)
				);
				if (sFogEnabled) {
					settingsChanged |= ImGui::SliderFloat(
						"sigma##fog", &sFogSigma, 0.0001f, 10.0f, "%.4f",
						ImGuiSliderFlags_Logarithmic
					);
					settingsChanged |= ImGui::ColorEdit3(
						"albedo##fog", &sFogAlbedo.x
					);
					settingsChanged |= ImGui::SliderFloat(
						"distance##fog", &sFogDistanceMax, 1.0f, 100000.0f,
						"%.1f", ImGuiSliderFlags_Logarithmic
					);
				}
				ImGui::SeparatorText("camera");
				if (sFirstPerson) {
					ImGui::Text(
						"pos %.2f %.2f %.2f yaw %.3f pitch %.3f speed %.2f",
						camFp.position.x, camFp.position.y, camFp.position.z,
						camFp.yaw, camFp.pitch, fpSpeedMul
					);
				} else {
					ImGui::Text(
						"azimuth %.3f elevation %.3f distance %.3f fov %.3f",
						cam.azimuth, cam.elevation, cam.distance, cam.fovY
					);
				}
				ImGui::TextDisabled(
					sFirstPerson
						? "tab: orbit | wasd+qe move, drag look, scroll speed"
						: "tab: first person"
				);
				settingsChanged |= (
					ImGui::SliderFloat("fov", &cam.fovY, 0.2f, 2.5f)
				);
				camFp.fovY = cam.fovY;
				ImGui::SeparatorText("environment");
				// indices match ENV_MODE_* in environment.glsl
				static char const * const skEnvModes[] = {
					"furnace", "checkerboard", "black", "hdrmap (exr)",
				};
				settingsChanged |= (
					ImGui::Combo("mode", &sEnvMode, skEnvModes, 4)
				);
				if (sEnvMode == 3) { // ENV_MODE_HDRMAP
					// which .exr backs hdrmap mode; picks from settings.json's
					// environmentMapPaths scan. the actual load is deferred
					// out of the imgui scope (like the model switch below)
					std::string const curLabel = (
						sCurrentEnvMapPath.empty()
							? std::string("[none]")
							: std::filesystem::path(sCurrentEnvMapPath)
								.filename().string()
					);
					if (ImGui::BeginCombo("exr map", curLabel.c_str())) {
						if (sEnvMapList.empty()) {
							ImGui::TextDisabled(
								"none found; list environmentMapPaths in %s",
								settingsPath.string().c_str()
							);
						}
						for (ModelEntry const & entry : sEnvMapList) {
							std::string const label = (
								entry.path.filename().string()
							);
							bool const isCurrent = (
								entry.path.string() == sCurrentEnvMapPath
							);
							// full path as id: filenames can repeat across
							// roots
							ImGui::PushID(entry.path.string().c_str());
							if (
								ImGui::Selectable(label.c_str(), isCurrent)
								&& !isCurrent
							) {
								pendingEnvMapPath = entry.path.string();
							}
							if (isCurrent) {
								ImGui::SetItemDefaultFocus();
							}
							ImGui::PopID();
						}
						ImGui::EndCombo();
					}
				}
				settingsChanged |= (
					ImGui::SliderFloat("env intensity", &sEnvIntensity, 0.0f, 4.0f)
				);
				// backdrop only (primary-ray misses); the lighting the scene
				// receives keeps using env intensity above
				settingsChanged |= (
					ImGui::SliderFloat(
						"background intensity", &sEnvBackgroundIntensity,
						0.0f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic
					)
				);
				// no-op outside hdrmap mode; see GpuGlobalExtended.envNeeEnabled
				settingsChanged |= (
					ImGui::Checkbox("env NEE (MIS)", &sEnvNeeEnabled)
				);
				settingsChanged |= (
					ImGui::SliderFloat(
						"env rotation", &sEnvRotation, 0.0f, 6.28318530718f
					)
				);
				// biased -- see GpuGlobalExtended.fireflyClampLuminance
				settingsChanged |= (
					ImGui::Checkbox("firefly clamp", &sFireflyClampEnabled)
				);
				if (sFireflyClampEnabled) {
					settingsChanged |= (
						ImGui::SliderFloat(
							"clamp luminance", &sFireflyClampLuminance,
							0.1f, 200.0f, "%.1f", ImGuiSliderFlags_Logarithmic
						)
					);
				}
				ImGui::SeparatorText("render");
				{
					// -1 when the current internal resolution matches no
					// preset (a custom --resolution startup value)
					i32 resIdx = -1;
					for (i32 i = 0; i < (i32)IM_ARRAYSIZE(skRenderResPresets); ++i) {
						if (
							skRenderResPresets[i].w == sRenderW
							&& skRenderResPresets[i].h == sRenderH
						) {
							resIdx = i;
							break;
						}
					}
					i32 newIdx = resIdx;
					ImGui::Combo(
						"resolution",
						&newIdx,
						[](void * data, int idx) -> char const * {
							return (
								static_cast<RenderResPreset const *>(data)[idx]
								.label
							);
						},
						(void *)skRenderResPresets,
						(i32)IM_ARRAYSIZE(skRenderResPresets)
					);
					if (newIdx != resIdx && newIdx >= 0) {
						sRenderW = skRenderResPresets[newIdx].w;
						sRenderH = skRenderResPresets[newIdx].h;
						// waits for device idle and resizes every transient
						// image in place; the accumulation buffer is a
						// persistent image, so it is recreated by hand
						vkof::render_extent_set(sRenderW, sRenderH);
						vkof::image_destroy(ptAccumImage);
						fnCreateAccumImage();
						cam.aspect = (f32)sRenderW / (f32)sRenderH;
						camFp.aspect = cam.aspect;
						settingsChanged = true;
					}
				}
				ImGui::SliderFloat("exposure", &sExposure, 0.01f, 8.0f);
				settingsChanged |= (
					ImGui::Checkbox("antialias", &sAntialias)
				);
				settingsChanged |= ImGui::SliderInt(
					"max spp", &sMaxSpp, 0, 4096,
					sMaxSpp == 0 ? "0 (unlimited)" : "%d"
				);
				if (sMaxSpp > 0) {
					// gpu-side completion tally (pt-accumulate.comp's
					// atomicAdd): each pixel bumps it once, on the frame it
					// reaches the max-spp target, so a finished render reads
					// renderWidth * renderHeight. read off the host mapping a
					// frame or so stale, fine for a progress display
					u32 const completedPixels = fnCompletedPixels();
					u32 const totalPixels = sRenderW * sRenderH;
					char overlay[64];
					snprintf(
						overlay, sizeof(overlay), "%u / %u pixels",
						completedPixels, totalPixels
					);
					ImGui::ProgressBar(
						std::min(
							(f32)completedPixels / (f32)totalPixels, 1.0f
						),
						ImVec2(-FLT_MIN, 0.0f),
						overlay
					);
				}
				// deliberately not part of settingsChanged: accumulation is
				// rate-independent, so toggling the budget or moving the
				// target mid-run just changes how fast tiles refresh
				ImGui::Checkbox("frame budget", &sBudgetEnabled);
				if (sBudgetEnabled) {
					ImGui::SliderFloat(
						"target ms", &sBudgetTargetMs, 1.0f, 64.0f, "%.1f ms"
					);
					ImGui::SliderInt(
						"hold spp", &sTileWindowSpp, 1, 256
					);
					u32 const budgetTileTotal = (
						((sRenderW + kPtTileSize - 1u) / kPtTileSize)
						* ((sRenderH + kPtTileSize - 1u) / kPtTileSize)
					);
					ImGui::Text(
						"resolve %.2f ms\n%u / %u tiles",
						sResolveMsLast, sTileKLast, budgetTileTotal
					);
				}
				// not settingsChanged either: the focus region only redirects
				// which tiles refresh, accumulation everywhere stays valid
				ImGui::Checkbox(
					"focus region (right-click drag)", &sFocusMode
				);
				if (sFocusActive) {
					ImGui::SameLine();
					if (ImGui::SmallButton("clear region")) {
						sFocusActive = false;
					}
					u32 const gridW = (
						(sRenderW + kPtTileSize - 1u) / kPtTileSize
					);
					u32 const gridH = (
						(sRenderH + kPtTileSize - 1u) / kPtTileSize
					);
					FocusTileBounds const b = fnFocusTileBounds(gridW, gridH);
					ImGui::Text(
						"focused %u x %u tiles (outside pixels frozen)",
						b.x1 - b.x0 + 1u, b.y1 - b.y0 + 1u
					);
				} else if (sFocusMode) {
					ImGui::TextDisabled(
						"right-click drag the view to set a region"
					);
				}
				if (ImGui::Button("screenshot")) {
					sScreenshotPending = true;
				}
				if (!sLastScreenshotPath.empty()) {
					ImGui::SameLine();
					ImGui::TextDisabled(
						"%s",
						std::filesystem::path(sLastScreenshotPath)
							.filename().string().c_str()
					);
				}
				u32 const validationCount = vkof::validation_message_count();
				if (validationCount > 0u) {
					ImGui::TextColored(
						ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
						"%u validation message(s)", validationCount
					);
				}
				{
					u32 const * const nanCounts = (
						reinterpret_cast<u32 const *>(
							vkof::buffer_host_address(nanCountsBuffer).ptr()
						)
					);
					if (nanCounts[0] > 0u || nanCounts[1] > 0u) {
						ImGui::TextColored(
							ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
							"%u NaN / %u Inf pixels since accum reset",
							nanCounts[0], nanCounts[1]
						);
					}
				}
				bool const openMaterialPopup = sMaterialPopupPending;
				sMaterialPopupPending = false;
				for (std::string const & msg : sProbeMessages) {
					ImGui::TextColored(
						ImVec4(1.0f, 0.6f, 0.1f, 1.0f), "%s", msg.c_str()
					);
				}
				ImGui::End();
				// OpenPopup/BeginPopup must share the same id-stack context
				// to resolve to the same popup id -- calling OpenPopup while
				// still inside the "viewer" window above hashes the id
				// against that window's id stack, which BeginPopup below
				// (called with no window active) never matches, so the
				// popup would silently never open
				if (openMaterialPopup) {
					ImGui::OpenPopup("material inspector");
				}
			}
			if (sMode == ViewerMode::StageEdit) {
				// the render fills the whole glfw window (no embedded 3d
				// viewport sub-region), same assumption the probe-pixel
				// mouse mapping above already makes
				i32 windowW = 0, windowH = 0;
				glfwGetWindowSize(window, &windowW, &windowH);
				f32m44 const view = (
					sFirstPerson
						? srat::camera_fp_view(camFp)
						: srat::camera_orbit_view(cam)
				);
				f32m44 gizmoProj = (
					sFirstPerson
						? srat::camera_fp_proj(camFp)
						: srat::camera_orbit_proj(cam)
				);
				// f32m44_perspective bakes in a y-flip for vulkan's clip
				// space (see srat/core-math.hpp); ImGuizmo's own screen-
				// space math assumes the opengl convention it was written
				// against, so handing it the real (vulkan) projection
				// double-flips y relative to what's actually on screen --
				// the gizmo tracks the wrong screen position and drifts
				// off-screen as the camera moves. undo just that one term
				// for the copy imguizmo sees; the real proj used for
				// rendering (view/proj above, and everywhere else) is
				// untouched
				gizmoProj.m[5] = -gizmoProj.m[5];
				gizmoUsed = stage::stage_imgui_gizmo_edit(
					sStage, view, gizmoProj,
					f32v4 { 0.0f, 0.0f, (f32)windowW, (f32)windowH }
				);
				if (gizmoUsed) {
					stage_render_rebuild_tlas(sStageRender, sStage);
					settingsChanged = true;
				}
			}
			if (ImGui::BeginPopup("material inspector")) {
				bool matChanged = false;
				if (sMode == ViewerMode::StageEdit) {
					// stage_imgui_material_inspector applies + uploads
					// internally (it's the same path as any other stage
					// material edit), unlike file view's below
					if (
						sPickedModelDrawIndex >= 0
						&& (usize)sPickedModelDrawIndex
							< sStageRender.modelDrawIndexToInstance.size()
					) {
						usize const instIdx = (
							sStageRender.modelDrawIndexToInstance[
								(usize)sPickedModelDrawIndex
							]
						);
						matChanged = stage::stage_imgui_material_inspector(
							sStage, instIdx, (u32)sPickedMaterialIndex, &focusRequest
						);
					}
				} else {
					matChanged = fnDrawMaterialInspector(
						sScene.scene, sScene.materialsOverride, sPickedMaterialIndex
					);
					if (matChanged) {
						mor::scene_gpu_materials_upload(sScene.materialsOverride);
					}
				}
				if (matChanged) {
					settingsChanged = true;
				}
				ImGui::EndPopup();
			}

			// -- environment map switch, deferred out of the imgui scope like
			// the model switch below (the load waits for device idle and
			// executes its own upload work)
			if (!pendingEnvMapPath.empty()) {
				settingsChanged |= fnLoadEnvMap(pendingEnvMapPath);
			}
			// hdrmap mode with nothing resident (just selected in the ui, or
			// the picked file failed to load): try each scanned map in order,
			// fall back to furnace if none loads
			if (sEnvMode == 3 && sEnvTables.width == 0u) {
				for (ModelEntry const & entry : sEnvMapList) {
					if (fnLoadEnvMap(entry.path.string())) {
						break;
					}
				}
				if (sEnvTables.width == 0u) {
					printf(
						"no loadable exr environment map; back to furnace\n"
					);
					sEnvMode = 0;
				}
				settingsChanged = true;
			}

			// file view only ever shows one asset at a time now -- loading a
			// model auto-unloads any active vdb and vice versa (stage mode
			// is where more than one thing at once belongs)
			if (!pendingLoadPath.empty() && !sCurrentVdbPath.empty()) {
				pendingUnloadVdb = true;
			}
			if (!pendingVdbPath.empty() && !sCurrentModelPath.empty()) {
				pendingUnloadModel = true;
			}

			// -- model switch: tear down every piece of scene state and
			// rebuild from the file; the libraries are rescanned so new files
			// under the settings.json paths show up without a restart
			if (!pendingLoadPath.empty()) {
				scene_state_destroy(sScene);
				sModelList = scan_asset_paths(
					settingsPath, { "modelPaths" }, { ".gltf", ".glb" },
					/*warnMissingKey=*/true
				);
				sEnvMapList = scan_asset_paths(
					settingsPath, { "environmentMapPaths", "environmentMaps" },
					{ ".exr" }, /*warnMissingKey=*/false
				);
				sCurrentModelPath = pendingLoadPath;
				sScene = scene_state_create(sCurrentModelPath.c_str());
				if (sScene.blas.id == 0u) {
					break;
				}
				// re-fit framing to the new bounds; the orbit orientation
				// (azimuth/elevation) is the one piece of camera state kept
				fnReframeCamera(cam);
				// snap first-person to the reframed orbit too, so a model
				// load doesn't leave the camera at the old model's scale and
				// position
				fnSyncFpFromOrbit();
				settingsChanged = true;
			}

			// -- model unload: back to the empty scene (zero-instance tlas,
			// every ray reaches the environment); a loaded vdb stays and
			// takes over the camera framing
			if (pendingUnloadModel) {
				scene_state_destroy(sScene);
				sCurrentModelPath.clear();
				sScene = scene_state_create(nullptr);
				if (!sVdb.blob.empty()) {
					vdb_bounds(
						sVdb, sScene.boundsCenter, sScene.boundsExtent
					);
				}
				fnReframeCamera(cam);
				fnSyncFpFromOrbit();
				settingsChanged = true;
			}

			// -- vdb switch/unload: one grid per scene for now, torn down
			// wholesale like a model switch. the old buffer may still be
			// referenced by in-flight frames, so idle before destroying
			if (pendingUnloadVdb || !pendingVdbPath.empty()) {
				vkof::device_wait_idle();
				if (sVdb.gpuBuffer.id != 0u) {
					vkof::buffer_destroy(sVdb.gpuBuffer);
				}
				if (sVdb.handleBuffer.id != 0u) {
					vkof::buffer_destroy(sVdb.handleBuffer);
				}
				sVdb = {};
				sCurrentVdbPath.clear();
				if (!pendingVdbPath.empty()) {
					// openvdb throws on unreadable files; a failed pick
					// should drop back to no-vdb, not kill the viewer
					try {
						sVdb = ponder::vdb_load(pendingVdbPath.c_str());
					} catch (std::exception const & e) {
						printf(
							"failed to load vdb '%s': %s\n",
							pendingVdbPath.c_str(), e.what()
						);
					}
					if (sVdb.gpuBuffer.id != 0u) {
						sCurrentVdbPath = pendingVdbPath;
					}
					sVdbList = scan_asset_paths(
						settingsPath, { "vdbPaths" }, { ".vdb" },
						/*warnMissingKey=*/false
					);
				}
				// with no mesh in the scene the grid owns the framing, same
				// as the vdb-only startup path
				if (sCurrentModelPath.empty() && !sVdb.blob.empty()) {
					vdb_bounds(
						sVdb, sScene.boundsCenter, sScene.boundsExtent
					);
					fnReframeCamera(cam);
					fnSyncFpFromOrbit();
				}
				settingsChanged = true;
			}

			// -- new stage: an empty, unsaved stage; leaves file view's own
			// scene/vdb alone (they're simply not rendered while in stage
			// mode -- see the mode-gated buffer selection above)
			if (pendingNewStage) {
				sMode = ViewerMode::StageEdit;
				sStage = stage::Stage {};
				sStage.environment.mode = stage::EnvironmentMode::Furnace;
				stage_render_rebuild_blases(sStageRender, sStage);
				stage_render_rebuild_tlas(sStageRender, sStage);
				sStagePathBuffer[0] = '\0';
				fnApplyStageEnvironment();
				fnReframeCamera(cam);
				fnSyncFpFromOrbit();
				settingsChanged = true;
			}

			// -- open stage: parse + load every instance the file describes
			if (!pendingOpenStagePath.empty()) {
				stage::Stage loaded = (
					stage::stage_load(pendingOpenStagePath.c_str())
				);
				if (loaded.sourceDirectory.empty()) {
					printf(
						"failed to open stage '%s'\n",
						pendingOpenStagePath.c_str()
					);
					stage::stage_destroy(loaded);
				} else {
					sMode = ViewerMode::StageEdit;
					sStage = std::move(loaded);
					stage_render_rebuild_blases(sStageRender, sStage);
					stage_render_rebuild_tlas(sStageRender, sStage);
					std::strncpy(
						sStagePathBuffer, pendingOpenStagePath.c_str(),
						sizeof(sStagePathBuffer) - 1
					);
					fnApplyStageEnvironment();
					fnReframeCamera(cam);
					fnSyncFpFromOrbit();
					settingsChanged = true;
				}
			}

			// -- save stage: only reachable in stage mode (the button lives
			// there); writes to whatever path is currently in the buffer
			if (pendingSaveStage) {
				if (sStagePathBuffer[0] == '\0') {
					printf("no save path set\n");
				} else if (!stage::stage_save(sStage, sStagePathBuffer)) {
					printf("failed to save stage '%s'\n", sStagePathBuffer);
				}
			}

			// -- unload stage: tear down every stage-owned gpu resource and
			// drop back to file view; file view's own sScene/sVdb were never
			// touched while in stage mode, so there's nothing to reload.
			// stage_render_destroy frees sStageRender.tlas, which was the
			// bound acceleration structure this whole time (stage mode
			// rebinds to its own tlas on entry) -- rebind sScene's own tlas
			// immediately after, or the very next frame traces a ray query
			// against a freed handle
			if (pendingUnloadStage) {
				stage_render_destroy(sStageRender);
				stage::stage_destroy(sStage);
				vkof::acceleration_structure_set_tlas(sScene.tlas);
				sMode = ViewerMode::FileView;
				fnReframeCamera(cam);
				fnSyncFpFromOrbit();
				settingsChanged = true;
			}

			// -- add instance: append, then rebuild both the blas list
			// (structural change) and the tlas (so the new instance actually
			// renders this frame)
			if (!pendingAddGltfPath.empty()) {
				if (
					stage::stage_add_gltf_instance(
						sStage, pendingAddGltfPath.c_str()
					)
				) {
					stage_render_rebuild_blases(sStageRender, sStage);
					stage_render_rebuild_tlas(sStageRender, sStage);
					stageInstancesChanged = false;
					settingsChanged = true;
				} else {
					printf(
						"failed to add gltf instance '%s'\n",
						pendingAddGltfPath.c_str()
					);
				}
			}
			if (!pendingAddVdbPath.empty()) {
				if (
					stage::stage_add_vdb_instance(sStage, pendingAddVdbPath.c_str())
				) {
					settingsChanged = true;
				} else {
					printf(
						"failed to add vdb instance '%s'\n",
						pendingAddVdbPath.c_str()
					);
				}
			}

			// stage_imgui_edit's own edits (transform drags, visibility
			// toggles, material overrides, light/camera fields) don't touch
			// the instance list itself, only need the tlas + models-
			// indirect buffer refreshed to match -- unless an add-instance
			// above already did a full rebuild this frame
			if (stageInstancesChanged) {
				stage_render_rebuild_tlas(sStageRender, sStage);
				settingsChanged = true;
			}

			// mirror the live (shared, file-view-style) environment state
			// into sStage.environment every frame while in stage mode, so
			// "save stage" always persists whatever's currently live --
			// there's no separate stage-side environment editor to keep in
			// sync from instead (see fnApplyStageEnvironment's comment)
			if (sMode == ViewerMode::StageEdit) {
				sStage.environment.mode = (stage::EnvironmentMode)sEnvMode;
				sStage.environment.path = sCurrentEnvMapPath;
				sStage.environment.rotationRadians = sEnvRotation;
				sStage.environment.intensity = sEnvIntensity;
			}

			// -- focus: a gltf instance's "focus" button (stage_imgui_edit)
			// re-frames the orbit camera to that instance's world bounds,
			// same math as fnReframeCamera but against the clicked
			// instance's bounds instead of the whole active scene's
			if (focusRequest.requested) {
				cam.target = focusRequest.center;
				cam.distance = focusRequest.extent * 1.5f;
				cam.near = focusRequest.extent * 0.001f;
				cam.far = focusRequest.extent * 100.0f;
				fnSyncFpFromOrbit();
				settingsChanged = true;
			}

			// captured before fnUploadFrameState consumes (and clears) the
			// probe request, so the max-spp freeze path below can still run
			// resolve.comp for the material inspector
			bool const probeRequested = sProbePixel.x >= 0;
			// every reset trigger -- camera/gizmo drags, any settings edit,
			// startup, shader reload -- goes through View first instead of
			// forcing an immediate full-screen pathtrace reset: at high
			// resolutions that single-frame full dispatch is a multi-second
			// hang, not a hitch
			bool const otherReset = (
				settingsChanged || sFrameIndex == 0u || vkof::shader_reloaded()
			);
			if (probeRequested) {
				sFrameMode = FrameMode::PathTrace;
			} else if (cameraMoved || gizmoUsed || otherReset) {
				sFrameMode = FrameMode::View;
				sViewIdleSeconds = 0.0f;
			} else if (sFrameMode == FrameMode::View) {
				sViewIdleSeconds += ImGui::GetIO().DeltaTime;
				if (sViewIdleSeconds >= skCameraIdleSeconds) {
					sFrameMode = FrameMode::Clear;
				}
			} else {
				sFrameMode = FrameMode::PathTrace;
			}
			fnUploadFrameState();
			bool const pathTraceReset = (
				otherReset && sFrameMode == FrameMode::PathTrace
			);
			bool const freezeAccum = (
				sFrameMode == FrameMode::PathTrace && !pathTraceReset
				&& sMaxSpp > 0 && fnCompletedPixels() >= sRenderW * sRenderH
			);
			fnRenderFrame(sFrameMode, pathTraceReset, freezeAccum, probeRequested);
			if (sScreenshotPending) {
				sScreenshotPending = false;
				std::filesystem::path const screenshotDir = (
					settingsPath.parent_path() / "screenshots"
				);
				std::error_code ec;
				std::filesystem::create_directories(screenshotDir, ec);
				time_t const now = time(nullptr);
				tm tmv {};
				localtime_r(&now, &tmv);
				char stamp[32];
				strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tmv);
				std::string const outPath = (
					(
						screenshotDir / (
							std::filesystem::path(sCurrentModelPath)
								.stem().string()
							+ "-" + stamp + "-"
							+ std::to_string(sAccumFrames) + "spp.exr"
						)
					).string()
				);
				// blocks until the gpu is idle, so it captures exactly the
				// frame rendered above
				vkof::screenshot(ptAccumImage, outPath.c_str(), sExposure);
				sLastScreenshotPath = outPath;
				printf("wrote %s\n", outPath.c_str());
			}
			// see sProbeMessages' declaration for why this has to happen
			// right here rather than in next iteration's imgui block.
			// probe_reset() (rather than relying on render_graph_execute to
			// auto-clear) comes right after the copy so the validation
			// layer's debug-printf readback -- which isn't guaranteed to
			// land synchronously within the render_graph_execute call whose
			// fence covers it -- gets the largest possible window to have
			// arrived before its message is discarded
			sProbeMessages.clear();
			for (u32 i = 0u, count = vkof::probe_message_count(); i < count; ++i) {
				sProbeMessages.emplace_back(vkof::probe_message(i));
			}
			vkof::probe_reset();

			// see sMaterialPopupPending's declaration for why this has to
			// happen right here too, mirroring sProbeMessages above
			{
				GpuProbeResult const * const probeResult = (
					reinterpret_cast<GpuProbeResult const *>(
						vkof::buffer_host_address(probeResultBuffer).ptr()
					)
				);
				if (probeResult->modelDrawIndex >= 0) {
					sPickedModelDrawIndex = probeResult->modelDrawIndex;
					sPickedMaterialIndex = probeResult->materialIndex;
					sMaterialPopupPending = true;
					fnResetProbeResult();
				}
			}
		}
	}

	scene_state_destroy(sScene);
	if (sVdb.gpuBuffer.id != 0u) {
		vkof::buffer_destroy(sVdb.gpuBuffer);
	}
	if (sVdb.handleBuffer.id != 0u) {
		vkof::buffer_destroy(sVdb.handleBuffer);
	}
	stage_render_destroy(sStageRender);
	stage::stage_destroy(sStage);
	vkof::buffer_destroy(nanCountsBuffer);
	vkof::buffer_destroy(nanProbeCounterBuffer);
	vkof::buffer_destroy(probeResultBuffer);
	vkof::buffer_destroy(sppCounterBuffer);
	vkof::image_destroy(ptAccumImage);
	mor::sampler_cache_destroy();
	{
		ponder::Bluenoise bn = bluenoise;
		ponder::bluenoise_destroy(bn);
		ponder::ZeltnerTables zt = zeltnerTables;
		ponder::zeltner_tables_destroy(zt);
		ponder::EnergyTables et = energyTables;
		ponder::energy_tables_destroy(et);
		if (sEnvTables.width != 0u) {
			ponder::environment_tables_destroy(sEnvTables);
		}
		vkof::buffer_destroy(envMapHandlesBuffer);
	}
	vkof::pipeline_destroy(nanCheckPipeline);
	vkof::pipeline_destroy(ptAccumulatePipeline);
	vkof::pipeline_destroy(resolvePipeline);
	vkof::shutdown();
	return 0;
}
