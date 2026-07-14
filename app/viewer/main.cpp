// ponder viewer: opens a gltf/glb file via mor and renders it with the
// lib/ponder OpenPBR path tracer (app/shaders/resolve.comp +
// pt-accumulate.comp). windowed by default; --screenshot renders headless
// for --spp frames and writes a png instead.

#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>

#include <ponder/bluenoise.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/zeltner-tables.hpp>

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
#include <cmath>
#include <cstdio>
#include <cstring>
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
};

// material inspector popup rows; one row per GpuMorMaterial scalar/rgb
// parameter, "[tex]" suffix when that parameter also carries a texture
// (the constant factor still applies as a multiplier on top of it)
void fnDrawMaterialParam(
	char const * const label, GpuMorMaterialComponent1 const & c
) {
	ImGui::Text(
		"%-34s %7.4f%s", label, c.r, c.texture != 0u ? " [tex]" : ""
	);
}
void fnDrawMaterialParam(
	char const * const label, GpuMorMaterialComponent3 const & c
) {
	ImGui::Text(
		"%-34s %7.4f %7.4f %7.4f%s",
		label, c.rgb.x, c.rgb.y, c.rgb.z, c.texture != 0u ? " [tex]" : ""
	);
}

// every scalar/rgb parameter via mor-shared.h's MOR_MATERIAL_ALL_PARAMS
// x-macro, so a new material parameter shows up here automatically rather
// than needing a second hand-maintained field list
void fnDrawMaterialInspector(mor::Scene const & scene, i32 const materialIndex) {
	if (
		materialIndex < 0
		|| (u32)materialIndex >= mor::scene_material_count(scene)
	) {
		ImGui::TextDisabled("no material at that pixel");
		return;
	}
	std::string const name = (
		mor::scene_material_name(scene, (u32)materialIndex)
	);
	GpuMorMaterial const mat = (
		mor::scene_material_get(scene, (u32)materialIndex)
	);
	ImGui::Text("#%d %s", materialIndex, name.c_str());
	ImGui::Separator();
#define X(Type, field, lo, hi) fnDrawMaterialParam(#field, mat.field);
	MOR_MATERIAL_ALL_PARAMS(X)
#undef X
	ImGui::Separator();
	ImGui::Text("alphaCutoff %.4f", mat.alphaCutoff);
}

struct ViewerArgs {
	char const * gltfPath = nullptr;
	char const * screenshotPath = nullptr;
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
	// ENV_MODE_* in environment.glsl: 0=furnace, 1=checkerboard, 2=black
	i32 envMode = 0;
	// vulkan validation layers (debug_printf capture, validation_message,
	// the material inspector probe -- see vkof::init) off by default:
	// meaningful chunk of windowed startup time most interactive use
	// doesn't need. headless (--screenshot) always has it on regardless,
	// since the numeric tests depend on it
	bool debug = false;
};

void print_usage() {
	printf(
		"usage: viewer [path.gltf|.glb]\n"
		"  (no path: opens the first model found via settings.json's\n"
		"   modelPaths; the path is required with --screenshot)\n"
		"  [--screenshot <out.png>]  render headless, write png, exit\n"
		"  [--camera <azimuth,elevation,distance,fovY>]  orbit camera\n"
		"      (radians / world units; distance 0 = auto-fit)\n"
		"  [--resolution <w>x<h>]  render resolution (default 1280x720)\n"
		"  [--spp <n>]  accumulation frames before --screenshot (default 256)\n"
		"  [--env-intensity <f>]  environment radiance (default 1.0)\n"
		"  [--env-mode <furnace|checkerboard|black>]  (default furnace)\n"
		"  [--debug]  enable vulkan validation layers (slower startup;\n"
		"      needed for the material inspector probe / NaN probes /\n"
		"      validation messages -- always on with --screenshot)\n"
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
		} else if (strcmp(argv[i], "--env-mode") == 0 && i + 1 < argc) {
			char const * const mode = argv[++i];
			if (strcmp(mode, "furnace") == 0) {
				out.envMode = 0;
			} else if (strcmp(mode, "checkerboard") == 0) {
				out.envMode = 1;
			} else if (strcmp(mode, "black") == 0) {
				out.envMode = 2;
			} else {
				printf(
					"--env-mode expects furnace, checkerboard, or black\n"
				);
				return false;
			}
		} else if (strcmp(argv[i], "--debug") == 0) {
			out.debug = true;
		} else if (argv[i][0] == '-') {
			printf("unknown option '%s'\n", argv[i]);
			return false;
		} else {
			out.gltfPath = argv[i];
		}
	}
	// windowed mode can fall back to the settings.json model library, but a
	// headless screenshot needs to know exactly what to render
	if (out.screenshotPath != nullptr && out.gltfPath == nullptr) {
		return false;
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
	vkof::AccelerationStructureBlas blas {};
	vkof::AccelerationStructureTlas tlas {};
	f32v3 boundsCenter {};
	f32 boundsExtent = 0.01f;
	vkof::Buffer debugPcBuffer {};
	vkof::Buffer modelsIndirectBuffer {};
};

// loads a gltf/glb, uploads it, builds its blas + single-instance tlas, and
// binds the tlas; blas.id == 0 on failure (partial state already destroyed)
SceneState scene_state_create(char const * const path) {
	SceneState s;
	s.scene = mor::scene_create();
	mor::scene_load_gltf(s.scene, path);
	s.gpuScene = mor::scene_gpu_upload(s.scene);
	s.bufs = mor::scene_gpu_buffers(s.gpuScene);

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

	// single-model tlas; the scene never changes between loads, so one
	// build here is enough
	{
		vkof::RenderNode const tlasNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
		);
		vkof::TlasInstance const instance = {
			.blas = s.blas,
			.transform = f32m44_identity(),
			.instanceCustomIndex = 0u,
			.rayMask = 0xFFu,
		};
		vkof::render_node_callback({
			.node = tlasNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				vkof::tlas_build(
					cmd,
					s.tlas,
					srat::slice<vkof::TlasInstance const>(&instance, 1u)
				);
			},
		});
		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(&tlasNode, 1u),
			.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
		});
		vkof::render_node_destroy(tlasNode);
	}
	vkof::acceleration_structure_set_tlas(s.tlas);

	s.debugPcBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuDebugPC),
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
	mor::scene_gpu_destroy(s.gpuScene);
	mor::scene_destroy(s.scene);
	s = {};
}

// a scanned model, tagged with which modelPaths root directory it was
// found under so the imgui list (below) can group by root instead of
// showing one flat list across every configured library
struct ModelEntry {
	std::filesystem::path root;
	std::filesystem::path path;
};

// settings.json at the repo root selects where models come from:
//   { "modelPaths": ["assets/models", "/absolute/path/too"] }
// each listed directory (relative paths resolve against the settings
// file's own directory) is searched recursively for .gltf/.glb files
std::vector<ModelEntry> scan_models(
	std::filesystem::path const & settingsPath
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
	auto const pathsIt = doc.FindMember("modelPaths");
	if (pathsIt == doc.MemberEnd() || !pathsIt->value.IsArray()) {
		printf(
			"%s has no \"modelPaths\" array\n", settingsPath.string().c_str()
		);
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
			printf("modelPaths entry is not a directory: %s\n", dir.string().c_str());
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
			if (p.extension() == ".gltf" || p.extension() == ".glb") {
				found.push_back(p);
			}
		}
		std::sort(found.begin(), found.end());
		for (std::filesystem::path & p : found) {
			out.push_back({ dir, std::move(p) });
		}
	}
	return out;
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

	// -- settings.json model library at the repo root; rescanned on every
	// model load so newly added files show up without a restart
	std::filesystem::path const settingsPath = (
		appDir.parent_path().parent_path() / "settings.json"
	);
	std::vector<ModelEntry> sModelList = scan_models(settingsPath);

	// -- scene; an explicit positional path wins, otherwise the library's
	// first model
	std::string sCurrentModelPath = (
		args.gltfPath != nullptr
			? std::string(args.gltfPath)
			: (sModelList.empty() ? std::string() : sModelList[0].path.string())
	);
	if (sCurrentModelPath.empty()) {
		printf(
			"no model to load: pass a .gltf/.glb path or list modelPaths in %s\n",
			settingsPath.string().c_str()
		);
		return 1;
	}
	SceneState sScene = scene_state_create(sCurrentModelPath.c_str());
	if (sScene.blas.id == 0u) {
		return 1;
	}

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
	i32 sEnvMode = args.envMode;
	f32 sExposure = 1.0f;
	bool sAntialias = true;

	// -- material inspector: right-click sets sProbePixel for the next
	// upload/render, resolve.comp debugPrintfEXT's the hit's material
	// index back, and the imgui block below parses it out of
	// vkof::probe_message and opens a popup. (-1,-1) means no probe
	// pending; fnUploadFrameState resets it right after upload, so a probe
	// only ever fires for the one frame right after a click
	i32v2 sProbePixel { -1, -1 };
	i32 sPickedMaterialIndex = -1;
	// vkof::probe_message is only "stable until the next render_graph_execute
	// call" (see vkof.hpp); the gpu is pipelined deeply enough that a given
	// frame's debugPrintfEXT output doesn't actually surface until a later
	// render_graph_execute call waits on its fence, and that call clears
	// the buffer again before refilling it -- so reading probe_message
	// live from the imgui block (which always runs before this iteration's
	// own fnRenderFrame) never once caught a message; it was always either
	// not-yet-arrived or already overwritten (confirmed empirically:
	// probe_message_count() read 0 there every time). copied out
	// immediately after fnRenderFrame (below), the one point in the loop
	// guaranteed to run right after this iteration's render_graph_execute
	// actually returns, avoids the race entirely
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

	auto const fnUploadFrameState = [&]() {
		// each frame gets a fresh NAN_PROBE_LIMIT_PER_FRAME budget
		fnZeroNanProbeCounter();
		GpuDebugPC const debugPC = {
			.envIntensity = sEnvIntensity,
			.renderWidth = sRenderW,
			.renderHeight = sRenderH,
			.envMode = sEnvMode,
			.antialias = sAntialias ? 1u : 0u,
			.nanProbeCounterVa = (
				vkof::buffer_virtual_address(nanProbeCounterBuffer)
			),
			.probePixel = sProbePixel,
		};
		// one-shot: consumed by this upload, so it doesn't keep firing
		// resolve.comp's debugPrintfEXT on every subsequent frame
		sProbePixel = { -1, -1 };
		vkof::buffer_upload({
			.buffer = sScene.debugPcBuffer,
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(debugPC),
		});
		GpuResolveModelIndirect const modelDesc = {
			.meshlets = sScene.bufs.meshlets,
			.materials = sScene.bufs.materials,
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
	};

	u32 sFrameIndex = 0u;
	u32 sAccumFrames = 0u;

	auto const fnRenderFrame = [&](bool const resetAccum) {
		GpuGlobalPC const globalPC = {
			.time = 0.0f,
			.cameraPos = srat::camera_orbit_eye(cam),
			.exposure = sExposure,
			.pad0 = 0.0f,
			.viewProj = (
				srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam)
			),
			.debug = vkof::buffer_virtual_address(sScene.debugPcBuffer),
			.models = (
				vkof::buffer_virtual_address(sScene.modelsIndirectBuffer)
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

		// -- resolve node: trace + shade into colorTarget
		vkof::RenderNode const resolveNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::graphics })
		);
		vkof::render_node_add_image({
			.node = resolveNode,
			.image = colorTarget,
			.access = vkof::RenderNodeAccess::write,
		});
		vkof::render_node_callback({
			.node = resolveNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				GpuResolvePC const resolvePC = {
					.bluenoiseVa = (
						vkof::buffer_virtual_address(bluenoise.handleBuffer)
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
					.pad0 = 0u,
				};
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = resolvePipeline,
					.push = resolvePC,
					.threadgroupSize = u32v3 { 16u, 16u, 1u },
					.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
				});
			},
		});

		// -- accumulate node: temporal average + exposure + tonemap
		vkof::RenderNode const accumNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::graphics })
		);
		vkof::render_node_add_image({
			.node = accumNode,
			.image = colorTarget,
			.access = vkof::RenderNodeAccess::read,
		});
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
					.reset = resetAccum ? 1u : 0u,
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

		// -- nan-check node: tallies broken pixels in the raw radiance image
		vkof::RenderNode const nanCheckNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::graphics })
		);
		vkof::render_node_add_image({
			.node = nanCheckNode,
			.image = colorTarget,
			.access = vkof::RenderNodeAccess::read,
		});
		vkof::render_node_callback({
			.node = nanCheckNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				GpuNanCheckPC const nanCheckPC = {
					.countsVa = vkof::buffer_virtual_address(nanCountsBuffer),
					.inputHandle = colorHandle,
					.width = sRenderW,
					.height = sRenderH,
					.pad0 = 0u,
				};
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = nanCheckPipeline,
					.push = nanCheckPC,
					.threadgroupSize = u32v3 { 16u, 16u, 1u },
					.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
				});
			},
		});

		if (resetAccum) {
			fnZeroNanCounts();
		}

		vkof::RenderNode const nodes[] = { resolveNode, nanCheckNode, accumNode };
		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(nodes, 3u),
			.rootPushconstant = srat::slice_as_bytes(globalPC),
			.finalImage = headless ? vkof::TransientImage { 0 } : ptOutputTarget,
		});
		vkof::render_node_destroy(resolveNode);
		vkof::render_node_destroy(nanCheckNode);
		vkof::render_node_destroy(accumNode);

		++sFrameIndex;
		sAccumFrames = resetAccum ? 1u : sAccumFrames + 1u;
	};

	if (headless) {
		fnUploadFrameState();
		for (u32 frame = 0u; frame < std::max(args.spp, 1u); ++frame) {
			fnRenderFrame(frame == 0u);
		}
		vkof::screenshot(ptOutputTarget, args.screenshotPath);
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
			bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
			if (
				mouseFree
				&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)
					== GLFW_PRESS
				&& (dx != 0.0f || dy != 0.0f)
			) {
				srat::camera_orbit_rotate(cam, -dx * 0.005f, dy * 0.005f);
				cameraMoved = true;
			}
			if (
				mouseFree
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
			// material inspector: right-click probes the pixel under the
			// cursor (press edge only, not held) for its material index.
			// glfwGetCursorPos is in window/screen coordinates, the same
			// space as glfwGetWindowSize, not the internal render
			// resolution (sRenderW/sRenderH) -- the finalImage blit scales
			// between the two (see skRenderResPresets' comment), so the
			// click position needs rescaling to land on the right texel
			bool const rightMouseDown = (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
			);
			if (mouseFree && rightMouseDown && !rightMouseWasDown) {
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
				}
			}
			rightMouseWasDown = rightMouseDown;
			if (mouseFree && sScrollDelta != 0.0f) {
				srat::camera_orbit_zoom(cam, -sScrollDelta * 0.1f * cam.distance);
				cameraMoved = true;
			}
			sScrollDelta = 0.0f;

			bool settingsChanged = false;
			// set by clicking a model in the list below; the load happens
			// after the imgui window is finished, outside its scope
			std::string pendingLoadPath;
			{
				ImGui::Begin("viewer");
				ImGui::Text("%s", sCurrentModelPath.c_str());
				ImGui::Text(
					"%u spp | frame %u | %.1f fps",
					sAccumFrames, sFrameIndex, ImGui::GetIO().Framerate
				);
				ImGui::SeparatorText("models");
				if (sModelList.empty()) {
					ImGui::TextDisabled(
						"none found; list modelPaths in %s",
						settingsPath.string().c_str()
					);
				} else {
					ImGui::BeginChild(
						"model list", ImVec2(0.0f, 160.0f), ImGuiChildFlags_None
					);
					// grouped by modelPaths root (scan_models keeps each
					// root's entries contiguous and sorted), one collapsing
					// header per configured library directory
					std::filesystem::path const * openRoot = nullptr;
					bool rootOpen = false;
					for (ModelEntry const & entry : sModelList) {
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
						// parent-dir/filename: distinguishes the gltf/gltf-
						// binary variants sample-asset directories usually
						// keep side by side under the same model folder
						std::string const label = (
							entry.path.parent_path().filename().string()
							+ "/" + entry.path.filename().string()
						);
						bool const isCurrent = (
							entry.path.string() == sCurrentModelPath
						);
						// full path as id: labels repeat across roots (two
						// libraries can both have an "Avocado" folder)
						ImGui::PushID(entry.path.string().c_str());
						if (
							ImGui::Selectable(label.c_str(), isCurrent)
							&& !isCurrent
						) {
							pendingLoadPath = entry.path.string();
						}
						ImGui::PopID();
					}
					ImGui::EndChild();
				}
				ImGui::SeparatorText("camera");
				ImGui::Text(
					"azimuth %.3f elevation %.3f distance %.3f fov %.3f",
					cam.azimuth, cam.elevation, cam.distance, cam.fovY
				);
				settingsChanged |= (
					ImGui::SliderFloat("fov", &cam.fovY, 0.2f, 2.5f)
				);
				ImGui::SeparatorText("environment");
				static char const * const skEnvModes[] = {
					"furnace", "checkerboard", "black",
				};
				settingsChanged |= (
					ImGui::Combo("mode", &sEnvMode, skEnvModes, 3)
				);
				settingsChanged |= (
					ImGui::SliderFloat("env intensity", &sEnvIntensity, 0.0f, 4.0f)
				);
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
						settingsChanged = true;
					}
				}
				ImGui::SliderFloat("exposure", &sExposure, 0.01f, 8.0f);
				settingsChanged |= (
					ImGui::Checkbox("antialias", &sAntialias)
				);
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
				bool openMaterialPopup = false;
				printf("probe messages: %zu\n", sProbeMessages.size());
				for (std::string const & msg : sProbeMessages) {
					printf("message: %s\n", msg.c_str());
					i32 materialIndex = -1;
					if (
						sscanf(msg.c_str(), "MODEL ID: %d", &materialIndex) == 1
					) {
						sPickedMaterialIndex = materialIndex;
						openMaterialPopup = true;
						continue;
					}
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
			if (ImGui::BeginPopup("material inspector")) {
				fnDrawMaterialInspector(sScene.scene, sPickedMaterialIndex);
				ImGui::EndPopup();
			}

			// -- model switch: tear down every piece of scene state and
			// rebuild from the file; the library is rescanned so new files
			// under the settings.json paths show up without a restart
			if (!pendingLoadPath.empty()) {
				scene_state_destroy(sScene);
				sModelList = scan_models(settingsPath);
				sCurrentModelPath = pendingLoadPath;
				sScene = scene_state_create(sCurrentModelPath.c_str());
				if (sScene.blas.id == 0u) {
					break;
				}
				// re-fit framing to the new bounds; the orbit orientation
				// (azimuth/elevation) is the one piece of camera state kept
				cam.target = sScene.boundsCenter;
				cam.distance = sScene.boundsExtent * 1.5f;
				cam.near = sScene.boundsExtent * 0.001f;
				cam.far = sScene.boundsExtent * 100.0f;
				settingsChanged = true;
			}

			fnUploadFrameState();
			bool const resetAccum = (
				cameraMoved || settingsChanged || sFrameIndex == 0u
				|| vkof::shader_reloaded()
			);
			fnRenderFrame(resetAccum);
			// see sProbeMessages' declaration for why this has to happen
			// right here rather than in next iteration's imgui block
			sProbeMessages.clear();
			printf("probe message count: %u\n", vkof::probe_message_count());
			for (u32 i = 0u, count = vkof::probe_message_count(); i < count; ++i) {
				sProbeMessages.emplace_back(vkof::probe_message(i));
			}
		}
	}

	scene_state_destroy(sScene);
	vkof::buffer_destroy(nanCountsBuffer);
	vkof::buffer_destroy(nanProbeCounterBuffer);
	vkof::image_destroy(ptAccumImage);
	mor::sampler_cache_destroy();
	{
		ponder::Bluenoise bn = bluenoise;
		ponder::bluenoise_destroy(bn);
		ponder::ZeltnerTables zt = zeltnerTables;
		ponder::zeltner_tables_destroy(zt);
		ponder::EnergyTables et = energyTables;
		ponder::energy_tables_destroy(et);
	}
	vkof::pipeline_destroy(nanCheckPipeline);
	vkof::pipeline_destroy(ptAccumulatePipeline);
	vkof::pipeline_destroy(resolvePipeline);
	vkof::shutdown();
	return 0;
}
