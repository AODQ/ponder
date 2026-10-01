// ponder viewer, offline reference path tracer
// TODO move stage editor completely to its own library

#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>

#include <ponder/asset-browser.hpp>
#include <ponder/bluenoise.hpp>
#include <ponder/camera-controller.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/material-inspector.hpp>
#include <ponder/vdb.hpp>
#include <ponder/zeltner-tables.hpp>

#include <stage/stage.hpp>

#include "scene-state.hpp"

// cpu-side reads of the loaded grid blob (world bbox for camera framing)
#include <nanovdb/NanoVDB.h>

#include "shaders/resolve_pc.h"
#include "shaders/pt-accumulate-pc.h"
#include "shaders/nan-check-pc.h"
#include "shaders/pt-denoise-pc.h"
#include "shaders/bloom-extract-pc.h"
#include "shaders/bloom-blur-pc.h"
#include "shaders/bloom-composite-pc.h"

#include <imgui.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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

// -----------------------------------------------------------------------------
// -- gpu handle types
// -----------------------------------------------------------------------------

// mirrors util-environment-map.glsl's EnvironmentMapHandles layout exactly;
// GpuGlobalExtended.envMap (global_pc.h) is a VA into a buffer holding one
// of these, uploaded once alongside ponder::EnvironmentTables
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 aliasProb;
	u32 aliasIndex;
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

// TODO mode this to its own ponder library so it can be used with observer
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
	// (TODO REVIEW)
	// > 0 synthesizes a normal map from base color luminance for materials
	// that ship without one; see --gen-normals
	f32 generatedNormalStrength = 0.0f;
	// (TODO REVIEW)
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

// -----------------------------------------------------------------------------
// -- command line
// -----------------------------------------------------------------------------

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
		// (TODO REVIEW)
		"  [--gen-normals <f>]  for materials with no normal map, synthesize\n"
		"      one from base color luminance at this strength; fabricated\n"
		"      relief, not real height data (default 0, off)\n"
		// (TODO REVIEW)
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
		// (TODO REVIEW)
		} else if (strcmp(argv[i], "--gen-normals") == 0 && i + 1 < argc) {
			out.generatedNormalStrength = (f32)atof(argv[++i]);
		// (TODO REVIEW)
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
	// a vdb volume alone, or a --stage file (fully self-describing, so
	// there's nothing to interactively edit -- just render it). --new-stage
	// has none of the three (an empty, unsaved stage needs a window to
	// build anything in), so it's the one case still rejected here
	if (
		out.screenshotPath != nullptr
		&& out.gltfPath == nullptr
		&& out.vdbPath == nullptr
		&& out.stagePath == nullptr
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
ponder::Vdb sVdb {};
// optional second grid, loaded alongside sVdb when the file has one named
// "temperature" -- blackbody emission source, see util-blackbody.glsl.
// gpuBuffer.id == 0 means none loaded (a file without one, or none open)
ponder::Vdb sVdbTemperature {};

// loads path's "temperature"-named grid into sVdbTemperature if present,
// leaving it default-constructed (gpuBuffer.id == 0, emission disabled)
// otherwise -- most vdb files (e.g. a plain cloud/fog scan) have no
// temperature channel at all, so this is expected to no-op often
void fnLoadVdbTemperatureIfPresent(char const * const path) {
	sVdbTemperature = {};
	// called independent of whether the file's density grid loaded, so (like
	// the reload path's own density vdb_load) a bad/unreadable path must not
	// throw uncaught here
	try {
		for (std::string const & name : ponder::vdb_grid_names(path)) {
			if (name == "temperature") {
				sVdbTemperature = ponder::vdb_load(path, "temperature");
				break;
			}
		}
	} catch (std::exception const & e) {
		printf(
			"failed to load vdb temperature grid '%s': %s\n", path, e.what()
		);
	}
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
			.aliasProb = sEnvTables.aliasProbHandle,
			.aliasIndex = sEnvTables.aliasIndexHandle,
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
	vkof::Pipeline const denoisePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/pt-denoise.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomExtractPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-extract.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomBlurPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-blur.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomCompositePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-composite.comp").string().c_str()
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
	std::vector<ponder::ModelEntry> sModelList = ponder::scan_asset_paths(
		settingsPath, { "modelPaths" }, { ".gltf", ".glb" },
		/*warnMissingKey=*/true
	);
	std::vector<ponder::ModelEntry> sEnvMapList = ponder::scan_asset_paths(
		settingsPath, { "environmentMapPaths", "environmentMaps" },
		{ ".exr" }, /*warnMissingKey=*/false
	);
	std::vector<ponder::ModelEntry> sVdbList = ponder::scan_asset_paths(
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
		// independent of whether a "density"-named grid loaded above -- a
		// fire-only vdb can have "temperature" with no density grid at all
		fnLoadVdbTemperatureIfPresent(args.vdbPath);
	}

	// -- scene; only an explicit positional path loads anything. no
	// auto-loading the model library's first entry -- windowed mode with no
	// args just opens empty (headless still requires one of
	// gltf/vdb/stage, enforced in parse_args above)
	std::string sCurrentModelPath;
	// (TODO REVIEW)
	// set before any load: mor reads it per material as the gltf comes in
	mor::scene_set_generated_normal_strength(args.generatedNormalStrength);
	// (TODO REVIEW)
	bool const startInStage = args.newStage || args.stagePath != nullptr;
	if (args.gltfPath != nullptr) {
		sCurrentModelPath = args.gltfPath;
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
	// to follow up with ponder::camera_controller_sync_fp_from_orbit()
	// themselves (camController declared later) to keep first-person in
	// sync too
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

	// g-buffer for pt-denoise.comp's edge-stopping weights: world-space hit
	// position (w = hit flag) and geometric normal, refreshed every time
	// resolve.comp resolves a tile (see resolve.comp's dedicated primary-ray
	// trace). persistent + resolution-sized like the accum image above, so
	// it follows the exact same create/resize/destroy lifecycle
	vkof::Image gbufferPosImage {};
	vkof::Image gbufferNorImage {};
	vkof::Image gbufferAlbedoImage {};
	u32 gbufferPosStorageHandle = 0u;
	u32 gbufferNorStorageHandle = 0u;
	u32 gbufferAlbedoStorageHandle = 0u;
	auto const fnCreateGbufferImages = [&]() {
		gbufferPosImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		gbufferNorImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		gbufferAlbedoImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		gbufferPosStorageHandle = vkof::image_storage_handle({
			.image = gbufferPosImage,
			.mipLevel = 0u,
		});
		gbufferNorStorageHandle = vkof::image_storage_handle({
			.image = gbufferNorImage,
			.mipLevel = 0u,
		});
		gbufferAlbedoStorageHandle = vkof::image_storage_handle({
			.image = gbufferAlbedoImage,
			.mipLevel = 0u,
		});
	};
	fnCreateGbufferImages();

	// pt-denoise.comp's a-trous passes ping-pong between these two; the
	// first pass reads ptAccumImage directly and the last pass writes
	// straight to ptOutputTarget (see kDenoisePassCount below), so only
	// intermediate passes need a home
	vkof::Image denoisePingImage {};
	vkof::Image denoisePongImage {};
	u32 denoisePingStorageHandle = 0u;
	u32 denoisePongStorageHandle = 0u;
	auto const fnCreateDenoiseImages = [&]() {
		denoisePingImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		denoisePongImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		denoisePingStorageHandle = vkof::image_storage_handle({
			.image = denoisePingImage,
			.mipLevel = 0u,
		});
		denoisePongStorageHandle = vkof::image_storage_handle({
			.image = denoisePongImage,
			.mipLevel = 0u,
		});
	};
	fnCreateDenoiseImages();
	bool sDenoiseEnabled = false;

	// bloom: extract writes into bloomAImage, then blur passes ping-pong
	// between bloomAImage/bloomBImage (bilinear sampler reads + storage
	// writes), landing in whichever the last pass wrote
	vkof::Sampler const bloomBlurSampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::linear,
		.minFilter = vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	vkof::Image bloomAImage {};
	vkof::Image bloomBImage {};
	u32 bloomAStorageHandle = 0u;
	u32 bloomBStorageHandle = 0u;
	u32 bloomASamplerHandle = 0u;
	u32 bloomBSamplerHandle = 0u;
	auto const fnCreateBloomImages = [&]() {
		bloomAImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		bloomBImage = vkof::image_create({
			.width = sRenderW,
			.height = sRenderH,
			.depth = 1u,
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		bloomAStorageHandle = vkof::image_storage_handle({
			.image = bloomAImage,
			.mipLevel = 0u,
		});
		bloomBStorageHandle = vkof::image_storage_handle({
			.image = bloomBImage,
			.mipLevel = 0u,
		});
		bloomASamplerHandle = vkof::image_sampler_handle({
			.image = bloomAImage,
			.sampler = bloomBlurSampler,
		});
		bloomBSamplerHandle = vkof::image_sampler_handle({
			.image = bloomBImage,
			.sampler = bloomBlurSampler,
		});
	};
	fnCreateBloomImages();
	bool sBloomEnabled = true;
	f32 sBloomThreshold = 1.0f;
	f32 sBloomIntensity = 0.12f;
	constexpr u32 kBloomBlurPassCount = 5u;

	// -- environment settings (imgui-editable in windowed mode)
	f32 sEnvIntensity = args.envIntensity;
	// backdrop-only scale on primary-ray misses (resolve.comp); the historic
	// hard-coded 0.001 stays the default so the environment reads as distant
	// from the scene geometry until deliberately raised
	f32 sEnvBackgroundIntensity = 1.000f;
	i32 sEnvMode = args.envMode;
	// next-event estimation toward the env map, mis-combined with the
	// existing bsdf-sampled technique; strictly lower variance at equal
	// sample count once correct, and a no-op for every env mode other than
	// hdrmap, so on-by-default is safe
	bool sEnvNeeEnabled = args.envMis;
	// diagnostic: terminate every path after its first surface hit, so a
	// grid of test spheres (different material params) can't cross-light
	// each other -- see GpuGlobalExtended.envMapOnlyMode
	bool sEnvMapOnlyMode = false;
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
	// per vdb-blob (see GpuGlobalExtended), global for now. sigmaAbsorption/
	// sigmaScattering: see stage::VdbBlob's matching fields
	f32v3 sVdbSigmaAbsorption = { 0.0f, 0.0f, 0.0f };
	f32v3 sVdbSigmaScattering = { 1.0f, 1.0f, 1.0f };
	f32 sVdbDropletDiameter = 25.0f;
	// manual HG anisotropy override; see stage::VdbBlob::phaseAnisotropyOverride
	f32 sVdbPhaseAnisotropyOverride = -2.0f;
	// blackbody temperature/emission scale; see stage::VdbBlob::
	// temperatureScale/emissionScale and util-blackbody.glsl. only has any
	// effect when sVdbTemperature is loaded (fnLoadVdbTemperatureIfPresent
	// found a "temperature" grid)
	f32 sVdbTemperatureScale = 1.0f;
	f32 sVdbEmissionScale = 1.0f;

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
	ponder::CameraController camController = ponder::camera_controller_create({
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
	});
	srat::CameraOrbit & cam = camController.orbit;
	srat::CameraFirstPerson & camFp = camController.firstPerson;
	bool & sFirstPerson = camController.isFirstPerson;

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
		ponder::camera_controller_sync_fp_from_orbit(camController);
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
		// blackbody emission: a second, optional blob named "temperature" in
		// the same stage instance -- see stage::VdbBlob::emissionScale
		stage::VdbBlob const * const activeVdbTemperatureBlob = (
			[&]() -> stage::VdbBlob const * {
				if (sMode != ViewerMode::StageEdit || sStage.vdbInstances.empty()) {
					return nullptr;
				}
				for (stage::VdbBlob const & blob : sStage.vdbInstances[0].blobs) {
					if (blob.gridName == "temperature") { return &blob; }
				}
				return nullptr;
			}()
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
			.envMapOnlyMode = sEnvMapOnlyMode ? 1u : 0u,
			.fireflyClampLuminance = (
				sFireflyClampEnabled ? sFireflyClampLuminance : 0.0f
			),
			.envBackgroundIntensity = sEnvBackgroundIntensity,
			.vdb = activeVdbHandleVa,
			.vdbWorldToLocal = activeVdbWorldToLocal,
			// stage mode's blob owns these directly (edited in
			// stage_imgui_edit's blob inspector); file view keeps using its
			// own global sliders
			.vdbSigmaAbsorption = (
				activeVdbBlob != nullptr
					? activeVdbBlob->sigmaAbsorption : sVdbSigmaAbsorption
			),
			.vdbSigmaScattering = (
				activeVdbBlob != nullptr
					? activeVdbBlob->sigmaScattering : sVdbSigmaScattering
			),
			.vdbDropletDiameter = (
				activeVdbBlob != nullptr
					? activeVdbBlob->dropletDiameter : sVdbDropletDiameter
			),
			.vdbPhaseAnisotropyOverride = (
				activeVdbBlob != nullptr
					? activeVdbBlob->phaseAnisotropyOverride
					: sVdbPhaseAnisotropyOverride
			),
			.vdbTemperatureGrid = (
				sMode == ViewerMode::StageEdit
					? (
						activeVdbTemperatureBlob != nullptr
						&& activeVdbTemperatureBlob->vdb.gpuBuffer.id != 0u
					)
						? vkof::buffer_virtual_address(
							activeVdbTemperatureBlob->vdb.gpuBuffer
						)
						: 0u
					: (sVdbTemperature.gpuBuffer.id != 0u)
						? vkof::buffer_virtual_address(sVdbTemperature.gpuBuffer)
						: 0u
			),
			.vdbTemperatureScale = (
				activeVdbTemperatureBlob != nullptr
					? activeVdbTemperatureBlob->temperatureScale
					: sVdbTemperatureScale
			),
			.vdbEmissionScale = (
				activeVdbTemperatureBlob != nullptr
					? activeVdbTemperatureBlob->emissionScale
					: sVdbEmissionScale
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
		GpuGlobalPc const globalPC = {
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

		constexpr u32 kDenoisePassCount = 4u;
		constexpr u32 kBloomNodeCount = 2u + kBloomBlurPassCount;
		vkof::RenderNode nodes[3u + kDenoisePassCount + kBloomNodeCount];
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
			vkof::render_node_add_persistent_image({
				.node = resolveNode,
				.image = gbufferPosImage,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_add_persistent_image({
				.node = resolveNode,
				.image = gbufferNorImage,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_add_persistent_image({
				.node = resolveNode,
				.image = gbufferAlbedoImage,
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
							.gbufferPosHandle = gbufferPosStorageHandle,
							.gbufferNorHandle = gbufferNorStorageHandle,
							.gbufferAlbedoHandle = gbufferAlbedoStorageHandle,
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

		// -- denoise nodes: variance-adaptive a-trous spatial filter over
		// ptAccumImage's running mean (see pt-denoise.comp's header comment
		// for why this is spatial-only, no temporal reprojection). only
		// meaningful once there's a converging pathtrace average to filter,
		// and only affects the presented ptOutputTarget -- screenshots read
		// ptAccumImage directly (see the headless path below), so this can
		// never bias a ground-truth render, only the live view
		bool const runDenoise = (
			sDenoiseEnabled && frameMode == FrameMode::PathTrace
		);
		vkof::RenderNode denoiseNodes[kDenoisePassCount] {};
		if (runDenoise) {
			vkof::Image passInputImage = ptAccumImage;
			u32 passInputHandle = ptAccumStorageHandle;
			for (u32 i = 0u; i < kDenoisePassCount; ++i) {
				bool const isFinal = (i == kDenoisePassCount - 1u);
				vkof::Image const passOutputImage = (
					isFinal ? vkof::Image {}
						: (i % 2u == 0u ? denoisePingImage : denoisePongImage)
				);
				u32 const passOutputHandle = (
					isFinal ? ptOutputHandle
						: (
							i % 2u == 0u
								? denoisePingStorageHandle
								: denoisePongStorageHandle
						)
				);
				vkof::RenderNode const node = (
					vkof::render_node_create({
						.queue = vkof::CommandQueue::graphics,
					})
				);
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = gbufferPosImage,
					.access = vkof::RenderNodeAccess::read,
				});
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = gbufferNorImage,
					.access = vkof::RenderNodeAccess::read,
				});
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = ptAccumImage,
					.access = vkof::RenderNodeAccess::read,
				});
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = gbufferAlbedoImage,
					.access = vkof::RenderNodeAccess::read,
				});
				// pass 0's input IS ptAccumImage (already declared above)
				if (passInputImage.id != ptAccumImage.id) {
					vkof::render_node_add_persistent_image({
						.node = node,
						.image = passInputImage,
						.access = vkof::RenderNodeAccess::read,
					});
				}
				if (isFinal) {
					vkof::render_node_add_image({
						.node = node,
						.image = ptOutputTarget,
						.access = vkof::RenderNodeAccess::write,
					});
				} else {
					vkof::render_node_add_persistent_image({
						.node = node,
						.image = passOutputImage,
						.access = vkof::RenderNodeAccess::write,
					});
				}
				vkof::render_node_callback({
					.node = node,
					.callback = [&, i, passInputHandle, passOutputHandle,
						isFinal
					](vkof::CommandBuffer const & cmd) {
						GpuDenoisePC const denoisePC = {
							.inputHandle = passInputHandle,
							.outputHandle = passOutputHandle,
							.accumHandle = ptAccumStorageHandle,
							.gbufferPosHandle = gbufferPosStorageHandle,
							.gbufferNorHandle = gbufferNorStorageHandle,
							.gbufferAlbedoHandle = gbufferAlbedoStorageHandle,
							.stepSize = 1u << i,
							.positionSigma = fnActiveBoundsExtent() * 0.01f,
							.isFirstPass = i == 0u ? 1u : 0u,
							.isFinalPass = isFinal ? 1u : 0u,
						};
						vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
							.cmd = cmd,
							.pipeline = denoisePipeline,
							.push = denoisePC,
							.threadgroupSize = u32v3 { 16u, 16u, 1u },
							.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
						});
					},
				});
				denoiseNodes[i] = node;
				nodes[nodeCount++] = node;
				passInputImage = passOutputImage;
				passInputHandle = passOutputHandle;
			}
		}

		// -- bloom: bright-pass extract off ptAccumImage, a few widening
		// blur passes, then an additive composite onto whatever's currently
		// in ptOutputTarget (accum's or denoise's tonemapped result --
		// bloom runs after both, so it doesn't care which)
		bool const runBloom = (
			sBloomEnabled && frameMode == FrameMode::PathTrace
		);
		vkof::RenderNode bloomNodes[kBloomNodeCount] {};
		if (runBloom) {
			u32 bloomNodeIndex = 0u;

			vkof::RenderNode const extractNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_persistent_image({
				.node = extractNode,
				.image = ptAccumImage,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_add_persistent_image({
				.node = extractNode,
				.image = bloomAImage,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_callback({
				.node = extractNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					GpuBloomExtractPC const extractPC = {
						.inputHandle = ptAccumStorageHandle,
						.outputHandle = bloomAStorageHandle,
						.threshold = sBloomThreshold,
					};
					vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
						.cmd = cmd,
						.pipeline = bloomExtractPipeline,
						.push = extractPC,
						.threadgroupSize = u32v3 { 16u, 16u, 1u },
						.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
					});
				},
			});
			bloomNodes[bloomNodeIndex++] = extractNode;
			nodes[nodeCount++] = extractNode;

			// blur passes ping-pong between bloomAImage/bloomBImage; pass 0
			// reads what extract just wrote (bloomAImage)
			vkof::Image blurSrcImage = bloomAImage;
			u32 blurSrcSamplerHandle = bloomASamplerHandle;
			for (u32 i = 0u; i < kBloomBlurPassCount; ++i) {
				bool const toB = (blurSrcImage.id == bloomAImage.id);
				vkof::Image const blurDstImage = toB ? bloomBImage : bloomAImage;
				u32 const blurDstStorageHandle = (
					toB ? bloomBStorageHandle : bloomAStorageHandle
				);
				u32 const blurDstSamplerHandle = (
					toB ? bloomBSamplerHandle : bloomASamplerHandle
				);

				vkof::RenderNode const node = (
					vkof::render_node_create({
						.queue = vkof::CommandQueue::graphics,
					})
				);
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = blurSrcImage,
					.access = vkof::RenderNodeAccess::read,
				});
				vkof::render_node_add_persistent_image({
					.node = node,
					.image = blurDstImage,
					.access = vkof::RenderNodeAccess::write,
				});
				vkof::render_node_callback({
					.node = node,
					.callback = [&, i, blurSrcSamplerHandle, blurDstStorageHandle](
						vkof::CommandBuffer const & cmd
					) {
						GpuBloomBlurPC const blurPC = {
							.inputHandle = blurSrcSamplerHandle,
							.outputHandle = blurDstStorageHandle,
							.passIndex = i,
						};
						vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
							.cmd = cmd,
							.pipeline = bloomBlurPipeline,
							.push = blurPC,
							.threadgroupSize = u32v3 { 16u, 16u, 1u },
							.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
						});
					},
				});
				bloomNodes[bloomNodeIndex++] = node;
				nodes[nodeCount++] = node;
				blurSrcImage = blurDstImage;
				blurSrcSamplerHandle = blurDstSamplerHandle;
			}
			u32 const bloomFinalStorageHandle = (
				blurSrcImage.id == bloomAImage.id
					? bloomAStorageHandle : bloomBStorageHandle
			);

			vkof::RenderNode const compositeNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_persistent_image({
				.node = compositeNode,
				.image = blurSrcImage,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_add_image({
				.node = compositeNode,
				.image = ptOutputTarget,
				.access = vkof::RenderNodeAccess::readWrite,
			});
			vkof::render_node_callback({
				.node = compositeNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					GpuBloomCompositePC const compositePC = {
						.colorHandle = ptOutputHandle,
						.bloomHandle = bloomFinalStorageHandle,
						.intensity = sBloomIntensity,
					};
					vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
						.cmd = cmd,
						.pipeline = bloomCompositePipeline,
						.push = compositePC,
						.threadgroupSize = u32v3 { 16u, 16u, 1u },
						.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
					});
				},
			});
			bloomNodes[bloomNodeIndex++] = compositeNode;
			nodes[nodeCount++] = compositeNode;
		}

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
		if (runDenoise) {
			for (u32 i = 0u; i < kDenoisePassCount; ++i) {
				vkof::render_node_destroy(denoiseNodes[i]);
			}
		}
		if (runBloom) {
			for (u32 i = 0u; i < kBloomNodeCount; ++i) {
				vkof::render_node_destroy(bloomNodes[i]);
			}
		}

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
		ponder::camera_controller_install_scroll_callback(window);
		bool rightMouseWasDown = false;
		// in-flight focus-region drag (right button held in focus mode);
		// anchor is in window coordinates, same space as glfwGetCursorPos
		bool focusDragActive = false;
		ImVec2 focusDragAnchor { 0.0f, 0.0f };

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}

			vkof::imgui_begin();

			f64 curMouseX, curMouseY;
			glfwGetCursorPos(window, &curMouseX, &curMouseY);

			// set below by the imguizmo drag; shares cameraMoved's View-mode
			// treatment in the frame-mode state machine further down
			bool gizmoUsed = false;
			bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
			bool const cameraMoved = ponder::camera_controller_update(
				camController, window, fnActiveBoundsExtent(),
				ImGui::GetIO().DeltaTime
			);
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
			// set inside the "viewer" window below (outliner delete button,
			// models-panel drag-into-instances drop target); the actual
			// blas/tlas rebuild this implies is deferred to after
			// ImGui::End() -- see that block's own comment for why
			bool gltfInstanceCountChanged = false;
			stage::FocusRequest focusRequest;
			{
				ImGui::Begin("Scene");
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
					// see lib/stage/src/stage-imgui.cpp. the outliner's own
					// "delete" button and the models panel's drag-into-
					// instances drop target both mutate sStage.gltfInstances
					// directly, from inside this still-open "viewer" window
					// -- note the actual blas/tlas rebuild this implies is
					// deliberately NOT done here (see gltfInstanceCountChanged
					// below): stage_render_rebuild_blases calls
					// vkof::device_wait_idle() and destroys/recreates gpu
					// resources, which must never run while an ImGui::Begin()
					// window is still open -- an exception or abort partway
					// through unwinds past this window's ImGui::End() below,
					// corrupting imgui's window stack ("missing End()" for
					// "viewer"). found via exactly that crash report after
					// the drag-and-drop-into-instances feature landed
					usize const gltfCountBefore = sStage.gltfInstances.size();
					stageInstancesChanged |= (
						stage::stage_imgui_edit(sStage, &focusRequest)
					);
					gltfInstanceCountChanged = (
						sStage.gltfInstances.size() != gltfCountBefore
					);

					ImGui::SeparatorText("add gltf model");
					if (sModelList.empty()) {
						ImGui::TextDisabled(
							"none found; list modelPaths in %s",
							settingsPath.string().c_str()
						);
					} else {
						pendingAddGltfPath = ponder::asset_list_draw(
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
						pendingAddVdbPath = ponder::asset_list_draw(
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
						// grouped by modelPaths root (ponder::scan_asset_paths keeps
						// each root's entries contiguous and sorted), one
						// collapsing header per configured library directory
						pendingLoadPath = ponder::asset_list_draw(
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
						pendingVdbPath = ponder::asset_list_draw(
							sVdbList, sCurrentVdbPath, "vdb list", 100.0f
						);
					}
				}
				// file view only: stage mode edits the same knobs per-blob,
				// in stage_imgui_edit's own vdb instance inspector instead
				if (sMode == ViewerMode::FileView && sVdb.handleBuffer.id != 0u) {
					settingsChanged |= ImGui::DragFloat3(
						"sigma absorption##vdb", &sVdbSigmaAbsorption.x,
						0.01f, 0.0f, 100.0f, "%.3f"
					);
					settingsChanged |= ImGui::DragFloat3(
						"sigma scattering##vdb", &sVdbSigmaScattering.x,
						0.01f, 0.0f, 100.0f, "%.3f"
					);
					settingsChanged |= ImGui::SliderFloat(
						"droplet##vdb", &sVdbDropletDiameter, 5.0f, 50.0f
					);
					{
						bool anisotropyOverrideOn = (
							sVdbPhaseAnisotropyOverride >= -1.0f
						);
						if (
							ImGui::Checkbox(
								"manual anisotropy##vdb", &anisotropyOverrideOn
							)
						) {
							sVdbPhaseAnisotropyOverride = (
								anisotropyOverrideOn ? 0.0f : -2.0f
							);
							settingsChanged = true;
						}
						if (anisotropyOverrideOn) {
							ImGui::SameLine();
							settingsChanged |= ImGui::SliderFloat(
								"g##vdb", &sVdbPhaseAnisotropyOverride, -1.0f, 1.0f
							);
						}
					}
					if (sVdbTemperature.gpuBuffer.id != 0u) {
						settingsChanged |= ImGui::DragFloat(
							"temperature scale##vdb", &sVdbTemperatureScale,
							1.0f, 0.0f, 10000.0f, "%.3f",
							ImGuiSliderFlags_Logarithmic
						);
						settingsChanged |= ImGui::DragFloat(
							"emission scale##vdb", &sVdbEmissionScale,
							0.01f, 0.0f, 1000.0f, "%.3f",
							ImGuiSliderFlags_Logarithmic
						);
					} else {
						ImGui::TextDisabled(
							"no \"temperature\" grid in this file -- no emission"
						);
					}
				}
				ImGui::End();

				ImGui::Begin("Environment");
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
						for (ponder::ModelEntry const & entry : sEnvMapList) {
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
				// only env lighting
				settingsChanged |= (
					ImGui::Checkbox("env map lighting only", &sEnvMapOnlyMode)
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
				ImGui::End();

				ImGui::Begin("Render");
				ImGui::Text(
					"%u spp | frame %u | %.1f fps",
					sAccumFrames, sFrameIndex, ImGui::GetIO().Framerate
				);
				ImGui::SeparatorText("camera");
				if (sFirstPerson) {
					ImGui::Text(
						"pos %.2f %.2f %.2f yaw %.3f pitch %.3f speed %.2f",
						camFp.position.x, camFp.position.y, camFp.position.z,
						camFp.yaw, camFp.pitch, camController.flySpeedMultiplier
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
						vkof::image_destroy(gbufferPosImage);
						vkof::image_destroy(gbufferNorImage);
						vkof::image_destroy(gbufferAlbedoImage);
						fnCreateGbufferImages();
						vkof::image_destroy(denoisePingImage);
						vkof::image_destroy(denoisePongImage);
						fnCreateDenoiseImages();
						vkof::image_destroy(bloomAImage);
						vkof::image_destroy(bloomBImage);
						fnCreateBloomImages();
						cam.aspect = (f32)sRenderW / (f32)sRenderH;
						camFp.aspect = cam.aspect;
						settingsChanged = true;
					}
				}
				ImGui::SliderFloat("exposure", &sExposure, 0.01f, 8.0f);
				settingsChanged |= (
					ImGui::Checkbox("antialias", &sAntialias)
				);
				// deliberately not part of settingsChanged: the denoiser
				// only filters ptOutputTarget for display (pt-denoise.comp),
				// it never writes ptAccumImage, so toggling it has nothing
				// to invalidate -- the next frame just runs with or without
				// the filter nodes
				ImGui::Checkbox("denoise (a-trous)", &sDenoiseEnabled);
				// also just a display filter, same reasoning as denoise above
				ImGui::Checkbox("bloom", &sBloomEnabled);
				if (sBloomEnabled) {
					ImGui::SliderFloat(
						"threshold##bloom", &sBloomThreshold, 0.0f, 8.0f
					);
					ImGui::SliderFloat(
						"intensity##bloom", &sBloomIntensity, 0.0f, 1.0f
					);
				}
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
				// OpenPopup/BeginPopup must share the same id-stack context;
				// calling OpenPopup while still inside a window hashes the id
				// against that window's stack, which BeginPopup below never
				// matches
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

				// -- drag a model from stage_imgui_edit's "gltf models"
				// panel directly into the 3d view to place a new instance.
				// an invisible window covering the whole viewport, existing
				// ONLY while a compatible drag is actually in flight
				// (GetDragDropPayload() gates this) -- deliberately no
				// ImGuiWindowFlags_NoInputs, since that would also disable
				// drag-drop hit-testing; safe without it because this
				// window only ever exists for the duration of an already-
				// active ImGui drag, which already owns mouse input for
				// that duration regardless of this window
				if (ImGui::GetDragDropPayload() != nullptr) {
					ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
					ImGui::SetNextWindowSize(ImVec2((f32)windowW, (f32)windowH));
					ImGui::SetNextWindowBgAlpha(0.0f);
					ImGui::Begin(
						"##viewport_drop_target", nullptr,
						ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
						| ImGuiWindowFlags_NoSavedSettings
						| ImGuiWindowFlags_NoFocusOnAppearing
						| ImGuiWindowFlags_NoBringToFrontOnFocus
						| ImGuiWindowFlags_NoNav
					);
					// BeginDragDropTarget() attaches to the LAST SUBMITTED
					// ITEM, not automatically to the enclosing window -- with
					// no item at all, it has nothing to hook onto (silently
					// finds no target) regardless of the window covering the
					// whole viewport. Dummy() submits one filling the window
					ImGui::Dummy(ImGui::GetContentRegionAvail());
					if (ImGui::BeginDragDropTarget()) {
						// new instance at the camera's current orbit pivot --
						// a much more useful default than the world origin,
						// which is often off-screen depending on the scene;
						// drag it into its final position afterward with the
						// usual gizmo/transform window, same as any other
						// newly-added instance
						stage::Transform const dropTransform {
							.position = cam.target,
						};
						if (
							stage::stage_imgui_accept_gltf_model_drop(
								sStage, dropTransform
							)
						) {
							gltfInstanceCountChanged = true;
						}
						ImGui::EndDragDropTarget();
					}
					ImGui::End();
				}

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
					matChanged = ponder::material_inspector_draw(
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
				for (ponder::ModelEntry const & entry : sEnvMapList) {
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
				sModelList = ponder::scan_asset_paths(
					settingsPath, { "modelPaths" }, { ".gltf", ".glb" },
					/*warnMissingKey=*/true
				);
				sEnvMapList = ponder::scan_asset_paths(
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
				ponder::camera_controller_sync_fp_from_orbit(camController);
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
				ponder::camera_controller_sync_fp_from_orbit(camController);
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
				if (sVdbTemperature.gpuBuffer.id != 0u) {
					vkof::buffer_destroy(sVdbTemperature.gpuBuffer);
				}
				if (sVdbTemperature.handleBuffer.id != 0u) {
					vkof::buffer_destroy(sVdbTemperature.handleBuffer);
				}
				sVdbTemperature = {};
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
					// independent of whether "density" loaded above -- a
					// fire-only vdb can have "temperature" with no density
					// grid at all
					fnLoadVdbTemperatureIfPresent(pendingVdbPath.c_str());
					sVdbList = ponder::scan_asset_paths(
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
					ponder::camera_controller_sync_fp_from_orbit(camController);
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
				ponder::camera_controller_sync_fp_from_orbit(camController);
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
					ponder::camera_controller_sync_fp_from_orbit(camController);
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
				ponder::camera_controller_sync_fp_from_orbit(camController);
				settingsChanged = true;
			}

			// -- add instance: append, then rebuild both the blas list
			// (structural change) and the tlas (so the new instance actually
			// renders this frame)
			if (!pendingAddGltfPath.empty()) {
				if (
					stage::stage_add_gltf_model(sStage, pendingAddGltfPath.c_str())
					>= 0
				) {
					settingsChanged = true;
				} else {
					printf(
						"failed to add gltf model '%s'\n",
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

			// every path that can change sStage.gltfInstances's size (outliner
			// delete, models-panel drag-drop, file-picker add above) funnels
			// through this one flag rather than rebuilding immediately at the
			// point of mutation -- see gltfInstanceCountChanged's declaration
			// comment for why the rebuild itself must happen out here, after
			// the "viewer" window's ImGui::End() has already run
			if (gltfInstanceCountChanged) {
				stage_render_rebuild_blases(sStageRender, sStage);
				stage_render_rebuild_tlas(sStageRender, sStage);
				stageInstancesChanged = false;
				settingsChanged = true;
			}
			if (sMode == ViewerMode::StageEdit) {
				stage::stage_flush_pending_gpu_destroys(sStage);
			}

			// stage_imgui_edit's own edits (transform drags, visibility
			// toggles, material overrides, light/camera fields) don't touch
			// the instance list itself, only need the tlas + models-
			// indirect buffer refreshed to match -- unless a full rebuild
			// above already covered it this frame
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
				ponder::camera_controller_sync_fp_from_orbit(camController);
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
	if (sVdbTemperature.gpuBuffer.id != 0u) {
		vkof::buffer_destroy(sVdbTemperature.gpuBuffer);
	}
	if (sVdbTemperature.handleBuffer.id != 0u) {
		vkof::buffer_destroy(sVdbTemperature.handleBuffer);
	}
	stage_render_destroy(sStageRender);
	stage::stage_destroy(sStage);
	vkof::buffer_destroy(nanCountsBuffer);
	vkof::buffer_destroy(nanProbeCounterBuffer);
	vkof::buffer_destroy(probeResultBuffer);
	vkof::buffer_destroy(sppCounterBuffer);
	vkof::image_destroy(ptAccumImage);
	vkof::image_destroy(gbufferPosImage);
	vkof::image_destroy(gbufferNorImage);
	vkof::image_destroy(gbufferAlbedoImage);
	vkof::image_destroy(denoisePingImage);
	vkof::image_destroy(denoisePongImage);
	vkof::image_destroy(bloomAImage);
	vkof::image_destroy(bloomBImage);
	vkof::sampler_destroy(bloomBlurSampler);
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
