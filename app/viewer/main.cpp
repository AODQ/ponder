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
#include <ponder/debug-path.hpp>
#include <ponder/dlss-rr.hpp>
#include <ponder/regir-debug.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/material-inspector.hpp>
#include <ponder/material-media.hpp>
#include <ponder/timeline.hpp>
#include <ponder/vdb.hpp>
#include <ponder/wavefront.hpp>
#include <ponder/zeltner-tables.hpp>

#include <stage/stage.hpp>

#include "scene-state.hpp"

#include <nanovdb/NanoVDB.h>

#include "shaders/resolve_pc.h"
#include "shaders/pt-accumulate-pc.h"
#include "shaders/nan-check-pc.h"
#include "shaders/manifold-probe-pc.h"
#include "shaders/manifold-shade-pc.h"
#include "shaders/pt-denoise-pc.h"
#include "shaders/bloom-extract-pc.h"
#include "shaders/bloom-blur-pc.h"
#include "shaders/bloom-composite-pc.h"
#include "shaders/display-dither-pc.h"

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

// mirrors util-environment-map.glsl's EnvironmentMapHandles layout
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 aliasProb;
	u32 aliasIndex;
};

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

struct ViewerArgs {
	char const * gltfPath = nullptr;
	char const * screenshotPath = nullptr;
	char const * vdbPath = nullptr;
	u32 width = 1280u;
	u32 height = 720u;
	u32 spp = 256u;
	// paths per pass, 0 for the ui's default
	u32 pathBudget = 0u;
	bool regir = false;
	bool cameraOverride = false;
	f32 cameraAzimuth = 0.0f;
	f32 cameraElevation = 0.3f;
	f32 cameraDistance = 0.0f;
	f32 cameraFovY = 1.0f;
	f32 envIntensity = 1.0f;
	f32 backgroundIntensity = 1.0f;
	i32 envMode = 0;
	bool envModeExplicit = false;
	char const * envExrPath = nullptr;
	bool envMis = true;
	bool meshMis = true;
	bool manifoldNee = false;
	f32 fireflyClampLuminance = 0.0f;
	f32 budgetMs = 0.0f;
	f32 generatedNormalStrength = 0.0f;
	bool debug = false;
	char const * stagePath = nullptr;
	bool newStage = false;
	bool dlssRr = false;
	ponder::DlssRrQuality dlssRrQuality = ponder::DlssRrQuality::Dlaa;
};

void print_usage() {
	printf(
		"usage: viewer [path.gltf|.glb]\n"
		"  [--screenshot <out.png>]  render headless, write png, exit\n"
		"  [--camera <azimuth,elevation,distance,fovY>]  orbit camera\n"
		"      (radians / world units; distance 0 = auto-fit)\n"
		"  [--resolution <w>x<h>]  render resolution (default 1280x720)\n"
		"  [--spp <n>]  samples per pixel before --screenshot (default 256)\n"
		"  [--render-resolution <w>x<h>]  paths traced per pass; passes walk\n"
		"      the image when it has more pixels (default 854x480)\n"
		"  [--vdb <path.vdb>]  load a density grid\n"
		"  [--env-intensity <f>]  environment radiance (default 1.0)\n"
		"  [--background-intensity <f>]  scales the environment camera rays\n"
		"      see directly; lighting is unchanged (default 1.0)\n"
		"  [--env-mode <furnace|checkerboard|black|hdrmap>]  (default furnace)\n"
		"  [--env-exr <path.exr>]  load an exr environment map; implies\n"
		"      --env-mode hdrmap unless --env-mode is also given\n"
		"  [--env-mis <0|1>]  env map next-event estimation (default 1)\n"
		"  [--mesh-mis <0|1>] emissive-mesh next-event estimation (default 1)\n"
		"  [--regir]  enable regir many-light sampling\n"
		"  [--manifold-nee <0|1>] nee through refracting casters (default 0)\n"
		"  [--firefly-clamp <f>]  cap a sample's luminance after exposure\n"
		"      before it's written out; <= 0 disables (default off)\n"
		"  [--budget-ms <f>]  start with the resolve frame budget enabled\n"
		"      at this gpu-ms target; <= 0 disables (default off)\n"
		"  [--gen-normals <f>]  for materials with no normal map, synthesize\n"
		"      one from base color luminance at this strength; fabricated\n"
		"      relief, not real height data (default 0, off)\n"
		"  [--debug]  enable vulkan validation layers (slower startup;\n"
		"      needed for NaN probes / validation messages -- always on\n"
		"      with --screenshot)\n"
		"  [--stage <path.json>]  start in stage edit mode with this file\n"
		"      loaded (skips the gltf/vdb-required startup check)\n"
		"  [--new-stage]  start in stage edit mode with a fresh, empty,\n"
		"      unsaved stage\n"
		"  [--dlss-rr]  with --screenshot: --spp frames through DLSS ray\n"
		"      reconstruction instead of accumulation\n"
		"  [--dlss-rr-upscale <dlaa|quality|balanced|performance|ultra>]\n"
		"      trace below the resolution and let RR upscale (default dlaa)\n"
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
		} else if (
			strcmp(argv[i], "--render-resolution") == 0 && i + 1 < argc
		) {
			u32 w, h;
			if (sscanf(argv[++i], "%ux%u", &w, &h) != 2 || w == 0u || h == 0u) {
				printf("--render-resolution expects <w>x<h>\n");
				return false;
			}
			out.pathBudget = w * h;
		} else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
			out.spp = (u32)atoi(argv[++i]);
		} else if (strcmp(argv[i], "--env-intensity") == 0 && i + 1 < argc) {
			out.envIntensity = (f32)atof(argv[++i]);
		} else if (
			strcmp(argv[i], "--background-intensity") == 0 && i + 1 < argc
		) {
			out.backgroundIntensity = (f32)atof(argv[++i]);
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
		}
		else if (strcmp(argv[i], "--mesh-mis") == 0 && i + 1 < argc) {
			out.meshMis = atoi(argv[++i]) != 0;
		} else if (strcmp(argv[i], "--manifold-nee") == 0 && i + 1 < argc) {
			out.manifoldNee = atoi(argv[++i]) != 0;
		} else if (strcmp(argv[i], "--firefly-clamp") == 0 && i + 1 < argc) {
			out.fireflyClampLuminance = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--budget-ms") == 0 && i + 1 < argc) {
			out.budgetMs = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--gen-normals") == 0 && i + 1 < argc) {
			out.generatedNormalStrength = (f32)atof(argv[++i]);
		} else if (strcmp(argv[i], "--debug") == 0) {
			out.debug = true;
		} else if (strcmp(argv[i], "--stage") == 0 && i + 1 < argc) {
			out.stagePath = argv[++i];
		} else if (strcmp(argv[i], "--regir") == 0) {
			out.regir = true;
		} else if (strcmp(argv[i], "--dlss-rr") == 0) {
			out.dlssRr = true;
		} else if (strcmp(argv[i], "--dlss-rr-upscale") == 0 && i + 1 < argc) {
			char const * const mode = argv[++i];
			using ponder::DlssRrQuality;
			if (strcmp(mode, "dlaa") == 0) {
				out.dlssRrQuality = DlssRrQuality::Dlaa;
			} else if (strcmp(mode, "quality") == 0) {
				out.dlssRrQuality = DlssRrQuality::Quality;
			} else if (strcmp(mode, "balanced") == 0) {
				out.dlssRrQuality = DlssRrQuality::Balanced;
			} else if (strcmp(mode, "performance") == 0) {
				out.dlssRrQuality = DlssRrQuality::Performance;
			} else if (strcmp(mode, "ultra") == 0) {
				out.dlssRrQuality = DlssRrQuality::UltraPerformance;
			} else {
				printf("unknown --dlss-rr-upscale '%s'\n", mode);
				return false;
			}
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

ponder::Vdb sVdb {};
ponder::Vdb sVdbTemperature {};

void fnLoadVdbTemperatureIfPresent(char const * const path) {
	sVdbTemperature = {};
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
// -- stage mode
// -----------------------------------------------------------------------------

enum class ViewerMode { FileView, StageEdit };

enum class FrameMode : i32 { View = 0, Clear = 1, PathTrace = 2 };

constexpr f32 skCameraIdleSeconds = 0.2f;

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

	// -- environment map
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
		// care for in-flight frames
		vkof::device_wait_idle();
		if (sEnvTables.width != 0u) {
			ponder::environment_tables_destroy(sEnvTables);
			sEnvTables = {};
		}
		sCurrentEnvMapPath.clear();
		sEnvTables = ponder::environment_tables_create(path);
		if (sEnvTables.width == 0u) {
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
		args.envMode = 0;
	}

	// -- pipelines
	char const * const includePaths[] = {
		PONDER_SHADER_DIR,
		MOR_INCLUDE_DIR,
	};
	ponder::Wavefront wavefront = ponder::wavefront_create();
	ponder::Regir regir = ponder::regir_create();
	vkof::Pipeline const ptAccumulatePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/pt-accumulate.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const nanCheckPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/nan-check.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const manifoldProbePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/manifold-probe.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const manifoldShadePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/manifold-shade.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const denoisePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/pt-denoise.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomExtractPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-extract.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomBlurPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-blur.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const bloomCompositePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/bloom-composite.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const displayDitherPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appDir / "shaders/display-dither.slang").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);

	// -- broken-pixel counters
	vkof::Buffer const nanCountsBuffer = vkof::buffer_create({
		.byteCount = 2u * sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	// -- manifold probe counts
	vkof::Buffer const manifoldCountsBuffer = vkof::buffer_create({
		.byteCount = 4u * sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnZeroManifoldCounts = [&]() {
		u32 const zero[4] = { 0u, 0u, 0u, 0u };
		vkof::buffer_upload({
			.buffer = manifoldCountsBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(zero), sizeof(zero)
			),
		});
	};
	fnZeroManifoldCounts();
	vkof::Buffer const manifoldCasterBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuManifoldMeshData),
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

	// -- nan probe printf
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

	// -- material inspector probe result
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

	// -- spp counter
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

	// -- settings.json model + environment-map libraries
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
	ponder::AssetTagIndex const sAssetTagIndex = ponder::asset_tag_index_load(
		settingsPath.parent_path() / "assets" / "index.json"
	);
	auto sModelTagFilter = ponder::AssetTagFilter {
		.index = &sAssetTagIndex, .selected = {}, .matchAny = false,
	};

	// -- vdb
	std::string sCurrentVdbPath;
	if (args.vdbPath != nullptr) {
		sVdb = ponder::vdb_load(args.vdbPath);
		if (sVdb.gpuBuffer.id != 0u) {
			sCurrentVdbPath = args.vdbPath;
		}
		fnLoadVdbTemperatureIfPresent(args.vdbPath);
	}

	// -- scene
	std::string sCurrentModelPath;
	mor::scene_set_generated_normal_strength(args.generatedNormalStrength);
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
	if (sCurrentModelPath.empty() && !sVdb.blob.empty()) {
		vdb_bounds(sVdb, sScene.boundsCenter, sScene.boundsExtent);
	}

	// -- viewer mode
	ViewerMode sMode = ViewerMode::FileView;
	stage::Stage sStage {};
	StageRenderState sStageRender {};
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
	// tonemapped display image; float so bloom's read-modify-write and the
	// final dither see unquantized values
	vkof::TransientImage const ptOutputTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1u,
		.isDoubleBuffered = false,
	});
	// presented image: ptOutputTarget dithered in srgb-encoded space, stored
	// linear; the blit to the srgb swapchain encodes and quantizes it
	vkof::TransientImage const displayTarget = vkof::transient_image_create({
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.scaleWidth = 1.0f,
		.scaleHeight = 1.0f,
		.mipLevels = 1u,
		.isDoubleBuffered = false,
	});
	u32 sRenderW = args.width;
	u32 sRenderH = args.height;
	// paths traced per pass; passes walk the image when it has more pixels
	i32 sRenderBudgetIdx = 0;
	// --render-resolution, until the ui picks a preset
	u32 sPathBudgetOverride = args.pathBudget;
	auto const fnPathBudgetWanted = [&]() -> u32 {
		if (sPathBudgetOverride > 0u) { return sPathBudgetOverride; }
		return (
			skRenderResPresets[sRenderBudgetIdx].w
			* skRenderResPresets[sRenderBudgetIdx].h
		);
	};
	// the budget the path buffers hold
	u32 sPathBudget = fnPathBudgetWanted();
	// the size the wavefront traces; below the render size when dlss-rr upscales
	auto sTraceSize = u32v2 { sRenderW, sRenderH };

	vkof::Image ptAccumImage {};
	u32 ptAccumStorageHandle = 0u;
	vkof::Sampler const magnifierSampler = (
		vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::nearest,
			.minFilter = vkof::SamplerFilter::nearest,
			.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
		})
	);
	ImTextureID ptAccumImguiId {};
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
		if (ptAccumImguiId) {
			vkof::image_imgui_id_destroy(ptAccumImguiId);
			ptAccumImguiId = {};
		}
		sTraceSize = u32v2 { sRenderW, sRenderH };
		ponder::wavefront_resize(wavefront, sRenderW, sRenderH, sPathBudget);
	};
	fnCreateAccumImage();

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

	vkof::Buffer manifoldBendPointBuffer {};
	auto const fnCreateManifoldBendPoints = [&]() {
		manifoldBendPointBuffer = vkof::buffer_create({
			.byteCount = (
				static_cast<u64>(sRenderW) * sRenderH
				* sizeof(GpuManifoldBendPoint)
			),
			.memory = vkof::BufferMemory::DeviceOnly,
		});
	};
	fnCreateManifoldBendPoints();

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

	// -- dlss ray reconstruction, off at startup; created on first use
	ponder::DlssRr sDlssRr {};
	bool sDlssRrEnabled = false;
	auto sDlssRrPreset = ponder::DlssRrPreset::Default;
	auto sDlssRrQuality = args.dlssRrQuality;
	if (args.dlssRr) {
		if (ponder::dlss_rr_supported()) {
			sDlssRrEnabled = true;
		} else {
			printf(
				"--dlss-rr ignored: %s\n", ponder::dlss_rr_unsupported_reason()
			);
		}
	}

	// -- bloom
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
	bool sBloomEnabled = false;
	f32 sBloomThreshold = 1.0f;
	f32 sBloomIntensity = 0.12f;
	constexpr u32 kBloomBlurPassCount = 5u;

	// -- environment settings
	f32 sEnvIntensity = args.envIntensity;
	f32 sEnvBackgroundIntensity = args.backgroundIntensity;
	i32 sEnvMode = args.envMode;
	bool sEnvNeeEnabled = args.envMis;
	bool sMeshNeeEnabled = args.meshMis;
	bool sManifoldNeeEnabled = args.manifoldNee;
	f32v3 sManifoldPointWo { 0.0f, 3.0f, 0.0f };
	f32v3 sManifoldDebugColor { 0.0f, 0.4f, 0.8f };
	bool sRegirEnabled = args.regir;
	bool sRegirJitter = false;
	u32 sRegirCellCount = 65536u;
	f32 sRegirCellSize = 1.0f;
	i32 sRegirMaxAge = 16;
	// grid stats and probe inspector, the "regir" window
	ponder::RegirDebugState sRegirDebug {};
	auto const environment_light_power = [&]() -> f32 {
		if (!sEnvNeeEnabled || sEnvMode != 3) {
			return 0.0f;
		}
		return sEnvTables.luminanceIntegral * sEnvIntensity;
	};
	bool sEnvMapOnlyMode = false;
	f32 sEnvRotation = 0.0f;
	auto const fnApplyStageEnvironment = [&]() {
		sEnvMode = (i32)sStage.environment.mode;
		sEnvRotation = sStage.environment.rotationRadians;
		sEnvIntensity = sStage.environment.intensity;
		if (sEnvMode == 3 && !sStage.environment.path.empty()) {
			fnLoadEnvMap(sStage.environment.path);
		}
	};
	bool sFireflyClampEnabled = args.fireflyClampLuminance > 0.0f;
	f32 sFireflyClampLuminance = (
		args.fireflyClampLuminance > 0.0f ? args.fireflyClampLuminance : 20.0f
	);
	f32 sExposure = 1.0f;
	bool sAntialias = true;
	bool sSpectral = true;
	bool sBlackholeEnabled = false;
	f32v3 sBlackholeOri = { 0.0f, 0.0f, 0.0f };
	// -- ambient medium
	i32 sAmbientMediumType = (i32)ponder::WavefrontAmbientMediumType::None;
	f32v3 sAmbientMediumScattering = { 1.0f, 1.0f, 1.0f };
	f32v3 sAmbientMediumAbsorption = { 0.0f, 0.0f, 0.0f };
	f32 sAmbientMediumDensity = 0.01f;
	i32 sAmbientMediumPhase = (i32)ponder::WavefrontAmbientMediumPhase::HenyeyGreenstein;
	f32 sAmbientMediumAnisotropy = 0.0f;
	f32 sAmbientMediumDraineAlpha = 1.0f;
	f32 sAmbientMediumDropletDiameter = 10.0f;
	f32 sAmbientMediumHeightOrigin = 0.0f;
	f32 sAmbientMediumHeightFalloff = 5.0f;

	// -- ground heating index field
	bool sGroundHeatingEnabled = false;
	f32 sGroundHeatingGroundHeight = 0.0f;
	f32 sGroundHeatingHazeHeight = 1.0f;
	f32 sGroundHeatingGradient = 0.0f;
	f32 sGroundHeatingNoiseAmplitude = 0.0f;
	f32 sGroundHeatingNoiseScale = 1.0f;

	f32 sBlackholeMass = 0.0f;
	f32 sBlackholeSpin = 0.9f;
	f32 sBlackholeLensRadius = 1000.0f;
	i32 sBlackholeMaxChords = 1024;
	bool sBlackholeDiskEnabled = true;
	f32 sBlackholeDiskInnerRadius = 6.0f;
	f32 sBlackholeDiskOuterRadius = 20.0f;
	f32 sBlackholeDiskTemperature = 6000.0f;
	f32 sBlackholeDiskBrightness = 1.0f;
	f32 sBlackholeDiskOpticalDepth = 5.0f;
	f32 sBlackholeDiskOpticalDepthExponent = 1.5f;
	f32 sBlackholeStepFactor = 0.05f;
	// -- blackhole bullshit
	double sBlackholeDiskSeconds = 0.0;
	f32 sBlackholeDiskOrbitsPerSecond = 0.25f;
	bool sBlackholeDiskAnimate = false;
	bool sBlackholeDiskPrograde = true;
	f32 sBlackholeDiskNoiseAmplitude = 0.3f;
	f32 sBlackholeDiskNoiseScale = 4.0f;
	f32 sBlackholeDiskHotspotRadius = 8.0f;
	f32 sBlackholeDiskHotspotTemperature = 4000.0f;
	f32 sBlackholeDiskHotspotSize = 1.0f;
	// -- camera shutter
	f32 sShutterSeconds = 0.0f;
	// -- camera lens
	f32 sLensRadius = 0.0f;
	f32 sFocalDistance = 5.0f;
	// lens edits outside the settings ui; resets the next frame
	bool sLensChanged = false;
	auto const fnBlackholeMetersPerSecond = [&]() -> double {
		double const innerPeriod = (
			2.0 * 3.14159265358979
			* std::pow((double)sBlackholeDiskInnerRadius, 1.5)
			* (double)sBlackholeMass
		);
		return (double)sBlackholeDiskOrbitsPerSecond * innerPeriod;
	};
	auto const fnBlackholeDiskTime = [&]() -> f32 {
		double const innerPeriod = (
			2.0 * 3.14159265358979
			* std::pow((double)sBlackholeDiskInnerRadius, 1.5)
			* (double)sBlackholeMass
		);
		double const orbits = (
			std::fmod(
				sBlackholeDiskSeconds * (double)sBlackholeDiskOrbitsPerSecond,
				1000.0
			)
		);
		return (f32)(orbits * innerPeriod);
	};

	// -- vdb volume
	f32v3 sVdbSigmaAbsorption = { 0.0f, 0.0f, 0.0f };
	f32v3 sVdbSigmaScattering = { 1.0f, 1.0f, 1.0f };
	f32 sVdbDropletDiameter = 25.0f;
	f32 sVdbPhaseAnisotropyOverride = -2.0f;
	f32 sVdbTemperatureScale = 1.0f;
	f32 sVdbEmissionScale = 1.0f;

	// -- homogeneous fog params
	bool sFogEnabled = false;
	f32 sFogSigma = 0.1f;
	f32v3 sFogAlbedo = { 1.0f, 1.0f, 1.0f };
	f32 sFogDistanceMax = 1000.0f;
	i32 sMaxSpp = 0;
	bool sScreenshotPending = false;
	std::string sLastScreenshotPath;

	// -- material inspector
	i32v2 sProbePixel { -1, -1 };
	i32 sPickedMaterialIndex = -1;
	i32 sPickedModelDrawIndex = -1;
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

	// -- --stage / --new-stage
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
		stage_render_rebuild_tlas(
			sStageRender, sStage, sMeshNeeEnabled, environment_light_power()
		);
		fnApplyStageEnvironment();
		fnReframeCamera(cam);
		ponder::camera_controller_sync_fp_from_orbit(camController);
	}

	FrameMode sFrameMode = FrameMode::PathTrace;
	f32 sViewIdleSeconds = 0.0f;

	u32 sFrameIndex = 0u;

	struct RegirBinding {
		u64 guards;
		u64 cells;
		u64 reservoirs;
		u32 cellCount;
	};
	auto const fnRegirBinding = [&]() -> RegirBinding {
		bool const stageMode = sMode == ViewerMode::StageEdit;
		vkof::Buffer const guards = (
			stageMode
				? sStageRender.regirGuardBuffer : sScene.regirGuardBuffer
		);
		vkof::Buffer const cells = (
			stageMode ? sStageRender.regirCellBuffer : sScene.regirCellBuffer
		);
		vkof::Buffer const reservoirs = (
			stageMode
				? sStageRender.regirReservoirBuffer
				: sScene.regirReservoirBuffer
		);
		if (guards.id == 0u) {
			return RegirBinding { 0u, 0u, 0u, 0u };
		}
		return RegirBinding {
			vkof::buffer_virtual_address(guards),
			vkof::buffer_virtual_address(cells),
			vkof::buffer_virtual_address(reservoirs),
			stageMode ? sStageRender.regirCellCount : sScene.regirCellCount,
		};
	};

	auto const fnUploadFrameState = [&]() {
		RegirBinding const regir = fnRegirBinding();
		fnZeroNanProbeCounter();
		if (sMode == ViewerMode::FileView) {
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
				.emitterLightOffset = 0u,
				.emitterLightCount = 0u,
			};
			{
				std::vector<GpuResolveModelIndirect> models { modelDesc };
				std::vector<mor::Scene> const scenes { sScene.scene };
				std::vector<mor::GpuMaterials> const overrides {
					sScene.materialsOverride
				};
				std::vector<std::vector<mor::MeshletAreaInfo> const *> const
					areas { &sScene.meshletAreas };
				EmitterListRebuild const rebuilt = (
					emitter_lights_rebuild(
						models, scenes, overrides, areas, sMeshNeeEnabled,
						environment_light_power(),
						sScene.emitterLightsBuffer, sScene.emitterLightCapacity
					)
				);
				sScene.emitterLightCount = rebuilt.count;
				sScene.envLightIndex = rebuilt.envLightIndex;
				if (rebuilt.clipped > 0u && rebuilt.clipped != sScene.emitterClipped) {
					printf(
						"[nee] emitter list clipped: %u emissive meshlets dropped,"
						" %u kept. the dropped ones still render, by bsdf"
						" sampling alone\n",
						rebuilt.clipped, rebuilt.count
					);
				}
				sScene.emitterClipped = rebuilt.clipped;
				sScene.emitterBuildMs = rebuilt.buildMs;
				vkof::buffer_upload({
					.buffer = sScene.modelsIndirectBuffer,
					.byteOffset = 0u,
					.data = srat::slice_as_bytes(models[0]),
				});
			}
		}

		stage::VdbBlob const * const activeVdbBlob = (
			sMode == ViewerMode::StageEdit
				&& !sStage.vdbInstances.empty()
				&& !sStage.vdbInstances[0].blobs.empty()
					? &sStage.vdbInstances[0].blobs[0]
					: nullptr
		);
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
			.emitterLights = (
				sMode == ViewerMode::StageEdit
					? (
						sStageRender.emitterLightsBuffer.id != 0u
							? vkof::buffer_virtual_address(
								sStageRender.emitterLightsBuffer
							)
							: 0u
					)
					: (
						sScene.emitterLightsBuffer.id != 0u
							? vkof::buffer_virtual_address(
								sScene.emitterLightsBuffer
							)
							: 0u
					)
			),
			.emitterLightCount = (
				sMode == ViewerMode::StageEdit
					? sStageRender.emitterLightCount : sScene.emitterLightCount
			),
			.meshNeeEnabled = sMeshNeeEnabled ? 1u : 0u,
			.manifoldNeeEnabled = sManifoldNeeEnabled ? 1u : 0u,
			.envLightIndex = (
				sMode == ViewerMode::StageEdit
					? sStageRender.envLightIndex : sScene.envLightIndex
			),
			.emitterNeeDebug = 0u,
			.regirGuards = regir.guards,
			.regirCells = regir.cells,
			.regirReservoirs = regir.reservoirs,
			.regirCellCount = regir.cellCount,
			.regirFrameIndex = sFrameIndex,
			.regirCellSize = sRegirCellSize,
			// the old pipeline's grid is no longer allocated; regir is new-core only
			.regirEnabled = 0u,
			.envMapOnlyDepth = (
				sEnvMapOnlyMode ? 1u : (sFrameMode == FrameMode::View ? 2u : 0u)
			),
			.fireflyClampLuminance = (
				sFireflyClampEnabled ? sFireflyClampLuminance : 0.0f
			),
			.envBackgroundIntensity = sEnvBackgroundIntensity,
			.vdb = activeVdbHandleVa,
			.vdbWorldToLocal = activeVdbWorldToLocal,
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
		sProbePixel = { -1, -1 };
		vkof::buffer_upload({
			.buffer = (
				sMode == ViewerMode::StageEdit
					? sStageRender.debugPcBuffer : sScene.debugPcBuffer
			),
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(extended),
		});
	};

	u32 sAccumFrames = 0u;
	ponder::Timeline sTimeline {};
	ponder::DebugPathState sDebugPath {};
	ponder::MaterialMedia sMaterialMedia {};
	FrameMode sPreviousFrameMode = FrameMode::View;

	// -- frame budget
	bool sBudgetEnabled = true;
	f32 sBudgetTargetMs = args.budgetMs > 0.0f ? args.budgetMs : 8.0f;
	double sResolveMsLast = 0.0;
	struct ResolveFrameHistory {
		bool resolveRan;
		// wavefront steps [stepFirst, stepFirst + stepCount) the resolve node
		// dispatched, on pathCount paths
		u32 stepFirst;
		u32 stepCount;
		u32 pathCount;
		// resolve's position in that frame's node list; the regir fill can
		// precede it
		u32 resolveNodeIndex;
	};
	ResolveFrameHistory sResolveHistory[2] = {};
	// gpu ms per million paths of each pass step, 0 until measured; steps
	// differ a lot (bounce 0 trace + shade vs camera, output, late bounces)
	double sStepCost[ponder::skWavefrontPassStepCount] = {};

	// -- wavefront pass accumulation
	// counts accumulation resets; a pass accumulates only if it started
	// after the latest reset
	u32 sResetGeneration = 0u;
	u32 sPassGeneration = 0u;
	// generation of the accumulated passes, ~0 when none
	u32 sAccumGeneration = ~0u;

	// -- focus region
	bool sFocusActive = false;
	f32v2 sFocusMin { 0.0f, 0.0f };
	f32v2 sFocusMax { 1.0f, 1.0f };
	// the focus rectangle in pixels, grown to whole wavefront tiles
	auto const fnFocusRegion = [&]() -> ponder::WavefrontRegion {
		u32 const tile = ponder::skWavefrontTileSize;
		auto const fnAxis = [&](f32 const lo, f32 const hi, u32 const size) {
			u32 const first = (
				std::min((u32)(lo * (f32)size), size - 1u) / tile * tile
			);
			u32 const last = (
				std::min(((u32)(hi * (f32)size) + tile - 1u) / tile * tile, size)
			);
			return u32v2 { first, std::max(last, first + 1u) - first };
		};
		u32v2 const x = fnAxis(sFocusMin.x, sFocusMax.x, sRenderW);
		u32v2 const y = fnAxis(sFocusMin.y, sFocusMax.y, sRenderH);
		return ponder::WavefrontRegion {
			.origin = { x.x, y.x },
			.extent = { x.y, y.y },
		};
	};

	auto const fnRenderFrame = [&](
		FrameMode const frameMode,
		bool const pathTraceReset,
		bool const freezeAccum,
		bool const probeRequested,
		bool const rrReset
	) {
		bool const timelineSingleSample = (
			!sTimeline.playing || (sBlackholeEnabled && sBlackholeDiskAnimate)
		);
		bool const accumReset = (
			frameMode == FrameMode::View
			|| (
				frameMode == FrameMode::PathTrace
				&& (
					pathTraceReset
					|| sPreviousFrameMode == FrameMode::Clear
					|| timelineSingleSample
				)
			)
		);
		bool const regirView = sRegirEnabled && sRegirDebug.viewMode != 0;
		// dlss ray reconstruction: one whole pass per frame, never accumulated
		bool const rrActive = sDlssRrEnabled && !regirView;
		bool const runResolve = (
			rrActive
			|| frameMode == FrameMode::View
			|| (
				frameMode == FrameMode::PathTrace && (!freezeAccum || probeRequested)
			)
		);

		// -- path budget: dlss-rr takes every trace pixel each frame
		u32v2 const traceSize = (
			rrActive
				? ponder::dlss_rr_render_size(
					sDlssRrQuality, u32v2 { sRenderW, sRenderH }
				)
				: u32v2 { sRenderW, sRenderH }
		);
		// whole tiles: rr needs the full image every frame, and edge tiles'
		// padding slots count against the budget
		u32 const traceTilesW = (
			(traceSize.x + ponder::skWavefrontTileSize - 1u)
			/ ponder::skWavefrontTileSize
		);
		u32 const traceTilesH = (
			(traceSize.y + ponder::skWavefrontTileSize - 1u)
			/ ponder::skWavefrontTileSize
		);
		u32 const pathBudget = (
			rrActive
				? traceTilesW * traceTilesH * ponder::skWavefrontTileSlotCount
				: fnPathBudgetWanted()
		);
		if (pathBudget != sPathBudget || traceSize != sTraceSize) {
			sPathBudget = pathBudget;
			sTraceSize = traceSize;
			ponder::wavefront_resize(
				wavefront, traceSize.x, traceSize.y, sPathBudget
			);
		}

		// -- wavefront pass
		// a debug capture records a whole pass, from its camera step
		if (sDebugPath.capturePending || rrActive) {
			ponder::wavefront_pass_restart(wavefront);
		}
		if (accumReset) {
			++sResetGeneration;
		}
		bool const passStarts = (
			runResolve && !regirView && wavefront.passStep == 0u
		);
		if (passStarts) {
			sPassGeneration = sResetGeneration;
		}

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
			.pad1 = wavefront.globalVa,
			.pad2 = 0u,
			.pad3 = 0u,
		};
		ponder::WavefrontCamera const viewCamera = (
			sFirstPerson
				? ponder::WavefrontCamera {
					.position = camFp.position,
					.forward = srat::camera_fp_forward(camFp),
					.fovY = camFp.fovY,
				}
				: ponder::WavefrontCamera {
					.position = srat::camera_orbit_eye(cam),
					.forward = cam.target - srat::camera_orbit_eye(cam),
					.fovY = cam.fovY,
				}
		);
		ponder::wavefront_global_update(
			wavefront,
			regir,
			ponder::WavefrontGlobalInfo {
				.camera = viewCamera,
				.blackhole = ponder::WavefrontBlackhole {
					.ori = sBlackholeOri,
					.mass = sBlackholeEnabled ? sBlackholeMass : 0.0f,
					.spin = sBlackholeSpin,
					.lensRadius = sBlackholeLensRadius,
					.maxChords = (u32)sBlackholeMaxChords,
					.diskInnerRadius = sBlackholeDiskInnerRadius,
					.diskOuterRadius = (
						sBlackholeDiskEnabled ? sBlackholeDiskOuterRadius : 0.0f
					),
					.diskTemperature = sBlackholeDiskTemperature,
					.diskBrightness = sBlackholeDiskBrightness,
					.diskOpticalDepth = sBlackholeDiskOpticalDepth,
					.diskOpticalDepthExponent = sBlackholeDiskOpticalDepthExponent,
					.stepFactor = sBlackholeStepFactor,
					.diskTime = fnBlackholeDiskTime(),
					.diskRotation = sBlackholeDiskPrograde ? 1.0f : -1.0f,
					.diskNoiseAmplitude = sBlackholeDiskNoiseAmplitude,
					.diskNoiseScale = sBlackholeDiskNoiseScale,
					.diskHotspotRadius = sBlackholeDiskHotspotRadius,
					.diskHotspotTemperature = sBlackholeDiskHotspotTemperature,
					.diskHotspotSize = sBlackholeDiskHotspotSize,
				},
				.ambientMedium = ponder::WavefrontAmbientMedium {
					.type = (ponder::WavefrontAmbientMediumType)sAmbientMediumType,
					.sigmaAbsorption = sAmbientMediumAbsorption * sAmbientMediumDensity,
					.sigmaScattering = sAmbientMediumScattering * sAmbientMediumDensity,
					.phase = (ponder::WavefrontAmbientMediumPhase)sAmbientMediumPhase,
					.anisotropy = sAmbientMediumAnisotropy,
					.draineAlpha = sAmbientMediumDraineAlpha,
					.dropletDiameter = sAmbientMediumDropletDiameter,
					.heightOrigin = sAmbientMediumHeightOrigin,
					.heightFalloff = sAmbientMediumHeightFalloff,
				},
				.materialMediumTableVa = (
					ponder::material_media_table_va(sMaterialMedia)
				),
				.materialMediumRecordsVa = (
					ponder::material_media_records_va(sMaterialMedia)
				),
				.materialMediumModelCount = sMaterialMedia.modelCount,
				.sceneEmitterLightsVa = (
					sMode == ViewerMode::StageEdit
						? (
							sStageRender.emitterLightsBuffer.id != 0u
								? vkof::buffer_virtual_address(
									sStageRender.emitterLightsBuffer
								)
								: 0u
						)
						: (
							sScene.emitterLightsBuffer.id != 0u
								? vkof::buffer_virtual_address(
									sScene.emitterLightsBuffer
								)
								: 0u
						)
				),
				.sceneEmitterLightCount = (
					sMode == ViewerMode::StageEdit
						? sStageRender.emitterLightCount : sScene.emitterLightCount
				),
				.meshNeeEnabled = sMeshNeeEnabled,
				.envNeeEnabled = sEnvNeeEnabled,
				.environment = ponder::WavefrontEnvironment {
					.mode = (
						sEnvMode == 3 && sEnvTables.width == 0u
						? ponder::WavefrontEnvironmentMode::Furnace
						: (ponder::WavefrontEnvironmentMode)sEnvMode
					),
					.intensity = sEnvIntensity,
					.rotation = sEnvRotation,
					.radianceHandle = sEnvTables.radianceHandle,
					.pdfHandle = sEnvTables.pdfHandle,
					.aliasProbHandle = sEnvTables.aliasProbHandle,
					.aliasIndexHandle = sEnvTables.aliasIndexHandle,
					.emitterIndex = (
						sMode == ViewerMode::StageEdit
						? sStageRender.envLightIndex
						: sScene.envLightIndex
					),
					.backgroundIntensity = sEnvBackgroundIntensity,
				},
				.groundHeating = ponder::WavefrontGroundHeating {
					.enabled = sGroundHeatingEnabled,
					.groundHeight = sGroundHeatingGroundHeight,
					.hazeHeight = sGroundHeatingHazeHeight,
					.gradient = sGroundHeatingGradient,
					.noiseAmplitude = sGroundHeatingNoiseAmplitude,
					.noiseScale = sGroundHeatingNoiseScale,
				},
				.regir = ponder::RegirSettings {
					.enabled = sRegirEnabled,
					.cellSize = sRegirCellSize,
					// rr never advances the timeline; reseed every frame instead
					.fillIndex = rrActive ? sFrameIndex : sTimeline.frame,
					.maxAge = (u32)sRegirMaxAge,
					// frame stays 0 while the camera moves and stops when paused.
					// rr needs stable representatives; claiming a cell sets one
					.refresh = (
						!rrActive && sTimeline.playing && sTimeline.frame > 0u
					),
					.jitter = sRegirJitter || rrActive,
				},
				.modelsVa = vkof::buffer_virtual_address(
					sMode == ViewerMode::StageEdit
						? sStageRender.modelsIndirectBuffer
						: sScene.modelsIndirectBuffer
				),
				.shutterOpen = 0.0f,
				.shutterDuration = (
					(f32)(sShutterSeconds * fnBlackholeMetersPerSecond())
				),
				.lensRadius = sLensRadius,
				.focalDistance = sFocalDistance,
				.frameIndex = sFrameIndex,
				// rr needs fresh noise every frame, also while the timeline
				// holds its frame
				.sampleIndex = rrActive ? sFrameIndex : sTimeline.frame,
				.width = traceSize.x,
				.height = traceSize.y,
				.region = (
					sFocusActive && !probeRequested && !rrActive
						? std::optional(fnFocusRegion())
						: std::nullopt
				),
				.debugPath = sDebugPath.capturePending,
				.debugPixel = sDebugPath.pixel,
				.debugCamera = ponder::WavefrontCamera {
					.position = sDebugPath.camera.position,
					.forward = sDebugPath.camera.forward,
					.fovY = sDebugPath.camera.fovY,
				},
			}
		);

		u32 const colorHandle = vkof::transient_image_storage_handle({
			.image = colorTarget,
			.mipLevel = 0u,
		});
		u32 const ptOutputHandle = vkof::transient_image_storage_handle({
			.image = ptOutputTarget,
			.mipLevel = 0u,
		});

		constexpr u32 kDenoisePassCount = 4u;
		constexpr u32 kBloomNodeCount = 2u + kBloomBlurPassCount;
		vkof::RenderNode nodes[
			5u + kDenoisePassCount + kBloomNodeCount + 4u
		];
		u32 nodeCount = 0u;

		// -- regir fill node: refills claimed cells before integrate reads them
		// a pass keeps one fill throughout
		vkof::RenderNode regirFillNode {};
		bool const runRegirFill = (
			runResolve && sRegirEnabled && (passStarts || regirView)
		);
		if (runRegirFill) {
			regirFillNode = ponder::regir_fill(regir);
			nodes[nodeCount++] = regirFillNode;
		}

		// -- resolve node: wavefront integrate into colorTarget
		// colorTarget holds a finished sample only when passComplete
		vkof::RenderNode resolveNode {};
		bool passComplete = false;
		u32 stepFirst = 0u;
		u32 stepCount = 0u;
		// region sweeps the finished pass completed
		u32 sweepCount = 0u;
		if (runResolve) {
			bool const fullPass = (
				rrActive || headless || !sBudgetEnabled || sStepCost[0] <= 0.0
				|| sDebugPath.capturePending || probeRequested
			);
			// -- fill the frame with the pass's next steps up to the target
			u32 stepBudget = ponder::skWavefrontPassStepCount;
			if (!fullPass) {
				double const megaPaths = (double)wavefront.passPathCount * 1e-6;
				double const targetMs = (double)sBudgetTargetMs * 0.9;
				double predictedMs = 0.0;
				stepBudget = 0u;
				for (
					u32 step = wavefront.passStep;
					step < ponder::skWavefrontPassStepCount;
					++step
				) {
					predictedMs += sStepCost[step] * megaPaths;
					if (stepBudget > 0u && predictedMs > targetMs) { break; }
					++stepBudget;
				}
			}
			if (rrActive) {
				ponder::dlss_rr_prepare(
					sDlssRr, traceSize, u32v2 { sRenderW, sRenderH }
				);
			}
			auto const integrateInfo = ponder::WavefrontIntegrateInfo {
				.outputImage = colorTarget,
				.outputHandle = colorHandle,
				.aov = ponder::WavefrontAov {
					.positionImage = gbufferPosImage,
					.positionHandle = gbufferPosStorageHandle,
					.normalImage = gbufferNorImage,
					.normalHandle = gbufferNorStorageHandle,
					.albedoImage = gbufferAlbedoImage,
					.albedoHandle = gbufferAlbedoStorageHandle,
				},
				.rrGuide = (
					rrActive
						? std::optional(ponder::dlss_rr_guide(sDlssRr))
						: std::nullopt
				),
				.globalJitter = (
					rrActive
						? std::optional(ponder::dlss_rr_jitter(sDlssRr))
						: std::nullopt
				),
				.spectral = sSpectral,
				.debugPath = sDebugPath.capturePending,
				// the clamp is display-referred: luminance after exposure
				.fireflyClampLuminance = (
					sFireflyClampEnabled
						? sFireflyClampLuminance / std::max(sExposure, 1e-6f)
						: 0.0f
				),
				.width = traceSize.x,
				.height = traceSize.y,
				.stepBudget = stepBudget,
			};
			if (regirView) {
				ponder::regir_view_prepare(regir);
				resolveNode = (
					ponder::regir_view(regir, ponder::RegirViewInfo {
						.outputImage = colorTarget,
						.outputHandle = colorHandle,
						.mode = (u32)sRegirDebug.viewMode,
						.exposure = sExposure,
						.width = sRenderW,
						.height = sRenderH,
					})
				);
				passComplete = true;
				sweepCount = 1u;
			} else {
				ponder::WavefrontIntegrateResult const integrate = (
					ponder::wavefront_integrate(wavefront, integrateInfo)
				);
				resolveNode = integrate.node;
				passComplete = integrate.passComplete;
				stepCount = integrate.stepCount;
				sweepCount = integrate.sweepCount;
				stepFirst = (
					(passComplete ? ponder::skWavefrontPassStepCount : wavefront.passStep)
					- stepCount
				);
			}
			nodes[nodeCount++] = resolveNode;
		}
		{
			double predictedMs = 0.0;
			for (u32 step = stepFirst; step < stepFirst + stepCount; ++step) {
				predictedMs += (
					sStepCost[step] * (double)wavefront.passPathCount * 1e-6
				);
			}
			vkof::profiler_value("wavefront predicted ms", predictedMs);
			vkof::profiler_value("wavefront measured ms", sResolveMsLast);
		}
		vkof::profiler_value("wavefront steps", (double)stepCount);
		vkof::profiler_value("wavefront passes", passComplete ? 1.0 : 0.0);

		// -- accumulate a finished pass, unless a reset came after its start
		bool const passCurrent = (
			regirView ? !accumReset : sPassGeneration == sResetGeneration
		);
		bool const addSample = (
			runResolve && !freezeAccum && passComplete && !rrActive
		);
		bool const addReset = (
			addSample
			&& (
				!passCurrent
				|| (regirView ? accumReset : sAccumGeneration != sPassGeneration)
			)
		);
		if (addSample) {
			sAccumGeneration = (
				passCurrent && !regirView ? sPassGeneration : ~0u
			);
		}

		// -- regir debug node: resolves the inspected probe's reservoirs after
		// this frame's fill, when a readback is due
		vkof::RenderNode regirDebugNode { 0u };
		if (runRegirFill) {
			regirDebugNode = ponder::regir_debug_node(regir, sRegirDebug);
			if (regirDebugNode.id != 0u) {
				nodes[nodeCount++] = regirDebugNode;
			}
		}

		// -- accumulate node: xyz running sum, resolved to linear srgb + display
		vkof::RenderNode accumNode {};
		if (!rrActive) {
			accumNode = ponder::wavefront_accumulate(
				wavefront,
				ponder::WavefrontAccumulateInfo {
					.sampleImage = colorTarget,
					.sampleHandle = colorHandle,
					.resolvedImage = ptAccumImage,
					.resolvedHandle = ptAccumStorageHandle,
					.outputImage = ptOutputTarget,
					.outputHandle = ptOutputHandle,
					.addSample = addSample,
					.reset = addReset,
					.exposure = sExposure,
					.width = sRenderW,
					.height = sRenderH,
				}
			);
		}

		// -- nan-check node: tallies broken pixels in the raw radiance image
		vkof::RenderNode manifoldProbeNode {};
		vkof::RenderNode manifoldShadeNode {};
		if (
			sManifoldNeeEnabled
			&& !freezeAccum
			&& passComplete
			&& frameMode == FrameMode::PathTrace
			&& sScene.bufs.triangleCount > 0u
		) {
			manifoldProbeNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_persistent_image({
				.node = manifoldProbeNode,
				.image = gbufferPosImage,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_add_persistent_image({
				.node = manifoldProbeNode,
				.image = gbufferNorImage,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_callback({
				.node = manifoldProbeNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					GpuManifoldMeshData const caster = {
						.positionVa = sScene.bufs.positions,
						.attributeVa = sScene.bufs.attributes,
						.indexVa = sScene.bufs.flatIndices,
						.modelMatrix = f32m44_identity(),
						.triangleCount = sScene.bufs.triangleCount,
						.pad = 0u,
					};
					vkof::buffer_upload({
						.buffer = manifoldCasterBuffer,
						.byteOffset = 0u,
						.data = srat::slice<u8 const>(
							reinterpret_cast<u8 const *>(&caster),
							sizeof(caster)
						),
					});
					GpuManifoldProbePC const probePC = {
						.casterVa = (
							vkof::buffer_virtual_address(manifoldCasterBuffer)
						),
						.pointWo = sManifoldPointWo,
						.gbufferPosHandle = gbufferPosStorageHandle,
						.gbufferNorHandle = gbufferNorStorageHandle,
						.width = sRenderW,
						.height = sRenderH,
						.frameIndex = sFrameIndex,
						.countsVa = (
							vkof::buffer_virtual_address(manifoldCountsBuffer)
						),
						.bendPointsVa = (
							vkof::buffer_virtual_address(manifoldBendPointBuffer)
						),
					};
					vkof::cmd_dispatch_pushconst(
						vkof::CmdDispatchPushconst {
							.cmd = cmd,
							.pipeline = manifoldProbePipeline,
							.push = probePC,
							.threadgroupSize = u32v3 { 16u, 16u, 1u },
							.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
						}
					);
				},
			});
			nodes[nodeCount++] = manifoldProbeNode;

			manifoldShadeNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
				})
			);
			vkof::render_node_add_image({
				.node = manifoldShadeNode,
				.image = colorTarget,
				.access = vkof::RenderNodeAccess::readWrite,
			});
			vkof::render_node_callback({
				.node = manifoldShadeNode,
				.callback = [&](vkof::CommandBuffer const & cmd) {
					GpuManifoldShadePC const shadePC = {
						.bendPointsVa = (
							vkof::buffer_virtual_address(manifoldBendPointBuffer)
						),
						.colorHandle = colorHandle,
						.width = sRenderW,
						.height = sRenderH,
						.pad0 = 0u,
						.debugColor = sManifoldDebugColor,
						.pad1 = 0u,
					};
					vkof::cmd_dispatch_pushconst(
						vkof::CmdDispatchPushconst {
							.cmd = cmd,
							.pipeline = manifoldShadePipeline,
							.push = shadePC,
							.threadgroupSize = u32v3 { 16u, 16u, 1u },
							.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
						}
					);
				},
			});
			nodes[nodeCount++] = manifoldShadeNode;
		}

		vkof::RenderNode nanCheckNode {};
		// a partial pass leaves the rest of the sample image undefined
		bool const passCoversRegion = (
			wavefront.passPathCount == wavefront.passSlotCount
		);
		ponder::WavefrontRegion const nanRegion = wavefront.passRegion;
		if (
			!freezeAccum && passComplete && passCoversRegion
			&& frameMode == FrameMode::PathTrace
		) {
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
					{
						u32 const originX = nanRegion.origin.x;
						u32 const originY = nanRegion.origin.y;
						u32 const extentX = nanRegion.extent.x;
						u32 const extentY = nanRegion.extent.y;
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
		// -- dlss ray reconstruction nodes, in place of accumulate
		ponder::DlssRrNodes rrNodes {};
		if (rrActive) {
			rrNodes = ponder::dlss_rr_nodes(
				sDlssRr,
				ponder::DlssRrFrameInfo {
					.colorImage = colorTarget,
					.colorHandle = colorHandle,
					.aov = ponder::WavefrontAov {
						.positionImage = gbufferPosImage,
						.positionHandle = gbufferPosStorageHandle,
						.normalImage = gbufferNorImage,
						.normalHandle = gbufferNorStorageHandle,
						.albedoImage = gbufferAlbedoImage,
						.albedoHandle = gbufferAlbedoStorageHandle,
					},
					.resolvedImage = ptAccumImage,
					.resolvedHandle = ptAccumStorageHandle,
					.outputImage = ptOutputTarget,
					.outputHandle = ptOutputHandle,
					.exposure = sExposure,
					.camera = viewCamera,
					.quality = sDlssRrQuality,
					.preset = sDlssRrPreset,
					.reset = rrReset,
				}
			);
			nodes[nodeCount++] = rrNodes.prepare;
			nodes[nodeCount++] = rrNodes.evaluate;
			nodes[nodeCount++] = rrNodes.resolve;
		} else {
			nodes[nodeCount++] = accumNode;
		}

		// -- denoise nodes
		bool const runDenoise = sDenoiseEnabled && !rrActive;
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

		// -- bloom
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
			fnZeroManifoldCounts();
			fnZeroSppCounter();
		}

		// -- display dither node: the 8-bit presented image
		vkof::RenderNode ditherNode {};
		if (!headless) {
			u32 const displayHandle = vkof::transient_image_storage_handle({
				.image = displayTarget,
				.mipLevel = 0u,
			});
			ditherNode = (
				vkof::render_node_create({
					.queue = vkof::CommandQueue::graphics,
					.debugName = "display dither",
				})
			);
			vkof::render_node_add_image({
				.node = ditherNode,
				.image = ptOutputTarget,
				.access = vkof::RenderNodeAccess::read,
			});
			vkof::render_node_add_image({
				.node = ditherNode,
				.image = displayTarget,
				.access = vkof::RenderNodeAccess::write,
			});
			vkof::render_node_callback({
				.node = ditherNode,
				.callback = [&, displayHandle](vkof::CommandBuffer const & cmd) {
					GpuDisplayDitherPC const ditherPC = {
						.inputHandle = ptOutputHandle,
						.outputHandle = displayHandle,
						.frameIndex = sFrameIndex,
					};
					vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
						.cmd = cmd,
						.pipeline = displayDitherPipeline,
						.push = ditherPC,
						.threadgroupSize = u32v3 { 16u, 16u, 1u },
						.invocationCount = u32v3 { sRenderW, sRenderH, 1u },
					});
				},
			});
			nodes[nodeCount++] = ditherNode;
		}

		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(nodes, nodeCount),
			.rootPushconstant = srat::slice_as_bytes(globalPC),
			.finalImage = headless ? vkof::TransientImage { 0 } : displayTarget,
		});
		if (!headless) {
			vkof::render_node_destroy(ditherNode);
		}
		if (runRegirFill) {
			vkof::render_node_destroy(regirFillNode);
			if (regirDebugNode.id != 0u) {
				vkof::render_node_destroy(regirDebugNode);
			}
			// blocking, and only every refreshInterval frames
			ponder::regir_debug_readback(
				regir, sRegirDebug, sTimeline.frame, sFrameIndex, sRegirCellSize
			);
		}
		if (runResolve) {
			vkof::render_node_destroy(resolveNode);
		}
		if (!freezeAccum) {
			vkof::render_node_destroy(nanCheckNode);
			vkof::render_node_destroy(manifoldProbeNode);
			vkof::render_node_destroy(manifoldShadeNode);
		}
		if (rrActive) {
			vkof::render_node_destroy(rrNodes.prepare);
			vkof::render_node_destroy(rrNodes.evaluate);
			vkof::render_node_destroy(rrNodes.resolve);
		} else {
			vkof::render_node_destroy(accumNode);
		}
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

		// -- frame-budget measurement
		{
			u32 const slot = sFrameIndex % 2u;
			ResolveFrameHistory const prev = sResolveHistory[slot];
			if (prev.resolveRan) {
				double const gpuMs = vkof::node_gpu_ms(prev.resolveNodeIndex);
				if (gpuMs > 0.0 && prev.stepCount > 0u && prev.pathCount > 0u) {
					sResolveMsLast = gpuMs;
					double const megaPaths = (double)prev.pathCount * 1e-6;
					u32 const stepLast = prev.stepFirst + prev.stepCount;
					// -- first measurement: spread evenly
					if (sStepCost[0] <= 0.0) {
						for (double & cost : sStepCost) {
							cost = gpuMs / ((double)prev.stepCount * megaPaths);
						}
					}
					// -- scale the frame's steps toward the measured total;
					// frames cover different step mixes, so the per-step
					// split converges
					double predictedMs = 0.0;
					for (u32 step = prev.stepFirst; step < stepLast; ++step) {
						predictedMs += sStepCost[step] * megaPaths;
					}
					double const scale = (
						1.0 + 0.3 * (gpuMs / predictedMs - 1.0)
					);
					for (u32 step = prev.stepFirst; step < stepLast; ++step) {
						sStepCost[step] *= scale;
					}
				}
			}
			sResolveHistory[slot] = ResolveFrameHistory {
				.resolveRan = runResolve,
				.stepFirst = stepFirst,
				.stepCount = stepCount,
				.pathCount = wavefront.passPathCount,
				.resolveNodeIndex = runRegirFill ? 1u : 0u,
			};
		}

		++sFrameIndex;
		// -- the timeline advances one frame per finished region sweep
		if (
			sTimeline.playing
			&& frameMode == FrameMode::PathTrace
			&& addSample
		) {
			sTimeline.frame += sweepCount;
		}
		sAccumFrames = (
			addReset
				? 1u
				: (addSample ? sAccumFrames + 1u : sAccumFrames)
		);
		sPreviousFrameMode = frameMode;
		// region sweeps this frame completed; an rr frame is one whole pass
		if (rrActive) { return passComplete ? 1u : 0u; }
		return addSample ? sweepCount : 0u;
	};

	if (headless) {
		fnUploadFrameState();
		// a pass samples only part of the image when the path budget is
		// smaller, so count sweeps, not frames
		u32 sweeps = 0u;
		for (u32 frame = 0u; sweeps < std::max(args.spp, 1u); ++frame) {
			sweeps += fnRenderFrame(
				FrameMode::PathTrace, /*pathTraceReset=*/frame == 0u, false, false,
				/*rrReset=*/frame == 0u
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
			u32 const * const nanCounts = (
				reinterpret_cast<u32 const *>(
					vkof::buffer_host_address(nanCountsBuffer).ptr()
				)
			);
			if (sManifoldNeeEnabled) {
				u32 const * const manifoldCounts = (
					reinterpret_cast<u32 const *>(
						vkof::buffer_host_address(manifoldCountsBuffer).ptr()
					)
				);
				printf(
					"[manifold] %u seeds, %u converged (%.1f%%),"
					" %u left surface\n",
					manifoldCounts[0], manifoldCounts[1],
					manifoldCounts[0] > 0u
						? 100.0 * (f64)manifoldCounts[1] / (f64)manifoldCounts[0]
						: 0.0,
					manifoldCounts[2]
				);
			}
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
		bool leftMouseWasDown = false;
		bool focusDragActive = false;
		ImVec2 focusDragAnchor { 0.0f, 0.0f };
		bool sRunInBackground = true;

		while (!glfwWindowShouldClose(window)) {
			if (
				!sRunInBackground
				&& glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_FALSE
			) {
				glfwWaitEventsTimeout(0.1);
				continue;
			}
			glfwPollEvents();
			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}

			vkof::imgui_begin();

			f64 curMouseX, curMouseY;
			glfwGetCursorPos(window, &curMouseX, &curMouseY);

			bool gizmoUsed = false;
			bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
			// hold v: left-drag selects a tile region, right-click clears it
			bool const tileSelectHeld = (
				!ImGui::GetIO().WantCaptureKeyboard
				&& glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS
			);
			bool const cameraMoved = (
				!ponder::debug_path_picking()
				&& !tileSelectHeld
				&& !focusDragActive
				&& ponder::camera_controller_update(
					camController, window, fnActiveBoundsExtent(),
					ImGui::GetIO().DeltaTime
				)
			);
			if (ponder::debug_path_picking() || tileSelectHeld || focusDragActive) {
				ponder::camera_controller_scroll_discard();
			}
			// -- b + wheel: lens radius, geometric steps; below the floor is 0
			if (
				ImGui::IsKeyDown(ImGuiKey_B) && mouseFree
				&& !ImGui::GetIO().WantTextInput
				&& ImGui::GetIO().MouseWheel != 0.0f
			) {
				f32 constexpr kFloor = 0.001f;
				f32 const radius = (
					std::max(sLensRadius, kFloor)
					* std::pow(1.5f, ImGui::GetIO().MouseWheel)
				);
				sLensRadius = radius < kFloor ? 0.0f : std::min(radius, 0.1f);
				sLensChanged = true;
			}
			{
				i32 windowW, windowH;
				glfwGetWindowSize(window, &windowW, &windowH);
				f32v2 const windowSize = { (f32)windowW, (f32)windowH };
				if (!ptAccumImguiId) {
					ptAccumImguiId = (
						vkof::image_imgui_id({
							.image = ptAccumImage,
							.sampler = magnifierSampler,
						})
					);
				}
				auto const viewCamera = ponder::DebugPathCamera {
					.position = (
						sFirstPerson ? camFp.position : srat::camera_orbit_eye(cam)
					),
					.forward = (
						sFirstPerson
							? srat::camera_fp_forward(camFp)
							: cam.target - srat::camera_orbit_eye(cam)
					),
					.fovY = sFirstPerson ? camFp.fovY : cam.fovY,
					.renderWidth = sRenderW,
					.renderHeight = sRenderH,
				};
				ponder::debug_path_input(
					sDebugPath,
					ponder::DebugPathInputInfo {
						.camera = viewCamera,
						.image = ptAccumImguiId,
						.renderWidth = sRenderW,
						.renderHeight = sRenderH,
						.windowSize = windowSize,
					}
				);
				ponder::debug_path_overlay(sDebugPath, viewCamera, windowSize);
				if (sRegirEnabled) {
					ponder::regir_debug_overlay(sRegirDebug, viewCamera, windowSize);
				}
			}
			bool const timelineReset = ponder::timeline_imgui(sTimeline);
			ponder::debug_path_imgui(sDebugPath);
			if (sDebugPath.regirInspectSlot != ponder::skWavefrontRegirSlotInvalid) {
				ponder::regir_debug_select(sRegirDebug, sDebugPath.regirInspectSlot);
				sDebugPath.regirInspectSlot = ponder::skWavefrontRegirSlotInvalid;
			}
			if (sDebugPath.regirClearRequested) {
				ponder::regir_debug_select(
					sRegirDebug, ponder::skWavefrontRegirSlotInvalid
				);
				sDebugPath.regirClearRequested = false;
			}
			if (sRegirEnabled) {
				ponder::regir_debug_imgui(
					sRegirDebug,
					sFirstPerson ? camFp.position : srat::camera_orbit_eye(cam)
				);
			}
			bool const rightMouseDown = (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
			);
			bool const leftMouseDown = (
				glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS
			);
			if (tileSelectHeld || focusDragActive) {
				if (mouseFree && rightMouseDown && !rightMouseWasDown) {
					focusDragActive = false;
					sFocusActive = false;
				}
				if (
					tileSelectHeld && mouseFree
					&& leftMouseDown && !leftMouseWasDown
				) {
					focusDragActive = true;
					focusDragAnchor = ImVec2((f32)curMouseX, (f32)curMouseY);
				}
				if (focusDragActive && leftMouseDown) {
					ImGui::GetForegroundDrawList()->AddRect(
						focusDragAnchor,
						ImVec2((f32)curMouseX, (f32)curMouseY),
						IM_COL32(255, 200, 0, 255)
					);
				}
				if (focusDragActive && !leftMouseDown) {
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
			} else if (
				mouseFree
				&& rightMouseDown
				&& !rightMouseWasDown
				&& !ponder::debug_path_picking()
			) {
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
			leftMouseWasDown = leftMouseDown;
			if (sFocusActive) {
				i32 windowW, windowH;
				glfwGetWindowSize(window, &windowW, &windowH);
				if (windowW > 0 && windowH > 0) {
					ponder::WavefrontRegion const r = fnFocusRegion();
					f32 const sx = (f32)windowW / (f32)sRenderW;
					f32 const sy = (f32)windowH / (f32)sRenderH;
					ImVec2 const p0 {
						(f32)r.origin.x * sx,
						(f32)r.origin.y * sy,
					};
					ImVec2 const p1 {
						(f32)(r.origin.x + r.extent.x) * sx,
						(f32)(r.origin.y + r.extent.y) * sy,
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

			bool settingsChanged = (
				sRegirDebug.viewChanged || sLensChanged
			);
			sRegirDebug.viewChanged = false;
			sLensChanged = false;
			if (sRegirDebug.clearRequested) {
				ponder::regir_clear(regir);
				ponder::regir_debug_select(
					sRegirDebug, ponder::skWavefrontRegirSlotInvalid
				);
				sRegirDebug.clearRequested = false;
				settingsChanged = true;
			}
			std::string pendingLoadPath;
			std::string pendingEnvMapPath;
			std::string pendingVdbPath;
			bool pendingUnloadModel = false;
			bool pendingUnloadVdb = false;
			bool pendingNewStage = false;
			std::string pendingOpenStagePath;
			bool pendingSaveStage = false;
			bool pendingUnloadStage = false;
			std::string pendingAddGltfPath;
			std::string pendingAddVdbPath;
			bool stageInstancesChanged = false;
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
						if (ImGui::TreeNode("tag filter")) {
							ponder::asset_tag_filter_draw(
								sModelTagFilter, "model tag filter"
							);
							ImGui::TreePop();
						}
						pendingLoadPath = ponder::asset_list_draw(
							sModelList, sCurrentModelPath, "model list", 160.0f,
							&sModelTagFilter
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
				static char const * const skEnvModes[] = {
					"furnace", "checkerboard", "black", "hdrmap (exr)",
				};
				settingsChanged |= (
					ImGui::Combo("mode", &sEnvMode, skEnvModes, 4)
				);
				if (sEnvMode == 3) {
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
				ImGui::Text("environment intensity");
				settingsChanged |= (
					ImGui::SliderFloat(
						"##env intensity",
						&sEnvIntensity,
						0.0f,
						4.0f,
						"%.3f",
						ImGuiSliderFlags_Logarithmic
					)
				);
				ImGui::Text("background intensity");
				settingsChanged |= (
					ImGui::SliderFloat(
						"###background intensity", &sEnvBackgroundIntensity,
						0.0f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic
					)
				);
				settingsChanged |= (
					ImGui::Checkbox("mesh NEE (MIS)", &sMeshNeeEnabled)
					|| ImGui::Checkbox("env NEE (MIS)", &sEnvNeeEnabled)
					|| ImGui::Checkbox("manifold NEE", &sManifoldNeeEnabled)
					|| ImGui::Checkbox("regir", &sRegirEnabled)
				);
				if (sRegirEnabled) {
					ImGui::SliderInt(
						"regir max age", &sRegirMaxAge, 0, 1024, "%d frames",
						ImGuiSliderFlags_Logarithmic
					);
					settingsChanged |= (
						ImGui::Checkbox("regir cell jitter", &sRegirJitter)
					);
					ImGui::SetItemTooltip(
						"nee looks up a random neighbouring cell, hiding cell seams"
					);
				}
				if (sManifoldNeeEnabled) {
					settingsChanged |= ImGui::DragFloat3(
						"manifold pointWo", &sManifoldPointWo.x, 0.05f
					);
				}
				{
					bool const stageMode = sStageRender.tlas.id != 0u;
					u32 const emitterCount = (
						stageMode
							? sStageRender.emitterLightCount
							: sScene.emitterLightCount
					);
					u32 const emitterClipped = (
						stageMode ? sStageRender.emitterClipped : sScene.emitterClipped
					);
					f32 const emitterBuildMs = (
						stageMode ? sStageRender.emitterBuildMs : sScene.emitterBuildMs
					);
					ImGui::Text(
						"light list: %u entries, %.3f ms", emitterCount, emitterBuildMs
					);
					if (emitterClipped > 0u) {
						ImGui::TextColored(
							ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
							"  %u emitters clipped (bsdf-sampled only)",
							emitterClipped
						);
					}
				}
				settingsChanged |= (
					ImGui::Checkbox("env map lighting only", &sEnvMapOnlyMode)
				);
				settingsChanged |= (
					ImGui::SliderFloat(
						"env rotation", &sEnvRotation, 0.0f, 6.28318530718f
					)
				);
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
				ImGui::Checkbox("run in background", &sRunInBackground);
				ImGui::SetItemTooltip("off: rendering stops while the window is unfocused");
				ImGui::SeparatorText("camera");
				if (sFirstPerson) {
					ImGui::Text(
						"pos %.2f %.2f %.2f",
						camFp.position.x, camFp.position.y, camFp.position.z
					);
					ImGui::Text("yaw %.3f", camFp.yaw);
					ImGui::Text("pitch %.3f", camFp.pitch);
					ImGui::Text("speed %.2f", camController.flySpeedMultiplier);
				} else {
					ImGui::Text("azimuth %.3f", cam.azimuth);
					ImGui::Text("elevation %.3f", cam.elevation);
					ImGui::Text("distance %.3f", cam.distance);
				}
				ImGui::TextUnformatted("fov");
				settingsChanged |= (
					ImGui::SliderFloat("##fov", &cam.fovY, 0.2f, 2.5f)
				);
				camFp.fovY = cam.fovY;
				ImGui::TextUnformatted("lens radius (m)");
				settingsChanged |= (
					ImGui::SliderFloat(
						"##lensradius",
						&sLensRadius,
						0.0f,
						0.1f,
						"%.4f",
						ImGuiSliderFlags_Logarithmic
					)
				);
				ImGui::TextUnformatted("focal distance (m)");
				settingsChanged |= (
					ImGui::DragFloat(
						"##focaldistance",
						&sFocalDistance,
						0.01f,
						0.01f,
						1000.0f,
						"%.2f"
					)
				);
				ImGui::SeparatorText("render");
				{
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
					auto const fnPresetLabel = (
						[](void * data, int idx) -> char const * {
							return (
								static_cast<RenderResPreset const *>(data)[idx]
								.label
							);
						}
					);
					i32 newIdx = resIdx;
					ImGui::TextUnformatted("resolution");
					ImGui::SetNextItemWidth(-FLT_MIN);
					ImGui::Combo(
						"##resolution",
						&newIdx,
						fnPresetLabel,
						(void *)skRenderResPresets,
						(i32)IM_ARRAYSIZE(skRenderResPresets)
					);
					ImGui::TextUnformatted("render resolution");
					ImGui::SetNextItemWidth(-FLT_MIN);
					if (
						ImGui::Combo(
							"##render resolution",
							&sRenderBudgetIdx,
							fnPresetLabel,
							(void *)skRenderResPresets,
							(i32)IM_ARRAYSIZE(skRenderResPresets)
						)
					) {
						sPathBudgetOverride = 0u;
						settingsChanged = true;
					}
					if (newIdx != resIdx && newIdx >= 0) {
						sRenderW = skRenderResPresets[newIdx].w;
						sRenderH = skRenderResPresets[newIdx].h;
						vkof::render_extent_set(sRenderW, sRenderH);
						vkof::image_destroy(ptAccumImage);
						fnCreateAccumImage();
						vkof::image_destroy(gbufferPosImage);
						vkof::image_destroy(gbufferNorImage);
						vkof::image_destroy(gbufferAlbedoImage);
						fnCreateGbufferImages();
						vkof::buffer_destroy(manifoldBendPointBuffer);
						fnCreateManifoldBendPoints();
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
				// the firefly clamp scales with exposure, so samples change
				if (
					ImGui::SliderFloat(
						"exposure",
						&sExposure,
						0.01f,
						256.0f,
						"%.2f", ImGuiSliderFlags_Logarithmic
					)
					&& sFireflyClampEnabled
				) {
					settingsChanged = true;
				}
				settingsChanged |= (
					ImGui::Checkbox("antialias", &sAntialias)
				);
				settingsChanged |= ImGui::Checkbox("spectral", &sSpectral);
				ImGui::Checkbox("denoise (a-trous)", &sDenoiseEnabled);
				{
					bool const rrSupported = ponder::dlss_rr_supported();
					ImGui::BeginDisabled(!rrSupported);
					settingsChanged |= (
						ImGui::Checkbox("DLSS-RR (realtime)", &sDlssRrEnabled)
					);
					ImGui::EndDisabled();
					if (!rrSupported) {
						ImGui::SameLine();
						ImGui::TextDisabled(
							"(%s)", ponder::dlss_rr_unsupported_reason()
						);
					}
					if (sDlssRrEnabled) {
						constexpr char const * skPresetLabels[] = {
							"default", "D", "E", "F",
						};
						constexpr ponder::DlssRrPreset skPresets[] = {
							ponder::DlssRrPreset::Default,
							ponder::DlssRrPreset::D,
							ponder::DlssRrPreset::E,
							ponder::DlssRrPreset::F,
						};
						i32 presetIndex = 0;
						for (i32 i = 0; i < 4; ++i) {
							if (skPresets[i] == sDlssRrPreset) { presetIndex = i; }
						}
						if (
							ImGui::Combo(
								"preset##dlssrr", &presetIndex, skPresetLabels, 4
							)
						) {
							sDlssRrPreset = skPresets[presetIndex];
						}
						constexpr char const * skQualityLabels[] = {
							"DLAA", "quality", "balanced", "performance",
							"ultra performance",
						};
						constexpr ponder::DlssRrQuality skQualities[] = {
							ponder::DlssRrQuality::Dlaa,
							ponder::DlssRrQuality::Quality,
							ponder::DlssRrQuality::Balanced,
							ponder::DlssRrQuality::Performance,
							ponder::DlssRrQuality::UltraPerformance,
						};
						i32 qualityIndex = 0;
						for (i32 i = 0; i < 5; ++i) {
							if (skQualities[i] == sDlssRrQuality) { qualityIndex = i; }
						}
						if (
							ImGui::Combo(
								"upscale##dlssrr", &qualityIndex, skQualityLabels, 5
							)
						) {
							sDlssRrQuality = skQualities[qualityIndex];
						}
						ImGui::TextDisabled(
							"traces %ux%u", sTraceSize.x, sTraceSize.y
						);
						if (sDlssRr.evaluateFailed) {
							ImGui::TextColored(
								ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
								"RR evaluate failed, see the log"
							);
						}
					}
				}
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
				ImGui::Checkbox("frame budget", &sBudgetEnabled);
				if (sBudgetEnabled) {
					ImGui::SliderFloat(
						"target ms", &sBudgetTargetMs, 1.0f, 64.0f, "%.1f ms"
					);
				}
				if (sFocusActive) {
					if (ImGui::SmallButton("clear region")) {
						sFocusActive = false;
					}
					ponder::WavefrontRegion const r = fnFocusRegion();
					ImGui::Text(
						"focused %u x %u px (outside pixels frozen)",
						r.extent.x, r.extent.y
					);
				} else {
					ImGui::TextDisabled(
						"hold v + left-drag: focus region, v + right-click: clear"
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
				if (sManifoldNeeEnabled) {
					u32 const * const manifoldCounts = (
						reinterpret_cast<u32 const *>(
							vkof::buffer_host_address(manifoldCountsBuffer).ptr()
						)
					);
					u32 const attempted = manifoldCounts[0];
					f32 const rate = (
						attempted > 0u
							? 100.0f * f32(manifoldCounts[1]) / f32(attempted)
							: 0.0f
					);
					ImGui::Text(
						"manifold: %u seeds, %.1f%% converged, %u left surface",
						attempted, (f64)rate, manifoldCounts[2]
					);
				}
				bool const openMaterialPopup = sMaterialPopupPending;
				sMaterialPopupPending = false;
				for (std::string const & msg : sProbeMessages) {
					ImGui::TextColored(
						ImVec4(1.0f, 0.6f, 0.1f, 1.0f), "%s", msg.c_str()
					);
				}
				ImGui::End();

				ImGui::Begin("Blackhole");
				settingsChanged |= (
					ImGui::Checkbox("enabled##blackhole", &sBlackholeEnabled)
				);
				if (sBlackholeEnabled) {
					f32 const extent = fnActiveBoundsExtent();
					if (ImGui::Button("place in view##blackhole")) {
						f32v3 const eye = (
							sFirstPerson
							? camFp.position
							: srat::camera_orbit_eye(cam)
						);
						f32v3 const target = (
							sFirstPerson
							? camFp.position + srat::camera_fp_forward(camFp) * extent
							: cam.target
						);
						sBlackholeOri = eye + (target - eye) * 0.5f;
						sBlackholeMass = extent * 0.02f;
						settingsChanged = true;
					}
					settingsChanged |= (
						ImGui::DragFloat3(
							"position##blackhole",
							&sBlackholeOri.x,
							extent * 0.01f
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"mass (m)##blackhole",
							&sBlackholeMass,
							extent * 0.001f,
							0.0f,
							extent,
							"%.5f"
						)
					);
					settingsChanged |= (
						ImGui::SliderFloat(
							"spin (a/M)##blackhole",
							&sBlackholeSpin,
							-0.998f,
							0.998f,
							"%.3f"
						)
					);

					ImGui::SeparatorText("integration");
					settingsChanged |= (
						ImGui::DragFloat(
							"lens radius (M)##blackhole",
							&sBlackholeLensRadius,
							10.0f,
							10.0f,
							100000.0f,
							"%.0f"
						)
					);
					settingsChanged |= (
						ImGui::SliderInt(
							"max chords##blackhole",
							&sBlackholeMaxChords,
							16,
							4096
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"step factor##blackhole",
							&sBlackholeStepFactor,
							0.001f,
							0.005f,
							0.5f,
							"%.3f"
						)
					);

					ImGui::SeparatorText("disk");
					settingsChanged |= (
						ImGui::Checkbox("enabled##blackholedisk", &sBlackholeDiskEnabled)
					);
					if (sBlackholeDiskEnabled) {
						settingsChanged |= (
							ImGui::DragFloatRange2(
								"radii (M)##blackholedisk",
								&sBlackholeDiskInnerRadius,
								&sBlackholeDiskOuterRadius,
								0.1f,
								2.0f,
								200.0f,
								"%.1f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"temperature (K)##blackholedisk",
								&sBlackholeDiskTemperature,
								50.0f,
								1000.0f,
								100000.0f,
								"%.0f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"brightness##blackholedisk",
								&sBlackholeDiskBrightness,
								0.01f,
								0.0f,
								100.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"optical depth##blackholedisk",
								&sBlackholeDiskOpticalDepth,
								0.05f,
								0.0f,
								100.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"optical depth falloff##blackholedisk",
								&sBlackholeDiskOpticalDepthExponent,
								0.01f,
								0.0f,
								8.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::Checkbox(
								"prograde##blackholedisk",
								&sBlackholeDiskPrograde
							)
						);

						ImGui::SeparatorText("disk animation");
						ImGui::Checkbox("animate##blackholedisk", &sBlackholeDiskAnimate);
						ImGui::DragFloat(
							"inner orbits per second##blackholedisk",
							&sBlackholeDiskOrbitsPerSecond,
							0.01f,
							0.0f,
							10.0f,
							"%.2f"
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"shutter (s)##blackholedisk",
								&sShutterSeconds,
								0.01f,
								0.0f,
								10.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"noise amplitude##blackholedisk",
								&sBlackholeDiskNoiseAmplitude,
								0.01f,
								0.0f,
								1.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"noise scale##blackholedisk",
								&sBlackholeDiskNoiseScale,
								0.05f,
								0.1f,
								64.0f,
								"%.2f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"hotspot radius (M)##blackholedisk",
								&sBlackholeDiskHotspotRadius,
								0.1f,
								2.0f,
								200.0f,
								"%.1f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"hotspot temperature (K)##blackholedisk",
								&sBlackholeDiskHotspotTemperature,
								50.0f,
								0.0f,
								100000.0f,
								"%.0f"
							)
						);
						settingsChanged |= (
							ImGui::DragFloat(
								"hotspot size (M)##blackholedisk",
								&sBlackholeDiskHotspotSize,
								0.05f,
								0.05f,
								50.0f,
								"%.2f"
							)
						);
					}
				}
				ImGui::End();

				ImGui::Begin("Ambient medium");
				{
					char const * const ambientMediumTypes[] = {
						"none", "homogeneous", "height fog",
					};
					settingsChanged |= (
						ImGui::Combo(
							"type##ambientMedium",
							&sAmbientMediumType,
							ambientMediumTypes,
							IM_ARRAYSIZE(ambientMediumTypes)
						)
					);
					if (sAmbientMediumType != (i32)ponder::WavefrontAmbientMediumType::None) {
						settingsChanged |= (
							ImGui::DragFloat(
								"density##ambientMedium",
								&sAmbientMediumDensity,
								0.0005f,
								0.0f,
								10.0f,
								"%.4f"
							)
						);
						ImGui::TextUnformatted("scattering (σs per meter)");
						ImGui::SetNextItemWidth(-FLT_MIN);
						settingsChanged |= (
							ImGui::ColorEdit3(
								"##ambientMediumscattering",
								&sAmbientMediumScattering.x,
								ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR
							)
						);
						ImGui::TextUnformatted("absorption (σa per meter)");
						ImGui::SetNextItemWidth(-FLT_MIN);
						settingsChanged |= (
							ImGui::ColorEdit3(
								"##ambientMediumabsorption",
								&sAmbientMediumAbsorption.x,
								ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR
							)
						);

						ImGui::SeparatorText("phase");
						char const * const ambientMediumPhases[] = {
							"henyey-greenstein", "draine", "fog (droplets)",
						};
						settingsChanged |= (
							ImGui::Combo(
								"##ambientMediumphase",
								&sAmbientMediumPhase,
								ambientMediumPhases,
								IM_ARRAYSIZE(ambientMediumPhases)
							)
						);
						if (
							sAmbientMediumPhase
							!= (i32)ponder::WavefrontAmbientMediumPhase::Fog
						) {
							settingsChanged |= (
								ImGui::SliderFloat(
									"g##ambientMedium", &sAmbientMediumAnisotropy, -0.99f, 0.99f
								)
							);
						}
						if (
							sAmbientMediumPhase
							== (i32)ponder::WavefrontAmbientMediumPhase::Draine
						) {
							settingsChanged |= (
								ImGui::SliderFloat(
									"α##ambientMedium", &sAmbientMediumDraineAlpha, 0.0f, 50.0f
								)
							);
						}
						if (
							sAmbientMediumPhase
							== (i32)ponder::WavefrontAmbientMediumPhase::Fog
						) {
							ImGui::TextUnformatted("droplet diameter (µm)");
							ImGui::SetNextItemWidth(-FLT_MIN);
							settingsChanged |= (
								ImGui::SliderFloat(
									"##ambientMediumdroplet",
									&sAmbientMediumDropletDiameter,
									5.0f,
									50.0f
								)
							);
						}

						if (sAmbientMediumType == (i32)ponder::WavefrontAmbientMediumType::HeightFog) {
							ImGui::SeparatorText("height");
							settingsChanged |= (
								ImGui::DragFloat(
									"origin##ambientMedium", &sAmbientMediumHeightOrigin, 0.05f
								)
							);
							settingsChanged |= (
								ImGui::DragFloat(
									"falloff##ambientMedium",
									&sAmbientMediumHeightFalloff,
									0.05f,
									0.01f,
									1000.0f
								)
							);
						}
					}
				}
				if (ImGui::CollapsingHeader("ground heating")) {
					settingsChanged |= (
						ImGui::Checkbox("enabled##groundHeating", &sGroundHeatingEnabled)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"ground height##groundHeating",
							&sGroundHeatingGroundHeight,
							0.05f
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"haze height##groundHeating",
							&sGroundHeatingHazeHeight,
							0.05f,
							0.001f,
							1000.0f
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"gradient##groundHeating",
							&sGroundHeatingGradient,
							0.0001f,
							0.0f,
							1.0f,
							"%.5f"
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"noise amplitude##groundHeating",
							&sGroundHeatingNoiseAmplitude,
							0.0001f,
							0.0f,
							1.0f,
							"%.5f"
						)
					);
					settingsChanged |= (
						ImGui::DragFloat(
							"noise scale##groundHeating",
							&sGroundHeatingNoiseScale,
							0.01f,
							0.01f,
							100.0f
						)
					);
				}
				ImGui::End();
				if (openMaterialPopup) {
					ImGui::OpenPopup("material inspector");
				}

				// -- every material of the scene, by model
				std::vector<ponder::MaterialMediaEntry> materialEntries;
				auto const fnAddMaterials = [&](
					i32 const modelDrawIndex,
					std::string const & modelName,
					mor::Scene const & scene
				) {
					u32 const count = mor::scene_material_count(scene);
					for (u32 m = 0u; m < count; ++m) {
						materialEntries.emplace_back(ponder::MaterialMediaEntry {
							.modelDrawIndex = modelDrawIndex,
							.materialIndex = (i32)m,
							.modelName = modelName,
							.materialName = mor::scene_material_name(scene, m),
						});
					}
				};
				if (sMode == ViewerMode::StageEdit) {
					auto const & instances = sStageRender.modelDrawIndexToInstance;
					for (usize i = 0u; i < instances.size(); ++i) {
						stage::GltfInstance const & instance = (
							sStage.gltfInstances[instances[i]]
						);
						stage::GltfModel const & model = (
							sStage.gltfModels[instance.modelIndex]
						);
						fnAddMaterials(
							(i32)i,
							instance.name,
							sStage.gltfMeshes[model.meshIndex].scene
						);
					}
				} else if (sScene.blas.id != 0u) {
					fnAddMaterials(
						0,
						std::filesystem::path(sCurrentModelPath).filename().string(),
						sScene.scene
					);
				}
				if (
					ponder::material_media_list_imgui(
						sMaterialMedia,
						materialEntries,
						sPickedModelDrawIndex,
						sPickedMaterialIndex
					)
				) {
					ImGui::OpenPopup("material inspector");
				}
			}
			if (sMode == ViewerMode::StageEdit) {
				i32 windowW = 0, windowH = 0;
				glfwGetWindowSize(window, &windowW, &windowH);

				// -- model drag
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
					ImGui::Dummy(ImGui::GetContentRegionAvail());
					if (ImGui::BeginDragDropTarget()) {
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
				gizmoProj.m[5] = -gizmoProj.m[5];
				gizmoUsed = stage::stage_imgui_gizmo_edit(
					sStage, view, gizmoProj,
					f32v4 { 0.0f, 0.0f, (f32)windowW, (f32)windowH }
				);
				if (gizmoUsed) {
					stage_render_rebuild_tlas(
						sStageRender, sStage, sMeshNeeEnabled, environment_light_power()
					);
					settingsChanged = true;
				}
			}
			if (ImGui::BeginPopup("material inspector")) {
				bool matChanged = false;
				if (sMode == ViewerMode::StageEdit) {
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
				matChanged |= (
					ponder::material_media_imgui(
						sMaterialMedia, sPickedModelDrawIndex, sPickedMaterialIndex
					)
				);
				if (matChanged) {
					settingsChanged = true;
				}
				ImGui::EndPopup();
			}

			// -- environment map switch
			if (!pendingEnvMapPath.empty()) {
				settingsChanged |= fnLoadEnvMap(pendingEnvMapPath);
			}
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

			if (!pendingLoadPath.empty() && !sCurrentVdbPath.empty()) {
				pendingUnloadVdb = true;
			}
			if (!pendingVdbPath.empty() && !sCurrentModelPath.empty()) {
				pendingUnloadModel = true;
			}

			// the pass in flight and the regir cells index the old scene
			auto const fnSceneReplaced = [&]() {
				ponder::wavefront_pass_restart(wavefront);
				ponder::regir_clear(regir);
			};

			// -- model switch
			if (!pendingLoadPath.empty()) {
				sMaterialMedia.media.clear();
				ponder::material_media_upload(sMaterialMedia);
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
				fnSceneReplaced();
				if (sScene.blas.id == 0u) {
					break;
				}
				fnReframeCamera(cam);
				ponder::camera_controller_sync_fp_from_orbit(camController);
				settingsChanged = true;
			}

			// -- model unload
			if (pendingUnloadModel) {
				sMaterialMedia.media.clear();
				ponder::material_media_upload(sMaterialMedia);
				scene_state_destroy(sScene);
				sCurrentModelPath.clear();
				sScene = scene_state_create(nullptr);
				fnSceneReplaced();
				if (!sVdb.blob.empty()) {
					vdb_bounds(
						sVdb, sScene.boundsCenter, sScene.boundsExtent
					);
				}
				fnReframeCamera(cam);
				ponder::camera_controller_sync_fp_from_orbit(camController);
				settingsChanged = true;
			}

			// -- vdb switch/unload
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
					fnLoadVdbTemperatureIfPresent(pendingVdbPath.c_str());
					sVdbList = ponder::scan_asset_paths(
						settingsPath, { "vdbPaths" }, { ".vdb" },
						/*warnMissingKey=*/false
					);
				}
				if (sCurrentModelPath.empty() && !sVdb.blob.empty()) {
					vdb_bounds(
						sVdb, sScene.boundsCenter, sScene.boundsExtent
					);
					fnReframeCamera(cam);
					ponder::camera_controller_sync_fp_from_orbit(camController);
				}
				settingsChanged = true;
			}

			// -- new stage
			if (pendingNewStage) {
				sMode = ViewerMode::StageEdit;
				sStage = stage::Stage {};
				sStage.environment.mode = stage::EnvironmentMode::Furnace;
				stage_render_rebuild_blases(sStageRender, sStage);
				stage_render_rebuild_tlas(
					sStageRender, sStage, sMeshNeeEnabled, environment_light_power()
				);
				fnSceneReplaced();
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
					stage_render_rebuild_tlas(
						sStageRender,
						sStage,
						sMeshNeeEnabled,
						environment_light_power()
					);
					fnSceneReplaced();
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

			// -- save stage
			if (pendingSaveStage) {
				if (sStagePathBuffer[0] == '\0') {
					printf("no save path set\n");
				} else if (!stage::stage_save(sStage, sStagePathBuffer)) {
					printf("failed to save stage '%s'\n", sStagePathBuffer);
				}
			}

			// -- unload stage
			if (pendingUnloadStage) {
				stage_render_destroy(sStageRender);
				stage::stage_destroy(sStage);
				vkof::acceleration_structure_set_tlas(sScene.tlas);
				sMode = ViewerMode::FileView;
				fnSceneReplaced();
				fnReframeCamera(cam);
				ponder::camera_controller_sync_fp_from_orbit(camController);
				settingsChanged = true;
			}

			// -- add instance
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

			if (gltfInstanceCountChanged) {
				stage_render_rebuild_blases(sStageRender, sStage);
				stage_render_rebuild_tlas(
					sStageRender, sStage, sMeshNeeEnabled, environment_light_power()
				);
				fnSceneReplaced();
				stageInstancesChanged = false;
				settingsChanged = true;
			}
			if (sMode == ViewerMode::StageEdit) {
				stage::stage_flush_pending_gpu_destroys(sStage);
			}

			if (stageInstancesChanged) {
				stage_render_rebuild_tlas(
					sStageRender, sStage, sMeshNeeEnabled, environment_light_power()
				);
				fnSceneReplaced();
				settingsChanged = true;
			}

			if (sMode == ViewerMode::StageEdit) {
				sStage.environment.mode = (stage::EnvironmentMode)sEnvMode;
				sStage.environment.path = sCurrentEnvMapPath;
				sStage.environment.rotationRadians = sEnvRotation;
				sStage.environment.intensity = sEnvIntensity;
			}

			// -- focus
			{
				f32v3 center;
				f32 extent;
				if (ponder::debug_path_focus(sDebugPath, center, extent)) {
					focusRequest.requested = true;
					focusRequest.center = center;
					focusRequest.extent = extent;
				}
				if (
					sRegirEnabled
					&& ponder::regir_debug_focus(sRegirDebug, center, extent)
				) {
					focusRequest.requested = true;
					focusRequest.center = center;
					focusRequest.extent = extent;
				}
			}
			if (focusRequest.requested) {
				cam.target = focusRequest.center;
				cam.distance = focusRequest.extent * 1.5f;
				cam.near = focusRequest.extent * 0.001f;
				cam.far = focusRequest.extent * 100.0f;
				ponder::camera_controller_sync_fp_from_orbit(camController);
				settingsChanged = true;
			}

			bool const probeRequested = sProbePixel.x >= 0;
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
			if (sBlackholeDiskAnimate) {
				sBlackholeDiskSeconds = ponder::timeline_seconds(sTimeline);
			}
			if (sTimeline.playing && (cameraMoved || gizmoUsed || otherReset)) {
				sTimeline.frame = 0u;
			}
			if (
				sDebugPath.captured
				&& sDebugPath.path.sampleIndex != sTimeline.frame
			) {
				sDebugPath.capturePending = true;
			}
			if (sDebugPath.capturePending) {
				ponder::wavefront_debug_path_prepare(wavefront);
			}
			bool const pathTraceReset = (
				(otherReset || timelineReset) && sFrameMode == FrameMode::PathTrace
			);
			bool const freezeAccum = (
				sFrameMode == FrameMode::PathTrace && !pathTraceReset
				&& sMaxSpp > 0 && fnCompletedPixels() >= sRenderW * sRenderH
			);
			fnRenderFrame(
				sFrameMode, pathTraceReset, freezeAccum, probeRequested,
				/*rrReset=*/otherReset || timelineReset
			);
			if (sDebugPath.capturePending) {
				ponder::wavefront_debug_path_read(wavefront, sDebugPath.path);
				sDebugPath.capturePending = false;
				sDebugPath.captured = true;
			}
			if (
				ponder::debug_path_material_pick(
					sDebugPath, sPickedModelDrawIndex, sPickedMaterialIndex
				)
			) {
				sMaterialPopupPending = true;
			}
			if (ponder::debug_path_focus_pick(sDebugPath, sFocalDistance)) {
				sLensChanged = true;
			}
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
				vkof::screenshot(ptAccumImage, outPath.c_str(), sExposure);
				sLastScreenshotPath = outPath;
				printf("wrote %s\n", outPath.c_str());
			}
			sProbeMessages.clear();
			for (u32 i = 0u, count = vkof::probe_message_count(); i < count; ++i) {
				sProbeMessages.emplace_back(vkof::probe_message(i));
			}
			vkof::probe_reset();

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

	ponder::material_media_destroy(sMaterialMedia);
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
	if (ptAccumImguiId) {
		vkof::image_imgui_id_destroy(ptAccumImguiId);
	}
	vkof::sampler_destroy(magnifierSampler);
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
	vkof::pipeline_destroy(manifoldProbePipeline);
	vkof::pipeline_destroy(manifoldShadePipeline);
	vkof::buffer_destroy(manifoldCountsBuffer);
	vkof::buffer_destroy(manifoldCasterBuffer);
	vkof::buffer_destroy(manifoldBendPointBuffer);
	vkof::pipeline_destroy(ptAccumulatePipeline);
	ponder::regir_debug_destroy(sRegirDebug);
	ponder::regir_destroy(regir);
	ponder::dlss_rr_destroy(sDlssRr);
	ponder::wavefront_destroy(wavefront);
	vkof::shutdown();
	return 0;
}
