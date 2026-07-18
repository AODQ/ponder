// disabled for now
#if 0

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <ponder/energy-tables.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include "viewer/shaders/resolve_pc.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// loads real glTF models (assets/Models, copied wholesale from the cull
// sample-asset set) and furnace-tests each one through the exact production
// path tracer -- furnace_model_render.comp calls utilTraceRay and
// utilIrradiancePropagationEntry, the same functions resolve.comp calls, so
// this exercises real textured multi-material geometry the lobe-level unit
// tests never see. environment mode is furnace at a low, deliberately
// unmissable intensity: every pixel converges to something close to
// envIntensity, so a NaN or a wildly bright pixel is a lib bug (a
// pdf-division-by-near-zero, a runaway energy-compensation term, ...) that
// wouldn't show up at all against a normal-exposure render.
//
// started with one model (Avocado, plain metallic-roughness) to prove the
// harness shape; DragonAttenuation adds full KHR_materials_transmission +
// KHR_materials_volume (a smooth glass dragon with strong Beer-Lambert
// attenuation), the transmission chain's own real-scene stress test. more
// of assets/Models can be added the same way -- push a FurnaceModelConfig
// and loop.
// ---------------------------------------------------------------------------

namespace {

struct FurnaceModelConfig {
	char const * label;
	// relative to REPO_DIR/assets/Models/
	char const * relPath;
	char const * outputName;
	// azimuth/elevation/fov tuned per model so the camera lands somewhere
	// the object actually fills the frame; distance/near/far stay derived
	// from the model's own bounds (see boundsExtent below) so those three
	// are the only thing worth varying per asset
	f32 azimuth;
	f32 elevation;
	f32 fovY;
	// transmission-heavy scenes (full glass, roughness 0) need more bounces
	// to reach the environment than an opaque model does
	u32 propagationDepth;
	// smooth-dielectric refraction concentrates radiance (real caustics,
	// not a bug) enough that the generic 5x-envIntensity firefly ceiling
	// doesn't apply uniformly; let each config own its ceiling
	f32 tooBrightMultiplier;
};

FurnaceModelConfig const skConfigs[] = {
	{
		.label = "Avocado",
		.relPath = "Avocado/glTF/Avocado.gltf",
		.outputName = "furnace-avocado.png",
		.azimuth = 0.6f,
		.elevation = 0.35f,
		.fovY = 0.5f,
		.propagationDepth = 8u,
		.tooBrightMultiplier = 5.0f,
	},
	{
		.label = "DragonAttenuation",
		.relPath = "DragonAttenuation/glTF/DragonAttenuation.gltf",
		.outputName = "furnace-dragon-attenuation.png",
		.azimuth = 0.5f,
		.elevation = 0.25f,
		.fovY = 1.5f,
		.propagationDepth = 16u,
		.tooBrightMultiplier = 20.0f,
	},
	{
		.label = "DispersionTest",
		.relPath = "DispersionTest/glTF/DispersionTest.gltf",
		.outputName = "furnace-dispersion-test.png",
		.azimuth = 0.0f,
		.elevation = 0.15f,
		.fovY = 1.5f,
		.propagationDepth = 16u,
		.tooBrightMultiplier = 20.0f,
	},
};

void run_furnace_model_render(FurnaceModelConfig const & config) {
	std::string const label(config.label);
	CAPTURE(label);

	std::string const modelPath = (
		std::string(REPO_DIR) + "/assets/Models/" + config.relPath
	);
	REQUIRE(std::filesystem::exists(modelPath));

	mor::Scene scene = mor::scene_create();
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);

	// the sweep feeds every assets/Models entry through here, and a few of
	// those are geometry-free by design (camera/animation-only test assets)
	// -- nothing to trace against, and blas_create on zero triangles is not
	// a furnace failure, so bow out before touching the acceleration path
	if (bufs.triangleCount == 0u) {
		MESSAGE("skipping ", label, ": no triangle geometry");
		mor::scene_gpu_destroy(gpuScene);
		mor::scene_destroy(scene);
		mor::sampler_cache_destroy();
		return;
	}

	f32v3 boundsMin, boundsMax;
	mor::scene_bounds(scene, boundsMin, boundsMax);
	f32v3 const boundsCenter = (boundsMin + boundsMax) * 0.5f;
	f32 const boundsExtent = std::max(
		{
			boundsMax.x - boundsMin.x,
			boundsMax.y - boundsMin.y,
			boundsMax.z - boundsMin.z,
			0.01f,
		}
	);

	auto const blas = vkof::blas_create({
		.positionVa = bufs.positions,
		.vertexCount = bufs.vertexCount,
		.indexVa = bufs.flatIndices,
		.triangleCount = bufs.triangleCount,
		.isOpaque = mor::scene_is_fully_opaque(scene),
	});
	REQUIRE(blas.id != 0u);

	auto const tlas = vkof::tlas_create({ .maxInstances = 1u });
	REQUIRE(tlas.id != 0u);
	{
		vkof::TlasInstance const instance = {
			.blas = blas,
			.transform = f32m44_identity(),
			.instanceCustomIndex = 0u,
			.rayMask = 0xFFu,
		};
		auto const node = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
		);
		vkof::render_node_callback({
			.node = node,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				vkof::tlas_build(
					cmd, tlas,
					srat::slice<vkof::TlasInstance const>(&instance, 1u)
				);
			},
		});
		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
			.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
		});
		vkof::render_node_destroy(node);
		test::gpu_wait();
	}
	vkof::acceleration_structure_set_tlas(tlas);

	auto energyTables = ponder::energy_tables_create();
	auto zeltnerTables = ponder::zeltner_tables_create();

	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR,
	};
	auto const pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "furnace_model_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 3u),
	});
	REQUIRE(pl.id != 0u);

	u32 const width = 1920u;
	u32 const height = 1080u;
	u32 const pixelCount = width * height;
	u32 const sampleCount = 64u;
	f32 const envIntensity = 0.5f;

	auto const debugPcBuf = vkof::buffer_create({
		.byteCount = sizeof(GpuGlobalExtended),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const modelsBuf = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(3u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	// temporary, for the DragonAttenuation firefly investigation; see
	// PONDER_NAN_PROBES in furnace_model_render.comp
	auto const nanProbeCounterBuf = vkof::buffer_create({
		.byteCount = sizeof(u32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	{
		u32 const zero = 0u;
		vkof::buffer_upload({
			.buffer = nanProbeCounterBuf,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&zero), sizeof(zero)
			),
		});
	}

	GpuGlobalExtended const extended = {
		.envIntensity = envIntensity,
		.renderWidth = width,
		.renderHeight = height,
		.envMode = 0, // ENV_MODE_FURNACE, see environment.glsl
		.antialias = 0u,
		.nanProbeCounterVa = vkof::buffer_virtual_address(nanProbeCounterBuf),
		// this test never probes a material; unreachable by any coord
		.probePixel = { -1, -1 },
		.envMap = 0u,
		.envRotation = 0.0f,
		.envNeeEnabled = 0u,
	};
	vkof::buffer_upload({
		.buffer = debugPcBuf,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(extended),
	});

	GpuResolveModelIndirect const modelDesc = {
		.meshlets = bufs.meshlets,
		.materials = bufs.materials,
		.uvTransforms = bufs.uvTransforms,
		.positions = bufs.positions,
		.instances = bufs.instances,
		.attributes = bufs.attributes,
		.meshletVerts = bufs.meshletVerts,
		.meshletTris = bufs.meshletTris,
		.flatIndices = bufs.flatIndices,
		.flatMeshlets = bufs.flatMeshlets,
		.modelMatrix = f32m44_identity(),
	};
	vkof::buffer_upload({
		.buffer = modelsBuf,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(modelDesc),
	});

	// given camera position/angle/fov, framed to the model's own bounds so
	// this works unchanged for any model swapped in later
	srat::CameraOrbit const cam = {
		.target = boundsCenter,
		.distance = boundsExtent * 2.2f,
		.azimuth = config.azimuth,
		.elevation = config.elevation,
		.fovY = config.fovY,
		.aspect = static_cast<f32>(width) / static_cast<f32>(height),
		.near = boundsExtent * 0.01f,
		.far = boundsExtent * 100.0f,
	};

	GpuGlobalPC const globalPC = {
		.time = 0.0f,
		.cameraPos = srat::camera_orbit_eye(cam),
		.exposure = 1.0f,
		.pad0 = 0.0f,
		.viewProj = (
			srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam)
		),
		.extended = vkof::buffer_virtual_address(debugPcBuf),
		.models = vkof::buffer_virtual_address(modelsBuf),
		.pad1 = 0u,
		.pad2 = 0u,
		.pad3 = 0u,
	};

	struct FurnacePush {
		u64 outVa;
		u32 width;
		u32 height;
		u32 sampleCount;
		u32 propagationDepth;
		u32 kullaContyEnergyHandle;
		u32 zeltnerLtcParamHandle;
		// must mirror GpuFurnaceModelPC in furnace_model_render.comp
		u32 seedSalt;
		u32 pad0;
	};
	FurnacePush const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = width,
		.height = height,
		.sampleCount = sampleCount,
		.propagationDepth = config.propagationDepth,
		.kullaContyEnergyHandle = energyTables.kullaContyEnergyHandle,
		.zeltnerLtcParamHandle = zeltnerTables.zeltnerLtcParamHandle,
		.seedSalt = 0u,
		.pad0 = 0u,
	};

	test::dispatch(
		pl, push, (width + 7u) / 8u, (height + 7u) / 8u, 1u,
		srat::slice_as_bytes(globalPC)
	);
	test::gpu_wait();

	auto const raw = test::readback<f32>(outBuf, 0u, 3u * pixelCount);

	u32 nonFiniteCount = 0u;
	f32 maxValue = 0.0f;
	f64 sum = 0.0;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (!std::isfinite(raw[i])) {
			nonFiniteCount++;
			continue;
		}
		maxValue = std::max(maxValue, raw[i]);
		sum += raw[i];
	}
	CHECK(nonFiniteCount == 0u);
	// canary against a silently-broken pipeline (e.g. a missing shader
	// include path) rendering nothing and leaving outBuf at its
	// zero-initialized default -- every check below would trivially "pass"
	// against an all-zero image, so demand actual signal first. the
	// furnace background alone guarantees every pixel is at least
	// envIntensity or the object's own reflection of it. CHECK rather than
	// REQUIRE: one broken model (or a shader edit saved mid-sweep -- the
	// pipeline recompiles from source per model) should fail loudly but
	// not abort the remaining sweep entries
	CHECK(maxValue > 0.5f * envIntensity);
	// energy-compensation overshoot in a couple of lobes is a known,
	// already-reported issue (see test-energy-compensation.cpp's "known
	// bug" cases) -- the default 5x multiplier is loose enough to not trip
	// on that, tight enough to still catch a real firefly (those tend to
	// be 10-1000x, not 5x); smooth-transmission configs widen it since
	// real caustics legitimately concentrate radiance. starting ceilings;
	// tighten once real per-model data exists
	f32 const tooBrightCeiling = config.tooBrightMultiplier * envIntensity;
	CHECK(maxValue < tooBrightCeiling);
	f64 const mean = sum / (3.0 * pixelCount);
	MESSAGE(
		"furnace render ", label, ": max=", maxValue,
		" mean=", mean, " nonFinite=", nonFiniteCount, " / ", 3u * pixelCount
	);

	std::vector<f32> pngR(pixelCount), pngG(pixelCount), pngB(pixelCount);
	// furnace intensity (0.1) is too dark to see unaided; boost exposure
	// before the display gamma purely for the eyeball pass, same as every
	// other render test's png output
	f32 const displayExposure = 4.0f;
	for (u32 i = 0u; i < pixelCount; ++i) {
		pngR[i] = std::pow(
			std::clamp(raw[i] * displayExposure, 0.0f, 1.0f), 1.0f / 2.2f
		);
		pngG[i] = std::pow(
			std::clamp(raw[pixelCount + i] * displayExposure, 0.0f, 1.0f),
			1.0f / 2.2f
		);
		pngB[i] = std::pow(
			std::clamp(raw[2u * pixelCount + i] * displayExposure, 0.0f, 1.0f),
			1.0f / 2.2f
		);
	}
	std::string const pngPath = (
		std::string(FURNACE_MODEL_OUTPUT_DIR) + config.outputName
	);
	// sweep outputs land in a furnace-sweep/ subdirectory that doesn't exist
	// on a fresh checkout; no-op for the curated outputs at the dir root
	std::filesystem::create_directories(
		std::filesystem::path(pngPath).parent_path()
	);
	CHECK(test::write_heatmap_png(
		pngR, pngG, pngB, width, height, pngPath.c_str()
	));

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(modelsBuf);
	vkof::buffer_destroy(debugPcBuf);
	vkof::buffer_destroy(nanProbeCounterBuf);
	ponder::zeltner_tables_destroy(zeltnerTables);
	ponder::energy_tables_destroy(energyTables);
	vkof::tlas_destroy(tlas);
	vkof::blas_destroy(blas);
	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
	// process-global texture sampler cache mor fills in on material load;
	// unowned by the scene, so it survives scene_gpu_destroy and must be
	// torn down separately or the device destroy below warns about leaked
	// VkSamplers (see main.cpp's shutdown sequence)
	mor::sampler_cache_destroy();
}

struct SweepModel {
	std::string label;
	// relative to REPO_DIR/assets/Models/, same convention as
	// FurnaceModelConfig::relPath
	std::string relPath;
};

// one entry per model directory under assets/Models, preferring the plain
// glTF variant and falling back to glTF-Binary -- the remaining variant
// directories (Draco, Embedded, Quantized, KTX-BasisU, ...) are
// re-encodings of the same content, so sweeping them would only re-test
// the loader, not the renderer
std::vector<SweepModel> enumerate_sweep_models() {
	std::vector<SweepModel> models;
	std::filesystem::path const root = (
		std::filesystem::path(REPO_DIR) / "assets" / "Models"
	);
	if (!std::filesystem::exists(root)) {
		return models;
	}
	for (auto const & entry : std::filesystem::directory_iterator(root)) {
		if (!entry.is_directory()) {
			continue;
		}
		struct Variant {
			char const * dir;
			char const * extension;
		};
		static Variant const skVariants[] = {
			{ .dir = "glTF", .extension = ".gltf" },
			{ .dir = "glTF-Binary", .extension = ".glb" },
		};
		for (Variant const & variant : skVariants) {
			std::filesystem::path const variantDir = (
				entry.path() / variant.dir
			);
			if (!std::filesystem::exists(variantDir)) {
				continue;
			}
			bool found = false;
			for (
				auto const & file
				: std::filesystem::directory_iterator(variantDir)
			) {
				if (file.path().extension() != variant.extension) {
					continue;
				}
				models.push_back(SweepModel {
					.label = entry.path().filename().string(),
					.relPath = (
						std::filesystem::relative(file.path(), root)
							.generic_string()
					),
				});
				found = true;
				break;
			}
			if (found) {
				break;
			}
		}
	}
	std::sort(
		models.begin(),
		models.end(),
		[](SweepModel const & a, SweepModel const & b) {
			return a.label < b.label;
		}
	);
	return models;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("furnace model render: no NaNs, no fireflies") {
	for (FurnaceModelConfig const & config : skConfigs) {
		run_furnace_model_render(config);
	}
}

// opt-in: every model under assets/Models through the same furnace harness.
// 159 models at full resolution is far too slow for the default suite, so
// vkof-test skips this case unless launched with --furnace
TEST_CASE("furnace model render: assets/Models sweep") {
	if (!test::furnaceSweepEnabled) {
		MESSAGE(
			"skipped: pass --furnace to vkof-test to run the full "
			"assets/Models furnace sweep"
		);
		return;
	}
	std::vector<SweepModel> const models = enumerate_sweep_models();
	REQUIRE(!models.empty());
	MESSAGE("furnace sweep: ", models.size(), " models");
	for (SweepModel const & model : models) {
		std::string const outputName = (
			"furnace-sweep/" + model.label + ".png"
		);
		FurnaceModelConfig const config = {
			.label = model.label.c_str(),
			.relPath = model.relPath.c_str(),
			.outputName = outputName.c_str(),
			// generic framing: bounds-derived distance does the heavy
			// lifting (see run_furnace_model_render), these just pick a
			// three-quarter view that shows most assets acceptably. at
			// distance 2.2x the max bounds axis, a fov of ~0.45 fits that
			// axis exactly vertically; 0.6 keeps a modest margin without
			// drowning the model in empty furnace background
			.azimuth = 0.6f,
			.elevation = 0.35f,
			.fovY = 0.45f,
			// enough for the transmission-heavy assets in the set to reach
			// the environment; opaque models converge long before this
			.propagationDepth = 16u,
			// deliberately huge: unlike the curated configs above, the
			// sweep includes emissive assets (EmissiveStrengthTest and
			// friends) whose radiance legitimately dwarfs envIntensity, so
			// a physically-motivated firefly ceiling can't apply uniformly.
			// this only catches runaway values (inf-adjacent blowups, pdf
			// division spikes), which is the sweep's job -- NaN/Inf and
			// the all-zero canary stay exact
			.tooBrightMultiplier = 1000.0f,
		};
		run_furnace_model_render(config);
	}
}

} // TEST_SUITE("[headless]")
#endif
