#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <ponder/emitter-tables.hpp>
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
// loads real glTF models (assets/models-categorized, copied wholesale from the cull
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
// of assets/models-categorized can be added the same way -- push a FurnaceModelConfig
// and loop.
// ---------------------------------------------------------------------------

namespace {

struct FurnaceModelConfig {
	char const * label;
	// relative to REPO_DIR/assets/models-categorized/
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
	// (TODO REVIEW)
	// tier 3.11 of plan-nee.md: build the scene's real emitter list and turn
	// mesh nee on. only meaningful for a model that actually has emissive
	// materials, and left off for the furnace configs above
	bool meshNee;
	// (TODO REVIEW)
};

// what the caller needs to compare two runs of the same model
struct FurnaceModelResult {
	f64 mean;
	f32 maxValue;
	u32 nonFiniteCount;
};

FurnaceModelConfig const skConfigs[] = {
	{
		.label = "Avocado",
		.relPath = "test-core/Avocado.glb",
		.outputName = "furnace-avocado.png",
		.azimuth = 0.6f,
		.elevation = 0.35f,
		.fovY = 0.5f,
		.propagationDepth = 8u,
		.tooBrightMultiplier = 5.0f,
		.meshNee = false,
	},
	{
		.label = "DragonAttenuation",
		.relPath = "test-transparency/DragonAttenuation.glb",
		.outputName = "furnace-dragon-attenuation.png",
		.azimuth = 0.5f,
		.elevation = 0.25f,
		.fovY = 1.5f,
		.propagationDepth = 16u,
		.tooBrightMultiplier = 20.0f,
		.meshNee = false,
	},
	{
		.label = "DispersionTest",
		.relPath = "test-transparency/DispersionTest.glb",
		.outputName = "furnace-dispersion-test.png",
		.azimuth = 0.0f,
		.elevation = 0.15f,
		.fovY = 1.5f,
		.propagationDepth = 16u,
		.tooBrightMultiplier = 20.0f,
		.meshNee = false,
	},
};

FurnaceModelResult run_furnace_model_render(FurnaceModelConfig const & config) {
	std::string const label(config.label);
	CAPTURE(label);

	std::string const modelPath = (
		std::string(REPO_DIR) + "/assets/models-categorized/" + config.relPath
	);
	REQUIRE(std::filesystem::exists(modelPath));

	mor::Scene scene = mor::scene_create();
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);

	// the sweep feeds every assets/models-categorized entry through here, and a few of
	// those are geometry-free by design (camera/animation-only test assets)
	// -- nothing to trace against, and blas_create on zero triangles is not
	// a furnace failure, so bow out before touching the acceleration path
	if (bufs.triangleCount == 0u) {
		MESSAGE("skipping ", label, ": no triangle geometry");
		mor::scene_gpu_destroy(gpuScene);
		mor::scene_destroy(scene);
		mor::sampler_cache_destroy();
		return FurnaceModelResult { 0.0, 0.0f, 0u };
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
		.pathCompute = TEST_SHADER_DIR "slang/entry/furnace_model_render.slang",
		.includePaths = srat::slice<char const * const>(includePaths, 3u),
	});
	REQUIRE(pl.id != 0u);

	// a nan/firefly probe wants paths, and paths are pixels x samples. the
	// default halves each axis (a quarter of the work) and keeps the sample
	// count, so per-pixel noise -- which is what the max-value ceiling is
	// sensitive to -- is unchanged
	u32 const width = test::thoroughEnabled ? 1920u : 960u;
	u32 const height = test::thoroughEnabled ? 1080u : 540u;
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

	// (TODO REVIEW)
	// tier 3.11: the scene's own emitter list, built exactly as the viewer
	// builds it per frame -- real gltf meshlets, real world areas, real
	// emissive factors, rather than the hand-authored quads tiers 0-3 use
	std::vector<GpuEmitterLight> emitterLights;
	vkof::Buffer emitterBuf {};
	ponder::EmitterRange emitterRange { 0u, 0u, 0u };
	if (config.meshNee) {
		std::vector<mor::MeshletAreaInfo> const areas = (
			mor::scene_meshlet_area_info(scene)
		);
		std::vector<ponder::EmitterMeshletInput> inputs;
		ponder::emitter_meshlet_inputs_from_scene(
			scene, mor::GpuMaterials {},
			srat::slice<mor::MeshletAreaInfo const>(areas.data(), areas.size()),
			inputs
		);
		emitterRange = ponder::emitter_lights_append(
			emitterLights,
			srat::slice<ponder::EmitterMeshletInput const>(
				inputs.data(), inputs.size()
			),
			f32m44_identity(), /*modelDrawIndex=*/0u, /*maxLights=*/65536u
		);
		(void)ponder::emitter_lights_finalize(emitterLights);
		if (!emitterLights.empty()) {
			emitterBuf = test::upload_bytes(
				emitterLights.data(),
				emitterLights.size() * sizeof(GpuEmitterLight)
			);
		}
	}
	// (TODO REVIEW)

	// value-initialised then assigned rather than one designated initializer:
	// GpuGlobalExtended keeps growing, and -Werror=missing-field-initializers
	// turns every new field into a build break here
	GpuGlobalExtended extended {};
	extended.envIntensity = envIntensity;
	extended.renderWidth = width;
	extended.renderHeight = height;
	// ENV_MODE_FURNACE, see environment.glsl
	extended.envMode = 0;
	extended.antialias = 0u;
	extended.nanProbeCounterVa = (
		vkof::buffer_virtual_address(nanProbeCounterBuf)
	);
	// this test never probes a material; unreachable by any coord
	extended.probePixel = { -1, -1 };
	extended.envMap = 0u;
	extended.envRotation = 0.0f;
	extended.envNeeEnabled = 0u;
	extended.envBackgroundIntensity = 1.0f;
	// (TODO REVIEW)
	extended.emitterLights = (
		emitterBuf.id != 0u ? vkof::buffer_virtual_address(emitterBuf) : 0u
	);
	extended.emitterLightCount = (u32)emitterLights.size();
	extended.meshNeeEnabled = config.meshNee ? 1u : 0u;
	// the environment is not in this list, so it keeps the whole pick mass on
	// its own side and every entry here is a mesh emitter
	extended.envLightIndex = (u32)emitterLights.size();
	// (TODO REVIEW)
	extended.vdbWorldToLocal = f32m44_identity();
	extended.vdbPhaseAnisotropyOverride = -2.0f;
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
		// the backward (bsdf-hit) lookup scans only this model's own range
		.emitterLightOffset = emitterRange.offset,
		.emitterLightCount = emitterRange.count,
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

	GpuGlobalPc const globalPC = {
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
	if (emitterBuf.id != 0u) { vkof::buffer_destroy(emitterBuf); }
	mor::sampler_cache_destroy();
	return FurnaceModelResult { mean, maxValue, nonFiniteCount };
}

struct SweepModel {
	std::string label;
	// relative to REPO_DIR/assets/models-categorized/, same convention as
	// FurnaceModelConfig::relPath
	std::string relPath;
};

// one entry per model file under assets/models-categorized/<category>/. the
// tree now holds a single already-chosen variant per model rather than the old
// per-model glTF/glTF-Binary/Draco/... directories, so there is nothing to
// prefer between -- take every gltf/glb in the tree
std::vector<SweepModel> enumerate_sweep_models() {
	std::vector<SweepModel> models;
	std::filesystem::path const root = (
		std::filesystem::path(REPO_DIR) / "assets" / "models-categorized"
	);
	if (!std::filesystem::exists(root)) {
		return models;
	}
	for (
		auto const & entry
		: std::filesystem::recursive_directory_iterator(root)
	) {
		if (!entry.is_regular_file()) {
			continue;
		}
		std::filesystem::path const & path = entry.path();
		std::string const extension = path.extension().string();
		if (extension != ".gltf" && extension != ".glb") {
			continue;
		}
		models.push_back(SweepModel {
			.label = path.stem().string(),
			.relPath = (
				std::filesystem::relative(path, root).generic_string()
			),
		});
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
		(void)run_furnace_model_render(config);
	}
}

// opt-in: every model under assets/models-categorized through the same furnace harness.
// 159 models at full resolution is far too slow for the default suite, so
// vkof-test skips this case unless launched with --furnace
TEST_CASE("furnace model render: assets/models-categorized sweep") {
	if (!test::furnaceSweepEnabled) {
		MESSAGE(
			"skipped: pass --furnace to vkof-test to run the full "
			"assets/models-categorized furnace sweep"
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
			// the sweep is a nan/firefly probe over the whole asset set; the
			// nee integration gets its own case below on the emissive models
			.meshNee = false,
		};
		(void)run_furnace_model_render(config);
	}
}

// ---------------------------------------------------------------------------
// tier 3.11 of plan-nee.md -- emissive-mesh nee against real authored geometry.
//
// Every other tier-3 case runs on hand-built buffers that are deliberately
// clean: flat quads, one material, no degenerate triangles, no curved emitter
// self-shadowing. This is the only one that points the sampler at geometry
// somebody actually modelled, where it meets slivers, zero-area triangles,
// grazing configurations and emissive meshlets spread over many materials.
//
// assets/models-categorized/test-emissive exists, so the emission is the
// asset's own rather than forced on -- a truer test than overwriting the
// materials, and it exercises the real gltf emissive load path with it.
// ---------------------------------------------------------------------------
TEST_CASE("furnace model render: emissive models agree with nee on and off") {
	struct EmissiveModel {
		char const * label;
		char const * relPath;
		char const * outputName;
	};
	static EmissiveModel const skEmissive[] = {
		{
			.label = "EmissiveStrengthTest",
			.relPath = "test-emissive/EmissiveStrengthTest.glb",
			.outputName = "furnace-nee-emissive-strength.png",
		},
		{
			.label = "CompareEmissiveStrength",
			.relPath = "test-emissive/CompareEmissiveStrength.glb",
			.outputName = "furnace-nee-compare-emissive-strength.png",
		},
		{
			.label = "PlaysetLightTest",
			.relPath = "test-emissive/PlaysetLightTest.glb",
			.outputName = "furnace-nee-playset-light.png",
		},
	};

	for (EmissiveModel const & model : skEmissive) {
		std::string const label(model.label);
		CAPTURE(label);
		FurnaceModelConfig config = {
			.label = model.label,
			.relPath = model.relPath,
			.outputName = model.outputName,
			.azimuth = 0.6f,
			.elevation = 0.35f,
			.fovY = 0.6f,
			.propagationDepth = 8u,
			// emissive assets are authored far above envIntensity by design
			// (EmissiveStrengthTest sweeps KHR_materials_emissive_strength up
			// to 1000), so a physical firefly ceiling cannot apply -- this
			// only catches inf-adjacent blowups, and NaN stays exact
			.tooBrightMultiplier = 100000.0f,
			.meshNee = true,
		};
		FurnaceModelResult const on = run_furnace_model_render(config);
		config.meshNee = false;
		config.outputName = "furnace-nee-off-scratch.png";
		FurnaceModelResult const off = run_furnace_model_render(config);

		CHECK(on.nonFiniteCount == 0u);
		CHECK(off.nonFiniteCount == 0u);
		REQUIRE(off.mean > 0.0);

		// no paired-realization sigma here: these renders are expensive and
		// the emissive assets are dominated by directly-visible emitters,
		// which both configs resolve identically and nearly noiselessly. a
		// percent-scale relative gate is the honest instrument -- it catches
		// the double-count and energy-loss failures the earlier tiers found
		// (those were 15-70%), and does not pretend to resolve better
		f64 const relative = std::abs(on.mean - off.mean) / off.mean;
		MESSAGE(
			"nee on/off ", label, ": on=", on.mean, " off=", off.mean,
			" relative=", relative * 100.0, "%"
		);
		CHECK(relative < 0.03);
	}
}

} // TEST_SUITE("[headless]")

