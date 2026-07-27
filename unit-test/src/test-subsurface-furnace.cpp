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
#include <vector>

// assembled-lobe furnace/energy test for the full subsurface material,
// eighth milestone in the openpbr port-verification series: renders a
// closed-mesh scene (assets/Models/Box, a simple cube -- any closed
// manifold works, this just needs *some* interior volume for the walk to
// traverse) with subsurfaceWeight = 1 through the exact production path
// tracer (utilTraceRay + utilIrradiancePropagationEntry, same calls
// resolve.comp makes). every prior subsurface test exercised one primitive
// in isolation with a bespoke minimal shader (albedo->sigma inversion, the
// walk step, the entry lobe's sampler/pdf); this is the only one that
// actually drives openPbrSampleWo's subsurface-entry branch ->
// utilSubsurfaceWalk -> utilLoadSurfaceHit's real mesh-hit unpacking ->
// the exit vertex's transmission-reuse refraction+nee, wired together the
// way resolve.comp would use them.
//
// material is built from scratch (mor::GpuMorMaterial{}, every component
// untextured) rather than using the box asset's own material, so the
// subsurface parameters are fully controlled and known.

namespace {

GpuMorMaterialComponent1 comp1(f32 const v) {
	return { v, 0u, 0, 0u };
}
GpuMorMaterialComponent3 comp3(f32v3 const v) {
	return { v, 0u, 0u };
}

GpuMorMaterial make_subsurface_material(
	f32v3 const subsurfaceColor,
	f32 const subsurfaceRadius,
	f32 const subsurfaceScatterAnisotropy,
	f32v3 const subsurfaceRadiusScale = { 1.0f, 1.0f, 1.0f }
) {
	GpuMorMaterial mat;
	mat.baseWeight = comp1(1.0f);
	mat.baseColor = comp3({ 0.8f, 0.8f, 0.8f });
	mat.baseMetalness = comp1(0.0f);
	mat.baseDiffuseRoughness = comp1(0.0f);
	mat.specularWeight = comp1(1.0f);
	mat.specularColor = comp3({ 1.0f, 1.0f, 1.0f });
	mat.specularRoughness = comp1(0.3f);
	mat.specularRoughnessAnisotropy = comp1(0.0f);
	mat.specularRoughnessAnisotropyRotation = comp1(0.0f);
	mat.specularIor = comp1(1.4f);
	mat.transmissionWeight = comp1(0.0f);
	mat.transmissionColor = comp3({ 1.0f, 1.0f, 1.0f });
	mat.transmissionDepth = comp1(0.0f);
	mat.transmissionScatter = comp3({ 0.0f, 0.0f, 0.0f });
	mat.transmissionScatterAnisotropy = comp1(0.0f);
	mat.transmissionDispersionScale = comp1(0.0f);
	mat.transmissionDispersionAbbeNumber = comp1(20.0f);
	mat.subsurfaceWeight = comp1(1.0f);
	mat.subsurfaceColor = comp3(subsurfaceColor);
	mat.subsurfaceRadius = comp1(subsurfaceRadius);
	mat.subsurfaceRadiusScale = comp3(subsurfaceRadiusScale);
	mat.subsurfaceScatterAnisotropy = comp1(subsurfaceScatterAnisotropy);
	mat.coatWeight = comp1(0.0f);
	mat.coatColor = comp3({ 1.0f, 1.0f, 1.0f });
	mat.coatRoughness = comp1(0.0f);
	mat.coatRoughnessAnisotropy = comp1(0.0f);
	mat.coatIor = comp1(1.6f);
	mat.coatDarkening = comp1(1.0f);
	mat.fuzzWeight = comp1(0.0f);
	mat.fuzzColor = comp3({ 1.0f, 1.0f, 1.0f });
	mat.fuzzRoughness = comp1(0.5f);
	mat.emissionLuminance = comp1(0.0f);
	mat.emissionColor = comp3({ 0.0f, 0.0f, 0.0f });
	mat.thinFilmWeight = comp1(0.0f);
	mat.thinFilmThickness = comp1(0.5f);
	mat.thinFilmIor = comp1(1.4f);
	mat.geometryOpacity = comp1(1.0f);
	mat.geometryThinWalled = comp1(0.0f);
	mat.geometryNormalTexture = 0u;
	mat.geometryNormalUvTransform = 0u;
	mat.geometryTangentTexture = 0u;
	mat.geometryTangentUvTransform = 0u;
	mat.geometryCoatNormalTexture = 0u;
	mat.geometryCoatNormalUvTransform = 0u;
	mat.geometryCoatTangentTexture = 0u;
	mat.geometryCoatTangentUvTransform = 0u;
	mat.alphaCutoff = 0.0f;
	return mat;
}

struct SubsurfaceFurnacePush {
	u64 outVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 propagationDepth;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 seedSalt;
	u32 pad0;
};

struct RenderResult {
	std::vector<f32> rgb;
	u32 width;
	u32 height;
};

RenderResult render_subsurface_box(
	f32v3 const subsurfaceColor,
	f32 const subsurfaceRadius,
	f32 const subsurfaceScatterAnisotropy,
	u32 const seedSalt,
	f32v3 const subsurfaceRadiusScale = { 1.0f, 1.0f, 1.0f }
) {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);
	REQUIRE(bufs.triangleCount > 0u);

	// full material override -- see make_subsurface_material's header
	// comment for why this replaces the box's own material entirely
	// rather than layering on top of it
	mor::GpuMaterials const materialsOverride = (
		mor::scene_gpu_materials_create(scene)
	);
	u32 const materialCount = mor::scene_gpu_materials_count(materialsOverride);
	REQUIRE(materialCount > 0u);
	// index 0 is mor's reserved default material, which no gltf-loaded
	// primitive references -- overriding only it silently does nothing, so
	// every slot gets the probe material
	for (u32 i = 0u; i < materialCount; ++i) {
		mor::scene_material_override_scalars(
			materialsOverride, i,
			make_subsurface_material(
				subsurfaceColor, subsurfaceRadius, subsurfaceScatterAnisotropy,
				subsurfaceRadiusScale
			)
		);
	}
	mor::scene_gpu_materials_upload(materialsOverride);

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
		.pathCompute = TEST_SHADER_DIR "subsurface_furnace_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 3u),
	});
	REQUIRE(pl.id != 0u);

	u32 const width = 256u;
	u32 const height = 256u;
	u32 const pixelCount = width * height;
	u32 const sampleCount = 32u;
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

	GpuGlobalExtended const extended = {
		.envIntensity = envIntensity,
		.renderWidth = width,
		.renderHeight = height,
		.envMode = 0, // ENV_MODE_FURNACE, see environment.glsl
		.antialias = 0u,
		.frameMode = 0,
		.nanProbeCounterVa = 0u,
		.probePixel = { -1, -1 },
		.probeResult = 0u,
		.envMap = 0u,
		.envRotation = 0.0f,
		.envNeeEnabled = 1u,
		.envMapOnlyMode = 0u,
		.fireflyClampLuminance = 0.0f,
		.envBackgroundIntensity = 1.0f,
		.vdb = 0u,
		.vdbWorldToLocal = f32m44_identity(),
		.vdbSigmaAbsorption = { 0.0f, 0.0f, 0.0f },
		.vdbSigmaScattering = { 0.0f, 0.0f, 0.0f },
		.vdbDropletDiameter = 20.0f,
		.vdbPhaseAnisotropyOverride = -2.0f,
		.vdbTemperatureGrid = 0u,
		.vdbTemperatureScale = 0.0f,
		.vdbEmissionScale = 0.0f,
		.fogThickness = 0.0f,
		.fogAlbedo = { 1.0f, 1.0f, 1.0f },
		.fogDistanceMax = 0.0f,
	};
	vkof::buffer_upload({
		.buffer = debugPcBuf,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(extended),
	});

	GpuResolveModelIndirect const modelDesc = {
		.meshlets = bufs.meshlets,
		.materials = mor::scene_gpu_materials_va(materialsOverride),
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

	srat::CameraOrbit const cam = {
		.target = boundsCenter,
		.distance = boundsExtent * 2.2f,
		.azimuth = 0.6f,
		.elevation = 0.35f,
		.fovY = 0.6f,
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

	SubsurfaceFurnacePush const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = width,
		.height = height,
		.sampleCount = sampleCount,
		.propagationDepth = 24u,
		.kullaContyEnergyHandle = energyTables.kullaContyEnergyHandle,
		.zeltnerLtcParamHandle = zeltnerTables.zeltnerLtcParamHandle,
		.seedSalt = seedSalt,
		.pad0 = 0u,
	};

	test::dispatch(
		pl, push, (width + 7u) / 8u, (height + 7u) / 8u, 1u,
		srat::slice_as_bytes(globalPC)
	);
	test::gpu_wait();

	auto const raw = test::readback<f32>(outBuf, 0u, 3u * pixelCount);

	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(modelsBuf);
	vkof::buffer_destroy(debugPcBuf);
	vkof::pipeline_destroy(pl);
	vkof::blas_destroy(blas);
	vkof::tlas_destroy(tlas);
	ponder::energy_tables_destroy(energyTables);
	ponder::zeltner_tables_destroy(zeltnerTables);
	mor::scene_gpu_materials_destroy(materialsOverride);
	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
	mor::sampler_cache_destroy();

	return { raw, width, height };
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("subsurface furnace: real scene through utilSubsurfaceWalk, no NaN, sane energy") {
	f32v3 const subsurfaceColor = { 0.6f, 0.5f, 0.4f };
	// smaller than the box's own extent so the walk takes several internal
	// bounces before reaching the boundary, rather than exiting on the
	// first traced segment every time
	f32 const subsurfaceRadius = 0.3f;
	f32 const envIntensity = 0.5f;

	auto const result = (
		render_subsurface_box(subsurfaceColor, subsurfaceRadius, 0.0f, 1u)
	);
	u32 const pixelCount = result.width * result.height;

	u32 nonFiniteCount = 0u;
	f32 maxValue = 0.0f;
	f64 sum = 0.0;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (!std::isfinite(result.rgb[i])) {
			nonFiniteCount++;
			continue;
		}
		maxValue = std::max(maxValue, result.rgb[i]);
		sum += result.rgb[i];
	}
	CAPTURE(nonFiniteCount);
	CAPTURE(maxValue);
	CHECK(nonFiniteCount == 0u);

	// canary against a silently-broken pipeline rendering nothing (same
	// convention as the disabled furnace_model_render suite): the furnace
	// background alone guarantees every pixel is at least envIntensity
	CHECK(maxValue > 0.5f * envIntensity);

	// no firefly blowup; a subsurface walk with nee at the exit vertex is
	// higher-variance than a plain diffuse bounce (multiple internal
	// scatter events before any light-carrying estimate lands), so this
	// ceiling is looser than a typical opaque-material furnace test, not
	// tightened further without real per-scene data
	f32 const tooBrightCeiling = 20.0f * envIntensity;
	u32 tooBrightCount = 0u;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (std::isfinite(result.rgb[i]) && result.rgb[i] > tooBrightCeiling) {
			tooBrightCount++;
		}
	}
	CAPTURE(tooBrightCount);
	CHECK(tooBrightCount == 0u);

	MESSAGE("subsurface furnace: mean=", sum / (3.0 * pixelCount));
}

TEST_CASE("subsurface furnace: absorptive (dark) color stays darker than reflective") {
	// two renders differing only in subsurfaceColor: a dark, mostly-
	// absorptive material should end up dimmer on average than a bright,
	// mostly-scattering one -- the one directional energy relationship
	// simple enough to assert without a reference render
	auto const dark = render_subsurface_box({ 0.05f, 0.05f, 0.05f }, 0.3f, 0.0f, 2u);
	auto const bright = render_subsurface_box({ 0.95f, 0.95f, 0.95f }, 0.3f, 0.0f, 3u);

	u32 const pixelCount = dark.width * dark.height;
	f64 darkSum = 0.0;
	f64 brightSum = 0.0;
	u32 darkNonFinite = 0u;
	u32 brightNonFinite = 0u;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (std::isfinite(dark.rgb[i])) { darkSum += dark.rgb[i]; }
		else { darkNonFinite++; }
		if (std::isfinite(bright.rgb[i])) { brightSum += bright.rgb[i]; }
		else { brightNonFinite++; }
	}
	CHECK(darkNonFinite == 0u);
	CHECK(brightNonFinite == 0u);
	CAPTURE(darkSum / (3.0 * pixelCount));
	CAPTURE(brightSum / (3.0 * pixelCount));
	CHECK(darkSum < brightSum);
}

TEST_CASE("subsurface furnace: small radius (many bounces) doesn't truncate the short-MFP channel to black") {
	// regression case for the reported bug: skSubsurfaceWalkMaxSteps used
	// to be a hard 256-step cap that behaved as a silent, biased
	// truncation (not an unbiased russian-roulette termination) -- with a
	// small radius relative to the object (here 2% of the unit box, the
	// same rough ratio the reported real scene used: 0.052 on a
	// head-sized object) combined with a skin-like radiusScale spread
	// (red's mean free path several times blue's), the short-mfp channel
	// (blue) needed enough bounces to reliably exceed 256 and got zeroed
	// regardless of its true albedo -- the only symptom-relief was
	// inverting the physically-correct ratio to keep blue under the cap.
	// this checks blue doesn't collapse to near-zero relative to red at
	// a small radius, now that russian roulette (not a hard step count)
	// bounds the walk
	auto const result = (
		render_subsurface_box(
			{ 0.6f, 0.5f, 0.4f }, 0.02f, 0.0f, 6u, { 1.0f, 0.35f, 0.15f }
		)
	);
	u32 const pixelCount = result.width * result.height;

	u32 nonFiniteCount = 0u;
	f64 redSum = 0.0, greenSum = 0.0, blueSum = 0.0;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f32 const r = result.rgb[i];
		f32 const g = result.rgb[pixelCount + i];
		f32 const b = result.rgb[2u * pixelCount + i];
		if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
			nonFiniteCount++;
			continue;
		}
		redSum += r;
		greenSum += g;
		blueSum += b;
	}
	CHECK(nonFiniteCount == 0u);
	CAPTURE(redSum / pixelCount);
	CAPTURE(greenSum / pixelCount);
	CAPTURE(blueSum / pixelCount);
	// blue must contribute *something* non-negligible relative to red --
	// a near-total truncation (the bug) would show up as blue at some
	// tiny fraction (<1%) of red, not as a merely-dimmer-but-present
	// channel
	CHECK(blueSum > 0.0);
	CHECK(blueSum > 0.01 * redSum);
	// the reverse direction matters equally: per-step hero resampling (a
	// later regression, caught separately) starved the *long*-mfp
	// channel (red) instead, since a walk needing many green/blue-scale
	// steps gives red comparatively few chances to be hero -- rendered
	// as "completely blue, red nearly absent" even in plain furnace
	// lighting with no nee at all. neither channel should be a tiny
	// fraction of the others in either direction
	CHECK(redSum > 0.0);
	CHECK(redSum > 0.01 * blueSum);
	CHECK(greenSum > 0.01 * redSum);
	CHECK(greenSum > 0.01 * blueSum);
}

TEST_CASE("subsurface furnace: extreme chromatic radiusScale stays finite, no blowup") {
	// regression case for the reported bug: a user typed radiusScale
	// (255, 70, 40) directly (ctrl+click text entry bypasses the
	// documented [0,1] slider range) and got a blown-out blue result --
	// traced to the old weighted-delta-tracking scheme's per-channel
	// weight compounding unboundedly across null-collision retries for
	// exactly this kind of strongly chromatic sigma_t. the single-sample
	// ratio-tracking replacement has no retry loop to compound across, so
	// this should render with the same finite/sane-energy guarantees as
	// every other case here even at this out-of-range input
	auto const result = (
		render_subsurface_box(
			{ 0.6f, 0.5f, 0.4f }, 0.3f, 0.0f, 5u, { 255.0f, 70.0f, 40.0f }
		)
	);
	u32 const pixelCount = result.width * result.height;

	u32 nonFiniteCount = 0u;
	f32 maxValue = 0.0f;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (!std::isfinite(result.rgb[i])) {
			nonFiniteCount++;
			continue;
		}
		maxValue = std::max(maxValue, result.rgb[i]);
	}
	CAPTURE(nonFiniteCount);
	CAPTURE(maxValue);
	CHECK(nonFiniteCount == 0u);
	// envIntensity is 0.5 inside render_subsurface_box; a blown-up channel
	// would blow well past any plausible multiple of that
	CHECK(maxValue < 50.0f * 0.5f);
}

struct SubsurfaceProbePush {
	u64 outVa;
	u32 width;
	u32 height;
	u32 fieldSelect;
};

TEST_CASE("DEBUG: probe material.subsurfaceColor at the hit point") {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);

	mor::GpuMaterials const materialsOverride = (
		mor::scene_gpu_materials_create(scene)
	);
	// subsurfaceColor, coatColor, and subsurfaceRadiusScale are all given
	// distinct, easily-identifiable values so the probe can tell which
	// field's bytes actually land where
	GpuMorMaterial probeMat = (
		make_subsurface_material(
			{ 0.6f, 0.5f, 0.4f }, 0.3f, 0.0f, { 0.3f, 0.5f, 0.7f }
		)
	);
	probeMat.coatColor = comp3({ 0.11f, 0.22f, 0.33f });
	// see the loop in the furnace helper above: index 0 is mor's reserved
	// default material and is not what the box's meshlet references
	u32 const probeMaterialCount = (
		mor::scene_gpu_materials_count(materialsOverride)
	);
	for (u32 i = 0u; i < probeMaterialCount; ++i) {
		mor::scene_material_override_scalars(materialsOverride, i, probeMat);
	}
	{
		// CPU-only sanity check, no GPU roundtrip: isolates whether the
		// override lands correctly in m->cpu before upload even happens
		GpuMorMaterial const cpuCheck = (
			mor::scene_gpu_materials_get(materialsOverride, 0u)
		);
		MESSAGE(
			"CPU-side subsurfaceColor after override = (",
			cpuCheck.subsurfaceColor.rgb.x, ",", cpuCheck.subsurfaceColor.rgb.y,
			",", cpuCheck.subsurfaceColor.rgb.z, ")"
		);
		MESSAGE(
			"CPU-side coatColor after override = (",
			cpuCheck.coatColor.rgb.x, ",", cpuCheck.coatColor.rgb.y,
			",", cpuCheck.coatColor.rgb.z, ")"
		);
		MESSAGE(
			"CPU-side subsurfaceRadiusScale after override = (",
			cpuCheck.subsurfaceRadiusScale.rgb.x, ",",
			cpuCheck.subsurfaceRadiusScale.rgb.y, ",",
			cpuCheck.subsurfaceRadiusScale.rgb.z, ")"
		);
	}
	mor::scene_gpu_materials_upload(materialsOverride);

	auto const blas = vkof::blas_create({
		.positionVa = bufs.positions,
		.vertexCount = bufs.vertexCount,
		.indexVa = bufs.flatIndices,
		.triangleCount = bufs.triangleCount,
		.isOpaque = mor::scene_is_fully_opaque(scene),
	});
	auto const tlas = vkof::tlas_create({ .maxInstances = 1u });
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

	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR,
	};
	auto const pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_material_probe.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 3u),
	});
	REQUIRE(pl.id != 0u);

	u32 const width = 64u;
	u32 const height = 64u;
	u32 const pixelCount = width * height;

	auto const modelsBuf = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(12u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GpuResolveModelIndirect const modelDesc = {
		.meshlets = bufs.meshlets,
		.materials = mor::scene_gpu_materials_va(materialsOverride),
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
	srat::CameraOrbit const cam = {
		.target = boundsCenter,
		.distance = boundsExtent * 2.2f,
		.azimuth = 0.6f,
		.elevation = 0.35f,
		.fovY = 0.6f,
		.aspect = static_cast<f32>(width) / static_cast<f32>(height),
		.near = boundsExtent * 0.01f,
		.far = boundsExtent * 100.0f,
	};

	auto const debugPcBuf = vkof::buffer_create({
		.byteCount = sizeof(GpuGlobalExtended),
		.memory = vkof::BufferMemory::HostWritable,
	});
	GpuGlobalExtended const extended {};
	vkof::buffer_upload({
		.buffer = debugPcBuf,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(extended),
	});

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

	char const * const fieldNames[3] = {
		"subsurfaceColor", "coatColor", "subsurfaceRadiusScale"
	};
	for (u32 fieldSelect = 0u; fieldSelect < 3u; ++fieldSelect) {
		SubsurfaceProbePush const push {
			.outVa = vkof::buffer_virtual_address(outBuf),
			.width = width,
			.height = height,
			.fieldSelect = fieldSelect,
		};

		test::dispatch(
			pl, push, (width + 7u) / 8u, (height + 7u) / 8u, 1u,
			srat::slice_as_bytes(globalPC)
		);
		test::gpu_wait();

		auto const raw = test::readback<f32>(outBuf, 0u, 12u * pixelCount);

		f32 rMin = 1e9f, rMax = -1e9f, gMin = 1e9f, gMax = -1e9f, bMin = 1e9f, bMax = -1e9f;
		f32 rawRMin = 1e9f, rawRMax = -1e9f, rawGMin = 1e9f, rawGMax = -1e9f,
			rawBMin = 1e9f, rawBMax = -1e9f;
		f32 directXMin = 1e9f, directXMax = -1e9f, directYMin = 1e9f, directYMax = -1e9f,
			directZMin = 1e9f, directZMax = -1e9f;
		f32 fwXMin = 1e9f, fwXMax = -1e9f, fwYMin = 1e9f, fwYMax = -1e9f,
			fwZMin = 1e9f, fwZMax = -1e9f;
		u32 hitCount = 0u;
		for (u32 i = 0u; i < pixelCount; ++i) {
			f32 const r = raw[i];
			f32 const g = raw[pixelCount + i];
			f32 const b = raw[2u * pixelCount + i];
			if (r < -0.5f) { continue; }
			hitCount++;
			rMin = std::min(rMin, r); rMax = std::max(rMax, r);
			gMin = std::min(gMin, g); gMax = std::max(gMax, g);
			bMin = std::min(bMin, b); bMax = std::max(bMax, b);
			f32 const rawR = raw[3u * pixelCount + i];
			f32 const rawG = raw[4u * pixelCount + i];
			f32 const rawB = raw[5u * pixelCount + i];
			rawRMin = std::min(rawRMin, rawR); rawRMax = std::max(rawRMax, rawR);
			rawGMin = std::min(rawGMin, rawG); rawGMax = std::max(rawGMax, rawG);
			rawBMin = std::min(rawBMin, rawB); rawBMax = std::max(rawBMax, rawB);
			f32 const directX = raw[6u * pixelCount + i];
			f32 const directY = raw[7u * pixelCount + i];
			f32 const directZ = raw[8u * pixelCount + i];
			directXMin = std::min(directXMin, directX); directXMax = std::max(directXMax, directX);
			directYMin = std::min(directYMin, directY); directYMax = std::max(directYMax, directY);
			directZMin = std::min(directZMin, directZ); directZMax = std::max(directZMax, directZ);
			f32 const fwX = raw[9u * pixelCount + i];
			f32 const fwY = raw[10u * pixelCount + i];
			f32 const fwZ = raw[11u * pixelCount + i];
			fwXMin = std::min(fwXMin, fwX); fwXMax = std::max(fwXMax, fwX);
			fwYMin = std::min(fwYMin, fwY); fwYMax = std::max(fwYMax, fwY);
			fwZMin = std::min(fwZMin, fwZ); fwZMax = std::max(fwZMax, fwZ);
		}
		MESSAGE("field=", fieldNames[fieldSelect], " hitCount=", hitCount, " of ", pixelCount);
		MESSAGE("  r range [", rMin, ",", rMax, "]");
		MESSAGE("  g range [", gMin, ",", gMax, "]");
		MESSAGE("  b range [", bMin, ",", bMax, "]");
		MESSAGE(
			"  RAW FLOAT subsurfaceColor r range [", rawRMin, ",", rawRMax, "]"
		);
		MESSAGE(
			"  RAW FLOAT subsurfaceColor g range [", rawGMin, ",", rawGMax, "]"
		);
		MESSAGE(
			"  RAW FLOAT subsurfaceColor b range [", rawBMin, ",", rawBMax, "]"
		);
		MESSAGE(
			"  DIRECT FIELD subsurfaceColor x range [", directXMin, ",", directXMax, "]"
		);
		MESSAGE(
			"  DIRECT FIELD subsurfaceColor y range [", directYMin, ",", directYMax, "]"
		);
		MESSAGE(
			"  DIRECT FIELD subsurfaceColor z range [", directZMin, ",", directZMax, "]"
		);
		MESSAGE(
			"  FIELDWISE-FN subsurfaceColor x range [", fwXMin, ",", fwXMax, "]"
		);
		MESSAGE(
			"  FIELDWISE-FN subsurfaceColor y range [", fwYMin, ",", fwYMax, "]"
		);
		MESSAGE(
			"  FIELDWISE-FN subsurfaceColor z range [", fwZMin, ",", fwZMax, "]"
		);
	}

	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(modelsBuf);
	vkof::buffer_destroy(debugPcBuf);
	vkof::pipeline_destroy(pl);
	vkof::blas_destroy(blas);
	vkof::tlas_destroy(tlas);
	mor::scene_gpu_materials_destroy(materialsOverride);
	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
	mor::sampler_cache_destroy();
}

void debug_color_ratio(f32v3 const subsurfaceColor, u32 const seedSalt) {
	auto const result = (
		render_subsurface_box(subsurfaceColor, 0.3f, 0.0f, seedSalt)
	);
	u32 const pixelCount = result.width * result.height;
	f64 rSum = 0.0, gSum = 0.0, bSum = 0.0;
	u32 nonFiniteCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f32 const r = result.rgb[i];
		f32 const g = result.rgb[pixelCount + i];
		f32 const b = result.rgb[2u * pixelCount + i];
		if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
			nonFiniteCount++;
			continue;
		}
		rSum += r;
		gSum += g;
		bSum += b;
	}
	CHECK(nonFiniteCount == 0u);
	MESSAGE(
		"color=(", subsurfaceColor.x, ",", subsurfaceColor.y, ",",
		subsurfaceColor.z, ") seedSalt=", seedSalt
	);
	MESSAGE("r=", rSum / pixelCount, " g=", gSum / pixelCount, " b=", bSum / pixelCount);
}

TEST_CASE("DEBUG: subsurface furnace color ratio vs target subsurfaceColor") {
	debug_color_ratio({ 0.6f, 0.5f, 0.4f }, 7u);
	debug_color_ratio({ 0.6f, 0.5f, 0.4f }, 42u);
	debug_color_ratio({ 0.4f, 0.6f, 0.4f }, 7u);
	debug_color_ratio({ 0.4f, 0.4f, 0.6f }, 7u);
	debug_color_ratio({ 0.2f, 0.5f, 0.8f }, 7u);
}

} // TEST_SUITE("[headless]")
