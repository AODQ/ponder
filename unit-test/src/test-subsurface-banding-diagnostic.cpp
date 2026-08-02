#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <ponder/energy-tables.hpp>
#include <ponder/zeltner-tables.hpp>
#include <ponder/environment-tables.hpp>
#include "util.hpp"

#include "viewer/shaders/resolve_pc.h"

#include <cmath>
#include <vector>

// diagnostic-only, not a regression test: renders a curved subsurface mesh
// (Suzanne, real HDR environment lighting with nee) through the exact
// production path tracer, the same way test-subsurface-furnace.cpp's Box
// scene does, so the reported dark/bright banding on a real character face
// can be inspected from an actual pixel buffer instead of a screenshot.
// writes a tonemapped png to unit-test/output/ for visual inspection.

namespace {

GpuMorMaterialComponent1 comp1(f32 const v) {
	return { v, 0u, 0, 0u };
}
GpuMorMaterialComponent3 comp3(f32v3 const v) {
	return { v, 0u, 0u };
}

GpuMorMaterial make_skin_material(
	f32 const subsurfaceWeight, f32 const specularWeight = 1.0f
) {
	GpuMorMaterial mat;
	mat.baseWeight = comp1(1.0f);
	mat.baseColor = comp3({ 0.8f, 0.8f, 0.8f });
	mat.baseMetalness = comp1(0.0f);
	mat.baseDiffuseRoughness = comp1(0.0f);
	mat.specularWeight = comp1(specularWeight);
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
	mat.subsurfaceWeight = comp1(subsurfaceWeight);
	mat.subsurfaceColor = comp3({ 0.72f, 0.45f, 0.36f });
	mat.subsurfaceRadius = comp1(0.2f);
	mat.subsurfaceRadiusScale = comp3({ 1.0f, 1.0f, 1.0f });
	mat.subsurfaceScatterAnisotropy = comp1(0.0f);
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

RenderResult render_subsurface_suzanne(
	f32 const subsurfaceWeight, f32 const specularWeight = 1.0f,
	u32 const seedSalt = 1u
) {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR) + "/assets/models-categorized/test-core/Suzanne/Suzanne.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);
	REQUIRE(bufs.triangleCount > 0u);

	mor::GpuMaterials const materialsOverride = (
		mor::scene_gpu_materials_create(scene)
	);
	u32 const materialCount = mor::scene_gpu_materials_count(materialsOverride);
	REQUIRE(materialCount > 0u);
	// index 0 is mor's reserved default material, which no gltf-loaded
	// primitive references -- overriding only it silently does nothing
	for (u32 i = 0u; i < materialCount; ++i) {
		mor::scene_material_override_scalars(
			materialsOverride, i,
			make_skin_material(subsurfaceWeight, specularWeight)
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

	std::string const envPath = (
		std::string(REPO_DIR)
		+ "/assets/environments/brown_photostudio_02_1k.exr"
	);
	REQUIRE(std::filesystem::exists(envPath));
	auto envTables = ponder::environment_tables_create(envPath);

	struct GpuEnvironmentMapHandles {
		u32 radiance;
		u32 pdf;
		u32 aliasProb;
		u32 aliasIndex;
	};
	GpuEnvironmentMapHandles const envHandles = {
		.radiance = envTables.radianceHandle,
		.pdf = envTables.pdfHandle,
		.aliasProb = envTables.aliasProbHandle,
		.aliasIndex = envTables.aliasIndexHandle,
	};
	auto const envHandlesBuf = vkof::buffer_create({
		.byteCount = sizeof(GpuEnvironmentMapHandles),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = envHandlesBuf,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(envHandles),
	});

	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR,
	};
	auto const pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_furnace_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 3u),
	});
	REQUIRE(pl.id != 0u);

	u32 const width = 512u;
	u32 const height = 512u;
	u32 const pixelCount = width * height;
	u32 const sampleCount = 128u;

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
		.envIntensity = 4.0f,
		.renderWidth = width,
		.renderHeight = height,
		.envMode = 3, // ENV_MODE_HDRMAP, see environment.glsl
		.antialias = 0u,
		.frameMode = 0,
		.nanProbeCounterVa = 0u,
		.probePixel = { -1, -1 },
		.probeResult = 0u,
		.envMap = vkof::buffer_virtual_address(envHandlesBuf),
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
		.distance = boundsExtent * 1.6f,
		.azimuth = 0.25f,
		.elevation = 0.15f,
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
		.propagationDepth = 8u,
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
	vkof::buffer_destroy(envHandlesBuf);
	vkof::pipeline_destroy(pl);
	vkof::blas_destroy(blas);
	vkof::tlas_destroy(tlas);
	ponder::environment_tables_destroy(envTables);
	ponder::energy_tables_destroy(energyTables);
	ponder::zeltner_tables_destroy(zeltnerTables);
	mor::scene_gpu_materials_destroy(materialsOverride);
	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
	mor::sampler_cache_destroy();

	return { raw, width, height };
}

// reinhard tonemap so a real hdr env's highlights don't just clip to white
// in the diagnostic png
f32 tonemap(f32 const v) {
	f32 const reinhard = v / (1.0f + std::max(v, 0.0f));
	return std::pow(std::clamp(reinhard, 0.0f, 1.0f), 1.0f / 2.2f);
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("subsurface banding diagnostic: suzanne under real hdr env, weight 1.0") {
	auto const result = render_subsurface_suzanne(1.0f);
	u32 const pixelCount = result.width * result.height;
	std::vector<f32> r(pixelCount), g(pixelCount), b(pixelCount);
	u32 nonFiniteCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f32 const rv = result.rgb[i];
		f32 const gv = result.rgb[pixelCount + i];
		f32 const bv = result.rgb[2u * pixelCount + i];
		if (!std::isfinite(rv) || !std::isfinite(gv) || !std::isfinite(bv)) {
			nonFiniteCount++;
			r[i] = g[i] = b[i] = 0.0f;
			continue;
		}
		r[i] = tonemap(rv);
		g[i] = tonemap(gv);
		b[i] = tonemap(bv);
	}
	CHECK(nonFiniteCount == 0u);
	std::string const outPath = (
		std::string(REPO_DIR)
		+ "/unit-test/output/subsurface_banding_diagnostic_w1.png"
	);
	bool const wrote = (
		test::write_heatmap_png(
			r, g, b, result.width, result.height, outPath.c_str()
		)
	);
	CHECK(wrote);
	MESSAGE("wrote ", outPath);
}

TEST_CASE("subsurface banding diagnostic: suzanne under real hdr env, weight 0.6") {
	auto const result = render_subsurface_suzanne(0.6f);
	u32 const pixelCount = result.width * result.height;
	std::vector<f32> r(pixelCount), g(pixelCount), b(pixelCount);
	u32 nonFiniteCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f32 const rv = result.rgb[i];
		f32 const gv = result.rgb[pixelCount + i];
		f32 const bv = result.rgb[2u * pixelCount + i];
		if (!std::isfinite(rv) || !std::isfinite(gv) || !std::isfinite(bv)) {
			nonFiniteCount++;
			r[i] = g[i] = b[i] = 0.0f;
			continue;
		}
		r[i] = tonemap(rv);
		g[i] = tonemap(gv);
		b[i] = tonemap(bv);
	}
	CHECK(nonFiniteCount == 0u);
	std::string const outPath = (
		std::string(REPO_DIR)
		+ "/unit-test/output/subsurface_banding_diagnostic_w0.6.png"
	);
	bool const wrote = (
		test::write_heatmap_png(
			r, g, b, result.width, result.height, outPath.c_str()
		)
	);
	CHECK(wrote);
	MESSAGE("wrote ", outPath);
}

TEST_CASE("DEBUG: subsurface color isolation, specularWeight=0") {
	auto const result = render_subsurface_suzanne(1.0f, 0.0f);
	u32 const pixelCount = result.width * result.height;
	f64 rSum = 0.0, gSum = 0.0, bSum = 0.0;
	u32 nonFiniteCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f32 const rv = result.rgb[i];
		f32 const gv = result.rgb[pixelCount + i];
		f32 const bv = result.rgb[2u * pixelCount + i];
		if (!std::isfinite(rv) || !std::isfinite(gv) || !std::isfinite(bv)) {
			nonFiniteCount++;
			continue;
		}
		rSum += rv;
		gSum += gv;
		bSum += bv;
	}
	CHECK(nonFiniteCount == 0u);
	MESSAGE("r=", rSum / pixelCount, " g=", gSum / pixelCount, " b=", bSum / pixelCount);
	MESSAGE("r/b ratio=", rSum / bSum);
}

} // TEST_SUITE("[headless]")

TEST_SUITE("[headless]") {

// gltf diffuse transmission + volume must reach the walk's parameters: the
// scattering albedo as the observed diffuse reflectance (the walk derives
// absorption from it, so a white color renders colorless) and
// attenuationDistance as the mean free path
TEST_CASE("gltf diffuse transmission: volume attenuation maps onto subsurface") {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/models-categorized/test-transparency/ScatteringSkull.glb"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuMaterials const mats = mor::scene_gpu_materials_create(scene);

	// index 0 is mor's reserved default; the asset's single material is 1
	REQUIRE(mor::scene_gpu_materials_count(mats) == 2u);
	GpuMorMaterial const m = mor::scene_gpu_materials_get(mats, 1u);

	// diffuseTransmissionFactor 1, no diffuseTransmissionColorFactor (so
	// cgltf's 1,1,1 default). the color comes from volume_scatter's
	// multiscatterColorFactor, which is why the asset reads blue
	CHECK(m.subsurfaceWeight.r == doctest::Approx(1.0f));
	CHECK(m.subsurfaceColor.rgb.x == doctest::Approx(0.16827136f));
	CHECK(m.subsurfaceColor.rgb.y == doctest::Approx(0.52711987f));
	CHECK(m.subsurfaceColor.rgb.z == doctest::Approx(0.59062535f));
	// per-channel mfp = d / -log(attenuationColor), normalized so the
	// longest channel is 1 and the absolute length sits in the radius
	CHECK(m.subsurfaceRadius.r == doctest::Approx(0.023392631f));
	CHECK(m.subsurfaceRadiusScale.rgb.x == doctest::Approx(0.72209930f));
	CHECK(m.subsurfaceRadiusScale.rgb.y == doctest::Approx(1.0f));
	CHECK(m.subsurfaceRadiusScale.rgb.z == doctest::Approx(0.78039026f));
	// isotropic phase function: g=1 is the sheet's all-transmit split and
	// means something else entirely to the walk
	CHECK(m.subsurfaceScatterAnisotropy.r == doctest::Approx(0.0f));
	CHECK(m.geometryThinWalled.r == doctest::Approx(0.0f));

	mor::scene_gpu_materials_destroy(mats);
	mor::scene_destroy(scene);
}

}

TEST_SUITE("[headless]") {

// is the reported banding stochastic (variance, which dwivedi guiding would
// reduce) or structural (bias/geometry, which it would not)? render the same
// scene under several independent seeds: noise decorrelates across seeds,
// a structural artifact lands in the same pixels every time
TEST_CASE("subsurface banding: stochastic or structural") {
	std::vector<RenderResult> renders;
	for (u32 seed = 1u; seed <= 4u; ++seed) {
		renders.push_back(render_subsurface_suzanne(1.0f, 1.0f, seed));
	}
	REQUIRE(renders.size() == 4u);
	size_t const n = renders[0].rgb.size();

	// per-pixel mean and cross-seed relative deviation, over lit pixels only
	f64 meanRelDev = 0.0;
	f64 maxRelDev = 0.0;
	u32 counted = 0u;
	for (size_t i = 0u; i < n; ++i) {
		f64 mean = 0.0;
		for (auto const & r : renders) { mean += (f64)r.rgb[i]; }
		mean /= (f64)renders.size();
		if (mean < 1e-3) { continue; }
		f64 var = 0.0;
		for (auto const & r : renders) {
			f64 const d = (f64)r.rgb[i] - mean;
			var += d * d;
		}
		var /= (f64)renders.size();
		f64 const relDev = std::sqrt(var) / mean;
		meanRelDev += relDev;
		maxRelDev = std::max(maxRelDev, relDev);
		++counted;
	}
	REQUIRE(counted > 0u);
	meanRelDev /= (f64)counted;

	MESSAGE("lit samples = ", counted, " of ", n);
	MESSAGE("mean cross-seed relative deviation = ", meanRelDev);
	MESSAGE("max  cross-seed relative deviation = ", maxRelDev);
	// diagnostic only: the number is the finding, not a pass/fail bar
	CHECK(std::isfinite(meanRelDev));
}

}
