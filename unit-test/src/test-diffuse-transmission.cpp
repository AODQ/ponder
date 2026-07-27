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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// gltf KHR_materials_diffuse_transmission -> openpbr, checked against the
// khronos conformance asset. the thin case maps exactly: gltf's
// (1-f)*baseColor reflected / f*colorFactor transmitted is reproduced by
// subsurfaceWeight=f, subsurfaceColor=colorFactor and the sheet's g=1
// all-transmit split. the volume case (ScatteringSkull) lives in
// test-subsurface-banding-diagnostic.cpp

namespace {

struct MaterialExpectation {
	char const * name;
	f32 factor;
	f32v3 color;
	bool factorTextured;
	bool colorTextured;
};

// DiffuseTransmissionTest.gltf, all 20 materials carrying the extension
std::vector<MaterialExpectation> const skExpectations = {
	{ "Factor 0.0", 0.0f, { 1.0f, 1.0f, 1.0f }, false, false },
	{ "Factor 0.25", 0.25f, { 1.0f, 1.0f, 1.0f }, false, false },
	{ "Factor 0.5", 0.5f, { 1.0f, 1.0f, 1.0f }, false, false },
	{ "Factor 0.75", 0.75f, { 1.0f, 1.0f, 1.0f }, false, false },
	{ "Factor 1.0", 1.0f, { 1.0f, 1.0f, 1.0f }, false, false },
	{ "ColorFactor 0.0", 0.0f, { 1.0f, 0.0f, 0.0f }, false, false },
	{ "ColorFactor 0.25", 0.25f, { 1.0f, 0.0f, 0.0f }, false, false },
	{ "ColorFactor 0.5", 0.5f, { 1.0f, 0.0f, 0.0f }, false, false },
	{ "ColorFactor 0.75", 0.75f, { 1.0f, 0.0f, 0.0f }, false, false },
	{ "ColorFactor 1.0", 1.0f, { 1.0f, 0.0f, 0.0f }, false, false },
	{ "Texture 0.0", 0.0f, { 1.0f, 1.0f, 1.0f }, true, false },
	{ "Texture 0.25", 0.25f, { 1.0f, 1.0f, 1.0f }, true, false },
	{ "Texture 0.5", 0.5f, { 1.0f, 1.0f, 1.0f }, true, false },
	{ "Texture 0.75", 0.75f, { 1.0f, 1.0f, 1.0f }, true, false },
	{ "Texture 1.0", 1.0f, { 1.0f, 1.0f, 1.0f }, true, false },
	{ "ColorTexture 0.0", 0.0f, { 1.0f, 1.0f, 1.0f }, false, true },
	{ "ColorTexture 0.25", 0.25f, { 1.0f, 1.0f, 1.0f }, false, true },
	{ "ColorTexture 0.5", 0.5f, { 1.0f, 1.0f, 1.0f }, false, true },
	{ "ColorTexture 0.75", 0.75f, { 1.0f, 1.0f, 1.0f }, false, true },
	{ "ColorTexture 1.0", 1.0f, { 1.0f, 1.0f, 1.0f }, false, true },
};

struct DiffuseTransmissionPush {
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

// renders the conformance asset through the production path tracer.
// forceOpaque zeroes every subsurfaceWeight, which is the same scene with
// diffuse transmission switched off -- the two renders must differ
std::vector<f32> render_conformance_asset(bool const forceOpaque) {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/Models/DiffuseTransmissionTest/glTF/DiffuseTransmissionTest.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());
	mor::GpuScene gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);
	REQUIRE(bufs.triangleCount > 0u);

	mor::GpuMaterials const materials = mor::scene_gpu_materials_create(scene);
	if (forceOpaque) {
		u32 const count = mor::scene_gpu_materials_count(materials);
		for (u32 i = 0u; i < count; ++i) {
			GpuMorMaterial mat = mor::scene_gpu_materials_get(materials, i);
			mat.subsurfaceWeight.r = 0.0f;
			mat.subsurfaceWeight.texture = 0u;
			mor::scene_material_override_scalars(materials, i, mat);
		}
		mor::scene_gpu_materials_upload(materials);
	}

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

	u32 const width = 256u;
	u32 const height = 256u;
	u32 const pixelCount = width * height;

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
		.envMode = 3,
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
		.materials = mor::scene_gpu_materials_va(materials),
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
		.distance = boundsExtent * 1.4f,
		.azimuth = 0.0f,
		.elevation = 0.1f,
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
	DiffuseTransmissionPush const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = width,
		.height = height,
		.sampleCount = 64u,
		.propagationDepth = 8u,
		.kullaContyEnergyHandle = energyTables.kullaContyEnergyHandle,
		.zeltnerLtcParamHandle = zeltnerTables.zeltnerLtcParamHandle,
		.seedSalt = 1u,
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
	mor::scene_gpu_materials_destroy(materials);
	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
	mor::sampler_cache_destroy();
	return raw;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("gltf diffuse transmission: conformance asset maps onto the sheet") {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/Models/DiffuseTransmissionTest/glTF/DiffuseTransmissionTest.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());

	u32 const count = mor::scene_material_count(scene);
	u32 matched = 0u;
	for (MaterialExpectation const & want : skExpectations) {
		i32 index = -1;
		for (u32 i = 0u; i < count; ++i) {
			if (mor::scene_material_name(scene, i) == want.name) {
				index = (i32)i;
				break;
			}
		}
		CAPTURE(want.name);
		REQUIRE(index >= 0);
		++matched;
		GpuMorMaterial const m = mor::scene_material_get(scene, (u32)index);

		// diffuseTransmissionFactor -> subsurfaceWeight
		CHECK(m.subsurfaceWeight.r == doctest::Approx(want.factor));
		// diffuseTransmissionColorFactor -> subsurfaceColor
		CHECK(m.subsurfaceColor.rgb.x == doctest::Approx(want.color.x));
		CHECK(m.subsurfaceColor.rgb.y == doctest::Approx(want.color.y));
		CHECK(m.subsurfaceColor.rgb.z == doctest::Approx(want.color.z));
		// the sheet's all-transmit split, and no volume means thin-walled
		CHECK(m.subsurfaceScatterAnisotropy.r == doctest::Approx(1.0f));
		CHECK(m.geometryThinWalled.r == doctest::Approx(1.0f));

		// the factor texture stores strength in A, the color texture in rgb
		if (want.factorTextured) {
			CHECK(m.subsurfaceWeight.texture != 0u);
			CHECK(
				(m.subsurfaceWeight.swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK)
				== 3
			);
		} else {
			CHECK(m.subsurfaceWeight.texture == 0u);
		}
		if (want.colorTextured) {
			CHECK(m.subsurfaceColor.texture != 0u);
		} else {
			CHECK(m.subsurfaceColor.texture == 0u);
		}
	}
	CHECK(matched == skExpectations.size());

	mor::scene_destroy(scene);
}

TEST_CASE("gltf diffuse transmission: changes the render, finite everywhere") {
	auto const lit = render_conformance_asset(/*forceOpaque=*/false);
	auto const opaque = render_conformance_asset(/*forceOpaque=*/true);
	REQUIRE(lit.size() == opaque.size());

	u32 nonFinite = 0u;
	u32 negative = 0u;
	f64 litSum = 0.0;
	f64 absDiff = 0.0;
	for (size_t i = 0u; i < lit.size(); ++i) {
		if (!std::isfinite(lit[i]) || !std::isfinite(opaque[i])) {
			++nonFinite;
			continue;
		}
		if (lit[i] < 0.0f) {
			++negative;
		}
		litSum += (f64)lit[i];
		absDiff += std::fabs((f64)lit[i] - (f64)opaque[i]);
	}
	CAPTURE(litSum);
	CAPTURE(absDiff);
	CHECK(nonFinite == 0u);
	CHECK(negative == 0u);
	// the scene actually rendered something
	CHECK(litSum > 0.0);
	// switching diffuse transmission off must visibly change the image;
	// this is what fails if the extension never reaches the shader
	CHECK(absDiff / (f64)lit.size() > 1e-3);
}

} // TEST_SUITE("[headless]")

TEST_SUITE("[headless]") {

// gltf tints transmitted light by base color (specular_btdf * base_color),
// openpbr uses a separate transmission_color. without the bridge a
// transmissionFactor=1 shade renders as untinted white glass -- the
// LightsPunctualLamp newspaper shade is exactly that case
TEST_CASE("gltf transmission: base color tints transmission") {
	mor::Scene scene = mor::scene_create();
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/Models/LightsPunctualLamp/glTF/LightsPunctualLamp.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::scene_load_gltf(scene, modelPath.c_str());

	u32 const count = mor::scene_material_count(scene);
	u32 transmissive = 0u;
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial const m = mor::scene_material_get(scene, i);
		if (m.transmissionWeight.r <= 0.0f && m.transmissionWeight.texture == 0u) {
			continue;
		}
		++transmissive;
		CAPTURE(i);
		// no volume on this asset, so the tint stays the base color
		CHECK(m.transmissionDepth.r == doctest::Approx(0.0f));
		CHECK(m.transmissionColor.texture == m.baseColor.texture);
		CHECK(m.transmissionColor.uvTransform == m.baseColor.uvTransform);
		CHECK(m.transmissionColor.rgb.x == doctest::Approx(m.baseColor.rgb.x));
		CHECK(m.transmissionColor.rgb.y == doctest::Approx(m.baseColor.rgb.y));
		CHECK(m.transmissionColor.rgb.z == doctest::Approx(m.baseColor.rgb.z));
		// the shade's newspaper arrives through a texture, not a factor
		CHECK(m.transmissionColor.texture != 0u);
	}
	CHECK(transmissive > 0u);

	mor::scene_destroy(scene);
}

}
