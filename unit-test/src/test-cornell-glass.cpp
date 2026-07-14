#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/energy-tables.hpp>
#include "util.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// three path-tracer validation scenes built on cornell_glass_render.comp,
// the medium-tracking sibling of cornell_render.comp's plain openpbr shader
// (see that shader's header comment for the single-flag insideMedium
// argument this suite depends on):
//
// - a solid dielectric sphere floating inside the air cavity of a hollow
//   glass shell, IOR swept per config
// - a hollow glass sphere sitting on a checkerboard floor against an open
//   background, IOR swept per config
// - a large flat plane tiled with a ~2500-sphere grid whose per-instance
//   height follows a radiating sine ripple, faking a field of balls frozen
//   mid-bounce
//
// each test writes one (or several) high-resolution pngs as the primary
// validation artifact; numeric checks are sanity guards (no NaNs, no
// energy blowup), not per-pixel oracles -- the same philosophy as
// "cornell: full render" and "cornell: openpbr material configs" in
// test-cornell.cpp.
// ---------------------------------------------------------------------------

namespace {

// mirrors InstanceRecord in cornell_glass_render.comp (scalar packing)
struct InstanceRecord {
	u64 vertexVa;
	u64 indexVa;
	f32 baseColor[3];
	f32 baseWeight;
	f32 baseMetalness;
	f32 baseDiffuseRoughness;
	f32 specularWeight;
	f32 specularRoughness;
	f32 specularColor[3];
	f32 specularRoughnessAnisotropy;
	f32 emission[3];
	f32 specularIor;
	f32 transmissionWeight;
	f32 transmissionColor[3];
};
static_assert(sizeof(InstanceRecord) == 96u);

InstanceRecord record_diffuse(
	u64 const vertexVa, u64 const indexVa,
	f32 const r, f32 const g, f32 const b,
	f32 const emission = 0.0f
) {
	return InstanceRecord {
		.vertexVa = vertexVa,
		.indexVa = indexVa,
		.baseColor = { r, g, b },
		.baseWeight = 1.0f,
		.baseMetalness = 0.0f,
		.baseDiffuseRoughness = 0.0f,
		.specularWeight = 0.0f,
		.specularRoughness = 0.3f,
		.specularColor = { 1.0f, 1.0f, 1.0f },
		.specularRoughnessAnisotropy = 0.0f,
		.emission = { emission, emission, emission },
		.specularIor = 1.5f,
		.transmissionWeight = 0.0f,
		.transmissionColor = { 1.0f, 1.0f, 1.0f },
	};
}

InstanceRecord record_glossy(
	u64 const vertexVa, u64 const indexVa,
	f32 const r, f32 const g, f32 const b,
	f32 const roughness, f32 const specularWeight
) {
	InstanceRecord rec = record_diffuse(vertexVa, indexVa, r, g, b);
	rec.specularWeight = specularWeight;
	rec.specularRoughness = roughness;
	return rec;
}

// deterministic per-cell pseudo-random value, used to assign sphere size
// tier and material kind across the bouncing-spheres grid without pulling
// in <random> for a fixed, reproducible scene
u32 hash2(u32 const a, u32 const b) {
	u32 h = a * 374761393u + b * 668265263u;
	h = (h ^ (h >> 13u)) * 1274126177u;
	return h ^ (h >> 16u);
}

f32 hash2_unit(u32 const a, u32 const b) {
	return static_cast<f32>(hash2(a, b)) / 4294967295.0f;
}

InstanceRecord record_varied(
	u64 const vertexVa, u64 const indexVa,
	f32 const r, f32 const g, f32 const b,
	u32 const kind
) {
	struct Kind { f32 roughness; f32 specularWeight; f32 metalness; };
	// polished/brushed metal, glossy/satin plastic, matte rubber, chalky
	// diffuse -- picked per instance so the field reads as a mixed bag of
	// materials rather than one lobe config recolored by height
	static Kind const kinds[6] = {
		{ 0.08f, 1.0f, 1.0f },
		{ 0.35f, 1.0f, 1.0f },
		{ 0.12f, 0.6f, 0.0f },
		{ 0.4f, 0.35f, 0.0f },
		{ 0.7f, 0.08f, 0.0f },
		{ 0.9f, 0.02f, 0.0f },
	};
	Kind const k = kinds[kind % 6u];
	InstanceRecord rec = record_diffuse(vertexVa, indexVa, r, g, b);
	rec.specularWeight = k.specularWeight;
	rec.specularRoughness = k.roughness;
	rec.baseMetalness = k.metalness;
	return rec;
}

InstanceRecord record_glass(
	u64 const vertexVa, u64 const indexVa,
	f32 const ior,
	f32 const tintR, f32 const tintG, f32 const tintB
) {
	return InstanceRecord {
		.vertexVa = vertexVa,
		.indexVa = indexVa,
		.baseColor = { 1.0f, 1.0f, 1.0f },
		.baseWeight = 1.0f,
		.baseMetalness = 0.0f,
		.baseDiffuseRoughness = 0.0f,
		.specularWeight = 1.0f,
		// not exactly 0: openPbrTransmissionEvaluateF's D/V terms use
		// mat.specularRoughness directly (unlike the reflect/sample paths,
		// which route through openPbrRoughnessAlpha's 1e-5 floor), so a
		// literal 0 evaluates a near-zero BTDF at the near-but-not-exactly
		// mirror-aligned half vector vndf sampling actually draws --
		// throughput collapses to ~0 and the glass renders solid black.
		// same reason the rest of this suite's mirror-like metals use 0.02
		// rather than 0 (see test-cornell.cpp's mirror-silver config)
		.specularRoughness = 0.02f,
		.specularColor = { 1.0f, 1.0f, 1.0f },
		.specularRoughnessAnisotropy = 0.0f,
		.emission = { 0.0f, 0.0f, 0.0f },
		.specularIor = ior,
		.transmissionWeight = 1.0f,
		.transmissionColor = { tintR, tintG, tintB },
	};
}

// mirrors the PC block in cornell_glass_render.comp
struct Push {
	u64 outVa;
	u64 sceneVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 kullaContyHandle;
	u32 bounceCount;
	f32 cameraOriginX;
	f32 cameraOriginY;
	f32 cameraOriginZ;
	f32 cameraPitch;
};

test::Mesh quad_mesh(
	test::Vertex const v0, test::Vertex const v1,
	test::Vertex const v2, test::Vertex const v3
) {
	return test::Mesh {
		{ v0, v1, v2, v3 },
		{ 0u, 1u, 2u, 0u, 2u, 3u },
	};
}

f32m44 translate(f32 const x, f32 const y, f32 const z) {
	return f32m44 {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		x, y, z, 1.0f,
	};
}
static f32m44 const skIdentity = translate(0.0f, 0.0f, 0.0f);

struct RenderConfig {
	char const * name;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 bounceCount;
	f32 cameraOrigin[3];
	f32 cameraPitch;
	// generous multiple of the scene's emitter Le: catches a genuine
	// energy-conservation blowup (e.g. a medium-tracking sign error
	// amplifying throughput every bounce) without flagging a legitimate
	// rare caustic-focused firefly pixel, which pure reflect/refract glass
	// can produce even in a physically correct renderer
	f32 energyCeiling;
};

void render_and_check(
	vkof::Pipeline const pl,
	std::vector<InstanceRecord> const & records,
	RenderConfig const & cfg,
	u32 const kullaContyHandle,
	std::string const & outputPath
) {
	CAPTURE(cfg.name);
	u32 const pixelCount = cfg.width * cfg.height;

	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(3u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto sceneBuf = vkof::buffer_create({
		.byteCount = records.size() * sizeof(InstanceRecord),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = sceneBuf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(records.data()),
			records.size() * sizeof(InstanceRecord)
		),
	});

	Push const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.sceneVa = vkof::buffer_virtual_address(sceneBuf),
		.width = cfg.width,
		.height = cfg.height,
		.sampleCount = cfg.sampleCount,
		.kullaContyHandle = kullaContyHandle,
		.bounceCount = cfg.bounceCount,
		.cameraOriginX = cfg.cameraOrigin[0],
		.cameraOriginY = cfg.cameraOrigin[1],
		.cameraOriginZ = cfg.cameraOrigin[2],
		.cameraPitch = cfg.cameraPitch,
	};
	test::dispatch(pl, push, cfg.width / 8u, cfg.height / 8u);
	test::gpu_wait();

	auto const raw = test::readback<f32>(outBuf, 0u, 3u * pixelCount);

	u32 nanCount = 0u;
	f32 minValue = 0.0f;
	f32 maxValue = 0.0f;
	f64 sum = 0.0;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (!std::isfinite(raw[i])) {
			nanCount++;
			continue;
		}
		minValue = std::min(minValue, raw[i]);
		maxValue = std::max(maxValue, raw[i]);
		sum += raw[i];
	}
	CHECK(nanCount == 0u);
	CHECK(minValue >= 0.0f);
	CHECK(maxValue < cfg.energyCeiling);
	f64 const mean = sum / (3.0 * pixelCount);
	CHECK(mean > 0.0005);

	std::vector<f32> pngR(pixelCount), pngG(pixelCount), pngB(pixelCount);
	for (u32 i = 0u; i < pixelCount; ++i) {
		pngR[i] = std::pow(std::clamp(raw[i], 0.0f, 1.0f), 1.0f / 2.2f);
		pngG[i] = std::pow(
			std::clamp(raw[pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
		);
		pngB[i] = std::pow(
			std::clamp(raw[2u * pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
		);
	}
	CHECK(test::write_heatmap_png(
		pngR, pngG, pngB, cfg.width, cfg.height, outputPath.c_str()
	));

	vkof::buffer_destroy(sceneBuf);
	vkof::buffer_destroy(outBuf);
}

vkof::Pipeline glass_pipeline_create() {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "cornell_glass_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);
	return pl;
}

}

TEST_SUITE("[headless]") {

// ---------------------------------------------------------------------------
// a solid dielectric sphere nested inside the air cavity of a hollow glass
// shell, inside the standard cornell box. every glass interface here has
// air (never another glass) on its far side, exactly the assumption
// cornell_glass_render.comp's single-flag insideMedium toggle depends on:
// outer shell outer surface (air -> shell glass), outer shell inner
// surface (shell glass -> cavity air), inner sphere surface twice (cavity
// air -> inner glass -> cavity air), then the shell's inner/outer surfaces
// again on the way back out
// ---------------------------------------------------------------------------

TEST_CASE("cornell: nested dielectric spheres") {
	// overlapped wall extent so adjacent walls share corners; see
	// test-cornell.cpp's header comment for why this matters (blas
	// watertightness is not guaranteed across blases)
	f32 const e = 1.05f;
	std::vector<test::Mesh> meshes {
		// floor (y=-1)
		quad_mesh(
			{ -e, -1.0f, -e }, { e, -1.0f, -e },
			{ e, -1.0f, e }, { -e, -1.0f, e }
		),
		// ceiling (y=+1)
		quad_mesh(
			{ -e, 1.0f, -e }, { -e, 1.0f, e },
			{ e, 1.0f, e }, { e, 1.0f, -e }
		),
		// back wall (z=+1)
		quad_mesh(
			{ -e, -e, 1.0f }, { e, -e, 1.0f },
			{ e, e, 1.0f }, { -e, e, 1.0f }
		),
		// left wall (x=-1), red
		quad_mesh(
			{ -1.0f, -e, -e }, { -1.0f, -e, e },
			{ -1.0f, e, e }, { -1.0f, e, -e }
		),
		// right wall (x=+1), green
		quad_mesh(
			{ 1.0f, -e, -e }, { 1.0f, e, -e },
			{ 1.0f, e, e }, { 1.0f, -e, e }
		),
		// area light, wound so the geometric normal points down (-y); see
		// test-cornell.cpp's identical light quad for why this matters
		quad_mesh(
			{ -0.25f, 0.998f, -0.25f }, { 0.25f, 0.998f, -0.25f },
			{ 0.25f, 0.998f, 0.25f }, { -0.25f, 0.998f, 0.25f }
		),
		// hollow shell, outer surface
		test::make_sphere_mesh(0.5f),
		// hollow shell, inner surface (the cavity the inner sphere floats in)
		test::make_sphere_mesh(0.42f),
		// solid inner sphere, floating in the shell's cavity
		test::make_sphere_mesh(0.15f),
	};
	// all three spheres share one center, on the floor
	f32 const sphereCenterY = -0.5f;
	std::vector<test::GeometryInstance> const instances {
		{ 0u, skIdentity, 0u },
		{ 1u, skIdentity, 1u },
		{ 2u, skIdentity, 2u },
		{ 3u, skIdentity, 3u },
		{ 4u, skIdentity, 4u },
		{ 5u, skIdentity, 5u },
		{ 6u, translate(0.0f, sphereCenterY, 0.0f), 6u },
		{ 7u, translate(0.0f, sphereCenterY, 0.0f), 7u },
		{ 8u, translate(0.0f, sphereCenterY, 0.0f), 8u },
	};
	auto const scene = test::geometry_scene_create(meshes, instances);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto pl = glass_pipeline_create();
	auto energyTables = ponder::energy_tables_create();
	f32 const skLe = 15.0f;

	struct Config { char const * name; f32 iorOuter; f32 iorInner; f32 tintB; };
	Config const configs[] = {
		// barely-there shell around a dense gem
		{ "thin-shell-dense-core", 1.05f, 2.4f, 1.0f },
		// dense shell around a barely-there core (core nearly disappears)
		{ "dense-shell-thin-core", 2.4f, 1.05f, 1.0f },
		// ordinary glass shell, water-like core
		{ "glass-shell-water-core", 1.5f, 1.33f, 1.0f },
		// matched glass IORs but the inner sphere is tinted blue, so the
		// two surfaces are only distinguishable by the tint, not by bend
		// angle -- a useful visual cross-check that transmissionColor
		// applies independently per instance
		{ "matched-ior-tinted-core", 1.5f, 1.5f, 0.6f },
	};

	for (auto const & config : configs) {
		std::vector<InstanceRecord> records {
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[0]),
				vkof::buffer_virtual_address(scene.indexBuffers[0]),
				0.73f, 0.73f, 0.73f
			),
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[1]),
				vkof::buffer_virtual_address(scene.indexBuffers[1]),
				0.73f, 0.73f, 0.73f
			),
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[2]),
				vkof::buffer_virtual_address(scene.indexBuffers[2]),
				0.73f, 0.73f, 0.73f
			),
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[3]),
				vkof::buffer_virtual_address(scene.indexBuffers[3]),
				0.63f, 0.065f, 0.05f
			),
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[4]),
				vkof::buffer_virtual_address(scene.indexBuffers[4]),
				0.14f, 0.45f, 0.091f
			),
			record_diffuse(
				vkof::buffer_virtual_address(scene.vertexBuffers[5]),
				vkof::buffer_virtual_address(scene.indexBuffers[5]),
				0.78f, 0.78f, 0.78f, skLe
			),
			record_glass(
				vkof::buffer_virtual_address(scene.vertexBuffers[6]),
				vkof::buffer_virtual_address(scene.indexBuffers[6]),
				config.iorOuter, 1.0f, 1.0f, 1.0f
			),
			record_glass(
				vkof::buffer_virtual_address(scene.vertexBuffers[7]),
				vkof::buffer_virtual_address(scene.indexBuffers[7]),
				config.iorOuter, 1.0f, 1.0f, 1.0f
			),
			record_glass(
				vkof::buffer_virtual_address(scene.vertexBuffers[8]),
				vkof::buffer_virtual_address(scene.indexBuffers[8]),
				config.iorInner, 1.0f, 1.0f, config.tintB
			),
		};

		RenderConfig const renderCfg {
			.name = config.name,
			.width = 3840u,
			.height = 2160u,
			// measured ~10.8s/config @ 256spp on this gpu; 128 lands each
			// config around ~5.4s
			.sampleCount = 128u,
			.bounceCount = 14u,
			.cameraOrigin = { 0.0f, 0.0f, -3.0f },
			.cameraPitch = 0.0f,
			.energyCeiling = 10.0f * skLe,
		};
		render_and_check(
			pl, records, renderCfg, energyTables.kullaContyEnergyHandle,
			std::string(CORNELL_OUTPUT_DIR "cornell-glass-nested-")
				+ config.name + ".png"
		);
	}

	vkof::pipeline_destroy(pl);
	ponder::energy_tables_destroy(energyTables);
	test::geometry_scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// a hollow glass sphere sitting on a checkerboard floor with an otherwise
// open (black) background: rays that miss the floor/sphere/light escape to
// infinity, so refraction through the shell reads clearly against the
// checker pattern behind it
// ---------------------------------------------------------------------------

TEST_CASE("cornell: hollow glass sphere over checkerboard") {
	f32 const tileHalf = 0.3f;
	f32 const tileSize = 2.0f * tileHalf;
	u32 const tilesPerSide = 14u;

	std::vector<test::Mesh> meshes {
		// checker tile, instanced across the floor grid below
		quad_mesh(
			{ -tileHalf, 0.0f, -tileHalf }, { tileHalf, 0.0f, -tileHalf },
			{ tileHalf, 0.0f, tileHalf }, { -tileHalf, 0.0f, tileHalf }
		),
		// area light above the sphere, facing down (-y)
		quad_mesh(
			{ -0.4f, 0.0f, -0.4f }, { 0.4f, 0.0f, -0.4f },
			{ 0.4f, 0.0f, 0.4f }, { -0.4f, 0.0f, 0.4f }
		),
		// hollow shell, outer surface
		test::make_sphere_mesh(0.4f),
		// hollow shell, inner surface
		test::make_sphere_mesh(0.32f),
	};

	std::vector<test::GeometryInstance> instances;
	std::vector<InstanceRecord> baseRecords;
	u32 customIndex = 0u;
	for (u32 ix = 0u; ix < tilesPerSide; ++ix) {
		for (u32 iz = 0u; iz < tilesPerSide; ++iz) {
			f32 const cx = (
				static_cast<f32>(ix) - 0.5f * (tilesPerSide - 1u)
			) * tileSize;
			f32 const cz = (
				static_cast<f32>(iz) - 0.5f * (tilesPerSide - 1u)
			) * tileSize;
			instances.push_back({
				0u, translate(cx, -1.0f, cz), customIndex
			});
			bool const isDark = ((ix + iz) & 1u) == 0u;
			f32 const shade = isDark ? 0.08f : 0.85f;
			baseRecords.push_back(record_diffuse(0u, 0u, shade, shade, shade));
			customIndex++;
		}
	}
	u32 const lightIndex = customIndex++;
	instances.push_back({ 1u, translate(0.0f, 2.5f, 0.0f), lightIndex });
	f32 const skLe = 30.0f;
	baseRecords.push_back(record_diffuse(0u, 0u, 1.0f, 1.0f, 1.0f, skLe));

	f32 const sphereCenterY = -0.6f;
	u32 const outerIndex = customIndex++;
	instances.push_back({
		2u, translate(0.0f, sphereCenterY, 0.0f), outerIndex
	});
	baseRecords.push_back(InstanceRecord {});
	u32 const innerIndex = customIndex++;
	instances.push_back({
		3u, translate(0.0f, sphereCenterY, 0.0f), innerIndex
	});
	baseRecords.push_back(InstanceRecord {});

	auto const scene = test::geometry_scene_create(meshes, instances);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	// fill in the per-instance geometry vas now that the scene buffers
	// exist; tile/light records were built before scene creation since
	// they all share the same two meshes (index 0 and 1)
	for (u32 i = 0u; i < instances.size(); ++i) {
		baseRecords[i].vertexVa = vkof::buffer_virtual_address(
			scene.vertexBuffers[instances[i].meshIndex]
		);
		baseRecords[i].indexVa = vkof::buffer_virtual_address(
			scene.indexBuffers[instances[i].meshIndex]
		);
	}

	auto pl = glass_pipeline_create();
	auto energyTables = ponder::energy_tables_create();

	struct Config { char const * name; f32 ior; };
	Config const configs[] = {
		{ "ior-thin-1.05", 1.05f },
		{ "ior-water-1.33", 1.33f },
		{ "ior-glass-1.5", 1.5f },
		{ "ior-diamond-2.4", 2.4f },
	};

	for (auto const & config : configs) {
		std::vector<InstanceRecord> records = baseRecords;
		records[outerIndex] = record_glass(
			vkof::buffer_virtual_address(scene.vertexBuffers[2]),
			vkof::buffer_virtual_address(scene.indexBuffers[2]),
			config.ior, 1.0f, 1.0f, 1.0f
		);
		records[innerIndex] = record_glass(
			vkof::buffer_virtual_address(scene.vertexBuffers[3]),
			vkof::buffer_virtual_address(scene.indexBuffers[3]),
			config.ior, 1.0f, 1.0f, 1.0f
		);

		RenderConfig const renderCfg {
			.name = config.name,
			.width = 3840u,
			.height = 2160u,
			// measured ~2.7s/config @ 256spp on this gpu; 512 lands each
			// config around ~5.5s
			.sampleCount = 512u,
			.bounceCount = 12u,
			.cameraOrigin = { 0.0f, 1.0f, -2.6f },
			.cameraPitch = 0.35f,
			.energyCeiling = 10.0f * skLe,
		};
		render_and_check(
			pl, records, renderCfg, energyTables.kullaContyEnergyHandle,
			std::string(CORNELL_OUTPUT_DIR "cornell-glass-checker-")
				+ config.name + ".png"
		);
	}

	vkof::pipeline_destroy(pl);
	ponder::energy_tables_destroy(energyTables);
	test::geometry_scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// a flat plane tiled with a ~2500-sphere grid; each instance's height
// follows a radiating sine ripple (a fixed snapshot, not an actual
// simulation) so the field reads as balls frozen mid-bounce at varying
// heights. purely an opaque-material scene -- exercises tlas instancing at
// scale, not the medium-tracking transmission path the other two tests
// target
// ---------------------------------------------------------------------------

TEST_CASE("cornell: plane of bouncing spheres") {
	f32 const floorHalf = 25.0f;
	// four size tiers, cumulative-weighted so big spheres are rare and
	// small ones dominate ("big ones and lots of small ones"); spacing
	// below is chosen against the largest tier so no two adjacent
	// instances can ever touch, regardless of which tiers land next to
	// each other
	f32 const sizeTiers[4] = { 0.16f, 0.10f, 0.065f, 0.035f };
	f32 const sizeCumulativeWeight[4] = { 0.04f, 0.16f, 0.44f, 1.0f };
	u32 const gridSideX = 50u;
	u32 const gridSideZ = 50u;
	f32 const spacing = 0.42f;
	// z starts just in front of the camera and recedes away from it,
	// rather than a grid centered on the camera (which would place
	// instances behind/inside the near clip); x stays centered for a wide
	// field
	f32 const zNear = 1.5f;
	f32 const rippleFreq = 2.2f;
	f32 const rippleAmplitude = 0.6f;

	std::vector<test::Mesh> meshes {
		// floor
		quad_mesh(
			{ -floorHalf, -1.0f, -floorHalf }, { floorHalf, -1.0f, -floorHalf },
			{ floorHalf, -1.0f, floorHalf }, { -floorHalf, -1.0f, floorHalf }
		),
		// area light, facing down (-y)
		quad_mesh(
			{ -1.2f, 0.0f, -1.2f }, { 1.2f, 0.0f, -1.2f },
			{ 1.2f, 0.0f, 1.2f }, { -1.2f, 0.0f, 1.2f }
		),
		// one shared sphere mesh per size tier, each instanced many times
		// across the grid below (radius is baked into the mesh, never a
		// tlas scale; see make_sphere_mesh's header comment)
		test::make_sphere_mesh(sizeTiers[0]),
		test::make_sphere_mesh(sizeTiers[1]),
		test::make_sphere_mesh(sizeTiers[2]),
		test::make_sphere_mesh(sizeTiers[3]),
	};

	std::vector<test::GeometryInstance> instances;
	std::vector<InstanceRecord> baseRecords;
	u32 customIndex = 0u;

	instances.push_back({ 0u, skIdentity, customIndex });
	baseRecords.push_back(record_diffuse(0u, 0u, 0.5f, 0.5f, 0.5f));
	customIndex++;

	u32 const lightIndex = customIndex++;
	f32 const skLe = 45.0f;
	instances.push_back({ 1u, translate(0.0f, 4.5f, 0.0f), lightIndex });
	baseRecords.push_back(record_diffuse(0u, 0u, 1.0f, 1.0f, 1.0f, skLe));

	// low balls read blue-ish (resting), high balls read orange (airborne)
	f32 const lowColor[3] = { 0.15f, 0.35f, 0.85f };
	f32 const highColor[3] = { 0.95f, 0.55f, 0.15f };
	for (u32 ix = 0u; ix < gridSideX; ++ix) {
		for (u32 iz = 0u; iz < gridSideZ; ++iz) {
			f32 const x = (
				static_cast<f32>(ix) - 0.5f * (gridSideX - 1u)
			) * spacing;
			f32 const z = zNear + static_cast<f32>(iz) * spacing;

			f32 const sizeP = hash2_unit(ix, iz);
			u32 tier = 3u;
			for (u32 t = 0u; t < 4u; ++t) {
				if (sizeP < sizeCumulativeWeight[t]) { tier = t; break; }
			}
			f32 const radius = sizeTiers[tier];
			u32 const meshIndex = 2u + tier;

			f32 const dist = std::sqrt(x * x + z * z);
			f32 const heightFrac = std::abs(std::sin(dist * rippleFreq));
			f32 const y = -1.0f + radius + rippleAmplitude * heightFrac;
			instances.push_back({
				meshIndex, translate(x, y, z), customIndex
			});

			f32 const rr = lowColor[0] + (highColor[0] - lowColor[0]) * heightFrac;
			f32 const gg = lowColor[1] + (highColor[1] - lowColor[1]) * heightFrac;
			f32 const bb = lowColor[2] + (highColor[2] - lowColor[2]) * heightFrac;
			u32 const kind = hash2(ix + 7919u, iz + 104729u) % 6u;
			baseRecords.push_back(
				record_varied(0u, 0u, rr, gg, bb, kind)
			);
			customIndex++;
		}
	}

	auto const scene = test::geometry_scene_create(meshes, instances);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	std::vector<InstanceRecord> records = baseRecords;
	for (u32 i = 0u; i < instances.size(); ++i) {
		records[i].vertexVa = vkof::buffer_virtual_address(
			scene.vertexBuffers[instances[i].meshIndex]
		);
		records[i].indexVa = vkof::buffer_virtual_address(
			scene.indexBuffers[instances[i].meshIndex]
		);
	}

	auto pl = glass_pipeline_create();
	auto energyTables = ponder::energy_tables_create();

	// low and close to the floor, grazing down the field so the nearest
	// (largest-reading) spheres sit close in frame and the grid recedes
	// to a horizon, rather than a top-down view that reads every instance
	// as similarly tiny
	RenderConfig const renderCfg {
		.name = "bouncing-spheres",
		.width = 3840u,
		.height = 2160u,
		// measured ~3.7s @ 512spp on this gpu; 640 lands around ~4.6s
		.sampleCount = 640u,
		.bounceCount = 6u,
		.cameraOrigin = { 0.0f, 0.05f, -2.6f },
		.cameraPitch = 0.11f,
		.energyCeiling = 1.5f * skLe,
	};
	render_and_check(
		pl, records, renderCfg, energyTables.kullaContyEnergyHandle,
		CORNELL_OUTPUT_DIR "cornell-bouncing-spheres.png"
	);

	vkof::pipeline_destroy(pl);
	ponder::energy_tables_destroy(energyTables);
	test::geometry_scene_destroy(scene);
}

}
