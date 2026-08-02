#include <doctest/doctest.h>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>

#include <cmath>
#include <filesystem>
#include <string>

// gltf KHR_materials_volume attenuation -> openpbr transmission_depth.
// pins each of the four authoring cases separately, because they map to
// three different renderer behaviors and only one of them is "beer-lambert
// with the authored distance"

namespace {

// the one material in the shipped assets that authors attenuationDistance
constexpr char const * skSunglassesPath = (
	"/assets/models-categorized/production-scene/SunglassesKhronos.glb"
);

}

TEST_SUITE("[headless]") {

TEST_CASE("gltf volume: authored attenuationDistance reaches transmissionDepth") {
	std::string const modelPath = std::string(REPO_DIR) + skSunglassesPath;
	REQUIRE(std::filesystem::exists(modelPath));
	mor::Scene scene = mor::scene_create();
	mor::scene_load_gltf(scene, modelPath.c_str());

	u32 const count = mor::scene_material_count(scene);
	// nose_pads: transmissionFactor 1, thicknessFactor 0.01,
	// attenuationDistance 0.007, attenuationColor white
	u32 withDepth = 0u;
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial const m = mor::scene_material_get(scene, i);
		if (m.transmissionDepth.r <= 0.0f) {
			continue;
		}
		++withDepth;
		CAPTURE(i);
		CHECK(m.transmissionDepth.r == doctest::Approx(0.007f));
		// a real interior is what makes the depth mean anything; the shader
		// only runs beer-lambert on a solid dielectric
		CHECK(m.geometryThinWalled.r == doctest::Approx(0.0f));
	}
	CHECK(withDepth == 1u);

	mor::scene_destroy(scene);
}

TEST_CASE("gltf volume: unauthored attenuationDistance stays a surface tint") {
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/models-categorized/production-scene/StainedGlassLamp"
		+ "/StainedGlassLamp.gltf"
	);
	REQUIRE(std::filesystem::exists(modelPath));
	mor::Scene scene = mor::scene_create();
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
		// no material in this asset authors attenuationDistance. cgltf leaves
		// it FLT_MAX, which as a beer-lambert length absorbs nothing and would
		// silently drop the tint from both the surface and the volume path
		CHECK(std::isfinite(m.transmissionDepth.r));
		CHECK(m.transmissionDepth.r == doctest::Approx(0.0f));
	}
	CHECK(transmissive > 0u);

	mor::scene_destroy(scene);
}

TEST_CASE("gltf volume: transmission without a volume is thin-walled") {
	std::string const modelPath = std::string(REPO_DIR) + skSunglassesPath;
	REQUIRE(std::filesystem::exists(modelPath));
	mor::Scene scene = mor::scene_create();
	mor::scene_load_gltf(scene, modelPath.c_str());

	u32 const count = mor::scene_material_count(scene);
	u32 thinTransmissive = 0u;
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial const m = mor::scene_material_get(scene, i);
		if (m.transmissionWeight.r <= 0.0f) {
			continue;
		}
		if (m.geometryThinWalled.r <= 0.0f) {
			continue;
		}
		++thinTransmissive;
		CAPTURE(i);
		// lens_interior/lens_exterior carry no KHR_materials_volume, so there
		// is no interior to attenuate through -- depth must stay 0 rather
		// than describe a medium the shader will never enter
		CHECK(m.transmissionDepth.r == doctest::Approx(0.0f));
	}
	// both lens materials
	CHECK(thinTransmissive == 2u);

	mor::scene_destroy(scene);
}

} // TEST_SUITE("[headless]")
