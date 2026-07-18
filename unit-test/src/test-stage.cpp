#include <doctest/doctest.h>

#include <stage/stage.hpp>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <ponder/vdb.hpp>

#include <filesystem>
#include <fstream>
#include <string>

// ---------------------------------------------------------------------------
// stage: json scene-file load/save round trips, real gltf/vdb instance
// loading through mor/ponder, nested composition, and a repeated
// load/destroy stress case
// ---------------------------------------------------------------------------

namespace {

// unique-per-test-case scratch dir, wiped clean before use so a previous
// failed run's leftovers can't leak into this one
std::filesystem::path stage_test_dir(char const * const name) {
	std::filesystem::path const dir = (
		std::filesystem::temp_directory_path() / "ponder-stage-test" / name
	);
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);
	std::filesystem::create_directories(dir);
	return dir;
}

void write_file(
	std::filesystem::path const & path, std::string const & text
) {
	std::ofstream out(path, std::ios::binary);
	out << text;
}

// mor reserves material index 0 for its own default material (see
// lib/mor/src/mor.cpp: "index 0 is always the default material"), so an
// authored material's index depends on load order rather than being 0 --
// look it up by name the same way stage::stage_gltf_instance_apply_overrides
// does, instead of assuming a fixed index
u32 find_material_index(mor::Scene const & scene, std::string const & name) {
	u32 const count = mor::scene_material_count(scene);
	for (u32 i = 0u; i < count; ++i) {
		if (mor::scene_material_name(scene, i) == name) { return i; }
	}
	FAIL("material '", name, "' not found");
	return 0u;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("stage: minimal scene without instances parses cameras/lights/env") {
	std::filesystem::path const dir = stage_test_dir("minimal");
	std::string const json = (
		"{"
		"\"version\":1,"
		"\"name\":\"minimal\","
		"\"metersPerUnit\":2.0,"
		"\"activeCamera\":\"main\","
		"\"cameras\":[{"
		"\"name\":\"main\","
		"\"position\":[1.0,2.0,3.0],"
		"\"fovYDegrees\":60.0"
		"}],"
		"\"environment\":{"
		"\"mode\":\"checkerboard\","
		"\"intensity\":1.5"
		"},"
		"\"lights\":[{"
		"\"name\":\"key\","
		"\"radiance\":[2.0,3.0,4.0],"
		"\"twoSided\":true"
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	stage::Stage const s = (
		stage::stage_load((dir / "scene.json").string().c_str())
	);
	REQUIRE(!s.sourceDirectory.empty());
	CHECK(s.name == "minimal");
	CHECK(s.metersPerUnit == doctest::Approx(2.0f));
	CHECK(s.activeCamera == "main");

	REQUIRE(s.cameras.size() == 1u);
	CHECK(s.cameras[0].position.x == doctest::Approx(1.0f));
	CHECK(s.cameras[0].position.y == doctest::Approx(2.0f));
	CHECK(s.cameras[0].position.z == doctest::Approx(3.0f));
	CHECK(s.cameras[0].fovYDegrees == doctest::Approx(60.0f));

	CHECK(s.environment.mode == stage::EnvironmentMode::Checkerboard);
	CHECK(s.environment.intensity == doctest::Approx(1.5f));

	REQUIRE(s.lights.size() == 1u);
	CHECK(s.lights[0].radiance.y == doctest::Approx(3.0f));
	CHECK(s.lights[0].twoSided == true);

	CHECK(s.gltfInstances.empty());
	CHECK(s.vdbInstances.empty());
}

TEST_CASE("stage: gltf instance loads a real mor::Scene and applies overrides") {
	std::filesystem::path const dir = stage_test_dir("gltf-instance");
	std::string const gltfPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);
	REQUIRE(std::filesystem::exists(gltfPath));

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"instances\":[{"
		"\"type\":\"gltf\","
		"\"name\":\"box\","
		"\"path\":\"" + gltfPath + "\","
		"\"materialOverrides\":[{"
		"\"material\":\"Red\","
		"\"fields\":{"
		"\"baseColor\":{\"value\":[0.0,1.0,0.0]},"
		"\"baseMetalness\":{\"value\":0.5}"
		"}"
		"}]"
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	stage::Stage s = stage::stage_load((dir / "scene.json").string().c_str());
	REQUIRE(!s.sourceDirectory.empty());
	REQUIRE(s.gltfInstances.size() == 1u);

	stage::GltfInstance const & inst = s.gltfInstances[0];
	CHECK(inst.scene.id != 0u);
	CHECK(inst.gpuScene.id != 0u);
	CHECK(inst.materials.id != 0u);
	// index 0 is mor's reserved default material, so Box.gltf's one
	// authored "Red" material lands at index 1
	REQUIRE(mor::scene_material_count(inst.scene) == 2u);
	u32 const redIndex = find_material_index(inst.scene, "Red");

	GpuMorMaterial const mat = (
		mor::scene_gpu_materials_get(inst.materials, redIndex)
	);
	CHECK(mat.baseColor.rgb.x == doctest::Approx(0.0f));
	CHECK(mat.baseColor.rgb.y == doctest::Approx(1.0f));
	CHECK(mat.baseColor.rgb.z == doctest::Approx(0.0f));
	CHECK(mat.baseMetalness.r == doctest::Approx(0.5f));

	stage::stage_destroy(s);
}

TEST_CASE("stage: vdb instance loads a real ponder::Vdb per blob with sigma") {
	ponder::vdb_initialize();
	std::filesystem::path const dir = stage_test_dir("vdb-instance");
	std::string const vdbPath = (
		std::string(REPO_DIR) + "/assets/vdb/bunny_cloud.vdb"
	);
	REQUIRE(std::filesystem::exists(vdbPath));

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"instances\":[{"
		"\"type\":\"vdb\","
		"\"name\":\"cloud\","
		"\"path\":\"" + vdbPath + "\","
		"\"blobs\":[{"
		"\"grid\":\"density\","
		"\"albedo\":[0.1,0.2,0.3],"
		"\"sigmaScale\":2.5,"
		"\"dropletDiameter\":30.0"
		"}]"
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	stage::Stage s = stage::stage_load((dir / "scene.json").string().c_str());
	REQUIRE(!s.sourceDirectory.empty());
	REQUIRE(s.vdbInstances.size() == 1u);
	REQUIRE(s.vdbInstances[0].blobs.size() == 1u);

	stage::VdbBlob const & blob = s.vdbInstances[0].blobs[0];
	CHECK(blob.gridName == "density");
	CHECK(blob.vdb.gpuBuffer.id != 0u);
	CHECK(blob.vdb.handleBuffer.id != 0u);
	CHECK(blob.vdb.densityMax > 0.0f);
	CHECK(blob.albedo.x == doctest::Approx(0.1f));
	CHECK(blob.sigmaScale == doctest::Approx(2.5f));
	CHECK(blob.dropletDiameter == doctest::Approx(30.0f));

	stage::stage_destroy(s);
}

TEST_CASE("stage: save then reload round-trips fields and rebased asset paths") {
	ponder::vdb_initialize();
	std::filesystem::path const srcDir = stage_test_dir("roundtrip-src");
	std::filesystem::path const dstDir = stage_test_dir("roundtrip-dst");
	std::string const gltfPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);
	std::string const vdbPath = (
		std::string(REPO_DIR) + "/assets/vdb/smoke.vdb"
	);
	REQUIRE(std::filesystem::exists(gltfPath));
	REQUIRE(std::filesystem::exists(vdbPath));

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"name\":\"roundtrip\","
		"\"metersPerUnit\":0.5,"
		"\"activeCamera\":\"main\","
		"\"cameras\":[{"
		"\"name\":\"main\","
		"\"position\":[1.0,2.0,3.0],"
		"\"fovYDegrees\":50.0"
		"}],"
		"\"environment\":{"
		"\"mode\":\"hdri\","
		"\"path\":\"env.exr\","
		"\"rotationRadians\":0.5,"
		"\"intensity\":2.0"
		"},"
		"\"lights\":[{"
		"\"name\":\"key\","
		"\"radiance\":[5.0,4.0,3.0],"
		"\"width\":2.0,"
		"\"height\":1.5,"
		"\"twoSided\":true"
		"}],"
		"\"instances\":[{"
		"\"type\":\"gltf\","
		"\"name\":\"box\","
		"\"path\":\"" + gltfPath + "\","
		"\"transform\":{\"position\":[1.0,0.0,0.0]},"
		"\"materialOverrides\":[{"
		"\"material\":\"Red\","
		"\"fields\":{\"baseMetalness\":{\"value\":0.75}}"
		"}]"
		"},{"
		"\"type\":\"vdb\","
		"\"name\":\"vol\","
		"\"path\":\"" + vdbPath + "\","
		"\"blobs\":[{"
		"\"grid\":\"density\","
		"\"albedo\":[0.4,0.5,0.6],"
		"\"sigmaScale\":3.0"
		"}]"
		"}]"
		"}"
	);
	write_file(srcDir / "scene.json", json);

	stage::Stage a = stage::stage_load((srcDir / "scene.json").string().c_str());
	REQUIRE(!a.sourceDirectory.empty());
	REQUIRE(a.gltfInstances.size() == 1u);
	REQUIRE(a.vdbInstances.size() == 1u);
	REQUIRE(stage::stage_save(a, (dstDir / "reloaded.json").string().c_str()));

	stage::Stage b = (
		stage::stage_load((dstDir / "reloaded.json").string().c_str())
	);
	REQUIRE(!b.sourceDirectory.empty());

	CHECK(b.name == "roundtrip");
	CHECK(b.metersPerUnit == doctest::Approx(0.5f));
	CHECK(b.activeCamera == "main");
	REQUIRE(b.cameras.size() == 1u);
	CHECK(b.cameras[0].position.x == doctest::Approx(1.0f));
	CHECK(b.cameras[0].position.y == doctest::Approx(2.0f));
	CHECK(b.cameras[0].position.z == doctest::Approx(3.0f));
	CHECK(b.cameras[0].fovYDegrees == doctest::Approx(50.0f));

	CHECK(b.environment.mode == stage::EnvironmentMode::Hdri);
	CHECK(b.environment.rotationRadians == doctest::Approx(0.5f));
	CHECK(b.environment.intensity == doctest::Approx(2.0f));

	REQUIRE(b.lights.size() == 1u);
	CHECK(b.lights[0].radiance.x == doctest::Approx(5.0f));
	CHECK(b.lights[0].width == doctest::Approx(2.0f));
	CHECK(b.lights[0].twoSided == true);

	REQUIRE(b.gltfInstances.size() == 1u);
	// path was resolved absolute in a, rebased relative to dstDir on save,
	// then re-resolved absolute again on reload -- should land on the same
	// file regardless of which directory it's expressed relative to
	CHECK(std::filesystem::equivalent(b.gltfInstances[0].path, gltfPath));
	CHECK(b.gltfInstances[0].transform.position.x == doctest::Approx(1.0f));
	REQUIRE(b.gltfInstances[0].materialOverrides.size() == 1u);
	REQUIRE(b.gltfInstances[0].materialOverrides[0].fields.size() == 1u);
	CHECK(
		b.gltfInstances[0].materialOverrides[0].fields[0].scalarValue
		== doctest::Approx(0.75f)
	);
	// the reload re-ran mor::scene_load_gltf and re-applied the override --
	// confirm it actually took on the freshly loaded material, not just that
	// the descriptor round-tripped
	u32 const redIndex = (
		find_material_index(b.gltfInstances[0].scene, "Red")
	);
	GpuMorMaterial const mat = (
		mor::scene_gpu_materials_get(b.gltfInstances[0].materials, redIndex)
	);
	CHECK(mat.baseMetalness.r == doctest::Approx(0.75f));

	REQUIRE(b.vdbInstances.size() == 1u);
	CHECK(std::filesystem::equivalent(b.vdbInstances[0].path, vdbPath));
	REQUIRE(b.vdbInstances[0].blobs.size() == 1u);
	CHECK(b.vdbInstances[0].blobs[0].albedo.x == doctest::Approx(0.4f));
	CHECK(b.vdbInstances[0].blobs[0].sigmaScale == doctest::Approx(3.0f));

	stage::stage_destroy(a);
	stage::stage_destroy(b);
}

TEST_CASE("stage: nested stage instance composes a child scene") {
	std::filesystem::path const dir = stage_test_dir("nested");

	std::string const childJson = (
		"{"
		"\"version\":1,"
		"\"name\":\"child\","
		"\"lights\":[{\"name\":\"childLight\",\"radiance\":[9.0,8.0,7.0]}]"
		"}"
	);
	write_file(dir / "child.json", childJson);

	std::string const parentJson = (
		"{"
		"\"version\":1,"
		"\"name\":\"parent\","
		"\"instances\":[{"
		"\"type\":\"stage\","
		"\"name\":\"childRef\","
		"\"path\":\"child.json\","
		"\"transform\":{\"position\":[10.0,0.0,0.0]}"
		"}]"
		"}"
	);
	write_file(dir / "parent.json", parentJson);

	stage::Stage parent = (
		stage::stage_load((dir / "parent.json").string().c_str())
	);
	REQUIRE(!parent.sourceDirectory.empty());
	REQUIRE(parent.stageInstances.size() == 1u);

	{
		stage::StageInstance const & ref = parent.stageInstances[0];
		REQUIRE(ref.child != nullptr);
		CHECK(ref.child->name == "child");
		REQUIRE(ref.child->lights.size() == 1u);
		CHECK(ref.child->lights[0].radiance.x == doctest::Approx(9.0f));
		CHECK(ref.transform.position.x == doctest::Approx(10.0f));
	}

	stage::stage_destroy(parent);
}

TEST_CASE("stage: load failures return the documented empty-sourceDirectory sentinel") {
	std::filesystem::path const dir = stage_test_dir("failures");

	SUBCASE("nonexistent file") {
		stage::Stage const s = stage::stage_load(
			(dir / "does-not-exist.json").string().c_str()
		);
		CHECK(s.sourceDirectory.empty());
	}

	SUBCASE("malformed json") {
		write_file(dir / "malformed.json", "{ not json ");
		stage::Stage const s = stage::stage_load(
			(dir / "malformed.json").string().c_str()
		);
		CHECK(s.sourceDirectory.empty());
	}

	SUBCASE("version newer than supported") {
		write_file(dir / "future.json", "{\"version\":999999}");
		stage::Stage const s = stage::stage_load(
			(dir / "future.json").string().c_str()
		);
		CHECK(s.sourceDirectory.empty());
	}
}

TEST_CASE("stage: unknown instance type/material/field are warned, not fatal") {
	std::filesystem::path const dir = stage_test_dir("unknowns");
	std::string const gltfPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"instances\":[{"
		"\"type\":\"banana\","
		"\"name\":\"nope\""
		"},{"
		"\"type\":\"gltf\","
		"\"name\":\"box\","
		"\"path\":\"" + gltfPath + "\","
		"\"materialOverrides\":[{"
		"\"material\":\"NoSuchMaterial\","
		"\"fields\":{\"notARealField\":{\"value\":1.0}}"
		"},{"
		"\"material\":\"Red\","
		"\"fields\":{"
		"\"notARealField\":{\"value\":1.0},"
		"\"baseWeight\":{\"value\":0.5}"
		"}"
		"}]"
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	stage::Stage s = stage::stage_load((dir / "scene.json").string().c_str());
	REQUIRE(!s.sourceDirectory.empty());
	// the unknown "banana" type is skipped entirely, never stored
	CHECK(s.gltfInstances.size() == 1u);
	CHECK(s.vdbInstances.empty());
	CHECK(s.alembicInstances.empty());
	CHECK(s.stageInstances.empty());

	// the recognized field on the recognized material still applied despite
	// its sibling override (unknown material) and sibling field (unknown
	// field) both failing to resolve
	u32 const redIndex = find_material_index(s.gltfInstances[0].scene, "Red");
	GpuMorMaterial const mat = (
		mor::scene_gpu_materials_get(s.gltfInstances[0].materials, redIndex)
	);
	CHECK(mat.baseWeight.r == doctest::Approx(0.5f));

	stage::stage_destroy(s);
}

TEST_CASE("stage: relative asset paths resolve against the scene file directory") {
	std::filesystem::path const dir = stage_test_dir("relative-paths");
	std::filesystem::path const gltfPath = (
		std::filesystem::path(REPO_DIR) / "assets/Models/Box/glTF/Box.gltf"
	);
	REQUIRE(std::filesystem::exists(gltfPath));
	std::filesystem::path const relative = (
		std::filesystem::relative(gltfPath, dir)
	);

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"instances\":[{"
		"\"type\":\"gltf\","
		"\"name\":\"box\","
		"\"path\":\"" + relative.string() + "\""
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	stage::Stage s = stage::stage_load((dir / "scene.json").string().c_str());
	REQUIRE(s.gltfInstances.size() == 1u);
	CHECK(std::filesystem::equivalent(s.gltfInstances[0].path, gltfPath));
	CHECK(s.gltfInstances[0].scene.id != 0u);

	stage::stage_destroy(s);
}

TEST_CASE("stage: repeated load/destroy of a mixed scene many times") {
	ponder::vdb_initialize();
	std::filesystem::path const dir = stage_test_dir("stress");
	std::string const gltfPath = (
		std::string(REPO_DIR) + "/assets/Models/Box/glTF/Box.gltf"
	);
	std::string const vdbPath = (
		std::string(REPO_DIR) + "/assets/vdb/smoke.vdb"
	);
	REQUIRE(std::filesystem::exists(gltfPath));
	REQUIRE(std::filesystem::exists(vdbPath));

	write_file(dir / "child.json", "{\"version\":1,\"name\":\"stress-child\"}");

	std::string const json = (
		"{"
		"\"version\":1,"
		"\"instances\":[{"
		"\"type\":\"gltf\","
		"\"name\":\"box\","
		"\"path\":\"" + gltfPath + "\","
		"\"materialOverrides\":[{"
		"\"material\":\"Red\","
		"\"fields\":{\"baseWeight\":{\"value\":0.5}}"
		"}]"
		"},{"
		"\"type\":\"vdb\","
		"\"name\":\"vol\","
		"\"path\":\"" + vdbPath + "\","
		"\"blobs\":[{\"grid\":\"density\"}]"
		"},{"
		"\"type\":\"stage\","
		"\"name\":\"childRef\","
		"\"path\":\"child.json\""
		"}]"
		"}"
	);
	write_file(dir / "scene.json", json);

	static constexpr u32 skIterations = 30u;
	std::string const scenePath = (dir / "scene.json").string();
	for (u32 i = 0u; i < skIterations; ++i) {
		stage::Stage s = stage::stage_load(scenePath.c_str());
		REQUIRE_MESSAGE(!s.sourceDirectory.empty(), "iteration ", i);
		REQUIRE_MESSAGE(s.gltfInstances.size() == 1u, "iteration ", i);
		REQUIRE_MESSAGE(s.gltfInstances[0].scene.id != 0u, "iteration ", i);
		REQUIRE_MESSAGE(s.gltfInstances[0].gpuScene.id != 0u, "iteration ", i);
		REQUIRE_MESSAGE(s.vdbInstances.size() == 1u, "iteration ", i);
		REQUIRE_MESSAGE(
			s.vdbInstances[0].blobs[0].vdb.gpuBuffer.id != 0u, "iteration ", i
		);
		REQUIRE_MESSAGE(s.stageInstances.size() == 1u, "iteration ", i);
		REQUIRE_MESSAGE(s.stageInstances[0].child != nullptr, "iteration ", i);
		stage::stage_destroy(s);
	}
}

} // TEST_SUITE("[headless]")
