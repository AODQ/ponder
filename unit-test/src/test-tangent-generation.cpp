#include <doctest/doctest.h>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <srat/core-math.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// mor generates a per-vertex tangent basis from uvs whenever a gltf omits
// TANGENT, since a normal map is meaningless without one. these tests pin
// that generation against ground truth by loading a model twice: once as
// authored (tangents present) and once from a copy with the TANGENT
// attribute stripped out of the json.
//
// the handedness convention is the fragile half -- gltf's bitangent,
// cross(normal, tangent.xyz) * tangent.w, runs along *decreasing* v -- so it
// gets its own assertion rather than riding along on the direction check
// ---------------------------------------------------------------------------

namespace {

// a gltf whose primitives ship TANGENT, so the authored data can referee
constexpr char const * skModelRelPath = (
	"BarramundiFish/glTF/BarramundiFish.gltf"
);

std::filesystem::path model_path() {
	return (
		std::filesystem::path(REPO_DIR) / "assets" / "Models" / skModelRelPath
	);
}

// copies the model beside its buffers/textures with every "TANGENT":<n> entry
// removed, so mor takes the generation path on an otherwise identical asset
std::filesystem::path write_stripped_copy() {
	std::filesystem::path const dir = (
		std::filesystem::temp_directory_path() / "ponder-tangent-test"
	);
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);
	std::filesystem::create_directories(dir);

	std::filesystem::path const src = model_path();
	for (auto const & entry : std::filesystem::directory_iterator(src.parent_path())) {
		if (!entry.is_regular_file()) { continue; }
		std::filesystem::copy_file(
			entry.path(), dir / entry.path().filename(),
			std::filesystem::copy_options::overwrite_existing
		);
	}

	std::filesystem::path const gltf = dir / src.filename();
	std::string json;
	{
		std::ifstream in(gltf, std::ios::binary);
		json.assign(
			std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()
		);
	}

	// "TANGENT":<digits>, plus whichever comma keeps the object well formed
	u32 stripped = 0u;
	for (;;) {
		size_t const key = json.find("\"TANGENT\"");
		if (key == std::string::npos) { break; }
		size_t end = json.find_first_of(",}", key);
		REQUIRE(end != std::string::npos);
		size_t begin = key;
		if (json[end] == ',') {
			++end;
		} else if (begin > 0u && json[begin - 1u] == ',') {
			--begin;
		}
		json.erase(begin, end - begin);
		++stripped;
	}
	REQUIRE(stripped > 0u);

	std::ofstream(gltf, std::ios::binary) << json;
	return gltf;
}

struct LoadedVerts {
	mor::Scene scene;
	std::vector<GpuMorVertexAttribute> attributes;
};

LoadedVerts load(std::filesystem::path const & path) {
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.string().c_str());
	u32 const count = mor::scene_vertex_count(scene);
	REQUIRE(count > 0u);
	GpuMorVertexAttribute const * const attrs = (
		mor::scene_vertex_attributes(scene)
	);
	return {
		scene, std::vector<GpuMorVertexAttribute>(attrs, attrs + count),
	};
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("tangent generation: matches authored TANGENT on a stripped copy") {
	REQUIRE(std::filesystem::exists(model_path()));

	LoadedVerts const authored = load(model_path());
	LoadedVerts const generated = load(write_stripped_copy());

	// stripping an attribute must not disturb anything else about the load
	REQUIRE(authored.attributes.size() == generated.attributes.size());

	u32 compared = 0u, handednessMismatch = 0u, degenerate = 0u;
	f64 dotSum = 0.0;
	f32 worstDot = 1.0f;
	f32 worstPerp = 0.0f;
	for (size_t i = 0u; i < authored.attributes.size(); ++i) {
		GpuMorVertexAttribute const & a = authored.attributes[i];
		GpuMorVertexAttribute const & g = generated.attributes[i];

		f32v3 const gt = { g.tangent.x, g.tangent.y, g.tangent.z };
		// a vertex whose uv neighbourhood is degenerate gets left at zero on
		// purpose; the shader falls back to an arbitrary frisvad basis there
		if (f32v3_length(gt) < 0.5f) { ++degenerate; continue; }

		CHECK(f32v3_length(gt) == doctest::Approx(1.0f).epsilon(1e-3f));
		CHECK(std::abs(g.tangent.w) == doctest::Approx(1.0f));

		f32v3 const at = f32v3_normalize({ a.tangent.x, a.tangent.y, a.tangent.z });
		f32 const d = f32v3_dot(at, f32v3_normalize(gt));
		dotSum += d;
		worstDot = std::min(worstDot, d);
		if (a.tangent.w != g.tangent.w) { ++handednessMismatch; }

		// generated tangents are gram-schmidt'd against the normal
		f32 const perp = std::abs(f32v3_dot(f32v3_normalize(g.normal), gt));
		worstPerp = std::max(worstPerp, perp);
		++compared;
	}
	REQUIRE(compared > 0u);

	f32 const meanDot = (f32)(dotSum / compared);
	MESSAGE(
		"verts ", compared, " mean dot ", meanDot, " worst dot ", worstDot,
		" handedness mismatches ", handednessMismatch,
		" degenerate ", degenerate
	);

	// direction: uv-space gradients reproduce the authored tangent closely,
	// modulo the smoothing differences any two generators disagree on
	CHECK(meanDot > 0.99f);
	CHECK(worstPerp < 1e-3f);

	// handedness: an inverted convention flips essentially every vertex, so
	// this only tolerates the handful of uv-seam vertices where authored
	// tooling and a plain accumulate legitimately differ
	CHECK(handednessMismatch <= compared / 100u);

	mor::scene_destroy(authored.scene);
	mor::scene_destroy(generated.scene);
	mor::sampler_cache_destroy();
}

} // TEST_SUITE
