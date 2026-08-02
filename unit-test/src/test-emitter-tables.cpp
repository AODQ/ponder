#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <ponder/emitter-tables.hpp>
#include <srat/core-math.hpp>
#include "util.hpp"

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// tier 0 of plan-nee.md: the cpu-side emitter list build. no gpu, no gltf --
// emitter_lights_append/finalize take plain per-meshlet records, so the
// weighting, the flat-primitiveIndex bookkeeping and the cdf math are all
// testable in isolation from mor and from the sampler that consumes them.
//
// the properties under test are the ones the sampler's correctness rests on:
// a monotonic cdf whose last entry is exactly 1.0 (fnPickEmitterLight's
// binary search would otherwise leave a sliver of the [0,1) pick range past
// the end of the table), pick probabilities that match the powers they were
// built from, and emitter ranges that address triangles in the *flat*
// numbering ray hits report rather than the meshlet-local numbering the
// raster path uses.
// ---------------------------------------------------------------------------

namespace {

using ponder::EmitterMeshletInput;
using ponder::EmitterRange;

EmitterMeshletInput make_input(
	u32 const primitiveIndexBase,
	u32 const triangleCount,
	f32 const objectArea,
	f32v3 const emissionColor,
	f32 const emissionLuminance
) {
	return EmitterMeshletInput {
		.primitiveIndexBase = primitiveIndexBase,
		.triangleCount = triangleCount,
		.objectArea = objectArea,
		.emissionColor = emissionColor,
		.emissionLuminance = emissionLuminance,
	};
}

srat::slice<EmitterMeshletInput const> as_slice(
	std::vector<EmitterMeshletInput> const & v
) {
	return srat::slice<EmitterMeshletInput const>(v.data(), v.size());
}

// per-entry pick probability implied by the cdf, i.e. what
// fnEmitterLightPickPdf returns in the shader
f32 pick_pdf(std::vector<GpuEmitterLight> const & lights, size_t const i) {
	f32 const prev = i == 0u ? 0.0f : lights[i - 1u].cdf;
	return lights[i].cdf - prev;
}

constexpr f32 skLuma(f32v3 const c) {
	return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

} // namespace

TEST_CASE("emitter tables: area scale factor") {
	// identity leaves object area untouched
	CHECK(
		ponder::emitter_transform_area_scale(f32m44_identity())
		== doctest::Approx(1.0f)
	);

	// a uniform scale s scales area by exactly s^2, the case cull's build
	// hard-coded; |det|^(2/3) must reproduce it rather than merely approximate
	for (f32 const s : { 0.25f, 0.5f, 2.0f, 7.5f }) {
		CHECK(
			ponder::emitter_transform_area_scale(f32m44_scale(s, s, s))
			== doctest::Approx(s * s).epsilon(1e-5f)
		);
	}

	// rotation is area preserving
	f32m44 const rotated = (
		f32m44_rotate_x(0.7f) * f32m44_rotate_y(-1.3f) * f32m44_rotate_z(2.1f)
	);
	CHECK(
		ponder::emitter_transform_area_scale(rotated)
		== doctest::Approx(1.0f).epsilon(1e-4f)
	);

	// translation does not scale area
	CHECK(
		ponder::emitter_transform_area_scale(f32m44_translate(5.0f, -2.0f, 9.0f))
		== doctest::Approx(1.0f)
	);

	// a degenerate (zero-determinant) transform flattens every triangle to
	// zero area, which must drop the emitter rather than divide by zero
	CHECK(
		ponder::emitter_transform_area_scale(f32m44_scale(1.0f, 0.0f, 1.0f))
		== doctest::Approx(0.0f)
	);

	// non-uniform scale is an isotropic estimate, not exact -- pin that it
	// lands between the smallest and largest true per-plane area scale so a
	// future exact implementation can only tighten this, never contradict it
	f32 const nonUniform = (
		ponder::emitter_transform_area_scale(f32m44_scale(1.0f, 2.0f, 4.0f))
	);
	CHECK(nonUniform > 1.0f * 2.0f);
	CHECK(nonUniform < 2.0f * 4.0f);
}

TEST_CASE("emitter tables: append filters non-emitters") {
	std::vector<EmitterMeshletInput> inputs;
	// emissive
	inputs.emplace_back(make_input(0u, 4u, 2.0f, { 1.0f, 1.0f, 1.0f }, 3.0f));
	// zero luminance
	inputs.emplace_back(make_input(4u, 4u, 2.0f, { 1.0f, 1.0f, 1.0f }, 0.0f));
	// black emission color
	inputs.emplace_back(make_input(8u, 4u, 2.0f, { 0.0f, 0.0f, 0.0f }, 5.0f));
	// zero area (degenerate meshlet)
	inputs.emplace_back(make_input(12u, 4u, 0.0f, { 1.0f, 1.0f, 1.0f }, 5.0f));
	// zero triangles
	inputs.emplace_back(make_input(16u, 0u, 2.0f, { 1.0f, 1.0f, 1.0f }, 5.0f));
	// emissive
	inputs.emplace_back(make_input(16u, 2u, 1.0f, { 1.0f, 0.0f, 0.0f }, 4.0f));

	std::vector<GpuEmitterLight> lights;
	EmitterRange const range = (
		ponder::emitter_lights_append(
			lights, as_slice(inputs), f32m44_identity(), /*modelDrawIndex=*/3u,
			/*maxLights=*/1024u
		)
	);

	REQUIRE(lights.size() == 2u);
	CHECK(range.offset == 0u);
	CHECK(range.count == 2u);
	CHECK(lights[0].primitiveIndexBase == 0u);
	CHECK(lights[0].triangleCount == 4u);
	CHECK(lights[0].modelDrawIndex == 3u);
	CHECK(lights[1].primitiveIndexBase == 16u);
	CHECK(lights[1].triangleCount == 2u);
	CHECK(lights[1].modelDrawIndex == 3u);

	// radiance is emissionColor * emissionLuminance, matching what the
	// bsdf-hit branch adds at a vertex
	CHECK(lights[1].radiance.x == doctest::Approx(4.0f));
	CHECK(lights[1].radiance.y == doctest::Approx(0.0f));
	CHECK(lights[1].radiance.z == doctest::Approx(0.0f));

	// power is worldArea * luma * luminance
	CHECK(lights[0].power == doctest::Approx(2.0f * 1.0f * 3.0f));
	CHECK(
		lights[1].power
		== doctest::Approx(1.0f * skLuma({ 1.0f, 0.0f, 0.0f }) * 4.0f)
	);
}

TEST_CASE("emitter tables: power scales with the instance transform") {
	std::vector<EmitterMeshletInput> inputs;
	inputs.emplace_back(make_input(0u, 2u, 3.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));

	// the same mesh instanced at two scales must weight by *world* area, so
	// the far-larger instance is picked proportionally more often
	std::vector<GpuEmitterLight> lights;
	(void)ponder::emitter_lights_append(
		lights, as_slice(inputs), f32m44_identity(), 0u, 1024u
	);
	(void)ponder::emitter_lights_append(
		lights, as_slice(inputs), f32m44_scale(3.0f, 3.0f, 3.0f), 1u, 1024u
	);
	REQUIRE(lights.size() == 2u);
	CHECK(lights[1].power == doctest::Approx(lights[0].power * 9.0f));

	// a zero-scale (degenerate) instance contributes nothing
	std::vector<GpuEmitterLight> flattened;
	EmitterRange const range = (
		ponder::emitter_lights_append(
			flattened, as_slice(inputs), f32m44_scale(1.0f, 0.0f, 1.0f), 0u, 1024u
		)
	);
	CHECK(flattened.empty());
	CHECK(range.count == 0u);
}

TEST_CASE("emitter tables: ranges are per-instance and contiguous") {
	std::vector<EmitterMeshletInput> a;
	a.emplace_back(make_input(0u, 1u, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	a.emplace_back(make_input(1u, 1u, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	std::vector<EmitterMeshletInput> b;
	b.emplace_back(make_input(0u, 1u, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));

	std::vector<GpuEmitterLight> lights;
	EmitterRange const rangeA = (
		ponder::emitter_lights_append(
			lights, as_slice(a), f32m44_identity(), 0u, 1024u
		)
	);
	EmitterRange const rangeB = (
		ponder::emitter_lights_append(
			lights, as_slice(b), f32m44_identity(), 1u, 1024u
		)
	);
	// an instance with no emissive meshlets still gets a valid empty range
	std::vector<EmitterMeshletInput> const empty;
	EmitterRange const rangeC = (
		ponder::emitter_lights_append(
			lights, as_slice(empty), f32m44_identity(), 2u, 1024u
		)
	);

	CHECK(rangeA.offset == 0u);
	CHECK(rangeA.count == 2u);
	CHECK(rangeB.offset == 2u);
	CHECK(rangeB.count == 1u);
	CHECK(rangeC.offset == 3u);
	CHECK(rangeC.count == 0u);
	// every entry in a range belongs to the model that range was built for
	for (u32 i = 0u; i < rangeB.count; ++i) {
		CHECK(lights[rangeB.offset + i].modelDrawIndex == 1u);
	}
}

TEST_CASE("emitter tables: maxLights clips without corrupting the list") {
	std::vector<EmitterMeshletInput> inputs;
	for (u32 i = 0u; i < 32u; ++i) {
		inputs.emplace_back(
			make_input(i, 1u, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f)
		);
	}

	std::vector<GpuEmitterLight> lights;
	EmitterRange const range = (
		ponder::emitter_lights_append(
			lights, as_slice(inputs), f32m44_identity(), 0u, /*maxLights=*/5u
		)
	);
	CHECK(lights.size() == 5u);
	CHECK(range.count == 5u);

	// a clipped list is still a valid distribution -- the dropped emitters
	// simply become unsampleable, which the bsdf technique covers at full
	// weight rather than being a bias
	f32 const totalPower = ponder::emitter_lights_finalize(lights);
	CHECK(totalPower == doctest::Approx(5.0f));
	CHECK(lights.back().cdf == doctest::Approx(1.0f));
}

TEST_CASE("emitter tables: finalize builds a monotonic normalized cdf") {
	std::vector<EmitterMeshletInput> inputs;
	// deliberately uneven powers so a uniform-cdf bug cannot pass
	inputs.emplace_back(make_input(0u, 1u, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	inputs.emplace_back(make_input(1u, 1u, 7.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	inputs.emplace_back(make_input(2u, 1u, 0.5f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	inputs.emplace_back(make_input(3u, 1u, 91.5f, { 1.0f, 1.0f, 1.0f }, 1.0f));

	std::vector<GpuEmitterLight> lights;
	(void)ponder::emitter_lights_append(
		lights, as_slice(inputs), f32m44_identity(), 0u, 1024u
	);
	f32 const totalPower = ponder::emitter_lights_finalize(lights);

	REQUIRE(lights.size() == 4u);
	CHECK(totalPower == doctest::Approx(100.0f));

	// strictly increasing, and the last entry is exactly 1.0 so no pick can
	// fall past the end of the table
	f32 prev = 0.0f;
	for (GpuEmitterLight const & l : lights) {
		CHECK(l.cdf > prev);
		prev = l.cdf;
	}
	CHECK(lights.back().cdf == 1.0f);

	// each entry's pick probability is its share of total power
	CHECK(pick_pdf(lights, 0u) == doctest::Approx(0.01f));
	CHECK(pick_pdf(lights, 1u) == doctest::Approx(0.07f));
	CHECK(pick_pdf(lights, 2u) == doctest::Approx(0.005f));
	CHECK(pick_pdf(lights, 3u) == doctest::Approx(0.915f));

	// pick probabilities are a partition of unity
	f32 sum = 0.0f;
	for (size_t i = 0u; i < lights.size(); ++i) {
		sum += pick_pdf(lights, i);
	}
	CHECK(sum == doctest::Approx(1.0f));
}

TEST_CASE("emitter tables: finalize on an empty list") {
	std::vector<GpuEmitterLight> lights;
	CHECK(ponder::emitter_lights_finalize(lights) == doctest::Approx(0.0f));
	CHECK(lights.empty());
}

TEST_CASE("emitter tables: cdf pick reproduces the power distribution") {
	// a wide dynamic range: one dominant emitter and a long tail of dim
	// ones, the configuration a uniform pick would handle worst
	std::vector<EmitterMeshletInput> inputs;
	inputs.emplace_back(make_input(0u, 1u, 500.0f, { 1.0f, 1.0f, 1.0f }, 1.0f));
	for (u32 i = 1u; i < 64u; ++i) {
		inputs.emplace_back(
			make_input(i, 1u, (f32)i * 0.25f, { 1.0f, 1.0f, 1.0f }, 1.0f)
		);
	}

	std::vector<GpuEmitterLight> lights;
	(void)ponder::emitter_lights_append(
		lights, as_slice(inputs), f32m44_identity(), 0u, 1024u
	);
	f32 const totalPower = ponder::emitter_lights_finalize(lights);
	REQUIRE(lights.size() == 64u);

	// mirrors fnPickEmitterLight's binary search exactly; a mismatch here
	// means the shader and the builder disagree about what the cdf means
	auto const pick = [&](f32 const u) {
		u32 lo = 0u;
		u32 hi = (u32)lights.size() - 1u;
		while (lo < hi) {
			u32 const mid = (lo + hi) / 2u;
			if (lights[mid].cdf < u) { lo = mid + 1u; } else { hi = mid; }
		}
		return lo;
	};

	// deterministic stratified sweep rather than an rng: every bin of the
	// [0,1) pick range is visited exactly once, so the histogram converges
	// to the exact distribution instead of a noisy estimate
	constexpr u32 skDrawCount = 1u << 20;
	std::vector<u32> histogram(lights.size(), 0u);
	for (u32 i = 0u; i < skDrawCount; ++i) {
		f32 const u = ((f32)i + 0.5f) / (f32)skDrawCount;
		u32 const idx = pick(u);
		REQUIRE(idx < lights.size());
		histogram[idx] += 1u;
	}

	for (size_t i = 0u; i < lights.size(); ++i) {
		f32 const observed = (f32)histogram[i] / (f32)skDrawCount;
		f32 const expected = lights[i].power / totalPower;
		// one draw of slack for the stratified grid landing mid-bucket
		CHECK(
			observed
			== doctest::Approx(expected).epsilon(1e-4f).scale(1.0f)
		);
	}

	// the extremes of the pick range must stay in bounds
	CHECK(pick(0.0f) == 0u);
	CHECK(pick(1.0f) == lights.size() - 1u);
}

// ---------------------------------------------------------------------------
// the flat-numbering cross-check. emitter_meshlet_inputs_from_scene
// reconstructs each meshlet's primitiveIndexBase by summing triangleCounts in
// meshlet order, on the assumption that mor's flatIndices/flatMeshlets (what a
// ray hit's primitiveIndex indexes) concatenate meshlets in that same order.
// if that assumption is ever wrong the sampler and the backward hit lookup
// would still agree with each other -- both read the same table -- so no
// chi-square or pdf round-trip could catch it. it would just silently light
// the wrong triangles. cull's light_shared.h warns about exactly this
// (meshlet-local triangleOffset numbering vs flat numbering), so it gets a
// direct test against the real uploaded buffer rather than a restatement of
// the builder's own arithmetic.
// ---------------------------------------------------------------------------

namespace {

struct CopyPush { u64 srcVa; u64 dstVa; u32 count; };

std::string emissive_model_path() {
	return (
		std::filesystem::path(REPO_DIR) / "assets" / "models-categorized"
		/ "test-emissive" / "EmissiveStrengthTest.glb"
	).string();
}

// mor keeps flatMeshlets gpu-only, so pull it back through a copy shader
std::vector<u32> download_flat_meshlets(u64 const srcVa, u32 const count) {
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "slang/entry/emitter_flat_meshlet_copy.slang" , .includePaths = test::slang_include_paths() }
		)
	);
	REQUIRE(pipeline.id != 0);
	vkof::Buffer dst = test::make_buffer_u32(count);
	CopyPush const push {
		.srcVa = srcVa,
		.dstVa = vkof::buffer_virtual_address(dst),
		.count = count,
	};
	test::dispatch(pipeline, push, (count + 63u) / 64u);
	std::vector<u32> const result = test::readback<u32>(dst, 0, count);
	vkof::buffer_destroy(dst);
	vkof::pipeline_destroy(pipeline);
	return result;
}

} // namespace

TEST_CASE("emitter tables: primitiveIndexBase matches mor's flat numbering") {
	std::string const path = emissive_model_path();
	REQUIRE(std::filesystem::exists(path));

	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	mor::GpuScene const gpuScene = mor::scene_gpu_upload(scene);
	mor::Buffers const bufs = mor::scene_gpu_buffers(gpuScene);
	REQUIRE(bufs.triangleCount > 0u);

	std::vector<mor::MeshletAreaInfo> const areas = (
		mor::scene_meshlet_area_info(scene)
	);
	REQUIRE(!areas.empty());

	std::vector<u32> const flatMeshlets = (
		download_flat_meshlets(bufs.flatMeshlets, bufs.triangleCount)
	);

	// every triangle in [base, base + triangleCount) must report the meshlet
	// the running sum says it belongs to
	u32 base = 0u;
	u32 checkedTriangles = 0u;
	for (u32 mi = 0u; mi < (u32)areas.size(); ++mi) {
		mor::MeshletAreaInfo const & info = areas[mi];
		for (u32 t = 0u; t < info.triangleCount; ++t) {
			REQUIRE(base + t < flatMeshlets.size());
			// a mismatch here means the emitter list would address the wrong
			// triangles entirely
			REQUIRE(flatMeshlets[base + t] == mi);
			++checkedTriangles;
		}
		base += info.triangleCount;
	}
	// the running sum must also account for the whole buffer, i.e. no
	// triangles exist outside any meshlet's range
	CHECK(base == bufs.triangleCount);
	CHECK(checkedTriangles == bufs.triangleCount);

	mor::scene_gpu_destroy(gpuScene);
	mor::scene_destroy(scene);
}

TEST_CASE("emitter tables: collects exactly the emissive meshlets of a gltf") {
	std::string const path = emissive_model_path();
	REQUIRE(std::filesystem::exists(path));

	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path.c_str());
	std::vector<mor::MeshletAreaInfo> const areas = (
		mor::scene_meshlet_area_info(scene)
	);
	REQUIRE(!areas.empty());

	std::vector<ponder::EmitterMeshletInput> inputs;
	ponder::emitter_meshlet_inputs_from_scene(
		scene,
		/*materialsOverride=*/mor::GpuMaterials {},
		srat::slice<mor::MeshletAreaInfo const>(areas.data(), areas.size()),
		inputs
	);

	// EmissiveStrengthTest is a grid of emitters at varying strengths, so a
	// build that silently collected nothing (or everything) is caught here
	REQUIRE(!inputs.empty());
	CHECK(inputs.size() < areas.size());

	// independently recompute which meshlets should have been collected, and
	// where they start, without going through the function under test
	u32 base = 0u;
	std::vector<ponder::EmitterMeshletInput> expected;
	for (mor::MeshletAreaInfo const & info : areas) {
		u32 const thisBase = base;
		base += info.triangleCount;
		GpuMorMaterial const mat = (
			mor::scene_material_get(scene, info.materialIndex)
		);
		if (mat.emissionLuminance.r <= 0.0f) {
			continue;
		}
		expected.emplace_back(ponder::EmitterMeshletInput {
			.primitiveIndexBase = thisBase,
			.triangleCount = info.triangleCount,
			.objectArea = info.area,
			.emissionColor = mat.emissionColor.rgb,
			.emissionLuminance = mat.emissionLuminance.r,
		});
	}

	REQUIRE(inputs.size() == expected.size());
	for (size_t i = 0u; i < inputs.size(); ++i) {
		CHECK(inputs[i].primitiveIndexBase == expected[i].primitiveIndexBase);
		CHECK(inputs[i].triangleCount == expected[i].triangleCount);
		CHECK(inputs[i].objectArea == doctest::Approx(expected[i].objectArea));
		CHECK(
			inputs[i].emissionLuminance
			== doctest::Approx(expected[i].emissionLuminance)
		);
	}

	// and the collected set must survive the append stage as real emitters
	std::vector<GpuEmitterLight> lights;
	ponder::EmitterRange const range = (
		ponder::emitter_lights_append(
			lights,
			srat::slice<ponder::EmitterMeshletInput const>(
				inputs.data(), inputs.size()
			),
			f32m44_identity(), /*modelDrawIndex=*/0u, /*maxLights=*/4096u
		)
	);
	CHECK(range.count > 0u);
	f32 const totalPower = ponder::emitter_lights_finalize(lights);
	CHECK(totalPower > 0.0f);
	CHECK(lights.back().cdf == doctest::Approx(1.0f));

	mor::scene_destroy(scene);
}
