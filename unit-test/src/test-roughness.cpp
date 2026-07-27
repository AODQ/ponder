#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

namespace {

struct RoughnessPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

std::vector<f32v2> evaluate_roughness_alpha(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "roughness_alpha_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v2)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	RoughnessPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v2>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32v4> evaluate_roughness_alpha_spec_crosscheck(
	std::vector<f32v2> const & samples
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "roughness_alpha_spec_crosscheck.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v2)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v4),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	RoughnessPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v4>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> evaluate_coat_roughened_roughness(std::vector<f32v3> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "coat_roughened_roughness_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v3)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	RoughnessPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------
// -- openPbrCoatRoughenedRoughness: verified to match the openpbr spec's
// -- r'_B formula exactly, no discrepancy found
// -----------------------------------------------------------------------

TEST_CASE("coat roughened roughness: identity at coatWeight=0") {
	std::vector<f32v3> const samples = {
		{ 0.1f, 0.5f, 0.0f }, { 0.5f, 0.1f, 0.0f }, { 0.8f, 0.9f, 0.0f },
	};
	auto const result = evaluate_coat_roughened_roughness(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i] == doctest::Approx(samples[i].x).epsilon(0.0001));
	}
}

TEST_CASE("coat roughened roughness: identity when coatRoughness=0, any coatWeight") {
	std::vector<f32v3> samples;
	for (f32 cw = 0.0f; cw <= 1.0f; cw += 0.2f) {
		samples.push_back({ 0.4f, 0.0f, cw });
	}
	auto const result = evaluate_coat_roughened_roughness(samples);
	for (f32 v : result) { CHECK(v == doctest::Approx(0.4).epsilon(0.0001)); }
}

TEST_CASE("coat roughened roughness: golden value at baseRoughness=0, coatWeight=1") {
	std::vector<f32v3> const samples = {
		{ 0.0f, 0.3f, 1.0f }, { 0.0f, 0.5f, 1.0f },
	};
	auto const result = evaluate_coat_roughened_roughness(samples);
	constexpr f32 kTwoQuarterRoot = 1.18920711500272106671f; // 2^(1/4)
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		f32 const expected = kTwoQuarterRoot * samples[i].y;
		CHECK(result[i] == doctest::Approx(expected).epsilon(0.001));
	}
}

TEST_CASE("coat roughened roughness: monotonic non-decreasing in coatWeight, >= baseRoughness (fuzz)") {
	u32 seed = 1u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 trial = 0; trial < 64u; ++trial) {
		f32 const baseR = next();
		f32 const coatR = next();
		std::vector<f32v3> samples;
		for (f32 cw = 0.0f; cw <= 1.0f; cw += 0.05f) { samples.push_back({ baseR, coatR, cw }); }
		auto const result = evaluate_coat_roughened_roughness(samples);

		f32 prev = -1.0f;
		for (u32 i = 0; i < result.size(); ++i) {
			CAPTURE(trial);
			CAPTURE(i);
			CHECK(result[i] >= baseR - 0.0001f);
			CHECK(result[i] >= prev - 0.0001f);
			prev = result[i];
		}
	}
}

// -----------------------------------------------------------------------
// -- openPbrRoughnessAlpha: alpha = roughness (linear, no perceptual
// -- remap), matching cull's shader-side formula 1:1. the openpbr spec's
// -- written formula squares roughness instead, and cull's own CPU-side
// -- debug-visualization code independently squares it too -- so cull's
// -- shader disagrees with the spec and with cull's own other code path.
// -- a prior version of util-roughness.glsl broke from cull to match the
// -- spec; reverted back to cull's linear convention for consistency
// -- with the rest of the port (see roughness_alpha_spec_crosscheck.comp
// -- / util-roughness-reference.glsl below, which documents this as a
// -- deliberate, permanent mismatch against the spec text).
// -----------------------------------------------------------------------

TEST_CASE("roughness alpha: golden value at anisotropy=0 is roughness (linear)") {
	std::vector<f32v2> const samples = {
		{ 0.1f, 0.0f }, { 0.3f, 0.0f }, { 0.6f, 0.0f }, { 0.9f, 0.0f },
	};
	auto const result = evaluate_roughness_alpha(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		f32 const roughness = samples[i].x;
		f32 const expected = roughness;
		CHECK(result[i].x == doctest::Approx(expected).epsilon(0.001));
		CHECK(result[i].y == doctest::Approx(expected).epsilon(0.001));
	}
}

TEST_CASE("roughness alpha: floored so the degenerate axis stays sampleable (fuzz)") {
	std::vector<f32v2> samples;
	u32 seed = 2u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 512u; ++i) { samples.push_back({ next(), next() }); }
	auto const result = evaluate_roughness_alpha(samples);
	for (auto const & a : result) {
		CHECK(a.x >= 1e-5f - 1e-9f);
		CHECK(a.y >= 1e-5f - 1e-9f);
	}
}

TEST_CASE("roughness alpha: deliberately diverges from the openpbr spec's r^2 mapping by a factor of roughness") {
	// spec.alphaT = r^2 * f(a) = r * (r * f(a)) = r * mine.alphaT, and
	// likewise for alphaB -- an exact relationship away from the 1e-5
	// floor clamp, which the roughness range below stays clear of
	std::vector<f32v2> samples;
	u32 seed = 3u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 512u; ++i) {
		samples.push_back({ 0.05f + next() * 0.95f, next() });
	}
	auto const result = evaluate_roughness_alpha_spec_crosscheck(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CAPTURE(samples[i].x);
		CAPTURE(samples[i].y);
		f32 const roughness = samples[i].x;
		f32v4 const & r = result[i];
		// r.xy is ponder's (linear), r.zw is the spec reference (squared)
		CHECK(r.z == doctest::Approx(roughness * r.x).epsilon(0.001));
		CHECK(r.w == doctest::Approx(roughness * r.y).epsilon(0.001));
		// and they must actually differ, not silently agree by coincidence
		CHECK(r.x != doctest::Approx(r.z).epsilon(0.001));
	}
}

} // TEST_SUITE("[headless]")
