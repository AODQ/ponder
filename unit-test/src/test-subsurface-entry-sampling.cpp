#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// tests for the subsurface-entry lobe added to
// lib/ponder/shaders/util-material-openpbr.glsl (OPENPBR_LOBE_SUBSURFACE_
// ENTRY, openPbrSubsurfaceEntrySampleWo / openPbrSubsurfaceEntryPdf in
// util-material-openpbr-subsurface.glsl): the entry event that starts a
// bounded subsurface random walk, split out of the existing diffuse budget
// by subsurfaceWeight the same way transmission already splits from it by
// transmissionWeight.
//
// two independent things need checking, mirroring how every other lobe's
// milestone split "does this direction distribution match its own pdf" from
// "does the discrete lobe pick land in the right proportions":
// 1. sampler-vs-eval chi-square for openPbrSubsurfaceEntrySampleWo/Pdf in
//    isolation (same convention as test-vdb-phase.cpp / test-thinfilm.cpp)
// 2. the discrete OPENPBR_LOBE_SUBSURFACE_ENTRY vs OPENPBR_LOBE_DIFFUSE pick
//    fraction out of the full openPbrSampleWo, which must track
//    subsurfaceWeight

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// -----------------------------------------------------------------------------
// subsurface_entry_histogram.comp / subsurface_entry_expected.comp runners
// -----------------------------------------------------------------------------

struct EntryPush {
	u64 va;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 nor;
};
static_assert(sizeof(EntryPush) == 32);

std::vector<u32> run_entry_histogram(
	f32v3 const nor, u32 const sampleCount, u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_entry_histogram.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const slotCount = kBinCount + 1u;
	auto histBuf = vkof::buffer_create({
		.byteCount = slotCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zeros(slotCount, 0u);
	vkof::buffer_upload({
		.buffer = histBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()),
			slotCount * sizeof(u32)
		),
	});

	EntryPush const push {
		.va = vkof::buffer_virtual_address(histBuf),
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.nor = nor,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1u) / kLocalSize);

	auto out = test::readback<u32>(histBuf, 0, slotCount);
	vkof::buffer_destroy(histBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_entry_expected(
	f32v3 const nor, u32 const sampleCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_entry_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	EntryPush const push {
		.va = vkof::buffer_virtual_address(buf),
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.nor = nor,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (kBinCount + kLocalSize - 1u) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, kBinCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// pearson chi-square + wilson-hilferty p-value, same convention as
// test-vdb-phase.cpp / test-thinfilm.cpp / test-mixture-lobe-selection.cpp
struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
};

ChiSquare chi_square(
	std::vector<u32> const & observed, std::vector<f32> const & expected
) {
	f64 statistic = 0.0;
	u32 cells = 0u;
	f64 pooledO = 0.0;
	f64 pooledE = 0.0;
	u64 observedTotal = 0u;
	f64 expectedTotal = 0.0;
	for (u32 i = 0; i < kBinCount; ++i) {
		observedTotal += observed[i];
		if (std::isnan(expected[i])) { continue; }
		expectedTotal += (f64)expected[i];
		if (expected[i] >= 5.0f) {
			f64 const d = (f64)observed[i] - (f64)expected[i];
			statistic += d * d / (f64)expected[i];
			cells++;
		} else {
			pooledO += (f64)observed[i];
			pooledE += (f64)expected[i];
		}
	}
	if (pooledE > 0.0) {
		f64 const d = pooledO - pooledE;
		statistic += d * d / pooledE;
		cells++;
	}
	f64 const dof = std::fmax((f64)cells - 1.0, 1.0);
	f64 const t = std::cbrt(statistic / dof);
	f64 const mu = 1.0 - 2.0 / (9.0 * dof);
	f64 const sigma = std::sqrt(2.0 / (9.0 * dof));
	f64 const z = (t - mu) / sigma;
	f64 const pValue = 0.5 * std::erfc(z / std::sqrt(2.0));
	return { statistic, dof, pValue, observedTotal, expectedTotal };
}

// -----------------------------------------------------------------------------
// subsurface_entry_selection_fraction.comp runner
// -----------------------------------------------------------------------------

struct SelectionPush {
	u64 pickCountVa;
	u32 sampleCount;
	u32 seedSalt;
	f32 subsurfaceWeight;
};
static_assert(sizeof(SelectionPush) == 24);

// must match OPENPBR_LOBE_* in util-material-openpbr.glsl
constexpr u32 kLobeDiffuse = 2u;
constexpr u32 kLobeSubsurfaceEntry = 5u;
constexpr u32 kLobeCount = 6u;

std::vector<u32> run_selection_fraction(
	f32 const subsurfaceWeight, u32 const sampleCount, u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_entry_selection_fraction.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto pickBuf = vkof::buffer_create({
		.byteCount = kLobeCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zeros(kLobeCount, 0u);
	vkof::buffer_upload({
		.buffer = pickBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()), kLobeCount * sizeof(u32)
		),
	});

	SelectionPush const push {
		.pickCountVa = vkof::buffer_virtual_address(pickBuf),
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.subsurfaceWeight = subsurfaceWeight,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1u) / kLocalSize);

	auto out = test::readback<u32>(pickBuf, 0, kLobeCount);
	vkof::buffer_destroy(pickBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

f64 binomial_z(u64 const successes, u64 const n, f64 const expectedP) {
	f64 const observedP = static_cast<f64>(successes) / static_cast<f64>(n);
	f64 const stderrP = std::sqrt(expectedP * (1.0 - expectedP) / (f64)n);
	return (observedP - expectedP) / stderrP;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// task: openPbrSubsurfaceEntrySampleWo's direction distribution matches
// openPbrSubsurfaceEntryPdf
// -----------------------------------------------------------------------------

TEST_CASE("subsurface entry: sampler matches pdf, +z normal") {
	constexpr u32 kSampleCount = 1u << 20;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };
	auto const hist = run_entry_histogram(nor, kSampleCount, 1u);
	auto const expected = run_entry_expected(nor, kSampleCount);

	CHECK(hist[kBinCount] == 0u);
	u32 nonFiniteExpected = 0u;
	for (f32 const e : expected) { if (!std::isfinite(e)) { nonFiniteExpected++; } }
	CHECK(nonFiniteExpected == 0u);

	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pValue);
	CHECK(cs.observedTotal == kSampleCount);
	CHECK(cs.expectedTotal == doctest::Approx((f64)kSampleCount).epsilon(0.02));
	CHECK(cs.pValue > 1e-4);

	// every sample must land in the hemisphere opposite nor (entering the
	// surface), never above it
	for (u32 i = 0; i < kBinCount; ++i) {
		u32 const thetaBin = i / kPhiBins;
		// theta < pi/2 is the +nor hemisphere; those bins must be empty
		if (thetaBin < kThetaBins / 2u) {
			CHECK(hist[i] == 0u);
		}
	}
}

TEST_CASE("subsurface entry: sampler matches pdf, off-axis normal") {
	constexpr u32 kSampleCount = 1u << 20;
	f32v3 const nor { 0.5f, -0.3f, std::sqrt(1.0f - 0.5f*0.5f - 0.3f*0.3f) };
	auto const hist = run_entry_histogram(nor, kSampleCount, 2u);
	auto const expected = run_entry_expected(nor, kSampleCount);

	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pValue);
	CHECK(cs.observedTotal == kSampleCount);
	CHECK(cs.expectedTotal == doctest::Approx((f64)kSampleCount).epsilon(0.02));
	CHECK(cs.pValue > 1e-4);
}

// -----------------------------------------------------------------------------
// task: the discrete diffuse-vs-subsurface-entry pick fraction out of the
// full openPbrSampleWo tracks subsurfaceWeight
// -----------------------------------------------------------------------------

TEST_CASE("subsurface entry: selection fraction matches subsurfaceWeight") {
	// 0.0/1.0 excluded: the z-test's normal approximation is degenerate at
	// p=0 or p=1 (zero expected variance); those exact-boundary cases are
	// covered directly by the next test case instead
	constexpr u32 kSampleCount = 1u << 18;
	std::vector<f32> const weights { 0.3f, 0.7f };
	for (u32 c = 0; c < weights.size(); ++c) {
		auto const picks = run_selection_fraction(weights[c], kSampleCount, c + 1u);
		u64 const total = (
			(u64)picks[kLobeDiffuse] + (u64)picks[kLobeSubsurfaceEntry]
		);
		CAPTURE(weights[c]);
		CAPTURE(total);
		// with specular/coat/fuzz/transmission/metalness weights all zero,
		// nearly every draw lands on one of these two lobes; a small
		// residual specular probability survives even at specularWeight=0
		// (utilMicrofacetDielectricAlbedo's ggx energy-compensation fit
		// doesn't extrapolate to exactly 0 at f0=0), unrelated to this
		// lobe -- not tightened to exact equality here
		CHECK(total > (u64)((f64)kSampleCount * 0.99));
		f64 const z = binomial_z(
			picks[kLobeSubsurfaceEntry], total, (f64)weights[c]
		);
		CAPTURE(z);
		CHECK(std::fabs(z) < 5.0);
	}
}

TEST_CASE("subsurface entry: zero weight never selects it, full weight always does") {
	// same small residual-specular leak as the fraction test above --
	// checked here via an upper/lower bound instead of exact equality
	constexpr u32 kSampleCount = 4096u;
	auto const zero = run_selection_fraction(0.0f, kSampleCount, 100u);
	CHECK(zero[kLobeSubsurfaceEntry] == 0u);
	CHECK(zero[kLobeDiffuse] > kSampleCount - 32u);

	auto const one = run_selection_fraction(1.0f, kSampleCount, 101u);
	CHECK(one[kLobeDiffuse] == 0u);
	CHECK(one[kLobeSubsurfaceEntry] > kSampleCount - 32u);
}

} // TEST_SUITE("[headless]")
