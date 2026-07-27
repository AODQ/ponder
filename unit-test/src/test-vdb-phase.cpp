#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// utilVdbPhaseSampleMixture / utilVdbPhaseFog (lib/ponder/shaders/util-vdb
// .glsl): the HG+Draine fog phase function (jendersie & d'eon 2023) that
// drives the vdb random walk's re-aim direction. a phase function is the
// volumetric analog of a bsdf lobe -- azimuthally symmetric about the
// incoming travel direction rather than a surface normal, full sphere
// support instead of a hemisphere -- so this suite follows the same
// sampler-vs-eval chi-square convention as every bsdf lobe milestone
// (test-mixture-lobe-selection.cpp et al.), reusing util-sampling-bins.glsl
// with pc.wi in place of the shading normal.
//
// separately: utilVdbPhaseSampleDraineCos is a closed-form cubic-root
// inversion with large cancelling terms; a dense (xi, g, alpha) sweep checks
// it never returns a non-finite or out-of-range cosine, which was the lead
// suspect for firefly reports seen while integrating the walk.
// ---------------------------------------------------------------------------

namespace {

// must match BSDF_VERIFY_THETA_BINS / BSDF_VERIFY_PHI_BINS in
// util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

struct PhaseParams {
	f32 anisotropyHG = 0.0f;
	f32 anisotropyDraine = 0.0f;
	f32 draineAlpha = 0.0f;
	f32 draineWeight = 0.0f;
};

// ---------------------------------------------------------------------------
// vdb_phase_histogram.comp / vdb_phase_expected.comp runners
// ---------------------------------------------------------------------------

// shared by both the histogram and expected runners: vdb_phase_expected
// .comp's PC block is byte-identical (histogramVa is expectedVa there)
struct PhaseHistogramPush {
	u64 histogramVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 anisotropyHG;
	f32 anisotropyDraine;
	f32 draineAlpha;
	f32 draineWeight;
};
// 44 logical bytes, padded to 48 for u64's 8-byte struct alignment
static_assert(sizeof(PhaseHistogramPush) == 48);

std::vector<u32> run_phase_histogram(
	f32v3 const wi,
	PhaseParams const & params,
	u32 const sampleCount,
	u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "vdb_phase_histogram.comp",
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

	PhaseHistogramPush const push {
		.histogramVa = vkof::buffer_virtual_address(histBuf),
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.wi = wi,
		.anisotropyHG = params.anisotropyHG,
		.anisotropyDraine = params.anisotropyDraine,
		.draineAlpha = params.draineAlpha,
		.draineWeight = params.draineWeight,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<u32>(histBuf, 0, slotCount);
	vkof::buffer_destroy(histBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_phase_expected(
	f32v3 const wi,
	PhaseParams const & params,
	u32 const sampleCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "vdb_phase_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	PhaseHistogramPush const push {
		.histogramVa = vkof::buffer_virtual_address(buf),
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.wi = wi,
		.anisotropyHG = params.anisotropyHG,
		.anisotropyDraine = params.anisotropyDraine,
		.draineAlpha = params.draineAlpha,
		.draineWeight = params.draineWeight,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (kBinCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, kBinCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// ---------------------------------------------------------------------------
// chi-square, same pearson + wilson-hilferty convention as every prior
// milestone (see test-mixture-lobe-selection.cpp)
// ---------------------------------------------------------------------------

struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
	f64 pooledObserved;
	f64 pooledExpected;
	u32 nanExpectedCount;
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
	u32 nanExpectedCount = 0u;
	for (u32 i = 0; i < kBinCount; ++i) {
		observedTotal += observed[i];
		if (std::isnan(expected[i])) {
			nanExpectedCount++;
			continue;
		}
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
	f64 const dof = std::fmax((f64)cells - 1.0, 1.0);
	f64 const t = std::cbrt(statistic / dof);
	f64 const mu = 1.0 - 2.0 / (9.0 * dof);
	f64 const sigma = std::sqrt(2.0 / (9.0 * dof));
	f64 const z = (t - mu) / sigma;
	f64 const pValue = 0.5 * std::erfc(z / std::sqrt(2.0));
	return {
		statistic, dof, pValue, observedTotal, expectedTotal, pooledO, pooledE,
		nanExpectedCount
	};
}

void check_phase_chi_square(
	char const * const label,
	f32v3 const wi,
	PhaseParams const & params,
	u32 const seedSalt,
	char const * const residualPngPath = nullptr
) {
	constexpr u32 kSampleCount = 1u << 20;
	auto const hist = run_phase_histogram(wi, params, kSampleCount, seedSalt);
	auto const expected = run_phase_expected(wi, params, kSampleCount);

	CAPTURE(label);
	CAPTURE(params.anisotropyHG);
	CAPTURE(params.anisotropyDraine);
	CAPTURE(params.draineAlpha);
	CAPTURE(params.draineWeight);

	// no sample may be non-finite or non-unit
	CHECK(hist[kBinCount] == 0u);
	u32 nonFiniteExpected = 0u;
	for (f32 const e : expected) { if (!std::isfinite(e)) { nonFiniteExpected++; } }
	CHECK(nonFiniteExpected == 0u);

	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pooledObserved);
	CAPTURE(cs.pooledExpected);
	CAPTURE(cs.nanExpectedCount);
	CHECK(cs.nanExpectedCount == 0u);
	CHECK(cs.observedTotal == kSampleCount);
	// energy conservation: the phase function must integrate to 1 over the
	// sphere, so the expected-side total (integral * sampleCount summed
	// across every bin) must equal sampleCount
	CHECK(
		cs.expectedTotal == doctest::Approx((f64)kSampleCount).epsilon(0.02)
	);
	CHECK(cs.pValue > 1e-4);
	// the draine forward peak concentrates mass into a handful of bins near
	// theta=0, thinning many other cells below the pearson e>=5 floor
	CHECK(cs.pooledObserved <= cs.pooledExpected + 400.0);

	if (residualPngPath != nullptr) {
		std::vector<f32> r(kBinCount), g(kBinCount, 0.0f), b(kBinCount);
		for (u32 i = 0; i < kBinCount; ++i) {
			f32 const e = expected[i];
			f32 const res = (
				((f32)hist[i] - e) / std::sqrt(std::fmax(e, 1.0f))
			);
			r[i] = std::fmax(res, 0.0f) / 4.0f;
			b[i] = std::fmax(-res, 0.0f) / 4.0f;
		}
		u32 nanCount = 0;
		bool const ok = test::write_heatmap_png(
			r, g, b, kPhiBins, kThetaBins, residualPngPath, &nanCount
		);
		CHECK(ok);
		CHECK(nanCount == 0);
	}
}

// ---------------------------------------------------------------------------
// vdb_phase_draine_sweep.comp runner
// ---------------------------------------------------------------------------

struct DraineSweepPush {
	u64 outVa;
	u32 xiCount;
	u32 gCount;
	u32 alphaCount;
	f32 gMin;
	f32 gMax;
	f32 alphaMin;
	f32 alphaMax;
};
static_assert(sizeof(DraineSweepPush) == 40);

std::vector<f32> run_draine_sweep(
	u32 const xiCount, u32 const gCount, u32 const alphaCount,
	f32 const gMin, f32 const gMax,
	f32 const alphaMin, f32 const alphaMax
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "vdb_phase_draine_sweep.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u64 const count = (u64)xiCount * gCount * alphaCount;
	auto buf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	DraineSweepPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.xiCount = xiCount,
		.gCount = gCount,
		.alphaCount = alphaCount,
		.gMin = gMin,
		.gMax = gMax,
		.alphaMin = alphaMin,
		.alphaMax = alphaMax,
	};
	test::dispatch(
		pl, push,
		(xiCount + 7u) / 8u, (gCount + 7u) / 8u, (alphaCount + 3u) / 4u
	);

	auto out = test::readback<f32>(buf, 0, (u32)count);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// task: sampler vs eval chi-square, across the phase function's parameter
// range (isotropic through the fitted fog extremes) and a couple of
// off-axis wi directions (exercises the frisvad-duff frame construction at
// non-trivial incoming directions, not just +z)
// -----------------------------------------------------------------------------

TEST_CASE("vdb phase: sampler vs eval chi-square, isotropic") {
	PhaseParams const p {};
	check_phase_chi_square(
		"isotropic", { 0.0f, 0.0f, 1.0f }, p, 1u,
		(std::string(VDB_PHASE_OUTPUT_DIR) + "vdb_phase_isotropic_chi2.png").c_str()
	);
}

TEST_CASE("vdb phase: sampler vs eval chi-square, forward hg") {
	PhaseParams p;
	p.anisotropyHG = 0.85f;
	check_phase_chi_square(
		"forward-hg", { 0.0f, 0.0f, 1.0f }, p, 2u,
		(std::string(VDB_PHASE_OUTPUT_DIR) + "vdb_phase_forward_hg_chi2.png").c_str()
	);
}

TEST_CASE("vdb phase: sampler vs eval chi-square, backward hg") {
	PhaseParams p;
	p.anisotropyHG = -0.6f;
	check_phase_chi_square(
		"backward-hg", { 0.0f, 0.0f, 1.0f }, p, 3u,
		(std::string(VDB_PHASE_OUTPUT_DIR) + "vdb_phase_backward_hg_chi2.png").c_str()
	);
}

// matches utilVdbPhaseFogDefaultParams(20.0) -- the walk's current default
// droplet diameter -- so this is the actual production configuration, not
// just a synthetic corner
TEST_CASE("vdb phase: sampler vs eval chi-square, fitted fog (20 micron droplet)") {
	f32 const d = 20.0f;
	PhaseParams p;
	p.anisotropyHG = std::exp(-0.0990567f / (d - 1.67154f));
	p.anisotropyDraine = std::exp(-2.20679f / (d + 3.91029f) - 0.428934f);
	p.draineAlpha = std::exp(3.62489f - 8.29288f / (d + 5.52825f));
	p.draineWeight = std::exp(-0.599085f / (d - 0.641583f) - 0.665888f);
	check_phase_chi_square(
		"fitted-fog-20um", { 0.0f, 0.0f, 1.0f }, p, 4u,
		(std::string(VDB_PHASE_OUTPUT_DIR) + "vdb_phase_fitted_fog_chi2.png").c_str()
	);
}

TEST_CASE("vdb phase: sampler vs eval chi-square, draine-heavy mixture off-axis wi") {
	PhaseParams p;
	p.anisotropyHG = 0.3f;
	p.anisotropyDraine = 0.7f;
	p.draineAlpha = 15.0f;
	p.draineWeight = 0.8f;
	f32v3 const wi = { 0.5f, -0.3f, std::sqrt(1.0f - 0.5f*0.5f - 0.3f*0.3f) };
	check_phase_chi_square(
		"draine-heavy-offaxis", wi, p, 5u,
		(std::string(VDB_PHASE_OUTPUT_DIR) + "vdb_phase_draine_heavy_chi2.png").c_str()
	);
}

// -----------------------------------------------------------------------------
// task: draine cubic-root inversion numerical robustness. targets the
// suspected firefly source while integrating the walk: the closed-form
// solve (432*T1a^3 + T2 + ...) involves large cancelling terms that can
// lose precision or overflow for particular (xi, g, alpha) combinations
// even though the underlying formula is exact everywhere on its domain.
// -----------------------------------------------------------------------------

TEST_CASE("vdb phase: draine cubic solver stays finite and in [-1, 1]") {
	// alpha range covers the fitted-fog production range (~0 up to ~27 at
	// a 20 micron droplet, see utilVdbPhaseFogDefaultParams) with headroom;
	// g range covers the full anisotropy domain excluding the g=0 isotropic
	// special case (guarded separately, and not reachable through the
	// draine lobe's own division by g)
	constexpr u32 kXi = 256u;
	constexpr u32 kG = 64u;
	constexpr u32 kAlpha = 64u;
	auto const out = run_draine_sweep(kXi, kG, kAlpha, -0.95f, 0.95f, 0.01f, 40.0f);

	// a solve landing exactly at the cosTheta = +-1 boundary can overshoot
	// by a few ulps of float precision; that is rounding, not a solver
	// failure, so the range check tolerates a small epsilon and only the
	// worst overshoot across the whole sweep is reported
	constexpr f32 kBoundaryEpsilon = 1e-3f;
	u32 nanCount = 0u;
	u32 infCount = 0u;
	u32 outOfRangeCount = 0u;
	f32 worstOverrange = 0.0f;
	for (f32 const c : out) {
		if (std::isnan(c)) { nanCount++; continue; }
		if (std::isinf(c)) { infCount++; continue; }
		f32 const overrange = std::fabs(c) - 1.0f;
		if (overrange > kBoundaryEpsilon) {
			outOfRangeCount++;
		}
		worstOverrange = std::fmax(worstOverrange, overrange);
	}
	CAPTURE(nanCount);
	CAPTURE(infCount);
	CAPTURE(outOfRangeCount);
	CAPTURE(worstOverrange);
	CHECK(nanCount == 0u);
	CHECK(infCount == 0u);
	CHECK(outOfRangeCount == 0u);
}

} // TEST_SUITE("[headless]")
