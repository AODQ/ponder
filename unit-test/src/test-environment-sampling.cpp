#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/environment-tables.hpp>
#include "util.hpp"

#include <tinyexr.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// sampler/pdf verification for lib/ponder's HDR environment map importance
// sampler (util-environment-map.glsl): chi-square sampler-vs-pdf statistics
// on the shared util-sampling-bins.glsl sphere grid (same technique as
// test-transmission-sampling.cpp / test-microfacet-sampling.cpp), plus a
// dense NaN/degenerate-value sweep over the seam/pole/xi-corner cases a
// stochastic test would rarely hit. exercises the real
// ponder::environment_tables_create loader, not hand-built gpu tables: each
// synthetic map is written to a temp .exr via tinyexr's SaveEXR, then loaded
// exactly the way the viewer loads a real HDRI.
//
// beyond the well-behaved sun map, the cases below feed the loader the
// degenerate inputs a real asset pipeline eventually produces: an all-black
// map (the cdf builder's uniform-fallback path), a half-black map (zero-mass
// rows inside an otherwise valid map), a single-bright-texel delta map (cdf
// binary-search stress), a 1x1 map, a missing file, and maps containing a
// NaN / negative texel (real HDRIs contain both; the two cases document the
// loader's current unsanitized behavior rather than a desired contract --
// see the reported finding before "fixing" either).
//
// everything runs at two rotations: rotation consistency between
// environmentMapUvToDir / DirToUv / ImportanceSample / Pdf / Radiance is
// exactly what env-map MIS unbiasedness rests on, and a rotation term
// dropped from any one of them cancels out of a rotation-0-only test.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// mirrors util-environment-map.glsl's EnvironmentMapHandles field order
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 conditionalCdf;
	u32 marginalCdf;
};

GpuEnvironmentMapHandles handles_from_tables(
	ponder::EnvironmentTables const & tables
) {
	return {
		.radiance = tables.radianceHandle,
		.pdf = tables.pdfHandle,
		.conditionalCdf = tables.conditionalCdfHandle,
		.marginalCdf = tables.marginalCdfHandle,
	};
}

// writes an rgba float exr and loads it back through the production loader,
// so every case exercises environment_tables_create end to end
ponder::EnvironmentTables tables_from_pixels(
	char const * const path,
	i32 const width,
	i32 const height,
	std::vector<f32> const & pixels
) {
	REQUIRE(pixels.size() == (size_t)width * height * 4);
	char const * err = nullptr;
	i32 const ret = SaveEXR(
		pixels.data(), width, height, 4, /*save_as_fp16=*/0, path, &err
	);
	if (ret != TINYEXR_SUCCESS) {
		MESSAGE("SaveEXR failed: ", err ? err : "unknown error");
		if (err) { FreeEXRErrorMessage(err); }
	}
	REQUIRE(ret == TINYEXR_SUCCESS);
	return ponder::environment_tables_create(path);
}

// fills one rgb value at (x, y); alpha is always 1
void set_pixel(
	std::vector<f32> & pixels,
	i32 const width,
	i32 const x,
	i32 const y,
	f32 const v
) {
	size_t const i = ((size_t)y * width + x) * 4;
	pixels[i + 0] = v;
	pixels[i + 1] = v;
	pixels[i + 2] = v;
	pixels[i + 3] = 1.0f;
}

// sun-map geometry shared by the chi-square and delta cases
constexpr i32 kSynthWidth = 128;
constexpr i32 kSynthHeight = 64;
constexpr i32 kSunXLo = kSynthWidth * 3 / 4 - 4;
constexpr i32 kSunXHi = kSynthWidth * 3 / 4 + 4;
constexpr i32 kSunYLo = kSynthHeight / 4 - 4;
constexpr i32 kSunYHi = kSynthHeight / 4 + 4;

// smooth, always-positive background (so no row/column collapses to the
// cdf builder's pure-black uniform fallback) plus one bright patch, so the
// importance sampler has something worth concentrating on and the
// quadrature-integrated pdf has an easily-reasoned-about shape
ponder::EnvironmentTables make_synthetic_env(char const * const path) {
	std::vector<f32> pixels((size_t)kSynthWidth * kSynthHeight * 4);
	for (i32 y = 0; y < kSynthHeight; ++y) {
		for (i32 x = 0; x < kSynthWidth; ++x) {
			f32 const bg = 0.1f + 0.05f * (f32)x / (f32)kSynthWidth;
			bool const isSun = (
				x >= kSunXLo && x < kSunXHi && y >= kSunYLo && y < kSunYHi
			);
			set_pixel(pixels, kSynthWidth, x, y, isSun ? 40.0f : bg);
		}
	}
	return tables_from_pixels(path, kSynthWidth, kSynthHeight, pixels);
}

// layout must match the scalar push block in
// environment_sampling_histogram.comp / environment_sampling_expected.comp
// (the expected shader declares the unused seedSalt field precisely so the
// two blocks stay byte-identical and can share this struct)
struct EnvSamplingPush {
	u64 bufferVa;
	GpuEnvironmentMapHandles handles;
	u32 sampleCount;
	u32 seedSalt;
	f32 rotation;
};

// observed side: histogram over the sphere bins; slot kBinCount counts
// non-finite / non-unit sampled directions
std::vector<u32> run_histogram(
	GpuEnvironmentMapHandles const & handles,
	u32 const sampleCount,
	u32 const seedSalt,
	f32 const rotation
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "environment_sampling_histogram.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const slotCount = kBinCount + 1u;
	auto buf = vkof::buffer_create({
		.byteCount = slotCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zeros(slotCount, 0u);
	vkof::buffer_upload({
		.buffer = buf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()),
			slotCount * sizeof(u32)
		),
	});

	EnvSamplingPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.handles = handles,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.rotation = rotation,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<u32>(buf, 0, slotCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// expected side: per-bin quadrature of environmentMapPdf, scaled by
// sampleCount
std::vector<f32> run_expected(
	GpuEnvironmentMapHandles const & handles,
	u32 const sampleCount,
	f32 const rotation
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "environment_sampling_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	EnvSamplingPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.handles = handles,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.rotation = rotation,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (kBinCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, kBinCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
	f64 pooledObserved;
	f64 pooledExpected;
};

// pearson chi-square with the standard small-expectation pooling: bins with
// E < 5 are folded into one pooled cell. p-value via the wilson-hilferty
// cube-root normal approximation (same technique as
// test-microfacet-sampling.cpp / test-transmission-sampling.cpp)
ChiSquare chi_square(
	std::vector<u32> const & observed,
	std::vector<f32> const & expected
) {
	f64 statistic = 0.0;
	u32 cells = 0u;
	f64 pooledO = 0.0;
	f64 pooledE = 0.0;
	u64 observedTotal = 0u;
	f64 expectedTotal = 0.0;
	for (u32 i = 0; i < kBinCount; ++i) {
		observedTotal += observed[i];
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
		statistic, dof, pValue, observedTotal, expectedTotal, pooledO, pooledE
	};
}

// full sampler-vs-pdf chi-square pass at one rotation: histogram vs
// quadrature, integration-to-one, pooled-mass allowance, residual heatmap.
// returns the observed histogram so callers can run cross-rotation checks
// on top (see the rotated chi-square case)
std::vector<u32> run_chi_square_case(
	GpuEnvironmentMapHandles const & handles,
	f32 const rotation,
	char const * const residualPngName
) {
	constexpr u32 kSampleCount = 1u << 20;
	auto const observed = run_histogram(
		handles, kSampleCount, /*seedSalt=*/0u, rotation
	);
	auto const expected = run_expected(handles, kSampleCount, rotation);

	// no sample may be non-finite or non-unit
	CHECK(observed[kBinCount] == 0u);

	auto const cs = chi_square(observed, expected);
	CAPTURE(rotation);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pooledObserved);
	CAPTURE(cs.pooledExpected);
	// every sample must land in some bin
	CHECK(cs.observedTotal == kSampleCount);
	// the pdf must integrate to ~1 over the sphere (2% quadrature slack)
	CHECK(
		cs.expectedTotal == doctest::Approx((f64)kSampleCount).epsilon(0.02)
	);
	// sampler and pdf describe the same distribution; the threshold is
	// loose enough (1e-4) that a correct pairing essentially never trips
	// it, while a real mismatch produces p ~ 0
	CHECK(cs.pValue > 1e-4);
	// no support boundary anywhere on the sphere (unlike the bsdf lobes'
	// cap-cut/tir curves), so the pooled (E < 5) allowance only needs to
	// cover ordinary sampling noise
	CHECK(cs.pooledObserved <= cs.pooledExpected + 64.0);

	// standardized residual (O - E) / sqrt(max(E, 1)), mapped to a +-4
	// sigma diverging red/blue image, same convention as the bsdf-lobe
	// sampling tests
	std::vector<f32> r(kBinCount), g(kBinCount, 0.0f), b(kBinCount);
	for (u32 i = 0; i < kBinCount; ++i) {
		f32 const e = expected[i];
		f32 const res = (
			((f32)observed[i] - e) / std::sqrt(std::fmax(e, 1.0f))
		);
		r[i] = std::fmax(res, 0.0f) / 4.0f;
		b[i] = std::fmax(-res, 0.0f) / 4.0f;
	}
	u32 nanCount = 0;
	std::string const pngPath = (
		std::string(ENVIRONMENT_OUTPUT_DIR) + residualPngName
	);
	bool const ok = test::write_heatmap_png(
		r, g, b, kPhiBins, kThetaBins, pngPath.c_str(), &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);

	return observed;
}

// layout must match the scalar push block in environment_map_sweep.comp
struct EnvSweepPush {
	u64 outVa;
	GpuEnvironmentMapHandles handles;
	u32 width;
	u32 height;
	f32 rotation;
};

// 7 floats per texel: directBad, pdfDirect, sampleNonFiniteCount,
// sampleMaxUnitLengthError, sampleMinPdf, sampleMaxPdf, roundTripUvError
constexpr u32 kSweepStride = 7u;

std::vector<f32> run_sweep(
	GpuEnvironmentMapHandles const & handles,
	u32 const width,
	u32 const height,
	f32 const rotation
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "environment_map_sweep.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = (u64)width * height * kSweepStride * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	EnvSweepPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.handles = handles,
		.width = width,
		.height = height,
		.rotation = rotation,
	};
	test::dispatch(pl, push, (width + 7u) / 8u, (height + 7u) / 8u);

	auto out = test::readback<f32>(buf, 0, (u64)width * height * kSweepStride);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

struct SweepStats {
	u32 directBad;
	u32 nonFinite;
	f32 maxLenErr;
	// texels (outside row 0, see the xi.y == 0 pole exception in the
	// original sweep case) whose sampled minPdf failed to be > 0
	u32 pdfZeroTexels;
	// min/max over every texel's own sampled pdf extrema
	f32 minPdf;
	f32 maxPdf;
	f32 maxRoundTripErr;
	// min/max of the direct-probe pdf over the whole grid
	f32 minPdfDirect;
	f32 maxPdfDirect;
};

// aggregates the sweep buffer; optionally writes the pdf/nan heatmap png
SweepStats sweep_stats(
	std::vector<f32> const & out,
	u32 const width,
	u32 const height,
	char const * const optPngName
) {
	SweepStats stats {
		.directBad = 0u,
		.nonFinite = 0u,
		.maxLenErr = 0.0f,
		.pdfZeroTexels = 0u,
		.minPdf = 1e30f,
		.maxPdf = 0.0f,
		.maxRoundTripErr = 0.0f,
		.minPdfDirect = 1e30f,
		.maxPdfDirect = 0.0f,
	};
	std::vector<f32> r(width * height);
	std::vector<f32> g(width * height, 0.0f);
	std::vector<f32> b(width * height);
	for (u32 y = 0; y < height; ++y) {
		for (u32 x = 0; x < width; ++x) {
			u32 const i = y * width + x;
			f32 const isBad = out[i * kSweepStride + 0u];
			f32 const pdfDirect = out[i * kSweepStride + 1u];
			f32 const nf = out[i * kSweepStride + 2u];
			f32 const lenErr = out[i * kSweepStride + 3u];
			f32 const minPdf = out[i * kSweepStride + 4u];
			f32 const maxPdf = out[i * kSweepStride + 5u];
			f32 const rtErr = out[i * kSweepStride + 6u];
			if (isBad > 0.5f) { stats.directBad++; }
			if (nf > 0.0f) {
				stats.nonFinite++;
				r[i] = g[i] = b[i] = std::nanf("");
				continue;
			}
			stats.maxLenErr = std::fmax(stats.maxLenErr, lenErr);
			stats.maxRoundTripErr = std::fmax(stats.maxRoundTripErr, rtErr);
			stats.minPdf = std::fmin(stats.minPdf, minPdf);
			stats.maxPdf = std::fmax(stats.maxPdf, maxPdf);
			stats.minPdfDirect = std::fmin(stats.minPdfDirect, pdfDirect);
			stats.maxPdfDirect = std::fmax(stats.maxPdfDirect, pdfDirect);
			// texel row 0's first xi substratum hits xi.y == 0 exactly,
			// which the marginal cdf maps to the exact north pole (v=0,
			// theta=0) -- environmentMapPdf's sinTheta guard correctly
			// returns 0 there (vanishing solid angle), same documented
			// exception style as ggx_bounded_sample_sweep's x==0 case
			if (y != 0u && !(minPdf > 0.0f)) { stats.pdfZeroTexels++; }
			f32 const v = pdfDirect / (pdfDirect + 1.0f);
			r[i] = v; g[i] = v; b[i] = v;
		}
	}
	if (optPngName != nullptr) {
		u32 nanCount = 0;
		std::string const pngPath = (
			std::string(ENVIRONMENT_OUTPUT_DIR) + optPngName
		);
		bool const ok = test::write_heatmap_png(
			r, g, b, width, height, pngPath.c_str(), &nanCount
		);
		CHECK(ok);
		CHECK(nanCount == stats.nonFinite);
	}
	return stats;
}

// standard sweep expectations for a well-formed, everywhere-positive map
void check_sweep_well_formed(SweepStats const & stats) {
	CHECK(stats.directBad == 0u);
	CHECK(stats.nonFinite == 0u);
	CHECK(stats.maxLenErr < 1e-3f);
	CHECK(stats.pdfZeroTexels == 0u);
	CHECK(stats.maxRoundTripErr < 1e-3f);
}

} // namespace

TEST_CASE("environment map: sampler-vs-pdf chi-square") {
	auto tables = make_synthetic_env(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env.exr"
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	run_chi_square_case(
		handles, /*rotation=*/0.0f, "environment_sampling_residual.png"
	);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: sampler-vs-pdf chi-square, rotated") {
	auto tables = make_synthetic_env(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_rotated.exr"
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// pi/2 is exactly 20 of the 80 phi bins, so the rotated histogram must
	// be the unrotated one circularly shifted by 20 -- run the full
	// chi-square at the rotation first (sampler/pdf consistency under
	// rotation), then locate the shift by cross-correlating the two phi
	// marginals. this is the check the chi-square alone cannot make: a
	// rotation term ignored consistently by BOTH the sampler and the pdf
	// still passes chi-square, but produces a zero shift here
	f32 const rotation = 0.5f * 3.14159265358979323846f;
	auto const rotated = run_chi_square_case(
		handles, rotation, "environment_sampling_residual_rotated.png"
	);
	auto const unrotated = run_histogram(
		handles, 1u << 20, /*seedSalt=*/1u, /*rotation=*/0.0f
	);

	std::vector<f64> phiRot(kPhiBins, 0.0);
	std::vector<f64> phiUnrot(kPhiBins, 0.0);
	for (u32 bin = 0; bin < kBinCount; ++bin) {
		phiRot[bin % kPhiBins] += (f64)rotated[bin];
		phiUnrot[bin % kPhiBins] += (f64)unrotated[bin];
	}
	u32 bestShift = 0u;
	f64 bestScore = -1.0;
	for (u32 shift = 0; shift < kPhiBins; ++shift) {
		f64 score = 0.0;
		for (u32 i = 0; i < kPhiBins; ++i) {
			score += phiUnrot[i] * phiRot[(i + shift) % kPhiBins];
		}
		if (score > bestScore) {
			bestScore = score;
			bestShift = shift;
		}
	}
	// the sun must land 20 phi bins away (either direction, depending on
	// the bin grid's phi handedness relative to the equirect u axis); the
	// failure mode this exists to catch -- rotation ignored somewhere --
	// shows up as a shift of 0
	u32 const wrapped = std::min(bestShift, kPhiBins - bestShift);
	CAPTURE(bestShift);
	CHECK(wrapped >= 19u);
	CHECK(wrapped <= 21u);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: NaN sweep heatmap") {
	auto tables = make_synthetic_env(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_sweep.exr"
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	constexpr u32 kWidth = 256u;
	constexpr u32 kHeight = 128u;
	auto const out = run_sweep(handles, kWidth, kHeight, /*rotation=*/0.0f);
	auto const stats = sweep_stats(
		out, kWidth, kHeight, "environment_map_nan_sweep.png"
	);
	check_sweep_well_formed(stats);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: NaN sweep heatmap, rotated") {
	auto tables = make_synthetic_env(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_sweep_rotated.exr"
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// deliberately not bin- or texel-aligned, so the seam (u wrap) lands
	// mid-texel and the round-trip check runs off the easy grid
	constexpr u32 kWidth = 256u;
	constexpr u32 kHeight = 128u;
	auto const out = run_sweep(handles, kWidth, kHeight, /*rotation=*/2.345f);
	auto const stats = sweep_stats(
		out, kWidth, kHeight, "environment_map_nan_sweep_rotated.png"
	);
	check_sweep_well_formed(stats);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: all-black map, uniform cdf fallback, zero pdf") {
	// an all-black map drives the loader's pure-black uniform fallback for
	// every conditional row AND the marginal: sampling must stay finite and
	// unit-length (the viewer can load a black hdri without crashing), while
	// the pdf is identically zero -- which is exactly what makes env
	// nee/mis a correct no-op (the envPdf > 0 gate in util-raytrace.glsl)
	constexpr i32 kWidth = 64;
	constexpr i32 kHeight = 32;
	std::vector<f32> pixels((size_t)kWidth * kHeight * 4, 0.0f);
	for (i32 y = 0; y < kHeight; ++y) {
		for (i32 x = 0; x < kWidth; ++x) {
			set_pixel(pixels, kWidth, x, y, 0.0f);
		}
	}
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_black.exr",
		kWidth, kHeight, pixels
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	auto const out = run_sweep(handles, 64u, 32u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 64u, 32u, nullptr);
	CHECK(stats.directBad == 0u);
	CHECK(stats.nonFinite == 0u);
	CHECK(stats.maxLenErr < 1e-3f);
	CHECK(stats.maxRoundTripErr < 1e-3f);
	// zero everywhere: both the sampled-side pdf and the direct probe
	CHECK(stats.maxPdf == 0.0f);
	CHECK(stats.maxPdfDirect == 0.0f);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: half-black map, zero-mass rows") {
	// bottom hemisphere exactly zero (the shape of any ground-masked hdri):
	// rows below the horizon carry no marginal mass and take the per-row
	// uniform fallback, rows above stay a normal distribution. the boundary
	// v = 0.5 aligns with both a texel edge and a theta-bin edge, so the
	// quadrature side has no straddling bin to soften the check
	constexpr i32 kWidth = 128;
	constexpr i32 kHeight = 64;
	std::vector<f32> pixels((size_t)kWidth * kHeight * 4);
	for (i32 y = 0; y < kHeight; ++y) {
		for (i32 x = 0; x < kWidth; ++x) {
			f32 const v = y < kHeight / 2 ? 0.25f : 0.0f;
			set_pixel(pixels, kWidth, x, y, v);
		}
	}
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_halfblack.exr",
		kWidth, kHeight, pixels
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// the sampler must never land in the zero half; the chi-square's pooled
	// cell covers every lower-hemisphere bin (E = 0 there), so the pooled
	// allowance check is what enforces that
	run_chi_square_case(
		handles, /*rotation=*/0.0f, "environment_sampling_residual_halfblack.png"
	);

	// every sampled direction resolves through the upper half, so the
	// standard positive-pdf sweep expectations still hold
	auto const out = run_sweep(handles, 128u, 64u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 128u, 64u, nullptr);
	check_sweep_well_formed(stats);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: single-bright-texel delta map") {
	// one lit texel on an otherwise black map: the near-delta distribution
	// a tiny sun in an otherwise black sky produces. every xi in [0,1)^2
	// must invert through the cdfs into that texel, with a pdf matching the
	// closed form W*H / (2 pi^2 sinTheta) over the texel's own theta span
	constexpr i32 kDeltaX = 96;
	constexpr i32 kDeltaY = 20;
	std::vector<f32> pixels((size_t)kSynthWidth * kSynthHeight * 4, 0.0f);
	for (i32 y = 0; y < kSynthHeight; ++y) {
		for (i32 x = 0; x < kSynthWidth; ++x) {
			set_pixel(pixels, kSynthWidth, x, y, 0.0f);
		}
	}
	set_pixel(pixels, kSynthWidth, kDeltaX, kDeltaY, 100.0f);
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_delta.exr",
		kSynthWidth, kSynthHeight, pixels
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	f64 const pi = 3.14159265358979323846;

	// -- sweep side: every sampled pdf must sit inside the texel's own
	// closed-form band. pdf texel value is exactly 1 (all importance in one
	// texel), so solid-angle pdf = W*H / (2 pi^2 sinTheta) with theta
	// anywhere inside the texel's v span
	auto const out = run_sweep(handles, 128u, 64u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 128u, 64u, nullptr);
	CHECK(stats.directBad == 0u);
	CHECK(stats.nonFinite == 0u);
	CHECK(stats.maxLenErr < 1e-3f);
	CHECK(stats.maxRoundTripErr < 1e-3f);
	f64 const theta0 = pi * (f64)kDeltaY / (f64)kSynthHeight;
	f64 const theta1 = pi * (f64)(kDeltaY + 1) / (f64)kSynthHeight;
	f64 const sinMin = std::fmin(std::sin(theta0), std::sin(theta1));
	f64 const sinMax = 1.0;
	f64 const pdfScale = (
		(f64)kSynthWidth * (f64)kSynthHeight / (2.0 * pi * pi)
	);
	// sinTheta <= 1 always, and over this texel's span sin is monotonic;
	// 1% slack for the f32 pipeline
	f64 const pdfLo = pdfScale / sinMax * 0.99;
	f64 const pdfHi = pdfScale / sinMin * 1.01;
	CAPTURE(stats.minPdf);
	CAPTURE(stats.maxPdf);
	CHECK((f64)stats.minPdf >= pdfLo);
	CHECK((f64)stats.maxPdf <= pdfHi);

	// -- histogram side: every sample must land in a sphere bin overlapping
	// the texel's theta band (phi bin conventions differ from equirect u,
	// so the theta band is the axis checked absolutely; phi locality is
	// checked as "at most two adjacent bins used")
	auto const observed = run_histogram(
		handles, 1u << 16, /*seedSalt=*/0u, /*rotation=*/0.0f
	);
	CHECK(observed[kBinCount] == 0u);
	u32 const thetaBinLo = (u32)(
		(u64)kThetaBins * (u64)kDeltaY / (u64)kSynthHeight
	);
	u32 const thetaBinHi = (u32)(
		(u64)kThetaBins * (u64)(kDeltaY + 1) / (u64)kSynthHeight
	);
	std::vector<u32> phiBinsUsed;
	for (u32 bin = 0; bin < kBinCount; ++bin) {
		if (observed[bin] == 0u) { continue; }
		u32 const thetaBin = bin / kPhiBins;
		u32 const phiBin = bin % kPhiBins;
		CAPTURE(thetaBin);
		CHECK(thetaBin >= thetaBinLo);
		CHECK(thetaBin <= thetaBinHi);
		if (
			std::find(phiBinsUsed.begin(), phiBinsUsed.end(), phiBin)
			== phiBinsUsed.end()
		) {
			phiBinsUsed.push_back(phiBin);
		}
	}
	// texel phi span (2 pi / 128) covers at most two adjacent 2 pi / 80 bins
	CHECK(phiBinsUsed.size() >= 1u);
	CHECK(phiBinsUsed.size() <= 2u);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: 1x1 map") {
	// smallest possible map: one texel is the whole sphere. exercises the
	// cdf binary search at count == 1 on both axes; the (uv-uniform)
	// sampler and the 1/(2 pi^2 sinTheta) pdf remain a consistent pair by
	// construction, so only well-formedness is checked here
	std::vector<f32> pixels(4);
	set_pixel(pixels, 1, 0, 0, 1.0f);
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_1x1.exr", 1, 1, pixels
	);
	REQUIRE(tables.width == 1u);
	REQUIRE(tables.height == 1u);
	auto const handles = handles_from_tables(tables);

	auto const out = run_sweep(handles, 64u, 32u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 64u, 32u, nullptr);
	check_sweep_well_formed(stats);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("environment map: loader failure returns empty tables") {
	auto const tables = ponder::environment_tables_create(
		ENVIRONMENT_OUTPUT_DIR "does_not_exist.exr"
	);
	// the viewer branches on width != 0 to decide whether an env map is
	// loaded at all; a missing/corrupt file must come back empty, not crash
	CHECK(tables.width == 0u);
	CHECK(tables.height == 0u);
	// nothing was created, so nothing to destroy
}

TEST_CASE("environment map: NaN input texel (documents unsanitized loader)") {
	// real HDRIs ship with NaN texels. the loader currently does NOT
	// sanitize: the NaN propagates through the importance sums, every
	// positivity test on the poisoned sums fails, and BOTH cdfs quietly
	// take their uniform fallback while the pdf zeroes out entirely --
	// sampling stays finite but importance sampling is silently lost for
	// the whole map, and the radiance texture still returns NaN wherever
	// the bad texel is filtered. this case documents that behavior; it is
	// a reported finding, not a contract -- if input sanitization lands in
	// environment_tables_create, flip these expectations
	constexpr i32 kWidth = 64;
	constexpr i32 kHeight = 32;
	std::vector<f32> pixels((size_t)kWidth * kHeight * 4);
	for (i32 y = 0; y < kHeight; ++y) {
		for (i32 x = 0; x < kWidth; ++x) {
			set_pixel(pixels, kWidth, x, y, 0.15f);
		}
	}
	set_pixel(pixels, kWidth, 10, 10, std::nanf(""));
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_nan.exr",
		kWidth, kHeight, pixels
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	auto const out = run_sweep(handles, 64u, 32u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 64u, 32u, nullptr);
	// sampling itself never produces a non-finite direction or pdf
	CHECK(stats.nonFinite == 0u);
	CHECK(stats.maxLenErr < 1e-3f);
	// the single NaN texel zeroed the ENTIRE pdf (totalSum is NaN, so
	// every positivity test fails into the zero/fallback branch)
	CHECK(stats.maxPdf == 0.0f);
	CHECK(stats.maxPdfDirect == 0.0f);
	// and the radiance texture still leaks NaN around the bad texel: the
	// direct probes near it fail their finite-radiance check
	CHECK(stats.directBad > 0u);
	MESSAGE(
		"NaN-texel map: directBad=", stats.directBad,
		" (NaN radiance reachable), pdf zeroed map-wide"
	);

	ponder::environment_tables_destroy(tables);
}

TEST_CASE(
	"environment map: negative input texel (documents unsanitized loader)"
) {
	// negative texels also occur in real assets. the loader keeps the
	// negative luminance: the affected row's cdf is non-monotonic and the
	// stored pdf is negative at that texel, which environmentMapPdf then
	// returns as a negative density -- the direct probes near the texel
	// fail their pdf >= 0 check. same reported-finding status as the NaN
	// case above
	constexpr i32 kWidth = 64;
	constexpr i32 kHeight = 32;
	std::vector<f32> pixels((size_t)kWidth * kHeight * 4);
	for (i32 y = 0; y < kHeight; ++y) {
		for (i32 x = 0; x < kWidth; ++x) {
			set_pixel(pixels, kWidth, x, y, 0.15f);
		}
	}
	set_pixel(pixels, kWidth, 10, 10, -50.0f);
	auto tables = tables_from_pixels(
		ENVIRONMENT_OUTPUT_DIR "synthetic_test_env_negative.exr",
		kWidth, kHeight, pixels
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	auto const out = run_sweep(handles, 64u, 32u, /*rotation=*/0.0f);
	auto const stats = sweep_stats(out, 64u, 32u, nullptr);
	// sampling stays finite, but a negative pdf is reachable
	CHECK(stats.nonFinite == 0u);
	CHECK(stats.maxLenErr < 1e-3f);
	CHECK(stats.directBad > 0u);
	CHECK(stats.minPdfDirect < 0.0f);
	MESSAGE(
		"negative-texel map: directBad=", stats.directBad,
		" minPdfDirect=", stats.minPdfDirect
	);

	ponder::environment_tables_destroy(tables);
}
