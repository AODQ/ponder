#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/bluenoise.hpp>
#include "util.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {

struct NoiseSeedsPush {
	u64 seedOutVa;
	u64 seed2OutVa;
	u32 bluenoiseHandle;
	u32 frameIndex;
	u32 salt;
	u32 width;
};

constexpr u32 kGrid = 128; // matches the stbn tile size

struct GeneratedSeeds {
	std::vector<f32> seed;  // kGrid*kGrid, bluenoise + pcg (cranley-patterson)
	std::vector<f32> seed2; // kGrid*kGrid*2, pure pcg control, no bluenoise
};

// width must be a multiple of 16 (matches the shader's local_size_x/y)
GeneratedSeeds generate_seeds(
	ponder::Bluenoise const & bn, u32 frameIndex, u32 salt, u32 width = kGrid
) {
	u32 const count = width * width;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "noise_generate_seeds.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto seedBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto seed2Buf = vkof::buffer_create({
		.byteCount = count * 2 * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	NoiseSeedsPush const push {
		.seedOutVa = vkof::buffer_virtual_address(seedBuf),
		.seed2OutVa = vkof::buffer_virtual_address(seed2Buf),
		.bluenoiseHandle = bn.handles[frameIndex % ponder::Bluenoise::kCount],
		.frameIndex = frameIndex,
		.salt = salt,
		.width = width,
	};
	test::dispatch(pl, push, width / 16u, width / 16u);

	GeneratedSeeds out;
	out.seed = test::readback<f32>(seedBuf, 0, count);
	out.seed2 = test::readback<f32>(seed2Buf, 0, count * 2);

	vkof::buffer_destroy(seedBuf);
	vkof::buffer_destroy(seed2Buf);
	vkof::pipeline_destroy(pl);
	return out;
}

f64 variance_of(std::vector<f64> const & v) {
	f64 mean = 0.0;
	for (f64 x : v) { mean += x; }
	mean /= (f64)v.size();
	f64 var = 0.0;
	for (f64 x : v) { f64 const d = x - mean; var += d * d; }
	return var / (f64)v.size();
}

// single invocation (pixel 0,0), `count` sequential pcg-chain samples
std::vector<f32> generate_chain(ponder::Bluenoise const & bn, u32 count) {
	struct NoiseChainPush {
		u64 dstVa;
		u32 bluenoiseHandle;
		u32 count;
	};

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "noise_generate_chain.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	NoiseChainPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.bluenoiseHandle = bn.handles[0],
		.count = count,
	};
	test::dispatch(pl, push, 1u);

	auto out = test::readback<f32>(buf, 0, count);

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

struct HistogramStats {
	u32 maxCount;
	u32 minCount;
	bool wroteOk;
};

// bins `values` into `bins` buckets and draws a bar-chart png, with a
// reference line marking where a perfectly flat histogram's bar top would
// sit. shared by the spatial and single-pixel-chain histogram tests.
HistogramStats draw_histogram_png(
	std::vector<f32> const & values, u32 bins, char const * path
) {
	constexpr u32 kBarWidth = 12;
	u32 const imgWidth = bins * kBarWidth;
	constexpr u32 kImgHeight = 300;

	std::vector<u32> histogram(bins, 0u);
	for (f32 v : values) {
		u32 const bin = std::min<u32>((u32)(v * (f32)bins), bins - 1);
		histogram[bin]++;
	}

	u32 maxCount = 0, minCount = ~0u;
	for (u32 c : histogram) {
		maxCount = std::max(maxCount, c);
		minCount = std::min(minCount, c);
	}
	f64 const expected = (f64)values.size() / (f64)bins;

	std::vector<u8> pixels(imgWidth * kImgHeight, 0); // black background

	// reference line marking where a perfectly flat histogram's bar top
	// would sit -- deviation from this line is the thing to look for
	u32 const refRow = (
		kImgHeight - 1
		- (u32)((expected / (f64)maxCount) * (f64)(kImgHeight - 1))
	);
	for (u32 x = 0; x < imgWidth; ++x) { pixels[refRow * imgWidth + x] = 96; }

	for (u32 bin = 0; bin < bins; ++bin) {
		u32 const barHeight = (u32)(
			((f64)histogram[bin] / (f64)maxCount) * (f64)(kImgHeight - 1)
		);
		for (u32 x = bin * kBarWidth; x < bin * kBarWidth + kBarWidth - 1; ++x) {
			for (u32 y = kImgHeight - barHeight; y < kImgHeight; ++y) {
				pixels[y * imgWidth + x] = 255;
			}
		}
	}

	std::filesystem::create_directories(NOISE_OUTPUT_DIR);
	int const ok = stbi_write_png(
		path, (int)imgWidth, (int)kImgHeight, 1, pixels.data(), (int)imgWidth
	);
	return { maxCount, minCount, ok != 0 };
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("noise: seeds are finite and in [0, 1)") {
	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const s = generate_seeds(bn, 0, 0);

	for (f32 v : s.seed) {
		CAPTURE(v);
		CHECK(std::isfinite(v));
		CHECK(v >= 0.0f);
		CHECK(v < 1.0f);
	}
	for (f32 v : s.seed2) {
		CAPTURE(v);
		CHECK(std::isfinite(v));
		CHECK(v >= 0.0f);
		CHECK(v < 1.0f);
	}

	ponder::bluenoise_destroy(bn);
}

TEST_CASE("noise: seed marginal distribution is uniform (chi-square)") {
	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const s = generate_seeds(bn, 0, 0);

	constexpr u32 kBins = 64;
	u32 histogram[kBins] = {};
	for (f32 v : s.seed) {
		u32 const bin = std::min<u32>((u32)(v * (f32)kBins), kBins - 1);
		histogram[bin]++;
	}

	f64 const expected = (f64)s.seed.size() / (f64)kBins;
	f64 chiSquare = 0.0;
	for (u32 i = 0; i < kBins; ++i) {
		f64 const diff = (f64)histogram[i] - expected;
		chiSquare += diff * diff / expected;
	}

	u32 const dof = kBins - 1;
	// E[chi-square] == dof for a true uniform sample. This bound is
	// deliberately generous (avoids flakiness) -- it's meant to catch gross
	// bugs (bit-shift errors, modulo bias, clumped ranges), not to be a
	// tight statistical test.
	CAPTURE(chiSquare);
	CAPTURE(dof);
	CHECK(chiSquare < 4.0 * (f64)dof);

	ponder::bluenoise_destroy(bn);
}

TEST_CASE("noise: bluenoise-seeded channel is spatially more even than pcg-only control") {
	// this is the actual "is it blue, not just uniform" check: bin the grid
	// into cells and compare the variance of per-cell means. blue noise
	// samples repel each other spatially, so local averages should be more
	// consistent (lower variance) than a pcg-only (spatially white) control.
	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const s = generate_seeds(bn, 0, 0);

	constexpr u32 kCellSize = 8;
	constexpr u32 kCellsPerAxis = kGrid / kCellSize;
	constexpr u32 kCellCount = kCellsPerAxis * kCellsPerAxis;
	constexpr u32 kSamplesPerCell = kCellSize * kCellSize;

	std::vector<f64> cellMeanBlue(kCellCount, 0.0);
	std::vector<f64> cellMeanWhite(kCellCount, 0.0);
	for (u32 y = 0; y < kGrid; ++y) {
		for (u32 x = 0; x < kGrid; ++x) {
			u32 const idx = y * kGrid + x;
			u32 const cellIdx = (
				(y / kCellSize) * kCellsPerAxis + (x / kCellSize)
			);
			cellMeanBlue[cellIdx]  += (f64)s.seed[idx];
			cellMeanWhite[cellIdx] += (f64)s.seed2[idx * 2 + 0]; // pure pcg
		}
	}
	for (auto & m : cellMeanBlue)  { m /= (f64)kSamplesPerCell; }
	for (auto & m : cellMeanWhite) { m /= (f64)kSamplesPerCell; }

	f64 const varBlue  = variance_of(cellMeanBlue);
	f64 const varWhite = variance_of(cellMeanWhite);
	CAPTURE(varBlue);
	CAPTURE(varWhite);
	CHECK(varBlue < varWhite);

	ponder::bluenoise_destroy(bn);
}

TEST_CASE("noise: seed distribution histogram (visual sanity check)") {
	// pools a large batch of seed samples -- one per pixel, across many
	// pixels -- buckets them into bins, and draws a bar-chart png. tests
	// spatial uniformity: are seeds assigned across many different pixels
	// flat across [0, 1). the chi-square test above is the precise numeric
	// check; this is the "look at it" complement.
	constexpr u32 kPoolGrid = 512; // 512*512 = 262144 samples
	constexpr u32 kBins = 64;

	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const s = generate_seeds(bn, 0, 0, kPoolGrid);

	auto const stats = draw_histogram_png(
		s.seed, kBins, NOISE_OUTPUT_DIR "noise_histogram_spatial.png"
	);
	CHECK(stats.wroteOk);
	MESSAGE(
		"wrote spatial histogram to "
		NOISE_OUTPUT_DIR "noise_histogram_spatial.png"
	);

	// loose guard against gross breakage (e.g. all samples landing in one
	// bin) -- the chi-square test above is the tight numeric check
	CAPTURE(stats.minCount);
	CAPTURE(stats.maxCount);
	CHECK((f64)stats.maxCount / (f64)std::max(stats.minCount, 1u) < 2.0);

	ponder::bluenoise_destroy(bn);
}

TEST_CASE("noise: single-pixel pcg-chain histogram (visual sanity check)") {
	// same idea as the spatial histogram above, but for ONE pixel's chain
	// iterated many times instead of many pixels sampled once each. tests
	// temporal uniformity: does repeatedly advancing a single pixel's seed
	// (as a path tracer does across bounces) stay flat across [0, 1), or
	// does the chain drift/cycle/clump over many iterations.
	constexpr u32 kCount = 262144; // matches the spatial pool size
	constexpr u32 kBins = 64;

	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const chain = generate_chain(bn, kCount);

	auto const stats = draw_histogram_png(
		chain, kBins, NOISE_OUTPUT_DIR "noise_histogram_chain.png"
	);
	CHECK(stats.wroteOk);
	MESSAGE(
		"wrote chain histogram to "
		NOISE_OUTPUT_DIR "noise_histogram_chain.png"
	);

	CAPTURE(stats.minCount);
	CAPTURE(stats.maxCount);
	CHECK((f64)stats.maxCount / (f64)std::max(stats.minCount, 1u) < 2.0);

	ponder::bluenoise_destroy(bn);
}

TEST_CASE("noise: seed changes across frame index") {
	auto bn = ponder::bluenoise_create(PONDER_ASSET_DIR);
	auto const s0 = generate_seeds(bn, 0, 0);
	auto const s1 = generate_seeds(bn, 1, 0);

	u32 mismatches = 0;
	for (u32 i = 0; i < s0.seed.size(); ++i) {
		if (s0.seed[i] != s1.seed[i]) { mismatches++; }
	}
	// expect the overwhelming majority of pixels to differ between frames
	CHECK(mismatches > s0.seed.size() / 2);

	ponder::bluenoise_destroy(bn);
}

} // TEST_SUITE("[headless]")
