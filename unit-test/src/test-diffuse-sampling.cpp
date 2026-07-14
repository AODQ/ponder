#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// sampler/pdf verification for the ported openpbr diffuse sampling pair
// (openPbrGlossyDiffuseSampleWo / openPbrGlossyDiffusePdf, i.e. the cosine
// hemisphere sampler utilCosineHemisphereSampleWo): NaN fuzz, materialx
// same-xi cross-check, chi-square sampler-vs-pdf statistics, closed-form
// mean-direction statistics, and heatmap visualization.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// must match sampling_histogram.comp / sampling_expected.comp
constexpr u32 kModeDiffuseWo = 2u;

constexpr f64 kPi = 3.14159265358979323846;

struct FlatPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

std::vector<f32> run_flat_shader(
	char const * const shaderPath,
	std::vector<f32> const & inFlat,
	u32 const inStride,
	u32 const outStride
) {
	u32 const count = (u32)inFlat.size() / inStride;
	REQUIRE(inFlat.size() == (size_t)count * inStride);
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = shaderPath,
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = inFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(inFlat.data()),
			inFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * outStride * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FlatPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count * outStride);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// layout must match the scalar push block in sampling_histogram.comp /
// sampling_expected.comp
struct SamplingStatsPush {
	u64 bufferVa;
	u32 mode;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32v2 alpha;
};
static_assert(sizeof(SamplingStatsPush) == 40);

std::vector<u32> run_histogram(u32 const sampleCount, u32 const seedSalt) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "sampling_histogram.comp",
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

	SamplingStatsPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.mode = kModeDiffuseWo,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.wi = { 0.0f, 0.0f, 1.0f },
		.alpha = { 0.5f, 0.5f },
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<u32>(buf, 0, slotCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_expected(u32 const sampleCount) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "sampling_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	SamplingStatsPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.mode = kModeDiffuseWo,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.wi = { 0.0f, 0.0f, 1.0f },
		.alpha = { 0.5f, 0.5f },
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
};

// pearson chi-square with E < 5 pooling and a wilson-hilferty p-value
// (same construction as test-microfacet-sampling.cpp)
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
	if (pooledE >= 5.0 || pooledO > 0.0) {
		f64 const d = pooledO - pooledE;
		statistic += d * d / std::fmax(pooledE, 1.0);
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

// deterministic lcg in [0, 1), matching the other test files
struct Rng {
	u32 seed;
	f32 next() {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	}
};

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("diffuse sampling: core mapping matches materialx mx_cosine_sample_hemisphere") {
	// same-xi comparison under the documented swizzle (ponder draws
	// cosTheta from u.x and phi from u.y; materialx the transpose):
	// both sides must produce the identical local direction
	std::vector<f32> flat;
	// grid including the exact endpoints the samplers can be fed
	// (fnUintToUniform emits [0, 1), so 0 is reachable and 1 - 2^-23 is
	// the largest representable draw)
	std::vector<f32> const grid = {
		0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99999988f,
	};
	for (f32 const x : grid) {
		for (f32 const y : grid) {
			flat.push_back(x);
			flat.push_back(y);
		}
	}
	Rng rng { 555u };
	for (u32 i = 0; i < 2048u; ++i) {
		flat.push_back(rng.next());
		flat.push_back(rng.next());
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "diffuse_sample_crosscheck.comp", flat, 2u, 8u
	);
	for (u32 i = 0; i < out.size() / 8u; ++i) {
		CAPTURE(i);
		// mineLocal == mxLocal componentwise
		CHECK(out[i*8+0] == doctest::Approx(out[i*8+3]).epsilon(1e-5));
		CHECK(out[i*8+1] == doctest::Approx(out[i*8+4]).epsilon(1e-5));
		CHECK(out[i*8+2] == doctest::Approx(out[i*8+5]).epsilon(1e-5));
		// openPbrGlossyDiffusePdf == mx_cosine_hemisphere_PDF at the
		// sampled cosine
		CHECK(out[i*8+7] == doctest::Approx(out[i*8+6]).epsilon(1e-5));
	}
}

TEST_CASE("diffuse pdf: golden values") {
	// pdf(mu) = mu / pi; checked through the crosscheck shader's pdf
	// channel at controlled cosines: xi.x = c^2 makes the sampled
	// cosTheta exactly c
	std::vector<f32> flat;
	std::vector<f32> const cosines = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
	for (f32 const c : cosines) {
		flat.push_back(c * c);
		flat.push_back(0.0f);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "diffuse_sample_crosscheck.comp", flat, 2u, 8u
	);
	for (u32 i = 0; i < cosines.size(); ++i) {
		CAPTURE(cosines[i]);
		f64 const expected = (f64)cosines[i] / kPi;
		CHECK(out[i*8+7] == doctest::Approx(expected).epsilon(1e-4));
	}
}

TEST_CASE("diffuse sampler: unit, above horizon, finite, pass-through equality (fuzz)") {
	std::vector<f32> flat;
	std::vector<f32v3> nors;
	Rng rng { 8080u };
	constexpr u32 kCount = 4096u;
	for (u32 i = 0; i < kCount; ++i) {
		// random unit normal
		f32 const z = rng.next() * 2.0f - 1.0f;
		f32 const phi = rng.next() * 6.2831853f;
		f32 const s = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
		f32v3 const nor { s * std::cos(phi), s * std::sin(phi), z };
		nors.push_back(nor);
		flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
		flat.push_back(rng.next()); flat.push_back(rng.next());
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "diffuse_sample_evaluate.comp", flat, 5u, 10u
	);
	u32 lenFailures = 0u;
	u32 horizonFailures = 0u;
	u32 pdfFailures = 0u;
	u32 passThroughFailures = 0u;
	for (u32 i = 0; i < kCount; ++i) {
		f32v3 const wo { out[i*10+0], out[i*10+1], out[i*10+2] };
		f32 const pdf = out[i*10+3];
		CAPTURE(i);
		CHECK(std::isfinite(wo.x));
		CHECK(std::isfinite(wo.y));
		CHECK(std::isfinite(wo.z));
		CHECK(std::isfinite(pdf));
		f32 const len = std::sqrt(wo.x*wo.x + wo.y*wo.y + wo.z*wo.z);
		if (std::fabs(len - 1.0f) > 1e-3f) { lenFailures++; }
		f32 const dotNorWo = (
			nors[i].x * wo.x + nors[i].y * wo.y + nors[i].z * wo.z
		);
		if (dotNorWo < -1e-4f) { horizonFailures++; }
		// pdf at the sampled direction: cos / pi, in (0, 1/pi]
		if (!(pdf >= -1e-6f && pdf <= 0.3184f)) { pdfFailures++; }
		// openPbrGlossyDiffuseSampleWo is a pass-through to
		// utilCosineHemisphereSampleWo: same xi in, same direction out
		if (
			out[i*10+0] != out[i*10+6] || out[i*10+1] != out[i*10+7]
			|| out[i*10+2] != out[i*10+8]
		) { passThroughFailures++; }
	}
	CHECK(lenFailures == 0u);
	CHECK(horizonFailures == 0u);
	CHECK(pdfFailures == 0u);
	CHECK(passThroughFailures == 0u);
}

TEST_CASE("diffuse sampler vs pdf: chi-square") {
	constexpr u32 kSampleCount = 1u << 20;
	std::vector<u32> const salts = { 400u, 401u, 402u };
	for (u32 const salt : salts) {
		auto const observed = run_histogram(kSampleCount, salt);
		auto const expected = run_expected(kSampleCount);
		CAPTURE(salt);
		CHECK(observed[kBinCount] == 0u);
		auto const cs = chi_square(observed, expected);
		CAPTURE(cs.statistic);
		CAPTURE(cs.dof);
		CHECK(cs.observedTotal == kSampleCount);
		CHECK(
			cs.expectedTotal
			== doctest::Approx((f64)kSampleCount).epsilon(0.02)
		);
		CHECK(cs.pValue > 1e-4);
	}
}

TEST_CASE("diffuse sampler vs pdf: chi-square residual heatmap") {
	constexpr u32 kSampleCount = 1u << 20;
	auto const observed = run_histogram(kSampleCount, 999u);
	auto const expected = run_expected(kSampleCount);

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
	bool const ok = test::write_heatmap_png(
		r, g, b, kPhiBins, kThetaBins,
		SAMPLING_OUTPUT_DIR "diffuse_chi2_residual.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

TEST_CASE("diffuse sampler: closed-form mean direction") {
	// for a cosine-weighted hemisphere, E[wo] = (0, 0, 2/3) exactly:
	// E[cos theta] = int 2 c^2 dc = 2/3, and the azimuthal components
	// vanish by symmetry. per-component tolerances are ~4 sigma of the
	// monte carlo mean (sigma_z = sqrt(1/18 / N), sigma_xy = sqrt(1/4 / N))
	constexpr u32 kCount = 1u << 18;
	std::vector<f32> flat;
	flat.reserve((size_t)kCount * 5u);
	Rng rng { 4242u };
	for (u32 i = 0; i < kCount; ++i) {
		flat.push_back(0.0f); flat.push_back(0.0f); flat.push_back(1.0f);
		flat.push_back(rng.next()); flat.push_back(rng.next());
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "diffuse_sample_evaluate.comp", flat, 5u, 10u
	);
	f64 mx = 0.0, my = 0.0, mz = 0.0;
	for (u32 i = 0; i < kCount; ++i) {
		mx += (f64)out[i*10+0];
		my += (f64)out[i*10+1];
		mz += (f64)out[i*10+2];
	}
	mx /= (f64)kCount;
	my /= (f64)kCount;
	mz /= (f64)kCount;
	f64 const sigmaXy = 4.0 * std::sqrt(0.25 / (f64)kCount);
	f64 const sigmaZ = 4.0 * std::sqrt((1.0 / 18.0) / (f64)kCount);
	CAPTURE(mx);
	CAPTURE(my);
	CAPTURE(mz);
	CHECK(std::fabs(mx) < sigmaXy);
	CHECK(std::fabs(my) < sigmaXy);
	CHECK(std::fabs(mz - 2.0 / 3.0) < sigmaZ);
}

} // TEST_SUITE("[headless]")
